"""艦隊ルータ(FleetLLM)+ vLLM バックエンドの GPU 無し検証。

OpenAI 互換の最小 JSON を返すスタブ HTTP サーバを threading で立て、
  (a) sticky routing(同じ agent_id は常に同じサーバ)
  (b) フェイルオーバ + cooldown 後の復帰
  (c) name が URL 非依存(D13 キャッシュキー安定)
  (d) CachedLLM と組んだとき同一プロンプトはキャッシュヒットしサーバを叩かない
を確認する。実 GPU/実 vLLM は本選環境で疎通(ここではスタブまで)。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from society.llm.cache import CachedLLM
from society.llm.fleet import FleetLLM
from society.llm.vllm import VllmBackend


# ---- OpenAI 互換の最小スタブ ----
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):        # テスト出力を汚さない
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            req = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            req = {}
        srv = self.server
        srv.hits.append(req)             # 受信リクエストを記録
        if srv.fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "down"}')
            return
        # どのサーバが応答したか分かるよう tag を JSON に載せる
        content = json.dumps({"action": "wander", "_server": srv.tag},
                             ensure_ascii=False)
        if self.path.endswith("/chat/completions"):
            body = {"choices": [{"message": {"content": content}}]}
        else:
            body = {"choices": [{"text": content}]}
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_stub(tag: str):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.hits = []
    srv.fail = False
    srv.tag = tag
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    return srv, url


class _Clock:
    """テスト用の手動時計(cooldown 復帰を決定論的に検証するため)。"""
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def stubs():
    created = []

    def make(tag: str):
        srv, url = _start_stub(tag)
        created.append(srv)
        return srv, url

    yield make
    for srv in created:
        srv.shutdown()


def _server_of(resp: str) -> str:
    return json.loads(resp)["_server"]


def _route(fleet: FleetLLM, agent: int) -> str:
    resp = fleet.generate("こんにちは", rng_key=f"deliberate/{agent}/0",
                          temperature=0.0, max_tokens=8)
    return _server_of(resp)


# ---- (a) sticky routing ----
def test_sticky_same_agent_same_server(stubs):
    _srv_a, url_a = stubs("A")
    _srv_b, url_b = stubs("B")
    fleet = FleetLLM([url_a, url_b], "qwen3-8b")

    first = {a: _route(fleet, a) for a in range(24)}
    # 何度呼んでも同じ agent は同じサーバ
    for _ in range(3):
        for a in range(24):
            assert _route(fleet, a) == first[a]
    # 24体で両サーバが使われている(sticky でも偏りすぎない = 分散している)
    assert set(first.values()) == {"A", "B"}


# ---- (b) フェイルオーバ + cooldown 後の復帰 ----
def test_failover_and_cooldown_recovery(stubs):
    srv_a, url_a = stubs("A")
    _srv_b, url_b = stubs("B")
    clock = _Clock()
    fleet = FleetLLM([url_a, url_b], "qwen3-8b", cooldown_s=5.0, now=clock)

    # 健全時に A へ吸着する agent を探す
    agent = next(a for a in range(50) if _route(fleet, a) == "A")

    # A を落とす → B へフェイルオーバ
    srv_a.fail = True
    assert _route(fleet, agent) == "B"
    # A は cooldown 中なので、同じ agent は B を使い続ける
    assert _route(fleet, agent) == "B"

    # A 復旧 + cooldown 経過 → sticky 先の A へ戻る
    srv_a.fail = False
    clock.advance(6.0)
    assert _route(fleet, agent) == "A"


def test_all_servers_down_returns_error(stubs):
    srv_a, url_a = stubs("A")
    srv_b, url_b = stubs("B")
    srv_a.fail = True
    srv_b.fail = True
    fleet = FleetLLM([url_a, url_b], "qwen3-8b")
    resp = fleet.generate("x", rng_key="deliberate/1/0",
                          temperature=0.0, max_tokens=8)
    assert resp.startswith("__vllm_error__") or resp.startswith("__fleet_error__")


# ---- (c) name が URL 非依存(D13)----
def test_name_is_url_independent():
    f1 = FleetLLM(["http://a:1"], "qwen3-8b")
    f2 = FleetLLM(["http://b:2", "http://c:3", "http://d:4"], "qwen3-8b")
    assert f1.name == f2.name == "fleet/qwen3-8b"
    v1 = VllmBackend("qwen3-8b", "http://whatever:9000")
    v2 = VllmBackend("qwen3-8b", "http://elsewhere:1234")
    assert v1.name == v2.name == "vllm/qwen3-8b"


# ---- (d) CachedLLM と組んだとき同一プロンプトはキャッシュヒット ----
def test_cached_fleet_hits_cache(tmp_path, stubs):
    srv_a, url_a = stubs("A")
    fleet = FleetLLM([url_a], "qwen3-8b")
    cached = CachedLLM(fleet, enabled=True, path=tmp_path / "cache.jsonl")

    r1, _id1, c1 = cached.generate("同じプロンプト", rng_key="deliberate/1/0",
                                   temperature=0.0, max_tokens=8)
    assert c1 is False
    hits_before = len(srv_a.hits)

    r2, _id2, c2 = cached.generate("同じプロンプト", rng_key="deliberate/1/0",
                                   temperature=0.0, max_tokens=8)
    assert c2 is True
    assert r2 == r1
    assert len(srv_a.hits) == hits_before      # 2回目はサーバを叩かない


# ---- tier seam: purpose 別プール ----
def test_tiers_route_by_purpose(stubs):
    _srv_a, url_a = stubs("A")   # reflect 専用
    _srv_b, url_b = stubs("B")   # default
    fleet = FleetLLM([url_a, url_b], "qwen3-8b",
                     tiers={"reflect": [url_a], "default": [url_b]})
    # reflect purpose は常に A、それ以外は常に B
    for a in range(10):
        r = fleet.generate("x", rng_key=f"reflect/{a}/0",
                           temperature=0.0, max_tokens=8)
        assert _server_of(r) == "A"
        r = fleet.generate("x", rng_key=f"deliberate/{a}/0",
                           temperature=0.0, max_tokens=8)
        assert _server_of(r) == "B"


# ---- vLLM バックエンド: chat フォールバックと JSON 抽出 ----
def test_vllm_completions_and_chat_extract(stubs):
    _srv, url = stubs("A")
    vb = VllmBackend("qwen3-8b", url)
    resp = vb.generate("hi", rng_key="deliberate/1/0",
                       temperature=0.0, max_tokens=8)
    assert json.loads(resp)["action"] == "wander"
