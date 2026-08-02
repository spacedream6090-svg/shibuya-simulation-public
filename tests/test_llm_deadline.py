"""LLM 1 呼の**絶対時限**(model.call_deadline_s)= 第86バッチ保守 M-1。

2026-08-02 夜間ラン実測: トークンが細々と流れ続ける病的生成では `model.timeout_s`
(= urlopen のソケットタイムアウト = **無通信区間**しか測れない)が一度も発火せず、
1 呼が 1 時間 47 分張り付いた。ここでは**その病的サーバをローカルに立てて**、

  (A) 実発火    : 細々と流し続けるサーバに対し、絶対時限で必ず切れる(秒で返る)
  (B) 非影響    : 即答する正常サーバでは 1 バイトも挙動が変わらない(応答も件数も)
  (C) 合流      : 超過は既存のタイムアウト→fallback 経路("__*_error__: ...")に合流する
  (D) 観測      : deadline_exceeded カウンタが増え、summary/watchdog_llm が拾える

を固定する。ソケットは 127.0.0.1 の**エフェメラルポート**(port=0)なので、xdist 並列でも
ポート衝突しない。全テストはスタブサーバ内で完結し、外部ネットワークへは一切出ない。
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from society.llm import deadline as dl
from society.llm.anthropic import AnthropicBackend
from society.llm.ollama import OllamaBackend
from society.llm.openai_compat import OpenAICompatBackend
from society.llm.vllm import VllmBackend


# --------------------------------------------------------------------------- #
# 病的サーバ(生の TCP。http.server は本文を一括で書くため「細々流す」を再現できない)
# --------------------------------------------------------------------------- #
class StubServer:
    """`mode` に応じて応答する最小 HTTP サーバ。

    - "fast"    : 完全な JSON を即返す(正常呼)。
    - "trickle" : Content-Length を宣言してから本文を **1 バイトずつ** 遅く流し続ける。
                  無通信区間は常に短いので **read timeout は永久に発火しない** =
                  夜間ランで観測された病的生成そのもの。
    """

    def __init__(self, mode: str, body: dict | None = None,
                 chunk_interval_s: float = 0.02):
        self.mode = mode
        self.body = json.dumps(body if body is not None else
                               {"response": "ok",
                                "choices": [{"text": "ok",
                                             "message": {"content": "ok"}}],
                                "content": [{"type": "text", "text": "ok"}]})
        self.chunk_interval_s = chunk_interval_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

    # -- 内部 -------------------------------------------------------------- #
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5.0)
            self._read_request(conn)
            raw = self.body.encode("utf-8")
            head = ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n")
            conn.sendall(head.encode("ascii"))
            if self.mode == "fast":
                conn.sendall(raw)
                return
            # trickle: 本文を 1 バイトずつ、**永久に終わらない**ペースで流す
            for i in range(len(raw)):
                if self._stop.is_set():
                    return
                conn.sendall(raw[i:i + 1])
                time.sleep(self.chunk_interval_s)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _read_request(conn: socket.socket) -> None:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1].strip())
        while len(rest) < length:
            chunk = conn.recv(4096)
            if not chunk:
                return
            rest += chunk


@pytest.fixture
def counter_reset():
    dl.reset_counter()
    yield
    dl.reset_counter()


# --------------------------------------------------------------------------- #
# (A)(C)(D) 実発火: 病的生成が絶対時限で切れて fallback 経路へ合流する
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make, prefix", [
    (lambda url: OllamaBackend("m", host=url, timeout_s=30.0, deadline_s=0.5),
     "__ollama_error__"),
    (lambda url: VllmBackend("m", url, timeout_s=30.0, deadline_s=0.5),
     "__vllm_error__"),
    (lambda url: OpenAICompatBackend("m", base_url=url, timeout_s=30.0,
                                     deadline_s=0.5), "__api_error__"),
])
def test_trickling_response_is_cut_by_deadline(make, prefix, counter_reset):
    """細々流し続けるサーバでも絶対時限で切れ、既存の fallback 文字列で返る。

    `timeout_s=30` は**発火しない**(無通信区間は 0.02s しかない)。時限 0.5s だけが効く。
    """
    srv = StubServer("trickle")
    try:
        be = make(srv.url)
        t0 = time.monotonic()
        out = be.generate("x", rng_key="plan/1/0", temperature=0.2, max_tokens=8)
        elapsed = time.monotonic() - t0
    finally:
        srv.close()
    assert out.startswith(prefix), f"fallback 経路に合流していない: {out[:120]}"
    assert "deadline" in out, f"時限超過であることが応答に出ていない: {out[:120]}"
    assert elapsed < 10.0, f"時限で切れていない(実測 {elapsed:.1f}s)"
    assert dl.exceeded_count() == 1


def test_anthropic_backend_also_has_the_deadline(counter_reset, monkeypatch):
    """HTTP を張る backend は 4 本とも同じ穴を持つので anthropic も塞いである。"""
    monkeypatch.setenv("_TEST_ANTHROPIC_KEY", "dummy-not-a-real-key")
    srv = StubServer("trickle")
    try:
        be = AnthropicBackend("m", api_key_env="_TEST_ANTHROPIC_KEY",
                              base_url=srv.url, timeout_s=30.0, deadline_s=0.5)
        out = be.generate("x", rng_key="plan/1/0", temperature=0.2, max_tokens=8)
    finally:
        srv.close()
    assert out.startswith("__anthropic_error__") and "deadline" in out
    assert dl.exceeded_count() == 1


# --------------------------------------------------------------------------- #
# (B) 非影響: 正常呼(即答)は挙動も件数も変わらない
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make", [
    lambda url, dls: OllamaBackend("m", host=url, timeout_s=5.0, deadline_s=dls),
    lambda url, dls: VllmBackend("m", url, timeout_s=5.0, deadline_s=dls),
    lambda url, dls: OpenAICompatBackend("m", base_url=url, timeout_s=5.0,
                                         deadline_s=dls),
])
@pytest.mark.parametrize("deadline_s", [300.0, 0.0])   # 既定値 / 無効(従来経路)
def test_normal_call_is_untouched(make, deadline_s, counter_reset):
    """既定 300s でも無効 0 でも、正常な応答は同一で超過カウンタは 0 のまま。"""
    srv = StubServer("fast")
    try:
        out = make(srv.url, deadline_s).generate(
            "x", rng_key="plan/1/0", temperature=0.2, max_tokens=8)
    finally:
        srv.close()
    assert out == "ok"
    assert dl.exceeded_count() == 0


def test_disabled_deadline_lets_the_pathological_call_hang(counter_reset):
    """0 以下 = 完全無効(従来経路)。時限では切れず、読取タイムアウトだけが効く。

    「既定を無効にすれば従来と同じ穴に戻る」= 挙動差が時限**だけ**に閉じている証拠。
    ここでは timeout_s も短くしてテストが停止しないようにする(trickle は
    無通信区間が短いので、実際には timeout_s でも切れず Content-Length ぶん流れ切る)。
    """
    srv = StubServer("trickle", body={"response": "ok"}, chunk_interval_s=0.005)
    try:
        be = OllamaBackend("m", host=srv.url, timeout_s=5.0, deadline_s=0.0)
        out = be.generate("x", rng_key="plan/1/0", temperature=0.2, max_tokens=8)
    finally:
        srv.close()
    assert out == "ok"                    # 最後まで待って読み切った(= 時限が無い)
    assert dl.exceeded_count() == 0


# --------------------------------------------------------------------------- #
# 配線(conf → backend)
# --------------------------------------------------------------------------- #
def test_config_key_default_is_300():
    from society.config import load_config
    cfg = load_config([])
    assert float(cfg.model.call_deadline_s) == 300.0


def test_config_override_reaches_the_backend(tmp_path):
    """model.call_deadline_s が ollama/vllm/router 子まで届く(mock は無関係)。"""
    from society.config import load_config
    from society.engine.simulation import Simulation
    cfg = load_config([f"run.out_dir={tmp_path.as_posix()}", "run.name=dl",
                       "run.n_agents=2", "run.n_steps=1",
                       "model.backend=ollama", "model.call_deadline_s=42"])
    sim = Simulation(cfg)
    raw = sim.llm.backend if hasattr(sim.llm, "backend") else sim.llm
    assert float(raw.deadline_s) == 42.0
    child = sim._build_router_child({"backend": "vllm", "name": "m"})
    assert float(child.deadline_s) == 42.0
    child2 = sim._build_router_child({"backend": "openai_compat", "name": "m",
                                      "call_deadline_s": 7})
    assert float(child2.deadline_s) == 7.0        # 子 spec の個別指定が勝つ


def test_fleet_passes_deadline_to_children():
    from society.llm.fleet import FleetLLM
    fl = FleetLLM(["http://a:8000", "http://b:8000"], "m", deadline_s=12.0,
                  tiers={"reflect": ["http://c:8000"],
                         "default": ["http://a:8000"]})
    assert {float(b.deadline_s) for b in fl._backend.values()} == {12.0}


# --------------------------------------------------------------------------- #
# 観測(summary / watchdog_llm)
# --------------------------------------------------------------------------- #
def test_summary_omits_the_key_when_no_call_was_cut(tmp_path):
    """0 件なら summary.json にキーが生えない(既存ランの summary とバイト一致)。"""
    from society.config import load_config
    from society.engine.simulation import Simulation
    dl.reset_counter()
    cfg = load_config([f"run.out_dir={tmp_path.as_posix()}", "run.name=nodl",
                       "run.n_agents=3", "run.n_steps=2"])
    sim = Simulation(cfg)
    sim.run()
    summ = json.loads((tmp_path / "nodl" / "summary.json").read_text(encoding="utf-8"))
    assert "llm_deadline_exceeded" not in summ


def test_watchdog_llm_reports_deadline_exceeded(tmp_path):
    """watchdog_llm が summary.json の件数を読んで警告対象にする。"""
    import importlib.util
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_m1_watchdog_llm", repo / "scripts" / "watchdog_llm.py")
    wl = importlib.util.module_from_spec(spec)
    sys.modules["_m1_watchdog_llm"] = wl
    spec.loader.exec_module(wl)

    run = tmp_path / "r"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(
        {"llm_calls": 10, "event_kinds": {"fallback": 0},
         "llm_deadline_exceeded": 3}), encoding="utf-8")
    assert wl.check_run(run)["deadline_exceeded"] == 3

    old = tmp_path / "old"                       # キーの無い旧ラン = 0 件扱い
    old.mkdir()
    (old / "summary.json").write_text(json.dumps({"llm_calls": 1}), encoding="utf-8")
    assert wl.check_run(old)["deadline_exceeded"] == 0

    none = tmp_path / "none"                     # summary 自体が無い = 欠測(None)
    none.mkdir()
    assert wl.check_run(none)["deadline_exceeded"] is None
