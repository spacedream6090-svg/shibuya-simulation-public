"""vLLM の HTTP 経路トグル `model.api_mode`(A8 実測 2026-08-17)のテスト。

背景(A8 トーナメント実測): 同一プロンプト束・同一 Qwen3-8B で
  raw /v1/completions 経路 = parse 成立 58.7%
  /v1/chat/completions 経路(chat template + chat_template_kwargs) = 99.0%
raw completions はチャットテンプレートが適用されず指示追従モードに入らないため。

受入基準:
  - 既定(api_mode 未指定)= /v1/completions 宛て・ボディは従来とバイト一致・キャッシュキー不変
  - api_mode="chat" = **最初から** /v1/chat/completions 宛て・enable_thinking が think に追随
  - think 境界ガード(think=True には response_format を送らない)は chat でも維持
  - 不正値は ValueError(format_mode と同じ流儀)
  - chat のときだけ cache_extra にマーカー(旧経路の応答を誤再生しない)
  - conf に api_mode が宣言済み(基底=completions / 本選 conf・プロファイル=chat)
"""
from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from society.config import load_config
from society.llm.cache import CachedLLM
from society.llm.fleet import FleetLLM
from society.llm.vllm import VllmBackend, cache_extra_for


def _capture(monkeypatch):
    """urlopen を横取りして (URL, 生ボディ) を記録する。"""
    sent: list[tuple[str, bytes]] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"text": '{"action":"wander"}',
                             "message": {"content": '{"action":"wander"}'}}]
            }).encode()

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.data))
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


# --------------------------------------------------------------------------- #
# (1) 既定 = 従来どおり completions(バイト不変)
# --------------------------------------------------------------------------- #
def test_default_hits_completions_with_legacy_body(monkeypatch):
    sent = _capture(monkeypatch)
    be = VllmBackend("m", "http://x:8000")
    assert be.api_mode == "completions" and be._mode == "completions"
    be.generate("p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    url, body = sent[0]
    assert url == "http://x:8000/v1/completions"
    legacy = json.dumps({"model": "m", "prompt": "p\n/no_think", "stream": False,
                         "temperature": 0.7, "max_tokens": 32,
                         "response_format": {"type": "json_object"}}).encode("utf-8")
    assert body == legacy, "既定なのに送出ボディが変わっている"


def test_default_think_true_body_unchanged(monkeypatch):
    """think=True(既定経路)も従来どおり: /no_think を付けず response_format も無し。"""
    sent = _capture(monkeypatch)
    VllmBackend("m").generate("p", rng_key="reflect/7/12", temperature=0.7,
                              max_tokens=32, think=True)
    body = json.loads(sent[0][1].decode("utf-8"))
    assert body["prompt"] == "p" and "response_format" not in body


# --------------------------------------------------------------------------- #
# (2) api_mode="chat" = 最初から chat/completions
# --------------------------------------------------------------------------- #
def test_chat_mode_hits_chat_endpoint_first(monkeypatch):
    sent = _capture(monkeypatch)
    be = VllmBackend("m", "http://x:8000", api_mode="chat")
    assert be._mode == "chat"
    be.generate("p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    url, raw = sent[0]
    assert url == "http://x:8000/v1/chat/completions"
    body = json.loads(raw.decode("utf-8"))
    assert body["messages"] == [{"role": "user", "content": "p"}]
    assert "prompt" not in body and "/no_think" not in raw.decode("utf-8")


@pytest.mark.parametrize("think", [False, True])
def test_chat_mode_enable_thinking_follows_flag(monkeypatch, think):
    sent = _capture(monkeypatch)
    VllmBackend("m", api_mode="chat").generate(
        "p", rng_key="reflect/7/12", temperature=0.7, max_tokens=32, think=think)
    body = json.loads(sent[0][1].decode("utf-8"))
    assert body["chat_template_kwargs"] == {"enable_thinking": think}


def test_chat_mode_think_guard_on_response_format(monkeypatch):
    """think 境界ガードは chat でも維持: False→response_format 有 / True→無。"""
    sent = _capture(monkeypatch)
    be = VllmBackend("m", api_mode="chat")
    be.generate("p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    assert json.loads(sent[0][1].decode("utf-8"))["response_format"] == {
        "type": "json_object"}
    sent.clear()
    be.generate("p", rng_key="reflect/7/12", temperature=0.7, max_tokens=32,
                think=True)
    assert "response_format" not in json.loads(sent[0][1].decode("utf-8"))
    sent.clear()
    VllmBackend("m", api_mode="chat", format_mode="none").generate(
        "p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    assert "response_format" not in json.loads(sent[0][1].decode("utf-8"))


def test_chat_mode_extracts_message_content(monkeypatch):
    _capture(monkeypatch)
    out = VllmBackend("m", api_mode="chat").generate(
        "p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    assert out == '{"action":"wander"}'


# --------------------------------------------------------------------------- #
# (3) 検証
# --------------------------------------------------------------------------- #
def test_invalid_api_mode_rejected():
    with pytest.raises(ValueError):
        VllmBackend("m", api_mode="chat_completions")
    with pytest.raises(ValueError):
        VllmBackend("m", api_mode="")
    with pytest.raises(ValueError):
        FleetLLM(["http://x:8000"], "m", api_mode="responses")


# --------------------------------------------------------------------------- #
# (4) キャッシュキー(D13)
# --------------------------------------------------------------------------- #
def test_cache_extra_contract():
    assert VllmBackend("m").cache_extra is None                      # 従来キー互換
    assert VllmBackend("m", format_mode="none").cache_extra == {"f": "none"}
    assert VllmBackend("m", api_mode="chat").cache_extra == {"api": "chat"}
    assert VllmBackend("m", format_mode="none",
                       api_mode="chat").cache_extra == {"f": "none", "api": "chat"}


def test_cache_extra_helper_is_pure():
    assert cache_extra_for("json") is None
    assert cache_extra_for("json", "completions") is None
    assert cache_extra_for("none", "completions") == {"f": "none"}
    assert cache_extra_for("json", "chat") == {"api": "chat"}


def test_cache_key_changes_only_for_chat():
    k_default = CachedLLM(VllmBackend("m"))._key("p", 0.7, 32, False)
    k_chat = CachedLLM(VllmBackend("m", api_mode="chat"))._key("p", 0.7, 32, False)
    assert k_default != k_chat, "chat が旧経路の応答を誤再生しうる"
    # 既定は format ノブ導入前後で不変(過去ランの llm_cache 再生互換)
    import hashlib
    blob = json.dumps({"model": "vllm/m", "t": 0.7, "m": 32, "think": False,
                       "p": "p"}, ensure_ascii=False, sort_keys=True)
    assert k_default == hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# (5) 艦隊(本選は 7 本 = FleetLLM 経路)へ透過する
# --------------------------------------------------------------------------- #
def test_fleet_propagates_api_mode(monkeypatch):
    fleet = FleetLLM(["http://a:8000", "http://b:8000"], "m", api_mode="chat",
                     tiers={"reflect": ["http://c:8000"]})
    assert fleet.api_mode == "chat"
    assert fleet.cache_extra == {"api": "chat"}
    assert all(b._mode == "chat" for b in fleet._backend.values())
    assert fleet._backend["http://c:8000"]._mode == "chat"   # tier だけの URL も
    sent = _capture(monkeypatch)
    fleet.generate("p", rng_key="deliberate/7/12", temperature=0.7, max_tokens=32)
    assert sent[0][0].endswith("/v1/chat/completions")


def test_fleet_default_is_completions():
    fleet = FleetLLM(["http://a:8000"], "m")
    assert fleet.api_mode == "completions" and fleet.cache_extra is None
    assert all(b._mode == "completions" for b in fleet._backend.values())


# --------------------------------------------------------------------------- #
# (6) conf 配線
# --------------------------------------------------------------------------- #
def test_config_default_is_completions():
    cfg = load_config([])
    assert str(cfg.model.get("api_mode", "completions")) == "completions"


def test_dotlist_override_reaches_backend():
    cfg = load_config(["model.api_mode=chat"])
    be = VllmBackend("m", api_mode=str(cfg.model.get("api_mode", "completions")))
    assert be._mode == "chat" and be.cache_extra == {"api": "chat"}


def test_finals_confs_declare_chat():
    """本選 conf と、そこへ貼るプロファイルの**両方**が chat を宣言している。

    片方だけだと「貼った側の model: ブロックに無い」=黙って completions へ戻る
    (plan_max_tokens 896 と同じ落とし穴)。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("conf/finals_observe.yaml", "conf/profiles/finals-vllm7.yaml"):
        node = OmegaConf.load(root / rel)
        assert str(node.model.get("api_mode", "")) == "chat", rel
