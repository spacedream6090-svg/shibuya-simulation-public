"""制約付きデコードノブ model.format(第33バッチ 計画A)のテスト。

設計(docs/research/constrained-decoding.md §5):
- 既定 format=json は従来挙動そのまま(payload・キャッシュキーとも従来と同一=過去ラン再生互換)。
- think=True の呼には format を送らない境界ガード(GBNF が qwen3 の思考を殺す実測)。
- none のみキャッシュキーが変わる(切替時の旧応答誤再生の防止)。
"""
from __future__ import annotations

import json

import pytest

from society.config import load_config
from society.llm.cache import CachedLLM
from society.llm.ollama import OllamaBackend
from society.llm.vllm import VllmBackend


def _capture_payload(monkeypatch):
    """urllib.request.urlopen を横取りして送信 payload を記録する。"""
    sent: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": '{"action":"wander"}'}).encode()

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data.decode("utf-8")))
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


# ------------------------------------------------------------------ payload
def test_default_json_payload_unchanged(monkeypatch):
    """既定(json・think=False)は従来どおり format:"json" を送る。"""
    sent = _capture_payload(monkeypatch)
    OllamaBackend("m").generate("p", rng_key="k", temperature=0.7,
                                max_tokens=32)
    assert sent[0]["format"] == "json"


def test_think_guard_omits_format(monkeypatch):
    """think=True の呼には format を送らない(思考殺しの境界ガード)。"""
    sent = _capture_payload(monkeypatch)
    OllamaBackend("m").generate("p", rng_key="k", temperature=0.7,
                                max_tokens=32, think=True)
    assert "format" not in sent[0] and sent[0]["think"] is True


def test_format_none_omits_format(monkeypatch):
    sent = _capture_payload(monkeypatch)
    OllamaBackend("m", format_mode="none").generate(
        "p", rng_key="k", temperature=0.7, max_tokens=32)
    assert "format" not in sent[0]


def test_invalid_format_rejected():
    with pytest.raises(ValueError):
        OllamaBackend("m", format_mode="yaml")
    with pytest.raises(ValueError):
        VllmBackend("m", format_mode="yaml")


def test_vllm_format_none_skips_response_format(monkeypatch):
    """vLLM: format=none なら response_format を一切送らない。think=True も同様。"""
    sent = _capture_payload(monkeypatch)
    VllmBackend("m", format_mode="none").generate(
        "p", rng_key="k", temperature=0.7, max_tokens=32)
    assert "response_format" not in sent[0]
    sent.clear()
    VllmBackend("m").generate("p", rng_key="k", temperature=0.7,
                              max_tokens=32, think=True)
    assert "response_format" not in sent[0]
    sent.clear()
    VllmBackend("m").generate("p", rng_key="k", temperature=0.7, max_tokens=32)
    assert sent[0]["response_format"] == {"type": "json_object"}


# ------------------------------------------------------------------ cache キー
class _Fixed:
    """応答固定のダミーバックエンド(cache_extra を任意設定)。"""

    def __init__(self, name="ollama/m", extra=None):
        self.name = name
        self.cache_extra = extra

    def generate(self, prompt, *, rng_key, temperature, max_tokens,
                 think=False):
        return "r"


def _legacy_key(name, prompt, t, m, think):
    """従来(第33バッチ以前)のキー式を再現=後方互換の照合対象。"""
    import hashlib
    blob = json.dumps({"model": name, "t": t, "m": m, "think": bool(think),
                       "p": prompt}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_cache_key_backward_compatible_for_default_json():
    """format=json(cache_extra=None)のキーは従来式と完全一致=過去ラン再生互換。"""
    c = CachedLLM(_Fixed(extra=None))
    assert c._key("p", 0.7, 32, False) == _legacy_key("ollama/m", "p", 0.7, 32, False)
    assert c._key("p", 0.7, 32, True) == _legacy_key("ollama/m", "p", 0.7, 32, True)


def test_cache_key_differs_for_format_none():
    """format=none は別キー(切替時に旧応答を誤再生しない)。"""
    k_json = CachedLLM(_Fixed(extra=None))._key("p", 0.7, 32, False)
    k_none = CachedLLM(_Fixed(extra={"f": "none"}))._key("p", 0.7, 32, False)
    assert k_json != k_none


def test_backend_cache_extra_contract():
    """バックエンドの cache_extra 規約: json→None / none→{"f":"none"}。"""
    assert OllamaBackend("m").cache_extra is None
    assert OllamaBackend("m", format_mode="none").cache_extra == {"f": "none"}
    assert VllmBackend("m").cache_extra is None
    assert VllmBackend("m", format_mode="none").cache_extra == {"f": "none"}


# ------------------------------------------------------------------ config 配線
def test_config_default_is_json():
    cfg = load_config([])
    assert str(cfg.model.get("format", "json")) == "json"


def test_dotlist_override_reaches_backend():
    cfg = load_config(["model.format=none"])
    b = OllamaBackend("m", format_mode=str(cfg.model.get("format", "json")))
    assert b.format_mode == "none" and b.cache_extra == {"f": "none"}
