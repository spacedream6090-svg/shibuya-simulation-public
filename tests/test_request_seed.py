"""β11(第117バッチ 2026-08-16。レーン B1 安全系): vLLM への request-level stable seed。

正典: docs/plans/beta-implementation-plan.md §1 β11 / docs/plans/external-audit-triage.md F14。

受入基準:
  - **既定 OFF では送出ボディがバイト単位で従来と同一**(seed キー自体を作らない)
  - ON で同一 rng_key → 同一 seed(プロセス跨ぎで安定)/ 材料が 1 つ違えば別 seed
  - seed は vLLM が受ける範囲(0 <= seed <= 0x7fffffff)
  - 艦隊(FleetLLM)でも同じ値(sticky 先が変わっても seed は動かない)
  - journal に seed が残る(OFF では欄自体を出さない)
  - conf `llm.request_seed.enabled` は基底 conf に既定 false で宣言済み・レジストリ登録済み
  - プロンプトを 1 バイトも変えない = CachedLLM のキャッシュキーが不変(過去ラン再生互換)
"""
from __future__ import annotations

import json

from society import registry as R
from society.config import load_config
from society.llm.cache import CachedLLM
from society.llm.fleet import FleetLLM
from society.llm.journal import LlmJournal, iter_records
from society.llm.vllm import VllmBackend, split_rng_key, stable_request_seed

KEY = "deliberate/7/12"


def _capture_payload(monkeypatch):
    """urllib.request.urlopen を横取りして送信 payload の **生バイト列**を記録する。"""
    sent: list[bytes] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"text": '{"action":"wander"}'}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent.append(req.data)
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


# --------------------------------------------------------------------------- #
# (1) OFF = 送出ボディが 1 バイトも変わらない
# --------------------------------------------------------------------------- #
def test_off_body_is_byte_identical(monkeypatch):
    """既定(request_seed 未指定)の送出ボディが従来と**バイト同一**であること。"""
    sent = _capture_payload(monkeypatch)
    VllmBackend("m").generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    legacy = json.dumps({"model": "m", "prompt": "p\n/no_think", "stream": False,
                         "temperature": 0.7, "max_tokens": 32,
                         "response_format": {"type": "json_object"}}).encode("utf-8")
    assert sent[0] == legacy, "OFF なのに送出ボディが変わっている"
    assert b"seed" not in sent[0]


def test_off_chat_body_has_no_seed(monkeypatch):
    sent = _capture_payload(monkeypatch)
    be = VllmBackend("m")
    be._mode = "chat"
    be.generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    assert b"seed" not in sent[0]


def test_default_backend_reports_no_seed():
    assert VllmBackend("m").request_seed is None
    assert VllmBackend("m").request_seed_for(KEY) is None


# --------------------------------------------------------------------------- #
# (2) ON = 安定 seed が載る
# --------------------------------------------------------------------------- #
def test_on_body_carries_seed(monkeypatch):
    sent = _capture_payload(monkeypatch)
    be = VllmBackend("m", request_seed=42)
    be.generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    body = json.loads(sent[0].decode("utf-8"))
    assert body["seed"] == stable_request_seed(42, KEY)
    # seed 以外は OFF のときと 1 つも変わらない(足したのは 1 欄だけ)
    assert {k: v for k, v in body.items() if k != "seed"} == {
        "model": "m", "prompt": "p\n/no_think", "stream": False,
        "temperature": 0.7, "max_tokens": 32,
        "response_format": {"type": "json_object"}}


def test_on_chat_body_carries_seed(monkeypatch):
    sent = _capture_payload(monkeypatch)
    be = VllmBackend("m", request_seed=42)
    be._mode = "chat"
    be.generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    assert json.loads(sent[0].decode("utf-8"))["seed"] == stable_request_seed(42, KEY)


def test_same_key_same_seed_different_key_different_seed():
    assert stable_request_seed(42, KEY) == stable_request_seed(42, KEY)
    assert stable_request_seed(42, KEY) != stable_request_seed(43, KEY)
    for other in ("deliberate/7/13", "deliberate/8/12", "plan/7/12",
                  "deliberate/7/12/retry"):
        assert stable_request_seed(42, KEY) != stable_request_seed(42, other), other


def test_seed_range_is_accepted_by_vllm():
    """vLLM/OpenAI 互換 API が受ける範囲(正の 31bit)に収まる。"""
    for i in range(200):
        s = stable_request_seed(42, f"deliberate/{i}/{i * 7}")
        assert isinstance(s, int) and 0 <= s <= 0x7FFFFFFF


def test_seed_is_process_stable():
    """プロセス非依存(blake2b)= 事後に同じ値を再計算できる(既知の固定値)。"""
    assert stable_request_seed(42, KEY) == stable_request_seed(42, "deliberate/7/12")
    assert stable_request_seed(0, "") == stable_request_seed(0, "")


def test_rng_key_split_does_not_fabricate():
    assert split_rng_key("deliberate/7/12") == ("deliberate", "7", "12", "")
    assert split_rng_key("plan/7/12/retry") == ("plan", "7", "12", "retry")
    assert split_rng_key("bare") == ("bare", "", "", "")


# --------------------------------------------------------------------------- #
# (3) 艦隊(FleetLLM)
# --------------------------------------------------------------------------- #
def test_fleet_children_share_the_same_seed():
    fleet = FleetLLM(["http://a:8000", "http://b:8000"], "m", request_seed=42)
    assert fleet.request_seed_for(KEY) == stable_request_seed(42, KEY)
    for child in fleet._backend.values():
        assert child.request_seed_for(KEY) == fleet.request_seed_for(KEY)


def test_fleet_default_is_off():
    fleet = FleetLLM(["http://a:8000"], "m")
    assert fleet.request_seed is None and fleet.request_seed_for(KEY) is None
    assert all(c.request_seed is None for c in fleet._backend.values())


# --------------------------------------------------------------------------- #
# (4) journal 記録(OFF では欄自体を出さない)
# --------------------------------------------------------------------------- #
class _Fixed:
    """応答固定のダミー backend(request_seed_for の有無を切り替える)。"""

    def __init__(self, seed=None):
        self.name = "vllm/m"
        self.cache_extra = None
        self._seed = seed

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        return "r"

    def request_seed_for(self, rng_key):
        return None if self._seed is None else stable_request_seed(self._seed, rng_key)


def _one_record(tmp_path, backend, name):
    j = LlmJournal(tmp_path / name)
    llm = CachedLLM(backend, enabled=False, journal=j)
    llm.generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    j.flush()
    return list(iter_records(tmp_path / name))[0]


def test_journal_records_seed_when_on(tmp_path):
    row = _one_record(tmp_path, _Fixed(seed=42), "on.jsonl.gz")
    assert row["seed"] == stable_request_seed(42, KEY)


def test_journal_has_no_seed_field_when_off(tmp_path):
    row = _one_record(tmp_path, _Fixed(seed=None), "off.jsonl.gz")
    assert "seed" not in row, "OFF なのに欄が増えている(既存ランと同形でない)"


def test_journal_has_no_seed_field_for_backends_without_the_hook(tmp_path):
    """mock / ollama / API 系(request_seed_for を持たない)は素通り。"""
    class _NoHook:
        name = "mock"
        cache_extra = None

        def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
            return "r"

    assert not hasattr(_NoHook(), "request_seed_for")
    row = _one_record(tmp_path, _NoHook(), "nohook.jsonl.gz")
    assert "seed" not in row


def test_journal_seed_matches_what_was_sent(tmp_path, monkeypatch):
    """記録した seed と実際に送出した seed が同じ(口が 1 つしかないことの検査)。"""
    sent = _capture_payload(monkeypatch)
    j = LlmJournal(tmp_path / "match.jsonl.gz")
    llm = CachedLLM(VllmBackend("m", request_seed=42), enabled=False, journal=j)
    llm.generate("p", rng_key=KEY, temperature=0.7, max_tokens=32)
    j.flush()
    row = list(iter_records(tmp_path / "match.jsonl.gz"))[0]
    assert row["seed"] == json.loads(sent[0].decode("utf-8"))["seed"]


# --------------------------------------------------------------------------- #
# (5) キャッシュキー不変(過去ランの llm_cache 再生互換)
# --------------------------------------------------------------------------- #
def test_cache_key_is_unaffected_by_request_seed():
    off = CachedLLM(VllmBackend("m"))._key("p", 0.7, 32, False)
    on = CachedLLM(VllmBackend("m", request_seed=42))._key("p", 0.7, 32, False)
    assert off == on, "seed がキャッシュキーに漏れている(過去ラン再生互換が壊れる)"


# --------------------------------------------------------------------------- #
# (6) conf 配線
# --------------------------------------------------------------------------- #
def test_conf_declares_request_seed_off_by_default():
    cfg = load_config([])
    assert cfg.llm.request_seed.enabled is False
    assert "llm.request_seed.enabled" in {f.id for f in R.FEATURES}
    assert "llm.request_seed.enabled" not in R.undeclared_toggles(cfg)


def test_simulation_wires_the_toggle(tmp_path):
    from society.engine.simulation import Simulation

    def _sim(name, **ov):
        dot = ["run.seed=42", "run.n_agents=4", "run.n_steps=2",
               "model.backend=mock"] + [f"{k}={v}" for k, v in ov.items()]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    assert _sim("off").llm_request_seed is None
    assert _sim("on", **{"llm.request_seed.enabled": "true"}).llm_request_seed == 42
