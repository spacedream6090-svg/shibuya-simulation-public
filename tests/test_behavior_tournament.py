"""A8 ミニ行動トーナメント(scripts/behavior_tournament.py)の検収。

ネットワークは一切使わない: 発行部は**缶詰応答の fake バックエンド**へ差し替え、
シナリオ束は小さな mock シム(LLM は society.llm.mock)から作る。

固定する性質:
  1. 束の決定論 — 同じ seed・同じ conf で 2 回作れば **同一の束**(bundle_sha256 一致)。
  2. 層別 — purpose 別の件数・時刻帯の散り・同一プロンプトの重複排除。
  3. 採点 — parse_action / classify_reject / 空 / 通信エラー / 応答長 の分類が仕様どおり。
  4. 配管 — TimedBackend が CachedLLM のキー材料(name / cache_extra / request seed)を
     素通しする(= 計測を挟んでもキャッシュキーが変わらない)。
  5. パイプライン全体 — 2 モデル分の発行 → 集計 → report.md / results.parquet /
     samples.md が数字入りで出る。

house style: scripts/ を path 追加して import(tests/test_analyze_founders.py に倣う)。
"""
from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))

import pyarrow.parquet as pq                        # noqa: E402

import behavior_tournament as bt                    # noqa: E402


# --------------------------------------------------------------------------- #
# 道具: 合成ジャーナル / 缶詰バックエンド
# --------------------------------------------------------------------------- #
def _rec(seq: int, purpose: str, agent: int, step: int, prompt: str,
         *, temperature: float = 0.7, max_tokens: int = 320,
         think: bool = False) -> dict:
    return {"seq": seq, "key": f"k{seq}", "rng_key": f"{purpose}/{agent}/{step}",
            "backend": "mock", "temperature": temperature,
            "max_tokens": max_tokens, "think": think, "cached": False,
            "prompt": prompt, "response": "{}"}


def _synth_records(n_agents: int = 12, steps_per_day: int = 144) -> list[dict]:
    """時刻帯もエージェントも散った合成ジャーナル(1 プロンプト 1 レコード)。"""
    out: list[dict] = []
    seq = 0
    for step in range(0, steps_per_day, 6):          # 24 step = 全時刻帯を覆う
        for agent in range(n_agents):
            out.append(_rec(seq, "deliberate", agent, step,
                            f"状況: step={step} agent={agent} の熟慮"))
            seq += 1
    for agent in range(n_agents):
        out.append(_rec(seq, "plan", agent, 30, f"朝の計画 agent={agent}"))
        seq += 1
        out.append(_rec(seq, "reflect", agent, 130, f"夜の内省 agent={agent}",
                        max_tokens=1200, think=True))
        seq += 1
    return out


class FakeBackend:
    """缶詰応答のバックエンド(LLMBackend 互換)。ネットワーク不使用。"""

    def __init__(self, name: str, responses, cache_extra=None):
        self.name = name
        self.cache_extra = cache_extra
        self._responses = responses      # purpose -> [応答, ...](巡回)
        self._lock = threading.Lock()
        self.seen: list[str] = []
        self.seed_keys: list[str] = []

    def request_seed_for(self, rng_key: str) -> int:
        self.seed_keys.append(rng_key)
        return 12345

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        purpose = rng_key.split("/")[0]
        with self._lock:
            idx = len(self.seen)
            self.seen.append(rng_key)
        pool = self._responses.get(purpose) or self._responses.get("default") or ["{}"]
        return pool[idx % len(pool)]


_GOOD = {
    "deliberate": ['{"action": "speak", "text": "今日は人が多いね"}',
                   '{"action": "post", "text": "スクランブル、混みすぎ"}',
                   '{"action": "wander"}'],
    "plan": ['{"action": "plan", "items": [{"what": "仕事", "minutes": 480}]}'],
    "reflect": ['{"action": "reflect", "summary": "働いた一日",'
                ' "belief": "焦らずやろう",'
                ' "salient": [{"text": "会話", "importance": 7}]}'],
    "default": ['{"action": "speak", "text": "はい"}'],
}
_BAD = {
    "deliberate": ['{"action": "speak", "text": "途中で切れ',      # broken_json(寛容修復で復活)
                   'ぜんぜんJSONじゃない文章',                      # broken_json
                   '{"action": "teleport", "to": "月"}',            # unknown_action
                   '{"action": "speak"}',                           # missing_field
                   '',                                              # 空応答
                   '__vllm_error__: HTTP 500 Internal Server Error'],  # 通信エラー
    "plan": ['{"action": "plan", "items": []}'],                     # missing_field
    "reflect": ['{"action": "reflect"}'],                            # missing_field
    "default": [''],
}


def _bundle(per_purpose: int = 6) -> list[dict]:
    return bt.select_scenarios(_synth_records(), per_purpose=per_purpose,
                               start_min=0, dt_min=10)


# --------------------------------------------------------------------------- #
# 1. シナリオ束(層別・決定論・重複排除)
# --------------------------------------------------------------------------- #
def test_select_scenarios_is_stratified_and_pure():
    scen = _bundle(per_purpose=12)
    counts = {}
    for s in scen:
        counts[s["purpose"]] = counts.get(s["purpose"], 0) + 1
    assert counts == {"deliberate": 12, "plan": 12, "reflect": 12}
    delib = [s for s in scen if s["purpose"] == "deliberate"]
    # 時刻帯 6 区分すべてに散っている(先頭だけ採っても層が偏らない)
    assert len({s["bucket"] for s in delib}) == bt.N_TOD_BUCKETS
    # エージェントも散る(同一個体に集中しない)
    assert len({s["agent_id"] for s in delib}) >= 10
    # 送出パラメータはジャーナルの実値を引き継ぐ(内省は think=True・上限が別)
    ref = [s for s in scen if s["purpose"] == "reflect"]
    assert all(s["think"] and s["max_tokens"] == 1200 for s in ref)
    assert all(not s["think"] and s["max_tokens"] == 320 for s in delib)
    # scenario_id は purpose 内で連番・一意
    assert len({s["scenario_id"] for s in scen}) == len(scen)


def test_select_scenarios_deterministic_and_dedupes():
    records = _synth_records()
    a = bt.select_scenarios(records, per_purpose=10)
    b = bt.select_scenarios(list(records), per_purpose=10)
    assert bt.bundle_hash(a) == bt.bundle_hash(b)
    assert [s["prompt"] for s in a] == [s["prompt"] for s in b]
    # 同一プロンプト+同一パラメータの重複は 1 件に畳まれる
    dup = [_rec(0, "deliberate", 1, 10, "まったく同じ状況"),
           _rec(1, "deliberate", 2, 20, "まったく同じ状況"),
           _rec(2, "deliberate", 3, 30, "別の状況")]
    out = bt.select_scenarios(dup, purposes=("deliberate",), per_purpose=10)
    assert len(out) == 2
    # think 違いは別要求として残す(キャッシュキーが違うので同一視できない)
    pair = [_rec(0, "reflect", 1, 10, "同文", think=False),
            _rec(1, "reflect", 1, 20, "同文", think=True)]
    assert len(bt.select_scenarios(pair, purposes=("reflect",), per_purpose=10)) == 2


def test_bundle_hash_detects_any_change():
    scen = _bundle()
    other = [dict(s) for s in scen]
    other[0]["prompt_sha256"] = "0" * 64
    assert bt.bundle_hash(scen) != bt.bundle_hash(other)
    swapped = [scen[1], scen[0]] + scen[2:]
    assert bt.bundle_hash(scen) != bt.bundle_hash(swapped)


def test_scenarios_roundtrip(tmp_path):
    scen = _bundle()
    path = tmp_path / "scenarios.jsonl"
    bt.write_scenarios(path, scen)
    assert bt.bundle_hash(bt.read_scenarios(path)) == bt.bundle_hash(scen)


# --------------------------------------------------------------------------- #
# 2. 採点(cognition の本物を使う)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,parse_ok,json_ok,action,reason", [
    ('{"action": "speak", "text": "やあ"}', True, True, "speak", ""),
    ('{"action": "wander"}', True, True, "stay", ""),
    ('{"action": "plan", "items": [{"what": "x"}]}', True, True, "plan", ""),
    ('{"action": "reflect", "belief": "b"}', True, True, "reflect", ""),
    ('{"action": "speak", "text": "切れた', True, True, "speak", ""),   # 寛容修復
    ("ただの文章", False, False, bt.REJECT_LABEL, "broken_json"),
    ("", False, False, bt.REJECT_LABEL, "broken_json"),
    ('{"action": "speak"}', False, True, bt.REJECT_LABEL, "missing_field"),
    ('{"text": "action がない"}', False, True, bt.REJECT_LABEL, "missing_field"),
    ('{"action": "teleport", "to": "月"}', False, True, bt.REJECT_LABEL,
     "unknown_action"),
    ("__vllm_error__: HTTP 500 x", False, False, bt.REJECT_LABEL, "broken_json"),
])
def test_score_response_classification(text, parse_ok, json_ok, action, reason):
    got = bt.score_response(text)
    assert got["parse_ok"] is parse_ok
    assert got["json_ok"] is json_ok
    assert got["action_type"] == action
    assert got["reject_reason"] == reason
    assert got["resp_chars"] == len(text)


def test_score_response_empty_and_error_flags():
    assert bt.score_response("")["empty"] is True
    assert bt.score_response("   ")["empty"] is True
    err = bt.score_response("__vllm_error__: HTTP 500 x")
    assert err["error"] is True and err["empty"] is False   # エラーは「空」に数えない
    assert bt.score_response('{"action": "speak", "text": "a"}')["error"] is False
    assert bt.score_response('{"action": "teleport"}')["unknown_action"] == "teleport"


def test_percentile_and_jsd():
    assert bt.percentile([], 0.5) == 0.0
    assert bt.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert bt.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    same = {"speak": 10.0, "post": 5.0}
    assert bt.jsd(same, dict(same)) == 0.0
    assert bt.jsd({"speak": 1.0}, {"post": 1.0}) == pytest.approx(1.0)
    mid = bt.jsd({"speak": 8.0, "post": 2.0}, {"speak": 2.0, "post": 8.0})
    assert 0.0 < mid < 1.0
    assert bt.jsd({}, {"speak": 1.0}) == 0.0                # 片側が空でも落ちない


# --------------------------------------------------------------------------- #
# 3. 配管(TimedBackend が CachedLLM のキー材料を素通しする)
# --------------------------------------------------------------------------- #
def test_timed_backend_is_transparent_to_cache_key():
    inner = FakeBackend("vllm/qwen3-8b", _GOOD, cache_extra={"f": "none"})
    timed = bt.TimedBackend(inner)
    assert timed.name == inner.name
    assert timed.cache_extra == {"f": "none"}
    assert timed.request_seed_for("deliberate/1/2") == 12345
    out = timed.generate("p", rng_key="deliberate/1/2", temperature=0.7,
                         max_tokens=320)
    assert out == _GOOD["deliberate"][0]
    lat = timed.latency_of("deliberate/1/2", bt.sha256_text("p"))
    assert lat is not None and lat >= 0.0
    # request_seed_for を持たないバックエンドでも落ちない(mock / ollama 経路)
    class _Bare:
        name = "bare"

        def generate(self, prompt, *, rng_key, temperature, max_tokens,
                     think=False):
            return "{}"
    assert bt.TimedBackend(_Bare()).request_seed_for("x/1/2") is None


def test_vllm_backend_wiring_uses_stable_request_seed():
    """A 側と B 側で **同じシナリオに同じ request seed** が乗る(既存 blake2b 導出)。

    ソケットは開かない(バックエンドを組んで送出ボディを作るところまで)。
    """
    from society.llm.vllm import stable_request_seed
    common = dict(timeout_s=5.0, deadline_s=30.0, format_mode="json", seed=20260817)
    a = bt.build_vllm_backend(model="Qwen3-8B", endpoint="http://a:8000", **common)
    b = bt.build_vllm_backend(model="Qwen3-14B", endpoint="http://b:8006", **common)
    key = "deliberate/12/34"
    assert a.request_seed_for(key) == b.request_seed_for(key) \
        == stable_request_seed(20260817, key)
    assert a.request_seed_for("reflect/12/34") != a.request_seed_for(key)
    # name はモデル名ベース(URL 非依存)= CachedLLM のキーが URL で割れない(D13)
    assert a.name == "vllm/Qwen3-8B" and b.name == "vllm/Qwen3-14B"
    body = a._completions_body("p", 0.7, 320, False, True, a.request_seed_for(key))
    assert body["seed"] == stable_request_seed(20260817, key)
    assert body["response_format"] == {"type": "json_object"}
    # seed 指定なし = 送出ボディに seed 欄を作らない(既存ランとバイト一致の既定)
    plain = bt.build_vllm_backend(model="Qwen3-8B", endpoint="http://a:8000",
                                  timeout_s=5.0, deadline_s=30.0,
                                  format_mode="json", seed=None)
    assert plain.request_seed_for(key) is None


def test_issue_records_rows_latency_and_parallel_path():
    scen = _bundle(per_purpose=6)
    backend = FakeBackend("vllm/fake-8b", _GOOD)
    rows, wall = bt.issue(scen, backend, side="A", model="fake-8b",
                          endpoint="http://x", workers=4)
    assert len(rows) == len(scen) == len(backend.seen)      # 全件が 1 回ずつ発行される
    assert wall >= 0.0
    assert {r["side"] for r in rows} == {"A"}
    assert all(not math.isnan(r["latency_s"]) for r in rows)
    assert all(r["cached"] is False for r in rows)          # 既定はキャッシュ無効
    assert all(r["parse_ok"] for r in rows)
    # 応答順が要求順どおり(並列でも取り違えない)
    assert [r["scenario_id"] for r in rows] == [s["scenario_id"] for s in scen]


def test_issue_with_cache_reuses_and_marks_cached(tmp_path):
    scen = _bundle(per_purpose=4)
    doubled = scen + [dict(s, scenario_id=s["scenario_id"] + "b") for s in scen]
    backend = FakeBackend("vllm/fake-8b", _GOOD)
    rows, _ = bt.issue(doubled, backend, side="A", model="fake-8b",
                       endpoint="http://x", workers=2,
                       cache_path=tmp_path / "llm_cache.A.jsonl")
    assert len(backend.seen) == len(scen)                   # 2 周目は実発行しない
    assert sum(1 for r in rows if r["cached"]) == len(scen)
    assert (tmp_path / "llm_cache.A.jsonl").exists()


# --------------------------------------------------------------------------- #
# 4. 集計とレポート
# --------------------------------------------------------------------------- #
def _two_sided_rows():
    scen = _bundle(per_purpose=6)
    rows_a, _ = bt.issue(scen, FakeBackend("vllm/fake-8b", _GOOD), side="A",
                         model="fake-8b", endpoint="http://a", workers=2)
    rows_b, _ = bt.issue(scen, FakeBackend("vllm/fake-14b", _BAD), side="B",
                         model="fake-14b", endpoint="http://b", workers=2)
    return scen, rows_a + rows_b


def test_aggregate_metrics_are_filled():
    scen, rows = _two_sided_rows()
    stats = bt.aggregate(rows)
    a = stats["A|ALL"]
    b = stats["B|ALL"]
    assert a["n"] == b["n"] == len(scen)
    assert a["parse_rate"] == 1.0 and a["json_rate"] == 1.0
    assert b["parse_rate"] < a["parse_rate"]                # 壊れた応答を出す側が下がる
    assert b["empty_rate"] > 0.0 and b["error_rate"] > 0.0
    assert set(b["reject"]) <= {"broken_json", "missing_field", "unknown_action"}
    assert b["reject"].get("unknown_action", 0) >= 1
    assert a["len_p50"] > 0 and a["len_p95"] >= a["len_p50"]
    assert a["lat_n"] == a["n"] and a["lat_p95"] >= a["lat_p50"] >= 0.0
    for purpose in ("deliberate", "plan", "reflect"):       # purpose 別内訳がある
        assert stats[f"A|{purpose}"]["n"] == 6
        assert stats[f"B|{purpose}"]["n"] == 6
    # 行動分布は type 別に埋まる(棄却も 1 カテゴリとして数える)
    assert "speak" in a["actions"]
    assert bt.REJECT_LABEL in b["actions"]
    assert bt.action_jsd(stats, "ALL") > 0.0


def test_report_and_outputs_have_numbers(tmp_path):
    scen, rows = _two_sided_rows()
    stats = bt.aggregate(rows)
    sides = {"A": {"model": "fake-8b", "endpoint": "http://a"},
             "B": {"model": "fake-14b", "endpoint": "http://b"}}
    meta = {"name": "t", "created": "2026-08-16T00:00:00+00:00",
            "bundle_sha256": bt.bundle_hash(scen), "workers": 2,
            "seed": 20260817, "format": "json"}
    report = bt.render_report(stats=stats, sides=sides,
                              purposes=["deliberate", "plan", "reflect"],
                              scenarios=scen, meta=meta,
                              walls={"A": 1.0, "B": 2.0})
    for needle in ("parse_action 成功率", "purpose 別内訳", "classify_reject",
                   "JSD", "判定材料", "限界", "deliberate", "reflect",
                   "fake-8b", "fake-14b"):
        assert needle in report, needle
    assert "%" in report and "|" in report
    assert "nan" not in report.lower()

    samples = bt.render_samples(scen, rows, per_purpose=2)
    assert samples.count("### prompt") == 6                 # 3 purpose × 2 件
    assert "### A —" in samples and "### B —" in samples
    assert scen[0]["prompt"] in samples                     # プロンプト全文が載る

    out = tmp_path / "results.parquet"
    bt.write_parquet(out, rows)
    table = pq.read_table(out)
    assert table.num_rows == len(rows)
    assert set(table.column_names) >= {
        "side", "model", "scenario_id", "purpose", "agent_id", "step",
        "latency_s", "parse_ok", "json_ok", "action_type", "reject_reason",
        "resp_chars", "response"}
    assert set(table.column("side").to_pylist()) == {"A", "B"}


def test_report_survives_single_side():
    scen = _bundle(per_purpose=4)
    rows, _ = bt.issue(scen, FakeBackend("vllm/fake-8b", _GOOD), side="A",
                       model="fake-8b", endpoint="http://a")
    report = bt.render_report(stats=bt.aggregate(rows),
                              sides={"A": {"model": "fake-8b",
                                           "endpoint": "http://a"}},
                              purposes=["deliberate", "plan", "reflect"],
                              scenarios=scen,
                              meta={"name": "solo", "created": "",
                                    "bundle_sha256": "0" * 64, "workers": 1,
                                    "seed": 1, "format": "json"},
                              walls={"A": 1.0})
    assert "B 側が無い" in report


# --------------------------------------------------------------------------- #
# 5. 束の生成が実シムの決定論に乗っていること(mock LLM・ネットワーク不使用)
# --------------------------------------------------------------------------- #
def test_build_bundle_from_mock_sim_is_reproducible(tmp_path):
    kwargs = dict(seed=4242, n_agents=10, n_steps=144, personas=None,
                  profile=None, env=None, sets=[], purposes=("deliberate",),
                  per_purpose=5)
    first, meta = bt.build_bundle(tmp_path / "src1", **kwargs)
    second, _ = bt.build_bundle(tmp_path / "src2", **kwargs)
    assert first, "mock シムから deliberate プロンプトが 1 件も採れていない"
    assert bt.bundle_hash(first) == bt.bundle_hash(second)
    assert [s["prompt"] for s in first] == [s["prompt"] for s in second]
    assert meta["dt_min"] == 10
    # 実プロンプトビルダーの出力そのもの(ヘッダの JSON 規約が入っている)
    assert '"action": "speak"' in first[0]["prompt"]
    assert first[0]["prompt_chars"] == len(first[0]["prompt"])
    # ジャーナルが残っている = 束の来歴を後から辿れる
    assert list((tmp_path / "src1").glob("llm_journal*.jsonl.gz"))


def test_main_build_only_writes_bundle(tmp_path):
    code = bt.main(["--name", "unit", "--out-dir", str(tmp_path), "--build-only",
                    "--n-agents", "10", "--n-steps", "144", "--personas", "",
                    "--per-purpose", "3", "--purposes", "deliberate,reflect",
                    "--scenario-seed", "77"])
    assert code == 0
    out = tmp_path / "tournament_unit"
    scen = bt.read_scenarios(out / "scenarios.jsonl")
    assert {s["purpose"] for s in scen} <= {"deliberate", "reflect"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_only"] is True
    assert manifest["bundle_sha256"] == bt.bundle_hash(scen)
    # 束だけのときは実 LLM 経路へ入らない(report は出さない)
    assert not (out / "report.md").exists()


def test_main_requires_endpoint_when_issuing(tmp_path):
    (tmp_path / "tournament_x").mkdir(parents=True)
    bt.write_scenarios(tmp_path / "tournament_x" / "scenarios.jsonl", _bundle(2))
    assert bt.main(["--name", "x", "--out-dir", str(tmp_path)]) == 2
