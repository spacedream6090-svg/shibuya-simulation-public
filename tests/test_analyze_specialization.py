"""第78バッチ Part C1: scripts/analyze_specialization.py(語彙の専門化スコア)の検証。

設計哲学 = **オラクルが無いので差分でしか主張しない**。したがってここで確かめるのは
  (a) 3 つの下位スコアが決定論で、既知の合成入力に対して期待どおりの向きに動くこと
  (b) 判定閾値が **引数必須**(未指定ならエラーで止まる=事後に閾値をいじれない)
  (c) 対照(propagation_off)が無ければ **合否を出さない**(NO_CONTROL)
  (d) mock ラン 2 本(通常 / propagation_off)で end-to-end に完走し、
      §A 実装健全性 と §B 現象の由来 が**別セクション**で出ること

house style: scripts/ を path 追加して import(tests/test_analyze_founders.py に倣う)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import analyze_specialization as sp                  # noqa: E402

from society.config import load_config               # noqa: E402
from society.engine.simulation import Simulation     # noqa: E402


class _Args:
    ngram = 2
    topics = 20
    max_df = 0.5
    ttr_budget = 20
    min_cluster = 2
    min_role_tokens = 10
    min_count = 2
    min_users = 2
    bin_steps = 144


def _rec(step, agent, text, n=2):
    return {"step": step, "agent": agent, "tokens": sp.tokens_of(text, n)}


# --------------------------------------------------------------------------- #
# (a) トークン化と下位スコアの単体
# --------------------------------------------------------------------------- #
def test_tokens_of_is_deterministic_and_drops_symbols():
    assert sp.tokens_of("あいう", 2) == ["あい", "いう"]
    assert sp.tokens_of("あ、い。う", 2) == ["あい", "いう"]     # 記号は落ちる
    assert sp.tokens_of("あ", 2) == ["あ"]                       # n 未満はそのまま
    assert sp.tokens_of("", 2) == []
    assert sp.tokens_of("あいう", 2) == sp.tokens_of("あいう", 2)


def test_jsd_bounds_and_symmetry():
    a, b = Counter({"x": 10}), Counter({"y": 10})
    assert sp._jsd(a, b) == 1.0                       # 完全に異なる分布 = 1
    assert sp._jsd(a, Counter({"x": 3})) == 0.0       # 同一分布 = 0
    c = Counter({"x": 3, "y": 7})
    assert abs(sp._jsd(a, c) - sp._jsd(c, a)) < 1e-12  # 対称


def test_topic_narrowing_detects_narrowing():
    """後半で語彙が狭まる合成データ → narrowing が正になる。"""
    recs = []
    for i in range(12):                               # 前半: 毎回違う語
        recs.append(_rec(i, i % 3, f"あか{chr(0x30a2 + i)}{chr(0x30f2 + i)}こんにち"))
    for i in range(12):                               # 後半: 同じ語の反復
        recs.append(_rec(100 + i, i % 3, "あかいこんにち"))
    out = sp.topic_narrowing(recs, 120, topics=5, max_df=1.0, budget=20,
                             min_cluster=2)
    assert out["value"] is not None and out["value"] > 0


def test_role_differentiation_separates_roles():
    recs = ([_rec(i, 1, "あいうあいうあいう") for i in range(10)]
            + [_rec(i, 2, "かきくかきくかきく") for i in range(10)])
    roles = {1: "医師", 2: "学生"}
    out = sp.role_differentiation(recs, roles, min_tokens=5)
    assert out["n_roles"] == 2 and out["value"] > 0.9   # 語が全く違う → 1 に近い
    same = sp.role_differentiation(
        [_rec(i, 1, "あいうあいう") for i in range(6)]
        + [_rec(i, 2, "あいうあいう") for i in range(6)], roles, min_tokens=5)
    assert same["value"] == 0.0


def test_role_differentiation_needs_two_roles():
    out = sp.role_differentiation([_rec(0, 1, "あいう")], {1: "医師"}, min_tokens=1)
    assert out["value"] is None and "2 群" in out["reason"]
    assert sp.role_differentiation([_rec(0, 1, "あいう")], {},
                                   min_tokens=1)["value"] is None


def test_vocab_persistence_prefers_long_lived_tokens():
    """毎日出る語 = 1.0 / 初日だけの語 = 低い。"""
    every = [_rec(d * 144 + 1, d % 2, "ずっとずっと") for d in range(5)]
    out = sp.vocab_persistence(every, 4 * 144 + 1, min_count=2, min_users=2,
                               bin_steps=144)
    assert out["value"] == 1.0                        # 初出日ビン〜末尾まで全ビンに出る
    burst = [_rec(1, 0, "いちどいちど"), _rec(2, 1, "いちどいちど")]
    out2 = sp.vocab_persistence(burst, 5 * 144, min_count=2, min_users=2,
                                bin_steps=144)
    assert out2["value"] is not None and out2["value"] < 0.5


def test_missing_data_reports_reason_not_zero():
    """算出不能を 0 と偽らない(None + 理由)。"""
    for out in (sp.topic_narrowing([], 10, topics=3, max_df=0.5, budget=10,
                                   min_cluster=2),
                sp.vocab_persistence([], 10, min_count=2, min_users=2,
                                     bin_steps=144)):
        assert out["value"] is None and out["reason"]


# --------------------------------------------------------------------------- #
# (b)(c)(d) end-to-end(mock ラン 2 本)
# --------------------------------------------------------------------------- #
def _mock_run(tmp_path, name: str, **ov):
    dot = ["run.seed=42", "run.n_agents=20", "run.n_steps=144",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    out = tmp_path / name
    Simulation(load_config(dot), out_dir=out).run()
    return out


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "analyze_specialization.py"),
         *[str(a) for a in args]],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(_ROOT))


def test_thresholds_are_required(tmp_path):
    """事前登録の流儀: 判定閾値を省くと**エラーで止まる**(既定値を埋め込まない)。"""
    run = _mock_run(tmp_path, "sp_req", **{"run.n_steps": 24, "run.n_agents": 8})
    p = _cli(run)
    assert p.returncode != 0
    assert "--min-delta" in (p.stderr + p.stdout)
    p2 = _cli(run, "--min-delta", "0.02")
    assert p2.returncode != 0 and "--alpha" in (p2.stderr + p2.stdout)


def test_no_control_gives_no_verdict(tmp_path):
    run = _mock_run(tmp_path, "sp_nc")
    p = _cli(run, "--min-delta", "0.02", "--alpha", "0.05",
             "--out", tmp_path / "sp_nc_out")
    assert p.returncode == 0, p.stderr
    res = json.loads((tmp_path / "sp_nc_out" / "specialization.json")
                     .read_text(encoding="utf-8"))
    assert res["control"] is None and res["comparison"] is None
    md = (tmp_path / "sp_nc_out" / "specialization_report.md").read_text(
        encoding="utf-8")
    assert "NO_CONTROL" in md and "判定不能" in md


def test_end_to_end_with_propagation_off_control(tmp_path):
    treat = _mock_run(tmp_path, "sp_treat")
    ctrl = _mock_run(tmp_path, "sp_ctrl", **{"ablate.propagation_off": "true"})
    out = tmp_path / "sp_out"
    p = _cli(treat, "--control", ctrl, "--min-delta", "0.02", "--alpha", "0.05",
             "--out", out)
    assert p.returncode == 0, p.stderr
    res = json.loads((out / "specialization.json").read_text(encoding="utf-8"))
    # 3 つの下位スコアが両条件で出る
    for cond in (res["treatment"], res["control"]):
        assert cond["n_runs"] == 1
        assert set(cond["means"]) == {"topic_narrowing", "role_differentiation",
                                      "vocab_persistence"}
    cmp_ = res["comparison"]
    assert cmp_["verdict"] in ("INTERACTION_DRIVEN", "NOT_INTERACTION_DRIVEN",
                              "UNDERPOWERED", "INCONCLUSIVE")
    # 条件 1 本ずつでは検定しない(正直に skip と書く)
    assert cmp_["perm"]["method"] == "skipped_n<2"
    assert cmp_["perm"]["p"] is None
    # 対照の manifest に ablate.propagation_off が残っている(来歴の確定)
    assert res["control"]["runs"][0]["manifest"]["ablate"]["propagation_off"] is True
    # 閾値がそのまま出力に載る(事前登録との突き合わせ用)
    assert res["params"]["min_delta"] == 0.02 and res["params"]["alpha"] == 0.05


def test_report_separates_soundness_and_phenomenon(tmp_path):
    """Part G: 実装健全性の検証と現象の由来の検証を**別セクション**で出す。"""
    treat = _mock_run(tmp_path, "sp_sec_t", **{"observer.state_hash.enabled": "true"})
    ctrl = _mock_run(tmp_path, "sp_sec_c", **{"ablate.propagation_off": "true"})
    out = tmp_path / "sp_sec_out"
    p = _cli(treat, "--control", ctrl, "--min-delta", "0.02", "--alpha", "0.05",
             "--out", out)
    assert p.returncode == 0, p.stderr
    md = (out / "specialization_report.md").read_text(encoding="utf-8")
    a = md.index("## §A 実装健全性の検証")
    b = md.index("## §B 現象の由来の検証")
    assert a < b, "2 節が分かれていない"
    # §A に来歴(指標ハッシュ・等級・呼数)、§B に差分が出る
    head = md[a:b]
    assert "指標定義ハッシュ" in head and "LLM 呼数" in head
    assert "再現性等級" in head
    assert "状態ハッシュチェーン" in head and "整合" in head
    assert "差分" in md[b:]
    res = json.loads((out / "specialization.json").read_text(encoding="utf-8"))
    snd = res["soundness"]
    assert snd["metrics_spec_consistent"] is True     # ラン直後なので必ず一致する
    assert snd["undeclared_toggles"] == []
    assert snd["tier_mismatch"] is False
    assert snd["state_hash"][treat.name]["ok"] is True
