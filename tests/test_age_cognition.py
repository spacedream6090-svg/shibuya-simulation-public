"""AGE-C: 年齢 → 思考/推論の量(第116バッチ 2026-08-15・既定 OFF)。

正典: docs/plans/age-diversity-plan.md §4-4。実装: src/society/cognition/age_cog.py +
src/society/factors/registry.age_cognition_params。

本テストが機械固定すること:
- **既定 OFF はバイト一致**(明示 OFF・純粋既定・未指定の 3 者で L1 完全一致)。
- **乱数ゼロ**: ON にしても draw を 1 粒も引かない(ON 同士 2 回で L1 完全一致 +
  写像関数が Generator を要求しない純関数であること)。
- **越えてはいけない線**: 予算(`LodBudget`)も発火順(`sort(key=(-drive,id))`)も
  高解像度層の選抜(`mind.tiers.high.select`)も年齢を読まない。
- **cap 拘束下で呼数ゼロ増**: 要求者が cap を超えるランで ON/OFF の LLM 呼数が完全一致。
- **文献の向き**: 高齢ほど周期が長い / 19 歳付近で閾値が最も低い(感覚探求のピーク)/
  ref_age で恒等。**単一の年齢係数ではない**(周期と閾値が別々の曲線)。
"""
from __future__ import annotations

import json

import pytest

from society.cognition import age_cog
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors.registry import age_cognition_params


def _sim(tmp_path, name, n=24, steps=48, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _n_calls(sim):
    return len(sim.logger.llm_calls)


# --------------------------------------------------------------- 純関数(写像)
def test_mapping_is_identity_at_ref_age():
    """ref_age では period_mult=1.0 / theta_delta=0.0(= 現行較正どおり)。"""
    cfg = age_cog.build_cfg({"enabled": True})
    p = age_cognition_params(int(cfg["ref_age"]), cfg)
    assert p["period_mult"] == pytest.approx(1.0, abs=1e-6)
    assert p["theta_delta"] == pytest.approx(0.0, abs=1e-6)


def test_period_grows_monotonically_with_age():
    """熟慮時間の年齢曲線(Queen 2012: 若→高で +63%)= 周期は年齢に単調増加。"""
    cfg = age_cog.build_cfg({"enabled": True})
    mults = [age_cognition_params(a, cfg)["period_mult"]
             for a in (15, 20, 30, 45, 60, 70, 82)]
    assert mults == sorted(mults), f"周期倍率が年齢に単調でない: {mults}"
    assert mults[0] < 1.0 < mults[-1], "ref_age の両側へ振れていない(対称化が壊れている)"
    # 若(20)→高(70)の比が文献の +63% に一致する(規格化しても比は不変)。
    lo = age_cognition_params(20, cfg)["period_mult"]
    hi = age_cognition_params(70, cfg)["period_mult"]
    assert hi / lo == pytest.approx(1.64, abs=0.01)


def test_threshold_curve_is_not_the_period_curve():
    """★単一の年齢係数ではない: 閾値は 19 歳で最小(感覚探求のピーク)= 周期と別の形。"""
    cfg = age_cog.build_cfg({"enabled": True})
    d = {a: age_cognition_params(a, cfg)["theta_delta"] for a in range(10, 83)}
    argmin = min(d, key=lambda a: d[a])
    assert argmin == 19, f"閾値 delta の最小が 19 歳でない(Steinberg 2018 の形が崩れた): {argmin}"
    assert d[19] < d[10] and d[19] < d[30], "10代後半の谷が出ていない"
    assert d[82] > 0.0 > d[19], "高齢側が上向いていない(処理速度低下の項が消えた)"
    # 周期は単調増加・閾値は非単調 = 2 本の別曲線であることの機械的な証拠。
    per = [age_cognition_params(a, cfg)["period_mult"] for a in range(10, 83)]
    thr = [d[a] for a in range(10, 83)]
    assert per == sorted(per) and thr != sorted(thr)


def test_mapping_draws_no_randomness():
    """写像は純関数(rng 引数を取らない)= 呼び出し位置に関係なく draw 順不変。"""
    import inspect
    sig = inspect.signature(age_cognition_params)
    assert list(sig.parameters) == ["age", "cfg"], \
        "age_cognition_params が乱数を要求している(零 draw 型が壊れた)"
    cfg = age_cog.build_cfg({"enabled": True})
    assert age_cognition_params(70, cfg) == age_cognition_params(70, cfg)


def test_gain_zero_is_identity():
    """gain=0 は年齢に依らず恒等(較正レバーが 0 で無風であること)。"""
    cfg = age_cog.build_cfg({"enabled": True, "period_gain": 0.0, "theta_gain": 0.0})
    for a in (6, 19, 38, 82):
        p = age_cognition_params(a, cfg)
        assert p["period_mult"] == pytest.approx(1.0)
        assert p["theta_delta"] == pytest.approx(0.0)


# --------------------------------------------------------------- OFF = 恒等
def test_off_hooks_are_identity(tmp_path):
    """OFF ではフックが厳密に恒等(倍率 1.0・加算 0.0)= 掛け算/足し算が無風。"""
    sim = _sim(tmp_path, "agec_off_hook", steps=1)
    a = sim.agents[0]
    a.age = 80
    assert age_cog.period_mult(sim, a) == 1.0
    assert age_cog.threshold_delta(sim, a) == 0.0
    assert getattr(a, "_age_cog", None) is None, "OFF なのにプロファイルを組んでいる"


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)= golden を守る。"""
    pure = _sim(tmp_path, "agec_pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "agec_expl_off", steps=144, **{"cognition.age.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(AGE-C の seam が no-op でない)"


# --------------------------------------------------------------- ON
def test_on_is_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(乱数を 1 本も足していない)。"""
    a = _sim(tmp_path, "agec_on_a", steps=144, **{"cognition.age.enabled": "true"})
    a.run()
    b = _sim(tmp_path, "agec_on_b", steps=144, **{"cognition.age.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


def test_on_changes_effective_threshold_distribution(tmp_path):
    """ON で年齢由来の閾値 delta が実際に散る(= drive 分布が動く)。OFF では全員 0。"""
    on = _sim(tmp_path, "agec_thr_on", n=40, steps=1,
              **{"cognition.age.enabled": "true"})
    deltas = sorted({round(age_cog.threshold_delta(on, a), 6) for a in on.agents})
    assert len(deltas) > 1, "ON なのに閾値 delta が 1 値しかない(年齢が読めていない)"
    assert min(deltas) < 0.0 or max(deltas) > 0.0
    off = _sim(tmp_path, "agec_thr_off", n=40, steps=1)
    assert {age_cog.threshold_delta(off, a) for a in off.agents} == {0.0}


def test_period_min_uses_age_multiplier(tmp_path):
    """`_period_min` に個体倍率が掛かる(周期の年齢差が実体になる)。cv=0 で決定論比較。"""
    from society.cognition import fire as fire_mod
    sim = _sim(tmp_path, "agec_period", n=6, steps=1,
               **{"cognition.age.enabled": "true",
                  "cognition.fire.enabled": "true",
                  "cognition.fire.period_cv_scale": "0"})
    young, old = sim.agents[0], sim.agents[1]
    young.age, old.age = 20, 78
    young._age_cog = old._age_cog = None
    p_young = fire_mod._period_min(sim, young, "walking", 0)
    p_old = fire_mod._period_min(sim, old, "walking", 0)
    assert p_old > p_young, f"高齢の基本周期が伸びていない: {p_young} vs {p_old}"


# --------------------------------------------------------------- R1(呼数)
def test_call_count_unchanged_under_binding_cap(tmp_path):
    """cap が binding するラン(要求者 >> cap)で ON/OFF の LLM 呼数が**完全一致**。

    ★これが AGE-C の R1 の本体: 年齢は「量の配分」でなく「順位の分布」に効く。
      総数は cap 固定のまま、*誰が* 撃つか(purpose の内訳)だけが動く。
    n=120 / cap=2 は「要求者が毎 step cap を超える」= 予算が深く binding する構成
    (age-diversity-plan §4-2 が本選構成について述べているのと同じ状況)。
    """
    ov = {"lod.max_llm_per_step": "2"}
    off = _sim(tmp_path, "agec_cap_off", n=120, steps=72, **ov)
    off.run()
    on = _sim(tmp_path, "agec_cap_on", n=120, steps=72,
              **{**ov, "cognition.age.enabled": "true"})
    on.run()
    assert _n_calls(on) == _n_calls(off), \
        f"cap 拘束下で呼数が動いた: off={_n_calls(off)} on={_n_calls(on)}"
    # ★同時に「内訳は動いている」ことも固定する(= 無風ではなく配分が変わった証拠)。
    import collections
    mix_off = collections.Counter(c.get("purpose") for c in off.logger.llm_calls)
    mix_on = collections.Counter(c.get("purpose") for c in on.logger.llm_calls)
    assert mix_off != mix_on, "総数も内訳も同じ = AGE-C が 1 ミリも効いていない"


def test_budget_and_tier_selection_never_read_age():
    """★越えてはいけない線: 予算・発火順・高解像度層の選抜に年齢が 1 文字も無い。"""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    word_age = re.compile(r"\bage\b")
    for rel in ("src/society/cognition/lod.py", "src/society/mind.py"):
        src = (root / rel).read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            assert not word_age.search(code), \
                f"{rel}:{i} に年齢の読み取りが入った(割当の年齢化): {line}"
    # 発火順(scheduler の requesters.sort)も年齢を見ない。
    sched = (root / "src/society/engine/scheduler.py").read_text(encoding="utf-8")
    m = re.search(r"requesters\.sort\((.*?)\)\n", sched, re.S)
    assert m and "age" not in m.group(1), "発火順のキーに年齢が入った"
