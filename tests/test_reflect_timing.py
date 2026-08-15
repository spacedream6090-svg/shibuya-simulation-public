"""RFX-A 内省の現実的タイミング / RFX-O 発火文脈の観測(第116バッチ 2026-08-15・既定 OFF)。

正典: docs/plans/reflection-leisure-plan.md §3。実装: src/society/cognition/reflect_timing.py。

本テストが機械固定すること:
- **既定 `mode: "sleep"` はバイト一致**(現行と完全同値。golden を守る)。
- **1 予約 = 1 発火**: 早期発火は `reflect_suppress_arm` で将来の予約 1 回を厳密に相殺する。
  二重発火が原理的に起こらないことを状態機械の単体で固定する。
- **発火点を足していない**: 予約を立てる場所は従来と同じ 3 箇所だけ(AST で固定)。
- **乱数ゼロ**: 発火判定が RngHub を 1 度も引かない(スパイで固定)+ ON 同士 2 回で L1 一致。
- **時刻分布が動く**: 居住者 100% 夜 → 夕方へ分散(実測)。
- **RFX-O**: OFF は payload にキーを生やさない / ON は語彙どおりのタグが載る。
- **no-fingerprint**: 日中用タスク文に機構語(一人・条件・トリガ…)が 1 語も無い。
- **JSON 契約キー不変**: どの変種でも summary/salient/belief(deep は +self/ties)のまま。
"""
from __future__ import annotations

import collections
import json
import re

import pytest

from society.cognition import reflect_timing, reflection
from society.config import load_config
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=60, steps=432, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "lod.budget.tiers.enabled=true"]      # DPH-B が RFX-A の前提(§3.5)
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _reflects(sim):
    return [e for e in sim.logger.events if e.kind == "reflect"]


# --------------------------------------------------------------------- 既定 OFF
def test_default_mode_matches_pure_default(tmp_path):
    """明示 sleep モードと純粋既定が L1 完全一致(144 step)= 現行と完全同値。"""
    pure = _sim(tmp_path, "rfx_pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "rfx_sleep", steps=144, **{"reflection.timing.mode": "sleep"})
    off.run()
    assert _l1(pure) == _l1(off), "sleep モードが既定と不一致(seam が no-op でない)"


def test_default_payload_has_no_new_keys(tmp_path):
    """RFX-O OFF では reflect payload に when / context が 1 つも生えない(L1 バイト一致)。"""
    sim = _sim(tmp_path, "rfx_nokey", steps=288)
    sim.run()
    ev = _reflects(sim)
    assert ev, "内省が 1 件も起きていない(テストが無風)"
    for e in ev:
        assert "when" not in e.payload and "context" not in e.payload


def test_default_task_text_is_byte_identical():
    """既定(moment/sleepy とも False)のタスク文が従来定数とバイト一致。"""
    t = reflection._reflect_task(deep=False, variety=False, agent_id=1, day=0)
    assert t == reflection._REFLECT_TASK
    d = reflection._reflect_task(deep=True, variety=False, agent_id=1, day=0)
    assert d == reflection._DEEP_REFLECT_TASK


# ------------------------------------------------- 1 予約 1 発火(状態機械の単体)
class _FakeSim:
    def __init__(self, mode):
        self._rfxcfg = reflect_timing.build_cfg({"mode": mode})


class _FakeAgent:
    def __init__(self):
        self.reflect_step = -1
        self.reflect_moment_day = -1
        self.reflect_suppress_arm = 0


def test_arm_is_identity_in_sleep_mode():
    """sleep モードの arm は従来と同じ `reflect_step = step + 1`(見送りが起きない)。"""
    sim, a = _FakeSim("sleep"), _FakeAgent()
    a.reflect_suppress_arm = 1                    # 立っていても sleep モードでは無視される
    reflect_timing.arm(sim, a, step=7)
    assert a.reflect_step == 8 and a.reflect_suppress_arm == 1


def test_early_fire_cancels_exactly_one_future_reservation():
    """★1 予約 1 発火: 早期発火 1 回が将来の予約を**ちょうど 1 回だけ**見送らせる。"""
    sim, a = _FakeSim("reflective_moment"), _FakeAgent()
    a.reflect_suppress_arm = 1                    # 早期発火した直後の状態
    reflect_timing.arm(sim, a, step=7)            # その夜の就寝
    assert a.reflect_step == -1, "見送られていない(二重発火になる)"
    assert a.reflect_suppress_arm == 0, "見送りフラグが消費されていない"
    reflect_timing.arm(sim, a, step=100)          # 翌日の就寝は通常どおり立つ
    assert a.reflect_step == 101


def test_arm_moments_sets_the_suppression(tmp_path):
    """arm_moments が発火させた個体は必ず (reflect_step=step, moment_day=-1, suppress=1)。"""
    sim = _sim(tmp_path, "rfx_state", n=30, steps=1,
               **{"reflection.timing.mode": "reflective_moment"})
    sim.run()
    fired = [a for a in sim.agents if int(getattr(a, "reflect_suppress_arm", 0))]
    for a in fired:
        assert reflect_timing.when_of(a) in reflect_timing.CONTEXTS
        assert a.reflect_moment_day == -1


# --------------------------------------------------------------------- 乱数ゼロ
def test_moment_decision_draws_no_randomness(tmp_path):
    """発火判定が RngHub を 1 度も引かない(hub.stream をスパイして固定)。"""
    sim = _sim(tmp_path, "rfx_norng", n=40, steps=1,
               **{"reflection.timing.mode": "reflective_moment"})
    calls = []
    orig = sim.hub.stream
    sim.hub.stream = lambda *a, **k: (calls.append(a), orig(*a, **k))[1]
    reflect_timing.arm_moments(sim, step=100, sim_min=100 * 10)
    assert not calls, f"発火判定が乱数 stream を引いた: {calls[:3]}"


def test_moment_mode_is_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(乱数を 1 本も足していない)。"""
    ov = {"reflection.timing.mode": "reflective_moment"}
    a = _sim(tmp_path, "rfx_det_a", steps=288, **ov)
    a.run()
    b = _sim(tmp_path, "rfx_det_b", steps=288, **ov)
    b.run()
    assert _l1(a) == _l1(b)


# --------------------------------------------------------------------- 効果
def test_hour_distribution_moves_out_of_the_night(tmp_path):
    """★H-R1: 22-06 時 100% だった内省が夕方へ分散する(本レーンの目的そのもの)。"""
    base = _sim(tmp_path, "rfx_hour_off", steps=432)
    base.run()
    on = _sim(tmp_path, "rfx_hour_on", steps=432,
              **{"reflection.timing.mode": "reflective_moment"})
    on.run()

    def night_share(sim):
        ev = _reflects(sim)
        assert ev
        n = sum(1 for e in ev if (e.sim_min % 1440) // 60 >= 22
                or (e.sim_min % 1440) // 60 < 6)
        return n / len(ev)

    off_share, on_share = night_share(base), night_share(on)
    assert off_share == 1.0, f"既定で夜 100% でない(前提が崩れた): {off_share}"
    assert on_share < 0.8, f"夜からの分散が起きていない: {on_share:.1%}"


def test_call_count_stays_within_run_boundary_effect(tmp_path):
    """★H-R2: 内省の件数は「1 予約 1 発火」を保つ。差はラン境界効果のみ(小さい)。

    ★正直な限界(計画書 §3.3): 夕方に早期発火した個体がその夜に眠らないままランが
    終わると、OFF では起きなかった内省が ON では起きる。**±1 日ぶん**の差が出うる。
    """
    base = _sim(tmp_path, "rfx_cnt_off", steps=432)
    base.run()
    on = _sim(tmp_path, "rfx_cnt_on", steps=432,
              **{"reflection.timing.mode": "reflective_moment"})
    on.run()
    n_off, n_on = len(_reflects(base)), len(_reflects(on))
    assert n_off > 0
    assert abs(n_on - n_off) / n_off < 0.15, \
        f"内省の件数がラン境界効果を超えて動いた: off={n_off} on={n_on}"


# --------------------------------------------------------------------- RFX-O
def test_context_tag_vocabulary(tmp_path):
    """RFX-O ON: reflect payload に when / context が載り、語彙が CONTEXTS に閉じる。"""
    sim = _sim(tmp_path, "rfx_tag", steps=432,
               **{"reflection.timing.mode": "reflective_moment",
                  "reflection.timing.context_tag": "true"})
    sim.run()
    ev = _reflects(sim)
    assert ev
    ctx = collections.Counter(e.payload["context"] for e in ev)
    assert set(ctx) <= set(reflect_timing.CONTEXTS), f"語彙外のタグ: {set(ctx)}"
    assert {e.payload["when"] for e in ev} <= {"moment", "sleep"}
    assert ctx.get("sleep", 0) < sum(ctx.values()), "早期発火が 1 件も無い"
    for e in ev:                                   # when と context の整合
        assert (e.payload["when"] == "sleep") == (e.payload["context"] == "sleep")


# --------------------------------------------------------------------- 窓
def test_window_from_min_uses_the_end_of_duty():
    """窓の始まり = max(work_end, part_time end, evening_floor)。夜勤は円環で解く。"""
    cfg = reflect_timing.build_cfg({"mode": "reflective_moment"})

    class _A:
        work_end_min = 17 * 60
        part_time = None
        work_wraps = False
    assert reflect_timing.window_from_min(_A(), cfg) == 18 * 60   # 早上がりでも夕方の下限

    class _B(_A):
        work_end_min = 21 * 60
    assert reflect_timing.window_from_min(_B(), cfg) == 21 * 60   # 遅番は本人の終業時刻

    class _C(_A):
        work_end_min = 6 * 60
        work_wraps = True
    assert reflect_timing.window_from_min(_C(), cfg) == 6 * 60    # 夜勤明けの朝が"帰路"

    class _D(_A):
        work_end_min = -1
        part_time = {"end_min": 22 * 60}
    assert reflect_timing.window_from_min(_D(), cfg) == 22 * 60


# --------------------------------------------------------------------- 契約
def test_no_new_firing_site():
    """★発火点を足していない: `reflect_step` を立てる場所は従来と同じ 3 箇所 + RFX-A の満期だけ。"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    sched = (root / "src/society/engine/scheduler.py").read_text(encoding="utf-8")
    # scheduler 側の直接代入は「DPH-B の繰り越し / 消費済みの -1」だけになっている。
    assigns = re.findall(r"agent\.reflect_step = ([^\n#]+)", sched)
    assert sorted(set(a.strip() for a in assigns)) == ["-1", "step + 1"], \
        f"scheduler に想定外の reflect_step 代入がある: {assigns}"
    assert sched.count("agent.reflect_step = step + 1") == 1, \
        "予約が reflect_timing.arm 以外の場所からも立っている"
    rt = (root / "src/society/cognition/reflect_timing.py").read_text(encoding="utf-8")
    stmts = [ln.split("#", 1)[0].strip() for ln in rt.splitlines()]
    stmts = [ln for ln in stmts if ln.startswith("agent.reflect_step =")]
    assert stmts == ["agent.reflect_step = step + 1",   # arm() 本体(予約)
                     "agent.reflect_step = step"], stmts  # 満期(arm_moments)


def test_daytime_task_has_no_mechanism_words():
    """no-fingerprint: 日中用タスク文に機構語が 1 語も無い / JSON 契約キーは不変。"""
    banned = ("一人", "条件", "トリガ", "発火", "予約", "内省的瞬間", "モード", "閾値")
    for t in (reflection._MOMENT_REFLECT_TASK, reflection._MOMENT_DEEP_REFLECT_TASK,
              reflection._SLEEPY_REFLECT_TASK):
        for w in banned:
            assert w not in t, f"機構語 {w} がタスク文に漏れた"
        for key in ("summary", "salient", "belief"):
            assert f'"{key}"' in t
    for key in ("self", "ties"):
        assert f'"{key}"' in reflection._MOMENT_DEEP_REFLECT_TASK
    # 日中用は「眠りにつく前に」と言わない(文言が嘘にならない)
    assert "眠りにつく前" not in reflection._MOMENT_REFLECT_TASK
    assert "眠りにつく前" not in reflection._MOMENT_DEEP_REFLECT_TASK


def test_sleep_task_rewrite_drops_the_insomniac_pattern():
    """就寝前テンプレの言い換え(Lemyre 2020): 能動的な再生と将来評価の文言が消える。"""
    old = reflection._REFLECT_TASK
    new = reflection._SLEEPY_REFLECT_TASK
    assert "順に思い出し" in old and "順に思い出し" not in new
    assert "明日の自分にどう影響するか" in old
    assert "明日の自分にどう影響するか" not in new
    assert reflection._reflect_task(deep=False, variety=False, agent_id=1,
                                    day=0, sleepy=True) == new
    # deep は言い換えの対象外(自己像の生成は別の仕事)
    assert reflection._reflect_task(deep=True, variety=False, agent_id=1,
                                    day=0, sleepy=True) == reflection._DEEP_REFLECT_TASK


def test_mode_is_validated():
    with pytest.raises(ValueError):
        reflect_timing.build_cfg({"mode": "whenever"})


def test_reflect_calls_ride_the_dph_b_life_lane(tmp_path):
    """★前提の確認(§3.5): 内省呼は DPH-B 二層予算の **life レーン**に入る。

    RFX-A は予算外の内省を「飽和帯」へ移すので、life レーンの先取り枠が無いと
    一般呼(social/reply)の granted 率を押し下げる。ここが life であることが前提条件。
    """
    from society.cognition.lod import PURPOSE_LANE
    assert PURPOSE_LANE["reflect"] == "life"
    sim = _sim(tmp_path, "rfx_lane", n=40, steps=288,
               **{"reflection.timing.mode": "reflective_moment"})
    assert sim.budget.tiers, "DPH-B が ON になっていない(テストの前提が崩れた)"
    assert sim.budget.caps["life"] > 0
    sim.run()
    assert _reflects(sim), "内省が 1 件も起きていない"
