"""シミュ開始時刻の設定可能化(run.start_tod)のテスト。

背景: Clock は開始 07:00 ハードコードだった。run.start_tod="HH:MM"(既定 "07:00"=現行値)を分 of day
へ解釈して Clock に渡す。day 境界・時刻表示・夜内省/朝計画・natural_start は全て clock 由来の sim_min
から決定論導出するため、開始時刻はこの 1 点の配線で全経路へ波及する。00:00 開始 + natural_start ON で
「全住民が自然な睡眠から初日の朝を迎える」=コールドスタート完全解を実証する(mock のみ・小規模)。

方針(既存 R1 の鉄則を継承):
- 既定 "07:00": 純粋既定と L1 バイト一致(golden は test_scenario が別途担保)。乱数消費ゼロ。
- "00:00": step0 の sim_min=0・時刻表示が正しい・day 境界が深夜0時(day=step//144)で正しく回る。
- 時刻機構(transit ダイヤ・bedtime・営業時間)は sim_min%1440 基準=開始時刻を変えても不変。
- 00:00 + natural_start ON: 初日朝に住民ほぼ全員の wake_up + day_plan。同 seed 決定論・呼数 k 非依存。
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society.config import load_config                          # noqa: E402
from society.engine.simulation import (                         # noqa: E402
    Simulation, _parse_start_tod, _natural_wake_step)
from society.world.clock import STEP_MINUTES                    # noqa: E402


# ------------------------------------------------------------ ヘルパ
def _sim(tmp_path, name, n=15, steps=96, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _residents(sim):
    return [a for a in sim.agents if not a.visitor and not a.commute]


def _hour(sim, step):
    return (sim.clock.sim_min(step) // 60) % 24


# ============================================================ (unit) パーサ
def test_parse_start_tod_variants():
    """"HH:MM" 文字列/分 int/60 進丸め値/None のどれでも分 of day(0..1439)へ落ちる。"""
    assert _parse_start_tod("07:00") == 420        # 既定=現行値
    assert _parse_start_tod("00:00") == 0
    assert _parse_start_tod("23:50") == 1430
    assert _parse_start_tod("09:30") == 570
    assert _parse_start_tod(None) == 420           # 欠落は既定 07:00
    assert _parse_start_tod(420) == 420            # 既に分
    assert _parse_start_tod(1430) == 1430          # OmegaConf の 60 進丸め(23:50→1430)も分として一致
    assert _parse_start_tod(0) == 0


# ============================================================ 既定 = バイト一致
def test_default_equals_explicit_0700(tmp_path):
    """明示 "07:00" と純粋既定(未指定)が L1 完全一致(既定値がノブの no-op)。"""
    d = _sim(tmp_path, "def")
    d.run()
    e = _sim(tmp_path, "exp0700", **{"run.start_tod": "07:00"})
    e.run()
    assert _l1(d) == _l1(e), "既定と明示 07:00 が L1 不一致(start_tod 既定が no-op でない)"


def test_default_clock_is_0700(tmp_path):
    """既定の clock は 07:00 起点(start_min=420・step0 の sim_min=420)。"""
    s = _sim(tmp_path, "clk", steps=1)
    assert s.clock.start_min == 420
    assert s.clock.sim_min(0) == 420
    assert s.clock.hour(0) == 7


# ============================================================ 00:00 開始: 時刻表示・sim_min
def test_midnight_sim_min_and_clock_display(tmp_path):
    """"00:00" 指定で step0 の sim_min=0・時刻表示(hour/sim_min)が正しい。"""
    s = _sim(tmp_path, "mid", steps=1, **{"run.start_tod": "00:00"})
    assert s.clock.start_min == 0
    assert s.clock.sim_min(0) == 0
    assert s.clock.hour(0) == 0
    assert s.clock.sim_min(6) == 60 and s.clock.hour(6) == 1     # 01:00
    assert s.clock.sim_min(90) == 900 and s.clock.hour(90) == 15  # 15:00
    assert s.clock.sim_min(143) == 1430 and s.clock.hour(143) == 23  # 23:50


def test_arbitrary_start_tod_offsets_clock(tmp_path):
    """任意の開始時刻(09:30)でも start_min と sim_min が一致(1 step=STEP_MINUTES 分)。"""
    s = _sim(tmp_path, "at0930", steps=1, **{"run.start_tod": "09:30"})
    assert s.clock.start_min == 570
    assert s.clock.sim_min(0) == 570
    assert s.clock.sim_min(3) == 570 + 3 * STEP_MINUTES


# ============================================================ day 境界
def test_midnight_day_boundary_rolls(tmp_path):
    """00:00 開始: day 境界=深夜0時=step の 144 の倍数(day=step//144)。2 日ぶんの境界を確認。"""
    s = _sim(tmp_path, "dayb0", steps=1, **{"run.start_tod": "00:00"})
    assert s.clock.day(0) == 0
    assert s.clock.day(143) == 0
    assert s.clock.day(144) == 1                    # 深夜0時で day1
    assert s.clock.day(287) == 1
    assert s.clock.day(288) == 2                    # 翌深夜0時で day2


def test_0700_day_boundary_unchanged(tmp_path):
    """07:00 開始(既定): day0=07:00〜24:00 の短い初日・境界は step102(sim_min=1440)=従来どおり。"""
    s = _sim(tmp_path, "dayb7", steps=1)
    assert s.clock.day(0) == 0
    assert s.clock.day(101) == 0
    assert s.clock.day(102) == 1                    # sim_min=420+1020=1440
    assert s.clock.day(245) == 1
    assert s.clock.day(246) == 2


def test_midnight_two_day_run_boundary(tmp_path):
    """00:00 開始で 2 日(288step)を実走し、イベントが day0/day1 の両日に跨る(日境界が正しく回る)。"""
    s = _sim(tmp_path, "run2d", n=12, steps=288, **{"run.start_tod": "00:00"})
    s.run()
    days = {s.clock.day(e.step) for e in s.logger.events if e.agent_id >= 0}
    assert 0 in days and 1 in days, f"2 日ランで day0/day1 の両日にイベントが無い: {sorted(days)}"
    # 深夜0時の日替わりで各 step の day が step//144 と厳密一致(絶対分基準の日境界)。
    for e in s.logger.events:
        assert s.clock.day(e.step) == e.step // 144


# ============================================================ 時刻機構(sim_min%1440 基準)は不変
def test_time_of_day_mechanisms_hold_at_midnight(tmp_path):
    """transit ダイヤ・bedtime 等は時刻(sim_min%1440)基準=開始時刻に依らず同じ時刻で同じ判定。"""
    mid = _sim(tmp_path, "tmid", steps=1, **{"run.start_tod": "00:00"})
    day = _sim(tmp_path, "tday", steps=1)                       # 07:00 開始(既定)
    # 同一の実時刻(正午=720 / 深夜3時=180)における transit 運行判定が開始時刻に依らず一致する。
    for tod in (180, 720, 1000, 1380):
        assert mid.transit.has_service(tod) == day.transit.has_service(tod)
    # bedtime は個体属性(時刻)=Clock 起点に非依存(名簿/生成が同一なら同一)。
    assert [a.bedtime_min for a in mid.agents] == [a.bedtime_min for a in day.agents]


# ============================================================ コールドスタート完全解
def test_midnight_natural_start_first_day_morning(tmp_path):
    """00:00 開始 + natural_start ON: 着席就寝した住民が全員初日の朝に wake_up+day_plan を出し、
    圏内に留まった大多数の住民に初日朝計画が出る(config→clock→natural_start の端から端までの配線)。"""
    sim = _sim(tmp_path, "cold", n=20, steps=96,
               **{"run.start_tod": "00:00", "run.natural_start": "true"})
    assert sim.clock.sim_min(0) % 1440 == 0, "00:00 開始になっていない(配線切れ)"
    res = _residents(sim)
    res_ids = {a.id for a in res}
    seated0 = {a.id for a in res if a.sleeping}     # 着席就寝=本ノブの直接対象
    assert seated0, "00:00 開始 natural_start ON なのに就寝着席した住民が0(前提の再確認が必要)"
    # 着席就寝の判定は導出関数と厳密一致(乱数を一切引かない決定論導出)。
    for a in res:
        want = _natural_wake_step(0, a.bedtime_min, a.sleep_steps)
        assert a.sleeping == (want is not None)
    sim.run()
    wake = [e for e in sim.logger.events if e.kind == "wake_up" and e.agent_id in res_ids]
    plan = [e for e in sim.logger.events if e.kind == "day_plan" and e.agent_id in res_ids]
    woke_ids = {e.agent_id for e in wake}
    planned_ids = {e.agent_id for e in plan}
    assert seated0 <= woke_ids, f"着席就寝した住民が初日に起床していない: {sorted(seated0 - woke_ids)}"
    assert seated0 <= planned_ids, f"着席就寝した住民の初日朝計画が出ていない: {sorted(seated0 - planned_ids)}"
    assert len(planned_ids) >= 0.8 * len(res_ids), \
        f"初日朝計画の住民カバレッジが低すぎる: {len(planned_ids)}/{len(res_ids)}"
    for e in wake:                                   # 起床は朝(正午前)= 深夜開始→自然な睡眠→翌朝
        assert 3 <= _hour(sim, e.step) < 12, \
            f"起床が朝の時間帯でない: id={e.agent_id} step={e.step} hour={_hour(sim, e.step)}"


def test_midnight_natural_start_beats_0700_off(tmp_path):
    """対照: 00:00+ON の初日朝計画カバレッジは、既定 07:00+OFF(コールドスタートの症状)を上回る。"""
    on = _sim(tmp_path, "cs_on", n=20, steps=96,
              **{"run.start_tod": "00:00", "run.natural_start": "true"})
    on.run()
    off = _sim(tmp_path, "cs_off", n=20, steps=96)              # 既定 07:00・natural_start OFF
    off.run()

    def day0_planned(sim):
        return {e.agent_id for e in sim.logger.events
                if e.kind == "day_plan" and 0 <= e.step < 96 and e.agent_id >= 0}

    assert len(day0_planned(on)) > len(day0_planned(off)), \
        "00:00+ON が 07:00+OFF より初日朝計画を増やしていない(コールドスタート改善が効いていない)"


# ============================================================ 決定論・k 非依存
def test_midnight_deterministic_same_seed(tmp_path):
    """同 seed 2 回で L1 完全一致(start_tod・natural_start とも乱数を一切引かない)。"""
    a = _sim(tmp_path, "det_a", n=20, steps=48,
             **{"run.start_tod": "00:00", "run.natural_start": "true"})
    a.run()
    b = _sim(tmp_path, "det_b", n=20, steps=48,
             **{"run.start_tod": "00:00", "run.natural_start": "true"})
    b.run()
    assert _l1(a) == _l1(b), "00:00 開始の決定論が崩れている(専用外の乱数漏れ?)"


def test_midnight_call_count_k_invariant(tmp_path):
    """00:00 開始でも LLM 呼数が k(writeback)非依存(発火ゲートに k/traits を入れない)。
    controls.mode=compute_matched は R1 呼数一致モード・予算飽和下で本数は「誰が発火するか」に不変。"""
    ov = {"run.start_tod": "00:00", "run.natural_start": "true",
          "lod.max_llm_per_step": 2, "controls.mode": "compute_matched"}
    free = _sim(tmp_path, "k_free", n=60, steps=48, **{**ov, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", n=60, steps=48, **{**ov, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls) > 0, \
        f"00:00 開始で呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"
