"""初日コールドスタート改善ノブ(run.natural_start)のテスト。

背景(統合mockリハーサル1万体で発見・devlog Entry 35): 全エージェントは既定 sleeping=False で
step0 に覚醒状態で開始するため、開始時刻が深夜でも起きたまま過ごし、初日の朝の起床(wake_up)と
朝計画(day_plan)がほぼ発火しない(2日目以降は正常)。

仕組み: run.natural_start=true で、run 開始時の着席時に「開始時刻が本人の自然な睡眠時間帯に
入っている居住者」を就寝状態・自宅・自然な起床 step(bedtime_min / sleep_steps からの決定論導出)で
初期化する。起床は既存の _phase_wake_and_returns の wake_up→_schedule_plan にそのまま乗る
(新しい起床経路を作らない=朝計画が初日から全員分出る)。

方針(既存 R1 の鉄則を継承):
- OFF(既定): 純粋既定と L1 バイト一致(golden_baseline_l1.json は tests/test_scenario.py が別途担保)。
  誰も就寝状態で着席させない・stream を一度も引かない(乱数消費ゼロ=決定論導出のみ)。
- ON(mock): (a)開始時刻が睡眠区間に入る居住者は step0 で自宅・sleeping=True(判定は決定論導出と一致)。
  (b)初日の朝(morning)に wake_up と day_plan が全住民分出る。(c)夜勤者/来街者/通勤者は対象外。
  (d)同 seed 2 回で L1 完全一致。(e)予算飽和下で ON の LLM 呼数が k∈{free,off} 非依存。(f)pool 経路の
  day0 着席でも初日朝計画が出る。
検証は mock のみ(実LLM 禁止)。開始時刻は Clock を midnight(00:00)へ差し替えて「深夜開始」を再現する
(既定の 07:00 では大半の居住者が既に起床済み=睡眠区間外のため、深夜開始が本ノブの主対象)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society.config import load_config                          # noqa: E402
from society.engine import simulation as sim_mod                # noqa: E402
from society.engine.simulation import Simulation, _natural_wake_step  # noqa: E402
from society.world.clock import Clock as _Clock, STEP_MINUTES   # noqa: E402


# ------------------------------------------------------------ ヘルパ
def _midnight(monkeypatch):
    """Clock を 00:00 開始へ差し替える(深夜開始=本ノブの主対象を再現)。構築時の着席で使われる。
    Simulation は run.start_tod を分へ解釈して Clock(start_min=...) を呼ぶため、渡される引数を
    無視して常に 00:00 を返すスタブにする(深夜開始の強制は据え置き=このガードの意図は不変)。"""
    monkeypatch.setattr(sim_mod, "Clock", lambda *a, **k: _Clock(start_hour=0))


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


# ============================================================ (unit) 導出関数
def test_wake_step_midnight_in_window():
    """深夜(00:00)開始で 22:00 就寝・42step(7h)睡眠 → 残り 5h=30step で起床(切り上げ)。"""
    # bedtime 22:00=1320 / dur=420 分 / start 00:00=0 → since=120 分 / remain=300 分 → 30 step
    assert _natural_wake_step(0, 1320, 42) == 30


def test_wake_step_daytime_start_already_awake():
    """日中(07:00)開始では 22:00 就寝の居住者は既に起床済み(区間外=None=覚醒で着席)。"""
    # start 07:00=420 / since=(420-1320)%1440=540 >= dur(420) → None
    assert _natural_wake_step(420, 1320, 42) is None


def test_wake_step_daytime_late_sleeper_still_asleep():
    """日中開始でも遅寝(23:50 就寝・49step)の居住者はまだ睡眠中=区間内で自然に働く。"""
    # start 07:00=420 / bed 23:50=1430 / dur=490 / since=(420-1430)%1440=430 < 490
    #   remain=60 分 → 6 step(08:00 起床)
    assert _natural_wake_step(420, 1430, 49) == 6


def test_wake_step_night_shift_daytime_sleep():
    """夜勤者(日中側 08:00 就寝)は昼に睡眠中 → 同じ式で睡眠着席になる(夜勤も対象=覚醒開始ではない)。"""
    # bed 08:00=480 / dur=420 / start 10:00=600 → since=120 < 420 → remain=300 → 30 step
    assert _natural_wake_step(600, 480, 42) == 30


def test_wake_step_boundary_and_positive():
    """ちょうど起床時刻(区間の端=exclusive)は None。区間内なら必ず >=1 step を返す。"""
    assert _natural_wake_step(300, 1320, 42) is None       # 1320+420=1740%1440=300 = 起床時刻
    assert _natural_wake_step(1739 % 1440, 1320, 42) == 1  # 起床 1 分前 → 1 step 切り上げ
    assert _natural_wake_step(0, 0, 0) is None              # dur=0 は対象外


# ============================================================ OFF: 純粋既定と一致・無着席
def test_off_no_agent_asleep_at_start(tmp_path):
    """OFF(既定)は誰も就寝状態で着席させない(構築直後に全員 sleeping=False)。"""
    sim = _sim(tmp_path, "off_seat", steps=1)
    assert all(not a.sleeping for a in sim.agents), "OFF なのに就寝状態で着席している"


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(96step)。ノブが no-op であることを固定する。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "nat_off", **{"run.natural_start": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(natural_start seam が no-op でない)"


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致(no-op の確度を上げる)。"""
    pure = _sim(tmp_path, "purel", n=12, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=12, steps=144, **{"run.natural_start": "false"})
    off.run()
    assert _l1(pure) == _l1(off)


# ============================================================ ON: step0 着席が導出と一致
def test_on_seats_match_derivation(tmp_path, monkeypatch):
    """深夜開始 ON: 各居住者の sleeping は導出関数(_natural_wake_step)と厳密に一致し、
    就寝者は自宅(home_node/home_building)・sleep_until=導出 step で着席する。少なくとも1体は就寝。"""
    _midnight(monkeypatch)
    sim = _sim(tmp_path, "seat", n=40, steps=1, **{"run.natural_start": "true"})
    start_tod = sim.clock.sim_min(0) % 1440
    assert start_tod == 0, "monkeypatch で 00:00 開始になっていない"
    n_asleep = 0
    for a in _residents(sim):
        want = _natural_wake_step(start_tod, a.bedtime_min, a.sleep_steps)
        assert a.sleeping == (want is not None), \
            f"着席就寝が導出と不一致: id={a.id} sleeping={a.sleeping} want={want}"
        if a.sleeping:
            n_asleep += 1
            assert a.sleep_until == want                    # 起床 step が導出と一致
            assert a.node == a.home_node                    # 自宅に居る
            if a.home_building and sim.city.has_building(a.home_building):
                assert a.building == a.home_building        # 実在の住宅建物内
                assert a.floor == a.home_floor
    assert n_asleep > 0, "深夜開始 ON なのに就寝状態で着席した居住者が1体もいない"


def test_on_excludes_visitor_and_commuter(tmp_path, monkeypatch):
    """来街者・流入通勤者(自宅が圏外)は対象外=就寝着席しない(居住者は着席する)。"""
    _midnight(monkeypatch)
    sim = _sim(tmp_path, "excl", n=8, steps=1, **{"run.natural_start": "true"})
    # 全員を覚醒へ戻し、睡眠区間に入る就寝時刻を与えたうえで属性だけ差し替えて再着席させる。
    for a in sim.agents:
        a.sleeping = False
        a.building = None
        a.visitor = False
        a.commute = False
        a.bedtime_min = 1320                                # 22:00(深夜開始=睡眠区間内)
        a.sleep_steps = 42
    res, vis, com = sim.agents[0], sim.agents[1], sim.agents[2]
    vis.visitor = True
    com.commute = True
    sim._apply_natural_start()
    assert res.sleeping, "居住者が就寝着席していない"
    assert not vis.sleeping, "来街者が就寝着席してしまった(対象外のはず)"
    assert not com.sleeping, "流入通勤者が就寝着席してしまった(対象外のはず)"


# ============================================================ ON: 初日の朝に全住民の wake/plan
def test_on_morning_wake_and_plan_for_all_residents(tmp_path, monkeypatch):
    """深夜開始 ON の初日(96step)で、着席就寝した住民は全員が朝の時間帯に wake_up と day_plan を出す。
    さらに圏内に留まった大多数の住民に初日朝計画が出る(圏外へ出た個体=exit_prob は自然に除かれる)。"""
    _midnight(monkeypatch)
    sim = _sim(tmp_path, "morning", n=20, steps=96, **{"run.natural_start": "true"})
    res = _residents(sim)
    res_ids = {a.id for a in res}
    seated0 = {a.id for a in res if a.sleeping}          # 着席就寝=本ノブの直接対象(睡眠中は圏外へ出られない)
    assert seated0, "深夜開始 ON なのに就寝着席した居住者が0(前提の再確認が必要)"
    sim.run()
    wake = [e for e in sim.logger.events if e.kind == "wake_up" and e.agent_id in res_ids]
    plan = [e for e in sim.logger.events if e.kind == "day_plan" and e.agent_id in res_ids]
    woke_ids = {e.agent_id for e in wake}
    planned_ids = {e.agent_id for e in plan}
    # 着席就寝した住民は必ず初日の朝に起床し、朝計画が出る(本ノブが保証する初日の朝の系列)。
    assert seated0 <= woke_ids, f"着席就寝した住民が初日に起床していない: {sorted(seated0 - woke_ids)}"
    assert seated0 <= planned_ids, f"着席就寝した住民の初日朝計画が出ていない: {sorted(seated0 - planned_ids)}"
    # 圏内に留まった大多数の住民に初日朝計画が出る(コールドスタート改善の効果)。
    assert len(planned_ids) >= 0.8 * len(res_ids), \
        f"初日朝計画の住民カバレッジが低すぎる: {len(planned_ids)}/{len(res_ids)}"
    # 起床はすべて朝(正午前)に起きる(深夜開始→自然な睡眠→翌朝の起床)。
    for e in wake:
        assert 3 <= _hour(sim, e.step) < 12, \
            f"起床が朝の時間帯でない: id={e.agent_id} step={e.step} hour={_hour(sim, e.step)}"


def test_off_daytime_start_no_morning_plan(tmp_path):
    """対照: 既定の 07:00 開始 + OFF では、22:00 就寝の住民は初日日中に朝計画が出ない(=改善対象の症状)。
    ノブ ON の効果(深夜開始で朝計画が出る)を裏づける最小の症状再現。"""
    sim = _sim(tmp_path, "sym", n=20, steps=96)             # 07:00 開始(既定)・OFF
    sim.run()
    # 07:00 開始では day0(07:00〜24:00 相当)の朝に住民が就寝→起床する経路が無い。
    day0_plans = [e for e in sim.logger.events
                  if e.kind == "day_plan" and 0 <= e.step < 96 and e.agent_id >= 0]
    # 少なくとも「全住民ぶんは出ていない」(コールドスタートの症状)ことを確認する。
    assert len(day0_plans) < len(_residents(sim)), \
        "07:00 開始 OFF で全住民ぶんの初日朝計画が出てしまっている(症状が再現しない=前提の再確認が必要)"


# ============================================================ ON: 決定論・OFF との差・k 非依存
def test_on_deterministic_same_seed(tmp_path, monkeypatch):
    """同 seed 2 回で L1 完全一致(natural_start は乱数を一切引かない=決定論導出)。"""
    _midnight(monkeypatch)
    a = _sim(tmp_path, "det_a", n=20, steps=48, **{"run.natural_start": "true"})
    a.run()
    b = _sim(tmp_path, "det_b", n=20, steps=48, **{"run.natural_start": "true"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(専用外の乱数漏れ?)"


def test_on_differs_from_off(tmp_path, monkeypatch):
    """深夜開始で ON は OFF と L1 が異なる(=着席就寝が実際に初日の行動系列を変えている)。"""
    _midnight(monkeypatch)
    on = _sim(tmp_path, "diff_on", n=20, steps=48, **{"run.natural_start": "true"})
    on.run()
    off = _sim(tmp_path, "diff_off", n=20, steps=48, **{"run.natural_start": "false"})
    off.run()
    assert _l1(on) != _l1(off), "深夜開始で ON と OFF が同一(着席就寝が効いていない)"


def test_on_seating_is_k_invariant(tmp_path, monkeypatch):
    """着席就寝の判定は k/traits を読まない: k∈{free,off} で構築直後の着席状態(就寝/起床 step/位置)が
    完全一致(発火ゲートに k を入れない=既存機構に乗るだけ、を構築レベルで固定する)。"""
    _midnight(monkeypatch)

    def seats(k):
        s = _sim(tmp_path, f"seat_{k}", n=40, steps=1,
                 **{"run.natural_start": "true", "k.writeback": k})
        return [(a.id, a.sleeping, a.sleep_until, a.node, a.building) for a in s.agents]

    assert seats("free") == seats("off"), "着席就寝が k に依存している(発火ゲートに k/traits が混入)"


def test_on_call_count_k_invariant(tmp_path, monkeypatch):
    """真の R1「呼数 k 非依存」: ON のまま k(writeback)を free/off に振っても LLM 呼数が不変。
    natural_start は物理量(開始時刻・bedtime/sleep・自宅)のみで着席し発火ゲートに k/traits を入れない。
    controls.mode=compute_matched はエンジンの R1 呼数一致モード(内省も off で実行し結果破棄=呼数一致)。
    予算飽和下では駆動発火 = Σ min(申請者数, 予算) となり「誰が発火するか」が変わっても本数は不変。"""
    _midnight(monkeypatch)
    ov = {"run.natural_start": "true", "lod.max_llm_per_step": 2,
          "controls.mode": "compute_matched"}
    free = _sim(tmp_path, "k_free", n=60, steps=48, **{**ov, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", n=60, steps=48, **{**ov, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls) > 0, \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ============================================================ pool 経路(小プール)
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない。test_pool_rotation と同型)。"""
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("npool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_sim(tmp_path, name, pool_dir, monkeypatch, steps, cap=80, **ov):
    _midnight(monkeypatch)
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_on_pool_day0_morning_plan(small_pool, tmp_path, monkeypatch):
    """pool 経路の day0 着席でも初日の朝計画が出る: 深夜開始 ON で pool 居住者が就寝着席し、
    初日の朝に wake_up と day_plan が発火する(2日目以降の途中日 enter には掛からない=day0 のみ)。"""
    sim = _pool_sim(tmp_path, "pool_on", small_pool, monkeypatch, steps=96,
                    **{"run.natural_start": "true"})
    # 構築直後: 睡眠区間に入る pool 居住者が就寝状態で着席している(1体以上)。
    seated = [a for a in sim.agents if a.sleeping]
    assert seated, "pool day0 着席で就寝状態の居住者が1体もいない(深夜開始 ON なのに)"
    for a in seated:
        assert not a.visitor and not a.commute              # 対象は居住者のみ
        assert a.node == a.home_node
    sim.run()
    wake = [e for e in sim.logger.events if e.kind == "wake_up" and 0 <= e.step < 96]
    plan = [e for e in sim.logger.events if e.kind == "day_plan" and 0 <= e.step < 96]
    assert wake, "pool 経路の初日に wake_up が1件も出ていない"
    assert plan, "pool 経路の初日に day_plan が1件も出ていない"
    assert any(3 <= (sim.clock.sim_min(e.step) // 60) % 24 < 12 for e in wake), \
        "pool 経路の初日起床が朝の時間帯に出ていない"


def test_pool_off_no_seating(small_pool, tmp_path):
    """pool 経路でも OFF(既定)は誰も就寝状態で着席させない(day0 着席は従来どおり覚醒)。"""
    dot = ["run.seed=42", "run.n_agents=10", "run.n_steps=1", "run.name=pool_off",
           "model.backend=mock", "pool.enabled=true", f"pool.dir={small_pool}",
           "pool.present_cap=80"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "pool_off")
    assert all(not a.sleeping for a in sim.agents), "pool OFF なのに就寝状態で着席している"
