"""都市・環境インフラのショック(現実ギャップ 後続波 H4 2026-07-07)= 災害/交通遅延・運休/インフラ障害 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): disaster/transit_delay/infra_outage が 0 件・新 stream "disaster" も引かない・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
  既存の摂動シナリオ(shock_closure/shock_event。test_scenario)の挙動も一切変えない(独立 phase)。
- 災害 ON: 発生日に disaster(onset)が出て 外出抑制(在宅)・強い grievance・交通麻痺(電車運休=
  has_service 停止)、解除で clear(単体・決定論)。
- 交通遅延 ON: transit_delay が出て grievance(単体)。運休=has_service 停止。
- インフラ ON: infra_outage(onset)が出て grievance+在宅娯楽/スマホ抑制(単体)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 災害(在宅)・運休は物理位置・移動を変え対面 co-location が変化しうる(career G5 / crowd G4 /
  健康 H1 / 世帯 H2 / 商業 H3 と同型)。よって呼数不変は compute_matched 下の k 不変性(k=free と k=off の
  generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import disaster
from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_DIS = {"disaster.enabled": "true"}
# 全機構を効かせた ON 設定(決定論/k 不変性/スモーク用)。days=[0]+duration=2 で run 全体が災害中。
_ON = {**_DIS,
       "disaster.days": "[0]", "disaster.duration_days": "2",
       "disaster.grievance": "0.06", "disaster.stay_home_bias": "0.7",
       "disaster.suspend_transit": "true",
       "disaster.delay_prob": "0.2", "disaster.suspend_prob": "0.3",
       "disaster.delay_grievance": "0.02",
       "disaster.outage_prob": "0.1", "disaster.outage_grievance": "0.03"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。H4 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"disaster.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H4 seam が no-op でない)"
    for k in ("disaster", "transit_delay", "infra_outage"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    assert pure.transit.suspended is False, "OFF で電車が運休している(交通麻痺 seam が漏れている)"


def test_off_noop_under_shock_scenario(tmp_path):
    """既存の摂動シナリオ(shock_closure)ON の世界でも disaster OFF は完全 no-op(独立 phase)。"""
    base = _sim(tmp_path, "scbase", steps=48, **{
        "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 6, duration_steps: 6, center: [0,0], radius_m: 120}"})
    base.run()
    off = _sim(tmp_path, "scoff", steps=48, **{
        "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 6, duration_steps: 6, center: [0,0], radius_m: 120}",
        "disaster.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "shock_closure ON で disaster seam が no-op でない"


# --------------------------------------------------------------------- 災害ショック
def test_disaster_onset_paralyzes_and_grievance(tmp_path):
    """災害 ON: 発生日に disaster(onset)+交通麻痺(運休=has_service 停止)+在宅抑制+grievance。"""
    sim = _sim(tmp_path, "onset", steps=1, **{
        **_DIS, "disaster.days": "[0]", "disaster.duration_days": "2",
        "disaster.grievance": "0.06", "disaster.stay_home_bias": "1.0",
        "disaster.suspend_transit": "true"})
    sm = sim.clock.sim_min(0)
    scheduler._phase_disaster(sim, 0, sm)
    onset = [e for e in _kind(sim, "disaster") if e.payload["phase"] == "onset"]
    assert onset and onset[0].payload["kind"] == "台風", "発生日に disaster(onset)が出ていない"
    assert onset[0].agent_id == -1, "災害は世界イベント(agent_id=-1)のはず"
    # 交通麻痺: 電車運休 = has_service 停止(駅経由の退出/帰還が不可)
    assert sim.transit.suspended is True and not sim.transit.has_service(sm), \
        "災害中なのに電車が運休していない(交通麻痺していない)"
    # 在宅抑制: 住民(非来街者)が在宅フラグ + プロンプト文脈
    assert any(disaster.is_homebound(a) for a in sim.agents if not a.visitor), \
        "災害中なのに在宅(外出抑制)の住民が居ない"
    assert sim.today_disaster_line and "台風" in sim.today_disaster_line, \
        "災害のプロンプト文脈が設定されていない"
    # grievance: 街内の人に不満(factors 経由・cause=disaster)
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "disaster"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "災害→grievance(disaster)が factors 経由で入っていない"


def test_disaster_homebound_agent_heads_home(tmp_path):
    """在宅抑制の住民は routine が在宅(自宅へ寄せる)を返す(病気の在宅と同型)。"""
    sim = _sim(tmp_path, "home", steps=1, **_DIS)
    a = next(x for x in sim.agents if not x.visitor and x.home_node)
    a._disaster_homebound = True
    a.sick, a.building, a.route, a.sleeping = False, None, [], False
    a.loc = "street"
    a.node = next(d for d in sim.dests if d != a.home_node)   # 自宅から離れた場所
    a.bedtime_min = 1320                                       # 昼(bedtime 帯外)
    action = routine.decide(a, 48, sim, "路上",
                            sim.hub.stream("decide", a.id, 48), has_company=False)
    assert action["type"] == "move_to" and action["dest"] == a.home_node, \
        f"在宅抑制の agent が帰宅していない: {action}"


def test_disaster_clears_after_duration(tmp_path):
    """災害 ON: duration_days 経過後に disaster(clear)= 運休解除・在宅解除・文脈解除。"""
    sim = _sim(tmp_path, "clear", steps=1, **{
        **_DIS, "disaster.days": "[0]", "disaster.duration_days": "1",
        "disaster.stay_home_bias": "1.0"})
    scheduler._phase_disaster(sim, 0, sim.clock.sim_min(0))       # day0: onset
    assert sim._disaster_active and sim.transit.suspended
    scheduler._phase_disaster(sim, 144, sim.clock.sim_min(144))   # day1: clear
    clear = [e for e in _kind(sim, "disaster") if e.payload["phase"] == "clear"]
    assert clear, "duration 経過後に disaster(clear)が出ていない"
    assert not sim._disaster_active and sim.transit.suspended is False, \
        "解除後も災害中/運休が残っている"
    assert sim.today_disaster_line is None, "解除後もプロンプト文脈が残っている"
    assert not any(disaster.is_homebound(a) for a in sim.agents), \
        "解除後も在宅フラグが残っている"


# --------------------------------------------------------------------- 交通の遅延・運休
def test_transit_delay_grievance_and_suspend(tmp_path):
    """交通遅延 ON(運休): transit_delay(運休)+ grievance + has_service 停止。"""
    sim = _sim(tmp_path, "delay", steps=1, **{
        **_DIS, "disaster.delay_prob": "1.0", "disaster.suspend_prob": "1.0",
        "disaster.delay_grievance": "0.05"})
    sm = sim.clock.sim_min(0)
    scheduler._phase_disaster(sim, 0, sm)
    td = _kind(sim, "transit_delay")
    assert td and td[0].payload["kind"] == "運休" and td[0].agent_id == -1, \
        "交通遅延 ON(運休)なのに transit_delay(運休)が出ていない"
    assert sim.transit.suspended is True and not sim.transit.has_service(sm), \
        "運休なのに電車が止まっていない"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "transit_delay"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "遅延→grievance(transit_delay)が factors 経由で入っていない"


def test_transit_delay_only_keeps_service(tmp_path):
    """交通遅延 ON(遅延のみ・suspend_prob=0): transit_delay(遅延)は出るが運休はしない。"""
    sim = _sim(tmp_path, "donly", steps=1, **{
        **_DIS, "disaster.delay_prob": "1.0", "disaster.suspend_prob": "0.0",
        "disaster.delay_grievance": "0.05"})
    scheduler._phase_disaster(sim, 0, sim.clock.sim_min(0))
    td = _kind(sim, "transit_delay")
    assert td and td[0].payload["kind"] == "遅延", "遅延のみ ON なのに遅延が出ていない"
    assert sim.transit.suspended is False, "遅延のみ(運休なし)なのに電車が止まっている"


# --------------------------------------------------------------------- インフラ障害
def test_infra_outage_grievance_and_suppresses_phone(tmp_path):
    """インフラ ON: infra_outage(onset)+ grievance + スマホ(_phone)抑制(通信断)。"""
    sim = _sim(tmp_path, "infra", steps=1, **{
        **_DIS, "disaster.outage_prob": "1.0", "disaster.outage_grievance": "0.04"})
    scheduler._phase_disaster(sim, 0, sim.clock.sim_min(0))
    io = [e for e in _kind(sim, "infra_outage") if e.payload["phase"] == "onset"]
    assert io and io[0].payload["kind"] == "停電" and io[0].agent_id == -1, \
        "インフラ ON なのに infra_outage(onset)が出ていない"
    assert disaster.infra_out(sim), "障害発生中フラグ(infra_out)が立っていない"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "infra_outage"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "障害→grievance(infra_outage)が factors 経由で入っていない"
    # 通信断: スマホ行動(_phone)は抑制されて None(生活麻痺)
    a = next(x for x in sim.agents if not x.sleeping and x.loc != "outside")
    assert scheduler._phone(sim, a, 0, sim.clock.sim_min(0)) is None, \
        "インフラ障害中でもスマホ行動が抑制されていない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """H4 全機構 ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 k 不変性
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    """H4 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_disaster_call_count_k_invariant(tmp_path):
    """災害(在宅)・運休は対面 co-location を変え ON!=OFF になりうる(実在するショックの必然)。だが
    R1 の本旨=「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。disaster の機構は
    day-index+物理位置+config+新 stream "disaster" のみ参照し、発火判断に k・内面状態(構成概念)を食わせないため、
    k=free と k=off で generate 呼数が完全一致する(career G5 / crowd G4 / 健康 H1 と同型)。"""
    free = _run_k(tmp_path, "dk_free", writeback="free")
    off = _run_k(tmp_path, "dk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"disaster の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "disaster"), "災害日なのに disaster が出ていない(機構が不発)"
