"""健康・疲労・病気・メンタル(現実ギャップ 後続波 H1 2026-07-07)= 疲労ゲージ/病気/引きこもり のテスト。

方針(既存の鉄則を継承):
- OFF(既定): illness/medical_visit/health_update が 0 件・新 stream "health" も引かない・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- 疲労 ON: 活動(勤務/移動)で fatigue が上がり睡眠で回復(単体・決定論)。高疲労で発火 effective_threshold
  が上がる(休息へ寄る=行動が鈍る)。
- 病気 ON: 確率で illness(onset)が出て欠勤(in_work_window=False)・受診(medical_visit + spend)、数日で recover。
- メンタル ON: 慢性高 grievance の持続で withdrawn(引きこもり)= 自由時間の行き先が自宅へ寄る(単体)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 病気(在宅)・疲労(休息)は物理位置を変え対面 co-location が変化しうる(career G5 / crowd G4 と同型)。
  よって呼数不変は compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import health
from society.cognition import drive, routine
from society.config import load_config
from society.engine.simulation import Simulation

_HEALTH = {"health.enabled": "true"}
# 各機構を効かせた ON 設定(決定論/k 不変性/スモーク用)。
_ON = {**_HEALTH,
       "health.fatigue_gain_work": "0.02", "health.fatigue_gain_move": "0.015",
       "health.fatigue_threshold_gain": "0.1", "health.onset_prob": "0.1",
       "health.medical_prob": "0.5", "health.mental_grievance_threshold": "0.3",
       "health.mental_withdraw_days": "1"}


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
    """明示 OFF と純粋既定が L1 完全一致(144 step)。H1 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"health.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H1 seam が no-op でない)"
    for k in ("illness", "medical_visit", "health_update"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 疲労ゲージ
def test_fatigue_accumulates_with_activity_and_recovers_in_sleep(tmp_path):
    """疲労 ON: 勤務/移動で fatigue が上がり、睡眠で回復(単体・決定論・RNG不要)。"""
    sim = _sim(tmp_path, "fatigue", steps=1,
               **{**_HEALTH, "health.fatigue_gain_work": "0.1",
                  "health.fatigue_gain_move": "0.05", "health.fatigue_recovery": "0.2"})
    a = sim.agents[0]
    a.loc, a.building, a.route, a.sleeping = "street", None, [], False
    a.fatigue = 0.0
    a.activity = "working"                             # 勤務で疲労が上がる
    for _ in range(3):
        health.tick_fatigue(a, sim.healthcfg, 0, 0, sim.logger)
    worked = a.fatigue
    assert worked > 0.0, "勤務で疲労が上がっていない"
    a.activity, a.route = "", ["x"]                    # 移動でも上がる
    health.tick_fatigue(a, sim.healthcfg, 0, 0, sim.logger)
    assert a.fatigue > worked, "移動で疲労が上がっていない"
    peak = a.fatigue
    a.sleeping, a.activity, a.route = True, "", []      # 睡眠で回復
    health.tick_fatigue(a, sim.healthcfg, 0, 0, sim.logger)
    assert a.fatigue < peak, "睡眠で疲労が回復していない"


def test_high_fatigue_dulls_behavior(tmp_path):
    """高疲労で発火 effective_threshold が上がる(休息へ寄る=行動が鈍る)。gain=0 で恒等。"""
    sim = _sim(tmp_path, "thr", steps=1,
               **{**_HEALTH, "health.fatigue_threshold_gain": "0.1",
                  "health.fatigue_high": "0.6"})
    a = sim.agents[0]
    a.drive_threshold = 0.55
    a.fatigue = 0.0
    assert health.fatigue_threshold_delta(a, sim.healthcfg) == 0.0, "低疲労で delta が 0 でない"
    a.fatigue = 0.9
    d = health.fatigue_threshold_delta(a, sim.healthcfg)
    assert d > 0.0, "高疲労なのに閾値 delta が上がらない"
    base = drive.effective_threshold(a, 0.0)
    tired = drive.effective_threshold(a, d)
    assert tired > base, "高疲労で発火閾値が上がっていない(行動が鈍らない)"


def test_fatigue_threshold_delta_zero_when_gain_zero(tmp_path):
    """fatigue_threshold_gain=0(既定)なら高疲労でも delta 0=恒等(バイト一致)。"""
    sim = _sim(tmp_path, "gain0", steps=1, **_HEALTH)   # gain 既定 0
    a = sim.agents[0]
    a.fatigue = 0.99
    assert health.fatigue_threshold_delta(a, sim.healthcfg) == 0.0


# --------------------------------------------------------------------- 病気
def test_illness_onset_absence_and_medical(tmp_path):
    """病気 ON: 確率で illness(onset)→ 欠勤(in_work_window=False)・受診(medical_visit + spend)。"""
    sim = _sim(tmp_path, "ill", steps=1,
               **{**_HEALTH, "health.onset_prob": "1.0", "health.medical_prob": "1.0",
                  "health.illness_days": "2", "health.medical_cost": "1500"})
    sim.run()
    onset = [e for e in _kind(sim, "illness") if e.payload["state"] == "onset"]
    assert onset, "発症 ON なのに illness(onset)が出ていない(非来街者が居ない?)"
    assert _kind(sim, "medical_visit"), "受診(medical_visit)が出ていない"
    assert [e for e in _kind(sim, "spend") if e.payload["cat"] == "medical"], \
        "受診の spend(cat=medical)が出ていない"
    assert all(not sim.agent_by_id[e.agent_id].visitor for e in onset), \
        "来街者が発症している(非来街者のみが対象のはず)"
    for e in onset:
        a = sim.agent_by_id[e.agent_id]
        assert a.sick and a.sick_until > 0, "発症者の sick フラグ/sick_until が立っていない"
        a.work_start_min, a.work_end_min = 8 * 60, 17 * 60
        assert not routine.in_work_window(a, 10 * 60), "病気なのに勤務時間帯扱い(欠勤していない)"
        assert not routine.in_part_time_window(a, 10 * 60), "病気なのにバイト時間帯扱い(欠勤していない)"


def test_illness_recovers_after_days(tmp_path):
    """病気 ON: illness_days 経過後に recover(sick 解除)。"""
    sim = _sim(tmp_path, "rec", steps=300,
               **{**_HEALTH, "health.onset_prob": "1.0", "health.medical_prob": "0.0",
                  "health.illness_days": "1"})
    sim.run()
    assert [e for e in _kind(sim, "illness") if e.payload["state"] == "onset"], "発症なし"
    assert [e for e in _kind(sim, "illness") if e.payload["state"] == "recover"], "回復なし"
    for a in sim.agents:
        if not a.visitor:
            assert not a.sick, "回復後も sick が残っている"


def test_sick_agent_heads_home(tmp_path):
    """病気の agent は routine が在宅(自宅へ寄せる)を返す(欠勤=在宅)。"""
    sim = _sim(tmp_path, "sickhome", steps=1, **_HEALTH)
    a = next(x for x in sim.agents if not x.visitor and x.home_node)
    a.sick, a.sick_until = True, 10_000
    a.building, a.route, a.sleeping = None, [], False
    a.loc = "street"
    a.node = next(d for d in sim.dests if d != a.home_node)  # 自宅から離れた場所
    a.bedtime_min = 1320                                     # 昼(bedtime 帯外)
    action = routine.decide(a, 48, sim, "路上",
                            sim.hub.stream("decide", a.id, 48), has_company=False)
    assert action["type"] == "move_to" and action["dest"] == a.home_node, \
        f"病気の agent が帰宅していない: {action}"


# --------------------------------------------------------------------- メンタル
def test_mental_withdrawal_on_chronic_grievance(tmp_path):
    """メンタル ON: 慢性高 grievance の持続で withdrawn(引きこもり)→ 自由時間に自宅へ寄る。"""
    sim = _sim(tmp_path, "mental", steps=1,
               **{**_HEALTH, "health.mental_grievance_threshold": "0.5",
                  "health.mental_withdraw_days": "2"})
    a = sim.agents[0]
    a.states["grievance"] = 0.9
    a.withdrawn, a.chronic_days = False, 0
    health.update_mental(a, sim.healthcfg, 0, 0, sim.logger)      # 1日目
    assert not a.withdrawn, "1日目で引きこもりになっている(閾値日数より早い)"
    health.update_mental(a, sim.healthcfg, 144, 144, sim.logger)  # 2日目=引きこもり
    assert a.withdrawn, "慢性高 grievance が続いても引きこもりにならない"
    hu = [e for e in _kind(sim, "health_update") if e.payload["name"] == "mental"]
    assert hu and hu[-1].payload["cause"] == "withdraw", "mental の health_update が出ていない"

    # 引きこもり: 自由時間の行き先が自宅へ寄る(社会的活動の低下)
    a.work_start_min, a.part_time, a.day_plan = -1, None, []
    a.building, a.route, a.sleeping, a.sick = None, [], False, False
    a.loc, a.stay_until, a.exit_intent = "street", -1, False
    a.bedtime_min = 1320
    a.node = next(d for d in sim.dests if d != a.home_node)
    action = routine.decide(a, 48, sim, "路上",
                            sim.hub.stream("decide", a.id, 48), has_company=False)
    assert action["type"] == "move_to" and action["dest"] == a.home_node, \
        f"引きこもりなのに自宅へ寄っていない: {action}"

    # grievance が下がれば解除
    a.states["grievance"] = 0.1
    health.update_mental(a, sim.healthcfg, 288, 288, sim.logger)
    assert not a.withdrawn, "grievance が下がっても引きこもりが解除されない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """健康 ON(疲労+病気+メンタル)同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
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
    """健康 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_health_call_count_k_invariant(tmp_path):
    """病気(在宅)・疲労(休息)は対面 co-location を変え ON!=OFF になりうる(実在する体調の必然)。だが
    R1 の本旨=「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。health の機構は
    暦日付+物理位置+config+新 stream "health"+grievance(k 非依存の state=イベント由来で writeback に無関係)
    のみ参照し、発火判断に k・内面状態(構成概念)を食わせないため、k=free と k=off で generate 呼数が完全一致する
    (career G5 / crowd G4 と同型)。"""
    free = _run_k(tmp_path, "hk_free", writeback="free")
    off = _run_k(tmp_path, "hk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"health の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "illness"), "発症日なのに illness が出ていない(機構が不発)"
