"""朝の一日計画(ユーザー要望 2026-07-06)のテスト。

1. 起床(来街者は帰還)の直後 step に、全員へ 1回だけ day_plan が出る(1人1日1回)。
2. planning.enabled=false は新イベント(day_plan/ride)を一切出さず決定論(旧挙動へ復帰)。
   ★ 追加の手動ゴールデン: 3機能すべて OFF で HEAD とイベント列 byte 一致を確認済み。
3. 計画項目(place/what)が routine の行き先(move_to)に反映される。
4. day_plan の総数は k∈{free,off} で同一(R1: 呼数 k 非依存)。
"""
from __future__ import annotations

import json

from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 10, steps: int = 1, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _run_stream(tmp_path, name: str, ov: list[str]):
    # 168 step = 翌日 11:00 まで(全員が一度は起床する=day_plan が現れる長さ)
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=168", f"run.name={name}",
           *ov]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
def test_wake_schedules_plan_for_every_agent(tmp_path):
    """起床させると全員に plan_step が予約され、次 step で全員に day_plan が1件出る。"""
    sim = _sim(tmp_path, "wakeplan", n=8)
    step = 5
    sim_min = sim.clock.sim_min(step)
    for a in sim.agents:
        a.sleeping, a.sleep_until = True, step        # この step で起床する条件
        a.loc, a.building = "street", None
        a.node = a.home_node
        a.x, a.y = sim.city.node_xy(a.node)
        a.plan_day, a.plan_step, a.day_plan = -1, -1, []

    scheduler._phase_wake_and_returns(sim, step, sim_min)
    wakes = [e for e in sim.logger.events if e.kind == "wake_up"]
    assert len(wakes) == len(sim.agents)
    assert all(a.plan_step == step + 1 for a in sim.agents)

    sim_min2 = sim.clock.sim_min(step + 1)
    scheduler._phase_planning(sim, step + 1, sim_min2)
    plans = [e for e in sim.logger.events if e.kind == "day_plan"]
    assert len(plans) == len(sim.agents)
    assert {e.agent_id for e in plans} == {a.id for a in sim.agents}

    # 二重生成しない(plan_step は消費済み=-1)。1人1日1回。
    scheduler._phase_planning(sim, step + 1, sim_min2)
    assert len([e for e in sim.logger.events if e.kind == "day_plan"]) \
        == len(sim.agents)


def test_plan_at_most_once_per_agent_per_day(tmp_path):
    """通しの mock ランで、各 (agent, day) の day_plan は高々1件。"""
    sim = _sim(tmp_path, "onceperday", n=12, steps=288,
               **{"observer.snapshot_every": 144})
    sim.run()
    per: dict = {}
    for e in sim.logger.events:
        if e.kind == "day_plan":
            key = (e.agent_id, e.sim_min // 1440)
            per[key] = per.get(key, 0) + 1
    assert per, "day_plan が一件も出ていない"
    assert all(v == 1 for v in per.values()), \
        f"(agent, day) で複数の day_plan が出た: {[k for k, v in per.items() if v > 1]}"


def test_planning_off_no_new_events_and_deterministic(tmp_path):
    """planning/taxi/bus をすべて OFF = 新イベントゼロ+決定論(旧挙動へ復帰)。"""
    off = ["planning.enabled=false", "transit_ride.taxi.enabled=false",
           "transit_ride.bus.enabled=false"]
    a = _run_stream(tmp_path, "off_a", off)
    b = _run_stream(tmp_path, "off_b", off)
    assert a == b, "OFF 同士でイベント列が一致しない(非決定論)"
    kinds = {k for (_s, _i, k, _p) in a}
    assert "day_plan" not in kinds and "ride" not in kinds
    # トグルが効く: 既定(planning ON)では day_plan が現れる
    on = _run_stream(tmp_path, "on", [])
    assert any(k == "day_plan" for (_s, _i, k, _p) in on)


def test_plan_place_and_category_drive_routine_destination(tmp_path):
    """place 指定(POI 名部分一致)と what カテゴリの両方が move_to の行き先になる。"""
    sim = _sim(tmp_path, "planc", n=10,
               **{"transit_ride.taxi.enabled": "false"})   # taxi 無効で行き先を確定
    step = 48                                              # 15:00 = 午後

    def _free(a):
        a.loc, a.building, a.route, a.sleeping = "street", None, [], False
        a.work_start_min, a.part_time, a.stay_until = -1, None, 0
        a.bedtime_min, a.money = 1380, 1e9

    # (1) place 指定 → その名を含む POI のノードへ
    a0 = sim.agents[0]
    _free(a0)
    a0.node = sim.city.gateways[0]
    poi = next(p for p in sim.city.poi_list if p["node"] != a0.node)
    a0.day_plan = [{"when": "午後", "what": "visit", "place": poi["name"],
                    "done": False}]
    act = routine.decide(a0, step, sim, "x",
                         sim.hub.stream("decide", a0.id, step), has_company=False)
    cand = {q["node"] for q in sim.city.poi_list
            if poi["name"] in q["name"] and q["node"] != a0.node}
    assert act["type"] == "move_to" and act["dest"] in cand
    assert a0.day_plan[0]["done"] is True                 # 消化済みにマーク

    # (2) place 空 + what カテゴリ → そのカテゴリの POI ノードへ
    shops = {p["node"] for p in sim.city.pois_by_cat("shop")}
    if shops:
        a1 = sim.agents[1]
        _free(a1)
        a1.node = sim.city.gateways[0]
        a1.day_plan = [{"when": "午後", "what": "shop", "place": "", "done": False}]
        act1 = routine.decide(a1, step, sim, "x",
                              sim.hub.stream("decide", a1.id, step),
                              has_company=False)
        assert act1["type"] == "move_to"
        assert act1["dest"] in {n for n in shops if n != a1.node}


def test_day_plan_count_is_k_invariant(tmp_path):
    """day_plan の総数は k∈{free,off} で同一(R1: 計画の呼数は k と無関係)。"""
    def count(wb: str) -> int:
        cfg = load_config(["run.seed=42", "run.n_agents=10", "run.n_steps=200",
                           f"run.name=kinv_{wb}", f"k.writeback={wb}"])
        sim = Simulation(cfg, out_dir=tmp_path / f"kinv_{wb}")
        sim.run()
        return sum(1 for e in sim.logger.events if e.kind == "day_plan")
    n_free, n_off = count("free"), count("off")
    assert n_free == n_off > 0, f"day_plan 数が k で異なる: free={n_free} off={n_off}"
