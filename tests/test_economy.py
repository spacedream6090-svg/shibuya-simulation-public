"""経済 v0(賃金+消費+バイト+心理接続。ユーザー決定 2026-07-05)のテスト。

1. 賃金: 勤務完遂(exit_building)→ wage イベント + 残高増。
2. 消費: 飲食到着 → spend + 残高減。残高不足 → 無料/回避で spend が出ない。
3. money_pressure: 残高を人工的に下げた agent に grievance 増 + state_update(money_pressure)。
4. バイト: part_time を持つ agent がシフト時間にバイト先へ向かう/入る。
5. 決定論: 同 seed 2回で money 系イベント列が一致。
"""
from __future__ import annotations

from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 20, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dotlist)
    return Simulation(cfg, out_dir=tmp_path / name)


def _pt_poi(sim):
    """建物入口のある shop/food POI(バイト先候補)。"""
    pool = sim.city.pois_by_cat("shop") + sim.city.pois_by_cat("food")
    for p in pool:
        if p.get("building") and sim.city.has_building(p["building"]):
            return p
    raise AssertionError("建物付きの shop/food POI が地図に無い")


# --------------------------------------------------------------------------- #
def test_wage_paid_on_work_completion(tmp_path):
    sim = _sim(tmp_path, "wage")
    a = sim.agents[0]
    bld = sim.city.buildings[0]
    a.building = bld["id"]
    a.work_building = bld["id"]
    a.work_start_min, a.work_end_min = 480, 1020    # 8:00〜17:00
    a.activity = "working"
    a.part_time = None
    a.wage = 12000.0
    a.money = 5000.0
    scheduler._apply(sim, a, {"type": "exit_building"}, step=60, sim_min=1030)
    assert a.money == 17000.0                        # 賃金で残高増
    wages = [e for e in sim.logger.events
             if e.kind == "wage" and e.agent_id == a.id]
    assert wages, "wage イベントが出ていない"
    assert wages[-1].payload["amount"] == 12000.0
    assert wages[-1].payload["balance"] == 17000.0


def test_no_wage_when_shift_not_over(tmp_path):
    """勤務時間内の退館は賃金を出さない(完遂していない)。"""
    sim = _sim(tmp_path, "nowage")
    a = sim.agents[0]
    bld = sim.city.buildings[0]
    a.building = bld["id"]
    a.work_building = bld["id"]
    a.work_start_min, a.work_end_min = 480, 1020
    a.activity = "working"
    a.part_time = None
    a.wage = 12000.0
    a.money = 5000.0
    scheduler._apply(sim, a, {"type": "exit_building"}, step=40, sim_min=800)
    assert a.money == 5000.0
    assert not [e for e in sim.logger.events
                if e.kind == "wage" and e.agent_id == a.id]


def test_meal_arrival_spends(tmp_path):
    sim = _sim(tmp_path, "spend")
    a = sim.agents[0]
    food = sim.city.pois_by_cat("food")[0]
    a.node = food["node"]
    a.money = 10000.0
    scheduler._charge_meal(sim, a, step=5, sim_min=700)
    spends = [e for e in sim.logger.events
              if e.kind == "spend" and e.agent_id == a.id]
    assert spends, "spend イベントが出ていない"
    price = sim.economy["prices"]["food"]
    assert a.money == 10000.0 - price
    assert spends[-1].payload["cat"] == "food"
    assert spends[-1].payload["balance"] == a.money


def test_broke_agent_avoids_paid_food(tmp_path):
    """残高が価格に満たない場合、食事の行き先選択で有料店を避ける(spend が出ない)。"""
    sim = _sim(tmp_path, "broke", **{"world.meal_prob": 1.0})
    a = sim.agents[0]
    a.loc, a.building, a.route, a.sleeping = "street", None, [], False
    a.work_start_min, a.part_time = -1, None
    a.stay_until = 0
    a.bedtime_min = 1380                             # 23:00(sim_min 720 は就寝前)
    a.node = sim.city.gateways[0]

    step = 30                                        # sim_min 720(12:00)= 食事時間帯
    a.money = 1e9                                    # 潤沢 → 食事を選ぶ
    rich = routine.decide(a, step, sim, "x",
                          sim.hub.stream("decide", a.id, step), has_company=False)
    a.money = 10.0                                   # 逼迫 → 有料店を避ける
    broke = routine.decide(a, step, sim, "x",
                           sim.hub.stream("decide", a.id, step), has_company=False)
    assert rich["type"] == "move_to" and rich.get("activity") == "eating"
    assert broke.get("activity") != "eating"


def test_money_pressure_raises_grievance(tmp_path):
    sim = _sim(tmp_path, "press")
    a = sim.agents[0]
    a.money = 100.0                                  # 閾値(2000)未満に人工的に下げる
    before = a.states["grievance"]
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    assert a.states["grievance"] > before
    ups = [e for e in sim.logger.events
           if e.kind == "state_update" and e.agent_id == a.id
           and e.payload["cause"] == "money_pressure"]
    assert ups, "money_pressure の state_update が出ていない"


def test_part_time_routine_heads_to_shift(tmp_path):
    sim = _sim(tmp_path, "pt")
    a = sim.agents[0]
    p = _pt_poi(sim)
    a.part_time = {"name": p["name"], "node": p["node"], "building": p["building"],
                   "floor": 1, "days": list(range(7)),
                   "start_min": 1020, "end_min": 1260, "hours": 4, "pay": 4400}
    a.loc, a.building, a.route, a.sleeping = "street", None, [], False
    a.work_start_min, a.stay_until = -1, 0
    a.bedtime_min = 1380

    # シフト時間(17:00〜21:00)= step 66 は sim_min 1080。別ノードにいる → 移動を開始
    step = 66
    a.node = sim.city.gateways[0] if sim.city.gateways[0] != p["node"] \
        else sim.city.gateways[-1]
    act = routine.decide(a, step, sim, "x",
                         sim.hub.stream("decide", a.id, step), has_company=False)
    assert act["type"] == "move_to" and act["dest"] == p["node"]

    # バイト先ノードに着いたら建物に入って勤務する
    a.node = p["node"]
    act2 = routine.decide(a, step + 1, sim, "x",
                          sim.hub.stream("decide", a.id, step + 1), has_company=False)
    assert act2["type"] == "enter_building"
    assert act2["building"] == p["building"]
    assert act2["activity"] == "working"


def test_part_time_shift_pays_wage(tmp_path):
    """バイトのシフト完遂(退館)で時給×時間を支給する。"""
    sim = _sim(tmp_path, "ptpay")
    a = sim.agents[0]
    p = _pt_poi(sim)
    a.part_time = {"name": p["name"], "node": p["node"], "building": p["building"],
                   "floor": 1, "days": list(range(7)),
                   "start_min": 1020, "end_min": 1260, "hours": 4, "pay": 4400}
    a.building = p["building"]
    a.work_building = ""                             # 本業は無し(学生・フリーター)
    a.work_start_min = -1
    a.activity = "working"
    a.wage = 0.0
    a.money = 3000.0
    scheduler._apply(sim, a, {"type": "exit_building"}, step=90, sim_min=1270)
    assert a.money == 3000.0 + 4400.0
    wages = [e for e in sim.logger.events
             if e.kind == "wage" and e.agent_id == a.id]
    assert wages and wages[-1].payload["amount"] == 4400.0


def test_money_events_deterministic(tmp_path):
    def run(name):
        cfg = load_config(["run.seed=5", "run.n_agents=20", "run.n_steps=72",
                           f"run.name={name}"])
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.run()
        return [(e.step, e.agent_id, e.kind, e.payload) for e in sim.logger.events
                if e.kind in ("wage", "spend")]
    a, b = run("m1"), run("m2")
    assert a == b, "同 seed で money 系イベント列が一致しない(非決定論)"
    assert a, "money 系イベントが一件も出ていない(経済が動いていない)"
