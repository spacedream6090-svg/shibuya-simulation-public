"""交通機関: タクシー+簡易バス(ユーザー要望 2026-07-06)のテスト。

1. 遠距離+所持金で taxi が発動(mode=car)し、到着課金で ride+spend(cat=taxi)・残高減。
   自家用車保有者は使わない。
2. taxi.enabled=false は不発(action を変えない)。
3. bus.enabled=true の合成路線で乗車(mode=car=短縮)・運賃(cat=bus)。
4. 既定(bus OFF)は buses 未構築で挙動不変(乗車が出ない)。
電車の車内・駅構内の高度化はしない(GTFS 実ダイヤ API は申請待ち)。
"""
from __future__ import annotations

import math

from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world.transit import BusNetwork


def _sim(tmp_path, name: str, n: int = 10, steps: int = 1, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _far_pair(sim, min_d: float = 700.0):
    """直線距離が min_d を超えるノードの組(src, dst)を返す。"""
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    src = nodes[0]
    sx, sy = sim.city.node_xy(src)
    for n in nodes:
        x, y = sim.city.node_xy(n)
        if math.hypot(x - sx, y - sy) > min_d:
            return src, n
    raise AssertionError("十分に離れたノード対が地図に無い")


# --------------------------------------------------------------------------- #
def test_taxi_fires_on_far_trip_and_charges(tmp_path):
    sim = _sim(tmp_path, "taxi", **{"transit_ride.taxi.prob": 1.0})
    a = sim.agents[0]
    a.has_car, a.money = False, 1_000_000.0
    src, dst = _far_pair(sim, 700.0)
    a.node = src

    extra = routine._ride_extra(a, sim, dst, step=5, sim_min=800)
    assert extra is not None, "遠距離+所持金でタクシーが発動しない"
    mode, ride = extra
    assert mode == "car" and ride["mode"] == "taxi"
    assert ride["fare"] > sim.ridecfg["taxi"]["base_fare"]   # 距離運賃が乗る
    assert ride["from"] == src and ride["to"] == dst

    # 自家用車保有者は使わない
    a.has_car = True
    assert routine._ride_extra(a, sim, dst, step=5, sim_min=800) is None
    a.has_car = False

    # 到着課金: ride + spend(cat=taxi)+ 残高減
    a.money = 5000.0
    scheduler._charge_ride(sim, a, {"mode": "taxi", "fare": 840.0,
                                    "from": src, "to": dst}, 6, 810)
    assert a.money == 5000.0 - 840.0
    rides = [e for e in sim.logger.events if e.kind == "ride"]
    spends = [e for e in sim.logger.events
              if e.kind == "spend" and e.payload["cat"] == "taxi"]
    assert rides and spends
    assert spends[-1].payload["balance"] == a.money


def test_augment_ride_sets_car_mode_and_ride(tmp_path):
    """発動時は move_to の mode を car に上書きし、action['ride'] を付ける。"""
    sim = _sim(tmp_path, "aug", **{"transit_ride.taxi.prob": 1.0})
    a = sim.agents[0]
    a.has_car, a.money = False, 1_000_000.0
    src, dst = _far_pair(sim, 700.0)
    a.node = src
    action = routine._augment_ride(
        a, sim, {"type": "move_to", "dest": dst, "mode": "walk"}, 5, 800)
    assert action["mode"] == "car"
    assert action["ride"]["mode"] == "taxi"


def test_taxi_disabled_does_not_fire(tmp_path):
    sim = _sim(tmp_path, "notaxi",
               **{"transit_ride.taxi.enabled": "false",
                  "transit_ride.taxi.prob": 1.0})
    a = sim.agents[0]
    a.has_car, a.money = False, 1_000_000.0
    src, dst = _far_pair(sim, 700.0)
    a.node = src
    assert routine._ride_extra(a, sim, dst, step=5, sim_min=800) is None
    out = routine._augment_ride(
        a, sim, {"type": "move_to", "dest": dst, "mode": "walk"}, 5, 800)
    assert "ride" not in out and out["mode"] == "walk"     # action は不変


def test_bus_synthetic_ride_shortcut_and_fare(tmp_path):
    sim = _sim(tmp_path, "bus", **{"transit_ride.bus.enabled": "true"})
    sim.ridecfg["taxi"]["enabled"] = False                # bus 経路を確定
    src, dst = _far_pair(sim, 300.0)
    sim.buses = BusNetwork([{"name": "循環バス", "stops": [src, dst]}],
                           sim.city, stop_radius_m=100.0, headway_steps=1)
    a = sim.agents[0]
    a.node, a.money = src, 1_000_000.0

    extra = routine._ride_extra(a, sim, dst, step=3, sim_min=600)
    assert extra is not None, "合成バス路線で乗車が発動しない"
    mode, ride = extra
    assert mode == "car"                                  # 車速で走る = 移動短縮
    assert ride["mode"] == "bus"
    assert ride["fare"] == sim.ridecfg["bus"]["fare"]
    assert ride["from"] == src and ride["to"] == dst
    # 短縮の裏付け: 走行モード(car)は徒歩より速い
    speeds = sim.cfg.world.modes.speeds
    assert float(speeds["car"]) > float(speeds["walk"])
    # 便(headway=1 step)は常に運行
    assert sim.buses.serves(600) is True

    # 運賃課金(cat=bus)
    a.money = 1000.0
    scheduler._charge_ride(sim, a, ride, 4, 610)
    assert a.money == 1000.0 - sim.ridecfg["bus"]["fare"]
    assert [e for e in sim.logger.events
            if e.kind == "spend" and e.payload["cat"] == "bus"]


def test_bus_headway_gates_service():
    """headway_steps>1 のとき、便のある step でだけ serves() が True。"""
    class _Stub:
        graph = type("G", (), {"nodes": set()})()
    # serves は sim_min のみ依存(停留所解決に city 不要)
    bus = BusNetwork([], _Stub(), stop_radius_m=100.0, headway_steps=3)
    assert bus.serves(0) is True          # step 0
    assert bus.serves(10) is False        # step 1
    assert bus.serves(30) is True         # step 3


def test_bus_off_by_default_no_ride(tmp_path):
    sim = _sim(tmp_path, "busoff")
    assert sim.buses is None
    assert sim.ridecfg["bus"]["enabled"] is False
    sim.ridecfg["taxi"]["enabled"] = False                # taxi も条件外へ
    a = sim.agents[0]
    a.node = sim.dests[0]
    assert routine._ride_extra(a, sim, sim.dests[1], 3, 600) is None
