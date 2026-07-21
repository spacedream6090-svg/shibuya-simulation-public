"""実バスダイヤの静的表 v-Ride-2 のテスト。

正典: docs/research/sumo-live-transit.md §3(バス pt・静的表推奨=SUMO を回さない軽量版)。
方針(既存の鉄則を継承):
- ビルド決定論: build_bus_table が合成 GTFS から同一入力で同一表を吐く(pandas 不使用・純 Python)。
- 実ダイヤ近似の待ち/所要: BusTable.find_ride が次便待ち wait_s + 区間所要 ride_s を実時刻表から返す。
- 既定 OFF 一致: bus_table 無効(既定)は sim.bus_table is None=従来の簡易バス近似のまま=ride に
  wait_s が付かない=バイト一致(ゴールデンを守る)。
- ON 経路: bus_table を実ノード整合の合成表で差せば _ride_extra が実ダイヤ乗車(wait_s/ride_s)を返す。
実バス取得は導線のみ(匿名直 DL は 403=要キー)。テストは合成 GTFS(中立名)でビルド経路を検証する。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from society.bus_table import BusTable
from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "bus_gtfs_synth"
ORIGIN = (35.6595, 139.70062)
BBOX = (35.656, 139.695, 35.6625, 139.706)


# --------------------------------------------------------------------------- helpers
def _sim(tmp_path, name, n=10, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


class _StubCity:
    """BusTable.find_ride が要求する最小 city(graph.nodes と node_xy)。"""

    def __init__(self, xy: dict):
        self._xy = dict(xy)
        self.graph = type("G", (), {"nodes": set(xy)})()
        self.meta = {}

    def node_xy(self, n):
        return self._xy[n]


# --------------------------------------------------------------------- ビルド
def test_build_deterministic_and_schema(tmp_path):
    """合成 GTFS から同一入力で同一表(generated_at 以外)=決定論。スキーマ・平日絞り・bbox 絞りを確認。"""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_bus_table import build_table

    a = build_table(FIXTURE, ORIGIN, BBOX, "合成テスト")
    b = build_table(FIXTURE, ORIGIN, BBOX, "合成テスト")
    # generated_at(日付)以外は完全一致
    a["_meta"].pop("generated_at"); b["_meta"].pop("generated_at")
    assert json.dumps(a, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b, ensure_ascii=False, sort_keys=True), "ビルドが非決定論"

    assert a["_meta"]["n_stops"] == 7           # P9 は bbox 外=除外
    assert a["_meta"]["n_routes"] == 2
    assert a["_meta"]["calendar"] == "weekday"
    r1 = next(r for r in a["routes"] if r["name"] == "系統M1")
    assert r1["stops"] == ["P1", "P2", "P3", "P4"]       # P9(圏外)は落ちる
    assert r1["cum_sec"] == [0.0, 180.0, 420.0, 600.0]   # 区間所要(便横断の中央値)
    assert r1["departures"] == [480, 495, 510]           # 平日のみ(週末 540 は除外)
    assert 540 not in r1["departures"], "週末便が平日ダイヤに混入"
    # cum_sec は単調非減少(順方向)
    assert all(r["cum_sec"] == sorted(r["cum_sec"]) for r in a["routes"])
    # 停留所には投影座標 x,y が載る(city 座標系)
    assert all("x" in s and "y" in s for s in a["stops"])


# --------------------------------------------------------------------- find_ride
def _table():
    return {
        "_meta": {"origin_latlon": list(ORIGIN), "step_seconds": 600},
        "stops": [
            {"id": "P1", "name": "停1", "x": 0.0, "y": 0.0},
            {"id": "P2", "name": "停2", "x": 100.0, "y": 0.0},
            {"id": "P3", "name": "停3", "x": 200.0, "y": 0.0},
            {"id": "P4", "name": "停4", "x": 300.0, "y": 0.0}],
        "routes": [{"name": "系統M1", "direction": "0",
                    "stops": ["P1", "P2", "P3", "P4"],
                    "cum_sec": [0.0, 180.0, 420.0, 600.0],
                    "departures": [480, 495, 510]}]}


def test_find_ride_wait_and_ride_from_timetable():
    """access 圏内・順方向で next 便待ち wait_s + 区間所要 ride_s を実時刻表どおり返す。"""
    city = _StubCity({"A": (5.0, 0.0), "B": (295.0, 0.0), "C": (5000.0, 5000.0)})
    bt = BusTable(_table(), city, access_radius_m=50.0)

    # 07:50(=470分)呼び: 次便は 08:00(480) → 待ち 600s、区間 P1→P4 = 600s
    r = bt.find_ride("A", "B", city, 470)
    assert r is not None and r["from"] == "P1" and r["to"] == "P4"
    assert r["wait_s"] == 600.0 and r["ride_s"] == 600.0
    assert r["line"] == "系統M1"

    # 08:05(=485)呼び: 次便は 08:15(495) → 待ち 600s
    r2 = bt.find_ride("A", "B", city, 485)
    assert r2["wait_s"] == 600.0 and r2["ride_s"] == 600.0

    # 逆方向(B→A)は順方向でない=乗れない
    assert bt.find_ride("B", "A", city, 470) is None
    # access 圏外(C)は乗れない
    assert bt.find_ride("A", "C", city, 470) is None
    # 最終便より後(08:40=520)は翌日始発へ回る=大きな待ち
    r3 = bt.find_ride("A", "B", city, 520)
    assert r3 is not None and r3["wait_s"] > 3600.0


def test_load_bus_table_missing_returns_none(tmp_path):
    """表ファイルが無い/壊れは None(呼び出し側は従来近似へ後退)。"""
    from society.bus_table import load_bus_table
    city = _StubCity({"A": (0.0, 0.0)})
    assert load_bus_table(tmp_path / "nope.json", city) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert load_bus_table(bad, city) is None


# --------------------------------------------------------------------- 既定 OFF
def test_default_config_bus_table_is_none(tmp_path):
    """既定(bus_table OFF)は sim.bus_table is None=SUMO なし・従来近似のまま。"""
    sim = _sim(tmp_path, "btoff")
    assert sim.bus_table is None
    assert load_config([]).transit_ride.bus_table.enabled is False


def test_off_bus_ride_has_no_wait_s(tmp_path):
    """簡易バス(bus.enabled=true・bus_table OFF)の ride は従来どおり wait_s を持たない=バイト一致。"""
    from society.world.transit import BusNetwork
    sim = _sim(tmp_path, "simplebus", **{"transit_ride.bus.enabled": "true"})
    assert sim.bus_table is None
    sim.ridecfg["taxi"]["enabled"] = False
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    src, dst = nodes[0], nodes[1]
    sim.buses = BusNetwork([{"name": "循環バス", "stops": [src, dst]}],
                           sim.city, stop_radius_m=100.0, headway_steps=1)
    a = sim.agents[0]
    a.node, a.money = src, 1_000_000.0
    extra = routine._ride_extra(a, sim, dst, step=3, sim_min=600)
    assert extra is not None and extra[1]["mode"] == "bus"
    assert "wait_s" not in extra[1] and "ride_s" not in extra[1], "OFF の bus に観測が漏れた"


# --------------------------------------------------------------------- ON 経路
def test_bus_table_on_real_timetable_ride_and_hold(tmp_path):
    """bus_table を実ノード整合の合成表で差すと _ride_extra が実ダイヤ乗車(wait_s/ride_s)を返し、
    _bus_ride_hold が到着 step の追加待ち(hold)+ delay_s を据える。"""
    sim = _sim(tmp_path, "bton", **{"transit_ride.bus.enabled": "true"})
    sim.ridecfg["taxi"]["enabled"] = False
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    src = nodes[0]
    sx, sy = sim.city.node_xy(src)
    dst = max(nodes, key=lambda n: math.hypot(*(p - q for p, q in
              zip(sim.city.node_xy(n), (sx, sy)))))
    dx, dy = sim.city.node_xy(dst)
    data = {"_meta": {"origin_latlon": list(sim.city.meta["origin_latlon"]),
                      "step_seconds": 600},
            "stops": [{"id": "S0", "name": "停0", "x": sx, "y": sy},
                      {"id": "S1", "name": "停1", "x": dx, "y": dy}],
            "routes": [{"name": "系統X", "direction": "0", "stops": ["S0", "S1"],
                        "cum_sec": [0.0, 600.0], "departures": [480, 495, 510]}]}
    sim.bus_table = BusTable(data, sim.city, access_radius_m=50.0)

    a = sim.agents[0]
    a.node, a.money, a.has_car = src, 1_000_000.0, False
    extra = routine._ride_extra(a, sim, dst, step=5, sim_min=470)
    assert extra is not None, "実ダイヤ表でバス乗車が発動しない"
    mode, ride = extra
    assert mode == "car" and ride["mode"] == "bus"
    assert ride["from"] == "S0" and ride["to"] == "S1"
    assert ride["wait_s"] == 600.0 and ride["ride_s"] == 600.0

    # _bus_ride_hold: hold と delay_s を据える(自由流は agent.node→dest で測る)
    a._ride_pending = ride
    a._taxi_hold_until = -1
    scheduler._bus_ride_hold(sim, a, dst, step=5)
    assert "delay_s" in a._ride_pending and a._ride_pending["delay_s"] >= 0.0
    assert getattr(a, "_taxi_hold_until", -1) >= 5, "次便待ちが到着 step の hold に反映されていない"

    # 到着課金で ride payload に wait_s/ride_s が載る(観測可能性)
    a.money = 5000.0
    scheduler._charge_ride(sim, a, a._ride_pending, 6, 480)
    rides = [e for e in sim.logger.events if e.kind == "ride"]
    assert rides and rides[-1].payload["mode"] == "bus"
    assert "wait_s" in rides[-1].payload and "ride_s" in rides[-1].payload
