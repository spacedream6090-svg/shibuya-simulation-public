"""②OD行列の純関数テスト(第31バッチ・分析スイート W2)。

scripts/analyze_od.py のトリップ抽出(route_start→arrive のペア)・目的帰属・
域外ゲートウェイゾーン・hour_bin・OD 集計を、合成イベントで検証する
(シミュ実行不要=決定論の軽量テスト・test_flows_grid と同流儀)。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import analyze_od as od   # noqa: E402


def _events(specs):
    """(kind, aid, step, x, y, payload) のリストから seq/sim_min 付き event dict 列を作る。"""
    out = []
    for seq, (kind, aid, step, x, y, payload) in enumerate(specs):
        out.append({"seq": seq, "step": step, "sim_min": step * 10,
                    "agent_id": aid, "kind": kind, "x": x, "y": y,
                    "payload": payload or {}})
    return out


# --------------------------------------------------------------- ビニング/時間
def test_hour_bin_and_district_zone():
    # hour_bin = (step % 144) // 6(1step=10分・6step=1時間)
    assert od.hour_bin_of_step(0) == 0
    assert od.hour_bin_of_step(6) == 1
    assert od.hour_bin_of_step(47) == 7          # 07:50 → bin 7
    assert od.hour_bin_of_step(144) == 0         # 翌日 00:00 → bin 0
    assert od.hour_bin_of_step(144 + 6) == 1
    # district_zone(100m メッシュ・負座標)
    assert od.district_zone(10, 10, 100) == "D:0:0"
    assert od.district_zone(250, 250, 100) == "D:2:2"
    assert od.district_zone(-5, -5, 100) == "D:-1:-1"


# --------------------------------------------------------------- 目的写像
def test_purpose_mappings():
    assert od._SPEND_PURPOSE["food"] == "food"
    assert od._SPEND_PURPOSE["nightlife"] == "leisure"
    assert od._SPEND_PURPOSE["fixed_cost"] == "other"
    assert od._BUILDING_PURPOSE["office"] == "work"
    assert od._BUILDING_PURPOSE["retail"] == "shop"
    assert od._BUILDING_PURPOSE["house?"] == "home"
    assert od._BUILDING_PURPOSE["generic"] == "other"


def test_attribute_purpose_priority_and_fallback():
    bk = {"b_off": "office", "b_gen": "generic"}
    # spend food が強い信号 → food
    w1 = _events([("spend", 1, 3, 0, 0, {"cat": "food"})])
    assert od.attribute_purpose(w1, bk) == "food"
    # 建物 office → work(generic は非情報 → 無視され work が勝つ)
    w2 = _events([("enter_building", 1, 3, 0, 0, {"building": "b_gen"}),
                  ("enter_building", 1, 3, 0, 0, {"building": "b_off"})])
    assert od.attribute_purpose(w2, bk) == "work"
    # 強い信号なし → day_plan.what を弱い信号として使う(meal → food)
    w3 = _events([("day_plan", 1, 3, 0, 0, {"plan": [{"what": "meal"}]})])
    assert od.attribute_purpose(w3, bk) == "food"
    # どれも決め手なし → other(捏造禁止)
    w4 = _events([("spend", 1, 3, 0, 0, {"cat": "fixed_cost"})])
    assert od.attribute_purpose(w4, bk) == "other"
    assert od.attribute_purpose([], bk) == "other"


# --------------------------------------------------------------- トリップ抽出
def test_interior_trip_and_purpose():
    """route_start(exit=False)→arrive の域内トリップ + 到着建物 kind から目的帰属。"""
    bk = {"b_off": "office"}
    evs = _events([
        ("route_start", 1, 0, 10, 10, {"dest": "n1", "exit": False}),
        ("arrive", 1, 2, 250, 250, {"node": "n1"}),
        ("enter_building", 1, 2, 250, 250, {"building": "b_off"}),
    ])
    res = od.extract_trips(evs, district_m=100, building_kinds=bk)
    assert len(res["trips"]) == 1
    t = res["trips"][0]
    assert (t["origin"], t["dest"]) == ("D:0:0", "D:2:2")
    assert t["hour_bin"] == 0
    assert t["purpose"] == "work"
    assert res["n_unpaired"] == 0


def test_outflow_trip_to_gateway():
    """route_start(exit=True)→arrive(node=gateway)= 流出。dest は域外ゾーン。"""
    evs = _events([
        ("route_start", 1, 0, 10, 10, {"dest": "nGW", "exit": True, "homing": True}),
        ("arrive", 1, 2, 900, 900, {"node": "nGW"}),
        ("exit_area", 1, 2, 900, 900, {"gateway": "nGW", "homing": True}),
    ])
    res = od.extract_trips(evs, district_m=100, building_kinds={})
    assert len(res["trips"]) == 1                 # exit_area では二重計上しない
    t = res["trips"][0]
    assert t["origin"] == "D:0:0"
    assert t["dest"] == "G:nGW"                   # 域外ゲートウェイゾーン
    assert t["purpose"] == "home"                 # homing=True → home
    # ゲートウェイ重心が観測 (x,y) で記録される
    assert res["gateway_xy"]["G:nGW"] == (900.0, 900.0)


def test_inflow_trip_from_gateway():
    """enter_area(域外→渋谷)直後の route_start は起点を域外ゲートウェイに差し替える。"""
    evs = _events([
        ("enter_area", 1, 0, -900, -900, {"gateway": "nIN", "via": "train"}),
        ("route_start", 1, 0, -900, -900, {"dest": "n3", "exit": False}),
        ("arrive", 1, 2, 50, 50, {"node": "n3"}),
        ("spend", 1, 2, 50, 50, {"cat": "food"}),
    ])
    res = od.extract_trips(evs, district_m=100, building_kinds={})
    assert len(res["trips"]) == 1
    t = res["trips"][0]
    assert t["origin"] == "G:nIN"                 # 域外起点(流入)
    assert t["dest"] == "D:0:0"
    assert t["purpose"] == "food"                 # 到着後の spend food
    assert res["gateway_xy"]["G:nIN"] == (-900.0, -900.0)


def test_interior_homing_falls_back_to_home():
    """到着後に決め手なしの域内トリップでも homing=True なら home に帰属する。"""
    evs = _events([
        ("route_start", 1, 0, 10, 10, {"dest": "n5", "exit": False, "homing": True}),
        ("arrive", 1, 2, 250, 250, {"node": "n5"}),
        # 到着後に建物入館も spend も無い(=通常なら other)
    ])
    res = od.extract_trips(evs, district_m=100, building_kinds={})
    assert len(res["trips"]) == 1
    assert res["trips"][0]["purpose"] == "home"
    # ただし強い信号(spend food)があれば homing より信号を優先する
    evs2 = _events([
        ("route_start", 2, 0, 10, 10, {"dest": "n6", "exit": False, "homing": True}),
        ("arrive", 2, 2, 250, 250, {"node": "n6"}),
        ("spend", 2, 2, 250, 250, {"cat": "food"}),
    ])
    res2 = od.extract_trips(evs2, district_m=100, building_kinds={})
    assert res2["trips"][0]["purpose"] == "food"


def test_unpaired_route_start_not_counted():
    """対応 arrive を欠く route_start はトリップにしない(捏造回避)。"""
    evs = _events([
        ("route_start", 1, 0, 10, 10, {"dest": "n9", "exit": False}),
        # arrive が来ない
    ])
    res = od.extract_trips(evs, district_m=100, building_kinds={})
    assert res["trips"] == []
    assert res["n_unpaired"] == 1
    assert res["n_route_start"] == 1
    assert res["n_arrive"] == 0


def test_od_rows_aggregation_and_normalization():
    """同一 (origin,dest,hour_bin,purpose) の集計 + trips_per_day 正規化。"""
    trips = [
        {"origin": "D:0:0", "dest": "D:2:2", "hour_bin": 0, "purpose": "work", "day": 0},
        {"origin": "D:0:0", "dest": "D:2:2", "hour_bin": 0, "purpose": "work", "day": 1},
        {"origin": "D:0:0", "dest": "D:1:1", "hour_bin": 3, "purpose": "food", "day": 1},
    ]
    cols = od.od_rows(trips, n_days=2)
    # 決定論ソート(origin,dest,hour_bin,purpose)。行1= D:0:0->D:1:1、行2= D:0:0->D:2:2
    assert cols["origin"] == ["D:0:0", "D:0:0"]
    assert cols["dest"] == ["D:1:1", "D:2:2"]
    assert cols["trips"] == [1, 2]
    assert cols["trips_per_day"] == [0.5, 1.0]    # ÷ n_days=2
    assert set(cols) == {"origin", "dest", "hour_bin", "purpose",
                         "trips", "trips_per_day"}


def test_empty_events_degenerate():
    """イベント 0 件 → トリップ 0・空列(データ不足を捏造しない)。"""
    res = od.extract_trips([], district_m=100, building_kinds={})
    assert res["trips"] == []
    cols = od.od_rows([], n_days=0)
    assert cols["origin"] == [] and cols["trips"] == []
    # レポートは「データ不足」を明記(値は捏造しない)
    rep = od.render_report("empty", cols, res, district_m=100, n_days=0)
    assert "データ不足" in rep
