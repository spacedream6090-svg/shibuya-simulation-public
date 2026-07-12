"""流入通勤者(commuter)+ POI 拡大(office/school/cinema/hall/landmark)のテスト。

バッチ E4(ユーザー要望 2026-07-06 / docs/research/shibuya-inflow.md):
- build_map.py の新カテゴリ抽出(合成 OSM)とハチ公フォールバック。
- build_personas.py の commuter 骨格(決定論・到着リード二峰の妥当域・出入口配分)。
- commuter が朝(範囲外)→ enter_area → 勤務 → 夕 homing → exit_area する(mock 1日)。
- 旧名簿(commute 無し)・旧地図(landmark 無し)で挙動が不変(no-op)。

ネットワークは使わない(合成 OSM + 既存/生成済みファイル + 手続き名簿)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from society.config import load_config
from society.engine.simulation import Simulation
from society.world.map import CityMap

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _load_build_map():
    spec = importlib.util.spec_from_file_location(
        "build_map_inflow", REPO / "scripts" / "build_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()


# --------------------------------------------------------------------------- #
# 合成 OSM: 地上グリッド + 各新カテゴリの POI ノード
# --------------------------------------------------------------------------- #
def _node(i, lat, lon, tags=None):
    e = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        e["tags"] = tags
    return e


def _way(i, nodes, tags):
    return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


MOCK_BBOX = (35.6585, 139.6990, 35.6600, 139.7025)


def _mock_raw(with_hachiko: bool):
    els = []
    # 地上 footway n1..n6(東へ)
    lons = [139.70000, 139.70010, 139.70020, 139.70030, 139.70040, 139.70050]
    for k, lon in enumerate(lons, 1):
        els.append(_node(k, 35.65920, lon))
    els.append(_way(101, [1, 2, 3, 4, 5, 6], {"highway": "footway"}))
    # 交差点維持の縦道
    els += [_node(7, 35.65910, 139.70020), _node(8, 35.65930, 139.70020)]
    els.append(_way(102, [7, 3, 8], {"highway": "footway"}))
    # 各新カテゴリの POI(ノード)
    els += [
        _node(20, 35.65921, 139.70001, {"office": "it", "name": "テスト商事"}),
        _node(21, 35.65921, 139.70011, {"amenity": "university", "name": "テスト大学"}),
        _node(22, 35.65921, 139.70021, {"amenity": "cinema", "name": "テストシネマ"}),
        _node(23, 35.65921, 139.70031, {"amenity": "events_venue", "name": "テストホール"}),
        _node(24, 35.65921, 139.70041, {"tourism": "artwork", "name": "テスト彫像"}),
        _node(25, 35.65921, 139.70051, {"amenity": "restaurant", "name": "テスト食堂"}),
    ]
    if with_hachiko:
        # ハチ公フォールバック近傍(35.6590, 139.7005)に landmark を置く
        els.append(_node(26, 35.65900, 139.70050,
                         {"amenity": "marketplace", "name": "忠犬ハチ公像"}))
    return {"elements": els}


# --------------------------------------------------------------------------- #
# 1. POI 拡大(build_map の新カテゴリ抽出)
# --------------------------------------------------------------------------- #
def test_build_map_new_categories():
    data = BM.build(_mock_raw(with_hachiko=True), MOCK_BBOX, None)
    cats = {p["cat"] for p in data["pois"]}
    for expect in ("office", "school", "cinema", "hall", "landmark"):
        assert expect in cats, f"{expect} POI が抽出されていない: {cats}"
    # 既存カテゴリも壊れない
    assert "food" in cats


def test_hachiko_present_from_osm_no_fallback():
    data = BM.build(_mock_raw(with_hachiko=True), MOCK_BBOX, None)
    landmarks = [p for p in data["pois"] if p["cat"] == "landmark"]
    assert any("ハチ公" in p["name"] for p in landmarks)
    assert data["meta"]["_stats"]["hachiko_source"] == "osm"
    assert not any(p["id"] == "p_hachiko_fallback" for p in data["pois"])


def test_hachiko_fallback_when_absent():
    data = BM.build(_mock_raw(with_hachiko=False), MOCK_BBOX, None)
    fb = [p for p in data["pois"] if p["id"] == "p_hachiko_fallback"]
    assert fb, "ハチ公像フォールバックが置かれていない"
    assert fb[0]["cat"] == "landmark" and "ハチ公" in fb[0]["name"]
    assert fb[0]["node"] in {n["id"] for n in data["nodes"]}
    assert data["meta"]["_stats"]["hachiko_source"] == "fallback"


def test_building_kind_reflects_office_from_poi():
    """POI(office)を内包する用途不明の建物は kind=office に格上げされる。"""
    raw = _mock_raw(with_hachiko=False)
    # n20(office, 35.65921/139.70001)を囲む建物(area>=100 で POI 紐付け対象)。
    # 用途タグは "yes"=generic → office POI 内包で kind=office へ格上げされることを見る。
    raw["elements"] += [
        _node(30, 35.65914, 139.69993), _node(31, 35.65914, 139.70009),
        _node(32, 35.65928, 139.70009), _node(33, 35.65928, 139.69993),
    ]
    raw["elements"].append(_way(300, [30, 31, 32, 33, 30],
                                {"building": "yes", "name": "無名ビル"}))
    data = BM.build(raw, MOCK_BBOX, None)
    hosts = [p.get("building") for p in data["pois"]
             if p["cat"] == "office" and p.get("building")]
    assert hosts, "office POI が建物に紐付いていない(テスト前提が崩れた)"
    bld = next(b for b in data["buildings"] if b["id"] == hosts[0])
    assert bld["kind"] in ("office", "public"), bld["kind"]


# --------------------------------------------------------------------------- #
# 2. commuter 骨格(build_personas)の決定論・分布妥当域
# --------------------------------------------------------------------------- #
def _load_build_personas():
    spec = importlib.util.spec_from_file_location(
        "build_personas_inflow", REPO / "scripts" / "build_personas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BP = _load_build_personas()
POP = json.loads((DATA / "shibuya_population.json").read_text(encoding="utf-8"))


def test_commuter_skeletons_deterministic():
    a = BP.sample_skeletons(200, POP, np.random.default_rng(7))
    b = BP.sample_skeletons(200, POP, np.random.default_rng(7))
    assert a == b, "同一 seed で骨格が一致しない(決定論が壊れている)"


def test_commuter_distribution_valid_range():
    sk = BP.sample_skeletons(600, POP, np.random.default_rng(3))
    coms = [s for s in sk if s.get("commute")]
    assert coms, "commuter が1人も生成されていない"
    # 全 commuter は visitor(下位型)かつ到着リードが妥当域
    assert all(s["visitor"] for s in coms)
    assert all(10 <= s["arrival_lead_min"] <= 120 for s in coms)
    assert all(s["commute_gateway"] in ("station", "edge") for s in coms)
    # 出入口は駅が大多数(~85-90%)
    st = sum(1 for s in coms if s["commute_gateway"] == "station")
    assert st / len(coms) > 0.7, f"駅ゲートウェイ比率が低い: {st}/{len(coms)}"
    # 二峰: 学生(大学生)の到着リード平均 < 会社員系の平均(通学が早め)
    stu = [s["arrival_lead_min"] for s in coms if s["occupation"] == "大学生"]
    wrk = [s["arrival_lead_min"] for s in coms if s["occupation"] != "大学生"]
    if stu and wrk:
        assert np.mean(stu) < np.mean(wrk), (np.mean(stu), np.mean(wrk))


def test_no_commuter_without_inflow_section():
    pop = {k: v for k, v in POP.items() if k != "inflow"}
    sk = BP.sample_skeletons(200, pop, np.random.default_rng(1))
    assert not any(s.get("commute") for s in sk)


# --------------------------------------------------------------------------- #
# 3. destinations() が landmark を余暇候補に含める(新地図のみ、旧は不変)
# --------------------------------------------------------------------------- #
def test_destinations_include_landmark(tmp_path):
    data = BM.build(_mock_raw(with_hachiko=True), MOCK_BBOX, None)
    p = tmp_path / "mock_map.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    c = CityMap(p)
    lmk_nodes = {q["node"] for q in c.poi_list if q["cat"] == "landmark"}
    assert lmk_nodes, "landmark POI が無い(テスト前提)"
    assert lmk_nodes <= set(c.destinations()), "landmark が目的地候補に入っていない"


def test_default_map_destinations_have_no_landmark():
    """旧コア地図(landmark cat 不在)では destinations は pois∪entrances のまま(不変)。"""
    c = CityMap(DATA / "shibuya_osm.json")
    assert not any(q["cat"] == "landmark" for q in c.poi_list)
    expect = set(c.pois()) | {b["entrance"] for b in c.buildings}
    assert set(c.destinations()) == expect


# --------------------------------------------------------------------------- #
# 4. commuter が朝流入 → 勤務 → 夕退出する(mock 1日、新名簿 + v6 地図)
# --------------------------------------------------------------------------- #
ROSTER = DATA / "personas_100_inflow.json"
V6 = DATA / "shibuya_osm_v6.json"


def _run_inflow(tmp_path, n_steps=144, seed=42):
    cfg = load_config(overrides=[
        "run.seed=%d" % seed, "run.n_agents=30", "run.n_steps=%d" % n_steps,
        "run.name=inflow", f"agents.personas_file={ROSTER}",
        f"world.map={V6}"])
    sim = Simulation(cfg, out_dir=tmp_path / "inflow")
    sim.run()
    return sim


@pytest.mark.skipif(not (ROSTER.exists() and V6.exists()),
                    reason="流入名簿 or v6 地図が未生成")
def test_commuter_inflow_work_exit(tmp_path):
    sim = _run_inflow(tmp_path)
    commuters = [a for a in sim.agents if a.commute]
    assert commuters, "commuter が名簿から生成されていない"
    # commuter は職場(office/school)が割り当たり、到着は勤務開始より前
    for a in commuters:
        assert a.work_start_min >= 0, f"commuter {a.id} に職場が無い"
        assert 0 <= a.arrival_min <= a.work_start_min, (a.arrival_min, a.work_start_min)

    ev = [(e.step, e.sim_min, e.agent_id, e.kind, e.payload)
          for e in sim.logger.events]
    com_ids = {a.id for a in commuters}
    enters = [(s, sm, aid) for (s, sm, aid, k, p) in ev
              if k == "enter_area" and aid in com_ids]
    exits = [(s, sm, aid, p) for (s, sm, aid, k, p) in ev
             if k == "exit_area" and aid in com_ids]
    assert enters, "commuter の朝の流入(enter_area)が出ていない"
    assert exits, "commuter の夕の流出(exit_area)が出ていない"
    # 各 commuter の初回の流入(=朝の到着。設計上の不変条件)は午前(5:00-11:00)
    first_enter: dict[int, int] = {}
    for (_s, sm, aid) in enters:
        first_enter.setdefault(aid, sm % 1440)
    assert first_enter, "commuter の初回流入が無い"
    assert all(5 * 60 <= m <= 11 * 60 for m in first_enter.values()), \
        sorted(first_enter.values())
    # 夕の流出は homing(帰宅)で、夕方以降(>=16:00 か翌0時前後の遅い裾)
    assert any(p.get("homing") for (_s, _sm, _a, p) in exits), "homing 帰宅が無い"
    # 少なくとも1人の commuter が勤務(working)に入る
    worked = any(e.payload.get("activity") == "working"
                 for e in sim.logger.events
                 if e.kind == "enter_building" and e.agent_id in com_ids)
    assert worked, "commuter が勤務(working)していない"


@pytest.mark.skipif(not (ROSTER.exists() and V6.exists()),
                    reason="流入名簿 or v6 地図が未生成")
def test_commuters_start_outside(tmp_path):
    """流入通勤者は step0 で範囲外(loc=outside)= 朝の到着待ち。非commuter は街の中。"""
    cfg = load_config(overrides=[
        "run.seed=42", "run.n_agents=30", "run.n_steps=1", "run.name=init",
        f"agents.personas_file={ROSTER}", f"world.map={V6}"])
    sim = Simulation(cfg, out_dir=tmp_path / "init")
    for a in sim.agents:
        if a.commute:
            assert a.loc == "outside", f"commuter {a.id} が範囲外に居ない"
            assert a.return_at >= 0
        else:
            assert a.loc == "street"


@pytest.mark.skipif(not (ROSTER.exists() and V6.exists()),
                    reason="流入名簿 or v6 地図が未生成")
def test_inflow_run_is_deterministic(tmp_path):
    a = _run_inflow(tmp_path / "a", n_steps=48)
    b = _run_inflow(tmp_path / "b", n_steps=48)
    la = [(e.step, e.agent_id, e.kind) for e in a.logger.events]
    lb = [(e.step, e.agent_id, e.kind) for e in b.logger.events]
    assert la == lb, "同一 seed で L1 が一致しない(決定論が壊れている)"
