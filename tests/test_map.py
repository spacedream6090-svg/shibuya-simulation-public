"""地図 v6 バッチのテスト。

対象:
- build_map.py の bbox/日付引数(Overpass QL 生成)とレイヤー判定
- 入口選択ロジック(モック OSM で main を最優先し、建物入口を最寄り道路ノードへ置換)
- 地下エッジが layer 属性を持ち、CityMap のグラフに載る(routing がそのまま使える)
- floorguide 拡張のスキーマ検証(地下階+connections)
- 生成済み wide 地図の基本整合(原点不変・ノード重複なし・CityMap ロード)
- CityMap の後方互換アクセサ(entrances / underground / connections)

ネットワークは使わない(合成 OSM + 既存/生成済みファイルのみ)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from society.world.map import CityMap

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _load_build_map():
    """scripts/build_map.py を(パッケージ外なので)動的ロードする。"""
    spec = importlib.util.spec_from_file_location(
        "build_map", REPO / "scripts" / "build_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()


# --- 合成 OSM: 地上グリッド + main 入口付き建物 + 地下 footway(tunnel) --------- #
def _node(i, lat, lon, tags=None):
    e = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        e["tags"] = tags
    return e


def _way(i, nodes, tags):
    return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


def _mock_raw():
    els = []
    # 東へ伸びる地上 footway n1..n5
    for k, lon in enumerate(
            [139.70062, 139.70071, 139.70081, 139.70091, 139.70102], 1):
        els.append(_node(k, 35.65950, lon))
    els.append(_way(101, [1, 2, 3, 4, 5], {"highway": "footway"}))
    # n3 を交差点に保つ縦の道
    els += [_node(6, 35.65940, 139.70081), _node(7, 35.65960, 139.70081)]
    els.append(_way(102, [6, 3, 7], {"highway": "footway"}))
    # 建物(東辺の中点に entrance=main。重心は西寄り=別ノードが最寄り)
    els += [_node(10, 35.65958, 139.70061), _node(11, 35.65958, 139.70102),
            _node(12, 35.65962, 139.70102), _node(13, 35.65962, 139.70061),
            _node(14, 35.65960, 139.70102, {"entrance": "main"})]
    els.append(_way(200, [10, 11, 14, 12, 13, 10],
                    {"building": "retail", "name": "テストビル",
                     "building:levels": "5"}))
    # 地下 footway(tunnel=yes)n5-n8-n9
    els += [_node(8, 35.65950, 139.70112), _node(9, 35.65950, 139.70122)]
    els.append(_way(103, [5, 8, 9], {"highway": "footway", "tunnel": "yes"}))
    return {"elements": els}


MOCK_BBOX = (35.6590, 139.6990, 35.6600, 139.7020)


# --------------------------------------------------------------------------- #
# 1. bbox / 日付引数のパース(Overpass QL 生成)
# --------------------------------------------------------------------------- #
def test_build_query_default_latest():
    q = BM.build_query((35.0, 139.0, 35.1, 139.1), None)
    assert "[out:json][timeout:180];" in q
    assert "[date:" not in q
    # bbox が全ブロックに展開される(12 個の (S,W,N,E))
    assert q.count("35.0,139.0,35.1,139.1") == 12


def test_build_query_attic_date():
    q = BM.build_query((35.0, 139.0, 35.1, 139.1), "2023-04-01")
    assert '[date:"2023-04-01T00:00:00Z"]' in q


def test_argparse_defaults_match_current_range():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=list(BM.DEFAULT_BBOX))
    ap.add_argument("--osm-date", default=None)
    ap.add_argument("--out", default="data/shibuya_osm.json")
    args = ap.parse_args([])
    assert tuple(args.bbox) == BM.DEFAULT_BBOX
    assert args.osm_date is None


def test_way_layer_levels_and_tunnel():
    assert BM.way_layer({"level": "-1"}) == -1
    assert BM.way_layer({"level": "-2;-1"}) == -2      # 最下層
    assert BM.way_layer({"layer": "0", "level": "-3"}) == 0   # layer 優先
    assert BM.way_layer({"tunnel": "yes"}) == -1
    assert BM.way_layer({"tunnel": "building_passage"}) == -1
    assert BM.way_layer({"bridge": "yes"}) == 1
    assert BM.way_layer({}) == 0


# --------------------------------------------------------------------------- #
# 2. 入口選択ロジック(main 優先)
# --------------------------------------------------------------------------- #
def test_main_entrance_overrides_building_entrance():
    data = BM.build(_mock_raw(), MOCK_BBOX, None)
    b = data["buildings"][0]
    # entrances に実データが入り、main が含まれる
    assert b.get("entrances"), "entrances が生成されていない"
    assert any(e["kind"] == "main" for e in b["entrances"])
    # 建物入口は重心最寄り(n1/n3)ではなく main 最寄りの n5 に置換されている
    assert b["entrance"] == "n5"
    assert data["meta"]["_stats"]["main_entrance_override"] == 1


def test_no_entrance_tag_keeps_centroid_logic():
    """entrance タグが無い建物は従来どおり重心最寄りノードのまま(entrances 無し)。"""
    raw = _mock_raw()
    # 建物ウェイから entrance ノード(id 14)を除去
    for el in raw["elements"]:
        if el.get("id") == 200:
            el["nodes"] = [10, 11, 12, 13, 10]
    raw["elements"] = [el for el in raw["elements"] if el.get("id") != 14]
    data = BM.build(raw, MOCK_BBOX, None)
    b = data["buildings"][0]
    node_ids = {n["id"] for n in data["nodes"]}
    assert "entrances" not in b
    # 重心最寄りの地上ノード(従来ロジック)。main 置換は起きない。
    assert b["entrance"] in node_ids
    assert data["meta"]["_stats"]["main_entrance_override"] == 0


# --------------------------------------------------------------------------- #
# 3. 地下エッジが layer 属性を持ちグラフに載る
# --------------------------------------------------------------------------- #
def test_underground_edge_has_layer_and_loads(tmp_path):
    data = BM.build(_mock_raw(), MOCK_BBOX, None)
    under = [e for e in data["edges"] if e.get("layer", 0) < 0]
    assert under, "地下エッジが生成されていない"
    assert all(e["layer"] < 0 for e in under)
    # CityMap にロードして graph に layer<0 のエッジが載ることを確認
    p = tmp_path / "mock_map.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    c = CityMap(p)
    assert c.underground_edges(), "CityMap.underground_edges が空"
    for u, v in c.underground_edges():
        assert c.graph.edges[u, v]["layer"] < 0
    assert c.underground_nodes()


# --------------------------------------------------------------------------- #
# 4. floorguide 拡張のスキーマ検証
# --------------------------------------------------------------------------- #
def test_floorguide_schema_and_connections():
    fg = json.loads((DATA / "floorguide_shibuya.json").read_text(encoding="utf-8"))
    assert fg["meta"]["sources"], "出典が空"
    buildings = fg["buildings"]
    assert buildings
    for b in buildings:
        assert b.get("match") and isinstance(b["match"], list)
        assert b.get("floors")
        for f in b["floors"]:
            assert set(f) >= {"f", "use", "label"}
            assert isinstance(f["f"], int)
        for c in b.get("connections", []):
            assert set(c) >= {"to_building", "via", "level"}
            assert isinstance(c["level"], int)
    # 駅・主要施設に地下階と接続が実在すること
    by_name = {b["match"][0]: b for b in buildings}
    station = by_name["渋谷駅"]
    assert any(f["f"] < 0 for f in station["floors"]), "駅に地下階が無い"
    assert station.get("connections"), "駅の接続が無い"
    ss = by_name["渋谷スクランブルスクエア"]
    assert any(f["f"] < 0 for f in ss["floors"])
    assert any(c["to_building"] == "渋谷駅" for c in ss["connections"])


# --------------------------------------------------------------------------- #
# 5. 生成済み wide 地図の基本整合
# --------------------------------------------------------------------------- #
WIDE = DATA / "shibuya_osm_wide.json"


@pytest.mark.skipif(not WIDE.exists(), reason="wide 地図が未生成")
def test_wide_map_integrity():
    raw = json.loads(WIDE.read_text(encoding="utf-8"))
    ids = [n["id"] for n in raw["nodes"]]
    assert len(ids) == len(set(ids)), "ノード id が重複している"
    # 原点(スクランブル交差点)不変: (0,0) 近傍にノードがある
    d0 = min(abs(n["x"]) + abs(n["y"]) for n in raw["nodes"])
    assert d0 < 20, f"原点近傍にノードが無い(原点がずれている?): {d0}"
    assert raw["meta"]["origin_latlon"] == list(BM.ORIGIN)
    assert raw["meta"]["version"] == 6
    # CityMap でロードでき、地下・入口が存在
    c = CityMap(WIDE)
    assert c.graph.number_of_nodes() > 2000
    assert c.underground_edges()
    assert any(c.building_entrances(b["id"]) for b in c.buildings)


# --------------------------------------------------------------------------- #
# 6. CityMap 後方互換アクセサ(旧地図では空を返す)
# --------------------------------------------------------------------------- #
def test_accessors_backward_compatible_on_core_map():
    """現行コア地図(v3、entrances 無し)でも新アクセサが例外を出さず空/既定を返す。"""
    c = CityMap(DATA / "shibuya_osm.json")
    # entrances は旧地図に無い → 全建物で []
    assert all(c.building_entrances(b["id"]) == [] for b in c.buildings[:50])
    assert c.has_real_entrances(c.buildings[0]["id"]) is False
    # 地下は v3 でも存在(layer<0)
    assert c.underground_nodes()
    assert c.underground_edges()
    # floorguide アクセサは名前で引ける(既定ファイルを遅延ロード)
    assert c.floor_guide("渋谷ヒカリエ") is not None
    conns = c.building_connections("渋谷スクランブルスクエア")
    assert conns and all("to_building" in x for x in conns)
    assert c.floor_guide("存在しない建物XYZ") is None
    assert c.building_connections("存在しない建物XYZ") == []
