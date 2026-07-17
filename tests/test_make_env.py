"""環境自動生成 v0(D2・scripts/make_env.py)のテスト。

検証(ネットワーク不使用=すべてモック/純関数):
- bbox パース(w,s,e,n → S,W,N,E・取り違え許容・不正拒否)
- 原点解決の3択(指定座標 / ランドマークPOI名 / bbox中心 / 既定)
- 地図の構造検証ロジック(連結性・要素数・ノードID一意)
- 交通の縮退宣言(徒歩の街)
- 合成 OSM データでの stage1 変換(build_map 汎用ビルダー経由で map.json 生成 → CityMap ロード)
- 渋谷名簿の機械的縮小流用(決定論)

build_map の汎用化(既定=渋谷値で従来同値)も間接的に確認する。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from society.world.map import CityMap

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ME = _load("make_env", "scripts/make_env.py")
BM = _load("build_map", "scripts/build_map.py")


# --------------------------------------------------------------- 合成 OSM
def _node(i, lat, lon, tags=None):
    e = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        e["tags"] = tags
    return e


def _way(i, nodes, tags):
    return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


def _mock_raw():
    """小さな連結グリッド + 名前つき建物 + カフェPOI + 名所ノード(原点PONモード用)。"""
    els = []
    # 東へ伸びる地上 footway n1..n5(下北沢近辺の任意座標)
    for k, lon in enumerate(
            [139.6660, 139.6670, 139.6680, 139.6690, 139.6700], 1):
        els.append(_node(k, 35.6613, lon))
    els.append(_way(101, [1, 2, 3, 4, 5], {"highway": "residential"}))
    # n3 を交差点に保つ縦の道
    els += [_node(6, 35.6603, 139.6680), _node(7, 35.6623, 139.6680)]
    els.append(_way(102, [6, 3, 7], {"highway": "footway"}))
    # 名前つき建物(retail)
    els += [_node(10, 35.6611, 139.6659), _node(11, 35.6611, 139.6701),
            _node(12, 35.6615, 139.6701), _node(13, 35.6615, 139.6659)]
    els.append(_way(200, [10, 11, 12, 13, 10],
                    {"building": "retail", "name": "テスト商店", "building:levels": "3"}))
    # カフェPOI(名前つきノード)
    els.append(_node(20, 35.6613, 139.6675, {"amenity": "cafe", "name": "純喫茶テスト"}))
    # 原点POI名モード用の名所ノード
    els.append(_node(21, 35.6613, 139.6685, {"tourism": "attraction", "name": "テスト広場"}))
    return {"elements": els}


# w,s,e,n の bbox(経度西,緯度南,経度東,緯度北)
MOCK_BBOX_WSEN = "139.6655,35.6600,139.6705,35.6626"
# build_map 順(S,W,N,E)
MOCK_BBOX = (35.6600, 139.6655, 35.6626, 139.6705)


# --------------------------------------------------------------- bbox パース
def test_parse_bbox_wsen_to_swne():
    assert ME.parse_bbox("139.66,35.65,139.67,35.66") == (35.65, 139.66, 35.66, 139.67)


def test_parse_bbox_space_and_swapped_tolerated():
    # 空白区切り + 西東/南北の取り違えを min/max で正規化
    assert ME.parse_bbox("139.67 35.66 139.66 35.65") == (35.65, 139.66, 35.66, 139.67)


def test_parse_bbox_rejects_wrong_count_and_degenerate():
    with pytest.raises(ValueError):
        ME.parse_bbox("139.66,35.65,139.67")            # 3 個
    with pytest.raises(ValueError):
        ME.parse_bbox("139.66,35.65,139.66,35.66")      # 経度幅0=退化
    with pytest.raises(ValueError):
        ME.parse_bbox("a,b,c,d")                        # 数でない


# --------------------------------------------------------------- 原点解決
def test_bbox_center():
    assert BM.bbox_center(MOCK_BBOX) == pytest.approx((35.6613, 139.668))


def test_find_poi_latlon_node_and_missing():
    raw = _mock_raw()
    assert BM.find_poi_latlon(raw, "純喫茶テスト") == pytest.approx((35.6613, 139.6675))
    assert BM.find_poi_latlon(raw, "存在しない") is None


def test_resolve_origin_three_choices_and_default():
    raw = _mock_raw()
    # (1) 指定座標
    o, mode = BM.resolve_origin(MOCK_BBOX, raw, latlon=(35.7, 139.7))
    assert o == (35.7, 139.7) and mode == "latlon"
    # (2) ランドマークPOI名
    o, mode = BM.resolve_origin(MOCK_BBOX, raw, poi="テスト広場")
    assert o == pytest.approx((35.6613, 139.6685)) and mode.startswith("poi:")
    # (3) bbox 中心
    o, mode = BM.resolve_origin(MOCK_BBOX, raw, bbox_center_mode=True)
    assert o == pytest.approx((35.6613, 139.668)) and mode == "bbox-center"
    # 既定=渋谷スクランブル(現行不変)
    o, mode = BM.resolve_origin(MOCK_BBOX, raw)
    assert o == BM.ORIGIN and "default" in mode


def test_resolve_origin_poi_missing_raises():
    with pytest.raises(RuntimeError):
        BM.resolve_origin(MOCK_BBOX, _mock_raw(), poi="無い名所")


# --------------------------------------------------------------- 構造検証
def test_validate_map_connected():
    data = BM.build(_mock_raw(), MOCK_BBOX, None, origin=BM.bbox_center(MOCK_BBOX),
                    landmarks=[], landmark_name_kws=(), hachiko_fallback=None)
    rep = ME.validate_map(data)
    assert rep["nodes"] > 0 and rep["edges"] > 0
    assert rep["connected"] is True and rep["n_components"] == 1
    assert rep["largest_frac"] == 1.0
    assert rep["unique_node_ids"] is True
    assert rep["ok"] is True


def test_validate_map_disconnected_flagged():
    data = {"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
            "edges": [{"u": "a", "v": "b"}, {"u": "c", "v": "d"}],
            "buildings": [], "pois": []}
    rep = ME.validate_map(data)
    assert rep["connected"] is False
    assert rep["n_components"] == 2
    assert rep["ok"] is False


def test_validate_map_duplicate_ids_flagged():
    data = {"nodes": [{"id": "a"}, {"id": "a"}],
            "edges": [{"u": "a", "v": "a"}], "buildings": [], "pois": []}
    rep = ME.validate_map(data)
    assert rep["unique_node_ids"] is False and rep["ok"] is False


# --------------------------------------------------------------- 交通の縮退
def test_transit_status_walking_when_no_key_or_targets():
    st = ME.transit_status(key_present=False, targets_defined=False)
    assert st["available"] is False and st["mode"] == "walking"
    # キーはあるが路線未定義 → やはり徒歩の街(v0)
    st2 = ME.transit_status(key_present=True, targets_defined=False)
    assert st2["available"] is False and st2["mode"] == "walking"


def test_transit_status_odpt_when_key_and_targets():
    st = ME.transit_status(key_present=True, targets_defined=True)
    assert st["available"] is True and st["mode"] == "odpt"


# --------------------------------------------------------------- stage1 合成変換
def test_stage1_synthetic_builds_and_loads(tmp_path):
    raw = _mock_raw()
    rep = ME.run_stage1(tmp_path, "testtown", "テストタウン", MOCK_BBOX, raw,
                        origin_bbox_center=True)
    # map.json が生成され、連結・構造検証 OK
    assert rep["ok"] is True and rep["connected"] is True
    assert rep["origin_mode"] == "bbox-center"
    map_path = tmp_path / "map.json"
    assert map_path.exists()
    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert data["meta"]["name"] == "testtown_osm"
    assert data["meta"]["origin_latlon"] == list(BM.bbox_center(MOCK_BBOX))
    # ハチ公フォールバックは別の街では無効(landmark 名所は入らない)
    assert data["meta"]["_stats"]["hachiko_source"] == "none"
    # CityMap がロードでき、グラフが連結(ノード連結性の成立)
    c = CityMap(map_path)
    assert c.graph.number_of_nodes() > 0
    import networkx as nx
    assert nx.is_connected(c.graph)


# --------------------------------------------------------------- 名簿の縮小流用
def test_reduce_roster_deterministic_subset():
    src = {"meta": {"generator": "x"},
           "personas": [{"name": f"p{i}", "age": 20 + i} for i in range(50)]}
    a = ME.reduce_roster(src, 10, seed=42)
    b = ME.reduce_roster(src, 10, seed=42)
    assert len(a["personas"]) == 10
    assert [p["name"] for p in a["personas"]] == [p["name"] for p in b["personas"]]
    assert a["meta"]["reduced_n"] == 10
    # n が母数超なら全件
    c = ME.reduce_roster(src, 999, seed=1)
    assert len(c["personas"]) == 50
