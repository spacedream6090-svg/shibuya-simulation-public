"""scripts/match_plateau.py の検証(実データ非依存・合成 footprint のみ)。

- ラスタ IoU の数値検証(解析解のある矩形重なり)
- 凹形状(L 字)のラスタ塗りが凸包でなく実形状であること
- match(): 同一ペア採用 / 半分ずらし棄却 / 遠距離候補外 / 競合の貪欲解決
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "match_plateau", REPO_ROOT / "scripts" / "match_plateau.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MP = _load_mod()


def _square(x0, y0, w, h):
    """左下 (x0,y0)・幅 w・高さ h の軸平行矩形 footprint。"""
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


# ----------------------------------------------------------------- ラスタ IoU 単体
def test_raster_iou_identical_is_one():
    sq = _square(0, 0, 10, 10)
    assert MP.raster_iou(sq, sq) == pytest.approx(1.0)


def test_raster_iou_half_shift_is_third():
    # 10x10 と、x 方向に 5 ずらした 10x10。重なり 5x10=50、和 150 → IoU=1/3。
    a = _square(0, 0, 10, 10)
    b = _square(5, 0, 10, 10)
    assert MP.raster_iou(a, b) == pytest.approx(1.0 / 3.0, abs=0.05)


def test_raster_iou_analytic_rectangles():
    # 10x10 と 6x10 が半分重なる: B=[7,0]-[13,10]、重なり x[7,10]=3 幅 ×10 = 30。
    # B 面積 60 の半分が重なる。和 = 100 + 60 - 30 = 130 → IoU = 30/130。
    a = _square(0, 0, 10, 10)
    b = _square(7, 0, 6, 10)
    assert MP.raster_iou(a, b) == pytest.approx(30.0 / 130.0, abs=0.05)


# ----------------------------------------------------------------- 凹形状(L 字)
def test_concave_L_notch_is_outside():
    # L 字: 凹(reflex)頂点 (2,2)。右上 (x>2,y>2) の切り欠きは形状外。
    L = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    # 切り欠き中央 (3,3) は「凸包なら内側」だが実形状では外側でなければならない
    notch = MP.points_in_polygon([[3.0, 3.0]], L)
    assert bool(notch[0]) is False
    # 実形状の内側の点は True(下辺の帯 (3,1) と 縦帯 (1,3))
    inside = MP.points_in_polygon([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]], L)
    assert all(bool(v) for v in inside)


def test_concave_L_iou_uses_real_shape_not_hull():
    # L 字どうしの IoU は 1.0(実形状塗り)。凸包で塗ると切り欠きを含み 1.0 にならない筈。
    L = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    assert MP.raster_iou(L, L) == pytest.approx(1.0)


# ----------------------------------------------------------------- match(): 採用/棄却
def test_match_identical_pair_adopted():
    osm = [{"id": "b1", "footprint": _square(0, 0, 10, 10)}]
    pla = [{"gml_id": "g1", "footprint": _square(0, 0, 10, 10)}]
    res = MP.match(osm, pla)
    assert "b1" in res["matches"]
    assert res["matches"]["b1"]["gml_id"] == "g1"
    assert res["matches"]["b1"]["iou"] == pytest.approx(1.0)
    assert res["matches"]["b1"]["dist_m"] == pytest.approx(0.0, abs=1e-6)
    assert res["unmatched_osm"] == []
    assert res["unmatched_plateau_count"] == 0


def test_match_half_shift_rejected():
    # IoU=1/3 < 0.4 → 不採用。候補内(重心距離 5m)だが棄却され unmatched に落ちる。
    osm = [{"id": "b1", "footprint": _square(0, 0, 10, 10)}]
    pla = [{"gml_id": "g1", "footprint": _square(5, 0, 10, 10)}]
    res = MP.match(osm, pla)
    assert res["matches"] == {}
    assert "b1" in res["unmatched_osm"]
    assert res["unmatched_plateau_count"] == 1


def test_match_far_pair_out_of_candidates():
    # 重心距離 30m(>25m の knn 半径)→ そもそも候補外 → 両者 unmatched。
    osm = [{"id": "b1", "footprint": _square(0, 0, 10, 10)}]
    pla = [{"gml_id": "g1", "footprint": _square(30, 0, 10, 10)}]
    res = MP.match(osm, pla)
    assert res["matches"] == {}
    assert res["unmatched_osm"] == ["b1"]
    assert res["unmatched_plateau_count"] == 1


def test_match_competition_greedy():
    # PLATEAU P1 に OSM O1(IoU=1.0)と O2(IoU≈0.82)が競合。
    # O1 が P1 を獲得。O2 は次候補 P2(IoU≈0.54)へ回る。
    P1 = _square(0, 0, 10, 10)          # 重心 (5,5)
    P2 = _square(4, 0, 10, 10)          # 重心 (9,5)
    O1 = _square(0, 0, 10, 10)          # 重心 (5,5) → P1 と同一(IoU=1.0)
    O2 = _square(1, 0, 10, 10)          # 重心 (6,5) → P1 と 0.818 / P2 と 0.538
    osm = [{"id": "o1", "footprint": O1}, {"id": "o2", "footprint": O2}]
    pla = [{"gml_id": "p1", "footprint": P1}, {"gml_id": "p2", "footprint": P2}]
    res = MP.match(osm, pla)
    assert res["matches"]["o1"]["gml_id"] == "p1"     # IoU 大が P1 を獲得
    assert res["matches"]["o2"]["gml_id"] == "p2"     # 敗者は次候補 P2 へ
    assert res["unmatched_osm"] == []
    assert res["unmatched_plateau_count"] == 0
    # 競合の勝者は敗者より高 IoU
    assert res["matches"]["o1"]["iou"] > res["matches"]["o2"]["iou"]


def test_match_output_schema_and_params():
    osm = [{"id": "b1", "footprint": _square(0, 0, 10, 10)}]
    pla = [{"gml_id": "g1", "footprint": _square(0, 0, 10, 10)}]
    res = MP.match(osm, pla, knn_radius_m=25, iou_min=0.4, grid_m=0.5)
    assert set(res.keys()) == {
        "matches", "unmatched_osm", "unmatched_plateau_count", "params"}
    assert res["params"] == {"knn_radius_m": 25, "iou_min": 0.4, "grid_m": 0.5}


def test_match_empty_inputs():
    # 空入力でも落ちない(PLATEAU 未生成→合成テストのみの経路を守る)。
    res = MP.match([], [])
    assert res["matches"] == {}
    assert res["unmatched_osm"] == []
    assert res["unmatched_plateau_count"] == 0
