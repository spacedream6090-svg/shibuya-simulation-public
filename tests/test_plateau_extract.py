"""scripts/plateau_extract.py の検証(実データ非依存・高速)。

- 合成 CityGML(2建物・LOD2 の直方体・GroundSurface 付き)を tmp に書いて抽出
  → 建物数・頂点/三角形数・gml_id・軸順 sniff を検証
- 局所接平面変換の数値検証(既知の緯度経度 → 期待 E/N ±0.5m)
- ear clipping: 凹五角形 → 三角形数 = n-2
- 3次メッシュコード: (35.6595,139.70062) → 53393596
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plateau_extract", REPO_ROOT / "scripts" / "plateau_extract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PE = _load_module()

LAT0, LON0 = 35.65950, 139.70062


# ------------------------------------------------------------- 合成 CityGML
def _cuboid_surfaces(lat, lon, dlat, dlon, height):
    """(lat,lon) を南西角とする直方体の6面ポリゴンを (surface_type, [(lat,lon,z)..]) で返す。
    各リングは閉じる(先頭=末尾)。"""
    c1 = (lat, lon)
    c2 = (lat, lon + dlon)
    c3 = (lat + dlat, lon + dlon)
    c4 = (lat + dlat, lon)
    corners = [c1, c2, c3, c4]

    def ring(z_by_corner):
        r = [(cc[0], cc[1], z) for cc, z in z_by_corner]
        r.append(r[0])
        return r

    ground = ("GroundSurface", ring([(c1, 0.0), (c2, 0.0), (c3, 0.0), (c4, 0.0)]))
    roof = ("RoofSurface", ring([(c1, height), (c2, height), (c3, height), (c4, height)]))
    walls = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        walls.append(("WallSurface",
                      ring([(a, 0.0), (b, 0.0), (b, height), (a, height)])))
    return [ground, roof] + walls


def _poslist_xml(pts):
    txt = " ".join(f"{lat} {lon} {z}" for (lat, lon, z) in pts)
    return (
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon><gml:exterior><gml:LinearRing>"
        f"<gml:posList>{txt}</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon>"
        "</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>")


def _building_xml(gml_id, surfaces):
    parts = [f'<bldg:Building gml:id="{gml_id}">']
    for surf_type, pts in surfaces:
        parts.append(
            f"<bldg:boundedBy><bldg:{surf_type}>"
            f"{_poslist_xml(pts)}"
            f"</bldg:{surf_type}></bldg:boundedBy>")
    parts.append("</bldg:Building>")
    return "".join(parts)


def _write_synthetic(tmp_path, buildings):
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<core:CityModel '
        'xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
        'xmlns:gml="http://www.opengis.net/gml">')
    body = "".join(
        f"<core:cityObjectMember>{b}</core:cityObjectMember>" for b in buildings)
    xml = header + body + "</core:CityModel>"
    p = tmp_path / "53393596_bldg_6697_op.gml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_extract_two_cuboids(tmp_path):
    # 建物A: 原点近く。建物B: 少し北東。どちらも LOD2 直方体(6面)。
    surfA = _cuboid_surfaces(35.6600, 139.7010, 0.0002, 0.0002, 12.0)
    surfB = _cuboid_surfaces(35.6605, 139.7020, 0.0002, 0.0002, 20.0)
    bxml = [_building_xml("bldg_test_A", surfA), _building_xml("bldg_test_B", surfB)]
    gml = _write_synthetic(tmp_path, bxml)

    res = PE.extract_buildings([gml], LAT0, LON0, clip_rect=None)
    bs = res["buildings"]
    assert len(bs) == 2
    # 軸順 sniff: 緯度が先(EPSG:6697)
    assert res["axis"] == (0, 1)
    # gml_id 一致
    assert {b["gml_id"] for b in bs} == {"bldg_test_A", "bldg_test_B"}
    for b in bs:
        # 直方体 6 面 × 2三角形 = 12
        assert b["n_tris"] == 12
        # 各面 4 頂点(閉環の重複は除去)× 6 面 = 24
        assert b["verts"].shape == (24, 3)
        assert b["lod"] == "LOD2"
        # footprint は GroundSurface(4頂点)由来
        assert len(b["footprint"]) == 4
    # 穴・スキップは無い
    assert res["counts"]["holes_ignored"] == 0
    assert res["counts"]["skipped_no_geom"] == 0
    assert res["counts"]["lod2"] == 2

    # 高さ: A の頂点 z 範囲は [0,12]
    bA = next(b for b in bs if b["gml_id"] == "bldg_test_A")
    assert bA["zmin"] == pytest.approx(0.0, abs=1e-6)
    assert bA["zmax"] == pytest.approx(12.0, abs=1e-6)


def test_clip_excludes_far_building(tmp_path):
    # A は原点近く、B は遥か遠く(緯度 +0.02度 ≒ 2.2km 北)。clip で B が落ちる。
    surfA = _cuboid_surfaces(35.6600, 139.7010, 0.0002, 0.0002, 12.0)
    surfB = _cuboid_surfaces(35.6800, 139.7010, 0.0002, 0.0002, 12.0)
    gml = _write_synthetic(tmp_path,
                           [_building_xml("A", surfA), _building_xml("B", surfB)])
    clip = [-300.0, -300.0, 300.0, 300.0]  # local-m ±300m
    res = PE.extract_buildings([gml], LAT0, LON0, clip_rect=clip)
    assert len(res["buildings"]) == 1
    assert res["buildings"][0]["gml_id"] == "A"
    assert res["counts"]["skipped_clip"] == 1


# ------------------------------------------------------------- 座標変換
def test_latlon_to_local_known():
    e, n = PE.latlon_to_local(35.66, 139.705, LAT0, LON0)
    # 独立計算値(スクリプトと同一公式): E≈396.16m, N≈55.57m
    assert e == pytest.approx(396.16, abs=0.5)
    assert n == pytest.approx(55.57, abs=0.5)
    # 原点は (0,0)
    e0, n0 = PE.latlon_to_local(LAT0, LON0, LAT0, LON0)
    assert e0 == pytest.approx(0.0, abs=1e-6)
    assert n0 == pytest.approx(0.0, abs=1e-6)


def test_local_latlon_roundtrip():
    lat, lon = PE.local_to_latlon(396.16, 55.57, LAT0, LON0)
    e, n = PE.latlon_to_local(lat, lon, LAT0, LON0)
    assert e == pytest.approx(396.16, abs=0.01)
    assert n == pytest.approx(55.57, abs=0.01)


def test_sniff_axis_order():
    assert PE.sniff_axis_order(35.66, 139.70) == (0, 1)   # 緯度先
    assert PE.sniff_axis_order(139.70, 35.66) == (1, 0)   # 経度先


# ------------------------------------------------------------- ear clipping
def test_triangulate_3d_concave_pentagon():
    # 凹五角形((2,1) が reflex)を z=0 平面に置く → n-2 = 3 三角形
    ring = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 1, 0), (0, 4, 0)]
    dedup, tris = PE.triangulate_3d(ring)
    assert len(dedup) == 5
    assert len(tris) == 3
    # 面積保存: 三角形の総和 = 元多角形の面積
    total = 0.0
    for (i, j, k) in tris:
        a, b, c = dedup[i], dedup[j], dedup[k]
        total += abs((b[0] - a[0]) * (c[1] - a[1])
                     - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
    # shoelace 面積
    area = abs(sum(ring[m][0] * ring[(m + 1) % 5][1]
                   - ring[(m + 1) % 5][0] * ring[m][1] for m in range(5))) / 2.0
    assert total == pytest.approx(area)


def test_triangulate_3d_vertical_wall():
    # 垂直な壁(xy に射影すると退化)でも三角形化できる
    ring = [(0, 0, 0), (10, 0, 0), (10, 0, 5), (0, 0, 5)]
    dedup, tris = PE.triangulate_3d(ring)
    assert len(tris) == 2


# ------------------------------------------------------------- メッシュコード
def test_meshcode3_origin():
    assert PE.meshcode3(35.6595, 139.70062) == "53393596"


def test_meshcode3_neighbors():
    # 原点の1つ西/南の3次メッシュ
    assert PE.meshcode3(35.6595, 139.6900) == "53393595"
    assert PE.meshcode3(35.6500, 139.70062) == "53393586"


def test_meshes_for_bbox_center():
    # 中心1km圏の bbox → 4枚(85/86/95/96)を含む
    codes = PE.meshes_for_bbox(35.65465, 139.69334, 35.66385, 139.70766)
    assert {"53393585", "53393586", "53393595", "53393596"} <= codes


# ------------------------------------------------------------- 巻き向きの復元
def _tri_normal(p, tri):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = p[tri[0]], p[tri[1]], p[tri[2]]
    u = (bx - ax, by - ay, bz - az)
    v = (cx - ax, cy - ay, cz - az)
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


@pytest.mark.parametrize("ring,expect", [
    # 上向き水平面(CCW を上から見た巻き)→ 法線 +z
    ([(0, 0, 5), (2, 0, 5), (2, 2, 5), (0, 2, 5)], (0, 0, 1)),
    # 下向き水平面(同じ頂点の逆順=床)→ 法線 -z が保存されること
    ([(0, 2, 5), (2, 2, 5), (2, 0, 5), (0, 0, 5)], (0, 0, -1)),
    # 南向きの壁(-y 向き)→ 投影軸の符号に関わらず -y が保存されること
    ([(0, 0, 0), (2, 0, 0), (2, 0, 3), (0, 0, 3)], (0, -1, 0)),
    # 北向きの壁(+y 向き)
    ([(2, 1, 0), (0, 1, 0), (0, 1, 3), (2, 1, 3)], (0, 1, 0)),
])
def test_triangulate_3d_preserves_winding(ring, expect):
    """三角形の表裏が元リング(CityGML の外向き巻き)と一致する
    (背面カリングで壁が消える崩れの回帰テスト)。"""
    p, tris = PE.triangulate_3d(ring)
    assert tris, "三角形化に失敗"
    for t in tris:
        n = _tri_normal(p, t)
        dot = sum(a * b for a, b in zip(n, expect))
        assert dot > 0, f"面が反転: tri={t} normal={n} expect={expect}"
