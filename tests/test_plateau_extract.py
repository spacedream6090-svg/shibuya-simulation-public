"""scripts/plateau_extract.py の検証(実データ非依存・高速)。

- 合成 CityGML(2建物・LOD2 の直方体・GroundSurface 付き)を tmp に書いて抽出
  → 建物数・頂点/三角形数・gml_id・軸順 sniff を検証
- 局所接平面変換の数値検証(既知の緯度経度 → 期待 E/N ±0.5m)
- ear clipping: 凹五角形 → 三角形数 = n-2
- 3次メッシュコード: (35.6595,139.70062) → 53393596
- terrain: 合成平面 TIN → 2m 格子ラスタ化で定数勾配を ±0.2m 復元
- extras: 合成 ubld(uro:UndergroundBuilding・z<0 の箱)→ 抽出と z 範囲
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
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


# ------------------------------------------------------------- terrain ラスタ化
def test_rasterize_planar_tin_recovers_gradient():
    """平面 TIN(z=a*x+b*y+c)を格子点群として与え、2m 格子へ最近傍3頂点距離加重
    ラスタ化 → 内部セルで定数勾配を ±0.2m で復元することを確認。"""
    a, b, c = 0.02, -0.01, 30.0
    step = 4.0
    coords = np.arange(-40.0, 40.0 + step, step)
    pts = [(float(x), float(y), a * x + b * y + c) for x in coords for y in coords]
    P = np.asarray(pts, dtype=np.float64)

    x0, y0, cell = -30.0, -30.0, 2.0
    nx = int(round((30.0 - x0) / cell)) + 1
    ny = nx
    H = PE.rasterize_tin_to_grid(P, x0, y0, cell, nx, ny, bucket_m=12.0, k=3)

    assert np.isfinite(H).all()
    max_err = 0.0
    for j in range(2, ny - 2):
        for i in range(2, nx - 2):
            x = x0 + i * cell
            y = y0 + j * cell
            expect = a * x + b * y + c
            max_err = max(max_err, abs(H[j, i] - expect))
    assert max_err < 0.2, f"平面復元誤差が大きい: {max_err:.3f} m"


def test_rasterize_empty_points_returns_nan():
    H = PE.rasterize_tin_to_grid(np.zeros((0, 3)), 0.0, 0.0, 2.0, 5, 5)
    assert H.shape == (5, 5)
    assert np.isnan(H).all()


def _planar_tin(a, b, c, lo=-40.0, hi=40.0, step=4.0):
    """平面 z=a*x+b*y+c を張る三角形 TIN(格子を対角線で 2 分割)。"""
    coords = np.arange(lo, hi + step, step)
    pts, idx = [], {}
    for x in coords:
        for y in coords:
            idx[(float(x), float(y))] = len(pts)
            pts.append((float(x), float(y), a * x + b * y + c))
    tris = []
    for i in range(len(coords) - 1):
        for j in range(len(coords) - 1):
            x0, x1 = float(coords[i]), float(coords[i + 1])
            y0, y1 = float(coords[j]), float(coords[j + 1])
            tris.append((idx[(x0, y0)], idx[(x1, y0)], idx[(x1, y1)]))
            tris.append((idx[(x0, y0)], idx[(x1, y1)], idx[(x0, y1)]))
    return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def test_rasterize_barycentric_is_exact_on_tin_surface():
    """重心座標補間は TIN 面上の厳密値(平面 TIN なら誤差 0)。
    最近傍 IDW(旧手法)は同じ入力で必ず有限の誤差を残す = 置換の根拠。"""
    a, b, c = 0.02, -0.01, 30.0
    P, T = _planar_tin(a, b, c)
    x0, y0, cell = -30.0, -30.0, 2.0
    nx = ny = int(round((30.0 - x0) / cell)) + 1
    H = PE.rasterize_tin_to_grid(P, x0, y0, cell, nx, ny, tris=T)
    assert np.isfinite(H).all()
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    want = a * (x0 + ii * cell) + b * (y0 + jj * cell) + c
    assert np.abs(H - want).max() < 1e-9
    # 旧経路(tris 無し)は同一入力で誤差が残る
    H_old = PE.rasterize_tin_to_grid(P, x0, y0, cell, nx, ny)
    assert np.abs(H_old - want).max() > 1e-6


def test_rasterize_barycentric_reproduces_tin_step():
    """段差(隣り合う三角形の z が不連続)を平滑化せずそのまま返す。
    IDW は近傍平均なので段差を鈍らせる = 実地形の崖が消える/偽の傾斜が出る。"""
    # x<0 は z=0、x>0 は z=5 の 2 枚(x=0 で段差)
    P = np.array([[-10, -10, 0.0], [0, -10, 0.0], [0, 10, 0.0], [-10, 10, 0.0],
                  [0.001, -10, 5.0], [10, -10, 5.0], [10, 10, 5.0], [0.001, 10, 5.0]],
                 dtype=np.float64)
    T = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    x0, y0, cell, nx, ny = -8.0, -8.0, 2.0, 9, 9
    H = PE.rasterize_tin_to_grid(P, x0, y0, cell, nx, ny, tris=T)
    xs = x0 + np.arange(nx) * cell
    for i, x in enumerate(xs):
        col = H[:, i]
        assert np.isfinite(col).all()
        # x=0 は下側パッチの縁(z=0)。段差は中間値を作らず 0 か 5 のどちらか。
        assert np.allclose(col, 0.0 if x <= 0 else 5.0), f"x={x} col={col[:3]}"
    # IDW(旧手法)は段差の両側を平均して中間値を作る = 崖が鈍る
    H_old = PE.rasterize_tin_to_grid(P, x0, y0, cell, nx, ny)
    mid = H_old[(H_old > 0.5) & (H_old < 4.5)]
    assert mid.size > 0


def test_rasterize_barycentric_nan_outside_hull_then_nearest_fill():
    """TIN の外は nan(=どの三角形にも属さない)で返り、最近傍埋めで解消される。
    「どこまでが実測でどこからが外挿か」を数えられることが重要(silent fill 禁止)。"""
    P, T = _planar_tin(0.01, 0.0, 5.0, lo=-10.0, hi=10.0, step=5.0)
    x0, y0, cell, nx, ny = -20.0, -20.0, 5.0, 9, 9
    H = PE.rasterize_barycentric(P, T, x0, y0, cell, nx, ny)
    n_out = int((~np.isfinite(H)).sum())
    assert n_out > 0                                   # 外周は TIN 外
    assert np.isfinite(H[4, 4])                        # 中心(0,0)は TIN 内
    n_filled = PE._fill_nearest(H, P, x0, y0, cell, nx, ny)
    assert n_filled == n_out
    assert np.isfinite(H).all()


def test_collect_tin_triangles_keeps_faces(tmp_path):
    """DEM から三角形(接続関係)を保持して読む。1 頂点でも矩形内なら三角形ごと残す。"""
    tris = []

    def node(x, y, z):
        lat, lon = PE.local_to_latlon(x, y, LAT0, LON0)
        return (lat, lon, z)

    # 矩形(±30m)内の三角形 1 枚 + 1 頂点だけ内側の三角形 1 枚 + 完全に外側 1 枚
    tris.append((node(0, 0, 10.0), node(10, 0, 11.0), node(0, 10, 12.0)))
    tris.append((node(25, 25, 13.0), node(400, 25, 14.0), node(25, 400, 15.0)))
    tris.append((node(600, 600, 16.0), node(700, 600, 17.0), node(600, 700, 18.0)))
    tri_xml = ["<gml:Triangle><gml:exterior><gml:LinearRing><gml:posList>"
               + " ".join(f"{la} {lo} {z}" for (la, lo, z) in (t[0], t[1], t[2], t[0]))
               + "</gml:posList></gml:LinearRing></gml:exterior></gml:Triangle>"
               for t in tris]
    relief = ('<dem:ReliefFeature gml:id="r"><dem:reliefComponent>'
              '<dem:TINRelief gml:id="t"><dem:tin><gml:TriangulatedSurface>'
              '<gml:trianglePatches>' + "".join(tri_xml)
              + '</gml:trianglePatches></gml:TriangulatedSurface></dem:tin>'
              '</dem:TINRelief></dem:reliefComponent></dem:ReliefFeature>')
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    _write_gml(dem_dir, "533935_dem_6697_op.gml", [relief])

    P, T, n = PE.collect_tin_triangles([dem_dir / "533935_dem_6697_op.gml"],
                                       LAT0, LON0, (0, 1),
                                       [-30.0, -30.0, 30.0, 30.0], progress=False)
    assert T.shape[1] == 3
    assert len(T) == 2                      # 完全に外側の 1 枚だけ落ちる
    assert n == P.shape[0] == 6             # 各三角形の 3 頂点(全頂点を保持)
    assert T.max() < P.shape[0] and T.min() >= 0
    # 頂点 z が保存されている(内側三角形の 10/11/12 が揃う)
    assert {round(float(z), 3) for z in P[:, 2]} >= {10.0, 11.0, 12.0}
    # 後方互換ラッパは頂点だけ返す
    P2, n2 = PE.collect_tin_points([dem_dir / "533935_dem_6697_op.gml"],
                                   LAT0, LON0, (0, 1),
                                   [-30.0, -30.0, 30.0, 30.0], progress=False)
    assert n2 == n and P2.shape == P.shape


# ------------------------------------------------------------- extras (ubld)
def _ubld_xml(gml_id, surfaces):
    """uro:UndergroundBuilding + bldg:lod1Solid(CompositeSurface)の合成 XML。"""
    parts = [f'<uro:UndergroundBuilding gml:id="{gml_id}">',
             "<bldg:lod1Solid><gml:Solid><gml:exterior><gml:CompositeSurface>"]
    for _surf_type, pts in surfaces:
        txt = " ".join(f"{lat} {lon} {z}" for (lat, lon, z) in pts)
        parts.append(
            "<gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>"
            f"<gml:posList>{txt}</gml:posList>"
            "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>")
    parts.append("</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod1Solid>")
    parts.append("</uro:UndergroundBuilding>")
    return "".join(parts)


def _write_ubld(tmp_path, bodies):
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<core:CityModel '
        'xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
        'xmlns:uro="https://www.geospatial.jp/iur/uro/3.2" '
        'xmlns:gml="http://www.opengis.net/gml">')
    body = "".join(
        f"<core:cityObjectMember>{b}</core:cityObjectMember>" for b in bodies)
    p = tmp_path / "53393596_ubld_6697_op.gml"
    p.write_text(header + body + "</core:CityModel>", encoding="utf-8")
    return p


def test_extract_ubld_underground_box(tmp_path):
    # 原点近くの LOD1 直方体(z=[0,12] 生標高)。ground0=15 差引きで z<0 が正常。
    surf = _cuboid_surfaces(35.6600, 139.7010, 0.0002, 0.0002, 12.0)
    gml = _write_ubld(tmp_path, [_ubld_xml("ubld_test_A", surf)])

    res = PE.extract_features([gml], LAT0, LON0, {"UndergroundBuilding"},
                              clip_rect=None, axis=None)
    structs = res["structures"]
    assert len(structs) == 1
    assert res["axis"] == (0, 1)               # 緯度先の sniff
    s = structs[0]
    assert s["gml_id"] == "ubld_test_A"
    assert s["lod"] == "LOD1"                   # lod1Solid → LOD1
    assert s["n_tris"] == 12                    # 6面×2
    assert s["verts"].shape == (24, 3)
    assert res["counts"]["lod1"] == 1

    U_V, U_F, U_off = PE._assemble_mesh(structs, ground0=15.0)
    assert U_V.dtype == np.float32 and U_F.dtype == np.int32
    assert U_V.shape == (24, 3)
    assert U_F.shape == (12, 3)
    assert U_off.tolist() == [0, 12]           # 1 構造・offset 規約
    # ground0=15 差引きで全頂点 z<0(地下)
    assert U_V[:, 2].max() < 0.0
    assert float(U_V[:, 2].min()) == pytest.approx(-15.0, abs=1e-3)
    assert float(U_V[:, 2].max()) == pytest.approx(-3.0, abs=1e-3)


def test_extract_features_clip_excludes_far(tmp_path):
    near = _cuboid_surfaces(35.6600, 139.7010, 0.0002, 0.0002, 12.0)
    far = _cuboid_surfaces(35.6800, 139.7010, 0.0002, 0.0002, 12.0)
    gml = _write_ubld(tmp_path, [_ubld_xml("near", near), _ubld_xml("far", far)])
    clip = [-300.0, -300.0, 300.0, 300.0]
    res = PE.extract_features([gml], LAT0, LON0, {"UndergroundBuilding"},
                              clip_rect=clip, axis=(0, 1))
    assert [s["gml_id"] for s in res["structures"]] == ["near"]
    assert res["counts"]["skipped_clip"] == 1


# ------------------------------------------------------------- 出力ファイル契約
def _brid_xml(gml_id, surfaces):
    """brid:Bridge + boundedBy 各面 lod2MultiSurface(実データと同じ幾何経路)。"""
    parts = [f'<brid:Bridge gml:id="{gml_id}">']
    for surf_type, pts in surfaces:
        txt = " ".join(f"{lat} {lon} {z}" for (lat, lon, z) in pts)
        parts.append(
            f"<brid:boundedBy><brid:{surf_type}>"
            "<brid:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
            "<gml:Polygon><gml:exterior><gml:LinearRing>"
            f"<gml:posList>{txt}</gml:posList>"
            "</gml:LinearRing></gml:exterior></gml:Polygon>"
            "</gml:surfaceMember></gml:MultiSurface></brid:lod2MultiSurface>"
            f"</brid:{surf_type}></brid:boundedBy>")
    parts.append("</brid:Bridge>")
    return "".join(parts)


def _write_gml(tmp_path, name, bodies, extra_ns=""):
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<core:CityModel '
        'xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
        'xmlns:brid="http://www.opengis.net/citygml/bridge/2.0" '
        'xmlns:dem="http://www.opengis.net/citygml/relief/2.0" '
        'xmlns:uro="https://www.geospatial.jp/iur/uro/3.2" '
        f'{extra_ns}'
        'xmlns:gml="http://www.opengis.net/gml">')
    body = "".join(
        f"<core:cityObjectMember>{b}</core:cityObjectMember>" for b in bodies)
    p = tmp_path / name
    p.write_text(header + body + "</core:CityModel>", encoding="utf-8")
    return p


def _write_map(tmp_path, nodes):
    import json
    m = {"meta": {"crs": "local-m"}, "nodes": nodes,
         "edges": [], "buildings": [], "pois": []}
    p = tmp_path / "map.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def _write_index(out_dir, ground0):
    import json
    (out_dir / "plateau_index.json").write_text(
        json.dumps({"ground0": ground0}), encoding="utf-8")


def test_run_extras_npz_contract(tmp_path):
    """run_extras が契約キー ubld_V/ubld_F/brid_V/brid_F を V=float32(n,3)・
    F=int32(m,3)・カテゴリ内 0-based で書き出すことを検証(下流契約の固定)。"""
    citygml_dir = tmp_path / "bldg"      # bldg タイル無しでも mesh_codes は算出される
    ubld_dir = tmp_path / "ubld"
    brid_dir = tmp_path / "brid"
    dem_dir = tmp_path / "dem"
    out_dir = tmp_path / "out"
    for d in (citygml_dir, ubld_dir, brid_dir, dem_dir, out_dir):
        d.mkdir()
    _write_index(out_dir, 15.0)
    map_path = _write_map(tmp_path, [{"x": 0.0, "y": 0.0}, {"x": 40.0, "y": 40.0}])

    usurf = _cuboid_surfaces(35.6600, 139.7010, 0.0002, 0.0002, 12.0)  # z[0,12]
    _write_gml(ubld_dir, "53393596_ubld_6697_op.gml",
               [_ubld_xml("ubld_A", usurf)])
    bsurf = _cuboid_surfaces(35.6598, 139.7008, 0.0002, 0.0002, 20.0)  # z[0,20]
    _write_gml(brid_dir, "53393596_brid_6697_op.gml",
               [_brid_xml("brid_A", bsurf)])

    PE.run_extras(str(citygml_dir), str(map_path), str(out_dir), str(dem_dir),
                  str(ubld_dir), str(brid_dir), (LAT0, LON0), 150.0, 50.0)

    npz = np.load(out_dir / "extras.npz")
    assert {"ubld_V", "ubld_F", "brid_V", "brid_F"} <= set(npz.files)
    for vkey, fkey in (("ubld_V", "ubld_F"), ("brid_V", "brid_F")):
        V, F = npz[vkey], npz[fkey]
        assert V.dtype == np.float32 and V.shape[1] == 3
        assert F.dtype == np.int32 and F.shape[1] == 3
        # カテゴリ内 0-based: 全 F インデックスが該当 V の範囲内
        assert F.min() >= 0 and F.max() < V.shape[0]
    # ubld は地下(ground0=15 差引きで z<0)
    assert npz["ubld_V"][:, 2].max() < 0.0


def test_run_terrain_npz_contract(tmp_path):
    """run_terrain が契約キー heights を float32・単位メートル・(ny,nx) で書き出し、
    terrain.json に quant を含めない(x0/y0/cell_m/nx/ny/ground0/n_tin_points)ことを検証。"""
    import json
    citygml_dir = tmp_path / "bldg"
    dem_dir = tmp_path / "dem"
    out_dir = tmp_path / "out"
    for d in (citygml_dir, dem_dir, out_dir):
        d.mkdir()
    _write_index(out_dir, 15.0)
    map_path = _write_map(tmp_path, [{"x": 0.0, "y": 0.0}, {"x": 20.0, "y": 20.0}])

    # 原点周りの平面 DEM: elev = 15 + 0.02*x - 0.01*y(local-m を lat/lon へ戻して posList 化)
    tris = []
    step = 8.0
    xs = np.arange(-80.0, 120.0 + step, step)
    ys = np.arange(-80.0, 120.0 + step, step)

    def node(x, y):
        lat, lon = PE.local_to_latlon(x, y, LAT0, LON0)
        return (lat, lon, 15.0 + 0.02 * x - 0.01 * y)

    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            a = node(xs[xi], ys[yi])
            b = node(xs[xi + 1], ys[yi])
            c = node(xs[xi + 1], ys[yi + 1])
            d = node(xs[xi], ys[yi + 1])
            tris.append((a, b, c))
            tris.append((a, c, d))

    tri_xml = ["<gml:Triangle><gml:exterior><gml:LinearRing><gml:posList>"
               + " ".join(f"{la} {lo} {z}" for (la, lo, z) in (t[0], t[1], t[2], t[0]))
               + "</gml:posList></gml:LinearRing></gml:exterior></gml:Triangle>"
               for t in tris]
    relief = ('<dem:ReliefFeature gml:id="dem_r"><dem:reliefComponent>'
              '<dem:TINRelief gml:id="dem_t"><dem:tin><gml:TriangulatedSurface>'
              '<gml:trianglePatches>' + "".join(tri_xml)
              + '</gml:trianglePatches></gml:TriangulatedSurface></dem:tin>'
              '</dem:TINRelief></dem:reliefComponent></dem:ReliefFeature>')
    _write_gml(dem_dir, "533935_dem_6697_op.gml", [relief])

    PE.run_terrain(str(citygml_dir), str(map_path), str(out_dir), str(dem_dir),
                   (LAT0, LON0), 150.0, 50.0, cell_m=2.0, terrain_buffer=50.0)

    meta = json.loads((out_dir / "terrain.json").read_text(encoding="utf-8"))
    assert set(meta) >= {"x0", "y0", "cell_m", "nx", "ny", "ground0",
                         "n_tin_points"}
    assert "quant" not in meta
    assert meta["n_tin_points"] > 0

    npz = np.load(out_dir / "terrain.npz")
    assert "heights" in npz.files
    H = npz["heights"]
    assert H.dtype == np.float32
    assert H.shape == (meta["ny"], meta["nx"])
    # 原点セル(x0+i*cell≈0, y0+j*cell≈0)は z=elev-ground0≈0
    i0 = int(round((0.0 - meta["x0"]) / meta["cell_m"]))
    j0 = int(round((0.0 - meta["y0"]) / meta["cell_m"]))
    assert abs(float(H[j0, i0])) < 0.3
