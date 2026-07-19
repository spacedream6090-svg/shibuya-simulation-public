"""PLATEAU CityGML(渋谷区2025)→ シミュ local-m 中間ファイル抽出。

下流(export_3d --plateau / 照合)が読む一次データを生成する。実 LLM 非依存・純粋な
ジオメトリ変換パイプライン。pandas/duckdb は使わず numpy+pyarrow+stdlib のみ。

処理概要:
  1. data/shibuya_osm.json の全座標(local-m)から緯度経度 bbox を逆算(+150m バッファ)。
     交差する3次メッシュコードを計算し、該当タイルの *.gml だけを読む。
  2. iterparse + elem.clear() で各 bldg:Building を streaming 抽出。名前空間は localname 照合
     (CityGML 2.x/3.x 両対応)。LOD2 の全ポリゴン、無ければ LOD1、それも無ければスキップ。
     footprint は bldg:GroundSurface 優先、無ければ最下層ポリゴンの xy 投影で代用。
  3. 局所接平面近似で local-m へ変換。E=(lon-lon0)*111320*cos(lat0), N=(lat-lat0)*111132.9。
  4. udx/dem の該当タイルから原点半径50m内標高の中央値を ground0 に(不可なら建物頂点zの1%分位)。
  5. footprint 重心が bbox+150m の外ならクリップ。
  6. 各ポリゴン(外環のみ・穴は無視して件数記録)を ear clipping で三角形化(export_3d を再利用)。
  7. data/plateau/plateau_mesh.npz + plateau_index.json を書き出す(緯度経度は残さず local-m のみ)。

使い方:
  python scripts/plateau_extract.py [--citygml-dir DIR] [--map data/shibuya_osm.json]
                                    [--out-dir data/plateau] [--dem-dir DIR]
                                    [--buffer 150] [--no-dem]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 局所接平面近似の定数(EPSG:6697 → local-m)
M_PER_DEG_LAT = 111132.9
M_PER_DEG_LON_EQ = 111320.0
DEFAULT_ORIGIN = (35.65950, 139.70062)  # (lat, lon) スクランブル交差点原点
DEFAULT_CITYGML_DIR = Path(
    r"C:\Users\塚本翔太\Desktop\13113_shibuya-ku_pref_2025_citygml_1_op\udx\bldg")

GML_ID = "{http://www.opengis.net/gml}id"

# LOD / 半題面 の localname 集合(名前空間非依存で照合)
LOD2_TAGS = {"lod2Solid", "lod2MultiSurface", "lod2Geometry"}
LOD1_TAGS = {"lod1Solid", "lod1MultiSurface", "lod1Geometry"}
LOD0_TAGS = {"lod0RoofEdge", "lod0FootPrint", "lod0Geometry", "lod0MultiSurface"}
SURFACE_TAGS = {"GroundSurface", "RoofSurface", "WallSurface",
                "OuterCeilingSurface", "OuterFloorSurface", "ClosureSurface"}


# --------------------------------------------------------------- ear clipping 再利用
def _load_triangulate():
    """scripts/export_3d.py の自作 ear clipping(凹対応)を import。
    scripts が package でないため importlib で直接ロードする(tests/test_export3d.py と同方式)。"""
    import importlib.util
    p = REPO_ROOT / "scripts" / "export_3d.py"
    spec = importlib.util.spec_from_file_location("export_3d", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triangulate


try:
    _triangulate2d = _load_triangulate()
except Exception:  # pragma: no cover - フォールバック(export_3d を読めない環境)
    # 出典: scripts/export_3d.py triangulate()(凹対応 ear clipping)をコピー。
    def _signed_area(ring):
        a = 0.0
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return a * 0.5

    def _point_in_tri(p, a, b, c):
        (px, py), (ax, ay), (bx, by), (cx, cy) = p, a, b, c
        d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
        d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
        d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def _triangulate2d(ring):
        pts = [tuple(p) for p in ring]
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts = pts[:-1]
        n = len(pts)
        if n < 3:
            return []
        idx = list(range(n))
        if _signed_area(pts) < 0:
            idx.reverse()
        tris = []
        guard = 0
        limit = 4 * n * n + 16
        while len(idx) > 3 and guard < limit:
            guard += 1
            m = len(idx)
            ear = False
            for k in range(m):
                i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
                a, b, c = pts[i0], pts[i1], pts[i2]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if cross <= 1e-12:
                    continue
                clean = True
                for j in idx:
                    if j in (i0, i1, i2):
                        continue
                    if _point_in_tri(pts[j], a, b, c):
                        clean = False
                        break
                if clean:
                    tris.append((i0, i1, i2))
                    idx.pop(k)
                    ear = True
                    break
            if not ear:
                break
        if len(idx) == 3:
            tris.append((idx[0], idx[1], idx[2]))
        return tris


def _newell_normal(pts):
    """3D 平面ポリゴンの法線(Newell 法)。pts は [(x,y,z),...]。"""
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def triangulate_3d(pts):
    """任意向きの3D平面ポリゴンを、法線の優越軸を落とした2D投影で ear clipping。
    戻り値: (dedup_pts[(x,y,z),...], tris[(i,j,k),...])。tris は dedup_pts への index。"""
    p = [tuple(v) for v in pts]
    if len(p) > 1 and p[0] == p[-1]:
        p = p[:-1]
    if len(p) < 3:
        return p, []
    nx, ny, nz = _newell_normal(p)
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        ring2 = [(x, y) for (x, y, _z) in p]
    elif ay >= ax:
        ring2 = [(x, z) for (x, _y, z) in p]
    else:
        ring2 = [(y, z) for (_x, y, z) in p]
    tris = _triangulate2d(ring2)
    # 元リング(CityGML の外向き巻き)と三角形の表裏を一致させる。_triangulate2d は
    # 投影面で CCW に正規化するため、投影軸の符号次第で表裏が反転し「正軸向き」に
    # 揃ってしまう(→建物の約半数が内向き殻=背面カリングで壁が消える崩れの原因)。
    for (i, j, k) in tris:
        (ax0, ay0, az0), (bx, by, bz), (cx, cy, cz) = p[i], p[j], p[k]
        ux, uy, uz = bx - ax0, by - ay0, bz - az0
        vx, vy, vz = cx - ax0, cy - ay0, cz - az0
        d = ((uy * vz - uz * vy) * nx + (uz * vx - ux * vz) * ny
             + (ux * vy - uy * vx) * nz)
        if d != 0.0:
            if d < 0.0:
                tris = [(a, c, b) for (a, b, c) in tris]
            break
    return p, tris


# --------------------------------------------------------------- 座標変換 / メッシュ
def latlon_to_local(lat, lon, lat0, lon0):
    """局所接平面近似(EPSG:6697 → local-m)。x=E(east), y=N(north)。"""
    e = (lon - lon0) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    n = (lat - lat0) * M_PER_DEG_LAT
    return e, n


def local_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / M_PER_DEG_LAT
    lon = lon0 + x / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
    return lat, lon


def meshcode3(lat, lon):
    """緯度経度 → 3次地域メッシュコード(8桁)。JIS X 0410。"""
    p = int(lat * 1.5)
    u = int(lon) - 100
    lat_min = lat * 60.0 - p * 40.0
    q = int(lat_min // 5.0)
    lon_min = (lon - (u + 100)) * 60.0
    v = int(lon_min // 7.5)
    lat_min2 = lat_min - q * 5.0
    r = int(lat_min2 // 0.5)
    lon_min2 = lon_min - v * 7.5
    s = int(lon_min2 // 0.75)
    return f"{p}{u}{q}{v}{r}{s}"


def meshes_for_bbox(latmin, lonmin, latmax, lonmax):
    """緯度経度 bbox と交差する3次メッシュコード集合。
    3次メッシュ格子(緯30秒=1/120度・経45秒=1/80度)のセル中心で網羅走査。"""
    codes = set()
    ilat0 = int(math.floor(latmin * 120))
    ilat1 = int(math.floor(latmax * 120))
    ilon0 = int(math.floor(lonmin * 80))
    ilon1 = int(math.floor(lonmax * 80))
    for ila in range(ilat0, ilat1 + 1):
        for ilo in range(ilon0, ilon1 + 1):
            lat = (ila + 0.5) / 120.0
            lon = (ilo + 0.5) / 80.0
            codes.add(meshcode3(lat, lon))
    return codes


def sniff_axis_order(a, b):
    """先頭 posList の (col0, col1) から緯度列/経度列を自動判定。
    戻り値 (lat_idx, lon_idx)。EPSG:6697 は (0,1)=「緯度 経度」の想定。"""
    if 34.0 <= a <= 37.0 and 138.0 <= b <= 141.0:
        return (0, 1)
    if 138.0 <= a <= 141.0 and 34.0 <= b <= 37.0:
        return (1, 0)
    # 判別不能: 既定(緯度先)にフォールバック
    return (0, 1)


# --------------------------------------------------------------- CityGML streaming
def _ln(tag):
    """名前空間を落とした localname。"""
    return tag.rsplit('}', 1)[-1]


def _parse_poslist(text):
    """gml:posList テキスト → [(v0, v1, z), ...](srsDimension=3 前提)。"""
    parts = text.split()
    out = []
    for i in range(0, len(parts) - 2, 3):
        out.append((float(parts[i]), float(parts[i + 1]), float(parts[i + 2])))
    return out


def stream_building_geoms(path):
    """1 CityGML ファイルを iterparse で streaming し、建物ごとの生ジオメトリを yield。
    メモリ節約: 各要素を end で clear、トップレベル member 終端で root.clear()。
    yield: {gml_id, lod2:[(surface, pts)], lod1:[(surface, pts)], n_holes}。
    pts は生の (col0, col1, z)(軸順は呼び出し側で解決)。"""
    ctx = ET.iterparse(str(path), events=("start", "end"))
    stack = []
    cur = None
    root = None
    for ev, el in ctx:
        name = _ln(el.tag)
        if ev == "start":
            if root is None:
                root = el
            stack.append(name)
            if name == "Building":
                cur = {"gml_id": el.get(GML_ID), "lod2": [], "lod1": [], "n_holes": 0}
            continue
        # ev == "end"
        if name == "posList":
            if cur is not None and el.text:
                lod = None
                for t in reversed(stack):
                    if t in LOD2_TAGS:
                        lod = "lod2"
                        break
                    if t in LOD1_TAGS:
                        lod = "lod1"
                        break
                    if t in LOD0_TAGS:
                        lod = "lod0"
                        break
                if lod in ("lod2", "lod1"):
                    if "interior" in stack:
                        cur["n_holes"] += 1
                    else:
                        surf = None
                        for t in stack:
                            if t in SURFACE_TAGS:
                                surf = t
                                break
                        pts = _parse_poslist(el.text)
                        if len(pts) >= 3:
                            cur[lod].append((surf, pts))
            stack.pop()
            el.clear()
            continue
        if name == "Building":
            if cur is not None:
                yield cur
                cur = None
            stack.pop()
            el.clear()
            continue
        stack.pop()
        el.clear()
        if root is not None and len(stack) <= 1:
            root.clear()


# --------------------------------------------------------------- 建物 → メッシュ
def _shoelace_xy(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _footprint(ground_polys, all_polys):
    """footprint(xy 列)を決める。GroundSurface があれば最大面積のもの、無ければ最下層。"""
    if ground_polys:
        poly = max(ground_polys, key=_shoelace_xy)
    else:
        poly = min(all_polys, key=lambda pl: sum(v[2] for v in pl) / max(len(pl), 1))
    return [(float(v[0]), float(v[1])) for v in poly]


def extract_buildings(files, lat0, lon0, clip_rect=None, axis=None,
                      progress=False):
    """複数タイルから建物ジオメトリを抽出。local-m へ変換・三角形化まで行う(z は生標高)。
    戻り値: {buildings:[...], axis, counts}。"""
    counts = {"skipped_no_geom": 0, "skipped_clip": 0, "holes_ignored": 0,
              "lod1": 0, "lod2": 0}
    buildings = []
    for path in files:
        t0 = time.time()
        n_here = 0
        for raw in stream_building_geoms(path):
            counts["holes_ignored"] += raw["n_holes"]
            if raw["lod2"]:
                polys, lod = raw["lod2"], "LOD2"
            elif raw["lod1"]:
                polys, lod = raw["lod1"], "LOD1"
            else:
                counts["skipped_no_geom"] += 1
                continue
            if axis is None:
                a, b, _z = polys[0][1][0]
                axis = sniff_axis_order(a, b)
                print(f"  [sniff] axis order (lat_idx,lon_idx)={axis} "
                      f"from first posList col0={a} col1={b}", flush=True)
            lat_i, lon_i = axis
            bverts = []
            btris = []
            ground_polys = []
            all_polys = []
            for surf, pts in polys:
                pts3 = []
                for tup in pts:
                    lat = tup[lat_i]
                    lon = tup[lon_i]
                    e, n = latlon_to_local(lat, lon, lat0, lon0)
                    pts3.append((e, n, tup[2]))
                dedup, tris = triangulate_3d(pts3)
                if not dedup:
                    continue
                base = len(bverts)
                bverts.extend(dedup)
                for (i, j, k) in tris:
                    btris.append((base + i, base + j, base + k))
                all_polys.append(dedup)
                if surf == "GroundSurface":
                    ground_polys.append(dedup)
            if not bverts:
                counts["skipped_no_geom"] += 1
                continue
            fp = _footprint(ground_polys, all_polys)
            cx = sum(p[0] for p in fp) / len(fp)
            cy = sum(p[1] for p in fp) / len(fp)
            if clip_rect is not None and not (
                    clip_rect[0] <= cx <= clip_rect[2]
                    and clip_rect[1] <= cy <= clip_rect[3]):
                counts["skipped_clip"] += 1
                continue
            V = np.asarray(bverts, dtype=np.float64)
            buildings.append({
                "gml_id": raw["gml_id"], "lod": lod, "verts": V,
                "tris": btris, "footprint": fp,
                "zmin": float(V[:, 2].min()), "zmax": float(V[:, 2].max()),
                "n_tris": len(btris),
            })
            counts["lod2" if lod == "LOD2" else "lod1"] += 1
            n_here += 1
        if progress:
            print(f"  {Path(path).name}  buildings_kept={n_here}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return {"buildings": buildings, "axis": axis, "counts": counts}


# --------------------------------------------------------------- DEM ground0
def compute_ground0_dem(dem_files, lat0, lon0, axis, radius=50.0):
    """DEM タイルの TIN 頂点から原点半径 radius[m] 内の標高中央値を返す。
    戻り値: (ground0 or None, n_points)。"""
    lat_i, lon_i = axis
    mlon = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    band_lat = radius * 1.5 / M_PER_DEG_LAT
    band_lon = radius * 1.5 / mlon
    r2 = radius * radius
    zs = []
    for path in dem_files:
        container = None
        cnt = 0
        for ev, el in ET.iterparse(str(path), events=("start", "end")):
            name = _ln(el.tag)
            if ev == "start":
                if name in ("trianglePatches", "TriangulatedSurface"):
                    container = el
                continue
            if name == "posList" and el.text:
                parts = el.text.split()
                for i in range(0, len(parts) - 2, 3):
                    lat = float(parts[i + lat_i])
                    lon = float(parts[i + lon_i])
                    if abs(lat - lat0) <= band_lat and abs(lon - lon0) <= band_lon:
                        e = (lon - lon0) * mlon
                        n = (lat - lat0) * M_PER_DEG_LAT
                        if e * e + n * n <= r2:
                            zs.append(float(parts[i + 2]))
                el.clear()
            elif name in ("Triangle", "Polygon"):
                cnt += 1
                if container is not None and cnt % 50000 == 0:
                    container.clear()
    if zs:
        return float(np.median(np.asarray(zs))), len(zs)
    return None, 0


# =============================================================== 新 stage: terrain
def collect_tin_points(dem_files, lat0, lon0, axis, rect, progress=True):
    """DEM の dem:TINRelief 内 gml:Triangle(gml:posList)頂点を iterparse で読み、
    local-m 矩形 rect=[xmin,ymin,xmax,ymax](+margin)内に落ちる頂点だけを収集する。

    重い全走査を避けるため、posList 先頭トークン(緯度)による粗プレフィルタで帯外の
    三角形を早期スキップ(既存 dem スキャンと同方式)。処理済み三角形はコンテナ単位で
    clear してメモリを一定に保つ。戻り値: (P[(e,n,z_raw)...] float64, n_points)。
    z は生標高(ground0 差引きは呼び出し側)。cm 量子化で重複頂点を除去。"""
    lat_i, lon_i = axis
    mlon = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    xmin, ymin, xmax, ymax = rect
    marg = 20.0
    latlo = lat0 + (ymin - marg) / M_PER_DEG_LAT
    lathi = lat0 + (ymax + marg) / M_PER_DEG_LAT
    lonlo = lon0 + (xmin - marg) / mlon
    lonhi = lon0 + (xmax + marg) / mlon
    pts = []
    for path in dem_files:
        t0 = time.time()
        kept = 0
        container = None
        cnt = 0
        for ev, el in ET.iterparse(str(path), events=("start", "end")):
            name = _ln(el.tag)
            if ev == "start":
                if name in ("trianglePatches", "TriangulatedSurface"):
                    container = el
                continue
            if name == "posList" and el.text:
                sp = el.text.split()
                if len(sp) >= 3:
                    try:
                        first = float(sp[lat_i])
                    except ValueError:
                        first = None
                    if first is not None and latlo - 0.002 <= first <= lathi + 0.002:
                        for i in range(0, len(sp) - 2, 3):
                            lat = float(sp[i + lat_i])
                            lon = float(sp[i + lon_i])
                            if latlo <= lat <= lathi and lonlo <= lon <= lonhi:
                                e = (lon - lon0) * mlon
                                n = (lat - lat0) * M_PER_DEG_LAT
                                if (xmin - marg <= e <= xmax + marg
                                        and ymin - marg <= n <= ymax + marg):
                                    pts.append((e, n, float(sp[i + 2])))
                                    kept += 1
                el.clear()
            elif name in ("Triangle", "Polygon"):
                cnt += 1
                if container is not None and cnt % 50000 == 0:
                    container.clear()
        if progress:
            print(f"  [terrain] {Path(path).name}  tin_pts_kept={kept}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    if pts:
        P = np.asarray(pts, dtype=np.float64)
        rk = np.round(P[:, :2] / 0.01).astype(np.int64)
        _, ui = np.unique(rk, axis=0, return_index=True)
        P = P[ui]
    else:
        P = np.zeros((0, 3), dtype=np.float64)
    return P, P.shape[0]


def _index_runs(arr):
    """単調非減少な整数配列 arr を (値, start, end) の連続ラン列へ。"""
    runs = []
    n = len(arr)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            runs.append((int(arr[start]), start, i))
            start = i
    runs.append((int(arr[start]), start, n))
    return runs


def rasterize_tin_to_grid(points, x0, y0, cell_m, nx, ny, bucket_m=12.0, k=3):
    """TIN 頂点群 points[(x,y,z)...] を規則格子へラスタ化した標高グリッド (ny,nx) を返す。

    手法(単純化・docstring 明記): 各格子点 (i,j) の座標 (x0+i*cell, y0+j*cell) について、
    一様バケット空間ハッシュで近傍 TIN 頂点を集め、**最近傍 k(=3)頂点の距離加重平均**
    (重み w=1/(distance+eps))で標高を推定する。平面 TIN(定数勾配)なら格子点を
    ±0.2m 程度で復元できる。近傍が全く得られないセルは nan(呼び出し側で 0 埋め)。

    バケット近傍は 1 リングから開始し、候補が k 未満なら最大 8 リングまで拡張する。
    格子点は列 i・行 j の連続ランごとにまとめて cdist をベクトル化評価する。"""
    H = np.full((ny, nx), np.nan, dtype=np.float64)
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] == 0 or nx <= 0 or ny <= 0:
        return H
    px, py, pz = P[:, 0], P[:, 1], P[:, 2]
    pbi = np.floor((px - x0) / bucket_m).astype(np.int64)
    pbj = np.floor((py - y0) / bucket_m).astype(np.int64)
    order = np.lexsort((pbj, pbi))
    sbi, sbj = pbi[order], pbj[order]
    change = np.empty(order.shape[0], dtype=bool)
    change[0] = True
    if order.shape[0] > 1:
        change[1:] = (sbi[1:] != sbi[:-1]) | (sbj[1:] != sbj[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], order.shape[0])
    buckets = {}
    for s, e in zip(starts, ends):
        buckets[(int(sbi[s]), int(sbj[s]))] = order[s:e]

    idx_x = np.arange(nx)
    idx_y = np.arange(ny)
    sample_x = x0 + idx_x * cell_m
    sample_y = y0 + idx_y * cell_m
    col_b = np.floor(idx_x * cell_m / bucket_m).astype(np.int64)
    row_b = np.floor(idx_y * cell_m / bucket_m).astype(np.int64)
    col_runs = _index_runs(col_b)
    row_runs = _index_runs(row_b)
    for (bcx, ci0, ci1) in col_runs:
        cx = sample_x[ci0:ci1]
        for (bcy, rj0, rj1) in row_runs:
            r = 1
            cand = np.zeros(0, dtype=np.int64)
            while True:
                idx_list = []
                for bx in range(bcx - r, bcx + r + 1):
                    for by in range(bcy - r, bcy + r + 1):
                        a = buckets.get((bx, by))
                        if a is not None:
                            idx_list.append(a)
                cand = np.concatenate(idx_list) if idx_list else np.zeros(0, np.int64)
                if cand.size >= k or r >= 8:
                    break
                r += 1
            if cand.size == 0:
                continue
            cy = sample_y[rj0:rj1]
            CX, CY = np.meshgrid(cx, cy)
            flatx = CX.ravel()
            flaty = CY.ravel()
            cpx = px[cand]
            cpy = py[cand]
            cpz = pz[cand]
            dx = flatx[:, None] - cpx[None, :]
            dy = flaty[:, None] - cpy[None, :]
            d2 = dx * dx + dy * dy
            kk = min(k, cand.size)
            nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            rows = np.arange(d2.shape[0])[:, None]
            dsel = np.sqrt(d2[rows, nn])
            zsel = cpz[nn]
            w = 1.0 / (dsel + 1e-6)
            zval = (w * zsel).sum(axis=1) / w.sum(axis=1)
            H[rj0:rj1, ci0:ci1] = zval.reshape(CY.shape)
    return H


# =============================================================== 新 stage: extras
def stream_feature_geoms(path, feature_tags):
    """stream_building_geoms の一般化。feature_tags(localname 集合)で feature 要素を切り出す。
    ubld=uro:UndergroundBuilding / brid=brid:Bridge など Building 以外の地物に対応する。
    ネストした部分要素(brid:BridgeConstructionElement 等)は feature_tags に含めず、親
    feature の文脈で幾何を集約する(cur が None のときだけ新規 feature を開始)。
    幾何抽出のロジック(LOD 判定・穴無視・SURFACE_TAGS)は building 版と同一。"""
    ctx = ET.iterparse(str(path), events=("start", "end"))
    stack = []
    cur = None
    root = None
    for ev, el in ctx:
        name = _ln(el.tag)
        if ev == "start":
            if root is None:
                root = el
            stack.append(name)
            if name in feature_tags and cur is None:
                cur = {"gml_id": el.get(GML_ID), "lod2": [], "lod1": [], "n_holes": 0}
            continue
        if name == "posList":
            if cur is not None and el.text:
                lod = None
                for t in reversed(stack):
                    if t in LOD2_TAGS:
                        lod = "lod2"
                        break
                    if t in LOD1_TAGS:
                        lod = "lod1"
                        break
                    if t in LOD0_TAGS:
                        lod = "lod0"
                        break
                if lod in ("lod2", "lod1"):
                    if "interior" in stack:
                        cur["n_holes"] += 1
                    else:
                        surf = None
                        for t in stack:
                            if t in SURFACE_TAGS:
                                surf = t
                                break
                        pts = _parse_poslist(el.text)
                        if len(pts) >= 3:
                            cur[lod].append((surf, pts))
            stack.pop()
            el.clear()
            continue
        if name in feature_tags and cur is not None:
            yield cur
            cur = None
            stack.pop()
            el.clear()
            continue
        stack.pop()
        el.clear()
        if root is not None and len(stack) <= 1:
            root.clear()


def extract_features(files, lat0, lon0, feature_tags, clip_rect=None, axis=None,
                     progress=False, label=""):
    """ubld/brid 等の地物を building と同じ経路で抽出(LOD2 優先→LOD1・triangulate_3d
    の巻き向き修正込み・重心 bbox クリップ)。戻り値: {structures, axis, counts}。
    structures[i] = {gml_id, lod, verts(Nx3 float64, z=生標高), tris, zmin, zmax, n_tris}。"""
    counts = {"skipped_no_geom": 0, "skipped_clip": 0, "holes_ignored": 0,
              "lod1": 0, "lod2": 0}
    structures = []
    for path in files:
        t0 = time.time()
        n_here = 0
        for raw in stream_feature_geoms(path, feature_tags):
            counts["holes_ignored"] += raw["n_holes"]
            if raw["lod2"]:
                polys, lod = raw["lod2"], "LOD2"
            elif raw["lod1"]:
                polys, lod = raw["lod1"], "LOD1"
            else:
                counts["skipped_no_geom"] += 1
                continue
            if axis is None:
                a, b, _z = polys[0][1][0]
                axis = sniff_axis_order(a, b)
            lat_i, lon_i = axis
            bverts = []
            btris = []
            for surf, pts in polys:
                pts3 = []
                for tup in pts:
                    e, n = latlon_to_local(tup[lat_i], tup[lon_i], lat0, lon0)
                    pts3.append((e, n, tup[2]))
                dedup, tris = triangulate_3d(pts3)
                if not dedup:
                    continue
                base = len(bverts)
                bverts.extend(dedup)
                for (i, j, kk) in tris:
                    btris.append((base + i, base + j, base + kk))
            if not bverts:
                counts["skipped_no_geom"] += 1
                continue
            V = np.asarray(bverts, dtype=np.float64)
            cx = float(V[:, 0].mean())
            cy = float(V[:, 1].mean())
            if clip_rect is not None and not (
                    clip_rect[0] <= cx <= clip_rect[2]
                    and clip_rect[1] <= cy <= clip_rect[3]):
                counts["skipped_clip"] += 1
                continue
            structures.append({
                "gml_id": raw["gml_id"], "lod": lod, "verts": V, "tris": btris,
                "zmin": float(V[:, 2].min()), "zmax": float(V[:, 2].max()),
                "n_tris": len(btris),
            })
            counts["lod2" if lod == "LOD2" else "lod1"] += 1
            n_here += 1
        if progress:
            print(f"  [{label}] {Path(path).name}  kept={n_here}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return {"structures": structures, "axis": axis, "counts": counts}


def _assemble_mesh(structures, ground0):
    """structures → (V float32 (z-ground0), F int32 (Nx3), offsets int32 (len+1))。
    build_outputs と同じオフセット規約(offsets[i]..offsets[i+1] が構造 i の面範囲)。"""
    V_parts = []
    F_rows = []
    offsets = []
    vbase = 0
    for s in structures:
        offsets.append(len(F_rows))
        V_parts.append(s["verts"])
        for (i, j, k) in s["tris"]:
            F_rows.append((vbase + i, vbase + j, vbase + k))
        vbase += s["verts"].shape[0]
    offsets.append(len(F_rows))
    if V_parts:
        V = np.vstack(V_parts).astype(np.float64)
    else:
        V = np.zeros((0, 3), dtype=np.float64)
    V[:, 2] -= ground0
    Vf = V.astype(np.float32)
    F = np.asarray(F_rows, dtype=np.int32).reshape(-1, 3) if F_rows \
        else np.zeros((0, 3), dtype=np.int32)
    offs = np.asarray(offsets, dtype=np.int32)
    return Vf, F, offs


# --------------------------------------------------------------- 組立 / 出力
def _load_city(map_path):
    return json.loads(Path(map_path).read_text(encoding="utf-8"))


def _collect_local_coords(city):
    xs, ys = [], []

    def add(x, y):
        xs.append(float(x))
        ys.append(float(y))

    for nd in city.get("nodes", []):
        if "x" in nd and "y" in nd:
            add(nd["x"], nd["y"])
    for e in city.get("edges", []):
        for pt in e.get("geometry", []):
            add(pt[0], pt[1])
    for b in city.get("buildings", []):
        for pt in b.get("footprint", []):
            add(pt[0], pt[1])
    for p in city.get("pois", []):
        if "x" in p and "y" in p:
            add(p["x"], p["y"])
    # railways は近隣駅まで伸びる長大ポリライン(±数km)でシミュ建物域を表さないため
    # bbox 算出からは除外する(nodes/edges/buildings/pois が中心1km圏を規定)。
    return xs, ys


def select_tiles(city, citygml_dir, origin, buffer_m):
    """sim 地図の local-m 範囲 → 緯度経度 bbox(+buffer)→ 交差3次メッシュ → 該当 gml。
    戻り値: (gml_paths, meta)。crs が local-m 前提(逆変換)、latlon ならそのまま。"""
    lat0, lon0 = origin
    xs, ys = _collect_local_coords(city)
    if not xs:
        raise SystemExit("shibuya_osm.json に座標が見つからない")
    crs = city.get("meta", {}).get("crs", "local-m")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if crs == "local-m":
        clip_rect = [xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m]
        corners = [
            local_to_latlon(clip_rect[0], clip_rect[1], lat0, lon0),
            local_to_latlon(clip_rect[2], clip_rect[1], lat0, lon0),
            local_to_latlon(clip_rect[2], clip_rect[3], lat0, lon0),
            local_to_latlon(clip_rect[0], clip_rect[3], lat0, lon0),
        ]
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        latmin, latmax = min(lats), max(lats)
        lonmin, lonmax = min(lons), max(lons)
    else:  # 座標が既に緯度経度
        dlat = buffer_m / M_PER_DEG_LAT
        dlon = buffer_m / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
        latmin, latmax = ymin - dlat, ymax + dlat
        lonmin, lonmax = xmin - dlon, xmax + dlon
        clip_rect = None

    codes = meshes_for_bbox(latmin, lonmin, latmax, lonmax)
    citygml_dir = Path(citygml_dir)
    all_gml = sorted(citygml_dir.glob("*_bldg_*_op.gml"))
    selected = [p for p in all_gml if p.name.split("_", 1)[0] in codes]
    meta = {
        "clip_rect": clip_rect,
        "latlon_bbox": [latmin, lonmin, latmax, lonmax],
        "mesh_codes": sorted(codes),
        "n_tiles_available": len(all_gml),
    }
    return selected, meta


def build_outputs(result, ground0, ground0_source, sel_meta, origin, buffer_m,
                  radius, n_dem_pts):
    """抽出結果 → (npz dict, index dict)。z は ground0 を差し引いて基準化。"""
    buildings = result["buildings"]
    V_parts = []
    F_rows = []
    offsets = []
    vbase = 0
    for b in buildings:
        offsets.append(len(F_rows))
        V_parts.append(b["verts"])
        for (i, j, k) in b["tris"]:
            F_rows.append((vbase + i, vbase + j, vbase + k))
        vbase += b["verts"].shape[0]
    offsets.append(len(F_rows))

    if V_parts:
        V = np.vstack(V_parts).astype(np.float64)
    else:
        V = np.zeros((0, 3), dtype=np.float64)
    V[:, 2] -= ground0
    Vf = V.astype(np.float32)
    F = np.asarray(F_rows, dtype=np.int32).reshape(-1, 3) if F_rows \
        else np.zeros((0, 3), dtype=np.int32)
    offs = np.asarray(offsets, dtype=np.int32)

    idx_buildings = []
    for b in buildings:
        idx_buildings.append({
            "gml_id": b["gml_id"],
            "footprint": [[round(x, 2), round(y, 2)] for (x, y) in b["footprint"]],
            "height": round(b["zmax"] - ground0, 3),
            "base": round(b["zmin"] - ground0, 3),
            "n_tris": b["n_tris"],
            "lod": b["lod"],
        })
    axis = result["axis"]
    index = {
        "crs": "local-m",
        "axes": "X=east,Y=north,Z=up",
        "n_buildings": len(buildings),
        "n_vertices": int(Vf.shape[0]),
        "n_triangles": int(F.shape[0]),
        "ground0": round(float(ground0), 4),
        "ground0_source": ground0_source,
        "bbox": sel_meta["clip_rect"],
        "buildings": idx_buildings,
        "params": {
            "origin_latlon": [origin[0], origin[1]],
            "buffer_m": buffer_m,
            "axis_order": list(axis) if axis else None,
            "mesh_codes": sel_meta["mesh_codes"],
            "latlon_bbox": sel_meta["latlon_bbox"],
            "dem_radius_m": radius,
            "dem_n_points": n_dem_pts,
            "counts": result["counts"],
        },
    }
    npz = {"V": Vf, "F": F, "building_offsets": offs}
    return npz, index


def run(citygml_dir, map_path, out_dir, dem_dir, origin, buffer_m, radius,
        use_dem=True):
    city = _load_city(map_path)
    selected, sel_meta = select_tiles(city, citygml_dir, origin, buffer_m)
    print(f"[tiles] mesh_codes={sel_meta['mesh_codes']}  selected={len(selected)}"
          f"/{sel_meta['n_tiles_available']} available", flush=True)
    for p in selected:
        print(f"  -> {p.name}", flush=True)
    if not selected:
        raise SystemExit("交差する PLATEAU タイルが無い(citygml-dir を確認)")

    clip_rect = sel_meta["clip_rect"]
    result = extract_buildings(selected, origin[0], origin[1],
                               clip_rect=clip_rect, progress=True)
    axis = result["axis"] or (0, 1)

    # ground0
    ground0 = None
    ground0_source = None
    n_dem_pts = 0
    if use_dem:
        dem_files = []
        prefixes = {c[:6] for c in sel_meta["mesh_codes"]}
        for p in sorted(Path(dem_dir).glob("*_dem_*_op.gml")):
            if p.name.split("_", 1)[0] in prefixes:
                dem_files.append(p)
        if dem_files:
            print(f"[dem] scanning {[p.name for p in dem_files]} (radius={radius}m)...",
                  flush=True)
            t0 = time.time()
            g0, n_dem_pts = compute_ground0_dem(dem_files, origin[0], origin[1],
                                                axis, radius=radius)
            print(f"[dem] points_in_radius={n_dem_pts}  {time.time() - t0:.1f}s",
                  flush=True)
            if g0 is not None:
                ground0 = g0
                ground0_source = "dem"
    if ground0 is None:
        # フォールバック: 全建物頂点 z の下位1%分位点
        all_z = [b["verts"][:, 2] for b in result["buildings"]]
        if all_z:
            ground0 = float(np.percentile(np.concatenate(all_z), 1.0))
        else:
            ground0 = 0.0
        ground0_source = "building_z_p01"
    print(f"[ground0] {ground0:.3f} m  source={ground0_source}", flush=True)

    npz, index = build_outputs(result, ground0, ground0_source, sel_meta,
                               origin, buffer_m, radius, n_dem_pts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "plateau_mesh.npz"
    idx_path = out_dir / "plateau_index.json"
    np.savez_compressed(npz_path, **npz)
    idx_path.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")

    c = result["counts"]
    npz_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"[done] buildings={index['n_buildings']}  vertices={index['n_vertices']}  "
          f"triangles={index['n_triangles']}", flush=True)
    print(f"       skipped_no_geom={c['skipped_no_geom']}  "
          f"skipped_clip={c['skipped_clip']}  holes_ignored={c['holes_ignored']}  "
          f"(LOD2={c['lod2']} LOD1={c['lod1']})", flush=True)
    print(f"       {npz_path}  ({npz_mb:.2f} MB)", flush=True)
    print(f"       {idx_path}", flush=True)
    return {"npz": npz_path, "index": idx_path, "ground0": ground0,
            "ground0_source": ground0_source, "n_buildings": index["n_buildings"],
            "npz_mb": npz_mb, "counts": c}


# --------------------------------------------------------------- 新 stage 実行系
def _select_by_mesh(directory, glob_pat, codes):
    """directory 内 glob_pat のうち、ファイル名先頭のメッシュコードが codes に入るものだけ。"""
    out = []
    for p in sorted(Path(directory).glob(glob_pat)):
        if p.name.split("_", 1)[0] in codes:
            out.append(p)
    return out


def _resolve_ground0(out_dir, dem_dir, origin, radius, codes):
    """新 stage の z 基準 ground0 を解決する。既存 plateau_index.json があればその値を
    そのまま用い(bldg 抽出と同一基準=15.18 を保証)、無ければ DEM から再計算する。"""
    idx = Path(out_dir) / "plateau_index.json"
    if idx.exists():
        try:
            g = json.loads(idx.read_text(encoding="utf-8")).get("ground0")
            if g is not None:
                return float(g), "plateau_index.json"
        except Exception:
            pass
    dem_files = _select_by_mesh(dem_dir, "*_dem_*_op.gml", {c[:6] for c in codes})
    g, _n = compute_ground0_dem(dem_files, origin[0], origin[1], (0, 1),
                                radius=radius)
    return (float(g) if g is not None else 0.0), "dem"


def run_terrain(citygml_dir, map_path, out_dir, dem_dir, origin, buffer_m, radius,
                cell_m=2.0, terrain_buffer=300.0):
    """terrain stage: 両 DEM の TIN を local-m 矩形(シミュ bbox+terrain_buffer)へクリップし、
    2m 格子の高さグリッドへラスタ化。z=標高-ground0[m] を float32(量子化しない)で保存。
    下流(export_3d の terrain_web 書き出し)が int16 量子化を担うため生メートルで渡す。"""
    city = _load_city(map_path)
    xs, ys = _collect_local_coords(city)
    if not xs:
        raise SystemExit("shibuya_osm.json に座標が見つからない")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x0 = xmin - terrain_buffer
    y0 = ymin - terrain_buffer
    x1 = xmax + terrain_buffer
    y1 = ymax + terrain_buffer
    nx = int(math.ceil((x1 - x0) / cell_m)) + 1
    ny = int(math.ceil((y1 - y0) / cell_m)) + 1
    rect = [x0, y0, x1, y1]
    axis = (0, 1)

    codes = set(_sim_mesh_codes(city, citygml_dir, origin, buffer_m))
    ground0, g0_src = _resolve_ground0(out_dir, dem_dir, origin, radius, codes)
    dem_files = sorted(Path(dem_dir).glob("*_dem_*_op.gml"))
    print(f"[terrain] rect(local-m)={[round(v, 1) for v in rect]}  "
          f"grid nx={nx} ny={ny} cell={cell_m}m  ground0={ground0:.3f}({g0_src})",
          flush=True)
    print(f"[terrain] scanning DEM: {[p.name for p in dem_files]}", flush=True)

    P, n_tin = collect_tin_points(dem_files, origin[0], origin[1], axis, rect)
    print(f"[terrain] tin_points(dedup)={n_tin}  rasterizing...", flush=True)
    t0 = time.time()
    H_elev = rasterize_tin_to_grid(P, x0, y0, cell_m, nx, ny)
    Hz = H_elev - ground0
    covered = np.isfinite(Hz)
    n_nan = int((~covered).sum())
    if n_nan:
        Hz = np.where(covered, Hz, 0.0)
    heights = Hz.astype(np.float32)  # (ny,nx) row-major・単位=メートル・量子化なし
    print(f"[terrain] raster done {time.time() - t0:.1f}s  "
          f"uncovered_cells={n_nan}", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "terrain.npz"
    json_path = out_dir / "terrain.json"
    np.savez_compressed(npz_path, heights=heights)
    zc = Hz[covered] if covered.any() else np.zeros(1)
    meta = {
        "x0": round(float(x0), 4), "y0": round(float(y0), 4),
        "cell_m": cell_m, "nx": nx, "ny": ny,
        "ground0": round(float(ground0), 4), "n_tin_points": int(n_tin),
        "z_min_m": round(float(zc.min()), 3), "z_max_m": round(float(zc.max()), 3),
        "uncovered_cells": n_nan, "ground0_source": g0_src,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    npz_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"[terrain] z range = [{meta['z_min_m']}, {meta['z_max_m']}] m", flush=True)
    print(f"       {npz_path}  ({npz_mb:.2f} MB)  {json_path}", flush=True)
    return {"npz": npz_path, "json": json_path, "nx": nx, "ny": ny,
            "z_min": meta["z_min_m"], "z_max": meta["z_max_m"],
            "n_tin_points": n_tin, "npz_mb": npz_mb}


def _sim_mesh_codes(city, citygml_dir, origin, buffer_m):
    _sel, meta = select_tiles(city, citygml_dir, origin, buffer_m)
    return meta["mesh_codes"]


def run_extras(citygml_dir, map_path, out_dir, dem_dir, ubld_dir, brid_dir,
               origin, buffer_m, radius):
    """extras stage: ubld(地下街)+brid(橋梁)を building 同様に抽出し、それぞれ
    U_V/U_F/U_off・B_V/B_F/B_off として extras.npz に保存。ubld は地下なので z<0 が正常。"""
    city = _load_city(map_path)
    selected, sel_meta = select_tiles(city, citygml_dir, origin, buffer_m)
    codes = set(sel_meta["mesh_codes"])
    clip_rect = sel_meta["clip_rect"]
    ubld_files = _select_by_mesh(ubld_dir, "*_ubld_*_op.gml", codes)
    brid_files = _select_by_mesh(brid_dir, "*_brid_*_op.gml", codes)
    ground0, g0_src = _resolve_ground0(out_dir, dem_dir, origin, radius, codes)
    print(f"[extras] ubld tiles={[p.name for p in ubld_files]}  "
          f"brid tiles={[p.name for p in brid_files]}", flush=True)
    print(f"[extras] ground0={ground0:.3f}({g0_src})  clip_rect={clip_rect}",
          flush=True)

    ures = extract_features(ubld_files, origin[0], origin[1],
                            {"UndergroundBuilding"}, clip_rect=clip_rect,
                            axis=(0, 1), progress=True, label="ubld")
    bres = extract_features(brid_files, origin[0], origin[1],
                            {"Bridge"}, clip_rect=clip_rect,
                            axis=(0, 1), progress=True, label="brid")
    # V=float32(n,3)・F=int32(m,3)。F はカテゴリ内 0-based(_assemble_mesh の vbase は
    # カテゴリ先頭から 0 で積むため、ubld_F は ubld_V を、brid_F は brid_V を指す)。
    ubld_V, ubld_F, U_off = _assemble_mesh(ures["structures"], ground0)
    brid_V, brid_F, B_off = _assemble_mesh(bres["structures"], ground0)

    def _zrange(V):
        if V.shape[0]:
            return [round(float(V[:, 2].min()), 3), round(float(V[:, 2].max()), 3)]
        return [0.0, 0.0]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "extras.npz"
    json_path = out_dir / "extras.json"
    # 契約キー: ubld_V/ubld_F/brid_V/brid_F。オフセットは任意(下流未使用)で併記。
    np.savez_compressed(npz_path, ubld_V=ubld_V, ubld_F=ubld_F,
                        brid_V=brid_V, brid_F=brid_F,
                        ubld_off=U_off, brid_off=B_off)
    meta = {
        "ubld": {
            "n_structures": len(ures["structures"]),
            "n_tris": int(ubld_F.shape[0]),
            "n_vertices": int(ubld_V.shape[0]),
            "z_range": _zrange(ubld_V),
            "counts": ures["counts"],
        },
        "brid": {
            "n_structures": len(bres["structures"]),
            "n_tris": int(brid_F.shape[0]),
            "n_vertices": int(brid_V.shape[0]),
            "z_range": _zrange(brid_V),
            "counts": bres["counts"],
        },
        "ground0": round(float(ground0), 4),
        "ground0_source": g0_src,
        "crs": "local-m", "axes": "X=east,Y=north,Z=up",
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    npz_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"[extras] ubld: structures={meta['ubld']['n_structures']} "
          f"tris={meta['ubld']['n_tris']} z={meta['ubld']['z_range']}", flush=True)
    print(f"[extras] brid: structures={meta['brid']['n_structures']} "
          f"tris={meta['brid']['n_tris']} z={meta['brid']['z_range']}", flush=True)
    print(f"       {npz_path}  ({npz_mb:.2f} MB)  {json_path}", flush=True)
    return {"npz": npz_path, "json": json_path, "meta": meta, "npz_mb": npz_mb}


def main(argv):
    # cp932 コンソールでの印字死対策(en-dash 等)。必須。
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PLATEAU CityGML → local-m 抽出")
    ap.add_argument("--citygml-dir", default=str(DEFAULT_CITYGML_DIR))
    ap.add_argument("--map", default=str(REPO_ROOT / "data" / "shibuya_osm.json"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "plateau"))
    ap.add_argument("--dem-dir", default=None,
                    help="既定は citygml-dir と同階層の dem/")
    ap.add_argument("--buffer", type=float, default=150.0)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--no-dem", action="store_true")
    # 追加 stage(既定=従来の bldg 抽出のみで完全互換)
    ap.add_argument("--terrain", action="store_true",
                    help="DEM TIN → 2m 高さグリッド(terrain.npz/json)")
    ap.add_argument("--extras", action="store_true",
                    help="ubld+brid → extras.npz/json")
    ap.add_argument("--all", action="store_true",
                    help="bldg + terrain + extras を全実行")
    ap.add_argument("--ubld-dir", default=None,
                    help="既定は citygml-dir と同階層の ubld/")
    ap.add_argument("--brid-dir", default=None,
                    help="既定は citygml-dir と同階層の brid/")
    ap.add_argument("--cell", type=float, default=2.0, help="terrain 格子間隔[m]")
    ap.add_argument("--terrain-buffer", type=float, default=300.0,
                    help="terrain クリップ矩形のシミュ bbox 外周[m]")
    a = ap.parse_args(argv)
    parent = Path(a.citygml_dir).parent
    dem_dir = a.dem_dir or str(parent / "dem")
    ubld_dir = a.ubld_dir or str(parent / "ubld")
    brid_dir = a.brid_dir or str(parent / "brid")

    do_terrain = a.terrain or a.all
    do_extras = a.extras or a.all
    # 既定(stage フラグ無し)は従来通り bldg のみ。--all は 3 stage 全実行。
    do_bldg = a.all or not (a.terrain or a.extras)

    if do_bldg:
        run(a.citygml_dir, a.map, a.out_dir, dem_dir, DEFAULT_ORIGIN,
            a.buffer, a.radius, use_dem=not a.no_dem)
    if do_terrain:
        run_terrain(a.citygml_dir, a.map, a.out_dir, dem_dir, DEFAULT_ORIGIN,
                    a.buffer, a.radius, cell_m=a.cell,
                    terrain_buffer=a.terrain_buffer)
    if do_extras:
        run_extras(a.citygml_dir, a.map, a.out_dir, dem_dir, ubld_dir, brid_dir,
                   DEFAULT_ORIGIN, a.buffer, a.radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
