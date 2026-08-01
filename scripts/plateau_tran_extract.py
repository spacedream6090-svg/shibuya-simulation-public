"""PLATEAU 道路 LOD3(udx/tran)→ 歩行可能面ポリゴン + 車道面 + LOD3 被覆マップ。【竹-1】

`docs/plans/highfidelity-3d-physics-plan.md` 第2段=竹-1。自前 SFM(`src/society/world/sfm_core.py`)
が現在持っていない「歩ける面の実形状」を CityGML から供給する一次データを作る。
現行は OSM の折れ線(幅なし)しか無い。

処理:
  1. シム地図(data/shibuya_osm.json)の local-m 範囲から緯度経度 bbox と交差する
     3次メッシュを求め、`udx/tran/<meshcode>_tran_6697_op.gml` だけを読む
     (`scripts/plateau_extract.py select_tiles` と同一規約)。
  2. `tran:TrafficArea` / `tran:AuxiliaryTrafficArea` の **lod3MultiSurface** のみを
     iterparse で streaming 抽出し、`function` コード(codelists/TrafficArea_function.xml)で分類:
       walk   = 2000 歩道部 / 2010 自転車歩行者道 / 2020 歩道 / 2030 自転車道
       road   = 1000 車道部 / 1010 車線 / 1020 車道交差部 / 1030 すりつけ / 1040 踏切道 …(1xxx)
       island = 3000 島 / 3010 交通島 / 3020 分離帯 …(3xxx・AuxiliaryTrafficArea)
       other  = 上記以外(6000 自転車駐車場 / 7000 自動車駐車場 など)
     未知コードは other に落としつつ **生コード別の件数を必ず記録**(黙って消さない)。
  3. 局所接平面変換(`plateau_extract` と同一式)→ 水平投影(xy)。z は面ごとに [min,max] を保持。
  4. **LOD3 被覆マップ**: シム矩形を cell_m 格子に切り、LOD3 ポリゴンが 1 枚でも覆うセルを 1 と
     する uint8 マスク。被覆外は下流(SFM)で OSM 線分にフォールバックする二層設計の前提。

出力(`data/plateau/`):
  - `tran_lod3.npz` … 幾何本体(int16×0.05m 量子化・決定論書き出し)
  - `tran_lod3.json` … メタ(件数・面積・被覆率・入力ファイル sha256・復元式)

依存は **stdlib + numpy のみ**。同一入力なら 2 回実行でバイト同一(scripts/det_npz.py)。

使い方:
    python scripts/plateau_tran_extract.py
    python scripts/plateau_tran_extract.py --tran-dir DIR --cell 5 --near 10 --out-dir data/plateau

実測(シム 4 タイル・2026-08-02):
    LOD3 ポリゴン 79,673(歩行系 47,023 / 車道系 29,222 / 島 3,428)・頂点 239,019
    面積 歩行系 244,099 m² / 車道系 389,906 m² / 島 5,158 m²
    被覆(5 m 格子・矩形 2.882 km²): 全体 22.1% / 歩行系 8.7% / near(10 m 膨張)44.1%
      = README の「LOD3.0 整備 1.41 km²」(bbox の約半分)と整合
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_det_npz():
    """scripts/det_npz.py を importlib で直接ロードする(scripts が package ではないため。
    sys.path に scripts/ を足すと共有 pytest プロセスで他モジュールを覆う危険があるので採らない。
    tests/test_export3d.py・plateau_extract._load_triangulate と同方式)。"""
    import importlib.util
    p = Path(__file__).resolve().parent / "det_npz.py"
    spec = importlib.util.spec_from_file_location("_det_npz_impl", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det_npz = _load_det_npz()

# --------------------------------------------------------------- 定数(plateau_extract 準拠)
M_PER_DEG_LAT = 111132.9
M_PER_DEG_LON_EQ = 111320.0
DEFAULT_ORIGIN = (35.65950, 139.70062)
DEFAULT_GROUND0 = 15.18
QUANT = 0.05          # scripts/export_3d.py PLATEAU_QUANT と同一

DEFAULT_CITYGML_ROOT = Path(
    r"C:\Users\塚本翔太\Desktop\13113_shibuya-ku_pref_2025_citygml_1_op")

GML_ID = "{http://www.opengis.net/gml}id"

# TrafficArea_function / AuxiliaryTrafficArea_function(codelists/*.xml の gml:name)
WALK_CODES = {"2000", "2010", "2020", "2030"}
ROAD_CODES = {"1000", "1010", "1020", "1030", "1040", "1050", "1060", "1070",
              "1080", "1090", "1100", "1110", "1120", "1130"}
ISLAND_CODES = {"3000", "3010", "3020"}
CLASS_WALK, CLASS_ROAD, CLASS_ISLAND, CLASS_OTHER = 0, 1, 2, 3
CLASS_NAMES = ("walk", "road", "island", "other")


def classify_function(code):
    """function コード → クラス番号。未知コードは先頭桁で推定し、それも駄目なら other。"""
    if code is None:
        return CLASS_OTHER
    c = code.strip()
    if c in WALK_CODES:
        return CLASS_WALK
    if c in ROAD_CODES:
        return CLASS_ROAD
    if c in ISLAND_CODES:
        return CLASS_ISLAND
    if c[:1] == "2":
        return CLASS_WALK
    if c[:1] == "1":
        return CLASS_ROAD
    if c[:1] == "3":
        return CLASS_ISLAND
    return CLASS_OTHER


# --------------------------------------------------------------- 座標 / メッシュ
def latlon_to_local(lat, lon, lat0=DEFAULT_ORIGIN[0], lon0=DEFAULT_ORIGIN[1]):
    """plateau_extract.latlon_to_local と同式。"""
    e = (lon - lon0) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    n = (lat - lat0) * M_PER_DEG_LAT
    return e, n


def local_to_latlon(x, y, lat0=DEFAULT_ORIGIN[0], lon0=DEFAULT_ORIGIN[1]):
    lat = lat0 + y / M_PER_DEG_LAT
    lon = lon0 + x / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
    return lat, lon


def meshcode3(lat, lon):
    """緯度経度 → 3次地域メッシュコード(JIS X 0410)。plateau_extract と同一。"""
    p = int(lat * 1.5)
    u = int(lon) - 100
    lat_min = lat * 60.0 - p * 40.0
    q = int(lat_min // 5.0)
    lon_min = (lon - (u + 100)) * 60.0
    v = int(lon_min // 7.5)
    r = int((lat_min - q * 5.0) // 0.5)
    s = int((lon_min - v * 7.5) // 0.75)
    return f"{p}{u}{q}{v}{r}{s}"


def meshes_for_bbox(latmin, lonmin, latmax, lonmax):
    codes = set()
    for ila in range(int(math.floor(latmin * 120)), int(math.floor(latmax * 120)) + 1):
        for ilo in range(int(math.floor(lonmin * 80)), int(math.floor(lonmax * 80)) + 1):
            codes.add(meshcode3((ila + 0.5) / 120.0, (ilo + 0.5) / 80.0))
    return codes


def sim_rect(map_path, buffer_m=150.0, origin=DEFAULT_ORIGIN):
    """data/shibuya_osm.json の local-m 全座標 → (clip_rect, latlon_bbox, meshcodes)。
    plateau_extract.select_tiles と同じ規則(railways は除外)。"""
    city = json.loads(Path(map_path).read_text(encoding="utf-8"))
    xs, ys = [], []
    for nd in city.get("nodes", []):
        if "x" in nd and "y" in nd:
            xs.append(float(nd["x"])); ys.append(float(nd["y"]))
    for e in city.get("edges", []):
        for pt in e.get("geometry", []):
            xs.append(float(pt[0])); ys.append(float(pt[1]))
    for b in city.get("buildings", []):
        for pt in b.get("footprint", []):
            xs.append(float(pt[0])); ys.append(float(pt[1]))
    for p in city.get("pois", []):
        if "x" in p and "y" in p:
            xs.append(float(p["x"])); ys.append(float(p["y"]))
    if not xs:
        raise SystemExit("shibuya_osm.json に座標が見つからない")
    rect = [min(xs) - buffer_m, min(ys) - buffer_m,
            max(xs) + buffer_m, max(ys) + buffer_m]
    corners = [local_to_latlon(rect[0], rect[1], *origin),
               local_to_latlon(rect[2], rect[1], *origin),
               local_to_latlon(rect[2], rect[3], *origin),
               local_to_latlon(rect[0], rect[3], *origin)]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    llbbox = [min(lats), min(lons), max(lats), max(lons)]
    return rect, llbbox, sorted(meshes_for_bbox(*llbbox))


# --------------------------------------------------------------- CityGML streaming
def _ln(tag):
    return tag.rsplit("}", 1)[-1]


def _parse_poslist(text):
    parts = text.split()
    return [(float(parts[i]), float(parts[i + 1]), float(parts[i + 2]))
            for i in range(0, len(parts) - 2, 3)]


AREA_TAGS = ("TrafficArea", "AuxiliaryTrafficArea")


def stream_lod3_areas(path, lod_tag="lod3MultiSurface"):
    """1 tran GML を streaming し、TrafficArea/AuxiliaryTrafficArea ごとに
    {gml_id, tag, function, polys:[[(lat_col,lon_col,z)...], ...]} を yield。

    polys は **外環のみ**(interior=穴は件数だけ数える)。lod_tag 配下の posList だけを拾う。"""
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
            if name in AREA_TAGS and cur is None:
                cur = {"gml_id": el.get(GML_ID), "tag": name, "function": None,
                       "polys": [], "n_holes": 0}
            continue
        # end
        if name == "function" and cur is not None and len(stack) >= 2 \
                and stack[-2] == cur["tag"]:
            if cur["function"] is None and el.text:
                cur["function"] = el.text.strip()
        elif name == "posList" and cur is not None and el.text:
            if lod_tag in stack:
                if "interior" in stack:
                    cur["n_holes"] += 1
                else:
                    pts = _parse_poslist(el.text)
                    if len(pts) >= 3:
                        cur["polys"].append(pts)
        elif name in AREA_TAGS and cur is not None:
            yield cur
            cur = None
        stack.pop()
        el.clear()
        if root is not None and len(stack) <= 1:
            root.clear()


# --------------------------------------------------------------- 幾何
def polygon_area_xy(P):
    """xy 多角形の絶対面積(shoelace)。P は (n,2)。"""
    if len(P) < 3:
        return 0.0
    x = P[:, 0]
    y = P[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def rect_intersects(pmin, pmax, rect):
    return not (pmax[0] < rect[0] or pmin[0] > rect[2]
                or pmax[1] < rect[1] or pmin[1] > rect[3])


def rasterize_polygon_edges(P, x0, y0, cell_m, nx, ny, mask):
    """多角形の輪郭が通るセルを 1 にする(保守的ラスタ化)。

    格子「点」判定だけだと **幅 < cell_m の細い歩道が丸ごと落ちる**。輪郭を
    cell_m/2 間隔で標本化してセルを塗ることで、細長いポリゴンも必ず被覆に現れる
    (= 「頂点を含むセルは必ず被覆」という不変量が成り立つ)。"""
    if P.shape[0] < 2:
        return mask
    A = P
    B = np.roll(P, -1, axis=0)
    seg = B - A
    L = np.hypot(seg[:, 0], seg[:, 1])
    nsub = np.maximum(1, np.ceil(L / (cell_m * 0.5)).astype(np.int64))
    for k in range(P.shape[0]):
        m = nsub[k]
        t = np.arange(m + 1, dtype=np.float64) / m
        xs = A[k, 0] + seg[k, 0] * t
        ys = A[k, 1] + seg[k, 1] * t
        ii = np.round((xs - x0) / cell_m).astype(np.int64)
        jj = np.round((ys - y0) / cell_m).astype(np.int64)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        if ok.any():
            mask[jj[ok], ii[ok]] = 1
    return mask


def dilate_mask(mask, radius_cells):
    """正方形構造要素(半径 radius_cells)の膨張。積分画像で O(N)・決定論。
    LOD3 整備「地区」の外形を近似し、二層フォールバックの境界に使う。"""
    r = int(radius_cells)
    if r <= 0:
        return mask.copy()
    m = (mask > 0).astype(np.int32)
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(m, axis=0), axis=1)
    ny, nx = m.shape
    j0 = np.clip(np.arange(ny) - r, 0, ny)
    j1 = np.clip(np.arange(ny) + r + 1, 0, ny)
    i0 = np.clip(np.arange(nx) - r, 0, nx)
    i1 = np.clip(np.arange(nx) + r + 1, 0, nx)
    s = (ii[np.ix_(j1, i1)] - ii[np.ix_(j0, i1)]
         - ii[np.ix_(j1, i0)] + ii[np.ix_(j0, i0)])
    return (s > 0).astype(np.uint8)


def rasterize_polygons(polys, x0, y0, cell_m, nx, ny, mask=None, edges=True):
    """xy 多角形群 → 被覆マスク(uint8 (ny,nx))。格子点 (x0+i*cell, y0+j*cell) が
    多角形の内側なら 1。交差数法(crossing number)をポリゴン bbox 内でベクトル評価する。

    edges=True(既定)なら輪郭が通るセルも塗る(rasterize_polygon_edges)。
    面積の計算にはこのマスクを使わない(面積は polygon_area_xy の合計を JSON に出す)。"""
    if mask is None:
        mask = np.zeros((ny, nx), dtype=np.uint8)
    for P in polys:
        P = np.asarray(P, dtype=np.float64)
        if P.shape[0] < 3:
            continue
        if P.shape[0] > 1 and P[0, 0] == P[-1, 0] and P[0, 1] == P[-1, 1]:
            P = P[:-1]
            if P.shape[0] < 3:
                continue
        if edges:
            rasterize_polygon_edges(P, x0, y0, cell_m, nx, ny, mask)
        i0 = max(0, int(math.ceil((P[:, 0].min() - x0) / cell_m)))
        i1 = min(nx - 1, int(math.floor((P[:, 0].max() - x0) / cell_m)))
        j0 = max(0, int(math.ceil((P[:, 1].min() - y0) / cell_m)))
        j1 = min(ny - 1, int(math.floor((P[:, 1].max() - y0) / cell_m)))
        if i0 > i1 or j0 > j1:
            continue
        gx = x0 + np.arange(i0, i1 + 1, dtype=np.float64) * cell_m
        gy = y0 + np.arange(j0, j1 + 1, dtype=np.float64) * cell_m
        GX = gx[None, :]
        GY = gy[:, None]
        inside = np.zeros((gy.size, gx.size), dtype=bool)
        xa, ya = P[:, 0], P[:, 1]
        xb, yb = np.roll(xa, -1), np.roll(ya, -1)
        for k in range(P.shape[0]):
            x1, y1, x2, y2 = xa[k], ya[k], xb[k], yb[k]
            if y1 == y2:
                continue
            cond = ((y1 > GY) != (y2 > GY))
            xint = x1 + (GY - y1) * (x2 - x1) / (y2 - y1)
            inside ^= cond & (GX < xint)
        sub = mask[j0:j1 + 1, i0:i1 + 1]
        np.copyto(sub, 1, where=inside)
    return mask


def quantize_xy(V, quant=QUANT):
    """local-m (N,2) → (相対整数, origin_q, dtype 名)。復元: xy = (q + origin_q) * quant。

    origin_q が整数なので量子化格子は plateau_web.json / tracks_bin と同じ
    グローバル 0.05 m 格子に乗る。int16 の相対幅(32767 単位 = 1638 m)を超える範囲を
    渡された場合は int32 へ落とす(シム bbox の実測は 32,649 単位 = 余裕 6 m しかない
    ので、bbox やバッファを少し広げるだけで即座に超える)。"""
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int16), np.zeros(2, dtype=np.int64), "int16"
    Q = np.round(V / quant).astype(np.int64)
    o = Q.min(axis=0)
    rel = Q - o
    if int(rel.max()) <= 32767:
        return rel.astype(np.int16), o, "int16"
    return rel.astype(np.int32), o, "int32"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _resolve_ground0(out_dir):
    p = Path(out_dir) / "plateau_index.json"
    if p.exists():
        try:
            g = json.loads(p.read_text(encoding="utf-8")).get("ground0")
            if g is not None:
                return float(g), "plateau_index.json"
        except Exception:
            pass
    return DEFAULT_GROUND0, "default"


# --------------------------------------------------------------- 本体
def extract(files, rect, ground0, origin=DEFAULT_ORIGIN, axis=(0, 1),
            progress=True):
    """tran GML 群 → {polys, cls, code, zrange, gml_ids, counts}。
    polys は local-m xy の list[np.ndarray (n,2)](閉じ点は落とす)。"""
    lat_i, lon_i = axis
    polys, cls, codes, zr, ids = [], [], [], [], []
    counts = {"areas": 0, "areas_no_lod3": 0, "polys_raw": 0, "polys_clipped": 0,
              "holes_ignored": 0, "degenerate": 0}
    code_hist = {}
    for path in files:
        t0 = time.time()
        n_here = 0
        for a in stream_lod3_areas(path):
            counts["areas"] += 1
            counts["holes_ignored"] += a["n_holes"]
            if not a["polys"]:
                counts["areas_no_lod3"] += 1
                continue
            c = classify_function(a["function"])
            key = a["function"] or "(none)"
            code_hist[key] = code_hist.get(key, 0) + 1
            for pts in a["polys"]:
                counts["polys_raw"] += 1
                arr = np.asarray(pts, dtype=np.float64)
                e, n = latlon_to_local(arr[:, lat_i], arr[:, lon_i], *origin)
                P = np.column_stack([e, n])
                if P.shape[0] > 1 and P[0, 0] == P[-1, 0] and P[0, 1] == P[-1, 1]:
                    P = P[:-1]
                if P.shape[0] < 3:
                    counts["degenerate"] += 1
                    continue
                pmin = P.min(axis=0)
                pmax = P.max(axis=0)
                if not rect_intersects(pmin, pmax, rect):
                    counts["polys_clipped"] += 1
                    continue
                polys.append(P)
                cls.append(c)
                codes.append(int(a["function"]) if (a["function"] or "").isdigit() else 0)
                zr.append((float(arr[:, 2].min() - ground0),
                           float(arr[:, 2].max() - ground0)))
                ids.append(a["gml_id"] or "")
                n_here += 1
        if progress:
            print(f"  [tran] {Path(path).name}  polys_kept={n_here}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return {"polys": polys, "cls": cls, "codes": codes, "zrange": zr,
            "gml_ids": ids, "counts": counts, "code_hist": code_hist}


def run(tran_dir, map_path, out_dir, origin=DEFAULT_ORIGIN, buffer_m=150.0,
        cell_m=5.0, near_m=10.0, progress=True):
    rect, llbbox, codes = sim_rect(map_path, buffer_m, origin)
    tran_dir = Path(tran_dir)
    files = [p for p in sorted(tran_dir.glob("*_tran_*_op.gml"))
             if p.name.split("_", 1)[0] in set(codes)]
    if not files:
        raise SystemExit(f"交差する tran タイルが無い: {tran_dir}")
    ground0, g0src = _resolve_ground0(out_dir)
    if progress:
        print(f"[tran] mesh_codes={codes}  files={[p.name for p in files]}", flush=True)
        print(f"[tran] rect={[round(v, 1) for v in rect]}  "
              f"ground0={ground0}({g0src})  cell={cell_m}m", flush=True)

    res = extract(files, rect, ground0, origin, progress=progress)
    polys = res["polys"]

    # --- 幾何を1本の配列に平積み
    if polys:
        allxy = np.vstack(polys)
    else:
        allxy = np.zeros((0, 2), dtype=np.float64)
    Q, origin_q, xy_dtype = quantize_xy(allxy)
    offs = np.zeros(len(polys) + 1, dtype=np.int32)
    for i, P in enumerate(polys):
        offs[i + 1] = offs[i] + P.shape[0]

    # --- 被覆マップ(シム矩形をぴったり覆う格子)
    x0, y0, x1, y1 = rect
    nx = int(math.ceil((x1 - x0) / cell_m)) + 1
    ny = int(math.ceil((y1 - y0) / cell_m)) + 1
    t0 = time.time()
    cover_all = np.zeros((ny, nx), dtype=np.uint8)
    cover_walk = np.zeros((ny, nx), dtype=np.uint8)
    for P, c in zip(polys, res["cls"]):
        rasterize_polygons([P], x0, y0, cell_m, nx, ny, cover_all)
        if c == CLASS_WALK:
            rasterize_polygons([P], x0, y0, cell_m, nx, ny, cover_walk)
    near_cells = int(round(near_m / cell_m))
    cover_near = dilate_mask(cover_all, near_cells)
    if progress:
        print(f"[tran] coverage raster {time.time() - t0:.1f}s", flush=True)

    areas = {n: 0.0 for n in CLASS_NAMES}
    for P, c in zip(polys, res["cls"]):
        areas[CLASS_NAMES[c]] += polygon_area_xy(P)
    n_by_class = {n: 0 for n in CLASS_NAMES}
    for c in res["cls"]:
        n_by_class[CLASS_NAMES[c]] += 1

    out_dir = Path(out_dir)
    npz = {
        "xy": Q,
        "origin_q": origin_q.astype(np.int64),
        "poly_offsets": offs,
        "poly_class": np.asarray(res["cls"], dtype=np.uint8),
        "poly_code": np.asarray(res["codes"], dtype=np.uint16),
        "poly_z": np.asarray(res["zrange"], dtype=np.float32).reshape(-1, 2),
        "cover_all": cover_all,
        "cover_walk": cover_walk,
        "cover_near": cover_near,
    }
    det_npz.save_npz(out_dir / "tran_lod3.npz", npz)
    npz_bytes = (out_dir / "tran_lod3.npz").stat().st_size

    ncell = nx * ny
    meta = {
        "schema": "plateau_tran_lod3/1",
        "crs": "local-m", "axes": "X=east,Y=north,Z=up",
        "quant_scale": QUANT,
        "dequant": "xy_m = (xy + origin_q) * quant_scale",
        "origin_latlon": [origin[0], origin[1]],
        "ground0": round(float(ground0), 4), "ground0_source": g0src,
        "clip_rect": [round(v, 3) for v in rect],
        "latlon_bbox": llbbox,
        "lod": "lod3MultiSurface",
        "n_polygons": len(polys),
        "n_vertices": int(Q.shape[0]),
        "xy_dtype": xy_dtype,
        "n_polygons_by_class": n_by_class,
        "area_m2_by_class": {k: round(v, 2) for k, v in areas.items()},
        "function_code_hist": dict(sorted(res["code_hist"].items())),
        "class_codes": {"walk": sorted(WALK_CODES), "road": sorted(ROAD_CODES),
                        "island": sorted(ISLAND_CODES)},
        "counts": res["counts"],
        "coverage": {
            "x0": round(float(x0), 3), "y0": round(float(y0), 3),
            "cell_m": cell_m, "nx": nx, "ny": ny, "n_cells": ncell,
            "rule": ("cover_* = 1 if the grid point (x0+i*cell, y0+j*cell) is inside "
                     "a LOD3 polygon OR a polygon outline passes through the cell "
                     "(conservative: thin sidewalks narrower than cell_m survive)"),
            "near_m": near_m, "near_cells": near_cells,
            "near_rule": "cover_near = square dilation of cover_all by near_cells",
            "n_covered_all": int(cover_all.sum()),
            "n_covered_walk": int(cover_walk.sum()),
            "n_covered_near": int(cover_near.sum()),
            "ratio_all": round(float(cover_all.sum()) / ncell, 5),
            "ratio_walk": round(float(cover_walk.sum()) / ncell, 5),
            "ratio_near": round(float(cover_near.sum()) / ncell, 5),
            "rect_area_km2": round((x1 - x0) * (y1 - y0) / 1e6, 4),
        },
        "source": {
            "dir": str(tran_dir),
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": sha256_file(p)} for p in files],
        },
        "npz": "tran_lod3.npz", "npz_bytes": npz_bytes,
        "attribution": ("国土交通省 Project PLATEAU「3D都市モデル 渋谷区(2025年度)」"
                        "CityGML(udx/tran・LOD3)を加工して作成"),
    }
    det_npz.dump_json(out_dir / "tran_lod3.json", meta, indent=1)
    if progress:
        print(f"[tran] polygons={len(polys)} vertices={Q.shape[0]} "
              f"by_class={n_by_class}", flush=True)
        print(f"[tran] area(m2)={meta['area_m2_by_class']}", flush=True)
        print(f"[tran] coverage all={meta['coverage']['ratio_all']:.3f} "
              f"walk={meta['coverage']['ratio_walk']:.3f} "
              f"near({near_m}m)={meta['coverage']['ratio_near']:.3f} "
              f"(rect {meta['coverage']['rect_area_km2']} km2)", flush=True)
        print(f"       {out_dir / 'tran_lod3.npz'} ({npz_bytes / 1e6:.2f} MB)  "
              f"{out_dir / 'tran_lod3.json'}", flush=True)
    return meta


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PLATEAU tran LOD3 → 歩行可能面 + 被覆マップ")
    ap.add_argument("--tran-dir", default=str(DEFAULT_CITYGML_ROOT / "udx" / "tran"))
    ap.add_argument("--map", default=str(REPO_ROOT / "data" / "shibuya_osm.json"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "plateau"))
    ap.add_argument("--buffer", type=float, default=150.0)
    ap.add_argument("--cell", type=float, default=5.0, help="被覆マップ格子間隔[m]")
    ap.add_argument("--near", type=float, default=10.0,
                    help="cover_near の膨張半径[m](LOD3 整備地区の近似)")
    a = ap.parse_args(argv)
    run(a.tran_dir, a.map, a.out_dir, DEFAULT_ORIGIN, a.buffer, a.cell, a.near)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
