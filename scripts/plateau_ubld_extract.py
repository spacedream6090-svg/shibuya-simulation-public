"""PLATEAU 地下街 LOD4.1(udx/ubld)→ 2D 壁線分 + 層分離フロア + ゲート点 + 表示メッシュ。【竹-2】

`docs/plans/highfidelity-3d-physics-plan.md` 第2段=竹-2。自前 SFM
(`src/society/world/sfm_core.py`)が docstring で「意図的に省略」と宣言している
**対壁斥力 f_iW** の入力(壁の線分集合と歩行可能面)を CityGML から供給する。

入力: `udx/ubld/53393596_ubld_6697_op.gml`(34 MB・渋谷駅周辺地下街・この 1 ファイルのみ)

処理:
  1. iterparse で lod4 幾何の posList を streaming 抽出し、**最内の境界面タグ**で種別付け
     (Door は WallSurface の opening 配下に入るので「最外」ではなく「最内」を採る)。
  2. `InteriorWallSurface` … Newell 法線が水平に近い面 = 鉛直面 → 水平投影 → **2D 線分**。
     CityGML の内壁は「面」であって厚みのある壁ではないので **同じ壁の表裏 2 枚**が別ポリゴンで
     来るうえ、1 枚の壁が高さ方向に複数ポリゴンへ分割されている。距離 + 角度の近接判定で
     **一意な線分へマージ**する(空間ハッシュ・決定論的な走査順)。dedup は **層ごと**
     (吹き抜け/階段室で複数フロアが同じ平面位置に壁を持つため、平面だけで潰すとフロアの壁が欠ける)。
  3. `FloorSurface` … **面積重み付き z ヒストグラムの山**で層に分離。SFM は 2D なので層分離が
     前提(地下街は T.P. 0.90〜17.97 m に床が重なる)。素朴なギャップ分割は
     **階段・スロープが層間を z 連続でつなぐため 1 塊に潰れる**(実測で確認済み)。
  4. `Door` … ポリゴン重心を **ゲート点**(既存 SignalGate と同型の通過制御に接続可能)。
  5. `ClosureSurface` … 仮想閉鎖面 = **開口部の境界**(層間/エリア間 接続グラフの辺の素材)。
  6. 表示用メッシュ(面種別タグ付き・int16×0.05m 量子化)を併産(レーンB 梅=塗り分け表示用)。

出力(`data/plateau/`):
  - `ubld_lod4.json`      … 壁線分 / 層 / ゲート / 開口 / 統計(人間可読・下流 SFM の一次入力)
  - `ubld_lod4_mesh.npz`  … 表示用三角メッシュ(kind 付き・決定論書き出し)

依存は **stdlib + numpy のみ**(三角形化は scripts/plateau_extract.triangulate_3d を再利用)。

使い方:
    python scripts/plateau_ubld_extract.py
    python scripts/plateau_ubld_extract.py --gml PATH --merge-tol 0.30 --merge-deg 8
    python scripts/plateau_ubld_extract.py --layer-sep 2.0 --layer-bin 0.5 --layer-min-frac 0.03

実測(渋谷駅周辺地下街・2026-08-02):
    層 4(local z = -13.25 / -10.25 / -6.75 / -4.75 m・床面積 2649 / 5792 / 2643 / 3678 m²)
    内壁 1,934 要素(うち 132 は開口部のみ)/ 5,687 面 → 鉛直 5,468 → 生線分 4,849
      → **一意 2,834 本**(層ごと dedup・総延長 7,338 m。層を無視すると 2,616 本)
    扉 188 要素(563 ポリゴン)→ ゲート 188 点 / 閉鎖面 218 要素 → 開口境界 225 本
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
QUANT = 0.05

DEFAULT_GML = Path(r"C:\Users\塚本翔太\Desktop\13113_shibuya-ku_pref_2025_citygml_1_op"
                   r"\udx\ubld\53393596_ubld_6697_op.gml")

GML_ID = "{http://www.opengis.net/gml}id"

# 境界面(最内優先で採る)。CityGML の bldg:_BoundarySurface 系 + 開口部。
SURFACE_TAGS = ("Door", "Window", "ClosureSurface", "FloorSurface",
                "CeilingSurface", "InteriorWallSurface", "WallSurface",
                "RoofSurface", "GroundSurface", "OuterCeilingSurface",
                "OuterFloorSurface", "IntBuildingInstallation")
KIND_ORDER = ("wall", "interior_wall", "floor", "ceiling", "roof", "ground",
              "closure", "door", "window", "installation", "other")
KIND_ID = {k: i for i, k in enumerate(KIND_ORDER)}
TAG_KIND = {
    "WallSurface": "wall", "InteriorWallSurface": "interior_wall",
    "FloorSurface": "floor", "CeilingSurface": "ceiling",
    "RoofSurface": "roof", "GroundSurface": "ground",
    "ClosureSurface": "closure", "Door": "door", "Window": "window",
    "IntBuildingInstallation": "installation",
    "OuterCeilingSurface": "ceiling", "OuterFloorSurface": "floor",
}
LOD4_TAGS = {"lod4MultiSurface", "lod4Solid", "lod4Geometry"}


# --------------------------------------------------------------- 座標
def latlon_to_local(lat, lon, lat0=DEFAULT_ORIGIN[0], lon0=DEFAULT_ORIGIN[1]):
    """plateau_extract.latlon_to_local と同式。"""
    e = (lon - lon0) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    n = (lat - lat0) * M_PER_DEG_LAT
    return e, n


def _load_triangulate():
    """scripts/plateau_extract.py の triangulate_3d(凹対応 ear clipping・巻き向き補正込み)
    を importlib で直接ロード(scripts が package でないため。tests/test_export3d.py と同方式)。"""
    p = REPO_ROOT / "scripts" / "plateau_extract.py"
    spec = importlib.util.spec_from_file_location("plateau_extract_for_ubld", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triangulate_3d


# --------------------------------------------------------------- CityGML streaming
def _ln(tag):
    return tag.rsplit("}", 1)[-1]


def _parse_poslist(text):
    parts = text.split()
    return [(float(parts[i]), float(parts[i + 1]), float(parts[i + 2]))
            for i in range(0, len(parts) - 2, 3)]


def innermost_surface(stack):
    """スタック中で **最も内側** の境界面タグ。Door/Window は WallSurface の
    opening 配下に入るため、最外(plateau_extract の規則)だと壁に吸われてしまう。"""
    for t in reversed(stack):
        if t in SURFACE_TAGS:
            return t
    return None


def stream_lod4_polygons(path, stats=None):
    """ubld GML を streaming し、lod4 幾何のポリゴンを 1 枚ずつ yield。

    stats(dict)を渡すと `stats["elements_seen"]` に **ファイル中の境界面要素数**
    (タグ別)を積む。「要素はあるが自前の面を持たない」= 開口部だけの
    InteriorWallSurface(実測 132 件)を後段で検出するために要る。

    yield: {"tag": 境界面タグ or None, "elem": 地物要素の通し番号(同一 Door/壁の面は同じ値),
            "elem_id": gml:id or None, "hole": bool, "pts": [(c0,c1,z), ...]}

    1 つの `bldg:Door` が複数のポリゴンを持つ(実測 188 要素 / 563 ポリゴン)ため、
    「扉 = ゲート 1 点」を得るには **要素単位の集約** が要る。elem がその鍵。
    穴(interior/LinearRing)は hole=True で通知だけする(幾何は使わない)。"""
    ctx = ET.iterparse(str(path), events=("start", "end"))
    stack = []
    elems = []          # 開いている境界面要素のスタック [(tag, seq, gml_id)]
    seen = None if stats is None else stats.setdefault("elements_seen", {})
    seq = 0
    root = None
    for ev, el in ctx:
        name = _ln(el.tag)
        if ev == "start":
            if root is None:
                root = el
            stack.append(name)
            if name in SURFACE_TAGS:
                elems.append((name, seq, el.get(GML_ID)))
                seq += 1
                if seen is not None:
                    seen[name] = seen.get(name, 0) + 1
            continue
        if name == "posList" and el.text:
            if LOD4_TAGS & set(stack):
                hole = len(stack) >= 3 and stack[-3] == "interior"
                pts = _parse_poslist(el.text)
                if len(pts) >= 3:
                    tag, es, eid = elems[-1] if elems else (None, -1, None)
                    yield {"tag": tag, "elem": es, "elem_id": eid,
                           "hole": hole, "pts": pts}
        elif name in SURFACE_TAGS and elems and elems[-1][0] == name:
            elems.pop()
        stack.pop()
        el.clear()
        if root is not None and len(stack) <= 1:
            root.clear()


# --------------------------------------------------------------- 幾何ユーティリティ
def newell_normal(P):
    """3D 平面ポリゴンの法線(Newell 法)。P は (n,3) ndarray。"""
    A = np.asarray(P, dtype=np.float64)
    B = np.roll(A, -1, axis=0)
    nx = float(np.sum((A[:, 1] - B[:, 1]) * (A[:, 2] + B[:, 2])))
    ny = float(np.sum((A[:, 2] - B[:, 2]) * (A[:, 0] + B[:, 0])))
    nz = float(np.sum((A[:, 0] - B[:, 0]) * (A[:, 1] + B[:, 1])))
    return nx, ny, nz


def is_vertical(P, max_cos=0.2):
    """面が鉛直か(法線が水平に近いか)。|nz| / |n| <= max_cos で判定。
    max_cos=0.2 は鉛直から約 11.5° までを鉛直とみなす。"""
    nx, ny, nz = newell_normal(P)
    m = math.sqrt(nx * nx + ny * ny + nz * nz)
    if m == 0.0:
        return False
    return abs(nz) / m <= max_cos


def project_segment(P):
    """鉛直ポリゴン(n,3)を水平投影して **最遠 2 点** を結ぶ 2D 線分にする。
    鉛直面は xy へ潰すと線分に退化するので、最遠点対がその線分そのものになる。
    頂点数は 1 面あたり数点なので総当たり O(n^2) で十分(決定論)。"""
    A = np.asarray(P, dtype=np.float64)[:, :2]
    n = A.shape[0]
    if n < 2:
        return None
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)
    k = int(np.argmax(d2))
    i, j = divmod(k, n)
    if d2[i, j] <= 0.0:
        return None
    return (float(A[i, 0]), float(A[i, 1]), float(A[j, 0]), float(A[j, 1]))


def canonical_segment(seg):
    """線分の端点を辞書順に正規化(向きの差を消す)。"""
    x1, y1, x2, y2 = seg
    if (x2, y2) < (x1, y1):
        return (x2, y2, x1, y1)
    return (x1, y1, x2, y2)


def segment_length(seg):
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def segment_angle(seg):
    """線分の向き(0..pi・無向)。"""
    a = math.atan2(seg[3] - seg[1], seg[2] - seg[0])
    if a < 0.0:
        a += math.pi
    if a >= math.pi:
        a -= math.pi
    return a


def merge_segments(segs, tol=0.30, deg=8.0):
    """壁線分の重複除去。CityGML の内壁は「表裏 2 枚 × 高さ分割」で同じ壁が何本にもなる。

    規則: 正規化端点が両方とも tol[m] 以内、かつ向きの差が deg[度] 以内なら同一とみなす。
    走査順は正規化端点の辞書順に固定(入力順に依存しない=決定論)。
    採用は「先に来た 1 本」。tol は既定 0.30 m(地下街の壁厚 + 測量誤差の目安)。

    戻り値: (unique[list[seg]], mapping[list[int]] 入力 i → unique index)。"""
    order = sorted(range(len(segs)), key=lambda i: canonical_segment(segs[i]))
    cell = max(tol, 1e-6)
    buckets = {}
    unique = []
    mapping = [-1] * len(segs)
    cos_tol = math.cos(math.radians(deg))
    for i in order:
        s = canonical_segment(segs[i])
        a = segment_angle(s)
        mx = (s[0] + s[2]) * 0.5
        my = (s[1] + s[3]) * 0.5
        bx = int(math.floor(mx / cell))
        by = int(math.floor(my / cell))
        hit = -1
        for ux in (bx - 1, bx, bx + 1):
            for uy in (by - 1, by, by + 1):
                for k in buckets.get((ux, uy), ()):
                    t = unique[k]
                    if math.hypot(s[0] - t[0], s[1] - t[1]) > tol:
                        continue
                    if math.hypot(s[2] - t[2], s[3] - t[3]) > tol:
                        continue
                    da = abs(a - segment_angle(t))
                    da = min(da, math.pi - da)
                    if math.cos(da) < cos_tol:
                        continue
                    hit = k
                    break
                if hit >= 0:
                    break
            if hit >= 0:
                break
        if hit < 0:
            hit = len(unique)
            unique.append(s)
            buckets.setdefault((bx, by), []).append(hit)
        mapping[i] = hit
    return unique, mapping


def layer_peaks(zs, weights, bin_m=0.5, min_sep_m=2.0, min_frac=0.03):
    """床面の **面積重み付き z ヒストグラム** から層(フロア)の代表 z を検出する。

    素朴なギャップ分割は地下街では失敗する: 階段・スロープ・エスカレータが層と層を
    z 方向に連続的につないでいるため、実測では -14.2〜+2.1 m が **1 塊** になる。
    そこで「床面積が集中している高さ」= ヒストグラムの山を層とみなす:

      1. bin_m 刻みで面積を集計。
      2. 重い bin から貪欲に採用。ただし既採用の層から min_sep_m 以上離れていること。
      3. 総面積の min_frac 未満しかない bin は層とみなさない(ノイズ床の除外)。
      4. 採用 bin の中心を z 昇順に並べて層 index とする。

    決定論: bin 重みの同値は bin index 昇順で解決する。
    戻り値: (peaks[list[float]], hist[list[(z_center, area)]])。"""
    if len(zs) == 0:
        return [], []
    z = np.asarray(zs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    lo = math.floor(float(z.min()) / bin_m) * bin_m
    hi = math.ceil(float(z.max()) / bin_m) * bin_m + bin_m
    nb = max(1, int(round((hi - lo) / bin_m)))
    edges = lo + np.arange(nb + 1) * bin_m
    idx = np.clip(((z - lo) / bin_m).astype(np.int64), 0, nb - 1)
    hist = np.zeros(nb, dtype=np.float64)
    np.add.at(hist, idx, w)
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = float(hist.sum())
    thr = total * min_frac
    order = sorted(range(nb), key=lambda i: (-hist[i], i))
    peaks = []
    for i in order:
        if hist[i] < thr or hist[i] <= 0.0:
            break
        c = float(centers[i])
        if all(abs(c - p) >= min_sep_m for p in peaks):
            peaks.append(c)
    peaks.sort()
    return peaks, [(round(float(centers[i]), 3), round(float(hist[i]), 2))
                   for i in range(nb) if hist[i] > 0.0]


def merge_by_layer(segs, zs, peaks, tol, deg):
    """線分を層へ割り当ててから **層内で** dedup する。

    segs[i] の層は zs[i][0](その面の下端)に最も近い層。層ごとに merge_segments を
    掛け、層番号昇順・層内は merge_segments の順で連結する(決定論)。
    戻り値: (unique_segs, unique_z[[zmin,zmax],...], unique_layer[list[int]])。"""
    n_layers = max(1, len(peaks))
    buckets = [[] for _ in range(n_layers)]
    for i, s in enumerate(segs):
        li = assign_layer(zs[i][0], peaks)
        buckets[max(0, li)].append(i)
    out_segs, out_z, out_layer = [], [], []
    for li, idxs in enumerate(buckets):
        if not idxs:
            continue
        sub = [segs[i] for i in idxs]
        uniq, mapping = merge_segments(sub, tol, deg)
        zr = [[float("inf"), float("-inf")] for _ in uniq]
        for k, u in enumerate(mapping):
            zr[u][0] = min(zr[u][0], zs[idxs[k]][0])
            zr[u][1] = max(zr[u][1], zs[idxs[k]][1])
        out_segs.extend(uniq)
        out_z.extend(zr)
        out_layer.extend([li if peaks else -1] * len(uniq))
    return out_segs, out_z, out_layer


def cluster_z(values, gap=1.5):
    """1 次元 z のギャップ分割クラスタリング(**本番では未使用・対照用**)。

    昇順に並べ、隣接差が gap[m] を超えたら層を切る。地下街の実データでは階段・スロープが
    層間を連続でつなぐため **全体が 1 層に潰れる**(tests で実証)。層分離は
    layer_peaks(面積重み付きヒストグラム)を使うこと。戻り値: (labels, layers)。決定論。"""
    if not len(values):
        return [], []
    idx = sorted(range(len(values)), key=lambda i: (values[i], i))
    labels = [0] * len(values)
    bounds = []
    cur = 0
    start = 0
    for k in range(1, len(idx)):
        if values[idx[k]] - values[idx[k - 1]] > gap:
            bounds.append((start, k))
            cur += 1
            start = k
        labels[idx[k]] = cur
    # 最小 z の要素は必ず層 0(labels の初期値 0 のまま)
    bounds.append((start, len(idx)))
    layers = []
    for li, (a, b) in enumerate(bounds):
        zs = [values[idx[t]] for t in range(a, b)]
        layers.append({"layer": li, "n": len(zs),
                       "z_min": round(float(min(zs)), 3),
                       "z_max": round(float(max(zs)), 3),
                       "z_median": round(float(np.median(zs)), 3)})
    return labels, layers


def assign_layer(z, peaks):
    """z[m] を最も近い層(代表 z)に割り当てる。層が無ければ -1。"""
    if not peaks:
        return -1
    best = 0
    bd = abs(z - peaks[0])
    for i, p in enumerate(peaks[1:], start=1):
        d = abs(z - p)
        if d < bd:
            bd = d
            best = i
    return best


def quantize_xyz(V, quant=QUANT):
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int16), np.zeros(3, dtype=np.int64)
    Q = np.round(V / quant).astype(np.int64)
    o = Q.min(axis=0)
    rel = Q - o
    if int(rel.max()) > 32767:
        raise ValueError("int16 overflow in ubld mesh")
    return rel.astype(np.int16), o


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
def run(gml_path, out_dir, origin=DEFAULT_ORIGIN, axis=(0, 1), merge_tol=0.30,
        merge_deg=8.0, layer_sep=2.0, layer_bin=0.5, layer_min_frac=0.03,
        min_seg_m=0.10, progress=True):
    gml_path = Path(gml_path)
    out_dir = Path(out_dir)
    ground0, g0src = _resolve_ground0(out_dir)
    lat_i, lon_i = axis
    triangulate_3d = _load_triangulate()
    if progress:
        print(f"[ubld] {gml_path.name}  ground0={ground0}({g0src})  "
              f"merge_tol={merge_tol}m/{merge_deg}deg  layer_sep={layer_sep}m",
              flush=True)

    t0 = time.time()
    counts = {"polygons": 0, "holes": 0, "untagged": 0, "degenerate": 0}
    kind_polys = {k: 0 for k in KIND_ORDER}
    kind_elems = {k: set() for k in KIND_ORDER}
    wall_segs = []
    wall_z = []
    closure_segs = []
    closure_z = []
    doors = {}                # elem -> [sx, sy, n, zmin, zmax]
    floor_z = []
    floor_area = []
    verts = []
    tris = []
    tri_kind = []
    vbase = 0
    n_wall_faces = 0
    n_wall_nonvertical = 0
    n_wall_too_short = 0

    stats = {}
    for rec in stream_lod4_polygons(gml_path, stats):
        if rec["hole"]:
            counts["holes"] += 1
            continue
        counts["polygons"] += 1
        tag = rec["tag"]
        kind = TAG_KIND.get(tag, "other")
        if tag is None:
            counts["untagged"] += 1
        kind_polys[kind] += 1
        kind_elems[kind].add(rec["elem"])
        A = np.asarray(rec["pts"], dtype=np.float64)
        e, n = latlon_to_local(A[:, lat_i], A[:, lon_i], *origin)
        P = np.column_stack([e, n, A[:, 2] - ground0])
        if P.shape[0] > 1 and np.array_equal(P[0], P[-1]):
            P = P[:-1]
        if P.shape[0] < 3:
            counts["degenerate"] += 1
            continue

        if kind == "interior_wall":
            n_wall_faces += 1
            if is_vertical(P):
                s = project_segment(P)
                if s is not None and segment_length(s) >= min_seg_m:
                    wall_segs.append(s)
                    wall_z.append((float(P[:, 2].min()), float(P[:, 2].max())))
                else:
                    n_wall_too_short += 1
            else:
                n_wall_nonvertical += 1
        elif kind == "closure":
            if is_vertical(P):
                s = project_segment(P)
                if s is not None and segment_length(s) >= min_seg_m:
                    closure_segs.append(s)
                    closure_z.append((float(P[:, 2].min()), float(P[:, 2].max())))
        elif kind == "door":
            d = doors.setdefault(rec["elem"],
                                 [0.0, 0.0, 0, float("inf"), float("-inf")])
            d[0] += float(P[:, 0].mean())
            d[1] += float(P[:, 1].mean())
            d[2] += 1
            d[3] = min(d[3], float(P[:, 2].min()))
            d[4] = max(d[4], float(P[:, 2].max()))
        elif kind == "floor":
            xy = P[:, :2]
            area = abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                             - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))) * 0.5
            floor_z.append(float(P[:, 2].mean()))
            floor_area.append(area)

        dedup, tri = triangulate_3d([tuple(v) for v in P])
        if tri:
            verts.extend(dedup)
            kid = KIND_ID[kind]
            for (i, j, k) in tri:
                tris.append((vbase + i, vbase + j, vbase + k))
                tri_kind.append(kid)
            vbase += len(dedup)

    if progress:
        print(f"[ubld] parsed {counts['polygons']} polygons  "
              f"{time.time() - t0:.1f}s", flush=True)

    # --- 床の面積重み付きヒストグラムで層分離(壁の dedup より **先**)
    peaks, zhist = layer_peaks(floor_z, floor_area, layer_bin, layer_sep,
                               layer_min_frac)

    # --- 壁線分の重複除去は **層ごと** に行う
    # 平面投影だけで潰すと、吹き抜け/階段室の壁のように複数フロアで同じ平面位置にある壁が
    # 1 本に潰れ、「そのフロアの壁集合」が欠ける。2D SFM はフロア単位で回すので、
    # 先に層へ割り当ててから層内で dedup する。
    t0 = time.time()
    uniq_walls, wz, wall_layer = merge_by_layer(wall_segs, wall_z, peaks,
                                                merge_tol, merge_deg)
    uniq_closures, cz, closure_layer = merge_by_layer(closure_segs, closure_z, peaks,
                                                      merge_tol, merge_deg)
    # 参考値: 層を無視した平面投影だけの一意本数(層をまたぐ重複がどれだけあるか)
    flat_walls, _ = merge_segments(wall_segs, merge_tol, merge_deg)
    flat_closures, _ = merge_segments(closure_segs, merge_tol, merge_deg)
    if progress:
        print(f"[ubld] wall segments {len(wall_segs)} -> {len(uniq_walls)} unique "
              f"(層無視なら {len(flat_walls)})  "
              f"closure {len(closure_segs)} -> {len(uniq_closures)}  "
              f"{time.time() - t0:.1f}s", flush=True)
    layers = [{"layer": i, "z": round(p, 3), "n_floor_polys": 0,
               "floor_area_m2": 0.0, "z_min": None, "z_max": None}
              for i, p in enumerate(peaks)]
    for z, a in zip(floor_z, floor_area):
        li = assign_layer(z, peaks)
        if li < 0:
            continue
        ly = layers[li]
        ly["n_floor_polys"] += 1
        ly["floor_area_m2"] += a
        ly["z_min"] = z if ly["z_min"] is None else min(ly["z_min"], z)
        ly["z_max"] = z if ly["z_max"] is None else max(ly["z_max"], z)
    for ly in layers:
        ly["floor_area_m2"] = round(ly["floor_area_m2"], 2)
        ly["z_min"] = None if ly["z_min"] is None else round(ly["z_min"], 3)
        ly["z_max"] = None if ly["z_max"] is None else round(ly["z_max"], 3)

    door_pts = []
    for elem in sorted(doors):
        sx, sy, cnt, zmin, zmax = doors[elem]
        door_pts.append((sx / cnt, sy / cnt, zmin, zmax, cnt))
    door_layer = [assign_layer(d[2], peaks) for d in door_pts]

    # --- 表示メッシュ
    V = np.asarray(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    Q, oq = quantize_xyz(V)
    F = np.asarray(tris, dtype=np.int64).reshape(-1, 3) if tris \
        else np.zeros((0, 3), dtype=np.int64)
    mesh = {
        "xyz": Q, "origin_q": oq.astype(np.int64),
        "tri": F.astype(np.uint32),
        "tri_kind": np.asarray(tri_kind, dtype=np.uint8),
        "kind_names": np.asarray(KIND_ORDER),
    }
    det_npz.save_npz(out_dir / "ubld_lod4_mesh.npz", mesh)
    mesh_bytes = (out_dir / "ubld_lod4_mesh.npz").stat().st_size

    def _r(v, nd=3):
        return round(float(v), nd)

    meta = {
        "schema": "plateau_ubld_lod4/1",
        "crs": "local-m", "axes": "X=east,Y=north,Z=up",
        "origin_latlon": [origin[0], origin[1]],
        "ground0": round(float(ground0), 4), "ground0_source": g0src,
        "scope": ("渋谷駅周辺地下街のみ(約 332m×410m)。シム bbox 全域の屋内は存在しない = "
                  "屋内 SFM は地下街ユースケースに限定される"),
        "params": {"merge_tol_m": merge_tol, "merge_angle_deg": merge_deg,
                   "vertical_max_cos": 0.2, "layer_sep_m": layer_sep,
                   "layer_bin_m": layer_bin, "layer_min_frac": layer_min_frac,
                   "min_segment_m": min_seg_m, "quant_scale": QUANT},
        "counts": counts,
        "polygons_by_kind": kind_polys,
        "elements_with_geometry_by_kind": {k: len(v) for k, v in kind_elems.items()},
        "elements_in_file_by_tag": dict(sorted(stats.get("elements_seen", {}).items())),
        "walls": {
            "n_elements_in_file": stats.get("elements_seen", {}).get(
                "InteriorWallSurface", 0),
            "n_elements_with_geometry": len(kind_elems["interior_wall"]),
            "n_elements_opening_only": (
                stats.get("elements_seen", {}).get("InteriorWallSurface", 0)
                - len(kind_elems["interior_wall"])),
            "opening_only_note": ("自前の面を持たず bldg:opening(Door/Window)だけの "
                                  "InteriorWallSurface = 出入口そのもの。壁線分には出ない"),
            "n_faces": n_wall_faces,
            "n_non_vertical_faces": n_wall_nonvertical,
            "n_vertical_faces_below_min_length": n_wall_too_short,
            "face_accounting": ("n_faces = 非鉛直 + 短すぎ(< min_segment_m) + n_segments_raw"),
            "n_segments_raw": len(wall_segs),
            "n_segments_unique": len(uniq_walls),
            "n_segments_unique_flat": len(flat_walls),
            "dedup_scope": ("層ごとに dedup(2D SFM がフロア単位で回るため)。"
                            "flat = 層を無視した平面投影だけの一意本数"),
            "n_segments_by_layer": [sum(1 for x in wall_layer if x == i)
                                    for i in range(max(1, len(peaks)))],
            "total_length_m": _r(sum(segment_length(s) for s in uniq_walls), 1),
        },
        "closures": {
            "n_elements": len(kind_elems["closure"]),
            "n_segments_raw": len(closure_segs),
            "n_segments_unique": len(uniq_closures),
            "n_segments_unique_flat": len(flat_closures),
        },
        "layers": layers,
        "layer_method": ("area-weighted z histogram peak picking "
                         "(gap clustering fails: stairs/ramps connect all levels)"),
        "floor_z_area_histogram": zhist,
        "doors": {"n_elements": len(door_pts),
                  "n_polygons": kind_polys["door"]},
        "mesh": {"file": "ubld_lod4_mesh.npz", "bytes": mesh_bytes,
                 "n_vertices": int(Q.shape[0]), "n_triangles": int(F.shape[0]),
                 "kind_names": list(KIND_ORDER),
                 "dequant": "xyz_m = (xyz + origin_q) * quant_scale"},
        "wall_segments": [
            {"xy": [_r(s[0], 2), _r(s[1], 2), _r(s[2], 2), _r(s[3], 2)],
             "z": [_r(wz[i][0], 2), _r(wz[i][1], 2)], "layer": wall_layer[i]}
            for i, s in enumerate(uniq_walls)],
        "closure_segments": [
            {"xy": [_r(s[0], 2), _r(s[1], 2), _r(s[2], 2), _r(s[3], 2)],
             "z": [_r(cz[i][0], 2), _r(cz[i][1], 2)], "layer": closure_layer[i]}
            for i, s in enumerate(uniq_closures)],
        "gates": [
            {"xy": [_r(d[0], 2), _r(d[1], 2)],
             "z": [_r(d[2], 2), _r(d[3], 2)], "n_polys": d[4],
             "layer": door_layer[i]}
            for i, d in enumerate(door_pts)],
        "source": {"file": gml_path.name, "bytes": gml_path.stat().st_size,
                   "sha256": sha256_file(gml_path)},
        "attribution": ("国土交通省 Project PLATEAU「3D都市モデル 渋谷区(2025年度)」"
                        "CityGML(udx/ubld・LOD4.1)を加工して作成"),
    }
    det_npz.dump_json(out_dir / "ubld_lod4.json", meta, indent=1)
    js_bytes = (out_dir / "ubld_lod4.json").stat().st_size
    if progress:
        print(f"[ubld] layers={len(layers)}  "
              f"{[(ly['layer'], ly['z'], ly['floor_area_m2']) for ly in layers]}",
              flush=True)
        print(f"[ubld] walls_unique={len(uniq_walls)} "
              f"(total {meta['walls']['total_length_m']} m)  "
              f"closures={len(uniq_closures)}  gates={len(door_pts)}", flush=True)
        print(f"       {out_dir / 'ubld_lod4.json'} ({js_bytes / 1e6:.2f} MB)  "
              f"{out_dir / 'ubld_lod4_mesh.npz'} ({mesh_bytes / 1e6:.2f} MB)", flush=True)
    return meta


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PLATEAU ubld LOD4.1 → 壁線分+層+ゲート")
    ap.add_argument("--gml", default=str(DEFAULT_GML))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "plateau"))
    ap.add_argument("--merge-tol", type=float, default=0.30)
    ap.add_argument("--merge-deg", type=float, default=8.0)
    ap.add_argument("--layer-sep", type=float, default=2.0,
                    help="層の最小間隔[m](面積ヒストグラムの山の分離)")
    ap.add_argument("--layer-bin", type=float, default=0.5)
    ap.add_argument("--layer-min-frac", type=float, default=0.03)
    ap.add_argument("--min-seg", type=float, default=0.10)
    a = ap.parse_args(argv)
    run(a.gml, a.out_dir, DEFAULT_ORIGIN, (0, 1), a.merge_tol, a.merge_deg,
        a.layer_sep, a.layer_bin, a.layer_min_frac, a.min_seg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
