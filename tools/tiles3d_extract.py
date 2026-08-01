"""PLATEAU 3D Tiles(b3dm/Draco/WebP)→ シミュ local-m 量子化タイル抽出。【松案 A-1】

`docs/plans/highfidelity-3d-physics-plan.md` 第3段=松のデータパイプライン。
手元の **未展開 zip**(608 MiB)を `zipfile` で直読みし、シム bbox と交差する
`bldg_..._lod2`(テクスチャ付き LOD2.2)タイルを取り出して

  - ジオメトリ: Draco 解凍 → CESIUM_RTC 適用 → ECEF → WGS84 → 局所接平面(local-m)
                → **int16 × 0.05 m 量子化**(既存 `plateau_web.json` / `tracks_bin` と同一格子)
  - UV:         uint16 正規化(0..65535)
  - テクスチャ: 1 タイル 1 枚の WebP アトラスを 1/2 解像度で再エンコード
  - 属性:       batchTable(gml_id / bldg:usage / storeysAboveGround / address …)

を `data/plateau/tiles_lod2/` へ書き出す。

**REPLACE 階層の注意**: この tileset は `refine=REPLACE` なので、bbox と交差する 148 タイルには
祖先タイルが持つ **同一建物の低精細版** が混ざる(実測 7,164 batch / 一意 gml_id 6,478 /
重複 686 は全て祖先↔子孫関係)。そのまま全部描くと 686 棟が二重に出る。各タイルの npz に
`batch_shadowed`(= もっと深いタイルに同じ gml_id がある batch 番号)を入れてあるので、
描画側はその batch の三角形を捨てれば重複が消える。

**依存の隔離**: 本ファイルだけが `smtk_draco` と `Pillow` に依存する。
本線パイプライン(`scripts/`)は stdlib+numpy の原則を維持し、成果物(npz/webp/json)
だけを読む。`tools/` に置いてあるのはそのため(調査 §7 松 の緩和策)。

    pip install smtk_draco pillow

使い方:
    python tools/tiles3d_extract.py                       # 既定(148タイル・アトラス1/2)
    python tools/tiles3d_extract.py --limit 3 --out-dir /tmp/x    # 動作確認
    python tools/tiles3d_extract.py --no-atlas            # ジオメトリだけ
    python tools/tiles3d_extract.py --calibrate-geoid     # ジオイド高の実測(要 plateau_index.json)

実測(2026-08-02・148 タイル):
    V 2,122,356 / F 707,452 / batch 7,164(一意 gml_id 6,478・影 686)/ テクスチャ付き 110 枚
    Draco 解凍 **1.1 秒**(148 枚合計)・アトラス再エンコード 49.7 秒・全体 53 秒
    npz 10.87 MB + WebP(1/2)33.16 MB = 44.16 MB。1/4 なら WebP 9.93 MB
    CityGML 経路との突合(9,676 棟): 水平ずれ 中央値 0.1 mm / |誤差| p95 5.0 mm

座標系(scripts/plateau_extract.py と厳密に同一):
    原点 (lat0, lon0) = (35.65950, 139.70062) … スクランブル交差点
    E = (lon - lon0) * 111320.0 * cos(lat0)
    N = (lat - lat0) * 111132.9
    Z = 標高 - ground0(= 15.18, plateau_index.json の DEM 実測値)
    ただし 3D Tiles の高さは **楕円体高** なのでジオイド高 GEOID_M を差し引いてから
    ground0 を引く(§ GEOID_M の来歴コメント参照)。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_det_npz():
    """scripts/det_npz.py を importlib で直接ロードする(scripts が package ではないため。
    sys.path に scripts/ を足すと共有 pytest プロセスで他モジュールを覆う危険があるので採らない。
    tests/test_export3d.py・plateau_extract._load_triangulate と同方式)。"""
    import importlib.util
    p = REPO_ROOT / "scripts" / "det_npz.py"
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
PLATEAU_QUANT = 0.05          # scripts/export_3d.py PLATEAU_QUANT と同一 [m/単位]

# GRS80(JGD2011)= WGS84 と実質同一(扁平率の差 ~1e-12)
ELL_A = 6378137.0
ELL_F = 1.0 / 298.257223563

# 3D Tiles の z は楕円体高。CityGML(標高 T.P.)との差 = ジオイド高。
# 来歴: 本ツールの --calibrate-geoid を data/plateau/plateau_index.json(CityGML 由来
# 6,311 棟)に対して実行し、gml_id 一致した建物の (楕円体高最大 − 標高最大) を実測。
#   148 タイル / 突合 9,676 件 / うち core(最頻ビン ±0.25 m)6,484 件
#   core 中央値 36.7905 m・p05..p95 = 36.7242..36.8538(渋谷 bbox 2.9 km² 内でほぼ一定)
# 同時に水平ずれも実測: dE 中央値 -0.08 mm / dN -0.03 mm・|誤差| p95 = 5.0 mm
# = 3D Tiles 経路と CityGML 経路の局所座標が **ミリ精度で一致** している証拠。
GEOID_M = 36.7905

DEFAULT_ZIP = Path(
    r"C:\Users\塚本翔太\Desktop\13113_shibuya-ku_pref_2025_3dtiles_mvt_1_op.zip")
DEFAULT_TILESET = ("13113_shibuya-ku_pref_2025_citygml_1_op_bldg_3dtiles_"
                   "13113_shibuya-ku_lod2")
# シム bbox(data/plateau/plateau_index.json params.latlon_bbox と同一)
DEFAULT_BBOX = (35.65260195270707, 139.68931725223695,
                35.66645113688206, 139.7100199766684)

# batchTable から拾う属性(既定)。"attributes" は他キーの入れ子重複で 30.7 MB あるため既定で除外。
CURATED_ATTRS = (
    "gml_id", "meshcode", "feature_type", "gml:name", "core:creationDate",
    "bldg:class", "bldg:usage", "bldg:measuredHeight",
    "bldg:storeysAboveGround", "bldg:storeysBelowGround", "bldg:address",
    "uro:BuildingIDAttribute_uro:buildingID", "uro:lodType",
    "uro:BuildingDetailAttribute_uro:buildingRoofEdgeArea",
    "uro:BuildingDetailAttribute_uro:urbanPlanType",
    "uro:BuildingDetailAttribute_uro:landUseType",
)

GLTF_COMPONENT_NP = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
GLTF_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
              "MAT2": 4, "MAT3": 9, "MAT4": 16}


# =============================================================== 座標変換
def latlon_to_local(lat, lon, lat0=DEFAULT_ORIGIN[0], lon0=DEFAULT_ORIGIN[1]):
    """局所接平面近似(EPSG:6697 → local-m)。scripts/plateau_extract.latlon_to_local と同式。
    スカラ/ndarray どちらでも動く。"""
    e = (lon - lon0) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    n = (lat - lat0) * M_PER_DEG_LAT
    return e, n


def ecef_to_geodetic(x, y, z, iters=5):
    """ECEF(m)→ 測地緯度[deg]・経度[deg]・楕円体高[m]。Bowring 初期値 + 固定回数反復。

    反復回数を固定するのは決定論のため(収束判定で回数が入力依存に変わらない)。
    渋谷の高度域(±300 m)では 3 回で 1e-9 m 未満に収束する。"""
    a = ELL_A
    b = a * (1.0 - ELL_F)
    e2 = ELL_F * (2.0 - ELL_F)
    ep2 = (a * a - b * b) / (b * b)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.hypot(x, y)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3,
                     p - e2 * a * np.cos(th) ** 3)
    for _ in range(iters):
        sn = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sn * sn)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))
    sn = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sn * sn)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """測地座標 → ECEF(m)。ecef_to_geodetic の逆(往復テスト用)。"""
    a = ELL_A
    e2 = ELL_F * (2.0 - ELL_F)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - e2) + h) * np.sin(lat)
    return x, y, z


def gltf_ypup_to_ecef(P, rtc_center):
    """glTF(Y-up)頂点 → ECEF。3D Tiles の Y_UP_TO_Z_UP(X 軸 +90°回転)+ CESIUM_RTC 平行移動。
    (x, y, z)_gltf → (x, -z, y) + center。"""
    P = np.asarray(P, dtype=np.float64)
    c = np.asarray(rtc_center, dtype=np.float64)
    return np.column_stack([P[:, 0], -P[:, 2], P[:, 1]]) + c


def tile_positions_to_local(P, rtc_center, origin=DEFAULT_ORIGIN,
                            ground0=DEFAULT_GROUND0, geoid=GEOID_M):
    """glTF 頂点(N,3・Y-up・RTC 相対)→ local-m (E, N, Z)。Z = 楕円体高 − geoid − ground0。"""
    E = gltf_ypup_to_ecef(P, rtc_center)
    lat, lon, h = ecef_to_geodetic(E[:, 0], E[:, 1], E[:, 2])
    e, n = latlon_to_local(lat, lon, origin[0], origin[1])
    return np.column_stack([e, n, h - geoid - ground0])


# =============================================================== b3dm / glTF
def b3dm_split(buf):
    """b3dm → (featureTableJSON, featureTableBIN, batchTableJSON, batchTableBIN, glb)。

    b3dm ヘッダ = magic(4) version(4) byteLength(4) ftJSONlen(4) ftBINlen(4)
                  btJSONlen(4) btBINlen(4) = 28 B。以降 4 ブロックが並び、残りが素の GLB。"""
    if len(buf) < 28:
        raise ValueError("b3dm too short")
    magic, version, blen, ftjl, ftbl, btjl, btbl = struct.unpack_from("<4sIIIIII", buf, 0)
    if magic != b"b3dm":
        raise ValueError(f"not b3dm: {magic!r}")
    off = 28
    ftj = bytes(buf[off:off + ftjl]); off += ftjl
    ftb = bytes(buf[off:off + ftbl]); off += ftbl
    btj = bytes(buf[off:off + btjl]); off += btjl
    btb = bytes(buf[off:off + btbl]); off += btbl
    glb = bytes(buf[off:blen])
    return (json.loads(ftj) if ftj.strip() else {}), ftb, \
           (json.loads(btj) if btj.strip() else {}), btb, glb


def glb_split(glb):
    """GLB → (glTF JSON dict, BIN chunk bytes)。"""
    magic, version, total = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF":
        raise ValueError(f"not glb: {magic!r}")
    off = 12
    js = None
    binc = b""
    while off + 8 <= total:
        clen, ctype = struct.unpack_from("<II", glb, off)
        off += 8
        data = glb[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:      # 'JSON'
            js = json.loads(data)
        elif ctype == 0x004E4942:    # 'BIN\0'
            binc = bytes(data)
    if js is None:
        raise ValueError("glb has no JSON chunk")
    return js, binc


def bufferview_bytes(js, binc, index):
    bv = js["bufferViews"][index]
    o = bv.get("byteOffset", 0)
    return binc[o:o + bv["byteLength"]]


def _alloc(nbytes):
    """smtk_draco の copy_* は Cython 側で `bytes` 型を要求し、その領域へ直接書き込む
    (Blender の Draco ブリッジと同じ契約)。長さ 1 の bytes は CPython がキャッシュを
    返して他所を壊しうるので、2 バイト未満は許可しない。"""
    if nbytes < 2:
        raise ValueError(f"refusing to allocate {nbytes}B scratch bytes")
    return bytes(nbytes)


def draco_decode_primitive(js, binc, prim, decoder_factory=None):
    """KHR_draco_mesh_compression のプリミティブを解凍して
    {"indices": (M*3,), "POSITION": (N,3), "TEXCOORD_0": (N,2), ...} を返す。"""
    if decoder_factory is None:
        import smtk_draco
        decoder_factory = smtk_draco.Decoder
    ext = prim["extensions"]["KHR_draco_mesh_compression"]
    data = bytes(bufferview_bytes(js, binc, ext["bufferView"]))
    dec = decoder_factory()
    if not dec.decode(data):
        raise RuntimeError("draco decode failed")
    out = {"n_vertices": dec.get_vertex_count(), "n_indices": dec.get_index_count()}
    iacc = js["accessors"][prim["indices"]]
    ict = iacc["componentType"]
    if not dec.read_indices(ict):
        raise RuntimeError("draco read_indices failed")
    ibuf = _alloc(dec.get_index_byte_length())
    dec.copy_indices(ibuf)
    out["indices"] = np.frombuffer(ibuf, dtype=GLTF_COMPONENT_NP[ict]).copy()
    for name, aidx in prim["attributes"].items():
        did = ext["attributes"][name]
        acc = js["accessors"][aidx]
        if not dec.read_attribute(did, acc["componentType"], acc["type"]):
            raise RuntimeError(f"draco read_attribute failed: {name}")
        buf = _alloc(dec.get_attribute_byte_length(did))
        dec.copy_attribute(did, buf)
        ncomp = GLTF_NCOMP[acc["type"]]
        arr = np.frombuffer(buf, dtype=GLTF_COMPONENT_NP[acc["componentType"]])
        out[name] = arr.reshape(-1, ncomp).copy()
    return out


# =============================================================== 量子化
def quantize_xyz(V, quant=PLATEAU_QUANT):
    """local-m 頂点 (N,3) → (Q, origin_q, dtype_name)。

    復元式: xyz[m] = (Q + origin_q) * quant。origin_q は整数なので **量子化格子は
    plateau_web.json / tracks_bin と同一**(0.05 m のグローバル格子上に乗る)。
    タイル 1 枚が int16 の幅(65535 単位 = 3276 m)を超える場合だけ int32 に落とす
    (実データでは 148 枚中 1 枚=ルートタイルのみ 5.6 km 幅で該当)。"""
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 0:
        return (np.zeros((0, 3), dtype=np.int16), np.zeros(3, dtype=np.int64), "int16")
    Q = np.round(V / quant).astype(np.int64)
    origin = Q.min(axis=0)
    rel = Q - origin
    if int(rel.max()) <= 32767:
        return rel.astype(np.int16), origin, "int16"
    return rel.astype(np.int32), origin, "int32"


def dequantize_xyz(Q, origin_q, quant=PLATEAU_QUANT):
    return (Q.astype(np.int64) + np.asarray(origin_q, dtype=np.int64)) * quant


def quantize_uv(UV):
    """TEXCOORD_0 float32 (N,2) → uint16。復元式: uv = q / 65535。
    サンプラは CLAMP_TO_EDGE(33071)なので [0,1] 外は元々クランプされる=先に丸めてよい。"""
    if UV is None or len(UV) == 0:
        return np.zeros((0, 2), dtype=np.uint16), 0
    A = np.asarray(UV, dtype=np.float64)
    out_of_range = int(((A < 0.0) | (A > 1.0)).any(axis=1).sum())
    A = np.clip(A, 0.0, 1.0)
    return np.round(A * 65535.0).astype(np.uint16), out_of_range


# =============================================================== tileset 走査
def walk_tileset(root):
    """tileset.json の root → 深さ優先のノード列。各要素:
    {uri, region(rad 6要素), depth, parent, n_children, geometric_error}。"""
    nodes = []

    def rec(node, depth, parent):
        me = {
            "uri": (node.get("content") or {}).get("uri"),
            "region": (node.get("boundingVolume") or {}).get("region"),
            "depth": depth,
            "parent": parent,
            "n_children": len(node.get("children") or []),
            "geometric_error": node.get("geometricError"),
        }
        nodes.append(me)
        i = len(nodes) - 1
        for ch in node.get("children") or []:
            rec(ch, depth + 1, i)

    rec(root, 0, -1)
    return nodes


def region_latlon(region):
    """boundingVolume.region(rad: west,south,east,north,hmin,hmax)→ (s,w,n,e) [deg]。"""
    w, s, e, n = (math.degrees(region[i]) for i in range(4))
    return s, w, n, e


def intersects_bbox(region, bbox):
    """bbox = (lat_min, lon_min, lat_max, lon_max)。境界接触も交差とみなす。"""
    s, w, n, e = region_latlon(region)
    lat0, lon0, lat1, lon1 = bbox
    return not (e < lon0 or w > lon1 or n < lat0 or s > lat1)


def select_tiles(nodes, bbox):
    """bbox と交差し content を持つノードの index 列(深さ優先順=決定論)。"""
    return [i for i, nd in enumerate(nodes)
            if nd["uri"] and nd["region"] and intersects_bbox(nd["region"], bbox)]


def ancestors(nodes, i):
    out = []
    p = nodes[i]["parent"]
    while p >= 0:
        out.append(p)
        p = nodes[p]["parent"]
    return out


def shadow_map(nodes, selected, gml_ids_by_tile):
    """3D Tiles の refine=REPLACE 階層では **同じ gml_id が祖先タイルにも低精細で入る**。
    選択集合内で同じ gml_id を持つタイルのうち **最も深い(=高精細)** ものだけを残し、
    それ以外を「影(shadowed)」として印を付ける。深さ同点なら選択順の先勝ち。

    戻り値: {tile_index: set(batch_index)}(= 描画時に飛ばすべき batch)。"""
    best = {}
    for ti in selected:
        d = nodes[ti]["depth"]
        for g in gml_ids_by_tile.get(ti, []):
            cur = best.get(g)
            if cur is None or d > cur[0]:
                best[g] = (d, ti)
    shadowed = {}
    for ti in selected:
        s = set()
        for bi, g in enumerate(gml_ids_by_tile.get(ti, [])):
            if best.get(g, (None, ti))[1] != ti:
                s.add(bi)
        shadowed[ti] = s
    return shadowed


# =============================================================== 本体
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _resolve_ground0(explicit):
    if explicit is not None:
        return float(explicit), "cli"
    p = REPO_ROOT / "data" / "plateau" / "plateau_index.json"
    if p.exists():
        try:
            g = json.loads(p.read_text(encoding="utf-8")).get("ground0")
            if g is not None:
                return float(g), "plateau_index.json"
        except Exception:
            pass
    return DEFAULT_GROUND0, "default"


def _atlas_bytes(js, binc):
    """images[0] の WebP バイト列(無ければ None)。"""
    imgs = js.get("images") or []
    if not imgs:
        return None
    im = imgs[0]
    if "bufferView" not in im:
        return None
    return bytes(bufferview_bytes(js, binc, im["bufferView"]))


def reencode_atlas(raw, scale=0.5, quality=85, method=6):
    """WebP バイト列 → (縮小 WebP バイト列, (w0,h0), (w1,h1))。Pillow のみ使用。"""
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    w0, h0 = im.size
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    w1 = max(1, int(round(w0 * scale)))
    h1 = max(1, int(round(h0 * scale)))
    if (w1, h1) != (w0, h0):
        im = im.resize((w1, h1), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=quality, method=method)
    return buf.getvalue(), (w0, h0), (w1, h1)


def extract_tile(raw_b3dm, origin, ground0, geoid, decoder_factory=None):
    """1 タイル分の b3dm バイト列 → 量子化済みメッシュ dict + batchTable + アトラス原バイト。"""
    ft, _ftb, bt, _btb, glb = b3dm_split(raw_b3dm)
    js, binc = glb_split(glb)
    rtc = (js.get("extensions") or {}).get("CESIUM_RTC", {}).get("center", [0.0, 0.0, 0.0])
    prims = (js.get("meshes") or [{}])[0].get("primitives") or []
    V_parts, UV_parts, B_parts, T_parts = [], [], [], []
    prim_tri_off = [0]
    prim_tex = []
    uv_out_of_range = 0
    vbase = 0
    for prim in prims:
        d = draco_decode_primitive(js, binc, prim, decoder_factory)
        P = d["POSITION"].astype(np.float64)
        L = tile_positions_to_local(P, rtc, origin, ground0, geoid)
        idx = d["indices"].astype(np.int64).reshape(-1, 3) + vbase
        V_parts.append(L)
        T_parts.append(idx)
        if "TEXCOORD_0" in d:
            UV_parts.append(d["TEXCOORD_0"])
            prim_tex.append(1)
        else:
            UV_parts.append(np.zeros((P.shape[0], 2), dtype=np.float32))
            prim_tex.append(0)
        bid = d.get("_BATCHID")
        B_parts.append(np.zeros(P.shape[0], dtype=np.int64) if bid is None
                       else bid.reshape(-1).astype(np.int64))
        vbase += P.shape[0]
        prim_tri_off.append(prim_tri_off[-1] + idx.shape[0])
    if V_parts:
        V = np.vstack(V_parts)
        UVf = np.vstack(UV_parts)
        T = np.vstack(T_parts)
        B = np.concatenate(B_parts)
    else:
        V = np.zeros((0, 3)); UVf = np.zeros((0, 2)); T = np.zeros((0, 3), np.int64)
        B = np.zeros(0, np.int64)
    Q, origin_q, qdt = quantize_xyz(V)
    UV, uv_out_of_range = quantize_uv(UVf)
    mesh = {
        "xyz": Q,
        "origin_q": origin_q.astype(np.int64),
        "uv": UV,
        "tri": T.astype(np.uint32),
        "batch": B.astype(np.uint16),
        "prim_tri_offsets": np.asarray(prim_tri_off, dtype=np.int32),
        "prim_textured": np.asarray(prim_tex, dtype=np.uint8),
    }
    meta = {
        "batch_length": int(ft.get("BATCH_LENGTH", 0)),
        "n_vertices": int(V.shape[0]),
        "n_triangles": int(T.shape[0]),
        "n_primitives": len(prims),
        "xyz_dtype": qdt,
        "uv_out_of_range": uv_out_of_range,
        "local_bbox": ([round(float(V[:, i].min()), 3) for i in range(3)]
                       + [round(float(V[:, i].max()), 3) for i in range(3)]
                       if V.shape[0] else None),
        "rtc_center": [float(v) for v in rtc],
    }
    return mesh, meta, bt, _atlas_bytes(js, binc)


def run(zip_path, tileset, out_dir, bbox, origin, ground0, geoid,
        atlas_scale=0.5, atlas_quality=85, limit=None, do_atlas=True,
        all_attrs=False, progress=True, attrs_out=None):
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    attrs_out = Path(attrs_out) if attrs_out else \
        (REPO_ROOT / "data" / "plateau" / "tiles_batch_attrs.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    zf = zipfile.ZipFile(zip_path)
    ts = json.loads(zf.read(f"{tileset}/tileset.json"))
    nodes = walk_tileset(ts["root"])
    selected = select_tiles(nodes, bbox)
    if limit:
        selected = selected[:limit]
    if progress:
        print(f"[tiles3d] tileset={tileset}", flush=True)
        print(f"[tiles3d] nodes={len(nodes)}  selected={len(selected)}  "
              f"bbox={bbox}", flush=True)

    # --- pass 1: batchTable(gml_id)だけ先に読み、REPLACE 階層の影を確定する
    gml_by_tile = {}
    for ti in selected:
        raw = zf.read(f"{tileset}/{nodes[ti]['uri']}")
        _ft, _fb, bt, _bb, _glb = b3dm_split(raw)
        gml_by_tile[ti] = list(bt.get("gml_id") or [])
    shadow = shadow_map(nodes, selected, gml_by_tile)

    keys = None if all_attrs else set(CURATED_ATTRS)
    tiles_meta = []
    attrs_by_tile = {}
    t_draco = 0.0
    t_atlas = 0.0
    n_npz_bytes = 0
    n_atlas_bytes = 0
    n_atlas_src_bytes = 0
    for k, ti in enumerate(selected):
        nd = nodes[ti]
        raw = zf.read(f"{tileset}/{nd['uri']}")
        t0 = time.time()
        mesh, meta, bt, atlas_raw = extract_tile(raw, origin, ground0, geoid)
        t_draco += time.time() - t0
        sh = sorted(shadow.get(ti, ()))
        mesh["batch_shadowed"] = np.asarray(sh, dtype=np.uint16)
        npz_name = f"tile_{k:03d}.npz"
        det_npz.save_npz(out_dir / npz_name, mesh)
        npz_bytes = (out_dir / npz_name).stat().st_size
        n_npz_bytes += npz_bytes

        atlas_name = None
        atlas_bytes = 0
        src_size = dst_size = None
        if atlas_raw is not None:
            n_atlas_src_bytes += len(atlas_raw)
            if do_atlas:
                t0 = time.time()
                enc, src_size, dst_size = reencode_atlas(
                    atlas_raw, atlas_scale, atlas_quality)
                t_atlas += time.time() - t0
                atlas_name = f"atlas_{k:03d}.webp"
                (out_dir / atlas_name).write_bytes(enc)
                atlas_bytes = len(enc)
                n_atlas_bytes += atlas_bytes

        s, w, n, e = region_latlon(nd["region"])
        tm = {
            "id": k, "uri": nd["uri"], "node": ti, "depth": nd["depth"],
            "geometric_error": nd["geometric_error"],
            "latlon_bbox": [round(s, 8), round(w, 8), round(n, 8), round(e, 8)],
            "height_range": [round(nd["region"][4], 3), round(nd["region"][5], 3)],
            "npz": npz_name, "npz_bytes": npz_bytes,
            "atlas": atlas_name, "atlas_bytes": atlas_bytes,
            "atlas_src_bytes": len(atlas_raw) if atlas_raw is not None else 0,
            "atlas_src_size": list(src_size) if src_size else None,
            "atlas_size": list(dst_size) if dst_size else None,
            "n_shadowed_batches": len(sh),
            "b3dm_bytes": len(raw),
        }
        tm.update(meta)
        tiles_meta.append(tm)

        sub = {}
        for kk, vv in bt.items():
            if kk == "attributes":
                continue
            if keys is not None and kk not in keys:
                continue
            sub[kk] = vv
        sub["_shadowed_batches"] = sh
        attrs_by_tile[npz_name] = sub

        if progress and (k % 20 == 0 or k == len(selected) - 1):
            print(f"  [{k + 1}/{len(selected)}] {nd['uri']}  "
                  f"V={meta['n_vertices']} T={meta['n_triangles']} "
                  f"npz={npz_bytes / 1024:.0f}KB atlas={atlas_bytes / 1024:.0f}KB",
                  flush=True)

    if progress:
        print("[tiles3d] hashing source zip (608MiB) ...", flush=True)
    zsha = sha256_file(zip_path)

    all_gml = [g for ti in selected for g in gml_by_tile[ti]]
    n_shadow = sum(len(shadow[ti]) for ti in selected)
    index = {
        "schema": "plateau_tiles_lod2/1",
        "crs": "local-m", "axes": "X=east,Y=north,Z=up",
        "quant_scale": PLATEAU_QUANT,
        "uv_scale": 65535,
        "dequant": "xyz_m = (xyz + origin_q) * quant_scale ; uv = uv / uv_scale",
        "origin_latlon": [origin[0], origin[1]],
        "ground0": round(float(ground0), 4),
        "geoid_m": geoid,
        "latlon_bbox": list(bbox),
        "n_tiles": len(tiles_meta),
        "n_vertices": sum(t["n_vertices"] for t in tiles_meta),
        "n_triangles": sum(t["n_triangles"] for t in tiles_meta),
        "n_batches": sum(t["batch_length"] for t in tiles_meta),
        "n_unique_gml_id": len(set(all_gml)),
        "n_shadowed_batches": n_shadow,
        "n_textured_tiles": sum(1 for t in tiles_meta if t["atlas_src_bytes"]),
        "bytes": {
            "npz_total": n_npz_bytes,
            "atlas_total": n_atlas_bytes,
            "atlas_source_total": n_atlas_src_bytes,
            "b3dm_source_total": sum(t["b3dm_bytes"] for t in tiles_meta),
        },
        "atlas": {"scale": atlas_scale, "quality": atlas_quality,
                  "format": "WEBP", "enabled": bool(do_atlas)},
        "source": {
            "zip": zip_path.name,
            "zip_sha256": zsha,
            "zip_bytes": zip_path.stat().st_size,
            "tileset": tileset,
            "tileset_nodes": len(nodes),
        },
        "attribution": ("国土交通省 Project PLATEAU「3D都市モデル 渋谷区(2025年度)」"
                        "3D Tiles 版を加工して作成"),
        "tiles": tiles_meta,
    }
    det_npz.dump_json(out_dir / "index.json", index, indent=1)
    det_npz.dump_json(attrs_out,
                      {"schema": "plateau_tiles_batch_attrs/1",
                       "source_tileset": tileset,
                       "keys": "all" if all_attrs else list(CURATED_ATTRS),
                       "note": ("batchTable の 'attributes' キーは他キーの入れ子重複"
                                "(148タイルで 30.7MB)なので常に除外"),
                       "tiles": attrs_by_tile})
    if progress:
        el = time.time() - t_all
        print(f"[tiles3d] done {el:.1f}s  (draco {t_draco:.1f}s / atlas {t_atlas:.1f}s)",
              flush=True)
        print(f"[tiles3d] npz {n_npz_bytes / 1e6:.1f} MB  "
              f"atlas {n_atlas_bytes / 1e6:.1f} MB (src {n_atlas_src_bytes / 1e6:.1f} MB)",
              flush=True)
    return index


# =============================================================== ジオイド較正
def calibrate_geoid(zip_path, tileset, bbox, origin, ground0, index_json=None,
                    max_tiles=None):
    """CityGML 由来 plateau_index.json と gml_id で突合し、(楕円体高 − 標高) を実測する。
    戻り値: {n, median, p25, p75, dE_median, dN_median, ...}。GEOID_M の来歴根拠。"""
    index_json = index_json or (REPO_ROOT / "data" / "plateau" / "plateau_index.json")
    ref = json.loads(Path(index_json).read_text(encoding="utf-8"))
    byid = {b["gml_id"]: b for b in ref["buildings"]}
    zf = zipfile.ZipFile(zip_path)
    ts = json.loads(zf.read(f"{tileset}/tileset.json"))
    nodes = walk_tileset(ts["root"])
    selected = select_tiles(nodes, bbox)
    if max_tiles:
        selected = selected[:max_tiles]
    dz, de, dn = [], [], []
    for ti in selected:
        raw = zf.read(f"{tileset}/{nodes[ti]['uri']}")
        ft, _fb, bt, _bb, glb = b3dm_split(raw)
        js, binc = glb_split(glb)
        rtc = (js.get("extensions") or {}).get("CESIUM_RTC", {}).get("center")
        if rtc is None:
            continue
        gml = bt.get("gml_id") or []
        for prim in (js.get("meshes") or [{}])[0].get("primitives") or []:
            d = draco_decode_primitive(js, binc, prim)
            if "_BATCHID" not in d:
                continue
            L = tile_positions_to_local(d["POSITION"].astype(np.float64), rtc,
                                        origin, 0.0, 0.0)   # 生の楕円体高のまま
            bid = d["_BATCHID"].reshape(-1)
            for b in range(int(ft.get("BATCH_LENGTH", 0))):
                g = gml[b] if b < len(gml) else None
                if g not in byid:
                    continue
                m = bid == b
                if not m.any():
                    continue
                r = byid[g]
                fp = np.asarray(r["footprint"], dtype=np.float64)
                dz.append(float(L[m, 2].max()) - (r["height"] + ground0))
                de.append(float(L[m, 0].min()) - float(fp[:, 0].min()))
                dn.append(float(L[m, 1].min()) - float(fp[:, 1].min()))
    if not dz:
        return {"n": 0}
    A = np.asarray(dz)
    # 生の中央値は汚染される: REPLACE 階層の粗いタイルは同じ gml_id を **LOD1(箱)** で
    # 持つため zmax が LOD2.2 と一致せず、タイル境界で分割された建物も最大値が欠ける。
    # 1 cm ビンのヒストグラム最頻値の周り ±0.25 m だけを採るのが頑健な推定。
    lo, hi = float(A.min()), float(A.max())
    nb = max(1, int(round((hi - lo) / 0.01)) + 1)
    h, edges = np.histogram(A, bins=nb, range=(lo, lo + nb * 0.01))
    peak = float(edges[int(np.argmax(h))] + 0.005)
    core = A[np.abs(A - peak) <= 0.25]
    return {
        "n": len(dz),
        "n_core": int(core.size),
        "geoid_core_median": round(float(np.median(core)), 4) if core.size else None,
        "geoid_core_p05": round(float(np.percentile(core, 5)), 4) if core.size else None,
        "geoid_core_p95": round(float(np.percentile(core, 95)), 4) if core.size else None,
        "geoid_peak_bin": round(peak, 4),
        "geoid_median": round(float(np.median(A)), 4),
        "geoid_p25": round(float(np.percentile(A, 25)), 4),
        "geoid_p75": round(float(np.percentile(A, 75)), 4),
        "geoid_min": round(float(A.min()), 4),
        "geoid_max": round(float(A.max()), 4),
        "note": ("core = ヒストグラム最頻値 ±0.25 m。raw の下振れは REPLACE 階層の "
                 "LOD1 タイル/タイル境界分割によるもの"),
        "dE_median_m": round(float(np.median(de)), 6),
        "dN_median_m": round(float(np.median(dn)), 6),
        "dE_p95_abs_m": round(float(np.percentile(np.abs(de), 95)), 6),
        "dN_p95_abs_m": round(float(np.percentile(np.abs(dn), 95)), 6),
    }


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PLATEAU 3D Tiles → local-m 量子化タイル")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--tileset", default=DEFAULT_TILESET)
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "data" / "plateau" / "tiles_lod2"))
    ap.add_argument("--bbox", default=None,
                    help="lat0,lon0,lat1,lon1(既定=シム bbox)")
    ap.add_argument("--ground0", type=float, default=None)
    ap.add_argument("--geoid", type=float, default=GEOID_M)
    ap.add_argument("--atlas-scale", type=float, default=0.5)
    ap.add_argument("--atlas-quality", type=int, default=85)
    ap.add_argument("--no-atlas", action="store_true")
    ap.add_argument("--all-attrs", action="store_true",
                    help="batchTable の全キーを出力(既定は curated)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attrs-out", default=None,
                    help="既定 data/plateau/tiles_batch_attrs.json")
    ap.add_argument("--calibrate-geoid", action="store_true")
    a = ap.parse_args(argv)

    bbox = tuple(float(v) for v in a.bbox.split(",")) if a.bbox else DEFAULT_BBOX
    ground0, g0src = _resolve_ground0(a.ground0)
    print(f"[tiles3d] ground0={ground0} ({g0src})  geoid={a.geoid}", flush=True)
    if a.calibrate_geoid:
        r = calibrate_geoid(a.zip, a.tileset, bbox, DEFAULT_ORIGIN, ground0,
                            max_tiles=a.limit)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0
    run(a.zip, a.tileset, a.out_dir, bbox, DEFAULT_ORIGIN, ground0, a.geoid,
        atlas_scale=a.atlas_scale, atlas_quality=a.atlas_quality,
        limit=a.limit, do_atlas=not a.no_atlas, all_attrs=a.all_attrs,
        attrs_out=a.attrs_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
