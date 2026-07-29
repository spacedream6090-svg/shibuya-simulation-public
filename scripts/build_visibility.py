"""視点グリッド × 広告面の**可視行列**をシム外で事前計算する(C0 第68バッチ 2026-07-29)。

計画書 docs/plans/twin-physics-vision-affordance-plan.md レーン1 C0 行の実装。
**シム本体には一切触らない**(このスクリプトは `src/society` を読むだけ。ランタイムでの利用=
`street.py` の視認判定への LOS 配線は後続 B-L1 の仕事)。

    python scripts/build_visibility.py \
        --faces conf/visibility/example_faces.yaml \
        --heights data/building_heights_shibuya.json \
        --bbox -250 -250 250 250 --cell 2.0 --eye 1.5 \
        --out runs/_visibility/core500
    # 任意: --dem data/plateau(地表 z を視点・面・建物基部に加算)
    #       --near-edge-m 8(道路ポリラインから 8m 以内=歩行可能面の近似に視点を限定)
    #       --chunk 4096(視点チャンク。1250万ペア級を分割処理する)

---------------------------------------------------------------------------
なぜ事前計算か / 計算量の設計
---------------------------------------------------------------------------
可視性は **O(視点 n × 面 m)** で閉じる。視点×視点の O(n²) 相互可視は計算しない(必要が無い)。
500m 四方・2m 格子 = 62,500 セル(屋外はその 6〜7 割)× 面 200 枚 = 1,250 万ペアで、事前計算
すればランタイムはテーブル参照 1 回で済む。ランタイム側でレイキャストしないことが
「LLM 呼数・計算予算に幾何を交絡させない」(計画書 §5-1)ための構造的な担保でもある。

---------------------------------------------------------------------------
2.5D LOS の定式化(このスクリプトの中核。テストで検算している)
---------------------------------------------------------------------------
遮蔽物は「フットプリント多角形 × 実高さ(平らな屋根)」の角柱で近似する。
面の水平位置 F=(fx,fy) は 1 点で、面サンプル(下端・中心・上端)は z だけが違う。よって
**2D の交差計算は 3 サンプルで共有できる**。F を扇の要にして視点 P へ向かう線分を
    Q(s) = F + s·(P − F),  s ∈ [0,1]
と置くと、視線の高さは水平距離に対して線形なので、apex 高 fz のとき
    z_ray(s) = fz·(1 − s) + pz·s        (pz = 視点の目線高)
となる。交差点(建物 j の壁を横切る点)の高さ上限を top_z(j) とすると
    遮蔽 ⟺ top_z > fz·(1 − s) + pz·s
        ⟺ (top_z − pz·s) / (1 − s) > fz                      … (1 − s > 0 なので不等号は保存)
右辺が fz だけの式になったので、交差ごとの量
    g = (top_z − pz·s) / (1 − s)
の **最大値 G = max_j g_j** を視点ごとに一度求めておけば、3 つの面サンプル z に対する判定は
    可視(サンプル k) ⟺ G ≤ fz_k
の比較 3 回で済む。g は fz に依らないので、1 回の (視点 × 辺) 行列演算で 3 サンプルを解ける。
G は fz について単調なので可視集合は 上端 ⊇ 中心 ⊇ 下端 と入れ子になり、可視サンプル数は
0/1/2/3 のいずれか = frac_visible が 0 / 0.5 / 1 の 3 値に落ちる(部分可視の定義が一意)。

**自建物は遮蔽から除外する**(面が載っている壁で自分自身を遮ってしまうため)。その結果
「法線を持たない面は自建物を透過して見える」ことになるので、面には法線を書くのが原則
(法線があれば背面カリング `--max-incidence-deg` で裏側は落ちる。凸なフットプリントなら
面の前方半空間に自建物の躯体は無いので、自建物除外は無害になる)。

---------------------------------------------------------------------------
性能(実測と、次に効く一手)
---------------------------------------------------------------------------
刈り込みは 2 段だけ:(a) 面ごとに `--max-dist-m` と背面カリングで視点を落とす
(b)「面 ∪ 視点チャンク」の bbox に交わる遮蔽辺だけを候補にする。あとは numpy の
(視点 × 辺)行列を `_MAX_CELLS` で内部分割しながら回す。既定地図・700m 四方・cell 2m ×
面 200 枚 = **1,331 万ペアを 37.5 秒・ピーク RSS 211MB** で完走した(2026-07-29 実測)。
足りなくなったときの次の一手は **角度スイープ**(面から見た各辺の方位区間を求め、方位で
ソートした視点の連続区間だけに交差判定を掛ける)。総当たりが辺数に比例するのに対し、
辺 1 本あたり「その辺が張る立体角に入る視点」だけになるので数十倍効く。現状は不要。

---------------------------------------------------------------------------
VAI(OOH 業界標準の視認価値)との対応
---------------------------------------------------------------------------
OOH の視認価値指標(VAI 系)は概ね **サイズ・距離・角度・照明・滞留**の 5 変数で組まれる。
本行列が持つのは**幾何で解ける 3 つ**:
    サイズ  … 面プロファイルの `w_m` × (z_top − z_base)(行列には持たず meta.faces に運ぶ)
    距離    … `dist_m` 列(目線点 → 面中心の 3D 距離)
    角度    … `incidence_deg` 列(面法線と「面→視点」ベクトルの水平角。0 = 正対)
残る **照明(lighting)と滞留(dwell)は将来列**。照明はリポに実装が無く(実査 §1.5: grep 0 件)、
滞留は L1 の在場時間から事後に接続する量なので、可視行列(静的幾何)には持たせない。
**この行列は「見えうるか」までしか言わない**(見たか=視認は確率判定で street.py の管轄)。

---------------------------------------------------------------------------
決定論
---------------------------------------------------------------------------
乱数を一切使わない。同じ入力・同じパラメータ(--chunk を含む)なら
`visibility_matrix.parquet` は**バイト一致**する(行順 = 視点グリッド順 × face_id 昇順、
浮動小数の丸めも固定)。`visibility_meta.json` のうち実行ごとに変わる値(計算時間・出力サイズ・
ライブラリ版)は `runtime` キー **1 つに隔離**してあり、それを除いた部分もバイト一致する。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.world.map import CityMap                      # noqa: E402

SCHEMA = "visibility-matrix-1.0"
FACES_SCHEMA = "visibility-faces-1.0"

# 面プロファイルで受理するキー(未知キーはエラー=実験条件の取り違え防止。worldmod と同じ流儀)
_FACE_KEYS = {"id", "slot", "building", "kind", "x", "y", "z_base", "z_top",
              "normal_deg", "w_m", "note"}
_FACE_KINDS = {"large", "normal"}

# (視点 × 辺)行列の 1 バッチあたり要素数の上限(メモリ上限 ≈ この値 × 8B × 数本)
_MAX_CELLS = 4_000_000


# =========================================================================== 入力
def load_faces(path: str | Path) -> tuple[dict, list[dict]]:
    """広告面プロファイル(YAML)を読み、(meta, faces) を返す。検証つき・乱数ゼロ。

    faces は **id 昇順**に並べ替えて返す(行列の列順=決定論)。
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    meta = dict(doc.get("meta") or {})
    schema = str(meta.get("schema", ""))
    if schema and schema != FACES_SCHEMA:
        raise ValueError(f"未知の面プロファイル schema: {schema!r}(期待 {FACES_SCHEMA!r})")
    raw = doc.get("faces")
    if not isinstance(raw, list) or not raw:
        raise ValueError("faces: 面のリストが空(または list ではない)")
    faces: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"faces[{i}] が mapping ではない")
        unknown = set(item) - _FACE_KEYS
        if unknown:
            raise ValueError(f"faces[{i}] に未知のキー: {sorted(unknown)}")
        fid = str(item.get("id", "")).strip()
        if not fid:
            raise ValueError(f"faces[{i}] に id が無い")
        if fid in seen:
            raise ValueError(f"face id が重複: {fid!r}")
        seen.add(fid)
        missing = [k for k in ("x", "y", "z_base", "z_top") if item.get(k) is None]
        if missing:
            raise ValueError(f"{fid}: 必須キーが無い: {missing}")
        kind = str(item.get("kind", "normal"))
        if kind not in _FACE_KINDS:
            raise ValueError(f"{fid}: kind は {sorted(_FACE_KINDS)} のいずれか(受領 {kind!r})")
        z_base = float(item["z_base"])
        z_top = float(item["z_top"])
        if not (z_top > z_base):
            raise ValueError(f"{fid}: z_top ({z_top}) は z_base ({z_base}) より大きいこと")
        nd = item.get("normal_deg")
        faces.append({
            "id": fid,
            "slot": (str(item["slot"]) if item.get("slot") is not None else None),
            "building": (str(item["building"]) if item.get("building") is not None else None),
            "kind": kind,
            "x": float(item["x"]), "y": float(item["y"]),
            "z_base": z_base, "z_top": z_top,
            "normal_deg": (None if nd is None else float(nd) % 360.0),
            "w_m": (None if item.get("w_m") is None else float(item["w_m"])),
            "note": (None if item.get("note") is None else str(item["note"])),
        })
    faces.sort(key=lambda f: f["id"])
    return meta, faces


def normal_vec(normal_deg: float) -> tuple[float, float]:
    """方位角 [deg](0=北=+Y, 90=東=+X, 時計回り)→ 単位ベクトル (nx, ny)。"""
    a = math.radians(float(normal_deg))
    return math.sin(a), math.cos(a)


# =========================================================================== 幾何
class Scene:
    """遮蔽計算に必要な静的幾何(建物フットプリント+実高さ)の配列表現。

    - poly_*: 全建物のフットプリント(屋内判定=点in多角形に使う。高さの有無は問わない)
    - e_*   : **高さを持つ建物のみ**の辺(遮蔽計算に使う)。高さ不明の建物は遮蔽しない
              (欠測を 0m とも無限大とも解釈しない=正直な縮退)。
    """

    def __init__(self, city: CityMap, ground_z=None) -> None:
        self.bld_index: dict[str, int] = {}
        px: list[float] = []
        py: list[float] = []
        off: list[int] = [0]
        eax: list[float] = []
        eay: list[float] = []
        ebx: list[float] = []
        eby: list[float] = []
        etz: list[float] = []
        ebi: list[int] = []
        self.n_with_height = 0
        self.src_counts: dict[str, int] = {}
        for bi, b in enumerate(city.buildings):
            self.bld_index[b["id"]] = bi
            fp = b["footprint"]
            for x, y in fp:
                px.append(float(x))
                py.append(float(y))
            off.append(len(px))
            h = b.get("height_m")
            if h is None:
                continue
            self.n_with_height += 1
            src = str(b.get("height_src", "?"))
            self.src_counts[src] = self.src_counts.get(src, 0) + 1
            cx, cy = b["centroid"]
            base = 0.0 if ground_z is None else float(ground_z(cx, cy))
            top = base + float(h)
            n = len(fp)
            for k in range(n):
                ax, ay = fp[k]
                bx, by = fp[(k + 1) % n]
                if ax == bx and ay == by:
                    continue                      # 閉じた環の重複点=長さ 0 の辺は捨てる
                eax.append(float(ax))
                eay.append(float(ay))
                ebx.append(float(bx))
                eby.append(float(by))
                etz.append(top)
                ebi.append(bi)
        self.poly_x = np.asarray(px, dtype=np.float64)
        self.poly_y = np.asarray(py, dtype=np.float64)
        self.poly_off = np.asarray(off, dtype=np.int64)
        self.e_ax = np.asarray(eax, dtype=np.float64)
        self.e_ay = np.asarray(eay, dtype=np.float64)
        self.e_bx = np.asarray(ebx, dtype=np.float64)
        self.e_by = np.asarray(eby, dtype=np.float64)
        self.e_top = np.asarray(etz, dtype=np.float64)
        self.e_bld = np.asarray(ebi, dtype=np.int64)
        self.e_minx = np.minimum(self.e_ax, self.e_bx)
        self.e_maxx = np.maximum(self.e_ax, self.e_bx)
        self.e_miny = np.minimum(self.e_ay, self.e_by)
        self.e_maxy = np.maximum(self.e_ay, self.e_by)
        self.n_buildings = len(city.buildings)
        self.n_edges = int(self.e_ax.size)


def _pip(xs: np.ndarray, ys: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """点in多角形(交差数法・ベクトル化)。境界上の扱いは実装依存だが決定論。"""
    x1, y1 = xs, ys
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (y1[None, :] > py[:, None]) != (y2[None, :] > py[:, None])
    dy = (y2 - y1)[None, :]
    safe = np.where(dy == 0.0, 1.0, dy)
    xin = (x2 - x1)[None, :] * (py[:, None] - y1[None, :]) / safe + x1[None, :]
    hit = cond & (px[:, None] < xin)
    return (hit.sum(axis=1) % 2) == 1


def inside_any_building(scene: Scene, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """各点が**いずれかの建物フットプリント内部**かどうか(屋内=視点から除外する)。"""
    inside = np.zeros(px.shape, dtype=bool)
    for k in range(scene.poly_off.size - 1):
        s, e = int(scene.poly_off[k]), int(scene.poly_off[k + 1])
        if e - s < 3:
            continue
        xs, ys = scene.poly_x[s:e], scene.poly_y[s:e]
        cand = (~inside) & (px >= xs.min()) & (px <= xs.max()) \
            & (py >= ys.min()) & (py <= ys.max())
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            continue
        inside[idx] = _pip(xs, ys, px[idx], py[idx])
    return inside


def near_edge_grid(city: CityMap, x0: float, y0: float, cell: float,
                   nx: int, ny: int, radius: float) -> np.ndarray:
    """道路ポリラインから radius [m] 以内のセル(= 歩行可能面の近似)の真偽格子(nx*ny,)。

    地下エッジ(layer < 0)は屋外の街路ではないので除外する。デッキ(layer > 0)は含む。
    """
    step = max(cell * 0.5, 0.25)
    pts_x: list[float] = []
    pts_y: list[float] = []
    for _u, _v, d in city.graph.edges(data=True):
        if int(d.get("layer", 0)) < 0:
            continue
        geom = d["geometry"]
        for (ax, ay), (bx, by) in zip(geom, geom[1:]):
            L = math.hypot(bx - ax, by - ay)
            n = max(1, int(L / step))
            for i in range(n + 1):
                t = i / n
                pts_x.append(ax + (bx - ax) * t)
                pts_y.append(ay + (by - ay) * t)
    mask = np.zeros(nx * ny, dtype=bool)
    if not pts_x:
        return mask
    P = np.asarray(pts_x, dtype=np.float64)
    Q = np.asarray(pts_y, dtype=np.float64)
    cix = np.floor((P - x0) / cell).astype(np.int64)
    ciy = np.floor((Q - y0) / cell).astype(np.int64)
    k = int(math.ceil(radius / cell))
    r2 = radius * radius
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            jx = cix + dx
            jy = ciy + dy
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            if not ok.any():
                continue
            cx = x0 + (jx + 0.5) * cell
            cy = y0 + (jy + 0.5) * cell
            ok &= ((cx - P) ** 2 + (cy - Q) ** 2) <= r2
            if ok.any():
                mask[(jy * nx + jx)[ok]] = True
    return mask


def make_grid(bbox: tuple[float, float, float, float], cell: float):
    """bbox と格子幅から視点候補(セル中心)を作る。返り値 (ix, iy, x, y, nx, ny)。

    行順は **iy 昇順 → ix 昇順**(row-major)。座標は x0 + (ix + 0.5)·cell。
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("--bbox は x0 < x1 かつ y0 < y1 であること")
    if cell <= 0:
        raise ValueError("--cell は正の値であること")
    nx = int(math.floor((x1 - x0) / cell + 1e-9))
    ny = int(math.floor((y1 - y0) / cell + 1e-9))
    if nx < 1 or ny < 1:
        raise ValueError("--cell が bbox に対して大きすぎる(格子が 0 セル)")
    ix = np.tile(np.arange(nx, dtype=np.int64), ny)
    iy = np.repeat(np.arange(ny, dtype=np.int64), nx)
    x = x0 + (ix + 0.5) * cell
    y = y0 + (iy + 0.5) * cell
    return ix, iy, x, y, nx, ny


# =========================================================================== LOS
def max_block_g(vpx: np.ndarray, vpy: np.ndarray, vpz: np.ndarray,
                fx: float, fy: float,
                ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray,
                top: np.ndarray) -> np.ndarray:
    """視点群 → 面位置 (fx, fy) の視線に対する遮蔽指標 G(視点ごとの最大値)を返す。

    G = max over 交差 of (top_z − pz·s) / (1 − s)。面サンプル高 fz に対して
    **遮蔽 ⟺ G > fz**(= 可視 ⟺ G ≤ fz)。交差が無ければ −inf。
    モジュール docstring の導出を参照。乱数ゼロ・純関数。
    """
    m = vpx.size
    if m == 0 or ax.size == 0:
        return np.full(m, -np.inf, dtype=np.float64)
    d1x = vpx - fx
    d1y = vpy - fy
    d2x = bx - ax
    d2y = by - ay
    afx = ax - fx
    afy = ay - fy
    s_num = afx * d2y - afy * d2x                                   # (E,)
    out = np.empty(m, dtype=np.float64)
    batch = max(1, int(_MAX_CELLS // max(1, ax.size)))
    for lo in range(0, m, batch):
        hi = min(m, lo + batch)
        p1x = d1x[lo:hi, None]
        p1y = d1y[lo:hi, None]
        denom = p1x * d2y[None, :] - p1y * d2x[None, :]
        ok = denom != 0.0
        den = np.where(ok, denom, 1.0)
        s = s_num[None, :] / den
        u = (afx[None, :] * p1y - afy[None, :] * p1x) / den
        ok &= (s > 0.0) & (s < 1.0) & (u >= 0.0) & (u <= 1.0)
        one_minus_s = np.where(ok, 1.0 - s, 1.0)
        g = (top[None, :] - vpz[lo:hi, None] * s) / one_minus_s
        out[lo:hi] = np.where(ok, g, -np.inf).max(axis=1)
    return out


# =========================================================================== 本体
def build_matrix(scene: Scene, faces: list[dict], vp, params: dict, sink, log=print):
    """可視行列の行を生成する(視点チャンク外側・面内側のループ)。

    vp = (ix, iy, x, y, z) の numpy 配列タプル。可視行はチャンクごとに `sink.write(block)` へ
    流す(全行を RAM に溜めない=1250万ペア級でもメモリはチャンク 1 個ぶんで頭打ち)。
    行順は (視点 index 昇順, face_id 昇順) = 決定論。返り値は統計 dict。
    """
    ix, iy, vx, vy, vz = vp
    n_vp = vx.size
    max_dist = float(params["max_dist_m"])
    max_inc = float(params["max_incidence_deg"])
    chunk = max(1, int(params["chunk"]))
    md2 = max_dist * max_dist

    fx = np.array([f["x"] for f in faces], dtype=np.float64)
    fy = np.array([f["y"] for f in faces], dtype=np.float64)
    own = np.array([scene.bld_index.get(f["building"] or "", -1) for f in faces],
                   dtype=np.int64)
    fz = np.array([[f["_z_base"], f["_z_mid"], f["_z_top"]] for f in faces],
                  dtype=np.float64)

    per_face = [{"n_rows": 0, "n_full": 0, "n_partial": 0, "sum_dist": 0.0,
                 "min_dist": math.inf, "max_dist": 0.0} for _ in faces]
    n_considered = 0

    for lo in range(0, n_vp, chunk):
        hi = min(n_vp, lo + chunk)
        cvx, cvy, cvz = vx[lo:hi], vy[lo:hi], vz[lo:hi]
        cminx, cmaxx = float(cvx.min()), float(cvx.max())
        cminy, cmaxy = float(cvy.min()), float(cvy.max())
        buf: list[tuple] = []
        for fi, face in enumerate(faces):
            dx = cvx - fx[fi]
            dy = cvy - fy[fi]
            d2 = dx * dx + dy * dy
            sel = (d2 <= md2) & (d2 > 0.0)
            if not sel.any():
                continue
            inc = None
            if face["normal_deg"] is not None:
                nx_, ny_ = normal_vec(face["normal_deg"])
                dist2d = np.sqrt(d2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cosv = (nx_ * dx + ny_ * dy) / np.where(dist2d > 0, dist2d, 1.0)
                inc = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
                sel &= inc <= max_inc
                if not sel.any():
                    continue
            idx = np.flatnonzero(sel)
            n_considered += int(idx.size)
            sx, sy, sz = cvx[idx], cvy[idx], cvz[idx]
            # --- 遮蔽辺の前フィルタ: 「面 ∪ このチャンクの視点」の bbox に交わる辺のみ
            xmin = min(cminx, float(fx[fi]))
            xmax = max(cmaxx, float(fx[fi]))
            ymin = min(cminy, float(fy[fi]))
            ymax = max(cmaxy, float(fy[fi]))
            ecand = (scene.e_maxx >= xmin) & (scene.e_minx <= xmax) \
                & (scene.e_maxy >= ymin) & (scene.e_miny <= ymax)
            if own[fi] >= 0:
                ecand &= scene.e_bld != own[fi]
            ei = np.flatnonzero(ecand)
            G = max_block_g(sx, sy, sz, float(fx[fi]), float(fy[fi]),
                            scene.e_ax[ei], scene.e_ay[ei],
                            scene.e_bx[ei], scene.e_by[ei], scene.e_top[ei])
            n_vis = ((G <= fz[fi, 0]).astype(np.int8)
                     + (G <= fz[fi, 1]).astype(np.int8)
                     + (G <= fz[fi, 2]).astype(np.int8))
            keep = n_vis > 0
            if not keep.any():
                continue
            kidx = idx[keep]
            frac = np.where(n_vis[keep] == 3, 1.0, 0.5)
            dz = fz[fi, 1] - sz[keep]
            dist = np.sqrt(d2[kidx] + dz * dz)
            incv = (np.full(kidx.size, np.nan) if inc is None else inc[kidx])
            buf.append((kidx, fi, frac, dist, incv))
            st = per_face[fi]
            st["n_rows"] += int(kidx.size)
            st["n_full"] += int((frac == 1.0).sum())
            st["n_partial"] += int((frac == 0.5).sum())
            st["sum_dist"] += float(dist.sum())
            st["min_dist"] = min(st["min_dist"], float(dist.min()))
            st["max_dist"] = max(st["max_dist"], float(dist.max()))
        if not buf:
            continue
        # --- チャンク内を (視点 index, face index) で安定ソート = 全体で決定論の行順
        kidx = np.concatenate([b[0] for b in buf])
        fidx = np.concatenate([np.full(b[0].size, b[1], dtype=np.int64) for b in buf])
        frac = np.concatenate([b[2] for b in buf])
        dist = np.concatenate([b[3] for b in buf])
        incv = np.concatenate([b[4] for b in buf])
        order = np.lexsort((fidx, kidx))
        kidx, fidx = kidx[order], fidx[order]
        sink.write({"vp_ix": ix[lo:hi][kidx], "vp_iy": iy[lo:hi][kidx],
                    "vp_x": cvx[kidx], "vp_y": cvy[kidx], "face_ix": fidx,
                    "visible": frac[order],
                    "dist_m": np.round(dist[order], 2),
                    "incidence_deg": np.round(incv[order], 2)})
        log(f"[build_visibility]   視点 {hi}/{n_vp} … 行 {sink.n_rows}")
    return {"n_pairs_considered": n_considered, "per_face": per_face}


class MatrixWriter:
    """可視行列 parquet の逐次書き出し(視点チャンク 1 個 = 行グループ 1 個)。

    列順・型・行グループの切り方を固定するので、同じ入力・同じ --chunk なら**バイト一致**。
    行が 1 本も無くてもスキーマだけの parquet を残す(下流の分岐を減らすため)。
    """

    SCHEMA = pa.schema([
        ("vp_ix", pa.int32()), ("vp_iy", pa.int32()),
        ("vp_x", pa.float32()), ("vp_y", pa.float32()),
        ("face_id", pa.string()),
        ("visible", pa.float32()), ("dist_m", pa.float32()),
        ("incidence_deg", pa.float32()),
    ])

    def __init__(self, path: Path, faces: list[dict]) -> None:
        self.face_ids = np.array([f["id"] for f in faces], dtype=object)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._w = pq.ParquetWriter(path, self.SCHEMA, compression="snappy")
        self.n_rows = 0

    def write(self, blk: dict) -> None:
        inc = blk["incidence_deg"]
        tbl = pa.Table.from_arrays([
            pa.array(blk["vp_ix"], type=pa.int32()),
            pa.array(blk["vp_iy"], type=pa.int32()),
            pa.array(blk["vp_x"], type=pa.float32()),
            pa.array(blk["vp_y"], type=pa.float32()),
            pa.array(self.face_ids[blk["face_ix"]], type=pa.string()),
            pa.array(blk["visible"], type=pa.float32()),
            pa.array(blk["dist_m"], type=pa.float32()),
            pa.array(inc, type=pa.float32(), mask=np.isnan(inc)),
        ], schema=self.SCHEMA)
        self._w.write_table(tbl)
        self.n_rows += tbl.num_rows

    def close(self) -> None:
        self._w.close()


# =========================================================================== CLI
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="視点グリッド × 広告面の可視行列を事前計算する(シム外・決定論)")
    ap.add_argument("--faces", type=Path, required=True,
                    help="広告面プロファイル YAML(例 conf/visibility/example_faces.yaml)")
    ap.add_argument("--map", type=Path, default=REPO_ROOT / "data" / "shibuya_osm.json")
    ap.add_argument("--heights", type=Path, default=None,
                    help="建物実高さ表(A1: data/building_heights_shibuya.json)。"
                         "省略すると遮蔽物ゼロ=全面可視になる(警告を出す)")
    ap.add_argument("--fallback-m-per-level", type=float, default=3.5,
                    help="高さ表に無い建物の推定高 = levels x この値(attach_heights と同義)")
    ap.add_argument("--dem", type=Path, default=None,
                    help="地形フォルダ(terrain.npz+terrain.json がある。例 data/plateau)。"
                         "指定時は視点・面・建物基部に地表 z を加算する")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="視点グリッドの範囲 [m]。省略時は面群の bbox を --max-dist で膨らませる")
    ap.add_argument("--cell", type=float, default=2.0, help="視点格子の一辺 [m](既定 2.0)")
    ap.add_argument("--eye", type=float, default=1.5, help="目線高 [m](既定 1.5)")
    ap.add_argument("--max-dist-m", type=float, default=200.0,
                    help="この距離を超える視点は不可視として捨てる [m](既定 200)")
    ap.add_argument("--max-incidence-deg", type=float, default=85.0,
                    help="面法線からの水平角がこれを超えたら不可視(背面カリング。既定 85)")
    ap.add_argument("--near-edge-m", type=float, default=None,
                    help="道路ポリラインからこの距離以内のセルだけを視点にする(歩行可能面の近似)")
    ap.add_argument("--chunk", type=int, default=4096, help="視点チャンク(既定 4096)")
    ap.add_argument("--out", type=Path, required=True, help="出力フォルダ")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.perf_counter()
    log = print

    for p in (args.faces, args.map):
        if not Path(p).exists():
            print(f"[build_visibility] 入力が見つからない: {p}", file=sys.stderr)
            return 2
    faces_meta, faces = load_faces(args.faces)
    log(f"[build_visibility] 面 {len(faces)} 枚 ({args.faces})")

    city = CityMap(args.map)
    heights_stat = None
    if args.heights is not None:
        if not Path(args.heights).exists():
            print(f"[build_visibility] 高さ表が見つからない: {args.heights}", file=sys.stderr)
            return 2
        heights_stat = city.attach_heights(args.heights, args.fallback_m_per_level)
        log(f"[build_visibility] 高さ配線 {heights_stat}")
    else:
        log("[build_visibility] ★--heights 未指定: 遮蔽物ゼロ(全面可視)の行列になる")

    ground = None
    dem_meta = None
    if args.dem is not None:
        from society.world.elevation import ElevationGrid     # noqa: WPS433(任意依存)
        grid = ElevationGrid.load(args.dem)
        if grid is None:
            print(f"[build_visibility] 地形が読めない(terrain.npz/json 不在): {args.dem}",
                  file=sys.stderr)
            return 2
        ground = grid.height_at
        dem_meta = {"dir": str(args.dem), "nx": grid.nx, "ny": grid.ny,
                    "cell_m": grid.cell_m, "x0": grid.x0, "y0": grid.y0}
        log(f"[build_visibility] 地形 {dem_meta['nx']}x{dem_meta['ny']} cell={grid.cell_m}m")

    scene = Scene(city, ground_z=ground)
    log(f"[build_visibility] 建物 {scene.n_buildings}(高さ有 {scene.n_with_height}"
        f" {scene.src_counts})・遮蔽辺 {scene.n_edges}")

    # ---- 面の絶対高さ(地表 z を加算)+ 自建物の解決
    for f in faces:
        gz = 0.0 if ground is None else float(ground(f["x"], f["y"]))
        f["_z_base"] = f["z_base"] + gz
        f["_z_top"] = f["z_top"] + gz
        f["_z_mid"] = (f["_z_base"] + f["_z_top"]) / 2.0
        f["_ground_z"] = gz
        if f["building"] is not None and f["building"] not in scene.bld_index:
            print(f"[build_visibility] 面 {f['id']}: building={f['building']} は地図に無い"
                  "(自建物の遮蔽除外は効かない)", file=sys.stderr)

    # ---- 視点グリッド
    if args.bbox is None:
        pad = float(args.max_dist_m)
        bbox = (min(f["x"] for f in faces) - pad, min(f["y"] for f in faces) - pad,
                max(f["x"] for f in faces) + pad, max(f["y"] for f in faces) + pad)
    else:
        bbox = tuple(args.bbox)
    ix, iy, gx, gy, nx, ny = make_grid(bbox, args.cell)
    n_cells = gx.size
    keep = ~inside_any_building(scene, gx, gy)
    n_outdoor = int(keep.sum())
    n_near = None
    if args.near_edge_m is not None:
        near = near_edge_grid(city, bbox[0], bbox[1], args.cell, nx, ny,
                              float(args.near_edge_m))
        keep &= near
        n_near = int(keep.sum())
    ix, iy, gx, gy = ix[keep], iy[keep], gx[keep], gy[keep]
    gz = np.full(gx.size, float(args.eye), dtype=np.float64)
    if ground is not None:
        gz = gz + np.array([ground(float(a), float(b)) for a, b in zip(gx, gy)],
                           dtype=np.float64)
    log(f"[build_visibility] 格子 {nx}x{ny}={n_cells} → 屋外 {n_outdoor}"
        + (f" → 道路近傍 {n_near}" if n_near is not None else "")
        + f"(視点 {gx.size})")

    params = {"cell_m": args.cell, "eye_m": args.eye,
              "max_dist_m": args.max_dist_m,
              "max_incidence_deg": args.max_incidence_deg,
              "near_edge_m": args.near_edge_m, "chunk": args.chunk,
              "fallback_m_per_level": args.fallback_m_per_level,
              "bbox": [float(v) for v in bbox]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "visibility_matrix.parquet"
    writer = MatrixWriter(mpath, faces)
    try:
        stats = build_matrix(scene, faces, (ix, iy, gx, gy, gz), params,
                             sink=writer, log=log)
    finally:
        writer.close()
    n_rows = writer.n_rows

    n_vp = int(gx.size)
    by_face = {}
    for f, st in zip(faces, stats["per_face"]):
        by_face[f["id"]] = {
            "n_rows": st["n_rows"],
            "rate": (round(st["n_rows"] / n_vp, 6) if n_vp else 0.0),
            "n_full": st["n_full"], "n_partial": st["n_partial"],
            "mean_dist_m": (round(st["sum_dist"] / st["n_rows"], 2)
                            if st["n_rows"] else None),
            "min_dist_m": (round(st["min_dist"], 2) if st["n_rows"] else None),
            "max_dist_m": (round(st["max_dist"], 2) if st["n_rows"] else None),
        }
    meta = {
        "schema": SCHEMA,
        "generated_by": "scripts/build_visibility.py",
        "purpose": "視点グリッド × 広告面の 2.5D 可視行列(疎=可視の行のみ)。"
                   "シム外の事前計算物であり、シム本体はまだこれを読まない(消費者は B-L1)。",
        "params": params,
        "inputs": {
            "map": str(args.map), "faces": str(args.faces),
            "faces_profile": faces_meta.get("name"),
            "heights": (str(args.heights) if args.heights else None),
            "heights_stat": heights_stat, "dem": dem_meta,
        },
        "grid": {"nx": nx, "ny": ny, "n_cells": n_cells, "n_outdoor": n_outdoor,
                 "n_near_edge": n_near, "n_viewpoints": n_vp,
                 "order": "iy 昇順 → ix 昇順(row-major)。座標 = x0 + (i+0.5)·cell"},
        "occluders": {"n_buildings": scene.n_buildings,
                      "n_with_height": scene.n_with_height,
                      "height_src": dict(sorted(scene.src_counts.items())),
                      "n_edges": scene.n_edges},
        "faces": [{k: f[k] for k in
                   ("id", "slot", "building", "kind", "x", "y", "z_base", "z_top",
                    "normal_deg", "w_m")} | {"ground_z_m": f["_ground_z"]}
                  for f in faces],
        "counts": {"n_faces": len(faces), "n_viewpoints": n_vp,
                   "n_pairs_total": n_vp * len(faces),
                   "n_pairs_considered": stats["n_pairs_considered"],
                   "n_rows": n_rows,
                   "n_full": sum(v["n_full"] for v in by_face.values()),
                   "n_partial": sum(v["n_partial"] for v in by_face.values())},
        "visibility": {
            "overall_rate": (round(n_rows / (n_vp * len(faces)), 6)
                             if n_vp and faces else 0.0),
            "by_face": by_face,
        },
        "columns": {
            "vp_ix/vp_iy": "視点セルの格子 index(bbox 原点基準)",
            "vp_x/vp_y": "視点セル中心の地図ローカル座標 [m]",
            "face_id": "面 ID(プロファイルの faces[].id)",
            "visible": "可視割合 1.0=上端/中心/下端すべて可視・0.5=一部可視(0.0 の行は格納しない)",
            "dist_m": "目線点 → 面中心の 3D 距離 [m](VAI の距離変数)",
            "incidence_deg": "面法線と『面→視点』ベクトルの水平角 [deg]・0=正対"
                             "(VAI の角度変数。法線を持たない面は null)",
            "_future": "VAI の残り 2 変数 = 照明(lighting)・滞留(dwell)は将来列。"
                       "照明はリポに実装が無く、滞留は L1 の在場時間から事後接続する量なので"
                       "静的な可視行列には持たせない",
        },
        "determinism": "乱数ゼロ。同一入力・同一パラメータ(--chunk 含む)で "
                       "visibility_matrix.parquet はバイト一致。可変値は runtime キーに隔離。",
    }
    elapsed = time.perf_counter() - t0
    meta["runtime"] = {
        "elapsed_sec": round(elapsed, 3),
        "matrix_bytes": mpath.stat().st_size,
        "numpy": np.__version__, "pyarrow": pa.__version__,
        "note": "★ここだけ実行ごとに変わる(決定論の検証はこのキーを除いて行う)",
    }
    (out_dir / "visibility_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")

    log(f"[build_visibility] 行 {n_rows} / ペア {n_vp * len(faces)}"
        f"(可視率 {meta['visibility']['overall_rate']:.4f})")
    log(f"[build_visibility] 出力 {mpath} "
        f"({mpath.stat().st_size / 1024.0:.1f} KB) / {elapsed:.1f}s")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
