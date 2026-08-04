"""Jülich Pedestrian Dynamics Data Archive の軌跡 txt → 較正用の派生 CSV。

    python -m reference.physics_bench.juelich \
        --raw data/juelich_ped --out reference/physics_bench/data

生データ(`data/juelich_ped/`・.gitignore 済み・約 150 MB)はコミットしない。
本モジュールが作る **小さな派生 CSV だけ** を `reference/physics_bench/data/` に置く。

────────────────────────────────────────────────────────────────────────
出典・ライセンス(帰属表示。README.md にも再掲)
  Forschungszentrum Jülich, Institute for Advanced Simulation (IAS-7),
  Pedestrian Dynamics Data Archive — https://ped.fz-juelich.de/database
  ライセンス: **CC Attribution 4.0 International (CC BY 4.0)**
  取得日: 2026-08-05

  | 本モジュールでの名前 | 実験 | DOI |
  |---|---|---|
  | `hermes_uni_open`   | HERMES 一方向流・開境界   | 10.34735/ped.2009.14 |
  | `hermes_uni_closed` | HERMES 一方向流・閉境界   | 10.34735/ped.2009.13 |
  | `basigo_uni_corr`   | BaSiGo 一方向流コリドー   | 10.34735/ped.2013.6  |
  | `hermes_bottleneck` | HERMES ボトルネック       | 10.34735/ped.2009.6  |
  | `basigo_crossing90` | BaSiGo 90°交差流          | 10.34735/ped.2013.4  |

  加工内容(= 派生 CSV が生データとどう違うか。CC BY 4.0 の "indicate if changes
  were made" に対応):
    1. 座標の単位を m に統一(2009 HERMES と 2013 crossing_90 は cm、
       2013 uni_corr は m。→ §単位判定)。frame → 秒(÷fps)。
    2. z 列(頭部高さ/3D 位置)を **捨てる**(2D 較正のため)。
       ただし単位判定の手がかりとしてのみ読む。
    3. 速度 = 位置の中心差分(窓 ±0.5 s)。端の 0.5 s は速度を持たないので落とす。
    4. 測定区画内の古典密度(頭数 ÷ 面積)と、区画内個体の流れ方向速度成分の算術平均。
    5. 0.5 s ごとに間引いて標本化(生データは 16 fps)。
    6. 密度ビン(既定 0.1 /m²)ごとに平均・標準偏差・10/90 パーセンタイルへ集計。
  **生の軌跡そのものは再配布していない**(集計統計のみ)。
────────────────────────────────────────────────────────────────────────

§単位判定(実測に基づく・捏造なし)
  - `# id frame x/m y/m z/m` のようなヘッダ行があればそれに従う(2013 uni_corr)。
  - ヘッダが無い場合は z 列(= 頭部高さ)の中央値で判定する:
      中央値 > 10  → cm(実測: 2009 HERMES で 155〜183、crossing_90 で 176)
      それ以外     → m (実測: uni_corr で 1.871)
    実験ページの記載("x-coordinate [cm]")とも一致することを確認済み。
  - fps はヘッダ `# framerate: N fps` があればそれを使う。無ければ実験ページ記載の
    16 fps。★uni_corr_500_02 だけヘッダが **25 fps** で、これはページ本文の
    「run 02 は 25 fps・別カメラ」という註と一致する(= ヘッダを信じるのが正しい)。

§幾何(metadata JSON を一次情報にする — 研究文書 §1.3 の指示)
  各 run の `wkt_geometry`(外周 + 壁のリング)と `parameter`(通路幅・開口幅)を読む。
  **corridor5 の寸法食い違い(5 m 説 vs 8×3 m 説)は metadata で決着**:
    `2013unidirectional_metadata.json` の全 9 run が
    `corridor width [m] = 5`, `corridor length [m] = 18` を持ち、
    WKT の壁も y=0 と y=5(幅 5 m)・x=-9..9(長さ 18 m)で一致する。
    → **「5 m」は通路の *幅*、長さは 18 m。どちらの説も不正確だった。**

§測定区画(本モジュールの選択。文献の指定ではないので明記する)
  一方向流は **流れ方向 2 m × 通路全幅** の矩形を通路中央に置く(古典法)。
  RiMEA Test 4 が指定する 2×2 m は通路幅 2 m 未満の実験には置けないため、
  「流れ方向 2 m」だけを共有し、横断方向は通路全幅を採る(Jülich の慣行と同じ)。
  交差流は中央に 2×2 m(RiMEA Test 4 の寸法どおり)。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

DEFAULT_FPS = 16.0            # 実験ページ記載(2009 HERMES / 2013 crossing_90 / uni_corr)
SPEED_HALF_WINDOW_S = 0.5     # 中心差分の片側窓 [s](全窓 1.0 s = Jülich の慣行)
SAMPLE_STRIDE_S = 0.5         # 派生 CSV の標本間隔 [s]
MEAS_LENGTH_M = 2.0           # 測定区画の流れ方向長さ [m](RiMEA Test 4 と同じ 2 m)
STEADY_TRIM = 0.10            # 定常窓 = 在室区間の中央 80%(両端 10% を捨てる)

# ── データセット定義(ファイル名 → 実験・DOI・幾何規則) ──────────────────
#   zip: data/juelich_ped/ 直下のファイル名
#   meta: metadata JSON のファイル名
#   axis: 流れ方向の座標軸("x" or "y")
#   kind: "corridor" | "bottleneck" | "crossing"
DATASETS = {
    "hermes_uni_open": dict(
        zip="2009unidirectional_open_trajectories_txt.zip",
        meta="2009unidirectional_open_metadata.json",
        doi="10.34735/ped.2009.14", page="corridor3", kind="corridor",
        axis="y", width_key="corridor_width [m]", cross_lo=0.0),
    "hermes_uni_closed": dict(
        zip="2009unidirectional_closed_trajectories_txt.zip",
        meta="2009unidirectional_closed_metadata.json",
        doi="10.34735/ped.2009.13", page="corridor3", kind="corridor",
        axis="y", width_key="b_Corridor [m]", cross_lo=0.0),
    "basigo_uni_corr": dict(
        zip="2013unidirectional_trajectories_txt.zip",
        meta="2013unidirectional_metadata.json",
        doi="10.34735/ped.2013.6", page="corridor5", kind="corridor",
        axis="x", width_key="corridor width [m]", cross_lo=0.0),
    "hermes_bottleneck": dict(
        zip="2009bottleneck_trajectories_txt.zip",
        meta="2009bottleneck_metadata.json",
        doi="10.34735/ped.2009.6", page="hermes_bottleneck", kind="bottleneck",
        axis="y", width_key="b_Exit [m]", cross_lo=None),
    "basigo_crossing90": dict(
        zip="2013crossing_90_trajectories_txt.zip",
        meta="2013crossing_90_metadata.json",
        doi="10.34735/ped.2013.4", page="crossing_90", kind="crossing",
        axis=None, width_key="corridor width [m]", cross_lo=None),
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _num(s, default=None):
    """'1.8m' や '2.4' から先頭の数値を取り出す(metadata の表記ゆれ対策)。"""
    if s is None:
        return default
    m = _NUM.search(str(s))
    return float(m.group(0)) if m else default


def wkt_rings(wkt):
    """WKT POLYGON 文字列 → [ [(x,y), …], … ](外周 + 穴。単位 m)。"""
    if not wkt:
        return []
    rings = []
    for body in re.findall(r"\(([^()]*)\)", wkt):
        pts = []
        for tok in body.split(","):
            v = _NUM.findall(tok)
            if len(v) >= 2:
                pts.append((float(v[0]), float(v[1])))
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def ring_bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


# ─────────────────────────────────────────────────────────────────────────────
# 軌跡 txt のパース
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Traj:
    """1 run 分の軌跡(すべて SI 単位: m, s)。"""
    run: str
    ids: np.ndarray        # (M,) int64
    frame: np.ndarray      # (M,) int64
    x: np.ndarray          # (M,) float64 [m]
    y: np.ndarray          # (M,) float64 [m]
    fps: float
    unit: str              # "cm" | "m"(生データ側の単位。記録用)
    n_ped: int
    meta: dict = field(default_factory=dict)

    @property
    def t(self):
        return self.frame / self.fps


def parse_traj_text(text, run, fps_default=DEFAULT_FPS):
    """`ID frame x y z` 形式のテキスト → Traj(単位を m に正規化)。

    列は空白区切り。`#` 始まりはヘッダ(fps・単位の宣言に使う)。"""
    fps = None
    unit_hdr = None
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        m = re.search(r"framerate\s*:\s*([\d.]+)\s*fps", line)
        if m:
            fps = float(m.group(1))
        if re.search(r"\bx\s*/\s*m\b", line):
            unit_hdr = "m"
        elif re.search(r"\bx\s*/\s*cm\b", line):
            unit_hdr = "cm"
    arr = np.loadtxt(io.StringIO(text), comments="#", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ids = arr[:, 0].astype(np.int64)
    frame = arr[:, 1].astype(np.int64)
    x, y = arr[:, 2].copy(), arr[:, 3].copy()
    z = arr[:, 4] if arr.shape[1] > 4 else None

    # 単位判定: ヘッダ優先。無ければ z(頭部高さ)の中央値で判定する。
    if unit_hdr is not None:
        unit = unit_hdr
    elif z is not None and z.size:
        unit = "cm" if float(np.median(np.abs(z))) > 10.0 else "m"
    else:
        # z 列が無い異例。x,y の広がりで判定(数百 = cm)。
        unit = "cm" if float(np.nanmax(np.abs(np.concatenate([x, y])))) > 60.0 else "m"
    if unit == "cm":
        x /= 100.0
        y /= 100.0
    return Traj(run=run, ids=ids, frame=frame, x=x, y=y,
                fps=float(fps if fps else fps_default), unit=unit,
                n_ped=int(np.unique(ids).size))


def load_dataset(raw_dir, name, runs=None):
    """データセット名 → [(Traj, run_meta), …]。runs=None なら zip 内の全 run。"""
    spec = DATASETS[name]
    zpath = os.path.join(raw_dir, spec["zip"])
    mpath = os.path.join(raw_dir, spec["meta"])
    with open(mpath, encoding="utf-8") as fh:
        meta = json.load(fh)["experiment"]
    by_run = {r["run_name"]: r for r in meta.get("run", [])}
    out = []
    with zipfile.ZipFile(zpath) as zf:
        for entry in sorted(zf.namelist()):
            if not entry.lower().endswith(".txt"):
                continue
            run = os.path.splitext(os.path.basename(entry))[0]
            if runs is not None and run not in runs:
                continue
            rm = by_run.get(run)
            if rm is None:
                # metadata に無い run(旧名など)は幾何が引けないので使わない
                continue
            text = zf.read(entry).decode("utf-8", errors="replace")
            out.append((parse_traj_text(text, run), rm))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 測定(密度・速度)
# ─────────────────────────────────────────────────────────────────────────────
def velocities(tr, half_window_s=SPEED_HALF_WINDOW_S):
    """中心差分による速度 (vx, vy) [m/s] と有効フラグ。

    同一 ID の frame f について (pos[f+W] − pos[f−W]) / (2W/fps)。
    両端に W frame 分の欠けがある標本は無効(NaN)。追跡の欠損 frame は
    「その ID の frame 列に f±W が存在するか」で厳密に判定する(内挿しない)。"""
    W = int(round(half_window_s * tr.fps))
    n = tr.ids.shape[0]
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    # (id, frame) → 行 index の辞書を id ごとに作る
    order = np.lexsort((tr.frame, tr.ids))
    ids_s, fr_s = tr.ids[order], tr.frame[order]
    x_s, y_s = tr.x[order], tr.y[order]
    bounds = np.flatnonzero(np.diff(ids_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    for s, e in zip(starts, ends):
        f = fr_s[s:e]
        if f.size < 2 * W + 1:
            continue
        # frame が連続とは限らないので searchsorted で f−W / f+W を厳密に引く
        lo = np.searchsorted(f, f - W)
        hi = np.searchsorted(f, f + W)
        ok = (lo < f.size) & (hi < f.size)
        ok[ok] = (f[lo[ok]] == f[ok] - W) & (f[hi[ok]] == f[ok] + W)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        dt = 2.0 * W / tr.fps
        vx[order[s + idx]] = (x_s[s + hi[idx]] - x_s[s + lo[idx]]) / dt
        vy[order[s + idx]] = (y_s[s + hi[idx]] - y_s[s + lo[idx]]) / dt
    return vx, vy


def fd_samples(tr, rect, axis, stride_s=SAMPLE_STRIDE_S, steady_trim=STEADY_TRIM):
    """測定区画 rect=(x0,y0,x1,y1) の (t, ρ, v_along, v_speed, n) 標本列。

    ρ = 区画内頭数 ÷ 面積(古典法)。v_along = 流れ方向速度成分の算術平均
    (符号は run 全体の流れ向きで正になるよう揃える)。空(n=0)の frame は捨てる。
    steady は「区画が空でない frame 区間」の中央 (1−2·trim) を 1 とするフラグ。"""
    x0, y0, x1, y1 = rect
    area = (x1 - x0) * (y1 - y0)
    vx, vy = velocities(tr)
    v_ax = vx if axis == "x" else vy
    inside = ((tr.x >= x0) & (tr.x < x1) & (tr.y >= y0) & (tr.y < y1)
              & np.isfinite(v_ax))
    if not inside.any():
        return np.zeros((0, 5))
    # 流れ向き(区画内の中央値の符号)。ring/開境界とも run 内で一貫。
    sgn = 1.0 if float(np.median(v_ax[inside])) >= 0 else -1.0

    step = max(1, int(round(stride_s * tr.fps)))
    fr = tr.frame[inside]
    f_lo, f_hi = int(fr.min()), int(fr.max())
    frames = np.arange(f_lo, f_hi + 1, step)
    # frame ごとに集計(frame 昇順にソートして区間切り出し)
    o = np.argsort(tr.frame[inside], kind="stable")
    fin = tr.frame[inside][o]
    van = v_ax[inside][o] * sgn
    sp = np.hypot(vx[inside], vy[inside])[o]
    rows = []
    lo = np.searchsorted(fin, frames, side="left")
    hi = np.searchsorted(fin, frames, side="right")
    for f, a, b in zip(frames, lo, hi):
        n = int(b - a)
        if n == 0:
            continue
        rows.append((f / tr.fps, n / area, float(van[a:b].mean()),
                     float(sp[a:b].mean()), n))
    if not rows:
        return np.zeros((0, 5))
    out = np.array(rows, dtype=np.float64)
    # 定常窓フラグ(在室区間の中央 80%)
    t = out[:, 0]
    lo_t = t.min() + steady_trim * (t.max() - t.min())
    hi_t = t.max() - steady_trim * (t.max() - t.min())
    steady = ((t >= lo_t) & (t <= hi_t)).astype(np.float64)
    return np.column_stack([out, steady])


def free_spans(rings, line_value, along_axis):
    """線(along_axis = line_value)が通る「多角形内部」の区間列 [(lo,hi), …]。

    外周 + 穴の全辺と線の交点を求め、even-odd 規則で内側区間を切り出す
    (穴あき多角形でもこれで正しい)。壁の bbox を使う近似ではないので、
    uni_corr のように壁に切欠き(ドア枠)がある幾何でも取りこぼさない。"""
    cuts = []
    ai = 1 if along_axis == "y" else 0        # 線を固定する座標の index
    ci = 1 - ai                               # 走査する座標の index
    for ring in rings:
        pts = ring if ring[0] == ring[-1] else list(ring) + [ring[0]]
        for p, q in zip(pts[:-1], pts[1:]):
            a0, a1 = p[ai], q[ai]
            if (a0 <= line_value < a1) or (a1 <= line_value < a0):
                frac = (line_value - a0) / (a1 - a0)
                cuts.append(p[ci] + frac * (q[ci] - p[ci]))
    cuts.sort()
    return [(cuts[k], cuts[k + 1]) for k in range(0, len(cuts) - 1, 2)]


def corridor_rect(spec, run_meta):
    """一方向流コリドーの測定区画 (x0,y0,x1,y1) [m] を metadata から作る。

    通路幅 b は `parameter` の値を一次情報とし(研究文書 §1.3 の指示)、
    WKT の内部区間のうち **幅が b に最も近いもの** を通路本体とみなす。
    測定区画は「流れ方向 2 m × 通路全幅」を通路中央に置く。"""
    b = _num(run_meta.get("parameter", {}).get(spec["width_key"]))
    if b is None:
        return None, None
    rings = wkt_rings(run_meta.get("wkt_geometry"))
    if len(rings) < 2:
        return None, None
    holes = [ring_bbox(r) for r in rings[1:]]
    axis = spec["axis"]
    half = MEAS_LENGTH_M / 2.0
    # 側壁 = 流れ方向の広がりが最大の穴 2 本 → その中点が測定断面の位置
    key = (lambda h: h[3] - h[1]) if axis == "y" else (lambda h: h[2] - h[0])
    side = sorted(holes, key=key, reverse=True)[:2]
    if axis == "y":
        centre = 0.5 * (min(h[1] for h in side) + max(h[3] for h in side))
    else:
        centre = 0.5 * (min(h[0] for h in side) + max(h[2] for h in side))
    spans = free_spans(rings, centre, axis)
    if not spans:
        return None, None
    lo, hi = min(spans, key=lambda s: abs((s[1] - s[0]) - b))
    if abs((hi - lo) - b) > 0.25:            # metadata と WKT が食い違う run は捨てる
        return None, None
    if axis == "y":
        return (lo, centre - half, hi, centre + half), hi - lo
    return (centre - half, lo, centre + half, hi), hi - lo


def bottleneck_line(run_meta, spec):
    """ボトルネックの (開口中心 x, 開口幅 b, 開口の y 座標) を metadata から。

    b は `parameter['b_Exit [m]']` を一次情報とする(WKT の穴の隙間は
    実測で b と最大 0.10 m ずれる = 壁厚の描き方の違い。README に明記)。"""
    b = _num(run_meta.get("parameter", {}).get(spec["width_key"]))
    rings = wkt_rings(run_meta.get("wkt_geometry"))
    holes = [ring_bbox(r) for r in rings[1:]]
    if len(holes) < 2 or b is None:
        return None
    left = min(holes, key=lambda h: h[0])
    right = max(holes, key=lambda h: h[2])
    gap_lo, gap_hi = left[2], right[0]
    x_centre = 0.5 * (gap_lo + gap_hi)
    y_line = 0.5 * (min(h[1] for h in holes) + max(h[3] for h in holes))
    return dict(x_centre=x_centre, b=b, y_line=y_line,
                wkt_gap=gap_hi - gap_lo)


def bottleneck_flow(tr, line, trim=0.10):
    """開口線 y=y_line の通過時刻から流量 J [1/s] と specific flow J/b [1/(m·s)]。

    各 ID の「初めて y_line を上から下へ跨いだ時刻」を線形内挿で求める。
    定常部だけを使う(通過ランク順の中央 (1−2·trim))。"""
    y_line = line["y_line"]
    order = np.lexsort((tr.frame, tr.ids))
    ids_s, fr_s = tr.ids[order], tr.frame[order]
    y_s, x_s = tr.y[order], tr.x[order]
    bounds = np.flatnonzero(np.diff(ids_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [ids_s.shape[0]]])
    times = []
    for s, e in zip(starts, ends):
        yy = y_s[s:e]
        cross = np.flatnonzero((yy[:-1] > y_line) & (yy[1:] <= y_line))
        if cross.size == 0:
            continue
        k = int(cross[0])
        y0, y1 = yy[k], yy[k + 1]
        f0, f1 = fr_s[s + k], fr_s[s + k + 1]
        frac = 0.0 if y0 == y1 else (y0 - y_line) / (y0 - y1)
        # 開口の外(横に大きく外れた通過)は数えない
        xc = x_s[s + k] + frac * (x_s[s + k + 1] - x_s[s + k])
        if abs(xc - line["x_centre"]) > 0.5 * line["b"] + 0.30:
            continue
        times.append((f0 + frac * (f1 - f0)) / tr.fps)
    times = np.sort(np.array(times, dtype=np.float64))
    if times.size < 10:
        return None
    lo = int(round(trim * times.size))
    hi = int(round((1.0 - trim) * times.size))
    sel = times[lo:hi]
    dur = float(sel[-1] - sel[0])
    if dur <= 0:
        return None
    j = (sel.size - 1) / dur
    return dict(n_cross=int(times.size), n_steady=int(sel.size),
                duration_s=dur, J=j, J_specific=j / line["b"],
                t_first=float(times[0]), t_last=float(times[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# 集計(基本図のビン化 + 10/90 パーセンタイル包絡線)
# ─────────────────────────────────────────────────────────────────────────────
def bin_fd(rho, v, edges):
    """密度ビンごとの (中心, n, mean, sd, p10, p50, p90)。RiMEA Test 16 流の包絡線。"""
    rows = []
    idx = np.digitize(rho, edges) - 1
    for k in range(edges.size - 1):
        sel = idx == k
        n = int(sel.sum())
        if n < 5:                     # 標本 5 未満のビンは出さない(ばらつき過大)
            continue
        vv = v[sel]
        rows.append((0.5 * (edges[k] + edges[k + 1]), n, float(vv.mean()),
                     float(vv.std(ddof=1)) if n > 1 else 0.0,
                     float(np.percentile(vv, 10)), float(np.percentile(vv, 50)),
                     float(np.percentile(vv, 90))))
    return rows


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(
                (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in r) + "\n")
    return path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
def build(raw_dir, out_dir, bin_width=0.1, rho_max=6.0, verbose=True):
    """生データ → 派生 CSV 一式。戻り値は成果物の要約 dict。"""
    os.makedirs(out_dir, exist_ok=True)
    edges = np.arange(0.0, rho_max + bin_width, bin_width)
    summary = {"license": "CC BY 4.0",
               "source": "Forschungszentrum Juelich, Pedestrian Dynamics Data Archive",
               "source_url": "https://ped.fz-juelich.de/database",
               "accessed": "2026-08-05", "datasets": {}, "files": {}}

    fd_rows = []          # 全一方向流の標本
    run_rows = []         # run ごとの要約
    for name in ("hermes_uni_closed", "hermes_uni_open", "basigo_uni_corr"):
        spec = DATASETS[name]
        path = os.path.join(raw_dir, spec["zip"])
        if not os.path.exists(path):
            summary["datasets"][name] = {"status": "missing", "path": path}
            continue
        got = load_dataset(raw_dir, name)
        n_s = 0
        for tr, rm in got:
            rect, b = corridor_rect(spec, rm)
            if rect is None:
                continue
            s = fd_samples(tr, rect, spec["axis"])
            if s.shape[0] == 0:
                continue
            for t, rho, van, sp, n, steady in s:
                fd_rows.append((name, tr.run, round(float(t), 3), float(rho),
                                float(van), float(sp), int(n), int(steady)))
            n_s += s.shape[0]
            sel = s[:, 5] > 0
            run_rows.append((name, tr.run, spec["doi"], float(b), tr.fps, tr.unit,
                             tr.n_ped, int(s.shape[0]),
                             float(s[sel, 1].mean()) if sel.any() else float("nan"),
                             float(s[sel, 2].mean()) if sel.any() else float("nan"),
                             round(float(rect[0]), 3), round(float(rect[1]), 3),
                             round(float(rect[2]), 3), round(float(rect[3]), 3)))
        summary["datasets"][name] = {"status": "ok", "runs": len(got), "samples": n_s,
                                     "doi": spec["doi"]}
        if verbose:
            print(f"  {name}: {len(got)} runs, {n_s} FD samples")

    if fd_rows:
        _write_csv(os.path.join(out_dir, "juelich_fd_samples.csv"),
                   ["dataset", "run", "t_s", "rho_per_m2", "v_along_mps",
                    "speed_mps", "n_in_area", "steady"], fd_rows)
        _write_csv(os.path.join(out_dir, "juelich_fd_runs.csv"),
                   ["dataset", "run", "doi", "corridor_width_m", "fps", "raw_unit",
                    "n_pedestrians", "n_samples", "rho_mean_steady",
                    "v_along_mean_steady", "rect_x0", "rect_y0", "rect_x1", "rect_y1"],
                   run_rows)
        hdr = ["rho_per_m2", "n", "v_mean_mps", "v_sd_mps",
               "v_p10_mps", "v_p50_mps", "v_p90_mps"]
        arr = np.array([(r[3], r[4]) for r in fd_rows if r[7] == 1], dtype=np.float64)
        _write_csv(os.path.join(out_dir, "juelich_fd_binned.csv"), hdr,
                   bin_fd(arr[:, 0], arr[:, 1], edges))
        # データセット別(境界条件で FD が変わるため。閉境界 ug が周期シムの直接の対応物)
        for ds in ("hermes_uni_closed", "hermes_uni_open", "basigo_uni_corr"):
            a = np.array([(r[3], r[4]) for r in fd_rows if r[0] == ds and r[7] == 1],
                         dtype=np.float64)
            if a.size:
                _write_csv(os.path.join(out_dir, f"juelich_fd_binned_{ds}.csv"), hdr,
                           bin_fd(a[:, 0], a[:, 1], edges))
                summary["datasets"][ds]["steady_samples"] = int(a.shape[0])
        summary["fd_total_samples"] = len(fd_rows)
        summary["fd_steady_samples"] = int(arr.shape[0])

    # ── ボトルネック ──
    spec = DATASETS["hermes_bottleneck"]
    bpath = os.path.join(raw_dir, spec["zip"])
    if os.path.exists(bpath):
        rows = []
        for tr, rm in load_dataset(raw_dir, "hermes_bottleneck"):
            line = bottleneck_line(rm, spec)
            if line is None:
                continue
            fl = bottleneck_flow(tr, line)
            if fl is None:
                continue
            rows.append((tr.run, spec["doi"], line["b"], round(line["wkt_gap"], 3),
                         rm.get("number_participants"), fl["n_cross"], fl["n_steady"],
                         round(fl["duration_s"], 3), fl["J"], fl["J_specific"]))
        if rows:
            _write_csv(os.path.join(out_dir, "juelich_bottleneck_flow.csv"),
                       ["run", "doi", "b_exit_m", "wkt_gap_m", "n_participants",
                        "n_crossed", "n_steady", "duration_s", "J_per_s",
                        "J_specific_per_ms"], rows)
        summary["datasets"]["hermes_bottleneck"] = {
            "status": "ok", "runs": len(rows), "doi": spec["doi"]}
        if verbose:
            print(f"  hermes_bottleneck: {len(rows)} runs")
    else:
        summary["datasets"]["hermes_bottleneck"] = {"status": "missing", "path": bpath}

    # ── 90°交差流(相対判定用。単方向 FD より下に来るかだけを見る) ──
    spec = DATASETS["basigo_crossing90"]
    cpath = os.path.join(raw_dir, spec["zip"])
    if os.path.exists(cpath):
        rows = []
        got = load_dataset(raw_dir, "basigo_crossing90")
        for tr, rm in got:
            rings = wkt_rings(rm.get("wkt_geometry"))
            if len(rings) < 5:
                continue
            holes = [ring_bbox(r) for r in rings[1:]]
            # 4 隅のブロック(面積上位 4 つ。柱ありの run では小さな穴が増える)に
            # 囲まれた中央部 = 交差ゾーン。その中心に 2×2 m を置く。
            holes = sorted(holes, key=lambda h: (h[2] - h[0]) * (h[3] - h[1]),
                           reverse=True)[:4]
            if len(holes) < 4:
                continue
            ox = sum(0.5 * (h[0] + h[2]) for h in holes) / 4.0
            oy = sum(0.5 * (h[1] + h[3]) for h in holes) / 4.0
            lft = [h for h in holes if 0.5 * (h[0] + h[2]) < ox]
            rgt = [h for h in holes if 0.5 * (h[0] + h[2]) > ox]
            bot = [h for h in holes if 0.5 * (h[1] + h[3]) < oy]
            top = [h for h in holes if 0.5 * (h[1] + h[3]) > oy]
            if not (lft and rgt and bot and top):
                continue
            cx = 0.5 * (max(h[2] for h in lft) + min(h[0] for h in rgt))
            cy = 0.5 * (max(h[3] for h in bot) + min(h[1] for h in top))
            rect = (cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0)
            vx, vy = velocities(tr)
            m = ((tr.x >= rect[0]) & (tr.x < rect[2]) & (tr.y >= rect[1])
                 & (tr.y < rect[3]) & np.isfinite(vx))
            if not m.any():
                continue
            step = max(1, int(round(SAMPLE_STRIDE_S * tr.fps)))
            fr = tr.frame[m]
            sp = np.hypot(vx[m], vy[m])
            for f in range(int(fr.min()), int(fr.max()) + 1, step):
                sel = fr == f
                n = int(sel.sum())
                if n == 0:
                    continue
                rows.append((tr.run, round(f / tr.fps, 3), n / 4.0,
                             float(sp[sel].mean()), n))
        if rows:
            _write_csv(os.path.join(out_dir, "juelich_crossing90_fd.csv"),
                       ["run", "t_s", "rho_per_m2", "speed_mps", "n_in_area"], rows)
        summary["datasets"]["basigo_crossing90"] = {
            "status": "ok", "runs": len(got), "samples": len(rows),
            "doi": spec["doi"],
            "note": "central 2x2 m; relative judgement only (multi-directional)"}
        if verbose:
            print(f"  basigo_crossing90: {len(got)} runs, {len(rows)} samples")
    else:
        summary["datasets"]["basigo_crossing90"] = {"status": "missing", "path": cpath}

    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".csv"):
            p = os.path.join(out_dir, fn)
            summary["files"][fn] = {"bytes": os.path.getsize(p), "sha256": sha256_of(p)}
    with open(os.path.join(out_dir, "SOURCES.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 較正側から使うローダ(派生 CSV → numpy)
# ─────────────────────────────────────────────────────────────────────────────
def load_binned(path):
    """juelich_fd_binned.csv → dict of arrays。"""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {k: np.atleast_1d(raw[k]) for k in raw.dtype.names}


def reference_speed(binned, rho, max_gap=0.15):
    """ビン化 FD を線形内挿して ρ の参照速度を返す。近傍ビンが無ければ NaN。"""
    xs, ys = binned["rho_per_m2"], binned["v_mean_mps"]
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    out = np.interp(rho, xs, ys, left=np.nan, right=np.nan)
    near = np.min(np.abs(rho[:, None] - xs[None, :]), axis=1)
    out[near > max_gap] = np.nan
    return out


def reference_envelope(binned, rho, max_gap=0.15):
    """ρ における (p10, p90) を線形内挿(RiMEA Test 16 流の包絡線)。"""
    xs = binned["rho_per_m2"]
    o = np.argsort(xs)
    xs = xs[o]
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    lo = np.interp(rho, xs, binned["v_p10_mps"][o], left=np.nan, right=np.nan)
    hi = np.interp(rho, xs, binned["v_p90_mps"][o], left=np.nan, right=np.nan)
    near = np.min(np.abs(rho[:, None] - xs[None, :]), axis=1)
    lo[near > max_gap] = np.nan
    hi[near > max_gap] = np.nan
    return lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=os.path.join("data", "juelich_ped"))
    ap.add_argument("--out", default=os.path.join("reference", "physics_bench", "data"))
    ap.add_argument("--bin-width", type=float, default=0.1)
    args = ap.parse_args(argv)
    print(f"[juelich] raw={args.raw} out={args.out}")
    s = build(args.raw, args.out, bin_width=args.bin_width)
    total = sum(v["bytes"] for v in s["files"].values())
    print(f"[juelich] wrote {len(s['files'])} CSV, total {total/1e6:.2f} MB")
    for k, v in s["files"].items():
        print(f"   {k}: {v['bytes']} B  sha256={v['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
