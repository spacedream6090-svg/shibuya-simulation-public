"""物理 2 レバー(第154)の検証ハーネス — 密度適応 dt(B)と近傍 cap 引き下げ(D)。

    python scripts/bench_physics_levers.py [--out DIR] [--quick] [--sizes 600,1500,3000]

何を測るか
----------
本選(conf/finals_observe.yaml)の 3 ゾーンを模した**合成シナリオ**を、本体エンジン
(`society.world.orca_core.OrcaCrowd` / `society.physics._CalibratedCrowd`)で回し、
4 構成の **コストと破綻統計** を前後比較する:

    base   … 現行(dt_sub=0.1 固定・neighbor_cap=12)
    B      … 密度適応 dt(在場者数 → dt 係数。積分時間の総量は保存)
    D      … 近傍 cap 7(Ballerini の位相的近傍 6-7)
    B+D    … 両方

破綻統計(すべて「値が悪化していないこと」を人が読んで判断するために出す):
    min_gap_m          体表間の最小すき間 [m](負 = 重なり)
    overlap_pair_rate  重なっているペアの割合
    jump_max_m         1 サブステップの最大変位 [m]
    accel_p99          |Δv|/dt_eff の 99 分位 [m/s²](ヒストグラム分解能 0.1)
    wall_clear_m       壁面クリアランス [m](負 = 壁へめり込み。corridor シナリオのみ)
    speed_mean         平均速さ [m/s](= 基本図の作業点が動いていないかの粗い代理)

設計上の約束
------------
  - **`reference/` は 1 バイトも読まない・書かない**(シナリオも指標もこのファイルで完結)。
  - 回すのは **src/society の本体エンジン**(ベンチ専用の派生クラスを作らない)。
  - 乱数は初期配置だけ(seed 固定)= 同じコマンドで同じ JSON が出る。
  - 出力の既定は `experiments/`(.gitignore 済み)。

シナリオ(本選ゾーンの面積 840 m² = スクランブル polygon 実測をそのまま採る)
------------------------------------------------------------------------------
  crossing … 29.0 × 29.0 m の開放平面・engine=orca・壁なし・4 方向の交差流
             (= スクランブル交差点。ORCA を置いた理由そのもの)
  corridor … 93.3 × 9.0 m・engine=sfm・両側に壁・対向流(= センター街を同面積に伸ばした形)
  どちらも「同じ広さに人数を増やす」= 夕方ラッシュの密度上昇そのものを掃く。

  初期配置は**格子**(乱数ゼロ・決定論)。境界に達した個体は**希望方向を反転**させる
  (位置は連続 = 瞬間移動を作らないので jump/accel の統計が汚れない。周期境界のゴーストを
  持ち込むより単純で、密度を一定に保てる)。

★正直な限界(幾何): 本リポジトリの体半径は平均 0.299 m なので、**重なりゼロで置ける
  上限密度は六方最密で 3.22 人/m²**。n=3000 / 840 m² = 3.57 人/m² はこれを超えるので、
  min_gap < 0 は **base を含む全構成で幾何的に不可避**である。その水準で読むべきは
  「B/D が base より悪化させたか」であって「重なりが出たかどうか」ではない。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society import physics as P                              # noqa: E402
from society.world import orca_core, sfm_core                 # noqa: E402
from society.world.indoor_flow import body_radius, desired_speed  # noqa: E402

# ---- 本選の作業点(conf/finals_observe.yaml)------------------------------- #
DT_BASE = 0.1               # 全ゾーン dt_sub(8/22 J1 判定で 0.05 → 0.1)
CAP_BASE = 12               # ゾーン宣言の neighbor_cap
CAP_LOW = 7                 # レバーD の候補(Ballerini 6-7)
THRESHOLDS = ((500, 2.0), (2000, 4.0))
RECHECK = 20
FAR_KW = dict(far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5, far_taper_m=1.0)

AREA = 840.0                # スクランブル polygon の実測面積 [m²](両シナリオ共通)
CROSS_L = 29.0              # 29×29 = 841 m²
CORR_W = 9.0                # センター街の実幅。長さは面積を揃えて 93.3 m
CORR_L = AREA / CORR_W
HEX_MAX_RHO = 3.22          # 半径平均 0.299 m の六方最密 = 重なりゼロで置ける上限 [人/m²]
T_TOTAL = 20.0              # 積分する秒数(quick は 6 s)
ACC_BIN, ACC_BINS = 0.1, 1000


# =========================================================================== #
# 共通ユーティリティ
# =========================================================================== #
def agent_params(n):
    """(v0, radius)。本体と同じ安定ハッシュ(乱数ゼロ)。"""
    v0 = np.array([desired_speed(i) for i in range(n)], dtype=np.float64)
    rad = np.array([body_radius(i) for i in range(n)], dtype=np.float64)
    return v0, rad


def lattice(n, w, h, margin=0.4):
    """格子の初期配置(**乱数ゼロ・完全決定論**)。同じ n なら全構成で同じ初期状態。

    格子にするのは棄却サンプリングが高密度で収束しないため(置けなかった個体を
    重ねて置くと、初期の巨大斥力が破綻統計を全部支配してしまう)。
    """
    iw, ih = w - 2 * margin, h - 2 * margin
    cols = max(1, int(round(math.sqrt(n * iw / ih))))
    rows = int(math.ceil(n / cols))
    xs = margin + (np.arange(cols) + 0.5) * (iw / cols)
    ys = margin + (np.arange(rows) + 0.5) * (ih / rows)
    out = np.zeros((n, 2))
    for i in range(n):
        out[i] = (xs[i % cols], ys[i // cols])
    return out


class Stats:
    """破綻統計の入れ物(加速度は本体と同じ固定ビンのヒストグラム = メモリ O(1))。"""

    def __init__(self):
        self.hist = np.zeros(ACC_BINS, dtype=np.int64)
        self.n = 0
        self.jump_max = 0.0
        self.min_gap = math.inf
        self.wall_clear = math.inf
        self.ov_pairs = 0
        self.tot_pairs = 0
        self.speed_sum = 0.0
        self.speed_n = 0
        self.sep_iters_max = 0        # 分離パスが上限に当たったかの監視(ORCA のみ)

    def add_motion(self, pos, prev_pos, vel, prev_vel, dt, mask):
        if not mask.any():
            return
        disp = np.linalg.norm(pos[mask] - prev_pos[mask], axis=1)
        self.jump_max = max(self.jump_max, float(disp.max()))
        dv = np.linalg.norm(vel[mask] - prev_vel[mask], axis=1) / dt
        idx = np.clip(dv / ACC_BIN, 0.0, float(ACC_BINS - 1)).astype(np.int64)
        self.hist += np.bincount(idx, minlength=ACC_BINS)
        self.n += int(mask.sum())
        sp = np.linalg.norm(vel[mask], axis=1)
        self.speed_sum += float(sp.sum())
        self.speed_n += int(mask.sum())

    def add_overlap(self, pos, radius):
        gap = orca_core.min_gap(pos, radius)
        if math.isfinite(gap):
            self.min_gap = min(self.min_gap, gap)
        reach = 2.0 * float(radius.max())
        ii, jj = sfm_core.neighbor_pairs(pos, reach)
        if ii.size:
            up = ii < jj
            ii, jj = ii[up], jj[up]
            d = np.linalg.norm(pos[ii] - pos[jj], axis=1)
            self.ov_pairs += int(np.count_nonzero(d < radius[ii] + radius[jj]))
        n = pos.shape[0]
        self.tot_pairs += n * (n - 1) // 2

    def add_wall_clearance(self, pos, radius, y_lo, y_hi):
        """水平な 2 枚壁(y=y_lo / y=y_hi)からのクリアランス(負 = めり込み)。"""
        c = np.minimum(pos[:, 1] - y_lo, y_hi - pos[:, 1]) - radius
        self.wall_clear = min(self.wall_clear, float(c.min()))

    def p99(self):
        if not self.n:
            return None
        target = 0.99 * self.n
        acc = 0
        for i, c in enumerate(self.hist):
            acc += int(c)
            if acc >= target:
                return (i + 1) * ACC_BIN
        return ACC_BINS * ACC_BIN

    def out(self):
        return {
            "min_gap_m": (None if not math.isfinite(self.min_gap)
                          else round(self.min_gap, 4)),
            "overlap_pair_rate": (round(self.ov_pairs / self.tot_pairs, 6)
                                  if self.tot_pairs else None),
            "jump_max_m": round(self.jump_max, 4),
            "accel_p99": self.p99(),
            "wall_clear_m": (None if not math.isfinite(self.wall_clear)
                             else round(self.wall_clear, 4)),
            "speed_mean": (round(self.speed_sum / self.speed_n, 4)
                           if self.speed_n else None),
            "sep_iters_max": int(self.sep_iters_max),
        }


def dt_schedule(n_members, t_total, dt_base, adaptive):
    """サブステップ幅の列。**規則そのものは本体(`physics.next_dt` / `_dt_factor`)を呼ぶ**
    = ベンチと本体が食い違いようがない(コピーを持たない)。

    塊(`RECHECK` 回)ごとに係数を引き直す。在場者数は本ハーネスでは一定なので、
    係数は塊ごとに同じ値になる。
    """
    if not adaptive:
        n = max(1, int(round(t_total / dt_base)))
        return [dt_base] * n
    out = []
    t, k = 0.0, 0
    n_max = max(1, int(round(t_total / dt_base)))
    factor = 1.0
    while k < n_max:
        if k % RECHECK == 0:
            factor = P._dt_factor(THRESHOLDS, n_members)
        dt_eff, t = P.next_dt(t, t_total, dt_base, factor)
        if dt_eff <= 0.0:
            break
        out.append(dt_eff)
        k += 1
        if t >= t_total:
            break
    return out


# =========================================================================== #
# シナリオ A: 交差点(ORCA・壁なし・4 方向交差流・正方トーラス)
# =========================================================================== #
def _reflect(e, pos, axis, lo, hi):
    """境界に達した個体の**希望方向だけ**を反転する(位置は連続 = 瞬間移動を作らない)。"""
    out = pos[:, axis] < lo
    e[out, axis] = np.abs(e[out, axis])
    out = pos[:, axis] > hi
    e[out, axis] = -np.abs(e[out, axis])


def run_crossing(n, cap, adaptive, cognitive, t_total=T_TOTAL, dt_base=DT_BASE,
                 warm_frac=0.3, sep_iters=64):
    v0, radius = agent_params(n)
    pos = lattice(n, CROSS_L, CROSS_L)
    # 4 方向(東西南北)。id 昇順の決定論割当 = 乱数を使わない。
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    e = dirs[np.arange(n) % 4].copy()
    vel = e * v0[:, None]
    cog = ({"cognitive": True, "cog_neighbors": cap, "cog_sectors": 16,
            "cog_fov_deg": 360.0} if cognitive else {})
    sched = dt_schedule(n, t_total, dt_base, adaptive)
    st = Stats()
    warm = int(len(sched) * warm_frac)
    t0 = time.perf_counter()
    for si, dt in enumerate(sched):
        crowd = orca_core.OrcaCrowd(
            pos, vel, pos + e * 60.0, v0, radius, walls=(),
            neighbor_cap=cap, tau=2.0, tau_obst=2.0, neighbor_dist=10.0,
            wall_range=2.0, v_max_factor=1.3, arrive_radius=1.0,
            pref_noise=0.0, rng=None, radius_margin=0.05,
            separation_iters=sep_iters, **cog)
        prev_p, prev_v = crowd.pos.copy(), crowd.vel.copy()
        crowd.step(dt)
        st.sep_iters_max = max(st.sep_iters_max, int(crowd.last_sep_iters))
        pos, vel = crowd.pos.copy(), crowd.vel.copy()
        _reflect(e, pos, 0, 0.0, CROSS_L)
        _reflect(e, pos, 1, 0.0, CROSS_L)
        if si >= warm:
            st.add_motion(pos, prev_p, vel, prev_v, dt, np.ones(n, dtype=bool))
            if si % 5 == 0:
                st.add_overlap(pos, radius)
    wall_s = time.perf_counter() - t0
    return _pack(st, wall_s, sched, t_total)


# =========================================================================== #
# シナリオ B: 通路(SFM・両側に壁・対向流・長手方向トーラス)
# =========================================================================== #
def run_corridor(n, cap, adaptive, cognitive, t_total=T_TOTAL, dt_base=DT_BASE,
                 warm_frac=0.3):
    v0, radius = agent_params(n)
    pos = lattice(n, CORR_L, CORR_W)
    e = np.zeros((n, 2))
    e[::2, 0] = 1.0
    e[1::2, 0] = -1.0
    vel = e * v0[:, None]
    walls = (((-10.0, 0.0), (CORR_L + 10.0, 0.0)),
             ((-10.0, CORR_W), (CORR_L + 10.0, CORR_W)))
    cog = ({"cognitive": True, "cog_neighbors": cap, "cog_sectors": 16,
            "cog_fov_deg": 360.0} if cognitive else {})
    sched = dt_schedule(n, t_total, dt_base, adaptive)
    st = Stats()
    warm = int(len(sched) * warm_frac)
    t0 = time.perf_counter()
    for si, dt in enumerate(sched):
        crowd = P._CalibratedCrowd(
            pos, vel, pos + e * 60.0, v0, radius=radius, rng=None, noise=0.0,
            arrive_radius=1.0, walls=walls, wall_range=2.0,
            neighbor_cap=cap, v_max_factor=1.3, **FAR_KW, **cog)
        prev_p, prev_v = crowd.pos.copy(), crowd.vel.copy()
        crowd.step(dt)
        pos, vel = crowd.pos.copy(), crowd.vel.copy()
        _reflect(e, pos, 0, 0.0, CORR_L)
        if si >= warm:
            st.add_motion(pos, prev_p, vel, prev_v, dt, np.ones(n, dtype=bool))
            st.add_wall_clearance(pos, radius, 0.0, CORR_W)
            if si % 5 == 0:
                st.add_overlap(pos, radius)
    wall_s = time.perf_counter() - t0
    return _pack(st, wall_s, sched, t_total)


def _pack(st, wall_s, sched, t_total):
    out = st.out()
    out["substeps"] = len(sched)
    out["integrated_s"] = round(float(sum(sched)), 9)
    out["time_accounting_err_s"] = round(abs(sum(sched) - t_total), 12)
    out["wall_s"] = round(wall_s, 4)
    out["wall_ms_per_sim_s"] = round(1000.0 * wall_s / t_total, 3)
    return out


# =========================================================================== #
# 掃引
# =========================================================================== #
CONFIGS = (
    ("base", CAP_BASE, False),
    ("B_adaptive_dt", CAP_BASE, True),
    ("D_cap7", CAP_LOW, False),
    ("B+D", CAP_LOW, True),
)


def sweep(sizes, t_total, cognitive):
    rows = []
    for scen, fn, area in (("crossing", run_crossing, CROSS_L * CROSS_L),
                           ("corridor", run_corridor, CORR_W * CORR_L)):
        for n in sizes:
            if n / area > HEX_MAX_RHO:
                print(f"  [note] {scen} n={n}: 密度 {n / area:.2f} > 六方最密 "
                      f"{HEX_MAX_RHO} = **全構成で重なり不可避**(幾何の限界)",
                      flush=True)
            base = None
            for name, cap, adaptive in CONFIGS:
                r = fn(n, cap, adaptive, cognitive, t_total=t_total)
                r.update(scenario=scen, n=n, density=round(n / area, 3),
                         config=name, neighbor_cap=cap, adaptive=adaptive,
                         cognitive=cognitive)
                if name == "base":
                    base = r
                r["speedup_x"] = (round(base["wall_s"] / r["wall_s"], 3)
                                  if r["wall_s"] > 0 else None)
                rows.append(r)
                print(f"  {scen:9s} n={n:5d} {name:14s} "
                      f"wall={r['wall_s']:8.2f}s x{r['speedup_x']:<6} "
                      f"sub={r['substeps']:5d} gap={r['min_gap_m']} "
                      f"jump={r['jump_max_m']} p99={r['accel_p99']} "
                      f"wall_clear={r['wall_clear_m']} v={r['speed_mean']}",
                      flush=True)
    return rows


def sweep_sep(sizes, t_total, seps, cognitive):
    """レバーS 掃引: 事後分離パスの反復上限を **本選の作業点**(B+D = 適応 dt + cap 7)で振る。

    ORCA だけの話なので crossing シナリオのみ。base(レバー無し・上限 64)を並べて出す。
    """
    rows = []
    area = CROSS_L * CROSS_L
    for n in sizes:
        ref = run_crossing(n, CAP_BASE, False, cognitive, t_total=t_total,
                           sep_iters=64)
        ref.update(scenario="crossing", n=n, density=round(n / area, 3),
                   config="base(no levers)", neighbor_cap=CAP_BASE,
                   adaptive=False, cognitive=cognitive, separation_iters=64)
        ref["speedup_x"] = 1.0
        rows.append(ref)
        _print_sep(ref)
        base = None
        for sep in seps:
            r = run_crossing(n, CAP_LOW, True, cognitive, t_total=t_total,
                             sep_iters=sep)
            r.update(scenario="crossing", n=n, density=round(n / area, 3),
                     config=f"B+D sep{sep}", neighbor_cap=CAP_LOW,
                     adaptive=True, cognitive=cognitive, separation_iters=sep)
            base = base or r
            r["speedup_x"] = round(ref["wall_s"] / r["wall_s"], 3)
            r["speedup_vs_sep64_x"] = round(base["wall_s"] / r["wall_s"], 3)
            rows.append(r)
            _print_sep(r)
    return rows


def _print_sep(r):
    print(f"  n={r['n']:5d} {r['config']:16s} wall={r['wall_s']:7.2f}s "
          f"x{r['speedup_x']:<6} gap={r['min_gap_m']:+.4f} "
          f"ov={r['overlap_pair_rate']} jump={r['jump_max_m']} "
          f"p99={r['accel_p99']} v={r['speed_mean']} "
          f"sep_hit={r['sep_iters_max']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPO, "experiments",
                                                  "physics_levers"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sizes", default="600,1500,3000")
    ap.add_argument("--cognitive", action="store_true",
                    help="本選と同じ認知的近傍 ON で測る(cap は cog_neighbors へ入る)")
    ap.add_argument("--sep", default="",
                    help="レバーS 掃引(例 64,24,16)。指定すると crossing のみを"
                         " B+D の作業点で回し、分離パス上限だけを振る")
    args = ap.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    t_total = 6.0 if args.quick else T_TOTAL
    seps = tuple(int(s) for s in args.sep.split(",") if s.strip())
    print(f"[bench] sizes={sizes} t_total={t_total}s dt_base={DT_BASE} "
          f"cognitive={args.cognitive} sep={seps or '-'}", flush=True)
    rows = (sweep_sep(sizes, t_total, seps, args.cognitive) if seps
            else sweep(sizes, t_total, args.cognitive))
    os.makedirs(args.out, exist_ok=True)
    blob = {
        "meta": {"dt_base": DT_BASE, "t_total": t_total, "sizes": list(sizes),
                 "thresholds": [list(p) for p in THRESHOLDS],
                 "recheck_every": RECHECK, "cap_base": CAP_BASE,
                 "cap_low": CAP_LOW, "cognitive": args.cognitive,
                 "sep_sweep": list(seps),
                 "python": platform.python_version(), "numpy": np.__version__},
        "rows": rows,
    }
    path = os.path.join(args.out, "levers_sep.json" if seps else "levers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()
