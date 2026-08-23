"""物理レバーの検証ハーネス — 第154(適応 dt・近傍 cap・分離反復)と第155 C1(所有の距離有界化)。

    python scripts/bench_physics_levers.py [--out DIR] [--quick] [--sizes 600,1500,3000]
    python scripts/bench_physics_levers.py --ownership 0,50,100,200 [--own-sizes 600,2000]

`--ownership` は**合成シナリオではなく本体そのもの**を回す(第155 C1)。理由は、C1 が
変えるのはエンジンの中身ではなく「**誰をゾーンに載せるか**」だからで、合成シナリオ側に
X を注入しても意味のある数字が出ない(そこには所有もゲートも経路も無い)。代わりに
`conf/zones_shibuya.yaml` の実 3 ゾーンを mock LLM で回し、X ごとに

  ・所有人数(zone_occupancy)/ 密度(zone_density_mean)/ 滞在(zone_dwell_mean_s)
  ・実測の局所密度(Perception.body.local_density = polygon 近傍の**本当の**密度)
  ・基本図の作業点(実効速度 = ゾーン内グラフ経路長 span_m ÷ 滞在 dwell_s)
  ・破綻統計(体表間 min_gap / 1 サブステップ変位 jump_max / 加速度 p99 / 逆走率 /
    グラフ復帰の跳び handover_jump / 分離反復の上限張り付き)
  ・軌跡差(X=0 との最終座標のずれ・退場 step のずれ)
  ・コスト(physics.phase の実測秒 / step)

を並べて出す。**判断材料を出すだけで合否は判定しない**(この repo の他のベンチと同じ)。

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


# =========================================================================== #
# 第155 C1: 所有の距離有界化(**本体を回す**。合成シナリオでは測れない)
# =========================================================================== #
OWN_PROFILE = os.path.join(_REPO, "conf", "zones_shibuya.yaml")


def own_spec(raw):
    """掃引の 1 水準。`"0"`/`"off"` = 無制限 / 数値 = euclid X [m] / `"route"` = route_arrival。

    Returns: (label, dotlist, ownership_max_dist_m)
    """
    s = str(raw).strip().lower()
    if s in ("route", "route_arrival", "arrival"):
        return "route", ["physics_levers.ownership_mode=route_arrival"], None
    x = float(s)
    if x <= 0.0:
        return "off", [], 0.0
    return f"euclid{int(x)}", ["physics_levers.ownership_mode=euclid",
                               f"physics_levers.ownership_max_dist_m={x}"], x


def _own_run(n, steps, spec, seed=42):
    """1 水準ぶん本体を steps 回す(`spec` は `own_spec` が返す 3 つ組)。"""
    from society import physics as _P
    from society.config import load_config
    from society.engine import scheduler
    from society.engine.simulation import Simulation

    label, dots, _x = spec
    cfg = load_config([f"run.seed={seed}", f"run.n_agents={n}",
                       f"run.n_steps={steps}", f"run.name=own_{label}_n{n}",
                       "model.backend=mock", "observer.snapshot_every=100000"]
                      + list(dots),
                      profile=OWN_PROFILE)
    sim = Simulation(cfg, out_dir=os.path.join(_REPO, "experiments",
                                               "_own", f"{label}_n{n}"))
    box = {"phys_s": 0.0}
    real_phase = _P.phase

    def timed(s, step, sim_min):
        t = time.perf_counter()
        try:
            return real_phase(s, step, sim_min)
        finally:
            box["phys_s"] += time.perf_counter() - t

    _P.phase = timed
    per_zone: dict = {}
    try:
        t0 = time.perf_counter()
        for step in range(steps):
            scheduler.run_step(sim, step)
            for zid, z in (getattr(sim, "_phys_state", None) or {}).get(
                    "by_zone", {}).items():
                acc = per_zone.setdefault(zid, {"occ": [], "dens": [],
                                                "wait": [], "sub": []})
                acc["occ"].append(float(z["occupancy_mean"]))
                acc["dens"].append(float(z["density"]))
                acc["wait"].append(int(z["waiting"]))
                acc["sub"].append(int(z["sub_steps"]))
        wall = time.perf_counter() - t0
    finally:
        _P.phase = real_phase
    return sim, box["phys_s"], wall, per_zone


def _own_metrics(sim, phys_s, wall, per_zone, steps):
    """1 ラン ぶんの観測量を dict に畳む(判定はしない)。"""
    from society import physics as _P

    sc = _P.scalars(sim) or {}
    cont = _P.continuity(sim) or {}
    span: dict = {}
    speeds: list = []
    exit_step: dict = {}
    for e in sim.logger.events:
        if e.kind != "zone_gate":
            continue
        p = e.payload
        key = (e.agent_id, p.get("zone"))
        if p.get("dir") == "enter":
            span[key] = float(p.get("span_m", 0.0))
        elif p.get("dir") == "exit":
            exit_step.setdefault(key, int(e.step))
            d = float(p.get("dwell_s", 0.0))
            s = span.pop(key, None)
            if s is not None and d > 0.0:
                speeds.append(s / d)
    dens_body = [float(b["local_density"])
                 for b in (getattr(a, "_phys_body", None) for a in sim.agents)
                 if b and b.get("local_density") is not None]
    blocked, body_v = [], []
    for a in sim.agents:
        b = getattr(a, "_phys_body", None)
        if not b or b.get("blocked") is None:
            continue
        blocked.append(float(b["blocked"]))
        # blocked = 1 − v_mean/v0 なので、実測の平均速さは v0·(1−blocked) で戻せる。
        # ★これが基本図の縦軸で、`v_eff`(= span_m ÷ dwell_s)より信頼できる:
        #   span_m はグラフ経路長(直前ノード起点)なので、個体が既に進んでいたぶん
        #   過大評価になる。有界化するとその偏りが相対的に大きく出る。
        body_v.append(desired_speed(a.id) * (1.0 - float(b["blocked"])))
    final = {a.id: (round(float(a.x), 6), round(float(a.y), 6)) for a in sim.agents}
    zones_out = {zid: {"occupancy_mean": round(sum(v["occ"]) / len(v["occ"]), 3),
                       "density_mean": round(sum(v["dens"]) / len(v["dens"]), 5),
                       "waiting_mean": round(sum(v["wait"]) / len(v["wait"]), 2),
                       "sub_steps_mean": round(sum(v["sub"]) / len(v["sub"]), 1)}
                 for zid, v in sorted(per_zone.items())}
    return {
        "occ_mean_total": round(sum(z["occupancy_mean"] for z in zones_out.values()), 3),
        "density_mean_total": round(
            sum(z["density_mean"] for z in zones_out.values()) / max(1, len(zones_out)), 5),
        "waiting_mean_total": round(
            sum(z["waiting_mean"] for z in zones_out.values()), 2),
        "zone_occupancy_last": sc.get("zone_occupancy"),
        "zone_density_mean_last": (round(sc["zone_density_mean"], 4)
                                   if sc.get("zone_density_mean") is not None else None),
        "zone_dwell_mean_s": (round(sc["zone_dwell_mean_s"], 2)
                              if sc.get("zone_dwell_mean_s") is not None else None),
        "enter_total": sc.get("zone_gate_enter_total"),
        "exit_total": sc.get("zone_gate_exit_total"),
        "by_zone": zones_out,
        "body_density_mean": (round(sum(dens_body) / len(dens_body), 3)
                              if dens_body else None),
        "body_density_max": (round(max(dens_body), 3) if dens_body else None),
        "body_blocked_mean": (round(sum(blocked) / len(blocked), 3)
                              if blocked else None),
        "body_speed_mean": (round(sum(body_v) / len(body_v), 4) if body_v else None),
        "body_n": len(body_v),
        "v_eff_mean": (round(sum(speeds) / len(speeds), 4) if speeds else None),
        "v_eff_n": len(speeds),
        "min_gap_m": (round(cont["min_gap_m"], 4)
                      if cont.get("min_gap_m") is not None else None),
        "jump_max_m": (round(cont["jump_max_m"], 4)
                       if cont.get("jump_max_m") is not None else None),
        "handover_jump_max_m": (round(cont["handover_jump_max_m"], 3)
                                if cont.get("handover_jump_max_m") is not None else None),
        "gate_accel_p99": cont.get("gate_accel_p99"),
        "interior_accel_p99": cont.get("interior_accel_p99"),
        "gate_reversal_rate": (round(cont["gate_reversal_rate"], 4)
                               if cont.get("gate_reversal_rate") is not None else None),
        "interior_reversal_rate": (round(cont["interior_reversal_rate"], 4)
                                   if cont.get("interior_reversal_rate") is not None
                                   else None),
        "sep_iters_max": cont.get("sep_iters_max"),
        "sub_steps_total": cont.get("sub_steps_total"),
        "physics_s_per_step": round(phys_s / steps, 3),
        "wall_s_per_step": round(wall / steps, 3),
        "_final_xy": final,
        "_exit_step": {f"{aid}|{zid}": st for (aid, zid), st in exit_step.items()},
    }


def sweep_ownership(sizes, specs, steps):
    rows = []
    for n in sizes:
        base_cost = None
        base_enter = None
        base_xy: dict = {}
        base_exit: dict = {}
        for spec in specs:
            label, _dots, x = spec
            sim, phys_s, wall, per_zone = _own_run(n, steps, spec)
            m = _own_metrics(sim, phys_s, wall, per_zone, steps)
            m.update(n=n, steps=steps, config=label, ownership_max_dist_m=x)
            xy = m.pop("_final_xy")
            ex = m.pop("_exit_step")
            if base_cost is None:                     # 先頭の水準が基準(通常 off = 現行)
                base_cost, base_xy, base_exit = m["physics_s_per_step"], xy, ex
                base_enter = m["enter_total"]
            m["physics_speedup_x"] = (round(base_cost / m["physics_s_per_step"], 3)
                                      if m["physics_s_per_step"] > 0 else None)
            # ---- 捕捉率(基準の入場件数に対する比)----
            m["capture_frac"] = (round(m["enter_total"] / base_enter, 4)
                                 if base_enter else None)
            # ---- 軌跡差(基準 X との正直な突き合わせ)----
            moved = [math.hypot(xy[k][0] - base_xy[k][0], xy[k][1] - base_xy[k][1])
                     for k in base_xy if k in xy]
            diff = [d for d in moved if d > 1e-9]
            m["traj_moved_agents"] = len(diff)
            m["traj_moved_frac"] = (round(len(diff) / len(moved), 4) if moved else None)
            m["traj_disp_mean_m"] = (round(sum(diff) / len(diff), 2) if diff else 0.0)
            m["traj_disp_max_m"] = (round(max(diff), 2) if diff else 0.0)
            keys = set(base_exit) | set(ex)
            same = sum(1 for k in keys if base_exit.get(k) == ex.get(k))
            m["exit_step_same_frac"] = round(same / len(keys), 4) if keys else None
            m["exit_pairs_base_only"] = len(set(base_exit) - set(ex))
            m["exit_pairs_new_only"] = len(set(ex) - set(base_exit))
            rows.append(m)
            print(f"  n={n:5d} {label:10s} capture={m['capture_frac']} "
                  f"enter={m['enter_total']:5} occ_mean={m['occ_mean_total']} "
                  f"dens_mean={m['density_mean_total']} "
                  f"body_dens={m['body_density_mean']} dwell={m['zone_dwell_mean_s']} "
                  f"v_body={m['body_speed_mean']} "
                  f"gap={m['min_gap_m']} jump={m['jump_max_m']} "
                  f"p99g={m['gate_accel_p99']} wait={m['waiting_mean_total']} "
                  f"phys={m['physics_s_per_step']}s x{m['physics_speedup_x']} "
                  f"moved={m['traj_moved_frac']}", flush=True)
    return rows


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
    ap.add_argument("--ownership", default="",
                    help="第155 C1 掃引(例 0,50,100,200,route)。0=無制限(現行)/ "
                         "数値=euclid のしきい [m] / route=route_arrival。指定すると"
                         "合成シナリオではなく **conf/zones_shibuya.yaml の実 3 ゾーンを"
                         "本体で**回す")
    ap.add_argument("--own-sizes", default="600,2000",
                    help="--ownership のときの人数(既定 600,2000)")
    ap.add_argument("--own-steps", type=int, default=12,
                    help="--ownership のときの step 数(既定 12。mock・24 step 以内)")
    args = ap.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    t_total = 6.0 if args.quick else T_TOTAL
    seps = tuple(int(s) for s in args.sep.split(",") if s.strip())
    specs = tuple(own_spec(s) for s in args.ownership.split(",") if s.strip())
    if specs:
        own_sizes = tuple(int(s) for s in args.own_sizes.split(",") if s.strip())
        print(f"[bench] ownership sweep {[s[0] for s in specs]} sizes={own_sizes} "
              f"steps={args.own_steps} profile={os.path.relpath(OWN_PROFILE, _REPO)}",
              flush=True)
        rows = sweep_ownership(own_sizes, specs, args.own_steps)
        os.makedirs(args.out, exist_ok=True)
        blob = {"meta": {"mode": "ownership", "levels": [s[0] for s in specs],
                         "sizes": list(own_sizes), "steps": args.own_steps,
                         "python": platform.python_version(),
                         "numpy": np.__version__},
                "rows": rows}
        path = os.path.join(args.out, "levers_ownership.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print(f"[bench] wrote {path}")
        return
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
