"""物理痩身 第二段B/C の検証ハーネス — 基本図 / ボトルネック / 性能 / 破綻統計。

    python scripts/bench_crowd_cognition.py --out <dir> [--quick]

何を測るか(発注の検収 1・2・4 に対応)
--------------------------------------
  (1) **基本図**(密度-速度)と**ボトルネック流量**を OFF / B / C / B+C の 4 構成で測り、
      Jülich 実測(`reference/physics_bench/data`)からの乖離を表にする。
      判定は P4-1 と同じ物差し: ±20% 帯(領域 I–II)/ 単調性 / 10-90 パーセンタイル包絡線 /
      specific flow J/w の受入帯 [1.5, 2.3]。
  (2) **性能**: 500/2000/8000 体の 1 体あたり時間(SFM forces / far 項 / ORCA step)と、
      密度 1–10 /m² での単価の伸び。
  (4) **破綻チェック**: 壁貫通(最小クリアランス)・重なり率・逆走率を OFF 比で出す。

設計上の約束
------------
  - `reference/physics_bench` は **読み取りのみ**(参照曲線・指標関数を import して使う。
    1 バイトも書き換えず、生成物も置かない。出力の既定は `experiments/`= .gitignore 済み)。
    シナリオ側はここに自前で持つ: 本ハーネスが回すのは
    **src/society の本体エンジン**(`physics._CalibratedCrowd` / `orca_core.OrcaCrowd`)で、
    ベンチ側の `ExtendedCrowd` ではない = 「本体で何が起きるか」を測る。
  - 周期通路のゴーストは「積分後に位置と速度を戻す」= ベンチ `WallCrowd(frozen=…)` と同じ
    意味(SFM の力は相手の速度に依らないので、ゴーストの速度は他個体へ影響しない)。
  - 乱数は初期配置だけ(seed 固定)。積分ループ内はゼロ = 同じコマンドで同じ JSON が出る。
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

from reference.physics_bench import juelich, metrics          # noqa: E402
from society import physics as P                              # noqa: E402
from society.world import indoor_flow, orca_core, sfm_core    # noqa: E402

# ── P4-1 と同じ幾何・測定窓(較正資産と同じ物差しで測るため)────────────────────
FD_LENGTH, FD_WIDTH = 20.0, 3.0
FD_AREA = FD_LENGTH * FD_WIDTH
FD_DENSITIES = (0.5, 1.0, 1.5, 2.0, 3.0)
FD_T_TOTAL = 60.0
FD_WARM_FRAC = 0.667
DT = 0.05
GOAL_AHEAD = 50.0
BOTTLENECK_WIDTHS = (0.8, 1.0, 1.2, 1.6, 2.4)
BN_N, BN_T = 80, 45.0
BN_REF_JS = 1.9              # Seyfried et al. 2009 / Rupprecht et al. 2011
BAND_TOL = 0.20

# finals の far_field(P4-2 較正値)。第二段C はこの項を置換する。
FAR_KW = dict(far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5, far_taper_m=1.0)

# 4 構成。B = 認知的近傍 / C = 遠方場の密度場化。
CONFIGS = {
    "OFF": {},
    "B": {"cognitive": True},
    "C": {"far_mode": "field"},
    "B+C": {"cognitive": True, "far_mode": "field"},
}


# ═════════════════════════════════════════════════════════════════════════════
# エンジン(本体 src を回すための最小ラッパ)
# ═════════════════════════════════════════════════════════════════════════════
class HarnessCrowd(P._CalibratedCrowd):
    """`_CalibratedCrowd` + 周期像ゴースト(frozen)。**力の経路は本体そのまま**。

    ゴーストは「斥力源だが動かない」。`WallCrowd(frozen=…)` は積分前に速度を 0 にするが、
    SFM の力は相手の速度に依存しない(駆動項は自分の速度のみ・斥力は位置のみ)ので、
    積分後に位置と速度を戻すのと**同じ結果**になる。
    """

    def __init__(self, *a, frozen=None, **kw):
        super().__init__(*a, **kw)
        n = self.pos.shape[0]
        self.frozen = (np.zeros(n, dtype=bool) if frozen is None
                       else np.asarray(frozen, dtype=bool).reshape(-1).copy())

    def step(self, dt=0.05):
        keep_p = self.pos[self.frozen].copy()
        super().step(dt)
        self.pos[self.frozen] = keep_p
        self.vel[self.frozen] = 0.0
        return self.pos


def agent_params(n):
    """(v0, radius)。ベンチ `engines.agent_params` と同じ安定ハッシュ(乱数ゼロ)。"""
    v0 = np.array([indoor_flow.desired_speed(i) for i in range(n)], dtype=np.float64)
    rad = np.array([indoor_flow.body_radius(i) for i in range(n)], dtype=np.float64)
    return v0, rad


def _init_positions(rng, n, x_lo, x_hi, y_lo, y_hi, radius, tries=400):
    """重なりなしの初期配置(棄却サンプリング。ベンチ scenarios._init_positions と同型)。"""
    pos = np.zeros((n, 2))
    for i in range(n):
        c = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
        for _ in range(tries):
            c = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
            if i == 0:
                break
            d = np.linalg.norm(pos[:i] - c, axis=1)
            if (d > radius[:i] + radius[i] + 0.05).all():
                break
        pos[i] = c
    return pos


def _corridor_walls(length, width, pad=10.0):
    return [((-pad, 0.0), (length + pad, 0.0)),
            ((-pad, width), (length + pad, width))]


def _kw(config, far_on=True, cog=None, field=None):
    """4 構成のどれかの名前 → エンジンのキーワード引数。"""
    out = dict(FAR_KW) if far_on else {}
    spec = dict(CONFIGS[config])
    if spec.pop("cognitive", False):
        out.update(cognitive=True, **(cog or {}))
    if spec.pop("far_mode", None) == "field" and far_on:
        out.update(far_mode="field", **(field or {}))
    assert not spec, spec
    return out


# ═════════════════════════════════════════════════════════════════════════════
# A. 周期通路の基本図(密度-速度)
# ═════════════════════════════════════════════════════════════════════════════
def run_fd_periodic(n_agents, engine_kw, length=FD_LENGTH, width=FD_WIDTH, dt=DT,
                    t_total=FD_T_TOTAL, seed=20260801, ghost_range=None):
    """周期通路。戻り値 (pos_t, vel_t, wrap_t, mean_v0)。"""
    v0, radius = agent_params(n_agents)
    rng = np.random.default_rng(seed)
    pos = _init_positions(rng, n_agents, 0.2, length - 0.2, 0.4, width - 0.4, radius)
    vel = np.zeros((n_agents, 2))
    vel[:, 0] = v0
    # ゴースト帯は far のカットオフ(体表間 cutoff + 体径)を覆う幅にする
    g_range = (2.5 * FAR_KW["far_b2"] + 1.0) if ghost_range is None else float(ghost_range)
    n_steps = int(round(t_total / dt))
    big_p = np.zeros((n_steps + 1, n_agents, 2))
    big_v = np.zeros((n_steps + 1, n_agents, 2))
    wrap = np.zeros((n_steps + 1, n_agents), dtype=bool)
    cur_p, cur_v = pos.copy(), vel.copy()
    big_p[0], big_v[0] = cur_p, cur_v
    walls = _corridor_walls(length, width)
    for t in range(1, n_steps + 1):
        left = cur_p[:, 0] < g_range
        right = cur_p[:, 0] > length - g_range
        gp = np.concatenate([cur_p[left] + [length, 0.0],
                             cur_p[right] - [length, 0.0]], axis=0)
        gr = np.concatenate([radius[left], radius[right]], axis=0)
        g = gp.shape[0]
        allp = np.concatenate([cur_p, gp], axis=0)
        allv = np.concatenate([cur_v, np.zeros((g, 2))], axis=0)
        allr = np.concatenate([radius, gr], axis=0)
        # 目標は常に真正面 GOAL_AHEAD m 先(希望方向 = +x を厳密に保つ)。
        # ゴーストの目標は自分自身 = 希望方向 0(斥力源としてだけ効く)。
        goal = np.concatenate([cur_p + np.array([GOAL_AHEAD, 0.0]), gp], axis=0)
        allv0 = np.concatenate([v0, np.ones(g)], axis=0)
        frozen = np.concatenate([np.zeros(n_agents, bool), np.ones(g, bool)])
        crowd = HarnessCrowd(pos=allp, vel=allv, goal=goal, v0=allv0, radius=allr,
                             walls=walls, neighbor_cap=indoor_flow.NEIGHBOR_CAP,
                             frozen=frozen, **engine_kw)
        crowd.step(dt)
        cur_p = crowd.pos[:n_agents].copy()
        cur_v = crowd.vel[:n_agents].copy()
        w = cur_p[:, 0] >= length
        if w.any():
            cur_p[w, 0] -= length
        wrap[t] = w
        big_p[t], big_v[t] = cur_p, cur_v
    return big_p, big_v, wrap, float(v0.mean())


def fd_sweep(engine_kw, densities=FD_DENSITIES, t_total=FD_T_TOTAL,
             warm_frac=FD_WARM_FRAC, extras=False):
    rows = {"rho": [], "v": [], "speed": [], "v0": []}
    if extras:
        rows["overlap_pair_rate"] = []
        rows["min_gap_m"] = []
        rows["wall_clearance_m"] = []
        rows["reversal_rate"] = []
        rows["stall_rate"] = []
    for rho in densities:
        n = int(round(rho * FD_AREA))
        pos_t, vel_t, wrap_t, mv0 = run_fd_periodic(n, engine_kw, t_total=t_total)
        t0 = int(pos_t.shape[0] * warm_frac)
        rows["rho"].append(float(rho))
        rows["v"].append(float(vel_t[t0:, :, 0].mean()))
        rows["speed"].append(float(np.linalg.norm(vel_t[t0:], axis=2).mean()))
        rows["v0"].append(mv0)
        if extras:
            _, radius = agent_params(n)
            ov = metrics.overlap_stats(pos_t[t0:], radius, sample_every=10)
            cont = metrics.continuity_stats(
                pos_t[t0:], vel_t[t0:], wrap_t[t0:], DT,
                lambda p: np.zeros(p.shape[0], dtype=bool))
            rows["overlap_pair_rate"].append(ov["overlap_pair_rate"])
            rows["min_gap_m"].append(ov["min_gap_m"])
            rows["wall_clearance_m"].append(metrics.wall_clearance_min(
                pos_t[t0:], radius, _corridor_walls(FD_LENGTH, FD_WIDTH),
                sample_every=10))
            rows["reversal_rate"].append(cont["reversal_rate_per_agent_s"])
            rows["stall_rate"].append(cont["interior"]["stall_rate"])
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# B. ボトルネック(specific flow J/w)
# ═════════════════════════════════════════════════════════════════════════════
def _bottleneck_walls(room_len, room_width, w, exit_len, pad=1.0):
    lo, hi = (room_width - w) / 2.0, (room_width + w) / 2.0
    return [((0.0, 0.0), (0.0, room_width)),
            ((0.0, 0.0), (room_len, 0.0)),
            ((0.0, room_width), (room_len, room_width)),
            ((room_len, 0.0), (room_len, lo)),
            ((room_len, hi), (room_len, room_width)),
            ((room_len, lo), (room_len + exit_len + pad, lo)),
            ((room_len, hi), (room_len + exit_len + pad, hi))]


class _Trace:
    def __init__(self, pos, active, dt):
        self.pos = pos
        self.active = active
        self.dt = dt


def run_bottleneck(engine_kw, w=1.2, n_agents=BN_N, room_len=8.0, room_width=5.0,
                   exit_len=2.0, dt=DT, t_total=BN_T, seed=20260805):
    v0, radius = agent_params(n_agents)
    rng = np.random.default_rng(seed)
    pos = _init_positions(rng, n_agents, 0.45, room_len - 0.45,
                          0.45, room_width - 0.45, radius)
    vel = np.zeros((n_agents, 2))
    y_c = room_width / 2.0
    walls = _bottleneck_walls(room_len, room_width, w, exit_len)
    n_steps = int(round(t_total / dt))
    big_p = np.zeros((n_steps + 1, n_agents, 2))
    act = np.ones((n_steps + 1, n_agents), dtype=bool)
    done = np.zeros(n_agents, dtype=bool)
    x_out = room_len + exit_len
    cur_p, cur_v = pos.copy(), vel.copy()
    big_p[0] = cur_p
    park = np.stack([np.arange(n_agents) * 20.0,
                     np.full(n_agents, -1000.0)], axis=1)
    for t in range(1, n_steps + 1):
        past = cur_p[:, 0] >= room_len
        goal = np.stack([np.where(past, cur_p[:, 0] + GOAL_AHEAD, room_len + 1.0),
                         np.where(past, cur_p[:, 1], y_c)], axis=1)
        crowd = HarnessCrowd(pos=cur_p, vel=cur_v, goal=goal, v0=v0, radius=radius,
                             walls=walls, neighbor_cap=indoor_flow.NEIGHBOR_CAP,
                             **engine_kw)
        crowd.step(dt)
        cur_p, cur_v = crowd.pos.copy(), crowd.vel.copy()
        done |= cur_p[:, 0] > x_out
        if done.any():
            cur_p[done] = park[done]
            cur_v[done] = 0.0
        act[t] = ~done
        big_p[t] = cur_p
    return _Trace(big_p, act, dt), room_len


def bottleneck_sweep(engine_kw, widths=BOTTLENECK_WIDTHS):
    rows = []
    for w in widths:
        tr, line_x = run_bottleneck(engine_kw, w=w)
        r = metrics.specific_flow(tr, line_x, w)
        r["w"] = w
        r["n_evacuated"] = int((~tr.active[-1]).sum())
        rows.append(r)
    slope, icept, r2 = metrics.linear_fit_r2([r["w"] for r in rows],
                                             [r["J"] for r in rows])
    js = [r["J_specific"] for r in rows if np.isfinite(r["J_specific"])]
    return {"rows": rows, "J_vs_w": {"slope": slope, "intercept": icept, "r2": r2},
            "J_specific_mean": float(np.mean(js)) if js else float("nan"),
            "in_band_1.5_2.3": [bool(1.5 <= x <= 2.3) for x in js]}


# ═════════════════════════════════════════════════════════════════════════════
# C. Jülich 参照曲線(P4-1 の `calibrate.Reference` と同じ作り方)
# ═════════════════════════════════════════════════════════════════════════════
class Reference:
    def __init__(self, data_dir, v0_sim):
        self.binned = juelich.load_binned(os.path.join(data_dir,
                                                       "juelich_fd_binned.csv"))
        self.vf, self.gamma, self.rho_max = metrics.fit_kladek(
            self.binned["rho_per_m2"], self.binned["v_mean_mps"],
            weights=self.binned["n"])
        self.v0_sim = float(v0_sim)

    def curve(self, rho, v0=None):
        """自由速度をシムの平均 v0 に差し替えた参照速度(P4-1 と同じ再スケール)。"""
        vfs = (np.full(np.shape(rho), self.v0_sim) if v0 is None
               else np.asarray(v0, dtype=np.float64))
        return np.array([metrics.weidmann_speed(np.array([r]), v_f=float(f),
                                                gamma=self.gamma,
                                                rho_max=self.rho_max)[0]
                         for r, f in zip(np.atleast_1d(rho), np.atleast_1d(vfs))])

    def envelope(self, rho, v0=None):
        lo, hi = juelich.reference_envelope(self.binned, rho)
        vfs = (np.full(np.shape(rho), self.v0_sim) if v0 is None
               else np.asarray(v0, dtype=np.float64))
        s = vfs / self.vf
        return lo * s, hi * s


def judge(sweep, ref):
    rho = np.array(sweep["rho"], dtype=np.float64)
    v = np.array(sweep["v"], dtype=np.float64)
    vref = ref.curve(rho, v0=sweep["v0"])
    band = metrics.band_check(v, vref, tol=BAND_TOL)
    sel = (rho >= 0.5) & (rho <= 2.0)                # 領域 I–II(P4-1 §2.3)
    band12 = metrics.band_check(v[sel], vref[sel], tol=BAND_TOL)
    lo, hi = ref.envelope(rho, v0=sweep["v0"])
    env = metrics.envelope_coverage(v, lo, hi)
    mono = metrics.monotonicity(rho, v)
    rel = np.abs(np.array([x for x in band["rel_err"] if x is not None]))
    return {"v": [float(x) for x in v],
            "v_ref": [float(x) for x in vref],
            "rel_err": band["rel_err"],
            "rmse_rel": float(math.sqrt(float((rel ** 2).mean()))) if rel.size else None,
            "band_pass_all": band["pass_rate"],
            "band_pass_I_II": band12["pass_rate"],
            "max_abs_rel_err": band.get("max_abs_rel_err"),
            "monotonic_violations": mono["violations"],
            "monotonic_max_increase": mono["max_increase"],
            "envelope_inside_rate": env["inside_rate"]}


# ═════════════════════════════════════════════════════════════════════════════
# D. 性能
# ═════════════════════════════════════════════════════════════════════════════
def _time(fn, reps=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def _synthetic(n, dens, seed=99):
    rng = np.random.default_rng(seed)
    s = math.sqrt(n / dens)
    pos = rng.uniform(0.0, s, size=(n, 2))
    radius = rng.uniform(sfm_core.RADIUS_MIN, sfm_core.RADIUS_MAX, size=n)
    return dict(pos=pos, vel=np.zeros((n, 2)),
                goal=pos + np.array([GOAL_AHEAD, 0.0]),
                v0=np.full(n, 1.3), radius=radius, neighbor_cap=12)


def perf_section(quick=False):
    out = {"sfm_forces_us_per_agent": {}, "far_term_us_per_agent": {},
           "orca_step_ms": {}, "density_scan_us_per_agent": {},
           "cognitive_neighbors_mean": {}}
    ns = (500, 2000) if quick else (500, 2000, 8000)
    for n in ns:
        base = _synthetic(n, 2.0)
        row = {}
        for name in ("OFF", "B"):
            c = sfm_core.Crowd(**base, **({} if name == "OFF" else {"cognitive": True}))
            row[name] = _time(c.forces) * 1e6 / n
        out["sfm_forces_us_per_agent"][n] = row
        far = {}
        for name, kw in (("pair(OFF)", {}), ("field(C)", {"far_mode": "field",
                                                          "field_update_every": 1})):
            c = P._CalibratedCrowd(**_synthetic(n, 3.0), **FAR_KW, **kw)
            fn = c._far_forces_field if kw else c._far_forces
            far[name] = _time(fn, reps=5) * 1e6 / n
        out["far_term_us_per_agent"][n] = far
        if n <= 2000:
            orow = {}
            for name in ("OFF", "B"):
                base_o = _synthetic(n, 2.0)
                base_o.pop("neighbor_cap")
                c = orca_core.OrcaCrowd(
                    **base_o, neighbor_cap=12,
                    **({} if name == "OFF" else {"cognitive": True}))
                orow[name] = _time(lambda c=c: c.step(DT), reps=1) * 1e3
            out["orca_step_ms"][n] = orow
    n = 3000
    for dens in (1.0, 2.0, 4.0, 6.0, 10.0):
        base = _synthetic(n, dens, seed=7)
        row = {}
        for name in ("OFF", "B"):
            c = sfm_core.Crowd(**base, **({} if name == "OFF" else {"cognitive": True}))
            row[name] = _time(c.forces) * 1e6 / n
        cp = P._CalibratedCrowd(**base, **FAR_KW)
        cf = P._CalibratedCrowd(**base, **FAR_KW, far_mode="field",
                                field_update_every=1)
        row["far_pair"] = _time(cp._far_forces, reps=5) * 1e6 / n
        row["far_field"] = _time(cf._far_forces_field, reps=5) * 1e6 / n
        out["density_scan_us_per_agent"][dens] = row
        e = np.zeros_like(base["pos"])
        e[:, 0] = 1.0
        _o, valid = sfm_core.visual_neighbors(
            base["pos"], base["pos"], e, 12,
            2.0 * float(base["radius"].max()) + sfm_core.CUTOFF_M)
        ii, _jj = sfm_core.neighbor_pairs(
            base["pos"], 2.0 * float(base["radius"].max()) + sfm_core.CUTOFF_M)
        out["cognitive_neighbors_mean"][dens] = {
            "B_selected_mean": float(valid.sum(axis=1).mean()),
            "B_selected_max": int(valid.sum(axis=1).max()),
            "OFF_candidates_mean": float(np.bincount(ii, minlength=n).mean())}
    return out


# ═════════════════════════════════════════════════════════════════════════════
def _fmt_table(res):
    lines = []
    lines.append("=== 基本図(周期通路 20x3 m・dt=0.05・t=60s・窓 t>=40s)===")
    hdr = "  rho  " + "".join(f"{c:>12}" for c in CONFIGS) + "     ref"
    lines.append(hdr)
    for i, rho in enumerate(FD_DENSITIES):
        row = f" {rho:4.1f}  "
        for c in CONFIGS:
            row += f"{res['fd'][c]['v'][i]:12.3f}"
        row += f"{res['fd']['OFF']['judge']['v_ref'][i]:8.3f}"
        lines.append(row)
    lines.append("")
    lines.append("  判定" + "".join(f"{c:>12}" for c in CONFIGS))
    for key, label in (("band_pass_I_II", "±20%帯 I-II"),
                       ("band_pass_all", "±20%帯 全点"),
                       ("rmse_rel", "相対RMSE"),
                       ("max_abs_rel_err", "最大|相対誤差|"),
                       ("monotonic_violations", "単調性違反"),
                       ("envelope_inside_rate", "包絡線内率")):
        row = f" {label:<14}"
        for c in CONFIGS:
            v = res["fd"][c]["judge"][key]
            row += f"{v:12.3f}" if isinstance(v, float) else f"{v:>12}"
        lines.append(row)
    lines.append("")
    lines.append("=== ボトルネック specific flow J/w [1/(m s)](参照 1.9・受入帯 1.5-2.3)===")
    lines.append("   w  " + "".join(f"{c:>12}" for c in CONFIGS))
    for i, w in enumerate(BOTTLENECK_WIDTHS):
        row = f" {w:4.1f}"
        for c in CONFIGS:
            row += f"{res['bottleneck'][c]['rows'][i]['J_specific']:12.3f}"
        lines.append(row)
    row = " mean"
    for c in CONFIGS:
        row += f"{res['bottleneck'][c]['J_specific_mean']:12.3f}"
    lines.append(row)
    row = " J(w)R2"
    for c in CONFIGS:
        row += f"{res['bottleneck'][c]['J_vs_w']['r2']:12.3f}"
    lines.append(row)
    lines.append("")
    lines.append("=== 破綻統計(基本図ランから。ρ ごと)===")
    for key, label in (("overlap_pair_rate", "重なりペア率"),
                       ("min_gap_m", "最小すき間[m]"),
                       ("wall_clearance_m", "壁クリアランス[m](負=貫通)"),
                       ("reversal_rate", "逆走率[回/体s]"),
                       ("stall_rate", "立往生率")):
        lines.append(f" -- {label}")
        lines.append("  rho  " + "".join(f"{c:>12}" for c in CONFIGS))
        for i, rho in enumerate(FD_DENSITIES):
            row = f" {rho:4.1f}  "
            for c in CONFIGS:
                row += f"{res['fd'][c][key][i]:12.4f}"
            lines.append(row)
    lines.append("")
    lines.append("=== 性能 ===")
    lines.append(" SFM forces() [us/agent] (rho=2)")
    for n, r in res["perf"]["sfm_forces_us_per_agent"].items():
        lines.append(f"  N={n:>5}  OFF={r['OFF']:7.2f}  B={r['B']:7.2f}"
                     f"  ({r['B'] / r['OFF']:.2f}x)")
    lines.append(" far 項 [us/agent] (rho=3)")
    for n, r in res["perf"]["far_term_us_per_agent"].items():
        lines.append(f"  N={n:>5}  pair={r['pair(OFF)']:7.2f}"
                     f"  field={r['field(C)']:7.3f}"
                     f"  ({r['pair(OFF)'] / r['field(C)']:.0f}x)")
    lines.append(" ORCA step() [ms] (rho=2)")
    for n, r in res["perf"]["orca_step_ms"].items():
        lines.append(f"  N={n:>5}  OFF={r['OFF']:8.1f}  B={r['B']:8.1f}")
    lines.append(" 密度掃引 [us/agent] (N=3000)")
    lines.append("   rho   SFM_OFF     SFM_B  far_pair far_field  B近傍数  OFF候補数")
    for d, r in res["perf"]["density_scan_us_per_agent"].items():
        nb = res["perf"]["cognitive_neighbors_mean"][d]
        lines.append(f"  {d:5.1f} {r['OFF']:9.2f} {r['B']:9.2f} {r['far_pair']:9.2f}"
                     f" {r['far_field']:9.3f} {nb['B_selected_mean']:8.2f}"
                     f" {nb['OFF_candidates_mean']:10.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # ★出力は experiments/(.gitignore 済み)。`reference/physics_bench/` は**読み取り専用**で、
    #   そこへ生成物を落とすと未追跡ファイルが増えて事故のもとになる(ステージ禁止の掟)。
    ap.add_argument("--out", default=os.path.join(_REPO, "experiments",
                                                  "physics_cognition"))
    ap.add_argument("--data", default=os.path.join(_REPO, "reference",
                                                   "physics_bench", "data"))
    ap.add_argument("--quick", action="store_true",
                    help="密度点と幅を間引く(配線の確認用)")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--update-every", type=int, default=None,
                    help="C の格子更新周期(既定 = conf 既定値)。1 と 10 の比較用")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dens = (0.5, 1.5, 3.0) if args.quick else FD_DENSITIES
    widths = (1.0, 2.4) if args.quick else BOTTLENECK_WIDTHS
    t_total = 24.0 if args.quick else FD_T_TOTAL
    field_kw = ({} if args.update_every is None
                else {"field_update_every": int(args.update_every)})

    t_start = time.perf_counter()
    v0_sim = float(agent_params(180)[0].mean())
    ref = Reference(args.data, v0_sim)
    res = {"generated_by": "scripts/bench_crowd_cognition.py",
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "platform": platform.platform()},
           "config": {"densities": list(dens), "widths": list(widths),
                      "t_total_s": t_total, "warm_frac": FD_WARM_FRAC, "dt": DT,
                      "far_field": FAR_KW, "field_kw": field_kw,
                      "cognitive_defaults": {
                          "neighbors": sfm_core.COG_NEIGHBORS_DEFAULT,
                          "sectors": sfm_core.COG_SECTORS_DEFAULT,
                          "fov_deg": sfm_core.COG_FOV_DEG_DEFAULT}},
           "reference": {"source": "Julich Pedestrian Dynamics Data Archive (CC BY 4.0)",
                         "kladek_fit": {"v_f": ref.vf, "gamma": ref.gamma,
                                        "rho_max": ref.rho_max},
                         "v0_sim_nominal": v0_sim},
           "fd": {}, "bottleneck": {}}

    for name in CONFIGS:
        print(f"[FD ] {name} ...", flush=True)
        t0 = time.perf_counter()
        sw = fd_sweep(_kw(name, field=field_kw), densities=dens, t_total=t_total,
                      extras=True)
        sw["judge"] = judge(sw, ref)
        sw["wall_s"] = time.perf_counter() - t0
        res["fd"][name] = sw
        print(f"      v={[round(x, 3) for x in sw['v']]} "
              f"band(I-II)={sw['judge']['band_pass_I_II']:.2f} "
              f"({sw['wall_s']:.0f}s)", flush=True)

    for name in CONFIGS:
        print(f"[BN ] {name} ...", flush=True)
        t0 = time.perf_counter()
        bn = bottleneck_sweep(_kw(name, field=field_kw), widths=widths)
        bn["wall_s"] = time.perf_counter() - t0
        res["bottleneck"][name] = bn
        print(f"      Js={[round(r['J_specific'], 3) for r in bn['rows']]} "
              f"({bn['wall_s']:.0f}s)", flush=True)

    if not args.skip_perf:
        print("[PERF] ...", flush=True)
        res["perf"] = perf_section(quick=args.quick)
    res["wall_s_total"] = time.perf_counter() - t_start

    path = os.path.join(args.out, "cognition_bench.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    if "perf" in res:
        globals()["FD_DENSITIES"] = tuple(dens)
        globals()["BOTTLENECK_WIDTHS"] = tuple(widths)
        text = _fmt_table(res)
        print()
        print(text)
        with open(os.path.join(args.out, "cognition_bench.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(f"[done] {path}  ({res['wall_s_total'] / 60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
