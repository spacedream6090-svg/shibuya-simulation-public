"""P4-3 較正ハーネス — Tordeux 型 V(s) + 壁斥力(**src/ は 1 バイトも触らない**)。

    python -m reference.physics_bench.calibrate_p43 --out reference/physics_bench/out

P4-1(`calibrate.py`)との関係
------------------------------
`calibrate.py` と `out/calib_results.json` は **P4-1 の成果物としてそのまま残す**
(再実行して上書きしない)。本スクリプトはその確定事項を前提に受け取る:

  * λ = 0.5 据え置き(λ 先行実験が文献値 0.06–0.12 の仮説を棄却した)
  * far_field 推奨値 (A2, B2) = (0.119, 1.890)・cutoff_factor 2.5・taper 1.0 m・dt 0.05
  * 長距離項だけでは不合格(単調性 2 違反・ρ=3.0 で +2.9 倍・J/w が実測の 1/2.2〜2.8)
  * 壁斥力 WALL_A=2000 / WALL_B=0.08 が片側 +0.15 m の過剰クリアランスを要求し、
    w ≤ 0.9 m の開口は**単独歩行者でも通れない**

本スクリプトが探索するもの(4 変数)
------------------------------------
  T        … V(s) の時間ギャップ [s]
  ℓ        … V(s) の体長パラメータ [m]
  WALL_B   … 壁斥力の特性距離 [m](= 有効クリアランスを決める唯一のレバー)
  far_field on/off … 長距離項を併用するか(離散 2 水準の外側ループ)

★受入は「大域密度」ではなく **局所測定密度**(RiMEA Test 4 の 2×2 m 区画)で作る。
  P4-1 の実測で ρ_global=0.5 に対し測定密度 0.73、ρ_global=2.0 に対し 0.93 と
  食い違い、大域密度で作った FD は「どの密度の話をしているか」が曖昧になるため
  (README §6-5)。過渡は 20〜35 s なので **先頭 35 s を破棄**する
  (RiMEA が許す 10 s では本モデルの過渡が抜けきらない = README §6-6)。

【決定論】乱数はシナリオの初期配置のみ(seed 固定)。Nelder-Mead は自前実装で
  乱数ゼロ・安定ソート。同じコマンドを 2 回走らせれば同じ JSON が出る。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    __package__ = "reference.physics_bench"

from . import calibrate as C                          # noqa: E402
from . import engines, metrics, scenarios             # noqa: E402

# ── P4-1 で確定した前提(ここは探索しない)──────────────────────────────────
LAMBDA_FIXED = 0.5              # 本体既定。λ 先行実験が文献値仮説を棄却(P4-1 §3.2(1))
FAR_A2 = 0.119                  # P4-1 推奨 A2 [m/s²]
FAR_B2 = 1.890                  # P4-1 推奨 B2 [m]
DT_MAIN = 0.05                  # P4-1 推奨 dt
WALL_B_BASE = 0.08              # 現行 src 既定(Helbing2000 escape-panic 値)

# ── 測定プロトコル(局所測定密度)────────────────────────────────────────────
FD_LENGTH, FD_WIDTH = C.FD_LENGTH, C.FD_WIDTH        # 20 m × 3 m 周期通路
FD_AREA = C.FD_AREA
DISCARD_S = 35.0                # 過渡の実測(20〜35 s)を完全に外す
SEARCH_DENSITIES = (0.5, 1.0, 1.5, 2.0)
FINAL_DENSITIES = (0.5, 1.0, 1.5, 2.0, 3.0)
SEARCH_T_TOTAL = 50.0
FINAL_T_TOTAL = 60.0
WEIGHTS = C.CAL_WEIGHTS
MONO_PENALTY = C.MONO_PENALTY
BN_PENALTY = C.BN_PENALTY
BN_REF_JS = C.BN_REF_JS
BAND_TOL = C.BAND_TOL

# ── 壁較正の格子 ─────────────────────────────────────────────────────────────
WALL_B_GRID = (0.08, 0.06, 0.05, 0.04, 0.03, 0.02)
SINGLE_WIDTHS = (0.7, 0.8, 0.9, 1.0, 1.2, 1.6)
BN_ROOM = dict(room_len=8.0, room_width=5.0, exit_len=2.0)
BN_INNER = dict(w=1.2, n_agents=80, t_total=30.0, **BN_ROOM)
BN_SWEEP_N, BN_SWEEP_T = 80, 45.0

# ── V(s) の粗グリッド ────────────────────────────────────────────────────────
#   原論文(単列歩行)の使用値は T=1 s, ℓ=0.3 m。本リポジトリは体径 0.5–0.7 m なので
#   ℓ はそれより大きい側を厚めに取り、T は「間隔 1.4 m で自由速度に達する」あたりから
#   両側へ振る。
GRID_T = (0.15, 0.3, 0.55, 0.9)
GRID_L = (0.2, 0.4, 0.6)


# ═════════════════════════════════════════════════════════════════════════════
# エンジン引数の組み立て
# ═════════════════════════════════════════════════════════════════════════════
def make_kw(*, v_of_s=False, vos_T=1.0, vos_l=0.3, far=True, wall_b=WALL_B_BASE):
    """engine_kw(ExtendedSfmEngine 用)。far=False で長距離項を切る。"""
    sk = {"lambda_aniso": LAMBDA_FIXED, "lambda_far": LAMBDA_FIXED,
          "cutoff_mode": "per_term", "far_taper": engines.FAR_TAPER_DEFAULT,
          "wall_b": float(wall_b),
          "v_of_s": bool(v_of_s), "vos_T": float(vos_T), "vos_l": float(vos_l)}
    sk["a2"] = float(FAR_A2) if far else 0.0
    sk["b2"] = float(FAR_B2)
    return {"solver_kw": sk}


def _ghost(far):
    return C.far_ghost_range(FAR_B2) if far else None


# ═════════════════════════════════════════════════════════════════════════════
# 基本図(1 ランから **大域密度** と **局所測定密度** の両方を採る)
# ═════════════════════════════════════════════════════════════════════════════
def measurement_rects():
    """測定区画 3 枚 = **流れ方向 2 m × 通路全幅**(主 1 + 対照 2)。

    ★RiMEA Test 4 の「2×2 m」をそのまま置くと、幅 3 m の通路では横断方向の外側
      各 0.5 m が測定から落ちる。高密度では個体が壁際へ押し出されるので、
      **測定密度が名目密度より系統的に低く出る**(実測: 名目 2.0 → 測定 1.43)。
      実測側(`juelich.py`)も「流れ方向 2 m × 通路全幅」で測っている(README §1.4)ので、
      **シムと実測を同じ物差しにする**ためにこちらへ揃える。
      流れ方向の 2 m と「主 1 + 対照 2」は RiMEA の指定どおり。
    """
    xs = (FD_LENGTH / 2.0, FD_LENGTH / 2.0 + FD_LENGTH / 4.0,
          FD_LENGTH / 2.0 - FD_LENGTH / 4.0)
    return [(x - 1.0, 0.0, x + 1.0, FD_WIDTH) for x in xs]


def local_fd(trace, rects, discard_s=DISCARD_S):
    """RiMEA Test 4 の 2×2 m 測定区画 **3 枚をプール**して局所 FD を 1 点測る。

    ★なぜ主区画 1 枚ではなくプールなのか(実測に基づく設計):
      主区画 1 枚(面積 4 m² = 通路 60 m² の 6.7%)だと、高密度で群れが「塊と空隙」に
      分離したときに測定密度が名目密度と逆転する(実測: 名目 1.5 → 測定 1.42、
      名目 2.0 → 測定 1.33)。**測り方由来の非単調**をモデルの非単調と取り違えるので、
      RiMEA が主 1 + 対照 2 を置く流儀そのままに 3 枚を合併して測る(面積 12 m²)。
      主区画と対照区画の食い違い(spread)は診断値として別に残す。
    """
    dt = trace.dt
    T = trace.pos.shape[0]
    t0 = min(T - 1, int(round(discard_s / dt)))
    act = getattr(trace, "active", None)
    area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)
    rho_t, v_t, sp_t, occ = [], [], [], 0
    for t in range(t0, T):
        p, v = trace.pos[t], trace.vel[t]
        m = np.zeros(p.shape[0], dtype=bool)
        for x0, y0, x1, y1 in rects:
            m |= ((p[:, 0] >= x0) & (p[:, 0] < x1)
                  & (p[:, 1] >= y0) & (p[:, 1] < y1))
        if act is not None:
            m = m & act[t]
        n = int(m.sum())
        rho_t.append(n / area)
        if n:
            occ += 1
            v_t.append(float(v[m, 0].mean()))
            sp_t.append(float(np.linalg.norm(v[m], axis=1).mean()))
    return {"rho": float(np.mean(rho_t)) if rho_t else float("nan"),
            "v": float(np.mean(v_t)) if v_t else float("nan"),
            "speed_abs": float(np.mean(sp_t)) if sp_t else float("nan"),
            "n_steps": len(rho_t), "area_m2": area,
            "occupancy": occ / max(1, len(rho_t))}


def fd_sweep(kw, far, densities=SEARCH_DENSITIES, t_total=SEARCH_T_TOTAL,
             discard_s=DISCARD_S):
    """密度掃引。返り値に rho_meas / v_meas(局所測定)と rho_global / v_global を併記。"""
    rects = measurement_rects()
    out = {"rho_nominal": [], "rho_meas": [], "v_meas": [], "speed_meas": [],
           "flow_meas": [], "occupancy": [], "spread_v": [], "spread_rho": [],
           "v_global": [], "speed_global": [], "v0": [], "escaped": [], "n": [],
           "t_total": t_total, "discard_s": discard_s}
    for rho in densities:
        n = int(round(rho * FD_AREA))
        tr = scenarios.run_fd_periodic("sfm_ext", n, length=FD_LENGTH, width=FD_WIDTH,
                                       dt=DT_MAIN, t_total=t_total, engine_kw=kw,
                                       ghost_range=_ghost(far))
        m = local_fd(tr, rects, discard_s=discard_s)
        diag = metrics.rimea_test4_fd(tr, rects, discard_s=discard_s)
        t0 = int(round(discard_s / tr.dt))
        out["rho_nominal"].append(float(rho))
        out["rho_meas"].append(m["rho"])
        out["v_meas"].append(m["v"])
        out["speed_meas"].append(m["speed_abs"])
        out["flow_meas"].append(m["rho"] * m["v"])
        out["occupancy"].append(m["occupancy"])
        out["spread_v"].append(diag.get("spread_v"))
        out["spread_rho"].append(diag.get("spread_rho"))
        y = tr.pos[-1, :, 1]
        out["escaped"].append(int(((y < 0.0) | (y > FD_WIDTH)).sum()))
        out["n"].append(int(n))
        out["v_global"].append(float(tr.vel[t0:, :, 0].mean()))
        out["speed_global"].append(float(np.linalg.norm(tr.vel[t0:], axis=2).mean()))
        out["v0"].append(float(tr.meta["mean_v0"]))
    return out


def bottleneck_inner(kw):
    tr = scenarios.run_bottleneck("sfm_ext", dt=DT_MAIN, engine_kw=kw, **BN_INNER)
    return metrics.specific_flow(tr, tr.meta["line_x"], tr.meta["w"])


def objective(sw, ref, bn=None):
    """L = 重み付き相対誤差²(**局所測定密度で参照曲線を引く**)+ 単調性 + ボトルネック。"""
    rho_m = np.array(sw["rho_meas"], dtype=np.float64)
    v = np.array(sw["v_meas"], dtype=np.float64)
    vref = ref.curve(rho_m, rescaled=True, v0=sw["v0"])
    w = np.array([WEIGHTS.get(float(r), 1.0) for r in sw["rho_nominal"]],
                 dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(vref) & (vref > 1e-6)
    if not ok.any():
        return 1e9, {}
    rel = np.zeros_like(v)
    rel[ok] = (v[ok] - vref[ok]) / vref[ok]
    fd_term = float((w[ok] * rel[ok] ** 2).sum() / w[ok].sum())
    mono = metrics.monotonicity(rho_m, v)
    order = np.argsort(rho_m)
    vv = v[order]
    inc = np.maximum(0.0, vv[1:] - vv[:-1])
    inc = inc[np.isfinite(inc)]
    mono_term = float((inc ** 2).sum())
    parts = {"fd": fd_term, "mono": mono_term, "rel_err": [float(x) for x in rel],
             "mono_violations": mono["violations"],
             "mono_max_increase": mono["max_increase"]}
    total = fd_term + MONO_PENALTY * mono_term
    if bn is not None:
        js = bn.get("J_specific", float("nan"))
        js = 0.0 if not np.isfinite(js) else js
        bn_term = ((js - BN_REF_JS) / BN_REF_JS) ** 2
        parts["bottleneck"] = float(bn_term)
        parts["J_specific"] = float(js)
        total += BN_PENALTY * bn_term
    parts["total"] = float(total)
    return float(total), parts


# ═════════════════════════════════════════════════════════════════════════════
# (0) self-check
# ═════════════════════════════════════════════════════════════════════════════
def section_selfcheck(res):
    out = {"extended_vs_base": engines.verify_extended_matches_base()}
    a = scenarios.run_fd_periodic("sfm", 90, t_total=6.0, dt=0.05)
    ha = metrics.array_hash(a.pos, a.vel)
    # v_of_s=False かつ a2=0 かつ wall_b 既定 = sfm とバイト一致
    c = scenarios.run_fd_periodic(
        "sfm_ext", 90, t_total=6.0, dt=0.05,
        engine_kw={"solver_kw": {"a2": 0.0, "v_of_s": False,
                                 "wall_b": WALL_B_BASE}})
    hc = metrics.array_hash(c.pos, c.vel)
    out["vos_off_identical"] = {"hash_sfm": ha, "hash_sfm_ext": hc,
                                "identical": ha == hc}
    # start_xy 既定 None が従来の run_bottleneck と一致
    b1 = scenarios.run_bottleneck("sfm", w=1.2, n_agents=20, t_total=6.0)
    b2 = scenarios.run_bottleneck("sfm", w=1.2, n_agents=20, t_total=6.0,
                                  start_xy=None)
    out["bottleneck_start_xy_default"] = {
        "identical": metrics.array_hash(b1.pos, b1.vel)
        == metrics.array_hash(b2.pos, b2.vel)}
    res["self_check"] = out
    return out


# ═════════════════════════════════════════════════════════════════════════════
# (1) 壁斥力の較正
# ═════════════════════════════════════════════════════════════════════════════
def section_wall(res, wall_bs=WALL_B_GRID, widths=SINGLE_WIDTHS):
    """(i) 単独通過 (ii) 壁貫通ゼロ (iii) J/w を、WALL_B ごとに測る。"""
    rows = []
    r_single = float(engines.agent_params([0])[1][0])   # agent 0 の体半径
    y_c = BN_ROOM["room_width"] / 2.0
    for wb in wall_bs:
        kw = make_kw(far=False, wall_b=wb)              # 壁の効きだけを見る = 対人は既定
        single = {}
        clear = {}
        for w in widths:
            walls = scenarios._bottleneck_walls(
                BN_ROOM["room_len"], BN_ROOM["room_width"], w, BN_ROOM["exit_len"])
            tr = scenarios.run_bottleneck("sfm_ext", w=w, n_agents=1, dt=DT_MAIN,
                                          t_total=20.0, engine_kw=kw,
                                          start_xy=(1.0, y_c), **BN_ROOM)
            single[str(w)] = bool(tr.meta["n_evacuated"] == 1)
            clear[str(w)] = metrics.wall_clearance_min(
                tr.pos, engines.agent_params([0])[1], walls, active_t=tr.active)
        rows.append({"wall_b": wb, "single_pass": single,
                     "clearance_min_m": clear,
                     "min_pass_width_m": _min_pass(single),
                     "penetration": bool(min(
                         [v for v in clear.values() if np.isfinite(v)] or [0.0]) < 0.0)})
    res["wall_single"] = {"radius_m": r_single, "widths": list(widths), "rows": rows}
    return rows


def _min_pass(single):
    ok = [float(k) for k, v in single.items() if v]
    return min(ok) if ok else None


def _through_partition(tr, room_len, room_width, w):
    """隔壁(開口以外)を**貫通**した個体数。開口線を越えた瞬間の |y − y_c| で判定する。"""
    y_c = room_width / 2.0
    x = tr.pos[:, :, 0]
    cross = (x[:-1] < room_len) & (x[1:] >= room_len)
    bad = 0
    for i in range(x.shape[1]):
        k = np.flatnonzero(cross[:, i])
        if k.size and abs(float(tr.pos[k[0] + 1, i, 1]) - y_c) > w / 2.0 + 1e-3:
            bad += 1
    return int(bad)


def section_wall_flow(res, wall_bs, far=False):
    """WALL_B ごとのボトルネック掃引(J/w と J(w) の線形性)。"""
    out = {}
    for wb in wall_bs:
        kw = make_kw(far=far, wall_b=wb)
        rows = []
        for w in scenarios.BOTTLENECK_WIDTHS:
            walls = scenarios._bottleneck_walls(
                BN_ROOM["room_len"], BN_ROOM["room_width"], w, BN_ROOM["exit_len"])
            tr = scenarios.run_bottleneck("sfm_ext", w=w, n_agents=BN_SWEEP_N,
                                          dt=DT_MAIN, t_total=BN_SWEEP_T,
                                          engine_kw=kw, **BN_ROOM)
            r = metrics.specific_flow(tr, tr.meta["line_x"], w)
            r["w"] = w
            r["n_evacuated"] = int(tr.meta["n_evacuated"])
            r["n_through_partition"] = _through_partition(tr, BN_ROOM["room_len"],
                                                          BN_ROOM["room_width"], w)
            r["wall_clearance_min_m"] = metrics.wall_clearance_min(
                tr.pos, engines.agent_params(list(range(BN_SWEEP_N)))[1], walls,
                active_t=tr.active, sample_every=4)
            rows.append(r)
        slope, icept, r2 = metrics.linear_fit_r2([r["w"] for r in rows],
                                                 [r["J"] for r in rows])
        fin = [r["J_specific"] for r in rows if np.isfinite(r["J_specific"])]
        out[f"wall_b={wb}"] = {
            "wall_b": wb, "far_field": far, "rows": rows,
            "J_vs_w_linear": {"slope": slope, "intercept": icept, "r2": r2},
            "J_specific_mean": float(np.mean(fin)) if fin else float("nan"),
            "in_band_1.5_2.3": [bool(1.5 <= x <= 2.3) for x in fin],
            "min_clearance_m": float(min(r["wall_clearance_min_m"] for r in rows))}
    res.setdefault("wall_flow", {}).update(out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# (2) V(s) の粗グリッド → (3) Nelder-Mead
# ═════════════════════════════════════════════════════════════════════════════
def section_grid(res, ref, wall_b, far, label):
    rows = []
    for T, ell in itertools.product(GRID_T, GRID_L):
        t0 = time.perf_counter()
        kw = make_kw(v_of_s=True, vos_T=T, vos_l=ell, far=far, wall_b=wall_b)
        sw = fd_sweep(kw, far)
        bn = bottleneck_inner(kw)
        tot, parts = objective(sw, ref, bn)
        rows.append({"T": T, "l": ell, "far_field": far, "wall_b": wall_b,
                     "loss": tot, "parts": parts,
                     "rho_meas": sw["rho_meas"], "v_meas": sw["v_meas"],
                     "v_global": sw["v_global"],
                     "wall_s": time.perf_counter() - t0})
    rows.sort(key=lambda r: r["loss"])
    res.setdefault("grid", {})[label] = rows
    return rows


def section_refine(res, ref, x0, wall_b, far, label, max_eval=20):
    """(log T, log ℓ) の 2 変数 Nelder-Mead(決定論・自前実装)。"""
    evals = []

    def loss(p):
        T = float(min(max(np.exp(p[0]), 0.05), 5.0))
        ell = float(min(max(np.exp(p[1]), 0.05), 1.5))
        kw = make_kw(v_of_s=True, vos_T=T, vos_l=ell, far=far, wall_b=wall_b)
        sw = fd_sweep(kw, far)
        bn = bottleneck_inner(kw)
        tot, parts = objective(sw, ref, bn)
        evals.append({"T": T, "l": ell, "loss": tot, "parts": parts,
                      "rho_meas": sw["rho_meas"], "v_meas": sw["v_meas"]})
        return tot

    start = np.array([np.log(x0[0]), np.log(x0[1])], dtype=np.float64)
    best, fbest = metrics.nelder_mead(loss, start, step=np.array([0.35, 0.30]),
                                      max_eval=max_eval)
    out = {"x0": list(x0), "far_field": far, "wall_b": wall_b, "n_eval": len(evals),
           "best": {"T": float(min(max(np.exp(best[0]), 0.05), 5.0)),
                    "l": float(min(max(np.exp(best[1]), 0.05), 1.5))},
           "best_loss": fbest, "evals": evals}
    res.setdefault("refine", {})[label] = out
    return out


# ═════════════════════════════════════════════════════════════════════════════
# (4) 受入判定表(A〜D)
# ═════════════════════════════════════════════════════════════════════════════
def acceptance(sw, ref, bn_sweep=None, tol=BAND_TOL):
    rho = np.array(sw["rho_meas"], dtype=np.float64)
    v = np.array(sw["v_meas"], dtype=np.float64)
    v0 = sw["v0"]
    vref = ref.curve(rho, rescaled=True, v0=v0)
    band = metrics.band_check(v, vref, tol=tol)
    sel = (rho >= 0.5) & (rho <= 2.0)
    band_i2 = metrics.band_check(v[sel], vref[sel], tol=tol)
    lo, hi = ref.envelope(rho, v0=v0)
    env = metrics.envelope_coverage(v, lo, hi)
    mono = metrics.monotonicity(rho, v)
    esc = np.array(sw.get("escaped") or [0] * len(rho), dtype=np.int64)
    clean = esc == 0
    mono_clean = metrics.monotonicity(rho[clean], v[clean])
    d, p = metrics.ks_two_sample(v / max(ref.v0_sim, 1e-9),
                                 vref / max(ref.v0_sim, 1e-9))
    churn = [float(a - b) for a, b in zip(sw["speed_meas"], sw["v_meas"])]
    out = {
        "rho_nominal": sw["rho_nominal"], "rho_meas": sw["rho_meas"],
        "v_meas": sw["v_meas"], "v_ref_at_rho_meas": [float(x) for x in vref],
        "v_global": sw["v_global"],
        "A_band_pm20_region_I_II": {"pass_rate": band_i2["pass_rate"],
                                    "n": band_i2["n_judged"],
                                    "max_abs_rel_err": band_i2.get("max_abs_rel_err"),
                                    "pass": bool(band_i2["pass_rate"] == 1.0)},
        "A_band_pm20_all": {"pass_rate": band["pass_rate"], "rel_err": band["rel_err"],
                            "max_abs_rel_err": band.get("max_abs_rel_err")},
        "B_monotonic": {"violations": mono["violations"],
                        "max_increase_mps": mono["max_increase"],
                        "worst_pair": mono["worst_pair"], "pass": mono["ok"]},
        # ★壁貫通で個体が消えた密度点は「その密度を測っていない」ので別掲する
        "B_monotonic_clean": {"n_points": int(clean.sum()),
                              "violations": mono_clean["violations"],
                              "max_increase_mps": mono_clean["max_increase"],
                              "pass": mono_clean["ok"]},
        "escaped_agents": [int(x) for x in esc], "n_agents": sw.get("n"),
        "C_envelope_p10_p90": {"inside_rate": env["inside_rate"], "n": env["n_judged"],
                               "below": env["below"], "above": env["above"],
                               "pass": bool(env["n_judged"]
                                            and env["inside_rate"] >= 0.8)},
        "KS_vs_reference_curve": {"D": d, "p": p, "note": "少数点なので近似・参考値"},
        "churn_speed_minus_forward": churn,
        "churn_pass": bool(max(churn) < 0.15),
        "occupancy": sw["occupancy"], "spread_v": sw["spread_v"],
    }
    if bn_sweep is not None:
        js = [r["J_specific"] for r in bn_sweep["rows"]]
        fin = [x for x in js if np.isfinite(x)]
        out["D_bottleneck"] = {
            "widths": [r["w"] for r in bn_sweep["rows"]], "J_specific": js,
            "n_finite": len(fin),
            "mean": float(np.mean(fin)) if fin else float("nan"),
            "in_band_1.5_2.3": [bool(1.5 <= x <= 2.3) for x in fin],
            "J_vs_w_r2": bn_sweep["J_vs_w_linear"]["r2"],
            "min_clearance_m": bn_sweep["min_clearance_m"],
            "pass": bool(fin and len(fin) == len(js)
                         and all(1.5 <= x <= 2.3 for x in fin)
                         and bn_sweep["J_vs_w_linear"]["r2"] >= 0.9)}
    return out


def evaluate_final(res, ref, label, *, v_of_s, T, ell, far, wall_b):
    kw = make_kw(v_of_s=v_of_s, vos_T=T, vos_l=ell, far=far, wall_b=wall_b)
    sw = fd_sweep(kw, far, densities=FINAL_DENSITIES, t_total=FINAL_T_TOTAL)
    bs = section_wall_flow({}, (wall_b,), far=far)[f"wall_b={wall_b}"]
    acc = acceptance(sw, ref, bs)
    res.setdefault("finals", {})[label] = {
        "params": {"v_of_s": v_of_s, "T": T, "l": ell, "far_field": far,
                   "wall_b": wall_b, "lambda": LAMBDA_FIXED, "dt": DT_MAIN,
                   "a2": FAR_A2 if far else 0.0, "b2": FAR_B2},
        "sweep": sw, "bottleneck": bs, "acceptance": acc}
    return sw, acc


# ═════════════════════════════════════════════════════════════════════════════
def _plots(res, ref, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        res["plots_error"] = repr(exc)
        return
    rr = np.linspace(0.15, 3.6, 200)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    b = ref.binned
    ax.fill_between(b["rho_per_m2"], b["v_p10_mps"], b["v_p90_mps"],
                    color="0.75", alpha=0.35, label="Julich 10/90 pct envelope")
    ax.plot(rr, ref.curve(rr, rescaled=True), "k-", lw=1.6,
            label=f"reference rescaled to sim v0={ref.v0_sim:.3f}")
    for label, f in sorted(res.get("finals", {}).items()):
        ax.plot(f["sweep"]["rho_meas"], f["sweep"]["v_meas"], "o-", lw=1.4, ms=5,
                label=label)
    ax.set_xlabel("measured density in 2x2 m rect [1/m^2]")
    ax.set_ylabel("forward speed in rect [m/s]")
    ax.set_title("P4-3: FD on LOCAL measurement density (discard 35 s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "calib_p43_fd.png"), dpi=140)
    plt.close(fig)

    ws = res.get("wall_single", {})
    if ws:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for row in ws["rows"]:
            xs = ws["widths"]
            ys = [1 if row["single_pass"][str(w)] else 0 for w in xs]
            ax.plot(xs, ys, "o-", label=f"WALL_B={row['wall_b']}")
        ax.set_xlabel("opening width w [m]")
        ax.set_ylabel("single pedestrian passes (1=yes)")
        ax.set_title("P4-3: wall repulsion calibration (single-agent control)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "calib_p43_wall.png"), dpi=140)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join("reference", "physics_bench", "out"))
    ap.add_argument("--data", default=os.path.join("reference", "physics_bench", "data"))
    ap.add_argument("--refine-eval", type=int, default=20)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    t_start = time.perf_counter()
    v0_sim = float(np.mean(engines.agent_params(list(range(180)))[0]))
    ref = C.Reference(args.data, v0_sim)
    res = {"generated_by": "reference/physics_bench/calibrate_p43.py",
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "platform": platform.platform()},
           "inherited_from_p4_1": {"lambda": LAMBDA_FIXED, "a2": FAR_A2, "b2": FAR_B2,
                                   "cutoff_factor": engines.FAR_CUTOFF_FACTOR,
                                   "taper_m": engines.FAR_TAPER_DEFAULT,
                                   "dt": DT_MAIN, "wall_b_current": WALL_B_BASE},
           "config": {"densities_search": list(SEARCH_DENSITIES),
                      "densities_final": list(FINAL_DENSITIES),
                      "discard_s": DISCARD_S, "t_total_search": SEARCH_T_TOTAL,
                      "t_total_final": FINAL_T_TOTAL,
                      "measurement": "RiMEA Test 4 2x2 m rect (LOCAL density)",
                      "grid_T": list(GRID_T), "grid_l": list(GRID_L),
                      "wall_b_grid": list(WALL_B_GRID),
                      "weights": WEIGHTS, "mono_penalty": MONO_PENALTY,
                      "bn_penalty": BN_PENALTY, "bn_ref_js": BN_REF_JS,
                      "band_tol": BAND_TOL},
           "reference": {"source": "Julich Pedestrian Dynamics Data Archive (CC BY 4.0)",
                         "kladek_fit": {"v_f": ref.vf, "gamma": ref.gamma,
                                        "rho_max": ref.rho_max},
                         "v0_sim_nominal": v0_sim}}

    if args.probe_only:
        for far in (True, False):
            kw = make_kw(v_of_s=True, vos_T=0.5, vos_l=0.5, far=far)
            t0 = time.perf_counter()
            sw = fd_sweep(kw, far)
            print(f"far={far}: {time.perf_counter()-t0:.1f}s "
                  f"rho_meas={[round(x,2) for x in sw['rho_meas']]} "
                  f"v_meas={[round(x,3) for x in sw['v_meas']]}")
        return 0

    print("[0] self check ...")
    sc = section_selfcheck(res)
    print("    extended==base:", sc["extended_vs_base"]["identical"],
          "| v_of_s off == sfm:", sc["vos_off_identical"]["identical"],
          "| start_xy default:", sc["bottleneck_start_xy_default"]["identical"])

    print("[1a] wall calibration: single-pedestrian passage ...")
    wrows = section_wall(res)
    for r in wrows:
        print(f"     WALL_B={r['wall_b']}: min pass width={r['min_pass_width_m']} "
              f"penetration={r['penetration']}")
    # 受入 (i): w=0.9 を単独で通れる / (iii) 壁貫通ゼロ
    cand = [r["wall_b"] for r in wrows
            if r["min_pass_width_m"] is not None and r["min_pass_width_m"] <= 0.9
            and not r["penetration"]]
    print("     candidates (pass w<=0.9, no penetration):", cand)

    print("[1b] wall calibration: bottleneck J/w ...")
    probe_bs = tuple([WALL_B_BASE] + [b for b in cand if b != WALL_B_BASE])
    wf = section_wall_flow(res, probe_bs, far=False)
    # ★順位付けの規約(明示):
    #   失格 = 隔壁を**貫通**した個体が 1 人でも居る(n_through_partition > 0)。
    #     体表が壁面へ押し込まれる(min_clearance_m < 0)のは **現行既定でも常時起きる**
    #     ので失格条件にしない(診断値として残す)。
    #   順位 = ① 通過が成立した開口幅の数(多いほど良い)→ ② mean|J/w − 1.9|(小さいほど良い)
    #   同点(②の差 < 0.02)は **b が大きい方**(= Helbing2000 の 0.08 からの乖離が小さい方)。
    ranked = []
    for k, o in sorted(wf.items()):
        js = [r["J_specific"] for r in o["rows"]]
        pierced = sum(int(r["n_through_partition"]) for r in o["rows"])
        print(f"     {k}: J/w={[None if not np.isfinite(x) else round(x,2) for x in js]}"
              f" R2={o['J_vs_w_linear']['r2']:.3f} "
              f"min_clear={o['min_clearance_m']:.3f} pierced={pierced}")
        fin = [x for x in js if np.isfinite(x)]
        o["n_through_partition_total"] = pierced
        o["mean_abs_err_vs_ref"] = (float(np.mean([abs(x - BN_REF_JS) for x in fin]))
                                    if fin else float("nan"))
        if pierced or not fin:
            continue
        ranked.append((-len(fin), o["mean_abs_err_vs_ref"], -o["wall_b"], o))
    ranked.sort()
    best = ranked[0][3] if ranked else wf[f"wall_b={WALL_B_BASE}"]
    # 同点帯(mean|J/w−1.9| の差 < 0.02)の中では b が最大のものを採る
    if ranked:
        tie = [r for r in ranked if r[0] == ranked[0][0]
               and r[1] - ranked[0][1] < 0.02]
        best = max(tie, key=lambda r: r[3]["wall_b"])[3]
    best_wall = best["wall_b"]
    # 保守側の別解 = 目標(i)(w=0.9 m 単独通過)を満たす **最大の** b
    conservative = max(cand) if cand else WALL_B_BASE
    res["wall_recommended"] = {
        "wall_b": best_wall,
        "mean_abs_err_vs_1.9": best["mean_abs_err_vs_ref"],
        "n_widths_with_flow": int(sum(1 for r in best["rows"]
                                      if np.isfinite(r["J_specific"]))),
        "candidates": cand,
        "conservative_wall_b": conservative,
        "rule": "①通過成立幅数 ②mean|J/w-1.9| ③同点なら b 最大(文献値 0.08 に近い方)。"
                "失格は隔壁貫通のみ(min_clearance<0 は既定でも起きるので失格にしない)"}
    print(f"     -> WALL_B = {best_wall} (conservative={conservative})")

    print("[2] V(s) coarse grid ...")
    grids = {}
    for far in (True, False):
        label = f"far={'on' if far else 'off'}"
        rows = section_grid(res, ref, best_wall, far, label)
        grids[label] = rows
        print(f"     {label}: best loss={rows[0]['loss']:.4f} "
              f"@ T={rows[0]['T']} l={rows[0]['l']} "
              f"(mono_viol={rows[0]['parts'].get('mono_violations')})")
    best_label = min(grids, key=lambda k: grids[k][0]["loss"])
    best_far = best_label == "far=on"
    top = grids[best_label][0]

    print(f"[3] Nelder-Mead refine ({best_label}) ...")
    rf = section_refine(res, ref, (top["T"], top["l"]), best_wall, best_far,
                        best_label, max_eval=args.refine_eval)
    print(f"     best T={rf['best']['T']:.4f} l={rf['best']['l']:.4f} "
          f"loss={rf['best_loss']:.4f}")

    T_best, l_best = rf["best"]["T"], rf["best"]["l"]
    res["recommended"] = {
        "v_of_s": {"enabled": True, "T": T_best, "l": l_best},
        "far_field": {"enabled": best_far, "a2": FAR_A2, "b2": FAR_B2,
                      "cutoff_factor": engines.FAR_CUTOFF_FACTOR,
                      "taper_m": engines.FAR_TAPER_DEFAULT},
        "wall": {"b": best_wall, "a": 2000.0},
        "lambda_aniso": LAMBDA_FIXED, "dt_s": DT_MAIN}

    print("[4] final evaluation ...")
    evaluate_final(res, ref, "baseline_src_defaults", v_of_s=False, T=1.0, ell=0.3,
                   far=False, wall_b=WALL_B_BASE)
    evaluate_final(res, ref, "p4_1_far_only", v_of_s=False, T=1.0, ell=0.3,
                   far=True, wall_b=WALL_B_BASE)
    evaluate_final(res, ref, "p4_3_wall_only", v_of_s=False, T=1.0, ell=0.3,
                   far=False, wall_b=best_wall)
    evaluate_final(res, ref, "p4_3_tuned", v_of_s=True, T=T_best, ell=l_best,
                   far=best_far, wall_b=best_wall)
    for label, f in sorted(res["finals"].items()):
        a = f["acceptance"]
        print(f"     {label}: A={a['A_band_pm20_region_I_II']['pass_rate']:.2f} "
              f"B_viol={a['B_monotonic']['violations']} "
              f"C={a['C_envelope_p10_p90']['inside_rate']:.2f} "
              f"D_mean={a['D_bottleneck']['mean']:.2f}")

    res["wall_s_total"] = time.perf_counter() - t_start
    if not args.skip_plots:
        print("[5] plots ...")
        _plots(res, ref, args.out)
    path = os.path.join(args.out, "calib_p43_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(f"[done] {path}  ({res['wall_s_total']/60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
