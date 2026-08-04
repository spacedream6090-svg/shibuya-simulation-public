"""P4-1 較正ハーネス — Jülich 実測基本図への合わせ込み(**src/ は 1 バイトも触らない**)。

    python -m reference.physics_bench.calibrate --out reference/physics_bench/out

構成(実行順):
  0. self-check   … 拡張が既定経路をバイト単位で変えていないことの実測
  1. λ 先行実験   … 最も安い実験。λ ∈ {0.06,0.08,0.12,0.5} を既存の周期通路で回し、
                    基本図の非単調性(0.70@1.5 → 0.98@3.0 問題)への λ 単独の効果を測る
  2. 本較正       … 粗グリッド (A2,B2,λ) → Nelder-Mead 精密化。dt は外側ループ
  3. ボトルネック … w ∈ {0.8,1.0,1.2,1.6,2.4} m の specific flow J/w
  4. 受入判定表   … ±20% 帯 / 包絡線内率 / 単調性 / RiMEA Test 4 相当

出力: out/calib_results.json(全評価点)/ out/calib_fd.png / out/calib_lambda.png
      / out/calib_bottleneck.png

【参照曲線の作り方(ここが本ハーネスの肝)】
  Jülich の一方向流 3 実験(閉境界 ug / 開境界 uo / BaSiGo uni_corr)の実測 FD を
  Kladek 式に当てはめて (v_f, γ, ρ_max) を得る。**そのうち v_f だけをシムの平均希望
  速度 v0_sim に差し替えた曲線**を参照にする。
    理由: A2・B2・λ は「混雑したときどれだけ減速するか」を決めるパラメータで、
    自由速度そのものは決めない(v0 は agent_id 安定ハッシュ由来の別パラメータ)。
    自由速度の差(実測 ≈1.41 m/s vs シム 1.133 m/s)を較正で埋めようとすると、
    形状の当てはめが自由速度のオフセットに引きずられる。
    → 既存ベンチが「Weidmann 再スケール版」を併記していたのと同じ処置を、
      参照曲線を Weidmann から **Jülich 実測当てはめ** に差し替えて行う。
  絶対値(再スケールなし)の比較も results に併記する(judgement_absolute)。

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

from . import engines, juelich, metrics, scenarios     # noqa: E402

# ── 較正の格子と規模(§3.3 のコスト見積りを 60 分に収める設定) ──
# 密度点。**探索ループ内は ρ=3.0 を外す**(下の理由)。最終評価と λ 先行実験では入れる。
#   理由 1(文献): 研究文書 §2.3 は ±20% 帯を **ρ ∈ [0.5, 2.0](領域 I–II)** に限り、
#     領域 III–IV は単調性だけを課す。スクランブルの作業域も領域 I–II。
#   理由 2(コスト): ρ=3.0(N=180 + 周期像)は 1 掃引の実測コストの約 6 割を占める。
#     60 分予算に収めるための間引き(指示書の「超えそうなら密度点を間引いて明記」に対応)。
CAL_DENSITIES = (0.5, 1.0, 1.5, 2.0)           # 探索ループ内
FINAL_DENSITIES = (0.5, 1.0, 1.5, 2.0, 3.0)    # 最終評価・λ 先行実験・受入判定
CAL_WEIGHTS = {0.5: 1.0, 1.0: 1.5, 1.5: 2.0, 2.0: 2.0, 3.0: 1.0}   # 領域 I–II を重く
FD_LENGTH, FD_WIDTH = 20.0, 3.0                # 既存 fd_periodic と同一
FD_AREA = FD_LENGTH * FD_WIDTH
# 1 ラン 50 s。測定窓は t ∈ [35, 50] s(warm_frac=0.70)。最終評価だけ 60 s([40,60])。
# ★根拠(実測・5 s 窓の平均前進速度):
#   ρ=3.0 / a2=0(現行 src 相当): −1.36 −1.39 −0.81 +0.10 +0.73 +0.88 → 20 s 以降定常
#   ρ=1.5 / a2=0.03(長距離項あり): −0.37 −0.48 −0.48 −0.44 −0.43 −0.31 +0.06 +0.26
#     +0.25 +0.30 +0.31 +0.29 → **35 s 以降でようやく定常**
#   長距離項を入れると過渡が倍近く伸びる。t_total を 24〜32 s にすると過渡を測って
#   しまい、パラメータの効果と区別がつかない(実際に v<0 を「定常値」として拾った)。
CAL_T_TOTAL = 50.0
CAL_WARM_FRAC = 0.70
FINAL_T_TOTAL = 60.0
FINAL_WARM_FRAC = 0.667
DT_GRID = (0.05, 0.02)                         # 外側ループ(最適化変数には入れない)
MONO_PENALTY = 10.0                            # 単調性違反ペナルティの係数
BN_PENALTY = 0.5                               # ボトルネック項の重み
BN_REF_JS = 1.9                                # 参照 specific flow [1/(m·s)](§2.4)
BAND_TOL = 0.20                                # 判定 A: ±20%

# 粗グリッド(4×3×3 = 36 評価 ≤ 60)。A_far/B_far の事前分布は研究文書 §6.1 の表
#   (A_far ∈ [0.04, 0.5] / B_far ∈ [1.6, 3.2])。
# ★A2 の上限を 0.5(VISSIM P0)に置き、それより上は取らない。予備実測で
#   A2=0.5・B2=2.8・λ=0.1 は ρ≥1.5 で前進速度が負(群れが後退)になり、
#   これ以上強くしても意味がないことを確認済み。
GRID_A2 = (0.03, 0.08, 0.20, 0.50)   # [m/s²] Johansson 楕円 0.04 / 円形 0.42 / VISSIM 0.5
GRID_B2 = (1.6, 2.8, 4.0)            # [m]    Johansson 1.65 / 3.22 / VISSIM 2.8
GRID_LAM = (0.06, 0.10, 0.50)        # λ      Johansson 0.06 / VISSIM 0.1 / 本体既定 0.5
LAMBDA_PROBE = (0.06, 0.08, 0.12, 0.50)

# ボトルネック(較正ループ内は 1 条件だけ = コスト抑制)
BN_INNER = dict(w=1.2, n_agents=80, room_len=8.0, room_width=5.0, t_total=30.0)
BN_SWEEP_N = 80
BN_SWEEP_T = 45.0


# ═════════════════════════════════════════════════════════════════════════════
def _mean_forward(tr, warm_frac=CAL_WARM_FRAC):
    t0 = int(tr.pos.shape[0] * warm_frac)
    return float(tr.vel[t0:, :, 0].mean())


def _speed_abs(tr, warm_frac=CAL_WARM_FRAC):
    t0 = int(tr.pos.shape[0] * warm_frac)
    return float(np.linalg.norm(tr.vel[t0:], axis=2).mean())


def far_ghost_range(b2, cutoff_factor=engines.FAR_CUTOFF_FACTOR):
    """長距離項のカットオフを覆うゴースト帯幅 [m](周期性を厳密に保つため)。"""
    return float(cutoff_factor * b2 + 0.7 + 0.3)     # + 体径 + 余裕


def make_kw(a2, b2, lam, cutoff_mode="per_term", taper=engines.FAR_TAPER_DEFAULT):
    """engine_kw(ExtendedSfmEngine 用)を組む。"""
    return {"solver_kw": {"a2": float(a2), "b2": float(b2),
                          "lambda_aniso": float(lam), "lambda_far": float(lam),
                          "cutoff_mode": cutoff_mode, "far_taper": float(taper)}}


def fd_sweep(a2, b2, lam, dt, densities=CAL_DENSITIES, t_total=CAL_T_TOTAL,
             warm_frac=CAL_WARM_FRAC, cutoff_mode="per_term",
             taper=engines.FAR_TAPER_DEFAULT, engine="sfm_ext"):
    """密度掃引 → {"rho","v","speed","v0"}(1 パラメータ組の基本図)。

    v0 は **その密度のラン固有の平均希望速度**(agent_id 安定ハッシュ由来なので
    N が違えば平均も違う)。参照曲線の再スケールに点ごとに使う = 既存ベンチが
    「Weidmann 再スケール版」で mean_v0 を使っていたのと同じ処置。"""
    kw = make_kw(a2, b2, lam, cutoff_mode, taper)
    gr = far_ghost_range(b2) if a2 > 0 else None
    rho_l, v_l, sp_l, v0_l = [], [], [], []
    for rho in densities:
        n = int(round(rho * FD_AREA))
        tr = scenarios.run_fd_periodic(engine, n, length=FD_LENGTH, width=FD_WIDTH,
                                       dt=dt, t_total=t_total, engine_kw=kw,
                                       ghost_range=gr)
        rho_l.append(float(rho))
        v_l.append(_mean_forward(tr, warm_frac))
        sp_l.append(_speed_abs(tr, warm_frac))
        v0_l.append(float(tr.meta["mean_v0"]))
    return {"rho": rho_l, "v": v_l, "speed": sp_l, "v0": v0_l,
            "t_total": t_total, "warm_frac": warm_frac, "dt": dt}


def bottleneck_inner(a2, b2, lam, dt, cutoff_mode="per_term",
                     taper=engines.FAR_TAPER_DEFAULT):
    """較正ループ内で使う 1 条件のボトルネック(specific flow)。"""
    kw = make_kw(a2, b2, lam, cutoff_mode, taper)
    tr = scenarios.run_bottleneck("sfm_ext", dt=dt, engine_kw=kw, **BN_INNER)
    return metrics.specific_flow(tr, tr.meta["line_x"], tr.meta["w"])


# ═════════════════════════════════════════════════════════════════════════════
class Reference:
    """Jülich 実測 → 参照曲線・包絡線・比較の一式。"""

    def __init__(self, data_dir, v0_sim):
        self.data_dir = data_dir
        p = os.path.join(data_dir, "juelich_fd_binned.csv")
        self.binned = juelich.load_binned(p)
        w = self.binned["n"]
        self.vf, self.gamma, self.rho_max = metrics.fit_kladek(
            self.binned["rho_per_m2"], self.binned["v_mean_mps"], weights=w)
        self.v0_sim = float(v0_sim)
        # 実測そのままの点(内挿)と、自由速度をシムに合わせた再スケール曲線
        self.per_dataset = {}
        for ds in ("hermes_uni_closed", "hermes_uni_open", "basigo_uni_corr"):
            q = os.path.join(data_dir, f"juelich_fd_binned_{ds}.csv")
            if os.path.exists(q):
                self.per_dataset[ds] = juelich.load_binned(q)

    def curve(self, rho, rescaled=True, v0=None):
        """参照速度。rescaled=True なら自由速度を v0(点ごと。既定 v0_sim)へ差し替える。"""
        rho = np.asarray(rho, dtype=np.float64)
        if not rescaled:
            return metrics.weidmann_speed(rho, v_f=self.vf, gamma=self.gamma,
                                          rho_max=self.rho_max)
        vfs = (np.full(rho.shape, self.v0_sim) if v0 is None
               else np.asarray(v0, dtype=np.float64))
        return np.array([metrics.weidmann_speed(np.array([r]), v_f=float(f),
                                                gamma=self.gamma,
                                                rho_max=self.rho_max)[0]
                         for r, f in zip(np.atleast_1d(rho), np.atleast_1d(vfs))])

    def raw_points(self, rho):
        return juelich.reference_speed(self.binned, rho)

    def envelope(self, rho, rescaled=True, v0=None):
        lo, hi = juelich.reference_envelope(self.binned, rho)
        if rescaled:
            vfs = (np.full(np.shape(rho), self.v0_sim) if v0 is None
                   else np.asarray(v0, dtype=np.float64))
            s = vfs / self.vf
            lo, hi = lo * s, hi * s
        return lo, hi


def objective(sweep, ref, bn=None):
    """目的関数 L = 重み付き相対誤差² + 単調性ペナルティ + ボトルネック誤差²。"""
    rho = np.array(sweep["rho"], dtype=np.float64)
    v = np.array(sweep["v"], dtype=np.float64)
    vref = ref.curve(rho, rescaled=True, v0=sweep.get("v0"))
    w = np.array([CAL_WEIGHTS.get(float(r), 1.0) for r in rho], dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(vref) & (vref > 1e-6)
    if not ok.any():
        return 1e9, {}
    rel = np.zeros_like(v)
    rel[ok] = (v[ok] - vref[ok]) / vref[ok]
    fd_term = float((w[ok] * rel[ok] ** 2).sum() / w[ok].sum())
    inc = np.maximum(0.0, v[1:] - v[:-1])
    mono_term = float((inc ** 2).sum())
    parts = {"fd": fd_term, "mono": mono_term,
             "rel_err": [float(x) for x in rel]}
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
def section_selfcheck(res):
    """拡張が既定経路をバイト単位で変えていないことの実測(src/ 無改変の裏づけ)。"""
    out = {"extended_vs_base": engines.verify_extended_matches_base()}
    # ghost_range の既定値(None)が従来定数と同一挙動であること
    a = scenarios.run_fd_periodic("sfm", 90, t_total=6.0, dt=0.05)
    b = scenarios.run_fd_periodic("sfm", 90, t_total=6.0, dt=0.05,
                                  ghost_range=scenarios.GHOST_RANGE)
    ha, hb = metrics.array_hash(a.pos, a.vel), metrics.array_hash(b.pos, b.vel)
    # sfm_ext(a2=0)は sfm とバイト一致するはず
    c = scenarios.run_fd_periodic("sfm_ext", 90, t_total=6.0, dt=0.05,
                                  engine_kw={"solver_kw": {"a2": 0.0}})
    hc = metrics.array_hash(c.pos, c.vel)
    out["ghost_range_default_identical"] = {"hash_none": ha, "hash_explicit": hb,
                                            "identical": ha == hb}
    out["sfm_ext_a2zero_identical"] = {"hash_sfm": ha, "hash_sfm_ext": hc,
                                       "identical": ha == hc}
    res["self_check"] = out
    return out


def section_lambda_probe(res, ref, quick=False):
    """§1 λ 先行実験(最安)。λ を単独で振って非単調性がどう動くかだけを見る。"""
    dts = (0.05,) if quick else (0.05, 0.02)
    out = {"note": "a2=0(長距離項なし)= 現行 src と同じ力構造。λ だけを振る対照実験",
           "lambda_values": list(LAMBDA_PROBE), "runs": {}}
    for dt in dts:
        for lam in LAMBDA_PROBE:
            sw = fd_sweep(0.0, engines.FAR_B_DEFAULT, lam, dt,
                          densities=FINAL_DENSITIES, t_total=FINAL_T_TOTAL,
                          warm_frac=FINAL_WARM_FRAC)
            mono = metrics.monotonicity(sw["rho"], sw["v"])
            vref = ref.curve(np.array(sw["rho"]), rescaled=True, v0=sw["v0"])
            band = metrics.band_check(sw["v"], vref, tol=BAND_TOL)
            lo, hi = ref.envelope(np.array(sw["rho"]), v0=sw["v0"])
            env = metrics.envelope_coverage(sw["v"], lo, hi)
            out["runs"][f"dt={dt}.lam={lam}"] = {
                "dt": dt, "lambda": lam, "rho": sw["rho"], "v": sw["v"],
                "speed_abs": sw["speed"], "v0": sw["v0"],
                "v_ref_rescaled": [float(x) for x in vref],
                "monotonicity": mono, "band_pass_rate": band["pass_rate"],
                "band_rel_err": band["rel_err"],
                "envelope_inside_rate": env["inside_rate"]}
    res["lambda_probe"] = out
    return out


def section_grid(res, ref, dt, use_bottleneck=True, quick=False):
    """§2-① 粗グリッド。平坦な谷を見るため全評価点を残す(Johansson の教訓 1)。"""
    a2s = GRID_A2[:2] if quick else GRID_A2
    b2s = GRID_B2[:2] if quick else GRID_B2
    lams = GRID_LAM[:2] if quick else GRID_LAM
    rows = []
    for a2, b2, lam in itertools.product(a2s, b2s, lams):
        t0 = time.perf_counter()
        sw = fd_sweep(a2, b2, lam, dt)
        bn = bottleneck_inner(a2, b2, lam, dt) if use_bottleneck else None
        tot, parts = objective(sw, ref, bn)
        rows.append({"a2": a2, "b2": b2, "lambda": lam, "dt": dt,
                     "rho": sw["rho"], "v": sw["v"], "speed_abs": sw["speed"],
                     "loss": tot, "parts": parts,
                     "monotonicity": metrics.monotonicity(sw["rho"], sw["v"]),
                     "wall_s": time.perf_counter() - t0})
    rows.sort(key=lambda r: r["loss"])
    res.setdefault("grid", {})[f"dt={dt}"] = rows
    return rows


def section_refine(res, ref, x0, dt, use_bottleneck=True, max_eval=30):
    """§2-② Nelder-Mead 精密化(自前実装・決定論)。探索は (log A2, log B2, λ)。"""
    evals = []

    def loss(p):
        a2 = float(np.exp(p[0]))
        b2 = float(np.exp(p[1]))
        lam = float(p[2])
        # 事前分布の外は強いペナルティ(探索が発散しないための箱)
        pen = 0.0
        a2 = min(max(a2, 0.01), 5.0)
        b2 = min(max(b2, 0.5), 8.0)
        if lam < 0.02 or lam > 1.0:
            pen += 10.0 * (max(0.02 - lam, 0.0) + max(lam - 1.0, 0.0)) ** 2
            lam = min(max(lam, 0.02), 1.0)
        sw = fd_sweep(a2, b2, lam, dt)
        bn = bottleneck_inner(a2, b2, lam, dt) if use_bottleneck else None
        tot, parts = objective(sw, ref, bn)
        evals.append({"a2": a2, "b2": b2, "lambda": lam, "loss": tot + pen,
                      "parts": parts, "rho": sw["rho"], "v": sw["v"]})
        return tot + pen

    start = np.array([np.log(x0[0]), np.log(x0[1]), x0[2]], dtype=np.float64)
    best, fbest = metrics.nelder_mead(loss, start, step=np.array([0.5, 0.3, 0.05]),
                                      max_eval=max_eval)
    out = {"x0": list(x0), "dt": dt, "n_eval": len(evals),
           "best": {"a2": float(np.exp(best[0])), "b2": float(np.exp(best[1])),
                    "lambda": float(min(max(best[2], 0.02), 1.0))},
           "best_loss": fbest, "evals": evals}
    res.setdefault("refine", {})[f"dt={dt}"] = out
    return out


def section_bottleneck_sweep(res, params, dt, label):
    """§3 開口幅掃引 w ∈ {0.8,1.0,1.2,1.6,2.4} m。specific flow と J(b) の線形性。"""
    a2, b2, lam = params
    kw = make_kw(a2, b2, lam)
    rows = []
    for w in scenarios.BOTTLENECK_WIDTHS:
        tr = scenarios.run_bottleneck("sfm_ext", w=w, n_agents=BN_SWEEP_N,
                                      dt=dt, t_total=BN_SWEEP_T, engine_kw=kw)
        r = metrics.specific_flow(tr, tr.meta["line_x"], w)
        r["w"] = w
        r["n_evacuated"] = tr.meta["n_evacuated"]
        rows.append(r)
    ws = [r["w"] for r in rows]
    js = [r["J"] for r in rows]
    slope, icept, r2 = metrics.linear_fit_r2(ws, js)
    out = {"params": {"a2": a2, "b2": b2, "lambda": lam, "dt": dt}, "rows": rows,
           "J_vs_w_linear": {"slope": slope, "intercept": icept, "r2": r2},
           "J_specific_mean": float(np.nanmean([r["J_specific"] for r in rows])),
           "reference_J_specific": BN_REF_JS,
           "reference_note": "Seyfried et al. 2009 / Rupprecht et al. 2011, b=0.6-2.5 m"}
    res.setdefault("bottleneck_sweep", {})[label] = out
    return out


def section_rimea_t4(res, params, dt, label, densities=FINAL_DENSITIES):
    """§4 RiMEA Test 4 相当(2×2 m 測定区画・先頭 10 s 破棄・複数密度点)。"""
    a2, b2, lam = params
    kw = make_kw(a2, b2, lam)
    gr = far_ghost_range(b2) if a2 > 0 else None
    rects = metrics.measurement_rects(FD_LENGTH, FD_WIDTH)
    rows = []
    for rho in densities:
        n = int(round(rho * FD_AREA))
        tr = scenarios.run_fd_periodic("sfm_ext", n, length=FD_LENGTH, width=FD_WIDTH,
                                       dt=dt, t_total=FINAL_T_TOTAL,
                                       engine_kw=kw, ghost_range=gr)
        m = metrics.rimea_test4_fd(tr, rects, discard_s=metrics.RIMEA_T4_DISCARD_S)
        m["rho_nominal"] = rho
        # ★RiMEA が許す「先頭 10 s 破棄」では本モデルの過渡(実測 20〜35 s)を
        #   抜けきらない。過渡を完全に外した窓も併記して差を可視化する。
        m2 = metrics.rimea_test4_fd(tr, rects, discard_s=35.0)
        m["v_discard35"] = m2["v"]
        m["rho_discard35"] = m2["rho"]
        rows.append(m)
    res.setdefault("rimea_test4", {})[label] = {
        "params": {"a2": a2, "b2": b2, "lambda": lam, "dt": dt},
        "rect_main": list(rects[0]), "rects": [list(r) for r in rects],
        "note": "discard_s=10 は RiMEA 4.1.1 の規定。discard_s=35 は本モデルの"
                "過渡(実測 20-35 s)を外した対照",
        "rows": rows}
    return rows


def acceptance(sweep, ref, bn_sweep=None, rimea=None, tol=BAND_TOL):
    """受入判定表(研究文書 §2.3 の A/B/C + ボトルネック + RiMEA Test 4)。"""
    rho = np.array(sweep["rho"], dtype=np.float64)
    v = np.array(sweep["v"], dtype=np.float64)
    v0 = sweep.get("v0")
    vref = ref.curve(rho, rescaled=True, v0=v0)
    band = metrics.band_check(v, vref, tol=tol)
    # 帯判定は ρ ∈ [0.5, 2.0](領域 I–II)に限る(§2.3)
    sel = (rho >= 0.5) & (rho <= 2.0)
    band_i2 = metrics.band_check(v[sel], vref[sel], tol=tol)
    lo, hi = ref.envelope(rho, v0=v0)
    env = metrics.envelope_coverage(v, lo, hi)
    mono = metrics.monotonicity(rho, v)
    d, p = metrics.ks_two_sample(v / max(ref.v0_sim, 1e-9), vref / max(ref.v0_sim, 1e-9))
    churn = [float(a - b) for a, b in zip(sweep["speed"], sweep["v"])]
    out = {
        "A_band_pm20_region_I_II": {"pass_rate": band_i2["pass_rate"],
                                    "n": band_i2["n_judged"],
                                    "max_abs_rel_err": band_i2.get("max_abs_rel_err"),
                                    "pass": bool(band_i2["pass_rate"] == 1.0)},
        "A_band_pm20_all": {"pass_rate": band["pass_rate"],
                            "rel_err": band["rel_err"],
                            "max_abs_rel_err": band.get("max_abs_rel_err")},
        "B_monotonic": {"violations": mono["violations"],
                        "max_increase_mps": mono["max_increase"],
                        "worst_pair": mono["worst_pair"], "pass": mono["ok"]},
        "C_envelope_p10_p90": {"inside_rate": env["inside_rate"],
                               "n": env["n_judged"], "below": env["below"],
                               "above": env["above"],
                               "pass": bool(env["n_judged"] and env["inside_rate"] >= 0.8)},
        "KS_vs_reference_curve": {"D": d, "p": p,
                                  "note": "少数点(5)なので近似。参考値扱い"},
        "churn_speed_minus_forward": churn,
        "churn_pass": bool(max(churn) < 0.15),
    }
    if bn_sweep is not None:
        js = [r["J_specific"] for r in bn_sweep["rows"]]
        fin = [x for x in js if np.isfinite(x)]
        out["D_bottleneck"] = {
            "J_specific": js, "widths": [r["w"] for r in bn_sweep["rows"]],
            "n_finite": len(fin),
            "mean": float(np.mean(fin)) if fin else float("nan"),
            "in_band_1.5_2.3": [bool(1.5 <= x <= 2.3) for x in fin],
            "J_vs_w_r2": bn_sweep["J_vs_w_linear"]["r2"],
            "pass": bool(fin and all(1.5 <= x <= 2.3 for x in fin)
                         and bn_sweep["J_vs_w_linear"]["r2"] >= 0.9)}
    if rimea is not None:
        out["E_rimea_test4"] = {
            "rho_measured": [r["rho"] for r in rimea],
            "v_measured": [r["v"] for r in rimea],
            "flow": [r["flow"] for r in rimea],
            "spread_v_main_vs_controls": [r.get("spread_v") for r in rimea],
            "note": "2x2 m 測定区画・先頭 10 s 破棄。主区画と対照区画の食い違いを併記"}
    return out


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

    # (1) 基本図の重ね描き
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    b = ref.binned
    ax.errorbar(b["rho_per_m2"], b["v_mean_mps"], yerr=b["v_sd_mps"], fmt="o",
                ms=3, lw=0.8, color="0.35", alpha=0.7,
                label="Julich measured (binned mean+-sd)")
    ax.fill_between(b["rho_per_m2"], b["v_p10_mps"], b["v_p90_mps"],
                    color="0.75", alpha=0.35, label="Julich 10/90 pct envelope")
    ax.plot(rr, ref.curve(rr, rescaled=False), "k--", lw=1.2,
            label=f"Kladek fit (vf={ref.vf:.2f})")
    ax.plot(rr, ref.curve(rr, rescaled=True), "k-", lw=1.6,
            label=f"reference rescaled to sim v0={ref.v0_sim:.3f}")
    ax.plot(rr, metrics.weidmann_speed(rr), ":", color="0.4", lw=1.0,
            label="Weidmann 1993 (vf=1.34)")
    for label, sw, style in res.get("_fd_overlays", []):
        ax.plot(sw["rho"], sw["v"], style, ms=6, lw=1.4, label=label)
    ax.set_xlabel("density [1/m^2]")
    ax.set_ylabel("forward speed [m/s]")
    ax.set_title("P4-1 fundamental diagram: sim vs Julich")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "calib_fd.png"), dpi=140)
    plt.close(fig)

    # (2) λ 先行実験
    lp = res.get("lambda_probe", {}).get("runs", {})
    if lp:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for key, r in sorted(lp.items()):
            if r["dt"] != 0.02 and any(v["dt"] == 0.02 for v in lp.values()):
                continue
            ax.plot(r["rho"], r["v"], "o-", lw=1.3,
                    label=f"lambda={r['lambda']} (dt={r['dt']})")
        ax.plot(rr, ref.curve(rr, rescaled=True), "k-", lw=1.6, label="reference")
        ax.set_xlabel("density [1/m^2]")
        ax.set_ylabel("forward speed [m/s]")
        ax.set_title("lambda pre-experiment (a2=0, i.e. current src force structure)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "calib_lambda.png"), dpi=140)
        plt.close(fig)

    # (3) ボトルネック
    bs = res.get("bottleneck_sweep", {})
    if bs:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
        for label, o in bs.items():
            ws = [r["w"] for r in o["rows"]]
            axes[0].plot(ws, [r["J"] for r in o["rows"]], "o-", label=label)
            axes[1].plot(ws, [r["J_specific"] for r in o["rows"]], "o-", label=label)
        jp = os.path.join(ref.data_dir, "juelich_bottleneck_flow.csv")
        if os.path.exists(jp):
            d = np.genfromtxt(jp, delimiter=",", names=True)
            axes[0].plot(np.atleast_1d(d["b_exit_m"]), np.atleast_1d(d["J_per_s"]),
                         "ks", label="Julich measured")
            axes[1].plot(np.atleast_1d(d["b_exit_m"]),
                         np.atleast_1d(d["J_specific_per_ms"]), "ks",
                         label="Julich measured")
        axes[1].axhspan(1.5, 2.3, color="0.8", alpha=0.5, label="accept band")
        axes[0].set_xlabel("opening width w [m]")
        axes[0].set_ylabel("flow J [1/s]")
        axes[1].set_xlabel("opening width w [m]")
        axes[1].set_ylabel("specific flow J/w [1/(m s)]")
        for a in axes:
            a.grid(alpha=0.3)
            a.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "calib_bottleneck.png"), dpi=140)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join("reference", "physics_bench", "out"))
    ap.add_argument("--data", default=os.path.join("reference", "physics_bench", "data"))
    ap.add_argument("--quick", action="store_true", help="格子と密度点を間引く")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--no-bottleneck-in-loop", action="store_true",
                    help="較正ループからボトルネック項を外す(時間短縮)")
    ap.add_argument("--refine-eval", type=int, default=30)
    ap.add_argument("--probe-only", action="store_true",
                    help="1 評価分の所要時間だけ測って終了(予算見積り)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    t_start = time.perf_counter()
    v0_sim = float(np.mean(engines.agent_params(list(range(180)))[0]))
    ref = Reference(args.data, v0_sim)
    res = {"generated_by": "reference/physics_bench/calibrate.py",
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "platform": platform.platform()},
           "config": {"densities_search": list(CAL_DENSITIES),
                      "densities_final": list(FINAL_DENSITIES), "weights": CAL_WEIGHTS,
                      "t_total_s": CAL_T_TOTAL, "warm_frac": CAL_WARM_FRAC,
                      "t_total_final_s": FINAL_T_TOTAL,
                      "warm_frac_final": FINAL_WARM_FRAC,
                      "far_cutoff_factor": engines.FAR_CUTOFF_FACTOR,
                      "far_taper_m": engines.FAR_TAPER_DEFAULT,
                      "far_neighbor_cap": engines.FAR_NEIGHBOR_CAP_DEFAULT,
                      "fd_geometry_m": [FD_LENGTH, FD_WIDTH],
                      "grid_a2": list(GRID_A2), "grid_b2": list(GRID_B2),
                      "grid_lambda": list(GRID_LAM), "dt_grid": list(DT_GRID),
                      "mono_penalty": MONO_PENALTY, "bn_penalty": BN_PENALTY,
                      "bn_ref_js": BN_REF_JS, "band_tol": BAND_TOL,
                      "bottleneck_in_loop": not args.no_bottleneck_in_loop},
           "reference": {"source": "Julich Pedestrian Dynamics Data Archive (CC BY 4.0)",
                         "kladek_fit": {"v_f": ref.vf, "gamma": ref.gamma,
                                        "rho_max": ref.rho_max},
                         "v0_sim_nominal": v0_sim,
                         "densities": list(FINAL_DENSITIES),
                         "curve_rescaled_to_v0_sim": [
                             float(x) for x in ref.curve(np.array(FINAL_DENSITIES))],
                         "raw_binned_at_densities": [
                             float(x) for x in ref.raw_points(np.array(FINAL_DENSITIES))],
                         "envelope_p10_p90_rescaled": [
                             [float(x) for x in a]
                             for a in ref.envelope(np.array(FINAL_DENSITIES))]}}
    res["_fd_overlays"] = []

    if args.probe_only:
        t0 = time.perf_counter()
        sw = fd_sweep(0.5, 2.8, 0.1, 0.05)
        t1 = time.perf_counter()
        bn = bottleneck_inner(0.5, 2.8, 0.1, 0.05)
        t2 = time.perf_counter()
        print(f"fd_sweep(dt=0.05): {t1-t0:.1f}s   bottleneck: {t2-t1:.1f}s")
        print("  v:", [round(x, 3) for x in sw["v"]])
        print("  Js:", bn.get("J_specific"))
        t0 = time.perf_counter()
        fd_sweep(0.5, 2.8, 0.1, 0.02)
        print(f"fd_sweep(dt=0.02): {time.perf_counter()-t0:.1f}s")
        return 0

    use_bn = not args.no_bottleneck_in_loop
    print("[0] self check ...")
    sc = section_selfcheck(res)
    print("    extended==base:", sc["extended_vs_base"]["identical"],
          "| ghost default:", sc["ghost_range_default_identical"]["identical"],
          "| sfm_ext(a2=0)==sfm:", sc["sfm_ext_a2zero_identical"]["identical"])

    print("[1] lambda pre-experiment ...")
    lp = section_lambda_probe(res, ref, quick=args.quick)
    for k, r in sorted(lp["runs"].items()):
        print(f"    {k}: v={[round(x,3) for x in r['v']]} "
              f"mono_viol={r['monotonicity']['violations']} "
              f"band={r['band_pass_rate']:.2f}")

    dt_main = 0.05
    print(f"[2a] coarse grid (dt={dt_main}) ...")
    rows = section_grid(res, ref, dt_main, use_bottleneck=use_bn, quick=args.quick)
    print(f"     {len(rows)} evals, best loss={rows[0]['loss']:.4f} "
          f"@ A2={rows[0]['a2']} B2={rows[0]['b2']} lam={rows[0]['lambda']}")

    print(f"[2b] Nelder-Mead refine (dt={dt_main}, max_eval={args.refine_eval}) ...")
    rf = section_refine(res, ref, (rows[0]["a2"], rows[0]["b2"], rows[0]["lambda"]),
                        dt_main, use_bottleneck=use_bn, max_eval=args.refine_eval)
    best = rf["best"]
    print(f"     best A2={best['a2']:.4f} B2={best['b2']:.4f} "
          f"lam={best['lambda']:.4f} loss={rf['best_loss']:.4f}")

    params = (best["a2"], best["b2"], best["lambda"])
    res["recommended"] = {"a2_mps2": params[0], "b2_m": params[1],
                          "lambda_aniso": params[2], "dt_s": dt_main,
                          "cutoff_mode": "per_term",
                          "far_cutoff_m": engines.FAR_CUTOFF_FACTOR * params[1],
                          "far_taper_m": engines.FAR_TAPER_DEFAULT,
                          "short_range_frozen": {"a_N": 2000.0, "b_m": 0.08,
                                                 "note": "A/m=25 m/s^2 は VISSIM P0 と一致"}}

    print("[3] final evaluation at both dt ...")
    finals = {}
    for dt in DT_GRID:
        sw = fd_sweep(*params, dt, densities=FINAL_DENSITIES,
                      t_total=FINAL_T_TOTAL, warm_frac=FINAL_WARM_FRAC)
        bn = bottleneck_inner(*params, dt) if use_bn else None
        tot, parts = objective(sw, ref, bn)
        bs = section_bottleneck_sweep(res, params, dt, f"tuned_dt={dt}")
        rt = section_rimea_t4(res, params, dt, f"tuned_dt={dt}")
        finals[f"dt={dt}"] = {"sweep": sw, "loss": tot, "parts": parts,
                              "acceptance": acceptance(sw, ref, bs, rt)}
        res["_fd_overlays"].append(
            (f"tuned A2={params[0]:.3f} B2={params[1]:.2f} lam={params[2]:.3f} "
             f"(dt={dt})", sw, "o-"))
        print(f"     dt={dt}: loss={tot:.4f} v={[round(x,3) for x in sw['v']]}")

    print("[4] baseline (current src defaults: a2=0, lambda=0.5) ...")
    for dt in DT_GRID:
        sw = fd_sweep(0.0, engines.FAR_B_DEFAULT, 0.5, dt,
                      densities=FINAL_DENSITIES, t_total=FINAL_T_TOTAL,
                      warm_frac=FINAL_WARM_FRAC)
        bs = section_bottleneck_sweep(res, (0.0, engines.FAR_B_DEFAULT, 0.5), dt,
                                      f"baseline_dt={dt}")
        rt = section_rimea_t4(res, (0.0, engines.FAR_B_DEFAULT, 0.5), dt,
                              f"baseline_dt={dt}")
        tot, parts = objective(sw, ref, None)
        finals[f"baseline_dt={dt}"] = {"sweep": sw, "loss": tot, "parts": parts,
                                       "acceptance": acceptance(sw, ref, bs, rt)}
        res["_fd_overlays"].append((f"baseline src defaults (dt={dt})", sw, "s--"))
        print(f"     baseline dt={dt}: loss={tot:.4f} v={[round(x,3) for x in sw['v']]}")

    res["finals"] = finals
    res["wall_s_total"] = time.perf_counter() - t_start

    if not args.skip_plots:
        print("[5] plots ...")
        _plots(res, ref, args.out)
    overlays = res.pop("_fd_overlays", [])
    res["fd_overlays"] = [{"label": lb, "rho": sw["rho"], "v": sw["v"]}
                          for lb, sw, _ in overlays]

    path = os.path.join(args.out, "calib_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(f"[done] {path}  ({res['wall_s_total']/60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
