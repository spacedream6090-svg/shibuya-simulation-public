#!/usr/bin/env python
"""Phase E: k スイープ横断解析 CLI(N × k / FSS 対応)。

    python scripts/analyze_sweep.py "runs/pilot_*"
    python scripts/analyze_sweep.py "runs/fss_*"

各 run の config.yaml から run.n_agents と k 実効値を読み、(N, k) で group する。
  k 実効値:  off=0.0 / degraded=degraded_alpha / free=1.0 / sham=0.0(=結合ゼロだが
             計算量同一の R1 対照。連続 k 軸からは別扱い=除外して扱う)
条件横断で以下を runs/_sweep/ に出力する:
  - R^2_external / R^2_internal(seed 階層ブートストラップ 95%CI, B=2000)
  - EWS var_tau / ac1_tau(adoption_frac 系列, measure.ews。三角測量の3本目)
  - fires 合計(R1 監査: 条件間で計算量交絡が無いか)/ compute 合計
  - seed_divergence(同一 (N,k) 内の seed ペア間 adoption_frac 系列の平均 L2 距離)
  - llm_health 3 列(llm_calls_total / llm_cache_hit_rate / llm_fallback_rate の**ラン最終値**。
    observer.llm_health.enabled=false のランは列が無いので n/a 表示=そのランを落とさない)
  - k*(N) 推定雛形: R^2(k) 最大勾配点 / seed_divergence 最大増分点 / EWS τ 最大点の
    3候補 → fss.json + fig_fss.png(k* vs 1/N。N<3 水準なら外挿せず「insufficient」)

traits が無い run では R^2 は None(skip)。図中テキストは英語。
乱数は numpy.default_rng(0) 固定。単一 N の旧形式ラン名(N トークン無し)でも回帰しない。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

from society.observer import measure as m  # noqa: E402

# 連続 k 軸(対照 sham は除外)上での条件並び優先
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000"]
# 別バッチが L2 に足しうる派生系列(存在すれば EWS を追加観測。無くても動く)
#   deviation_mean(第55バッチ タスクA): 裁量ペルソナ逸脱率の日別平均。k を上げると逸脱が増えるか=
#   R²(k) の解釈材料(measure.collective_series が L2 全数値列を素通しするので per-run 系列として自然に載る)。
#   joint_*(第62バッチ 承諾内生化): 承諾率・較正乖離(フェーズ2合否KPI=計画書§3)・内生判定率・履行率。
#   endogenous_accept×k の交互作用(H3)の per-run 系列材料(列が無いラン=OFF では素通り)。
#   quality_magnitude_mean(第65バッチ 関係の質の内生化): 会話由来 magnitude の日別平均
#   (会話の厚みが関係の増減量に効いた度合い。列が無いラン=OFF では素通り)。
#   echo_*/self_similarity_mean/transmission_novel*(第70バッチ IDEA① エコー計測): **常設列**。
#   「同じ語が繰り返し出た」が伝播なのか LLM の反復癖なのかの切り分け材料。造語 k* の主張は
#   transmission_novel_rate が低い(=エコーだらけの)ランでは無効になりうるので、k 掃引の
#   EWS/系列と並べて必ず目視できるようにする。observer.echo.enabled=false のランでは素通り。
_EXTRA_L2_SERIES = ("speech_diversity", "deviation_mean",
                    "joint_accept_rate", "joint_accept_calib_gap",
                    "joint_endo_share", "joint_fulfill_rate",
                    "quality_magnitude_mean",
                    "echo_max", "self_similarity_mean", "echo_utterance_rate",
                    "transmission_novel", "transmission_novel_rate",
                    "active_relations_mean", "dormant_total", "rekindle_total")

# ---- LLM 健全性 KPI(observer.llm_health.enabled=false のランでは **列が存在しない**)---- #
# 源は observer/aggregate.py の 3 列。**いずれもラン開始からの累積**(per-step 差分ではない)
# ので、系列としてではなく「**ランの最終行の値**」= ラン全体の要約として横断表に載せる。
#   llm_calls_total     … 累積呼数 → 最終値 = ラン総呼数(fires/compute と同族の計算量監査量。
#                          セル内では **合計**する)
#   llm_cache_hit_rate  … 累積 hits/累積 calls → 最終値が既に**ラン全体の率**(レート化不要。
#                          セル内では seed 階層ブートストラップの平均+95%CI)
#   llm_fallback_rate   … 同上(発話・熟考のパース失敗率の**下限**)
# ★resume の限界: 3 列とも**プロセス開始**からの累積(observer/silence.py:25 が同じ性質を明記)。
#   途中 resume したランでは最終値が resume 後の区間だけを表すので、`llm_calls_total` は
#   過小に出る(率 2 列は resume 後区間の率になる)。横断表は「1 プロセスで完走したラン」を前提。
# ★条件間の絶対比較に使わない: aggregate.py の但し書きどおり `llm_fallback_rate` の分母は
#   総呼数(plan/reflect/recall/null 込み)なので、purpose 構成が動くと同じ失敗率でも値が動く。
#   本表は **同一構成内での急増検知**の材料であって k 依存性の主張には使えない(図も出さない)。
_LLM_HEALTH_COLS = ("llm_calls_total", "llm_cache_hit_rate", "llm_fallback_rate")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "figure.dpi": 150, "font.size": 10,
})


# --------------------------------------------------------------------------- #
# 条件の読み取り(N, k 実効値, ラベル, 対照フラグ)
# --------------------------------------------------------------------------- #
def _fmt_alpha(a) -> str:
    return "%g" % float(a)


def _cond_label(writeback, alpha) -> str:
    if writeback == "degraded" and alpha is not None:
        return f"dg{_fmt_alpha(alpha)}"
    return str(writeback)


def _k_eff(writeback, alpha):
    """連続 k 軸上の実効値。sham は結合ゼロ(=0)だが対照として別扱いする。"""
    if writeback == "off":
        return 0.0
    if writeback == "free":
        return 1.0
    if writeback == "sham":
        return 0.0
    if writeback == "degraded":
        return float(alpha) if alpha is not None else None
    return None


def _read_cond(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    wb = OmegaConf.select(cfg, "k.writeback")
    if isinstance(wb, bool):  # YAML は無引用の off を False と読む(config.py と同じ正規化)
        wb = "off" if not wb else "free"
    wb = str(wb)
    alpha = OmegaConf.select(cfg, "k.degraded_alpha")
    alpha = float(alpha) if alpha is not None else None
    return {
        "n": int(cfg.run.n_agents),
        "seed": int(cfg.run.seed),
        "writeback": wb,
        "alpha": alpha,
        "label": _cond_label(wb, alpha),
        "k_eff": _k_eff(wb, alpha),
        "is_control": wb == "sham",
    }


# --------------------------------------------------------------------------- #
# 集計補助
# --------------------------------------------------------------------------- #
def _nan_none(v):
    """nan/None を None に正規化(ブートストラップ平均を汚染させない)。"""
    try:
        return None if v is None or math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


def _last_valid(series):
    """系列の**最後の有効値**(末尾の None/nan は飛ばす)。全欠測・列なしなら None。

    llm_health の 3 列は累積量なので「最終行の値 = ラン全体の要約」。列が存在しないラン
    (observer.llm_health.enabled=false)では `collective` にキー自体が無く None を返す
    = 欠測として落ちずに N/A 表示される。
    """
    if not series:
        return None
    for v in reversed(series):
        fv = _nan_none(v)
        if fv is not None:
            return fv
    return None


def _sum_present(rs, key):
    """列を持つランだけを合計する。**1 本も持たなければ None**(0 で埋めない)。"""
    vals = [r[key] for r in rs if r.get(key) is not None]
    return float(sum(vals)) if vals else None


def _bootstrap_ci(values, b=2000):
    """seed ブロックを resample する階層ブートストラップの平均と 95%CI。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        v = float(arr[0])
        return {"mean": v, "lo": v, "hi": v, "n": 1}
    rng = np.random.default_rng(0)  # 決定論
    n = arr.size
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(b)])
    return {"mean": float(arr.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": n}


def _agg_metric(rs, key):
    """seed ごとに平均 → seed ブロックで階層ブートストラップした CI。"""
    by_seed = defaultdict(list)
    for r in rs:
        v = r.get(key)
        if v is not None:
            by_seed[r["seed"]].append(float(v))
    per_seed = [float(np.mean(v)) for v in by_seed.values()]
    return _bootstrap_ci(per_seed)


def _seed_divergence(rs):
    """同一 (N,k) 内、異なる seed の run ペア間 adoption_frac L2 距離の平均。"""
    dists = []
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if rs[i]["seed"] == rs[j]["seed"]:
                continue
            a = np.asarray(rs[i]["adoption_frac"], dtype=float)
            bb = np.asarray(rs[j]["adoption_frac"], dtype=float)
            length = min(a.size, bb.size)
            if length == 0:
                continue
            dists.append(float(np.sqrt(np.sum((a[:length] - bb[:length]) ** 2))))
    return float(np.mean(dists)) if dists else None


def _order_labels(cells):
    """非対照は k_eff 昇順、対照(sham)は末尾。"""
    def key(lab):
        c = cells[lab]
        ke = c["k_eff"] if c["k_eff"] is not None else 99.0
        return (1 if c["is_control"] else 0, ke, lab)
    return sorted(cells, key=key)


# --------------------------------------------------------------------------- #
# k*(N) 推定雛形(3候補)
# --------------------------------------------------------------------------- #
def _kaxis_points(cells):
    """連続 k 軸(対照除外)を k_eff 昇順で返す [(k, label, cell), ...]。"""
    pts = [(c["k_eff"], lab, c) for lab, c in cells.items()
           if not c["is_control"] and c["k_eff"] is not None]
    pts.sort(key=lambda t: t[0])
    return pts


def _steepest_drop_midpoint(xs, ys):
    """隣接区間で最も急な下降(R^2 が最も落ちる)勾配の中点。<2 有効点なら None。"""
    best, best_slope = None, None
    for i in range(len(xs) - 1):
        y0, y1, dx = ys[i], ys[i + 1], xs[i + 1] - xs[i]
        if y0 is None or y1 is None or dx == 0:
            continue
        slope = (y1 - y0) / dx
        if best_slope is None or slope < best_slope:
            best_slope, best = slope, 0.5 * (xs[i] + xs[i + 1])
    return best


def _steepest_rise_midpoint(xs, ys):
    """隣接区間で最も急な上昇(seed_divergence が最も跳ねる)勾配の中点。"""
    best, best_slope = None, None
    for i in range(len(xs) - 1):
        y0, y1, dx = ys[i], ys[i + 1], xs[i + 1] - xs[i]
        if y0 is None or y1 is None or dx == 0:
            continue
        slope = (y1 - y0) / dx
        if best_slope is None or slope > best_slope:
            best_slope, best = slope, 0.5 * (xs[i] + xs[i + 1])
    return best


def _argmax_x(xs, ys):
    """値が最大となる k(EWS τ のピーク)。全 None なら None。"""
    best, best_y = None, None
    for x, y in zip(xs, ys):
        if y is None:
            continue
        if best_y is None or y > best_y:
            best_y, best = y, x
    return best


def _kstar_for_N(cells):
    pts = _kaxis_points(cells)
    ks = [p[0] for p in pts]
    r2 = [p[2]["R2_external"]["mean"] for p in pts]
    dv = [p[2]["seed_divergence"] for p in pts]
    ews = [p[2]["EWS_var"]["mean"] for p in pts]
    return {
        "k_grid": ks,
        "r2_external": r2,
        "seed_divergence": dv,
        "ews_var_tau": ews,
        "kstar_r2_gradient": _steepest_drop_midpoint(ks, r2),
        "kstar_divergence_jump": _steepest_rise_midpoint(ks, dv),
        "kstar_ews_peak": _argmax_x(ks, ews),
    }


def _linfit_extrapolate(inv_n, kstar):
    """k* vs 1/N を最小二乗直線に載せ、1/N->0(熱力学極限)の切片を返す。"""
    xs = [x for x, y in zip(inv_n, kstar) if y is not None]
    ys = [y for y in kstar if y is not None]
    if len(xs) < 2:
        return None
    a, b = np.polyfit(np.asarray(xs), np.asarray(ys), 1)  # y = a*x + b
    return {"slope": float(a), "intercept_at_inf_N": float(b)}


# --------------------------------------------------------------------------- #
# 図
# --------------------------------------------------------------------------- #
def _cell_mean(cell, key):
    v = cell.get(key)
    if isinstance(v, dict):
        return v.get("mean"), (v.get("lo"), v.get("hi"))
    return v, None


def _fig_vs_k(table, N_levels, key, ylabel, title, path, with_ci=True):
    """x=k_eff の連続軸に N ごとの折れ線。対照 sham は白抜き四角で別表示。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_val = False
    for i, N in enumerate(N_levels):
        col = PALETTE[i % len(PALETTE)]
        cells = table[N]
        pts = _kaxis_points(cells)
        xs, ys, los, his = [], [], [], []
        for k, _lab, c in pts:
            mean, ci = _cell_mean(c, key)
            if mean is None:
                continue
            xs.append(k)
            ys.append(mean)
            if with_ci and ci and ci[0] is not None:
                los.append(mean - ci[0])
                his.append(ci[1] - mean)
            else:
                los.append(0.0)
                his.append(0.0)
        if xs:
            any_val = True
            ax.errorbar(xs, ys, yerr=[los, his] if with_ci else None,
                        fmt="o-", color=col, capsize=3, lw=1.6, label=f"N={N}")
        # 対照(sham)= 白抜き四角(結合ゼロ・計算量同一)
        for lab, c in cells.items():
            if not c["is_control"]:
                continue
            mean, _ = _cell_mean(c, key)
            if mean is None:
                continue
            ax.scatter([c["k_eff"]], [mean], marker="s", s=48,
                       facecolors="white", edgecolors=col, linewidths=1.4,
                       zorder=3, label=f"N={N} {lab}(ctrl)")
    if not any_val:
        ax.text(0.5, 0.5, f"no data for {key}", transform=ax.transAxes,
                ha="center", va="center")
    else:
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("k (off=0, degraded=alpha, free=1; sham=control at 0)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_r2_by_k(table, N_levels, path):
    """R^2 external(実線)/ internal(破線)を N ごとに連続 k 軸で。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_val = False
    for i, N in enumerate(N_levels):
        col = PALETTE[i % len(PALETTE)]
        cells = table[N]
        pts = _kaxis_points(cells)
        for key, style, tag in [("R2_external", "o-", "ext"),
                                ("R2_internal", "s--", "int")]:
            xs, ys, los, his = [], [], [], []
            for k, _lab, c in pts:
                mean, ci = _cell_mean(c, key)
                if mean is None:
                    continue
                xs.append(k)
                ys.append(mean)
                los.append(mean - ci[0] if ci and ci[0] is not None else 0.0)
                his.append(ci[1] - mean if ci and ci[1] is not None else 0.0)
            if xs:
                any_val = True
                ax.errorbar(xs, ys, yerr=[los, his], fmt=style, color=col,
                            capsize=3, lw=1.5, label=f"N={N} {tag}")
    if not any_val:
        ax.text(0.5, 0.5, "no traits in any run -> R^2 unavailable",
                transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("k (off=0, degraded=alpha, free=1)")
    ax.set_ylabel("R^2 (Y ~ traits)")
    ax.set_title("Explained variance of world-change by traits vs k")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_fss(fss, path):
    """k* vs 1/N。3 推定量を別系列で。N<3 水準なら外挿せず注記。"""
    by_n = fss["by_N"]
    Ns = sorted(int(n) for n in by_n)
    inv_n = [1.0 / n for n in Ns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    estimators = [("kstar_r2_gradient", PALETTE[0], "R^2 max-drop"),
                  ("kstar_divergence_jump", PALETTE[3], "divergence max-jump"),
                  ("kstar_ews_peak", PALETTE[2], "EWS var peak")]
    any_val = False
    for est, col, lab in estimators:
        ys = [by_n[str(n)].get(est) for n in Ns]
        xs = [x for x, y in zip(inv_n, ys) if y is not None]
        yv = [y for y in ys if y is not None]
        if yv:
            any_val = True
            ax.scatter(xs, yv, color=col, s=55, label=lab, zorder=3)
        if fss["sufficient"]:
            fit = _linfit_extrapolate(inv_n, ys)
            if fit is not None:
                xline = np.linspace(0.0, max(inv_n) * 1.05, 20)
                ax.plot(xline, fit["slope"] * xline + fit["intercept_at_inf_N"],
                        color=col, lw=1.0, ls="--", alpha=0.7)
                ax.scatter([0.0], [fit["intercept_at_inf_N"]], color=col,
                           marker="*", s=130, zorder=4)
    if not any_val:
        ax.text(0.5, 0.5, "no k* candidates (need >=2 k levels per N)",
                transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend(fontsize=8, frameon=False, title="k* estimator")
    if not fss["sufficient"]:
        ax.text(0.5, 0.92,
                f"insufficient N levels ({fss['n_levels']} < 3): "
                "no extrapolation to 1/N->0",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=PALETTE[3])
    ax.set_xlabel("1 / N")
    ax.set_ylabel("estimated k*")
    ax.set_title("Finite-size scaling of k* (three triangulating estimators)")
    ax.set_xlim(left=-0.005)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _ci_str(cell):
    if not isinstance(cell, dict) or cell.get("mean") is None:
        return "n/a"
    return f"{cell['mean']:.3f} [{cell['lo']:.3f},{cell['hi']:.3f}]"


def _num(v, fmt="{:.4f}"):
    return fmt.format(v) if v is not None else "n/a"


def _write_report(path, N_levels, table, fss):
    lines = ["# k-sweep report (N x k)\n"]
    lines.append(f"- N levels: {N_levels}")
    all_labels = sorted({lab for N in N_levels for lab in table[N]})
    lines.append(f"- conditions: {all_labels}\n")

    lines.append("## Per-N tables\n")
    for N in N_levels:
        cells = table[N]
        lines.append(f"### N = {N}\n")
        lines.append("| k | k_eff | ctrl | n_runs | n_seeds | R2_ext (95%CI) | "
                     "R2_int (95%CI) | EWS_var (adopt) | EWS_ac1 | fires | "
                     "compute | seed_div |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in _order_labels(cells):
            c = cells[lab]
            ke = _num(c["k_eff"], "{:.2f}")
            lines.append(
                f"| {lab} | {ke} | {'Y' if c['is_control'] else ''} | "
                f"{c['n_runs']} | {c['n_seeds']} | {_ci_str(c['R2_external'])} | "
                f"{_ci_str(c['R2_internal'])} | {_ci_str(c['EWS_var'])} | "
                f"{_ci_str(c['EWS_ac1'])} | {c['fires_total']} | "
                f"{c['compute_total']} | {_num(c['seed_divergence'])} |")
        lines.append("")

    lines.append("## LLM health (observer.llm_health, cumulative columns)\n")
    lines.append("Run-final values of the three cumulative L2 columns. Runs with "
                 "`observer.llm_health.enabled=false` have **no such columns** and "
                 "show n/a (they are not dropped from any other row). "
                 "`llm_calls` is summed over the runs of the cell; the two rates "
                 "are already run-level ratios, so they get the seed-hierarchical "
                 "bootstrap mean [95%CI].\n")
    lines.append("| N | k | runs w/ llm_health | llm_calls (sum) | "
                 "cache_hit_rate | fallback_rate |")
    lines.append("|---|---|---|---|---|---|")
    for N in N_levels:
        cells = table[N]
        for lab in _order_labels(cells):
            c = cells[lab]
            lines.append(
                f"| {N} | {lab} | {c['llm_health_runs']}/{c['n_runs']} | "
                f"{_num(c['llm_calls_total'], '{:.0f}')} | "
                f"{_ci_str(c['llm_cache_hit_rate'])} | "
                f"{_ci_str(c['llm_fallback_rate'])} |")
    lines.append("")
    lines.append("> Two honest limits carried over from `observer/aggregate.py` and "
                 "`observer/silence.py`: (1) all three are cumulative **from process "
                 "start**, so a resumed run reports only its post-resume segment "
                 "(llm_calls under-counts); (2) `fallback_rate` counts only "
                 "utterance/deliberation parse failures over **all** LLM calls, so it "
                 "is a LOWER BOUND and its denominator moves with purpose mix -- use "
                 "it to detect a spike within one configuration, NOT to compare k "
                 "levels. No figure is drawn for it on purpose.\n")

    lines.append("## Triangulation of k* (three legs)\n")
    lines.append("Legs: (1) R^2(Y~traits) FALLS toward free (traits stop "
                 "explaining who changes the world), (2) seed_divergence RISES "
                 "(path dependence), (3) EWS var_tau of adoption_frac PEAKS "
                 "(critical slowing down). k* is where they coincide.\n")
    lines.append("| N | k*(R2 max-drop) | k*(divergence max-jump) | "
                 "k*(EWS var peak) |")
    lines.append("|---|---|---|---|")
    for N in N_levels:
        f = fss["by_N"][str(N)]
        lines.append(f"| {N} | {_num(f['kstar_r2_gradient'], '{:.3f}')} | "
                     f"{_num(f['kstar_divergence_jump'], '{:.3f}')} | "
                     f"{_num(f['kstar_ews_peak'], '{:.3f}')} |")
    lines.append("")

    lines.append("## Finite-size scaling (k* vs 1/N)\n")
    if fss["sufficient"]:
        lines.append(f"- {fss['n_levels']} N levels -> line fit + "
                     "extrapolation to 1/N->0 (thermodynamic k*):")
        ex = fss.get("extrapolation", {})
        for est, val in ex.items():
            if val is not None:
                lines.append(f"  - {est}: k*_inf = "
                             f"{val['intercept_at_inf_N']:.3f} "
                             f"(slope {val['slope']:.3f})")
    else:
        lines.append(f"- {fss['n_levels']} N level(s) < 3: **insufficient** "
                     "for FSS. Points plotted without extrapolation "
                     "(see fig_fss.png).")
    lines.append("")

    lines.append("## R1 audit (compute confound)\n")
    lines.append("fires and compute must be ~flat across k within each N: a "
                 "k-trend there means compute is confounded with the effect. "
                 "sham (control) has ZERO coupling at MATCHED compute -- the "
                 "three legs must NOT appear for sham.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def analyze_sweep(pattern, out=None, allow_tier_mismatch: bool = False):
    """glob パターン(または dir リスト)を横断解析し、出力ディレクトリを返す。"""
    if isinstance(pattern, (list, tuple)):
        candidates = list(pattern)
    else:
        candidates = sorted(glob.glob(pattern))
    dirs = [d for d in candidates
            if os.path.isfile(os.path.join(d, "config.yaml"))
            and os.path.isfile(os.path.join(d, "l1_events.parquet"))]
    if not dirs:
        raise SystemExit(f"no runs matched: {pattern}")
    # 第72バッチ: ラン間比較ガード(run_mode / 再現性等級の混在は既定で拒否)。
    from society.registry import guard_or_die
    guard_or_die(dirs, allow_mismatch=allow_tier_mismatch)

    runs = []
    for d in dirs:
        cond = _read_cond(d)
        events = m.load_events(d)
        agents = m.load_agents(d)
        l2 = m.load_l2(d)
        traits, tnames, _src = m.load_traits(d, agents)
        n_agents = len(agents) or (max((e["agent_id"] for e in events),
                                       default=-1) + 1)
        feats = m.agent_features(events, agents, traits)
        collective = m.collective_series(events, l2, n_agents)
        r2 = m.r2_traits(feats, tnames)
        ews_adopt = m.ews(collective["adoption_frac"])
        ews_entropy = m.ews(collective["vocab_entropy"])
        # defensive: 別バッチが L2 に足しうる派生系列(存在すれば拾う)
        ews_extra = {}
        for col in _EXTRA_L2_SERIES:
            vals = collective.get(col)
            if vals and any(v is not None for v in vals):
                e = m.ews([v if v is not None else 0.0 for v in vals])
                ews_extra[col] = {"var_tau": _nan_none(e["var_tau"]),
                                  "ac1_tau": _nan_none(e["ac1_tau"])}
        # LLM 健全性 3 列(累積列 → 最終行の値。列が無いラン=llm_health OFF では None)
        llm_health = {col: _last_valid(collective.get(col))
                      for col in _LLM_HEALTH_COLS}
        runs.append({
            "dir": d, **cond, **llm_health,
            "has_llm_health": any(v is not None for v in llm_health.values()),
            "r2_ext": r2["external"]["r2"], "r2_int": r2["internal"]["r2"],
            "fires": sum(f["n_fires_received"] for f in feats),
            "compute": sum(f["compute"] for f in feats),
            "adoption_frac": collective["adoption_frac"],
            "ews_adopt_var": _nan_none(ews_adopt["var_tau"]),
            "ews_adopt_ac1": _nan_none(ews_adopt["ac1_tau"]),
            "ews_entropy_var": _nan_none(ews_entropy["var_tau"]),
            "ews_entropy_ac1": _nan_none(ews_entropy["ac1_tau"]),
            "ews_extra": ews_extra,
        })

    # (N, label) で group
    by_nk = defaultdict(list)
    for r in runs:
        by_nk[(r["n"], r["label"])].append(r)

    N_levels = sorted({r["n"] for r in runs})
    table = {N: {} for N in N_levels}
    for (N, lab), rs in by_nk.items():
        table[N][lab] = {
            "k_eff": rs[0]["k_eff"],
            "is_control": rs[0]["is_control"],
            "n_runs": len(rs),
            "n_seeds": len({r["seed"] for r in rs}),
            "R2_external": _agg_metric(rs, "r2_ext"),
            "R2_internal": _agg_metric(rs, "r2_int"),
            "EWS_var": _agg_metric(rs, "ews_adopt_var"),
            "EWS_ac1": _agg_metric(rs, "ews_adopt_ac1"),
            "EWS_entropy_var": _agg_metric(rs, "ews_entropy_var"),
            "fires_total": int(sum(r["fires"] for r in rs)),
            "compute_total": int(sum(r["compute"] for r in rs)),
            "seed_divergence": _seed_divergence(rs),
            # LLM 健全性(llm_health OFF のランしか無いセルは総数 None / CI mean None = n/a)
            "llm_calls_total": _sum_present(rs, "llm_calls_total"),
            "llm_cache_hit_rate": _agg_metric(rs, "llm_cache_hit_rate"),
            "llm_fallback_rate": _agg_metric(rs, "llm_fallback_rate"),
            "llm_health_runs": int(sum(1 for r in rs if r["has_llm_health"])),
        }

    # k*(N) 推定 + FSS
    by_N_fss = {str(N): _kstar_for_N(table[N]) for N in N_levels}
    n_levels = len(N_levels)
    fss = {
        "n_levels": n_levels,
        "sufficient": n_levels >= 3,
        "N_levels": N_levels,
        "by_N": by_N_fss,
    }
    if fss["sufficient"]:
        inv_n = [1.0 / N for N in N_levels]
        fss["extrapolation"] = {
            est: _linfit_extrapolate(
                inv_n, [by_N_fss[str(N)][est] for N in N_levels])
            for est in ("kstar_r2_gradient", "kstar_divergence_jump",
                        "kstar_ews_peak")}

    out = out or os.path.join(os.path.dirname(dirs[0]) or "runs", "_sweep")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "sweep_table.json"), "w", encoding="utf-8") as fh:
        json.dump({"N_levels": N_levels, "table": table,
                   "runs": [{k: r[k] for k in ("dir", "n", "label", "k_eff",
                                               "is_control", "seed", "r2_ext",
                                               "r2_int", "ews_adopt_var",
                                               "ews_adopt_ac1", "fires",
                                               "compute", "llm_calls_total",
                                               "llm_cache_hit_rate",
                                               "llm_fallback_rate")}
                            for r in runs]},
                  fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "seed_divergence.json"), "w",
              encoding="utf-8") as fh:
        json.dump({f"N{N}|{lab}": table[N][lab]["seed_divergence"]
                   for N in N_levels for lab in table[N]},
                  fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "fss.json"), "w", encoding="utf-8") as fh:
        json.dump(fss, fh, ensure_ascii=False, indent=1)

    _fig_r2_by_k(table, N_levels, os.path.join(out, "fig_r2_by_k.png"))
    _fig_vs_k(table, N_levels, "fires_total", "total drive fires",
              "Compute-confound audit (fires vs k)",
              os.path.join(out, "fig_fires_by_k.png"), with_ci=False)
    _fig_vs_k(table, N_levels, "seed_divergence",
              "mean L2 distance (adoption_frac)", "Seed divergence vs k",
              os.path.join(out, "fig_divergence.png"), with_ci=False)
    _fig_vs_k(table, N_levels, "EWS_var",
              "EWS var_tau (adoption_frac)", "Early-warning signal vs k",
              os.path.join(out, "fig_ews_by_k.png"), with_ci=True)
    _fig_fss(fss, os.path.join(out, "fig_fss.png"))
    _write_report(os.path.join(out, "report_sweep.md"), N_levels, table, fss)

    print(f"[sweep] {len(runs)} runs, N={N_levels}, "
          f"conditions per N={[len(table[N]) for N in N_levels]} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase E cross-condition k sweep")
    ap.add_argument("pattern", help='glob, e.g. "runs/pilot_*" or "runs/fss_*"')
    ap.add_argument("--out", default=None, help="output dir (default runs/_sweep)")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="run_mode / 再現性等級が異なるラン同士の横断解析を明示的に許可する"
                         "(既定は拒否)")
    args = ap.parse_args()
    analyze_sweep(args.pattern, args.out,
                  allow_tier_mismatch=args.allow_tier_mismatch)


if __name__ == "__main__":
    main()
