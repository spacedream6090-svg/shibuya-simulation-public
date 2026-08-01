"""P2 比較ベンチの実行本体。

    python -m reference.physics_bench.run_bench --out reference/physics_bench/out

生成物:
    out/results.json          全指標(決定論ハッシュ・速度・基本図・レーン・境界)
    out/fd_dt*.png            基本図(密度-速度)+ Weidmann(1993) 重ね描き
    out/dt_stability.png      積分安定性(前進速度 vs 物理 dt)
    out/traj_counterflow_*.png / out/traj_crossing.png   軌跡プロット

決定論の検査は「同一プロセス内 2 回走」+「別プロセス実行の results.json 突合」の 2 段。
後者は README の手順どおり本スクリプトを 2 回実行して out/results.json の
determinism.*.hash_run1 を比較する(このファイルは実行のたび同じ値になるはず)。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

# 直接実行(python reference/physics_bench/run_bench.py)でも動くように
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "reference.physics_bench"

from . import metrics, scenarios          # noqa: E402
from .engines import NEIGHBOR_CAP, NEIGHBOR_DIST  # noqa: E402

ENGINE_KINDS = ("sfm", "orca")
FD_DENSITIES = (0.2, 0.5, 1.0, 1.5, 2.0, 3.0)     # [1/m^2]  通路 20m x 3m = 60 m^2
FD_AREA = 60.0
CF_LENGTH = {3.0: 60.0, 6.0: 30.0}                # 幅員違いでも面密度を 200/180 に揃える
CF_N = 200


def _traj_hash(tr):
    return metrics.array_hash(tr.pos, tr.vel)


def _mean_forward(tr, warm_frac=0.5):
    """定常窓(後半)の平均前進速度(x 成分)。基本図の速度側。"""
    t0 = int(tr.pos.shape[0] * warm_frac)
    return float(tr.vel[t0:, :, 0].mean())


def _transient_min(tr, window_s=1.0):
    """過渡の激しさ: 1秒窓ごとの平均前進速度の最小値 [m/s]。

    高密度の初期条件で指数斥力 + 前進 Euler が「群れを後ろへ吹き飛ばす」現象を定量化する。
    負の値が出たら、その時間帯は群れが平均として後退している(非物理)。"""
    w = max(1, int(round(window_s / tr.dt)))
    n = tr.pos.shape[0] // w
    if n == 0:
        return float("nan")
    return float(min(tr.vel[i * w:(i + 1) * w, :, 0].mean() for i in range(n)))


# ─────────────────────────────────────────────────────────────────────────────
def section_determinism(res, quick):
    """同一入力 2 回走で軌跡がバイト一致するか(評価軸①)。"""
    t = 20.0 if quick else 40.0
    cases = {
        "fd_periodic_rho1.5": lambda k: scenarios.run_fd_periodic(k, 90, t_total=t),
        "counterflow_w3": lambda k: scenarios.run_counterflow(
            k, width=3.0, length=CF_LENGTH[3.0], n_agents=CF_N, t_total=t),
        "crossing": lambda k: scenarios.run_crossing(k, n_agents=CF_N, t_total=t),
    }
    out = {}
    for name, fn in cases.items():
        for kind in ENGINE_KINDS:
            h1 = _traj_hash(fn(kind))
            h2 = _traj_hash(fn(kind))
            out[f"{name}.{kind}"] = {
                "hash_run1": h1, "hash_run2": h2, "byte_identical": h1 == h2}
    res["determinism"] = out


def section_throughput(res, quick):
    """速度(評価軸④)。agent·step/s と 200体×1時間相当の実時間を実測から外挿。"""
    t = 20.0 if quick else 30.0
    out = {}
    for kind in ENGINE_KINDS:
        for name, fn in (("counterflow_w3", lambda k: scenarios.run_counterflow(
                              k, width=3.0, length=CF_LENGTH[3.0], n_agents=CF_N, t_total=t)),
                         ("crossing", lambda k: scenarios.run_crossing(
                              k, n_agents=CF_N, t_total=t))):
            t0 = time.perf_counter()
            tr = fn(kind)
            el = time.perf_counter() - t0
            steps = tr.pos.shape[0] - 1
            asps = CF_N * steps / el
            out[f"{name}.{kind}"] = {
                "wall_s": el, "steps": steps, "agent_steps_per_s": asps,
                "sim_s_simulated": t,
                "realtime_factor": t / el,                       # sim秒 / 実秒
                "wall_s_for_200agents_1h_dt0.1": 200 * 36000 / asps,
                "wall_s_for_200agents_1h_dt0.05": 200 * 72000 / asps,
            }
    res["throughput"] = out


def section_fundamental_diagram(res, quick, dts=(0.1, 0.02)):
    """基本図(評価軸③較正可能性)。周期一方向通路で密度を振る。

    測定窓はランの**後半**(高密度では前半に強い過渡があるため)。過渡そのものは
    `transient_min_*`(1秒窓の平均前進速度の最小値)に別途記録する。"""
    t = 30.0 if quick else 60.0
    out = {}
    for dt in dts:
        rows = []
        for rho in FD_DENSITIES:
            n = int(round(rho * FD_AREA))
            row = {"rho": rho, "n_agents": n,
                   "weidmann_vf1.34": float(metrics.weidmann_speed(np.array([rho]))[0])}
            for kind in ENGINE_KINDS:
                tr = scenarios.run_fd_periodic(kind, n, t_total=t, dt=dt)
                row[f"v_{kind}"] = _mean_forward(tr)
                row[f"speed_{kind}"] = float(
                    np.linalg.norm(tr.vel[tr.pos.shape[0] // 2:], axis=2).mean())
                row[f"transient_min_{kind}"] = _transient_min(tr)
                row["mean_v0"] = tr.meta["mean_v0"]
            row["weidmann_rescaled"] = float(
                metrics.weidmann_speed(np.array([rho]), v_f=row["mean_v0"])[0])
            rows.append(row)
        out[f"dt={dt}"] = rows
    res["fundamental_diagram"] = out


def section_dt_stability(res, quick):
    """前進 Euler の積分安定性: 物理 dt を振ったときの前進速度の収束(評価軸①/⑤)。"""
    t = 30.0 if quick else 60.0
    dts = (0.5, 0.2, 0.1, 0.05, 0.02) if not quick else (0.2, 0.1, 0.05)
    out = {}
    for rho in (1.0, 3.0):
        n = int(round(rho * FD_AREA))
        for kind in ENGINE_KINDS:
            vals = {}
            for dt in dts:
                tr = scenarios.run_fd_periodic(kind, n, t_total=t, dt=dt)
                sp = np.linalg.norm(tr.vel[tr.pos.shape[0] // 2:], axis=2)
                vals[str(dt)] = {"v_forward": _mean_forward(tr),
                                 "speed_abs": float(sp.mean()),
                                 "speed_max": float(sp.max()),
                                 "transient_min": _transient_min(tr)}
            out[f"rho{rho}.{kind}"] = vals
    res["dt_stability"] = out


def section_calibration_probe(res, quick):
    """較正可能性(評価軸③): パラメータを動かしたとき基本図がどれだけ動くか。"""
    t = 30.0 if quick else 50.0   # 過渡を抜けた後半窓で比べる(絶対値は fundamental_diagram 節)
    dt = 0.02
    sfm_sets = {
        # 既定 = Helbing, Farkas & Vicsek (2000) の escape-panic パラメータ(sfm_core の確定値)
        "default_A2000_B0.08": {},
        # Helbing & Molnár (1995) の通常歩行パラメータ相当。V0=2.1 m^2/s^2, sigma=0.3 m を
        # 本コアの |f|=A·exp((r_i+r_j-d)/B) 形に読み替えると A ≈ m·V0/sigma = 80·2.1/0.3 = 560 N,
        # B = sigma = 0.3 m(距離の基準点が違うため厳密な等価ではない=近似の目安)。
        "HelbingMolnar1995_A560_B0.3": {"solver_kw": {"a": 560.0, "b": 0.3}},
        "B0.3_onlyrange": {"solver_kw": {"b": 0.3}},
        "A500_B0.6": {"solver_kw": {"a": 500.0, "b": 0.6}},
        "tau1.0": {"solver_kw": {"tau": 1.0}},
    }
    orca_sets = {
        "default_tau2": {},
        "tau4": {"tau": 4.0},
        "tau8": {"tau": 8.0},
        "tau4_radius1.3": {"tau": 4.0, "solver_kw": {"radius_scale": 1.3}},
    }
    out = {}
    for rho in (1.0, 2.0, 3.0):
        n = int(round(rho * FD_AREA))
        rec = {"weidmann_vf1.34": float(metrics.weidmann_speed(np.array([rho]))[0])}
        for label, kw in sfm_sets.items():
            rec[f"sfm.{label}"] = _mean_forward(
                scenarios.run_fd_periodic("sfm", n, t_total=t, dt=dt, engine_kw=kw))
        for label, kw in orca_sets.items():
            rec[f"orca.{label}"] = _mean_forward(
                scenarios.run_fd_periodic("orca", n, t_total=t, dt=dt, engine_kw=kw))
        out[f"rho{rho}"] = rec
    res["calibration_probe"] = out


def section_counterflow(res, quick, traces):
    """対向流(幅員違い)。レーン形成・速度・重なり(評価軸②③)。"""
    t = 30.0 if quick else 60.0
    out = {}
    for width in (3.0, 6.0):
        L = CF_LENGTH[width]
        for kind in ENGINE_KINDS:
            tr = scenarios.run_counterflow(kind, width=width, length=L,
                                           n_agents=CF_N, t_total=t)
            traces[f"counterflow_w{width:g}_{kind}"] = tr
            sign = np.array(tr.meta["dir_sign"])
            T = tr.pos.shape[0]
            phis, nulls, bands = [], [], []
            for ti in range(T // 2, T, 10):
                act = tr.active[ti]
                phi, nul, _ = metrics.lane_order(
                    tr.pos[ti][act], sign[act], 0.0, width,
                    n_bins=max(4, int(width * 2)), x_range=(0.15 * L, 0.85 * L))
                if np.isfinite(phi):
                    phis.append(phi)
                    nulls.append(nul)
                    bands.append(metrics.band_count(tr.pos[ti][act], sign[act], 0.0, width,
                                                    n_bins=max(4, int(width * 2))))
            sp = np.linalg.norm(tr.vel, axis=2)
            m = tr.active.copy()
            m[:T // 2] = False
            fwd = (tr.vel[:, :, 0] * sign[None, :])
            gate_fn = (lambda p, L=L: (p[:, 0] < 3.0) | (p[:, 0] > L - 3.0))
            out[f"w{width:g}.{kind}"] = {
                "width": width, "length": L, "density": CF_N / (width * L),
                "mean_speed": float(sp[m].mean()),
                "mean_forward_speed": float(fwd[m].mean()),
                "efficiency": float(fwd[m].mean() / tr.meta["mean_v0"]),
                "lane_order_phi": float(np.mean(phis)) if phis else float("nan"),
                "lane_order_null": float(np.mean(nulls)) if nulls else float("nan"),
                "band_count": float(np.mean(bands)) if bands else float("nan"),
                "mean_wait_s": tr.meta["mean_wait_s"],
                **metrics.overlap_stats(tr.pos, _radius_of(tr), 20, tr.active),
                "continuity": metrics.continuity_stats(tr.pos, tr.vel, tr.wrap, tr.dt,
                                                       gate_fn, tr.active),
            }
    res["counterflow"] = out

    # 密度掃引(幅 3m 固定)。「どの密度で詰まるか」が両候補の実用差になる。
    sweep = {}
    for n in (40, 80, 120, 200):
        L, width = CF_LENGTH[3.0], 3.0
        for kind in ENGINE_KINDS:
            tr = scenarios.run_counterflow(kind, width=width, length=L,
                                           n_agents=n, t_total=t)
            sign = np.array(tr.meta["dir_sign"])
            T = tr.pos.shape[0]
            m = tr.active.copy()
            m[:T // 2] = False
            phis, nulls = [], []
            for ti in range(T // 2, T, 10):
                act = tr.active[ti]
                phi, nul, _ = metrics.lane_order(tr.pos[ti][act], sign[act], 0.0, width,
                                                 n_bins=6, x_range=(0.15 * L, 0.85 * L))
                if np.isfinite(phi):
                    phis.append(phi)
                    nulls.append(nul)
            fwd = tr.vel[:, :, 0] * sign[None, :]
            sweep[f"n{n}.{kind}"] = {
                "n_agents": n, "density": n / (width * L),
                "efficiency": float(fwd[m].mean() / tr.meta["mean_v0"]),
                "mean_forward_speed": float(fwd[m].mean()),
                "lane_order_phi": float(np.mean(phis)) if phis else float("nan"),
                "lane_order_null": float(np.mean(nulls)) if nulls else float("nan"),
            }
    res["counterflow_density_sweep"] = sweep


def _radius_of(tr):
    from .engines import agent_params
    _, r = agent_params(list(range(tr.pos.shape[1])))
    return r


def section_crossing(res, quick, traces):
    """4方向交差流(スクランブル風)。"""
    t = 30.0 if quick else 60.0
    out = {}
    for kind in ENGINE_KINDS:
        tr = scenarios.run_crossing(kind, n_agents=CF_N, t_total=t)
        traces[f"crossing_{kind}"] = tr
        T = tr.pos.shape[0]
        m = tr.active.copy()
        m[:T // 2] = False
        sp = np.linalg.norm(tr.vel, axis=2)
        size = tr.meta["size"]
        gate_fn = (lambda p, s=size: (p[:, 0] < 3.0) | (p[:, 0] > s - 3.0) |
                                     (p[:, 1] < 3.0) | (p[:, 1] > s - 3.0))
        out[kind] = {
            "mean_speed": float(sp[m].mean()),
            "efficiency": float(sp[m].mean() / tr.meta["mean_v0"]),
            **metrics.overlap_stats(tr.pos, _radius_of(tr), 20, tr.active),
            "continuity": metrics.continuity_stats(tr.pos, tr.vel, tr.wrap, tr.dt,
                                                   gate_fn, tr.active),
        }
    res["crossing"] = out


def section_gate(res, quick):
    """境界縫合性(評価軸②): 流入/流出ゲートの規則による急停止・振動・重なりの差。"""
    t = 20.0 if quick else 40.0
    out = {}
    width, L = 3.0, CF_LENGTH[3.0]
    for mode in ("blind", "guarded"):
        for kind in ENGINE_KINDS:
            tr = scenarios.run_counterflow(kind, width=width, length=L, n_agents=CF_N,
                                           t_total=t, gate_mode=mode)
            gate_fn = (lambda p, L=L: (p[:, 0] < 3.0) | (p[:, 0] > L - 3.0))
            out[f"{mode}.{kind}"] = {
                **metrics.continuity_stats(tr.pos, tr.vel, tr.wrap, tr.dt, gate_fn, tr.active),
                **metrics.overlap_stats(tr.pos, _radius_of(tr), 20, tr.active),
                "n_gate_events": int(tr.wrap.sum()),
                "mean_wait_s": tr.meta["mean_wait_s"],
            }
    res["gate_stitching"] = out


def section_notes(res, quick):
    """自前資産の性質を「主張」ではなく実測で示すメモ。"""
    t, L = 15.0, CF_LENGTH[3.0]

    def _cf(kind, **kw):
        return _traj_hash(scenarios.run_counterflow(
            kind, width=3.0, length=L, n_agents=100, t_total=t, engine_kw=kw))

    h0 = _cf("sfm")
    h1 = _cf("sfm", solver_kw={"noise": 2.0, "noise_seed": 7})
    h1b = _cf("sfm", solver_kw={"noise": 2.0, "noise_seed": 7})
    o0 = _cf("orca")
    o1 = _cf("orca", solver_kw={"pref_noise": 0.2, "noise_seed": 7})
    o1b = _cf("orca", solver_kw={"pref_noise": 0.2, "noise_seed": 7})
    res["engine_notes"] = {
        "wallcrowd_drops_fluctuation_xi": {
            "hash_noise0": h0, "hash_noise2.0": h1, "hash_noise2.0_run2": h1b,
            "identical": h0 == h1,
            "changes_trajectory": h0 != h1, "reproducible": h1 == h1b,
            "fixed_in": "竹-3 (2026-08-02)",
            "note": "【2026-08-01 の実測=バグ】indoor_flow.WallCrowd.forces() が "
                    "sfm_core.Crowd.forces() を上書きして揺らぎ項 ξ を落としており、壁ありの経路では "
                    "noise パラメータが完全に無効だった(identical=true)。"
                    "【竹-3 で修正済み】壁斥力 f_iW と近傍 cap を sfm_core.Crowd の引数へ移し、"
                    "WallCrowd は forces() を上書きしなくなった=ξ は壁ありでも効く。"
                    "以後この項目は identical=false・changes_trajectory=true・reproducible=true "
                    "(同 seed でバイト一致=確率項と決定論の両立)が正常値。"
                    "★ただし counterflow w6(ρ=1.11/m²)で ξ を入れてもレーン秩序 φ は改善しない"
                    "(初期配置 seed 5 本の対照で φ 平均 0.264→0.219〜0.255・seed 間 sd 0.11〜0.14 の"
                    "散らばりの内側)。ξ の大きさは未較正で、実装既定は noise=0.0(無効)のまま。"},
        "orca_pref_noise_works_and_is_deterministic": {
            "hash_noise0": o0, "hash_pref0.2_run1": o1, "hash_pref0.2_run2": o1b,
            "changes_trajectory": o0 != o1, "reproducible": o1 == o1b,
            "note": "orca_min の pref_noise は seed 固定の専用 Generator を消費するので、"
                    "揺らぎを入れても同一 seed なら軌跡はバイト一致(確率項と決定論は両立する)。"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# プロット
# ─────────────────────────────────────────────────────────────────────────────
def _plots(res, traces, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 基本図
    for key, rows in res["fundamental_diagram"].items():
        rho = np.array([r["rho"] for r in rows])
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        grid = np.linspace(0.05, 4.5, 200)
        ax.plot(grid, metrics.weidmann_speed(grid), "k-", lw=2,
                label="Weidmann 1993 ($v_f$=1.34)")
        ax.plot(grid, metrics.weidmann_speed(grid, v_f=rows[0]["mean_v0"]), "k--", lw=1.4,
                label=f"Weidmann rescaled ($v_f$={rows[0]['mean_v0']:.2f})")
        ax.plot(rho, [r["v_sfm"] for r in rows], "o-", color="#c1440e", label="SFM (self-built)")
        ax.plot(rho, [r["v_orca"] for r in rows], "s-", color="#1f5f8b", label="ORCA (self-built)")
        ax.set_xlabel(r"density $\rho$ [1/m$^2$]")
        ax.set_ylabel("mean forward speed [m/s]")
        ax.set_title(f"Fundamental diagram — periodic 20x3 m corridor, {key}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fd_{key.replace('=', '')}.png"), dpi=130)
        plt.close(fig)

    # dt 安定性
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rho in zip(axes, (1.0, 3.0)):
        for kind, c in (("sfm", "#c1440e"), ("orca", "#1f5f8b")):
            d = res["dt_stability"][f"rho{rho}.{kind}"]
            xs = sorted(float(k) for k in d)
            ax.plot(xs, [d[str(x)]["v_forward"] for x in xs], "o-", color=c,
                    label=f"{kind} forward")
            ax.plot(xs, [d[str(x)]["speed_abs"] for x in xs], "^--", color=c, alpha=0.5,
                    label=f"{kind} |v|")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("physics dt [s]")
        ax.set_title(rf"$\rho$={rho} /m$^2$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("speed [m/s]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Integration stability: forward speed vs physics dt")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dt_stability.png"), dpi=130)
    plt.close(fig)

    # 対向流の密度掃引(効率とレーン秩序)
    sw = res["counterflow_density_sweep"]
    ns = sorted({v["n_agents"] for v in sw.values()})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind, c in (("sfm", "#c1440e"), ("orca", "#1f5f8b")):
        rho = [sw[f"n{n}.{kind}"]["density"] for n in ns]
        axes[0].plot(rho, [sw[f"n{n}.{kind}"]["efficiency"] for n in ns], "o-", color=c, label=kind)
        axes[1].plot(rho, [sw[f"n{n}.{kind}"]["lane_order_phi"] for n in ns], "o-", color=c,
                     label=f"{kind} $\\phi$")
        axes[1].plot(rho, [sw[f"n{n}.{kind}"]["lane_order_null"] for n in ns], "x--", color=c,
                     alpha=0.6, label=f"{kind} null")
    axes[0].set_ylabel("forward speed / $v_0$")
    axes[1].set_ylabel(r"lane order $\phi$")
    for ax in axes:
        ax.set_xlabel(r"density $\rho$ [1/m$^2$]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Counterflow (3 m corridor): flow efficiency and lane order vs density")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "counterflow_sweep.png"), dpi=130)
    plt.close(fig)

    # 軌跡
    for name, tr in traces.items():
        T = tr.pos.shape[0]
        t0 = T // 2
        fig, ax = plt.subplots(figsize=(9, 3.4) if "counterflow" in name else (5.4, 5.2))
        if "counterflow" in name:
            sign = np.array(tr.meta["dir_sign"])
            for i in range(0, tr.pos.shape[1], 3):
                m = tr.active[t0:, i]
                if m.sum() < 5:
                    continue
                seg = tr.pos[t0:, i][m]
                cut = np.nonzero(np.abs(np.diff(seg[:, 0])) > 3.0)[0]
                pieces = np.split(seg, cut + 1)
                for pc in pieces:
                    ax.plot(pc[:, 0], pc[:, 1], lw=0.6,
                            color="#c1440e" if sign[i] > 0 else "#1f5f8b")
            ax.set_xlim(0, tr.meta["length"])
            ax.set_ylim(0, tr.meta["width"])
            ax.set_aspect("equal")
        else:
            stream = np.array(tr.meta["stream"])
            cols = ["#c1440e", "#1f5f8b", "#2e7d32", "#6a1b9a"]
            for i in range(0, tr.pos.shape[1], 3):   # 3 は 4 と互いに素 = 4 ストリームすべて拾う
                m = tr.active[t0:, i]
                if m.sum() < 5:
                    continue
                seg = tr.pos[t0:, i][m]
                cut = np.nonzero(np.linalg.norm(np.diff(seg, axis=0), axis=1) > 3.0)[0]
                for pc in np.split(seg, cut + 1):
                    ax.plot(pc[:, 0], pc[:, 1], lw=0.6, color=cols[stream[i]])
            ax.set_xlim(0, tr.meta["size"])
            ax.set_ylim(0, tr.meta["size"])
            ax.set_aspect("equal")
        ax.set_title(f"{name} (2nd half of run)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"traj_{name}.png"), dpi=130)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--quick", action="store_true", help="短い時間で通す(検算用)")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--plots-only", action="store_true",
                    help="既存の results.json から図だけ作り直す(軌跡シナリオのみ再実行)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.plots_only:
        with open(os.path.join(args.out, "results.json"), encoding="utf-8") as f:
            res = json.load(f)
        t = 30.0 if args.quick else 60.0
        traces = {}
        for width in (3.0, 6.0):
            for kind in ENGINE_KINDS:
                traces[f"counterflow_w{width:g}_{kind}"] = scenarios.run_counterflow(
                    kind, width=width, length=CF_LENGTH[width], n_agents=CF_N, t_total=t)
        for kind in ENGINE_KINDS:
            traces[f"crossing_{kind}"] = scenarios.run_crossing(kind, n_agents=CF_N, t_total=t)
        _plots(res, traces, args.out)
        print("replotted into", args.out)
        return res

    res = {"env": {"python": platform.python_version(), "numpy": np.__version__,
                   "platform": platform.platform(), "quick": args.quick,
                   "neighbor_cap": NEIGHBOR_CAP, "neighbor_dist_orca": NEIGHBOR_DIST,
                   "rvo2_binding": "unavailable (see orca_min.py docstring)"}}
    traces = {}
    t_all = time.perf_counter()
    for name, fn in (("determinism", lambda: section_determinism(res, args.quick)),
                     ("throughput", lambda: section_throughput(res, args.quick)),
                     ("fundamental_diagram", lambda: section_fundamental_diagram(res, args.quick)),
                     ("dt_stability", lambda: section_dt_stability(res, args.quick)),
                     ("calibration_probe", lambda: section_calibration_probe(res, args.quick)),
                     ("counterflow", lambda: section_counterflow(res, args.quick, traces)),
                     ("crossing", lambda: section_crossing(res, args.quick, traces)),
                     ("gate_stitching", lambda: section_gate(res, args.quick)),
                     ("engine_notes", lambda: section_notes(res, args.quick))):
        t0 = time.perf_counter()
        fn()
        print(f"[{name}] {time.perf_counter() - t0:.1f}s", flush=True)
    res["env"]["total_wall_s"] = time.perf_counter() - t_all

    if not args.skip_plots:
        _plots(res, traces, args.out)

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, sort_keys=True)
    print("wrote", os.path.join(args.out, "results.json"))
    return res


if __name__ == "__main__":
    main()
