"""測定指標 — 決定論ハッシュ・基本図・レーン形成・境界縫合性。

出典(数値の根拠。捏造なし・URL は docs/research/physics-engine-selection.md に再掲):
  - Weidmann (1993) の基本図(Kladek 式):
        v(ρ) = v_f · [ 1 − exp( −γ · (1/ρ − 1/ρ_max) ) ]
        v_f = 1.34 m/s, γ = 1.913 m^-2, ρ_max = 5.4 m^-2
    (Kretz, "The Social Force Model and its Relation to the Kladek Formula",
     arXiv:1512.01426 が Weidmann のパラメータをこの値で引用)
  - レーン形成の秩序変数は Feliciani & Nishinari (2016) PRE 94, 032304 が
    2次元の order parameter を定義しているが、本ベンチでは**その式は使わず**、
    横断方向ビンごとの方向分離度という簡略版を自前定義する(下記 lane_order の註)。
    有限サイズ由来のベースラインを必ず併記する。
"""
from __future__ import annotations

import hashlib

import numpy as np

# ── Weidmann (1993) / Kladek 式の確定パラメータ ──
WEIDMANN_VF = 1.34        # 自由歩行速度 [m/s]
WEIDMANN_GAMMA = 1.913    # ゲージ定数 [1/m^2]
WEIDMANN_RHO_MAX = 5.4    # 停止密度 [1/m^2]


def weidmann_speed(rho, v_f=WEIDMANN_VF, gamma=WEIDMANN_GAMMA, rho_max=WEIDMANN_RHO_MAX):
    """Weidmann(1993)の速度-密度関係 v(ρ)。ρ<=0 は v_f、ρ>=ρ_max は 0 にクリップ。"""
    rho = np.asarray(rho, dtype=np.float64)
    out = np.zeros_like(rho)
    ok = (rho > 1e-9) & (rho < rho_max)
    out[ok] = v_f * (1.0 - np.exp(-gamma * (1.0 / rho[ok] - 1.0 / rho_max)))
    out[rho <= 1e-9] = v_f
    return np.clip(out, 0.0, v_f)


# ─────────────────────────────────────────────────────────────────────────────
# 決定論
# ─────────────────────────────────────────────────────────────────────────────
def array_hash(*arrays):
    """float64 バイト列の SHA-256(同一入力→同一軌跡の「バイト一致」判定に使う)。"""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 基本図(密度-速度)
# ─────────────────────────────────────────────────────────────────────────────
def fd_sample(pos, vel, rect):
    """測定矩形 rect=(x0,y0,x1,y1) 内の (密度 [1/m^2], 空間平均速さ [m/s], 人数)。

    古典的な測定法(measurement area 内の頭数 / 面積、および領域内個体の速さの算術平均)。
    Voronoi 法(Steffen & Seyfried 2010)は使わない=低密度でのばらつきは大きい。"""
    x0, y0, x1, y1 = rect
    m = ((pos[:, 0] >= x0) & (pos[:, 0] < x1) &
         (pos[:, 1] >= y0) & (pos[:, 1] < y1))
    n = int(m.sum())
    area = (x1 - x0) * (y1 - y0)
    if n == 0:
        return 0.0, float("nan"), 0
    speed = float(np.linalg.norm(vel[m], axis=1).mean())
    return n / area, speed, n


# ─────────────────────────────────────────────────────────────────────────────
# レーン形成(対向流)
# ─────────────────────────────────────────────────────────────────────────────
def lane_order(pos, direction, y_lo, y_hi, n_bins=12, x_range=None,
               n_null=20, null_seed=12345):
    """横断方向ビンの方向分離度 φ ∈ [0,1] と、方向ラベルを無作為化した帰無値。

    定義(自前の簡略版・Feliciani&Nishinari2016 の 2D order parameter とは別物):
        横断方向 [y_lo,y_hi] を n_bins 等分。各ビン b の右向き人数 n⁺_b・左向き n⁻_b から
            φ = Σ_b (n⁺_b − n⁻_b)² / n_b  ÷  Σ_b n_b
        完全分離(各ビンが単一方向)で φ=1、各ビンが 50:50 で φ→0。
    有限サイズのため無秩序でも φ>0 になる。方向ラベルを n_null 回シャッフルした
    平均 φ_null を必ず併記し、「φ が φ_null を有意に超えるか」でのみレーン形成を主張する。
    """
    if x_range is not None:
        m = (pos[:, 0] >= x_range[0]) & (pos[:, 0] < x_range[1])
        pos, direction = pos[m], direction[m]
    if pos.shape[0] == 0:
        return float("nan"), float("nan"), 0
    edges = np.linspace(y_lo, y_hi, n_bins + 1)
    b = np.clip(np.digitize(pos[:, 1], edges) - 1, 0, n_bins - 1)

    def _phi(dirs):
        num = 0.0
        tot = 0
        for k in range(n_bins):
            sel = b == k
            nb = int(sel.sum())
            if nb == 0:
                continue
            npos = int((dirs[sel] > 0).sum())
            nneg = nb - npos
            num += (npos - nneg) ** 2 / nb
            tot += nb
        return num / tot if tot else float("nan")

    phi = _phi(direction)
    rng = np.random.default_rng(null_seed)     # 帰無分布専用(シナリオの乱数とは無関係)
    nulls = [_phi(rng.permutation(direction)) for _ in range(n_null)]
    return float(phi), float(np.mean(nulls)), int(pos.shape[0])


def band_count(pos, direction, y_lo, y_hi, n_bins=12):
    """横断ビン列の優勢方向の符号反転数 + 1 = 見かけのレーン本数(空ビンは無視)。"""
    edges = np.linspace(y_lo, y_hi, n_bins + 1)
    b = np.clip(np.digitize(pos[:, 1], edges) - 1, 0, n_bins - 1)
    signs = []
    for k in range(n_bins):
        sel = b == k
        if not sel.any():
            continue
        s = np.sign((direction[sel] > 0).sum() - (direction[sel] <= 0).sum())
        if s != 0:
            signs.append(int(s))
    if not signs:
        return 0
    return 1 + sum(1 for a, c in zip(signs, signs[1:]) if a != c)


# ─────────────────────────────────────────────────────────────────────────────
# 境界縫合性(流入/流出ゲート)と数値健全性
# ─────────────────────────────────────────────────────────────────────────────
def continuity_stats(pos_t, vel_t, wrap_t, dt, gate_mask_fn, active_t=None):
    """軌跡列から「急停止・振動・不連続」の定量指標を返す。

    pos_t/vel_t: (T,N,2)、wrap_t: (T,N) bool(その step でゲート再投入されたか)、
    active_t: (T,N) bool(待機列に退避している個体を除外する)。
    gate_mask_fn(pos)-> (N,) bool: ゲート近傍帯の判定(区間を呼び出し側が決める)。

    返す指標:
      accel_p99 / accel_max : |Δv|/dt の分位点 [m/s^2](急停止・急加速の強さ)
      stall_rate            : |v| < 0.1 m/s の標本割合(立ち往生)
      reversal_rate         : 進行方向成分の符号反転回数 / (体·秒)(振動)
      いずれも gate 近傍 / それ以外(interior)で分けて出す。
      jump_max              : ワープを除いた 1 step 最大変位 [m](瞬間移動の検出)
    """
    T, N = pos_t.shape[0], pos_t.shape[1]
    if active_t is None:
        active_t = np.ones((T, N), dtype=bool)
    dv = np.linalg.norm(np.diff(vel_t, axis=0), axis=2) / dt        # (T-1,N)
    speed = np.linalg.norm(vel_t, axis=2)                           # (T,N)
    # ワープ step / 待機列の個体は Δv を無効化(ゲート再投入自体の不連続は別途 jump で見る)
    valid = (~wrap_t[1:]) & active_t[1:] & active_t[:-1]
    gate = np.zeros((T, N), dtype=bool)
    for t in range(T):
        gate[t] = gate_mask_fn(pos_t[t]) & active_t[t]

    def _agg(sel_dv, sel_sp):
        out = {}
        d = dv[sel_dv]
        out["accel_p99"] = float(np.percentile(d, 99)) if d.size else float("nan")
        out["accel_max"] = float(d.max()) if d.size else float("nan")
        s = speed[sel_sp]
        out["stall_rate"] = float((s < 0.1).mean()) if s.size else float("nan")
        out["mean_speed"] = float(s.mean()) if s.size else float("nan")
        return out

    g_dv = gate[1:] & valid
    i_dv = (~gate[1:]) & valid
    res = {"gate": _agg(g_dv, gate), "interior": _agg(i_dv, (~gate) & active_t)}

    # 振動: x 速度成分の符号反転(ワープ step を跨がない)
    sx = np.sign(vel_t[:, :, 0])
    flip = (sx[1:] * sx[:-1] < 0) & valid
    dur = valid.sum(axis=0) * dt                                    # (N,) 有効時間
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(dur > 0, flip.sum(axis=0) / np.maximum(dur, 1e-9), np.nan)
    res["reversal_rate_per_agent_s"] = float(np.nanmean(rate))
    res["gate_reversal_rate_per_agent_s"] = float(
        (flip & gate[1:]).sum() / max(float((gate[1:] & valid).sum()) * dt, 1e-9))

    step_disp = np.linalg.norm(np.diff(pos_t, axis=0), axis=2)
    step_disp = np.where(valid, step_disp, 0.0)
    res["jump_max_m"] = float(step_disp.max()) if step_disp.size else float("nan")
    return res


def overlap_stats(pos_t, radius, sample_every=10, active_t=None):
    """体の重なり(めり込み)の程度。最小中心間距離と、重なり標本の割合。

    SFM は接触項を持たない(sfm_core の註)ため深い重なりが起きうる。ORCA は
    速度制約なので原理的に重なりにくい。両者の「体の非圧縮性」の差を測る。"""
    worst = np.inf
    over = 0
    tot = 0
    for t in range(0, pos_t.shape[0], sample_every):
        sel = slice(None) if active_t is None else np.nonzero(active_t[t])[0]
        p = pos_t[t][sel]
        r = radius[sel]
        if p.shape[0] < 2:
            continue
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        rr = r[:, None] + r[None, :]
        worst = min(worst, float((d - rr).min()))
        iu = np.triu_indices(p.shape[0], 1)
        over += int((d[iu] < rr[iu]).sum())
        tot += iu[0].size
    return {"min_gap_m": worst, "overlap_pair_rate": over / tot if tot else float("nan")}


# ═════════════════════════════════════════════════════════════════════════════
# P4-1 追加分 — 較正のための測定プロトコルと受入判定
#   (a) RiMEA Test 4 の測定プロトコル(2×2 m 測定区画・先頭 10 s 破棄・複数密度点)
#   (b) ボトルネックの specific flow J/w
#   (c) 実測包絡線との比較(区間内率 + KS 距離)
#   (d) 単調性 dv/dρ ≤ 0 の検査
# 出典:
#   - RiMEA Guideline 4.1.1 (2025-09-11) Test 4 "Measurement of the fundamental
#     diagram": 通路 1000 m × 10 m・2×2 m の測定区画 3 つ(主 1・対照 2)・
#     密度 0.5/1/2/3/4/5/6 P/m²・各密度 60 秒平均・**最初の 10 秒は過渡として無視可**・
#     flow = speed × density。(docs/research/p4-calibration-research.md §5.2)
#   - RiMEA Test 16 "1D Fundamental Diagram": 参照は実試験データの
#     **10%/90% パーセンタイル包絡線**。本ベンチでは RiMEA 配布の XLSX ではなく
#     **Jülich 実測から同じ流儀で作った包絡線**を使う(理由は README の限界節)。
#   - ボトルネック specific flow の参照値 ≈1.9 (m·s)⁻¹(b=0.6–2.5 m):
#     Seyfried et al. 2009 / Rupprecht et al. 2011(同 §2.4)。
#     **本リポジトリが Jülich 実測(b=2.4–5.0 m)から測り直した値は README に併記**。
# ═════════════════════════════════════════════════════════════════════════════

RIMEA_T4_DENSITIES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)   # [1/m^2] RiMEA Test 4 指定
RIMEA_T4_DISCARD_S = 10.0     # 先頭 10 秒は過渡として無視してよい(RiMEA 4.1.1)
RIMEA_T4_AREA_M = 2.0         # 測定区画の一辺 [m]


def measurement_rects(length, width, area_m=RIMEA_T4_AREA_M, n_rects=3):
    """RiMEA Test 4 流の測定区画(主 1 + 対照 2)を通路の中央付近に等間隔で置く。

    通路幅が area_m 未満なら横断方向は通路全幅にする(2×2 m が置けないため)。
    返り値は [(x0,y0,x1,y1), …](先頭が主区画)。"""
    h = area_m / 2.0
    y_c = width / 2.0
    y0, y1 = ((max(0.0, y_c - h), min(width, y_c + h)) if width >= area_m
              else (0.0, width))
    xs = [length / 2.0]
    for k in range(1, n_rects):
        off = (k + 1) // 2 * (length / (n_rects + 1))
        xs.append(length / 2.0 + (off if k % 2 else -off))
    return [(x - h, y0, x + h, y1) for x in xs]


def rimea_test4_fd(trace, rects, discard_s=RIMEA_T4_DISCARD_S, measure_s=None):
    """RiMEA Test 4 の測定プロトコルで基本図 1 点を測る。

    trace: scenarios.Trace(pos (T,N,2) / vel (T,N,2) / dt / active (T,N))。
    rects: measurement_rects() の出力(主区画 = rects[0]、残りは対照)。

    先頭 discard_s 秒を捨て、以後 measure_s 秒(None なら残り全部)を平均する。
    各 step で「区画内の頭数 / 面積」= 瞬間密度、「区画内個体の前進速度(x 成分)の
    算術平均」= 瞬間速度。時間平均は step 等重み(区画が空の step は密度 0 として
    数え、速度平均からは除く)。flow = density × speed(RiMEA 4.1.1 の定義)。

    返り値: {"rho","v","speed_abs","flow","n_steps","occupancy",
             "controls":[…], "spread_rho","spread_v"}
      spread_* は主区画と対照区画の食い違い(測定区画の置き方への感度)。"""
    dt = trace.dt
    T = trace.pos.shape[0]
    t0 = min(T - 1, int(round(discard_s / dt)))
    t1 = T if measure_s is None else min(T, t0 + int(round(measure_s / dt)))
    act = getattr(trace, "active", None)

    def _one(rect):
        x0, y0, x1, y1 = rect
        area = (x1 - x0) * (y1 - y0)
        rho_t, v_t, sp_t, occ = [], [], [], 0
        for t in range(t0, t1):
            p, v = trace.pos[t], trace.vel[t]
            m = ((p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1))
            if act is not None:
                m = m & act[t]
            n = int(m.sum())
            rho_t.append(n / area)
            if n:
                occ += 1
                v_t.append(float(v[m, 0].mean()))
                sp_t.append(float(np.linalg.norm(v[m], axis=1).mean()))
        rho = float(np.mean(rho_t)) if rho_t else float("nan")
        vv = float(np.mean(v_t)) if v_t else float("nan")
        sp = float(np.mean(sp_t)) if sp_t else float("nan")
        return {"rho": rho, "v": vv, "speed_abs": sp, "flow": rho * vv,
                "n_steps": len(rho_t), "occupancy": occ / max(1, len(rho_t))}

    main = _one(rects[0])
    ctrl = [_one(r) for r in rects[1:]]
    out = dict(main)
    out["controls"] = ctrl
    if ctrl:
        rr = [c["rho"] for c in ctrl] + [main["rho"]]
        vv = [c["v"] for c in ctrl] + [main["v"]]
        out["spread_rho"] = float(np.nanmax(rr) - np.nanmin(rr))
        out["spread_v"] = float(np.nanmax(vv) - np.nanmin(vv))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# (b) ボトルネックの specific flow
# ─────────────────────────────────────────────────────────────────────────────
def specific_flow(trace, line_x, width, trim=0.10, direction=+1):
    """開口線 x=line_x の通過時刻列から J [1/s] と specific flow J/w [1/(m·s)]。

    各個体の **最初の通過時刻** を線形内挿で求め、通過ランク順の中央 (1−2·trim) を
    定常部として J = (n−1)/(t_last − t_first) を作る(Jülich 実測側 juelich.py の
    bottleneck_flow と同一手順 = シムと実測を同じ物差しで測る)。
    width は開口幅 [m](= specific flow の分母)。"""
    P = trace.pos
    T, N = P.shape[0], P.shape[1]
    act = getattr(trace, "active", None)
    if act is None:
        act = np.ones((T, N), dtype=bool)
    x = P[:, :, 0]
    if direction < 0:
        x = -x
        line_x = -line_x
    times = []
    for i in range(N):
        col = x[:, i]
        ok = act[:, i]
        cross = np.flatnonzero((col[:-1] < line_x) & (col[1:] >= line_x)
                               & ok[:-1] & ok[1:])
        if cross.size == 0:
            continue
        k = int(cross[0])
        a, b = col[k], col[k + 1]
        frac = 0.0 if b == a else (line_x - a) / (b - a)
        times.append((k + frac) * trace.dt)
    times = np.sort(np.array(times, dtype=np.float64))
    if times.size < 5:
        return {"n_cross": int(times.size), "J": float("nan"),
                "J_specific": float("nan"), "duration_s": float("nan")}
    lo = int(round(trim * times.size))
    hi = int(round((1.0 - trim) * times.size))
    sel = times[lo:hi]
    dur = float(sel[-1] - sel[0]) if sel.size >= 2 else 0.0
    if sel.size < 2 or dur <= 0:
        return {"n_cross": int(times.size), "J": float("nan"),
                "J_specific": float("nan"), "duration_s": dur}
    j = (sel.size - 1) / dur
    return {"n_cross": int(times.size), "n_steady": int(sel.size),
            "duration_s": dur, "J": j, "J_specific": j / float(width),
            "t_first": float(times[0]), "t_last": float(times[-1])}


def linear_fit_r2(x, y):
    """最小二乗の直線当てはめ → (slope, intercept, R²)。J(b) の線形性検査用
    (研究文書 §2.4: 流量は幅に対して段階状ではなく線形に増える)。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([x, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ sol
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(sol[0]), float(sol[1]), r2


# ─────────────────────────────────────────────────────────────────────────────
# (c) 実測包絡線との比較(区間内率 + KS 距離)
# ─────────────────────────────────────────────────────────────────────────────
def envelope_coverage(v_sim, lo, hi):
    """各密度点で v_sim が [lo, hi](実測 10/90 パーセンタイル)に入る割合。

    lo/hi が NaN の点(参照データが無い密度)は判定から除く。
    返り値: {"inside_rate","n_judged","below","above","details"}"""
    v = np.atleast_1d(np.asarray(v_sim, dtype=np.float64))
    lo = np.atleast_1d(np.asarray(lo, dtype=np.float64))
    hi = np.atleast_1d(np.asarray(hi, dtype=np.float64))
    ok = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return {"inside_rate": float("nan"), "n_judged": 0,
                "below": 0, "above": 0, "details": []}
    inside = (v[ok] >= lo[ok]) & (v[ok] <= hi[ok])
    return {"inside_rate": float(inside.mean()), "n_judged": int(ok.sum()),
            "below": int((v[ok] < lo[ok]).sum()),
            "above": int((v[ok] > hi[ok]).sum()),
            "details": [bool(b) for b in inside]}


def ks_two_sample(a, b):
    """2 標本 Kolmogorov–Smirnov: (D, p)。scipy 非依存の自前実装。

    D = sup|F_a − F_b|。p は漸近式
        Q(λ) = 2 Σ_{j≥1} (−1)^{j−1} exp(−2 j² λ²),
        λ = (√n_e + 0.12 + 0.11/√n_e) · D,   n_e = n_a·n_b/(n_a+n_b)
    (Numerical Recipes の標準近似)。**小標本では近似**である旨は README に明記。
    RiMEA Test 16 が KS 検定で突合する流儀(SingleFileComparator)に合わせた検収用。"""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, allv, side="right") / n1
    cdf2 = np.searchsorted(b, allv, side="right") / n2
    d = float(np.abs(cdf1 - cdf2).max())
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    p, sign = 0.0, 1.0
    for j in range(1, 101):
        term = float(np.exp(-2.0 * (j * lam) ** 2))
        p += sign * term
        sign = -sign
        if term < 1e-12:
            break
    return d, float(min(1.0, max(0.0, 2.0 * p)))


# ─────────────────────────────────────────────────────────────────────────────
# (d) 単調性 dv/dρ ≤ 0
# ─────────────────────────────────────────────────────────────────────────────
def monotonicity(rho, v, eps=0.0):
    """密度掃引の非単調性を測る。Weidmann の 4 領域はすべて非増加(研究文書 §2.1)。

    返り値: {"violations": 違反した隣接ペア数, "max_increase": 最大の増分 [m/s],
             "worst_pair": (ρ_k, ρ_{k+1}), "ok": bool}
    eps は許容(seed 間 sd などを入れる)。"""
    rho = np.asarray(rho, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    o = np.argsort(rho)
    rho, v = rho[o], v[o]
    ok = np.isfinite(v)
    rho, v = rho[ok], v[ok]
    if v.size < 2:
        return {"violations": 0, "max_increase": 0.0, "worst_pair": None, "ok": True}
    inc = v[1:] - v[:-1]
    bad = inc > eps
    k = int(np.argmax(inc))
    return {"violations": int(bad.sum()), "max_increase": float(inc.max()),
            "worst_pair": (float(rho[k]), float(rho[k + 1])),
            "ok": bool(not bad.any())}


def band_check(v_sim, v_ref, tol=0.20):
    """判定 A: |v_sim − v_ref| / v_ref ≤ tol(既定 ±20%・研究文書 §2.3)。"""
    v = np.atleast_1d(np.asarray(v_sim, dtype=np.float64))
    r = np.atleast_1d(np.asarray(v_ref, dtype=np.float64))
    ok = np.isfinite(v) & np.isfinite(r) & (np.abs(r) > 1e-9)
    if not ok.any():
        return {"pass_rate": float("nan"), "n_judged": 0, "rel_err": []}
    rel = np.full(v.shape, np.nan)
    rel[ok] = (v[ok] - r[ok]) / r[ok]
    return {"pass_rate": float((np.abs(rel[ok]) <= tol).mean()),
            "n_judged": int(ok.sum()),
            "rel_err": [None if not np.isfinite(x) else float(x) for x in rel],
            "max_abs_rel_err": float(np.abs(rel[ok]).max())}


# ─────────────────────────────────────────────────────────────────────────────
# 決定論的 Nelder-Mead(scipy 非依存)と Kladek 形の当てはめ
# ─────────────────────────────────────────────────────────────────────────────
def nelder_mead(fn, x0, step=None, max_iter=400, tol=1e-9, max_eval=None):
    """決定論的 Nelder-Mead。返り値 (x_best, f_best)。

    標準係数(反射 1.0 / 拡大 2.0 / 収縮 0.5 / 縮小 0.5)。乱数を一切引かない。
    同値のときの並べ替えは安定ソート = 同一入力で同一結果。
    max_eval を与えると目的関数の評価回数で打ち切る(計算時間の上限管理)。"""
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    if step is None:
        step = np.where(np.abs(x0) > 1e-12, 0.10 * np.abs(x0), 0.05)
    step = np.broadcast_to(np.asarray(step, dtype=np.float64), (n,)).copy()
    n_eval = [0]

    def _f(p):
        n_eval[0] += 1
        return fn(p)

    def _budget():
        return max_eval is not None and n_eval[0] >= max_eval

    sim = np.vstack([x0] + [x0 + np.eye(n)[k] * step[k] for k in range(n)])
    fv = np.array([_f(p) for p in sim], dtype=np.float64)
    for _ in range(max_iter):
        if _budget():
            break
        o = np.argsort(fv, kind="stable")
        sim, fv = sim[o], fv[o]
        if abs(fv[-1] - fv[0]) <= tol * (abs(fv[0]) + tol):
            break
        cen = sim[:-1].mean(axis=0)
        xr = cen + 1.0 * (cen - sim[-1])
        fr = _f(xr)
        if fr < fv[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            fe = _f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)
            fc = _f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                for k in range(1, n + 1):
                    fv[k] = _f(sim[k])
    o = np.argsort(fv, kind="stable")
    return sim[o][0], float(fv[o][0])


def fit_kladek(rho, v, weights=None, x0=(1.34, 1.913, 5.4)):
    """実測 (ρ, v) に Kladek 式を当てはめて (v_f, γ, ρ_max) を返す。

    Jülich 実測から「なめらかな参照曲線」を作るための当てはめ。
    scipy 非依存(上の nelder_mead)。重み付き二乗誤差を最小化する。"""
    rho = np.asarray(rho, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)

    def loss(p):
        vf, g, rm = p
        if vf <= 0 or g <= 0 or rm <= rho.max() * 1.001:
            return 1e9
        pred = weidmann_speed(rho, v_f=vf, gamma=g, rho_max=rm)
        return float((w * (pred - v) ** 2).sum() / w.sum())

    best, _ = nelder_mead(loss, np.array(x0, dtype=np.float64))
    return tuple(float(x) for x in best)
