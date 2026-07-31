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
