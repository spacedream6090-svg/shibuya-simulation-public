"""比較シナリオ — 両候補を **同一の初期条件・同一の駆動規則** で回す。

  A. fd_periodic  : 周期境界の一方向通路(周期像=ゴーストで厳密な周期性)。
                    密度を振って基本図(密度-速度)を測り Weidmann(1993) と重ねる。
  B. counterflow  : 幅員違い(3m / 6m)の通路での対向流 200 体。レーン形成を観察。
  C. crossing     : 開放正方領域の4方向交差流(スクランブル風)200 体。
  D. gate_stress  : B と同じ通路で流入/流出ゲートの規則を変えて境界縫合性を測る。

【ゲート(流入/流出)の 2 方式】
  blind   : 出口を越えた個体を即座に入口へ再投入(占有チェックなし)。
            = P3 の素朴実装。重なり生成の有無を測るための対照。
  guarded : 出口を越えた個体は「待機列」へ退避(遠方の駐機スロットに置き、
            力学から切り離す)。入口候補位置が空いている個体だけを id 昇順で入場させ、
            **退場時の速度をそのまま復元する**(P3 の「初期速度をグラフ側から連続に引き継ぐ」に対応)。

【決定論】
  初期配置のみ np.random.default_rng(seed) を使い(シナリオ生成時の 1 回だけ)、
  積分ループ内では乱数を一切引かない。ゲートの入場判定は id 昇順の逐次判定=順序固定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engines import GHOST_RANGE, NEIGHBOR_DIST, agent_params, make_engine

PARK_X = 0.0            # 待機列(駐機)スロット。本体から十分遠く、互いに NEIGHBOR_DIST 以上離す。
PARK_Y = -1000.0
PARK_PITCH = 20.0
GOAL_AHEAD = 50.0       # 目標点を進行方向 50m 先に置く=希望方向が厳密に単位方向ベクトルになる


@dataclass
class Trace:
    """1 ラン分の記録。決定論ハッシュと全指標はここから計算する。"""
    pos: np.ndarray            # (T,N,2)
    vel: np.ndarray            # (T,N,2)
    wrap: np.ndarray           # (T,N) bool  その step でゲート再投入/退場した
    active: np.ndarray         # (T,N) bool  領域内にいる(待機列でない)
    dt: float
    meta: dict = field(default_factory=dict)


def _corridor_walls(length, width, pad=10.0):
    """通路の 2 本の壁(y=0 と y=width)。x はゲート外側まで延長する。"""
    return [((-pad, 0.0), (length + pad, 0.0)),
            ((-pad, width), (length + pad, width))]


def _init_positions(rng, n, x_lo, x_hi, y_lo, y_hi, radius, tries=400):
    """重なりなしの初期配置(棄却サンプリング・rng は初期化時のみ消費)。"""
    pos = np.zeros((n, 2))
    for i in range(n):
        for _ in range(tries):
            c = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
            if i == 0:
                break
            d = np.linalg.norm(pos[:i] - c, axis=1)
            if (d > radius[:i] + radius[i] + 0.05).all():
                break
        pos[i] = c
    return pos


def _park_slot(i):
    return np.array([PARK_X + i * PARK_PITCH, PARK_Y])


# ─────────────────────────────────────────────────────────────────────────────
# A. 周期境界の一方向通路(基本図の測定)
# ─────────────────────────────────────────────────────────────────────────────
def run_fd_periodic(engine_kind, n_agents, length=20.0, width=3.0, dt=0.1,
                    t_total=50.0, seed=20260801, record_every=1, engine_kw=None):
    ids = list(range(n_agents))
    v0, radius = agent_params(ids)
    rng = np.random.default_rng(seed)
    pos = _init_positions(rng, n_agents, 0.2, length - 0.2, 0.4, width - 0.4, radius)
    vel = np.zeros((n_agents, 2))
    vel[:, 0] = v0                       # 初速=希望速度(過渡を短く)
    goal = pos + np.array([GOAL_AHEAD, 0.0])

    eng = make_engine(engine_kind, pos=pos, vel=vel, goal=goal, v0=v0, radius=radius,
                      walls=_corridor_walls(length, width), neighbor_dist=NEIGHBOR_DIST,
                      **(engine_kw or {}))

    n_steps = int(round(t_total / dt))
    P = np.zeros((n_steps + 1, n_agents, 2))
    V = np.zeros((n_steps + 1, n_agents, 2))
    W = np.zeros((n_steps + 1, n_agents), dtype=bool)
    P[0], V[0] = eng.pos, eng.vel

    for t in range(1, n_steps + 1):
        # 目標は常に真正面 50m 先(希望方向 = +x を厳密に保つ)
        eng.goal[:, 0] = eng.pos[:, 0] + GOAL_AHEAD
        eng.goal[:, 1] = eng.pos[:, 1]
        # 周期像(ゴースト): 端から GHOST_RANGE 以内の個体を ±length にコピー
        left = eng.pos[:, 0] < GHOST_RANGE
        right = eng.pos[:, 0] > length - GHOST_RANGE
        gp = np.concatenate([eng.pos[left] + [length, 0.0],
                             eng.pos[right] - [length, 0.0]], axis=0)
        gv = np.concatenate([eng.vel[left], eng.vel[right]], axis=0)
        gr = np.concatenate([eng.radius[left], eng.radius[right]], axis=0)
        eng.set_ghosts(gp, gv, gr)

        eng.step(dt)

        wrapped = eng.pos[:, 0] >= length
        if wrapped.any():
            eng.pos[wrapped, 0] -= length            # 周期折返し(ゴーストがあるので力学は連続)
        W[t] = wrapped
        P[t], V[t] = eng.pos, eng.vel

    active = np.ones_like(W)
    return Trace(pos=P[::record_every], vel=V[::record_every], wrap=W[::record_every],
                 active=active[::record_every], dt=dt * record_every,
                 meta={"scenario": "fd_periodic", "engine": engine_kind,
                       "n_agents": n_agents, "length": length, "width": width,
                       "area_m2": length * width,
                       "global_density": n_agents / (length * width),
                       "mean_v0": float(v0.mean())})


# ─────────────────────────────────────────────────────────────────────────────
# ゲート付き通路の共通ループ(B / D で使う)
# ─────────────────────────────────────────────────────────────────────────────
def _gate_admit(eng, parked, park_vel, entry_x, entry_y, dir_sign, min_gap=0.10,
                guarded=True):
    """待機列の個体を id 昇順で入場させる。guarded=False なら占有チェックなしで即入場。"""
    admitted = np.zeros(parked.shape[0], dtype=bool)
    idx = np.nonzero(parked)[0]
    for i in idx:
        cand = np.array([entry_x[i], entry_y[i]])
        if guarded:
            others = (~parked) | admitted
            others[i] = False
            if others.any():
                d = np.linalg.norm(eng.pos[others] - cand, axis=1)
                if (d < eng.radius[others] + eng.radius[i] + min_gap).any():
                    continue
        eng.pos[i] = cand
        eng.vel[i] = park_vel[i]
        admitted[i] = True
    parked[admitted] = False
    return admitted


def run_counterflow(engine_kind, width=3.0, n_agents=200, length=30.0, dt=0.1,
                    t_total=60.0, seed=20260801, gate_mode="guarded", engine_kw=None):
    ids = list(range(n_agents))
    v0, radius = agent_params(ids)
    rng = np.random.default_rng(seed)
    pos = _init_positions(rng, n_agents, 0.5, length - 0.5, 0.4, width - 0.4, radius)
    sign = np.where(np.arange(n_agents) % 2 == 0, 1.0, -1.0)   # 偶数=右向き / 奇数=左向き
    vel = np.zeros((n_agents, 2))
    vel[:, 0] = sign * v0
    goal = pos + np.stack([sign * GOAL_AHEAD, np.zeros(n_agents)], axis=1)

    eng = make_engine(engine_kind, pos=pos, vel=vel, goal=goal, v0=v0, radius=radius,
                      walls=_corridor_walls(length, width), neighbor_dist=NEIGHBOR_DIST,
                      **(engine_kw or {}))

    guarded = (gate_mode == "guarded")
    parked = np.zeros(n_agents, dtype=bool)
    park_vel = np.zeros((n_agents, 2))
    entry_y = pos[:, 1].copy()
    entry_x = np.where(sign > 0, 0.0, length)

    n_steps = int(round(t_total / dt))
    P = np.zeros((n_steps + 1, n_agents, 2))
    V = np.zeros((n_steps + 1, n_agents, 2))
    W = np.zeros((n_steps + 1, n_agents), dtype=bool)
    A = np.ones((n_steps + 1, n_agents), dtype=bool)
    P[0], V[0] = eng.pos.copy(), eng.vel.copy()
    wait_steps = np.zeros(n_agents)

    for t in range(1, n_steps + 1):
        adm = _gate_admit(eng, parked, park_vel, entry_x, entry_y, sign, guarded=guarded)
        eng.goal[:, 0] = eng.pos[:, 0] + sign * GOAL_AHEAD
        eng.goal[:, 1] = eng.pos[:, 1]
        eng.step(dt)

        # 出口通過 → 待機列へ(速度を保存して駐機スロットへ退避)
        out = (~parked) & (((sign > 0) & (eng.pos[:, 0] > length)) |
                           ((sign < 0) & (eng.pos[:, 0] < 0.0)))
        for i in np.nonzero(out)[0]:
            park_vel[i] = eng.vel[i]
            eng.pos[i] = _park_slot(i)
            eng.vel[i] = 0.0
        parked |= out
        # 駐機個体は毎ステップ位置・速度を固定(力学から切り離す)
        for i in np.nonzero(parked)[0]:
            eng.pos[i] = _park_slot(i)
            eng.vel[i] = 0.0
        wait_steps += parked

        W[t] = out | adm
        A[t] = ~parked
        P[t], V[t] = eng.pos.copy(), eng.vel.copy()

    return Trace(pos=P, vel=V, wrap=W, active=A, dt=dt,
                 meta={"scenario": "counterflow", "engine": engine_kind,
                       "n_agents": n_agents, "length": length, "width": width,
                       "gate_mode": gate_mode, "dir_sign": sign.tolist(),
                       "mean_wait_s": float(wait_steps.mean() * dt),
                       "mean_v0": float(v0.mean())})


# ─────────────────────────────────────────────────────────────────────────────
# C. 4方向交差流(スクランブル風・開放領域=壁なし)
# ─────────────────────────────────────────────────────────────────────────────
def run_crossing(engine_kind, n_agents=200, size=20.0, band=6.0, dt=0.1,
                 t_total=60.0, seed=20260801, gate_mode="guarded", engine_kw=None):
    ids = list(range(n_agents))
    v0, radius = agent_params(ids)
    rng = np.random.default_rng(seed)
    lo, hi = (size - band) / 2.0, (size + band) / 2.0

    stream = np.arange(n_agents) % 4          # 0:→ 1:← 2:↑ 3:↓
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])[stream]
    pos = np.zeros((n_agents, 2))
    for i in range(n_agents):
        for _ in range(400):
            along = rng.uniform(0.5, size - 0.5)
            across = rng.uniform(lo + 0.3, hi - 0.3)
            c = np.array([along, across]) if stream[i] < 2 else np.array([across, along])
            if i == 0:
                break
            if (np.linalg.norm(pos[:i] - c, axis=1) > radius[:i] + radius[i] + 0.05).all():
                break
        pos[i] = c
    vel = dirs * v0[:, None]
    goal = pos + dirs * GOAL_AHEAD

    eng = make_engine(engine_kind, pos=pos, vel=vel, goal=goal, v0=v0, radius=radius,
                      walls=None, neighbor_dist=NEIGHBOR_DIST, **(engine_kw or {}))

    guarded = (gate_mode == "guarded")
    parked = np.zeros(n_agents, dtype=bool)
    park_vel = np.zeros((n_agents, 2))
    # 入口は各ストリームの上流端。横断座標は退場時の値を引き継ぐ(帯の中で連続)。
    entry_along = np.where(stream == 0, 0.0,
                  np.where(stream == 1, size,
                  np.where(stream == 2, 0.0, size)))
    entry_across = np.where(stream < 2, pos[:, 1], pos[:, 0]).copy()

    n_steps = int(round(t_total / dt))
    P = np.zeros((n_steps + 1, n_agents, 2))
    V = np.zeros((n_steps + 1, n_agents, 2))
    W = np.zeros((n_steps + 1, n_agents), dtype=bool)
    A = np.ones((n_steps + 1, n_agents), dtype=bool)
    P[0], V[0] = eng.pos.copy(), eng.vel.copy()

    def _entry_xy():
        ex = np.where(stream < 2, entry_along, entry_across)
        ey = np.where(stream < 2, entry_across, entry_along)
        return ex, ey

    for t in range(1, n_steps + 1):
        ex, ey = _entry_xy()
        adm = _gate_admit(eng, parked, park_vel, ex, ey, None, guarded=guarded)
        eng.goal[:] = eng.pos + dirs * GOAL_AHEAD
        eng.step(dt)

        along_now = np.where(stream < 2, eng.pos[:, 0], eng.pos[:, 1])
        fwd = np.where((stream == 0) | (stream == 2), 1.0, -1.0)
        out = (~parked) & (((fwd > 0) & (along_now > size)) | ((fwd < 0) & (along_now < 0.0)))
        for i in np.nonzero(out)[0]:
            park_vel[i] = eng.vel[i]
            entry_across[i] = eng.pos[i, 1] if stream[i] < 2 else eng.pos[i, 0]
            eng.pos[i] = _park_slot(i)
            eng.vel[i] = 0.0
        parked |= out
        for i in np.nonzero(parked)[0]:
            eng.pos[i] = _park_slot(i)
            eng.vel[i] = 0.0

        W[t] = out | adm
        A[t] = ~parked
        P[t], V[t] = eng.pos.copy(), eng.vel.copy()

    return Trace(pos=P, vel=V, wrap=W, active=A, dt=dt,
                 meta={"scenario": "crossing", "engine": engine_kind,
                       "n_agents": n_agents, "size": size, "band": band,
                       "gate_mode": gate_mode, "stream": stream.tolist(),
                       "mean_v0": float(v0.mean())})
