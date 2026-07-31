"""比較対象 2 候補の共通ラッパ(同一 API で同一シナリオを回すため)。

  (a) SFM  … 自前資産 src/society/world/sfm_core.Crowd を拡張した
             indoor_flow.WallCrowd を **読み取り利用のみ** で薄く包む。
             src/ は 1 バイトも変更しない(import して使うだけ)。
  (b) ORCA … reference/physics_bench/orca_min.OrcaCrowd(本ディレクトリ内の自前最小実装)。

共通 API:
    eng.pos    (N,2) float64   位置 [m]
    eng.vel    (N,2) float64   速度 [m/s]
    eng.goal   (N,2) float64   目標点 [m](書き換え可)
    eng.v0     (N,)            希望速度 [m/s]
    eng.radius (N,)            半径 [m]
    eng.set_ghosts(pos, vel, radius)   周期境界のゴースト(斥力源のみ)
    eng.step(dt)                       1 サブステップ前進

公平性のため両者で揃えた条件:
    - 半径 radius・希望速度 v0 は同一(indoor_flow の agent_id 安定ハッシュ由来)
    - 近傍上限 neighbor_cap = 12(SFM 側の既定値に合わせる)
    - 最大速度 = 1.3 * v0(Helbing2000 の v_max。ORCA 側もこの円で LP を解く)
    - 壁は同一の線分集合
"""
from __future__ import annotations

import os
import sys

import numpy as np

# src/ を import path に載せる(pyproject の [tool.pytest.ini_options] pythonpath=["src"] と同じ)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from society.world import indoor_flow          # noqa: E402  (読み取り利用のみ)
from society.world.sfm_core import V_MAX_FACTOR  # noqa: E402

from . import orca_min                          # noqa: E402

NEIGHBOR_CAP = indoor_flow.NEIGHBOR_CAP          # 12
# ORCA の近傍探索半径 [m]。SFM 側の実効到達距離(斥力カットオフ rr+2.0 ≈ 2.6m)より広く取る
# (ORCA は tau=2s 先の速度障害物を見るので数 m 先の相手が効く)。周期境界のゴースト帯幅も
# この値以上にする(GHOST_RANGE)。
NEIGHBOR_DIST = 6.0
GHOST_RANGE = 6.5


class SfmEngine:
    """自前 SFM(WallCrowd)の薄いラッパ。

    WallCrowd は「pos/vel/goal から 1 ステップ進める」だけの状態機械なので、
    ゴースト(周期境界の像)は毎ステップ実体+ゴーストの結合配列で作り直す。
    ゴーストは frozen=True(斥力源だが動かない)。SFM の力は相手の速度に依存しない
    (駆動項は自分の速度のみ、斥力は位置のみ)ので、frozen ゴーストで周期境界は厳密。
    """

    name = "sfm"

    def __init__(self, pos, vel, goal, v0, radius, walls=None,
                 neighbor_cap=NEIGHBOR_CAP, solver_kw=None, **kw):
        # solver_kw は Crowd/WallCrowd のコンストラクタ引数(a, b, tau, mass, lambda_aniso,
        # wall_a, wall_b, noise …)へそのまま渡す。src/ を書き換えずに較正感度を測るための唯一の口。
        # noise>0 を使うときは noise_seed から専用 Generator を1個作って毎ステップ引き継ぐ
        # (= 同一 seed なら同一軌跡。決定論は seed 固定で保たれる)。
        self.solver_kw = dict(solver_kw or {})
        seed = self.solver_kw.pop("noise_seed", None)
        if self.solver_kw.get("noise", 0.0) and "rng" not in self.solver_kw:
            self.solver_kw["rng"] = np.random.default_rng(0 if seed is None else seed)
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2).copy()
        self.vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2).copy()
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2).copy()
        self.v0 = np.asarray(v0, dtype=np.float64).reshape(-1).copy()
        self.radius = np.asarray(radius, dtype=np.float64).reshape(-1).copy()
        self.walls = list(walls or [])
        self.neighbor_cap = int(neighbor_cap)
        self.ghost_pos = np.zeros((0, 2))
        self.ghost_vel = np.zeros((0, 2))
        self.ghost_radius = np.zeros(0)

    def set_ghosts(self, gpos, gvel, gradius):
        self.ghost_pos = np.asarray(gpos, dtype=np.float64).reshape(-1, 2)
        self.ghost_vel = np.asarray(gvel, dtype=np.float64).reshape(-1, 2)
        self.ghost_radius = np.asarray(gradius, dtype=np.float64).reshape(-1)

    def step(self, dt=0.1):
        n = self.pos.shape[0]
        g = self.ghost_pos.shape[0]
        if g:
            pos = np.concatenate([self.pos, self.ghost_pos], axis=0)
            vel = np.concatenate([self.vel, np.zeros((g, 2))], axis=0)
            goal = np.concatenate([self.goal, self.ghost_pos], axis=0)   # dir=0
            v0 = np.concatenate([self.v0, np.ones(g)], axis=0)
            rad = np.concatenate([self.radius, self.ghost_radius], axis=0)
            frozen = np.concatenate([np.zeros(n, bool), np.ones(g, bool)])
        else:
            pos, vel, goal = self.pos, self.vel, self.goal
            v0, rad = self.v0, self.radius
            frozen = np.zeros(n, bool)

        crowd = indoor_flow.WallCrowd(
            pos=pos, vel=vel, goal=goal, v0=v0, radius=rad,
            walls=self.walls or None, frozen=frozen,
            neighbor_cap=self.neighbor_cap, **self.solver_kw)
        crowd.step(dt)
        self.pos = crowd.pos[:n].copy()
        self.vel = crowd.vel[:n].copy()
        return self.pos


class OrcaEngine:
    """自前最小 ORCA のラッパ(API を SfmEngine に揃えるだけ)。"""

    name = "orca"

    def __init__(self, pos, vel, goal, v0, radius, walls=None,
                 neighbor_cap=NEIGHBOR_CAP, tau=2.0, tau_obst=2.0,
                 neighbor_dist=NEIGHBOR_DIST, solver_kw=None, **kw):
        sk = dict(solver_kw or {})
        radius = np.asarray(radius, dtype=np.float64) * float(sk.pop("radius_scale", 1.0))
        seed = sk.pop("noise_seed", None)
        if sk.get("pref_noise", 0.0) and "rng" not in sk:
            sk["rng"] = np.random.default_rng(0 if seed is None else seed)
        self._c = orca_min.OrcaCrowd(pos=pos, vel=vel, goal=goal, v0=v0, radius=radius,
                                     walls=walls, neighbor_cap=neighbor_cap,
                                     tau=tau, tau_obst=tau_obst,
                                     neighbor_dist=neighbor_dist,
                                     v_max_factor=V_MAX_FACTOR, **sk)

    # pos/vel/goal は下位オブジェクトへ委譲(シナリオ側が in-place 書換できるように)
    @property
    def pos(self):
        return self._c.pos

    @pos.setter
    def pos(self, v):
        self._c.pos = v

    @property
    def vel(self):
        return self._c.vel

    @vel.setter
    def vel(self, v):
        self._c.vel = v

    @property
    def goal(self):
        return self._c.goal

    @goal.setter
    def goal(self, v):
        self._c.goal = v

    @property
    def v0(self):
        return self._c.v0

    @property
    def radius(self):
        return self._c.radius

    def set_ghosts(self, gpos, gvel, gradius):
        self._c.set_ghosts(gpos, gvel, gradius)

    def step(self, dt=0.1):
        return self._c.step(dt)


ENGINES = {"sfm": SfmEngine, "orca": OrcaEngine}


def make_engine(kind, **kw):
    return ENGINES[kind](**kw)


def agent_params(ids):
    """agent_id 列 → (v0, radius)。src/society/world/indoor_flow の安定ハッシュを流用。

    v0 ∈ [1.0, 1.4) m/s、radius ∈ [0.25, 0.35) m。乱数 stream を一切引かない=決定論。"""
    v0 = np.array([indoor_flow.desired_speed(i) for i in ids], dtype=np.float64)
    rad = np.array([indoor_flow.body_radius(i) for i in ids], dtype=np.float64)
    return v0, rad
