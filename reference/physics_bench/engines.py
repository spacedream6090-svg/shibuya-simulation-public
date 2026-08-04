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
from society.world.sfm_core import (CUTOFF_M, V_MAX_FACTOR,     # noqa: E402
                                    _EXP_ARG_MAX)

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


# ═════════════════════════════════════════════════════════════════════════════
# P4-1: 拡張 SFM(**ベンチ内に閉じる**。src/society/world/sfm_core.py は 1 バイトも
#       変更しない。ここでの実験結果を見て、P4-2 で本体に conf ゲート付きで入れる)
#
# 追加するもの(docs/research/p4-calibration-research.md §6.1 の発見に対応):
#   (i)   **第 2 項 = 長距離 social 項**。VISSIM 製品版 SFM は 2 項構造で、
#         短距離・等方 (A=25 m/s², B=0.2 m) と 長距離・異方 (A=0.5 m/s², B=2.8 m)
#         を持つ[文献: Kretz, Hengst & Vortisch, arXiv:0805.1788 の P0 表]。
#         本リポジトリの既定 A=2000 N ÷ m=80 kg = 25 m/s² は **短距離項と一致**し、
#         長距離項が丸ごと欠けている(研究文書 §6.1)。
#         → f2_ij = m · A2 · exp((r_i+r_j − d)/B2) · w2(φ) · n_ij
#           ★ A2 の単位は **m/s²**(力ではなく加速度)。VISSIM の表記に合わせる。
#           ★ 指数の基準点は本リポジトリの短距離項と同じ **体表間隔**(r_i+r_j−d)。
#             VISSIM の定義式は論文に明示されていないので等価性は主張しない
#             (研究文書 §8-7 の限界)。
#   (ii)  λ(異方性)を engine_kw で可変化。本体既定 0.5 に対し較正済み文献値は
#         0.06–0.12(VISSIM 0.1 / Johansson 円形 0.12 / 楕円 0.06)。
#         短距離項の λ は Crowd の `lambda_aniso` をそのまま使い、長距離項には
#         独立の `lambda_far`(既定 = lambda_aniso)を持たせる。
#   (iii) **CUTOFF_M の不連続対策**(研究文書 §6.1b)。B2=2.8 m の項を
#         体表間隔 2.0 m で垂直に切ると exp(−2.0/2.8)=0.49 = 接触時の 49% が
#         残ったまま断ち切られ、Köster 2013 の言う「右辺の不連続」を作り込む。
#         → `far_cutoff`(項別カットオフ)+ `far_taper`(C¹ smoothstep テーパー)。
#
# 【後方互換の保証】a2 <= 0 のとき _far_forces は 1 度も評価されず、forces() は
#   `super().forces()` をそのまま返す = 既存 SfmEngine とバイト一致。
#   この不変条件は verify_extended_matches_base() が実測で確認する。
# ═════════════════════════════════════════════════════════════════════════════
FAR_A_DEFAULT = 0.0       # A2 [m/s²]。0 = 長距離項 OFF(= 既存挙動とバイト一致)
FAR_B_DEFAULT = 2.8       # B2 [m]。VISSIM P0 の B_social
# 項別カットオフ = FAR_CUTOFF_FACTOR·B2。2.5 で exp(−2.5)=8.2% が残る位置で切り、
# その手前 FAR_TAPER_DEFAULT [m] を C¹ smoothstep で 0 へ落とす。
# ★ここは **モデル定義の一部**。較正で得る A2 はこの打ち切り規則とセットでしか
#   意味を持たない(切ると失う裾を A2 が吸収する)。P4-2 で本体へ移すときは
#   同じ規則(factor と taper)をそのまま持ち込むこと。
# ★2.5 にしている理由は計算量: 周期通路のゴースト帯幅がこの値で決まり、
#   3.0 → 2.5 で 1 評価あたり約 2.8 倍速くなる(60 分予算に収めるため)。
FAR_CUTOFF_FACTOR = 2.5
FAR_TAPER_DEFAULT = 1.0   # C¹ テーパー幅 [m](カットオフ手前 1 m で滑らかに 0 へ)
# 長距離項の近傍上限。既定 None = **上限なし**(距離カットオフ + テーパーだけで切る)。
# 理由: ρ=3 /m² では最近傍 12 体はすべて半径 ~1.1 m 以内に入るので、cap 12 を掛けると
#   「長距離項」が実質短距離項になってしまい B2 がほぼ効かなくなる(実測で確認)。
#   VISSIM の n=5 相当の人数制限は短距離項側(src の neighbor_cap=12)に残っている。
FAR_NEIGHBOR_CAP_DEFAULT = None


class ExtendedCrowd(indoor_flow.WallCrowd):
    """WallCrowd(= src の SFM)に長距離 social 項を足した **ベンチ専用** サブクラス。

    src/ を継承して使うだけで、src/ 側は 1 バイトも変えない。
    forces() は `super().forces()` の結果に加算するだけなので、短距離項・壁項・
    駆動項・揺らぎ ξ・近傍 cap の挙動は **本体と完全に同一**。

    Args(追加分のみ):
        a2:        長距離項の強さ A2 [m/s²](0 = 無効。既定 0)
        b2:        長距離項の特性距離 B2 [m](既定 2.8 = VISSIM P0)
        lambda_far: 長距離項の異方性 λ2(None = 短距離項と同じ lambda_aniso)
        far_cutoff: 長距離項の体表間隔カットオフ [m](None = FAR_CUTOFF_FACTOR·b2)
        far_taper:  カットオフ手前のテーパー幅 [m](0 = 硬い打ち切り)
        far_neighbor_cap: 長距離項の近傍上限(None = 短距離項と同じ neighbor_cap)
        cutoff_mode: "per_term"(既定・上の far_cutoff を使う)/
                     "hard"(短距離項と同じ CUTOFF_M=2.0 で切る = 現行構造の再現。
                            不連続の悪さを測るための対照)
    """

    def __init__(self, *args, a2=FAR_A_DEFAULT, b2=FAR_B_DEFAULT, lambda_far=None,
                 far_cutoff=None, far_taper=FAR_TAPER_DEFAULT,
                 far_neighbor_cap=FAR_NEIGHBOR_CAP_DEFAULT,
                 cutoff_mode="per_term", **kw):
        super().__init__(*args, **kw)
        self.a2 = float(a2)
        self.b2 = float(b2)
        self.lambda_far = float(self.lam if lambda_far is None else lambda_far)
        if cutoff_mode not in ("per_term", "hard"):
            raise ValueError("cutoff_mode must be 'per_term' or 'hard'")
        self.cutoff_mode = cutoff_mode
        if cutoff_mode == "hard":
            self.far_cutoff = float(CUTOFF_M)
            self.far_taper = 0.0
        else:
            self.far_cutoff = float(far_cutoff if far_cutoff is not None
                                    else FAR_CUTOFF_FACTOR * self.b2)
            self.far_taper = float(max(0.0, min(far_taper, self.far_cutoff)))
        self.far_neighbor_cap = (None if far_neighbor_cap is None
                                 else int(far_neighbor_cap))

    # ── 幾何(diff/d/nij/cosφ)を 1 度だけ作るための内部ヘルパ ────────────────
    def _pair_geometry(self):
        e, _ = self._desired_dir()
        diff = self.pos[:, None, :] - self.pos[None, :, :]       # j → i
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        rr = self.radius[:, None] + self.radius[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            nij = diff / d[:, :, None]
        nij = np.nan_to_num(nij)
        cosphi = -np.einsum("ik,ijk->ij", e, nij)
        return e, d, rr, nij, cosphi

    def _far_contrib(self, d, rr, nij, cosphi):
        """長距離 social 項の (N,N,2) 寄与。単位は N(= m · A2 · exp(…))。"""
        n = d.shape[0]
        gap = d - rr                                             # 体表間隔 [m]
        arg = np.clip(-gap / self.b2, a_min=None, a_max=_EXP_ARG_MAX)
        mag = self.mass * self.a2 * np.exp(arg)                  # [N]
        w = self.lambda_far + (1.0 - self.lambda_far) * (1.0 + cosphi) / 2.0
        valid = self.active[None, :] & (gap <= self.far_cutoff)
        if self.far_neighbor_cap is not None and self.far_neighbor_cap < n - 1:
            order = np.argsort(d, axis=1, kind="stable")
            rank = np.argsort(order, axis=1, kind="stable")
            valid = valid & (rank < self.far_neighbor_cap)
        if self.far_taper > 0.0:
            # C¹ smoothstep: gap <= c−wt で 1、gap >= c で 0、両端で微分 0
            c, wt = self.far_cutoff, self.far_taper
            u = np.clip((c - gap) / wt, 0.0, 1.0)
            mag = mag * (u * u * (3.0 - 2.0 * u))
        contrib = (w * mag)[:, :, None] * nij
        contrib[~valid] = 0.0
        return contrib

    def _short_contrib(self, d, rr, nij, cosphi):
        """短距離項の (N,N,2) 寄与。**sfm_core.Crowd.forces() と同一式・同一順序**。"""
        n = d.shape[0]
        arg = np.clip((rr - d) / self.b, a_min=None, a_max=_EXP_ARG_MAX)
        mag = self.a * np.exp(arg)
        w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
        valid = self.active[None, :] & (d <= rr + CUTOFF_M)
        if self.neighbor_cap is not None and self.neighbor_cap < n - 1:
            order = np.argsort(d, axis=1, kind="stable")
            rank = np.argsort(order, axis=1, kind="stable")
            valid = valid & (rank < self.neighbor_cap)
        contrib = (w * mag)[:, :, None] * nij
        contrib[~valid] = 0.0
        return contrib

    def _far_forces(self):
        """長距離 social 項だけの合力 (N,2)。参照経路(検証用・低速)。"""
        if self.pos.shape[0] < 2:
            return np.zeros_like(self.pos)
        _, d, rr, nij, cosphi = self._pair_geometry()
        return self._far_contrib(d, rr, nij, cosphi).sum(axis=1)

    def forces(self):
        """f = 駆動 + 短距離斥力 + **長距離斥力** + 壁 + ξ。

        a2 <= 0 なら super().forces() をそのまま返す(= 本体とバイト一致)。
        a2 > 0 のときは幾何(diff/d/nij/cosφ)を 1 度だけ作って両項に使い回す
        (素朴に super().forces() + _far_forces() とすると幾何を 2 度作るため約 1.8 倍遅い)。
        **短距離項の式・マスク・合算順序は sfm_core.Crowd.forces() と同一**にしてあり、
        verify_extended_matches_base() が naive 経路とのバイト一致を実測で固定する。"""
        if self.a2 <= 0.0:
            return super().forces()                  # ← 完全後方互換(バイト一致)
        n = self.pos.shape[0]
        if n > 1:
            e, d, rr, nij, cosphi = self._pair_geometry()
        else:
            e = self._desired_dir()[0]
            d = rr = nij = cosphi = None
        # ↓ 加算の順序は forces_naive()(= 本体 forces() + 長距離項)と厳密に同じ
        f = self.mass * (self.v0[:, None] * e - self.vel) / self.tau
        if n > 1:
            f = f + self._short_contrib(d, rr, nij, cosphi).sum(axis=1)
        if self.wall_field is not None:
            f = f + self._wall_forces()
        if self.noise > 0.0 and self.rng is not None:
            f = f + self.mass * self.rng.normal(0.0, self.noise, size=(n, 2))
        if n > 1:
            f = f + self._far_contrib(d, rr, nij, cosphi).sum(axis=1)
        f[~self.active] = 0.0
        return f

    def forces_naive(self):
        """検証用の素朴経路: super().forces() + _far_forces()(幾何を 2 度作る)。"""
        f = super().forces()
        if self.a2 > 0.0:
            f = f + self._far_forces()
        f[~self.active] = 0.0
        return f


class ExtendedSfmEngine(SfmEngine):
    """SfmEngine と同一 API で、下位を ExtendedCrowd に差し替えるだけのラッパ。

    engine_kw={"solver_kw": {"a2": 0.5, "b2": 2.8, "lambda_aniso": 0.1, …}}
    のように渡す(SfmEngine の solver_kw と同じ口)。"""

    name = "sfm_ext"

    def step(self, dt=0.1):
        n = self.pos.shape[0]
        g = self.ghost_pos.shape[0]
        if g:
            pos = np.concatenate([self.pos, self.ghost_pos], axis=0)
            vel = np.concatenate([self.vel, np.zeros((g, 2))], axis=0)
            goal = np.concatenate([self.goal, self.ghost_pos], axis=0)
            v0 = np.concatenate([self.v0, np.ones(g)], axis=0)
            rad = np.concatenate([self.radius, self.ghost_radius], axis=0)
            frozen = np.concatenate([np.zeros(n, bool), np.ones(g, bool)])
        else:
            pos, vel, goal = self.pos, self.vel, self.goal
            v0, rad = self.v0, self.radius
            frozen = np.zeros(n, bool)
        crowd = ExtendedCrowd(
            pos=pos, vel=vel, goal=goal, v0=v0, radius=rad,
            walls=self.walls or None, frozen=frozen,
            neighbor_cap=self.neighbor_cap, **self.solver_kw)
        crowd.step(dt)
        self.pos = crowd.pos[:n].copy()
        self.vel = crowd.vel[:n].copy()
        return self.pos


def verify_extended_matches_base(seed=20260805, n=40):
    """a2=0 の ExtendedCrowd が WallCrowd と **バイト一致** することの実測。

    src/ 無改変の主張の裏づけ(拡張が既定経路に一切影響しないこと)。
    戻り値 {"identical": bool, "hash_base": …, "hash_ext": …}。"""
    import hashlib

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 10.0, size=(n, 2))
    vel = rng.normal(0.0, 0.3, size=(n, 2))
    goal = pos + np.array([20.0, 0.0])
    ids = list(range(n))
    v0, rad = agent_params(ids)
    walls = [((-5.0, 0.0), (25.0, 0.0)), ((-5.0, 6.0), (25.0, 6.0))]
    outs = []
    for cls, kw in ((indoor_flow.WallCrowd, {}),
                    (ExtendedCrowd, {"a2": 0.0}),
                    (ExtendedCrowd, {"a2": 0.0, "b2": 2.8, "far_cutoff": 9.0})):
        c = cls(pos=pos.copy(), vel=vel.copy(), goal=goal.copy(), v0=v0,
                radius=rad, walls=walls, neighbor_cap=NEIGHBOR_CAP, **kw)
        for _ in range(50):
            c.step(0.05)
        h = hashlib.sha256()
        for a in (c.pos, c.vel):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        outs.append(h.hexdigest())
    # a2>0 のとき、融合経路 forces() と素朴経路 forces_naive() がバイト一致するか
    c = ExtendedCrowd(pos=pos.copy(), vel=vel.copy(), goal=goal.copy(), v0=v0,
                      radius=rad, walls=walls, neighbor_cap=NEIGHBOR_CAP,
                      a2=0.25, b2=2.8)
    fused, naive = c.forces(), c.forces_naive()
    fused_eq = bool(np.array_equal(fused, naive))
    for _ in range(10):                     # 数ステップ進めた状態でも確認
        c.step(0.05)
    fused_eq = fused_eq and bool(np.array_equal(c.forces(), c.forces_naive()))
    return {"identical": len(set(outs)) == 1, "hash_base": outs[0],
            "hash_ext_a2zero": outs[1], "hash_ext_a2zero_farcutoff": outs[2],
            "fused_equals_naive": fused_eq,
            "fused_naive_max_abs_diff": float(np.abs(fused - naive).max())}


ENGINES = {"sfm": SfmEngine, "orca": OrcaEngine, "sfm_ext": ExtendedSfmEngine}


def make_engine(kind, **kw):
    return ENGINES[kind](**kw)


def agent_params(ids):
    """agent_id 列 → (v0, radius)。src/society/world/indoor_flow の安定ハッシュを流用。

    v0 ∈ [1.0, 1.4) m/s、radius ∈ [0.25, 0.35) m。乱数 stream を一切引かない=決定論。"""
    v0 = np.array([indoor_flow.desired_speed(i) for i in ids], dtype=np.float64)
    rad = np.array([indoor_flow.body_radius(i) for i in ids], dtype=np.float64)
    return v0, rad
