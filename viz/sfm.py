"""最小 Social Force Model (SFM) コア — 渋谷スクランブル領域の微視歩行者物理。

外部ライブラリ(PySocialForce 等)は一切使わない完全自前の最小実装。
数式・パラメータはすべて公開文献のもの(出典を各所に明記):

  - Helbing & Molnár (1995) "Social Force Model for Pedestrian Dynamics",
    Phys. Rev. E 51, 4282-4286.
        駆動項  f_drive = m (v0·e − v) / τ           … 希望速度 v0・希望方向 e へ緩和
        対人斥力 |f_ij| = A·exp((r_i+r_j − d_ij)/B)   … 指数ポテンシャルの勾配(円形近似)
        異方性重み w = λ + (1−λ)(1 + cosφ)/2         … 前方の相手ほど強く効く(視野)
  - Helbing, Farkas & Vicsek (2000) "Simulating dynamical features of escape panic",
    Nature 407, 487-490.
        確定パラメータ  A = 2000 N, B = 0.08 m, τ = 0.5 s, m = 80 kg,
                        r ∈ [0.25, 0.35] m(一様), v_max = 1.3·v0
  詳細と出典アクセス日は docs/research/social-force-crowd.md §1 を参照。

────────────────────────────────────────────────────────────────────────
本コアで【意図的に省略した項】(第1版・docs/plans の指示に従う):
  (1) 対壁斥力 f_iW(壁・障害物からの斥力)。
      → 理由: 案a は「スクランブル領域(円)」だけを扱い、建物外壁・車道縁石など
        の障害物ジオメトリを持ち込まない。境界は円の外に出たら退場する扱いのみ。
  (2) 物理接触項 — body force  k·g(r_ij−d_ij)·n_ij(弾性反発)と
      sliding friction  κ·g(r_ij−d_ij)·Δv_t·t_ij(接線摩擦)。
      (Helbing2000 のパニック弾性/摩擦。k=1.2e5, κ=2.4e5)
      → 理由: これらは高密度の押し合い・すり抜け防止(体の非圧縮)を担う項で、
        本用途(低〜中密度の「揉まれ方」= 減速・回避・レーン形成の再現)には不要。
  ∴ 本コアは超高密度(LOS F 相当)の圧潰・faster-is-slower は再現対象外。
    社会的斥力 A·exp(...) は接触で有限値 A に留まるため、稀な深い重なりは
    速度上限クリップ(v_max)で数値的に抑える(接触項の代替ではない)。
────────────────────────────────────────────────────────────────────────

決定論について:
  乱数は numpy.random.Generator を引数で受け取り、コア内部では seed しない。
  揺らぎ項 ξ(既定 noise=0.0 で無効)を有効化したときだけ Generator を消費する。
  noise=0.0 のときコアは完全に決定論的(同一入力→同一 float 配列)。
"""
from __future__ import annotations

import numpy as np

# ── 確定パラメータ(Helbing & Molnár 1995 / Helbing, Farkas & Vicsek 2000) ──
#    リサーチ docs/research/social-force-crowd.md §1.2 表・§1.3
TAU_DEFAULT = 0.5        # 緩和(反応)時間 [s]           … Helbing2000
A_DEFAULT = 2000.0       # 社会的斥力の強さ [N]          … Helbing2000
B_DEFAULT = 0.08         # 斥力の特性距離 [m]            … Helbing2000
MASS_DEFAULT = 80.0      # 歩行者質量 [kg]               … Helbing2000
LAMBDA_DEFAULT = 0.5     # 異方性(視野)係数 c≈0.5      … Helbing&Molnár1995
V_MAX_FACTOR = 1.3       # 最高速度 = 希望速度 × 1.3      … Helbing2000
RADIUS_MIN = 0.25        # 半径下限 [m](一様分布)       … Helbing2000
RADIUS_MAX = 0.35        # 半径上限 [m]
CUTOFF_M = 2.0           # 斥力カットオフ半径 [m]。exp((r−d)/B) は B=0.08 で数m先は
#                          ~0(d=接触+0.5mで数N)。§1.4 の近傍リスト/カットオフに対応。
_EXP_ARG_MAX = 4.0       # exp 引数の上限(深い重なり時の overflow/暴走を防ぐ安全弁)


class Crowd:
    """群衆の状態(位置・速度・目標・希望速度)と 1 サブステップの Euler 積分。

    すべて numpy 配列で保持し、力の評価はベクトル化(全ペア O(n²)+カットオフ)。
    数百体×数千サブステップ程度なら十分軽量(リサーチ §1.4)。

    Args:
        pos:    (N,2) 初期位置 [m]      (X=east, Y=north)
        vel:    (N,2) 初期速度 [m/s]
        goal:   (N,2) 目標点(出口)[m]
        v0:     (N,)  個体別 希望速度 [m/s]
        radius: (N,)  個体別 半径 [m]。None なら rng で U(0.25,0.35) を引く。
        active: (N,)  bool。False の個体は力を及ぼさず・受けず・動かない。
        rng:    numpy.random.Generator。radius 抽選と揺らぎ ξ に使う(内部 seed 禁止)。
        noise:  揺らぎ ξ の標準偏差 [m/s²]。0.0(既定)で決定論・ξ 無効。
        arrive_radius: 目標到達判定半径 [m]。
    """

    def __init__(self, pos, vel, goal, v0, radius=None, active=None,
                 mass=MASS_DEFAULT, tau=TAU_DEFAULT, a=A_DEFAULT, b=B_DEFAULT,
                 lambda_aniso=LAMBDA_DEFAULT, v_max_factor=V_MAX_FACTOR,
                 rng=None, noise=0.0, arrive_radius=0.5):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2).copy()
        n = self.pos.shape[0]
        self.vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2).copy()
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2).copy()
        self.v0 = np.asarray(v0, dtype=np.float64).reshape(-1).copy()
        if radius is None:
            if rng is None:
                radius = np.full(n, 0.5 * (RADIUS_MIN + RADIUS_MAX))
            else:
                radius = rng.uniform(RADIUS_MIN, RADIUS_MAX, size=n)
        self.radius = np.asarray(radius, dtype=np.float64).reshape(-1).copy()
        self.active = (np.ones(n, dtype=bool) if active is None
                       else np.asarray(active, dtype=bool).reshape(-1).copy())
        self.mass = float(mass)
        self.tau = float(tau)
        self.a = float(a)
        self.b = float(b)
        self.lam = float(lambda_aniso)
        self.v_max = v_max_factor * self.v0        # (N,) 個体別 最高速度
        self.rng = rng
        self.noise = float(noise)
        self.arrive_radius = float(arrive_radius)

    # ── 希望方向 e_i(目標へ向かう単位ベクトル) ──
    def _desired_dir(self):
        d = self.goal - self.pos                    # (N,2)
        dist = np.linalg.norm(d, axis=1)            # (N,)
        e = np.zeros_like(d)
        nz = dist > 1e-9
        e[nz] = d[nz] / dist[nz, None]
        return e, dist

    # ── 全力の合成(駆動 + 対人斥力[+揺らぎ]) ──
    def forces(self):
        """各個体に働く合力 (N,2) [N] を返す。非アクティブ個体は 0。"""
        n = self.pos.shape[0]
        e, _ = self._desired_dir()

        # 駆動項: f_drive = m (v0 e − v) / τ   (Helbing&Molnár1995)
        f = self.mass * (self.v0[:, None] * e - self.vel) / self.tau

        if n > 1:
            # 対人斥力(全ペア)。diff_ij = pos_i − pos_j(j から i へ向かうベクトル)
            diff = self.pos[:, None, :] - self.pos[None, :, :]     # (N,N,2)
            d = np.linalg.norm(diff, axis=2)                       # (N,N)
            np.fill_diagonal(d, np.inf)                            # 自己ペア除外
            rr = self.radius[:, None] + self.radius[None, :]       # (N,N) 半径和
            # |f_ij| = A·exp((r_i+r_j − d)/B) — 指数斥力(接触 d=rr で A)
            arg = np.clip((rr - d) / self.b, a_min=None, a_max=_EXP_ARG_MAX)
            mag = self.a * np.exp(arg)                             # (N,N)
            with np.errstate(invalid="ignore", divide="ignore"):
                nij = diff / d[:, :, None]                         # j→i 単位ベクトル
            nij = np.nan_to_num(nij)
            # 異方性 w = λ + (1−λ)(1+cosφ)/2、φ = e_i と (i→j 方向) の成す角
            #   i→j 方向 = −nij。cosφ = e_i · (−nij)。前方(cosφ=1)で w=1、後方で w=λ。
            cosphi = -np.einsum("ik,ijk->ij", e, nij)              # (N,N)
            w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
            # マスク: 非アクティブ相手 j / カットオフ外 / 自己 は寄与 0
            valid = self.active[None, :] & (d <= rr + CUTOFF_M)
            contrib = (w * mag)[:, :, None] * nij                 # (N,N,2)
            contrib[~valid] = 0.0
            f_rep = contrib.sum(axis=1)                           # (N,2)
            f = f + f_rep

        # 揺らぎ ξ(既定 noise=0 で無効・決定論)
        if self.noise > 0.0 and self.rng is not None:
            f = f + self.mass * self.rng.normal(0.0, self.noise, size=(n, 2))

        f[~self.active] = 0.0
        return f

    def step(self, dt=0.1):
        """1 サブステップの前進 Euler 積分。位置・速度を in-place 更新する。

        v_new = v + (F/m)·dt → 速さを v_max でクリップ → x += v_new·dt。
        非アクティブ個体は速度 0・不動(数値安全弁も兼ねる)。
        """
        f = self.forces()
        acc = f / self.mass
        v_new = self.vel + acc * dt
        # 最高速度クリップ(個体別 v_max = 1.3 v0)。深い重なり時の暴走もこれで抑える。
        speed = np.linalg.norm(v_new, axis=1)
        over = speed > self.v_max
        if np.any(over):
            v_new[over] *= (self.v_max[over] / speed[over])[:, None]
        v_new[~self.active] = 0.0
        self.vel = v_new
        self.pos = self.pos + v_new * dt
        return self.pos

    def arrived(self):
        """目標到達(距離 < arrive_radius)した個体の bool 配列 (N,)。"""
        d = np.linalg.norm(self.goal - self.pos, axis=1)
        return d < self.arrive_radius
