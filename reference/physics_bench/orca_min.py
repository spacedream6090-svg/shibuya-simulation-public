"""最小 ORCA (Optimal Reciprocal Collision Avoidance) — 自前実装(numpy + 純Python LP)。

【なぜ自前実装なのか(実測に基づく正直な報告)】
  本機(Windows 11 / Python 3.12.10)では RVO2 の Python バインディングを導入できなかった。
    - `pip install rvo2` / `pyrvo2` / `python-rvo2` / `rvo2-python`
        → いずれも "ERROR: Could not find a version that satisfies the requirement
           (from versions: none)" = PyPI に配布物なし(2026-08-01 実測)。
    - 既知のバインディング(sybrenstuvel/Python-RVO2, mit-acl/Python-RVO2)は
      Cython + CMake + C++ コンパイラでのソースビルドが必須。本機は
        cl.exe → 未検出 / gcc → 未検出 / cmake → 未検出 / Cython → 未インストール
      であり、ビルドツールチェーン(Visual Studio Build Tools 等)の新規導入は
      本タスクのスコープ(pip のみ)を超えるため実施しなかった。
  よって「RVO2/ORCA 候補」は本ファイルの自前最小実装で評価する。

【移植の忠実さ】
  アルゴリズムは原論文と参照実装に忠実:
    van den Berg, Guy, Lin & Manocha (2011) "Reciprocal n-Body Collision Avoidance",
      Robotics Research (ISRR 2009), Springer STAR 70, pp.3-19.
    参照実装 RVO2 (UNC GAMMA, Apache-2.0) の Agent.cpp:
      computeNewVelocity / linearProgram1 / linearProgram2 / linearProgram3
    https://gamma.cs.unc.edu/RVO2/
  ORCA 半平面の構成(cut-off circle / left leg / right leg / collision 分岐)と
  3段 LP(LP1=1直線上の区間, LP2=2D の最近点, LP3=実行不能時の距離最小化)の
  数式・分岐条件・タイブレークをそのまま写した。

【意図的に変えた点(限界の明示)】
  (1) 浮動小数は C++ 実装の float(32bit) ではなく float64。決定論には無影響
      (演算順序は固定)。RVO2 C++ とのバイト一致は主張しない。
  (2) 静的障害物(壁)の扱いを簡略化した。RVO2 の障害物 ORCA(線分に対する速度障害物、
      左右足と隣接障害物の凸性処理)は移植していない。代わりに
        「壁最近点までの距離 d と法線 n(壁→個体)による半平面制約
             n·v >= (r - d) / tau_obst 」
      を障害物線(LP3 で必ず保持される hard constraint)として与える。
      直線壁の廊下では ORCA 障害物処理とほぼ等価だが、鋭角コーナー・薄い障害物・
      障害物端点の回り込みでは RVO2 と挙動が変わりうる。
  (3) kd-tree 近傍探索は使わず、全ペア距離 + 距離昇順の上位 k 近傍
      (安定ソート = index 昇順タイブレーク)。O(N^2) だが N<=200 では十分。

【決定論】
  乱数を一切引かない。エージェント処理順は index 昇順固定、近傍順は
  (距離昇順, index 昇順) 固定、集約順序も固定。同一入力 → 同一 float64 配列。
"""
from __future__ import annotations

import math

import numpy as np

RVO_EPSILON = 1e-5          # RVO2 の RVO_EPSILON と同値(平行判定のしきい値)


# ─────────────────────────────────────────────────────────────────────────────
# 線形計画(RVO2 Agent.cpp の linearProgram1/2/3 の移植)
#   線は (px, py, dx, dy) のタプル。実行可能領域は det(dir, v - point) >= 0
#   (= RVO2 の「det(line.direction, line.point - result) > 0 なら違反」と同値)。
# ─────────────────────────────────────────────────────────────────────────────
def _lp1(lines, line_no, radius, opt_x, opt_y, direction_opt):
    """直線 lines[line_no] 上で、先行する全制約と速度円 |v|<=radius を満たす最適点。

    満たす点が存在しなければ None(RVO2 の linearProgram1 が false を返す場合)。"""
    px, py, dx, dy = lines[line_no]
    dot = px * dx + py * dy
    disc = dot * dot + radius * radius - (px * px + py * py)
    if disc < 0.0:
        return None                      # 速度円が制約直線を完全に無効化
    sq = math.sqrt(disc)
    t_left = -dot - sq
    t_right = -dot + sq

    for i in range(line_no):
        qx, qy, ex, ey = lines[i]
        denom = dx * ey - dy * ex                       # det(dir_lineNo, dir_i)
        numer = ex * (py - qy) - ey * (px - qx)         # det(dir_i, p_lineNo - p_i)
        if abs(denom) <= RVO_EPSILON:
            if numer < 0.0:
                return None                             # 平行かつ実行不能
            continue
        t = numer / denom
        if denom >= 0.0:
            if t < t_right:
                t_right = t                             # 右から拘束
        else:
            if t > t_left:
                t_left = t                              # 左から拘束
        if t_left > t_right:
            return None

    if direction_opt:
        t = t_right if (opt_x * dx + opt_y * dy) > 0.0 else t_left
    else:
        t = dx * (opt_x - px) + dy * (opt_y - py)
        if t < t_left:
            t = t_left
        elif t > t_right:
            t = t_right
    return (px + t * dx, py + t * dy)


def _lp2(lines, radius, opt_x, opt_y, direction_opt):
    """全制約下で opt に最も近い(または opt 方向へ最も進む)速度。

    Returns: (fail_index, rx, ry)。fail_index == len(lines) なら実行可能。"""
    if direction_opt:
        rx, ry = opt_x * radius, opt_y * radius
    else:
        sq = opt_x * opt_x + opt_y * opt_y
        if sq > radius * radius:
            s = radius / math.sqrt(sq)
            rx, ry = opt_x * s, opt_y * s
        else:
            rx, ry = opt_x, opt_y

    for i in range(len(lines)):
        px, py, dx, dy = lines[i]
        if dx * (py - ry) - dy * (px - rx) > 0.0:       # det(dir, point - result) > 0 = 違反
            res = _lp1(lines, i, radius, opt_x, opt_y, direction_opt)
            if res is None:
                return i, rx, ry                        # RVO2: result = tempResult; return i
            rx, ry = res
    return len(lines), rx, ry


def _lp3(lines, num_obst_lines, begin_line, radius, rx, ry):
    """実行不能時の後退(RVO2 linearProgram3): 最大違反距離を最小化する速度を返す。"""
    distance = 0.0
    n = len(lines)
    for i in range(begin_line, n):
        pxi, pyi, dxi, dyi = lines[i]
        if dxi * (pyi - ry) - dyi * (pxi - rx) > distance:
            proj = list(lines[:num_obst_lines])         # 障害物線は必ず保持(hard)
            for j in range(num_obst_lines, i):
                pxj, pyj, dxj, dyj = lines[j]
                det_ij = dxi * dyj - dyi * dxj
                if abs(det_ij) <= RVO_EPSILON:
                    if dxi * dxj + dyi * dyj > 0.0:
                        continue                        # 同方向の平行 → 冗長
                    npx, npy = 0.5 * (pxi + pxj), 0.5 * (pyi + pyj)
                else:
                    num = dxj * (pyi - pyj) - dyj * (pxi - pxj)
                    t = num / det_ij
                    npx, npy = pxi + t * dxi, pyi + t * dyi
                ndx, ndy = dxj - dxi, dyj - dyi
                nl = math.hypot(ndx, ndy)
                if nl <= RVO_EPSILON:
                    continue                            # 正規化不能(RVO2 は 0 除算になる箇所)
                proj.append((npx, npy, ndx / nl, ndy / nl))
            tx, ty = rx, ry
            ok, nrx, nry = _lp2(proj, radius, -dyi, dxi, True)
            if ok < len(proj):
                rx, ry = tx, ty                         # 数値誤差による失敗 → 直前値を保持
            else:
                rx, ry = nrx, nry
            distance = dxi * (pyi - ry) - dyi * (pxi - rx)
    return rx, ry


# ─────────────────────────────────────────────────────────────────────────────
# ORCA 半平面の構成(ベクトル化)
# ─────────────────────────────────────────────────────────────────────────────
def build_agent_lines(pos, vel, radius, all_pos, all_vel, all_rad,
                      nb_idx, nb_valid, tau, dt):
    """(N,K,4) の ORCA 線パラメータと (N,K) の有効フラグを返す。

    RVO2 Agent.cpp computeNewVelocity の対エージェント部をそのまま numpy 化。
    nb_idx: (N,K) 近傍 index(all_* 配列への index)。nb_valid=False の箇所は無視。
    all_*:  実体 + 周期ゴーストを結合した配列(近傍探索の対象)。
    """
    p_i = pos[:, None, :]                               # (N,1,2)
    v_i = vel[:, None, :]
    r_i = radius[:, None]                               # (N,1)
    p_j = all_pos[nb_idx]                               # (N,K,2)
    v_j = all_vel[nb_idx]
    r_j = all_rad[nb_idx]                               # (N,K)

    x = p_j - p_i                                       # relativePosition
    v = v_i - v_j                                       # relativeVelocity
    r = r_i + r_j                                       # combinedRadius
    dist_sq = (x * x).sum(axis=2)
    r_sq = r * r
    inv_tau = 1.0 / tau
    inv_dt = 1.0 / dt

    collide = dist_sq <= r_sq

    # ── 非衝突枝 ──
    w = v - x * inv_tau
    w_len_sq = (w * w).sum(axis=2)
    dot1 = (w * x).sum(axis=2)
    cutoff = (dot1 < 0.0) & (dot1 * dot1 > r_sq * w_len_sq)

    w_len = np.sqrt(np.maximum(w_len_sq, 1e-300))
    unit_w = w / w_len[:, :, None]
    dir_cut = np.stack([unit_w[:, :, 1], -unit_w[:, :, 0]], axis=2)
    u_cut = (r * inv_tau - w_len)[:, :, None] * unit_w

    leg = np.sqrt(np.maximum(dist_sq - r_sq, 0.0))
    det_xw = x[:, :, 0] * w[:, :, 1] - x[:, :, 1] * w[:, :, 0]
    ds = np.where(dist_sq > 1e-300, dist_sq, 1.0)
    left = np.stack([x[:, :, 0] * leg - x[:, :, 1] * r,
                     x[:, :, 0] * r + x[:, :, 1] * leg], axis=2) / ds[:, :, None]
    right = -np.stack([x[:, :, 0] * leg + x[:, :, 1] * r,
                       -x[:, :, 0] * r + x[:, :, 1] * leg], axis=2) / ds[:, :, None]
    dir_leg = np.where((det_xw > 0.0)[:, :, None], left, right)
    dot2 = (v * dir_leg).sum(axis=2)
    u_leg = dot2[:, :, None] * dir_leg - v

    # ── 衝突枝(dt の cut-off circle に射影) ──
    w_c = v - x * inv_dt
    w_c_len = np.sqrt(np.maximum((w_c * w_c).sum(axis=2), 1e-300))
    unit_wc = w_c / w_c_len[:, :, None]
    dir_col = np.stack([unit_wc[:, :, 1], -unit_wc[:, :, 0]], axis=2)
    u_col = (r * inv_dt - w_c_len)[:, :, None] * unit_wc

    direction = np.where(collide[:, :, None], dir_col,
                         np.where(cutoff[:, :, None], dir_cut, dir_leg))
    u = np.where(collide[:, :, None], u_col,
                 np.where(cutoff[:, :, None], u_cut, u_leg))
    point = v_i + 0.5 * u                               # line.point = velocity + 0.5 u

    lines = np.concatenate([point, direction], axis=2)  # (N,K,4)
    return lines, nb_valid


def wall_lines(pos, radius, wall_p1, wall_p2, tau_obst, wall_range):
    """壁半平面制約 n·v >= (r - d)/tau_obst を (N,M,4)+(N,M) で返す(簡略化・上註 (2))。"""
    if wall_p1 is None or len(wall_p1) == 0:
        n = pos.shape[0]
        return np.zeros((n, 0, 4)), np.zeros((n, 0), dtype=bool)
    e = wall_p2 - wall_p1                               # (M,2)
    len2 = (e * e).sum(axis=1)
    len2 = np.where(len2 > 1e-12, len2, 1.0)
    rel = pos[:, None, :] - wall_p1[None, :, :]         # (N,M,2)
    t = np.clip((rel * e[None, :, :]).sum(axis=2) / len2[None, :], 0.0, 1.0)
    closest = wall_p1[None, :, :] + t[:, :, None] * e[None, :, :]
    dvec = pos[:, None, :] - closest                    # 壁→個体
    d = np.linalg.norm(dvec, axis=2)
    nvec = dvec / np.maximum(d, 1e-12)[:, :, None]
    c = (radius[:, None] - d) / tau_obst                # 制約 n·v >= c
    point = c[:, :, None] * nvec
    direction = np.stack([nvec[:, :, 1], -nvec[:, :, 0]], axis=2)
    lines = np.concatenate([point, direction], axis=2)
    valid = d < (radius[:, None] + wall_range)
    return lines, valid


# ─────────────────────────────────────────────────────────────────────────────
# ORCA シミュレータ
# ─────────────────────────────────────────────────────────────────────────────
class OrcaCrowd:
    """ORCA による 1 ステップ速度計算 + 前進積分。

    Args:
        pos/vel: (N,2) 位置・速度 [m], [m/s]
        radius:  (N,)  半径 [m]
        v0:      (N,)  希望速度 [m/s]
        walls:   [((x1,y1),(x2,y2)), ...] 壁線分(簡略半平面制約として使う)
        tau:     対エージェント時間地平 [s](RVO2 の timeHorizon)
        tau_obst: 対壁時間地平 [s]
        neighbor_cap: 近傍上限 K
        neighbor_dist: 近傍とみなす最大距離 [m]
        v_max_factor: 最大速度 = v0 * factor(SFM 側と同条件にする)
    """

    def __init__(self, pos, vel, goal, v0, radius, walls=None, neighbor_cap=12,
                 tau=2.0, tau_obst=2.0, neighbor_dist=10.0, wall_range=2.0,
                 v_max_factor=1.3, arrive_radius=0.5, pref_noise=0.0, rng=None):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2).copy()
        self.vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2).copy()
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2).copy()
        self.v0 = np.asarray(v0, dtype=np.float64).reshape(-1).copy()
        self.radius = np.asarray(radius, dtype=np.float64).reshape(-1).copy()
        self.v_max = v_max_factor * self.v0
        self.neighbor_cap = int(neighbor_cap)
        self.tau = float(tau)
        self.tau_obst = float(tau_obst)
        self.neighbor_dist = float(neighbor_dist)
        self.wall_range = float(wall_range)
        self.arrive_radius = float(arrive_radius)
        segs = list(walls or [])
        if segs:
            self._wp1 = np.array([[s[0][0], s[0][1]] for s in segs], dtype=np.float64)
            self._wp2 = np.array([[s[1][0], s[1][1]] for s in segs], dtype=np.float64)
        else:
            self._wp1 = self._wp2 = None
        # 対称性の破れ用の希望速度ゆらぎ [m/s]。ORCA は決定論的な対称配置で
        # デッドロックしやすい(RVO2 でも実運用では pref velocity に微小摂動を入れる)。
        # rng を外から渡す=乱数は専用 stream・同一 seed で同一軌跡(決定論は保たれる)。
        self.pref_noise = float(pref_noise)
        self.rng = rng
        # 周期境界用のゴースト(位置・速度・半径)。空なら通常。
        self.ghost_pos = np.zeros((0, 2))
        self.ghost_vel = np.zeros((0, 2))
        self.ghost_radius = np.zeros(0)

    def set_ghosts(self, gpos, gvel, gradius):
        self.ghost_pos = np.asarray(gpos, dtype=np.float64).reshape(-1, 2)
        self.ghost_vel = np.asarray(gvel, dtype=np.float64).reshape(-1, 2)
        self.ghost_radius = np.asarray(gradius, dtype=np.float64).reshape(-1)

    def _pref_velocity(self):
        d = self.goal - self.pos
        dist = np.linalg.norm(d, axis=1)
        e = np.zeros_like(d)
        nz = dist > 1e-9
        e[nz] = d[nz] / dist[nz, None]
        pref = e * self.v0[:, None]
        if self.pref_noise > 0.0 and self.rng is not None:
            pref = pref + self.rng.normal(0.0, self.pref_noise, size=pref.shape)
        return pref

    def step(self, dt=0.1):
        n = self.pos.shape[0]
        # 実体 + ゴーストの結合配列(近傍探索の対象)。実体は先頭 n。
        all_pos = np.concatenate([self.pos, self.ghost_pos], axis=0)
        all_vel = np.concatenate([self.vel, self.ghost_vel], axis=0)
        all_rad = np.concatenate([self.radius, self.ghost_radius], axis=0)
        m = all_pos.shape[0]

        # 近傍(距離昇順・index 昇順タイブレーク = 決定論)
        diff = self.pos[:, None, :] - all_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[np.arange(n), np.arange(n)] = np.inf
        k = min(self.neighbor_cap, max(m - 1, 1))
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]
        nb_valid = np.take_along_axis(dist, order, axis=1) < self.neighbor_dist

        a_lines, a_valid = build_agent_lines(self.pos, self.vel, self.radius,
                                             all_pos, all_vel, all_rad,
                                             order, nb_valid, self.tau, dt)
        o_lines, o_valid = wall_lines(self.pos, self.radius, self._wp1, self._wp2,
                                      self.tau_obst, self.wall_range)

        v_pref = self._pref_velocity()
        new_vel = np.empty_like(self.vel)
        a_lines_l = a_lines.tolist()
        o_lines_l = o_lines.tolist()
        a_valid_l = a_valid.tolist()
        o_valid_l = o_valid.tolist()
        for i in range(n):
            obst = [tuple(l) for l, ok in zip(o_lines_l[i], o_valid_l[i]) if ok]
            agents = [tuple(l) for l, ok in zip(a_lines_l[i], a_valid_l[i]) if ok]
            lines = obst + agents
            n_obst = len(obst)
            vmax = float(self.v_max[i])
            ok, rx, ry = _lp2(lines, vmax, float(v_pref[i, 0]), float(v_pref[i, 1]), False)
            if ok < len(lines):
                rx, ry = _lp3(lines, n_obst, ok, vmax, rx, ry)
            new_vel[i, 0] = rx
            new_vel[i, 1] = ry

        self.vel = new_vel
        self.pos = self.pos + new_vel * dt
        return self.pos

    def arrived(self):
        d = np.linalg.norm(self.goal - self.pos, axis=1)
        return d < self.arrive_radius
