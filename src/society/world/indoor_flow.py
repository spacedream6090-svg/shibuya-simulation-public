"""屋内ミクロ人流拡張 — 遷移駆動 Social Force Model(壁斥力 + ドアウェイポイント + 接触ペア抽出)。

sfm_core.Crowd(開放領域の対人斥力・駆動項)を屋内へ拡張する層。時間設計は「遷移駆動」:
1 認知 step(=10分)は不変で、エージェントが空間遷移(部屋→部屋、部屋→階段/EV コア)を
起こした瞬間だけ dt=0.2s のサブステップ積分で微視軌跡を作る。常時ミリング(全員が毎秒動く)は
採らない=遷移が起きたフロア・時間帯だけコストを払う。

拡張点(sfm_core からの差分):
  - 壁斥力 f_iW = A·exp((r_i − d_iW)/B)·n_iW(A=2000 N, B=0.08 m。Helbing2000 の壁項)。
    壁セグメント(vision.building_walls の形式 [((x1,y1),(x2,y2)), …])への最近点距離 d_iW と
    法線 n_iW(壁→個体)で法線方向の斥力を与える。
    【意図的な省略(コード註で正直に明記)】: 壁の物理接触項 — body force k·g(r−d)·n_iW と
      sliding friction κ·g(r−d)·Δv_t·t_iW — は入れない。屋内 10 分粒度の遷移は
      低密度(廊下・ドア前で数体)で、押し合いの非圧縮(体のめり込み防止)より
      「避けて通る」減速・迂回が主。指数斥力 A·exp(...) と速度上限クリップで十分近似でき、
      高密度の圧潰(faster-is-slower)は屋内遷移の観測対象外。この近似の限界は本註のとおり。
  - ドア = ウェイポイント(吸引ポテンシャルではない)。経路は waypoint 連鎖を順に goal 切替して
    追従する(ドア開口の中心点を経由点に置くだけ=開口を通れば壁斥力に阻まれない)。
  - 近傍計算 cap(既定 12・id 昇順で決定論選択)。対人斥力は各個体につき最近傍 cap 体のみ寄与。
  - 接触ペア抽出(遭遇観測の基盤): 積分中、中心間距離 < r_contact が連続 T_sub サブステップ以上
    続いたペアを (id_a, id_b, kind, duration_s) で記録。kind は passing(双方が移動中)/
    coplace(一方が静止=静止者 or 経路完了者)の 2 種。id_a < id_b 正規化・記録順は決定論。

決定論(R1 doctrine #4):
  id 昇順の固定配列・dt/上限固定・積分内乱数ゼロ。個体パラメータ(希望速度 v0・半径 radius)は
  agent_id からの blake2b 安定ハッシュ(run.seed 非依存・毎回同値)。src/society/ontology.py の
  _stable_uniform と同一流儀(hashlib=プロセス跨ぎ安定・RngHub の乱数列に無風)。

no-fingerprint(この層は world/ = 走査対象): 座標・幾何しか見ない(因子名・地名を一切書かない)。

B1/B3 との境界:
  ドア開口情報は本モジュールの doors_from_layout(layout)= vision._walls_from_layout が開ける
  開口(zone の廊下側の辺の中央)を独立に導出する自己完結ヘルパで代用する。B1 の
  doors_from_layout(vision.py 側で並行実装中)との接続、および scheduler への配線は B3 で行う。
"""
from __future__ import annotations

import hashlib
import math
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from society.world.sfm_core import Crowd, CUTOFF_M, _EXP_ARG_MAX

# ── 屋内拡張の確定パラメータ ──
WALL_A = 2000.0      # 壁斥力の強さ [N]        … Helbing2000 の壁項(対人と同値の標準値)
WALL_B = 0.08        # 壁斥力の特性距離 [m]     … Helbing2000
NEIGHBOR_CAP = 12    # 対人斥力の近傍上限(最近傍のみ寄与・id 昇順で決定論選択)
V0_MIN, V0_MAX = 1.0, 1.4       # 希望速度の一様域 [m/s](agent_id 安定ハッシュで抽選)
RADIUS_MIN, RADIUS_MAX = 0.25, 0.35  # 半径の一様域 [m](Helbing2000)
_WALL_CUTOFF_M = 2.0             # 壁斥力カットオフ [m](exp は B=0.08 で数m先 ~0)


# ─────────────────────────────────────────────────────────────────────────────
# 個体パラメータ(agent_id からの安定ハッシュ。run.seed 非依存・毎回同値)
# ─────────────────────────────────────────────────────────────────────────────
def _stable_uniform(pid: str, salt: str = "") -> float:
    """blake2b 安定ハッシュ由来の一様値 [0,1)。ontology._stable_uniform と同一流儀。

    hashlib はプロセス跨ぎで安定(PYTHONHASHSEED の salt を受けない)=別ラン・resume でも
    同一 pid は常に同一値。RngHub の stream を一切引かない(乱数列に無風)。salt で v0/radius 等の
    独立な一様値を派生する。"""
    h = hashlib.blake2b(f"{salt}\x1f{pid}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def desired_speed(agent_id) -> float:
    """agent_id → 希望速度 v0 ∈ [V0_MIN, V0_MAX)(安定ハッシュ・決定論)。"""
    u = _stable_uniform(str(agent_id), "v0")
    return V0_MIN + u * (V0_MAX - V0_MIN)


def body_radius(agent_id) -> float:
    """agent_id → 半径 radius ∈ [RADIUS_MIN, RADIUS_MAX)(安定ハッシュ・決定論)。"""
    u = _stable_uniform(str(agent_id), "radius")
    return RADIUS_MIN + u * (RADIUS_MAX - RADIUS_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# 幾何ヘルパ(自己完結。vision の内部関数には依存しない=B1 の変更から独立)
# ─────────────────────────────────────────────────────────────────────────────
def _seg_cross(p1, p2, p3, p4) -> bool:
    """線分 p1p2 と p3p4 が真に交差するか(端点接触・共線は非交差=寛容側)。

    vision._seg_cross と同一の向き付け判定(ドアの開口=線分の隙間を塞がない寛容側)。"""
    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = orient(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])
    d2 = orient(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1])
    d3 = orient(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    d4 = orient(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def segment_hits_wall(p, q, walls) -> bool:
    """線分 p-q が walls のいずれかと交差するか(壁貫通=非到達判定)。"""
    for w in walls:
        if _seg_cross(p, q, w[0], w[1]):
            return True
    return False


def _point_seg(px, py, ax, ay, bx, by):
    """点 (px,py) から線分 (ax,ay)-(bx,by) への最近点と距離を返す。"""
    ex, ey = bx - ax, by - ay
    len2 = ex * ex + ey * ey
    if len2 <= 1e-12:
        cx, cy = ax, ay
    else:
        t = ((px - ax) * ex + (py - ay) * ey) / len2
        t = min(1.0, max(0.0, t))
        cx, cy = ax + t * ex, ay + t * ey
    return (cx, cy), math.hypot(px - cx, py - cy)


# ─────────────────────────────────────────────────────────────────────────────
# ドア開口の導出(vision._walls_from_layout が開ける開口を独立再現する自己完結ヘルパ)
# ─────────────────────────────────────────────────────────────────────────────
def doors_from_layout(layout: dict) -> list:
    """間取り → 各 zone の廊下側の辺の中央にあるドア開口の中心点 [(x,y), …]。

    vision._walls_from_layout の開口位置(zone の廊下側辺の中央・半幅 min(0.8,幅*0.25))と
    同じ x/y に開口中心を置く=このドア点を経由すれば開口(壁の隙間)を通る。zone index の順は
    layout["zones"] と同順(側A[0..nA-1] → 側B[nA..])。B3 で B1 の doors_from_layout に差し替える。
    """
    zones = layout["zones"]
    n_a = layout["nA"]
    doors = []
    if layout["horiz"]:
        _, yc0, _, yc1 = layout["corridor"]
        for idx, (zx0, _zy0, zx1, _zy1) in enumerate(zones):
            xm = (zx0 + zx1) / 2.0
            y = yc1 if idx < n_a else yc0        # 側A=上(通路の上辺 yc1)・側B=下(yc0)
            doors.append((xm, y))
    else:
        xc0, _yc0, xc1, _yc1 = layout["corridor"]
        for idx, (_zx0, zy0, _zx1, zy1) in enumerate(zones):
            ym = (zy0 + zy1) / 2.0
            x = xc1 if idx < n_a else xc0        # 側A=右(通路の右辺 xc1)・側B=左(xc0)
            doors.append((x, ym))
    return doors


def _zone_center(layout: dict, idx: int):
    zx0, zy0, zx1, zy1 = layout["zones"][idx]
    return ((zx0 + zx1) / 2.0, (zy0 + zy1) / 2.0)


def _corridor_center_cross(layout: dict):
    """廊下中心線の「横断」座標(horiz なら中心 y、vert なら中心 x)を返す。"""
    cx0, cy0, cx1, cy1 = layout["corridor"]
    if layout["horiz"]:
        return (cy0 + cy1) / 2.0
    return (cx0 + cx1) / 2.0


def _corridor_approach(layout: dict, door):
    """ドア点に対応する廊下中央線上の接近点(ドアから廊下芯へ寄せた経由点)。"""
    cross = _corridor_center_cross(layout)
    if layout["horiz"]:
        return (door[0], cross)
    return (cross, door[1])


# ─────────────────────────────────────────────────────────────────────────────
# 経路生成(純関数): zone→自zoneドア→通路内中継点→目的zoneドア→zone内目標
# ─────────────────────────────────────────────────────────────────────────────
def route_waypoints(layout: dict, doors: list, src_zone, dst_zone,
                    start=None, walls=None, cell: float = 0.5) -> list:
    """直交間取りでの waypoint 連鎖(決定論・純関数)。

    通常経路(3〜5 点): [src ドア, src 廊下接近点, dst 廊下接近点, dst ドア, dst 内目標]。
    dst_zone が "core"(階段/EV コア)なら [src ドア, src 廊下接近点, コア接近点] の 3 点。
    重複する連続点は畳む。start(遷移の出発点)は goal 列に含めない(積分の初期位置)。

    walls を渡した場合のみ、連続 waypoint の直線が壁と交差する区間を 0.5m グリッド BFS 距離場
    フォールバックで迂回に差し替える(非凸レイアウト・開口位置ずれ対策)。walls=None なら
    直線 waypoint のまま(直交間取りの正常経路は開口を通るので通常は交差しない)。
    """
    src_door = tuple(doors[src_zone])
    src_appr = _corridor_approach(layout, src_door)

    if dst_zone == "core":
        core_goal = _core_approach(layout, src_door)
        wps = [src_door, src_appr, core_goal]
    else:
        dst_door = tuple(doors[dst_zone])
        dst_appr = _corridor_approach(layout, dst_door)
        dst_target = _zone_center(layout, dst_zone)
        wps = [src_door, src_appr, dst_appr, dst_door, dst_target]

    wps = _dedupe(wps)

    if walls:
        anchor = tuple(start) if start is not None else _zone_center(layout, src_zone)
        wps = _repair_with_bfs(layout, walls, anchor, wps, cell)
    return wps


def _core_approach(layout: dict, src_door):
    """コア(階段/EV。無開口の中央矩形)への接近目標 — 廊下芯上・コア近辺の手前に置く。

    コアは廊下芯上にあり壁で囲われ開口が無い。実運用の階段/EV アクセス点は本層では厳密化せず、
    廊下からコアの手前(半径ぶん外側)に目標を置く近似(この近似は註のとおり=B3 で精緻化余地)。"""
    cx0, cy0, cx1, cy1 = layout["core"]
    cross = _corridor_center_cross(layout)
    clr = RADIUS_MAX + 0.1
    if layout["horiz"]:
        # 廊下は水平。src が芯の左右どちら側かでコアの近い辺の外側へ。
        return (cx0 - clr, cross) if src_door[0] <= (cx0 + cx1) / 2.0 else (cx1 + clr, cross)
    return (cross, cy0 - clr) if src_door[1] <= (cy0 + cy1) / 2.0 else (cross, cy1 + clr)


def _dedupe(pts, eps: float = 1e-9) -> list:
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(tuple(p))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BFS 距離場フォールバック(0.5m グリッド・純関数・キャッシュ可能)
# ─────────────────────────────────────────────────────────────────────────────
def _repair_with_bfs(layout, walls, anchor, wps, cell) -> list:
    """anchor→wps を順に辿り、壁と交差する区間だけ BFS 迂回で差し替える。"""
    chain = [anchor] + list(wps)
    out = [anchor]
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        if segment_hits_wall(a, b, walls):
            detour = _grid_bfs_path(layout, walls, a, b, cell)
            # detour は a を含み b を含む。連続畳み込みで a の重複を避けて連結。
            out.extend(detour[1:])
        else:
            out.append(b)
    return _dedupe(out)


def _grid_bfs_path(layout, walls, start, goal, cell: float = 0.5) -> list:
    """0.5m グリッド BFS 距離場で start→goal の壁回避経路を返す(決定論・純関数)。

    layout の bbox から格子ノードを構築し、壁近傍(clearance 内)ノードと壁を跨ぐ辺を禁止。
    goal から BFS で距離場を作り、start から最急降下で辿る。直線区間は畳んで waypoint 化する。
    経路が無ければ [start, goal](フォールバック不能=呼び出し側で壁交差が残る)を返す。"""
    x0, y0, x1, y1 = layout["bbox"]
    pad = cell
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    nx = int(math.ceil((x1 - x0) / cell)) + 1
    ny = int(math.ceil((y1 - y0) / cell)) + 1
    clearance = RADIUS_MAX                       # 半径ぶんは壁から離す=ドア開口(半幅0.8)は通れる

    def node_xy(i, j):
        return (x0 + i * cell, y0 + j * cell)

    def blocked_node(i, j):
        px, py = node_xy(i, j)
        for w in walls:
            _, d = _point_seg(px, py, w[0][0], w[0][1], w[1][0], w[1][1])
            if d < clearance:
                return True
        return False

    free = [[not blocked_node(i, j) for j in range(ny)] for i in range(nx)]

    def nearest_free(pt):
        # 最近傍の自由ノード(格子座標)。真上のノードが塞がっていれば同心リングで探索。
        gi = min(nx - 1, max(0, int(round((pt[0] - x0) / cell))))
        gj = min(ny - 1, max(0, int(round((pt[1] - y0) / cell))))
        if free[gi][gj]:
            return (gi, gj)
        for r in range(1, max(nx, ny)):
            best = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = gi + di, gj + dj
                    if 0 <= i < nx and 0 <= j < ny and free[i][j]:
                        cand = (i, j)
                        if best is None or cand < best:   # id 昇順に相当する決定論タイブレーク
                            best = cand
            if best is not None:
                return best
        return (gi, gj)

    si, sj = nearest_free(start)
    gi, gj = nearest_free(goal)

    # goal からの BFS 距離場(4近傍・壁を跨ぐ辺は禁止)。近傍順は固定=決定論。
    INF = float("inf")
    dist = [[INF] * ny for _ in range(nx)]
    dist[gi][gj] = 0.0
    queue = [(gi, gj)]
    head = 0
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while head < len(queue):
        ci, cj = queue[head]
        head += 1
        base = dist[ci][cj]
        cxy = node_xy(ci, cj)
        for di, dj in steps:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not free[ni][nj]:
                continue
            if dist[ni][nj] != INF:
                continue
            nxy = node_xy(ni, nj)
            if segment_hits_wall(cxy, nxy, walls):
                continue
            dist[ni][nj] = base + 1.0
            queue.append((ni, nj))

    if dist[si][sj] == INF:
        return [tuple(start), tuple(goal)]       # 到達不能=迂回不能(呼び出し側に委ねる)

    # start から最急降下(近傍の最小 dist を辿る。ties は固定近傍順=決定論)。
    path_nodes = [(si, sj)]
    ci, cj = si, sj
    guard = nx * ny + 4
    while (ci, cj) != (gi, gj) and guard > 0:
        guard -= 1
        best = None
        best_d = dist[ci][cj]
        cxy = node_xy(ci, cj)
        for di, dj in steps:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not free[ni][nj]:
                continue
            if segment_hits_wall(cxy, node_xy(ni, nj), walls):
                continue
            if dist[ni][nj] < best_d:
                best_d = dist[ni][nj]
                best = (ni, nj)
        if best is None:
            break
        path_nodes.append(best)
        ci, cj = best

    # ノード列 → 座標。直線に並ぶ中間点を畳んで waypoint 化。始点/終点は実座標に固定。
    coords = [node_xy(i, j) for (i, j) in path_nodes]
    coords[0] = tuple(start)
    coords[-1] = tuple(goal)
    return _collapse_collinear(coords)


def _collapse_collinear(pts, eps: float = 1e-6) -> list:
    if len(pts) <= 2:
        return _dedupe(pts)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(cross) > eps:                     # 曲がり角のみ残す
            out.append(pts[i])
    out.append(pts[-1])
    return _dedupe(out)


# ─────────────────────────────────────────────────────────────────────────────
# WallCrowd: sfm_core.Crowd + 壁斥力 + 近傍 cap + frozen(静止者)
# ─────────────────────────────────────────────────────────────────────────────
class WallCrowd(Crowd):
    """Crowd の屋内拡張。壁斥力・対人斥力の近傍 cap・静止個体(frozen)を追加する。

    frozen 個体(静止者 = 斥力源 + 接触対象だが動かない)は active=True のまま(=他個体へ斥力を
    及ぼし距離計算に入る)、step で速度 0・不動に固定する。Crowd.active(False=力を及ぼさない)とは
    別概念(静止者は「見える・押すが動かない」)。
    """

    def __init__(self, pos, vel, goal, v0, radius, walls=None, frozen=None,
                 wall_a=WALL_A, wall_b=WALL_B, neighbor_cap=NEIGHBOR_CAP,
                 arrive_radius=0.5, **kw):
        super().__init__(pos, vel, goal, v0, radius=radius,
                         arrive_radius=arrive_radius, **kw)
        n = self.pos.shape[0]
        self.frozen = (np.zeros(n, dtype=bool) if frozen is None
                       else np.asarray(frozen, dtype=bool).reshape(-1).copy())
        self.wall_a = float(wall_a)
        self.wall_b = float(wall_b)
        self.neighbor_cap = int(neighbor_cap)
        # 壁セグメントを (M,2) の端点配列へ(空なら壁斥力オフ)。
        segs = list(walls or [])
        if segs:
            self._wp1 = np.array([[s[0][0], s[0][1]] for s in segs], dtype=np.float64)
            self._wp2 = np.array([[s[1][0], s[1][1]] for s in segs], dtype=np.float64)
        else:
            self._wp1 = self._wp2 = None

    def forces(self):
        """駆動 + 対人斥力(近傍 cap)+ 壁斥力。非アクティブは 0。決定論(乱数ゼロ)。"""
        n = self.pos.shape[0]
        e, _ = self._desired_dir()
        # 駆動項: f_drive = m (v0 e − v) / τ(sfm_core.Crowd と同式)
        f = self.mass * (self.v0[:, None] * e - self.vel) / self.tau

        if n > 1:
            # 対人斥力(全ペア)。diff_ij = pos_i − pos_j(j から i へ向かう)
            diff = self.pos[:, None, :] - self.pos[None, :, :]     # (N,N,2)
            d = np.linalg.norm(diff, axis=2)                       # (N,N)
            np.fill_diagonal(d, np.inf)
            rr = self.radius[:, None] + self.radius[None, :]
            arg = np.clip((rr - d) / self.b, a_min=None, a_max=_EXP_ARG_MAX)
            mag = self.a * np.exp(arg)
            with np.errstate(invalid="ignore", divide="ignore"):
                nij = diff / d[:, :, None]
            nij = np.nan_to_num(nij)
            cosphi = -np.einsum("ik,ijk->ij", e, nij)
            w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
            valid = self.active[None, :] & (d <= rr + CUTOFF_M)
            # 近傍 cap: 各個体につき最近傍 neighbor_cap 体のみ寄与。距離昇順の安定ソートで順位
            # を作り(id 昇順=配列 index 昇順でタイブレーク)、cap 位以内だけ残す=決定論選択。
            if self.neighbor_cap < n - 1:
                order = np.argsort(d, axis=1, kind="stable")       # 距離昇順の列 index
                rank = np.argsort(order, axis=1, kind="stable")    # 各 j の順位
                valid = valid & (rank < self.neighbor_cap)
            contrib = (w * mag)[:, :, None] * nij
            contrib[~valid] = 0.0
            f = f + contrib.sum(axis=1)

        # 壁斥力 f_iW = A·exp((r_i − d_iW)/B)·n_iW(法線=壁→個体)。接触項(body/friction)は省略。
        if self._wp1 is not None:
            f = f + self._wall_forces()

        f[~self.active] = 0.0
        return f

    def _wall_forces(self):
        """各個体への壁斥力の合力 (N,2)。壁セグメントへの最近点距離と法線で計算(ベクトル化)。"""
        pos = self.pos                                   # (N,2)
        p1, p2 = self._wp1, self._wp2                    # (M,2)
        eseg = p2 - p1                                   # (M,2)
        len2 = (eseg * eseg).sum(axis=1)                 # (M,)
        len2_safe = np.where(len2 > 1e-12, len2, 1.0)
        rel = pos[:, None, :] - p1[None, :, :]           # (N,M,2)
        t = (rel * eseg[None, :, :]).sum(axis=2) / len2_safe[None, :]   # (N,M)
        t = np.clip(t, 0.0, 1.0)
        closest = p1[None, :, :] + t[:, :, None] * eseg[None, :, :]     # (N,M,2)
        dvec = pos[:, None, :] - closest                 # (N,M,2) 壁→個体
        dist = np.linalg.norm(dvec, axis=2)              # (N,M)
        with np.errstate(invalid="ignore", divide="ignore"):
            nvec = dvec / dist[:, :, None]
        nvec = np.nan_to_num(nvec)
        arg = np.clip((self.radius[:, None] - dist) / self.wall_b,
                      a_min=None, a_max=_EXP_ARG_MAX)
        mag = self.wall_a * np.exp(arg)                  # (N,M)
        mag = np.where(dist <= self.radius[:, None] + _WALL_CUTOFF_M, mag, 0.0)
        return (mag[:, :, None] * nvec).sum(axis=1)      # (N,2)

    def step(self, dt=0.1):
        """1 サブステップ Euler。frozen 個体は速度 0・不動(静止者=見える/押すが動かない)。"""
        f = self.forces()
        acc = f / self.mass
        v_new = self.vel + acc * dt
        speed = np.linalg.norm(v_new, axis=1)
        over = speed > self.v_max
        if np.any(over):
            v_new[over] *= (self.v_max[over] / speed[over])[:, None]
        v_new[~self.active] = 0.0
        v_new[self.frozen] = 0.0                         # 静止者は不動(斥力源には残る)
        self.vel = v_new
        self.pos = self.pos + v_new * dt
        return self.pos


# ─────────────────────────────────────────────────────────────────────────────
# 積分エントリ(純関数): 1回の空間遷移の軌跡サンプル + 接触ペア記録
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class IndoorParams:
    """遷移駆動 SFM の積分パラメータ(conf 化前提。既定は屋内 10 分粒度の低密度遷移向け)。"""
    dt: float = 0.2                 # サブステップ [s]
    max_substeps: int = 900         # 積分の上限サブステップ数(=180s。遷移1回の安全上限)
    sample_interval: int = 5        # 軌跡の等間隔サブサンプル間隔(5*0.2=1s ごと)
    r_contact: float = 1.2          # 接触判定の中心間距離 [m]。近接遭遇=パーソナルスペース
    #                                 (proxemics の personal distance ~0.45–1.2m)の上端。
    #                                 中心間 1.2m ≈ 体表間 ~0.6m の「すれ違い・同席」を遭遇とみなす。
    t_sub_contact: int = 3          # 接触成立に要する連続サブステップ数(=0.6s 以上・瞬間ブラシ除外)
    neighbor_cap: int = NEIGHBOR_CAP
    wall_a: float = WALL_A
    wall_b: float = WALL_B
    arrive_radius: float = 0.5      # waypoint 到達判定半径 [m]
    bfs_cell: float = 0.5           # BFS フォールバック格子 [m]


# 遷移積分の結果(repr 比較で決定論検証できるよう平明な tuple/list 構造で返す)。
TransitionResult = namedtuple("TransitionResult", ["samples", "contacts", "n_substeps"])


def integrate_transition(layout, walls, doors, movers, bystanders, params=None):
    """1 回の空間遷移について移動者の軌跡を積分し (軌跡サンプル, 接触ペア, サブステップ数) を返す。

    movers:     [{agent_id, src_zone, dst_zone(int or "core"), start?=(x,y)}]。start 省略時は
                src_zone の中心。route_waypoints で goal 連鎖を作り順に追従する。
    bystanders: [{agent_id, pos=(x,y)}]。静止者=斥力源 + 接触対象としてのみ参加(動かさない)。
    返り値 TransitionResult:
      samples:  [(agent_id, t_s, x, y)] 移動者の等間隔サブサンプル軌跡(t 昇順→id 昇順)。
      contacts: [(id_a, id_b, kind, duration_s)] kind∈{passing, coplace}・id_a<id_b・
                (id_a,id_b) 昇順。距離 < r_contact が連続 t_sub_contact 以上のペア。
    決定論: 全参加者を agent_id 昇順の固定配列に並べ、積分内で乱数を一切引かない。
    """
    p = params or IndoorParams()

    # ── 参加者を agent_id 昇順の固定配列へ(移動者 + 静止者) ──
    recs = []
    for m in movers:
        aid = int(m["agent_id"])
        # 明示 route が与えられればそれを使う(呼び出し側が経路を既に持つ場合・テスト用)。
        # 無ければ src_zone/dst_zone から route_waypoints で生成(直交間取りの標準経路)。
        explicit = m.get("route")
        if explicit is not None:
            route = [tuple(w) for w in explicit]
            start = tuple(m["start"]) if m.get("start") is not None else (
                route[0] if route else (0.0, 0.0))
        else:
            start = (tuple(m["start"]) if m.get("start") is not None
                     else _zone_center(layout, m["src_zone"]))
            route = route_waypoints(layout, doors, m["src_zone"], m["dst_zone"],
                                    start=start, walls=walls, cell=p.bfs_cell)
        recs.append({"id": aid, "pos": start, "mover": True, "route": route, "wp": 0,
                     "done": len(route) == 0})
    for b in bystanders:
        recs.append({"id": int(b["agent_id"]), "pos": tuple(b["pos"]), "mover": False,
                     "route": None, "wp": 0, "done": True})
    recs.sort(key=lambda r: r["id"])
    n = len(recs)

    pos = np.array([r["pos"] for r in recs], dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    radius = np.array([body_radius(r["id"]) for r in recs], dtype=np.float64)
    v0 = np.array([desired_speed(r["id"]) if r["mover"] else 1.0 for r in recs],
                  dtype=np.float64)
    frozen = np.array([not r["mover"] for r in recs], dtype=bool)
    goal = np.array([_first_goal(r) for r in recs], dtype=np.float64)

    crowd = WallCrowd(pos=pos, vel=vel, goal=goal, v0=v0, radius=radius, walls=walls,
                      frozen=frozen, neighbor_cap=p.neighbor_cap,
                      wall_a=p.wall_a, wall_b=p.wall_b, arrive_radius=p.arrive_radius)

    mover_idx = [i for i, r in enumerate(recs) if r["mover"]]
    samples = []
    # 接触トラッキング: (i,j)index → [連続長, 全サブステップで双方移動中か]
    ongoing = {}
    contacts = []
    ids = [r["id"] for r in recs]

    def moving_now(i):
        return recs[i]["mover"] and not recs[i]["done"]

    n_used = 0
    for k in range(1, p.max_substeps + 1):
        # ── waypoint 前進(goal 到達で次へ・最終到達で done→frozen) ──
        for i in mover_idx:
            r = recs[i]
            if r["done"]:
                continue
            gx, gy = crowd.goal[i]
            if math.hypot(crowd.pos[i, 0] - gx, crowd.pos[i, 1] - gy) < p.arrive_radius:
                r["wp"] += 1
                if r["wp"] < len(r["route"]):
                    crowd.goal[i] = r["route"][r["wp"]]
                else:
                    r["done"] = True
                    crowd.frozen[i] = True

        crowd.step(p.dt)
        n_used = k

        # ── 接触検出(全参加者ペア。中心間距離 < r_contact) ──
        cur_move = [moving_now(i) for i in range(n)]
        dm = crowd.pos[:, None, :] - crowd.pos[None, :, :]
        dd = np.linalg.norm(dm, axis=2)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                if dd[i, j] < p.r_contact:
                    key = (i, j)
                    seen.add(key)
                    both = cur_move[i] and cur_move[j]
                    if key in ongoing:
                        ongoing[key][0] += 1
                        ongoing[key][1] = ongoing[key][1] and both
                    else:
                        ongoing[key] = [1, both]
        # 途切れたペアを finalize(この step で接触していない=区間終了)
        for key in list(ongoing.keys()):
            if key not in seen:
                _finalize_contact(key, ongoing.pop(key), ids, contacts, p)

        # ── 軌跡サンプル(等間隔サブサンプル) ──
        if k % p.sample_interval == 0:
            t_s = k * p.dt
            for i in mover_idx:
                samples.append((ids[i], t_s, float(crowd.pos[i, 0]), float(crowd.pos[i, 1])))

        if all(recs[i]["done"] for i in mover_idx):
            break

    # 未 finalize の接触を締める
    for key, val in ongoing.items():
        _finalize_contact(key, val, ids, contacts, p)

    contacts.sort(key=lambda c: (c[0], c[1], c[2]))
    return TransitionResult(samples=samples, contacts=contacts, n_substeps=n_used)


def _first_goal(rec):
    if rec["mover"] and rec["route"]:
        return rec["route"][0]
    return rec["pos"]


def _finalize_contact(key, val, ids, contacts, params):
    length, all_moving = val
    if length < params.t_sub_contact:
        return
    i, j = key
    a, b = ids[i], ids[j]
    if a > b:
        a, b = b, a
    kind = "passing" if all_moving else "coplace"
    contacts.append((a, b, kind, length * params.dt))
