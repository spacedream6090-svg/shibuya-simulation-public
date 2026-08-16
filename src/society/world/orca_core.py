"""ORCA コア(竹-4 で src へ昇格)— 多方向交差流ゾーン専用の局所物理エンジン。

出自と昇格の理由
----------------
本 module は `reference/physics_bench/orca_min.py`(第79バッチの比較プロトタイプ)の移植版。
**reference 側は 1 バイトも変更していない**(比較レポートの再現性を壊さないため)。
昇格の根拠は `docs/research/physics-engine-selection.md` の「★ P2 決定(2026-08-02・確定)」節:

  既定エンジン = 自前 SFM / **多方向交差流ゾーンのみ ORCA**(ゾーン別ハイブリッド)

  | 実測 | SFM | ORCA |
  |---|---|---|
  | 交差流の速度効率 | 0.530 | **0.924** |
  | 交差流の内部立往生率 | 16.5% | 0.3% |
  | 交差流の方向反転 [回/体/s] | 0.681 | **0.139** |
  | ゲート帯 加速度 p99(guarded)[m/s²] | 9.34 | **5.15** |

アルゴリズムは原論文と参照実装に忠実:
  van den Berg, Guy, Lin & Manocha (2011) "Reciprocal n-Body Collision Avoidance",
    Robotics Research (ISRR 2009), Springer STAR 70, pp.3-19.
  参照実装 RVO2 (UNC GAMMA, Apache-2.0) の Agent.cpp:
    computeNewVelocity / linearProgram1 / linearProgram2 / linearProgram3
  https://gamma.cs.unc.edu/RVO2/
ORCA 半平面の構成(cut-off circle / left leg / right leg / collision の 4 分岐)と
3 段 LP(LP1=1直線上の区間, LP2=2D の最近点, LP3=実行不能時の距離最小化)の
数式・分岐条件・タイブレークをそのまま写してある。

reference からの**追加点**(= P2 決定の条件 3「重なり対策」)
------------------------------------------------------------
ORCA が保証するのは「**連続時間**での衝突回避」であって、離散 step の前進積分では
わずかに重なる(ベンチ実測: 重なりペア率 0.55%、最深 **−0.207 m**。SFM は接触斥力が
硬いので重なりゼロ)。連続時間の保証が離散化で破れるのは ORCA の既知の性質で、
実務では (a) 計画半径に安全マージンを載せる (b) 事後に位置を分離する、の 2 段で潰す。
本 module は両方を**決定論を壊さない形で**入れた:

  (a) **半径マージン** `radius_margin`(既定 0.05 m)
      ORCA 線・壁半平面の構成にだけ `r_plan = r_body + margin` を使う。
      身体半径 `radius` そのものは分離パス・重なり計測・描画のために保つ
      (= 「計画は太めに、判定は実寸で」)。
  (b) **事後分離パス** `separate_positions()`
      重なっているペアの検出はベクトル化、**解消は (i<j) 辞書順昇順の逐次**(Gauss-Seidel)。
      各ペアでめり込み量の半分ずつを中心線方向へ押し戻し、重なりが消えるか
      **固定上限回数**に達するまで繰り返す(上限に当たっても静かに打ち切る=無限ループ無し)。
      逐次にしたのは実測の結果: 全ペア同時補正(Jacobi)は連鎖する重なりで収束が遅く、
      16 体の密集で 8 反復では −1.2 cm 残った。ペア順が固定なので逐次でも
      **同一入力 → 同一 float64 配列**(決定論は不変)。
      速度は書き換えない(位置ベースの補正。速度まで触ると ORCA の相互性が壊れる)。
      壁へ押し込む可能性はある(次サブステップの壁半平面が押し返す)= 正直な限界。
      ★ 「速度層(ORCA)+ 位置層(決定論的 push-apart)」の二層構成は実装業界の定石で、
        Recast/Detour の `dtCrowd` が同型を採る(位置積分の**後**に押し戻しを 4 反復・
        `COLLISION_RESOLVE_FACTOR = 0.7f`・**完全重複は agent index の大小比較で
        決定論的に振り分ける**)。本 module は補正係数を落とさず(0.5+0.5 の全量補正)
        反復上限まで回す設計にした = 「収束したら gap ≥ 0」をテストで固定できる。
        退化ペア(中心が完全一致)のタイブレークは dtCrowd に倣い速度の法線+index 順に採る。
        出典: recastnavigation/DetourCrowd/Source/DetourCrowd.cpp
      ★ **正直な限界**: 反復は上限で打ち切るので、幾何的に無理な密集(例: 24 体が
        半径 0.25 m の塊に居て 0.6 m ずつ離れたい)では上限に当たって残留めり込みが出る。
        その場合の残差は返り値 `iters == max_iters` から検出できる。

  **なぜ重なりが原理的に起きるのか(仕様上の必然)**: ORCA の制約は半平面なので、
  近接する少数のエージェントだけで実行可能集合が**空**になりうる。RVO2 はそのとき
  「全制約を線形緩和して最小通過距離を最大化する速度」を返す(= 衝突を許す速度を返す)。
  すなわち **RVO2 は衝突回避を保証しない**(Ben Sunshine-Hill, "RVO and ORCA: How They
  Really Work", Game AI Pro 3 ch.19)。既に重なっている相手には
  "Collision. Project on cut-off circle of time timeStep." の枝が効くが、これは
  **次の 1 ステップで重なりを増やさない**だけで即時解消はしない(RVO2 `Agent.cc` 実測)。
  → 事後分離パスは「あれば良い」ではなく **必須**。

対称配置のデッドロックと揺らぎ
------------------------------
ORCA が保証するのは衝突回避であって**進捗ではない**。完全対称配置(円周 16 体が
対蹠点を目指す)ではベンチ実測で **0/16 到達**(最小中心間距離は 2r ちょうどに張り付き、
衝突は一度も起きない)。RVO2 本家の例題 `examples/Blocks.cc` も希望速度に微小乱数を足し、
コメントに "Perturb a little to avoid deadlocks due to perfect symmetry." と明記している
(**実測**: 方向は一様角、振幅は `0.0001F`= 単位速度の 0.01% オーダー)。
本 module も `pref_noise`(希望速度への微小ゆらぎ)を持つが、**Generator は必ず外から
受け取る**(内部 seed 禁止)。呼び出し側が用途別 named stream を渡す限り、
「同一 seed の 2 ラン → 同一軌跡(バイト一致)」は保たれる(ベンチ実測で確認済み)。
★ 既定値 0.05 [m/s] は **本家の 1e-4 より 3 桁大きい**。本家は毎ステップ方向を引き直す
  一様角の摂動、本コアは成分ごとの正規分布という違いがあり、ベンチの円周 16 体試験では
  0.01 で 0/16・0.05 で 16/16 到達だった(実測)。**未較正の実験レバー**であることを
  明記しておく(較正は P4 の仕事)。0.0 にすれば乱数を 1 本も引かない。

静的障害物(壁)の扱い — 移植で**意図的に簡略化**した点
--------------------------------------------------------
RVO2 の障害物 ORCA(`Agent.cc` 実測)は、**有向**線分 (obstacle1 → obstacle1->next_)
ごとに次を行い、線分 1 本につき ORCA 線を **高々 1 本**しか出さない:
  1. `alreadyCovered` 判定 — 既に構築済みの障害物線で覆われている速度障害物は捨てる
     ("Check if velocity obstacle of obstacle is already taken care of by previously
       constructed obstacle ORCA lines.")
  2. 衝突 3 ケース(左頂点 / 右頂点 / 線分本体)。左頂点は `isConvex_` でなければ無視、
     右頂点は隣接エッジが担当するなら無視 ("or if it will be taken care of by
     neighboring obstacle")。**障害物の衝突枝は timeStep を使わない**(cut-off circle を
     dt で作るのはエージェント間だけ)。
  3. left leg / right leg(端点への接線)。斜視では 1 頂点が両 leg を定義し、非凸頂点では
     leg が cut-off 線の延長になる。
  4. **foreign leg**: leg が隣接エッジに食い込む凸頂点では隣のエッジの cut-off へ差し替え、
     速度がその「他所の leg」に射影されたら **制約を一切追加しない**。
  5. 左右 cut-off circle / cut-off line / 左右 leg のうち**現在速度に最も近い 1 つ**へ射影。
  6. **相互責任なし**: 対エージェントは `point = v + 0.5·u`(責任折半)だが障害物線に 0.5 は
     掛からない = 壁は避けてくれない前提で全責任をエージェントが負う。
本 module はこれを移植せず、
  「壁最近点までの距離 d と法線 n(壁→個体)による半平面 `n·v ≥ (r_plan − d)/τ_obst`」
で代用する(hard constraint として LP3 の先頭に置く点は同じ)。
**直線壁の通路ではほぼ等価**だが、次の場合は本家と挙動が変わりうる(既知の限界):
  - **壁端の回り込みができない**: cut-off circle / leg 由来の「端点を回り込む速度」が
    無いので、有限線分の端で不当に禁止された速度が残る。
  - **鋭角のコーナー・複数壁の交差**: `alreadyCovered` と foreign-leg で本家が落としている
    **重複制約が全部残る**ので、実行可能領域が痩せて停止しやすい。
  - **薄い障害物**: 本家の障害物は `direction_` を持つ**向き付き輪郭で片側からしか効かない**
    が、本 module の半平面は無向で両側から効く(挟み撃ちになりうる)。
  - **非凸(内側)コーナー**: `isConvex_` の特別扱いが無いので leg が壁の中を向く枝が無い。
ORCA + 静的障害物の失敗様式そのものは文献にも報告がある: 壁を背にしたゴールでは
「北向きの速度が全部壁の静的速度障害物に塞がれてゴールに永遠に到達できない」、
コーナーでは「望みの軌道から遥かに逸れるか、そのまま詰まる … in general leads to
frequent deadlocks」(Ben Sunshine-Hill, "RVO and ORCA: How They Really Work",
Game AI Pro 3 ch.19)。同章は **参照実装が論文どおりではなく、停止距離より遠い静的障害物との
衝突を無視している**ことも指摘している(= 本家も実務上の逃げを入れている)。
→ したがって本 module は **開放平面の多方向交差流ゾーン**に用途を限定する
   (壁のある領域・通路・対向流は SFM = P2 決定の割当規則)。壁は「ゾーンの縁を
   越えさせない柵」程度の軽い用途にとどめること。

候補列挙の空間ハッシュ化(物理痩身 第一段A。2026-08-16)
--------------------------------------------------------
本家 RVO2 は `maxNeighbors` + `neighborDist` を **kd-tree** で選抜する。本 module は
cap=12 を持ちながら選抜を全体 argsort(O(N² log N))でやっていたので、
`sfm_core.knn_neighbors`(拡大リングのセル探索)へ置換した。
neighbor_dist=10 m を素直にセル化すると候補が πR²ρ で膨らむため、
**cap=12 が効いている事実**を使って「ring·cell 以内に k 体見つかった行から確定」する。
同じ空間ハッシュを (a) 事後分離パスの重なり検出 (b) `min_gap` にも使う。
いずれも **有効近傍の集合・順序・値は全ペア版と完全一致**する(`pair_hash=False` で
旧経路が残り、tests/test_physics_hash.py がバイト一致を機械固定する)。
`min_gap` だけはカットオフを持たない量なので、探索半径 R で得た最小すき間 g が
g ≤ R − 2·r_max を満たすまで R を倍にする(満たせば g が全ペアの最小そのもの)。

決定論
------
乱数は `pref_noise > 0` かつ `rng` が渡されたときだけ引く(それ以外は 1 本も引かない)。
エージェント処理順は index 昇順固定、近傍順は (距離昇順, index 昇順) 固定、
分離パスの合算順序も固定。float は float64(RVO2 C++ の float32 とのバイト一致は主張しない)。
"""
from __future__ import annotations

import math

import numpy as np

from . import sfm_core as _sfm     # PointField(点集合の空間ハッシュ)を共用する

RVO_EPSILON = 1e-5          # RVO2 の RVO_EPSILON と同値(平行判定のしきい値)

# ---- 既定パラメータ(ベンチ `reference/physics_bench` と同値。出典は上記 docstring) ----
TAU_DEFAULT = 2.0            # 対エージェント時間地平 [s](RVO2 の timeHorizon)
TAU_OBST_DEFAULT = 2.0       # 対壁時間地平 [s]
NEIGHBOR_CAP_DEFAULT = 12    # 近傍上限(SFM 側の neighbor_cap と揃える)
NEIGHBOR_DIST_DEFAULT = 10.0  # 近傍とみなす最大距離 [m]
WALL_RANGE_DEFAULT = 2.0     # 壁半平面を張る距離 [m]
V_MAX_FACTOR = 1.3           # 最高速度 = 希望速度 × 1.3(Helbing2000・SFM 側と同条件)
RADIUS_MARGIN_DEFAULT = 0.05  # 計画半径の安全マージン [m](P2 決定 条件3(a))
SEPARATION_ITERS_DEFAULT = 64  # 事後分離パスの最大反復回数(P2 決定 条件3(b))
#   実測: ORCA の離散化で生じる程度の重なり(最深 −0.21 m)を 16 体の密集配置に与えると、
#   逐次補正でも収束に 32〜64 反復要る(8 反復では −1.8 cm 残る)。重なりが無ければ
#   ループは 1 回目の検出で即 break するので、上限を上げても既定経路のコストは増えない。
SEPARATION_SLACK_M = 1e-6    # 分離後に残す極小の余白 [m](float 誤差で gap<0 に戻らないため)
MIN_GAP_SEED_M = 0.5         # min_gap のセル探索の初期余裕 [m](密な群衆なら 1 巡で確定)
PREF_NOISE_DEFAULT = 0.05    # 対称性の破れ [m/s]。RVO2 examples/Blocks.cc の実務に倣う


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
    radius / all_rad は **計画半径**(= 身体半径 + margin)を渡すこと。
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
    """壁半平面制約 n·v >= (r - d)/tau_obst を (N,M,4)+(N,M) で返す(簡略化・冒頭 docstring)。"""
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
# 事後分離パス(P2 決定 条件3(b))— 決定論を壊さない重なり解消
# ─────────────────────────────────────────────────────────────────────────────
def min_gap_dense(pos, radius) -> float:
    """全ペア triu の最小すき間(**参照実装**。min_gap とビット一致)。"""
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    iu, ju = np.triu_indices(n, k=1)
    d = np.linalg.norm(pos[iu] - pos[ju], axis=1)
    return float((d - (radius[iu] + radius[ju])).min())


def min_gap(pos, radius, pair_hash=True) -> float:
    """体表間の最小すき間 [m]。負なら重なっている(= min(d_ij − r_i − r_j))。

    2 体未満なら +inf(重なりの概念が無い)。

    ★空間ハッシュ経路でも **値は全ペア版とビット一致**(min は順序非依存・厳密):
      半径 R のセル探索で得た最小すき間 g が g ≤ R − 2·r_max を満たすなら、
      列挙されなかったペアは d > R すなわち gap > R − 2·r_max ≥ g なので、
      g は全ペアの最小そのもの。満たさないときは R を倍にして繰り返す
      (= 群衆が疎で「最も近いペアすら遠い」ときだけ探索半径が伸びる)。
    """
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    if not pair_hash or n <= _sfm._ALL_PAIRS_MAX_N:
        return min_gap_dense(pos, radius)
    pos = np.asarray(pos, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    rr_max = 2.0 * float(radius.max())
    span = float(max(pos[:, 0].max() - pos[:, 0].min(),
                     pos[:, 1].max() - pos[:, 1].min()))
    reach = rr_max + MIN_GAP_SEED_M
    while True:
        ii, jj = _sfm.neighbor_pairs(pos, reach)
        up = ii < jj                            # triu と同じペア集合(min は対称)
        ii, jj = ii[up], jj[up]
        if ii.shape[0]:
            d = np.linalg.norm(pos[ii] - pos[jj], axis=1)
            g = float((d - (radius[ii] + radius[jj])).min())
            if g <= reach - rr_max:
                return g                        # 未列挙のペアは必ずこれより大きい
        if reach >= span + rr_max:              # もう全ペアを覆っている
            return min_gap_dense(pos, radius)
        reach *= 2.0


def _overlap_pairs(out, radius, slack, pair_hash):
    """重なりペア (i<j) を **辞書順昇順**で返す + 各ペアの必要間隔 rr。

    全ペア triu 経路と **同じ集合・同じ順序**(候補は上位集合 → dist < rr で厳密に絞る)。
    rr の計算順序も (r_i + r_j) + slack で同一なのでビット一致。
    """
    n = out.shape[0]
    if pair_hash and n > _sfm._ALL_PAIRS_MAX_N:
        reach = 2.0 * float(radius.max()) + float(slack)
        # reach が体径ぶん(~0.7 m)しかない = 候補は元々少ないので、格子は粗く(cell=reach・
        # リング 1 周 = 9 セル)して**問い合わせ回数**を減らす方が速い。分離パスは 1 サブ
        # ステップで最大 64 回呼ばれるため、ここの定数が効く(結果は cell に依存しない)。
        ii, jj = _sfm.neighbor_pairs(out, reach, cell=reach)
        up = ii < jj
        ii, jj = ii[up], jj[up]
    else:
        ii, jj = np.triu_indices(n, k=1)
    if ii.shape[0] == 0:
        return ii, jj, np.zeros(0), np.zeros(0)
    rr = radius[ii] + radius[jj] + float(slack)
    dist = np.linalg.norm(out[ii] - out[jj], axis=1)
    hit = dist < rr
    return ii[hit], jj[hit], rr[hit], dist[hit]


def separate_positions(pos, radius, max_iters=SEPARATION_ITERS_DEFAULT,
                       slack=SEPARATION_SLACK_M, vel=None, pair_hash=True):
    """重なりを **決定論的に** 解消する位置ベースの事後パス(in-place ではない)。

    重なりペア (i<j) を辞書順昇順に列挙し、めり込み量 (r_i+r_j+slack − d) の半分ずつを
    中心線方向へ相互に押し戻す。全ペアを同時に補正して 1 反復とし、重なりが消えるか
    `max_iters` 回に達するまで繰り返す(連鎖する重なりのため)。

    Returns: (new_pos, iters_used, moved_max_m)
      iters_used == 0 は「最初から重なりゼロ = 1 mm も動かしていない」を意味する。
    ★ 合算は np.bincount(index 昇順の逐次加算)= 同一入力 → 同一 float64 配列。
    ★ 速度は触らない(位置ベース補正)。壁へ押し込む可能性は残る(次サブステップの
      壁項が押し返す)= 冒頭 docstring の限界のとおり。
    ★ 中心が完全一致(d≈0)する退化ペアのタイブレークは dtCrowd に倣う: 速度がある側の
      **法線**へ押す(i<j 固定なので符号も一意)。速度も 0 なら +x(決定論的な既定方向)。
    """
    n = pos.shape[0]
    out = np.array(pos, dtype=np.float64, copy=True)
    if n < 2 or max_iters <= 0:
        return out, 0, 0.0
    radius = np.asarray(radius, dtype=np.float64)
    vv = (np.asarray(vel, dtype=np.float64).reshape(-1, 2)
          if vel is not None else None)
    start = out.copy()
    used = 0
    for _ in range(int(max_iters)):
        # ★検出だけ空間ハッシュ化(候補は上位集合 → dist < rr で厳密に絞る)。
        #   重なりペアは体径 + slack 以内にしか居ないので、探索半径は 2·r_max + slack。
        iu, ju, rr, _d = _overlap_pairs(out, radius, slack, pair_hash)
        if iu.size == 0:
            break
        used += 1
        # 退化ペア用の決定論的な押し出し方向(i の速度の法線 → 無ければ +x)
        if vv is not None:
            perp = np.stack([-vv[iu, 1], vv[iu, 0]], axis=1)
            plen = np.linalg.norm(perp, axis=1)
            fallback = np.where(plen[:, None] > 1e-12,
                                perp / np.maximum(plen, 1e-12)[:, None],
                                np.array([1.0, 0.0]))
        else:
            fallback = np.tile(np.array([1.0, 0.0]), (iu.shape[0], 1))
        # ★ 検出はベクトル化・**解消は逐次(Gauss-Seidel)**。全ペア同時補正(Jacobi)は
        #   連鎖する重なりで収束が遅く、実測で「16 体の密集で 8 反復では −1.2 cm 残る」
        #   (32 反復でようやく −2.5 µm)。逐次なら数反復で消える。ペア順は (i,j) 昇順に
        #   固定してあるので、逐次でも **同一入力 → 同一 float64 配列**(決定論は不変)。
        for p in range(iu.shape[0]):
            i, j = int(iu[p]), int(ju[p])
            dx = out[i, 0] - out[j, 0]
            dy = out[i, 1] - out[j, 1]
            d = math.hypot(dx, dy)
            need = float(rr[p])
            if d >= need:
                continue                                # 直前のペア解消で既に離れた
            if d <= 1e-12:
                ux, uy = float(fallback[p, 0]), float(fallback[p, 1])
            else:
                ux, uy = dx / d, dy / d
            half = 0.5 * (need - d)
            out[i, 0] += half * ux
            out[i, 1] += half * uy
            out[j, 0] -= half * ux
            out[j, 1] -= half * uy
    moved_max = float(np.linalg.norm(out - start, axis=1).max()) if used else 0.0
    return out, used, moved_max


# ─────────────────────────────────────────────────────────────────────────────
# ORCA シミュレータ
# ─────────────────────────────────────────────────────────────────────────────
class OrcaCrowd:
    """ORCA による 1 サブステップの速度計算 + 前進積分 + 事後分離。

    Args:
        pos/vel: (N,2) 位置・速度 [m], [m/s]
        goal:    (N,2) 目標点 [m]
        v0:      (N,)  希望速度 [m/s]
        radius:  (N,)  **身体**半径 [m](計画には radius+radius_margin を使う)
        walls:   [((x1,y1),(x2,y2)), ...] 壁線分(簡略半平面制約。冒頭 docstring の限界)
        tau / tau_obst: 対エージェント / 対壁 時間地平 [s]
        neighbor_cap / neighbor_dist: 近傍上限 / 近傍距離 [m]
        radius_margin: 計画半径の安全マージン [m](P2 決定 条件3(a))
        separation_iters: 事後分離パスの最大反復(0 で無効化=reference と同一挙動)
        pref_noise: 希望速度への微小ゆらぎ [m/s](対称性の破れ)。>0 なら rng 必須
        rng: numpy.random.Generator(**外から渡す**。内部 seed 禁止)
    """

    def __init__(self, pos, vel, goal, v0, radius, walls=None,
                 neighbor_cap=NEIGHBOR_CAP_DEFAULT, tau=TAU_DEFAULT,
                 tau_obst=TAU_OBST_DEFAULT, neighbor_dist=NEIGHBOR_DIST_DEFAULT,
                 wall_range=WALL_RANGE_DEFAULT, v_max_factor=V_MAX_FACTOR,
                 arrive_radius=0.5, pref_noise=0.0, rng=None,
                 radius_margin=RADIUS_MARGIN_DEFAULT,
                 separation_iters=SEPARATION_ITERS_DEFAULT,
                 pair_hash=True):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2).copy()
        self.vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2).copy()
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2).copy()
        self.v0 = np.asarray(v0, dtype=np.float64).reshape(-1).copy()
        self.radius = np.asarray(radius, dtype=np.float64).reshape(-1).copy()
        self.radius_margin = float(radius_margin)
        self.r_plan = self.radius + self.radius_margin
        self.v_max = v_max_factor * self.v0
        self.neighbor_cap = int(neighbor_cap)
        self.tau = float(tau)
        self.tau_obst = float(tau_obst)
        self.neighbor_dist = float(neighbor_dist)
        self.wall_range = float(wall_range)
        self.arrive_radius = float(arrive_radius)
        self.separation_iters = int(separation_iters)
        # 近傍選抜・重なり検出・min_gap の候補列挙を空間ハッシュで行うか。
        # False = 全ペア(検証用の参照経路)。**結果は両者でビット一致**。
        self.pair_hash = bool(pair_hash)
        segs = list(walls or [])
        if segs:
            self._wp1 = np.array([[s[0][0], s[0][1]] for s in segs], dtype=np.float64)
            self._wp2 = np.array([[s[1][0], s[1][1]] for s in segs], dtype=np.float64)
        else:
            self._wp1 = self._wp2 = None
        self.pref_noise = float(pref_noise)
        self.rng = rng
        # 周期境界用のゴースト(位置・速度・半径)。空なら通常。
        self.ghost_pos = np.zeros((0, 2))
        self.ghost_vel = np.zeros((0, 2))
        self.ghost_radius = np.zeros(0)
        # 直近 step の分離パスの実績(観測・テスト用)
        self.last_sep_iters = 0
        self.last_sep_moved_m = 0.0

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

    def step(self, dt=0.05):
        n = self.pos.shape[0]
        if n == 0:
            self.last_sep_iters = 0
            self.last_sep_moved_m = 0.0
            return self.pos
        # 実体 + ゴーストの結合配列(近傍探索の対象)。実体は先頭 n。
        all_pos = np.concatenate([self.pos, self.ghost_pos], axis=0)
        all_vel = np.concatenate([self.vel, self.ghost_vel], axis=0)
        all_rad = np.concatenate([self.r_plan,
                                  self.ghost_radius + self.radius_margin], axis=0)
        m = all_pos.shape[0]

        # 近傍(距離昇順・index 昇順タイブレーク = 決定論)
        k = min(self.neighbor_cap, max(m - 1, 1))
        if self.pair_hash and m > _sfm._ALL_PAIRS_MAX_N:
            # ★セル法の拡大リング探索。全体 argsort(O(N² log N))を廃す。
            #   有効近傍の集合も順序も全ペア版と完全一致する(sfm_core.knn_neighbors)。
            #   無効枠の index は 0 で埋める(build_agent_lines の該当要素は
            #   nb_valid=False で必ず捨てられるので、何を入れても結果に影響しない)。
            order, nb_valid = _sfm.knn_neighbors(self.pos, all_pos, k,
                                                 self.neighbor_dist)
        else:
            diff = self.pos[:, None, :] - all_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist[np.arange(n), np.arange(n)] = np.inf
            order = np.argsort(dist, axis=1, kind="stable")[:, :k]
            nb_valid = np.take_along_axis(dist, order, axis=1) < self.neighbor_dist

        a_lines, a_valid = build_agent_lines(self.pos, self.vel, self.r_plan,
                                             all_pos, all_vel, all_rad,
                                             order, nb_valid, self.tau, dt)
        o_lines, o_valid = wall_lines(self.pos, self.r_plan, self._wp1, self._wp2,
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
        # ---- 事後分離パス(P2 決定 条件3(b)。iters=0 なら何もしない=reference と同一)----
        if self.separation_iters > 0:
            self.pos, self.last_sep_iters, self.last_sep_moved_m = separate_positions(
                self.pos, self.radius, max_iters=self.separation_iters, vel=self.vel,
                pair_hash=self.pair_hash)
        else:
            self.last_sep_iters = 0
            self.last_sep_moved_m = 0.0
        return self.pos

    def arrived(self):
        d = np.linalg.norm(self.goal - self.pos, axis=1)
        return d < self.arrive_radius

    def min_gap(self) -> float:
        """体表間の最小すき間 [m](負 = 重なり)。分離パス後は >= 0 が期待値。"""
        return min_gap(self.pos, self.radius, pair_hash=self.pair_hash)
