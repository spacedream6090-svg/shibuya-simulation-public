"""最小 Social Force Model (SFM) コア — 開放領域の微視歩行者物理。

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
        対壁斥力  f_iW = A_w·exp((r_i − d_iW)/B_w)·n_iW   … 壁→歩行者の法線方向
  詳細と出典アクセス日は docs/research/social-force-crowd.md §1 を参照。

────────────────────────────────────────────────────────────────────────
対壁斥力 f_iW(竹-3 バッチで追加。docs/research/physics-engine-selection.md「P2 決定」):
  壁は 2D 線分集合 walls=[((x1,y1),(x2,y2)), …] として **任意引数** で受け取る。
    walls=None(既定)= 壁項そのものが 1 度も評価されない = 従来の開放領域コアと
    **完全後方互換**(既存の呼び出しは 1 バイトも挙動が変わらない)。
  最近点探索は一様格子の空間ハッシュ(WallField)で行う(N×M の全ペア走査をしない)。
    ハッシュは「距離 ≤ reach の壁を必ず含む」上位集合を返す(AABB をまたぐセル登録 +
    reach/cell 分のリング探索)ので、結果は全ペア版と **完全に同値**(ビット一致)。
    テスト tests/test_sfm_walls.py がこの同値性を全ペア参照実装との比較で固定する。
    ★正直な例外: 壁が探索セル数より少ない極小ジオメトリ(既定で M ≤ 9)では、リング探索より
      全ペアの方が安いので自動でそちらへ落ちる(WallField.pairs の安全弁)。結果は同一で、
      O(N·M) が問題になる規模(壁が多い=地下街・通路網)では必ずハッシュ経路を通る。
  最近傍1本ではなく閾値距離(既定 r_i + 2.0 m)以内の **全壁セグメントの合力**(Helbing 標準)。
  合算順序は (個体 index, 壁 index) 昇順に固定(np.bincount の逐次加算)= 決定論。

本コアで【意図的に省略した項】:
  (1) 物理接触項 — body force  k·g(r_ij−d_ij)·n_ij(弾性反発)と
      sliding friction  κ·g(r_ij−d_ij)·Δv_t·t_ij(接線摩擦)。
      (Helbing2000 のパニック弾性/摩擦。k=1.2e5, κ=2.4e5)。壁側の接触項も同様に省略。
      → 理由: これらは高密度の押し合い・すり抜け防止(体の非圧縮)を担う項で、
        本用途(低〜中密度の「揉まれ方」= 減速・回避・レーン形成の再現)には不要。
  ∴ 本コアは超高密度(LOS F 相当)の圧潰・faster-is-slower は再現対象外。
    社会的斥力 A·exp(...) は接触で有限値 A に留まるため、稀な深い重なりは
    速度上限クリップ(v_max)で数値的に抑える(接触項の代替ではない)。
────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────
対人ペアの候補列挙(物理痩身 第一段A。2026-08-16。docs/research/crowd-attention-physics.md):
  対人斥力には元から距離カットオフ(体表間 2.0 m)があるのに、候補列挙だけが O(N²) だった。
  壁項の `WallField` と**同型**の空間ハッシュ `PointField` を点集合に張り、
  「候補 = カットオフ内を必ず含む上位集合 → 正確な距離判定は従来コード」へ置換した。
    - 候補は (i, j) 辞書順に整列してから合算する(np.bincount)。現行の `contrib.sum(axis=1)`
      は軸 1 が最内周でないため**逐次加算**なので、両者の和は厳密に一致する。
    - 近傍 cap の選抜も「候補 → (距離, index) 昇順で k 選抜」へ置換。行内の候補が既に
      j 昇順なので、距離の安定ソートが現行 argsort(kind="stable") のタイブレークを再現する。
  → 力・速度・位置は **1 ビットも変わらない**(pair_hash=False で旧全ペア経路が残る。
    tests/test_physics_hash.py が両経路のバイト一致を密度 3 水準×形状 3 種×seed 3 本で固定)。
────────────────────────────────────────────────────────────────────────

決定論について:
  乱数は numpy.random.Generator を引数で受け取り、コア内部では seed しない。
  揺らぎ項 ξ(既定 noise=0.0 で無効)を有効化したときだけ Generator を消費する。
  noise=0.0 のときコアは完全に決定論的(同一入力→同一 float 配列)。
  noise>0 でも「同一 seed の Generator を渡せば同一軌跡」= 決定論は seed 固定で保たれる。
  ★ ξ は forces() の **唯一の合成点**で足す。屋内拡張(indoor_flow.WallCrowd)が
    forces() を上書きして ξ を落としていたバグ(壁ありで noise 完全無効)は竹-3 で解消
    した = 壁・近傍 cap は本コアの引数になり、派生クラスは forces() を上書きしない。
"""
from __future__ import annotations

import math

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

# ── 対壁斥力 f_iW の確定パラメータ(Helbing, Farkas & Vicsek 2000 の壁項) ──
WALL_A_DEFAULT = 2000.0  # 壁斥力の強さ A_w [N](対人 A と同値の標準値)
WALL_B_DEFAULT = 0.08    # 壁斥力の特性距離 B_w [m]
WALL_RANGE_M = 2.0       # 壁斥力の作用距離 [m](d_iW > r_i + これ で寄与 0)
_MAX_CELLS_PER_SEG = 4096  # 1 セグメントが占める格子セル数の上限(超えた壁は常時候補へ退避)
_CELL_CLIP = 1 << 30     # 格子座標のクリップ(int64 キー i*2^32+j の桁溢れ防止)
_KEY_STRIDE = 1 << 32


class WallField:
    """壁セグメント集合 + 一様格子の空間ハッシュ(最近点探索の全ペア走査を避ける)。

    walls: [((x1,y1),(x2,y2)), …](2D 線分の列。vision.building_walls と同形式)。

    格子は「各セグメントの AABB が重なるセル全部」に seg index を登録する。点 p から
    距離 R 以内に最近点 c を持つセグメントは必ず c を含むセルに登録されており、そのセルは
    p のセルから Chebyshev 距離 ceil(R/cell) 以内にある。したがってリング探索
    (2·ring+1)² セルの合併は **距離 R 以内の全セグメントの上位集合**になる(取りこぼし無し)。
    → 空間ハッシュ版と全ペア版の力は(寄与 0 の余分な候補が混じるだけで)完全に一致する。

    AABB が巨大なセグメント(_MAX_CELLS_PER_SEG 超)は格子に載せず「常時候補」に退避する
    (登録セル数の爆発を防ぐ安全弁。正しさは変わらない=上位集合のまま)。
    """

    __slots__ = ("p1", "p2", "n_seg", "cell", "_ox", "_oy", "_keys", "_start",
                 "_segs", "_always")

    def __init__(self, walls, cell=None):
        segs = [((float(s[0][0]), float(s[0][1])), (float(s[1][0]), float(s[1][1])))
                for s in (walls or [])]
        self.n_seg = len(segs)
        if self.n_seg == 0:
            raise ValueError("WallField needs at least one wall segment")
        self.p1 = np.array([[s[0][0], s[0][1]] for s in segs], dtype=np.float64)
        self.p2 = np.array([[s[1][0], s[1][1]] for s in segs], dtype=np.float64)
        self.cell = float(cell if cell else (WALL_RANGE_M + RADIUS_MAX))
        if not (self.cell > 0.0):
            raise ValueError("cell must be > 0")

        lo = np.minimum(self.p1, self.p2)
        hi = np.maximum(self.p1, self.p2)
        self._ox = float(lo[:, 0].min())
        self._oy = float(lo[:, 1].min())
        i0 = np.floor((lo[:, 0] - self._ox) / self.cell).astype(np.int64)
        i1 = np.floor((hi[:, 0] - self._ox) / self.cell).astype(np.int64)
        j0 = np.floor((lo[:, 1] - self._oy) / self.cell).astype(np.int64)
        j1 = np.floor((hi[:, 1] - self._oy) / self.cell).astype(np.int64)
        n_cells = (i1 - i0 + 1) * (j1 - j0 + 1)
        big = n_cells > _MAX_CELLS_PER_SEG
        self._always = np.nonzero(big)[0].astype(np.int64)   # 常時候補(seg index 昇順)

        keys, owners = [], []
        for m in np.nonzero(~big)[0]:
            ii = np.arange(i0[m], i1[m] + 1, dtype=np.int64)
            jj = np.arange(j0[m], j1[m] + 1, dtype=np.int64)
            k = (ii[:, None] * _KEY_STRIDE + jj[None, :]).ravel()
            keys.append(k)
            owners.append(np.full(k.shape[0], m, dtype=np.int64))
        if keys:
            key_all = np.concatenate(keys)
            own_all = np.concatenate(owners)
            order = np.lexsort((own_all, key_all))           # key 昇順・同 key 内は seg 昇順
            key_all, own_all = key_all[order], own_all[order]
            uniq, start = np.unique(key_all, return_index=True)
            self._keys = uniq
            self._start = np.append(start, key_all.shape[0]).astype(np.int64)
            self._segs = own_all
        else:
            self._keys = np.zeros(0, dtype=np.int64)
            self._start = np.zeros(1, dtype=np.int64)
            self._segs = np.zeros(0, dtype=np.int64)

    # ── (個体, 壁)候補ペアの列挙 ──────────────────────────────────────────
    def pairs(self, pos, reach):
        """距離 reach 以内の壁を必ず含む候補ペア (agent_idx, seg_idx) を返す。

        戻り値は (個体 index, 壁 index) の辞書順昇順・重複なし(合算順序=決定論)。"""
        n = pos.shape[0]
        if n == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        ring = int(math.ceil(max(float(reach), 0.0) / self.cell))
        # 安全弁: 探索リングが壁本数より大きくなる(=格子が細かすぎる/作用距離が極端)なら
        # 全ペアの方が安く、結果も同一(候補は上位集合なので力はビット一致)。
        n_off = (2 * ring + 1) ** 2
        if n_off >= self.n_seg or n_off * n > 4_000_000:
            return self.all_pairs(n)
        ai = np.clip(np.floor((pos[:, 0] - self._ox) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)
        aj = np.clip(np.floor((pos[:, 1] - self._oy) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)

        keys_out = []
        if self._keys.shape[0]:
            offs = np.arange(-ring, ring + 1, dtype=np.int64)
            di = np.repeat(offs, offs.shape[0])
            dj = np.tile(offs, offs.shape[0])
            # (n, n_off) の問い合わせキー → 一括 searchsorted
            qk = ((ai[:, None] + di[None, :]) * _KEY_STRIDE
                  + (aj[:, None] + dj[None, :]))
            agent_q = np.repeat(np.arange(n, dtype=np.int64), di.shape[0])
            qk = qk.ravel()
            idx = np.searchsorted(self._keys, qk)
            hit = (idx < self._keys.shape[0])
            hit[hit] = self._keys[idx[hit]] == qk[hit]
            idx = idx[hit]
            agent_q = agent_q[hit]
            cnt = (self._start[idx + 1] - self._start[idx])
            total = int(cnt.sum())
            if total:
                rep = np.repeat(np.arange(cnt.shape[0], dtype=np.int64), cnt)
                offs_in = (np.arange(total, dtype=np.int64)
                           - np.repeat(np.cumsum(cnt) - cnt, cnt))
                seg = self._segs[self._start[idx][rep] + offs_in]
                keys_out.append(agent_q[rep] * self.n_seg + seg)
        if self._always.shape[0]:
            keys_out.append((np.arange(n, dtype=np.int64)[:, None] * self.n_seg
                             + self._always[None, :]).ravel())
        if not keys_out:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        # 重複除去 + 昇順(np.unique は昇順 = (個体, 壁) の辞書順)
        flat = np.unique(np.concatenate(keys_out))
        return (flat // self.n_seg, flat % self.n_seg)

    def all_pairs(self, n):
        """全ペア (agent_idx, seg_idx) 昇順(空間ハッシュとの同値検証・小規模向け参照実装)。"""
        a = np.repeat(np.arange(n, dtype=np.int64), self.n_seg)
        s = np.tile(np.arange(self.n_seg, dtype=np.int64), n)
        return (a, s)


# ─────────────────────────────────────────────────────────────────────────────
# 点集合の空間ハッシュ(PointField)— 対人ペアの候補列挙を O(N²) から外す
# ─────────────────────────────────────────────────────────────────────────────
# 設計は WallField と**同型**(docs/research/crowd-attention-physics.md 案A):
#   「セル法で候補 = 上位集合を出す → 正確な距離判定は従来コードのまま」。
#   全力項には既に距離カットオフがある(対人 rr+2.0 m・far rr+4.725 m・知覚 2.0 m・
#   ORCA 10 m)ので、カットオフ外の寄与は**元々ちょうど 0.0**。候補が上位集合であれば
#   落ちるのは +0.0 の加算だけで、x + 0.0 == x(IEEE754・厳密)。
#
# ★ビット一致の要(3 条件。tests/test_physics_hash.py が機械固定する)
#   (a) 候補は「カットオフ内を必ず含む」上位集合 = 実効近傍集合が完全一致
#   (b) neighbor_cap 選抜のタイブレークまで一致
#       (現行 argsort(kind="stable") の (距離, index) 昇順を再現)
#   (c) 加算順序が現行と同一 — 候補を (i, j) 辞書順に整列してから合算する。
#       現行の `contrib.sum(axis=1)` は **軸 1 が最内周でないため逐次加算**であり
#       (numpy は最内周の連続軸を縮約するときだけ pairwise summation を使う)、
#       `np.bincount`(index 昇順の逐次加算)と厳密に同じ和になる。
#       ★-0.0 の縮退も起きない: bincount の初期値は +0.0 で、0.0 + (−0.0) = +0.0。
#       全ペア版の行和も自己ペア(+0.0)を必ず含むので、どちらも −0.0 にはならない。
_POINT_CELL_DIV = 2       # 格子辺 = reach / これ(リング探索 (2·div+1)² セル)。実測で最速域。
_ALL_PAIRS_MAX_N = 48     # これ以下は全ペアの方が安い(結果は同一 = 上位集合のまま)
_PAIR_BLOCK = 1 << 20     # 1 ブロックあたりの候補ペア上限(行境界で切る = 加算順序は不変)


class PointField:
    """点集合 + 一様格子の空間ハッシュ(対人ペアの候補列挙)。`WallField` の点版。

    格子は各点を 1 セルに登録する。点 p から距離 R 以内の点は必ず p のセルから
    Chebyshev 距離 ceil(R/cell) 以内のセルに居るので、リング探索 (2·ring+1)² セルの
    合併は **距離 R 以内の全点の上位集合**になる(取りこぼし無し)。

    セル内の登録順は **点 index 昇順**(`kind="stable"` の argsort)。
    """

    __slots__ = ("pos", "n", "cell", "_ox", "_oy", "_keys", "_start", "_order")

    def __init__(self, pos, cell):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
        self.n = self.pos.shape[0]
        self.cell = float(cell)
        if not (self.cell > 0.0):
            raise ValueError("cell must be > 0")
        if self.n == 0:
            self._ox = self._oy = 0.0
            self._keys = np.zeros(0, dtype=np.int64)
            self._start = np.zeros(1, dtype=np.int64)
            self._order = np.zeros(0, dtype=np.int64)
            return
        self._ox = float(self.pos[:, 0].min())
        self._oy = float(self.pos[:, 1].min())
        key = self._cell_key(self.pos)
        order = np.argsort(key, kind="stable")      # 同一セル内は点 index 昇順
        ks = key[order]
        uniq, start = np.unique(ks, return_index=True)
        self._order = order
        self._keys = uniq
        self._start = np.append(start, ks.shape[0]).astype(np.int64)

    def _cell_key(self, p):
        ci = np.clip(np.floor((p[:, 0] - self._ox) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)
        cj = np.clip(np.floor((p[:, 1] - self._oy) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)
        return ci * _KEY_STRIDE + cj

    # ── 候補列挙(行ごとにまとまるが、行内の点 index は昇順ではない)────────── #
    def candidates(self, qpos, ring):
        """(query 行 index, 点 index) の候補列。ring セル分のリング探索。"""
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1, 2)
        nq = qpos.shape[0]
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        if nq == 0 or self.n == 0:
            return empty
        offs = np.arange(-int(ring), int(ring) + 1, dtype=np.int64)
        di = np.repeat(offs, offs.shape[0])
        dj = np.tile(offs, offs.shape[0])
        ai = np.clip(np.floor((qpos[:, 0] - self._ox) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)
        aj = np.clip(np.floor((qpos[:, 1] - self._oy) / self.cell),
                     -_CELL_CLIP, _CELL_CLIP).astype(np.int64)
        qk = ((ai[:, None] + di[None, :]) * _KEY_STRIDE
              + (aj[:, None] + dj[None, :])).ravel()
        qrow = np.repeat(np.arange(nq, dtype=np.int64), di.shape[0])
        idx = np.searchsorted(self._keys, qk)
        hit = idx < self._keys.shape[0]
        hit[hit] = self._keys[idx[hit]] == qk[hit]
        idx = idx[hit]
        qrow = qrow[hit]
        if idx.shape[0] == 0:
            return empty
        cnt = self._start[idx + 1] - self._start[idx]
        total = int(cnt.sum())
        if total == 0:
            return empty
        rep = np.repeat(np.arange(cnt.shape[0], dtype=np.int64), cnt)
        offs_in = (np.arange(total, dtype=np.int64)
                   - np.repeat(np.cumsum(cnt) - cnt, cnt))
        pt = self._order[self._start[idx][rep] + offs_in]
        return qrow[rep], pt


def all_point_pairs(n):
    """全ペア (i, j)(i≠j)の辞書順昇順 — 空間ハッシュとの同値検証・小規模の参照経路。"""
    ii = np.repeat(np.arange(n, dtype=np.int64), n)
    jj = np.tile(np.arange(n, dtype=np.int64), n)
    keep = ii != jj
    return ii[keep], jj[keep]


def neighbor_pairs(pos, reach, cell=None, hash_ok=True):
    """距離 reach 以内を必ず含む候補ペア (i, j) を **(i, j) 辞書順昇順**で返す。

    自己ペア (i, i) は含まない。重複も無い(各点はちょうど 1 セルに登録される)。
    戻り値の並びが辞書順であることが「合算順序 = 現行の行和と同一」の担保
    (本 module 冒頭 PointField の (c))。
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    n = pos.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    reach = float(max(reach, 0.0))
    if (not hash_ok) or n <= _ALL_PAIRS_MAX_N or reach <= 0.0:
        return all_point_pairs(n)
    c = float(cell) if cell else (reach / _POINT_CELL_DIV)
    if not (c > 0.0):
        return all_point_pairs(n)
    field = PointField(pos, c)
    ring = int(math.ceil(reach / field.cell))
    if (2 * ring + 1) ** 2 >= n:      # 安全弁: 探索セル数が点数を超えるなら全ペアが安い
        return all_point_pairs(n)
    qrow, pt = field.candidates(pos, ring)
    keep = qrow != pt
    qrow, pt = qrow[keep], pt[keep]
    flat = np.sort(qrow * n + pt)     # (i, j) 辞書順(重複なし = unique 不要)
    return flat // n, flat % n


def pair_blocks(ii, n, max_pairs=_PAIR_BLOCK):
    """候補ペア列を **行境界で** ブロックに切る [(row_lo, row_hi, p_lo, p_hi), …]。

    行(= 個体 i)を跨いで合算しないので、ブロック分割は加算順序を 1 ビットも変えない。
    """
    p = int(ii.shape[0])
    if p == 0:
        return [(0, n, 0, 0)]
    per_row = max(1.0, p / max(n, 1))
    rows = max(1, int(max_pairs / per_row))
    if rows >= n:
        return [(0, n, 0, p)]
    out = []
    for lo in range(0, n, rows):
        hi = min(lo + rows, n)
        p_lo = int(np.searchsorted(ii, lo, side="left"))
        p_hi = int(np.searchsorted(ii, hi, side="left"))
        out.append((lo, hi, p_lo, p_hi))
    return out


def cap_pairs(pos, ii, jj, n, cap):
    """候補ペアを「各行 (距離, index) 昇順の最近傍 cap 体」へ絞る(辞書順で返す)。

    現行 `argsort(d, kind="stable")` → `rank < cap` と **選抜集合もタイブレークも一致**する:
      - 行内の候補は既に j 昇順 → 距離の安定ソートは (距離, index) 昇順と同義。
      - カットオフ外の点が cap 枠を奪うことは無い(距離が遠いので必ず後順位)。
        逆に候補に混じる「reach 内だが cutoff 外」の点は valid=False で寄与 0 のまま。
    """
    if ii.shape[0] == 0:
        return ii, jj
    cnt = np.bincount(ii, minlength=n)
    kmax = int(cnt.max())
    if kmax <= cap:                    # 誰も cap に掛からない = 絞る必要が無い
        return ii, jj
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(cnt)[:-1]])
    rowpos = np.arange(ii.shape[0], dtype=np.int64) - starts[ii]
    jm = np.full((n, kmax), n, dtype=np.int64)          # 埋め草 = n(実在しない index)
    jm[ii, rowpos] = jj
    dm = np.full((n, kmax), np.inf)
    diff = pos[ii] - pos[jj]
    dm[ii, rowpos] = np.linalg.norm(diff, axis=1)
    order = np.argsort(dm, axis=1, kind="stable")[:, :cap]   # (距離, index) 昇順
    sel = np.take_along_axis(jm, order, axis=1)
    sel.sort(axis=1)                                   # 合算順序 = j 昇順へ戻す
    out_i = np.repeat(np.arange(n, dtype=np.int64), cap)
    out_j = sel.ravel()
    keep = out_j != n
    return out_i[keep], out_j[keep]


def knn_neighbors(qpos, points, k, max_dist, exclude_self=True):
    """各 query 行の「距離 max_dist 未満の近傍のうち (距離, index) 昇順の先頭 k 個」。

    戻り値 (order, valid): どちらも (nq, k)。valid[r] は先頭詰め(True が前)。
    現行 ORCA の `argsort(dist)[:, :k]` + `dist < neighbor_dist` と **完全一致**する
    (行の有効分は距離昇順の prefix なので、選抜集合も順序も同じ)。

    実装は**拡大リング探索**: 格子辺 c を密度から選び、ring=1,2,4,… と広げる。
    ring·c 以内に k 個見つかった行はそこで確定(それより近い点は全部列挙済み)。
    ring·c ≥ max_dist に達したら、max_dist 以内は完全に覆えているので確定。
    → ORCA の neighbor_dist=10 m を素直にセル化すると候補が πR²ρ で膨らむ問題を回避する
      (cap=12 = 認知的近傍が効いている以上、10 m 全部を列挙する必要が無い)。
    """
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    nq = qpos.shape[0]
    m = points.shape[0]
    k = int(k)
    order = np.zeros((nq, k), dtype=np.int64)
    valid = np.zeros((nq, k), dtype=bool)
    if nq == 0 or m == 0 or k <= 0:
        return order, valid
    max_dist = float(max(max_dist, 0.0))
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    area = max(float(hi[0] - lo[0]), 1e-9) * max(float(hi[1] - lo[1]), 1e-9)
    # 半径 c の円に期待 2k 体が入る格子辺(9 セルで ~5.7k 体 = ring 1 でほぼ確定する)
    cell = math.sqrt(max(2.0 * k * area / m, 1e-12) / math.pi)
    cell = min(max(cell, 1e-9), max_dist if max_dist > 0.0 else 1.0)
    field = PointField(points, cell)
    ring_max = max(1, int(math.ceil(max_dist / field.cell)))
    todo = np.arange(nq, dtype=np.int64)
    ring = 1
    while todo.size:
        r = min(ring, ring_max)
        qrow, pt = field.candidates(qpos[todo], r)
        src = todo[qrow]
        if exclude_self:
            keep = ~((src == pt) & (pt < nq))
            src, pt = src[keep], pt[keep]
        flat = np.sort(src * m + pt)
        ii, jj = flat // m, flat % m
        d = np.linalg.norm(qpos[ii] - points[jj], axis=1)
        cover = (r >= ring_max)
        if cover:
            done_mask = np.ones(todo.shape[0], dtype=bool)
        else:
            inside = np.bincount(ii[d <= r * field.cell], minlength=nq)
            done_mask = inside[todo] >= k
        done = todo[done_mask]
        if done.size:
            sel = np.zeros(nq, dtype=bool)
            sel[done] = True
            sub = sel[ii]
            _knn_fill(order, valid, ii[sub], jj[sub], d[sub], k, m, max_dist)
        todo = todo[~done_mask]
        if cover:
            break
        ring *= 2
    return order, valid


def _knn_fill(order, valid, ii, jj, d, k, m, max_dist):
    nq = order.shape[0]
    keep = d < max_dist
    ii, jj, d = ii[keep], jj[keep], d[keep]
    if ii.shape[0] == 0:
        return
    cnt = np.bincount(ii, minlength=nq)
    kmax = int(cnt.max())
    rows = np.nonzero(cnt)[0]
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(cnt)[:-1]])
    rowpos = np.arange(ii.shape[0], dtype=np.int64) - starts[ii]
    jm = np.full((nq, kmax), m, dtype=np.int64)
    dm = np.full((nq, kmax), np.inf)
    jm[ii, rowpos] = jj
    dm[ii, rowpos] = d
    kk = min(k, kmax)
    o = np.argsort(dm[rows], axis=1, kind="stable")[:, :kk]   # (距離, index) 昇順
    sel = np.take_along_axis(jm[rows], o, axis=1)
    ok = np.take_along_axis(dm[rows], o, axis=1) < max_dist
    order[rows, :kk] = np.where(ok, sel, 0)
    valid[rows, :kk] = ok


def wall_forces_from_pairs(pos, radius, field, ai, si,
                           wall_a=WALL_A_DEFAULT, wall_b=WALL_B_DEFAULT,
                           wall_range=WALL_RANGE_M):
    """候補ペア列から対壁斥力の合力 (N,2) を組む(Helbing2000 の f_iW)。

        f_iW = A_w · exp((r_i − d_iW)/B_w) · n_iW        (d_iW > r_i + wall_range で 0)

    d_iW = 個体中心から壁線分への最近点距離、n_iW = 壁 → 個体の単位法線。
    合算は ai 昇順・同一個体内は si 昇順の **逐次加算**(np.bincount)に固定するので、
    候補集合が「寄与 0 を除いて同じ」であれば全ペア版とビット単位で一致する。
    """
    n = pos.shape[0]
    f = np.zeros((n, 2), dtype=np.float64)
    if ai.shape[0] == 0:
        return f
    a = field.p1[si]                                  # (P,2) 線分始点
    eseg = field.p2[si] - a                           # (P,2) 線分ベクトル
    len2 = (eseg * eseg).sum(axis=1)
    len2_safe = np.where(len2 > 1e-12, len2, 1.0)
    p = pos[ai]
    t = np.clip(((p - a) * eseg).sum(axis=1) / len2_safe, 0.0, 1.0)
    closest = a + t[:, None] * eseg
    dvec = p - closest                                # 壁 → 個体
    dist = np.linalg.norm(dvec, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nvec = dvec / dist[:, None]
    nvec = np.nan_to_num(nvec)
    r_i = radius[ai]
    arg = np.clip((r_i - dist) / wall_b, a_min=None, a_max=_EXP_ARG_MAX)
    mag = np.where(dist <= r_i + wall_range, wall_a * np.exp(arg), 0.0)
    contrib = mag[:, None] * nvec
    f[:, 0] = np.bincount(ai, weights=contrib[:, 0], minlength=n)
    f[:, 1] = np.bincount(ai, weights=contrib[:, 1], minlength=n)
    return f


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
        walls:  壁線分 [((x1,y1),(x2,y2)), …] or None(既定)。None なら対壁斥力 f_iW は
                一切評価されない=従来の開放領域コアと完全同一(後方互換)。
        wall_a / wall_b / wall_range: f_iW の A_w [N] / B_w [m] / 作用距離 [m]。
        wall_hash: True(既定)= 空間ハッシュで最近点探索。False = 全ペア(検証用の参照経路。
                結果は空間ハッシュ版とビット一致する=テストで固定)。
        wall_cell: 空間ハッシュ格子の一辺 [m]。None(既定)= wall_range + 半径上限。
                値を変えても結果は不変(探索の粒度が変わるだけ)。
        neighbor_cap: 対人斥力の近傍上限(最近傍 cap 体のみ寄与・距離昇順+index 昇順の
                決定論選択)。None(既定)= 上限なし=従来どおり全近傍。
        pair_hash: True(既定)= 対人斥力の候補列挙を空間ハッシュ(PointField)で行う。
                False = 全ペア(検証用の参照経路)。**結果は両者でビット一致**
                (tests/test_physics_hash.py が固定)。
    """

    def __init__(self, pos, vel, goal, v0, radius=None, active=None,
                 mass=MASS_DEFAULT, tau=TAU_DEFAULT, a=A_DEFAULT, b=B_DEFAULT,
                 lambda_aniso=LAMBDA_DEFAULT, v_max_factor=V_MAX_FACTOR,
                 rng=None, noise=0.0, arrive_radius=0.5,
                 walls=None, wall_a=WALL_A_DEFAULT, wall_b=WALL_B_DEFAULT,
                 wall_range=WALL_RANGE_M, wall_hash=True, wall_cell=None,
                 neighbor_cap=None, pair_hash=True):
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
        # ── 対壁斥力 f_iW(walls=None なら field=None = 壁項を一度も評価しない) ──
        self.wall_a = float(wall_a)
        self.wall_b = float(wall_b)
        self.wall_range = float(wall_range)
        self.wall_hash = bool(wall_hash)
        # 空間ハッシュ格子 [m]。既定 = 作用距離 + 半径上限(= リング 1 周で reach を覆う)。
        self.wall_cell = float(wall_cell) if wall_cell else (self.wall_range + RADIUS_MAX)
        self.wall_field = (WallField(walls, cell=self.wall_cell)
                           if walls is not None and len(walls) else None)
        self.neighbor_cap = None if neighbor_cap is None else int(neighbor_cap)
        self.pair_hash = bool(pair_hash)

    # ── 希望方向 e_i(目標へ向かう単位ベクトル) ──
    def _desired_dir(self):
        d = self.goal - self.pos                    # (N,2)
        dist = np.linalg.norm(d, axis=1)            # (N,)
        e = np.zeros_like(d)
        nz = dist > 1e-9
        e[nz] = d[nz] / dist[nz, None]
        return e, dist

    # ── 対壁斥力 f_iW(walls を渡したときだけ評価される) ──
    def _wall_forces(self):
        """壁からの斥力の合力 (N,2)。最近点探索は空間ハッシュ(wall_hash=False で全ペア)。"""
        field = self.wall_field
        if field is None:
            return np.zeros_like(self.pos)
        if self.wall_hash:
            reach = float(self.radius.max()) + self.wall_range if self.radius.size else \
                self.wall_range
            ai, si = field.pairs(self.pos, reach)
        else:
            ai, si = field.all_pairs(self.pos.shape[0])
        return wall_forces_from_pairs(self.pos, self.radius, field, ai, si,
                                      wall_a=self.wall_a, wall_b=self.wall_b,
                                      wall_range=self.wall_range)

    # ── 全力の合成(駆動 + 対人斥力 + 対壁斥力[+揺らぎ]) ──
    def forces(self):
        """各個体に働く合力 (N,2) [N] を返す。非アクティブ個体は 0。

        f_i = f_drive + Σ_j f_ij + Σ_W f_iW + ξ
        ★ ξ はここが唯一の合成点(派生クラスは forces() を上書きしない)。壁ありでも
          noise>0 なら ξ が効く(竹-3: WallCrowd が ξ を落としていたバグの構造的解消)。
        """
        n = self.pos.shape[0]
        e, _ = self._desired_dir()

        # 駆動項: f_drive = m (v0 e − v) / τ   (Helbing&Molnár1995)
        f = self.mass * (self.v0[:, None] * e - self.vel) / self.tau

        if n > 1:
            f = f + (self._repulsion_hash(e) if self.pair_hash
                     else self._repulsion_dense(e))

        # 対壁斥力 f_iW = A_w·exp((r_i − d_iW)/B_w)·n_iW(walls 未指定なら評価しない)
        if self.wall_field is not None:
            f = f + self._wall_forces()

        # 揺らぎ ξ(既定 noise=0 で無効・決定論)
        if self.noise > 0.0 and self.rng is not None:
            f = f + self.mass * self.rng.normal(0.0, self.noise, size=(n, 2))

        f[~self.active] = 0.0
        return f

    # ── 対人斥力 Σ_j f_ij ────────────────────────────────────────────────── #
    def _repulsion_dense(self, e):
        """全ペア (N,N) の対人斥力(**参照実装**。空間ハッシュ版と結果はビット一致)。

        ★竹-3 まではこの本体が forces() に直書きされていた。式・マスク・合算順序は
          1 バイトも変えていない(pair_hash=False で通る経路 = 同値性テストの基準)。
        """
        n = self.pos.shape[0]
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
        # 近傍 cap(None=上限なし=従来経路): 各個体につき最近傍 cap 体のみ寄与。距離昇順の
        # 安定ソートで順位を作り(index 昇順でタイブレーク)、cap 位以内だけ残す=決定論選択。
        if self.neighbor_cap is not None and self.neighbor_cap < n - 1:
            order = np.argsort(d, axis=1, kind="stable")      # 距離昇順の列 index
            rank = np.argsort(order, axis=1, kind="stable")   # 各 j の順位
            valid = valid & (rank < self.neighbor_cap)
        contrib = (w * mag)[:, :, None] * nij                 # (N,N,2)
        contrib[~valid] = 0.0
        return contrib.sum(axis=1)                            # (N,2)

    def _repulsion_hash(self, e):
        """空間ハッシュで候補を絞った対人斥力(**_repulsion_dense とビット一致**)。

        候補 reach = 2·r_max + CUTOFF_M。実効カットオフ d ≤ r_i+r_j+CUTOFF_M は
        必ず reach 以内なので、候補は上位集合 = 落ちるのは寄与 0.0 の加算だけ。
        """
        n = self.pos.shape[0]
        reach = 2.0 * float(self.radius.max()) + CUTOFF_M
        ii, jj = neighbor_pairs(self.pos, reach)
        cap = self.neighbor_cap
        if cap is not None and cap < n - 1:
            ii, jj = cap_pairs(self.pos, ii, jj, n, cap)
        f = np.zeros((n, 2), dtype=np.float64)
        for _r0, _r1, p0, p1 in pair_blocks(ii, n):
            if p1 <= p0:
                continue
            si, sj = ii[p0:p1], jj[p0:p1]
            diff = self.pos[si] - self.pos[sj]                 # j → i
            d = np.linalg.norm(diff, axis=1)
            rr = self.radius[si] + self.radius[sj]
            arg = np.clip((rr - d) / self.b, a_min=None, a_max=_EXP_ARG_MAX)
            mag = self.a * np.exp(arg)
            with np.errstate(invalid="ignore", divide="ignore"):
                nij = diff / d[:, None]
            nij = np.nan_to_num(nij)
            cosphi = -(e[si, 0] * nij[:, 0] + e[si, 1] * nij[:, 1])
            w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
            valid = self.active[sj] & (d <= rr + CUTOFF_M)
            contrib = (w * mag)[:, None] * nij
            contrib[~valid] = 0.0
            # 合算は (i, j) 昇順の逐次加算(np.bincount)= 行和とビット一致
            f[_r0:_r1, 0] = np.bincount(si - _r0, weights=contrib[:, 0],
                                        minlength=_r1 - _r0)
            f[_r0:_r1, 1] = np.bincount(si - _r0, weights=contrib[:, 1],
                                        minlength=_r1 - _r0)
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


# ─────────────────────────────────────────────────────────────────────────────
# 歩行者信号ゲート(描画専用・決定論。docs/research/pedestrian-signals.md §3.2 案a)
# ─────────────────────────────────────────────────────────────────────────────
class SignalGate:
    """歩行者信号の決定論ゲート(赤=停止線で待機・青=一斉横断=全方向一斉横断の見せ場)。

    信号は周期関数で、位相は【絶対時刻(秒)】から決定論計算できる(乱数ゼロ)。
    サイクル C = 歩行者青 g + 青点滅 f + 赤 r。位相 φ(t) = (t + offset) mod C。
    横断可能(go)= φ < g + f(青+青点滅。青点滅は「渡り始めたら渡り切る」許容窓として
    横断可能に含める。docs/research/pedestrian-signals.md §3.1 の g=47s 相当)。

    SFM 連携(§3.2): 赤の間に停止線(=横断の手前)へ着いた個体を active=False で滞留させ、
    青で一斉に active=True へ解放する。Crowd.active(False の個体は力を及ぼさず・受けず・
    動かない)機構をそのまま使うため、curb での赤待ち→青での一斉横断が自然に出る。

    全方向一斉横断は「全歩行者信号が同相で青になる特殊ケース」= 同一 gate
    (同 offset)で領域内の全歩行者を一括ゲートすることで表現する。

    位相・秒数はすべて config or サイドカー由来の固定値。内部で乱数を一切引かない
    (同一入力 → 同一 bool =決定論)。

    ★装置としての同一性(actor model P2 / src/society/devices.py)
    ------------------------------------------------------------
    本クラスは「位相 = 絶対時刻の純関数」= **記憶を持たない固定スケジュール装置**
    (DEVS の δ_int だけを持ち、状態ベクトルが空)であり、装置層が言う device の
    条件を最初から満たしている。足りなかったのは **同一性**(どの交差点の信号なのかを
    外から指す名前)だけなので、`crossing_id`(交差点表 = world/zones.py の
    `signal.crossing_id`)から安定 id を導いて `device_id` に持たせる。
    ★**時刻計算(位相・青 / 青点滅 / 赤の窓)は 1 バイトも変えていない**
      (既存 tests/test_crossing_signals.py が無修正で緑であることがその証拠)。
      `crossing_id` 未指定(既存の全呼び出し)では `device_id` は None のままで、
      属性が 2 つ増える以外に一切の差分が無い。
    """

    def __init__(self, cycle_s, green_s, flash_s=0.0, offset_s=0.0,
                 crossing_id=None, device_id=None):
        self.cycle_s = float(cycle_s)
        self.green_s = float(green_s)
        self.flash_s = float(flash_s)
        self.offset_s = float(offset_s)
        # 装置としての同一性(既存呼び出しでは両方 None = 従来と同一の振る舞い)。
        # ★接頭辞は society/devices.py の SIGNAL_PREFIX と同じ文字列でなければならない
        #   (tests/test_devices.py が両者の一致を機械固定する)。ここで society 層を
        #   import しないのは、sfm_core が「純粋な物理コア」であるという既存の線引きを
        #   崩さないため(viz/sfm.py からの再export 経路も society に依存しない)。
        self.crossing_id = (None if crossing_id is None else int(crossing_id))
        self.device_id = (device_id if device_id is not None
                          else (None if self.crossing_id is None
                                else f"signal:{self.crossing_id}"))
        if self.cycle_s <= 0.0:
            raise ValueError("cycle_s must be > 0")
        if not (0.0 <= self.green_s <= self.cycle_s):
            raise ValueError("green_s must be within [0, cycle_s]")
        if self.green_s + self.flash_s > self.cycle_s:
            raise ValueError("green_s + flash_s must be <= cycle_s")

    def phase(self, sim_sec):
        """絶対時刻 sim_sec [s] における位相 [0, cycle_s)。"""
        return (float(sim_sec) + self.offset_s) % self.cycle_s

    def is_green(self, sim_sec):
        """歩行者(全方向)青(点灯)か = φ < green_s。"""
        return self.phase(sim_sec) < self.green_s

    def is_flashing(self, sim_sec):
        """青点滅(横断中は渡り切れるが新規開始は本来抑制)か = green_s ≤ φ < green_s+flash_s。"""
        p = self.phase(sim_sec)
        return self.green_s <= p < self.green_s + self.flash_s

    def can_cross(self, sim_sec):
        """横断してよい(青 or 青点滅)か = φ < green_s + flash_s(残りは赤=待機)。

        SFM ゲートの「解放」判定に使う(赤で curb 滞留 → 青で一斉横断)。"""
        return self.phase(sim_sec) < self.green_s + self.flash_s

    @classmethod
    def scramble(cls, cycle_s, green_s, flash_s=0.0, crossing_id=None):
        """全方向同相ゲート(offset 0 = 全方向同相で一斉に青)を作る。

        `crossing_id` は装置としての同一性のためだけの任意引数(既定 None = 従来と同一)。
        """
        return cls(cycle_s, green_s, flash_s, offset_s=0.0,
                   crossing_id=crossing_id)
