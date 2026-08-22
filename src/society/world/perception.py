"""知覚(誰の声が聞こえるか)。同じ文脈(路上どうし/同じ建物の同じ階)のみ。

非ブロードキャスト原則: 範囲外(outside)のエージェントには何も聞こえない。

スケール: 全対全走査 O(n²) を避けるため、step ごとに1回だけ空間グリッド索引
(セル = perception_radius)を build_index() で構築し、hearers_of は近傍9セルだけを
走査する。**返り値の内容・順序は全対全走査(下の legacy 経路)とバイト一致**する
(cell = radius なので半径内の点は必ず ±1 セルに入り、距離判定は従来どおり残す)。
索引を使うのは位置が安定なフェーズ(_phase_drive/_decide)だけ。位置が動く _apply では
従来どおり agents リストを渡す(索引が古くなるため=live 走査で完全一致)。

擬似視覚(壁による遮蔽): occluder を渡すか install_occluder() で本層に据えると、距離・
同一階フィルタを通ったペアに視線(LOS)判定を追加する(vision.py の VisionOccluder)。
occluder が無いとき(既定)は挙動が従来と完全にバイト一致=遮蔽なし。occluder は幾何しか
見ないので no-fingerprint 契約を破らない。

── 有界化(声の段階 + 最寄り K)。正典 docs/plans/hearer-cap-plan.md §2 ────────────
`hearers_of` / `PerceptIndex.hearers` は 2 つの**任意**引数を受け取る:

  radius_eff … 「この発話が実際に届く距離」[m]。声の段階(作用点0)の実体で、
               **索引のセル半径(= world.perception_radius_m)以下へクランプ**する
               (r ≤ セル幅なら半径内の点は必ず ±1 セルに入る = 走査セルは 9 のまま)。
  cap        … 聞き手の最大人数(位相的近傍)。**距離二乗 昇順 → id 昇順**で選抜し、
               返り値は従来どおり **id 昇順**(下流契約不変)。規則は第141 S15
               (scheduler._attention_limited)と完全に同一で、乱数を 1 粒も引かない。

**どちらも既定(cap=0 / radius_eff=None)なら下の分岐へ 1 度も入らない**: 既存の列挙
コードが 1 行も変わらずそのまま走る = golden L1 バイト一致(R1 をコード構造で保証)。

有界経路に入ったときだけ 9 セルの座標を numpy 化(**セル単位で遅延構築・step 内で再利用**)
して距離判定と選抜をベクトル化する。半径判定は `np.hypot` = `math.hypot` と同一の丸め、
順位は `dx*dx+dy*dy` = S15 と同一の鍵なので、ループ実装と**同じ集合・同じ順序**になる
(tests が機械照合する)。遮蔽器が実際に据わっているときはベクトル化を捨ててループへ
後退する(正しさ優先。本選は遮蔽未配線 = occluder None)。
"""
from __future__ import annotations

import math

import numpy as np

# 本層に据えた既定の遮蔽器(None=遮蔽なし=従来と完全同一)。引数 occluder が優先。
# 全 hearers_of 呼び出し(索引/legacy 双方)に一括で効かせる本番トグルの据え付け先。
_OCCLUDER = None


def install_occluder(occluder) -> None:
    """モジュール全体の既定遮蔽器を据える(シミュ初期化で1回)。None で解除。"""
    global _OCCLUDER
    _OCCLUDER = occluder


def clear_occluder() -> None:
    global _OCCLUDER
    _OCCLUDER = None


def active_occluder():
    return _OCCLUDER


def _resolve(occluder):
    """引数優先・無ければ据え付けの既定遮蔽器。両方 None なら遮蔽なし。"""
    return occluder if occluder is not None else _OCCLUDER


def _context(agent) -> tuple:
    if agent.loc == "outside":
        return ("outside", agent.id)   # 誰とも重ならない
    if agent.building:
        return ("bld", agent.building, agent.floor)
    return ("street",)


# --------------------------------------------------------------------------- #
# 声の段階(作用点0)= 発話の種類 → 声の大きさ → 実効半径 の決定論写像
#   正典: docs/plans/hearer-cap-plan.md §2「作用点0」。
#   音響: 会話音声 ~60dB@1m・距離が倍で −6dB・街頭騒音 70-95dBA → 通常声の了解圏は数 m。
#   屋内(騒音 ~50dB)は同じ声が 2-3 倍届く。段階の**選び方**は engine 側(発話文脈からの
#   決定論写像)で、本層は「段階 → 距離」の物理表しか持たない(no-fingerprint)。
# --------------------------------------------------------------------------- #
SPEECH_LEVELS: tuple[str, ...] = ("normal", "raised", "shout")

SPEECH_DEFAULTS: dict = {
    "enabled": False,                                   # 既定 OFF = 半径一律(現行と完全同一)
    "normal": {"outdoor_m": 5.0, "indoor_m": 10.0},     # 通常の会話声
    "raised": {"outdoor_m": 12.0, "indoor_m": 20.0},    # 張り上げ(+9dB ≒ 2-3 倍)
    "shout": {"outdoor_m": 30.0, "indoor_m": 40.0},     # 叫び(+18-24dB ≒ 8-16 倍)
}


def build_speech_cfg(raw, max_radius_m: float) -> dict:
    """conf の `world.speech_levels` を正準化する(純関数・既定 OFF=現行と完全同一)。

    すべての距離は **[0, max_radius_m] へクランプ**する(= 索引のセル幅を超えない)。
    上限を超える値を書いても静かに落ちるのではなく、`world.perception_radius_m` まで
    切り下げられる(走査セル範囲を広げる実装を持たない = 9 セルの不変条件を守るため)。
    """
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                                   # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    raw = dict(raw or {})
    top = float(max_radius_m)
    cfg: dict = {"enabled": bool(raw.get("enabled", SPEECH_DEFAULTS["enabled"]))}
    for level in SPEECH_LEVELS:
        base = dict(SPEECH_DEFAULTS[level])
        got = dict(raw.get(level, {}) or {})
        row = {}
        for side in ("outdoor_m", "indoor_m"):
            try:
                v = float(got.get(side, base[side]))
            except (TypeError, ValueError):
                v = float(base[side])
            row[side] = min(max(0.0, v), top)            # ★ ≤ perception_radius_m を強制
        cfg[level] = row
    return cfg


def speech_radius(speech_cfg: dict, level: str, speaker) -> float:
    """段階 × 屋内外 → 実効半径 [m](純関数)。未知の段階は保守側 normal へ倒す。"""
    row = speech_cfg.get(level) or speech_cfg["normal"]
    indoor = _context(speaker)[0] == "bld"
    return float(row["indoor_m"] if indoor else row["outdoor_m"])


def speech_cfg_of(sim) -> dict:
    """`world.speech_levels` を遅延構築して sim にキャッシュする(呼び手 2 つの共有口)。

    キャッシュ属性 `_speech_levels_cfg` は L1/L2/乱数のどこにも現れない(conversation._cfg と
    同型)。既定 OFF のとき返る dict の `enabled` は False = 呼び手は現行経路へ落ちる。"""
    cfg = getattr(sim, "_speech_levels_cfg", None)
    if cfg is None:
        try:
            world = sim.cfg.get("world", {}) or {}
            raw = world.get("speech_levels", None)
            top = float(world.get("perception_radius_m", 0.0) or 0.0)
        except Exception:                               # noqa: BLE001 (旧 config 互換)
            raw, top = None, 0.0
        cfg = build_speech_cfg(raw, top)
        sim._speech_levels_cfg = cfg
    return cfg


def nearest_k(speaker, candidates: list, cap: int) -> list:
    """候補から最寄り K 人を選び **id 昇順**で返す(第141 S15 `_attention_limited` と同一規則)。

    順序規則: **距離二乗 昇順 → id 昇順**(平方根も乱数も使わない完全な決定論)。
    cap<=0 または候補数<=cap のときは「id 昇順に並べ替えるだけ」= 絞らないときと同一集合・
    同一順序(呼び手が既に id 昇順を期待しているため並べ替えは常に行う)。
    """
    if cap <= 0 or len(candidates) <= cap:
        return sorted(candidates, key=lambda a: a.id)
    sx, sy = float(speaker.x), float(speaker.y)

    def _key(h):
        dx, dy = float(h.x) - sx, float(h.y) - sy
        return (dx * dx + dy * dy, int(h.id))
    return sorted(sorted(candidates, key=_key)[:cap], key=lambda a: int(a.id))


class PerceptIndex:
    """step 単位の空間ハッシュ索引。cell = (context, floor(x/r), floor(y/r))。

    睡眠中・範囲外は聞き手になり得ないので索引から除外する(legacy 走査の
    `other.sleeping` / context 不一致スキップと同値)。位置が確定したフェーズで
    一度だけ構築し、その間だけ hearers_of に渡す。
    """

    __slots__ = ("radius", "_inv", "cells", "_occ", "_np", "_nb")

    def __init__(self, radius: float, occluder=None):
        self.radius = float(radius)
        self._inv = (1.0 / self.radius) if self.radius > 0 else 0.0
        self.cells: dict[tuple, list] = {}
        self._occ = occluder
        # 有界経路(cap / radius_eff)のときだけ育つ numpy キャッシュ。
        #   _np … セル 1 個の (x, y, id) 配列(近傍 9 個の中心セルから共有される)
        #   _nb … 中心セルごとの 9 セル連結(**同じセルに居る話者どうしが共有する**)。
        #         これが無いと連結と平坦化が話者ごとに O(近傍総数) 掛かり、
        #         ベクトル化の利得を打ち消す(密なセルでは 3 桁の差になる)。
        # 既定経路では **どちらも 1 度も触らない**(空 dict のまま)= 追加コストも状態もゼロ。
        # 索引は step ごとに作り直されるのでキャッシュの寿命も 1 step(陳腐化しない)。
        self._np: dict[tuple, tuple] = {}
        self._nb: dict[tuple, tuple] = {}

    def _cell_xy(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x * self._inv), math.floor(y * self._inv))

    def add(self, agent) -> None:
        if agent.sleeping:
            return
        ctx = _context(agent)
        if ctx[0] == "outside":
            return
        cx, cy = self._cell_xy(agent.x, agent.y)
        self.cells.setdefault((ctx, cx, cy), []).append(agent)

    # ---- 有界経路(cap / radius_eff)。既定では 1 度も呼ばれない ------------------
    def _cell_arrays(self, key: tuple, bucket: list) -> tuple:
        """セル 1 個の (x, y, id) を numpy 化して step 内でキャッシュする(遅延構築)。

        索引は step ごとに作り直される(build_index)ので、キャッシュの寿命も 1 step。
        位置が動くフェーズでは索引そのものを使わない規約なので、値が古くなることはない。
        """
        arr = self._np.get(key)
        if arr is None:
            n = len(bucket)
            arr = (np.fromiter((float(a.x) for a in bucket), np.float64, n),
                   np.fromiter((float(a.y) for a in bucket), np.float64, n),
                   np.fromiter((int(a.id) for a in bucket), np.int64, n))
            self._np[key] = arr
        return arr

    def _neighborhood(self, ctx: tuple, cx: int, cy: int):
        """中心セルの近傍 9 セルを連結した (xs, ys, ids, flat) を返す(遅延構築・共有)。

        **同じセルに居る話者は同じ近傍を見る**ので、連結と平坦化はセルにつき 1 回で済む。
        近傍に誰も居なければ None。"""
        key = (ctx, cx, cy)
        if key in self._nb:
            return self._nb[key]
        cells = self.cells
        xs_l: list = []
        ys_l: list = []
        ids_l: list = []
        objs: list = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ck = (ctx, cx + dx, cy + dy)
                bucket = cells.get(ck)
                if not bucket:
                    continue
                xa, ya, ia = self._cell_arrays(ck, bucket)
                xs_l.append(xa)
                ys_l.append(ya)
                ids_l.append(ia)
                objs.append(bucket)
        if not objs:
            self._nb[key] = None
            return None
        nb = (xs_l[0] if len(xs_l) == 1 else np.concatenate(xs_l),
              ys_l[0] if len(ys_l) == 1 else np.concatenate(ys_l),
              ids_l[0] if len(ids_l) == 1 else np.concatenate(ids_l),
              objs[0] if len(objs) == 1 else [a for b in objs for a in b])
        self._nb[key] = nb
        return nb

    def _hearers_bounded(self, speaker, cap: int, radius_eff) -> list:
        """半径 radius_eff(≤ セル幅)+ 最寄り cap 人で列挙する(ベクトル化・決定論)。"""
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return []
        radius = self.radius if radius_eff is None else min(float(radius_eff),
                                                            self.radius)
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        sx, sy, sid = float(speaker.x), float(speaker.y), int(speaker.id)
        cells = self.cells
        occ = _resolve(self._occ)
        if occ is not None:
            # 遮蔽器が実際に据わっている: ベクトル化を捨ててループで正しさを取る。
            cands = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = cells.get((ctx, cx + dx, cy + dy))
                    if not bucket:
                        continue
                    for other in bucket:
                        if other.id == sid:
                            continue
                        if math.hypot(other.x - sx, other.y - sy) <= radius:
                            if not occ.blocks(speaker, other):
                                cands.append(other)
            return nearest_k(speaker, cands, cap)
        nb = self._neighborhood(ctx, cx, cy)
        if nb is None:
            return []
        xs, ys, ids, flat = nb
        dx_a = xs - sx
        dy_a = ys - sy
        # ① 外接正方形で粗く落とす(hypot(dx,dy) >= max(|dx|,|dy|) なので**厳密に等価**な
        #    前フィルタ)。声の段階(5m)のように半径がセル幅(40m)より小さいとき、
        #    高価な hypot に渡す要素数が桁で減る。
        box = np.flatnonzero((np.abs(dx_a) <= radius) & (np.abs(dy_a) <= radius)
                             & (ids != sid))
        if box.size == 0:
            return []
        # ② 半径判定は np.hypot(= math.hypot と同一の丸め)= ループ実装と同じ集合。
        keep = box[np.hypot(dx_a[box], dy_a[box]) <= radius]
        n = int(keep.size)
        if n == 0:
            return []
        if cap <= 0 or n <= cap:
            sel = keep[np.argsort(ids[keep], kind="stable")]     # id 昇順(従来の並び)
            return [flat[int(i)] for i in sel]
        # 順位鍵は距離二乗(S15 と同一)。argpartition で cap 件の窓を切り、境界の同値だけ
        # 厳密に (距離二乗, id) で並べ直す = 全体ソートを避けた O(n) 選抜(決定論)。
        d2 = dx_a[keep] * dx_a[keep] + dy_a[keep] * dy_a[keep]
        part = np.argpartition(d2, cap - 1)[:cap]
        thresh = float(d2[part].max())
        cand = np.flatnonzero(d2 <= thresh)
        order = np.lexsort((ids[keep][cand], d2[cand]))
        chosen = keep[cand[order[:cap]]]
        chosen = chosen[np.argsort(ids[chosen], kind="stable")]  # 返りは id 昇順
        return [flat[int(i)] for i in chosen]

    def hearers(self, speaker, cap: int = 0, radius_eff=None) -> list:
        # ★既定(cap=0 かつ radius_eff=None)ではこの分岐に入らず、下の従来コードが
        #   1 行も変わらずそのまま走る(golden L1 バイト一致の構造的保証)。
        if cap > 0 or radius_eff is not None:
            return self._hearers_bounded(speaker, int(cap), radius_eff)
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return []
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        radius = self.radius
        sx, sy, sid = speaker.x, speaker.y, speaker.id
        cells = self.cells
        occ = _resolve(self._occ)              # None=遮蔽なし=従来と完全同一
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = cells.get((ctx, cx + dx, cy + dy))
                if not bucket:
                    continue
                for other in bucket:
                    if other.id == sid:
                        continue
                    if math.hypot(other.x - sx, other.y - sy) <= radius:
                        if occ is None or not occ.blocks(speaker, other):
                            result.append(other)
        return sorted(result, key=lambda a: a.id)


def salience_gate(items: list, scores: list, k: int) -> list:
    """同時知覚 item を上位 K 件に絞る注意の容量制約ゲート(Cowan 2001: 焦点は約4チャンク)。

    items と scores は同順の並び。scores は上流(factors 層)が算出した**不透明な float**で、
    本層はその意味を知らずソートするだけ(価値名・valence 語をここに晒さない=no-fingerprint)。

    - k<=0(=∞/無効)または item 数<=K のとき: 入力を**順序そのまま**返す(既定=従来と同一集合・同一順序)。
    - k>0 のとき: score 上位 K を選び、**元の並び順を保って**返す(決定論。同点は先着=index 昇順)。
    """
    n = len(items)
    if k <= 0 or n <= k:
        return list(items)
    top = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    keep = set(top)
    return [it for i, it in enumerate(items) if i in keep]


def build_index(agents, radius_m: float, occluder=None) -> PerceptIndex:
    """位置が確定した時点の全 agent から空間索引を1回だけ構築する。

    occluder(既定 None)を渡すと索引経由の hearers に視線遮蔽を効かせる。None のときは
    据え付けの既定遮蔽器(install_occluder)へ後退し、それも無ければ遮蔽なし=従来同一。
    """
    idx = PerceptIndex(radius_m, occluder=occluder)
    for a in agents:
        idx.add(a)
    return idx


def hearers_of(speaker, agents_or_index, radius_m: float, occluder=None,
               *, cap: int = 0, radius_eff=None) -> list:
    """speaker の発話が聞こえるエージェント(agent_id 昇順=決定論)。

    第2引数は agent の反復可能列(従来=全対全 live 走査)または PerceptIndex
    (空間索引=近傍9セルのみ走査)。どちらでも返り値は一致する。

    occluder(既定 None)= 視線遮蔽器。None のときは索引に据えた/install_occluder した
    既定へ後退し、遮蔽器が一切無ければ従来と完全にバイト一致(=遮蔽なし)。

    cap(既定 0=無制限)/ radius_eff(既定 None=従来半径)= 有界化の 2 引数
    (docs/plans/hearer-cap-plan.md §2)。**両方とも既定なら下の従来コードがそのまま走る**。
    radius_eff は索引経路では索引のセル幅へクランプされる(9 セル走査の不変条件)。
    """
    bounded = cap > 0 or radius_eff is not None
    if isinstance(agents_or_index, PerceptIndex):
        if occluder is None:
            return (agents_or_index.hearers(speaker, cap, radius_eff) if bounded
                    else agents_or_index.hearers(speaker))
        # 明示の occluder が索引の据え付けより優先(呼び出し側の意図を尊重)。
        saved = agents_or_index._occ
        agents_or_index._occ = occluder
        try:
            return (agents_or_index.hearers(speaker, cap, radius_eff) if bounded
                    else agents_or_index.hearers(speaker))
        finally:
            agents_or_index._occ = saved
    ctx = _context(speaker)
    occ = _resolve(occluder)                   # None=遮蔽なし=従来と完全同一
    radius = radius_m if radius_eff is None else float(radius_eff)
    result = []
    for other in agents_or_index:
        if other.id == speaker.id or other.sleeping or _context(other) != ctx:
            continue
        if math.hypot(other.x - speaker.x, other.y - speaker.y) <= radius:
            if occ is None or not occ.blocks(speaker, other):
                result.append(other)
    if bounded:                                # 最寄り K 選抜(S15 と同一規則)
        return nearest_k(speaker, result, cap)
    return sorted(result, key=lambda a: a.id)
