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
"""
from __future__ import annotations

import math

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


class PerceptIndex:
    """step 単位の空間ハッシュ索引。cell = (context, floor(x/r), floor(y/r))。

    睡眠中・範囲外は聞き手になり得ないので索引から除外する(legacy 走査の
    `other.sleeping` / context 不一致スキップと同値)。位置が確定したフェーズで
    一度だけ構築し、その間だけ hearers_of に渡す。
    """

    __slots__ = ("radius", "_inv", "cells", "_occ")

    def __init__(self, radius: float, occluder=None):
        self.radius = float(radius)
        self._inv = (1.0 / self.radius) if self.radius > 0 else 0.0
        self.cells: dict[tuple, list] = {}
        self._occ = occluder

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

    def hearers(self, speaker) -> list:
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


def hearers_of(speaker, agents_or_index, radius_m: float, occluder=None) -> list:
    """speaker の発話が聞こえるエージェント(agent_id 昇順=決定論)。

    第2引数は agent の反復可能列(従来=全対全 live 走査)または PerceptIndex
    (空間索引=近傍9セルのみ走査)。どちらでも返り値は一致する。

    occluder(既定 None)= 視線遮蔽器。None のときは索引に据えた/install_occluder した
    既定へ後退し、遮蔽器が一切無ければ従来と完全にバイト一致(=遮蔽なし)。
    """
    if isinstance(agents_or_index, PerceptIndex):
        if occluder is None:
            return agents_or_index.hearers(speaker)
        # 明示の occluder が索引の据え付けより優先(呼び出し側の意図を尊重)。
        saved = agents_or_index._occ
        agents_or_index._occ = occluder
        try:
            return agents_or_index.hearers(speaker)
        finally:
            agents_or_index._occ = saved
    ctx = _context(speaker)
    occ = _resolve(occluder)                   # None=遮蔽なし=従来と完全同一
    result = []
    for other in agents_or_index:
        if other.id == speaker.id or other.sleeping or _context(other) != ctx:
            continue
        if math.hypot(other.x - speaker.x, other.y - speaker.y) <= radius_m:
            if occ is None or not occ.blocks(speaker, other):
                result.append(other)
    return sorted(result, key=lambda a: a.id)
