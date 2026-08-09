"""経路探索(モード対応: walk / bicycle / car)。A* + OD キャッシュ。

車は車道のみ、自転車は階段以外、徒歩は全部。経路が無ければ徒歩にフォールバック。
"""
from __future__ import annotations

import math

import networkx as nx

from .map import DRIVABLE, NO_BICYCLE, CityMap

# networkx 3.x は astar_path を **2段** で包む(backends.py の `argmap(_do_nothing)(self)`):
#   nx.astar_path = argmap トランポリン → `_dispatchable` オブジェクト → 素の実装
# `__wrapped__` を1段だけ辿ると **dispatcher 本体** に着地するため、呼ぶたびに
# `_call_if_no_backends_installed`(backend 引数の検査 → orig_func への委譲)を通ってしまう。
# 最後まで辿れば素の実装に直接届く = 同一アルゴリズム・同一 tie-break のまま検査分だけ消える
# (本リポジトリは backend= を一切渡さないので分岐の結果は常に「素の実装を呼ぶ」で同値)。
# 剥がせない版では元の callable をそのまま返す(=現行と同じ呼び先)。
def _undispatched(func):
    """networkx の dispatch/argmap ラッパを剥がして素の実装関数を返す。"""
    seen: set[int] = set()
    f = func
    while True:
        nxt = getattr(f, "orig_func", None) or getattr(f, "__wrapped__", None)
        if nxt is None or id(nxt) in seen:
            return f
        seen.add(id(nxt))
        f = nxt


_ASTAR = _undispatched(nx.astar_path)


def _allowed(mode: str, klass: str) -> bool:
    if mode == "car":
        return klass in DRIVABLE
    if mode == "bicycle":
        return klass not in NO_BICYCLE
    return True


class Router:
    def __init__(self, city: CityMap):
        self.city = city
        self._cache: dict[tuple[str, str, str], list[str] | None] = {}
        self._sub: dict[str, nx.Graph] = {}

    def _any_closed(self) -> bool:
        """封鎖中(edge["closed"])のエッジが1つでもあるか。摂動シナリオが無い間は False。"""
        return any(d.get("closed") for _u, _v, d in self.city.graph.edges(data=True))

    def invalidate(self) -> None:
        """OD キャッシュとモード別部分グラフを捨てる(封鎖の発動・解除で呼ばれる)。

        封鎖が無い平時には呼ばれない=既定挙動には一切触れない。次回 route() 時に
        現在の closed フラグを反映した部分グラフが再構築される。"""
        self._cache.clear()
        self._sub.clear()

    def _node_attrs(self):
        """`city.graph` のノード属性 dict を直に引く索引(`graph.nodes[n]` と同一オブジェクト)。

        A* のヒューリスティックは 1 経路で数十万回呼ばれる。`CityMap.node_xy` 経由だと
        1 回ごとに Python 関数呼び出し + `NodeView.__getitem__` が挟まる一方、返る x/y は
        この dict から取り出した **同じ float オブジェクト**。よって直に引いても値は完全同一で、
        A* の展開順・tie-break・経路は 1 ノードも変わらない(ノード座標は CityMap 構築時に
        1 回だけ書かれ、走行中に書き換わる箇所は無い)。`_node` を持たない実装では
        公開 NodeView に後退する(その場合は現行と同じ経路)。
        """
        graph = self.city.graph
        nodes = getattr(graph, "_node", None)
        return nodes if nodes is not None else graph.nodes

    def _graph(self, mode: str) -> nx.Graph:
        if mode not in self._sub:
            if mode == "walk" and not self._any_closed():
                self._sub[mode] = self.city.graph       # 平時: 現行と同一(コピーしない)
            else:
                g = nx.Graph()
                for u, v, d in self.city.graph.edges(data=True):
                    if d.get("closed"):
                        continue                        # 封鎖エッジは経路網から除外
                    if _allowed(mode, d["klass"]):
                        g.add_edge(u, v, **d)
                self._sub[mode] = g
        return self._sub[mode]

    def route(self, src: str, dst: str, mode: str = "walk") -> tuple[list[str], str]:
        """(ノード列, 実際に使うモード)。モードで到達不能なら徒歩に落とす。"""
        if src == dst:
            return [src], "walk"
        for try_mode in ([mode, "walk"] if mode != "walk" else ["walk"]):
            key = (try_mode, src, dst)
            if key not in self._cache:
                g = self._graph(try_mode)
                if src not in g or dst not in g:
                    self._cache[key] = None
                else:
                    # ヒューリスティックの座標は **常に city.graph** から引く(モード別
                    # 部分グラフ g は add_edge で作られ x/y を持たない)= 従来と同じ出典。
                    node_attrs = self._node_attrs()
                    hypot = math.hypot

                    def h(a: str, b: str) -> float:
                        da = node_attrs[a]
                        db = node_attrs[b]
                        return hypot(da["x"] - db["x"], da["y"] - db["y"])
                    try:
                        self._cache[key] = _ASTAR(g, src, dst,
                                                  heuristic=h, weight="length")
                    except nx.NetworkXNoPath:
                        self._cache[key] = None
            if self._cache[key] is not None:
                return list(self._cache[key]), try_mode
        return [src], "walk"   # 完全に到達不能(孤立ノード)は動かない

    def route_length(self, path: list[str]) -> float:
        return sum(self.city.edge_length(u, v) for u, v in zip(path, path[1:]))
