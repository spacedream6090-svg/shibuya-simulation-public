"""経路探索(モード対応: walk / bicycle / car)。A* + OD キャッシュ。

車は車道のみ、自転車は階段以外、徒歩は全部。経路が無ければ徒歩にフォールバック。
"""
from __future__ import annotations

import math

import networkx as nx

from .map import DRIVABLE, NO_BICYCLE, CityMap


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
                    def h(a: str, b: str) -> float:
                        ax, ay = self.city.node_xy(a)
                        bx, by = self.city.node_xy(b)
                        return math.hypot(ax - bx, ay - by)
                    try:
                        self._cache[key] = nx.astar_path(g, src, dst,
                                                         heuristic=h, weight="length")
                    except nx.NetworkXNoPath:
                        self._cache[key] = None
            if self._cache[key] is not None:
                return list(self._cache[key]), try_mode
        return [src], "walk"   # 完全に到達不能(孤立ノード)は動かない

    def route_length(self, path: list[str]) -> float:
        return sum(self.city.edge_length(u, v) for u, v in zip(path, path[1:]))
