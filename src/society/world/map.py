"""道路ネットワーク+建物+POI+ゲートウェイ(地図 v3)。

v3: 全建物(用途 kind 付き)+実名 POI +垂直レイヤー(-1=地下街, 0=地上, 1=デッキ)
    + car_gateways(背景交通の発生点)。v2/v1 も読める(interface は seam として不変)。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx

DRIVABLE = {"primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service"}
NO_BICYCLE = {"steps"}

# 既定のフロアガイド(駅・主要商業施設の実フロア構成+接続)。存在すれば遅延ロード。
_DEFAULT_FLOORGUIDE = (Path(__file__).resolve().parents[3]
                       / "data" / "floorguide_shibuya.json")


class CityMap:
    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: dict = raw.get("meta", {})
        self.graph = nx.Graph()

        for node in raw["nodes"]:
            self.graph.add_node(node["id"], x=float(node["x"]), y=float(node["y"]),
                                poi=node.get("poi"), name=node.get("name"),
                                gateway=bool(node.get("gateway")),
                                layer=int(node.get("layer", 0)))

        for edge in raw["edges"]:
            u, v = edge["u"], edge["v"]
            geometry = edge.get("geometry")
            if not geometry:  # v1 互換: 直線
                geometry = [list(self.node_xy(u)), list(self.node_xy(v))]
            length = edge.get("length") or _poly_len(geometry)
            self.graph.add_edge(u, v, length=round(float(length), 1),
                                klass=edge.get("klass", "footway"),
                                layer=int(edge.get("layer", 0)),
                                geometry=[tuple(p) for p in geometry], u0=u)

        self.buildings: list[dict] = raw.get("buildings", [])
        self._bld_by_id = {b["id"]: b for b in self.buildings}
        self._bld_at_node: dict[str, list[dict]] = defaultdict(list)
        for b in self.buildings:
            cx = sum(p[0] for p in b["footprint"]) / len(b["footprint"])
            cy = sum(p[1] for p in b["footprint"]) / len(b["footprint"])
            b["centroid"] = (round(cx, 1), round(cy, 1))
            self._bld_at_node[b["entrance"]].append(b)

        self.gateways: list[str] = sorted(
            n for n, d in self.graph.nodes(data=True) if d.get("gateway"))
        self.car_gateways: list[str] = [
            n for n in raw.get("car_gateways", []) if n in self.graph]
        self.station_node: str | None = next(
            (n for n, d in self.graph.nodes(data=True) if d.get("poi") == "station"),
            None)

        # ---- POI(v3: 実名の店・会社・施設)----
        self.poi_list: list[dict] = [
            p for p in raw.get("pois", []) if p.get("node") in self.graph]
        self._poi_by_cat: dict[str, list[dict]] = defaultdict(list)
        self._poi_by_bld: dict[str, list[dict]] = defaultdict(list)
        self._poi_by_node: dict[str, list[dict]] = defaultdict(list)
        for p in self.poi_list:
            self._poi_by_cat[p["cat"]].append(p)
            self._poi_by_node[p["node"]].append(p)
            if p.get("building"):
                self._poi_by_bld[p["building"]].append(p)

        # ---- 住宅系建物(家の割当先。明示的な住宅を優先)----
        explicit = [b for b in self.buildings if b["kind"] == "residential"]
        maybe = [b for b in self.buildings if b["kind"] == "house?"]
        self.residential_buildings: list[dict] = \
            sorted(explicit, key=lambda b: b["id"]) + \
            sorted(maybe, key=lambda b: b["id"])

    # ---- ノード ----
    def node_xy(self, node: str) -> tuple[float, float]:
        d = self.graph.nodes[node]
        return d["x"], d["y"]

    def node_name(self, node: str) -> str:
        d = self.graph.nodes[node]
        return d.get("name") or "路上"

    def pois(self) -> list[str]:
        return sorted(n for n, d in self.graph.nodes(data=True) if d.get("poi"))

    def destinations(self) -> list[str]:
        """目的地候補 = 名前つき地点 + 建物入口 + 待ち合わせ名所(landmark POI)。

        landmark(ハチ公像等)を余暇・待ち合わせの行き先候補に含める(ユーザー要望
        2026-07-06)。旧地図は landmark cat が不在=候補は変わらない(バイト一致)。
        重複除去・ソート=決定論。"""
        dests = set(self.pois())
        dests.update(b["entrance"] for b in self.buildings)
        dests.update(p["node"] for p in self.poi_list
                     if p.get("cat") == "landmark")
        return sorted(dests)

    # ---- エッジ・幾何 ----
    def edge_length(self, u: str, v: str) -> float:
        return self.graph.edges[u, v]["length"]

    def xy_along(self, u: str, v: str, offset: float) -> tuple[float, float]:
        """u→v へ offset(m) 進んだ地点の座標(道の折れ線に沿う)。"""
        data = self.graph.edges[u, v]
        geometry = data["geometry"]
        if data["u0"] != u:
            geometry = list(reversed(geometry))
        remaining = max(0.0, min(offset, data["length"]))
        for a, b in zip(geometry, geometry[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg <= 0:
                continue
            if remaining <= seg:
                f = remaining / seg
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            remaining -= seg
        return geometry[-1]

    def node_layer(self, node: str) -> int:
        return self.graph.nodes[node].get("layer", 0)

    # ---- 建物 ----
    def buildings_at(self, node: str) -> list[dict]:
        return self._bld_at_node.get(node, [])

    def building(self, bld_id: str) -> dict:
        return self._bld_by_id[bld_id]

    def has_building(self, bld_id: str) -> bool:
        return bld_id in self._bld_by_id

    # ---- POI ----
    def pois_by_cat(self, cat: str) -> list[dict]:
        return self._poi_by_cat.get(cat, [])

    def pois_in_building(self, bld_id: str, floor: int | None = None) -> list[dict]:
        ps = self._poi_by_bld.get(bld_id, [])
        if floor is None:
            return ps
        return [p for p in ps if p.get("floor", 0) == floor]

    def pois_at_node(self, node: str) -> list[dict]:
        return self._poi_by_node.get(node, [])

    def place_label(self, node: str) -> str:
        """プロンプト用の場所名: ランドマーク名 > 最寄り POI > 路上(レイヤー付き)。"""
        d = self.graph.nodes[node]
        if d.get("name"):
            return d["name"]
        pois = self._poi_by_node.get(node)
        if pois:
            return f"{pois[0]['name']}の前"
        lyr = d.get("layer", 0)
        if lyr < 0:
            return "地下通路(渋谷ちかみち)"
        if lyr > 0:
            return "ペデストリアンデッキ"
        return "路上"

    # ---- 入口(実データ由来。v6 地図のみ。旧地図は空を返す=後方互換)----
    def building_entrances(self, bld_id: str) -> list[dict]:
        """建物の実入口 [{x, y, kind}]。entrance タグが無い建物・旧地図では []。"""
        b = self._bld_by_id.get(bld_id)
        return list(b.get("entrances", [])) if b else []

    def has_real_entrances(self, bld_id: str) -> bool:
        return bool(self.building_entrances(bld_id))

    # ---- 地下(垂直レイヤー<0 のノード/エッジ。ちかみち等)----
    def underground_nodes(self) -> list[str]:
        """地下(layer<0)ノードの id 一覧(ソート=決定論)。"""
        return sorted(n for n, d in self.graph.nodes(data=True)
                      if d.get("layer", 0) < 0)

    def underground_edges(self) -> list[tuple[str, str]]:
        """地下(layer<0)エッジの (u, v) 一覧。routing はこれらを通常どおり使える。"""
        return sorted((u, v) if u <= v else (v, u)
                      for u, v, d in self.graph.edges(data=True)
                      if d.get("layer", 0) < 0)

    # ---- フロアガイド(駅・主要商業施設の実フロア構成+接続。遅延ロード)----
    def _floorguide(self) -> dict:
        """floorguide_shibuya.json を初回アクセス時に読み、以後キャッシュ。無ければ空。"""
        cache = getattr(self, "_floorguide_cache", None)
        if cache is None:
            path = _DEFAULT_FLOORGUIDE
            if path.exists():
                cache = json.loads(path.read_text(encoding="utf-8"))
            else:
                cache = {"buildings": []}
            self._floorguide_cache = cache
        return cache

    @staticmethod
    def _fg_matches(rec: dict, name: str) -> bool:
        return any(m and (m in name or name in m) for m in rec.get("match", []))

    def floor_guide(self, name: str) -> dict | None:
        """名前でフロアガイドを引く(部分一致)。フロア構成 dict を返す。無ければ None。"""
        if not name:
            return None
        for rec in self._floorguide().get("buildings", []):
            if self._fg_matches(rec, name):
                return rec
        return None

    def building_connections(self, name: str) -> list[dict]:
        """駅⇄ビル/ビル⇄ビルの実接続 [{to_building, via, level}]。無ければ []。"""
        rec = self.floor_guide(name)
        return list(rec.get("connections", [])) if rec else []


def _poly_len(points) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))
