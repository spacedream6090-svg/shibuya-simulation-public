"""実バスダイヤの静的表による乗車結合 v-Ride-2(本体側・純ロジック)。

設計正典: docs/research/sumo-live-transit.md §3(バス pt・静的表推奨=SUMO を回さない軽量版)。
役割: build_bus_table.py が GTFS から吐いた **停留所・系統・発時刻・区間所要のコンパクト静的表**
  (data/odpt/bus_table_shibuya.json)を読み、出発/目的が停留所 access 圏内のとき「次便の待ち時間 +
  区間所要」を実ダイヤ近似で返す。既存 BusNetwork.find_ride の「停留所<100m なら乗れる」近似の格上げ。

この読み手を src/society 直下に置く理由: 表(=data)に実バス停名が載る場合でも、engine/cognition/world
  の基盤コードに地名リテラルを入れない(D1-W2 地名禁止ガード)。表はデータ、読み手は素の JSON 読取のみ。

決定論(絶対条件): 乱数・wall-clock・k/traits/belief を一切参照しない純幾何+純ダイヤ計算。
  find_ride は座標(city.node_xy)と静的表の発時刻だけを見る。呼び出し側 routine._ride_extra の乗車判断も
  非LLM=呼数を1本も足さない。既定 OFF(transit_ride.bus_table.enabled=false)は BusTable を生成しない。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

STEP_SECONDS = 600.0        # 正準 Δt での 1 step = 600s。**本 module では未使用**
                            # (ダイヤは実時刻[分]で扱うので Δt に依存しない)
_LON_M = 111320.0           # sumo_pipeline.project / build_map と同一の局所接平面定数
_LAT_M = 110540.0


def _project(lat: float, lon: float, origin) -> tuple[float, float]:
    """経緯度 → 原点基準ローカル平面 m(build_map._make_project / sumo_pipeline.project と同一式)。"""
    o_lat, o_lon = origin
    cos_lat = math.cos(math.radians(o_lat))
    return (lon - o_lon) * _LON_M * cos_lat, (lat - o_lat) * _LAT_M


class BusTable:
    """静的表(実ダイヤ近似)。serves は常に True 扱い(便の有無は次便探索で判定=時刻表が担う)。

    find_ride(from_node, to_node, city, call_min) → {line, from, to, wait_s, ride_s} or None。
    出発/目的が同一系統の停留所 access 圏(<radius)にあり順方向(from の停留所 index < to の停留所 index)の
    とき、呼時刻 call_min(シミュ内分)から次便の発時刻を引いて待ち時間、区間 cum_sec 差から乗車時間を返す。
    """

    def __init__(self, data: dict, city, *, origin=None, access_radius_m: float = 150.0):
        self.radius = float(access_radius_m)
        self.meta = dict(data.get("_meta", {}) or {})
        # 表の原点(なければ引数 origin=地図原点)。x,y が表にあればそれを優先(build 時に地図原点で投影済)。
        tbl_origin = self.meta.get("origin_latlon")
        self.origin = tuple(tbl_origin) if tbl_origin else (tuple(origin) if origin else None)
        self._stop_xy: dict[str, tuple[float, float]] = {}
        self._stop_name: dict[str, str] = {}
        for s in data.get("stops", []):
            sid = str(s.get("id"))
            self._stop_name[sid] = str(s.get("name", sid))
            if s.get("x") is not None and s.get("y") is not None:
                self._stop_xy[sid] = (float(s["x"]), float(s["y"]))
            elif self.origin is not None and s.get("lat") is not None:
                self._stop_xy[sid] = _project(float(s["lat"]), float(s["lon"]), self.origin)
        # 系統: stops(停留所 id 列)・cum_sec(先頭からの累積所要秒)・departures(先頭停の平日発時刻・分)。
        self.routes: list[dict] = []
        for r in data.get("routes", []):
            stops = [str(sid) for sid in r.get("stops", [])]
            cum = [float(c) for c in r.get("cum_sec", [])]
            deps = sorted(int(d) for d in r.get("departures", []))
            if len(stops) >= 2 and len(cum) == len(stops) and deps:
                self.routes.append({
                    "name": str(r.get("name", "系統")),
                    "direction": str(r.get("direction", "")),
                    "stops": stops, "cum_sec": cum, "departures": deps})

    # ---- 停留所解決 ----
    def _nearest_idx(self, x: float, y: float, route: dict):
        """route の停留所列の中で (x,y) に最も近い停留所 index を radius 内で返す(なければ None)。決定論。"""
        best_i, best_d = None, self.radius
        for i, sid in enumerate(route["stops"]):
            xy = self._stop_xy.get(sid)
            if xy is None:
                continue
            d = math.hypot(xy[0] - x, xy[1] - y)
            if d <= best_d:
                best_i, best_d = i, d
        return best_i

    def _schedule(self, route: dict, i: int, j: int, call_min: int) -> tuple[float, float]:
        """停留所 index i→j の (wait_s, ride_s)。次便の先頭発時刻から i 停到着時刻を作り、呼時刻からの
        待ちを、cum_sec 差から乗車時間を得る。当日の次便が無ければ翌日始発へ回す(+1440 分)。"""
        cum = route["cum_sec"]
        ride_s = max(0.0, cum[j] - cum[i])
        call_sec = float(call_min) * 60.0
        offset_i = cum[i]
        arrivals_i = [float(d) * 60.0 + offset_i for d in route["departures"]]
        future = [a for a in arrivals_i if a >= call_sec]
        board = min(future) if future else (min(arrivals_i) + 1440.0 * 60.0)
        return board - call_sec, ride_s

    def find_ride(self, from_node: str, to_node: str, city, call_min: int):
        """出発/目的が同一系統の停留所 access 圏内かつ順方向なら {line, from, to, wait_s, ride_s}。
        複数系統が乗れるときは (待ち時間, 系統名) 昇順で決定論選択。乗れなければ None(徒歩/タクシーへ)。"""
        if from_node not in city.graph.nodes or to_node not in city.graph.nodes:
            return None
        fx, fy = city.node_xy(from_node)
        tx, ty = city.node_xy(to_node)
        best = None
        for route in self.routes:
            i = self._nearest_idx(fx, fy, route)
            j = self._nearest_idx(tx, ty, route)
            if i is None or j is None or j <= i:
                continue
            wait_s, ride_s = self._schedule(route, i, j, call_min)
            key = (round(wait_s, 3), route["name"])
            if best is None or key < best[0]:
                best = (key, route, i, j, wait_s, ride_s)
        if best is None:
            return None
        _key, route, i, j, wait_s, ride_s = best
        return {"line": route["name"], "from": route["stops"][i],
                "to": route["stops"][j], "wait_s": wait_s, "ride_s": ride_s}


def load_bus_table(path, city, *, origin=None, access_radius_m: float = 150.0):
    """静的表 JSON を読み BusTable を返す(不在/壊れは None=呼び出し側は従来近似へ後退)。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    table = BusTable(data, city, origin=origin, access_radius_m=access_radius_m)
    return table if table.routes else None
