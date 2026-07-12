"""公共交通(定刻ダイヤ)。ダイヤ乱れなし=ファイルの時刻どおり決定論的に運行。

現状のダイヤは公表の始発・終電・運転間隔から生成した近似(ファイルに明記)。
★実ダイヤへの切替(v3): conf の transit.gtfs_dir に GTFS(stops.txt / stop_times.txt /
trips.txt / routes.txt)を置いたフォルダを指定すると、渋谷駅の実発車時刻から
路線ごとの始発・終電・間隔を再構成して置き換える。
GTFS の入手 = ODPT(公共交通オープンデータセンター)。developer.odpt.org で無料の
開発者登録(確認に最大2営業日)→ アクセストークン発行 → ckan.odpt.org から
事業者別 GTFS を取得(JR-East / TokyoMetro / Tokyu)。
役割: (1) 駅経由の退出/帰還を運行時間帯に制約(終電後は帰って来られない)
      (2) 到着ごとの人の波(パルス)の源。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _to_min(text: str) -> int:
    parts = text.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _load_gtfs(gtfs_dir: Path) -> list[dict] | None:
    """GTFS から渋谷停車の路線別 始発/終電/中央間隔を再構成する(最小実装)。"""
    try:
        stops = list(csv.DictReader(
            (gtfs_dir / "stops.txt").open(encoding="utf-8-sig")))
        shibuya_ids = {s["stop_id"] for s in stops if "渋谷" in s.get("stop_name", "")
                       or "Shibuya" in s.get("stop_name", "")}
        if not shibuya_ids:
            return None
        trips = {t["trip_id"]: t["route_id"] for t in csv.DictReader(
            (gtfs_dir / "trips.txt").open(encoding="utf-8-sig"))}
        routes = {r["route_id"]: (r.get("route_long_name")
                                  or r.get("route_short_name") or r["route_id"])
                  for r in csv.DictReader(
                      (gtfs_dir / "routes.txt").open(encoding="utf-8-sig"))}
        deps: dict[str, list[int]] = {}
        for st in csv.DictReader(
                (gtfs_dir / "stop_times.txt").open(encoding="utf-8-sig")):
            if st["stop_id"] in shibuya_ids and st.get("departure_time"):
                route = routes.get(trips.get(st["trip_id"], ""), "不明")
                deps.setdefault(route, []).append(_to_min(st["departure_time"]))
        lines = []
        for name, times in sorted(deps.items()):
            times.sort()
            gaps = sorted(b - a for a, b in zip(times, times[1:]) if 0 < b - a < 60)
            headway = gaps[len(gaps) // 2] if gaps else 5
            first, last = times[0] % 1440, times[-1]
            lines.append({"name": name, "first": f"{first // 60}:{first % 60:02d}",
                          "last": f"{last // 60}:{last % 60:02d}",
                          "headway_min": headway,
                          "_first": first, "_last": last % 2880,
                          "source": "GTFS(実ダイヤ)"})
        return lines or None
    except (OSError, KeyError, ValueError):
        return None


class Transit:
    def __init__(self, path: str | Path, gtfs_dir: str | Path | None = None):
        self.source = "approximation"
        # 運休フラグ(都市・環境ショック 後続波 H4。既定 False=通常運行)。災害/運休の日に
        # disaster 層が True にすると駅経由の退出/帰還が不可(交通麻痺)。disaster OFF では常に
        # False のまま=has_service は現行挙動とバイト一致。
        self.suspended = False
        # 簡易バスの路線定義(ユーザー要望 2026-07-06)。既定 OFF なので通常は休眠。
        # ファイルの "bus_lines"(中立名の停留所ノード列)を素の list として保持する。
        raw_file = json.loads(Path(path).read_text(encoding="utf-8"))
        self.bus_lines: list[dict] = list(raw_file.get("bus_lines", []))
        lines = None
        if gtfs_dir:
            lines = _load_gtfs(Path(gtfs_dir))
            if lines:
                self.source = "gtfs"
        if lines is None:
            lines = raw_file["lines"]
            for line in lines:
                line["_first"] = _to_min(line["first"])
                line["_last"] = _to_min(line["last"])  # 24:30 → 1470(翌日 0:30)
        self.lines: list[dict] = lines

    def lines_in_service(self, sim_min: int) -> list[str]:
        """この時刻(シミュ内分)に運行中の路線名。10分 step ≧ 最短間隔なので在/不在で十分。"""
        mod = sim_min % 1440
        names = []
        for line in self.lines:
            first, last = line["_first"], line["_last"]
            in_service = (first <= mod <= last) if last < 1440 else \
                         (mod >= first or mod <= last - 1440)
            if in_service:
                names.append(line["name"])
        return names

    def has_service(self, sim_min: int) -> bool:
        if self.suspended:                     # 運休(災害/運休の日。既定 False=通常運行=不変)
            return False
        return bool(self.lines_in_service(sim_min))


class BusNetwork:
    """簡易バス(ユーザー要望 2026-07-06)。既定 OFF = 本番トグル。

    電車の車内・駅構内の高度化はしない(GTFS 実ダイヤ API は申請待ち)。ここは
    「同一路線の停留所が近い出発/目的の間を、便(headway)ごとに短い区間乗車で結ぶ」
    最小実装。has_service 流儀を踏襲し serves() で便の有無を、find_ride() で乗車可否を
    決定論的に返す(乱数なし)。停留所は node id か {node:...} で与え、地図に存在する
    ノードだけを採用する(合成データ・簡易 JSON の両方に対応)。"""

    def __init__(self, lines, city, stop_radius_m: float = 100.0,
                 headway_steps: int = 1):
        self.radius = float(stop_radius_m)
        self.headway = max(1, int(headway_steps))
        self.lines: list[dict] = []
        for ln in (lines or []):
            stops: list[tuple[str, float, float]] = []
            for s in ln.get("stops", []):
                node = s if isinstance(s, str) else \
                    (s.get("node") if isinstance(s, dict) else None)
                if node is not None and node in city.graph.nodes:
                    x, y = city.node_xy(node)
                    stops.append((node, x, y))
            if len(stops) >= 2:
                self.lines.append({"name": ln.get("name", "循環バス"),
                                   "stops": stops})

    def serves(self, sim_min: int) -> bool:
        """この step に便があるか(headway ごと)。既定 headway=1 step なら常に True。"""
        return ((sim_min // 10) % self.headway) == 0

    def _nearest(self, x: float, y: float, stops):
        best, best_d = None, self.radius
        for node, sx, sy in stops:
            d = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = node, d
        return best

    def find_ride(self, from_node: str, to_node: str, city):
        """出発・目的の両方が同一路線の停留所の近く(<radius)なら {line, from, to} を返す。"""
        if from_node not in city.graph.nodes or to_node not in city.graph.nodes:
            return None
        fx, fy = city.node_xy(from_node)
        tx, ty = city.node_xy(to_node)
        for ln in self.lines:
            a = self._nearest(fx, fy, ln["stops"])
            b = self._nearest(tx, ty, ln["stops"])
            if a is not None and b is not None and a != b:
                return {"line": ln["name"], "from": a, "to": b}
        return None
