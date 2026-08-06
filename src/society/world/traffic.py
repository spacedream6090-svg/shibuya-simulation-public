"""背景交通(自動車フロー)。2 モード。

- ambient(既定): エージェントではない「通過車両」を実規模で流す(現行実装)。
    発生 = 車用ゲートウェイ間を Poisson 到着でスポーン(時間帯プロファイル付き)、
    移動 = 車道グラフの最短経路を道なりに進む。決定論 = stream("traffic", step)。
- od: 車を個体化する。各車が出発地(ゲートウェイ)と目的地(別ゲートウェイ=通過交通、
    または域内の POI 近傍ノード)を決め、一方通行を尊重した最短路を辿る。信号は
    ノードごとの決定論的な平均遅延、車線数はエッジ容量(超過で渋滞減速)として効かせる。
    サイドカー data/traffic_features_shibuya.json(scripts/build_traffic.py 生成)を読む。

od は「本番だけ ON」を想定(config: world.traffic.mode)。既定 ambient は挙動不変。
このモジュールは走行の物理だけを扱い、エージェントの内部状態には一切触れない。
乱数は必ず RngHub.stream の新規ストリームから取る(D13)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

from .geom import rdp

REPO_ROOT = Path(__file__).resolve().parents[3]
STEP_SECONDS = 600                       # 正準 Δt での 1 step = 10 分 = 600 秒
STEP_MINUTES = 10                        # 正準 Δt。実行時は TrafficFlow.step_minutes

# 時間帯プロファイル(0時〜23時の相対重み。朝夕ピーク・深夜減)
HOURLY = [0.20, 0.12, 0.10, 0.10, 0.15, 0.35, 0.60, 1.00, 1.35, 1.25,
          1.05, 1.05, 1.10, 1.05, 1.05, 1.10, 1.20, 1.40, 1.45, 1.25,
          1.00, 0.80, 0.55, 0.35]
_HOURLY_SUM = sum(HOURLY)
_HOURLY_MEAN = _HOURLY_SUM / 24.0


def _hash_frac(text: str) -> float:
    """文字列 → [0,1) の決定論値(プロセス非依存の sha256)。信号の位相・赤率に使う。"""
    h = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:6], "big")
    return (h % 100000) / 100000.0


class TrafficFlow:
    def __init__(self, city, router, hub, *, enabled: bool = True,
                 cars_per_day: int = 30000, max_log: int = 120,
                 step_minutes: int = STEP_MINUTES):
        self.city = city
        self.router = router
        self.hub = hub
        self.enabled = enabled and len(getattr(city, "car_gateways", [])) >= 2
        self.cars_per_day = int(cars_per_day)
        self.max_log = int(max_log)
        # 中央 Δt(第79バッチ)。既定 10 = 従来の直書き(600 秒 / 6 step 毎時 / 144 step 毎日)。
        # ★TODO(§1.2 B4 / R6・第94バッチ OBS-U2 で棚卸し): 以下 4 行は Clock と同じ式を
        #   **自前で再導出している第2の Δt 源**である(clock.py:26 の「派生量の単一源」設計に
        #   反する)。TrafficFlow は Clock を持たず step_minutes だけを受け取る配線なので今は
        #   追随するが、Clock 側の定義だけを将来変えると静かに不整合になる。直すなら
        #   コンストラクタで clock を受け取り steps_per_hour / steps_per_day / step_seconds を
        #   委譲する(第94では挙動を 1 ビットも変えないため据え置き)。
        self.step_minutes = int(step_minutes)
        self.steps_per_hour = max(1, 60 // self.step_minutes)
        self.steps_per_day = max(1, 1440 // self.step_minutes)
        self.step_seconds = float(self.step_minutes) * 60.0
        # 速度スケール(m/step)= RATE 類。Δt=10 で厳密に 1.0(恒等=バイト一致)。
        self._v = self.step_minutes / float(STEP_MINUTES)
        self.gates = list(getattr(city, "car_gateways", []))
        self.cars: list[dict] = []          # {route:[nodes], node, offset, speed}
        self.n_active = 0                   # step をまたいで走行中の車
        self.last_n = 0                     # この step に走った車(通過含む)
        self.total_spawned = 0
        # ---- モード(既定 ambient=現行挙動)。ensure_mode() で config から遅延設定 ----
        self.mode = "ambient"
        self._configured = False
        self._last_cars: list[dict] = []    # od: payload 用の現在位置スナップショット
        self.max_cars_log = 200
        self.total_arrived = 0              # od: 目的地/域外に到達して消えた車
        self.sum_travel_steps = 0          # od: 旅行 step の総和(平均の分子)
        self.jam_events = 0                 # od: 容量超過で減速した (車, step) 回数

    # ------------------------------------------------------------------ #
    # モード設定(scheduler が毎 step 呼ぶ。ambient では何もしない)
    # ------------------------------------------------------------------ #
    def ensure_mode(self, cfg) -> None:
        """config から od の遅延セットアップを 1 度だけ行う。ambient は完全に no-op。"""
        if self._configured:
            return
        self._configured = True
        tcfg = cfg.world.get("traffic", {}) if hasattr(cfg, "world") else {}
        if str(tcfg.get("mode", "ambient")) == "od":
            self.setup_od(tcfg)

    def setup_od(self, tcfg) -> None:
        """od モードの初期化: サイドカー読み込み+車道有向グラフ+信号+出入口。

        失敗(ファイル無し・ゲートウェイ不足)時は ambient のまま(sim は止めない)。
        """
        if not self.enabled:
            return
        feat = self._load_features(tcfg.get("features_file"))
        if feat is None:
            return
        self.through_ratio = float(tcfg.get("through_ratio", 0.7))
        self.capacity_per_lane = int(tcfg.get("capacity_per_lane", 8))
        self.default_lanes = int(tcfg.get("default_lanes", 2))
        self.signal_cycle_s = float(tcfg.get("signal_cycle_s", 90.0))
        self.od_cars_per_day = int(tcfg.get("od_cars_per_day", 8000))
        self.max_cars_log = int(tcfg.get("max_cars_log", 200))
        self.free_speed = (float(tcfg.get("free_speed_min", 3000.0)),
                           float(tcfg.get("free_speed_max", 5000.0)))

        dg = self._build_car_graph(feat)
        self.dg = dg
        self.od_gateways = sorted(g for g in feat.get("gateways", []) if g in dg)
        if len(self.od_gateways) < 2:
            return                          # 出入口が足りない → ambient のまま
        # 信号: ノードごとに決定論の「赤率」。位相は node id ハッシュでずらす。
        # ★1 step(600s)は信号周期(既定 90s)より長い。緑赤の瞬時状態は意味を持たない
        #   ので、通過する車には「一様到着を仮定した期待待ち時間」を距離換算で差し引く近似。
        self.signal_red = {s["node"]: 0.35 + 0.20 * _hash_frac(s["node"])
                           for s in feat.get("signals", []) if s["node"] in dg}
        self.signal_nodes = set(self.signal_red)
        self.od_dest_nodes = self._domain_dest_nodes(dg)
        self._route_cache: dict[tuple[str, str], list[str] | None] = {}
        self.mode = "od"

    def _load_features(self, path) -> dict | None:
        p = Path(str(path)) if path else (REPO_ROOT / "data"
                                          / "traffic_features_shibuya.json")
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _build_car_graph(self, feat: dict) -> nx.DiGraph:
        """サイドカー edges("u|v": {lanes,oneway,class})→ 一方通行を尊重した有向車道グラフ。

        oneway: 0=双方向, +1=u→v のみ, -1=v→u のみ。長さ・幾何はコア地図から引く。
        """
        dg = nx.DiGraph()
        city = self.city
        for key, attr in feat.get("edges", {}).items():
            u, v = key.split("|", 1)
            if not (city.graph.has_edge(u, v)):
                continue
            length = city.edge_length(u, v)
            lanes = attr.get("lanes") or self.default_lanes
            ow = int(attr.get("oneway", 0))
            if ow >= 0:                     # u→v 許可(双方向 or u→v のみ)
                dg.add_edge(u, v, length=length, lanes=int(lanes))
            if ow <= 0:                     # v→u 許可(双方向 or v→u のみ)
                dg.add_edge(v, u, length=length, lanes=int(lanes))
        return dg

    def _domain_dest_nodes(self, dg: nx.DiGraph) -> list[str]:
        """域内目的地の候補 = 駐車可能とみなす POI 近傍の車道ノード(重複除去・ソート)。"""
        city = self.city
        graph_nodes = list(dg.nodes)
        xy = {n: city.node_xy(n) for n in graph_nodes}
        dests: set[str] = set()
        for poi in getattr(city, "poi_list", []):
            px, py = float(poi["x"]), float(poi["y"])
            best, best_d = None, 1e18
            for n in graph_nodes:
                nx_, ny_ = xy[n]
                d = (nx_ - px) ** 2 + (ny_ - py) ** 2
                if d < best_d:
                    best, best_d = n, d
            if best is not None:
                dests.add(best)
        return sorted(dests) if dests else sorted(graph_nodes)

    # ------------------------------------------------------------------ #
    # 走行 1 step
    # ------------------------------------------------------------------ #
    def step(self, step: int, sim_min: int) -> list[dict]:
        """1step 進める。戻り値 = この step に各車が通った折れ線(可視化・ログ用)。"""
        if not self.enabled:
            return []
        if self.mode == "od":
            return self._step_od(step, sim_min)

        # ---------------- ambient(現行実装。一切変更しない)----------------
        rng = self.hub.stream("traffic", step)
        lam = (self.cars_per_day * HOURLY[(sim_min % 1440) // 60]
               / (_HOURLY_SUM * self.steps_per_hour))
        for _ in range(int(rng.poisson(lam))):
            car = self._spawn(rng)
            if car:
                self.cars.append(car)
                self.total_spawned += 1

        segs: list[dict] = []
        alive: list[dict] = []
        for car in self.cars:
            pts: list[tuple[float, float]] = []
            x, y = self.city.xy_along(car["node"], car["route"][0], car["offset"]) \
                if car["route"] else self.city.node_xy(car["node"])
            pts.append((x, y))
            budget = car["speed"]
            while budget > 0 and car["route"]:
                edge_len = self.city.edge_length(car["node"], car["route"][0])
                remain = edge_len - car["offset"]
                if budget >= remain:
                    budget -= remain
                    car["node"] = car["route"].pop(0)
                    car["offset"] = 0.0
                    pts.append(self.city.node_xy(car["node"]))
                else:
                    car["offset"] += budget
                    budget = 0.0
                    pts.append(self.city.xy_along(car["node"], car["route"][0],
                                                  car["offset"]))
            # 形状を保って間引く(等間隔間引きは曲がり角が落ちて道路外に見えた)
            pts = rdp(pts, epsilon=5.0, max_points=24)
            segs.append({"pts": [[round(p[0], 1), round(p[1], 1)] for p in pts]})
            if car["route"]:
                alive.append(car)
        self.cars = alive
        self.n_active = len(alive)
        self.last_n = len(segs)
        return segs

    def _spawn(self, rng) -> dict | None:
        for _ in range(4):                   # 到達不能ペアは引き直し(最大4回)
            i = int(rng.integers(len(self.gates)))
            j = int(rng.integers(len(self.gates)))
            if i == j:
                continue
            path, mode = self.router.route(self.gates[i], self.gates[j], "car")
            if mode == "car" and len(path) >= 2:
                # 実勢速度 20〜35km/h(信号込み)→ 1step(10分)あたり 3300〜5800m
                speed = float(rng.uniform(3300.0, 5800.0))
                if self._v != 1.0:              # m/step = RATE(Δt=10 では通らない)
                    speed *= self._v
                return {"route": path[1:], "node": path[0], "offset": 0.0,
                        "speed": speed}
        return None

    # ------------------------------------------------------------------ #
    # od 走行
    # ------------------------------------------------------------------ #
    def _od_route(self, src: str, dst: str) -> list[str] | None:
        key = (src, dst)
        if key not in self._route_cache:
            try:
                self._route_cache[key] = nx.shortest_path(self.dg, src, dst,
                                                           weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._route_cache[key] = None
        cached = self._route_cache[key]
        return list(cached) if cached else None

    def _od_spawn_one(self, gate: str, rng) -> dict | None:
        """1 台の車を出発地 gate から発生させ、OD と経路(一方通行尊重)を決める。"""
        through = bool(rng.random() < self.through_ratio)
        for _ in range(6):                   # 到達不能な OD は引き直し
            if through:
                other = self.od_gateways[int(rng.integers(len(self.od_gateways)))]
                dst = other
            else:
                dst = self.od_dest_nodes[int(rng.integers(len(self.od_dest_nodes)))]
            if dst == gate:
                continue
            path = self._od_route(gate, dst)
            if path and len(path) >= 2:
                speed = float(rng.uniform(*self.free_speed))
                return {"route": path[1:], "node": path[0], "offset": 0.0,
                        "speed": speed}
        return None

    def _signal_delay_m(self, node: str, speed: float) -> float:
        """信号 1 基あたりの期待遅延を距離(m)に換算。r=赤率、周期 C のとき
        一様到着の期待待ち = r²·C/2。それを当該速度で走れた距離に読み替えて差し引く。"""
        r = self.signal_red.get(node, 0.0)
        delay_s = r * r * self.signal_cycle_s * 0.5
        return speed * delay_s / self.step_seconds

    def _step_od(self, step: int, sim_min: int) -> list[dict]:
        weight = HOURLY[(sim_min % 1440) // 60]
        base = self.od_cars_per_day / float(self.steps_per_day) * (weight / _HOURLY_MEAN)
        lam_gate = base / len(self.od_gateways)
        # スポーン: 出入口ごとに Poisson。乱数は新 stream("car_spawn", gate_idx, step)。
        for gi, gate in enumerate(self.od_gateways):
            rng = self.hub.stream("car_spawn", gi, step)
            for _ in range(int(rng.poisson(lam_gate))):
                car = self._od_spawn_one(gate, rng)
                if car:
                    car["id"] = self.total_spawned
                    car["spawn"] = step
                    self.cars.append(car)
                    self.total_spawned += 1

        # 車線容量: この step の各有向エッジの車数を先に数える(超過で減速)。
        occ: dict[tuple[str, str], int] = {}
        for car in self.cars:
            if car["route"]:
                key = (car["node"], car["route"][0])
                occ[key] = occ.get(key, 0) + 1

        segs: list[dict] = []
        cars_pos: list[dict] = []
        alive: list[dict] = []
        for car in self.cars:
            node, route, offset = car["node"], car["route"], car["offset"]
            lanes = self._edge_lanes(node, route[0]) if route else self.default_lanes
            cap = self.capacity_per_lane * lanes
            count = occ.get((node, route[0]), 0) if route else 0
            factor = 1.0 if count <= cap else max(0.35, cap / count)
            if factor < 1.0:
                self.jam_events += 1

            pts: list[tuple[float, float]] = []
            x, y = self.city.xy_along(node, route[0], offset) if route \
                else self.city.node_xy(node)
            pts.append((x, y))
            budget = car["speed"] * factor
            while budget > 0 and route:
                edge_len = self.city.edge_length(node, route[0])
                remain = edge_len - offset
                if budget >= remain:
                    budget -= remain
                    node = route.pop(0)
                    offset = 0.0
                    pts.append(self.city.node_xy(node))
                    if route and node in self.signal_nodes:   # 信号ノードを通過
                        budget = max(0.0, budget
                                     - self._signal_delay_m(node, car["speed"]))
                else:
                    offset += budget
                    budget = 0.0
                    pts.append(self.city.xy_along(node, route[0], offset))
            car["node"], car["offset"] = node, offset
            pts = rdp(pts, epsilon=5.0, max_points=24)
            segs.append({"pts": [[round(p[0], 1), round(p[1], 1)] for p in pts]})
            cx, cy = self.city.xy_along(node, route[0], offset) if route \
                else self.city.node_xy(node)
            cars_pos.append({"id": car["id"], "x": round(cx, 1), "y": round(cy, 1)})
            if route:                                  # まだ目的地に着いていない
                alive.append(car)
            else:                                      # 到達 → despawn
                self.total_arrived += 1
                self.sum_travel_steps += step - car["spawn"] + 1

        self.cars = alive
        self.n_active = len(alive)
        self.last_n = len(segs)
        self._last_cars = cars_pos[:self.max_cars_log]
        return segs

    def _edge_lanes(self, u: str, v: str) -> int:
        if self.dg.has_edge(u, v):
            return int(self.dg.edges[u, v].get("lanes", self.default_lanes))
        return self.default_lanes

    # ------------------------------------------------------------------ #
    # ログ payload の追加分(ambient では空 = 現行と完全同一)
    # ------------------------------------------------------------------ #
    def log_extra(self) -> dict:
        if self.mode != "od":
            return {}
        return {"mode": "od", "cars": self._last_cars}
