"""交通 v2(信号・車線・OD)のテスト。

- ambient: 変更前に採取したゴールデン(seed 42, 72 step)と完全一致 = 現行挙動を保持。
- build_traffic.build_features: 合成 OSM+コアで車線/一方通行/信号/ゲートウェイ抽出(無ネット)。
- od: スポーン決定論 / 一方通行尊重 / 通過車の到達・消滅 / 容量超過で減速 / payload 後方互換。
- mock 完走: world.traffic.mode=od で 20人×48step が exit 0、車の統計が出る。

od 系は実生成物 data/traffic_features_shibuya.json を使う(無ければ skip)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society.config import load_config
from society.engine.simulation import Simulation
from society.rng import RngHub
from society.world.clock import Clock
from society.world.map import CityMap
from society.world.routing import Router
from society.world.traffic import TrafficFlow

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CORE = DATA / "shibuya_osm.json"
FEATURES = DATA / "traffic_features_shibuya.json"

# 変更前(周期的背景交通)に採取したゴールデン。ambient の挙動不変の担保。
AMBIENT_DIGEST = "b74967f55daf98b0b5621eed90a787088ba1106b4786aea79617ad14fdbb68e2"


def _load_build_traffic():
    spec = importlib.util.spec_from_file_location(
        "build_traffic", REPO / "scripts" / "build_traffic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BT = _load_build_traffic()
_needs_features = pytest.mark.skipif(
    not FEATURES.exists(), reason="traffic_features 未生成(build_traffic.py)")


# --------------------------------------------------------------------------- #
# 共有(重い CityMap ロードを1回に)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def city():
    return CityMap(CORE)


@pytest.fixture(scope="module")
def od_cfg():
    return load_config(overrides=["world.traffic.mode=od"])


def _make_od_tf(city, cfg, seed: int) -> TrafficFlow:
    tf = TrafficFlow(city, Router(city), RngHub(seed), enabled=True,
                     cars_per_day=30000, max_log=120)
    tf.ensure_mode(cfg)
    return tf


def _od_trace(tf: TrafficFlow, steps: int) -> list:
    clk = Clock()
    out = []
    for step in range(steps):
        segs = tf.step(step, clk.sim_min(step))
        out.append((len(segs), tf.total_spawned, tf.total_arrived,
                    tuple((c["id"], c["x"], c["y"]) for c in tf._last_cars)))
    return out


# --------------------------------------------------------------------------- #
# 1. ambient ゴールデン(現行挙動の保持)
# --------------------------------------------------------------------------- #
def test_ambient_matches_prechange_golden(city):
    hub, clk = RngHub(42), Clock()
    tf = TrafficFlow(city, Router(city), hub, enabled=True,
                     cars_per_day=30000, max_log=120)
    rows = []
    for step in range(72):
        segs = tf.step(step, clk.sim_min(step))
        rows.append([step, len(segs), tf.total_spawned,
                     [s["pts"] for s in segs]])
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    assert hashlib.sha256(blob.encode()).hexdigest() == AMBIENT_DIGEST
    assert tf.mode == "ambient"
    assert tf.log_extra() == {}                 # ambient は payload 追加なし


def test_ambient_unaffected_by_ensure_mode_default(city):
    """既定 config(mode 未指定=ambient)で ensure_mode を呼んでも ambient のまま。"""
    cfg = load_config()
    tf = TrafficFlow(city, Router(city), RngHub(1), enabled=True,
                     cars_per_day=30000, max_log=120)
    tf.ensure_mode(cfg)
    assert tf.mode == "ambient"
    assert tf.log_extra() == {}


# --------------------------------------------------------------------------- #
# 2. build_traffic.build_features(合成データ・無ネット)
# --------------------------------------------------------------------------- #
def test_build_features_lanes_oneway_signal_gateway():
    proj = BT.BM.project
    lls = {1: (35.6595, 139.7000), 2: (35.6595, 139.70055),
           3: (35.6595, 139.7006), 4: (35.6595, 139.7009)}
    raw = {"elements": [
        {"type": "node", "id": 1, "lat": lls[1][0], "lon": lls[1][1]},
        {"type": "node", "id": 2, "lat": lls[2][0], "lon": lls[2][1],
         "tags": {"highway": "traffic_signals"}},
        {"type": "node", "id": 3, "lat": lls[3][0], "lon": lls[3][1]},
        {"type": "node", "id": 4, "lat": lls[4][0], "lon": lls[4][1]},
        # way A: n1-n2-n3 一方通行(yes=ノード順) 2車線
        {"type": "way", "id": 10, "nodes": [1, 2, 3],
         "tags": {"highway": "secondary", "oneway": "yes", "lanes": "2"}},
        # way B: n3-n4 双方向 1車線
        {"type": "way", "id": 11, "nodes": [3, 4],
         "tags": {"highway": "secondary", "lanes": "1"}},
    ]}
    p1, p2, p3, p4 = (proj(*lls[i]) for i in (1, 2, 3, 4))
    core = {
        "meta": {"name": "synthetic"},
        "nodes": [{"id": "n1", "x": p1[0], "y": p1[1]},
                  {"id": "n3", "x": p3[0], "y": p3[1]},
                  {"id": "n4", "x": p4[0], "y": p4[1]}],
        "edges": [
            {"u": "n1", "v": "n3", "klass": "secondary", "layer": 0,
             "geometry": [list(p1), list(p2), list(p3)]},
            {"u": "n3", "v": "n4", "klass": "secondary", "layer": 0,
             "geometry": [list(p3), list(p4)]},
        ],
        "car_gateways": ["n1", "n4"],
    }
    bbox = (35.658, 139.699, 35.661, 139.702)       # ノードは内部(縁から>60m)
    feat = BT.build_features(raw, core, bbox)

    assert feat["edges"]["n1|n3"] == {"lanes": 2, "oneway": 1,
                                      "class": "secondary"}
    assert feat["edges"]["n3|n4"] == {"lanes": 1, "oneway": 0,
                                      "class": "secondary"}
    # 信号は最寄り車道ノード(n3)へ写像される
    assert [s["node"] for s in feat["signals"]] == ["n3"]
    # 内部ノードのみ=厳密ゲートウェイ空 → コア地図の car_gateways へフォールバック
    assert feat["gateways"] == ["n1", "n4"]
    assert feat["meta"]["n_edges"] == 2
    assert feat["meta"]["lane_data_rate"] == 1.0


def test_build_features_reverse_oneway():
    """oneway=-1(way ノード順の逆)→ コア edge の u→v が禁止(oneway=-1)。"""
    proj = BT.BM.project
    a, b = (35.6595, 139.7000), (35.6595, 139.7006)
    raw = {"elements": [
        {"type": "node", "id": 1, "lat": a[0], "lon": a[1]},
        {"type": "node", "id": 2, "lat": b[0], "lon": b[1]},
        {"type": "way", "id": 10, "nodes": [1, 2],
         "tags": {"highway": "tertiary", "oneway": "-1"}},
    ]}
    pa, pb = proj(*a), proj(*b)
    core = {"meta": {}, "nodes": [{"id": "n1", "x": pa[0], "y": pa[1]},
                                  {"id": "n2", "x": pb[0], "y": pb[1]}],
            "edges": [{"u": "n1", "v": "n2", "klass": "tertiary", "layer": 0,
                       "geometry": [list(pa), list(pb)]}],
            "car_gateways": ["n1", "n2"]}
    feat = BT.build_features(raw, core, (35.658, 139.699, 35.661, 139.702))
    assert feat["edges"]["n1|n2"]["oneway"] == -1


# --------------------------------------------------------------------------- #
# 3. od: グラフが一方通行を尊重
# --------------------------------------------------------------------------- #
@_needs_features
def test_od_graph_respects_oneway(city, od_cfg):
    tf = _make_od_tf(city, od_cfg, 1)
    assert tf.mode == "od"
    assert len(tf.od_gateways) >= 2
    feat = json.loads(FEATURES.read_text(encoding="utf-8"))
    checked = 0
    for key, attr in feat["edges"].items():
        u, v = key.split("|", 1)
        if not city.graph.has_edge(u, v):
            continue
        ow = int(attr["oneway"])
        if ow == 1:
            assert tf.dg.has_edge(u, v) and not tf.dg.has_edge(v, u)
            checked += 1
        elif ow == -1:
            assert tf.dg.has_edge(v, u) and not tf.dg.has_edge(u, v)
            checked += 1
        else:
            assert tf.dg.has_edge(u, v) and tf.dg.has_edge(v, u)
    assert checked > 0                            # 一方通行エッジが実在する


@_needs_features
def test_od_routes_only_use_directed_edges(city, od_cfg):
    """最短路は有向グラフ上のみ=逆走(禁止方向の通過)を含まない。"""
    tf = _make_od_tf(city, od_cfg, 2)
    found = False
    for src in tf.od_gateways:
        for dst in tf.od_gateways:
            if src == dst:
                continue
            path = tf._od_route(src, dst)
            if path and len(path) >= 3:
                for a, b in zip(path, path[1:]):
                    assert tf.dg.has_edge(a, b), "経路が有向エッジ外を通っている"
                found = True
                break
        if found:
            break
    assert found, "多ホップの通過経路が見つからない"


# --------------------------------------------------------------------------- #
# 4. od: スポーン決定論
# --------------------------------------------------------------------------- #
@_needs_features
def test_od_spawn_is_deterministic(city, od_cfg):
    t1 = _od_trace(_make_od_tf(city, od_cfg, 7), 30)
    t2 = _od_trace(_make_od_tf(city, od_cfg, 7), 30)
    assert t1 == t2
    t3 = _od_trace(_make_od_tf(city, od_cfg, 8), 30)
    assert t1 != t3                               # seed 違いは別系列


# --------------------------------------------------------------------------- #
# 5. od: 通過車がゲートウェイ→ゲートウェイに到達して消える
# --------------------------------------------------------------------------- #
@_needs_features
def test_od_through_car_reaches_and_despawns(city, od_cfg):
    tf = _make_od_tf(city, od_cfg, 3)
    # 出入口間で経路が引ける通過 OD を1つ選ぶ
    src = dst = path = None
    for s in tf.od_gateways:
        for d in tf.od_gateways:
            if s == d:
                continue
            p = tf._od_route(s, d)
            if p and len(p) >= 2:
                src, dst, path = s, d, p
                break
        if path:
            break
    assert path is not None, "到達可能な通過 OD が無い"
    tf.od_cars_per_day = 0                         # 新規スポーンを止めて注入車だけ追う
    tf.cars = [{"route": path[1:], "node": path[0], "offset": 0.0,
                "speed": 3000.0, "id": 999, "spawn": 0}]
    tf.total_spawned = 1
    clk = Clock()
    for step in range(1, 60):
        tf.step(step, clk.sim_min(step))
        if not tf.cars:
            break
    assert tf.cars == [], "通過車が消えない(despawn していない)"
    assert tf.total_arrived == 1
    assert tf.city.node_xy(dst)                    # 目的地=別ゲートウェイに到達


# --------------------------------------------------------------------------- #
# 6. od: 車線容量の超過で減速(渋滞)
# --------------------------------------------------------------------------- #
@_needs_features
def test_od_capacity_overflow_decelerates(city, od_cfg):
    tf = _make_od_tf(city, od_cfg, 4)
    tf.od_cars_per_day = 0                         # スポーン停止
    u, v = next(iter(tf.dg.edges()))
    lanes = tf._edge_lanes(u, v)
    cap = tf.capacity_per_lane * lanes
    clk = Clock()

    def _car(k):
        return {"route": [v], "node": u, "offset": 0.0,
                "speed": 3000.0, "id": k, "spawn": 0}

    tf.jam_events = 0                             # 1台=容量内=減速なし
    tf.cars = [_car(0)]
    tf.step(1, clk.sim_min(1))
    assert tf.jam_events == 0

    tf.jam_events = 0                             # 容量超過=全車が減速
    tf.cars = [_car(k) for k in range(cap + 20)]
    tf.step(2, clk.sim_min(2))
    assert tf.jam_events > 0


# --------------------------------------------------------------------------- #
# 7. payload 後方互換 + mock 完走(20人×48step, exit 0)
# --------------------------------------------------------------------------- #
def _traffic_payloads(parquet_path: Path) -> list:
    rows = pq.read_table(parquet_path).to_pylist()
    return [json.loads(r["payload"]) for r in rows
            if r["kind"] == "traffic_flow" and r["payload"]]


def test_ambient_run_payload_backward_compatible(tmp_path):
    cfg = load_config(overrides=["run.seed=5", "run.n_agents=8",
                                 "run.n_steps=24", "run.name=amb"])
    Simulation(cfg, out_dir=tmp_path / "amb").run()
    tfs = _traffic_payloads(tmp_path / "amb" / "l1_events.parquet")
    assert tfs
    for p in tfs:
        assert set(p) >= {"n", "total", "segs"}   # 既存キー不変
        assert "mode" not in p                     # ambient は追加キー無し


@_needs_features
def test_od_mock_run_completes_with_stats(tmp_path):
    cfg = load_config(overrides=[
        "run.seed=5", "run.n_agents=20", "run.n_steps=48", "run.name=odrun",
        "world.traffic.mode=od"])
    sim = Simulation(cfg, out_dir=tmp_path / "odrun")
    summary = sim.run()

    assert summary["n_events"] > 100
    assert sim.traffic.mode == "od"
    assert sim.traffic.total_spawned > 0
    assert sim.traffic.total_arrived > 0

    tfs = _traffic_payloads(tmp_path / "odrun" / "l1_events.parquet")
    assert tfs
    for p in tfs:
        assert set(p) >= {"n", "total", "segs"}   # 後方互換キーは維持
    assert any(p.get("mode") == "od" for p in tfs)
    assert any(p.get("cars") for p in tfs)         # 車の現在位置が載る
