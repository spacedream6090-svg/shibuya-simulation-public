"""SUMO ライブ連成タクシー配車 v-Ride-1 のテスト。

正典: docs/research/sumo-live-transit.md(ハイブリッド連成(b)・taxi device・dispatch=traci・
go/no-go 4条件)+ docs/research/traffic-indirect-effects.md(§5 チェックリスト・遅延の観測可能性)。

方針(既存の鉄則を継承):
- **既定 OFF**: transit_ride.live.enabled=false のとき sim.taxi_live is None=SUMO を一切起動しない=
  純粋既定と L1 完全一致(ゴールデン golden_baseline_l1.json を守る)・taxi_unmatched 0 件・
  ride payload に wait_s 等が付かない。
- **mock backend**(SUMO 不要の純関数ブリッジ): 決定論(同 seed 2回で L1 バイト一致)・
  予約→wait_s/ride_s→到着 step 追加待ちの整合・観測記録(wait_s/ride_s/delay_s)・
  R1 の k 不変性(compute_matched 下で k=free==k=off 呼数一致)・連成が LLM を1本も足さないこと。
- **traci backend**(実 SUMO): SUMO があれば実走して同 seed 2回で (wait_s, ride_s, delay_s, hold_steps)
  列がバイト一致することを示す(go/no-go 条件②)。SUMO 不在環境では skip(import/他テストは壊れない)。
検証は mock を主・実 SUMO は環境ガード付き(validation-runs-short: ≤数分の小規模のみ)。
"""
from __future__ import annotations

import json
import math

import pytest

from society import transit_live
from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_MAP_ORIGIN_NET = "runs/demo_event_200a3d/sumo/shibuya_osm_car.net.xml"


# --------------------------------------------------------------------------- helpers
def _sim(tmp_path, name, n=20, steps=30, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _FixedLLM:
    """挙動を固定する backend(応答をプロンプト非依存に)。呼数だけ数える(test_commerce と同型)。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


_MOCK = {"transit_ride.taxi.prob": 1.0,
         "transit_ride.live.enabled": "true", "transit_ride.live.backend": "mock"}


def _sumo_available() -> bool:
    """実 SUMO(sumo 実行ファイル + traci python)が使えるか。無ければ SUMO 必須テストは skip。"""
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import sumo_pipeline as sp
        if not sp.find_bin("sumo"):
            return False
        sp.ensure_tools_on_path()
        import traci   # noqa: F401
        import sumolib  # noqa: F401
        return Path(_MAP_ORIGIN_NET).is_file()
    except Exception:
        return False


# --------------------------------------------------------------------- 既定 OFF
def test_off_matches_pure_default_and_no_bridge(tmp_path):
    """明示 live OFF と純粋既定が L1 完全一致・taxi_live is None=SUMO 非起動・観測列も不在。"""
    pure = _sim(tmp_path, "pure", steps=30, **{"transit_ride.taxi.prob": 1.0})
    pure.run()
    off = _sim(tmp_path, "off", steps=30,
               **{"transit_ride.taxi.prob": 1.0, "transit_ride.live.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "live OFF が純粋既定と不一致(連成 seam が no-op でない)"
    assert pure.taxi_live is None and off.taxi_live is None, "既定で taxi_live が生成されている"
    assert not _kind(pure, "taxi_unmatched"), "OFF で taxi_unmatched が出ている"
    for e in _kind(pure, "ride"):
        assert "wait_s" not in e.payload, "OFF の ride に wait_s が付いている(観測が漏れている)"


def test_default_config_live_is_none():
    """既定 config で live_cfg is None(=SUMO を起動しない)。"""
    assert transit_live.live_cfg(load_config([])) is None


# --------------------------------------------------------------------- mock 決定論
def test_mock_coupling_deterministic(tmp_path):
    """mock backend の連成 ON 同士 2 回で L1 完全一致(乗車列・到着・全イベントの決定論)。"""
    a = _sim(tmp_path, "det_a", **_MOCK)
    a.run()
    b = _sim(tmp_path, "det_b", **_MOCK)
    b.run()
    assert type(a.taxi_live).__name__ == "TaxiLive", "mock で taxi_live が生成されていない"
    assert _l1(a) == _l1(b), "mock ON の決定論が崩れている(同 seed でバイト不一致)"
    assert _kind(a, "ride"), "mock ON で乗車(ride)が 1 件も出ていない"


# --------------------------------------------------------------------- 観測記録
def test_ride_payload_records_wait_ride_delay(tmp_path):
    """ライブ乗車の ride payload に wait_s/ride_s/delay_s が必ず載る(条件(5)遅延の観測可能性)。
    delay_s>=0(自由流超過)・quantize_hold 整合・未配車は taxi_unmatched で観測可能化。"""
    sim = _sim(tmp_path, "obs", **_MOCK)
    sim.run()
    rides = [e for e in _kind(sim, "ride") if e.payload["mode"] == "taxi"]
    assert rides, "ライブ taxi 乗車が 1 件も出ていない"
    for e in rides:
        assert "wait_s" in e.payload and "ride_s" in e.payload \
            and "delay_s" in e.payload, f"ride payload に観測欠落: {e.payload}"
        assert e.payload["wait_s"] >= 0 and e.payload["ride_s"] >= 0
        assert e.payload["delay_s"] >= 0, "delay_s(自由流超過)が負"
    # 少なくとも 1 件は捕まらない(未配車)を観測(mock は決定論分岐で unmatched を含む設計)
    assert _kind(sim, "taxi_unmatched"), "taxi_unmatched(捕まらない)が観測できていない"


def test_quantize_hold_pure():
    """到着 step 追加待ちの量子化は純関数: ceil((wait+delay)/600)・追加遅延ゼロは 0。"""
    assert transit_live.quantize_hold(0.0, 0.0) == 0
    assert transit_live.quantize_hold(300.0, 0.0) == 1
    assert transit_live.quantize_hold(600.0, 0.0) == 1
    assert transit_live.quantize_hold(601.0, 0.0) == 2
    assert transit_live.quantize_hold(300.0, 400.0) == 2      # 700s → 2 step


def test_reservation_pickup_dropoff_consistency(tmp_path):
    """予約→wait_s/ride_s→delay_s/hold_steps の整合(TaxiLive + MockBridge の単体)。"""
    sim = _sim(tmp_path, "cons", n=4, steps=1, **_MOCK)
    tl = sim.taxi_live
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    src = nodes[0]
    sx, sy = sim.city.node_xy(src)
    dst = max(nodes, key=lambda n: math.hypot(*(a - b for a, b in
              zip(sim.city.node_xy(n), (sx, sy)))))
    fx, fy = sim.city.node_xy(src)
    tx, ty = sim.city.node_xy(dst)
    seen_match = seen_unmatch = False
    for sm in range(400, 900, 20):     # 複数時刻で予約を投げ、matched/unmatched 双方を確認
        res = tl.request(agent_id=0, from_node=src, to_node=dst,
                         from_xy=(fx, fy), to_xy=(tx, ty), step=1, sim_min=sm)
        if not res["matched"]:
            seen_unmatch = True
            continue
        seen_match = True
        assert res["wait_s"] > 0 and res["ride_s"] > 0
        free_s = math.hypot(fx - tx, fy - ty) / tl.car_m_per_s
        assert abs(res["delay_s"] - max(0.0, res["ride_s"] - free_s)) < 0.2
        assert res["hold_steps"] == transit_live.quantize_hold(res["wait_s"], res["delay_s"])
    assert seen_match, "matched な予約が 1 件も出ていない"
    assert tl.stats()["requests"] > 0


# --------------------------------------------------------------------- R1 呼数
def _run_k(tmp_path, name, *, writeback):
    sim = _sim(tmp_path, name, **{**_MOCK, "controls.mode": "compute_matched",
                                  "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_llm_count_k_invariant_under_live(tmp_path):
    """ライブ連成 ON でも generate 呼数は k(writeback)に依存しない(R1 の中核)。連成は物理量
    (位置・時刻・予約・fleet)だけを見て配車・遅延を決め、k/belief を一切食わせないため、
    compute_matched 下で k=free と k=off の呼数が完全一致する(commerce/career と同型)。"""
    free = _run_k(tmp_path, "cm_free", writeback="free")
    off = _run_k(tmp_path, "cm_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"ライブ連成で呼数が k 依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "ride"), "ライブ ON なのに乗車が出ていない(連成が不発)"


def test_coupling_adds_zero_llm_calls(tmp_path):
    """連成フック(_taxi_live_dispatch → TaxiLive.request → bridge.reserve)は LLM を1本も呼ばない
    (呼数不変=MobiVerse 同思想)。1件の配車要求の前後で generate 呼数が不変であることを直接示す。"""
    sim = _sim(tmp_path, "zero", n=6, steps=1, **_MOCK)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    a = sim.agents[0]
    a.has_car, a.money = False, 1_000_000.0
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    sx, sy = sim.city.node_xy(nodes[0])
    dst = max(nodes, key=lambda n: math.hypot(*(p - q for p, q in
              zip(sim.city.node_xy(n), (sx, sy)))))
    a.node = nodes[0]
    a._ride_pending = {"mode": "taxi", "fare": 800.0, "from": a.node, "to": dst}
    before = sim.llm.calls
    scheduler._taxi_live_dispatch(sim, a, dst, step=1, sim_min=800)
    assert sim.llm.calls == before, "連成フックが LLM を呼んでいる(呼数不変の違反)"
    assert sim.taxi_live.stats()["requests"] == 1, "配車要求が記録されていない"


# --------------------------------------------------------------------- SUMO 実走(環境ガード)
@pytest.mark.skipif(not _sumo_available(),
                    reason="SUMO(sumo 実行ファイル+traci+car net)不在=実走テストは skip")
def test_sumo_bridge_determinism_same_seed(tmp_path):
    """[SUMO 実走] 同 seed 2 回で (wait_s, ride_s, delay_s, hold_steps) 列がバイト一致(go/no-go ②)。
    実 SUMO の taxi device + dispatch=traci をサブプロセスで回し、予約→pickup→dropoff の実秒が
    決定論に再現することを示す。座標は既定地図(=demo net と同一原点)の実ノードから与える。"""
    sim = _sim(tmp_path, "sumo_coords", n=2, steps=1, **{"transit_ride.taxi.prob": 1.0})
    origin = tuple(sim.city.meta["origin_latlon"])
    nodes = sim.dests or sorted(sim.city.graph.nodes)
    sx, sy = sim.city.node_xy(nodes[0])
    far = sorted(nodes, key=lambda n: -math.hypot(*(p - q for p, q in
                 zip(sim.city.node_xy(n), (sx, sy)))))
    trips = [(nodes[0], far[0]), (nodes[0], far[1]), (far[2], far[0])]
    lc = transit_live.live_cfg(load_config(
        ["transit_ride.live.enabled=true", "transit_ride.live.backend=traci",
         "transit_ride.live.n_taxis=6"]))

    def one_run():
        import sys
        from pathlib import Path
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from sumo_taxi_bridge import SumoTaxiBridge
        br = SumoTaxiBridge(lc, net_hint=_MAP_ORIGIN_NET, origin=origin)
        br.start()
        tl = transit_live.TaxiLive(lc, br)
        out = []
        try:
            for i, (a, b) in enumerate(trips):
                fx, fy = sim.city.node_xy(a)
                tx, ty = sim.city.node_xy(b)
                r = tl.request(agent_id=i, from_node=a, to_node=b,
                               from_xy=(fx, fy), to_xy=(tx, ty),
                               step=1, sim_min=600 + i * 10)
                out.append(r)
        finally:
            tl.close()
        return out

    r1 = one_run()
    r2 = one_run()
    assert r1 == r2, f"実 SUMO の配車が非決定論(同 seed でバイト不一致): {r1} != {r2}"
    assert any(r["matched"] for r in r1), "実 SUMO で 1 件も配車できていない"
    for r in r1:
        if r["matched"]:
            assert r["wait_s"] >= 0 and r["ride_s"] > 0 and r["delay_s"] >= 0
