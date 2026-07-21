"""サービスの実体化 スライス③(理美容/クリニック/塾/ジム/クリーニング等。既定 OFF)のテスト。

正典: docs/research/economy-goods-services.md §7 ③。R1 の鉄則を継承:
- OFF(既定): L1 が純粋既定と完全一致(バイト一致)・service_use 0 件・来店バイアスも効果も生えない・
  "service" stream も引かない。
- ON: (a)service_dest が近傍の service POI へ寄せ来店予定を退避。(b)到着で課金(spend cat=service)+ 効果
      (factors on_service=気分/活力/学習)+ 記憶 + service_use。(c)効果は band で対象 state を選ぶ(mood/vitality=
      grievance−、learning=efficacy+)。(d)決定論(同 seed 2回で L1 一致)。(e)k∈{free,off}で LLM 呼数一致
      (来店判定に k を食わせない=k 不変性)。(f)schema 登録。(g)streaming 集計 service_usage。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤48 step)。追加 LLM 呼ゼロ・乱数は "service" stream のみ。
"""
from __future__ import annotations

import json

from society import services as services_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer import stream as st
from society.observer.schema import EVENT_KINDS

_ON = {"services.enabled": "true"}
_OFF = {"services.enabled": "false"}
_ISL = {"prompts.interstitial.enabled": "true"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _service_node(sim):
    return sim.city.pois_by_cat("service")[0]["node"]


# ------------------------------------------------------ (f) schema 登録
def test_schema_registered():
    assert "service_use" in EVENT_KINDS, "service_use が schema 未登録"


# ------------------------------------------------------ OFF: 純粋既定と L1 完全一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。OFF では service_use も来店予定も生えない。"""
    pure = _sim(tmp_path, "pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=48, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(services seam が no-op でない)"
    assert not _kind(off, "service_use"), "OFF で service_use が出ている"
    for a in pure.agents:
        assert getattr(a, "_service_pending", None) is None, "OFF なのに来店予定が生えている"


def test_service_dest_off_returns_none(tmp_path):
    """OFF では service_dest が即 None を返す(来店バイアスなし・stream も引かない)。"""
    sim = _sim(tmp_path, "destoff", **_OFF)
    a = sim.agents[0]
    assert services_mod.service_dest(a, sim, 5, 700) is None


# ------------------------------------------------------ (a) 来店バイアス(service_dest)
def test_service_dest_biases_to_service_poi(tmp_path):
    """ON で近傍に service POI があれば service_dest がそのノードを返し来店予定を退避する(利用率を高くして強制ヒット)。"""
    sim = _sim(tmp_path, "dest", **{**_ON})
    node = _service_node(sim)
    x, y = sim.city.node_xy(node)
    a = sim.agents[0]
    a.node = sim.city.pois_by_cat("food")[0]["node"]     # 現在地は別ノード(同ノード除外を回避)
    a.x, a.y = x, y                                       # 位置は service POI の近傍(半径内)
    a.money = 100000.0
    a.age = 30
    for svc in sim.servicescfg["services"].values():     # 利用率を上げて必ずヒットさせる(決定論の抽選確認)
        svc["daily_rate"] = 1000.0
    m = 12 * 60                                           # 営業時間帯(9-20 時)
    dest = services_mod.service_dest(a, sim, 5, m)
    assert dest is not None, "近傍に service POI があるのに来店へ寄っていない"
    assert dest != a.node, "現在地と同じノードを返している(来店にならない)"
    key, pn = a._service_pending
    assert pn == dest and key in sim.servicescfg["services"]
    svc = sim.servicescfg["services"][key]                # dest には選ばれたサービスの POI がある
    assert services_mod._poi_name(sim, dest, svc) is not None, "来店先に該当サービス POI が無い"


# ------------------------------------------------------ (b)(c) 受給=課金+効果+記憶+service_use
def test_charge_service_charges_and_effects(tmp_path):
    """到着で課金(spend cat=service)+ 効果(mood→grievance−)+ service_use を出し会計は家計から引かれる。"""
    sim = _sim(tmp_path, "charge", **_ON)
    node = _service_node(sim)
    a = sim.agents[0]
    a.node = node
    a.x, a.y = sim.city.node_xy(node)
    a.money = 100000.0
    a._service_pending = ("service", node)               # 汎用対人サービス(mood・磁 0.015)
    g0 = a.states.get("grievance", 0.0)
    m0 = a.money
    scheduler._charge_service(sim, a, 5, 700)
    uses = _kind(sim, "service_use")
    spends = _kind(sim, "spend")
    assert len(uses) == 1 and uses[0].payload["service"] == "service"
    assert uses[0].payload["node"] == node and "cost" in uses[0].payload
    assert len(spends) == 1 and spends[0].payload["cat"] == "service"
    price = sim.servicescfg["services"]["service"]["price"]
    assert abs(m0 - a.money - price) < 1e-6, "サービス代が家計から引かれていない(会計不整合)"
    assert a.states.get("grievance", 0.0) < g0, "mood 効果で grievance が下がっていない"
    assert getattr(a, "_service_pending", None) is None, "受給後に来店予定がクリアされていない"


def test_charge_service_wrong_node_noop(tmp_path):
    """予定ノードと現在地が違えば受給しない(寄り道/中断で流れた場合の安全)。"""
    sim = _sim(tmp_path, "wrong", **_ON)
    node = _service_node(sim)
    a = sim.agents[0]
    a.node = sim.city.pois_by_cat("food")[0]["node"]     # 予定と違うノードに居る
    a.x, a.y = sim.city.node_xy(a.node)
    a.money = 100000.0
    a._service_pending = ("service", node)
    scheduler._charge_service(sim, a, 5, 700)
    assert not _kind(sim, "service_use") and not _kind(sim, "spend")
    assert getattr(a, "_service_pending", None) is None


def test_learning_effect_raises_efficacy(tmp_path):
    """learning 帯(塾)の効果は efficacy を上げる(band で対象 state が変わる)。"""
    sim = _sim(tmp_path, "learn", **_ON)
    a = sim.agents[0]
    e0 = a.states.get("efficacy", 0.0)
    services_mod.apply_effect(sim, a, "lesson", 5, 700)
    assert a.states.get("efficacy", 0.0) > e0, "learning 効果で efficacy が上がっていない"


# ------------------------------------------------------ (d) 決定論(同 seed 2回で完全一致)
def test_on_deterministic(tmp_path):
    """services ON 同士 2 回で L1 完全一致(決定論・"service" stream のみ・mock)。"""
    a = _sim(tmp_path, "det_a", steps=30, **{**_ON, **_ISL})
    a.run()
    b = _sim(tmp_path, "det_b", steps=30, **{**_ON, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# ------------------------------------------------------ (e) k 不変性(発火判断に k を食わせない)
class _FixedLLM:
    """内容非依存の固定応答 backend。呼数だけを数える。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, steps=24, **{**ov, "prompts.interstitial.enabled": "true"})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_k_invariant(tmp_path):
    """services ON で k=free と k=off の LLM 呼数が一致(来店=co-location 変化だが判定に k を食わせない)。"""
    free = _run_fixed(tmp_path, "k_free", **{**_ON, "k.writeback": "free"})
    off = _run_fixed(tmp_path, "k_off", **{**_ON, "k.writeback": "off"})
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"k∈{{free,off}} で呼数が変化(k 不変性 違反): free={free.llm.calls} off={off.llm.calls}"


# ------------------------------------------------------ (g) streaming 集計
def test_service_usage_aggregation(tmp_path):
    """streaming 集計 service_usage が件数・支出・サービス別内訳を有界1パスで出す。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _ev(step, aid, kind, **payload):
        return {"step": step, "sim_min": step * 10, "agent_id": aid,
                "kind": kind, "payload": payload}

    ev = [
        _ev(0, 1, "service_use", node="n1", service="grooming", cost=4000),
        _ev(0, 2, "service_use", node="n2", service="gym", cost=2000),
        _ev(1, 3, "service_use", node="n3", service="grooming", cost=4000),
        _ev(1, 4, "spend", cat="service", amount=2000),   # spend は service_usage の対象外
    ]
    rd = tmp_path / "agg"
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": [e["step"] for e in ev], "sim_min": [e["sim_min"] for e in ev],
        "agent_id": [e["agent_id"] for e in ev], "kind": [e["kind"] for e in ev],
        "x": [0.0] * len(ev), "y": [0.0] * len(ev),
        "payload": [json.dumps(e["payload"], ensure_ascii=False, sort_keys=True) for e in ev],
    }
    pq.write_table(pa.table(cols), rd / "l1_events.parquet")
    (rd / "summary.json").write_text(json.dumps({"n_steps": 1}), encoding="utf-8")
    out = st.service_usage(str(rd))
    assert out["n_use"] == 3 and out["cost_total"] == 10000.0
    assert out["by_service"]["grooming"] == {"n": 2, "cost": 8000.0}
    assert out["by_service"]["gym"] == {"n": 1, "cost": 2000.0}
