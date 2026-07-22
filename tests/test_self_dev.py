"""T4 自助努力 affordance(第52バッチ)のテスト。

「自力で成長・改善できる」可能性を用意する層(成長を強制せず選べる状態=affordance)。既存のサービス
受給(第46バッチ services.charge_service)に、反復利用の累積 agent.self_dev(塾=skill / ジム=fitness)と、
経済成果への1本だけの間接経路(skill → 賃金乗数)を足す。R1 の鉄則を継承:
- OFF(既定 services.self_dev.enabled=false): 会計もイベントも発火も一切動かさない=ゴールデン L1 バイト一致。
- ON + wage_coef=0: 会計不変(賃金乗数 1.0)。ON + wage_coef>0: skill を持つ者の賃金だけが増える。
- 累積は逓減(飽和)形で、決定論(RNG ゼロ)。発火判定に累積状態を読ませない(k=free/off の LLM 呼数一致)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤48 step)。
"""
from __future__ import annotations

import json

from society import services as services_mod
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import aggregate as agg

# services ON のみ(self_dev は既定 OFF)/ services + self_dev ON(wage_coef=0.0 既定)。
_SVC = {"services.enabled": "true"}
_SD = {"services.enabled": "true", "services.self_dev.enabled": "true"}


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


def _acct(sim):
    """会計イベント(賃金・消費)だけを取り出す(自助努力は金の動きに触れないはず)。"""
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events if e.kind in ("wage", "spend")]


# ------------------------------------------------------ OFF 既定一致(ゴールデン層)
def test_self_dev_off_matches_pure_default(tmp_path):
    """self_dev 明示 OFF(services も既定 OFF)が純粋既定と L1 完全一致。self_dev も蓄積しない。"""
    pure = _sim(tmp_path, "pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "sd_off", steps=48, **{"services.self_dev.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "self_dev seam が純粋既定を汚している"
    for a in pure.agents:
        assert getattr(a, "self_dev", {}) == {}, "OFF なのに self_dev が蓄積している"


def test_self_dev_off_is_noop_over_services(tmp_path):
    """services ON + self_dev OFF が services ON のみと L1 完全一致(self_dev seam が no-op)。"""
    base = _sim(tmp_path, "svc_base", steps=48, **_SVC)
    base.run()
    off = _sim(tmp_path, "svc_sdoff", steps=48,
               **{**_SVC, "services.self_dev.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "self_dev OFF が services 層を動かしている"
    assert not any("dev_value" in e.payload for e in _kind(off, "service_use"))


def test_off_helpers_are_noop(tmp_path):
    """self_dev OFF では賃金乗数=1.0・accrue は None(state を触らない)。"""
    sim = _sim(tmp_path, "noop", **_SVC)             # services ON・self_dev 既定 OFF
    a = sim.agents[0]
    assert services_mod.self_dev_wage_mult(sim, a) == 1.0
    assert services_mod.accrue_self_dev(sim, a, "lesson", 0, 0) is None
    assert a.self_dev == {}


# ------------------------------------------------------ ON + coef=0 で会計不変
def test_on_coef0_accounting_invariant(tmp_path):
    """self_dev ON + wage_coef=0 は会計(wage/spend の全列)が OFF と完全一致(乗数 1.0)。"""
    off = _sim(tmp_path, "c0_off", steps=48,
               **{**_SVC, "services.self_dev.enabled": "false"})
    off.run()
    on = _sim(tmp_path, "c0_on", steps=48,
              **{**_SD, "services.self_dev.remember_threshold": "999"})  # 記憶発火を封じ会計だけを見る
    on.run()
    assert _acct(off) == _acct(on), "self_dev ON(coef=0)が会計イベントを動かした"


# ------------------------------------------------------ ON 決定論(同 seed 2回で一致)
def test_on_deterministic(tmp_path):
    """self_dev ON(coef>0・decay>0)同士 2 回で L1 完全一致(決定論・RNG ゼロ)。"""
    ov = {**_SD, "services.self_dev.wage_coef": "0.1",
          "services.self_dev.decay": "0.01", "prompts.interstitial.enabled": "true"}
    a = _sim(tmp_path, "det_a", steps=30, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", steps=30, **ov)
    b.run()
    assert _l1(a) == _l1(b), "self_dev ON の決定論が崩れている"


# ------------------------------------------------------ 累積が逓減形で増える
def test_accrual_is_saturating(tmp_path):
    """反復受給で dev_axis(塾→skill)が単調増・増分は逓減(飽和形=無限成長を防ぐ)。"""
    sim = _sim(tmp_path, "accr", **_SD)
    a = sim.agents[0]
    a.self_dev = {}
    vals = []
    for i in range(6):
        axis, v = services_mod.accrue_self_dev(sim, a, "lesson", i, i * 10)
        assert axis == "skill"
        vals.append(v)
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), "累積が増えていない"
    deltas = [vals[0]] + [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert all(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1)), \
        "増分が逓減していない(飽和形でない)"
    # 別サービスは別軸(gym→fitness)。skill は不変=軸ごとに独立に累積する。
    axis2, _ = services_mod.accrue_self_dev(sim, a, "gym", 99, 990)
    assert axis2 == "fitness" and a.self_dev["fitness"] > 0.0
    assert a.self_dev["skill"] == vals[-1]


def test_accrual_appended_to_service_use_payload(tmp_path):
    """受給の service_use payload に累積後の軸値が追記される(既存キー不変・追加のみ)。"""
    sim = _sim(tmp_path, "payload", **_SD)
    node = sim.city.pois_by_cat("service")[0]["node"]
    a = sim.agents[0]
    a.node = node
    a.x, a.y = sim.city.node_xy(node)
    a.money = 100000.0
    a.self_dev = {}
    a._service_pending = ("lesson", node)            # 塾(skill 軸)を強制受給
    scheduler._charge_service(sim, a, 5, 700)
    uses = _kind(sim, "service_use")
    assert len(uses) == 1
    p = uses[0].payload
    assert p["service"] == "lesson" and "node" in p and "cost" in p and "poi" in p
    assert p["dev_axis"] == "skill" and p["dev_value"] > 0.0
    assert a.self_dev["skill"] == p["dev_value"]


# ------------------------------------------------------ 賃金乗数が効く(coef>0 で賃金増)
def test_wage_multiplier_boosts_pay(tmp_path):
    """wage_coef>0 のとき skill を持つ者の支給が (1+coef×skill) 倍になり、持たぬ者は不変。"""
    sim = _sim(tmp_path, "wage", **{**_SD, "services.self_dev.wage_coef": "0.1"})
    assert abs(sim.servicescfg["self_dev"]["wage_coef"] - 0.1) < 1e-12
    a = sim.agents[0]
    a.self_dev = {"skill": 1.0}
    assert abs(services_mod.self_dev_wage_mult(sim, a) - 1.1) < 1e-9
    a.money = 0.0
    scheduler._pay_wage(sim, a, 10000.0, 5, 700, source="gig")
    assert abs(a.money - 11000.0) < 1e-6, "賃金乗数が支給に効いていない"
    # 累積のない個体は乗数 1.0(自力で伸ばした者だけが得をする)
    b = sim.agents[1]
    b.self_dev = {}
    b.money = 0.0
    scheduler._pay_wage(sim, b, 10000.0, 5, 700, source="gig")
    assert abs(b.money - 10000.0) < 1e-6, "skill=0 なのに賃金が変わった"


def test_wage_axis_only_skill(tmp_path):
    """賃金へ結ぶのは skill のみ(fitness は累積・観測するが経済経路を持たない=1本だけの経路)。"""
    sim = _sim(tmp_path, "axis", **{**_SD, "services.self_dev.wage_coef": "0.1"})
    a = sim.agents[0]
    a.self_dev = {"fitness": 1.0}                     # skill は無し
    assert services_mod.self_dev_wage_mult(sim, a) == 1.0


# ------------------------------------------------------ 日次自然減衰(decay)
def test_daily_decay_erodes(tmp_path):
    """decay>0 は日次で乗算減衰(維持しないと薄れる)。decay=0 は完全 no-op。"""
    sim = _sim(tmp_path, "decay", **{**_SD, "services.self_dev.decay": "0.1"})
    a = sim.agents[0]
    a.self_dev = {"skill": 1.0}
    services_mod.self_dev_daily(sim, 0, 0)
    assert abs(a.self_dev["skill"] - 0.9) < 1e-9
    sim.servicescfg["self_dev"]["decay"] = 0.0
    services_mod.self_dev_daily(sim, 0, 0)
    assert a.self_dev["skill"] == 0.9, "decay=0 なのに減衰した"


# ------------------------------------------------------ 発火判定に不使用(k 不変性)
class _FixedLLM:
    """内容非依存の固定応答 backend。呼数だけを数える(test_services と同型)。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, steps=24,
               **{**ov, "prompts.interstitial.enabled": "true"})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_k_invariant(tmp_path):
    """self_dev ON(coef>0)で k=free と k=off の LLM 呼数が一致=累積状態を発火判定に食わせていない。"""
    ov = {**_SD, "services.self_dev.wage_coef": "0.1"}
    free = _run_fixed(tmp_path, "k_free", **{**ov, "k.writeback": "free"})
    off = _run_fixed(tmp_path, "k_off", **{**ov, "k.writeback": "off"})
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"k∈{{free,off}} で呼数が変化(累積状態の漏洩): free={free.llm.calls} off={off.llm.calls}"


# ------------------------------------------------------ L2 採用率系の列(OFF=None)
def test_l2_columns_present_only_when_on(tmp_path):
    """L2: 自助サービス利用者数・平均累積値は self_dev OFF で None(列なし)、ON で値を出す。"""
    off = _sim(tmp_path, "l2off", **_SVC)             # self_dev 既定 OFF
    assert agg.AGGREGATORS["n_self_dev_users"](off) is None
    assert agg.AGGREGATORS["avg_self_dev"](off) is None
    on = _sim(tmp_path, "l2on", **_SD)
    on.agents[0].self_dev = {"skill": 0.3}
    on.agents[1].self_dev = {"fitness": 0.5}
    assert agg.AGGREGATORS["n_self_dev_users"](on) == 2
    assert abs(agg.AGGREGATORS["avg_self_dev"](on) - 0.4) < 1e-9


# ------------------------------------------------------ resume 一致(checkpoint)
def test_self_dev_survives_checkpoint(tmp_path):
    """agent.self_dev が checkpoint save→load で保存される(Agent ごと pickle 経路)。"""
    ov = {**_SD, "services.self_dev.wage_coef": "0.1"}
    sim = _sim(tmp_path, "ckA", steps=10, **ov)
    a = sim.agents[0]
    a.self_dev = {"skill": 0.42, "fitness": 0.17}
    path = tmp_path / "ck.pkl.gz"
    checkpoint.save(sim, 5, path)
    sim2 = _sim(tmp_path, "ckB", steps=10, **ov)      # 同一 config(name は hash 除外)→ load 可
    assert checkpoint.load(sim2, path) == 5
    a2 = sim2.agent_by_id[a.id]
    assert a2.self_dev == {"skill": 0.42, "fitness": 0.17}, \
        "resume で self_dev が保存されていない"
