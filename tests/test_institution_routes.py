"""制度改変の3ルート(現実ギャップ Wave G3 2026-07-07)= 職域/民主/執行 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): labor_action/vote_cast/vote_result/enforcement が 0 件・イベント列は純粋既定と
  L1 完全一致(ゴールデン golden_baseline_l1.json を守る)。
- 労働争議(職域): 被雇用者の propose が labor_action として記録され、賛同が同じ org の同僚に偏る。
- 投票(民主): 露出者が閾値到達で決定論投票し、過半数で可決/否決(vote_cast/vote_result)。
- 執行: prohibit ルール下で近傍の警察官が違反者を執行(罰金 + grievance)。
- 決定論: ON 同士2回で L1 完全一致。R1 呼数不変(FixedLLM で ON==OFF)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from collections import defaultdict

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.tools import _is_employee


def _sim(tmp_path, name, n=30, steps=1, *, routes=None, orgs=False,
         personas=False, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if orgs or personas:
        dot.append("agents.personas_file=data/personas_80.json")
    if orgs:
        dot.append("organizations.enabled=true")
    for r in (routes or []):
        dot.append(f"institution_routes.{r}.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


_G3_KINDS = ("labor_action", "vote_cast", "vote_result", "enforcement")


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。G3 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"institution_routes.labor.enabled": "false",
                  "institution_routes.vote.enabled": "false",
                  "institution_routes.enforcement.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G3 seam が no-op でない)"
    for k in _G3_KINDS:
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 労働争議(職域)
def test_labor_action_and_colleague_bias(tmp_path):
    """被雇用者の propose が labor_action として記録され、同じ org の同僚(未採用)が職場露出で賛同する。"""
    sim = _sim(tmp_path, "labor", n=40, orgs=True, routes=["labor"])
    sim.run()                                          # steps=1: 組織台帳を配属
    byorg = defaultdict(list)
    for ag in sim.agents:
        if _is_employee(ag):
            byorg[ag.org_id].append(ag)
    pair = next((v for v in byorg.values() if len(v) >= 2), None)
    assert pair, "同僚ペアが見つからない(personas_80 / org_assignments_80 を確認)"
    a, b = pair[0], pair[1]
    c = next(ag for ag in sim.agents
             if getattr(ag, "org_id", None) != a.org_id)   # 別 org / 未配属
    text = "残業代を上げてほしい"
    for ag in (a, b, c):
        ag.adopted.discard(text)
    sim.tools._propose(sim, a, {"type": "propose", "text": text},
                       step=10, sim_min=100)
    la = _kind(sim, "labor_action")
    assert la and la[-1].payload["org"] == a.org_id \
        and la[-1].payload["initiator"] == a.id, "labor_action が起点者/org で記録されていない"
    sim.tools._proposal_support(sim, step=11, sim_min=110)
    pr = next(p for p in sim.tools.proposals.values() if p["text"] == text)
    assert b.id in pr["supporters"], "同僚が職場露出で賛同していない(職域の偏りが効いていない)"
    assert c.id not in pr["supporters"], "非同僚(未採用)が賛同している(偏りが広すぎる)"


# --------------------------------------------------------------------- 投票(民主)
def test_vote_cast_result_pass_and_reject(tmp_path):
    """露出者が閾値到達で決定論投票し、過半数 yes で可決 / 過半数 no で否決(vote_cast/vote_result)。"""
    sim = _sim(tmp_path, "vote", n=10, routes=["vote"],
               **{"institution_routes.vote.grievance_threshold": 0.3})
    sim.run()

    # (a) 可決: author + 3 露出者、うち 3 人が不満高(yes)・1 人が不満0で opinion 中立(no)
    text = "夜間の騒音を規制しよう"
    a = sim.agents[0]
    sim.tools._propose(sim, a, {"type": "propose", "text": text},
                       step=5, sim_min=60)
    for i in (1, 2, 3):
        sim.agents[i].adopted.add(text)
    for i in (0, 1, 2):
        sim.agents[i].states["grievance"] = 0.9
        sim.agents[i].opinion = 0.0
    sim.agents[3].states["grievance"] = 0.0
    sim.agents[3].opinion = 0.0
    sim.tools._proposal_support(sim, step=6, sim_min=70)
    pid = next(p["id"] for p in sim.tools.proposals.values() if p["text"] == text)
    res = [e for e in _kind(sim, "vote_result") if e.payload["proposal_id"] == pid]
    casts = [e for e in _kind(sim, "vote_cast") if e.payload["proposal_id"] == pid]
    assert res and res[-1].payload["yes"] == 3 and res[-1].payload["no"] == 1 \
        and res[-1].payload["passed"] is True, "過半数 yes で可決していない"
    assert len(casts) == 4, "露出者全員の vote_cast が出ていない"
    assert _kind(sim, "proposal_passed"), "可決なのに proposal_passed が出ていない"

    # (b) 否決: 全員 不満0・opinion 中立 → yes 0 で否決
    text2 = "花火大会をもっと増やそう"
    d = sim.agents[4]
    sim.tools._propose(sim, d, {"type": "propose", "text": text2},
                       step=7, sim_min=80)
    for i in (4, 5, 6, 7):
        sim.agents[i].adopted.add(text2)
        sim.agents[i].states["grievance"] = 0.0
        sim.agents[i].opinion = 0.0
    sim.tools._proposal_support(sim, step=8, sim_min=90)
    pid2 = next(p["id"] for p in sim.tools.proposals.values() if p["text"] == text2)
    res2 = [e for e in _kind(sim, "vote_result") if e.payload["proposal_id"] == pid2]
    assert res2 and res2[-1].payload["passed"] is False, "全員 no なのに否決していない"


# --------------------------------------------------------------------- 執行
def test_enforcement_penalizes_violator(tmp_path):
    """prohibit ルール下で、同ノードの警察官が禁止カテゴリの場所に居る違反者に罰金 + grievance。"""
    sim = _sim(tmp_path, "enf", n=10, routes=["enforcement"])
    sim.run()
    rb = sim.rulebook
    cat = "nightlife"
    node = next((p["node"] for p in sim.city.poi_list if p["cat"] == cat), None)
    assert node, "nightlife POI が地図に無い(テスト前提の破綻)"
    rec = rb.enact({"type": "prohibit", "target_cat": cat}, name="夜間規制",
                   proposer=0, step=0, day=0)
    assert rec is not None, "prohibit ルールが制定されていない(validate 失敗)"

    officer, violator = sim.agents[0], sim.agents[1]
    officer.occupation = "警察官"
    violator.occupation = "会社員"                      # 公務員でない=執行対象
    for ag in (officer, violator):
        ag.loc = "street"
        ag.sleeping = False
        ag.building = None
        ag.node = node
        ag.x, ag.y = sim.city.node_xy(node)
    violator.money = 5000.0
    g0 = violator.states.get("grievance", 0.0)
    scheduler._phase_enforcement(sim, step=20, sim_min=22 * 60)   # 22時=終日窓内
    enf = _kind(sim, "enforcement")
    assert enf, "enforcement が出ていない(執行が起きていない)"
    p = enf[-1].payload
    assert p["officer"] == officer.id and p["target"] == violator.id
    assert p["rule_id"] == rec["id"] and p["penalty"] == 1000.0
    assert violator.money == 4000.0, "罰金が引かれていない"
    assert violator.states["grievance"] > g0, "執行で grievance が上がっていない"

    # 警察官不在なら執行なし(officer を退場させると no-op)
    n_before = len(_kind(sim, "enforcement"))
    officer.loc = "outside"
    scheduler._phase_enforcement(sim, step=21, sim_min=22 * 60)
    assert len(_kind(sim, "enforcement")) == n_before, "警察官不在でも執行が起きている"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """3ルート ON(+組織)同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, orgs=True,
             routes=["labor", "vote", "enforcement"])
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, orgs=True,
             routes=["labor", "vote", "enforcement"])
    b.run()
    assert _l1(a) == _l1(b), "3ルート ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, *, routes):
    sim = _sim(tmp_path, name, n=30, steps=144, orgs=True,
               routes=(["labor", "vote", "enforcement"] if routes else None))
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend なら 3ルート ON/OFF で generate 呼数が完全一致(R1: 非LLM機構=呼数不変)。"""
    on = _run_fixed(tmp_path, "cc_on", routes=True)
    off = _run_fixed(tmp_path, "cc_off", routes=False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
