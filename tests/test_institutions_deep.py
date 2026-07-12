"""制度深化(第9バッチ 2026-07-07)+ 改善修正 P2/P3 のテスト。

対象(調査: docs/research/rights-institutions-gap.md / sim-improvement-analysis.md):
- 審議・パブコメ段階(institution_routes.deliberation): 署名到達→審議入り→反対多数で否決/
  少数で可決経路へ(proposal_review イベント)。
- 供託金(institution_routes.vote.deposit): 払えなければ演説どまり、可決/得票1割で返還、未満は没収。
- 権利創設・宣言型(rules.allow_declare): declare 型 rule の制定と知覚行(norm_line)描画。
- P2 来街者の財布補充(economy.visitor_refresh): 帰宅から戻るたび allowance まで補充。
- P3 発話の定型化ガード(prompts.variety_hint): 状況文への注意書き注入。
すべて既定 OFF=純粋既定と L1 完全一致(ゴールデンは test_scenario が別途担保)。mock のみ。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _seed_proposal(sim, text="夜の騒音をなくす取り組み", n_supporters=10):
    """テスト用: 提案を直接登録し、署名を閾値以上に積む(id 昇順=決定論)。"""
    tools = sim.tools
    pid = tools._proposal_seq
    tools._proposal_seq += 1
    tools.proposals[pid] = {
        "id": pid, "author": sim.agents[0].id, "text": text,
        "supporters": set(a.id for a in sim.agents[:n_supporters]),
        "passed": False, "rule": None}
    return tools.proposals[pid]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。新イベント(proposal_review/deposit)は出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"institution_routes.deliberation.enabled": "false",
                  "institution_routes.vote.deposit": "0",
                  "rules.allow_declare": "false",
                  "economy.visitor_refresh": "false",
                  "prompts.variety_hint": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(制度深化 seam が no-op でない)"
    for k in ("proposal_review", "deposit"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 審議・パブコメ
def test_deliberation_reject_when_opposed(tmp_path):
    """署名到達→審議入り(start)→認知者の反対が多い→否決(reject)。成立しない。"""
    sim = _sim(tmp_path, "delib_rej", steps=1,
               **{"institution_routes.deliberation.enabled": "true",
                  "institution_routes.deliberation.days": "1",
                  "institution_routes.deliberation.reject_ratio": "0.5"})
    sim.run()
    pr = _seed_proposal(sim, n_supporters=10)
    sim.tools._proposal_support(sim, step=10, sim_min=100)      # day0: 審議入り
    rv = _kind(sim, "proposal_review")
    assert rv and rv[-1].payload["phase"] == "start", "審議入り(start)が出ていない"
    assert not pr["passed"]
    for a in sim.agents[10:16]:            # パブコメ窓で提案を知った反対者6人(> 10×0.5)。
        a.adopted.add(pr["text"])          # 立場表明: 決定論 no(grievance 低・opinion 中立に固定)
        a.states["grievance"] = 0.0
        a.opinion = 0.0
    sim.tools._proposal_support(sim, step=20, sim_min=200)      # day0: 反対を積む(まだ審議中)
    assert not pr["passed"] and not pr.get("decided")
    assert len(pr["review"]["opposed"]) == 6, "露出した反対者が opposed に積まれていない"
    sim.tools._proposal_support(sim, step=154, sim_min=1540)    # day1: 満了 → 否決
    rv = _kind(sim, "proposal_review")
    assert rv[-1].payload["phase"] == "reject" and rv[-1].payload["opposed"] >= 6, \
        f"反対多数なのに否決されていない: {rv[-1].payload}"
    assert pr.get("decided") and not pr["passed"], "否決後に成立してしまっている"
    assert not _kind(sim, "proposal_passed")


def test_deliberation_approve_when_unopposed(tmp_path):
    """審議で反対が集まらなければ approve → 従来の署名成立(proposal_passed)へ。"""
    sim = _sim(tmp_path, "delib_ok", steps=1,
               **{"institution_routes.deliberation.enabled": "true",
                  "institution_routes.deliberation.days": "1"})
    sim.run()
    pr = _seed_proposal(sim, n_supporters=10)
    sim.tools._proposal_support(sim, step=10, sim_min=100)      # day0: 審議入り
    sim.tools._proposal_support(sim, step=154, sim_min=1540)    # day1: 満了 → 可決経路
    rv = _kind(sim, "proposal_review")
    phases = [e.payload["phase"] for e in rv]
    assert phases == ["start", "approve"], f"審議の遷移が不正: {phases}"
    assert pr["passed"] and _kind(sim, "proposal_passed"), "approve 後に成立していない"


# --------------------------------------------------------------------- 供託金
def test_deposit_bars_and_refunds(tmp_path):
    """供託金: 払えなければ演説どまり(受理なし)。可決なら返還、得票不足なら没収。"""
    ov = {"institution_routes.vote.enabled": "true",
          "institution_routes.vote.deposit": "5000",
          "institution_routes.vote.refund_share": "0.1"}
    sim = _sim(tmp_path, "dep", steps=1, **ov)
    sim.run()
    poor, rich = sim.agents[1], sim.agents[2]
    poor.money = 1000.0
    n0 = len(sim.tools.proposals)
    sim.tools._propose(sim, poor, {"type": "propose", "text": "屋台をもっと増やす"},
                       step=5, sim_min=50)
    assert len(sim.tools.proposals) == n0, "供託金が払えないのに提案が受理された"
    dep = _kind(sim, "deposit")
    assert dep and dep[-1].payload["phase"] == "insufficient"

    rich.money = 20000.0
    sim.tools._propose(sim, rich, {"type": "propose", "text": "広場に花を植える"},
                       step=6, sim_min=60)
    assert rich.money == 15000.0, "供託金が拠出されていない"
    pr = sim.tools.proposals[max(sim.tools.proposals)]
    assert pr["deposit"] == 5000.0
    assert _kind(sim, "deposit")[-1].payload["phase"] == "paid"

    # 可決 → 返還(全員 yes: grievance を閾値超に)
    for a in sim.agents:
        a.states["grievance"] = 0.9
    pr["supporters"] = set(a.id for a in sim.agents[:10])
    sim.tools._resolve_vote(sim, pr, step=10, sim_min=100,
                            vcfg=__import__("society.tools", fromlist=["routes_of"])
                            .routes_of(sim)["vote"])
    assert _kind(sim, "deposit")[-1].payload["phase"] == "refund"
    assert rich.money == 20000.0, "可決なのに供託金が返還されていない"

    # 没収: 全員 no(grievance 低・opinion 中立)で新提案を開票
    for a in sim.agents:
        a.states["grievance"] = 0.0
        a.opinion = 0.0
    rich.money = 20000.0
    sim.tools._propose(sim, rich, {"type": "propose", "text": "街灯を増やす"},
                       step=11, sim_min=110)
    pr2 = sim.tools.proposals[max(sim.tools.proposals)]
    pr2["supporters"] = set(a.id for a in sim.agents[:10])
    from society.tools import routes_of
    sim.tools._resolve_vote(sim, pr2, step=12, sim_min=120, vcfg=routes_of(sim)["vote"])
    assert _kind(sim, "deposit")[-1].payload["phase"] == "forfeit", "得票不足なのに没収されない"
    assert rich.money == 15000.0, "没収なのに残高が戻っている"


# --------------------------------------------------------------------- 権利創設・宣言型
def test_declare_rule_gated_and_rendered(tmp_path):
    """declare 型: allow_declare=false は降格(不変)。true なら制定され知覚行に「宣言」が載る。"""
    from society import recursion as rec_mod
    off = _sim(tmp_path, "dec_off", steps=1)
    off.run()
    assert off.rulebook.enact({"type": "declare", "title": "多様性を認め合う"},
                              name="宣言テスト", proposer=0, step=0, day=0) is None, \
        "allow_declare=false なのに declare が制定された"

    on = _sim(tmp_path, "dec_on", steps=1, **{"rules.allow_declare": "true",
                                              "recursion.enabled": "true"})
    on.run()
    rec = on.rulebook.enact({"type": "declare", "title": "多様性を認め合う"},
                            name="パートナーシップ宣言", proposer=0, step=0, day=0)
    assert rec is not None and rec["type"] == "declare"
    assert on.rulebook.public_rule(rec) == {"type": "declare", "title": "多様性を認め合う"}
    line = on.recursion.norm_line(on)
    assert line and "宣言: 多様性を認め合う" in line, f"知覚行に宣言が載っていない: {line}"
    # fee/curfew アクセサは declare を素通し(価格・行き先に作用しない)
    assert on.rulebook.fee_price("cafe", 500.0) == 500.0
    assert not on.rulebook.has_curfew() and not on.rulebook.has_prohibit()


# --------------------------------------------------------------------- P2 来街者の財布補充
def test_visitor_refresh_on_return(tmp_path):
    """visitor_refresh=true: 帰宅から戻った来街者の手持ちが allowance まで補充される(wage 記録)。"""
    sim = _sim(tmp_path, "refresh", steps=1, **{"economy.visitor_refresh": "true"})
    sim.run()
    a = sim.agents[0]
    gate = next(n for n in sim.city.gateways if n != sim.city.station_node)
    a.visitor, a.money = True, 500.0
    a.loc, a.sleeping = "outside", False
    a.return_at, a.return_gateway = 0, gate
    scheduler._phase_wake_and_returns(sim, step=10, sim_min=600)
    assert a.loc == "street" and a.money == sim.economy["allowance_visitor"], \
        "帰還したのに財布が補充されていない"
    w = [e for e in _kind(sim, "wage") if e.payload.get("source") == "home_refill"]
    assert w and w[-1].agent_id == a.id and w[-1].payload["amount"] == 19500.0

    off = _sim(tmp_path, "refresh_off", steps=1)
    off.run()
    b = off.agents[0]
    gate2 = next(n for n in off.city.gateways if n != off.city.station_node)
    b.visitor, b.money = True, 500.0
    b.loc, b.sleeping = "outside", False
    b.return_at, b.return_gateway = 0, gate2
    scheduler._phase_wake_and_returns(off, step=10, sim_min=600)
    assert b.money == 500.0, "OFF なのに財布が補充された"


# --------------------------------------------------------------------- P3 発話の定型化ガード
def test_variety_hint_injection(tmp_path):
    """variety_hint=true: social/reply/solo の状況文に注意書きが載る。false は一字も足さない。"""
    from society.cognition import deliberate
    sim = _sim(tmp_path, "hint", steps=1)
    sim.run()
    a = sim.agents[0]
    for surprise, reply_to in (("social", None), ("solo", None),
                               ("reply", ("太郎", "やあ"))):
        on = deliberate.build_prompt(a, place_name="路上", surprise=surprise,
                                     nearby_names=[], sim_min=600, step=3,
                                     reply_to=reply_to, variety_hint=True)
        off = deliberate.build_prompt(a, place_name="路上", surprise=surprise,
                                      nearby_names=[], sim_min=600, step=3,
                                      reply_to=reply_to)
        assert ("決まり文句" in on) or ("オウム返し" in on) or ("報告文" in on), \
            f"{surprise}: variety_hint の注意書きが載っていない"
        assert "決まり文句" not in off and "オウム返し" not in off, \
            f"{surprise}: OFF なのに注意書きが載っている"
