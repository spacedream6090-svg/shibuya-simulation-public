"""行政・税・公務員(バッチB 2026-07-06)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): tax/civic_service/public_budget が 0 件・既存イベント列に無影響(純粋既定と完全一致)。
- ON: 源泉徴収の整合(手取り+税=名目)/ 消費税の内訳(価格不変)/ 公務員給与が該当予算から歳出 /
  困窮→給付 / public_budget の会計整合(Σ歳入−Σ歳出=残高変化)/ 同一設定2回で決定論。
- 検証は mock のみ(実LLM 禁止)。全ラン非LLM。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.government import Government, build_government_cfg


def _sim(tmp_path, name: str, n: int = 12, steps: int = 1,
         gov: bool = True, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if gov:
        dot.append("government.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    return Simulation(cfg, out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と government キー無しの純粋既定が L1 完全一致(既存イベント列に無影響)。"""
    pure = _sim(tmp_path, "pure", n=15, steps=144, gov=False)
    pure.run()
    off = _sim(tmp_path, "expl_off", n=15, steps=144, gov=False,
               **{"government.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(行政 seam が no-op でない)"
    for k in ("tax", "civic_service", "public_budget"):
        assert not [e for e in pure.logger.events if e.kind == k], f"{k} が OFF で出ている"
    # 経済(既存)は動いている=行政 OFF でも既存イベントは通常どおり
    assert any(e.kind in ("wage", "spend") for e in pure.logger.events)


# --------------------------------------------------------------------- 源泉徴収
def test_wage_withholding_integrity(tmp_path):
    """賃金支給で所得税(→nation)+住民税(→ward6:metro4)を源泉徴収。手取り+税=名目。"""
    sim = _sim(tmp_path, "wh")
    a = sim.agents[0]
    bld = sim.city.buildings[0]
    a.building = bld["id"]
    a.work_building = bld["id"]
    a.work_start_min, a.work_end_min = 480, 1020
    a.activity = "working"
    a.part_time = None
    a.wage = 12000.0
    a.money = 5000.0
    scheduler._apply(sim, a, {"type": "exit_building"}, step=60, sim_min=1030)

    wages = [e for e in sim.logger.events if e.kind == "wage" and e.agent_id == a.id]
    assert wages, "wage イベントが出ていない"
    w = wages[-1].payload
    assert w["gross"] == 12000.0
    assert abs(w["amount"] + w["tax"] - w["gross"]) < 0.5, "手取り+税=名目 が崩れている"
    assert abs(a.money - (5000.0 + w["amount"])) < 0.5, "残高が手取りぶん増えていない"

    taxes = [e for e in sim.logger.events if e.kind == "tax" and e.agent_id == a.id]
    tos = {(e.payload["tax"], e.payload["to"]) for e in taxes}
    assert ("income", "nation") in tos
    assert ("resident", "ward") in tos and ("resident", "metro") in tos
    assert abs(sum(e.payload["amount"] for e in taxes) - w["tax"]) < 0.5
    # 住民税は区6:都4
    rw = next(e.payload["amount"] for e in taxes
              if e.payload["tax"] == "resident" and e.payload["to"] == "ward")
    rm = next(e.payload["amount"] for e in taxes
              if e.payload["tax"] == "resident" and e.payload["to"] == "metro")
    assert abs(rw / rm - 1.5) < 0.05, "住民税の区:都 が 6:4 でない"


# --------------------------------------------------------------------- 消費税
def test_consumption_tax_price_unchanged(tmp_path):
    """消費時、価格は名目不変(内税)。消費税を国:地方=78:22 で内訳計上。"""
    sim = _sim(tmp_path, "ct")
    a = sim.agents[0]
    food = sim.city.pois_by_cat("food")[0]
    a.node = food["node"]
    a.money = 10000.0
    scheduler._charge_meal(sim, a, step=5, sim_min=700)

    price = sim.economy["prices"]["food"]
    spends = [e for e in sim.logger.events if e.kind == "spend" and e.agent_id == a.id]
    assert spends and spends[-1].payload["amount"] == price, "価格の名目が変わっている"
    assert a.money == 10000.0 - price, "税で追加減少している(内税でない)"

    ct = [e for e in sim.logger.events
          if e.kind == "tax" and e.payload["tax"] == "consumption"]
    total = sum(e.payload["amount"] for e in ct)
    assert abs(total - price * 0.08 / 1.08) < 0.5, "food は軽減8%の内税でない"
    nat = sum(e.payload["amount"] for e in ct if e.payload["to"] == "nation")
    loc = sum(e.payload["amount"] for e in ct if e.payload["to"] == "metro")
    assert nat > 0 and loc > 0
    assert abs(nat / total - 0.78) < 0.02, "国:地方=78:22 でない"


def test_consumption_standard_rate_for_nonfood(tmp_path):
    """food 以外(shop 等)は標準 10% の内税。"""
    sim = _sim(tmp_path, "ct10")
    a = sim.agents[0]
    a.money = 100000.0
    scheduler._spend(sim, a, 2200.0, "shop", step=5, sim_min=700)
    ct = [e for e in sim.logger.events
          if e.kind == "tax" and e.payload["tax"] == "consumption"]
    assert abs(sum(e.payload["amount"] for e in ct) - 2200.0 * 0.10 / 1.10) < 0.5


# --------------------------------------------------------------------- 公務員給与
def test_civil_servant_pay_from_correct_budget(tmp_path):
    """区職員=ward予算 / 警察官=metro予算 から歳出。源泉徴収も掛かる。"""
    sim = _sim(tmp_path, "pay")
    gov = scheduler._gov(sim)
    a, b = sim.agents[0], sim.agents[1]
    a.occupation = "区職員"
    b.occupation = "警察官"
    ward_wage = sim.economy["wages"]["区職員"]
    police_wage = sim.economy["wages"]["警察官"]
    scheduler._gov_payroll(sim, gov, step=0, sim_min=0)

    assert abs(gov.day_exp["ward"] - ward_wage) < 0.5, "区職員給与が ward の歳出でない"
    assert abs(gov.day_exp["metro"] - police_wage) < 0.5, "警察官給与が metro の歳出でない"
    civil = [e for e in sim.logger.events
             if e.kind == "wage" and e.payload.get("source") == "civil"]
    assert {e.agent_id for e in civil} == {a.id, b.id}
    # 支給に伴う住民税は該当予算へ戻る(歳入)=循環
    assert gov.day_rev["ward"] > 0 and gov.day_rev["nation"] > 0


# --------------------------------------------------------------------- 給付
def test_benefit_to_needy_from_ward(tmp_path):
    """所持金が閾値未満の居住者へ ward から給付(civic_service)。潤沢者には出さない。"""
    sim = _sim(tmp_path, "ben")
    gov = scheduler._gov(sim)
    a, b = sim.agents[0], sim.agents[1]
    a.visitor, a.money, a.account = False, 100.0, 0.0
    b.visitor, b.money, b.account = False, 1_000_000.0, 0.0
    ward0 = gov.balance["ward"]
    scheduler._gov_benefits(sim, gov, step=0, sim_min=0)

    amt = gov.cfg["benefit_amount"]
    assert a.money == 100.0 + amt, "困窮者に給付されていない"
    assert b.money == 1_000_000.0, "潤沢者に給付されている"
    cs = [e for e in sim.logger.events if e.kind == "civic_service"]
    assert any(e.agent_id == a.id for e in cs)
    assert not any(e.agent_id == b.id for e in cs)
    assert all(e.payload["level"] == "ward" for e in cs)
    assert abs(gov.balance["ward"] - (ward0 - amt * len(cs))) < 0.5, "ward の歳出と残高が不整合"


def test_visitor_not_eligible_for_benefit(tmp_path):
    """来街者(街の外に生活基盤)は給付対象外。"""
    sim = _sim(tmp_path, "benv")
    gov = scheduler._gov(sim)
    a = sim.agents[0]
    a.visitor, a.money, a.account = True, 0.0, 0.0
    scheduler._gov_benefits(sim, gov, step=0, sim_min=0)
    assert a.money == 0.0
    assert not [e for e in sim.logger.events
                if e.kind == "civic_service" and e.agent_id == a.id]


# --------------------------------------------------------------------- 会計整合
def test_budget_accounting_invariant_unit():
    """Government の会計: level ごとに Σ歳入−Σ歳出 = 最後の残高 − 最初の残高。"""
    gov = Government(build_government_cfg({"enabled": True}))
    logged = list(gov.daily(0))                 # 初回=残高アンカー(歳入歳出0)
    gov.collect("ward", 1000.0)
    gov.expense("ward", 300.0)
    gov.collect("metro", 500.0)
    gov.collect("nation", 800.0)
    gov.expense("nation", 200.0)
    logged += gov.daily(1)                       # day0 を締める
    gov.collect("ward", 50.0)
    gov.expense("metro", 70.0)
    logged += gov.daily(2)                       # day1 を締める
    for lv in gov.LEVELS:
        recs = [r for r in logged if r["level"] == lv]
        srev = sum(r["revenue"] for r in recs)
        sexp = sum(r["expense"] for r in recs)
        assert abs((srev - sexp) - (recs[-1]["balance"] - recs[0]["balance"])) < 1e-6


def test_run_on_produces_events_and_accounts(tmp_path):
    """ラン(日境界跨ぎ)で public_budget/tax が出て、会計整合が成り立つ。世界レベル id=-1。"""
    sim = _sim(tmp_path, "onrun", n=12, steps=115)   # step102 で day0→1 を跨ぐ
    sim.run()
    pb = [e for e in sim.logger.events if e.kind == "public_budget"]
    assert pb, "public_budget が出ていない"
    assert all(e.agent_id == -1 for e in pb), "public_budget が世界レベル(id=-1)でない"
    assert any(e.kind == "tax" for e in sim.logger.events), "tax が出ていない"
    for lv in ("ward", "metro", "nation"):
        recs = [e.payload for e in pb if e.payload["level"] == lv]
        assert len(recs) >= 2
        srev = sum(r["revenue"] for r in recs)
        sexp = sum(r["expense"] for r in recs)
        assert abs((srev - sexp) - (recs[-1]["balance"] - recs[0]["balance"])) < 1.0


# --------------------------------------------------------------------- 決定論
def test_deterministic_on(tmp_path):
    """同一設定 ON を2回実行してイベント列が完全一致。"""
    def run(name):
        sim = _sim(tmp_path, name, n=12, steps=115)
        sim.run()
        return _l1(sim)
    assert run("d1") == run("d2"), "行政 ON で非決定論(イベント列が一致しない)"
