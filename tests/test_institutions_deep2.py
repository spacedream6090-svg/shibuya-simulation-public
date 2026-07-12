"""制度深化 第2弾(第10バッチ 2026-07-08)= 勾留・解雇規制/退職金・営業許可 のテスト。

対象(調査: docs/research/rights-institutions-gap.md §F/§C/§J):
- 勾留(enforcement.detention_steps): 執行時に違反者を数step行動停止(detention イベント。
  会話・発火・移動なし)。時間による自由の剥奪=司法過程の最小形。
- 解雇規制(career.severance_days / unfair_ratio): 退職金(wage source="severance")+
  不当解雇の生活不安増幅(unemployment payload に unfair)。
- 営業許可(tools.permit_steps / permit_deny_prob): 却下=開業費を払わず出店なし、
  許可待ちの間は販売なし(open_at)。
すべて既定 0/0=純粋既定と L1 完全一致(ゴールデンは test_scenario が別途担保)。mock のみ。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_ENF = {"rules.enabled": "true",
        "institution_routes.enforcement.enabled": "true",
        "agents.personas_file": "data/personas_100_civic.json"}


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


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(全ノブ0)と純粋既定が L1 完全一致(144 step)。新イベントは出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"institution_routes.enforcement.detention_steps": "0",
                  "career.severance_days": "0.0", "career.unfair_ratio": "0.0",
                  "tools.permit_steps": "0", "tools.permit_deny_prob": "0.0"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(制度深化2 seam が no-op でない)"
    for k in ("detention", "venture_permit"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 勾留
def _setup_violation(sim):
    """テスト用: prohibit ルール制定 + 警察官と違反者を同じ nightlife ノードへ配置。"""
    rec = sim.rulebook.enact({"type": "prohibit", "target_cat": "nightlife",
                              "from_h": 0, "to_h": 24},
                             name="夜の店の立入禁止", proposer=0, step=0, day=0)
    assert rec is not None
    officer = next(a for a in sim.agents if a.occupation == "警察官")
    civilian = next(a for a in sim.agents
                    if a.occupation not in ("警察官", "区職員", "消防士"))
    node = next(p["node"] for p in sim.city.poi_list if p["cat"] == "nightlife")
    for a in (officer, civilian):
        a.loc, a.sleeping, a.building = "street", False, None
        a.node = node
        a.x, a.y = sim.city.node_xy(node)
    return officer, civilian


def test_detention_stops_agent(tmp_path):
    """執行+勾留: detention イベントが出て、拘束中は発火・行動せず、明けたら復帰する。"""
    sim = _sim(tmp_path, "det", n=100, steps=1,   # n=100=名簿全員(警察官を確実に含む)
               **{**_ENF, "institution_routes.enforcement.detention_steps": "3"})
    sim.run()
    officer, civ = _setup_violation(sim)
    civ.money = 5000.0
    step = 10
    scheduler._phase_enforcement(sim, step, sim.clock.sim_min(step))
    ev = _kind(sim, "detention")
    assert ev and ev[-1].payload["target"] == civ.id \
        and ev[-1].payload["steps"] == 3, "detention が記録されていない"
    assert civ.detained_until == step + 3 and civ.route == []
    # 拘束中: _decide は stay のみ(返答・発火なし)
    act = scheduler._decide(sim, civ, step + 1, sim.clock.sim_min(step + 1))
    assert act == {"type": "stay"}, "拘束中に行動している"
    # 拘束明け: 通常の行動決定に戻る(stay 以外もあり得る=型だけ確認)
    act2 = scheduler._decide(sim, civ, step + 3, sim.clock.sim_min(step + 3))
    assert isinstance(act2, dict) and "type" in act2


def test_no_detention_when_zero(tmp_path):
    """detention_steps=0(既定): 執行しても拘束なし(罰金のみ=従来と同一)。"""
    sim = _sim(tmp_path, "nodet", n=100, steps=1, **_ENF)
    sim.run()
    officer, civ = _setup_violation(sim)
    civ.money = 5000.0
    scheduler._phase_enforcement(sim, 10, sim.clock.sim_min(10))
    assert _kind(sim, "enforcement"), "執行そのものが起きていない(テスト前提の不備)"
    assert not _kind(sim, "detention") and civ.detained_until == 0


# --------------------------------------------------------------------- 解雇規制
def test_severance_and_unfair_layoff(tmp_path):
    """解雇時: 退職金(日給×日数)が支給され、不当解雇(ratio=1.0)は payload に記録される。"""
    sim = _sim(tmp_path, "sev", steps=1,
               **{"organizations.enabled": "true",
                  "agents.personas_file": "data/personas_100_civic.json",
                  "career.enabled": "true", "career.layoff_prob": "0.0",
                  "career.severance_days": "5",
                  "career.unfair_ratio": "1.0"})
    sim.run()                                          # run 中は解雇なし(day0 の全員失業を防ぐ)
    sim.careercfg["layoff_prob"] = 1.0                 # 手動フェーズでだけ確定解雇
    from society import organizations
    emp = next(a for a in sim.agents                 # career は来街者を対象外とする
               if organizations.is_employee(a) and a.wage > 0 and not a.visitor)
    wage = a_wage = float(emp.wage)
    money0 = emp.money
    sim._career_day = -1                                # 日次ガードをリセット
    scheduler._phase_career(sim, step=144, sim_min=1440)
    lost = [e for e in _kind(sim, "unemployment")
            if e.payload.get("state") == "lost" and e.agent_id == emp.id]
    assert lost, "layoff_prob=1.0 なのに失業していない"
    assert lost[-1].payload.get("unfair") is True, "unfair_ratio=1.0 なのに不当解雇でない"
    sev = [e for e in _kind(sim, "wage")
           if e.payload.get("source") == "severance" and e.agent_id == emp.id]
    assert sev and sev[-1].payload["amount"] == round(wage * 5, 1), \
        f"退職金が日給×5でない: {sev}"
    assert emp.money == money0 + a_wage * 5


# --------------------------------------------------------------------- 営業許可
def _venture_agent(sim):
    a = sim.agents[0]
    a.loc, a.sleeping, a.building = "street", False, None
    a.money = 50000.0
    return a


def test_permit_denied_refunds(tmp_path):
    """deny_prob=1.0: 出店は却下され、開業費は引かれない(venture_permit denied)。"""
    sim = _sim(tmp_path, "deny", steps=1, **{"tools.permit_deny_prob": "1.0"})
    sim.run()
    a = _venture_agent(sim)
    sim.tools._open_venture(sim, a, {"name": "テスト屋台", "offer": "コーヒー"},
                            step=10, sim_min=100)
    assert a.id not in sim.tools.ventures, "却下なのに出店している"
    assert a.money == 50000.0, "却下なのに開業費が引かれている"
    ev = _kind(sim, "venture_permit")
    assert ev and ev[-1].payload["outcome"] == "denied"


def test_permit_wait_gates_sales(tmp_path):
    """permit_steps=6: 許可待ちの間は販売なし、開業時刻を過ぎると販売できる。"""
    sim = _sim(tmp_path, "wait", steps=1, **{"tools.permit_steps": "6",
                                             "tools.buy_prob": "1.0"})
    sim.run()
    a = _venture_agent(sim)
    sim.tools._open_venture(sim, a, {"name": "待ち屋台", "offer": "パン"},
                            step=10, sim_min=100)
    v = sim.tools.ventures[a.id]
    assert v["open_at"] == 16
    gr = _kind(sim, "venture_permit")
    assert gr and gr[-1].payload["outcome"] == "granted" \
        and gr[-1].payload["open_step"] == 16
    buyer = sim.agents[1]
    buyer.loc, buyer.sleeping, buyer.building = "street", False, None
    buyer.node, buyer.money = a.node, 10000.0
    n0 = len(_kind(sim, "venture_sale"))
    sim.tools._buy_at_ventures(sim, buyer, step=12, sim_min=120)   # 許可待ち中
    assert len(_kind(sim, "venture_sale")) == n0, "許可待ち中に販売している"
    sim.tools._buy_at_ventures(sim, buyer, step=16, sim_min=160)   # 開業後
    assert len(_kind(sim, "venture_sale")) == n0 + 1, "開業後に販売できていない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """制度深化2 全ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    ov = {**_ENF, "institution_routes.enforcement.detention_steps": "3",
          "organizations.enabled": "true", "career.enabled": "true",
          "career.severance_days": "5", "career.unfair_ratio": "0.3",
          "tools.permit_steps": "6", "tools.permit_deny_prob": "0.15"}
    a = _sim(tmp_path, "det_a", steps=144, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", steps=144, **ov)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
