"""制度深化 第3弾(2026-07-08)= 最低賃金の床・代表制議会・立退き・破産サイクル のテスト。

対象(rights-institutions-gap の見送り分 + 最低賃金の一次確認):
- 最低賃金の床(economy.min_wage_hourly): 一次確認=東京都最低賃金 1,226円/時
  (2025-10-03 発効・東京労働局)。>0 で雇用者賃金へ床を適用。自営は対象外。
- 代表制議会(institution_routes.assembly): 意見最近傍の決定論選挙(council_elected)+
  提案の採決を露出者でなく議会が行う(vote_cast に council 印)。
- 立退き(economy.accounts.eviction_days): 滞納の継続→住居喪失(路上就寝・家賃停止)
  →完済で再入居(eviction イベントの evicted/rehoused サイクル)。
- 破産(economy.accounts.bankruptcy_days): 免責(rent_due 消滅)+自由財産以外の圧縮+
  制限期間(出店不可)+屋台の強制閉店(bankruptcy イベント)。
すべて既定 OFF=純粋既定と L1 完全一致。mock のみ・乱数なし(決定論)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.economy import build_economy
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


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(全ノブ0)と純粋既定が L1 完全一致(144 step)。新イベントは出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"economy.min_wage_hourly": "0.0",
                  "institution_routes.assembly.enabled": "false",
                  "economy.accounts.eviction_days": "0",
                  "economy.accounts.bankruptcy_days": "0"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(制度深化3 seam が no-op でない)"
    for k in ("council_elected", "eviction", "bankruptcy"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 最低賃金の床
def test_min_wage_floor():
    """床 1226円/h: バイト時給と店員日給(<床×8h)が持ち上がり、自営は対象外で不変。"""
    base = build_economy(None)
    assert base["wages"]["part_time_hourly"] == 1100 and base["wages"]["店員"] == 9000, \
        "既定賃金が変わっている(テスト前提の更新が必要)"
    floored = build_economy({"min_wage_hourly": 1226,
                             "wages": {"自営": 5000}})       # 自営を床×8h 未満にして対象外を確認
    w = floored["wages"]
    assert w["part_time_hourly"] == 1226, "バイト時給に床が効いていない"
    assert w["店員"] == 1226 * 8, "店員日給が床×8h に持ち上がっていない"
    assert w["会社員"] == 12000, "床以上の日給が変わった(床は max のはず)"
    assert w["自営"] == 5000, "自営(労働者でない)に床が掛かっている"
    assert floored["min_wage_hourly"] == 1226


# --------------------------------------------------------------------- 代表制議会
def test_assembly_election_and_council_vote(tmp_path):
    """day0 に決定論選挙(council_elected)→ 採決は露出者でなく議会が行う(council 印)。"""
    sim = _sim(tmp_path, "asm", steps=1,
               **{"institution_routes.vote.enabled": "true",
                  "institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.size": "3"})
    sim.run()
    ev = _kind(sim, "council_elected")
    assert ev and ev[0].payload["term"] == 1, "day0 に議会が選ばれていない"
    council = sim.council
    residents = {a.id for a in sim.agents if not a.visitor}
    assert len(council["members"]) == 3
    assert set(council["members"]) <= residents, "来街者が議員になっている"
    # 2回目の呼び出し(同日・任期内)では改選しない
    scheduler._phase_assembly(sim, step=2, sim_min=20)
    assert len(_kind(sim, "council_elected")) == 1, "任期内に改選している"
    # 採決: supporters でなく議会が投票する(vote_cast は size 件・council 印つき)
    author = sorted(residents)[0]
    pr = {"id": 999, "author": author, "text": "テスト提案",
          "supporters": set(sorted(residents)[:10]), "decided": False}
    vcfg = sim.routescfg["vote"]
    n0 = len(_kind(sim, "vote_cast"))
    sim.tools._resolve_vote(sim, pr, step=10, sim_min=100, vcfg=vcfg)
    casts = _kind(sim, "vote_cast")[n0:]
    assert len(casts) == 3, f"議会採決の投票数が議席数でない: {len(casts)}"
    assert all(c.payload.get("council") is True for c in casts)
    assert {c.agent_id for c in casts} == set(council["members"])
    assert _kind(sim, "vote_result")[-1].payload.get("by") == "council"


# --------------------------------------------------------------------- 立退き
def test_eviction_and_rehouse_cycle(tmp_path):
    """滞納2日→立退き(路上就寝・grievance+)→完済→再入居のサイクルが閉じる。"""
    sim = _sim(tmp_path, "evict", steps=1,
               **{"economy.accounts.enabled": "true",
                  "economy.accounts.eviction_days": "2"})
    sim.run()
    acc = sim.economy["accounts"]
    a = next(x for x in sim.agents if not x.visitor and x.home_building)
    a.rent_due, a.account = 5000.0, 0.0
    g0 = float(a.states.get("grievance", 0.0))
    scheduler._eviction_bankruptcy_day(sim, a, acc, step=144, sim_min=1440)
    assert a.arrears_days == 1 and not a.evicted
    scheduler._eviction_bankruptcy_day(sim, a, acc, step=288, sim_min=2880)
    ev = _kind(sim, "eviction")
    assert a.evicted and ev and ev[-1].payload["phase"] == "evicted"
    assert float(a.states["grievance"]) > g0, "立退きの生活不安が入っていない"
    # 物理: 立退き中は自宅建物に入れない=路上就寝
    a.sleeping, a.building = False, None
    scheduler._apply(sim, a, {"type": "go_to_bed"}, step=300, sim_min=3000)
    assert a.sleeping and a.building is None, "立退き中に自宅建物へ入っている"
    # 完済 → 再入居(サイクルの閉じ)
    a.rent_due = 0.0
    scheduler._eviction_bankruptcy_day(sim, a, acc, step=432, sim_min=4320)
    assert not a.evicted and _kind(sim, "eviction")[-1].payload["phase"] == "rehoused"


# --------------------------------------------------------------------- 破産
def test_bankruptcy_discharge_and_restriction(tmp_path):
    """滞納2日→破産: 免責(rent_due=0)+自由財産まで圧縮+屋台の強制閉店+出店制限。"""
    sim = _sim(tmp_path, "bank", steps=1,
               **{"economy.accounts.enabled": "true",
                  "economy.accounts.bankruptcy_days": "2",
                  "economy.accounts.bankruptcy_keep": "10000",
                  "economy.accounts.bankruptcy_restrict_days": "1"})
    sim.run()
    acc = sim.economy["accounts"]
    a = next(x for x in sim.agents if not x.visitor)
    a.loc, a.sleeping, a.building = "street", False, None
    a.money = 50000.0
    sim.tools._open_venture(sim, a, {"name": "破産前の屋台", "offer": "コーヒー"},
                            step=10, sim_min=100)
    assert a.id in sim.tools.ventures, "テスト前提: 出店できていない"
    a.rent_due, a.account = 8000.0, 30000.0
    scheduler._eviction_bankruptcy_day(sim, a, acc, step=144, sim_min=1440)
    scheduler._eviction_bankruptcy_day(sim, a, acc, step=288, sim_min=2880)
    ev = _kind(sim, "bankruptcy")
    assert ev and ev[-1].payload["venture_closed"] is True
    assert a.rent_due == 0.0, "免責されていない"
    assert a.money + a.account == 10000.0, "自由財産(keep)まで圧縮されていない"
    assert a.bankrupt_until == 288 + 144
    assert a.id not in sim.tools.ventures, "屋台が強制閉店されていない"
    closes = _kind(sim, "venture_close")
    assert closes and closes[-1].payload.get("reason") == "bankruptcy"
    # 制限期間中は出店できない(却下=venture_permit reason=bankruptcy)
    a.money = 50000.0
    sim.tools._open_venture(sim, a, {"name": "再挑戦の屋台", "offer": "パン"},
                            step=300, sim_min=3000)
    assert a.id not in sim.tools.ventures
    perms = _kind(sim, "venture_permit")
    assert perms and perms[-1].payload.get("reason") == "bankruptcy"
    # 制限明けは出店できる(免責後の再出発=サイクルの閉じ)
    sim.tools._open_venture(sim, a, {"name": "再出発の屋台", "offer": "パン"},
                            step=288 + 144, sim_min=4320)
    assert a.id in sim.tools.ventures, "制限明けに出店できていない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """制度深化3 全ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    ov = {"economy.min_wage_hourly": "1226",
          "economy.accounts.enabled": "true",
          "economy.accounts.eviction_days": "2",
          "economy.accounts.bankruptcy_days": "4",
          "institution_routes.vote.enabled": "true",
          "institution_routes.assembly.enabled": "true"}
    a = _sim(tmp_path, "d3_a", steps=144, **ov)
    a.run()
    b = _sim(tmp_path, "d3_b", steps=144, **ov)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
