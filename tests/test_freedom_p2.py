"""生活の自己決定 P2(D3 未実装棚卸し。docs/plans/agent-freedom-plan.md #6-#10)のテスト。

方針(既存の鉄則を継承):
- OFF(既定 全 false): 純粋既定と L1 完全一致(ゴールデン golden_baseline_l1.json を守る)。
  新イベント(move_home/partnership_declined)は 0 件・L2 の自由度列は不在(None=列なし)。
- ON: 各アクション(#6-#10)のハンドラ単体 — move_home で home が変わり敷金が引かれる/buy で
  spend(chosen:true)/study イベント/partnership の成立・不成立の両分岐/無許可出店→執行で罰金・閉店。
- R1: _FixedLLM で ON/OFF の generate 呼数が完全一致(メニュー1呼の内容差のみ=呼数不変)。
- 決定論: ON 同士2回で L1 完全一致。choice_points/exercised の計数。
検証は mock / _FixedLLM のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.cognition.deliberate import build_prompt, parse_action
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.aggregate import AGGREGATORS

# 5項目すべて ON(+ closeness を読む partnership 用に relations、閾値用に household)。
_ALL_ON = {
    "freedom.p2.move_home": "true", "freedom.p2.buy": "true",
    "freedom.p2.study": "true", "freedom.p2.partnership": "true",
    "freedom.p2.deviance": "true",
}


def _sim(tmp_path, name, n=15, steps=48, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=48"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _put_near(a, b):
    """b を a と同じ路上ノード・同座標に置く(対面 co-location を作る)。"""
    for x in (a, b):
        x.loc = "street"
        x.sleeping = False
        x.building = None
        x.floor = 0
    b.node = a.node
    b.x, b.y = a.x, a.y


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(全 false)と純粋既定が L1 完全一致。P2 新イベント 0 件・L2 自由度列は不在。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "off", steps=144,
               **{f"freedom.p2.{k}": "false"
                  for k in ("move_home", "buy", "study", "partnership", "deviance")})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(P2 seam が no-op でない)"
    for k in ("move_home", "partnership_declined"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    # L2: 自由度の観測列は OFF では None=列なし(L2 バイト不変)
    assert AGGREGATORS["freedom_choice_points"](pure) is None
    assert AGGREGATORS["freedom_exercised"](pure) is None
    # メニューは既定プロンプトに載らない
    p = build_prompt(pure.agents[0], place_name="自宅", surprise="solo",
                     nearby_names=[], step=1)
    assert "生活の選択" not in p, "OFF なのにプロンプトに P2 メニューがある"


def test_parse_action_p2():
    """P2 アクションの解釈(必須キー・permit 通過)。壊れは None。"""
    assert parse_action('{"action":"move_home","area":"神南"}') == {
        "type": "move_home", "area": "神南"}
    assert parse_action('{"action":"buy","cat":"food"}') == {"type": "buy", "cat": "food"}
    assert parse_action('{"action":"study","topic":"歴史"}') == {
        "type": "study", "topic": "歴史"}
    assert parse_action('{"action":"propose_partnership","to":"花子"}') == {
        "type": "propose_partnership", "to": "花子"}
    assert parse_action('{"action":"break_up"}') == {"type": "break_up"}
    assert parse_action('{"action":"propose_partnership"}') is None
    ov = parse_action('{"action":"open_venture","name":"店","offer":"x","permit":false}')
    assert ov["permit"] is False and ov["type"] == "open_venture"
    ov2 = parse_action('{"action":"open_venture","name":"店","offer":"x"}')
    assert "permit" not in ov2                      # 既定(許可あり)は permit を付けない=不変


# --------------------------------------------------------------------- #6 move_home
def test_move_home_relocates_and_charges_deposit(tmp_path):
    """#6: 敷金を払えるとき空き住戸へ転居し、home が変わり敷金が引かれ move_home が出る。"""
    sim = _sim(tmp_path, "mh", **{"freedom.p2.move_home": "true"})
    a = sim.agents[0]
    a.money = 100000.0
    a.account = 0.0
    old = a.home_building
    scheduler._apply(sim, a, {"type": "move_home", "area": None}, 0, 0)
    assert a.home_building != old, "home_building が変わっていない"
    assert a.home_node == sim.city.building(a.home_building)["entrance"]
    assert a.money == 50000.0, f"敷金(50000)が引かれていない: {a.money}"
    mh = _kind(sim, "move_home")
    assert mh and mh[0].payload == {"from": old, "to": a.home_building, "deposit": 50000.0}
    # 空き住戸=他エージェントの home でない
    occupied = {x.home_building for x in sim.agents if x is not a}
    assert a.home_building not in occupied


def test_move_home_blocked_without_deposit(tmp_path):
    """#6: 敷金に満たない所持金では引っ越せない(客観条件のゲート)。"""
    sim = _sim(tmp_path, "mh2", **{"freedom.p2.move_home": "true"})
    a = sim.agents[0]
    a.money = 1000.0
    a.account = 0.0
    old = a.home_building
    scheduler._apply(sim, a, {"type": "move_home", "area": None}, 0, 0)
    assert a.home_building == old and not _kind(sim, "move_home")


# --------------------------------------------------------------------- #7 buy
def test_buy_spends_with_chosen_flag(tmp_path):
    """#7: 発火時の buy が既存 _spend を通し、spend payload に chosen:true が付く。"""
    sim = _sim(tmp_path, "buy", **{"freedom.p2.buy": "true"})
    a = sim.agents[0]
    a.money = 100000.0
    scheduler._apply(sim, a, {"type": "buy", "cat": "food"}, 0, 0)
    sp = [e for e in _kind(sim, "spend") if e.payload.get("chosen")]
    assert sp, "chosen:true の spend が出ていない"
    assert sp[0].payload["cat"] == "food" and sp[0].payload["amount"] > 0


# --------------------------------------------------------------------- #8 study
def test_study_records_event(tmp_path):
    """#8: study が記録され、その場に留まる(効果は記録のみ=賃金経路なし)。"""
    sim = _sim(tmp_path, "study", **{"freedom.p2.study": "true"})
    a = sim.agents[0]
    scheduler._apply(sim, a, {"type": "study", "topic": "数学"}, 5, 50)
    st = _kind(sim, "study")
    assert st and st[0].payload["subject"] == "数学" and st[0].payload["chosen"] is True
    assert a.stay_until >= 5 + 2, "その場に留まっていない"


# --------------------------------------------------------------------- #9 partnership
def test_partnership_forms_on_high_closeness(tmp_path):
    """#9: closeness が閾値以上なら既存の世帯結合処理で partner_formed が出て双方が交際に。"""
    sim = _sim(tmp_path, "pf", **{"freedom.p2.partnership": "true"})
    a, b = sim.agents[0], sim.agents[1]
    a.partner_id = b.partner_id = None
    _put_near(a, b)
    a.mem.relations[b.id] = {"name": b.name, "count": 9, "last_step": 0,
                             "last": "", "closeness": 20.0}     # 閾値 15 超
    scheduler._apply(sim, a, {"type": "propose_partnership", "to": b.name}, 0, 0)
    assert a.partner_id == b.id and b.partner_id == a.id
    assert _kind(sim, "partner_formed")
    assert [e for e in _kind(sim, "life_event") if e.payload["kind"] == "partner"]
    assert not _kind(sim, "partnership_declined")


def test_partnership_declines_on_low_closeness(tmp_path):
    """#9: closeness が閾値未満なら partnership_declined(新 kind)。交際は成立しない。"""
    sim = _sim(tmp_path, "pd", **{"freedom.p2.partnership": "true"})
    a, b = sim.agents[0], sim.agents[1]
    a.partner_id = b.partner_id = None
    _put_near(a, b)
    a.mem.relations[b.id] = {"name": b.name, "count": 2, "last_step": 0,
                             "last": "", "closeness": 3.0}       # 閾値 15 未満
    scheduler._apply(sim, a, {"type": "propose_partnership", "to": b.name}, 0, 0)
    assert a.partner_id is None and b.partner_id is None
    pd = _kind(sim, "partnership_declined")
    assert pd and pd[0].payload["to"] == b.id
    assert not _kind(sim, "partner_formed")


def test_break_up_separates_partners(tmp_path):
    """#9: break_up は既存 relation_break + life_event(breakup)で双方の partner_id を外す。"""
    sim = _sim(tmp_path, "bu", **{"freedom.p2.partnership": "true"})
    a, b = sim.agents[0], sim.agents[1]
    a.partner_id, b.partner_id = b.id, a.id
    scheduler._apply(sim, a, {"type": "break_up"}, 0, 0)
    assert a.partner_id is None and b.partner_id is None
    rb = _kind(sim, "relation_break")
    assert rb and rb[0].payload["cause"] == "breakup" and rb[0].payload["other"] == b.id
    assert [e for e in _kind(sim, "life_event") if e.payload["kind"] == "breakup"]


# --------------------------------------------------------------------- #10 deviance
def test_unpermitted_venture_opens_and_is_enforced(tmp_path):
    """#10: 無許可出店は permitted:false で即開店し、近傍の警察官が摘発(罰金+閉店)。"""
    sim = _sim(tmp_path, "dev", **{**{"freedom.p2.deviance": "true"},
                                   "institution_routes.enforcement.enabled": "true"})
    owner, officer = sim.agents[0], sim.agents[1]
    officer.occupation = "警察官"
    owner.money = 100000.0
    _put_near(owner, officer)
    tools = sim.tools
    tools.apply(sim, owner, {"type": "open_venture", "name": "無許可屋台",
                             "offer": "軽食", "permit": False}, 0, 0)
    v = tools.ventures.get(owner.id)
    assert v is not None and v["permitted"] is False
    vo = [e for e in _kind(sim, "venture_open") if e.payload.get("permitted") is False]
    assert vo, "venture_open に permitted:false が付いていない"
    before = owner.money
    scheduler._phase_enforcement(sim, 0, 0)
    enf = _kind(sim, "enforcement")
    assert enf and enf[0].payload["target"] == owner.id, "摘発(enforcement)が出ていない"
    assert owner.money < before, "罰金が引かれていない"
    assert owner.id not in tools.ventures, "無許可出店が閉店されていない"
    assert _kind(sim, "venture_close"), "venture_close が出ていない"


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: P2 ON(全 true)/OFF で generate 呼数が完全一致(288 step)。"""
    def run(name, on):
        ov = _ALL_ON if on else {}
        sim = _sim(tmp_path, name, steps=288, **ov)
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    on = run("r1_on", True)
    off = run("r1_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"


# --------------------------------------------------------------------- 決定論
def test_on_deterministic(tmp_path):
    """P2 ON(全 true)同士 2 回で L1 完全一致(mock・決定論)。"""
    a = _sim(tmp_path, "det_a", steps=144, **_ALL_ON)
    a.run()
    b = _sim(tmp_path, "det_b", steps=144, **_ALL_ON)
    b.run()
    assert _l1(a) == _l1(b), "P2 ON の決定論が崩れている"


# --------------------------------------------------------------------- 観測(計数)
def test_choice_points_and_exercised_counted(tmp_path):
    """ON: メニュー提示(choice_points)と行使(exercised)が計数され L2 列に出る。"""
    sim = _sim(tmp_path, "cnt", steps=144,
               **{**_ALL_ON, "relations.enabled": "true", "household.enabled": "true"})
    cum = {"choice_points": 0, "exercised": 0}
    for step in range(sim.cfg.run.n_steps):
        scheduler.run_step(sim, step)
        cum["choice_points"] += sim.freedom_stats["choice_points"]
        cum["exercised"] += sim.freedom_stats["exercised"]
    assert cum["choice_points"] > 0, "メニュー提示が1度も計数されていない"
    assert cum["exercised"] > 0, "P2 行使が1度も計数されていない"
    # L2 列は ON では int(0 以上)で出る(OFF は None=列なし)
    assert isinstance(AGGREGATORS["freedom_choice_points"](sim), int)
    assert isinstance(AGGREGATORS["freedom_exercised"](sim), int)
