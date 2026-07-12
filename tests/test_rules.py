"""制度DSL(ホワイトリスト型ルールの自動制定。ユーザー構想 2026-07-06)。

エージェントが propose で制度を作り、署名 25% で成立した瞬間、機械可読 rule が
シミュ世界の実効ルール(価格・支給・行き先抑制・定期イベント)として自動制定される。

検証項目(4型それぞれ 検証→成立→実効):
  fee          : spend/price が変わる
  bonus        : 対象行動で入金
  curfew       : 時間帯×カテゴリの行き先が抑制される(緩く統計)
  weekly_event : 期日にニュース + 場所ブースト
加えて: 不正 rule の降格 / 同時上限 / 有効期限 / resume / 観測列 / OFF 不変。
"""
from __future__ import annotations

import json
import types

from society.config import load_config
from society.economy import price_of
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.rng import RngHub
from society.rules import RuleBook, apply_bonus, build_rules_cfg

_REF = {"food": 900.0, "cafe": 500.0, "shop": 2500.0, "nightlife": 1800.0,
        "taxi": 500.0, "bus": 230.0}


def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


def _l1(sim) -> list:
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run_l1(tmp_path, name: str, n: int = 20, steps: int = 144, **ov) -> list:
    sim = _sim(tmp_path, name, n=n, steps=steps, **ov)
    sim.run()
    return _l1(sim)


def _free_at(agent, node) -> None:
    agent.loc = "street"
    agent.building = None
    agent.floor = 0
    agent.sleeping = False
    agent.route = []
    agent.exit_intent = False
    agent.homing = False
    agent.activity = ""
    agent.node = node
    agent.x, agent.y = 50.0, 50.0
    agent.stay_until = 0


def _book(**cfg) -> RuleBook:
    return RuleBook(build_rules_cfg(cfg), dict(_REF))


# ============================================================ 1. fee
def test_fee_rule_adjusts_price(tmp_path):
    """fee ルール成立 → price_of / _poi_price が合算適用(残高判定にも反映)。"""
    sim = _sim(tmp_path, "fee", n=4)
    rec = sim.rulebook.enact({"type": "fee", "target_cat": "food", "delta": 300},
                             name="値上げ", proposer=0, step=0, day=0)
    assert rec is not None
    assert price_of("food", sim.economy, sim.rulebook) == 1200.0   # 900 + 300
    from society.cognition.routine import _poi_price
    assert _poi_price(sim, {"cat": "food"}) == 1200.0
    # 他カテゴリは不変
    assert price_of("shop", sim.economy, sim.rulebook) == 2500.0


def test_fee_clamped_to_ratio():
    """複数 fee の合算が ±fee_max_ratio(既定 50%)でクランプされる。"""
    rb = _book()
    for _ in range(2):
        rb.enact({"type": "fee", "target_cat": "food", "delta": 300},
                 name="n", proposer=0, step=0, day=0)
    # 900 + 600 = 1500 だが上限 900*1.5 = 1350 にクランプ
    assert rb.fee_price("food", 900.0) == 1350.0


# ============================================================ 2. bonus
def test_bonus_pays_on_event_attend(tmp_path):
    """bonus(event_attend)成立 → イベント参加者へ入金 + rule_bonus ログ。"""
    sim = _sim(tmp_path, "bonus_ev", n=6, **{"tools.attend_base": 1.0})
    host, a1, a2 = sim.agents[0], sim.agents[1], sim.agents[2]
    node = host.node
    for ag in (host, a1, a2):
        _free_at(ag, node)
    sim.rulebook.enact({"type": "bonus", "behavior": "event_attend", "amount": 300},
                       name="参加奨励", proposer=0, step=0, day=0)
    sim.tools.apply(sim, host, {"type": "host_event", "title": "集まろう",
                                "hours_later": 1}, 0, sim.clock.sim_min(0))
    ev = sim.tools.events[0]
    sim.tools.invite(a1, 0)
    sim.tools.invite(a2, 0)
    before1 = a1.money
    start = ev["start_step"]
    sim.tools.phase(sim, start, sim.clock.sim_min(start))
    assert a1.id in ev["attendees"]
    assert a1.money == before1 + 300
    bonuses = [e for e in sim.logger.events
               if e.kind == "rule_bonus" and e.agent_id == a1.id]
    assert bonuses and bonuses[0].payload["behavior"] == "event_attend"
    assert sim.rulebook.active[0]["spent"] >= 300     # 累計支出を記録


def test_bonus_apply_direct(tmp_path):
    """apply_bonus(park)= scheduler の公園到着で使う関数の直接検証(入金+記録)。"""
    sim = _sim(tmp_path, "bonus_park", n=4)
    rec = sim.rulebook.enact({"type": "bonus", "behavior": "park", "amount": 200},
                             name="公園", proposer=0, step=0, day=0)
    agent = sim.agents[0]
    before = agent.money
    paid = apply_bonus(sim.rulebook, sim, agent, "park", 0, 0)
    assert paid == 200 and agent.money == before + 200
    assert rec["spent"] == 200
    assert [e for e in sim.logger.events if e.kind == "rule_bonus"]
    # 対象外 behavior には支給しない(乱数も副作用もなし)
    assert apply_bonus(sim.rulebook, sim, agent, "flyer_view", 0, 0) == 0.0


# ============================================================ 3. curfew
def test_curfew_suppresses_statistically():
    """curfew(weight=0.2)= 該当時間帯×カテゴリの行き先を ~80% 抑制(緩く統計)。"""
    rb = _book()
    rb.enact({"type": "curfew", "target_cat": "nightlife", "from_h": 18,
              "to_h": 23, "weight": 0.2}, name="夜間抑制", proposer=0, step=0, day=0)
    from society.cognition import routine
    city = types.SimpleNamespace(pois_at_node=lambda n: [{"cat": "nightlife"}])
    sim = types.SimpleNamespace(rulebook=rb, city=city, hub=RngHub(0))
    hour20 = 20 * 60
    suppressed = 0
    trials = 400
    for i in range(trials):
        ag = types.SimpleNamespace(id=i)
        if routine._curfew_suppressed(ag, sim, "X", hour20, 0):
            suppressed += 1
    rate = suppressed / trials
    assert 0.6 < rate < 0.95, rate                    # weight=0.2 → ~0.8 抑制
    # 時間外(hour 10)は抑制なし
    hour10 = 10 * 60
    assert not any(routine._curfew_suppressed(types.SimpleNamespace(id=i), sim,
                                              "X", hour10, 0) for i in range(50))
    # curfew ルールが無いと常に False(乱数を引かない=不変)
    rb2 = _book()
    sim2 = types.SimpleNamespace(rulebook=rb2, city=city, hub=RngHub(0))
    assert not any(routine._curfew_suppressed(types.SimpleNamespace(id=i), sim2,
                                              "X", hour20, 0) for i in range(50))


# ============================================================ 4. weekly_event
def test_weekly_event_fires_news_and_boost(tmp_path):
    """weekly_event(every_days=1)= 期日にニュース配信 + 場所を余暇候補へブースト。"""
    sim = _sim(tmp_path, "weekly", n=6)
    poi = sim.city.poi_list[0]
    rec = sim.rulebook.enact({"type": "weekly_event", "title": "朝市",
                              "place": poi["name"], "every_days": 1},
                             name="朝市", proposer=0, step=0, day=0)
    assert rec is not None
    # day1 の日次境界で発火(step 102 → sim_min 1440 → day 1)
    step, sim_min = 102, sim.clock.sim_min(102)
    assert sim_min // 1440 == 1
    scheduler._phase_rules(sim, step, sim_min)
    assert any(n["title"] == "定期イベント" for n in sim.net.news), "定期ニュース未配信"
    fires = [e for e in sim.logger.events if e.kind == "rule_weekly_fire"]
    assert fires and fires[0].payload["place"] == poi["name"]
    assert poi["node"] in sim.rulebook.today_boost, "場所がブーストされていない"


# ============================================================ 不正 rule → 降格
def test_invalid_rule_demotes_to_text(tmp_path):
    """未知カテゴリ/過大 delta は enact されず、提案は文言だけの制度として成立。"""
    rb = _book()
    assert rb.enact({"type": "fee", "target_cat": "BOGUS", "delta": 100},
                    name="n", proposer=0, step=0, day=0) is None
    assert rb.enact({"type": "fee", "target_cat": "food", "delta": 999999},
                    name="n", proposer=0, step=0, day=0) is None   # 過大(>50%)
    assert rb.enact({"type": "bonus", "behavior": "park", "amount": 999999},
                    name="n", proposer=0, step=0, day=0) is None   # 上限超過
    assert not rb.active

    # 統合: 不正 rule 付き提案は成立するが実効ルール化しない(壊れない)
    sim = _sim(tmp_path, "demote", n=8)                # 閾値 0.25*8 = 2
    author = sim.agents[0]
    sim.tools.apply(sim, author, {"type": "propose", "text": "渋谷に緑を",
                                  "rule": {"type": "fee", "target_cat": "空", "delta": 1}},
                    0, 0)
    for a in (sim.agents[1], sim.agents[2]):
        a.adopted.add("渋谷に緑を")
    sim.tools.phase(sim, 1, 10)
    assert sim.tools.proposals[0]["passed"]
    assert not sim.rulebook.active
    assert not [e for e in sim.logger.events if e.kind == "institution_rule"]
    assert [e for e in sim.logger.events if e.kind == "proposal_passed"]


def test_valid_rule_enacts_through_proposal(tmp_path):
    """有効 rule 付き提案の成立 → institution_rule + 新制度ニュース + fee 実効。"""
    sim = _sim(tmp_path, "enact", n=8)
    author = sim.agents[0]
    sim.tools.apply(sim, author, {"type": "propose", "text": "食事を安く",
                                  "rule": {"type": "fee", "target_cat": "food",
                                           "delta": -100}}, 0, 0)
    for a in (sim.agents[1], sim.agents[2]):
        a.adopted.add("食事を安く")
    sim.tools.phase(sim, 1, 10)
    assert sim.tools.proposals[0]["passed"]
    assert sim.rulebook.active and sim.rulebook.active[0]["type"] == "fee"
    inst = [e for e in sim.logger.events if e.kind == "institution_rule"]
    assert inst and inst[0].payload["proposer"] == author.id
    assert any(n["title"] == "新制度" for n in sim.net.news)
    assert price_of("food", sim.economy, sim.rulebook) == 800.0   # 900 - 100


# ============================================================ 上限・期限
def test_max_active_evicts_oldest():
    """同時アクティブ上限を超えると古い順に失効する。"""
    rb = _book(max_active=2)
    for i in range(3):
        rb.enact({"type": "bonus", "behavior": "park", "amount": 100},
                 name=f"r{i}", proposer=i, step=i, day=0)
    evicted = rb.evict_over_limit()
    assert len(evicted) == 1 and evicted[0]["proposer"] == 0   # 最古が失効
    assert len(rb.active) == 2 and [r["proposer"] for r in rb.active] == [1, 2]


def test_duration_expiry():
    """duration_days の有効期限を過ぎたルールが日次境界で失効する。"""
    rb = _book(duration_days=1)
    rec = rb.enact({"type": "bonus", "behavior": "park", "amount": 100},
                   name="r", proposer=0, step=0, day=0)
    assert rec["expire_step"] == 144
    assert rb.advance_day(0, 0) is not None            # day0: まだ有効
    assert rb.active
    expired, weekly = rb.advance_day(1, 145)           # day1 step145 >= 144
    assert rec in expired and not rb.active


# ============================================================ resume
def test_resume_preserves_rulebook(tmp_path):
    """rulebook が checkpoint を跨いで有効(active ルール・fee 実効が復元)。"""
    sim = _sim(tmp_path, "rb_save", n=6)
    assert sim.rulebook.enact({"type": "fee", "target_cat": "food", "delta": 300},
                              name="n", proposer=0, step=0, day=0) is not None
    path = sim.out_dir / "checkpoint" / "ckpt-000005.pkl.gz"
    checkpoint.save(sim, 5, path)
    sim2 = _sim(tmp_path, "rb_load", n=6)
    assert checkpoint.load(sim2, path) == 5
    assert len(sim2.rulebook.active) == 1
    assert sim2.rulebook.active[0]["type"] == "fee"
    assert price_of("food", sim2.economy, sim2.rulebook) == 1200.0


# ============================================================ 観測
def test_measure_rules_enacted_column():
    """measure が rules_enacted 列を足す(institution_rule があるときのみ・y_external 不変)。"""
    from society.observer import measure

    def ev(step, aid, kind, **payload):
        return {"step": step, "sim_min": step, "agent_id": aid, "kind": kind,
                "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
                "payload": payload}

    meta = [{"id": 3, "name": "A"}, {"id": 4, "name": "B"}]
    events = [ev(0, 3, "institution_rule", rule_id=0, type="fee", proposer=3, rule={})]
    feats = {f["id"]: f for f in measure.agent_features(events, meta)}
    assert feats[3]["rules_enacted"] == 1
    assert feats[4]["rules_enacted"] == 0
    assert feats[3]["Y_external"] == 0                 # y_external には混ざらない
    # institution_rule が無ければ列を足さない(後方互換)
    plain = measure.agent_features([ev(0, 3, "speak", hearers=[], items=[], text="x")],
                                   meta)
    assert all("rules_enacted" not in f for f in plain)


def test_aggregate_n_active_rules_gated(tmp_path):
    """n_active_rules は rules 有効時のみ L2 列に出る(無効は列なし=L2 不変)。"""
    from society.observer.aggregate import collect
    on = _sim(tmp_path, "nar_on", n=6)
    assert "n_active_rules" in collect(on)
    off = _sim(tmp_path, "nar_off", n=6, **{"rules.enabled": "false"})
    assert "n_active_rules" not in collect(off)


# ============================================================ 配線・OFF 不変・決定論
def test_offer_text_and_parse_rule(tmp_path):
    """有効時は propose に rule スロットを併記し、parse_action が rule を受理する。"""
    from society.cognition import deliberate
    sim = _sim(tmp_path, "offer", n=4)
    ag = sim.agents[0]
    txt = sim.tools.offer_text(sim, ag)
    assert txt is not None and '"rule"' in txt
    off = _sim(tmp_path, "offer_off", n=4, **{"rules.enabled": "false"})
    assert '"rule"' not in (off.tools.offer_text(off, off.agents[0]) or "")
    act = deliberate.parse_action(
        '{"action":"propose","text":"X","rule":{"type":"fee",'
        '"target_cat":"food","delta":-100}}')
    assert act["type"] == "propose" and act["rule"]["type"] == "fee"


def test_disabled_clean_and_enabled_deterministic(tmp_path):
    """rules 無効: rule_* イベントが1件も出ない。有効(既定): 同 seed で決定論。"""
    off = _run_l1(tmp_path, "r_off", n=20, steps=144, **{"rules.enabled": "false"})
    off_kinds = {row[2] for row in off}
    assert not (off_kinds & {"institution_rule", "rule_bonus", "rule_expired",
                             "rule_weekly_fire"}), "無効なのに rule_* が出ている"
    a = _run_l1(tmp_path, "r_on1", n=20, steps=144)
    b = _run_l1(tmp_path, "r_on2", n=20, steps=144)
    assert a == b, "有効ランが同 seed で非決定論"
    # 提案が新プロンプトでも作られている(制度DSL 経路が生きている)
    assert any(row[2] == "proposal" for row in a)
