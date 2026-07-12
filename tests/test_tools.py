"""「世界を変える」ツール群(host_event/post_flyer/found_group/propose/open_venture)。

4 軸(モノ=出店 / 制度=提案 / 人=勉強会 / 虚構=イベント・ビラ・グループ)を最小で覆う。
中立提示・自然使用の観察が目的。R1(提示は k と無関係)/ R4(効果は客観カウント)を前提に、
各ツールの発火→効果を確認する。決定論と no-fingerprint は既存テストと本ファイル末尾で担保。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dotlist)
    return Simulation(cfg, out_dir=tmp_path / name)


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


# --------------------------------------------------------------------------- #
def test_host_event_announces_attends_and_teaches(tmp_path):
    """開催→SNS告知(event_id 付き)→参加者が会場に→event_attend→勉強会で語を教育。"""
    sim = _sim(tmp_path, "host", n=6, **{"tools.attend_base": 1.0})
    host, a1, a2 = sim.agents[0], sim.agents[1], sim.agents[2]
    node = host.node
    for ag in (host, a1, a2):
        _free_at(ag, node)
    # 主催者は自分の造語をタイトルに入れる(勉強会=教育軸の帯域)
    sim.labels.coin(host, "シブり現象", step=0, sim_min=0, logger=sim.logger)
    title = "シブり現象を語る会"
    sim.tools.apply(sim, host, {"type": "host_event", "title": title,
                                "hours_later": 1}, 0, sim.clock.sim_min(0))
    ev = sim.tools.events[0]
    # 開催宣言 + 自動 SNS 告知(event_id で参加告知に紐付く)
    assert [e for e in sim.logger.events if e.kind == "event_host"]
    assert any(p.get("event_id") == 0 for p in sim.net.posts), "告知SNSが出ていない"
    # 告知を見た2人に invite(SNS 経由の代わりにここでは直接付与=同じ機構)
    sim.tools.invite(a1, 0)
    sim.tools.invite(a2, 0)
    start = ev["start_step"]
    sim.tools.phase(sim, start, sim.clock.sim_min(start))
    attend = {e.agent_id for e in sim.logger.events if e.kind == "event_attend"}
    assert {a1.id, a2.id} <= attend, "参加者の event_attend が出ていない"
    # 終了時: タイトルの語を2回分聴取 → 閾値2を1回の教育で満たす(採用)
    end = ev["end_step"]
    sim.tools.phase(sim, end, sim.clock.sim_min(end))
    assert "シブり現象" in a1.adopted and "シブり現象" in a2.adopted, "教育で採用に至らない"


def test_flyer_view_hears_words_and_expires(tmp_path):
    """貼る→別エージェント到着で flyer_view + 語の聴取(ユニーク)→ ttl 失効。"""
    sim = _sim(tmp_path, "flyer", n=4)
    author, viewer = sim.agents[0], sim.agents[1]
    node = author.node
    _free_at(author, node)
    sim.labels.coin(author, "ザワめき", step=0, sim_min=0, logger=sim.logger)
    text = "ここで「ザワめき」が起きてる"
    sim.tools.apply(sim, author, {"type": "post_flyer", "text": text}, 0, 0)
    assert sim.tools.flyers_by_node[node], "ビラが登録されていない"
    # 通行人が到着 → 閲覧 + 語の聴取(channel=flyer の transmission)
    viewer.node = node
    sim.tools.on_arrive(sim, viewer, 1, 10)
    views = [e for e in sim.logger.events
             if e.kind == "flyer_view" and e.agent_id == viewer.id]
    assert views, "flyer_view が出ていない"
    trans = [e for e in sim.logger.events
             if e.kind == "transmission" and e.agent_id == viewer.id
             and e.payload.get("channel") == "flyer"]
    assert trans, "ビラ経由の語の聴取(transmission channel=flyer)が無い"
    # ユニーク: 再到着では二重に閲覧しない
    sim.tools.on_arrive(sim, viewer, 2, 20)
    views2 = [e for e in sim.logger.events
              if e.kind == "flyer_view" and e.agent_id == viewer.id]
    assert len(views2) == len(views), "同じビラを二重閲覧している"
    # ttl 失効
    f = sim.tools.flyers_by_node[node][0]
    sim.tools.phase(sim, f["expire_step"], sim.clock.sim_min(f["expire_step"]))
    assert not sim.tools.flyers_by_node[node], "ttl 切れのビラが撤去されない"
    assert [e for e in sim.logger.events if e.kind == "flyer_expire"]


def test_group_founding_and_join_with_mutual_follow(tmp_path):
    """結成→(名前を知り関係者がいる者が)加入→相互フォロー。"""
    sim = _sim(tmp_path, "group", n=6, **{"tools.group_join_prob": 1.0})
    founder, joiner = sim.agents[0], sim.agents[1]
    sim.tools.apply(sim, founder, {"type": "found_group", "name": "渋谷夜会",
                                   "purpose": "ゆるくつながる"}, 0, 0)
    g = sim.tools.groups[0]
    assert founder.id in g["members"]
    assert [e for e in sim.logger.events if e.kind == "group_found"]
    # 加入条件: (a) グループ名を採用済み + (b) メンバーが関係台帳にいる
    joiner.adopted.add("渋谷夜会")
    joiner.mem.record_contact(founder.id, founder.name, 0)
    sim.tools.phase(sim, 1, 10)
    assert joiner.id in g["members"], "条件を満たす者が加入していない"
    assert [e for e in sim.logger.events
            if e.kind == "group_join" and e.agent_id == joiner.id]
    # 相互フォロー
    assert founder.id in sim.net.follows.get(joiner.id, set())
    assert joiner.id in sim.net.follows.get(founder.id, set())


def test_proposal_exposure_signs_and_passes(tmp_path):
    """提案→露出2回で署名→閾値到達で proposal_passed + 世界ニュース配信。"""
    sim = _sim(tmp_path, "prop", n=8)     # 閾値 = 0.25 * 8 = 2 人
    author = sim.agents[0]
    text = "渋谷に緑を増やそう"
    sim.tools.apply(sim, author, {"type": "propose", "text": text}, 0, 0)
    pr = sim.tools.proposals[0]
    assert author.id in pr["supporters"], "起案者は初期署名者"
    # 別エージェントが露出2回(complex contagion)で採用=署名
    sup = sim.agents[1]
    for _ in range(2):
        scheduler._hear_words(sim, sup, [text], author.id, "sns", 0, 0)
    assert text in sup.adopted
    sim.tools.phase(sim, 1, 10)
    assert sup.id in pr["supporters"]
    assert [e for e in sim.logger.events if e.kind == "proposal_support"]
    # 署名が全体の 25%(=2人)到達 → 成立 + ニュース
    assert pr["passed"], "閾値到達で成立していない"
    assert [e for e in sim.logger.events if e.kind == "proposal_passed"]
    assert any("住民提案" in a["title"] for a in sim.net.news), "世界ニュースが出ていない"


def test_venture_offer_gated_by_money_and_sale_flow(tmp_path):
    """所持金不足では提示されない/開業→通行人が購入→売上(客 spend・店主 sale)。"""
    sim = _sim(tmp_path, "vent", n=6, **{"tools.buy_prob": 1.0})
    owner = sim.agents[0]
    node = owner.node
    _free_at(owner, node)
    # 所持金 < venture_cost では open_venture を提示しない(R1: 客観条件のみ)
    owner.money = 1000.0
    assert "open_venture" not in (sim.tools.offer_text(sim, owner) or "")
    owner.money = 50000.0
    assert "open_venture" in (sim.tools.offer_text(sim, owner) or "")
    # 開業(開業費を差し引く)
    cost = sim.tools.cfg["venture_cost"]
    sim.tools.apply(sim, owner, {"type": "open_venture", "name": "渋谷コーヒー",
                                 "offer": "コーヒー"}, 0, 0)
    assert owner.money == 50000.0 - cost
    assert owner.id in sim.tools.ventures
    assert [e for e in sim.logger.events if e.kind == "venture_open"]
    # 通行人が到着して購入(buy_prob=1.0 で確定)
    cust = sim.agents[1]
    cust.node = node
    cust.money = 5000.0
    before = owner.money
    sim.tools.on_arrive(sim, cust, 1, 10)
    price = sim.tools.cfg["venture_price"]
    sales = [e for e in sim.logger.events if e.kind == "venture_sale"]
    assert sales, "venture_sale が出ていない"
    assert owner.money == before + price          # 店主に売上
    assert cust.money == 5000.0 - price           # 客が支払い


def test_measure_adds_tool_columns_without_touching_y_external():
    """measure が列を追加し、既存 Y_external の合成式は変えない。"""
    from society.observer import measure

    def ev(step, aid, kind, **payload):
        return {"step": step, "sim_min": step, "agent_id": aid, "kind": kind,
                "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
                "payload": payload}

    events = [
        ev(0, 0, "event_host", event_id=0, title="t"),
        ev(1, 1, "event_attend", event_id=0, host=0, title="t"),
        ev(2, 0, "group_found", group_id=0, name="g", founder=0),
        ev(3, 1, "group_join", group_id=0, name="g", founder=0),
        ev(4, 0, "venture_open", name="v"),
        ev(5, 0, "venture_sale", amount=800, balance=800, buyer=1),
    ]
    meta = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}]
    feats = {f["id"]: f for f in measure.agent_features(events, meta)}
    a0 = feats[0]
    assert a0["events_hosted"] == 1
    assert a0["event_attendees_drawn"] == 1
    assert a0["groups_founded"] == 1
    assert a0["group_members_recruited"] == 1
    assert a0["ventures_opened"] == 1
    assert a0["venture_revenue"] == 800
    assert a0["Y_external"] == 0        # ツール列は Y_external に混ざらない


def test_tools_run_is_deterministic_and_actually_used(tmp_path):
    """フルラン(mock 40人1日相当)で決定論 + ツールが実際に使われている。"""
    def run(name):
        cfg = load_config(["run.seed=9", "run.n_agents=30", "run.n_steps=144",
                           f"run.name={name}"])
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.run()
        return [(e.step, e.agent_id, e.kind,
                 json.dumps(e.payload, sort_keys=True, ensure_ascii=False))
                for e in sim.logger.events]

    a = run("t1")
    b = run("t2")
    assert a == b, "同 seed でツール入りランが一致しない(非決定論)"
    kinds = {e[2] for e in a}
    assert kinds & {"group_found", "proposal", "event_host", "flyer_post",
                    "venture_open"}, "ツールが1つも使われていない"
