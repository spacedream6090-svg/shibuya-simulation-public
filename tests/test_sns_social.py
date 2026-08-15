"""SNS 反応(#14)・founder タイムライン優先枠(#13)・イベント屋内参加(#12)。

- いいね/リシェアの状態更新 + リシェア経由の transmission(channel="sns")。
- founder 優先枠が member の timeline に必ず載る / priority 空の読者は旧ロジック。
- 屋内エージェント(勤務・睡眠でない)のイベント参加(退館→再試行)。
- 決定論(同 seed 同結果)。
"""
from __future__ import annotations

import json

import numpy as np

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.net.internet import Internet


def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


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
def test_like_reshare_state_and_reshare_transmission(tmp_path):
    sim = _sim(tmp_path, "react", n=4)
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    sim.labels.coin(a, "リシェア語", step=0, sim_min=0, logger=sim.logger)
    pid = sim.net.post(a.id, "リシェア語がいい", ["リシェア語"], 0)

    # いいね: 状態更新のみ + 著者 id を返す
    assert sim.net.react(b.id, pid, "like", 0, author_name=a.name) == a.id
    assert b.id in sim.net.posts[pid]["likes"]

    # リシェア: reshares +1、"RT @元著者名: 本文"、items 同一、rt_of=元
    sim.net.react(b.id, pid, "reshare", 0, author_name=a.name)
    assert sim.net.posts[pid]["reshares"] == 1
    rt = sim.net.posts[-1]
    assert rt["author"] == b.id
    assert rt["items"] == ["リシェア語"]
    assert rt["text"].startswith("RT @") and rt.get("rt_of") == pid

    # リシェア経由でフォロワーが語を聴取 → transmission(channel="sns", from=リシェア主)
    scheduler._hear_words(sim, c, rt["items"], rt["author"], "sns", 1, 10)
    trans = [e for e in sim.logger.events
             if e.kind == "transmission" and e.agent_id == c.id
             and e.payload.get("channel") == "sns"]
    assert trans, "リシェア経由の SNS transmission が無い"
    assert trans[0].payload.get("from") == b.id


def test_sns_react_helper_logs_event_and_drives_author(tmp_path):
    sim = _sim(tmp_path, "reacth", n=4)
    a, b = sim.agents[0], sim.agents[1]
    pid = sim.net.post(a.id, "やあ", [], 0)
    before = a.drive
    scheduler._sns_react(sim, b, sim.net.posts[pid], "like", "sns_like", 1, 10)
    likes = [e for e in sim.logger.events if e.kind == "sns_like"]
    assert likes and likes[0].payload["author"] == a.id
    assert likes[0].payload["post_id"] == pid
    assert a.drive >= before               # 著者に drive.add("sns")(states は不変)


def test_timeline_priority_includes_founder_empty_is_old_logic():
    net = Internet(feed_size=3)
    net.init_follows([0, 1, 5, 9], np.random.default_rng(0), k=0)  # フォローなし
    net.post(9, "founder", [], 0)                    # pid 0(最古)
    for i in range(5):
        net.post(5, f"noise{i}", [], 0)              # pid 1..5

    # priority 空 → 完全に旧ロジック: (followed+rest)[-3:] = pid 3,4,5(著者9は落ちる)
    tl0 = net.timeline_for(0)
    assert [p["id"] for p in tl0] == [3, 4, 5]
    assert 9 not in [p["author"] for p in tl0]

    # priority あり → 著者9(founder)の投稿を必ず含む
    net.set_priority(1, 9)
    tl1 = net.timeline_for(1)
    assert 9 in [p["author"] for p in tl1]
    assert len(tl1) <= 3


def test_group_join_sets_priority_and_founder_post_reaches_member(tmp_path):
    sim = _sim(tmp_path, "grp", n=6, **{"tools.group_join_prob": 1.0})
    founder, joiner = sim.agents[0], sim.agents[1]
    sim.tools.apply(sim, founder, {"type": "found_group", "name": "渋谷夜会",
                                   "purpose": "つながる"}, 0, 0)
    g = sim.tools.groups[0]
    joiner.adopted.add("渋谷夜会")
    joiner.mem.record_contact(founder.id, founder.name, 0)
    sim.tools.phase(sim, 1, 10)
    assert joiner.id in g["members"]
    assert founder.id in sim.net.priority.get(joiner.id, set())

    # タイムラインを別著者で溢れさせても founder の新着は必ず載る
    other = sim.agents[3].id
    sim.net.follow(joiner.id, other)
    pid = sim.net.post(founder.id, "founderの告知", [], 2)
    for i in range(sim.net.feed_size + 3):
        sim.net.post(other, f"n{i}", [], 2)
    tl = sim.net.timeline_for(joiner.id)
    assert pid in [p["id"] for p in tl], "founder の投稿が優先枠で載っていない"


def test_indoor_agent_attends_event(tmp_path):
    sim = _sim(tmp_path, "indoor", n=6, **{"tools.attend_base": 1.0})
    host, indoor = sim.agents[0], sim.agents[1]
    _free_at(host, host.node)
    sim.tools.apply(sim, host, {"type": "host_event", "title": "屋内会",
                                "hours_later": 1}, 0, sim.clock.sim_min(0))
    ev = sim.tools.events[0]

    # indoor: 建物の中(勤務でも睡眠でもない)+ 招待済み
    indoor.building = "b_test"
    indoor.loc = "street"
    indoor.sleeping = False
    indoor.activity = ""
    indoor.route = []
    indoor.exit_intent = False
    indoor.homing = False
    indoor.node = "elsewhere"
    sim.tools.invite(indoor, ev["id"])

    start = ev["start_step"]
    sim.tools.phase(sim, start, sim.clock.sim_min(start))
    # 屋内枝: 退館誘発(stay_until=step)+ 参加予約
    assert indoor._attending_event == ev["id"]
    assert indoor.stay_until == start
    assert indoor.id not in ev["attendees"]        # まだ屋内 = 未参加

    # 退館して会場ノードに出た → 再試行(event_retry)で参加確定
    indoor.building = None
    indoor.node = ev["node"]
    indoor.loc = "street"
    sim.tools.phase(sim, start + 1, sim.clock.sim_min(start + 1))
    assert indoor.id in ev["attendees"]
    assert [e for e in sim.logger.events
            if e.kind == "event_attend" and e.agent_id == indoor.id]


def test_sns_opinion_run_is_deterministic(tmp_path):
    def run(name):
        cfg = load_config(["run.seed=21", "run.n_agents=12", "run.n_steps=60",
                           f"run.name={name}"])
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.run()
        return [(e.step, e.agent_id, e.kind,
                 json.dumps(e.payload, sort_keys=True, ensure_ascii=False))
                for e in sim.logger.events]

    assert run("s1") == run("s2")
