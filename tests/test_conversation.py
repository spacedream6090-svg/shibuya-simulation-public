"""対面会話=確定発火 + 返答保証(ユーザー仕様 2026-07-05)のテスト。

1. 対面確定発火: 同席状態でゲージ≥閾値の申請が抽選抜き(lottery None)で granted。
2. 返答保証: 話しかけられた相手は次の意思決定で trigger "reply" の LLM を出す。
3. 収束: conv_max_turns を絞れば長時間回しても返答が無限連鎖しない(緩く確認)。

k 不変性(face 発火も出来事駆動なので k 間で発火数が揃う)は tests/test_drive.py が担保。
"""
from __future__ import annotations

from collections import Counter

from society.cognition import drive
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dotlist)
    return Simulation(cfg, out_dir=tmp_path / name)


def _colocate(*agents) -> None:
    node = agents[0].node
    for ag in agents:
        ag.loc = "street"
        ag.building = None
        ag.floor = 0
        ag.sleeping = False
        ag.node = node
        ag.x, ag.y = 50.0, 50.0
        ag.conv_turns_left = 3
        ag.conv_cooldown_until = 0


def test_face_fire_is_granted_without_lottery(tmp_path):
    sim = _sim(tmp_path, "face")
    a, b = sim.agents[0], sim.agents[1]
    _colocate(a, b)
    for ag in (a, b):
        ag.drive = 0.95
        ag.drive_threshold = 0.3
        ag.refractory_until = 0
    scheduler._phase_drive(sim, step=0, sim_min=0)
    reqs = [e for e in sim.logger.events if e.kind == "drive_request"
            and e.agent_id in (a.id, b.id)]
    assert reqs, "申請イベントが出ていない"
    face = [e for e in reqs if e.payload["mode"] == "face"]
    assert face, "同席申請が face モードになっていない"
    assert all(e.payload["lottery"] is None for e in face), "face に抽選が混入"
    assert any(e.payload["granted"] for e in face), "face 確定発火が granted でない"


def test_reply_is_guaranteed(tmp_path):
    sim = _sim(tmp_path, "reply")
    a, b = sim.agents[0], sim.agents[1]
    _colocate(a, b)
    # a が話す → 最も近い同席者 b に返答権が渡る
    scheduler._apply(sim, a, {"type": "speak", "text": "こんにちは、いい天気ですね",
                              "use_items": []}, step=0, sim_min=0)
    assert b._reply_to is not None and b._reply_to[0] == a.id
    # 次 step: b の意思決定で抽選なしに返答 LLM が発火
    scheduler._decide(sim, b, step=1, sim_min=10)
    reply_ev = [e for e in sim.logger.events
                if e.kind == "llm_deliberate" and e.agent_id == b.id
                and e.payload.get("trigger") == "reply"]
    assert reply_ev, "返答の llm_deliberate(trigger=reply)が出ていない"
    assert b.conv_turns_left == 2, "返答後に会話ターンが減っていない"
    assert b._reply_to is None, "返答後に返答権が消費されていない"


def test_conversation_converges(tmp_path):
    """conv_max_turns を絞ると返答は有界(無限連鎖しない)。"""
    max_turns, cooldown, steps = 2, 6, 200
    sim = _sim(tmp_path, "conv", n=30, steps=steps,
               **{"drive.conv_max_turns": max_turns,
                  "drive.conv_cooldown_steps": cooldown})
    sim.run()
    replies = Counter(e.agent_id for e in sim.logger.events
                      if e.kind == "llm_deliberate"
                      and e.payload.get("trigger") == "reply")
    # 1エージェントの返答は「セッション上限 × 起こり得るセッション数」で頭打ち。
    bound = max_turns * (steps // cooldown + 2)
    assert replies, "返答が一度も起きていない(機能が無効)"
    worst = max(replies.values())
    assert worst <= bound, f"返答が有界でない: {worst} > {bound}(連鎖の疑い)"
