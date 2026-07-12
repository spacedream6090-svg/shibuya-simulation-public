"""意見力学(Friedkin-Johnsen #16)の検証。

- expose の式一致・anchor 不変量・収束・clip(純関数)。
- speak 経由で opinion_shift イベントが出て聞き手の意見が更新される。
- measure.agent_features の c_opinion_moved 計上(source 帰属)。
- 既定ラン(mock 10人×48step)で Y_external が opinion 導入前と不変
  (従来4成分の和のままで、opinion/SNS 反応は混ざらない)。
"""
from __future__ import annotations

import types

from society import opinion
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.lang.sentiment import valence
from society.observer import measure


def _sim(tmp_path, name: str, n: int = 4, steps: int = 1, **ov) -> Simulation:
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


def _ev(step, agent_id, kind, **payload):
    return {"step": step, "sim_min": step, "agent_id": agent_id, "kind": kind,
            "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
            "payload": payload}


# --------------------------------------------------------------------------- #
def test_expose_formula_invariant_convergence_clip():
    # 単一 exposure の式一致: o ← (1−s)·anchor + s·((1−w)·o + w·v)
    ag = types.SimpleNamespace(opinion=0.2, opinion_anchor=0.1,
                               opinion_susceptibility=0.6)
    v, w = 0.8, 0.2
    social = (1 - w) * 0.2 + w * 0.8
    expect = (1 - 0.6) * 0.1 + 0.6 * social
    assert abs(opinion.expose(ag, v, w) - expect) < 1e-12

    # anchor 不変量: o==anchor かつ v==anchor なら動かない
    ag2 = types.SimpleNamespace(opinion=0.3, opinion_anchor=0.3,
                                opinion_susceptibility=0.7)
    assert abs(opinion.expose(ag2, 0.3, 0.5) - 0.3) < 1e-12

    # anchor への収束: v=anchor を繰り返し曝露 → 意見が anchor(=0)へ寄る
    ag3 = types.SimpleNamespace(opinion=0.9, opinion_anchor=0.0,
                                opinion_susceptibility=0.5)
    for _ in range(50):
        ag3.opinion = opinion.expose(ag3, 0.0, 0.2)
    assert abs(ag3.opinion) < 0.9 and abs(ag3.opinion) < 1e-3

    # clip: [−1, 1] を超えない
    hi = types.SimpleNamespace(opinion=0.0, opinion_anchor=1.0,
                               opinion_susceptibility=1.0)
    assert opinion.expose(hi, 5.0, 1.0) == 1.0
    lo = types.SimpleNamespace(opinion=0.0, opinion_anchor=-1.0,
                               opinion_susceptibility=1.0)
    assert opinion.expose(lo, -5.0, 1.0) == -1.0


def test_speak_emits_opinion_shift_and_updates_hearer(tmp_path):
    sim = _sim(tmp_path, "opsh", n=4)
    speaker, hearer = sim.agents[0], sim.agents[1]
    for other in sim.agents[2:]:                # 余計な聞き手を排除
        other.loc = "outside"
    _free_at(speaker, speaker.node)
    _free_at(hearer, speaker.node)
    speaker.x, speaker.y = 50.0, 50.0
    hearer.x, hearer.y = 52.0, 52.0
    hearer.opinion = 0.0                        # 制御値: Δ = s·w·v が厳密に出る
    hearer.opinion_anchor = 0.0
    hearer.opinion_susceptibility = 0.5

    text = "今日は最高に楽しい"
    v = valence(text)
    assert v > 0.0
    scheduler._apply(sim, speaker,
                     {"type": "speak", "text": text, "use_items": []}, 0, 0)

    shifts = [e for e in sim.logger.events
              if e.kind == "opinion_shift" and e.agent_id == hearer.id]
    assert shifts, "opinion_shift が出ていない"
    assert shifts[0].payload["source"] == speaker.id
    expected = 0.5 * sim.opinioncfg["w_face"] * v
    assert abs(hearer.opinion - expected) < 1e-9


def test_c_opinion_moved_attributed_to_source():
    events = [
        _ev(0, 1, "opinion_shift", source=0, old=0.0, new=0.2),
        _ev(1, 2, "opinion_shift", source=0, old=0.1, new=-0.1),
        _ev(2, 3, "opinion_shift", source=1, old=0.0, new=0.05),
    ]
    meta = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
    feats = {f["id"]: f for f in measure.agent_features(events, meta)}
    assert abs(feats[0]["c_opinion_moved"] - 0.4) < 1e-9   # |0.2|+|0.2|
    assert abs(feats[1]["c_opinion_moved"] - 0.05) < 1e-9
    assert feats[3]["c_opinion_moved"] == 0
    # c_opinion_moved は Y_external に混ざらない
    for f in feats.values():
        assert f["Y_external"] == 0


def test_y_external_unchanged_by_opinion(tmp_path):
    """既定ラン(mock 10人×48step): Y_external は従来4成分の和のまま。"""
    cfg = load_config(["run.seed=5", "run.n_agents=10", "run.n_steps=48",
                       "run.name=yext"])
    sim = Simulation(cfg, out_dir=tmp_path / "yext")
    sim.run()
    events = measure.load_events(str(sim.out_dir))
    meta = measure.load_agents(str(sim.out_dir))
    feats = measure.agent_features(events, meta)
    assert feats
    for row in feats:
        assert row["Y_external"] == (row["c_transmission"] + row["c_label_adopt"]
                                     + row["c_new_relations"] + row["c_sns_reach"])
    # 新列は存在するが Y_external とは独立に並ぶだけ
    assert all("c_opinion_moved" in r for r in feats)
    assert all("sns_likes_received" in r for r in feats)
    # 実際に意見更新が起きている(mock ランで opinion_shift が出る)
    assert any(e["kind"] == "opinion_shift" for e in events)
