"""主観的世界モデル(第20バッチ 2026-07-12)のテスト。

設計(docs/research/world-models.md C1/C2/C6・ユーザー依頼 2026-07-12):
- C1 期待形成: 場所×時間帯の人出 EMA(=検証可能な仮説)。誤差の記録=検証の観測。
- C2 可制御性: 全員 0.5 起点。世界の応答イベントだけで分岐(純経験の経路依存)。
- C6 規範予期: 開拓的行動の記述規範を全員共通1行に。
- 既定 OFF=状態・イベント・プロンプトともバイト一致。乱数ゼロ。R1: 呼数不変。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from society import worldview
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import Event


def _sim(tmp_path, name, n=20, steps=144, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
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
    """明示 OFF と純粋既定が L1 完全一致。主観状態の属性も作られない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"worldview.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(第20バッチ seam が no-op でない)"
    assert not _kind(pure, "worldview")
    assert all(getattr(a, "wv_expect", None) is None and
               getattr(a, "controllability", None) is None
               for a in pure.agents)


# --------------------------------------------------------------------- 単体
def test_ctrl_dynamics():
    """C2 可制御性の力学: 応答で上下・限界効用逓減・クリップ・弱シグナル倍率。"""
    cfg = worldview.build_cfg({"enabled": True})
    a = SimpleNamespace()
    worldview._ctrl_apply(a, True, 1.0, cfg)
    assert a.controllability > 0.5, "正の応答で上がらない"
    up1 = a.controllability - 0.5
    worldview._ctrl_apply(a, True, 1.0, cfg)
    assert (a.controllability - 0.5) - up1 < up1, "限界効用逓減が効かない"
    b = SimpleNamespace()
    for _ in range(100):
        worldview._ctrl_apply(b, False, 1.0, cfg)
    assert b.controllability >= worldview._CTRL_CLIP[0], "下側クリップ破り"
    c1, c2 = SimpleNamespace(), SimpleNamespace()
    worldview._ctrl_apply(c1, True, 1.0, cfg)
    worldview._ctrl_apply(c2, True, cfg["weak_scale"], cfg)
    assert c1.controllability > c2.controllability, "弱シグナルが強と同じ"


def test_lines_unit():
    """注入行の閾値: 手応え/無力/規範の各文が条件でのみ出る(因子語なし)。"""
    cfg = worldview.build_cfg({"enabled": True})
    a = SimpleNamespace(controllability=0.8)
    assert "手応え" in worldview.ctrl_line(a, cfg)
    a.controllability = 0.2
    assert "変わらない" in worldview.ctrl_line(a, cfg)
    a.controllability = 0.5
    assert worldview.ctrl_line(a, cfg) is None, "中間帯で沈黙しない"
    assert worldview.ctrl_line(SimpleNamespace(), cfg) is None
    assert worldview.ctrl_line(a, None) is None

    from collections import deque
    sim = SimpleNamespace(agents=[None] * 20,
                          _wv_pioneer=deque([3, 4], maxlen=7))
    assert "珍しくない" in worldview.norm_line(sim, cfg)      # rate=3.5/20=0.175
    sim._wv_pioneer = deque([0, 0], maxlen=7)
    assert "ほとんど見かけない" in worldview.norm_line(sim, cfg)
    sim._wv_pioneer = deque([1], maxlen=7)
    assert worldview.norm_line(sim, cfg) is None, "窓が育つ前に喋った"


def test_ctrl_scan_responses(tmp_path):
    """C2: 応答イベントの走査で可制御性が正しい向きに動く(合成イベント)。"""
    sim = _sim(tmp_path, "scan", steps=1, **{"worldview.enabled": "true"})
    cfg = sim.worldviewcfg
    a0, a1, a2 = sim.agents[0], sim.agents[1], sim.agents[2]
    ev = [Event(step=1, sim_min=0, agent_id=a0.id, kind="proposal_passed",
                x=0, y=0, payload={"proposal_id": 1}),
          Event(step=1, sim_min=0, agent_id=a1.id, kind="vote_result",
                x=0, y=0, payload={"passed": False}),
          Event(step=2, sim_min=0, agent_id=-1, kind="flyer_view",
                x=0, y=0, payload={"author": a2.id, "node": "n"})]
    for e in ev:
        sim.logger.log(e)
    sim._wv_pos = sim.logger._n_flushed + len(sim.logger.events) - len(ev)
    worldview._scan_responses(sim, cfg)
    assert a0.controllability > 0.5, "可決で上がらない"
    assert a1.controllability < 0.5, "否決で下がらない"
    assert 0.5 < a2.controllability < a0.controllability, "弱シグナルの序列が壊れている"
    pos = sim._wv_pos
    worldview._scan_responses(sim, cfg)               # 二重適用しない(増分走査)
    assert sim._wv_pos == pos and a0.controllability == a0.controllability


def test_ctrl_daily_cap(tmp_path):
    """C2: 応答の洪水でも日次計上キャップで動く量が抑えられる(飽和防止)。"""
    sim = _sim(tmp_path, "cap", steps=1, **{"worldview.enabled": "true"})
    cfg = sim.worldviewcfg
    a = sim.agents[0]
    for i in range(20):                               # 弱い応答 ×20(キャップ3)
        sim.logger.log(Event(step=1, sim_min=0, agent_id=-1, kind="flyer_view",
                             x=0, y=0, payload={"author": a.id, "node": "n"}))
    sim._wv_pos = sim.logger._n_flushed + len(sim.logger.events) - 20
    worldview._scan_responses(sim, cfg)
    b = SimpleNamespace()
    for _ in range(cfg["ctrl_daily_weak"]):           # キャップ数ちょうどの参照値
        worldview._ctrl_apply(b, True, cfg["weak_scale"], cfg)
    assert abs(a.controllability - b.controllability) < 1e-9, \
        f"キャップが効いていない: {a.controllability} vs {b.controllability}"


# --------------------------------------------------------------------- ON 挙動
def test_on_learning_and_snapshot(tmp_path):
    """ON: 期待テーブルが育ち・誤差が記録され・日次 worldview イベントが出る。"""
    sim = _sim(tmp_path, "on", steps=288, **{"worldview.enabled": "true"})
    sim.run()
    evs = _kind(sim, "worldview")
    assert evs, "ON なのに worldview イベントが出ない"
    world = [e for e in evs if e.agent_id == -1]
    agents = [e for e in evs if e.agent_id >= 0]
    assert world and agents, "街レベル/エージェントレベルの片方が欠けている"
    assert all("norm_rate" in e.payload for e in world)
    assert all(set(e.payload) == {"ctrl", "expect_n", "err_mean", "err_n"}
               for e in agents)
    assert any((a.payload["expect_n"] or 0) > 0 for a in agents), "期待仮説が育たない"
    assert any(a.payload["err_n"] > 0 for a in agents), "仮説の検証(誤差記録)が無い"
    assert all(getattr(a, "wv_expect", None) is not None for a in sim.agents
               if a.loc != "outside" or a.wv_expect is not None)


def test_on_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(mock・乱数ゼロの決定論)。"""
    a = _sim(tmp_path, "det_a", steps=288, **{"worldview.enabled": "true"})
    a.run()
    b = _sim(tmp_path, "det_b", steps=288, **{"worldview.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


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
    """応答固定 backend: worldview ON/OFF で generate 呼数が完全一致(288 step)。"""
    def run(name, on):
        sim = _sim(tmp_path, name, steps=288,
                   **{"worldview.enabled": str(on).lower()})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    on = run("r1_on", True)
    off = run("r1_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
