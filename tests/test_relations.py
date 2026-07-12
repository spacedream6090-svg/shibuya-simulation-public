"""社会関係の質(現実ギャップ Wave G2 2026-07-07)= tier / decay / reputation / 派閥 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): closeness/tier フィールドを一切足さず、relation_tier/relation_break/
  reputation_update が 0 件・イベント列は純粋既定と L1 完全一致(ゴールデンを守る)。
- tier: ポジ交流の蓄積で 知人→友人→親友 へ上がり relation_tier が出る。
- decay: ネガ交流・長期不在で weight が下がり tier が落ちて relation_break が出る。
- reputation: 語の採用・被傾聴で reputation_update が出る(日次で風化)。
- プロンプト: tier がある相手に「友人」等が載る。OFF では従来の「N回話した仲」。
- 決定論: ON 同士2回で L1 完全一致。R1 呼数不変(FixedLLM で ON==OFF)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import relations
from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 20, steps: int = 1, *,
         rel: bool | None = None, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if rel is not None:
        dot.append(f"relations.enabled={'true' if rel else 'false'}")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind: str):
    return [e for e in sim.logger.events if e.kind == kind]


def _fresh_pair(sim):
    """関係台帳が空の 2 体を返す(初期の顔なじみ関係を除去してから始める)。"""
    a, b = sim.agents[0], sim.agents[1]
    a.mem.relations.pop(b.id, None)
    b.mem.relations.pop(a.id, None)
    return a, b


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定(config.yaml の relations.enabled=false)が L1 完全一致(144 step)。

    OFF 時は closeness/tier フィールドが付かず、G2 の新イベントが 1 件も出ない(seam が no-op)。
    """
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, rel=False)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G2 seam が no-op でない)"
    for s in (pure, off):
        for k in ("relation_tier", "relation_break", "reputation_update"):
            assert not _kind(s, k), f"OFF で {k} が出ている"
        for a in s.agents:
            for r in a.mem.relations.values():
                assert "closeness" not in r and "tier" not in r, \
                    "OFF で closeness/tier フィールドが台帳に付いている"


# --------------------------------------------------------------------- tier
def test_tier_of_pure_function():
    """tier_of: closeness 閾値で 0..3 を決定論算出(純関数)。"""
    cfg = relations.build_cfg({"enabled": True})     # 既定: 2 / 5 / 12
    assert relations.tier_of(0.0, cfg) == 0
    assert relations.tier_of(1.9, cfg) == 0
    assert relations.tier_of(2.0, cfg) == 1
    assert relations.tier_of(4.9, cfg) == 1
    assert relations.tier_of(5.0, cfg) == 2
    assert relations.tier_of(12.0, cfg) == 3
    assert relations.tier_of(-3.0, cfg) == 0


def test_tier_progression_emits_relation_tier(tmp_path):
    """ポジ交流の蓄積で closeness が伸び、知人(1)→友人(2)→親友(3)へ上がり relation_tier が出る。"""
    sim = _sim(tmp_path, "tier", n=10, rel=True)
    cfg = sim.relationscfg
    a, b = _fresh_pair(sim)
    for i in range(12):                              # 12 回のポジ交流 = closeness 12 = 親友
        relations.note_contact(a, b.id, b.name, "いいね", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger)
    rel = a.mem.relations[b.id]
    assert rel["closeness"] == 12.0 and rel["tier"] == 3
    reached = {e.payload["tier"] for e in _kind(sim, "relation_tier")
               if e.payload["other"] == b.id}
    assert {1, 2, 3} <= reached, f"知人/友人/親友の relation_tier が揃っていない: {reached}"


# --------------------------------------------------------------------- decay / break
def test_decay_absence_lowers_tier_and_breaks(tmp_path):
    """長期不在(前日以前が最終接触)で日次減衰 → tier が落ち relation_break(cause=absence)。"""
    sim = _sim(tmp_path, "decay", n=10, rel=True)
    cfg = sim.relationscfg
    a, b = _fresh_pair(sim)
    for i in range(5):                               # closeness 5 = 友人(tier 2)
        relations.note_contact(a, b.id, b.name, "", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger)
    assert a.mem.relations[b.id]["tier"] == 2
    sim._rel_day = -1
    scheduler._phase_relations_day(sim, step=200, sim_min=1440)   # day 1 の境界
    rel = a.mem.relations[b.id]
    assert rel["closeness"] == 4.0 and rel["tier"] == 1, "不在減衰で tier が落ちていない"
    brk = [e for e in _kind(sim, "relation_break") if e.payload["other"] == b.id]
    assert brk and brk[-1].payload["from_tier"] == 2 and brk[-1].payload["to_tier"] == 1
    assert brk[-1].payload["cause"] == "absence"


def test_negative_contact_breaks_relation(tmp_path):
    """ネガ交流(−neg_weight)で closeness が下がり tier が落ちて relation_break(cause=contact)。"""
    sim = _sim(tmp_path, "neg", n=10, rel=True)
    cfg = sim.relationscfg
    a, b = _fresh_pair(sim)
    for i in range(5):                               # closeness 5 = 友人(tier 2)
        relations.note_contact(a, b.id, b.name, "", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger)
    assert a.mem.relations[b.id]["tier"] == 2
    relations.note_contact(a, b.id, b.name, "最悪だ", -1.0, cfg,   # −2 → closeness 3 = 知人
                           step=6, sim_min=60, logger=sim.logger)
    rel = a.mem.relations[b.id]
    assert rel["closeness"] == 3.0 and rel["tier"] == 1
    brk = [e for e in _kind(sim, "relation_break")
           if e.payload["other"] == b.id and e.payload["cause"] == "contact"]
    assert brk and brk[-1].payload["from_tier"] == 2 and brk[-1].payload["to_tier"] == 1


# --------------------------------------------------------------------- reputation
def test_reputation_update_and_decay(tmp_path):
    """語の採用・被傾聴で reputation_update(スコア加算)、日次で風化(decay)。"""
    sim = _sim(tmp_path, "rep", n=10, rel=True)
    cfg = sim.relationscfg
    a = sim.agents[0]
    relations.gain_reputation(a, cfg["rep_adopted"], "own_adopted", 0, 0, sim.logger)
    relations.gain_reputation(a, cfg["rep_mention"], "mention", 1, 10, sim.logger)
    ups = _kind(sim, "reputation_update")
    assert len(ups) == 2
    assert ups[0].payload["cause"] == "own_adopted" and ups[0].payload["new"] == 1.0
    assert ups[1].payload["cause"] == "mention"
    assert abs(getattr(a, "_reputation") - 1.2) < 1e-9
    relations.reputation_decay(a, cfg, 2, 20, sim.logger)     # −rep_decay_per_day(0.5)
    assert abs(a._reputation - 0.7) < 1e-9
    assert _kind(sim, "reputation_update")[-1].payload["cause"] == "decay"


def test_reputation_wired_to_own_adopted(tmp_path):
    """配線: _hear_words で自分の造語が採用されると creator に reputation_update が出る。"""
    sim = _sim(tmp_path, "repwire", n=10, rel=True)
    creator, listener = sim.agents[0], sim.agents[1]
    # creator の造語を item として登録し、listener を採用直前(閾値-1 回既聴)に置く
    item = sim.labels.coin(creator, "ズレ语", step=0, sim_min=0, logger=sim.logger,
                           context={})
    assert item is not None
    word = item.text
    # adopt_threshold 回 hear させて listener に採用させる(最後の hear で own_adopted 発火)
    thr = sim.labels.adopt_threshold
    for s in range(thr):
        scheduler._hear_words(sim, listener, [word], creator.id, "face",
                              step=s + 1, sim_min=(s + 1) * 10)
    assert word in listener.adopted, "listener が語を採用していない(閾値設定を確認)"
    reps = [e for e in _kind(sim, "reputation_update")
            if e.agent_id == creator.id and e.payload["cause"] == "own_adopted"]
    assert reps, "造語採用で creator の reputation_update が出ていない"


# --------------------------------------------------------------------- プロンプト
def test_prompt_shows_tier_and_off_shows_count(tmp_path):
    """tier がある相手には「友人」等が載る。relation_line=None(OFF 相当)は従来の「N回話した仲」。"""
    sim = _sim(tmp_path, "prompt", n=10, rel=True)
    cfg = sim.relationscfg
    a, b = _fresh_pair(sim)
    for i in range(5):                               # 友人(tier 2)
        relations.note_contact(a, b.id, b.name, "", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger)
    rline, repline = relations.social_lines(a, [b.id], cfg, tools=sim.tools)
    assert rline is not None and f"{b.name}とは友人" in rline
    prompt = deliberate.build_prompt(a, place_name="路上", surprise="solo",
                                     nearby_names=[b.name], nearby_ids=[b.id],
                                     sim_min=600, step=3,
                                     relation_line=rline, reputation_line=repline)
    assert "友人" in prompt, "tier がプロンプトに載っていない"

    # OFF 相当: relation_line 未指定 → build_prompt は従来の mem.relation_line に後退
    c, d = sim.agents[2], sim.agents[3]
    c.mem.relations.pop(d.id, None)
    c.mem.record_contact(d.id, d.name, 0)
    c.mem.record_contact(d.id, d.name, 1)            # count=2(closeness なし)
    p_off = deliberate.build_prompt(c, place_name="路上", surprise="solo",
                                    nearby_names=[d.name], nearby_ids=[d.id],
                                    sim_min=600, step=3)
    assert f"{d.name}とは2回話した仲" in p_off and "友人" not in p_off


def test_prompt_reputation_line(tmp_path):
    """評判が閾値以上のとき「街で名が知られている」1行が載る(未満では載らない)。"""
    sim = _sim(tmp_path, "prep", n=10, rel=True)
    cfg = sim.relationscfg
    a = sim.agents[0]
    _, rep_lo = relations.social_lines(a, [], cfg, tools=sim.tools)
    assert rep_lo is None, "評判 0 なのに評判行が出ている"
    a._reputation = cfg["rep_known_threshold"]       # 閾値ちょうど
    _, rep_hi = relations.social_lines(a, [], cfg, tools=sim.tools)
    assert rep_hi is not None and "知られている" in rep_hi


# --------------------------------------------------------------------- 決定論 / R1
def test_on_deterministic_and_events(tmp_path):
    """relations ON 同士2回で L1 完全一致。ON で新イベントが実際に出る(配線が効く)。"""
    a = _sim(tmp_path, "on_a", n=30, steps=144, rel=True)
    a.run()
    b = _sim(tmp_path, "on_b", n=30, steps=144, rel=True)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    assert _kind(a, "relation_tier"), "ON で relation_tier が出ていない"
    assert _kind(a, "reputation_update"), "ON で reputation_update が出ていない"


class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, *, rel, n=30, steps=144):
    sim = _sim(tmp_path, name, n=n, steps=steps, rel=rel)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend なら ON/OFF で generate 呼数が完全一致(R1: 非LLM機構=呼数不変)。"""
    on = _run_fixed(tmp_path, "cc_on", rel=True)
    off = _run_fixed(tmp_path, "cc_off", rel=False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    # ON は非LLM 機構で新イベントを出す / OFF は 1 件も出さない(配線と OFF no-op の両確認)
    assert _kind(on, "reputation_update"), "ON で reputation_update が出ていない(配線未接続)"
    for k in ("relation_tier", "relation_break", "reputation_update"):
        assert not _kind(off, k), f"OFF で {k} が出ている"
