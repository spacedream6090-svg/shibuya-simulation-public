"""engaged モード(第87バッチ・cognition.engaged)のテスト。

正典: docs/plans/source/design-discussion-20260802.md §8 /
      docs/plans/dayplan-engaged-plan.md 第87。

守るもの(検収基準の順)
  (1) 既定 OFF = 純粋既定と L1 バイト一致・プロンプト不変・新 kind ゼロ件・L2 に列なし・
      manifest にキーなし・state 不在・LLM 呼数不変・**新 stream を 1 本も引かない**
  (2) 状態機械の芯(純関数): ヒステリシスで**発振しない**・θ_out<θ_in・最短滞在
  (3) 突入 5 条件 / 脱出 4 条件がそれぞれ効く
  (4) 補助規則: 不応期・両者 ENGAGED 会話成立(片方 AUTOPILOT なら不成立)・
      テンプレ応答は LLM 呼ゼロ・ターン上限の切り上げ・プリエンプトの兆しメモリ 1 行
  (5) ON ラン: 同 seed 2 ラン L1 一致・resume==straight・k 不変・呼び出しサイトを増やさない
  (6) no-fingerprint: 機構語・実験条件語・因子名がプロンプト全文に出現しない
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society.cognition import deliberate
from society.cognition import engaged as EG
from society.cognition import fire as FIRE
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

FIRE_ON = {"cognition.fire.enabled": "true"}
ON = {**FIRE_ON, "cognition.engaged.enabled": "true"}
NEW_KINDS = {"episode_start", "episode_end", "episode_closing",
             "engaged_template", "sign_memory"}
L2_COLS = ("engaged_episodes_total", "engaged_turns_total",
           "engaged_stay_min_total", "engaged_turns_mean",
           "engaged_template_total")


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _spy(sim) -> list:
    seen: list = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    return seen


class _CountingHub:
    """stream 派生を数えるプロキシ(test_fire と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


_C = EG.build_cfg({"enabled": True})


class _Agent:
    """状態機械の単体検証用のごく薄い代役(sim も世界も要らない検査のため)。"""

    def __init__(self, aid=1, name="X"):
        self.id = aid
        self.name = name
        self.x = self.y = 0.0
        self._reply_to = None
        self._engaged = None
        self._engaged_refr = None


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.cognition.engaged.enabled) is False
    assert EG.build_cfg(None)["enabled"] is False


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。state 不在・L2 に列なし・manifest 無風。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **{"cognition.engaged.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(engaged seam が no-op でない)"
    assert getattr(off, "_engaged_state", None) is None, "OFF なのに _engaged_state が生えた"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    for c in L2_COLS:
        assert c not in cols, f"OFF なのに L2 に {c} 列"


def test_off_emits_no_new_event_kinds_and_no_prompt_section(tmp_path):
    sim = _sim(tmp_path, "off_prompt")
    seen = _spy(sim)
    sim.run()
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]
    assert EG.PROMPT_MARK not in "\n".join(seen), \
        "OFF なのに終結の宣言路がプロンプトに載った"


def test_off_llm_call_count_unchanged(tmp_path):
    a = _sim(tmp_path, "cc_pure")
    a.run()
    b = _sim(tmp_path, "cc_off", **{"cognition.engaged.enabled": "false"})
    b.run()
    assert a.llm.calls == b.llm.calls > 0


def test_off_draws_no_new_stream(tmp_path):
    """既定 OFF では engaged 由来の stream("mock_end")を 1 本も引かない。"""
    sim = _sim(tmp_path, "off_stream")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert "mock_end" not in sim.hub.counts


def test_requires_fire_enabled(tmp_path):
    """cognition.fire が OFF のまま engaged ON にしても完全 no-op(fire ON が前提)。"""
    base = _sim(tmp_path, "nf_base")
    base.run()
    on = _sim(tmp_path, "nf_on", **{"cognition.engaged.enabled": "true"})
    on.run()
    assert not EG.enabled(on), "fire OFF なのに engaged が実効になっている"
    assert _l1(base) == _l1(on)
    assert getattr(on, "_engaged_state", None) is None


def test_event_kinds_and_registry_declared():
    for k in NEW_KINDS:
        assert k in EVENT_KINDS, f"{k} が EVENT_KINDS に未登録"
    feat = {f.id: f for f in R.FEATURES}["cognition.engaged.enabled"]
    assert feat.repro_tier == "journal"
    assert feat.affects_k is True, "affects_k の宣言(エピソードが発火を束ねる)が落ちた"
    assert feat.fingerprint_risk == "possible", "終結の宣言路がプロンプトに出ることの宣言"


def test_no_undeclared_toggles():
    assert R.undeclared_toggles(load_config()) == []


# --------------------------------------------------------------------------- #
# (B) 状態機械の芯 = ヒステリシス(検収基準 2)
# --------------------------------------------------------------------------- #
def test_theta_out_is_strictly_below_theta_in():
    assert EG.theta_out(1.0, _C) == 0.5
    # 比が 1 以上に設定されてもクランプされる(ヒステリシスの成立条件を構造で強制)
    bad = EG.build_cfg({"enabled": True, "theta_out_ratio": 5.0})
    assert bad["theta_out_ratio"] < 1.0
    assert EG.theta_out(1.0, bad) < 1.0


def test_hysteresis_holds_state_inside_the_band():
    """θ_out < S < θ_in の帯では**状態が変わらない**(Schmitt トリガの意味論)。"""
    theta = 1.0
    band = 0.75                                    # θ_out=0.5 < 0.75 < θ_in=1.0
    # 帯から始めると入らない
    assert EG.simulate([band] * 5, theta, _C) == [EG.AUTOPILOT] * 5
    # 一度超えて帯へ落ちても抜けない
    states = EG.simulate([1.5] + [band] * 5, theta, _C)
    assert states == [EG.ENGAGED] * 6


def test_no_oscillation_at_the_boundary():
    """閾値ちょうどで S が振動しても、突入/脱出が交互に起きない(発振の直接検査)。"""
    theta = 1.0
    series = [0.1] + [1.01, 0.99] * 20             # θ_in をまたいで毎 tick 振動
    states = EG.simulate(series, theta, _C)
    flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert flips == 1, f"境界で発振している(遷移 {flips} 回): {states[:8]}"
    assert states[-1] == EG.ENGAGED
    # ヒステリシスを外す(ratio=0.999…)と同じ系列で**発振する**= 検査自体が効いている
    flat = EG.build_cfg({"enabled": True, "theta_out_ratio": 0.999,
                         "min_stay_min": 0})
    flat_states = EG.simulate(series, theta, flat)
    flat_flips = sum(1 for a, b in zip(flat_states, flat_states[1:]) if a != b)
    assert flat_flips > 10, "対照(ヒステリシスなし)が発振しない=検査が無意味"


def test_min_stay_blocks_immediate_exit():
    """最短滞在の間は減衰で抜けない(O'Brien & Arkin 2020 の dithering 対策)。"""
    cfg = EG.build_cfg({"enabled": True, "min_stay_min": 30})
    assert EG.should_exit_decay(0.0, 1.0, cfg, dwell_min=0) is False
    assert EG.should_exit_decay(0.0, 1.0, cfg, dwell_min=20) is False
    assert EG.should_exit_decay(0.0, 1.0, cfg, dwell_min=30) is True
    assert EG.should_exit_decay(0.0, 1.0, cfg) is True          # 未指定=従来判定


def test_refractory_raises_entry_threshold():
    """不応期中は θ_in が refractory_mult 倍になる(= 同じ S では入れない)。"""
    assert EG.should_enter(1.5, 1.0, _C, refractory=False) is True
    assert EG.should_enter(1.5, 1.0, _C, refractory=True) is False   # θ_in→2.0
    assert EG.should_enter(2.5, 1.0, _C, refractory=True) is True
    assert EG.should_enter(1.5, 0.0, _C) is False                    # θ=0 は発火しない


def test_turn_cap_is_kind_specific():
    assert EG.turn_cap_of(EG.TALK, _C) == _C["turn_cap"] == 12
    assert EG.turn_cap_of(EG.REPLAN, _C) == _C["replan_cap"] == 3
    assert EG.turn_cap_of(EG.REFLECT, _C) == 3


def test_high_res_is_deterministic_and_draws_no_rng():
    """高解像度層の選定は _stable_hash の純関数(同じ id は常に同じ・割合が効く)。"""
    agents = [_Agent(i) for i in range(2000)]
    cfg0 = EG.build_cfg({"enabled": True, "reflect_frac": 0.0})
    cfg1 = EG.build_cfg({"enabled": True, "reflect_frac": 1.0})
    assert not any(EG.high_res(a, cfg0) for a in agents)
    assert all(EG.high_res(a, cfg1) for a in agents)
    cfg = EG.build_cfg({"enabled": True, "reflect_frac": 0.05})
    picked = [a.id for a in agents if EG.high_res(a, cfg)]
    assert picked == [a.id for a in agents if EG.high_res(a, cfg)]    # 決定論
    assert 0.02 < len(picked) / len(agents) < 0.09, len(picked) / len(agents)


# --------------------------------------------------------------------------- #
# (C) closing move(終結の宣言路)
# --------------------------------------------------------------------------- #
def test_parse_action_passes_end_flag_through():
    """`end` 欄は**あるときだけ**行動 dict に現れる(無ければ従来と完全同一)。"""
    plain = deliberate.parse_action('{"action":"speak","text":"やあ"}')
    assert plain == {"type": "speak", "text": "やあ", "use_items": []}
    assert "end" not in plain
    closed = deliberate.parse_action('{"action":"speak","text":"またね","end":true}')
    assert closed["end"] is True
    assert deliberate.parse_action(
        '{"action":"speak","text":"x","end":false}').get("end") is None
    # 別名も読む(実 LLM の揺れ)
    assert deliberate.parse_action(
        '{"action":"dm","text":"またね","end_conversation":true}')["end"] is True


def test_prompt_section_only_on_talk_turns(tmp_path):
    sim = _sim(tmp_path, "sec", n_steps=1, **ON)
    a = sim.agents[0]
    assert EG.prompt_section(sim, a, "reply") is None            # エピソード無し
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    assert EG.PROMPT_MARK in EG.prompt_section(sim, a, "reply")
    assert EG.PROMPT_MARK in EG.prompt_section(sim, a, "social")
    assert EG.prompt_section(sim, a, "post") is None             # 会話以外には出さない
    assert EG.prompt_section(sim, a, "solo") is None
    a._engaged["kind"] = EG.REPLAN
    assert EG.prompt_section(sim, a, "reply") is None             # 会話エピソード以外
    a._engaged["kind"] = EG.TALK
    a._engaged["wrapup"] = True
    assert "締めくくる" in EG.prompt_section(sim, a, "reply")


def test_closing_requires_both_sides(tmp_path):
    """片方の別れ挨拶だけでは解消しない(Schegloff & Sacks の terminal exchange)。"""
    sim = _sim(tmp_path, "close", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, b.id, "mock")
    b._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, a.id, "mock")
    EG.note_closing(sim, a, b, 0, 0)
    assert a._engaged["closed_self"] and not a._engaged["closed_other"]
    assert b._engaged["closed_other"] and not b._engaged["closed_self"]
    EG.update(sim, 0, 0, {}, sim.agents)                 # まだ双方揃っていない
    assert a._engaged is not None and b._engaged is not None
    EG.note_closing(sim, b, a, 0, 0)
    EG.update(sim, 0, 10, {}, sim.agents)
    assert a._engaged is None and b._engaged is None
    ends = [e for e in sim.logger.events if e.kind == "episode_end"]
    assert len(ends) == 2 and all(e.payload["exit"] == EG.X_RESOLVED for e in ends)


# --------------------------------------------------------------------------- #
# (D) 補助規則(検収基準 4)
# --------------------------------------------------------------------------- #
def test_familiarity_uses_closeness_then_contacts(tmp_path):
    sim = _sim(tmp_path, "fam", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    assert EG.familiarity_ok(a, b.id, sim.engagedcfg) is False    # 台帳なし=初対面
    a.mem.record_contact(b.id, b.name, 0)
    assert EG.familiarity_ok(a, b.id, sim.engagedcfg) is False    # count=1 < 2
    a.mem.record_contact(b.id, b.name, 1)
    assert EG.familiarity_ok(a, b.id, sim.engagedcfg) is True     # count=2
    # closeness があるとき(relations ON 相当)はそちらを見る
    a.mem.relations[b.id]["closeness"] = 0.1
    assert EG.familiarity_ok(a, b.id, sim.engagedcfg) is False
    a.mem.relations[b.id]["closeness"] = 5.0
    assert EG.familiarity_ok(a, b.id, sim.engagedcfg) is True


def test_template_reply_calls_no_llm(tmp_path):
    """関係の薄い相手への定型応答は LLM を 1 本も呼ばない(呼数が減る向きの seam)。"""
    sim = _sim(tmp_path, "tmpl", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a._reply_to = (b.id, "こんにちは")
    assert EG.reply_mode(sim, a) == "template"
    before = sim.llm.calls
    action = scheduler._decide(sim, a, 0, 0)
    assert action["type"] == "speak" and action["text"] == EG.TEMPLATE
    assert sim.llm.calls == before, "テンプレ応答なのに LLM を呼んだ"
    assert a._reply_to is None and EG.episode_of(a) is None
    assert _kind(sim, "engaged_template"), "engaged_template が記録されていない"
    # 親密な相手なら従来の LLM 返答経路(= "engage")
    a.mem.record_contact(b.id, b.name, 0)
    a.mem.record_contact(b.id, b.name, 1)
    a._reply_to = (b.id, "こんにちは")
    assert EG.reply_mode(sim, a) == "engage"


def test_handoff_requires_speaker_engaged(tmp_path):
    """会話は両者 ENGAGED が成立条件: 話者が AUTOPILOT なら返答権を渡さない。"""
    sim = _sim(tmp_path, "hand", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    assert EG.handoff_ok(sim, a, b) is False                      # AUTOPILOT
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_NEED, 0, 0, None, "mock")
    assert EG.handoff_ok(sim, a, b) is False                      # 会話ではない
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    assert EG.handoff_ok(sim, a, b) is True
    off = _sim(sim.out_dir.parent, "hand_off", n_steps=1)
    assert EG.handoff_ok(off, off.agents[0], off.agents[1]) is True   # OFF は常に True


def test_turn_cap_inserts_one_wrapup_turn(tmp_path):
    """会話はターン上限で**切り上げターンを 1 回挟んで**から終わる(§8 脱出 3)。"""
    sim = _sim(tmp_path, "cap", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    a._engaged["turns"] = sim.engagedcfg["turn_cap"]
    EG.update(sim, 0, 10, {}, sim.agents)
    assert a._engaged is not None and a._engaged["wrapup"] is True
    assert "締めくくる" in EG.prompt_section(sim, a, "reply")
    EG.update(sim, 1, 20, {}, sim.agents)                 # 切り上げターンを撃ち終えた
    assert a._engaged is None
    end = _kind(sim, "episode_end")[-1]
    assert end.payload["exit"] == EG.X_TURN_CAP


def test_thinking_episode_has_no_wrapup(tmp_path):
    """思考系(再計画/内省)は replan_cap で即終了(会話の礼儀は要らない)。"""
    sim = _sim(tmp_path, "cap2", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_NEED, 0, 0, None, "mock")
    a._engaged["turns"] = sim.engagedcfg["replan_cap"]
    EG.update(sim, 0, 10, {}, sim.agents)
    assert a._engaged is None
    assert _kind(sim, "episode_end")[-1].payload["exit"] == EG.X_TURN_CAP


def test_preempt_writes_one_line_sign_memory(tmp_path):
    """プリエンプトは中断内容を**1 行だけ**兆しメモリへ書き、割り込み側へ席を譲る(§8 脱出 4)。"""
    sim = _sim(tmp_path, "pre", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, b.id, "mock")
    a._engaged["turns"] = 2
    before = len(a.mem.buffer)
    EG.update(sim, 0, 30, {a.id: {"reason": FIRE.INTERNAL}}, sim.agents)
    end = _kind(sim, "episode_end")[-1]
    assert end.payload["exit"] == EG.X_PREEMPT and end.payload["kind"] == EG.TALK
    assert len(a.mem.buffer) == before + 1, "兆しメモリが 1 行でない"
    ep = a.mem.buffer[-1]
    assert ep.kind == "sign" and b.name in ep.text and "あとで考えよう" in ep.text
    assert _kind(sim, "sign_memory")
    # **割り込み**なので、同じ tick のうちに割り込み側(欲求臨界)のエピソードへ移る
    new = EG.episode_of(a)
    assert new and new["kind"] == EG.REPLAN and new["trigger"] == EG.T_NEED


def test_preempt_can_be_switched_off(tmp_path):
    sim = _sim(tmp_path, "pre_off", n_steps=1, **ON,
               **{"cognition.engaged.sign_memory": "false"})
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.REFLECT, EG.T_SCHEDULED, 0, 0, None, "mock")
    before = len(a.mem.buffer)
    EG.update(sim, 0, 30, {a.id: {"reason": FIRE.INTERNAL}}, sim.agents)
    assert _kind(sim, "episode_end")[-1].payload["exit"] == EG.X_PREEMPT
    assert len(a.mem.buffer) == before, "sign_memory=false なのに記憶へ書いた"
    assert not _kind(sim, "sign_memory")


def test_replan_episode_is_not_preempted_by_its_own_trigger(tmp_path):
    """再計画中は「割り込まれる側」ではない(すでに同じ用件の中にいる)。"""
    sim = _sim(tmp_path, "pre2", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_EXCEPTION, 0, 0, None, "mock")
    a._reply_to = (sim.agents[1].id, "…")           # 減衰も止めて他要因を排除
    EG.update(sim, 0, 30, {a.id: {"reason": FIRE.INTERNAL}}, sim.agents)
    assert a._engaged is not None


def test_refractory_is_set_on_exit_and_blocks_reentry(tmp_path):
    sim = _sim(tmp_path, "refr", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_SALIENCE, 0, 0, None, "mock")
    EG._end(sim, a, 0, 100, EG.X_DECAY)
    assert a._engaged_refr == {"until": 130, "trigger": EG.T_SALIENCE}
    assert EG._refractory(a, EG.T_SALIENCE, 120) is True
    assert EG._refractory(a, EG.T_SALIENCE, 130) is False   # 期限は開区間
    assert EG._refractory(a, EG.T_NEED, 120) is False       # **同種刺激**だけ


def test_resolved_only_when_plan_validates(tmp_path):
    """再計画の解消は「検証を通る計画」のときだけ(後退した計画は解消ではない)。"""
    sim = _sim(tmp_path, "res", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_EXCEPTION, 0, 0, None, "mock")
    EG.note_resolved(sim, a)
    assert a._engaged["resolved"] is True
    EG.update(sim, 0, 10, {}, sim.agents)
    assert a._engaged is None
    assert _kind(sim, "episode_end")[-1].payload["exit"] == EG.X_RESOLVED
    # 会話エピソードには効かない(会話の解消は双方の別れ挨拶だけ)
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    EG.note_resolved(sim, a)
    assert "resolved" not in a._engaged


def test_note_turn_counts_only_inside_an_episode(tmp_path):
    sim = _sim(tmp_path, "turn", n_steps=1, **ON)
    a = sim.agents[0]
    EG.note_turn(sim, a, 0, 0)                       # エピソード無し=no-op
    assert getattr(sim, "_engaged_state", None) is None
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    EG.note_turn(sim, a, 0, 30)
    assert a._engaged["turns"] == 1 and a._engaged["last_turn_min"] == 30
    assert sim._engaged_state["turns"] == 1


# --------------------------------------------------------------------------- #
# (E) 突入 5 条件(検収基準 3)
# --------------------------------------------------------------------------- #
def test_entry_social_addressed_bypasses_salience(tmp_path):
    """話しかけられたら S 計算をバイパスして即突入(S=0・θ 未達でも入る)。"""
    sim = _sim(tmp_path, "e_soc", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a.mem.record_contact(b.id, b.name, 0)
    a.mem.record_contact(b.id, b.name, 1)
    a._reply_to = (b.id, "やあ")
    sim._fire_s = {a.id: (0.0, [])}
    EG.update(sim, 0, 0, {}, [a])
    ep = EG.episode_of(a)
    assert ep and ep["kind"] == EG.TALK and ep["trigger"] == EG.T_SOCIAL
    assert ep["partner"] == b.id


def test_entry_social_declined_for_thin_relations(tmp_path):
    """関係の薄い定型接触では突入しない(1 ターンのテンプレで流す)。"""
    sim = _sim(tmp_path, "e_soc2", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a._reply_to = (b.id, "やあ")
    EG.update(sim, 0, 0, {}, [a])
    assert EG.episode_of(a) is None


def test_entry_plan_exception(tmp_path):
    sim = _sim(tmp_path, "e_exc", n_steps=1, **ON)
    a = sim.agents[0]
    a._plan_exception = 60
    EG.update(sim, 0, 60, {}, [a])
    ep = EG.episode_of(a)
    assert ep and ep["kind"] == EG.REPLAN and ep["trigger"] == EG.T_EXCEPTION


def test_entry_need_and_salience_and_scheduled(tmp_path):
    sim = _sim(tmp_path, "e_rest", n_steps=1, **ON)
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    sim._fire_s = {b.id: (99.0, [])}
    sim._fire_plan_due = (c.id,)
    EG.update(sim, 0, 0, {a.id: {"reason": FIRE.INTERNAL},
                          b.id: {"reason": FIRE.SALIENCE}}, [a, b, c])
    assert EG.episode_of(a)["trigger"] == EG.T_NEED
    assert EG.episode_of(b)["trigger"] == EG.T_SALIENCE
    ep_c = EG.episode_of(c)
    assert ep_c["trigger"] == EG.T_SCHEDULED and ep_c["kind"] == EG.REPLAN
    # 予定思考は「既にこの tick に 1 回考えている」ので turns=1 から始まる
    assert ep_c["turns"] == 1


def test_entry_salience_needs_to_exceed_theta_in(tmp_path):
    """salience 発火でも S が θ_in を超えていなければエピソードにはしない。"""
    sim = _sim(tmp_path, "e_sal", n_steps=1, **ON)
    a = sim.agents[0]
    sim._fire_s = {a.id: (0.0, [])}
    EG.update(sim, 0, 0, {a.id: {"reason": FIRE.SALIENCE}}, [a])
    assert EG.episode_of(a) is None


def test_entry_reflect_only_for_high_res_layer(tmp_path):
    """夜の内省がエピソードになるのは高解像度層だけ(§8 突入 5)。"""
    sim = _sim(tmp_path, "e_ref", n_steps=1, n_agents=40, **ON,
               **{"cognition.engaged.reflect_frac": "1.0"})
    for a in sim.agents:
        a.reflect_step = 0
    EG.update(sim, 0, 0, {}, sim.agents)
    assert all(EG.episode_of(a)["kind"] == EG.REFLECT for a in sim.agents)
    zero = _sim(tmp_path, "e_ref0", n_steps=1, n_agents=40, **ON,
                **{"cognition.engaged.reflect_frac": "0.0"})
    for a in zero.agents:
        a.reflect_step = 0
    EG.update(zero, 0, 0, {}, zero.agents)
    assert all(EG.episode_of(a) is None for a in zero.agents)


def test_pre_tick_advances_only_live_episodes(tmp_path):
    """生きているエピソードだけがキューの「今」へ繰り上がる(点 → 区間の実体)。"""
    sim = _sim(tmp_path, "pretick", n_steps=1, **ON)
    a, b = sim.agents[0], sim.agents[1]
    sim.cogq.schedule(a.id, 999, FIRE.PERIODIC)
    sim.cogq.schedule(b.id, 999, FIRE.PERIODIC)
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    EG.pre_tick(sim, 0, 100)
    assert sim.cogq.pending[a.id]["at"] == 100
    assert sim.cogq.pending[a.id]["reason"] == EG.CONTINUE
    assert sim.cogq.pending[b.id]["at"] == 999


# --------------------------------------------------------------------------- #
# (F) ON ラン(検収基準 5)
# --------------------------------------------------------------------------- #
def test_on_run_emits_episodes_and_l2_matches_state(tmp_path):
    sim = _sim(tmp_path, "on_run", n_steps=144, n_agents=40, **ON)
    summary = sim.run()
    starts, ends = _kind(sim, "episode_start"), _kind(sim, "episode_end")
    assert starts and ends and len(starts) >= len(ends)
    st = sim._engaged_state
    assert st["episodes"] == len(starts)
    for e in ends:
        assert e.payload["exit"] in EG.EXITS
        assert e.payload["kind"] in EG.KINDS
        assert e.payload["model"] == "mock"        # 第88 までは backend 名
    rows = pq.read_table(tmp_path / "on_run" / "l2_metrics.parquet").to_pylist()
    assert rows[-1]["engaged_episodes_total"] == float(st["episodes"])
    assert rows[-1]["engaged_turns_total"] == float(st["turns"])
    prov = summary["engaged"]
    assert prov["episodes"] == st["episodes"]
    assert sum(prov["by_trigger"].values()) == st["episodes"]
    assert set(prov["by_trigger"]) <= set(EG.TRIGGERS)
    assert set(prov["by_exit"]) <= set(EG.EXITS)


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "det_a", n_steps=144, n_agents=40, **ON)
    a.run()
    b = _sim(tmp_path, "det_b", n_steps=144, n_agents=40, **ON)
    b.run()
    assert _l1(a) == _l1(b)
    assert a.llm.calls == b.llm.calls


class _FixedLLM:
    """内容非依存の固定応答 LLM(test_day_plan と同型)。

    mock は**プロンプト全文**で乱数を引くので、k=free/off でプロンプトが変われば
    発話内容が変わり、そこから co-location が変わって呼数も揺れる(engaged 以前から
    ある性質で、fire 単独 ON でも 30 体 144 step で ±2 の差が出る)。ここで見たいのは
    「**engaged の機構が k を読んでいないか**」なので、内容の揺れを固定して切り分ける。
    """

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """compute_matched の下で k=free と k=off の呼数が完全一致する(R1)。

    engaged の判定入力は S(観測チャンネル)・ニーズゲージ・同席・接触台帳・conf だけで、
    **beliefs(k の作用点)を 1 つも読まない**ことの機械的な固定。
    """
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=30, **ON,
                   **{"controls.mode": "compute_matched", "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("k_free", "free")
    off = _run("k_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"engaged の呼数が k に依存(R1 違反): {free.llm.calls} vs {off.llm.calls}"
    assert free._engaged_state["episodes"] == off._engaged_state["episodes"] > 0


def test_on_adds_no_new_llm_call_site(tmp_path):
    """呼び出しサイト(purpose)の集合が engaged ON で増えない(§6 の予算整合)。"""
    base = _sim(tmp_path, "site_base", n_steps=144, n_agents=40, **FIRE_ON)
    base.run()
    on = _sim(tmp_path, "site_on", n_steps=144, n_agents=40, **ON)
    on.run()
    purposes = {c["purpose"] for c in on.logger.llm_calls}
    assert purposes <= {c["purpose"] for c in base.logger.llm_calls}, \
        f"engaged が新しい呼び出しサイトを作った: {purposes}"


def test_resume_matches_straight(tmp_path):
    """mid-day resume が straight と L1/L2/L3 一致(_engaged_state の checkpoint 中央管理)。

    生きているエピソード本体(agent._engaged)と不応期(_engaged_refr)は agents pickle に
    自然同梱されるので、中央管理するのは L2 5 列と summary の材料だけ(第86 と同じ型)。
    """
    ov = {**ON, "observer.checkpoint_every": 48}
    split, total, n = 48, 96, 25
    straight_dir = tmp_path / "eg_straight"
    straight = Simulation(_cfg("eg_straight", total, n, **ON), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "episode_start"), "テスト前提が崩れた(エピソードが立っていない)"

    d = tmp_path / "eg_resumed"
    sim1 = Simulation(_cfg("eg_resumed", split, n, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    assert _kind(sim1, "episode_start"), "分割点より前にエピソードが無い(検証にならない)"
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("eg_resumed", total, n, **ov), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(engaged resume)"

    # 直接検証: タリーとエピソード本体・不応期が round-trip で復元される(空回り防止)
    sim3 = Simulation(_cfg("eg_inspect", split, n, **ov),
                      out_dir=tmp_path / "eg_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._engaged_state == sim1._engaged_state is not None
    assert sim3._engaged_state["episodes"] > 0
    assert [EG.episode_of(a) for a in sim3.agents] == \
           [EG.episode_of(a) for a in sim1.agents]
    assert [getattr(a, "_engaged_refr", None) for a in sim3.agents] == \
           [getattr(a, "_engaged_refr", None) for a in sim1.agents]


def test_off_manifest_has_no_engaged_key(tmp_path):
    from society.cognition import calib as CALIB
    off = _sim(tmp_path, "prov_off", n_steps=1, **FIRE_ON)
    assert "engaged" not in (CALIB.provenance(off) or {})
    on = _sim(tmp_path, "prov_on", n_steps=1, **ON)
    assert "engaged" in CALIB.provenance(on)
    assert CALIB.provenance(on)["engaged"]["turn_cap"] == 12


# --------------------------------------------------------------------------- #
# (G) no-fingerprint(検収基準 6)
# --------------------------------------------------------------------------- #
# (a) 本 module の機構語 = ラン中のどのプロンプトにも出てはならない
_FORBIDDEN_EVERYWHERE = (
    "engaged", "ENGAGED", "autopilot", "AUTOPILOT", "エピソード", "自動操縦",
    "不応期", "ヒステリシス", "episode_start", "episode_end", "theta_out",
    "ターン上限", "兆しメモリ", "プリエンプト", "状態機械",
)
# (b) 会話プロンプト(終結の宣言路が載る呼び)に出てはならない語。発火機構・実験条件・因子名。
_FORBIDDEN_IN_TALK_PROMPT = _FORBIDDEN_EVERYWHERE + (
    "periodic", "salience", "cog_fire", "認知イベント", "閾値", "予測誤差",
    "flat_traits", "experiment", "ablate", "mock", "シミュレーション",
    "アブレーション", "efficacy", "grievance", "ownership", "nfc",
    "risk_tolerance", "internal_locus",
)


def test_no_fingerprint_in_prompts(tmp_path):
    sim = _sim(tmp_path, "fp", n_steps=144, n_agents=40, **ON)
    seen = _spy(sim)
    sim.run()
    blob = "\n".join(seen)
    for word in _FORBIDDEN_EVERYWHERE:
        assert word not in blob, f"機構語 '{word}' がプロンプトに漏れた"
    talk = [p for p in seen if EG.PROMPT_MARK in p]
    assert talk, "会話ターンが 1 度も起きていない(検査が空振り)"
    for prompt in talk:
        for word in _FORBIDDEN_IN_TALK_PROMPT:
            assert word not in prompt, f"会話プロンプトに '{word}' が出た"


def test_prompt_section_is_the_only_prompt_change(tmp_path):
    """engaged がプロンプトへ足すのは**末尾 1 行だけ**(それ以外は 1 バイトも変わらない)。

    ラン全体の比較では発話内容・記憶・同席者が当然変わるので、seam そのものを
    `build_prompt` の直呼びで固定する(第85 の契約テストと同じ切り分けの流儀)。
    """
    sim = _sim(tmp_path, "seam", n_steps=1, **ON)
    a = sim.agents[0]
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    kwargs = dict(place_name="どこか", surprise="reply", nearby_names=["誰か"],
                  reply_to=("誰か", "やあ"), sim_min=600, step=1)
    base = deliberate.build_prompt(a, **kwargs)
    section = EG.prompt_section(sim, a, "reply")
    withsec = deliberate.build_prompt(a, engaged_section=section, **kwargs)
    assert withsec == base + "\n" + section
    assert deliberate.build_prompt(a, engaged_section=None, **kwargs) == base


def test_mock_marks_stay_in_sync_with_the_section():
    """mock 側のマーカー定数が engaged 側の節と食い違っていない(層をまたぐ文字列の同期)。"""
    from society.llm.mock import MockBackend
    assert MockBackend._END_MARK == EG.PROMPT_MARK
    cfg = load_config(["cognition.engaged.enabled=true"])
    assert MockBackend._WRAPUP_MARK in (
        f"{EG.PROMPT_MARK} そろそろこの話は締めくくるつもりで、"), \
        "切り上げターンのマーカーが本文と一致しない"
    assert bool(cfg.cognition.engaged.wrapup) is True


def test_mock_end_flag_only_with_section(tmp_path):
    """mock が end を返すのは終結の宣言路がプロンプトにあるときだけ(OFF は無風)。"""
    sim = _sim(tmp_path, "mock_end", n_steps=1, **ON)
    body = sim.llm.backend.generate("場所: どこか\n何か話してください",
                                    rng_key="x", temperature=0.7, max_tokens=64)
    assert "end" not in json.loads(body)
    marked = sim.llm.backend.generate(
        f"場所: どこか\n{EG.PROMPT_MARK} 切り上げたいなら end を書いてください",
        rng_key="x", temperature=0.7, max_tokens=64)
    data = json.loads(marked)
    if data.get("action") in ("speak", "dm"):
        assert isinstance(data.get("end"), bool)


def test_off_golden_l1_bytes(tmp_path):
    """既定 OFF の L1 parquet が fire 単独 ON と完全一致(engaged seam が恒等)。"""
    a = _sim(tmp_path, "g_a", n_steps=72, n_agents=20, **FIRE_ON)
    a.run()
    b = _sim(tmp_path, "g_b", n_steps=72, n_agents=20, **FIRE_ON,
             **{"cognition.engaged.enabled": "false"})
    b.run()
    ta = pq.read_table(Path(tmp_path) / "g_a" / "l1_events.parquet").to_pylist()
    tb = pq.read_table(Path(tmp_path) / "g_b" / "l1_events.parquet").to_pylist()
    assert ta == tb
