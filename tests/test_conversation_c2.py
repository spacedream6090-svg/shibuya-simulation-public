"""会話3層 C2/C3(P2 スライス S3)= 構造化会話(LLMなし)+ すれ違い集計のテスト。

設計: docs/plans/p2-interstitial-design.md §1 S3 / docs/research/interstitial-life.md §4.3。
方針(既存 R1 の鉄則を継承):
- OFF(既定): L1 が純粋既定とバイト一致・conversation イベント0件・agent に _c2/_c3 の
  実行時フィールドを一切生やさない(ゴールデン golden_baseline_l1.json は tests/test_scenario.py
  が別途担保)。新 stream `c2_meet` を一度も引かない。
- ON(mock ≤24step): (a)conversation イベントが出る。(b)同 seed 2回で L1 完全一致(決定論)。
  (c)追加 LLM 呼ゼロ = 予算飽和下で ON と OFF の LLM 呼数が一致(C2→C1 昇格は固定予算の
  再配分であって拡張ではない)+ 呼数が k(writeback)条件に非依存。(d)会話数/人日が目標
  30±50%圏(24step=4時間分 → ×6 で日換算)。(e)機械効果(関係値・意見が動く)。(f)C2→C1 昇格
  (drive ゲージが押し上がる)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤24step)。乱数は会話成立の抽選 c2_meet 1本のみ。

※ 対面 C1(フル LLM 発話の確定発火・返答保証)は別機構で tests/test_conversation.py が担保する。
"""
from __future__ import annotations

import json

from society import conversation
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS
from society.world.perception import build_index

_ON = {"conversation.enabled": "true"}
_OFF = {"conversation.enabled": "false"}


def _sim(tmp_path, name, n=60, steps=24, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _convs(sim):
    return [e for e in sim.logger.events if e.kind == "conversation"]


def _scene(tmp_path, name, **ov):
    """2 個体(a, b)を同席させ、他は就寝で index から外す(会話を2者に閉じる=決定論)。"""
    sim = _sim(tmp_path, name, n=6, steps=1, **ov)
    a, b = sim.agents[0], sim.agents[1]
    for o in sim.agents:
        if o not in (a, b):
            o.sleeping = True
    for ag in (a, b):
        ag.loc = "street"
        ag.building = None
        ag.floor = 0
        ag.sleeping = False
        ag.node = a.node
        ag.x, ag.y = 50.0, 50.0
        ag.drive = 0.0
    return sim, a, b


def _run_c2(sim, step=0, sim_min=600):
    sim.percept_index = build_index(
        sim.agents, float(sim.cfg.world.perception_radius_m))
    conversation.run_phase(sim, step, sim_min)


# ------------------------------------------------------------ schema 登録
def test_conversation_kind_registered():
    assert "conversation" in EVENT_KINDS


# ------------------------------------------------------------ OFF: 純粋既定と一致 + 無副作用
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(24step)。OFF では実行時フィールドも作られない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(C2 seam が no-op でない)"
    assert not any(e.kind == "conversation" for e in off.logger.events)
    for a in pure.agents:
        assert not hasattr(a, "_c2_day"), "OFF なのに _c2_day が生えている"
        assert not hasattr(a, "_c3_pass"), "OFF なのに _c3_pass が生えている"
        assert not hasattr(a, "_c2_cooldown_until")


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致(no-op の確度を上げる)。"""
    pure = _sim(tmp_path, "purel", n=15, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=15, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# ------------------------------------------------------------ (a) conversation が出る
def test_on_emits_conversations(tmp_path):
    """ON の mock 24step で conversation イベントが出る(payload の型を固定)。"""
    sim = _sim(tmp_path, "emit", **_ON)
    sim.run()
    convs = _convs(sim)
    assert convs, "ON なのに conversation イベントが1件も出ていない"
    p = convs[0].payload
    assert set(p) == {"with", "topic", "tone", "outcome", "acts"}
    assert isinstance(p["with"], int) and isinstance(p["acts"], int)
    assert p["topic"] in {"smalltalk", "catchup", "opinion", "novelty"}
    assert p["tone"] in {"friendly", "neutral", "cool"}
    assert p["outcome"] in {"uneventful", "opinion_gap", "rapport_shift", "new_word"}


# ------------------------------------------------------------ (b) 決定論(同 seed 2回で一致)
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(c2_meet 以外の乱数漏れ?)"


def test_on_differs_from_off(tmp_path):
    """ON は OFF と L1 が異なる(=C2 が実際にイベント・機械効果を注入している)。"""
    on = _sim(tmp_path, "diff_on", **_ON)
    on.run()
    off = _sim(tmp_path, "diff_off", **_OFF)
    off.run()
    assert _l1(on) != _l1(off)


# ------------------------------------------------------------ (c) 追加 LLM 呼ゼロ
def test_call_count_invariant_under_saturated_budget(tmp_path):
    """予算飽和下では ON と OFF の LLM 呼数が一致する(C2→C1 昇格=固定予算の再配分であって
    拡張ではない=追加 LLM 呼ゼロ・R1)。

    lod.max_llm_per_step を小さくして毎 step 予算を飽和させる(n=100 で全 step が飽和)。
    飽和下では総発火数 = Σ min(申請者数, 予算) = step 数 × 予算 で ON/OFF とも同一になり、
    C2 が「誰が発火するか」を変えても「何本発火するか」は変えないことを示す。予算に余裕がある
    非飽和時は昇格が実際に発火を増やす(=密度機構として意図した挙動)ので、この検査は飽和下で行う。"""
    ov = {"lod.max_llm_per_step": 2}
    on = _sim(tmp_path, "cc_on", n=100, **{**_ON, **ov})
    on.run()
    off = _sim(tmp_path, "cc_off", n=100, **{**_OFF, **ov})
    off.run()
    assert len(on.logger.llm_calls) == len(off.logger.llm_calls) > 0, \
        f"予算飽和下で呼数が不一致: on={len(on.logger.llm_calls)} off={len(off.logger.llm_calls)}"


def test_call_count_k_invariant(tmp_path):
    """真の R1「呼数 k 非依存」: ON のまま k(writeback)を free/off に振っても LLM 呼数が不変。

    C2 の drive 押し上げは opinion/relations/観測量のみを入力にし、k 由来量(belief 書き戻し)を
    一切見ないため、呼数は k 条件間で完全一致する(reflection-drift.md §3 の R1 監査と同型)。"""
    free = _sim(tmp_path, "k_free", **{**_ON, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", **{**_ON, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ------------------------------------------------------------ (d) 会話数/人日が目標圏
def test_density_per_person_day(tmp_path):
    """ON の mock 24step で、会話数/人日(×6 日換算)が目標 30±50% 圏 [15,45] に入る。

    在場密度依存(interstitial-life §6)なので、~30/人日 の初期値較正の参照密度として n=200 を使う
    (co-presence が薄い小規模 mock では上限が下がるため)。24step=4時間 → ×6 で日換算(タスク指定)。"""
    n = 200
    sim = _sim(tmp_path, "dens", n=n, steps=24, **_ON)
    sim.run()
    e = len(_convs(sim))
    per_person_day = 2.0 * e / n * (144.0 / 24.0)   # 1会話=2人・24step→144step(×6)
    assert 15.0 <= per_person_day <= 45.0, \
        f"会話密度が目標圏外: {per_person_day:.1f}/人日(events={e})"


# ------------------------------------------------------------ (e) 機械効果(意見・関係が動く)
def test_opinion_shifts_via_fj(tmp_path):
    """C2 会話で双方の意見が相手側へ FJ 更新される(k 非依存・opinion_shift を2件記録)。"""
    sim, a, b = _scene(tmp_path, "op", **{**_ON, "conversation.c2.meet_prob": 1.0})
    a.opinion, a.opinion_anchor = 0.9, 0.9
    b.opinion, b.opinion_anchor = -0.9, -0.9
    _run_c2(sim)
    assert _convs(sim), "同席2者で会話が成立していない"
    assert a.opinion < 0.9 and b.opinion > -0.9, "意見が相手側へ動いていない(FJ 未適用)"
    assert sum(1 for e in sim.logger.events if e.kind == "opinion_shift") == 2


def test_relations_update(tmp_path):
    """C2 会話で関係台帳が両方向に更新される(OFF=record_contact の count、ON=closeness)。"""
    # relations OFF: count が両方向で増える(従来の record_contact)
    sim, a, b = _scene(tmp_path, "rel", **{**_ON, "conversation.c2.meet_prob": 1.0})
    _run_c2(sim)
    assert b.id in a.mem.relations and a.id in b.mem.relations, "関係台帳が更新されていない"
    assert a.mem.relations[b.id]["count"] >= 1
    assert "closeness" not in a.mem.relations[b.id], "relations OFF で closeness が付いた"
    # relations ON: closeness(符号つき)が付く。反論スタンス=負の交流でネガ側へ。
    sim2, a2, b2 = _scene(tmp_path, "rel_on",
                          **{**_ON, "conversation.c2.meet_prob": 1.0,
                             "relations.enabled": "true"})
    a2.opinion, a2.opinion_anchor = 0.9, 0.9
    b2.opinion, b2.opinion_anchor = -0.9, -0.9      # 強い意見差 → 反論 → 負の交流
    _run_c2(sim2)
    assert "closeness" in a2.mem.relations[b2.id], "relations ON で closeness が付かない"
    assert a2.mem.relations[b2.id]["closeness"] < 0.0, "反論なら closeness は負へ動く"


def test_vocabulary_contagion(tmp_path):
    """C2 会話で片方の未知語がもう片方へ complex contagion 経路(labels.on_hear)で伝わる。"""
    sim, a, b = _scene(tmp_path, "vocab", **{**_ON, "conversation.c2.meet_prob": 1.0})
    word = "ヨセアツメ"
    a.adopted.add(word)
    item = sim.labels.items.new_item("vocab", word, a.id, 0)
    sim.labels.text_to_item[word] = item
    _run_c2(sim)
    assert b.heard_counts.get(item.item_id, 0) >= 1, "未知語の聴取(heard_counts)が起きていない"
    assert any(e.kind == "transmission" for e in sim.logger.events), \
        "transmission(伝播系譜)が記録されていない"
    conv = _convs(sim)[0].payload
    assert conv["topic"] == "novelty" and conv["outcome"] == "new_word"


# ------------------------------------------------------------ (f) C2→C1 昇格(drive 押し上げ)
def test_promotion_pushes_drive(tmp_path):
    """C2 会話が drive ゲージを押し上げ(addressed)、顕著な帰結(強い意見差)はより強く押す。

    押し上げは既存 drive 経由 → 次以降の step の _phase_drive がフル LLM 発話(C1)を発火する接続点。"""
    # 平穏(意見一致・語なし): addressed のみ
    calm, ca, cb = _scene(tmp_path, "calm", **{**_ON, "conversation.c2.meet_prob": 1.0})
    ca.opinion = cb.opinion = 0.5
    ca.opinion_anchor = cb.opinion_anchor = 0.5
    _run_c2(calm)
    assert _convs(calm)[0].payload["outcome"] == "uneventful"
    assert ca.drive > 0.0 and cb.drive > 0.0, "会話で drive が全く上がっていない"
    calm_push = ca.drive

    # 強い意見差: addressed + state_change(gap 比例)で平穏より強く押す
    gap, ga, gb = _scene(tmp_path, "gap", **{**_ON, "conversation.c2.meet_prob": 1.0})
    ga.opinion, ga.opinion_anchor = 0.9, 0.9
    gb.opinion, gb.opinion_anchor = -0.9, -0.9
    _run_c2(gap)
    assert _convs(gap)[0].payload["outcome"] == "opinion_gap"
    assert ga.drive > calm_push, "強い意見差の昇格押し上げが平穏より弱い(昇格が効いていない)"


# ------------------------------------------------------------ C3(すれ違い集計)
def test_c3_counters(tmp_path):
    """C3: 同席は会話有無に関わらず「すれ違った」に数え、会話成立で「挨拶」に数える。日次カウンタ。"""
    # 会話成立 → pass も greet も相手を含む
    met, a, b = _scene(tmp_path, "c3met", **{**_ON, "conversation.c2.meet_prob": 1.0})
    _run_c2(met)
    assert b.id in a._c3_pass and b.id in a._c3_greet
    assert a.id in b._c3_pass and a.id in b._c3_greet
    # 会話不成立(meet_prob=0)→ すれ違いのみ(pass)・挨拶(greet)は空・イベント0件
    miss, a2, b2 = _scene(tmp_path, "c3miss", **{**_ON, "conversation.c2.meet_prob": 0.0})
    _run_c2(miss)
    assert b2.id in a2._c3_pass and not a2._c3_greet
    assert not _convs(miss), "会話不成立なのに conversation イベントが出た"


# ------------------------------------------------------------ 会話内容の決定論写像(純関数)
def test_plan_conversation_deterministic_mapping(tmp_path):
    """plan_conversation は両者状態から型・帰結を決定論写像する(同入力2回で完全一致)。"""
    sim, a, b = _scene(tmp_path, "plan", **_ON)
    c2 = conversation._cfg(sim)["c2"]
    # 意見一致 → 同意スタンス
    a.opinion, b.opinion = 0.5, 0.5
    p_agree = conversation.plan_conversation(a, b, sim.opinioncfg, c2)
    assert p_agree["stance"] == "agree"
    assert conversation.plan_conversation(a, b, sim.opinioncfg, c2) == p_agree
    # 強い意見差 → 反論スタンス・cool トーン・opinion_gap 帰結
    a.opinion, b.opinion = 0.95, -0.95
    p_opp = conversation.plan_conversation(a, b, sim.opinioncfg, c2)
    assert p_opp["stance"] == "oppose" and p_opp["tone"] == "cool"
    assert p_opp["outcome"] == "opinion_gap"
    assert 3 <= p_opp["acts"] <= 5
