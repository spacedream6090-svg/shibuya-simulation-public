"""内面本格版(現実ギャップ 後続波 H6、既定 OFF)= 離散感情・長期目標・趣味 のテスト。

設計正典: docs/design-candidates/gap-implementation-plan.md(§後続波)/
          docs/lit/neuroscience__emotion-interest-attention.md §5.2(本格版)。
検証の柱(既存の鉄則を継承):
- OFF 既定 == 純粋既定の L1 完全一致(144 step)。**最重要**(ゴールデン protecting = バイト一致)。
- 離散感情 ON(+affect ON): core affect の象限からラベルが決まり emotion_label が出てプロンプトに載る。
- 長期目標 ON: needs/traits から目標が決まり long_goal が出てプロンプトに載る(単体・決定論)。
- 趣味 ON: 趣味文脈がプロンプトに載る / 余暇の行き先が趣味の場所へ寄る(単体)。
- 決定論(ON 同士2回=L1 一致)。ON で L1 が変わる。
- R1: 注入のみ(leisure_bias=0)は FixedLLM で ON==OFF。行き先バイアス込みは compute_matched 下の
  k 不変性(k=free==k=off の呼数一致)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from society.config import load_config
from society.engine.simulation import Simulation
from society import inner_life


# --------------------------------------------------------------------------- helpers
def _sim(tmp_path, name: str, n: int = 20, steps: int = 24, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=12"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim) -> list:
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# 全 ON(感情=affect ON 前提 / 目標 / 趣味 注入のみ)。行き先バイアスは既定 0(=純粋注入)。
def _on_ov() -> dict:
    return {"inner_life.enabled": "true",
            "affect.enabled": "true", "affect.arousal_gain": 0.15}


# ============================================================ 1. cfg / 恒等の核
def test_cfg_default_off():
    assert inner_life.build_cfg({})["enabled"] is False
    assert inner_life.build_cfg(None)["enabled"] is False
    c = inner_life.build_cfg({"enabled": True})
    assert c["enabled"] is True
    assert c["emotion"]["enabled"] is True and c["goals"]["enabled"] is True
    assert c["hobbies"]["leisure_bias"] == 0.0        # 既定は注入のみ(純粋 ON==OFF)


# ============================================================ 2. 離散感情ラベル(純関数・§1.2)
def test_emotion_label_quadrants():
    ecfg = inner_life.build_cfg({"enabled": True})["emotion"]
    # 高覚醒 × 負 = 不安 / 高覚醒 × 正 = 高揚 / 低覚醒 × 正 = 穏やか / 低覚醒 × 負 = しょんぼり
    assert inner_life.emotion_label(-0.6, 0.8, ecfg)[0] == "不安"
    assert inner_life.emotion_label(0.6, 0.8, ecfg)[0] == "高揚"
    assert inner_life.emotion_label(0.6, 0.1, ecfg)[0] == "穏やか"
    assert inner_life.emotion_label(-0.6, 0.1, ecfg)[0] == "しょんぼり"
    # 中立 valence(deadzone 内): 高覚醒=高ぶり / 低覚醒=平静
    assert inner_life.emotion_label(0.0, 0.8, ecfg)[0] == "高ぶり"
    assert inner_life.emotion_label(0.0, 0.1, ecfg)[0] == "平静"
    # 純関数(同入力=同出力)
    assert inner_life.emotion_label(0.6, 0.8, ecfg) == inner_life.emotion_label(0.6, 0.8, ecfg)


def test_mood_valence_from_states():
    assert inner_life._mood_valence({"efficacy": 0.8, "grievance": 0.1}) > 0
    assert inner_life._mood_valence({"efficacy": 0.2, "grievance": 0.9}) < 0
    assert inner_life._mood_valence({}) == 0.5        # 既定 efficacy0.5-grievance0.0


def test_update_emotion_logs_on_change():
    cfg = inner_life.build_cfg({"enabled": True})
    logs = []
    logger = SimpleNamespace(log=lambda e: logs.append(e))
    ag = SimpleNamespace(id=1, x=0.0, y=0.0, arousal=0.8,
                         states={"efficacy": 0.8, "grievance": 0.1})
    inner_life.update_emotion(ag, cfg, step=0, sim_min=0, logger=logger)
    assert ag._emotion_label == "高揚" and "高揚" in ag._emotion_phrase
    assert len(logs) == 1 and logs[0].kind == "emotion_label"
    inner_life.update_emotion(ag, cfg, step=1, sim_min=10, logger=logger)  # 不変
    assert len(logs) == 1, "ラベル不変なのに emotion_label が再記録された(sparse でない)"
    ag.arousal, ag.states = 0.1, {"efficacy": 0.2, "grievance": 0.9}       # しょんぼりへ
    inner_life.update_emotion(ag, cfg, step=2, sim_min=20, logger=logger)
    assert ag._emotion_label == "しょんぼり" and len(logs) == 2


# ============================================================ 3. 長期目標・趣味(決定論)
def test_long_goal_from_value_profile():
    # 刺激最優位 → 「新しいことに挑戦…」/ 関係最優位 → 「仲間を…」
    stim = SimpleNamespace(_needs_profile={"stimulation": 0.95, "security": 0.2,
                                           "relatedness": 0.2, "competence": 0.2,
                                           "autonomy": 0.2}, occupation="会社員")
    assert "挑戦" in inner_life.long_goal(stim)
    rel = SimpleNamespace(_needs_profile={"stimulation": 0.2, "security": 0.2,
                                          "relatedness": 0.95, "competence": 0.2,
                                          "autonomy": 0.2}, occupation="会社員")
    assert "仲間" in inner_life.long_goal(rel)


def test_hobbies_from_occupation():
    band = SimpleNamespace(occupation="バンドマン", traits={})
    assert inner_life.hobbies(band)[0] == "音楽・ライブ"
    ap = SimpleNamespace(occupation="アパレル店員", traits={})
    assert inner_life.hobbies(ap)[0] == "ファッション・買い物"


# ============================================================ 4. OFF: バイト一致(最重要)
def test_off_matches_pure_default(tmp_path):
    """inner_life.enabled=false なら、他パラメータが非ゼロでも純粋既定と L1 完全一致(144 step)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "off_params", steps=144,
               **{"inner_life.enabled": "false",
                  "inner_life.hobbies.leisure_bias": 0.5,
                  "inner_life.goals.leisure_bias": 0.5})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(inner_life seam が no-op でない)"
    for k in ("emotion_label", "long_goal"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# ============================================================ 5. ON: イベント + プロンプト注入
def test_long_goal_logged_and_in_prompt(tmp_path):
    from society.cognition import deliberate
    sim = _sim(tmp_path, "goal_on", steps=2, **_on_ov())
    sim.run()
    lg = _kind(sim, "long_goal")
    assert lg, "long_goal が 1 件も出ていない(目標が付与されていない)"
    agent = sim.agents[0]
    assert getattr(agent, "life_goal", None), "life_goal が設定されていない"
    icfg = sim.innerlifecfg
    line = inner_life.goal_line(agent, icfg)
    assert line and agent.life_goal in line
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3,
                                     goal_line=line)
    assert agent.life_goal in prompt, "goal_line がプロンプトに載っていない"
    plain = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                    nearby_names=[], sim_min=600, step=3)
    assert "長期的な目標" not in plain, "goal_line=None なのに目標語が載っている"


def test_emotion_label_on_and_in_prompt(tmp_path):
    from society.cognition import deliberate
    sim = _sim(tmp_path, "emo_on", n=30, steps=144, **_on_ov())
    sim.run()
    el = _kind(sim, "emotion_label")
    assert el, "emotion_label が 1 件も出ていない(affect ON でも感情ラベルが動いていない)"
    # プロンプト注入: agent に感情句が保持され、emotion_line として載る(affect ON 前提)
    agent = next((a for a in sim.agents if getattr(a, "_emotion_phrase", None)), None)
    assert agent is not None, "感情句が保持された agent が居ない"
    line = inner_life.emotion_line(agent, sim.innerlifecfg, affect_on=True)
    assert line and "今の気持ち" in line
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3,
                                     emotion_line=line)
    assert "今の気持ち" in prompt


def test_emotion_needs_affect_on(tmp_path):
    """affect OFF では感情ラベルは無効(emotion_label 0 件・emotion_line=None)。"""
    sim = _sim(tmp_path, "emo_no_affect", steps=48,
               **{"inner_life.enabled": "true", "affect.enabled": "false"})
    sim.run()
    assert not _kind(sim, "emotion_label"), "affect OFF なのに emotion_label が出ている"
    agent = sim.agents[0]
    assert inner_life.emotion_line(agent, sim.innerlifecfg, affect_on=False) is None
    # 目標・趣味は affect と無関係に出る
    assert _kind(sim, "long_goal"), "affect OFF でも long_goal は出るはず"


def test_hobby_line_in_prompt(tmp_path):
    from society.cognition import deliberate
    sim = _sim(tmp_path, "hob_on", steps=2, **_on_ov())
    sim.run()
    agent = sim.agents[0]
    assert getattr(agent, "hobbies", None), "hobbies が付与されていない"
    line = inner_life.hobby_line(agent, sim.innerlifecfg)
    assert line and "趣味・関心" in line
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3,
                                     hobby_line=line)
    assert "趣味・関心" in prompt


# ============================================================ 6. 趣味の行き先バイアス(単体)
def test_hobby_dest_biases_leisure(tmp_path):
    """leisure_bias=1.0 で趣味の POI カテゴリのノードへ寄る。bias=0(既定)は乱数を引かず None。"""
    from society.cognition import routine
    sim = _sim(tmp_path, "hob_bias", n=20, steps=1,
               **{"inner_life.enabled": "true",
                  "inner_life.hobbies.leisure_bias": 1.0})
    sim.run()
    agent = sim.agents[0]
    cat = inner_life._HOBBY_CAT.get(agent.hobbies[0])
    cat_nodes = {p["node"] for p in sim.city.pois_by_cat(cat) if p["node"] != agent.node}
    if cat_nodes:                                     # 趣味カテゴリの POI がある地図なら寄る
        dest = inner_life.hobby_dest(agent, sim, step=7, sim_min=600)
        assert dest in cat_nodes, "bias=1.0 なのに趣味の場所へ寄っていない"
    # bias=0(注入のみ)は乱数を引かず None(既定挙動)
    sim0 = _sim(tmp_path, "hob_nobias", n=20, steps=1,
                **{"inner_life.enabled": "true"})
    sim0.run()
    assert inner_life.hobby_dest(sim0.agents[0], sim0, step=7, sim_min=600) is None


# ============================================================ 7. 決定論 / ON で L1 が変わる
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", n=30, steps=72, **_on_ov()); a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=72, **_on_ov()); b.run()
    assert _l1(a) == _l1(b), "inner_life ON が同 seed で非決定論"


def test_on_changes_l1(tmp_path):
    off = _sim(tmp_path, "chg_off", steps=48,
               **{"affect.enabled": "true", "affect.arousal_gain": 0.15}); off.run()
    on = _sim(tmp_path, "chg_on", steps=48, **_on_ov()); on.run()
    assert _l1(off) != _l1(on), "inner_life ON で L1 が変わらない(効いていない)"


# ============================================================ 8. R1: 呼数不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, ov, steps=30, n=20, seed=42) -> int:
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"] + [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim.llm.calls


def test_injection_only_call_count_invariant(tmp_path):
    """FixedLLM: inner_life ON(注入のみ・leisure_bias=0)と OFF で generate 呼数が完全一致。

    感情/目標/趣味のプロンプト注入は内容のみ=発火判断(drive)に触れない → 呼数は不変(R1)。
    affect は両側 ON(threshold_gain=0 で発火に非接続)にして arousal 経路を揃え、inner_life の
    注入だけを差分にする。"""
    base = {"affect.enabled": "true", "affect.arousal_gain": 0.15,
            "affect.threshold_gain": 0.0}
    off = _run_fixed(tmp_path, "cc_off", base)
    on = _run_fixed(tmp_path, "cc_on", {**base, "inner_life.enabled": "true"})
    assert off == on and off > 0, (off, on)


def _run_bias_k(tmp_path, name, *, writeback) -> "Simulation":
    """趣味の行き先バイアス ON を compute_matched(k 掃引で使う対照)下で回し呼数を数える。"""
    dot = ["run.seed=42", "run.n_agents=30", "run.n_steps=144", f"run.name={name}",
           "inner_life.enabled=true", "inner_life.hobbies.leisure_bias=1.0",
           "controls.mode=compute_matched", f"k.writeback={writeback}"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_hobby_bias_call_count_k_invariant(tmp_path):
    """趣味の行き先バイアスは対面 co-location を変えうる(ON!=OFF)。だが R1 の本旨=「呼数が
    k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。趣味の移動は名簿・config・
    物理位置・専用 stream のみ参照し k・内面状態を一切読まないため、k=free と k=off で呼数が完全一致する。"""
    free = _run_bias_k(tmp_path, "bk_free", writeback="free")
    off = _run_bias_k(tmp_path, "bk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"趣味バイアスの呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# ============================================================ 9. mock 全 ON 完走 + 内面イベント
def test_mock_full_on_smoke(tmp_path):
    sim = _sim(tmp_path, "smoke", n=30, steps=144, **_on_ov())
    summary = sim.run()
    assert summary["n_steps"] == 144
    assert _kind(sim, "long_goal"), "long_goal が出ていない"
    assert _kind(sim, "emotion_label"), "emotion_label が出ていない(arousal が動いていない)"
