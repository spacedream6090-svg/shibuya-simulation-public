"""V-P1(prompts.p1)= プロンプト一貫性修正のテスト。

正典: docs/research/dialogue-coherence-and-model.md §1.3 / §6.1 / §9-1。
実装: src/society/cognition/prompt_p1.py + deliberate.build_prompt(p1_purpose)。

検収の 4 本柱(タスク指定):
  (a) OFF = 4 purpose すべてのプロンプトがバイト不変(実ランのプロンプトを捕まえて突合)
  (b) ON  = 各 purpose のプロンプトに矛盾指示が無い(禁止パターンの機械検査)
  (c) 応答固定バックエンドのランで **呼数・rng_key 列・L1 がバイト一致**
      (= 変わったのはプロンプト文字列だけ)
  (d) 新旧プロンプトのサンプルがファイルへ出る(A8 の目視素材)

★実 LLM は 1 本も呼ばない(mock / 応答固定スパイのみ)。
★MockBackend は**プロンプト全文で乱数を引く**(`hub.stream("mock", rng_key, prompt)`)ので、
  素の mock で ON/OFF の L1 を比べると「プロンプトが変わった」ことだけを理由に応答が変わる。
  したがって (c) は**応答固定バックエンド**で測る(= 唯一の差分をプロンプト文字列に絞る)。
"""
from __future__ import annotations

import json

import pytest

from society.cognition import planning as planning_mod
from society.cognition import prompt_p1 as P
from society.cognition import reflection as reflection_mod
from society.cognition.deliberate import _COMMON_HEADER, build_prompt
from society.config import load_config
from society.engine.simulation import Simulation

# 現行(OFF)ヘッダの先頭 2 行 = 行動メニューの入口。OFF ではどの purpose にも在る。
_MENU_HEAD = "出力は次のいずれかの JSON 1個のみ(キー名は厳守):"


# =========================================================================== #
# スタブ(build_prompt が触る口だけを持つ最小の入れ物)
# =========================================================================== #
class _Mem:
    day_summaries = ["昨日はよく歩いた"]

    def recent(self, n):
        return ["出来事A", "出来事B"]

    def retrieve(self, *a, **k):
        return ["記憶X"]

    def query_ex(self, *a, **k):
        class _R:
            hits = ["想起Y"]
            failed = False
            cue = "手掛かり"
        return _R()

    def relation_line(self, ids):
        return None


class _Agent:
    id = 0
    name = "見本"
    persona = "私はテスト用の人間です。"
    activity = "working"
    states: dict = {}
    beliefs = ["世界は変わる"]
    adopted = {"語1"}
    said = ["さっきの一言"]
    money = 100000
    self_model = {"self": "よく歩く人", "ties": "甲"}
    implicit_self = "少し疲れている"
    mem = _Mem()


def _prompts(p1: bool, city: str = "見本町") -> dict[str, str]:
    """4 purpose のプロンプトを、実装の入口(= 実ランと同じ関数)から組む。"""
    agent = _Agent()
    dp = {"min_blocks": 4, "max_blocks": 8, "max_conting": 3}
    return {
        "deliberate": build_prompt(
            agent, place_name="見本通り", surprise="social",
            nearby_names=["甲", "乙"], nearby_ids=[1, 2],
            nearby_pois=["店A"], sim_min=510, step=51, city_name=city,
            p1_purpose="deliberate" if p1 else None),
        "plan": planning_mod.build_plan_prompt(
            agent, place_name="見本通り", sim_min=510, step=51,
            city_name=city, day_plan=dp, p1=p1),
        "plan_simple": planning_mod.build_plan_prompt(
            agent, place_name="見本通り", sim_min=510, step=51,
            city_name=city, p1=p1),
        "reflect": reflection_mod.build_reflect_request(
            agent, step=51, sim_min=510, place_name="自宅",
            date_line=None, weather_line=None, reflect_cfg=None,
            reflect_variety=False, interstitial_digest=None,
            interstitial=False, city_name=city, max_tokens=896,
            think=False, recalled=[], recall_fail=None, p1=p1)["prompt"],
        "recall": reflection_mod.build_recall_request(
            agent, step=51, place_name="自宅", city_name=city,
            p1=p1)["prompt"],
    }


def _purpose_of(name: str) -> str:
    return "plan" if name.startswith("plan") else name


# =========================================================================== #
# (0) 既定 OFF の宣言(conf / registry / cfg 正準化)
# =========================================================================== #
def test_default_is_off():
    cfg = load_config()
    assert bool(cfg.prompts.p1.enabled) is False
    assert cfg.model.plan_temperature is None
    assert cfg.model.recall_temperature is None
    assert P.build_cfg(None) == {"enabled": False}
    assert P.build_cfg({}) == {"enabled": False}
    assert P.build_cfg({"enabled": "yes"})["enabled"] is True


def test_finals_profile_turns_it_on():
    """本選プロファイルは ON + 予算 896 + 温度分化(根拠コメントは conf 側)。"""
    cfg = load_config(profile="conf/finals_observe.yaml")
    assert bool(cfg.prompts.p1.enabled) is True
    assert int(cfg.model.plan_max_tokens) == 896
    assert 0.2 <= float(cfg.model.plan_temperature) <= 0.3
    assert 0.2 <= float(cfg.model.recall_temperature) <= 0.3
    assert int(cfg.model.reflect_max_tokens) == 768        # 別判断=据え置き


def test_registry_declares_the_toggle():
    from society import registry
    ids = {f.id for f in registry.FEATURES}
    assert "prompts.p1.enabled" in ids, "機能レジストリに未宣言のトグルがある"


def test_cfg_of_is_safe_on_stub_sim():
    """sim スタブ(cfg なし)でも黙って OFF に落ちる(例外を投げない)。"""
    class _S:
        pass
    assert P.enabled(_S()) is False
    assert P.purpose_for(_S(), "reflect") is None
    assert P.plan_temperature(_S()) is None
    assert P.recall_temperature(_S()) is None


def test_unknown_purpose_is_rejected():
    with pytest.raises(ValueError):
        P.header("speak", base="x")
    with pytest.raises(ValueError):
        P.purpose_for(object(), "speak")


# =========================================================================== #
# (a) OFF = バイト不変
# =========================================================================== #
def test_off_kwarg_equals_omitting_it():
    """`p1_purpose=None` を明示するのと、渡さないのは完全に同じ文字列。"""
    agent = _Agent()
    kw = dict(place_name="見本通り", surprise="solo", nearby_names=[],
              sim_min=510, step=51, city_name="見本町")
    assert build_prompt(agent, **kw) == build_prompt(agent, p1_purpose=None, **kw)


def test_off_keeps_the_shared_header_everywhere():
    """OFF は 4 purpose すべてが現行どおり = 共有ヘッダ(行動メニュー)を持つ。"""
    # 現行ヘッダそのもの(街名だけ差し替え)。ここが一致する限り冒頭はバイト不変。
    head = _COMMON_HEADER.replace("あなたはの街で", "あなたは見本町の街で")
    assert _MENU_HEAD in head                      # 足場の自己点検
    for name, prompt in _prompts(p1=False).items():
        assert prompt.startswith(head), f"{name}: OFF の冒頭が現行ヘッダと違う"
        for line in P.DISCIPLINE:
            assert line not in prompt, f"{name}: OFF なのに規律行が入っている"


def test_off_reproduces_the_documented_contradiction():
    """★§1.3 の矛盾が OFF では**実在する**ことを固定する(反証可能性)。

    ここが落ちるようになったら「直す前の姿」が変わったということなので、
    P1 の前提(何を直しているのか)から確認し直すこと。
    """
    off = _prompts(p1=False)
    for name in ("plan", "plan_simple", "reflect", "recall"):
        a = P.audit(off[name], _purpose_of(name))
        assert a["directives"] == 2, \
            f"{name}: OFF の出力形式指示が 2 本でない({a['directives']} 本)"
        assert a["menu_tokens"], f"{name}: OFF なのに行動メニュー語彙が無い"
        assert a["speech_only"], f"{name}: OFF なのに発話専用の指示が無い"
    assert P.audit(off["deliberate"], "deliberate")["directives"] == 1


# =========================================================================== #
# (b) ON = 矛盾指示が無い(機械検査)
# =========================================================================== #
def test_on_has_exactly_one_output_directive_per_purpose():
    for name, prompt in _prompts(p1=True).items():
        a = P.audit(prompt, _purpose_of(name))
        assert a["directives"] == 1, \
            f"{name}: 出力形式指示が 1 本でない({a['directives']} 本)"


def test_on_removes_the_action_menu_from_non_deliberate():
    on = _prompts(p1=True)
    for name in ("plan", "plan_simple", "reflect", "recall"):
        a = P.audit(on[name], _purpose_of(name))
        assert a["menu_tokens"] == [], \
            f"{name}: 行動メニュー語彙が残っている: {a['menu_tokens']}"
        assert a["speech_only"] == [], \
            f"{name}: 発話専用の書式指示が残っている: {a['speech_only']}"
        assert _MENU_HEAD not in on[name]
    # deliberate のヘッダは**1 バイトも変えない**(行動メニューは deliberate の本体)
    assert _MENU_HEAD in on["deliberate"]
    assert on["deliberate"].startswith(_prompts(p1=False)["deliberate"].split("\n")[0])


def test_on_injects_the_three_discipline_lines_right_after_persona():
    on = _prompts(p1=True)
    for name, prompt in on.items():
        lines = prompt.split("\n")
        idx = lines.index(_Agent.persona)
        got = lines[idx + 1:idx + 1 + len(P.DISCIPLINE)]
        assert got == list(P.DISCIPLINE), f"{name}: 規律行の位置がペルソナ直後でない"


def test_discipline_lines_are_neutral():
    """規律 3 行は中立(機構語・実験条件語・因子名・地名を 1 語も含まない)。"""
    banned = ("発火", "閾値", "驚き", "条件", "モデル", "シミュレーション",
              "エージェント", "実験", "指標", "スコア")
    joined = "".join(P.DISCIPLINE)
    for word in banned:
        assert word not in joined, f"規律行に機構語 {word} が混ざっている"
    assert len(P.DISCIPLINE) == 3
    assert all(len(line) <= 60 for line in P.DISCIPLINE), "規律行が簡潔でない"


def test_on_keeps_the_json_contract_keys():
    """ON でも各 purpose の JSON 契約キーは 1 つも変わらない(パーサ互換)。"""
    on = _prompts(p1=True)
    assert '"action": "plan"' in on["plan"] and '"blocks"' in on["plan"]
    assert '"action": "plan"' in on["plan_simple"] and '"items"' in on["plan_simple"]
    for key in ('"action": "reflect"', '"summary"', '"salient"', '"belief"'):
        assert key in on["reflect"]
    assert '"action": "recall"' in on["recall"] and '"query"' in on["recall"]


def test_on_keeps_the_mock_branch_markers():
    """ON でも mock の分岐マーカーは生きている(検収ランが黙って別経路へ行かない)。"""
    on = _prompts(p1=True)
    assert "一日の予定表:" in on["plan"]          # day_plan v1
    assert "今日一日の計画" in on["plan_simple"]   # 従来の計画タスク
    assert "内省してください" in on["reflect"]


def test_audit_reports_without_judging():
    a = P.audit(_prompts(p1=True)["reflect"], "reflect")
    assert set(a) == {"purpose", "directives", "menu_tokens", "speech_only",
                      "has_discipline"}
    assert a["has_discipline"] is True
    with pytest.raises(ValueError):
        P.audit("x", "speak")


def test_on_keeps_the_speech_hint_for_deliberate():
    """発話専用の注意行は deliberate では**残す**(そこでは矛盾でないので削らない)。"""
    on = _prompts(p1=True)
    assert P.audit(on["deliberate"], "deliberate")["speech_only"] == \
        list(P.SPEECH_ONLY_LINES)


# =========================================================================== #
# (c) 実ラン: 呼数・rng_key 列・L1 がバイト一致(変わるのはプロンプトだけ)
# =========================================================================== #
class _FixedLLM:
    """応答を固定するバックエンド(= 世界をプロンプト非依存にする)。呼びを全部記録。"""

    def __init__(self):
        self.hits = 0                              # finalize が参照
        self.calls: list[dict] = []
        self._resp = json.dumps({"action": "speak", "text": "x"},
                                ensure_ascii=False)

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls.append({"rng_key": str(rng_key), "prompt": prompt,
                           "temperature": float(temperature),
                           "max_tokens": int(max_tokens)})
        return self._resp, str(len(self.calls)), False

    def generate_many(self, reqs, workers=8):
        return [self.generate(r["prompt"], rng_key=r["rng_key"],
                              temperature=r["temperature"],
                              max_tokens=r["max_tokens"],
                              think=r.get("think", False)) for r in reqs]


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name: str, on: bool, n_steps: int = 168):
    dot = ["run.seed=42", "run.n_agents=8", f"run.n_steps={n_steps}",
           f"run.name={name}", "memory.agentic_pull=true"]
    if on:
        dot.append("prompts.p1.enabled=true")
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    spy = _FixedLLM()
    sim.llm = spy
    sim.run()
    return sim, spy


def _by_purpose(spy) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for call in spy.calls:
        head = call["rng_key"].split("/", 1)[0]
        out.setdefault(head, []).append(call["prompt"])
    return out


def test_run_on_off_is_identical_except_prompt_text(tmp_path):
    """★R1 の要: ON/OFF で呼数・rng_key 列・L1 が完全一致し、差はプロンプトだけ。"""
    sim_off, off = _run(tmp_path, "p1_off", on=False)
    sim_on, on = _run(tmp_path, "p1_on", on=True)
    assert len(off.calls) == len(on.calls) > 0
    assert [c["rng_key"] for c in off.calls] == [c["rng_key"] for c in on.calls]
    assert _l1(sim_off) == _l1(sim_on), "ON で L1 が動いた(プロンプト以外が変わっている)"
    assert any(a["prompt"] != b["prompt"] for a, b in zip(off.calls, on.calls)), \
        "ON なのにプロンプトが 1 本も変わっていない"


def test_run_off_prompts_are_untouched(tmp_path):
    """(a) OFF の実ラン: 全 purpose のプロンプトが現行どおり(規律行ゼロ・メニュー有り)。"""
    _sim, spy = _run(tmp_path, "p1_off_prompts", on=False)
    seen = _by_purpose(spy)
    assert {"deliberate", "plan", "reflect", "recall"} <= set(seen), sorted(seen)
    for head, prompts in seen.items():
        for prompt in prompts:
            assert _MENU_HEAD in prompt, head
            for line in P.DISCIPLINE:
                assert line not in prompt, head


def test_run_on_prompts_pass_the_machine_audit(tmp_path):
    """(b) ON の実ラン: purpose ごとに出力指示 1 本・メニュー混入ゼロ・規律 3 行あり。"""
    _sim, spy = _run(tmp_path, "p1_on_prompts", on=True)
    seen = _by_purpose(spy)
    assert {"deliberate", "plan", "reflect", "recall"} <= set(seen), sorted(seen)
    for head, prompts in seen.items():
        purpose = head if head in P.PURPOSES else "deliberate"
        for prompt in prompts:
            a = P.audit(prompt, purpose)
            assert a["directives"] == 1, f"{head}: 指示 {a['directives']} 本"
            assert a["has_discipline"], f"{head}: 規律行が無い"
            if purpose != "deliberate":
                assert a["menu_tokens"] == [], f"{head}: {a['menu_tokens']}"
                assert a["speech_only"] == [], f"{head}: {a['speech_only']}"


def test_run_is_deterministic_when_on(tmp_path):
    """ON 同士 2 回で L1 完全一致(乱数構造が動いていない)。"""
    a, _ = _run(tmp_path, "p1_det_a", on=True, n_steps=48)
    b, _ = _run(tmp_path, "p1_det_b", on=True, n_steps=48)
    assert _l1(a) == _l1(b)


# =========================================================================== #
# 温度・上限の seam(P1 トグルとは独立に効く)
# =========================================================================== #
def test_temperature_seams_default_to_current_values(tmp_path):
    """未設定(null)なら plan=model.temperature / recall=0.7 = 従来と完全同一。"""
    cfg = load_config()
    _sim, spy = _run(tmp_path, "p1_temp_default", on=False)
    plan = [c for c in spy.calls if c["rng_key"].startswith("plan/")]
    recall = [c for c in spy.calls if c["rng_key"].startswith("recall/")]
    assert plan and recall
    assert all(c["temperature"] == float(cfg.model.temperature) for c in plan)
    assert all(c["temperature"] == 0.7 for c in recall)


def test_temperature_seams_are_applied_when_set(tmp_path):
    """設定すると plan / recall だけがその温度になる(発話・内省へ漏れない)。"""
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=168",
           "run.name=p1_temp_set", "memory.agentic_pull=true",
           "model.plan_temperature=0.3", "model.recall_temperature=0.2"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "p1_temp_set")
    spy = _FixedLLM()
    sim.llm = spy
    sim.run()
    plan = [c for c in spy.calls if c["rng_key"].startswith("plan/")]
    recall = [c for c in spy.calls if c["rng_key"].startswith("recall/")]
    other = [c for c in spy.calls
             if not c["rng_key"].startswith(("plan/", "recall/"))]
    assert plan and recall and other
    assert all(c["temperature"] == 0.3 for c in plan)
    assert all(c["temperature"] == 0.2 for c in recall)
    assert all(c["temperature"] not in (0.3, 0.2) for c in other), \
        "plan/recall 用の温度が他 purpose へ漏れた"


def test_call_count_invariant_to_temperature_and_budget(tmp_path):
    """温度・上限を変えても呼数は 1 本も動かない(R1)。"""
    _s0, spy0 = _run(tmp_path, "p1_cnt_a", on=False)
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=168",
           "run.name=p1_cnt_b", "memory.agentic_pull=true",
           "model.plan_temperature=0.3", "model.recall_temperature=0.2",
           "model.plan_max_tokens=896"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "p1_cnt_b")
    spy1 = _FixedLLM()
    sim.llm = spy1
    sim.run()
    assert len(spy0.calls) == len(spy1.calls) > 0
    plan = [c for c in spy1.calls if c["rng_key"].startswith("plan/")]
    assert plan and all(c["max_tokens"] == 896 for c in plan)


# =========================================================================== #
# (d) 新旧サンプルのファイル出力(A8 の目視素材)
# =========================================================================== #
def test_dump_samples_writes_both_sides(tmp_path):
    path = P.dump_samples(tmp_path / "p1_prompt_samples.md")
    text = open(path, encoding="utf-8").read()
    for purpose in P.PURPOSES:
        assert f"## {purpose}" in text
    assert text.count("### 旧") == len(P.PURPOSES)
    assert text.count("### 新") == len(P.PURPOSES)
    assert _MENU_HEAD in text                      # 旧側に行動メニューが写っている
    for line in P.DISCIPLINE:
        assert line in text                        # 新側に規律行が写っている
