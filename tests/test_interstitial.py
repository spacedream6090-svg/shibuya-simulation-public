"""ナラティブ補間(P2 スライス S2)= 行間の機械ダイジェスト注入のテスト。

設計: docs/plans/p2-interstitial-design.md §1 S2 / docs/research/interstitial-life.md §4.2。
方針(既存 R1 の鉄則を継承):
- OFF(既定): build_prompt がバイト同一・実行時バッファ(_isl_buf)も作られない・L1 が
  純粋既定と完全一致(ゴールデン golden_baseline_l1.json は tests/test_scenario.py が別途担保)。
- ON: (a)前回発火以降の客観ダイジェスト「この間のこと:」がプロンプトに載る。
  (b)同 seed 2回で L1 完全一致(決定論)。(c)LLM 呼数は OFF と同一(追加呼ゼロ=k 非依存)。
  (d)実行時バッファは有界(最大30件のリングバッファ)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤24 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.schema import Event

_ON = {"prompts.interstitial.enabled": "true"}
_OFF = {"prompts.interstitial.enabled": "false"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = [f"run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# ------------------------------------------------------ OFF: build_prompt バイト同一
def test_build_prompt_seam_byte_identical_off(tmp_path):
    """interstitial_digest=None(既定)は引数なしの build_prompt とバイト同一。ON は1行増える。"""
    sim = _sim(tmp_path, "seam", steps=1)
    sim.run()
    agent = sim.agents[0]
    base = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3)
    none = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3,
                                   interstitial_digest=None)
    assert none == base, "interstitial_digest=None が既定と非同一(バイト一致が崩れている)"
    dig = "この間のこと: 今日の予定は3件中2件を済ませた"
    on = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                 nearby_names=[], sim_min=600, step=3,
                                 interstitial_digest=dig)
    assert dig in on and on != base, "ON でダイジェスト行がプロンプトに載っていない"
    assert "この間のこと" not in base, "OFF なのにダイジェストの語がプロンプトに出ている"


# ------------------------------------------------------ OFF: L1 が純粋既定と一致 + バッファ不在
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(24 step)。OFF では実行時バッファも作られない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(interstitial seam が no-op でない)"
    for a in pure.agents:
        assert not hasattr(a, "_isl_buf"), "OFF なのに実行時バッファ _isl_buf が生えている"


# ------------------------------------------------------ ダイジェスト生成器(客観列挙・決定論)
def test_digest_builder_objective(tmp_path):
    """_isl_digest はバッファ+計画から客観ダイジェストを組む(意味づけしない・決定論)。"""
    sim = _sim(tmp_path, "dig", steps=1)
    sim.run()
    agent = sim.agents[0]
    agent.day_plan = [{"when": "朝", "what": "shop", "place": "", "done": True},
                      {"when": "昼", "what": "meal", "place": "", "done": False}]
    # 立ち寄った場所(重複は畳む・先出し順)/ 会った人 / 起きたこと の順で列挙
    agent._isl_buf = [("place", "北の広場"), ("place", "北の広場"),
                      ("person", "花子"), ("act", "買い物や支払いをした")]
    d1 = scheduler._isl_digest(agent)
    d2 = scheduler._isl_digest(agent)      # 乱数不使用=呼ぶたび同一
    assert d1 == d2, "ダイジェストが非決定論(乱数を引いている)"
    assert d1 is not None and d1.startswith("この間のこと: ")
    assert "1件を済ませた" in d1 and "北の広場に立ち寄った" in d1
    assert "花子と会った" in d1 and "買い物や支払いをした" in d1
    assert d1.count("北の広場") == 1, "重複した場所が畳まれていない"
    # 空バッファ・計画未消化なら None(=build_prompt が1行も足さない)
    empty = sim.agents[1]
    empty.day_plan = []
    empty._isl_buf = []
    assert scheduler._isl_digest(empty) is None


# ------------------------------------------------------ (a) ダイジェストがプロンプトに載る
class _RecordingLLM:
    """内側 LLM に委譲しつつ、渡されたプロンプト全文を記録する薄いラッパ(mock 24step 用)。"""

    def __init__(self, inner):
        self._inner = inner
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self._inner.generate(prompt, **kw)

    @property
    def calls(self):
        return self._inner.calls

    @property
    def hits(self):
        return self._inner.hits


def test_digest_appears_in_prompt_on(tmp_path):
    """ON の mock 24step で、実際の発火プロンプトにダイジェスト「この間のこと:」が載る。"""
    sim = _sim(tmp_path, "appear", n=30, **_ON)
    sim.llm = _RecordingLLM(sim.llm)
    sim.run()
    assert any("この間のこと:" in p for p in sim.llm.prompts), \
        "ON の発火プロンプトにダイジェストが一度も載っていない"


# ------------------------------------------------------ (b) 決定論(同 seed 2回で完全一致)
def test_on_deterministic(tmp_path):
    """interstitial ON 同士 2 回で L1 完全一致(決定論・mock 24 step)。"""
    a = _sim(tmp_path, "det_a", **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


def test_on_differs_from_off(tmp_path):
    """ON は OFF と L1 が異なる(=補間が実際に注入されプロンプト依存の mock 出力が動く)。"""
    on = _sim(tmp_path, "diff_on", **_ON)
    on.run()
    off = _sim(tmp_path, "diff_off", **_OFF)
    off.run()
    assert _l1(on) != _l1(off), "ON と OFF が同一(ダイジェストが注入されていない疑い)"


# ------------------------------------------------------ (c) LLM 呼数が OFF と同一(追加呼ゼロ)
class _FixedLLM:
    """プロンプト内容に依存しない固定応答 backend。呼数だけを数える(内容→下流分岐を無効化して
    『補間が generate() を1本も足さない』ことを純粋に測る)。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, enabled):
    sim = _sim(tmp_path, name, **{"prompts.interstitial.enabled": enabled})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_call_count_invariant(tmp_path):
    """補間 ON/OFF で LLM 呼数が完全一致(追加 LLM 呼ゼロ=呼数 k 非依存・R1)。

    固定 LLM は内容非依存=下流(発火)が ON/OFF で同一なので、呼数差が出るのは補間自身が
    generate() を足したときだけ。載せているのはプロンプト内容のみなので呼数は不変。"""
    on = _run_fixed(tmp_path, "cc_on", "true")
    off = _run_fixed(tmp_path, "cc_off", "false")
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"補間で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


# ------------------------------------------------------ (d) バッファが有界(30件上限)
def test_buffer_bounded(tmp_path):
    """実行時リングバッファは最大 30 件(超過は最古を捨てる)。"""
    sim = _sim(tmp_path, "bound", steps=1, **_ON)
    sim.run()
    aid = sim.agents[0].id
    start = len(sim.logger.events)
    for i in range(50):                    # 前回発火以降に 50 件の到着が積まれる状況を作る
        sim.logger.log(Event(step=0, sim_min=0, agent_id=aid, kind="arrive",
                             x=0.0, y=0.0, payload={"name": f"場所{i}"}))
    scheduler._isl_accumulate(sim, start)
    buf = getattr(sim.agents[0], "_isl_buf", [])
    assert len(buf) == scheduler._ISL_BUF_MAX == 30, \
        f"バッファが有界でない: len={len(buf)}(上限={scheduler._ISL_BUF_MAX})"
    # 最古が捨てられ最新側が残る(リングバッファの向き)
    assert ("place", "場所49") in buf and ("place", "場所19") not in buf
