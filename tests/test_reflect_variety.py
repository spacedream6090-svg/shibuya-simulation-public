"""A4 内省プロンプト改善(prompts.reflect_variety)のテスト。

要件(docs/plans/unimplemented-inventory.md A4・第20バッチ検収):
 - OFF: 内省タスク文が現行(_REFLECT_TASK / _DEEP_REFLECT_TASK)とバイト一致(ゴールデン不変)。
 - ON: belief 説明のバリアントが agent×day で決定論的に変わる(同入力2回同一・異なる入力で相違あり)。
 - JSON 契約キー(summary / salient / belief / self / ties)は ON/OFF とも不変。
 - mock 24step の ON ランが決定論(同 seed で L1 一致)。1日回すと reflect が発火し、ON では
   プロンプトに定型句回避の1行とバリアントが載る一方、契約(payload キー)は保たれる。
"""
from __future__ import annotations

import json

from society.cognition.reflection import (
    _BELIEF_VARIANTS,
    _DEEP_REFLECT_TASK,
    _REFLECT_TASK,
    _VARIETY_NOTE,
    _reflect_task,
)
from society.config import load_config
from society.engine.simulation import Simulation
from society.rng import _stable_hash

_ORIG = "今日の経験からの考え方の変化・結論を一文"   # 元の雛形例文(冒頭句アンカー)


# --------------------------------------------------------------- unit: OFF 不変
def test_off_matches_current_prompt():
    """OFF は個体・日に依らず従来のプロンプト定数と完全一致(ゴールデンを守る)。"""
    for deep in (False, True):
        base = _DEEP_REFLECT_TASK if deep else _REFLECT_TASK
        for aid, day in ((0, 0), (3, 5), (99, 12)):
            assert _reflect_task(deep=deep, variety=False,
                                 agent_id=aid, day=day) == base


# --------------------------------------------------- unit: ON の決定論・ローテーション
def test_on_is_deterministic_per_agent_day():
    """同じ (agent, day) の入力は2回引いても同一・OFF とは異なる。"""
    a = _reflect_task(deep=False, variety=True, agent_id=3, day=5)
    b = _reflect_task(deep=False, variety=True, agent_id=3, day=5)
    assert a == b
    assert a != _REFLECT_TASK


def test_on_rotates_over_agent_and_day():
    """複数の (agent, day) で実際に2種類以上のバリアントが出る=ローテーションしている。"""
    seen = {_reflect_task(deep=False, variety=True, agent_id=a, day=d)
            for a in range(12) for d in range(12)}
    assert len(seen) >= 2
    # 同一 agent でも day が変われば変わりうる(少なくとも1組の相違を具体的に示す)。
    assert any(
        _reflect_task(deep=False, variety=True, agent_id=1, day=d1)
        != _reflect_task(deep=False, variety=True, agent_id=1, day=d2)
        for d1 in range(12) for d2 in range(12))


def test_variant_selection_uses_stable_hash():
    """バリアント選択は乱数でなく rng._stable_hash(プロセス非依存)で決まる。"""
    for aid in (1, 7, 42):
        for day in (0, 3, 9):
            idx = _stable_hash(f"reflect_variety/{aid}/{day}") % len(_BELIEF_VARIANTS)
            got = _reflect_task(deep=True, variety=True, agent_id=aid, day=day)
            assert _BELIEF_VARIANTS[idx] in got


# ------------------------------------------------ unit: JSON 契約キー + 定型句回避
def test_json_contract_keys_unchanged():
    """summary / salient / belief(deep は self / ties も)のキーは ON/OFF とも不変。"""
    for deep in (False, True):
        for variety in (False, True):
            t = _reflect_task(deep=deep, variety=variety, agent_id=2, day=4)
            for key in ('"action": "reflect"', '"summary"', '"salient"', '"belief"'):
                assert key in t
            if deep:
                assert '"self"' in t and '"ties"' in t


def test_variety_note_and_variants_are_clean():
    """ON は定型句回避の1行を足し、元の雛形例文は消える。バリアントは 3〜5 個・意味等価。"""
    on = _reflect_task(deep=False, variety=True, agent_id=2, day=4)
    off = _reflect_task(deep=False, variety=False, agent_id=2, day=4)
    assert _VARIETY_NOTE.strip() in on and "決まり文句" in on
    assert "決まり文句" not in off
    assert _ORIG not in on                    # 元の belief 例文まるごとは ON では消える
    assert 3 <= len(_BELIEF_VARIANTS) <= 5
    for v in _BELIEF_VARIANTS:                # バリアント自身は雛形の冒頭句を含まない
        assert "今日の経験から" not in v


# ------------------------------------------ integration: mock ランの決定論・発火・契約
def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name, n_steps, **ov):
    dot = ["run.seed=42", "run.n_agents=15", f"run.n_steps={n_steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return sim


def test_mock_on_24step_is_deterministic(tmp_path):
    """ON の 24step mock ランは同 seed で L1 完全一致(乱数構造・発火が不変=R1)。"""
    a = _run(tmp_path, "on_a", 24, **{"prompts.reflect_variety": "true"})
    b = _run(tmp_path, "on_b", 24, **{"prompts.reflect_variety": "true"})
    assert _l1(a) == _l1(b)


class _ReflectSpy:
    """sim.llm を包み reflect 呼の prompt を記録する。委譲は素通し=挙動は不変。"""

    def __init__(self, inner):
        self.inner = inner
        self.reflect_prompts: list[str] = []

    def generate(self, prompt, **kw):
        if str(kw.get("rng_key", "")).startswith("reflect/"):
            self.reflect_prompts.append(prompt)
        return self.inner.generate(prompt, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_mock_on_reflect_prompt_uses_variant_and_fires(tmp_path):
    """1日回すと reflect が発火。ON では定型句回避の1行+バリアントが載り、雛形例文は消える。
    reflect イベントの JSON 契約(payload キー)は保たれる。"""
    cfg = load_config(["run.seed=42", "run.n_agents=15", "run.n_steps=144",
                       "run.name=on_full", "prompts.reflect_variety=true"])
    sim = Simulation(cfg, out_dir=tmp_path / "on_full")
    sim.llm = _ReflectSpy(sim.llm)
    sim.run()
    assert sim.llm.reflect_prompts, "内省が1件も発火していない(step 数/就寝時刻を確認)"
    for p in sim.llm.reflect_prompts:
        assert "決まり文句" in p            # 定型句回避の1行が入っている
        assert _ORIG not in p               # 元の雛形例文は消えている
    revs = [e for e in sim.logger.events if e.kind == "reflect"]
    assert revs
    for e in revs:
        for key in ("mode", "written_back", "belief", "summary", "n_salient"):
            assert key in e.payload


def test_mock_off_reflect_prompt_is_original(tmp_path):
    """既定 OFF では reflect プロンプトが従来どおり=雛形例文を含み・注意文を含まない。"""
    cfg = load_config(["run.seed=42", "run.n_agents=15", "run.n_steps=144",
                       "run.name=off_full"])
    sim = Simulation(cfg, out_dir=tmp_path / "off_full")
    sim.llm = _ReflectSpy(sim.llm)
    sim.run()
    assert sim.llm.reflect_prompts
    for p in sim.llm.reflect_prompts:
        assert _ORIG in p
        assert "決まり文句" not in p
