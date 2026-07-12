"""model.plan_max_tokens seam(タスク2)のテスト。

朝の一日計画(day_plan)の LLM 呼び出しへ渡る max_tokens が:
  (a) plan_max_tokens 未設定(既定 0)時 = model.max_tokens(=既定挙動 完全不変)
  (b) plan_max_tokens 設定時 = その値
であること、かつ設定の有無で LLM 呼数が不変(R1: 呼数は上限値に依存しない)であることを、
応答固定 backend(他テストの _FixedLLM 流儀)+ 呼び出しスパイで検証する。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine.simulation import Simulation


class _SpyLLM:
    """応答を固定し(=挙動をプロンプト非依存に固定)、各呼び出しの (rng_key, max_tokens) を記録。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0                    # simulation.finalize が参照(cache ヒット数)
        self.records: list[tuple[str, int]] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        self.records.append((rng_key, int(max_tokens)))
        return self.response, str(self.calls), False


def _run(tmp_path, name: str, overrides: list[str]):
    # 168 step = 翌日 11:00 まで(全員が一度は起床=day_plan が現れる長さ)。応答は speak 固定。
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=168", f"run.name={name}",
           *overrides]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    spy = _SpyLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.llm = spy
    sim.run()
    plan_mts = [mt for (rk, mt) in spy.records if str(rk).startswith("plan/")]
    return cfg, spy, plan_mts


def test_default_plan_tokens_equal_max_tokens(tmp_path):
    """(a) 未設定(既定 0)なら plan の max_tokens は model.max_tokens と一致=不変。"""
    cfg, spy, plan_mts = _run(tmp_path, "plan_default", [])
    assert int(cfg.model.plan_max_tokens) == 0            # 既定=0(未設定相当)
    assert plan_mts, "plan 呼び出しが1件も観測されなかった"
    assert all(mt == int(cfg.model.max_tokens) for mt in plan_mts), \
        f"plan の max_tokens が model.max_tokens と一致しない: {set(plan_mts)}"


def test_set_plan_tokens_is_used(tmp_path):
    """(b) 設定時はその値が plan 呼び出しへ渡る(発話等の他呼び出しには影響しない)。"""
    cfg, spy, plan_mts = _run(tmp_path, "plan_set", ["model.plan_max_tokens=448"])
    assert int(cfg.model.plan_max_tokens) == 448
    assert plan_mts, "plan 呼び出しが1件も観測されなかった"
    assert all(mt == 448 for mt in plan_mts), \
        f"plan の max_tokens が 448 になっていない: {set(plan_mts)}"
    # seam は plan に限定: plan 以外(発話=max_tokens / 内省=reflect_max_tokens 等)へ
    # plan 用の上限 448 は一切漏れない。
    non_plan = [mt for (rk, mt) in spy.records if not str(rk).startswith("plan/")]
    assert non_plan, "plan 以外の呼び出しが観測されなかった"
    assert 448 not in non_plan, f"plan 用の上限が plan 以外へ漏れた: {sorted(set(non_plan))}"


def test_call_count_invariant_to_plan_tokens(tmp_path):
    """設定の有無で LLM 呼数(総数・plan 数とも)は不変(R1)。"""
    cfg0, spy0, plan0 = _run(tmp_path, "cnt_default", [])
    cfg1, spy1, plan1 = _run(tmp_path, "cnt_set", ["model.plan_max_tokens=448"])
    assert spy0.calls == spy1.calls > 0, \
        f"総呼数が上限設定で変化した: {spy0.calls} vs {spy1.calls}"
    assert len(plan0) == len(plan1) > 0, \
        f"plan 呼数が上限設定で変化した: {len(plan0)} vs {len(plan1)}"
