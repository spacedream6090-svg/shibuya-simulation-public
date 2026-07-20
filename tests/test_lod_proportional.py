"""P2-S6a: LLM予算のN比例化(lod.n_proportional・第38バッチ 2026-07-20)のテスト。

ON で step 上限が ceil(density × n_agents) に置換され、OFF(既定)では従来の
max_llm_per_step がそのまま使われること。docs/plans/million-scale.md §2.3 の seam。
"""
from __future__ import annotations

from society.config import load_config
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=40, **ov):
    dot = [f"run.seed=42", f"run.n_agents={n}", "run.n_steps=2",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_default_off_uses_fixed_cap(tmp_path):
    sim = _sim(tmp_path, "off", n=40)
    assert sim.budget.max_per_step == 300


def test_on_replaces_cap_with_density_times_n(tmp_path):
    sim = _sim(tmp_path, "on40", n=40, **{"lod.n_proportional.enabled": "true"})
    assert sim.budget.max_per_step == 6          # ceil(0.15×40)=6(float誤差で7にならない)


def test_on_scales_linearly_with_n(tmp_path):
    caps = {}
    for n in (40, 200, 1000):
        sim = _sim(tmp_path, f"on{n}", n=n,
                   **{"lod.n_proportional.enabled": "true"})
        caps[n] = sim.budget.max_per_step
    assert caps == {40: 6, 200: 30, 1000: 150}


def test_on_custom_density_and_floor(tmp_path):
    sim = _sim(tmp_path, "dens", n=10,
               **{"lod.n_proportional.enabled": "true",
                  "lod.n_proportional.density": "0.5"})
    assert sim.budget.max_per_step == 5
    tiny = _sim(tmp_path, "tiny", n=1,
                **{"lod.n_proportional.enabled": "true",
                   "lod.n_proportional.density": "0.01"})
    assert tiny.budget.max_per_step == 1         # 下限1(0にはしない)
