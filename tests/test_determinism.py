"""D14-2: 同一 seed で2回走らせて L1 が完全一致(決定論リプレイ D13)。"""
from __future__ import annotations

import pyarrow.parquet as pq

from society.config import load_config
from society.engine.simulation import Simulation


def _run(tmp_path, name: str):
    cfg = load_config(overrides=[
        "run.seed=11", "run.n_agents=10", "run.n_steps=72", f"run.name={name}"])
    Simulation(cfg, out_dir=tmp_path / name).run()
    return pq.read_table(tmp_path / name / "l1_events.parquet")


def test_same_seed_same_events(tmp_path):
    t1 = _run(tmp_path, "a")
    t2 = _run(tmp_path, "b")
    assert t1.num_rows == t2.num_rows
    assert t1.to_pylist() == t2.to_pylist()


def test_different_seed_different_events(tmp_path):
    t1 = _run(tmp_path, "c")
    cfg = load_config(overrides=[
        "run.seed=12", "run.n_agents=10", "run.n_steps=72", "run.name=d"])
    Simulation(cfg, out_dir=tmp_path / "d").run()
    t2 = pq.read_table(tmp_path / "d" / "l1_events.parquet")
    assert t1.to_pylist() != t2.to_pylist()
