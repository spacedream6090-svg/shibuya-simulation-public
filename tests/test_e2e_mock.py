"""D14-1: Mock LLM で end-to-end が無クラッシュ完走し、ログ一式が出る = P0 の合格判定。"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society.config import load_config
from society.engine.simulation import Simulation


def test_e2e_mock_runs_and_logs(tmp_path):
    cfg = load_config(overrides=[
        "run.seed=42", "run.n_agents=10", "run.n_steps=144", "run.name=e2e"])
    sim = Simulation(cfg, out_dir=tmp_path / "e2e")
    summary = sim.run()

    # 完走・イベントが出ている
    assert summary["n_events"] > 100
    kinds = summary["event_kinds"]
    for required in ["route_start", "move_segment", "arrive", "speak", "reflect"]:
        assert kinds.get(required, 0) > 0, f"{required} が1件も出ていない"

    # LLM(mock)が LOD 経由で発火している
    assert summary["llm_calls"] > 0

    # L1 parquet が読める・スキーマ列が揃っている
    table = pq.read_table(tmp_path / "e2e" / "l1_events.parquet")
    assert set(table.column_names) >= {
        "step", "sim_min", "agent_id", "kind", "x", "y", "payload"}

    # 出力一式
    for name in ["l1_events.parquet", "l2_metrics.parquet",
                 "l3_snapshots.parquet", "summary.json", "config.yaml"]:
        assert (tmp_path / "e2e" / name).exists(), f"{name} がない"


def test_movement_is_continuous_not_teleport(tmp_path):
    """ユーザー要望(2026-07-03): 瞬間移動禁止 — 1step の移動距離はそのモードの速度以内。"""
    cfg = load_config(overrides=[
        "run.seed=7", "run.n_agents=10", "run.n_steps=72", "run.name=move"])
    sim = Simulation(cfg, out_dir=tmp_path / "move")
    sim.run()
    table = pq.read_table(tmp_path / "move" / "l1_events.parquet")
    rows = table.to_pylist()
    speeds = {k: float(v) for k, v in dict(cfg.world.modes.speeds).items()}
    moves = [r for r in rows if r["kind"] == "move_segment"]
    assert moves, "move_segment が出ていない"
    for row in moves:
        p = json.loads(row["payload"])
        limit = speeds[p.get("mode", "walk")] + 1e-6
        assert p["dist_m"] <= limit, \
            f"1stepで {p['dist_m']}m 移動({p.get('mode')}) = テレポートの疑い"


def test_provenance_and_vocab_logged(tmp_path):
    """ユーザー指定ログ(語彙・伝播系譜)が実際に記録される。"""
    cfg = load_config(overrides=[
        "run.seed=3", "run.n_agents=20", "run.n_steps=288", "run.name=prov"])
    sim = Simulation(cfg, out_dir=tmp_path / "prov")
    summary = sim.run()
    kinds = summary["event_kinds"]
    assert kinds.get("vocab_coin", 0) > 0, "新語が一つも生まれていない"
    assert summary["n_transmissions"] > 0, "伝播が一件も記録されていない"
