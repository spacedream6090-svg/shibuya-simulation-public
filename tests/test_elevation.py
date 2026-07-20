"""3D Phase 0: 標高 z 列(world/elevation.py + world.elevation 配線)のテスト。

設計: docs/research/3d-movement.md §7 Phase 0(ユーザー決定 2026-07-20「P1 に同梱」)。
- OFF(既定): sim.elevation は None・move_segment/arrive payload に "z" キーが無い
  (ゴールデン L1 バイト一致は tests/test_scenario.py が別途担保)。
- ON: 全 move_segment/arrive payload に z が付く・値は ElevationGrid.height_at(x,y) と一致・
  同 seed 再現(決定論)・LLM 呼数不変(記録専用=読み出しは純関数)。
- 単体: 双一次補間の厳密性(平面格子)・端クランプ・load の縮退(ファイル無し=None)。
補間は scripts/export_3d.py の _sample_terrain_gz と同値(sim 由来 z とエクスポート z の一致)。
"""
from __future__ import annotations

import json

import numpy as np

from society.config import load_config
from society.engine.simulation import Simulation
from society.world.elevation import ElevationGrid


# ------------------------------------------------------------ 合成地形
def _plane_grid(tmp_path, name="terr", x0=-2000.0, y0=-2000.0,
                cell_m=2000.0, nx=3, ny=3, ax=2.0, by=3.0):
    """h = ax*ix + by*iy の平面格子を terrain.npz + terrain.json で書く。"""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    H = np.asarray([[ax * ix + by * iy for ix in range(nx)]
                    for iy in range(ny)], dtype=np.float32)
    np.savez(d / "terrain.npz", heights=H)
    (d / "terrain.json").write_text(
        json.dumps({"x0": x0, "y0": y0, "cell_m": cell_m,
                    "nx": nx, "ny": ny}), encoding="utf-8")
    return d


def _sim(tmp_path, name, n=40, steps=60, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# ------------------------------------------------------------ 単体: 補間
def test_bilinear_exact_on_plane():
    """平面格子(単位セル)では格子内で厳密に平面値を返す(小数1桁丸めのみ)。"""
    g = ElevationGrid([[0.0, 2.0], [3.0, 5.0]], x0=0.0, y0=0.0, cell_m=1.0)
    assert g.height_at(0.0, 0.0) == 0.0
    assert g.height_at(1.0, 0.0) == 2.0
    assert g.height_at(0.0, 1.0) == 3.0
    assert g.height_at(1.0, 1.0) == 5.0
    assert g.height_at(0.5, 0.5) == 2.5      # (0+2+3+5)/4


def test_clamp_outside_grid():
    """格子外は端セルにクランプ(外挿しない)。"""
    g = ElevationGrid([[0.0, 2.0], [3.0, 5.0]], x0=0.0, y0=0.0, cell_m=1.0)
    assert g.height_at(-10.0, -10.0) == 0.0
    assert g.height_at(10.0, 10.0) == 5.0
    assert g.height_at(10.0, -10.0) == 2.0


def test_cell_scale_and_origin():
    """原点・セル幅がスケールしても補間位置が正しい。"""
    g = ElevationGrid([[0.0, 4.0], [8.0, 12.0]], x0=-100.0, y0=-100.0,
                      cell_m=200.0)
    assert g.height_at(0.0, 0.0) == 6.0      # 中心=4値平均
    assert g.height_at(-100.0, 100.0) == 8.0


def test_load_roundtrip_and_missing(tmp_path):
    d = _plane_grid(tmp_path)
    g = ElevationGrid.load(d)
    assert g is not None and (g.nx, g.ny) == (3, 3)
    assert g.height_at(-2000.0, -2000.0) == 0.0
    assert ElevationGrid.load(tmp_path / "empty") is None


# ------------------------------------------------------------ OFF: 既定
def test_off_no_attr_no_keys(tmp_path):
    sim = _sim(tmp_path, "off", steps=24)
    sim.run()
    assert sim.elevation is None
    for e in sim.logger.events:
        if e.kind in ("move_segment", "arrive"):
            assert "z" not in e.payload


# ------------------------------------------------------------ ON: mock
def _on_sim(tmp_path, name, steps=60, seed=42):
    d = _plane_grid(tmp_path)
    return _sim(tmp_path, name, steps=steps, seed=seed,
                **{"world.elevation.enabled": "true",
                   "world.elevation.dir": d.as_posix()})


def test_on_payloads_carry_z(tmp_path):
    """ON: 全 move_segment/arrive に z が付き、値は height_at(x,y) と一致。"""
    sim = _on_sim(tmp_path, "on")
    sim.run()
    assert sim.elevation is not None
    moves = [e for e in sim.logger.events if e.kind == "move_segment"]
    arrives = [e for e in sim.logger.events if e.kind == "arrive"]
    assert moves and arrives                 # 朝の移動が発生している(60step=10時間)
    for e in moves + arrives:
        assert "z" in e.payload
        assert e.payload["z"] == sim.elevation.height_at(e.x, e.y)
    # 平面格子なので位置が違えば z も変わる=定数でないことを確認
    zs = {e.payload["z"] for e in moves}
    assert len(zs) >= 2


def test_on_deterministic_same_seed(tmp_path):
    a = _on_sim(tmp_path, "det_a", steps=30)
    a.run()
    b = _on_sim(tmp_path, "det_b", steps=30)
    b.run()
    assert _l1(a) == _l1(b)


def test_on_llm_calls_unchanged(tmp_path):
    """z は記録専用=LLM 呼数は OFF と完全一致(R1)。"""
    off = _sim(tmp_path, "calls_off", steps=30)
    off.run()
    on = _on_sim(tmp_path, "calls_on", steps=30)
    on.run()
    assert on.llm.calls == off.llm.calls
