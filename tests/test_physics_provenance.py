# -*- coding: utf-8 -*-
"""第155: physics.provenance の summary.json 配線(G1/C1 を縦煙の summary から読むため)。

規約は tests/test_summary_provenance.py の 6 層と同じ:
  ・ゾーン未使用(= ``_phys_state`` 無し)では **キー自体を出さない**(既存ラン無風)
  ・ON では ``summary["physics"] == physics.provenance(sim)``(summary 側で作り直さない)
配線の取りこぼし検知(finalize から呼ばれていること)は
tests/test_summary_provenance.py の④(AST 走査)が自動で担う。
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from society import physics
from society.config import load_config
from society.engine.simulation import Simulation

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R],
         [_ZONE_R, _ZONE_R], [-_ZONE_R, _ZONE_R]]


def _cfg(name, n=30, steps=8, zone_specs=None):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    cfg = load_config(dot)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
    return cfg


def _run(tmp_path, name, **kw):
    sim = Simulation(_cfg(name, **kw), out_dir=tmp_path / name)
    sim.run()
    return sim


def _summary(sim):
    return json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))


def test_no_physics_key_when_zones_are_off(tmp_path):
    sim = _run(tmp_path, "prov_phys_off")
    assert physics.provenance(sim) is None
    assert "physics" not in _summary(sim), \
        "ゾーン未使用のランの summary に physics キーが生えた(既存ラン・golden が無風でなくなる)"


def test_summary_carries_physics_provenance_when_zones_run(tmp_path):
    zone = {"id": "z1", "engine": "orca", "dt_sub": 0.05, "polygon": list(_POLY)}
    sim = _run(tmp_path, "prov_phys_on", zone_specs=[zone])
    prov = physics.provenance(sim)
    assert prov is not None, "ゾーン付きランで provenance が None(ON セットの取り違え)"
    summary = _summary(sim)
    assert "physics" in summary, "physics が summary に無い(配線が外れている)"
    assert summary["physics"] == prov, "physics: summary 側で観測量を作り直している"
    # 縦煙で読む中身が実際に居ることを固定(G1=sub_steps_total / C1=levers.ownership_mode)。
    assert "sub_steps_total" in prov["continuity"]
    assert prov["continuity"]["sub_steps_total"] > 0
    assert set(prov["levers"]) == {"ownership_mode", "ownership_max_dist_m",
                                   "min_gap_every"}
    for z in prov["by_zone_last_step"].values():
        assert {"occupancy", "occupancy_mean", "density", "waiting",
                "sub_steps"} <= set(z)


def test_physics_key_is_appended_at_the_tail(tmp_path):
    """キー順の保存(第109 D2 と同じ規約): 既存キーの後ろに足す。"""
    zone = {"id": "z1", "engine": "orca", "dt_sub": 0.05, "polygon": list(_POLY)}
    sim = _run(tmp_path, "prov_phys_tail", zone_specs=[zone])
    keys = list(_summary(sim))
    assert keys.index("files") < keys.index("physics")
    assert keys.index("elapsed_sec") < keys.index("physics")
