"""B段 seam: logistic 発火 + LogNormal 閾値。既定(fixed / normal)は現行と不変。"""
from __future__ import annotations

import statistics

import numpy as np

from society.config import load_config
from society.engine.simulation import Simulation
from society.factors.registry import drive_params, sample_traits


def _run(tmp_path, name, **overrides):
    dot = ["run.n_agents=16", "run.n_steps=90", f"run.name={name}",
           "observer.snapshot_every=45"]
    dot += [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [(e.step, e.agent_id, e.kind, e.payload) for e in sim.logger.events]


def test_logistic_firing_completes_and_requests(tmp_path):
    """drive.firing=logistic で完走し、drive_request(発火含む)が出る。"""
    sim = _run(tmp_path, "logi", **{"drive.firing": "logistic"})
    reqs = [e for e in sim.logger.events if e.kind == "drive_request"]
    assert reqs, "logistic で drive_request が出ていない"
    assert any(e.payload["granted"] for e in reqs), "logistic で発火がゼロ"


def test_fixed_firing_is_golden(tmp_path):
    """drive.firing=fixed(既定)は firing 未指定と完全一致(現行コードパス不変)。"""
    a = _run(tmp_path, "fx_explicit", **{"drive.firing": "fixed"})
    b = _run(tmp_path, "fx_default")
    assert _l1(a) == _l1(b)


def test_lognormal_threshold_distribution(tmp_path):
    """lognormal 閾値: 全て正(clip 域内)・中央値が normal 近傍(moment matching)。"""
    rng_t = np.random.default_rng(0)
    norm, logn = [], []
    for i in range(2000):
        tr = sample_traits(rng_t)
        norm.append(drive_params(tr, np.random.default_rng(500 + i), "normal")[0])
        logn.append(drive_params(tr, np.random.default_rng(500 + i), "lognormal")[0])
    assert all(0.30 <= x <= 0.85 for x in logn)      # 全て正・clip 域内
    assert all(x > 0 for x in logn)
    assert abs(statistics.median(norm) - statistics.median(logn)) < 0.03


def test_lognormal_threshold_wires_through_sim(tmp_path):
    """factors.threshold_dist=lognormal が手続き生成経路で配線され完走する。"""
    sim = _run(tmp_path, "logn_sim", **{"factors.threshold_dist": "lognormal"})
    for a in sim.agents:
        assert 0.30 <= a.drive_threshold <= 0.85


def test_normal_threshold_draw_count_unchanged():
    """normal(既定)と lognormal で乱数消費本数が同一 = 既定の再現性に影響しない。

    同じ rng を渡したとき、閾値の後段(fire_weight)が両分布で完全一致する
    (= 閾値サンプルが 1 draw で、後続の draw 位置がずれない)。
    """
    tr = sample_traits(np.random.default_rng(1))
    _, fw_normal = drive_params(tr, np.random.default_rng(9), "normal")
    _, fw_lognorm = drive_params(tr, np.random.default_rng(9), "lognormal")
    assert fw_normal == fw_lognorm
