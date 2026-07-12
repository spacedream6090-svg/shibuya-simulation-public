"""E2: 内省しやすさの時間ドリフト(発火閾値ドリフト、既定 OFF)のテスト。

設計: docs/research/reflection-drift.md。実効閾値 = clip(drive_threshold + theta_drift,
0.30, 0.85)。3 力(馴化↑ / 鋭敏化↓ / 自発回復→0)を機構レベルで検証し、OFF 不変・clip・
k 間発火数±20%(R1)・決定論をシミュレーションで確認する。
"""
from __future__ import annotations

import json

import numpy as np

from society.cognition import drive
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors.registry import drift_params


# --------------------------------------------------------------------------- #
# 機構レベル(drive.py の純関数)
# --------------------------------------------------------------------------- #
def _agent(drift_p):
    class A:
        pass
    a = A()
    a.drive = 0.0
    a._drive_reasons = {}
    a.theta_drift = 0.0
    a.drift_p = drift_p
    a.drive_threshold = 0.55
    a.conv_turns_left = 3
    a.conv_cooldown_until = 0
    a.drive_mods = None
    return a


def _cfg(**drift):
    d = {"enabled": True, "habit_rate": 0.0, "sens_rate": 0.0,
         "recovery_rate": 0.0, "max_abs": 0.15}
    d.update(drift)
    return drive.build_cfg({"drift": d})


def test_off_is_identity_noop():
    """drift_p None(=OFF): 出来事/発火/step を重ねても theta_drift=0・実効閾値=drive_threshold。"""
    a = _agent(None)
    cfg = drive.build_cfg({})                      # drift 未指定 = 無効
    for _ in range(50):
        drive.add(a, "sns", cfg)
    drive.on_fire(a, 3, cfg)
    drive.step_tick(a, cfg, 4)
    assert a.theta_drift == 0.0
    assert drive.effective_threshold(a) == a.drive_threshold


def test_habituation_raises_threshold():
    """低顕著の単調入力(sns)が続くと theta_drift↑=実効閾値が上がる(内省が減る)。"""
    cfg = _cfg(habit_rate=0.05)
    a = _agent({"habit_rate": 0.05, "sens_rate": 0.0, "recovery_rate": 0.0,
                "direction_bias": 0.0})
    base = drive.effective_threshold(a)
    for _ in range(30):
        drive.add(a, "sns", cfg)
    assert a.theta_drift > 0.0
    assert drive.effective_threshold(a) > base


def test_firing_sensitizes_lowers_threshold():
    """発火経験(練習効果)は theta_drift↓=実効閾値が下がる(乗ってくる)。"""
    cfg = _cfg(sens_rate=0.03)
    a = _agent({"habit_rate": 0.0, "sens_rate": 0.03, "recovery_rate": 0.0,
                "direction_bias": 0.0})
    a.drive = 0.5
    drive.on_fire(a, 5, cfg)
    assert a.theta_drift < 0.0
    assert drive.effective_threshold(a) < a.drive_threshold


def test_novelty_sensitizes():
    """新奇/強イベント(初訪問)は脱馴化=theta_drift↓。"""
    cfg = _cfg(sens_rate=0.05)
    a = _agent({"habit_rate": 0.0, "sens_rate": 0.05, "recovery_rate": 0.0,
                "direction_bias": 0.0})
    drive.add(a, "novel_place", cfg)
    assert a.theta_drift < 0.0


def test_recovery_returns_to_zero():
    """自発回復: 出来事が馴化に効かない設定で step を重ねると theta_drift が 0 へ縮む。"""
    cfg = _cfg(recovery_rate=0.1)
    a = _agent({"habit_rate": 0.0, "sens_rate": 0.0, "recovery_rate": 0.1,
                "direction_bias": 0.0})
    a.theta_drift = 0.1
    for i in range(20):
        drive.step_tick(a, cfg, i)
    assert 0.0 < a.theta_drift < 0.1               # 単調に 0 へ近づく(まだ 0 ではない)


def test_clip_bounds_theta_drift():
    """theta_drift は ±max_abs を超えない(暴走防止)。"""
    cfg = _cfg(habit_rate=1.0, max_abs=0.15)
    a = _agent({"habit_rate": 1.0, "sens_rate": 0.0, "recovery_rate": 0.0,
                "direction_bias": 0.0})
    for _ in range(200):
        drive.add(a, "company", cfg)
    assert a.theta_drift <= 0.15 + 1e-9
    assert drive.effective_threshold(a) <= 0.85 + 1e-9


def test_drift_params_sensitizer_vs_habituator():
    """NFC 高=鋭敏化型(sens>habit)、NFC 低/内的統制低=馴化型(habit>sens)。"""
    base = {"habit_rate": 0.02, "sens_rate": 0.02, "recovery_rate": 0.01}
    hi = drift_params({"nfc": 0.95, "internal_locus": 0.7, "risk_tolerance": 0.5},
                      np.random.default_rng(0), base)
    lo = drift_params({"nfc": 0.05, "internal_locus": 0.2, "risk_tolerance": 0.5},
                      np.random.default_rng(0), base)      # 同 seed=speed 一致で db を隔離
    assert hi["direction_bias"] > lo["direction_bias"]
    assert hi["sens_rate"] > hi["habit_rate"]              # sensitizer
    assert lo["habit_rate"] > lo["sens_rate"]              # habituator


# --------------------------------------------------------------------------- #
# シミュレーションレベル
# --------------------------------------------------------------------------- #
def _run(tmp_path, name, **ov):
    dot = ["run.seed=42", "run.n_agents=20", "run.n_steps=60", f"run.name={name}",
           "observer.snapshot_every=20"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _granted(sim):
    return sum(1 for e in sim.logger.events
               if e.kind == "drive_request" and e.payload["granted"])


def test_off_leaves_theta_zero_and_no_column(tmp_path):
    """既定 OFF: 全 agent の theta_drift=0、L2 に mean_theta_drift 列が出ない。"""
    sim = _run(tmp_path, "d_off")
    assert all(a.theta_drift == 0.0 and a.drift_p is None for a in sim.agents)
    assert all("mean_theta_drift" not in row for row in sim.logger.metrics)


def test_on_wires_and_shifts(tmp_path):
    """ON: drift_p が全員に付き、theta_drift が動く/L2 mean_theta_drift・L3 theta_drift が出る。"""
    sim = _run(tmp_path, "d_on", **{"drive.drift.enabled": "true",
                                    "drive.drift.habit_rate": 0.05,
                                    "drive.drift.sens_rate": 0.05})
    assert all(a.drift_p is not None for a in sim.agents)
    assert any(abs(a.theta_drift) > 1e-6 for a in sim.agents)
    assert all(abs(a.theta_drift) <= 0.15 + 1e-9 for a in sim.agents)
    l2 = sim.logger.metrics
    assert any("mean_theta_drift" in row for row in l2)
    snaps = sim.logger.snapshots
    assert snaps
    last = json.loads(snaps[-1]["state"])
    assert all("theta_drift" in ag for ag in last["agents"])


def test_fire_count_stable_across_k_with_drift(tmp_path):
    """R1: drift ON でも k∈{free,off} の発火数が乖離しない(±20%)。ドリフトは k 非依存。"""
    fires = {}
    for mode in ("free", "off"):
        sim = _run(tmp_path, f"dk_{mode}",
                   **{"drive.drift.enabled": "true", "k.writeback": mode})
        fires[mode] = _granted(sim)
    lo, hi = sorted(fires.values())
    assert lo > 0
    assert hi <= lo * 1.2 + 3, f"drift ON で k 間発火数が乖離: {fires}(R1 監査)"


def test_drift_on_deterministic(tmp_path):
    """同 seed 2 回で drift ON の L1(閾値含む)が完全一致(決定論)。"""
    def l1(name):
        sim = _run(tmp_path, name, **{"drive.drift.enabled": "true"})
        return [(e.step, e.agent_id, e.kind, json.dumps(e.payload, sort_keys=True,
                                                        ensure_ascii=False))
                for e in sim.logger.events]
    assert l1("det1") == l1("det2")
