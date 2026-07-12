"""欲求駆動発火(Phase A)のテスト。

1. 発火が時間的に分散する(旧: 全員が序盤に同期バースト)
2. ゲージ更新は信念(beliefs)に依存しない(R1: k との計算量交絡の機構レベル防止)
3. k 条件を変えても発火回数がほぼ同じ(許容±20%。世界の分岐による揺らぎは許す)
4. 抽選落ちで数十%減衰する(ユーザー仕様)
"""
from __future__ import annotations

import json
from collections import Counter

from society.config import load_config
from society.engine.simulation import Simulation


def _run(tmp_path, name: str, **overrides):
    dotlist = [f"run.n_agents=20", f"run.n_steps=60", f"run.name={name}",
               "observer.snapshot_every=30"]
    dotlist += [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(dotlist)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def test_fires_are_dispersed(tmp_path):
    sim = _run(tmp_path, "drv1")
    fire_steps = [e.step for e in sim.logger.events
                  if e.kind == "drive_request"
                  and json.loads(json.dumps(e.payload))["granted"]]
    assert len(fire_steps) >= 10, "発火がほとんど起きていない"
    counts = Counter(fire_steps)
    # 1つの step に発火が集中しない(最大でも全体の 35% 未満)
    assert max(counts.values()) < max(3, 0.35 * len(fire_steps))
    # 発火が広い時間帯に散っている
    assert len(counts) >= 8


def test_drive_ignores_beliefs(tmp_path):
    """ゲージ入力は出来事のみ: beliefs を書き換えてもゲージ増分は不変。"""
    from society.cognition import drive

    class A:
        pass

    cfg = drive.build_cfg({})
    a1, a2 = A(), A()
    for a in (a1, a2):
        a.drive = 0.0
        a._drive_reasons = {}
    a2.beliefs = ["世界は変えられる", "自分がやるしかない"]
    for a in (a1, a2):
        drive.add(a, "novel_place", cfg)
        drive.add(a, "addressed", cfg)
    assert a1.drive == a2.drive


def test_fire_count_stable_across_k(tmp_path):
    fires = {}
    for mode in ("free", "off"):
        sim = _run(tmp_path, f"drvk_{mode}", **{"k.writeback": mode})
        fires[mode] = sum(1 for e in sim.logger.events
                          if e.kind == "drive_request" and e.payload["granted"])
    lo, hi = sorted(fires.values())
    assert lo > 0
    assert hi <= lo * 1.2 + 3, f"k条件間で発火回数が乖離: {fires}(R1 監査)"


def test_reject_decays_gauge():
    from society.cognition import drive

    class A:
        pass

    a = A()
    a.drive = 0.8
    a._drive_reasons = {"novel_place": 0.5}
    cfg = drive.build_cfg({"fail_decay": 0.30})
    drive.on_reject(a, cfg)
    assert abs(a.drive - 0.8 * 0.7) < 1e-9      # 数十%減衰
    assert a._drive_reasons                      # 理由は保持(再申請に備える)
