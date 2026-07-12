"""SNS/DM 架橋距離(第22バッチ P2)のテスト。

設計(docs/plans/legacy-adoption.md P2): sns_geo.enabled=true のときだけ、SNS/DM 経由の
transmission payload に送り手との物理距離 dist_m を追記する。乱数・LLM 呼数は変えない
(キーを足すだけ)。既定 OFF は純粋既定と L1 バイト一致。
"""
from __future__ import annotations

import json
import math

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _cfg(name, n=20, steps=24, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n=20, steps=24, **ov):
    return Simulation(_cfg(name, n, steps, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _trans(sim):
    return [e for e in sim.logger.events if e.kind == "transmission"]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。dist_m はどこにも現れない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"sns_geo.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "sns_geo OFF が純粋既定と不一致(seam が no-op でない)"
    assert not any("dist_m" in e.payload for e in _trans(pure))


def test_off_no_dist_even_for_dm(tmp_path):
    """既定(キー未指定)では DM 経由でも dist_m は付かない。"""
    sim = _sim(tmp_path, "offdm")
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "テスト語", step=0, sim_min=0, logger=sim.logger)
    scheduler._hear_words(sim, b, ["テスト語"], a.id, "dm", 0, 0)
    ev = _trans(sim)[-1]
    assert ev.payload["channel"] == "dm" and "dist_m" not in ev.payload


# --------------------------------------------------------------------- ON
def test_dm_carries_dist_when_on(tmp_path):
    """ON: DM の transmission に送り手との距離が載る(値も一致)。"""
    sim = _sim(tmp_path, "on", **{"sns_geo.enabled": "true"})
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "テスト語", step=0, sim_min=0, logger=sim.logger)
    scheduler._hear_words(sim, b, ["テスト語"], a.id, "dm", 0, 0)
    ev = _trans(sim)[-1]
    assert ev.payload["channel"] == "dm"
    assert ev.payload["dist_m"] == round(math.hypot(a.x - b.x, a.y - b.y), 1)


def test_sns_carries_dist_when_on(tmp_path):
    """ON: SNS 経由も同様(旧実装の本命=ネットが物理距離を架橋した実測)。"""
    sim = _sim(tmp_path, "onsns", **{"sns_geo.enabled": "true"})
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "テスト語", step=0, sim_min=0, logger=sim.logger)
    scheduler._hear_words(sim, b, ["テスト語"], a.id, "sns", 0, 0)
    assert _trans(sim)[-1].payload["dist_m"] == round(
        math.hypot(a.x - b.x, a.y - b.y), 1)


def test_face_and_media_have_no_dist(tmp_path):
    """ON でも対面(共在が前提)と送り手なし(メディア=-1)には付けない。"""
    sim = _sim(tmp_path, "onface", **{"sns_geo.enabled": "true"})
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "テスト語", step=0, sim_min=0, logger=sim.logger)
    scheduler._hear_words(sim, b, ["テスト語"], a.id, "face", 0, 0)
    assert "dist_m" not in _trans(sim)[-1].payload
    scheduler._hear_words(sim, b, ["テスト語"], -1, "sns", 0, 0)   # 送り手が実在しない
    assert "dist_m" not in _trans(sim)[-1].payload
