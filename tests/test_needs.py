"""欲求プロファイルの個人差プラグイン(needs、既定 OFF)のテスト = Wave2-2。

検証の柱:
- ON: エージェント間で欲求プロファイル(倍率ベクトル)が実際に分散し、「何を不快と感じるか」の
  個体差が congestion 等の感度差として出る。発火の顔ぶれ・分布も個体で変わる。
- OFF(既定): drive の挙動が現行と完全一致(needs_mods 不在=恒等 → 発火列がバイト一致、
  effective_threshold 不変)。
- R1(k 非依存): 同一 seed/agent で k=free と k=off のプロファイルが同一。needs ON でも発火数が
  k 間で偏らない(mock で free/off の発火数 ±20% 以内)。
- 決定論(同設定2回=同一列)/ mock 24step 完走。

steward 統合前提: engine/simulation.py は未配線のため、本テストは needs.precompute を
Simulation 構築直後(run 前)に直接呼ぶことで ON 経路を再現する(psych.sdt ブロックと同型の 3 行)。
"""
from __future__ import annotations

import json
from collections import Counter

from society.cognition import drive
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors import registry
from society import needs


# --------------------------------------------------------------------------- helpers
def _sim(tmp_path, name: str, n: int = 20, steps: int = 24, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=12"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


def _l1(sim) -> list:
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _fires_by_agent(sim) -> Counter:
    return Counter(e.agent_id for e in sim.logger.events
                   if e.kind == "drive_request" and e.payload["granted"])


def _run_on(tmp_path, name, cfg=None, **ov) -> Simulation:
    """ON 経路: 構築直後に needs.precompute を挿す(steward 配線の代替)。"""
    sim = _sim(tmp_path, name, **ov)
    needs.precompute(sim.agents, sim.hub, cfg or needs.build_cfg({"enabled": True}))
    sim.run()
    return sim


# ============================================================ 1. cfg / 恒等の核
def test_needs_cfg_default_off():
    assert needs.build_cfg({})["enabled"] is False
    assert needs.build_cfg(None)["enabled"] is False
    assert needs.build_cfg({"enabled": True})["enabled"] is True


def test_center_profile_is_identity():
    """全次元 0.5(中立)のプロファイル → 全 reason で倍率 1.0(=現行と一致)。"""
    cfg = needs.build_cfg({"enabled": True})
    center = {d: 0.5 for d in registry.NEEDS_DIMS}
    mods = needs.mods_from_profile(center, cfg)
    assert set(mods) == set(drive.DEFAULT_WEIGHTS)   # 全 10 reason を網羅
    for r, m in mods.items():
        assert abs(m - 1.0) < 1e-9, (r, m)


# ============================================================ 2. drive.add の seam
def test_drive_add_applies_needs_mods():
    """drive.add が needs_mods の不透明倍率を scale に掛ける(reason 文字列しか見ない)。"""
    cfg = drive.build_cfg({})

    class A:
        pass

    a = A(); a.drive = 0.0; a._drive_reasons = {}
    a.needs_mods = {"congestion": 1.5}
    drive.add(a, "congestion", cfg)
    assert abs(a.drive - drive.DEFAULT_WEIGHTS["congestion"] * 1.5) < 1e-9
    # 倍率 dict に無い reason は 1.0(現行どおり)
    b = A(); b.drive = 0.0; b._drive_reasons = {}
    b.needs_mods = {"congestion": 1.5}
    drive.add(b, "sns", cfg)
    assert abs(b.drive - drive.DEFAULT_WEIGHTS["sns"]) < 1e-9
    # needs_mods 属性が無いオブジェクトは恒等(既定 OFF 経路=バイト一致)
    c = A(); c.drive = 0.0; c._drive_reasons = {}
    drive.add(c, "congestion", cfg)
    assert abs(c.drive - drive.DEFAULT_WEIGHTS["congestion"]) < 1e-9


def test_needs_and_sdt_compose_multiplicatively():
    """drive_mods(SDT)と needs_mods(needs)は乗算合成(非衝突)。"""
    cfg = drive.build_cfg({})

    class A:
        pass

    a = A(); a.drive = 0.0; a._drive_reasons = {}
    a.drive_mods = {"novel_place": 1.5}
    a.needs_mods = {"novel_place": 1.2}
    drive.add(a, "novel_place", cfg)
    assert abs(a.drive - drive.DEFAULT_WEIGHTS["novel_place"] * 1.5 * 1.2) < 1e-9


# ============================================================ 3. 「何を不快と感じるか」の個体差
def test_congestion_sensitivity_varies_by_profile():
    """安全志向が高い個体は混雑を強く不快(倍率>1)・刺激欲求が高い個体は耐性(倍率<1)。"""
    cfg = needs.build_cfg({"enabled": True})
    safe = {d: 0.5 for d in registry.NEEDS_DIMS}
    safe["security"], safe["stimulation"] = 0.95, 0.15
    seeker = {d: 0.5 for d in registry.NEEDS_DIMS}
    seeker["security"], seeker["stimulation"] = 0.15, 0.95
    m_safe = needs.mods_from_profile(safe, cfg)["congestion"]
    m_seeker = needs.mods_from_profile(seeker, cfg)["congestion"]
    assert m_safe > 1.0 > m_seeker, (m_safe, m_seeker)
    # 逆に novel_place は刺激欲求側で欲求↑
    assert (needs.mods_from_profile(seeker, cfg)["novel_place"]
            > needs.mods_from_profile(safe, cfg)["novel_place"])


def test_profile_direction_from_traits_and_age():
    """registry.needs_profile の向き: 若く高リスクは刺激↑、老いて低リスクは安全↑。"""
    import numpy as np
    cfg = needs.build_cfg({"enabled": True})
    rng = np.random.default_rng(0)
    young_bold = registry.needs_profile(
        {"nfc": 0.9, "internal_locus": 0.5, "risk_tolerance": 0.9},
        age=20, occupation="無職", rng=rng, cfg={**cfg, "profile_sigma": 0.0})
    old_cautious = registry.needs_profile(
        {"nfc": 0.5, "internal_locus": 0.5, "risk_tolerance": 0.1},
        age=68, occupation="無職", rng=rng, cfg={**cfg, "profile_sigma": 0.0})
    assert young_bold["stimulation"] > old_cautious["stimulation"]
    assert old_cautious["security"] > young_bold["security"]


# ============================================================ 4. ON: プロファイル分散(sim)
def test_needs_on_profiles_diverge(tmp_path):
    """ON: 全員に needs_mods が付き、倍率ベクトルが個体間で分散する。特に congestion 感度が
    1.0 の両側に散る(=何を不快と感じるかの多様性)。"""
    sim = _sim(tmp_path, "diverge", n=30)
    needs.precompute(sim.agents, sim.hub, needs.build_cfg({"enabled": True}))
    mods = [a.needs_mods for a in sim.agents]
    assert all(m is not None for m in mods)
    cong = [m["congestion"] for m in mods]
    assert any(c > 1.0 for c in cong) and any(c < 1.0 for c in cong), cong
    # 少なくとも 3 reason で個体差(全員同値でない)
    varied = [r for r in drive.DEFAULT_WEIGHTS
              if len({m[r] for m in mods}) > 1]
    assert len(varied) >= 3, varied


# ============================================================ 5. OFF: バイト一致(発火列/閾値)
def test_needs_off_is_byte_identical(tmp_path):
    """needs_mods 不在(OFF)と全 reason=1.0 の恒等倍率で L1 が完全一致(seam が中立)。"""
    off = _sim(tmp_path, "off_absent", n=20, steps=48)
    off.run()
    ident = _sim(tmp_path, "off_identity", n=20, steps=48)
    for a in ident.agents:                       # 恒等倍率(全 reason 1.0)を明示付与
        a.needs_mods = {r: 1.0 for r in drive.DEFAULT_WEIGHTS}
    ident.run()
    assert _l1(off) == _l1(ident)


def test_needs_does_not_touch_threshold():
    """needs はゲージ入力の倍率のみ。effective_threshold は一切変えない。"""
    import types
    a = types.SimpleNamespace(drive_threshold=0.55, theta_drift=0.0,
                              needs_mods={"congestion": 2.0})
    assert drive.effective_threshold(a) == 0.55


# ============================================================ 6. ON で発火の顔ぶれ・分布が変わる
def test_needs_on_changes_firing(tmp_path):
    """同一 seed/world で ON と OFF を比べると L1 が変わり、個人別の発火分布も変わる。"""
    off = _sim(tmp_path, "fire_off", n=20, steps=48)
    off.run()
    on = _sim(tmp_path, "fire_on", n=20, steps=48)
    needs.precompute(on.agents, on.hub, needs.build_cfg({"enabled": True}))
    on.run()
    assert _l1(off) != _l1(on), "needs ON で L1 が変わらない(効いていない)"
    assert _fires_by_agent(off) != _fires_by_agent(on), "個人別発火分布が不変"


# ============================================================ 7. R1: k 非依存
def test_profile_is_k_independent(tmp_path):
    """同一 seed/agent なら k=free と k=off でプロファイル(needs_mods)が完全一致。"""
    def mods_for(mode):
        sim = _sim(tmp_path, f"k_{mode}", n=20, steps=1, **{"k.writeback": mode})
        needs.precompute(sim.agents, sim.hub, needs.build_cfg({"enabled": True}))
        return {a.id: a.needs_mods for a in sim.agents}
    free, off = mods_for("free"), mods_for("off")
    assert free == off, "k 条件でプロファイルが変わっている(R1 違反)"


def test_fire_count_stable_across_k_with_needs(tmp_path):
    """needs ON でも発火回数が k 間で偏らない(±20%、R1 監査)。"""
    fires = {}
    for mode in ("free", "off"):
        sim = _run_on(tmp_path, f"kfire_{mode}", n=20, steps=60,
                      **{"k.writeback": mode})
        fires[mode] = sum(1 for e in sim.logger.events
                          if e.kind == "drive_request" and e.payload["granted"])
    lo, hi = sorted(fires.values())
    assert lo > 0, fires
    assert hi <= lo * 1.2 + 3, f"needs ON で k 間の発火が乖離: {fires}(R1)"


# ============================================================ 8. 決定論 / 24step 完走
def test_needs_on_deterministic(tmp_path):
    a = _run_on(tmp_path, "det_a", n=20, steps=24)
    b = _run_on(tmp_path, "det_b", n=20, steps=24)
    assert _l1(a) == _l1(b), "needs ON が同 seed で非決定論"


def test_mock_24step_runs_on_and_off(tmp_path):
    on = _run_on(tmp_path, "smoke_on", n=20, steps=24)
    off = _sim(tmp_path, "smoke_off", n=20, steps=24)
    off.run()
    assert len(_l1(on)) > 50 and len(_l1(off)) > 50
