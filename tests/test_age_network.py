"""AGE-D: 社会ネットワーク次数の年齢曲線(第116バッチ 2026-08-15・既定 OFF)。

正典: docs/plans/age-diversity-plan.md §4-6。実装: src/society/friends.py。

本テストが機械固定すること:
- **既定 OFF はバイト一致**(倍率が厳密に 1.0 = 次数計算が恒等)。
- **曲線の形**が文献どおり(Bhattacharya 2016: 25 歳ピーク・45-55 台地・55 以降再減少)。
- ★**加齢が削るのは外層だけ**: ON にしても **親友(tier3)の人数は 1 人も変わらない**。
  減るのは友人(tier2)と知人(tier1)だけ。
- **乱数ゼロ**(friend_graph は全 hashlib。ON でも draw 順が動かない)。
"""
from __future__ import annotations

import json

import pytest

from society import friends
from society.config import load_config
from society.engine.simulation import Simulation


def _cfg(**ov):
    return friends.build_cfg({"enabled": True, "age_degree": True, **ov})


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------- OFF
def test_off_multiplier_is_exactly_one():
    """OFF では倍率が**厳密に** 1.0(float の 1.0 == 恒等 = 次数の式を通らない)。"""
    cfg = friends.build_cfg({"enabled": True})     # age_degree 未指定 = False
    for a in (6, 19, 25, 45, 70, 90):
        assert friends.age_degree_mult(a, cfg) == 1.0


def test_off_matches_pure_default(tmp_path):
    """friend_graph ON かつ age_degree OFF が、age_degree キー未指定と L1 完全一致。"""
    base = _sim(tmp_path, "aged_base", steps=72, **{"friend_graph.enabled": "true"})
    base.run()
    off = _sim(tmp_path, "aged_off", steps=72,
               **{"friend_graph.enabled": "true", "friend_graph.age_degree": "false"})
    off.run()
    assert _l1(base) == _l1(off), "age_degree=false が既定と不一致(seam が no-op でない)"


# --------------------------------------------------------------------- 曲線の形
def test_curve_peaks_at_25_and_plateaus_45_to_55():
    """Bhattacharya 2016 の形: 25 歳が最大 / 45-55 が台地 / 55 以降また減少。"""
    cfg = _cfg()
    m = {a: friends.age_degree_mult(a, cfg) for a in range(15, 86)}
    assert max(m, key=lambda a: m[a]) == 25, "ピークが 25 歳でない(逆 J 字が崩れた)"
    assert m[25] == pytest.approx(1.0), "ref_age での規格化が壊れている"
    assert m[45] == pytest.approx(m[50]) == pytest.approx(m[55]), "45-55 が台地でない"
    assert m[70] < m[55] and m[85] < m[70], "55 以降の再減少が無い"
    assert m[15] < m[25], "青年期がピークを超えている"
    # 範囲外は端の値で平ら(外挿して偽の精度を作らない)
    assert friends.age_degree_mult(6, cfg) == friends.age_degree_mult(15, cfg)
    assert friends.age_degree_mult(99, cfg) == friends.age_degree_mult(85, cfg)


def test_curve_is_clipped():
    """min/max のクリップが効く(暴走防止)。"""
    cfg = _cfg(age_degree_min=0.9, age_degree_max=0.95)
    for a in (15, 25, 85):
        assert 0.9 <= friends.age_degree_mult(a, cfg) <= 0.95


# --------------------------------------------------- ★内核は触らない(形の制約)
def test_close_tier_is_never_touched(tmp_path):
    """★AGE-D ON で **親友(tier3)の人数は 1 人も変わらない**。減るのは tier2/tier1 だけ。

    「加齢が削るのは外層(周辺・同僚)で内核ではない」= 4 つの独立ソースが一致する所見
    (Bruine de Bruin: 周辺 r=−.13 / 親友 r=.01 / English & Carstensen 2014 / Wrzus 2013)。
    """
    def tiers(sim):
        out = {}
        for a in sim.agents:
            if a.visitor:
                continue
            c = {1: 0, 2: 0, 3: 0}
            for rel in a.mem.relations.values():
                t = int(rel.get("tier", 0))
                if t in c:
                    c[t] += 1
            out[a.id] = c
        return out

    ov = {"friend_graph.enabled": "true", "relations.enabled": "true"}
    off = _sim(tmp_path, "aged_tier_off", n=60, steps=1, **ov)
    on = _sim(tmp_path, "aged_tier_on", n=60, steps=1,
              **{**ov, "friend_graph.age_degree": "true"})
    t_off, t_on = tiers(off), tiers(on)
    assert t_off and t_on
    for aid, c in t_off.items():
        assert t_on[aid][3] == c[3], \
            f"agent {aid} の親友(tier3)が動いた: {c[3]} -> {t_on[aid][3]}"
    weak_off = sum(c[1] + c[2] for c in t_off.values())
    weak_on = sum(c[1] + c[2] for c in t_on.values())
    assert weak_on != weak_off, "弱紐帯が 1 本も動いていない(AGE-D が無風)"


def test_on_is_deterministic_and_draws_nothing(tmp_path):
    """ON 同士 2 回で L1 完全一致(friend_graph は全 hashlib = 乱数 stream を引かない)。"""
    ov = {"friend_graph.enabled": "true", "friend_graph.age_degree": "true"}
    a = _sim(tmp_path, "aged_on_a", steps=72, **ov)
    a.run()
    b = _sim(tmp_path, "aged_on_b", steps=72, **ov)
    b.run()
    assert _l1(a) == _l1(b)
