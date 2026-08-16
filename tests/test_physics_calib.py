"""P4-2 / P4-3 歩行物理較正(`physics.sfm.far_field` / `v_of_s` / `wall`)のテスト。

正典: docs/research/p4-calibration-research.md §4(Tordeux 型 V(s))・§6.1(2 項構造)
      docs/plans/if-sv-p4-plan.md P4-2 / P4-3 行
      reference/physics_bench/out/calib_results.json(P4-1)
      reference/physics_bench/out/calib_p43_results.json(P4-3)

方針(R1 の鉄則を継承):
- OFF(既定): conf にキーが増えても **既定ランは 1 バイトも変わらない**。
  `_calib_kwargs()` が空 dict = `_build_engine` は素の `sfm_core.Crowd` を
  従来の引数のまま構築する(派生クラスすら作らない)。
- ON: 同 seed 2 ラン一致 / 1 サブステップ変位 ≤ v_max·dt / 壁貫通ゼロ /
  較正した壁で w=1.0 m(および 0.9 m)の開口を単独歩行者が通れる。
- **重い較正はここでは回さない**。受入数値は `reference/physics_bench` 側の
  `calib_p43_results.json` が正典で、ここが固定するのは「不変条件」だけ。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

from society import physics as P
from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import sfm_core as _sfm, zones

REPO = Path(__file__).resolve().parents[1]

# P4-3 較正値(reference/physics_bench/out/calib_p43_results.json)。
# **conf の既定ではない**(既定は現行値 0.08 のまま)。ON 経路の検査にだけ使う。
CAL_WALL_B = 0.03          # 較正ハーネスの推奨(README §9。保守側の別解は 0.06)
FAR_ON = {"enabled": True, "a2": 0.119, "b2": 1.890,
          "cutoff_factor": 2.5, "taper_m": 1.0}
VOS_ON = {"enabled": True, "T": 0.482, "l": 0.297}

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R], [-_ZONE_R, _ZONE_R]]


def _sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _zone(zid="z1", engine="sfm", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _cfg(name, n=20, steps=6, zone_specs=None, calib=None, drop_calib_keys=False):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    cfg = load_config(dot)
    if drop_calib_keys:
        # 「P4 で足した conf キーが 1 つも無かった世界」= 昇格前の conf と同じ形
        for k in ("far_field", "v_of_s", "wall"):
            cfg.physics.sfm.pop(k, None)
    if calib is not None:
        for k, v in calib.items():
            OmegaConf.update(cfg, f"physics.sfm.{k}", v, force_add=True)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
    return cfg


def _run(tmp_path, name, **kw):
    sim = Simulation(_cfg(name, **kw), out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _members(pos, vel, goal, v0, radius):
    return [{"pos": tuple(p), "vel": tuple(v), "wp_xy": tuple(g),
             "v0": float(s), "radius": float(r)}
            for p, v, g, s, r in zip(pos, vel, goal, v0, radius)]


# =========================================================================== #
# ① 既定 OFF = golden L1 バイト一致(conf にキーが増えても既定ランは無風)
# =========================================================================== #
def test_default_off_is_byte_identical_to_pre_p4_conf(tmp_path):
    """`physics.sfm.{far_field,v_of_s,wall}` を **conf から丸ごと消しても** L1 が同一。

    = P4-2/P4-3 で足した conf キーの既定値が現行挙動と完全に恒等であることの証明。
    """
    with_keys = _run(tmp_path, "calib_keys", steps=6)
    without = _run(tmp_path, "calib_nokeys", steps=6, drop_calib_keys=True)
    assert _l1(with_keys) == _l1(without)
    # 既定では較正引数が 1 つも立たない = 従来の sfm_core.Crowd 経路
    assert P._calib_kwargs(with_keys) == {}
    assert P.calib_describe(with_keys) == {}
    # conf を消した側も同じ既定へ正準化される
    assert without.physcfg["sfm"]["far_field"]["enabled"] is False
    assert without.physcfg["sfm"]["v_of_s"]["enabled"] is False
    assert without.physcfg["sfm"]["wall"] == {"a": _sfm.WALL_A_DEFAULT,
                                              "b": _sfm.WALL_B_DEFAULT}


def test_conf_defaults_equal_the_core_defaults():
    """conf の既定値が `sfm_core` の既定値と一致する(片方だけ動かす事故の検出)。"""
    raw = yaml.safe_load((REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))
    s = raw["physics"]["sfm"]
    assert s["far_field"]["enabled"] is False
    assert s["v_of_s"]["enabled"] is False
    # 新設 `wall:` ブロックと竹-3 の宣言 wall_A / wall_B が食い違っていないこと
    assert float(s["wall"]["a"]) == float(s["wall_A"]) == _sfm.WALL_A_DEFAULT
    assert float(s["wall"]["b"]) == float(s["wall_B"]) == _sfm.WALL_B_DEFAULT


# =========================================================================== #
# ② OFF で力計算が既存と数値恒等(軌跡 SHA-256 一致)
# =========================================================================== #
def _scenario(cls, **kw):
    """壁つき通路 40 体を 200 サブステップ回す小シナリオ(決定論・乱数ゼロ)。"""
    n = 40
    rng = np.random.default_rng(20260805)
    pos = rng.uniform(0.0, 10.0, size=(n, 2))
    vel = rng.normal(0.0, 0.3, size=(n, 2))
    goal = pos + np.array([20.0, 0.0])
    v0 = 1.0 + 0.4 * (np.arange(n) % 5) / 4.0
    radius = 0.25 + 0.10 * (np.arange(n) % 3) / 2.0
    walls = [((-5.0, 0.0), (25.0, 0.0)), ((-5.0, 6.0), (25.0, 6.0))]
    c = cls(pos=pos.copy(), vel=vel.copy(), goal=goal.copy(), v0=v0, radius=radius,
            walls=walls, neighbor_cap=12, **kw)
    for _ in range(200):
        c.step(0.05)
    return c


def test_off_path_builds_the_plain_core_and_is_bit_identical():
    """較正 OFF では `_CalibratedCrowd` を作らず、作っても素の Crowd とバイト一致。"""
    cfgp = zones.build_cfg({"zones_enabled": True, "zones": [_zone()]}, REPO)
    zone = cfgp["zones"][0]
    pos = np.array([[0.0, 0.0], [2.0, 0.5]])
    mem = _members(pos, np.zeros((2, 2)), pos + np.array([10.0, 0.0]),
                   [1.2, 1.1], [0.30, 0.32])
    eng_off = P._build_engine(zone, mem, None, {})
    assert type(eng_off) is _sfm.Crowd, "OFF で派生クラスが作られている"
    eng_on = P._build_engine(zone, mem, None, {"far_a2": 0.119})
    assert type(eng_on) is P._CalibratedCrowd

    base = _scenario(_sfm.Crowd)
    same = _scenario(P._CalibratedCrowd)                       # 追加引数すべて既定 OFF
    same2 = _scenario(P._CalibratedCrowd, far_a2=0.0, v_of_s=False,
                      far_b2=1.89, vos_T=0.3, vos_l=0.4)
    assert _sha(base.pos, base.vel) == _sha(same.pos, same.vel)
    assert _sha(base.pos, base.vel) == _sha(same2.pos, same2.vel)
    # 壁パラメータを既定値と同値で明示しても変わらない(= conf 既定の恒等性)
    explicit = _scenario(_sfm.Crowd, wall_a=_sfm.WALL_A_DEFAULT,
                         wall_b=_sfm.WALL_B_DEFAULT)
    assert _sha(base.pos, base.vel) == _sha(explicit.pos, explicit.vel)


def test_on_changes_the_trajectory():
    """ON は実際に軌跡を変える(= テストが「何も起きない」を通してしまわない)。"""
    base = _scenario(_sfm.Crowd)
    far = _scenario(P._CalibratedCrowd, far_a2=0.119, far_b2=1.890)
    vos = _scenario(P._CalibratedCrowd, v_of_s=True, vos_T=0.3, vos_l=0.4)
    wall = _scenario(_sfm.Crowd, wall_b=CAL_WALL_B)
    hs = {_sha(c.pos, c.vel) for c in (base, far, vos, wall)}
    assert len(hs) == 4, "ON が軌跡を変えていない(または衝突している)"


def test_src_matches_the_calibration_bench_bit_for_bit():
    """昇格版が `reference/physics_bench` の ExtendedCrowd と **バイト一致**。

    = 「較正で得た数値がそのまま本体で再現される」ことの機械的な証明
      (test_physics_zones.py の ORCA 昇格検査と同じ流儀)。
    """
    import importlib
    import sys as _sys
    if str(REPO) not in _sys.path:
        _sys.path.insert(0, str(REPO))
    bench = importlib.import_module("reference.physics_bench.engines")
    cases = (
        (dict(a2=0.119, b2=1.890, lambda_aniso=0.5, lambda_far=0.5,
              cutoff_mode="per_term", far_taper=1.0),
         dict(far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5, far_taper_m=1.0)),
        (dict(a2=0.0, v_of_s=True, vos_T=0.3, vos_l=0.4),
         dict(v_of_s=True, vos_T=0.3, vos_l=0.4)),
        (dict(a2=0.119, b2=1.890, lambda_aniso=0.5, lambda_far=0.5,
              cutoff_mode="per_term", far_taper=1.0, v_of_s=True, vos_T=0.3,
              vos_l=0.4, wall_b=CAL_WALL_B),
         dict(far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5, far_taper_m=1.0,
              v_of_s=True, vos_T=0.3, vos_l=0.4, wall_b=CAL_WALL_B)),
    )
    for bench_kw, src_kw in cases:
        a = _scenario(bench.ExtendedCrowd, **bench_kw)
        b = _scenario(P._CalibratedCrowd, **src_kw)
        assert a.pos.tobytes() == b.pos.tobytes(), f"位置が違う: {src_kw}"
        assert a.vel.tobytes() == b.vel.tobytes(), f"速度が違う: {src_kw}"


# =========================================================================== #
# ③ ON: 同 seed 2 ラン一致
# =========================================================================== #
def test_on_same_seed_two_runs_are_identical(tmp_path):
    calib = {"far_field": FAR_ON, "v_of_s": VOS_ON,
             "wall": {"a": 2000.0, "b": CAL_WALL_B}}
    a = _run(tmp_path, "cal_on_a", steps=4, zone_specs=[_zone()], calib=calib)
    b = _run(tmp_path, "cal_on_b", steps=4, zone_specs=[_zone()], calib=calib)
    assert _l1(a) == _l1(b)
    assert [e for e in a.logger.events if e.kind == "zone_gate"], \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    # 較正が実際に効いている(引数が立っている)
    kw = P._calib_kwargs(a)
    assert kw["far_a2"] == 0.119 and kw["v_of_s"] is True and kw["wall_b"] == CAL_WALL_B
    d = P.calib_describe(a)
    assert set(d) == {"far_field", "v_of_s", "wall"}


def test_the_zone_engine_class_follows_the_conf(monkeypatch):
    """ラン中に実際に構築されるエンジンのクラスが conf に従う(配線の直接検査)。

    OFF: `sfm_core.Crowd`(**派生クラスを一度も作らない**)/ ON: `_CalibratedCrowd`。
    """
    seen: list[type] = []
    orig = P._build_engine

    def spy(zone, members, rng, calib=None, cog=None):
        # cog = 第二段B(認知的近傍)の引数。既定 OFF では常に空 dict。
        eng = orig(zone, members, rng, calib, cog)
        seen.append(type(eng))
        return eng

    monkeypatch.setattr(P, "_build_engine", spy)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        seen.clear()
        Simulation(_cfg("cls_off", steps=4, zone_specs=[_zone()]),
                   out_dir=base / "off").run()
        assert seen and set(seen) == {_sfm.Crowd}, f"OFF で {set(seen)} が作られた"
        seen.clear()
        Simulation(_cfg("cls_on", steps=4, zone_specs=[_zone()],
                        calib={"far_field": FAR_ON, "v_of_s": VOS_ON,
                               "wall": {"a": 2000.0, "b": CAL_WALL_B}}),
                   out_dir=base / "on").run()
        assert seen and set(seen) == {P._CalibratedCrowd}, f"ON で {set(seen)} が作られた"


# =========================================================================== #
# ④ 1 サブステップ変位 ≤ v_max·dt(較正 ON でも壊れない不変条件)
# =========================================================================== #
def test_one_substep_displacement_never_exceeds_vmax_dt():
    """v_max クリップは較正 ON でも効く(V(s) は希望速度だけを下げる)。"""
    n, dt = 60, 0.05
    rng = np.random.default_rng(7)
    pos = np.stack([rng.uniform(0.5, 12.0, n), rng.uniform(0.6, 2.4, n)], axis=1)
    v0 = 1.0 + 0.4 * (np.arange(n) % 5) / 4.0
    radius = 0.25 + 0.10 * (np.arange(n) % 3) / 2.0
    walls = [((-2.0, 0.0), (20.0, 0.0)), ((-2.0, 3.0), (20.0, 3.0))]
    for kw in ({}, {"far_a2": 0.119, "far_b2": 1.890},
               {"v_of_s": True, "vos_T": 0.3, "vos_l": 0.4},
               {"far_a2": 0.119, "far_b2": 1.890, "v_of_s": True,
                "vos_T": 0.3, "vos_l": 0.4, "wall_b": CAL_WALL_B}):
        c = P._CalibratedCrowd(pos=pos.copy(), vel=np.zeros((n, 2)),
                               goal=pos + np.array([40.0, 0.0]), v0=v0,
                               radius=radius, walls=walls, neighbor_cap=12, **kw)
        limit = c.v_max * dt + 1e-12
        for _ in range(400):
            prev = c.pos.copy()
            c.step(dt)
            disp = np.linalg.norm(c.pos - prev, axis=1)
            assert (disp <= limit).all(), f"{kw}: 変位が v_max·dt を超えた"


def test_v_of_s_only_lowers_the_desired_speed():
    """V(s) ≤ v0 が常に成り立ち、前方が空いていれば v0 に戻る(定義そのもの)。"""
    v0 = np.array([1.2, 1.2])
    radius = np.array([0.30, 0.30])
    # 追従列(後続 0 の 0.6 m 前方に 1 = ほぼ体が触れている)
    pos = np.array([[0.0, 0.0], [0.6, 0.0]])
    c = P._CalibratedCrowd(pos=pos, vel=np.zeros((2, 2)),
                           goal=pos + np.array([10.0, 0.0]), v0=v0, radius=radius,
                           v_of_s=True, vos_T=0.3, vos_l=0.4)
    s = c.front_spacing()
    assert np.isclose(s[0], 0.6) and not np.isfinite(s[1]), "前方判定が壊れている"
    v = c.v_of_s_speed()
    assert np.isclose(v[0], (0.6 - 0.4) / 0.3), "V(s) が式どおりでない"
    assert v[0] < v0[0] and v[1] == v0[1], "前方が空いている側まで減速している"
    assert (v <= v0 + 1e-12).all()
    # s ≥ l + T·v0 なら v0 に戻る(頭打ち)
    free = np.array([[0.0, 0.0], [5.0, 0.0]])
    c3 = P._CalibratedCrowd(pos=free, vel=np.zeros((2, 2)),
                            goal=free + np.array([10.0, 0.0]), v0=v0, radius=radius,
                            v_of_s=True, vos_T=0.3, vos_l=0.4)
    assert (c3.v_of_s_speed() == v0).all()
    # 横にずれた相手(体幅より外)は「前方」に数えない
    side = np.array([[0.0, 0.0], [0.8, 1.5]])
    c2 = P._CalibratedCrowd(pos=side, vel=np.zeros((2, 2)),
                            goal=side + np.array([10.0, 0.0]), v0=v0, radius=radius,
                            v_of_s=True, vos_T=0.3, vos_l=0.4)
    assert not np.isfinite(c2.front_spacing()[0])


def test_far_field_falls_to_zero_continuously_at_the_cutoff():
    """長距離項は打ち切り点で **不連続を作らない**(C¹ テーパー。Köster 2013 対策)。"""
    a2, b2, factor, taper = 0.119, 1.890, 2.5, 1.0
    cutoff = factor * b2
    radius = np.array([0.30, 0.30])
    mags = []
    gaps = np.linspace(cutoff - taper - 0.2, cutoff + 0.3, 60)
    for gap in gaps:
        pos = np.array([[0.0, 0.0], [0.60 + gap, 0.0]])
        c = P._CalibratedCrowd(pos=pos, vel=np.zeros((2, 2)),
                               goal=pos + np.array([10.0, 0.0]),
                               v0=np.array([1.2, 1.2]), radius=radius,
                               far_a2=a2, far_b2=b2, far_cutoff_factor=factor,
                               far_taper_m=taper)
        mags.append(float(np.linalg.norm(c._far_forces()[0])))
    mags = np.array(mags)
    assert mags[gaps >= cutoff].max() == 0.0, "打ち切りの外側で力が残っている"
    assert mags[0] > 0.0
    # 隣接サンプル間の跳びが最大値の 10% 未満 = 垂直に切り落としていない
    assert np.abs(np.diff(mags)).max() < 0.10 * mags.max()


# =========================================================================== #
# ⑤ 壁貫通ゼロ(較正 ON)
# =========================================================================== #
def test_no_wall_penetration_with_calibration_on():
    """壁の向こう側に goal を置いて押し当て続けても、誰も通路の外へ出ない。"""
    # 幾何・人数・seed は tests/test_sfm_walls.py の同名検査と揃える(既定で通る条件)
    length, width, n, dt = 20.0, 3.0, 40, 0.05
    walls = ([((k * 2.5, 0.0), ((k + 1) * 2.5, 0.0)) for k in range(8)]
             + [((k * 2.5, width), ((k + 1) * 2.5, width)) for k in range(8)]
             + [((length, 0.0), (length, width))])
    rng = np.random.default_rng(11)
    pos = np.stack([rng.uniform(1.0, 8.0, n), rng.uniform(0.6, width - 0.6, n)], axis=1)
    radius = rng.uniform(0.25, 0.35, n)
    v0 = np.full(n, 1.34)
    goal = np.tile(np.array([length + 20.0, width / 2.0]), (n, 1))
    for kw in ({"wall_b": CAL_WALL_B},
               {"wall_b": CAL_WALL_B, "far_a2": 0.119, "far_b2": 1.890,
                "v_of_s": True, "vos_T": 0.3, "vos_l": 0.4}):
        c = P._CalibratedCrowd(pos=pos.copy(), vel=np.zeros((n, 2)), goal=goal,
                               v0=v0, radius=radius, walls=walls, neighbor_cap=12,
                               **kw)
        for _ in range(600):
            c.step(dt)
        assert (c.pos[:, 1] > 0.0).all() and (c.pos[:, 1] < width).all(), \
            f"{kw}: 側壁を貫通した"
        assert (c.pos[:, 0] < length).all(), f"{kw}: 終端壁を貫通した"


# =========================================================================== #
# ⑥ 開口の通過可能性(較正した壁)
# =========================================================================== #
def _opening_walls(w, room_len=8.0, room_width=5.0, exit_len=2.0):
    lo, hi = (room_width - w) / 2.0, (room_width + w) / 2.0
    return [((0.0, 0.0), (0.0, room_width)),
            ((0.0, 0.0), (room_len, 0.0)),
            ((0.0, room_width), (room_len, room_width)),
            ((room_len, 0.0), (room_len, lo)),
            ((room_len, hi), (room_len, room_width)),
            ((room_len, lo), (room_len + exit_len + 1.0, lo)),
            ((room_len, hi), (room_len + exit_len + 1.0, hi))]


def _single_passes(w, wall_b, t_total=20.0, dt=0.05):
    """単独歩行者が幅 w の開口を通れるか(P4-1 が見つけた過剰クリアランスの対照)。"""
    room_len, room_width = 8.0, 5.0
    y_c = room_width / 2.0
    pos = np.array([[1.0, y_c]])
    c = P._CalibratedCrowd(pos=pos, vel=np.zeros((1, 2)),
                           goal=np.array([[room_len + 1.0, y_c]]),
                           v0=np.array([1.2]), radius=np.array([0.33]),
                           walls=_opening_walls(w, room_len, room_width),
                           neighbor_cap=12, wall_b=wall_b)
    for _ in range(int(round(t_total / dt))):
        if c.pos[0, 0] >= room_len:
            c.goal[0] = [c.pos[0, 0] + 50.0, c.pos[0, 1]]
        c.step(dt)
        if c.pos[0, 0] > room_len + 1.5:
            return True
    return False


def test_calibrated_wall_opens_the_narrow_gate():
    """w=1.0 m は既定でも通れる。**w=0.9 m は較正した壁でだけ**通れる(P4-3 の狙い)。"""
    assert _single_passes(1.0, _sfm.WALL_B_DEFAULT), "w=1.0 m が既定で通れない"
    assert _single_passes(1.0, CAL_WALL_B), "w=1.0 m が較正値で通れない"
    # P4-1 実測: 既定 B_w=0.08 は片側 +0.15 m の余分なクリアランスを要求する
    assert not _single_passes(0.9, _sfm.WALL_B_DEFAULT), \
        "既定で w=0.9 m が通れてしまう(P4-1 の実測と食い違う)"
    assert _single_passes(0.9, CAL_WALL_B), "較正値でも w=0.9 m が通れない"


# =========================================================================== #
# 宣言・検証
# =========================================================================== #
def test_registry_declares_the_calibration_toggles():
    ids = {f.id: f for f in R.FEATURES}
    for fid in ("physics.sfm.far_field.enabled", "physics.sfm.v_of_s.enabled"):
        f = ids[fid]
        assert f.repro_tier == "strict"
        assert f.affects_k is False
        assert f.fingerprint_risk == "none"
        assert f.off_value is False


def test_build_cfg_rejects_bad_calibration():
    for bad in ({"far_field": {"enabled": True, "a2": 0.0}},
                {"far_field": {"enabled": True, "b2": 0.0}},
                {"v_of_s": {"enabled": True, "T": 0.0}},
                {"wall": {"b": 0.0}}):
        try:
            zones.build_cfg({"sfm": bad}, REPO)
            raise AssertionError(f"不正な較正が素通りした: {bad}")
        except ValueError:
            pass
    try:
        zones.build_cfg({"sfm": {"far_field": {"nope": 1}}}, REPO)
        raise AssertionError("未知キーが素通りした")
    except KeyError:
        pass
