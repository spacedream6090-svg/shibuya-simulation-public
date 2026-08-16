"""物理痩身 第一段A(2026-08-16)— **軌跡ビット一致の同値変換**を機械固定する。

正典: docs/research/crowd-attention-physics.md 「案A: 同値セル法 + 観測層痩身」

何を証明するか
--------------
全力項には既に距離カットオフがある(対人 rr+2.0 m・far rr+4.725 m・知覚 2.0 m・
ORCA 10 m)のに候補列挙だけが O(N²) だった。それをセル法(`sfm_core.PointField`)へ
置換したが、**動力学は 1 ビットも変えていない**。その根拠は 3 つで、本ファイルは
そのすべてを機械で固定する:

  (a) 候補が「カットオフ内を必ず含む上位集合」= 実効近傍集合が完全一致
      → test_neighbor_pairs_is_a_superset_in_lexicographic_order
  (b) neighbor_cap 選抜の**タイブレークまで**一致(現行 argsort(stable) の
      (距離, index) 昇順を再現)→ 格子配置(距離が厳密に同値になる)で検査
  (c) 加算順序が現行と同一 — 候補を (i, j) 辞書順に整列してから合算する。
      現行の `contrib.sum(axis=1)` は軸 1 が最内周でないため**逐次加算**であり、
      `np.bincount`(index 昇順の逐次加算)と厳密に同じ和になる。
      → test_numpy_row_sum_is_sequential_like_bincount がこの前提自体を固定する。

比較の相手は「修正前の実装そのもの」である:
  - `sfm_core.Crowd._repulsion_dense`(旧 forces() の本体をそのまま残したもの)
  - `physics._CalibratedCrowd._far_forces_dense`(旧 _far_forces そのまま)
  - `orca_core.min_gap_dense` / `pair_hash=False` の分離パス・近傍選抜
  - `_accumulate` は旧実装を **本ファイル内に参照実装として保持**して突き合わせる
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from society import physics as P
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import orca_core as _orca, sfm_core as _sfm, zones

REPO = Path(__file__).resolve().parents[1]

# 密度 3 水準 [人/m²] × ゾーン形状 3 種 × seed 3 本(発注の検収条件)
DENSITIES = (0.3, 1.5, 3.0)
SHAPES = ("square", "wide", "clump")
SEEDS = (11, 22, 33)


def _layout(n, dens, shape, rng):
    """3 形状の配置。square=正方/wide=細長い帯/clump=偏った塊(密度が不均一)。"""
    area = n / dens
    if shape == "square":
        s = math.sqrt(area)
        return rng.uniform(0.0, s, size=(n, 2))
    if shape == "wide":
        h = math.sqrt(area / 8.0)
        return np.stack([rng.uniform(0.0, 8.0 * h, size=n),
                         rng.uniform(0.0, h, size=n)], axis=1)
    s = math.sqrt(area)
    k = n // 2
    return np.concatenate([rng.normal(s * 0.5, s * 0.06, size=(k, 2)),
                           rng.uniform(0.0, s, size=(n - k, 2))], axis=0)


def _crowd_pair(cls, n, dens, shape, seed, **kw):
    rng = np.random.default_rng(seed)
    pos = _layout(n, dens, shape, rng)
    vel = rng.normal(0.0, 0.4, size=(n, 2))
    goal = rng.uniform(-20.0, 60.0, size=(n, 2))
    v0 = rng.uniform(1.0, 1.6, size=n)
    radius = rng.uniform(_sfm.RADIUS_MIN, _sfm.RADIUS_MAX, size=n)
    args = (pos, vel, goal, v0)
    if cls is _orca.OrcaCrowd:
        a = cls(*args, radius, pair_hash=False, **kw)
        b = cls(*args, radius, pair_hash=True, **kw)
    else:
        a = cls(*args, radius=radius, pair_hash=False, **kw)
        b = cls(*args, radius=radius, pair_hash=True, **kw)
    active = rng.random(n) > 0.05
    a.active = active.copy()
    b.active = active.copy()
    return a, b


def _lattice(nx, ny, step):
    xs = np.arange(nx) * step
    ys = np.arange(ny) * step
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


# =========================================================================== #
# (0) 同値の前提そのもの — numpy の縮約が逐次加算であること
# =========================================================================== #
def test_numpy_row_sum_is_sequential_like_bincount():
    """`(N,N,2).sum(axis=1)` == index 昇順の逐次加算 == `np.bincount`。

    これが (c) の土台。軸 1 は最内周(連続軸)ではないので numpy は pairwise
    summation を使わず、`out += in` の逐次縮約になる。**この前提が崩れたら
    セル法の合算はビット一致しなくなる**ので、前提を直接固定しておく。
    """
    rng = np.random.default_rng(0)
    for n in (2, 5, 17, 64, 129):
        arr = rng.normal(size=(n, n, 2))
        ref = arr.sum(axis=1)
        seq = np.zeros((n, 2))
        for j in range(n):
            seq += arr[:, j, :]
        ii = np.repeat(np.arange(n), n)
        bc = np.stack([np.bincount(ii, weights=arr[:, :, 0].ravel(), minlength=n),
                       np.bincount(ii, weights=arr[:, :, 1].ravel(), minlength=n)],
                      axis=1)
        assert ref.tobytes() == seq.tobytes(), f"逐次加算でない(n={n})"
        assert ref.tobytes() == bc.tobytes(), f"bincount と一致しない(n={n})"


# =========================================================================== #
# (1) 候補列挙(PointField)そのものの性質
# =========================================================================== #
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_neighbor_pairs_is_a_superset_in_lexicographic_order(dens, shape):
    """(a) 候補は「reach 以内を必ず含む」上位集合 / (c) 並びは (i,j) 辞書順・重複なし。"""
    rng = np.random.default_rng(7)
    n = 400
    pos = _layout(n, dens, shape, rng)
    for reach in (0.5, 2.7, 5.5):
        ii, jj = _sfm.neighbor_pairs(pos, reach)
        assert np.all(ii != jj), "自己ペアが混じっている"
        flat = ii * n + jj
        assert np.all(np.diff(flat) > 0), "(i,j) 辞書順・重複なしでない"
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        want = np.argwhere(d <= reach)
        got = set(zip(ii.tolist(), jj.tolist()))
        missing = [tuple(p) for p in want if tuple(p) not in got]
        assert not missing, f"reach 内の取りこぼし {missing[:5]}"


def test_neighbor_pairs_matches_all_pairs_for_small_inputs():
    """小規模の安全弁(全ペア経路)でも並びは同じ辞書順。"""
    rng = np.random.default_rng(2)
    for n in (2, 3, 10):
        pos = rng.uniform(0, 5, size=(n, 2))
        ii, jj = _sfm.neighbor_pairs(pos, 100.0)
        ai, aj = _sfm.all_point_pairs(n)
        assert ii.tolist() == ai.tolist() and jj.tolist() == aj.tolist()


# =========================================================================== #
# (2) SFM: 力・速度・位置のビット一致(密度 3 × 形状 3 × seed 3)
# =========================================================================== #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_sfm_hash_matches_dense_bit_for_bit(dens, shape, seed):
    """全サブステップの forces / vel / pos が **1 バイトも違わない**。"""
    walls = [((-5.0, -5.0), (40.0, -5.0)), ((-5.0, -5.0), (-5.0, 40.0))]
    for cap in (None, 12):
        a, b = _crowd_pair(_sfm.Crowd, 300, dens, shape, seed,
                           neighbor_cap=cap, walls=walls)
        assert a.pair_hash is False and b.pair_hash is True
        for _ in range(8):
            fa, fb = a.forces(), b.forces()
            assert fa.tobytes() == fb.tobytes(), f"force 不一致 cap={cap}"
            a.step(0.05)
            b.step(0.05)
            assert a.pos.tobytes() == b.pos.tobytes(), f"pos 不一致 cap={cap}"
            assert a.vel.tobytes() == b.vel.tobytes(), f"vel 不一致 cap={cap}"


@pytest.mark.parametrize("nx,ny,step", [(10, 10, 0.55), (16, 9, 0.7), (7, 7, 1.4)])
@pytest.mark.parametrize("cap", [None, 3, 12])
def test_sfm_cap_tiebreak_matches_on_exact_distance_ties(nx, ny, step, cap):
    """(b) 格子配置 = 距離が厳密に同値。cap 選抜のタイブレークまで一致する。"""
    pos = _lattice(nx, ny, step)
    n = pos.shape[0]
    radius = np.full(n, 0.3)                     # 半径も同一 → 完全同値の距離が大量に出る
    kw = dict(radius=radius, neighbor_cap=cap)
    goal = pos + np.array([100.0, 0.0])
    a = _sfm.Crowd(pos, np.zeros((n, 2)), goal, np.full(n, 1.3), pair_hash=False, **kw)
    b = _sfm.Crowd(pos, np.zeros((n, 2)), goal, np.full(n, 1.3), pair_hash=True, **kw)
    for _ in range(10):
        assert a.forces().tobytes() == b.forces().tobytes()
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()


def test_sfm_degenerate_configurations_match():
    """完全重複・単独・遠距離 2 体など退化配置でもビット一致(0 除算経路を含む)。"""
    cases = [np.zeros((6, 2)),
             np.array([[0.0, 0.0], [0.0, 0.0], [80.0, 0.0]]),
             np.array([[0.0, 0.0], [60.0, 60.0]]),
             np.array([[0.0, 0.0]])]
    for pos in cases:
        n = pos.shape[0]
        kw = dict(radius=np.full(n, 0.3), neighbor_cap=12)
        goal = pos + np.array([10.0, 0.0])
        a = _sfm.Crowd(pos, np.zeros((n, 2)), goal, np.full(n, 1.2),
                       pair_hash=False, **kw)
        b = _sfm.Crowd(pos, np.zeros((n, 2)), goal, np.full(n, 1.2),
                       pair_hash=True, **kw)
        for _ in range(4):
            assert a.forces().tobytes() == b.forces().tobytes()
            a.step(0.05)
            b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes()


# =========================================================================== #
# (3) 較正 far 項(_CalibratedCrowd)のビット一致
# =========================================================================== #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_far_field_hash_matches_dense_bit_for_bit(dens, shape, seed):
    """finals の far_field(a2=0.119 / b2=1.890 / cutoff 2.5 / taper 1.0)で一致。"""
    kw = dict(neighbor_cap=12, far_a2=0.119, far_b2=1.890,
              far_cutoff_factor=2.5, far_taper_m=1.0)
    a, b = _crowd_pair(P._CalibratedCrowd, 250, dens, shape, seed, **kw)
    for _ in range(8):
        assert a._far_forces_dense().tobytes() == b._far_forces().tobytes()
        assert a.forces().tobytes() == b.forces().tobytes()
        a.step(0.05)
        b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes()
        assert a.vel.tobytes() == b.vel.tobytes()


def test_far_field_with_v_of_s_matches():
    """v_of_s(既定 OFF・本選でも OFF)を立てても一致する。"""
    kw = dict(neighbor_cap=12, far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5,
              far_taper_m=1.0, v_of_s=True, vos_T=0.482, vos_l=0.297)
    a, b = _crowd_pair(P._CalibratedCrowd, 200, 1.5, "square", 5, **kw)
    for _ in range(6):
        assert a.forces().tobytes() == b.forces().tobytes()
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()


# =========================================================================== #
# (4) ORCA: 近傍選抜 / 分離パス / min_gap
# =========================================================================== #
def _dense_nb(pos, all_pos, cap, ndist):
    n = pos.shape[0]
    m = all_pos.shape[0]
    dist = np.linalg.norm(pos[:, None, :] - all_pos[None, :, :], axis=2)
    dist[np.arange(n), np.arange(n)] = np.inf
    k = min(cap, max(m - 1, 1))
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(dist, order, axis=1) < ndist


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_knn_neighbors_matches_the_full_argsort_selection(dens, shape):
    """ORCA 近傍選抜: 有効近傍の**集合も順序も**全体 argsort 版と完全一致。"""
    rng = np.random.default_rng(4)
    n = 300
    pos = _layout(n, dens, shape, rng)
    ghosts = rng.uniform(-10.0, 40.0, size=(25, 2))
    all_pos = np.concatenate([pos, ghosts], axis=0)
    for cap, nd in ((12, 10.0), (4, 3.0), (12, 0.8), (40, 10.0)):
        o1, v1 = _dense_nb(pos, all_pos, cap, nd)
        o2, v2 = _sfm.knn_neighbors(pos, all_pos, min(cap, max(all_pos.shape[0] - 1, 1)),
                                    nd)
        for i in range(n):
            want = [int(x) for x, ok in zip(o1[i], v1[i]) if ok]
            got = [int(x) for x, ok in zip(o2[i], v2[i]) if ok]
            assert want == got, f"近傍列が違う i={i} cap={cap} nd={nd}"


@pytest.mark.parametrize("nx,ny,step", [(14, 14, 0.7), (18, 6, 1.1)])
def test_knn_neighbors_tiebreak_on_lattice(nx, ny, step):
    """(b) 距離が厳密同値でも (距離, index) 昇順のタイブレークが一致する。"""
    pos = _lattice(nx, ny, step)
    for cap, nd in ((12, 10.0), (6, 2.0), (5, 1.0)):
        o1, v1 = _dense_nb(pos, pos, cap, nd)
        o2, v2 = _sfm.knn_neighbors(pos, pos, min(cap, pos.shape[0] - 1), nd)
        for i in range(pos.shape[0]):
            want = [int(x) for x, ok in zip(o1[i], v1[i]) if ok]
            got = [int(x) for x, ok in zip(o2[i], v2[i]) if ok]
            assert want == got, f"タイブレーク不一致 i={i}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_orca_hash_matches_dense_bit_for_bit(dens, shape, seed):
    """ORCA の pos / vel / min_gap / 分離反復数が全サブステップで一致。"""
    walls = [((-5.0, -5.0), (60.0, -5.0))]
    a, b = _crowd_pair(_orca.OrcaCrowd, 220, dens, shape, seed, walls=walls,
                       neighbor_cap=12, neighbor_dist=10.0, separation_iters=64)
    for _ in range(6):
        a.step(0.05)
        b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes(), "pos 不一致"
        assert a.vel.tobytes() == b.vel.tobytes(), "vel 不一致"
        assert a.last_sep_iters == b.last_sep_iters
        assert a.last_sep_moved_m == b.last_sep_moved_m
        assert a.min_gap() == b.min_gap()


def test_separate_positions_resolution_order_is_unchanged():
    """重なりが実際に連鎖する密集で、逐次解消の**順序込み**で一致する。"""
    rng = np.random.default_rng(5)
    pos = rng.normal(0.0, 0.45, size=(140, 2))
    radius = np.full(140, 0.3)
    vel = rng.normal(0.0, 0.5, size=(140, 2))
    ref = _orca.separate_positions(pos, radius, max_iters=64, vel=vel, pair_hash=False)
    got = _orca.separate_positions(pos, radius, max_iters=64, vel=vel, pair_hash=True)
    assert ref[0].tobytes() == got[0].tobytes()
    assert ref[1] == got[1] > 0, "重なりが起きていない = テストが空回り"
    assert ref[2] == got[2]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", (0.01, 0.3, 3.0))
def test_min_gap_matches_the_full_pair_value(dens, shape):
    """min_gap は疎な配置(探索半径の拡大が起きる)でも全ペア版と厳密一致。"""
    rng = np.random.default_rng(6)
    n = 300
    pos = _layout(n, dens, shape, rng)
    radius = rng.uniform(0.25, 0.35, size=n)
    assert _orca.min_gap(pos, radius) == _orca.min_gap_dense(pos, radius)


def test_min_gap_on_a_regular_lattice():
    lat = _lattice(20, 20, 3.0)
    r = np.full(lat.shape[0], 0.3)
    assert _orca.min_gap(lat, r) == _orca.min_gap_dense(lat, r)


# =========================================================================== #
# (5) 観測層 `_accumulate` — 旧実装(参照)との一致
# =========================================================================== #
def _accumulate_legacy(zone, members, engine, prev_pos, prev_vel, dt, gate_xy,
                       cont, st, pcfg):
    """痩身**前**の `physics._accumulate`(2026-08-16 以前)をそのまま写した参照実装。"""
    pos = engine.pos
    vel = engine.vel
    dv = np.linalg.norm(vel - prev_vel, axis=1) / dt
    disp = np.linalg.norm(pos - prev_pos, axis=1)
    cont["jump_max_m"] = max(cont["jump_max_m"], float(disp.max()))
    for i, rec in enumerate(members):
        band = "gate" if zone.near_gate(float(pos[i, 0]), float(pos[i, 1]),
                                        gate_xy) else "interior"
        b = cont[band]
        idx = int(dv[i] / P._ACC_BIN)
        b["hist"][min(idx, P._ACC_BINS - 1)] += 1
        b["n"] += 1
        b["samples"] += 1
        ex, ey = rec["seg_dir"]
        s = vel[i, 0] * ex + vel[i, 1] * ey
        sign = 1 if s > 0 else (-1 if s < 0 else 0)
        if sign and rec["sign"] and sign != rec["sign"]:
            b["flip"] += 1
        if sign:
            rec["sign"] = sign
    radius = np.array([r["radius"] for r in members], dtype=np.float64)
    gap = _orca.min_gap_dense(pos, radius)
    if math.isfinite(gap):
        st["min_gap_m"] = gap if st["min_gap_m"] is None else min(st["min_gap_m"], gap)
    st["sep_iters_max"] = max(st["sep_iters_max"],
                              int(getattr(engine, "last_sep_iters", 0)))
    speed = np.linalg.norm(vel, axis=1)
    if len(members) > 1:
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        rr = radius[:, None] + radius[None, :]
    else:
        d = None
    dens_r = float(pcfg["density_radius_m"])
    gap_m = float(pcfg["contact_gap_m"])
    for i, rec in enumerate(members):
        rec["speed_sum"] += float(speed[i])
        rec["speed_n"] += 1
        rec["step_n"] += 1
        if d is not None:
            rec["dens_sum"] += int((d[i] < dens_r).sum())
            if bool((d[i] < rr[i] + gap_m).any()):
                rec["contact_n"] += 1


class _FakeEngine:
    def __init__(self, pos, vel, sep=0):
        self.pos = pos
        self.vel = vel
        self.last_sep_iters = sep


def _fake_members(n, rng):
    out = []
    for _ in range(n):
        ang = float(rng.uniform(0, 2 * math.pi))
        out.append({"radius": float(rng.uniform(0.25, 0.35)),
                    "seg_dir": (math.cos(ang), math.sin(ang)),
                    "sign": int(rng.integers(-1, 2)),
                    "speed_sum": 0.0, "speed_n": 0, "step_n": 0,
                    "contact_n": 0, "dens_sum": 0})
    return out


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
@pytest.mark.parametrize("n", (1, 2, 200))
def test_accumulate_matches_the_pre_slimming_implementation(n, dens, shape):
    """帯・ヒスト・反転・min_gap・密度・接触・速度和が **すべて同値**。"""
    rng = np.random.default_rng(hash((n, dens, shape)) % (2 ** 31))
    zone = zones.build_cfg(OmegaConf.create({
        "zones_enabled": True,
        "zones": [{"id": "z", "engine": "sfm", "dt_sub": 0.05,
                   "polygon": [[-60, -60], [60, -60], [60, 60], [-60, 60]]}],
    }), REPO)["zones"][0]
    gate_xy = ((0.0, 0.0), (12.0, 4.0), (-8.0, 9.0))
    pcfg = {"density_radius_m": 2.0, "contact_gap_m": 0.05}
    members_a = _fake_members(n, np.random.default_rng(1))
    members_b = copy.deepcopy(members_a)
    cont_a, cont_b = P._new_cont(), P._new_cont()
    st_a, st_b = P._new_state(), P._new_state()
    for _ in range(5):
        pos = _layout(n, dens, shape, rng) - 20.0
        prev_pos = pos + rng.normal(0.0, 0.02, size=(n, 2))
        vel = rng.normal(0.0, 0.6, size=(n, 2))
        prev_vel = vel + rng.normal(0.0, 0.3, size=(n, 2))
        eng_a = _FakeEngine(pos, vel, sep=3)
        eng_b = _FakeEngine(pos, vel, sep=3)
        _accumulate_legacy(zone, members_a, eng_a, prev_pos, prev_vel, 0.05,
                           gate_xy, cont_a, st_a, pcfg)
        P._accumulate(zone, members_b, eng_b, prev_pos, prev_vel, 0.05,
                      gate_xy, cont_b, st_b, pcfg)
    assert cont_a == cont_b, "境界連続性の集計が違う"
    assert st_a["min_gap_m"] == st_b["min_gap_m"]
    assert st_a["sep_iters_max"] == st_b["sep_iters_max"]
    assert members_a == members_b, "個体別の身体観測が違う"


@pytest.mark.parametrize("shape", SHAPES)
def test_near_gate_mask_matches_the_scalar_predicate(shape):
    zone = zones.build_cfg(OmegaConf.create({
        "zones_enabled": True,
        "zones": [{"id": "z", "engine": "sfm", "dt_sub": 0.05,
                   "polygon": [[-60, -60], [60, -60], [60, 60], [-60, 60]]}],
    }), REPO)["zones"][0]
    rng = np.random.default_rng(8)
    pos = _layout(300, 1.5, shape, rng) - 15.0
    gate_xy = ((0.0, 0.0), (5.0, -3.0), (-11.0, 7.0))
    got = P._near_gate_mask(zone, pos, gate_xy)
    want = np.array([zone.near_gate(float(pos[i, 0]), float(pos[i, 1]), gate_xy)
                     for i in range(pos.shape[0])])
    assert got.tolist() == want.tolist()


def test_admit_blocked_matches_the_brute_force_scan():
    """入場ゲートの占有判定(旧 O(W·M) 総当たり)と bool が完全一致。"""
    rng = np.random.default_rng(12)
    for n_m, n_w in ((0, 5), (10, 8), (120, 60), (400, 200)):
        members = [{"pos": tuple(p), "radius": float(r)}
                   for p, r in zip(rng.uniform(0, 30, size=(n_m, 2)),
                                   rng.uniform(0.25, 0.35, size=n_m))]
        queue = [{"pos": tuple(p), "radius": float(r)}
                 for p, r in zip(rng.uniform(0, 30, size=(n_w, 2)),
                                 rng.uniform(0.25, 0.35, size=n_w))]
        gap = 0.10
        want = []
        for rec in queue:
            px, py = rec["pos"]
            hit = False
            for other in members:
                ox, oy = other["pos"]
                need = rec["radius"] + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    hit = True
                    break
            want.append(hit)
        got = P._admit_blocked(queue, members, gap)
        if got is None:
            assert n_m <= P._ADMIT_HASH_MIN
        else:
            assert got.tolist() == want


# =========================================================================== #
# (6) ゾーン物理 ON の mock 短ランで L1 バイト一致
# =========================================================================== #
_ZR = 25.0
_POLY = [[-_ZR, -_ZR], [_ZR, -_ZR], [_ZR, _ZR], [-_ZR, _ZR]]
_FAR_ON = {"enabled": True, "a2": 0.119, "b2": 1.890,
           "cutoff_factor": 2.5, "taper_m": 1.0}


def _zone_spec(zid, engine):
    return {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}


def _run_zones(tmp_path, name, pair_hash, monkeypatch, engine="orca", steps=6, n=40):
    """`pair_hash` を強制したうえでゾーン物理 ON の mock ランを回す。"""
    real_crowd = _sfm.Crowd
    real_calib = P._CalibratedCrowd
    real_orca = _orca.OrcaCrowd

    class Crowd(real_crowd):
        def __init__(self, *a, **kw):
            kw["pair_hash"] = pair_hash
            super().__init__(*a, **kw)

    class Calib(real_calib):
        def __init__(self, *a, **kw):
            kw["pair_hash"] = pair_hash
            super().__init__(*a, **kw)

    class Orca(real_orca):
        def __init__(self, *a, **kw):
            kw["pair_hash"] = pair_hash
            super().__init__(*a, **kw)

    monkeypatch.setattr(P._sfm, "Crowd", Crowd)
    monkeypatch.setattr(P, "_CalibratedCrowd", Calib)
    monkeypatch.setattr(P._orca, "OrcaCrowd", Orca)
    cfg = load_config(["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
                       f"run.name={name}", "model.backend=mock",
                       "observer.snapshot_every=144"])
    OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
    cfg.physics.zones = [_zone_spec("z1", engine)]
    OmegaConf.update(cfg, "physics.sfm.far_field", _FAR_ON, force_add=True)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind, e.x, e.y,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


@pytest.mark.parametrize("engine", ("orca", "sfm"))
def test_zone_run_l1_is_byte_identical_between_hash_and_dense(tmp_path, monkeypatch,
                                                              engine):
    """ゾーン物理 ON の mock 短ラン: セル法 ON/OFF で **L1 が完全一致**。

    SFM ゾーンでは finals と同じ far_field(P4-2)を立てた状態で比較する。
    """
    a = _run_zones(tmp_path, f"zh_dense_{engine}", False, monkeypatch, engine=engine)
    b = _run_zones(tmp_path, f"zh_fast_{engine}", True, monkeypatch, engine=engine)
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)
    assert P.scalars(a) == P.scalars(b)


# =========================================================================== #
# (7) 計算量: 人数 2 倍で時間が ~2 倍(O(N·k))に近づく
# =========================================================================== #
def _time_forces(n, dens, reps=3):
    rng = np.random.default_rng(99)
    pos = _layout(n, dens, "square", rng)
    c = _sfm.Crowd(pos, np.zeros((n, 2)), pos + 30.0, np.full(n, 1.3),
                   radius=rng.uniform(0.25, 0.35, size=n), neighbor_cap=12)
    c.forces()                                   # ウォームアップ
    t0 = time.perf_counter()
    for _ in range(reps):
        c.forces()
    return (time.perf_counter() - t0) / reps


@pytest.mark.slow
def test_scaling_is_close_to_linear_in_population():
    """密度を保ったまま人数 2 倍 → 時間はおよそ 2 倍(緩い係数で確認)。

    旧実装は O(N²)(cap 込みで O(N² log N))なので、この不等式は原理的に通らない。
    """
    dens = 2.0
    t1 = _time_forces(2000, dens)
    t2 = _time_forces(4000, dens)
    t3 = _time_forces(8000, dens)
    assert t2 / t1 < 3.2, f"2000→4000 が線形から外れている ({t2 / t1:.2f}x)"
    assert t3 / t2 < 3.2, f"4000→8000 が線形から外れている ({t3 / t2:.2f}x)"
    assert t3 / t1 < 8.0, f"2000→8000 が線形から外れている ({t3 / t1:.2f}x)"
