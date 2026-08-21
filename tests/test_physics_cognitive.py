"""物理痩身 第二段B/C — 認知的近傍 と 遠方場の密度場化。

正典: docs/research/crowd-attention-physics.md 「案B」「案C」
実装: src/society/world/sfm_core.py `visual_neighbors` / `Crowd._repulsion_cognitive`
      src/society/world/orca_core.py `OrcaCrowd.step`(選抜だけ差し替え)
      src/society/physics.py `_CalibratedCrowd._far_forces_field`

本ファイルが機械固定するもの
----------------------------
(1) **既定 OFF は 1 バイトも変えない**(R1)。conf キーを丸ごと消したランと L1 が一致し、
    エンジン側でも新引数を既定値で明示しても軌跡がバイト一致する。
(2) B の選抜が **定義そのもの**(視野円錐 → セクタ最前 → 距離昇順 k)と一致する。
    比較相手は本ファイル内の素朴な二重ループ参照実装で、格子配置(距離が厳密同値)の
    タイブレークまで突き合わせる。
(3) B の**実効近傍が密度に依らず有界**(≤ min(k, sectors))であること。
(4) C の接続係数が解析形どおりで、**一様密度ではペア版と同じ合力**になること
    (= 基本図の作業点が保存される、という案C の設計主張そのもの)。
(5) ON でも決定論(同一入力 → 同一 float64・同 seed 2 ラン一致)で、乱数消費が増えないこと。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from society import physics as P
from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import orca_core as _orca, sfm_core as _sfm, zones

REPO = Path(__file__).resolve().parents[1]

DENSITIES = (0.5, 2.0, 6.0)
SHAPES = ("square", "wide", "clump")
FAR_ON = {"enabled": True, "a2": 0.119, "b2": 1.890,
          "cutoff_factor": 2.5, "taper_m": 1.0}
FAR_KW = dict(far_a2=0.119, far_b2=1.890, far_cutoff_factor=2.5, far_taper_m=1.0)

_ZR = 25.0
_POLY = [[-_ZR, -_ZR], [_ZR, -_ZR], [_ZR, _ZR], [-_ZR, _ZR]]


def _layout(n, dens, shape, rng):
    """tests/test_physics_hash.py と同じ 3 形状(square / wide / clump)。"""
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


def _lattice(nx, ny, step):
    xs = np.arange(nx) * step
    ys = np.arange(ny) * step
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _unit_dirs(pos, goal):
    d = goal - pos
    n = np.linalg.norm(d, axis=1)
    out = np.zeros_like(d)
    nz = n > 1e-12
    out[nz] = d[nz] / n[nz, None]
    return out


# =========================================================================== #
# (0) 参照実装 — 「見えている近傍」の定義をそのまま二重ループで書いたもの
# =========================================================================== #
def _visual_reference(qpos, points, edir, k, max_dist, sectors, fov_deg,
                      exclude_self=True):
    """定義そのもの: ① |φ| ≤ fov/2 ② 各セクタの最前 1 体 ③ 距離昇順 k 体。

    タイブレークは (距離, 点 index) の昇順。**全ペアを列挙して素朴に選ぶだけ**の参照実装で、
    固定するのは「候補列挙(セル法・早期終了)と選抜の論理」であって浮動小数の丸めではない。
    そのため距離・方位の**算術式は実装と同一のもの**を使う(`np.linalg.norm` と
    逆数掛け)。ここを `math.hypot` や `du/d` に変えると 1 ulp の差でセクタ境界の
    振り分けが揺れ、テストが論理ではなく丸めを見てしまう。
    """
    nq, m = qpos.shape[0], points.shape[0]
    half = min(max(0.5 * math.radians(float(fov_deg)), 1e-6), math.pi)
    cos_half = math.cos(half)
    en = np.linalg.norm(edir, axis=1)
    e = np.where(en[:, None] > 1e-12,
                 edir / np.maximum(en, 1e-12)[:, None],
                 np.array([1.0, 0.0]))
    ii = np.repeat(np.arange(nq, dtype=np.int64), m)
    jj = np.tile(np.arange(m, dtype=np.int64), nq)
    if exclude_self:
        keep = ~((ii == jj) & (jj < nq))
        ii, jj = ii[keep], jj[keep]
    du = points[jj] - qpos[ii]
    d = np.linalg.norm(du, axis=1)
    inv = np.where(d > 1e-12, 1.0 / np.maximum(d, 1e-12), 0.0)
    ux, uy = du[:, 0] * inv, du[:, 1] * inv
    cs = e[ii, 0] * ux + e[ii, 1] * uy
    sn = e[ii, 0] * uy - e[ii, 1] * ux
    seen = (cs >= cos_half) & (d < max_dist)
    ii, jj, d = ii[seen], jj[seen], d[seen]
    phi = np.arctan2(sn[seen], cs[seen])
    sec = np.clip(np.floor((phi + half) / (2.0 * half) * sectors),
                  0, sectors - 1).astype(np.int64)
    best: list[dict[int, tuple[float, int]]] = [{} for _ in range(nq)]
    for p in range(ii.shape[0]):
        row = best[int(ii[p])]
        key = int(sec[p])
        cand = (float(d[p]), int(jj[p]))
        cur = row.get(key)
        if cur is None or cand < cur:
            row[key] = cand
    return [[j for _, j in sorted(row.values())[:k]] for row in best]


def _selected(order, valid):
    return [[int(x) for x, ok in zip(order[i], valid[i]) if ok]
            for i in range(order.shape[0])]


# =========================================================================== #
# (1) B: 選抜が定義と一致する
# =========================================================================== #
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_visual_neighbors_matches_the_definition(dens, shape):
    rng = np.random.default_rng(21)
    n = 240
    pos = _layout(n, dens, shape, rng)
    e = _unit_dirs(pos, rng.uniform(-20.0, 60.0, size=(n, 2)))
    for k, sectors, fov, md in ((12, 16, 200.0, 2.7), (12, 16, 200.0, 10.0),
                                (6, 12, 360.0, 2.7), (12, 16, 90.0, 4.0),
                                (3, 16, 200.0, 2.7), (12, 4, 200.0, 2.7)):
        got = _selected(*_sfm.visual_neighbors(pos, pos, e, k, md,
                                               sectors=sectors, fov_deg=fov))
        want = _visual_reference(pos, pos, e, k, md, sectors, fov)
        assert got == want, f"選抜が定義と違う k={k} S={sectors} fov={fov} md={md}"


@pytest.mark.parametrize("nx,ny,step", [(14, 14, 0.7), (18, 6, 1.1), (10, 10, 0.55)])
def test_visual_neighbors_tiebreak_on_exact_distance_ties(nx, ny, step):
    """格子配置 = 距離が厳密に同値。セクタ最前・k 選抜のタイブレークまで定義どおり。"""
    pos = _lattice(nx, ny, step)
    n = pos.shape[0]
    for e in (np.tile(np.array([1.0, 0.0]), (n, 1)),
              np.tile(np.array([1.0, 1.0]) / math.sqrt(2.0), (n, 1))):
        got = _selected(*_sfm.visual_neighbors(pos, pos, e, 12, 2.7))
        want = _visual_reference(pos, pos, e, 12, 2.7,
                                 _sfm.COG_SECTORS_DEFAULT, _sfm.COG_FOV_DEG_DEFAULT)
        assert got == want


def test_visual_neighbors_handles_degenerate_inputs():
    """完全重複・単独・向きゼロ・遠方だけ、でも落ちずに定義どおり。"""
    cases = [np.zeros((6, 2)),
             np.array([[0.0, 0.0], [0.0, 0.0], [80.0, 0.0]]),
             np.array([[0.0, 0.0], [60.0, 60.0]]),
             np.array([[0.0, 0.0]])]
    for pos in cases:
        n = pos.shape[0]
        for e in (np.zeros((n, 2)), np.tile(np.array([0.0, 1.0]), (n, 1))):
            got = _selected(*_sfm.visual_neighbors(pos, pos, e, 12, 2.7))
            want = _visual_reference(pos, pos, e, 12, 2.7,
                                     _sfm.COG_SECTORS_DEFAULT,
                                     _sfm.COG_FOV_DEG_DEFAULT)
            assert got == want


def test_visual_neighbors_with_ghost_points():
    """query と points が別集合(ORCA のゴースト込み配列)でも定義どおり。"""
    rng = np.random.default_rng(33)
    pos = _layout(200, 2.0, "square", rng)
    ghosts = rng.uniform(-8.0, 20.0, size=(30, 2))
    allp = np.concatenate([pos, ghosts], axis=0)
    e = _unit_dirs(pos, rng.uniform(-20.0, 40.0, size=(200, 2)))
    got = _selected(*_sfm.visual_neighbors(pos, allp, e, 12, 10.0))
    want = _visual_reference(pos, allp, e, 12, 10.0,
                             _sfm.COG_SECTORS_DEFAULT, _sfm.COG_FOV_DEG_DEFAULT)
    assert got == want


# =========================================================================== #
# (2) B: 実効近傍が密度に依らず有界(発注の「有界であることをテストで固定」)
# =========================================================================== #
@pytest.mark.parametrize("dens", (0.5, 1.0, 2.0, 4.0, 6.0, 10.0))
def test_cognitive_neighborhood_is_bounded_regardless_of_density(dens):
    """力を及ぼす相手は **必ず min(k, sectors) 体以下**(密度を 20 倍しても変わらない)。

    ★これが案B の要点。距離カットオフだけの旧経路は候補数が ρπR² で増え続ける。
    """
    rng = np.random.default_rng(int(dens * 100) + 5)
    n = 1500
    pos = _layout(n, dens, "square", rng)
    radius = rng.uniform(_sfm.RADIUS_MIN, _sfm.RADIUS_MAX, size=n)
    e = _unit_dirs(pos, pos + np.array([50.0, 0.0]))
    reach = 2.0 * float(radius.max()) + _sfm.CUTOFF_M
    k, sectors = 12, 16
    _order, valid = _sfm.visual_neighbors(pos, pos, e, k, reach,
                                          sectors=sectors, fov_deg=200.0)
    cnt = valid.sum(axis=1)
    assert cnt.max() <= min(k, sectors), "上限 min(k, sectors) を超えた"
    # 旧経路(距離カットオフのみ)は同じ配置で密度に比例して増える = 対照
    ii, _jj = _sfm.neighbor_pairs(pos, reach)
    raw = np.bincount(ii, minlength=n).max()
    assert raw >= cnt.max(), "参照(距離カットオフのみ)より多く選んでいる"


@pytest.mark.slow
def test_cognitive_per_agent_cost_plateaus_in_density():
    """1 体あたりの forces() 時間が高密度側で **頭打ち**になる(旧経路は増え続ける)。

    ★正直な読み方: 有界になるのは「遮蔽の地平 horizon が距離カットオフの内側に入って
      から」で、それより疎な領域ではカットオフが binding するので旧経路と同じ増え方をする。
      ここが固定するのは ρ=4 → ρ=10(2.5 倍)で新経路の単価が **1.3 倍以内**に収まること。
    """
    def _t(crowd, reps=3):
        crowd.forces()
        t0 = time.perf_counter()
        for _ in range(reps):
            crowd.forces()
        return (time.perf_counter() - t0) / reps / crowd.pos.shape[0]

    rng = np.random.default_rng(77)
    n = 3000
    got = {}
    for dens in (4.0, 10.0):
        pos = _layout(n, dens, "square", rng)
        radius = rng.uniform(_sfm.RADIUS_MIN, _sfm.RADIUS_MAX, size=n)
        kw = dict(pos=pos, vel=np.zeros((n, 2)), goal=pos + np.array([50.0, 0.0]),
                  v0=np.full(n, 1.3), radius=radius, neighbor_cap=12)
        got[dens] = (_t(_sfm.Crowd(**kw)), _t(_sfm.Crowd(**kw, cognitive=True)))
    off_ratio = got[10.0][0] / got[4.0][0]
    cog_ratio = got[10.0][1] / got[4.0][1]
    assert cog_ratio < 1.3, f"認知的近傍が密度で伸びている ({cog_ratio:.2f}x)"
    assert cog_ratio < off_ratio, f"旧経路より伸びが悪い ({cog_ratio:.2f} vs {off_ratio:.2f})"


# =========================================================================== #
# (3) B: 既定 OFF は 1 バイトも変わらない / ON は実際に変わる / 決定論
# =========================================================================== #
def _sfm_pair(cls, n, dens, shape, seed, **kw):
    rng = np.random.default_rng(seed)
    pos = _layout(n, dens, shape, rng)
    vel = rng.normal(0.0, 0.4, size=(n, 2))
    goal = rng.uniform(-20.0, 60.0, size=(n, 2))
    v0 = rng.uniform(1.0, 1.6, size=n)
    radius = rng.uniform(_sfm.RADIUS_MIN, _sfm.RADIUS_MAX, size=n)
    return dict(pos=pos, vel=vel, goal=goal, v0=v0, radius=radius, **kw)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dens", DENSITIES)
def test_cognitive_off_is_bit_identical_to_the_previous_path(dens, shape):
    """`cognitive=False` なら他の新引数を何に設定しても軌跡がビット一致(= 完全な no-op)。"""
    walls = [((-5.0, -5.0), (40.0, -5.0)), ((-5.0, -5.0), (-5.0, 40.0))]
    kw = _sfm_pair(_sfm.Crowd, 250, dens, shape, 11, neighbor_cap=12, walls=walls)
    a = _sfm.Crowd(**kw)
    b = _sfm.Crowd(**kw, cognitive=False, cog_neighbors=3, cog_sectors=5,
                   cog_fov_deg=33.0)
    for _ in range(8):
        assert a.forces().tobytes() == b.forces().tobytes()
        a.step(0.05)
        b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes()
        assert a.vel.tobytes() == b.vel.tobytes()


def test_cognitive_on_changes_the_trajectory():
    """ON が実際に軌跡を変える(= テストが『何も起きない』を通してしまわない)。"""
    kw = _sfm_pair(_sfm.Crowd, 200, 2.0, "square", 4, neighbor_cap=12)
    a = _sfm.Crowd(**kw)
    b = _sfm.Crowd(**kw, cognitive=True)
    for _ in range(20):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() != b.pos.tobytes(), "認知的近傍が力を 1 ビットも変えていない"


@pytest.mark.parametrize("dens", DENSITIES)
def test_cognitive_is_deterministic(dens):
    """同一入力 → 同一 float64 配列(2 個体を並走させて全サブステップ突合)。"""
    kw = _sfm_pair(_sfm.Crowd, 200, dens, "clump", 9, neighbor_cap=12,
                   cognitive=True)
    a, b = _sfm.Crowd(**kw), _sfm.Crowd(**kw)
    for _ in range(10):
        assert a.forces().tobytes() == b.forces().tobytes()
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()


def _orca_kw(n, dens, seed, **kw):
    rng = np.random.default_rng(seed)
    pos = _layout(n, dens, "square", rng)
    return dict(pos=pos, vel=rng.normal(0.0, 0.3, size=(n, 2)),
                goal=rng.uniform(-20.0, 60.0, size=(n, 2)),
                v0=rng.uniform(1.0, 1.6, size=n),
                radius=rng.uniform(0.25, 0.35, size=n), **kw)


def test_orca_cognitive_off_is_bit_identical():
    kw = _orca_kw(200, 2.0, 12, neighbor_cap=12, neighbor_dist=10.0,
                  separation_iters=64)
    a = _orca.OrcaCrowd(**kw)
    b = _orca.OrcaCrowd(**kw, cognitive=False, cog_neighbors=4, cog_sectors=6,
                        cog_fov_deg=45.0)
    for _ in range(6):
        a.step(0.05)
        b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes()
        assert a.vel.tobytes() == b.vel.tobytes()


def test_orca_cognitive_on_changes_the_trajectory_and_stays_deterministic():
    kw = _orca_kw(180, 2.0, 13, neighbor_cap=12, neighbor_dist=10.0,
                  separation_iters=64)
    base = _orca.OrcaCrowd(**kw)
    a = _orca.OrcaCrowd(**kw, cognitive=True)
    b = _orca.OrcaCrowd(**kw, cognitive=True)
    for _ in range(6):
        base.step(0.05)
        a.step(0.05)
        b.step(0.05)
        assert a.pos.tobytes() == b.pos.tobytes(), "決定論が壊れている"
    assert base.pos.tobytes() != a.pos.tobytes(), "選抜を変えたのに軌跡が同じ"


def test_orca_cognitive_consumes_no_extra_random_draws():
    """★選抜の希望方向は goal−pos から作る = `pref_noise` の draw 順が 1 ビットも動かない。"""
    tail = []
    for cog in (False, True):
        rng = np.random.default_rng(4242)
        c = _orca.OrcaCrowd(**_orca_kw(120, 2.0, 15, neighbor_cap=12,
                                       neighbor_dist=10.0, pref_noise=0.05,
                                       rng=rng, separation_iters=32),
                            cognitive=cog)
        for _ in range(6):
            c.step(0.05)
        tail.append(rng.normal(size=3).tolist())
    assert tail[0] == tail[1], "認知的近傍が乱数を余分に消費している"


# =========================================================================== #
# (4) C: 接続係数と「一様密度でペア版と一致」
# =========================================================================== #
def _calib(n, dens, shape, seed, **kw):
    rng = np.random.default_rng(seed)
    pos = _layout(n, dens, shape, rng)
    return P._CalibratedCrowd(pos=pos, vel=np.zeros((n, 2)),
                              goal=pos + np.array([100.0, 0.0]),
                              v0=np.full(n, 1.3),
                              radius=rng.uniform(0.25, 0.35, size=n),
                              neighbor_cap=12, **FAR_KW, **kw)


def test_far_field_coefficients_follow_the_analytic_form():
    """c_drag / c_grad が class docstring の I₁ / I₂ 式と一致する(独立に積分し直す)。"""
    c = _calib(300, 2.0, "square", 3, far_mode="field")
    rr = 2.0 * float(c.radius.mean())
    big_r = rr + c.far_cutoff
    r = np.linspace(0.0, big_r, 200001)[1:]
    gap = r - rr
    kern = c.mass * c.far_a2 * np.exp(np.clip(-gap / c.far_b2, None, 4.0))
    u = np.clip((c.far_cutoff - gap) / c.far_taper, 0.0, 1.0)
    kern = np.where(gap <= c.far_cutoff, kern * (u * u * (3.0 - 2.0 * u)), 0.0)
    i1 = float(np.trapezoid(kern * r, r))
    i2 = float(np.trapezoid(kern * r * r, r))
    assert c._c_drag == pytest.approx((1.0 - c.lam) * math.pi / 2.0 * i1, rel=1e-4)
    assert c._c_grad == pytest.approx((1.0 + c.lam) * math.pi / 2.0 * i2, rel=1e-4)
    assert c._c_drag > 0.0 and c._c_grad > 0.0


@pytest.mark.parametrize("dens", (1.0, 2.0, 3.0))
def test_field_far_matches_the_pair_far_in_a_uniform_crowd(dens):
    """★案C の設計主張: 一様密度では場版とペア版の合力が一致する(= FD の作業点が保存)。

    縁の効果を外すため、広い一様矩形の**中央部**の個体だけで平均する。
    """
    side = 40.0
    n = int(round(dens * side * side))
    rng = np.random.default_rng(int(dens * 10) + 1)
    pos = rng.uniform(0.0, side, size=(n, 2))
    kw = dict(pos=pos, vel=np.zeros((n, 2)), goal=pos + np.array([100.0, 0.0]),
              v0=np.full(n, 1.3), radius=rng.uniform(0.25, 0.35, size=n),
              neighbor_cap=12)
    pair = P._CalibratedCrowd(**kw, **FAR_KW)
    field = P._CalibratedCrowd(**kw, **FAR_KW, far_mode="field")
    core = ((pos[:, 0] > 8.0) & (pos[:, 0] < side - 8.0)
            & (pos[:, 1] > 8.0) & (pos[:, 1] < side - 8.0))
    fp = pair._far_forces()[core, 0].mean()
    ff = field._far_forces_field()[core, 0].mean()
    assert fp < 0.0, "ペア版の far 項が進行方向の抵抗になっていない(前提が壊れた)"
    assert ff / fp == pytest.approx(1.0, abs=0.12), \
        f"一様密度で合力が合わない pair={fp:.3f} field={ff:.3f}"


def test_density_grid_conserves_the_head_count():
    """密度格子は人数を保存する(平滑してもゼロ詰めの外へ漏れない)。"""
    for blur in (0, 1, 2, 4):
        c = _calib(400, 2.0, "clump", 6, far_mode="field", field_blur=blur)
        rho, _gx, _gy, _ox, _oy = c._density_grid()
        assert float(rho.sum()) * c.field_cell_m ** 2 == pytest.approx(400.0, rel=1e-9)


def test_field_far_pushes_out_of_a_density_peak():
    """密度の山の中では**外向き**の力が出る(勾配項の符号の直接確認)。"""
    rng = np.random.default_rng(8)
    blob = rng.normal(0.0, 1.6, size=(400, 2))
    far = rng.uniform(-30.0, 30.0, size=(40, 2))
    pos = np.concatenate([blob, far], axis=0)
    n = pos.shape[0]
    c = P._CalibratedCrowd(pos=pos, vel=np.zeros((n, 2)),
                           goal=pos + np.array([100.0, 0.0]), v0=np.full(n, 1.3),
                           radius=np.full(n, 0.3), neighbor_cap=12,
                           **FAR_KW, far_mode="field")
    f = c._far_forces_field()
    ring = (np.linalg.norm(pos, axis=1) > 1.0) & (np.linalg.norm(pos, axis=1) < 3.0)
    radial = (f[ring, 0] * pos[ring, 0] + f[ring, 1] * pos[ring, 1]) \
        / np.linalg.norm(pos[ring], axis=1)
    assert float(radial.mean()) > 0.0, "密度の山から外向きに押されていない"


def test_far_mode_pair_is_the_untouched_path():
    """既定 `far_mode="pair"` は第一段A までと同じ経路(場の格子を 1 度も作らない)。"""
    a = _calib(200, 2.0, "square", 2)
    assert a.far_mode == "pair" and a._c_drag == 0.0 and a._c_grad == 0.0
    for _ in range(5):
        a.step(0.05)
    assert a._field_grid is None, "pair 経路で密度格子が作られている"
    b = _calib(200, 2.0, "square", 2, far_mode="field")
    for _ in range(5):
        b.step(0.05)
    assert b._field_grid is not None
    assert a.pos.tobytes() != b.pos.tobytes()


def test_field_update_every_is_deterministic_and_caches():
    """粗い更新周期でも決定論。格子の作り直し回数は `update_every` に従う。"""
    made = []
    for every in (1, 10):
        c = _calib(200, 2.0, "square", 5, far_mode="field", field_update_every=every)
        orig = c._density_grid
        cnt = [0]

        def spy(_orig=orig, _cnt=cnt):
            _cnt[0] += 1
            return _orig()

        c._density_grid = spy
        for _ in range(20):
            c.step(0.05)
        made.append(cnt[0])
    assert made == [20, 2], f"格子の作り直し回数が周期どおりでない {made}"
    a = _calib(200, 2.0, "square", 5, far_mode="field", field_update_every=10)
    b = _calib(200, 2.0, "square", 5, far_mode="field", field_update_every=10)
    for _ in range(20):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()


@pytest.mark.slow
def test_field_far_cost_is_flat_in_density():
    """C の far 項は O(N + セル数) = **密度に対して平坦**(ペア版は ρ に比例して伸びる)。"""
    def _t(fn, n, reps=5):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps / n

    n = 3000
    pair_t, field_t = {}, {}
    for dens in (1.0, 6.0):
        c1 = _calib(n, dens, "square", 4)
        c2 = _calib(n, dens, "square", 4, far_mode="field", field_update_every=1)
        pair_t[dens] = _t(c1._far_forces, n)
        field_t[dens] = _t(c2._far_forces_field, n)
    assert field_t[6.0] / field_t[1.0] < 1.5, "密度で密度場の単価が伸びている"
    assert pair_t[6.0] / pair_t[1.0] > 1.8, "ペア版が伸びていない(対照が壊れた)"
    assert field_t[6.0] < pair_t[6.0] / 10.0, "密度場が 1 桁も速くない"


# =========================================================================== #
# (5) conf / registry / 正準化
# =========================================================================== #
def test_conf_defaults_are_off():
    raw = yaml.safe_load((REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))
    p = raw["physics"]
    assert p["cognitive"]["enabled"] is False
    assert p["density_far"]["enabled"] is False
    # 既定値がコア側の既定と食い違っていないこと(片方だけ動かす事故の検出)
    assert int(p["cognitive"]["neighbors"]) == _sfm.COG_NEIGHBORS_DEFAULT
    assert int(p["cognitive"]["sectors"]) == _sfm.COG_SECTORS_DEFAULT
    assert float(p["cognitive"]["fov_deg"]) == _sfm.COG_FOV_DEG_DEFAULT
    assert float(p["density_far"]["cell_m"]) == P.FIELD_CELL_M_DEFAULT
    assert int(p["density_far"]["blur"]) == P.FIELD_BLUR_DEFAULT
    assert int(p["density_far"]["update_every"]) == P.FIELD_UPDATE_EVERY_DEFAULT


def test_registry_declares_the_second_stage_toggles():
    ids = {f.id: f for f in R.FEATURES}
    for fid in ("physics.cognitive.enabled", "physics.density_far.enabled"):
        f = ids[fid]
        assert f.repro_tier == "strict"
        assert f.affects_k is False
        assert f.fingerprint_risk == "none"
        assert f.off_value is False


def test_build_cfg_rejects_bad_values():
    for bad in ({"cognitive": {"enabled": True, "neighbors": 0}},
                {"cognitive": {"enabled": True, "sectors": 0}},
                {"cognitive": {"enabled": True, "fov_deg": 0.0}},
                {"cognitive": {"enabled": True, "fov_deg": 400.0}},
                {"density_far": {"enabled": True, "cell_m": 0.0}},
                {"density_far": {"enabled": True, "update_every": 0}}):
        with pytest.raises(ValueError):
            zones.build_cfg(bad, REPO)
    with pytest.raises(KeyError):
        zones.build_cfg({"cognitive": {"nope": 1}}, REPO)
    with pytest.raises(KeyError):
        zones.build_cfg({"density_far": {"nope": 1}}, REPO)


def test_kwargs_helpers_are_empty_when_off():
    class _Sim:
        physcfg = zones.build_cfg({}, REPO)

    assert P._cog_kwargs(_Sim()) == {}
    assert P.calib_describe(_Sim()) == {}


def test_kwargs_helpers_carry_the_conf_when_on():
    class _Sim:
        physcfg = zones.build_cfg(
            {"sfm": {"far_field": dict(FAR_ON)},
             "cognitive": {"enabled": True, "neighbors": 8, "sectors": 12,
                           "fov_deg": 180.0},
             "density_far": {"enabled": True, "cell_m": 1.5, "blur": 1,
                             "update_every": 4}}, REPO)

    sim = _Sim()
    assert P._cog_kwargs(sim) == {"cognitive": True, "cog_neighbors": 8,
                                  "cog_sectors": 12, "cog_fov_deg": 180.0}
    kw = P._calib_kwargs(sim)
    assert kw["far_mode"] == "field" and kw["field_cell_m"] == 1.5
    assert kw["field_blur"] == 1 and kw["field_update_every"] == 4
    d = P.calib_describe(sim)
    assert set(d) == {"far_field", "cognitive", "density_far"}


def test_density_far_is_a_no_op_without_far_field():
    """置換する当の項(far_field)が OFF なら密度場の引数は 1 つも立たない。"""
    class _Sim:
        physcfg = zones.build_cfg(
            {"density_far": {"enabled": True}}, REPO)

    assert P._calib_kwargs(_Sim()) == {}


# =========================================================================== #
# (6) ゾーン物理 ON の mock 短ラン
# =========================================================================== #
def _cfg(name, n=40, steps=6, engine="orca", cognitive=None, density_far=None,
         far=False, drop_keys=False):
    cfg = load_config(["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
                       f"run.name={name}", "model.backend=mock",
                       "observer.snapshot_every=144"])
    if drop_keys:
        for k in ("cognitive", "density_far"):
            cfg.physics.pop(k, None)
    if cognitive is not None:
        OmegaConf.update(cfg, "physics.cognitive", cognitive, force_add=True)
    if density_far is not None:
        OmegaConf.update(cfg, "physics.density_far", density_far, force_add=True)
    if far:
        OmegaConf.update(cfg, "physics.sfm.far_field", dict(FAR_ON), force_add=True)
    OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
    cfg.physics.zones = [{"id": "z1", "engine": engine, "dt_sub": 0.05,
                          "polygon": list(_POLY)}]
    return cfg


def _l1(sim):
    return [[e.step, e.agent_id, e.kind, e.x, e.y,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name, **kw):
    sim = Simulation(_cfg(name, **kw), out_dir=tmp_path / name)
    sim.run()
    return sim


@pytest.mark.parametrize("engine", ("orca", "sfm"))
def test_zone_run_off_is_byte_identical_without_the_new_conf_keys(tmp_path, engine):
    """★R1: 新しい conf キーを **丸ごと消しても** L1 が一致する = 既定 OFF は完全 no-op。"""
    a = _run(tmp_path, f"cog_keys_{engine}", engine=engine, far=True)
    b = _run(tmp_path, f"cog_nokeys_{engine}", engine=engine, far=True,
             drop_keys=True)
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)
    assert b.physcfg["cognitive"]["enabled"] is False
    assert b.physcfg["density_far"]["enabled"] is False


COG_ON = {"enabled": True, "neighbors": 12, "sectors": 16, "fov_deg": 360.0}
# 前方円錐のアブレーション(既定ではない。円錐経路も統合ランで 1 度は通しておく)
COG_CONE = dict(COG_ON, fov_deg=200.0)
DF_ON = {"enabled": True, "cell_m": 1.0, "blur": 2, "update_every": 10}


@pytest.mark.parametrize("engine", ("orca", "sfm"))
def test_zone_run_on_is_reproducible(tmp_path, engine):
    """ON: 同 seed 2 ランがビット一致する(決定論)。manifest 用の要約も生える。"""
    a = _run(tmp_path, f"cog_on_a_{engine}", engine=engine, far=True,
             cognitive=COG_ON, density_far=DF_ON)
    b = _run(tmp_path, f"cog_on_b_{engine}", engine=engine, far=True,
             cognitive=COG_ON, density_far=DF_ON)
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert _l1(a) == _l1(b), "同 seed 2 ランが一致しない(決定論が壊れている)"
    assert set(P.calib_describe(a)) >= {"cognitive", "density_far"}


@pytest.mark.parametrize("engine,kw", [
    # SFM は密度場 far(C)が常に効くので疎でも世界が変わる
    ("sfm", {"cognitive": COG_ON, "density_far": DF_ON}),
    # ORCA に far 項は無いので B だけが効く。既定(fov=360)の B は**疎なゾーンでは
    # no-op** になる(下のテストがそれを別途固定する)ので、ここは円錐を絞って
    # 「選抜が実際に差し替わっている」ことを確かめる。
    ("orca", {"cognitive": COG_CONE}),
])
def test_zone_run_on_changes_the_world(tmp_path, engine, kw):
    off = _run(tmp_path, f"cog_off_{engine}", engine=engine, far=True)
    on = _run(tmp_path, f"cog_chg_{engine}", engine=engine, far=True, **kw)
    assert _l1(on) != _l1(off), "ON が世界を 1 バイトも変えていない"


def test_cognitive_is_a_no_op_when_the_zone_is_sparse(tmp_path):
    """★疎なゾーンでは認知的近傍 = 距離最近傍(= 何も変わらない)。

    これは不具合ではなく**設計どおり**である: 遮蔽は「手前の人が奥の人を隠す」現象なので、
    隠す相手が居なければ見えている近傍と最近傍 k 体は一致する(Dachner et al. 2022 の
    「距離減衰の主因は遮蔽」= 混んでいないと減衰しない、の裏返し)。
    ゾーンの実効密度が低い ORCA の短ランでは、既定 fov=360 の B は L1 をビット単位で
    変えない。ここを固定しておかないと、将来「ON が効いていない」と誤読される。
    """
    off = _run(tmp_path, "cog_sparse_off", engine="orca", far=True)
    on = _run(tmp_path, "cog_sparse_on", engine="orca", far=True, cognitive=COG_ON)
    assert _l1(on) == _l1(off), \
        "疎なゾーンで既定の認知的近傍が最近傍と食い違った(前提が変わった)"


def test_zone_run_on_keeps_the_boundary_invariants(tmp_path):
    """ON でも境界連続性の不変条件(跳び・重なり)が壊れない。"""
    for name, kw in (("cog_inv_off", {}),
                     ("cog_inv_b", {"cognitive": COG_ON}),
                     ("cog_inv_c", {"density_far": DF_ON}),
                     ("cog_inv_bc", {"cognitive": COG_ON, "density_far": DF_ON}),
                     ("cog_inv_cone", {"cognitive": COG_CONE})):
        sim = _run(tmp_path, name, engine="sfm", far=True, **kw)
        cont = P.continuity(sim)
        assert cont, f"{name}: continuity が空(ゾーンを通っていない)"
        assert cont["jump_max_m"] < 1.0, f"{name}: 1 サブステップの跳びが大きすぎる"
        assert cont["min_gap_m"] is None or cont["min_gap_m"] > -0.5, \
            f"{name}: 深い重なりが出た {cont['min_gap_m']}"


# =========================================================================== #
# (7) P3 密度場の再構築コスト削減(2026-08-20)
#     ── `_box3` の同値高速化 と `density_far.carry_grid`(既定 OFF)
# =========================================================================== #
def _box3_reference(a):
    """P3 以前の `_box3` の**逐語コピー**。比較の相手はいつも「修正前の実装そのもの」。"""
    p = np.zeros((a.shape[0] + 2, a.shape[1] + 2), dtype=np.float64)
    p[1:-1, 1:-1] = a
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _density_grid_reference(crowd):
    """P3 以前の `_density_grid` の逐語コピー(作業領域を持ち回らない版)。"""
    cell = crowd.field_cell_m
    pad = crowd.field_blur + 2
    pos = crowd.pos
    ox = float(pos[:, 0].min()) - pad * cell
    oy = float(pos[:, 1].min()) - pad * cell
    nx = int(math.floor((float(pos[:, 0].max()) - ox) / cell)) + pad + 1
    ny = int(math.floor((float(pos[:, 1].max()) - oy) / cell)) + pad + 1
    nx, ny = max(nx, 3), max(ny, 3)
    ci = np.clip(np.floor((pos[:, 0] - ox) / cell), 0, nx - 1).astype(np.int64)
    cj = np.clip(np.floor((pos[:, 1] - oy) / cell), 0, ny - 1).astype(np.int64)
    sel = crowd.active
    cnt = np.bincount((ci * ny + cj)[sel], minlength=nx * ny)
    rho = cnt.reshape(nx, ny).astype(np.float64) / (cell * cell)
    for _ in range(crowd.field_blur):
        rho = _box3_reference(rho)
    gx = np.zeros_like(rho)
    gy = np.zeros_like(rho)
    gx[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * cell)
    gy[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * cell)
    return rho, gx, gy, ox, oy


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_box3_is_bit_identical_to_the_reference(seed):
    """★加算順序を 1 つも変えていないことの機械証明(プロパティテスト)。

    `x + y + z` は左結合なので、中間結果を毎回新しい配列に置くか同じ配列へ上書きするかで
    値は 1 ビットも変わらない ── その前提を実際の値で踏み固める。値は本番と同じ素性:
    密度格子は `np.bincount` の整数をセル面積で割ったもので、平滑を通ると整数/9.0 になる。
    形は 3〜300 の**非正方形**も含め、平滑は本番と同じ 2 回反復まで見る。
    """
    rng = np.random.default_rng(seed)
    for _ in range(24):
        nx = int(rng.integers(3, 301))
        ny = int(rng.integers(3, 301))
        kind = int(rng.integers(0, 3))
        if kind == 0:                     # bincount 由来(整数カウント / セル面積 1.0)
            a = rng.integers(0, 9, size=(nx, ny)).astype(np.float64)
        elif kind == 1:                   # それを 1 度平滑した素性(整数 / 9.0)
            a = rng.integers(0, 40, size=(nx, ny)).astype(np.float64) / 9.0
        else:                             # 一般の float(丸めの縁を踏ませる)
            a = rng.normal(0.0, 1e3, size=(nx, ny))
        pad = np.zeros((nx + 2, ny + 2), dtype=np.float64)
        acc = np.empty((nx, ny), dtype=np.float64)
        ref, got = a, a
        for i in range(2):
            ref = _box3_reference(ref)
            got = P._box3(got, pad, acc if i == 0 else None)
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert np.array_equal(got, ref), f"{nx}x{ny} kind={kind} blur={i + 1}"
        # バッファを渡さない(従来どおりの)呼び方も同じ
        assert np.array_equal(P._box3(P._box3(a)), ref)


@pytest.mark.parametrize("blur", (0, 1, 2, 3))
def test_density_grid_survives_scratch_reuse(blur):
    """作業領域を持ち回っても (a) 値がビット一致 (b) **前に返した場が壊れない**。

    (b) は carry_grid が場を次のエンジンへ渡すために要る性質(返す ρ は毎回新品)。
    格子の形は在場者の外接矩形で決まるので、途中で形が変わる配置を通す。
    """
    rng = np.random.default_rng(5)
    n = 300
    c = _calib(n, 2.0, "square", 5, far_mode="field", field_blur=blur)
    base = c.pos.copy()
    kept = []
    for k in range(5):
        c.pos = base + rng.uniform(-1.0, 1.0, size=(n, 2)) * (k + 1)
        ref = _density_grid_reference(c)
        got = c._density_grid()
        for g, r in zip(got[:3], ref[:3]):
            assert g.shape == r.shape and g.dtype == r.dtype
            assert np.array_equal(g, r), f"blur={blur} k={k}"
        assert got[3] == ref[3] and got[4] == ref[4]
        kept.append((got, ref))
    for j, (got, ref) in enumerate(kept):
        for g, r in zip(got[:3], ref[:3]):
            assert np.array_equal(g, r), \
                f"blur={blur}: {j} 回目に返した場が後の再構築で書き潰された"


def test_field_trajectory_is_bit_identical_after_the_speedup():
    """場モードの**軌跡**が旧 `_density_grid` 経路とビット一致(積分を通した端点検査)。"""
    a = _calib(400, 2.0, "clump", 9, far_mode="field", field_update_every=3)
    b = _calib(400, 2.0, "clump", 9, far_mode="field", field_update_every=3)
    b._density_grid = lambda: _density_grid_reference(b)
    for _ in range(30):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()
    assert a.vel.tobytes() == b.vel.tobytes()


def test_carry_grid_defaults_to_off_everywhere():
    """★R1: 新キーは既定 false(conf・正準化・読み取りのどこにも ON が漏れない)。"""
    raw = yaml.safe_load((REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))
    assert raw["physics"]["density_far"]["carry_grid"] is False
    assert zones.DENSITY_FAR_DEFAULTS["carry_grid"] is False
    cfg = zones.build_cfg({}, REPO)
    assert cfg["density_far"]["carry_grid"] is False

    class _Sim:
        physcfg = cfg

    assert P._carry_grid_on(_Sim()) is False
    # キーを丸ごと書かない conf でも既定 false(欠測を ON と読まない)
    assert P._carry_grid_on(type("S", (), {"physcfg": {}})()) is False
    assert P._carry_grid_on(type("S", (), {})()) is False


def test_carry_grid_is_declared_in_the_registry():
    f = {x.id: x for x in R.FEATURES}["physics.density_far.carry_grid"]
    assert f.repro_tier == "strict" and f.affects_k is False
    assert f.fingerprint_risk == "none" and f.off_value is False


def test_carry_grid_moves_the_field_to_the_new_engine():
    """ON: 場と再構築カウンタが新エンジンへ渡る / OFF: 渡らない(= 巻き戻る)。"""
    old = _calib(200, 2.0, "square", 3, far_mode="field", field_update_every=10)
    for _ in range(4):
        old.step(0.05)
    assert old._field_grid is not None and old._field_tick == 4
    new = _calib(200, 2.0, "square", 3, far_mode="field", field_update_every=10)
    assert new._field_grid is None and new._field_tick == 0
    P._carry_field(old, new)
    assert new._field_grid is old._field_grid and new._field_tick == 4
    # 作業領域は**渡さない**(渡すと次の再構築が持ち回り中の場を書き潰す)
    assert new._field_scratch is not old._field_scratch
    # 場を持たないエンジン(ORCA)や None が混ざっても落ちない = no-op
    P._carry_field(None, new)
    P._carry_field(old, None)


DF_CARRY = dict(DF_ON, carry_grid=True)


def _grid_rebuilds(tmp_path, name, **kw):
    """mock 短ランで `_density_grid` の呼び出し回数を数える。"""
    cnt = [0]
    orig = P._CalibratedCrowd._density_grid

    def spy(self):
        cnt[0] += 1
        return orig(self)

    P._CalibratedCrowd._density_grid = spy
    try:
        sim = _run(tmp_path, name, engine="sfm", far=True, **kw)
    finally:
        P._CalibratedCrowd._density_grid = orig
    return cnt[0], sim


def test_carry_grid_off_is_byte_identical(tmp_path):
    """★R1: carry_grid を明示 false にしても、書かないランと L1 がバイト一致。"""
    a = _run(tmp_path, "carry_off_a", engine="sfm", far=True, density_far=DF_ON)
    b = _run(tmp_path, "carry_off_b", engine="sfm", far=True,
             density_far=dict(DF_ON, carry_grid=False))
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)


def test_carry_grid_on_never_rebuilds_more_often(tmp_path):
    """ON は再構築回数を増やさない(= 周期の巻き戻しを取り除くだけ)。"""
    off_n, off = _grid_rebuilds(tmp_path, "carry_cnt_off", density_far=DF_ON)
    on_n, on = _grid_rebuilds(tmp_path, "carry_cnt_on", density_far=DF_CARRY)
    assert off_n > 0, "密度格子が 1 度も作られていない(テストが空回りしている)"
    assert on_n <= off_n, f"carry_grid ON で再構築が増えた off={off_n} on={on_n}"
    assert on.physcfg["density_far"]["carry_grid"] is True
    assert off.physcfg["density_far"]["carry_grid"] is False
    assert P.calib_describe(on)["density_far"]["carry_grid"] is True


def test_carry_grid_on_is_reproducible(tmp_path):
    """ON でも同 seed 2 ランがビット一致(場の持ち回りは決定論)。"""
    a = _run(tmp_path, "carry_rep_a", engine="sfm", far=True, density_far=DF_CARRY)
    b = _run(tmp_path, "carry_rep_b", engine="sfm", far=True, density_far=DF_CARRY)
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert _l1(a) == _l1(b)
    cont = P.continuity(a)
    assert cont["jump_max_m"] < 1.0
    assert cont["min_gap_m"] is None or cont["min_gap_m"] > -0.5


def test_the_density_field_never_reaches_the_checkpoint(tmp_path):
    """★場と作業領域は step ローカル(エンジンと同じ寿命)= checkpoint に載らない。

    載ってしまうと (a) 100 万セル級の格子が毎 checkpoint に乗り (b) resume==straight が
    「場を持ったまま再開したか」で割れる。中央管理される物理の状態は `_phys_state`
    (カウンタとヒストグラム)だけで、個体側は `_FIELDS` の値型だけである。
    carry_grid が場を渡すのは**同一 step の中のエンジン間**だけなので、ON でも同じ。
    """
    import pickle

    sim = _run(tmp_path, "carry_ckpt", engine="sfm", far=True, density_far=DF_CARRY)
    blob = P.state_of(sim)
    assert blob is not None and set(blob) >= {"by_zone", "cont"}
    dump = pickle.dumps(blob)
    for marker in (b"_CalibratedCrowd", b"_field_grid", b"_field_tick",
                   b"_field_scratch", b"ndarray"):
        assert marker not in dump, f"checkpoint 側に {marker!r} が載っている"
    assert not [f for f in P._FIELDS if "field" in f]
    for a in sim.agents:
        for f in ("_field_grid", "_field_tick", "_field_scratch"):
            assert not hasattr(a, f), f"個体に {f} が生えている(pickle に載る)"


# =========================================================================== #
# (8) P4 密度格子の**外延の有界化**(2026-08-21)
#     ── `density_far.clip_margin_m`(既定 0.0 = 無効 = 現行と 1 バイト同一)
#
# 発端(第145 の実測): 外延は「在場者の外接矩形」で決まるのに、ゾーンの所有は
# 「経路がゾーンを通り抜ける個体」へ**現在地(数百 m 先)から**始まる。そのため
# 38×29 m の広場に平均 117 万セル(~1 km 角)の場を毎回作っていた。
# 本節が機械固定するもの:
#   (a) 既定 0 では新しい分岐を **1 度も通らない**(= 現行と完全同一経路)。
#   (b) クリップの意味は「矩形の外のメンバーを名簿から外す」ことと**ビット一致**する。
#   (c) ★圏外を np.clip で縁のセルへ**押し込まない**(押し込む実装が実際に誤りである
#       ことを、却下した設計の逐語実装と突き合わせて数字で残す)。
#   (d) 圏外の読み出しは ρ=0・∇ρ=0(= 遠方場の力ちょうど 0)。
# =========================================================================== #
CLIP_MARGIN_M = 10.0
DF_CLIP = dict(DF_ON, clip_margin_m=CLIP_MARGIN_M)
# ゾーン _POLY(±25 m)の外接矩形 ± margin
CLIP_RECT = (-_ZR - CLIP_MARGIN_M, -_ZR - CLIP_MARGIN_M,
             _ZR + CLIP_MARGIN_M, _ZR + CLIP_MARGIN_M)


def _zone_of(poly=None, zid="z1"):
    """テスト用に 1 枚だけ Zone を作る(`bbox` が欲しいだけ)。"""
    return zones.build_cfg(
        {"zones_enabled": True,
         "zones": [{"id": zid, "engine": "sfm",
                    "polygon": list(poly or _POLY)}]}, REPO)["zones"][0]


def test_clip_margin_defaults_to_zero_everywhere():
    """★R1: 新キーは既定 0.0(conf・正準化・矩形の生成・エンジンのどこにも漏れない)。"""
    raw = yaml.safe_load((REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))
    assert float(raw["physics"]["density_far"]["clip_margin_m"]) == 0.0
    assert float(zones.DENSITY_FAR_DEFAULTS["clip_margin_m"]) == 0.0
    cfg = zones.build_cfg({}, REPO)
    assert cfg["density_far"]["clip_margin_m"] == 0.0
    z = _zone_of()
    # 既定では矩形そのものが作られない(= エンジンへ引数が生えない)
    assert P._field_clip(z, cfg["density_far"]) is None
    assert P._field_clip(None, {"clip_margin_m": CLIP_MARGIN_M}) is None
    assert P._field_clip(z, {}) is None          # キーを書かない conf も 0 と読む
    assert P._field_clip(z, None) is None
    # エンジン側の既定も None(新分岐の入口が閉じている)
    assert _calib(50, 2.0, "square", 1, far_mode="field").field_clip is None
    assert _calib(50, 2.0, "square", 1).field_clip is None


def test_clip_margin_is_declared_in_the_registry():
    f = {x.id: x for x in R.FEATURES}["physics.density_far.clip_margin_m"]
    assert f.repro_tier == "strict" and f.affects_k is False
    assert f.fingerprint_risk == "none" and f.off_value == 0.0


def test_clip_margin_rejects_a_negative_margin():
    with pytest.raises(ValueError):
        zones.build_cfg({"density_far": {"clip_margin_m": -1.0}}, REPO)
    # 0 は「無効」であって不正ではない
    assert zones.build_cfg({"density_far": {"clip_margin_m": 0.0}},
                           REPO)["density_far"]["clip_margin_m"] == 0.0


def test_the_clip_rectangle_is_the_zone_bbox_plus_the_margin():
    """矩形 = ゾーン polygon の外接矩形 ± margin(それ以上でも以下でもない)。"""
    z = _zone_of([[-24.0, -56.0], [14.0, -56.0], [14.0, -27.0], [-24.0, -27.0]])
    assert z.bbox == (-24.0, -56.0, 14.0, -27.0)
    got = P._field_clip(z, {"clip_margin_m": 10.0})
    assert got == (-34.0, -66.0, 24.0, -17.0)


def test_calib_kwargs_wires_the_rectangle_only_when_it_is_asked_for():
    """`field_clip` が生えるのは far_field ON × density_far ON × margin>0 のときだけ。"""
    z = _zone_of()

    def _sim(df):
        return type("S", (), {"physcfg": zones.build_cfg(
            {"sfm": {"far_field": dict(FAR_ON)}, "density_far": df}, REPO)})()

    assert "field_clip" not in P._calib_kwargs(_sim(dict(DF_ON)), z)
    assert "field_clip" not in P._calib_kwargs(_sim(dict(DF_CLIP)), None)
    kw = P._calib_kwargs(_sim(dict(DF_CLIP)), z)
    assert kw["field_clip"] == CLIP_RECT and kw["far_mode"] == "field"
    # density_far が OFF なら margin を書いても no-op(置換する当の項が無い)
    off = P._calib_kwargs(_sim({"enabled": False, "clip_margin_m": 10.0}), z)
    assert "field_clip" not in off and "far_mode" not in off
    # 要約(manifest)には出る = ON の値が記録に残る
    assert P.calib_describe(_sim(dict(DF_CLIP)))["density_far"]["clip_margin_m"] == 10.0


def test_clip_off_never_takes_the_new_branch(tmp_path):
    """★R1 の直接証明: 既定 0 のランでは `_clip_mask` が **1 度も呼ばれない**。"""
    calls = [0]
    orig = P._CalibratedCrowd._clip_mask

    def spy(self):
        calls[0] += 1
        return orig(self)

    P._CalibratedCrowd._clip_mask = spy
    try:
        sim = _run(tmp_path, "clip_off_branch", engine="sfm", far=True,
                   density_far=DF_ON)
    finally:
        P._CalibratedCrowd._clip_mask = orig
    assert [e.kind for e in sim.logger.events].count("zone_gate") > 0, \
        "ゾーンを 1 度も通っていない(テストが空回りしている)"
    assert calls[0] == 0, f"既定 OFF で新分岐を {calls[0]} 回通った"


def test_clip_off_is_byte_identical(tmp_path):
    """★R1: clip_margin_m を明示 0.0 にしても、書かないランと L1 がバイト一致。"""
    a = _run(tmp_path, "clip_off_a", engine="sfm", far=True, density_far=DF_ON)
    b = _run(tmp_path, "clip_off_b", engine="sfm", far=True,
             density_far=dict(DF_ON, clip_margin_m=0.0))
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)


def _clipped(n, dens, shape, seed, clip, **kw):
    return _calib(n, dens, shape, seed, far_mode="field", field_clip=clip, **kw)


def test_clip_is_bit_identical_when_everyone_is_inside():
    """全員が矩形の中なら、外延も値も軌跡もクリップ無しと 1 ビット違わない。

    これがクリップの定義そのもの(=「はみ出した分だけ」を切る操作)であり、
    ON にしても混雑したゾーンの中の物理は据え置かれる、という主張の機械証明でもある。
    """
    a = _calib(400, 2.0, "clump", 9, far_mode="field", field_update_every=3)
    lo = a.pos.min(axis=0) - 50.0
    hi = a.pos.max(axis=0) + 50.0
    b = _clipped(400, 2.0, "clump", 9, (lo[0], lo[1], hi[0], hi[1]),
                 field_update_every=3)
    ga, gb = a._density_grid(), b._density_grid()
    for x, y in zip(ga[:3], gb[:3]):
        assert np.array_equal(x, y)
    assert ga[3] == gb[3] and ga[4] == gb[4]
    for _ in range(30):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()
    assert a.vel.tobytes() == b.vel.tobytes()


def _with_strays(base_seed, clip, strays, **kw):
    """在場者(密な塊)+ 圏外を一人で歩いている個体、の 2 通りのエンジンを返す。

    返り値 (全員, 在場者だけ, 在場者数)。「クリップした場」と「圏外を名簿から外した場」が
    ビット一致することを見るための対。
    """
    rng = np.random.default_rng(base_seed)
    core = _layout(300, 2.0, "square", rng)
    both = np.concatenate([core, np.asarray(strays, dtype=np.float64)], axis=0)

    def _mk(p):
        m = p.shape[0]
        return P._CalibratedCrowd(pos=p.copy(), vel=np.zeros((m, 2)),
                                  goal=p + np.array([100.0, 0.0]),
                                  v0=np.full(m, 1.3), radius=np.full(m, 0.3),
                                  neighbor_cap=12, **FAR_KW, far_mode="field",
                                  field_clip=clip, **kw)

    return _mk(both), _mk(core), core.shape[0]


# 数百 m 先を一人で歩いている所有メンバー(finals で実際に起きている状況)
STRAYS = [[800.0, -600.0], [-430.0, 210.0], [90.0, 640.0]]
CORE_CLIP = (-30.0, -30.0, 60.0, 60.0)


def test_clipping_is_exactly_dropping_the_members_outside():
    """クリップ =「矩形の外を名簿から外す」。格子も読み出しもビット一致する。"""
    both, core, n = _with_strays(11, CORE_CLIP, STRAYS)
    gb, gc = both._density_grid(), core._density_grid()
    for x, y in zip(gb[:3], gc[:3]):
        assert x.shape == y.shape and np.array_equal(x, y), \
            "圏外の個体が圏内の場を動かした"
    assert gb[3] == gc[3] and gb[4] == gc[4], "圏外の個体が格子の原点を動かした"
    rb, grb = both._sample_field(gb)
    rc, grc = core._sample_field(gc)
    assert np.array_equal(rb[:n], rc) and np.array_equal(grb[:n], grc)


def test_the_field_force_on_a_member_outside_the_clip_is_exactly_zero():
    """圏外は ρ=0・∇ρ=0 → 遠方場の力ちょうど 0(孤立歩行者に混雑回避力は働かない)。"""
    both, _core, n = _with_strays(12, CORE_CLIP, STRAYS)
    grid = both._density_grid()
    rho, grad = both._sample_field(grid)
    assert np.array_equal(rho[n:], np.zeros(len(STRAYS)))
    assert np.array_equal(grad[n:], np.zeros((len(STRAYS), 2)))
    f = both._far_forces_field()
    assert np.array_equal(f[n:], np.zeros((len(STRAYS), 2))), \
        "圏外の個体に遠方場の力が残っている"
    assert np.abs(f[:n]).max() > 0.0, "圏内にも力が無い(テストが空回りしている)"


def _push_in_reference(crowd):
    """★**却下した設計**(圏外を np.clip で縁のセルへ押し込む)の逐語実装。

    なぜ却下したのかを数字で残すために置く: 押し込むと、数百 m 先を一人で歩いている
    個体の質量がクリップ矩形の**縁のセルへ山積み**になり、ゾーンの縁に居る個体が
    「そこに居ない群衆」の密度と勾配を読む。堆積させない(名簿から外す)のが正しい。
    ここは本体の `_density_grid` の外延計算だけを矩形固定に差し替えた同型のコードである。
    """
    cell = crowd.field_cell_m
    pad = crowd.field_blur + 2
    x0, y0, x1, y1 = crowd.field_clip
    pos = np.clip(crowd.pos, [x0, y0], [x1, y1])        # ← 押し込み(これが誤り)
    ox = x0 - pad * cell
    oy = y0 - pad * cell
    nx = int(math.floor((x1 - ox) / cell)) + pad + 1
    ny = int(math.floor((y1 - oy) / cell)) + pad + 1
    ci = np.clip(np.floor((pos[:, 0] - ox) / cell), 0, nx - 1).astype(np.int64)
    cj = np.clip(np.floor((pos[:, 1] - oy) / cell), 0, ny - 1).astype(np.int64)
    cnt = np.bincount((ci * ny + cj)[crowd.active], minlength=nx * ny)
    return cnt.reshape(nx, ny).astype(np.float64) / (cell * cell), ox, oy


def test_the_rejected_push_in_design_really_would_pile_density_on_the_edge():
    """押し込み版は縁に幽霊群衆を作る / 本実装は作らない(質量保存で数える)。"""
    both, _core, n = _with_strays(13, CORE_CLIP, STRAYS, field_blur=0)
    cell = both.field_cell_m
    grid = both._density_grid()
    mass = float(grid[0].sum()) * cell * cell
    assert math.isclose(mass, n, rel_tol=1e-9), \
        f"堆積した人数が圏内の人数と違う {mass} != {n}"
    push, pox, poy = _push_in_reference(both)
    assert math.isclose(float(push.sum()) * cell * cell, n + len(STRAYS),
                        rel_tol=1e-9), "押し込み版が圏外を数えていない(前提が違う)"
    # 押し込み先(矩形の縁)のセルには誰も居ないのに密度が立つ
    for sx, sy in STRAYS:
        px = min(max(sx, CORE_CLIP[0]), CORE_CLIP[2])
        py = min(max(sy, CORE_CLIP[1]), CORE_CLIP[3])
        i = int(math.floor((px - pox) / cell))
        j = int(math.floor((py - poy) / cell))
        assert push[i, j] > 0.0, "押し込み版の前提が崩れている"
        # 同じ場所に実在の在場者は 1 人も居ない = 幽霊
        d = np.linalg.norm(both.pos[:n] - np.array([px, py]), axis=1)
        assert d.min() > 5.0, "押し込み先の近くに本物が居る(テストの配置が悪い)"


def test_clip_bounds_the_grid_by_the_rectangle():
    """★本題: 遠方の所有メンバーが居ても格子は矩形ぶんで頭打ちになる。"""
    both, _core, _n = _with_strays(14, CORE_CLIP, STRAYS)
    nx, ny = both._density_grid()[0].shape
    cell, pad = both.field_cell_m, both.field_blur + 2
    lim_x = int(math.floor((CORE_CLIP[2] - CORE_CLIP[0]) / cell)) + 2 * pad + 1
    lim_y = int(math.floor((CORE_CLIP[3] - CORE_CLIP[1]) / cell)) + 2 * pad + 1
    assert nx <= lim_x and ny <= lim_y, f"外延が矩形を超えた {nx}x{ny}"
    # クリップ無しだと同じ配置で桁が変わる(この差が P4 の全部である)
    m = both.pos.shape[0]
    off = P._CalibratedCrowd(pos=both.pos.copy(), vel=np.zeros((m, 2)),
                             goal=both.pos + np.array([100.0, 0.0]),
                             v0=np.full(m, 1.3), radius=np.full(m, 0.3),
                             neighbor_cap=12, **FAR_KW, far_mode="field")
    assert off._density_grid()[0].size > 100 * (nx * ny), \
        "クリップ無しの格子が大きくない(テストが空回りしている)"


def test_clip_survives_an_empty_rectangle():
    """圏内が空(全員が矩形の外)でも落ちない = 場は全面 0・力は全員 0。"""
    n = 5
    pos = np.array([[500.0 + i, -400.0] for i in range(n)], dtype=np.float64)
    c = P._CalibratedCrowd(pos=pos, vel=np.zeros((n, 2)),
                           goal=pos + np.array([100.0, 0.0]),
                           v0=np.full(n, 1.3), radius=np.full(n, 0.3),
                           neighbor_cap=12, **FAR_KW, far_mode="field",
                           field_clip=CORE_CLIP)
    rho, gx, gy, _ox, _oy = c._density_grid()
    assert rho.shape == (3, 3) and not rho.any() and not gx.any() and not gy.any()
    assert np.array_equal(c._far_forces_field(), np.zeros((n, 2)))
    c.step(0.05)                                  # 積分まで通す(例外が出ないこと)


def test_clip_rejects_an_inverted_rectangle():
    with pytest.raises(ValueError):
        _clipped(20, 2.0, "square", 1, (10.0, 0.0, -10.0, 5.0))


def test_clip_on_is_reproducible(tmp_path):
    """ON: 同 seed 2 ランがビット一致(クリップは位置の比較だけ = 決定論)。"""
    a = _run(tmp_path, "clip_rep_a", engine="sfm", far=True, density_far=DF_CLIP)
    b = _run(tmp_path, "clip_rep_b", engine="sfm", far=True, density_far=DF_CLIP)
    assert [e.kind for e in a.logger.events].count("zone_gate") > 0
    assert _l1(a) == _l1(b), "同 seed 2 ランが一致しない(決定論が壊れている)"


def test_clip_on_keeps_the_boundary_invariants(tmp_path):
    """ON でも境界連続性の不変条件(跳び・重なり)が壊れない。"""
    for name, df in (("clip_inv_off", DF_ON), ("clip_inv_on", DF_CLIP),
                     ("clip_inv_carry", dict(DF_CLIP, carry_grid=True))):
        sim = _run(tmp_path, name, engine="sfm", far=True, density_far=df)
        cont = P.continuity(sim)
        assert cont, f"{name}: continuity が空(ゾーンを通っていない)"
        assert cont["jump_max_m"] < 1.0, f"{name}: 1 サブステップの跳びが大きすぎる"
        assert cont["min_gap_m"] is None or cont["min_gap_m"] > -0.5, \
            f"{name}: 深い重なりが出た {cont['min_gap_m']}"


class _CountingHub:
    """全 stream の draw を数えるプロキシ(ON の乱数消費不変を直接固定する)。"""

    def __init__(self, inner):
        self._inner = inner
        self.draws = 0

    def stream(self, *key):
        g = self._inner.stream(*key)
        outer = self

        class _W:
            def __getattr__(self, name):
                attr = getattr(g, name)
                if name in ("random", "integers", "choice", "normal", "shuffle",
                            "permutation", "uniform"):
                    def _wrapped(*a, **k):
                        outer.draws += 1
                        return attr(*a, **k)
                    return _wrapped
                return attr

        return _W()

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


def _draws(tmp_path, name, **kw):
    sim = Simulation(_cfg(name, **kw), out_dir=tmp_path / name)
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    return sim.hub.draws, sim


def test_clip_on_consumes_no_extra_randomness(tmp_path):
    """ON は乱数を 1 粒も増やさない(クリップは位置の比較だけ = draw ゼロ)。"""
    off_n, off = _draws(tmp_path, "clip_draw_off", engine="sfm", far=True,
                        density_far=DF_ON)
    on_n, _on = _draws(tmp_path, "clip_draw_on", engine="sfm", far=True,
                       density_far=DF_CLIP)
    assert [e.kind for e in off.logger.events].count("zone_gate") > 0
    assert off_n == on_n, f"乱数消費が変わった off={off_n} on={on_n}"


def test_the_clipped_field_never_reaches_the_checkpoint(tmp_path):
    """★クリップ矩形も場も step ローカル(エンジンと同じ寿命)= checkpoint に載らない。

    P3 の同名テストと同型。矩形はゾーン宣言から毎 step 作り直される定数なので、
    checkpoint に持たせる理由が無い(持たせると resume==straight がそこで割れる)。
    """
    import pickle

    sim = _run(tmp_path, "clip_ckpt", engine="sfm", far=True, density_far=DF_CLIP)
    blob = P.state_of(sim)
    assert blob is not None and set(blob) >= {"by_zone", "cont"}
    dump = pickle.dumps(blob)
    for marker in (b"_CalibratedCrowd", b"_field_grid", b"field_clip",
                   b"_field_scratch", b"ndarray"):
        assert marker not in dump, f"checkpoint 側に {marker!r} が載っている"
    for a in sim.agents:
        for f in ("_field_grid", "field_clip", "_field_scratch"):
            assert not hasattr(a, f), f"個体に {f} が生えている(pickle に載る)"
