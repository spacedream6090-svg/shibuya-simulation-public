"""竹-3: SFM 対壁斥力 f_iW(sfm_core)+ WallCrowd の ξ 欠落バグ修正の検収。

正典: docs/research/physics-engine-selection.md「★ P2 決定(2026-08-02・確定)」
      条件2「WallCrowd の ξ 欠落バグを竹-3 で修正」/ 表「壁のある領域は SFM が自然。
      f_iW は Helbing 標準項(竹-3 で追加)」。

検収項目:
  (A) f_iW の解析値一致(単一壁・端点・作用距離外・複数壁の合力=最近傍1本ではない)
  (B) 空間ハッシュ = 全ペアの結果が **ビット一致**(候補集合は距離内の壁の上位集合)+
      実際に枝刈りできている(候補ペア数 ≪ N×M)
  (C) walls=None(既定)の挙動不変 = 竹-3 以前(HEAD)のコアで採ったゴールデン sha256 と
      バイト一致(壁パラメータを変えても 1 バイトも動かない)
  (D) 壁非貫通(合成コリドー: 壁の向こう側に goal を置いて押し当てても越えない)
  (E) ξ 復活の実効: 壁ありでも noise>0 で軌跡が変わり、同 seed で完全再現(決定論維持)
  (F) WallCrowd 委譲後の既存挙動維持: forces() の上書きが無い(=ξ 欠落バグの構造的解消)・
      壁斥力/近傍 cap は sfm_core の引数・屋内積分の結果が竹-3 以前とバイト一致
  (G) conf physics.sfm の宣言値 = コード既定値、かつ bool トグルを持たない
      (= どこからも自動で ON にならない・registry.py の宣言対象外)

★ゴールデン sha256 は「竹-3 の変更前(git HEAD の sfm_core.py / indoor_flow.py)を
  そのまま実行して採った値」であり、この 3 本が緑である限り既定経路は 1 バイトも動いていない。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from society.world import indoor_flow as flow
from society.world import sfm_core as core
from society.world import vision

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 竹-3 変更前(HEAD)のコアで採った軌跡ゴールデン(scenario は下の _golden_* と同一) ──
GOLDEN_OPEN = "36603ed8e1f750d5b3208c358dd594424f65845506971443a683ec9d7023d983"
GOLDEN_INDOOR = "fb96c5945ae79ff5e3ae6a981ba53af59e96ed9323f4db15131c65002354dae5"
GOLDEN_WALLFORCE = "4215dd19fc8a071b7a04e8fb6378e6247cc467ed3c3d8f9fd27e2196dc622fc6"


def _sha(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def _office():
    return {"id": "b", "kind": "office",
            "footprint": [[0.0, 0.0], [40.0, 0.0], [40.0, 24.0], [0.0, 24.0]]}


# ═══════════════════════════ (A) f_iW の解析値一致 ═══════════════════════════
def _analytic(a_w, b_w, r, d):
    """|f_iW| = A_w·exp((r − d)/B_w)(Helbing, Farkas & Vicsek 2000)。"""
    return a_w * np.exp((r - d) / b_w)


def test_single_wall_force_matches_hand_computation():
    """単一の直線壁: f_iW = A_w·exp((r−d)/B_w)·n_iW を手計算値と厳密一致で照合。

    壁 = y 軸(x=0 の線分)。個体は (0.5, 0) ・半径 0.30 → d=0.5・n=(+1,0)。
    |f| = 2000·exp((0.30−0.50)/0.08) = 2000·exp(−2.5) = 164.16999… N。
    """
    walls = [((0.0, -10.0), (0.0, 10.0))]
    c = core.Crowd(pos=[[0.5, 0.0]], vel=[[0.0, 0.0]], goal=[[0.5, 0.0]],
                   v0=[0.0], radius=[0.30], walls=walls)
    f = c._wall_forces()
    expect = _analytic(core.WALL_A_DEFAULT, core.WALL_B_DEFAULT, 0.30, 0.50)
    assert f[0, 0] == pytest.approx(expect, rel=1e-12)
    assert f[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert expect == pytest.approx(164.1699972, rel=1e-8)      # 手計算値そのもの
    # 反対側(x<0)では法線が反転する(壁 → 個体の向き)
    c2 = core.Crowd(pos=[[-0.5, 0.0]], vel=[[0.0, 0.0]], goal=[[-0.5, 0.0]],
                    v0=[0.0], radius=[0.30], walls=walls)
    assert c2._wall_forces()[0, 0] == pytest.approx(-expect, rel=1e-12)


def test_wall_force_uses_nearest_point_on_segment_including_endpoint():
    """最近点が線分の端点に落ちる場合(t のクリップ)も距離・法線が正しい。

    壁 = (0,0)-(0,10)。個体 (0.3, −0.4) の最近点は端点 (0,0)、d=0.5・n=(0.6,−0.8)。
    """
    walls = [((0.0, 0.0), (0.0, 10.0))]
    c = core.Crowd(pos=[[0.3, -0.4]], vel=[[0.0, 0.0]], goal=[[0.3, -0.4]],
                   v0=[0.0], radius=[0.30], walls=walls)
    f = c._wall_forces()[0]
    mag = _analytic(core.WALL_A_DEFAULT, core.WALL_B_DEFAULT, 0.30, 0.50)
    assert f[0] == pytest.approx(0.6 * mag, rel=1e-12)
    assert f[1] == pytest.approx(-0.8 * mag, rel=1e-12)


def test_wall_force_is_zero_beyond_range():
    """d > r + wall_range では寄与 0(閾値の外は評価しても厳密に 0)。"""
    walls = [((0.0, -10.0), (0.0, 10.0))]
    inside = core.Crowd(pos=[[2.29, 0.0]], vel=[[0.0, 0.0]], goal=[[2.29, 0.0]],
                        v0=[0.0], radius=[0.30], walls=walls)
    outside = core.Crowd(pos=[[2.31, 0.0]], vel=[[0.0, 0.0]], goal=[[2.31, 0.0]],
                         v0=[0.0], radius=[0.30], walls=walls)
    assert inside._wall_forces()[0, 0] > 0.0                   # r+2.0=2.30 の内側
    assert np.array_equal(outside._wall_forces()[0], np.zeros(2))   # 外側は厳密 0


def test_all_walls_in_range_sum_not_only_nearest():
    """閾値内の **全壁セグメントの合力**(Helbing 標準)= 最近傍 1 本だけではない。

    直交する 2 壁(x 軸・y 軸)の角に居る個体は、両方から等量ずつ押される。
    """
    walls = [((0.0, -10.0), (0.0, 10.0)), ((-10.0, 0.0), (10.0, 0.0))]
    c = core.Crowd(pos=[[0.5, 0.5]], vel=[[0.0, 0.0]], goal=[[0.5, 0.5]],
                   v0=[0.0], radius=[0.30], walls=walls)
    f = c._wall_forces()[0]
    mag = _analytic(core.WALL_A_DEFAULT, core.WALL_B_DEFAULT, 0.30, 0.50)
    assert f[0] == pytest.approx(mag, rel=1e-12)
    assert f[1] == pytest.approx(mag, rel=1e-12)
    # 最近傍1本しか見ていなければ片方の成分が 0 になる = そうなっていないことを固定
    assert min(abs(f[0]), abs(f[1])) > 0.5 * mag


# ═══════════════════ (B) 空間ハッシュ = 全ペア(ビット一致 + 枝刈り) ═══════════════════
def _grid_walls(nx=12, ny=12, pitch=6.0):
    """格子状の壁(1 セグメント長 6m の升目)。空間ハッシュの枝刈りを測る合成ジオメトリ。"""
    ws = []
    for i in range(nx):
        for j in range(ny):
            x, y = i * pitch, j * pitch
            ws.append(((x, y), (x + pitch, y)))
            ws.append(((x, y), (x, y + pitch)))
    return ws


def test_spatial_hash_equals_all_pairs_bitwise():
    """空間ハッシュ探索と全ペア探索の f_iW が **ビット一致**(小〜中規模の無作為配置)。"""
    bld = _office()
    walls = vision.building_walls(bld, 1)
    rng = np.random.default_rng(4242)
    n = 200
    pos = np.stack([rng.uniform(-2.0, 42.0, n), rng.uniform(-2.0, 26.0, n)], axis=1)
    radius = rng.uniform(0.25, 0.35, n)
    kw = dict(vel=np.zeros((n, 2)), goal=pos + 1.0, v0=np.full(n, 1.2),
              radius=radius, walls=walls)
    hashed = core.Crowd(pos=pos, wall_hash=True, **kw)._wall_forces()
    brute = core.Crowd(pos=pos, wall_hash=False, **kw)._wall_forces()
    assert hashed.tobytes() == brute.tobytes(), "空間ハッシュと全ペアがビット一致しない"
    assert np.abs(hashed).sum() > 0.0                          # 実際に力が出ている


def test_spatial_hash_prunes_candidate_pairs():
    """空間ハッシュが実際に枝刈りしている(候補ペア数 ≪ N×M)= 全ペア走査をしていない。"""
    walls = _grid_walls()
    field = core.WallField(walls, cell=core.WALL_RANGE_M + core.RADIUS_MAX)
    rng = np.random.default_rng(7)
    n = 300
    pos = np.stack([rng.uniform(0.0, 72.0, n), rng.uniform(0.0, 72.0, n)], axis=1)
    ai, si = field.pairs(pos, 0.35 + core.WALL_RANGE_M)
    n_all = n * field.n_seg
    assert ai.shape[0] == si.shape[0]
    assert ai.shape[0] < 0.05 * n_all, f"枝刈りが効いていない({ai.shape[0]}/{n_all})"
    # 候補は (個体, 壁) 昇順・重複なし(合算順序の決定論)
    key = ai * field.n_seg + si
    assert np.array_equal(key, np.unique(key))
    # それでも結果は全ペアとビット一致(= 取りこぼしゼロ)
    radius = np.full(n, 0.30)
    got = core.wall_forces_from_pairs(pos, radius, field, ai, si)
    ref = core.wall_forces_from_pairs(pos, radius, field, *field.all_pairs(n))
    assert got.tobytes() == ref.tobytes()


def test_wall_cell_size_does_not_change_result():
    """空間ハッシュの格子サイズは探索の粒度でしかない = 変えても力はビット一致。"""
    walls = _grid_walls(6, 6)
    rng = np.random.default_rng(21)
    n = 80
    pos = np.stack([rng.uniform(0.0, 36.0, n), rng.uniform(0.0, 36.0, n)], axis=1)
    kw = dict(vel=np.zeros((n, 2)), goal=pos + 2.0, v0=np.full(n, 1.2),
              radius=rng.uniform(0.25, 0.35, n), walls=walls)
    base = core.Crowd(pos=pos, **kw)
    assert base.wall_cell == pytest.approx(core.WALL_RANGE_M + core.RADIUS_MAX)
    ref = base._wall_forces().tobytes()
    for cell in (0.5, 1.0, 5.0, 40.0):
        assert core.Crowd(pos=pos, wall_cell=cell, **kw)._wall_forces().tobytes() == ref


def test_spatial_hash_handles_oversized_segment():
    """AABB が巨大な壁(格子セル数が上限超)は「常時候補」へ退避されても結果は全ペアと一致。"""
    walls = [((-1e4, -1e4), (1e4, 1e4)), ((0.0, -5.0), (0.0, 5.0))]
    field = core.WallField(walls, cell=2.35)
    assert field._always.shape[0] == 1 and int(field._always[0]) == 0
    rng = np.random.default_rng(3)
    n = 40
    pos = np.stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-3.0, 3.0, n)], axis=1)
    radius = np.full(n, 0.30)
    ai, si = field.pairs(pos, 0.30 + core.WALL_RANGE_M)
    got = core.wall_forces_from_pairs(pos, radius, field, ai, si)
    ref = core.wall_forces_from_pairs(pos, radius, field, *field.all_pairs(n))
    assert got.tobytes() == ref.tobytes()


# ═══════════════ (C) walls=None(既定)の挙動不変 = 竹-3 以前とバイト一致 ═══════════════
def _golden_open_scenario(**kw):
    """開放領域(壁なし)24 体の収束シナリオ。竹-3 以前のコアで採った軌跡と比較する。"""
    n = 24
    ang = np.arange(n) * (2.0 * np.pi / n)
    pos = np.stack([8.0 * np.cos(ang), 8.0 * np.sin(ang)], axis=1)
    v0 = 1.0 + 0.4 * (np.arange(n) % 5) / 4.0
    radius = 0.25 + 0.10 * (np.arange(n) % 3) / 2.0
    c = core.Crowd(pos=pos, vel=np.zeros((n, 2)), goal=-pos, v0=v0, radius=radius,
                   noise=0.0, **kw)
    for _ in range(200):
        c.step(0.05)
    return c


def test_open_core_matches_pre_batch_golden_bytes():
    """walls を渡さない既存の呼び出しは竹-3 以前と **バイト一致**(完全後方互換)。"""
    c = _golden_open_scenario()
    assert c.wall_field is None and c.neighbor_cap is None
    assert _sha(c.pos, c.vel) == GOLDEN_OPEN


def test_walls_none_ignores_wall_parameters():
    """walls=None なら wall_a/wall_b/wall_range を何に変えても 1 バイトも動かない。"""
    base = _golden_open_scenario()
    weird = _golden_open_scenario(walls=None, wall_a=9.9e4, wall_b=1.5,
                                  wall_range=50.0, wall_hash=False)
    assert _sha(weird.pos, weird.vel) == _sha(base.pos, base.vel) == GOLDEN_OPEN
    # 空リストも「壁なし」扱い(旧 WallCrowd の `if segs:` と同じ判定)
    empty = _golden_open_scenario(walls=[])
    assert empty.wall_field is None
    assert _sha(empty.pos, empty.vel) == GOLDEN_OPEN


# ═══════════════════════════ (D) 壁非貫通(合成コリドー) ═══════════════════════════
def _corridor(length=20.0, width=3.0, pieces=8):
    """合成コリドー: 上下の側壁 + 終端の閉じた壁(goal はその向こう側に置く)。

    側壁は pieces 分割して渡す(実データの壁も細切れ線分で来るため)。分割数を 8 にすると
    セグメント数 17 > リング探索セル数 9 となり、**空間ハッシュ経路**が実際に使われる。"""
    step = length / pieces
    ws = []
    for k in range(pieces):
        x0, x1 = k * step, (k + 1) * step
        ws.append(((x0, 0.0), (x1, 0.0)))
        ws.append(((x0, width), (x1, width)))
    ws.append(((length, 0.0), (length, width)))
    return ws


def test_wall_non_penetration_in_synthetic_corridor():
    """壁の向こう側に goal を置いて押し当て続けても、誰も壁を越えない(合成コリドー)。"""
    length, width = 20.0, 3.0
    walls = _corridor(length, width)
    assert len(walls) == 17                                    # 空間ハッシュ経路を通る本数
    rng = np.random.default_rng(11)
    n = 40
    pos = np.stack([rng.uniform(1.0, 8.0, n), rng.uniform(0.6, width - 0.6, n)], axis=1)
    radius = rng.uniform(0.25, 0.35, n)
    c = core.Crowd(pos=pos, vel=np.zeros((n, 2)),
                   goal=np.tile([length + 20.0, width / 2.0], (n, 1)),
                   v0=np.full(n, 1.34), radius=radius, walls=walls, neighbor_cap=12)
    x_max, y_min, y_max = -1e9, 1e9, -1e9
    for _ in range(600):
        c.step(0.05)
        x_max = max(x_max, float(c.pos[:, 0].max()))
        y_min = min(y_min, float(c.pos[:, 1].min()))
        y_max = max(y_max, float(c.pos[:, 1].max()))
    assert x_max < length, f"終端壁を越えた(x_max={x_max})"
    assert y_min > 0.0 and y_max < width, f"側壁を越えた(y∈[{y_min},{y_max}])"
    # 駆動項に押されて終端壁の手前まで詰めている(壁斥力が効いているだけで固まっていない)
    assert x_max > length - 1.0


# ═══════════════════ (E) ξ 復活の実効(壁あり)+ 決定論 ═══════════════════
def _corridor_run(noise, seed=None, steps=200):
    walls = _corridor()
    rng0 = np.random.default_rng(20260802)
    n = 30
    pos = np.stack([rng0.uniform(1.0, 6.0, n), rng0.uniform(0.6, 2.4, n)], axis=1)
    radius = rng0.uniform(0.25, 0.35, n)
    kw = {}
    if noise > 0.0:
        kw = {"noise": noise, "rng": np.random.default_rng(seed)}
    c = flow.WallCrowd(pos=pos, vel=np.zeros((n, 2)),
                       goal=np.tile([40.0, 1.5], (n, 1)), v0=np.full(n, 1.2),
                       radius=radius, walls=walls, **kw)
    for _ in range(steps):
        c.step(0.05)
    return c


def test_fluctuation_xi_is_alive_with_walls():
    """★ξ 欠落バグの修正: 壁ありの WallCrowd でも noise>0 が軌跡を変える。

    修正前は WallCrowd.forces() が Crowd.forces() を上書きして ξ 項を落としており、
    **壁を使う経路では noise が完全に無効**だった(reference/physics_bench の
    engine_notes.wallcrowd_drops_fluctuation_xi = noise 0 と 2.0 の軌跡ハッシュが一致)。
    本テストは「noise>0 の軌跡が noise=0 と異なる」ことを固定して再発を防ぐ。
    """
    quiet = _corridor_run(0.0)
    noisy = _corridor_run(0.3, seed=7)
    assert _sha(quiet.pos, quiet.vel) != _sha(noisy.pos, noisy.vel), \
        "壁ありで noise が無視されている(ξ 欠落バグの再発)"


def test_fluctuation_xi_is_deterministic_per_seed():
    """ξ を入れても決定論は保たれる: 同 seed 2 ラン = バイト一致 / 別 seed = 別軌跡。"""
    a = _corridor_run(0.3, seed=7)
    b = _corridor_run(0.3, seed=7)
    d = _corridor_run(0.3, seed=8)
    assert a.pos.tobytes() == b.pos.tobytes() and a.vel.tobytes() == b.vel.tobytes()
    assert a.pos.tobytes() != d.pos.tobytes()


def test_fluctuation_default_off_is_byte_identical():
    """既定(noise=0)は ξ を一切引かない = 2 ランでバイト一致・rng を消費しない。"""
    a = _corridor_run(0.0)
    b = _corridor_run(0.0)
    assert a.pos.tobytes() == b.pos.tobytes()
    # rng を渡しても noise=0 なら Generator の状態は動かない(乱数列に無風)
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state["state"]["state"]
    c = flow.WallCrowd(pos=[[1.0, 1.5]], vel=[[0.0, 0.0]], goal=[[10.0, 1.5]],
                       v0=[1.2], radius=[0.3], walls=_corridor(), rng=rng, noise=0.0)
    for _ in range(20):
        c.step(0.05)
    assert rng.bit_generator.state["state"]["state"] == before


# ═════════════ (F) WallCrowd 委譲後の既存挙動維持(構造 + バイト) ═════════════
def test_wallcrowd_does_not_override_forces():
    """WallCrowd が forces() を上書きしていない = ξ を落とす経路が構造的に存在しない。"""
    assert flow.WallCrowd.forces is core.Crowd.forces
    assert flow.WallCrowd._wall_forces is core.Crowd._wall_forces
    # 壁パラメータの正典は sfm_core 側(屋内層は名前だけ再export)
    assert flow.WALL_A == core.WALL_A_DEFAULT == 2000.0
    assert flow.WALL_B == core.WALL_B_DEFAULT == 0.08
    assert flow._WALL_CUTOFF_M == core.WALL_RANGE_M == 2.0


def test_wallcrowd_equals_core_crowd_with_same_arguments():
    """WallCrowd(walls=…) と Crowd(walls=…, neighbor_cap=12) の合力がビット一致(委譲の証明)。"""
    walls = vision.building_walls(_office(), 1)
    rng = np.random.default_rng(99)
    n = 60
    pos = np.stack([rng.uniform(0.0, 40.0, n), rng.uniform(0.0, 24.0, n)], axis=1)
    kw = dict(vel=np.zeros((n, 2)), goal=pos + 3.0, v0=np.full(n, 1.2),
              radius=rng.uniform(0.25, 0.35, n), walls=walls)
    a = flow.WallCrowd(pos=pos, **kw).forces()
    b = core.Crowd(pos=pos, neighbor_cap=flow.NEIGHBOR_CAP, **kw).forces()
    assert a.tobytes() == b.tobytes()


def test_wallcrowd_forces_match_pre_batch_golden_bytes():
    """壁あり 120 体の合力が竹-3 以前(HEAD の密な N×M 実装)と **バイト一致**。"""
    walls = vision.building_walls(_office(), 1)
    rng = np.random.default_rng(20260802)
    n = 120
    pos = np.stack([rng.uniform(0.0, 40.0, n), rng.uniform(0.0, 24.0, n)], axis=1)
    radius = 0.25 + 0.10 * (np.arange(n) % 7) / 6.0
    c = flow.WallCrowd(pos=pos, vel=np.zeros((n, 2)), goal=pos + 5.0,
                       v0=np.full(n, 1.2), radius=radius, walls=walls)
    assert _sha(c.forces()) == GOLDEN_WALLFORCE


def test_indoor_transition_matches_pre_batch_golden_bytes():
    """屋内の遷移積分(実レイアウト)の軌跡・接触が竹-3 以前と **バイト一致**。"""
    bld = _office()
    lay = vision.building_layout(bld, 1)
    walls = vision.building_walls(bld, 1)
    doors = flow.doors_from_layout(lay)
    movers = [{"agent_id": 3, "src_zone": 0, "dst_zone": lay["nA"] + 1},
              {"agent_id": 1, "src_zone": 1, "dst_zone": lay["nA"]}]
    bys = [{"agent_id": 9, "pos": (lay["corridor"][0] + 8.0, 12.0)}]
    res = flow.integrate_transition(lay, walls, doors, movers, bys)
    xy = np.array([[t, x, y] for (_a, t, x, y) in res.samples], dtype=np.float64)
    assert _sha(xy) == GOLDEN_INDOOR
    assert len(res.samples) == 360
    assert res.contacts == [(3, 9, "coplace", 175.0)]


def test_neighbor_cap_none_is_the_default_and_cap_is_effective():
    """近傍 cap は None(=上限なし)が既定。cap を与えたときだけ最遠の相手が落ちる。"""
    n = 14
    ang = np.arange(n - 1) * (2.0 * np.pi / (n - 1))
    pos = np.vstack([[0.0, 0.0], np.stack([np.cos(ang), np.sin(ang)], axis=1)])
    kw = dict(vel=np.zeros((n, 2)), goal=np.tile([10.0, 0.0], (n, 1)),
              v0=np.full(n, 1.2), radius=np.full(n, 0.3))
    assert core.Crowd(pos=pos, **kw).neighbor_cap is None
    full = core.Crowd(pos=pos, **kw).forces()[0]
    capped = core.Crowd(pos=pos, neighbor_cap=6, **kw).forces()[0]
    assert not np.allclose(full, capped)


# ═══════════════════════ (G) conf 宣言(既定値のみ・自動 ON 無し) ═══════════════════════
def _conf_physics():
    doc = yaml.safe_load((REPO_ROOT / "conf" / "config.yaml").read_text(encoding="utf-8"))
    return doc["physics"]["sfm"]


def test_conf_physics_sfm_defaults_match_code():
    """conf physics.sfm の宣言値 = 実装の既定値(config とコードのズレ検知)。"""
    p = _conf_physics()
    assert p["wall_A"] == core.WALL_A_DEFAULT
    assert p["wall_B"] == core.WALL_B_DEFAULT
    assert p["wall_range"] == core.WALL_RANGE_M
    assert p["wall_hash_cell"] == pytest.approx(core.WALL_RANGE_M + core.RADIUS_MAX)
    assert p["noise"] == 0.0                       # ξ は既定 OFF(未較正)


def test_conf_physics_sfm_has_no_boolean_toggle():
    """physics.sfm は bool トグルを持たない = どこからも自動で ON にならない宣言専用ブロック。

    (bool リーフがあれば registry.py の repro_tier 宣言が必須になる = 第72バッチの規約。
     竹-3 では宣言だけを置き、エンジン切替 physics.zones[].engine は竹-4 の範囲。)
    """
    def _walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, bool):
            yield path
    assert list(_walk(_conf_physics())) == []
