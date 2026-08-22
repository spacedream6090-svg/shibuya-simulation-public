"""第151 レーン1: 空間索引のセル細分化(`world.perception_cell_m`)。

正典: docs/plans/index-cell-and-friend-cache-plan.md §1(リサーチ)/ §2-1(設計)/ §4(検収)。
実装: src/society/world/perception.py。

何を直したか: 索引のセル寸法は常に `perception_radius_m`(40m)で、声の段階(第149)が
入って実効半径が 5m になっても **40m セル × 9 個 =(120m)² の中の全員**を距離計算に
流していた(候補母集団 ~183 倍。250k 夕方 step の py-spy で `_hearers_bounded` が ~80%)。

守るもの(検収基準の順)
  (1) ★**既定 0 = 現行と完全一致**: 索引の状態(細格子を持たない)・返り値のオブジェクト
      同一性・mock ランの L1 バイト一致まで。
  (2) ★**取りこぼしゼロ**: cell_m>0 の索引が返す集合・順序が、あらゆる (半径, cap,
      屋内外, 遮蔽有無, セル寸法) で **粗格子 / legacy 全対全走査と完全一致**する。
      C2・speak ハンドラ・count_hearers・目撃チャネルが使う呼び口を全部通す。
  (3) リング半径 = ceil(r/cell)・大半径と疎な近傍は粗格子へ落ちる(退行しえない)。
  (4) 連結キャッシュが **(中心セル, リング半径)** キーで、半径階級を跨いでも壊れない。
  (5) 決定論(同 seed 2 ラン一致・入力順に依らない)・乱数 stream を 1 本も引かない。
  (6) 契約列挙ピン(conf 既定・finals 値・registry 宣言・凍結ファイル不触)。
検証は mock のみ(実 LLM 禁止・≤24step)。
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from omegaconf import OmegaConf

from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import perception as P

_REPO_FINALS = Path(__file__).resolve().parents[1] / "conf" / "finals_observe.yaml"


# --------------------------------------------------------------------------- #
# 共通ヘルパ(hearers_of が要求する duck-type だけを持つ最小の個体)
# --------------------------------------------------------------------------- #
class _A:
    __slots__ = ("id", "x", "y", "loc", "building", "floor", "sleeping")

    def __init__(self, i, x, y, *, building=None, floor=0, loc="street",
                 sleeping=False):
        self.id = int(i)
        self.x = float(x)
        self.y = float(y)
        self.loc = loc
        self.building = building
        self.floor = floor
        self.sleeping = sleeping

    def __repr__(self):                              # 失敗時の可読性
        return f"<A{self.id}@({self.x},{self.y})>"


class _BlocksOdd:
    """遮蔽器の duck-type(据わっているのでループ経路へ後退する)。"""

    def blocks(self, a, b) -> bool:
        return bool(b.id % 2)


def _mixed(n=900, side=120.0, seed=5) -> list:
    """路上 / 建物 3 階ぶん / 睡眠 / 圏外 を混ぜた場面(全 `_context` を通す)。"""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        x = rnd.uniform(-side / 2, side / 2)
        y = rnd.uniform(-side / 2, side / 2)
        m = i % 10
        if m < 5:
            out.append(_A(i, x, y))                                   # 路上
        elif m < 8:
            out.append(_A(i, x, y, building="B1", floor=i % 3,
                          loc="in_building"))                          # 屋内
        elif m == 8:
            out.append(_A(i, x, y, sleeping=True))                     # 睡眠(索引外)
        else:
            out.append(_A(i, x, y, loc="outside"))                     # 圏外(索引外)
    return out


def _dense(n=2600, side=30.0, seed=9) -> list:
    """細格子が実際に選ばれる程度に混んだ場面(`_FINE_MIN_GAIN` を越える)。"""
    rnd = random.Random(seed)
    return [_A(i, rnd.uniform(-side / 2, side / 2), rnd.uniform(-side / 2, side / 2))
            for i in range(n)]


def _ids(hs) -> list:
    return [h.id for h in hs]


_RADII = (0.0, 0.5, 1.0, 3.3, 5.0, 7.0, 10.0, 12.0, 17.5, 20.0, 25.0, 30.0,
          39.9, 40.0, None)
_CAPS = (0, 1, 2, 3, 15, 10 ** 6)
_CELLS = (1.0, 2.0, 3.7, 5.0, 7.5, 13.0, 20.0, 39.9)


# --------------------------------------------------------------------------- #
# (1) 既定 0 = 現行と完全一致
# --------------------------------------------------------------------------- #
def test_default_zero_builds_no_fine_grid():
    """既定 0 では細格子を **1 個も持たない**(状態も追加コストもゼロ)。"""
    idx = P.build_index(_mixed(), 40.0)
    assert idx.cell_m == 0.0
    assert idx._fcells is None and idx._fnp == {} and idx._fnb == {}
    for r in _RADII:
        assert idx._fine_ring(0.0 if r is None else r) == (0, 1.0)
    # 有界クエリを一通り撃っても細格子は生えない。
    for a in idx.cells[list(idx.cells)[0]][:20]:
        idx.hearers(a, 15, 5.0)
        idx.hearers(a)
    assert idx._fcells is None


def test_cell_m_at_or_above_radius_degenerates_to_off():
    """cell_m >= radius は細分化の意味が無いので **0(=現行)へ倒す**。"""
    for cm in (40.0, 40.1, 100.0, 0.0, -3.0, None):
        idx = P.build_index([_A(0, 0, 0)], 40.0, cell_m=cm)
        assert idx.cell_m == 0.0, cm


def test_default_zero_returns_identical_objects():
    """既定 0 の返り値は従来経路そのもの(**オブジェクト同一性**まで一致)。"""
    ags = _mixed()
    a_idx = P.build_index(ags, 40.0)
    b_idx = P.build_index(ags, 40.0, cell_m=0.0)
    for a in ags[:120]:
        x, y = a_idx.hearers(a), b_idx.hearers(a)
        assert [id(h) for h in x] == [id(h) for h in y]
        x, y = a_idx.hearers(a, 15, 5.0), b_idx.hearers(a, 15, 5.0)
        assert [id(h) for h in x] == [id(h) for h in y]


# --------------------------------------------------------------------------- #
# (2) 取りこぼしゼロ = 粗格子 / legacy と全入力で同一
# --------------------------------------------------------------------------- #
def test_fine_grid_matches_coarse_and_legacy_for_every_radius():
    """★本レーンの中核: 任意 (半径, cap, セル寸法) で 3 実装が同じ集合・同じ順序。"""
    ags = _mixed()
    coarse = P.build_index(ags, 40.0)
    for cm in _CELLS:
        fine = P.build_index(ags, 40.0, cell_m=cm)
        for r in _RADII:
            for cap in _CAPS:
                for a in ags[:60]:
                    want = _ids(coarse.hearers(a, cap, r))
                    assert _ids(fine.hearers(a, cap, r)) == want, (cm, r, cap, a)
                    assert _ids(P.hearers_of(a, ags, 40.0, cap=cap,
                                             radius_eff=r)) == want, (cm, r, cap, a)


def test_fine_grid_matches_in_a_dense_scene_where_it_is_actually_used():
    """密な場面(= 細格子が実際に選ばれる)でも 3 実装が一致する。"""
    ags = _dense()
    coarse = P.build_index(ags, 40.0)
    fine = P.build_index(ags, 40.0, cell_m=5.0)
    used = 0
    for r in (1.0, 5.0, 10.0, 20.0):
        for cap in (0, 15):
            for a in ags[:80]:
                want = _ids(coarse.hearers(a, cap, r))
                assert _ids(fine.hearers(a, cap, r)) == want, (r, cap, a)
                assert _ids(P.hearers_of(a, ags, 40.0, cap=cap,
                                         radius_eff=r)) == want
    used = len(fine._fnb)
    assert used > 0, "密な場面なのに細格子が 1 度も使われていない"


def test_fine_grid_matches_with_an_occluder_installed():
    """遮蔽器が据わるとループ経路へ後退する。細格子でも同じ集合になる。"""
    ags = _mixed()
    occ = _BlocksOdd()
    coarse = P.build_index(ags, 40.0, occluder=occ)
    for cm in (2.0, 5.0, 13.0):
        fine = P.build_index(ags, 40.0, occluder=occ, cell_m=cm)
        for r in (1.0, 5.0, 12.0, 20.0, 40.0):
            for cap in (0, 3, 15):
                for a in ags[:40]:
                    assert _ids(fine.hearers(a, cap, r)) == \
                        _ids(coarse.hearers(a, cap, r)), (cm, r, cap)


def test_count_hearers_and_default_hearers_are_untouched_by_cell_m():
    """count_hearers(目撃/遭遇チャネル)と既定 hearers は粗格子のまま = 現行と同一。"""
    for ags in (_mixed(), _dense()):
        coarse = P.build_index(ags, 40.0)
        for cm in (2.0, 5.0, 13.0):
            fine = P.build_index(ags, 40.0, cell_m=cm)
            for a in ags[:80]:
                assert P.count_hearers(a, fine, 40.0) == \
                    P.count_hearers(a, coarse, 40.0)
                assert P.count_hearers(a, fine, 40.0) == \
                    len(P.hearers_of(a, ags, 40.0))
                assert _ids(fine.hearers(a)) == _ids(coarse.hearers(a))


def test_boundary_distance_is_inclusive_on_the_fine_grid():
    """距離がちょうど半径の相手は圏内(<=)。細セル境界を跨いでも落ちない。"""
    # 5m セルの境界(x=5.0)を跨ぐ配置を、半径ちょうどで並べる。
    ags = [_A(0, 0.0, 0.0), _A(1, 5.0, 0.0), _A(2, -5.0, 0.0),
           _A(3, 0.0, 5.0), _A(4, 0.0, -5.0), _A(5, 3.0, 4.0),   # 距離 5 ちょうど
           _A(6, 5.000001, 0.0)]
    fine = P.build_index(ags, 40.0, cell_m=5.0)
    coarse = P.build_index(ags, 40.0)
    got = _ids(fine.hearers(ags[0], 0, 5.0))
    assert got == [1, 2, 3, 4, 5]
    assert got == _ids(coarse.hearers(ags[0], 0, 5.0))


# --------------------------------------------------------------------------- #
# (3) リング半径と「細格子へ回す/回さない」の規則
# --------------------------------------------------------------------------- #
def test_ring_radius_is_ceil_r_over_cell():
    idx = P.build_index([_A(0, 0, 0)], 40.0, cell_m=5.0)
    for r, want in ((0.0, 1), (1.0, 1), (5.0, 1), (5.1, 2), (10.0, 2),
                    (12.0, 3), (15.0, 3), (20.0, 4)):
        assert idx._fine_ring(r)[0] == want, r
        assert want * 5.0 >= r                      # 半径内の点が必ず ±ring セルに入る
    # ring > _FINE_RING_MAX(= 走査セルが増えすぎる)は粗格子へ落ちる。
    for r in (25.0, 30.0, 40.0):
        assert idx._fine_ring(r)[0] == 0, r


def test_large_cell_falls_back_when_the_area_gain_is_thin():
    """面積比の余裕が足りない(細分化の得が薄い)ときは粗格子へ落ちる。"""
    idx = P.build_index([_A(0, 0, 0)], 40.0, cell_m=30.0)
    for r in (5.0, 20.0, 30.0, 40.0):
        assert idx._fine_ring(r)[0] == 0, r


def test_sparse_neighbourhood_stays_on_the_coarse_grid():
    """疎な近傍では細格子が純損なので**現行経路のまま**(退行しえないことの構造的保証)。"""
    sparse = [_A(i, i * 37.0, 0.0) for i in range(40)]     # 近傍にほぼ誰も居ない
    idx = P.build_index(sparse, 40.0, cell_m=5.0)
    for a in sparse:
        idx.hearers(a, 15, 5.0)
    assert idx._fcells is None, "疎なのに細格子が構築された"
    # 密ならちゃんと使う。
    dense = P.build_index(_dense(), 40.0, cell_m=5.0)
    for a in dense.cells[list(dense.cells)[0]][:10]:
        dense.hearers(a, 15, 5.0)
    assert dense._fcells is not None


# --------------------------------------------------------------------------- #
# (4) 連結キャッシュが (中心セル, リング半径) キー
# --------------------------------------------------------------------------- #
def test_neighbourhood_cache_is_keyed_by_cell_and_ring():
    ags = _dense()
    idx = P.build_index(ags, 40.0, cell_m=5.0)
    a = ags[0]
    idx.hearers(a, 15, 5.0)                       # ring 1
    idx.hearers(a, 15, 20.0)                      # ring 4
    rings = {k[3] for k in idx._fnb}
    assert rings == {1, 4}, rings
    assert all(len(k) == 4 for k in idx._fnb)
    # 半径階級を混ぜても答えは変わらない(キャッシュが互いを壊さない)。
    coarse = P.build_index(ags, 40.0)
    for r in (5.0, 20.0, 5.0, 20.0, 10.0, 5.0):
        for b in ags[:40]:
            assert _ids(idx.hearers(b, 15, r)) == _ids(coarse.hearers(b, 15, r))


def test_result_is_independent_of_insertion_order():
    """索引に入れる順を変えても返り値は同一(細格子の走査順に依らない)。"""
    ags = _dense()
    shuffled = list(ags)
    random.Random(3).shuffle(shuffled)
    a_idx = P.build_index(ags, 40.0, cell_m=5.0)
    b_idx = P.build_index(shuffled, 40.0, cell_m=5.0)
    for r in (5.0, 10.0, 20.0):
        for cap in (0, 3, 15):
            for a in ags[:60]:
                assert _ids(a_idx.hearers(a, cap, r)) == \
                    _ids(b_idx.hearers(a, cap, r)), (r, cap)


# --------------------------------------------------------------------------- #
# (5) 実ラン(mock): 既定一致・ON でも決定論
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n=24, steps=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def test_explicit_default_matches_pure_default_run(tmp_path):
    """新キーを既定値で明示指定しても L1 が完全一致(1 バイトも動かない)。"""
    base = _sim(tmp_path, "cell_base")
    base.run()
    same = _sim(tmp_path, "cell_same", **{"world.perception_cell_m": 0.0})
    same.run()
    assert _l1(base) == _l1(same)


def test_cell_m_on_run_is_deterministic_and_matches_off(tmp_path):
    """cell_m ON のランは OFF と L1 バイト一致(層1 = 世界に触らない)かつ再現する。"""
    ov = {"world.speech_levels.enabled": "true", "world.c2_neighbors_max": 15,
          "conversation.enabled": "true"}
    off = _sim(tmp_path, "cell_off", **ov)
    off.run()
    on_a = _sim(tmp_path, "cell_on_a", **{**ov, "world.perception_cell_m": 5.0})
    on_a.run()
    on_b = _sim(tmp_path, "cell_on_b", **{**ov, "world.perception_cell_m": 5.0})
    on_b.run()
    assert _l1(on_a) == _l1(on_b)                 # 決定論
    assert _l1(on_a) == _l1(off)                  # 世界は 1 ビットも動かない


def test_l1_identical_when_the_fine_path_is_actually_taken(tmp_path, monkeypatch):
    """★門②(局所の混み具合)を外して**細格子を必ず通す**実ランでも L1 がバイト一致。

    小さな mock ランは近傍が疎なので通常は粗格子のまま = 細格子の配線が実ランで
    素通りしてしまう。`_FINE_MIN_GAIN` を 0 に落として C2・speak・観測の全経路を
    細格子で走らせ、それでも世界が 1 ビットも動かないことを確かめる。"""
    ov = {"world.speech_levels.enabled": "true", "world.c2_neighbors_max": 15,
          "conversation.enabled": "true"}
    off = _sim(tmp_path, "cell_forced_off", **ov)
    off.run()

    monkeypatch.setattr(P, "_FINE_MIN_GAIN", 0.0)
    seen = {"fine": 0}
    orig = P.PerceptIndex._fine_neighborhood

    def _spy(self, ctx, cx, cy, ring):
        seen["fine"] += 1
        return orig(self, ctx, cx, cy, ring)

    monkeypatch.setattr(P.PerceptIndex, "_fine_neighborhood", _spy)
    on = _sim(tmp_path, "cell_forced_on", **{**ov, "world.perception_cell_m": 5.0})
    on.run()
    assert seen["fine"] > 0, "門を外したのに細格子経路を 1 度も通っていない"
    assert _l1(on) == _l1(off)


def test_scheduler_builds_the_index_with_the_configured_cell(tmp_path):
    sim = _sim(tmp_path, "cell_wired", steps=2,
               **{"world.perception_cell_m": 5.0})
    sim.run()
    assert P.cell_m_of(sim) == 5.0
    assert sim.percept_index.cell_m == 5.0
    off = _sim(tmp_path, "cell_wired_off", steps=2)
    off.run()
    assert off.percept_index.cell_m == 0.0


def test_no_random_stream_is_drawn_by_the_fine_grid():
    """細格子は乱数を 1 粒も引かない(全て幾何と決定論の整列)。"""
    ags = _dense()
    idx = P.build_index(ags, 40.0, cell_m=5.0)
    before = random.random()
    st = random.getstate()
    for a in ags[:100]:
        idx.hearers(a, 15, 5.0)
    assert random.getstate() == st
    assert before == before


# --------------------------------------------------------------------------- #
# (6) 契約列挙ピン
# --------------------------------------------------------------------------- #
def test_conf_default_is_off():
    cfg = load_config()
    assert float(cfg.world.perception_cell_m) == 0.0


def test_finals_profile_declares_the_registered_value():
    fin = OmegaConf.load(_REPO_FINALS)
    assert float(fin.world.perception_cell_m) == 5.0
    # finals の細セル寸法は通常声(屋外)の半径に一致している = 定石どおりの較正。
    assert float(fin.world.perception_cell_m) == \
        float(fin.world.speech_levels.normal.outdoor_m)


def test_registry_declares_the_new_key():
    f = R.BY_ID.get("world.perception_cell_m")
    assert f is not None, "world.perception_cell_m がレジストリ未宣言"
    assert f.repro_tier == "strict"
    assert f.fingerprint_risk == "none"
    assert f.affects_k is False           # 層1 = LLM 呼の発生点も本数も変わらない
    assert f.off_value == 0.0
    assert R.undeclared_toggles(load_config()) == []


def test_touched_files_are_not_frozen():
    from society.observer import metrics_spec as MS
    for rel in ("src/society/world/perception.py",
                "src/society/engine/scheduler.py",
                "src/society/registry.py"):
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"


def test_ring_bound_is_mathematically_sound():
    """ceil(r/cell) セルで取りこぼさないことを座標の総当たりで確かめる。"""
    rnd = random.Random(17)
    for cell in (1.0, 2.5, 5.0, 7.5):
        inv = 1.0 / cell
        for _ in range(4000):
            x = rnd.uniform(-500.0, 500.0)
            r = rnd.uniform(0.0, 4.0 * cell)
            ring = int(math.ceil(r / cell))
            while ring * cell < r:
                ring += 1
            for other in (x - r, x + r, x - r / 2, x + r / 2):
                assert abs(math.floor(other * inv) - math.floor(x * inv)) <= ring
