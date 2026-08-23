"""第154 A1/A2: `perception.has_hearers`(存在判定モード = 最初の 1 件で打ち切り)のテスト。

正典: docs/plans/step-time-audit.md §3 A1/A2・§4 の適用順 1/2。

背景: 「同席者が居るか否か」の **1 ビット**しか読まないのに 40m 圏を全列挙して id 昇順へ
整列していた呼び手が 3 か所(`scheduler._phase_drive` の face 判定)残っており、第152 が
`_decide_g` で `count_hearers` へ落とした 1 か所も**人数すら要らない**(250k 夕方の
py-spy で `_decide_g:3074` が 7.5% + `_phase_drive:2929` が 5.7%)。

守るもの(検収基準の順)
  (1) ★同値: `has_hearers(...) == (count_hearers(...) > 0)` が**全入力で**成り立つ
      (索引経路 / legacy 経路 / 遮蔽あり・なし / 密・疎 / 境界ちょうど / ±ULP 帯 /
       屋内外 / 就寝 / 範囲外 / 空セル / ブロック境界)。
      `count == len(hearers_of)` は第150 が機械照合済み(tests/test_count_hearers.py)なので、
      ここを繋ぐと `has_hearers == bool(hearers_of)` が従う。
  (2) ★早期打ち切りが**実際に効いている**(全部計算してから any() ではない):
      密なセルでは近傍 9 セルの連結配列(`_nc`)を 1 本も作らず、触るセルも 1 個で済む。
      遮蔽器経路は `blocks` の呼び出し回数が `count` より少ない。
  (3) 契約: cap / radius_eff を受け取らない(`count_hearers` と同じ引数面)。
  (4) 索引キャッシュを共有しても `hearers` / `count` / 有界経路の答えが変わらない。
  (5) ★実ラン(mock)の L1 完全一致: `scheduler.has_hearers` を**修正前の式**
      (`bool(hearers_of(...))`)へ差し替えた B ランと現行 A ランが 1 バイトも違わない。
  (6) 回帰ガード: `_phase_drive` / `_decide_g` が同席判定でリストを作らない。

検証は mock のみ(実 LLM 禁止・≤24step)。乱数は 1 本も引かない。
"""
from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world import perception as P


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_count_hearers.py / test_decide_company_count.py と同型)
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

    def __repr__(self):
        return f"<A{self.id}@({self.x},{self.y})>"


class _BlocksOdd:
    def blocks(self, a, b) -> bool:
        return bool(b.id % 2)


class _CountingOcc:
    """`blocks` の呼び出し回数を数える遮蔽器(遮蔽はしない)。"""

    def __init__(self):
        self.calls = 0

    def blocks(self, a, b) -> bool:
        self.calls += 1
        return False


def _grid(n_side: int = 12, pitch: float = 3.0) -> list:
    out = []
    i = 0
    for gx in range(n_side):
        for gy in range(n_side):
            out.append(_A(i, gx * pitch, gy * pitch))
            i += 1
    return out


def _assert_equiv(speaker, agents, radius, occluder=None, label=""):
    """索引経路 / legacy 経路の双方で `has == (count > 0) == bool(hearers_of)` を釘打つ。"""
    idx = P.build_index(agents, radius)
    want = bool(P.hearers_of(speaker, agents, radius, occluder=occluder))
    for src, tag in ((idx, "index"), (agents, "legacy")):
        cnt = P.count_hearers(speaker, src, radius, occluder=occluder)
        got = P.has_hearers(speaker, src, radius, occluder=occluder)
        assert isinstance(got, bool), (label, tag, type(got))
        assert got == (cnt > 0), (label, tag, speaker, cnt, got)
        assert got == want, (label, tag, speaker, want, got)
    return want


# =========================================================================== #
# (1) ★同値 — has == (count > 0)
# =========================================================================== #
def test_has_equals_count_gt_zero_on_a_grid():
    agents = _grid()
    seen_true = seen_false = 0
    for speaker in agents:
        got = _assert_equiv(speaker, agents, 40.0, label="grid")
        seen_true += int(got)
        seen_false += int(not got)
    assert seen_true > 0, "全員 False のグリッドでは検算にならない"


@pytest.mark.parametrize("radius", [0.0, 1.0, 2.9999, 3.0, 5.0, 12.5, 40.0, 400.0])
def test_has_equals_count_across_radii(radius):
    agents = _grid()
    seen = set()
    for speaker in agents:
        seen.add(_assert_equiv(speaker, agents, radius, label=f"r={radius}"))
    assert seen, "空の入力"


def test_sparse_world_is_all_false():
    """誰も圏内に居ない配置では全員 False(片側だけの検算にしない)。"""
    agents = [_A(i, i * 500.0, 0.0) for i in range(30)]
    for speaker in agents:
        assert _assert_equiv(speaker, agents, 40.0, label="sparse") is False


def test_single_agent_world_and_empty_neighbourhood():
    solo = [_A(0, 1.0, 2.0)]
    assert _assert_equiv(solo[0], solo, 40.0, label="solo") is False
    lonely = _A(999, 10_000.0, 10_000.0)
    agents = _grid() + [lonely]
    assert _assert_equiv(lonely, agents, 40.0, label="lonely") is False


def test_outside_hears_nobody():
    agents = _grid()
    speaker = _A(500, 0.0, 0.0, loc="outside")
    assert _assert_equiv(speaker, agents + [speaker], 40.0, label="outside") is False


def test_sleeping_and_other_contexts_are_excluded():
    """睡眠中・屋外/屋内・階違い・範囲外の除外規則が `count_hearers` と同一。"""
    speaker = _A(0, 0.0, 0.0, building="b1", floor=3)
    agents = [
        speaker,
        _A(1, 1.0, 1.0, building="b1", floor=3),                  # 同じ階=聞こえる
        _A(2, 1.0, 1.0, building="b1", floor=4),                  # 別の階
        _A(3, 1.0, 1.0, building="b2", floor=3),                  # 別の建物
        _A(4, 1.0, 1.0),                                          # 路上
        _A(5, 1.0, 1.0, building="b1", floor=3, sleeping=True),    # 睡眠中
        _A(7, 1.0, 1.0, building="b1", floor=3, loc="outside"),    # 範囲外
    ]
    assert _assert_equiv(speaker, agents, 40.0, label="ctx-in") is True
    assert _assert_equiv(agents[4], agents, 40.0, label="ctx-street") is False
    # 相手が全員就寝している路上の話者 → False
    lone = [_A(100, 0.0, 0.0)] + [_A(100 + i, i * 1.0, 0.0, sleeping=True)
                                  for i in range(1, 20)]
    assert _assert_equiv(lone[0], lone, 40.0, label="asleep") is False


def test_has_equals_count_at_scale():
    """数千体の高密度セルでも 1 件もずれない(決定論の擬似ランダム配置)。"""
    agents = [_A(i, (i * 37) % 79 * 0.5, (i * 53) % 83 * 0.5) for i in range(3000)]
    for speaker in (agents[0], agents[7], agents[1234], agents[2999]):
        assert _assert_equiv(speaker, agents, 40.0, label="scale") is True
    idx = P.build_index(agents, 40.0)
    assert P.count_hearers(agents[1234], idx, 40.0) > 100, "密度が足りない検算"


def test_has_equals_count_on_scattered_float_coordinates():
    """格子ではない「汚い」float 座標でも一致する(丸めの取り違えを検出)。"""
    agents = []
    for i in range(1200):
        ang = (i * 0.7391) % (2.0 * math.pi)
        rad = 0.113 * ((i * 977) % 401)
        agents.append(_A(i, rad * math.cos(ang) + 61.7, rad * math.sin(ang) - 13.3))
    for speaker in agents[::37]:
        _assert_equiv(speaker, agents, 40.0, label="scatter")


# --------------------------------------------------------------------------- #
# 境界 — 距離ちょうど半径 / 半径まわり ±ULP の帯(第150 の既知罠)
# --------------------------------------------------------------------------- #
def test_exact_boundary_distance_is_inside():
    """距離がちょうど半径のとき圏内(`<=`)。厳密に表せる三角形で固定する。"""
    speaker = _A(0, 0.0, 0.0)
    inside = [speaker, _A(1, 24.0, 32.0)]              # hypot = 40.0 ちょうど
    assert _assert_equiv(speaker, inside, 40.0, label="edge-in") is True
    outside = [speaker,
               _A(5, math.nextafter(40.0, math.inf), 0.0),   # 1 ULP だけ外
               _A(6, 24.0, math.nextafter(32.0, math.inf))]  # わずかに外
    assert _assert_equiv(speaker, outside, 40.0, label="edge-out") is False


@pytest.mark.parametrize("radius", [5.0, 40.0])
def test_ulp_band_around_the_radius_never_diverges(radius, monkeypatch):
    """★半径の ±数 ULP に散らした点でも `count > 0` と 1 度もずれない。

    `np.hypot` と `math.hypot` は最大 1 ULP ずれうる(第150 の既知罠)。`_exists` の
    ベクトル化経路(②)は `_count` と同じ `_radius_band` の帯で `math.hypot` へ裁定を
    落としているので、帯に入る点しか居ない配置でも判定が割れない。
    ★ここは**必ず②を通す**(⓪ループ / ①先読みは `math.hypot` 直で自明に一致するため、
      素通りさせるとこのテストが空回りする)。
    """
    monkeypatch.setattr(P, "_EXISTS_LOOP_MAX", 0)     # ⓪ を無効化
    monkeypatch.setattr(P, "_EXISTS_PROBE_MAX", 0)    # ① を無効化 = ②(numpy)一択
    lo, hi = P._radius_band(radius)
    speaker = _A(0, 0.0, 0.0)
    band_only: list = [speaker]
    i = 1
    hit_band = 0
    for a in range(180):
        ang = a * (math.pi / 90.0)
        for shift in range(-3, 4):
            d = radius
            for _ in range(abs(shift)):
                d = math.nextafter(d, math.inf if shift > 0 else -math.inf)
            x, y = d * math.cos(ang), d * math.sin(ang)
            h = float(np.hypot(x, y))
            if lo < h <= hi:
                hit_band += 1
            band_only.append(_A(i, x, y))
            i += 1
    assert hit_band > 0, "帯に 1 点も入っていない(このテストが空回りしている)"
    _assert_equiv(speaker, band_only, radius, label=f"band r={radius}")
    # 帯に入る点「だけ」を残した配置(= `h <= lo` の即決枝が 1 度も使えない)でも一致する
    only = [speaker] + [a for a in band_only[1:]
                        if lo < float(np.hypot(a.x, a.y)) <= hi]
    assert len(only) > 1, "帯だけの配置が作れていない"
    _assert_equiv(speaker, only, radius, label=f"band-only r={radius}")


def test_zero_radius_matches_exact_overlaps_only():
    speaker = _A(0, 3.5, -2.25)
    same = [speaker, _A(1, 3.5, -2.25), _A(4, 9.0, 9.0)]
    assert _assert_equiv(speaker, same, 0.0, label="r0-hit") is True
    off = [speaker, _A(3, math.nextafter(3.5, math.inf), -2.25), _A(4, 9.0, 9.0)]
    assert _assert_equiv(speaker, off, 0.0, label="r0-miss") is False


# --------------------------------------------------------------------------- #
# 走査の構造 — 3 経路(⓪疎ループ / ①中心セル先読み / ②9セル連結の全長パス)の境界
# --------------------------------------------------------------------------- #
def _far_corner_filler(n: int, start_id: int) -> list:
    """半径 40 の話者(0.5, 0.5)から**必ず圏外**で、かつ**同じ粗セル**に入る詰め物。

    粗セルは幅 40m なので [0,40) × [0,40) が中心セル。その遠い角(≈38m 付近)は
    原点近傍から 53m 以上離れる = 圏外(40m)。
    """
    return [_A(start_id + k, 38.0 + (k % 17) * 0.1, 38.0 + (k // 17 % 17) * 0.1)
            for k in range(n)]


@pytest.mark.parametrize(
    "n_filler",
    [0, 1,
     P._EXISTS_PROBE_MAX - 2, P._EXISTS_PROBE_MAX - 1, P._EXISTS_PROBE_MAX,
     P._EXISTS_PROBE_MAX + 1,
     P._EXISTS_LOOP_MAX - 1, P._EXISTS_LOOP_MAX, P._EXISTS_LOOP_MAX + 1,
     2 * P._EXISTS_LOOP_MAX + 3, 600])
def test_hit_behind_the_probe_window_is_still_found(n_filler):
    """★経路の切り替え点で取りこぼさない: 唯一の圏内個体が**先読み窓の外**でも True。

    圏外の詰め物を中心セルの先頭に置くので、`_EXISTS_PROBE_MAX` を超えたところに
    唯一の圏内個体が来る = ①が必ず外れて②へ落ちる経路も通る。詰め物の数で
    ⓪(疎ループ)/ ①+② の切り替えも跨ぐ。
    """
    speaker = _A(0, 0.5, 0.5)
    agents = [speaker] + _far_corner_filler(n_filler, 1) + [_A(10 ** 6, 1.0, 1.0)]
    assert _assert_equiv(speaker, agents, 40.0, label=f"tail n={n_filler}") is True
    # 圏内個体を外すと False(詰め物だけでは当たらない = 検算が空回りしていない)
    assert _assert_equiv(speaker, agents[:-1], 40.0,
                         label=f"tail-none n={n_filler}") is False


def test_all_three_scan_paths_agree_across_the_switch_points(monkeypatch):
    """★しきい値(`_EXISTS_LOOP_MAX` / `_EXISTS_PROBE_MAX`)は**答えを変えない**。

    どの値に差し替えても同じ bool = 「速さだけのつまみ」であることの機械証明
    (`world.perception_fine_gate` の粗/細同値と同型の釘)。
    """
    agents = [_A(i, (i * 37) % 79 * 0.7, (i * 53) % 83 * 0.7) for i in range(500)]
    agents += [_A(900 + i, 500.0 + i * 3.0, 500.0) for i in range(20)]   # 疎な一群
    agents += [_A(950 + i, 9000.0 + i * 100.0, 0.0) for i in range(10)]  # 孤立
    idx0 = P.build_index(agents, 40.0)
    want = [P.count_hearers(s, idx0, 40.0) > 0 for s in agents]
    assert any(want) and not all(want), "片側しか出ていない入力集合"
    for loop_max, probe_max in ((0, 0), (0, 1), (1, 0), (8, 4), (64, 32),
                                (10 ** 9, 1), (0, 10 ** 9), (10 ** 9, 10 ** 9)):
        monkeypatch.setattr(P, "_EXISTS_LOOP_MAX", loop_max)
        monkeypatch.setattr(P, "_EXISTS_PROBE_MAX", probe_max)
        idx = P.build_index(agents, 40.0)
        got = [P.has_hearers(s, idx, 40.0) for s in agents]
        assert got == want, (loop_max, probe_max)
        # legacy(索引なし)経路はしきい値と無関係に同じ
        assert [P.has_hearers(s, agents, 40.0) for s in agents] == want


def test_hit_only_in_a_non_centre_cell_is_found():
    """中心セル優先で走査しても、隣接セルにしか居ない相手を取りこぼさない(全 8 方向)。"""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            speaker = _A(0, 20.0, 20.0)                 # セル (0,0) の中央
            other = _A(1, 20.0 + dx * 30.0, 20.0 + dy * 30.0)
            # 距離 30 or 42.4 → 半径 45 なら常に圏内・セルは必ず (dx, dy) 側
            assert P._radius_band(45.0)                 # 帯キャッシュを温める(副作用なし)
            got = _assert_equiv(speaker, [speaker, other], 45.0,
                                label=f"cell {dx},{dy}")
            assert got is True, (dx, dy)


def test_speaker_not_in_the_index_still_matches_count():
    """話者が索引に載っていない(睡眠/範囲外)場合も `count` と同じ答え。"""
    others = _grid(5, 4.0)
    for flag in ({"sleeping": True}, {"loc": "outside"}):
        speaker = _A(777, 4.0, 4.0, **flag)
        idx = P.build_index(others + [speaker], 40.0)   # add() が話者を弾く
        assert P.has_hearers(speaker, idx, 40.0) == \
            (P.count_hearers(speaker, idx, 40.0) > 0)


# =========================================================================== #
# (2) ★早期打ち切りが実際に効いている
# =========================================================================== #
def test_exists_does_not_build_the_full_neighbourhood():
    """★密なセルでは 9 セル連結(`_nc`)も per-cell numpy 配列(`_np`)も 1 本も作らない。

    「全部計算してから any()」なら `_count` と同じ `_count_arrays` を通るはずなので、
    ここが両方とも空であることが早期打ち切り(①中心セル先読み)の構造的証拠。
    """
    agents = [_A(i, 5.0 + (i % 20) * 0.5, 5.0 + (i // 20) * 0.5) for i in range(400)]
    # 近傍 8 セルにも人を置く(「連結すると重い」状況を実際に作る)
    for j in range(400):
        agents.append(_A(1000 + j, 45.0 + (j % 20) * 0.5, 5.0 + (j // 20) * 0.5))
    idx = P.build_index(agents, 40.0)
    assert P.has_hearers(agents[0], idx, 40.0) is True
    assert idx._nc == {}, "9 セル連結を作っている(早期打ち切りになっていない)"
    assert idx._np == {}, "per-cell numpy 配列を作っている(先読みが numpy へ落ちている)"
    # 参照側(`_count`)は連結を作る = 上の断言が「そもそも作られない実装」ではない証拠
    idx2 = P.build_index(agents, 40.0)
    P.count_hearers(agents[0], idx2, 40.0)
    assert idx2._nc, "_count が連結を作っていない(前提が崩れた)"


def test_sparse_neighbourhood_never_touches_numpy():
    """疎な近傍(⓪ループ経路)は numpy キャッシュを 1 本も作らない(固定費ゼロ)。"""
    agents = [_A(i, i * 500.0, 0.0) for i in range(50)]
    idx = P.build_index(agents, 40.0)
    for speaker in agents:
        assert P.has_hearers(speaker, idx, 40.0) is False
    assert idx._np == {} and idx._nc == {}


def test_dense_miss_falls_through_to_the_full_pass():
    """①が外れる配置では②(9 セル連結の全長パス)へ落ちて、正しく False を返す。"""
    speaker = _A(0, 0.5, 0.5)
    # 中心セルの遠い角に大量に詰める(全員 40m 圏外)+ 隣セルにも圏外で大量に置く
    agents = [speaker] + _far_corner_filler(300, 1)
    agents += [_A(10_000 + j, 70.0 + (j % 17) * 0.3, 70.0 + (j // 17) * 0.3)
               for j in range(300)]
    idx = P.build_index(agents, 40.0)
    assert P.count_hearers(speaker, idx, 40.0) == 0, "前提が崩れた(圏内が居る)"
    assert P.has_hearers(speaker, idx, 40.0) is False
    assert idx._nc, "②(9 セル連結)へ落ちていない"


def test_exists_stops_early_in_the_occluder_loop():
    """遮蔽器経路も最初の 1 件で打ち切る(`blocks` の呼び出しが `count` より少ない)。"""
    agents = [_A(i, 5.0 + (i % 20) * 0.5, 5.0 + (i // 20) * 0.5) for i in range(400)]
    idx_h = P.build_index(agents, 40.0)
    idx_c = P.build_index(agents, 40.0)
    occ_h, occ_c = _CountingOcc(), _CountingOcc()
    assert P.has_hearers(agents[0], idx_h, 40.0, occluder=occ_h) is True
    assert P.count_hearers(agents[0], idx_c, 40.0, occluder=occ_c) > 0
    assert occ_h.calls == 1, occ_h.calls
    assert occ_c.calls > 100, occ_c.calls


def test_legacy_path_stops_early_too():
    agents = [_A(i, 1.0 * i, 0.0) for i in range(500)]
    occ_h, occ_c = _CountingOcc(), _CountingOcc()
    assert P.has_hearers(agents[0], agents, 40.0, occluder=occ_h) is True
    assert P.count_hearers(agents[0], agents, 40.0, occluder=occ_c) > 0
    assert occ_h.calls == 1 and occ_c.calls > 1


# =========================================================================== #
# (3) 遮蔽 — 引数 / 索引据え付け / モジュール既定のどれでも `count > 0` と同値
# =========================================================================== #
def test_occluder_argument_matches_count():
    agents = _grid(6, 2.0)
    for speaker in agents:
        _assert_equiv(speaker, agents, 40.0, occluder=_BlocksOdd(), label="occ-arg")
    # 遮蔽で全滅する配置(偶数 id の相手が 1 人も居ない)では False へ落ちる
    speaker = _A(0, 0.0, 0.0)
    only_odd = [speaker, _A(1, 1.0, 0.0), _A(3, 2.0, 0.0)]
    assert _assert_equiv(speaker, only_odd, 40.0, occluder=_BlocksOdd(),
                         label="occ-all-blocked") is False


def test_occluder_installed_on_the_index_is_honoured():
    agents = _grid(6, 2.0)
    speaker = agents[14]
    occ = _CountingOcc()
    idx = P.build_index(agents, 40.0, occluder=occ)
    assert P.has_hearers(speaker, idx, 40.0) is True
    assert occ.calls > 0, "索引据え付けの遮蔽器が使われていない"


def test_module_level_occluder_is_honoured():
    agents = _grid(6, 2.0)
    speaker = agents[14]
    idx = P.build_index(agents, 40.0)
    P.install_occluder(_BlocksOdd())
    try:
        want = P.count_hearers(speaker, idx, 40.0) > 0
        assert P.has_hearers(speaker, idx, 40.0) == want
        assert P.has_hearers(speaker, agents, 40.0) == want
    finally:
        P.clear_occluder()
    assert P.active_occluder() is None
    assert P.has_hearers(speaker, idx, 40.0) == \
        (P.count_hearers(speaker, idx, 40.0) > 0)


# =========================================================================== #
# (4) 契約 — 引数面 / 索引キャッシュの共有
# =========================================================================== #
def test_has_hearers_takes_no_bounding_arguments():
    """`count_hearers` と同じ引数面(cap / radius_eff は受け取らない)。"""
    sig = inspect.signature(P.has_hearers)
    assert list(sig.parameters) == list(inspect.signature(P.count_hearers).parameters)
    assert list(sig.parameters) == ["speaker", "agents_or_index", "radius_m",
                                    "occluder"]
    with pytest.raises(TypeError):
        P.has_hearers(_A(0, 0.0, 0.0), [], 40.0, cap=5)
    with pytest.raises(TypeError):
        P.has_hearers(_A(0, 0.0, 0.0), [], 40.0, radius_eff=5.0)


def test_sharing_the_index_cache_does_not_change_anything():
    """has_hearers が育てたセル配列キャッシュを、既定経路も count も有界経路も使える。"""
    agents = _grid()
    speaker = agents[57]
    idx_a = P.build_index(agents, 40.0)
    want_plain = [a.id for a in P.hearers_of(speaker, idx_a, 40.0)]
    want_cap = [a.id for a in P.hearers_of(speaker, idx_a, 40.0, cap=9)]
    want_cnt = P.count_hearers(speaker, idx_a, 40.0)

    idx_b = P.build_index(agents, 40.0)
    assert P.has_hearers(speaker, idx_b, 40.0) is True
    assert [a.id for a in P.hearers_of(speaker, idx_b, 40.0)] == want_plain
    assert [a.id for a in P.hearers_of(speaker, idx_b, 40.0, cap=9)] == want_cap
    assert P.count_hearers(speaker, idx_b, 40.0) == want_cnt
    # 逆順(count / 有界経路が育てたキャッシュを has が使う)でも一致
    idx_c = P.build_index(agents, 40.0)
    P.count_hearers(speaker, idx_c, 40.0)
    P.hearers_of(speaker, idx_c, 40.0, cap=9, radius_eff=20.0)
    assert P.has_hearers(speaker, idx_c, 40.0) is True
    # 索引は step ごとに作り直される前提: 同じ索引を何度引いても答えが揺れない
    for _ in range(3):
        assert P.has_hearers(speaker, idx_c, 40.0) is True


# =========================================================================== #
# (5) ★実ラン(mock)の L1 完全一致 — 修正前の式へ差し替えた B ランと比べる
# =========================================================================== #
def _cfg(name, n_steps=24, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def test_l1_matches_pre_fix_expression(tmp_path, monkeypatch):
    """★A(現行 has_hearers)と B(修正前 bool(hearers_of))の L1 がバイト一致。

    `scheduler` 名前空間の `has_hearers` は `_phase_drive` の 3 か所と `_decide_g` の
    1 か所からしか呼ばれない(いずれも第154 A1/A2 の差し替え地点)ので、ここを
    「列挙して bool を取る」実装へ戻すことは修正前のコードを走らせることと厳密に同じ
    (引数も半径も知覚ソースも呼び順も評価順もそのまま)。
    """
    sim_a = Simulation(_cfg("has_a"), out_dir=tmp_path / "has_a")
    sim_a.run()
    l1_a = _l1(sim_a)
    assert l1_a, "L1 が空のランで比べても意味がない"

    calls = {"n": 0, "with": 0, "without": 0}

    def _legacy(speaker, src, radius, occluder=None):
        calls["n"] += 1
        got = bool(P.hearers_of(speaker, src, radius, occluder=occluder))
        calls["with" if got else "without"] += 1
        return got

    monkeypatch.setattr(scheduler, "has_hearers", _legacy)
    sim_b = Simulation(_cfg("has_b"), out_dir=tmp_path / "has_b")
    sim_b.run()

    assert calls["n"] > 0, "差し替えた経路が 1 度も通っていない(前提が崩れた)"
    # 片側しか出ていないと「同席あり」側の分岐(drive 加算 / face 発火)を 1 度も通らない
    assert calls["with"] > 0 and calls["without"] > 0, calls
    assert _l1(sim_b) == l1_a
    assert len(getattr(sim_b.logger, "llm_calls", [])) == \
        len(getattr(sim_a.logger, "llm_calls", []))


def test_real_run_agents_agree_with_count(tmp_path):
    """★実ラン(mock)の実個体・実索引で `has == (count > 0)` を全員ぶん照合する。"""
    sim = Simulation(_cfg("has_real", 8, 40), out_dir=tmp_path / "has_real")
    sim.run()
    radius = float(sim.cfg.world.perception_radius_m)
    idx = P.build_index(sim.agents, radius)
    checked = nonzero = 0
    for agent in sim.agents:
        want = P.count_hearers(agent, idx, radius) > 0
        assert P.has_hearers(agent, idx, radius) == want, agent.id
        assert P.has_hearers(agent, sim.agents, radius) == want, agent.id
        assert bool(P.hearers_of(agent, sim.agents, radius)) == want, agent.id
        checked += 1
        nonzero += int(want)
    assert checked == len(sim.agents)
    assert nonzero > 0, "全員 False のランで検算しても意味がない"


# =========================================================================== #
# (6) 回帰ガード — 同席判定でリストを作らない
# =========================================================================== #
def _body(fn) -> str:
    return "\n".join(ln for ln in inspect.getsource(fn).splitlines()
                     if not ln.lstrip().startswith("#"))


def test_decide_g_and_phase_drive_do_not_enumerate_for_a_bool():
    """`_decide_g` / `_phase_drive` の同席判定が列挙(`hearers_of`)へ戻っていない。"""
    dg = _body(scheduler._decide_g)
    assert "has_hearers(" in dg, "存在判定経路が消えている"
    assert "hearers_of(" not in dg, "リスト構築が戻っている(40m 圏の全列挙)"
    assert "count_hearers(" not in dg, "人数経路へ戻っている(1 ビットで足りる)"
    pd = _body(scheduler._phase_drive)
    assert pd.count("has_hearers(") == 3, "face 判定 3 か所が存在判定になっていない"
    assert "hearers_of(" not in pd, "リスト構築が戻っている(第152 と同型のバグ)"
