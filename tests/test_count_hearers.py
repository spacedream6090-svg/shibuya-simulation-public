"""第150 小修正: `perception.count_hearers`(人数だけを数える経路)のテスト。

背景: 観測チャンネル `ext.encounter`(cognition/channels.py)は「声が届く**人数**」しか
使わないのに `len(hearers_of(...))` を呼んでいた = 全個体ぶんのリスト構築(遮蔽判定 +
id 昇順ソート)を毎 step 捨てるために作っていた(25k 夕方の py-spy で step 時間の 15.8%)。

守るもの(検収基準の順)
  (1) ★同値: `count_hearers(...) == len(hearers_of(...))` が**全入力で**成り立つ
      (索引経路 / legacy 経路 / 遮蔽あり・なし / 境界距離ちょうど / 空セル / 屋内外)。
  (2) 半径判定の境界: 距離がちょうど半径のとき圏内(<=)。半径まわり ±数 ULP の帯でも
      `hearers_of`(= `math.hypot`)と 1 件もずれない(ベクトル化の裁定が効いている)。
  (3) cap / radius_eff を受け取らない = このチャンネルの現行 semantics を保つ。
  (4) 索引のキャッシュを共有しても `hearers` / 有界経路の答えが変わらない。
  (5) 実ラン(mock)の `ext.encounter` が `len(hearers_of(...))` と一致する
      = G 観測の出力値がバイト一致。
検証は mock のみ(実 LLM 禁止・≤24step)。乱数は 1 本も引かない。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from society.cognition import channels as CH
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import perception as P


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_hearer_cap.py と同型の最小 duck-type)
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


class _NeverBlocks:
    def __init__(self):
        self.calls = 0

    def blocks(self, a, b) -> bool:
        self.calls += 1
        return False


class _BlocksOdd:
    def blocks(self, a, b) -> bool:
        return bool(b.id % 2)


def _grid(n_side: int = 12, pitch: float = 3.0) -> list:
    out = []
    i = 0
    for gx in range(n_side):
        for gy in range(n_side):
            out.append(_A(i, gx * pitch, gy * pitch))
            i += 1
    return out


def _both(speaker, agents, radius, occluder=None):
    """(索引経路の count, legacy 経路の count, 参照 len(hearers_of)) を返す。"""
    idx = P.build_index(agents, radius)
    want = len(P.hearers_of(speaker, idx, radius, occluder=occluder))
    want_legacy = len(P.hearers_of(speaker, agents, radius, occluder=occluder))
    assert want == want_legacy, "前提が崩れた(索引と legacy で hearers_of が不一致)"
    return (P.count_hearers(speaker, idx, radius, occluder=occluder),
            P.count_hearers(speaker, agents, radius, occluder=occluder),
            want)


# --------------------------------------------------------------------------- #
# (1) ★同値 — count == len(hearers_of)
# --------------------------------------------------------------------------- #
def test_count_equals_len_on_a_grid():
    agents = _grid()
    for speaker in agents:
        got_idx, got_legacy, want = _both(speaker, agents, 40.0)
        assert got_idx == want, speaker
        assert got_legacy == want, speaker
    # 「全部 0 で偶然一致」ではない
    assert P.count_hearers(agents[57], P.build_index(agents, 40.0), 40.0) > 0


@pytest.mark.parametrize("radius", [0.0, 1.0, 2.9999, 3.0, 5.0, 12.5, 40.0, 400.0])
def test_count_equals_len_across_radii(radius):
    agents = _grid()
    for speaker in (agents[0], agents[7], agents[57], agents[100], agents[-1]):
        got_idx, got_legacy, want = _both(speaker, agents, radius)
        assert got_idx == want == got_legacy, (speaker, radius)


def test_count_is_zero_in_an_empty_neighborhood():
    """近傍 9 セルに誰も居ない(遠く離れた孤立個体)= 0 件。"""
    lonely = _A(999, 10_000.0, 10_000.0)
    agents = _grid() + [lonely]
    got_idx, got_legacy, want = _both(lonely, agents, 40.0)
    assert want == 0 and got_idx == 0 and got_legacy == 0


def test_count_of_a_single_agent_world_is_zero():
    solo = [_A(0, 1.0, 2.0)]
    got_idx, got_legacy, want = _both(solo[0], solo, 40.0)
    assert (got_idx, got_legacy, want) == (0, 0, 0)


def test_outside_hears_nobody():
    agents = _grid()
    speaker = _A(500, 0.0, 0.0, loc="outside")
    agents = agents + [speaker]
    got_idx, got_legacy, want = _both(speaker, agents, 40.0)
    assert (got_idx, got_legacy, want) == (0, 0, 0)


def test_sleeping_and_other_contexts_are_excluded():
    """睡眠中・屋外/屋内・階違いの除外規則が `hearers_of` と同一。"""
    speaker = _A(0, 0.0, 0.0, building="b1", floor=3)
    agents = [
        speaker,
        _A(1, 1.0, 1.0, building="b1", floor=3),                   # 同じ階=聞こえる
        _A(2, 1.0, 1.0, building="b1", floor=4),                   # 別の階
        _A(3, 1.0, 1.0, building="b2", floor=3),                   # 別の建物
        _A(4, 1.0, 1.0),                                           # 路上
        _A(5, 1.0, 1.0, building="b1", floor=3, sleeping=True),     # 睡眠中
        _A(6, 2.0, 2.0, building="b1", floor=3),                   # 同じ階=聞こえる
        _A(7, 1.0, 1.0, building="b1", floor=3, loc="outside"),     # 範囲外
    ]
    got_idx, got_legacy, want = _both(speaker, agents, 40.0)
    assert want == 2, want
    assert got_idx == 2 and got_legacy == 2
    # 路上の話者から見ると路上の 1 人だけ
    got_idx, got_legacy, want = _both(agents[4], agents, 40.0)
    assert (got_idx, got_legacy, want) == (0, 0, 0)


def test_count_equals_len_at_scale():
    """数千体の高密度セルでも 1 件もずれない(決定論の擬似ランダム配置)。"""
    agents = [_A(i, (i * 37) % 79 * 0.5, (i * 53) % 83 * 0.5) for i in range(3000)]
    idx = P.build_index(agents, 40.0)
    for speaker in (agents[0], agents[7], agents[1234], agents[2999]):
        want = len(P.hearers_of(speaker, idx, 40.0))
        assert P.count_hearers(speaker, idx, 40.0) == want, speaker
        assert P.count_hearers(speaker, agents, 40.0) == want, speaker
    assert P.count_hearers(agents[1234], idx, 40.0) > 100, "密度が足りない検算"


def test_count_equals_len_on_scattered_float_coordinates():
    """格子ではない「汚い」float 座標でも一致する(丸めの取り違えを検出)。"""
    agents = []
    for i in range(1200):
        ang = (i * 0.7391) % (2.0 * math.pi)
        rad = 0.113 * ((i * 977) % 401)
        agents.append(_A(i, rad * math.cos(ang) + 61.7, rad * math.sin(ang) - 13.3))
    idx = P.build_index(agents, 40.0)
    for speaker in agents[::37]:
        want = len(P.hearers_of(speaker, idx, 40.0))
        assert P.count_hearers(speaker, idx, 40.0) == want, speaker
        assert P.count_hearers(speaker, agents, 40.0) == want, speaker


# --------------------------------------------------------------------------- #
# (2) 境界 — 距離ちょうど半径 / 半径まわり ±数 ULP の帯
# --------------------------------------------------------------------------- #
def test_exact_boundary_distance_is_inside():
    """距離がちょうど半径のとき圏内(`<=`)。厳密に表せる三角形で固定する。"""
    speaker = _A(0, 0.0, 0.0)
    others = [
        _A(1, 24.0, 32.0),    # hypot = 40.0 ちょうど(3-4-5 の 8 倍)
        _A(2, 40.0, 0.0),     # 40.0 ちょうど
        _A(3, 0.0, -40.0),    # 40.0 ちょうど
        _A(4, -24.0, -32.0),  # 40.0 ちょうど
        _A(5, math.nextafter(40.0, math.inf), 0.0),   # 1 ULP だけ外
        _A(6, 24.0, math.nextafter(32.0, math.inf)),  # わずかに外
    ]
    agents = [speaker] + others
    got_idx, got_legacy, want = _both(speaker, agents, 40.0)
    assert want == 4, want
    assert got_idx == 4 and got_legacy == 4


@pytest.mark.parametrize("radius", [5.0, 40.0])
def test_ulp_band_around_the_radius_never_diverges(radius):
    """★半径の ±数 ULP に散らした点でも `hearers_of` と 1 件もずれない。

    `np.hypot` と `math.hypot` は最大 1 ULP ずれうるので、ベクトル化が素朴だとここで
    落ちる(帯の裁定が効いていることの証明)。角度を振って帯に実際に入る点を作る。
    """
    agents = [_A(0, 0.0, 0.0)]
    i = 1
    hit_band = 0
    lo, hi = P._radius_band(radius)
    for a in range(180):
        ang = a * (math.pi / 90.0)
        for shift in range(-3, 4):
            d = radius
            for _ in range(abs(shift)):
                d = math.nextafter(d, math.inf if shift > 0 else -math.inf)
            x, y = d * math.cos(ang), d * math.sin(ang)
            agents.append(_A(i, x, y))
            h = float(np.hypot(x, y))
            if lo < h <= hi:
                hit_band += 1
            i += 1
    assert hit_band > 0, "帯に 1 点も入っていない(このテストが空回りしている)"
    speaker = agents[0]
    got_idx, got_legacy, want = _both(speaker, agents, radius)
    assert got_idx == want, (got_idx, want)
    assert got_legacy == want, (got_legacy, want)


def test_radius_band_brackets_the_radius():
    for radius in (0.5, 5.0, 40.0, 1234.5):
        lo, hi = P._radius_band(radius)
        assert lo < radius < hi
        step = hi
        for _ in range(P._BAND_ULP):
            step = math.nextafter(step, -math.inf)
        assert step == radius, "hi が半径から所定の ULP だけ離れていない"


def test_zero_radius_counts_only_exact_overlaps():
    speaker = _A(0, 3.5, -2.25)
    agents = [speaker, _A(1, 3.5, -2.25), _A(2, 3.5, -2.25),
              _A(3, math.nextafter(3.5, math.inf), -2.25), _A(4, 9.0, 9.0)]
    got_idx, got_legacy, want = _both(speaker, agents, 0.0)
    assert want == 2, want
    assert got_idx == 2 and got_legacy == 2


# --------------------------------------------------------------------------- #
# (3) 遮蔽 — 引数 / 索引据え付け / モジュール既定のどれでも `hearers_of` と同値
# --------------------------------------------------------------------------- #
def test_count_matches_len_with_an_occluder_argument():
    agents = _grid(6, 2.0)
    for speaker in agents:
        got_idx, got_legacy, want = _both(speaker, agents, 40.0,
                                          occluder=_BlocksOdd())
        assert got_idx == want == got_legacy, speaker
    # 遮蔽ありは遮蔽なしより必ず少ない(空回り防止)
    idx = P.build_index(agents, 40.0)
    assert P.count_hearers(agents[14], idx, 40.0, occluder=_BlocksOdd()) < \
        P.count_hearers(agents[14], idx, 40.0)


def test_occluder_installed_on_the_index_is_honoured():
    agents = _grid(6, 2.0)
    speaker = agents[14]
    occ = _NeverBlocks()
    idx = P.build_index(agents, 40.0, occluder=occ)
    want = len(P.hearers_of(speaker, idx, 40.0))
    assert occ.calls > 0, "索引据え付けの遮蔽器が使われていない"
    before = occ.calls
    assert P.count_hearers(speaker, idx, 40.0) == want
    assert occ.calls > before, "count 側で遮蔽器が呼ばれていない(ループ経路へ落ちていない)"


def test_module_level_occluder_is_honoured():
    agents = _grid(6, 2.0)
    speaker = agents[14]
    idx = P.build_index(agents, 40.0)
    P.install_occluder(_BlocksOdd())
    try:
        want = len(P.hearers_of(speaker, idx, 40.0))
        assert P.count_hearers(speaker, idx, 40.0) == want
        assert P.count_hearers(speaker, agents, 40.0) == want
        assert all(a.id % 2 == 0 for a in P.hearers_of(speaker, idx, 40.0))
    finally:
        P.clear_occluder()
    assert P.active_occluder() is None
    # 解除後は遮蔽なしの人数へ戻る
    assert P.count_hearers(speaker, idx, 40.0) == len(P.hearers_of(speaker, idx, 40.0))


# --------------------------------------------------------------------------- #
# (4) 契約 — cap / radius_eff を受け取らない・索引キャッシュを壊さない
# --------------------------------------------------------------------------- #
def test_count_hearers_takes_no_bounding_arguments():
    """このチャンネルの意味は「物理的に声が届く人数」= 第149 の cap/声の段階に依らない。"""
    import inspect
    sig = inspect.signature(P.count_hearers)
    assert list(sig.parameters) == ["speaker", "agents_or_index", "radius_m",
                                    "occluder"]
    with pytest.raises(TypeError):
        P.count_hearers(_A(0, 0.0, 0.0), [], 40.0, cap=5)
    with pytest.raises(TypeError):
        P.count_hearers(_A(0, 0.0, 0.0), [], 40.0, radius_eff=5.0)


def test_count_is_unaffected_by_cap_and_speech_radius():
    """cap を掛けた `hearers_of` より多い = 絞られていない現行 semantics のまま。"""
    agents = _grid()
    speaker = agents[57]
    idx = P.build_index(agents, 40.0)
    full = P.count_hearers(speaker, idx, 40.0)
    assert full == len(P.hearers_of(speaker, idx, 40.0))
    assert full > len(P.hearers_of(speaker, idx, 40.0, cap=6))
    assert full > len(P.hearers_of(speaker, idx, 40.0, radius_eff=5.0))


def test_sharing_the_index_cache_does_not_change_hearers():
    """count が育てた numpy キャッシュを、既定経路も有界経路もそのまま使える。"""
    agents = _grid()
    speaker = agents[57]
    idx_a = P.build_index(agents, 40.0)
    idx_b = P.build_index(agents, 40.0)
    want_plain = [a.id for a in P.hearers_of(speaker, idx_a, 40.0)]
    want_cap = [a.id for a in P.hearers_of(speaker, idx_a, 40.0, cap=9)]
    # 先に count を回してキャッシュを育ててから同じことをする
    P.count_hearers(speaker, idx_b, 40.0)
    assert [a.id for a in P.hearers_of(speaker, idx_b, 40.0)] == want_plain
    assert [a.id for a in P.hearers_of(speaker, idx_b, 40.0, cap=9)] == want_cap
    # 逆順(有界経路が育てたキャッシュを count が使う)でも一致
    idx_c = P.build_index(agents, 40.0)
    P.hearers_of(speaker, idx_c, 40.0, cap=9, radius_eff=20.0)
    assert P.count_hearers(speaker, idx_c, 40.0) == len(want_plain)


# --------------------------------------------------------------------------- #
# (5) G 観測の出力値がバイト一致(channels.observe を実物で照合)
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 8, n_agents: int = 40, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_observe_encounter_column_matches_len_hearers_of(tmp_path):
    """★実ラン(mock)の `ext.encounter` が `len(hearers_of(...))` と 1 件も違わない。"""
    sim = Simulation(_cfg("cnt_obs", 8, 40,
                          **{"cognition.channels.enabled": "true"}),
                     out_dir=tmp_path / "cnt_obs")
    sim.run()
    col = [c.column for c in CH.CHANNELS].index("ext_encounter")
    radius = float(sim.cfg.world.perception_radius_m)
    rows = CH.observe(sim, 0, 0, 0)
    assert rows, "観測行が空(前提が崩れた)"
    by_id = {int(a.id): a for a in sim.agents}
    checked = nonzero = 0
    for row in rows:
        agent = by_id[int(row[2])]
        want = float(len(P.hearers_of(agent, sim.agents, radius)))
        got = row[len(CH.KEY_COLUMNS) + col]
        assert got == want, f"agent {agent.id}: {got} != {want}"
        checked += 1
        nonzero += int(want > 0)
    assert checked == len(sim.agents)
    assert nonzero > 0, "全員 0 のランで検算しても意味がない"
