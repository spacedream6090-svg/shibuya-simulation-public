"""歩車信号の同一化(``world.traffic.signal_from_crossings``。**既定 OFF**)のテスト。

何を守るのか
------------
同じ交差点なのに、車の信号(``world/traffic.py``)と歩行者の信号
(``world/sfm_core.SignalGate`` + ``data/crossings_shibuya.json``)が**別々の数字**で
動いていた:

  - 車     … 赤率 = 0.35 + 0.20 × sha256(node)、周期 = conf の全交差点共通の定数(90 秒)
  - 歩行者 … 交差点表の実測(渋谷: 周期 140 秒・青 37 秒・青点滅 10 秒 = 横断可能 47 秒)

= **同一交差点の 2 つの現示が統計的に独立**という、地図にも実測にも無い状態である。
本トグルは「交差点表に載っているノードだけ、車側もその交差点自身から導く」= 歩車が
同じ 1 台の装置になる、というところまでを行う(それ以上は主張しない)。

検収基準(この順に固定する)
  ① 出荷既定が false・レジストリ宣言あり・**挙動変更トグル**であることの明示
  ② OFF = 1 ビットも変わらない(交差点表を**読みもしない** = 従来の hash 経路)
  ③ 式: r = 1 − (green_s+flash_s)/cycle_s(140/37/10 → 0.664285…)+ clamp は保険
  ④ ON: 表に居るノードだけ表由来・**居ないノードは hash 値のままバイト一致**
  ⑤ 遅延式が**その交差点自身の周期**を使う(全交差点共通の定数ではない)
  ⑥ 決定論(静的データだけの純関数。同じ地図なら何度組んでも同じ表)
  ⑦ ON は実際に走行を変える(= 観測の追加ではないことを正直に固定する)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from society import registry as R
from society.config import load_config
from society.rng import RngHub
from society.world import traffic as T
from society.world.clock import Clock
from society.world.map import CityMap
from society.world.routing import Router
from society.world.traffic import TrafficFlow

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CORE = DATA / "shibuya_osm.json"
FEATURES = DATA / "traffic_features_shibuya.json"
CROSSINGS = DATA / "crossings_shibuya.json"

_needs_data = pytest.mark.skipif(
    not (CORE.exists() and FEATURES.exists() and CROSSINGS.exists()),
    reason="地図/交差点サイドカーが未生成")

#: 渋谷スクランブルの実測(交差点表 signal_defaults)。表の全 80 行がこの値。
CYCLE_S, GREEN_S, FLASH_S = 140.0, 37.0, 10.0
#: 期待する車の赤率 = 1 − 47/140(clamp は効かない = 帯の内側)
EXPECT_R = 1.0 - (GREEN_S + FLASH_S) / CYCLE_S


# --------------------------------------------------------------------------- #
# 共有(重い CityMap ロードを 1 回に)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def city():
    return CityMap(CORE)


def _tf(city, cfg, seed: int = 42) -> TrafficFlow:
    tf = TrafficFlow(city, Router(city), RngHub(seed), enabled=True,
                     cars_per_day=30000, max_log=120)
    tf.ensure_mode(cfg)
    return tf


def _od_cfg(**ov):
    dot = ["world.traffic.mode=od"] + [f"{k}={v}" for k, v in ov.items()]
    return load_config(overrides=dot)


def _trace(tf: TrafficFlow, steps: int = 24) -> list:
    clk = Clock()
    return [(len(tf.step(step, clk.sim_min(step))), tf.total_spawned,
             tf.total_arrived, tf.jam_events) for step in range(steps)]


# =========================================================================== #
# ① 出荷既定・宣言(挙動変更トグルであることを明示する)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.world.traffic.signal_from_crossings) is False
    assert str(cfg.world.traffic.crossings_file) == "data/crossings_shibuya.json"


def test_registry_declares_the_toggle():
    feat = {f.id: f for f in R.FEATURES}["world.traffic.signal_from_crossings"]
    assert feat.repro_tier == "strict"      # 静的データだけの純関数(乱数ゼロ)
    assert feat.affects_k is False          # generate() の呼び出し点不変
    assert feat.fingerprint_risk == "none"  # プロンプトへ 1 バイトも足さない
    assert feat.off_value is False
    # ★挙動変更であることが宣言文に書かれている(観測の追加だと誤読させない)
    assert "挙動変更" in feat.description
    assert R.undeclared_toggles(load_config()) == []


# =========================================================================== #
# ③ 式(純関数。地図が無くても動く)
# =========================================================================== #
def test_vehicle_red_ratio_formula():
    """r = 1 − (green+flash)/cycle。140/37/10 → 0.6642857…(= 1 − 47/140)。"""
    got = T.vehicle_red_ratio(CYCLE_S, GREEN_S, FLASH_S)
    assert got == EXPECT_R == pytest.approx(0.6642857142857143)
    assert round(got, 3) == 0.664
    # 現行の交差点表は 80 行すべてこの値 = clamp は 1 度も効かない
    assert T.RED_RATIO_MIN < got < T.RED_RATIO_MAX


def test_vehicle_red_ratio_clamp_is_only_a_guard():
    """clamp [0.20, 0.80] は**壊れた表行への保険**であって調整つまみではない。"""
    assert T.vehicle_red_ratio(100.0, 100.0, 10.0) == T.RED_RATIO_MIN   # 常に青
    assert T.vehicle_red_ratio(100.0, 0.0, 0.0) == T.RED_RATIO_MAX      # 常に赤
    assert T.vehicle_red_ratio(0.0, 37.0, 10.0) == 0.0                  # 周期不明→0
    assert (T.RED_RATIO_MIN, T.RED_RATIO_MAX) == (0.20, 0.80)


def test_crossing_id_join_is_an_identity_not_a_neighbour_search():
    """結合は OSM node id の完全一致だけ(近傍探索で同一性を捏造しない)。"""
    assert T.crossing_id_of_node("n1499530621") == 1499530621
    assert T.crossing_id_of_node("1499530621") is None    # "n" が無い = 別系の id
    assert T.crossing_id_of_node("nabc") is None
    assert T.crossing_id_of_node("") is None


@_needs_data
def test_table_loader_keeps_only_signalized_rows():
    """信号のある行だけを読む(信号無しの横断歩道は cycle_s を持たない = 入れない)。"""
    table = T.load_signalized_crossings(CROSSINGS)
    raw = json.loads(CROSSINGS.read_text(encoding="utf-8"))["crossings"]
    assert len(table) == sum(1 for r in raw if r.get("signal"))
    assert len(table) == 80 and len(raw) == 192
    for row in table.values():
        assert row["cycle_s"] > 0.0
        assert (row["cycle_s"], row["green_s"], row["flash_s"]) == \
            (CYCLE_S, GREEN_S, FLASH_S)
    assert T.load_signalized_crossings(DATA / "no_such_file.json") == {}


# =========================================================================== #
# ② OFF = 1 ビットも変わらない
# =========================================================================== #
@_needs_data
def test_off_keeps_the_hash_path_untouched(city):
    off = _tf(city, _od_cfg())
    assert off.mode == "od"
    assert off.signal_cycle_of == {} and off.n_signals_unified == 0
    for node, r in off.signal_red.items():
        assert r == 0.35 + 0.20 * T._hash_frac(node)   # 従来式そのまま
    # 遅延式は conf の全交差点共通の周期を使う(空 dict なので分岐に入らない)
    node = sorted(off.signal_red)[0]
    r = off.signal_red[node]
    assert off._signal_delay_m(node, 4000.0) == \
        4000.0 * (r * r * off.signal_cycle_s * 0.5) / off.step_seconds


@_needs_data
def test_off_and_explicit_false_are_identical(city):
    a = _tf(city, _od_cfg())
    b = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "false"}))
    assert a.signal_red == b.signal_red
    assert _trace(a) == _trace(b)


# =========================================================================== #
# ④⑤ ON の中身
# =========================================================================== #
@_needs_data
def test_on_unified_nodes_take_the_crossing_values(city):
    on = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"}))
    off = _tf(city, _od_cfg())
    assert on.n_signals_unified == len(on.signal_cycle_of) > 0
    assert set(on.signal_cycle_of) <= set(on.signal_red)
    table = T.load_signalized_crossings(CROSSINGS)
    for node in on.signal_cycle_of:                     # 表に居るノード
        cid = T.crossing_id_of_node(node)
        assert cid in table
        assert on.signal_red[node] == EXPECT_R          # 1 − 47/140 = 0.664285…
        assert on.signal_cycle_of[node] == CYCLE_S      # その交差点自身の周期
        assert on.signal_red[node] != off.signal_red[node], "hash と同値 = 検収の空回り"
    # ★表に居ないノードは**従来の hash 値のままバイト一致**(欠測を埋めない)
    rest = set(on.signal_red) - set(on.signal_cycle_of)
    assert rest, "検収の空回り(全ノードが表に居る)"
    for node in rest:
        assert on.signal_red[node] == off.signal_red[node]


@_needs_data
def test_on_covers_the_measured_share_of_signal_nodes(city):
    """被覆率を数字で固定する(『全部の信号が実測になった』と誤読させない)。"""
    on = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"}))
    assert on.n_signals_unified == 23        # 実測: 車道側 69 本中 23 本 ≈ 33%
    assert len(on.signal_red) == 69
    assert len(on.signal_nodes) == 69, "signal_nodes は同一化で増減しない"


@_needs_data
def test_on_delay_uses_that_crossings_own_cycle(city):
    """遅延 r²·C/2 の C が**その交差点自身の周期**になる(全交差点共通の定数ではない)。"""
    on = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"}))
    node = sorted(on.signal_cycle_of)[0]
    want = 4000.0 * (EXPECT_R * EXPECT_R * CYCLE_S * 0.5) / on.step_seconds
    assert on._signal_delay_m(node, 4000.0) == want
    assert on.signal_cycle_s == 90.0, "conf の既定(90s)は残っている"
    # 表に居ないノードは従来どおり conf の周期
    other = sorted(set(on.signal_red) - set(on.signal_cycle_of))[0]
    r = on.signal_red[other]
    assert on._signal_delay_m(other, 4000.0) == \
        4000.0 * (r * r * on.signal_cycle_s * 0.5) / on.step_seconds


@_needs_data
def test_on_missing_crossings_file_falls_back_silently(city):
    """交差点表が読めないときは hash のまま(ランを止めない・偽の値を作らない)。"""
    on = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true",
                              "world.traffic.crossings_file": "data/nope.json"}))
    off = _tf(city, _od_cfg())
    assert on.signal_cycle_of == {} and on.n_signals_unified == 0
    assert on.signal_red == off.signal_red


# =========================================================================== #
# ⑥⑦ 決定論と「本当に挙動が変わる」ことの固定
# =========================================================================== #
@_needs_data
def test_on_is_deterministic(city):
    a = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"}))
    b = _tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"}))
    assert a.signal_red == b.signal_red
    assert a.signal_cycle_of == b.signal_cycle_of
    assert _trace(a) == _trace(b), "同 seed 2 ランで走行が一致しない(決定論の破れ)"


@_needs_data
def test_on_actually_changes_the_driving(city):
    """★これは観測の追加ではなく**挙動変更**である、ということ自体の固定。"""
    off = _trace(_tf(city, _od_cfg()), steps=36)
    on = _trace(_tf(city, _od_cfg(**{"world.traffic.signal_from_crossings": "true"})),
                steps=36)
    assert off != on, "ON で走行が 1 ビットも変わらない(トグルが効いていない)"
    # 発生(スポーン)は乱数側なので同一 = 変わるのは走りだけ
    assert [row[1] for row in off] == [row[1] for row in on]


@_needs_data
def test_ambient_mode_is_untouched_by_the_toggle(city):
    """ambient(既定モード)には 1 バイトも効かない(od 専用のトグル)。"""
    cfg = load_config(overrides=["world.traffic.signal_from_crossings=true"])
    tf = _tf(city, cfg)
    assert tf.mode == "ambient"
    assert tf.signal_cycle_of == {} and not hasattr(tf, "signal_red")
