"""純オーバヘッド除去(挙動不変リファクタ)の同値性テスト。

このスライスは **速くするだけで出力を1バイトも変えない** ことが受入条件なので、
最適化した各点が「元の実装と同じ値を返す」ことを部品単位で機械固定する。

- L1 バイト一致そのものは
  ``tests/test_scenario.py::test_baseline_matches_prechange_golden``
  (変更前ゴールデンとの完全一致)と ``tests/test_determinism.py`` が担保する。
  本ファイルはその **内訳** を押さえ、将来この最適化に手が入ったとき
  「どこが壊れたか」が 1 テストで分かるようにする。
- 対象:
  1. ``lang.sentiment.valence`` の有界メモ化(純関数=同一入力に同一値)。
  2. ``world.routing`` の networkx dispatch 剥がし(素の実装に届いている+経路同一)。
  3. ``world.routing.Router`` の A* ヒューリスティック直引き(``node_xy`` 版と同値)。
  4. ``world.routing.Router.route`` の OD キャッシュ(呼び出し側へ毎回新しいリスト)。
  5. ``cognition.routine._lynch_pool`` の事前計算(素朴な再計算とバイト一致)。
  6. ``cognition.routine._world_prob`` の config 巻き上げ(値同一・キー単位で遅延)。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from society.cognition import routine
from society.config import load_config
from society.lang import sentiment
from society.world.map import CityMap
from society.world.routing import Router, _undispatched

_TEXTS = [
    "今日はとても楽しい一日だった",
    "最悪だ、もう嫌になる",
    "駅前で友人に会った",
    "すごく美味しいご飯を食べた😊",
    "楽しくない",
    "",
    "渋谷は人が多い",
]


# ------------------------------------------------------------------ 1. valence
def test_valence_memoization_matches_uncached_and_is_bounded():
    """メモ化は純粋な高速化 = 素の実装(``__wrapped__``)と全入力で同値。

    lru_cache は上限つき(無制限だと長時間ランで発話文字列が際限なく溜まる)。
    """
    assert hasattr(sentiment.valence, "cache_info"), "valence がメモ化されていない"
    assert sentiment.valence.cache_info().maxsize == 65536, "メモ化の上限が無い/変わった"

    uncached = sentiment.valence.__wrapped__          # デコレータを剥がした素の実装
    for text in _TEXTS:
        assert sentiment.valence(text) == uncached(text), f"メモ化で値が変わった: {text!r}"


def test_valence_cache_hits_and_does_not_leak_between_inputs():
    """同一入力は 2 回目以降ヒットし、異なる入力の値が混ざらない(キー取り違えなし)。"""
    sentiment.valence.cache_clear()
    first = [sentiment.valence(t) for t in _TEXTS]
    assert sentiment.valence.cache_info().misses == len(_TEXTS)
    assert sentiment.valence.cache_info().hits == 0

    second = [sentiment.valence(t) for t in _TEXTS]
    assert second == first
    assert sentiment.valence.cache_info().hits == len(_TEXTS)
    # 値が同じでも「別入力に別エントリ」であること(1 エントリに畳まれていない)
    assert sentiment.valence.cache_info().currsize == len(set(_TEXTS))


# ------------------------------------------------------------------ 2/3/4. 経路
@pytest.fixture(scope="module")
def city() -> CityMap:
    cfg = load_config(["run.n_agents=1", "run.n_steps=1", "run.name=_perf_refactor"])
    return CityMap(cfg.world.map)


def _od_pairs(city: CityMap, n: int = 40) -> list[tuple[str, str]]:
    """決定論に選んだ OD ペア(名簿順・ハッシュ不使用)。"""
    nodes = sorted(city.graph.nodes)
    step = max(1, len(nodes) // n)
    picked = nodes[::step][:n]
    return [(picked[i], picked[-1 - i]) for i in range(len(picked) // 2)]


def test_undispatched_reaches_the_bare_networkx_implementation():
    """dispatch ラッパを **最後まで** 剥がして素の実装に届いている。

    ``__wrapped__`` を 1 段しか辿らないと networkx 3.x では dispatcher 本体
    (``_dispatchable``)に着地し、呼ぶたびに backend 検査を通ってしまう。
    """
    from society.world import routing

    bare = routing._ASTAR
    assert callable(bare)
    # dispatcher オブジェクトでも argmap トランポリンでもない = これ以上剥がすものが無い
    assert getattr(bare, "orig_func", None) is None
    assert getattr(bare, "__wrapped__", None) is None
    assert bare.__module__.endswith("shortest_paths.astar")
    # 剥がせない callable を渡しても壊れない(旧 networkx 互換の後退路)
    def plain(x):
        return x
    assert _undispatched(plain) is plain


def test_undispatched_astar_returns_identical_paths(city: CityMap):
    """素の実装 ``_ASTAR`` と公開 API ``nx.astar_path`` は同一経路(tie-break 込み)。"""
    from society.world import routing

    graph = city.graph

    def h(a: str, b: str) -> float:
        ax, ay = city.node_xy(a)
        bx, by = city.node_xy(b)
        return math.hypot(ax - bx, ay - by)

    checked = 0
    for src, dst in _od_pairs(city):
        try:
            want = nx.astar_path(graph, src, dst, heuristic=h, weight="length")
        except nx.NetworkXNoPath:
            continue
        got = routing._ASTAR(graph, src, dst, heuristic=h, weight="length")
        assert got == want
        checked += 1
    assert checked >= 5, "経路が取れる OD が少なすぎて検査になっていない"


def test_router_heuristic_is_bitwise_identical_to_node_xy(city: CityMap):
    """ノード属性 dict の直引きは ``CityMap.node_xy`` と **同じ float** を返す。

    ヒューリスティックの値が 1bit でも違えば A* の展開順が変わりうるので、
    hex 表現まで一致させる(= 経路が変わらないことの根拠)。
    """
    router = Router(city)
    node_attrs = router._node_attrs()
    for src, dst in _od_pairs(city):
        ax, ay = city.node_xy(src)
        bx, by = city.node_xy(dst)
        want = math.hypot(ax - bx, ay - by)
        da, db = node_attrs[src], node_attrs[dst]
        got = math.hypot(da["x"] - db["x"], da["y"] - db["y"])
        assert got.hex() == want.hex()
        # 同一オブジェクトを引いている(コピーではない=更新が反映される)
        assert da["x"] is city.graph.nodes[src]["x"]


def test_router_routes_match_node_xy_reference_implementation(city: CityMap):
    """`Router.route` の経路が、変更前の実装(node_xy 版 h)と完全一致する。"""
    from society.world import routing

    router = Router(city)
    graph = city.graph

    def old_h(a: str, b: str) -> float:                # 変更前のヒューリスティック
        ax, ay = city.node_xy(a)
        bx, by = city.node_xy(b)
        return math.hypot(ax - bx, ay - by)

    checked = 0
    for src, dst in _od_pairs(city):
        if src == dst:
            continue
        try:
            want = nx.astar_path(graph, src, dst, heuristic=old_h, weight="length")
        except nx.NetworkXNoPath:
            continue
        got, mode = router.route(src, dst, "walk")
        assert mode == "walk"
        assert got == want, f"経路が変わった: {src}->{dst}"
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize("mode", ["car", "bicycle"])
def test_mode_subgraph_routes_still_use_city_coordinates(city: CityMap, mode: str):
    """モード別部分グラフは `add_edge` 由来で x/y を持たない。

    ヒューリスティックが誤って部分グラフ側のノード属性を引くと KeyError になるか、
    座標が消えて経路が変わる。city.graph から引いていることを、実際に car/bicycle で
    経路を取り、変更前実装(node_xy 版 h + 同じ部分グラフ)と突き合わせて固定する。
    """
    from society.world import routing

    router = Router(city)
    sub = router._graph(mode)
    # 前提: 部分グラフのノードは属性を持たない(=直引き先を間違えると即座に壊れる)
    some_node = next(iter(sub.nodes))
    assert "x" not in sub.nodes[some_node]

    def old_h(a: str, b: str) -> float:
        ax, ay = city.node_xy(a)
        bx, by = city.node_xy(b)
        return math.hypot(ax - bx, ay - by)

    checked = 0
    for src, dst in _od_pairs(city):
        if src == dst or src not in sub or dst not in sub:
            continue
        got, used = router.route(src, dst, mode)
        try:
            want = routing._ASTAR(sub, src, dst, heuristic=old_h, weight="length")
        except nx.NetworkXNoPath:
            assert used == "walk"              # 到達不能は徒歩へ後退(従来どおり)
            continue
        assert used == mode and got == want
        checked += 1
    assert checked >= 1, f"{mode} で到達可能な OD が 1 つも無く検査になっていない"


def test_route_returns_a_fresh_list_so_callers_cannot_corrupt_the_cache(city: CityMap):
    """OD キャッシュは毎回コピーを返す(scheduler が route を pop で消費するため)。"""
    router = Router(city)
    src, dst = next((a, b) for a, b in _od_pairs(city)
                    if a != b and len(router.route(a, b, "walk")[0]) > 2)

    first, _ = router.route(src, dst, "walk")
    snapshot = list(first)
    first.pop(0)                                   # scheduler と同じ消費の仕方
    first.append("BOGUS-NODE")

    second, _ = router.route(src, dst, "walk")
    assert second == snapshot, "キャッシュが呼び出し側の破壊的操作に汚染された"
    assert second is not first
    third, _ = router.route(src, dst, "walk")
    assert third is not second, "同じリストオブジェクトを使い回している"


# ------------------------------------------------------------------ 5. Lynch
def _lynch_destination_pre(agent, sim, rng):
    """変更前の ``_lynch_destination``(素朴な毎回再計算)。同値性の参照実装。"""
    score = sim.landmark_score
    cands = [d for d in sim.dests if d != agent.node and d in score]
    if not cands:
        return agent.node
    weights = np.array([max(0.0, float(score[d])) for d in cands], dtype=float)
    total = weights.sum()
    if total <= 0.0:
        return cands[int(rng.integers(len(cands)))]
    weights /= total
    return cands[int(rng.choice(len(cands), p=weights))]


def _lynch_sim(n_dests: int = 400, zero_weights: bool = False):
    """dests と landmark_score だけを持つ最小 sim(engine を起動せずに同値を測る)。"""
    dests = [f"n{i:04d}" for i in range(n_dests)]
    # 一部の dest は score を持たない = 候補から落ちる経路も踏む
    score = {}
    for i, d in enumerate(dests):
        if i % 7 == 3:
            continue
        score[d] = 0.0 if zero_weights else (0.0 if i % 11 == 0 else (i % 97) / 13.0)
    # score にしか無いノード(dests に無い)も混ぜて、集合演算の向きを固定する
    score["ghost"] = 5.0
    return SimpleNamespace(dests=dests, landmark_score=score)


@pytest.mark.parametrize("zero_weights", [False, True])
def test_lynch_pool_precompute_matches_naive_recomputation(zero_weights):
    """事前計算版と素朴な再計算版が、全ての現在地・複数シードで同一の行き先を返す。

    ``total <= 0`` の一様抽選路(zero_weights=True)も含めて突き合わせる。
    """
    sim_new = _lynch_sim(zero_weights=zero_weights)
    sim_old = _lynch_sim(zero_weights=zero_weights)
    for i, node in enumerate(sim_new.dests + ["ghost", "not-a-node"]):
        agent = SimpleNamespace(node=node)
        for seed in (0, 1, 7):
            got = routine._lynch_destination(
                agent, sim_new, np.random.default_rng(seed))
            want = _lynch_destination_pre(
                agent, sim_old, np.random.default_rng(seed))
            assert got == want, f"node={node} seed={seed}"


def test_lynch_pool_weights_are_bytewise_identical_to_naive_array():
    """重み配列(と正規化に使う総和)が素朴構築とバイト単位で一致する。

    ``np.delete`` で 1 点抜いた配列と、内包表記から作り直した配列が別物だと
    ``sum`` の丸めが変わって抽選結果がずれうるので、そこを直接固定する。
    """
    sim = _lynch_sim()
    score = sim.landmark_score
    pool, pool_w, index = routine._lynch_pool(sim)

    for node in [sim.dests[0], sim.dests[5], sim.dests[123], "ghost"]:
        naive_cands = [d for d in sim.dests if d != node and d in score]
        naive_w = np.array([max(0.0, float(score[d])) for d in naive_cands],
                           dtype=float)
        hit = index.get(node)
        fast_w = pool_w.copy() if hit is None else np.delete(pool_w, hit)
        fast_cands = pool if hit is None else pool[:hit] + pool[hit + 1:]

        assert fast_cands == naive_cands
        assert fast_w.dtype == naive_w.dtype
        assert fast_w.tobytes() == naive_w.tobytes(), "重み配列がバイト一致しない"
        assert fast_w.sum().hex() == naive_w.sum().hex(), "総和の丸めが違う"


def test_lynch_pool_is_cached_and_rebuilt_when_dests_change():
    """同じ sim では作り直さない(キャッシュが効いている)。長さが変われば作り直す。"""
    sim = _lynch_sim()
    pool_a, w_a, _ = routine._lynch_pool(sim)
    pool_b, w_b, _ = routine._lynch_pool(sim)
    assert pool_a is pool_b and w_a is w_b

    sim.dests = sim.dests[:100]
    pool_c, _, index_c = routine._lynch_pool(sim)
    assert pool_c is not pool_a
    assert set(pool_c) <= set(sim.dests)
    assert all(index_c[d] == i for i, d in enumerate(pool_c))


def test_lynch_destination_does_not_mutate_the_cached_weights():
    """返り値の正規化(``weights /= total``)がキャッシュ側の配列を壊さない。"""
    sim = _lynch_sim()
    _, pool_w, _ = routine._lynch_pool(sim)
    before = pool_w.tobytes()
    for node in (sim.dests[0], "ghost", sim.dests[9]):
        routine._lynch_destination(SimpleNamespace(node=node), sim,
                                   np.random.default_rng(3))
    assert routine._lynch_pool(sim)[1].tobytes() == before


# ------------------------------------------------------------------ 6. config
def test_world_prob_matches_omegaconf_and_resolves_lazily_per_key():
    """巻き上げた確率定数は OmegaConf 直読みと同値。キーは使われた分だけ解決される。"""
    cfg = load_config(["run.n_agents=1", "run.n_steps=1", "run.name=_perf_cfg"])
    sim = SimpleNamespace(cfg=cfg)

    assert routine._world_prob(sim, "meal_prob") == float(cfg.world.meal_prob)
    assert set(sim._routine_world_probs) == {"meal_prob"}, "未参照キーまで解決している"

    assert routine._world_prob(sim, "exit_prob") == float(cfg.world.exit_prob)
    assert routine._world_prob(sim, "building_enter_prob") == \
        float(cfg.world.building_enter_prob)
    assert set(sim._routine_world_probs) == {
        "meal_prob", "exit_prob", "building_enter_prob"}

    # 2 回目以降もキャッシュから同じ値(型も float)
    again = routine._world_prob(sim, "meal_prob")
    assert again == float(cfg.world.meal_prob) and isinstance(again, float)


def test_world_prob_handles_zero_without_re_resolving_forever():
    """値 0.0 でも「未キャッシュ」と誤判定しない(falsy 判定バグの回帰止め)。"""
    cfg = load_config(["run.n_agents=1", "run.n_steps=1", "run.name=_perf_cfg0",
                       "world.meal_prob=0.0"])
    sim = SimpleNamespace(cfg=cfg)
    assert routine._world_prob(sim, "meal_prob") == 0.0
    assert sim._routine_world_probs == {"meal_prob": 0.0}
