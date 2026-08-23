"""第154 A5: `zones.node_in` のゾーン別 frozenset メモ化のテスト。

正典: docs/plans/step-time-audit.md §2 G6 / §3 A5 / §4 の適用順 6・付録 3。

背景: `physics._run_zone:492` は「在街かつ経路持ちの全個体 × 3 ゾーン」に対して
`zones.route_span` を呼び、`route_span` は**経路の全ノード**に `node_in` を掛ける。
`node_in` は `graph.nodes[node]` の辞書引き + `Zone.contains` の ray casting を
**毎回やり直していた**(キャッシュ皆無)のに、地図は静的 = 純関数であり、
「答えの集合を返す関数」(`inside_nodes`)が既に存在していた。

守るもの(検収基準の順)
  (1) ★全ノード突合: メモ化あり `node_in` == メモ化なし `_node_in_uncached` が
      **実地図の全ノード × 全ゾーン**で一致する(合成ゾーン + 実プロファイル両方)。
  (2) `inside_nodes` / `gates_of` / `route_span` の返り値が新旧で完全一致。
  (3) ★キャッシュが pickle / checkpoint / deepcopy に**乗らない**
      (`_node_memo` は dataclass の field ではなく `__getstate__` が除く)。
      1 度も引いていない Zone の instance `__dict__` は空のまま(属性が生えない)。
  (4) 純関数性: 何度引いても同じ答え・`Zone` の等価性/表示/`fields()` が変わらない・
      別グラフを渡したら作り直す・未知ノードは従来どおり `KeyError`。
  (5) 実ラン(mock)の L1 バイト一致(物理ゾーン ON、≤24step)。

検証は mock のみ(実 LLM 禁止・≤24step)。乱数は 1 本も引かない。
"""
from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society.config import load_config
from society.engine.simulation import Simulation
from society.world import zones as Z

REPO = Path(__file__).resolve().parents[1]
SHIBUYA_PROFILE = REPO / "conf" / "zones_shibuya.yaml"

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R],
         [-_ZONE_R, _ZONE_R]]


def _zone_spec(zid="z1", engine="orca", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _cfg(name, n=20, steps=2, zone_specs=None, profile=None, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot, profile=profile)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
    return cfg


def _sim(tmp_path, name, **kw):
    return Simulation(_cfg(name, **kw), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """合成ゾーン 2 枚(原点まわり / 平行移動)を持つ mock sim(モジュール内 1 回)。"""
    d = tmp_path_factory.mktemp("zmemo")
    off = [[x + 120.0, y + 60.0] for x, y in _POLY]
    sim = Simulation(
        _cfg("zmemo_synth",
             zone_specs=[_zone_spec("z1"),
                         _zone_spec("z2", polygon=off)]),
        out_dir=d / "synth")
    return sim


# =========================================================================== #
# (1) ★全ノード突合 — メモ化あり == メモ化なし
# =========================================================================== #
def _sweep_all_nodes(zone, graph):
    """全ノードで新旧を突合し、(内側件数, 総件数) を返す。"""
    n_in = 0
    for node in graph.nodes:
        want = Z._node_in_uncached(zone, graph, node)
        got = Z.node_in(zone, graph, node)
        assert got is want, (zone.id, node, want, got)
        n_in += int(want)
    return n_in, len(graph.nodes)


def test_node_in_matches_the_uncached_predicate_on_every_node(synth):
    graph = synth.city.graph
    total_in = 0
    for zone in synth.physcfg["zones"]:
        n_in, n_all = _sweep_all_nodes(zone, graph)
        assert n_all > 0, "地図にノードが無い(前提が崩れた)"
        total_in += n_in
    assert total_in > 0, "どのゾーンにも内側ノードが無い = 空回りの検算"


def test_node_in_matches_on_the_real_shibuya_profile(tmp_path):
    """★実プロファイル(渋谷 3 空間 + 実地図)の全ノードで一致する。"""
    if not SHIBUYA_PROFILE.exists():          # プロファイル不在の環境ではスキップ
        pytest.skip("conf/zones_shibuya.yaml が無い")
    sim = _sim(tmp_path, "zmemo_shibuya", profile=str(SHIBUYA_PROFILE))
    graph = sim.city.graph
    zones_ = sim.physcfg["zones"]
    assert len(zones_) == 3, [z.id for z in zones_]
    for zone in zones_:
        n_in, n_all = _sweep_all_nodes(zone, graph)
        assert n_in > 0, f"{zone.id} の内側ノードが 0(宣言か地図が変わった)"
        assert n_in < n_all, f"{zone.id} が地図全体を飲み込んでいる"


def test_layers_filter_is_preserved(synth):
    """`zone.layers` 非空の縦レイヤー条件も新旧で一致する(幾何だけに縮退しない)。"""
    graph = synth.city.graph
    base = synth.physcfg["zones"][0]
    layered = dataclasses.replace(base, id="z_layer", layers=(0,))
    n_in, _ = _sweep_all_nodes(layered, graph)
    plain, _ = _sweep_all_nodes(base, graph)
    assert n_in <= plain, "レイヤー条件が緩くなっている"
    weird = dataclasses.replace(base, id="z_layer9", layers=(9,))
    n9, _ = _sweep_all_nodes(weird, graph)
    assert n9 == 0, "存在しないレイヤーでノードが残っている"


# =========================================================================== #
# (2) 下流(inside_nodes / gates_of / route_span)の返り値が変わらない
# =========================================================================== #
def test_inside_nodes_and_gates_are_unchanged(synth):
    graph = synth.city.graph
    for zone in synth.physcfg["zones"]:
        want = tuple(sorted(n for n in graph.nodes
                            if Z._node_in_uncached(zone, graph, n)))
        got = Z.inside_nodes(zone, graph)
        assert got == want, zone.id
        assert list(got) == sorted(got), "id 昇順が崩れている"
        # 2 度目(キャッシュ命中)も同じ
        assert Z.inside_nodes(zone, graph) == want
        gates = Z.gates_of(zone, graph)
        assert list(gates) == sorted(gates)
        for g in gates:
            assert g not in set(got), "ゲートがゾーンの内側にある"


def test_route_span_is_unchanged(synth):
    """`route_span` の (path, rest) が新旧で完全一致する(実地図の経路で総当たり)。"""
    graph = synth.city.graph
    zone = synth.physcfg["zones"][0]
    ins = list(Z.inside_nodes(zone, graph))
    gates = list(Z.gates_of(zone, graph))
    assert ins and gates, "テスト前提が崩れた(ゾーンに道路が無い)"

    def _span_uncached(zone_, graph_, node, route):
        seq = [node] + list(route or ())
        flags = [Z._node_in_uncached(zone_, graph_, n) for n in seq]
        try:
            first_in = flags.index(True)
        except ValueError:
            return None
        for j in range(first_in + 1, len(seq)):
            if not flags[j]:
                return seq[:j + 1], seq[j + 1:]
        return None

    checked = hits = 0
    for gate in gates:
        for inside in ins:
            for tail in ([], [gate], [gate, gates[0]]):
                route = [inside] + tail
                want = _span_uncached(zone, graph, gate, route)
                got = Z.route_span(zone, graph, gate, route)
                assert got == want, (gate, route)
                checked += 1
                hits += int(got is not None)
    assert checked > 0
    assert hits > 0, "所有される経路が 1 本も無い = 空回りの検算"


# =========================================================================== #
# (3) ★キャッシュが pickle / checkpoint / deepcopy に乗らない
# =========================================================================== #
def test_cache_is_not_a_dataclass_field(synth):
    names = {f.name for f in dataclasses.fields(Z.Zone)}
    assert "_node_memo" not in names, "メモ化キャッシュが dataclass の field になっている"
    assert Z.Zone._node_memo is None, "クラス既定が None でない"


def test_untouched_zone_grows_no_instance_attribute(synth):
    """1 度も引いていない Zone には属性が生えない(instance `__dict__` が field だけ)。"""
    zone = dataclasses.replace(synth.physcfg["zones"][0], id="z_fresh")
    assert "_node_memo" not in zone.__dict__
    keys_before = list(zone.__dict__)
    assert zone._node_memo is None                      # クラス属性が読まれるだけ
    assert list(zone.__dict__) == keys_before


def test_pickle_does_not_carry_the_cache(synth):
    """★キャッシュを育てた Zone を pickle しても、バイト列は育てる前と完全一致。"""
    zone = dataclasses.replace(synth.physcfg["zones"][0], id="z_pickle")
    before = pickle.dumps(zone, protocol=pickle.HIGHEST_PROTOCOL)
    Z.inside_nodes(zone, synth.city.graph)              # キャッシュを育てる
    assert "_node_memo" in zone.__dict__, "キャッシュが実際に育っていない(前提が崩れた)"
    after = pickle.dumps(zone, protocol=pickle.HIGHEST_PROTOCOL)
    assert after == before, "pickle バイト列にキャッシュが載っている"
    back = pickle.loads(after)
    assert "_node_memo" not in back.__dict__
    assert back == zone, "pickle 往復で Zone の等価性が壊れている"
    # 復元した Zone でも同じ答えが出る(地図から引き直せる純キャッシュ)
    assert Z.inside_nodes(back, synth.city.graph) == \
        Z.inside_nodes(zone, synth.city.graph)


def test_deepcopy_does_not_carry_the_cache(synth):
    zone = dataclasses.replace(synth.physcfg["zones"][0], id="z_copy")
    Z.inside_nodes(zone, synth.city.graph)
    clone = copy.deepcopy(zone)
    assert "_node_memo" not in clone.__dict__
    assert clone == zone


def test_cache_does_not_pull_the_graph_into_the_pickle(synth):
    """キャッシュはグラフへの強参照を持つが、pickle には 1 バイトも漏れない。"""
    zone = dataclasses.replace(synth.physcfg["zones"][0], id="z_graph")
    Z.inside_nodes(zone, synth.city.graph)
    memo = zone.__dict__["_node_memo"]
    assert memo[0] is synth.city.graph, "graph を同一性で持っていない"
    blob = pickle.dumps(zone, protocol=pickle.HIGHEST_PROTOCOL)
    # 地図まるごとを抱き込んでいれば桁が違う(ゾーン宣言だけなら数 KB)
    assert len(blob) < 64_000, len(blob)


# =========================================================================== #
# (4) 純関数性 / 例外 / 別グラフ
# =========================================================================== #
def test_repeated_queries_are_stable(synth):
    graph = synth.city.graph
    zone = synth.physcfg["zones"][0]
    node = Z.inside_nodes(zone, graph)[0]
    assert all(Z.node_in(zone, graph, node) for _ in range(5))
    outside = next(n for n in graph.nodes if not Z._node_in_uncached(zone, graph, n))
    assert not any(Z.node_in(zone, graph, outside) for _ in range(5))


def test_unknown_node_still_raises_key_error(synth):
    """未知ノードは従来(`graph.nodes[node]`)と同じ `KeyError`(黙って False にしない)。"""
    graph = synth.city.graph
    zone = synth.physcfg["zones"][0]
    with pytest.raises(KeyError):
        Z._node_in_uncached(zone, graph, "n_does_not_exist")
    with pytest.raises(KeyError):
        Z.node_in(zone, graph, "n_does_not_exist")


def test_a_different_graph_rebuilds_the_cache(synth):
    """別のグラフを渡したらメモを作り直す(同一性キー)。"""
    import networkx as nx

    zone = dataclasses.replace(synth.physcfg["zones"][0], id="z_two_graphs")
    g1 = synth.city.graph
    ins1 = Z.inside_nodes(zone, g1)
    g2 = nx.Graph()
    g2.add_node("a", x=0.0, y=0.0, layer=0)             # ゾーン中心 = 内側
    g2.add_node("b", x=10_000.0, y=0.0, layer=0)        # 遠方 = 外側
    assert Z.node_in(zone, g2, "a") is True
    assert Z.node_in(zone, g2, "b") is False
    assert Z.inside_nodes(zone, g2) == ("a",)
    # g1 へ戻ると作り直して元の答えに戻る
    assert Z.inside_nodes(zone, g1) == ins1


def test_zone_identity_and_repr_are_unchanged(synth):
    """メモ化で `__eq__` / `__repr__` / `fields()` が 1 文字も変わらない。"""
    a = dataclasses.replace(synth.physcfg["zones"][0], id="z_eq")
    b = dataclasses.replace(synth.physcfg["zones"][0], id="z_eq")
    assert a == b and repr(a) == repr(b)
    Z.inside_nodes(a, synth.city.graph)                 # 片方だけキャッシュを育てる
    assert a == b, "キャッシュが等価性に漏れている"
    assert repr(a) == repr(b), "キャッシュが repr に漏れている"
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


# =========================================================================== #
# (5) 実ラン(mock)の L1 バイト一致
# =========================================================================== #
def test_zone_run_l1_matches_the_uncached_predicate(tmp_path, monkeypatch):
    """★物理ゾーン ON の mock ラン(≤24step)が、メモ化を殺した B ランと L1 バイト一致。"""
    sim_a = _sim(tmp_path, "zmemo_a", steps=12, n=30, zone_specs=[_zone_spec()])
    sim_a.run()
    l1_a = _l1(sim_a)
    assert l1_a, "L1 が空のランで比べても意味がない"

    calls = {"n": 0}

    def _uncached(zone, graph, node):
        calls["n"] += 1
        return Z._node_in_uncached(zone, graph, node)

    monkeypatch.setattr(Z, "node_in", _uncached)
    sim_b = _sim(tmp_path, "zmemo_b", steps=12, n=30, zone_specs=[_zone_spec()])
    sim_b.run()

    assert calls["n"] > 0, "差し替えた経路が 1 度も通っていない(前提が崩れた)"
    assert _l1(sim_b) == l1_a
