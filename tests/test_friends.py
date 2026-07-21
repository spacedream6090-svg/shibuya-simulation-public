"""友人グラフ生成(関係性パッケージ完結 第45バッチ S-R2)= homophily+所属+Dunbar のテスト。

方針(既存の鉄則を継承):
- OFF(既定): friend_graph_built が 0 件・relations 台帳に closeness が注入されない・イベント列は
  純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- ON: 居住者に friend 辺が張られ friend_graph_built が n_edges>0 で出る。来街者は対象外。
- 次数分布が層設定と整合(各居住者に close_min 以上の親友 tier3・tier3 ⊆ tier2+ の入れ子)。
- run.seed 非依存: 属性固定(名簿)+ w_same_area=0 で run.seed を変えても注入辺(tier)が不変。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine.simulation import Simulation

_ON = {"friend_graph.enabled": "true", "relations.enabled": "true"}


def _sim(tmp_path, name, n=100, steps=1, roster=None, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    if roster:
        dot += [f"agents.personas_file=data/{roster}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _friend_rels(agent):
    """friend グラフが注入した関係(closeness 付き)を {other_id: tier} で返す。

    顔なじみ経路は closeness を持たない=除外される(注入辺だけを抽出できる)。"""
    return {oid: int(rel.get("tier", 0)) for oid, rel in agent.mem.relations.items()
            if "closeness" in rel}


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。friend_graph_built は出ず closeness も注入されない。"""
    pure = _sim(tmp_path, "pure", n=20, steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", n=20, steps=144, **{"friend_graph.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(friend_graph seam が no-op でない)"
    assert not _kind(pure, "friend_graph_built"), "OFF で friend_graph_built が出ている"
    assert all(not _friend_rels(a) for a in pure.agents), \
        "OFF なのに友人辺(closeness)が注入されている"


# --------------------------------------------------------------------- 生成・注入
def test_friend_graph_builds_and_injects(tmp_path):
    """friend_graph ON: friend_graph_built が n_edges>0 で出て、居住者に tier≥2 の友人辺が注入される。"""
    sim = _sim(tmp_path, "build", n=100, roster="personas_100_civic.json", **_ON)
    ev = _kind(sim, "friend_graph_built")
    assert len(ev) == 1, "friend_graph_built が1件出ていない"
    assert ev[0].payload["n_edges"] > 0, "友人辺が1本も張られていない"
    residents = [a for a in sim.agents if not a.visitor]
    assert any(any(t >= 2 for t in _friend_rels(a).values()) for a in residents), \
        "友人(tier2)以上の辺が1本も無い"
    # 来街者は対象外(注入辺を持たない)。
    for a in sim.agents:
        if a.visitor:
            assert not _friend_rels(a), "来街者に友人辺が注入されている(対象外のはず)"


def test_degree_distribution_matches_layers(tmp_path):
    """次数分布が Dunbar 層設定と整合: 各居住者に close_min 以上の親友(tier3)・tier3 ⊆ tier2+ の
    入れ子・mean_degree(イベント)が実次数の平均と一致する。"""
    sim = _sim(tmp_path, "deg", n=100, roster="personas_100_civic.json", **_ON)
    fc = sim.friendcfg
    residents = [a for a in sim.agents if not a.visitor]
    assert len(residents) > fc["close_min"] + 2, "居住者が少なすぎて層検証にならない"
    degs = []
    for a in residents:
        rels = _friend_rels(a)
        close = [oid for oid, t in rels.items() if t >= 3]
        friend_plus = [oid for oid, t in rels.items() if t >= 2]
        assert len(close) >= fc["close_min"], \
            f"親友(tier3)が close_min 未満: {len(close)} < {fc['close_min']}"
        assert set(close).issubset(set(friend_plus)), "tier3 が tier2+ の部分集合でない(入れ子が壊れている)"
        degs.append(len(rels))
    ev = _kind(sim, "friend_graph_built")[0]
    mean_deg = sum(degs) / len(degs)
    assert abs(mean_deg - ev.payload["mean_degree"]) < 1e-3, \
        f"mean_degree(イベント)が実次数平均と不一致: {ev.payload['mean_degree']} vs {mean_deg}"


# --------------------------------------------------------------------- run.seed 非依存
def test_run_seed_independent(tmp_path):
    """run.seed を変えても友人辺(注入 tier)が不変。属性は名簿で固定・w_same_area=0(home は直接ランで
    rng 割当=run.seed 依存のため近接項を切る)=辺は (id ペア, friend_graph.seed)+固定属性の純関数。"""
    ov = {**_ON, "friend_graph.w_same_area": "0.0"}
    a = _sim(tmp_path, "seed_a", n=100, roster="personas_100_civic.json",
             **{**ov, "run.seed": "1"})
    b = _sim(tmp_path, "seed_b", n=100, roster="personas_100_civic.json",
             **{**ov, "run.seed": "999999"})
    sig_a = {r.id: sorted(_friend_rels(r).items()) for r in a.agents if not r.visitor}
    sig_b = {r.id: sorted(_friend_rels(r).items()) for r in b.agents if not r.visitor}
    assert sig_a and sig_a == sig_b, "run.seed を変えると友人辺が変わる(run.seed 非依存が崩れている)"


# --------------------------------------------------------------------- 決定論
def test_on_deterministic(tmp_path):
    """friend_graph+relations ON 同士 2 回で L1 完全一致(決定論・mock 100 step)。"""
    a = _sim(tmp_path, "det_a", n=20, steps=100, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=20, steps=100, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "friend_graph ON の決定論が崩れている"
