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
from pathlib import Path

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


# --------------------------------------------------------------- β4 初期関係の較正
# 第117 レーンB3。正典: docs/research/initial-relations-improvement.md §0 R2 / §1.3 /
#   §4 機構4 (4-d) / §8.2。何を守るか:
#   (a) 層別 margin を書かなければ従来どおり(単一 margin)= 既定は 1 バイトも動かない
#   (b) 書けば層ごとに注入 closeness が変わり、**tier は昇格しない**(上の閾値に届かない)
#   (c) 減衰 1.0/日 の下で「接触ゼロで何日その層に留まるか」が伸びる(13 日蒸発の是正)
def _closeness_by_tier(sim):
    """**起動直後の**注入辺を {tier: {closeness}} で返す(同 tier は同値のはず)。

    run() 後に呼ぶと交流と減衰で値が動くので、build 直後の sim にだけ使う。"""
    out: dict[int, set] = {}
    for a in sim.agents:
        for rel in a.mem.relations.values():
            if "closeness" in rel:
                out.setdefault(int(rel.get("tier", 0)), set()).add(
                    round(float(rel["closeness"]), 6))
    return out


_LAYERED = {"friend_graph.margin_close": "15.0",
            "friend_graph.margin_friend": "6.5",
            "friend_graph.margin_acq": "2.5"}


def test_layered_margin_absent_is_the_single_margin(tmp_path):
    """層別 margin を書かない(既定 None)なら従来どおり全層 `margin` = L1 バイト一致。"""
    plain = _sim(tmp_path, "mg_plain", n=100, steps=60,
                 roster="personas_100_civic.json", **_ON)
    plain.run()
    explicit = _sim(tmp_path, "mg_null", n=100, steps=60,
                    roster="personas_100_civic.json", **_ON,
                    **{"friend_graph.margin_close": "null",
                       "friend_graph.margin_friend": "null",
                       "friend_graph.margin_acq": "null"})
    explicit.run()
    assert _l1(plain) == _l1(explicit), "層別 margin の未設定が no-op でない"
    fresh = _sim(tmp_path, "mg_fresh", n=100, roster="personas_100_civic.json", **_ON)
    assert _closeness_by_tier(fresh) == {3: {12.5}, 2: {5.5}, 1: {2.5}}, \
        "既定の注入値が『閾値 + margin(0.5)』から動いている"


def test_layered_margin_injects_per_tier_without_promoting(tmp_path):
    """層別 margin が層ごとに効き、**上の層へ昇格しない**(tier_of と整合)。"""
    from society import relations as REL

    sim = _sim(tmp_path, "mg_layered", n=100, roster="personas_100_civic.json",
               **_ON, **_LAYERED)
    got = _closeness_by_tier(sim)
    assert got.get(3) == {27.0} and got.get(2) == {11.5} and got.get(1) == {4.5}
    rc = sim.relationscfg
    for tier, vals in got.items():
        for clo in vals:
            assert REL.tier_of(clo, rc) == tier, \
                f"注入値 {clo} が tier {tier} から外れる(昇格/降格している)"


def test_layered_margin_survives_the_daily_decay(tmp_path):
    """★13 日蒸発の是正: 接触ゼロで層に留まる日数が伸びる(減衰 1.0/日 は変えない)。

    §1.3 の表(現行)= 親友/友人/知人とも **1 日で 1 tier 落ちる**。
    層別 margin 後 = 親友 15 日 / 友人 6 日 / 知人 2 日 は落ちない。
    """
    from society import relations as REL

    rc = _sim(tmp_path, "mg_days", n=20, **_ON).relationscfg
    thr = {3: rc["tier_close"], 2: rc["tier_friend"], 1: rc["tier_acquaintance"]}
    decay = float(rc["decay_per_day"])

    def _days_in_tier(clo: float, tier: int) -> int:
        """接触ゼロで何日その tier に留まるか(1.0/日 の減衰)。"""
        d = 0
        while REL.tier_of(clo - decay * (d + 1), rc) >= tier:
            d += 1
            if d > 400:
                break
        return d

    before = {t: _days_in_tier(thr[t] + 0.5, t) for t in (3, 2, 1)}
    after = {3: _days_in_tier(thr[3] + 15.0, 3),
             2: _days_in_tier(thr[2] + 6.5, 2),
             1: _days_in_tier(thr[1] + 2.5, 1)}
    assert before == {3: 0, 2: 0, 1: 0}, f"前提が崩れた(現行の蒸発): {before}"
    assert after == {3: 15, 2: 6, 1: 2}, f"較正後の滞留日数が想定と違う: {after}"


def test_finals_profile_carries_the_r2_calibration():
    """本選 conf に R2 推奨値が入っている(基底 conf は不変のまま)。"""
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    fin = OmegaConf.load(root / "conf" / "finals_observe.yaml").friend_graph
    assert (float(fin.margin_close), float(fin.margin_friend),
            float(fin.margin_acq)) == (15.0, 6.5, 2.5)
    assert (int(fin.close_min), int(fin.close_max)) == (1, 3)
    assert (int(fin.friend_min), int(fin.friend_max)) == (4, 8)
    assert int(fin.acq_extra) == 24 and float(fin.age_scale) == 9.0
    base = OmegaConf.load(root / "conf" / "config.yaml").friend_graph
    assert float(base.margin) == 0.5 and "margin_close" not in base, \
        "基底 conf が動いている(R1 違反)"
    assert (int(base.close_min), int(base.acq_extra)) == (3, 20)
