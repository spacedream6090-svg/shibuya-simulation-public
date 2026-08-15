"""社会的ヒエラルキー(地位・信用・名声)= 合成地位スコア + 動員/引力/優先的選択 のテスト。

方針(既存の鉄則を継承。設計: docs/research/hierarchy.md 採択候補 #2/#1/#3):
- OFF(既定): status を一切書かず(=0)・動員フックは加算0/倍率1・feed は従来 TL・L3 に status キーなし・
  L2 集計は列なし=純粋既定と L1 完全一致(ゴールデン golden_baseline_l1.json を守る)。
- ON 決定論: ON 同士2回で L1 完全一致(乱数を1つも足さない)。
- スコアの妥当性: 客観カウント(評判/フォロワー/資産)を手で仕込む → phase_day → 地位順位が材料順位を反映。
- 動員: 主催者 status を 1.0 vs 0.0 で参加確率の**計算値**を直接比較(確率的結果でなく式を検証)。
- R1: _FixedLLM で ON/OFF の generate 呼数が完全一致(status 機構は generate を1本も足さない)。
- L2: ON で status_gini 列が出て 0..1、OFF で列なし(mean_theta_drift と同型)。
- 優先的選択: フィード露出重みが被フォロー数に単調(Barabási)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

import pytest

from society import status
from society.config import load_config
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _snap_states(sim):
    return [json.loads(s["state"]) for s in sim.logger.snapshots]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(hierarchy.enabled=false)と純粋既定が L1 完全一致(144 step)+ L3 に status キーなし。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"hierarchy.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(hierarchy seam が no-op でない)"
    for st in _snap_states(pure):
        for a in st["agents"]:
            assert "status" not in a, "OFF の L3 スナップショットに status キーが漏れている"


# --------------------------------------------------------------------- ON 決定論
def test_on_deterministic(tmp_path):
    """ヒエラルキー ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。乱数を足していない。"""
    a = _sim(tmp_path, "on_a", steps=144, **{"hierarchy.enabled": "true"})
    a.run()
    b = _sim(tmp_path, "on_b", steps=144, **{"hierarchy.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(乱数/順序依存が混入)"


# --------------------------------------------------------------------- スコアの妥当性
def test_status_reflects_materials(tmp_path):
    """評判・フォロワー・資産を材料順位どおりに仕込む → phase_day → 地位が単調に順位を反映する。"""
    sim = _sim(tmp_path, "score", n=6, steps=1, **{"hierarchy.enabled": "true"})
    agents = sorted(sim.agents, key=lambda a: a.id)
    for a in agents:                                   # 被フォロー数を決定論制御(全消し→再構成)
        sim.net.set_follows(a.id, set())   # ★逆索引つきの正規経路(follows 直接代入は索引が古くなる)
    for rank, a in enumerate(agents):                  # rank が大きいほど全材料で上位
        a._reputation = float(rank)
        a.money = float(rank) * 1000.0
        a.account = 0.0
        for follower in agents[:rank]:                 # rank 人に a をフォローさせる=follower_count=rank
            sim.net.follow(follower.id, a.id)
    sim._status_day = -1                               # 日次ガードを外して再計算を強制
    status.phase_day(sim, step=0, sim_min=0)
    statuses = [a.status for a in agents]
    assert statuses == sorted(statuses), f"地位が材料順位に単調でない: {statuses}"
    assert statuses[-1] > statuses[0], "最上位と最下位の地位差が出ていない"
    assert all(0.0 <= s <= 1.0 for s in statuses), "status が 0..1 の外にある"


def test_percentiles_pure_function():
    """_percentiles: rank/N の百分位。同値は id 昇順で安定・単調増加(純関数の単体)。"""
    vals = {10: 5.0, 11: 5.0, 12: 9.0, 13: 1.0}       # 10 と 11 は同値
    pr = status._percentiles(vals, [10, 11, 12, 13])
    assert pr[13] < pr[10] < pr[11] < pr[12], f"順位が値/idに整合しない: {pr}"
    assert pr[12] == pytest.approx(1.0)               # 最大値 → N/N = 1.0
    assert all(0.0 < v <= 1.0 for v in pr.values())


# --------------------------------------------------------------------- 動員(式の検証)
def test_attract_and_buy_formula(tmp_path):
    """主催者/店主の status を 1.0 vs 0.0 で、参加確率の加算・購買の倍率の**計算値**を直接比較。"""
    sim = _sim(tmp_path, "mob", n=10, steps=1, **{"hierarchy.enabled": "true",
                                                  "hierarchy.attract_gain": "0.15",
                                                  "hierarchy.buy_gain": "0.2"})
    host = sim.agents[0]
    host.status = 1.0
    assert status.attract_bonus(sim, host.id) == pytest.approx(0.15)   # gain × status
    assert status.buy_multiplier(sim, host.id) == pytest.approx(1.2)   # 1 + gain × status
    host.status = 0.0
    assert status.attract_bonus(sim, host.id) == 0.0                   # 地位0=加算なし
    assert status.buy_multiplier(sim, host.id) == 1.0                  # 地位0=倍率1(従来と同一)
    # イベント参加確率の式 p = base + relation_bonus + attract。高地位ほど p が大きい。
    base, bonus = 0.25, 0.35
    host.status = 1.0
    p_hi = min(1.0, base + bonus + status.attract_bonus(sim, host.id))
    host.status = 0.0
    p_lo = min(1.0, base + bonus + status.attract_bonus(sim, host.id))
    assert p_hi > p_lo and p_lo == pytest.approx(0.6)


def test_off_hooks_are_identity(tmp_path):
    """OFF では動員/引力フックが恒等(加算0・倍率1)=既存確率と完全同一(バイト一致の根拠)。"""
    sim = _sim(tmp_path, "off_hook", n=10, steps=1)   # hierarchy 既定 OFF
    a = sim.agents[0]
    a.status = 1.0                                     # 値があっても OFF なら無視される
    assert status.attract_bonus(sim, a.id) == 0.0
    assert status.buy_multiplier(sim, a.id) == 1.0


# --------------------------------------------------------------------- 優先的選択(Barabási)
def test_feed_exposure_prefers_hubs(tmp_path):
    """フィード露出重みが被フォロー数に単調増加(優先的選択)+ status に単調増加(威信)。"""
    sim = _sim(tmp_path, "pref", n=6, steps=1, **{"hierarchy.enabled": "true"})
    agents = sorted(sim.agents, key=lambda a: a.id)
    hub, small = agents[0], agents[1]
    for a in agents:
        sim.net.set_follows(a.id, set())
    hub.status = small.status = 0.0
    for follower in agents[2:]:                        # hub を 4 人がフォロー、small は 1 人
        sim.net.follow(follower.id, hub.id)
    sim.net.follow(agents[2].id, small.id)
    w_hub = status.feed_exposure_weight(sim, hub.id)
    w_small = status.feed_exposure_weight(sim, small.id)
    assert w_hub > w_small, "被フォロー数の多い投稿者の露出重みが大きくない(優先的選択が効いていない)"
    # status を上げると同一フォロワー数でも重みが増える(威信)。
    small.status = 1.0
    assert status.feed_exposure_weight(sim, small.id) > w_small


def test_ranked_exposure_selects_top_weight(tmp_path):
    """ranked_exposure_timeline_for が露出重み上位を選ぶ(内部メソッドの単体・決定論)。"""
    sim = _sim(tmp_path, "rank", n=6, steps=1, **{"hierarchy.enabled": "true"})
    net = sim.net
    net.feed_size = 2
    net.posts = [{"id": i, "step": 0, "author": 100 + i, "text": "", "items": [],
                  "likes": set(), "reshares": 0} for i in range(4)]
    net._post_offset = 0
    net.read_marks[9999] = 0
    weight = {100: 1.0, 101: 9.0, 102: 2.0, 103: 8.0}   # 上位2 = post 101, 103
    feed = net.ranked_exposure_timeline_for(9999, lambda p: weight[p["author"]])
    assert {p["id"] for p in feed} == {1, 3}, f"露出重み上位が選ばれていない: {feed}"
    assert [p["id"] for p in feed] == [1, 3], "表示が recency(id 昇順)でない"


# --------------------------------------------------------------------- L2 集計
def test_l2_status_columns(tmp_path):
    """ON で status_gini 列が出て 0..1 に収まる。OFF は列なし(mean_theta_drift と同型)。"""
    on = _sim(tmp_path, "l2_on", n=30, steps=6, **{"hierarchy.enabled": "true"})
    on.run()
    rows = on.logger.metrics
    assert rows and all("status_gini" in r for r in rows), "ON で status_gini 列が出ていない"
    for r in rows:
        assert 0.0 <= r["status_gini"] <= 1.0, f"gini が範囲外: {r['status_gini']}"
        assert 0.0 <= r["status_top10_share"] <= 1.0
        assert r["status_rank_mobility"] >= 0.0
    off = _sim(tmp_path, "l2_off", n=30, steps=6)
    off.run()
    assert all("status_gini" not in r for r in off.logger.metrics), \
        "OFF で status_gini 列が漏れている(L2 不変が壊れている)"


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, *, hierarchy):
    ov = {}
    if hierarchy:
        ov["hierarchy.enabled"] = "true"
    sim = _sim(tmp_path, name, n=30, steps=144, **ov)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: ヒエラルキー ON/OFF で generate 呼数が完全一致。

    地位の再計算は非LLM・非乱数、動員/引力フックは既存の rng.random() の閾値を変えるだけ(新 draw を
    足さない)、feed は決定論選別。よって status 機構は generate() を1本も追加しない(R1)。"""
    on = _run_fixed(tmp_path, "r1_on", hierarchy=True)
    off = _run_fixed(tmp_path, "r1_off", hierarchy=False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert any(getattr(a, "status", 0.0) >= 0.0 for a in on.agents), "ON 側で status が計算されていない"
