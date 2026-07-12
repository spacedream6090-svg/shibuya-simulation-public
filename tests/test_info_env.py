"""情報環境の非対称(現実ギャップ Wave G6 2026-07-07)= 推薦/バイラル/炎上 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): feed_rank/viral_cascade/misinfo が 0 件・TL は従来の時系列・イベント列は純粋既定と
  L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。最重要。
- 推薦 ON: TL が意見整合(opinion.alignment)で並べ替わり feed_rank(boosted/filtered)が出る。
- バイラル ON: 高フォロワー author の投稿の reach が加重され viral_cascade が出る。
- 炎上 ON: 誤情報判定 → misinfo(post/correction/flame)が出る。任意で炎上→grievance(factors 経由)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 物理不変な機構(viral/misinfo)は FixedLLM で ON==OFF。推薦は「どの投稿を読むか」を変えるため
  FixedLLM で ON!=OFF になりうる=compute_matched 下の k 不変性で担保(career G5 / crowd G4 と同型)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

import numpy as np

from society import opinion
from society.config import load_config
from society.engine.simulation import Simulation
from society.net import infoenv
from society.net.internet import Internet

_ALL_ON = {"info_env.enabled": "true",
           "info_env.recommendation.enabled": "true",
           "info_env.influence.enabled": "true",
           "info_env.misinfo.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。G6 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"info_env.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G6 seam が no-op でない)"
    for k in ("feed_rank", "viral_cascade", "misinfo"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    assert pure.infoenvcfg["enabled"] is False


# --------------------------------------------------------------------- 意見整合(純関数)
def test_alignment_pure_function():
    assert opinion.alignment(1.0, 1.0) == 1.0
    assert opinion.alignment(1.0, -1.0) == 0.0
    assert opinion.alignment(0.0, 0.0) == 1.0
    assert abs(opinion.alignment(0.5, -0.5) - 0.5) < 1e-12
    # クリップ: 範囲外でも [0,1]
    assert opinion.alignment(1.0, -5.0) == 0.0


# --------------------------------------------------------------------- 推薦: ランキング(単体)
def test_ranked_timeline_selects_by_score():
    """ranked_timeline_for が score 上位 feed_size を選び、boosted/filtered を正しく返す。"""
    net = Internet(feed_size=2)
    net.init_follows([0, 1, 2, 3], np.random.default_rng(0), k=0)
    net.post(1, "aligned", [], 0)      # id0(最古・最も整合)
    net.post(2, "neg-a", [], 0)        # id1
    net.post(3, "neg-b", [], 0)        # id2(最新)
    score = {0: 1.0, 1: 0.0, 2: 0.0}   # id0 だけ整合が高い
    feed, boosted, filtered = net.ranked_timeline_for(
        0, score_of=lambda p: score[p["id"]])
    # 時系列窓(last 2)= {1,2}。整合上位 2 = {0, 2}(タイは id 昇順で 2 が残る)。
    assert [p["id"] for p in feed] == [0, 2]        # 表示は recency(id 昇順)
    assert boosted == [0]                            # 整合ゆえ引き上げ
    assert filtered == [1]                           # 不整合ゆえ間引き
    # 候補が feed_size 以下なら選別しない(boosted/filtered 空)
    net2 = Internet(feed_size=6)
    net2.init_follows([0, 1], np.random.default_rng(0), k=0)
    net2.post(1, "x", [], 0)
    f2, b2, fi2 = net2.ranked_timeline_for(0, score_of=lambda p: 1.0)
    assert [p["id"] for p in f2] == [0] and b2 == [] and fi2 == []


# --------------------------------------------------------------------- 推薦: feed_rank
def test_recommendation_reorders_and_logs_feed_rank(tmp_path):
    """推薦 ON: 意見整合の高い(古い)投稿が引き上げられ、不整合が間引かれ、feed_rank が出る。"""
    sim = _sim(tmp_path, "rec", n=6, steps=1,
               **{**_ALL_ON, "info_env.influence.enabled": "false",
                  "info_env.misinfo.enabled": "false", "net.feed_size": "2"})
    viewer = sim.agents[0]
    viewer.opinion = 1.0                             # とても肯定的な意見
    sim.net.post(1, "😊😄👍", [], 0)                 # id0 肯定(整合)・最古
    sim.net.post(2, "😢😡💢", [], 0)                 # id1 否定
    sim.net.post(3, "😢😡💢", [], 0)                 # id2 否定・最新
    feed = infoenv.timeline(sim, viewer, step=3, sim_min=30)
    assert [p["id"] for p in feed] == [0, 2], "整合投稿(id0)が引き上げられていない"
    fr = _kind(sim, "feed_rank")
    assert fr and fr[0].agent_id == viewer.id, "feed_rank が出ていない"
    assert 0 in fr[0].payload["boosted"], "肯定投稿が boosted に無い"
    assert 1 in fr[0].payload["filtered"], "否定投稿が filtered に無い"

    # OFF(推薦無効)なら従来 TL(時系列 last 2 = id1,id2)・feed_rank なし
    sim2 = _sim(tmp_path, "recoff", n=6, steps=1, **{"net.feed_size": "2"})
    v2 = sim2.agents[0]
    v2.opinion = 1.0
    sim2.net.post(1, "😊😄👍", [], 0)
    sim2.net.post(2, "😢😡💢", [], 0)
    sim2.net.post(3, "😢😡💢", [], 0)
    feed2 = infoenv.timeline(sim2, v2, step=3, sim_min=30)
    assert [p["id"] for p in feed2] == [1, 2], "OFF なのに TL が並べ替わっている"
    assert not _kind(sim2, "feed_rank"), "OFF で feed_rank が出ている"


# --------------------------------------------------------------------- バイラル
def test_viral_cascade_weights_high_follower(tmp_path):
    """バイラル ON: 高フォロワー author の投稿の reach が加重され viral_cascade が出る。"""
    sim = _sim(tmp_path, "viral", n=6, steps=1,
               **{"info_env.enabled": "true", "info_env.influence.enabled": "true",
                  "info_env.influence.follower_threshold": "2"})
    author = sim.agents[1].id
    for f in (2, 3, 4):                              # author を明示的に3人がフォロー
        sim.net.follows[f].add(author)
    pid = sim.net.post(author, "拡散して", [], 0)
    sim.net.react(5, pid, "reshare", 0, author_name=sim.agents[1].name)  # RT 生成
    before = sim.net.posts[pid]["reshares"]
    sim._infoenv_watermark = 0
    infoenv.phase(sim, step=1, sim_min=sim.clock.sim_min(1))
    vc = _kind(sim, "viral_cascade")
    assert vc, "viral_cascade が出ていない"
    p = vc[0].payload
    assert p["post_id"] == pid and p["author"] == author
    assert p["reach"] > 2, "reach が閾値を超えて加重されていない"
    assert sim.net.posts[pid]["reshares"] > before, "reshares が加重されていない"
    # 同じ元投稿は1回だけ加重(重複防止)
    n0 = len(_kind(sim, "viral_cascade"))
    sim.net.react(0, pid, "reshare", 1, author_name=sim.agents[1].name)
    infoenv.phase(sim, step=2, sim_min=sim.clock.sim_min(2))
    assert len(_kind(sim, "viral_cascade")) == n0, "同一元投稿が二重に加重されている"


# --------------------------------------------------------------------- 炎上/誤情報
def test_misinfo_events_and_optional_grievance(tmp_path):
    """炎上 ON: 誤情報判定 → misinfo(post/correction/flame)が出る。炎上→grievance(任意)も検証。"""
    sim = _sim(tmp_path, "mis", n=6, steps=1,
               **{"info_env.enabled": "true", "info_env.misinfo.enabled": "true",
                  "info_env.misinfo.rate": "1.0",
                  "info_env.misinfo.correction_prob": "1.0",
                  "info_env.misinfo.flame_grievance": "0.05"})
    author = sim.agents[1].id
    pid = sim.net.post(author, "うわさ話", [], 0)
    sim._infoenv_watermark = 0
    infoenv.phase(sim, step=1, sim_min=sim.clock.sim_min(1))
    kinds = [e.payload["kind"] for e in _kind(sim, "misinfo")
             if e.payload["post_id"] == pid]
    assert "post" in kinds and "correction" in kinds and "flame" in kinds, \
        f"misinfo の3種が揃っていない: {kinds}"
    # 炎上→発信者の grievance(factors 経由。cause=misinfo_flame の state_update が出る)
    su = [e for e in _kind(sim, "state_update")
          if e.payload.get("cause") == "misinfo_flame" and e.agent_id == author]
    assert su, "flame_grievance>0 なのに grievance の state_update が出ていない"

    # flame_grievance=0(既定)なら grievance は不変(観測イベントのみ)
    sim2 = _sim(tmp_path, "mis0", n=6, steps=1,
                **{"info_env.enabled": "true", "info_env.misinfo.enabled": "true",
                   "info_env.misinfo.rate": "1.0"})
    a2 = sim2.agents[1].id
    p2 = sim2.net.post(a2, "うわさ話", [], 0)
    sim2._infoenv_watermark = 0
    infoenv.phase(sim2, step=1, sim_min=sim2.clock.sim_min(1))
    assert _kind(sim2, "misinfo"), "misinfo イベントが出ていない"
    assert not [e for e in _kind(sim2, "state_update")
                if e.payload.get("cause") == "misinfo_flame"], \
        "flame_grievance=0 なのに grievance が動いている"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """情報環境 全 ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_ALL_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_ALL_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    # 実際に新イベントが出ていること(機構が動いている)
    assert _kind(a, "misinfo"), "ON なのに misinfo が出ていない"


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


# 応答固定でも SNS 投稿を実際に生む(誤情報/推薦の入口=投稿を用意する)ため post 応答にする。
_FIXED_POST = json.dumps({"action": "post", "text": "うわさの話"}, ensure_ascii=False)


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n=30, steps=144, **ov)
    sim.llm = _FixedLLM(_FIXED_POST)
    sim.run()
    return sim


def test_r1_fixedllm_viral_misinfo_invariant(tmp_path):
    """物理不変な機構(バイラル/炎上)は FixedLLM で ON==OFF(generate を1本も足さない)。

    viral=拡散カウンタの加重(観測量)/ misinfo=観測イベント+訂正カウンタ(flame_grievance=0 は
    drive 非接続)。どちらも「どの投稿を読むか/heard 語」を変えないため、応答固定 backend で
    generate 呼数が完全一致する。"""
    on = _run_fixed(tmp_path, "vm_on",
                    **{"info_env.enabled": "true",
                       "info_env.influence.enabled": "true",
                       "info_env.misinfo.enabled": "true"})
    off = _run_fixed(tmp_path, "vm_off", **{"info_env.enabled": "false"})
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"viral/misinfo で呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert _kind(on, "misinfo"), "ON 側で misinfo が動いていない"


def _run_k(tmp_path, name, *, writeback):
    """情報環境 全 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ALL_ON, "controls.mode": "compute_matched",
                  "k.writeback": writeback})
    sim.llm = _FixedLLM(_FIXED_POST)
    sim.run()
    return sim


def test_r1_call_count_k_invariant(tmp_path):
    """推薦は「どの投稿を読むか」を変え FixedLLM で ON!=OFF になりうる(SNS=内容/ネットワーク層の必然)。
    だが R1 の本旨=「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。
    情報環境の3機構は k・内面状態(構成概念)を一切読まず暦・config・意見・フォロワー・新 stream のみ
    参照するため、k=free と k=off で generate 呼数が完全一致する(career G5 / crowd G4 と同型)。"""
    free = _run_k(tmp_path, "k_free", writeback="free")
    off = _run_k(tmp_path, "k_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"情報環境の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "misinfo"), "ON なのに misinfo が出ていない(機構が不発)"
