"""第142 RAM 根本原因の修正 + ガード(GRP / C3P / NET ガード / S9 / S12 / S15)のテスト。

正典: docs/plans/ram-rootcause-and-fix-plan.md
実装: src/society/net/contact_formation.py(GRP)/ src/society/tools.py(呼び出し側)/
      src/society/conversation.py(C3P)/ src/society/net/internet.py(NET ガード)/
      src/society/observer/relations.py(S9)/ src/society/engine/simulation.py(S12)/
      src/society/engine/scheduler.py(S15)

守るもの(検収基準の順)
  (0) 出荷既定 = 全キーが基底 conf に宣言済み・既定値は**現行と同値**(= 無効)。
      レジストリ宣言(repro_tier / affects_k / fingerprint_risk)も同時に固定する。
  (1) GRP: 既定(all)は現行どおり「メンバー全員と相互 add_contact」。bounded は
      **創設者(+決定論 k 人)だけ**と contacts を結び、follows を 1 本も張らず、
      **乱数を 1 粒も引かず**、contacts_max の押し出しを貫通しない。
  (2) C3P: cap=0 は現行と同一。cap>0 は「その日の相異なる人数」が cap で**飽和**する。
  (3) NET ガード: hard cap 到達で**拒否**(押し出しはしない)+ 拒否カウンタ。
      auto_follow ぶんの follows も連動してスキップ。既定 0 は完全に現行同一。
  (4) S9: `_last` の内部表現を変えても**出力 rows がバイト不変**(旧実装と機械比較)。
      -0.0 / NaN という実数の 2 つの罠も旧実装と同じ行を出す。
  (5) S15: 近傍 K 人選抜の決定論(距離二乗昇順 → id 昇順)・cap=0 は素通し。
  (6) S12: gc_freeze は ON でも世界・L1 を 1 バイトも変えない。
  (7) S7 / S8 の調査結果を conf の事実として固定する(実装ではなく**記録**)。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from society import conversation as CONV
from society import registry as R
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.net import contact_formation as CF
from society.net.internet import Internet
from society.observer import relations as ORL
from society.rng import RngHub
from society.world.perception import build_index

_REPO_FINALS = "conf/finals_observe.yaml"


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n=6, steps=1, seed=42, **ov) -> Simulation:
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _CountingHub:
    """stream 派生を数えるプロキシ(test_contact_formation と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _ExplodingHub:
    """乱数を 1 粒でも引いたら落ちる hub(= 「引かない」ことの機械証明)。"""

    def stream(self, *key):                        # pragma: no cover - 呼ばれたら失敗
        raise AssertionError(f"乱数を引いた: {key}")


class _Mem:
    def __init__(self):
        self.relations: dict[int, dict] = {}


class _Agent:
    def __init__(self, aid):
        self.id = aid
        self.x = float(aid)
        self.y = 0.0
        self.mem = _Mem()


class _StubSim:
    """contact_formation が触る口だけを持つスタブ(test_contact_formation と同型)。"""

    def __init__(self, n=8, hub=None, **cfg_ov):
        self.net = Internet(feed_size=6)
        self.net.init_follows(list(range(n)), np.random.default_rng(7), k=0)
        self.agents = [_Agent(i) for i in range(n)]
        self.agent_by_id = {a.id: a for a in self.agents}
        self.hub = hub if hub is not None else RngHub(7)
        self.netcfg = {"contact_formation": CF.build_cfg({"enabled": True,
                                                          **cfg_ov})}


# --------------------------------------------------------------------------- #
# (0) 出荷既定 = 現行と同値 + レジストリ宣言
# --------------------------------------------------------------------------- #
NEW_KEYS = {
    "net.contact_formation.group_join_mode": "all",
    "net.contact_formation.group_join_k": 0,
    "net.contacts_hard_max": 0,
    "net.follows_hard_max": 0,
    "conversation.c3_distinct_cap": 0,
    "engine.gc_freeze": False,
    "world.attention_hearers_max": 0,
}


def test_shipped_defaults_are_the_current_behaviour():
    """新キーは全部**基底 conf に宣言済み**で、既定値は現行と同値(= 無効)。"""
    cfg = load_config()
    for dotted, want in NEW_KEYS.items():
        got = R._select(cfg, dotted)
        assert got == want, f"{dotted} の出荷既定が {got}(期待 {want})"
    assert CF.build_cfg(None)["group_join_mode"] == "all"
    assert CF.build_cfg(None)["group_join_k"] == 0
    assert CF.build_cfg(None) == CF.DEFAULTS
    assert CONV.build_cfg(None)["c3_distinct_cap"] == 0


def test_new_keys_are_declared_in_the_registry():
    """新トグルはレジストリ宣言必須(第113 の未宣言トグル走査に落ちない)。"""
    for dotted in NEW_KEYS:
        assert dotted in R.BY_ID, f"{dotted} がレジストリ未宣言"
    assert R.undeclared_toggles(load_config()) == [], "未宣言の bool トグルがある"
    # 層3(世界の力学に触る)だけが affects_k=True を名乗る。
    assert R.BY_ID["world.attention_hearers_max"].affects_k is True
    for dotted in NEW_KEYS:
        if dotted != "world.attention_hearers_max":
            assert R.BY_ID[dotted].affects_k is False, f"{dotted} が affects_k を主張"
        assert R.BY_ID[dotted].repro_tier == "strict", f"{dotted} の等級が strict でない"
        assert R.BY_ID[dotted].fingerprint_risk == "none"


def test_build_cfg_falls_back_on_hostile_values():
    """未知の mode / 負の k は現行挙動(all / 0)へ倒す。"""
    got = CF.build_cfg({"group_join_mode": "everyone", "group_join_k": -5})
    assert got["group_join_mode"] == "all" and got["group_join_k"] == 0
    assert CONV.build_cfg({"c3_distinct_cap": -3})["c3_distinct_cap"] == 0
    assert CONV.build_cfg({"c3_distinct_cap": None})["c3_distinct_cap"] == 0


def test_finals_profile_declares_the_layer1_and_layer2_fixes():
    """本選 conf の値を固定する(層3 は**入れていない**ことも固定する)。"""
    fin = OmegaConf.load(_REPO_FINALS)
    assert str(fin.net.contact_formation.group_join_mode) == "bounded"
    assert int(fin.net.contact_formation.group_join_k) == 0
    assert int(fin.net.contacts_hard_max) == 2000
    assert int(fin.net.follows_hard_max) == 4000
    assert int(fin.conversation.c3_distinct_cap) == 2000
    assert bool(fin.engine.gc_freeze) is True
    # 層3(ユーザー判断待ち)は finals に**書かない**。
    assert "attention_hearers_max" not in (fin.world or {}), \
        "S15(層3)が事前登録なしに finals へ入っている"
    assert "relations_max" not in (fin.get("memory", {}) or {}), \
        "S7(層3)が事前登録なしに finals へ入っている"


# --------------------------------------------------------------------------- #
# (1) GRP: グループ加入接続の有界化(主犯根治)
# --------------------------------------------------------------------------- #
def _found_and_join(sim, name="渋谷夜会"):
    """結成 → 条件を満たす 1 人が加入(test_tools と同じ最小シナリオ)。"""
    founder, joiner = sim.agents[0], sim.agents[1]
    sim.tools.apply(sim, founder, {"type": "found_group", "name": name,
                                   "purpose": "ゆるくつながる"}, 0, 0)
    for other in sim.agents[2:]:                   # 全員をメンバーにして「太い」群を作る
        sim.tools.groups[0]["members"].add(other.id)
        sim.tools.member_of[other.id].add(0)
    joiner.adopted.add(name)
    joiner.mem.record_contact(founder.id, founder.name, 0)
    sim.tools.phase(sim, 1, 10)
    return founder, joiner


def test_group_join_default_still_links_every_member(tmp_path):
    """既定(all)は現行どおり **メンバー全員と相互 add_contact + 相互フォロー**。

    ★これはバグの再現テストでもある(主犯の挙動を明文で固定してから塞ぐ)。
    """
    sim = _sim(tmp_path, "grp_all", n=6, **{"tools.group_join_prob": 1.0})
    founder, joiner = _found_and_join(sim)
    members = sim.tools.groups[0]["members"]
    assert joiner.id in members
    others = {m for m in members if m != joiner.id}
    assert sim.net.contacts[joiner.id] == others, "全員と知り合いになっていない"
    assert others <= sim.net.follows[joiner.id], "自動フォローが張られていない"


def test_group_join_bounded_links_only_the_founder(tmp_path):
    """bounded は**創設者だけ**と contacts を結び、follows を 1 本も張らない。"""
    sim = _sim(tmp_path, "grp_bounded", n=6,
               **{"tools.group_join_prob": 1.0,
                  "net.contact_formation.enabled": "true",
                  "net.contact_formation.group_join_mode": "bounded"})
    before_follows = {a.id: set(sim.net.follows.get(a.id, set()))
                      for a in sim.agents}
    founder, joiner = _found_and_join(sim)
    members = sim.tools.groups[0]["members"]
    assert joiner.id in members and len(members) >= 5
    assert sim.net.contacts[joiner.id] == {founder.id}, \
        "創設者以外とも縁を張っている(有界化が効いていない)"
    assert len(sim.net.contacts[joiner.id]) <= 1 + 0, "エッジが (1+k) を超えた"
    for aid, was in before_follows.items():        # follows は 1 本も動かない
        assert sim.net.follows.get(aid, set()) == was, f"{aid} の follows が動いた"
    # 加入の記録(group_join)は従来どおり出る / 新しい L1 は 1 件も足さない。
    joins = [e for e in sim.logger.events if e.kind == "group_join"]
    assert [e for e in joins if e.agent_id == joiner.id]
    assert not [e for e in sim.logger.events if e.kind == CF.KIND_ACQUAINT], \
        "bounded が L1 を増やしている(加入件数ぶん膨らむ)"
    # #13 の優先枠(創設者の投稿の到達保証)は現行のまま維持されている。
    assert founder.id in sim.net.priority.get(joiner.id, set())


def test_group_join_bounded_does_not_change_random_consumption(tmp_path):
    """bounded は乱数の**本数も名前も**変えない(加入抽選 "group" は分岐の前)。"""
    counts = {}
    for name, ov in (("rng_all", {}),
                     ("rng_bounded",
                      {"net.contact_formation.enabled": "true",
                       "net.contact_formation.group_join_mode": "bounded"})):
        sim = _sim(tmp_path, name, n=6,
                   **{"tools.group_join_prob": 1.0, **ov})
        sim.hub = _CountingHub(sim.hub)
        _found_and_join(sim)
        counts[name] = dict(sim.hub.counts)
    assert counts["rng_all"] == counts["rng_bounded"], \
        f"乱数消費が変わった: {counts}"


def test_on_group_join_draws_no_random_at_all():
    """`on_group_join` は乱数を 1 粒も引かない(hub に触れたら落ちる)。"""
    sim = _StubSim(n=8, hub=_ExplodingHub(), group_join_mode="bounded")
    got = CF.on_group_join(sim, joiner_id=5, founder_id=0, members=set(range(8)))
    assert got == [0]


def test_on_group_join_k_picks_the_lowest_ids_deterministically():
    """k>0 の追加選抜は `sorted(members)` の先頭 k 人(自分・創設者は除く)。"""
    sim = _StubSim(n=8, group_join_mode="bounded", group_join_k=2)
    got = CF.on_group_join(sim, joiner_id=5, founder_id=3,
                           members={0, 1, 2, 3, 5, 7})
    assert got == [0, 1, 3], f"決定論の選抜になっていない: {got}"
    assert sim.net.contacts[5] == {0, 1, 3}
    assert len(sim.net.contacts[5]) <= 1 + 2
    for aid in range(8):                           # follows は不触
        assert sim.net.follows[aid] == set()


def test_on_group_join_is_idempotent_and_skips_unwired_agents():
    """既知の相手は張り直さない / SNS 未配線の相手は結ばない(add_contact と同条件)。"""
    sim = _StubSim(n=4, group_join_mode="bounded")
    assert CF.on_group_join(sim, 1, 0, {0, 1}) == [0]
    assert CF.on_group_join(sim, 1, 0, {0, 1}) == [], "冪等でない(2 度目も結んだ)"
    assert CF.on_group_join(sim, 1, 99, {0, 1, 99}) == [], "未配線の相手と結んだ"


def test_on_group_join_respects_the_snc_contact_cap():
    """contacts_max を貫通しない(押し出しは SNC と同じ `_evict_contacts` 経由)。"""
    sim = _StubSim(n=8, group_join_mode="bounded", contacts_max=2)
    for other in (5, 6, 7):                        # 先に 3 本張っておく
        sim.net.add_contact(1, other, auto_follow=False)
    assert CF.on_group_join(sim, 1, 0, {0, 1}) == [0]
    assert len(sim.net.contacts[1]) <= 2, "上限を貫通した"
    assert 0 in sim.net.contacts[1], "いま結んだ相手が押し出された"


def test_group_join_mode_all_is_byte_identical_to_the_pure_default(tmp_path):
    """明示 all(+ SNC OFF)は純粋既定と L1 バイト一致(seam が完全な no-op)。"""
    pure = _sim(tmp_path, "grp_pure", n=15, steps=144)
    pure.run()
    expl = _sim(tmp_path, "grp_expl", n=15, steps=144,
                **{"net.contact_formation.group_join_mode": "all",
                   "net.contact_formation.group_join_k": 0})
    expl.run()
    assert _l1(expl) == _l1(pure), "GRP の seam が既定ランを動かしている"


def test_group_join_bounded_is_inert_while_snc_is_off():
    """bounded は **SNC OFF では効かない**(上限規律に相乗りする設計の明文化)。"""
    off = CF.build_cfg({"enabled": False, "group_join_mode": "bounded"})
    sim = _StubSim(n=4)
    sim.netcfg = {"contact_formation": off}
    assert CF.group_join_bounded(sim) is False
    sim.netcfg = {"contact_formation": CF.build_cfg(
        {"enabled": True, "group_join_mode": "bounded"})}
    assert CF.group_join_bounded(sim) is True


# --------------------------------------------------------------------------- #
# (2) C3P: すれ違い集計の有界化
# --------------------------------------------------------------------------- #
def _c3_scene(tmp_path, name, n=8, phases=1, **ov):
    """n 体を同座標へ集めて `phases` 回フェーズを回す(全員が全員とすれ違う密な場面)。"""
    sim = _sim(tmp_path, name, n=n, steps=phases,
               **{"conversation.enabled": "true", **ov})
    for ag in sim.agents:
        ag.loc = "street"
        ag.building = None
        ag.floor = 0
        ag.sleeping = False
        ag.node = sim.agents[0].node
        ag.x, ag.y = 50.0, 50.0
    for step in range(phases):
        sim.percept_index = build_index(
            sim.agents, float(sim.cfg.world.perception_radius_m))
        CONV.run_phase(sim, step, 600 + step * 10)   # 同一日(day 境界を跨がない)
    return sim


# すれ違いだけを密にする(会話は成立させない)/ 会話を毎 step 別の相手と成立させる の 2 条件。
_C3_PASS_ONLY = {"conversation.c2.meet_prob": 0.0}
_C3_GREET_MANY = {"conversation.c2.meet_prob": 1.0,
                  "conversation.c2.cooldown_steps": 1}


def test_c3_cap_zero_counts_every_distinct_passerby(tmp_path):
    """既定 0 = 無制限(現行と同値): 密な場面では全員ぶん数える。"""
    sim = _c3_scene(tmp_path, "c3_off", n=8, **_C3_PASS_ONLY)
    sizes = [len(a._c3_pass) for a in sim.agents]
    assert max(sizes) >= 4, f"テストの前提(密な場面)が崩れている: {sizes}"


def test_c3_cap_saturates_the_pass_sets(tmp_path):
    """cap>0 は「その日すれ違った相異なる人数」を cap で飽和させる。"""
    off = _c3_scene(tmp_path, "c3_pass_off", n=8, **_C3_PASS_ONLY)
    on = _c3_scene(tmp_path, "c3_pass_on", n=8,
                   **{"conversation.c3_distinct_cap": 2, **_C3_PASS_ONLY})
    assert max(len(a._c3_pass) for a in off.agents) > 2, "前提(飽和が起きる密度)が崩れた"
    for a in on.agents:
        assert len(a._c3_pass) <= 2, "cap を超えて数えている"
    assert max(len(a._c3_pass) for a in on.agents) == 2, "飽和まで届いていない"


def _greet_after_priming(tmp_path, name, cap):
    """既に cap 件たまった状態から**もう 1 回**会話させ、greet 集合の伸びを見る。

    ★同一日のうちに 2 度目のフェーズを回す(`_roll_day` は日が変わらない限り集合を
      リセットしないので、1 回目で据えた `_c2_day` のまま人工的な初期値を仕込める)。
    """
    sim = _c3_scene(tmp_path, name, n=6, phases=1,
                    **{"conversation.c3_distinct_cap": cap,
                       "conversation.c2.meet_prob": 1.0,
                       "conversation.c2.cooldown_steps": 0})
    for ag in sim.agents:
        ag._c3_greet = {900, 901}                  # = cap(2)に到達済みの状態
    sim.percept_index = build_index(
        sim.agents, float(sim.cfg.world.perception_radius_m))
    n_before = len([e for e in sim.logger.events if e.kind == "conversation"])
    CONV.run_phase(sim, 1, 610)
    n_after = len([e for e in sim.logger.events if e.kind == "conversation"])
    assert n_after > n_before, "2 度目の会話が成立していない(前提が崩れた)"
    return sim


def test_c3_cap_saturates_the_greet_sets(tmp_path):
    """会話側(_c3_greet)も同じ飽和規則(片方だけ直し忘れていないことの証明)。"""
    on = _greet_after_priming(tmp_path, "c3_greet_on", cap=2)
    for a in on.agents:
        assert a._c3_greet == {900, 901}, "cap 到達後も greet を数えている"
    off = _greet_after_priming(tmp_path, "c3_greet_off", cap=0)
    assert any(a._c3_greet != {900, 901} for a in off.agents), \
        "cap=0 なのに数えていない(テストが何も証明していない)"


def test_c3_cap_does_not_change_the_conversations(tmp_path):
    """飽和しても会話の成立・L1 は 1 件も変わらない(数え方だけの話)。"""
    off = _c3_scene(tmp_path, "c3_l1_off", n=8, phases=6, **_C3_GREET_MANY)
    on = _c3_scene(tmp_path, "c3_l1_on", n=8, phases=6,
                   **{"conversation.c3_distinct_cap": 1, **_C3_GREET_MANY})
    assert _l1(on) == _l1(off), "C3P が会話そのものを変えている"


def test_c3_cap_zero_matches_the_pure_default_run(tmp_path):
    """明示 0 は純粋既定と L1 バイト一致(24step の実ラン)。"""
    pure = _sim(tmp_path, "c3_pure", n=15, steps=24,
                **{"conversation.enabled": "true"})
    pure.run()
    expl = _sim(tmp_path, "c3_zero", n=15, steps=24,
                **{"conversation.enabled": "true",
                   "conversation.c3_distinct_cap": 0})
    expl.run()
    assert _l1(expl) == _l1(pure)


# --------------------------------------------------------------------------- #
# (3) NET ガード: Internet 水準の安全上限
# --------------------------------------------------------------------------- #
def _wired(n=6, **kw) -> Internet:
    net = Internet(feed_size=6, **kw)
    net.init_follows(list(range(n)), np.random.default_rng(3), k=0)
    return net


def test_net_guard_default_zero_is_the_current_behaviour():
    """既定 0 では 1 件も拒否せず、拒否カウンタも動かない。"""
    net = _wired(6)
    for other in range(1, 6):
        net.add_contact(0, other)
    assert net.contacts[0] == {1, 2, 3, 4, 5}
    assert net.follows[0] == {1, 2, 3, 4, 5}       # auto_follow は従来どおり
    assert net.n_contact_rejects == 0 and net.n_follow_rejects == 0


def test_net_guard_rejects_per_side_and_counts():
    """cap 到達側の挿入**だけ**を拒否する(押し出しはしない)。"""
    net = _wired(6, contacts_hard_max=2)
    net.add_contact(0, 1, auto_follow=False)
    net.add_contact(0, 2, auto_follow=False)
    assert net.contacts[0] == {1, 2} and net.n_contact_rejects == 0
    net.add_contact(0, 3, auto_follow=False)       # 0 は満杯 / 3 は空き
    assert net.contacts[0] == {1, 2}, "既存の縁が押し出された(拒否のみのはず)"
    assert net.contacts[3] == {0}, "空いている側にも入っていない"
    assert net.n_contact_rejects == 1
    net.add_contact(0, 1, auto_follow=False)       # 既知の再投入は拒否ではない
    assert net.n_contact_rejects == 1


def test_net_guard_skips_the_auto_follow_when_the_contact_is_rejected():
    """contacts を拒否した側の自動フォローも張らない(縁が無いのに購読だけ残さない)。"""
    net = _wired(6, contacts_hard_max=1)
    net.add_contact(0, 1)
    assert net.contacts[0] == {1} and net.follows[0] == {1}
    net.add_contact(0, 2)
    assert net.contacts[0] == {1}, "拒否されていない"
    assert 2 not in net.follows[0], "縁が無いのにフォローだけ張った"
    assert net.n_contact_rejects == 1 and net.n_follow_rejects == 1


def test_net_guard_caps_follows_and_keeps_the_reverse_index_consistent():
    """follows 側も拒否のみ。逆索引 `_followers` は旧全走査と一致し続ける。"""
    net = _wired(8, follows_hard_max=2)
    net.follow(0, 1)
    net.follow(0, 2)
    net.follow(0, 3)                               # 拒否
    assert net.follows[0] == {1, 2} and net.n_follow_rejects == 1
    net.follow(0, 1)                               # 既知 = 冪等(拒否ではない)
    assert net.n_follow_rejects == 1
    for author in range(8):
        assert net.follower_count(author) == net._follower_count_scan(author)


def test_net_guard_is_wired_from_conf(tmp_path):
    """conf の値が Internet まで届く(配線の証明)。"""
    sim = _sim(tmp_path, "netguard", n=4,
               **{"net.contacts_hard_max": 11, "net.follows_hard_max": 13})
    assert sim.net.contacts_hard_max == 11 and sim.net.follows_hard_max == 13
    plain = _sim(tmp_path, "netguard_off", n=4)
    assert plain.net.contacts_hard_max == 0 and plain.net.follows_hard_max == 0


def test_net_guard_survives_old_checkpoints():
    """旧 checkpoint 由来(属性を持たない)の Internet でも落ちない=クラス既定へ後退。"""
    net = _wired(4)
    for attr in ("contacts_hard_max", "follows_hard_max",
                 "n_contact_rejects", "n_follow_rejects"):
        del net.__dict__[attr]                     # 旧 pickle の __dict__ を再現
    net.add_contact(0, 1)
    net.follow(0, 2)
    assert net.contacts[0] == {1} and {1, 2} <= net.follows[0]


def test_net_guard_default_run_is_byte_identical(tmp_path):
    """明示 0 は純粋既定と L1 バイト一致。"""
    pure = _sim(tmp_path, "ng_pure", n=15, steps=144)
    pure.run()
    expl = _sim(tmp_path, "ng_zero", n=15, steps=144,
                **{"net.contacts_hard_max": 0, "net.follows_hard_max": 0})
    expl.run()
    assert _l1(expl) == _l1(pure)


# --------------------------------------------------------------------------- #
# (4) S9: observer G5 `_last` の圧縮(出力 rows はバイト不変)
# --------------------------------------------------------------------------- #
class _LegacyRelationsDaily(ORL.RelationsDaily):
    """S9 **以前**の実装((aid,oid) タプル鍵 + 4 つ組の値)。比較のための参照実装。"""

    def capture(self, sim, day: int) -> int:       # noqa: D102 - 旧コードの写し
        n = 0
        day = int(day)
        for agent in getattr(sim, "agents", ()) or ():
            aid = int(agent.id)
            mem = getattr(agent, "mem", None)
            rels = (getattr(mem, "relations", None) or {}) if mem is not None else {}
            for oid in sorted(rels):
                rel = rels[oid]
                if not isinstance(rel, dict):
                    continue
                val = ORL._value(rel)
                key = (aid, int(oid))
                if self._last.get(key) == val:
                    continue
                self._last[key] = val
                self.rows.append((day, aid, int(oid), val[0], val[1], val[2],
                                  val[3], None, None))
                n += 1
            if not self.passing:
                continue
            npass = len(getattr(agent, "_c3_pass", None) or ())
            ngreet = len(getattr(agent, "_c3_greet", None) or ())
            if npass or ngreet:
                self.rows.append((day, aid, ORL.SELF_ID, None, None, None, None,
                                  int(npass), int(ngreet)))
                n += 1
        return n


class _RelSim:
    def __init__(self, agents):
        self.agents = agents


def _rel_scenario():
    """S9 の等価性を突くための台帳の系列(欠測・-0.0・NaN・無変化・変化を全部含む)。"""
    a0, a1 = _Agent(0), _Agent(1)
    sim = _RelSim([a0, a1])
    days = []
    days.append({0: {1: {"closeness": 1.5, "tier": 2, "count": 3}},
                 1: {0: {"count": 1}}})                        # 初出(tier 欠測あり)
    days.append({0: {1: {"closeness": 1.5, "tier": 2, "count": 3}},
                 1: {0: {"count": 1}}})                        # 完全な無変化
    days.append({0: {1: {"closeness": -0.00001, "tier": 2, "count": 3}},
                 1: {0: {"count": 2, "dormant": True}}})       # round → -0.0
    days.append({0: {1: {"closeness": 0.0, "tier": 2, "count": 3}},
                 1: {0: {"count": 2, "dormant": True}}})       # -0.0 == 0.0(無変化)
    days.append({0: {1: {"closeness": float("nan"), "tier": 2, "count": 3}},
                 1: {0: {"count": 2, "dormant": True}}})       # NaN
    days.append({0: {1: {"closeness": float("nan"), "tier": 2, "count": 3}},
                 1: {0: {"count": 2, "dormant": True}}})       # NaN(毎回「変化」)
    days.append({0: {1: {"closeness": 4.25, "tier": 3, "count": 4}},
                 1: {0: {"count": 3}}})                        # 復帰
    return sim, (a0, a1), days


def _rows_with(cls, tmp_path, name):
    sim, agents, days = _rel_scenario()
    sc = cls(tmp_path / name)
    for day, state in enumerate(days):
        for agent in agents:
            agent.mem.relations = {k: dict(v) for k, v in state[agent.id].items()}
        sc.capture(sim, day)
    return sc.rows


def _norm(rows):
    """NaN を含む行も比較できるように正規化する(NaN は同じ札へ潰す)。"""
    out = []
    for r in rows:
        out.append(tuple("nan" if isinstance(x, float) and math.isnan(x) else x
                         for x in r))
    return out


def test_s9_output_rows_are_byte_identical_to_the_legacy_memo(tmp_path):
    """★S9 の要件: 内部表現を変えても**出力 rows が 1 行も変わらない**。"""
    new = _rows_with(ORL.RelationsDaily, tmp_path, "s9_new")
    old = _rows_with(_LegacyRelationsDaily, tmp_path, "s9_old")
    assert _norm(new) == _norm(old), "S9 が出力を変えた"
    assert new, "テストの前提(行が出る)が崩れている"


def test_s9_memo_is_int_keyed_bytes(tmp_path):
    """`_last` の実体が int → bytes になっている(= 目的のメモリ削減が効いている)。"""
    sim, agents, days = _rel_scenario()
    sc = ORL.RelationsDaily(tmp_path / "s9_repr")
    for agent in agents:
        agent.mem.relations = {k: dict(v) for k, v in days[0][agent.id].items()}
    sc.capture(sim, 0)
    assert sc._last, "_last が空(前提が崩れている)"
    for key, val in sc._last.items():
        assert isinstance(key, int) and isinstance(val, bytes)
        assert len(val) == ORL._PACK.size == 25


def test_s9_pack_treats_negative_zero_as_equal_and_nan_as_always_new():
    """実数の 2 つの罠を明示的に固定する(-0.0 は同値 / NaN は基準に据えない)。"""
    assert ORL._pack((-0.0, 1, 2, False)) == ORL._pack((0.0, 1, 2, False))
    assert ORL._pack((float("nan"), 1, 2, False)) is None
    # 欠測(None)は 0 と混ざらない(欠測を 0 で埋めないという既定則)。
    assert ORL._pack((None, None, 0, False)) != ORL._pack((0.0, 0, 0, False))
    assert ORL._key(3, 7) != ORL._key(7, 3)
    assert ORL._key(3, -1) != ORL._key(3, 1)


def test_s9_reload_last_uses_the_same_packing(tmp_path):
    """part から読み戻した基準でも「無変化なら 1 行も出さない」が成り立つ。"""
    out = tmp_path / "s9_reload"
    sim, agents, days = _rel_scenario()
    for agent in agents:
        agent.mem.relations = {k: dict(v) for k, v in days[0][agent.id].items()}
    first = ORL.RelationsDaily(out)
    first.capture(sim, 0)
    first.flush_segment()
    again = ORL.RelationsDaily(out)                # resume 相当(新プロセス)
    again._resumed = True
    assert again.capture(sim, 1) == 0, "読み戻した基準が効いていない(全対が初出)"


# --------------------------------------------------------------------------- #
# (5) S15: 聴衆の注意上限(層3・既定 OFF)
# --------------------------------------------------------------------------- #
class _AttnSim:
    def __init__(self, cap):
        self.cfg = {"world": {"attention_hearers_max": cap}}


class _Pt:
    def __init__(self, aid, x, y):
        self.id, self.x, self.y = aid, float(x), float(y)


def test_attention_cap_zero_is_a_pass_through():
    """既定 0 では**同じリストをそのまま返す**(コピーもしない = 完全な no-op)。"""
    hearers = [_Pt(1, 1, 0), _Pt(2, 2, 0)]
    got = scheduler._attention_limited(_AttnSim(0), _Pt(0, 0, 0), hearers)
    assert got is hearers


def test_attention_cap_keeps_the_nearest_k_deterministically():
    """距離二乗の昇順 → 同値は id 昇順。返り値は id 昇順(下流の走査順を変えない)。"""
    speaker = _Pt(0, 0.0, 0.0)
    hearers = [_Pt(1, 9.0, 0.0), _Pt(2, 1.0, 0.0), _Pt(3, 0.0, 3.0),
               _Pt(4, -3.0, 0.0), _Pt(5, 2.0, 0.0)]
    # 距離 = 9 / 1 / 3 / 3 / 2 → 近い順に id 2(1)・id 5(2)・**同値 3.0 の 3 と 4 は id 昇順で 3**。
    got = scheduler._attention_limited(_AttnSim(3), speaker, hearers)
    assert [h.id for h in got] == [2, 3, 5], "近傍 K 人の選抜が決定論でない"
    tie = [_Pt(7, 1.0, 0.0), _Pt(3, -1.0, 0.0), _Pt(5, 0.0, 1.0)]
    got2 = scheduler._attention_limited(_AttnSim(2), speaker, tie)
    assert [h.id for h in got2] == [3, 5], "同値の解決が id 昇順でない"


def test_attention_cap_does_not_touch_shorter_audiences():
    """cap 以下の聴衆はそのまま(絞りの分岐に入らない)。"""
    hearers = [_Pt(1, 1, 0), _Pt(2, 2, 0)]
    got = scheduler._attention_limited(_AttnSim(5), _Pt(0, 0, 0), hearers)
    assert got is hearers


def test_attention_cap_zero_run_is_byte_identical(tmp_path):
    """明示 0 は純粋既定と L1 バイト一致。"""
    pure = _sim(tmp_path, "attn_pure", n=15, steps=144)
    pure.run()
    expl = _sim(tmp_path, "attn_zero", n=15, steps=144,
                **{"world.attention_hearers_max": 0})
    expl.run()
    assert _l1(expl) == _l1(pure)


# --------------------------------------------------------------------------- #
# (6) S12: gc.freeze(世界も L1 も 1 バイトも変えない)
# --------------------------------------------------------------------------- #
def test_gc_freeze_changes_nothing_observable(tmp_path):
    """ON でも L1 はバイト一致(GC の話であって世界の話ではない)。"""
    import gc
    off = _sim(tmp_path, "gcf_off", n=15, steps=24)
    off.run()
    try:
        on = _sim(tmp_path, "gcf_on", n=15, steps=24, **{"engine.gc_freeze": "true"})
        assert getattr(on, "_gc_frozen", None) == "init", "初期化後の freeze が走っていない"
        on.run()
        assert _l1(on) == _l1(off), "gc_freeze が出力を変えた"
    finally:
        gc.unfreeze()                              # 他テストへ影響を残さない


def test_gc_freeze_default_is_a_noop(tmp_path):
    """既定 false では freeze を呼ばない(印が付かない)。"""
    sim = _sim(tmp_path, "gcf_default", n=4)
    assert getattr(sim, "_gc_frozen", None) is None


# --------------------------------------------------------------------------- #
# (7) S7 / S8 の調査結果を conf の事実として固定する(実装ではなく記録)
# --------------------------------------------------------------------------- #
def test_s7_relations_max_is_already_a_conf_key_and_unbounded_by_default():
    """S7: `memory.relations_max` は**既に conf 化済み**で、既定 0 = 無制限。

    ★調査の結論(設計書 §2 #2 の「設計上限あり」は**現行 conf では成立していない**):
      基底 conf も本選プロファイルも 0(= LRU 退避なし)なので、在場し続ける個体の
      関係台帳は in-RAM で青天井である(退場する個体だけ `pool.relations_cap` で切られる)。
      値を入れるのは認知容量を絞る**力学変更**(層3)なのでユーザー判断待ち。
    """
    base = load_config()
    assert int(base.memory.relations_max) == 0
    fin = OmegaConf.load(_REPO_FINALS)
    assert "relations_max" not in (fin.get("memory", {}) or {})
    assert int(base.pool.relations_cap) > 0, "退場時の切り取りまで無効になっている"


def test_s8_fact_beliefs_already_has_a_cap():
    """S8: `_fact_beliefs` は **既に `beliefs.max_beliefs_per_agent` で有界**(32 件)。

    ★書き込み経路は `truth_ledger._set_belief` の 1 本だけで、そこが超過分を
      「取得 step の古い順 → キー昇順」で落とす決定論の evict を持っている。
      `truth_ledger.py` は**凍結 SPEC_FILES** なので新しい cap は足さない(足す必要もない)。
    """
    from society import truth_ledger as TL
    from society.observer import metrics_spec as MS
    assert int(load_config().beliefs.max_beliefs_per_agent) == 32
    assert TL.DEFAULTS["max_beliefs_per_agent"] == 32
    assert "src/society/truth_ledger.py" in MS.SPEC_FILES, "凍結の前提が崩れている"


def test_touched_files_are_not_frozen():
    """本レーンが触ったファイルが凍結 SPEC_FILES に 1 つも入っていない。"""
    from society.observer import metrics_spec as MS
    touched = (
        "src/society/tools.py",
        "src/society/conversation.py",
        "src/society/net/contact_formation.py",
        "src/society/net/internet.py",
        "src/society/observer/relations.py",
        "src/society/engine/simulation.py",
        "src/society/engine/scheduler.py",
        "src/society/registry.py",
    )
    for rel in touched:
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"
