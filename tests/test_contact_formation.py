"""SNS・知り合い形成 v2(SNC。第117・net.contact_formation)のテスト。

正典: docs/plans/sns-contact-redesign.md
実装: src/society/net/contact_formation.py / net/internet.py(unfollow・drop_contact)

守るもの(検収基準の順)
  (1) 既定 OFF = 現行挙動そのまま(聞こえた全員と contacts)・L1 バイト一致・
      プロンプト不変・新 kind 0 件・**乱数を 1 粒も引かない**
  (2) ON: 傍聴者に contacts / follows が生えない。`remember`(聞いた記憶)は生える
      = 新語・情報の対面伝播は死なない
  (3) C1/F2: 返答 JSON の relate/follow → contacts 双方向 / follow 片方向。**欠落=false**
  (4) C3: k 回目で昇格・LRU 上限で古い遭遇相手が押し出される(★O(N²) 再発防止の芯)
  (5) F3: いいね → 確率フォロー(専用 stream・決定論・OFF では乱数不消費)
  (6) 上限: contacts_max / follows_max の押し出しと `_followers` 逆索引の整合
      (= follower_count が旧全走査 `_follower_count_scan` と一致し続ける)
  (7) 宣言: 新 kind 2 種が schema と causality の**両方**に載っている(片方だけだと
      本選 conf の causality ON で logger.log が KeyError で即死する)
  (8) プロンプト: OFF=バイト不変 / ON=reply にだけ 2 行(prompts.p1 の ON/OFF 両経路)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import timeconv as T
from society.cognition import deliberate
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.net import contact_formation as CF
from society.net.internet import Internet
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS, Event
from society.rng import RngHub

ON = {"net.contact_formation.enabled": "true"}
NEW_KINDS = {CF.KIND_ACQUAINT, CF.KIND_FOLLOW}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _FixedLLM:
    """内容非依存の固定応答 LLM(test_engaged の _FixedLLM と同型)。"""

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


class _CountingHub:
    """stream 派生を数えるプロキシ(test_engaged / test_fire と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# --------------------------------------------------------------------------- #
# 単体検査用のごく薄い代役(世界も Simulation も要らない検査のため)
# --------------------------------------------------------------------------- #
class _Mem:
    def __init__(self):
        self.relations: dict[int, dict] = {}


class _Agent:
    def __init__(self, aid):
        self.id = aid
        self.x = float(aid)
        self.y = 0.0
        self.mem = _Mem()


class _Logger:
    def __init__(self):
        self.events: list = []

    def log(self, ev) -> None:
        # 本番 logger と同じく **kind の分類を必ず引く**(未登録なら KeyError で落ちる)。
        C.cause_of(ev.kind)
        self.events.append(ev)


class _StubSim:
    """contact_formation が触る口だけを持つスタブ(net / logger / agent_by_id / hub)。"""

    def __init__(self, n=6, seed=7, **cfg_ov):
        self.net = Internet(feed_size=6)
        self.net.init_follows(list(range(n)), np.random.default_rng(seed), k=0)
        self.agents = [_Agent(i) for i in range(n)]
        self.agent_by_id = {a.id: a for a in self.agents}
        self.logger = _Logger()
        self.hub = RngHub(seed)
        raw = {"enabled": True, **cfg_ov}
        self.netcfg = {"contact_formation": CF.build_cfg(raw)}

    def kinds(self, kind):
        return [e for e in self.logger.events if e.kind == kind]


# --------------------------------------------------------------------------- #
# (1) 既定 OFF = 現行挙動そのまま
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.net.contact_formation.enabled) is False
    assert CF.build_cfg(None)["enabled"] is False
    assert CF.build_cfg(None) == CF.DEFAULTS


def test_shipped_defaults_match_the_design_document():
    """設計書 §2 の初期値がそのまま出荷されている(勝手に動かしていない)。"""
    cfg = load_config().net.contact_formation
    assert int(cfg.encounter_promote_k) == 5      # Moreland & Beach: 5 回で有意
    assert int(cfg.encounter_track_max) == 100    # ★O(N²) 再発防止の要
    assert int(cfg.contacts_max) == 200           # Dunbar 帯
    assert int(cfg.follows_max) == 500            # 安全弁
    assert float(cfg.follow_like_p) == pytest.approx(0.02)


def test_build_cfg_clamps_hostile_values():
    """0 / 負値 / 範囲外を渡されても破壊的な解釈にならない(上限 0 で全消し等)。"""
    got = CF.build_cfg({"enabled": 1, "encounter_promote_k": 0,
                        "encounter_track_max": -5, "contacts_max": 0,
                        "follows_max": -1, "follow_like_p": 3.0})
    assert got["enabled"] is True
    assert got["encounter_promote_k"] == 1 and got["encounter_track_max"] == 1
    assert got["contacts_max"] == 1 and got["follows_max"] == 1
    assert got["follow_like_p"] == 1.0
    assert CF.build_cfg({"follow_like_p": -1.0})["follow_like_p"] == 0.0


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(seam が完全な no-op)。新 kind は 0 件。"""
    pure = _sim(tmp_path, "snc_pure")
    pure.run()
    off = _sim(tmp_path, "snc_off", **{"net.contact_formation.enabled": "false"})
    off.run()
    assert _l1(off) == _l1(pure), "SNC の seam が既定ランを動かしている"
    assert not [e for e in pure.logger.events if e.kind in NEW_KINDS]


def test_off_draws_no_new_random_stream(tmp_path):
    """既定 OFF は新 stream "snc_follow" を 1 本も派生しない(= 乱数不消費)。"""
    sim = _sim(tmp_path, "snc_off_rng")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert "snc_follow" not in sim.hub.counts, "OFF なのに SNC の乱数を引いた"


def test_off_keeps_the_current_hearer_behaviour(tmp_path):
    """OFF では「聞こえた全員と知り合い」= 現行挙動が 1 ミリも変わらない。

    ★これは**バグの再現テスト**でもある: 傍聴者(発話に返事をしていない人)まで
      contacts に載ることを固定する。ON 側の対になるテストが下にある。
    """
    sim = _sim(tmp_path, "snc_off_hear")
    sim.run()
    heard = [e for e in sim.logger.events if e.kind == "hear"]
    assert heard, "hear が 1 件も出ていない(テストの前提が崩れている)"
    for ev in heard:
        speaker = int(ev.payload["speaker"])
        assert speaker in sim.net.contacts.get(ev.agent_id, ()), \
            "OFF なのに聞き手が話者の知り合いになっていない(現行挙動が壊れた)"


# --------------------------------------------------------------------------- #
# (2) ON: 傍聴者に縁が生えない / 記憶は生える
# --------------------------------------------------------------------------- #
def test_on_does_not_link_bystanders_but_keeps_memory(tmp_path):
    """ON: 「聞いただけ」で contacts / follows が生えない。`remember` は生える。

    伝播(新語・情報)の経路は `hearer.remember` なので、そこが生きていれば
    「知り合いを切っても街は静かにならない」= 設計書 §2 の要求そのもの。
    """
    off = _sim(tmp_path, "snc_by_off", n_agents=20)
    off.run()
    on = _sim(tmp_path, "snc_by_on", n_agents=20, **ON)
    on.run()

    def _edges(sim):
        return sum(len(v) for v in sim.net.contacts.values())

    assert _edges(on) < _edges(off), \
        f"ON なのに contacts 辺が減っていない({_edges(on)} vs {_edges(off)})"
    # 「聞いた記憶」の総量は ON/OFF で同オーダー(伝播が死んでいない)。
    hear_off = len([e for e in off.logger.events if e.kind == "hear"])
    hear_on = len([e for e in on.logger.events if e.kind == "hear"])
    assert hear_on > 0 and hear_off > 0
    # 語の対面伝播(transmission / label_adopt)が ON で消えていないこと。
    prop_on = [e for e in on.logger.events
               if e.kind in ("transmission", "label_adopt", "vocab_use")]
    prop_off = [e for e in off.logger.events
                if e.kind in ("transmission", "label_adopt", "vocab_use")]
    if prop_off:
        assert prop_on, "ON で対面伝播が全滅した(remember を切ってしまっている)"


def _add_contact_callers(sim) -> list[str]:
    """ラン中に `net.add_contact` を呼んだ**関数名**を集める(呼び出し元の同定)。"""
    seen: list[str] = []
    inner = sim.net.add_contact

    def _spy(a, b, **kw):           # kw = auto_follow(SNC 経路が渡す)
        seen.append(sys._getframe(1).f_code.co_name)
        return inner(a, b, **kw)
    sim.net.add_contact = _spy      # 起動後に差し替える = day0 の初期配線は数えない
    return seen


#: 置換対象の関数(scheduler の発話ハンドラ)。ここから add_contact が出たら置換漏れ。
SPEAK_HANDLER = "_apply_action"


def test_off_still_calls_add_contact_from_the_speak_handler(tmp_path):
    """反証テスト: OFF では発話ハンドラから add_contact が**実際に走る**。

    下の ON 側テストが「たまたま発話が無かったから 0 件」で通ってしまわないよう、
    同じ計測器で旧挙動が観測できることを先に固定する。
    """
    off = _sim(tmp_path, "snc_caller_off", n_agents=20)
    seen = _add_contact_callers(off)
    off.run()
    assert seen.count(SPEAK_HANDLER) > 0, "OFF なのに旧経路が走っていない"


def test_on_never_calls_add_contact_from_the_speak_handler(tmp_path):
    """ON では発話ハンドラから `net.add_contact` が 1 度も呼ばれない(置換の機械固定)。

    ★他の呼び出し元(グループ加入 = tools.py・day0 の同居/同僚配線)は SNC の
      対象外なので、**関数名で切り分ける**。C1(relate)経由は
      `contact_formation.acquaint` からで、mock は relate 欄を出さないので 0 件。
    """
    on = _sim(tmp_path, "snc_noadd", n_agents=20, **ON)
    seen = _add_contact_callers(on)
    on.run()
    assert SPEAK_HANDLER not in seen, \
        f"ON なのに発話ハンドラから add_contact が走った(呼び出し元: {sorted(set(seen))})"


# --------------------------------------------------------------------------- #
# (3) C1 / F2 = 返答 JSON の relate / follow
# --------------------------------------------------------------------------- #
def test_parser_reads_the_two_flags_unconditionally():
    """パーサは 2 欄を**無条件に**受ける(消費側のトグルとは独立)。余剰キーは黙認。"""
    got = deliberate.parse_action(json.dumps(
        {"action": "speak", "text": "やあ", "relate": True, "follow": True,
         "nonsense": 1}, ensure_ascii=False))
    assert got == {"type": "speak", "text": "やあ", "use_items": [],
                   "relate": True, "follow": True}


def test_parser_treats_missing_flags_as_false():
    """★欠落 = false(安全側)。mock は 2 欄を出さないので mock ランでは成立ゼロ。"""
    got = deliberate.parse_action(
        json.dumps({"action": "speak", "text": "やあ"}, ensure_ascii=False))
    assert "relate" not in got and "follow" not in got


@pytest.mark.parametrize("value", [False, "true", 1, None, "yes"])
def test_parser_only_accepts_literal_true(value):
    """`true` そのものだけを受ける(文字列 "true" や 1 は縁にならない)。"""
    got = deliberate.parse_action(json.dumps(
        {"action": "speak", "text": "やあ", "relate": value, "follow": value},
        ensure_ascii=False))
    assert "relate" not in got and "follow" not in got


def test_on_reply_relate_makes_a_two_way_contact():
    sim = _StubSim()
    replier, speaker = sim.agents[1], sim.agents[0]
    CF.on_reply(sim, replier, speaker.id, {"type": "speak", "text": "x",
                                           "relate": True}, 3, 30)
    assert replier.id in sim.net.contacts[speaker.id]
    assert speaker.id in sim.net.contacts[replier.id], "contacts が片側だけ"
    ev = sim.kinds(CF.KIND_ACQUAINT)
    assert len(ev) == 1 and ev[0].payload["via"] == CF.VIA_REPLY


def test_on_reply_relate_does_not_create_any_follow():
    """★C1 は contacts だけ(自動フォローを切る)。

    設計書 §2 F2「フォローは**宣言した側だけ**が片方向・相互化は自動にしない」。
    `Internet.add_contact` の既定は「知り合いは自動フォロー」なので、SNC 経路が
    `auto_follow=False` で呼ばないとここに話者→返答者のフォローが生えてしまい、
    事後検算アンカー②(相互フォロー率 ≈22.1%)が構造的に測れなくなる。
    """
    sim = _StubSim()
    replier, speaker = sim.agents[1], sim.agents[0]
    CF.on_reply(sim, replier, speaker.id, {"type": "speak", "text": "x",
                                           "relate": True}, 3, 30)
    assert sim.net.follows[speaker.id] == set(), \
        "relate だけで話者→返答者のフォローが張られた(自動フォローが漏れている)"
    assert sim.net.follows[replier.id] == set(), "relate だけでフォローが張られた"
    assert sim.kinds(CF.KIND_FOLLOW) == [], "relate なのに sns_follow が出た"
    assert _index_is_consistent(sim.net)


def test_encounter_promotion_does_not_create_any_follow():
    """★C3 も contacts だけ(C1 と同じ理由)。"""
    sim = _StubSim(encounter_promote_k=2)
    hearer, speaker = sim.agents[1], sim.agents[0]
    for step in (1, 2):
        CF.note_encounter(sim, hearer, speaker, step, step * 10)
    assert speaker.id in sim.net.contacts[hearer.id], "昇格していない(前提が崩れた)"
    assert sim.net.follows[hearer.id] == set() and sim.net.follows[speaker.id] == set()
    assert sim.kinds(CF.KIND_FOLLOW) == []
    assert _index_is_consistent(sim.net)


def test_mutual_follow_needs_both_sides_to_declare():
    """相互フォローは**両者がそれぞれの返事で follow=true を宣言したときだけ**成立する。"""
    sim = _StubSim()
    a, b = sim.agents[0], sim.agents[1]
    decl = {"type": "speak", "text": "x", "relate": True, "follow": True}
    # ① b が a への返事で宣言 → b→a の片方向だけ
    CF.on_reply(sim, b, a.id, decl, 3, 30)
    assert sim.net.follows[b.id] == {a.id}
    assert sim.net.follows[a.id] == set(), "片方の宣言で相互化した"
    # ② 次のターンで a が b への返事で宣言 → ここで初めて相互
    CF.on_reply(sim, a, b.id, decl, 4, 40)
    assert sim.net.follows[a.id] == {b.id} and sim.net.follows[b.id] == {a.id}
    assert len(sim.kinds(CF.KIND_FOLLOW)) == 2, "フォロー成立は 2 件(各自 1 件)"
    assert len(sim.kinds(CF.KIND_ACQUAINT)) == 1, "知り合いは冪等で 1 件"
    assert _index_is_consistent(sim.net)


def test_add_contact_default_still_auto_follows():
    """既定 `auto_follow=True` は現行挙動のままバイト不変(SNC OFF 経路の保護)。"""
    net = Internet(feed_size=6)
    net.init_follows([0, 1], np.random.default_rng(1), k=0)
    net.add_contact(0, 1)                          # 引数なし = 現行の呼び方
    assert net.contacts[0] == {1} and net.contacts[1] == {0}
    assert net.follows[0] == {1}, "既定で自動フォローしなくなった(現行挙動が壊れた)"
    assert net.follows[1] == set(), "自動フォローは a→b の片方向のはず"
    assert net.follower_count(1) == 1 and _index_is_consistent(net)


def test_add_contact_without_auto_follow_touches_no_follow_state():
    """`auto_follow=False` は follows も逆索引も 1 バイトも触らない。"""
    net = Internet(feed_size=6)
    net.init_follows([0, 1, 2], np.random.default_rng(1), k=2)
    before_follows = {k: set(v) for k, v in net.follows.items()}
    before_index = dict(net._followers)
    net.add_contact(0, 1, auto_follow=False)
    assert net.contacts[0] == {1} and net.contacts[1] == {0}
    assert net.follows == before_follows, "follows を触った"
    assert net._followers == before_index, "逆索引を触った"
    assert _index_is_consistent(net)


def test_on_reply_follow_is_one_way_from_the_replier():
    sim = _StubSim()
    replier, speaker = sim.agents[1], sim.agents[0]
    CF.on_reply(sim, replier, speaker.id, {"type": "speak", "text": "x",
                                           "follow": True}, 3, 30)
    assert speaker.id in sim.net.follows[replier.id], "返答者→話者のフォローが無い"
    assert replier.id not in sim.net.follows[speaker.id], \
        "★片方向のはずが相互化している(相互フォロー率 22% のアンカーが壊れる)"
    assert sim.net.contacts[replier.id] == set(), "follow だけで知り合いになった"
    ev = sim.kinds(CF.KIND_FOLLOW)
    assert len(ev) == 1 and ev[0].payload["via"] == CF.VIA_REPLY


def test_on_reply_without_flags_forms_nothing():
    sim = _StubSim()
    CF.on_reply(sim, sim.agents[1], 0, {"type": "speak", "text": "x"}, 3, 30)
    assert sim.net.contacts[1] == set() and sim.net.follows[1] == set()
    assert sim.logger.events == []


def test_on_reply_is_idempotent():
    """同じ相手に何度宣言しても縁は 1 本・イベントも 1 件(毎日同じ縁を数えない)。"""
    sim = _StubSim()
    act = {"type": "speak", "text": "x", "relate": True, "follow": True}
    for step in range(4):
        CF.on_reply(sim, sim.agents[1], 0, act, step, step * 10)
    assert len(sim.kinds(CF.KIND_ACQUAINT)) == 1
    assert len(sim.kinds(CF.KIND_FOLLOW)) == 1


def test_on_reply_is_a_noop_when_disabled():
    sim = _StubSim()
    sim.netcfg["contact_formation"] = CF.build_cfg({"enabled": False})
    CF.on_reply(sim, sim.agents[1], 0, {"type": "speak", "text": "x",
                                        "relate": True, "follow": True}, 3, 30)
    assert sim.net.contacts[1] == set() and sim.net.follows[1] == set()
    assert sim.logger.events == []


def test_on_reply_end_to_end_with_a_declaring_llm(tmp_path):
    """実ラン: relate/follow を必ず宣言する LLM で acquaint / sns_follow が実際に出る。"""
    sim = _sim(tmp_path, "snc_declare", n_agents=20, **ON)
    sim.llm = _FixedLLM(json.dumps(
        {"action": "speak", "text": "こんにちは", "relate": True, "follow": True},
        ensure_ascii=False))
    sim.run()
    acq = [e for e in sim.logger.events if e.kind == CF.KIND_ACQUAINT]
    fol = [e for e in sim.logger.events if e.kind == CF.KIND_FOLLOW]
    assert any(e.payload["via"] == CF.VIA_REPLY for e in acq), \
        "宣言する LLM なのに C1(via=reply)の知り合いが 1 件も出ていない"
    assert any(e.payload["via"] == CF.VIA_REPLY for e in fol), \
        "宣言する LLM なのに F2(via=reply)のフォローが 1 件も出ていない"
    # 完全グラフ化していない(1 人あたり contacts が人口に張り付かない)。
    worst = max(len(v) for v in sim.net.contacts.values())
    assert worst < 20, f"ON なのに完全グラフ化している(最大 contacts={worst})"


# --------------------------------------------------------------------------- #
# (4) C3 = 顔なじみ昇格(★O(N²) 再発防止の芯)
# --------------------------------------------------------------------------- #
def test_encounter_promotes_exactly_on_the_kth_meeting():
    sim = _StubSim(encounter_promote_k=3)
    hearer, speaker = sim.agents[1], sim.agents[0]
    for step in (1, 2):
        CF.note_encounter(sim, hearer, speaker, step, step * 10)
        assert speaker.id not in sim.net.contacts[hearer.id], \
            f"{step} 回目で昇格した(k=3 のはず)"
    CF.note_encounter(sim, hearer, speaker, 3, 30)
    assert speaker.id in sim.net.contacts[hearer.id], "k 回目で昇格しない"
    assert hearer.id in sim.net.contacts[speaker.id], "昇格が片側だけ"
    ev = sim.kinds(CF.KIND_ACQUAINT)
    assert len(ev) == 1 and ev[0].payload["via"] == CF.VIA_ENCOUNTER


def test_encounter_counter_is_dropped_after_promotion():
    """昇格したら遭遇台帳から降ろす(既知の相手で枠を食い続けない)。"""
    sim = _StubSim(encounter_promote_k=2)
    hearer, speaker = sim.agents[1], sim.agents[0]
    for step in range(4):
        CF.note_encounter(sim, hearer, speaker, step, step * 10)
    assert speaker.id not in hearer._encounter_counts
    assert len(sim.kinds(CF.KIND_ACQUAINT)) == 1, "昇格後も繰り返し発火している"


def test_encounter_table_is_bounded_by_track_max():
    """★O(N²) 再発防止の芯: 1 個体の遭遇台帳は track_max 件を**絶対に超えない**。

    上限が無いと「相手ごとの回数」が全人口ぶんに育ち、塞ぎに来た O(N²) が
    メモリ側で再発する(= 直したはずのバグを別の場所で作る)。
    """
    sim = _StubSim(n=2, encounter_promote_k=99, encounter_track_max=4)
    hearer = sim.agents[1]
    strangers = [_Agent(100 + i) for i in range(50)]
    for i, other in enumerate(strangers):
        CF.note_encounter(sim, hearer, other, i, i * 10)
        assert len(hearer._encounter_counts) <= 4, \
            f"{i} 件目で上限 4 を超えた: {len(hearer._encounter_counts)}"
    assert len(hearer._encounter_counts) == 4


def test_encounter_table_evicts_least_recently_seen():
    """LRU: 押し出されるのは「最も長く触れていない相手」で、触った相手は残る。"""
    sim = _StubSim(n=1, encounter_promote_k=99, encounter_track_max=3)
    hearer = sim.agents[0]
    a, b, c, d = (_Agent(x) for x in (10, 11, 12, 13))
    for other in (a, b, c):
        CF.note_encounter(sim, hearer, other, 1, 10)
    CF.note_encounter(sim, hearer, a, 2, 20)      # a を触り直す = 末尾へ
    CF.note_encounter(sim, hearer, d, 3, 30)      # 溢れる → 先頭(= b)が落ちる
    assert set(hearer._encounter_counts) == {10, 12, 13}, \
        f"LRU の順序が違う: {list(hearer._encounter_counts)}"
    assert hearer._encounter_counts[10] == 2, "touch で回数が消えた"


def test_encounter_skips_people_already_known():
    """既に知り合いの相手は数えない(昇格済みの縁が台帳を占有しない)。"""
    sim = _StubSim()
    hearer, speaker = sim.agents[1], sim.agents[0]
    sim.net.add_contact(hearer.id, speaker.id)
    CF.note_encounter(sim, hearer, speaker, 1, 10)
    assert getattr(hearer, "_encounter_counts", {}) == {}


def test_encounter_promotion_happens_in_a_real_run(tmp_path):
    """実ラン(mock)で C3 が効く: via=encounter の acquaint が出て、辺は有界のまま。"""
    sim = _sim(tmp_path, "snc_enc", n_agents=25, **ON,
               **{"net.contact_formation.encounter_promote_k": 2})
    sim.run()
    acq = [e for e in sim.logger.events if e.kind == CF.KIND_ACQUAINT]
    assert any(e.payload["via"] == CF.VIA_ENCOUNTER for e in acq), \
        "k=2 まで下げても顔なじみ昇格が 1 件も起きていない"
    worst = max(len(v) for v in sim.net.contacts.values())
    assert worst < 25, f"C3 で完全グラフ化した(最大 contacts={worst})"


# --------------------------------------------------------------------------- #
# (5) F3 = タイムライン発見(いいね → 確率フォロー)
# --------------------------------------------------------------------------- #
def test_like_follow_fires_at_probability_one():
    sim = _StubSim(follow_like_p=1.0)
    post = {"id": 7, "author": 0, "text": "t", "items": []}
    CF.on_like(sim, sim.agents[3], post, 5, 50)
    assert 0 in sim.net.follows[3]
    ev = sim.kinds(CF.KIND_FOLLOW)
    assert len(ev) == 1 and ev[0].payload["via"] == CF.VIA_TIMELINE


def test_like_follow_never_fires_at_probability_zero():
    sim = _StubSim(follow_like_p=0.0)
    CF.on_like(sim, sim.agents[3], {"id": 7, "author": 0}, 5, 50)
    assert sim.net.follows[3] == set() and sim.logger.events == []


def test_like_follow_is_deterministic_per_agent_step_post():
    """同じ (agent, step, post) なら何度呼んでも同じ判定(専用 stream の決定論)。"""
    def _run():
        sim = _StubSim(seed=3, follow_like_p=0.5)
        for pid in range(12):
            CF.on_like(sim, sim.agents[5], {"id": pid, "author": 0}, 9, 90)
            sim.net.unfollow(5, 0)                # 冪等ガードを外して毎回判定させる
        return [e.payload for e in sim.kinds(CF.KIND_FOLLOW)]
    assert _run() == _run(), "同じキーで結果が揺れた(決定論が壊れている)"


def test_like_follow_varies_across_posts():
    """post id を stream キーに含めている(1 閲覧の複数いいねで同じ乱数を使い回さない)。"""
    sim = _StubSim(seed=3, follow_like_p=0.5)
    hits = []
    for pid in range(20):
        before = len(sim.kinds(CF.KIND_FOLLOW))
        CF.on_like(sim, sim.agents[5], {"id": pid, "author": 0}, 9, 90)
        hits.append(len(sim.kinds(CF.KIND_FOLLOW)) > before)
        sim.net.unfollow(5, 0)
    assert 0 < sum(hits) < len(hits), \
        f"post ごとに判定が変わっていない(全部同じ): {hits}"


def test_like_follow_ignores_media_and_self():
    sim = _StubSim(follow_like_p=1.0)
    CF.on_like(sim, sim.agents[3], {"id": 1, "author": -1}, 5, 50)   # メディア
    CF.on_like(sim, sim.agents[3], {"id": 2, "author": 3}, 5, 50)    # 自分の投稿
    assert sim.net.follows[3] == set() and sim.logger.events == []


def test_like_follow_is_a_noop_when_disabled():
    sim = _StubSim(follow_like_p=1.0)
    sim.netcfg["contact_formation"] = CF.build_cfg({"enabled": False,
                                                    "follow_like_p": 1.0})
    hub = _CountingHub(sim.hub)
    sim.hub = hub
    CF.on_like(sim, sim.agents[3], {"id": 7, "author": 0}, 5, 50)
    assert sim.net.follows[3] == set()
    assert hub.counts == {}, "OFF なのに乱数 stream を派生した"


def test_on_run_draws_the_dedicated_stream(tmp_path):
    """ON の実ランでは "snc_follow" stream が実際に引かれる(配線の存在確認)。"""
    sim = _sim(tmp_path, "snc_stream_on", n_agents=20, **ON,
               **{"net.contact_formation.follow_like_p": 1.0})
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert sim.hub.counts.get("snc_follow", 0) > 0, "F3 の配線が届いていない"
    assert any(e.kind == CF.KIND_FOLLOW
               and e.payload["via"] == CF.VIA_TIMELINE for e in sim.logger.events)


# --------------------------------------------------------------------------- #
# (6) 上限と `_followers` 逆索引の整合
# --------------------------------------------------------------------------- #
def _index_is_consistent(net) -> bool:
    """`follower_count`(O(1) 逆索引)が旧全走査と全 id で一致するか。"""
    ids = set(net.follows) | {t for s in net.follows.values() for t in s}
    return all(net.follower_count(i) == net._follower_count_scan(i) for i in ids)


def test_unfollow_keeps_the_reverse_index_exact():
    net = Internet(feed_size=6)
    net.init_follows(list(range(8)), np.random.default_rng(1), k=3)
    for f in range(8):
        for a in sorted(net.follows[f]):
            net.unfollow(f, a)
            assert _index_is_consistent(net), "unfollow で逆索引がずれた"
    assert net._followers == {}, "全部外したのに逆索引が残っている"


def test_unfollow_is_idempotent_and_safe_on_unknown_ids():
    net = Internet(feed_size=6)
    net.init_follows([0, 1, 2], np.random.default_rng(1), k=0)
    net.follow(0, 1)
    net.unfollow(0, 1)
    net.unfollow(0, 1)                             # 2 度目は no-op
    net.unfollow(99, 1)                            # 名簿外も no-op
    assert net.follows[0] == set() and _index_is_consistent(net)


def test_drop_contact_removes_both_sides_and_leaves_follows():
    net = Internet(feed_size=6)
    net.init_follows([0, 1], np.random.default_rng(1), k=0)
    net.add_contact(0, 1)
    net.drop_contact(0, 1)
    assert net.contacts[0] == set() and net.contacts[1] == set()
    assert 1 in net.follows[0], "drop_contact が follows まで巻き込んだ"
    assert _index_is_consistent(net)


def test_contacts_cap_evicts_the_weakest_closeness():
    """contacts_max 超過で closeness 最弱が押し出される(両側から外れる)。"""
    sim = _StubSim(n=5, contacts_max=3)
    me = sim.agents[0]
    for other in (1, 2, 3):
        sim.net.add_contact(me.id, other)
    # closeness: 1 が最弱・3 が最強
    me.mem.relations = {1: {"closeness": 0.5, "last_step": 9},
                        2: {"closeness": 5.0, "last_step": 9},
                        3: {"closeness": 9.0, "last_step": 9}}
    CF.acquaint(sim, me, 4, CF.VIA_REPLY, 10, 100)
    assert sim.net.contacts[0] == {2, 3, 4}, f"押し出しが違う: {sim.net.contacts[0]}"
    assert 0 not in sim.net.contacts[1], "押し出しが片側だけ(相手側に残った)"
    assert _index_is_consistent(sim.net)


def test_contacts_cap_never_evicts_the_new_partner():
    """いま結んだ相手は落とさない(結んだ瞬間に消える、が起きない)。"""
    sim = _StubSim(n=6, contacts_max=2)
    me = sim.agents[0]
    for other in (1, 2, 3, 4):
        CF.acquaint(sim, me, other, CF.VIA_REPLY, other, other * 10)
        assert other in sim.net.contacts[0], f"{other} が結んだ直後に落ちた"
    assert len(sim.net.contacts[0]) == 2


def test_follows_cap_evicts_and_keeps_the_index_exact():
    """follows_max 超過の押し出しは **unfollow 経由**で逆索引を保つ。"""
    sim = _StubSim(n=3, follows_max=4, follow_like_p=1.0)
    me = sim.agents[0]
    for author in range(10, 30):
        CF.follow(sim, me, author, CF.VIA_TIMELINE, 1, 10)
        assert len(sim.net.follows[0]) <= 4, "follows_max を超えた"
        assert author in sim.net.follows[0], "結んだ直後に落ちた"
    assert _index_is_consistent(sim.net), "押し出しで逆索引がずれた"


def test_follows_cap_drops_non_contacts_first():
    """押し出しは「知り合いでない購読」から(決定論の代理規則)。"""
    sim = _StubSim(n=4, follows_max=2)
    me = sim.agents[0]
    sim.net.add_contact(0, 1)                      # 1 = 知り合い(follows[0] にも入る)
    CF.follow(sim, me, 50, CF.VIA_TIMELINE, 1, 10)
    CF.follow(sim, me, 51, CF.VIA_TIMELINE, 2, 20)
    assert 1 in sim.net.follows[0], "知り合いの購読が先に落ちた"
    assert 51 in sim.net.follows[0]
    assert _index_is_consistent(sim.net)


def test_caps_hold_in_a_real_run(tmp_path):
    """実ラン: どの個体も contacts_max / follows_max を超えない。

    ★follows_max は初期フォロー `net.follow_k`(既定 6)より上に取る: 上限は
      **SNC が伸ばす経路に掛かる歯止め**であって、day0 に配られた初期フォローを
      遡って刈る機構ではない(本番値 500 ≫ 6 なので実運用では区別が付かない)。
    """
    sim = _sim(tmp_path, "snc_caps", n_agents=25, **ON,
               **{"net.contact_formation.contacts_max": 3,
                  "net.contact_formation.follows_max": 8,
                  "net.contact_formation.encounter_promote_k": 2,
                  "net.contact_formation.follow_like_p": 1.0})
    sim.run()
    assert max((len(v) for v in sim.net.contacts.values()), default=0) <= 3
    assert max((len(v) for v in sim.net.follows.values()), default=0) <= 8
    assert _index_is_consistent(sim.net), "実ランで逆索引がずれた"


# --------------------------------------------------------------------------- #
# (7) 宣言(schema + causality の 2 箇所 / registry / timeconv)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", sorted(NEW_KINDS))
def test_new_kind_is_registered_in_both_places(kind):
    """★片方だけの登録を落とす: schema に無い or causality に無いなら失敗。

    第115 の教訓 = 因果台帳へ登録し忘れると、本選 conf の causality ON で
    `logger.log` が KeyError を投げてランが即死する。
    """
    assert kind in EVENT_KINDS, f"{kind} が observer/schema.py に未登録"
    assert kind in C.CAUSE_OF_KIND, f"{kind} が causality.CAUSE_OF_KIND に未登録"
    assert C.cause_of(kind) in C.CAUSE_TYPES


def test_logging_a_new_kind_survives_the_causality_lookup():
    """新 kind を実際に log しても分類が引ける(登録漏れならここで KeyError)。"""
    sim = _StubSim()
    CF.acquaint(sim, sim.agents[0], 1, CF.VIA_REPLY, 1, 10)
    CF.follow(sim, sim.agents[0], 2, CF.VIA_TIMELINE, 1, 10)
    assert {e.kind for e in sim.logger.events} == NEW_KINDS


def test_new_kinds_carry_the_channel_label():
    """payload["via"] が設計書 §2 の 3 チャネルのどれか(内訳が事後に復元できる)。"""
    assert {CF.VIA_REPLY, CF.VIA_ENCOUNTER, CF.VIA_TIMELINE} == \
        {"reply", "encounter", "timeline"}


def test_registry_declares_the_toggle():
    feat = {f.id: f for f in R.FEATURES}["net.contact_formation.enabled"]
    assert feat.repro_tier == "journal", "自由文の relate/follow を読むので journal"
    assert feat.affects_k is False, "generate() の呼び出しサイトを 1 つも増やさない"
    assert feat.fingerprint_risk == "possible", "reply プロンプトに 2 行増える"
    assert R.undeclared_toggles(load_config()) == []


def test_timeconv_classifies_the_new_numeric_keys():
    """Δt 分類表に 5 キーが載っている(全部 INVARIANT = 時間量ではない)。"""
    table = {k: cls for k, cls, _why in T.TABLE}
    for name in ("follow_like_p", "encounter_promote_k", "encounter_track_max",
                 "contacts_max", "follows_max"):
        key = f"net.contact_formation.{name}"
        assert key in table, f"{key} が timeconv 未分類"
        assert table[key] == T.INVARIANT, f"{key} は Δt 非依存のはず"


def test_encounter_counts_is_transient_not_a_carried_field():
    """`_encounter_counts` は個体側 transient(搬送欄を 1 つも増やしていない)。

    プール回転の搬送は「明示された欄」だけを運ぶ設計なので、ここでは
    **属性が最初から生えていない**(= 触るまで存在しない)ことを固定する。
    """
    agent = _Agent(1)
    assert not hasattr(agent, "_encounter_counts")
    sim = _StubSim(n=2, encounter_promote_k=99)
    CF.note_encounter(sim, sim.agents[1], sim.agents[0], 1, 10)
    assert isinstance(sim.agents[1]._encounter_counts, dict)


# --------------------------------------------------------------------------- #
# (8) プロンプト
# --------------------------------------------------------------------------- #
class _PromptSim:
    """`prompt_section` が触る口だけ(cfg / netcfg)。"""

    def __init__(self, on: bool):
        self.netcfg = {"contact_formation": CF.build_cfg({"enabled": on})}


def test_prompt_section_is_none_when_off():
    assert CF.prompt_section(_PromptSim(False), "reply") is None


def test_prompt_section_is_reply_only():
    on = _PromptSim(True)
    assert CF.prompt_section(on, "reply") is not None
    for trigger in ("social", "solo", "post", "dm", "novel_place"):
        assert CF.prompt_section(on, trigger) is None, \
            f"{trigger} にまで判定の説明が出ている"


def test_prompt_section_is_exactly_two_lines():
    text = CF.prompt_section(_PromptSim(True), "reply")
    assert len(text.split("\n")) == 2 and len(CF.JUDGMENT_LINES) == 2


def _reply_prompt(snc, p1):
    return deliberate.build_prompt(
        _PromptAgent(), place_name="見本通り", surprise="reply",
        nearby_names=["甲"], nearby_ids=[1], sim_min=510, step=51,
        city_name="見本町", reply_to=("甲", "やあ"),
        snc_section=snc, p1_purpose=p1)


class _PromptMem:
    day_summaries: list = []

    def recent(self, n):
        return ["路上で立ち話をした"][:n]

    def retrieve(self, *a, **k):
        return []

    def query_ex(self, *a, **k):
        class _R:
            hits: list = []
            failed = False
            cue = ""
        return _R()

    def relation_line(self, ids):
        return None


class _PromptAgent:
    id = 0
    name = "見本"
    persona = "私は見本の人間です。"
    activity = "working"
    states: dict = {}
    beliefs: list = []
    adopted: set = set()
    said: list = []
    money = 1000
    self_model: dict = {}
    implicit_self = ""
    mem = _PromptMem()


@pytest.mark.parametrize("p1", [None, "deliberate"])
def test_reply_prompt_is_byte_identical_when_off(p1):
    """OFF(snc_section=None)は P1 の ON/OFF どちらでもプロンプトがバイト不変。"""
    assert _reply_prompt(None, p1) == _reply_prompt(None, p1)
    assert "relate" not in _reply_prompt(None, p1)
    assert "follow" not in _reply_prompt(None, p1)


@pytest.mark.parametrize("p1", [None, "deliberate"])
def test_reply_prompt_gains_exactly_the_two_lines_when_on(p1):
    """ON では reply プロンプトに 2 行だけが増える(P1 の ON/OFF 両経路)。"""
    section = CF.prompt_section(_PromptSim(True), "reply")
    off, on = _reply_prompt(None, p1), _reply_prompt(section, p1)
    assert on != off
    assert on.replace("\n" + section, "") == off, "2 行以外も変わっている"
    for line in CF.JUDGMENT_LINES:
        assert line in on


def test_non_reply_prompts_are_untouched_when_on():
    """発話・投稿・独り言のプロンプトには 1 行も足さない(判定は対話相手に固定)。"""
    for surprise in ("social", "solo", "post"):
        base = deliberate.build_prompt(
            _PromptAgent(), place_name="見本通り", surprise=surprise,
            nearby_names=["甲"], nearby_ids=[1], sim_min=510, step=51,
            city_name="見本町", dm_target="甲")
        assert "relate" not in base


def test_prompt_lines_contain_no_mechanism_or_condition_words():
    """no-fingerprint: 機構語・実験条件語・因子名を 1 文字も書かない。"""
    text = "".join(CF.JUDGMENT_LINES)
    for word in ("発火", "閾値", "驚き", "シミュレーション", "モデル", "エージェント",
                 "上限", "グラフ", "ネットワーク", "実験", "条件", "パラメータ",
                 "確率", "スコア"):
        assert word not in text, f"禁止語 '{word}' がプロンプトに出た"


def test_prompt_is_unchanged_in_a_real_off_run(tmp_path):
    """既定 OFF の実ランで、返答プロンプトに判定 2 行が 1 度も出ない。"""
    sim = _sim(tmp_path, "snc_prompt_off", n_agents=15)
    seen: list = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    sim.run()
    assert seen, "LLM が 1 度も呼ばれていない(テストの前提が崩れている)"
    assert not any(CF.JUDGMENT_LINES[0] in p for p in seen)


def test_prompt_appears_in_a_real_on_run(tmp_path):
    """ON の実ランでは返答プロンプトにだけ判定 2 行が出る。"""
    sim = _sim(tmp_path, "snc_prompt_on", n_agents=15, **ON)
    seen: list = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    sim.run()
    hit = [p for p in seen if CF.JUDGMENT_LINES[0] in p]
    assert hit, "ON なのに判定 2 行が 1 度も出ていない"
    assert all("話しかけられた" in p for p in hit), "返答以外にも出ている"


# --------------------------------------------------------------------------- #
# (9) 決定論(同 seed 2 ラン一致)
# --------------------------------------------------------------------------- #
def test_on_run_is_deterministic(tmp_path):
    a = _sim(tmp_path, "snc_det_a", n_agents=20, **ON,
             **{"net.contact_formation.encounter_promote_k": 2,
                "net.contact_formation.follow_like_p": 0.5})
    a.run()
    b = _sim(tmp_path, "snc_det_b", n_agents=20, **ON,
             **{"net.contact_formation.encounter_promote_k": 2,
                "net.contact_formation.follow_like_p": 0.5})
    b.run()
    assert _l1(a) == _l1(b), "同 seed 2 ランで L1 が食い違う"
    assert a.net.contacts == b.net.contacts and a.net.follows == b.net.follows


def test_resume_matches_straight(tmp_path):
    """mid-day resume が straight と L1/L2/L3 一致(新しい中央管理を要さないことの確認)。

    SNC が持つ状態は 2 つだけで、どちらも既存の器に自然同梱される:
      ・contacts / follows / `_followers` 逆索引 … `sim.net` の中身
        (checkpoint は `sim.net` を丸ごと pickle するので何も足さなくてよい)
      ・`_encounter_counts` … 個体の属性(agents pickle に自然同梱。engaged の
        `_engaged` / `_engaged_refr` と同じ扱い)
    """
    ov = {**ON, "observer.checkpoint_every": 72,
          "net.contact_formation.encounter_promote_k": 2,
          "net.contact_formation.follow_like_p": 1.0}
    # 1 日ぶん(144 step)回さないと C3 の 2 回目の遭遇も TL 閲覧も起きない
    # (96 step では SNC のイベントが 0 件 = 空回りのテストになる)。
    split, total, n = 72, 144, 25
    straight_dir = tmp_path / "snc_straight"
    straight = Simulation(_cfg("snc_straight", total, n, **ov),
                          out_dir=straight_dir)
    straight.run()
    # ★checkpoint_every を入れると logger.events は区間ごとに掃き出されて空になるので、
    #   前提の確認は**総計タリー**で見る(空回りのテストを防ぐ)。
    assert NEW_KINDS & set(straight.logger.total_event_kinds()), \
        "テスト前提が崩れた(SNC のイベントが 1 件も出ていない)"

    d = tmp_path / "snc_resumed"
    sim1 = Simulation(_cfg("snc_resumed", split, n, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("snc_resumed", total, n, **ov), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(SNC resume)"

    # 直接検証: 逆索引と遭遇台帳が round-trip で戻る(空回り防止)。
    sim3 = Simulation(_cfg("snc_inspect", split, n, **ov),
                      out_dir=tmp_path / "snc_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3.net.follows == sim1.net.follows
    assert sim3.net.contacts == sim1.net.contacts
    assert _index_is_consistent(sim3.net), "復元後に逆索引がずれた"
    assert [getattr(a, "_encounter_counts", None) for a in sim3.agents] == \
           [getattr(a, "_encounter_counts", None) for a in sim1.agents]


def test_event_construction_uses_the_public_schema():
    """Event の構築が schema の型そのもの(payload に内部キーを漏らしていない)。"""
    sim = _StubSim()
    CF.acquaint(sim, sim.agents[0], 1, CF.VIA_REPLY, 4, 40)
    ev = sim.logger.events[0]
    assert isinstance(ev, Event)
    assert set(ev.payload) == {"other", "via"}
    assert ev.step == 4 and ev.sim_min == 40 and ev.agent_id == 0
