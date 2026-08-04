"""IF-A(行為 → 思考の来歴リンク IF-1 + 監査 §5 の穴 2 件)のテスト。

正典
  - docs/research/llm-world-interface-audit.md §4(llm_call_id の現状)・§5(穴)・§6 IF-1
  - docs/research/if-lane-research.md §4(W3C PROV-DM の wasInformedBy・(llm_call_id, role) 対)
  - docs/plans/if-sv-p4-plan.md 「IF-A」行

守るもの(検収基準の順)
  (1) 既定 OFF = ゴールデン L1 バイト一致・純粋既定と一致・新キー/新 kind が 1 件も出ない
  (2) ON: LLM が決めた行為の L1 イベントに (llm_call_id, llm_role) が付き、
      l1b_llm と **(call_id, agent_id, step) で一意に**引ける(同 step 複数発火の曖昧さの解消)
  (3) role の語彙は l1b_llm の purpose と同一(新語彙を作っていない)
  (4) 刻むのは**行為者自身のイベントだけ**(聞き手の hear には付かない)
  (5) 穴2: 熟慮経路の plan/recall/reflect は ON のときだけ
      fallback{reason:"misrouted_action"} を出す(世界への作用は従来どおり no-op)
  (6) 穴1: contingency(CONDS 7 種 × THENS 5 種)の決定論評価・適用。
      前提機構が OFF の条件は**常に不成立**(捏造しない)
  (7) ON の決定論(同 seed 2 ラン一致)と resume==straight
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society import provlink as PL
from society import registry as R
from society.cognition import day_plan as DP
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

LINK_ON = {"observer.llm_link": "true"}
DAYPLAN_ON = {"planning.day_plan.enabled": "true"}
CONT_ON = {**DAYPLAN_ON, "planning.day_plan.use_contingency": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_day_plan.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    """ゴールデン比較の形(step / agent_id / kind / payload)。"""
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l1x(sim):
    """決定論比較の形(llm_call_id 列まで含める)。"""
    return [[e.step, e.agent_id, e.kind, e.llm_call_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _linked(sim):
    """payload に llm_role が入っているイベント(= 来歴の辺が張られたもの)。"""
    return [e for e in sim.logger.events if "llm_role" in e.payload]


class _FixedLLM:
    """固定応答スタブ(test_day_plan._FixedLLM と同型)。call_id は呼び番号。"""

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_defaults_are_off():
    cfg = load_config()
    assert bool(cfg.observer.llm_link) is False
    assert bool(cfg.planning.day_plan.use_contingency) is False
    assert DP.build_cfg(None)["use_contingency"] is False


def test_off_matches_golden(tmp_path):
    """既定 OFF は変更前ゴールデンと一字一句一致(IF-A の seam が完全な no-op)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "link_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-A の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(llm_call_id 列まで含めて)。"""
    pure = _sim(tmp_path, "link_pure")
    pure.run()
    off = _sim(tmp_path, "link_off",
               **{"observer.llm_link": "false",
                  "planning.day_plan.use_contingency": "false"})
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_no_new_keys_or_kinds(tmp_path):
    """OFF では llm_role / plan_cont_fire / misrouted_action が 1 件も出ない。"""
    sim = _sim(tmp_path, "link_off_keys")
    sim.run()
    assert not _linked(sim), "OFF なのに payload に llm_role が生えた"
    assert not _kind(sim, "plan_cont_fire"), "OFF なのに plan_cont_fire が出た"
    assert not [e for e in _kind(sim, "fallback")
                if e.payload.get("reason") == "misrouted_action"], \
        "OFF なのに misrouted_action が出た"
    # 行為イベントの llm_call_id は従来どおり空のまま(思考系イベントにだけ入る)
    for e in sim.logger.events:
        if e.kind in ("speak", "sns_post", "dm", "spend"):
            assert e.llm_call_id is None, f"OFF なのに {e.kind} に id が入った"


def test_off_never_stamps_the_temp_key(tmp_path):
    """OFF では行為 dict に一時キーそのものが積まれない(構造による保証)。"""
    sim = _sim(tmp_path, "link_nokey", n_steps=1)
    action = {"type": "speak", "text": "やあ"}
    PL.stamp(sim, action, "call-1", "social")
    assert PL.PROV_KEY not in action
    assert PL.take(action) is None


def test_registry_and_schema_declared():
    feats = {f.id: f for f in R.FEATURES}
    link = feats["observer.llm_link"]
    assert link.repro_tier == "strict" and link.affects_k is False
    assert link.fingerprint_risk == "none"
    cont = feats["planning.day_plan.use_contingency"]
    assert cont.repro_tier == "journal" and cont.affects_k is False
    assert cont.fingerprint_risk == "none"
    assert "plan_cont_fire" in EVENT_KINDS


# --------------------------------------------------------------------------- #
# (B) ON: (llm_call_id, role) の付与(検収基準 2〜4)
# --------------------------------------------------------------------------- #
def test_on_links_speech_to_the_llm_call(tmp_path):
    """deliberate → speak の行為イベントが l1b_llm と一意に突き合う。"""
    sim = _sim(tmp_path, "link_on")
    sim.run()
    speaks = _kind(sim, "speak")
    assert speaks, "テスト前提が崩れた(発話が 1 件も無い)"
    for e in speaks:
        assert e.llm_call_id is None, "ON にしていないのに id が付いた"

    on = _sim(tmp_path, "link_on2", **LINK_ON)
    on.run()
    speaks = _kind(on, "speak")
    assert speaks
    for e in speaks:
        assert e.llm_call_id, "ON なのに speak に llm_call_id が無い"
        assert "llm_role" in e.payload
        rows = [r for r in on.logger.llm_calls
                if r["llm_call_id"] == e.llm_call_id
                and r["agent_id"] == e.agent_id and r["step"] == e.step]
        assert len(rows) == 1, \
            f"(call_id, agent, step) で一意に引けない: {len(rows)} 件"
        assert rows[0]["purpose"] == e.payload["llm_role"], \
            "role が l1b_llm の purpose と食い違う(語彙の単一源が崩れた)"


def test_on_role_vocabulary_is_the_l1b_purpose_set(tmp_path):
    """role は l1b_llm の purpose の部分集合(新語彙を 1 つも作っていない)。"""
    sim = _sim(tmp_path, "link_role", **LINK_ON)
    sim.run()
    roles = {e.payload["llm_role"] for e in _linked(sim)}
    purposes = {r["purpose"] for r in sim.logger.llm_calls}
    assert roles, "ON なのに辺が 1 本も張られていない"
    assert roles <= purposes, f"l1b に無い role 語彙が出た: {roles - purposes}"


def test_on_does_not_stamp_other_agents_events(tmp_path):
    """聞き手側の hear は刻まない(別の主体の activity なので辺を張らない)。"""
    sim = _sim(tmp_path, "link_hear", **LINK_ON)
    sim.run()
    hears = _kind(sim, "hear")
    assert hears, "テスト前提が崩れた(聴取が 1 件も無い)"
    for e in hears:
        assert "llm_role" not in e.payload, "聞き手の hear に辺が張られた"
        assert e.llm_call_id is None


def test_on_parquet_joins_one_to_one(tmp_path):
    """成果物(parquet)側でも join できる = 事後解析が (step, agent) join に依存しない。"""
    sim = _sim(tmp_path, "link_pq", **LINK_ON)
    sim.run()
    l1 = pq.read_table(tmp_path / "link_pq" / "l1_events.parquet").to_pylist()
    l1b = pq.read_table(tmp_path / "link_pq" / "l1b_llm.parquet").to_pylist()
    index = {}
    for r in l1b:
        index.setdefault((r["llm_call_id"], r["agent_id"], r["step"]), []).append(r)
    acts = [r for r in l1
            if r["kind"] in ("speak", "sns_post", "dm") and r["llm_call_id"]]
    assert acts, "ON なのに来歴付きの行為イベントが parquet に無い"
    for r in acts:
        hit = index.get((r["llm_call_id"], r["agent_id"], r["step"]))
        assert hit is not None and len(hit) == 1
        assert json.loads(r["payload"])["llm_role"] == hit[0]["purpose"]


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "link_det_a", **LINK_ON)
    a.run()
    b = _sim(tmp_path, "link_det_b", **LINK_ON)
    b.run()
    assert _l1x(a) == _l1x(b), "llm_link ON の決定論が崩れている"


def test_on_adds_no_llm_calls(tmp_path):
    """辺を張るだけ=LLM 呼数は 1 本も動かない(affects_k=False の実測)。"""
    off = _sim(tmp_path, "link_k_off")
    off.run()
    on = _sim(tmp_path, "link_k_on", **LINK_ON)
    on.run()
    assert on.llm.calls == off.llm.calls > 0


def test_plan_block_execution_links_to_the_morning_call(tmp_path):
    """day_plan のブロック実行(role="plan")が朝の計画呼へ辿れる。"""
    sim = _sim(tmp_path, "link_plan", n_steps=288, **LINK_ON, **DAYPLAN_ON,
               **{"run.start_tod": "05:00", "run.natural_start": "true"})
    sim.run()
    assert _kind(sim, "plan_created"), "テスト前提が崩れた(計画が立っていない)"
    routed = [e for e in _kind(sim, "route_start")
              if e.payload.get("llm_role") == "plan"]
    assert routed, "計画ブロック由来の移動に role='plan' の辺が張られていない"
    for e in routed:
        rows = [r for r in sim.logger.llm_calls
                if r["llm_call_id"] == e.llm_call_id and r["purpose"] == "plan"]
        assert rows, "role='plan' の id が朝の計画呼を指していない"


def test_plan_dict_has_no_call_id_when_off(tmp_path):
    """OFF では plan に call_id キー自体が生えない(状態も checkpoint も不変)。"""
    sim = _sim(tmp_path, "link_plan_off", n_steps=1, **DAYPLAN_ON)
    a = sim.agents[0]
    DP.apply(sim, a, 0, 8 * 60, "{壊れた", "cid-1")
    assert "call_id" not in a._dayplan
    on = _sim(tmp_path, "link_plan_on", n_steps=1, **LINK_ON, **DAYPLAN_ON)
    b = on.agents[0]
    DP.apply(on, b, 0, 8 * 60, "{壊れた", "cid-1")
    assert b._dayplan["call_id"] == "cid-1"


# --------------------------------------------------------------------------- #
# (C) 穴2: 熟慮経路の plan / recall / reflect(検収基準 5)
# --------------------------------------------------------------------------- #
_MISROUTED = json.dumps(
    {"action": "plan",
     "items": [{"when": "昼", "what": "散歩", "place": "公園"}]},
    ensure_ascii=False)


def _misrouted_run(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=72, n_agents=12, **ov)
    sim.llm = _FixedLLM(_MISROUTED)
    sim.run()
    return sim


def test_misrouted_action_is_silent_when_off(tmp_path):
    sim = _misrouted_run(tmp_path, "mis_off")
    assert not [e for e in _kind(sim, "fallback")
                if e.payload.get("reason") == "misrouted_action"]


def test_misrouted_action_is_counted_when_on(tmp_path):
    """ON でだけ fallback{reason:"misrouted_action"} が出る(世界への作用は no-op のまま)。"""
    off = _misrouted_run(tmp_path, "mis_base")
    on = _misrouted_run(tmp_path, "mis_on", **LINK_ON)
    mis = [e for e in _kind(on, "fallback")
           if e.payload.get("reason") == "misrouted_action"]
    assert mis, "ON なのに misrouted_action が 1 件も出ない"
    assert {e.payload["action"] for e in mis} == {"plan"}
    for e in mis:                                  # 来歴の辺も張られている(原因の呼が判る)
        assert e.llm_call_id and e.payload["llm_role"]
    # 世界への作用は不変: misrouted 以外のイベント列は OFF と完全一致
    def _rest(s):
        return [[e.step, e.agent_id, e.kind] for e in s.logger.events
                if not (e.kind == "fallback"
                        and e.payload.get("reason") == "misrouted_action")]
    assert _rest(off) == _rest(on), "明示分岐が世界を動かしている(no-op でない)"


def test_misrouted_branch_covers_all_three_verbs(tmp_path):
    """plan / recall / reflect の 3 種すべてが明示分岐に入る(直接適用)。"""
    sim = _sim(tmp_path, "mis_three", n_steps=1, **LINK_ON)
    a = sim.agents[0]
    before = len(sim.logger.events)
    for kind in ("plan", "recall", "reflect"):
        scheduler._apply(sim, a, {"type": kind}, 0, 0)
    out = sim.logger.events[before:]
    assert [e.payload["action"] for e in out] == ["plan", "recall", "reflect"]
    assert all(e.kind == "fallback"
               and e.payload["reason"] == "misrouted_action" for e in out)
    # 3 語すべてが parse_action の既知動詞(= undefined_action に落ちない領域)
    from society.cognition.deliberate import KNOWN_ACTIONS
    assert {"plan", "recall", "reflect"} <= KNOWN_ACTIONS


# --------------------------------------------------------------------------- #
# (D) 穴1: contingency の消費(検収基準 6)
# --------------------------------------------------------------------------- #
def _cont_sim(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, **CONT_ON, **ov)
    return sim, sim.agents[0]


def _block(start=8 * 60, end=9 * 60, place="food", act="meal", node="n",
           **extra):
    row = {"start": start, "end": end, "place": place, "act": act, "with": [],
           "purpose": DP.ACT_PURPOSE[act], "priority": "should",
           "flex": "slideable", "reason": "", "note": "", "node": node,
           "state": "todo", "slid": 0}
    row.update(extra)
    return row


def _plan(agent, blocks, cont, day=0):
    agent._dayplan = {"schema": DP.SCHEMA, "day": day, "version": 1,
                      "model": "mock", "mood": "", "carry": "", "src": "llm",
                      "blocks": blocks, "cont": cont}
    return agent._dayplan


# ---- 条件 7 種(単体・合成状態で決定論に発火)---------------------------------- #
def test_cond_rain(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_rain")
    b = _block()
    assert DP._cond_rain(sim, a, b, DP.cfg_of(sim), 0) is False   # 天気なし=不成立
    sim.today_weather = {"cond": "晴", "temp_hi": 20}
    assert DP._cond_rain(sim, a, b, DP.cfg_of(sim), 0) is False
    sim.today_weather = {"cond": "雨", "temp_hi": 14}
    assert DP._cond_rain(sim, a, b, DP.cfg_of(sim), 0) is True
    sim.today_weather = {"cond": "雪", "temp_hi": 1}
    assert DP._cond_rain(sim, a, b, DP.cfg_of(sim), 0) is True


def test_cond_crowded(tmp_path):
    from society import envfeedback as EF
    sim, a = _cont_sim(tmp_path, "c_crowd", **{"env.feedback.enabled": "true",
                                               "env.feedback.poi.enabled": "true"})
    node = sorted(sim.dests)[0]
    b = _block(node=node)
    assert DP._cond_crowded(sim, a, b, DP.cfg_of(sim), 0) is False
    EF.state(sim)["poi_hold"][node] = 999          # 待ち行列で塞がっている
    assert DP._cond_crowded(sim, a, b, DP.cfg_of(sim), 0) is True


def test_cond_crowded_is_false_without_envfeedback(tmp_path):
    """前提機構 OFF では常に不成立(欠測を『混んでいる』と捏造しない)。"""
    sim, a = _cont_sim(tmp_path, "c_crowd_off")
    b = _block(node=sorted(sim.dests)[0])
    assert DP._cond_crowded(sim, a, b, DP.cfg_of(sim), 0) is False


def test_cond_closed(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_closed", **{"commerce.enabled": "true"})
    cfg = DP.cfg_of(sim)
    b = _block(place="food")                       # food は 11:00〜23:00
    assert DP._cond_closed(sim, a, b, cfg, 9 * 60) is True
    assert DP._cond_closed(sim, a, b, cfg, 13 * 60) is False
    assert DP._cond_closed(sim, a, _block(place="home"), cfg, 9 * 60) is False


def test_cond_closed_is_false_without_commerce(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_closed_off")
    assert DP._cond_closed(sim, a, _block(place="food"), DP.cfg_of(sim),
                           4 * 60) is False


def test_cond_tired(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_tired", **{"health.enabled": "true"})
    cfg = DP.cfg_of(sim)
    high = float(sim.healthcfg["fatigue_high"])
    a.fatigue = high - 0.01
    assert DP._cond_tired(sim, a, _block(), cfg, 0) is False
    a.fatigue = high
    assert DP._cond_tired(sim, a, _block(), cfg, 0) is True


def test_cond_tired_is_false_without_health(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_tired_off")
    a.fatigue = 1.0
    assert DP._cond_tired(sim, a, _block(), DP.cfg_of(sim), 0) is False


def test_cond_no_money(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_money", **{"economy.enabled": "true"})
    cfg = DP.cfg_of(sim)
    a.money = 0.0
    assert DP._cond_no_money(sim, a, _block(place="food"), cfg, 0) is True
    a.money = 10 ** 6
    assert DP._cond_no_money(sim, a, _block(place="food"), cfg, 0) is False
    a.money = 0.0                                  # 価格の定義が無いカテゴリは判定不能=False
    assert DP._cond_no_money(sim, a, _block(place="office", act="work"),
                             cfg, 0) is False


def test_cond_late(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_late")
    cfg = DP.cfg_of(sim)
    assert DP._cond_late(sim, a, _block(), cfg, 0) is False
    assert DP._cond_late(sim, a, _block(slid=30), cfg, 0) is True


def test_cond_invited(tmp_path):
    sim, a = _cont_sim(tmp_path, "c_invited")
    cfg = DP.cfg_of(sim)
    b = _block(8 * 60, 9 * 60)
    assert DP._cond_invited(sim, a, b, cfg, 0) is False
    a.joint_today = {"poi": "x", "band": (12 * 60, 13 * 60),
                     "activity": "meal", "activity_tag": "", "tier": 1}
    assert DP._cond_invited(sim, a, b, cfg, 0) is False   # 時間帯が重ならない
    a.joint_today["band"] = (8 * 60 + 30, 10 * 60)
    assert DP._cond_invited(sim, a, b, cfg, 0) is True


def test_every_cond_and_then_has_an_implementation():
    """列挙(CONDS/THENS)と実装表が完全に対応する(片方だけ増えたら落ちる)。"""
    assert set(DP.COND_FN) == set(DP.CONDS)
    assert set(DP.THEN_FN) == set(DP.THENS)
    assert set(DP.INDOOR_SWAP) == set(DP.OUTDOOR_PLACES)
    assert set(DP.INDOOR_SWAP.values()) <= set(DP.PLACE_CATS)
    assert not set(DP.INDOOR_SWAP.values()) & set(DP.OUTDOOR_PLACES), \
        "swap_indoor の差し替え先が屋外カテゴリになっている"


# ---- 対処 5 種(単体)----------------------------------------------------------- #
def test_then_skip(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_skip")
    b = _block()
    assert DP._then_skip(sim, a, b, DP.cfg_of(sim), 0, 0) is True
    assert b["state"] == "dropped"
    ev = _kind(sim, "plan_block_drop")
    assert len(ev) == 1 and ev[0].payload["reason"] == "cont_skip"
    assert sim._dayplan_state["drop"] == 1, "drop の保存則(件数=イベント数)が崩れた"


def test_then_postpone(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_post")
    cfg = DP.cfg_of(sim)
    b = _block(8 * 60, 9 * 60)
    assert DP._then_postpone(sim, a, b, cfg, 0, 0) is True
    assert b["start"] == 8 * 60 + cfg["postpone_min"]
    assert b["end"] - b["start"] == 60, "ずらしで継続まで変わった(最小摂動でない)"
    assert b["slid"] == cfg["postpone_min"]
    assert _kind(sim, "plan_slide") and sim._dayplan_state["slide"] == 1


def test_then_postpone_drops_at_slide_cap(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_post_cap")
    cfg = dict(DP.cfg_of(sim), max_slide_min=30)
    b = _block(8 * 60, 9 * 60)
    assert DP._then_postpone(sim, a, b, cfg, 0, 0) is True
    assert b["state"] == "dropped"
    assert _kind(sim, "plan_block_drop")[0].payload["reason"] == "cont_postpone"


def test_then_go_home(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_home")
    cfg = DP.cfg_of(sim)
    b = _block(place="food", act="meal")
    assert DP._then_go_home(sim, a, b, cfg, 0, 0) is True
    assert b["place"] == "home" and b["act"] == "home"
    assert b["node"] == a.home_node
    assert DP._then_go_home(sim, a, b, cfg, 0, 0) is False    # もう自宅=空振り
    a.home_node = ""                                          # 自宅が無ければ削る
    c = _block(place="food")
    assert DP._then_go_home(sim, a, c, cfg, 0, 0) is True
    assert c["state"] == "dropped"
    assert _kind(sim, "plan_block_drop")[-1].payload["reason"] == "cont_no_place"


def test_then_swap_indoor(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_swap")
    cfg = DP.cfg_of(sim)
    assert DP._then_swap_indoor(sim, a, _block(place="food"), cfg, 0, 0) is False
    b = _block(place="street", act="walk", node=None)
    assert DP._then_swap_indoor(sim, a, b, cfg, 12 * 60, 12 * 60) is True
    assert b["place"] not in DP.OUTDOOR_PLACES
    assert b["node"] is not None or b["place"] == "home"


def test_then_shorten(tmp_path):
    sim, a = _cont_sim(tmp_path, "t_short")
    cfg = DP.cfg_of(sim)
    b = _block(8 * 60, 9 * 60)
    assert DP._then_shorten(sim, a, b, cfg, 0, 0) is True
    assert b["end"] == 9 * 60 - cfg["shorten_min"]
    tight = _block(8 * 60, 8 * 60 + cfg["min_dur_min"])
    assert DP._then_shorten(sim, a, tight, cfg, 0, 0) is False   # もう最小=空振り


# ---- 発火の作法(順序・冪等・OFF)---------------------------------------------- #
def test_contingency_fires_first_match_only_once(tmp_path):
    sim, a = _cont_sim(tmp_path, "cf_once")
    cfg = DP.cfg_of(sim)
    sim.today_weather = {"cond": "雨", "temp_hi": 12}
    b = _block(slid=30)
    plan = _plan(a, [b], [{"if": "rain", "then": "shorten"},
                          {"if": "late", "then": "skip"}])
    assert DP.apply_contingency(sim, a, plan, b, cfg, 0, 8 * 60) is True
    ev = _kind(sim, "plan_cont_fire")
    assert len(ev) == 1
    assert ev[0].payload["cond"] == "rain" and ev[0].payload["then"] == "shorten"
    assert ev[0].payload["block"] == 0 and ev[0].payload["applied"] is True
    assert b["state"] == "todo", "先頭一致でない対処(skip)まで適用された"
    # 2 回目は発火しない(1 ブロック 1 回=先送りの振動を作らない)
    assert DP.apply_contingency(sim, a, plan, b, cfg, 1, 8 * 60) is False
    assert len(_kind(sim, "plan_cont_fire")) == 1


def test_contingency_records_unapplied_firing(tmp_path):
    """条件は成立したが対処が空振り(既に屋内)なら applied=false で残す。"""
    sim, a = _cont_sim(tmp_path, "cf_noop")
    sim.today_weather = {"cond": "雨", "temp_hi": 12}
    b = _block(place="food")
    plan = _plan(a, [b], [{"if": "rain", "then": "swap_indoor"}])
    DP.apply_contingency(sim, a, plan, b, DP.cfg_of(sim), 0, 8 * 60)
    ev = _kind(sim, "plan_cont_fire")
    assert len(ev) == 1 and ev[0].payload["applied"] is False


def test_contingency_is_not_consumed_when_toggle_off(tmp_path):
    """use_contingency OFF(第86 と同一)では plan_action が cont を読まない。"""
    sim = _sim(tmp_path, "cf_off", n_steps=1, **DAYPLAN_ON)
    a = sim.agents[0]
    sim.today_weather = {"cond": "雨", "temp_hi": 12}
    node = next(n for n in sorted(sim.dests) if n != a.node)
    plan = _plan(a, [_block(8 * 60, 9 * 60, place="food", node=node)],
                 [{"if": "rain", "then": "skip"}])
    rng = sim.hub.stream("decide", a.id, 0)
    DP.plan_action(a, sim, 8 * 60, 0, rng)
    assert not _kind(sim, "plan_cont_fire")
    assert plan["blocks"][0]["state"] == "done", "OFF なのに計画が中止された"


def test_contingency_runs_through_plan_action(tmp_path):
    """plan_action の作用点で発火し、中止されたブロックはその step を空き時間にする。"""
    sim, a = _cont_sim(tmp_path, "cf_exec")
    sim.today_weather = {"cond": "雨", "temp_hi": 12}
    node = next(n for n in sorted(sim.dests) if n != a.node)
    plan = _plan(a, [_block(8 * 60, 9 * 60, place="food", node=node)],
                 [{"if": "rain", "then": "skip"}])
    rng = sim.hub.stream("decide", a.id, 0)
    assert DP.plan_action(a, sim, 8 * 60, 0, rng) is None, \
        "中止したのに行き先が返った"
    assert _kind(sim, "plan_cont_fire")
    assert plan["blocks"][0]["state"] == "dropped"


def test_contingency_run_is_deterministic(tmp_path):
    """ON ラン(288 step)の同 seed 2 ラン一致 + 乱数を 1 本も足していない。"""
    ov = {**CONT_ON, **LINK_ON, "weather.enabled": "true",
          "commerce.enabled": "true",
          "run.start_tod": "05:00", "run.natural_start": "true"}
    a = _sim(tmp_path, "cf_det_a", n_steps=288, **ov)
    a.run()
    b = _sim(tmp_path, "cf_det_b", n_steps=288, **ov)
    b.run()
    assert _l1x(a) == _l1x(b), "contingency ON の決定論が崩れている"


# --------------------------------------------------------------------------- #
# (E) resume(検収基準 7)
# --------------------------------------------------------------------------- #
def test_resume_matches_straight(tmp_path):
    """llm_link + day_plan + contingency ON で resume==straight(parquet バイト一致)。"""
    ov = {**CONT_ON, **LINK_ON, "run.start_tod": "00:00",
          "run.natural_start": "true"}
    split, total = 60, 120
    straight_dir = tmp_path / "lk_straight"
    straight = Simulation(_cfg("lk_straight", total, 15, **ov),
                          out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "plan_created"), "テスト前提が崩れた(計画が立っていない)"

    d = tmp_path / "lk_resumed"
    sim1 = Simulation(_cfg("lk_resumed", split, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("lk_resumed", total, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(llm_link/contingency resume)"


# --------------------------------------------------------------------------- #
# (F) no-fingerprint(プロンプトを 1 バイトも変えない)
# --------------------------------------------------------------------------- #
def test_prompts_are_byte_identical_between_off_and_on(tmp_path):
    """観測を足しただけ = 全プロンプトが OFF と 1 バイトも変わらない。"""
    def _prompts(name, **ov):
        sim = _sim(tmp_path, name, n_steps=72, n_agents=12, **ov)
        seen: list = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            seen.append(prompt)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return seen

    off = _prompts("fp_off", **DAYPLAN_ON)
    on = _prompts("fp_on", **CONT_ON, **LINK_ON)
    assert off and off == on, "IF-A でプロンプトが変わった(no-fingerprint 違反)"
