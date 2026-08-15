"""DPH レーン(O = 観測 4 点 / C = 日跨ぎブロック / B = 二層予算 + FIFO)の検収。

正典: docs/plans/dayplan-horizon-plan.md §1.2 / §1.5 / §3.3 / §4.2 / §6

守るもの(検収基準の順)
  (1) **既定 OFF = 1 バイトも動かない**: 3 レーンとも純粋既定と L1 完全一致・新 kind 0 件・
      summary にキーなし・state 不在・LLM 呼数完全一致。
  (2) DPH-O は **ON でも世界不変**: 観測 ON / OFF で
      「行動列(L1 から観測イベントを除いた残り)」「乱数消費列」「LLM 呼数」が完全一致。
      = 観測がシムを変えないことの機械証明。
  (3) DPH-C: 24-29 時表記の受理 / end<start → +1440 / 「23:00-02:00」が 0 時をまたいで
      実行される / OFF では潰れることと、その潰れが wrap_clipped に数えられること /
      夜勤(18:00→02:00)の骨格が組めること / **1 暦日 1 計画**ガードとの整合。
  (4) DPH-B: レーン分割で reply が予約枠を必ず取れる(飢餓の解消)/ 総呼数は増えない
      (used <= cap を常に満たす)/ FIFO の全順序 / max_defer_steps 超過で骨格へ落ちる。
  (5) resume == straight(3 レーン ON)。
  (6) レジストリ宣言と本選 conf の整合。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society.cognition import day_plan as DP
from society.cognition import lod as LOD
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import starvation as SV
from society.observer import state_hash
from society.observer.schema import EVENT_KINDS

_REPO = Path(__file__).resolve().parents[1]

OBS = {"observer.starvation.enabled": "true"}
WRAP = {"planning.day_plan.enabled": "true",
        "planning.day_plan.wrap_blocks": "true"}
TIERS = {"lod.budget.tiers.enabled": "true"}
NEW_KINDS = ("reply_dropped", "plan_skipped", "reflect_dropped")


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


def _l1_without_obs(sim):
    """観測イベントを取り除いた「世界の行動列」。"""
    return [r for r in _l1(sim) if r[2] not in NEW_KINDS]


def _calls(sim) -> int:
    return len(sim.logger.llm_calls)


def _run(tmp_path, name, n_steps=144, n_agents=15, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim


def _world_state(sim, step: int) -> dict:
    """正準状態から「L1 の行数」だけを落としたもの(観測は行数しか動かしてよくない)。"""
    st = json.loads(state_hash.canonical_state(sim, step))
    st["world"].pop("n_events", None)
    return st


def _blk(start, end, place="food", act="meal", priority="should",
         flex="slideable", aim="sustenance"):
    return {"reason": "そうしたい", "start": start, "end": end, "place": place,
            "act": act, "with": [], "aim": aim, "priority": priority,
            "flex": flex, "note": ""}


def _resp(blocks):
    return json.dumps({"action": "plan", "mood": "ふつう", "carry": "",
                       "blocks": blocks, "if_then": []}, ensure_ascii=False)


# =========================================================================== #
# (1) 既定 OFF = 1 バイトも動かない
# =========================================================================== #
def test_all_three_lanes_default_off_is_byte_identical(tmp_path):
    """3 レーンを明示 false で置いても、キーを書かない純粋既定と L1 完全一致。"""
    base = _run(tmp_path, "dph_base")
    off = _run(tmp_path, "dph_off",
               **{"observer.starvation.enabled": "false",
                  "planning.day_plan.wrap_blocks": "false",
                  "lod.budget.tiers.enabled": "false"})
    assert _l1(base) == _l1(off)
    assert _calls(base) == _calls(off)


def test_default_off_emits_no_new_kinds_and_no_state(tmp_path):
    sim = _run(tmp_path, "dph_nokind")
    kinds = {e.kind for e in sim.logger.events}
    for k in NEW_KINDS:
        assert k in EVENT_KINDS, f"{k} が EVENT_KINDS に登録されていない"
        assert k not in kinds, f"既定 OFF なのに {k} が出ている"
    assert getattr(sim, "_starvation_state", None) is None
    assert SV.provenance(sim) is None
    assert sim.budget.counters is None and sim.budget.tiers is None


def test_default_off_summary_has_no_starvation_key(tmp_path):
    sim = _run(tmp_path, "dph_sum", n_steps=24)
    summary = json.load(open(sim.out_dir / "summary.json", encoding="utf-8"))
    assert "starvation" not in summary


# =========================================================================== #
# (2) DPH-O は ON でも世界不変(観測がシムを変えないことの機械証明)
# =========================================================================== #
def test_observation_on_does_not_change_the_world(tmp_path):
    """観測 ON/OFF で行動列・LLM 呼数・乱数消費が完全一致する。

    cap を絞って予算を binding にし(= reply_dropped が実際に発火する構成)、
    それでも世界が 1 バイトも動かないことを見る。
    """
    ov = {"lod.max_llm_per_step": 4, "planning.day_plan.enabled": "true"}
    off = _run(tmp_path, "obs_off", n_steps=120, n_agents=30, **ov)
    on = _run(tmp_path, "obs_on", n_steps=120, n_agents=30, **ov, **OBS)
    assert _l1(off) == _l1_without_obs(on), "観測 ON で行動列が変わった"
    assert _calls(off) == _calls(on), "観測 ON で LLM 呼数が変わった"
    # 乱数消費列: rng_key は「誰がどの step にどの stream を引いたか」の全記録なので、
    # これが一致していれば観測が draw を 1 粒も足していない(= 世界が同じ乱数を見た)。
    assert [c.get("llm_call_id") for c in off.logger.llm_calls] \
        == [c.get("llm_call_id") for c in on.logger.llm_calls]
    # 世界の最終状態(全個体の正準行 + 世界行)。**n_events だけは除く** — 観測イベントを
    # 足した分だけ増えるのは当たり前で、そこが増えないなら記録していないことになる。
    assert _world_state(off, 120) == _world_state(on, 120), \
        "観測 ON で世界の最終状態が変わった"
    prov = SV.provenance(on)
    assert prov["reply_dropped"] > 0, "この構成で reply が落ちていない(前提が崩れた)"
    assert {e.kind for e in on.logger.events} & set(NEW_KINDS)


def test_reply_dropped_matches_budget_denials(tmp_path):
    """reply_dropped の件数 = purpose 別カウンタの reply.denied(二重計上も欠落もない)。"""
    on = _run(tmp_path, "obs_cnt", n_steps=120, n_agents=30,
              **{"lod.max_llm_per_step": 4}, **OBS)
    prov = SV.provenance(on)
    n_ev = sum(1 for e in on.logger.events if e.kind == "reply_dropped")
    assert n_ev == prov["reply_dropped"] > 0
    assert prov["llm_budget_by_purpose"]["reply"]["denied"] == n_ev


def test_new_kinds_are_classified_in_the_causal_ledger():
    """★新 kind は `CAUSE_OF_KIND` に無いと `logger.log` が KeyError で落とす。

    本選 conf は `observer.causality.enabled: true` なので、分類を忘れると
    **最初の飢餓イベント 1 件でランが即死する**(既定 OFF のテストでは出ない事故)。
    """
    from society.observer import causality as C
    for kind in NEW_KINDS:
        assert kind in C.CAUSE_OF_KIND, f"{kind} が因果台帳に未分類(ランが即死する)"
        assert C.CAUSE_OF_KIND[kind] in C.CAUSE_TYPES
        assert C.cause_of(kind)                    # KeyError を投げない


def test_finals_combination_survives_a_starvation_event(tmp_path):
    """★本選と同じ組み合わせ(観測 ON × 因果台帳 ON × cap 拘束)で実際に落ちない。

    ラン中に飢餓イベントが 1 件以上出ることまで確かめる(0 件だと「落ちないこと」を
    確かめたことにならない = 前提が崩れたまま緑になる)。
    """
    from society.observer import causality as C
    common = {"lod.max_llm_per_step": 4,
              "observer.causality.enabled": "true",
              "planning.day_plan.enabled": "true",
              "planning.day_plan.wrap_blocks": "true"}
    # (a) 二層予算 OFF = 返事が落ちる経路 / (b) ON かつ繰り越しゼロ = 計画と内省が落ちる経路。
    #     どちらも本選 conf が通る道なので両方で「落ちない」ことを確かめる。
    cases = {
        "reply": dict(common),
        "plan": dict(common, **TIERS,
                     **{"lod.budget.tiers.max_defer_steps": 0}),
    }
    seen: set[str] = set()
    for name, ov in cases.items():
        sim = _run(tmp_path, f"finals_mix_{name}", n_steps=144, n_agents=60,
                   **ov, **OBS)
        starved = [e for e in sim.logger.events if e.kind in NEW_KINDS]
        assert starved, f"[{name}] 飢餓イベントが 1 件も出ていない(前提が崩れた)"
        for e in starved:
            assert e.cause_type == C.CAUSE_OF_KIND[e.kind], \
                f"{e.kind}: cause_type が刻まれていない/表と食い違う({e.cause_type})"
        seen |= {e.kind for e in starved}
        # 因果台帳 ON でも観測 4 点の集計は素通りする(summary が書けている)
        assert SV.provenance(sim)["llm_budget_by_purpose"]
    assert set(NEW_KINDS) <= seen, f"実ランで一度も出ていない kind: {set(NEW_KINDS) - seen}"


def test_budget_purposes_cover_every_take_site(tmp_path):
    """予算を取りに行く全ての site が purpose 別に数え分けられている。"""
    on = _run(tmp_path, "obs_purp", n_steps=120, n_agents=30,
              **{"lod.max_llm_per_step": 4}, **OBS)
    seen = set(SV.provenance(on)["llm_budget_by_purpose"])
    assert seen <= set(LOD.PURPOSE_LANE), f"未知の purpose: {seen - set(LOD.PURPOSE_LANE)}"
    assert {"face", "media", "reply"} <= seen


def test_plan_skipped_is_recorded(tmp_path):
    """予約 step に眠っていた/街の外に居た個体が L1 に 1 件残る(現行は静かに消える)。"""
    on = _run(tmp_path, "obs_skip", n_steps=144, n_agents=40,
              **{"planning.day_plan.enabled": "true"}, **OBS)
    ev = [e for e in on.logger.events if e.kind == "plan_skipped"]
    prov = SV.provenance(on)
    assert sum(prov["plan_skipped"].values()) == len(ev)
    assert all(e.payload["reason"] in ("outside", "sleeping", "defer_cap")
               for e in ev)


def test_wrap_clipped_counts_the_silent_crush(tmp_path):
    """「23:00-02:00」が潰される瞬間が数えられる(現行は round に埋もれる)。"""
    sim = _sim(tmp_path, "wc", n_steps=1, n_agents=4,
               **{"planning.day_plan.enabled": "true"}, **OBS)
    cfg = DP.cfg_of(sim)
    blocks, conts, errs = DP.validate_schema(
        {"action": "plan", "blocks": [_blk("23:00", "02:00")] * 4}, cfg)
    assert not [e for e in errs if e.startswith("bad_")]
    assert blocks[0]["start"] == 1380 and blocks[0]["end"] == 120
    out, _c2, ops = DP.repair(sim, sim.agents[0], blocks, conts, cfg, 0)
    assert SV.state(sim)["wrap_clipped"] == 4, "日跨ぎの圧潰が数えられていない"
    assert "wrap_clipped" not in ops, "REPAIR_OPS に私設キーが漏れている(L2 が動く)"
    assert set(ops) == set(DP.REPAIR_OPS)
    assert all(b["end"] - b["start"] == cfg["min_dur_min"] for b in out)


# =========================================================================== #
# (3) DPH-C 日跨ぎブロック
# =========================================================================== #
def test_hhmm_accepts_24h_notation_only_when_wrap_is_on():
    assert DP._hhmm("24:00") is None and DP._hhmm("26:00") is None
    assert DP._hhmm("23:59") == 1439
    assert DP._hhmm("24:00", 1800) == 1440
    assert DP._hhmm("26:30", 1800) == 1590
    assert DP._hhmm("30:00", 1800) is None          # 終端の外は受けない
    assert DP._hhmm("12:60", 1800) is None          # 分は 60 未満のまま


def test_wrap_on_reads_end_before_start_as_next_day():
    cfg = DP.build_cfg({"enabled": True, "wrap_blocks": True})
    blocks, _c, errs = DP.validate_schema(
        {"action": "plan", "blocks": [_blk("23:00", "02:00", place="nightlife",
                                           act="leisure", aim="enjoyment")] * 4},
        cfg)
    assert errs == [] or all(not e.startswith("bad_end") for e in errs)
    assert blocks[0]["start"] == 1380 and blocks[0]["end"] == 1560


def test_wrap_off_crushes_the_block_wrap_on_keeps_it(tmp_path):
    """同じ入力に対する OFF / ON の差そのもの(§1.2 のプローブ実測の再現)。"""
    rows = [_blk("23:00", "02:00", place="nightlife", act="leisure",
                 aim="enjoyment")] * 4
    for name, wrap, want in (("wrapoff", False, 10), ("wrapon", True, 180)):
        sim = _sim(tmp_path, name, n_steps=1, n_agents=4,
                   **{"planning.day_plan.enabled": "true",
                      "planning.day_plan.wrap_blocks": str(wrap).lower()})
        cfg = DP.cfg_of(sim)
        blocks, conts, _e = DP.validate_schema(
            {"action": "plan", "blocks": [dict(r) for r in rows]}, cfg)
        out, _c, _o = DP.repair(sim, sim.agents[0], blocks, conts, cfg, 0)
        assert out, f"wrap={wrap} で全ブロックが落ちた"
        assert out[0]["end"] - out[0]["start"] == want, \
            f"wrap={wrap} で継続 {out[0]['end'] - out[0]['start']} 分"


def test_wrap_block_executes_across_midnight():
    """23:00-02:00 のブロックが翌 01:00 に「いま実行すべき」と判定される。"""
    cfg_on = DP.build_cfg({"enabled": True, "wrap_blocks": True})
    cfg_off = DP.build_cfg({"enabled": True})
    plan = {"day": 3, "blocks": [{"start": 1380, "end": 1560, "state": "todo"}]}
    at_2300 = 3 * 1440 + 1380
    at_0100 = 4 * 1440 + 60
    assert DP.current_block(plan, at_2300, cfg_on) is not None
    assert DP.current_block(plan, at_0100, cfg_on) is not None, "0 時をまたげていない"
    # OFF では % 1440 判定なので翌 01:00 は窓の外(= 現行の挙動)
    assert DP.current_block(plan, at_0100, cfg_off) is None


def test_wrap_block_runs_after_midnight_end_to_end(tmp_path):
    """★実行系まで通しで: 23:00-02:00 の予定が翌 01:00 に動き出す(OFF は動かない)。"""
    got = {}
    for name, wrap in (("e2e_off", False), ("e2e_on", True)):
        sim = _sim(tmp_path, name, n_steps=1, n_agents=8,
                   **{"planning.day_plan.enabled": "true",
                      "planning.day_plan.wrap_blocks": str(wrap).lower()})
        a = sim.agents[0]
        DP.apply(sim, a, 0, 600,                   # 朝 10:00 に立てた計画
                 _resp([_blk("23:00", "02:00", place="food", act="meal")] * 4),
                 None)
        blocks = a._dayplan["blocks"]
        got[f"{name}_dur"] = blocks[0]["end"] - blocks[0]["start"]
        n0 = sum(1 for e in sim.logger.events if e.kind == "plan_block_start")
        rng = sim.hub.stream("decide", a.id, 1)
        DP.plan_action(a, sim, 1440 + 60, 1, rng, None)   # 翌日 01:00
        got[name] = sum(1 for e in sim.logger.events
                        if e.kind == "plan_block_start") - n0
    assert got["e2e_off_dur"] == 10 and got["e2e_on_dur"] == 180
    assert got["e2e_off"] == 0, "OFF なのに 0 時をまたいで計画が動いた"
    assert got["e2e_on"] == 1, "wrap ON でも 0 時をまたいだ予定が実行されない"


def test_wrap_plan_survives_midnight_only_as_far_as_its_own_tail():
    """地平は変えない: 前日の計画は自分の日跨ぎブロックが伸びている間だけ生き延びる。"""
    cfg = DP.build_cfg({"enabled": True, "wrap_blocks": True})

    class _A:
        _dayplan = {"day": 3, "blocks": [{"start": 1380, "end": 1560,
                                          "state": "todo"}]}
    a = _A()
    assert DP._plan_of(a, 3 * 1440 + 1400, cfg) is not None      # 当日 23:20
    assert DP._plan_of(a, 4 * 1440 + 60, cfg) is not None        # 翌 01:00(尾の中)
    assert DP._plan_of(a, 4 * 1440 + 130, cfg) is None           # 翌 02:10(尾の外)
    assert DP._plan_of(a, 5 * 1440 + 10, cfg) is None            # 翌々日は問答無用で失効

    class _B:                                      # 日跨ぎブロックが無い計画は従来どおり
        _dayplan = {"day": 3, "blocks": [{"start": 600, "end": 700, "state": "todo"}]}
    assert DP._plan_of(_B(), 4 * 1440 + 1, cfg) is None


def test_night_shift_skeleton_needs_wrap():
    """18:00→02:00 の夜勤が骨格で組める(OFF では 18:10 に潰れる)。"""
    class _A:
        work_start_min = 1080          # 18:00
        work_end_min = 120             # 02:00(日跨ぎ = 台帳の close<open)
        part_time = None
        node = ""
        money = 0.0
    for wrap, want in ((False, 10), (True, 480)):
        cfg = DP.build_cfg({"enabled": True, "wrap_blocks": wrap})
        rows = DP.skeleton(None, _A(), cfg)
        assert rows[0]["end"] - rows[0]["start"] == want, \
            f"wrap={wrap} の夜勤骨格が {rows[0]['end'] - rows[0]['start']} 分"


def test_wrap_on_keeps_one_plan_per_calendar_day(tmp_path):
    """★「1 暦日 1 計画」ガードとの整合: wrap ON でも 2 本目は立たない。"""
    sim = _run(tmp_path, "wrap_guard", n_steps=300, n_agents=25, **WRAP, **OBS)
    per = Counter((e.agent_id, e.sim_min // 1440)
                  for e in sim.logger.events if e.kind == "plan_created")
    assert per and max(per.values()) == 1, f"同暦日に 2 本立った: {per.most_common(3)}"


def test_wrap_on_does_not_change_the_number_of_plan_calls(tmp_path):
    """DPH-C は朝の呼数を 1 も動かさない(地平ガードに触っていない証拠)。"""
    off = _run(tmp_path, "wrap_calls_off", n_steps=200, n_agents=25,
               **{"planning.day_plan.enabled": "true"})
    on = _run(tmp_path, "wrap_calls_on", n_steps=200, n_agents=25, **WRAP)
    n_off = Counter(c["purpose"] for c in off.logger.llm_calls)
    n_on = Counter(c["purpose"] for c in on.logger.llm_calls)
    assert n_off["plan"] == n_on["plan"], f"plan 呼数が動いた: {n_off} vs {n_on}"


def test_wrap_off_is_byte_identical_to_day_plan_alone(tmp_path):
    a = _run(tmp_path, "wrap_off_a", n_steps=144, n_agents=15,
             **{"planning.day_plan.enabled": "true"})
    b = _run(tmp_path, "wrap_off_b", n_steps=144, n_agents=15,
             **{"planning.day_plan.enabled": "true",
                "planning.day_plan.wrap_blocks": "false"})
    assert _l1(a) == _l1(b)


# =========================================================================== #
# (4) DPH-B 二層予算 + FIFO
# =========================================================================== #
def test_tiered_budget_never_exceeds_the_cap():
    """総量は増えない(どのレーンから取っても used <= cap)。"""
    tiers = LOD.build_budget_cfg({"enabled": True, "reply_share": 0.2,
                                  "life_share": 0.3})
    b = LOD.LodBudget(10, tiers=tiers)
    got = sum(1 for _ in range(100) if b.take("media"))
    assert got == b.caps["general"] == 5, "general が自分の枠を越えた"
    assert sum(1 for _ in range(100) if b.take("reply")) == 2
    assert sum(1 for _ in range(100) if b.take("plan")) == 3
    assert b.used == 10 <= b.max_per_step


def test_reserved_lanes_are_never_eaten_by_general():
    """発火(general)が先に走り切っても、返答保証の予約枠は必ず残る。"""
    tiers = LOD.build_budget_cfg({"enabled": True})
    b = LOD.LodBudget(100, tiers=tiers)
    for _ in range(1000):                          # general が全力で取りに行く
        b.take("media")
    assert b.take("reply") is True, "予約枠が発火に食われている(DPH-B の要件違反)"
    assert b.take("plan") is True


def test_reserved_lane_may_borrow_general_slack():
    """逆向き(予約超過 → general の余り)は許す = 遊ばせない。"""
    tiers = LOD.build_budget_cfg({"enabled": True, "reply_share": 0.1,
                                  "life_share": 0.1})
    b = LOD.LodBudget(10, tiers=tiers)
    n = sum(1 for _ in range(100) if b.take("reply"))
    assert n == 9, f"reply が general の余りを借りられていない(取れたのは {n})"
    assert b.take("media") is False


def test_tiers_off_budget_is_identical_to_the_legacy_counter():
    b = LOD.LodBudget(3)
    assert [b.take() for _ in range(5)] == [True, True, True, False, False]
    b.reset()
    assert b.take("reply") is True and b.tiers is None


def _reply_rate(sim) -> float:
    row = SV.provenance(sim)["llm_budget_by_purpose"].get(
        "reply", {"granted": 0, "denied": 0})
    return row["granted"] / max(1, row["granted"] + row["denied"])


def test_tiers_relieve_the_reply_starvation(tmp_path):
    """★実測: cap 拘束下で reply の飢餓が解消する(§1.5 の cap60 実験の縮小再現)。

    実測(60 体 × 144 step・cap 4・mock・seed 42):
        OFF … 返答保証の成立率 2.6%(reply 呼 2 件 / 落ちた返事 74 件)
        ON  … 成立率 100%      (reply 呼 23 件 / 落ちた返事 0 件)・総呼数は 502 → 218
    「話しかけられたのに誰も返さない街」が、総呼数を 1 も増やさずに解消する。
    """
    ov = {"lod.max_llm_per_step": 4, "planning.day_plan.enabled": "true"}
    off = _run(tmp_path, "tier_off", n_steps=144, n_agents=60, **ov, **OBS)
    on = _run(tmp_path, "tier_on", n_steps=144, n_agents=60, **ov, **OBS, **TIERS)
    n_off = Counter(c["purpose"] for c in off.logger.llm_calls)
    n_on = Counter(c["purpose"] for c in on.logger.llm_calls)
    r_off, r_on = _reply_rate(off), _reply_rate(on)
    assert r_off < 0.5, f"前提が崩れた(OFF で reply が枯渇していない: {r_off:.3f})"
    assert r_on > 0.9, f"二層予算でも reply が枯渇したまま: {r_on:.3f}"
    assert n_on["reply"] > n_off["reply"], \
        f"二層予算で reply 呼が増えていない: {n_off['reply']} → {n_on['reply']}"
    assert SV.provenance(on)["reply_dropped"] < SV.provenance(off)["reply_dropped"]
    # 総量は増えない(plan/reflect が予算の中へ入るので ON は必ず OFF 以下)
    assert sum(n_on.values()) <= sum(n_off.values()), \
        f"総呼数が増えた: {sum(n_off.values())} → {sum(n_on.values())}"


def test_tiers_cap_the_per_step_calls(tmp_path):
    """ON では **予算外呼が 1 本も無い** = 1 step の総呼数が cap を超えない。"""
    cap = 4
    on = _run(tmp_path, "tier_cap", n_steps=144, n_agents=40,
              **{"lod.max_llm_per_step": cap,
                 "planning.day_plan.enabled": "true"}, **TIERS)
    per_step = Counter(c["step"] for c in on.logger.llm_calls)
    assert per_step and max(per_step.values()) <= cap, \
        f"1 step の呼数が cap を超えた: {max(per_step.values())} > {cap}"


def test_tiers_off_run_is_byte_identical(tmp_path):
    a = _run(tmp_path, "tier_id_a", n_steps=100, n_agents=20,
             **{"planning.day_plan.enabled": "true"})
    b = _run(tmp_path, "tier_id_b", n_steps=100, n_agents=20,
             **{"planning.day_plan.enabled": "true",
                "lod.budget.tiers.enabled": "false"})
    assert _l1(a) == _l1(b)


def test_defer_cap_falls_back_to_the_skeleton(tmp_path):
    """繰り越し上限を超えた計画は **LLM ゼロ**で骨格へ落ちる(失わない)。"""
    on = _run(tmp_path, "tier_defer", n_steps=200, n_agents=60,
              **{"lod.max_llm_per_step": 2, "planning.day_plan.enabled": "true",
                 "lod.budget.tiers.max_defer_steps": 1}, **TIERS, **OBS)
    skipped = [e for e in on.logger.events
               if e.kind == "plan_skipped" and e.payload["reason"] == "defer_cap"]
    assert skipped, "繰り越し上限に到達しなかった(構成が緩すぎる)"
    fb = [e for e in on.logger.events
          if e.kind == "plan_created" and e.payload["src"] == "skeleton"]
    assert len(fb) >= len(skipped)
    # 繰り越し上限で据えた骨格には **LLM 呼が 1 本も紐付かない**(呼数を増やしていない証拠)。
    # ★同じ src="skeleton" でも「応答が壊れて後退した」経路は call_id を持つので、
    #   ここでは「call_id を持たない骨格が skipped と同数以上ある」ことで分離する。
    free = [e for e in fb if e.llm_call_id is None]
    assert len(free) >= len(skipped), \
        f"LLM ゼロの骨格が足りない: {len(free)} < {len(skipped)}"
    for e in skipped:
        assert e.payload["waited"] >= 1


def test_defer_is_fifo_by_first_reservation_then_id():
    """キューの全順序 = (最初に予約された step, agent_id)。"""
    class _A:
        def __init__(self, aid, first):
            self.id = aid
            self.plan_due_step = first
    rows = [_A(9, -1), _A(3, -1), _A(7, 4), _A(1, 6)]
    rows.sort(key=lambda a: (scheduler._defer_first(a, "plan_due_step", 10), a.id))
    assert [a.id for a in rows] == [7, 1, 3, 9]


# =========================================================================== #
# (5) resume == straight(3 レーン ON)
# =========================================================================== #
def _rows(run_dir: Path, stem="l1_events"):
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def test_resume_matches_straight_with_all_lanes_on(tmp_path):
    ov = {"planning.day_plan.enabled": "true",
          "planning.day_plan.wrap_blocks": "true",
          "observer.starvation.enabled": "true",
          "lod.budget.tiers.enabled": "true",
          "lod.max_llm_per_step": "6"}
    straight = tmp_path / "dph_straight"
    Simulation(_cfg("dph_straight", 40, 20, **ov), out_dir=straight).run()

    resumed = tmp_path / "dph_resumed"
    every = {"observer.checkpoint_every": "20"}
    s1 = Simulation(_cfg("dph_resumed", 20, 20, **ov, **every), out_dir=resumed)
    for step in range(20):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 20, resumed / "checkpoint" / "ckpt-000020.pkl.gz")
    s1.logger.flush_segment()
    s2 = Simulation(_cfg("dph_resumed", 40, 20, **ov, **every), out_dir=resumed)
    s2.run(resume_from=resumed)

    assert _rows(straight) == _rows(resumed), "resume != straight(L1)"
    a = json.load(open(straight / "summary.json", encoding="utf-8"))["starvation"]
    b = json.load(open(resumed / "summary.json", encoding="utf-8"))["starvation"]
    assert a == b, f"starvation の累積タリーが resume で食い違う: {a} vs {b}"


# =========================================================================== #
# (6) 宣言と本選 conf の整合
# =========================================================================== #
def test_new_toggles_are_declared():
    ids = {f.id for f in R.FEATURES}
    for fid in ("observer.starvation.enabled", "planning.day_plan.wrap_blocks",
                "lod.budget.tiers.enabled"):
        assert fid in ids, f"{fid} がレジストリに無い"
    by_id = {f.id: f for f in R.FEATURES}
    # 観測は記録専用 = 呼数を動かさない / 二層予算は正直に affects_k=True
    assert by_id["observer.starvation.enabled"].affects_k is False
    assert by_id["planning.day_plan.wrap_blocks"].affects_k is False
    assert by_id["lod.budget.tiers.enabled"].affects_k is True
    for fid in ("observer.starvation.enabled", "planning.day_plan.wrap_blocks",
                "lod.budget.tiers.enabled"):
        assert by_id[fid].fingerprint_risk == "none"


def test_finals_profile_turns_the_three_lanes_on():
    fin = R.flatten_bools(__import__("omegaconf").OmegaConf.load(
        _REPO / "conf" / "finals_observe.yaml"))
    for fid in ("observer.starvation.enabled", "planning.day_plan.wrap_blocks",
                "lod.budget.tiers.enabled"):
        assert fin.get(fid) is True, f"本選 conf で {fid} が ON になっていない"


def test_finals_keeps_the_fire_block_commented_out():
    """D1 は「8/15 診断の実測を見てから」= この時点では 1 行も解凍されていない。"""
    text = (_REPO / "conf" / "finals_observe.yaml").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("fire:"), "cognition.fire が解凍されている"
        assert not stripped.startswith("watch:"), "cognition.watch が解凍されている"
    assert "# fire:" in text, "解凍待ちの fire 行(コメント)が消えている"
    from omegaconf import OmegaConf
    cog = OmegaConf.load(_REPO / "conf" / "finals_observe.yaml").get("cognition")
    assert "fire" not in cog and "watch" not in cog and "engaged" not in cog


# =========================================================================== #
# (7) DPH-O ⑤ = step あたりの used / cap(第117 レーンB3・**観測のみ**)
# =========================================================================== #
def test_used_per_step_is_absent_when_observation_is_off(tmp_path):
    """既定 OFF では summary に starvation ごと出ない(= llm_budget も出ない)。"""
    sim = _run(tmp_path, "sb_off", n_steps=24)
    assert SV.provenance(sim) is None
    assert "step_budget" not in (getattr(sim, "_starvation_state", None) or {})


def test_used_per_step_observation_does_not_change_the_world(tmp_path):
    """★観測 ON/OFF で行動列・LLM 呼数・世界の最終状態が完全一致(⑤ を足しても不変)。"""
    ov = {"lod.max_llm_per_step": 4, "planning.day_plan.enabled": "true",
          "lod.budget.tiers.enabled": "true"}
    off = _run(tmp_path, "sb_world_off", n_steps=120, n_agents=30, **ov)
    on = _run(tmp_path, "sb_world_on", n_steps=120, n_agents=30, **ov, **OBS)
    assert _l1(off) == _l1_without_obs(on), "⑤ の観測で行動列が変わった"
    assert _calls(off) == _calls(on), "⑤ の観測で LLM 呼数が変わった"
    assert [c.get("llm_call_id") for c in off.logger.llm_calls] \
        == [c.get("llm_call_id") for c in on.logger.llm_calls]
    assert _world_state(off, 120) == _world_state(on, 120), \
        "⑤ の観測で世界の最終状態が変わった"
    assert SV.provenance(on)["llm_budget"]["steps"] == 120


def test_used_per_step_matches_the_granted_calls(tmp_path):
    """used の総和 = 予算が許可した呼の総数(二重計上も欠落もない)。"""
    cap = 4
    on = _run(tmp_path, "sb_sum", n_steps=120, n_agents=30,
              **{"lod.max_llm_per_step": cap, "planning.day_plan.enabled": "true"},
              **OBS)
    prov = SV.provenance(on)
    lb = prov["llm_budget"]
    assert lb["steps"] == 120
    assert lb["used_total"] == prov["llm_budget_granted_total"] > 0, \
        "used の総和が purpose 別 granted の総和と食い違う"
    assert lb["used_per_step"]["mean"] == round(lb["used_total"] / 120, 4)
    assert 0 <= lb["used_per_step"]["p95"] <= cap
    assert lb["cap_per_step_mean"] == float(cap)
    assert 0.0 < lb["used_over_cap_mean"] <= 1.0, \
        "充填率が (0, 1] の外(used <= cap の硬い上限が壊れている)"
    assert lb["used_over_cap_mean"] == round(lb["used_per_step"]["mean"] / cap, 4)


def test_used_p95_is_the_nearest_rank_of_the_histogram():
    """p95 はヒストグラムからの最近傍順位法(numpy 非依存・step 列を持たない)。"""
    assert SV._p95_from_hist({}) == 0
    assert SV._p95_from_hist({0: 100}) == 0
    # 100 件中 95 件が 1・5 件が 9 → ceil(0.95*100)=95 件目 = 1
    assert SV._p95_from_hist({1: 95, 9: 5}) == 1
    # 100 件中 94 件が 1・6 件が 9 → 95 件目 = 9
    assert SV._p95_from_hist({1: 94, 9: 6}) == 9
    assert SV._p95_from_hist({0: 1, 4: 1}) == 4


def test_tight_cap_shows_a_binding_budget(tmp_path):
    """cap を締めると充填率が上がる = 「cap が binding だったか」が事後に読める。"""
    ov = {"planning.day_plan.enabled": "true"}
    tight = _run(tmp_path, "sb_tight", n_steps=120, n_agents=40,
                 **{"lod.max_llm_per_step": 2}, **ov, **OBS)
    loose = _run(tmp_path, "sb_loose", n_steps=120, n_agents=40,
                 **{"lod.max_llm_per_step": 200}, **ov, **OBS)
    t = SV.provenance(tight)["llm_budget"]["used_over_cap_mean"]
    l = SV.provenance(loose)["llm_budget"]["used_over_cap_mean"]
    assert t > l, f"cap を締めても充填率が上がらない: tight={t} loose={l}"


# =========================================================================== #
# (8) DPH-B の繰り越し予約 staleness(第117 レーンB3 追加発注)
#
#   `plan_due_step` は「最初に予約された step」の印で、`plan_step` とセットでしか意味を
#   持たない。退場ヘルパ(`health._exit_world` = 死亡 / `population._leave_world` = 転出)は
#   `plan_step` を -1 に落とすのに印を残していたため、同じ実体が次に予約を受けた日に
#   `_defer_first` が何日も前の step を返し、`waited` が巨大 = 初回から max_defer_steps
#   超過扱いで **LLM 計画を一度も撃たずに骨格へ落ちる**。
#   ★どちらのヘルパもプール回転で再実体化された退場者へ毎 step 冪等に貼り直されるので
#     (health.phase / population の ready フック)、印の寿命は実体の寿命より長くなりうる。
# =========================================================================== #
def _carry_over_one_plan(sim, agent, step: int):
    """予算ゼロで 1 回だけ繰り越させ、(plan_step, plan_due_step) を立てる。"""
    for a in sim.agents:
        a.plan_step, a.plan_due_step = -1, -1
        a.sleeping, a.loc = False, "street"
    agent.plan_step = step
    scheduler._phase_planning(sim, step, sim.clock.sim_min(step))
    assert agent.plan_step == step + 1 and agent.plan_due_step == step, \
        "前提が崩れた(予算ゼロでも繰り越されていない)"


def _next_reservation_is_fresh(sim, agent, later: int) -> dict:
    """ずっと後の step で予約し直したときに「待たされた人」扱いされないか。"""
    for a in sim.agents:
        a.plan_step, a.sleeping, a.loc = -1, False, "street"
    agent.plan_step = later
    n0 = len(sim.logger.events)
    scheduler._phase_planning(sim, later, sim.clock.sim_min(later))
    new = sim.logger.events[n0:]
    return {"plan_step": agent.plan_step,
            "defer_cap": [e for e in new if e.kind == "plan_skipped"
                          and e.payload["reason"] == "defer_cap"],
            "skeleton": [e for e in new if e.kind == "plan_created"
                         and e.payload.get("src") == "skeleton"]}


def _stale_sim(tmp_path, name):
    """tiers ON・予算ゼロ・繰り越し上限 1・観測 ON(= 骨格落ちが L1 に出る)。"""
    return _sim(tmp_path, name, n_steps=1, n_agents=6,
                **{"lod.max_llm_per_step": 0,
                   "lod.budget.tiers.max_defer_steps": 1,
                   "planning.day_plan.enabled": "true"}, **TIERS, **OBS)


def test_death_clears_the_carry_over_reservation(tmp_path):
    """★死亡(`health._exit_world`)が繰り越しの印を残さない。"""
    from society import health as H

    sim = _stale_sim(tmp_path, "stale_death")
    agent = sim.agents[0]
    _carry_over_one_plan(sim, agent, 10)
    H._exit_world(agent)
    assert agent.plan_step == -1, "退場で予約が畳まれていない(前提が崩れた)"
    assert int(getattr(agent, "plan_due_step", -1)) == -1, \
        "★繰り越しの印が宙に浮いている(plan_due_step が残った)"
    # 帰結: ずっと後の日に予約し直しても「待たされた人」にならない
    got = _next_reservation_is_fresh(sim, agent, 200)
    assert got["defer_cap"] == [], \
        f"新しい予約が初回から defer_cap 扱いになった: {got['defer_cap'][0].payload}"
    assert got["skeleton"] == [], "LLM 計画を撃つ前に骨格へ落ちた"
    assert got["plan_step"] == 201, "普通に翌 step へ繰り越されていない"


def test_emigration_clears_the_carry_over_reservation(tmp_path):
    """★転出(`population._leave_world`)が繰り越しの印を残さない(死亡版と同型)。"""
    from society import population as P

    sim = _stale_sim(tmp_path, "stale_emigrate")
    agent = sim.agents[0]
    _carry_over_one_plan(sim, agent, 10)
    P._leave_world(agent)
    assert agent.plan_step == -1
    assert int(getattr(agent, "plan_due_step", -1)) == -1, \
        "★繰り越しの印が宙に浮いている(plan_due_step が残った)"
    got = _next_reservation_is_fresh(sim, agent, 200)
    assert got["defer_cap"] == [] and got["skeleton"] == []
    assert got["plan_step"] == 201


def test_exit_helpers_grow_no_attribute_when_tiers_are_off(tmp_path):
    """★既定(tiers OFF)では属性を 1 つも生やさない = pickle も L1 もバイト不変。"""
    from society import health as H
    from society import population as P

    sim = _sim(tmp_path, "stale_off", n_steps=1, n_agents=4)
    assert sim.budget.tiers is None
    a, b = sim.agents[0], sim.agents[1]
    for x in (a, b):
        assert not hasattr(x, "plan_due_step")
    H._exit_world(a)
    P._leave_world(b)
    for x in (a, b):
        assert not hasattr(x, "plan_due_step"), \
            "tiers OFF なのに繰り越し用の属性が生えた(既定挙動が変わっている)"
