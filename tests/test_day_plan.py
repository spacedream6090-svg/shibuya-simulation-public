"""day_plan v1(第86バッチ・planning.day_plan)のテスト。

正典: docs/plans/source/design-discussion-20260802.md §6・§7 /
      docs/plans/dayplan-engaged-plan.md 第86。

守るもの(検収基準の順)
  (1) 既定 OFF = 純粋既定と L1 バイト一致・プロンプト不変・新 kind ゼロ件・L2 に列なし・
      summary にキーなし・state 不在・LLM 呼数不変
  (2) スキーマ検証の全分岐(列挙・個数・型・時刻)
  (3) 修復の決定論(同入力 → 同修復)と操作子ごとの効き(round/slide/truncate/substitute/
      drop/clip)。ずらしは**最小摂動**
  (4) 物理検証(移動時間・営業時間・場所カテゴリの実在)
  (5) フォールバック(壊れた JSON → 前日計画 → ペルソナ既定ルーチン)。世界を止めない
  (6) 場所解決の決定論(習慣 → 距離 → node 昇順)
  (7) 割り込み: could+droppable は **LLM 呼を 1 本も増やさず**削られる / slideable はずれる /
      **must が脅かされたときだけ** replan が鳴る / plan_exception は fire ON のときだけ
  (8) ON ラン: 同 seed 2 ラン L1 一致・resume==straight・k 不変・L2 が state と一致
  (9) no-fingerprint: 機構語・実験条件語・因子名がプロンプト全文に出現しない
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import registry as R
from society.cognition import day_plan as DP
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

ON = {"planning.day_plan.enabled": "true"}
NEW_KINDS = {"plan_created", "plan_repair", "plan_fallback", "plan_block_start",
             "plan_block_drop", "plan_slide", "plan_replan"}
L2_COLS = ("plan_blocks_mean", "plan_exec_total", "plan_repair_total",
           "plan_fallback_total", "plan_drop_total", "plan_slide_total",
           "plan_replan_total")


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


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _spy(sim) -> list:
    seen: list = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    return seen


def _blk(start, end, place="food", act="meal", priority="should",
         flex="slideable", aim="sustenance", **extra):
    """LLM 生出力 1 ブロック(reason は先頭に置く=スキーマ指示と同順)。"""
    row = {"reason": "そうしたい", "start": start, "end": end, "place": place,
           "act": act, "with": [], "aim": aim, "priority": priority,
           "flex": flex, "note": ""}
    row.update(extra)
    return row


def _resp(blocks, cont=None, **meta):
    body = {"action": "plan", "mood": "ふつう", "carry": "",
            "blocks": blocks, "if_then": cont or []}
    body.update(meta)
    return json.dumps(body, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.planning.day_plan.enabled) is False
    assert DP.build_cfg(None)["enabled"] is False


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。state 不在・L2 に列なし・summary 無風。"""
    pure = _sim(tmp_path, "pure")
    pure_summary = pure.run()
    off = _sim(tmp_path, "expl_off", **{"planning.day_plan.enabled": "false"})
    off_summary = off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(day_plan seam が no-op でない)"
    assert getattr(off, "_dayplan_state", None) is None, "OFF なのに _dayplan_state が生えた"
    assert "day_plan" not in pure_summary and "day_plan" not in off_summary
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    for c in L2_COLS:
        assert c not in cols, f"OFF なのに L2 に {c} 列"


def test_off_emits_no_new_event_kinds_and_keeps_old_prompt(tmp_path):
    """OFF では新 kind 0 件・プロンプトに day_plan マーカーが無く従来の計画タスクが載る。"""
    sim = _sim(tmp_path, "off_prompt")
    seen = _spy(sim)
    sim.run()
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]
    blob = "\n".join(seen)
    assert DP.PROMPT_MARK not in blob, "OFF なのに day_plan のスキーマ節がプロンプトに載った"
    assert "今日一日の計画" in blob, "従来の計画タスクが消えている"
    assert _kind(sim, "day_plan"), "従来の day_plan イベントが消えている"


def test_off_llm_call_count_unchanged(tmp_path):
    a = _sim(tmp_path, "cc_pure")
    a.run()
    b = _sim(tmp_path, "cc_off", **{"planning.day_plan.enabled": "false"})
    b.run()
    assert a.llm.calls == b.llm.calls > 0


def test_requires_planning_enabled(tmp_path):
    """planning.enabled=false のまま day_plan ON にしても完全 no-op(朝の呼びが前提)。"""
    base = _sim(tmp_path, "np_base", **{"planning.enabled": "false"})
    base.run()
    on = _sim(tmp_path, "np_on", **{"planning.enabled": "false", **ON})
    on.run()
    assert not DP.enabled(on), "planning OFF なのに day_plan が実効になっている"
    assert _l1(base) == _l1(on)
    assert getattr(on, "_dayplan_state", None) is None


def test_event_kinds_and_registry_declared():
    for k in NEW_KINDS:
        assert k in EVENT_KINDS, f"{k} が EVENT_KINDS に未登録"
    feat = {f.id: f for f in R.FEATURES}["planning.day_plan.enabled"]
    assert feat.repro_tier == "journal"
    assert feat.affects_k is True, "affects_k の宣言(plan_exception が発火点を動かす)が落ちた"
    assert feat.fingerprint_risk == "possible"


# --------------------------------------------------------------------------- #
# (B) スキーマ検証(検収基準 2)
# --------------------------------------------------------------------------- #
_C = DP.build_cfg({"enabled": True})


def test_validate_schema_accepts_well_formed():
    blocks, conts, errs = DP.validate_schema(
        json.loads(_resp([_blk("08:00", "09:00"), _blk("12:00", "13:00"),
                          _blk("15:00", "16:00"), _blk("19:00", "20:00")],
                         [{"if": "rain", "then": "skip"}])), _C)
    assert errs == [], errs
    assert len(blocks) == 4 and len(conts) == 1
    assert blocks[0]["start"] == 480 and blocks[0]["end"] == 540
    assert blocks[0]["state"] == "todo" and blocks[0]["node"] is None


def test_validate_schema_flags_broken_and_empty():
    assert DP.validate_schema(None, _C) == ([], [], ["broken_json:root"])
    assert DP.validate_schema({}, _C)[2] == ["bad_action:root", "no_blocks:root"]
    assert "no_blocks:root" in DP.validate_schema(
        {"action": "plan", "blocks": []}, _C)[2]
    assert "not_object:b0" in DP.validate_schema(
        {"action": "plan", "blocks": ["x"]}, _C)[2]
    # 綴り違い(if_then / contingency)はどちらも読む
    assert DP.validate_schema(
        {"action": "plan", "blocks": [_blk("08:00", "09:00")],
         "contingency": [{"if": "rain", "then": "skip"}]}, _C)[1] == \
        [{"if": "rain", "then": "skip"}]


def test_validate_schema_flags_every_bad_enum():
    """列挙外は欄ごとに個別のエラーコードで出る(値は None のまま=修復側が埋める)。"""
    bad = _blk("08:00", "09:00", place="宇宙", act="踊る", aim="なんとなく",
               priority="超重要", flex="ぐにゃ")
    blocks, _c, errs = DP.validate_schema(_j([bad]), _C)
    for code in ("bad_place:b0", "bad_act:b0", "bad_aim:b0",
                 "bad_priority:b0", "bad_flex:b0"):
        assert code in errs, f"{code} が検出されていない: {errs}"
    b = blocks[0]
    assert b["place"] is None and b["act"] is None and b["purpose"] is None
    assert b["priority"] is None and b["flex"] is None


def test_validate_schema_flags_bad_times():
    blocks, _c, errs = DP.validate_schema(
        _j([_blk("25:70", None), _blk("あさ", "ひる")]), _C)
    assert "bad_start:b0" in errs and "bad_end:b0" in errs
    assert "bad_start:b1" in errs and "bad_end:b1" in errs
    assert blocks[0]["start"] is None and blocks[0]["end"] is None


def test_validate_schema_counts_blocks():
    few = DP.validate_schema(_j([_blk("08:00", "09:00")]), _C)[2]
    assert "too_few:root" in few
    many = DP.validate_schema(
        _j([_blk(f"{6 + i:02d}:00", f"{6 + i:02d}:30") for i in range(10)]), _C)[2]
    assert "too_many:root" in many


def test_validate_schema_contingency_branches():
    raw = _j([_blk("08:00", "09:00")] * 4,
             cont=[{"if": "rain", "then": "skip"}, {"if": "宇宙", "then": "skip"},
                   "not a dict", {"if": "tired", "then": "go_home"},
                   {"if": "closed", "then": "postpone"},
                   {"if": "crowded", "then": "swap_indoor"}])
    _b, conts, errs = DP.validate_schema(raw, _C)
    assert "bad_conting:c1" in errs and "not_object:c2" in errs
    assert "too_many_conting:root" in errs
    assert len(conts) == 4                         # 妥当な 4 件が残る(clip は修復側)


def test_place_alias_is_absorbed():
    """列挙外だが意味の近い語(cafe/park/…)は 1 段だけ吸収する = substitute の材料。"""
    blocks, _c, errs = DP.validate_schema(
        _j([_blk("08:00", "09:00", place="cafe"),
            _blk("10:00", "11:00", place="park"),
            _blk("12:00", "13:00", place="theatre"),
            _blk("14:00", "15:00", place="movie")]), _C)
    assert [b["place"] for b in blocks] == ["food", "leisure", "hall", "cinema"]
    assert not [e for e in errs if e.startswith("bad_place")]


def test_hhmm_parsing_edges():
    assert DP._hhmm("00:00") == 0 and DP._hhmm("23:59") == 1439
    assert DP._hhmm("8:5") == 8 * 60 + 5
    assert DP._hhmm(480) == 480 and DP._hhmm("480") == 480
    for bad in ("24:00", "12:60", "-1:00", "", None, True, "あ", 1441):
        assert DP._hhmm(bad) is None, bad


def test_enums_are_grounded_in_existing_vocabulary():
    """活動語彙は既存の計画語彙の部分集合(新語を足していない)= 既存写像が改変なしで効く。"""
    from society.cognition import plan_schema as PS
    from society.cognition import routine as RT
    assert set(DP.ACTS) <= set(PS._DEFAULT_VOCAB) | {"meetup"}
    assert set(DP.ACTS) <= set(RT._WHAT_CAT), "routine の what→cat3 写像に無い活動語がある"
    assert set(DP.ACT_PLACE) == set(DP.ACTS) == set(DP.ACT_PURPOSE)
    assert set(DP.PURPOSE_CAT) == set(DP.PURPOSES)
    assert set(DP.PURPOSE_CAT.values()) == {"mandatory", "maintenance", "discretionary"}
    assert set(DP.ACT_PLACE.values()) <= set(DP.PLACE_CATS)


# --------------------------------------------------------------------------- #
# (C) 修復(検収基準 3)
# --------------------------------------------------------------------------- #
def _agent_sim(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, **{**ON, **ov})
    return sim, sim.agents[0]


def test_repair_is_deterministic(tmp_path):
    """同じ入力からは常に同じ修復が出る(乱数ゼロ)。2 回通して完全一致。"""
    sim, a = _agent_sim(tmp_path, "det_rep")
    raw = _j([_blk("08:07", "09:03"), _blk("08:50", "10:00", place="cafe"),
              _blk("13:00", "12:00"), _blk("18:00", "19:00", priority="must",
                                           flex="fixed")])
    outs = []
    for _ in range(2):
        b, c, e = DP.validate_schema(raw, DP.cfg_of(sim))
        b, c, ops = DP.repair(sim, a, b, c, DP.cfg_of(sim), 0)
        outs.append((json.dumps(b, ensure_ascii=False, sort_keys=True), ops))
    assert outs[0] == outs[1], "同入力なのに修復結果が違う(決定論が崩れている)"


def test_repair_rounds_times_to_grid(tmp_path):
    sim, a = _agent_sim(tmp_path, "round")
    b, c, _e = DP.validate_schema(_j([_blk("08:07", "09:03")]), DP.cfg_of(sim))
    b, _c, ops = DP.repair(sim, a, b, c, DP.cfg_of(sim), 0)
    assert ops["round"] == 1
    assert b[0]["start"] % 10 == 0 and b[0]["end"] % 10 == 0
    assert b[0]["start"] == 490 and b[0]["end"] == 540


def test_repair_fills_missing_times_and_enforces_min_duration(tmp_path):
    sim, a = _agent_sim(tmp_path, "fill")
    b, c, _e = DP.validate_schema(
        _j([_blk("08:00", None), _blk(None, "12:00"), _blk(None, None),
            _blk("15:00", "15:00")]), DP.cfg_of(sim))
    b, _c, ops = DP.repair(sim, a, b, c, DP.cfg_of(sim), 0)
    assert ops["round"] >= 4
    for row in b:
        assert row["end"] - row["start"] >= DP.cfg_of(sim)["min_dur_min"]


def test_repair_truncates_over_long_block(tmp_path):
    sim, a = _agent_sim(tmp_path, "trunc")
    cfg = DP.cfg_of(sim)
    b, c, _e = DP.validate_schema(_j([_blk("00:00", "23:00")]), cfg)
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["truncate"] >= 1
    assert b[0]["end"] - b[0]["start"] <= cfg["max_dur_min"]


def test_repair_slides_overlapping_blocks_minimally(tmp_path):
    """重なり/移動時間不足は**後ろへずらす**(最小摂動)。継続は保つ。"""
    sim, a = _agent_sim(tmp_path, "slide")
    cfg = DP.cfg_of(sim)
    b, c, _e = DP.validate_schema(
        _j([_blk("08:00", "10:00", place="food"),
            _blk("08:30", "09:30", place="shop")]), cfg)
    before_dur = [row["end"] - row["start"] for row in b]
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["slide"] >= 1, ops
    assert len(b) == 2
    assert b[1]["start"] >= b[0]["end"], "重なりが解消していない"
    assert [row["end"] - row["start"] for row in b] == before_dur, \
        "ずらしで継続時間まで変わった(最小摂動でない)"
    assert b[1]["slid"] > 0


def test_repair_truncates_previous_when_next_is_fixed(tmp_path):
    """次が fixed(動かせない)なら前を短縮して固定を守る(ずらしでなく truncate)。"""
    sim, a = _agent_sim(tmp_path, "fixnext")
    cfg = DP.cfg_of(sim)
    b, c, _e = DP.validate_schema(
        _j([_blk("08:00", "12:00", place="home", act="home", flex="slideable"),
            _blk("11:00", "12:00", place="work", act="work",
                 priority="must", flex="fixed", aim="livelihood")]), cfg)
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["truncate"] >= 1, ops
    assert b[1]["start"] == 11 * 60, "fixed が動かされた"
    assert b[0]["end"] <= b[1]["start"], "fixed の前が短縮されていない"


def test_repair_substitutes_missing_enums(tmp_path):
    """欠損/列挙外の欄は既定へ寄せる(act→place / act→aim / priority / flex)。"""
    sim, a = _agent_sim(tmp_path, "subst")
    cfg = DP.cfg_of(sim)
    b, c, _e = DP.validate_schema(
        _j([_blk("08:00", "09:00", place="宇宙", aim="?", priority="?", flex="?",
                 act="shop")]), cfg)
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["substitute"] >= 4, ops
    assert b[0]["place"] == DP.ACT_PLACE["shop"]
    assert b[0]["purpose"] == DP.ACT_PURPOSE["shop"]
    assert b[0]["priority"] == "should" and b[0]["flex"] == "slideable"


def test_repair_drops_block_with_no_act_and_no_place(tmp_path):
    sim, a = _agent_sim(tmp_path, "nodrop")
    cfg = DP.cfg_of(sim)
    b, c, _e = DP.validate_schema(
        _j([_blk("08:00", "09:00", place="宇宙", act="踊る"),
            _blk("10:00", "11:00")]), cfg)
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["drop"] >= 1 and len(b) == 1


def test_repair_clips_to_max_blocks_lowest_priority_first(tmp_path):
    """件数上限は **LLM が宣言した priority の弱い方・遅い方**から落とす。"""
    sim, a = _agent_sim(tmp_path, "clip")
    cfg = DP.cfg_of(sim)
    rows = [_blk(f"{6 + i:02d}:00", f"{6 + i:02d}:30",
                 priority="must" if i < 4 else "could",
                 flex="slideable") for i in range(10)]
    b, c, _e = DP.validate_schema(_j(rows), cfg)
    b, _c, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert ops["clip"] == 2, ops
    assert len(b) <= cfg["max_blocks"]
    assert sum(1 for row in b if row["priority"] == "must") == 4, \
        "must が clip で落ちた(弱い方から落としていない)"


def test_repair_clips_contingency(tmp_path):
    sim, a = _agent_sim(tmp_path, "clipc")
    cfg = DP.cfg_of(sim)
    raw = _j([_blk("08:00", "09:00")] * 4,
             cont=[{"if": "rain", "then": "skip"},
                   {"if": "tired", "then": "go_home"},
                   {"if": "closed", "then": "postpone"},
                   {"if": "crowded", "then": "swap_indoor"}])
    b, c, _e = DP.validate_schema(raw, cfg)
    _b, c2, ops = DP.repair(sim, a, b, c, cfg, 0)
    assert len(c2) == cfg["max_conting"] and ops["clip"] >= 1


# --------------------------------------------------------------------------- #
# (D) 物理検証(検収基準 4)
# --------------------------------------------------------------------------- #
def test_travel_min_is_a_lower_bound_and_monotone(tmp_path):
    sim, a = _agent_sim(tmp_path, "travel")
    cfg = DP.cfg_of(sim)
    nodes = sorted(sim.dests)[:6]
    assert DP.travel_min(sim, cfg, nodes[0], nodes[0]) == cfg["transfer_min"]
    assert DP.travel_min(sim, cfg, None, nodes[0]) == cfg["transfer_min"]
    far = max(nodes[1:], key=lambda n: DP._dist(sim, nodes[0], n))
    near = min(nodes[1:], key=lambda n: DP._dist(sim, nodes[0], n))
    assert DP.travel_min(sim, cfg, nodes[0], far) >= \
        DP.travel_min(sim, cfg, nodes[0], near)


def test_validate_physical_flags_unreachable_and_resolves_nodes(tmp_path):
    """移動時間の下限すら満たさない並びを unreachable として挙げ、node を埋める。"""
    sim, a = _agent_sim(tmp_path, "phys")
    cfg = DP.cfg_of(sim)
    b, _c, _e = DP.validate_schema(
        _j([_blk("08:00", "09:00", place="food"),
            _blk("09:00", "10:00", place="shop")]), cfg)
    errs = DP.validate_physical(sim, a, b, cfg, 0)
    assert any(e.startswith("unreachable") for e in errs), errs
    assert all(row["node"] is not None for row in b), "解決結果が node に入っていない"


def test_validate_physical_flags_unresolvable_category(tmp_path):
    sim, a = _agent_sim(tmp_path, "phys_np")
    cfg = DP.cfg_of(sim)
    a.home_node = ""                              # 自宅が無い = home が解けない
    b, _c, _e = DP.validate_schema(
        _j([_blk("08:00", "09:00", place="home", act="home")]), cfg)
    assert "no_place:b0" in DP.validate_physical(sim, a, b, cfg, 0)


def test_validate_physical_flags_overflow(tmp_path):
    sim, a = _agent_sim(tmp_path, "phys_ov")
    cfg = dict(DP.cfg_of(sim), day_end_min=600)
    b, _c, _e = DP.validate_schema(_j([_blk("18:00", "19:00")]), cfg)
    assert any(e.startswith("overflow") for e in DP.validate_physical(sim, a, b, cfg, 0))


# --------------------------------------------------------------------------- #
# (E) フォールバック(検収基準 5)
# --------------------------------------------------------------------------- #
def test_fallback_to_skeleton_on_broken_json(tmp_path):
    """壊れた JSON を注入 → 前日計画が無いのでペルソナ既定ルーチンへ後退(世界を止めない)。"""
    sim, a = _agent_sim(tmp_path, "fb_skel")
    DP.apply(sim, a, 0, 8 * 60, "{壊れた", None)
    ev = _kind(sim, "plan_fallback")
    assert len(ev) == 1 and ev[0].payload["kind"] == "skeleton"
    assert a._dayplan["blocks"], "フォールバックしても計画が空(世界が止まる)"
    assert a._dayplan["src"] == "skeleton"
    assert sim._dayplan_state["fallback"] == 1


def test_fallback_to_prev_day_when_available(tmp_path):
    """前日の計画があるならそちらが優先(骨格より人物固有の情報が多い)。"""
    sim, a = _agent_sim(tmp_path, "fb_prev")
    good = _resp([_blk("08:00", "09:00", place="food", act="meal"),
                  _blk("11:00", "12:00", place="shop", act="shop"),
                  _blk("14:00", "15:00", place="leisure", act="leisure"),
                  _blk("18:00", "19:00", place="home", act="home")])
    DP.apply(sim, a, 0, 8 * 60, good, None)
    assert a._dayplan["src"] == "llm"
    DP.apply(sim, a, 200, 1440 + 8 * 60, "ぐちゃぐちゃ", None)
    ev = _kind(sim, "plan_fallback")
    assert len(ev) == 1 and ev[0].payload["kind"] == "prev_day"
    assert a._dayplan["day"] == 1 and a._dayplan["blocks"]


def test_fallback_counts_are_per_model(tmp_path):
    """修復・フォールバックの件数と率が **モデル別** に summary へ残る(§7 の追加指標)。"""
    sim, a = _agent_sim(tmp_path, "fb_model")
    DP.apply(sim, a, 0, 8 * 60, "{", None)
    prov = DP.provenance(sim)
    mid = DP.model_id(sim)
    assert mid == "mock"
    assert prov["by_model"][mid]["fallback_plans"] == 1
    assert prov["by_model"][mid]["fallback_rate"] == 1.0
    assert prov["by_model"][mid]["fallback_kinds"] == {"skeleton": 1}


def test_skeleton_respects_work_schedule(tmp_path):
    sim, a = _agent_sim(tmp_path, "skel")
    cfg = DP.cfg_of(sim)
    a.work_start_min, a.work_end_min, a.part_time = 9 * 60, 17 * 60, None
    rows = DP.skeleton(sim, a, cfg)
    assert rows[0]["act"] == "work" and rows[0]["priority"] == "must"
    assert rows[0]["start"] == 9 * 60 and rows[0]["end"] == 17 * 60
    a.work_start_min, a.work_end_min = -1, 0
    rows = DP.skeleton(sim, a, cfg)
    assert all(r["act"] != "work" for r in rows) and len(rows) >= 3


# --------------------------------------------------------------------------- #
# (F) 場所解決(検収基準 6)
# --------------------------------------------------------------------------- #
def test_resolve_place_is_deterministic_habit_then_distance(tmp_path):
    """習慣(訪問回数)が最優先。同数なら近い順、さらに node 昇順で全順序。"""
    sim, a = _agent_sim(tmp_path, "resolve")
    first = DP.resolve_place(sim, a, "food", 12 * 60, from_node=a.node)
    assert first is not None
    for _ in range(3):
        assert DP.resolve_place(sim, a, "food", 12 * 60, from_node=a.node) == first
    others = [p["node"] for p in sim.city.pois_by_cat("food")
              if p["node"] not in (first, a.node)]
    assert others, "テスト前提が崩れた(food POI が 1 か所しかない)"
    a.visits[others[0]] += 5                      # 「いつもの店」を作る
    assert DP.resolve_place(sim, a, "food", 12 * 60, from_node=a.node) == others[0], \
        "訪問履歴(習慣)が行き先解決に効いていない"


def test_resolve_place_education_falls_back_to_school_cat(tmp_path):
    """★第98是正: 本線地図では education 1 件 vs school 68 件(W2-4 実測)で学業ブロックが
    事実上解決不能だった。MAP_FALLBACK_CATS の追加照会で school 側の POI が候補に入る。"""
    sim, a = _agent_sim(tmp_path, "resolve_fb")
    real = sim.city.pois_by_cat
    food = list(real("food"))
    assert food, "テスト前提が崩れた(food POI が無い)"
    asked: list[str] = []

    def fake(cat):
        asked.append(cat)
        if cat == "education":
            return []                       # 地図側の education はほぼ空(実測の縮図)
        if cat == "school":
            return food                     # school 側には実在する(中身は既存 POI を流用)
        return real(cat)

    sim.city.pois_by_cat = fake
    got = DP.resolve_place(sim, a, "education", 12 * 60, from_node=a.node)
    assert got is not None, "fallback 照会が効いていない(学業ブロックが解決不能のまま)"
    assert asked[:2] == ["education", "school"]      # 自カテゴリ→fallback の順で照会
    assert got in {p["node"] for p in food}


def test_resolve_place_without_fallback_queries_only_its_own_cat(tmp_path):
    """fallback の無いカテゴリは従来と完全同値=照会は自カテゴリ 1 回のみ。"""
    sim, a = _agent_sim(tmp_path, "resolve_nofb")
    real = sim.city.pois_by_cat
    asked: list[str] = []

    def fake(cat):
        asked.append(cat)
        return real(cat)

    sim.city.pois_by_cat = fake
    before = DP.resolve_place(sim, a, "food", 12 * 60, from_node=a.node)
    assert asked == ["food"]                         # 余計な照会ゼロ
    assert "food" not in DP.MAP_FALLBACK_CATS        # 前提: fallback 表は education のみ
    assert set(DP.MAP_FALLBACK_CATS) == {"education"}
    assert before is not None


def test_resolve_place_specials(tmp_path):
    sim, a = _agent_sim(tmp_path, "resolve_sp")
    assert DP.resolve_place(sim, a, "home", 0) == a.home_node
    assert DP.resolve_place(sim, a, "street", 0) is None
    a.work_node, a.part_time = "", None
    assert DP.resolve_place(sim, a, "work", 0) is None
    a.work_node = sorted(sim.dests)[0]
    assert DP.resolve_place(sim, a, "work", 0) == a.work_node


# --------------------------------------------------------------------------- #
# (G) priority × flex の割り込み(検収基準 7)
# --------------------------------------------------------------------------- #
class _CountingLLM:
    """generate の本数を数えるだけのプロキシ(割り込みが LLM を呼ばないことの固定)。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        return self._inner.generate(prompt, **kw)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _plan_on(sim, a, blocks, day_min=0):
    """検証・修復を通さず計画を直接据える(割り込みだけを見るため)。"""
    a._dayplan = {"schema": DP.SCHEMA, "day": day_min // 1440, "version": 1,
                  "model": "mock", "mood": "", "carry": "", "src": "llm",
                  "blocks": blocks, "cont": []}
    return a._dayplan


def _b(start, end, priority, flex, place="food", act="meal", node="n"):
    return {"start": start, "end": end, "place": place, "act": act, "with": [],
            "purpose": DP.ACT_PURPOSE[act], "priority": priority, "flex": flex,
            "reason": "", "note": "", "node": node, "state": "todo", "slid": 0}


def test_could_droppable_is_dropped_without_any_llm_call(tmp_path):
    """時間が食われた could+droppable は **LLM を 1 本も呼ばず** ルールが削る。"""
    sim, a = _agent_sim(tmp_path, "drop_could")
    sim.llm = _CountingLLM(sim.llm)
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "could", "droppable")])
    DP._sweep(sim, a, plan, DP.cfg_of(sim), 100, 13 * 60)
    assert plan["blocks"][0]["state"] == "dropped"
    ev = _kind(sim, "plan_block_drop")
    assert len(ev) == 1 and ev[0].payload["reason"] == "grace"
    assert sim.llm.calls == 0, "割り込みで LLM が呼ばれた(無料でないといけない)"
    assert not _kind(sim, "plan_replan"), "could を削るのに再計画が鳴った"


def test_slideable_slides_then_is_dropped_at_cap(tmp_path):
    """slideable は後ろへずれ、累積ずらしが上限を超えたら削られる。"""
    sim, a = _agent_sim(tmp_path, "slide_cap")
    cfg = dict(DP.cfg_of(sim), max_slide_min=60)
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "should", "slideable")])
    DP._sweep(sim, a, plan, cfg, 100, 8 * 60 + 50)
    b = plan["blocks"][0]
    assert b["state"] == "todo" and b["start"] == 8 * 60 + 50 and b["slid"] == 50
    assert _kind(sim, "plan_slide")
    DP._sweep(sim, a, plan, cfg, 110, 10 * 60)
    assert plan["blocks"][0]["state"] == "dropped"
    assert _kind(sim, "plan_block_drop")[-1].payload["reason"] == "slide_cap"


def test_fixed_non_must_is_dropped_when_missed(tmp_path):
    sim, a = _agent_sim(tmp_path, "fixed_miss")
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "should", "fixed")])
    DP._sweep(sim, a, plan, DP.cfg_of(sim), 100, 13 * 60)
    assert plan["blocks"][0]["state"] == "dropped"
    assert _kind(sim, "plan_block_drop")[0].payload["reason"] == "fixed_past"


def test_replan_fires_only_when_must_is_threatened(tmp_path):
    """must が脅かされたときだけ再計画が鳴る(should/could だけなら 0 件)。"""
    sim, a = _agent_sim(tmp_path, "replan_gate")
    cfg = DP.cfg_of(sim)
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "should", "droppable"),
                             _b(8 * 60, 9 * 60, "could", "droppable")])
    DP._sweep(sim, a, plan, cfg, 100, 13 * 60)
    assert not _kind(sim, "plan_replan"), "must でないのに再計画が鳴った"
    # must(8:00-9:00 slideable)は「今」= 13:00 へずれる。13:30 の could はまだ猶予内なので
    # sweep には拾われないが、ずらした先の must 窓(13:00-14:00)と重なるので replan が空ける。
    plan2 = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "must", "slideable"),
                              _b(13 * 60 + 30, 14 * 60 + 30, "could", "droppable")])
    DP._sweep(sim, a, plan2, cfg, 101, 13 * 60)
    ev = _kind(sim, "plan_replan")
    assert len(ev) == 1 and ev[0].payload["n_must"] == 1
    assert ev[0].payload["freed"] == 1, "ずらした先で must と重なる could が空けられていない"
    assert plan2["version"] == 2, "plan_version が上がっていない"
    assert plan2["blocks"][0]["start"] == 13 * 60, "must が今へずらされていない"
    assert plan2["blocks"][1]["state"] == "dropped"


def test_replan_retires_unmovable_missed_must(tmp_path):
    """動かせず窓も過ぎた must は退役させる(でないと毎 step 再計画が鳴り続ける)。"""
    sim, a = _agent_sim(tmp_path, "replan_retire")
    cfg = DP.cfg_of(sim)
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "must", "fixed")])
    DP._sweep(sim, a, plan, cfg, 100, 13 * 60)
    assert plan["blocks"][0]["state"] == "dropped"
    assert _kind(sim, "plan_block_drop")[0].payload["reason"] == "missed"
    DP._sweep(sim, a, plan, cfg, 101, 13 * 60 + 10)
    assert len(_kind(sim, "plan_replan")) == 1, "退役後も再計画が鳴り続けている"


def test_replan_is_idempotent_within_one_step(tmp_path):
    sim, a = _agent_sim(tmp_path, "replan_idem")
    cfg = DP.cfg_of(sim)
    plan = _plan_on(sim, a, [_b(8 * 60, 9 * 60, "must", "slideable")])
    for _ in range(3):
        DP._sweep(sim, a, plan, cfg, 100, 13 * 60)
    assert len(_kind(sim, "plan_replan")) == 1


def test_plan_exception_only_touches_cogq_when_fire_is_on(tmp_path):
    """plan_exception は cognition.fire OFF(既定)で完全 no-op、ON でだけ内部発火を前倒す。"""
    off, a_off = _agent_sim(tmp_path, "pe_off")
    DP.note_plan_exception(off, a_off, 8 * 60)
    assert getattr(off, "cogq", None) is None or not off.cogq.pending
    assert not hasattr(a_off, "_plan_exception")

    on, a_on = _agent_sim(tmp_path, "pe_on", **{"cognition.fire.enabled": "true"})
    on.cogq.schedule(int(a_on.id), 9 * 60, "periodic")
    DP.note_plan_exception(on, a_on, 8 * 60)
    rec = on.cogq.pending[int(a_on.id)]
    assert rec["at"] == 8 * 60 and rec["reason"] == "internal", rec
    assert a_on._plan_exception == 8 * 60


# --------------------------------------------------------------------------- #
# (H) ON ラン(検収基準 8)
# --------------------------------------------------------------------------- #
def test_on_run_emits_events_and_l2_matches_state(tmp_path):
    sim = _sim(tmp_path, "on_run", n_steps=288, **ON)
    summary = sim.run()
    assert _kind(sim, "plan_created"), "ON なのに計画が 1 件も立っていない"
    assert _kind(sim, "plan_block_start"), "ブロックが 1 つも実行されていない"
    st = sim._dayplan_state
    assert st["plans"] == len(_kind(sim, "plan_created"))
    assert st["exec"] == len(_kind(sim, "plan_block_start"))
    assert st["drop"] == len(_kind(sim, "plan_block_drop"))
    assert st["slide"] == len(_kind(sim, "plan_slide"))
    assert st["replan"] == len(_kind(sim, "plan_replan"))
    assert st["repair"] == len(_kind(sim, "plan_repair"))
    assert st["fallback"] == len(_kind(sim, "plan_fallback"))
    rows = pq.read_table(tmp_path / "on_run" / "l2_metrics.parquet").to_pylist()
    last = rows[-1]
    assert last["plan_exec_total"] == float(st["exec"])
    assert last["plan_repair_total"] == float(st["repair"])
    assert last["plan_replan_total"] == float(st["replan"])
    assert last["plan_blocks_mean"] == round(st["blocks"] / st["plans"], 6)
    assert summary["day_plan"]["plans"] == st["plans"]
    # 旧 day_plan イベント(自由文の計画)は ON では出ない = 経路が 1 本になっている
    assert not _kind(sim, "day_plan")


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "det_a", n_steps=288, **ON)
    a.run()
    b = _sim(tmp_path, "det_b", n_steps=288, **ON)
    b.run()
    assert _l1(a) == _l1(b), "day_plan ON の決定論が崩れている"


class _FixedLLM:
    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """day_plan ON のまま compute_matched 下で k=free / k=off の呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n_steps=110,
                   **{**ON, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("dpk_free", "free")
    off = _run("dpk_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"day_plan の呼数が k に依存(R1 違反): {free.llm.calls} vs {off.llm.calls}"


def test_on_adds_no_extra_llm_call_site(tmp_path):
    """ON でも朝の計画は **1 人 1 回のまま**(再試行なし・修復とフォールバックは全て決定論)。

    ★正直な限界: 総呼数そのものは ON/OFF で一致しない。day_plan は行き先を変える=
      co-location が変わる=遭遇由来の発火が変わるという **間接経路**があるからで、
      これは commerce / health / disaster と同型の既知の性質(registry の affects_k の
      注記どおり「呼び出し点を足す/減らす」ことだけを本テストは固定する)。
    """
    on = _sim(tmp_path, "nx_on", n_steps=288, **ON)
    on.run()
    purposes = [r["purpose"] for r in on.logger.llm_calls]
    n_plan = purposes.count("plan")
    assert n_plan == len(_kind(on, "plan_created")) > 0, \
        f"朝の計画呼が 1 人 1 回でない(再試行が混ざった): {n_plan}"
    assert "plan_retry" not in purposes, "day_plan 経路が再試行を撃っている"

    off = _sim(tmp_path, "nx_off", n_steps=288)
    off.run()
    assert set(purposes) == set(r["purpose"] for r in off.logger.llm_calls), \
        "day_plan ON で LLM 呼び出し点の種類が変わった"


def test_resume_matches_straight(tmp_path):
    """day_plan ON の resume==straight(_dayplan_state の checkpoint 中央管理)。

    分割点(60)より前に計画が立ち終わっている必要があるので、00:00 開始 + natural_start で
    「初日の朝に住民がまとめて起きて計画を立てる」配置にする(第79 test_start_tod と同じ手)。
    """
    ov = {**ON, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 60, 120
    straight_dir = tmp_path / "dp_straight"
    straight = Simulation(_cfg("dp_straight", total, 15, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "plan_created"), "テスト前提が崩れた(計画が立っていない)"

    d = tmp_path / "dp_resumed"
    sim1 = Simulation(_cfg("dp_resumed", split, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    assert _kind(sim1, "plan_created"), "分割点より前に計画が立っていない(検証にならない)"
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("dp_resumed", total, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(day_plan resume)"
    # 直接検証: _dayplan_state と計画本体が round-trip で復元される(空回り防止)
    sim3 = Simulation(_cfg("dp_inspect", split, 15, **ov,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "dp_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._dayplan_state == sim1._dayplan_state is not None
    assert sim3._dayplan_state["plans"] > 0
    assert [getattr(a, "_dayplan", None) for a in sim3.agents] == \
           [getattr(a, "_dayplan", None) for a in sim1.agents]


# --------------------------------------------------------------------------- #
# (I) no-fingerprint(検収基準 9)
# --------------------------------------------------------------------------- #
# (a) 本 module の機構語 = **ラン中のどのプロンプトにも**出てはならない
_FORBIDDEN_EVERYWHERE = (
    "day_plan", "plan_version", "plan_exception", "plan_created", "plan_repair",
    "contingency", "フォールバック", "修復", "ルールエンジン", "実行不能",
    "priority×flex",
)
# (b) 計画プロンプト(day_plan の節が載る呼び)に出てはならない語。発火機構・実験条件・因子名。
#     ★他の呼び(発話・内省)は既存機構が「驚き:」等を正当に載せるので、この検査は
#      **計画プロンプトだけ**に適用する(第82 test_watch と同じ切り分けの流儀)。
_FORBIDDEN_IN_PLAN_PROMPT = _FORBIDDEN_EVERYWHERE + (
    "periodic", "salience", "cog_fire", "認知イベント", "発火", "驚き", "閾値",
    "予測誤差", "スキーマ", "検証", "flat_traits", "experiment", "ablate",
    "mock", "シミュレーション", "アブレーション",
    "efficacy", "grievance", "ownership", "nfc", "risk_tolerance", "internal_locus",
)


def test_no_mechanism_or_factor_vocabulary_reaches_the_prompt(tmp_path):
    sim = _sim(tmp_path, "dp_fp", n_steps=144, **ON)
    seen = _spy(sim)
    sim.run()
    assert seen
    blob = "\n".join(seen)
    assert DP.PROMPT_MARK in blob, "ON なのに計画の書式がプロンプトに載っていない"
    for token in _FORBIDDEN_EVERYWHERE:
        assert token not in blob, f"どのプロンプトにも出てはいけない語が漏れた: {token}"
    plans = [p for p in seen if DP.PROMPT_MARK in p]
    assert plans
    plan_blob = "\n".join(plans)
    for token in _FORBIDDEN_IN_PLAN_PROMPT:
        assert token not in plan_blob, f"計画プロンプトに漏れている: {token}"


def test_schema_prompt_is_neutral_and_puts_reason_first():
    """理由欄が先頭(自己回帰の生成順)であることと、中立語彙であることを固定する。"""
    text = DP.schema_prompt(_C)
    body = text[text.index("各区切りの項目"):]
    assert body.index("reason:") < body.index("start /"), \
        "reason が先頭に置かれていない(構造化出力の推論品質を落とす並び)"
    for token in _FORBIDDEN_IN_PLAN_PROMPT:
        assert token not in text, token
    for member in DP.PRIORITIES + DP.FLEXES:
        assert member in text, f"列挙値 {member} が指示に出ていない"


# --------------------------------------------------------------------------- #
# (J) 24 step ON 実測(検収の数値報告)
# --------------------------------------------------------------------------- #
def test_24step_on_smoke_reports_rates(tmp_path):
    """24 step ON: 計画が立ち、ブロックが実行され、修復率が算出できる(数値の健全性のみ固定)。

    05:00 開始 + natural_start = 「起床が 24 step の窓に入る」配置(既定 07:00 開始では
    住民は起きたところから始まるので、初日の朝計画が 24 step 内に来ない)。
    """
    sim = _sim(tmp_path, "dp24", n_steps=24, n_agents=40, **ON,
               **{"run.start_tod": "05:00", "run.natural_start": "true"})
    summary = sim.run()
    dp = summary["day_plan"]
    assert dp["plans"] > 0 and dp["blocks"] >= dp["plans"]
    exec_rate = dp["executed"] / max(1, dp["blocks"])
    row = dp["by_model"]["mock"]
    assert 0.0 <= exec_rate <= 1.0
    assert 0.0 <= row["repair_rate"] <= 1.0 and 0.0 <= row["fallback_rate"] <= 1.0
    assert row["conflict_repair_rate"] <= row["repair_rate"], \
        "矛盾修復率が全修復率を超えている(丸めだけの計画まで数えている)"
    assert set(row["repair_ops"]) <= set(DP.REPAIR_OPS)
    assert set(row["fallback_kinds"]) <= set(DP.FALLBACK_KINDS)
    # ブロック数はスキーマの範囲へ収まっている(clip が効いている)
    for e in _kind(sim, "plan_created"):
        assert 1 <= e.payload["n"] <= DP.cfg_of(sim)["max_blocks"]
        assert e.payload["n_cont"] <= DP.cfg_of(sim)["max_conting"]


def _j(blocks, cont=None):
    """テスト用: 生 dict(json.loads 済み相当)を作る。"""
    return {"action": "plan", "mood": "", "carry": "",
            "blocks": blocks, "if_then": cont or []}


def test_note_resolved_fires_on_clean_plan_not_on_fallback(tmp_path, monkeypatch):
    """第87 脱出条件(1)の回帰固定(IF-B 検収で発見した実バグ)。

    fallback の「後退なし」値は ""(apply 内の初期値)で None は入らないため、
    `if fallback is None:` では engaged.note_resolved が一度も呼ばれなかった。
    真偽値判定(`if not fallback:`)で「検証を通る計画=解消」が発火することと、
    フォールバック(応急処置)では発火しないことの両側を固定する。
    """
    from society.cognition import engaged as EG
    calls = []
    monkeypatch.setattr(EG, "note_resolved", lambda sim, agent: calls.append(agent.id))
    sim, a = _agent_sim(tmp_path, "resolve_fire")
    good = _resp([_blk("08:00", "09:00", place="food", act="meal"),
                  _blk("18:00", "19:00", place="home", act="home")])
    DP.apply(sim, a, 0, 8 * 60, good, None)
    assert calls == [a.id], "検証を通った計画で note_resolved が発火していない"
    DP.apply(sim, a, 200, 1440 + 8 * 60, "{壊れた", None)
    assert calls == [a.id], "フォールバックは解消ではないのに note_resolved が発火した"
