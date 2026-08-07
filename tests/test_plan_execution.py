"""朝の計画と実行動の突合レイヤ(W2-4)の自己検定 — `scripts/analyze_plan_execution.py`。

対象は**解析側だけ**(`src/` / `conf/` はゼロタッチ・凍結 SPEC_FILES にも含まれない)。

守るもの(検収基準の順)
  (1) 対応表を写していない: 場所語彙は `day_plan.PLACE_CATS` の import のみで構成され、
      解析側に地図カテゴリ名の文字列リテラルが 1 つも無い(AST で固定)
  (2) `resolve_place` の特別分岐(home / work / street)が 3 つのままであること(AST)
  (3) 合成 L1 で 3 段の遵守率が手計算と一致する(PROV 経路)
  (4) `llm_role` を落とした同じ L1 で **経路(b) へ自動フォールバック**し、
      遵守率は完全に同値になる(経路が変わっても測定値は変わらない)
  (5) 台帳再生: slide / drop / replan / contingency(添字による差し戻し)
  (6) 未解決は当てにいかず unknown に落ちる(street 委譲・移動イベント無し・地図外)
  (7) day_plan OFF ラン = 遵守率は **0.0 ではなく null**
  (8) 空ラン・欠損ランで落ちない
  (9) 語彙被覆(地図 × PLACE_CATS)= 台帳の「266 中 5」を置き換える測り方
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import analyze_plan_execution as APE                 # noqa: E402

from society.cognition import day_plan as DP         # noqa: E402


# =========================================================================== #
# 共通の合成材料
# =========================================================================== #
def _ev(seq, step, sim_min, agent, kind, payload, call_id=None):
    return {"seq": seq, "step": step, "sim_min": sim_min, "agent_id": agent,
            "kind": kind, "payload": dict(payload), "llm_call_id": call_id}


def _mapidx():
    """最小の地図索引(nF=食/nS=店/nX=夜/nH=自宅/nW=職場建物内)。"""
    return {"path": "synthetic",
            "nodes": {"nF", "nS", "nX", "nH", "nW", "nP"},
            "node_cats": {"nF": {"food"}, "nS": {"shop"}, "nX": {"nightlife"}},
            "cat_counts": {"food": 1, "shop": 1, "nightlife": 1},
            "building_nodes": {"bW": {"nW"}},
            "n_pois": 3}


def _agents():
    return {1: {"home": "nH", "work_building": "bW", "work_nodes": {"nW"}}}


def _blocks():
    return [
        {"start": 480, "end": 540, "place": "food", "act": "meal",
         "aim": "sustenance", "priority": "should", "flex": "slideable"},
        {"start": 600, "end": 660, "place": "shop", "act": "shop",
         "aim": "errand", "priority": "should", "flex": "slideable"},
        {"start": 720, "end": 780, "place": "home", "act": "home",
         "aim": "rest", "priority": "must", "flex": "fixed"},
        {"start": 840, "end": 900, "place": "street", "act": "walk",
         "aim": "health", "priority": "could", "flex": "droppable"},
        {"start": 960, "end": 1020, "place": "work", "act": "work",
         "aim": "livelihood", "priority": "must", "flex": "fixed"},
    ]


def _start_payload(b, start=None, slid=0):
    return {"act": b["act"], "place": b["place"], "aim": b["aim"],
            "priority": b["priority"], "flex": b["flex"],
            "start": int(start if start is not None else b["start"]),
            "slid": int(slid), "version": 1}


def _scenario(with_role=True):
    """1 体 1 日・5 ブロック。手計算の正解つき(下の test が固定する)。"""
    bs = _blocks()
    role = {"llm_role": "plan"} if with_role else {}
    cid = "c1"
    rs_cid = cid if with_role else None
    ev = [
        _ev(0, 40, 400, 1, "plan_created",
            {"n": 5, "version": 1, "src": "llm", "model": "mock", "n_cont": 0,
             "blocks": bs}, cid),
        # ---- ブロック0: 予定どおり food へ ------------------------------------
        _ev(1, 48, 480, 1, "plan_block_start", _start_payload(bs[0])),
        _ev(2, 48, 480, 1, "route_start", {"dest": "nF", **role}, rs_cid),
        # ---- ブロック1: 日中に 40 分ずらされ、夜の店へ行った ------------------
        _ev(3, 64, 640, 1, "plan_slide",
            {"act": "shop", "place": "shop", "start": 640, "slid": 40,
             "priority": "should"}),
        _ev(4, 64, 640, 1, "plan_block_start", _start_payload(bs[1], 640, 40)),
        _ev(5, 64, 640, 1, "route_start", {"dest": "nX", **role}, rs_cid),
        # ---- ブロック2: 自宅へ(予定どおり)----------------------------------
        _ev(6, 72, 720, 1, "plan_block_start", _start_payload(bs[2])),
        _ev(7, 72, 720, 1, "route_start", {"dest": "nH", **role}, rs_cid),
        # ---- ブロック3: street = 行き先を習慣ポリシーへ委譲(移動イベント無し)--
        _ev(8, 84, 840, 1, "plan_block_start", _start_payload(bs[3])),
        # ---- ブロック4: 実行されず削除 ---------------------------------------
        _ev(9, 100, 1000, 1, "plan_block_drop",
            {"act": "work", "place": "work", "start": 960, "priority": "must",
             "flex": "fixed", "reason": "missed"}),
    ]
    return ev


def _summarize(events, mapidx=None, agents=None, cfg=None):
    return APE.summarize("runs/_synthetic", events, cfg or DP.build_cfg(None),
                         mapidx if mapidx is not None else _mapidx(),
                         agents if agents is not None else _agents())


# =========================================================================== #
# (1)(2) 対応表を写していない = 正典は day_plan.py 側
# =========================================================================== #
def test_place_vocabulary_is_a_partition_of_day_plan_place_cats():
    """15 種は 3 分類の直和であり、**import した DP.PLACE_CATS が唯一の源**。"""
    parts = (set(APE.AGENT_RELATIVE_PLACES) | set(APE.DELEGATED_PLACES)
             | set(APE.MAP_PLACES))
    assert parts == set(DP.PLACE_CATS)
    assert not (set(APE.AGENT_RELATIVE_PLACES) & set(APE.DELEGATED_PLACES))
    assert not (set(APE.MAP_PLACES) & set(APE.AGENT_RELATIVE_PLACES))
    assert not (set(APE.MAP_PLACES) & set(APE.DELEGATED_PLACES))
    # 順序も PLACE_CATS の並びから導出されている(勝手な並べ替えをしない)
    assert list(APE.MAP_PLACES) == [p for p in DP.PLACE_CATS if p in APE.MAP_PLACES]


def test_analyzer_source_has_no_copied_place_category_literals():
    """★地図カテゴリ名の文字列リテラルが解析側に 1 つも無い(= 語彙の二重定義なし)。

    home / work / street は `resolve_place` の**特別分岐**に 1 対 1 対応する
    3 語だけを許す(下の AST テストがその 3 つを day_plan 側で固定する)。
    """
    src = Path(APE.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lits = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    leaked = sorted(lits & set(APE.MAP_PLACES))
    assert leaked == [], f"場所カテゴリ名がベタ書きされている: {leaked}"
    assert lits & set(APE.AGENT_RELATIVE_PLACES) == set(APE.AGENT_RELATIVE_PLACES)


def test_resolve_place_special_branches_are_exactly_three():
    """`day_plan.resolve_place` の特別分岐が 3 つのままであること(乖離検知)。

    分岐が増えた/名前が変わった瞬間にここが落ちる = 解析側の 3 分類を直せ、の合図。
    """
    src = Path(DP.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_place")
    special = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != "place":
            continue
        for op, cmp_ in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(cmp_, ast.Constant):
                special.add(cmp_.value)
    assert special == set(APE.AGENT_RELATIVE_PLACES) | set(APE.DELEGATED_PLACES)
    # 残りは必ず地図の POI カテゴリ索引を引く(= 解析側の逆引きと同じ欄)
    assert "pois_by_cat" in ast.dump(fn)


def test_analyzer_reuses_day_plan_config_normalizer():
    """conf の正準化は DP.build_cfg をそのまま呼ぶ(既定値の写しを持たない)。"""
    cfg = APE.load_cfg("runs/does_not_exist")
    assert cfg == DP.build_cfg(None)
    assert cfg["round_min"] == DP.DEFAULTS["round_min"]


# =========================================================================== #
# (3) PROV 経路での 3 段遵守率(手計算)
# =========================================================================== #
def test_prov_path_three_stage_rates_match_hand_computation():
    res = _summarize(_scenario(with_role=True))
    assert res["day_plan"]["present"] is True
    at = res["attribution"]
    assert at["path"] == "prov"
    assert at["by_source"] == {"prov": 3, "step_join": 0, "none": 1, "detour": 0}
    assert at["n_route_start_role_plan"] == 3

    d = res["compliance"]["denominator"]
    assert (d["blocks_planned"], d["blocks_executed"], d["blocks_dropped"],
            d["blocks_untouched"]) == (5, 4, 1, 0)

    s = res["compliance"]["stage"]
    # ① 時間帯: ブロック1 だけ日中にずらされた → 3/4
    assert (s["time"]["n"], s["time"]["ok"]) == (4, 3)
    assert s["time"]["rate"] == pytest.approx(0.75)
    # ② 活動種別: 4/4(contingency 無し)
    assert (s["act"]["n"], s["act"]["ok"], s["act"]["rate"]) == (4, 4, 1.0)
    # ③ 場所解決: 判定できたのは food/shop/home の 3 件、一致は food と home
    assert (s["place"]["n"], s["place"]["ok"]) == (3, 2)
    assert s["place"]["rate"] == pytest.approx(0.6667, abs=1e-4)
    # 未解決 = street 委譲の 1 件(実行 4 件の 25%)
    assert s["place"]["unresolved"] == 1
    assert s["place"]["by_reason"] == {"street_delegated": 1}
    assert s["place"]["unresolved_rate"] == pytest.approx(0.25)
    # 3 段すべて = ブロック0 と 2
    assert (s["all_three"]["n"], s["all_three"]["ok"]) == (3, 2)


def test_by_place_breakdown_is_keyed_on_the_morning_declaration():
    res = _summarize(_scenario())
    bp = res["compliance"]["by_place"]
    assert set(bp) == {"food", "shop", "home", "street", "work"}
    assert bp["food"] == {"planned": 1, "executed": 1, "dropped": 0, "untouched": 0,
                          "time_ok": 1, "act_ok": 1, "place_decided": 1,
                          "place_ok": 1, "unresolved": 0}
    assert bp["shop"]["place_decided"] == 1 and bp["shop"]["place_ok"] == 0
    assert bp["street"]["unresolved"] == 1 and bp["street"]["place_decided"] == 0
    assert bp["work"] == {"planned": 1, "executed": 0, "dropped": 1, "untouched": 0,
                          "time_ok": 0, "act_ok": 0, "place_decided": 0,
                          "place_ok": 0, "unresolved": 0}
    assert res["compliance"]["by_drop_reason"] == {"missed": 1}


def test_mismatch_sample_carries_the_node_and_its_real_categories():
    res = _summarize(_scenario())
    samples = res["compliance"]["mismatch_samples"]
    assert len(samples) == 1
    s = samples[0]
    assert s["planned_place"] == "shop" and s["node"] == "nX"
    assert s["node_cats"] == ["nightlife"]          # 実際に行った場所の種類を出す
    assert s["node_src"] == "prov"


# =========================================================================== #
# (4) llm_link 無しラン = 経路(b)へ自動フォールバックし、測定値は同値
# =========================================================================== #
def test_step_join_fallback_when_llm_link_is_off():
    on = _summarize(_scenario(with_role=True))
    off = _summarize(_scenario(with_role=False))
    assert off["attribution"]["path"] == "step_join"
    assert off["attribution"]["n_route_start_role_plan"] == 0
    assert off["attribution"]["by_source"] == {"prov": 0, "step_join": 3,
                                               "none": 1, "detour": 0}
    # ★経路が変わっても遵守率は 1 桁も動かない(帰属の証拠だけが違う)
    assert off["compliance"]["stage"] == on["compliance"]["stage"]
    assert off["compliance"]["by_place"] == on["compliance"]["by_place"]


def test_prov_join_ignores_events_of_another_call_id():
    """PROV 経路は **その朝の呼**の刻印しか拾わない(別の呼の移動を混ぜない)。"""
    ev = _scenario(with_role=True)
    for e in ev:
        if e["kind"] == "route_start":
            e["llm_call_id"] = "OTHER"              # 別の思考由来の移動
    res = _summarize(ev)
    # role は "plan" のままだが call_id が違う → PROV では拾えず同 step join へ落ちる
    assert res["attribution"]["by_source"]["prov"] == 0
    assert res["attribution"]["by_source"]["step_join"] == 3


# =========================================================================== #
# (5) 台帳再生: slide / replan / contingency
# =========================================================================== #
def test_slide_is_attributed_and_updates_the_current_window():
    days, stats = APE.build_ledger(_scenario(), DP.build_cfg(None))
    assert len(days) == 1
    b1 = days[0]["blocks"][1]
    assert b1["n_slide"] == 1 and b1["slid_seen"] == 40
    assert (b1["cur"]["start"], b1["cur"]["end"]) == (640, 700)   # 継続 60 分は保つ
    assert b1["plan"]["start"] == 600                             # 朝の宣言は不変
    assert b1["exec"]["exact_match"] is True                      # ずらし後の窓で一致
    assert sum(stats["unmatched"].values()) == 0
    assert sum(stats["ambiguous"].values()) == 0


def test_replan_moves_must_blocks_without_a_per_block_event():
    """`plan_replan` はブロック単位のイベントを出さない → cfg から決定論再現する。"""
    bs = [{"start": 480, "end": 540, "place": "food", "act": "meal",
           "aim": "sustenance", "priority": "must", "flex": "slideable"}]
    ev = [_ev(0, 40, 400, 1, "plan_created",
              {"n": 1, "version": 1, "src": "llm", "blocks": bs}, "c1"),
          _ev(1, 60, 600, 1, "plan_replan",
              {"version": 2, "n_must": 1, "freed": 0, "retired": 0, "at": 600}),
          _ev(2, 60, 600, 1, "plan_block_start", _start_payload(bs[0], 600, 120)),
          _ev(3, 60, 600, 1, "route_start", {"dest": "nF", "llm_role": "plan"}, "c1")]
    days, stats = APE.build_ledger(ev, DP.build_cfg(None))
    b = days[0]["blocks"][0]
    assert (b["cur"]["start"], b["cur"]["end"]) == (600, 660)
    assert b["exec"] is not None and b["exec"]["exact_match"] is True
    assert stats["replan_mismatch"] == 0
    res = _summarize(ev)
    s = res["compliance"]["stage"]
    assert (s["time"]["ok"], s["time"]["n"]) == (0, 1)   # 時間帯は守られていない
    assert (s["place"]["ok"], s["place"]["n"]) == (1, 1)  # 場所は守られた
    assert s["all_three"]["ok"] == 0


def test_contingency_block_index_overrides_an_ambiguous_drop_attribution():
    """★`plan_cont_fire` だけがブロック添字を持つ = 正典。誤帰属は差し戻す。"""
    same = {"start": 600, "end": 660, "place": "food", "act": "meal",
            "aim": "sustenance", "priority": "should", "flex": "slideable"}
    ev = [_ev(0, 40, 400, 1, "plan_created",
              {"n": 2, "src": "llm", "blocks": [dict(same), dict(same)]}, "c1"),
          _ev(1, 60, 600, 1, "plan_block_drop",
              {"act": "meal", "place": "food", "start": 600, "priority": "should",
               "flex": "slideable", "reason": "cont_skip"}),
          _ev(2, 60, 600, 1, "plan_cont_fire",
              {"cond": "closed", "then": "skip", "block": 1, "act": "meal",
               "place": "food", "start": 600, "applied": True})]
    days, stats = APE.build_ledger(ev, DP.build_cfg(None))
    blocks = days[0]["blocks"]
    assert blocks[0]["state"] == "todo"             # 誤帰属は差し戻された
    assert blocks[1]["state"] == "dropped"
    assert blocks[1]["cont"]["then"] == "skip"
    assert stats["cont_reattributed"] == 1
    assert stats["ambiguous"]["drop"] == 1          # 多義だったことは隠さず数える


def test_contingency_go_home_rewrites_act_place_and_aim_from_the_canonical_table():
    """go_home は act/place/aim を変える。aim は DP.ACT_PURPOSE から引き直す。"""
    bs = [{"start": 600, "end": 660, "place": "shop", "act": "shop",
           "aim": "errand", "priority": "should", "flex": "slideable"}]
    ev = [_ev(0, 40, 400, 1, "plan_created", {"n": 1, "src": "llm", "blocks": bs},
              "c1"),
          _ev(1, 60, 600, 1, "plan_cont_fire",
              {"cond": "rain", "then": "go_home", "block": 0, "act": "home",
               "place": "home", "start": 600, "applied": True}),
          _ev(2, 60, 600, 1, "plan_block_start",
              {"act": "home", "place": "home", "aim": DP.ACT_PURPOSE["home"],
               "priority": "should", "flex": "slideable", "start": 600,
               "slid": 0, "version": 1}),
          _ev(3, 60, 600, 1, "route_start", {"dest": "nH", "llm_role": "plan"}, "c1")]
    days, stats = APE.build_ledger(ev, DP.build_cfg(None))
    b = days[0]["blocks"][0]
    assert b["cur"]["aim"] == DP.ACT_PURPOSE["home"]
    assert b["exec"] is not None and sum(stats["unmatched"].values()) == 0
    # 朝の宣言(shop)から見れば 場所も活動も逸脱 = 遵守していない
    res = _summarize(ev)
    s = res["compliance"]["stage"]
    assert s["act"]["ok"] == 0
    assert s["place_intent"]["ok"] == 0
    assert s["place"]["n"] == 1 and s["place"]["ok"] == 0   # nH は shop ではない
    assert res["compliance"]["denominator"]["blocks_with_contingency"] == 1


def test_events_without_a_plan_are_counted_as_orphans_not_silently_dropped():
    ev = [_ev(0, 48, 480, 1, "plan_block_start",
              _start_payload(_blocks()[0]))]        # plan_created が無い
    _days, stats = APE.build_ledger(ev, DP.build_cfg(None))
    assert stats["orphan_events"] == 1


# =========================================================================== #
# (6) 未解決は当てにいかない
# =========================================================================== #
def test_unresolved_reasons_are_reported_separately_and_never_guessed():
    bs = [{"start": 480, "end": 540, "place": "cinema", "act": "leisure",
           "aim": "enjoyment", "priority": "should", "flex": "slideable"},
          {"start": 600, "end": 660, "place": "food", "act": "meal",
           "aim": "sustenance", "priority": "should", "flex": "slideable"},
          {"start": 720, "end": 780, "place": "food", "act": "meal",
           "aim": "sustenance", "priority": "should", "flex": "slideable"}]
    ev = [_ev(0, 40, 400, 1, "plan_created", {"n": 3, "src": "llm", "blocks": bs},
              "c1"),
          # cinema は合成地図に 1 件も無い = 原理的に満たされえない
          _ev(1, 48, 480, 1, "plan_block_start", _start_payload(bs[0])),
          _ev(2, 48, 480, 1, "route_start", {"dest": "nF", "llm_role": "plan"}, "c1"),
          # 同 step に移動イベントが無い(既に目的地に居た等)
          _ev(3, 60, 600, 1, "plan_block_start", _start_payload(bs[1])),
          # 地図に無いノードへ着いた
          _ev(4, 72, 720, 1, "plan_block_start", _start_payload(bs[2])),
          _ev(5, 72, 720, 1, "route_start", {"dest": "nZZZ", "llm_role": "plan"},
              "c1")]
    res = _summarize(ev)
    s = res["compliance"]["stage"]
    assert s["place"]["n"] == 0 and s["place"]["rate"] is None   # 判定できた件 0
    assert s["place"]["by_reason"] == {"category_absent_on_map": 1,
                                       "no_move_event": 1, "node_off_map": 1}
    assert s["place"]["unresolved"] == 3
    assert s["all_three"]["n"] == 0 and s["all_three"]["rate"] is None


def test_home_and_work_fall_back_to_unknown_without_agents_json():
    res = _summarize(_scenario(), agents={})
    s = res["compliance"]["stage"]
    assert s["place"]["by_reason"]["home_unknown"] == 1
    assert s["place"]["n"] == 2                     # food と shop だけが判定可


def test_work_match_uses_the_building_approximation():
    st, detail = APE.place_status("work", "nW", 1, _mapidx(), _agents())
    assert (st, detail) == ("match", "approx_building")
    st2, _d = APE.place_status("work", "nF", 1, _mapidx(), _agents())
    assert st2 == "mismatch"


def test_detour_is_recorded_as_a_deviation_with_the_planned_node_kept():
    bs = [{"start": 480, "end": 540, "place": "food", "act": "meal",
           "aim": "sustenance", "priority": "should", "flex": "slideable"}]
    ev = [_ev(0, 40, 400, 1, "plan_created", {"n": 1, "src": "llm", "blocks": bs},
              "c1"),
          _ev(1, 48, 480, 1, "plan_block_start", _start_payload(bs[0])),
          _ev(2, 48, 480, 1, "detour", {"act": "meal", "to": "nS",
                                        "from_dest": "nF"}),
          _ev(3, 48, 480, 1, "route_start", {"dest": "nS", "llm_role": "plan"},
              "c1")]
    days, _stats = APE.build_ledger(ev, DP.build_cfg(None))
    APE.attach_nodes(days, ev)
    ex = days[0]["blocks"][0]["exec"]
    assert ex["node"] == "nS" and ex["node_planned"] == "nF"
    res = _summarize(ev)
    assert res["compliance"]["denominator"]["detours"] == 1
    assert res["compliance"]["stage"]["place"]["ok"] == 0   # 実際に行った先で判定


# =========================================================================== #
# (7)(8) OFF ラン / 空ラン
# =========================================================================== #
def test_day_plan_off_run_reports_null_not_zero():
    """★測る対象が存在しないことを明示する(0.0 と偽らない)。"""
    ev = [_ev(0, 10, 100, 1, "route_start", {"dest": "nF"}, None)]
    res = _summarize(ev)
    assert res["day_plan"]["present"] is False
    assert res["day_plan"]["n_plan_created"] == 0
    assert "OFF" in res["day_plan"]["reason"]
    assert res["compliance"] is None and res["attribution"] is None
    assert res["ledger"] is None
    md = APE.render(res)
    assert "null" in md and "未測定" in md


def test_empty_run_does_not_crash_and_still_reports_vocabulary():
    res = _summarize([])
    assert res["day_plan"]["present"] is False
    assert res["vocabulary"]["resolvable"] == ["food", "nightlife", "shop"]
    assert APE.render(res)


def test_missing_map_and_missing_agents_are_not_fatal():
    res = APE.summarize("runs/_synthetic", _scenario(), DP.build_cfg(None), None, {})
    assert res["vocabulary"]["map"] is None
    s = res["compliance"]["stage"]
    assert s["place"]["by_reason"]["no_map"] == 2       # food / shop は地図が要る
    assert s["place"]["by_reason"]["home_unknown"] == 1
    assert APE.render(res)


def test_zero_denominator_rates_are_none_never_zero():
    assert APE._rate(0, 0) is None
    assert APE._rate(0, 3) == 0.0


def test_load_events_raises_a_clear_error_when_l1_is_absent(tmp_path):
    with pytest.raises(SystemExit):
        APE.load_events(str(tmp_path))


def test_load_events_preserves_file_order_and_filters_kinds(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [(3, 300, 1, "speak", "{}", None),
            (1, 100, 1, "plan_created", json.dumps({"n": 0, "blocks": []}), "c1"),
            (2, 200, 1, "route_start", json.dumps({"dest": "nF"}), "c1")]
    table = pa.table({
        "step": pa.array([r[0] for r in rows], pa.int32()),
        "sim_min": pa.array([r[1] for r in rows], pa.int32()),
        "agent_id": pa.array([r[2] for r in rows], pa.int32()),
        "kind": pa.array([r[3] for r in rows], pa.string()),
        "x": pa.array([0.0] * 3, pa.float32()),
        "y": pa.array([0.0] * 3, pa.float32()),
        "payload": pa.array([r[4] for r in rows], pa.string()),
        "rng_stream": pa.array([""] * 3, pa.string()),
        "llm_call_id": pa.array([r[5] for r in rows], pa.string())})
    pq.write_table(table, tmp_path / "l1_events.parquet")
    ev = APE.load_events(str(tmp_path))
    assert [e["kind"] for e in ev] == ["plan_created", "route_start"]  # 記録順のまま
    assert [e["seq"] for e in ev] == [1, 2]
    assert ev[0]["llm_call_id"] == "c1"


# =========================================================================== #
# (9) 語彙被覆 = 台帳の「266 中 5」を置き換える測り方
# =========================================================================== #
def test_place_coverage_counts_categories_not_node_names():
    cov = APE.place_coverage(_mapidx())
    assert cov["map_backed"] == list(APE.MAP_PLACES)
    assert cov["resolvable"] == ["food", "nightlife", "shop"]
    assert "cinema" in cov["unresolvable"] and "hall" in cov["unresolvable"]
    assert cov["coverage_rate"] == pytest.approx(3 / len(APE.MAP_PLACES), abs=1e-4)


def test_place_coverage_education_resolves_via_school_fallback():
    """★2026-08-07 是正後の真実: 地図の `school` は day_plan.MAP_FALLBACK_CATS で
    `education` の追加照会先になった=学業ブロックは解決可能・踏み潰し警告も解消。
    fallback の無い別名(theater→hall)は引き続き踏み潰しとして検知される。"""
    idx = _mapidx()
    idx["cat_counts"]["school"] = 68                # 地図には学校が 68 件ある
    idx["cat_counts"]["theater"] = 7                # hall は 0 件のまま=真の踏み潰し
    cov = APE.place_coverage(idx)
    assert "education" in cov["resolvable"]         # fallback(school)経由で解決可能
    assert cov["n_pois_effective"]["education"] == 68
    assert cov["n_pois_by_place"]["education"] == 0  # 生の件数は正直に 0 のまま
    assert cov["fallback_cats"].get("education") == ["school"]
    shadow = {s["alias"]: s for s in cov["alias_shadowed"]}
    assert "school" not in shadow                   # 受け皿ができたので解消
    assert "theater" in shadow
    assert shadow["theater"]["rewritten_to"] == "hall"
    assert shadow["theater"]["n_pois_target"] == 0
    assert "school" in cov["map_cats_not_in_plan_vocab"]


def test_place_status_matches_education_reached_via_school_poi():
    """resolve_place の fallback で選ばれた school ノードが mismatch にならない。"""
    idx = _mapidx()
    idx["nodes"].add("nSch")
    idx["node_cats"]["nSch"] = {"school"}
    idx["cat_counts"]["school"] = 1
    st, _ = APE.place_status("education", "nSch", 1, idx, _agents())
    assert st == "match"
    st2, det2 = APE.place_status("education", "nF", 1, idx, _agents())
    assert st2 == "mismatch"                        # food ノードは受理集合の外のまま
    idx2 = _mapidx()                                # school も education も 0 件の地図
    st3, det3 = APE.place_status("education", "nF", 1, idx2, _agents())
    assert (st3, det3) == ("unknown", "category_absent_on_map")


def test_place_coverage_on_the_real_shipped_map_if_present():
    """実地図(あれば)でも落ちず、被覆率が [0,1] に収まる。"""
    path = _ROOT / "data" / "shibuya_osm_wide_v7.json"
    if not path.exists():
        pytest.skip("実地図が無い環境")
    idx = APE.load_map(str(path))
    assert idx is not None and idx["n_pois"] > 0
    cov = APE.place_coverage(idx)
    assert 0.0 <= cov["coverage_rate"] <= 1.0
    assert set(cov["resolvable"]) | set(cov["unresolvable"]) == set(APE.MAP_PLACES)


def test_map_path_resolution_reads_world_map_from_the_run_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "world:\n  map: data/shibuya_osm.json\n", encoding="utf-8")
    got = APE.map_path_of(str(tmp_path))
    assert got is not None and got.endswith("shibuya_osm.json")


def test_render_is_pure_text_and_mentions_both_paths():
    md = APE.render(_summarize(_scenario()))
    assert "帰属経路" in md and "prov" in md
    assert "3 段のはしご" in md
    assert "place 別の内訳" in md
