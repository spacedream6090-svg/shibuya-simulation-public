"""SV2-B の解析器 7 本の自己検定(合成データ・既知の正解つき)。

対象(いずれも `scripts/` 新設・読み取り専用・`src/`/`conf/` ゼロタッチ):
  analyze_seed_variance.py        S-05  seed 間分散の分散分解(事前登録 §3-G)
  analyze_persona_consistency.py  S-06  個体レベル長期一貫性
  analyze_layers.py               S-12  伝播チャネル三層(事前登録 §1.5)
  analyze_org_form.py             S-09  組織形態(事前登録 §3-I)
  analyze_mas_failures.py         S-10  MAST 失敗様式の L1 判定可能サブセット
  analyze_ipf_fidelity.py         S-14  IPF 再現誤差(SRMSE / TAE / CE)
  analyze_stereotype.py           S-18  ステレオタイプ増幅(Monroe 2008 log-odds)

house style: `scripts/` を path 追加して import(tests/test_analyze_founders.py に倣う)。
検定の方針 = **既知の正解を持つ小さな合成データで、定義そのものを固定する**。
乱数を使う量(ブートストラップ)は seed 固定で**再現同値**を固定する。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import analyze_ipf_fidelity as aif                   # noqa: E402
import analyze_layers as al                          # noqa: E402
import analyze_mas_failures as amf                   # noqa: E402
import analyze_org_form as aof                       # noqa: E402
import analyze_persona_consistency as apc            # noqa: E402
import analyze_seed_variance as asv                  # noqa: E402
import analyze_stereotype as ast_                    # noqa: E402


def _ev(step, agent, kind, payload=None):
    """analyze_layers / analyze_mas_failures が読む形(agent キー)。"""
    return {"step": step, "agent": agent, "kind": kind, "payload": payload or {}}


def _ev2(step, agent, kind, payload=None):
    """analyze_org_form が読む形(agent_id キー = analyze_communities の流儀)。"""
    return {"step": step, "agent_id": agent, "kind": kind, "payload": payload or {}}


# =========================================================================== #
# S-05 analyze_seed_variance
# =========================================================================== #
def test_seed_variance_decompose_matches_hand_computation():
    """分解の定義(事前登録 §3-G.1)を手計算で固定する。"""
    by_cond = {"A": {1: 10.0, 2: 12.0, 3: 14.0},      # mean 12, var(ddof=1) = 4
               "B": {1: 20.0, 2: 22.0, 3: 24.0}}      # mean 22, var(ddof=1) = 4
    d = asv.decompose(by_cond)
    assert d["n_conditions"] == 2
    assert d["V_seed"] == pytest.approx(4.0)          # 条件内分散の平均
    assert d["V_condition"] == pytest.approx(50.0)    # Var([12, 22], ddof=1) = 50
    assert d["ratio"] == pytest.approx(12.5)
    assert d["share_condition"] + d["share_seed"] == pytest.approx(1.0)


def test_seed_variance_degenerate_zero_seed_variance_is_not_a_pass():
    """★V_seed == 0 を「条件差が圧倒的」と読ませない(専用ラベルにする)。"""
    by_cond = {"A": {1: 1.0, 2: 1.0, 3: 1.0}, "B": {1: 5.0, 2: 5.0, 3: 5.0}}
    d = asv.decompose(by_cond)
    assert d["V_seed"] == 0
    v = asv.verdict(d, asv.bootstrap_ratio_ci(by_cond, b=50))
    assert v["label"] == asv.LABEL_DEGENERATE
    assert v["claimable"] is False


def test_seed_variance_insufficient_seeds_is_not_claimable_even_if_g1_g2_hold():
    """G3(seed 本数 >= 3)不成立なら、G1/G2 が立っても主張しない。"""
    by_cond = {"A": {1: 10.0, 2: 10.5}, "B": {1: 30.0, 2: 30.5}}   # seed 2 本ずつ
    d = asv.decompose(by_cond)
    ci = asv.bootstrap_ratio_ci(by_cond, b=200)
    v = asv.verdict(d, ci)
    assert v["G1"] is True                       # 条件差は圧倒的に大きい
    assert v["label"] == asv.LABEL_INSUFFICIENT_SEEDS
    assert v["claimable"] is False


def test_seed_variance_single_condition_is_insufficient_conditions():
    d = asv.decompose({"A": {1: 1.0, 2: 2.0, 3: 3.0}})
    v = asv.verdict(d, {"lo": None, "hi": None})
    assert v["label"] == asv.LABEL_INSUFFICIENT_CONDITIONS
    assert d["V_condition"] is None


def test_seed_variance_claimable_when_g1_g2_g3_all_hold():
    by_cond = {"A": {1: 1.0, 2: 1.1, 3: 0.9, 4: 1.05},
               "B": {1: 9.0, 2: 9.1, 3: 8.9, 4: 9.05}}
    d = asv.decompose(by_cond)
    ci = asv.bootstrap_ratio_ci(by_cond, b=500)
    v = asv.verdict(d, ci)
    assert v == {"label": asv.LABEL_OK, "G1": True, "G2": True, "G3": True,
                 "claimable": True}


def test_seed_variance_bootstrap_is_deterministic_under_fixed_seed():
    by_cond = {"A": {1: 1.0, 2: 2.0, 3: 3.0}, "B": {1: 5.0, 2: 6.0, 3: 7.0}}
    a = asv.bootstrap_ratio_ci(by_cond, b=300, seed=0)
    b = asv.bootstrap_ratio_ci(by_cond, b=300, seed=0)
    c = asv.bootstrap_ratio_ci(by_cond, b=300, seed=1)
    assert a == b                                  # 同 seed = バイト同値
    assert a["n_valid"] > 0
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"]) or a["n_valid"] != c["n_valid"]


def test_seed_variance_spread_is_median_min_max():
    d = asv.decompose({"A": {1: 1.0}, "B": {1: 5.0}, "C": {1: 3.0}})
    s = asv.spread(d)
    assert (s["min"], s["median"], s["max"], s["range"]) == (1.0, 3.0, 5.0, 4.0)


# =========================================================================== #
# S-06 analyze_persona_consistency
# =========================================================================== #
def test_persona_consistency_tvd_definition():
    assert apc.tvd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert apc.tvd([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert apc.tvd([0.75, 0.25], [0.25, 0.75]) == pytest.approx(0.5)


def _profile_events():
    """個体 1/2 は毎日同じ配分(一貫)、個体 3 は日ごとに全く違う配分(非一貫)。"""
    spd = apc.STEPS_PER_DAY
    ev = []
    for day in range(4):
        base = day * spd
        for _ in range(4):
            ev.append({"step": base, "agent": 1, "kind": "speak"})
        for _ in range(4):
            ev.append({"step": base, "agent": 2, "kind": "arrive"})
        kind = "speak" if day % 2 == 0 else "arrive"
        for _ in range(4):
            ev.append({"step": base, "agent": 3, "kind": kind})
    return ev


def test_persona_consistency_within_lt_between_for_stable_agents():
    prof = apc.build_profiles(_profile_events())
    assert prof["agents"] == [1, 2, 3]
    assert len(prof["days"]) == 4
    within = apc.within_agent_tvd(prof)
    assert within[1] == pytest.approx(0.0)         # 毎日同じ = 個体内 TVD 0
    assert within[2] == pytest.approx(0.0)
    assert within[3] == pytest.approx(1.0)         # 毎日反転 = 個体内 TVD 1
    between = apc.between_agent_tvd(prof)
    assert all(v > 0 for v in between.values())    # 個体 1 と 2 は常に別配分


def test_persona_consistency_ratio_and_exchange_unit_is_day_pair(tmp_path):
    prof = apc.build_profiles(_profile_events())
    rows = apc.per_agent_signflip(prof, mc=200)
    # 交換単位は「日ペア」= D の行数が日ペア数(4 日 → 3 ペア)
    assert rows and all(r["n_day_pairs"] == 3 for r in rows)
    assert sorted(r["agent"] for r in rows) == [1, 2, 3]


def test_persona_consistency_absent_days_are_not_zero_vectors():
    """イベントが 1 件も無い (個体, 日) を present=False にする(ゼロ配分と偽らない)。"""
    spd = apc.STEPS_PER_DAY
    ev = [{"step": 0, "agent": 1, "kind": "speak"},
          {"step": 2 * spd, "agent": 1, "kind": "speak"},
          {"step": spd, "agent": 2, "kind": "speak"}]
    prof = apc.build_profiles(ev)
    i = prof["agents"].index(1)
    j = prof["days"].index(1)
    assert bool(prof["present"][i, j]) is False


def test_persona_consistency_no_data_returns_no_data():
    prof = apc.build_profiles([])
    assert prof["agents"] == []
    assert apc.within_agent_tvd(prof) == {}


# =========================================================================== #
# S-12 analyze_layers
# =========================================================================== #
def test_layers_channel_map_includes_event_as_offline():
    """★実測で見つかった 6 種目 `event` が offline に写像されていることを固定する。"""
    assert al.CHANNEL_LAYER["event"] == "offline"
    assert al.CHANNEL_LAYER["face"] == "offline"
    assert al.CHANNEL_LAYER["dm"] == "online"
    assert al.CHANNEL_LAYER["sns"] == "online"
    assert al.CHANNEL_LAYER["search"] == "broadcast"
    assert al.CHANNEL_LAYER["news"] == "broadcast"
    assert len(al.CHANNEL_LAYER) == 6


def test_layers_interaction_graph_layer_assignment():
    ev = [_ev(0, 1, "speak", {"hearers": [2, 3]}),
          _ev(1, 1, "dm", {"to": 4}),
          _ev(2, 5, "sns_read", {"authors": [6, -1]}),
          _ev(3, 7, "sns_like", {"author": 8}),
          _ev(4, 9, "news_read", {"titles": ["a", "b"]})]
    out = al.build_layer_graphs(ev)
    assert set(out["W"]["offline"]) == {(1, 2), (1, 3)}
    assert set(out["W"]["online"]) == {(1, 4), (5, 6), (7, 8)}
    # -1 相手の sns_read 1 件 + news_read 2 件 = 到達 3 件・到達人数 2 名
    assert out["broadcast"] == {"reach_events": 3, "reach_agents": 2}


def test_layers_broadcast_never_becomes_a_person_edge():
    ev = [_ev(0, 1, "sns_read", {"authors": [-1, -1]}),
          _ev(1, 2, "sns_like", {"author": -1})]
    out = al.build_layer_graphs(ev)
    assert out["W"]["offline"] == {} and out["W"]["online"] == {}
    assert out["broadcast"]["reach_events"] == 3


def test_layers_transmission_warns_on_unmapped_channel():
    ev = [_ev(0, 2, "transmission", {"from": 1, "channel": "telepathy"})]
    out = al.build_transmission_graphs(ev)
    assert out["channel_stats"]["telepathy"]["layer"] == al.UNMAPPED
    assert any("未知の channel" in w for w in out["warnings"])
    assert set(out["W"][al.UNMAPPED]) == {(1, 2)}


def test_layers_transmission_warns_when_person_layer_has_minus1_sender():
    """層の宣言を機械で検算する(offline に -1 が出たら前提が壊れている)。"""
    ev = [_ev(0, 2, "transmission", {"from": -1, "channel": "face"})]
    out = al.build_transmission_graphs(ev)
    assert out["channel_stats"]["face"]["n_from_minus1"] == 1
    assert any("前提が壊れている" in w for w in out["warnings"])
    assert out["broadcast"]["reach_events"] == 1


def test_layers_transmission_event_channel_goes_offline():
    ev = [_ev(0, 2, "transmission", {"from": 1, "channel": "event"}),
          _ev(1, 3, "transmission", {"from": 1, "channel": "sns"}),
          _ev(2, 4, "transmission", {"from": -1, "channel": "search"})]
    out = al.build_transmission_graphs(ev)
    assert set(out["W"]["offline"]) == {(1, 2)}
    assert set(out["W"]["online"]) == {(1, 3)}
    assert "broadcast" not in out["W"]                    # broadcast は辺を張らない
    assert out["broadcast"]["reach_events"] == 1


def test_layers_metrics_on_star_and_edge_overlap():
    star = {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0}
    mm = al.layer_metrics(star)
    assert mm["n_nodes"] == 4 and mm["n_edges"] == 3
    assert mm["max_degree"] == 3
    assert mm["mean_degree"] == pytest.approx(1.5)
    ov = al.edge_overlap(star, {(0, 1): 1.0, (5, 6): 1.0})
    assert ov["n_both"] == 1
    assert ov["jaccard"] == pytest.approx(1 / 4)


def test_layers_gini_bounds():
    assert al.gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)
    assert al.gini([0, 0, 0, 4]) > 0.7


# =========================================================================== #
# S-09 analyze_org_form
# =========================================================================== #
def test_org_form_freeman_star_is_one_and_complete_is_zero():
    """Freeman C_D: スター型 = 1・完全グラフ = 0(定義の固定)。"""
    star = {0: {1, 2, 3, 4}, 1: {0}, 2: {0}, 3: {0}, 4: {0}}
    assert aof.freeman_degree_centralization(star) == pytest.approx(1.0)
    comp = {i: {j for j in range(4) if j != i} for i in range(4)}
    assert aof.freeman_degree_centralization(comp) == pytest.approx(0.0)
    assert aof.freeman_degree_centralization({0: {1}, 1: {0}}) is None   # n<3


def test_org_form_grc_chain_is_higher_than_cycle():
    """GRC: 木/鎖(階層あり)> 有向巡回(フラット)。"""
    chain = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    cycle = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (3, 0): 1.0}
    g_chain = aof.global_reaching_centrality(chain)
    g_cycle = aof.global_reaching_centrality(cycle)
    assert g_chain["status"] == "OK" and g_chain["n_scc"] == 4
    assert g_chain["grc"] > g_cycle["grc"]
    assert g_cycle["grc"] == pytest.approx(0.0, abs=1e-9)   # 巡回は完全にフラット


def test_org_form_grc_zero_on_strongly_connected_is_declared_degenerate():
    """★実測で判明した落とし穴: 強連結なら GRC は構造的に 0 になる。

    「階層が無い」と読ませないために専用 status を立てる(実測: night_llm_100a3d の
    100 ノード / 1,936 有向辺で SCC=1 → GRC 0.0)。
    """
    cycle = {(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}
    out = aof.global_reaching_centrality(cycle)
    assert out["n_scc"] == 1
    assert out["grc"] == pytest.approx(0.0, abs=1e-9)
    assert out["status"] == "DEGENERATE_STRONGLY_CONNECTED"
    assert "判別できない" in out["note"]


def test_org_form_grc_skips_too_large_graph_loudly():
    big = {(i, i + 1): 1.0 for i in range(10)}
    out = aof.global_reaching_centrality(big, max_nodes=3)
    assert out["status"] == "SKIPPED_TOO_LARGE" and out["grc"] is None


def test_org_form_guimera_amaral_bridge_has_high_P():
    """2 モジュールの橋渡し役は参加係数 P が高く、モジュール内ハブは z が高い。

    モジュール 0 = {0,1,2,6}(0 が内ハブ)/ モジュール 3 = {3,4,5}(3 が内ハブ)。
    辺 0–3 が唯一の橋なので、0 と 3 だけが P > 0 になる。
    """
    adj = {0: {1, 2, 6, 3}, 1: {0}, 2: {0}, 6: {0},
           3: {0, 4, 5}, 4: {3}, 5: {3}}
    part = {0: 0, 1: 0, 2: 0, 6: 0, 3: 3, 4: 3, 5: 3}
    ga = aof.guimera_amaral(adj, part)
    assert ga["n"] == 7
    assert ga["P"][0] > ga["P"][1] == 0.0    # 0 は他モジュールへ 1 本持つ = 橋
    assert ga["P"][3] > ga["P"][4] == 0.0
    assert ga["z"][0] > ga["z"][1]           # 内部次数が最大 = director 側
    assert ga["z"][3] > ga["z"][4]
    # 参加係数の定義を手計算で固定: 0 は module0 へ 3 本 / module3 へ 1 本
    assert ga["P"][0] == pytest.approx(1 - ((3 / 4) ** 2 + (1 / 4) ** 2))


def test_org_form_ga_thresholds_are_reference_only():
    """★閾値による 4 分類は参考値であって主判定ではない(件数だけ出る)。"""
    z = {0: 3.0, 1: 0.0}
    P = {0: 0.7, 1: 0.1}
    rc = aof.ga_role_counts(z, P)
    assert rc["hub_and_bridge"] == 1 and rc["worker"] == 1
    assert sum(rc.values()) == 2


def test_org_form_no_eigenvector_centrality_anywhere():
    """★事前登録 §3-I の「eigenvector は採らない」を機械で固定する。"""
    src = (_ROOT / "scripts" / "analyze_org_form.py").read_text(encoding="utf-8")
    assert "eigenvector_centrality(" not in src
    assert "NOT_USED" in src


def test_org_form_window_graph_reuse_is_the_frozen_free_path():
    """`analyze_communities.build_window_graph` を再利用している(グラフを再実装しない)。"""
    spd = aof.STEPS_PER_DAY
    ev = []
    for d in range(3):
        for t in range(4):                       # min_weight 足切りを越えるだけ反復する
            ev.append(_ev2(d * spd + t, 0, "speak", {"hearers": [1, 2]}))
            ev.append(_ev2(d * spd + t, 1, "speak", {"hearers": [0]}))
    graph = aof.ac.build_window_graph(ev, aof.ac.MIN_WEIGHT)
    assert graph["nodes"] == {0, 1, 2}
    adj = {u: set(v) for u, v in graph["adj"].items()}
    # 辺は 0–1 と 0–2 の 2 本 = 中心 0 のスター型 → C_D = 1.0
    assert aof.freeman_degree_centralization(adj) == pytest.approx(1.0)


# =========================================================================== #
# S-10 analyze_mas_failures
# =========================================================================== #
def test_mas_fm13_counts_only_immediate_self_repetition():
    ev = [_ev(0, 1, "speak", {"text": "あ"}),
          _ev(1, 1, "speak", {"text": "あ"}),      # 直前と同一 = 反復
          _ev(2, 2, "speak", {"text": "あ"}),      # 別人 = 反復ではない
          _ev(3, 1, "speak", {"text": "い"}),
          _ev(4, 1, "speak", {"text": "あ"})]      # 間に別発話 = 直前反復ではない
    out = amf.fm13_step_repetition(ev)
    assert out["n_speak"] == 5 and out["n_immediate_repeat"] == 1
    assert out["rate"] == pytest.approx(0.2)


def test_mas_fm25_no_response_and_invite_none():
    ev = [_ev(0, 1, "speak", {"hearers": [2, 3]}),
          _ev(2, 2, "speak", {"text": "はい"}),          # 2 は窓内に応答
          _ev(0, 4, "joint_invite", {"verdict": "none"}),
          _ev(0, 5, "joint_invite", {"verdict": "accept"})]
    out = amf.fm25_ignored_input(ev, window=6)
    assert out["n_addressed"] == 2 and out["n_no_response"] == 1     # 3 が無応答
    assert out["rate_no_response"] == pytest.approx(0.5)
    assert out["n_invite_verdict_none"] == 1
    assert out["rate_invite_none"] == pytest.approx(0.5)


def test_mas_fm26_blank_place_is_excluded_from_denominator():
    """place が空の計画項目は「検証不能」= 分母外(0 として数えない)。"""
    ev = [_ev(0, 1, "day_plan", {"plan": [{"place": "カフェ"}, {"place": ""},
                                          {"place": "公園"}]}),
          _ev(5, 1, "arrive", {"name": "カフェ"})]
    out = amf.fm26_reasoning_action_mismatch(ev)
    assert out["n_plan_items"] == 3
    assert out["n_plan_items_without_place"] == 1
    assert out["n_verifiable_places"] == 2 and out["n_not_reached"] == 1
    assert out["rate"] == pytest.approx(0.5)
    assert out["status"] == "IMPLEMENTED"          # 語彙が半分重なるので妥当性は保たれる


def test_mas_fm26_detects_name_space_mismatch():
    """★実測で判明した落とし穴: 計画の place(自由文)と到達名(地図ノード名)が
    別の名前空間だと、この指標は「文字列の不一致」を測ってしまう。機械で検知する。"""
    ev = [_ev(0, 1, "day_plan", {"plan": [{"place": "渋谷のカフェ"},
                                          {"place": "自宅"}]}),
          _ev(5, 1, "arrive", {"name": "路上"}),
          _ev(6, 1, "arrive", {"name": "渋谷駅ハチ公口"})]
    out = amf.fm26_reasoning_action_mismatch(ev)
    assert out["status"] == "NAME_SPACE_MISMATCH"
    assert out["n_vocab_overlap"] == 0
    assert out["rate"] == pytest.approx(1.0)       # 数字は 100% だが失敗率ではない
    assert "文字列の不一致" in out["note"]


def test_mas_fm31_group_with_no_join_is_premature():
    spd = amf.STEPS_PER_DAY
    ev = [_ev(0, 1, "group_found", {"group_id": 1}),
          _ev(0, 2, "group_found", {"group_id": 2}),
          _ev(spd, 3, "group_join", {"group_id": 1}),
          _ev(10 * spd, 4, "group_join", {"group_id": 2})]   # 窓外の加入は救わない
    out = amf.fm31_premature_termination(ev, premature_days=3)
    assert out["n_groups_founded"] == 2
    assert out["n_groups_no_join_within_window"] == 1
    assert out["rate_groups"] == pytest.approx(0.5)


def test_mas_fm23_js_divergence_bounds_and_no_threshold():
    a = amf.char_ngrams("あいうえお")
    assert amf.js_divergence(a, a) == pytest.approx(0.0, abs=1e-12)
    assert amf.js_divergence(a, amf.char_ngrams("かきくけこ")) == pytest.approx(1.0)
    assert amf.js_divergence(a, {}) is None
    ev = [_ev(0, 1, "group_found", {"group_id": 7, "purpose": "音楽", "name": "会"}),
          _ev(1, 1, "speak", {"text": "音楽の会をやろう"})]
    out = amf.fm23_task_derailment(ev)
    assert out["status"] == "IMPLEMENTED" and out["n_groups"] == 1
    assert "閾値は置かない" in out["note"]


def test_mas_not_implemented_modes_all_carry_a_reason():
    """★判定しない様式には必ず理由が付く(理由を書くのが本スクリプトの本体)。"""
    assert len(amf.NOT_IMPLEMENTED) == 8
    for fm, (name, reason) in amf.NOT_IMPLEMENTED.items():
        assert fm.startswith("FM-") and name and len(reason) > 20
    assert 6 + len(amf.NOT_IMPLEMENTED) == 14      # 実装 6 + 未実装 8 = MAST の 14 様式


def test_mas_fm15_is_marked_proxy_not_implemented():
    """代理指標に `PROXY` を付け続ける(「終了条件を認識していない」とは書かない)。"""
    ev = [_ev(0, 1, "speak", {"hearers": [2]}) for _ in range(15)]
    out = amf.fm15_termination_unaware(ev, turn_cap=12)
    assert out["status"] == "PROXY"
    assert out["n_over_cap"] == 1 and out["max_turns"] == 15


# =========================================================================== #
# S-14 analyze_ipf_fidelity
# =========================================================================== #
_POP = {
    "meta": {"status": "test", "note": "synthetic"},
    "age_bands": [{"band": [18, 24], "share": 0.5}, {"band": [25, 34], "share": 0.5}],
    "gender": {"女": 0.5, "男": 0.5},
    "occupations": [{"name": "A", "share": 0.5}, {"name": "B", "share": 0.5}],
}


def test_ipf_perfect_fit_is_zero_error():
    recs = ([{"age": 20, "gender": "女", "occupation": "A"}] * 50
            + [{"age": 30, "gender": "男", "occupation": "B"}] * 50)
    res = aif.analyze(recs, _POP, "synthetic")
    for key in ("age_band", "gender", "occupation"):
        a = res["attributes"][key]
        assert a["TAE"] == pytest.approx(0.0)
        assert a["CE"] == pytest.approx(0.0)
        assert a["SRMSE"] == pytest.approx(0.0)


def test_ipf_tae_ce_srmse_hand_computation():
    """TAE / CE / SRMSE を手計算で固定する(Voas & Williamson 2001)。"""
    recs = ([{"age": 20, "gender": "女", "occupation": "A"}] * 60
            + [{"age": 30, "gender": "男", "occupation": "B"}] * 40)
    res = aif.analyze(recs, _POP, "synthetic")
    a = res["attributes"]["age_band"]
    assert a["TAE"] == pytest.approx(20.0)        # |60-50| + |40-50|
    assert a["CE"] == pytest.approx(10.0)         # TAE/2 = 誤分類個体数
    # RMSE = sqrt((100+100)/2) = 10, 平均期待 = 50 → SRMSE = 0.2
    assert a["SRMSE"] == pytest.approx(0.2)


def test_ipf_out_of_target_categories_are_reported_not_dropped():
    """★目標に無いカテゴリ・帯外の年齢を黙って捨てない。"""
    recs = [{"age": 90, "gender": "他", "occupation": "議員"}] * 10
    res = aif.analyze(recs, _POP, "synthetic")
    for key in ("age_band", "gender", "occupation"):
        assert res["attributes"][key]["n_not_in_target"] == 10
    assert aif.assign_age_band(90, _POP["age_bands"]) == aif.NOT_IN_TARGET
    assert aif.assign_age_band(20, _POP["age_bands"]) == "18-24"
    assert aif.assign_age_band(None, _POP["age_bands"]) == aif.NOT_IN_TARGET


def test_ipf_zero_cells_are_counted():
    recs = [{"age": 20, "gender": "女", "occupation": "A"}] * 10
    res = aif.analyze(recs, _POP, "synthetic")
    assert res["attributes"]["age_band"]["n_zero_cells"] == 1   # 25-34 が空


def test_ipf_no_data_is_declared():
    res = aif.analyze([], _POP, "synthetic")
    assert res["status"] == "NO_DATA" and res["n_records"] == 0


def test_ipf_real_population_json_is_loadable_and_shares_sum_to_one():
    """実データの目標分布が読め、share が 1 に十分近いことを確認する(読むだけ)。"""
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    tgt = aif.target_shares(pop)
    for key in ("age_band", "gender", "occupation"):
        assert tgt[key]
        assert sum(tgt[key].values()) == pytest.approx(1.0, abs=0.02)


# =========================================================================== #
# S-18 analyze_stereotype
# =========================================================================== #
def test_stereotype_two_sided_p_matches_known_values():
    """scipy を使わずに正規分布の裾を出す(erfc 実装の妥当性)。"""
    assert ast_.two_sided_p(1.96) == pytest.approx(0.05, abs=0.001)
    assert ast_.two_sided_p(0.0) == pytest.approx(1.0)
    assert ast_.two_sided_p(-1.96) == pytest.approx(0.05, abs=0.001)


def test_stereotype_char_ngrams_are_not_words():
    g = ast_.char_ngrams("あいう")
    assert g == {"あい": 1, "いう": 1, "あいう": 1}       # n=2 と n=3 の両方


def test_stereotype_log_odds_sign_and_symmetry():
    ci = {"XX": 100, "YY": 1}
    cj = {"XX": 1, "YY": 100}
    prior, a0 = ast_.build_prior({"XX": 101, "YY": 101}, a0=10.0)
    res = ast_.log_odds_dirichlet(ci, cj, prior, a0)
    assert res["XX"]["z"] > 0 and res["YY"]["z"] < 0
    rev = ast_.log_odds_dirichlet(cj, ci, prior, a0)
    assert res["XX"]["delta"] == pytest.approx(-rev["XX"]["delta"], abs=1e-6)


def test_stereotype_prior_shrinks_point_estimate_of_rare_ngrams():
    """informative Dirichlet prior が低頻度 n-gram の **点推定 δ̂ を縮める**(Monroe 2008)。

    ★ z ではなく δ̂ で固定するのが正しい。事前分布が小さいと δ̂ は発散する一方で
    分散項も同時に膨らむため、|z| は必ずしも単調に動かない(実測で確認済み)。
    正則化の実体は「点推定が背景分布へ引き寄せられること」である。
    """
    ci, cj = {"AA": 3, "BB": 300}, {"AA": 0, "BB": 300}
    small, s0 = ast_.build_prior({"AA": 3, "BB": 600}, a0=1.0)
    large, l0 = ast_.build_prior({"AA": 3, "BB": 600}, a0=500.0)
    d_small = ast_.log_odds_dirichlet(ci, cj, small, s0)["AA"]["delta"]
    d_large = ast_.log_odds_dirichlet(ci, cj, large, l0)["AA"]["delta"]
    assert abs(d_large) < abs(d_small)


def test_stereotype_zero_count_group_still_yields_finite_statistics():
    """★事前分布のおかげで「片方 0 件」でもゼロ除算せず有限値が出る(prior を置く理由)。"""
    ci, cj = {"AA": 3, "BB": 300}, {"AA": 0, "BB": 300}
    prior, a0 = ast_.build_prior({"AA": 3, "BB": 600}, a0=500.0)
    rec = ast_.log_odds_dirichlet(ci, cj, prior, a0)["AA"]
    assert math.isfinite(rec["delta"]) and math.isfinite(rec["z"])
    assert rec["z"] > 0 and 0.0 <= rec["p"] <= 1.0


def test_stereotype_one_vs_rest_marks_the_distinctive_group():
    by_group = {"g1": {"XX": 200, "ZZ": 10}, "g2": {"YY": 200, "ZZ": 10}}
    total = {"XX": 200, "YY": 200, "ZZ": 20}
    out = ast_.marked_ngrams(by_group, total, a0=10.0, min_count=1, top=5)
    top_g1 = [t["ngram"] for t in out["g1"]["top"] if t["z"] > 0]
    top_g2 = [t["ngram"] for t in out["g2"]["top"] if t["z"] > 0]
    assert "XX" in top_g1 and "YY" in top_g2
    assert "XX" not in top_g2


def test_stereotype_min_count_filters_rare_ngrams():
    by_group = {"g1": {"XX": 100, "RR": 1}, "g2": {"YY": 100}}
    total = {"XX": 100, "YY": 100, "RR": 1}
    out = ast_.marked_ngrams(by_group, total, a0=10.0, min_count=5, top=10)
    assert all(t["ngram"] != "RR" for t in out["g1"]["top"])


def test_stereotype_intersection_rule_is_preserved():
    """★Cheng らの積集合則: 片方だけ有意な n-gram は交差セルに出さない。"""
    a = {"20代": {"all_significant": ["XX", "YY"]}}
    b = {"学生": {"all_significant": ["YY", "ZZ"]}, "無職": {"all_significant": ["QQ"]}}
    inter = ast_.intersect_marked(a, b)
    assert inter == {"20代 × 学生": ["YY"]}


def test_stereotype_age_band_assignment():
    assert ast_.age_band_of(20) == "18-24"
    assert ast_.age_band_of(65) == "65-79"
    assert ast_.age_band_of(5) is None
    assert ast_.age_band_of("x") is None


# =========================================================================== #
# end-to-end: 合成ランディレクトリ(parquet ローダまで通す)
# =========================================================================== #
@pytest.fixture(scope="module")
def mini_run(tmp_path_factory):
    """最小の合成ラン(l1_events.parquet + agents.json)を tmp に作る。

    3 名 × 3 日。offline/online/broadcast の全層と、6 種の channel を 1 件ずつ含む。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = tmp_path_factory.mktemp("mini_run")
    spd = apc.STEPS_PER_DAY                       # 144(1 step = 10 sim 分)
    rows = []

    def add(step, agent, kind, payload):
        rows.append((int(step), int(step) * 10, int(agent), kind,
                     json.dumps(payload, ensure_ascii=False)))

    for day in range(3):
        b = day * spd
        for t in range(5):                       # 足切り(min_weight)を越える反復
            add(b + t, 0, "speak", {"hearers": [1, 2], "text": "みんなで音楽をやろう"})
            add(b + t, 1, "speak", {"hearers": [0], "text": "いいね賛成する"})
            add(b + t, 2, "speak", {"hearers": [0], "text": "わたしは写真がすきだ"})
        add(b + 6, 0, "dm", {"to": 1, "text": "こんばんは音楽の件"})
        add(b + 7, 1, "sns_read", {"authors": [0, -1], "n_posts": 2})
        add(b + 8, 2, "sns_like", {"author": 0, "post_id": day})
        add(b + 9, 2, "news_read", {"titles": ["議会の改選"]})
        add(b + 10, 0, "arrive", {"name": "カフェ", "first_visit": day == 0})
        add(b + 1, 0, "day_plan", {"n": 2, "plan": [{"place": "カフェ", "what": "work"},
                                                    {"place": "図書館", "what": "study"}]})
    # 6 種すべての channel を 1 件ずつ(層の写像の end-to-end 検算)
    for i, ch in enumerate(sorted(al.CHANNEL_LAYER)):
        frm = -1 if al.CHANNEL_LAYER[ch] == "broadcast" else 0
        add(20 + i, 1, "transmission", {"item_id": f"vocab-{i}", "from": frm,
                                        "channel": ch})
    add(30, 0, "group_found", {"group_id": 1, "founder": 0, "purpose": "音楽",
                               "name": "音楽の会"})
    add(31, 1, "group_join", {"group_id": 1, "founder": 0, "name": "音楽の会"})
    add(32, 2, "group_found", {"group_id": 2, "founder": 2, "purpose": "写真",
                               "name": "写真の会"})
    add(33, 2, "joint_invite", {"invitee": 1, "verdict": "none", "accepted": False})
    add(34, 0, "sns_post", {"text": "音楽の会をはじめました"})

    rows.sort()
    tbl = pa.table({
        "step": pa.array([r[0] for r in rows], pa.int64()),
        "sim_min": pa.array([r[1] for r in rows], pa.int64()),
        "agent_id": pa.array([r[2] for r in rows], pa.int64()),
        "kind": pa.array([r[3] for r in rows], pa.string()),
        "payload": pa.array([r[4] for r in rows], pa.string()),
    })
    pq.write_table(tbl, str(run / "l1_events.parquet"))
    (run / "agents.json").write_text(json.dumps([
        {"id": 0, "name": "a", "age": 30, "gender": "女", "occupation": "会社員"},
        {"id": 1, "name": "b", "age": 22, "gender": "男", "occupation": "大学生"},
        {"id": 2, "name": "c", "age": 45, "gender": "女", "occupation": "写真家"},
    ], ensure_ascii=False), encoding="utf-8")
    return str(run)


def test_e2e_layers_maps_all_six_channels(mini_run):
    res = al.analyze(mini_run)
    assert res["status"] == "OK"
    stats = res["transmission"]["channel_stats"]
    assert set(stats) == set(al.CHANNEL_LAYER)
    for ch, st in stats.items():
        assert st["layer"] == al.CHANNEL_LAYER[ch]
    assert res["transmission"]["warnings"] == []          # 写像の前提が壊れていない
    assert res["interaction"]["layers"]["offline"]["n_edges"] == 2   # 0-1, 0-2
    assert res["interaction"]["layers"]["online"]["n_edges"] == 2    # 0-1(dm/read), 0-2
    assert res["interaction"]["broadcast"]["reach_events"] > 0
    assert res["transmission"]["layers"]["offline"]["n_edges"] == 1  # face+event は同ペア
    md = al.render(res)
    assert "三層" in md and "broadcast" in md


def test_e2e_org_form_runs_and_reports_all_four_axes(mini_run):
    res = aof.analyze(mini_run, window_days=7)
    assert res["status"] == "OK" and res["windows"]
    w = res["windows"][0]
    assert w["freeman_C_D"] is not None
    # 合成ランの会話網は強連結なので、GRC は退化ラベルが立つのが正しい
    assert w["grc"]["status"] in ("OK", "DEGENERATE_STRONGLY_CONNECTED")
    assert w["grc"]["n_scc"] >= 1
    assert w["ga_n"] >= 2
    assert res["mode"]["role_counts"]
    assert res["spec"]["eigenvector_centrality"].startswith("NOT_USED")
    md = aof.render(res)
    assert "Freeman C_D" in md and "Palla" in md


def test_e2e_mas_failures_scores_six_modes(mini_run):
    res = amf.analyze(mini_run)
    assert res["status"] == "OK"
    assert res["coverage"] == {"n_total_modes": 14, "n_scored": 6,
                               "n_not_implemented": 8}
    # 「図書館」は計画されたが到達していない → 未到達が 3 日分
    assert res["modes"]["FM-2.6"]["n_not_reached"] == 3
    assert res["modes"]["FM-3.1"]["n_groups_founded"] == 2
    assert res["modes"]["FM-2.5"]["n_invite_verdict_none"] == 1
    md = amf.render(res)
    assert "判定しない様式" in md


def test_e2e_stereotype_runs_and_marks_groups(mini_run):
    res = ast_.analyze(mini_run, top=5, min_count=2)
    assert res["status"] == "OK"
    assert res["n_utterances"] > 0
    assert set(res["attributes"]) == {"age_band", "gender", "occupation"}
    assert res["attributes"]["occupation"]                      # 群が立っている
    ast_.render(res)


def test_e2e_persona_consistency_runs(mini_run):
    res = apc.analyze(mini_run, mc=100)
    assert res["status"] == "OK"
    assert res["n_agents"] == 3 and res["n_days"] == 3
    assert res["spec"]["exchange_unit"].startswith("day_pair")
    apc.render(res)


def test_e2e_ipf_fidelity_on_run_agents(mini_run):
    recs = aif.load_run_agents(mini_run)
    assert len(recs) == 3
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(
        encoding="utf-8"))
    res = aif.analyze(recs, pop, "mini_run")
    assert res["status"] == "OK"
    for key in ("age_band", "gender", "occupation"):
        assert res["attributes"][key]["SRMSE"] is not None
    aif.render(res)


# =========================================================================== #
# 共通: R1(凍結資産・依存)の機械固定
# =========================================================================== #
_NEW_SCRIPTS = ("analyze_seed_variance.py", "analyze_persona_consistency.py",
                "analyze_layers.py", "analyze_org_form.py", "analyze_mas_failures.py",
                "analyze_ipf_fidelity.py", "analyze_stereotype.py")


@pytest.mark.parametrize("name", _NEW_SCRIPTS)
def test_new_scripts_do_not_import_scipy(name):
    """依存は numpy / pyarrow / networkx / 標準ライブラリのみ(`scipy` は依存に無い)。"""
    src = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "import scipy" not in src
    assert "from scipy" not in src


@pytest.mark.parametrize("name", _NEW_SCRIPTS)
def test_new_scripts_are_read_only_towards_src_and_conf(name):
    """★`src/` と `conf/` へ書き込む経路をコードに持たない(読み取り専用の機械固定)。"""
    src = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
    for bad in ('open(os.path.join(_ROOT, "src"', "conf/config.yaml", "shutil.copy",
                "os.remove", "shutil.rmtree"):
        assert bad not in src


def test_frozen_spec_files_are_untouched_by_this_batch():
    """★`metrics_spec_hash` の SPEC_FILES を 1 本も改変していないこと。

    ハッシュそのものは `tests/test_metrics_spec.py` 系が見る。ここでは
    **本バッチが凍結 4 本の解析器を import しかしていない**ことを固定する。
    """
    frozen = ("analyze_norms.py", "analyze_specialization.py", "analyze_beliefs.py")
    for name in _NEW_SCRIPTS:
        src = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
        for f in frozen:
            mod = f[:-3]
            assert f"import {mod}" not in src, f"{name} が凍結 {f} を import している"
    # diagnose_stationarity は **import のみ許される**(複製しない)= S-06 の設計
    apc_src = (_ROOT / "scripts" / "analyze_persona_consistency.py").read_text(
        encoding="utf-8")
    assert "from diagnose_stationarity import" in apc_src


def test_preregistration_declares_six_channels_including_event():
    """★事前登録 §1.5 が 6 種目 `event` を offline として宣言していること(文書と実装の一致)。"""
    doc = (_ROOT / "docs" / "plans" / "stationarity-preregistration.md").read_text(
        encoding="utf-8")
    assert "`event`" in doc
    for ch in al.CHANNEL_LAYER:
        assert f"`{ch}`" in doc


def test_report_template_declaration_sections_are_not_empty_stubs():
    """★S-08 / S-15 / S-17 の宣言欄が空枠のままでないこと。"""
    doc = (_ROOT / "docs" / "plans" / "observation-report-template.md").read_text(
        encoding="utf-8")
    for marker in ("### 7.2 S-08", "### 7.3 S-15", "### 7.4 S-17"):
        assert marker in doc
    assert "Fluency Fallacy" in doc          # S-17 の近縁概念(§7.4)
    assert "片側検査" in doc                  # S-08 の非対称な読み方(§7.2)
    assert "崩壊ラン" in doc                  # S-15 の除外規則(§7.3)


def test_judge_report_carries_survey_scope_line():
    """★S-07: judge.py が「world_changer の単一次元にのみ用いる」を出力に書くこと。"""
    src = (_ROOT / "scripts" / "judge.py").read_text(encoding="utf-8")
    assert "world_changer の単一次元にのみ" in src
    assert "§4.4" in src
    # 静的文字列のみ(実験条件・数値を埋め込まない)
    assert "サーベイ §4.4" in src


def test_all_new_scripts_have_main_guard_so_import_is_side_effect_free():
    for name in _NEW_SCRIPTS:
        src = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in src


def test_rounding_helpers_are_none_safe_and_nan_safe():
    for mod in (asv, al, aof, amf, aif, ast_):
        assert mod._r(None) is None
        assert mod._r(float("nan")) is None
        assert mod._r(float("inf")) is None
        assert mod._r(1.23456789, 3) == pytest.approx(1.235)
    assert math.isclose(asv._r(2.5, 1), 2.5)
