"""IF-E フェーズ1: scripts/analyze_accounting.py(部門別会計行列 + 貨幣保存の検査)の検証。

正典: docs/research/if-lane-research.md §5(Caiani et al. 2016 の検査法2本)。
house style は tests/test_analyze_luck.py / tests/test_analyze_founders.py に倣う。

確かめること
------------
A. **検査器そのものの正しさ**(合成イベント列)
   1. 保存が成り立つ系 → 検査① PASS・void 漏れ 0。
   2. 意図的に漏らした系(残高だけ動いてイベントが無い / イベントだけあって残高が動かない)
      → 検査① が FAIL として検出する。
   3. フロー行列のゼロ和と void 行・列の意味(貨幣の創出/消滅)。
   4. 分類器の自己整合(flows_for と move_for の二重写像が一致する)。
   5. 行政の内部整合(revenue/expense/balance)と外部整合(行列の純額 vs 残高変化)。
   6. serve→spend の突合(スタッフ有り=customer 鍵 / スタッフ不在=同 step 同 cat の記録順)。
B. **小ラン実測の回帰固定**(mock ≤288 step・LLM は mock・シム本体ゼロタッチ)
   7. 家計の貨幣保存が成り立つ(=分類器が家計側の金の経路を1本も取りこぼしていない)。
   8. **漏れの内訳が既知リストに閉じている**(新しい漏れ経路が増えたら fail)。
   9. **未分類の金額キーを持つイベント種が無い**(= 金の経路の追加を検知する装置)。
  10. 既知の断絶 revenue_est ≫ 実測 spend が実測として再現する(ゼロと偽らない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))

import analyze_accounting as aa                      # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ev(step, agent, kind, payload=None, x=0.0, y=0.0):
    return {"step": step, "agent_id": agent, "kind": kind,
            "payload": payload or {}, "x": x, "y": y}


def _run(events, stock, budget=(), ledger=(), serve=(), other=(), **cfg):
    base = {"accounts_on": False, "government_on": False, "pool_on": False,
            "org_ledger_on": False, "indoor_fields_on": False}
    base.update(cfg)
    return {"cfg": base, "events": list(events), "stock": dict(stock),
            "budget": list(budget), "ledger": list(ledger), "serve": list(serve),
            "other_money": list(other), "run_dir": "<synthetic>"}


# ===========================================================================
# A-1. 保存が成り立つ合成系 → 検査① PASS
# ===========================================================================
def test_conserving_system_passes_check1():
    """賃金 +1000 → 消費 −300 を、L3 残高がそのとおり動く系で組む。残差ゼロで PASS。"""
    events = [
        _ev(1, 7, "wage", {"amount": 1000.0, "balance": 6000.0}),
        _ev(2, 7, "spend", {"amount": 300.0, "balance": 5700.0, "cat": "food"}),
    ]
    stock = {0: {7: 5000.0}, 4: {7: 5700.0}}
    rep = aa.analyze(_run(events, stock))
    c = rep["checks"][aa.HOUSEHOLD]
    assert c["observable"] and c["pass"]
    assert c["total_residual"] == 0.0
    assert rep["classifier_consistency"]["pass"]


def test_zero_flow_zero_stock_is_trivially_conserved():
    events = []
    stock = {0: {1: 100.0}, 12: {1: 100.0}}
    c = aa.analyze(_run(events, stock))["checks"][aa.HOUSEHOLD]
    assert c["pass"] and c["total_abs_flow"] == 0.0


# ===========================================================================
# A-2. 意図的に漏らした系 → 検査① が FAIL として検出
# ===========================================================================
def test_leak_stock_moved_without_event_is_detected():
    """残高だけ 500 増えてフローが無い(=イベント化されていない送金)を検出する。"""
    events = [_ev(1, 7, "wage", {"amount": 1000.0, "balance": 6000.0})]
    stock = {0: {7: 5000.0}, 4: {7: 6500.0}}          # 実際は +1500 動いた
    c = aa.analyze(_run(events, stock))["checks"][aa.HOUSEHOLD]
    assert not c["pass"]
    assert c["total_residual"] == pytest.approx(500.0)
    assert c["windows"][0]["kinds"] == ["wage"]        # どの窓・どの種の傍で起きたかが出る


def test_leak_event_without_stock_move_is_detected():
    """フローはあるのに残高が動かない(=clamp / 記帳だけ)を検出する。"""
    events = [_ev(1, 7, "spend", {"amount": 800.0, "balance": 0.0, "cat": "shop"})]
    stock = {0: {7: 500.0}, 4: {7: 0.0}}               # 実際は −500 しか動けない(床でクリップ)
    c = aa.analyze(_run(events, stock))["checks"][aa.HOUSEHOLD]
    assert not c["pass"]
    assert c["total_residual"] == pytest.approx(300.0)   # 記帳 800 − 実移動 500


def test_pool_rotation_is_separated_not_counted_as_leak():
    """個体の出入り(pool)は残差に混ぜず pool_in/pool_out として分離表示する。"""
    events = []
    stock = {0: {1: 100.0}, 12: {1: 100.0, 2: 9999.0}}   # 2 が途中参加
    c = aa.analyze(_run(events, stock))["checks"][aa.HOUSEHOLD]
    assert c["pass"]
    assert c["windows"][0]["pool_in"] == pytest.approx(9999.0)
    assert c["windows"][0]["residual"] == 0.0


def test_rel_tol_is_an_argument_not_a_hardcoded_zero():
    """許容誤差は相対誤差で、引数で動く(浮動小数の丸めで厳密ゼロは不成立=文献の示唆4)。"""
    events = [_ev(1, 7, "wage", {"amount": 1000.0, "balance": 6000.0})]
    stock = {0: {7: 5000.0}, 4: {7: 6000.5}}            # 0.5 の丸め残差
    assert not aa.analyze(_run(events, stock), rel_tol=1e-6)["checks"][aa.HOUSEHOLD]["pass"]
    assert aa.analyze(_run(events, stock), rel_tol=1e-2)["checks"][aa.HOUSEHOLD]["pass"]


# ===========================================================================
# A-3. フロー行列と void(検査②)
# ===========================================================================
def test_matrix_is_double_entry_so_all_sector_nets_sum_to_zero():
    events = [
        _ev(1, 7, "wage", {"amount": 1000.0, "balance": 6000.0}),
        _ev(2, 7, "spend", {"amount": 300.0, "balance": 5700.0, "cat": "food"}),
        _ev(3, 7, "spend", {"amount": 100.0, "balance": 5600.0, "cat": "venture"}),
    ]
    rep = aa.analyze(_run(events, {0: {7: 5000.0}, 4: {7: 5600.0}}))
    nets = [v["net"] for v in rep["sector_sums"].values()]
    assert sum(nets) == pytest.approx(0.0)


def test_void_row_and_column_quantify_created_and_destroyed_money():
    """客の払った金の受け取り手が居ない=消滅 / 出所の無い賃金=創出。両方が void に出る。"""
    events = [
        _ev(1, 7, "wage", {"amount": 1000.0, "balance": 6000.0}),          # source なし=本業
        _ev(2, 7, "spend", {"amount": 300.0, "balance": 5700.0, "cat": "food"}),
    ]
    rep = aa.analyze(_run(events, {0: {7: 5000.0}, 4: {7: 5700.0}}))
    assert rep["totals"]["money_created"] == pytest.approx(1000.0)
    assert rep["totals"]["money_destroyed"] == pytest.approx(300.0)
    assert rep["leaks"]["wage:work"]["created"] == pytest.approx(1000.0)
    assert rep["leaks"]["spend:food"]["destroyed"] == pytest.approx(300.0)


def test_venture_purchase_is_not_a_leak_because_the_owner_exists():
    """屋台の売上は店主(家計)へ実際に入るので void を通らない=保存が成り立つ既存経路。"""
    events = [
        _ev(1, 7, "spend", {"amount": 500.0, "balance": 4500.0, "cat": "venture"}),
        _ev(1, 9, "venture_sale", {"amount": 500.0, "balance": 500.0, "buyer": 7}),
    ]
    rep = aa.analyze(_run(events, {0: {7: 5000.0, 9: 0.0}, 4: {7: 4500.0, 9: 500.0}}))
    assert rep["totals"]["money_created"] == 0.0
    assert rep["totals"]["money_destroyed"] == 0.0
    assert rep["checks"][aa.HOUSEHOLD]["pass"]
    assert rep["sector_sums"][aa.VENTURE]["net"] == pytest.approx(0.0)


def test_vc_dividend_splits_venture_sale_between_owner_and_bank():
    events = [_ev(1, 9, "venture_sale",
                  {"amount": 500.0, "balance": 400.0, "buyer": 7, "dividend": 100.0})]
    fl = aa.flows_for("venture_sale", events[0]["payload"], {})
    edges = {(f.src, f.dst): f.amount for f in fl}
    assert edges[(aa.VENTURE, aa.HOUSEHOLD)] == pytest.approx(400.0)
    assert edges[(aa.VENTURE, aa.BANK)] == pytest.approx(100.0)


def test_withdraw_is_internal_transfer_not_a_flow():
    """現金⇄口座は家計内部の資産振替。行列に出さず、残高合計も動かさない。"""
    assert aa.flows_for("withdraw", {"amount": 2000.0}, {}) == []
    mv = aa.move_for("withdraw", 3, {"amount": 2000.0}, accounts_on=True)
    assert len(mv) == 1 and mv[0].total == pytest.approx(0.0)
    assert (mv[0].cash, mv[0].acct) == (2000.0, -2000.0)


def test_wage_source_mapping_covers_every_documented_source():
    """civil=行政 / home_refill=街外 / gig・salary・severance・無指定=void(出所なし)。"""
    assert aa.WAGE_SOURCE_SECTOR["civil"][0] == aa.GOVERNMENT
    assert aa.WAGE_SOURCE_SECTOR["home_refill"][0] == aa.EXTERNAL
    for s in ("gig", "salary", "severance"):
        assert aa.WAGE_SOURCE_SECTOR[s][0] == aa.VOID
    assert aa._WAGE_DEFAULT[0] == aa.VOID


def test_spend_destination_only_venture_has_a_receiver():
    assert aa.spend_destination("venture")[0] == aa.VENTURE
    for cat in ("food", "cafe", "shop", "nightlife", "taxi", "bus", "lodging", "fixed_cost"):
        assert aa.spend_destination(cat)[0] == aa.VOID


def test_consumption_tax_is_carved_out_of_the_price_not_added():
    """内税=価格は名目不変。tax は家計の残高を動かさず、納税者は spend の受け取り側。"""
    assert aa.move_for("tax", 7, {"tax": "consumption", "amount": 90.0}, False) == []
    events = [
        _ev(1, 7, "spend", {"amount": 1000.0, "balance": 4000.0, "cat": "food"}),
        _ev(1, 7, "tax", {"tax": "consumption", "amount": 90.0, "to": "nation", "base": 1000.0}),
    ]
    rep = aa.analyze(_run(events, {0: {7: 5000.0}, 4: {7: 4000.0}}, government_on=True))
    assert rep["checks"][aa.HOUSEHOLD]["pass"]           # 家計の流出は 1000 だけ(1090 ではない)
    m = rep["matrix"]
    assert m[f"{aa.HOUSEHOLD}->{aa.VOID}"] == pytest.approx(1000.0)
    assert m[f"{aa.VOID}->{aa.GOVERNMENT}"] == pytest.approx(90.0)


def test_income_tax_on_civil_wage_stays_inside_government():
    """公務員給与の源泉税は行政の内部振替=行列に出さない(歳出 gross − 歳入 tax = 手取り)。"""
    events = [
        _ev(1, 7, "tax", {"tax": "income", "amount": 200.0, "to": "nation", "base": 1000.0}),
        _ev(1, 7, "wage", {"amount": 800.0, "balance": 5800.0, "source": "civil",
                           "gross": 1000.0, "tax": 200.0}),
    ]
    rep = aa.analyze(_run(events, {0: {7: 5000.0}, 4: {7: 5800.0}}, government_on=True))
    m = rep["matrix"]
    assert m[f"{aa.GOVERNMENT}->{aa.HOUSEHOLD}"] == pytest.approx(800.0)
    assert f"{aa.GOVERNMENT}->{aa.GOVERNMENT}" not in m
    assert rep["totals"]["money_created"] == 0.0


def test_theft_destroys_money_because_the_offender_receives_nothing():
    """diversity.py: victim.money が減るだけで offender は受け取らない(実装どおりに記帳)。"""
    ev = _ev(5, 3, "crime", {"kind": "theft", "victim": 8, "offender": 3, "amount": 400.0})
    mv = aa.move_for("crime", 3, ev["payload"], False)
    assert len(mv) == 1 and mv[0].agent_id == 8 and mv[0].total == pytest.approx(-400.0)
    fl = aa.flows_for("crime", ev["payload"], {})
    assert (fl[0].src, fl[0].dst) == (aa.HOUSEHOLD, aa.VOID)


def test_enforcement_fine_goes_to_government_only_when_government_is_on():
    p = {"rule_id": None, "officer": 1, "target": 9, "penalty": 500.0}
    assert aa.flows_for("enforcement", p, {"government_on": True})[0].dst == aa.GOVERNMENT
    assert aa.flows_for("enforcement", p, {"government_on": False})[0].dst == aa.VOID
    mv = aa.move_for("enforcement", 1, p, False)         # agent_id は警察官・減るのは target
    assert mv[0].agent_id == 9 and mv[0].total == pytest.approx(-500.0)


def test_deposit_escrow_round_trip_nets_to_zero():
    paid = aa.flows_for("deposit", {"amount": 1000.0, "phase": "paid"}, {})
    back = aa.flows_for("deposit", {"amount": 1000.0, "phase": "refund"}, {})
    forf = aa.flows_for("deposit", {"amount": 1000.0, "phase": "forfeit"}, {})
    assert (paid[0].src, paid[0].dst) == (aa.HOUSEHOLD, aa.VOID)
    assert (back[0].src, back[0].dst) == (aa.VOID, aa.HOUSEHOLD)
    assert (forf[0].src, forf[0].dst) == (aa.VOID, aa.GOVERNMENT)
    assert aa.flows_for("deposit", {"amount": 1000.0, "phase": "insufficient"}, {}) == []
    assert aa.move_for("deposit", 1, {"amount": 1000.0, "phase": "forfeit"}, False) == []


def test_move_home_deposit_is_both_a_flow_and_a_stock_move():
    """片方だけに書くと分類器の自己整合が壊れる(この対称性が回帰の要)。"""
    p = {"from": "b1", "to": "b2", "deposit": 30000.0}
    assert aa.flows_for("move_home", p, {})[0].amount == pytest.approx(30000.0)
    assert aa.move_for("move_home", 4, p, False)[0].total == pytest.approx(-30000.0)
    assert "move_home" in aa.MONEY_KINDS and "move_home" in aa.HOUSEHOLD_KINDS


def test_classifier_consistency_detects_an_asymmetric_classifier():
    """flows_for 側だけ金を動かした人工ケースは自己整合が FAIL になる(検出器が生きている)。"""
    flows = [aa.Flow(aa.VOID, aa.HOUSEHOLD, 100.0, "x")]
    assert not aa.classifier_consistency(flows, [])["pass"]
    assert aa.classifier_consistency(flows, [aa.Move(1, cash=100.0)])["pass"]


# ===========================================================================
# A-5. 行政の検査①(内部整合 + 外部整合)
# ===========================================================================
def test_government_internal_consistency_detects_a_broken_ledger():
    ok = [{"step": 0, "level": "ward", "revenue": 0.0, "expense": 0.0, "balance": 100.0},
          {"step": 144, "level": "ward", "revenue": 50.0, "expense": 20.0, "balance": 130.0}]
    bad = [{"step": 0, "level": "ward", "revenue": 0.0, "expense": 0.0, "balance": 100.0},
           {"step": 144, "level": "ward", "revenue": 50.0, "expense": 20.0, "balance": 999.0}]
    assert aa.conserve_government(ok, [], 1e-6)["pass"]
    assert not aa.conserve_government(bad, [], 1e-6)["pass"]


def test_government_external_consistency_uses_the_flow_matrix():
    """行政の残高変化 +30 が、行列の行政純額 (+50 −20) と一致するか。"""
    rows = [{"step": 0, "level": "ward", "revenue": 0.0, "expense": 0.0, "balance": 100.0},
            {"step": 144, "level": "ward", "revenue": 50.0, "expense": 20.0, "balance": 130.0}]
    flows = [aa.Flow(aa.VOID, aa.GOVERNMENT, 50.0, "tax:consumption", step=10),
             aa.Flow(aa.GOVERNMENT, aa.HOUSEHOLD, 20.0, "civic_benefit", step=20)]
    ext = aa.conserve_government(rows, flows, 1e-6)["external"]
    assert ext["applicable"] and ext["pass"] and ext["residual"] == 0.0
    bad = aa.conserve_government(rows, flows[:1], 1e-6)["external"]     # 歳出を1本落とす
    assert not bad["pass"] and bad["residual"] == pytest.approx(-20.0)


# ===========================================================================
# A-6. serve → spend の突合
# ===========================================================================
def _spendrow(step, aid, cat, amount, x=0.0, y=0.0):
    return {"step": step, "agent_id": aid, "cat": cat, "amount": amount, "x": x, "y": y}


def test_serve_match_staffed_uses_customer_key():
    spends = [_spendrow(5, 11, "food", 900.0), _spendrow(5, 12, "food", 800.0)]
    serves = [{"step": 5, "agent_id": 3, "x": 0.0, "y": 0.0,
               "payload": {"cat": "food", "customer": 12, "org_id": "co_a"}}]
    r = aa.match_serve_to_spend(serves, spends)
    assert r["n_matched"] == 1 and r["by_org"]["co_a"] == pytest.approx(800.0)


def test_serve_match_unstaffed_consumes_same_step_same_cat_in_log_order():
    """スタッフ不在 serve は customer を持たず、座標も jitter でずれる → 記録順で対応づける。"""
    spends = [_spendrow(5, 11, "food", 900.0), _spendrow(5, 12, "food", 800.0)]
    serves = [{"step": 5, "agent_id": -1, "x": 999.0, "y": 999.0,
               "payload": {"cat": "food", "unstaffed": True, "org_id": "co_a"}},
              {"step": 5, "agent_id": -1, "x": 999.0, "y": 999.0,
               "payload": {"cat": "food", "unstaffed": True, "org_id": None}}]
    r = aa.match_serve_to_spend(serves, spends)
    assert r["n_matched"] == 2 and r["n_unstaffed"] == 2
    assert r["by_org"]["co_a"] == pytest.approx(900.0)
    assert r["by_org"][aa.UNKNOWN_ORG] == pytest.approx(800.0)      # org 不明を正直に別立て


def test_serve_match_never_double_counts_one_spend():
    """max_serve_per_event>1 で同じ spend に2本 serve が立っても金額は1回だけ。"""
    spends = [_spendrow(5, 11, "food", 900.0)]
    serves = [{"step": 5, "agent_id": 3, "x": 0, "y": 0,
               "payload": {"cat": "food", "customer": 11, "org_id": "co_a"}},
              {"step": 5, "agent_id": 4, "x": 0, "y": 0,
               "payload": {"cat": "food", "customer": 11, "org_id": "co_a"}}]
    r = aa.match_serve_to_spend(serves, spends)
    assert r["n_matched"] == 1 and r["n_dup_serve"] == 1
    assert r["by_org"]["co_a"] == pytest.approx(900.0)


def test_revenue_gap_reports_est_minus_org_resolved_actual():
    ledger = [{"day": 0, "org_id": "co_a", "revenue_est": 18000.0, "wage_paid": 12000.0}]
    spends = [_spendrow(5, 11, "food", 900.0)]
    serves = [{"step": 5, "agent_id": 3, "x": 0, "y": 0,
               "payload": {"cat": "food", "customer": 11, "org_id": "co_a"}}]
    g = aa.revenue_gap(ledger, serves, spends)
    assert g["total_revenue_est"] == pytest.approx(18000.0)
    assert g["known_org_spend"] == pytest.approx(900.0)
    assert g["gap"] == pytest.approx(17100.0)
    assert g["ratio"] == pytest.approx(20.0)


# ===========================================================================
# A-9. 金の経路の追加を検知する装置
# ===========================================================================
def test_unclassified_money_kind_detector_fires_on_a_new_path():
    novel = [{"kind": "mystery_transfer", "payload": {"amount": 500.0}}]
    out = aa.unclassified_money_kinds(novel)
    assert out["mystery_transfer"]["n"] == 1 and out["mystery_transfer"]["keys"] == ["amount"]


def test_unclassified_detector_ignores_known_and_derived_kinds():
    known = [{"kind": "spend", "payload": {"amount": 1.0}},
             {"kind": "ride", "payload": {"fare": 1.0}},          # _spend 経由=派生
             {"kind": "service_use", "payload": {"cost": 1.0}},
             {"kind": "speak", "payload": {"text": "hi"}}]
    assert aa.unclassified_money_kinds(known) == {}


def test_missing_snapshots_reports_unobservable_instead_of_crashing():
    """L3 が無いランでも落ちず、「残高観測 不可」と正直に出す(ゼロ残差と偽らない)。"""
    rep = aa.analyze(_run([_ev(1, 7, "wage", {"amount": 1000.0})], {}))
    c = rep["checks"][aa.HOUSEHOLD]
    assert c["observable"] is False and "l3_snapshots" in c["reason"]
    assert rep["totals"]["money_created"] == pytest.approx(1000.0)   # 検査②は動く
    assert isinstance(aa.render_markdown(rep), str)


def test_read_config_extracts_only_the_toggles_it_needs(tmp_path):
    """runs/<name>/config.yaml の素朴パーサが必要な5トグルを経路で正しく引く。"""
    p = tmp_path / "config.yaml"
    p.write_text(
        "economy:\n  enabled: true\n  accounts:\n    enabled: true\n"
        "government:\n  enabled: false\n"
        "work:\n  service:\n    enabled: true\n    indoor_fields: true\n"
        "    ledger:\n      enabled: false\n"
        "pool:\n  enabled: true\n", encoding="utf-8")
    cfg = aa._read_config(tmp_path)
    assert cfg["accounts_on"] is True
    assert cfg["government_on"] is False
    assert cfg["indoor_fields_on"] is True
    assert cfg["org_ledger_on"] is False
    assert cfg["pool_on"] is True


def test_leak_family_folds_spend_categories_but_keeps_paths_distinct():
    assert aa.leak_family("spend:food") == aa.leak_family("spend:cafe") == "spend"
    assert aa.leak_family("wage:gig") == "wage"
    assert aa.leak_family("venture_setup_cost") == "venture_setup_cost"


# ===========================================================================
# B. 小ラン実測の回帰固定(mock・シム本体ゼロタッチ)
# ===========================================================================
#: **実測から正直に作った既知の漏れ経路**(2026-08-05・第95バッチ)。ゼロと偽らない。
#:   spend  客が払った金の受け取り手が世界に居ない(店舗 POI が会計主体でない)
#:   wage   賃金の出所が居ない(org の残高が減らない。civil/home_refill は除く)
#:   tax    内税・源泉税を「販売側/雇用側」= void が納付している形になる(spend/wage の裏面)
#:   org    org_ledger の revenue_est / wage_paid がどの主体の残高とも接続していない
#:   venture_setup_cost  開業費の受け取り手(内装業者等)が世界に居ない
KNOWN_LEAK_FAMILIES = {"spend", "wage", "tax", "org", "venture_setup_cost"}

#: 既知の漏れ経路(上に加えて、トグル/確率次第で現れるもの。実測 acct_b ラン等で確認済み)。
#:   rent   家賃の受け取り手(家主)が世界に居ない(economy.accounts.enabled=true で出る)
#:   deposit / candidacy  供託金の預かり主体(escrow)が世界に居ない
#:   theft   窃盗は被害者から減るだけで加害者が受け取らない
#:   bankruptcy  破産の資産圧縮は消滅として処理される
#:   rule_bonus  制度DSL の bonus は財源(行政予算)から出ていない
#:   move_home   転居の敷金の受け取り手(家主)が世界に居ない
KNOWN_LEAK_FAMILIES_ALL = KNOWN_LEAK_FAMILIES | {
    "rent", "deposit", "candidacy", "theft", "bankruptcy", "rule_bonus", "move_home"}


@pytest.fixture(scope="module")
def econ_run(tmp_path_factory):
    """mock 小ラン(288 step = 2日・40体)。経済まわりを一通り点火して検査の網羅性を上げる。"""
    from society.config import load_config
    from society.engine.simulation import Simulation

    out = tmp_path_factory.mktemp("acct")
    cfg = load_config([
        "run.seed=20260805", "run.n_agents=40", "run.n_steps=288", "run.name=acct",
        "model.backend=mock", "observer.snapshot_every=12",
        "agents.personas_file=data/personas_80.json",
        "organizations.enabled=true",
        "work.service.enabled=true", "work.service.ledger.enabled=true",
        "work.service.indoor_fields=true",
        "government.enabled=true",
        "commerce.inventory.enabled=true", "commerce.inventory.b2b.enabled=true",
        "economy.fixed_cost_daily=300", "economy.visitor_refresh=true",
    ])
    Simulation(cfg, out_dir=out / "acct").run()
    return aa.analyze(aa.load_run(out / "acct"), rel_tol=1e-4)


@pytest.fixture(scope="module")
def econ_run_accounts(tmp_path_factory):
    """口座 E5 + 銀行 E-W1 を ON にした変種。move_for の現金/口座分岐(card/cash・rent・
    withdraw・interest_paid)を実測で踏む。合計 total が正しいことは検査①が保証する。"""
    from society.config import load_config
    from society.engine.simulation import Simulation

    out = tmp_path_factory.mktemp("acctb")
    cfg = load_config([
        "run.seed=20260805", "run.n_agents=40", "run.n_steps=288", "run.name=acctb",
        "model.backend=mock", "observer.snapshot_every=12",
        "agents.personas_file=data/personas_80.json",
        "organizations.enabled=true",
        "work.service.enabled=true", "work.service.ledger.enabled=true",
        "work.service.indoor_fields=true",
        "government.enabled=true",
        "economy.fixed_cost_daily=300", "economy.visitor_refresh=true",
        "economy.accounts.enabled=true", "economy.accounts.payday_dom=1",
        "economy.bank.enabled=true", "economy.bank.deposit_rate=3.65",
    ])
    Simulation(cfg, out_dir=out / "acctb").run()
    return aa.analyze(aa.load_run(out / "acctb"), rel_tol=1e-4)


def test_measured_household_money_is_conserved_with_accounts_on(econ_run_accounts):
    """口座 ON: 現金/口座の内訳が payload から一意でない種があっても **合計** は保存する。"""
    rep = econ_run_accounts
    assert rep["config"]["accounts_on"]
    c = rep["checks"][aa.HOUSEHOLD]
    assert c["observable"] and c["pass"], \
        f"口座 ON で家計の貨幣保存が破れた: 残差={c['total_residual']}"
    assert rep["classifier_consistency"]["pass"]
    assert rep["unclassified_money_kinds"] == {}
    fams = set(rep["leak_families"])
    assert not (fams - KNOWN_LEAK_FAMILIES_ALL), f"未知の漏れ経路: {sorted(fams - KNOWN_LEAK_FAMILIES_ALL)}"
    assert "rent" in fams, "家賃の漏れ(家主不在)が消えている"
    assert rep["event_counts"].get("interest_paid", 0) > 0, "銀行 ON が点火していない"


def test_measured_household_money_is_conserved(econ_run):
    """検査①: 家計の Δ(現金+口座) = Σフロー。**分類器が金の経路を1本も落としていない**の意。"""
    c = econ_run["checks"][aa.HOUSEHOLD]
    assert c["observable"], "L3 スナップショットが読めていない"
    assert c["total_abs_flow"] > 100_000.0, "実測の流量が小さすぎる(ランが点火していない)"
    assert c["pass"], f"家計の貨幣保存が破れた: 残差={c['total_residual']} 相対={c['relative_residual']}"


def test_measured_government_budget_is_consistent(econ_run):
    gv = econ_run["checks"][aa.GOVERNMENT]
    assert gv["observable"] and gv["pass"]
    assert gv["external"]["applicable"] and gv["external"]["pass"]


def test_measured_classifier_is_self_consistent(econ_run):
    assert econ_run["classifier_consistency"]["pass"]


def test_measured_leak_families_are_closed_under_the_known_list(econ_run):
    """**監視テスト**: 漏れの内訳が既知リストに閉じている。新しい金の経路が増えたら fail。"""
    fams = set(econ_run["leak_families"])
    unknown = fams - KNOWN_LEAK_FAMILIES_ALL
    assert not unknown, (
        f"未知の漏れ経路が増えた: {sorted(unknown)}。"
        "scripts/analyze_accounting.py の分類器と本テストの既知リストを更新すること")
    assert KNOWN_LEAK_FAMILIES <= fams, (
        f"既知の漏れが消えた(=接続されたなら喜ばしい): 不足 {sorted(KNOWN_LEAK_FAMILIES - fams)}")


def test_measured_no_unclassified_money_carrying_kind(econ_run):
    """**金の経路の追加を検知する装置**: 金額キーを持つのに未分類な種が 0 であること。"""
    unc = econ_run["unclassified_money_kinds"]
    assert unc == {}, (
        f"会計検査が見ていない金の経路の候補: {unc}。"
        "MONEY_KINDS / DERIVED_MONEY_KINDS のどちらへ入るかを判断して追加すること")


def test_measured_revenue_est_is_disconnected_from_customer_spend(econ_run):
    """既知の断絶(監査 §3)の実測。**ゼロと偽らず**、乖離が実在することを固定する。"""
    g = econ_run["revenue_gap"]
    assert g["total_revenue_est"] > 0.0, "org_ledger が書かれていない(ledger トグル)"
    assert g["n_serve"] > 0 and g["n_serve_without_spend"] == 0, "serve→spend 突合に穴がある"
    assert g["gap"] > 0.0 and g["ratio"] > 5.0, (
        f"revenue_est と実測 spend の乖離が消えている(ratio={g['ratio']})。"
        "接続が実装されたなら本テストを更新すること")
    # revenue_est = 日給×margin / wage_paid = 日給 なので比は revenue_margin(既定 1.5)に一致する
    assert g["total_revenue_est"] == pytest.approx(g["total_wage_paid"] * 1.5, rel=1e-9)


def test_measured_leak_share_is_large_and_dominated_by_multiple_paths(econ_run):
    """フェーズ2の判断根拠: 漏れは1経路に一意帰着せず、少なくとも3族が主要。"""
    t = econ_run["totals"]
    assert t["leak_share"] > 0.5, "漏れが小さいなら接続の優先度は下がる(前提が変わった)"
    by_fam: dict[str, float] = {}
    for tag, v in econ_run["leaks"].items():
        by_fam[aa.leak_family(tag)] = by_fam.get(aa.leak_family(tag), 0.0) \
            + v["created"] + v["destroyed"]
    top = sorted(by_fam.values(), reverse=True)
    assert len(top) >= 3 and top[0] < sum(top) * 0.8, (
        "漏れが単一経路に一意帰着している(= フェーズ2の接続条件が満たされた)")
