"""経済深化 E(第37バッチ 2026-07-19)= E-W3 消費/決済・E-W1 銀行・E-W2 VC のテスト。

方針(既存の鉄則を継承。docs/research/economy-abm-research.md §3/§5/§6/§7):
- OFF(既定): consumption/payment/bank/vc とも enabled=false。配線前なので loan_grant/
  loan_repay/interest_paid/vc_investment は1件も出ず、既定 run は決定論(2回で L1 完全一致)。
  = ゴールデン(tests/test_scenario.py)バイト同一・test_economy.py 不変 の土台。
- E-W3 消費: consumption_profile が traits/所得から個体別の費目配分と貯蓄率を決定論導出
  (エンゲル: 所得↑で食料↓)。budget_amount が可処分所得で支出を絞る(予算制約)。
- E-W3 決済: choose_payment が §6 経産省2024 のキャッシュレス比 42.8% へ較正(大数で ±5%)。
  金額帯(card_threshold)で大額=カード寄り・少額=現金寄り(既存閾値ロジックと整合)。
- E-W1 銀行: credit_score の承認/拒否境界、金利、返済スケジュール完済、延滞→破産接続。
- E-W2 VC: vc_score の判定数値例(通過/拒否)、出資と持分、以後の売上からの配当。
検証は mock のみ(実LLM 禁止=R1: 追加 generate ゼロ)。
"""
from __future__ import annotations

import json

import numpy as np

from society import commerce, economy
from society.config import load_config
from society.engine.simulation import Simulation


# --------------------------------------------------------------------- helpers
def _run(tmp_path, name, n=15, steps=72, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


_NEUTRAL = {"risk_tolerance": 0.5, "internal_locus": 0.5, "nfc": 0.5}


# ===================================================================== OFF 不変
def test_off_defaults_and_no_new_events(tmp_path):
    """既定は4機能とも OFF。配線前=新イベント 0 件、既定 run は決定論(L1 完全一致=バイト一致)。"""
    econ = economy.build_economy(None)
    for k in ("consumption", "payment", "bank", "vc"):
        assert econ[k]["enabled"] is False, f"{k} が既定 OFF でない(バイト一致を破る)"

    a = _run(tmp_path, "off_a")
    b = _run(tmp_path, "off_b")
    assert _l1(a) == _l1(b), "既定 run の決定論が崩れている"
    for k in ("loan_grant", "loan_repay", "interest_paid", "vc_investment"):
        assert not _kind(a, k), f"配線前なのに {k} が出ている(OFF 不変違反)"


# ============================================================= E-W3 消費(配分・予算)
def test_consumption_profile_individual_difference(tmp_path):
    """個体差: risk_tolerance 高=discretionary(nightlife)厚め・食料薄め・貯蓄率低(決定論)。"""
    cfg = economy.build_consumption_cfg({"enabled": True})
    inc = cfg["income_ref"]
    hi = economy.consumption_profile({**_NEUTRAL, "risk_tolerance": 0.9}, inc, cfg)
    lo = economy.consumption_profile({**_NEUTRAL, "risk_tolerance": 0.1}, inc, cfg)
    assert hi["shares"]["nightlife"] > lo["shares"]["nightlife"], "高 risk が discretionary 厚めでない"
    assert hi["shares"]["food"] < lo["shares"]["food"], "高 risk が食料薄めでない"
    assert hi["savings_rate"] < lo["savings_rate"], "高 risk の貯蓄率が低くない"
    # 個体差が実在する(同一プロファイルでない)
    assert hi["shares"] != lo["shares"]


def test_consumption_engel_law(tmp_path):
    """エンゲルの法則: 所得が低い個体ほど食料シェアが厚い(engel_income_slope<0)。"""
    cfg = economy.build_consumption_cfg({"enabled": True})
    low = economy.consumption_profile(_NEUTRAL, 0.5 * cfg["income_ref"], cfg)
    high = economy.consumption_profile(_NEUTRAL, 1.5 * cfg["income_ref"], cfg)
    assert low["shares"]["food"] > high["shares"]["food"], "低所得ほど食料シェアが厚くない(エンゲル)"


def test_budget_constraint_scales_with_disposable(tmp_path):
    """予算制約: 可処分所得が低いほど支出額が下がる。高所得は income_factor=1 で base×個体ウェイト。"""
    cfg = economy.build_consumption_cfg({"enabled": True})
    prof = economy.consumption_profile(_NEUTRAL, cfg["income_ref"], cfg)
    lo = economy.budget_amount(prof, "food", 900.0, income=50000.0, fixed_cost=0.0, cfg=cfg)
    hi = economy.budget_amount(prof, "food", 900.0, income=300000.0, fixed_cost=0.0, cfg=cfg)
    assert lo < hi, "低所得で支出が絞られていない(予算制約が効いていない)"
    # 高所得(可処分 ≥ income_ref)は income_factor=1 → base×(shares/base_shares)
    mult = prof["shares"]["food"] / cfg["budget_shares"]["food"]
    assert abs(hi - 900.0 * mult) < 1e-6, "高所得の支出が base×個体ウェイトに一致しない"
    # budget_shares 外のカテゴリ(cafe 等)は base_amount のまま(不変)
    assert economy.budget_amount(prof, "cafe", 500.0, cfg["income_ref"], 0.0, cfg) == 500.0


# ============================================================= E-W3 決済(確率較正)
def test_payment_cashless_calibration_42_8(tmp_path):
    """大数で cashless 比 ≈ 42.8%(§6 経産省2024)±5%。中間帯・中立嗜好の 1 draw で検証。"""
    cfg = economy.build_payment_cfg({"enabled": True})
    rng = np.random.default_rng(2026)
    n = 20000
    cashless = sum(1 for _ in range(n)
                   if economy.choose_payment(1000.0, 0.5, rng, cfg) == "cashless")
    frac = cashless / n
    assert abs(frac - 0.428) <= 0.05, f"キャッシュレス比が 42.8%±5% を外れる: {frac:.3f}"


def test_payment_threshold_consistency(tmp_path):
    """既存カード/現金閾値と整合: 大額 ≥ card_threshold=カード寄り(cashless↑)、少額=現金寄り(↓)。"""
    cfg = economy.build_payment_cfg({"enabled": True})
    p_large = economy.payment_p_cashless(4000.0, 0.5, cfg)   # ≥ 3000 → カード寄り
    p_mid = economy.payment_p_cashless(1000.0, 0.5, cfg)     # 中間帯 → 基準
    p_small = economy.payment_p_cashless(300.0, 0.5, cfg)    # < 500 → 現金寄り
    assert p_large > p_mid > p_small, f"金額帯の単調性が崩れている: {p_large}/{p_mid}/{p_small}"
    assert abs(p_mid - cfg["cashless_prob"]) < 1e-9, "中間帯・中立で基準(42.8%)にならない"
    # 個体嗜好: cashless 好き(pref=1)>中立>現金好き(pref=0)
    assert (economy.payment_p_cashless(1000.0, 1.0, cfg)
            > p_mid > economy.payment_p_cashless(1000.0, 0.0, cfg))


def test_payment_stream_deterministic(tmp_path):
    """同 seed の payment stream で method 列が一致(決定論・新 stream で既存 draw を乱さない)。"""
    cfg = economy.build_payment_cfg({"enabled": True})
    seq_a = [economy.choose_payment(1200.0, 0.5, np.random.default_rng(7), cfg)]
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = [economy.choose_payment(1200.0, 0.5, r1, cfg) for _ in range(50)]
    b = [economy.choose_payment(1200.0, 0.5, r2, cfg) for _ in range(50)]
    assert a == b, "同 seed で決済手段の列が一致しない(非決定論)"
    assert seq_a  # 使用(lint)


# ============================================================= E-W1 銀行(与信・返済)
def test_credit_score_approve_reject_boundary(tmp_path):
    """与信スコアの境界: 高所得・高資産・良好返済=承認、低所得・高延滞=拒否(客観条件のみ)。"""
    cfg = economy.build_bank_cfg({"enabled": True})
    strong = economy.credit_score(income=200000, assets=500000, repayment=1.0,
                                  arrears_days=0, has_group=False, cfg=cfg)
    weak = economy.credit_score(income=20000, assets=10000, repayment=0.0,
                                arrears_days=60, has_group=False, cfg=cfg)
    assert economy.loan_approved(strong, cfg), "健全な借り手が承認されない"
    assert not economy.loan_approved(weak, cfg), "延滞・低資産の借り手が拒否されない"
    # 数値例(§3): strong = 0.4+0.3+0.2 = 0.9
    assert abs(strong - 0.9) < 1e-9, f"strong スコアの数値例が違う: {strong}"


def test_credit_group_guarantee_and_rate(tmp_path):
    """連帯保証グループ(§3 社会的担保)でスコア加点。金利は低スコアで上昇・満点で base_rate。"""
    cfg = economy.build_bank_cfg({"enabled": True})
    base = economy.credit_score(100000, 100000, 0.5, 0, has_group=False, cfg=cfg)
    grouped = economy.credit_score(100000, 100000, 0.5, 0, has_group=True, cfg=cfg)
    assert abs((grouped - base) - cfg["group_guarantee_bonus"]) < 1e-9, "グループ加点が効いていない"
    assert economy.loan_rate(0.2, cfg) > economy.loan_rate(0.9, cfg), "低スコアで金利が上がらない"
    assert abs(economy.loan_rate(1.0, cfg) - cfg["base_rate"]) < 1e-9, "満点で base_rate にならない"
    assert economy.loan_limit(150000, cfg) == 150000 * cfg["max_loan_ratio"]


def test_loan_repayment_schedule_completes(tmp_path):
    """返済スケジュール: 残高十分なら term_days/installment_days 回で完済(remaining→0)。"""
    cfg = economy.build_bank_cfg({"enabled": True})   # term 90 / installment 30 → 3 回
    loan = economy.open_loan(90000.0, score=0.9, day=0, cfg=cfg)
    assert loan["n_installments"] == 3
    assert abs(loan["total_due"] - loan["per"] * 3) < 1e-6
    statuses = []
    for day in range(0, 200):
        if economy.loan_due(loan, day):
            pay, st = economy.repay_installment(loan, balance=1e9, day=day)
            statuses.append(st)
            assert pay > 0.0
    assert loan["remaining"] <= 1e-9 and loan["paid_installments"] == 3, "完済していない"
    assert statuses[-1] == "complete" and statuses.count("complete") == 1
    assert not economy.loan_defaulted(loan, cfg), "完済したのに延滞判定"


def test_loan_arrears_connects_to_bankruptcy(tmp_path):
    """延滞→破産接続: 残高 0 で返済不能が続くと arrears_days が閾値到達→loan_defaulted=True。"""
    cfg = economy.build_bank_cfg({"enabled": True, "default_arrears_days": 30})
    loan = economy.open_loan(90000.0, score=0.9, day=0, cfg=cfg)
    defaulted_day = None
    for day in range(0, 400):
        if economy.loan_due(loan, day):
            pay, st = economy.repay_installment(loan, balance=0.0, day=day)
            assert pay == 0.0 and st == "arrears"
        if economy.loan_defaulted(loan, cfg):
            defaulted_day = day
            break
    assert defaulted_day is not None, "延滞が続いても破産接続(loan_defaulted)にならない"
    assert loan["remaining"] > 0.0, "未返済のはずが remaining が 0"
    # Bank 会計: 貸倒処理で loans から除去・premium 引上げ(§1 信用引締め)
    bank = economy.Bank(cfg)
    bank.grant(agent_id=1, principal=90000.0, score=0.9, day=0)
    before = bank.premium
    loss = bank.write_off(1)
    assert loss > 0.0 and 1 not in bank.loans and bank.premium > before


# ================================================================= E-W2 VC(出資・配当)
def test_vc_score_pass_and_reject(tmp_path):
    """VC スコアの数値例(§4 観測量のみ): 好調 venture は閾値超、低調は未満。"""
    cfg = economy.build_vc_cfg({"enabled": True})
    strong = commerce.vc_score(sales_total=8000, occ=6, network_degree=10, cfg=cfg)
    weak = commerce.vc_score(sales_total=1000, occ=1, network_degree=2, cfg=cfg)
    assert abs(strong - 1.0) < 1e-9, f"満点 venture のスコアが 1.0 でない: {strong}"
    assert strong >= cfg["threshold"] and weak < cfg["threshold"], "閾値の通過/拒否が逆"
    # 候補選抜: 閾値超を上位 n_deals 件(決定論・同点キー昇順)
    picks = commerce.vc_candidates([(1, strong), (2, weak), (3, 0.7)], cfg)
    assert picks == [(1, 1.0)], f"上位 1 件の選抜が違う: {picks}"


def test_vc_fund_invest_and_dividend(tmp_path):
    """出資: ticket を支出・持分を記録。以後の売上から配当(売上×持分×dividend_rate)を差引き。"""
    cfg = economy.build_vc_cfg({"enabled": True})
    fund = commerce.VCFund(cfg)
    assert fund.can_invest()
    ticket = fund.invest(owner_id=1)
    assert ticket == cfg["ticket"]
    assert fund.balance == cfg["fund_initial"] - cfg["ticket"]
    assert fund.equity[1] == cfg["equity_share"]
    # 配当の数値例: 売上 800 × 持分 0.2 × dividend_rate 0.5 = 80
    d = fund.collect_dividend(owner_id=1, sale_amount=800.0)
    assert abs(d - 80.0) < 1e-9, f"配当の数値例が違う: {d}"
    assert abs(commerce.vc_dividend(800.0, 0.2, cfg) - 80.0) < 1e-9
    # 非出資先からは配当を取らない
    assert fund.collect_dividend(owner_id=2, sale_amount=800.0) == 0.0


def test_vc_fund_depletion_stops_investing(tmp_path):
    """ファンド原資が枯れると出資停止(§4 希少性=競争)。"""
    cfg = economy.build_vc_cfg({"enabled": True, "fund_initial": 50000, "ticket": 50000})
    fund = commerce.VCFund(cfg)
    assert fund.can_invest()
    fund.invest(owner_id=1)
    assert not fund.can_invest(), "原資が枯れても出資可能のまま"
