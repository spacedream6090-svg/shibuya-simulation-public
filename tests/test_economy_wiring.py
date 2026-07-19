"""経済深化 E の「配線」テスト(第37バッチ 2026-07-19)= 純ロジックがシミュ本体に接続されたことの検証。

economy.py/commerce.py の純ロジック(消費/決済/銀行/VC・全て既定 OFF)は test_economy_deep.py が
単体検証済み。本ファイルは scheduler.py/tools.py への**配線**(フック)が効くことだけを見る:
OFF=バイト不変 / consumption ON=予算圧縮 / payment ON=method+口座現金分岐 / bank ON=利息・融資→返済→
完済・延滞→破産接続 / VC ON=出資→配当差引き。検証は mock のみ(実LLM 禁止=R1: 追加 generate ゼロ)。

────────────────────────────────────────────────────────────────────────────────────
配線位置の対応表(economy.py 冒頭 E セクション TODO → 実配線箇所 file:line。所有ファイルのみ変更)
────────────────────────────────────────────────────────────────────────────────────
[E-W3 消費 budget_amount]
  - 共通ヘルパ _budget_amount                     src/society/engine/scheduler.py:188
  - _charge_meal の _spend 前に置換              src/society/engine/scheduler.py:582
  - _charge_ride の _spend 前に置換(taxi/bus→transport 写像)  scheduler.py:592
  - _buy_at_ventures の _spend 前に置換(買い手)  src/society/tools.py:1493
[E-W3 決済 choose_payment]
  - _spend で method 選択+cashless=口座/cash=現金 分岐  scheduler.py:515-516(_spend:490)
[E-W1 利息 daily_interest]
  - _phase_daily の居住者ループで account に付与+interest_paid  scheduler.py:2084(_phase_daily:2035)
[E-W1 融資 grant/return]
  - 共通ヘルパ _maybe_loan(遅延 _bank 構築+score+grant+loan_grant)  scheduler.py:205 / _bank:—
  - 現金不足点① 家賃引落                          scheduler.py:1960(_phase_accounts_day:1925)
  - 現金不足点② move_home 敷金                    scheduler.py:2358(_apply_move_home:2348)
  - 現金不足点③ 出店費 _open_venture             src/society/tools.py:732-735(_open_venture:723)
  - 日次返済フェーズ _phase_bank_day(loan_due→repay_installment→loan_repay、
      loan_defaulted→bank.write_off+rent_due へ接続)  scheduler.py:241、run_step 呼出 3171
[E-W2 VC invest/dividend]
  - tools.phase の review_period_days ごと _vc_review(遅延 VCFund+vc_score→vc_candidates→
      invest+入金+vc_investment)  src/society/tools.py:1034(呼出 807、_vc_fund:1026)
  - _buy_at_ventures 売上から collect_dividend を店主取り分から差引き  src/society/tools.py:1501
────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json

from society import commerce, economy
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_NEW_KINDS = ("loan_grant", "loan_repay", "interest_paid", "vc_investment")
_NEUTRAL = {"risk_tolerance": 0.5, "internal_locus": 0.5, "nfc": 0.5}


# --------------------------------------------------------------------- helpers
def _sim(tmp_path, name, n=8, steps=1, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind, agent_id=None):
    return [e for e in sim.logger.events if e.kind == kind
            and (agent_id is None or e.agent_id == agent_id)]


def _enable_accounts(sim, **ov):
    sim.economy["accounts"] = economy.build_accounts_cfg({"enabled": True, **ov})


# ============================================================= OFF=バイト不変(ゴールデン)
def test_off_is_byte_identical_and_no_new_events(tmp_path):
    """4機能 OFF(既定)は決定論で L1 完全一致、配線由来の新イベントも spend.method も一切出ない。

    ゴールデン(tests/test_scenario.py)のバイト一致は別ファイルで担保。ここでは配線が入った状態でも
    OFF が既定 run に無影響(新経路を通らない=新 stream を引かない)ことを固定する。"""
    a = _sim(tmp_path, "off_a", steps=72)
    a.run()
    b = _sim(tmp_path, "off_b", steps=72)
    b.run()
    assert _l1(a) == _l1(b), "OFF 既定 run の決定論が崩れている(配線が OFF で副作用)"
    for k in _NEW_KINDS:
        assert not _kind(a, k), f"OFF なのに {k} が出ている(配線が OFF で発火)"
    assert not [e for e in a.logger.events
                if e.kind == "spend" and "method" in e.payload], \
        "OFF なのに spend payload に method が付いている"


# ============================================================= E-W3 消費(予算圧縮)
def test_consumption_wired_compresses_spend(tmp_path):
    """consumption ON: _charge_ride の運賃が個体の予算制約で圧縮され、spend/ride とも圧縮額で一致。

    OFF は base のまま(バイト不変)。budget_shares 外(venture)は ON でも不変(会計保存)。"""
    sim = _sim(tmp_path, "cons")
    a = sim.agents[0]
    a.loc, a.node, a.visitor = "street", a.node, False
    a.money, a.period_income, a.traits = 100000.0, 0.0, dict(_NEUTRAL)

    # OFF: 運賃は base のまま
    scheduler._charge_ride(sim, a, {"mode": "taxi", "fare": 900.0}, 0, 0)
    off_ride = _kind(sim, "ride", a.id)[-1]
    off_spend = [e for e in _kind(sim, "spend", a.id) if e.payload.get("cat") == "taxi"][-1]
    assert off_ride.payload["fare"] == 900.0 and off_spend.payload["amount"] == 900.0

    # ON: 可処分所得が低い個体は運賃が圧縮される(spend と ride が一致)
    sim.economy["consumption"] = economy.build_consumption_cfg({"enabled": True})
    scheduler._charge_ride(sim, a, {"mode": "taxi", "fare": 900.0}, 1, 10)
    on_ride = _kind(sim, "ride", a.id)[-1]
    on_spend = [e for e in _kind(sim, "spend", a.id) if e.payload.get("cat") == "taxi"][-1]
    assert on_spend.payload["amount"] < 900.0, "consumption ON で運賃が圧縮されていない(配線未接続)"
    assert on_ride.payload["fare"] == on_spend.payload["amount"], "ride.fare と spend.amount が不一致"

    # ヘルパ単体: taxi→transport で圧縮、budget_shares 外(venture)は不変、OFF は恒等
    assert scheduler._budget_amount(sim, a, "taxi", 900.0) < 900.0
    assert scheduler._budget_amount(sim, a, "venture", 900.0) == 900.0
    sim.economy["consumption"] = economy.build_consumption_cfg(None)   # OFF
    assert scheduler._budget_amount(sim, a, "taxi", 900.0) == 900.0


# ============================================================= E-W3 決済(method+口座現金分岐)
def test_payment_wired_method_and_pot_branch(tmp_path):
    """payment ON: spend payload に method が出現し、cashless=口座引落 / cash=現金引落 に分岐する。"""
    sim = _sim(tmp_path, "pay")
    _enable_accounts(sim)
    a = sim.agents[0]
    a.traits, a.account, a.money = dict(_NEUTRAL), 100000.0, 50000.0

    # cashless 強制(cashless_prob=1・pref_slope=0・small=0 → p=1) → 口座から引落
    sim.economy["payment"] = economy.build_payment_cfg(
        {"enabled": True, "cashless_prob": 1.0, "pref_slope": 0.0, "small_amount": 0.0})
    scheduler._spend(sim, a, 1000.0, "food", 0, 0)
    e = _kind(sim, "spend", a.id)[-1]
    assert e.payload["method"] == "cashless", "cashless が選ばれていない(method 未配線)"
    assert e.payload["src"] == "card" and a.account == 99000.0, "cashless で口座から引落されていない"
    assert a.money == 50000.0, "cashless なのに現金が減っている"

    # cash 強制(cashless_prob=0・pref_slope=0・large_boost=0 → p=0) → 現金から引落
    sim.economy["payment"] = economy.build_payment_cfg(
        {"enabled": True, "cashless_prob": 0.0, "pref_slope": 0.0, "large_boost": 0.0})
    acc0 = a.account
    scheduler._spend(sim, a, 1000.0, "food", 1, 10)
    e = _kind(sim, "spend", a.id)[-1]
    assert e.payload["method"] == "cash", "cash が選ばれていない"
    assert e.payload["src"] == "cash" and a.money == 49000.0, "cash で現金から引落されていない"
    assert a.account == acc0, "cash なのに口座が減っている"


# ============================================================= E-W1 銀行(利息)
def test_bank_wired_interest_accrues(tmp_path):
    """bank ON: 居住者の口座残高に日次利息が付与され interest_paid が出る(_phase_daily 配線)。"""
    sim = _sim(tmp_path, "int", **{"economy.wages.自営": 0})
    _enable_accounts(sim)
    sim.economy["bank"] = economy.build_bank_cfg({"enabled": True, "deposit_rate": 3.65})  # 日利1%
    a = sim.agents[0]
    a.loc, a.visitor, a.account, a.money = "street", False, 100000.0, 5000.0
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    ip = _kind(sim, "interest_paid", a.id)
    assert ip, "bank ON で interest_paid が出ていない(利息未配線)"
    assert abs(ip[-1].payload["amount"] - 1000.0) < 1e-6, "日次利息(残高×日利1%)が一致しない"
    assert a.account >= 101000.0, "利息が口座に加算されていない"


# ============================================================= E-W1 銀行(融資→返済→完済)
def test_bank_wired_loan_grant_and_repay_completes(tmp_path):
    """bank ON: 現金不足で融資(loan_grant・口座入金)→ 日次返済フェーズで完済(loan_repay complete)。"""
    sim = _sim(tmp_path, "loan")
    _enable_accounts(sim)
    sim.economy["bank"] = economy.build_bank_cfg(
        {"enabled": True, "term_days": 3, "installment_days": 1})   # 3 回で完済
    a = sim.agents[0]
    a.visitor, a.period_income, a.money, a.account = False, 150000.0, 0.0, 0.0

    got = scheduler._maybe_loan(sim, a, 30000.0, step=0, sim_min=0)
    assert got == 30000.0 and a.account == 30000.0, "融資が口座に入金されていない"
    lg = _kind(sim, "loan_grant", a.id)
    assert lg and lg[-1].payload["amount"] == 30000.0, "loan_grant が出ていない(融資未配線)"
    assert a.id in sim.bank.loans, "Bank に融資が登録されていない"

    a.account = 100000.0                             # 返済原資は十分
    for d in range(1, 5):                            # 返済期日(day 1,2,3)を跨いで駆動
        sim._bank_day = -1
        scheduler._phase_bank_day(sim, step=d * 144, sim_min=d * 1440)
    lr = _kind(sim, "loan_repay", a.id)
    assert lr and lr[-1].payload["status"] == "complete", "完済(loan_repay complete)に至っていない"
    assert a.id not in sim.bank.loans, "完済したのに融資が残っている"
    assert a.account < 100000.0, "返済で口座が減っていない"


def test_bank_wired_loan_at_rent_shortage(tmp_path):
    """現金不足点①(家賃引落): 口座が家賃に足りないとき融資が実行され、家賃が滞納にならない。"""
    sim = _sim(tmp_path, "rentloan", **{"economy.wages.自営": 0})
    _enable_accounts(sim, payday_dom=25, rent_share=0.30)
    sim.economy["bank"] = economy.build_bank_cfg({"enabled": True})
    a = sim.agents[0]
    a.visitor, a.period_income, a.money, a.account, a.rent_due = False, 150000.0, 0.0, 0.0, 0.0
    # rent_dom = payday%30+1 = 26 → block_day%30==25 → block_day=25 → step=25*144
    sim._acct_day = -1
    scheduler._phase_accounts_day(sim, step=25 * 144, sim_min=25 * 1440)
    assert _kind(sim, "loan_grant", a.id), "家賃引落の現金不足点で融資が実行されていない"
    assert a.rent_due == 0.0, "融資で家賃を完済できていない(滞納が残る)"


# ============================================================= E-W1 銀行(延滞→破産接続)
def test_bank_wired_default_connects_bankruptcy(tmp_path):
    """延滞が閾値到達 → bank.write_off(貸倒・premium 引上げ)+ 未回収残を rent_due へ接続。"""
    sim = _sim(tmp_path, "default")
    _enable_accounts(sim)
    sim.economy["bank"] = economy.build_bank_cfg(
        {"enabled": True, "term_days": 3, "installment_days": 1, "default_arrears_days": 2})
    a = sim.agents[0]
    a.visitor, a.period_income, a.money, a.account, a.rent_due = False, 150000.0, 0.0, 0.0, 0.0
    scheduler._maybe_loan(sim, a, 30000.0, step=0, sim_min=0)
    a.account = 0.0                                  # 返済原資なし=延滞し続ける
    for d in range(1, 4):                            # day 1(延滞1)→ day 2(延滞2=default)
        sim._bank_day = -1
        scheduler._phase_bank_day(sim, step=d * 144, sim_min=d * 1440)
    dflt = [e for e in _kind(sim, "loan_repay", a.id)
            if e.payload.get("status") == "defaulted"]
    assert dflt, "延滞が続いても loan_defaulted(貸倒)に接続していない"
    assert a.id not in sim.bank.loans, "貸倒後も融資が残っている"
    assert a.rent_due > 0.0, "未回収残が家賃滞納(rent_due)へ接続されていない(破産サイクル未接続)"
    assert sim.bank.write_offs > 0.0 and sim.bank.premium > 0.0, "貸倒計上/信用引締めが起きていない"


# ============================================================= E-W2 VC(出資→配当差引き)
def test_vc_wired_invest_and_dividend(tmp_path):
    """VC ON: _vc_review で開店中 venture に出資(vc_investment・口座入金)→ 以後の売上から配当を差引き。"""
    sim = _sim(tmp_path, "vc")
    _enable_accounts(sim)
    sim.economy["vc"] = economy.build_vc_cfg({"enabled": True, "review_period_days": 1})
    tools = sim.tools
    owner = sim.agents[0]
    owner.account, owner.money = 0.0, 0.0
    node = owner.node
    for ag in sim.agents:                            # 在館数(market 代理)を稼ぐため全員を同ノードへ
        ag.loc, ag.node, ag.sleeping = "street", node, False
    v = {"owner": owner.id, "node": node, "name": "屋台", "offer": "", "price": 800.0,
         "opened_step": 0, "last_sale_step": 0, "open_at": 0, "permitted": True,
         "sales_total": 8000.0, "fulltime": False}
    tools.ventures[owner.id] = v
    tools.ventures_by_node[node].append(v)

    sim._vc_review_day = -1
    tools._vc_review(sim, step=0, sim_min=0)
    inv = _kind(sim, "vc_investment", owner.id)
    assert inv, "VC ON で vc_investment が出ていない(出資未配線)"
    ticket = sim.economy["vc"]["ticket"]
    assert owner.account == ticket, "出資額が owner.account に入金されていない"
    assert abs(sim.vc_fund.equity[owner.id] - sim.economy["vc"]["equity_share"]) < 1e-9

    # 配当: 買い手が購入 → 店主の取り分から配当(売上×持分×dividend_rate=800×0.2×0.5=80)を差引き
    tools.cfg["buy_prob"] = 1.0                       # 購入抽選を必ず通す
    buyer = sim.agents[1]
    buyer.money, buyer.account, buyer.node = 10000.0, 0.0, node
    acc0 = owner.account
    tools._buy_at_ventures(sim, buyer, step=1, sim_min=10)
    sale = _kind(sim, "venture_sale", owner.id)[-1]
    assert sale.payload.get("dividend") == 80.0, "売上から VC 配当が差し引かれていない(配当未配線)"
    assert abs((owner.account - acc0) - 720.0) < 1e-6, "店主の取り分が(売上−配当)になっていない"


# ============================================================= 決定論(ON 同士2回一致)
def test_wired_on_deterministic(tmp_path):
    """全機能 ON 同士 2 回の L1 完全一致(配線が決定論=新 stream 'payment' 以外の乱数を足さない)。"""
    def _run_on(name):
        sim = _sim(tmp_path, name, steps=72, **{"economy.accounts.enabled": "true"})
        sim.economy["consumption"] = economy.build_consumption_cfg({"enabled": True})
        sim.economy["payment"] = economy.build_payment_cfg({"enabled": True})
        sim.economy["bank"] = economy.build_bank_cfg({"enabled": True})
        sim.economy["vc"] = economy.build_vc_cfg({"enabled": True})
        sim.run()
        return sim
    assert _l1(_run_on("on_a")) == _l1(_run_on("on_b")), "ON の決定論が崩れている"
