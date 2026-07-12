"""E5: 口座(銀行)概念(既定 OFF)のテスト。

OFF 不変 / 初期分割(口座8割・現金2割)/ 給料日まとめ支給 / 家賃引き落とし(繰越)/
カード・現金の使い分け / 現金不足→自動引き出し / 逼迫判定=合算 / resume 跨ぎ一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1", "economy.accounts.enabled=true"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _resident(sim):
    return next(a for a in sim.agents if not a.visitor)


# --------------------------------------------------------------------------- #
def test_off_is_byte_identical(tmp_path):
    """既定 OFF と明示 OFF で L1 が完全一致、かつ account=0・money 未分割。"""
    def l1(name, **ov):
        dot = ["run.seed=42", "run.n_agents=15", "run.n_steps=40", f"run.name={name}"]
        dot += [f"{k}={v}" for k, v in ov.items()]
        sim = Simulation(load_config(dot), out_dir=tmp_path / name)
        sim.run()
        return sim
    a = l1("acc_def")                                         # 既定(accounts off)
    b = l1("acc_off", **{"economy.accounts.enabled": "false"})  # 明示 off
    la = [(e.step, e.agent_id, e.kind, json.dumps(e.payload, sort_keys=True))
          for e in a.logger.events]
    lb = [(e.step, e.agent_id, e.kind, json.dumps(e.payload, sort_keys=True))
          for e in b.logger.events]
    assert la == lb
    assert all(x.account == 0.0 for x in a.agents)


def test_initial_split_residents_only(tmp_path):
    """ON: 居住者は口座8割/現金2割に分割。来街者は口座を持たない(account=0)。"""
    sim = _sim(tmp_path, "split")
    res = [a for a in sim.agents if not a.visitor and (a.money + a.account) > 0]
    assert res, "居住者が居ない(地図/名簿の前提崩れ)"
    for a in res:
        total = a.money + a.account
        assert abs(a.account - round(total * 0.8)) <= 1.0      # 口座8割
        assert a.account > a.money                             # 大半は口座
    assert all(a.account == 0.0 for a in sim.agents if a.visitor)


def test_wage_goes_to_account(tmp_path):
    """賃金は口座へ入金(現金は不変)。payload に "to":"account"。"""
    sim = _sim(tmp_path, "wage_acc")
    a = _resident(sim)
    cash0, acc0 = a.money, a.account
    scheduler._pay_wage(sim, a, 5000.0, step=0, sim_min=420)
    assert a.account == acc0 + 5000.0 and a.money == cash0
    w = [e for e in sim.logger.events if e.kind == "wage" and e.agent_id == a.id][-1]
    assert w.payload["to"] == "account"
    assert w.payload["account"] == round(a.account, 1)


def test_card_vs_cash_and_withdraw(tmp_path):
    """支払い: ≥card_threshold は口座(カード)、未満は現金、現金不足は ATM 引き出し。"""
    sim = _sim(tmp_path, "pay")
    a = _resident(sim)
    a.account, a.money = 100000.0, 5000.0
    scheduler._spend(sim, a, 4000.0, "shop", step=1, sim_min=420)   # ≥3000 → カード
    assert a.account == 96000.0 and a.money == 5000.0
    sp = [e for e in sim.logger.events if e.kind == "spend"][-1]
    assert sp.payload["src"] == "card"

    scheduler._spend(sim, a, 900.0, "food", step=2, sim_min=430)    # <3000 現金足りる
    assert a.money == 4100.0
    assert [e for e in sim.logger.events if e.kind == "spend"][-1].payload["src"] == "cash"
    assert not [e for e in sim.logger.events if e.kind == "withdraw"]

    a.money, a.account = 100.0, 50000.0
    scheduler._spend(sim, a, 900.0, "food", step=3, sim_min=440)    # 現金不足 → 引き出し
    wd = [e for e in sim.logger.events if e.kind == "withdraw"]
    assert wd, "現金不足で withdraw が出ていない"
    assert wd[-1].payload["amount"] == 20000.0                      # atm_withdraw 基本額
    assert a.money >= 0.0 and a.account == 30000.0                  # 50000-20000
    assert abs(a.money - (100.0 + 20000.0 - 900.0)) < 1e-6


def test_payday_monthly_lump_sum(tmp_path):
    """給料日: 月給者は economy.wages×勤務日数をまとめて口座へ支給し work_days をリセット。"""
    sim = _sim(tmp_path, "payday", **{"economy.accounts.payday_dom": 1})
    a = _resident(sim)
    a.wage, a.work_days, a.account, a.period_income = 12000.0, 20, 1000.0, 0.0
    sim._acct_day = -1
    scheduler._phase_accounts_day(sim, step=0, sim_min=420)          # block_day 0 → dom 1
    assert a.account == 1000.0 + 12000.0 * 20
    assert a.work_days == 0 and a.last_salary == 240000.0
    sal = [e for e in sim.logger.events
           if e.kind == "wage" and e.payload.get("source") == "salary"]
    assert sal and sal[-1].payload["amount"] == 240000.0


def test_rent_charged_day_after_payday(tmp_path):
    """給料日翌日: 家賃=月収相当×rent_share を口座から引き落とし。残高不足は繰越。"""
    sim = _sim(tmp_path, "rent", **{"economy.accounts.payday_dom": 1})
    a = _resident(sim)
    # 家賃日 = payday(1) の翌日 = dom 2 = block_day 1(step 144)
    a.period_income, a.account, a.rent_due = 100000.0, 50000.0, 0.0
    sim._acct_day = 0
    scheduler._phase_accounts_day(sim, step=144, sim_min=420 + 144 * 10)
    assert a.account == 20000.0                                      # 50000 - 100000*0.3
    assert a.period_income == 0.0 and a.rent_due == 0.0
    rent = [e for e in sim.logger.events if e.kind == "rent"]
    assert rent and rent[-1].payload["amount"] == 30000.0

    # 残高不足 → 未払いを繰越(rent_due>0)
    b = next(x for x in sim.agents if x.id != a.id and not x.visitor)
    b.period_income, b.account, b.rent_due = 100000.0, 10000.0, 0.0
    sim._acct_day = 0
    scheduler._phase_accounts_day(sim, step=144, sim_min=420 + 144 * 10)
    assert b.account == 0.0 and abs(b.rent_due - 20000.0) < 1e-6     # 30000-10000 繰越


def test_money_pressure_uses_combined_balance(tmp_path):
    """逼迫判定は現金+口座の合算(口座が潤沢なら現金が薄くても逼迫しない)。"""
    sim = _sim(tmp_path, "press", steps=1)
    a = _resident(sim)
    a.money, a.account = 100.0, 50000.0
    before = a.states["grievance"]
    sim._econ_day, sim._acct_day = -1, -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    ups = [e for e in sim.logger.events
           if e.kind == "state_update" and e.agent_id == a.id
           and e.payload["cause"] == "money_pressure"]
    assert not ups and a.states["grievance"] == before


def test_resume_matches_straight_with_accounts(tmp_path):
    """口座 ON で「一気 160step」と「80+resume」の L1/L2/L3 が完全一致(状態が跨いで復元)。"""
    ov = {"run.seed": 42, "run.n_agents": 20, "model.backend": "mock",
          "economy.accounts.enabled": "true", "economy.accounts.payday_dom": 2}

    def cfg(name, n_steps, **extra):
        dot = [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        dot += [f"run.n_steps={n_steps}", f"run.name={name}"]
        return load_config(dot)

    def rows(run_dir, stem):
        return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()

    straight = tmp_path / "acc_straight"
    Simulation(cfg("acc_straight", 150), out_dir=straight).run()   # 150>144=暦日境界を跨ぐ

    resumed = tmp_path / "acc_resume"
    every = {"observer.checkpoint_every": 80}
    sim1 = Simulation(cfg("acc_resume", 80, **every), out_dir=resumed)
    for step in range(80):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 80, resumed / "checkpoint" / "ckpt-000080.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(cfg("acc_resume", 150, **every), out_dir=resumed).run(resume_from=resumed)

    assert rows(straight, "l1_events") == rows(resumed, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert rows(straight, stem) == rows(resumed, stem), f"{stem} 不一致"
