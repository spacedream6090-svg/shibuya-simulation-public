"""レーンE(第117)= **経済の正しさ**と**観測のエンジン分離**の検収。

正典: docs/plans/codex-review-triage.md の B6 / B7 / B8 / A9 / B9

  B6 消費税を**実支払**基準へ(名目課税だと床クリップ時に受け手の net が負に落ちる)
  B7 死者(dead=True)を経済 5 フェーズから除外(★loc=="outside" は除外条件にしない)
  B8 出生時の世帯二重所属を塞ぐ(別世帯の夫婦は出生を受理しない)
  A9 provenance の伝播回数をプロンプトから切り離す(transmissions_count 併設)
  B9 starvation 観測の agent 属性書き込みを observer 側へ

house style は tests/test_org_accounting.py / tests/test_dph.py に倣う。
どのテストも **mock backend のみ**(実 LLM を 1 度も呼ばない)。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from society import economy_sfc as SFC                 # noqa: E402
from society import population as P                    # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine import checkpoint, scheduler       # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402
from society.observer import starvation as SV          # noqa: E402
from society.observer.provenance import Item, ItemStore  # noqa: E402

STARV_KINDS = ("reply_dropped", "plan_skipped", "reflect_dropped")


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_dph.py / test_org_accounting.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=12, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# =========================================================================== #
# B6  消費税を実支払(actual)基準へ
# =========================================================================== #
#: 消費税と org 会計の両方を点火する(= 課税と受け手配分の両側が同時に動く)。
B6_ON = {"government.enabled": "true", "economy.org_accounting.enabled": "true"}


def _gov_of(sim):
    """行政主体(scheduler が遅延構築する。simulation.py は編集不可のため)。"""
    return scheduler._gov(sim)


def _gov_total(sim) -> float:
    return sum(_gov_of(sim).balance.values())


def _tax_of(sim, base: float, cat: str) -> float:
    """行政が定める内税額(government.py:182 の式そのもの)= 比較の基準。"""
    national, local, _rate = _gov_of(sim).consumption_tax(float(base), str(cat))
    return float(national) + float(local)


def _yen_eq(a: float, b: float) -> bool:
    """**1 円単位で同値**か。行政の残高は 3 主体(桁の違う初期残高)へ分けて加算されるので、
    合計の最終桁は加算順で 0.5 ulp 動く。1e-6 は 1 円の 100 万分の 1 = 「1 円単位の同値」を
    主張するには十分に厳しく、浮動小数の丸めによる偽陽性だけを落とす。"""
    return abs(float(a) - float(b)) < 1e-6


def _row_out_total(sim) -> float:
    """RoW へ出た累計(受け手 org が解決できない消費はここへ入る)。"""
    st = SFC.state_of(sim) or {}
    return sum(float(v.get("out", 0.0)) for v in (st.get("row") or {}).values())


def _spend_probe(tmp_path, name, *, money: float, nominal: float, cat: str = "shop"):
    """1 件の消費を実行し、家計 / 行政 / 受け手の増減を返す。

    org 台帳を持たない小ラン(organizations OFF)なので受け手は必ず RoW
    (``unknown_payee``)= 受け手側の増減が ``row.out`` に丸ごと出る。
    """
    sim = _sim(tmp_path, name, n_steps=1, n_agents=8, **B6_ON)
    agent = sim.agents[0]
    agent.money = float(money)
    agent.account = 0.0
    agent.building = ""                       # resolve_payee を必ず外す(= RoW 受け手)
    agent.node = ""
    gov0 = _gov_total(sim)
    row0 = _row_out_total(sim)
    n0 = len(sim.logger.events)
    scheduler._spend(sim, agent, float(nominal), cat, step=0, sim_min=0)
    spend_ev = [e for e in sim.logger.events[n0:] if e.kind == "spend"]
    return {
        "sim": sim, "agent": agent,
        "actual": float(money) - agent.money,      # 家計から実際に減った額
        "d_gov": _gov_total(sim) - gov0,
        "d_row": _row_out_total(sim) - row0,
        "payload": spend_ev[0].payload if spend_ev else {},
    }


def test_b6_no_clip_is_bit_identical_to_the_old_nominal_basis(tmp_path):
    """(a) クリップが無ければ 実支払 == 名目 なので新旧の税額は**厳密に同値**。"""
    nominal = 1100.0
    r = _spend_probe(tmp_path, "b6_noclip", money=500000.0, nominal=nominal)
    assert r["actual"] == nominal, "そもそもクリップが起きている(前提が崩れている)"
    old_tax = _tax_of(r["sim"], nominal, "shop")        # 旧実装の課税標準 = 名目
    new_tax = _tax_of(r["sim"], r["actual"], "shop")    # 新実装の課税標準 = 実支払
    assert new_tax == old_tax, "クリップ無しで税額が動いた(同値性の破れ)"
    assert _yen_eq(r["d_gov"], old_tax), "行政の歳入が旧挙動と 1 円でも違う"
    assert _yen_eq(r["d_row"], nominal - old_tax), "受け手が受け取る額が旧挙動と違う"
    assert "paid" not in r["payload"], "クリップしていないのに paid が載っている"


def test_b6_clipped_tax_is_bounded_by_actual_and_payee_stays_non_negative(tmp_path):
    """(b) クリップ有り: 0 <= tax <= 実支払 かつ 受け手 net >= 0。"""
    # ★nominal/11 > actual になるよう選ぶ(= 旧挙動なら net が必ず負に落ちる領域)
    r = _spend_probe(tmp_path, "b6_clip", money=50.0, nominal=1100.0)
    actual = r["actual"]
    assert actual == 50.0, "床クリップが起きていない(前提が崩れている)"
    tax = r["d_gov"]
    assert 0.0 <= tax <= actual, f"税が実支払を超えている: tax={tax} actual={actual}"
    assert r["d_row"] >= 0.0, f"受け手の net が負: {r['d_row']}"
    assert _yen_eq(r["d_row"], actual - tax)
    assert r["payload"].get("paid") == round(actual, 1), "床クリップが正直開示されていない"


def test_b6_conservation_sums_to_zero_under_clipping(tmp_path):
    """(c) 総和 Σ=0(家計の減 + 行政の増 + RoW へ出た額 = 0)がクリップ下でも閉じる。"""
    r = _spend_probe(tmp_path, "b6_sigma", money=50.0, nominal=1100.0)
    total = (-r["actual"]) + r["d_gov"] + r["d_row"]
    assert abs(total) < 1e-6, f"保存則が破れている: Σ={total}"


def test_b6_falsify_old_nominal_basis_overtaxes_and_drains_the_payee(tmp_path):
    """(d) ★反証: 旧挙動(名目課税)を**実際に実行**すると税 > 実支払・net < 0 になる。"""
    r = _spend_probe(tmp_path, "b6_falsify", money=50.0, nominal=1100.0)
    sim, agent, actual = r["sim"], r["agent"], r["actual"]
    old_tax = _tax_of(sim, 1100.0, "shop")              # 旧実装の課税標準 = 名目
    assert old_tax > actual, "反証の前提が崩れている(旧挙動でも税が実支払を超えない)"
    assert actual - old_tax < 0.0, "旧挙動なら受け手 net は負でなければならない"

    # 旧実装の行政計上を**そのまま呼ぶ**: 実支払 50 円の客から old_tax(=100 円)徴収する
    gov0 = _gov_total(sim)
    scheduler._record_consumption_tax(sim, agent, 1100.0, "shop", 0, 0)
    d_gov_old = _gov_total(sim) - gov0
    assert _yen_eq(d_gov_old, old_tax)
    assert d_gov_old > actual, \
        "旧挙動は『払えなかった客』から実支払を超える税を取っていた(無からの貨幣創出)"


# =========================================================================== #
# B7  死者を経済フェーズから除外
# =========================================================================== #
B7_ON = {
    "economy.accounts.enabled": "true", "economy.accounts.payday_dom": 1,
    "economy.bank.enabled": "true", "economy.bank.deposit_rate": 3.65,
    "economy.fixed_cost_daily": 300,
    "economy.wage_profile.enabled": "true",
    "government.enabled": "true",
}


def _arm_econ(agent, occupation: str) -> None:
    """5 フェーズの全部が「この個体に用がある」状態に置く(生死以外は同一条件)。"""
    agent.dead = False
    agent.visitor = False
    agent.evicted = False
    agent.sick = False
    agent.occupation = occupation
    agent.money = 0.0                       # < benefit_threshold = 給付の対象
    agent.account = 100000.0                # 利息・家賃・固定費の原資
    agent.wage = 12000.0
    agent.work_days = 3                     # 給料日にまとめ支給される勤務日数
    agent.period_income = 200000.0          # 家賃 = これ × rent_share
    agent.rent_due = 0.0


def _run_econ_phases(sim, day: int) -> None:
    """B7 が対象とする 5 フェーズだけを 1 日ぶん直接駆動する(LLM を 1 度も呼ばない)。"""
    step, sim_min = day * 144, day * 1440
    sim._wage_day = -1
    sim._econ_day = -1
    sim._acct_day = -1
    scheduler._phase_wage_profile(sim, step, sim_min)
    scheduler._phase_daily(sim, step, sim_min)          # 内部で _phase_accounts_day
    gov = scheduler._gov(sim)
    scheduler._gov_payroll(sim, gov, step, sim_min)
    scheduler._gov_benefits(sim, gov, step, sim_min)


def test_b7_dead_agent_moves_not_one_yen_while_the_control_does(tmp_path):
    """死者の口座は 1 円も動かない。**同条件の生者(対照)は動く**= 旧挙動なら死者も動いた。

    死者と対照の差は ``dead`` フラグ 1 つだけなので、対照が動いたという事実が
    そのまま「dead ゲートが無ければ死者も同じだけ動いていた」ことの反証になる。
    """
    sim = _sim(tmp_path, "b7_dead", n_steps=1, n_agents=12, **B7_ON)
    dead, ctrl = sim.agents[0], sim.agents[1]
    _arm_econ(dead, "警察官")                 # 公務員 = _gov_payroll の対象
    _arm_econ(ctrl, "警察官")
    dead.dead = True
    snap = (dead.money, dead.account, dead.work_days, dead.period_income, dead.rent_due)
    n0 = len(sim.logger.events)
    for day in range(3):
        _run_econ_phases(sim, day)
    assert (dead.money, dead.account, dead.work_days,
            dead.period_income, dead.rent_due) == snap, \
        "死者の口座・勤務日数・月収相当が経済フェーズに動かされている"
    assert (ctrl.money, ctrl.account) != (snap[0], snap[1]), \
        "対照(生者)が動いていない = このテストは何も証明していない"
    # 対照側でどのフェーズが実際に発火したかを名指しで固定する
    # (= 死者側の「1 件も無い」が「そもそも誰も動かない」ではないことの証明)
    fired = {(e.kind, str(e.payload.get("source") or ""))
             for e in sim.logger.events[n0:] if e.agent_id == ctrl.id}
    for want in (("wage", "salary"),      # _phase_accounts_day(月給まとめ)
                 ("rent", ""),            # _phase_accounts_day(家賃引落)
                 ("interest_paid", ""),   # _phase_daily(預金利息)
                 ("spend", ""),           # _phase_daily(固定費 300 円/日)
                 ("wage", "civil")):      # _gov_payroll(公務員給与)
        assert want in fired, f"対照で {want} が発火していない(前提が崩れている)"
    assert not [e for e in sim.logger.events[n0:] if e.agent_id == dead.id], \
        "死者に経済イベントが 1 件でも出ている"


def test_b7_dead_agent_is_not_settled_by_the_wage_profile_phase(tmp_path):
    """_phase_wage_profile(第112 WAGE / 第114 §ROLE)が死者を清算しない。"""
    sim = _sim(tmp_path, "b7_wage", n_steps=1, n_agents=12, **B7_ON)
    dead, ctrl = sim.agents[0], sim.agents[1]
    for a in (dead, ctrl):
        _arm_econ(a, "タクシー運転手")          # §ROLE の歩合職 = 在場なら毎日支給される
    dead.dead = True
    sim._wage_day = -1
    scheduler._phase_wage_profile(sim, 0, 0)
    assert int(getattr(dead, "rp_settled_day", -1)) == -1, "死者が清算されている"
    assert (dead.money, dead.account) == (0.0, 100000.0), "死者に日銭が入っている"
    assert int(getattr(ctrl, "rp_settled_day", -1)) == 0, \
        "対照が清算されていない = このテストは何も証明していない"
    assert ctrl.account > 100000.0, "対照に日銭が入っていない(前提が崩れている)"


def test_b7_dead_agent_receives_no_welfare_benefit(tmp_path):
    """_gov_benefits(困窮者給付)が死者へ給付しない。"""
    sim = _sim(tmp_path, "b7_benefit", n_steps=1, n_agents=12, **B7_ON)
    gov = scheduler._gov(sim)
    dead, ctrl = sim.agents[0], sim.agents[1]
    for a in (dead, ctrl):
        _arm_econ(a, "会社員")
        a.account = 0.0                     # 現金 + 口座 < benefit_threshold
    dead.dead = True
    ward0 = gov.balance["ward"]
    scheduler._gov_benefits(sim, gov, 0, 0)
    assert dead.money == 0.0, "死者へ給付が振り込まれている"
    assert ctrl.money > 0.0, "対照へ給付が出ていない = このテストは何も証明していない"
    assert gov.balance["ward"] < ward0, "区の歳出が立っていない(前提が崩れている)"


def test_b7_absent_but_alive_agent_still_receives_catchup_pay(tmp_path):
    """★除外条件は dead **だけ**: 街の外に居る生者への不在時キャッチアップ支給は生きている。

    第112 WAGE の設計仕様(振込は在不在に関わらず着金する)。ここが壊れると
    プール回転層・通勤者の給与が構造的に消える。
    """
    sim = _sim(tmp_path, "b7_outside", n_steps=1, n_agents=12, **B7_ON)
    agent = sim.agents[0]
    _arm_econ(agent, "議員")                    # §ROLE の月額(payday_dom=21)
    agent.loc = "outside"                       # ★街の外に居る(が生きている)
    agent.rp_settled_day = 0                    # 最後の清算は day0
    before = agent.money + agent.account
    sim._wage_day = -1
    # day25 まで不在 → 戻った最初の清算日に (0, 25] の給料日(dom=21 → day20)を拾う
    scheduler._phase_wage_profile(sim, 25 * 144, 25 * 1440)
    assert agent.money + agent.account > before, \
        "不在(outside・非 dead)個体のキャッチアップ支給が消えている"
    assert int(getattr(agent, "rp_settled_day", -1)) == 25


# =========================================================================== #
# B8  出生時の世帯二重所属
# =========================================================================== #
BIRTH_ON = {
    "population.births.enabled": "true",
    "population.births.interval_days": "1.0",   # 位相に関わらず毎日 due
    "population.births.spread": "0.0",
    "population.births.couple_share": "1.0",
    "population.births.age_min": "0", "population.births.age_max": "99",
    "population.births.min_days": "0",
    "population.births.household_spouses": "false",   # partner_id 経路だけを見る
}


def _birth_pair(tmp_path, name, hh_lo, hh_hi):
    """1 組だけが出生候補になる小ランを作る(他の全員は候補から外す)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=10, **BIRTH_ON)
    for a in sim.agents:                        # 既存の縁を消して候補を 1 組に絞る
        a.partner_id = None
        a.visitor = True
    lo, hi = sim.agents[0], sim.agents[1]
    if lo.id > hi.id:
        lo, hi = hi, lo
    for x, hh in ((lo, hh_lo), (hi, hh_hi)):
        x.visitor = False
        x.dead = False
        x.pop_emigrated = False
        x.age = 30
        x.housemates = []
        x.household_id = hh
    lo.partner_id = hi.id
    hi.partner_id = lo.id
    st = P._state(sim)
    return sim, st, lo, hi


def test_b8_split_household_pair_is_skipped_with_a_reason(tmp_path):
    """① 別世帯どうしの夫婦は出生を受理しない(候補には数え、理由を台帳に残す)。"""
    sim, st, lo, hi = _birth_pair(tmp_path, "b8_split", "H_A", "H_B")
    P._births_day(sim, st, day=0, step=0, sim_min=0)
    assert st["counts"]["birth"] == 0 and st["births"] == []
    assert st["considered"]["birth"] == 1, "候補として数えられていない(黙って消えている)"
    assert st["skipped"].get("birth:split_household") == 1
    assert lo.housemates == [] and hi.housemates == []
    assert (lo.household_id, hi.household_id) == ("H_A", "H_B"), "世帯を勝手に統合している"


def test_b8_same_household_pair_still_gives_birth(tmp_path):
    """①' 反証の対をなす対照: 世帯が同じなら**同じ組が**出生する(門だけが効いている)。"""
    sim, st, lo, hi = _birth_pair(tmp_path, "b8_same", "H_A", "H_A")
    P._births_day(sim, st, day=0, step=0, sim_min=0)
    assert st["counts"]["birth"] == 1, "対照が出生していない = ①は何も証明していない"
    assert st["skipped"].get("birth:split_household") is None


def test_b8_one_unbound_parent_merges_into_the_existing_household(tmp_path):
    """② 片方が無所属なら既存世帯へ合流し、子の所属世帯はちょうど 1 つになる。"""
    sim, st, lo, hi = _birth_pair(tmp_path, "b8_merge", "H_A", None)
    P._births_day(sim, st, day=0, step=0, sim_min=0)
    assert st["counts"]["birth"] == 1
    row = st["births"][0]
    child = int(row["child_id"])
    assert row["household_id"] == "H_A"
    assert lo.household_id == hi.household_id == "H_A", \
        "無所属だった親の household_id が合流先へ揃っていない"
    assert child in lo.housemates and child in hi.housemates
    assert len({lo.household_id, hi.household_id}) == 1, "子が 2 世帯に属している"


def test_b8_both_unbound_parents_found_a_new_household(tmp_path):
    """③ 両方無所属なら従来どおり世帯を 1 つ新設する(挙動を変えていない)。"""
    sim, st, lo, hi = _birth_pair(tmp_path, "b8_new", None, None)
    P._births_day(sim, st, day=0, step=0, sim_min=0)
    assert st["counts"]["birth"] == 1
    hh = st["births"][0]["household_id"]
    assert hh.startswith("pb")
    assert lo.household_id == hi.household_id == hh
    assert lo.household_kind == hi.household_kind == "family"


def test_b8_falsify_unguarded_birth_puts_the_child_in_two_households(tmp_path):
    """★反証: 門を通さず ``_birth`` を直接呼ぶ(= B8 以前に到達していた状態)と二重所属。"""
    sim, st, lo, hi = _birth_pair(tmp_path, "b8_falsify", "H_A", "H_B")
    P._birth(sim, st, lo, hi, together=0, day=0, step=0, sim_min=0)
    child = int(st["births"][0]["child_id"])
    assert child in lo.housemates and child in hi.housemates
    assert lo.household_id != hi.household_id, \
        "反証の前提が崩れている(別世帯のままでなければ二重所属は起きない)"


# =========================================================================== #
# A9  provenance: 伝播回数をプロンプトから切り離す
# =========================================================================== #
class _NullLogger:
    """ItemStore.transmit が要求する最小の logger(単体テスト用)。"""

    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


def _store_with(n: int):
    store = ItemStore()
    logger = _NullLogger()
    item = store.new_item("vocab", "ためし語", creator=1, step=0)
    for i in range(n):
        store.transmit(logger, item, step=i, sim_min=i * 10,
                       from_agent=1, to_agent=2 + i, channel="face", x=0.0, y=0.0)
    return store, item, logger


def test_a9_counter_is_always_equal_to_len(tmp_path):
    """併設カウンタは常に ``len(transmissions)`` と同値(更新点が 1 箇所しかない)。"""
    _store, item, logger = _store_with(7)
    assert item.transmissions_count == len(item.transmissions) == 7
    assert len(logger.events) == 7, "L1 の transmission 行が減っている"


def test_a9_counter_matches_len_for_every_item_after_a_run(tmp_path):
    """実ラン全体でも Σ 同値(語彙・噂のどの経路を通っても崩れない)。"""
    sim = _sim(tmp_path, "a9_run", n_steps=144, n_agents=15)
    sim.run()
    items = list(sim.items.items.values())
    assert items, "1 つも Item が生えていない(前提が崩れている)"
    for it in items:
        assert it.transmissions_count == len(it.transmissions), \
            f"{it.item_id}: counter={it.transmissions_count} len={len(it.transmissions)}"


def test_a9_search_prompt_string_is_unchanged(tmp_path):
    """プロンプト文字列(「N 回拡散」)が修正前後で 1 バイトも変わらない。"""
    sim = _sim(tmp_path, "a9_prompt", n_steps=1, n_agents=8)
    speaker = sim.agents[0]
    word = "ためし語"
    item = sim.labels.coin(speaker, word, step=0, sim_min=0, logger=sim.logger)
    for i, listener in enumerate(sim.agents[1:5]):
        sim.items.transmit(sim.logger, item, step=i, sim_min=i * 10,
                           from_agent=speaker.id, to_agent=listener.id,
                           channel="face", x=0.0, y=0.0)
    src = ("メディア発表" if item.creator == -1
           else f"{sim.agent_by_id[item.creator].name}が言い始めた")
    expected = (f"「{word}」: 最近使われ始めた言葉({src}、"
                f"{len(item.transmissions)}回拡散)")     # ← 旧実装の式そのもの
    got = scheduler._search_index(sim, word)
    assert got and got[0] == expected, f"プロンプトが変わった: {got[:1]} != {expected}"


def test_a9_old_pickle_without_the_counter_backfills_from_the_list(tmp_path):
    """旧 checkpoint 互換: counter を持たない pickle を load しても落ちず同値が保たれる。"""
    _store, item, _lg = _store_with(5)
    stale = Item(item_id=item.item_id, kind=item.kind, text=item.text,
                 creator=item.creator, born_step=item.born_step,
                 transmissions=list(item.transmissions))
    del stale.__dict__["transmissions_count"]           # A9 以前の pickle を再現する
    revived = pickle.loads(pickle.dumps(stale))
    assert revived.transmissions_count == len(revived.transmissions) == 5


def test_a9_checkpoint_roundtrip_preserves_the_counter(tmp_path):
    """checkpoint save → load で counter == len が保たれる(labels 同梱の経路)。"""
    cfg = _cfg("a9_ckpt", n_steps=144, n_agents=12)
    sim = Simulation(cfg, out_dir=tmp_path / "a9_ckpt")
    sim.run()
    path = tmp_path / "a9_ckpt" / "ck.pkl.gz"
    checkpoint.save(sim, 144, path)
    before = {k: v.transmissions_count for k, v in sim.items.items.items()}
    assert before, "checkpoint に Item が 1 つも載っていない(前提が崩れている)"

    revived = Simulation(_cfg("a9_ckpt", n_steps=144, n_agents=12),
                         out_dir=tmp_path / "a9_ckpt_r")
    checkpoint.load(revived, path)
    items = revived.items.items
    assert {k: v.transmissions_count for k, v in items.items()} == before
    for it in items.values():
        assert it.transmissions_count == len(it.transmissions)


# =========================================================================== #
# B9  starvation 観測の agent 属性書き込みを observer 側へ
# =========================================================================== #
def test_b9_observation_writes_no_attribute_onto_agents(tmp_path):
    """① 観測 ON / OFF で全 agent の ``__dict__`` キー集合が完全に一致する。"""
    off = _sim(tmp_path, "b9_off", n_steps=144, n_agents=15,
               **{"observer.starvation.enabled": "false"})
    off.run()
    on = _sim(tmp_path, "b9_on", n_steps=144, n_agents=15,
              **{"observer.starvation.enabled": "true"})
    on.run()
    keys_off = {a.id: frozenset(a.__dict__) for a in off.agents}
    keys_on = {a.id: frozenset(a.__dict__) for a in on.agents}
    assert keys_on == keys_off, \
        "観測 ON が agent へ属性を書いている(= checkpoint バイトが観測で変わる)"
    assert all("_plan_expired_day" not in ks for ks in keys_on.values())


def test_b9_action_stream_is_identical_with_observation_on(tmp_path):
    """② 観測イベント 3 種を除いた L1(= 世界の行動列)が ON / OFF で完全一致。"""
    off = _sim(tmp_path, "b9_l1_off", n_steps=144, n_agents=15,
               **{"observer.starvation.enabled": "false"})
    off.run()
    on = _sim(tmp_path, "b9_l1_on", n_steps=144, n_agents=15,
              **{"observer.starvation.enabled": "true"})
    on.run()
    assert [r for r in _l1(on) if r[2] not in STARV_KINDS] == _l1(off)
    assert len(on.logger.llm_calls) == len(off.logger.llm_calls)


def test_b9_dedup_counts_one_agent_day_and_keeps_the_step_tally(tmp_path):
    """③ 重複排除が効く(同日 2 回で agent_days は 1・翌日は改めて 1 増える)。"""
    sim = _sim(tmp_path, "b9_dedup", n_steps=1, n_agents=8,
               **{"observer.starvation.enabled": "true"})
    agent = sim.agents[0]
    SV.note_plan_expired_awake(sim, agent, 0)          # day0
    SV.note_plan_expired_awake(sim, agent, 30)         # day0(同日 2 度目)
    st = SV.state(sim)
    assert st["plan_expired_awake_steps"] == 2
    assert st["plan_expired_awake_agent_days"] == 1, "同日に二重計上されている"
    SV.note_plan_expired_awake(sim, agent, 1440)       # day1
    assert st["plan_expired_awake_steps"] == 3
    assert st["plan_expired_awake_agent_days"] == 2, "翌日が新しい 1 日と数えられていない"
    assert "_plan_expired_day" not in agent.__dict__, "観測が agent へ書いている"
    assert st["plan_expired_seen_day"] == 1
    assert list(st["plan_expired_seen"]) == [int(agent.id)], \
        "印が当日ぶんに刈り込まれていない(保持が有界でない)"


def test_b9_dedup_state_survives_a_state_restored_without_the_new_keys(tmp_path):
    """旧 checkpoint 互換: 新キーを持たない state を復元しても落ちない。"""
    sim = _sim(tmp_path, "b9_oldstate", n_steps=1, n_agents=8,
               **{"observer.starvation.enabled": "true"})
    st = SV.state(sim)
    del st["plan_expired_seen_day"]                    # B9 以前の state を再現する
    del st["plan_expired_seen"]
    SV.note_plan_expired_awake(sim, sim.agents[0], 0)
    assert st["plan_expired_awake_agent_days"] == 1
    assert st["plan_expired_seen"] == {int(sim.agents[0].id): 1}
