"""賃金多様性 WAGE(第112バッチ 2026-08-13。既定 OFF)のテスト。

ユーザー要求:「L2 の本業日給 0 の穴を実装で塞ぐ。ただし月給なのか日給か・給料が
振り込まれるタイミング・ボーナスの有無・職種に見合った金額か —— 給料にも多様性がある実装に」。

検証:
  純関数   職種群の判定 / 職種適合額 / 最低賃金の床 / 給料日の分布 / 賞与の割当 /
           日割り / 支給形態 / 対象判定(学生・公務員・自営・無給・L4 来街者を弾く)
  決定論   seed 非依存・同じキーなら同じプラン(resume / 回転再入で不変)
  搬送     dehydrate/hydrate 往復(既定値では**キーを 1 つも足さない**)
  税       支給周期別の年換算(月給を ×245 して最高税率にしない)
  統合 ON  L2 に賃金イベントが出る / 給料日に発火 / 回転を跨いで勤務日が消えない
  OFF 不変 既定 OFF で L1 バイト一致(golden は tests/test_scenario)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society import economy as E                        # noqa: E402
from society.agents.agent import Agent                  # noqa: E402
from society.agents.memory import MemoryStore           # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.government import Government, build_government_cfg  # noqa: E402
from society.world import pool as pool_mod              # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return E.build_wage_profile_cfg(None)


@pytest.fixture(scope="module")
def cfg_floor():
    """東京都最低賃金 1,226 円/時 の床つき(本選プロファイルと同じ)。"""
    return E.build_wage_profile_cfg(None, min_wage_hourly=1226.0)


def _org(industry="IT", band="10-19"):
    return {"industry_key": industry, "size": {"band": band}}


# ======================================================== 職種群の判定(表の順序が意味論)
@pytest.mark.parametrize("occupation,expected", [
    ("エンジニア", "expert"), ("開発プロジェクト担当", "expert"),
    ("土木施工管理技士", "expert"), ("弁護士", "expert"), ("看護師", "expert"),
    ("店長", "manager"),                 # ★「店員」より先に判定されなければならない
    ("警備司令", "manager"),
    ("夜間清掃", "cleaning"),            # ★現業より先に判定されなければならない
    ("常駐警備", "manual"), ("トラック運転手", "manual"), ("仕分け作業員", "manual"),
    ("設備保守員", "manual"),
    ("営業", "sales"), ("不動産営業", "sales"), ("仕入担当", "sales"),
    ("事務員", "clerical"), ("金融事務", "clerical"), ("受付", "clerical"),
    ("コーポレート", "clerical"),
    ("アパレル店員", "service"), ("接客係", "service"), ("バリスタ", "service"),
    ("深夜ホール", "service"),
    ("教員", "professional"), ("デザイナー", "professional"), ("調理師", "professional"),
    ("塾講師", "instructor"), ("語学講師", "instructor"),
    ("経営者", "exec"),
    ("会社員", "general"),               # 表に無い語は general(既定)
])
def test_occupation_group(occupation, expected):
    assert E.occupation_group(occupation, "") == expected


def test_group_rules_all_map_to_a_known_multiplier():
    """判定表の右辺は全て GROUP_MULT に実在する(死んだ群名の検出)。"""
    for _word, grp in E.OCC_GROUP_RULES:
        assert grp in E.GROUP_MULT, f"未定義の職種群: {grp}"


def test_role_is_used_when_occupation_is_unknown():
    """職業名が表に無ければ台帳の role を見る(pool の L5 など)。"""
    assert E.occupation_group("よくわからない肩書", "販売スタッフ") == "service"


# ======================================================== 職種適合額(ユーザー要求の水準)
def test_pay_matches_the_occupation(cfg):
    """職種に見合った額になっている(ユーザーが挙げた水準に収まる)。

    帯の判定は「規模帯を跨いだ中央値」で行う(規模で ±16% 動くのが仕様そのものなので、
    1 点だけを固定すると規模の効果を殺してしまう)。"""
    cases = [                       # (職業, 産業, 月額の下限, 月額の上限)
        ("エンジニア", "IT", 330_000, 480_000),
        ("営業", "WR", 250_000, 350_000),
        ("事務員", "IT", 210_000, 290_000),
        ("店長", "FB", 220_000, 320_000),
        ("常駐警備", "SV", 175_000, 250_000),
        ("夜間清掃", "SV", 150_000, 220_000),
        ("教員", "ED", 250_000, 340_000),
    ]
    for occ, ind, lo, hi in cases:
        grp = E.occupation_group(occ, "")
        vals = [E.monthly_wage(ind, grp, band, f"{occ}#{i}", cfg)
                for band in E.BAND_MULT for i in range(12)]
        vals.sort()
        med = vals[len(vals) // 2]
        assert lo <= med <= hi, f"{occ}: 月額中央値 {med:,.0f} が {lo:,}〜{hi:,} の外"


def test_industry_ordering_is_respected(cfg):
    """同じ職種でも産業で額が違う(情報通信 > 卸小売 > 宿泊飲食)。"""
    def m(ind):
        return E.monthly_wage(ind, "general", "50-99", "same_key", cfg)
    assert m("IT") > m("WR") > m("FB")


def test_size_band_premium_is_monotone(cfg):
    """同じ職種・同じ産業でも規模帯が大きいほど高い(企業規模間賃金格差)。"""
    vals = [E.monthly_wage("IT", "general", b, "same_key", cfg)
            for b in ("1-4", "5-9", "10-19", "20-29", "30-49", "50-99", "100-299", "300+")]
    assert vals == sorted(vals)
    assert vals[-1] > vals[0] * 1.3


def test_individual_spread_is_within_bounds_and_deterministic(cfg):
    """個体差は ±spread に収まり、同じキーなら何度呼んでも同じ(乱数ゼロ)。"""
    base = E.monthly_wage("IT", "general", "50-99", "k0", dict(cfg, spread=0.0))
    vals = [E.monthly_wage("IT", "general", "50-99", f"k{i}", cfg) for i in range(500)]
    for v in vals:
        assert base * 0.85 - 10 <= v <= base * 1.15 + 10
    assert len(set(vals)) > 400                      # 実際にばらけている(定数でない)
    assert vals[7] == E.monthly_wage("IT", "general", "50-99", "k7", cfg)


def test_minimum_wage_floor_lifts_the_lowest_paid(cfg, cfg_floor):
    """最低賃金の床(1,226 円/時 × 8h × 20日)が零細飲食の接客を法定水準へ持ち上げる。"""
    assert cfg["min_monthly"] == 0.0                 # 既定は床なし=従来と同一
    assert cfg_floor["min_monthly"] == pytest.approx(1226.0 * 8 * 20)
    bare = E.monthly_wage("FB", "service", "1-4", "barista", cfg)
    lifted = E.monthly_wage("FB", "service", "1-4", "barista", cfg_floor)
    assert bare < cfg_floor["min_monthly"], "床が要らないなら検査が空回りしている"
    assert lifted == pytest.approx(cfg_floor["min_monthly"])


# ======================================================== 給料日・支給形態・賞与
def test_payday_distribution_matches_the_convention():
    """給料日は {10,15,20,25,月末} に散り、25 日が最頻(民間の慣行)。"""
    c = Counter(E.payday_dom_of(f"co_{i:06d}") for i in range(20000))
    assert set(c) == {0, 10, 15, 20, 25}
    share = {k: v / 20000 for k, v in c.items()}
    assert share[25] == pytest.approx(0.40, abs=0.02)
    assert share[25] > max(share[k] for k in (0, 10, 15, 20))
    for dom, w in E.PAYDAY_WEIGHTS:                   # 全区分が実際に出る
        assert share[dom] == pytest.approx(w, abs=0.02)


def test_payday_is_shared_within_one_workplace(cfg):
    """同じ職場の人は同じ給料日(給料日は会社の属性であって個人の属性ではない)。"""
    plans = [E.wage_plan(occupation="営業", role="営業", org=_org(), org_id="co_it_00007",
                         key=f"L2_{i:08d}", cfg=cfg) for i in range(30)]
    assert len({p["payday"] for p in plans}) == 1
    assert len({p["monthly"] for p in plans}) > 20     # 額は個体差でばらける


def test_pay_period_is_mostly_monthly_and_never_weekly(cfg):
    """支給形態は月給が既定。日給は接客/現業/清掃の小規模職場だけ。週給は存在しない。"""
    periods = {E.pay_period_of(g, b, f"k{i}", cfg)
               for g in E.GROUP_MULT for b in E.BAND_MULT for i in range(40)}
    assert periods <= {"monthly", "daily"}, "週給など未定義の支給形態が出た"
    # 事務・専門は規模に関わらず必ず月給
    assert all(E.pay_period_of("clerical", b, f"k{i}", cfg) == "monthly"
               for b in E.BAND_MULT for i in range(50))
    # 小規模の接客は一部が日給(少数派)
    got = [E.pay_period_of("service", "1-4", f"k{i}", cfg) for i in range(4000)]
    share = got.count("daily") / len(got)
    assert 0.10 < share < 0.20
    # 中〜大規模の接客は日給にならない(日払いの慣行は小規模に偏る)
    assert all(E.pay_period_of("service", "100-299", f"k{i}", cfg) == "monthly"
               for i in range(200))


def test_bonus_is_richer_at_larger_workplaces(cfg):
    """賞与は規模が大きいほど「支給する職場の割合」も「支給月数」も高い。"""
    def stats(band):
        plans = [E.bonus_plan_of(f"co_{i:06d}", band, 25, cfg) for i in range(3000)]
        paid = [p for p in plans if p]
        return len(paid) / len(plans), (paid[0]["mult"] if paid else 0.0)
    small_share, small_mult = stats("1-4")
    big_share, big_mult = stats("300+")
    assert small_share < big_share
    assert small_mult < big_mult
    assert small_mult == pytest.approx(cfg["bonus"]["months_min"])
    assert big_mult == pytest.approx(cfg["bonus"]["months_max"])


def test_bonus_never_lands_on_the_payday(cfg):
    """賞与日は給料日と重ならない(同一 step に賃金を 2 本重ねない設計の根拠)。"""
    for payday in (0, 5, 10, 15, 20, 25):
        for i in range(400):
            p = E.bonus_plan_of(f"co_{i:06d}", "100-299", payday, cfg)
            if p:
                assert p["dom"] != payday
                assert set(p["months"]) <= {6, 7, 12}


def test_bonus_can_be_disabled(cfg):
    off = E.build_wage_profile_cfg({"bonus": {"enabled": False}})
    assert all(E.bonus_plan_of(f"co_{i}", "300+", 25, off) is None for i in range(50))


# ======================================================== 日割り・年収
def test_salary_is_prorated_for_a_partial_month(cfg):
    """所定労働日数に満たない月は日割り、満勤なら月額そのもの(超過しない)。"""
    plan = {"monthly": 300000.0}
    assert E.salary_amount(plan, 20, cfg) == pytest.approx(300000.0)
    assert E.salary_amount(plan, 10, cfg) == pytest.approx(150000.0)
    assert E.salary_amount(plan, 0, cfg) == 0.0
    assert E.salary_amount(plan, 30, cfg) == pytest.approx(300000.0)   # 残業手当は無い


def test_annual_income_includes_bonus(cfg):
    plan = E.wage_plan(occupation="エンジニア", role="エンジニア", org=_org("IT", "300+"),
                       org_id="co_it_00001", key="L2_00000001", cfg=cfg)
    expected = plan["monthly"] * 12.0
    if plan["bonus"]:
        expected += plan["monthly"] * plan["bonus"]["mult"] * len(plan["bonus"]["months"])
    assert plan["annual"] == pytest.approx(expected)
    assert plan["daily"] == pytest.approx(plan["monthly"] / cfg["month_workdays"], abs=0.1)


# ======================================================== 対象判定(二重支給の禁止)
@pytest.mark.parametrize("occupation,role,ok", [
    ("営業", "営業", True),               # ★WAGE_CAT に無い台帳ロール名(198,264 人)= 対象
    ("常駐警備", "スタッフ", True),
    ("会社員", "", True),
    ("エンジニア", "", True),
    ("大学生", "", False),                # 学業は無給
    ("無職", "", False),
    ("路上生活者", "", False),
    ("街頭演説者", "", False),            # 選挙運動は賃金労働でない
    ("フリーランス", "", False),          # 自営 = gig の日銭が払っている
    ("配達員", "", False),
    ("区職員", "", False),                # 公務員 = government のペイロールが払っている
    ("警察官", "", False),
    ("消防士", "", False),
    ("エンジニア", "学生", False),        # 学生ロールは賃金の受け手でない
])
def test_wage_eligible(occupation, role, ok):
    assert E.wage_eligible(occupation, role) is ok


def _agent(**kw):
    base = dict(id=1, name="甲", age=30, occupation="営業", persona="p",
                traits={}, states={}, mem=MemoryStore())
    base.update({k: v for k, v in kw.items() if k in
                 ("id", "occupation", "visitor", "commute", "work_start_min")})
    a = Agent(**base)
    for k, v in kw.items():
        if k not in base:
            setattr(a, k, v)
    return a


def test_visitors_without_a_local_job_are_not_paid(cfg):
    """L4(非定期来街 707,778 人)は勤務窓を持っていても対象外。

    persona._pick_workplace は職業名「会社員」に勤務窓を与えてしまう(L4 に 240,661 人)が、
    その職場は渋谷ではない(visitor かつ commute=false かつ org_id 無し)。"""
    tourist = _agent(occupation="会社員", visitor=True, commute=False, work_start_min=540)
    assert E.assign_wage_plan(tourist, {}, cfg) is None
    assert tourist.wage == 0.0
    # 通勤者(L2/L5)と居住者(L1)は対象
    commuter = _agent(occupation="会社員", visitor=True, commute=True, work_start_min=540)
    assert E.assign_wage_plan(commuter, {}, cfg) is not None
    resident = _agent(occupation="会社員", visitor=False, commute=False, work_start_min=540)
    assert E.assign_wage_plan(resident, {}, cfg) is not None


def test_no_work_window_means_no_plan(cfg):
    """勤務窓を持たない個体(失職・未就業)は対象外。"""
    idle = _agent(occupation="営業", visitor=False, work_start_min=-1)
    assert E.assign_wage_plan(idle, {}, cfg) is None


def test_assign_is_idempotent_and_sets_the_daily_wage(cfg):
    """冪等(2 回呼んでも同じ)で、agent.wage に日額が載る。"""
    a = _agent(occupation="エンジニア", visitor=True, commute=True, work_start_min=540,
               org_id="co_it_00001", org_role="エンジニア", pool_pid="L2_00000042")
    p1 = E.assign_wage_plan(a, {"co_it_00001": _org("IT", "20-29")}, cfg)
    p2 = E.assign_wage_plan(a, {"co_it_00001": _org("IT", "20-29")}, cfg)
    assert p1 is p2
    assert a.wage == pytest.approx(p1["daily"])
    assert a.wage > 0.0
    assert p1["industry"] == "IT" and p1["band"] == "20-29"


def test_plan_is_stable_across_rehydration(cfg):
    """同じ pool_pid・同じ台帳なら別インスタンスでも同じプラン(回転再入・resume で不変)。"""
    book = {"co_it_00001": _org("IT", "20-29")}
    def build(agent_id):
        a = _agent(id=agent_id, occupation="エンジニア", visitor=True, commute=True,
                   work_start_min=540, org_id="co_it_00001", org_role="エンジニア",
                   pool_pid="L2_00000042")
        return E.assign_wage_plan(a, book, cfg)
    # agent.id が違っても pool_pid が同じなら同一(id は密割当なので安定キーは pid)
    assert build(1) == build(999)


def test_school_staff_use_the_education_industry(cfg):
    """学校(規模帯を持たない)は教育・学習支援業の水準・既定帯で評価する。"""
    school = {"school_type": "区立小学校", "workplace_poi": {}}
    p = E.wage_plan(occupation="教員", role="教員", org=school, org_id="sc_es_01",
                    key="L2_00099999", cfg=cfg)
    assert p["industry"] == "ED" and p["band"] == cfg["default_band"]
    assert 250_000 <= p["monthly"] <= 340_000


# ======================================================== 税(支給周期別の年換算)
def test_income_tax_annualization_by_pay_period():
    """★既存の欠陥の修正: 月給まとめを日給前提の ×245 で年換算すると最高税率になる。"""
    gov = Government(build_government_cfg({"enabled": True}))
    monthly = 300_000.0
    naive = gov.income_tax(monthly)                       # 年収 7,350 万円扱い(誤り)
    correct = gov.income_tax(monthly, annual=monthly * 12)  # 年収 360 万円(正しい)
    assert correct < naive
    assert correct / monthly < naive / monthly            # 実効税率が下がっている
    # 日給は従来式のまま(annual 未指定 = 1 バイトも変わらない)
    daily = 12_000.0
    assert gov.income_tax(daily) == gov.income_tax(
        daily, annual=daily * gov.cfg["annual_workdays"])


def test_income_tax_default_is_byte_identical():
    """annual=None は既存の全呼び出しと完全同一(既定経路の不変)。"""
    gov = Government(build_government_cfg({"enabled": True}))
    for g in (0.0, 1.0, 9_000.0, 12_000.0, 250_000.0):
        assert gov.income_tax(g) == gov.income_tax(g, annual=None)


# ======================================================== 退避スリム状態の搬送
def test_dehydrate_keeps_the_default_dict_untouched():
    """既定値(口座 OFF / WAGE OFF)では退避 dict にキーを 1 つも足さない(バイト一致の芯)。"""
    a = Agent(id=1, name="甲", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    assert "econ" not in pool_mod.dehydrate(a)


def test_dehydrate_carries_the_unsettled_work(tmp_path):
    """★回転で毎回 0 に戻っていた欄が持続する(月給が原理的に成立するための条件)。"""
    a = Agent(id=1, name="甲", age=30, occupation="営業", persona="p",
              traits={}, states={}, mem=MemoryStore())
    a.period_income = 123456.0
    a.work_days = 3
    a.rent_due = 5000.0
    a.arrears_days = 2
    a.last_salary = 250000.0
    a.evicted = True
    a.bankrupt_until = 4321
    a.wp_days = 7
    a.wp_settled_day = 9
    a.wp_bonus_pending = True
    state = pool_mod.dehydrate(a)
    assert state["econ"]["wp_days"] == 7

    b = Agent(id=2, name="甲", age=30, occupation="営業", persona="p",
              traits={}, states={}, mem=MemoryStore())
    pool_mod.hydrate(b, state)
    assert b.period_income == 123456.0
    assert b.work_days == 3
    assert b.rent_due == 5000.0
    assert b.arrears_days == 2
    assert b.last_salary == 250000.0
    assert b.evicted is True
    assert b.bankrupt_until == 4321
    assert b.wp_days == 7                       # ★未清算の勤務実績が消えない
    assert b.wp_settled_day == 9
    assert b.wp_bonus_pending is True


def test_hydrate_tolerates_old_states():
    """旧 退避辞書(econ キーなし)からは属性を 1 つも生やさない。"""
    b = Agent(id=2, name="甲", age=30, occupation="営業", persona="p",
              traits={}, states={}, mem=MemoryStore())
    pool_mod.hydrate(b, {"beliefs": [], "money": 100.0})
    assert b.work_days == 0 and b.period_income == 0.0


# ======================================================== 統合(mock・小規模)
def _sim(tmp_path, name, steps=1, n=30, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


_ON = {"economy.wage_profile.enabled": "true", "economy.accounts.enabled": "true"}
#: 実暦モードで**給料日を必ず跨ぐ**短縮構成。6/24 開始で 7 日走ると 25 日(最頻)と
#  30 日(月末)の両方が窓に入る。block_day%30+1 の既定モードだと 1,2,3… にしかならず、
#  10 日ランでは給料日に永遠に到達しない(= 本選で塞いだ穴そのもの)。
_CAL = {"world.calendar.enabled": "true",
        "world.calendar.start_date": "2026-06-24",
        "economy.wage_profile.calendar": "true"}
_ON_CAL = dict(_ON, **_CAL)
_PAYDAY_WINDOW_STEPS = 144 * 7


def test_off_is_byte_identical(tmp_path):
    """既定 OFF と明示 OFF で L1 が完全一致(新機構が 1 行も通らない)。"""
    def l1(name, **ov):
        sim = _sim(tmp_path, name, steps=40, n=15, **ov)
        sim.run()
        return [(e.step, e.agent_id, e.kind, json.dumps(e.payload, sort_keys=True))
                for e in sim.logger.events]
    assert l1("wp_def") == l1("wp_off", **{"economy.wage_profile.enabled": "false"})


def test_off_leaves_no_attributes(tmp_path):
    """OFF では賃金プランも清算カウンタも 1 体も生えない。"""
    sim = _sim(tmp_path, "wp_noattr", steps=2)
    sim.run()
    assert all(not hasattr(a, "_wage_plan") for a in sim.agents)
    assert all(not hasattr(a, "wp_settled_day") for a in sim.agents)


def test_on_pays_and_diversifies(tmp_path):
    """ON: 給料日に wage(source=salary/daily)が出て、額が 1 種類でない。"""
    sim = _sim(tmp_path, "wp_on", steps=_PAYDAY_WINDOW_STEPS, n=30, **_ON_CAL)
    sim.run()
    wages = [e for e in sim.logger.events if e.kind == "wage"
             and e.payload.get("source") in ("salary", "daily")]
    assert wages, "ON なのに賃金イベントが 1 件も出ていない"
    amounts = {round(float(e.payload["amount"]), 1) for e in wages}
    assert len(amounts) > 1, "賃金が単一の額に潰れている(多様性が無い)"


def test_on_payday_is_the_only_day_money_arrives(tmp_path):
    """月給者への支給は給料日にだけ起きる(毎日ではない)。"""
    sim = _sim(tmp_path, "wp_payday", steps=_PAYDAY_WINDOW_STEPS, n=30, **_ON_CAL)
    sim.run()
    salaries = [e for e in sim.logger.events
                if e.kind == "wage" and e.payload.get("source") == "salary"]
    assert salaries, "給料日が 1 度も来ていない(実暦の給料日判定が壊れている)"
    per_agent_days = Counter()
    for e in salaries:
        per_agent_days[(e.agent_id, e.sim_min // 1440)] += 1
    assert all(v == 1 for v in per_agent_days.values())     # 同じ日に 2 度払わない
    days = {e.sim_min // 1440 for e in salaries}
    assert len(days) < 5, "毎日支給になっている(給料日の判定が効いていない)"
    # 支給された日は実際に「暦の給料日」である(25 日 or その月の末日)
    import datetime
    for d in days:
        date = datetime.date(2026, 6, 24) + datetime.timedelta(days=int(d))
        nxt = (date.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        last = (nxt - datetime.timedelta(days=1)).day
        assert date.day in (10, 15, 20, 25, last), f"{date} は給料日でない"


def test_on_block_mode_reaches_a_payday_without_the_calendar(tmp_path):
    """calendar=false(既定)でも 30 日周期で給料日は来る(暦 OFF のランの後退動作)。"""
    sim = _sim(tmp_path, "wp_block", steps=144 * 11, n=20,
               **dict(_ON, **{"economy.wage_profile.payday_weights": "[[10,1.0]]"}))
    sim.run()
    salaries = [e for e in sim.logger.events
                if e.kind == "wage" and e.payload.get("source") == "salary"]
    assert salaries, "30 日周期モードで給料日が来ていない"
    assert {e.sim_min // 1440 for e in salaries} == {9}      # day9 → dom=10


def test_on_does_not_double_pay_via_settle_work(tmp_path):
    """勤務完遂の都度払い(_settle_work)と日次清算が二重に走らない。"""
    sim = _sim(tmp_path, "wp_nodouble", steps=144 * 2, n=40, **_ON)
    sim.run()
    covered = {a.id for a in sim.agents if getattr(a, "_wage_plan", None)}
    plain = [e for e in sim.logger.events if e.kind == "wage"
             and e.agent_id in covered and "source" not in e.payload]
    assert not plain, "プラン保持者に source 無しの本業賃金(都度払い)が出ている"


def test_on_same_agent_never_gets_two_wages_in_one_step(tmp_path):
    """同一 agent・同一 step に賃金を 2 本重ねない(源泉税の帰属突合が壊れないため)。"""
    sim = _sim(tmp_path, "wp_onestep", steps=_PAYDAY_WINDOW_STEPS, n=30,
               **dict(_ON_CAL, **{"government.enabled": "true"}))
    sim.run()
    seen = Counter((e.step, e.agent_id) for e in sim.logger.events if e.kind == "wage")
    assert seen and max(seen.values()) == 1


def test_on_tax_uses_the_right_annualization(tmp_path):
    """月給の源泉徴収が「日給前提の ×245」で最高税率にならない。"""
    sim = _sim(tmp_path, "wp_tax", steps=_PAYDAY_WINDOW_STEPS, n=30,
               **dict(_ON_CAL, **{"government.enabled": "true"}))
    sim.run()
    top = sim.government.cfg["income_brackets"][-1]["rate"]
    salaries = [e for e in sim.logger.events
                if e.kind == "wage" and e.payload.get("source") == "salary"
                and e.payload.get("gross")]
    assert salaries, "課税された月給が 1 件も無い(検査が空回り)"
    for e in salaries:
        rate = float(e.payload["tax"]) / float(e.payload["gross"])
        assert rate < top, f"月給に最高税率 {top} が掛かっている(年換算の補正が効いていない)"


# ======================================================== プール(L2 = 塞いだ穴そのもの)
_ORGS_BOOK = "data/organizations_shibuya_wide11k.json"


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。test_pool_rotation と同型。"""
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("wagepool")
    orgs = json.loads((_ROOT / "data" / _ORGS_BOOK.split("/")[-1]
                       ).read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop, total_target=1_000_000)
    return out


def _pool_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}",
           "organizations.enabled=true", f"organizations.book={_ORGS_BOOK}",
           "work.bind_workplace.enabled=true",
           f"work.bind_workplace.book={_ORGS_BOOK}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_l2_workers_finally_get_paid(small_pool, tmp_path):
    """★塞いだ穴そのもの: 台帳ロール名の L2 従業者に賃金が実際に届く。

    OFF では彼らの agent.wage は 0(職業名が WAGE_CAT に無い)= 1 円も稼がない。"""
    off = Simulation(_pool_cfg("wp_l2_off", small_pool, n_steps=1),
                     out_dir=tmp_path / "l2off")
    unmapped = [a for a in off.agents
                if getattr(a, "org_id", None)
                and a.occupation not in E.WAGE_CAT
                and int(getattr(a, "work_start_min", -1)) >= 0]
    assert unmapped, "台帳ロール名の L2 が 1 人も居ない(検査の前提が崩れている)"
    assert all(a.wage == 0.0 for a in unmapped), "OFF なのに日給が付いている"

    on = Simulation(_pool_cfg("wp_l2_on", small_pool, n_steps=_PAYDAY_WINDOW_STEPS,
                              **_ON_CAL), out_dir=tmp_path / "l2on")
    covered = [a for a in on.agents
               if getattr(a, "org_id", None) and a.occupation not in E.WAGE_CAT]
    assert covered, "ON 側に台帳ロール名の L2 が居ない"
    paid = [a for a in covered if a.wage > 0.0]
    assert len(paid) > 0.8 * len(covered), \
        f"ON でも日給 0 の L2 が多すぎる({len(covered) - len(paid)}/{len(covered)})"
    on.run()
    ids = {a.id for a in paid}
    wages = [e for e in on.logger.events if e.kind == "wage" and e.agent_id in ids]
    assert wages, "★L2 に賃金イベントが 1 件も出ていない(穴が塞がっていない)"
    # 職業ごとに額が違う(産業 × 職種群 × 規模帯 × 個体差)
    assert len({round(a.wage, 1) for a in paid}) > 10, "日給が数種類に潰れている"


def test_rotation_keeps_the_unsettled_work(small_pool, tmp_path):
    """回転(退場 → 再来街)を跨いでも未清算の勤務日数が消えない(月給成立の条件)。

    presence の曜日は day%7 なので day5/6 が週末 = workday_shift の L2 が一斉に退場する。
    その瞬間の退避辞書に「平日 5 日ぶんの未清算勤務」が載っていなければ、月給は原理的に
    成立しない(戻ってきた月曜に 0 から数え直しになる)。"""
    sim = Simulation(_pool_cfg("wp_rot", small_pool, n_steps=144 * 6, **_ON_CAL),
                     out_dir=tmp_path / "rot")
    sim.run()
    carried = [st for st in sim._dormant._d.values()
               if (st.get("econ") or {}).get("wp_days")]
    assert carried, "勤務実績を持って退場した個体が 1 人も居ない(搬送の検査が空回り)"
    assert max(st["econ"]["wp_days"] for st in carried) >= 4, \
        "退場時に積まれた勤務日数が少なすぎる(日次カウントが効いていない)"
    # 在場中の個体も日をまたいで積み上がっている
    live = [a for a in sim.agents if getattr(a, "_wage_plan", None)
            and getattr(a, "wp_settled_day", -1) >= 0]
    assert live and max(int(getattr(a, "wp_days", 0)) for a in live) >= 2


def test_catchup_pays_a_payday_that_passed_while_away(tmp_path):
    """不在中に過ぎた給料日を、戻った最初の清算日に拾う(振込は在不在に関わらず着金する)。"""
    from society.engine import scheduler
    sim = _sim(tmp_path, "wp_catchup", steps=1, n=5, **_ON_CAL)
    cfg = sim.economy["wage_profile"]
    # 6/24 開始。day1=6/25 が給料日。day0 に清算して day4(6/28)まで不在だった個体は、
    # 「(0, 4] に 25 日が来たか」で拾える。
    assert scheduler._wage_fires(sim, cfg, 0, 4, 25) is True
    assert scheduler._wage_fires(sim, cfg, 0, 0, 25) is False   # まだ来ていない
    assert scheduler._wage_fires(sim, cfg, 1, 4, 25) is False   # もう清算済み
    # 月末(0)は「その月の末日」に解決される(6月=30日 → day6)
    assert scheduler._wage_fires(sim, cfg, 5, 6, 0) is True
    assert scheduler._wage_fires(sim, cfg, 5, 5, 0) is False
    # 賞与は月でも絞る(6月の 5 日は窓の外なので発火しない)
    assert scheduler._wage_fires(sim, cfg, -1, 6, 25, months=(12,)) is False
    assert scheduler._wage_fires(sim, cfg, -1, 6, 25, months=(6,)) is True


def test_pool_on_resume_byte_matches_straight(small_pool, tmp_path):
    """WAGE ON でも「一気 vs 中断→resume」の l1_events が完全一致(P3 検収基準)。"""
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    st = tmp_path / "st"
    Simulation(_pool_cfg("wp_rst", small_pool, n_steps=300, **_ON_CAL), out_dir=st).run()

    rs = tmp_path / "rs"
    s1 = Simulation(_pool_cfg("wp_rrs", small_pool, n_steps=150,
                              **dict(_ON_CAL, **{"observer.checkpoint_every": 150})),
                    out_dir=rs)
    for step in range(150):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 150, rs / "checkpoint" / "ckpt-000150.pkl.gz")
    s1._save_pool_sidecar(150)
    s1.logger.flush_segment()
    s2 = Simulation(_pool_cfg("wp_rrs", small_pool, n_steps=300,
                              **dict(_ON_CAL, **{"observer.checkpoint_every": 150})),
                    out_dir=rs)
    s2.run(resume_from=rs)
    assert rows(st) == rows(rs), "WAGE ON の resume が straight と byte 不一致"
