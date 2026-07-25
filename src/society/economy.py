"""経済 v0(ユーザー決定 2026-07-05: 賃金+消費+バイト+心理接続)。

原則:
- 職業 → 賃金カテゴリ・持ち金レンジ・バイト有無 の写像はここ(persona/simulation 側)。
  factors/ は職業を見ない(R9)。engine は precompute 済みの金額(agent.wage 等)しか
  見ないので、no-fingerprint 契約(engine が因子名を名指ししない)にも触れない。
- 金額は conf の economy ブロックで可変(決め打ちを1か所に閉じる)。

役割:
- build_economy: conf → 実行時の economy dict。
- wage_amount: 職業カテゴリ → 本業の日給(勤務完遂で支給)。
- initial_money: 職業・来街者フラグ → 手持ち初期値(rng)。
- assign_part_time: 学生・フリーター等に実在 POI(shop/food)のバイトを割当。
- price_of: 消費カテゴリ → 価格(食事/買い物/nightlife)。
"""
from __future__ import annotations

import numpy as np

# 職業 → 本業の賃金カテゴリ(economy.wages のキー)。None = 無給(学生の学業・無職)
WAGE_CAT: dict[str, str | None] = {
    "会社員": "会社員", "エンジニア": "会社員", "デザイナー": "会社員",
    "カフェ店員": "店員", "アパレル店員": "店員", "美容師": "店員",
    "大学生": None,                      # 学業は無給(バイトで稼ぐ)
    "フリーランス": "自営", "写真家": "自営", "バンドマン": "自営",
    "配達員": "自営", "無職": None,
}

# ---- 公務員(行政・税・公務員バッチ 2026-07-06。docs/research/shibuya-government.md)----
# 職業 → 給与の出所となる行政主体。給与は government の日次ペイロール(scheduler)で予算から支給し、
# 源泉徴収(所得税+住民税)を掛ける。区職員=ward予算 / 警察官・消防士=metro予算(帰属の現実準拠:
# 警視庁・東京消防庁は東京都の公安職地方公務員=区職員ではない)。WAGE_CAT には入れない
# (勤務地写像 persona._pick_workplace は別バッチ所有で編集不可のため、通常の勤務完遂経路ではなく
#  government のペイロールで支給する=二重支給や gig 経路の混入を避ける)。
CIVIL_SERVANTS: dict[str, str] = {"区職員": "ward", "警察官": "metro", "消防士": "metro"}


def civil_servant_pay(occupation: str, economy: dict) -> tuple[float, str] | None:
    """公務員なら (日給, 給与の出所 level) を返す。非公務員は None。

    日給は economy.wages の該当キー。実額(行政職 ≈630万/年・公安職 ≈700-800万/年)を
    会社員=12,000円/日 の圧縮スケールに実比率で寄せた値(docs/research §8)。
    """
    fund = CIVIL_SERVANTS.get(occupation)
    if fund is None:
        return None
    return float(economy["wages"].get(occupation, 0.0)), fund

# 職業 → 手持ち初期値レンジ(円)。来街者は allowance_visitor を使う(下記)
MONEY_INIT: dict[str, tuple[int, int]] = {
    "会社員": (50000, 150000), "エンジニア": (50000, 150000),
    "デザイナー": (40000, 120000),
    "カフェ店員": (10000, 40000), "アパレル店員": (10000, 40000),
    "美容師": (20000, 60000),
    "大学生": (5000, 30000),
    "フリーランス": (20000, 100000), "写真家": (15000, 80000),
    "バンドマン": (5000, 30000), "配達員": (15000, 50000),
    "無職": (3000, 20000),
    # 公務員(安定収入 → 会社員並みの手持ちレンジ)
    "区職員": (50000, 150000), "警察官": (50000, 150000), "消防士": (50000, 150000),
}
_MONEY_DEFAULT = (20000, 80000)

# バイトを持つ職業(学生・フリーター等)。実在 POI(shop/food)で夕方シフト
PART_TIME_OCC = {"大学生", "無職", "バンドマン"}


def build_accounts_cfg(raw: dict | None) -> dict:
    """口座(銀行)概念 E5 の設定(既定 OFF)。100 日ラン想定=月次の資金繰りを表現。

    payday_dom = 給料日(シミュ内暦の何日目。1日=144step、run開始日=1日として day%30)。
    ON 時: 月給者は日割りでなく給料日に月給まとめ支給、翌日に家賃を口座から引き落とす。
    card_threshold 以上の支払いは口座(カード)、未満は現金。現金不足は最寄り ATM で引出。
    """
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "payday_dom": int(raw.get("payday_dom", 25)),
        "rent_share": float(raw.get("rent_share", 0.30)),
        "card_threshold": float(raw.get("card_threshold", 3000)),
        "atm_withdraw": float(raw.get("atm_withdraw", 20000)),
        "cash_share": float(raw.get("cash_share", 0.20)),   # 初期資産の現金比率(残りは口座)
        # ---- 立退き・破産サイクル(制度深化3 2026-07-08。既定 0/0=どちらも無効=E5 従来と完全同一)----
        # 現実の対応: 家賃3ヶ月滞納→信頼関係破壊の法理で契約解除(立退き)、支払不能→自己破産
        # (免責=借金消滅+自由財産以外の資産処分+資格・活動の制限期間)。日数は圧縮スケール可。
        "eviction_days": int(raw.get("eviction_days", 0)),          # 滞納がこの日数続くと立退き
        "bankruptcy_days": int(raw.get("bankruptcy_days", 0)),      # 滞納がこの日数続くと自己破産
        "bankruptcy_keep": float(raw.get("bankruptcy_keep", 10000)),  # 破産後も手元に残る自由財産(円)
        "bankruptcy_restrict_days": int(raw.get("bankruptcy_restrict_days", 30)),  # 出店等の制限(日)
        "eviction_grievance": float(raw.get("eviction_grievance", 0.10)),    # 立退きの生活不安
        "bankruptcy_grievance": float(raw.get("bankruptcy_grievance", 0.15)),  # 破産の生活不安
    }


def build_economy(raw: dict | None) -> dict:
    raw = dict(raw or {})
    wages = {"会社員": 12000, "自営": 10000, "店員": 9000, "part_time_hourly": 1100,
             # 公務員の日給(docs/research §8: 実額の年収比を会社員=12000 基準に圧縮)。
             # government OFF 時は誰も読まない(WAGE_CAT 外・gig=自営のみ)=既定挙動に無影響。
             "区職員": 13000, "警察官": 15000, "消防士": 14000}
    wages.update({str(k): float(v) for k, v in dict(raw.get("wages", {})).items()})
    # ---- 最低賃金の床(制度深化3 2026-07-08)。既定 0.0=床なし=従来と完全同一 ----
    # 一次確認: 東京都最低賃金 1,226円/時(2025-10-03 発効。東京労働局・地域別最低賃金)。
    # 床は「雇用される労働者」の賃金のみに適用(最低賃金法の適用対象)。自営(フリーランス・
    # 配達員等)は労働者でないため対象外=床を掛けない(現実の制度と同じ穴をそのまま持つ)。
    # 日給カテゴリは 8h 換算(床×8)で持ち上げる。part_time_hourly は時給なのでそのまま比較。
    min_wage = float(raw.get("min_wage_hourly", 0.0))
    if min_wage > 0.0:
        wages["part_time_hourly"] = max(float(wages["part_time_hourly"]), min_wage)
        for cat in ("会社員", "店員", "区職員", "警察官", "消防士"):
            wages[cat] = max(float(wages.get(cat, 0.0)), min_wage * 8.0)
    prices = {"food": 900, "cafe": 500, "shop": 2500, "nightlife": 1800,
              "leisure": 0}
    prices.update({str(k): float(v) for k, v in dict(raw.get("prices", {})).items()})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "wages": wages,
        "min_wage_hourly": min_wage,                    # 観測・検査用(床の適用は上で済み)
        "prices": prices,
        "allowance_visitor": float(raw.get("allowance_visitor", 20000)),
        # 来街者の財布補充(改善 P2 第9バッチ): 帰宅(範囲外)から戻るたび手持ちを
        # allowance_visitor まで戻す。来街者は反復収入が無く長期ランで恒久破綻する
        # (sim-improvement-analysis.md P2)ことへの対処。既定 false=補充なし=従来と完全同一。
        "visitor_refresh": bool(raw.get("visitor_refresh", False)),
        "money_pressure_threshold": float(raw.get("money_pressure_threshold", 2000)),
        # 固定費(光熱費・サブスク等。Wave G1 2026-07-07): 家賃(rent)以外の日次固定支出=
        # 恒常的な生活圧。既定 0.0=控除ゼロ=イベントなし=バイト一致。★ON 推奨: 例 300。
        "fixed_cost_daily": float(raw.get("fixed_cost_daily", 0.0)),
        "accounts": build_accounts_cfg(raw.get("accounts")),
        # ---- 経済深化 E(第37バッチ 2026-07-19。全て既定 OFF=誰も読まない=バイト一致)----
        # E-W3 消費行動(家計調査2024)/ 決済手段(経産省2024)/ E-W1 銀行(預金利息・融資)/
        # E-W2 VC(ベンチャー出資)。config.yaml に該当ブロックが無ければ raw.get(...)=None →
        # 各 build_*_cfg が既定 OFF を返す。scheduler/tools が読むまで完全 no-op(新イベント0件)。
        "consumption": build_consumption_cfg(raw.get("consumption")),
        "payment": build_payment_cfg(raw.get("payment")),
        "bank": build_bank_cfg(raw.get("bank")),
        "vc": build_vc_cfg(raw.get("vc")),
    }


def build_career_cfg(raw) -> dict:
    """キャリア転換(Wave G5。失業/求職/転職/起業転換)の設定を正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける。すべて既定 OFF(enabled=false)= scheduler の career 日次
    フェーズと tools の起業転換が完全 no-op(unemployment/job_change/venture_fulltime を1件も出さず・
    新 stream "career" も引かない=乱数消費不変=ゴールデン golden_baseline_l1.json を守る)。

    パラメータ(ON 時のみ効く。確率は被雇用者/失業者ごとの日次確率):
      layoff_prob            被雇用者が職を失う日次確率(失業)
      switch_prob            被雇用者が別 org へ移る日次確率(転職。失業と排他=失業判定を優先)
      rehire_prob            失業者が新 org へ再就職する日次確率(求職)
      unemployment_grievance 失業時の生活不安=不満(grievance+。factors 経由の不透明 magnitude。0=金銭のみ)
      venture_fulltime_sales 起業転換の売上閾値(open_venture の累計売上がこれ以上で本業化。0=起業転換 OFF)
      venture_fulltime_days  起業転換の最短経過日数(開店からこの日数以上)
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "layoff_prob": float(raw.get("layoff_prob", 0.02)),
        "switch_prob": float(raw.get("switch_prob", 0.02)),
        "rehire_prob": float(raw.get("rehire_prob", 0.30)),
        "unemployment_grievance": float(raw.get("unemployment_grievance", 0.05)),
        "venture_fulltime_sales": float(raw.get("venture_fulltime_sales", 8000.0)),
        "venture_fulltime_days": int(raw.get("venture_fulltime_days", 2)),
        # 解雇規制の最小形(制度深化2 第10バッチ。解雇権濫用法理のモデル。既定 0/0=従来と完全同一):
        #   severance_days        解雇時の退職金=日給×この日数(0=退職金なし)
        #   unfair_ratio          解雇のうち「不当解雇」の割合(>0 のときだけ "career" stream を
        #                         追加 draw=既定は draw 数も不変)。不当解雇は生活不安を増幅
        #   unfair_grievance_mult 不当解雇時の unemployment_grievance の倍率
        "severance_days": float(raw.get("severance_days", 0.0)),
        "unfair_ratio": float(raw.get("unfair_ratio", 0.0)),
        "unfair_grievance_mult": float(raw.get("unfair_grievance_mult", 2.0)),
        # career選択由来化(内部可動性 第60バッチ b。既定 OFF=求職 tool を出さない=バイト一致)。
        # ON: LLM の行動空間に求職 tool(job_search)を1件足す(既存 tool 選択枠内=呼数不変)。
        # 発火→ mobility.match_job(定員空きの org を決定論選択)→ 既存 switch_org/rehire を呼ぶ。
        # 確率駆動 switch_prob とは独立(両立可=干渉しない)。
        "by_choice": {"enabled": bool(dict(raw.get("by_choice", {}) or {})
                                      .get("enabled", False))},
    }


def split_account(money: float, visitor: bool, economy: dict) -> tuple[float, float]:
    """初期資産を(現金, 口座)へ分割。口座 OFF/来街者は現金のまま(account=0)。

    居住者は口座8割/現金2割(cash_share で可変)。来街者は街の外に家=口座を持たない。
    rng は引かない(決定論)=口座 OFF 経路では money を一切動かさない(バイト一致)。"""
    acc = economy.get("accounts") if economy else None
    if not acc or not acc.get("enabled") or visitor:
        return money, 0.0
    cash = float(round(money * float(acc["cash_share"])))
    return cash, float(money - cash)


def wage_amount(occupation: str, has_workplace: bool, economy: dict) -> float:
    """本業の勤務完遂で支給する日給。職場を持たない/無給職は 0。"""
    if not has_workplace:
        return 0.0
    cat = WAGE_CAT.get(occupation)
    if cat is None:
        return 0.0
    return float(economy["wages"].get(cat, 0.0))


def gig_profile(occupation: str, economy: dict) -> dict | None:
    """自営(固定職場を持たない)層の日銭プロファイル。

    WAGE_CAT が「自営」相当の職業(フリーランス/写真家/バンドマン/配達員 等)は本業の
    勤務完遂が発生しないため wage_amount が 0 になる(経済 v0 の穴)。ここでその層に
    「日額の元手」を与える。実際の支給額は日次で出来高係数を掛けて決める(engine 側)。
    自営に当たらなければ None(= 支給なし)。wages 辞書の「自営」日額は本関数だけが読む。"""
    if WAGE_CAT.get(occupation) == "自営":
        return {"daily_base": float(economy["wages"].get("自営", 0.0))}
    return None


def initial_money(occupation: str, visitor: bool, rng: np.random.Generator,
                  economy: dict) -> float:
    """手持ち初期値。来街者は街の外に家=賃金なし、持ち金のみ(allowance レンジ)。"""
    if visitor:
        base = economy["allowance_visitor"]
        return float(round(rng.uniform(0.5 * base, 1.5 * base), -2))
    lo, hi = MONEY_INIT.get(occupation, _MONEY_DEFAULT)
    return float(round(rng.uniform(lo, hi), -2))


def assign_part_time(occupation: str, visitor: bool, city,
                     rng: np.random.Generator, economy: dict) -> dict | None:
    """学生・フリーター等に実在 POI(shop/food)のバイトを割当。

    シフト = 週3日・夕方(17:00〜18:00 開始)・4h(個人差は rng)。建物入口のある
    POI を優先(勤務完遂=退館の判定に乗せるため)。来街者にはバイトを割り当てない。
    """
    if visitor or occupation not in PART_TIME_OCC:
        return None
    pool = city.pois_by_cat("shop") + city.pois_by_cat("food")
    with_bld = [p for p in pool if p.get("building")
                and city.has_building(p["building"])]
    pool = with_bld or pool
    if not pool:
        return None
    p = pool[int(rng.integers(len(pool)))]
    bld = p.get("building", "")
    floor = int(p.get("floor", 0))
    if bld and floor == 0 and city.has_building(bld):
        levels = int(city.building(bld)["levels"])
        floor = int(rng.integers(1, levels + 1))
    days = sorted(int(d) for d in rng.choice(7, size=3, replace=False))
    start = 17 * 60 + int(rng.integers(0, 7)) * 10        # 17:00〜18:00 開始
    hours = 4
    hourly = float(economy["wages"].get("part_time_hourly", 1100))
    return {
        "name": p["name"], "node": p["node"], "building": bld,
        "floor": max(1, floor) if bld else 0,
        "days": days, "start_min": start, "end_min": start + hours * 60,
        "hours": hours, "pay": hourly * hours,
    }


def price_of(cat: str, economy: dict, rulebook=None) -> float:
    """消費カテゴリの価格。制度DSL の fee ルール(rulebook)があれば合算適用する。

    rulebook が None / 該当 fee ルール無しなら基準価格をそのまま返す(=既定不変)。
    """
    base = float(economy["prices"].get(cat, 0.0))
    if rulebook is not None:
        return rulebook.fee_price(cat, base)
    return base


# ==================================================================== 経済深化 E
# 第37バッチ 2026-07-19。docs/research/economy-abm-research.md(§3/§5/§6/§7)に基づく。
# 3機能(E-W3 消費・E-W1 銀行・E-W2 VC)とも既存の流儀で組む:
#   既定 OFF / 非LLM / 決定論 or 新 stream / OFF 時バイト一致(ゴールデン golden_baseline_l1.json)。
# R1 総括: generate()(LLM呼び出し)を1本も足さない=呼数不変。判定は observables
#   (money/account/period_income/sales_total/occupancy/relations)と config のみを参照し、
#   k・内面構成概念(efficacy/grievance 等)を発火判断に食わせない(研究доク §7【要注意 B】)。
# ★本モジュールの関数・クラスは「純ロジック(ログしない・sim を触らない)」。イベント記録
#   (loan_grant/loan_repay/interest_paid/vc_investment)と乱数 stream の配線は scheduler/tools
#   スチュワードが後で行う(下記 TODO)。
#
# TODO(スチュワード配線。scheduler.py/tools.py は本エージェント編集不可のため未配線):
#   E-W3 消費: _charge_meal/_charge_ride/tools._buy_at_ventures で _spend 前に、consumption.enabled
#     なら amount = budget_amount(consumption_profile(agent.traits, agent.period_income, ccfg),
#     cat, base, agent.period_income, sim.economy["fixed_cost_daily"], ccfg) に置換。
#   E-W3 決済: _spend で payment.enabled なら method = choose_payment(amount,
#     payment_pref(agent.traits), sim.hub.stream("payment", agent.id, step), pcfg) を payload に付与し、
#     cashless=口座(既存カード経路)/ cash=現金 で引落を分岐(現状の card_threshold 分岐と整合)。
#   E-W1 利息: _phase_daily で bank.enabled かつ _accounts_on なら居住者ごと i=daily_interest(
#     agent.account, bcfg); agent.account+=i; interest_paid をログ。
#   E-W1 融資: 現金不足点(rent 引落・move_home 敷金・_open_venture 出店費)で、_bank(sim)(=_gov 同型の
#     Bank 遅延構築)を使い score=bank.score(period_income, money+account, 返済実績, arrears_days,
#     グループ所属)→ loan_approved かつ need≤loan_limit なら loan=bank.grant(id, need, score, day);
#     agent.account+=need; loan_grant をログ。日次返済フェーズで loan_due→repay_installment→残高控除+
#     loan_repay ログ、loan_defaulted なら bank.write_off + 既存 accounts の破産サイクルへ接続。
#   E-W2 VC: tools.phase の review_period_days ごとに sim.vc_fund(=VCFund 遅延構築)で開店中の全 venture を
#     commerce.vc_score(sales_total, commerce.occupancy(sim, node), len(owner.mem.relations), vccfg)
#     → commerce.vc_candidates → can_invest なら invest+入金+vc_investment ログ。tools._buy_at_ventures の
#     売上で fund.collect_dividend(owner, price) を owner 取り分から差引き。


def _to_container(raw) -> dict:
    """dotlist / OmegaConf どちらでも受けて素の dict に正準化(career/commerce と同型)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})


def _clip(x: float, lo: float, hi: float) -> float:
    x = float(x)
    return lo if x < lo else hi if x > hi else x


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _trait_dev(traits: dict, key: str) -> float:
    """trait(0..1、既定 0.5=中立)を中立中心の偏差 [-1,1] へ。未設定=0(中立)。"""
    return (float((traits or {}).get(key, 0.5)) - 0.5) * 2.0


# -------------------------------------------------------------- E-W3 消費行動 + 決済
def build_consumption_cfg(raw) -> dict:
    """E-W3 消費行動(家計調査2024 単身世帯の費目構成=§5)。既定 OFF=現行 spend と完全同一。

    budget_shares: 単身世帯 2024 の費目シェア(食料 28.4% 等)。個体差は consumption_profile が
      traits から決定論導出(エンゲル: 所得↑で食料↓、risk_tolerance で discretionary の厚み)。
    予算制約: 可処分=(period_income − 固定費)×(1−貯蓄率)。所得が低いほど支出を絞る(§7 E-W3)。
    """
    raw = _to_container(raw)
    # §5 家計調査2024 単身世帯: 食料28.4 / 家具+被服=shop6.6 / 教養娯楽=nightlife12.0 /
    #   その他(交際費等)=social14.5 / 交通通信=transport12.1。住居・光熱・医療は別機構が処理。
    shares = {"food": 0.284, "shop": 0.066, "nightlife": 0.120,
              "social": 0.145, "transport": 0.121}
    shares.update({str(k): float(v)
                   for k, v in dict(raw.get("budget_shares", {})).items()})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "budget_shares": shares,
        "engel_income_slope": float(raw.get("engel_income_slope", -0.15)),  # 所得↑で食料↓
        "share_spread": float(raw.get("share_spread", 0.5)),        # 個体差の幅(相対)
        "savings_rate_base": float(raw.get("savings_rate_base", 0.20)),
        "savings_rate_spread": float(raw.get("savings_rate_spread", 0.15)),
        "income_ref": float(raw.get("income_ref", 170000.0)),        # 単身世帯 月消費≈17万(§5)
        "min_income_factor": float(raw.get("min_income_factor", 0.4)),  # 支出圧縮の下限
    }


def build_payment_cfg(raw) -> dict:
    """決済手段の確率モデル(経産省2024=§6)。既定 OFF=spend payload に method を足さない=不変。

    cashless_prob 42.8%(2024 実績)を基準に、金額帯(card_threshold)と個体嗜好で method を選ぶ。
    既存 _spend のカード/現金閾値ロジックと整合(大額=カード=cashless 寄り、少額=現金寄り)。"""
    raw = _to_container(raw)
    return {
        "enabled": bool(raw.get("enabled", False)),
        "cashless_prob": float(raw.get("cashless_prob", 0.428)),   # §6 経産省2024
        "card_threshold": float(raw.get("card_threshold", 3000.0)),  # 既存 accounts キー流用
        "small_amount": float(raw.get("small_amount", 500.0)),      # これ未満=少額(現金寄り)
        "large_boost": float(raw.get("large_boost", 0.5)),         # 大額→cashless へ引上げ(割合)
        "small_penalty": float(raw.get("small_penalty", 0.3)),     # 少額→cashless を引下げ(割合)
        "pref_slope": float(raw.get("pref_slope", 0.3)),           # 個体嗜好の効き(±0.5×slope)
    }


def consumption_profile(traits: dict, income: float, cfg: dict) -> dict:
    """個体別の消費プロファイル(traits から決定論導出=乱数を引かない)。

    戻り: {"shares": {cat: 配分ウェイト}, "savings_rate": 貯蓄率}。
    - エンゲル: 所得↑で食料シェア↓(engel_income_slope)。
    - 個体差: risk_tolerance 高=食料↓・discretionary(nightlife/social)厚め・貯蓄率↓、
             internal_locus 高=計画的=貯蓄率↑。
    - 正規化: 配分の総和を budget_shares の総和へ戻す(総消費性向は不変・費目配分だけ動かす)。"""
    base = cfg["budget_shares"]
    ref = float(cfg["income_ref"]) or 1.0
    inc_dev = (float(income) - ref) / ref                 # 基準所得で 0
    risk = _trait_dev(traits, "risk_tolerance")           # [-1,1]
    locus = _trait_dev(traits, "internal_locus")
    spread = float(cfg["share_spread"])
    raw_shares: dict[str, float] = {}
    for cat, s in base.items():
        w = float(s)
        if cat == "food":                                 # エンゲル + risk で薄く
            w = s * (1.0 + cfg["engel_income_slope"] * inc_dev) * (1.0 - 0.3 * spread * risk)
        elif cat in ("nightlife", "social"):              # discretionary は risk で厚く
            w = s * (1.0 + spread * risk)
        raw_shares[cat] = max(0.0, w)
    tot = sum(raw_shares.values()) or 1.0
    scale = (sum(base.values()) or 1.0) / tot
    shares = {k: v * scale for k, v in raw_shares.items()}
    sr = cfg["savings_rate_base"] + cfg["savings_rate_spread"] * (0.5 * locus - 0.5 * risk)
    return {"shares": shares, "savings_rate": _clip(sr, 0.0, 0.9)}


def budget_amount(profile: dict, cat: str, base_amount: float, income: float,
                  fixed_cost: float, cfg: dict) -> float:
    """予算制約下の実消費額(§7 E-W3)。budget_shares 外のカテゴリは base_amount のまま(不変)。

    - 個体ウェイト mult = shares[cat]/base_shares[cat](その個体が cat をどれだけ重視するか)。
    - 予算制約: 可処分=(income−fixed_cost)×(1−貯蓄率)を income_ref と比べ、低いほど支出を絞る
      (income_factor ∈ [min_income_factor, 1.0])。低所得個体ほど安く済ませる。"""
    base = cfg["budget_shares"]
    b = float(base.get(cat, 0.0))
    if b <= 0.0:
        return float(base_amount)
    mult = float(profile["shares"].get(cat, b)) / b
    disposable = max(0.0, float(income) - float(fixed_cost)) * (1.0 - profile["savings_rate"])
    income_factor = _clip(disposable / (float(cfg["income_ref"]) or 1.0),
                          float(cfg["min_income_factor"]), 1.0)
    return float(base_amount) * mult * income_factor


def payment_pref(traits: dict) -> float:
    """キャッシュレス嗜好(0..1、0.5=中立)を traits から決定論導出。risk/nfc 高=新決済を採りやすい。"""
    risk = float((traits or {}).get("risk_tolerance", 0.5))
    nfc = float((traits or {}).get("nfc", 0.5))
    return _clip01(0.5 + 0.5 * ((risk - 0.5) + (nfc - 0.5)))


def payment_p_cashless(amount: float, pref: float, cfg: dict) -> float:
    """金額×個体嗜好からキャッシュレス確率を返す(§6 較正)。基準は cashless_prob=42.8%。

    金額 ≥ card_threshold は大額=カード寄り(cashless↑)、small_amount 未満は少額=現金寄り
    (cashless↓)。個体嗜好 pref(0.5=中立)で ±。中間帯かつ pref 中立なら p=cashless_prob。"""
    thr = float(cfg["card_threshold"])
    small = float(cfg["small_amount"])
    p = float(cfg["cashless_prob"])
    a = float(amount)
    if a >= thr:
        p += (1.0 - p) * float(cfg["large_boost"])
    elif a < small:
        p *= (1.0 - float(cfg["small_penalty"]))
    p += float(cfg["pref_slope"]) * (float(pref) - 0.5)
    return _clip01(p)


def choose_payment(amount: float, pref: float, rng, cfg: dict) -> str:
    """決済手段 "cashless" / "cash" を新 stream("payment", ...) の1 draw で選ぶ(§6)。

    rng は sim.hub.stream("payment", agent.id, step)(既存 draw 順を乱さない専用 stream)。
    大数で cashless 比 ≈ cashless_prob(42.8%)へ較正。スチュワードは戻り値を spend payload の
    method に載せ、cashless=口座(カード/QR/電子マネー)/ cash=現金 で引落を分ける(会計不変)。"""
    return "cashless" if float(rng.random()) < payment_p_cashless(amount, pref, cfg) else "cash"


# -------------------------------------------------------------------- E-W1 銀行
def build_bank_cfg(raw) -> dict:
    """E-W1 銀行(預金利息・融資)。§1 信用回路 / §3 与信スコア(5C→4項)。既定 OFF=完全 no-op。

    deposit_rate 年利0.2%(→日割)。融資は base_rate 年利3% + 低スコアで premium 上乗せ。
    与信スコア = w·norm(income/assets/repayment) − w·norm(arrears)(+ グループ連帯保証ボーナス)。
    返済 term_days 90日を installment_days ごとの定期返済。延滞→既存 accounts の破産サイクルへ接続。
    ★既存 deposit イベント kind(供託金)とは衝突しない(§7【要注意 A】=新 kind のみ使用)。"""
    raw = _to_container(raw)
    w = {"income": 0.4, "assets": 0.3, "repayment": 0.2, "arrears": 0.1}   # §3 5C→4項
    w.update({str(k): float(v)
              for k, v in dict(raw.get("score_weights", {})).items()})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "deposit_rate": float(raw.get("deposit_rate", 0.002)),      # 預金 年利0.2%
        "base_rate": float(raw.get("base_rate", 0.03)),             # 融資 年利3%
        "premium_slope": float(raw.get("premium_slope", 0.05)),     # 低スコアで金利上乗せ
        "score_weights": w,
        "approve_threshold": float(raw.get("approve_threshold", 0.35)),  # score≥これ で承認
        "max_loan_ratio": float(raw.get("max_loan_ratio", 3.0)),    # 与信上限=月収相当×これ
        "group_guarantee_bonus": float(raw.get("group_guarantee_bonus", 0.1)),  # §3 社会的担保
        "term_days": int(raw.get("term_days", 90)),                 # 返済総期間
        "installment_days": int(raw.get("installment_days", 30)),   # 定期返済の周期
        "income_ref": float(raw.get("income_ref", 150000.0)),       # 正規化基準(月収相当)
        "assets_ref": float(raw.get("assets_ref", 300000.0)),
        "arrears_ref": float(raw.get("arrears_ref", 90.0)),
        "default_arrears_days": int(raw.get("default_arrears_days", 30)),  # 延滞→破産接続の日数
        "initial_capital": float(raw.get("initial_capital", 0.0)),
        "ration_premium_step": float(raw.get("ration_premium_step", 0.01)),  # §1 信用引締め
    }


def daily_interest(balance: float, cfg: dict) -> float:
    """預金の日次利息(年利 deposit_rate の日割・365)。残高>0 のみ。0 以下は 0(利息なし)。"""
    b = float(balance)
    if b <= 0.0:
        return 0.0
    return b * float(cfg["deposit_rate"]) / 365.0


def credit_score(income: float, assets: float, repayment: float,
                 arrears_days: float, has_group: bool, cfg: dict) -> float:
    """与信スコア(§3 5C→4項、決定論・乱数なし)。observables のみ=k 非依存(R1)。

    income=period_income(月収相当) / assets=money+account / repayment=返済実績[0..1]
    (履歴なし=0.5 neutral を渡す想定) / arrears_days=滞納日数(家賃 arrears を再利用)。
    has_group=連帯保証グループ所属(§3 社会的担保)。戻り値はおよそ [−0.1, 1.0+bonus]。"""
    w = cfg["score_weights"]
    ni = _clip01(float(income) / (float(cfg["income_ref"]) or 1.0))
    na = _clip01(float(assets) / (float(cfg["assets_ref"]) or 1.0))
    nr = _clip01(float(repayment))
    nar = _clip01(float(arrears_days) / (float(cfg["arrears_ref"]) or 1.0))
    score = (w["income"] * ni + w["assets"] * na + w["repayment"] * nr
             - w["arrears"] * nar)
    if has_group:
        score += float(cfg["group_guarantee_bonus"])
    return score


def loan_approved(score: float, cfg: dict) -> bool:
    """与信可否(score ≥ approve_threshold)。客観条件のみ(§7 E-W1 R1)。"""
    return float(score) >= float(cfg["approve_threshold"])


def loan_rate(score: float, cfg: dict) -> float:
    """個別金利 = base_rate + premium_slope×(1 − score)(§1 低スコア/高レバレッジで上昇)。"""
    return float(cfg["base_rate"]) + float(cfg["premium_slope"]) * (1.0 - _clip01(score))


def loan_limit(income: float, cfg: dict) -> float:
    """与信上限 = 月収相当(period_income)× max_loan_ratio。"""
    return float(income) * float(cfg["max_loan_ratio"])


def open_loan(principal: float, score: float, day: int, cfg: dict,
              extra_premium: float = 0.0) -> dict:
    """融資の実行データ(定期返済スケジュール)を組む。ログ・入金はスチュワードが行う。

    総返済額 = 元本×(1 + 実効金利×term_days/365)(単利)。installment_days ごとに n 回の定額返済。
    extra_premium=銀行の信用引締め premium(§1 貸倒発生で街全体に上乗せ)を金利へ加算。"""
    rate = loan_rate(score, cfg) + float(extra_premium)
    term = int(cfg["term_days"])
    inst_days = max(1, int(cfg["installment_days"]))
    n = max(1, term // inst_days)
    total = float(principal) * (1.0 + rate * term / 365.0)
    per = total / n
    return {"principal": float(principal), "rate": float(rate), "term_days": term,
            "installment_days": inst_days, "n_installments": n, "per": float(per),
            "total_due": float(total), "remaining": float(total),
            "paid_installments": 0, "arrears_days": 0,
            "opened_day": int(day), "next_due_day": int(day) + inst_days,
            "score": float(score)}


def loan_due(loan: dict, day: int) -> bool:
    """この日に返済回収を試みるべきか(未完済 かつ 返済期日到来)。"""
    return loan["remaining"] > 1e-9 and int(day) >= int(loan["next_due_day"])


def repay_installment(loan: dict, balance: float, day: int) -> tuple[float, str]:
    """1回分の定期返済を試みる(スチュワードが 1日1回・loan_due の時に呼ぶ)。

    戻り: (実返済額, status)。status = complete(完済) / paid(1回分完遂) / arrears(不能→延滞)。
    balance が定額(per)以上なら満額返済し次回期日を installment_days 先へ。不足なら1円も引かず
    延滞日数を +1、翌日再試行(部分返済はしない=分割の粒度を保つ)。実際の残高控除はスチュワード。"""
    if loan["remaining"] <= 1e-9:
        return 0.0, "complete"
    due = min(float(loan["per"]), float(loan["remaining"]))
    if float(balance) + 1e-9 >= due:
        loan["remaining"] = max(0.0, float(loan["remaining"]) - due)
        loan["paid_installments"] += 1
        loan["next_due_day"] = int(day) + int(loan["installment_days"])
        loan["arrears_days"] = 0
        return float(due), ("complete" if loan["remaining"] <= 1e-9 else "paid")
    loan["arrears_days"] += 1
    loan["next_due_day"] = int(day) + 1                    # 翌日再試行
    return 0.0, "arrears"


def loan_defaulted(loan: dict, cfg: dict) -> bool:
    """延滞が破産接続の閾値に達したか(既存 accounts の破産サイクルへ渡す判定)。"""
    return int(loan.get("arrears_days", 0)) >= int(cfg["default_arrears_days"])


class Bank:
    """銀行の会計主体(government.Government の雛形=§1)。scheduler が _gov と同型に遅延構築する。

    残高(capital)・貸出残(loans_outstanding)・貸倒(write_offs)・信用引締め premium を持つ。
    最後の貸手=capital が負でも貸せる(nation 予算補填で近似=研究доク §1)。ログはしない
    (loan_grant/loan_repay/interest_paid はスチュワードが記録)。個々の融資は loans[agent_id]。"""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.capital = float(cfg.get("initial_capital", 0.0))
        self.loans_outstanding = 0.0
        self.write_offs = 0.0
        self.premium = 0.0                                # §1 貸倒で街全体の金利を一段上げる
        self.loans: dict[int, dict] = {}                  # agent_id -> loan dict

    def score(self, income, assets, repayment, arrears_days, has_group) -> float:
        return credit_score(income, assets, repayment, arrears_days, has_group, self.cfg)

    def grant(self, agent_id: int, principal: float, score: float, day: int) -> dict:
        """融資を実行し loans に登録(呼び出し前に loan_approved / loan_limit を確認)。"""
        loan = open_loan(principal, score, day, self.cfg, extra_premium=self.premium)
        self.loans[int(agent_id)] = loan
        self.capital -= float(principal)
        self.loans_outstanding += float(loan["total_due"])
        return loan

    def receive(self, amount: float) -> None:
        self.capital += float(amount)
        self.loans_outstanding = max(0.0, self.loans_outstanding - float(amount))

    def write_off(self, agent_id: int) -> float:
        """貸倒処理(破産接続時)。未回収残を損金計上し loans から除去。§1 ration で premium 上げ。"""
        loan = self.loans.pop(int(agent_id), None)
        if loan is None:
            return 0.0
        loss = float(loan.get("remaining", 0.0))
        self.loans_outstanding = max(0.0, self.loans_outstanding - loss)
        self.write_offs += loss
        self.premium += float(self.cfg.get("ration_premium_step", 0.0))
        return loss


# -------------------------------------------------------------------- E-W2 VC
def build_vc_cfg(raw) -> dict:
    """E-W2 VC/出資(§4)。既定 OFF=完全 no-op(vc_investment 0 件=バイト一致)。

    判定は観測可能な代理変数のみ(traction=売上履歴 / market=在館数 / network=関係次数)。
    efficacy 等の内部構成概念(conviction)は使わない=研究доク §7【要注意 B】の k 非依存に従う。
    出資判定・配当ロジックは commerce.py(occupancy と同居=市場代理の近傍)に置く。"""
    raw = _to_container(raw)
    w = {"traction": 0.5, "market": 0.3, "network": 0.2}   # §4(conviction は除外=k安全)
    w.update({str(k): float(v) for k, v in dict(raw.get("weights", {})).items()})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "review_period_days": int(raw.get("review_period_days", 5)),  # 定期審査の周期
        "ticket": float(raw.get("ticket", 50000.0)),          # 1件の出資額(§4 圧縮スケール)
        "n_deals_per_review": int(raw.get("n_deals_per_review", 1)),  # 1周期の件数(希少=競争)
        "weights": w,
        "threshold": float(raw.get("threshold", 0.6)),        # 正規化スコア≥これ で出資
        "equity_share": float(raw.get("equity_share", 0.2)),  # 出資と引換の持分
        "dividend_rate": float(raw.get("dividend_rate", 0.5)),  # 以後の売上×持分×これ を配当
        "fund_initial": float(raw.get("fund_initial", 1000000.0)),  # ファンド原資(枯れると停止)
        "traction_ref": float(raw.get("traction_ref", 8000.0)),  # 正規化基準(career 転換閾値近傍)
        "market_ref": float(raw.get("market_ref", 6.0)),      # 在館数の基準(commerce 混雑近傍)
        "network_ref": float(raw.get("network_ref", 10.0)),   # 関係次数の基準
    }
