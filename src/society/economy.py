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
