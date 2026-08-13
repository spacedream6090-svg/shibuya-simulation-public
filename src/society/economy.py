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
- 賃金多様性 WAGE(既定 OFF): 職種適合額・支給周期・給料日・賞与の純関数群(末尾 §WAGE)。
"""
from __future__ import annotations

import hashlib

import numpy as np

# 職業 → 本業の賃金カテゴリ(economy.wages のキー)。None = 無給(学生の学業・無職)
WAGE_CAT: dict[str, str | None] = {
    "会社員": "会社員", "エンジニア": "会社員", "デザイナー": "会社員",
    "カフェ店員": "店員", "アパレル店員": "店員", "美容師": "店員",
    "大学生": None,                      # 学業は無給(バイトで稼ぐ)
    "フリーランス": "自営", "写真家": "自営", "バンドマン": "自営",
    "配達員": "自営", "無職": None,
    # ---- 街頭の生業(Wave 4: street_life レーンが増やした職業。2026-08-10)----
    # 賃金カテゴリは「雇用形態の粗い代理」であって職種の格付けではない。判定基準は 1 つだけ:
    #   固定職場を持ち雇われている=店員/会社員 / 固定職場を持たず出来高=自営 / 賃金労働でない=None。
    "ティッシュ配り": "店員",              # 販促代理店に時給で雇われる街頭スタッフ(固定拠点なしだが被用者)
    "ストリートミュージシャン": "自営",      # 投げ銭=出来高。既存「バンドマン」と同じ扱い
    "キッチンカー営業者": "自営",           # 移動販売の個人事業主(車両=自前の生産手段)
    "路上占い師": "自営",                 # 面談ごとの出来高。固定拠点なし
    "募金スタッフ": "店員",                # 街頭募金の多くは有給スタッフ(時給)。寄付金そのものは本人の所得ではない
    "路上支援員": "会社員",                # NPO・社会福祉法人・区委託のアウトリーチ常勤職員(固定給)
    # ★無給の 2 種(None = 勤務完遂で 1 円も入らない)。理由が違うので別々に書く。
    "街頭演説者": None,                   # 選挙運動は賃金労働ではない(公選法の運動員買収禁止)。政治活動に賃金を与えない
    "路上生活者": None,                   # 定職なし。持ち金レンジ(MONEY_INIT)だけで表現し、賃金経路には載せない
    # ---- 都市オペレーション(Wave 4: city_ops レーンが増やす L5 役割。2026-08-10)----
    # 公共の現業職は「どの主体に雇われているか」で分ける。区の清掃事務所・消防は公費、
    # ビルメンテナンス会社・物流会社は民間。★CIVIL_SERVANTS には入れない(下のコメント参照)。
    "清掃作業員": "区職員",                # 区の清掃事業(一般廃棄物)= 公費の現業職
    "救急隊員": "消防士",                  # 救急隊員は消防吏員(東京消防庁)= 消防士と同じ公安職の給与水準
    "納品ドライバー": "会社員",             # 運輸・物流会社の被用者(ギグの「配達員」= 自営 とは別物)
    "夜間清掃員": "店員",                  # ビルメンテナンス会社の深夜現業(時給ベース)。深夜割増は経済 v0 に無い
    # ---- 交通(Wave 4)----
    "車掌": "会社員",                     # 鉄道事業者の正社員。既存 L5 の「駅員/運転士」と同じ雇われ方
}

# ---- 公務員(行政・税・公務員バッチ 2026-07-06。docs/research/shibuya-government.md)----
# 職業 → 給与の出所となる行政主体。給与は government の日次ペイロール(scheduler)で予算から支給し、
# 源泉徴収(所得税+住民税)を掛ける。区職員=ward予算 / 警察官・消防士=metro予算(帰属の現実準拠:
# 警視庁・東京消防庁は東京都の公安職地方公務員=区職員ではない)。WAGE_CAT には入れない
# (勤務地写像 persona._pick_workplace は別バッチ所有で編集不可のため、通常の勤務完遂経路ではなく
#  government のペイロールで支給する=二重支給や gig 経路の混入を避ける)。
CIVIL_SERVANTS: dict[str, str] = {"区職員": "ward", "警察官": "metro", "消防士": "metro"}

# ★「清掃作業員」「救急隊員」は公費で働くが CIVIL_SERVANTS ではなく WAGE_CAT 側に置いた。理由:
#   CIVIL_SERVANTS 経路は government の日次ペイロール(予算 + 源泉徴収)を前提にしており、勤務地写像
#   persona._pick_workplace(別バッチ所有)へ職業を載せないと成立しない。この 2 種は city_ops が
#   L5 の役割として職場ごと与える層なので、通常の「勤務完遂 → 日給」経路の方が二重支給の危険がない。
#   給与水準だけを公務員のもの(区職員 / 消防士)から借りている。この 2 集合は**交わらない**
#   (tests/test_census_calibration.py が機械で固定)。


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
    # ---- 街頭の生業(Wave 4。レンジは WAGE_CAT の区分と揃える)----
    "ティッシュ配り": (10000, 40000),        # 時給の街頭スタッフ = 店員相当
    "ストリートミュージシャン": (5000, 30000),  # バンドマンと同じ(投げ銭の日銭暮らし)
    "キッチンカー営業者": (30000, 120000),    # 仕入れ・釣銭の運転資金を持ち歩く分だけ厚い
    "路上占い師": (10000, 50000),
    "募金スタッフ": (10000, 40000),
    "路上支援員": (30000, 100000),
    "街頭演説者": (20000, 80000),           # 無給だが生活基盤は別にある想定(既定レンジ相当)
    "路上生活者": (0, 3000),                # ★唯一、下限を 0 にする区分(所持金がほぼ無い状態を表す)
    # ---- 都市オペレーション(Wave 4)----
    "清掃作業員": (50000, 150000),          # 区職員相当の安定収入
    "救急隊員": (50000, 150000),            # 消防士相当
    "納品ドライバー": (30000, 100000),
    "夜間清掃員": (15000, 50000),
    # ---- 交通(Wave 4)----
    "車掌": (40000, 120000),
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
        # ---- 賃金多様性 WAGE(既定 OFF=誰も読まない=バイト一致)。末尾 §WAGE を参照 ----
        # ★最低賃金の床は wage_profile にも渡す(対象は全員「雇用される労働者」= 最低賃金法の
        #   適用対象)。min_wage_hourly=0.0(既定)なら床なし=従来と完全同一。
        "wage_profile": build_wage_profile_cfg(raw.get("wage_profile"),
                                               min_wage_hourly=min_wage),
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
#   k・内面構成概念(efficacy/grievance 等)を発火判断に食わせない(研究ドク §7【要注意 B】)。
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
    最後の貸手=capital が負でも貸せる(nation 予算補填で近似=研究ドク §1)。ログはしない
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
    efficacy 等の内部構成概念(conviction)は使わない=研究ドク §7【要注意 B】の k 非依存に従う。
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


# =============================================================== §WAGE 賃金多様性
# ユーザー要求(2026-08-13):「L2 の本業日給 0 の穴を実装で塞ぐ。ただし月給なのか日給か・
# 給料が振り込まれるタイミング・ボーナスの有無・職種に見合った金額か —— 給料にも多様性がある
# 実装に」。
#
# 何が穴だったか(実測):
#   ① プールの L2(域内従業者 224,240 人)のうち 198,264 人は職業名が WAGE_CAT(25 語)に
#      無いので `wage_amount` が 0 を返す = **本業の日給が構造的に 0 円**。
#   ② 残る層も wage は「会社員 12,000 / 店員 9,000 / 自営 10,000」の**実質 3 値**で、
#      台帳が持っている産業(14 種)・規模帯(8 区分)・役割(40 種)・職業名(146 種)の
#      情報が 1 つも賃金に反映されていなかった。
#   ③ 支給周期は「勤務完遂ごとの日割り」か「口座 ON の月給まとめ」の 2 択で、給料日は
#      全員同一(payday_dom=25)・賞与は存在しない。
#
# ここが持つのは**純関数だけ**(ログしない・sim を触らない・乱数を 1 粒も引かない)。
# 個体差・給料日・賞与の割当はすべて blake2b の安定ハッシュ(= プロセス跨ぎ・seed 非依存・
# resume 不変)で決める = **新しい乱数 stream を 1 本も足さない**。
# エンジン側の配線(日次清算・支給・搬送)は engine/scheduler.py の `_phase_wage_profile`。
#
# 金額の根拠(公表統計の水準に寄せた実額。圧縮スケールは掛けない):
#   産業別「きまって支給する現金給与額」(賃金構造基本統計調査・一般労働者)
#   企業規模別賃金格差(同上。小規模ほど低い)
#   賞与: 支給事業所割合は規模が大きいほど高く、支給月数も大きい(毎月勤労統計)
#   給料日: 25 日が最頻、次いで 15 日 / 10 日 / 月末 / 20 日(民間の一般的な慣行)
#   支給形態: 日本の常用労働者は圧倒的に月給制。**週給は作らない** — 週給制は米英の慣行で、
#     労働基準法 24 条「毎月 1 回以上・一定期日払い」の下では月給が既定。ここで週給を足すと
#     「多様性」ではなく非実在の制度を世界に置くことになる。

#: 産業キー(組織台帳 industry_key)→ 月額の基準(円)。所定内給与の水準。
INDUSTRY_MONTHLY: dict[str, float] = {
    "IT": 380000.0,     # 情報通信業
    "FI": 395000.0,     # 金融業・保険業
    "PS": 390000.0,     # 学術研究・専門・技術サービス業
    "CN": 355000.0,     # 建設業
    "RE": 345000.0,     # 不動産業・物品賃貸業
    "ED": 330000.0,     # 教育・学習支援業
    "MF": 320000.0,     # 製造業
    "WR": 310000.0,     # 卸売業・小売業
    "TR": 310000.0,     # 運輸業・郵便業
    "CS": 305000.0,     # 複合サービス事業
    "MW": 300000.0,     # 医療・福祉
    "SV": 285000.0,     # サービス業(他に分類されないもの)
    "AM": 280000.0,     # 娯楽業
    "LS": 275000.0,     # 生活関連サービス業
    "FB": 260000.0,     # 宿泊業・飲食サービス業
}

#: 規模帯(組織台帳 size.band)→ 月額の係数(企業規模間賃金格差)。
BAND_MULT: dict[str, float] = {
    "1-4": 0.84, "5-9": 0.88, "10-19": 0.92, "20-29": 0.95,
    "30-49": 0.98, "50-99": 1.00, "100-299": 1.06, "300+": 1.16,
}
_BAND_ORDER: tuple[str, ...] = ("1-4", "5-9", "10-19", "20-29",
                                "30-49", "50-99", "100-299", "300+")

#: 職種群 → 月額の係数(産業基準に対する相対)。役割・職業名の**内容**で決まる。
GROUP_MULT: dict[str, float] = {
    "exec": 1.30,          # 経営層
    "expert": 1.10,        # 有資格・高度専門(士業・医療専門職・技術者・設計/施工管理)
    "manager": 1.05,       # 現場の長(店長・教室長・司令)
    "general": 1.00,       # 総合職・企画・担当(区分の既定)
    "sales": 0.98,         # 営業・仕入・渉外
    "professional": 0.95,  # 専門職(意匠・制作・教員・調理・施術)
    "instructor": 0.78,    # 講師(学習塾・語学・音楽。教員より低い実態)
    "manual": 0.75,        # 現業(警備・保守・運搬・製造・仕分)
    "clerical": 0.72,      # 事務・受付・管理事務
    "service": 0.68,       # 接客・販売・補助
    "cleaning": 0.66,      # 清掃(現業のうち最も低い層)
}
_GROUP_DEFAULT = "general"

#: 職業名 / 役割名 → 職種群の判定表。**先頭から部分一致**(具体的な語ほど先に置く)。
#: ここに無い語は "general"。台帳に無い職業を発明しないのと同じ規律で、語は
#: data/organizations_shibuya_census.json の roles と persona プールの occupation にある語だけ。
OCC_GROUP_RULES: tuple[tuple[str, str], ...] = (
    # ---- 経営層 ----
    ("経営者", "exec"), ("支配人", "exec"), ("所長", "exec"),
    # ---- 現場の長(「店長」は「店員」より先に判定する)----
    ("店長", "manager"), ("教室長", "manager"), ("司令", "manager"),
    ("責任者", "manager"),
    # ---- 清掃(現業より先。「夜間清掃」「清掃作業員」を拾う)----
    ("清掃", "cleaning"),
    # ---- 有資格・高度専門 ----
    ("弁護士", "expert"), ("税理士", "expert"), ("会計士", "expert"),
    ("薬剤師", "expert"), ("看護師", "expert"), ("エンジニア", "expert"),
    ("開発", "expert"), ("設計", "expert"), ("施工管理", "expert"),
    ("技士", "expert"), ("アナリスト", "expert"), ("コンサルタント", "expert"),
    ("プロダクトマネージャー", "expert"),
    # ---- 講師(教員より先。塾/語学/音楽の講師は教員と水準が違う)----
    ("講師", "instructor"),
    # ---- 専門職 ----
    ("教員", "professional"), ("デザイナー", "professional"),
    ("ディレクター", "professional"), ("プランナー", "professional"),
    ("調理師", "professional"), ("職人", "professional"),
    ("美容師", "professional"), ("セラピスト", "professional"),
    ("整復師", "professional"), ("鍼灸師", "professional"),
    ("衛生士", "professional"), ("エステティシャン", "professional"),
    ("カメラマン", "professional"), ("技術者", "professional"),
    ("インストラクター", "professional"), ("登録販売者", "professional"),
    ("相談員", "professional"), ("スタイリスト", "professional"),
    ("クリーニング師", "professional"), ("プロデューサー", "professional"),
    # ---- 現業(「販売員」より先に「作業員」等を判定)----
    ("警備", "manual"), ("巡回", "manual"), ("保守", "manual"),
    ("運転手", "manual"), ("ドライバー", "manual"), ("配送", "manual"),
    ("配車", "manual"), ("倉庫", "manual"), ("作業員", "manual"),
    ("仕分", "manual"), ("荷役", "manual"), ("回収", "manual"),
    ("製造", "manual"), ("印刷", "manual"), ("仕上げ", "manual"),
    ("生産管理", "manual"), ("設備", "manual"),
    # ---- 営業・仕入 ----
    ("営業", "sales"), ("外交員", "sales"), ("バイヤー", "sales"),
    ("仕入", "sales"), ("販売運営", "sales"),
    ("コーディネーター", "sales"), ("アドバイザー", "sales"),
    # ---- 事務(「事務」は「金融事務」「医療事務」等も拾う)----
    ("事務", "clerical"), ("受付", "clerical"), ("オペレーター", "clerical"),
    ("管理員", "clerical"), ("在庫管理", "clerical"), ("レンタル管理", "clerical"),
    ("制作進行", "clerical"), ("パラリーガル", "clerical"),
    ("フロント", "clerical"), ("窓口", "clerical"), ("法務", "clerical"),
    ("財務", "clerical"), ("採用", "clerical"), ("コーポレート", "clerical"),
    # ---- 接客・販売・補助 ----
    ("販売員", "service"), ("店員", "service"), ("接客", "service"),
    ("ホール", "service"), ("キッチン", "service"), ("バリスタ", "service"),
    ("バーテンダー", "service"), ("フロア", "service"), ("音響", "service"),
    ("スタッフ", "service"), ("アシスタント", "service"),
    ("補助", "service"), ("助手", "service"), ("介護士", "service"),
    ("部員", "service"), ("係", "service"),
)

#: 支給形態が日給になりうる職種群(日雇い・日払いの慣行がある層)。
#: それ以外は必ず月給(労基法 24 条の「毎月 1 回以上・一定期日払い」の下では月給が既定)。
DAILY_GROUPS: frozenset[str] = frozenset({"service", "manual", "cleaning"})

#: 給料日(暦の何日)の分布。0 = 月末。25 日が最頻という民間の慣行に寄せた重み。
PAYDAY_WEIGHTS: tuple[tuple[int, float], ...] = (
    (10, 0.15), (15, 0.18), (20, 0.12), (25, 0.40), (0, 0.15),
)

#: 賞与の支給日候補(夏・冬とも上旬〜中旬が慣行)。給料日と重ならない日を選ぶ。
BONUS_DOMS: tuple[int, ...] = (5, 10, 15)


def build_wage_profile_cfg(raw, min_wage_hourly: float = 0.0) -> dict:
    """賃金多様性 WAGE の設定を正準化(既定 OFF=現行挙動と完全同一)。

    ON 時に効くもの(すべて決定論・乱数ゼロ・LLM 呼数不変):
      月額 = 産業基準 × 職種群係数 × 規模帯係数 × 個体差(±spread)
      日額 = 月額 / month_workdays(既存 `agent.wage` の意味=1 日ぶんの賃金 に載せる)
      支給周期 = 月給(既定)/ 日給(日雇い慣行のある職種群のうち daily_share)
      給料日 = 職場(org)ごとに {10,15,20,25,月末} へ分散(同じ会社の人は同じ給料日)
      賞与   = 職場ごとに支給有無・支給月・支給日・月数を割当(規模が大きいほど手厚い)

    calendar=true のときだけ「暦の実日付」で給料日・賞与月を判定する(要
    world.calendar.enabled)。false のときは口座 E5 と同じ「run 開始日=1 日」の
    30 日周期で判定する(= 短いランでも給料日が必ず来る)。

    min_wage_hourly(economy.min_wage_hourly の実効値。既定 0.0=床なし): 対象は全員
    「雇用される労働者」なので**最低賃金法の適用対象**。月額の床 = 時給床 × 8h ×
    month_workdays を張る(これが無いと零細飲食店の接客が法定水準を割る=実測で発見)。
    """
    raw = _to_container(raw)
    industry = dict(INDUSTRY_MONTHLY)
    industry.update({str(k): float(v)
                     for k, v in dict(raw.get("industry_monthly", {})).items()})
    group = dict(GROUP_MULT)
    group.update({str(k): float(v)
                  for k, v in dict(raw.get("group_mult", {})).items()})
    band = dict(BAND_MULT)
    band.update({str(k): float(v) for k, v in dict(raw.get("band_mult", {})).items()})
    bonus_raw = _to_container(raw.get("bonus"))
    month_workdays = max(1, int(raw.get("month_workdays", 20)))
    hours = float(raw.get("day_hours", 8.0))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "calendar": bool(raw.get("calendar", False)),
        "month_workdays": month_workdays,
        "day_hours": hours,
        "min_monthly": max(0.0, float(min_wage_hourly)) * hours * month_workdays,
        "spread": max(0.0, float(raw.get("spread", 0.15))),
        "default_monthly": float(raw.get("default_monthly", 330000.0)),
        "school_industry": str(raw.get("school_industry", "ED")),
        "default_band": str(raw.get("default_band", "10-19")),
        "daily_share": _clip01(float(raw.get("daily_share", 0.15))),
        "industry_monthly": industry,
        "group_mult": group,
        "band_mult": band,
        "payday_weights": _payday_weights(raw.get("payday_weights")),
        "bonus": {
            "enabled": bool(bonus_raw.get("enabled", True)),
            "summer_months": [int(m) for m in
                              (bonus_raw.get("summer_months") or (6, 7))],
            "winter_month": int(bonus_raw.get("winter_month", 12)),
            "doms": [int(d) for d in (bonus_raw.get("doms") or BONUS_DOMS)],
            "months_min": float(bonus_raw.get("months_min", 1.0)),
            "months_max": float(bonus_raw.get("months_max", 2.5)),
            "share_min": _clip01(float(bonus_raw.get("share_min", 0.35))),
            "share_max": _clip01(float(bonus_raw.get("share_max", 0.95))),
        },
    }


def _payday_weights(raw) -> tuple[tuple[int, float], ...]:
    """給料日の分布を正準化。dict {"25": 0.4, ...} / list [[25, 0.4], ...] のどちらも受ける。

    未指定なら PAYDAY_WEIGHTS(民間の慣行)。重みは合計 1 に正規化する(設定の書き間違いで
    「どの給料日にも当たらない個体」が出ないようにする)。日の昇順で畳むので決定論。"""
    if not raw:
        return PAYDAY_WEIGHTS
    if isinstance(raw, dict):
        pairs = [(int(k), float(v)) for k, v in raw.items()]
    else:
        pairs = [(int(p[0]), float(p[1])) for p in raw]
    pairs = [(d, w) for d, w in sorted(pairs) if w > 0.0]
    if not pairs:
        return PAYDAY_WEIGHTS
    total = sum(w for _d, w in pairs)
    return tuple((d, w / total) for d, w in pairs)


def stable_unit(salt: str, key: str) -> float:
    """(用途 salt, 安定キー)→ [0,1) の一様値。blake2b=決定論・RngHub 無風・プロセス跨ぎ安定。

    work._stable_uniform / ontology._stable_uniform と同流儀。**乱数 stream を 1 本も引かない**
    ので、ON にしても既存の draw 順・golden は一切動かない(R1)。"""
    h = hashlib.blake2b(f"{salt}\x1f{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def occupation_group(occupation: str, role: str = "") -> str:
    """職業名・役割名 → 職種群(GROUP_MULT のキー)。表に無ければ "general"。

    職業名を先に見る(台帳の role より具体的=第111 レーン B の職業対応表が入っている)。
    判定は**部分一致の先頭優先**なので、表の並び順が意味論そのものである
    (例: 「店長」を「店員」より先に置く / 「清掃」を現業より先に置く)。"""
    for text in (str(occupation or ""), str(role or "")):
        if not text:
            continue
        for word, grp in OCC_GROUP_RULES:
            if word in text:
                return grp
    return _GROUP_DEFAULT


def band_of(org: dict | None, cfg: dict) -> str:
    """組織 → 規模帯。学校・帯を持たない組織は cfg の既定帯。"""
    band = str(((org or {}).get("size") or {}).get("band") or "")
    return band if band in cfg["band_mult"] else str(cfg["default_band"])


def industry_of(org: dict | None, cfg: dict) -> str | None:
    """組織 → 産業キー。学校(school_type)は school_industry(既定 ED)。無所属は None。"""
    if not org:
        return None
    key = str(org.get("industry_key") or "")
    if key:
        return key
    if org.get("school_type"):
        return str(cfg["school_industry"])
    return None


def monthly_wage(industry_key, group: str, band: str, key: str, cfg: dict) -> float:
    """月額(円)= 産業基準 × 職種群係数 × 規模帯係数 × 個体差(1±spread)。

    個体差は安定ハッシュのみ(乱数ゼロ)。10 円単位へ丸める(給与明細の粒度)。
    最後に最低賃金の床(min_monthly。既定 0=床なし)で持ち上げる。"""
    base = float(cfg["industry_monthly"].get(str(industry_key or ""),
                                             cfg["default_monthly"]))
    m = base * float(cfg["group_mult"].get(group, GROUP_MULT[_GROUP_DEFAULT]))
    m *= float(cfg["band_mult"].get(str(band), 1.0))
    spread = float(cfg["spread"])
    if spread > 0.0:
        m *= 1.0 + spread * (2.0 * stable_unit("wage_indiv", key) - 1.0)
    return float(round(max(m, float(cfg.get("min_monthly", 0.0)), 0.0), -1))


def pay_period_of(group: str, band: str, key: str, cfg: dict) -> str:
    """支給形態 "monthly" / "daily"。日給になりうるのは DAILY_GROUPS かつ小規模職場のみ。

    日本の常用労働者は圧倒的に月給制なので、日給は**少数派**として置く
    (daily_share=0.15 = 対象群の 15%)。週給は日本の慣行に無いので作らない。"""
    if group not in DAILY_GROUPS:
        return "monthly"
    if str(band) not in ("1-4", "5-9"):          # 日払いの慣行は小規模職場に偏る
        return "monthly"
    return "daily" if stable_unit("wage_period", key) < float(cfg["daily_share"]) \
        else "monthly"


def payday_dom_of(key: str, cfg: dict | None = None) -> int:
    """給料日(暦の何日。0=月末)。同じ職場の人が同じ給料日になるよう key は org id を渡す。

    cfg を渡すと economy.wage_profile.payday_weights の分布を使う(未指定=民間の慣行)。"""
    weights = (cfg or {}).get("payday_weights") or PAYDAY_WEIGHTS
    u = stable_unit("wage_payday", str(key))
    acc = 0.0
    for dom, w in weights:
        acc += w
        if u < acc:
            return int(dom)
    return int(weights[-1][0])


def _band_rank(band: str) -> float:
    """規模帯 → [0,1] の位置(賞与の手厚さの補間に使う)。"""
    try:
        i = _BAND_ORDER.index(str(band))
    except ValueError:
        return 0.5
    return i / float(len(_BAND_ORDER) - 1)


def bonus_plan_of(key: str, band: str, payday: int, cfg: dict) -> dict | None:
    """賞与の計画(支給しない職場は None)。同じ職場の人は同じ支給月・支給日になる。

    支給の有無・月数は**規模帯**で決まる(規模が大きいほど支給割合も月数も高い実態)。
    支給日は給料日と重ならないものを選ぶ(同一 step に 2 本の賃金を重ねない設計)。"""
    bcfg = cfg["bonus"]
    if not bcfg["enabled"]:
        return None
    rank = _band_rank(band)
    share = float(bcfg["share_min"]) + (float(bcfg["share_max"])
                                        - float(bcfg["share_min"])) * rank
    if stable_unit("wage_bonus_on", str(key)) >= share:
        return None
    months = sorted(int(m) for m in bcfg["summer_months"]) or [6]
    summer = months[int(stable_unit("wage_bonus_mon", str(key))
                        * len(months)) % len(months)]
    doms = [int(d) for d in bcfg["doms"] if int(d) != int(payday)] \
        or [int(d) for d in bcfg["doms"]]
    dom = doms[int(stable_unit("wage_bonus_dom", str(key)) * len(doms)) % len(doms)]
    mult = float(bcfg["months_min"]) + (float(bcfg["months_max"])
                                        - float(bcfg["months_min"])) * rank
    return {"months": sorted({int(summer), int(bcfg["winter_month"])}),
            "dom": int(dom), "mult": float(round(mult, 3))}


def wage_plan(*, occupation: str, role: str, org: dict | None, org_id,
              key: str, cfg: dict) -> dict:
    """1 人ぶんの賃金プラン(決定論・乱数ゼロ)。engine はこの dict しか読まない。

    key = 個体の安定キー(pool の pid か agent.id)。同じ key・同じ台帳なら
    run.seed が変わっても同じプランになる(= resume / ローテーション再入で不変)。"""
    group = occupation_group(occupation, role)
    band = band_of(org, cfg)
    industry = industry_of(org, cfg)
    monthly = monthly_wage(industry, group, band, key, cfg)
    period = pay_period_of(group, band, key, cfg)
    org_key = str(org_id or "") or f"occ:{group}"       # 無所属は職種群で束ねる
    payday = payday_dom_of(org_key, cfg)
    bonus = bonus_plan_of(org_key, band, payday, cfg) if period == "monthly" else None
    workdays = int(cfg["month_workdays"])
    daily = float(round(monthly / workdays, 1))
    n_bonus = len(bonus["months"]) if bonus else 0
    annual = monthly * 12.0 + (monthly * bonus["mult"] * n_bonus if bonus else 0.0)
    return {"monthly": float(monthly), "daily": daily, "period": period,
            "payday": int(payday), "bonus": bonus, "group": group,
            "band": str(band), "industry": industry,
            "org": (str(org_id) if org_id else None),
            "annual": float(annual)}


def wage_eligible(occupation: str, role: str) -> bool:
    """賃金プランの対象か(**支給経路の二重化を避けるための排他判定**)。

    対象外にするもの:
      学生            … 賃金の受け手でない(organizations の役割名)
      公務員          … government の日次ペイロール(予算+源泉徴収)が既に払っている
      自営(WAGE_CAT) … `gig_profile` の日銭(出来高)が既に払っている
      明示的な無給    … WAGE_CAT が None(学業・政治活動・路上生活者)
    それ以外(WAGE_CAT に無い 198,264 人の台帳ロール名を含む)は対象。"""
    occ = str(occupation or "")
    if str(role or "") == "学生":
        return False
    if occ in CIVIL_SERVANTS:
        return False
    if occ in WAGE_CAT and WAGE_CAT[occ] is None:
        return False
    if WAGE_CAT.get(occ) == "自営":
        return False
    return True


#: `agent._wage_plan` の「未計算」を表す番人(None = 計算済みだが対象外)。
_WAGE_UNSET = "__unset__"


def wage_target(agent, record: dict | None = None) -> bool:
    """この個体に賃金プランを与えるか(= この街で雇われて働いているか)。

    条件は 3 つとも**観測可能な素性のみ**(k・内面構成概念を一切読まない):
      ① 職種が対象(`wage_eligible`。学生・公務員・自営・無給を除く)
      ② 勤務窓を持つ(work_start_min>=0。職場を持たない層は賃金が発生しない)
      ③ **この街で働いている**: 居住者 / org 配属 / 通勤者 のいずれか
    ③ が要るのは L4(非定期来街 707,778 人)を弾くため。彼らは `persona._pick_workplace`
    が職業名から勤務窓を与えてしまう(会社員 240,661 人)が、勤務先は渋谷ではない
    (record が visitor かつ commute=false かつ org_id 無し)。"""
    role = str(getattr(agent, "org_role", "") or (record or {}).get("role", "") or "")
    if not wage_eligible(getattr(agent, "occupation", ""), role):
        return False
    if int(getattr(agent, "work_start_min", -1)) < 0:
        return False
    if getattr(agent, "org_id", None):
        return True
    if not getattr(agent, "visitor", False):
        return True
    return bool(getattr(agent, "commute", False))


def assign_wage_plan(agent, orgs: dict | None, cfg: dict,
                     record: dict | None = None) -> dict | None:
    """賃金プランを 1 体へ与える(冪等・決定論・乱数ゼロ)。対象外は None。

    副作用は 2 つだけ: `agent._wage_plan`(プラン or None)と `agent.wage`
    (= 1 日ぶんの賃金。既存の意味にそのまま載せ替える)。**再来街(hydrate)や resume で
    何度呼んでも同じ値**になる(キーは pool の pid、無ければ agent.id)。

    `agent.wage` を書き換えるのは意図的で、IF-E2 の org 初期預金(配属者の日給合計 ×
    month_days × σ)・退職金(日給 × severance_days)・org の産出見積り(日給 ×
    revenue_margin)が全部この 1 欄を読んでいるため = 賃金の多様性がそこまで一貫して届く。"""
    # ★**肯定だけを記憶する**: 一度プランが付いた個体は同じものを返し続ける(冪等)。
    #   対象外の判定は毎回やり直す —— 対象外の理由の 1 つ「勤務窓が無い」は career の
    #   求職・再就職で**後から解消しうる**ので、否定をキャッシュすると「途中で職に就いた人が
    #   永久に無給」という静かなバグになる。再判定は文字列の辞書引き数回で済む。
    cached = getattr(agent, "_wage_plan", _WAGE_UNSET)
    if cached is not _WAGE_UNSET and cached is not None:
        return cached
    if not wage_target(agent, record):
        agent._wage_plan = None
        return None
    role = str(getattr(agent, "org_role", "") or (record or {}).get("role", "") or "")
    org_id = getattr(agent, "org_id", None)
    org = (orgs or {}).get(str(org_id)) if org_id else None
    key = str(getattr(agent, "pool_pid", "") or getattr(agent, "id", 0))
    plan = wage_plan(occupation=getattr(agent, "occupation", ""), role=role,
                     org=org, org_id=org_id, key=key, cfg=cfg)
    agent._wage_plan = plan
    agent.wage = float(plan["daily"])
    return plan


def salary_amount(plan: dict, worked_days: int, cfg: dict) -> float:
    """給料日に支給する月給(円)。所定労働日数に満たない月は**日割り**(実務どおり)。

    満勤(worked_days >= month_workdays)なら月額そのもの。残業・皆勤手当は持たない
    (この機構が主張するのは所定内給与の水準だけ)。"""
    n = max(0, int(worked_days))
    if n <= 0:
        return 0.0
    ratio = min(1.0, n / float(cfg["month_workdays"]))
    return float(round(float(plan["monthly"]) * ratio, 1))


def bonus_amount(plan: dict) -> float:
    """賞与 1 回ぶんの支給額(月額 × 支給月数)。賞与の無い職場は 0。"""
    b = plan.get("bonus")
    if not b:
        return 0.0
    return float(round(float(plan["monthly"]) * float(b["mult"]), 1))
