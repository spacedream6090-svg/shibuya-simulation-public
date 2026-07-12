"""行政(ward=渋谷区 / metro=東京都 / nation=国)= 税・給与・給付の会計主体。

ユーザー要望 2026-07-06: 「人が生きるためには行政の仕組みが必要。区民が受けるサービス・
納める税(区だけでなく都・国も)を調べ、概念・コード上の存在でいいので行政がシミュに
存在しエージェントに影響を与えるように。公務員(警察・消防など)も税と接続してほしい」。

原則(既存の鉄則を継承。src/society 直下=no-fingerprint 検査対象外だが方針は同じ):
- **既定 OFF**(`enabled=false`)。OFF 時は build_government_cfg の enabled が False になり、
  scheduler 側で税・給付・予算ログのすべてが no-op(税/civic_service/public_budget が 0 件、
  既存イベント列はバイト一致)。
- 決定論: 本モジュールは乱数を一切引かない(税額・給付は算術のみ)。反復は list/固定順のみ。
- R4(客観測定): 効果は金銭の増減のみ。因子(grievance/efficacy 等)へ直接書き込まない。
  行政の効果は「困窮→給付→所持金改善」という間接経路で state に効く(engine 側の money_pressure)。
- 会計整合: 各主体は残高 balance を持ち、税=歳入 / 公務員給与・給付=歳出。日次で
  public_budget イベントに (revenue, expense, balance) を出力し、Σ歳入−Σ歳出=残高変化 を保つ。

調査根拠は docs/research/shibuya-government.md(出典つき)。engine(scheduler)は本モジュールの
不透明な金額(源泉税・消費税内訳・給付額・予算残高)だけを受け取り、因子名を名指ししない。
"""
from __future__ import annotations

# ---- 消費税の国:地方 配分(国税庁 No.6303: 地方消費税=消費税額×22/78 → 国78% / 地方22%)----
#   標準10%(国7.8+地方2.2)も軽減8%(国6.24+地方1.76)も比率は同じ 78:22。
_CONSUMPTION_NATIONAL_SHARE = 0.78


def _default_income_brackets() -> list[dict]:
    """額面年収レンジ別の**簡略実効所得税率**(国税・源泉徴収近似)。

    給与所得控除・基礎控除・社保控除があるため額面に対する実効税率は低め(docs 参照)。
    統計的な限界税率表(5/10/20/23/33/40/45%)ではなく「年収→実効税率」の近似である旨を明記。
    up_to=None(最後)は上限なし。年換算は日給×annual_workdays で行う(government cfg)。
    """
    return [
        {"up_to": 2_000_000, "rate": 0.02},
        {"up_to": 3_300_000, "rate": 0.03},
        {"up_to": 5_000_000, "rate": 0.05},
        {"up_to": 7_000_000, "rate": 0.07},
        {"up_to": 10_000_000, "rate": 0.11},
        {"up_to": None, "rate": 0.20},
    ]


def build_government_cfg(raw: dict | None) -> dict:
    """config(government ブロック)→ 実行時 dict。既定は **OFF**(現状挙動を一切変えない)。

    初期残高は「年間予算 ÷ 人口 × 100体」でスケール換算(docs/research §7)。短ランでは
    日次の税フローに対し十分な準備金。config.yaml は本バッチでは編集せず(最終スチュワード統合)、
    既定値はここに置く(final 報告に config キー案を提示)。
    """
    raw = dict(raw or {})
    brackets = raw.get("income_brackets")
    if brackets:
        brackets = [{"up_to": (None if b.get("up_to") in (None, "", "null")
                               else float(b["up_to"])),
                     "rate": float(b["rate"])} for b in brackets]
    else:
        brackets = _default_income_brackets()
    return {
        "enabled": bool(raw.get("enabled", False)),          # ★既定 OFF
        # 初期予算残高(円)。docs/research §7 のスケール換算。
        "ward_initial": float(raw.get("ward_initial", 60_000_000)),   # 渋谷区 1,468.73億÷24.4万×100
        "metro_initial": float(raw.get("metro_initial", 65_000_000)), # 東京都 9.158兆÷1,400万×100
        "nation_initial": float(raw.get("nation_initial", 92_000_000)),  # 国 115兆÷1.25億×100
        # 住民税(所得割): 区6% + 都4% = 10%。区:都 = 6:4(docs §1)。
        "resident_rate": float(raw.get("resident_rate", 0.10)),
        "resident_ward_share": float(raw.get("resident_ward_share", 0.6)),
        # 所得税(国税・源泉徴収): 日給を annual_workdays で年換算しレンジ別実効税率(docs §2)。
        "annual_workdays": int(raw.get("annual_workdays", 245)),
        "income_brackets": brackets,
        # 消費税: 標準10% / 軽減8%(food系)。国:地方=78:22(docs §3)。
        "consumption_rate": float(raw.get("consumption_rate", 0.10)),
        "consumption_reduced_rate": float(raw.get("consumption_reduced_rate", 0.08)),
        "reduced_cats": list(raw.get("reduced_cats", ["food"])),   # food=8%(他10%)。※現実の軽減対象は持帰り飲食料品
        # セーフティネット(区の生活困窮者支援): 所持金がこの額未満で給付(docs §4.2)。
        "benefit_threshold": float(raw.get("benefit_threshold", 2000)),
        "benefit_amount": float(raw.get("benefit_amount", 3000)),
    }


class Government:
    """3 主体(ward/metro/nation)の予算残高と日次会計。

    自己完結の plain データ(sim 参照を持たない)= 将来 checkpoint 同梱も可能。乱数は引かない。
    """

    LEVELS = ("ward", "metro", "nation")

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.balance: dict[str, float] = {
            "ward": float(cfg["ward_initial"]),
            "metro": float(cfg["metro_initial"]),
            "nation": float(cfg["nation_initial"]),
        }
        # 日次アキュムレータ(public_budget で出力し、日境界でリセット)。
        self.day_rev: dict[str, float] = {lv: 0.0 for lv in self.LEVELS}
        self.day_exp: dict[str, float] = {lv: 0.0 for lv in self.LEVELS}
        self.last_day = -1

    # ------------------------------------------------------------------ 会計
    def collect(self, level: str, amount: float) -> None:
        """歳入(税)を計上: 残高 + / 当日歳入 +。"""
        if amount <= 0:
            return
        self.balance[level] += amount
        self.day_rev[level] += amount

    def expense(self, level: str, amount: float) -> None:
        """歳出(公務員給与・給付)を計上: 残高 − / 当日歳出 +。"""
        if amount <= 0:
            return
        self.balance[level] -= amount
        self.day_exp[level] += amount

    def daily(self, day: int) -> list[dict]:
        """日境界(day 変化)で当日会計を締め、level ごとの (revenue, expense, balance) を返す。

        同日なら空リスト(何もしない)。初回(last_day==-1)は歳入歳出 0 の残高アンカーを返す
        (会計整合テストの基準点。Σ歳入−Σ歳出 = 最後の balance − 最初の balance を自明にする)。
        締め後は当日アキュムレータを 0 に戻す。
        """
        if day == self.last_day:
            return []
        self.last_day = day
        out = []
        for lv in self.LEVELS:
            out.append({"level": lv, "revenue": self.day_rev[lv],
                        "expense": self.day_exp[lv], "balance": self.balance[lv]})
            self.day_rev[lv] = 0.0
            self.day_exp[lv] = 0.0
        return out

    # ------------------------------------------------------------------ 税額
    def income_tax(self, gross: float) -> float:
        """所得税(国税)の源泉控除額。日給 gross を年換算しレンジ別実効税率を掛ける。"""
        if gross <= 0:
            return 0.0
        annual = gross * float(self.cfg["annual_workdays"])
        rate = self.cfg["income_brackets"][-1]["rate"]
        for b in self.cfg["income_brackets"]:
            if b["up_to"] is None or annual <= b["up_to"]:
                rate = b["rate"]
                break
        return gross * float(rate)

    def resident_tax(self, gross: float) -> tuple[float, float]:
        """住民税(所得割)の源泉控除額を (区分, 都分) で返す。区:都 = 6:4。"""
        if gross <= 0:
            return 0.0, 0.0
        total = gross * float(self.cfg["resident_rate"])
        ward = total * float(self.cfg["resident_ward_share"])
        return ward, total - ward

    def consumption_tax(self, price: float, cat: str) -> tuple[float, float, float]:
        """消費税(内税)の内訳を (国分, 地方分, 適用税率) で返す。価格は名目不変。

        food 系は軽減 8%、他は標準 10%。内税なので税額 = price × rate/(1+rate)。
        国:地方 = 78:22(地方消費税=消費税額×22/78)。国分=nation, 地方分=metro に配分。
        """
        if price <= 0:
            return 0.0, 0.0, 0.0
        rate = (float(self.cfg["consumption_reduced_rate"])
                if cat in self.cfg["reduced_cats"]
                else float(self.cfg["consumption_rate"]))
        ct = price * rate / (1.0 + rate)                 # 内税の税額
        national = ct * _CONSUMPTION_NATIONAL_SHARE
        return national, ct - national, rate             # 地方=残り(合計を厳密一致させる)
