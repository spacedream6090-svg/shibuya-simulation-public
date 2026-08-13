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
#   D1-W3: この値も含め制度値は institutions ブロック(既定=institutions.py の現行値)から来る。
#   下の module 定数は歴史的な既定の記録用。実際の按分は self.cfg["consumption_national_share"]。
_CONSUMPTION_NATIONAL_SHARE = 0.78


def _default_income_brackets() -> list[dict]:
    """額面年収レンジ別の**簡略実効所得税率**(国税・源泉徴収近似)。

    給与所得控除・基礎控除・社保控除があるため額面に対する実効税率は低め(docs 参照)。
    統計的な限界税率表(5/10/20/23/33/40/45%)ではなく「年収→実効税率」の近似である旨を明記。
    up_to=None(最後)は上限なし。年換算は日給×annual_workdays で行う(government cfg)。
    D1-W3: 既定の正準は institutions.build_cfg(None)(値の二重管理を避けここへ委譲)。
    """
    from . import institutions as _inst_mod
    return _inst_mod.build_cfg(None)["income_brackets"]


def build_government_cfg(raw: dict | None, institutions: dict | None = None) -> dict:
    """config(government ブロック + institutions ブロック)→ 実行時 dict。既定 **OFF**(現状不変)。

    D1-W3: 税率・予算初期値・給付など**制度値**は institutions ブロック(既定=現行コード値)から来る。
    ``institutions`` は simulation.py が institutions.build_cfg で1回だけ正準化した dict(平キー)。
    None のときは institutions.build_cfg(None)=現行コード既定を使う(直接構築する既存テスト互換)。
    ``government`` ブロック(raw)は ``enabled`` を持ち、個別の制度値上書きがあれば institutions より
    優先する(config.yaml の「ここで上書き可能」注記の後方互換)。

    初期残高は「年間予算 ÷ 人口 × 100体」でスケール換算(docs/research §7)。
    """
    from . import institutions as _inst_mod
    raw = dict(raw or {})
    # 制度値の基底 = institutions ブロック(正準済み dict)。未指定なら現行コード既定。
    gov = dict(institutions) if institutions is not None else _inst_mod.build_cfg(None)
    # 後方互換: government ブロックに個別の制度値上書きがあれば優先する。
    if raw.get("income_brackets"):
        gov["income_brackets"] = _inst_mod.build_cfg(
            {"income_brackets": raw["income_brackets"]})["income_brackets"]
    _casts = {
        "ward_initial": float, "metro_initial": float, "nation_initial": float,
        "resident_rate": float, "resident_ward_share": float,
        "annual_workdays": int, "consumption_rate": float,
        "consumption_reduced_rate": float, "consumption_national_share": float,
        "benefit_threshold": float, "benefit_amount": float,
    }
    for key, cast in _casts.items():
        if raw.get(key) is not None:
            gov[key] = cast(raw[key])
    if raw.get("reduced_cats"):
        gov["reduced_cats"] = list(raw["reduced_cats"])
    gov["enabled"] = bool(raw.get("enabled", False))          # ★既定 OFF
    return gov


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
        # 予算執行の係数(議会現実化 realism ON の予算承認フック。既定 1.0=従来と完全同一)。
        # 議会が区(ward)の期次予算を否決すると tools 側が cut_ratio(既定0.8)を書き込み、以後の
        # 区の歳出執行がその倍率に絞られる。区以外(都・国)は対象外。1.0 のときは expense が
        # amount を一切いじらない=バイト一致(ゴールデン維持)。
        self.exec_ratio: float = 1.0

    # ------------------------------------------------------------------ 会計
    def collect(self, level: str, amount: float) -> None:
        """歳入(税)を計上: 残高 + / 当日歳入 +。"""
        if amount <= 0:
            return
        self.balance[level] += amount
        self.day_rev[level] += amount

    def expense(self, level: str, amount: float) -> None:
        """歳出(公務員給与・給付)を計上: 残高 − / 当日歳出 +。

        予算執行フック(議会現実化 realism ON): 区(ward)の歳出は議会の予算承認で決まる
        exec_ratio を掛ける(否決なら cut_ratio へ絞られる)。既定 exec_ratio==1.0 のときは
        amount を一切変えない(amount×1.0 のバイト一致は避け、乗算そのものをスキップ)=従来と完全同一。"""
        if amount <= 0:
            return
        if level == "ward" and self.exec_ratio != 1.0:
            amount = amount * self.exec_ratio
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
    def income_tax(self, gross: float, annual: float | None = None) -> float:
        """所得税(国税)の源泉控除額。gross を年換算しレンジ別実効税率を掛ける。

        annual=None(既定・既存の全呼び出し)は従来どおり「日給 × annual_workdays」で
        年換算する = 1 バイトも変わらない。

        ★annual を渡す口(賃金多様性 WAGE で新設): 年換算の分母は**支給周期で違う**。
          日給を 245 倍するのは日給者にだけ正しく、月給まとめ(例 24 万)をそのまま 245 倍
          すると年収 5,880 万円扱いになって最高税率が掛かる(既存の欠陥)。月給・賞与は
          「その人の年収」を呼び出し側が渡し、税率だけをそこから引く(税額は gross×率)。
        """
        if gross <= 0:
            return 0.0
        annual = (gross * float(self.cfg["annual_workdays"])
                  if annual is None else float(annual))
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
        national = ct * float(self.cfg.get("consumption_national_share",
                                            _CONSUMPTION_NATIONAL_SHARE))
        return national, ct - national, rate             # 地方=残り(合計を厳密一致させる)
