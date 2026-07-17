"""制度値(institutions)の正準化(基盤モデル抽出 D1-W3・第35バッチ)。

「機構=基盤・値=環境」の分離の第3段。government.py に **コード埋没** していた日本の制度値
(所得税ブラケット・消費税率と国地方按分・住民税率と区都按分・年間労働日・区/都/国の予算初期値・
区の生活困窮者給付)を、conf/config.yaml の ``institutions`` ブロック(既定=現行コード値)へ
外出しし、ここで1回だけ正準化して government へ渡す(深い get を散らさない=envpack.build_cfg 流儀)。

3層設計(docs/plans/env-classification.md):
  ② 共有参照(ref/institutions_jp.yaml)= 国・都道府県スコープの値。env.yaml の pref セレクタで引く。
  ③ 環境固有 = 市区町村・地点スコープの値(区の予算・給付など)。

このモジュール自身は地名リテラルを持たない(基盤層)。実値は必ず conf/config.yaml の
``institutions`` ブロック(既定=現行コード値)から与える。W4 の EnvPack ローダが
env.yaml の ``institutions.pref: tokyo`` を ref から解決してこのブロックへ展開する。

既定値の出所(=現行の government.py 埋没値。ゴールデン/本番の挙動を一切変えないため一致必須):
  income_brackets            所得税 実効税率(額面年収レンジ→実効税率の近似・累進の圧縮)
  consumption_rate/…         消費税 標準10%/軽減8%・国:地方=78:22・軽減対象 food
  resident_rate/ward_share   住民税 10%(区6%+都4%)・区:都=6:4
  annual_workdays            年間労働日 245(日給→年換算)
  ward/metro/nation_initial  予算初期残高(年予算÷人口×100体スケール)
  benefit_threshold/amount   区の生活困窮者給付(所持金がこの額未満で給付)
"""
from __future__ import annotations

# ---- 既定値(=government.py の現行コード値。institutions ブロック未指定時に効く)----
# 所得税 実効税率ブラケット。up_to=None(最後)は上限なし。年換算は日給×annual_workdays。
# ※ ref/institutions_jp.yaml の national.income_tax_effective と一致させる(値の二重管理のズレ検知)。
_DEFAULT_INCOME_BRACKETS: list[dict] = [
    {"up_to": 2_000_000, "rate": 0.02},
    {"up_to": 3_300_000, "rate": 0.03},
    {"up_to": 5_000_000, "rate": 0.05},
    {"up_to": 7_000_000, "rate": 0.07},
    {"up_to": 10_000_000, "rate": 0.11},
    {"up_to": None, "rate": 0.20},
]
_DEFAULT_CONSUMPTION_RATE = 0.10
_DEFAULT_CONSUMPTION_REDUCED_RATE = 0.08
_DEFAULT_CONSUMPTION_NATIONAL_SHARE = 0.78   # 国:地方=78:22(地方消費税=消費税額×22/78)
_DEFAULT_REDUCED_CATS = ["food"]
_DEFAULT_RESIDENT_RATE = 0.10                 # 住民税 標準税率(区6%+都4%)
_DEFAULT_RESIDENT_WARD_SHARE = 0.6            # 区:都=6:4(特別区の配分近似)
_DEFAULT_ANNUAL_WORKDAYS = 245
_DEFAULT_WARD_INITIAL = 60_000_000            # 渋谷区 1,468.73億÷24.4万×100(市区町村スコープ=③)
_DEFAULT_METRO_INITIAL = 65_000_000           # 東京都 9.158兆÷1,400万×100(都道府県スコープ)
_DEFAULT_NATION_INITIAL = 92_000_000          # 国 115兆÷1.25億×100(国スコープ)
_DEFAULT_BENEFIT_THRESHOLD = 2000             # 区の生活困窮者支援(市区町村スコープ=③)
_DEFAULT_BENEFIT_AMOUNT = 3000


def _brackets(raw_brackets) -> list[dict]:
    """所得税ブラケットを正準化。up_to は None(上限なし)or float。rate は float。"""
    if not raw_brackets:
        return [dict(b) for b in _DEFAULT_INCOME_BRACKETS]
    out: list[dict] = []
    for b in raw_brackets:
        b = dict(b)
        up = b.get("up_to")
        out.append({
            "up_to": (None if up in (None, "", "null") else float(up)),
            "rate": float(b["rate"]),
        })
    return out


def build_cfg(raw: dict | None) -> dict:
    """conf の ``institutions`` ブロックを正準化する。raw が None/空でも現行コード値で成立する。

    返す dict は government の実行時 cfg にそのまま流し込めるフラット形(ward_initial 等の平キー)。
    government.build_government_cfg がこれを既定として受け、``enabled`` を被せて Government を作る。
    """
    raw = dict(raw or {})
    consumption = dict(raw.get("consumption", {}) or {})
    resident = dict(raw.get("resident", {}) or {})
    labor = dict(raw.get("labor", {}) or {})
    budget = dict(raw.get("budget", {}) or {})
    benefit = dict(raw.get("benefit", {}) or {})

    reduced_cats = consumption.get("reduced_cats", raw.get("reduced_cats"))
    reduced_cats = ([str(c) for c in reduced_cats] if reduced_cats
                    else list(_DEFAULT_REDUCED_CATS))

    return {
        "income_brackets": _brackets(raw.get("income_brackets")),
        "consumption_rate": float(consumption.get("rate", _DEFAULT_CONSUMPTION_RATE)),
        "consumption_reduced_rate": float(
            consumption.get("reduced_rate", _DEFAULT_CONSUMPTION_REDUCED_RATE)),
        "consumption_national_share": float(
            consumption.get("national_share", _DEFAULT_CONSUMPTION_NATIONAL_SHARE)),
        "reduced_cats": reduced_cats,
        "resident_rate": float(resident.get("rate", _DEFAULT_RESIDENT_RATE)),
        "resident_ward_share": float(
            resident.get("ward_share", _DEFAULT_RESIDENT_WARD_SHARE)),
        "annual_workdays": int(labor.get("annual_workdays", _DEFAULT_ANNUAL_WORKDAYS)),
        "ward_initial": float(budget.get("ward_initial", _DEFAULT_WARD_INITIAL)),
        "metro_initial": float(budget.get("metro_initial", _DEFAULT_METRO_INITIAL)),
        "nation_initial": float(budget.get("nation_initial", _DEFAULT_NATION_INITIAL)),
        "benefit_threshold": float(benefit.get("threshold", _DEFAULT_BENEFIT_THRESHOLD)),
        "benefit_amount": float(benefit.get("amount", _DEFAULT_BENEFIT_AMOUNT)),
    }
