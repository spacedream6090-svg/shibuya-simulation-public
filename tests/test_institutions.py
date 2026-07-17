"""制度値 institutions(基盤モデル抽出 D1-W3)のテスト。

検証:
- institutions.build_cfg の既定 = 現行 government.py コード値(=ゴールデン/本番の挙動を変えない)。
- conf/config.yaml の institutions ブロック既定 = コード既定(config とコードのズレ検知)。
- 国・都道府県スコープの値が ref/institutions_jp.yaml と一致(値の二重管理のズレ検知)。
- config の institutions 値を変えると government(税・給付)の挙動に反映される。
- institutions 未指定の build_government_cfg は従来挙動と同一(直接構築の既存テスト互換)。
検証は非LLM のみ。
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from society import institutions
from society.config import load_config
from society.government import Government, build_government_cfg

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "ref" / "institutions_jp.yaml"


# ----------------------------------------------------- 既定 = 現行コード値
def test_defaults_are_current_code_values():
    """institutions.build_cfg(None) が現行 government.py 埋没値と一致(ゴールデン死守の要)。"""
    c = institutions.build_cfg(None)
    assert c["income_brackets"] == [
        {"up_to": 2_000_000.0, "rate": 0.02},
        {"up_to": 3_300_000.0, "rate": 0.03},
        {"up_to": 5_000_000.0, "rate": 0.05},
        {"up_to": 7_000_000.0, "rate": 0.07},
        {"up_to": 10_000_000.0, "rate": 0.11},
        {"up_to": None, "rate": 0.20},
    ]
    assert c["consumption_rate"] == 0.10
    assert c["consumption_reduced_rate"] == 0.08
    assert c["consumption_national_share"] == 0.78
    assert c["reduced_cats"] == ["food"]
    assert c["resident_rate"] == 0.10
    assert c["resident_ward_share"] == 0.6
    assert c["annual_workdays"] == 245
    assert c["ward_initial"] == 60_000_000.0
    assert c["metro_initial"] == 65_000_000.0
    assert c["nation_initial"] == 92_000_000.0
    assert c["benefit_threshold"] == 2000.0
    assert c["benefit_amount"] == 3000.0


def test_base_config_institutions_equals_code_defaults():
    """基底 config.yaml の institutions ブロック既定 = コード既定(config とコードのズレ検知)。"""
    cfg = load_config([])
    raw = OmegaConf.to_container(cfg.institutions, resolve=True)
    assert institutions.build_cfg(raw) == institutions.build_cfg(None)


# ----------------------------------------------------- ref との一致(② 共有参照)
def test_ref_agreement_national_and_pref():
    """国・都道府県スコープの institutions 既定 = ref/institutions_jp.yaml(値の二重管理のズレ検知)。"""
    ref = OmegaConf.to_container(OmegaConf.load(REF), resolve=True)
    c = institutions.build_cfg(None)

    # 所得税ブラケット(国スコープ): ref は upto キー、config は up_to キー。
    ref_brackets = [{"up_to": b.get("upto"), "rate": float(b["rate"])}
                    for b in ref["national"]["income_tax_effective"]]
    got_brackets = [{"up_to": (None if b["up_to"] is None else float(b["up_to"])),
                     "rate": b["rate"]} for b in c["income_brackets"]]
    assert got_brackets == ref_brackets, "所得税ブラケットが ref と不一致"

    # 消費税(国スコープ)
    rc = ref["national"]["consumption"]
    assert c["consumption_rate"] == rc["rate"]
    assert c["consumption_reduced_rate"] == rc["reduced"]
    assert c["consumption_national_share"] == rc["national_share"]
    assert c["reduced_cats"] == list(rc["reduced_cats"])

    # 住民税: rate=国スコープ / ward_share=都(特別区)スコープ
    assert c["resident_rate"] == ref["national"]["resident_tax"]["rate"]
    split = ref["prefectures"]["tokyo"]["special_ward_resident_split"]
    assert c["resident_ward_share"] == split[0]
    assert abs((split[0] / split[1]) - 1.5) < 1e-9    # 区:都=6:4

    # 年間労働日(国スコープ)
    assert c["annual_workdays"] == ref["national"]["labor"]["annual_workdays"]


# ----------------------------------------------------- config → government への反映
def test_income_bracket_change_reflects_in_income_tax():
    """institutions.income_brackets を変えると所得税額が変わる(config が government を駆動)。"""
    base = Government(build_government_cfg({"enabled": True}))
    gross = 12000.0                                   # 会社員日給 → 年 245 万 → 既定 3% 帯
    assert abs(base.income_tax(gross) - gross * 0.03) < 1e-6

    hot = institutions.build_cfg({"income_brackets": [
        {"up_to": 3_300_000, "rate": 0.40}, {"up_to": None, "rate": 0.50}]})
    g = Government(build_government_cfg({"enabled": True}, institutions=hot))
    assert abs(g.income_tax(gross) - gross * 0.40) < 1e-6, "ブラケット変更が反映されない"


def test_consumption_rate_and_share_change_reflects():
    """institutions.consumption の税率・国地方按分を変えると消費税内訳が変わる。"""
    inst = institutions.build_cfg({"consumption": {
        "rate": 0.20, "reduced_rate": 0.05, "national_share": 0.5,
        "reduced_cats": ["food"]}})
    g = Government(build_government_cfg({"enabled": True}, institutions=inst))
    nat, loc, rate = g.consumption_tax(1200.0, "shop")   # 非 food = 標準 20%
    assert rate == 0.20
    ct = 1200.0 * 0.20 / 1.20
    assert abs((nat + loc) - ct) < 1e-6
    assert abs(nat - ct * 0.5) < 1e-6, "国:地方 按分が反映されない"


def test_benefit_and_budget_change_reflects():
    """給付額・予算初期残高が institutions から来る。"""
    inst = institutions.build_cfg({
        "benefit": {"threshold": 5000, "amount": 7000},
        "budget": {"ward_initial": 111, "metro_initial": 222, "nation_initial": 333}})
    g = Government(build_government_cfg({"enabled": True}, institutions=inst))
    assert g.cfg["benefit_threshold"] == 5000.0 and g.cfg["benefit_amount"] == 7000.0
    assert g.balance == {"ward": 111.0, "metro": 222.0, "nation": 333.0}


# ----------------------------------------------------- 後方互換
def test_government_none_institutions_matches_code_defaults():
    """institutions を渡さない build_government_cfg は現行コード既定と同一(既存テスト互換)。"""
    a = build_government_cfg({"enabled": True})
    b = build_government_cfg({"enabled": True},
                             institutions=institutions.build_cfg(None))
    assert a == b
    assert a["consumption_national_share"] == 0.78    # 新キーも既定で入る


def test_government_block_override_wins_over_institutions():
    """government ブロックの個別上書きは institutions より優先(config.yaml の後方互換注記)。"""
    inst = institutions.build_cfg(None)               # consumption_rate=0.10
    g = build_government_cfg({"enabled": True, "consumption_rate": 0.25}, institutions=inst)
    assert g["consumption_rate"] == 0.25
