"""経済センサス較正(Wave 4 IV-1)= 組織台帳の産業構成を令和3年経済センサスへ接地する層のテスト。

方針(既存の鉄則を継承):
- **合成入力に対する純関数の単体テスト**が主。生 xlsx(data/realworld/ = .gitignore)にも
  組織台帳(data/ の生成物)にも依存しない形で、パーサ・集計・割付の算術を固定する。
  実データが手元にある環境でだけ動く検査は `pytest.mark.skipif` で明示的に飛ばす。
- 生成器の後方互換: `--census` 無指定の出力が**従来と 1 バイトも変わらない**ことを、
  コミット済み台帳 data/organizations_shibuya_wide11k.json との完全一致で固定する
  (--night-shifts と同じ線引き)。
- economy.WAGE_CAT / MONEY_INIT への追記は**既定ランでは誰も引かない**(不活性)ことと、
  そういうエージェントが居れば既存の賃金機構をそのまま通ることの両方を固定する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from society import economy

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS_JSON = REPO_ROOT / "data" / "realworld" / "census_r3" / "station_area_industry.json"
RAW_DIR = REPO_ROOT / "data" / "realworld" / "census_r3" / "raw"
LEDGER_11K = REPO_ROOT / "data" / "organizations_shibuya_wide11k.json"
WIDE_MAP = REPO_ROOT / "data" / "shibuya_osm_wide_v7.json"


def _load_script(name: str):
    """scripts/<name>.py を module として読む(scripts はパッケージではないので直読み)。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CAL = _load_script("calibrate_orgs_census")
ORG = _load_script("build_orgs")


# ============================================================ 合成 xlsx(標準ライブラリだけで作る)
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _col_ref(i: int) -> str:
    """0 起点の列番号 → `A` / `AA`(セル参照の列部分)。"""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def make_xlsx(rows: list[list[str]]) -> bytes:
    """行×セル(文字列)から最小限の xlsx バイト列を作る。

    値はすべて inlineStr で書く(sharedStrings を使わない)ので、read_xlsx_rows の
    inlineStr 経路と数値経路の両方が「文字列として」返ることを実際に踏む。
    """
    out = [f'<worksheet xmlns="{_NS}"><sheetData>']
    for r, row in enumerate(rows, start=1):
        out.append(f'<row r="{r}">')
        for c, val in enumerate(row):
            if val == "":
                continue
            esc = (str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            out.append(f'<c r="{_col_ref(c)}{r}" t="inlineStr"><is><t>{esc}</t></is></c>')
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/content-types"/>')
        zf.writestr("xl/workbook.xml",
                    f'<workbook xmlns="{_NS}"><sheets><sheet name="synth" sheetId="1"/>'
                    f'</sheets></workbook>')
        zf.writestr("xl/worksheets/sheet1.xml", "".join(out))
    return buf.getvalue()


def synth_dai_rows() -> list[list[str]]:
    """第28表(大分類×町丁目)の骨格を 2 分類 × 3 町丁目で再現した合成表。

    実表と同じ罠を全部踏ませる: ①表題行が『事業所数』『従業者数』を含む ②見出しがルビ連結
    ③結合セル由来の空欄 ④町行と丁目行の階層 ⑤皆無セル `-` ⑥末尾の注記行。
    """
    return [
        ["第28表　区市町村、町丁目、産業大分類別民営事業所数及び従業者数ベツ"],
        [""],
        ["区市町村\n町丁目", "", "", "総数ソウスウ", "", "産業大分類サンギョウダイブンルイ"],
        ["", "", "", "", "", "G\n情報通信業", "", "M\n宿泊業，飲食サービス業", ""],
        ["", "", "", "事業所数ジギョウショスウ", "従業者数ジュウギョウシャスウ",
         "事業所数ジギョウショスウ", "従業者数ジュウギョウシャスウ",
         "事業所数ジギョウショスウ", "従業者数ジュウギョウシャスウ"],
        ["", "", ""],
        ["架空区", "", "", "300", "3000", "100", "2000", "200", "1000"],
        ["", "甲町", "", "30", "300", "10", "200", "20", "100"],          # 町(丁目あり=末端でない)
        ["", "", "甲町１丁目", "20", "250", "8", "180", "12", "70"],
        ["", "", "甲町２丁目", "10", "50", "2", "20", "8", "30"],
        ["", "乙町", "", "5", "40", "1", "10", "4", "30"],                # 町(丁目なし=末端)
        ["", "丙町", "", "7", "-", "-", "-", "7", "60"],                  # 皆無セル
        ["注）　町丁目が不詳の事業所が存在するため、町丁目の合計は総数と一致しない場合がある。"],
    ]


def synth_chu_rows() -> list[list[str]]:
    """第30表(中分類×町丁目)の骨格を再現した合成表(4 行 1 組の町丁目ブロック)。"""
    head = ["区市町村\n町丁目\n事業所数\n従業者数", "", "", "", "", "総数",
            "G", "", "", "M", "", ""]
    codes = ["", "", "", "", "", "", "", "39", "40", "", "75", "76"]
    names = ["", "", "", "", "", "", "情報通信業", "情報サービス業",
             "インターネット附随サービス業", "宿泊業，飲食サービス業", "宿泊業", "飲食店"]

    def block(vals_est, vals_emp, vals_m, vals_f):
        return [
            ["", "", "", "事業所数ジギョウショスウ", ""] + vals_est,
            ["", "", "", "従業者数 ", ""] + vals_emp,
            ["", "", "", "", "男"] + vals_m,
            ["", "", "", "", "女"] + vals_f,
        ]

    rows = [["第30表　区市町村、町丁目、産業中分類別民営事業所数及び男女別従業者数"], [""],
            head, codes, names, ["", "", "", "", ""]]
    rows.append(["架空区", "", "", "", ""])
    rows += block(["300", "100", "60", "40", "200", "50", "150"],
                  ["3000", "2000", "1200", "800", "1000", "400", "600"],
                  ["1600", "1100", "700", "400", "500", "200", "300"],
                  ["1300", "900", "500", "400", "500", "200", "300"])
    rows.append(["", "甲町", "", "", ""])
    rows += block(["30", "10", "6", "4", "20", "5", "15"],
                  ["300", "200", "120", "80", "100", "40", "60"],
                  ["160", "110", "70", "40", "50", "20", "30"],
                  ["130", "90", "50", "40", "50", "20", "30"])
    rows.append(["", "", "甲町１丁目", "", ""])
    rows += block(["20", "8", "5", "3", "12", "3", "9"],
                  ["250", "180", "110", "70", "70", "30", "40"],
                  ["130", "100", "60", "40", "35", "15", "20"],
                  ["120", "80", "50", "30", "35", "15", "20"])
    rows.append(["", "", "甲町２丁目", "", ""])
    rows += block(["10", "2", "1", "1", "8", "2", "6"],
                  ["50", "20", "10", "10", "30", "10", "20"],
                  ["30", "10", "5", "5", "15", "5", "10"],
                  ["20", "10", "5", "5", "15", "5", "10"])
    rows.append(["", "乙町", "", "", ""])
    rows += block(["5", "1", "1", "-", "4", "1", "3"],
                  ["40", "10", "10", "-", "30", "10", "20"],
                  ["20", "5", "5", "-", "15", "5", "10"],
                  ["20", "5", "5", "-", "15", "5", "10"])
    rows.append(["注１）　従業者数の総数には、男女別が不詳の従業者を含む。"])
    return rows


# ============================================================ (1) パーサ(合成 xlsx)
def test_parse_dai_reads_hierarchy_and_classes():
    """大分類表: 表題行を見出しと誤認せず、区/町/丁目の 3 階層と分類別の対を読む。"""
    rows = CAL.read_xlsx_rows(make_xlsx(synth_dai_rows()))
    recs, n_missing = CAL.parse_dai(rows)
    by = {(r["block"], r["code"]): r for r in recs}
    assert by[("架空区", "総数")]["establishments"] == 300
    assert by[("架空区", "総数")]["level"] == "ward"
    assert by[("甲町１丁目", "G")]["establishments"] == 8
    assert by[("甲町１丁目", "G")]["employees"] == 180
    assert by[("甲町１丁目", "G")]["name"] == "情報通信業"
    assert by[("甲町１丁目", "G")]["level"] == "block"
    assert by[("甲町１丁目", "G")]["town"] == "甲町"
    assert by[("乙町", "M")]["level"] == "town" and by[("乙町", "M")]["employees"] == 30
    # 皆無セル(`-`)は 0 で埋めず None のまま数える
    assert by[("丙町", "G")]["establishments"] is None
    assert n_missing == 3                       # 丙町: 総数従業者 + G の 2 セル
    # 注記行はレコードにしない
    assert not [r for r in recs if r["block"].startswith("注")]


def test_parse_dai_rejects_table_without_pairs():
    """見出しが無い表は黙って空を返さず ValueError(欠測を偽値で埋めない掟)。"""
    rows = CAL.read_xlsx_rows(make_xlsx([["表題だけ"], ["", ""], ["甲町", "1"]]))
    with pytest.raises(ValueError):
        CAL.parse_dai(rows)


def test_parse_chu_reads_middle_classes_and_sex():
    """中分類表: 4 行 1 組(事業所数/従業者数/男/女)と 2 桁コード列を読む。"""
    rows = CAL.read_xlsx_rows(make_xlsx(synth_chu_rows()))
    recs, n_missing = CAL.parse_chu(rows)
    by = {(r["block"], r["code"], r["field"]): r["value"] for r in recs}
    assert by[("甲町１丁目", "39", "establishments")] == 5
    assert by[("甲町１丁目", "39", "employees")] == 110
    assert by[("甲町１丁目", "39", "employees_male")] == 60
    assert by[("甲町１丁目", "39", "employees_female")] == 50
    assert by[("甲町１丁目", "76", "establishments")] == 9
    assert by[("乙町", "40", "establishments")] is None      # 皆無セル
    assert n_missing == 4                                    # 乙町 40 の 4 フィールド
    # 大分類の小計列も別 class_level として拾う(第28表との突き合わせ用)
    mj = {(r["block"], r["code"]): r["value"] for r in recs
          if r["class_level"] == "大分類" and r["field"] == "employees"}
    assert mj[("甲町１丁目", "G")] == 180
    # 中分類には所属大分類が付く
    mids = {r["code"]: r["major"] for r in recs if r["class_level"] == "中分類"}
    assert mids["39"] == "G" and mids["76"] == "M"


# ============================================================ (2) 集計の算術
def _synth_records():
    dai, _ = CAL.parse_dai(CAL.read_xlsx_rows(make_xlsx(synth_dai_rows())))
    chu, _ = CAL.parse_chu(CAL.read_xlsx_rows(make_xlsx(synth_chu_rows())))
    return dai, chu


def test_leaf_blocks_excludes_towns_that_have_blocks():
    """丁目を持つ町(甲町)は末端ではない = 町行と丁目行の二重計上を構造的に防ぐ。"""
    dai, _ = _synth_records()
    leaves = CAL.leaf_blocks(dai)
    assert "甲町" not in leaves
    assert {"甲町１丁目", "甲町２丁目", "乙町", "丙町"} <= leaves


def test_aggregate_sums_only_leaf_rows():
    """エリア集計は末端行の単純和(町行を混ぜない)。"""
    dai, chu = _synth_records()
    agg = CAL.aggregate(dai, chu, ("甲町１丁目", "甲町２丁目", "乙町"))
    assert agg["area"]["establishments"] == 20 + 10 + 5
    assert agg["area"]["employees"] == 250 + 50 + 40
    assert agg["by_major"]["G"]["establishments"] == 8 + 2 + 1
    assert agg["by_major"]["M"]["employees"] == 70 + 30 + 30
    assert agg["by_middle"]["39"]["establishments"] == 5 + 1 + 1
    assert agg["by_middle"]["76"]["employees"] == 40 + 20 + 20
    assert agg["by_block"]["乙町"] == {"establishments": 5, "employees": 40}
    assert agg["checks"]["majors_sum_matches_total"] is True
    assert agg["checks"]["dai_vs_chu_match"] is True


def test_aggregate_rejects_unknown_block():
    """エリア指定に表へ無い町丁目(半角数字の取り違え等)があれば必ず落とす。"""
    dai, chu = _synth_records()
    with pytest.raises(ValueError, match="全角数字"):
        CAL.aggregate(dai, chu, ("甲町1丁目",))


def test_sim_buckets_split_major_by_middle_class():
    """大分類 N を中分類 78/79(LS)と 80(AM)へ割る写像が実際に効く。"""
    by_major = {"N": {"code": "N", "name": "生活関連", "establishments": 100, "employees": 1000}}
    by_middle = {
        "78": {"code": "78", "name": "洗濯理容", "major": "N", "establishments": 60, "employees": 400},
        "79": {"code": "79", "name": "その他生活", "major": "N", "establishments": 15, "employees": 100},
        "80": {"code": "80", "name": "娯楽業", "major": "N", "establishments": 25, "employees": 500},
    }
    buckets, unmapped = CAL.sim_buckets(by_major, by_middle)
    assert buckets["LS"]["establishments"] == 75 and buckets["LS"]["employees"] == 500
    assert buckets["AM"]["establishments"] == 25 and buckets["AM"]["employees"] == 500
    assert buckets["AM"]["census_only"] is True and buckets["LS"]["census_only"] is False
    # 台帳に受け皿の無い大分類は 0 で計上され、他バケットへ混ざらない
    assert set(unmapped) == set(CAL.UNMAPPED_MAJORS)
    assert all(d["establishments"] == 0 for d in unmapped.values())


# ============================================================ (3) 写像表そのもの(閉じた表)
def test_mapping_is_a_closed_list():
    """sim バケット ∪ 未写像大分類 = 経済センサス大分類 A..R の全体(取りこぼしゼロ・重複ゼロ)。"""
    majors = [chr(c) for c in range(ord("A"), ord("R") + 1)]
    covered: list[str] = []
    for b in CAL.SIM_BUCKETS:
        covered += b["majors"]
    # 中分類で割っているバケット(LS/AM)だけが同じ大分類を共有してよい
    shared = {m for m in covered if covered.count(m) > 1}
    assert shared == {"N"}
    for b in CAL.SIM_BUCKETS:
        if b["majors"] == ["N"]:
            assert b.get("middles"), f"大分類 N を共有するなら中分類の指定が要る: {b['key']}"
    assert set(covered) | set(CAL.UNMAPPED_MAJORS) == set(majors)
    assert not (set(covered) & set(CAL.UNMAPPED_MAJORS))


def test_mapping_keys_exist_in_generator():
    """写像先の industry_key は必ず生成器の業種定義に在る(綴り違いを機械で潰す)。"""
    known = {s["key"] for s in ORG.ALL_INDUSTRY_SPEC}
    assert {b["key"] for b in CAL.SIM_BUCKETS} <= known
    # census_only のキーは既定の INDUSTRY_SPEC に居てはならない(居ると既定分布が動く)
    default_keys = {s["key"] for s in ORG.INDUSTRY_SPEC}
    for b in CAL.SIM_BUCKETS:
        if b.get("census_only"):
            assert b["key"] not in default_keys


# ============================================================ (4) 割付の算術
def test_rescale_to_quota_hits_target_and_keeps_shape():
    """相対形を保ったまま合計が目標に一致し、どの社も 1 人以上になる。"""
    raw = [1, 1, 2, 40, 400]
    out = ORG._rescale_to_quota(raw, 1000)
    assert sum(out) == 1000
    assert all(v >= 1 for v in out)
    assert out == sorted(out)                 # 単調(大きい社ほど大きいまま)
    assert out[-1] > out[-2] * 5              # 裾の長さが潰れていない


def test_rescale_to_quota_degenerate_and_empty():
    assert ORG._rescale_to_quota([], 100) == []
    assert ORG._rescale_to_quota([5, 5, 5], 2) == [1, 1, 1]     # 目標 < 件数 → 全社 1 人
    assert sum(ORG._rescale_to_quota([1, 1, 1], 3)) == 3


def test_band_of_covers_all_bands():
    assert ORG._band_of(1) == "1-4"
    assert ORG._band_of(9) == "5-9"
    assert ORG._band_of(299) == "100-299"
    assert ORG._band_of(300) == "300+"
    assert ORG._band_of(99999) == "300+"       # 上限超も 300+ に丸める(帯を発明しない)


# ============================================================ (5) --census(合成センサス)
def _synth_census(buckets: dict[str, tuple[int, int]]) -> dict:
    return {"_meta": {"source": "合成", "attribution": "テスト用", "survey": {},
                      "area_definition": {"name": "合成エリア"}},
            "area": {"blocks": ["合成町"],
                     "establishments": sum(v[0] for v in buckets.values()),
                     "employees": sum(v[1] for v in buckets.values())},
            "sim_buckets": {k: {"industry": k, "census_majors": [], "census_middles": [],
                                "census_only": False,
                                "establishments": v[0], "employees": v[1]}
                            for k, v in buckets.items()}}


def test_census_mode_follows_industry_shares():
    """--census: 社数は事業所数シェア、産業別従業者総和は従業者シェアへ最大剰余法で一致する。"""
    census = _synth_census({"IT": (300, 30000), "FB": (600, 6000), "AM": (100, 4000)})
    companies = ORG.build_companies_dist(set(), 1000, 7, census=census)
    assert len(companies) == 1000
    by_key: dict[str, list[dict]] = {}
    for c in companies:
        by_key.setdefault(c["industry_key"], []).append(c)
    assert {k: len(v) for k, v in by_key.items()} == {"IT": 300, "FB": 600, "AM": 100}
    # 従業者: 目標総和 = 1000 * 40000/1000 = 40,000 をシェアで割付
    total = sum(int(c["size"]["employees"]) for c in companies)
    assert total == 40000
    assert sum(int(c["size"]["employees"]) for c in by_key["IT"]) == 30000
    assert sum(int(c["size"]["employees"]) for c in by_key["FB"]) == 6000
    assert sum(int(c["size"]["employees"]) for c in by_key["AM"]) == 4000
    # 規模帯ラベルは従業者数と矛盾しない(帯は駆動側ではなく派生ラベル)
    for c in companies:
        assert c["size"]["band"] == ORG._band_of(c["size"]["employees"])


def test_census_mode_is_deterministic():
    census = _synth_census({"IT": (300, 30000), "FB": (600, 6000), "AM": (100, 4000)})
    a = ORG.build_companies_dist(set(), 500, 3, census=census)
    b = ORG.build_companies_dist(set(), 500, 3, census=census)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_census_mode_emits_nightlife_and_service_workplaces():
    """娯楽業(AM)は nightlife、生活関連(LS)は service の職場カテゴリで生える。"""
    census = _synth_census({"AM": (100, 1000), "LS": (100, 1000), "WR": (100, 1000)})
    companies = ORG.build_companies_dist(set(), 300, 11, census=census)
    cat = {c["industry_key"]: ORG.SHIFT_BY_CAT for c in companies}   # 参照の存在確認
    assert cat
    am = [c for c in companies if c["industry_key"] == "AM"]
    ls = [c for c in companies if c["industry_key"] == "LS"]
    wr = [c for c in companies if c["industry_key"] == "WR"]
    assert am and ls and wr
    assert all(c["shift_pattern"] == ORG.SHIFT_BY_CAT["nightlife"] for c in am)
    assert all(c["shift_pattern"] == ORG.SHIFT_BY_CAT["service"] for c in ls)
    assert all(c["shift_pattern"] == ORG.SHIFT_BY_CAT["shop"] for c in wr)   # WR は据え置き
    # 娯楽業を切り出したので LS の表示名から「娯楽」が消える
    assert all(c["industry"] == "生活関連サービス業" for c in ls)
    assert all(c["industry"] == "娯楽業" for c in am)
    # nightlife は日跨ぎ(close < open)= 夜勤語彙と同じ読み方になる
    sp = ORG.SHIFT_BY_CAT["nightlife"]
    assert sp["close"] < sp["open"]


def test_census_mode_night_shift_uses_rescaled_size():
    """--census --night-shifts: 夜勤枠は**割り直したあと**の従業者数で判定される。"""
    census = _synth_census({"AM": (10, 1000)})
    companies = ORG.build_companies_dist(set(), 10, 5, night_shifts=True, census=census)
    assert all("night_shift" in c for c in companies), "AM は min_employees=3 を全社が満たす"
    for c in companies:
        assert c["size"]["employees"] >= ORG.CENSUS_NIGHT_SHIFT_BY_KEY["AM"]["min_employees"]
    # センサス専用の夜勤枠は既定の夜勤表へ漏れていない(別レーンの不変条件を壊さない)
    assert "AM" not in ORG.NIGHT_SHIFT_BY_KEY
    assert set(ORG.NIGHT_SHIFT_BY_KEY) <= {s["key"] for s in ORG.INDUSTRY_SPEC}
    assert ORG.night_shift_for("AM", 10)["open"] == "22:00"
    # OFF なら 1 社も生えない
    off = ORG.build_companies_dist(set(), 10, 5, night_shifts=False, census=census)
    assert not [c for c in off if "night_shift" in c]


def test_census_rejects_unknown_industry_key():
    """台帳に定義の無いキーが混ざったセンサス集計は黙って無視せず落とす。"""
    census = _synth_census({"IT": (10, 100), "ZZ": (10, 100)})
    with pytest.raises(ValueError, match="ZZ"):
        ORG.build_companies_dist(set(), 20, 1, census=census)


def test_census_skips_industries_with_zero_establishments():
    census = _synth_census({"IT": (10, 100), "MF": (0, 0)})
    companies = ORG.build_companies_dist(set(), 10, 1, census=census)
    assert {c["industry_key"] for c in companies} == {"IT"}


def test_load_census_requires_sim_buckets(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"area": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="sim_buckets"):
        ORG.load_census(p)


# ============================================================ (6) 無指定はバイト一致
@pytest.mark.skipif(not (LEDGER_11K.exists() and WIDE_MAP.exists()),
                    reason="正準台帳/広域地図が無い環境(生成物は .gitignore ではないが再生成要)")
def test_no_census_flag_is_byte_identical_to_committed_ledger():
    """--census を渡さない生成結果は、コミット済み 11k 台帳と**完全一致**する。

    (meta.night_shifts は Wave 4 III-1 が追加したキーで、台帳ファイルの再生成前から在る差分。
     ここで見るのは本バッチが 1 バイトも動かしていないこと = companies/schools の完全一致。)
    """
    old = json.loads(LEDGER_11K.read_text(encoding="utf-8"))
    new = ORG.build_ledger_dist(ORG.DEFAULT_WIDE_MAP, 11000, 42)
    assert json.dumps(new["companies"], ensure_ascii=False, sort_keys=False) == \
        json.dumps(old["companies"], ensure_ascii=False, sort_keys=False)
    assert new["schools"] == old["schools"]
    changed = {k for k in set(old["meta"]) | set(new["meta"])
               if old["meta"].get(k) != new["meta"].get(k)}
    assert changed == {"night_shifts"}, f"meta が想定外に動いた: {changed}"
    # 既定モードの meta にセンサス由来のキーは 1 つも生えない
    assert "census" not in new["meta"]
    assert new["meta"]["mode"] == "distribution-driven"
    assert new["meta"]["generator"] == "scripts/build_orgs.py --dist"


def test_no_census_flag_keeps_default_spec_untouched():
    """既定の業種表・カテゴリ表に、センサス専用の追加が漏れ出していない。"""
    assert "AM" not in {s["key"] for s in ORG.INDUSTRY_SPEC}
    assert {s["key"] for s in ORG.ALL_INDUSTRY_SPEC} - {s["key"] for s in ORG.INDUSTRY_SPEC} \
        == {s["key"] for s in ORG.CENSUS_EXTRA_INDUSTRY_SPEC}
    assert ORG.INDUSTRY_SPEC[ORG.INDUSTRY_SPEC.index(
        ORG.IND_BY_KEY["LS"])]["cat"] == "shop"           # 既定の LS は据え置き
    assert ORG._census_spec(ORG.IND_BY_KEY["LS"], False) is ORG.IND_BY_KEY["LS"]
    assert ORG._census_spec(ORG.IND_BY_KEY["LS"], True)["cat"] == "service"
    # place_on_buildings の差し替え引数は既定 None = 従来の cat を引く
    assert ORG.census_cat(ORG.IND_BY_KEY["LS"], False) == "shop"
    assert ORG.census_cat(ORG.IND_BY_KEY["LS"], True) == "service"


# ============================================================ (7) WAGE_CAT / MONEY_INIT の追記
NEW_OCCUPATIONS = (
    # street_life レーン(8)
    "ティッシュ配り", "ストリートミュージシャン", "キッチンカー営業者", "路上占い師",
    "街頭演説者", "募金スタッフ", "路上生活者", "路上支援員",
    # city_ops レーン(4)
    "清掃作業員", "納品ドライバー", "夜間清掃員", "救急隊員",
    # 交通(1)
    "車掌",
)


def test_new_occupations_are_registered_in_both_tables():
    for occ in NEW_OCCUPATIONS:
        assert occ in economy.WAGE_CAT, f"WAGE_CAT に {occ} が無い"
        assert occ in economy.MONEY_INIT, f"MONEY_INIT に {occ} が無い"


def test_wage_cat_values_resolve_in_default_economy():
    """WAGE_CAT の値(None 以外)は必ず economy.wages の実在キー = 0 円に黙って落ちない。"""
    econ = economy.build_economy(None)
    for occ, cat in economy.WAGE_CAT.items():
        if cat is None:
            continue
        assert cat in econ["wages"], f"{occ} → {cat} が wages に無い"
        assert float(econ["wages"][cat]) > 0.0


def test_wage_cat_and_civil_servants_are_disjoint():
    """公務員ペイロール経路と通常の賃金経路は交わらない(二重支給の構造的な禁止)。"""
    assert not (set(economy.WAGE_CAT) & set(economy.CIVIL_SERVANTS))


def test_new_occupations_resolve_through_wage_machinery():
    """合成エージェント(職場あり/なし)が既存の賃金機構をそのまま通る。"""
    econ = economy.build_economy(None)
    expected = {
        "ティッシュ配り": "店員", "ストリートミュージシャン": "自営", "キッチンカー営業者": "自営",
        "路上占い師": "自営", "募金スタッフ": "店員", "路上支援員": "会社員",
        "街頭演説者": None, "路上生活者": None,
        "清掃作業員": "区職員", "救急隊員": "消防士", "納品ドライバー": "会社員",
        "夜間清掃員": "店員", "車掌": "会社員",
    }
    assert expected == {occ: economy.WAGE_CAT[occ] for occ in NEW_OCCUPATIONS}
    for occ, cat in expected.items():
        got = economy.wage_amount(occ, True, econ)
        assert got == (0.0 if cat is None else float(econ["wages"][cat])), occ
        # 職場が無ければ本業の日給は出ない(既存規約)
        assert economy.wage_amount(occ, False, econ) == 0.0
        # 自営だけが gig プロファイルを持つ
        prof = economy.gig_profile(occ, econ)
        assert (prof is not None) == (cat == "自営"), occ


def test_new_occupations_initial_money_in_range():
    econ = economy.build_economy(None)
    rng = np.random.default_rng(0)
    for occ in NEW_OCCUPATIONS:
        lo, hi = economy.MONEY_INIT[occ]
        assert 0 <= lo <= hi
        for _ in range(20):
            m = economy.initial_money(occ, False, rng, econ)
            assert lo - 100 <= m <= hi + 100, f"{occ}: {m} が [{lo}, {hi}] から外れる"
    # 路上生活者だけが下限 0(意図的な唯一の例外)
    zeros = [o for o, (lo, _) in economy.MONEY_INIT.items() if lo == 0]
    assert zeros == ["路上生活者"]


def test_new_occupations_are_inert_for_default_runs():
    """既定ランの職業語彙には 1 語も混ざらない = 追記はゴールデンに触れない。"""
    for occ in NEW_OCCUPATIONS:
        assert occ not in economy.PART_TIME_OCC
        assert occ not in economy.CIVIL_SERVANTS
    roster = REPO_ROOT / "data" / "personas_100_inflow.json"
    if roster.exists():
        doc = json.loads(roster.read_text(encoding="utf-8"))
        personas = doc["personas"] if isinstance(doc, dict) else doc
        assert not ({p.get("occupation") for p in personas} & set(NEW_OCCUPATIONS))


# ============================================================ (8) 実データ(在るときだけ)
@pytest.mark.skipif(not (RAW_DIR / "ka21_dai.xlsx").exists(),
                    reason="生 xlsx は data/realworld/(.gitignore)= 取得済み環境でのみ検査")
def test_real_station_area_totals():
    """実表の駅周辺13町丁目 = 9,872 事業所 / 222,848 人(第28表と第30表が一致)。"""
    doc = CAL.build_document(RAW_DIR)
    assert doc["area"]["establishments"] == 9872
    assert doc["area"]["employees"] == 222848
    assert doc["checks"]["majors_sum_matches_total"] is True
    assert doc["checks"]["dai_vs_chu_match"] is True
    assert doc["by_major"]["G"]["employees"] == 52761        # 情報通信業
    assert doc["by_major"]["R"]["employees"] == 39304        # サービス業(他に分類されないもの)
    assert doc["by_major"]["I"]["employees"] == 28376        # 卸売業・小売業
    assert doc["by_major"]["M"]["employees"] == 22260        # 宿泊業・飲食サービス業
    assert doc["by_major"]["M"]["establishments"] == 1619
    assert doc["sim_buckets"]["AM"]["establishments"] == 254  # 娯楽業(中分類 80)
    assert doc["sim_buckets"]["LS"]["employees"] + doc["sim_buckets"]["AM"]["employees"] \
        == doc["by_major"]["N"]["employees"]


@pytest.mark.skipif(not CENSUS_JSON.exists(),
                    reason="集計 JSON は data/realworld/(.gitignore)= 生成済み環境でのみ検査")
def test_real_census_ledger_has_nightlife_workplaces():
    """実測センサスで台帳を組むと nightlife 職場が 254 社ぶん生える(接客の穴が閉じる)。"""
    census = ORG.load_census(CENSUS_JSON)
    n = int(census["area"]["establishments"])
    ledger = ORG.build_ledger_dist(ORG.DEFAULT_WIDE_MAP, n, 42, False, census, "test")
    cats = ledger["meta"]["census"]["workplace_cat_counts"]
    assert cats["nightlife"] == census["sim_buckets"]["AM"]["establishments"]
    assert cats["service"] > 0 and cats["food"] > 0 and cats["shop"] > 0 and cats["office"] > 0
    assert set(cats) == {"office", "shop", "food", "service", "nightlife"}
    st = ledger["meta"]["realized"]
    assert abs(st["employees_total"] - census["area"]["employees"]) <= n   # 丸め以内
