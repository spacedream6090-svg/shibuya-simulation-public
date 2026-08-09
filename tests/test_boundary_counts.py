"""P4 境界較正データのテスト(2026-08-10)。

対象は **scripts/rw_fetch/{transport_census,odpt_passenger_survey}.py と
scripts/build_boundary_counts.py だけ**。`src/` `conf/` には1バイトも触れていない。

★ **ネットワークには一切出ない。** HTTP の唯一の出口 `common._urlopen` を差し替え、
   フィクスチャは実データの「形だけ」を写した **完全な合成データ**を使う
   (実データの抜粋は1バイトもコミットしない = 再配布制限のあるソースが混ざるため)。

構成:
  A. xlsx      標準ライブラリだけの xlsx 読み(共有文字列・結合セル・欠番列)
  B. セル      ルビ剥がし・数値化・時間帯見出し
  C. パーサ    5表それぞれの純関数(見出し違いは例外・欠測は None + 計上)
  D. ODPT      includeAlighting の単位分け・改札共有の重複畳み・キー無しの無取得
  E. counts    正規化・スキーマ検証・counter_id の一意性・按分(derived)
  F. 境界      src/ が触らないこと・日次スケジューラに結線していないこと
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_boundary_counts as bbc  # noqa: E402
from rw_fetch import common  # noqa: E402
from rw_fetch import odpt_passenger_survey as ps  # noqa: E402
from rw_fetch import transport_census as tc  # noqa: E402


# ============================================================ 合成 xlsx を作る
_SHEET_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               "<sheetData>")


def _col_name(i: int) -> str:
    name = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        name = chr(65 + rem) + name
    return name


def make_xlsx(rows: list[list], sheet_name: str = "Sheet1") -> bytes:
    """行列 → 最小限の xlsx バイト列。文字列は共有文字列表に入れる(実ファイルと同じ形)。

    `None` を置いた列は **セルごと省略**する(結合セル/欠番列の再現)。
    """
    shared: list[str] = []
    index: dict[str, int] = {}
    body = [_SHEET_HEAD]
    for r, row in enumerate(rows, start=1):
        body.append(f'<row r="{r}">')
        for c, val in enumerate(row):
            if val is None:
                continue
            ref = f"{_col_name(c)}{r}"
            if isinstance(val, (int, float)):
                body.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                text = str(val)
                if text not in index:
                    index[text] = len(shared)
                    shared.append(text)
                body.append(f'<c r="{ref}" t="s"><v>{index[text]}</v></c>')
        body.append("</row>")
    body.append("</sheetData></worksheet>")

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    sst = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
           f'count="{len(shared)}" uniqueCount="{len(shared)}">'
           + "".join(f"<si><t>{esc(s)}</t></si>" for s in shared) + "</sst>")
    wb = ('<?xml version="1.0" encoding="UTF-8"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
          f'<sheets><sheet name="{esc(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", wb)
        zf.writestr("xl/sharedStrings.xml", sst)
        zf.writestr("xl/worksheets/sheet1.xml", "".join(body))
    return buf.getvalue()


# ============================================================ 合成フィクスチャ(実データではない)
# 実表の**列の形だけ**を写した架空の数字。実データの抜粋は使わない。
FIX_STATION_FLOW = [
    ["駅別発着・駅間通過人員表　　単位：人／日エキベツ"],
    [None, "定期券利用者テイキ", None, None, None, None, None,
     "普通券利用者", None, None, None, None, None, "合計ゴウケイ"],
    [None, "下　り", None, None, "上　り", None, None, "下　り", None, None,
     "上　り", None, None, "下　り", None, None, "上　り"],
    ["駅名エキメイ", "乗車ジョウシャ", "降車コウシャ", "通過ツウカ",
     "乗車ジョウシャ", "降車コウシャ", "通過ツウカ",
     "乗車ジョウシャ", "降車コウシャ", "通過ツウカ",
     "乗車ジョウシャ", "降車コウシャ", "通過ツウカ",
     "乗車ジョウシャ", "降車コウシャ", "通過ツウカ",
     "乗車ジョウシャ", "降車コウシャ", "通過ツウカ"],
    ["架空本線カクウホンセン"],                                  # 路線の小見出し(値ゼロ列)
    ["テスト駅", 100, 200, 300, 110, 210, 310, 10, 20, 30, 11, 21, 31,
     110, 220, 330, 121, 231, 341],
    ["よその駅", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ["湘南新宿ライン"],                                          # ルビ無しの路線名(剥がしすぎ検知)
    ["テスト駅", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, "", 18, 19, 20, 21, 22],
]

FIX_TIME_DIST = [
    ["利用目的別乗車降車時刻分布リヨウモクテキ", None, None, "単位：％タンイ"],
    ["利用目的リヨウモクテキ", "乗降ジョウコウ", "～6時台", "7時台", "8時台", "0時台～", "合計"],
    ["通勤ツウキン", "乗車ジョウシャ", 0.1, 0.4, 0.5, 0.0, 1],
    [None, "降車コウシャ", 0.05, 0.25, 0.7, 0.0, 1],
    ["合計ゴウケイ", "乗車ジョウシャ", 0.2, 0.3, 0.4, 0.1, 1],
    [None, "降車コウシャ", 0.1, 0.2, 0.6, 0.1, 1],
    ["注）これは注記行なので取り込まない"],
]

FIX_TRANSFER = [
    ["No.", "ターミナル名", "到着線名", "到着線\n（上り下り）", "行先線名", "行先線（上り下り）",
     "終日\n（人／日）", "ピーク時\n（人／1時間）", "ピーク時間ジカン"],
    ["1", "テスト駅", "初乗りハツノ", None, "A線", "上り", 1000, 400, "7:45～8:44"],
    [None, None, None, None, "B線", "下り", 500, 200, "7:45～8:44"],
    [None, None, "A線", "上り", "B線", "下り", 3000, 1500, "7:45～8:44"],
    [None, None, None, "下り", "B線", "上り", 2000, 900, "7:45～8:44"],
    [None, None, "A線", "上り", "最終降車", None, 4000, 1800, "7:45～8:44"],
    [None, None, None, None, None, "**　初乗り計　　*:", 1500, 600, None],
    [None, None, None, None, None, "**　乗換え計　　*:", 5000, 2400, None],
    ["2", "よその駅", "初乗り", None, "C線", "上り", 7, 7, "7:45～8:44"],
]

FIX_LINE_CAP = [
    ["路線別着時間帯別駅間輸送定員表ロセンベツ", None, None, "単位：人タンイニン"],
    ["事業者名ジギョウシャメイ", "路線名ロセンメイ", "方向ホウコウ", "発駅名ハツエキメイ",
     "着駅名チャクエキメイ", "始発～\n 6:59シハツ", "7:00～\n 7:29", "24:00～\n終発シュウハツ"],
    ["架空鉄道カクウテツドウ", "架空本線カクウホンセン", "下りクダ", "前の駅", "テスト駅", 100, 200, 300],
    [None, None, None, "テスト駅", "次の駅", 110, 210, 310],
    [None, None, None, "よそ1", "よそ2", 1, 2, 3],
]

FIX_OD13 = [
    ["■初乗り・最終降車駅間移動人員ハツノ"],
    ["出発駅シュッパツエキ", "到着駅トウチャクエキ", "移動人員イドウジンイン"],
    ["架空鉄道 テスト駅", "架空鉄道 よその駅", 100],
    ["架空鉄道 テスト駅", "別会社 よその駅", 50],
    ["別会社 テスト駅", "架空鉄道 よその駅", 25],
    ["架空鉄道 よその駅", "架空鉄道 テスト駅", 200],
    ["架空鉄道 よその駅", "架空鉄道 まったく別の駅", 999],
    # 部分一致で拾ってはいけない紛らわしい駅名(実データの 高座渋谷 に相当)。
    ["架空鉄道 前テスト駅", "架空鉄道 よその駅", 777],
]


# ============================================================ A. xlsx 読み
def test_read_xlsx_shared_strings_and_numbers():
    data = make_xlsx([["あ", 1, 2.5], ["い", "", 3]])
    rows = tc.read_xlsx_rows(data)
    assert rows[0][0] == "あ" and rows[0][1] == "1" and rows[0][2] == "2.5"
    assert rows[1][0] == "い"


def test_read_xlsx_missing_cells_become_empty():
    """結合セル由来の欠番列は空文字で埋まる(位置がずれない)。"""
    data = make_xlsx([["a", None, None, "d"]])
    assert tc.read_xlsx_rows(data) == [["a", "", "", "d"]]


def test_read_xlsx_rejects_non_zip():
    with pytest.raises(ValueError):
        tc.read_xlsx_rows(b"not a zip at all")


def test_read_xlsx_sheet_index_out_of_range():
    with pytest.raises(ValueError):
        tc.read_xlsx_rows(make_xlsx([["a"]]), sheet_index=5)


# ============================================================ B. セルの下ごしらえ
@pytest.mark.parametrize("raw,want", [
    ("駅名エキメイ", "駅名"),
    ("東海道本線トウカイドウホンセン", "東海道本線"),
    ("湘南新宿ライン", "湘南新宿ライン"),      # ★ ルビではないカタカナを剥がさない
    ("鶴見線（１）", "鶴見線（１）"),
    ("渋谷", "渋谷"),
    ("", ""),
])
def test_strip_ruby(raw, want):
    assert tc.strip_ruby(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("始発～6:59シハツ", "始発～6:59"),
    ("24:00～終発シュウハツ", "24:00～終発"),
    ("7:00～\n 7:29", "7:00～7:29"),
])
def test_clean_time_label(raw, want):
    assert tc.clean_time_label(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("121926", 121926), ("1,234", 1234), ("6.44E-2", 6.44e-2),
    ("", None), ("-", None), ("　", None), ("あ", None),
])
def test_to_number(raw, want):
    assert tc.to_number(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("～6時台", "hour:le06"), ("7時台", "hour:07"), ("23時台", "hour:23"),
    ("0時台～", "hour:ge00"),
])
def test_hour_bin(raw, want):
    assert bbc.hour_bin(raw) == want


def test_hour_bin_rejects_garbage():
    with pytest.raises(ValueError):
        bbc.hour_bin("なにか")


# ============================================================ C. パーサ
def test_parse_station_flow_shape_and_filter():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_STATION_FLOW))
    recs, n_missing = tc.parse_station_flow(rows, stations=("テスト駅",))
    assert {r["station"] for r in recs} == {"テスト駅"}
    # 2 行 × 18 列
    assert len(recs) == 36
    assert n_missing == 1                      # 空セルを1つ仕込んである
    hit = [r for r in recs if r["line"] == "架空本線" and r["ticket"] == "合計"
           and r["direction"] == "下り" and r["measure"] == "乗車"]
    assert len(hit) == 1 and hit[0]["value"] == 110
    # ルビ無しの路線名が剥がされていない
    assert "湘南新宿ライン" in {r["line"] for r in recs}


def test_parse_station_flow_all_stations_when_no_filter():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_STATION_FLOW))
    recs, _ = tc.parse_station_flow(rows)
    assert {r["station"] for r in recs} == {"テスト駅", "よその駅"}


def test_parse_station_flow_missing_header_raises():
    rows = tc.read_xlsx_rows(make_xlsx([["ぜんぜん違う表"], ["a", "b"]]))
    with pytest.raises(ValueError):
        tc.parse_station_flow(rows)


def test_parse_time_dist_bins_and_purposes():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_TIME_DIST))
    recs, n_missing = tc.parse_time_dist(rows)
    assert n_missing == 0
    assert len(recs) == 4 * 4                  # 2目的 × 乗降2 × 4区分
    assert {r["purpose"] for r in recs} == {"通勤", "合計"}
    assert {r["bin"] for r in recs} == {"～6時台", "7時台", "8時台", "0時台～"}
    # 注記行が purpose として紛れ込んでいない
    assert not any("注" in r["purpose"] for r in recs)
    for purpose in ("通勤", "合計"):
        for flow in ("乗車", "降車"):
            got = sum(r["share"] for r in recs if r["purpose"] == purpose and r["flow"] == flow)
            assert got == pytest.approx(1.0)


def test_parse_transfer_kinds_and_totals():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_TRANSFER))
    recs, _ = tc.parse_transfer(rows, terminals=("テスト駅",))
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    assert kinds == {"gate_entry": 2, "transfer": 2, "gate_exit": 1, "total": 2}
    # 小計は**行先線名ではなく行先線(上り下り)の列**に置かれている
    totals = {r["total_label"]: r["daily"] for r in recs if r["kind"] == "total"}
    assert totals == {"初乗り計": 1500, "乗換え計": 5000}
    # 到着線の方向は結合セルなので前方補完される
    down = [r for r in recs if r["kind"] == "transfer" and r["from_direction"] == "下り"]
    assert len(down) == 1 and down[0]["daily"] == 2000
    assert all(r["peak_window"] == "7:45～8:44" for r in recs if r["kind"] != "total")


def test_parse_transfer_terminal_filter():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_TRANSFER))
    recs, _ = tc.parse_transfer(rows, terminals=("よその駅",))
    assert {r["terminal"] for r in recs} == {"よその駅"}


def test_parse_line_capacity_matches_either_endpoint():
    rows = tc.read_xlsx_rows(make_xlsx(FIX_LINE_CAP))
    recs, _ = tc.parse_line_capacity(rows, stations=("テスト駅",))
    pairs = {(r["from_station"], r["to_station"]) for r in recs}
    assert pairs == {("前の駅", "テスト駅"), ("テスト駅", "次の駅")}
    assert {r["arrival_bin"] for r in recs} == {"始発～6:59", "7:00～7:29", "24:00～終発"}
    assert all(r["operator"] == "架空鉄道" and r["line"] == "架空本線" for r in recs)


def test_parse_station_od_exact_station_match():
    """`前テスト駅` を `テスト駅` に混ぜない(実データの 高座渋谷/渋谷 に相当)。"""
    rows = tc.read_xlsx_rows(make_xlsx(FIX_OD13))
    recs, _ = tc.parse_station_od(rows, stations=("テスト駅",))
    assert all("前テスト駅" not in (r["from_station"], r["to_station"]) for r in recs)
    dep = sum(r["passengers"] for r in recs if r["from_station"] == "テスト駅")
    arr = sum(r["passengers"] for r in recs if r["to_station"] == "テスト駅")
    assert dep == 175 and arr == 200
    assert {r["from_operator"] for r in recs} <= {"架空鉄道", "別会社"}


# ============================================================ D. ODPT PassengerSurvey
def _survey(same_as, stations, include_alighting, years):
    return {"owl:sameAs": same_as, "odpt:station": stations,
            "odpt:operator": "odpt.Operator:X",
            "odpt:includeAlighting": include_alighting,
            "odpt:passengerSurveyObject": [
                {"odpt:surveyYear": y, "odpt:passengerJourneys": v} for y, v in years.items()]}


def test_is_station_exact_suffix():
    rec = _survey("a", ["odpt.Station:JR-East.Yamanote.Shibuya"], False, {2024: 1})
    assert ps.is_station(rec, "Shibuya")
    other = _survey("b", ["odpt.Station:Odakyu.Enoshima.KozaShibuya"], True, {2024: 1})
    assert not ps.is_station(other, "Shibuya")


def test_dedupe_merges_shared_gate_records():
    """メトロ渋谷の半蔵門線/副都心線のように、駅集合も値も同一なら1件に畳む。"""
    stations = ["odpt.Station:M.Hanzomon.Shibuya", "odpt.Station:M.Fukutoshin.Shibuya"]
    a = _survey("S:M.Fukutoshin.Shibuya", stations, True, {2024: 751998})
    b = _survey("S:M.Hanzomon.Shibuya", list(reversed(stations)), True, {2024: 751998})
    kept, dropped = ps.dedupe_records([a, b])
    assert len(kept) == 1 and dropped == ["S:M.Hanzomon.Shibuya"]
    assert kept[0]["_merged_from"] == ["S:M.Hanzomon.Shibuya"]


def test_dedupe_keeps_genuinely_different_records():
    """東急の東横線/田園都市線は別改札の別計上 = 畳んではいけない。"""
    a = _survey("S:Tokyu.Toyoko.Shibuya", ["odpt.Station:Tokyu.Toyoko.Shibuya"], True, {2024: 1})
    b = _survey("S:Tokyu.DenEnToshi.Shibuya",
                ["odpt.Station:Tokyu.DenEnToshi.Shibuya"], True, {2024: 2})
    kept, dropped = ps.dedupe_records([a, b])
    assert len(kept) == 2 and dropped == []


def test_to_records_measure_follows_include_alighting():
    op = ps.OPERATOR_BY_KEY["jr_east"]
    boarding = ps.to_records([_survey("x", ["s.Shibuya"], False, {2024: 100})], op)[0]
    both = ps.to_records([_survey("y", ["s.Shibuya"], True, {2024: 100})], op)[0]
    assert boarding["measure"] == "boarding"
    assert both["measure"] == "boarding_alighting"
    assert boarding["latest_year"] == 2024 and boarding["latest_value"] == 100


def test_common_year_and_totals_align_years():
    """事業者ごとに系列の開始年が違うので、合計は必ず共通年度でとる。"""
    op = ps.OPERATOR_BY_KEY["jr_east"]
    recs = ps.to_records([_survey("a", ["s.Shibuya"], False, {2023: 10, 2024: 11})], op)
    recs += ps.to_records([_survey("b", ["s.Shibuya"], True, {2024: 20, 2025: 21})],
                          ps.OPERATOR_BY_KEY["keio"])
    assert ps.common_year(recs) == 2024
    assert ps.totals_for_year(recs, 2024) == {"boarding": 11, "boarding_alighting": 20}
    summary = ps.summarize(recs)
    assert summary["common_year"] == 2024
    # 最新年度は事業者でずれる = latest 合計は年度が混ざる。警告が残っていること。
    assert "足してはいけない" in summary["warning"]


def test_year_series_skips_broken_entries():
    rec = {"odpt:passengerSurveyObject": [
        {"odpt:surveyYear": 2024, "odpt:passengerJourneys": 5},
        {"odpt:surveyYear": "bad", "odpt:passengerJourneys": 9},
        "not-a-dict"]}
    assert ps.year_series(rec) == {2024: 5}


def test_fetch_without_keys_writes_nothing(tmp_path, monkeypatch):
    """キーが1つも無ければ **1リクエストも出さず・1ファイルも書かない**。"""
    calls = []

    def _boom(req, timeout):                     # pragma: no cover - 呼ばれたら失敗
        calls.append(req)
        raise AssertionError("キーが無いのに HTTP に出た")

    monkeypatch.setattr(common, "_urlopen", _boom)
    common.reset_request_count()
    doc, rows = ps.fetch_all(tmp_path, open_key="", challenge_key="")
    assert doc is None and calls == [] and common.request_count() == 0
    assert not list(tmp_path.rglob("*.json"))
    assert rows and all(not r["ok"] for r in rows)
    assert all("key-missing" in (r["error"] or "") for r in rows)


def test_plan_never_contains_a_key():
    common.register_secret("SUPERSECRETKEY123456")
    try:
        for _, url in ps.plan():
            assert "SUPERSECRETKEY123456" not in url
            assert common.MASK in url
    finally:
        common.clear_secrets()


def test_display_url_masks_but_api_url_carries_key():
    params = {"odpt:operator": "odpt.Operator:Tokyu"}
    assert common.MASK in ps.display_url("open", params)
    assert "KEY123456" in ps.api_url("open", params, "KEY123456")
    # ★ 保存経路に載るのは display 側だけ。scrub を通せば消える。
    common.register_secret("KEY123456")
    try:
        assert "KEY123456" not in common.scrub(ps.api_url("open", params, "KEY123456"))
    finally:
        common.clear_secrets()


# ============================================================ E. counts の正規化
def _docs_from_fixtures():
    """合成 xlsx をパースして、取得器が書くのと同じ形の文書を組む(HTTP 不使用)。"""
    out = {}
    for key, fixture in (("station_flow", FIX_STATION_FLOW), ("time_dist", FIX_TIME_DIST),
                         ("transfer", FIX_TRANSFER), ("line_capacity", FIX_LINE_CAP),
                         ("station_od13", FIX_OD13)):
        table = tc.TABLE_BY_KEY[key]
        rows = tc.read_xlsx_rows(make_xlsx(fixture))
        recs, _ = tc.run_parser(table, rows, ("テスト駅",))
        out[key] = {"_meta": {"survey_year": table["survey_year"], "regime": table["regime"]},
                    "data": recs}
    return out


def _survey_doc():
    op = ps.OPERATOR_BY_KEY["tokyu"]
    recs = ps.to_records([
        _survey("odpt.PassengerSurvey:Tokyu.Toyoko.Shibuya",
                ["odpt.Station:Tokyu.Toyoko.Shibuya"], True, {2023: 10, 2024: 11}),
        _survey("odpt.PassengerSurvey:Tokyu.DenEnToshi.Shibuya",
                ["odpt.Station:Tokyu.DenEnToshi.Shibuya"], True, {2023: 20, 2024: 21}),
    ], op)
    recs += ps.to_records([_survey("odpt.PassengerSurvey:JR-East.Shibuya",
                                   ["odpt.Station:JR-East.Yamanote.Shibuya"], False,
                                   {2023: 30, 2024: 31})], ps.OPERATOR_BY_KEY["jr_east"])
    return {"_meta": {}, "data": recs}


def _build(monkeypatch, tmp_path, *, with_derived=True):
    monkeypatch.setattr(tc, "load_saved", lambda root: _docs_from_fixtures())
    monkeypatch.setattr(ps, "load_saved", lambda root: _survey_doc())
    return bbc.build(tmp_path, station="テスト駅", with_derived=with_derived)


def test_build_produces_valid_schema(monkeypatch, tmp_path):
    doc = _build(monkeypatch, tmp_path)
    assert bbc.validate_counts(doc) == []
    assert doc["_meta"]["n_records"] == len(doc["counts"])
    for rec in doc["counts"]:
        for field in bbc.REQUIRED_FIELDS:
            assert field in rec


def test_counter_ids_are_unique_per_bin_year_source(monkeypatch, tmp_path):
    """★ 別々の観測が同じ名前で潰れないこと(東横線/田園都市線・上り/下りの衝突)。"""
    doc = _build(monkeypatch, tmp_path)
    keys = [(c["counter_id"], c["time_bin"], c["year"], c["source"]) for c in doc["counts"]]
    assert len(keys) == len(set(keys))


def test_validate_catches_duplicate_counter_id():
    rec = bbc._rec("a", "loc", "daily", 1, "s", 2024, measure="m", unit="u")
    doc = {"_meta": {"n_records": 2}, "counts": [rec, dict(rec)]}
    assert any("重複" in p for p in bbc.validate_counts(doc))


def test_validate_catches_bad_fields():
    good = bbc._rec("a", "loc", "daily", 1, "s", 2024, measure="m", unit="u")
    bad = dict(good, observed_count=-5)
    assert any("負" in p for p in bbc.validate_counts({"_meta": {}, "counts": [bad]}))
    missing = {k: v for k, v in good.items() if k != "location"}
    assert any("location" in p for p in bbc.validate_counts({"_meta": {}, "counts": [missing]}))
    unknown = dict(good, double_count="なにか")
    assert any("double_count" in p for p in bbc.validate_counts({"_meta": {}, "counts": [unknown]}))
    derived = dict(good, kind="derived", derived_from=[])
    assert any("derived_from" in p for p in bbc.validate_counts({"_meta": {}, "counts": [derived]}))


def test_rec_rejects_unknown_double_count():
    with pytest.raises(ValueError):
        bbc._rec("a", "l", "daily", 1, "s", 2024, measure="m", unit="u", double_count="でたらめ")


def test_measures_carry_double_count_flags(monkeypatch, tmp_path):
    """事業者ごとの数え方の違いが counts に印として残っていること。"""
    doc = _build(monkeypatch, tmp_path)
    by_id = {}
    for c in doc["counts"]:
        by_id.setdefault(c["counter_id"], c)
    jr = by_id["station.shibuya.jr-east.boarding"]
    tokyu = by_id["station.shibuya.tokyu.toyoko.boarding_alighting"]
    assert jr["measure"] == "boarding" and jr["double_count"] == "none"
    assert tokyu["measure"] == "boarding_alighting"
    assert tokyu["double_count"] == "through_passengers"
    # 路線別の乗車は改札内乗換の二重計上を名乗る
    line = [c for c in doc["counts"] if c["measure"] == "line_boarding"]
    assert line and all(c["double_count"] == "intra_station_transfer" for c in line)
    # 輸送定員は需要ではないと名乗る
    cap = [c for c in doc["counts"] if c["measure"] == "transport_capacity"]
    assert cap and all(c["double_count"] == "supply_not_demand" for c in cap)
    # 乗換施設調査は部分集計だと名乗る
    tr = [c for c in doc["counts"] if c["source"] == "transport_census.transfer"]
    assert tr and all(c["double_count"] == "partial_survey" for c in tr)


def test_gate_flow_totals_are_summed_per_operator(monkeypatch, tmp_path):
    doc = _build(monkeypatch, tmp_path)
    by_id = {c["counter_id"]: c for c in doc["counts"]
             if c["source"] == "transport_census.station_od13"}
    assert by_id["station.shibuya.gate.all.gate_entry"]["observed_count"] == 175
    assert by_id["station.shibuya.gate.all.gate_exit"]["observed_count"] == 200
    assert by_id["station.shibuya.gate.架空鉄道.gate_entry"]["observed_count"] == 150
    assert by_id["station.shibuya.gate.別会社.gate_entry"]["observed_count"] == 25
    assert all(c["double_count"] == "none" for c in by_id.values())


def test_derived_hourly_is_level_times_share(monkeypatch, tmp_path):
    doc = _build(monkeypatch, tmp_path)
    derived = [c for c in doc["counts"] if c["kind"] == "derived"]
    assert derived, "按分レコードが出ていない"
    entry = [c for c in derived if c["measure"] == "gate_entry"]
    # 水準 175 × 合計/乗車の構成比 (0.2, 0.3, 0.4, 0.1)
    got = {c["time_bin"]: c["observed_count"] for c in entry}
    assert got == {"hour:le06": 35.0, "hour:07": 52.5, "hour:08": 70.0, "hour:ge00": 17.5}
    assert all(len(c["derived_from"]) == 2 for c in derived)
    assert all(c["source"] == "derived" and c["unit"] == "persons_per_hour" for c in derived)
    # 按分の合計は水準に戻る(構成比が1に和すため)
    assert sum(got.values()) == pytest.approx(175.0)


def test_no_derived_when_disabled(monkeypatch, tmp_path):
    doc = _build(monkeypatch, tmp_path, with_derived=False)
    assert not [c for c in doc["counts"] if c["kind"] == "derived"]


def test_build_survives_missing_sources(tmp_path, monkeypatch):
    """未取得の表があっても落ちず、`sources_missing` に正直に書く。"""
    monkeypatch.setattr(tc, "load_saved", lambda root: {})
    monkeypatch.setattr(ps, "load_saved", lambda root: None)
    doc = bbc.build(tmp_path)
    assert doc["counts"] == []
    assert set(doc["_meta"]["sources_missing"]) == {
        "station_flow", "time_dist", "transfer", "line_capacity", "station_od13"}
    assert any("counts が空" in p for p in bbc.validate_counts(doc))


def test_meta_carries_attribution_and_caveats(monkeypatch, tmp_path):
    doc = _build(monkeypatch, tmp_path)
    meta = doc["_meta"]
    assert meta["attribution"] and "国土交通省" in meta["attribution"]
    assert meta["license"]
    assert any("二重" in n or "数え方" in n for n in meta["notes"])
    assert meta["redistribution"] == "restricted"     # ODPT が混ざるので厳しい方に倒す
    assert "portal_mapping" in meta and set(meta["portal_mapping"]) == {"station", "edge"}


def test_cli_writes_file_and_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tc, "load_saved", lambda root: _docs_from_fixtures())
    monkeypatch.setattr(ps, "load_saved", lambda root: _survey_doc())
    rc = bbc.main(["--root", str(tmp_path), "--station", "テスト駅", "--report"])
    assert rc == 0
    path = bbc.out_path(tmp_path)
    assert path.exists()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert bbc.validate_counts(doc) == []
    assert "counter_id" in doc["counts"][0]


def test_cli_offline_makes_zero_requests(monkeypatch, tmp_path, capsys):
    def _boom(req, timeout):                     # pragma: no cover
        raise AssertionError("--offline なのに HTTP に出た")

    monkeypatch.setattr(common, "_urlopen", _boom)
    common.reset_request_count()
    assert bbc.main(["--root", str(tmp_path), "--offline"]) == 0
    assert common.request_count() == 0
    out = capsys.readouterr().out
    assert "mlit.go.jp" in out and common.MASK in out


# ============================================================ F. 境界
def test_src_does_not_read_these_modules():
    """`src/` は取得器も counts ファイルも読まない(較正専用の切り分けを機械固定)。"""
    needles = ("rw_fetch", "boundary_counts", "build_boundary_counts",
               "transport_census", "odpt_passenger_survey", "data/realworld")
    hits = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in needles:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
    assert hits == [], f"src/ が現実データ側を参照している: {hits}"


def test_not_wired_into_daily_scheduler():
    """センサス/乗降者数は静的ソース = 日次スケジューラに結線しない。"""
    daily = REPO_ROOT / "scripts" / "rw_fetch_daily.py"
    if daily.exists():
        text = daily.read_text(encoding="utf-8", errors="ignore")
        assert "transport_census" not in text
        assert "odpt_passenger_survey" not in text
    assert "transport_census" not in tuple(__import__("rw_fetch.ledger",
                                                     fromlist=["DAILY_SOURCES"]).DAILY_SOURCES)


def test_data_dir_is_gitignored():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/realworld/" in text


def test_attribution_registered_for_both_sources():
    for key in ("transport_census", "odpt_passenger_survey"):
        att = common.ATTRIBUTION[key]
        assert att["attribution"] and att["license"] and att["source_url"]
    assert common.ATTRIBUTION["odpt_passenger_survey"]["redistribution"] == "restricted"


def test_no_api_key_literals_in_source():
    """取得器のソースにキーらしき長い英数字リテラルが無いこと。"""
    import re
    for name in ("transport_census.py", "odpt_passenger_survey.py"):
        text = (REPO_ROOT / "scripts" / "rw_fetch" / name).read_text(encoding="utf-8")
        assert not re.search(r"['\"][0-9a-f]{32,}['\"]", text)
    text = (REPO_ROOT / "scripts" / "build_boundary_counts.py").read_text(encoding="utf-8")
    assert not re.search(r"['\"][0-9a-f]{32,}['\"]", text)
