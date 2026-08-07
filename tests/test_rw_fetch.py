"""RW-U1 現実データフェッチャーのテスト(W2-1 / 2026-08-07)。

対象は **scripts/rw_fetch/ だけ**。`src/` `conf/` には1バイトも触れていない
(= 8/12 フリーズ対象外の整理を機械検査でも固定する。§H)。

★ **ネットワークには一切出ない。** HTTP の唯一の出口 `common._urlopen` を差し替え、
   フィクスチャは実データの「形だけ」を写した最小の合成データを使う。
   `--offline` が本当に 0 リクエストであることも機械固定する(§G)。

構成:
  A. common     秘密のスクラブ / HTTP リトライ / `_meta` / 保存 / 日付
  B. amedas     10分値パース・欠測計上・日次まとめ・保持10日・8ブロック取得
  C. wbgt       実況℃ と 予測0.1℃ の**単位を混ぜない**・空欄は null・提供期間
  D. jinryu     BOM 付き CSV・既取得スキップ(0リクエスト)・DCAT 照合
  E. odpt_rt    キーが URL には載るが**ログ・保存物には一切載らない**・dct:valid・README
  F. jma_xml    Atom パース・東京抽出・外れ値日ラベル
  G. ledger/cli 取得台帳・未取得日検出(URGENT/LOST)・--offline の 0 リクエスト固定
  H. 境界       src/ が rw_fetch / data/realworld を参照しないこと・.gitignore
"""
from __future__ import annotations

import json
import sys
import urllib.error
from datetime import date, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rw_fetch import amedas, cli, common, jma_xml, ledger, odpt_rt, shibuya_jinryu, wbgt  # noqa: E402

JST = common.JST
DAY = date(2026, 8, 7)


# ============================================================ 偽ネットワーク
class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body, self.status = body, status

    def read(self):
        return self._body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeNet:
    """URL 部分一致 → (status, body) の表。未登録は 404。呼ばれた URL を全部記録する。"""

    def __init__(self, table=None, default_status: int = 404):
        self.table = list((table or {}).items())
        self.calls: list[str] = []
        self.default_status = default_status

    def add(self, pattern: str, body, status: int = 200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.table.append((pattern, (status, body)))
        return self

    def __call__(self, req, timeout):
        url = req.full_url
        self.calls.append(url)
        for pattern, (status, body) in self.table:
            if pattern in url:
                if status == 200:
                    return _FakeResponse(body, 200)
                raise urllib.error.HTTPError(url, status, "fake", None, None)
        raise urllib.error.HTTPError(url, self.default_status, "fake-missing", None, None)


class NoNet:
    """呼ばれたら記録して落ちる(= 1本でも出たらテストが失敗する)。"""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, req, timeout):
        self.calls.append(req.full_url)
        raise AssertionError("ネットワークに出てはいけない場面で HTTP 要求が発生した")


@pytest.fixture(autouse=True)
def _clean_state():
    common.clear_secrets()
    common.reset_request_count()
    yield
    common.clear_secrets()
    common.reset_request_count()


@pytest.fixture
def net(monkeypatch):
    f = FakeNet()
    monkeypatch.setattr(common, "_urlopen", f)
    return f


@pytest.fixture
def nonet(monkeypatch):
    f = NoNet()
    monkeypatch.setattr(common, "_urlopen", f)
    return f


# ============================================================ フィクスチャ(実データの形だけ)
AMEDAS_BLOCK = {
    "20260807180000": {"pressure": [1005.0, 0], "normalPressure": [1008.0, 0],
                       "temp": [30.1, 0], "humidity": [70, 0], "sun10m": [10, 0],
                       "precipitation10m": [0.0, 0], "wind": [3.3, 0],
                       "windDirection": [7, 0], "maxTempTime": {"hour": 13, "minute": 20}},
    "20260807181000": {"pressure": [1005.1, 0], "temp": [29.8, 0], "humidity": [71, 0],
                       "sun10m": [10, 0], "precipitation10m": [0.5, 0], "wind": [3.0, 0],
                       "windDirection": [7, 0]},
    "20260807182000": {"pressure": [1005.0, 0], "temp": [None, 1], "humidity": [None, 1],
                       "sun10m": [9, 0], "precipitation10m": [0.0, 0], "wind": [2.8, 0],
                       "windDirection": [7, 0]},
}

WBGT_EST_CSV = "Date,Time,44132\n2026/8/1,1:00,25.0\n2026/8/1,2:00,24.5\n2026/8/1,3:00,\n"
WBGT_FORECAST_CSV = (",,2026080721,2026080724,2026080803\n"
                     "44132,2026/08/07 20:25, 260, 250,\n")

JINRYU_CSV = ("﻿名称,種別,年,月,属性,人数 の合計,ObjectId\n"
              "宮益坂,通行人口,2018年,1月,居住者,5000,1\n"
              "道玄坂,通行人口,2018年,1月,来街者,1300000,2\n"
              "渋谷駅中心エリア,滞在人口,2018年,1月,来街者,8033000,3\n")

ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <title>高頻度フィード</title>
 <entry>
  <title>気象警報・注意報</title>
  <id>urn:uuid:aaa</id>
  <updated>2026-08-16T05:00:00Z</updated>
  <author><name>気象庁</name></author>
  <link type="application/xml" href="https://example.invalid/aaa.xml"/>
  <content type="text">東京都気象警報・注意報</content>
 </entry>
 <entry>
  <title>府県天気予報</title>
  <id>urn:uuid:bbb</id>
  <updated>2026-08-16T06:00:00Z</updated>
  <author><name>大阪管区気象台</name></author>
  <link type="application/xml" href="https://example.invalid/bbb.xml"/>
  <content type="text">大阪府天気予報</content>
 </entry>
</feed>
"""

ODPT_TRAININFO = [
    {"@id": "urn:ucode:1", "owl:sameAs": "odpt.TrainInformation:TokyoMetro.Ginza",
     "dc:date": "2026-08-07T20:47:18+09:00", "dct:valid": "2026-08-07T20:52:18+09:00",
     "odpt:railway": "odpt.Railway:TokyoMetro.Ginza",
     "odpt:trainInformationText": {"ja": "一部の列車に遅れが出ています。"}},
    {"@id": "urn:ucode:2", "owl:sameAs": "odpt.TrainInformation:TokyoMetro.Marunouchi",
     "dc:date": "2026-08-07T20:47:18+09:00", "dct:valid": "2026-08-07T20:52:18+09:00",
     "odpt:railway": "odpt.Railway:TokyoMetro.Marunouchi",
     "odpt:trainInformationText": {"ja": "平常どおり運転しています。"}},
]
ODPT_TRAINS = [
    {"owl:sameAs": "odpt.Train:JR-East.Yamanote.1968G", "odpt:delay": 300,
     "dc:date": "2026-08-07T20:47:18+09:00", "dct:valid": "2026-08-07T20:52:18+09:00",
     "odpt:railway": "odpt.Railway:JR-East.Yamanote"},
    {"owl:sameAs": "odpt.Train:JR-East.Yamanote.1970G", "odpt:delay": 0,
     "dc:date": "2026-08-07T20:47:18+09:00", "dct:valid": "2026-08-07T20:52:18+09:00",
     "odpt:railway": "odpt.Railway:JR-East.Yamanote"},
]


def _register_amedas(net_, d=DAY, hours=amedas.BLOCK_HOURS):
    for hh in hours:
        net_.add(f"/point/44132/{d.strftime('%Y%m%d')}_{hh}.json", AMEDAS_BLOCK)
    return net_


# ============================================================ A. common
def test_scrub_masks_registered_secret():
    common.register_secret("SUPERSECRETKEY12345")
    out = common.scrub("https://api.example/?acl:consumerKey=SUPERSECRETKEY12345&x=1")
    assert "SUPERSECRETKEY12345" not in out
    assert common.MASK in out


def test_scrub_masks_url_encoded_form():
    common.register_secret("abc/def+ghi")
    out = common.scrub("k=abc%2Fdef%2Bghi")
    assert "abc%2Fdef%2Bghi" not in out and common.MASK in out


def test_register_secret_ignores_too_short_and_empty():
    common.register_secret("")
    common.register_secret(None)
    common.register_secret("ab")            # MIN_SECRET_LEN 未満
    assert common.n_secrets() == 0
    assert common.scrub("ab") == "ab"


def test_log_goes_through_scrub(capsys):
    common.register_secret("KEYKEYKEY123")
    common.log("url=https://x/?k=KEYKEYKEY123")
    out = capsys.readouterr().out
    assert "KEYKEYKEY123" not in out and common.MASK in out


def test_write_json_scrubs_secret(tmp_path):
    common.register_secret("LEAKYLEAKY999")
    p = common.write_json(tmp_path / "a.json", {"url": "https://x/?k=LEAKYLEAKY999"})
    text = p.read_text(encoding="utf-8")
    assert "LEAKYLEAKY999" not in text and common.MASK in text


def test_write_bytes_scrubs_secret(tmp_path):
    common.register_secret("RAWLEAK12345")
    p = common.write_bytes(tmp_path / "a.bin", b"before RAWLEAK12345 after")
    assert b"RAWLEAK12345" not in p.read_bytes()


def test_http_get_success(net):
    net.add("example.invalid/ok", b"hello")
    res = common.http_get("https://example.invalid/ok", retries=0)
    assert res.ok and res.status == 200 and res.body == b"hello"
    assert res.attempts == 1 and common.request_count() == 1


def test_http_get_retries_on_503_with_exponential_backoff(net):
    net.add("example.invalid/flaky", b"", status=503)
    slept: list[float] = []
    res = common.http_get("https://example.invalid/flaky", retries=2,
                          backoff=1.0, sleep_fn=slept.append)
    assert res.status == 503 and res.attempts == 3
    assert slept == [1.0, 2.0]            # 指数バックオフ・小回数
    assert common.request_count() == 3


def test_http_get_does_not_retry_on_404(net):
    """404 は確定した答え。アメダスの保持境界検出を鈍らせないため再試行しない。"""
    slept: list[float] = []
    res = common.http_get("https://example.invalid/gone", retries=3, sleep_fn=slept.append)
    assert res.status == 404 and res.attempts == 1 and slept == []


def test_http_get_error_text_never_contains_url(net, monkeypatch):
    common.register_secret("KEYINURL9999")

    def boom(req, timeout):
        raise OSError("connection failed to " + req.full_url)

    monkeypatch.setattr(common, "_urlopen", boom)
    res = common.http_get("https://example.invalid/x?k=KEYINURL9999", retries=0)
    assert res.error == "OSError"
    assert "KEYINURL9999" not in (res.error or "") and "KEYINURL9999" not in res.url_masked


def test_build_meta_carries_attribution_and_license():
    meta = common.build_meta("amedas", module="amedas.py", urls=["https://x"],
                             n_records=3, n_missing=1)
    assert "気象庁" in meta["attribution"] and meta["license_url"]
    assert meta["n_records"] == 3 and meta["n_missing"] == 1
    assert meta["schema_version"] == common.SCHEMA_VERSION
    assert "状態同化" in meta["usage_note"]


def test_validate_document_detects_missing_meta_and_count_mismatch():
    assert common.validate_document({"data": []}) == ["_meta が無い"]
    meta = common.build_meta("wbgt", module="wbgt.py", urls=[], n_records=5, n_missing=0)
    probs = common.validate_document({"_meta": meta, "data": [1, 2]})
    assert any("n_records" in p for p in probs)


def test_parse_date_and_range():
    assert common.parse_date("20260815") == date(2026, 8, 15)
    assert common.parse_date("2026-08-15") == date(2026, 8, 15)
    rng = common.date_range(date(2026, 8, 1), date(2026, 8, 3))
    assert rng == [date(2026, 8, 1), date(2026, 8, 2), date(2026, 8, 3)]
    assert common.date_range(date(2026, 8, 3), date(2026, 8, 1)) == []


def test_day_dir_is_date_partitioned(tmp_path):
    p = common.day_dir(tmp_path, "amedas", date(2026, 8, 15))
    assert p.parts[-3:] == ("2026", "08", "2026-08-15")


# ============================================================ B. amedas
def test_parse_point_block_keeps_values_and_records_flags():
    recs = amedas.parse_point_block(AMEDAS_BLOCK)
    assert [r["timestamp"] for r in recs] == sorted(AMEDAS_BLOCK)
    assert recs[0]["obs_time_jst"] == "2026-08-07T18:00:00+09:00"
    assert recs[0]["values"]["temp"] == 30.1          # 値は書き換えない
    assert "flags" not in recs[0]
    assert recs[2]["flags"]["temp"] == 1              # 品質フラグ 0 以外は記録
    assert recs[2]["values"]["temp"] is None
    # [値,フラグ] 以外の形(時刻オブジェクト)はそのまま残す
    assert recs[0]["values"]["maxTempTime"] == {"hour": 13, "minute": 20}


def test_parse_point_block_rejects_bad_shape():
    with pytest.raises(ValueError):
        amedas.parse_point_block({"not-a-timestamp": {"temp": [1, 0]}})
    with pytest.raises(ValueError):
        amedas.parse_point_block([1, 2, 3])


def test_count_missing_counts_bad_cells_without_double_counting():
    recs = amedas.parse_point_block(AMEDAS_BLOCK)
    miss = amedas.count_missing(recs)
    # 3時刻 × 7要素 = 21 セル。18:20 の temp/humidity が null かつ flagged = 2 セル。
    # 18:10 と 18:20 は normalPressure を持たないが CORE_FIELDS 外なので無関係。
    assert miss["n_cells"] == 21
    assert miss["n_missing"] == 2
    assert miss["n_null"] == 2 and miss["n_flagged"] == 2
    assert miss["per_field"]["temp"] == 1


def test_summarize_day_never_invents_values():
    recs = amedas.parse_point_block(AMEDAS_BLOCK)
    s = amedas.summarize_day(recs)
    assert s["temp_max"] == 30.1 and s["temp_min"] == 29.8
    assert s["n_temp_obs"] == 2                 # フラグ付きは統計から外す
    assert s["precipitation_sum_mm"] == 0.5
    empty = amedas.summarize_day([])
    assert empty["temp_max"] is None and empty["temp_mean"] is None   # 0 で埋めない


def test_within_retention_boundary_is_ten_days():
    today = date(2026, 8, 15)
    assert amedas.within_retention(date(2026, 8, 5), today)      # 10日前 = 実測 200
    assert not amedas.within_retention(date(2026, 8, 4), today)  # 11日前 = 圏外
    assert not amedas.within_retention(date(2026, 8, 16), today)  # 未来


def test_plan_day_lists_eight_blocks_and_no_network(nonet):
    plan = amedas.plan_day(DAY)
    assert len(plan) == amedas.N_BLOCKS == 8
    assert all("20260807_" in url for _, url in plan)
    assert nonet.calls == []


def test_fetch_day_writes_document_and_ledger_row(tmp_path, net):
    _register_amedas(net)
    doc, rows = amedas.fetch_day(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None, today=DAY)
    assert doc is not None and len(net.calls) == 8
    path = amedas.day_path(tmp_path, DAY)
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["_meta"]["complete"] is True and saved["_meta"]["n_blocks_ok"] == 8
    assert saved["_meta"]["daily_summary"]["temp_max"] == 30.1
    assert len(saved["data"]) == 3          # 8ブロックとも同じ3時刻 → マージで重複が畳まれる
    assert rows[0]["ok"] and rows[0]["source"] == "amedas"
    assert rows[0]["extra"]["complete"] is True


def test_fetch_day_partial_when_some_blocks_404(tmp_path, net):
    _register_amedas(net, hours=("00", "03", "06"))
    doc, rows = amedas.fetch_day(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None, today=DAY)
    assert doc["_meta"]["n_blocks_ok"] == 3 and doc["_meta"]["complete"] is False
    assert rows[0]["ok"] is True and rows[0]["extra"]["complete"] is False


def test_fetch_day_all_404_returns_none_and_failed_row(tmp_path, net):
    doc, rows = amedas.fetch_day(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None, today=DAY)
    assert doc is None
    assert rows[0]["ok"] is False and rows[0]["error"] == "all-blocks-failed"
    assert not amedas.day_path(tmp_path, DAY).exists()      # 空ファイルを作らない


def test_fetch_day_saves_raw_on_parse_failure(tmp_path, net):
    net.add("/point/44132/20260807_00.json", b"{ this is not json")
    doc, rows = amedas.fetch_day(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None, today=DAY)
    raws = list((tmp_path / "amedas").glob("**/_raw_failed/*.json"))
    assert raws and raws[0].read_bytes() == b"{ this is not json"   # 黙って空にしない
    assert doc is None


def test_missing_days_detects_gaps_from_files(tmp_path, net):
    _register_amedas(net, d=date(2026, 8, 5))
    amedas.fetch_day(tmp_path, date(2026, 8, 5), sleep=0, sleep_fn=lambda s: None,
                     today=date(2026, 8, 7))
    miss = amedas.missing_days(tmp_path, date(2026, 8, 7), window_days=3)
    assert date(2026, 8, 5) not in miss
    assert date(2026, 8, 6) in miss and date(2026, 8, 4) in miss


# ============================================================ C. wbgt
def test_parse_est_csv_keeps_celsius_and_nulls_blanks():
    recs, points = wbgt.parse_est_csv(WBGT_EST_CSV)
    assert points == ["44132"] and len(recs) == 3
    assert recs[0] == {"date": "2026/8/1", "time": "1:00", "values": {"44132": 25.0}}
    assert recs[2]["values"]["44132"] is None       # 未到来 = 空欄 → null(0 で埋めない)
    assert wbgt.count_est_missing(recs) == 1


def test_parse_forecast_csv_keeps_tenth_degree_units():
    recs = wbgt.parse_forecast_csv(WBGT_FORECAST_CSV)
    assert len(recs) == 1 and recs[0]["point"] == "44132"
    assert recs[0]["generated_at"] == "2026/08/07 20:25"
    vals = [s["wbgt_0p1degC"] for s in recs[0]["series"]]
    assert vals == [260, 250, None]                # ★ 26.0 に換算しない
    assert wbgt.count_forecast_missing(recs) == 1


def test_est_and_forecast_units_are_declared_and_different():
    assert "degC" in wbgt.UNITS["est"] and "0.1degC" in wbgt.UNITS["forecast"]
    assert wbgt.UNITS["est"] != wbgt.UNITS["forecast"]


def test_parse_rejects_wrong_headers():
    with pytest.raises(ValueError):
        wbgt.parse_est_csv("A,B,C\n1,2,3\n")
    with pytest.raises(ValueError):
        wbgt.parse_forecast_csv(",,notatimestamp\n44132,x, 260\n")


def test_in_service_period_covers_finals_window():
    assert wbgt.in_service_period(date(2026, 8, 15))
    assert wbgt.in_service_period(date(2026, 8, 30))
    assert not wbgt.in_service_period(date(2026, 4, 21))
    assert not wbgt.in_service_period(date(2026, 10, 22))


def test_fetch_day_writes_est_and_forecast_with_units(tmp_path, net):
    net.add("/est15WG/dl/wbgt_44132_202608.csv", WBGT_EST_CSV)
    net.add("/prev15WG/dl/yohou_44132.csv", WBGT_FORECAST_CSV)
    docs, rows = wbgt.fetch_day(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None)
    assert len(docs) == 2 and all(r["ok"] for r in rows)
    est = json.loads(wbgt.est_path(tmp_path, "44132", 2026, 8).read_text(encoding="utf-8"))
    fcs = json.loads(wbgt.forecast_path(tmp_path, "44132", DAY).read_text(encoding="utf-8"))
    assert est["_meta"]["kind"] == "est" and fcs["_meta"]["kind"] == "forecast"
    assert est["_meta"]["units"] == fcs["_meta"]["units"] == wbgt.UNITS
    assert est["_meta"]["n_missing"] == 1
    assert "環境省" in est["_meta"]["attribution"]


def test_fetch_est_saves_raw_csv(tmp_path, net):
    net.add("/est15WG/dl/wbgt_44132_202608.csv", WBGT_EST_CSV)
    wbgt.fetch_est(tmp_path, DAY, sleep_fn=lambda s: None)
    raw = wbgt.est_path(tmp_path, "44132", 2026, 8).with_suffix(".csv")
    assert raw.exists() and raw.read_text(encoding="utf-8") == WBGT_EST_CSV


def test_fetch_est_404_records_failure_not_fake_data(tmp_path, net):
    doc, rows = wbgt.fetch_est(tmp_path, DAY, sleep_fn=lambda s: None)
    assert doc is None and rows[0]["ok"] is False and rows[0]["http_status"] == 404
    assert not wbgt.est_path(tmp_path, "44132", 2026, 8).exists()


def test_fetch_est_out_of_season_404_is_annotated(tmp_path, net):
    _, rows = wbgt.fetch_est(tmp_path, date(2026, 1, 10), sleep_fn=lambda s: None)
    assert "提供期間外" in rows[0]["error"]


# ============================================================ D. shibuya_jinryu
def test_parse_csv_strips_bom_and_types_the_count_column():
    recs, header, n_missing = shibuya_jinryu.parse_csv(JINRYU_CSV)
    assert header[0] == "名称"                       # BOM が剥がれている
    assert len(recs) == 3 and n_missing == 0
    assert recs[0]["名称"] == "宮益坂" and recs[0]["人数 の合計"] == 5000
    assert recs[2]["人数 の合計"] == 8033000


def test_parse_csv_blank_count_is_null_not_zero():
    recs, _, n_missing = shibuya_jinryu.parse_csv(
        "名称,種別,年,月,属性,人数 の合計,ObjectId\n宮益坂,通行人口,2018年,1月,居住者,,1\n")
    assert recs[0]["人数 の合計"] is None and n_missing == 1


def test_parse_csv_rejects_unexpected_header():
    with pytest.raises(ValueError):
        shibuya_jinryu.parse_csv("a,b,c\n1,2,3\n")


def test_summarize_lists_locations_and_years():
    recs, header, _ = shibuya_jinryu.parse_csv(JINRYU_CSV)
    s = shibuya_jinryu.summarize(recs, header)
    assert s["n_locations"] == 3 and "道玄坂" in s["locations"]
    assert s["years"] == ["2018年"] and set(s["kinds"]) == {"通行人口", "滞在人口"}


def test_six_datasets_have_distinct_item_ids():
    ids = [d["item_id"] for d in shibuya_jinryu.DATASETS]
    assert len(ids) == 6 and len(set(ids)) == 6


def test_verify_against_dcat_detects_missing_item():
    feed = {"dataset": [{"identifier": d["item_id"]} for d in shibuya_jinryu.DATASETS[:5]]}
    probs = shibuya_jinryu.verify_against_dcat(feed)
    assert len(probs) == 1 and shibuya_jinryu.DATASETS[5]["key"] in probs[0]


def test_fetch_all_skips_already_fetched_without_any_request(tmp_path, net):
    net.add("/api/download/v1/items/", JINRYU_CSV)
    docs, rows = shibuya_jinryu.fetch_all(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None)
    assert len(docs) == 6 and len(net.calls) == 6
    net.calls.clear()
    docs2, rows2 = shibuya_jinryu.fetch_all(tmp_path, DAY, sleep=0, sleep_fn=lambda s: None)
    assert docs2 == [] and rows2 == [] and net.calls == []     # 月次 = 取り直さない


def test_fetch_one_records_usage_scope_l1_only(tmp_path, net):
    net.add("/api/download/v1/items/", JINRYU_CSV)
    doc, _ = shibuya_jinryu.fetch_one(tmp_path, shibuya_jinryu.DATASETS[0], DAY,
                                      sleep_fn=lambda s: None)
    assert "L1" in doc["_meta"]["usage_scope"] and "CC BY 4.0" in doc["_meta"]["license"]
    assert "KDDI" in doc["_meta"]["attribution"]


# ============================================================ E. odpt_rt
def test_display_url_never_contains_a_key():
    url = odpt_rt.display_url("challenge", "odpt:Train",
                              {"odpt:railway": "odpt.Railway:JR-East.Yamanote"})
    assert url.endswith("acl:consumerKey=" + common.MASK)


def test_plan_is_key_free_and_makes_no_request(nonet):
    plan = odpt_rt.plan()
    assert plan and all(common.MASK in url for _, url in plan)
    assert any("odpt:TrainInformation" in u for _, u in plan)
    assert any("odpt:Train?" in u for _, u in plan)
    assert nonet.calls == []


def test_split_expired_drops_only_stale_records():
    now = datetime(2026, 8, 7, 20, 55, tzinfo=JST)
    valid, expired = odpt_rt.split_expired(ODPT_TRAINS, now)
    assert valid == [] and len(expired) == 2
    now2 = datetime(2026, 8, 7, 20, 50, tzinfo=JST)
    valid2, expired2 = odpt_rt.split_expired(ODPT_TRAINS, now2)
    assert len(valid2) == 2 and expired2 == []


def test_shibuya_railways_filters_out_other_lines():
    kept = odpt_rt.shibuya_railways(ODPT_TRAININFO)
    assert len(kept) == 1
    assert kept[0]["odpt:railway"] == "odpt.Railway:TokyoMetro.Ginza"


def test_delay_summary_is_none_when_no_data():
    assert odpt_rt.delay_summary([]) == {"n": 0, "max_s": None, "mean_s": None, "n_delayed": 0}
    s = odpt_rt.delay_summary(ODPT_TRAINS)
    assert s["max_s"] == 300 and s["n_delayed"] == 1 and s["mean_s"] == 150.0


def test_env_var_names_match_existing_scripts():
    """既存 scripts/fetch_odpt.py と同じ環境変数名を踏襲していること。"""
    text = (REPO_ROOT / "scripts" / "fetch_odpt.py").read_text(encoding="utf-8")
    assert odpt_rt.OPEN_KEY_ENV in text and odpt_rt.CHALLENGE_KEY_ENV in text


def test_fetch_snapshot_key_reaches_the_wire_but_never_a_file_or_log(tmp_path, net, capsys):
    """★キーは実リクエスト URL には載るが、ログ・保存物には1バイトも出ない。"""
    key = "FAKE-OPEN-KEY-0123456789"
    common.register_secret(key)
    net.add("api.odpt.org/api/v4/odpt:TrainInformation", ODPT_TRAININFO)
    now = datetime(2026, 8, 7, 20, 50, tzinfo=JST)
    docs, rows = odpt_rt.fetch_snapshot(tmp_path, now=now, open_key=key, challenge_key="",
                                        sleep=0, sleep_fn=lambda s: None)
    assert any(key in c for c in net.calls)              # 送信はされている
    out = capsys.readouterr()
    assert key not in out.out and key not in out.err
    for path in (tmp_path / "odpt_rt").glob("**/*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes(), path
    for row in rows:
        assert key not in json.dumps(row, ensure_ascii=False)
    assert docs and docs[0]["_meta"]["source_urls"]
    assert all(common.MASK in u for u in docs[0]["_meta"]["source_urls"])


def test_fetch_snapshot_without_key_makes_no_request(tmp_path, net):
    docs, rows = odpt_rt.fetch_snapshot(tmp_path, now=datetime(2026, 8, 7, 20, 50, tzinfo=JST),
                                        open_key="", challenge_key="",
                                        sleep=0, sleep_fn=lambda s: None)
    assert net.calls == [] and docs == []
    assert rows and all(r["error"].startswith("key-missing:") for r in rows)


def test_fetch_snapshot_filters_shibuya_lines_and_records_expiry(tmp_path, net):
    net.add("api.odpt.org/api/v4/odpt:TrainInformation", ODPT_TRAININFO)
    net.add("api-challenge.odpt.org/api/v4/odpt:Train?", ODPT_TRAINS)
    now = datetime(2026, 8, 7, 20, 55, tzinfo=JST)      # dct:valid=20:52 を過ぎている
    docs, _ = odpt_rt.fetch_snapshot(tmp_path, now=now, open_key="OPENKEY123456",
                                     challenge_key="CHALKEY123456", sleep=0,
                                     sleep_fn=lambda s: None)
    info = [d for d in docs if d["_meta"]["datatype"] == "odpt:TrainInformation"][0]
    assert info["data"] == [] and info["_meta"]["n_expired"] == 1   # 丸ノ内線は元から除外
    train = [d for d in docs if d["_meta"]["datatype"] == "odpt:Train"][0]
    # 在線は JR 3路線に問い合わせる(山手/埼京/湘南新宿) → 2本 × 3 = 6 が期限切れ
    assert train["_meta"]["n_expired"] == 6 and train["data"] == []


def test_readme_declares_redistribution_restriction(tmp_path):
    path = odpt_rt.ensure_readme(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "再配布不可" in text and "gitignore" in text
    assert odpt_rt.OPEN_KEY_ENV in text and odpt_rt.CHALLENGE_KEY_ENV in text
    assert common.ATTRIBUTION["odpt_rt"]["redistribution"] == "restricted"


# ============================================================ F. jma_xml
def test_parse_atom_extracts_entries_and_flags():
    entries = jma_xml.parse_atom(ATOM_XML)
    assert len(entries) == 2
    assert entries[0]["title"] == "気象警報・注意報"
    assert entries[0]["link"] == "https://example.invalid/aaa.xml"
    assert entries[0]["author"] == "気象庁"
    assert entries[0]["tokyo_related"] is True and entries[0]["outlier_candidate"] is True
    assert entries[1]["tokyo_related"] is False and entries[1]["outlier_candidate"] is False


def test_parse_atom_rejects_broken_xml():
    with pytest.raises(ValueError):
        jma_xml.parse_atom("<feed><entry>")


def test_outlier_labels_group_by_jst_day():
    labels = jma_xml.outlier_labels(jma_xml.parse_atom(ATOM_XML))
    assert list(labels) == ["2026-08-16"]           # 05:00Z = 14:00 JST 同日
    assert labels["2026-08-16"] == ["気象警報・注意報"]


def test_only_long_feeds_are_used():
    """R4(1日10GB超で IP 遮断)対策: 高頻度フィードは取らない。"""
    assert all(f.endswith("_l") for f in jma_xml.FEEDS)
    assert len(jma_xml.FEEDS) == 4


def test_fetch_all_writes_documents_and_labels(tmp_path, net):
    for feed in jma_xml.FEEDS:
        net.add(f"/feed/{feed}.xml", ATOM_XML)
    now = datetime(2026, 8, 7, 9, 0, tzinfo=JST)
    docs, rows = jma_xml.fetch_all(tmp_path, now=now, sleep=0, sleep_fn=lambda s: None)
    assert len(docs) == 4 and all(r["ok"] for r in rows)
    assert docs[0]["_meta"]["n_tokyo_related"] == 1
    assert docs[0]["_meta"]["outlier_labels_by_day"] == {"2026-08-16": ["気象警報・注意報"]}
    assert "気象庁" in docs[0]["_meta"]["attribution"]


def test_fetch_feed_saves_raw_on_parse_failure(tmp_path, net):
    net.add("/feed/extra_l.xml", b"<feed><entry>")
    doc, rows = jma_xml.fetch_feed(tmp_path, "extra_l", now=datetime(2026, 8, 7, 9, tzinfo=JST),
                                   retries=0, sleep_fn=lambda s: None)
    assert doc is None and rows[0]["ok"] is False
    assert list((tmp_path / "jma_xml").glob("**/_raw_failed/*.xml"))


# ============================================================ G. ledger / cli
def test_ledger_append_and_read_roundtrip(tmp_path):
    rows = [ledger.make_entry("amedas", "44132/2026-08-07", ok=True, http_status=200,
                              n_records=18, date_jst="2026-08-07"),
            ledger.make_entry("wbgt", "est/44132", ok=False, http_status=404,
                              date_jst="2026-08-07", error="HTTP 404")]
    assert ledger.append(tmp_path, rows) == 2
    ledger.append(tmp_path, rows[0])
    back = ledger.read_all(tmp_path)
    assert len(back) == 3 and back[0]["source"] == "amedas" and back[1]["ok"] is False


def test_ledger_append_scrubs_secrets(tmp_path):
    common.register_secret("LEDGERLEAK123")
    ledger.append(tmp_path, {"source": "x", "note": "k=LEDGERLEAK123", "ok": True})
    assert "LEDGERLEAK123" not in ledger.ledger_path(tmp_path).read_text(encoding="utf-8")


def test_ledger_keeps_broken_lines_visible(tmp_path):
    p = ledger.ledger_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"source":"a","ok":true}\nnot json at all\n', encoding="utf-8")
    rows = ledger.read_all(tmp_path)
    assert len(rows) == 2 and rows[1]["_broken"] == "not json at all"


def test_day_status_distinguishes_partial_from_ok(tmp_path):
    rows = [
        ledger.make_entry("amedas", "a", ok=True, date_jst="2026-08-01",
                          extra={"complete": True}),
        ledger.make_entry("amedas", "b", ok=True, date_jst="2026-08-02",
                          extra={"complete": False}),
        ledger.make_entry("amedas", "c", ok=False, date_jst="2026-08-03"),
    ]
    assert ledger.day_status(rows, "amedas", "2026-08-01") == "ok"
    assert ledger.day_status(rows, "amedas", "2026-08-02") == "partial"
    assert ledger.day_status(rows, "amedas", "2026-08-03") == "failed"
    assert ledger.day_status(rows, "amedas", "2026-08-04") == "missing"


def test_missing_days_lists_every_non_ok_day():
    rows = [ledger.make_entry("amedas", "a", ok=True, date_jst="2026-08-02",
                              extra={"complete": True})]
    miss = ledger.missing_days(rows, "amedas", date(2026, 8, 1), date(2026, 8, 3))
    assert miss == ["2026-08-01", "2026-08-03"]


def test_gap_report_splits_urgent_from_lost_at_the_ten_day_boundary():
    """★生命線: 保持10日以内 = まだ取れる(URGENT) / 超過 = 二度と取れない(LOST)。"""
    today = date(2026, 8, 20)
    rows = [ledger.make_entry("amedas", "x", ok=True, date_jst=d.isoformat(),
                              extra={"complete": True})
            for d in common.date_range(date(2026, 7, 21), date(2026, 8, 20))
            if d not in (date(2026, 8, 12), date(2026, 8, 5))]
    rep = ledger.gap_report(rows, "amedas", today, lookback_days=20)
    assert "2026-08-12" in rep["urgent"]        # 8日前 = 保持10日以内
    assert "2026-08-05" in rep["lost"]          # 15日前 = 回復不能
    assert rep["retention_days"] == 10


def test_gap_report_treats_today_as_pending():
    today = date(2026, 8, 20)
    rep = ledger.gap_report([], "amedas", today, lookback_days=2)
    assert rep["pending"] == ["2026-08-20"]
    assert "2026-08-20" not in rep["urgent"]


def test_format_report_marks_urgent_and_lost():
    today = date(2026, 8, 20)
    rows = [ledger.make_entry("amedas", "x", ok=True, date_jst="2026-08-19",
                              extra={"complete": True})]
    text = ledger.format_report(rows, today, sources=["amedas"], lookback_days=12)
    assert "URGENT" in text and "LOST" in text and "amedas" in text
    assert ledger.format_report([], today) .endswith("台帳が空。まだ一度も取得していない。")


def test_retention_table_matches_research_document():
    assert ledger.RETENTION_DAYS["amedas"] == 10        # 実測
    assert ledger.RETENTION_DAYS["odpt_rt"] == 0        # dct:valid 5分
    assert ledger.RETENTION_DAYS["wbgt"] is None        # 月次累積 = 後追い可


def test_cli_offline_makes_zero_requests(tmp_path, nonet, capsys):
    rc = cli.main(["--offline", "--source", "all", "--out-dir", str(tmp_path),
                   "--date", "2026-08-07"])
    out = capsys.readouterr().out
    assert rc == 0
    assert nonet.calls == [] and common.request_count() == 0
    assert "1 本も送っていません" in out
    assert "20260807_00.json" in out and "yohou_44132.csv" in out


def test_cli_offline_never_prints_a_key(tmp_path, nonet, capsys, monkeypatch):
    monkeypatch.setenv(odpt_rt.OPEN_KEY_ENV, "ENVKEYSHOULDNOTAPPEAR")
    common.register_secret("ENVKEYSHOULDNOTAPPEAR")
    cli.main(["--offline", "--source", "odpt_rt", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "ENVKEYSHOULDNOTAPPEAR" not in out and "設定済み" in out


def test_cli_source_selection():
    assert cli._sources_from_args(None) == list(cli.DAILY_SOURCES)
    assert cli._sources_from_args(["all"]) == list(cli.ALL_SOURCES)
    assert cli._sources_from_args(["amedas,wbgt"]) == ["amedas", "wbgt"]
    with pytest.raises(SystemExit):
        cli._sources_from_args(["nope"])


def test_cli_report_and_verify_make_zero_requests(tmp_path, nonet, capsys):
    ledger.append(tmp_path, ledger.make_entry("amedas", "x", ok=True, date_jst="2026-08-06",
                                              extra={"complete": True}))
    assert cli.main(["--report", "--out-dir", str(tmp_path), "--date", "2026-08-07"]) == 0
    assert cli.main(["--verify", "--out-dir", str(tmp_path)]) == 0
    assert nonet.calls == [] and common.request_count() == 0
    assert "取得台帳レポート" in capsys.readouterr().out


def test_cli_verify_flags_a_document_without_meta(tmp_path, nonet, capsys):
    (tmp_path / "amedas").mkdir(parents=True)
    (tmp_path / "amedas" / "bad.json").write_text('{"data": []}', encoding="utf-8")
    assert cli.main(["--verify", "--out-dir", str(tmp_path)]) == 1
    assert "_meta が無い" in capsys.readouterr().err


def test_cli_end_to_end_daily_run(tmp_path, net, capsys):
    _register_amedas(net)
    net.add("/est15WG/dl/wbgt_44132_202608.csv", WBGT_EST_CSV)
    net.add("/prev15WG/dl/yohou_44132.csv", WBGT_FORECAST_CSV)
    for feed in jma_xml.FEEDS:
        net.add(f"/feed/{feed}.xml", ATOM_XML)
    rc = cli.main(["--out-dir", str(tmp_path), "--date", "2026-08-07",
                   "--sleep", "0", "--retries", "0"])
    assert rc == 0
    rows = ledger.read_all(tmp_path)
    assert {r["source"] for r in rows} == {"amedas", "wbgt", "jma_xml"}
    assert all(r["ok"] for r in rows)
    assert (tmp_path / "README.md").exists()
    assert "再配布" in (tmp_path / "README.md").read_text(encoding="utf-8")
    # 保存物はすべてスキーマ検証を通る
    assert cli.main(["--verify", "--out-dir", str(tmp_path)]) == 0


def test_cli_continues_after_a_source_fails(tmp_path, net):
    """部分失敗で止まらない・失敗も台帳に残る。"""
    for feed in jma_xml.FEEDS:
        net.add(f"/feed/{feed}.xml", ATOM_XML)      # アメダスと WBGT は 404 のまま
    rc = cli.main(["--out-dir", str(tmp_path), "--date", "2026-08-07",
                   "--sleep", "0", "--retries", "0"])
    rows = ledger.read_all(tmp_path)
    assert rc == 1                                   # 失敗ソースがあるので非ゼロ
    assert any(r["source"] == "jma_xml" and r["ok"] for r in rows)
    assert any(r["source"] == "amedas" and not r["ok"] for r in rows)


def test_cli_backfill_fetches_missing_days_within_retention(tmp_path, net):
    for d in (date(2026, 8, 6), date(2026, 8, 7)):
        _register_amedas(net, d=d)
    rc = cli.main(["--out-dir", str(tmp_path), "--date", "2026-08-07", "--source", "amedas",
                   "--backfill", "--sleep", "0", "--retries", "0"])
    assert rc == 0
    assert amedas.day_path(tmp_path, date(2026, 8, 7)).exists()
    assert amedas.day_path(tmp_path, date(2026, 8, 6)).exists()


# ============================================================ H. 境界(フリーズ整理の機械検査)
def test_src_never_references_rw_fetch_or_realworld():
    """8/12 フリーズ対象外の条件: `src/` から本件への参照がゼロであること。"""
    hits = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "rw_fetch" in text or "data/realworld" in text or "realworld" in text:
            hits.append(str(path))
    assert hits == []


def test_rw_fetch_never_imports_the_simulation():
    """逆向き: 取得器がシミュ本体(src/)や conf/ に依存しないこと。"""
    bad = []
    for path in (REPO_ROOT / "scripts" / "rw_fetch").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")) and ("society" in s or "conf" in s
                                                       or s.startswith("from src")):
                bad.append(f"{path.name}: {s}")
    assert bad == []


def test_gitignore_excludes_realworld_data():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/realworld/" in text


def test_entrypoint_script_exists_and_is_thin():
    path = REPO_ROOT / "scripts" / "rw_fetch_daily.py"
    text = path.read_text(encoding="utf-8")
    assert "rw_fetch.cli" in text and len(text.splitlines()) < 40


def test_all_sources_declare_attribution():
    for src in cli.ALL_SOURCES:
        att = common.ATTRIBUTION[src]
        assert att["attribution"] and att["license"] and att["source_url"]
