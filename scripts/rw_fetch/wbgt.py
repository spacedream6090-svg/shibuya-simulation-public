#!/usr/bin/env python3
"""A2: 環境省 暑さ指数(WBGT)電子情報提供サービス・地点 44132。

エンドポイント(`docs/research/rw-data-acquisition.md` §2-2 で 2026-08-07 に実接続検証):
  実況(月次累積) https://www.wbgt.env.go.jp/est15WG/dl/wbgt_{地点}_{YYYYMM}.csv
  予測           https://www.wbgt.env.go.jp/prev15WG/dl/yohou_{地点}.csv

★ **単位が違う**: 実況 = 1時間値・**℃** / 予測 = 3時間値・**0.1℃**(260 = 26.0℃)。
   本モジュールは **正規化しない**。両方とも生値のまま保存し、`_meta.units` に単位を明記する
   (換算は下流の設計判断。ここで割ると「どちらの単位で保存されているか」が分からなくなる)。

★ 実況は**月初からの累積ファイル**なので後追いに強い(月内なら1回取れば全部入る)。
   予測は上書きされるので、取りたい日に取らないと消える。
★ 提供期間は毎年 **4月22日〜10月21日**。期間外の 404 は異常ではない(`in_service_period`)。
"""
from __future__ import annotations

import csv
import io
from datetime import date as _date
from pathlib import Path

from . import common, ledger

SOURCE = "wbgt"
EST_BASE = "https://www.wbgt.env.go.jp/est15WG/dl"
PREV_BASE = "https://www.wbgt.env.go.jp/prev15WG/dl"
POINT = "44132"
POINT_INFO = {
    "id": "44132",
    "name_ja": "東京",
    "master_address": "文京区白山　小石川植物園",
    "master_latlon_note": "地点マスタの緯度経度は 35°41.5', 139°45.0'(= アメダス東京 = 北の丸公園)"
                          "で、所在地テキスト(小石川植物園)と食い違う [要一次確認]",
    "distance_from_shibuya_km_range": [5.8, 7.9],
    "obs_start": "2010-05-01",
}

# 提供期間(毎年 4/22〜10/21)。
SERVICE_START = (4, 22)
SERVICE_END = (10, 21)

UNITS = {
    "est": "degC (実況・1時間値。CSV の値をそのまま保持)",
    "forecast": "0.1degC (予測・3時間値。260 = 26.0℃。**換算していない**)",
}

CAVEATS = [
    "実況(℃)と予測(0.1℃)で単位が異なる。本ファイルは換算しておらず、_meta.units のとおり生値。",
    "渋谷区内に提供地点は無い。最寄はアメダスと同じ 44132「東京」で渋谷から約6〜8km。",
    "実況 CSV の未到来時刻セルは空欄。空欄は null として保持し、n_missing に数える(埋めない)。",
    "実況の時刻表記には 24:00 がある(その日の終端)。元表記のまま保持し、日跨ぎ正規化はしない。",
    "提供期間は毎年4月22日〜10月21日。期間外は 404 が正常。",
]


# ------------------------------------------------------------------ URL / パス
def est_url(point: str = POINT, year: int = 2026, month: int = 8) -> str:
    return f"{EST_BASE}/wbgt_{point}_{year:04d}{month:02d}.csv"


def forecast_url(point: str = POINT) -> str:
    return f"{PREV_BASE}/yohou_{point}.csv"


def est_path(root: Path, point: str, year: int, month: int) -> Path:
    return common.month_dir(root, SOURCE, year, month) / f"wbgt_est_{point}_{year:04d}{month:02d}.json"


def forecast_path(root: Path, point: str, d: _date) -> Path:
    return common.day_dir(root, SOURCE, d) / f"wbgt_forecast_{point}_{d.strftime('%Y%m%d')}.json"


def in_service_period(d: _date) -> bool:
    return SERVICE_START <= (d.month, d.day) <= SERVICE_END


def plan_day(d: _date, point: str = POINT) -> list[tuple[str, str]]:
    return [
        (f"WBGT 実況(月次累積・℃) {point} {d.year}-{d.month:02d}", est_url(point, d.year, d.month)),
        (f"WBGT 予測(3時間値・0.1℃) {point}", forecast_url(point)),
    ]


# ------------------------------------------------------------------ パース
def _rows(text: str) -> list[list[str]]:
    return [r for r in csv.reader(io.StringIO(text)) if r]


def parse_est_csv(text: str) -> tuple[list[dict], list[str]]:
    """実況 CSV → (レコード列, 地点ID列)。**空欄は null**(0 で埋めない)。

    形: `Date,Time,44132` / `2026/8/1,1:00,25.0`。都道府県一括版は地点列が複数。
    """
    rows = _rows(text)
    if not rows:
        raise ValueError("実況 CSV が空")
    header = [c.strip() for c in rows[0]]
    if len(header) < 3 or header[0].lower() != "date" or header[1].lower() != "time":
        raise ValueError(f"実況 CSV のヘッダが想定と違う: {header[:4]}")
    points = header[2:]
    out: list[dict] = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        values: dict = {}
        for i, pid in enumerate(points):
            cell = r[i + 2].strip() if len(r) > i + 2 else ""
            if cell == "":
                values[pid] = None
                continue
            try:
                values[pid] = float(cell)
            except ValueError:
                values[pid] = None
        out.append({"date": r[0].strip(), "time": r[1].strip(), "values": values})
    return out, points


def parse_forecast_csv(text: str) -> list[dict]:
    """予測 CSV → 地点ごとのレコード。**0.1℃ のまま**(換算しない)。

    形: 1行目 `,,2026080721,2026080724,...` / 2行目以降 `44132,2026/08/07 20:25, 260, ...`。
    """
    rows = _rows(text)
    if len(rows) < 2:
        raise ValueError("予測 CSV に行が足りない")
    stamps = [c.strip() for c in rows[0][2:]]
    if not stamps or not all(s.isdigit() for s in stamps):
        raise ValueError(f"予測 CSV の時刻ヘッダが想定と違う: {rows[0][:4]}")
    out: list[dict] = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        series = []
        for i, ts in enumerate(stamps):
            cell = r[i + 2].strip() if len(r) > i + 2 else ""
            try:
                val = int(cell) if cell != "" else None
            except ValueError:
                val = None
            series.append({"timestamp": ts, "wbgt_0p1degC": val})
        out.append({"point": r[0].strip(), "generated_at": r[1].strip(), "series": series})
    return out


def count_est_missing(records: list[dict]) -> int:
    return sum(1 for r in records for v in r["values"].values() if v is None)


def count_forecast_missing(records: list[dict]) -> int:
    return sum(1 for r in records for s in r["series"] if s["wbgt_0p1degC"] is None)


# ------------------------------------------------------------------ 取得
def _document(kind: str, records, *, url: str, n_missing: int, extra: dict) -> dict:
    meta = common.build_meta(
        SOURCE, module="wbgt.py", urls=[url], n_records=len(records), n_missing=n_missing,
        units=UNITS, caveats=CAVEATS,
        notes=[f"kind={kind}", "実況℃ と 予測0.1℃ を正規化せず生値のまま保存している。"],
        extra=dict(extra, kind=kind))
    return {"_meta": meta, "data": records}


def fetch_est(root: Path, d: _date, *, point: str = POINT, timeout: float = 30.0,
              retries: int = 2, save_raw: bool = True, sleep_fn=None) -> tuple[dict | None, list[dict]]:
    """実況(月次累積)を取得。月内なら後追い可能なので取り逃しに強い。"""
    url = est_url(point, d.year, d.month)
    res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    if not res.ok:
        note = "" if in_service_period(d) else "(提供期間外 4/22〜10/21 のため 404 は正常)"
        return None, [ledger.make_entry(
            SOURCE, f"est/{point}/{d.year:04d}{d.month:02d}", ok=False, http_status=res.status,
            date_jst=d.isoformat(), error=(res.error or "failed") + note)]
    try:
        records, points = parse_est_csv(res.text())
    except ValueError as exc:
        raw = common.save_raw_on_failure(
            root, SOURCE, f"wbgt_est_{point}_{d.strftime('%Y%m')}.csv", res.body or b"", d)
        return None, [ledger.make_entry(
            SOURCE, f"est/{point}/{d.year:04d}{d.month:02d}", ok=False, http_status=res.status,
            date_jst=d.isoformat(), error=f"parse: {exc}", path=str(raw))]
    n_missing = count_est_missing(records)
    doc = _document("est", records, url=url, n_missing=n_missing,
                    extra={"point": dict(POINT_INFO, id=point), "points_in_file": points,
                           "year": d.year, "month": d.month,
                           "in_service_period": in_service_period(d)})
    path = est_path(root, point, d.year, d.month)
    common.write_json(path, doc)
    if save_raw:
        common.write_bytes(path.with_suffix(".csv"), res.body or b"")
    return doc, [ledger.make_entry(
        SOURCE, f"est/{point}/{d.year:04d}{d.month:02d}", ok=True, http_status=200,
        n_records=len(records), n_missing=n_missing, path=str(path), date_jst=d.isoformat(),
        extra={"complete": True, "kind": "est"})]


def fetch_forecast(root: Path, d: _date, *, point: str = POINT, timeout: float = 30.0,
                   retries: int = 2, save_raw: bool = True,
                   sleep_fn=None) -> tuple[dict | None, list[dict]]:
    """予測(3時間値・0.1℃)を取得。**上書き配信なので後追い不可。**"""
    url = forecast_url(point)
    res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    if not res.ok:
        note = "" if in_service_period(d) else "(提供期間外 4/22〜10/21 のため 404 は正常)"
        return None, [ledger.make_entry(
            SOURCE, f"forecast/{point}", ok=False, http_status=res.status,
            date_jst=d.isoformat(), error=(res.error or "failed") + note,
            extra={"kind": "forecast"})]
    try:
        records = parse_forecast_csv(res.text())
    except ValueError as exc:
        raw = common.save_raw_on_failure(
            root, SOURCE, f"yohou_{point}_{d.strftime('%Y%m%d')}.csv", res.body or b"", d)
        return None, [ledger.make_entry(
            SOURCE, f"forecast/{point}", ok=False, http_status=res.status,
            date_jst=d.isoformat(), error=f"parse: {exc}", path=str(raw),
            extra={"kind": "forecast"})]
    n_missing = count_forecast_missing(records)
    doc = _document("forecast", records, url=url, n_missing=n_missing,
                    extra={"point": dict(POINT_INFO, id=point), "fetch_date_jst": d.isoformat(),
                           "in_service_period": in_service_period(d)})
    path = forecast_path(root, point, d)
    common.write_json(path, doc)
    if save_raw:
        common.write_bytes(path.with_suffix(".csv"), res.body or b"")
    return doc, [ledger.make_entry(
        SOURCE, f"forecast/{point}", ok=True, http_status=200, n_records=len(records),
        n_missing=n_missing, path=str(path), date_jst=d.isoformat(),
        extra={"complete": True, "kind": "forecast"})]


def fetch_day(root: Path, d: _date, *, point: str = POINT, timeout: float = 30.0,
              retries: int = 2, sleep: float = 1.0, save_raw: bool = True,
              sleep_fn=None) -> tuple[list[dict], list[dict]]:
    """実況 + 予測 をまとめて取得。戻り値 = (文書リスト, 台帳行)。"""
    docs, rows = [], []
    doc, r = fetch_est(root, d, point=point, timeout=timeout, retries=retries,
                       save_raw=save_raw, sleep_fn=sleep_fn)
    rows += r
    if doc:
        docs.append(doc)
    common.polite_sleep(sleep, sleep_fn)
    doc, r = fetch_forecast(root, d, point=point, timeout=timeout, retries=retries,
                            save_raw=save_raw, sleep_fn=sleep_fn)
    rows += r
    if doc:
        docs.append(doc)
    return docs, rows
