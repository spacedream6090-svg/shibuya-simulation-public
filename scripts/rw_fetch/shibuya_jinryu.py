#!/usr/bin/env python3
"""A3: 渋谷区オープンデータ「Location Analyzer による通行/滞在人口」6 データセット。

エンドポイント(`docs/research/rw-data-acquisition.md` §2-3 で 2026-08-07 に実接続検証):
  DCAT カタログ https://city-shibuya-data.opendata.arcgis.com/api/feed/dcat-us/1.1.json
  CSV           https://city-shibuya-data.opendata.arcgis.com/api/download/v1/items/{itemId}/csv?layers=0

★ **月次データ・2018年1月〜2024年12月で完結**しているので、取得は本選期間中に **1回**で足りる
   (毎日叩く意味がない = `needs_fetch` が既取得を検出して既定でスキップする)。
★ 位置づけは **L1 パラメータ較正の主柱**。2024年12月で終わるので **本選期間の事後検証には
   使えない**(リスク R8)。この切り分けは `_meta.usage_scope` に固定して書く。
★ ライセンス CC BY 4.0。帰属表示は `_meta.attribution`(KDDI・技研商事の指定表記を含む)。
"""
from __future__ import annotations

import csv
import io
from datetime import date as _date
from pathlib import Path

from . import common, ledger

SOURCE = "shibuya_jinryu"
HUB = "https://city-shibuya-data.opendata.arcgis.com"
DCAT_URL = f"{HUB}/api/feed/dcat-us/1.1.json"

# §2-3 の実測 itemId(6本)。DCAT フィードで名称の一致を検証できる(`verify_against_dcat`)。
DATASETS: list[dict] = [
    {"key": "tsuko_attribute", "item_id": "e0747cc11bec4076b3fbbb5adaf8a667",
     "title": "Location Analyzerによる通行人口データ（属性別）", "kind": "通行人口", "axis": "属性"},
    {"key": "tsuko_age", "item_id": "bcaecf7e95524341990eb710f4b259c1",
     "title": "Location Analyzerによる通行人口データ（年代別）", "kind": "通行人口", "axis": "年代"},
    {"key": "tsuko_sex", "item_id": "ed98b7cf800042279c10a4d6c33b51c1",
     "title": "Location Analyzerによる通行人口データ（性別）", "kind": "通行人口", "axis": "性別"},
    {"key": "taizai_attribute", "item_id": "ed587c1134334e259c8cbcefaaf29b2a",
     "title": "Location Analyzerによる滞在人口データ（属性別）", "kind": "滞在人口", "axis": "属性"},
    {"key": "taizai_age", "item_id": "8cde42f2013941d08665424331be221e",
     "title": "Location Analyzerによる滞在人口データ（年代別）", "kind": "滞在人口", "axis": "年代"},
    {"key": "taizai_sex", "item_id": "4ea66bd8bec444fab1e2afbeb72e52df",
     "title": "Location Analyzerによる滞在人口データ（性別）", "kind": "滞在人口", "axis": "性別"},
]
DATASET_BY_KEY = {d["key"]: d for d in DATASETS}

USAGE_SCOPE = ("L1 パラメータ較正専用。2018-01〜2024-12 の月次で、時間帯・日種の次元が無く、"
               "本選期間(2026年8月)の事後検証には使えない(docs/research/rw-data-acquisition.md R8)。")

CAVEATS = [
    "月次合計人数(千人単位に丸め)。時間帯別・平日休日別の次元は無い。",
    "2018年1月〜2024年12月で系列が終わる。2026年8月の検証には使えない(L1較正専用)。",
    "auスマートフォンユーザーのうち個別同意を得たユーザーの集計値であり、実人口ではない。",
    "CSV は UTF-8 BOM 付き。BOM は読み込み時に除去しているが値そのものは無加工。",
]


# ------------------------------------------------------------------ URL / パス
def csv_url(item_id: str) -> str:
    return f"{HUB}/api/download/v1/items/{item_id}/csv?layers=0"


def geojson_url(item_id: str) -> str:
    return f"{HUB}/api/download/v1/items/{item_id}/geojson?layers=0"


def out_path(root: Path, key: str, d: _date) -> Path:
    return common.day_dir(root, SOURCE, d) / f"{key}.json"


def already_fetched(root: Path, key: str) -> Path | None:
    """既に取得済みならそのパス(月次なので取り直す必要が無い)。"""
    hits = sorted((Path(root) / SOURCE).glob(f"**/{key}.json"))
    return hits[-1] if hits else None


def plan(datasets=None) -> list[tuple[str, str]]:
    return [(f"渋谷区人流 {d['title']}", csv_url(d["item_id"]))
            for d in (datasets or DATASETS)]


# ------------------------------------------------------------------ パース
def parse_csv(text: str) -> tuple[list[dict], list[str], int]:
    """人流 CSV → (レコード列, ヘッダ, 欠測セル数)。BOM 除去のみ・値は無加工。

    形: `名称,種別,年,月,属性,人数 の合計,ObjectId`(軸列の名前はデータセットで変わる)。
    """
    if text.startswith("﻿"):
        text = text[1:]
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if len(rows) < 2:
        raise ValueError("人流 CSV に行が足りない")
    header = [c.strip() for c in rows[0]]
    if "名称" not in header or not any("人数" in h for h in header):
        raise ValueError(f"人流 CSV のヘッダが想定と違う: {header}")
    value_cols = [h for h in header if "人数" in h]
    out: list[dict] = []
    n_missing = 0
    for r in rows[1:]:
        rec: dict = {}
        for i, h in enumerate(header):
            cell = r[i].strip() if i < len(r) else ""
            if h in value_cols:
                if cell == "":
                    rec[h] = None
                    n_missing += 1
                else:
                    try:
                        rec[h] = float(cell) if "." in cell else int(cell)
                    except ValueError:
                        rec[h] = None
                        n_missing += 1
            else:
                rec[h] = cell
        out.append(rec)
    return out, header, n_missing


def summarize(records: list[dict], header: list[str]) -> dict:
    """空間・時間のカバレッジ要約(較正で最初に見る数字)。"""
    names = sorted({r.get("名称", "") for r in records if r.get("名称")})
    years = sorted({r.get("年", "") for r in records if r.get("年")})
    kinds = sorted({r.get("種別", "") for r in records if r.get("種別")})
    return {"n_rows": len(records), "columns": header, "locations": names,
            "n_locations": len(names), "years": years, "kinds": kinds}


def verify_against_dcat(feed_obj, datasets=None) -> list[str]:
    """DCAT フィードに itemId と表題が実在するかを照合する(スキーマ漂流の検知)。"""
    datasets = datasets or DATASETS
    text = common.canonical_json(feed_obj)
    problems = []
    for d in datasets:
        if d["item_id"] not in text:
            problems.append(f"DCAT に itemId が無い: {d['key']} ({d['item_id']})")
    return problems


# ------------------------------------------------------------------ 取得
def fetch_one(root: Path, ds: dict, d: _date, *, timeout: float = 60.0, retries: int = 2,
              save_raw: bool = True, sleep_fn=None) -> tuple[dict | None, list[dict]]:
    url = csv_url(ds["item_id"])
    res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    if not res.ok:
        return None, [ledger.make_entry(
            SOURCE, ds["key"], ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=res.error or "failed")]
    try:
        records, header, n_missing = parse_csv(res.text())
    except ValueError as exc:
        raw = common.save_raw_on_failure(root, SOURCE, f"{ds['key']}.csv", res.body or b"", d)
        return None, [ledger.make_entry(
            SOURCE, ds["key"], ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=f"parse: {exc}", path=str(raw))]
    meta = common.build_meta(
        SOURCE, module="shibuya_jinryu.py", urls=[url], n_records=len(records),
        n_missing=n_missing, caveats=CAVEATS,
        notes=["月次データにつき期間中1回の取得で足りる(既取得なら --force なしではスキップ)。"],
        extra={"dataset": ds, "usage_scope": USAGE_SCOPE,
               "summary": summarize(records, header), "fetch_date_jst": d.isoformat()})
    doc = {"_meta": meta, "data": records}
    path = out_path(root, ds["key"], d)
    common.write_json(path, doc)
    if save_raw:
        common.write_bytes(path.with_suffix(".csv"), res.body or b"")
    return doc, [ledger.make_entry(
        SOURCE, ds["key"], ok=True, http_status=200, n_records=len(records),
        n_missing=n_missing, path=str(path), date_jst=d.isoformat(),
        extra={"complete": True})]


def fetch_all(root: Path, d: _date, *, datasets=None, timeout: float = 60.0, retries: int = 2,
              sleep: float = 1.0, force: bool = False, save_raw: bool = True,
              sleep_fn=None) -> tuple[list[dict], list[dict]]:
    """6 データセットを取得。既取得のものは `force` でなければ **リクエストを出さずに** 飛ばす。"""
    datasets = list(datasets or DATASETS)
    docs, rows = [], []
    for i, ds in enumerate(datasets):
        if not force:
            got = already_fetched(root, ds["key"])
            if got:
                common.log(f"  [jinryu] skip {ds['key']} (取得済み: {got.name})")
                continue
        doc, r = fetch_one(root, ds, d, timeout=timeout, retries=retries,
                           save_raw=save_raw, sleep_fn=sleep_fn)
        rows += r
        if doc:
            docs.append(doc)
        if i < len(datasets) - 1:
            common.polite_sleep(sleep, sleep_fn)
    return docs, rows
