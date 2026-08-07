#!/usr/bin/env python3
"""A1: 気象庁アメダス(bosai 配信 JSON)・地点 44132「東京」の 10 分値。

エンドポイント(`docs/research/rw-data-acquisition.md` §2-1 で 2026-08-07 に実接続検証):
  最新観測時刻  https://www.jma.go.jp/bosai/amedas/data/latest_time.txt
  地点3時間ブロック https://www.jma.go.jp/bosai/amedas/data/point/{id}/{yyyymmdd}_{HH}.json
  ({HH} = 00,03,06,09,12,15,18,21。1リクエストで 10 分値×最大 18 = 3 時間分)

★ **保持は約10日**(実測: 10日前=200 / 13日前=404)。本選期間 8/15〜30 の 10 分値は
   期間中に取らないと**永久に失われる**(リスク R1)。→ `ledger.gap_report` が未取得日を出す。

★ 渋谷区内に気象庁の観測点は無い。44132「東京」は**北の丸公園**にあり渋谷駅から約 5.76km NE。
   都市キャニオンの実効気温より系統的に低い。**補正はしない**(生値を凍結・リスク R7)。

値の形: 各項目は `[値, 品質フラグ]`(フラグ 0 = 正常)。本モジュールは
**値を書き換えない**。フラグが 0 でないセルと、値が null のセルを `n_missing` に数えるだけ。
"""
from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path

from . import common, ledger

SOURCE = "amedas"
BASE = "https://www.jma.go.jp/bosai/amedas"
STATION = "44132"
STATION_INFO = {
    "id": "44132",
    "name_ja": "東京",
    "site": "北の丸公園(千代田区)",
    "lat": 35.6917, "lon": 139.7506,
    "distance_from_shibuya_km": 5.76,
    "kind": "A(官署) = 気温・湿度・気圧・風・日照・降水の全要素",
}

# 3時間ブロックの開始時刻。1日 = 8 ブロック = 8 リクエスト。
BLOCK_HOURS = ("00", "03", "06", "09", "12", "15", "18", "21")
N_BLOCKS = len(BLOCK_HOURS)
RETENTION_DAYS = 10          # 実測(§2-1 の 200/404 境界)

# 欠測を数える対象(44132 が実際に観測している要素)。
CORE_FIELDS = ("temp", "humidity", "pressure", "precipitation10m", "wind",
               "windDirection", "sun10m")

CAVEATS = [
    "渋谷区内に気象庁の観測点は無い。44132「東京」は北の丸公園にあり渋谷駅から約5.76km NE。",
    "北の丸公園は緑地内のため、渋谷スクランブル交差点の実効気温より系統的に低い"
    "(都市キャニオン・アスファルト輻射なし)。本ファイルは生の観測値で都市バイアス補正を含まない。",
    "44132 は 2014-12-02 に大手町から北の丸公園へ移転しており、移転前後で系列に不連続がある。",
    "bosai の point JSON は公式 API として仕様公開されたものではない(気象庁サイトの内部"
    "エンドポイント)ため、予告なく形式が変わり得る(リスク R2)。",
    "保持期間は約10日。過ぎた日の 10 分値は後追い取得できない(リスク R1)。",
]

UNITS = {
    "temp": "degC", "humidity": "%", "pressure": "hPa", "normalPressure": "hPa",
    "precipitation10m": "mm/10min", "precipitation1h": "mm/h", "precipitation3h": "mm/3h",
    "precipitation24h": "mm/24h", "wind": "m/s", "windDirection": "16方位(1=NNE..16=N)",
    "sun10m": "min/10min", "sun1h": "h", "snow": "cm",
    "maxTemp": "degC", "minTemp": "degC", "gust": "m/s",
}


# ------------------------------------------------------------------ URL / パス
def latest_time_url() -> str:
    return f"{BASE}/data/latest_time.txt"


def block_url(d: _date, hh: str, station: str = STATION) -> str:
    return f"{BASE}/data/point/{station}/{d.strftime('%Y%m%d')}_{hh}.json"


def day_path(root: Path, d: _date, station: str = STATION) -> Path:
    return common.day_dir(root, SOURCE, d) / f"amedas_{station}_{d.strftime('%Y%m%d')}.json"


def plan_day(d: _date, station: str = STATION) -> list[tuple[str, str]]:
    """`--offline` 用の取得予定(説明, URL)。ネットワークには一切出ない。"""
    return [(f"アメダス {station} {d.isoformat()} {hh}:00〜(10分値×18)", block_url(d, hh, station))
            for hh in BLOCK_HOURS]


def within_retention(d: _date, today: _date, retention_days: int = RETENTION_DAYS) -> bool:
    """後追い取得がまだ間に合うか(実測境界 10 日)。"""
    age = (today - d).days
    return 0 <= age <= retention_days


# ------------------------------------------------------------------ パース
def _ts_to_iso(ts: str) -> str | None:
    """`20260807180000` → `2026-08-07T18:00:00+09:00`(観測時刻は JST)。"""
    if len(ts) != 14 or not ts.isdigit():
        return None
    try:
        dt = datetime(int(ts[0:4]), int(ts[4:6]), int(ts[6:8]),
                      int(ts[8:10]), int(ts[10:12]), int(ts[12:14]), tzinfo=common.JST)
    except ValueError:
        return None
    return dt.isoformat()


def _split_value(v):
    """`[値, フラグ]` を (値, フラグ) に。それ以外の形はそのまま (値, None)。"""
    if isinstance(v, list) and len(v) == 2 and isinstance(v[1], (int, float)):
        return v[0], int(v[1])
    return v, None


def parse_point_block(obj) -> list[dict]:
    """3時間ブロック JSON → 時刻順のレコード列。**値は一切書き換えない。**

    形が想定と違えば ValueError で落とす(黙って空にしない = リスク R2)。
    """
    if not isinstance(obj, dict):
        raise ValueError(f"point ブロックが dict でない: {type(obj).__name__}")
    out: list[dict] = []
    for ts in sorted(obj):
        fields = obj[ts]
        iso = _ts_to_iso(ts)
        if iso is None:
            raise ValueError(f"観測時刻キーが想定形式(YYYYMMDDhhmmss)でない: {ts!r}")
        if not isinstance(fields, dict):
            raise ValueError(f"観測値が dict でない: {ts}={type(fields).__name__}")
        values: dict = {}
        flags: dict = {}
        for key in sorted(fields):
            val, flag = _split_value(fields[key])
            values[key] = val
            if flag is not None and flag != 0:
                flags[key] = flag
        rec = {"timestamp": ts, "obs_time_jst": iso, "values": values}
        if flags:
            rec["flags"] = flags
        out.append(rec)
    return out


def merge_blocks(blocks: list[list[dict]]) -> list[dict]:
    """複数ブロックのレコードを時刻でマージ(重複は後勝ちにせず先勝ち = 同値のはず)。"""
    seen: dict[str, dict] = {}
    for recs in blocks:
        for r in recs:
            seen.setdefault(r["timestamp"], r)
    return [seen[k] for k in sorted(seen)]


def count_missing(records: list[dict], fields=CORE_FIELDS) -> dict:
    """欠測セルを数える。**埋めない・推定しない**(リスク R6)。

    absent = 項目キー自体が無い / null = 値が None / flagged = 品質フラグが 0 以外。
    """
    absent = null = flagged = bad_cells = 0
    per_field: dict[str, int] = {}
    for r in records:
        vals = r.get("values", {})
        fl = r.get("flags", {})
        for f in fields:
            bad = False
            if f not in vals:
                absent += 1
                bad = True
            elif vals[f] is None:
                null += 1
                bad = True
            if f in fl:
                flagged += 1
                bad = True
            if bad:
                bad_cells += 1
                per_field[f] = per_field.get(f, 0) + 1
    # n_missing = **使えないセルの個数**(absent/null/flagged は重なりうるので内訳は別建て)。
    return {"n_missing": bad_cells, "n_absent": absent, "n_null": null,
            "n_flagged": flagged, "per_field": per_field,
            "n_cells": len(records) * len(fields)}


def _numeric(records, field):
    out = []
    for r in records:
        v = r.get("values", {}).get(field)
        if isinstance(v, (int, float)) and field not in r.get("flags", {}):
            out.append(float(v))
    return out


def summarize_day(records: list[dict]) -> dict:
    """日次まとめ。**値が1つも無い項目は None**(0 で埋めない)。"""
    temps = _numeric(records, "temp")
    hums = _numeric(records, "humidity")
    prec = _numeric(records, "precipitation10m")
    wind = _numeric(records, "wind")
    sun = _numeric(records, "sun10m")
    return {
        "n_timestamps": len(records),
        "first_obs_jst": records[0]["obs_time_jst"] if records else None,
        "last_obs_jst": records[-1]["obs_time_jst"] if records else None,
        "temp_max": max(temps) if temps else None,
        "temp_min": min(temps) if temps else None,
        "temp_mean": round(sum(temps) / len(temps), 3) if temps else None,
        "n_temp_obs": len(temps),
        "humidity_mean": round(sum(hums) / len(hums), 3) if hums else None,
        "precipitation_sum_mm": round(sum(prec), 3) if prec else None,
        "wind_max": max(wind) if wind else None,
        "sun_sum_min": round(sum(sun), 3) if sun else None,
    }


# ------------------------------------------------------------------ 文書組み立て
def build_document(records: list[dict], d: _date, *, urls: list[str], blocks: list[dict],
                   station: str = STATION, fetched_at_utc: str | None = None) -> dict:
    miss = count_missing(records)
    n_ok = sum(1 for b in blocks if b.get("ok"))
    complete = n_ok == N_BLOCKS
    meta = common.build_meta(
        SOURCE, module="amedas.py", urls=urls, n_records=len(records),
        n_missing=miss["n_missing"], units=UNITS, caveats=CAVEATS,
        fetched_at_utc=fetched_at_utc,
        notes=[f"1日 = {N_BLOCKS} ブロック(3時間ごと)。取得できたブロック {n_ok}/{N_BLOCKS}。",
               "品質フラグ 0 以外のセルは値を残したうえで flags に記録し、欠測として数える。"],
        extra={
            "station": dict(STATION_INFO, id=station),
            "date_jst": d.isoformat(),
            "blocks": blocks,
            "n_blocks_ok": n_ok,
            "n_blocks_expected": N_BLOCKS,
            "complete": complete,
            "missing_breakdown": miss,
            "retention_days": RETENTION_DAYS,
            "daily_summary": summarize_day(records),
        })
    return {"_meta": meta, "data": records}


# ------------------------------------------------------------------ 取得
def fetch_day(root: Path, d: _date, *, station: str = STATION, timeout: float = 30.0,
              retries: int = 2, sleep: float = 1.0, sleep_fn=None,
              today: _date | None = None) -> tuple[dict | None, list[dict]]:
    """1日ぶん(8ブロック)を取得して保存する。戻り値 = (保存した文書 | None, 台帳行)。

    404 は「その時刻の観測がまだ無い(未来)」か「保持期間切れ」のどちらか。
    **どちらも失敗として正直に記録し、値を捏造しない。**
    """
    today = today or common.now_jst().date()
    rows: list[dict] = []
    urls: list[str] = []
    blocks: list[dict] = []
    parsed: list[list[dict]] = []
    if not within_retention(d, today):
        age = (today - d).days
        common.log(f"  [amedas] ⚠ {d.isoformat()} は {age} 日前 = 保持{RETENTION_DAYS}日を超過。"
                   "404 が返る見込み(10分値は回復不能)。", err=True)
    for i, hh in enumerate(BLOCK_HOURS):
        url = block_url(d, hh, station)
        urls.append(url)
        res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
        blk = {"hour": hh, "ok": False, "http_status": res.status, "n_records": 0,
               "error": res.error}
        if res.ok:
            try:
                recs = parse_point_block(json.loads(res.text()))
            except (ValueError, TypeError) as exc:
                raw = common.save_raw_on_failure(
                    root, SOURCE, f"{d.strftime('%Y%m%d')}_{hh}.json", res.body or b"", d)
                blk["error"] = f"parse: {type(exc).__name__}"
                blk["raw_saved"] = str(raw)
                common.log(f"  [amedas] パース失敗 {d} {hh} → 生JSONを保存: {raw}", err=True)
            else:
                blk["ok"] = True
                blk["n_records"] = len(recs)
                parsed.append(recs)
        blocks.append(blk)
        if i < len(BLOCK_HOURS) - 1:
            common.polite_sleep(sleep, sleep_fn)

    records = merge_blocks(parsed)
    n_ok = sum(1 for b in blocks if b["ok"])
    if n_ok == 0:
        rows.append(ledger.make_entry(
            SOURCE, f"{station}/{d.isoformat()}", ok=False,
            http_status=blocks[0]["http_status"] if blocks else None,
            n_records=0, n_missing=0, date_jst=d.isoformat(),
            error="all-blocks-failed",
            extra={"complete": False, "n_blocks_ok": 0, "n_blocks_expected": N_BLOCKS}))
        return None, rows

    doc = build_document(records, d, urls=urls, blocks=blocks, station=station)
    path = day_path(root, d, station)
    common.write_json(path, doc)
    rows.append(ledger.make_entry(
        SOURCE, f"{station}/{d.isoformat()}", ok=True,
        http_status=200, n_records=len(records),
        n_missing=doc["_meta"]["n_missing"], path=str(path), date_jst=d.isoformat(),
        extra={"complete": n_ok == N_BLOCKS, "n_blocks_ok": n_ok,
               "n_blocks_expected": N_BLOCKS}))
    return doc, rows


def fetch_latest_time(*, timeout: float = 15.0, retries: int = 1) -> str | None:
    """最新観測時刻(疎通確認用の最小リクエスト1本)。"""
    res = common.http_get(latest_time_url(), timeout=timeout, retries=retries)
    return res.text().strip() if res.ok else None


def missing_days(root: Path, today: _date, *, station: str = STATION,
                 window_days: int = RETENTION_DAYS) -> list[_date]:
    """保存済みファイルの実在から未取得日を出す(台帳が壊れていても効く二重の網)。"""
    out = []
    for d in common.date_range(today - timedelta(days=window_days), today - timedelta(days=1)):
        if not day_path(root, d, station).exists():
            out.append(d)
    return out
