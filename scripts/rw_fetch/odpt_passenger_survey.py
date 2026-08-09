#!/usr/bin/env python3
"""P4境界較正 B: ODPT `odpt:PassengerSurvey`(駅別の年度乗降/乗車人員)。

`odpt_rt.py` と同じ流儀: **APIキーは環境変数からのみ**(`ODPT_API_KEY` /
`ODPT_CHALLENGE_API_KEY`)、値はコード・ログ・保存物・報告に一切出さない。
リアルタイムではないので日次スケジューラには結線しない(年1回更新の静的データ)。

------------------------------------------------------------------ 実測(2026-08-10)
枠の振り分け(実接続で確認):
  オープン枠 api.odpt.org           = TokyoMetro のみ(JR-East/Tokyu/Keio は 0 件)
  チャレンジ枠 api-challenge.odpt.org = JR-East / Tokyu / Keio(TokyoMetro は 0 件)

★ **二重計上の判別キーが API に載っている**: `odpt:includeAlighting`。
     false = 乗車人員のみ(JR東日本)     … 改札を通った**入場**だけを数える
     true  = 乗降人員(東急・メトロ・京王)… 入場 + 出場。単純比較すると約2倍ずれる。
   さらに東急・メトロ・京王の「乗降人員」は**直通・改札内乗換の通過客を含む**ため、
   駅の外(街)と出入りした人数ではない。渋谷はこの差が特に大きい(乗換が支配的)。

★ **メトロ渋谷は同じ実数が2レコードに出る**: `TokyoMetro.Hanzomon.Shibuya` と
   `TokyoMetro.Fukutoshin.Shibuya` は `odpt:station` 配列も値も完全に同一(半蔵門線と
   副都心線で改札を共有するため)。素朴に足すと 75 万人/日 を二重に数える。
   `dedupe_records()` が (駅集合, 年度別の値) の同一性で1件に畳む。

★ 再配布: チャレンジ枠は再配布不可の可能性(odpt_rt.py と同じ R5)。`data/realworld/` は
   .gitignore 済みで、README も置く。論文・提出物には生データではなく集計値のみ。
"""
from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from urllib.parse import quote, urlencode

from . import common, ledger

SOURCE = "odpt_passenger_survey"
DATATYPE = "odpt:PassengerSurvey"
OPEN_BASE = "https://api.odpt.org/api/v4/"
CHALLENGE_BASE = "https://api-challenge.odpt.org/api/v4/"
OPEN_KEY_ENV = "ODPT_API_KEY"
CHALLENGE_KEY_ENV = "ODPT_CHALLENGE_API_KEY"

# 渋谷駅に乗り入れる4事業者。scope = そのデータが返る枠(2026-08-10 実測)。
OPERATORS: list[dict] = [
    {"key": "jr_east", "label": "JR東日本", "operator": "odpt.Operator:JR-East",
     "scope": "challenge", "measure": "boarding",
     "note": "乗車人員のみ(odpt:includeAlighting=false)"},
    {"key": "tokyu", "label": "東急電鉄", "operator": "odpt.Operator:Tokyu",
     "scope": "challenge", "measure": "boarding_alighting",
     "note": "乗降人員。東横線と田園都市線が別レコード"},
    {"key": "keio", "label": "京王電鉄", "operator": "odpt.Operator:Keio",
     "scope": "challenge", "measure": "boarding_alighting",
     "note": "乗降人員。井の頭線のみ"},
    {"key": "tokyo_metro", "label": "東京メトロ", "operator": "odpt.Operator:TokyoMetro",
     "scope": "open", "measure": "boarding_alighting",
     "note": "乗降人員。半蔵門線と副都心線は改札共有で同一値が2レコード出る"},
]
OPERATOR_BY_KEY = {o["key"]: o for o in OPERATORS}

# 対象駅。ODPT の駅識別子は `odpt.Station:<事業者>.<路線>.<駅>` なので末尾で判定する。
DEFAULT_STATION = "Shibuya"

CAVEATS = [
    "★ odpt:includeAlighting=false(JR東日本)は**乗車人員のみ**、true(東急・メトロ・京王)は"
    "**乗降人員**。単位が違うので絶対に素で足し引きしてはいけない。",
    "乗降人員には直通運転・改札内乗換の通過客が含まれる。駅の外(街)と出入りした人数ではない。"
    "渋谷は乗換が支配的なので、街への流入量としては大きく過大になる。",
    "東京メトロ渋谷は半蔵門線・副都心線で改札を共有しており、同一実数が2レコードに現れる。"
    "dedupe_records() で1件に畳んでいる(n_duplicates に件数を残す)。",
    "東急渋谷は東横線と田園都市線が別レコード。これは別改札の別計上で重複ではない。",
    "surveyYear は年度。事業者ごとに開始年が違う(京王は 2024 年度以降しか出ない)。",
    "チャレンジ枠(JR東日本・東急・京王)は再配布不可の可能性がある。生データを"
    "リポジトリ・論文・提出物に載せないこと。",
]

README_TEXT = """# data/realworld/odpt_passenger_survey/ — 取り扱い注意

**公共交通オープンデータセンター(ODPT)** の `odpt:PassengerSurvey`(駅別 年度別 乗降者数)です。

## 再配布の制限(重要)

- `scope: "challenge"` のファイル(**JR東日本・東急・京王**)は公共交通オープンデータ
  チャレンジの参加者限定データで、**再配布不可の可能性があります**。
  - リポジトリにコミットしない(`data/realworld/` は .gitignore 済み)。
  - 論文・発表資料・提出物には生データを添付しない(集計値・グラフのみ)。
- `scope: "open"`(**東京メトロ**)はオープン枠ですが、出典表示義務は同じです。

## 出典表示(必須)

> 本データは公共交通オープンデータセンターのデータを利用して作成。

## 数え方が事業者でそろっていない(最重要)

| 事業者 | `odpt:includeAlighting` | 意味 |
|---|---|---|
| JR東日本 | `false` | **乗車人員のみ**(改札入場だけ) |
| 東急・京王・東京メトロ | `true` | **乗降人員**(入場+出場、直通・乗換の通過客込み) |

素朴に合計すると JR だけ半分に数えることになります。境界流量への使い方は
`docs/research/boundary-calibration-data.md` を参照してください。

## API キー

キーは環境変数 `ODPT_API_KEY` / `ODPT_CHALLENGE_API_KEY` からのみ読みます。
**キーの値はこのディレクトリのどのファイルにも書かれていません**(保存直前に機械的にマスク)。
"""


# ------------------------------------------------------------------ URL
def _query(params: dict) -> str:
    return urlencode(params, safe=":", quote_via=quote)


def api_url(scope: str, params: dict, key: str | None) -> str:
    """実要求 URL(キー入り)。**この戻り値は絶対に print / 保存しない。**"""
    base = CHALLENGE_BASE if scope == "challenge" else OPEN_BASE
    full = dict(params)
    if key:
        full["acl:consumerKey"] = key
    return f"{base}{DATATYPE}?{_query(full)}"


def display_url(scope: str, params: dict) -> str:
    """表示・保存用 URL(consumerKey は常に *** 固定)。"""
    base = CHALLENGE_BASE if scope == "challenge" else OPEN_BASE
    return f"{base}{DATATYPE}?{_query(params)}&acl:consumerKey={common.MASK}"


def out_path(root: Path) -> Path:
    return Path(root) / SOURCE / "shibuya.json"


def ensure_readme(root: Path) -> Path:
    path = Path(root) / SOURCE / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != README_TEXT:
        path.write_text(README_TEXT, encoding="utf-8")
    return path


def plan(operators=None) -> list[tuple[str, str]]:
    """`--offline` 用。キーを渡さないので URL に載りようがない。"""
    return [(f"乗降者数 {o['label']} ({o['scope']})",
             display_url(o["scope"], {"odpt:operator": o["operator"]}))
            for o in (operators or OPERATORS)]


# ------------------------------------------------------------------ パース(純関数)
def is_station(record: dict, station: str = DEFAULT_STATION) -> bool:
    """`odpt:station` のいずれかが `.<station>` で終わるか(部分一致で拾いすぎない)。"""
    stations = record.get("odpt:station") or []
    if isinstance(stations, str):
        stations = [stations]
    return any(str(s).rsplit(".", 1)[-1] == station for s in stations)


def year_series(record: dict) -> dict[int, int]:
    """`odpt:passengerSurveyObject` → `{年度: 人数}`。壊れた要素は黙って捨てず飛ばす。"""
    out: dict[int, int] = {}
    for obj in record.get("odpt:passengerSurveyObject") or []:
        if not isinstance(obj, dict):
            continue
        year, val = obj.get("odpt:surveyYear"), obj.get("odpt:passengerJourneys")
        if isinstance(year, int) and isinstance(val, (int, float)):
            out[year] = int(val)
    return out


def _signature(record: dict) -> tuple:
    """重複判定の鍵 = (駅集合, 年度別の値)。改札共有の同一計上を1件に畳むため。"""
    stations = record.get("odpt:station") or []
    if isinstance(stations, str):
        stations = [stations]
    return (tuple(sorted(str(s) for s in stations)),
            tuple(sorted(year_series(record).items())))


def dedupe_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """同一計上(メトロ渋谷の半蔵門線/副都心線)を1件に畳む。戻り値 = (残した, 畳んだ識別子)。"""
    seen: dict[tuple, dict] = {}
    dropped: list[str] = []
    for rec in records:
        sig = _signature(rec)
        if sig in seen:
            dropped.append(str(rec.get("owl:sameAs") or rec.get("@id") or "?"))
            kept = seen[sig].setdefault("_merged_from", [])
            kept.append(str(rec.get("owl:sameAs") or "?"))
            continue
        seen[sig] = rec
    return list(seen.values()), dropped


def to_records(raw: list[dict], op: dict) -> list[dict]:
    """API のレコードを、こちらの語彙(単位を明示した形)に写す。値は無加工。"""
    out: list[dict] = []
    for rec in raw:
        include_alighting = bool(rec.get("odpt:includeAlighting"))
        series = year_series(rec)
        out.append({
            "operator_key": op["key"],
            "operator_label": op["label"],
            "operator": rec.get("odpt:operator") or op["operator"],
            "scope": op["scope"],
            "same_as": rec.get("owl:sameAs"),
            "railways": rec.get("odpt:railway") or [],
            "stations": rec.get("odpt:station") or [],
            "include_alighting": include_alighting,
            # 事業者で数え方が違う。ここで単位を必ず名前にする(後段が取り違えないように)。
            "measure": "boarding_alighting" if include_alighting else "boarding",
            "unit": "persons_per_day",
            "years": {str(y): v for y, v in sorted(series.items())},
            "latest_year": max(series) if series else None,
            "latest_value": series[max(series)] if series else None,
            "merged_from": rec.get("_merged_from") or [],
            "dc_date": rec.get("dc:date"),
        })
    return out


def common_year(records: list[dict]) -> int | None:
    """**全事業者がそろって値を持つ最新の年度**。無ければ None。

    事業者ごとに系列の開始年が違う(京王は 2024 年度以降・メトロは 2020 年度以降)。
    年度をまたいで足すと年の違うものを合計してしまうので、必ずここで揃える。
    """
    if not records:
        return None
    sets = []
    for op in sorted({r["operator_key"] for r in records}):
        years: set[int] = set()
        for r in records:
            if r["operator_key"] == op:
                years |= {int(y) for y in r["years"]}
        sets.append(years)
    shared = set.intersection(*sets) if sets else set()
    return max(shared) if shared else None


def totals_for_year(records: list[dict], year: int) -> dict[str, int]:
    """指定年度の measure 別合計。**measure が違う値は別々に積む**(足してはいけない)。"""
    out: dict[str, int] = {}
    for r in records:
        val = r["years"].get(str(year))
        if val is None:
            continue
        out[r["measure"]] = out.get(r["measure"], 0) + int(val)
    return out


def summarize(records: list[dict]) -> dict:
    """境界較正で最初に見る要約(単位ごと・**年度をそろえて**合計する)。"""
    by_measure: dict[str, int] = {}
    for r in records:
        if r["latest_value"] is None:
            continue
        by_measure[r["measure"]] = by_measure.get(r["measure"], 0) + int(r["latest_value"])
    years = sorted({int(y) for r in records for y in r["years"]})
    cy = common_year(records)
    return {
        "n_records": len(records),
        # ↓ 事業者ごとに最新年度が違うので、これは**年度が混ざった合計**。参考値。
        "latest_by_measure": by_measure,
        "latest_year_by_operator": {r["operator_key"]: r["latest_year"] for r in records},
        "common_year": cy,
        "common_year_by_measure": totals_for_year(records, cy) if cy else {},
        "year_min": years[0] if years else None,
        "year_max": years[-1] if years else None,
        "operators": sorted({r["operator_key"] for r in records}),
        "warning": "measure が違う値どうしを足してはいけない(boarding と boarding_alighting)。"
                   "合計を使うなら common_year_by_measure(年度をそろえた方)を使うこと。",
    }


# ------------------------------------------------------------------ 取得
def fetch_all(root: Path, d: _date | None = None, *, operators=None, station: str = DEFAULT_STATION,
              timeout: float = 45.0, retries: int = 1, sleep: float = 1.0, sleep_fn=None,
              open_key: str | None = None, challenge_key: str | None = None,
              force: bool = False) -> tuple[dict | None, list[dict]]:
    """4事業者の PassengerSurvey を取り、渋谷の行だけを1ファイルにまとめる。

    **キーが無い枠はリクエストを出さずに飛ばし、台帳に理由を残して何も書かない。**
    全枠のキーが無ければ `(None, 台帳行)` を返す(空ファイルを作らない)。
    """
    d = d or common.now_jst().date()
    operators = list(operators or OPERATORS)
    open_key = open_key if open_key is not None else common.resolve_key(OPEN_KEY_ENV)
    challenge_key = (challenge_key if challenge_key is not None
                     else common.resolve_key(CHALLENGE_KEY_ENV))
    keys = {"open": open_key, "challenge": challenge_key}

    if not force and out_path(root).exists():
        common.log("  [passenger_survey] skip (取得済み)")
        return None, [ledger.make_entry(SOURCE, "shibuya", ok=True, n_records=0,
                                        date_jst=d.isoformat(), path=str(out_path(root)),
                                        extra={"skipped": "already-fetched", "complete": True})]

    missing = [o for o in operators if not keys.get(o["scope"])]
    if len(missing) == len(operators):
        envs = sorted({OPEN_KEY_ENV if o["scope"] == "open" else CHALLENGE_KEY_ENV
                       for o in operators})
        common.log(f"[passenger_survey] APIキーが無いので**何も取得せず何も書かない**。"
                   f"環境変数 {' / '.join(envs)} を設定してから再実行すること。", err=True)
        return None, [ledger.make_entry(
            SOURCE, o["key"], ok=False, date_jst=d.isoformat(),
            error=f"key-missing:{OPEN_KEY_ENV if o['scope'] == 'open' else CHALLENGE_KEY_ENV}",
            extra={"scope": o["scope"]}) for o in operators]

    ensure_readme(root)
    records: list[dict] = []
    urls: list[str] = []
    rows: list[dict] = []
    n_dupes: list[str] = []
    for i, op in enumerate(operators):
        key = keys.get(op["scope"])
        env = OPEN_KEY_ENV if op["scope"] == "open" else CHALLENGE_KEY_ENV
        if not key:
            rows.append(ledger.make_entry(SOURCE, op["key"], ok=False, date_jst=d.isoformat(),
                                          error=f"key-missing:{env}", extra={"scope": op["scope"]}))
            continue
        params = {"odpt:operator": op["operator"]}
        url = api_url(op["scope"], params, key)      # ← 絶対に print / 保存しない
        res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
        res.url_masked = display_url(op["scope"], params)
        urls.append(res.url_masked)
        if not res.ok:
            rows.append(ledger.make_entry(SOURCE, op["key"], ok=False, http_status=res.status,
                                          date_jst=d.isoformat(), error=res.error or "failed",
                                          extra={"scope": op["scope"]}))
            continue
        try:
            data = json.loads(res.text())
        except ValueError:
            rows.append(ledger.make_entry(SOURCE, op["key"], ok=False, http_status=res.status,
                                          date_jst=d.isoformat(), error="parse: not-json",
                                          extra={"scope": op["scope"]}))
            continue
        if not isinstance(data, list):
            rows.append(ledger.make_entry(SOURCE, op["key"], ok=False, http_status=res.status,
                                          date_jst=d.isoformat(), error="unexpected-response",
                                          extra={"scope": op["scope"]}))
            continue
        hits = [r for r in data if isinstance(r, dict) and is_station(r, station)]
        kept, dropped = dedupe_records(hits)
        n_dupes += dropped
        records += to_records(kept, op)
        rows.append(ledger.make_entry(
            SOURCE, op["key"], ok=True, http_status=200, n_records=len(kept),
            date_jst=d.isoformat(),
            extra={"scope": op["scope"], "n_total": len(data), "n_station": len(hits),
                   "n_duplicates": len(dropped), "complete": True}))
        if i < len(operators) - 1:
            common.polite_sleep(sleep, sleep_fn)

    if not records:
        common.log("[passenger_survey] 渋谷のレコードが1件も取れなかった(保存しない)。", err=True)
        return None, rows

    meta = common.build_meta(
        SOURCE, module="odpt_passenger_survey.py", urls=urls, n_records=len(records),
        n_missing=sum(1 for r in records if r["latest_value"] is None),
        caveats=CAVEATS, units={"latest_value": "persons_per_day"},
        notes=["年1回更新の静的データ。日次スケジューラには結線していない。",
               "measure 欄で数え方(boarding / boarding_alighting)を明示している。",
               f"対象駅 = {station}(odpt:station の末尾一致)。"],
        extra={"datatype": DATATYPE, "station": station,
               "n_duplicates": len(n_dupes), "duplicate_ids": n_dupes,
               "summary": summarize(records),
               "scopes_used": sorted({o["scope"] for o in operators if keys.get(o["scope"])}),
               "fetch_date_jst": d.isoformat()})
    doc = {"_meta": meta, "data": records}
    common.write_json(out_path(root), doc)
    common.log(f"  [passenger_survey] {len(records)} 件 (重複畳み {len(n_dupes)} 件)")
    return doc, rows


def load_saved(root: Path) -> dict | None:
    """保存済み JSON を読む(ネットワーク不使用)。"""
    path = out_path(root)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None
