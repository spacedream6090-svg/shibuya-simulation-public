#!/usr/bin/env python3
"""過去の実気象データ(気象庁 東京・日別)の取得と凍結(天候の生成型実データ化 前半)。

背景: `src/society/weather.py` は 100% 合成で、8月 = (32, 25, 雨0.30, 雪0.0) に ±3℃ の
一様ゆらぎを足すだけ。**連続する猛暑という現象が構造的に出せない**
(docs/research/dt-snapshot-reproposal-notes.md §2.3 / §3.1)。ユーザー決定
「天候は現実と必ず同期しなくてよい。**データからサンプリング/データを反映した生成**でよい」に従い、
本スクリプトは **その"データ"を一度だけ取ってきて凍結する**役を担う。

方式(決定論の保護。odpt-integration.md / data/odpt/.gitkeep と同一の大原則):
  **オフライン取得 → data/snapshot/ に静的凍結 → シミュ本体はこの静的ファイルのみ読む**。
  シミュ実行中のネットワーク呼び出しは一切しない(本スクリプトは事前に1回走らせる道具)。

出典・ライセンス:
  気象庁ホームページ「過去の気象データ検索(日ごとの値)」
  https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php
  気象庁のコンテンツは権利表記のない限り**公共データ利用規約(第1.0版)**に準拠
  (https://www.jma.go.jp/jma/kishou/info/coment.html。商用可・**出典明示必須**・
   CC BY 4.0 互換。加工した場合はその旨を別途明記すること)。
  → 出力 JSON の meta.source に出典・URL・取得日時・ライセンスを必ず埋める。
  robots.txt: www.data.jma.go.jp / www.jma.go.jp とも **404(=クロール制限の宣言なし)**を実測
  (2026-08-01)。それでも既定 1.5 秒の間隔を空け、1 実行あたりの要求数を月数×年数に限る。

観測点の注意(**重要・§下記 CAVEATS にも入る**):
  - **渋谷区内に気象庁の気温観測点は無い**。最寄の官署は「東京」(prec_no=44 / block_no=47662 /
    アメダス番号 44132)で、**北の丸公園**にあり渋谷駅から約 5.8km NE。
  - 東京は **2014-12-02 に大手町から北の丸公園へ移転**している
    (https://www.jma.go.jp/jma/kishou/know/kansoku/info/20141202_tokyo_rojo.html)。
    移転前後で気温系列に不連続がある(報道ベースで年平均 約 −0.9℃・日最低 約 −1.4℃)。
    → 30年を一括で使うと**系統的に低温側へ引かれる**。fit 側(scripts/fit_weather_gen.py)は
      既定で移転後(2015年以降)の窓を使い、全期間は感度分析としてのみ併記する。
  - 公式値をそのまま渋谷スクランブル交差点の気温とみなすと**都市バイアス分だけ低い**。
    本スクリプトは補正を**しない**(生の観測値を凍結する)。補正は下流の設計判断。

使い方:
  # 30年ぶんの8月を取得して凍結(既定)
  python scripts/fetch_weather_history.py
  # 年・月・出力先を明示
  python scripts/fetch_weather_history.py --years 1996-2025 --months 8 \
      --out data/snapshot/weather_tokyo_aug.json
  # 取得済み JSON のスキーマ + SHA-256 だけを検証(ネットワーク不使用)
  python scripts/fetch_weather_history.py --verify --out data/snapshot/weather_tokyo_aug.json
  # 生 HTML も残す(再現・監査用。既定は残さない)
  python scripts/fetch_weather_history.py --raw-dir /tmp/jma_raw

出力(data/snapshot/weather_tokyo_aug.json):
  {"meta": {schema_version, generated_by, fetched_at_utc, station, source, columns,
            years, months, n_days, n_missing, caveats, payload_sha256},
   "days": [{"date","year","month","day","temp_hi","temp_lo","temp_mean","precip_mm",
             "humid_mean","humid_min","sunshine_h","wind_mean","snow_cm",
             "cond_day","cond_night","flags"}, ...]}

  payload_sha256 = sha256(正準 JSON of {"station","columns","days"})。
  **fetched_at_utc は payload に含めない**ので、同じ年月を取り直せば**同じハッシュ**になる
  (= 取得の決定論を機械検査できる)。

取得できない場合の方針(ユーザー指示):
  ネットワーク不通・レイアウト変更などで取得できないときは**捏造せず**、
  `--emit-schema` で手動投入用の空スキーマ(取得手順書つき)を書き出して正直に報告する。
    python scripts/fetch_weather_history.py --emit-schema --out data/snapshot/weather_tokyo_aug.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
DEFAULT_OUT = REPO_ROOT / "data" / "snapshot" / "weather_tokyo_aug.json"
USER_AGENT = ("shibuya-simulation/weather-history-fetch "
              "(research use; JMA PDL1.0 attribution kept in output)")

# 観測点レジストリ(daily_s1.php = 官署=気圧/湿度/日照まで揃う地点)。
STATIONS = {
    "tokyo": {
        "key": "tokyo",
        "name_ja": "東京",
        "prec_no": 44,          # 都府県・地方 = 東京
        "block_no": 47662,      # 東京(官署)
        "amedas_no": 44132,     # 環境省 WBGT の地点番号と同じ番号体系
        "lat": 35.6917, "lon": 139.7506,   # 北の丸公園(2014-12-02 移転後)
        "note": ("渋谷区内に気象庁の気温観測点は無いため最寄官署を使う。"
                 "渋谷駅から約 5.8km NE。2014-12-02 に大手町から北の丸公園へ移転。"),
    },
}

# daily_s1.php の日別表の列(2026-08-01 実測: 1976〜2025年の8月すべてで 21 セル固定)。
# 添字はセル配列(先頭 = 日)基準。
COLUMNS = [
    {"i": 0,  "key": "day",         "label": "日",                      "unit": ""},
    {"i": 1,  "key": "pres_local",  "label": "気圧(現地)",             "unit": "hPa"},
    {"i": 2,  "key": "pres_sea",    "label": "気圧(海面)",             "unit": "hPa"},
    {"i": 3,  "key": "precip_mm",   "label": "降水量 合計",              "unit": "mm"},
    {"i": 4,  "key": "precip_1h",   "label": "降水量 最大1時間",         "unit": "mm"},
    {"i": 5,  "key": "precip_10m",  "label": "降水量 最大10分間",        "unit": "mm"},
    {"i": 6,  "key": "temp_mean",   "label": "気温 平均",                "unit": "degC"},
    {"i": 7,  "key": "temp_hi",     "label": "気温 最高",                "unit": "degC"},
    {"i": 8,  "key": "temp_lo",     "label": "気温 最低",                "unit": "degC"},
    {"i": 9,  "key": "humid_mean",  "label": "湿度 平均",                "unit": "%"},
    {"i": 10, "key": "humid_min",   "label": "湿度 最小",                "unit": "%"},
    {"i": 11, "key": "wind_mean",   "label": "風速 平均",                "unit": "m/s"},
    {"i": 12, "key": "wind_max",    "label": "最大風速",                 "unit": "m/s"},
    {"i": 13, "key": "wind_max_dir", "label": "最大風速 風向",           "unit": ""},
    {"i": 14, "key": "gust_max",    "label": "最大瞬間風速",             "unit": "m/s"},
    {"i": 15, "key": "gust_max_dir", "label": "最大瞬間風速 風向",       "unit": ""},
    {"i": 16, "key": "sunshine_h",  "label": "日照時間",                 "unit": "h"},
    {"i": 17, "key": "snow_cm",     "label": "降雪 合計",                "unit": "cm"},
    {"i": 18, "key": "snow_depth",  "label": "最深積雪",                 "unit": "cm"},
    {"i": 19, "key": "cond_day",    "label": "天気概況(06-18)",        "unit": ""},
    {"i": 20, "key": "cond_night",  "label": "天気概況(18-翌06)",      "unit": ""},
]
N_CELLS = len(COLUMNS)

# 出力 days[] に残す列(気象生成器の較正に要る分のみ。気圧・風向は落とす)。
KEEP = ["temp_hi", "temp_lo", "temp_mean", "precip_mm", "humid_mean", "humid_min",
        "sunshine_h", "wind_mean", "snow_cm", "cond_day", "cond_night"]
# 「--」を 0.0(現象なし)とみなしてよい列。気温・湿度は欠測扱い(None)にする。
DASH_IS_ZERO = {"precip_mm", "precip_1h", "precip_10m", "snow_cm", "snow_depth", "sunshine_h"}

# 気象庁の値の注記(https://www.data.jma.go.jp/stats/data/mdrr/docs/csv_dl_readme.html ほか)。
#   「--」現象なし / 「)」準正常値 / 「]」資料不足値 / 「#」利用に適さない / 「×」欠測 / 空欄 未観測
ANNOT = {")": "quasi_normal", "]": "insufficient", "#": "unusable", "*": "annotated"}

CAVEATS = [
    "渋谷区内に気象庁の気温観測点は存在しない。最寄官署『東京』(北の丸公園)は渋谷駅から約5.8km NE。",
    "『東京』は2014-12-02に大手町から北の丸公園へ移転しており、移転前後で系列に不連続がある"
    "(報道ベースで年平均 約-0.9℃・日最低 約-1.4℃)。長期窓の一括利用は低温側バイアスを持つ。",
    "都市キャノピー(渋谷スクランブル交差点)の実効気温は官署値より高い。本ファイルは生の観測値であり"
    "都市バイアス補正を含まない。補正は下流の設計判断。",
    "気象庁の掲載値は過去に遡って修正されることがある(取得日時を meta.fetched_at_utc に記録)。",
    "日別値のみ。時刻別気温・全天日射量・WBGT(暑さ指数)実測は本ファイルに含まれない。",
]


# ------------------------------------------------------------------ HTTP / パース
def daily_url(prec_no: int, block_no: int, year: int, month: int) -> str:
    """気象庁『過去の気象データ検索・日ごとの値』の URL(官署 daily_s1)。"""
    return ("https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php"
            f"?prec_no={prec_no}&block_no={block_no}&year={year}&month={month}&day=&view=")


def http_get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    # 実測(2026-08-01): 当該ページは UTF-8。将来 Shift_JIS へ戻った場合に備えフォールバック。
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


_ROW_RE = re.compile(r'<tr class="mtx" style="text-align:right;">(.*?)</tr>', re.S)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _cell_text(html: str) -> str:
    return _TAG_RE.sub("", html).replace("&nbsp;", " ").strip()


def _num(text: str, key: str) -> tuple[float | None, str | None]:
    """気象庁の数値セルを (値, 注記) に。欠測は (None, 理由)。"""
    t = text.strip()
    if t == "" or t == "×":
        return None, "missing"
    if t.startswith("--"):
        return (0.0, None) if key in DASH_IS_ZERO else (None, "no_phenomenon")
    flag = None
    while t and t[-1] in ")]#*":
        flag = ANNOT.get(t[-1], "annotated")
        t = t[:-1].strip()
    if t in ("", "--", "×"):
        return (0.0, flag) if key in DASH_IS_ZERO else (None, "missing")
    try:
        return float(t), flag
    except ValueError:
        return None, "unparsable"


def parse_daily_html(html: str, year: int, month: int) -> list[dict]:
    """daily_s1.php の HTML から1か月ぶんの日別レコードを取り出す(ネットワーク非依存=テスト可能)。

    レイアウトが想定(21セル)と違う行があれば ValueError で**落とす**(黙って埋めない)。
    """
    rows = _ROW_RE.findall(html)
    if not rows:
        raise ValueError(f"日別表の行が見つからない (year={year} month={month})")
    out: list[dict] = []
    for ri, row in enumerate(rows):
        cells = [_cell_text(c) for c in _TD_RE.findall(row)]
        if len(cells) != N_CELLS:
            raise ValueError(f"列数が想定と違う: {len(cells)} != {N_CELLS} "
                             f"(year={year} month={month} row={ri + 1})")
        vals: dict = {}
        flags: dict = {}
        for col in COLUMNS:
            key, raw = col["key"], cells[col["i"]]
            if key in ("day", "wind_max_dir", "gust_max_dir", "cond_day", "cond_night"):
                vals[key] = raw
                continue
            v, f = _num(raw, key)
            vals[key] = v
            if f:
                flags[key] = f
        try:
            day = int(re.sub(r"[^0-9]", "", vals["day"]))
        except ValueError:
            raise ValueError(f"日付セルが読めない: {vals['day']!r} "
                             f"(year={year} month={month} row={ri + 1})") from None
        rec = {
            "date": f"{year:04d}-{month:02d}-{day:02d}",
            "year": year, "month": month, "day": day,
        }
        for key in KEEP:
            v = vals.get(key)
            if isinstance(v, float):
                v = round(v, 3)
            rec[key] = v
        if flags:
            rec["flags"] = {k: v for k, v in sorted(flags.items()) if k in KEEP}
            if not rec["flags"]:
                rec.pop("flags")
        out.append(rec)
    return out


# ------------------------------------------------------------------ 文書の組み立て・検証
def _canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_sha256(doc: dict) -> str:
    """観測ペイロードのみのハッシュ(取得日時を含めない=再取得で同値になる)。"""
    payload = {"station": doc["meta"]["station"],
               "columns": doc["meta"]["columns"],
               "days": doc["days"]}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def build_document(days: list[dict], station: dict, years: list[int], months: list[int],
                   urls: list[str], fetched_at_utc: str) -> dict:
    n_missing = sum(1 for d in days if d.get("temp_hi") is None or d.get("temp_lo") is None)
    doc = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "scripts/fetch_weather_history.py",
            "fetched_at_utc": fetched_at_utc,
            "station": {k: station[k] for k in
                        ("key", "name_ja", "prec_no", "block_no", "amedas_no", "lat", "lon", "note")},
            "source": {
                "name": "気象庁 過去の気象データ検索(日ごとの値)",
                "url_template": daily_url(station["prec_no"], station["block_no"], 0, 0)
                                .replace("year=0", "year={year}").replace("month=0", "month={month}"),
                "urls": urls,
                "license": "公共データ利用規約(第1.0版) / PDL1.0(商用可・出典明示必須・CC BY 4.0 互換)",
                "license_url": "https://www.jma.go.jp/jma/kishou/info/coment.html",
                "attribution": "出典: 気象庁ホームページ(https://www.data.jma.go.jp/stats/etrn/)",
                "modified_note": "HTML 表を機械的に抽出し JSON 化した(値の加工・補正はしていない)。",
            },
            "columns": [{"key": c["key"], "label": c["label"], "unit": c["unit"]}
                        for c in COLUMNS if c["key"] in KEEP],
            "years": sorted(years),
            "months": sorted(months),
            "n_days": len(days),
            "n_missing_temp": n_missing,
            "caveats": list(CAVEATS),
        },
        "days": days,
    }
    doc["meta"]["payload_sha256"] = payload_sha256(doc)
    return doc


def validate_document(doc: dict) -> list[str]:
    """スキーマ検証。問題のリストを返す(空 = 合格)。捏造検出のため必須列は厳格に見る。"""
    problems: list[str] = []
    if not isinstance(doc, dict) or "meta" not in doc or "days" not in doc:
        return ["トップレベルに meta / days が無い"]
    meta, days = doc["meta"], doc["days"]
    if meta.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version が {meta.get('schema_version')}(期待 {SCHEMA_VERSION})")
    for key in ("generated_by", "fetched_at_utc", "station", "source", "columns",
                "years", "months", "n_days", "caveats", "payload_sha256"):
        if key not in meta:
            problems.append(f"meta.{key} が無い")
    src = meta.get("source", {})
    for key in ("name", "urls", "license", "license_url", "attribution"):
        if not src.get(key):
            problems.append(f"meta.source.{key} が空(出典明示は PDL1.0 の必須条件)")
    if not isinstance(days, list) or not days:
        problems.append("days が空")
        return problems
    seen: set[str] = set()
    for i, d in enumerate(days):
        for key in ("date", "year", "month", "day", "temp_hi", "temp_lo", "precip_mm"):
            if key not in d:
                problems.append(f"days[{i}] に {key} が無い")
        date = d.get("date")
        if date in seen:
            problems.append(f"days[{i}] 日付重複: {date}")
        seen.add(date)
        if d.get("temp_hi") is not None and d.get("temp_lo") is not None:
            if d["temp_lo"] > d["temp_hi"]:
                problems.append(f"days[{i}] temp_lo > temp_hi: {date}")
        p = d.get("precip_mm")
        if p is not None and p < 0:
            problems.append(f"days[{i}] precip_mm < 0: {date}")
    if meta.get("n_days") != len(days):
        problems.append(f"meta.n_days={meta.get('n_days')} != len(days)={len(days)}")
    got = payload_sha256(doc)
    if got != meta.get("payload_sha256"):
        problems.append(f"payload_sha256 不一致: 記載 {meta.get('payload_sha256')} / 実算 {got}")
    return problems


def empty_schema_document(station: dict, years: list[int], months: list[int]) -> dict:
    """取得できないときの手動投入用テンプレ(days は空。手順書を meta に入れる)。"""
    doc = build_document([], station, years, months, urls=[],
                         fetched_at_utc="MANUAL-FILL-ME")
    doc["meta"]["manual_entry_instructions"] = [
        "1. https://www.data.jma.go.jp/risk/obsdl/ を開く。",
        "2. 地点 = 東京(都府県 = 東京)、項目 = 日別値の『最高気温』『最低気温』『平均気温』"
        "『降水量の日合計』『平均湿度』『最小相対湿度』『日照時間』『平均風速』を選ぶ。",
        "3. 期間 = 対象年の 8/1〜8/31 を年ごとに指定(連続する期間で複数年を一括指定も可)。",
        "4. 表示オプションで『均質番号』『品質情報』を付けると欠測/準正常値が判別しやすい。",
        "5. ダウンロードした CSV(Shift_JIS / CRLF)を days[] のスキーマへ手で写す。",
        "6. days[] を埋めたら `python scripts/fetch_weather_history.py --verify --out <path>` で"
        "スキーマと payload_sha256 を再計算・検証する(--rehash で書き戻し)。",
        "7. 出典表示『出典: 気象庁ホームページ』は必ず meta.source.attribution に残すこと。",
    ]
    doc["meta"]["days_schema"] = {
        "date": "YYYY-MM-DD", "year": "int", "month": "int", "day": "int",
        "temp_hi": "float degC | null", "temp_lo": "float degC | null",
        "temp_mean": "float degC | null", "precip_mm": "float mm (現象なしは 0.0)",
        "humid_mean": "float % | null", "humid_min": "float % | null",
        "sunshine_h": "float h", "wind_mean": "float m/s | null",
        "snow_cm": "float cm", "cond_day": "str 天気概況(昼)", "cond_night": "str 天気概況(夜)",
        "flags": "dict(列 -> quasi_normal|insufficient|unusable|missing) 省略可",
    }
    doc["meta"]["payload_sha256"] = payload_sha256(doc)
    return doc


# ------------------------------------------------------------------ CLI
def _parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="気象庁 過去の気象データ(日別)の取得と凍結")
    ap.add_argument("--station", default="tokyo", choices=sorted(STATIONS))
    ap.add_argument("--years", default="1996-2025",
                    help="例 1996-2025 / 2015-2025,1994 (既定 1996-2025 = 30年)")
    ap.add_argument("--months", default="8", help="例 8 / 7,8,9 (既定 8)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sleep", type=float, default=1.5, help="要求間隔[秒](既定 1.5)")
    ap.add_argument("--raw-dir", default=None, help="生 HTML の保存先(監査用・既定は保存しない)")
    ap.add_argument("--from-dir", default=None,
                    help="ネットワークを使わず、このディレクトリの保存済み HTML から組み立てる")
    ap.add_argument("--verify", action="store_true", help="取得せず既存 --out を検証するだけ")
    ap.add_argument("--rehash", action="store_true",
                    help="--verify と併用: payload_sha256 を再計算して書き戻す(手動投入後の確定用)")
    ap.add_argument("--emit-schema", action="store_true",
                    help="取得せず手動投入用の空スキーマを書き出す(取得不能時のフォールバック)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    station = STATIONS[args.station]
    years = _parse_years(args.years)
    months = _parse_years(args.months)

    if args.verify:
        if not out.exists():
            print(f"[NG] ファイルが無い: {out}", file=sys.stderr)
            return 2
        doc = json.loads(out.read_text(encoding="utf-8"))
        if args.rehash:
            doc["meta"]["payload_sha256"] = payload_sha256(doc)
            _write(out, doc)
            print(f"[rehash] payload_sha256 = {doc['meta']['payload_sha256']}")
        problems = validate_document(doc)
        if problems:
            for p in problems:
                print(f"[NG] {p}", file=sys.stderr)
            return 1
        print(f"[OK] {out} スキーマ合格 / days={len(doc['days'])} / "
              f"payload_sha256={doc['meta']['payload_sha256']}")
        return 0

    if args.emit_schema:
        doc = empty_schema_document(station, years, months)
        _write(out, doc)
        print(f"[emit-schema] 手動投入用テンプレを書いた: {out}")
        print("  取得手順は meta.manual_entry_instructions を参照。")
        return 0

    days: list[dict] = []
    urls: list[str] = []
    failures: list[str] = []
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
    from_dir = Path(args.from_dir) if args.from_dir else None
    total = len(years) * len(months)
    n = 0
    for year in years:
        for month in months:
            n += 1
            url = daily_url(station["prec_no"], station["block_no"], year, month)
            name = f"daily_s1_{station['block_no']}_{year:04d}{month:02d}.html"
            try:
                if from_dir:
                    html = (from_dir / name).read_text(encoding="utf-8")
                else:
                    html = http_get(url)
                    if raw_dir:
                        (raw_dir / name).write_text(html, encoding="utf-8")
                recs = parse_daily_html(html, year, month)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                failures.append(f"{year}-{month:02d}: {type(exc).__name__}: {exc}")
                print(f"  [{n}/{total}] {year}-{month:02d} 失敗: {exc}", file=sys.stderr)
                continue
            days.extend(recs)
            urls.append(url)
            print(f"  [{n}/{total}] {year}-{month:02d} 取得 {len(recs)} 日")
            if not from_dir and n < total:
                time.sleep(max(0.0, args.sleep))

    if not days:
        print("[NG] 1日も取得できなかった。--emit-schema で手動投入テンプレへ切り替えること。",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    days.sort(key=lambda d: d["date"])
    fetched = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")
    doc = build_document(days, station, years, months, urls, fetched)
    if failures:
        doc["meta"]["fetch_failures"] = failures
        doc["meta"]["payload_sha256"] = payload_sha256(doc)   # failures は payload 外
    problems = validate_document(doc)
    if problems:
        for p in problems:
            print(f"[NG] {p}", file=sys.stderr)
        return 1
    _write(out, doc)
    print(f"[OK] {out}")
    print(f"  days={len(days)} years={years[0]}-{years[-1]} months={months} "
          f"missing_temp={doc['meta']['n_missing_temp']}")
    print(f"  payload_sha256={doc['meta']['payload_sha256']}")
    if failures:
        print(f"  ★ 取得失敗 {len(failures)} 件(meta.fetch_failures に記録)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
