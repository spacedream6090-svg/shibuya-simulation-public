#!/usr/bin/env python3
"""RW-U1 取得台帳: いつ・何を・何件・欠測いくつ・HTTP ステータス を1行ずつ追記する。

`docs/research/rw-data-acquisition.md` リスク **R9(取得漏れの静かな蓄積)** の対策。
`data/realworld/_ledger.jsonl` に **1行1レコードで追記**(JSON Lines = 途中で落ちても
既存行が壊れない)。研究文書は `_ledger.json` と書いているが、R9 本文の「1行ずつ追記」に
忠実な JSONL を採る(読む側は `read_all` を通せば同じ)。

最重要の機能は **未取得日の検出**。アメダスは保持 **10 日**(実測・リスク R1)なので、
「まだ間に合う欠け(URGENT)」と「もう二度と取れない欠け(LOST)」を分けて報告する。
"""
from __future__ import annotations

import json
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from . import common

LEDGER_NAME = "_ledger.jsonl"

# ソース別の後追い可能日数(None = 期限なし)。docs/research/rw-data-acquisition.md §4-2。
RETENTION_DAYS: dict[str, int | None] = {
    "amedas": 10,        # 実測: point/{id}/{date}_{HH}.json は 10 日で 404
    "wbgt": None,        # 実況は月初からの累積 = 月内なら後追い可
    "wbgt_forecast": 0,  # 予測は上書きされる = その日に取らないと消える
    "shibuya_jinryu": None,
    "odpt_rt": 0,        # dct:valid=5分。後追い不可
    "jma_xml": 3,        # 長期フィードは「数日分」を掲載
    # ↓ P4 境界較正の静的ソース。5年に1度/年1回の更新なので期限の概念が無い(日次対象外)。
    "transport_census": None,
    "odpt_passenger_survey": None,
}

# 日次実行で毎日取れているべきソース(未取得日の検出対象)。
DAILY_SOURCES = ("amedas", "wbgt", "jma_xml")


def ledger_path(root: Path) -> Path:
    return Path(root) / LEDGER_NAME


def make_entry(source: str, target: str, *, ok: bool, http_status=None, n_records: int = 0,
               n_missing: int = 0, path: str | None = None, error: str | None = None,
               date_jst: str | None = None, ts_utc: str | None = None,
               extra: dict | None = None) -> dict:
    """台帳1行。`date_jst` は「そのデータが属する日」(取得日ではない)。"""
    row = {
        "ts_utc": ts_utc or common.iso_utc(),
        "date_jst": date_jst or common.now_jst().date().isoformat(),
        "source": source,
        "target": common.scrub(target),
        "ok": bool(ok),
        "http_status": http_status,
        "n_records": int(n_records),
        "n_missing": int(n_missing),
        "path": common.scrub(path) if path else None,
        "error": common.scrub(error) if error else None,
    }
    if extra:
        row["extra"] = extra
    return row


def append(root: Path, rows) -> int:
    """台帳へ追記する。戻り値 = 書いた行数。**追記のみ・既存行は決して書き換えない。**"""
    if isinstance(rows, dict):
        rows = [rows]
    rows = list(rows)
    if not rows:
        return 0
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        text = common.scrub(json.dumps(row, ensure_ascii=False, sort_keys=True))
        lines.append(text)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(rows)


def read_all(root: Path) -> list[dict]:
    """台帳を読む。壊れた行は捨てずに `{"_broken": 生テキスト}` として返す(黙って消さない)。"""
    path = ledger_path(root)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            out.append({"_broken": line})
            continue
        out.append(obj if isinstance(obj, dict) else {"_broken": line})
    return out


# ------------------------------------------------------------------ 集計
def ok_days(rows, source: str) -> set[str]:
    """そのソースで **成功した** 日(date_jst)の集合。"""
    return {r.get("date_jst") for r in rows
            if r.get("source") == source and r.get("ok") and r.get("date_jst")}


def day_status(rows, source: str, day: str) -> str:
    """`ok` / `partial` / `failed` / `missing`。

    `partial` = 成功行はあるが `extra.complete` が False(アメダスの 8 ブロックが揃っていない等)。
    """
    same = [r for r in rows if r.get("source") == source and r.get("date_jst") == day]
    if not same:
        return "missing"
    oks = [r for r in same if r.get("ok")]
    if not oks:
        return "failed"
    completes = [r.get("extra", {}).get("complete") for r in oks]
    if any(c is True for c in completes):
        return "ok"
    if any(c is False for c in completes):
        return "partial"
    return "ok"


def missing_days(rows, source: str, start: _date, end: _date) -> list[str]:
    """start..end のうち `ok` でない日(= partial / failed / missing)を返す。"""
    return [d.isoformat() for d in common.date_range(start, end)
            if day_status(rows, source, d.isoformat()) != "ok"]


def gap_report(rows, source: str, today: _date, *, lookback_days: int = 30) -> dict:
    """未取得日を「まだ間に合う(URGENT)」と「もう取れない(LOST)」に分けて返す。

    保持期間 `RETENTION_DAYS[source]` を超えた欠けは後追いできない = LOST。
    当日(today)は取得途中でありうるので `pending` として別に数える。
    """
    retention = RETENTION_DAYS.get(source)
    start = today - timedelta(days=lookback_days)
    urgent, lost, pending, partial = [], [], [], []
    for d in common.date_range(start, today):
        day = d.isoformat()
        st = day_status(rows, source, day)
        if st == "ok":
            continue
        if d == today:
            pending.append(day)
            continue
        if st == "partial":
            partial.append(day)
        age = (today - d).days
        if retention is None:
            urgent.append(day)
        elif age <= retention:
            urgent.append(day)
        else:
            lost.append(day)
    return {
        "source": source,
        "today": today.isoformat(),
        "retention_days": retention,
        "lookback_days": lookback_days,
        "urgent": urgent,       # まだ後追い取得できる欠け → 今すぐ取る
        "lost": lost,           # 保持期間を過ぎた欠け → 二度と取れない(正直に記録するだけ)
        "partial": partial,     # 成功はしたが不完全な日
        "pending": pending,     # 当日(まだ途中でありうる)
        "n_ok": len(ok_days(rows, source)),
    }


def summary(rows) -> dict:
    """ソース別の件数・最終成功時刻・失敗件数。"""
    out: dict[str, dict] = {}
    for r in rows:
        src = r.get("source")
        if not src:
            continue
        s = out.setdefault(src, {"n_rows": 0, "n_ok": 0, "n_fail": 0, "n_records": 0,
                                 "n_missing": 0, "last_ok_utc": None, "errors": {}})
        s["n_rows"] += 1
        s["n_records"] += int(r.get("n_records") or 0)
        s["n_missing"] += int(r.get("n_missing") or 0)
        if r.get("ok"):
            s["n_ok"] += 1
            ts = r.get("ts_utc")
            if ts and (s["last_ok_utc"] is None or ts > s["last_ok_utc"]):
                s["last_ok_utc"] = ts
        else:
            s["n_fail"] += 1
            key = str(r.get("error") or r.get("http_status") or "unknown")
            s["errors"][key] = s["errors"].get(key, 0) + 1
    return out


def format_report(rows, today: _date, *, sources=None, lookback_days: int = 30) -> str:
    """人が読む取得台帳レポート(`--report`)。ネットワーク不使用。"""
    sources = list(sources or DAILY_SOURCES)
    summ = summary(rows)
    lines = ["=" * 72,
             f"RW 取得台帳レポート (今日 = {today.isoformat()} JST / 全 {len(rows)} 行)",
             "=" * 72]
    if not rows:
        lines.append("台帳が空。まだ一度も取得していない。")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{'source':<16}{'rows':>6}{'ok':>6}{'fail':>6}{'records':>10}"
                 f"{'missing':>9}  last_ok_utc")
    for src in sorted(summ):
        s = summ[src]
        lines.append(f"{src:<16}{s['n_rows']:>6}{s['n_ok']:>6}{s['n_fail']:>6}"
                     f"{s['n_records']:>10}{s['n_missing']:>9}  {s['last_ok_utc'] or '-'}")
        if s["errors"]:
            errs = ", ".join(f"{k}×{v}" for k, v in sorted(s["errors"].items()))
            lines.append(f"{'':<16}  失敗内訳: {errs}")
    lines.append("")
    lines.append("-" * 72)
    lines.append("未取得日の検出(★アメダスは保持10日 = 過ぎたら永久に取れない)")
    lines.append("-" * 72)
    n_urgent = n_lost = 0
    for src in sources:
        rep = gap_report(rows, src, today, lookback_days=lookback_days)
        n_urgent += len(rep["urgent"])
        n_lost += len(rep["lost"])
        ret = "無期限" if rep["retention_days"] is None else f"{rep['retention_days']}日"
        lines.append(f"[{src}] 保持={ret}")
        if rep["urgent"]:
            lines.append(f"  ⚠ URGENT 今すぐ後追い取得できる欠け {len(rep['urgent'])} 日: "
                         + ", ".join(rep["urgent"]))
        if rep["lost"]:
            lines.append(f"  ✖ LOST 保持期間切れで回復不能 {len(rep['lost'])} 日: "
                         + ", ".join(rep["lost"]))
        if rep["partial"]:
            lines.append(f"  △ PARTIAL 不完全な日 {len(rep['partial'])} 日: "
                         + ", ".join(rep["partial"]))
        if not (rep["urgent"] or rep["lost"] or rep["partial"]):
            lines.append("  OK 欠けなし")
    lines.append("")
    lines.append(f"合計: URGENT {n_urgent} 日 / LOST {n_lost} 日")
    if n_urgent:
        lines.append("→ `python scripts/rw_fetch_daily.py --backfill` で後追い取得できる。")
    return "\n".join(lines)
