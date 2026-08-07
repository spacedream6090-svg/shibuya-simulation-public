#!/usr/bin/env python3
"""A5: 気象庁 防災情報XML(Atom フィード)。**外れ値日ラベル**の材料。

用途は気象そのものではなく、`real-world-feedback.md` §4-3「大規模イベント／異常気象が
重なる日は外れ値日としてラベルし較正データから除外」の機械化。

フィード(`docs/research/rw-data-acquisition.md` §2-5):
  長期(毎時更新・数日分) https://www.data.jma.go.jp/developer/xml/feed/{regular,extra,eqvol,other}_l.xml
  高頻度(毎分更新)       同 `_l` なし ← **使わない**

★ 気象庁は「**1日10GB以上のダウンロードで IP を遮断**」と明示している(リスク R4)。
   よって**長期フィードのみ・1時間間隔**に限定する(1日 24〜96 リクエスト)。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date as _date
from datetime import datetime
from pathlib import Path

from . import common, ledger

SOURCE = "jma_xml"
FEED_BASE = "https://www.data.jma.go.jp/developer/xml/feed"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# 長期フィードのみ(高頻度フィードは取らない = R4 対策)。
FEEDS = ("regular_l", "extra_l", "eqvol_l", "other_l")
FEED_LABEL = {
    "regular_l": "定時(長期)", "extra_l": "随時=警報・注意報(長期)",
    "eqvol_l": "地震火山(長期)", "other_l": "その他(長期)",
}

# 東京地方の抽出キーワード(表題・本文・発表官署のいずれかに現れる)。
TOKYO_KEYWORDS = ("東京", "伊豆諸島", "小笠原", "関東甲信")

# 外れ値日ラベルに使う表題キーワード(該当したら「その日は平常ではない」と印を付ける)。
OUTLIER_KEYWORDS = ("特別警報", "警報", "注意報", "熱中症警戒アラート", "台風", "地震",
                    "噴火", "記録的短時間大雨")

CAVEATS = [
    "長期フィード(_l)のみを1時間間隔で取得している。高頻度フィードは R4(IP遮断)対策で使わない。",
    "東京地方の抽出は表題・本文・発表官署のキーワード一致であり、電文本体(XML詳細)は読んでいない。",
    "外れ値日ラベルは候補の提示であって確定ではない。最終判断は較正側で人が行う。",
]


# ------------------------------------------------------------------ URL / パス
def feed_url(feed: str) -> str:
    return f"{FEED_BASE}/{feed}.xml"


def out_path(root: Path, d: _date, feed: str, stamp: str) -> Path:
    return common.day_dir(root, SOURCE, d) / f"{feed}_{stamp}.json"


def plan(feeds=None) -> list[tuple[str, str]]:
    return [(f"防災情報XML {FEED_LABEL.get(f, f)}", feed_url(f)) for f in (feeds or FEEDS)]


# ------------------------------------------------------------------ パース
def _text(elem, tag: str) -> str:
    node = elem.find(ATOM_NS + tag)
    return (node.text or "").strip() if node is not None else ""


def parse_atom(xml_text: str) -> list[dict]:
    """Atom フィード → エントリ列。壊れていれば ParseError/ValueError で落とす(空にしない)。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Atom パース失敗: {exc}") from None
    entries = root.findall(ATOM_NS + "entry")
    out: list[dict] = []
    for e in entries:
        author = e.find(ATOM_NS + "author")
        link = e.find(ATOM_NS + "link")
        rec = {
            "id": _text(e, "id"),
            "title": _text(e, "title"),
            "updated": _text(e, "updated"),
            "content": _text(e, "content"),
            "author": (_text(author, "name") if author is not None else ""),
            "link": (link.get("href") if link is not None else ""),
        }
        rec["tokyo_related"] = is_tokyo(rec)
        rec["outlier_candidate"] = is_outlier(rec)
        out.append(rec)
    return out


def is_tokyo(entry: dict, keywords=TOKYO_KEYWORDS) -> bool:
    blob = " ".join(str(entry.get(k, "")) for k in ("title", "content", "author"))
    return any(k in blob for k in keywords)


def is_outlier(entry: dict, keywords=OUTLIER_KEYWORDS) -> bool:
    blob = " ".join(str(entry.get(k, "")) for k in ("title", "content"))
    return any(k in blob for k in keywords)


def outlier_labels(entries: list[dict]) -> dict[str, list[str]]:
    """東京関連かつ外れ値候補のエントリを **日(JST)ごと** にまとめる。

    戻り値: {"2026-08-16": ["東京都気象警報・注意報", ...]}(重複は畳む・順序は安定)。
    """
    out: dict[str, list[str]] = {}
    for e in entries:
        if not (e.get("tokyo_related") and e.get("outlier_candidate")):
            continue
        raw = e.get("updated") or ""
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        day = dt.astimezone(common.JST).date().isoformat() if dt else "unknown"
        titles = out.setdefault(day, [])
        title = e.get("title") or e.get("content") or "?"
        if title not in titles:
            titles.append(title)
    return {k: out[k] for k in sorted(out)}


# ------------------------------------------------------------------ 取得
def fetch_feed(root: Path, feed: str, *, now=None, timeout: float = 30.0, retries: int = 2,
               save_raw: bool = False, sleep_fn=None) -> tuple[dict | None, list[dict]]:
    now = now or common.now_jst()
    d = now.date()
    stamp = now.strftime("%H%M%S")
    url = feed_url(feed)
    res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    if not res.ok:
        return None, [ledger.make_entry(
            SOURCE, feed, ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=res.error or "failed")]
    try:
        entries = parse_atom(res.text())
    except ValueError as exc:
        raw = common.save_raw_on_failure(root, SOURCE, f"{feed}_{stamp}.xml", res.body or b"", d)
        return None, [ledger.make_entry(
            SOURCE, feed, ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=f"parse: {exc}", path=str(raw))]
    labels = outlier_labels(entries)
    n_tokyo = sum(1 for e in entries if e["tokyo_related"])
    meta = common.build_meta(
        SOURCE, module="jma_xml.py", urls=[url], n_records=len(entries), n_missing=0,
        caveats=CAVEATS,
        notes=[f"feed={feed} ({FEED_LABEL.get(feed, feed)})",
               "tokyo_related / outlier_candidate は本モジュールが付けた派生フラグ"
               "(原文は title/content/updated がそのまま入っている)。"],
        extra={"feed": feed, "snapshot_jst": common.iso_jst(now),
               "n_tokyo_related": n_tokyo,
               "n_outlier_candidates": sum(1 for e in entries if e["outlier_candidate"]),
               "outlier_labels_by_day": labels})
    doc = {"_meta": meta, "data": entries}
    path = out_path(root, d, feed, stamp)
    common.write_json(path, doc)
    if save_raw:
        common.write_bytes(path.with_suffix(".xml"), res.body or b"")
    return doc, [ledger.make_entry(
        SOURCE, feed, ok=True, http_status=200, n_records=len(entries), n_missing=0,
        path=str(path), date_jst=d.isoformat(),
        extra={"complete": True, "n_tokyo_related": n_tokyo})]


def fetch_all(root: Path, *, feeds=None, now=None, timeout: float = 30.0, retries: int = 2,
              sleep: float = 1.5, save_raw: bool = False,
              sleep_fn=None) -> tuple[list[dict], list[dict]]:
    feeds = list(feeds or FEEDS)
    docs, rows = [], []
    for i, f in enumerate(feeds):
        doc, r = fetch_feed(root, f, now=now, timeout=timeout, retries=retries,
                            save_raw=save_raw, sleep_fn=sleep_fn)
        rows += r
        if doc:
            docs.append(doc)
        if i < len(feeds) - 1:
            common.polite_sleep(sleep, sleep_fn)
    return docs, rows
