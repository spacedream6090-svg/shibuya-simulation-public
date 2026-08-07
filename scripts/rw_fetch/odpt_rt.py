#!/usr/bin/env python3
"""A4: ODPT リアルタイム(運行情報 TrainInformation + 在線 Train)。

`docs/research/rw-data-acquisition.md` §2-4 の実測に基づく振り分け:
  odpt:TrainInformation  オープン枠 = TokyoMetro(銀座/半蔵門/副都心)
                         チャレンジ枠 = Tokyu(東横/田園都市) / Keio(井の頭)
  odpt:Train(在線)      チャレンジ枠 = JR-East のみ(山手/埼京/湘南新宿)。
                         東急・京王・メトロは在線を出していない。

**APIキーは環境変数からのみ読む**(既存 `scripts/fetch_odpt.py` と同じ名前を踏襲):
  オープン枠 `ODPT_API_KEY` / チャレンジ枠 `ODPT_CHALLENGE_API_KEY`。
  値はコード・ログ・保存物・報告に一切出さない(`common.register_secret` → `scrub`)。

★ **チャレンジ枠のデータは再配布不可の可能性**(リスク R5)。保存先に README を必ず置き、
   論文・提出物には生データではなく集計値・グラフのみを載せる。`data/realworld/` は
   .gitignore 済み。
★ `dct:valid`(= `dc:date` + 5分)を過ぎたレコードは ODPT ガイドラインにより使用しない。
   本モジュールは期限切れを `data` から外し、件数と識別子を `_meta` に残す(黙って消さない)。
"""
from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

from . import common, ledger

SOURCE = "odpt_rt"
OPEN_BASE = "https://api.odpt.org/api/v4/"
CHALLENGE_BASE = "https://api-challenge.odpt.org/api/v4/"
OPEN_KEY_ENV = "ODPT_API_KEY"
CHALLENGE_KEY_ENV = "ODPT_CHALLENGE_API_KEY"

# 渋谷駅に乗り入れる9路線。scope = そのデータが取れる枠(実測 §2-4)。
RAILWAYS: list[dict] = [
    {"key": "jr_yamanote", "label": "JR山手線", "operator": "odpt.Operator:JR-East",
     "railway": "odpt.Railway:JR-East.Yamanote", "train": "challenge", "info": None},
    {"key": "jr_saikyo", "label": "JR埼京線", "operator": "odpt.Operator:JR-East",
     "railway": "odpt.Railway:JR-East.SaikyoKawagoe", "train": "challenge", "info": None},
    {"key": "jr_shonanshinjuku", "label": "JR湘南新宿ライン", "operator": "odpt.Operator:JR-East",
     "railway": "odpt.Railway:JR-East.ShonanShinjuku", "train": "challenge", "info": None},
    {"key": "tokyu_toyoko", "label": "東急東横線", "operator": "odpt.Operator:Tokyu",
     "railway": "odpt.Railway:Tokyu.Toyoko", "train": None, "info": "challenge"},
    {"key": "tokyu_denentoshi", "label": "東急田園都市線", "operator": "odpt.Operator:Tokyu",
     "railway": "odpt.Railway:Tokyu.DenEnToshi", "train": None, "info": "challenge"},
    {"key": "keio_inokashira", "label": "京王井の頭線", "operator": "odpt.Operator:Keio",
     "railway": "odpt.Railway:Keio.Inokashira", "train": None, "info": "challenge"},
    {"key": "metro_ginza", "label": "東京メトロ銀座線", "operator": "odpt.Operator:TokyoMetro",
     "railway": "odpt.Railway:TokyoMetro.Ginza", "train": None, "info": "open"},
    {"key": "metro_hanzomon", "label": "東京メトロ半蔵門線", "operator": "odpt.Operator:TokyoMetro",
     "railway": "odpt.Railway:TokyoMetro.Hanzomon", "train": None, "info": "open"},
    {"key": "metro_fukutoshin", "label": "東京メトロ副都心線", "operator": "odpt.Operator:TokyoMetro",
     "railway": "odpt.Railway:TokyoMetro.Fukutoshin", "train": None, "info": "open"},
]

CAVEATS = [
    "チャレンジ枠(api-challenge)のデータは再配布不可の可能性がある。論文・提出物には"
    "生データではなく集計値・グラフのみを載せること(リスク R5)。",
    "dct:valid(= dc:date + 5分)を過ぎたレコードは使用しない。期限切れは data から外し、"
    "件数を _meta.n_expired に記録している。",
    "リアルタイム情報を表示する際はデータ生成時刻(dc:date)と出典を必ず併記すること。",
    "JR-East は TrainInformation を提供していない(運行障害は GTFS-RT Alert 側)。",
]

README_TEXT = """# data/realworld/odpt_rt/ — 取り扱い注意

このディレクトリの中身は **公共交通オープンデータセンター(ODPT)** のデータです。

## 再配布の制限(重要)

- `scope: "challenge"` と記録されたファイル(JR東日本の在線 `odpt:Train`、
  東急・京王の運行情報 `odpt:TrainInformation`)は **公共交通オープンデータチャレンジの
  参加者限定データ**であり、**再配布不可の可能性があります**。
- したがって:
  - リポジトリにコミットしない(`data/realworld/` は .gitignore 済み)。
  - 論文・発表資料・提出物には **生データを添付しない**。載せてよいのは集計値・グラフのみ。
  - 第三者へファイルを渡さない。
- `scope: "open"`(東京メトロの運行情報)はオープンデータ枠ですが、**出典表示義務**は同じです。

## 出典表示(必須)

> 本データは公共交通オープンデータセンターのデータを利用して作成。

リアルタイム情報を表示するときは、併せて **データ生成時刻(`dc:date`)** を明記してください。

## 有効期限

各レコードの `dct:valid` は `dc:date + 5分` です。ODPT ガイドラインは有効範囲を超えた
データの使用を認めていません。取得器は期限切れレコードを `data` から外し、件数だけを
`_meta.n_expired` に残しています。

## API キー

キーは環境変数 `ODPT_API_KEY` / `ODPT_CHALLENGE_API_KEY` からのみ読みます。
**キーの値はこのディレクトリのどのファイルにも書かれていません**(取得器が保存直前に
機械的にマスクします)。
"""


# ------------------------------------------------------------------ URL
def _query(params: dict) -> str:
    return urlencode(params, safe=":", quote_via=quote)


def api_url(scope: str, datatype: str, params: dict, key: str | None) -> str:
    """実要求 URL(キー入り)。**この戻り値は絶対に print / 保存しない。**"""
    base = CHALLENGE_BASE if scope == "challenge" else OPEN_BASE
    full = dict(params)
    if key:
        full["acl:consumerKey"] = key
    return f"{base}{datatype}?{_query(full)}"


def display_url(scope: str, datatype: str, params: dict) -> str:
    """表示・保存用 URL。consumerKey は常に *** 固定(キーを渡さないので漏れようがない)。"""
    base = CHALLENGE_BASE if scope == "challenge" else OPEN_BASE
    return f"{base}{datatype}?{_query(params)}&acl:consumerKey={common.MASK}"


def out_path(root: Path, d: _date, datatype: str, stamp: str) -> Path:
    return common.day_dir(root, SOURCE, d) / f"{datatype}_{stamp}.json"


def ensure_readme(root: Path) -> Path:
    path = Path(root) / SOURCE / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != README_TEXT:
        path.write_text(README_TEXT, encoding="utf-8")
    return path


def plan(railways=None) -> list[tuple[str, str]]:
    """`--offline` 用。キーは渡さないので URL に載らない。"""
    railways = list(railways or RAILWAYS)
    out: list[tuple[str, str]] = []
    for scope in ("open", "challenge"):
        ops = sorted({r["operator"] for r in railways if r["info"] == scope})
        for op in ops:
            out.append((f"運行情報 TrainInformation ({scope}) {op}",
                        display_url(scope, "odpt:TrainInformation", {"odpt:operator": op})))
    for r in railways:
        if r["train"]:
            out.append((f"在線 Train ({r['train']}) {r['label']}",
                        display_url(r["train"], "odpt:Train", {"odpt:railway": r["railway"]})))
    return out


# ------------------------------------------------------------------ 有効期限
def _parse_dt(text):
    if not isinstance(text, str) or not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def split_expired(records: list[dict], now: datetime) -> tuple[list[dict], list[str]]:
    """`dct:valid` を過ぎたレコードを外す。戻り値 = (有効, 期限切れの識別子)。"""
    valid, expired = [], []
    for rec in records:
        vt = _parse_dt(rec.get("dct:valid"))
        if vt is not None and vt < now:
            expired.append(str(rec.get("owl:sameAs") or rec.get("@id") or "?"))
        else:
            valid.append(rec)
    return valid, expired


def shibuya_railways(records: list[dict], railways=None) -> list[dict]:
    """渋谷9路線に該当するレコードだけを残す。"""
    wanted = {r["railway"] for r in (railways or RAILWAYS)}
    return [r for r in records if r.get("odpt:railway") in wanted]


def delay_summary(trains: list[dict]) -> dict:
    """在線レコードの遅延(秒)の要約。値が無ければ None(0 と偽らない)。"""
    delays = [r["odpt:delay"] for r in trains
              if isinstance(r.get("odpt:delay"), (int, float))]
    if not delays:
        return {"n": 0, "max_s": None, "mean_s": None, "n_delayed": 0}
    return {"n": len(delays), "max_s": max(delays),
            "mean_s": round(sum(delays) / len(delays), 2),
            "n_delayed": sum(1 for x in delays if x > 0)}


# ------------------------------------------------------------------ 取得
def _fetch(scope: str, datatype: str, params: dict, key: str, *, timeout: float,
           retries: int, sleep_fn=None):
    url = api_url(scope, datatype, params, key)     # ← 絶対に print / 保存しない
    res = common.http_get(url, timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    res.url_masked = display_url(scope, datatype, params)
    if not res.ok:
        return None, res
    try:
        data = json.loads(res.text())
    except ValueError:
        return None, res
    if not isinstance(data, list):
        res.error = "unexpected-response"
        return None, res
    return data, res


def fetch_snapshot(root: Path, *, now=None, railways=None, timeout: float = 30.0,
                   retries: int = 1, sleep: float = 1.0, sleep_fn=None,
                   open_key: str | None = None,
                   challenge_key: str | None = None) -> tuple[list[dict], list[dict]]:
    """運行情報 + 在線の1スナップショットを取得する。戻り値 = (文書リスト, 台帳行)。

    キー未設定の枠は **リクエストを出さずに** 飛ばし、台帳へ理由を残す。
    """
    now = now or common.now_jst()
    railways = list(railways or RAILWAYS)
    open_key = open_key if open_key is not None else common.resolve_key(OPEN_KEY_ENV)
    challenge_key = (challenge_key if challenge_key is not None
                     else common.resolve_key(CHALLENGE_KEY_ENV))
    keys = {"open": open_key, "challenge": challenge_key}
    ensure_readme(root)
    d = now.date()
    stamp = now.strftime("%H%M%S")
    docs, rows = [], []

    # --- 1) 運行情報(事業者単位)
    info_records: list[dict] = []
    info_urls: list[str] = []
    n_expired_total = 0
    expired_ids: list[str] = []
    for scope in ("open", "challenge"):
        ops = sorted({r["operator"] for r in railways if r["info"] == scope})
        for op in ops:
            if not keys[scope]:
                rows.append(ledger.make_entry(
                    SOURCE, f"TrainInformation/{scope}/{op}", ok=False,
                    date_jst=d.isoformat(),
                    error=f"key-missing:{OPEN_KEY_ENV if scope == 'open' else CHALLENGE_KEY_ENV}",
                    extra={"scope": scope}))
                continue
            params = {"odpt:operator": op}
            data, res = _fetch(scope, "odpt:TrainInformation", params, keys[scope],
                               timeout=timeout, retries=retries, sleep_fn=sleep_fn)
            info_urls.append(res.url_masked)
            if data is None:
                rows.append(ledger.make_entry(
                    SOURCE, f"TrainInformation/{scope}/{op}", ok=False,
                    http_status=res.status, date_jst=d.isoformat(),
                    error=res.error or "failed", extra={"scope": scope}))
                continue
            kept = shibuya_railways(data, railways)
            kept, exp = split_expired(kept, now)
            n_expired_total += len(exp)
            expired_ids += exp
            for rec in kept:
                rec["_scope"] = scope
            info_records += kept
            rows.append(ledger.make_entry(
                SOURCE, f"TrainInformation/{scope}/{op}", ok=True, http_status=200,
                n_records=len(kept), date_jst=d.isoformat(),
                extra={"scope": scope, "n_total": len(data), "n_expired": len(exp),
                       "complete": True}))
            common.polite_sleep(sleep, sleep_fn)

    if info_urls:
        meta = common.build_meta(
            SOURCE, module="odpt_rt.py", urls=info_urls, n_records=len(info_records),
            n_missing=0, caveats=CAVEATS,
            notes=["渋谷9路線に該当するレコードのみ保持している。",
                   "dct:valid 超過は data から外し n_expired に件数を残した。"],
            extra={"datatype": "odpt:TrainInformation", "snapshot_jst": common.iso_jst(now),
                   "n_expired": n_expired_total, "expired_ids": expired_ids,
                   "scopes_used": [s for s in ("open", "challenge") if keys[s]]})
        doc = {"_meta": meta, "data": info_records}
        common.write_json(out_path(root, d, "traininformation", stamp), doc)
        docs.append(doc)

    # --- 2) 在線(路線単位・チャレンジ枠 JR東のみ)
    train_records: list[dict] = []
    train_urls: list[str] = []
    n_expired_tr = 0
    expired_tr: list[str] = []
    for r in railways:
        scope = r["train"]
        if not scope:
            continue
        if not keys[scope]:
            rows.append(ledger.make_entry(
                SOURCE, f"Train/{scope}/{r['key']}", ok=False, date_jst=d.isoformat(),
                error=f"key-missing:{OPEN_KEY_ENV if scope == 'open' else CHALLENGE_KEY_ENV}",
                extra={"scope": scope}))
            continue
        params = {"odpt:railway": r["railway"]}
        data, res = _fetch(scope, "odpt:Train", params, keys[scope],
                           timeout=timeout, retries=retries, sleep_fn=sleep_fn)
        train_urls.append(res.url_masked)
        if data is None:
            rows.append(ledger.make_entry(
                SOURCE, f"Train/{scope}/{r['key']}", ok=False, http_status=res.status,
                date_jst=d.isoformat(), error=res.error or "failed", extra={"scope": scope}))
            continue
        kept, exp = split_expired(data, now)
        n_expired_tr += len(exp)
        expired_tr += exp
        for rec in kept:
            rec["_scope"] = scope
        train_records += kept
        rows.append(ledger.make_entry(
            SOURCE, f"Train/{scope}/{r['key']}", ok=True, http_status=200,
            n_records=len(kept), date_jst=d.isoformat(),
            extra={"scope": scope, "n_total": len(data), "n_expired": len(exp),
                   "complete": True}))
        common.polite_sleep(sleep, sleep_fn)

    if train_urls:
        meta = common.build_meta(
            SOURCE, module="odpt_rt.py", urls=train_urls, n_records=len(train_records),
            n_missing=0, caveats=CAVEATS,
            notes=["odpt:delay(秒)を含む。dct:valid 超過は除外済み。"],
            extra={"datatype": "odpt:Train", "snapshot_jst": common.iso_jst(now),
                   "n_expired": n_expired_tr, "expired_ids": expired_tr,
                   "delay_summary": delay_summary(train_records)})
        doc = {"_meta": meta, "data": train_records}
        common.write_json(out_path(root, d, "train", stamp), doc)
        docs.append(doc)

    return docs, rows
