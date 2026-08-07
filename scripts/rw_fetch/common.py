#!/usr/bin/env python3
"""RW-U1 共通基盤: HTTP・秘密のスクラブ・`_meta` 付与・保存・時刻。

ネットワークの**唯一の出口**は `_urlopen` 1点に絞ってある(テストはここだけを差し替えれば
「1リクエストも出ない」ことを機械的に固定できる)。`http_get` は必ずこれを通る。

秘密(APIキー)の扱い:
  `register_secret(値)` に登録した文字列は、`log()` / `write_json()` / `write_bytes()` /
  `scrub()` を通るすべての出力から `***` に置換される。**値は登録以外の目的で保持しない。**
  この置換が効くことは tests/test_rw_fetch.py が機械検査する。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "realworld"
SCHEMA_VERSION = 1
GENERATED_BY = "scripts/rw_fetch/"
USER_AGENT = ("shibuya-simulation/rw-fetch/1.0 "
              "(research use; non-commercial; attribution kept in _meta)")
JST = timezone(timedelta(hours=9))

MASK = "***"
# 短すぎる文字列を秘密として登録すると無関係な出力まで潰れるため下限を設ける。
MIN_SECRET_LEN = 6

# リトライして意味のある HTTP ステータス。404 は**確定した答え**なので再試行しない
# (アメダスの保持境界 10 日の検出がリトライで鈍るのを防ぐ)。
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# ------------------------------------------------------------------ 出典・ライセンス表
# docs/research/rw-data-acquisition.md ⑤「出典表記の義務」の表をそのまま機械可読にしたもの。
ATTRIBUTION: dict[str, dict] = {
    "amedas": {
        "source_name": "気象庁 アメダス(bosai 配信 JSON)",
        "source_url": "https://www.jma.go.jp/bosai/amedas/",
        "license": "公共データ利用規約(第1.0版) / PDL1.0(商用可・出典明示必須・CC BY 4.0 互換)",
        "license_url": "https://www.jma.go.jp/jma/kishou/info/coment.html",
        "attribution": "出典: 気象庁ホームページ(アメダス)。加工して作成。",
        "redistribution": "ok",
    },
    "wbgt": {
        "source_name": "環境省 熱中症予防情報サイト 暑さ指数(WBGT)電子情報提供サービス",
        "source_url": "https://www.wbgt.env.go.jp/",
        "license": "出典明記が必須(他媒体へ転記する場合は環境省である旨を明記)",
        "license_url": "https://www.wbgt.env.go.jp/data_service.php",
        "attribution": "出典: 環境省熱中症予防情報サイト",
        "redistribution": "ok",
    },
    "shibuya_jinryu": {
        "source_name": "渋谷区オープンデータ Location Analyzer 通行/滞在人口",
        "source_url": "https://city-shibuya-data.opendata.arcgis.com/",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0",
        "attribution": ("渋谷区オープンデータ(CC BY 4.0)／データ提供: "
                        "KDDI・技研商事インターナショナル「KDDI Location Analyzer」"),
        "redistribution": "ok",
    },
    "odpt_rt": {
        "source_name": "公共交通オープンデータセンター(ODPT) リアルタイム情報",
        "source_url": "https://www.odpt.org/",
        "license": "ODPT 利用規約(出典表示義務・派生物にも継承)",
        "license_url": "https://developer.odpt.org/",
        "attribution": ("本データは公共交通オープンデータセンターのデータを利用して作成。"
                        "リアルタイム情報の表示時はデータ生成時刻(dc:date)を必ず併記すること。"),
        # ★チャレンジ枠は再配布不可の可能性 → data/realworld/odpt_rt/README.md に明記する。
        "redistribution": "restricted",
    },
    "jma_xml": {
        "source_name": "気象庁 防災情報XML(Atom フィード)",
        "source_url": "https://xml.kishou.go.jp/xmlpull.html",
        "license": "公共データ利用規約(第1.0版) / PDL1.0",
        "license_url": "https://www.jma.go.jp/jma/kishou/info/coment.html",
        "attribution": "出典: 気象庁ホームページ(防災情報XML)。加工して作成。",
        "redistribution": "ok",
    },
}

# シミュ本体が読まないことを人にも伝えるための共通注記。
USAGE_NOTE = ("較正(ラン前)と事後検証(ラン後)にのみ使う。**状態同化はしない**。"
              "シミュ本体(src/)はこのディレクトリを一切読まない。")


# ================================================================ 秘密のスクラブ
_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """出力から潰すべき文字列(APIキー等)を登録する。値はここ以外に保存しない。"""
    if not value:
        return
    v = str(value)
    if len(v) < MIN_SECRET_LEN:
        return
    _SECRETS.add(v)


def clear_secrets() -> None:
    _SECRETS.clear()


def n_secrets() -> int:
    """登録件数(値は返さない)。"""
    return len(_SECRETS)


def scrub(text) -> str:
    """登録済みの秘密を `***` に置換する。URL エンコード形も潰す。"""
    s = text if isinstance(text, str) else str(text)
    for sec in _SECRETS:
        if sec in s:
            s = s.replace(sec, MASK)
        quoted = urllib.parse.quote(sec, safe="")
        if quoted != sec and quoted in s:
            s = s.replace(quoted, MASK)
    return s


def scrub_bytes(data: bytes) -> bytes:
    out = data
    for sec in _SECRETS:
        b = sec.encode("utf-8", "ignore")
        if b and b in out:
            out = out.replace(b, MASK.encode("ascii"))
    return out


def contains_secret(text) -> bool:
    s = text if isinstance(text, str) else str(text)
    return any(sec in s for sec in _SECRETS)


def log(msg="", *, err: bool = False) -> None:
    """標準出力/標準エラーへの唯一の出口。必ず scrub を通す。"""
    print(scrub(msg), file=(sys.stderr if err else sys.stdout))


def resolve_key(env_name: str) -> str | None:
    """環境変数から API キーを解決し、同時に秘密として登録する(値は返すだけ)。

    プロセス環境に無ければ Windows の User 環境変数(HKCU\\Environment)を読む
    (`scripts/fetch_odpt.py::_resolve_key` と同型・同じ環境変数名を踏襲)。
    """
    val = os.environ.get(env_name)
    if not val and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v, _ = winreg.QueryValueEx(k, env_name)
                val = str(v) or None
        except OSError:
            val = None
    if val:
        register_secret(val)
    return val or None


# ================================================================ 時刻
def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def now_jst() -> datetime:
    return datetime.now(JST).replace(microsecond=0)


def iso_utc(dt: datetime | None = None) -> str:
    return (dt or now_utc()).astimezone(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def iso_jst(dt: datetime | None = None) -> str:
    return (dt or now_jst()).astimezone(JST).replace(microsecond=0).isoformat()


def parse_date(spec: str) -> _date:
    """`YYYY-MM-DD` / `YYYYMMDD` を date に。"""
    s = spec.strip()
    if len(s) == 8 and s.isdigit():
        return _date(int(s[:4]), int(s[4:6]), int(s[6:]))
    return _date.fromisoformat(s)


def date_range(start: _date, end: _date) -> list[_date]:
    """start..end(両端含む)。"""
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# ================================================================ HTTP
_REQUEST_COUNT = 0


def request_count() -> int:
    """このプロセスが `http_get` 経由で試みた HTTP 要求数(`--offline` の 0 固定に使う)。"""
    return _REQUEST_COUNT


def reset_request_count() -> None:
    global _REQUEST_COUNT
    _REQUEST_COUNT = 0


def _urlopen(req, timeout):  # pragma: no cover - テストは必ずここを差し替える
    """ネットワークの唯一の出口。テストはこの1点をモンキーパッチする。"""
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class HttpResult:
    url_masked: str
    status: int | None = None
    body: bytes | None = None
    error: str | None = None
    attempts: int = 0
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.body is not None

    def text(self, encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932")) -> str:
        return decode_text(self.body or b"", encodings)


def decode_text(body: bytes, encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932")) -> str:
    for enc in encodings:
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", "replace")


def http_get(url: str, *, timeout: float = 30.0, retries: int = 2, backoff: float = 1.0,
             sleep_fn=None, headers: dict | None = None) -> HttpResult:
    """GET。指数バックオフで**小回数**だけ再試行する。例外文面は型名/コードのみ返す。

    URL にキーが載り得るので、戻り値・ログには `scrub` 済みのものしか出さない。
    """
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    res = HttpResult(url_masked=scrub(url))
    t0 = time.monotonic()
    global _REQUEST_COUNT
    n_try = max(1, int(retries) + 1)
    for i in range(n_try):
        res.attempts += 1
        _REQUEST_COUNT += 1
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with _urlopen(req, timeout) as resp:
                res.status = getattr(resp, "status", None) or resp.getcode()
                res.body = resp.read()
            res.error = None if res.status == 200 else f"HTTP {res.status}"
            if res.status == 200 or res.status not in RETRY_STATUS:
                break
        except urllib.error.HTTPError as exc:      # exc は URL を含むので code だけ使う
            res.status, res.body, res.error = exc.code, None, f"HTTP {exc.code}"
            if exc.code not in RETRY_STATUS:
                break
        except Exception as exc:                   # ネット/デコード等: 型名のみ
            res.status, res.body, res.error = None, None, type(exc).__name__
        if i < n_try - 1:
            sleep_fn(backoff * (2 ** i))
    res.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return res


def polite_sleep(seconds: float, sleep_fn=None) -> None:
    """相手サーバへの礼儀としての間隔。0 以下なら何もしない。"""
    if seconds and seconds > 0:
        (time.sleep if sleep_fn is None else sleep_fn)(seconds)


# ================================================================ `_meta` と保存
def build_meta(source_key: str, *, module: str, urls: list[str], n_records: int,
               n_missing: int, units: dict | None = None, caveats: list[str] | None = None,
               notes: list[str] | None = None, extra: dict | None = None,
               fetched_at_utc: str | None = None) -> dict:
    """保存物に必ず付ける `_meta`(既存 `scripts/fetch_odpt.py` の `_meta` 規約と同型)。"""
    att = ATTRIBUTION.get(source_key, {})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": f"{GENERATED_BY}{module}",
        "source_key": source_key,
        "source_name": att.get("source_name", source_key),
        "source_url": att.get("source_url", ""),
        "source_urls": [scrub(u) for u in urls],
        "license": att.get("license", ""),
        "license_url": att.get("license_url", ""),
        "attribution": att.get("attribution", ""),
        "redistribution": att.get("redistribution", "unknown"),
        "fetched_at_utc": fetched_at_utc or iso_utc(),
        "n_records": int(n_records),
        "n_missing": int(n_missing),
        "units": dict(units or {}),
        "caveats": list(caveats or []),
        "notes": list(notes or []),
        "usage_note": USAGE_NOTE,
        "modified_note": "取得した生値をそのまま構造化しただけで、補間・補正・単位換算はしていない。",
    }
    if extra:
        meta.update(extra)
    return meta


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, obj) -> Path:
    """JSON 保存。**書く直前に必ず scrub** する(キーが保存物に落ちる経路を潰す)。"""
    text = json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
    safe = scrub(text)
    if safe != text:
        log(f"[warn] 保存物に秘密が含まれていたため {MASK} に置換した: {path.name}", err=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe, encoding="utf-8")
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    """生バイト列の保存(パース失敗時の証拠保全 = リスク R2)。ここでも scrub する。"""
    safe = scrub_bytes(data)
    if safe != data:
        log(f"[warn] 生データに秘密が含まれていたため {MASK} に置換した: {path.name}", err=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(safe)
    return path


def day_dir(root: Path, source: str, d: _date) -> Path:
    """日付パーティション `data/realworld/<source>/<YYYY>/<MM>/<YYYY-MM-DD>/`。"""
    return Path(root) / source / f"{d.year:04d}" / f"{d.month:02d}" / d.isoformat()


def month_dir(root: Path, source: str, year: int, month: int) -> Path:
    return Path(root) / source / f"{year:04d}" / f"{month:02d}"


def save_raw_on_failure(root: Path, source: str, name: str, body: bytes,
                        d: _date | None = None) -> Path:
    """パース失敗時に生バイト列を `_raw_failed/` へ退避する(黙って空にしない)。"""
    base = day_dir(root, source, d) if d is not None else Path(root) / source
    return write_bytes(base / "_raw_failed" / name, body)


REQUIRED_META_KEYS = ("schema_version", "generated_by", "source_key", "source_name",
                      "license", "attribution", "fetched_at_utc", "n_records", "n_missing")


def validate_document(doc) -> list[str]:
    """保存物のスキーマ検証(ネットワーク不使用)。問題のリストを返す(空 = 合格)。"""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["トップレベルが dict でない"]
    if "_meta" not in doc:
        return ["_meta が無い"]
    meta = doc["_meta"]
    if not isinstance(meta, dict):
        return ["_meta が dict でない"]
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            problems.append(f"_meta.{key} が無い")
    if not meta.get("attribution"):
        problems.append("_meta.attribution が空(出典明示は全ソース共通の必須条件)")
    if "data" not in doc:
        problems.append("data が無い")
    elif isinstance(doc["data"], list) and meta.get("n_records") not in (None, len(doc["data"])):
        problems.append(f"_meta.n_records={meta.get('n_records')} != len(data)={len(doc['data'])}")
    return problems


def reconfigure_stdio() -> None:
    """Windows コンソールでの日本語出力の文字化け対策(失敗しても致命ではない)。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
