#!/usr/bin/env python3
"""ODPT(公共交通オープンデータセンター)静的データ取得 CLI。

方式(決定論の保護): **オフライン取得 → data/odpt/ に静的キャッシュ → シミュ本体は
キャッシュのみ読む**。シミュ実行中に本 API を叩くことは禁止(実行時のネット呼び出しは
再現性・決定論を壊すため)。本スクリプトは「事前に1回走らせてキャッシュを作る」道具。

APIキー: 環境変数からのみ読む(コード・リポジトリにキーを書かない)。既定は ODPT_API_KEY、
  --key-env で別名(例: ODPT_CHALLENGE_API_KEY=チャレンジ参加者キー)に切替可。プロセス環境に
  無ければ Windows の User 環境変数(HKCU\\Environment)へフォールバック(エディタ再起動不要)。
  未設定なら「<キー名> を設定してください」と表示して終了コード 1(--dry-run を除く)。
  キーの値はログ・ファイル・標準出力に一切出さない(表示 URL は *** でマスク)。

チャレンジ限定データ(公共交通オープンデータチャレンジ): 参加者キーで追加データに
  アクセスできる場合がある。エンドポイントが別の場合は --api-base で切替
  (候補: https://api-challenge.odpt.org/api/v4/)。保存先は --out-dir data/odpt_challenge
  を推奨(限定データは再配布不可の可能性 → .gitignore 済み)。

対象: 渋谷駅に乗り入れる路線に絞る(JR 山手/埼京/湘南新宿、東急 東横/田園都市、
  京王 井の頭、東京メトロ 銀座/半蔵門/副都心)。取得種別:
    odpt:Railway(路線メタ・駅順)/ odpt:Station(駅・位置)/
    odpt:StationTimetable(渋谷駅の駅時刻表)/ [任意] odpt:TrainTimetable(列車時刻表)。

出典・ライセンス: 公共交通オープンデータセンター(ODPT) https://www.odpt.org/
  API v4 https://api.odpt.org/api/v4/ 。ODPT 利用規約(出典表示義務等)に従うこと。
  取得データの内訳・シミュ写像・規約要件は docs/research/odpt-integration.md を参照。

使い方:
  python scripts/fetch_odpt.py --dry-run          # 取得予定 URL を一覧表示(ネット無し・キー不要)
  ODPT_API_KEY=xxxx python scripts/fetch_odpt.py  # 実取得して data/odpt/ に保存
  python scripts/fetch_odpt.py --only 東横        # ラベル/キー部分一致で対象を絞る
  python scripts/fetch_odpt.py --train-timetable  # 列車時刻表(重い・任意)も取得
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

# requests があれば使う。無ければ標準ライブラリ urllib にフォールバック(pyproject 非依存)。
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - 環境依存
    requests = None  # type: ignore

for _stream in (sys.stdout, sys.stderr):  # 日本語出力の文字化け対策(失敗しても致命ではない)
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass

API_BASE = "https://api.odpt.org/api/v4/"
KEY_ENV = "ODPT_API_KEY"     # main() で --key-env に差し替え可
_CONSUMER_KEY = ""           # main() で解決。絶対に print/log/保存しない
USER_AGENT = "shibuya-simulation-odpt-fetch/1.0 (+offline cache builder)"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "odpt"

# 取得データに付す出典・免責メタ(キーは含めない)。
SOURCE_META = {
    "source": "公共交通オープンデータセンター (Public Transportation Open Data Center, ODPT)",
    "source_url": "https://www.odpt.org/",
    "api_base": API_BASE,
    "api_version": "v4",
    "attribution": "本データは公共交通オープンデータセンターのデータを利用して作成。"
                   "出典表示および ODPT 利用規約の遵守が必要。",
    "disclaimer": "データの正確性・完全性は保証されない。ODPT および各事業者は本データの"
                  "利用によって生じた損害について責任を負わない。",
    "usage_note": "決定論保護のため、シミュ実行中はこの静的キャッシュのみ読む"
                  "(実行時の API 呼び出しは禁止)。詳細: docs/research/odpt-integration.md",
}


@dataclass
class Target:
    key: str                 # 出力ファイル名に使う ASCII スラッグ
    label: str               # 日本語ラベル(表示・メタ用)
    operator: str            # odpt:operator sameAs
    railway: str             # odpt:Railway sameAs(最善知の既定値。live で自己修復)
    shibuya_station: str     # 渋谷駅 odpt:Station sameAs(既定値。live で自己修復)
    title_keywords: tuple    # Railway 自己修復時のタイトル一致キーワード


# 渋谷駅に乗り入れる対象路線(ODPT v4 識別子は既知の最善値。live 取得時に
# owl:sameAs が空なら operator 一覧から title 一致で自動補正する=綴り揺れに強い)。
TARGETS: list[Target] = [
    Target("jr_yamanote", "JR山手線", "odpt.Operator:JR-East",
           "odpt.Railway:JR-East.Yamanote",
           "odpt.Station:JR-East.Yamanote.Shibuya", ("山手", "Yamanote")),
    Target("jr_saikyo", "JR埼京線", "odpt.Operator:JR-East",
           "odpt.Railway:JR-East.SaikyoKawagoe",
           "odpt.Station:JR-East.SaikyoKawagoe.Shibuya", ("埼京", "Saikyo")),
    Target("jr_shonanshinjuku", "JR湘南新宿ライン", "odpt.Operator:JR-East",
           "odpt.Railway:JR-East.ShonanShinjuku",
           "odpt.Station:JR-East.ShonanShinjuku.Shibuya", ("湘南新宿", "ShonanShinjuku")),
    Target("tokyu_toyoko", "東急東横線", "odpt.Operator:Tokyu",
           "odpt.Railway:Tokyu.Toyoko",
           "odpt.Station:Tokyu.Toyoko.Shibuya", ("東横", "Toyoko")),
    Target("tokyu_denentoshi", "東急田園都市線", "odpt.Operator:Tokyu",
           "odpt.Railway:Tokyu.DenEnToshi",
           "odpt.Station:Tokyu.DenEnToshi.Shibuya", ("田園都市", "DenEnToshi", "DenentToshi")),
    Target("keio_inokashira", "京王井の頭線", "odpt.Operator:Keio",
           "odpt.Railway:Keio.Inokashira",
           "odpt.Station:Keio.Inokashira.Shibuya", ("井の頭", "井ノ頭", "Inokashira")),
    Target("metro_ginza", "東京メトロ銀座線", "odpt.Operator:TokyoMetro",
           "odpt.Railway:TokyoMetro.Ginza",
           "odpt.Station:TokyoMetro.Ginza.Shibuya", ("銀座", "Ginza")),
    Target("metro_hanzomon", "東京メトロ半蔵門線", "odpt.Operator:TokyoMetro",
           "odpt.Railway:TokyoMetro.Hanzomon",
           "odpt.Station:TokyoMetro.Hanzomon.Shibuya", ("半蔵門", "Hanzomon")),
    Target("metro_fukutoshin", "東京メトロ副都心線", "odpt.Operator:TokyoMetro",
           "odpt.Railway:TokyoMetro.Fukutoshin",
           "odpt.Station:TokyoMetro.Fukutoshin.Shibuya", ("副都心", "Fukutoshin")),
]


# ------------------------------------------------------------------ URL 構築
def _query_string(params: dict) -> str:
    """コロンを保持したままクエリを組み立てる(ODPT は acl:consumerKey / odpt:railway を素で受ける)。"""
    return urlencode(params, safe=":", quote_via=quote)


def _url(datatype: str, params: dict) -> str:
    return f"{API_BASE}{datatype}?{_query_string(params)}"


def _display_url(datatype: str, params: dict) -> str:
    """表示用 URL。キーは必ず *** でマスクする(params にキーは含めない前提)。"""
    return f"{API_BASE}{datatype}?{_query_string(params)}&acl:consumerKey=***"


# ------------------------------------------------------------------ APIキー解決
def _resolve_key(key_env: str) -> str | None:
    """環境変数からキーを解決する。値は返すだけで、表示・保存は絶対にしない。

    プロセス環境に無ければ Windows の User 環境変数(HKCU\\Environment)を直接読む
    (エディタ起動後に設定された User 変数はプロセスに伝播しないため)。"""
    val = os.environ.get(key_env)
    if val:
        return val
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v, _ = winreg.QueryValueEx(k, key_env)
                return str(v) or None
        except OSError:
            return None
    return None


# ------------------------------------------------------------------ HTTP
def _fetch_json(datatype: str, params: dict, timeout: float):
    """(records:list|None, error:str|None) を返す。例外文字列は握りつぶす(URL=キー漏洩防止)。"""
    full_params = dict(params)
    full_params["acl:consumerKey"] = _CONSUMER_KEY
    url = _url(datatype, full_params)  # ← このローカル変数は絶対に print/log しない
    try:
        if requests is not None:
            resp = requests.get(url, timeout=timeout,
                                 headers={"User-Agent": USER_AGENT})
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"
            data = resp.json()
        else:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"          # e には URL が含まれるが code だけ返す
    except Exception as e:                     # ネット/JSON エラー等: 型名のみ(本文は出さない)
        return None, type(e).__name__
    if not isinstance(data, list):
        return None, "unexpected-response"
    return data, None


# ------------------------------------------------------------------ 保存
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(out_dir: Path, name: str, meta: dict, records: list) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    obj = {"_meta": {**SOURCE_META, **meta, "fetched_at": _now(),
                     "count": len(records)},
           "data": records}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 取得予定(dry-run)
def _plan(t: Target, want_train: bool) -> list[tuple[str, str]]:
    """(説明, マスク済み URL) のリスト。実取得もこの順で叩く。"""
    plan = [
        (f"{t.label} / 路線メタ(駅順)",
         _display_url("odpt:Railway", {"owl:sameAs": t.railway})),
        (f"{t.label} / 駅一覧(位置)",
         _display_url("odpt:Station", {"odpt:railway": t.railway})),
        (f"{t.label} / 渋谷駅 駅時刻表",
         _display_url("odpt:StationTimetable", {"odpt:station": t.shibuya_station})),
    ]
    if want_train:
        plan.append((f"{t.label} / 列車時刻表(任意・重い)",
                     _display_url("odpt:TrainTimetable", {"odpt:railway": t.railway})))
    return plan


def _print_plan(targets: list[Target], want_train: bool) -> None:
    print("=" * 70)
    print("ODPT 取得予定 URL(--dry-run。acl:consumerKey は *** でマスク)")
    print(f"ベース: {API_BASE}")
    print("=" * 70)
    n = 0
    for t in targets:
        print(f"\n[{t.key}] {t.label}  operator={t.operator}")
        for desc, url in _plan(t, want_train):
            n += 1
            print(f"  - {desc}")
            print(f"      {url}")
    print(f"\n取得予定リクエスト数: {n}")


# ------------------------------------------------------------------ 実取得
def _find_shibuya_station(stations: list) -> str | None:
    """Station レコード群から渋谷駅の sameAs を探す(.Shibuya 末尾 or タイトル 渋谷)。"""
    for s in stations:
        same = s.get("owl:sameAs", "")
        if same.endswith(".Shibuya"):
            return same
        title = s.get("dc:title") or (s.get("odpt:stationTitle") or {}).get("ja", "")
        if title == "渋谷":
            return same
    return None


def _repair_railway(t: Target, timeout: float) -> tuple[list, str, str | None]:
    """railway sameAs が空振りしたとき operator 一覧から title 一致で補正する。

    戻り値: (railwayレコード群, 有効な railway sameAs, 渋谷駅 sameAs|None)。"""
    recs, err = _fetch_json("odpt:Railway", {"odpt:operator": t.operator}, timeout)
    if err or not recs:
        return [], t.railway, None
    for r in recs:
        title = (r.get("dc:title") or (r.get("odpt:railwayTitle") or {}).get("ja", ""))
        title = title or ""
        if any(k in title or k in r.get("owl:sameAs", "") for k in t.title_keywords):
            same = r.get("owl:sameAs", t.railway)
            shibuya = None
            for st in r.get("odpt:stationOrder", []):
                sid = st.get("odpt:station", "")
                if sid.endswith(".Shibuya"):
                    shibuya = sid
                    break
            return [r], same, shibuya
    return recs, t.railway, None


def _run_fetch(targets: list[Target], out_dir: Path, want_train: bool,
               timeout: float, pause: float) -> int:
    index = {"_meta": {**SOURCE_META, "fetched_at": _now()}, "lines": []}
    total = 0
    print("ODPT 取得開始(→ %s)" % out_dir)
    for t in targets:
        railway_id = t.railway
        shibuya_id = t.shibuya_station
        summary = {"key": t.key, "label": t.label, "operator": t.operator,
                   "railway": railway_id, "railway_count": 0, "station_count": 0,
                   "shibuya_station": shibuya_id, "station_timetable_count": 0,
                   "train_timetable_count": 0, "notes": []}

        # 1) Railway(路線メタ・駅順)
        railways, err = _fetch_json("odpt:Railway", {"owl:sameAs": railway_id}, timeout)
        if err or not railways:
            summary["notes"].append(f"railway owl:sameAs 空振り({err or '0件'}) → operator 一覧で補正")
            railways, railway_id, discovered = _repair_railway(t, timeout)
            summary["railway"] = railway_id
            if discovered:
                shibuya_id = discovered
                summary["shibuya_station"] = shibuya_id
        railways = railways or []
        summary["railway_count"] = len(railways)
        _write(out_dir, f"railway_{t.key}.json",
               {"datatype": "odpt:Railway", "label": t.label, "railway": railway_id},
               railways)
        time.sleep(pause)

        # 2) Station(駅・位置)
        stations, err = _fetch_json("odpt:Station", {"odpt:railway": railway_id}, timeout)
        stations = stations or []
        if err:
            summary["notes"].append(f"station error: {err}")
        summary["station_count"] = len(stations)
        _write(out_dir, f"stations_{t.key}.json",
               {"datatype": "odpt:Station", "label": t.label, "railway": railway_id},
               stations)
        # 渋谷駅 ID が既定で取れないときは Station から探す
        if stations and not any(s.get("owl:sameAs") == shibuya_id for s in stations):
            found = _find_shibuya_station(stations)
            if found:
                shibuya_id = found
                summary["shibuya_station"] = shibuya_id
                summary["notes"].append("渋谷駅 sameAs を Station から自動解決")
        time.sleep(pause)

        # 3) StationTimetable(渋谷駅の駅時刻表)
        stt, err = _fetch_json("odpt:StationTimetable",
                               {"odpt:station": shibuya_id}, timeout)
        stt = stt or []
        if err:
            summary["notes"].append(f"station_timetable error: {err}")
        if not stt:
            summary["notes"].append("駅時刻表 0 件(JR 等は StationTimetable 非提供の場合あり)")
        summary["station_timetable_count"] = len(stt)
        _write(out_dir, f"station_timetable_{t.key}.json",
               {"datatype": "odpt:StationTimetable", "label": t.label,
                "railway": railway_id, "station": shibuya_id}, stt)
        time.sleep(pause)

        # 4) TrainTimetable(任意・重い)
        if want_train:
            tt, err = _fetch_json("odpt:TrainTimetable",
                                  {"odpt:railway": railway_id}, timeout)
            tt = tt or []
            if err:
                summary["notes"].append(f"train_timetable error: {err}")
            summary["train_timetable_count"] = len(tt)
            _write(out_dir, f"train_timetable_{t.key}.json",
                   {"datatype": "odpt:TrainTimetable", "label": t.label,
                    "railway": railway_id}, tt)
            time.sleep(pause)

        line_total = (summary["railway_count"] + summary["station_count"]
                      + summary["station_timetable_count"]
                      + summary["train_timetable_count"])
        total += line_total
        index["lines"].append(summary)
        note = ("  ⚠ " + " / ".join(summary["notes"])) if summary["notes"] else ""
        print(f"  [{t.key:>18}] {t.label:<10} "
              f"Railway={summary['railway_count']} "
              f"Station={summary['station_count']} "
              f"StationTimetable={summary['station_timetable_count']} "
              f"TrainTimetable={summary['train_timetable_count']}{note}")

    idx_path = out_dir / "_index.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 70)
    print(f"合計レコード数: {total}  路線数: {len(targets)}")
    print(f"索引: {idx_path}")
    print("出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/ "
          "— 利用規約に従い出典表示すること。")
    return 0


# ------------------------------------------------------------------ CLI
def _select(only: str | None) -> list[Target]:
    if not only:
        return list(TARGETS)
    sel = [t for t in TARGETS if only in t.label or only in t.key]
    return sel


def main(argv: list[str] | None = None) -> int:
    global API_BASE, KEY_ENV, _CONSUMER_KEY
    p = argparse.ArgumentParser(
        description="ODPT(公共交通オープンデータ)静的キャッシュ取得。"
                    "APIキーは環境変数から読む(既定 ODPT_API_KEY、--key-env で切替)。")
    p.add_argument("--dry-run", action="store_true",
                   help="取得予定 URL を一覧表示(ネット無し・キー不要)")
    p.add_argument("--only", default=None,
                   help="ラベル/キー部分一致で対象路線を絞る(例: 東横, metro_ginza)")
    p.add_argument("--train-timetable", action="store_true",
                   help="odpt:TrainTimetable(列車時刻表・重い)も取得する")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT),
                   help="保存先(既定: data/odpt/。チャレンジ枠は data/odpt_challenge 推奨)")
    p.add_argument("--key-env", default="ODPT_API_KEY",
                   help="APIキーを読む環境変数名(例: ODPT_CHALLENGE_API_KEY)")
    p.add_argument("--api-base", default=API_BASE,
                   help=f"API ベース URL(既定: {API_BASE}。"
                        "チャレンジ枠候補: https://api-challenge.odpt.org/api/v4/)")
    p.add_argument("--timeout", type=float, default=30.0, help="1リクエストのタイムアウト秒")
    p.add_argument("--pause", type=float, default=0.5,
                   help="リクエスト間の待機秒(API への配慮)")
    args = p.parse_args(argv)

    KEY_ENV = args.key_env
    API_BASE = args.api_base if args.api_base.endswith("/") else args.api_base + "/"
    SOURCE_META["api_base"] = API_BASE

    targets = _select(args.only)
    if not targets:
        print(f"該当路線なし: --only {args.only!r}", file=sys.stderr)
        return 1

    if args.dry_run:
        _print_plan(targets, args.train_timetable)
        keyset = bool(_resolve_key(KEY_ENV))
        print(f"\n[dry-run] {KEY_ENV}: {'設定済み' if keyset else '未設定'}(値は表示しません)")
        print("[dry-run] 実取得は行っていません。実行するには --dry-run を外してください。")
        return 0

    key = _resolve_key(KEY_ENV)
    if not key:
        print(f"{KEY_ENV} を設定してください(User 環境変数でも可)", file=sys.stderr)
        return 1
    _CONSUMER_KEY = key

    return _run_fetch(targets, Path(args.out_dir), args.train_timetable,
                      args.timeout, args.pause)


if __name__ == "__main__":
    raise SystemExit(main())
