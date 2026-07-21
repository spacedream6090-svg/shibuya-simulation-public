#!/usr/bin/env python3
"""ODPT 静的 GTFS(zip)取得 CLI(第10バッチ 2026-07-08 / 第48バッチ バス対応 2026-07-22)。

ODPT のファイル配信 `…/api/v4/files/{事業者}/data/{事業者}-{Train|Bus}-GTFS.zip` から
鉄道/バスの静的 GTFS を取得して data/ 配下にキャッシュする。fetch_odpt.py と同じ原則:
**オフライン取得 → 静的キャッシュ → シミュはキャッシュのみ読む**(実行中の API 呼び出し禁止)。

APIキー: 環境変数からのみ読む(--key-env で名前を指定。値はログ・出力に一切出さない)。fetch_odpt.py の
  _resolve_key を共用(プロセス環境 → Windows User 環境変数の順)。**まず匿名(キーなし)で試し**、
  401/403 のときだけキーで再試行する(静的 GTFS はキー不要で配布される事業者があるため)。

提供状況(2026-07-08/07-22 実測):
  - TokyoMetro : 鉄道 GTFS=オープン枠(api.odpt.org・ODPT_API_KEY)に実在
  - Keio       : 鉄道 GTFS=チャレンジ枠(api-challenge.odpt.org・ODPT_CHALLENGE_API_KEY)に実在
  - JR-East    : 鉄道 GTFS **未投入**(投入され次第このスクリプトで取得可)
  - 都営バス(Toei/Bus): files/ への匿名直 DL は **HTTP 403**(要キー)。静的バス GTFS-JP の
                 配布口・データセットは ckan.odpt.org を要確認(GTFS-RT は b_bus_gtfs_rt-toei)。
  チャレンジ限定データは再配布不可の可能性 → data/odpt_challenge/ は .gitignore 済み。

使い方:
  python scripts/fetch_gtfs_odpt.py --operator TokyoMetro                 # 鉄道 GTFS(既定 kind=Train)
  python scripts/fetch_gtfs_odpt.py --operator Keio --challenge
  python scripts/fetch_gtfs_odpt.py --operator Toei --kind Bus            # 都営バス静的 GTFS(匿名優先)
  python scripts/fetch_gtfs_odpt.py --operator Toei --kind Bus --file ToeiBus-GTFS.zip

出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/(利用規約の出典表示に従う)。
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fetch_odpt import _resolve_key  # noqa: E402  キー解決(値は表示しない)を共用

OPEN_BASE = "https://api.odpt.org/api/v4/"
CHALLENGE_BASE = "https://api-challenge.odpt.org/api/v4/"
USER_AGENT = "shibuya-simulation-gtfs-fetch/1.0 (+offline cache builder)"


def _try_download(url_no_key: str, key: str | None, timeout: float):
    """(blob|None, note) を返す。まず匿名(キーなし)で試し、401/403 なら key があれば再試行。
    静的 GTFS はキー不要で配布される事業者があるため匿名優先。key は URL 以外に一切出さない。"""
    attempts = [("匿名", url_no_key)]
    if key:
        attempts.append(("キー", f"{url_no_key}?acl:consumerKey={key}"))
    last = "不明"
    for label, full in attempts:
        req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), f"{label}取得成功"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}({label})"       # 本文は出さない(キー漏洩防止)
            if e.code == 404:
                return None, f"HTTP 404({label}): 未提供"
            # 401/403 は次(キー)へ。他はそのまま次候補も試す。
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}({label})"
    return None, last


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ODPT 静的 GTFS(zip)取得。キーは環境変数から(匿名優先)。")
    p.add_argument("--operator", required=True,
                   help="事業者名(URL の {事業者} 部分。例: TokyoMetro, Keio, JR-East, Toei)")
    p.add_argument("--kind", default="Train", choices=["Train", "Bus"],
                   help="GTFS 種別(既定 Train。Bus=都営バス等の静的バス GTFS)")
    p.add_argument("--file", default=None,
                   help="zip ファイル名を明示(既定: {operator}-{kind}-GTFS.zip)")
    p.add_argument("--challenge", action="store_true",
                   help="チャレンジ枠(api-challenge + ODPT_CHALLENGE_API_KEY + "
                        "data/odpt_challenge/gtfs/)を使う")
    p.add_argument("--key-env", default=None,
                   help="キーの環境変数名(既定: ODPT_API_KEY / --challenge 時 ODPT_CHALLENGE_API_KEY)")
    p.add_argument("--out-dir", default=None,
                   help="保存先(既定: data/odpt/gtfs/ / --challenge 時 data/odpt_challenge/gtfs/)")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args(argv)

    base = CHALLENGE_BASE if args.challenge else OPEN_BASE
    key_env = args.key_env or ("ODPT_CHALLENGE_API_KEY" if args.challenge else "ODPT_API_KEY")
    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO_ROOT / "data" / ("odpt_challenge" if args.challenge else "odpt") / "gtfs"

    key = _resolve_key(key_env)                     # 無ければ None=匿名だけ試す(捏造せず正直に失敗記録)
    name = args.file or f"{args.operator}-{args.kind}-GTFS.zip"
    url = f"{base}files/{args.operator}/data/{name}"      # 表示用(キーなし)
    print(f"取得: {url}(匿名優先 / 失敗時 acl:consumerKey=*** [{key_env}"
          f"{'設定済' if key else '未設定'}])")

    blob, note = _try_download(url, key, args.timeout)
    if blob is None:
        print(f"取得失敗: {note}"
              + ("(静的バス GTFS は事業者/枠により未提供・要キーの場合あり。"
                 "提供状況は ckan.odpt.org を確認)" if args.kind == "Bus" else ""),
              file=sys.stderr)
        return 1
    if blob[:4] != b"PK\x03\x04":
        print("zip でない応答(先頭マジック不一致)。保存を中止。", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(blob)
    print(f"→ {path}({len(blob):,} bytes・{note})")
    print("出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/ "
          "— 利用規約に従い出典表示すること。チャレンジ限定データはコミット禁止(.gitignore 済み)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
