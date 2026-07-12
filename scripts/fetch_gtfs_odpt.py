#!/usr/bin/env python3
"""ODPT 静的 GTFS(zip)取得 CLI(第10バッチ続き 2026-07-08)。

ODPT のファイル配信 `…/api/v4/files/{事業者}/data/{事業者}-Train-GTFS.zip` から
鉄道の静的 GTFS を取得して data/ 配下にキャッシュする。fetch_odpt.py と同じ原則:
**オフライン取得 → 静的キャッシュ → シミュはキャッシュのみ読む**(実行中の API 呼び出し禁止)。

APIキー: 環境変数からのみ読む(--key-env で名前を指定。値はログ・出力に一切出さない)。
  プロセス環境に無ければ Windows の User 環境変数(HKCU\\Environment)へフォールバック。

提供状況(2026-07-08 実測):
  - TokyoMetro : オープン枠(api.odpt.org・ODPT_API_KEY)に実在
  - Keio       : チャレンジ枠(api-challenge.odpt.org・ODPT_CHALLENGE_API_KEY)に実在
  - JR-East    : **未投入**(チャレンジ2026の告知には「GTFS形式の時刻表データ」とあるが
                 現時点で files/ にも API にも無い。投入され次第このスクリプトで取得可)
  チャレンジ限定データは再配布不可の可能性 → data/odpt_challenge/ は .gitignore 済み。

使い方:
  python scripts/fetch_gtfs_odpt.py --operator TokyoMetro
  python scripts/fetch_gtfs_odpt.py --operator Keio --challenge
  python scripts/fetch_gtfs_odpt.py --operator JR-East --challenge   # 投入待ち(404なら未提供)

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ODPT 静的 GTFS(zip)取得。キーは環境変数から。")
    p.add_argument("--operator", required=True,
                   help="事業者名(URL の {事業者} 部分。例: TokyoMetro, Keio, JR-East)")
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

    key = _resolve_key(key_env)
    if not key:
        print(f"{key_env} を設定してください(User 環境変数でも可)", file=sys.stderr)
        return 1

    name = f"{args.operator}-Train-GTFS.zip"
    url = f"{base}files/{args.operator}/data/{name}"      # 表示用(キーなし)
    print(f"取得: {url}(acl:consumerKey=***)")
    req = urllib.request.Request(f"{url}?acl:consumerKey={key}",
                                 headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"HTTP 404: {args.operator} の静的 GTFS は未提供(JR東は投入待ち。"
                  "提供開始後に再実行を)", file=sys.stderr)
        else:
            print(f"HTTP {e.code}", file=sys.stderr)   # 本文は出さない(キー漏洩防止)
        return 1
    except Exception as e:
        print(f"取得失敗: {type(e).__name__}", file=sys.stderr)
        return 1
    if blob[:4] != b"PK\x03\x04":
        print("zip でない応答(先頭マジック不一致)。保存を中止。", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(blob)
    print(f"→ {path}({len(blob):,} bytes)")
    print("出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/ "
          "— 利用規約に従い出典表示すること。チャレンジ限定データはコミット禁止(.gitignore 済み)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
