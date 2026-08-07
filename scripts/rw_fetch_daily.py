#!/usr/bin/env python3
"""RW-U1 現実世界データ日次取得のエントリポイント(薄い起動器)。

  python scripts/rw_fetch_daily.py --offline   # ドライラン(HTTP要求 0 本)
  python scripts/rw_fetch_daily.py --help

本体は `scripts/rw_fetch/`(cli.py がオーケストレータ)。設計の一次資料は
`docs/research/rw-data-acquisition.md`。**シミュ本体(src/ conf/)には一切触れない**。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rw_fetch.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
