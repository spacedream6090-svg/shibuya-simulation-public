"""RW-U1: 現実世界データの取得器群(オフライン取得 → 静的凍結)。

設計・エンドポイント・ライセンス・リスクの一次資料は
`docs/research/rw-data-acquisition.md`(2026-08-07・全ソース実接続検証済み)。

大原則(既存 `scripts/fetch_odpt.py` / `scripts/fetch_weather_history.py` の流儀を継承):
  1. **オフライン取得 → data/realworld/ に静的凍結 → シミュ本体は一切読まない。**
     `src/` からの参照ゼロ・`conf/` へのキー追加ゼロ(8/12 フリーズ対象外の条件)。
  2. **APIキーは環境変数からのみ読む。** 値をコード・ログ・保存物・報告に一切出さない
     (`common.register_secret` に登録 → `common.scrub` が出力経路を機械的に潰す)。
  3. **欠測を偽の値で埋めない**(リスク R6)。欠測は `null` + `_meta.n_missing` + 取得台帳。
  4. **パース失敗時は生バイト列をそのまま保存して失敗を報告する**(リスク R2。黙って空にしない)。

使い方: `python scripts/rw_fetch_daily.py --offline`(ドライラン)/ `--help`。
"""
from __future__ import annotations

__all__ = ["common", "ledger", "amedas", "wbgt", "shibuya_jinryu", "odpt_rt", "jma_xml", "cli"]
