"""モデル人間らしさテストバッテリー(第90バッチ)。

正典: docs/plans/source/design-discussion-20260802.md §4 / docs/plans/dayplan-engaged-plan.md 第90。
設計メモ: docs/research/model-battery-design.md。

目的は**合否判定ではなく**、候補モデルの「人間らしさプロファイル」を作り fleet 内の
役割(心=会話/思考/価値判断 を誰に任せるか)を決めることである。

構成:
  metrics.py   … 純関数の指標計算(I/O なし・既知値でテスト可能)
  reference.py … 対照統計の読み込みと**来歴の強制**(出典・ライセンス・取得日が無ければ拒否)
  stimuli.py   … A〜E 5層の刺激生成(同一プロンプト・同一シード=モデル横断で不変)
  clients.py   … モデル呼び出しアダプタ(ollama / openai互換 / プラセボ)
  harness.py   … 実行 CLI(raw JSONL + manifest を data/battery/raw/<model>/ に書く)
  report.py    … 集計 CLI(モデル×層プロファイル表 + プラセボ健全性判定 + JSON/MD)

★このパッケージは src/ を一切書き換えない(読み取り/import のみ)。
"""
from __future__ import annotations

HARNESS_VERSION = "1"
