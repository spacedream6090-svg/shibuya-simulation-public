# shibuya-simulation

*A large-scale LLM social simulation of Shibuya — built to observe, not to intervene.*

現実の渋谷を舞台に、LLM を心に持つ住民たちが暮らす人工社会をつくり、そこで**自然に起きること**を記録するためのシミュレーターです。何かを起こすための仕掛けは入れません。街と物理と制度だけを用意し、起きたことをすべてログに残し、あとから観察します。

## 何が起きる場所か

住民は一人ひとりが LLM の心(と、LLM を呼ぶまでもない日課をこなす身体)を持ち、計画を立て、働き、買い物をし、話し、関係をつくります。以下はすべて台本なしに、住民の行動の積み重ねとして立ち現れます。

- **生活** — 1日の計画と実行([cognition/](src/society/cognition/))・仕事と商い([work.py](src/society/work.py) / [commerce.py](src/society/commerce.py) / [services.py](src/society/services.py))・世帯と住まい([household.py](src/society/household.py))・欲求と健康([needs.py](src/society/needs.py) / [health.py](src/society/health.py))
- **経済** — 貨幣の保存を部門勘定で検査できる経済([economy_sfc.py](src/society/economy_sfc.py))・企業と賃金([organizations.py](src/society/organizations.py))・企業間取引([b2b.py](src/society/b2b.py))・税と歳出([government.py](src/society/government.py))
- **組織と関係** — 就労・結成・労務([organizations.py](src/society/organizations.py) / [party.py](src/society/party.py))・友人・交際・共同行動([relations.py](src/society/relations.py) / [friends.py](src/society/friends.py) / [joint.py](src/society/joint.py)・認知上限は [dunbar.py](src/society/dunbar.py))
- **情報と文化** — 会話([conversation.py](src/society/conversation.py))・ゴシップと噂([gossip.py](src/society/gossip.py) / [rumors.py](src/society/rumors.py))・新語・ラベルの誕生と伝播([labeling/](src/society/labeling/))・SNS/メディア([media.py](src/society/media.py))・世界観の書き換わり([worldview.py](src/society/worldview.py))・場所に残る痕跡([traces.py](src/society/traces.py))
- **空間と物理** — OSM 実道路網の上での移動・群衆物理(Social Force Model・[physics.py](src/society/physics.py))・建物内の階と間取り・信号・公共交通([transit_live.py](src/society/transit_live.py))
- **環境** — 実測パラメータから生成する天候([weather_gen.py](src/society/weather_gen.py))・災害([disaster.py](src/society/disaster.py))・偶発事([chance.py](src/society/chance.py))・路上イベント([street.py](src/society/street.py))

住民には出店・提案・結成・イベント開催・ビラ配りなど「世界に手を加える」行動もひととおり開かれています([tools.py](src/society/tools.py))。それを使う者が現れるかどうかも、観察対象です。

## 観察装置としての設計原則

1. **観測がシムを変えない** — 記録・観測のコードは世界を 1 バイトも動かしません。観測を追加するときは「OFF で既存ランとログがバイト一致・ON でも行動列が不変」をテストで機械的に固定します。エージェント側に観測の痕跡(プロンプト差分・LLM 呼数差)は残しません。
2. **エンジンは因子を名指ししない** — 「世界を変えたくなる度」のような変数は実装しません。内面はエージェント自身の経験から生成され、指標はすべて観測者が**事後に**ログから計算します([observer/](src/society/observer/))。
3. **ミクロが正典・マクロは集約** — 集計値のための別モデルを持ちません。街の人流も経済も、個々のエージェントの状態の集約としてだけ現れます。
4. **決定論と再現性** — 中央集権シード([rng.py](src/society/rng.py))+LLM 応答キャッシュ+checkpoint/resume(分割実行が一気通し実行とログ全層で一致)。同じ入力からは同じ歴史が再生されます。
5. **現実との整合** — 実地図(OSM)・実フロア構成・公共交通の実ダイヤ・公表統計からの人口合成・実測気象。現実側の観測データ(気象・人流など)は日次で取得し([scripts/rw_fetch/](scripts/rw_fetch/))、パラメータ較正と事後検証に使います。

## アーキテクチャ

```
src/society/
  engine/        # scheduler(認知LOD)・simulation・checkpoint/resume
  world/         # 空間・道路網・知覚・時計・屋内
  agents/        # 状態容器・ペルソナ・記憶
  cognition/     # 1日の計画・routine(非LLM)/deliberate(LLM)・内省
  llm/           # バックエンド抽象(mock / Ollama / vLLM / OpenAI互換)・応答キャッシュ
  actions/  tools.py   # 行動プリミティブ・「世界に手を加える」ツール群
  labeling/ lang/      # ラベル・新語の生成と伝播
  observer/      # L1 イベントログ・スキーマ・サイドカー(世界には書き戻さない)
  economy_sfc.py organizations.py rumors.py traces.py worldview.py ...  # 各サブシステム
scripts/         # 実行・データ構築・解析(analyze_* 約30本)・実データ取得(rw_fetch/)
viz/             # 2D/3D ビューア・統合ハブ・ライブモニタ
conf/config.yaml # すべての機能はここのスイッチで宣言(規模もパラメータ)
tests/           # 4,000 本超の自動テスト
```

## 動かし方

Python **3.10+**。依存は [pyproject.toml](pyproject.toml) に宣言。LLM は既定で Mock(GPU 不要・決定論)、実 LLM は Ollama / vLLM / OpenAI 互換サーバを `model.backend` で選択します。

> **データについて**: 地図・ダイヤ・人口名簿など `data/`・`env/` 一式は、出典のライセンス(OSM の ODbL、交通事業者の規約等)に従い本リポジトリでは配布していません。本リポジトリはコードと設計の提示を目的としています。

```bash
pip install -e .
pytest -q          # テストスイート
```

| コマンド | 役割 |
|---|---|
| `python scripts/run.py [dotlist上書き]` | シミュ実行。例: `run.seed=7 run.n_agents=100 model.backend=ollama` |
| `python scripts/analyze.py runs/<name>` | 単一ランの基本観測(特徴量・カスケード・ネットワーク・図・report.md) |
| `python scripts/analyze_accounting.py runs/<name>` ほか `analyze_*` 群 | 会計検査・噂・規範・専門分化・組織形成・計画遵守・創発語検知など約30本。すべて L1 ログからの事後計算 |
| `python viz/make_viewer.py runs/<name>` | 2D 地図ビューア+ダッシュボード(HTML) |
| `python viz/make_viewer3d.py runs/<name>` | 3D ビューア(建物高さ・地形・屋内) |
| `python viz/make_hub.py runs/<name>` | ラン成果物を 1 枚のタブ型 HTML に統合 |
| `python scripts/live_viewer.py runs/<name>` | 走行中ランのライブモニタ |

出力はすべて `runs/<name>/` 配下(L1 イベント / L1b LLM / L2 metrics / L3 snapshot の Parquet + `summary.json`)。

## 規模の工学

設計目標は**現実の渋谷と同等の規模(約25万人)を10日間**。これを観察しきるための実装:

- **ストリーミング解析**([scripts/l1_stream.py](scripts/l1_stream.py)) — イベント種別の列レベル絞り込み+row-group 統計による枝刈り。全展開比で最大 252 倍・メモリ定数(204 GiB 展開の解消)。解析スクリプトはこの経路に統一。
- **有界 finalize** — ログ確定処理のピークを row-group 1 個に(一括結合では 25万人×10日は「書き終わり」で落ちる)。一時ファイル+アトミック置換でクラッシュ安全。
- **checkpoint / resume** — 全サブシステムの状態を保存し、分割実行が一気通し実行とログ全層で一致。
- **認知 LOD** — 驚き・予測誤差ゲートで LLM 呼び出しを絞る(日課は非 LLM で進む)。
- **検収の文化** — 4,000 本超のテスト。新機能は既定 OFF+「OFF でログがバイト一致」を機械固定。判定ロジック不変の主張は AST 比較で証明する。

## 倫理

実在の場所・施設名は使いますが、**実在の個人・団体は登場させず**、シミュ内の全出来事・人物は**架空**です。ペルソナは公表統計からの手続き生成のみ。詳細は [ETHICS.md](ETHICS.md)。

## ライセンス

- **コード**: [Apache License 2.0](LICENSE)([NOTICE](NOTICE) 併置)
- **同梱物**: three.js([viz/vendor/](viz/vendor/)・MIT)/ 群衆物理の較正データ([reference/physics_bench/](reference/physics_bench/)・Jülich 歩行実験データ派生・CC BY 4.0・帰属表示は同フォルダ内)
- **データ非配布**: `data/`・`env/` は OSM 由来(ODbL)・公共交通(事業者規約)・商業施設の公開情報などライセンスが混在するため、リポジトリに含めていません
