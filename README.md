# shibuya-simulation

渋谷を舞台にした**大規模 LLM 人工社会シミュレーション**。無数の AI 住民が自律的に文化を進化させる
「生命のように変化する人工社会」を作り、その上で **「世界を変えようとする人」の創発**を計測する研究基盤。

## 研究課題

> **世界を変えようとする個体は、生まれつき存在するのか、環境から創発するのか。**

改変者を直接実装しない。経験と社会環境で変化する内部状態から、改変者が自然発生するかを**観測**する。
反証可能な形に落とすと次のようになる:

- **被説明変数 Y** = 4層(空間 / 資源 / 象徴 / social network)への**連続的な書き換え量**
  (POI 新設・資源プール形成・ラベル採用の S 字到達・伝播・新規関係・SNS 到達など、客観カウント)。
- **k** = 経験→内部状態の結合強度。主軸は「ソロ内省での**信念の書き戻し自由度**」(`free | degraded | sham | off`)。
- Y を初期パラメータ(traits)で回帰した **決定係数 R²(k)** を k で掃引する。
  R² が高い→初期条件支配(「生まれつき」寄り)、R² が低く経路依存が強い→創発レジーム。
- init-determined → path-dominated への **相転移点 k\*** を、R² 低下・seed 発散・早期警戒シグナル(EWS)の
  三角測量で探す。交絡を切るため **sham / null / compute 一定**の対照を必須で回す(R1)。

詳細な設計・マイルストーン・OPEN 台帳は [`docs/design.md`](docs/design.md) を参照。

## セットアップ

Python **3.10+**。依存は [`pyproject.toml`](pyproject.toml) に宣言(`numpy` / `networkx` / `omegaconf` / `pyarrow`)。
解析・可視化には追加で `matplotlib`、テストには `pytest` を使う。

```bash
# 依存のインストール(いずれか)
pip install -e .                 # pyproject の依存を取り込む
pip install numpy networkx omegaconf pyarrow matplotlib pytest   # 明示指定でも可

# テスト(pyproject の pythonpath=src で src/ が解決される)
pytest -q
```

各スクリプトは冒頭で `src/` を `sys.path` に挿入するため、**インストールなしでも直接実行できる**。
実 LLM を使う場合のみ [Ollama](https://ollama.com/) をローカルに用意する(既定は GPU 不要の Mock)。

## 主要コマンド

| コマンド | 役割 |
|---|---|
| `python scripts/run.py [dotlist上書き]` | シミュ実行。既定 10体×144step・Mock LLM。例: `run.seed=7 run.n_agents=30 k.writeback=sham` |
| `python scripts/analyze.py runs/<name>` | 単一 run の観測・計測(agent 特徴量 / カスケード / network / EWS / R² / 図・`report.md`) |
| `python scripts/run_sweep.py --modes off sham free --seeds 1 2 3 --agents 60 --steps 144 --prefix pilot` | k 掃引・FSS ラン(直列)。`--agents` を多水準にすると有限サイズスケーリング |
| `python scripts/analyze_sweep.py "runs/pilot_*"` | 条件横断解析(R²(k) の seed 階層ブートストラップ CI・EWS・seed 発散・計算量交絡監査) |
| `python scripts/judge.py runs/<name> [--backend ollama --model qwen3:4b --judges 3]` | LLM-judge ハーネス(**補助**指標)。判定間 Fleiss κ を計算。既定 `--backend mock`(決定論) |
| `python scripts/bench.py --agents 10 40 80 --steps 60` | スケーリング・ベンチ(step 壁時計 / LLM 呼数 / ピークメモリ)。`runs/_bench/bench.json` + markdown 表 |
| `python scripts/build_personas.py 80 [--no-llm]` | ペルソナ名簿の生成(IPF 骨格×尺度分布×LLM 文章化)。`data/personas_80.json` |
| `python scripts/build_icebreak.py --personas data/personas_40.json [--mock]` | 実験前の初期関係(初対面会話)を生成。全 k 条件で同一ファイルを読み交絡を排除 |
| `python viz/make_viewer.py runs/<name>` | 実行結果 → HTML ビューア(地図 `viewer.html` + ダッシュボード `dashboard.html`) |

出力はすべて `runs/<name>/` 配下(L1 イベント/L1b LLM/L2 metrics/L3 snapshot の Parquet + `summary.json`)。

## アーキテクチャ概要

```
src/society/
  config.py  rng.py       # 設定ローダ(規模=パラメータ) / 中央集権シード(再現性)
  llm/                    # バックエンド抽象(mock / ollama、応答キャッシュ、tier ルータ seam)
  world/                  # 空間・知覚(4チャネル)・時計・道路網・交通・交通機関
  agents/                 # 状態容器 / persona / memory / 構成概念バリデータ
  factors/                # trait 初期 vs state 創発の registry・state 更新則
  cognition/              # LOD(驚き/予測誤差ゲート)・routine(非LLM)・deliberate(LLM)・内省
  actions/  tools.py      # 動詞プリミティブ・「世界を変える」ツール(出店/提案/結成/イベント/ビラ)
  labeling/               # ラベル/新語の伝播(全員記録しない・drift する)
  observer/               # 完全可観測ログ・スキーマ・事後計測(研究者frame)
  engine/                 # scheduler(LOD ルーティング)・simulation・metrics
scripts/  tests/  viz/  conf/config.yaml  data/  docs/
```

- **scale = 設定パラメータ**。規模変更が容易(§6 規模ラダー: A 骨格 / B 創発調律 / C 本番)。
- **観測と本体の frame 分離**: シミュ本体は「起きたこと」を記録するだけ。測定・集計はすべて事後に
  L1 ログから([`src/society/observer/`](src/society/observer/))。

設計思想・確定事項(DECIDED)・未確定の継ぎ目(OPEN)は [`docs/design.md`](docs/design.md)、
批判的監査は [`docs/risk-register.md`](docs/risk-register.md)。

## 制約(不変原則)

- **決定論・再現性**: 中央集権シード(`rng.py`)+ 応答キャッシュ(`llm/cache.py`)。Mock は
  プロンプトと rng キーだけから応答が決まり、呼び出し順に依存しない。
- **no-fingerprint(R9)**: engine は因子を名指ししない。`world_change_drive=f(...)` の類は禁止。
  内的状態(効力感・不満・当事者意識)はエージェントに自己生成させず、**観測者が事後に**行動ログから測る。
  state 更新器には行動ログのみを渡し、初期 trait を配線レベルで除外する。
- **交絡制御(R1)**: k 掃引には sham(計算量同一・結合ゼロ)/ null / compute 一定の対照を併走させ、
  3信号(R² 低下・seed 発散・EWS)が対照で出ないことを確認して初めて k\* を主張する。
- **判定の循環回避(R4)**: 世界改変は客観カウントで計上。LLM-judge は補助・κ ≥ 0.7 でのみ採用・
  本体へ逆流しない([`scripts/judge.py`](scripts/judge.py))。

## 倫理

実在の場所・施設名は使うが**実在の個人・団体は登場させず**、シミュ内の全出来事・人物は**架空**である。
ペルソナは公表統計からの手続き生成、データは合成データのみ。詳細は [`ETHICS.md`](ETHICS.md) を参照。

## ライセンス

- **コード**: [Apache License 2.0](LICENSE)([NOTICE](NOTICE) 併置)
- **データ**: コードとは**別ライセンス**。同梱データは出典別に以下に従う。
  - OpenStreetMap 由来(`data/shibuya_osm*.json` / `env/shimokita/*`):
    © OpenStreetMap contributors・[ODbL 1.0](https://opendatacommons.org/licenses/odbl/)(派生データベースは ODbL を継承)
  - 全国の人流オープンデータ(`data/jinryu/`): 国土交通省(株式会社 Agoop 提供データより作成)・
    政府標準利用規約 2.0(CC BY 4.0 互換)。出典・定義・DL 元は [`data/jinryu/SOURCE.md`](data/jinryu/SOURCE.md)
  - 公共交通オープンデータセンター(`data/odpt/`): [ODPT の利用規約](https://developer.odpt.org/)に従う。
    出典表示は各ファイルの `_meta` に付与済み。データの権利は各公共交通事業者等に帰属
  - フロアガイド事実データ(`data/floorguide_shibuya.json` / `data/floor_layouts.json`):
    公開フロアガイド等からの**カテゴリ事実のみ**(店名・ブランド名不記載。出典は各ファイル `meta.sources`)
  - 組織台帳(`data/organizations_shibuya_wide11k.json`): 手続き生成の**合成データ**(実在企業・学校名なし)
  - three.js(`viz/vendor/`): MIT([`viz/vendor/LICENSE`](viz/vendor/LICENSE) 同梱)
