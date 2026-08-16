# 本選 GPU 待ち・計算量削減 E1 実測チェックリスト(第24バッチ A1)

本選機(7 GPU)が手に入ってから実測する項目の手順書。**実測せずに数値を約束しない。**
文献値は「文献値」と明記し、我々の実測で置き換える。関連: [compute-optimization.md](../docs/plans/compute-optimization.md)
E1-1〜E1-4 / 起動スクリプト [launch-vllm-finals.ps1](launch-vllm-finals.ps1) / プロファイル
[conf/profiles/finals-vllm7.yaml](../conf/profiles/finals-vllm7.yaml)。

前提の不変条件(全項目で守る):
- **R1(呼数の k 非依存)**: どの最適化も LLM 呼数と RNG draw 列を変えてはならない。最適化は
  「同じ呼を安く/速く」だけ。呼数が動く設計は不採用(compute-optimization.md の方針)。
- **ゴールデン L1**: 既定 config での L1 バイト一致(pytest の golden)は最適化後も緑のまま。
- 検証は mock スモーク + 実 LLM 短ラン(≤2日)で(validation-runs-short 準拠)。長期ランはしない。

---

## E4 ★本選の正準起動手順 — 「conf を1枚に凍結してから起動する」(A8・2026-08-16)

**結論から**: 本選は `conf/finals_observe.yaml` を `--profile` に直接渡して起動**しない**。
`finals_observe.yaml`(用途=世界)と `conf/profiles/finals-vllm7.yaml`(用途=配線)を
[`scripts/freeze_config.py`](../scripts/freeze_config.py) で**解決済みの1枚**へ落とし、
**その1枚だけ**を `--profile` に渡す。理由は 2 つ:

1. **正準コマンドは、そのままでは起動できない(仕様)。** `finals_observe.yaml` は
   `model.backend` を基底の `mock` のまま残してある(縦煙とドライランをそのまま通すため)。
   一方 `run.n_agents: 250000` なので、**β6 の mock fail-fast ガード**
   (`scripts/run.py::check_mock_production`: `n_agents >= 10,000` ∧ `backend == "mock"` ∧
   `allow_mock_production == false` → **起動時 RuntimeError**)に必ず当たる。
   これは**バグではなく設計**で、「配線を忘れたまま 25 万体を mock で回す」事故を潰す保険。
   したがって `--profile conf/finals_observe.yaml` 単独起動は**常に失敗するのが正しい**。
2. **起動条件を事後に再構成できないから。** 実際の設定は「基底 < env < profile < dotlist」の
   4 段重ねで決まる。dotlist を人間が正確に覚えていない限り、何で回したかは事後に復元できない。
   凍結すると sha256 が付き、`run_manifest` / 進捗報告と突き合わせられる。

### 手順(本番直前に 1 回)

```bash
# ① 世界(finals_observe)と 配線(finals-vllm7)を合流した「解決済み1枚」を作る。
#    ★freeze_config は --profile を 1 本しか取らない(load_config の仕様)。合流は下の
#      どちらかで行う。両者の出力は sha256 まで一致することを 2026-08-16 に実測確認済み。
#
#  (推奨) 事前合流: 2 ファイルを OmegaConf で重ねた中間 profile を作り、それを凍結する
python - <<'PY'
from omegaconf import OmegaConf
m = OmegaConf.merge(OmegaConf.load("conf/finals_observe.yaml"),
                    OmegaConf.load("conf/profiles/finals-vllm7.yaml"))
open("/tmp/finals_merged_profile.yaml", "wb").write(OmegaConf.to_yaml(m).encode("utf-8"))
PY
python scripts/freeze_config.py --profile /tmp/finals_merged_profile.yaml

#  (別解) dotlist で model ブロックを流し込む(13 個。URL 配列も dotlist で通る)
python scripts/freeze_config.py --profile conf/finals_observe.yaml \
  model.backend=vllm model.cache=true model.format=json model.temperature=0.7 \
  model.timeout_s=120 model.max_tokens=320 model.plan_max_tokens=448 \
  model.reflect_max_tokens=768 model.reflect_think=false model.name=qwen3:8b \
  'model.servers=[http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005,http://localhost:8006]' \
  'model.tiers.reflect=[http://localhost:8000]' \
  'model.tiers.default=[http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005,http://localhost:8006]'

# ② 出た1枚を目視 → sha256 を控える(既定 conf/finals_<YYYYMMDD>_frozen.yaml + .sha256)
#    確認するのは最低この 5 行: model.backend / model.servers の本数 /
#    run.n_agents / pool.present_cap / world.calendar.start_date
grep -nE "backend:|n_agents:|present_cap:|start_date:" conf/finals_*_frozen.yaml

# ③ 本選はこの1行だけで起動する(dotlist を 1 つも打たない = 打ち間違いようがない)
python scripts/run.py --profile conf/finals_<YYYYMMDD>_frozen.yaml run.name=finals1
```

### 注意(踏むと痛い順)

- **凍結ファイルはコミットしない。** 本番直前に生成する運用成果物であって正典ではない
  (正典は 基底 conf + profile)。公開ミラーの除外域でもないのでコミットすると公開される。
- **Δt≠10 の凍結ファイルを `--profile` で使わない。** `--profile` は基底へ重ねてから
  **もう一度** `apply_dt` を通す。正準の `run.dt_min: 10` では恒等パスなので 1 バイトも
  変わらないが、Δt≠10 だと二重変換になる(根拠と同値性テストは `tests/test_launch_guard.py` の
  β6 節 — freeze_config.py の docstring が挙げる `tests/test_freeze_config.py` は**存在しない**
  ファイル名で、実体はこちら)。
- **`world.calendar.start_date` が `auto` でないこと**を ② で必ず見る。`auto` は
  `load_config` が**凍結を作った日**へ解決するので、凍結日と起動日がずれると
  曜日・給料日・generated 天候が意図と変わる(本選 conf は `2026-08-22` 固定にしてある)。
- **watchdog 経由でも同じ1枚を渡す**(下の E3 のコマンドの `--profile` を凍結ファイルへ差し替える)。

---

## E0 ★checkpoint / dormant 世代の剪定禁止(第114 G2・2026-08-14 確定)

**本選ランの `checkpoint/` は 1 世代も消さない。** `ckpt-NNNNNN.pkl.gz` と同 step の
`dormant-NNNNNN.pkl.gz` は**必ず対で**残す(片方だけでは在場者しか復元できない)。

理由: **ペルソナ文の本文・記憶ストリーム・関係台帳の全対**は、checkpoint と dormant
サイドカーが**唯一の複製**である。第114 の GT ロガー(memory.parquet / relations.parquet)は
**日境界の粒度**しか持たず、**半日粒度の完全状態は checkpoint にしか無い**。
これが消えると「観測痕跡から内部状態をどこまで復元できるか」という問い自体が
真値を失って成立しなくなる(復元実験の中止に直結する)。詳細と剪定順序は
[finals-reliability-plan.md §1.1](../docs/plans/finals-reliability-plan.md)。

運用の具体:

1. バックアップは **`python scripts/backup_run.py --run-dir <run_dir> --dest <dest> --ckpt-generations 999`**
   で回す。★既定は `--ckpt-generations 2` = **直近 2 世代しか転送しない**。
   既定のままだと「20 世代のうち 18 世代が手元に無い」という事故になる。
2. ディスク逼迫時に落とす順序は ① `indoor_tracks_*`(ON なら)② `llm_journal`
   ③ それでもだめならユーザー判断。**checkpoint と dormant は最後**。
   `rm checkpoint/ckpt-*` を「容量が足りないから」で実行しない。
3. watchdog の「3 世代バックアップ」はノード内のローリング複製であって**世代保全ではない**。
   世代保全は上の 1. のバックアップだけが担う。
4. ラン終了後の撤収でも checkpoint ディレクトリは丸ごと持ち帰る(容量は 8/15 に実測。
   計画値 8.4GB+)。

---

## E3 ★watchdog / backup の本選値(第121 レーンB3・2026-08-15)

**コードの既定値は 1 つも変えていない**(開発用の小ランで回している既存テストを守るため)。
本選は**下のコマンドラインで明示的に上書きする**。根拠と引数の意味は
[scripts/watchdog.py](../scripts/watchdog.py) の冒頭 docstring と `--help` にも同じものがある。

```bash
python scripts/watchdog.py --run-dir runs/finals1 \
  --stall-step-sec <8/15 実測の 1 step 秒> --stall-factor 6 \
  --disk-warn-gb 50 --disk-crit-gb 20 \
  -- run.out_dir=runs run.name=finals1 --profile conf/finals_observe.yaml
```

| 項目 | 既定 | **本選値** | なぜ |
|---|---:|---:|---|
| 停滞判定 | `--stall-min 20`(= 20 分固定) | `--stall-step-sec <実測>` + `--stall-factor 6` | 25万体の 1 step は開発機の数百倍。進捗は checkpoint 粒度でしか見えないので、**1 step も進まないうちに kill する**のが最悪の事故。判定は `max(--stall-min, step_sec × factor)` = **必ず待つ側へ倒れる**(1 step 90 秒なら 20 分側が勝ち、1 step 600 秒なら 60 分側が勝つ) |
| ディスク警告 | `--disk-warn-gb 20` | **50** | E0 で checkpoint / dormant を 1 世代も剪定しない。警告が出た時点で世代コピー数回ぶんの余裕が要る |
| ディスク致命 | `--disk-crit-gb 5` | **20** | 世代バックアップ 1 回ぶん(数 GB)を書き切れる残量で「最後の一声」を出す |
| 世代保全 | `backup_run.py --ckpt-generations 2` | **999** | E0。既定のままだと 20 世代のうち 18 世代が手元に残らない(§E0 の 1.) |

★ `--stall-step-sec` の実測値は **8/15 診断(D1-a と同じラン)の壁時計 ÷ step 数**で足りる。
測っていない場合は引数を渡さなければ従来どおり `--stall-min` だけで動く(退路つき)。

---

## E1-1 speculative decoding(無損失・文献値 最大 ~2.8x)

**狙い**: 目標モデルと同一分布のまま decode を速くする(出力の意味は変えない)。
**未検証**: EAGLE 等の draft は本選機で未疎通。まず ngram draft(モデル追加不要)で疎通 → 次に
独立 draft(qwen3:0.6b 級)や EAGLE ヘッドを比較。draft 選定は本選機の実測で決める。

手順:
1. `launch-vllm-finals.ps1 -Speculative` が表示するコマンドを finals ノードで起動(prefix cache も同時 ON)。
2. **正しさ(無損失)の確認は temperature=0(greedy)で**:
   - spec あり/なしで **同一プロンプト → 出力バイト一致** を確認(greedy なら spec は定義上ロスレス)。
   - ★temperature=0.7 の本番設定では、spec あり/なしでサンプラの乱数消費が変わり **トークン列は
     一致しない**(分布は一致)。よって 0.7 では「バイト一致」を無損失の判定に使わない。
     0.7 の健全性は (a) acceptance rate が想定域、(b) distinct-n・自己反復率がベース比で不変、
     (c) R1 呼数不変、で見る。
3. **速度(before/after)**: 同一ワークロードで tok/s を実測。
   - サーバ側: vLLM の Prometheus `/metrics`(`vllm:*` 系。`avg_generation_throughput_toks_per_s`、
     spec の `spec_decode_*`/acceptance rate)。
   - クライアント側: `scripts/bench.py --lod` は現状 **ollama 直**(eval_count 経由)。vLLM の tok/s は
     上記 `/metrics` か、`--lod` に vLLM 分岐(応答長÷wall の概算 or /metrics 読み)を足して測る
     (概算は「概算」と明記)。
4. 記録: `runs/_bench/` に before/after の tok/s・acceptance rate・出力サンプルを残す。**倍率は実測値のみ記載**。

参考(手元 ollama qwen3:4b の実測 baseline・第24バッチ): decode ~172–176 tok/s(GPU 共有下)、
fallback 0%。本選機・vLLM・8B では別物 → 本選で測り直す。

---

## E1-2 prefix cache + sticky の実機疎通(文献値 ヒット率 ~96%)

**狙い**: ペルソナ+履歴の共通接頭辞の再計算を省く。FleetLLM の sticky 割当(agent_id →
サーバ安定割当)と両輪。配線は実装済み・**未実測**。

手順:
1. `--enable-prefix-caching` 付きで起動(launch スクリプト既定 ON)。
2. `conf/profiles/finals-vllm7.yaml` で 6 体×20 step 程度の配線スモーク。
3. ヒット率を vLLM `/metrics` の `vllm:gpu_prefix_cache_hit_rate`(または hit/query カウンタ)で確認。
   sticky が効いていれば同一エージェントの2回目以降の接頭辞がヒットする。
4. sticky の効きを見るには、tiers/servers を1本 → 複数に増やしてヒット率が保たれるかを比較
   (agent_id 割当が分散しても各エージェントは同じサーバに貼り付く=ヒット率維持が期待挙動)。

注意: 応答キャッシュ(llm_cache.jsonl)とは別物。prefix cache は「初回計算の接頭辞再利用」=
GPU 内 KV の話。llm_cache は「同一呼のバイト再生」=再現性の実体。混同しない。

---

## E1-3 reflect 出力上限の右サイズ化(実測は第24バッチで完了・本選は反映判断のみ)

第24バッチで既存 10 ラン(reflect_think=false)から実測済み。詳細は `scripts/bench.py --analyze-runs`
の出力(`runs/_bench/bench_reflect_sizing.json`)。要点:
- reflect raw 応答 p95=411 / max=459 字(実測 1.855 字/tok で ~247 tok)。深い内省の self/ties を
  足しても ~350 tok。現行 `reflect_max_tokens=2048` は **reflect_think=true 時代の名残**で、
  think=false では約 8x 過大。**推奨 768**(観測 max の ~2.2x 余裕)。
- ★条件付き: think=true に戻すなら思考が予算を食うため 768 では飢える(1600 以上へ)。
- plan_max_tokens=448 / max_tokens=320 は実測上 **妥当**(それぞれ観測 max=543字/367字 < 上限)。
- **効果の性質**: ollama 単一ストリームでは上限は非拘束(モデルが早く自然停止)= 速度は変わらない。
  **vLLM では max_tokens ぶんの KV スロットを予約する**ため、reflect の上限を下げると連続バッチングの
  同時実行スロットが増える → 高並列時のスループット改善に効く。本選機で before/after を実測して確認。

本選での作業: `conf/profiles/finals-vllm7.yaml` の `reflect_max_tokens: 768` で高並列時の
tok/s を 2048 と比較(KV スロット増の効果測定)。conf 本体(config/daily/production)への反映は
主エージェント判断。

---

## E1-4 step 内独立発火の一括発行 seam(R1 制約下で何が可能か・検討メモ)

**目的**: 7 GPU の充填率向上。1 step 内に**互いに独立**に発火する LLM 呼(主に deliberate 系)を
まとめて艦隊へ同時発行し、GPU を遊ばせない。

R1・決定論の制約下で **可能なこと**:
- 1 step 内の deliberate 呼は、プロンプト構築の RNG draw が **呼の前に**完了していて、応答が
  同 step 内の後続の RNG draw に影響しない限り、**発行順序を保ったまま並行実行**できる
  (transport だけ並列化・意味は不変)。呼数も RNG 列も変わらない=R1 維持。
- 応答の**書き戻し(cache 追記・イベント記録)は決定論順序**で行う(agent_id 昇順など固定順)。
  並行で受け取っても適用は固定順=バイト一致を保つ。
- FleetLLM は既に agent_id で sticky 分散するので、集約点で `(prompt, rng_key, params)` の
  タプル列を作って concurrent 発行 → sticky 先へ散る、が素直な実装。

**やってはいけない/できないこと**:
- 応答が同 step 内の**次の呼のプロンプトに入る**依存(逐次)のある呼は束ねられない
  (例: recall→reflect の2段は逐次。deliberate 同士は独立)。
- 束ねることで**呼数や draw 順が変わる**設計は不可(R1 違反)。「崩れたら大モデルへ再送」型の
  動的エスカレーションも呼数変動=不採用(multi-model-lod.md 設計判断2)。
- バッチ化で**キャッシュキーやイベント順序が変わる**なら不可(ゴールデン L1 が壊れる)。

**着手前に確認すべき seam**(コード実測タスク・本選前でも mock で検証可):
1. scheduler の deliberate 発火点(`_arouse`/deliberate 呼)で、応答が同 step 内の RNG を
   引かないことをコードで確認(引くなら独立でない)。
2. 集約点を1つ入れて「順序保存の並行発行 → 固定順で書き戻し」に置き換え、**mock でゴールデン
   L1 バイト一致**を確認(seam が意味を変えないことの証明)。
3. 実 LLM 短ランで tok/s(充填率)before/after を実測。**倍率は実測値のみ**記載。

現状: seam 未実装(src/society は本バッチ変更禁止)。本メモは着手時の設計制約の固定用。
実装は主エージェント(agent-core 担当)が scheduler に集約点を1つ入れる形が素直。

---

## E2 進捗報告(Discord)の運用手順(第116 レーン・2026-08-15)

正典: [progress-reporting-plan.md](../docs/plans/progress-reporting-plan.md)(レーン1+2 を実装)。
実装: [scripts/report_progress.py](../scripts/report_progress.py) / ラッパ: [report-progress.ps1](report-progress.ps1) /
テスト: `tests/test_report_progress.py`。

**性質**: run-dir を**読むだけ**のサイドカー。ラン本体にも watchdog にも 1 行も触らない。
書くのは `<run>/_progress/` 配下だけ・**終了コードは常に 0**・L1 は 1 バイトも読まない。
`live_viewer.py` と併走してよい(どちらも `_open_shared` = 共有読み)。

### E2-0 ★事前準備(ユーザー作業・1 回だけ)

1. Discord で**投稿先チャンネルを 1 本**作る(非公開でよい)。
2. チャンネル設定 → 連携サービス → ウェブフック → 新規作成 → **URL をコピー**。
3. 環境変数へ入れる(**URL はリポジトリにもチャットにも貼らない**):

       [Environment]::SetEnvironmentVariable('SHIBUYA_DISCORD_WEBHOOK','<URL>','User')

   退路: `%USERPROFILE%\.shibuya\discord_webhook.txt`(1 行・リポジトリ外)に置くと
   `ops/report-progress.ps1` が読んで子プロセスの環境変数としてだけ渡す。
4. **失効させたくなったら**: チャンネル設定でウェブフックを削除 → 旧 URL は即死。
   新規発行 → 環境変数を差し替え → レポーターを再起動(`reporter_state.json` の
   `disabled` が `true` なら `false` へ戻す)。

### E2-1 疎通(投稿の前に必ず通す)

    # ① 投稿せず本文だけ確認(runs/ のどのランでもよい。診断ランでも過去ランでもよい)
    python scripts/report_progress.py runs/<run> --dry-run
    #    → <run>/_progress/dryrun/*.json に「送るはずだった JSON」が出る。中身を目視。
    # ② 本番投稿 1 回(環境変数が入っていれば投稿する)
    python scripts/report_progress.py runs/<run>
    # ③ 常駐(Linux/tmux)  または  タスクスケジューラ(Windows)
    python scripts/report_progress.py runs/<run> --interval 900

**★8/15 に確認すること**: GPU 機から `discord.com:443` へ出られるか。
出られない場合は `backup_run.py --dest` で**ローカル PC へ日次 pull した先**を `--run-dir` に
指して**ローカル側から投稿**する(報告が 1 日遅れになるだけで機構は同一)。

### E2-2 出るもの(3 系統)

| 系統 | 形 | 頻度 |
|---|---|---|
| ハートビート | **1 通を編集し続ける**(チャンネルを汚さない) | 既定 10 分ごと(`--heartbeat-min`) |
| 日次ダイジェスト | 新規 embed + `rollup.html` 添付(≤8MB) | **シミュ日の境界**(本選 = 実時間 7〜16 h おき・全 10 通強) |
| アラート | 新規 embed | 状態が**遷移したときだけ**(state / 進捗停止 / ディスク / LLM fallback / 再起動回数 / 退行判定) |

抑制(クールダウン 30 分・毎時 6 通・ヒステリシス)で送らなかった件数は、**次の日次ダイジェストに
必ず数を出す**(silent cap 禁止)。重大(failed / disk critical / 進捗停止)は抑制を跨いで必ず通る。

### E2-3 途中取り出し(ビューアに落とす)

    python scripts/report_progress.py runs/<run> --extract --day 3
    #  → <run>/_progress/day-03/{l2_metrics.parquet, digest.json, summary.json, rollup.html}

- `--day` は **0 始まり**(`make_viewer.py --daily-rollup` / `analyze_structure` と同定義)。
- `digest.json` は **確報**(最新 checkpoint の mtime 以前の part 由来)と **速報**を別の節に
  分けて残す。投稿本文は速報で、必ず「暫定」と確報 step を明記する。
- `day-NN/` の `config.yaml` と `l1_events.parquet` は **report_progress.py が置いた合成物**
  (`make_viewer` に真の start_min / Δt を渡すためだけの足場。同 dir の `_SHIMS.txt` に明記)。
  L1 の中身は 1 バイトも含まない。
- 地図つきのライブ画面が要るときは `scripts/live_viewer.py`(別プロセス・併走可)。

### E2-4 事故時の切り分け

| 症状 | 見るところ |
|---|---|
| 何も投稿されない | `<run>/_progress/reporter.log`(1 行/サイクル)。環境変数未設定なら「dryrun へ書く」と出る |
| 404 で止まった | webhook が削除/再生成された。環境変数を差し替え → `reporter_state.json` の `disabled` を `false` へ |
| 日次が飛んだ | `reporter_state.json` の `posted_days` から該当 day を消せば次サイクルで再投稿 |
| ハートビートが増殖 | `heartbeat_id` が消えている(メッセージが削除された)。実害なし・1 通に収束する |
| ランに影響が出た? | **構造的に出ない**。それでも疑うなら止めてよい(止めてもランは何も変わらない) |
