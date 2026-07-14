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
