# =============================================================================
# launch-vllm-finals.ps1 — 本選 GPU 7枚での vLLM 艦隊 起動「手順書」(第24バッチ A5 / M4)
#
# ★これは手順書スクリプト。既定は **dry-run(コマンドを表示するだけ・何も起動しない)**。
#   実際に起動するのは -Execute を明示したときだけ。本選ノードは Linux 想定なので、通常は
#   本スクリプトが表示する各コマンドを finals ノードのシェルへ貼って使う(Windows ローカルに
#   vLLM が入っていれば -Execute で Start-Process 起動も可=あくまで疎通テスト用)。
#
# トポロジ(ops topology A = 単一データセンターノードに 7 GPU):
#   GPU0→port8000, GPU1→8001, ... GPU6→8006 に vLLM を1本ずつ固定。
#   FleetLLM(conf/profiles/finals-vllm7.yaml)が agent_id で sticky 割当 → prefix cache が効く。
#
# 回収する「タダ飯」(E1・compute-optimization.md):
#   --enable-prefix-caching     : ペルソナ+履歴の共通接頭辞を再利用(sticky と両輪。無損失)。
#   --speculative-config        : speculative decoding。目標モデルと同一分布=無損失で最大~2.8倍。
#                                 ★EAGLE 等の draft は本選機で未検証 → 起動疎通 + ゴールデン一致 +
#                                   tok/s before/after を必ず実測してから本番採用(数値は約束しない)。
#
# -----------------------------------------------------------------------------
# β10 モデル・サンプリングの完全凍結(第117バッチ 2026-08-16)
#   正典: docs/plans/beta-implementation-plan.md §1 β10 / external-audit-triage.md F15。
#
#   ① --generation-config vllm(★本バッチで追加)
#      vLLM の既定は `--generation-config auto` = **モデルリポジトリ同梱の
#      generation_config.json をサンプリング既定として採用する**。つまり未指定の
#      temperature / top_p / top_k / repetition_penalty が「そのモデルの作者が置いた値」に
#      なり、モデルを差し替えると**こちらの conf を 1 文字も変えていないのに分布が変わる**。
#      `vllm` を指定すると vLLM 自身の中立な既定に固定され、実際に効くのは
#      「リクエストで明示した値」だけになる = 実験条件が起動側で閉じる。
#
#   ② モデル表記は **HF repo id + revision 固定を推奨**(第一候補 Qwen/Qwen3-8B-AWQ)
#      `-Model qwen3:8b` のような ollama 風タグは vLLM では「served-model-name(別名)」に
#      しかならず、**実体がどの重みか**を後から確定できない。本選では
#        vllm serve Qwen/Qwen3-8B-AWQ --revision <commit sha> --served-model-name qwen3-8b ...
#      の形にして、run_manifest の launch 欄(起動側申告)へ同じ文字列を書き写すこと。
#      候補サイズ: Qwen3-8B-AWQ 約6.11GB / Qwen3-14B-AWQ 約9.99GB(A5000 24GB × 7 枚)。
#      ★revision を省くと HF 側の main が動いた瞬間に「同じ名前で違う重み」になる。
#
#   ③ sampling は**全部明示**する(client 側 = conf/finals 側の責務)
#        temperature … conf `model.temperature`(既定 0.7)= 送出ボディに必ず載る
#        max_tokens  … conf `model.max_tokens` / `model.reflect_max_tokens` = 必ず載る
#        top_p/top_k … ★現状 **client は送っていない** = ① の既定に従う。値を主張したい
#                       なら vllm.py の body へ足す(送っていない値を manifest に書かない)。
#        seed        … conf `llm.request_seed.enabled=true`(β11)で 1 リクエストごとに
#                       blake2b(run_seed, agent_id, step, purpose, ordinal) を送る。
#                       OFF ではサーバ側のグローバル乱数 = 同じ入力でも応答が揺れる。
# -----------------------------------------------------------------------------
#
# 使い方:
#   powershell -NoProfile -File ops/launch-vllm-finals.ps1                 # dry-run(表示のみ)
#   powershell -NoProfile -File ops/launch-vllm-finals.ps1 -Model qwen3:8b -Speculative
#   powershell -NoProfile -File ops/launch-vllm-finals.ps1 -Execute       # ローカル疎通(要 vLLM)
#
# 起動後の検収(別ターミナル):
#   python scripts/bench.py --lod --backend ... (vLLM 経路の tok/s)          ※現状 --lod は ollama 直
#   python scripts/check_llm_backends.py --backend openai_compat --base-url http://localhost:8000/v1 --model qwen3:8b
#   python scripts/run.py --profile conf/profiles/finals-vllm7.yaml run.n_agents=6 run.n_steps=20  # 配線スモーク
# =============================================================================

param(
    [int]    $NumGpus       = 7,
    [int]    $BasePort      = 8000,
    [string] $Model         = "qwen3:8b",
    [double] $GpuMemUtil    = 0.90,
    [int]    $MaxModelLen   = 8192,          # KV 予算。reflect_max_tokens=768 でも接頭辞が長いので余裕を持たせる
    [switch] $Speculative,                   # --speculative-config を付ける(未検証=疎通確認前提)
    [string] $DraftModel    = "qwen3:0.6b",  # speculative の draft(小モデル。EAGLE ヘッドなら別指定)
    [int]    $NumSpecTokens = 3,
    [switch] $EnablePrefixCache = $true,     # 既定 ON(タダ飯)
    [switch] $Execute                        # 付けたときだけ実際に起動(既定は表示のみ)
)

$ErrorActionPreference = "Stop"

Write-Host "=== vLLM finals 起動プラン ($NumGpus GPU / port $BasePort-$($BasePort+$NumGpus-1)) ===" -ForegroundColor Cyan
Write-Host "model=$Model  gpu_mem_util=$GpuMemUtil  max_model_len=$MaxModelLen  prefix_cache=$EnablePrefixCache  speculative=$Speculative"
if (-not $Execute) {
    Write-Host "[dry-run] -Execute 未指定 = コマンドを表示するだけで起動しません。" -ForegroundColor Yellow
}
Write-Host ""

for ($i = 0; $i -lt $NumGpus; $i++) {
    $port = $BasePort + $i
    $gpu  = $i

    # 共通フラグ
    $flags = @(
        "--port $port",
        "--served-model-name $Model",
        "--gpu-memory-utilization $GpuMemUtil",
        "--max-model-len $MaxModelLen",
        # β10: サンプリング既定をモデル同梱の generation_config.json ではなく vLLM 側へ固定する
        # (auto のままだとモデル差し替えで未指定パラメータが黙って変わる。上部の注記 ① を参照)。
        "--generation-config vllm"
    )
    if ($EnablePrefixCache) { $flags += "--enable-prefix-caching" }
    if ($Speculative) {
        # ★vLLM のバージョンでフラグ形が異なる(新: --speculative-config JSON / 旧: --speculative-model 等)。
        #   下は新形の例。EAGLE を使うなら method を "eagle" にし draft を EAGLE ヘッドへ差し替える。未検証。
        $spec = '{""method"":""ngram"",""num_speculative_tokens"":' + $NumSpecTokens + '}'
        # draft モデル併用(独立 draft の例。EAGLE では不要):
        # $spec = '{""method"":""eagle"",""model"":""' + $DraftModel + '"",""num_speculative_tokens"":' + $NumSpecTokens + '}'
        $flags += "--speculative-config `"$spec`""
    }
    $flagStr = ($flags -join " ")

    # 本選ノード(Linux)向けの1行(GPU 固定 + バックグラウンド + ログ)
    $linuxCmd = "CUDA_VISIBLE_DEVICES=$gpu vllm serve $Model $flagStr > vllm_gpu$gpu.log 2>&1 &"
    Write-Host "# GPU$gpu -> port $port" -ForegroundColor Green
    Write-Host "  $linuxCmd"

    if ($Execute) {
        # Windows ローカルでの疎通テスト起動(vLLM が入っていれば)。GPU 固定は環境変数で。
        $env:CUDA_VISIBLE_DEVICES = "$gpu"
        Start-Process -FilePath "vllm" `
            -ArgumentList "serve $Model $flagStr" `
            -RedirectStandardOutput "vllm_gpu$gpu.log" `
            -RedirectStandardError  "vllm_gpu$gpu.err.log" `
            -NoNewWindow
        Write-Host "  [execute] started (log: vllm_gpu$gpu.log)" -ForegroundColor DarkGray
    }
}

Write-Host ""
if ($Model -notmatch "/") {
    # β10: HF repo id(`org/repo`)でない = 実体の重みが事後に確定できない表記。
    Write-Host "[β10 注意] -Model '$Model' は HF repo id ではありません。本選は" -ForegroundColor Yellow
    Write-Host "          'Qwen/Qwen3-8B-AWQ' + --revision <commit sha> で起動し、" -ForegroundColor Yellow
    Write-Host "          別名は --served-model-name で付けること(冒頭の注記 ② を参照)。" -ForegroundColor Yellow
}
Write-Host "次の手順: ops/finals-compute-checklist.md(speculative の before/after 実測・prefix cache ヒット率確認)"
if (-not $Execute) { Write-Host "(何も起動していません=dry-run)" -ForegroundColor Yellow }
