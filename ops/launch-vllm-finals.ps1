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
        "--max-model-len $MaxModelLen"
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
Write-Host "次の手順: ops/finals-compute-checklist.md(speculative の before/after 実測・prefix cache ヒット率確認)"
if (-not $Execute) { Write-Host "(何も起動していません=dry-run)" -ForegroundColor Yellow }
