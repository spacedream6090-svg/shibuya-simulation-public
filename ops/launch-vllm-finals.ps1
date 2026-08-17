# =============================================================================
# launch-vllm-finals.ps1 — 本選 GPU 7枚での vLLM 艦隊 起動「手順書」(第24バッチ A5 / M4)
#
# ★これは手順書スクリプト。既定は **dry-run(コマンドを表示するだけ・何も起動しない)**。
#   実際に起動するのは -Execute を明示したときだけ。本選ノードは Linux 想定なので、通常は
#   本スクリプトが表示する各コマンドを finals ノードのシェルへ貼って使う(Windows ローカルに
#   vLLM が入っていれば -Execute で Start-Process 起動も可=あくまで疎通テスト用)。
#
# トポロジ(第139 MIX-1 = 混合モデル艦隊 5+2。正典 docs/plans/model-mix-plan.md):
#   GPU0-4 → port 8000-8004 = Qwen3-8B-AWQ ×5(会話 = default tier)
#   GPU5-6 → port 8005-8006 = Qwen3-14B-AWQ ×2(思考 = reflect / plan tier)
#   conf/profiles/finals-vllm7.yaml の servers/tiers と **1:1 対応**(ここを変えたら必ず両方変える。
#   8005/8006 に 14B を起動していないのに conf が宣言していると、思考呼が 8B を叩き続ける)。
#   FleetLLM(finals-vllm7.yaml)が agent_id で sticky 割当 → prefix cache が効く。
#   ★A8 実測(2026-08-17): 本選経路は api_mode: chat(第138)。5+2 は MIX-3 リハーサルで確定する
#     暫定値(縮退線: 思考タイア滞留>1step or T10>140h → 6+1 へ)。
#
# 回収する「タダ飯」(E1・compute-optimization.md):
#   --enable-prefix-caching     : ペルソナ+履歴の共通接頭辞を再利用(sticky と両輪。無損失)。
#   --speculative-config        : speculative decoding。★MIX リサーチ(2026-08-17)= 高並列では
#                                 逆効果(EAGLE は batch=1 の1.96倍 → batch=128 で1.21倍・32+並列で
#                                 標準より遅い報告)+ AWQ 併用はバグ帯 → **本選は使わない**。
#                                 低並列レイテンシ用途が生じたときのみ AngelSlim EAGLE3 を実機検証。
#
# -----------------------------------------------------------------------------
# β10 モデル・サンプリングの完全凍結(第117バッチ 2026-08-16)
#   正典: docs/plans/beta-implementation-plan.md §1 β10 / external-audit-triage.md F15。
#
#   ① --generation-config vllm(★第117で追加)
#      vLLM の既定は `--generation-config auto` = **モデルリポジトリ同梱の
#      generation_config.json をサンプリング既定として採用する**。つまり未指定の
#      temperature / top_p / top_k / repetition_penalty が「そのモデルの作者が置いた値」に
#      なり、モデルを差し替えると**こちらの conf を 1 文字も変えていないのに分布が変わる**。
#      `vllm` を指定すると vLLM 自身の中立な既定に固定され、実際に効くのは
#      「リクエストで明示した値」だけになる = 実験条件が起動側で閉じる。
#
#   ② モデル表記は **HF repo id + revision 固定**(下の $Fleet 表に埋め込み済み)
#      revision は GPU 機での A8 実測検収に使った実体と同一。★revision を省くと HF 側の main が
#      動いた瞬間に「同じ名前で違う重み」になる。served-model-name(別名)は conf の
#      name / tiers.model と一致させること。run_manifest の launch 欄へ同じ文字列を書き写す。
#      サイズ: Qwen3-8B-AWQ 約6.11GB / Qwen3-14B-AWQ 約9.99GB(A5000 24GB)。
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
#   powershell -NoProfile -File ops/launch-vllm-finals.ps1 -Execute       # ローカル疎通(要 vLLM)
#
# 起動後の検収(別ターミナル):
#   python scripts/check_llm_backends.py --backend openai_compat --base-url http://localhost:8000/v1 --model qwen3:8b
#   python scripts/check_llm_backends.py --backend openai_compat --base-url http://localhost:8005/v1 --model qwen3:14b
#   python scripts/run.py --profile conf/profiles/finals-vllm7.yaml run.n_agents=6 run.n_steps=20  # 配線スモーク
# =============================================================================

param(
    [int]    $BasePort      = 8000,
    [double] $GpuMemUtil    = 0.90,
    [int]    $MaxModelLen   = 8192,          # KV 予算。reflect_max_tokens=768 でも接頭辞が長いので余裕を持たせる
    [switch] $EnablePrefixCache = $true,     # 既定 ON(タダ飯)
    [switch] $Execute                        # 付けたときだけ実際に起動(既定は表示のみ)
)

$ErrorActionPreference = "Stop"

# MIX-1 混合艦隊 5+2(第139)。conf/profiles/finals-vllm7.yaml の tiers と 1:1 対応。
# 8B へ戻す(6+1 縮退等)ときは該当行の Repo/Rev/Name を 8B の値へ書き換え、conf 側の tiers も直す。
$Fleet = @(
    @{ Gpu = 0; Repo = "Qwen/Qwen3-8B-AWQ";  Rev = "4da05a8edb55c6046cce958586c33b61da07bb79"; Name = "qwen3:8b"  },
    @{ Gpu = 1; Repo = "Qwen/Qwen3-8B-AWQ";  Rev = "4da05a8edb55c6046cce958586c33b61da07bb79"; Name = "qwen3:8b"  },
    @{ Gpu = 2; Repo = "Qwen/Qwen3-8B-AWQ";  Rev = "4da05a8edb55c6046cce958586c33b61da07bb79"; Name = "qwen3:8b"  },
    @{ Gpu = 3; Repo = "Qwen/Qwen3-8B-AWQ";  Rev = "4da05a8edb55c6046cce958586c33b61da07bb79"; Name = "qwen3:8b"  },
    @{ Gpu = 4; Repo = "Qwen/Qwen3-8B-AWQ";  Rev = "4da05a8edb55c6046cce958586c33b61da07bb79"; Name = "qwen3:8b"  },
    @{ Gpu = 5; Repo = "Qwen/Qwen3-14B-AWQ"; Rev = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"; Name = "qwen3:14b" },
    @{ Gpu = 6; Repo = "Qwen/Qwen3-14B-AWQ"; Rev = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"; Name = "qwen3:14b" }
)

Write-Host "=== vLLM finals 起動プラン(MIX-1 混合艦隊 5+2 / port $BasePort-$($BasePort+$Fleet.Count-1)) ===" -ForegroundColor Cyan
Write-Host "gpu_mem_util=$GpuMemUtil  max_model_len=$MaxModelLen  prefix_cache=$EnablePrefixCache  speculative=不採用(MIXリサーチ)"
if (-not $Execute) {
    Write-Host "[dry-run] -Execute 未指定 = コマンドを表示するだけで起動しません。" -ForegroundColor Yellow
}
Write-Host ""

foreach ($srv in $Fleet) {
    $gpu  = $srv.Gpu
    $port = $BasePort + $gpu

    # 共通フラグ(--revision = β10 実体固定・--served-model-name = conf の name/tiers.model と一致)
    $flags = @(
        "--revision $($srv.Rev)",
        "--served-model-name $($srv.Name)",
        "--port $port",
        "--gpu-memory-utilization $GpuMemUtil",
        "--max-model-len $MaxModelLen",
        # β10: サンプリング既定をモデル同梱の generation_config.json ではなく vLLM 側へ固定する
        # (auto のままだとモデル差し替えで未指定パラメータが黙って変わる。上部の注記 ① を参照)。
        "--generation-config vllm"
    )
    if ($EnablePrefixCache) { $flags += "--enable-prefix-caching" }
    $flagStr = ($flags -join " ")

    # 本選ノード(Linux)向けの1行(GPU 固定 + バックグラウンド + ログ)
    $linuxCmd = "CUDA_VISIBLE_DEVICES=$gpu vllm serve $($srv.Repo) $flagStr > vllm_gpu$gpu.log 2>&1 &"
    Write-Host "# GPU$gpu -> port $port  [$($srv.Name)]" -ForegroundColor Green
    Write-Host "  $linuxCmd"

    if ($Execute) {
        # Windows ローカルでの疎通テスト起動(vLLM が入っていれば)。GPU 固定は環境変数で。
        $env:CUDA_VISIBLE_DEVICES = "$gpu"
        Start-Process -FilePath "vllm" `
            -ArgumentList "serve $($srv.Repo) $flagStr" `
            -RedirectStandardOutput "vllm_gpu$gpu.log" `
            -RedirectStandardError  "vllm_gpu$gpu.err.log" `
            -NoNewWindow
        Write-Host "  [execute] started (log: vllm_gpu$gpu.log)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "次の手順: ops/finals-compute-checklist.md(prefix cache ヒット率確認・MIX-3 リハーサルで 5+2 確定)"
if (-not $Execute) { Write-Host "(何も起動していません=dry-run)" -ForegroundColor Yellow }
