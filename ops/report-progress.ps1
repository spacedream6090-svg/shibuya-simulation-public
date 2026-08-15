# =============================================================================
# report-progress.ps1 — 進捗報告サイドカーの実行ラッパ(第116 レーン: Discord 進捗報告)
#
# 正典: docs/plans/progress-reporting-plan.md / 実装: scripts/report_progress.py
#
# シミュレーション本体とは**完全に独立**した運用スクリプト。run-dir を読むだけで、
# 走行中のランへ read も write も一切しない(書くのは <run>/_progress/ 配下だけ)。
# 失敗しても必ず exit 0 で帰る(観測がランを殺してはならない)。
#
# 使い方(手動):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ops\report-progress.ps1 `
#       -RunDir "C:\...\runs\finals" -DryRun
#
# 使い方(タスクスケジューラ = 15 分ごと。★下の「登録」を参照):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ops\report-progress.ps1 `
#       -RunDir "C:\...\runs\finals"
#
# ★webhook URL の渡し方(**3 通りだけ**。どれも引数では渡さない)
#   1. ユーザー環境変数 SHIBUYA_DISCORD_WEBHOOK(推奨)
#        [Environment]::SetEnvironmentVariable('SHIBUYA_DISCORD_WEBHOOK','<URL>','User')
#      設定後は**新しい** PowerShell / タスクを起動しないと反映されない。
#   2. リポジトリ**外**のファイル %USERPROFILE%\.shibuya\discord_webhook.txt(1 行)
#      本スクリプトが読んで、子プロセスの環境変数としてだけ渡す(引数には載せない)。
#   3. 何も無ければ **投稿せず** <run>\_progress\dryrun\*.json に本文を書くだけ(安全側)。
#   URL を -RunDir のような引数で渡す口は**意図的に用意していない**
#   (引数はシェル履歴・タスクスケジューラの XML・タスクマネージャの一覧に残るため)。
#
# 登録(1 回だけ。15 分ごと・PC が寝ていたら次に可能なときに実行):
#   $act = New-ScheduledTaskAction -Execute "powershell.exe" `
#       -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Users\<you>\Desktop\shibuya-simulation\ops\report-progress.ps1" -RunDir "C:\...\runs\finals"'
#   $trg = New-ScheduledTaskTrigger -Once -At (Get-Date) `
#       -RepetitionInterval (New-TimeSpan -Minutes 15)
#   $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
#   Register-ScheduledTask -TaskName "shibuya-report-progress" -Action $act `
#       -Trigger $trg -Settings $set -Description "渋谷シム 進捗報告(読むだけ)"
#   解除:  Unregister-ScheduledTask -TaskName "shibuya-report-progress" -Confirm:$false
#
# Linux(本選機が Linux の場合)は systemd timer か tmux で同じことをする:
#   while true; do python scripts/report_progress.py "$RUN" --quiet; sleep 900; done
#   あるいは常駐 1 本:  python scripts/report_progress.py "$RUN" --interval 900
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$OutDir = "",
    [string]$Python = "python",
    [switch]$DryRun,          # 投稿せず本文をローカルへ(★初回はこれで確認する)
    [switch]$Extract,         # 取り出しのみ(投稿しない)
    [int]$Day = -1,           # --extract と併用する day(0 始まり)
    [string[]]$Extra = @()    # そのまま report_progress.py へ渡す追加オプション
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repo "scripts\report_progress.py"
$logDir = if ($OutDir) { $OutDir } else { Join-Path $RunDir "_progress" }
$log = Join-Path $logDir "report-progress-task.log"
$t0 = Get-Date

# ---- 1) webhook: 環境変数 → リポジトリ外のファイル → 無ければ dry-run 相当 ----
$mode = "env"
if (-not $env:SHIBUYA_DISCORD_WEBHOOK) {
    $secret = Join-Path $env:USERPROFILE ".shibuya\discord_webhook.txt"
    if (Test-Path $secret) {
        $url = (Get-Content $secret -TotalCount 1).Trim()
        if ($url) { $env:SHIBUYA_DISCORD_WEBHOOK = $url; $mode = "file" }
        Remove-Variable url -ErrorAction SilentlyContinue   # 変数に残さない
    } else {
        $mode = "none(投稿せずローカルへ書くだけ)"
    }
}

# ---- 2) 実行(★URL は引数に載せない。環境変数だけで渡る) ----
$argsList = @($script, $RunDir)
if ($OutDir)  { $argsList += @("--out-dir", $OutDir) }
if ($DryRun)  { $argsList += "--dry-run" }
if ($Extract) { $argsList += "--extract"; if ($Day -ge 0) { $argsList += @("--day", "$Day") } }
$argsList += $Extra

$env:PYTHONIOENCODING = "utf-8"
try {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
} catch { }

$out = & $Python @argsList 2>&1
$rc = $LASTEXITCODE

# ---- 3) 1 行ログ(★URL は絶対に書かない) ----
$sec = [int]((Get-Date) - $t0).TotalSeconds
$tail = ""
if ($out) {
    # レポーター自身の 1 行ログ(cycle / 例外 / extract)を優先して拾う。
    # --dry-run では送信本文の JSON も標準出力に出るので、末尾 1 行では役に立たない。
    $pick = $out | Where-Object { $_ -match 'cycle #|例外|extract |恒久停止|投稿失敗' } |
        Select-Object -Last 1
    if (-not $pick) { $pick = $out | Select-Object -Last 1 }
    $tail = "$pick" -replace '\s+', ' '
}
$tail = $tail -replace 'https?://\S*discord\S*', '<webhook>'
$line = "{0} rc={1} webhook={2} {3}s {4}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $rc, $mode, $sec, $tail
try { Add-Content -Path $log -Encoding UTF8 -Value $line } catch { }

# 観測がランを殺してはならない: 何があっても 0 で帰る
exit 0
