# =============================================================================
# backup-daily.ps1 — 渋谷シミュレーションの日次バックアップ(第28バッチ 2026-07-13)
#
# シミュレーション実装(src/scripts)とは独立した運用スクリプト。
# タスクスケジューラ「shibuya-simulation-daily-backup」が毎日 23:30 に実行する
# (その時刻にPCが起きていなければ、次に可能になったタイミングで実行)。
#
# 出力先: C:\Users\塚本翔太\Desktop\shibuya バックアップ\
#   mirror\                        リポジトリ全体の増分ミラー(robocopy /MIR。runs・.git 含む完全鏡像。
#                                  初回のみ全量コピー・以降は変更分だけなので速い)
#   snapshots\code-YYYY-MM-DD.zip  コード+docs+conf+data の日付つきスナップショット
#                                  (runs/ と .git を除く軽量版。直近14日分を保持・古いものは自動削除)
#   backup-log.txt                 実行ログ(1行/回。失敗時は MIRROR-ERROR と記録される)
#
# 手動実行:   powershell -NoProfile -ExecutionPolicy Bypass -File "この ファイル"
# 時刻変更:   タスクスケジューラで shibuya-simulation-daily-backup のトリガーを編集
# =============================================================================

$ErrorActionPreference = "Continue"
$src     = "C:\Users\塚本翔太\Desktop\shibuya-simulation"
$dest    = "C:\Users\塚本翔太\Desktop\shibuya バックアップ"
$mirror  = Join-Path $dest "mirror"
$snapDir = Join-Path $dest "snapshots"
$log     = Join-Path $dest "backup-log.txt"
$date    = Get-Date -Format "yyyy-MM-dd"
$t0      = Get-Date

New-Item -ItemType Directory -Force -Path $mirror, $snapDir | Out-Null

# ---- 1) 増分ミラー(完全鏡像・削除も同期。キャッシュ類のみ除外) ----
robocopy $src $mirror /MIR /R:2 /W:5 /XD "__pycache__" ".pytest_cache" /NFL /NDL /NP /NJH /NJS | Out-Null
$rc = $LASTEXITCODE   # robocopy: 0-7 = 成功系(コピー実施含む) / 8以上 = 失敗

# ---- 2) 日付つきコードスナップショット(runs/.git 除外=軽量) ----
$zipOk = $true
try {
    $tmp = Join-Path $env:TEMP "shibuya-snap"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    robocopy $src $tmp /E /R:2 /W:5 /XD "runs" ".git" "__pycache__" ".pytest_cache" /NFL /NDL /NP /NJH /NJS | Out-Null
    $zip = Join-Path $snapDir "code-$date.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }   # 同日再実行は上書き
    Compress-Archive -Path (Join-Path $tmp "*") -DestinationPath $zip -Force
    Remove-Item $tmp -Recurse -Force
} catch {
    $zipOk = $false
}

# ---- 3) 14日より古いスナップショットを削除 ----
Get-ChildItem $snapDir -Filter "code-*.zip" |
    Sort-Object Name -Descending | Select-Object -Skip 14 |
    Remove-Item -Force -ErrorAction SilentlyContinue

# ---- 4) ログ(1行/回) ----
$sec = [int]((Get-Date) - $t0).TotalSeconds
if ($rc -ge 8)       { $status = "MIRROR-ERROR(rc=$rc)" }
elseif (-not $zipOk) { $status = "ZIP-ERROR(mirror=OK)" }
else                 { $status = "OK" }
$zipInfo = ""
if ($zipOk -and (Test-Path $zip)) {
    $zipInfo = " zip={0:N1}MB" -f ((Get-Item $zip).Length / 1MB)
}
Add-Content -Path $log -Encoding UTF8 -Value ("{0} {1} {2}{3} {4}s" -f $date, (Get-Date -Format "HH:mm:ss"), $status, $zipInfo, $sec)

if ($rc -ge 8) { exit 1 } else { exit 0 }
