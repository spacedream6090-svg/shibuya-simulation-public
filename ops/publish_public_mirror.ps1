# publish_public_mirror.ps1 — 公開ミラー同期スクリプト
#
# 役割: ローカルの private リポジトリから「フィルタ済み公開ミラー」を再生成して公開リポへ push する。
#   - 全履歴から公開不可パスを除去(GPL 参照コード reference/2d-fire-sim/・主催メモ docs/AUTOMATA*・
#     他チーム分析 docs/research/hackathon1-analysis/)
#   - コミット作者/コミッタのメールアドレスを GitHub noreply へ書換(公開側のみ。ローカルは不変)
#   - ローカルの作業フォルダー・private リポは一切変更しない(一時 clone 上でのみ書換)
#
# 前提: pip install git-filter-repo 済み・公開先リポが GitHub 上に存在すること。
# git-filter-repo は決定論的(同一入力→同一ハッシュ)なので、2 回目以降の実行は fast-forward push になる。
# non-fast-forward で拒否された場合は private 側の履歴が書き換わっている(要調査。安易に -Force しない)。
#
# 使い方:
#   pwsh -File ops/publish_public_mirror.ps1                # フィルタ→push→後片付け
#   pwsh -File ops/publish_public_mirror.ps1 -NoPush        # フィルタまで(検証用に一時 clone を残す)

param(
    [string]$PublicRepo = "spacedream6090-svg/shibuya-simulation-public",
    [string]$Branch = "main",
    [switch]$NoPush,
    # 除外セット変更(2026-08-06 PUB-U1 決定)後の初回のみ必要。公開側の履歴を意図して書き換える。
    [switch]$ForcePush
)
$ErrorActionPreference = "Stop"

$src = Split-Path -Parent $PSScriptRoot   # リポジトリルート(ops/ の親)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "shibuya-public-mirror-$stamp"

# 公開側の作者情報(名前は維持・メールのみ noreply へ)。個人メールはスクリプトに書かず git から導出する。
$oldEmail = (git -C $src log -1 --format=%ae).Trim()
$oldName  = (git -C $src log -1 --format=%an).Trim()
$noreply  = "277403140+spacedream6090-svg@users.noreply.github.com"

git clone --no-local --branch $Branch $src $work
if ($LASTEXITCODE -ne 0) { throw "clone failed" }

$mailmap = Join-Path ([System.IO.Path]::GetTempPath()) "shibuya-mailmap-$stamp.txt"
Set-Content -Path $mailmap -Value "$oldName <$noreply> <$oldEmail>" -Encoding UTF8

Push-Location $work
try {
    # 除去パス(2026-08-06 PUB-U1 決定で拡張):
    #   - reference/2d-fire-sim: GPL-3.0 参照コード(本体未使用)
    #   - docs/ 全体: 内部設計・研究ノート(主催メモ docs/AUTOMATA*・他チーム分析を包含)
    #   - STATUS/IMPLEMENTED/PENDING: 内部台帳(公開に必要な .md は README/ETHICS のみ)
    #   - data/ 全体: ライセンス地雷2件を包含(商業施設/区サイト由来=転載不可・OSM 由来テーブル=ODbL
    #     share-alike が配布時発動)。安全側で全除外(dt-snapshot-integration-proposal §4)
    #   - env/ 全体: shimokita に OSM 生データ(_osm_raw.json)・shibuya も OSM/ODPT 由来値を参照
    # 残すもの: README.md / ETHICS.md / LICENSE / src / scripts / tests / conf / viz / ops / tools /
    #   reference/physics_bench(Jülich 派生 CSV は CC BY 4.0=README に帰属表示あり=再配布可)
    python -m git_filter_repo --invert-paths `
        --path reference/2d-fire-sim `
        --path docs `
        --path data `
        --path env `
        --path STATUS.md `
        --path IMPLEMENTED.md `
        --path PENDING.md `
        --mailmap $mailmap
    if ($LASTEXITCODE -ne 0) { throw "git filter-repo failed" }

    # 検証(push 前の機械チェック): 除去パスの残存ゼロ・旧メールの残存ゼロ
    # 注意: '^reference/' で引くと physics_bench(残すもの)を誤検知する(2026-08-06 に実バグとして修正)。
    $leftPaths = (git log --all --name-only --format= ) -match '^(reference/2d-fire-sim|docs/|data/|env/|STATUS\.md|IMPLEMENTED\.md|PENDING\.md)'
    if ($leftPaths) { throw "excluded paths still present in mirror history: $($leftPaths -join ', ')" }
    $emails = (git log --all --format='%ae%n%ce' | Sort-Object -Unique)
    if ($emails -contains $oldEmail) { throw "old author email still present in mirror history" }

    if ($NoPush) {
        Write-Host "NoPush: mirror left for inspection at $work"
    } else {
        git remote add public "https://github.com/$PublicRepo.git"
        if ($ForcePush) {
            git push --force public "${Branch}:${Branch}"
        } else {
            git push public "${Branch}:${Branch}"
        }
        if ($LASTEXITCODE -ne 0) { throw "push failed (non-fast-forward なら private 履歴の書換を疑う。除外セット変更後の初回は -ForcePush)" }
        Write-Host "published -> https://github.com/$PublicRepo"
    }
}
finally {
    Pop-Location
    Remove-Item -Force $mailmap -ErrorAction SilentlyContinue
    if (-not $NoPush) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
}
