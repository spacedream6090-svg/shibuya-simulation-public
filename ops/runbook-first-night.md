# 初夜ランブック — GPU機セットアップ後、今夜「実際に動かす」までの一本道

> 2026-08-16作成。前提: GPU機(A5000級×7・単一ノード・Linux想定)のOSセットアップ済み。
> ゴール: **今夜、実LLMで最初のランを完走させる**(段階スケールアップの1段目)。
> 参照: [launch-vllm-finals.ps1](launch-vllm-finals.ps1)(vLLM起動手順書)・[finals-compute-checklist.md](finals-compute-checklist.md)(E0剪定禁止/E1最適化/E2 Discord)・[decision-dashboard.md](../docs/plans/decision-dashboard.md)(D1 fire判定)。
> 各Phaseの結果(数字)をこちらへ貼ってもらえれば、fire GO/NO-GO・POP ON等の判定は私(Fable)が即応で行う。

---

## Phase 0 — 搬入と環境(見積り 60-90分)

### 0-1. リポジトリと Python
```bash
git clone <private-repo-url> shibuya-simulation && cd shibuya-simulation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 無ければ pyproject/README の依存節
export PYTHONIOENCODING=utf-8
```

### 0-2. gitignore対象データの搬入(★リポに入っていない3点)
| データ | 方法 |
|---|---|
| `data/persona_pool/`(v1・100万) | **推奨=決定論リビルド**(seed同一なら local と同一): `python scripts/build_persona_pool.py --help` で引数確認→第108と同じ seed/台帳で実行。**照合は G1**: ラン後の `run_manifest.json` の `inputs` sha256 がローカルPCのランと一致すること |
| `data/persona_pool_v2/` | 今夜は不要(v2切替は Phase 4)。作るなら `--v2 --childcare` を追加 |
| `runs/` 既存成果物・`data/realworld/` | 不要(本選ランはgenerated天候・比較解析はローカルPCで) |

### 0-3. vLLM 艦隊(7本)
```bash
# ローカルPCで dry-run 表示 → 出てきたコマンド7本をGPU機シェルへ貼る
powershell -NoProfile -File ops/launch-vllm-finals.ps1        # (ローカルで表示のみ)
# GPU機で7本起動後、疎通:
python scripts/check_llm_backends.py --backend openai_compat --base-url http://localhost:8000/v1 --model <model>
# (8000〜8006 全ポート)
```
※ speculative/prefix-cache は E1 手順(greedy でバイト一致→採否)。**今夜は prefix cache のみONで可**(specは時間があれば)。

### 0-4. 環境確認5点+運用4点(8/16監査反映)
1. `nvidia-smi`: 7 GPU 認識・VRAM
2. ディスク空き: **今夜は50GB以上あれば可**(本番10日ランは~70GB+バックアップ先。E0=checkpoint剪定禁止を忘れない)
3. `date`(機の時計。★本選 conf の `world.calendar.start_date` は **`2026-08-22` 固定**にしたので、
   ラン内の暦は機の日付に**依らない**。今夜の縮小ランで `--profile conf/finals_observe.yaml` を
   使うと初日が 8/22 = **土曜**になる点だけ意識する。`auto` を使う他 profile では実日付を拾う)
4. `curl -sI https://discord.com -o /dev/null -w "%{http_code}\n"`(Discord疎通・任意)
5. ODPT キーは**GPU機では不要**(RW取得はローカルPCのタスクのみ)
6. **`ulimit -n` を上げる**(実測1024=本番不足): ランを張る tmux シェルで `ulimit -n 65535`(vLLM7本+parquet+sockets)
7. `nvidia-smi topo -m` と `numactl --hardware` の出力を保存(2 NUMA機・偏りが大きい時だけ affinity を検討=盲目的pinningはしない)
8. vLLM `/metrics` を全7ポートで30〜60秒間隔保存(prefix hit・queue・KV使用率・TTFT→run artifactへ)
9. **持続試験**: 30〜60分連続推論で req/s のドリフト(熱/クロック)を確認——最初35で1時間後25なら壁時計見積りが壊れる

---

## Phase 1 — 配線スモーク→実測スモーク(見積り 60分)

### 1-1. 配線スモーク(5分)
```bash
python scripts/run.py --profile conf/profiles/finals-vllm7.yaml \
  run.n_agents=6 pool.present_cap=6 run.n_steps=20
```
完走+`l1b_llm.parquet` に実呼が出ればOK。
(このプロファイルは `model:` ブロックだけ = `pool.enabled` は基底の false なので `present_cap` は
不活性だが、**規模指定は常に対で書く**運用にしておく=打ち間違いの型を作らない。)

### 1-2. 実測スモーク(★今夜の最重要計測・約30-40分)
```bash
python scripts/run.py --profile conf/finals_observe.yaml \
  run.seed=42 run.n_agents=2000 pool.present_cap=2000 run.n_steps=144 run.name=night0_smoke \
  model.backend=<finals-vllm7のmodelブロック or dotlist>
```
★`pool.present_cap` は `run.n_agents` と**必ず対で**下げる。finals_observe は `pool.enabled: true`
なので**在場人口は present_cap 側が効く**。付け忘れると 2,000 体のつもりで 25 万体ぶんを組み立て、
**CPU高負荷・RSS 2.5GB・GPU 0%・run dir が空**のまま進まない(2026-08-16 に実地で踏んだ事故。
起動バナー2行目の `present_cap=` が nominal N と一致しているかを1秒で目視すること)。
記録するもの(summary.json と経過時刻から):
- **R_eff**(呼/秒・実効スループット)と **c**(エンジン秒/体/step)——**初の実機値**(いままで±50%誤差の外挿だった)
- fallback率・cache hit・peak RSS
→ この2値を **decision-dashboard D1-b の表に代入**して fire GO/NO-GO と本番規模の壁時計を確定する。**数字を貼ってもらえれば私が判定を返す。**

### 1-3. fire 呼数実測(mockで可・GPU不要・10分)
ダッシュボード **D1-a の手順そのまま**(mock 2ラン比較)。既にローカルPCでも実行可。

---

## Phase 2 — 今夜の初ラン(見積り 起動15分+走行数時間)

**推奨構成: 10,000体 × 1シミュ日(144step)× finals_observe(v1プール)**。
fire は Phase 1 の判定が **GO なら3行解凍**(conf の cognition ブロック・D1-c チェックリスト8項を上から確認)、判定が出ない/CONDITIONALなら**OFFのまま走らせる**(fireは後から開けられる。今夜の目的は「動く」こと)。

```bash
python scripts/run.py --profile conf/finals_observe.yaml \
  run.seed=42 run.n_agents=10000 pool.present_cap=10000 run.n_steps=144 run.name=scale1_10k \
  model.backend=<vllm7> &                                # ★present_cap は n_agents と対指定
# 併走(別ターミナル・すべて読み取り専用):
python scripts/watchdog.py --run-dir runs/scale1_10k
python scripts/report_progress.py runs/scale1_10k --dry-run          # 初回は必ずdry-run
# webhook を env に設定済みなら:
#   export SHIBUYA_DISCORD_WEBHOOK=<URL>   (URLはチャット/リポに貼らない)
#   python scripts/report_progress.py runs/scale1_10k --interval 900
python scripts/live_viewer.py runs/scale1_10k                        # 途中経過の目視
```

完走後:
```bash
python scripts/backup_run.py --run-dir runs/scale1_10k --dest <退避先> --ckpt-generations 999   # E0
python scripts/calibrate_report.py runs/scale1_10k                             # 較正指標
```

---

## Phase 3 — 夜間〜明日の判定メニュー(ランを回しながら/回した後)

| # | 項目 | 入力 | 判定者 |
|---|---|---|---|
| 1 | **fire GO/NO-GO** | Phase1のR_eff/c→D1-b式 | 数字をもらえれば私 |
| 2 | **POP転出/転入 ON** | scale1ランの `summary.population.per_day` を現実レート(転出7.8/転入8.4件日換算)と照合 | 同上(合えばconf 2行ON) |
| 3 | **PRES-A2 emergent ON** | scale1ランのRSS/R_eff | 同上(1行ON) |
| 4 | **v2プール切替** | GPU機で `--v2 --childcare` 生成→tier_quota再計算→縦煙48step→conf待機ブロックの1行 | 縦煙緑なら切替推奨 |
| 5 | policy_cache | resume前後の呼数差(scale1をresumeして比較) | 私 |
| 6 | 犯罪V1/V2 | `scripts/probe_deviance_choice.py` / `probe_victim_react.py`(docstringのコマンド・--allow-real-llm) | 実測値を記録 |
| 7 | 信頼性リハーサル | finals-reliability-plan の7本+watchdog閾値の本選値化 | 手順書どおり |
| 8 | **U-10 事前登録** | 本番10日ラン開始の直前に私から承認依頼(お約束どおり) | ユーザー |

## スケールアップの階段(ユーザー方針=少人数・短期間から)
```
今夜: 2,000×48step(計測) → 10,000×1日(初ラン)
明日: 50,000×1日 → 判定反映(fire/POP/A2/v2) → 100,000×1日
明後日〜: 250,000×1-2日(リハーサル・T10確定) → 本番 250,000×10日
```
各段で見るのは「完走・R_eff/cの線形性・RSS・飢餓カウンタ(summary.starvation)・保存則」。段間の設定変更は**1レバーずつ**(どの変更が効いたか分かるように)。

## 今夜のNG集(事故防止)
- 走行中run-dirへの `robocopy`/`cp -r` 直がけ禁止(backup_run.py経由=共有フラグ読み)
- `rm checkpoint/*` 絶対禁止(E0)・`--ckpt-generations 999` を常用
- webhook URLをコマンドライン引数に渡さない(env のみ・psに残るため)
- 走行中の conf 編集は次のランから(実行中ランはrun-dir内のconfコピーが正)
