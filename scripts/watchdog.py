"""本番シミュレーションの安全装置(watchdog)。

run.py を子プロセスとして起動・監督し、障害からの自動復旧を担う。
checkpoint / resume(D16)はエンジン側に実装済み。本モジュールはその上に載る
「落ちても勝手に立ち上がり、壊れても最後の健全な地点まで巻き戻す」外殻。

用法
----
    # 初回起動(run.py の引数は `--` の後ろにそのまま並べる)
    python scripts/watchdog.py --run-dir runs/prod1 -- \\
        run.out_dir=runs run.name=prod1 model.backend=vllm \\
        observer.checkpoint_every=72 run.n_steps=4032

    # 既存 run-dir の監督(checkpoint があれば --resume で再開する)
    python scripts/watchdog.py --run-dir runs/prod1 -- <同じ run.py 引数>

監督ループ(要点)
------------------
- **プロセス死**(exit != 0): 最新 checkpoint から `--resume` で再起動。
- **ストール**: run-dir の checkpoint / l1 part が `--stall-min`(既定20分)進まず、かつ
  プロセスが生きている → kill して再開。**`--stall-step-sec` を渡すと
  `max(--stall-min, --stall-step-sec × --stall-factor)`** で判定する(下記「本選値」)。
- **正常終了**: exit 0 かつ summary.json 存在(= n_steps 到達)→ 再起動せず終了。
- **リトライ上限** `--max-restarts`(既定10)超過 → status=failed で諦める。
- **連続即死**(起動 `--min-uptime-sec` 秒以内に進捗なく死ぬ)→ 指数バックオフ。
- **破損対策**: 同一 checkpoint からの再開失敗が2回続く / checkpoint が読めない →
  1世代前(バックアップ)へ巻き戻して再開。
- **完全世代だけを掴む**(第133 レーンR2): 再開元・進捗・バックアップの基準になる
  「最新 checkpoint」は **COMPLETE マーカーのある世代だけ**から選ぶ
  (= `society.engine.checkpoint.complete_generations` と同一規則。society を import
  できない環境では同一規則の stdlib 版 `complete_ckpts_stdlib` へ後退する)。
  書き込み途中で死んだ未コミット世代を掴んで、engine 側の明示エラーで即死する経路を塞ぐ。
- **fresh 再起動前の退避**(第133 レーンR2): checkpoint が 1 つも無い状態からの再起動
  (= `--resume` を付けない起動)は、run.py の A7 ガード(使用済み run dir の拒否)に
  弾かれて**再起動ループ**になる。そこで fresh 起動の直前に、run dir が既に使われて
  いれば **`<run-dir>.crash-<連番>` へ丸ごと rename して退避**し(同一ボリューム rename
  = 一瞬・1 バイトも消さない)、空の run dir から始める。`<run-dir>_backup` に世代が
  残っていれば同じ連番で一緒に退避する(前のランの世代を新しいランへ混ぜない)。
  **rename に失敗したら停止**する(黙って `run.allow_dirty_outdir` で重ね書きしない)。
- **バックアップ**: checkpoint が進むたび(または `--backup-every-min`)、checkpoint/ +
  config.yaml + l1 parts を `--backup-dir`(既定 `<run-dir>_backup`)へ世代コピー(直近3世代)。

LLM 健全性の監視(P0バッチ 2026-07-29 追加)
------------------------------------------
- ラン中に `l2_metrics(.part-*).parquet` の最新行の **`llm_fallback_rate`** を定期的に読み、
  `--fallback-warn`(既定 0.20)を超えたら watchdog.log と status.json に警告を出す。
- **警告のみ。プロセスは絶対に止めない**(パース失敗が増えても観察ランは続ける方が損失が小さい)。
- 列が無い(= `observer.llm_health.enabled=false` で回している)場合は起動後 1 回だけ
  「監視不能」を記録する。pyarrow が無い環境でも 1 回記録して監視を諦める(watchdog 本体の
  stdlib 限定方針を壊さないため import は関数内・失敗許容)。
- 事後(ラン終了後)の点検は `python scripts/watchdog_llm.py <run-dir>` を使う。

ディスク残量ガード(レーンREL 2026-08-12 追加)
----------------------------------------------
- 監視サイクルごと(`--disk-check-min`・既定5分)に **ラン対象ドライブの空き容量を1行**
  watchdog.log へ記録する(`disk run=... free=..GB state=ok`)。backup-dir が別ボリュームなら
  そちらも 1 行。
- `--disk-warn-gb`(既定20GB)を下回ると **警告**(コンソールにも出す)。
  `--disk-crit-gb`(既定5GB)を下回ると **致命警告**とし、さらに
  **世代バックアップ(= 新しい checkpoint コピー)を書く直前**に強調して警告する
  (数十GBのコピーがディスクを踏み抜く直前の最後の一声)。
- **警告のみ。ランは絶対に止めない**(llm_fallback_rate と同じ流儀)。何を消すか=
  checkpoint 世代の剪定か転送済み世代の削除か、は**人間の判断**に残す。
- 空き容量は `shutil.disk_usage`(stdlib)。取得できない環境では state="unknown" として
  1 回だけ記録し、以後は黙る(欠測を 0 と偽らない)。

★本選(25万体 10 シミュ日)の推奨値 — 既定値は 1 つも変えていない(第117 レーンB3)
------------------------------------------------------------------------------
正典: [ops/finals-compute-checklist.md](../ops/finals-compute-checklist.md) E3。
**既定は開発用の小さいラン向け**で、本選規模ではどれも小さすぎる:

- **停滞判定**: 25万体の 1 step は checkpoint 間隔より遥かに長い。**8/15 診断で測った
  1 step の実測秒**を `--stall-step-sec` に入れ、`--stall-factor`(既定6)を掛けた値と
  `--stall-min` の**大きい方**で判定する。1 step 90 秒なら 90×6=540 秒 = 9 分 → 20 分側が
  勝つ。1 step 600 秒なら 3,600 秒 = 60 分が勝つ。**「1 step も進まないうちに kill する」
  という最悪の事故**(= 進捗が checkpoint 粒度でしか見えないのに待ち時間が固定)を、
  実測 1 本で塞ぐための引数。
- **ディスク**: `--disk-warn-gb 50 --disk-crit-gb 20`。既定(20/5)は本選には小さい —
  E0(剪定禁止)で checkpoint と dormant を**1 世代も消さない**運用なので、警告が出た
  時点で数十 GB の余裕が要る(世代バックアップのコピー 1 回ぶんが数 GB)。
- **バックアップ**: watchdog の `--keep-backups` はノード内のローリング複製。
  **世代保全は `scripts/backup_run.py <run> <dest> --ckpt-generations 999`** だけが担う
  (backup_run 側の既定 2 のままだと 18 世代が手元に残らない = E0 事故)。

記録
----
- `<run-dir>/watchdog.log`: タイムスタンプ付きの全アクション。
- `<run-dir>/status.json`: {state, restarts, last_progress, llm_health, ...}。
- `<run-dir>/run.out.log`: 子プロセスの stdout/stderr(障害解析用)。

Windows 対応・標準ライブラリのみ(society は**任意** import = 失敗しても watchdog は
死なない。pyarrow も LLM 健全性監視のために関数内で任意 import し、無ければ監視を諦める)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_PY = REPO_ROOT / "scripts" / "run.py"
CKPT_GLOB = "ckpt-*.pkl.gz"
PART_GLOB = "*.part-*.parquet"
#: 第117 β7 の COMPLETE マーカー拡張子(`society/engine/checkpoint.COMPLETE_SUFFIX` と同値。
#: 本 module は society を import しない stdlib 限定なので値だけを写している)。
#: ★`CKPT_GLOB` は `.pkl.gz` 終端なのでマーカーには**掛からない** = 進捗検知や
#:   `_latest_ckpt` は無風。マーカーを明示的に扱うのは「本体と一緒に動かす」2 箇所だけ。
COMPLETE_SUFFIX = ".complete"
GB = float(1 << 30)

#: A7(第133 レーンR2): `scripts/run.py` の「非 resume 起動が**使用済み** run dir を踏むのを
#: 拒否する」ガードが見る痕跡の glob。`run.DIRTY_OUTDIR_GLOBS` と**同値**で、本 module は
#: stdlib 限定なので値だけを写している(tests/test_watchdog.py が run.py 側との一致を機械固定
#: する = run.py に glob が増えたらこちらが赤くなる)。
DIRTY_OUTDIR_GLOBS = (
    "*.part-*.parquet",
    "l1_events.parquet",
    "checkpoint/ckpt-*.pkl.gz",
    "checkpoint/dormant-*.pkl.gz",
    "llm_cache*.jsonl",
)
#: 退避先の名前。`runs/prod1` → `runs/prod1.crash-001`(既存を絶対に上書きしない連番)。
CRASH_SUFFIX_FMT = ".crash-{:03d}"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 「完全世代」の判定(第133 レーンR2)
#
#   第133 で COMPLETE マーカーは「本体 + pool サイドカー + 全 flush」が終わった後の
#   **世代トランザクションのコミット点**になった。resume 側(engine)はマーカーのある世代
#   しか掴まないので、watchdog が未コミット世代を「最新」とみなすと、
#     - `--resume` の起動が engine の明示エラーで即死する
#     - その世代をバックアップ世代としてコピーしてしまう
#   の 2 つが起きる。**判定規則を engine と 1 つにする**のが唯一の正しい直し方。
#
#   本体(`society.engine.checkpoint.complete_generations`)を任意 import して使い、
#   import できない環境(society 側の依存が壊れている等)では同一規則の stdlib 版へ
#   後退する。**watchdog 自身は絶対に死なない**が最優先(監督者が落ちるのが最悪)。
#   両者の一致は tests/test_watchdog.py が機械固定する。
# --------------------------------------------------------------------------- #
_CK_MOD: object | None = None
_CK_TRIED = False


def _checkpoint_module():
    """`society.engine.checkpoint` を任意 import(1 回だけ試す。失敗は None)。"""
    global _CK_MOD, _CK_TRIED
    if not _CK_TRIED:
        _CK_TRIED = True
        try:
            src = str(REPO_ROOT / "src")
            if src not in sys.path:
                sys.path.append(src)       # 末尾へ = watchdog 自身の import 解決を変えない
            from society.engine import checkpoint as _ck
            _CK_MOD = _ck
        except Exception:                  # 依存欠落・構文エラー等でも監督は続ける
            _CK_MOD = None
    return _CK_MOD


def complete_ckpts_stdlib(ckpt_dir: Path) -> list[Path]:
    """`checkpoint/` 内の**完全世代**を step 昇順で返す(stdlib 版・純関数)。

    `checkpoint.complete_generations` と同一規則:
      - COMPLETE マーカー(`<名前>.complete`)のある本体だけを候補にする。
      - ★後方互換: マーカーが **1 つも無い**ディレクトリ(第117 以前のラン・
        バックアップから戻した dir)は従来どおり全候補を許す。
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    cands = [p for p in d.glob(CKPT_GLOB) if _ckpt_step(p) is not None]
    if not cands:
        return []
    marked = [p for p in cands if p.with_name(p.name + COMPLETE_SUFFIX).exists()]
    if marked:                             # 1 つでもあれば「マーカー運用中」とみなす
        cands = marked
    return sorted(cands, key=lambda p: (_ckpt_step(p), p.name))


def dirty_outdir_hits(run_dir: Path, limit: int = 6) -> list[str]:
    """run dir に残る「使用済みの痕跡」(相対パス)。無ければ空 list(読むだけ)。

    `run.dirty_outdir_hits` と同型(向こうは cfg 経由・こちらは Path 直)。
    """
    root = Path(run_dir)
    if not root.is_dir():
        return []
    hits: list[str] = []
    for pat in DIRTY_OUTDIR_GLOBS:
        for p in sorted(root.glob(pat)):
            hits.append(p.relative_to(root).as_posix())
            if len(hits) >= limit:
                return hits
    return hits


# --------------------------------------------------------------------------- #
# ディスク残量(純関数で切り出す = 閾値判定を単体テストできるように)
# --------------------------------------------------------------------------- #
def disk_state(free_bytes: int | None, warn_gb: float, crit_gb: float) -> str:
    """空き容量の状態を返す: "unknown" | "critical" | "warn" | "ok"。

    - `free_bytes is None`(取得できなかった)は **"unknown"**。0 と偽らない。
    - 閾値 0 以下は「その段を使わない」の意味(--disk-warn-gb 0 で無効化)。
    - 致命判定が先。crit >= warn という設定ミスをしても「より厳しい方」が勝つだけで、
      判定が壊れることはない。
    """
    if free_bytes is None:
        return "unknown"
    free_gb = float(free_bytes) / GB
    if crit_gb > 0 and free_gb < crit_gb:
        return "critical"
    if warn_gb > 0 and free_gb < warn_gb:
        return "warn"
    return "ok"


def stall_seconds(stall_min: float, step_sec: float = 0.0,
                  factor: float = 6.0) -> float:
    """停滞と判定するまでの秒数(純関数 = 単体テストできるように切り出す)。

    - `step_sec <= 0`(既定)では **`--stall-min` そのまま** = 従来と 1 秒も変わらない。
    - `step_sec > 0` を渡すと「**1 step の実測時間 × 係数**」と `--stall-min` の
      **大きい方**を採る。小さい方を採らないのは、実測が短い(= 開発機で測った)ときに
      本選で誤 kill する側へ倒れないため。閾値は必ず「安全側 = 待つ側」へ動く。
    """
    base = float(stall_min) * 60.0
    if step_sec > 0.0 and factor > 0.0:
        return max(base, float(step_sec) * float(factor))
    return base


def _existing_ancestor(path: Path) -> Path | None:
    """まだ作られていない dir(例: backup-dir)でも測れるよう、実在する祖先まで遡る。"""
    p = path
    while True:
        if p.exists():
            return p
        if p.parent == p:
            return None
        p = p.parent


def disk_free_bytes(path: Path) -> int | None:
    """path が載っているボリュームの空きバイト。測れなければ None(捏造しない)。"""
    base = _existing_ancestor(Path(path))
    if base is None:
        return None
    try:
        return int(shutil.disk_usage(str(base)).free)
    except OSError:
        return None


def _volume_key(path: Path):
    """同一ボリュームかの判定キー(Windows はドライブ、POSIX は st_dev)。"""
    base = _existing_ancestor(Path(path))
    if base is None:
        return None
    try:
        return os.stat(str(base)).st_dev
    except OSError:
        return str(base.anchor) or None


def _ckpt_step(path: Path | None) -> int | None:
    """ckpt-NNNNNN.pkl.gz からステップ番号を取り出す。"""
    if path is None:
        return None
    try:
        return int(path.name[len("ckpt-"):].split(".")[0])
    except ValueError:
        return None


class Watchdog:
    def __init__(self, args: argparse.Namespace):
        self.run_dir = Path(args.run_dir).resolve()
        self.backup_dir = (Path(args.backup_dir).resolve()
                           if args.backup_dir
                           else Path(str(self.run_dir) + "_backup"))
        self.child_args = list(args.child_args)
        self.base_cmd = (shlex.split(args.cmd) if args.cmd
                         else [sys.executable, str(DEFAULT_RUN_PY)])
        # 停滞判定(既定 = --stall-min のみ。--stall-step-sec を渡したときだけ 1 step 連動)
        self.stall_sec = stall_seconds(args.stall_min,
                                       float(getattr(args, "stall_step_sec", 0.0) or 0.0),
                                       float(getattr(args, "stall_factor", 6.0) or 0.0))
        self.max_restarts = int(args.max_restarts)
        self.poll_sec = float(args.poll_sec)
        self.min_uptime_sec = float(args.min_uptime_sec)
        self.backoff_base = float(args.backoff_base_sec)
        self.backoff_cap = float(args.backoff_cap_sec)
        self.backup_every_sec = (float(args.backup_every_min) * 60.0
                                 if args.backup_every_min else 0.0)
        self.keep_gens = int(args.keep_backups)
        # ---- LLM 健全性監視(P0バッチ)----
        self.fallback_warn = float(getattr(args, "fallback_warn", 0.20) or 0.0)
        self.health_check_sec = float(getattr(args, "health_check_min", 10.0)) * 60.0
        self._health_next_t = 0.0
        self._health_unavailable_logged = False
        self._health_warned = False
        self.last_health: dict | None = None
        # ---- ディスク残量ガード(レーンREL)----
        self.disk_warn_gb = float(getattr(args, "disk_warn_gb", 20.0) or 0.0)
        self.disk_crit_gb = float(getattr(args, "disk_crit_gb", 5.0) or 0.0)
        self.disk_check_sec = float(getattr(args, "disk_check_min", 5.0) or 0.0) * 60.0
        self._disk_next_t = 0.0
        self._disk_unavailable_logged = False
        self.last_disk: dict | None = None

        # --- 監督状態 ---
        self.restarts = 0
        self.quick_death_streak = 0
        self.same_ckpt_fail = 0
        self.last_failed_ckpt_step: int | None = None
        self.last_backup_step: int | None = None
        self.last_backup_time = 0.0

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "watchdog.log"
        self.status_path = self.run_dir / "status.json"
        self.child_log = self.run_dir / "run.out.log"

    # ------------------------------------------------------------------ #
    # 記録
    # ------------------------------------------------------------------ #
    def log(self, msg: str, echo: bool = True) -> None:
        line = f"[{_now_iso()}] {msg}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if echo:
            # ★第133 レーンR2: コンソールが cp932(日本語 Windows の既定)だと、記録には
            #   問題なく書ける文字(em dash 等)が print で UnicodeEncodeError を投げる。
            #   例外は log() の呼び出し元へ伝播して**監督者ごと落とす** = 最悪の事故。
            #   ファイル(utf-8)への記録は既に済んでいるので、表示だけ落として続ける。
            try:
                print(f"[watchdog] {msg}", flush=True)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                print(f"[watchdog] {msg}".encode(enc, "replace").decode(enc, "replace"),
                      flush=True)

    def write_status(self, state: str, pid: int | None = None) -> None:
        data = {
            "state": state,                       # running|restarting|failed|done
            "restarts": self.restarts,
            "max_restarts": self.max_restarts,
            "last_progress": self._progress_desc(),
            "last_backup_step": self.last_backup_step,
            "run_dir": str(self.run_dir),
            "backup_dir": str(self.backup_dir),
            "pid": pid,
            "updated": _now_iso(),
        }
        if self.last_health is not None:      # P0バッチ: 直近の LLM 健全性(監視できたときのみ)
            data["llm_health"] = self.last_health
        if self.last_disk is not None:        # レーンREL: 直近のディスク残量
            data["disk"] = self.last_disk
        tmp = self.status_path.with_name(self.status_path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.status_path)

    def _progress_desc(self) -> dict | None:
        latest = self._latest_ckpt()
        if latest is None:
            return None
        return {
            "checkpoint": latest.name,
            "step": _ckpt_step(latest),
            "mtime": datetime.fromtimestamp(
                latest.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------------------ #
    # checkpoint / 進捗の観測(society 非依存の自前実装)
    # ------------------------------------------------------------------ #
    def _ckpt_dir(self, root: Path | None = None) -> Path:
        return (root or self.run_dir) / "checkpoint"

    def _complete_ckpts(self, root: Path | None = None) -> list[Path]:
        """`<root>/checkpoint/` の**完全世代**(COMPLETE マーカー付き)を step 昇順で返す。

        判定規則は engine と 1 つ(`_checkpoint_module` の解説を参照)。society を
        import できた場合は本体の `complete_generations` を、できない場合は同一規則の
        stdlib 版を使う。どちらの経路でも「名前から step が読めないファイル」は
        従来どおり候補にしない(= `_ckpt_step` フィルタは残す)。
        """
        d = self._ckpt_dir(root)
        if not d.is_dir():
            return []
        ck = _checkpoint_module()
        if ck is None:
            return complete_ckpts_stdlib(d)
        try:
            gens = [p for p in ck.complete_generations(root or self.run_dir)
                    if _ckpt_step(p) is not None]
        except Exception as e:                 # 本体側が落ちても監督は続ける
            self.log(f"complete_generations failed ({e}); using stdlib mirror",
                     echo=False)
            return complete_ckpts_stdlib(d)
        return sorted(gens, key=lambda p: (_ckpt_step(p), p.name))

    def _latest_ckpt(self, root: Path | None = None) -> Path | None:
        """最新の**完全世代**(未コミット世代は掴まない。第133 レーンR2)。"""
        gens = self._complete_ckpts(root)
        return gens[-1] if gens else None

    def _ckpt_step_now(self) -> int | None:
        return _ckpt_step(self._latest_ckpt())

    def _verify_ckpt(self, path: Path | None) -> int | None:
        """checkpoint が読める(gzip CRC + pickle + format/step)なら step を返す。"""
        if path is None or not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as f:
                blob = pickle.loads(f.read())
            if not isinstance(blob, dict):
                return None
            if blob.get("format") is None or blob.get("step") is None:
                return None
            return int(blob["step"])
        except Exception as e:                     # gzip/pickle 破損など
            self.log(f"checkpoint verify failed: {path.name}: {e}", echo=False)
            return None

    def _progress_fp(self) -> tuple:
        """ストール検知用の進捗指紋(checkpoint と l1 part の集合+mtime)。"""
        latest = self._latest_ckpt()
        ck = (latest.name, round(latest.stat().st_mtime, 3)) if latest else (None, 0.0)
        parts = sorted(self.run_dir.glob(PART_GLOB))
        pmtime = max((p.stat().st_mtime for p in parts), default=0.0)
        return (ck, len(parts), round(pmtime, 3))

    # ------------------------------------------------------------------ #
    # LLM 健全性の監視(P0バッチ 2026-07-29。警告のみ・プロセスは止めない)
    # ------------------------------------------------------------------ #
    def read_llm_health(self) -> dict | None:
        """run-dir の L2 から最新の LLM 健全性 3 列を読む。読めなければ None。

        探索順は「最も新しい part → canonical」。observer.llm_health.enabled=false の
        ランには列が無いので None を返す(捏造しない)。pyarrow が無い環境でも None。
        """
        try:
            import pyarrow.parquet as pq          # 任意 import(stdlib 限定方針の例外)
        except Exception:
            return None
        cands = sorted(self.run_dir.glob("l2_metrics.part-*.parquet"))
        canonical = self.run_dir / "l2_metrics.parquet"
        if canonical.exists():
            cands.append(canonical)
        for path in reversed(cands):
            try:
                tbl = pq.read_table(path)
            except Exception:
                continue
            cols = set(tbl.column_names)
            if not {"llm_calls_total", "llm_fallback_rate"} <= cols:
                continue
            if tbl.num_rows == 0:
                continue
            last = tbl.slice(tbl.num_rows - 1, 1).to_pylist()[0]
            return {
                "step": last.get("step"),
                "llm_calls_total": last.get("llm_calls_total"),
                "llm_fallback_rate": last.get("llm_fallback_rate"),
                "llm_cache_hit_rate": last.get("llm_cache_hit_rate"),
                "source": path.name,
            }
        return None

    def _maybe_check_llm_health(self, now: float) -> None:
        if self.fallback_warn <= 0.0 or now < self._health_next_t:
            return
        self._health_next_t = now + max(self.health_check_sec, 30.0)
        health = self.read_llm_health()
        if health is None:
            if not self._health_unavailable_logged:
                self._health_unavailable_logged = True
                self.log("LLM 健全性を監視できない(L2 に llm_fallback_rate 列が無い"
                         " or pyarrow 不在)。observer.llm_health.enabled=true で回すと"
                         "監視できる", echo=False)
            return
        self.last_health = health
        rate = health.get("llm_fallback_rate")
        if rate is None:
            return
        if float(rate) > self.fallback_warn:
            self.log(f"WARN llm_fallback_rate={float(rate):.4f} > "
                     f"{self.fallback_warn:.4f} (step={health.get('step')}, "
                     f"calls={health.get('llm_calls_total')}) — "
                     "モデル/プロンプト/バックエンドの点検を推奨(ランは継続する)")
            self._health_warned = True
        elif self._health_warned:                 # 復帰も記録する(閾値を跨ぎ直したとき)
            self._health_warned = False
            self.log(f"llm_fallback_rate が閾値以下へ復帰: {float(rate):.4f} "
                     f"(step={health.get('step')})", echo=False)

    # ------------------------------------------------------------------ #
    # ディスク残量ガード(レーンREL。警告のみ・プロセスは止めない)
    # ------------------------------------------------------------------ #
    def _disk_targets(self) -> list[tuple[str, Path]]:
        """測るべきパス。backup-dir が run-dir と別ボリュームなら 2 本になる。"""
        out: list[tuple[str, Path]] = [("run", self.run_dir)]
        kr, kb = _volume_key(self.run_dir), _volume_key(self.backup_dir)
        if kb is not None and kb != kr:
            out.append(("backup", self.backup_dir))
        return out

    def check_disk(self, tag: str | None = None) -> dict | None:
        """残量を 1 行ずつ記録し、閾値を跨いでいれば警告する。**止めない**。

        `tag` は「いつの点検か」(例: "before backup gen step=144")。致命閾値を
        下回っているときだけ、tag を添えて強調警告する = 大きなコピーを始める直前の
        最後の一声。何を消すかの判断は人間に残す(自動で剪定はしない)。
        """
        snaps: list[dict] = []
        for label, path in self._disk_targets():
            free = disk_free_bytes(path)
            state = disk_state(free, self.disk_warn_gb, self.disk_crit_gb)
            snap = {"label": label, "path": str(path), "state": state,
                    "free_gb": (round(free / GB, 2) if free is not None else None),
                    "warn_gb": self.disk_warn_gb, "crit_gb": self.disk_crit_gb,
                    "checked": _now_iso()}
            snaps.append(snap)
            if state == "unknown":
                if not self._disk_unavailable_logged:
                    self._disk_unavailable_logged = True
                    self.log(f"disk {label}={path} free=unknown "
                             "(shutil.disk_usage が測れない環境。以後は黙る)",
                             echo=False)
                continue
            free_gb = free / GB
            line = (f"disk {label}={path} free={free_gb:.1f}GB state={state} "
                    f"(warn<{self.disk_warn_gb:g}GB crit<{self.disk_crit_gb:g}GB)")
            self.log(line, echo=(state != "ok"))
            if state == "critical":
                self.log(f"*** DISK CRITICAL *** {label} の空きが {free_gb:.1f}GB "
                         f"(< {self.disk_crit_gb:g}GB)"
                         + (f" — {tag} の直前" if tag else "")
                         + "。checkpoint 世代の剪定 / 転送済みバックアップ世代の削除を"
                           "検討すること(ランは止めない=停止判断は人間)")
            elif state == "warn":
                self.log(f"WARN disk {label} の空きが {free_gb:.1f}GB "
                         f"(< {self.disk_warn_gb:g}GB)。残り日数×日次増分を見積もって"
                         "退避・剪定の計画を(ランは継続する)")
        # status.json 用: **常に run ボリュームの値を最上位**に置く(消費側が
        # `status["disk"]["free_gb"]` で必ず読める)。別ボリュームがあるときだけ targets を足す。
        primary = dict(snaps[0])
        if len(snaps) > 1:
            primary["targets"] = snaps
        self.last_disk = primary
        return self.last_disk

    def _maybe_check_disk(self, now: float) -> None:
        if self.disk_check_sec <= 0 or now < self._disk_next_t:
            return
        self._disk_next_t = now + max(self.disk_check_sec, 30.0)
        self.check_disk()

    # ------------------------------------------------------------------ #
    # バックアップ(世代管理)
    # ------------------------------------------------------------------ #
    def _gen_step(self, gen: Path) -> int:
        try:
            return int(gen.name.split("-")[1])
        except (IndexError, ValueError):
            return -1

    def _gens(self) -> list[Path]:
        if not self.backup_dir.is_dir():
            return []
        return sorted((p for p in self.backup_dir.glob("gen-*") if p.is_dir()),
                      key=lambda p: (self._gen_step(p), p.name))

    def _maybe_backup(self, step: int | None) -> None:
        if step is None:
            return
        now = time.monotonic()
        due = (step != self.last_backup_step)
        if not due and self.backup_every_sec > 0:
            due = (now - self.last_backup_time) >= self.backup_every_sec
        if due:
            self._do_backup(step)
            self.last_backup_step = step
            self.last_backup_time = now

    def _do_backup(self, step: int) -> None:
        # 世代コピー(checkpoint+part 群)は run 中で最も大きな書き込み。踏み抜く前に一声。
        # (--disk-check-min 0 = 残量監視そのものを切る、なのでここも黙る)
        if self.disk_check_sec > 0:
            self.check_disk(tag=f"backup gen step={step}")
            self._disk_next_t = time.monotonic() + max(self.disk_check_sec, 30.0)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        gen = self.backup_dir / f"gen-{step:06d}-{ts}"
        tmp = self.backup_dir / (gen.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            src_ck = self._ckpt_dir()
            if src_ck.is_dir():
                shutil.copytree(src_ck, tmp / "checkpoint",
                                ignore=shutil.ignore_patterns("corrupt", "*.tmp"),
                                dirs_exist_ok=True)
            for p in self.run_dir.glob(PART_GLOB):
                shutil.copy2(p, tmp / p.name)
            cfg = self.run_dir / "config.yaml"
            if cfg.exists():
                shutil.copy2(cfg, tmp / "config.yaml")
            os.replace(tmp, gen)                   # ディレクトリの原子的 rename
        except Exception as e:
            self.log(f"backup failed (step {step}): {e}")
            shutil.rmtree(tmp, ignore_errors=True)
            return
        self.log(f"backup gen created: {gen.name}")
        self._rotate_backups()

    def _rotate_backups(self) -> None:
        # 中断した .tmp の掃除
        for stray in self.backup_dir.glob("gen-*.tmp"):
            shutil.rmtree(stray, ignore_errors=True)
        gens = self._gens()
        for old in gens[:-self.keep_gens] if self.keep_gens > 0 else []:
            shutil.rmtree(old, ignore_errors=True)
            self.log(f"backup gen rotated out: {old.name}", echo=False)

    def _best_backup_gen(self, max_step_exclusive: int | None) -> Path | None:
        """検証を通る最新のバックアップ世代(必要なら step < max_step_exclusive)。"""
        for gen in reversed(self._gens()):
            step = self._gen_step(gen)
            if max_step_exclusive is not None and step >= max_step_exclusive:
                continue
            if self._verify_ckpt(self._latest_ckpt(gen)) is not None:
                return gen
        return None

    def _clear_run_state(self) -> None:
        """run-dir の checkpoint(隔離した corrupt は残す)と part を消す。

        ★COMPLETE マーカー(第117 β7)も一緒に消す = **孤児マーカーを作らない**
          (本体を消してマーカーだけ残すと「本体の無い完了印」が積み上がる)。
          `_restore_from_backup` はこの直後にバックアップ世代を copytree で被せるので、
          運んできた世代のマーカーは正しく揃った状態で置かれる。
        """
        for p in self._ckpt_dir().glob(CKPT_GLOB):
            try:
                p.unlink()
            except OSError:
                pass
        for p in self._ckpt_dir().glob(CKPT_GLOB + COMPLETE_SUFFIX):
            try:
                p.unlink()
            except OSError:
                pass
        for p in self.run_dir.glob(PART_GLOB):
            try:
                p.unlink()
            except OSError:
                pass

    def _restore_from_backup(self, max_step_exclusive: int | None) -> int | None:
        """整合したバックアップ世代(checkpoint + parts + config)を run-dir へ復元。"""
        gen = self._best_backup_gen(max_step_exclusive)
        if gen is None:
            return None
        step = self._gen_step(gen)
        self._clear_run_state()
        src_ck = self._ckpt_dir(gen)
        if src_ck.is_dir():
            shutil.copytree(src_ck, self._ckpt_dir(), dirs_exist_ok=True)
        for p in gen.glob(PART_GLOB):
            shutil.copy2(p, self.run_dir / p.name)
        cfg = gen / "config.yaml"
        if cfg.exists():
            shutil.copy2(cfg, self.run_dir / "config.yaml")
        self.log(f"restored from backup {gen.name} (step {step})")
        return step

    def _rollback_one_generation(self, failing_step: int) -> None:
        """疑わしい最新 checkpoint を隔離し、1世代前の健全な地点へ戻す。"""
        latest = self._latest_ckpt()
        if latest is not None and _ckpt_step(latest) == failing_step:
            quar = self._ckpt_dir() / "corrupt"
            quar.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(latest), str(quar / latest.name))
                self.log(f"quarantined suspect checkpoint {latest.name}")
                # ★COMPLETE マーカー(第117 β7)は**本体と同時に**隔離する。本体だけを
                #   corrupt/ へ移すと `<名前>.complete` が run-dir に取り残され、
                #   「本体の無い完了印」= 孤児マーカーができる。マーカーの意味は
                #   「この本体は最後まで書けた」なので、本体を疑って隔離した以上、
                #   その保証も一緒に降ろすのが正しい(片方だけ残さない)。
                marker = latest.with_name(latest.name + COMPLETE_SUFFIX)
                if marker.exists():
                    shutil.move(str(marker), str(quar / marker.name))
                    self.log(f"quarantined COMPLETE marker {marker.name}", echo=False)
                # ★COMPLETE マーカー(第117 β7)は**本体と同時に**隔離する。本体だけを
                #   corrupt/ へ移すと `<名前>.complete` が run-dir に取り残され、
                #   「本体の無い完了印」= 孤児マーカーができる。マーカーの意味は
                #   「この本体は最後まで書けた」なので、本体を疑って隔離した以上、
                #   その保証も一緒に降ろすのが正しい(片方だけ残さない)。
            except Exception as e:
                self.log(f"quarantine failed: {e}")
        # 整合スナップショット(parts 同梱)を最優先で復元
        restored = self._restore_from_backup(max_step_exclusive=failing_step)
        if restored is not None:
            return
        # バックアップが無ければ run-dir 内に残る前世代 checkpoint を使う
        prev = self._latest_ckpt()
        if prev is not None and self._verify_ckpt(prev) is not None:
            self.log(f"rolled back to in-place checkpoint {prev.name} "
                     f"(parts may include events past this step)")
            return
        self.log("no older good checkpoint (backup or in-place); "
                 "next attempt will restart from scratch")

    # ------------------------------------------------------------------ #
    # fresh 再起動の前の退避(第133 レーンR2 = run.py の A7 ガードとの整合)
    # ------------------------------------------------------------------ #
    def _crash_dest(self) -> tuple[Path, int]:
        """退避先 `<run-dir>.crash-<連番>` と連番。**既存を絶対に上書きしない**。"""
        n = 1
        while True:
            dest = self.run_dir.with_name(self.run_dir.name + CRASH_SUFFIX_FMT.format(n))
            if not dest.exists():
                return dest, n
            n += 1

    def _prepare_fresh_outdir(self) -> bool:
        """`--resume` を付けない起動の直前に run dir を空にする。続行可なら True。

        ★なぜ要るか: 完全世代の checkpoint が 1 つも無いクラッシュ(= 最初の世代を
          書き切る前に死んだ)からの復旧は `--resume` を付けられない。しかし前回の
          `*.part-*.parquet` や `llm_cache.jsonl` は残っているので、run.py の A7 ガードが
          `RuntimeError` で即死させる → watchdog が再起動 → また即死、の**再起動ループ**に
          なる(第133 の新ガードと watchdog の不整合)。

        ★直し方: 逃し弁(`run.allow_dirty_outdir=true`)で**混ぜて通す**のは最悪手
          (2 つのランのイベントが 1 本の canonical に結合され、事後に検出できない)。
          run dir を丸ごと `<name>.crash-<連番>` へ **rename して退避**し、空の dir から
          始める。同一ボリュームの rename なので 25 万体ぶんの part があっても一瞬で、
          **1 バイトも消さない**(事後解析に全部残る)。

        ★rename に失敗したら **False = 停止**。「退避できないから重ねて書く」は
          静かにデータを壊す方向なので、判断を人間へ返す。
        """
        hits = dirty_outdir_hits(self.run_dir)
        if not hits:
            return True
        dest, n = self._crash_dest()
        bdest = (self.backup_dir.with_name(self.backup_dir.name
                                           + CRASH_SUFFIX_FMT.format(n))
                 if self.backup_dir.is_dir() and any(self.backup_dir.glob("gen-*"))
                 else None)
        self.log("fresh restart into a used run dir "
                 f"(痕跡: {', '.join(hits)}) - quarantining to {dest.name}"
                 + (f" (+ backup → {bdest.name})" if bdest else ""))
        # ★バックアップを先に動かす: ここで失敗しても run dir は 1 バイトも動いていない
        #   (= 世界の状態が中途半端にならない)。
        if bdest is not None:
            try:
                os.rename(self.backup_dir, bdest)
            except OSError as e:
                self.log(f"*** STOP *** backup dir の退避に失敗: {self.backup_dir} "
                         f"→ {bdest}: {e}。前のランの世代を新しいランへ混ぜないため"
                         "ここで停止する(run dir は動かしていない)。手動で退避するか "
                         "--backup-dir を別の場所にして再実行すること")
                return False
        try:
            os.rename(self.run_dir, dest)
        except OSError as e:
            self.log(f"*** STOP *** run dir の退避に失敗: {self.run_dir} → {dest}: {e}。"
                     "run.allow_dirty_outdir で重ねて書くと前回の part が今回の成果物へ"
                     "結合され、事後に検出できない。ここで停止する"
                     + (f"(backup は既に {bdest.name} へ退避済み)" if bdest else ""))
            return False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # ここから先のログは**新しい空の run dir** へ書かれる(退避前のログは crash dir 側に
        # 丸ごと残っている)。両方に痕跡が残るよう、退避の事実をもう一度書く。
        self.log(f"quarantined previous run dir → {dest} "
                 + (f"(backup → {bdest}) " if bdest else "")
                 + "; starting fresh from an empty dir")
        self.last_backup_step = None
        self.last_backup_time = 0.0
        self.last_failed_ckpt_step = None
        self.same_ckpt_fail = 0
        return True

    # ------------------------------------------------------------------ #
    # 子プロセスの起動・監視
    # ------------------------------------------------------------------ #
    def _build_cmd(self, resume: bool) -> list[str]:
        cmd = list(self.base_cmd)
        if resume:
            cmd += ["--resume", str(self.run_dir)]
        cmd += self.child_args
        return cmd

    def _kill(self, proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
            return
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=15)
        except Exception:
            pass

    def _run_once(self, resume: bool) -> dict:
        cmd = self._build_cmd(resume)
        resume_step = self._ckpt_step_now() if resume else None
        self.log(f"launch ({'resume' if resume else 'initial'}): "
                 f"{' '.join(cmd)}  [resume_step={resume_step}]")
        start = time.monotonic()
        with self.child_log.open("ab") as clog:
            clog.write((f"\n==== attempt#{self.restarts} {_now_iso()} "
                        f"cmd={' '.join(cmd)} ====\n").encode("utf-8"))
            clog.flush()
            try:
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                                        stdout=clog, stderr=subprocess.STDOUT)
            except Exception as e:               # 起動そのものに失敗(retry 対象)
                self.log(f"failed to spawn child: {e}")
                return {"result": "crash", "exit_code": None, "uptime": 0.0,
                        "resume_step": resume_step, "made_progress": False}
            self.write_status("running", pid=proc.pid)

            last_fp = self._progress_fp()
            last_progress_t = time.monotonic()
            seen_step = self._ckpt_step_now()
            pending_step: int | None = None
            pending_since = 0.0
            stalled = False

            while proc.poll() is None:
                time.sleep(self.poll_sec)
                now = time.monotonic()
                fp = self._progress_fp()
                if fp != last_fp:
                    last_fp = fp
                    last_progress_t = now
                elif self.stall_sec > 0 and (now - last_progress_t) > self.stall_sec:
                    self.log(f"STALL: no progress for {now - last_progress_t:.0f}s "
                             f"(> {self.stall_sec:.0f}s); killing pid {proc.pid}")
                    self._kill(proc)
                    stalled = True
                    break
                # checkpoint が進んだら1ポーリング遅らせて(flush 完了待ち)バックアップ
                step = self._ckpt_step_now()
                if step is not None and step != seen_step:
                    seen_step = step
                    pending_step = step
                    pending_since = now
                elif (pending_step is not None and step == pending_step
                      and (now - pending_since) >= self.poll_sec):
                    self._maybe_backup(pending_step)
                    pending_step = None
                self._maybe_check_llm_health(now)   # P0バッチ: 警告のみ(kill しない)
                self._maybe_check_disk(now)         # レーンREL: 残量1行+閾値警告(止めない)

            uptime = time.monotonic() - start

        ret = None if stalled else proc.poll()
        final_step = self._ckpt_step_now()
        made_progress = (final_step is not None
                         and (resume_step is None or final_step > resume_step))

        if stalled:
            self._maybe_backup(final_step)
            return {"result": "stall", "exit_code": None, "uptime": uptime,
                    "resume_step": resume_step, "made_progress": made_progress}
        if ret == 0 and (self.run_dir / "summary.json").exists():
            return {"result": "done", "exit_code": 0, "uptime": uptime,
                    "resume_step": resume_step, "made_progress": made_progress}
        self._maybe_backup(final_step)
        return {"result": "crash", "exit_code": ret, "uptime": uptime,
                "resume_step": resume_step, "made_progress": made_progress}

    # ------------------------------------------------------------------ #
    # 監督ループ本体
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        self.log(f"watchdog start. run_dir={self.run_dir} "
                 f"backup_dir={self.backup_dir} max_restarts={self.max_restarts} "
                 f"stall={self.stall_sec:.0f}s")
        self.log("complete-generation rule: "
                 + ("society.engine.checkpoint.complete_generations"
                    if _checkpoint_module() is not None
                    else "stdlib mirror (society import unavailable)"), echo=False)
        if self.disk_check_sec > 0:                 # 起動直後に残量の基準線を 1 行残す
            self.check_disk(tag="watchdog start")
            self._disk_next_t = time.monotonic() + max(self.disk_check_sec, 30.0)
        if (self.run_dir / "summary.json").exists():
            self.log("summary.json already present; run already complete. "
                     "nothing to do (delete it to force a fresh run).")
            self.write_status("done")
            return 0

        while True:
            resume = self._latest_ckpt() is not None
            if resume:
                latest = self._latest_ckpt()
                if self._verify_ckpt(latest) is None:
                    self.log(f"latest checkpoint unreadable ({latest.name}); "
                             "restoring from backup before resume")
                    if self._restore_from_backup(max_step_exclusive=None) is None:
                        self.log("no usable backup to restore. status=failed")
                        self.write_status("failed")
                        return 1
                    resume = self._latest_ckpt() is not None

            # 第133 レーンR2: `--resume` を付けない起動は、使用済み run dir を踏むと
            # run.py の A7 ガードに弾かれて再起動ループになる。退避してから始める。
            if not resume and not self._prepare_fresh_outdir():
                self.write_status("failed")
                return 1

            outcome = self._run_once(resume)

            if outcome["result"] == "done":
                self.log(f"run COMPLETE (exit 0, summary.json present). "
                         f"restarts={self.restarts}")
                self.write_status("done")
                return 0

            self.restarts += 1
            self.log(f"attempt failed: result={outcome['result']} "
                     f"exit={outcome['exit_code']} uptime={outcome['uptime']:.1f}s "
                     f"progress={outcome['made_progress']} "
                     f"(restart {self.restarts}/{self.max_restarts})")

            if self.restarts > self.max_restarts:
                self.log("max restarts exceeded; giving up. status=failed")
                self.write_status("failed")
                return 1

            # 破損対策: 同一 checkpoint から2回続けて再開失敗 → 1世代前へ
            rs = outcome["resume_step"]
            if rs is not None and not outcome["made_progress"]:
                if rs == self.last_failed_ckpt_step:
                    self.same_ckpt_fail += 1
                else:
                    self.same_ckpt_fail = 1
                    self.last_failed_ckpt_step = rs
                if self.same_ckpt_fail >= 2:
                    self.log(f"checkpoint step {rs} failed "
                             f"{self.same_ckpt_fail}x with no progress; "
                             "rolling back one generation (corruption guard)")
                    self._rollback_one_generation(rs)
                    self.same_ckpt_fail = 0
                    self.last_failed_ckpt_step = None
            else:
                self.same_ckpt_fail = 0
                self.last_failed_ckpt_step = None

            # 連続即死(進捗なく短時間で死)→ 指数バックオフ
            if outcome["uptime"] < self.min_uptime_sec and not outcome["made_progress"]:
                self.quick_death_streak += 1
                delay = min(self.backoff_cap,
                            self.backoff_base * (2 ** (self.quick_death_streak - 1)))
                self.log(f"quick death (<{self.min_uptime_sec:.0f}s, no progress); "
                         f"backoff {delay:.1f}s (streak {self.quick_death_streak})")
                self.write_status("restarting")
                if delay > 0:
                    time.sleep(delay)
            else:
                self.quick_death_streak = 0
                self.write_status("restarting")


def parse_args(argv: list[str]) -> argparse.Namespace:
    # `--` 以降は子コマンド(run.py)への引数としてそのまま渡す
    if "--" in argv:
        idx = argv.index("--")
        wd_argv, child = argv[:idx], argv[idx + 1:]
    else:
        wd_argv, child = argv, []
    p = argparse.ArgumentParser(
        prog="watchdog", description="本番シミュの安全装置(自動再開・復旧)")
    p.add_argument("--run-dir", required=True,
                   help="監督する run ディレクトリ(run.py の出力先と一致させる)")
    p.add_argument("--backup-dir", default=None,
                   help="バックアップ世代の置き場(既定: <run-dir>_backup)")
    p.add_argument("--stall-min", type=float, default=20.0,
                   help="進捗が止まったと判定するまでの分(既定20=開発用の小ラン向け)。"
                        "★本選(25万体)は --stall-step-sec に 1 step の実測秒を渡して"
                        "自動連動させる(下記)。単独で使うなら 60 以上を推奨")
    p.add_argument("--stall-step-sec", type=float, default=0.0,
                   help="1 step の実測秒(既定0=使わない=--stall-min そのまま)。"
                        "渡すと停滞判定を max(--stall-min, これ × --stall-factor) にする。"
                        "★本選は 8/15 診断の実測 1 step 秒を入れる"
                        "(進捗は checkpoint 粒度でしか見えないので、1 step も進まないうちに"
                        "kill する事故をこの 1 本で塞ぐ)")
    p.add_argument("--stall-factor", type=float, default=6.0,
                   help="--stall-step-sec に掛ける安全係数(既定6)。"
                        "--stall-step-sec を渡さないときは一切効かない")
    p.add_argument("--max-restarts", type=int, default=10,
                   help="再起動の上限回数(既定10)。超えたら status=failed")
    p.add_argument("--poll-sec", type=float, default=5.0,
                   help="監視ポーリング間隔・秒(既定5)")
    p.add_argument("--min-uptime-sec", type=float, default=60.0,
                   help="この秒数以内に進捗なく死ぬ=即死とみなしバックオフ(既定60)")
    p.add_argument("--backoff-base-sec", type=float, default=2.0)
    p.add_argument("--backoff-cap-sec", type=float, default=60.0)
    p.add_argument("--backup-every-min", type=float, default=0.0,
                   help="時間ベースの追加バックアップ間隔・分(0=checkpoint 進捗ごとのみ)")
    p.add_argument("--keep-backups", type=int, default=3,
                   help="保持するバックアップ世代数(既定3)")
    p.add_argument("--fallback-warn", type=float, default=0.20,
                   help="L2 の llm_fallback_rate がこれを超えたら警告(既定0.20・0=監視しない)。"
                        "警告のみでランは止めない。observer.llm_health.enabled=true が前提")
    p.add_argument("--health-check-min", type=float, default=10.0,
                   help="LLM 健全性の点検間隔・分(既定10。最小30秒)")
    p.add_argument("--disk-warn-gb", type=float, default=20.0,
                   help="空き容量がこれを下回ったら警告・GB(既定20=安全側。0=警告しない)。"
                        "警告のみでランは止めない。★本選推奨 50"
                        "(E0=checkpoint/dormant を 1 世代も剪定しない運用のため)")
    p.add_argument("--disk-crit-gb", type=float, default=5.0,
                   help="空き容量がこれを下回ったら致命警告・GB(既定5)。世代バックアップを"
                        "書く直前にも強調して出す(0=致命段を使わない)。★本選推奨 20"
                        "(世代コピー 1 回ぶんの余裕を残して警告するため)")
    p.add_argument("--disk-check-min", type=float, default=5.0,
                   help="残量の点検・記録の間隔・分(既定5。最小30秒。0=監視しない)")
    p.add_argument("--cmd", default=None,
                   help="テスト用フック: 起動する基底コマンドを上書き"
                        "(既定は `python scripts/run.py`)")
    ns = p.parse_args(wd_argv)
    ns.child_args = child
    return ns


def main() -> None:
    ns = parse_args(sys.argv[1:])
    raise SystemExit(Watchdog(ns).run())


if __name__ == "__main__":
    main()
