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
  プロセスが生きている → kill して再開。
- **正常終了**: exit 0 かつ summary.json 存在(= n_steps 到達)→ 再起動せず終了。
- **リトライ上限** `--max-restarts`(既定10)超過 → status=failed で諦める。
- **連続即死**(起動 `--min-uptime-sec` 秒以内に進捗なく死ぬ)→ 指数バックオフ。
- **破損対策**: 同一 checkpoint からの再開失敗が2回続く / checkpoint が読めない →
  1世代前(バックアップ)へ巻き戻して再開。
- **バックアップ**: checkpoint が進むたび(または `--backup-every-min`)、checkpoint/ +
  config.yaml + l1 parts を `--backup-dir`(既定 `<run-dir>_backup`)へ世代コピー(直近3世代)。

記録
----
- `<run-dir>/watchdog.log`: タイムスタンプ付きの全アクション。
- `<run-dir>/status.json`: {state, restarts, last_progress, ...}。
- `<run-dir>/run.out.log`: 子プロセスの stdout/stderr(障害解析用)。

Windows 対応・標準ライブラリのみ(society を import しない)。
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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
        self.stall_sec = float(args.stall_min) * 60.0
        self.max_restarts = int(args.max_restarts)
        self.poll_sec = float(args.poll_sec)
        self.min_uptime_sec = float(args.min_uptime_sec)
        self.backoff_base = float(args.backoff_base_sec)
        self.backoff_cap = float(args.backoff_cap_sec)
        self.backup_every_sec = (float(args.backup_every_min) * 60.0
                                 if args.backup_every_min else 0.0)
        self.keep_gens = int(args.keep_backups)

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
            print(f"[watchdog] {msg}", flush=True)

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

    def _latest_ckpt(self, root: Path | None = None) -> Path | None:
        d = self._ckpt_dir(root)
        if not d.is_dir():
            return None
        cands = [p for p in d.glob(CKPT_GLOB) if _ckpt_step(p) is not None]
        if not cands:
            return None
        return max(cands, key=lambda p: _ckpt_step(p))

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
        """run-dir の checkpoint(隔離した corrupt は残す)と part を消す。"""
        for p in self._ckpt_dir().glob(CKPT_GLOB):
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
                   help="進捗が止まったと判定するまでの分(既定20)")
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
