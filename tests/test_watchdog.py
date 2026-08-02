"""本番安全装置(scripts/watchdog.py)のテスト。

実 run.py を使わずに高速化するため、ダミーの子プロセス(child.py)を watchdog に
食わせる。ダミーは checkpoint(gzip+pickle の {format,step})/ l1 part / summary.json を
本番と同じ命名で書き、モードに応じて「N 回クラッシュ後に成功」「進捗せずストール」
「常にクラッシュ」「複数 checkpoint を刻む」を演じる。watchdog は `--cmd` フックで
基底コマンドを差し替えられる(本番経路と同一の監督ループを通す)。

最後に実 run.py での煙テストを1本(mock 20人×48step, checkpoint_every=12)。
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# 第86バッチ保守 M-2: このモジュールは**実際に子プロセスを起動して監督ループを回す**
# (ダミー子の「N 回クラッシュ→成功」「進捗せずストール」は poll_sec とストール判定の
# **実時間**に依存し、最後の 1 本は実 run.py を丸ごと走らせる)。xdist の既定 `--dist load`
# では他のテストと同時に走って CPU を奪い合い、ストール判定が誤発火する並列フレークを
# 2 例観測した。loadgroup(pyproject の addopts)で同名グループを**同一ワーカーへ集約**し、
# サブプロセスを起こすテスト同士が並列に走らないようにする。単体・直列実行では無影響。
pytestmark = pytest.mark.xdist_group("subproc_run")


def _load_watchdog():
    spec = importlib.util.spec_from_file_location("watchdog_mod",
                                                  SCRIPTS / "watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WD = _load_watchdog()


# --------------------------------------------------------------------------- #
# ダミー子プロセス(標準ライブラリのみ)。tmp に書き出して watchdog に食わせる。
# --------------------------------------------------------------------------- #
DUMMY_SRC = r'''
import argparse, gzip, os, pickle, sys, time
from pathlib import Path


def latest_step(ckpt_dir):
    if not ckpt_dir.is_dir():
        return 0
    steps = []
    for p in ckpt_dir.glob("ckpt-*.pkl.gz"):
        try:
            steps.append(int(p.name[len("ckpt-"):].split(".")[0]))
        except ValueError:
            pass
    return max(steps) if steps else 0


def write_ckpt(ckpt_dir, step):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("ckpt-%06d.pkl.gz" % step)
    raw = pickle.dumps({"format": 1, "step": step, "pad": b"x" * 64})
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)


def write_part(run_dir, seg):
    (run_dir / ("l1_events.part-%04d.parquet" % seg)).write_bytes(
        b"PAR1-dummy-%d" % seg)


def write_summary(run_dir):
    (run_dir / "summary.json").write_text(
        '{"n_steps": 1, "dummy": true}', encoding="utf-8")


def bump(run_dir):
    f = run_dir / "_attempts.txt"
    n = (int(f.read_text()) if f.exists() else 0) + 1
    f.write_text(str(n))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume", default=None)   # watchdog が resume 時に付与
    ap.add_argument("--mode", required=True)
    ap.add_argument("--crashes", type=int, default=0)
    ap.add_argument("--inc", type=int, default=10)
    ap.add_argument("--n-ckpts", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--final-sleep", type=float, default=0.6)
    ap.add_argument("--stall-sleep", type=float, default=30.0)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    ckpt_dir = run_dir / "checkpoint"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt = bump(run_dir)

    if a.mode == "always_crash":
        sys.exit(1)                              # 進捗なく即死

    if a.mode == "crash_then_ok":
        write_ckpt(ckpt_dir, latest_step(ckpt_dir) + a.inc)  # 毎回1歩進む
        write_part(run_dir, attempt)
        if attempt <= a.crashes:
            sys.exit(1)
        write_summary(run_dir)
        sys.exit(0)

    if a.mode == "stall_then_ok":
        if attempt == 1:
            write_ckpt(ckpt_dir, latest_step(ckpt_dir) + a.inc)
            write_part(run_dir, attempt)
            time.sleep(a.stall_sleep)            # 以後進捗なし → watchdog が kill
            sys.exit(0)                          # (到達しない: kill される)
        write_ckpt(ckpt_dir, latest_step(ckpt_dir) + a.inc)
        write_summary(run_dir)
        sys.exit(0)

    if a.mode == "multi_ckpt":
        cur = latest_step(ckpt_dir)
        for i in range(1, a.n_ckpts + 1):
            write_ckpt(ckpt_dir, cur + a.inc * i)
            write_part(run_dir, i)
            time.sleep(a.sleep)                  # checkpoint を刻んでいく
        time.sleep(a.final_sleep)                # 最後の world をバックアップさせる猶予
        write_summary(run_dir)
        sys.exit(0)

    sys.exit(2)


if __name__ == "__main__":
    main()
'''


def _dummy(tmp_path: Path) -> Path:
    p = tmp_path / "child.py"
    p.write_text(DUMMY_SRC, encoding="utf-8")
    return p


def _cmd_str(dummy: Path) -> str:
    """`--cmd` に渡す基底コマンド。Windows のバックスラッシュ回避に posix パスで。"""
    py = Path(sys.executable).as_posix()
    return f"{shlex.quote(py)} {shlex.quote(dummy.as_posix())}"


def _run_watchdog(run_dir: Path, cmd: str, child: list[str], **opts) -> int:
    argv: list[str] = ["--run-dir", str(run_dir), "--cmd", cmd]
    for k, v in opts.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    argv += ["--"] + child
    ns = WD.parse_args(argv)
    return WD.Watchdog(ns).run()


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# (a) クラッシュ → 自動再開 → 最終成功
# --------------------------------------------------------------------------- #
def test_crash_then_auto_resume_final_success(tmp_path):
    run_dir = tmp_path / "run"
    dummy = _dummy(tmp_path)
    child = ["--run-dir", str(run_dir), "--mode", "crash_then_ok",
             "--crashes", "2", "--inc", "10"]
    rc = _run_watchdog(run_dir, _cmd_str(dummy), child,
                       poll_sec=0.1, stall_min=999, min_uptime_sec=0,
                       backoff_base_sec=0)
    assert rc == 0
    st = _status(run_dir)
    assert st["state"] == "done"
    assert st["restarts"] == 2, st                 # 2回クラッシュ→3回目で成功
    assert (run_dir / "summary.json").exists()
    log = (run_dir / "watchdog.log").read_text(encoding="utf-8")
    assert "resume" in log and "COMPLETE" in log
    # 各再開が最新 checkpoint から進んだ痕跡(step が 10,20,30 と伸びている)
    assert st["last_progress"]["step"] >= 30


# --------------------------------------------------------------------------- #
# (b) ストール検知 → kill → 再開 → 成功
# --------------------------------------------------------------------------- #
def test_stall_detected_kill_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    dummy = _dummy(tmp_path)
    child = ["--run-dir", str(run_dir), "--mode", "stall_then_ok",
             "--inc", "10", "--stall-sleep", "30"]
    rc = _run_watchdog(run_dir, _cmd_str(dummy), child,
                       poll_sec=0.1, stall_min=0.03,   # ~1.8s で停滞判定
                       min_uptime_sec=0, backoff_base_sec=0)
    assert rc == 0
    st = _status(run_dir)
    assert st["state"] == "done"
    assert st["restarts"] >= 1
    log = (run_dir / "watchdog.log").read_text(encoding="utf-8")
    assert "STALL" in log


# --------------------------------------------------------------------------- #
# (c) max-restarts 到達で諦め status=failed
# --------------------------------------------------------------------------- #
def test_max_restarts_exhausted_marks_failed(tmp_path):
    run_dir = tmp_path / "run"
    dummy = _dummy(tmp_path)
    child = ["--run-dir", str(run_dir), "--mode", "always_crash"]
    rc = _run_watchdog(run_dir, _cmd_str(dummy), child,
                       poll_sec=0.05, stall_min=999, max_restarts=3,
                       min_uptime_sec=9999,             # 即死扱い(バックオフ経路)
                       backoff_base_sec=0, backoff_cap_sec=0)
    assert rc == 1
    st = _status(run_dir)
    assert st["state"] == "failed"
    assert st["restarts"] > 3, st                   # 上限超過で諦める
    assert not (run_dir / "summary.json").exists()


# --------------------------------------------------------------------------- #
# (d) バックアップ世代が3つで回る
# --------------------------------------------------------------------------- #
def test_backup_generations_rotate_to_keep(tmp_path):
    run_dir = tmp_path / "run"
    dummy = _dummy(tmp_path)
    child = ["--run-dir", str(run_dir), "--mode", "multi_ckpt",
             "--n-ckpts", "5", "--inc", "10", "--sleep", "0.4",
             "--final-sleep", "0.6"]
    rc = _run_watchdog(run_dir, _cmd_str(dummy), child,
                       poll_sec=0.1, stall_min=999, keep_backups=3,
                       min_uptime_sec=0)
    assert rc == 0
    backup_dir = Path(str(run_dir) + "_backup")
    gens = sorted(backup_dir.glob("gen-*"))
    assert len(gens) == 3, [g.name for g in gens]   # 直近3世代だけ残る
    # 残った世代はいずれも checkpoint が健全に読める
    wd = WD.Watchdog(WD.parse_args(
        ["--run-dir", str(run_dir), "--cmd", _cmd_str(dummy), "--"] + child))
    for g in gens:
        assert wd._verify_ckpt(wd._latest_ckpt(g)) is not None


# --------------------------------------------------------------------------- #
# 実 run.py の煙テスト: mock 20人×48step / checkpoint_every=12 → 正常終了
# --------------------------------------------------------------------------- #
def test_real_run_py_smoke_completes(tmp_path):
    run_dir = tmp_path / "smoke"
    child = [f"run.out_dir={tmp_path.as_posix()}", "run.name=smoke",
             "model.backend=mock", "run.n_agents=20", "run.n_steps=48",
             "observer.checkpoint_every=12"]
    # --cmd を渡さない → 本番の scripts/run.py がそのまま起動される
    argv = ["--run-dir", str(run_dir), "--poll-sec", "0.5"] + ["--"] + child
    rc = WD.Watchdog(WD.parse_args(argv)).run()
    assert rc == 0
    st = _status(run_dir)
    assert st["state"] == "done"
    assert st["restarts"] == 0                       # クラッシュなしで一発完走
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "l1_events.parquet").exists()  # part は finalize で結合済み
    assert not list(run_dir.glob("l1_events.part-*.parquet"))
