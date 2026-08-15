"""本番安全装置(scripts/watchdog.py)のテスト。

実 run.py を使わずに高速化するため、ダミーの子プロセス(child.py)を watchdog に
食わせる。ダミーは checkpoint(gzip+pickle の {format,step})/ l1 part / summary.json を
本番と同じ命名で書き、モードに応じて「N 回クラッシュ後に成功」「進捗せずストール」
「常にクラッシュ」「複数 checkpoint を刻む」を演じる。watchdog は `--cmd` フックで
基底コマンドを差し替えられる(本番経路と同一の監督ループを通す)。

最後に実 run.py での煙テストを1本(mock 20人×48step, checkpoint_every=12)。
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import pickle
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
# (e) ディスク残量ガード(レーンREL 2026-08-12)
#
# ここは**子プロセスを起こさない純粋な単体テスト**。閾値判定は純関数 `disk_state` に
# 切り出してあるので、実際にディスクを埋めずに全分岐を踏める(障害注入 drill の
# 「ダミーファイルでディスクを枯らす」は 8/15 の実機側でやる)。
# --------------------------------------------------------------------------- #
GB = 1 << 30


def _wd(tmp_path, **opts):
    """子プロセスを起こさずに Watchdog インスタンスだけ作る。"""
    argv = ["--run-dir", str(tmp_path / "run")]
    for k, v in opts.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return WD.Watchdog(WD.parse_args(argv + ["--", "noop"]))


def _log(wd) -> str:
    return wd.log_path.read_text(encoding="utf-8")


def test_disk_state_thresholds():
    # 測れない = "unknown"(0 と偽らない)
    assert WD.disk_state(None, 20.0, 5.0) == "unknown"
    assert WD.disk_state(100 * GB, 20.0, 5.0) == "ok"
    assert WD.disk_state(20 * GB, 20.0, 5.0) == "ok"        # 閾値ちょうどは ok
    assert WD.disk_state(19 * GB, 20.0, 5.0) == "warn"
    assert WD.disk_state(5 * GB, 20.0, 5.0) == "warn"       # 致命はちょうどでは出ない
    assert WD.disk_state(4 * GB, 20.0, 5.0) == "critical"
    assert WD.disk_state(0, 20.0, 5.0) == "critical"
    # 閾値 0 = その段を使わない
    assert WD.disk_state(1, 0.0, 0.0) == "ok"
    assert WD.disk_state(1 * GB, 20.0, 0.0) == "warn"
    # 設定ミス(crit >= warn)でも「厳しい方が勝つ」だけで壊れない
    assert WD.disk_state(10 * GB, 5.0, 20.0) == "critical"


def test_disk_args_defaults_and_overrides(tmp_path):
    ns = WD.parse_args(["--run-dir", str(tmp_path)])
    assert (ns.disk_warn_gb, ns.disk_crit_gb, ns.disk_check_min) == (20.0, 5.0, 5.0)
    ns = WD.parse_args(["--run-dir", str(tmp_path), "--disk-warn-gb", "200",
                        "--disk-crit-gb", "50", "--disk-check-min", "0"])
    assert (ns.disk_warn_gb, ns.disk_crit_gb, ns.disk_check_min) == (200.0, 50.0, 0.0)
    wd = _wd(tmp_path, disk_warn_gb=200, disk_crit_gb=50, disk_check_min=0)
    assert wd.disk_warn_gb == 200.0 and wd.disk_crit_gb == 50.0
    assert wd.disk_check_sec == 0.0                        # 0 = 監視しない


def test_disk_free_bytes_uses_existing_ancestor(tmp_path, monkeypatch):
    """まだ作られていない backup-dir でも、実在する祖先で測れる。"""
    deep = tmp_path / "not" / "yet" / "there"
    assert WD._existing_ancestor(deep) == tmp_path
    assert WD.disk_free_bytes(deep) == WD.disk_free_bytes(tmp_path)
    assert WD.disk_free_bytes(deep) > 0

    def _boom(_p):                                 # 測れない環境(権限・特殊 FS 等)
        raise OSError("nope")

    monkeypatch.setattr(WD.shutil, "disk_usage", _boom)
    assert WD.disk_free_bytes(deep) is None        # 0 と偽らない


def test_check_disk_logs_one_line_and_stays_quiet_when_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 100 * GB)
    wd = _wd(tmp_path)
    wd.check_disk()
    lines = [ln for ln in _log(wd).splitlines() if "disk " in ln]
    assert len(lines) == 1, lines
    assert "free=100.0GB" in lines[0] and "state=ok" in lines[0]
    assert "[watchdog]" not in capsys.readouterr().out    # 正常時はコンソールに出さない


def test_check_disk_warns_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 10 * GB)
    wd = _wd(tmp_path)
    wd.check_disk()
    log = _log(wd)
    assert "state=warn" in log
    assert "WARN disk" in log
    assert "DISK CRITICAL" not in log
    assert wd.last_disk["state"] == "warn" and wd.last_disk["free_gb"] == 10.0


def test_check_disk_critical_is_emphasised_before_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 2 * GB)
    wd = _wd(tmp_path)
    wd.check_disk(tag="backup gen step=144")
    log = _log(wd)
    assert "*** DISK CRITICAL ***" in log
    assert "backup gen step=144 の直前" in log
    assert "ランは止めない" in log                          # 止める判断は人間に残す


def test_do_backup_checks_disk_first(tmp_path, monkeypatch):
    """世代コピー(run 中で最大の書き込み)の直前に必ず 1 回測る。"""
    seen = []
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: seen.append(p) or (3 * GB))
    wd = _wd(tmp_path)
    ck = wd.run_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "ckpt-000010.pkl.gz").write_bytes(b"dummy")
    wd._do_backup(10)
    assert seen, "backup 前に残量を測っていない"
    log = _log(wd)
    assert "*** DISK CRITICAL ***" in log and "backup gen step=10" in log
    assert "backup gen created" in log                     # 警告だけ・バックアップは続く
    assert len(list(wd.backup_dir.glob("gen-*"))) == 1


def test_disk_unknown_is_logged_once_then_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: None)
    wd = _wd(tmp_path)
    wd.check_disk()
    wd.check_disk()
    assert _log(wd).count("free=unknown") == 1
    assert wd.last_disk["state"] == "unknown" and wd.last_disk["free_gb"] is None


def test_disk_targets_split_when_backup_is_on_another_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 50 * GB)
    wd = _wd(tmp_path)
    assert len(wd._disk_targets()) == 1                     # 同一ボリューム = 1 本
    monkeypatch.setattr(WD, "_volume_key",
                        lambda p: "R" if "backup" in str(p) else "L")
    assert [lbl for lbl, _ in wd._disk_targets()] == ["run", "backup"]
    wd.check_disk()
    assert _log(wd).count("state=ok") == 2
    assert len(wd.last_disk["targets"]) == 2
    # 消費側が常に読める形(run ボリュームの値が最上位)であること
    assert wd.last_disk["label"] == "run" and wd.last_disk["free_gb"] == 50.0


def test_maybe_check_disk_respects_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 100 * GB)
    wd = _wd(tmp_path, disk_check_min=1)                    # 60s 間隔
    wd._maybe_check_disk(1000.0)
    wd._maybe_check_disk(1001.0)                            # 間隔内 = 測らない
    assert _log(wd).count("disk run=") == 1
    wd._maybe_check_disk(1000.0 + 61.0)
    assert _log(wd).count("disk run=") == 2


def test_disk_monitoring_off_writes_nothing(tmp_path, monkeypatch):
    """--disk-check-min 0 = 残量監視を丸ごと切る(世代バックアップ直前の点検も黙る)。"""
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 1 * GB)
    wd = _wd(tmp_path, disk_check_min=0)
    wd._maybe_check_disk(1000.0)
    ck = wd.run_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "ckpt-000010.pkl.gz").write_bytes(b"dummy")
    wd._do_backup(10)
    log = _log(wd)
    assert "disk " not in log and "DISK CRITICAL" not in log
    assert "backup gen created" in log             # バックアップ自体は普通に走る


def test_status_json_carries_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(WD, "disk_free_bytes", lambda p: 7 * GB)
    wd = _wd(tmp_path)
    wd.check_disk()
    wd.write_status("running", pid=123)
    st = _status(wd.run_dir)
    assert st["disk"]["state"] == "warn"
    assert st["disk"]["free_gb"] == 7.0
    assert st["disk"]["warn_gb"] == 20.0 and st["disk"]["crit_gb"] == 5.0


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


# --------------------------------------------------------------------------- #
# 停滞判定の本選値化(第117 レーンB3 β2)
#
#   既定は開発用の小ラン向け(20 分固定)。25万体では 1 step が checkpoint 粒度より長く
#   なりうるので「1 step の実測秒 × 係数」と連動できる引数を足した。**既定値は 1 つも
#   変えていない**(--stall-step-sec を渡さなければ従来と 1 秒も変わらない)。
# --------------------------------------------------------------------------- #
def test_stall_seconds_defaults_to_stall_min_only():
    assert WD.stall_seconds(20.0) == 1200.0
    assert WD.stall_seconds(20.0, 0.0) == 1200.0
    assert WD.stall_seconds(20.0, 0.0, 6.0) == 1200.0
    assert WD.stall_seconds(20.0, 90.0, 0.0) == 1200.0     # 係数 0 = 連動を切る


def test_stall_seconds_takes_the_larger_of_the_two():
    """必ず「待つ側」へ倒れる(短い実測で本選を誤 kill しない)。"""
    assert WD.stall_seconds(20.0, 90.0, 6.0) == 1200.0     # 90*6=540 < 1200 → 20 分側
    assert WD.stall_seconds(20.0, 600.0, 6.0) == 3600.0    # 600*6=3600 > 1200 → 1 step 側
    assert WD.stall_seconds(0.0, 600.0, 6.0) == 3600.0


def test_stall_args_defaults_are_unchanged_and_overridable(tmp_path):
    ns = WD.parse_args(["--run-dir", str(tmp_path)])
    assert (ns.stall_min, ns.stall_step_sec, ns.stall_factor) == (20.0, 0.0, 6.0)
    assert _wd(tmp_path).stall_sec == 1200.0               # 既定は従来どおり
    wd = _wd(tmp_path, stall_step_sec=600, stall_factor=6)
    assert wd.stall_sec == 3600.0
    wd = _wd(tmp_path, stall_min=90)                       # 単独指定も従来どおり
    assert wd.stall_sec == 5400.0


def test_help_carries_the_finals_recommendations():
    """本選推奨値(ディスク 50/20・stall 連動・ckpt-generations 999)が読める場所にある。

    ★「既定値は変えない代わりにヘルプと手順書へ書く」が β2 の契約なので、
      **その文言そのもの**を機械固定する(消えたら赤くなる)。
    """
    scripts = Path(WD.__file__).parent
    doc = WD.__doc__ or ""
    assert "--disk-warn-gb 50" in doc and "--disk-crit-gb 20" in doc
    assert "--ckpt-generations 999" in doc
    assert "--stall-step-sec" in doc
    parser_src = Path(WD.__file__).read_text(encoding="utf-8")
    assert "★本選推奨 50" in parser_src and "★本選推奨 20" in parser_src
    backup_src = (scripts / "backup_run.py").read_text(encoding="utf-8")
    assert "本選は必ず 999" in backup_src
    checklist = (scripts.parent / "ops" / "finals-compute-checklist.md"
                 ).read_text(encoding="utf-8")
    assert "--stall-step-sec" in checklist and "--disk-warn-gb 50" in checklist


# --------------------------------------------------------------------------- #
# COMPLETE マーカー(第117 β7)の取り回し(第117 レーンB3 追加発注2)
#
#   マーカーの意味は「この本体は最後まで書けた」。したがって**本体と運命を共にする**の
#   が唯一の正しい規約で、片方だけ残すと「本体の無い完了印」= 孤児マーカーができる。
#   ★`CKPT_GLOB` は `.pkl.gz` 終端なので進捗検知・`_latest_ckpt` はマーカーに掛からない
#     (= 既存挙動は無風)。明示的に扱うのは隔離と一掃の 2 箇所だけ。
# --------------------------------------------------------------------------- #
def _put_ckpt(run_dir: Path, step: int, *, marker: bool = True) -> Path:
    d = run_dir / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"ckpt-{step:06d}.pkl.gz"
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps({"format": 1, "step": step}))
    if marker:
        (d / (p.name + WD.COMPLETE_SUFFIX)).write_bytes(b"")
    return p


def test_rollback_quarantines_the_marker_with_its_body(tmp_path):
    """★本体を corrupt/ へ隔離したら、その COMPLETE マーカーも一緒に降りる。"""
    wd = _wd(tmp_path)
    run_dir = wd.run_dir
    _put_ckpt(run_dir, 100)
    _put_ckpt(run_dir, 200)
    ck = run_dir / "checkpoint"
    wd._rollback_one_generation(200)               # 200 を疑って隔離する
    quar = ck / "corrupt"
    assert (quar / "ckpt-000200.pkl.gz").exists(), "本体が隔離されていない(前提が崩れた)"
    assert (quar / "ckpt-000200.pkl.gz.complete").exists(), \
        "★マーカーが隔離されず run-dir に取り残された(孤児マーカー)"
    assert not (ck / "ckpt-000200.pkl.gz.complete").exists()
    # 巻き戻し先(100)は本体もマーカーも無傷
    assert (ck / "ckpt-000100.pkl.gz").exists()
    assert (ck / "ckpt-000100.pkl.gz.complete").exists()


def test_rollback_without_a_marker_still_works(tmp_path):
    """★後方互換: マーカーの無い(第117 以前の)checkpoint でも従来どおり隔離できる。"""
    wd = _wd(tmp_path)
    ck = wd.run_dir / "checkpoint"
    _put_ckpt(wd.run_dir, 100, marker=False)
    _put_ckpt(wd.run_dir, 200, marker=False)
    wd._rollback_one_generation(200)
    assert (ck / "corrupt" / "ckpt-000200.pkl.gz").exists()
    assert not list((ck / "corrupt").glob("*" + WD.COMPLETE_SUFFIX))
    assert "quarantined suspect checkpoint" in _log(wd)


def test_clear_run_state_leaves_no_orphan_markers(tmp_path):
    """本体を消すときはマーカーも消す(孤児マーカーを積み上げない)。"""
    wd = _wd(tmp_path)
    ck = wd.run_dir / "checkpoint"
    _put_ckpt(wd.run_dir, 100)
    _put_ckpt(wd.run_dir, 200)
    wd._clear_run_state()
    assert not list(ck.glob("ckpt-*.pkl.gz"))
    assert not list(ck.glob("*" + WD.COMPLETE_SUFFIX)), \
        "★本体を消したのにマーカーが残った(孤児マーカー)"


def test_marker_is_invisible_to_progress_and_latest(tmp_path):
    """マーカーは進捗指紋にも `_latest_ckpt` にも掛からない(既存挙動が無風)。"""
    wd = _wd(tmp_path)
    _put_ckpt(wd.run_dir, 100, marker=False)
    fp_before = wd._progress_fp()
    latest_before = wd._latest_ckpt()
    (wd.run_dir / "checkpoint" / "ckpt-000100.pkl.gz.complete").write_bytes(b"")
    assert wd._progress_fp() == fp_before, "マーカーを置いただけで進捗が動いた"
    assert wd._latest_ckpt() == latest_before
    # ★不変量の在処: `_ckpt_step` は名前の**前半だけ**を見るのでマーカーを渡せば 100 を
    #   返してしまう(下)。守っているのは「マーカーがそこへ渡らない」= 候補を作る glob が
    #   `.pkl.gz` 終端であること。ここを緩めると隔離・剪定の対象が静かに広がる。
    assert WD.CKPT_GLOB.endswith(".pkl.gz")
    assert not list((wd.run_dir / "checkpoint").glob(WD.CKPT_GLOB)) == []
    assert all(not p.name.endswith(WD.COMPLETE_SUFFIX)
               for p in (wd.run_dir / "checkpoint").glob(WD.CKPT_GLOB))
