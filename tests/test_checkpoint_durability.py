"""β7(第117バッチ 2026-08-16。レーン B1 安全系): checkpoint の stream 書き + COMPLETE。

正典: docs/plans/beta-implementation-plan.md §1 β7 / docs/plans/external-audit-triage.md F3・F4。

受入基準:
  - `pickle.dumps` の中間バイト列を作らず gzip ストリームへ直接書いても **往復は同一**
    (共有参照の保存契約 = labels.items ⇄ sim.items が同一実体、も維持される)
  - 書き終わりに fsync + 既存の原子的 rename(tmp を残さない)
  - rename 後に `<名前>.complete` マーカーを置く
  - `latest()` は **マーカーのある世代だけ**を最新候補にする(書きかけを掴まない)
  - **後方互換**: マーカーが 1 つも無いディレクトリは従来どおり全候補を許す
  - 書きかけの世代が混ざっていても resume == straight
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation


def _cfg(name: str, n_steps: int, **ov):
    dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _sim_at(tmp_path, name: str, steps: int, n_steps: int | None = None):
    """steps 回まわした Simulation を返す(checkpoint の材料)。"""
    sim = Simulation(_cfg(name, n_steps or steps), out_dir=tmp_path / name)
    for step in range(steps):
        scheduler.run_step(sim, step)
    return sim


# --------------------------------------------------------------------------- #
# (1) 往復同一(dumps → dump へ変えても契約が変わらない)
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_is_identical(tmp_path):
    sim = _sim_at(tmp_path, "rt", 12)
    path = checkpoint.save(sim, 12, tmp_path / "rt" / "checkpoint" / "ckpt-000012.pkl.gz")
    fresh = Simulation(_cfg("rt2", 24), out_dir=tmp_path / "rt2")
    assert checkpoint.load(fresh, path) == 12
    assert [a.id for a in fresh.agents] == [a.id for a in sim.agents]
    assert [a.money for a in fresh.agents] == [a.money for a in sim.agents]


def test_shared_references_survive_one_pickle_call(tmp_path):
    """docstring の契約: labels 内の ItemStore と sim.items が **同一実体**で戻る。"""
    sim = _sim_at(tmp_path, "shared", 8)
    path = checkpoint.save(sim, 8, tmp_path / "shared" / "checkpoint" / "ckpt-000008.pkl.gz")
    fresh = Simulation(_cfg("shared2", 16), out_dir=tmp_path / "shared2")
    checkpoint.load(fresh, path)
    assert fresh.items is fresh.labels.items, "共有参照が 2 実体に割れている"


def test_written_file_is_still_gzip_pickle(tmp_path):
    """既存の読み手(tests・解析)が使う gzip+pickle の読み方がそのまま通る。"""
    sim = _sim_at(tmp_path, "fmt", 6)
    path = checkpoint.save(sim, 6, tmp_path / "fmt" / "checkpoint" / "ckpt-000006.pkl.gz")
    with gzip.open(path, "rb") as f:
        blob = pickle.loads(f.read())          # 旧来の読み方(全読み)
    assert blob["format"] == checkpoint.FORMAT and blob["step"] == 6
    with gzip.open(path, "rb") as f:
        assert pickle.load(f)["step"] == 6     # ストリーム読み(新しい load と同じ)


# --------------------------------------------------------------------------- #
# (2) 原子性 + COMPLETE マーカー
# --------------------------------------------------------------------------- #
def test_complete_marker_and_no_tmp_left(tmp_path):
    sim = _sim_at(tmp_path, "mk", 6)
    ck = tmp_path / "mk" / "checkpoint"
    path = checkpoint.save(sim, 6, ck / "ckpt-000006.pkl.gz")
    marker = checkpoint.complete_marker(path)
    assert marker.name == "ckpt-000006.pkl.gz.complete"
    assert marker.exists() and marker.stat().st_size == 0, "空マーカーであること"
    assert list(ck.glob("*.tmp")) == [], "書きかけ tmp が残っている"


def test_marker_does_not_break_existing_globs(tmp_path):
    """watchdog / report_progress / backup_run の既存 glob に掛からない命名であること。"""
    sim = _sim_at(tmp_path, "gl", 6)
    ck = tmp_path / "gl" / "checkpoint"
    checkpoint.save(sim, 6, ck / "ckpt-000006.pkl.gz")
    assert [p.name for p in ck.glob("ckpt-*.pkl.gz")] == ["ckpt-000006.pkl.gz"]


# --------------------------------------------------------------------------- #
# (3) latest(): マーカー無しを最新候補から外す / 後方互換
# --------------------------------------------------------------------------- #
def test_latest_skips_unmarked_generation(tmp_path):
    sim = _sim_at(tmp_path, "sel", 6)
    ck = tmp_path / "sel" / "checkpoint"
    good = checkpoint.save(sim, 6, ck / "ckpt-000006.pkl.gz")
    # 書き込み途中で死んだ「より新しい世代」を捏造する(マーカーが無い)
    (ck / "ckpt-000012.pkl.gz").write_bytes(b"\x1f\x8b\x08\x00 torn")
    assert checkpoint.latest(tmp_path / "sel") == good


def test_latest_is_backward_compatible_without_any_marker(tmp_path):
    """マーカーが 1 つも無い(第117 以前の)ディレクトリは従来どおり step 最大を返す。"""
    ck = tmp_path / "old" / "checkpoint"
    ck.mkdir(parents=True)
    for step in (6, 12):
        (ck / f"ckpt-{step:06d}.pkl.gz").write_bytes(b"x")
    assert checkpoint.latest(tmp_path / "old").name == "ckpt-000012.pkl.gz"


def test_latest_none_when_empty(tmp_path):
    assert checkpoint.latest(tmp_path / "nothing") is None
    (tmp_path / "empty" / "checkpoint").mkdir(parents=True)
    assert checkpoint.latest(tmp_path / "empty") is None


# --------------------------------------------------------------------------- #
# (4) end-to-end: 書きかけが混ざっていても resume == straight
# --------------------------------------------------------------------------- #
def test_resume_ignores_torn_checkpoint_and_matches_straight(tmp_path):
    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 40), out_dir=straight).run()

    d = tmp_path / "resumed"
    every = {"observer.checkpoint_every": 20}
    sim1 = Simulation(_cfg("resumed", 20, **every), out_dir=d)
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, d / "checkpoint" / "ckpt-000020.pkl.gz")
    sim1.logger.flush_segment()
    # ★プロセスが死んだ瞬間の「書きかけの新しい世代」= 中身は壊れていてマーカーも無い。
    #   β7 以前はこちらが最新候補として選ばれ、resume が gzip エラーで即死していた。
    (d / "checkpoint" / "ckpt-000030.pkl.gz").write_bytes(b"\x1f\x8b\x08\x00 torn")

    sim2 = Simulation(_cfg("resumed", 40, **every), out_dir=d)
    sim2.run(resume_from=d)
    assert _rows(d) == _rows(straight), "resume の L1 が straight と不一致"


def test_torn_checkpoint_without_guard_would_fail(tmp_path):
    """反証: マーカー規則が無ければ壊れた世代を掴んで落ちる(規則が効いている証明)。"""
    ck = tmp_path / "proof" / "checkpoint"
    ck.mkdir(parents=True)
    torn = ck / "ckpt-000030.pkl.gz"
    torn.write_bytes(b"\x1f\x8b\x08\x00 torn")
    sim = Simulation(_cfg("proof", 40), out_dir=tmp_path / "proof")
    with pytest.raises((gzip.BadGzipFile, EOFError, OSError, pickle.UnpicklingError)):
        checkpoint.load(sim, torn)             # マーカー無しを直に読ませればこうなる


# =========================================================================== #
# A6(第118 レーンC): COMPLETE マーカーを**世代トランザクションの最後**へ
#
#   1 世代 = ① checkpoint 本体 ② pool サイドカー(pool ON) ③ 全 flush_segment。
#   旧順序は①の直後にマーカーを置いていたので、「マーカーは在るが②③が無い」世代が
#   生まれ得た(= latest() が完成扱いで掴み、L1 が最大 checkpoint 間隔ぶん欠落する)。
#   ここでは engine の順序そのものと、障害注入 3 窓(①後 / マーカー後 / 全完了)を固定する。
# =========================================================================== #
def test_save_without_complete_flag_leaves_no_marker(tmp_path):
    """`save(..., complete=False)` は本体だけを置く(= まだ latest() の候補にならない)。

    ★既存の後方互換規則(マーカーが**1 つも無い**ディレクトリは全候補を許す)は
      1 バイトも変えていないので、判定は「完成した世代が 1 つでもある dir」で見る。
    """
    sim = _sim_at(tmp_path, "nc", 12)
    ck = tmp_path / "nc" / "checkpoint"
    old = checkpoint.save(sim, 6, ck / "ckpt-000006.pkl.gz")            # 完成した世代
    path = checkpoint.save(sim, 12, ck / "ckpt-000012.pkl.gz", complete=False)
    assert path.exists(), "本体が置かれていない"
    assert not checkpoint.complete_marker(path).exists(), "マーカーが書かれている"
    assert checkpoint.latest(tmp_path / "nc") == old, "未コミット世代を latest が拾った"
    assert checkpoint.complete_generations(tmp_path / "nc") == [old]
    # 明示コミットで初めて候補になる(= 世代のコミット点が 1 点に集まっている)
    checkpoint.write_complete_marker(path)
    assert checkpoint.latest(tmp_path / "nc") == path
    assert checkpoint.complete_generations(tmp_path / "nc") == [old, path]


def test_engine_commits_the_marker_after_the_flush(tmp_path, monkeypatch):
    """★順序の直接証明: マーカーが書かれる瞬間、その世代の part は**もう在る**。

    旧順序(本体 rename → マーカー → sidecar → flush)ではここが空になる = このテストは
    順序が戻れば必ず落ちる(空回りしない)。
    """
    seen: list[list[str]] = []
    real = checkpoint.write_complete_marker

    def _spy(path):
        out_dir = Path(path).parent.parent
        seen.append(sorted(p.name for p in out_dir.glob("l1_events.part-*.parquet")))
        return real(path)

    monkeypatch.setattr(checkpoint, "write_complete_marker", _spy)
    d = tmp_path / "order"
    Simulation(_cfg("order", 40, **{"observer.checkpoint_every": 20}),
               out_dir=d).run()
    assert len(seen) == 2, f"checkpoint が 2 世代ぶん走っていない: {seen}"
    assert seen[0] == ["l1_events.part-0000.parquet"], \
        "1 世代目のマーカーが flush より前に書かれている"
    assert seen[1] == ["l1_events.part-0000.parquet", "l1_events.part-0001.parquet"], \
        "2 世代目のマーカーが flush より前に書かれている"


def test_resume_falls_back_from_an_uncommitted_generation(tmp_path):
    """障害注入①: 本体は完全に書けたがマーカー前に停止 → **一つ前の完全世代**から再開。

    ★`test_latest_skips_unmarked_generation` との違い: あちらの新しい世代は**壊れて**
      いるが、こちらは中身が完全に正しい checkpoint である。それでも「まだコミット
      されていない」= sidecar も part も無い世代なので掴んではいけない。
    """
    straight = tmp_path / "unc_straight"
    Simulation(_cfg("unc_straight", 40), out_dir=straight).run()

    d = tmp_path / "unc"
    every = {"observer.checkpoint_every": 20}
    sim1 = Simulation(_cfg("unc", 20, **every), out_dir=d)
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, d / "checkpoint" / "ckpt-000020.pkl.gz")   # 完成した世代
    sim1.logger.flush_segment()
    for step in range(20, 30):                       # ★次の世代の本体だけ書いて停止
        scheduler.run_step(sim1, step)
    uncommitted = checkpoint.save(sim1, 30, d / "checkpoint" / "ckpt-000030.pkl.gz",
                                  complete=False)
    assert uncommitted.exists() and not checkpoint.complete_marker(uncommitted).exists()
    assert checkpoint.latest(d).name == "ckpt-000020.pkl.gz", "未コミット世代を選んだ"

    sim2 = Simulation(_cfg("unc", 40, **every), out_dir=d)
    assert sim2._pick_resume_checkpoint(d).name == "ckpt-000020.pkl.gz"
    sim2.run(resume_from=d)
    assert _rows(d) == _rows(straight), "未コミット世代へ戻った resume が straight と不一致"


def test_resume_uses_the_generation_once_it_is_committed(tmp_path):
    """障害注入③(対照): 全部書き終わった世代は、その世代からそのまま再開する。"""
    straight = tmp_path / "cmt_straight"
    Simulation(_cfg("cmt_straight", 40), out_dir=straight).run()

    d = tmp_path / "cmt"
    every = {"observer.checkpoint_every": 10}
    Simulation(_cfg("cmt", 30, **every), out_dir=d).run()   # 30 step で正常終了
    marks = sorted(p.name for p in (d / "checkpoint").glob("*.complete"))
    assert marks == ["ckpt-000010.pkl.gz.complete", "ckpt-000020.pkl.gz.complete",
                     "ckpt-000030.pkl.gz.complete"], f"世代のコミットが欠けている: {marks}"

    sim2 = Simulation(_cfg("cmt", 40, **every), out_dir=d)
    assert sim2._pick_resume_checkpoint(d).name == "ckpt-000030.pkl.gz", \
        "完全世代なのに前へ戻った(過剰な後退)"
    sim2.run(resume_from=d)
    assert _rows(d) == _rows(straight)


def test_resume_refuses_when_later_segments_are_already_on_disk(tmp_path):
    """世代より**先**の part が残っていれば落とす(黙って L1 を二重にしない)。

    マーカーを最後に置く順序では「flush は済んだがマーカー前に停止」した世代が
    ありうる。そこで一つ前へ戻ると、確定済みの区間を再実行して finalize が同じ
    イベントを 2 回結合してしまう。checkpoint が運ぶ `log_seg` でこれを検出する。
    """
    d = tmp_path / "seg"
    every = {"observer.checkpoint_every": 20}
    sim1 = Simulation(_cfg("seg", 20, **every), out_dir=d)
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, d / "checkpoint" / "ckpt-000020.pkl.gz")
    sim1.logger.flush_segment()                       # part-0000(= 世代 20 の対)
    for step in range(20, 30):                        # ★次の世代は flush まで済んで
        scheduler.run_step(sim1, step)                #   マーカー前に停止した、という状態
    sim1.logger.flush_segment()
    assert sorted(p.name for p in d.glob("l1_events.part-*.parquet")) == [
        "l1_events.part-0000.parquet", "l1_events.part-0001.parquet"]

    sim2 = Simulation(_cfg("seg", 40, **every), out_dir=d)
    with pytest.raises(RuntimeError) as exc:
        sim2.run(resume_from=d)
    assert "part-0001" in str(exc.value) and "二重" in str(exc.value)
