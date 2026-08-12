"""日次バックアップ(scripts/backup_run.py)の検収。レーンREL 2026-08-12。

合成ランディレクトリ(実 parquet + gzip pickle の checkpoint)だけで完結する。
シミュ本体も実 LLM も使わない。固定するのは計画書 §3 が要求する 5 点:

  ① **確定 part の判別**   … footer が閉じていない書きかけ part を掴まない
  ② **増分性**             … 2 回目は差分ゼロで dest に 1 バイトも書かない
  ③ **マニフェスト照合**   … BagIt 式 manifest-sha256.txt が実体と一致し --verify が通る
  ④ **書き込み中のスキップ** … 未確定 part は tar に入らない(理由付きで記録される)
  ⑤ **copy 系のみ**        … 元を消してもバックアップからは消えない

★ ①と④は別物として書き分けている: ① は「footer が壊れている」= parquet 形式の検証、
④ は「checkpoint 境界より後にできた」= watchdog の巻き戻しで**中身が変わりうる**期間の
除外。前者はデータの完結性、後者は不変性の話で、落ちる理由が違う。
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_backup_run():
    spec = importlib.util.spec_from_file_location("backup_run_mod",
                                                  SCRIPTS / "backup_run.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclass は cls.__module__ を sys.modules 経由で引くので、exec 前に登録が要る
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


BR = _load_backup_run()


# --------------------------------------------------------------------------- #
# 合成ラン
# --------------------------------------------------------------------------- #
def _write_part(path: Path, n: int = 8) -> None:
    """本物の parquet(footer 込み)を書く。"""
    tbl = pa.table({"step": pa.array(range(n), pa.int64()),
                    "kind": pa.array([f"k{i}" for i in range(n)])})
    pq.write_table(tbl, path)


def _write_torn_part(path: Path) -> None:
    """**書き込み中**の part を模す: 実 parquet の末尾(footer)を削り落とす。"""
    _write_part(path, n=8)
    raw = path.read_bytes()
    path.write_bytes(raw[:-40])                    # footer 長 + 末尾 magic を破壊


def _write_ckpt(run_dir: Path, step: int, *, dormant: bool = True) -> None:
    d = run_dir / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    blob = {"format": 1, "step": step, "pad": b"x" * 128}
    with gzip.open(d / f"ckpt-{step:06d}.pkl.gz", "wb") as f:
        f.write(pickle.dumps(blob))
    if dormant:                                    # pool サイドカー(同 step で 1 世代)
        with gzip.open(d / f"dormant-{step:06d}.pkl.gz", "wb") as f:
            f.write(pickle.dumps({"pool_day": step // 144, "dormant": {}}))


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def _make_run(tmp_path: Path, name: str = "finals") -> Path:
    """走行中のランを模した合成ディレクトリ。

    時系列は mtime で作る: 確定 part(t0)→ checkpoint(t1)→ 未確定 part(t2)。
    """
    run = tmp_path / name
    (run / "checkpoint").mkdir(parents=True, exist_ok=True)
    t0 = time.time() - 3600
    t1 = t0 + 600
    t2 = t1 + 600

    for i in (0, 1):
        p = run / f"l1_events.part-{i:04d}.parquet"
        _write_part(p)
        _touch(p, t0)
    for stem in ("l1b_llm", "l2_metrics"):
        p = run / f"{stem}.part-0000.parquet"
        _write_part(p, n=4)
        _touch(p, t0)

    _write_ckpt(run, 72)
    for p in (run / "checkpoint").iterdir():
        _touch(p, t1)
    _write_ckpt(run, 0)                            # 古い世代(既定 2 世代なので残る)
    for p in (run / "checkpoint").glob("*-000000.pkl.gz"):
        _touch(p, t0)

    # checkpoint より後にできた part(= 未確定。中身は正しい parquet)
    p = run / "l1_events.part-0002.parquet"
    _write_part(p)
    _touch(p, t2)
    # 書きかけ(footer が無い)part
    _write_torn_part(run / "l1_events.part-0003.parquet")
    _touch(run / "l1_events.part-0003.parquet", t2)

    (run / "config.yaml").write_text("run:\n  name: finals\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"git": "deadbeef"}),
                                           encoding="utf-8")
    (run / "llm_cache.jsonl").write_text('{"k":"v"}\n', encoding="utf-8")
    (run / "run.out.log").write_text("child stdout\n" * 100, encoding="utf-8")
    (run / "heatmap.html").write_text("<html></html>", encoding="utf-8")
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "derived.json").write_text("{}", encoding="utf-8")
    return run


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """dest ツリーの全ファイルを {相対パス: (サイズ, sha256)} で。"""
    out: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = (
                p.stat().st_size,
                hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def _tar_members(tar_path: Path) -> set[str]:
    with tarfile.open(tar_path, "r|") as t:
        return {i.name for i in t if i.isfile()}


# --------------------------------------------------------------------------- #
# ① 確定 part の判別(footer 検証)+ ④ 書き込み中のスキップ
# --------------------------------------------------------------------------- #
def test_selects_only_confirmed_parts(tmp_path):
    run = _make_run(tmp_path)
    items, skips = BR.select_payload(run)
    rels = {i.rel for i in items}
    reasons = {s.rel: s.reason for s in skips}

    assert "l1_events.part-0000.parquet" in rels
    assert "l1_events.part-0001.parquet" in rels
    # ④ checkpoint より後 = 巻き戻しで書き換わりうるので取らない
    assert "l1_events.part-0002.parquet" not in rels
    assert "after-checkpoint-boundary" in reasons["l1_events.part-0002.parquet"]
    # ① footer が閉じていない = 書き込み中とみなす
    assert "l1_events.part-0003.parquet" not in rels
    assert "footer-incomplete" in reasons["l1_events.part-0003.parquet"]

    # checkpoint は直近 2 世代(step 0/72)の ckpt- と dormant- が対で入る
    assert {"checkpoint/ckpt-000072.pkl.gz", "checkpoint/dormant-000072.pkl.gz",
            "checkpoint/ckpt-000000.pkl.gz", "checkpoint/dormant-000000.pkl.gz"} <= rels
    # サイドカー・設定・台帳は同梱、派生物と巨大ログは除外
    assert {"config.yaml", "run_manifest.json", "llm_cache.jsonl"} <= rels
    assert "run.out.log" not in rels and "heatmap.html" not in rels
    assert "analysis/" in reasons and "derived-dir" in reasons["analysis/"]


def test_torn_parquet_never_reaches_the_archive(tmp_path):
    """④: 書きかけ part は tar のメンバにも累積マニフェストにも現れない。"""
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    rep = BR.backup(run, dest, quiet=True)
    members = _tar_members(dest / "finals" / BR.INCREMENTS_DIR / rep["increment"])
    assert "data/l1_events.part-0003.parquet" not in members
    assert "data/l1_events.part-0002.parquet" not in members
    manifest = (dest / "finals" / BR.MANIFEST_NAME).read_text(encoding="utf-8")
    assert "part-0003" not in manifest and "part-0002" not in manifest
    assert not rep["errors"]


def test_only_one_generation_kept_when_asked(tmp_path):
    run = _make_run(tmp_path)
    items, skips = BR.select_payload(run, ckpt_generations=1)
    rels = {i.rel for i in items}
    assert "checkpoint/ckpt-000072.pkl.gz" in rels
    assert "checkpoint/ckpt-000000.pkl.gz" not in rels
    assert any(s.rel == "checkpoint/ckpt-000000.pkl.gz"
               and "older-generation" in s.reason for s in skips)


def test_completed_run_drops_the_boundary_requirement(tmp_path):
    """完走ラン(summary.json あり)は書き手が居ないので境界を課さない。"""
    run = _make_run(tmp_path)
    (run / "summary.json").write_text('{"n_steps": 144}', encoding="utf-8")
    rels = {i.rel for i in BR.select_payload(run)[0]}
    assert "l1_events.part-0002.parquet" in rels          # 境界より後でも確定扱い
    assert "l1_events.part-0003.parquet" not in rels      # footer 不正は依然 NG


# --------------------------------------------------------------------------- #
# ② 増分性(2 回目は差分ゼロ・何も書かない)
# --------------------------------------------------------------------------- #
def test_second_run_is_zero_diff_and_writes_nothing(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    first = BR.backup(run, dest, quiet=True)
    assert first["transferred"] > 0 and not first["zero_diff"]

    before = _snapshot(dest)
    second = BR.backup(run, dest, quiet=True)
    after = _snapshot(dest)

    assert second["zero_diff"] is True
    assert second["transferred"] == 0
    assert second["increment"] is None
    assert after == before, "差分ゼロなのに dest が変化した"


def test_new_part_after_next_checkpoint_ships_alone(tmp_path):
    """増分の実体: 2 回目は**新しく確定した 1 本だけ**が tar に入る。"""
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    BR.backup(run, dest, quiet=True)

    # 次の checkpoint が打たれ、part-0002 が境界の内側に入った
    now = time.time()
    _write_ckpt(run, 144)
    for p in (run / "checkpoint").glob("*-000144.pkl.gz"):
        _touch(p, now)

    rep = BR.backup(run, dest, quiet=True)
    tar = dest / "finals" / BR.INCREMENTS_DIR / rep["increment"]
    payload = {m[len("data/"):] for m in _tar_members(tar) if m.startswith("data/")}
    assert payload == {"l1_events.part-0002.parquet",
                       "checkpoint/ckpt-000144.pkl.gz",
                       "checkpoint/dormant-000144.pkl.gz"}
    assert len(json.loads((dest / "finals" / BR.STATE_NAME)
                          .read_text(encoding="utf-8"))["increments"]) == 2


def test_dry_run_writes_nothing(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    rep = BR.backup(run, dest, dry_run=True, quiet=True)
    assert rep["transferred"] > 0
    assert not dest.exists() or _snapshot(dest) == {}


# --------------------------------------------------------------------------- #
# ③ マニフェスト照合(BagIt 式)+ --verify
# --------------------------------------------------------------------------- #
def test_manifest_matches_sources_and_verify_passes(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    BR.backup(run, dest, quiet=True)
    bag = dest / "finals"

    manifest = BR.parse_manifest((bag / BR.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest, "マニフェストが空"
    for rel, digest in manifest.items():
        src = run / rel
        assert digest == hashlib.sha256(src.read_bytes()).hexdigest(), rel
    # BagIt 式のタグファイル(完全準拠ではないが型は借りる)
    assert "BagIt-Version" in (bag / BR.BAGIT_NAME).read_text(encoding="utf-8")
    info = (bag / BR.BAGINFO_NAME).read_text(encoding="utf-8")
    assert f"Payload-Oxum: " in info and f".{len(manifest)}" in info

    rep = BR.verify(dest, "finals", quiet=True)
    assert rep["problems"] == []
    assert rep["checked"] == len(manifest) and rep["ok"] == rep["checked"]


def test_verify_detects_corruption(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    r = BR.backup(run, dest, quiet=True)
    tar = dest / "finals" / BR.INCREMENTS_DIR / r["increment"]
    raw = bytearray(tar.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                     # 転送中の 1 ビット化けを模す
    tar.write_bytes(bytes(raw))

    rep = BR.verify(dest, "finals", quiet=True)
    assert rep["problems"], "壊れた tar を見逃した"
    assert any("sha256" in p for p in rep["problems"])


def test_verify_written_round_trip(tmp_path):
    """書いた直後の読み直し照合(--verify-written)が通り、state も正しく更新される。"""
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    rep = BR.backup(run, dest, verify_written=True, quiet=True)
    assert not rep["errors"] and rep["transferred"] > 0
    state = json.loads((dest / "finals" / BR.STATE_NAME).read_text(encoding="utf-8"))
    assert len(state["increments"]) == 1
    assert BR.backup(run, dest, quiet=True)["zero_diff"] is True


def test_verify_reports_missing_increment(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    r = BR.backup(run, dest, quiet=True)
    (dest / "finals" / BR.INCREMENTS_DIR / r["increment"]).unlink()
    rep = BR.verify(dest, "finals", quiet=True)
    assert any("増分 tar が無い" in p for p in rep["problems"])


def test_tree_layout_is_directly_analysable_and_verifies(tmp_path):
    """restore drill 向き: 展開済みコピーがそのまま置かれ、照合も通る。"""
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    BR.backup(run, dest, layout="tree", quiet=True)
    data = dest / "finals" / BR.PAYLOAD_DIR
    assert (data / "l1_events.part-0000.parquet").exists()
    assert (data / "checkpoint" / "ckpt-000072.pkl.gz").exists()
    # 置いた part がそのまま pyarrow で読める(= 復元できることの最小の証拠)
    assert pq.read_table(data / "l1_events.part-0000.parquet").num_rows == 8
    assert BR.verify(dest, "finals", quiet=True)["problems"] == []


# --------------------------------------------------------------------------- #
# ⑤ copy 系のみ = 削除を伝播しない
# --------------------------------------------------------------------------- #
def test_source_deletion_does_not_propagate(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    r1 = BR.backup(run, dest, layout="both", quiet=True)
    assert not r1["errors"]

    victim = "l1_events.part-0000.parquet"
    (run / victim).unlink()                        # 人為ミス/巻き戻しで元が消えた
    r2 = BR.backup(run, dest, layout="both", quiet=True)

    assert victim in r2["missing_at_source"]
    assert (dest / "finals" / BR.PAYLOAD_DIR / victim).exists()
    assert victim in BR.parse_manifest(
        (dest / "finals" / BR.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert BR.verify(dest, "finals", quiet=True)["problems"] == []
    assert r2["zero_diff"] is True                 # 消えただけなら転送も発生しない


def test_rewritten_part_is_flagged_and_kept_in_both_increments(tmp_path):
    """巻き戻しで同名 part の中身が変わった場合: 警告し、旧版も残す(上書きしない)。"""
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    r1 = BR.backup(run, dest, quiet=True)
    old_tar = dest / "finals" / BR.INCREMENTS_DIR / r1["increment"]
    old_state = json.loads((dest / "finals" / BR.STATE_NAME).read_text(encoding="utf-8"))
    old_digest = old_state["files"]["l1_events.part-0000.parquet"]["sha256"]

    target = run / "l1_events.part-0000.parquet"
    _write_part(target, n=16)                      # 同名・別内容(resume で採番が衝突した想定)
    _touch(target, time.time() - 3600)             # 境界の内側に居続ける

    r2 = BR.backup(run, dest, quiet=True)
    assert "l1_events.part-0000.parquet" in r2["changed_existing"]
    new_state = json.loads((dest / "finals" / BR.STATE_NAME).read_text(encoding="utf-8"))
    assert new_state["files"]["l1_events.part-0000.parquet"]["sha256"] != old_digest
    assert old_tar.exists()                        # 旧版は古い増分にそのまま残る
    assert BR.verify(dest, "finals", quiet=True)["problems"] == []


# --------------------------------------------------------------------------- #
# CLI(手順書に貼る形がそのまま動くか)
# --------------------------------------------------------------------------- #
def test_cli_backup_then_verify(tmp_path, capsys):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    rc = BR.main(["--run-dir", str(run), "--dest", str(dest), "--json"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["transferred"] > 0 and rep["run_id"] == "finals"

    rc = BR.main(["--verify", "--dest", str(dest), "--run-id", "finals", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []


def test_cli_verify_needs_an_id(tmp_path):
    assert BR.main(["--verify", "--dest", str(tmp_path)]) == 2


def test_cli_missing_run_dir_is_a_clean_error(tmp_path, capsys):
    rc = BR.main(["--run-dir", str(tmp_path / "nope"), "--dest", str(tmp_path / "d")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err       # traceback を吐かない


def test_cli_verify_fails_on_broken_bag(tmp_path):
    run = _make_run(tmp_path)
    dest = tmp_path / "dest"
    BR.main(["--run-dir", str(run), "--dest", str(dest), "--quiet"])
    man = dest / "finals" / BR.MANIFEST_NAME
    man.write_text(man.read_text(encoding="utf-8").replace("a", "b", 1),
                   encoding="utf-8")
    assert BR.main(["--verify", "--dest", str(dest), "--run-id", "finals",
                    "--quiet"]) == 1


# --------------------------------------------------------------------------- #
# 細部
# --------------------------------------------------------------------------- #
def test_is_complete_parquet(tmp_path):
    good = tmp_path / "good.parquet"
    _write_part(good)
    torn = tmp_path / "torn.parquet"
    _write_torn_part(torn)
    empty = tmp_path / "empty.parquet"
    empty.write_bytes(b"")
    assert BR.is_complete_parquet(good) is True
    assert BR.is_complete_parquet(torn) is False
    assert BR.is_complete_parquet(empty) is False
    assert BR.parquet_ok(good, "pyarrow") is True
    assert BR.parquet_ok(torn, "magic") is False


def test_manifest_roundtrip():
    entries = {"a/b.parquet": {"sha256": "0" * 64, "size": 1},
               "c.json": {"sha256": "1" * 64, "size": 2}}
    text = BR.manifest_text(entries)
    assert text.splitlines()[0].startswith("0" * 64)
    assert "  data/a/b.parquet" in text
    assert BR.parse_manifest(text) == {"a/b.parquet": "0" * 64, "c.json": "1" * 64}


def test_no_checkpoint_yet_waits_for_quiescence(tmp_path):
    """checkpoint が 1 つも無い序盤は「静止した part だけ」を取る。"""
    run = tmp_path / "young"
    run.mkdir()
    fresh = run / "l1_events.part-0000.parquet"
    _write_part(fresh)
    old = run / "l1_events.part-0001.parquet"
    _write_part(old)
    _touch(old, time.time() - 3600)
    rels = {i.rel for i in BR.select_payload(run, min_age_sec=300.0)[0]}
    assert "l1_events.part-0001.parquet" in rels
    assert "l1_events.part-0000.parquet" not in rels


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BR.select_payload(tmp_path / "nope")
