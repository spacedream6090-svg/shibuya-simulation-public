"""追いかけ再生(P6: scripts/live_viewer.py)。**観測がシムを変えないこと**が最上位の検収。

検収の柱:
  A. 読み取り専用 —— live_viewer 併走あり/なしで本体ランの出力(L1/L2/L3/agents/journal)が
     バイト一致。ラン dir 直下には `_live/` 以外の新規ファイルを一切作らない。
  B. 書きかけ part を読まない —— parquet の末尾 magic/footer 長による完結判定。切り詰めた
     (= 書き込み中の)最新 part は取り込まれず、追いかけ位置も進まない。
     一方、より新しい part が現れた不完全 part は「クラッシュの残骸」として恒久スキップ。
  C. 増分 == 一括 —— 3 回に分けて追いかけた結果が、全 part が揃ってから 1 回で読んだ結果と一致。
  D. 実データ一致(統合) —— mock ランを**別プロセス**で走らせながら追いかけ、追いかけ位置が
     単調増加し、その時点の位置スナップショットとイベント計数が canonical L1 から独立に
     再計算した値と一致する。
  E. HTML 機械検査 —— 自己完結(外部 URL ゼロ)・script 均衡・プレースホルダ残留なし・
     背景 JSON が地図と一致・JSONP の受け口と再取得 URL が存在(第76バッチと同流儀)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LV = _load("live_viewer", "scripts/live_viewer.py")

L1_SCHEMA = pa.schema([
    ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
    ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
    ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])

START_MIN = 7 * 60


# ============================================================ 合成データ
def _ev(step, aid, kind, x=0.0, y=0.0, payload=None):
    return {"step": step, "sim_min": START_MIN + step * 10, "agent_id": aid,
            "kind": kind, "x": float(x), "y": float(y),
            "payload": json.dumps(payload, ensure_ascii=False) if payload else "",
            "rng_stream": "", "llm_call_id": ""}


def _write_l1_part(run_dir: Path, idx: int, rows: list) -> Path:
    cols = {k: [r[k] for r in rows] for k in L1_SCHEMA.names}
    path = run_dir / f"l1_events.part-{idx:04d}.parquet"
    pq.write_table(pa.table(cols, schema=L1_SCHEMA), path, compression="zstd")
    return path


def _write_l2_part(run_dir: Path, idx: int, rows: list) -> Path:
    keys = sorted({k for r in rows for k in r})
    path = run_dir / f"l2_metrics.part-{idx:04d}.parquet"
    pq.write_table(pa.table({k: [r.get(k) for r in rows] for k in keys}), path,
                   compression="zstd")
    return path


def _synthetic_map() -> dict:
    return {"meta": {"version": 1, "name": "mock", "attribution": "test"},
            "nodes": [{"id": "n1", "x": 0.0, "y": 0.0}],
            "edges": [{"u": "n1", "v": "n2", "geometry": [[-10.0, -10.0], [10.0, 10.0]]}],
            "buildings": [{"id": "b1", "name": "館", "levels": 2,
                           "footprint": [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]}],
            "pois": [],
            "railways": [{"name": "線", "kind": "rail",
                          "geometry": [[-5000.0, -5000.0], [5000.0, 5000.0]]}]}


def _make_run(tmp_path: Path, name: str = "r", n_steps: int = 40,
              checkpoint_every: int = 8) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        "run:\n"
        f"  n_steps: {n_steps}\n"
        '  start_tod: "07:00"\n'
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "observer:\n"
        f"  checkpoint_every: {checkpoint_every}\n"
        "  flush_every_steps: 0\n", encoding="utf-8")
    (run_dir / "agents.json").write_text(
        json.dumps([{"id": i, "name": f"住民{i}"} for i in range(6)], ensure_ascii=False),
        encoding="utf-8")
    return run_dir


def _truncate(path: Path, keep_ratio: float = 0.6) -> None:
    """書き込み途中の parquet を模す(末尾 footer/magic がまだ無い状態)。"""
    data = path.read_bytes()
    path.write_bytes(data[: max(8, int(len(data) * keep_ratio))])


def _hash_tree(run_dir: Path) -> dict:
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            out[p.relative_to(run_dir).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ============================================================ B. part 完結判定
def test_is_complete_parquet_true_for_finished_file(tmp_path):
    run_dir = _make_run(tmp_path)
    p = _write_l1_part(run_dir, 0, [_ev(0, 0, "arrive", 1, 2)])
    assert LV.is_complete_parquet(p)


@pytest.mark.parametrize("mangle", ["truncate", "empty", "garbage", "head_only"])
def test_is_complete_parquet_false_for_unfinished(tmp_path, mangle):
    run_dir = _make_run(tmp_path)
    p = _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, s) for s in range(20)])
    if mangle == "truncate":
        _truncate(p)
    elif mangle == "empty":
        p.write_bytes(b"")
    elif mangle == "garbage":
        p.write_bytes(b"not a parquet at all, but long enough to pass the size gate")
    else:                                    # 先頭 magic だけ書けている状態
        p.write_bytes(b"PAR1" + b"\x00" * 64)
    assert not LV.is_complete_parquet(p)


def test_is_complete_parquet_missing_file(tmp_path):
    assert not LV.is_complete_parquet(tmp_path / "nope.parquet")


def test_part_index_and_listing(tmp_path):
    run_dir = _make_run(tmp_path)
    for i in (2, 0, 1):
        _write_l1_part(run_dir, i, [_ev(i, 0, "arrive")])
    pq.write_table(pa.table({"a": [1]}), run_dir / "l1_events.parquet")   # canonical は無視
    parts = LV.list_parts(run_dir, "l1_events")
    assert [i for i, _ in parts] == [0, 1, 2]
    assert LV.part_index(parts[1][1]) == 1
    assert LV.part_index(run_dir / "l1_events.parquet") == -1


def test_writing_part_is_not_read(tmp_path):
    """最新 part が書き込み中(= 末尾 magic 無し)なら取り込まず、追いかけ位置も進まない。"""
    run_dir = _make_run(tmp_path)
    _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, 0) for s in range(8)])
    p1 = _write_l1_part(run_dir, 1, [_ev(s, 0, "arrive", s, 0) for s in range(8, 16)])
    _truncate(p1)
    st = LV.ChaseState(run_dir)
    d = st.poll()
    assert d["chase"]["step"] == 7, "書きかけ part を読んでしまっている"
    assert d["lag"]["parts_read"] == 1 and d["lag"]["parts_pending"] == 1
    assert d["status"] == "chasing"
    # 書き終わったら次の poll で取り込まれる
    _write_l1_part(run_dir, 1, [_ev(s, 0, "arrive", s, 0) for s in range(8, 16)])
    d = st.poll()
    assert d["chase"]["step"] == 15 and d["lag"]["parts_read"] == 2


def test_reader_never_blocks_the_sims_unlink(tmp_path):
    """★ドクトリンの核心: 読んでいる最中でもシム側の unlink を邪魔しない。

    実測した事故: 素の `open()` は Windows で FILE_SHARE_DELETE を立てないため、追いかけ側が
    part を開いている間に `logger._finalize_stream` の `p.unlink()` が
    PermissionError [WinError 32] で失敗し、**ランごと落ちた**。POSIX では元から起きない。
    """
    run_dir = _make_run(tmp_path)
    p = _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, 0) for s in range(8)])
    with LV._open_shared(p) as f:
        assert f.read(4) == b"PAR1"
        p.unlink()                            # ← ここが例外を投げたらドクトリン違反
    assert not p.exists()


def test_part_read_completes_even_if_unlinked_midway(tmp_path):
    """unlink 後もこちらのハンドルは有効=取りこぼしなく読み切れる(削除保留の意味論)。"""
    run_dir = _make_run(tmp_path)
    p = _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, 0) for s in range(8)])
    with LV._open_shared(p) as f:
        pf = pq.ParquetFile(f)
        p.unlink()
        tbl = pf.read(columns=["step", "kind"])
    assert tbl.num_rows == 8
    assert tbl.column("step").to_pylist() == list(range(8))


def test_completeness_check_also_uses_shared_open(tmp_path):
    """完結判定の一瞬の read でも unlink を弾かない(判定は毎 poll・全 part に走る)。"""
    run_dir = _make_run(tmp_path)
    p = _write_l1_part(run_dir, 0, [_ev(0, 0, "arrive")])
    assert LV.is_complete_parquet(p)
    p.unlink()                                # 判定でハンドルが残っていたらここで落ちる
    assert not LV.is_complete_parquet(p)


def test_part_vanishing_between_check_and_read(tmp_path, monkeypatch):
    """finalize は part を canonical へ結合したあと unlink する。完結判定と読み取りの間で
    消えるのは正常系であり、観測プロセスはそこで落ちてはいけない(実測で踏んだレース)。"""
    run_dir = _make_run(tmp_path)
    _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, 0) for s in range(8)])
    _write_l1_part(run_dir, 1, [_ev(s, 0, "arrive", s, 0) for s in range(8, 16)])
    _write_l2_part(run_dir, 0, [{"step": s, "n_moving": 1.0} for s in range(8)])
    real = LV.is_complete_parquet

    def vanishing(path):                      # 「完結している」と答えた直後に消える
        ok = real(path)
        if ok and path.name.endswith("part-0001.parquet"):
            path.unlink()
        return ok

    monkeypatch.setattr(LV, "is_complete_parquet", vanishing)
    st = LV.ChaseState(run_dir)
    d = st.poll()                             # 例外を投げないこと自体が検収
    assert d["chase"]["step"] == 7, "消えた part の手前まで進んでいるべき"
    assert any("消えた" in w for w in d["warnings"])
    assert st.next_l1 == 2, "消えた part を永久に読み直そうとしている"
    monkeypatch.undo()
    assert st.poll()["warnings"] == []


def test_dead_part_is_skipped_with_warning(tmp_path):
    """より新しい part が現れた不完全 part = クラッシュの残骸。警告して先へ進む。"""
    run_dir = _make_run(tmp_path)
    p0 = _write_l1_part(run_dir, 0, [_ev(s, 0, "arrive", s, 0) for s in range(8)])
    _truncate(p0)
    _write_l1_part(run_dir, 1, [_ev(s, 0, "arrive", s, 0) for s in range(8, 16)])
    st = LV.ChaseState(run_dir)
    d = st.poll()
    assert d["chase"]["step"] == 15
    assert any("スキップ" in w for w in d["warnings"])
    assert 0 in st.skipped
    assert st.poll()["warnings"] == [], "同じ警告を毎回出し続けない"


# ============================================================ 位置ビルダーの意味論
def _reference_positions(rows: list, upto_step: int) -> dict:
    """make_viewer.build_data と同じイベント意味論の独立実装(テスト側の参照)。

    live_viewer の増分実装とは別に書くことで「同じ数字を見ている」ことを機械検査する。
    """
    cur: dict = {}
    for r in sorted(rows, key=lambda r: r["step"]):
        if r["step"] > upto_step or r["agent_id"] < 0:
            continue
        k = r["kind"]
        if k not in LV.POS_KINDS:
            continue
        c = cur.setdefault(r["agent_id"], [0.0, 0.0, LV.W_ROAD])
        x, y = round(float(r["x"]), 1), round(float(r["y"]), 1)
        if k in ("move_segment", "arrive", "speak", "reflect"):
            c[0], c[1] = x, y
        elif k == "enter_building":
            c[0], c[1], c[2] = x, y, LV.W_INDOOR
        elif k in ("exit_building", "enter_area"):
            c[0], c[1], c[2] = x, y, LV.W_ROAD
        elif k == "exit_area":
            c[2] = LV.W_OUTSIDE
        elif k == "sleep_start":
            c[2] = LV.W_SLEEP
        elif k == "wake_up":
            c[2] = LV.W_ROAD if c[2] == LV.W_SLEEP else c[2]
    return cur


def test_position_state_machine_matches_reference(tmp_path):
    run_dir = _make_run(tmp_path)
    rows = [
        _ev(0, 0, "arrive", 5, 5), _ev(1, 0, "enter_building", 20, 20, {"building": "b1"}),
        _ev(2, 0, "floor_move", 20, 20, {"building": "b1", "floor": 2}),
        _ev(3, 0, "exit_building", 20, 20, {"building": "b1"}),
        _ev(4, 1, "exit_area", -50, -50, {"gateway": "g"}),
        _ev(5, 1, "enter_area", -40, -40, {"gateway": "g"}),
        _ev(6, 2, "sleep_start", 3, 3, {"until_step": 9}),
        _ev(7, 2, "wake_up", 3, 3, {"slept_steps": 1}),
        _ev(7, 3, "sleep_start", 9, 9, {}),
        _ev(7, -1, "traffic_flow", 0, 0, {"n": 5}),        # 世界イベントは位置に効かない
    ]
    _write_l1_part(run_dir, 0, rows)
    _write_l1_part(run_dir, 1, [_ev(8, 0, "move_segment", 11.26, -3.34)])
    st = LV.ChaseState(run_dir)
    d = st.poll()
    assert d["chase"]["step"] == 8
    assert st.pos == _reference_positions(rows + [_ev(8, 0, "move_segment", 11.26, -3.34)], 8)
    assert st.pos[0] == [11.3, -3.3, LV.W_ROAD]
    assert st.pos[1][2] == LV.W_ROAD and st.pos[2][2] == LV.W_ROAD
    assert st.pos[3][2] == LV.W_SLEEP
    assert -1 not in st.pos, "agent_id<0(世界イベント)を人として数えている"
    assert d["stat"] == {"n": 4, "road": 3, "indoor": 0, "outside": 0, "sleep": 1}


def test_trail_holds_previous_steps_only(tmp_path):
    run_dir = _make_run(tmp_path)
    rows = [_ev(s, 0, "move_segment", s * 1.0, 0.0) for s in range(6)]
    _write_l1_part(run_dir, 0, rows)
    st = LV.ChaseState(run_dir, trail_steps=3)
    d = st.poll()
    assert d["pos"] == [[5.0, 0.0, LV.W_ROAD]]
    assert d["trail"] == [[[3.0, 0.0]], [[4.0, 0.0]]], "残像は直近 step の履歴(現在位置は含めない)"


# ============================================================ C. 増分 == 一括
def _busy_rows(n_steps: int) -> list:
    rows = []
    for s in range(n_steps):
        for a in range(4):
            rows.append(_ev(s, a, "move_segment", s + a, s - a))
        if s % 5 == 0:
            rows.append(_ev(s, s % 4, "vocab_coin", 1, 1, {"item_id": f"i{s}", "text": f"語{s}"}))
        if s % 7 == 0:
            rows.append(_ev(s, (s + 1) % 4, "enter_building", 20, 20, {"building": "b1"}))
    return rows


def test_incremental_chase_equals_single_shot(tmp_path):
    """3 回に分けて追いかけた結果 == 全 part が揃ってから 1 回で読んだ結果。"""
    inc_dir = _make_run(tmp_path, "inc")
    all_dir = _make_run(tmp_path, "all")
    rows = _busy_rows(24)
    parts = [[r for r in rows if 8 * k <= r["step"] < 8 * (k + 1)] for k in range(3)]

    st_inc = LV.ChaseState(inc_dir)
    steps = []
    for k, rows in enumerate(parts):
        _write_l1_part(inc_dir, k, rows)
        _write_l2_part(inc_dir, k, [{"step": r, "n_moving": float(r % 3),
                                     "distinct_vocab_in_use": float(r)}
                                    for r in range(8 * k, 8 * (k + 1))])
        steps.append(st_inc.poll()["chase"]["step"])
    assert steps == sorted(steps) and steps[-1] == 23, "追いかけ位置が単調に進んでいない"

    for k, rows in enumerate(parts):
        _write_l1_part(all_dir, k, rows)
        _write_l2_part(all_dir, k, [{"step": r, "n_moving": float(r % 3),
                                     "distinct_vocab_in_use": float(r)}
                                    for r in range(8 * k, 8 * (k + 1))])
    d_all = LV.ChaseState(all_dir).poll()
    d_inc = st_inc.poll()
    for key in ("pos", "ids", "trail", "counts", "stat", "series"):
        assert d_inc[key] == d_all[key], f"{key} が増分と一括で食い違う"
    assert [(t["s"], t["k"], t["a"]) for t in d_inc["ticker"]] == \
           [(t["s"], t["k"], t["a"]) for t in d_all["ticker"]]


def test_join_midrun_reads_everything_available(tmp_path):
    """途中参加: part が既に 3 つある状態で起動しても、そこまでを一気に取り込む。"""
    run_dir = _make_run(tmp_path)
    rows = _busy_rows(24)
    for k in range(3):
        _write_l1_part(run_dir, k, [r for r in rows if 8 * k <= r["step"] < 8 * (k + 1)])
    d = LV.ChaseState(run_dir).poll()
    assert d["chase"]["step"] == 23 and d["lag"]["parts_read"] == 3
    assert d["chase"]["day"] == 0 and d["chase"]["tod"] == "10:50"      # 07:00 + 23*10 分
    assert d["chase"]["progress"] == pytest.approx(24 / 40)


# ============================================================ ティッカー / 系列
def test_ticker_and_counts(tmp_path):
    run_dir = _make_run(tmp_path)
    rows = [
        _ev(0, 0, "vocab_coin", 0, 0, {"item_id": "i1", "text": "ハチ公前スワップ"}),
        _ev(1, 1, "move_segment", 1, 1),                       # 見せ場ではない
        _ev(2, -1, "world_event", 0, 0, {"title": "区が発表", "word": "再開発"}),
        _ev(3, 2, "undefined_action", 0, 0, {"action": "踊る", "keys": ["a"]}),
        _ev(4, 3, "joint_invite", 0, 0, {"invitee": 4, "verdict": "accept"}),
    ]
    _write_l1_part(run_dir, 0, rows)
    _write_l1_part(run_dir, 1, [_ev(8, 0, "arrive", 0, 0)])
    d = LV.ChaseState(run_dir).poll()
    assert d["counts"] == {"joint_invite": 1, "undefined_action": 1,
                           "vocab_coin": 1, "world_event": 1}
    tick = {t["k"]: t for t in d["ticker"]}
    assert tick["vocab_coin"]["x"] == "ハチ公前スワップ"
    assert tick["vocab_coin"]["a"] == "住民0"
    assert tick["world_event"]["a"] == "世界"          # agent_id<0 は「世界」
    assert tick["world_event"]["x"] == "再開発"        # _TEXT_KEYS の優先順(word > title)
    assert tick["undefined_action"]["x"] == "踊る"
    assert [t["s"] for t in d["ticker"]] == [4, 3, 2, 0], "新しい順に並んでいない"
    assert tick["vocab_coin"]["t"] == "07:00"


def test_ticker_is_capped_and_keeps_newest(tmp_path):
    run_dir = _make_run(tmp_path)
    _write_l1_part(run_dir, 0, [_ev(s, 0, "vocab_coin", 0, 0, {"text": f"語{s}"})
                                for s in range(30)])
    _write_l1_part(run_dir, 1, [_ev(31, 0, "arrive")])
    d = LV.ChaseState(run_dir, ticker_max=5).poll()
    assert len(d["ticker"]) == 5
    assert [t["x"] for t in d["ticker"]] == [f"語{s}" for s in (29, 28, 27, 26, 25)]
    assert d["counts"]["vocab_coin"] == 30, "計数はティッカー上限に引きずられない"


def test_payload_text_is_honest():
    assert LV._payload_text("") == ""
    assert LV._payload_text(json.dumps({"text": "あ" * 80})).endswith("…")
    assert LV._payload_text(json.dumps({"zzz": 3})) == "zzz=3"     # 既知キーが無ければ生の kv
    assert LV._payload_text("{not json") == "{not json"


def test_series_selection_and_downsample(tmp_path):
    run_dir = _make_run(tmp_path)
    _write_l1_part(run_dir, 0, [_ev(0, 0, "arrive")])
    _write_l1_part(run_dir, 1, [_ev(1, 0, "arrive")])
    _write_l2_part(run_dir, 0, [{"step": s, "n_moving": float(s % 4),
                                 "llm_fallback_rate": s / 400.0,
                                 "mystery_column": 1.0} for s in range(200)])
    _write_l2_part(run_dir, 1, [{"step": 200, "n_moving": 1.0, "llm_fallback_rate": 0.5}])
    d = LV.ChaseState(run_dir, series_points=40, series_max=2).poll()
    keys = [s["key"] for s in d["series"]]
    assert keys == ["llm_fallback_rate", "n_moving"], "SERIES_PREF の優先順に従っていない"
    assert all(s["key"] != "mystery_column" for s in d["series"])
    fb = d["series"][0]
    assert len(fb["pts"]) <= 45 and fb["pts"][-1] == [200, 0.5]
    assert fb["last"] == 0.5 and fb["max"] >= 0.5


# ============================================================ 終了検知 / part 不在
def test_finish_detection_and_viewer_handoff(tmp_path):
    run_dir = _make_run(tmp_path, n_steps=16)
    _write_l1_part(run_dir, 0, [_ev(s, 0, "move_segment", s, 0) for s in range(8)])
    st = LV.ChaseState(run_dir)
    assert st.poll()["status"] == "chasing"
    # finalize 相当: part を消して canonical を書き、summary.json を出す
    for p in run_dir.glob("*.part-*.parquet"):
        p.unlink()
    pq.write_table(pa.table({"step": list(range(16)),
                             "n_moving": [float(s % 3) for s in range(16)]}),
                   run_dir / "l2_metrics.parquet")
    (run_dir / "summary.json").write_text(json.dumps({"n_steps": 16}), encoding="utf-8")
    d = st.poll()
    assert d["finished"] is True and d["status"] == "finished"
    assert "make_viewer.py" in d["viewer_cmd"]
    assert "step 8〜15" in d["note"], "追いかけ切れなかった末尾を正直に出していない"
    assert d["series"][0]["pts"][-1][0] == 15, "L2 canonical で系列が完結していない"
    assert d["series_step"] == 15 and d["chase"]["step"] == 7, \
        "地図と L2 で step がずれるとき、それを黙って混ぜない印(series_step)が要る"


def test_no_parts_config_is_reported(tmp_path):
    run_dir = _make_run(tmp_path, checkpoint_every=0)
    d = LV.ChaseState(run_dir).poll()
    assert d["status"] == "no_parts" and "checkpoint_every=0" in d["note"]


def test_waiting_before_first_flush(tmp_path):
    run_dir = _make_run(tmp_path)
    d = LV.ChaseState(run_dir).poll()
    assert d["status"] == "waiting" and d["chase"]["step"] is None
    assert d["pos"] == [] and d["ticker"] == []


def test_agents_json_absent_then_present(tmp_path):
    """名簿がまだ書かれていなくても落ちない(次の poll で名前が付く)。"""
    run_dir = _make_run(tmp_path)
    (run_dir / "agents.json").unlink()
    _write_l1_part(run_dir, 0, [_ev(0, 0, "vocab_coin", 0, 0, {"text": "語"})])
    _write_l1_part(run_dir, 1, [_ev(8, 0, "vocab_coin", 0, 0, {"text": "語2"})])
    st = LV.ChaseState(run_dir)
    d = st.poll()
    assert d["ticker"][0]["a"] == "agent0"


# ============================================================ 出力(JSONP / HTML)
def test_write_data_is_jsonp_and_atomic(tmp_path):
    out = tmp_path / "_live"
    p = LV.write_data(out, {"gen": 1, "hello": "世界"})
    text = p.read_text(encoding="utf-8")
    assert text.startswith("LIVE_DATA(") and text.rstrip().endswith(");")
    assert json.loads(text.strip()[len("LIVE_DATA("):-2])["hello"] == "世界"
    LV.write_data(out, {"gen": 2})
    assert json.loads(p.read_text(encoding="utf-8").strip()[len("LIVE_DATA("):-2])["gen"] == 2
    assert not list(out.glob("*.tmp")), "一時ファイルが残っている"


def test_background_bbox_excludes_rails(tmp_path):
    map_path = tmp_path / "m.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    bg = LV.build_background(map_path)
    assert bg["bbox"] == [-10.0, -10.0, 20.0, 20.0], "郊外まで伸びる線路が枠を潰している"
    assert len(bg["rails"]) == 1 and len(bg["roads"]) == 1 and len(bg["blds"]) == 1
    assert bg["attr"] == "test"


def test_background_missing_map_is_survivable(tmp_path):
    bg = LV.build_background(tmp_path / "nope.json")
    assert bg["roads"] == [] and len(bg["bbox"]) == 4


def test_html_is_self_contained_and_wellformed(tmp_path):
    map_path = tmp_path / "m.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    bg = LV.build_background(map_path)
    html = LV.render_html(bg, "prod1", 60.0)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert html.count("<script") == html.count("</script>") == 2
    for ph in ("__BG_JSON__", "__RUN_NAME__", "__INTERVAL_MS__", "__META_REFRESH__"):
        assert ph not in html, f"プレースホルダ {ph} が残っている"
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', html), "外部参照(CDN)がある"
    assert "function LIVE_DATA(" in html and 'live_data.js?v=' in html
    assert "生成が止まっている" in html, "生成側が止まったことをブラウザ側で検知していない"
    for ident in ("bg-data", "sparkh", "sparks", "tick", "fincmd", "b-lag", "b-step", "b-gen"):
        assert html.count('id="' + ident + '"') == 1, f"id={ident} が 1 個でない"
        assert 'getElementById("' + ident + '")' in html, f"id={ident} を誰も読んでいない"
    assert "60000" in html
    assert "<meta http-equiv=\"refresh\"" not in html          # 既定は JS ポーリング
    m = re.search(r'<script type="application/json" id="bg-data">(.*?)</script>', html, re.S)
    assert json.loads(m.group(1).replace("<\\/", "</")) == bg


def test_html_meta_refresh_mode(tmp_path):
    html = LV.render_html({"roads": [], "blds": [], "rails": [], "bbox": [0, 0, 1, 1],
                           "attr": "", "name": ""}, "r", 30.0, refresh="meta")
    assert '<meta http-equiv="refresh" content="30">' in html


def test_html_escapes_run_name():
    html = LV.render_html({"roads": [], "blds": [], "rails": [], "bbox": [0, 0, 1, 1],
                           "attr": "", "name": ""}, '<img src=x onerror=1>', 30.0)
    assert "<img src=x" not in html and "&lt;img src=x" in html


def test_max_dots_thinning_is_deterministic(tmp_path):
    run_dir = _make_run(tmp_path)
    _write_l1_part(run_dir, 0, [_ev(0, a, "arrive", a, a) for a in range(50)])
    _write_l1_part(run_dir, 1, [_ev(8, 0, "arrive", 0, 0)])
    a = LV.ChaseState(run_dir, max_dots=10).poll()
    b = LV.ChaseState(run_dir, max_dots=10).poll()
    assert len(a["ids"]) == 10 and a["ids"] == b["ids"] == sorted(a["ids"])
    assert a["pos"] == b["pos"]
    assert a["stat"]["n"] == 50, "間引きは描画だけ(人数の集計は全体)"


# ============================================================ A. 読み取り専用(単体)
def test_chase_does_not_touch_run_dir(tmp_path):
    run_dir = _make_run(tmp_path)
    for k in range(3):
        _write_l1_part(run_dir, k, [_ev(s, 0, "arrive", s, 0)
                                    for s in range(8 * k, 8 * (k + 1))])
        _write_l2_part(run_dir, k, [{"step": s, "n_moving": 1.0}
                                    for s in range(8 * k, 8 * (k + 1))])
    before = _hash_tree(run_dir)
    st = LV.ChaseState(run_dir)
    for _ in range(3):
        LV.write_data(run_dir / "_live", st.poll())
    after = _hash_tree(run_dir)
    for rel, h in before.items():
        assert after.get(rel) == h, f"{rel} が書き換わっている"
    new = set(after) - set(before)
    assert new and all(n.startswith("_live/") for n in new), f"_live/ の外に書いた: {new}"
    assert not any(re.match(r"[^/]+$", n) for n in new), "ラン dir 直下に何か作っている"


# ============================================================ D/A. 統合(別プロセスの実走)
RUN_ARGS = ["run.seed=11", "run.n_agents=14", "run.n_steps=48",
            "observer.checkpoint_every=6"]
STABLE_OUTPUTS = ("l1_events.parquet", "l2_metrics.parquet", "l3_snapshots.parquet",
                  "agents.json", "traits.json", "llm_cache.jsonl")
VOLATILE_SUMMARY_KEYS = ("elapsed_sec", "peak_rss_mb", "out_dir", "files")


def _spawn_run(out_root: Path, name: str) -> subprocess.Popen:
    """mock ランを別プロセスで起動。stderr はファイルへ(PIPE を読まずに放置=詰まりの元)。"""
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log = (out_root / f"{name}.err.log").open("wb")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "run.py"),
         *RUN_ARGS, f"run.name={name}", f"run.out_dir={out_root.as_posix()}"],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.DEVNULL, stderr=log)
    proc._err_log = out_root / f"{name}.err.log"        # 失敗時の診断用
    return proc


def _err(proc: subprocess.Popen) -> str:
    try:
        return proc._err_log.read_text(encoding="utf-8", errors="replace")[-2000:]
    except (AttributeError, OSError):
        return ""


def _chase_until_done(run_dir: Path, proc: subprocess.Popen, out_dir: Path,
                      timeout: float = 900.0) -> list:
    """別プロセスのランを追いかけ、各 poll のデータを記録して返す。

    timeout は「本当に固まった」だけを拾うための安全弁(mock ランの実測は数秒)。
    フルスイートを -n auto で回すと 20 並列に踏まれて桁で伸びるため大きく取る。
    """
    st = LV.ChaseState(run_dir, trail_steps=3)
    frames = []
    t0 = time.time()
    while True:
        if run_dir.is_dir():
            data = st.poll()
            LV.write_data(out_dir, data)
            frames.append(data)
            if data["finished"]:
                break
        if time.time() - t0 > timeout:
            proc.kill()
            pytest.fail(f"mock ランが {timeout}s 以内に終わらない "
                        f"(rc={proc.poll()} steps={[f['chase']['step'] for f in frames]}): "
                        f"{_err(proc)}")
        time.sleep(0.1)                    # flush 間隔(6 step)より十分細かく覗く
    return frames


@pytest.mark.slow
@pytest.mark.xdist_group("subproc_run")   # M-2: run.py を Popen で実走 = 直列化
def test_integration_chase_matches_real_data(tmp_path):
    """mock ランを別プロセスで実走させながら追いかけ、実データと突き合わせる。"""
    out_root = tmp_path / "runs"
    proc = _spawn_run(out_root, "chased")
    run_dir = out_root / "chased"
    frames = _chase_until_done(run_dir, proc, run_dir / "_live")
    proc.wait(timeout=300)

    steps = [f["chase"]["step"] for f in frames if f["chase"]["step"] is not None]
    assert steps and steps == sorted(steps), f"追いかけ位置が単調でない: {steps}"
    # 到達しうる最大は 41(steps 42-47 の最終 part は finalize が即座に結合して消すため)。
    # 負荷で 1 flush ぶん取りこぼしても落ちないよう 1 段だけ余裕を持たせる。
    assert steps[-1] >= 35, f"最後の完結 part まで届いていない: {steps}"

    chasing = [f for f in frames if f["status"] == "chasing"]
    assert chasing, "chasing 状態が一度も観測されない(実走中に追いつけていない)"
    mid = chasing[len(chasing) // 2]
    upto = mid["chase"]["step"]

    rows = pq.read_table(run_dir / "l1_events.parquet").to_pylist()
    ref = _reference_positions(rows, upto)
    got = {aid: list(p) for aid, p in zip(mid["ids"], mid["pos"])}
    assert got == {a: list(v) for a, v in ref.items()}, \
        "追いかけ位置スナップショットが canonical L1 の再計算と一致しない"

    n_hi = sum(1 for r in rows
               if r["kind"] in LV.HIGHLIGHT_KINDS and r["step"] <= upto)
    assert sum(mid["counts"].values()) == n_hi, "見せ場イベントの計数が実データと違う"
    seen = {(r["step"], r["agent_id"], r["kind"]) for r in rows}
    for t in mid["ticker"]:
        aid = -1 if t["a"] == "世界" else None
        cands = [s for s in seen if s[0] == t["s"] and s[2] == t["k"]]
        assert cands, f"ティッカーに実在しないイベント: {t}"
        if aid == -1:
            assert any(c[1] < 0 for c in cands)
    assert frames[-1]["finished"] and "make_viewer" in frames[-1]["viewer_cmd"]


@pytest.mark.slow
@pytest.mark.xdist_group("subproc_run")   # M-2: run.py を Popen で実走 = 直列化
def test_integration_observation_does_not_change_the_run(tmp_path):
    """★最上位の検収: live_viewer 併走あり/なしで本体ランの出力がバイト一致。"""
    out_root = tmp_path / "runs"
    plain = _spawn_run(out_root, "plain")
    assert plain.wait(timeout=900) == 0, _err(plain)

    watched = _spawn_run(out_root, "watched")
    watched_dir = out_root / "watched"
    _chase_until_done(watched_dir, watched, tmp_path / "_live_out")
    assert watched.wait(timeout=300) == 0, _err(watched)

    plain_dir = out_root / "plain"
    for name in STABLE_OUTPUTS:
        a, b = plain_dir / name, watched_dir / name
        assert a.exists() and b.exists(), f"{name}: plain={a.exists()} watched={b.exists()}"
        ba, bb = a.read_bytes(), b.read_bytes()
        assert ba == bb, (f"{name} が併走で変わった "
                          f"(plain {len(ba)}B / watched {len(bb)}B)")
    sa = json.loads((plain_dir / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((watched_dir / "summary.json").read_text(encoding="utf-8"))
    for k in VOLATILE_SUMMARY_KEYS:            # パス・実時間は本質的に一致しない
        sa.pop(k, None)
        sb.pop(k, None)
    assert sa == sb, "summary.json(パス/実時間を除く)が併走で変わった"
    # 追いかけ側のラン dir 直下には _live/ すら作っていない(--out-dir を外へ向けたため)
    assert not (watched_dir / "_live").exists()
    assert sorted(p.name for p in plain_dir.iterdir()) == \
           sorted(p.name for p in watched_dir.iterdir())


@pytest.mark.slow
@pytest.mark.xdist_group("subproc_run")   # M-2: run.py を Popen で実走 = 直列化
def test_cli_once_on_finished_run(tmp_path):
    """CLI: 終わったランに後から掛けても落ちない(part は既に消えている)。"""
    out_root = tmp_path / "runs"
    proc = _spawn_run(out_root, "done")
    assert proc.wait(timeout=900) == 0, _err(proc)
    out_dir = tmp_path / "live"
    rc = LV.main([str(out_root / "done"), "--once", "--quiet",
                  "--out-dir", str(out_dir)])
    assert rc == 0
    html = (out_dir / "live.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    data = json.loads((out_dir / "live_data.js").read_text(encoding="utf-8")
                      .strip()[len("LIVE_DATA("):-2])
    assert data["finished"] is True and data["gen"] == 1
    assert data["chase"]["step"] == 47, "L2 canonical から最終 step を拾えていない"


def test_cli_missing_run_dir_returns_2(tmp_path):
    assert LV.main([str(tmp_path / "nope"), "--once", "--quiet"]) == 2


# ============================================================ F. part 読みの有界化(W4-C)
# W2-2 残件#2: 1 part を `ParquetFile.read()` で丸ごと Arrow 表にしていたため、常駐が
# **part の行数に比例**していた(在場 25万の本線では checkpoint 1 回分の part が数千万行)。
# `iter_batches` + 列射影へ移した。ここで固定するのは 2 点:
#   (a) 出力が変わらない —— batch 境界をどこに置いても、また HEAD 実装と比べても同値。
#   (b) 構造的にピークが batch 単位 —— `ParquetFile.read` を**一度も呼ばない**(実行時に検知)。
_STABLE_KEYS = ("pos", "ids", "trail", "counts", "stat", "series", "chase", "lag")


def _stable(d: dict) -> dict:
    """壁時計依存(wrote_at/wrote_epoch/gen/lag.seconds/lag.text)を除いた比較用ビュー。

    lag の seconds/text は「最後の part の mtime から今まで」= 実行のたびに動くので外す。
    parts_read / rows_read / src は読み方が変われば動くので**残す**(有界化の検収対象)。"""
    out = {k: d[k] for k in _STABLE_KEYS if k in d}
    if "lag" in out:
        out["lag"] = {k: v for k, v in out["lag"].items()
                      if k not in ("seconds", "text")}
    out["ticker"] = [(t["s"], t["k"], t["a"], t["d"], t["t"], t["x"]) for t in d["ticker"]]
    return out


def _busy_run(tmp_path: Path, name: str, n_parts: int = 3, per: int = 8) -> Path:
    run_dir = _make_run(tmp_path, name, n_steps=n_parts * per, checkpoint_every=per)
    rows = _busy_rows(n_parts * per)
    for k in range(n_parts):
        _write_l1_part(run_dir, k, [r for r in rows if per * k <= r["step"] < per * (k + 1)])
        _write_l2_part(run_dir, k, [{"step": s, "n_moving": float(s % 3),
                                     "distinct_vocab_in_use": float(s)}
                                    for s in range(per * k, per * (k + 1))])
    return run_dir


@pytest.mark.parametrize("batch_rows", [1, 2, 3, 7, 64, 131_072])
def test_batch_size_does_not_change_results(tmp_path, monkeypatch, batch_rows):
    """batch 境界をどこに置いても結果は同一(= 有界化が意味論を変えていない)。

    part 内の step 境界・見せ場・start_min 復元がすべて batch をまたぐ大きさで回す。
    """
    ref_dir = _busy_run(tmp_path, f"ref{batch_rows}")
    ref = _stable(LV.ChaseState(ref_dir).poll())        # 既定 BATCH_ROWS
    monkeypatch.setattr(LV, "BATCH_ROWS", batch_rows)
    got_dir = _busy_run(tmp_path, f"got{batch_rows}")
    assert _stable(LV.ChaseState(got_dir).poll()) == ref


def test_batch_size_does_not_change_incremental_results(tmp_path, monkeypatch):
    """増分追いかけ(part ごとの poll)でも batch 境界に依らない。"""
    def _chase(run_dir: Path) -> list:
        st = LV.ChaseState(run_dir)
        return [_stable(st.poll()) for _ in range(3)]

    ref = _chase(_busy_run(tmp_path, "iref"))
    monkeypatch.setattr(LV, "BATCH_ROWS", 3)
    assert _chase(_busy_run(tmp_path, "igot")) == ref


def test_never_materializes_a_whole_part(tmp_path, monkeypatch):
    """`ParquetFile.read()` を **1 度も呼ばない**(呼べば例外=構造的根拠のランタイム検査)。

    これが「ピークが part 全体でなく batch 単位」の根拠。列射影だけでは part の行数に
    比例した Arrow 表が残るので、read を封じてなお通ることを条件にする。
    """
    def _boom(self, *a, **kw):                          # noqa: ANN001
        raise AssertionError("ParquetFile.read() = part 全体の実体化(有界化が壊れている)")

    monkeypatch.setattr(pq.ParquetFile, "read", _boom)
    run_dir = _busy_run(tmp_path, "noread")
    d = LV.ChaseState(run_dir).poll()
    assert d["chase"]["step"] == 23 and d["lag"]["parts_read"] == 3
    assert d["series"], "L2 系列も read() 無しで読めている"
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pq.write_table(pa.table({"step": list(range(24)),
                             "n_moving": [float(s % 3) for s in range(24)]}),
                   run_dir / "l2_metrics.parquet")
    assert LV.ChaseState(run_dir).poll()["finished"] is True   # canonical 経路も read 無し


def test_source_has_no_whole_table_read():
    """静的にも `ParquetFile.read(` / `read_table(` が残っていない(コメント/文字列を除く)。"""
    import ast
    src = (REPO_ROOT / "scripts" / "live_viewer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("read", "read_table", "read_row_group", "read_pandas"):
                owner = node.func.value
                name = getattr(owner, "id", None) or getattr(owner, "attr", None)
                if name in ("pf", "pq"):
                    bad.append(f"L{node.lineno}: {name}.{node.func.attr}(")
    assert not bad, "part 全体を実体化する呼び出しが残っている: " + ", ".join(bad)
    assert "iter_batches" in src


def test_max_step_helper_matches_full_scan(tmp_path):
    """`_max_step_of` は「step 列の最大」と厳密同値(統計あり/なしの両方)。"""
    import pyarrow.compute as pc
    rows = _busy_rows(24)
    for stats in (True, False):
        path = tmp_path / f"m{int(stats)}.parquet"
        cols = {k: [r[k] for r in rows] for k in L1_SCHEMA.names}
        pq.write_table(pa.table(cols, schema=L1_SCHEMA), path,
                       write_statistics=stats, row_group_size=7)
        with LV._open_shared(path) as fh:
            pf = pq.ParquetFile(fh)
            assert LV._max_step_of(pf, pc) == 23
    # 全 null の step 列 = 判断材料が無い → None(偽の値を作らない)
    path = tmp_path / "null.parquet"
    pq.write_table(pa.table({"step": pa.array([None, None], pa.int32())}), path)
    with LV._open_shared(path) as fh:
        assert LV._max_step_of(pq.ParquetFile(fh), pc) is None


@pytest.mark.xdist_group("live_viewer_head")
def test_matches_head_implementation(tmp_path):
    """HEAD 実装(part を丸読みしていた版)と画面データが完全一致(移行の受入条件)。"""
    try:
        src = subprocess.check_output(["git", "show", "HEAD:scripts/live_viewer.py"],
                                      cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
    except Exception:                                    # noqa: BLE001
        pytest.skip("git 不在(HEAD 版を取れない)")
    # 【第137 追随】live_viewer は viz/notable_events.py を「自分の親の親 = REPO_ROOT」相対で
    # 読む(kind レジストリの単一化)。HEAD 版をコピーして動かす本テストでも同じ相対配置
    # (scripts/ の下+隣に viz/)を再現しないと HEAD 側だけ FileNotFoundError で落ちる。
    # HEAD が旧版(レジストリ不読)なら viz コピーは単に使われないだけで無害。
    head_py = tmp_path / "scripts" / "live_viewer_head.py"
    head_py.parent.mkdir(parents=True, exist_ok=True)
    head_py.write_bytes(src)
    try:
        nsrc = subprocess.check_output(["git", "show", "HEAD:viz/notable_events.py"],
                                       cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        (tmp_path / "viz").mkdir(exist_ok=True)
        (tmp_path / "viz" / "notable_events.py").write_bytes(nsrc)
    except Exception:                                    # noqa: BLE001
        pass
    # `_load` は REPO_ROOT / rel で解決する。絶対パスを渡せば pathlib がそのまま採る。
    HEADLV = _load("live_viewer_head", str(head_py))
    head_dir = _busy_run(tmp_path, "hd")
    cur_dir = _busy_run(tmp_path, "cu")
    assert _stable(LV.ChaseState(cur_dir).poll()) == \
        _stable(HEADLV.ChaseState(head_dir).poll())
    # 終了後(L2 canonical から末尾を補完する経路)も一致
    for d in (head_dir, cur_dir):
        (d / "summary.json").write_text("{}", encoding="utf-8")
        pq.write_table(pa.table({"step": list(range(30)),
                                 "n_moving": [float(s % 3) for s in range(30)],
                                 "distinct_vocab_in_use": [float(s) for s in range(30)]}),
                       d / "l2_metrics.parquet")
    assert _stable(LV.ChaseState(cur_dir).poll()) == \
        _stable(HEADLV.ChaseState(head_dir).poll())
