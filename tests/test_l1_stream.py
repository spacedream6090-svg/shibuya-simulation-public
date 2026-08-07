"""W2-2: 解析パイプラインの 25万対応(`scripts/l1_stream.py` と移行先の同値固定)。

提案書 `docs/plans/proposal-dp-u3-observe-250k.md` §2-4 の実測 = 在場 25万 × 10 日の L1 は
42.7 GB・40.6 億イベント。`measure.load_events` の全件 RAM 展開は破綻するので、本選で
確実に使う解析を row-group 逐次へ移した。ここで固定するのは **移行前後の出力同値** である。

固定する不変条件
----------------
A. `l1_stream.iter_events` は `measure.load_events` と**バイト同値**(キー順まで)。
   kind 絞り込み・step 範囲は「`load_events` を同条件で filter した部分列」と同値。
B. チャンク跨ぎ(複数 row-group)・空ラン・欠測列(rng_stream/llm_call_id 無し)で崩れない。
C. `row_count` / `max_step` / `kind_counts` が全件展開と一致する。
D. 移行した 4 本(analyze_sweep / analyze_rumors / watchdog_llm / summarize_run /
   detect_regression)の出力が移行前の実装と一致する。
E. row-group 枝刈りが**実際に効いている**(読んだ row-group 数が減っている)。

pandas / duckdb は使わない(pyarrow + 標準ライブラリのみ)。実 LLM は呼ばない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import l1_stream as ls                              # noqa: E402
from society.observer import measure as m           # noqa: E402


# --------------------------------------------------------------------------- #
# 合成 L1(複数 row-group・多彩な kind・欠測列)
# --------------------------------------------------------------------------- #
_KINDS = ("speak", "hear", "transmission", "vocab_coin", "rumor_born",
          "rumor_stifle", "dm", "spend", "arrive", "fallback")


def _events(n_steps: int = 12, n_agents: int = 5) -> list[dict]:
    ev: list[dict] = []
    for s in range(n_steps):
        for a in range(n_agents):
            kind = _KINDS[(s * n_agents + a) % len(_KINDS)]
            payload: dict = {"n": s}
            if kind == "transmission":
                # 噂 / 語彙の両方が混ざるように(item_id 接頭辞での切り分けを実際に突く)
                payload |= {"item_id": ("rumor-r1" if (s + a) % 3 else "w1"),
                            "from": (a + 1) % n_agents, "channel": "face"}
            elif kind == "rumor_born":
                payload |= {"item_id": "rumor-r1", "src_kind": "enforcement",
                            "node": "hachiko", "knowers": [a, (a + 1) % n_agents]}
            elif kind == "rumor_stifle":
                payload |= {"item_id": "rumor-r1"}
            elif kind == "speak":
                payload |= {"hearers": [(a + 1) % n_agents], "text": "hi",
                            "items": ["w1"]}
            elif kind == "dm":
                payload |= {"to": (a + 2) % n_agents}
            elif kind == "vocab_coin":
                payload |= {"item_id": "w1", "text": "fizz"}
            ev.append({"step": s, "sim_min": 420 + s * 10, "agent_id": a,
                       "kind": kind, "x": float(a), "y": float(s),
                       "payload": payload})
    return ev


def _write_l1(rd: Path, events: list[dict], *, row_group_size: int | None = 7,
              with_optional_cols: bool = False, stem: str = "l1_events",
              part: int | None = None) -> Path:
    """`_write_run`(tests/test_streaming_analyze.py)と同じ書き方の最小版。

    rng_stream / llm_call_id は既定で**書かない**(両経路の None 補完を突く)。
    """
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": pa.array([int(e["step"]) for e in events], pa.int32()),
        "sim_min": pa.array([int(e["sim_min"]) for e in events], pa.int32()),
        "agent_id": pa.array([int(e["agent_id"]) for e in events], pa.int32()),
        "kind": pa.array([str(e["kind"]) for e in events], pa.string()),
        "x": pa.array([float(e["x"]) for e in events], pa.float32()),
        "y": pa.array([float(e["y"]) for e in events], pa.float32()),
        "payload": pa.array([json.dumps(e["payload"], ensure_ascii=False,
                                        sort_keys=True) for e in events],
                            pa.string()),
    }
    if with_optional_cols:
        cols["rng_stream"] = pa.array(["s"] * len(events), pa.string())
        cols["llm_call_id"] = pa.array([None] * len(events), pa.string())
    name = f"{stem}.parquet" if part is None else f"{stem}.part-{part:04d}.parquet"
    path = rd / name
    kw = {"row_group_size": row_group_size} if row_group_size else {}
    pq.write_table(pa.table(cols), path, **kw)
    return path


@pytest.fixture()
def run_dir(tmp_path) -> Path:
    rd = tmp_path / "run"
    _write_l1(rd, _events())
    return rd


# --------------------------------------------------------------------------- #
# A. iter_events の同値(全件・kind・step 範囲・列射影)
# --------------------------------------------------------------------------- #
def test_iter_events_matches_load_events(run_dir):
    """全件経路は `measure.load_events` と **JSON バイト同値**(キー順まで)。"""
    base = m.load_events(str(run_dir))
    got = list(ls.iter_events(str(run_dir)))
    assert got == base
    assert json.dumps(got, ensure_ascii=False) == json.dumps(base, ensure_ascii=False)
    assert [list(r) for r in got[:3]] == [list(r) for r in base[:3]]   # キー順


def test_multiple_row_groups_do_not_change_output(tmp_path):
    """チャンク跨ぎ: row-group サイズを変えても列は 1 バイトも変わらない。"""
    ev = _events()
    outs = []
    for i, rgs in enumerate((1, 3, 7, 10_000)):
        rd = tmp_path / f"rg{i}"
        _write_l1(rd, ev, row_group_size=rgs)
        outs.append(json.dumps(list(ls.iter_events(str(rd))), ensure_ascii=False))
        assert pq.ParquetFile(rd / "l1_events.parquet").num_row_groups >= 1
    assert len(set(outs)) == 1


def test_kind_filter_is_exact_subsequence(run_dir):
    base = m.load_events(str(run_dir))
    for kinds in ({"speak"}, {"transmission", "rumor_born"},
                  {"speak", "hear", "dm", "transmission"}, set(_KINDS)):
        want = [e for e in base if e["kind"] in kinds]
        assert list(ls.iter_events(str(run_dir), kinds=kinds)) == want


def test_kind_filter_unknown_and_empty(run_dir):
    assert list(ls.iter_events(str(run_dir), kinds={"no_such_kind"})) == []
    assert list(ls.iter_events(str(run_dir), kinds=set())) == []


def test_step_range_is_exact_subsequence(run_dir):
    base = m.load_events(str(run_dir))
    for lo, hi in ((0, 0), (3, 3), (5, 11), (None, 4), (8, None), (99, None)):
        want = [e for e in base
                if (lo is None or e["step"] >= lo) and (hi is None or e["step"] <= hi)]
        got = list(ls.iter_events(str(run_dir), step_min=lo, step_max=hi))
        assert got == want, (lo, hi)


def test_kind_and_step_combined(run_dir):
    base = m.load_events(str(run_dir))
    want = [e for e in base if e["kind"] in {"speak", "dm"} and 4 <= e["step"] <= 9]
    assert list(ls.iter_events(str(run_dir), kinds={"speak", "dm"},
                               step_min=4, step_max=9)) == want


def test_column_projection_matches_stream_events(run_dir):
    for cols in (["step", "kind"], ["agent_id", "payload"],
                 ["step", "sim_min", "agent_id", "kind", "payload"]):
        want = list(m.stream_events(str(run_dir), columns=cols))
        assert list(ls.iter_events(str(run_dir), columns=cols)) == want


def test_kind_filter_without_kind_in_projection(run_dir):
    """filter 用に読んだ `kind` 列が、要求していないのに出力へ混ざらないこと。"""
    got = list(ls.iter_events(str(run_dir), columns=["step", "agent_id"],
                              kinds={"speak"}))
    assert got and all(set(r) == {"step", "agent_id"} for r in got)
    base = m.load_events(str(run_dir))
    assert [(r["step"], r["agent_id"]) for r in got] == \
           [(e["step"], e["agent_id"]) for e in base if e["kind"] == "speak"]


# --------------------------------------------------------------------------- #
# B. 欠測列・空ラン・part 群
# --------------------------------------------------------------------------- #
def test_optional_columns_present(tmp_path):
    rd = tmp_path / "opt"
    _write_l1(rd, _events(), with_optional_cols=True)
    assert list(ls.iter_events(str(rd))) == m.load_events(str(rd))


def test_missing_columns_are_filled_with_none(run_dir):
    """rng_stream / llm_call_id を書いていないランでも None 補完の形が同じ。"""
    row = next(iter(ls.iter_events(str(run_dir))))
    assert row["rng_stream"] is None and row["llm_call_id"] is None
    assert list(row) == list(m.load_events(str(run_dir))[0])


def test_empty_run(tmp_path):
    rd = tmp_path / "empty"
    rd.mkdir()
    assert list(ls.iter_events(str(rd))) == []
    assert ls.l1_paths(str(rd)) == []
    assert ls.row_count(str(rd)) == 0
    assert ls.max_step(str(rd)) == -1
    assert dict(ls.kind_counts(str(rd))) == {}


def test_zero_row_l1(tmp_path):
    rd = tmp_path / "zero"
    _write_l1(rd, [], row_group_size=None)
    assert list(ls.iter_events(str(rd))) == []
    assert ls.row_count(str(rd)) == 0
    assert ls.max_step(str(rd)) == -1


def test_part_files_are_read_in_index_order(tmp_path):
    """canonical が無いランでは完結 part を index 昇順で連結して読む。"""
    ev = _events()
    rd = tmp_path / "parts"
    half = len(ev) // 2
    _write_l1(rd, ev[:half], part=0)
    _write_l1(rd, ev[half:], part=1)
    got = list(ls.iter_events(str(rd)))
    assert [(e["step"], e["agent_id"], e["kind"]) for e in got] == \
           [(e["step"], e["agent_id"], e["kind"]) for e in ev]
    assert ls.row_count(str(rd)) == len(ev)
    assert ls.max_step(str(rd)) == max(e["step"] for e in ev)


def test_canonical_wins_over_parts(tmp_path):
    """canonical があれば part は読まない(logger の finalize 後の二重計上を防ぐ)。"""
    ev = _events()
    rd = tmp_path / "both"
    _write_l1(rd, ev)
    _write_l1(rd, ev, part=0)
    assert ls.row_count(str(rd)) == len(ev)
    assert len(ls.l1_paths(str(rd))) == 1


def test_incomplete_part_is_skipped(tmp_path):
    rd = tmp_path / "wip"
    _write_l1(rd, _events(), part=0)
    (rd / "l1_events.part-0001.parquet").write_bytes(b"PAR1\x00\x00")  # 書きかけ
    assert [p.name for p in ls.l1_paths(str(rd))] == ["l1_events.part-0000.parquet"]


# --------------------------------------------------------------------------- #
# C. メタ系(row_count / max_step / kind_counts)
# --------------------------------------------------------------------------- #
def test_meta_helpers_match_full_scan(run_dir):
    base = m.load_events(str(run_dir))
    assert ls.row_count(str(run_dir)) == len(base)
    assert ls.max_step(str(run_dir)) == max(e["step"] for e in base)
    from collections import Counter
    assert dict(ls.kind_counts(str(run_dir))) == \
        dict(Counter(e["kind"] for e in base))


def test_max_step_uses_statistics_only(run_dir, monkeypatch):
    """統計が読めるファイルでは step 列を 1 行も走査しない(iter_batches を呼ばない)。"""
    calls = []
    orig = pq.ParquetFile.iter_batches

    def spy(self, *a, **kw):
        calls.append(kw.get("columns"))
        return orig(self, *a, **kw)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    assert ls.max_step(str(run_dir)) == 11
    assert calls == []


# --------------------------------------------------------------------------- #
# E. row-group 枝刈りが効いていること
# --------------------------------------------------------------------------- #
def test_step_pruning_skips_row_groups(tmp_path, monkeypatch):
    """末尾窓の読みで、読む row-group 数が全体より**確かに少ない**。"""
    ev = _events(n_steps=40, n_agents=5)
    rd = tmp_path / "prune"
    path = _write_l1(rd, ev, row_group_size=10)
    total_rg = pq.ParquetFile(path).num_row_groups
    assert total_rg >= 10

    seen = {}
    orig = pq.ParquetFile.iter_batches

    def spy(self, *a, **kw):
        seen["row_groups"] = kw.get("row_groups")
        return orig(self, *a, **kw)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    rows = list(ls.iter_events(str(rd), columns=["step"], step_min=38))
    assert seen["row_groups"] is not None
    assert 0 < len(seen["row_groups"]) < total_rg
    assert [r["step"] for r in rows] == [e["step"] for e in ev if e["step"] >= 38]


def test_kind_filter_on_file_without_kind_column(tmp_path):
    """kind 列が無いファイルに kind 絞り込みを掛けたら 0 行(偽の一致を作らない)。"""
    rd = tmp_path / "nokind"
    rd.mkdir(parents=True)
    pq.write_table(pa.table({"step": pa.array([0, 1, 2], pa.int32())}),
                   rd / "l1_events.parquet")
    assert list(ls.iter_events(str(rd), kinds={"speak"})) == []
    assert [r["step"] for r in ls.iter_events(str(rd), columns=["step"])] == [0, 1, 2]
    assert dict(ls.kind_counts(str(rd))) == {}


def test_pruning_disabled_when_statistics_absent(tmp_path):
    """統計が無い parquet では枝刈りしない(= 取りこぼさない)。"""
    ev = _events()
    rd = tmp_path / "nostats"
    rd.mkdir(parents=True)
    tbl = pa.table({
        "step": pa.array([e["step"] for e in ev], pa.int32()),
        "agent_id": pa.array([e["agent_id"] for e in ev], pa.int32()),
        "kind": pa.array([e["kind"] for e in ev], pa.string()),
    })
    pq.write_table(tbl, rd / "l1_events.parquet", row_group_size=7,
                   write_statistics=False)
    want = [e["step"] for e in ev if e["step"] >= 8]
    got = [r["step"] for r in ls.iter_events(str(rd), columns=["step"], step_min=8)]
    assert got == want
    assert ls.max_step(str(rd)) == max(e["step"] for e in ev)   # 走査へフォールバック


# --------------------------------------------------------------------------- #
# D-1. analyze_rumors: collect() と旧 load_events() 経路の同値
# --------------------------------------------------------------------------- #
def test_analyze_rumors_collect_matches_full_list(run_dir):
    import analyze_rumors as AR
    legacy = AR.summarize(AR.load_events(str(run_dir)), None)
    streamed = AR.summarize(AR.collect(str(run_dir)), None)
    assert json.dumps(streamed, ensure_ascii=False, sort_keys=True) == \
           json.dumps(legacy, ensure_ascii=False, sort_keys=True)
    assert streamed["counts"]["rumors_born"] > 0          # 空の同値で誤魔化さない
    assert streamed["counts"]["rumor_transmissions"] > 0


def test_analyze_rumors_collect_drops_non_tree_rows(run_dir):
    """`collect` が保持する行は「木に効く行」だけ(= 総行数に比例しない)。"""
    import analyze_rumors as AR
    b = AR.collect(str(run_dir))
    kept = {r["kind"] for r in b["events"]}
    assert kept <= {"rumor_born", "rumor_stifle", "transmission"}
    assert all(r["kind"] != "transmission" or AR.is_rumor(r["payload"].get("item_id"))
               for r in b["events"])
    assert len(b["events"]) < ls.row_count(str(run_dir))
    # 母集団は全 want-kind から集めるので、捨てた speak/hear/dm の相手も入る
    assert b["population_seen"] == set(
        AR._bundle_from_events(AR.load_events(str(run_dir)))["population_seen"])


def test_analyze_rumors_explicit_population(run_dir):
    import analyze_rumors as AR
    res = AR.summarize(AR.collect(str(run_dir)), 999)
    assert res["population"] == {"n": 999, "source": "explicit(--population)"}


def test_analyze_rumors_empty_run(tmp_path):
    import analyze_rumors as AR
    rd = tmp_path / "rz"
    _write_l1(rd, [], row_group_size=None)
    res = AR.summarize(AR.collect(str(rd)), None)
    assert res["counts"]["rumors_born"] == 0
    assert res["counts"]["rumor_transmissions"] == 0
    assert res["population"]["n"] == 0


# --------------------------------------------------------------------------- #
# D-2. detect_regression: 移行前後の act 集計が一致
# --------------------------------------------------------------------------- #
def _act_counts_legacy(run_dir: Path, window_steps: int):
    """移行前の実装(全 row-group を 2 回走査)。同値の対照として test 内に持つ。"""
    from society.observer import regression as RG
    path = run_dir / "l1_events.parquet"
    pf = pq.ParquetFile(path)
    max_step = -1
    for batch in pf.iter_batches(columns=["step"]):
        col = batch.column(0).to_pylist()
        if col:
            max_step = max(max_step, int(max(col)))
    if max_step < 0:
        return {}, -1, 0
    floor = max_step - int(window_steps) + 1
    acts: dict = {}
    for batch in pf.iter_batches(columns=["step", "agent_id", "kind"]):
        d = batch.to_pydict()
        for s, aid, kind in zip(d["step"], d["agent_id"], d["kind"]):
            if s < floor:
                continue
            idx = RG._ACT_INDEX.get(kind)
            if idx is None or aid is None or int(aid) < 0:
                continue
            row = acts.setdefault(int(aid), {})
            row[idx] = row.get(idx, 0) + 1
    return acts, max_step, max(1, max_step - floor + 1)


@pytest.mark.parametrize("window", [1, 4, 12, 1000])
def test_detect_regression_act_counts_unchanged(tmp_path, window):
    import detect_regression as DR
    rd = tmp_path / f"dr{window}"
    _write_l1(rd, _events(n_steps=20, n_agents=6), row_group_size=9)
    assert DR.act_counts_from_l1(rd, window) == _act_counts_legacy(rd, window)


def test_detect_regression_act_counts_nonempty(tmp_path):
    """空の同値で通してしまわないための下限(act 種が実際に拾えている)。"""
    import detect_regression as DR
    rd = tmp_path / "drx"
    _write_l1(rd, _events(n_steps=20, n_agents=6), row_group_size=9)
    acts, max_step, n = DR.act_counts_from_l1(rd, 1000)
    # n は旧実装と同じく max(1, max_step-floor+1) = 要求窓幅(ラン長で切り詰めない)
    assert acts and max_step == 19 and n == 1000
    acts2, _s, n2 = DR.act_counts_from_l1(rd, 5)
    assert n2 == 5 and acts2 and acts2 != acts


def test_detect_regression_act_counts_empty_run(tmp_path):
    import detect_regression as DR
    rd = tmp_path / "dre"
    rd.mkdir()
    assert DR.act_counts_from_l1(rd, 144) == ({}, -1, 0)


# --------------------------------------------------------------------------- #
# D-3. watchdog_llm: fallback 件数と l1b 集計
# --------------------------------------------------------------------------- #
def test_watchdog_llm_fallbacks_from_l1(tmp_path):
    import watchdog_llm as W
    rd = tmp_path / "wl"
    ev = _events()
    _write_l1(rd, ev)
    want = sum(1 for e in ev if e["kind"] == "fallback")
    assert want > 0
    assert W._l1_fallbacks(rd) == want                    # summary.json 不在 → L1 経路


def test_watchdog_llm_fallbacks_prefers_summary(tmp_path):
    import watchdog_llm as W
    rd = tmp_path / "wl2"
    _write_l1(rd, _events())
    (rd / "summary.json").write_text(json.dumps({"event_kinds": {"fallback": 7}}),
                                     encoding="utf-8")
    assert W._l1_fallbacks(rd) == 7


def test_watchdog_llm_fallbacks_missing_l1(tmp_path):
    import watchdog_llm as W
    rd = tmp_path / "wl3"
    rd.mkdir()
    assert W._l1_fallbacks(rd) is None


def test_watchdog_llm_l1b_stats_matches_full_read(tmp_path):
    import watchdog_llm as W
    rd = tmp_path / "wl4"
    rd.mkdir()
    n = 25
    tbl = pa.table({
        "purpose": pa.array([None if i % 7 == 0 else f"p{i % 3}" for i in range(n)],
                            pa.string()),
        "cached": pa.array([bool(i % 2) for i in range(n)], pa.bool_()),
    })
    pq.write_table(tbl, rd / "l1b_llm.parquet", row_group_size=4)

    rows = pq.read_table(rd / "l1b_llm.parquet").to_pylist()   # 旧実装
    from collections import Counter
    by, cby, hits = Counter(), Counter(), 0
    for r in rows:
        p = str(r.get("purpose") or "?")
        by[p] += 1
        if r.get("cached"):
            hits += 1
            cby[p] += 1
    legacy = {"llm_calls": len(rows), "cache_hits": hits,
              "by_purpose": {p: {"calls": c, "cache_hit_rate": round(cby[p] / c, 4)}
                             for p, c in sorted(by.items(), key=lambda kv: -kv[1])}}
    assert W._l1b_stats(rd) == legacy


def test_watchdog_llm_l1b_stats_missing(tmp_path):
    import watchdog_llm as W
    rd = tmp_path / "wl5"
    rd.mkdir()
    assert W._l1b_stats(rd) == {"llm_calls": None, "cache_hits": None,
                                "by_purpose": {}}


# --------------------------------------------------------------------------- #
# D-4. summarize_run: 逐次抽出器が全読み抽出器と同値
# --------------------------------------------------------------------------- #
def _write_panel(path: Path, rows: int, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "agent_id": pa.array(list(range(rows)), pa.int32()),
        "day": pa.array([i % 5 for i in range(rows)], pa.int32()),
        "trips": pa.array([None if i % 11 == 0 else i % 9 for i in range(rows)],
                          pa.int32()),
        "n_edges": pa.array([i % 4 for i in range(rows)], pa.int32()),
        "score": pa.array([0.5 * i for i in range(rows)], pa.float64()),
        "allnull": pa.array([None] * rows, pa.float64()),
        "label": pa.array([f"a{i}" for i in range(rows)], pa.string()),
    }), path, row_group_size=row_group_size)


@pytest.mark.parametrize("rows,rgs", [(37, 5), (1, 1), (100, 100)])
def test_summarize_run_generic_stream_matches_full_read(tmp_path, rows, rgs):
    import summarize_run as SR
    p = tmp_path / "panel" / "panel.parquet"
    _write_panel(p, rows, rgs)
    cols, schema = SR._read_parquet(str(p))
    assert SR._extract_generic_stream(str(p)) == SR._extract_generic(cols, schema)


def test_summarize_run_generic_stream_zero_rows(tmp_path):
    import summarize_run as SR
    p = tmp_path / "panel" / "empty.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"a": pa.array([], pa.int32())}), p)
    cols, schema = SR._read_parquet(str(p))
    assert SR._extract_generic_stream(str(p)) == SR._extract_generic(cols, schema)


@pytest.mark.parametrize("rgs", [3, 8, 1000])
def test_summarize_run_heatmap_stream_matches_full_read(tmp_path, rgs):
    import summarize_run as SR
    p = tmp_path / "panel" / "heatmap_grid.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 24
    pq.write_table(pa.table({
        "cell_x": pa.array([i % 4 for i in range(n)], pa.int32()),
        "cell_y": pa.array([i % 3 for i in range(n)], pa.int32()),
        "hour_bin": pa.array([i % 6 for i in range(n)], pa.int32()),
        "pass_count": pa.array([None if i % 5 == 0 else i for i in range(n)],
                               pa.int32()),
        "present_count": pa.array([i * 2 for i in range(n)], pa.int32()),
        "unique_agents": pa.array([i % 7 for i in range(n)], pa.int32()),
    }), p, row_group_size=rgs)
    cols, schema = SR._read_parquet(str(p))
    assert SR._extract_heatmap_grid_stream(str(p)) == \
        SR._extract_heatmap_grid(cols, schema)


def test_summarize_run_panel_dispatch_uses_stream(tmp_path):
    """`_extract_panel_table` が逐次経路を通り、全読み結果と一致する。"""
    import summarize_run as SR
    p = tmp_path / "panel" / "panel.parquet"
    _write_panel(p, 40, 6)
    cols, schema = SR._read_parquet(str(p))
    assert SR._extract_panel_table(str(p)) == SR._extract_generic(cols, schema)


# --------------------------------------------------------------------------- #
# D-5. analyze_sweep: 全件 load_events を残していないこと(退行防止)
# --------------------------------------------------------------------------- #
def test_analyze_sweep_has_no_full_load(tmp_path):
    """AST 固定: analyze_sweep から `load_events` の呼び出しが消えていること。"""
    import ast
    src = (_ROOT / "scripts" / "analyze_sweep.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "load_events" not in names


def test_analyze_sweep_streaming_equals_measure(tmp_path):
    """analyze_sweep が使う 3 関数が、全件版 measure と同じ値を返すこと。"""
    from society.observer import stream as st
    rd = tmp_path / "sw"
    _write_l1(rd, _events())
    events = m.load_events(str(rd))
    agents = m.load_agents(str(rd))
    traits, _tn, _s = m.load_traits(str(rd), agents)
    l2 = m.load_l2(str(rd))
    n_pure = len(agents) or (max((e["agent_id"] for e in events), default=-1) + 1)
    n_stream = len(agents) or (st.max_agent_id(str(rd)) + 1)
    assert n_pure == n_stream
    assert st.agent_features(str(rd), agents, traits) == \
        m.agent_features(events, agents, traits)
    assert st.collective_series(str(rd), l2, n_stream) == \
        m.collective_series(events, l2, n_pure)
