#!/usr/bin/env python
"""W2-2: L1 の共有ストリーミング読み(在場 25万 × 10 日ランの解析パイプライン用)。

なぜ要るか
----------
`docs/plans/proposal-dp-u3-observe-250k.md` §2-4 の実測: 在場 25万 × 10 日の L1 は
**42.7 GB・40.6 億イベント**(いずれも線形外挿の下界)。`measure.load_events` は
これを `list[dict]` に全件展開するので、**1 行あたり最小でも数百バイトの Python
オブジェクト**が要り、確実に破綻する。律速は「回す側」ではなく「測る側」である。

既存資産との関係(車輪の再発明をしない)
--------------------------------------
- `society.observer.measure.stream_events` — row-group/batch 逐次 + 列射影。**有界
  メモリの読み経路は既にここにある**。本モジュールはこれを置き換えない。
- `society.observer.stream` — measure の各集計関数の逐次版(凍結)。集計そのものが
  必要なら**まずこちらを使う**。本モジュールは「まだ逐次版が無い解析」の共有部品。
- `scripts/live_viewer.py` — `_open_shared` / `list_parts` / `is_complete_parquet`。
  Windows で **走行中のランの part を読んでも finalize の unlink を弾かない**開き方の
  単一の源。本モジュールはそれをそのまま借りる。
- `scripts/export_3d.py` の `iter_track_events` — row-group 単位読みの直近前例。

本モジュールが足すのは既存に無かった 3 点だけ
--------------------------------------------
1. **kind の Arrow レベル絞り込み**(`kinds=`)。`stream_events` は有界メモリだが
   *全行* を Python dict に実体化する。40.6 億行のうち解析が要るのが数百万行でも
   40.6 億個の dict を作って捨てる。`pyarrow.compute.is_in` で **RecordBatch を
   filter してから** 実体化するので、捨てる行の Python オブジェクト生成費が 0 になる。
2. **step 範囲の row-group 枝刈り**(`step_min` / `step_max`)。L1 は step 非減少で
   追記されるため `step` 列の row-group 統計(min/max)で**丸ごと読み飛ばせる**。
   末尾窓しか見ない解析(退行監視)が O(全行) → O(窓) になる。
   統計が無い row-group は**枝刈りしない**(= 欠測を偽の値で埋めない)。
3. **part 群の透過的な連結**。canonical(`l1_events.parquet`)があればそれだけを、
   無ければ**完結した** `l1_events.part-NNNN.parquet` を index 昇順で読む
   (走行中のランに対しても安全)。`stream_events` は canonical 専用。

出力同値の約束
--------------
`iter_events(run_dir)` が yield する dict は `measure.load_events(run_dir)` の要素と
**キー順まで含めて同一**(`_materialize` が measure の実体化ループを写経している)。
`kinds=K` を渡した場合は
`[e for e in measure.load_events(run_dir) if e["kind"] in K]` と同一列。
`step_min`/`step_max` を渡した場合は `step_min <= e["step"] <= step_max` の部分列。
いずれも `tests/test_l1_stream.py` でバイト同値を固定している。

O(1) 化するもの / 残るもの
--------------------------
- `row_count` … footer メタのみ。**O(1)**(1 行も読まない)。
- `max_step` … row-group 統計のみ。**O(row-group 数)**。統計が無ければ step 列を
  1 パス走査(その場合だけ O(行数) の *時間*。メモリは batch 分で有界)。
- `kind_counts` … kind 列だけを読み Arrow の `value_counts` で数える。Python 文字列を
  **1 個も作らない**。メモリ **O(distinct kind)**。
- `iter_*` … メモリは **O(batch_rows × 列数)** で有界。**時間は読んだ行数に比例**する。
  これは消えない: parquet は kind でクラスタ化されていないので、kind 絞り込みは
  「復号は要るが実体化は要らない」までしか下げられない。step 枝刈りだけが
  **読む行数そのもの**を減らせる唯一の軸である。
- **全件ソートが本質的に必要な集計は本モジュールでは解けない**。呼び出し側が
  「チャンク集約 → 最後に小さい集約表だけ持つ」へ設計を変える必要がある
  (例: `scripts/analyze_rumors.py` は「木に効く行だけを残して最後に並べ替える」)。

依存は pyarrow + 標準ライブラリのみ(pandas / duckdb は使わない)。副作用なし。
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Windows: 素の open() は FILE_SHARE_DELETE を立てないので、読んでいる最中の part を
# ランが unlink できずシム本体が finalize で落ちる(第77バッチの教訓)。実装の単一の源は
# live_viewer.py。detect_regression.py と同じ借り方をする。
from live_viewer import (                              # noqa: E402
    _open_shared, is_complete_parquet, list_parts,
)

# 列の正準定義は凍結側(observer/measure.py)が単一の源。写経して二重定義にしない。
try:                                                    # noqa: SIM105
    from society.observer.measure import (              # noqa: E402
        _L1_COLUMNS as L1_COLUMNS, _L1_INT_COLUMNS as L1_INT_COLUMNS,
    )
except ImportError:                                     # src を import できない配布形態
    L1_COLUMNS = ("step", "sim_min", "agent_id", "kind", "x", "y", "payload",
                  "rng_stream", "llm_call_id")
    L1_INT_COLUMNS = frozenset({"step", "sim_min", "agent_id"})

DEFAULT_BATCH_ROWS = 131_072
L1_STEM = "l1_events"


# --------------------------------------------------------------------------- #
# パス解決(canonical 優先・無ければ完結 part 群)
# --------------------------------------------------------------------------- #
def l1_paths(run_dir, stem: str = L1_STEM, *,
             complete_only: bool = True) -> list[Path]:
    """読むべき parquet を index 昇順で返す。

    canonical(`<stem>.parquet`)があればそれ 1 本。無ければ `<stem>.part-NNNN.parquet`
    を index 昇順で。`complete_only=True`(既定)は**書きかけの part を読まない**
    (footer 未確定の part は pyarrow が例外を投げる = 走行中ランへの安全弁)。
    """
    rd = Path(run_dir)
    canonical = rd / f"{stem}.parquet"
    if canonical.exists():
        return [canonical]
    if not rd.is_dir():
        return []
    parts = [p for _i, p in list_parts(rd, stem)]
    if complete_only:
        parts = [p for p in parts if is_complete_parquet(p)]
    return parts


def row_count(run_dir, stem: str = L1_STEM) -> int:
    """総行数を footer メタから返す(**1 行も読まない**)。part 群なら合計。"""
    total = 0
    for path in l1_paths(run_dir, stem):
        with _open_shared(path) as fh:
            total += pq.ParquetFile(fh).metadata.num_rows
    return int(total)


# --------------------------------------------------------------------------- #
# row-group 統計による step 枝刈り
# --------------------------------------------------------------------------- #
def _step_bounds(md, rg: int) -> tuple[int, int] | None:
    """row-group `rg` の step 列の (min, max)。統計が無ければ None(=枝刈りしない)。"""
    try:
        schema = md.schema
        names = [schema.column(i).name for i in range(md.num_columns)]
        j = names.index("step")
    except (ValueError, AttributeError):
        return None
    try:
        stats = md.row_group(rg).column(j).statistics
    except (IndexError, AttributeError):
        return None
    if stats is None or not stats.has_min_max:
        return None
    try:
        return int(stats.min), int(stats.max)
    except (TypeError, ValueError):
        return None


def _selected_row_groups(pf, step_min, step_max) -> list[int] | None:
    """step 範囲に交差する row-group の index。枝刈り不要なら None を返す。"""
    if step_min is None and step_max is None:
        return None
    md = pf.metadata
    keep: list[int] = []
    for rg in range(md.num_row_groups):
        b = _step_bounds(md, rg)
        if b is None:                      # 統計が無い = 判断材料が無い → 読む
            keep.append(rg)
            continue
        lo, hi = b
        if step_min is not None and hi < step_min:
            continue
        if step_max is not None and lo > step_max:
            continue
        keep.append(rg)
    return keep


def max_step(run_dir, stem: str = L1_STEM) -> int:
    """最終 step。row-group 統計だけで決まるなら **O(row-group 数)**。

    統計が無い row-group がある場合だけ、その row-group の step 列を走査する
    (メモリは batch 分で有界)。空なら -1。
    """
    mx = -1
    for path in l1_paths(run_dir, stem):
        with _open_shared(path) as fh:
            pf = pq.ParquetFile(fh)
            md = pf.metadata
            need_scan: list[int] = []
            for rg in range(md.num_row_groups):
                b = _step_bounds(md, rg)
                if b is None:
                    need_scan.append(rg)
                elif b[1] > mx:
                    mx = b[1]
            if need_scan:
                for batch in pf.iter_batches(columns=["step"],
                                             row_groups=need_scan,
                                             batch_size=DEFAULT_BATCH_ROWS):
                    if batch.num_rows == 0:
                        continue
                    v = pc.max(batch.column(0)).as_py()
                    if v is not None and int(v) > mx:
                        mx = int(v)
    return int(mx)


def kind_counts(run_dir, stem: str = L1_STEM) -> Counter:
    """kind → 件数。kind 列だけを読み Arrow の `value_counts` で数える。

    Python 文字列を 1 個も作らずに集計するので、40.6 億行でもメモリは
    **O(distinct kind)**(実測の distinct kind は 100 前後)。
    `summary.json` の `event_kinds` が使えるならそちらが速い(0 バイト読み)ことに注意。
    """
    out: Counter = Counter()
    for path in l1_paths(run_dir, stem):
        with _open_shared(path) as fh:
            pf = pq.ParquetFile(fh)
            if "kind" not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(columns=["kind"],
                                         batch_size=DEFAULT_BATCH_ROWS):
                arr = batch.column(0)
                if isinstance(arr, pa.DictionaryArray):
                    arr = arr.dictionary_decode()
                for st in pc.value_counts(arr):
                    out[st["values"].as_py()] += int(st["counts"].as_py())
    return out


# --------------------------------------------------------------------------- #
# 逐次読み(RecordBatch / 列 dict / イベント dict の 3 段)
# --------------------------------------------------------------------------- #
def _plan(available: set, columns) -> dict:
    """要求列 → 射影計画。`measure.stream_events` と同じ規約で組む。"""
    want = list(L1_COLUMNS) if columns is None else list(columns)
    cols = [c for c in want if c in available]          # 実在する列だけ射影
    return {
        "want": want,
        "cols": cols,
        "want_payload": "payload" in cols,
        "fill_none": [c for c in want if c not in available],
        "int_cols": [c for c in cols if c != "payload" and c in L1_INT_COLUMNS],
        "raw_cols": [c for c in cols if c != "payload" and c not in L1_INT_COLUMNS],
    }


def _filter_kinds(batch, kinds_arr):
    """kind が value_set に入る行だけの RecordBatch。null は落とす。"""
    mask = pc.is_in(batch.column("kind"), value_set=kinds_arr)
    mask = pc.fill_null(mask, False)
    return batch.filter(mask)


def _filter_steps(batch, step_min, step_max):
    """step が [step_min, step_max] に入る行だけの RecordBatch(境界は両端含む)。"""
    col = batch.column("step")
    mask = None
    if step_min is not None:
        mask = pc.greater_equal(col, pa.scalar(int(step_min), col.type))
    if step_max is not None:
        m2 = pc.less_equal(col, pa.scalar(int(step_max), col.type))
        mask = m2 if mask is None else pc.and_(mask, m2)
    return batch.filter(pc.fill_null(mask, False))


def iter_record_batches(run_dir, columns=None, *, kinds: Iterable | None = None,
                        step_min: int | None = None, step_max: int | None = None,
                        batch_rows: int = DEFAULT_BATCH_ROWS,
                        stem: str = L1_STEM) -> Iterator[pa.RecordBatch]:
    """射影 + kind 絞り込み + step 枝刈りを掛けた RecordBatch を逐次 yield する。

    絞り込みの結果 0 行になった batch は yield しない(呼び出し側の分岐を減らす)。
    メモリは O(batch_rows × 列数)。Python オブジェクトは 1 個も作らない段。
    """
    kinds_arr = None
    if kinds is not None:
        ks = sorted({str(k) for k in kinds})
        if not ks:                                      # 空集合 = 何も通さない
            return
        kinds_arr = pa.array(ks, pa.string())
    for path in l1_paths(run_dir, stem):
        with _open_shared(path) as fh:
            pf = pq.ParquetFile(fh)
            available = set(pf.schema_arrow.names)
            if kinds_arr is not None and "kind" not in available:
                # kind 列が無いファイルでは「どの行が条件を満たすか」を証明できない。
                # 全通しにすると偽の一致を作るので、1 行も出さない(欠測を偽の値で埋めない)。
                continue
            cols = [c for c in (list(L1_COLUMNS) if columns is None
                                else list(columns)) if c in available]
            read_cols = list(cols)
            # filter に要る列は、呼び出し側が要求していなくても読む(後で落とす)
            if kinds_arr is not None and "kind" not in read_cols:
                read_cols.append("kind")
            need_step = (step_min is not None or step_max is not None)
            if need_step and "step" in available and "step" not in read_cols:
                read_cols.append("step")
            if not read_cols:
                continue
            rgs = _selected_row_groups(pf, step_min, step_max) \
                if (need_step and "step" in available) else None
            if rgs is not None and not rgs:
                continue
            kw = {"columns": read_cols, "batch_size": int(batch_rows)}
            if rgs is not None:
                kw["row_groups"] = rgs
            for batch in pf.iter_batches(**kw):
                if kinds_arr is not None and "kind" in batch.schema.names:
                    batch = _filter_kinds(batch, kinds_arr)
                if need_step and "step" in batch.schema.names:
                    batch = _filter_steps(batch, step_min, step_max)
                if batch.num_rows == 0:
                    continue
                if len(read_cols) != len(cols):         # filter 用の余分な列を落とす
                    batch = batch.select(cols)
                yield batch


def iter_columns(run_dir, columns=None, **kw) -> Iterator[dict]:
    """`{列名: list}` の列 dict を逐次 yield(payload は **JSON 文字列のまま**)。

    zip ループで回す列指向の集計向け(`detect_regression.act_counts_from_l1` 系)。
    payload を dict にしたいなら `iter_events` を使う。
    """
    for batch in iter_record_batches(run_dir, columns, **kw):
        yield batch.to_pydict()


def _materialize(d: dict, n: int, plan: dict) -> Iterator[dict]:
    """列 dict → イベント dict 列。**`measure.stream_events` の実体化ループの写経**。

    キーの挿入順まで一致させる(int 列 → raw 列 → 欠測 None 列 → payload)。
    ここを measure と分岐させると出力同値が壊れるので、変更するときは
    `tests/test_l1_stream.py::test_iter_events_matches_load_events` を必ず見ること。
    """
    import json
    int_cols, raw_cols = plan["int_cols"], plan["raw_cols"]
    fill_none, want_payload = plan["fill_none"], plan["want_payload"]
    pays = d["payload"] if want_payload else None
    for i in range(n):
        row: dict = {}
        for c in int_cols:
            row[c] = int(d[c][i])
        for c in raw_cols:
            row[c] = d[c][i]
        for c in fill_none:
            row[c] = None
        if want_payload:
            raw = pays[i]
            try:
                row["payload"] = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                row["payload"] = {}
        yield row


def iter_events(run_dir, columns=None, *, kinds: Iterable | None = None,
                step_min: int | None = None, step_max: int | None = None,
                batch_rows: int = DEFAULT_BATCH_ROWS,
                stem: str = L1_STEM) -> Iterator[dict]:
    """`measure.load_events` の要素と**同一形**の dict を逐次 yield する。

    `kinds` / `step_min` / `step_max` は「`load_events` の結果をその条件で filter
    したもの」と厳密に同じ部分列を返す(順序も保つ)。絞り込みを掛けないときは
    `measure.stream_events` と完全に同じ列を返す。
    """
    plan = None
    for batch in iter_record_batches(run_dir, columns, kinds=kinds,
                                     step_min=step_min, step_max=step_max,
                                     batch_rows=batch_rows, stem=stem):
        if plan is None or plan["_schema"] != batch.schema.names:
            plan = _plan(set(batch.schema.names), columns)
            plan["_schema"] = batch.schema.names
        yield from _materialize(batch.to_pydict(), batch.num_rows, plan)


# --------------------------------------------------------------------------- #
# 汎用 parquet(L1 以外: panel/ 配下・l1b_llm 等)の逐次読み
# --------------------------------------------------------------------------- #
def iter_table_columns(path, columns=None,
                       batch_rows: int = DEFAULT_BATCH_ROWS) -> Iterator[dict]:
    """任意の parquet を列 dict で逐次 yield(L1 以外の表を全読みしないため)。

    在場 25万 × 10 日では `panel/panel.parquet` が 250万行になり、
    `read_table().to_pydict()` は数 GB の Python list を作る。行数に比例する集計
    (合計・最大・先頭/末尾)は本関数で有界にできる。
    """
    p = Path(path)
    with _open_shared(p) as fh:
        pf = pq.ParquetFile(fh)
        cols = None
        if columns is not None:
            available = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in available]
            if not cols:
                return
        kw = {"batch_size": int(batch_rows)}
        if cols is not None:
            kw["columns"] = cols
        for batch in pf.iter_batches(**kw):
            if batch.num_rows:
                yield batch.to_pydict()


def table_schema(path):
    """parquet の schema を footer だけから返す(1 行も読まない)。"""
    with _open_shared(Path(path)) as fh:
        return pq.ParquetFile(fh).schema_arrow


def table_rows(path) -> int:
    """parquet の行数を footer だけから返す(1 行も読まない)。"""
    with _open_shared(Path(path)) as fh:
        return int(pq.ParquetFile(fh).metadata.num_rows)


__all__ = [
    "DEFAULT_BATCH_ROWS", "L1_COLUMNS", "L1_INT_COLUMNS", "L1_STEM",
    "l1_paths", "row_count", "max_step", "kind_counts",
    "iter_record_batches", "iter_columns", "iter_events",
    "iter_table_columns", "table_schema", "table_rows",
]


def _main(argv=None) -> int:
    """CLI: ランの L1 の形(行数・最終 step・kind 内訳)を全件展開せずに出す。"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="L1 の形を全件展開せずに調べる")
    ap.add_argument("run_dir")
    ap.add_argument("--kinds", action="store_true", help="kind 内訳も出す(kind 列を1パス)")
    a = ap.parse_args(argv)
    out = {
        "run_dir": os.path.abspath(a.run_dir),
        "paths": [p.name for p in l1_paths(a.run_dir)],
        "rows": row_count(a.run_dir),
        "max_step": max_step(a.run_dir),
    }
    if a.kinds:
        out["kinds"] = dict(sorted(kind_counts(a.run_dir).items(),
                                   key=lambda kv: (-kv[1], kv[0])))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main())
