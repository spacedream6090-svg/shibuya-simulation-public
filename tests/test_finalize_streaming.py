"""W2-6 ①: finalize のメモリ有界化(observer.finalize.streaming。既定 OFF)のテスト。

背景(W2-2 の発見): `ObserverLogger._finalize_stream` は finalize 時に **全 part を
read_table して concat_tables** する。part 化の目的が「走行中の RAM を解放すること」なのに
最後の 1 回だけ全部を載せ直すので、ランのピークメモリは L1 の総量で決まる。在場 25万 ×
10 日の L1 は 42.7 GB・40.6 億イベント(proposal-dp-u3-observe-250k.md §2-4)なので、
**解析以前に「ランの書き終わり」で落ちる**。

受入条件:
  (1) 既定 OFF は現行経路と **parquet のバイト列まで一致**(明示 false も既定と同一)。
      part が 1 つも無いランは **ON でもバイト一致**(従来経路へ落ちる)。
  (2) ON は OFF と **行の内容・行順・スキーマが完全同値**(バイト一致は保証しない)。
      部分バッファ / 端数なし / part 1 個 / resume 跨ぎ / 列集合のずれた part 群 で固定。
  (3) 有界性の**構造的**根拠: ON の経路に「全部を 1 枚に載せる」識別子
      (`read_table` / `concat_tables`)が 1 つも無いことを AST で固定する。
  (4) レジストリ宣言(repro_tier / affects_k / fingerprint_risk)と未宣言検出。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import logger as logger_mod
from society.observer.logger import ObserverLogger

STEMS = ("l1_events", "l2_metrics", "l3_snapshots")


# --------------------------------------------------------------------------- #
# ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int, **ov):
    dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int, **ov) -> Path:
    d = tmp_path / name
    Simulation(_cfg(name, n_steps, **ov), out_dir=d).run()
    return d


def _rows(run_dir: Path, stem: str) -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _schema(run_dir: Path, stem: str):
    return pq.read_schema(Path(run_dir) / f"{stem}.parquet")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_same_content(a: Path, b: Path, stems=STEMS) -> None:
    """行の内容・行順・スキーマの完全同値(バイト一致は要求しない)。"""
    for stem in stems:
        pa_, pb = a / f"{stem}.parquet", b / f"{stem}.parquet"
        assert pa_.exists() == pb.exists(), f"{stem}: 片方だけ canonical が無い"
        if not pa_.exists():
            continue
        ra, rb = _rows(a, stem), _rows(b, stem)
        assert len(ra) == len(rb), f"{stem}: 行数不一致 {len(ra)} vs {len(rb)}"
        assert ra == rb, f"{stem}: 行内容 or 行順が不一致"
        assert _schema(a, stem).equals(_schema(b, stem)), f"{stem}: スキーマ不一致"


# --------------------------------------------------------------------------- #
# (1) 既定 OFF は 1 バイトも変わらない
# --------------------------------------------------------------------------- #
def test_default_is_off():
    cfg = load_config()
    assert cfg.observer.finalize.streaming is False
    assert int(cfg.observer.finalize.row_group_rows) == (1 << 20)
    lg = ObserverLogger(Path("."))
    assert lg.streaming_finalize is False
    assert lg.finalize_row_group_rows == logger_mod.FINALIZE_ROW_GROUP_ROWS


def test_default_equals_explicit_false_byte_for_byte(tmp_path):
    """明示 false は既定と **parquet のバイト列まで**一致(part を作るランで確認)。"""
    ov = {"observer.flush_every_steps": 12}
    a = _run(tmp_path, "def_off", 40, **ov)
    b = _run(tmp_path, "exp_off", 40, **{**ov, "observer.finalize.streaming": "false"})
    for stem in STEMS:
        assert _sha(a / f"{stem}.parquet") == _sha(b / f"{stem}.parquet"), \
            f"{stem}: 既定と明示 false でバイト列が違う"


def test_streaming_on_without_parts_is_byte_identical(tmp_path):
    """part が 1 つも無いラン(flush も checkpoint も無効)は ON でもバイト一致。

    ON が触るのは『part 群の結合』だけで、buffer 直書き経路には手を入れていない。"""
    a = _run(tmp_path, "np_off", 30)
    b = _run(tmp_path, "np_on", 30, **{"observer.finalize.streaming": "true"})
    assert not list(b.glob("l1_events.part-*.parquet"))
    for stem in STEMS:
        assert _sha(a / f"{stem}.parquet") == _sha(b / f"{stem}.parquet"), \
            f"{stem}: part 無しランで ON がバイト列を変えた"


# --------------------------------------------------------------------------- #
# (2) ON は行内容が完全同値
# --------------------------------------------------------------------------- #
def test_streaming_matches_concat_with_partial_last_buffer(tmp_path):
    """端数あり(40 step / flush 15 = part 2 個 + 残バッファ 10 step)で行内容が同値。"""
    ov = {"observer.flush_every_steps": 15}
    off = _run(tmp_path, "pb_off", 40, **ov)
    on = _run(tmp_path, "pb_on", 40, **{**ov, "observer.finalize.streaming": "true"})
    assert len(_rows(off, "l1_events")) > 0
    _assert_same_content(off, on)
    # part は結合後に削除され canonical と一時ファイルの残骸が残らない
    assert not list(on.glob("l1_events.part-*.parquet"))
    assert not list(on.glob("*.parquet.tmp")), "一時ファイルが残っている"


def test_streaming_matches_concat_with_no_remainder(tmp_path):
    """端数なし(40 step / flush 20 = 最後の flush で buffer が空)でも同値。"""
    ov = {"observer.flush_every_steps": 20}
    off = _run(tmp_path, "nr_off", 40, **ov)
    on = _run(tmp_path, "nr_on", 40, **{**ov, "observer.finalize.streaming": "true"})
    _assert_same_content(off, on)


def test_streaming_with_single_part(tmp_path):
    """part 1 個(= concat の分岐が tables[0] を素通しする側)でも同値。"""
    ov = {"observer.flush_every_steps": 40}
    off = _run(tmp_path, "sp_off", 40, **ov)
    on = _run(tmp_path, "sp_on", 40, **{**ov, "observer.finalize.streaming": "true"})
    _assert_same_content(off, on)


def test_streaming_row_group_rows_does_not_change_content(tmp_path):
    """row_group_rows を極端に小さくしても(= row-group が細切れでも)行内容は不変。

    ピークメモリの単位を変えるだけのつまみであることの固定。"""
    ov = {"observer.flush_every_steps": 12, "observer.finalize.streaming": "true"}
    big = _run(tmp_path, "rg_big", 40, **ov)
    small = _run(tmp_path, "rg_small", 40, **{**ov, "observer.finalize.row_group_rows": 7})
    _assert_same_content(big, small)
    md = pq.read_metadata(small / "l1_events.parquet")
    assert md.num_row_groups > pq.read_metadata(big / "l1_events.parquet").num_row_groups


def test_streaming_resume_matches_straight(tmp_path):
    """resume 跨ぎ(分割実行で前チャンクの canonical を先頭結合する経路)でも同値。"""
    ov = {"observer.finalize.streaming": "true"}
    straight = _run(tmp_path, "rs_straight", 40, **ov)

    d = tmp_path / "rs_resumed"
    every = {"observer.checkpoint_every": 20}
    sim1 = Simulation(_cfg("rs_resumed", 20, **every, **ov), out_dir=d)
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, d / "checkpoint" / "ckpt-000020.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("rs_resumed", 40, **every, **ov), out_dir=d)
    sim2.run(resume_from=d)

    _assert_same_content(straight, d)
    assert not list(d.glob("*.parquet.tmp"))


def test_streaming_empty_buffer_and_empty_run(tmp_path):
    """空ラン(n_steps=0)は ON/OFF とも canonical が 0 行で一致し、落ちない。"""
    off = _run(tmp_path, "e_off", 0)
    on = _run(tmp_path, "e_on", 0, **{"observer.finalize.streaming": "true"})
    assert len(_rows(off, "l1_events")) == 0
    _assert_same_content(off, on, stems=("l1_events",))


# --------------------------------------------------------------------------- #
# (2b) part 間でスキーマがずれても壊れない(ロガー単体)
# --------------------------------------------------------------------------- #
def _mk_logger(d: Path, streaming: bool) -> ObserverLogger:
    lg = ObserverLogger(d)
    lg.streaming_finalize = streaming
    return lg


def test_streaming_unifies_schema_across_parts(tmp_path):
    """列集合・型が part 間でずれても、欠測を null のまま揃えて 1 本にできる。

    `_rows_table` は part ごとに『その part に現れたキーの和集合』で表を作るので、
    L1b のような行ごとにキーが違う表では part 間で列がずれる(ある part では
    値が全 None = null 型になることもある)。既定経路の concat_tables はこれで例外を
    投げるが、streaming は unify_schemas(permissive)で通す。"""
    d = tmp_path / "schema_drift"
    lg = _mk_logger(d, streaming=True)
    lg.log_llm_call({"purpose": "solo", "cached": True, "extra": None})
    lg.flush_segment()
    lg.log_llm_call({"purpose": "reply", "cached": False, "extra": "x", "newcol": 3})
    lg.flush_segment()
    lg.log_llm_call({"purpose": "dm", "cached": True, "extra": "y", "newcol": 5})
    paths = lg.flush()

    rows = pq.read_table(paths["l1b_llm"]).to_pylist()
    assert [r["purpose"] for r in rows] == ["solo", "reply", "dm"]
    assert rows[0]["newcol"] is None and rows[1]["newcol"] == 3
    assert rows[0]["extra"] is None and rows[2]["extra"] == "y"
    assert not list(d.glob("l1b_llm.part-*.parquet"))


def test_streaming_preserves_part_order(tmp_path):
    """part の連結順(index 昇順 → 残バッファ)が既定経路と同一。"""
    for streaming in (False, True):
        d = tmp_path / f"order_{int(streaming)}"
        lg = _mk_logger(d, streaming)
        for i in range(5):
            lg.log_metrics(i, {"v": float(i)})
            lg.flush_segment()
        lg.log_metrics(5, {"v": 5.0})
        lg.flush()
        got = [r["step"] for r in pq.read_table(d / "l2_metrics.parquet").to_pylist()]
        assert got == [0, 1, 2, 3, 4, 5], f"streaming={streaming}: 順序が壊れた"


def test_streaming_leaves_sources_intact_when_writer_fails(tmp_path):
    """merge が途中で失敗しても canonical と part を壊さない(一時ファイル経由の担保)。"""
    d = tmp_path / "crash"
    lg = _mk_logger(d, streaming=True)
    lg.log_metrics(0, {"v": 1.0})
    lg.flush_segment()
    lg.log_metrics(1, {"v": 2.0})
    lg.flush_segment()
    parts_before = sorted(p.name for p in d.glob("l2_metrics.part-*.parquet"))
    assert len(parts_before) == 2

    boom = RuntimeError("disk full")

    class _Boom:
        def __init__(self, *a, **kw):
            raise boom

    orig = pq.ParquetWriter
    pq.ParquetWriter = _Boom
    try:
        try:
            lg.flush()
        except RuntimeError as exc:
            assert exc is boom
        else:                                          # pragma: no cover
            raise AssertionError("失敗が伝播していない")
    finally:
        pq.ParquetWriter = orig

    assert sorted(p.name for p in d.glob("l2_metrics.part-*.parquet")) == parts_before
    assert not (d / "l2_metrics.parquet").exists(), "壊れた canonical が残った"


# --------------------------------------------------------------------------- #
# (3) 有界性の構造的根拠(AST)
# --------------------------------------------------------------------------- #
def _fn_ast(func) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


#: 「全部を 1 枚に載せる」= ピークが総量で決まる原因になる識別子。
WHOLE_READ_IDENTIFIERS = {"read_table", "concat_tables"}


def test_streaming_path_never_materializes_everything():
    """ON の経路に `read_table` / `concat_tables` が 1 つも無い(= 有界性の構造的根拠)。

    ★対照として、既定経路には**両方ある**ことも同時に固定する。片方だけ見ていると
    「実装が別関数へ移っただけ」で空回りするため。"""
    on = _names(_fn_ast(ObserverLogger._finalize_streaming))
    assert not (on & WHOLE_READ_IDENTIFIERS), \
        f"streaming 経路に全読み識別子がある: {sorted(on & WHOLE_READ_IDENTIFIERS)}"
    # スキーマ解決も footer だけ(read_schema)で 1 行も読まない
    helper = _names(_fn_ast(ObserverLogger._merged_schema))
    assert not (helper & WHOLE_READ_IDENTIFIERS)
    assert "read_schema" in helper

    off = _names(_fn_ast(ObserverLogger._finalize_stream))
    assert WHOLE_READ_IDENTIFIERS <= off, \
        "既定経路から全読みが消えている(このテストが空回りしている)"


def test_streaming_reads_in_batches_and_bounds_row_groups():
    """ON の経路が row-group 単位の逐次読み(iter_batches)と行数上限の両方を使う。"""
    src = inspect.getsource(ObserverLogger._finalize_streaming)
    assert "iter_batches" in src, "逐次読みになっていない"
    assert "finalize_row_group_rows" in src, "row-group の行数上限が効いていない"
    assert "os.replace" in src, "一時ファイル経由の差し替えになっていない"


def test_default_path_is_unreachable_without_the_flag():
    """既定 OFF では streaming 分岐に入らない(分岐条件が属性 1 つだけ)。"""
    src = inspect.getsource(ObserverLogger._finalize_stream)
    assert "if self.streaming_finalize:" in src


# --------------------------------------------------------------------------- #
# (4) レジストリ
# --------------------------------------------------------------------------- #
def test_registry_declares_streaming_finalize():
    f = R.BY_ID["observer.finalize.streaming"]
    assert f.repro_tier == "strict"
    assert f.affects_k is False
    assert f.fingerprint_risk == "none"
    assert "42.7" in f.description or "25万" in f.description


def test_no_undeclared_toggles_after_adding_finalize_block():
    assert R.undeclared_toggles(load_config()) == []


def test_verify_mode_keeps_streaming_finalize_available():
    """strict なので verify モード(最も厳しい)でも自動 OFF されない。"""
    cfg = load_config(["run.mode=verify", "observer.finalize.streaming=true"])
    gated, rep = R.apply_mode(cfg)
    assert gated.observer.finalize.streaming is True
    assert "observer.finalize.streaming" not in {d["id"] for d in rep["auto_disabled"]}
