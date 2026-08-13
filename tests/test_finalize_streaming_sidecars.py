"""W4-E: streaming finalize のサイドカー横展開(observer.finalize.streaming。既定 OFF)のテスト。

背景
----
W2-6(第98バッチ)は `ObserverLogger` の finalize だけをメモリ有界化した。ところが
**同型の finalize を各サイドカーが自前で持っていた**(indoor_tracks / org_ledger / finance /
channels / cognition_g)。part 化の目的は「走行中の RAM を解放すること」なのに、finalize で
全 part を `read_table` → `concat_tables` すると **最後の 1 回だけ全部を載せ直す**ので、
そのファイルのピークメモリは総量で決まったままになる。

W4-E は実装を `society/observer/finalize.py` の `FinalizeStreamMixin` **1 本**へ括り出し、
**同一 conf キー** `observer.finalize.streaming` で全ファイルが同じモードになるようにした
(サイドカー用の別キーは作らない)。

受入条件(logger 側 tests/test_finalize_streaming.py と同じ様式)
  (1) 既定 OFF は **W4-E 以前の経路と parquet のバイト列まで一致**
      (本ファイル内に旧経路の参照実装を持ち、それと sha256 比較する)。
      part が 1 つも無いケースは **ON でもバイト一致**(従来経路へ落ちる)。
  (2) ON は OFF と **行の内容・行順・スキーマが完全同値**(バイト一致は保証しない)。
      空 / part 1 個 / 端数あり / 端数なし / resume 跨ぎ で固定。
  (3) 有界性の**構造的**根拠: ON の経路に `read_table` / `concat_tables` が 1 つも無い(AST)。
      加えて **全サイドカーが同一の関数オブジェクトを使う**こと(= 二重実装の再発防止)と、
      `src/society` のどこにも自前の `_finalize_stream` が残っていないことを固定する。
  (4) 配線: conf 1 キーが logger と全サイドカーへ届く(Simulation の属性を機械走査するので、
      **新しいサイドカーを足して配線を忘れたら落ちる**)。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from society import economy_sfc as economy_mod
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import channels as channels_mod
from society.observer import finalize as finalize_mod
from society.observer import indoor_tracks as indoor_mod
from society.observer import logger as logger_mod
from society.observer import org_ledger as org_mod
from society.observer.finalize import FinalizeStreamMixin

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "society"

#: 本 mixin を使う全クラス(logger + サイドカー 5 本)。
ALL_SINKS = (
    logger_mod.ObserverLogger,
    indoor_mod.IndoorTracks,
    org_mod.OrgLedger,
    channels_mod.ChannelsSidecar,
    channels_mod.CognitionGSidecar,
    economy_mod.FinanceLedger,
)


# --------------------------------------------------------------------------- #
# サイドカーごとの仕様(生成 / 1 行追加 / 現バッファの表)
# --------------------------------------------------------------------------- #
class Spec:
    def __init__(self, name, stems, make, add, table_of):
        self.name, self.stems = name, stems
        self.make, self.add, self.table_of = make, add, table_of

    def __repr__(self):                                    # pytest の id 用
        return self.name


def _indoor_table(sc, stem):
    if stem == indoor_mod._SAMPLES:
        return sc._samples_table(sc.samples) if sc.samples else None
    return sc._contacts_table(sc.contacts) if sc.contacts else None


def _rows_table(sc, _stem):
    return sc._table(sc.rows) if sc.rows else None


SPECS = [
    Spec("indoor_tracks",
         (indoor_mod._SAMPLES, indoor_mod._CONTACTS),
         lambda d: indoor_mod.IndoorTracks(d),
         lambda sc, i: (sc.add_samples([(i, i * 0.5, "bldg", i % 3,
                                         float(i), float(-i), i % 4)]),
                        sc.add_contacts([(float(i), i, i + 1, "meet",
                                          1.5, "bldg", i % 3)])),
         _indoor_table),
    Spec("org_ledger", ("org_ledger",),
         lambda d: org_mod.OrgLedger(d),
         lambda sc, i: sc.add_rows([(i // 2, f"org{i}", i, 1.5 * i, 2.5 * i, i, 10 * i)]),
         _rows_table),
    Spec("finance", ("finance",),
         lambda d: economy_mod.FinanceLedger(d),
         lambda sc, i: sc.add_rows([(i // 2, i, 1.0 * i, i, 0, -1.0 * i, 2.0 * i,
                                     3.0 * i, 4.0 * i, 5.0 * i, 6.0 * i,
                                     7.0 * i, 8.0 * i, f"ch{i}", 9.0 * i)]),
         _rows_table),
    Spec("channels", ("channels",),
         lambda d: channels_mod.ChannelsSidecar(d, ("alpha", "beta")),
         lambda sc, i: sc.add_rows([(i, 10 * i, i % 5, 0.25 * i, -0.5 * i)]),
         _rows_table),
    Spec("cognition_g", ("cognition_g",),
         lambda d: channels_mod.CognitionGSidecar(d, ("g_a", "theta")),
         lambda sc, i: sc.add_rows([(i, 10 * i, i % 5, 0.75 * i, 1.25 * i)]),
         _rows_table),
]

#: (名前, 行数, 「何行目の直後に flush_segment するか」)
SCENARIOS = [
    ("no_parts", 6, ()),                 # part 無し = buffer 直書き(ON でもバイト一致)
    ("parts_and_remainder", 7, (2, 4, 6)),   # part 3 個 + 残バッファ 1 行
    ("no_remainder", 6, (2, 4, 6)),      # part 3 個・残バッファ空
    ("single_part", 4, (4,)),            # part 1 個(concat が tables[0] を素通しする側)
    ("empty", 0, ()),                    # 何も無い = canonical を作らない
]


# --------------------------------------------------------------------------- #
# ヘルパ
# --------------------------------------------------------------------------- #
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build(spec: Spec, d: Path, streaming: bool, n_rows: int, flush_at) -> object:
    sc = spec.make(d)
    sc.streaming_finalize = streaming
    for i in range(n_rows):
        spec.add(sc, i)
        if (i + 1) in set(flush_at):
            sc.flush_segment()
    return sc


def _legacy_merge(parts, table, dest: Path) -> None:
    """W4-E **以前**の各サイドカーの既定経路を逐語で再現した参照実装。

    ここが本体と食い違えば「既定 OFF のバイト列が変わった」ということなので、
    **この関数は触ってはいけない**(直すべきは本体側)。
    """
    if not parts:
        pq.write_table(table, dest, compression="zstd")
        return
    tables = [pq.read_table(p) for p in parts]
    if table is not None and table.num_rows > 0:
        tables.append(table)
    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    pq.write_table(combined, dest, compression="zstd")


def _assert_same_content(a: Path, b: Path, stems) -> None:
    """行の内容・行順・スキーマの完全同値(バイト一致は要求しない)。"""
    for stem in stems:
        pa_, pb = a / f"{stem}.parquet", b / f"{stem}.parquet"
        assert pa_.exists() == pb.exists(), f"{stem}: 片方だけ canonical が無い"
        if not pa_.exists():
            continue
        ra = pq.read_table(pa_).to_pylist()
        rb = pq.read_table(pb).to_pylist()
        assert len(ra) == len(rb), f"{stem}: 行数不一致 {len(ra)} vs {len(rb)}"
        assert ra == rb, f"{stem}: 行内容 or 行順が不一致"
        assert pq.read_schema(pa_).equals(pq.read_schema(pb)), f"{stem}: スキーマ不一致"


# --------------------------------------------------------------------------- #
# (1) 既定 OFF は W4-E 以前とバイト列まで一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s[0])
def test_default_path_is_byte_identical_to_legacy(tmp_path, spec, scenario):
    label, n_rows, flush_at = scenario
    d = tmp_path / f"off_{spec.name}_{label}"
    sc = _build(spec, d, False, n_rows, flush_at)

    refs = {}                       # finalize は part を消すので**先に**参照を作る
    for stem in spec.stems:
        parts = sorted(d.glob(f"{stem}.part-*.parquet")) if d.is_dir() else []
        table = spec.table_of(sc, stem)
        if not parts and table is None:
            continue                # 何も出ないケース(empty)
        ref = tmp_path / f"ref_{spec.name}_{label}_{stem}.parquet"
        _legacy_merge(parts, table, ref)
        refs[stem] = ref

    sc.finalize()
    if n_rows == 0:
        assert not refs
        for stem in spec.stems:
            assert not (d / f"{stem}.parquet").exists(), "空なのに canonical を作った"
        return
    assert refs, "参照が 1 本も作られていない(シナリオ設定の誤り)"
    for stem, ref in refs.items():
        assert _sha(d / f"{stem}.parquet") == _sha(ref), \
            f"{spec.name}/{stem}: 既定 OFF のバイト列が W4-E 以前と違う"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_streaming_without_parts_is_byte_identical(tmp_path, spec):
    """part が 1 つも無ければ ON でも従来経路(buffer 直書き)= バイト一致。"""
    off = _build(spec, tmp_path / f"np_off_{spec.name}", False, 6, ())
    off.finalize()
    on = _build(spec, tmp_path / f"np_on_{spec.name}", True, 6, ())
    on.finalize()
    for stem in spec.stems:
        assert _sha(off.out_dir / f"{stem}.parquet") == _sha(on.out_dir / f"{stem}.parquet"), \
            f"{spec.name}/{stem}: part 無しで ON がバイト列を変えた"


# --------------------------------------------------------------------------- #
# (2) ON は OFF と行内容・行順・スキーマが完全同値
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s[0])
def test_streaming_matches_default(tmp_path, spec, scenario):
    label, n_rows, flush_at = scenario
    off = _build(spec, tmp_path / f"c_off_{spec.name}_{label}", False, n_rows, flush_at)
    off.finalize()
    on = _build(spec, tmp_path / f"c_on_{spec.name}_{label}", True, n_rows, flush_at)
    on.finalize()
    _assert_same_content(off.out_dir, on.out_dir, spec.stems)
    if n_rows:
        assert any((on.out_dir / f"{s}.parquet").exists() for s in spec.stems)
    for stem in spec.stems:                       # part は結合後に消え、残骸も残らない
        assert not list(on.out_dir.glob(f"{stem}.part-*.parquet"))
    assert not list(on.out_dir.glob("*.parquet.tmp")), "一時ファイルが残っている"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_streaming_resume_chunk_matches_default(tmp_path, spec):
    """分割実行(前チャンクの canonical を先頭に結合する経路)でも ON == OFF。

    第57バッチ タスクC の `_resumed` 経路。ここを取り違えると **resume で行が消える /
    二重になる**ので、サイドカーごとに固定する。
    """
    dirs = {}
    for streaming in (False, True):
        d = tmp_path / f"rs_{spec.name}_{int(streaming)}"
        first = _build(spec, d, streaming, 4, (2, 4))     # チャンク1 = part 2 個
        first.finalize()                                   # canonical が確定
        second = spec.make(d)                              # チャンク2(resume 相当)
        second.streaming_finalize = streaming
        second._resumed = True
        for i in range(4, 7):
            spec.add(second, i)
            if i == 5:
                second.flush_segment()
        second.finalize()
        dirs[streaming] = d
    _assert_same_content(dirs[False], dirs[True], spec.stems)
    for stem in spec.stems:                                # 7 行(0..6)が 1 度ずつ
        rows = pq.read_table(dirs[True] / f"{stem}.parquet").to_pylist()
        assert len(rows) == 7, f"{spec.name}/{stem}: resume で行数が {len(rows)}"


def test_row_group_rows_changes_layout_not_content(tmp_path):
    """row_group_rows は**ピークメモリの単位**を変えるだけで行内容は変えない(代表 1 本)。"""
    big = _build(SPECS[3], tmp_path / "rg_big", True, 7, (2, 4, 6))
    big.finalize_row_group_rows = 1 << 20
    big.finalize()
    small = _build(SPECS[3], tmp_path / "rg_small", True, 7, (2, 4, 6))
    small.finalize_row_group_rows = 2
    small.finalize()
    _assert_same_content(big.out_dir, small.out_dir, SPECS[3].stems)
    assert (pq.read_metadata(small.out_dir / "channels.parquet").num_row_groups
            > pq.read_metadata(big.out_dir / "channels.parquet").num_row_groups)


# --------------------------------------------------------------------------- #
# (3) 有界性の構造的根拠 + 二重実装の再発防止(AST / 同一性)
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

SHARED_METHODS = ("_finalize_stream", "_finalize_streaming",
                  "_sources", "_merged_schema", "_align")


def test_every_sink_uses_the_one_shared_implementation():
    """logger も 5 サイドカーも **同一の関数オブジェクト**を使う(= 二重実装が無い)。

    ★これが本タスクの本体。ここが緑なら、下の AST 検査 1 回で全ファイルの有界性が言える。
    """
    for cls in ALL_SINKS:
        assert issubclass(cls, FinalizeStreamMixin), f"{cls.__name__} が mixin を継承していない"
        for meth in SHARED_METHODS:
            assert getattr(cls, meth) is getattr(FinalizeStreamMixin, meth), \
                f"{cls.__name__}.{meth} が共通実装を上書きしている(二重実装の再発)"


def test_shared_streaming_path_never_materializes_everything():
    """ON の経路に `read_table` / `concat_tables` が 1 つも無い(= 有界性の構造的根拠)。

    ★対照として、既定経路には**両方ある**ことも同時に固定する(空回り防止)。"""
    on = _names(_fn_ast(FinalizeStreamMixin._finalize_streaming))
    assert not (on & WHOLE_READ_IDENTIFIERS), \
        f"streaming 経路に全読み識別子がある: {sorted(on & WHOLE_READ_IDENTIFIERS)}"
    helper = _names(_fn_ast(FinalizeStreamMixin._merged_schema))
    assert not (helper & WHOLE_READ_IDENTIFIERS)
    assert "read_schema" in helper                       # footer だけ = 1 行も読まない

    off = _names(_fn_ast(FinalizeStreamMixin._finalize_stream))
    assert WHOLE_READ_IDENTIFIERS <= off, \
        "既定経路から全読みが消えている(このテストが空回りしている)"

    src = inspect.getsource(FinalizeStreamMixin._finalize_streaming)
    assert "iter_batches" in src, "逐次読みになっていない"
    assert "finalize_row_group_rows" in src, "row-group の行数上限が効いていない"
    assert "os.replace" in src, "一時ファイル経由の差し替えになっていない"


def test_no_module_keeps_a_private_merge_implementation():
    """`src/society` のどこにも自前の `_finalize_stream` / `concat_tables` が残っていない。

    唯一の例外は `observer/finalize.py`(共通実装そのもの)。AST で見るので、
    docstring の「旧: concat_tables で…」という説明文には反応しない。
    """
    allowed = (REPO_SRC / "observer" / "finalize.py").resolve()
    offenders = []
    for path in sorted(REPO_SRC.rglob("*.py")):
        if path.resolve() == allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in {"_finalize_stream", "_finalize_streaming"}):
                offenders.append(f"{path.name}:{node.lineno} def {node.name}")
            if isinstance(node, ast.Attribute) and node.attr == "concat_tables":
                offenders.append(f"{path.name}:{node.lineno} concat_tables")
    assert not offenders, f"finalize の実装が分裂している: {offenders}"


# --------------------------------------------------------------------------- #
# (4) 配線: conf 1 キーが logger と全サイドカーへ届く
# --------------------------------------------------------------------------- #
_ALL_SIDECARS_ON = [
    "indoor.enabled=true", "indoor.tracks.enabled=true",
    "work.service.enabled=true", "work.service.ledger.enabled=true",
    "economy.org_accounting.enabled=true", "economy.org_accounting.sidecar=true",
    "cognition.channels.enabled=true",
    "cognition.fire.enabled=true", "cognition.g_update.enabled=true",
    "observer.roster_daily.enabled=true",       # A13 日次入場者名簿(レーン丙 2)
]


def _sinks_of(sim) -> dict:
    """Simulation が持つ「finalize する層」を**属性走査で**全部拾う。

    人手の列挙ではないので、**新しいサイドカーを足して配線を忘れたら**このテストが落ちる。
    """
    return {k: v for k, v in vars(sim).items() if isinstance(v, FinalizeStreamMixin)}


def _sim(tmp_path, name, *dots):
    dot = ["run.seed=42", "run.n_agents=12", "run.n_steps=1", f"run.name={name}",
           "model.backend=mock", *dots]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_conf_reaches_logger_and_every_sidecar(tmp_path):
    sim = _sim(tmp_path, "wire_on", *_ALL_SIDECARS_ON,
               "observer.finalize.streaming=true", "observer.finalize.row_group_rows=7")
    sinks = _sinks_of(sim)
    assert set(sinks) == {"logger", "indoor_tracks", "org_ledger_sc",
                          "finance_sc", "channels_sc", "cognition_g_sc",
                          "roster_sc"}, \
        f"finalize する層の顔ぶれが変わった: {sorted(sinks)}"
    for name, sink in sinks.items():
        assert sink.streaming_finalize is True, f"{name} に streaming が届いていない"
        assert sink.finalize_row_group_rows == 7, f"{name} に row_group_rows が届いていない"


def test_default_leaves_every_sidecar_on_the_legacy_path(tmp_path):
    """既定(conf 未指定)では 1 つも streaming にならない = 既存ランとバイト一致。"""
    sim = _sim(tmp_path, "wire_off", *_ALL_SIDECARS_ON)
    sinks = _sinks_of(sim)
    assert len(sinks) == 7
    for name, sink in sinks.items():
        assert sink.streaming_finalize is False, f"{name} が既定で streaming になっている"
        assert sink.finalize_row_group_rows == finalize_mod.FINALIZE_ROW_GROUP_ROWS


def test_apply_cfg_touches_nothing_when_off():
    """OFF では属性を 1 つも書かない(既定バイト不変の構造的根拠)。"""
    class _Probe(FinalizeStreamMixin):
        def __setattr__(self, k, v):               # pragma: no cover - 呼ばれたら失敗
            raise AssertionError(f"OFF なのに属性 {k} を書いた")

    probe = _Probe()
    finalize_mod.apply_cfg(probe, {"streaming": False, "row_group_rows": 7})
    finalize_mod.apply_cfg(probe, None)
    finalize_mod.apply_cfg(probe, {})
    assert probe.streaming_finalize is False


def test_cfg_of_config_defaults_are_the_legacy_path():
    cfg = load_config()
    got = finalize_mod.cfg_of_config(cfg)
    assert got == {"streaming": False,
                   "row_group_rows": finalize_mod.FINALIZE_ROW_GROUP_ROWS}
    assert finalize_mod.cfg_of_config({}) == got     # conf ブロックごと無くても既定 OFF


def test_no_new_conf_key_for_sidecars():
    """サイドカー用の別キーを増やしていない(1 つの判断で全ファイルが同じモード)。"""
    cfg = load_config()
    assert sorted(cfg.observer.finalize.keys()) == ["row_group_rows", "streaming"]
