"""finalize(part 群 → canonical parquet)の**共通実装 1 本**(W2-6 の本体 + W4-E の横展開)。

何を解く問題か
--------------
part 化(checkpoint / flush_every_steps)の目的は「走行中の RAM を解放すること」なのに、
finalize を素直に書くと `pq.read_table` した **全 part を `pa.concat_tables` で 1 枚に載せて
から**書くことになる。つまり **最後の 1 回だけ全部を載せ直す**ので、ランのピークメモリは
結局その表の総量で決まる。在場 25万 × 10 日の L1 は
`docs/plans/proposal-dp-u3-observe-250k.md` §2-4 の実測外挿で **42.7 GB・40.6 億イベント**
であり、1 プロセスに載らない = **解析以前に「ランの書き終わり」で落ちる**(W2-2 の発見)。

conf `observer.finalize.streaming`(既定 false)を ON にすると、part を **1 つずつ開き
row-group 単位で読み**、`pq.ParquetWriter` へ流し込む。ピークメモリは「書き出す row-group
1 個ぶん」(既定 1,048,576 行)だけで、**part 数にも総量にも依存しない**。

W4-E(第99バッチ 2026-08-08): サイドカーへの横展開 = 二重実装をやめる
--------------------------------------------------------------------
W2-6 は `ObserverLogger` にだけこの経路を実装した。ところが **同型の finalize を各サイドカーが
自前で持っていた**(indoor_tracks / org_ledger / finance / channels / cognition_g)。そこで
実装を本 module の `FinalizeStreamMixin` **1 本**に括り出し、全クラスがこれを継承する。

  - **同じ conf キーで全ファイルが同じモードになる**のが正(新 conf キーは増やさない)。
    「L1 だけ逐次書き・サイドカーだけ全載せ」という混在は、運用者が 1 つの判断で
    ピークメモリを決められなくなるので採らない。
  - 実装が 1 本なので、有界性の AST 検査(`read_table` / `concat_tables` 不在)も
    **1 箇所を見れば全クラスに効く**。テストは「各クラスの `_finalize_streaming` が
    本 module の関数と同一オブジェクトであること」を機械的に固定する
    (tests/test_finalize_streaming_sidecars.py)。

不変条件(既定 OFF は 1 バイトも変わらない)
--------------------------------------------
  - 既定 OFF は従来の `concat_tables` 経路そのもの(分岐に入る前に return する構造)。
  - ON でも **part が 1 つも無いラン**は従来の「buffer を直接 write_table」経路に落ちる
    = バイト一致。
  - ON の出力は **行の内容・行順・スキーマが OFF と完全同値**だが、**parquet のバイト列は
    一致しない**(row-group の切れ目が変わるため)。バイト比較ではなく行比較で固定する。
  - 副産物として **より安全**: OFF は canonical を直接上書きするので merge 中に落ちると
    canonical が壊れるが、ON は `<stem>.parquet.tmp` へ書いてから `os.replace` するので
    落ちても part と旧 canonical がそのまま残る。
  - part 間でスキーマがずれている場合(L1b のように行ごとにキー集合が違いうる表で、ある列が
    part-0 では全 None = null 型・part-1 では string になる等)、OFF の `concat_tables` は
    例外を投げるが、ON は `pa.unify_schemas(promote_options="permissive")` で統一してから
    書く(欠測は null で埋める = 偽の値を作らない)。

使う側の契約(mixin が触る属性はこれだけ)
------------------------------------------
  `self.out_dir`(Path)/ `self._resumed`(分割実行フラグ)/ `self.streaming_finalize` /
  `self.finalize_row_group_rows`。後ろ 2 つはクラス属性に既定値があるので、
  **conf を配線していないクラス・古い個体でも従来経路で動く**(欠測を偽の値で埋めない)。
"""
from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: streaming finalize の既定 row-group 行数(pyarrow の `write_table` 既定と同じ 2**20)。
#: これがそのまま ON 時のピークメモリの単位になる(conf `observer.finalize.row_group_rows`)。
FINALIZE_ROW_GROUP_ROWS = 1 << 20
#: part を読むときの RecordBatch 行数(l1_stream.DEFAULT_BATCH_ROWS と同値)。
FINALIZE_BATCH_ROWS = 131_072


class FinalizeStreamMixin:
    """part 群 + 残りバッファ → canonical parquet(既定経路 / streaming 経路の両方)。

    ★このクラスに**状態は持たせない**(バッファの持ち方は各サイドカーの自由)。
      触るのは out_dir / _resumed / streaming_finalize / finalize_row_group_rows だけ。
    """

    #: 既定値はクラス属性で持つ。__init__ で立て忘れたクラスや、属性を持たない古い個体でも
    #: **従来経路**へ落ちる(= 既定 OFF が壊れない側に倒す)。
    streaming_finalize: bool = False
    finalize_row_group_rows: int = FINALIZE_ROW_GROUP_ROWS
    _resumed: bool = False

    def _finalize_stream(self, stem: str, table: pa.Table | None) -> Path | None:
        """part 群 + 残りバッファを結合して canonical parquet を出す。

        part が無い(=checkpoint 無効)場合は従来どおり buffer を直接書く(byte 級同一)。
        分割実行(resume で clean finalize したチャンク)のときだけ既存 canonical を
        先頭に結合する(_resumed=False の fresh ランはこの分岐に一切入らない)。
        """
        parts = sorted(self.out_dir.glob(f"{stem}.part-*.parquet"))
        canonical = self.out_dir / f"{stem}.parquet"
        if not parts:
            if table is None:
                return None
            pq.write_table(table, canonical, compression="zstd")
            return canonical
        if self.streaming_finalize:            # W2-6: 有界メモリ経路(既定 OFF=ここに入らない)
            return self._finalize_streaming(stem, table, parts, canonical)
        tables = []
        if self._resumed and canonical.exists():
            tables.append(pq.read_table(canonical))
        tables += [pq.read_table(p) for p in parts]
        if table is not None and table.num_rows > 0:
            tables.append(table)
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        pq.write_table(combined, canonical, compression="zstd")
        for p in parts:
            p.unlink()
        return canonical

    # ------------------------------------------------------------------ #
    # W2-6: streaming finalize(observer.finalize.streaming=true のときだけ)
    # ------------------------------------------------------------------ #
    def _sources(self, parts: list[Path], canonical: Path) -> list[Path]:
        """結合する parquet を読む順に並べる(既定経路の tables の組み立てと同じ順序)。

        分割実行(resume で clean に finalize したチャンク)のときだけ前チャンクの
        canonical を先頭に置く。**この規則は既定経路と 1 文字も違わない**(同じ条件式)。
        """
        src: list[Path] = []
        if self._resumed and canonical.exists():
            src.append(canonical)
        return src + list(parts)

    def _merged_schema(self, sources: list[Path],
                       table: pa.Table | None) -> pa.Schema:
        """全ソースの schema を footer だけから読んで統一する(**1 行も読まない**)。

        part 間で列集合や型がずれうる(`_rows_table` は part ごとに「その part に現れた
        キーの和集合」で表を作るので、ある列が part-0 に無い / part-0 では全 None =
        null 型ということが起きる)。`promote_options="permissive"` は null → 具体型の
        昇格と欠測列の補完を許す = **欠測を偽の値で埋めず null のまま揃える**。
        """
        schemas = []
        for src in sources:
            schemas.append(pq.read_schema(src))
        if table is not None and table.num_rows > 0:
            schemas.append(table.schema)
        if len(schemas) == 1:
            return schemas[0]
        return pa.unify_schemas(schemas, promote_options="permissive")

    @staticmethod
    def _align(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
        """統一 schema へ揃える(欠測列は null で補い、型は cast する)。"""
        if tbl.schema.equals(schema):
            return tbl
        cols = []
        names = set(tbl.schema.names)
        for field in schema:
            if field.name in names:
                col = tbl.column(field.name).cast(field.type)
            else:                              # 欠測列は null で埋める(偽の値を作らない)
                col = pa.chunked_array([pa.nulls(tbl.num_rows, field.type)],
                                       type=field.type)
            cols.append(col)
        return pa.Table.from_arrays(cols, schema=schema)

    def _finalize_streaming(self, stem: str, table: pa.Table | None,
                            parts: list[Path], canonical: Path) -> Path:
        """part を 1 つずつ row-group 単位で読み、ParquetWriter へ流して canonical を作る。

        ピークメモリ = 「まだ書き出していない row-group 1 個ぶん」+「読み込み中の
        RecordBatch 1 個ぶん」だけで、**part 数にも総量にも依存しない**。
        全部を 1 枚の Table に載せる操作(`read_table` / `concat_tables`)を
        **この経路では 1 度も呼ばない**ことが有界性の構造的な根拠であり、
        tests/test_finalize_streaming.py が AST で固定している。

        出力は既定経路と **行の内容・行順・スキーマが同値**(バイト列は row-group の
        切れ目が変わるので一致しない)。書き込み先は一時ファイルで、成功したときだけ
        `os.replace` で canonical を差し替える(merge 中に落ちても canonical と part が
        そのまま残る = 既定経路より安全)。
        """
        sources = self._sources(parts, canonical)
        schema = self._merged_schema(sources, table)
        tmp = self.out_dir / f"{stem}.parquet.tmp"
        rg_rows = max(1, int(self.finalize_row_group_rows))
        buf: list[pa.RecordBatch] = []
        n_buf = 0

        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        try:
            def _drain() -> None:
                nonlocal buf, n_buf
                if not buf:
                    return
                writer.write_table(pa.Table.from_batches(buf, schema=schema))
                buf, n_buf = [], 0

            for src in sources:
                pf = pq.ParquetFile(src)
                try:
                    for batch in pf.iter_batches(batch_size=FINALIZE_BATCH_ROWS):
                        if batch.num_rows == 0:
                            continue
                        aligned = self._align(pa.Table.from_batches([batch]), schema)
                        buf.extend(aligned.to_batches())
                        n_buf += aligned.num_rows
                        if n_buf >= rg_rows:
                            _drain()
                finally:
                    pf.close()
            if table is not None and table.num_rows > 0:
                aligned = self._align(table, schema)
                buf.extend(aligned.to_batches())
                n_buf += aligned.num_rows
            _drain()
        finally:
            writer.close()
        os.replace(tmp, canonical)             # 読み終えてから差し替える(Windows でも可)
        for p in parts:
            p.unlink()
        return canonical


# --------------------------------------------------------------------------- #
# conf 配線(**唯一の源**。L1 もサイドカーもこの 2 関数しか通らない)
# --------------------------------------------------------------------------- #
def cfg_of_config(cfg) -> dict:
    """resolved config から finalize 設定を取り出す(欠測は従来経路の既定値)。

    conf ブロックごと無い古い config でも既定 OFF で動く(= 既定は 1 バイトも変わらない)。
    """
    node = None
    obs = cfg.get("observer", None) if hasattr(cfg, "get") else None
    if obs is not None:
        node = obs.get("finalize", None)
    if node is None:
        return {"streaming": False, "row_group_rows": FINALIZE_ROW_GROUP_ROWS}
    return {"streaming": bool(node.get("streaming", False)),
            "row_group_rows": int(node.get("row_group_rows",
                                           FINALIZE_ROW_GROUP_ROWS))}


def apply_cfg(sink, fincfg: dict | None) -> None:
    """1 つの出力層(logger / サイドカー)へ設定を配る。

    ★**OFF では属性を 1 つも触らない**。既定ランは各クラスの初期値のまま = 既存ランと
      バイト一致であることが、この 1 行の early return で構造的に保証される。
    """
    if not fincfg or not fincfg.get("streaming", False):
        return
    sink.streaming_finalize = True
    sink.finalize_row_group_rows = int(fincfg.get("row_group_rows",
                                                  FINALIZE_ROW_GROUP_ROWS))
