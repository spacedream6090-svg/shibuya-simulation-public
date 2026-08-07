"""observer L1/L1b/L2/L3 ロガー(D12)。追記専用・Parquet+zstd 出力。

frame 分離: シミュ本体は「起きたことを記録する」だけ。測定・集計は全て事後にこのログから。

D16 セグメント化: checkpoint 連携で flush_segment() が溜まったログを part-N として
書き出しメモリを解放する。finalize の flush() は part 群 + 残りバッファを結合して
従来どおり単一の l1_events.parquet 等を出す(part は削除)。checkpoint 無効時は part を
一切作らないので、出力・挙動は従来と完全同一(byte 級)。

W2-6 finalize のメモリ有界化(第98バッチ 2026-08-07・conf `observer.finalize.streaming`・既定 OFF)
----------------------------------------------------------------------------------------------
上の「結合」を素直に書くと `pq.read_table` した **全 part を `pa.concat_tables` で 1 枚に
載せてから** 書くことになる(下の `_finalize_stream` の既定経路)。part を作る目的が
「走行中の RAM を解放すること」なのに、**最後の 1 回だけ全部を載せ直す**ので、ランの
ピークメモリは結局 L1 の総量で決まる。在場 25万 × 10 日の L1 は
`docs/plans/proposal-dp-u3-observe-250k.md` §2-4 の実測外挿で **42.7 GB・40.6 億イベント**
であり、これを 1 プロセスに載せることはできない。つまり **解析以前に「ランの書き終わり」で
落ちる**。W2-2 で発見したこの穴を塞ぐのが本トグルである。

ON にすると part を **1 つずつ開き row-group 単位で読み**、`pq.ParquetWriter` へ流し込む。
ピークメモリは「書き出す row-group 1 個ぶん」(既定 1,048,576 行 = pyarrow の既定 row group)
だけで、**part 数にも L1 総量にも依存しない**。

  - **既定 OFF は 1 バイトも変わらない**(分岐に入る前に return する構造で固定してある)。
  - ON でも **part が 1 つも無いラン**(checkpoint も flush_every_steps も無効)は
    従来と同じ「buffer を直接 write_table」経路に落ちる = バイト一致。
  - ON の出力は **行の内容・行順・スキーマが OFF と完全同値**だが、**parquet の
    バイト列は一致しない**(row-group の切れ目が変わるため)。バイト比較ではなく
    行比較で固定してある(tests/test_finalize_streaming.py)。
  - 副産物として **より安全**: OFF は canonical を直接上書きするので merge 中に落ちると
    canonical が壊れるが、ON は `<stem>.parquet.tmp` へ書いてから `os.replace` するので
    落ちても part と旧 canonical がそのまま残る。
  - part 間でスキーマがずれている場合(L1b のように行ごとにキー集合が違いうる表で、
    ある列が part-0 では全 None = null 型・part-1 では string になる等)、OFF の
    `concat_tables` は例外を投げるが、ON は `pa.unify_schemas(promote_options="permissive")`
    で統一してから書く(欠測は null で埋める = 偽の値を作らない)。

★観察ランでは ON を推奨する。25万体ランで OFF のままだと finalize で落ちる。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import EVENT_KINDS, Event

#: streaming finalize の既定 row-group 行数(pyarrow の `write_table` 既定と同じ 2**20)。
#: これがそのまま ON 時のピークメモリの単位になる(conf `observer.finalize.row_group_rows`)。
FINALIZE_ROW_GROUP_ROWS = 1 << 20
#: part を読むときの RecordBatch 行数(l1_stream.DEFAULT_BATCH_ROWS と同値)。
FINALIZE_BATCH_ROWS = 131_072


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


class ObserverLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.events: list[Event] = []          # L1
        self.llm_calls: list[dict] = []        # L1b
        self.metrics: list[dict] = []          # L2(step ごと1行)
        self.snapshots: list[dict] = []        # L3
        # ---- D16 セグメント化(checkpoint 連携)----
        self._kinds_flushed: Counter = Counter()   # part へ出し済みイベント種別の累計
        self._n_flushed: int = 0                   # part へ出し済みイベント総数
        self._seg: int = self._next_seg()          # resume 時は既存 part の続きから採番
        # 第57バッチ タスクC: 意図的な分割実行(10日×3等)で各チャンクが clean に finalize しても
        # 前チャンクの canonical を失わないためのフラグ。resume 時のみ Simulation.run() が True にする。
        # fresh ラン(resume でない)では常に False=finalize 挙動は従来と完全同一(byte 級不変)。
        self._resumed: bool = False
        # ---- W2-6: finalize のメモリ有界化(conf observer.finalize.streaming。既定 OFF)----
        # False = 従来の「全 part を concat_tables」経路(1 バイトも変えない)。
        # True  = part を 1 つずつ row-group 単位で読み ParquetWriter へ流す。
        # 配線は engine/simulation.py の ObserverLogger 生成直後 1 箇所だけ。
        self.streaming_finalize: bool = False
        self.finalize_row_group_rows: int = FINALIZE_ROW_GROUP_ROWS
        # ---- LLM 健全性 KPI(P0バッチ 2026-07-29)の O(1) 累積カウンタ ----
        # flush_segment がバッファを空にしても失われないよう、記録時点で加算する。
        # **出力(L1/L1b/L2/L3 の中身)には一切現れない**内部カウンタ = 既存ランとバイト一致。
        # observer.llm_health.enabled=false のときは誰も読まない(L2 列も出ない)。
        self.n_fallback_events: int = 0     # L1 kind="fallback"(発話系のパース失敗)の累計
        self.n_llm_rows: int = 0            # L1b 行(= LLM 呼び出し)の累計
        self.n_llm_cached: int = 0          # うちキャッシュ命中の累計
        # ---- 未定義行動レジスタ(第70バッチ IDEA②)の O(1) 累積カウンタ ----
        # fallback と**排他**(scheduler._log_reject がどちらか片方だけを出す)。
        # 出力には現れない内部カウンタ = 既定 OFF なら 0 のまま・L1/L2 ともバイト一致。
        self.n_undefined_action_events: int = 0
        # ---- 行為 → 思考の来歴リンク IF-1(observer.llm_link。既定 OFF=常に None)----
        # scheduler._apply が「いまこの行為を適用中」のあいだだけ
        # (llm_call_id, role, 行為者 agent_id)を立てる。既定 OFF では
        # scheduler が一時キー(_prov)を 1 度も積まないので **None のまま**
        # = 下の分岐に入らない = 既存ランと L1 バイト一致。設計は society/provlink.py。
        self._prov: tuple | None = None

    # ---- 来歴スコープ(scheduler._apply が開閉する。observer.llm_link ON のみ)----
    def set_prov(self, call_id: str, role: str, agent_id: int) -> None:
        self._prov = (str(call_id), str(role), int(agent_id))

    def clear_prov(self) -> None:
        self._prov = None

    # ---- L1 ----
    def log(self, event: Event) -> None:
        if event.kind not in EVENT_KINDS:
            raise ValueError(
                f"未登録のイベント種類 '{event.kind}'。"
                f" observer/schema.py で register_event_kind() してから使うこと(D12 拡張契約)。"
            )
        if event.kind == "fallback":       # LLM 健全性 KPI(文字列比較1回だけ=ホットパス影響最小)
            self.n_fallback_events += 1
        elif event.kind == "undefined_action":     # 未定義行動レジスタ(既定 OFF では 0 件)
            self.n_undefined_action_events += 1
        # PROV の wasInformedBy 辺 1 本(IF-1)。**行為者自身のイベントだけ**に刻む
        # (聞き手の hear / opinion_shift は別の主体の activity なので刻まない)。
        # 既に llm_call_id を持つイベント(llm_deliberate / plan_created 等)は上書きしない。
        if self._prov is not None and event.agent_id == self._prov[2]:
            if event.llm_call_id is None:
                event.llm_call_id = self._prov[0]
            event.payload.setdefault("llm_role", self._prov[1])
        self.events.append(event)

    # ---- L1b ----
    def log_llm_call(self, row: dict) -> None:
        self.n_llm_rows += 1
        if row.get("cached"):
            self.n_llm_cached += 1
        self.llm_calls.append(row)

    # ---- L2 ----
    def log_metrics(self, step: int, values: dict) -> None:
        self.metrics.append({"step": step, **values})

    # ---- L3 ----
    def log_snapshot(self, step: int, world_state: dict) -> None:
        self.snapshots.append({"step": step, "state": _dumps(world_state)})

    # ---- 累計(finalize の summary 用。part へ出した分 + バッファ)----
    def total_n_events(self) -> int:
        return self._n_flushed + len(self.events)

    def total_event_kinds(self) -> Counter:
        kinds = Counter(self._kinds_flushed)
        kinds.update(e.kind for e in self.events)
        return kinds

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _l1_table(self, events: list[Event]) -> pa.Table:
        return pa.table({
            "step":        pa.array([e.step for e in events], pa.int32()),
            "sim_min":     pa.array([e.sim_min for e in events], pa.int32()),
            "agent_id":    pa.array([e.agent_id for e in events], pa.int32()),
            "kind":        pa.array([e.kind for e in events], pa.string()),
            "x":           pa.array([e.x for e in events], pa.float32()),
            "y":           pa.array([e.y for e in events], pa.float32()),
            "payload":     pa.array([_dumps(e.payload) for e in events], pa.string()),
            "rng_stream":  pa.array([e.rng_stream for e in events], pa.string()),
            "llm_call_id": pa.array([e.llm_call_id for e in events], pa.string()),
        })

    def _rows_table(self, rows: list[dict]) -> pa.Table:
        keys = sorted({k for row in rows for k in row})
        return pa.table({k: [row.get(k) for row in rows] for k in keys})

    def _next_seg(self) -> int:
        """既存 l1 part 群の最大 index + 1(resume で採番衝突を避ける)。"""
        mx = -1
        if self.out_dir.is_dir():
            for p in self.out_dir.glob("l1_events.part-*.parquet"):
                try:
                    mx = max(mx, int(p.name[len("l1_events.part-"):].split(".")[0]))
                except ValueError:
                    pass
        return mx + 1

    # ---- セグメント書き出し(checkpoint 時に呼ぶ)----
    def flush_segment(self) -> None:
        """溜まったログを part-N として書き出し、メモリを解放する。"""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n = self._seg
        if self.events:
            pq.write_table(self._l1_table(self.events),
                           self.out_dir / f"l1_events.part-{n:04d}.parquet",
                           compression="zstd")
            self._kinds_flushed.update(e.kind for e in self.events)
            self._n_flushed += len(self.events)
            self.events = []
        for stem, attr in (("l1b_llm", "llm_calls"), ("l2_metrics", "metrics"),
                           ("l3_snapshots", "snapshots")):
            rows = getattr(self, attr)
            if rows:
                pq.write_table(self._rows_table(rows),
                               self.out_dir / f"{stem}.part-{n:04d}.parquet",
                               compression="zstd")
                setattr(self, attr, [])
        self._seg += 1

    # ---- 出力(finalize)----
    def flush(self) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        l1 = self._finalize_stream("l1_events", self._l1_table(self.events),
                                   always=True)
        if l1 is not None:
            paths["l1"] = l1
        for stem, rows in [("l1b_llm", self.llm_calls),
                           ("l2_metrics", self.metrics),
                           ("l3_snapshots", self.snapshots)]:
            table = self._rows_table(rows) if rows else None
            p = self._finalize_stream(stem, table, always=False)
            if p is not None:
                paths[stem] = p
        return paths

    def _finalize_stream(self, stem: str, table: pa.Table | None,
                         always: bool) -> Path | None:
        """part 群 + 残りバッファを結合して canonical parquet を出す。

        part が無い(=checkpoint 無効)場合は従来どおり buffer を直接書く(byte 級同一)。
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
        # 分割実行(resume で clean に finalize したチャンク)のときだけ、前チャンクが書いた
        # canonical を先頭に含める。straight ラン / crash-recovery(前 finalize 無し=canonical 不在)
        # では発火しないため従来と完全同一(_resumed=False の fresh ランはこの分岐に一切入らない)。
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
        RecordBatch 1 個ぶん」だけで、**part 数にも L1 総量にも依存しない**。
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
