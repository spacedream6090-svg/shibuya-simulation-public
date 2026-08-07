"""observer L1/L1b/L2/L3 ロガー(D12)。追記専用・Parquet+zstd 出力。

frame 分離: シミュ本体は「起きたことを記録する」だけ。測定・集計は全て事後にこのログから。

D16 セグメント化: checkpoint 連携で flush_segment() が溜まったログを part-N として
書き出しメモリを解放する。finalize の flush() は part 群 + 残りバッファを結合して
従来どおり単一の l1_events.parquet 等を出す(part は削除)。checkpoint 無効時は part を
一切作らないので、出力・挙動は従来と完全同一(byte 級)。

W2-6 finalize のメモリ有界化(第98バッチ 2026-08-07・conf `observer.finalize.streaming`・既定 OFF)
----------------------------------------------------------------------------------------------
★実装は W4-E(第99バッチ 2026-08-08)で `observer/finalize.py` の `FinalizeStreamMixin` へ
括り出した(サイドカー 5 本が同型の finalize を自前で持っていたため = 二重実装の解消)。
本 class はそれを継承するだけで、経路も既定値も 1 バイトも変わらない。以下の説明は
そのまま有効(詳細と横展開の設計は `observer/finalize.py` の docstring)。

上の「結合」を素直に書くと `pq.read_table` した **全 part を `pa.concat_tables` で 1 枚に
載せてから** 書くことになる(`FinalizeStreamMixin._finalize_stream` の既定経路)。part を作る目的が
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
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import (FINALIZE_BATCH_ROWS, FINALIZE_ROW_GROUP_ROWS,  # noqa: F401
                       FinalizeStreamMixin)
from .schema import EVENT_KINDS, Event

# FINALIZE_ROW_GROUP_ROWS / FINALIZE_BATCH_ROWS は observer/finalize.py が唯一の定義。
# ここでの再輸出は既存の参照(tests・スクリプト)を壊さないためだけのもの。


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


class ObserverLogger(FinalizeStreamMixin):
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
        # 実装は FinalizeStreamMixin(observer/finalize.py)。配線は
        # engine/simulation.py の finalize.apply_cfg 1 箇所だけ(サイドカーと同じ関数)。
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
    # 結合の実装は FinalizeStreamMixin(observer/finalize.py)に 1 本だけ置いてある。
    # l1_events はバッファが空でも `_l1_table([])`(0 行の表)を渡すので canonical が必ず出る
    # (他の 3 本は行が無ければ None = ファイルを作らない、という従来どおりの線引き)。
    def flush(self) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        l1 = self._finalize_stream("l1_events", self._l1_table(self.events))
        if l1 is not None:
            paths["l1"] = l1
        for stem, rows in [("l1b_llm", self.llm_calls),
                           ("l2_metrics", self.metrics),
                           ("l3_snapshots", self.snapshots)]:
            table = self._rows_table(rows) if rows else None
            p = self._finalize_stream(stem, table)
            if p is not None:
                paths[stem] = p
        return paths
