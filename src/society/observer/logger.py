"""observer L1/L1b/L2/L3 ロガー(D12)。追記専用・Parquet+zstd 出力。

frame 分離: シミュ本体は「起きたことを記録する」だけ。測定・集計は全て事後にこのログから。

D16 セグメント化: checkpoint 連携で flush_segment() が溜まったログを part-N として
書き出しメモリを解放する。finalize の flush() は part 群 + 残りバッファを結合して
従来どおり単一の l1_events.parquet 等を出す(part は削除)。checkpoint 無効時は part を
一切作らないので、出力・挙動は従来と完全同一(byte 級)。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import EVENT_KINDS, Event


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

    # ---- L1 ----
    def log(self, event: Event) -> None:
        if event.kind not in EVENT_KINDS:
            raise ValueError(
                f"未登録のイベント種類 '{event.kind}'。"
                f" observer/schema.py で register_event_kind() してから使うこと(D12 拡張契約)。"
            )
        self.events.append(event)

    # ---- L1b ----
    def log_llm_call(self, row: dict) -> None:
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
        tables = [pq.read_table(p) for p in parts]
        if table is not None and table.num_rows > 0:
            tables.append(table)
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        pq.write_table(combined, canonical, compression="zstd")
        for p in parts:
            p.unlink()
        return canonical
