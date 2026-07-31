"""観測チャンネル サイドカー(第80バッチ 2026-08-01。cognition.channels.enabled ON のみ)。

`src/society/cognition/channels.py` が step ごとに計算した o_c(t) を、**L1 とは別の**
Parquet(`channels.parquet`)へ書き出すだけの層。第83バッチの θ 較正パイロットと
`scripts/measure_sigma.py` がこれを読む。

**記録と動力学の分離**(第58 indoor_tracks / 第73 beliefs_ledger と同一の設計原則):
動力学(scheduler)は本層のバッファを読まない。本層は世界状態を読むだけで書かない。
したがって ON にしても L1/L2/L3・乱数・LLM 呼数は 1 バイトも変わらない
(= サイドカーが 1 本増えるだけ)。

セグメント化/finalize は observer/logger.py・indoor_tracks.py と同一流儀
(checkpoint 連携で part 化 → finalize で結合、resume の `_resumed` フラグで分割実行
チャンクの canonical を先頭結合)。これにより resume==straight のバイト一致を保つ
= **途中再開でサイドカーが二重記録しない**。

列
--
  step, sim_min, agent_id … 固定キー(int32)
  <channel.column> …………… 値(float32・**欠測は null**。0 で埋めない)

値列の順序は `cognition.channels.CHANNELS` の並びが唯一の源(本層は並べ替えない)。
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STEM = "channels"


class ChannelsSidecar:
    """観測チャンネル行の追記バッファ + セグメント/finalize(IndoorTracks と対の設計)。"""

    def __init__(self, out_dir, columns):
        self.out_dir = Path(out_dir)
        self.columns: tuple[str, ...] = tuple(columns)
        self.rows: list[tuple] = []
        self._n_flushed = 0
        self._seg = self._next_seg()
        # 分割実行(resume)で clean finalize しても前チャンクの canonical を失わない。
        self._resumed = False

    # ---- 記録(観測層が積む。動力学はこのバッファを読まない)----
    #  メソッド名は既存サイドカー(org_ledger)と揃えた `add_rows`。動力学から触ってよい
    #  メソッドは tests/test_indoor_invariance.py の allowlist で機械的に固定されている
    #  (書き込み一方向=動力学→observer だけを許す静的検査)。
    def add_rows(self, rows: list) -> None:
        if rows:
            self.rows.extend(rows)

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _table(self, rows: list) -> pa.Table:
        cols = {
            "step":     pa.array([r[0] for r in rows], pa.int32()),
            "sim_min":  pa.array([r[1] for r in rows], pa.int32()),
            "agent_id": pa.array([r[2] for r in rows], pa.int32()),
        }
        for i, name in enumerate(self.columns):
            cols[name] = pa.array([r[3 + i] for r in rows], pa.float32())
        return pa.table(cols)

    def _next_seg(self) -> int:
        """既存 part 群の最大 index + 1(resume で採番衝突を避ける)。"""
        mx = -1
        if self.out_dir.is_dir():
            for p in self.out_dir.glob(f"{STEM}.part-*.parquet"):
                try:
                    mx = max(mx, int(p.name[len(f"{STEM}.part-"):].split(".")[0]))
                except ValueError:
                    pass
        return mx + 1

    # ---- セグメント書き出し(checkpoint / 定期 flush 時に呼ぶ)----
    def flush_segment(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            pq.write_table(self._table(self.rows),
                           self.out_dir / f"{STEM}.part-{self._seg:04d}.parquet",
                           compression="zstd")
            self._n_flushed += len(self.rows)
            self.rows = []
        self._seg += 1

    # ---- 出力(finalize)----
    def finalize(self) -> Path | None:
        """part 群 + 残りバッファを結合して canonical parquet を出す。

        part が無い(=checkpoint 無効)なら buffer を直接書く(byte 級同一)。分割実行
        (resume で clean finalize したチャンク)のときだけ既存 canonical を先頭に結合する。
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        table = self._table(self.rows) if self.rows else None
        parts = sorted(self.out_dir.glob(f"{STEM}.part-*.parquet"))
        canonical = self.out_dir / f"{STEM}.parquet"
        if not parts:
            if table is None:
                return None
            pq.write_table(table, canonical, compression="zstd")
            return canonical
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
