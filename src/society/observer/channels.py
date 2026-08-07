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
= **途中再開でサイドカーが二重記録しない**。★W4-E(第99バッチ)で結合の実装は
`observer/finalize.py` の `FinalizeStreamMixin` **1 本**へ括り出し、conf
`observer.finalize.streaming`(既定 false)が L1 と同じ 1 つの判断で本サイドカー
(`channels` / 派生の `cognition_g`)にも効くようにした。既定 OFF は従来経路のまま。

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

from .finalize import FinalizeStreamMixin

STEM = "channels"


class ChannelsSidecar(FinalizeStreamMixin):
    """観測チャンネル行の追記バッファ + セグメント/finalize(IndoorTracks と対の設計)。

    ★ファイル名の幹はクラス属性 `STEM`(第82バッチで g/θ 軌跡サイドカーが同じ機構を
      そのまま使うため)。列は int32 キー 3 本 + float32 値列という同じ形をしている。
    """

    STEM = STEM

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
            for p in self.out_dir.glob(f"{self.STEM}.part-*.parquet"):
                try:
                    mx = max(mx, int(p.name[len(f"{self.STEM}.part-"):].split(".")[0]))
                except ValueError:
                    pass
        return mx + 1

    # ---- セグメント書き出し(checkpoint / 定期 flush 時に呼ぶ)----
    def flush_segment(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            pq.write_table(self._table(self.rows),
                           self.out_dir / f"{self.STEM}.part-{self._seg:04d}.parquet",
                           compression="zstd")
            self._n_flushed += len(self.rows)
            self.rows = []
        self._seg += 1

    # ---- 出力(finalize)----
    # 結合(既定の concat 経路 / streaming 経路)は FinalizeStreamMixin が唯一の実装。
    # ここに自前の複製は置かない(W4-E)。stem はクラス属性 STEM なので派生
    # (CognitionGSidecar)もそのまま同じ経路を通る。
    def finalize(self) -> Path | None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._finalize_stream(self.STEM,
                                     self._table(self.rows) if self.rows else None)


class CognitionGSidecar(ChannelsSidecar):
    """感度 g / 閾値倍率 θ の**全軌跡**サイドカー(第82バッチ・g_update ON のみ)。

    設計 §2.7「`g_i(0)` と g の全軌跡をログする」/ §8「`g` ベクトルの時間発展 =
    注意の向き先の軌跡。**分散の拡大 = 役割分化の一次証拠**」。
    `scripts/analyze_g.py` がこの parquet だけを読んで g(0) 分散 vs Δg の分散分解を出す。

    ChannelsSidecar と同じ形(int32 キー 3 本 + float32 値列)なので実装は STEM だけ差す。
    観測層なので**動力学はこのバッファを読まない**(記録と動力学の分離)。
    """

    STEM = "cognition_g"
