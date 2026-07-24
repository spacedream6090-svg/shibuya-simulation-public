"""会社観測データ層 B4: 組織台帳の日次系列サイドカー(work.service.ledger.enabled ON のみ・既定 OFF)。

会社(org_id)ごとの日次会計/活動を L1 とは別の Parquet サイドカーへ書き出す。**記録と動力学の分離**
(indoor_tracks.py と同一の設計原則③): 動力学(scheduler)が per-day アキュムレータ(sim._org_day)へ
積んだ集計を、本層が日次境界で「読むだけ」で 1 行/社に整える(動力学は本層のバッファを読まない)。
ledger.enabled=false では本サイドカーは生成されず、L1・org_output は不変=バイト一致。

セグメント化/finalize は observer/logger.py・indoor_tracks.py と同一流儀(checkpoint 連携で part 化→
finalize で結合、resume の _resumed フラグで分割実行チャンクの canonical を先頭結合)。これにより
resume==straight の org_ledger バイト一致を保つ。

スキーマ(会社 UI タスク B7 がこの契約で読む=厳守):
  org_ledger.parquet: (day, org_id, production, revenue_est, wage_paid, serve_count, attendance_min)
  - day(int32)          : 日インデックス(sim_min // 1440。0 起点)。
  - org_id(str)         : 組織台帳の id(str)。同居複数社は別行。
  - production(int32)   : その日の勤務完遂→産出(production イベント)件数。
  - revenue_est(float)  : 売上の代理指標(日給×revenue_margin の合計。economy ON 時のみ非0)。
  - wage_paid(float)    : 支給賃金の合計(economy ON 時のみ非0)。
  - serve_count(int32)  : その社スタッフに帰属した接客(serve)件数(indoor_fields の org_id 解決に一致)。
  - attendance_min(int32): ミクロ在席分(office 系従業者が職務区画に居た step 数×10分。indoor ON 時のみ非0)。
値が無い列は 0。日次境界で活動があった社のみ 1 行(全列 0 の社は書かない)。
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_STEM = "org_ledger"


class OrgLedger:
    """組織日次系列の追記バッファ + セグメント/finalize(indoor_tracks.IndoorTracks と対の設計)。"""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        # 行: (day, org_id, production, revenue_est, wage_paid, serve_count, attendance_min)
        self.rows: list[tuple] = []
        self._n_flushed = 0
        self._seg = self._next_seg()
        # 分割実行(resume)で clean finalize しても前チャンクの canonical を失わない(logger と同流儀)。
        self._resumed = False

    # ---- 記録(動力学が日次境界で集計した 1 社 1 行を観測側が渡す)----
    def add_rows(self, rows: list) -> None:
        if rows:
            self.rows.extend(rows)

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _table(self, rows: list) -> pa.Table:
        return pa.table({
            "day":            pa.array([r[0] for r in rows], pa.int32()),
            "org_id":         pa.array([r[1] for r in rows], pa.string()),
            "production":     pa.array([r[2] for r in rows], pa.int32()),
            "revenue_est":    pa.array([r[3] for r in rows], pa.float64()),
            "wage_paid":      pa.array([r[4] for r in rows], pa.float64()),
            "serve_count":    pa.array([r[5] for r in rows], pa.int32()),
            "attendance_min": pa.array([r[6] for r in rows], pa.int32()),
        })

    def _next_seg(self) -> int:
        """既存 part 群の最大 index + 1(resume で採番衝突を避ける)。"""
        mx = -1
        if self.out_dir.is_dir():
            for p in self.out_dir.glob(f"{_STEM}.part-*.parquet"):
                try:
                    mx = max(mx, int(p.name[len(f"{_STEM}.part-"):].split(".")[0]))
                except ValueError:
                    pass
        return mx + 1

    # ---- セグメント書き出し(checkpoint 時に呼ぶ)----
    def flush_segment(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            pq.write_table(self._table(self.rows),
                           self.out_dir / f"{_STEM}.part-{self._seg:04d}.parquet",
                           compression="zstd")
            self._n_flushed += len(self.rows)
            self.rows = []
        self._seg += 1

    # ---- 出力(finalize)----
    def finalize(self) -> Path | None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._finalize_stream(self._table(self.rows) if self.rows else None)

    def _finalize_stream(self, table: pa.Table | None) -> Path | None:
        """part 群 + 残りバッファを結合して canonical parquet を出す(logger._finalize_stream と同流儀)。

        part が無い(=checkpoint 無効)なら buffer を直接書く(byte 級同一)。分割実行(resume で clean
        finalize したチャンク)のときだけ既存 canonical を先頭に結合する(_resumed=False の fresh は不変)。"""
        parts = sorted(self.out_dir.glob(f"{_STEM}.part-*.parquet"))
        canonical = self.out_dir / f"{_STEM}.parquet"
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
