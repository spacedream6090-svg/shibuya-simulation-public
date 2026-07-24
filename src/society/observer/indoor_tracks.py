"""屋内ミクロ軌跡サイドカー(B3 記録層。indoor.tracks.enabled ON のみ・既定 OFF)。

秒スケールの屋内軌跡(sub-step サンプル)と遭遇(接触ペア)を L1 とは別の Parquet サイドカーへ
書き出す。**記録と動力学の分離**(設計原則③): 動力学(scheduler の _phase_indoor)が step 内一時
状態へ積んだサンプル/遭遇を、本層が「読むだけ」で書く(動力学は本層のバッファを読まない)。
tracks.enabled=false でも L1(space_move)は不変=後の統合検収の分離条件。

セグメント化/finalize は observer/logger.py と同一流儀(checkpoint 連携で part 化→finalize で結合、
resume の _resumed フラグで分割実行チャンクの canonical を先頭結合)。これにより resume==straight の
tracks バイト一致を保つ。2 テーブルを別ファイルへ:
  - indoor_tracks_samples.parquet : (agent_id, t_s, building, floor, x, y, zone)
  - indoor_tracks_contacts.parquet: (t_s, id_a, id_b, kind, duration_s, building, floor)
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_SAMPLES = "indoor_tracks_samples"
_CONTACTS = "indoor_tracks_contacts"


class IndoorTracks:
    """屋内軌跡+遭遇の追記バッファ + セグメント/finalize(logger と対の設計)。"""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        # サンプル: (agent_id, t_s, building, floor, x, y, zone)
        self.samples: list[tuple] = []
        # 遭遇: (t_s, id_a, id_b, kind, duration_s, building, floor)
        self.contacts: list[tuple] = []
        self._n_samples_flushed = 0
        self._n_contacts_flushed = 0
        self._seg = self._next_seg()
        # 分割実行(resume)で clean finalize しても前チャンクの canonical を失わない(logger と同流儀)。
        self._resumed = False

    # ---- 記録(動力学が積んだ一時状態を観測側が渡す)----
    def add_samples(self, rows: list) -> None:
        if rows:
            self.samples.extend(rows)

    def add_contacts(self, rows: list) -> None:
        if rows:
            self.contacts.extend(rows)

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _samples_table(self, rows: list) -> pa.Table:
        return pa.table({
            "agent_id": pa.array([r[0] for r in rows], pa.int32()),
            "t_s":      pa.array([r[1] for r in rows], pa.float32()),
            "building": pa.array([r[2] for r in rows], pa.string()),
            "floor":    pa.array([r[3] for r in rows], pa.int32()),
            "x":        pa.array([r[4] for r in rows], pa.float32()),
            "y":        pa.array([r[5] for r in rows], pa.float32()),
            "zone":     pa.array([r[6] for r in rows], pa.int32()),
        })

    def _contacts_table(self, rows: list) -> pa.Table:
        return pa.table({
            "t_s":        pa.array([r[0] for r in rows], pa.float32()),
            "id_a":       pa.array([r[1] for r in rows], pa.int32()),
            "id_b":       pa.array([r[2] for r in rows], pa.int32()),
            "kind":       pa.array([r[3] for r in rows], pa.string()),
            "duration_s": pa.array([r[4] for r in rows], pa.float32()),
            "building":   pa.array([r[5] for r in rows], pa.string()),
            "floor":      pa.array([r[6] for r in rows], pa.int32()),
        })

    def _next_seg(self) -> int:
        """既存 part 群の最大 index + 1(resume で採番衝突を避ける。両 stem の最大を採る)。"""
        mx = -1
        if self.out_dir.is_dir():
            for stem in (_SAMPLES, _CONTACTS):
                for p in self.out_dir.glob(f"{stem}.part-*.parquet"):
                    try:
                        mx = max(mx, int(p.name[len(f"{stem}.part-"):].split(".")[0]))
                    except ValueError:
                        pass
        return mx + 1

    # ---- セグメント書き出し(checkpoint 時に呼ぶ)----
    def flush_segment(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n = self._seg
        if self.samples:
            pq.write_table(self._samples_table(self.samples),
                           self.out_dir / f"{_SAMPLES}.part-{n:04d}.parquet",
                           compression="zstd")
            self._n_samples_flushed += len(self.samples)
            self.samples = []
        if self.contacts:
            pq.write_table(self._contacts_table(self.contacts),
                           self.out_dir / f"{_CONTACTS}.part-{n:04d}.parquet",
                           compression="zstd")
            self._n_contacts_flushed += len(self.contacts)
            self.contacts = []
        self._seg += 1

    # ---- 出力(finalize)----
    def finalize(self) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        s = self._finalize_stream(_SAMPLES,
                                  self._samples_table(self.samples) if self.samples else None)
        if s is not None:
            paths["samples"] = s
        c = self._finalize_stream(_CONTACTS,
                                  self._contacts_table(self.contacts) if self.contacts else None)
        if c is not None:
            paths["contacts"] = c
        return paths

    def _finalize_stream(self, stem: str, table: pa.Table | None) -> Path | None:
        """part 群 + 残りバッファを結合して canonical parquet を出す(logger._finalize_stream と同流儀)。

        part が無い(=checkpoint 無効)なら buffer を直接書く(byte 級同一)。分割実行(resume で clean
        finalize したチャンク)のときだけ既存 canonical を先頭に結合する(_resumed=False の fresh は不変)。"""
        parts = sorted(self.out_dir.glob(f"{stem}.part-*.parquet"))
        canonical = self.out_dir / f"{stem}.parquet"
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
