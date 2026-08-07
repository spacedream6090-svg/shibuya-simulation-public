"""屋内ミクロ軌跡サイドカー(B3 記録層。indoor.tracks.enabled ON のみ・既定 OFF)。

秒スケールの屋内軌跡(sub-step サンプル)と遭遇(接触ペア)を L1 とは別の Parquet サイドカーへ
書き出す。**記録と動力学の分離**(設計原則③): 動力学(scheduler の _phase_indoor)が step 内一時
状態へ積んだサンプル/遭遇を、本層が「読むだけ」で書く(動力学は本層のバッファを読まない)。
tracks.enabled=false でも L1(space_move)は不変=後の統合検収の分離条件。

セグメント化/finalize は observer/logger.py と同一流儀(checkpoint 連携で part 化→finalize で結合、
resume の _resumed フラグで分割実行チャンクの canonical を先頭結合)。これにより resume==straight の
tracks バイト一致を保つ。★W4-E(第99バッチ)で結合の実装は `observer/finalize.py` の
`FinalizeStreamMixin` **1 本**へ括り出した(同型 finalize の二重実装をやめた)。これにより
conf `observer.finalize.streaming`(既定 false)が **L1 と同じ 1 つの判断で**本サイドカーにも効く
= ON なら part を row-group 単位で逐次書きしてピークメモリを有界化する。既定 OFF は従来経路のまま
= 1 バイトも変わらない。2 テーブルを別ファイルへ:
  - indoor_tracks_samples.parquet : (agent_id, t_s, building, floor, x, y, zone)
  - indoor_tracks_contacts.parquet: (t_s, id_a, id_b, kind, duration_s, building, floor)
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import FinalizeStreamMixin

_SAMPLES = "indoor_tracks_samples"
_CONTACTS = "indoor_tracks_contacts"


class IndoorTracks(FinalizeStreamMixin):
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

    # `_finalize_stream`(既定の concat 経路 / streaming 経路)は FinalizeStreamMixin が
    # 唯一の実装。ここに自前の複製は置かない(W4-E)。
