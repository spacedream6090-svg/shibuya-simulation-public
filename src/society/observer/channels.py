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


# --------------------------------------------------------------------------- #
# G6(第114 GT ロガー): 価値 4 軸の**充足** sat を channels.parquet の追加列にする
#
# 塞ぐ穴: `values.py` の `agent.sat`(4 軸の充足度 0-1)は「その人がいま何を欲しているか」の
# **直接ラベル**でありながら writer が 1 本も無く、ランの成果物のどこにも残らない
# (checkpoint にしか無く、世代を剪定すると消える)。needs / 内部状態は既に channels に
# 全時系列が載っているので、同じ 1 枚に 4 列足すのが最も安い。
#
# ★`cognition/channels.py` の `CHANNELS` には**足さない**。あちらは驚き S の入力の定義で、
#   `spec_sha256()` が σ_c 凍結ファイルを縛っている = 1 本足すと既存の較正が全部無効になる。
#   sat は S の入力ではない(発火式に入らない)ので、**観測側の追加列**として観測層に閉じる。
# ★列は既定 OFF(`cognition.channels.sat_columns`)。OFF では列自体が生えないので、
#   既存ランの channels.parquet とスキーマ・バイトが同一である。
# ★`values.py` は**読むだけ**(sat を 1 バイトも書き換えない)。
_SAT_PREFIX = "sat_"


def sat_columns() -> tuple[str, ...]:
    """sat 追加列の名前(`values.TAGS` の並びが唯一の源。ここに軸名を綴らない)。"""
    from ..values import TAGS
    return tuple(f"{_SAT_PREFIX}{t}" for t in TAGS)


def sat_values(agent) -> list:
    """個体 1 体ぶんの sat 値(`agent.sat` が無い = 機構 OFF なら **null**。0 で埋めない)。"""
    from ..values import TAGS
    sat = getattr(agent, "sat", None)
    if not isinstance(sat, dict):
        return [None] * len(TAGS)
    out = []
    for tag in TAGS:
        got = sat.get(tag)
        try:
            out.append(None if got is None else float(got))
        except (TypeError, ValueError):
            out.append(None)
    return out


def append_sat(sim, rows: list) -> list:
    """`cognition.channels.observe()` の行へ sat 4 列を継ぎ足す(**副作用なし**)。

    `observe()` は `sim.agents` を 1 周して 1 体 1 行を返すので、行と個体は**同じ並び**に
    なっている(行の 3 列目が agent_id という位置は `cognition/channels.KEY_COLUMNS` が固定)。
    その対応を id で 1 件ずつ検算し、崩れていたら辞書を作って引き直す(25 万体で毎 step
    辞書を作らないための最適化であって、正しさは id 検算の側が持つ)。
    """
    if not rows:
        return rows
    agents = list(getattr(sim, "agents", ()) or ())
    nulls = [None] * len(sat_columns())
    by_id = None
    out = []
    for i, row in enumerate(rows):
        aid = int(row[2])
        agent = agents[i] if i < len(agents) else None
        if agent is None or int(agent.id) != aid:
            if by_id is None:
                by_id = {int(a.id): a for a in agents}
            agent = by_id.get(aid)
        out.append((*row, *(sat_values(agent) if agent is not None else nulls)))
    return out


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
