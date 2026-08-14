"""G4 記憶ストリーム サイドカー ``memory.parquet``(``observer.memory_daily``・**既定 OFF**)。

何を解く問題か(現物の穴)
--------------------------
記憶ストリーム(``agents/memory.py`` の ``MemoryStore.episodes`` / ``buffer``)の**本文**は、
ランの成果物のどこにも残らない:

  - L1 は「何件になったか」しか出さない(本文は payload に無い)
  - checkpoint(半日ごとの完全状態)には本文が入るが、**世代を剪定した瞬間に消える**
  - 記憶そのものが日々消える: ``consolidate`` は就寝時に ``buffer`` を直近 5 件へ切り、
    ``store_cap`` 超過分は重要度の低い順に捨て、ACT-R 忘却(既定 OFF)は
    活性化の低い episodes を確率的に間引く

つまり「その人が何を覚えていたか」は**その日のうちにしか存在しない**。復元実験
(観測痕跡 → 内部状態)の**正解ラベル**としては最重要級なのに、事後には二度と取れない。

何をするか
----------
日境界に、在場している個体の ``mem.episodes`` と ``mem.buffer`` を 1 行 1 記憶で
そのまま書き出す(**その日の朝のスナップショット**)。差分ではなく毎日の全量なので、
「いつ何を忘れたか」が行の消失として読める(差分方式だと忘却が観測できない)。

  ★``agents/memory.py`` は**読むだけ**。本 module は世界状態を 1 バイトも書き換えず、
    L1 を 1 件も出さず、乱数を 1 粒も引かず、LLM 呼数を 1 も変えない
    (動力学は本 module のバッファを 1 度も読まない = ``observer/roster.py`` と同じ規律)。

規律(R1)
---------
- **既定 OFF**。``observer.memory_daily.enabled=false`` では本 module のオブジェクトを
  作らず、1 ファイルも書かない(= 既存ラン・golden 無風)。
- 分割実行(checkpoint / resume)は logger・他サイドカーと同一流儀(part 化 → finalize で
  結合)。``_day``(最後に撮った日)は resume 時に **前チャンクが実行し終えた step の日**
  へ据える(``Simulation.run`` が ``clock.sim_min(start-1)`` から渡す)ので、
  resume したランの行集合は一気通しのランと一致する(= resume==straight)。
  ★ここを「既存 part の最大 day から復元」にすると、**行が 1 つも無い日**(day0 の朝は
    まだ誰も何も覚えていない)を撮り直してしまい、一気通しと食い違う。

正直な限界
----------
撮るのは**日境界だけ**なので、最終日の**終わり**の状態は本サイドカーに入らない
(10 日ランなら day0〜day9 の 10 枚 = それぞれ「前日の終わり」)。終端の完全状態は
**最後の checkpoint** が持つ —— これが G2「checkpoint 世代の剪定禁止」の根拠の 1 つである。
``finalize`` で撮り足す案は採らない: 分割実行では各チャンクの finalize が**日の途中**で
走るので、一気通しのランと行集合が食い違う(resume==straight が壊れる)。

規模(正直な見積り)
--------------------
finals 構成(在場 25 万 × 10 日)で 1 個体あたり平均 20 件前後(``store_cap=120`` /
``buffer_cap=30`` が上限)とすると **約 5,000 万行**、zstd 圧縮後で **1.5-1.8 GB**。
これは本サイドカー群のなかで最大の 1 枚である(ディスク計画に必ず数えること)。
走行中のコストは日境界に在場者の記憶を 1 周する走査だけで、毎 step の走査はゼロ。

スキーマ(1 行 = 1 記憶)
-------------------------
``day`` / ``agent_id`` / ``src``(``"ep"`` = 統合済み顕著記憶 / ``"buf"`` = 未統合の緩衝)/
``idx``(その列の中での位置。0 が最古)/ ``step``(その記憶が生まれた step)/
``kind`` / ``importance`` / ``text``
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import FinalizeStreamMixin

_STEM = "memory"

#: サイドカーのスキーマ版(列を足したら上げる)。
SCHEMA = 1

#: ``src`` 列の値(統合済み / 未統合)。解析側が文字列を書き写さずに済むよう定数で出す。
SRC_EPISODE = "ep"
SRC_BUFFER = "buf"


def build_cfg(raw) -> dict:
    """conf の ``observer.memory_daily`` を正準化(既定 OFF)。

    ★``isinstance(raw, dict)`` で判定しないこと: conf は OmegaConf の ``DictConfig`` で、
      dict の部分型ではない(素の dict しか受けない書き方だと **ON にしても OFF に落ちる**)。
    """
    cfg = {"enabled": False, "text_chars": 0}
    if raw is not None and hasattr(raw, "get"):
        cfg["enabled"] = bool(raw.get("enabled", False))
        # 本文の切り詰め(0 = 切らない = 既定)。ディスクが逼迫したときの唯一のレバー。
        cfg["text_chars"] = max(0, int(raw.get("text_chars", 0) or 0))
    return cfg


def cfg_of_config(config) -> dict:
    """``sim.cfg`` から本層の設定を引く(未宣言の旧 config でも既定 OFF に落ちる)。"""
    try:
        obs = config.get("observer", None) or {}
        raw = obs.get("memory_daily", None)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        raw = None
    return build_cfg(raw)


class MemoryDaily(FinalizeStreamMixin):
    """記憶ストリームの日次スナップショット(``observer/roster.py`` と対の設計)。"""

    def __init__(self, out_dir: Path, text_chars: int = 0):
        self.out_dir = Path(out_dir)
        self.text_chars = max(0, int(text_chars))
        self.rows: list[tuple] = []
        self._day: int | None = None
        self._n_flushed = 0
        self._seg = self._next_seg()
        self._resumed = False

    # ---- 記録(観測側が日境界に在場者を 1 周する。動力学は本バッファを読まない)----
    def _text(self, raw) -> str:
        text = str(raw or "")
        return text[:self.text_chars] if self.text_chars else text

    def capture(self, sim, day: int) -> int:
        """在場個体の記憶を 1 記憶 1 行で追記する。戻り値 = 追記した行数。"""
        n = 0
        day = int(day)
        for agent in getattr(sim, "agents", ()) or ():
            mem = getattr(agent, "mem", None)
            if mem is None:
                continue
            aid = int(agent.id)
            for src, seq in ((SRC_EPISODE, getattr(mem, "episodes", None) or ()),
                             (SRC_BUFFER, getattr(mem, "buffer", None) or ())):
                for idx, ep in enumerate(seq):
                    self.rows.append((
                        day, aid, src, int(idx),
                        int(getattr(ep, "step", 0) or 0),
                        str(getattr(ep, "kind", "") or ""),
                        round(float(getattr(ep, "importance", 0.0) or 0.0), 3),
                        self._text(getattr(ep, "text", "")),
                    ))
                    n += 1
        return n

    def on_step(self, sim, step: int, sim_min: int) -> int:
        """日境界(と初回)にだけ :meth:`capture` を通す。それ以外の step は即 0。"""
        day = int(sim_min) // 1440
        if self._day is not None and day == self._day:
            return 0
        self._day = day
        return self.capture(sim, day)

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _table(self, rows: list) -> pa.Table:
        def col(i, typ):
            return pa.array([r[i] for r in rows], typ)
        return pa.table({
            "day":        col(0, pa.int32()),
            "agent_id":   col(1, pa.int32()),
            "src":        col(2, pa.string()),
            "idx":        col(3, pa.int32()),
            "step":       col(4, pa.int32()),
            "kind":       col(5, pa.string()),
            "importance": col(6, pa.float32()),
            "text":       col(7, pa.string()),
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
        return self._finalize_stream(_STEM,
                                     self._table(self.rows) if self.rows else None)
