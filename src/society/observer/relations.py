"""G5 関係台帳の日次差分 サイドカー ``relations.parquet``(``observer.relations_daily``・**既定 OFF**)。

何を解く問題か(現物の穴)
--------------------------
関係の**連続値**(``closeness``)の軌跡は、ランの成果物のどこにも残らない:

  - L1 に出るのは **tier を跨いだ瞬間**(``relation_tier`` / ``relation_break`` /
    ``relation_dormant`` / ``relation_rekindle``)だけで、段の内側の連続的な増減は無音
  - checkpoint(半日ごと)には台帳がまるごと入るが、**世代を剪定した瞬間に消える**
  - プール回転で退場した個体の台帳は dormant サイドカーへ移り、LRU 溢れで最古から捨てられる

「共在(すれ違い)→ 関係」の復元実験は、まさにこの連続値を**正解ラベル**として要求する。
すれ違いの側(C3 の日次カウンタ ``_c3_pass`` / ``_c3_greet``)も L1 には出ないので、
**説明変数と被説明変数が両方とも残らない**という状態だった。

何をするか
----------
日境界に 2 種類の行を追記する。

  1. **関係行**(``other_id >= 0``): ``mem.relations`` のうち **前日から値が変わった対だけ**
     (差分方式)。初出の対は必ず 1 行出る。値が動かない対は 1 行も出ない
     = 25 万人 × 数十件 × 10 日を全量で書くと数十 GB になるのを避ける唯一の手段。
  2. **すれ違い行**(``other_id = -1``): その個体の C3 日次カウンタ
     (``c3_pass`` = 今日すれ違った人数 / ``c3_greet`` = 今日会話した人数)。
     **どちらも 0 の個体は行を作らない**(欠行 = 0 と読む)。

  ★``day`` は「その日の**始まり**に見た値」= 前日の終わりの状態である。本サイドカーは
    run_step の先頭(在場ローテーションより前)で撮るので、C3 カウンタが
    ``conversation._roll_day`` に空へ戻される前の**前日ぶん**を掴む。退場する個体の
    最後の関係も、退場処理の前なのでここに残る。

  ★``agents/memory.py`` / ``conversation.py`` は**読むだけ**。世界状態を 1 バイトも
    書き換えず、L1 を 1 件も出さず、乱数を 1 粒も引かず、LLM 呼数を 1 も変えない。

規律(R1)
---------
- **既定 OFF**。``observer.relations_daily.enabled=false`` では本 module のオブジェクトを
  作らず、1 ファイルも書かない(= 既存ラン・golden 無風)。
- 分割実行(checkpoint / resume): ``_day``(最後に撮った日)は ``Simulation.run`` が
  ``clock.sim_min(start-1)`` から渡し、``_last``(前回出した値)は**既存の part /
  canonical から読み戻す**。両方そろって初めて resume==straight が成立する
  (``_last`` を失うと resume 直後に全対が「初出」として再出力される)。

正直な限界
----------
- 撮るのは**日境界だけ**なので、最終日の**終わり**の関係は本サイドカーに入らない。
  終端の完全な台帳は**最後の checkpoint**(と dormant サイドカー)が持つ ——
  これが G2「checkpoint 世代の剪定禁止」の根拠の 1 つである。``finalize`` で撮り足す案は
  採らない: 分割実行では各チャンクの finalize が**日の途中**で走るので、一気通しのランと
  行集合が食い違う(resume==straight が壊れる)。
- 差分方式なので「**関係が台帳から消えたこと**」(``relations_max`` の LRU 退避 /
  プール退避時の上位 ``rel_cap`` 件切り)は行として現れない。消滅の検出は
  「その後 1 度も行が出ない」という**不在**でしか読めない(全量方式なら読めるが 1 桁重い)。

規模(正直な見積り)
--------------------
finals 構成(在場 25 万 × 10 日)で、1 人 1 日あたり動く対は数件のオーダー。差分方式で
**+0.4 GB 前後**(全量方式だと 1 桁大きい)。走行中のコストは日境界に在場者の台帳を
1 周する走査だけで、毎 step の走査はゼロ。

スキーマ(1 行 = 1 対の変化 または 1 個体のすれ違い集計)
---------------------------------------------------------
``day`` / ``agent_id`` / ``other_id``(-1 = すれ違い行)/ ``closeness`` / ``tier`` /
``count`` / ``dormant`` / ``c3_pass`` / ``c3_greet``

  関係行は ``c3_*`` が null、すれ違い行は ``closeness`` / ``tier`` / ``count`` /
  ``dormant`` が null(**欠測を 0 で埋めない**という本リポジトリの既定則)。
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import FinalizeStreamMixin

_STEM = "relations"

#: サイドカーのスキーマ版(列を足したら上げる)。
SCHEMA = 1

#: すれ違い行(個体そのものの日次集計)を表す ``other_id``。
SELF_ID = -1


def build_cfg(raw) -> dict:
    """conf の ``observer.relations_daily`` を正準化(既定 OFF)。

    ★``isinstance(raw, dict)`` で判定しないこと: conf は OmegaConf の ``DictConfig`` で、
      dict の部分型ではない(素の dict しか受けない書き方だと **ON にしても OFF に落ちる**)。
    """
    cfg = {"enabled": False, "passing": True}
    if raw is not None and hasattr(raw, "get"):
        cfg["enabled"] = bool(raw.get("enabled", False))
        # C3 すれ違い行を出すか(既定 ON。関係行だけ欲しいときに切れる)。
        cfg["passing"] = bool(raw.get("passing", True))
    return cfg


def cfg_of_config(config) -> dict:
    """``sim.cfg`` から本層の設定を引く(未宣言の旧 config でも既定 OFF に落ちる)。"""
    try:
        obs = config.get("observer", None) or {}
        raw = obs.get("relations_daily", None)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        raw = None
    return build_cfg(raw)


def _value(rel: dict) -> tuple:
    """台帳エントリ → 差分比較に使う 4 つ組(欠測は None のまま = 0 で埋めない)。"""
    clo = rel.get("closeness")
    tier = rel.get("tier")
    return (None if clo is None else round(float(clo), 4),
            None if tier is None else int(tier),
            int(rel.get("count", 0) or 0),
            bool(rel.get("dormant", False)))


class RelationsDaily(FinalizeStreamMixin):
    """関係台帳の日次差分 + C3 すれ違い集計(``observer/roster.py`` と対の設計)。"""

    def __init__(self, out_dir: Path, passing: bool = True):
        self.out_dir = Path(out_dir)
        self.passing = bool(passing)
        self.rows: list[tuple] = []
        self._last: dict[tuple[int, int], tuple] = {}
        self._day: int | None = None
        self._n_flushed = 0
        self._seg = self._next_seg()
        self._resumed = False
        self._reloaded = False

    # ---- 記録(観測側が日境界に在場者を 1 周する。動力学は本バッファを読まない)----
    def capture(self, sim, day: int) -> int:
        """変化した関係 + C3 集計を追記する。戻り値 = 追記した行数。"""
        if self._resumed and not self._reloaded:   # 分割実行のときだけ既出値を読み戻す
            self._reloaded = True
            self._reload_last()
        n = 0
        day = int(day)
        for agent in getattr(sim, "agents", ()) or ():
            aid = int(agent.id)
            mem = getattr(agent, "mem", None)
            rels = (getattr(mem, "relations", None) or {}) if mem is not None else {}
            for oid in sorted(rels):               # 相手 id 昇順 = 決定論
                rel = rels[oid]
                if not isinstance(rel, dict):
                    continue
                val = _value(rel)
                key = (aid, int(oid))
                if self._last.get(key) == val:
                    continue                       # 前日から 1 つも動いていない = 行を作らない
                self._last[key] = val
                self.rows.append((day, aid, int(oid), val[0], val[1], val[2], val[3],
                                  None, None))
                n += 1
            if not self.passing:
                continue
            npass = len(getattr(agent, "_c3_pass", None) or ())
            ngreet = len(getattr(agent, "_c3_greet", None) or ())
            if npass or ngreet:                    # どちらも 0 の個体は行を作らない
                self.rows.append((day, aid, SELF_ID, None, None, None, None,
                                  int(npass), int(ngreet)))
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
            "day":       col(0, pa.int32()),
            "agent_id":  col(1, pa.int32()),
            "other_id":  col(2, pa.int32()),
            # ★float64: ``_last`` を parquet から読み戻して差分比較するので、書いた値と
            #   読んだ値が**ビット一致**しないと resume 直後に全対が「初出」になる
            #   (float32 だと round(x, 4) の丸め位置が往復で動きうる)。
            "closeness": col(3, pa.float64()),
            "tier":      col(4, pa.int32()),
            "count":     col(5, pa.int32()),
            "dormant":   col(6, pa.bool_()),
            "c3_pass":   col(7, pa.int32()),
            "c3_greet":  col(8, pa.int32()),
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

    def _reload_last(self) -> None:
        """既存の part / canonical から「最後に出した値」を読み戻す(**resume==straight の要**)。

        差分方式なので ``_last`` を失うと、resume 直後の日境界で在場者の全対が「初出」として
        もう一度書かれ、一気通しのランと行集合が食い違う。行は日昇順で追記されるので、
        canonical(前チャンク)→ part(このチャンク)の順に読んで**後勝ち**で畳めばよい。

        ★呼ぶのは ``_resumed`` が立っているときの初回 capture だけ(``__init__`` では
          呼ばない): 「前のランの残骸が入った out_dir へ fresh に書く」場合に旧ファイルを
          既出扱いして 1 行も書かなくなる事故を構造的に避ける(roster.py と同じ作法)。
        """
        if not self.out_dir.is_dir():
            return
        paths: list[Path] = []
        canonical = self.out_dir / f"{_STEM}.parquet"
        if canonical.exists():
            paths.append(canonical)
        paths += sorted(self.out_dir.glob(f"{_STEM}.part-*.parquet"))
        cols = ["agent_id", "other_id", "closeness", "tier", "count", "dormant"]
        for p in paths:
            try:
                tbl = pq.read_table(p, columns=cols)
            except Exception:                      # noqa: BLE001(壊れた part は黙って諦める)
                continue
            for row in tbl.to_pylist():
                oid = row["other_id"]
                if oid is None or int(oid) == SELF_ID:
                    continue                       # すれ違い行は差分の対象ではない
                self._last[(int(row["agent_id"]), int(oid))] = (
                    None if row["closeness"] is None else round(float(row["closeness"]), 4),
                    None if row["tier"] is None else int(row["tier"]),
                    int(row["count"] or 0),
                    bool(row["dormant"]),
                )

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
