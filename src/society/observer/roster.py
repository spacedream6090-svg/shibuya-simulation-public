"""A13 日次入場者名簿サイドカー ``roster.parquet``(``observer.roster_daily``・**既定 OFF**)。

何を解く問題か(現物の穴)
--------------------------
本リポジトリが持つ「名簿」は 2 枚しかなく、どちらもプール回転の途中入場者を写さない:

  - ``traits.json`` … ``Simulation.__init__`` が書く = **day0 の在場者だけ**
  - ``agents.json`` … 同じく __init__ で書き、会社観測 ON のランだけ finalize が
    再出力する = どちらの時点でも **その瞬間の在場者だけ**

finals 構成(名簿 100 万・在場 cap 25 万・10 日)では day1 以降に **20.9 万人**が入場する。
その人たちは L1/L2 には現れるのに、**素性(年齢・職業・住居・職場・traits)がどこにも
残らない**。事後解析(``observer/measure.py`` の R² 回帰・``deviation`` のペルソナ逸脱・
復元実験)は名簿を ``traits.json > agents.json`` の順に引くので、途中入場者は
**分母から丸ごと落ちていた**(= 観測されているのに素性が不明な 20.9 万人)。

何をするか
----------
日境界に「**まだ一度も名簿に載せていない在場者**」を 1 行ずつ追記するだけ。1 個体は
生涯 1 行(入場した日 = ``day`` 列)で、退場して再来街しても 2 行目は作らない
(名簿の保全が目的であって、在場の履歴は L1/L3 が既に持っている)。

  ★``traits.json`` を finalize で書き直す案は採らない: それだと「day0 の名簿」という
    既存成果物の意味が変わり(復元実験・既存解析の入力が別物になる)、既存ランと
    比較できなくなる。**新しいファイルを 1 枚足す**ほうが観測の来歴を壊さない。

規律(R1)
---------
- **既定 OFF**。``observer.roster_daily.enabled=false`` では本 module のオブジェクトを
  作らず、1 ファイルも書かず、``sim`` の属性も 1 つも生えない(= 既存ラン・golden 無風)。
- **書き手は観測側だけ**。動力学(scheduler / simulation)は本 module のバッファを
  1 度も読まない。読むのは ``sim.agents`` の**既存の属性**だけで、世界状態を 1 バイトも
  書き換えない・L1 を 1 件も出さない・乱数を 1 粒も引かない・LLM 呼数を 1 も変えない。
- ``traits`` の**個々の名前をコードに書かない**(no-fingerprint 契約)。1 本の JSON 文字列
  列 ``traits_json`` に丸ごと入れるので、trait 名が増減しても列は動かない。
- 分割実行(checkpoint / resume)は logger・他サイドカーと同一流儀
  (part 化 → finalize で結合)。``_seen`` は**既存の part / canonical から復元する**ので、
  resume したランの行集合は一気通しのランと一致する(= resume==straight)。

規模(正直な見積り)
--------------------
finals 構成(在場 25 万 × 10 日・途中入場 20.9 万)で行数は **約 46 万行**。
1 行は固定列 ≈ 150 B + ``traits_json`` ≈ 150-250 B なので、zstd 圧縮後で **+100 MB 前後**。
走行中のコストは日境界に在場者を 1 周する走査(25 万 × 10 日 = 250 万回の集合検査)だけで、
毎 step の走査はゼロ。

スキーマ(1 行 = 1 個体・追記のみ)
-----------------------------------
``day``(初めて名簿に載せた日)/ ``agent_id`` / ``pool_pid`` / ``name`` / ``age`` /
``gender`` / ``occupation`` / ``visitor`` / ``commute`` / ``arrival_min`` /
``residence_line`` / ``has_bicycle`` / ``has_car`` / ``home_node`` / ``home_building`` /
``home_floor`` / ``work_name`` / ``work_node`` / ``work_building`` / ``org_id`` /
``org_role`` / ``household_id`` / ``bedtime_min`` / ``money`` / ``wage`` / ``part_time`` /
``part_time_name`` / ``drive_threshold`` / ``fire_weight`` / ``traits_json``
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import FinalizeStreamMixin

_STEM = "roster"

#: サイドカーのスキーマ版(列を足したら上げる)。
SCHEMA = 1


def build_cfg(raw) -> dict:
    """conf の ``observer.roster_daily`` を正準化(既定 OFF)。

    ★``isinstance(raw, dict)`` で判定しないこと: conf は OmegaConf の ``DictConfig`` で、
      dict の部分型ではない(素の dict しか受けない書き方だと **ON にしても OFF に落ちる**)。
    """
    cfg = {"enabled": False}
    if raw is not None and hasattr(raw, "get"):
        cfg["enabled"] = bool(raw.get("enabled", False))
    return cfg


def cfg_of_config(config) -> dict:
    """``sim.cfg`` から本層の設定を引く(未宣言の旧 config でも既定 OFF に落ちる)。"""
    try:
        obs = config.get("observer", None) or {}
        raw = obs.get("roster_daily", None)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        raw = None
    return build_cfg(raw)


def _row(agent, day: int) -> tuple:
    """在場個体 1 体 → 1 行(**読むだけ**。属性が無い個体でも既定値へ落ちる)。"""
    pt = getattr(agent, "part_time", None)
    return (
        int(day),
        int(agent.id),
        str(getattr(agent, "pool_pid", "") or ""),
        str(getattr(agent, "name", "") or ""),
        int(getattr(agent, "age", 0) or 0),
        str(getattr(agent, "gender", "") or ""),
        str(getattr(agent, "occupation", "") or ""),
        bool(getattr(agent, "visitor", False)),
        bool(getattr(agent, "commute", False)),
        int(getattr(agent, "arrival_min", -1)),
        str(getattr(agent, "residence_line", "") or ""),
        bool(getattr(agent, "has_bicycle", False)),
        bool(getattr(agent, "has_car", False)),
        str(getattr(agent, "home_node", "") or ""),
        str(getattr(agent, "home_building", "") or ""),
        int(getattr(agent, "home_floor", 0) or 0),
        str(getattr(agent, "work_name", "") or ""),
        str(getattr(agent, "work_node", "") or ""),
        str(getattr(agent, "work_building", "") or ""),
        str(getattr(agent, "org_id", "") or ""),
        str(getattr(agent, "org_role", "") or ""),
        str(getattr(agent, "household_id", "") or ""),
        int(getattr(agent, "bedtime_min", 0) or 0),
        round(float(getattr(agent, "money", 0.0) or 0.0), 1),
        round(float(getattr(agent, "wage", 0.0) or 0.0), 1),
        bool(pt),
        str((pt or {}).get("name", "") or ""),
        round(float(getattr(agent, "drive_threshold", 0.0) or 0.0), 4),
        round(float(getattr(agent, "fire_weight", 0.0) or 0.0), 4),
        # ★trait 名を 1 つもコードに書かない(no-fingerprint 契約)。
        json.dumps({str(k): float(v) for k, v in
                    sorted((getattr(agent, "traits", None) or {}).items())},
                   ensure_ascii=False, sort_keys=True),
    )


class RosterDaily(FinalizeStreamMixin):
    """日次入場者名簿の追記バッファ + セグメント/finalize(``org_ledger.OrgLedger`` と対の設計)。"""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.rows: list[tuple] = []
        self._seen: set[int] = set()
        self._day: int | None = None
        self._n_flushed = 0
        self._seg = self._next_seg()
        self._resumed = False
        self._reloaded = False

    # ---- 記録(観測側が日境界に在場者を 1 周する。動力学は本バッファを読まない)----
    def capture(self, sim, day: int) -> int:
        """まだ名簿に載せていない在場者を追記する。戻り値 = 追記した行数(冪等)。"""
        if self._resumed and not self._reloaded:   # 分割実行のときだけ既出を読み戻す
            self._reloaded = True
            self._reload_seen()
        n = 0
        for agent in getattr(sim, "agents", ()) or ():
            aid = int(agent.id)
            if aid in self._seen:
                continue
            self._seen.add(aid)
            self.rows.append(_row(agent, int(day)))
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
            "day":             col(0, pa.int32()),
            "agent_id":        col(1, pa.int32()),
            "pool_pid":        col(2, pa.string()),
            "name":            col(3, pa.string()),
            "age":             col(4, pa.int32()),
            "gender":          col(5, pa.string()),
            "occupation":      col(6, pa.string()),
            "visitor":         col(7, pa.bool_()),
            "commute":         col(8, pa.bool_()),
            "arrival_min":     col(9, pa.int32()),
            "residence_line":  col(10, pa.string()),
            "has_bicycle":     col(11, pa.bool_()),
            "has_car":         col(12, pa.bool_()),
            "home_node":       col(13, pa.string()),
            "home_building":   col(14, pa.string()),
            "home_floor":      col(15, pa.int32()),
            "work_name":       col(16, pa.string()),
            "work_node":       col(17, pa.string()),
            "work_building":   col(18, pa.string()),
            "org_id":          col(19, pa.string()),
            "org_role":        col(20, pa.string()),
            "household_id":    col(21, pa.string()),
            "bedtime_min":     col(22, pa.int32()),
            "money":           col(23, pa.float64()),
            "wage":            col(24, pa.float64()),
            "part_time":       col(25, pa.bool_()),
            "part_time_name":  col(26, pa.string()),
            "drive_threshold": col(27, pa.float64()),
            "fire_weight":     col(28, pa.float64()),
            "traits_json":     col(29, pa.string()),
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

    def _reload_seen(self) -> None:
        """既存の part / canonical から ``agent_id`` を読み戻す(**resume==straight の要**)。

        ``_seen`` は「もう名簿に載せた個体」の集合で、checkpoint には載っていない。
        復元しないと resume 直後の日境界で在場者全員が 2 行目として追記され、
        一気通しのランと行集合が食い違う。列 1 本だけを読むので費用は小さい。

        ★呼ぶのは **``_resumed`` が立っているときの初回 capture だけ**(``__init__`` では
          呼ばない): 同一プロセス内では ``_seen`` はメモリに在るので読み戻す必要が無く、
          「前のランの残骸が入った out_dir へ fresh に書く」場合に旧ファイルを既出扱いして
          1 行も書かなくなる事故を構造的に避けられる。
        """
        if not self.out_dir.is_dir():
            return
        paths = sorted(self.out_dir.glob(f"{_STEM}.part-*.parquet"))
        canonical = self.out_dir / f"{_STEM}.parquet"
        if canonical.exists():
            paths.append(canonical)
        for p in paths:
            try:
                tbl = pq.read_table(p, columns=["agent_id"])
            except Exception:                      # noqa: BLE001(壊れた part は黙って諦める)
                continue
            self._seen.update(int(v) for v in tbl.column("agent_id").to_pylist())

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
