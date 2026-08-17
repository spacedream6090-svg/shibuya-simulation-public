"""レーンG 案B = 「意図の収束」の観測サイドカー ``gathering_intent.parquet``
(``observer.gathering_intent``・**既定 OFF**・**新 L1 kind ゼロ**)。

正典: docs/research/emergent-events-and-narrative-ui.md §3 案B(assembly ledger)。
対になる事後スクリプトは ``scripts/detect_gatherings.py``(案A の集合検知 + 突合)。

何を解く問題か(現物の穴)
--------------------------
本リポには「集まった事実」を測る器(``annual.check_surge`` の ``crowd_surge``)はあるが、
**「集まろうとした事実」を測る器が無い**。しかも意図の在り処を調べると、L1 からは
**原理的に復元できない欄**が 3 つある:

  ① **計画ブロックの解決済みノード** ``agent._dayplan["blocks"][i]["node"]``
     ``day_plan.repair`` は場所カテゴリ(``food`` / ``landmark`` …)を
     ``resolve_place`` で**具体的なノードへ解決して ``b["node"]`` に焼く**。ところが
     ``plan_created`` の payload が射影するのは ``{start, end, place, act, aim,
     priority, flex}`` の 7 欄だけで **``node`` が入っていない**。実行時に出る
     ``plan_block_start.node`` は schema.py が明記するとおり
     「**実行を決めたその瞬間の現在ノード(移動前)**」= 行き先ではない。
     → **「誰が・何時に・どのノードへ行くつもりだったか」は L1 に 1 件も残らない。**

  ② **予定帳の相手側の写し** ``agent.schedule``
     ``_record_appointments`` は L1 へ ``appointment`` を**話者視点で 1 件だけ**出す。
     聞き手側の帳簿にも同じ予定が記入されるが、その記入は L1 に出ない。しかも記入は
     **在場者だけ**に絞られる(レーン乙)ので、``with`` を素直に読む事後解析は
     不在だった相手を数え **過大評価する**。

  ③ **イベントの認知集合** ``agent._known_events``
     ``event_host`` の告知は SNS/DM で伝播し ``Tools.invite`` が個体の集合へ足すが、
     「いま誰がそのイベントを知っているか」は L1 に出ない(出るのは開催宣言と、
     抽選を通って**会場に着いた**人の ``event_attend`` だけ)。Generative Agents の
     パーティ計測が「招待を知った 12 人 → 来た 5 人」と書けたのは、まさにこの
     **分母**を持っていたからである。

本サイドカーはこの 3 つを日次で 1 枚に落とし、「(場所, 時間帯) セルへ意図が何人ぶん
収束したか」を残す。**実際に集まったか**は L1 側から
``scripts/detect_gatherings.py`` が検出し、本ファイルと突合する
(= 「意図 → 実現 / 不発」の対応表。★不発の「臨界まであと何人」は現実の観測では
取れない量で、本プロジェクト固有のデータになる)。

規律(R1・**世界を 1 バイトも動かさない**)
------------------------------------------
- **既定 OFF**。``observer.gathering_intent.enabled=false`` では本 module の
  オブジェクトを作らず、1 ファイルも書かず、``sim`` に属性も生えない
  (= 既存ラン・golden 無風)。
- **新しい L1 kind を 1 つも足さない**(``register_event_kind`` を 1 度も呼ばない)。
  したがって **ON でも L1 はバイト不変**であり、因果台帳 ``CAUSE_OF_KIND`` へ
  分類を足す必要もない。出口はサイドカー parquet 1 枚だけ。
- **読み手専用**。動力学(scheduler / simulation)は本 module のバッファを 1 度も
  読まない。読むのは ``sim.agents`` と ``sim.tools.events`` の**既存の属性**だけで、
  世界状態を 1 バイトも書き換えない・乱数を 1 粒も引かない・LLM 呼数を 1 も変えない・
  プロンプトを 1 バイトも変えない。
- **no-fingerprint**: 場所カテゴリ・活動語・時間帯語といった語彙をコードに 1 語も
  書かない(値は世界から読んだ文字列をそのまま列へ流すだけ)。時間帯語 → 分の
  写像は ``society/schedule.py`` の既存表を**単一の源として借りる**(写経しない)。

撮る位置(なぜ日境界ではないのか・なぜ 1 日 1 回では足りないのか)
------------------------------------------------------------------
朝の計画は 5-10 時に作られるので、**日境界(0 時)に撮ると当日の計画がまだ無い**
(前日の食べ残しが写るだけ)。そこで初回は ``capture_min``(既定 10:00)に撮る。
こうすると当日の計画ブロックは確定済み・大半は未実行 = **意図が実現する前**が撮れる。

ただしそれだけでは**イベントを取り逃す**。``tools._host_event`` は開催を
``hours_later`` = **1-6 時間後**にしか置けない(実測: 60 体 2 日の mock で開催 11 件の
うち 10 時前の宣言は 1 件だけ)。つまり「10 時に 1 枚」では、その日に生まれる集合の
種のほとんどが**まだ存在していない**。そこで ``repeat_min``(既定 180 分)ごとに
その日の終わりまで撮り直す。1 回の費用は在場者 1 周(属性を数個読むだけ)なので、
在場 25 万 × 5 回/日でも走行費は無視できる(``memory_daily`` の 1 日ぶんより軽い)。

同じセルが複数の ``cap_min`` で出るのは**仕様**である(意図がその日のうちに
積み上がる過程がそのまま残る)。事後解析は「セル毎に n_intent が最大の行」を採れば
ピーク意図が得られる(``scripts/detect_gatherings.py`` はそうしている)。

``lead_min``(セル開始 − 撮影時刻)を列に持たせてあるので、「撮った時点で既に
過ぎていたセル」(負値)は事後に落とせる = 判定を conf に固定しない。

スキーマ(1 行 = 1 セル = (対象日, 時間帯ビン, 場所種別, 場所))
---------------------------------------------------------------
``cap_day``(撮影した日)/ ``cap_min``(撮影した sim_min)/ ``day``(意図の対象日)/
``when_bin``(その日を ``slot_min`` で割ったビン番号。時刻不明は **-1**)/
``place_kind``(``node`` = 世界のノード / ``label`` = 会話から拾った場所名の生文字列 /
``category`` = 計画の場所カテゴリでノードへ解決できなかったもの)/ ``place`` /
``n_intent``(そのセルへ意図を持つ**相異なる個体数**)/ ``n_appointment`` / ``n_plan`` /
``n_event``(内訳。同一個体が複数経路で入ると各々に数えるので合計は n_intent 以上)/
``event_ids``(JSON 配列・昇順)/ ``lead_min`` / ``sample_ids``(JSON 配列・昇順・
``sample_cap`` 件まで)

正直な限界
----------
- ``place_kind="label"`` の場所名は会話テキストから正規表現で拾った**生の文字列**で、
  ノードへの束縛は行わない(``place_label_bind`` は造語専用の別機構)。したがって
  案A の検出結果と機械的に突合できるのは ``place_kind="node"`` のセルだけである。
  ここを推測で埋めない = 偽の対応表を作らない。
- ``sample_ids`` は ``sample_cap`` 件で切る(在場 25 万のランで全 id を持つと行が
  破裂する)。**総数 ``n_intent`` は切らない**ので「臨界まであと何人」は常に読める。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize import FinalizeStreamMixin

_STEM = "gathering_intent"

#: サイドカーのスキーマ版(列を足したら上げる)。
SCHEMA = 1

#: 場所の種別(3 語。**世界の語彙ではない**ので本 module に置いてよい)。
KIND_NODE = "node"
KIND_LABEL = "label"
KIND_CATEGORY = "category"

#: 意図の出所(3 語)。
SRC_APPOINTMENT = "appointment"
SRC_PLAN = "plan"
SRC_EVENT = "event"

_SRC_ORDER = (SRC_APPOINTMENT, SRC_PLAN, SRC_EVENT)


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
def build_cfg(raw) -> dict:
    """conf の ``observer.gathering_intent`` を正準化(既定 OFF)。

    ★``isinstance(raw, dict)`` で判定しないこと: conf は OmegaConf の ``DictConfig`` で
      dict の部分型ではない(素の dict しか受けない書き方だと **ON にしても OFF に落ちる**。
      ``observer/roster.py`` が踏んだ穴と同型)。
    """
    cfg = {"enabled": False, "slot_min": 30, "capture_min": 600,
           "repeat_min": 180, "min_intent": 2, "sample_cap": 32}
    if raw is not None and hasattr(raw, "get"):
        cfg["enabled"] = bool(raw.get("enabled", False))
        cfg["slot_min"] = max(1, min(1440, int(raw.get("slot_min", 30))))
        cfg["capture_min"] = max(0, min(1439, int(raw.get("capture_min", 600))))
        cfg["repeat_min"] = max(0, min(1440, int(raw.get("repeat_min", 180))))
        cfg["min_intent"] = max(1, int(raw.get("min_intent", 2)))
        cfg["sample_cap"] = max(0, int(raw.get("sample_cap", 32)))
    return cfg


def cfg_of_config(config) -> dict:
    """``sim.cfg`` から本層の設定を引く(未宣言の旧 config でも既定 OFF に落ちる)。"""
    try:
        obs = config.get("observer", None) or {}
        raw = obs.get("gathering_intent", None)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        raw = None
    return build_cfg(raw)


# --------------------------------------------------------------------------- #
# 時刻表現 → 分(語彙表は society/schedule.py が単一の源)
# --------------------------------------------------------------------------- #
def when_minutes(when) -> int | None:
    """予定帳の ``when``(``"HH:MM"`` または時間帯語)を「その日の分」へ。不明は None。

    ★時間帯語の表は ``society/schedule.py`` の ``_BAND_MIN`` を**借りる**(写経しない)。
      ここに語を書くと no-fingerprint 契約に触れるうえ、2 箇所で表がずれる。
      ``_when_sort`` は「時刻不明」を 1440(= 日の終端より後)へ寄せる仕様なので、
      その値をそのまま「不明」として畳む。
    """
    text = str(when or "")
    if not text:
        return None
    from .. import schedule as _schedule       # 遅延 import(循環と起動費を避ける)
    sort_key = getattr(_schedule, "_when_sort", None)
    if sort_key is None:                       # pragma: no cover - 旧版互換の保険
        return None
    try:
        mins = int(sort_key(text))
    except Exception:                          # noqa: BLE001(壊れた値は「不明」)
        return None
    return None if mins >= 1440 else mins


def _bin_of(minutes, slot_min: int) -> int:
    """その日の分 → 時間帯ビン番号(不明は **-1**。捏造しない)。"""
    if minutes is None:
        return -1
    return max(0, min(1440 - 1, int(minutes))) // int(slot_min)


# --------------------------------------------------------------------------- #
# セル集計(純関数に近い形で保つ = テストから直接叩ける)
# --------------------------------------------------------------------------- #
class _Cells:
    """(day, when_bin, place_kind, place) → 出所別の個体集合 + イベント id 集合。"""

    def __init__(self):
        self.by_key: dict[tuple, dict] = {}

    def add(self, day, when_bin, place_kind, place, src, agent_id,
            event_id=None) -> None:
        key = (int(day), int(when_bin), str(place_kind), str(place))
        cell = self.by_key.get(key)
        if cell is None:
            cell = {s: set() for s in _SRC_ORDER}
            cell["events"] = set()
            self.by_key[key] = cell
        cell[src].add(int(agent_id))
        if event_id is not None:
            cell["events"].add(int(event_id))

    def rows(self, cap_day: int, cap_min: int, slot_min: int,
             min_intent: int, sample_cap: int) -> list[tuple]:
        """閾値を超えたセルを決定論の順序(key 昇順)で 1 行ずつ。"""
        out: list[tuple] = []
        for key in sorted(self.by_key):
            day, when_bin, place_kind, place = key
            cell = self.by_key[key]
            ids = set()
            for src in _SRC_ORDER:
                ids |= cell[src]
            if len(ids) < int(min_intent):
                continue
            start_min = day * 1440 + (0 if when_bin < 0 else when_bin * int(slot_min))
            out.append((
                int(cap_day), int(cap_min), int(day), int(when_bin),
                str(place_kind), str(place), len(ids),
                len(cell[SRC_APPOINTMENT]), len(cell[SRC_PLAN]),
                len(cell[SRC_EVENT]),
                json.dumps(sorted(cell["events"]), ensure_ascii=False),
                int(start_min - int(cap_min)),
                json.dumps(sorted(ids)[:int(sample_cap)], ensure_ascii=False),
            ))
        return out


def collect(sim, cells: _Cells, slot_min: int) -> None:
    """在場個体を 1 周して意図セルを積む(**読むだけ**)。

    3 経路の順序は固定(予定帳 → 計画 → イベント)。集合はいずれも id で畳むので、
    走査順は出力に漏れない(``rows`` が key 昇順・id 昇順で並べ直す)。
    """
    events = _events_of(sim)
    for agent in getattr(sim, "agents", ()) or ():
        aid = int(getattr(agent, "id", -1))
        if aid < 0:
            continue
        # ① 予定帳(会話から自動記入。**相手側の写しは L1 に出ない**)
        for entry in (getattr(agent, "schedule", None) or ()):
            try:
                day = int(entry.get("day"))
            except (TypeError, ValueError):
                continue
            cells.add(day, _bin_of(when_minutes(entry.get("when")), slot_min),
                      KIND_LABEL, str(entry.get("place", "") or ""),
                      SRC_APPOINTMENT, aid)
        # ② 朝の計画ブロック(★解決済みノード ``node`` は L1 に 1 件も出ない)
        plan = getattr(agent, "_dayplan", None) or {}
        try:
            plan_day = int(plan.get("day"))
        except (TypeError, ValueError):
            plan_day = None
        if plan_day is not None:
            for block in (plan.get("blocks") or ()):
                if not hasattr(block, "get"):
                    continue
                start = block.get("start")
                if start is None:
                    continue
                node = block.get("node")
                if node:
                    kind, place = KIND_NODE, str(node)
                else:                              # 解決できていない = カテゴリのまま
                    kind, place = KIND_CATEGORY, str(block.get("place", "") or "")
                # 日跨ぎブロック(start >= 1440)は翌日のセルへ落とす(DPH-C wrap_blocks)。
                day = plan_day + int(start) // 1440
                cells.add(day, _bin_of(int(start) % 1440, slot_min),
                          kind, place, SRC_PLAN, aid)
        # ③ イベントの認知集合(★「誰が知っているか」は L1 に出ない = 参加率の分母)
        if events:
            known = set(getattr(agent, "_known_events", None) or ())
            attending = getattr(agent, "_attending_event", None)
            if attending is not None:
                known.add(int(attending))
            for eid in sorted(known):
                ev = events.get(int(eid))
                if ev is None or ev.get("ended"):
                    continue
                start_min = _event_start_min(sim, ev)
                if start_min is None:
                    continue
                cells.add(start_min // 1440,
                          _bin_of(start_min % 1440, slot_min),
                          KIND_NODE, str(ev.get("node", "") or ""),
                          SRC_EVENT, aid, event_id=int(eid))


def _events_of(sim) -> dict:
    """``sim.tools.events``(id → イベント辞書)。無ければ空 dict(捏造しない)。"""
    tools = getattr(sim, "tools", None)
    return getattr(tools, "events", None) or {}


def _event_start_min(sim, ev) -> int | None:
    """イベントの開始 step → sim_min(中央 Δt = ``sim.clock`` が単一の源)。"""
    try:
        start_step = int(ev.get("start_step"))
    except (TypeError, ValueError):
        return None
    clock = getattr(sim, "clock", None)
    if clock is None:                              # pragma: no cover - 旧 sim 互換
        return None
    try:
        return int(clock.sim_min(start_step))
    except Exception:                              # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# サイドカー本体
# --------------------------------------------------------------------------- #
class GatheringIntent(FinalizeStreamMixin):
    """意図収束の日次サイドカー(``observer/roster.py`` / ``memory.py`` と対の設計)。"""

    def __init__(self, out_dir: Path, *, slot_min: int = 30,
                 capture_min: int = 600, repeat_min: int = 180,
                 min_intent: int = 2, sample_cap: int = 32):
        self.out_dir = Path(out_dir)
        self.slot_min = int(slot_min)
        self.capture_min = int(capture_min)
        self.repeat_min = int(repeat_min)
        self.min_intent = int(min_intent)
        self.sample_cap = int(sample_cap)
        self.rows: list[tuple] = []
        self._last_target: int | None = None
        self._n_captures = 0
        self._n_flushed = 0
        self._seg = self._next_seg()
        self._resumed = False

    # ---- 記録(観測側が 1 日 1 回・在場者を 1 周する。動力学は本バッファを読まない)----
    def capture(self, sim, day: int, sim_min: int) -> int:
        """その日の意図セルを撮る。戻り値 = 追記した行数。"""
        cells = _Cells()
        collect(sim, cells, self.slot_min)
        rows = cells.rows(int(day), int(sim_min), self.slot_min,
                          self.min_intent, self.sample_cap)
        self.rows.extend(rows)
        return len(rows)

    def last_scheduled(self, sim_min: int) -> int | None:
        """``sim_min`` 以前で最後に予定されていた撮影時刻(絶対 sim_min)。無ければ None。

        撮影時刻は 1 日につき ``capture_min``, ``+repeat_min``, ``+2*repeat_min`` …
        (``repeat_min=0`` なら 1 日 1 回)。**純関数**なので resume の据え直しと
        走行中の判定が同じ式を通る = resume==straight が式の同一性から従う。
        """
        sim_min = int(sim_min)
        day, mod = divmod(sim_min, 1440)
        if mod < self.capture_min:                 # まだ当日の初回に達していない
            day -= 1
            mod += 1440
            if day < 0:
                return None
        off = mod - self.capture_min
        if self.repeat_min > 0:
            max_off = ((1439 - self.capture_min) // self.repeat_min) * self.repeat_min
            off = min((off // self.repeat_min) * self.repeat_min, max_off)
        else:
            off = 0
        return day * 1440 + self.capture_min + off

    def on_step(self, sim, step: int, sim_min: int) -> int:
        """予定された撮影時刻を跨いだ最初の step でだけ :meth:`capture` を通す。

        それ以外の step は整数演算数回で即 0(毎 step の在場者走査はゼロ)。
        """
        sim_min = int(sim_min)
        target = self.last_scheduled(sim_min)
        if target is None:
            return 0
        if self._last_target is not None and target <= self._last_target:
            return 0
        self._last_target = target
        self._n_captures += 1
        return self.capture(sim, sim_min // 1440, sim_min)

    def resume_at(self, prev_sim_min: int) -> None:
        """resume 時に「前チャンクが撮り終えた時刻」を据える(**resume==straight の要**)。

        前チャンクは ``prev_sim_min`` の step まで実行済みなので、その時刻までに
        予定されていた撮影は全て済んでいる。``last_scheduled`` は走行中の判定と
        **同じ純関数**なので、据えた値は一気通しのランの状態と厳密に一致する。
        """
        self._resumed = True
        self._last_target = self.last_scheduled(int(prev_sim_min))

    # ---- テーブル生成(part / 直接出力で同一スキーマを共有)----
    def _table(self, rows: list) -> pa.Table:
        def col(i, typ):
            return pa.array([r[i] for r in rows], typ)
        return pa.table({
            "cap_day":       col(0, pa.int32()),
            "cap_min":       col(1, pa.int32()),
            "day":           col(2, pa.int32()),
            "when_bin":      col(3, pa.int32()),
            "place_kind":    col(4, pa.string()),
            "place":         col(5, pa.string()),
            "n_intent":      col(6, pa.int32()),
            "n_appointment": col(7, pa.int32()),
            "n_plan":        col(8, pa.int32()),
            "n_event":       col(9, pa.int32()),
            "event_ids":     col(10, pa.string()),
            "lead_min":      col(11, pa.int32()),
            "sample_ids":    col(12, pa.string()),
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
        """part 群 + 残りバッファを結合する。

        ★1 行も溜まらなかった場合でも、**1 度でも撮ったなら空の表を書く**。
          「ON だったが閾値を超えたセルが 1 つも無かった」と「そもそも OFF だった」は
          事後解析にとって別の事実であり、ファイルの有無で区別できるようにする
          (欠測を「起きなかった」と読ませない)。
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        table = (self._table(self.rows)
                 if (self.rows or self._n_captures or self._n_flushed) else None)
        return self._finalize_stream(_STEM, table)
