#!/usr/bin/env python
"""集合イベントの事後検出(レーンG 案A + 案B の突合・**読み出し専用**)。

    python scripts/detect_gatherings.py                    # 最新ラン
    python scripts/detect_gatherings.py --run lane_g_on    # ラン名を指定
    python scripts/detect_gatherings.py --min-n 4 --ratio 2.0   # 閾値を変える

正典: docs/research/emergent-events-and-narrative-ui.md §3(案A 集合検知器 / 案B 意図台帳)。
対になる走行中サイドカーは ``src/society/observer/gathering.py``(既定 OFF・finals ON)。

何をするか
----------
**案A: 集合検知器** = ``annual.check_surge`` の ``crowd_surge``(年中行事日 + 固定集会ノード
限定・1 日 1 回)を、**全ノード・全日・全時間帯**へ一般化する。L1 の位置イベントから
「同一ノード × 同一時間帯に**同時に**居た集団」を復元し、**その場所・その時刻帯の
日常ベースラインからの逸脱**として「集合」を定義する。

**案B: 意図台帳** = ``appointment`` / ``plan_created`` / ``event_host`` を
(場所, 対象日, 時間帯)セルへ畳み、「その一点へ何人の意図が集まったか」を出す。

**突合** = 意図セル × 検出された集合 → 「意図 → 実現 / 不発」の対応表。
★**不発のセル**(意図は集まったのに集合が起きなかった)には
``short_by``(= 検出閾値まであと何人足りなかったか)を付ける。現実の社会調査では
「集まらなかった集会の参加予定者数」は原理的に観測できないので、これは本シム固有の量である。

**種の系譜** = 集合の参加者が共有していた情報オブジェクト(``item_id``)の伝播木を
``transmission`` から再構成する(誰から誰へ・何 hop・集合の何分前に届いたか)。
★**本文はパースしない**。「会話に (場所, 時刻) が書かれていたか」は判定せず、
「参加者が共有していた item は何で、どう伝わったか」だけを事実として出す
(§3 案C の「時刻付きの種」はまだ世界に存在しない = 無い物を有る事にしない)。

R1(観測がシムを変えない)
--------------------------
本スクリプトは runs/<name> の parquet と config.yaml を**読むだけ**で、``src/`` を
1 行も変更しない・ランを 1 バイトも書き換えない・LLM を 1 回も呼ばない。
出力は ``runs/<name>/gatherings.json`` と ``gatherings_report.md`` の 2 枚。

検出の定義(全て決定論)
------------------------
1. **在場の復元**: 個体ごとに「いま居るノード」を L1 の位置イベントで前方補完する。
   ``arrive`` / ``stay`` = そのノードに居る / ``route_start`` / ``move_segment`` = 移動中
   (どのノードにも数えない)/ ``exit_area`` = 街の外 / ``sleep_start`` = 就寝(数えない。
   ``check_surge`` が ``not a.sleeping`` で除くのと同じ規約)/ ``wake_up`` = 起床。
   ``enter_building`` / ``exit_building`` は**平面の点を動かさない**ので在場ノードは変えない
   (``live_viewer.POS_KINDS`` と同じ意味論)。
2. **時間帯ビン**: 1 日を ``--slot-min``(既定 30 分)で割る。ビンの値は
   **そのビン内の各 step の在場数の最大**(= 同時在場のピーク)。
3. **ベースライン**: 同一 (ノード, 日内ビン) の**他の日**の値の平均(leave-one-out)。
   2 日未満のランでは日別平均が作れないので、そのノードの**全ビン平均**へ後退し
   ``baseline_src`` に何を使ったかを必ず書く(欠測を偽の値で埋めない)。
4. **判定**: ``n_peak >= --min-n`` かつ ``n_peak / max(baseline, --baseline-floor) >= --ratio``
   を開始条件、``ratio_end``(既定 ``--ratio`` の 0.6 倍)を終了条件とするヒステリシスで、
   連続するビンを 1 件の集合へまとめる(``--dur`` ビン以上続いたものだけを採る)。
5. **帰属**(なぜ集まったか。優先順に最初に当たったもの):
   ``event``(参加者が同一 event_id の ``event_attend`` を持つ)→
   ``appointment``(同ノード・同時間帯の予定が参加者に在る)→
   ``plan``(``plan_block_start`` がそのノードで発火)→
   ``joint``(``joint_activity`` / ``joint_invite`` が参加者を結ぶ)→
   ``congestion``(同時刻帯に ``transit_delay`` / ``env_feedback`` / ``crowd_surge``)→
   ``none`` = **誰も企画していないのに集まった**(最も面白い箱)。

正直な限界
----------
- ``stay`` は ``freedom.explicit_nothing`` ON のランにしか出ない。OFF のランでも
  ``arrive`` の前方補完で在場は復元できるが、**個体が最後に到着した点に居続ける**という
  仮定が入る(街を出る / 移動を始める / 就寝するまで)。これは live_viewer / make_viewer が
  地図を描くときと同じ仮定で、それ以上の情報は L1 に無い。
- 場所の突合ができるのは意図セルのうち ``place_kind="node"`` のものだけ。会話から拾った
  場所名(``label``)と計画の場所カテゴリ(``category``)はノードへ束縛しない(推測しない)。
- ``plan_created`` の blocks には**解決済みノードが入っていない**(schema がその欄を射影
  していない)。したがって L1 だけから作る意図台帳の ``plan`` 経路は
  **カテゴリ止まり**である。ノード付きの計画意図が要るなら走行中サイドカー
  ``observer.gathering_intent`` を ON にする(本スクリプトは在れば自動で読む)。

依存は標準ライブラリ + pyarrow のみ(pandas / numpy は使わない)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")
if _HERE not in sys.path:                       # 同ディレクトリの l1_stream / run_dt を import
    sys.path.insert(0, _HERE)

import l1_stream as ls                          # noqa: E402  (W2-2: L1 の有界読み)
import run_dt                                   # noqa: E402  (W2-3: ランの Δt の単一の源)

# --------------------------------------------------------------------------- #
# 既定の閾値(全て CLI で上書きできる。ここは「何も指定しない時の値」でしかない)
# --------------------------------------------------------------------------- #
DEF_SLOT_MIN = 30           # 時間帯ビンの幅(分)
DEF_MIN_N = 10              # 集合とみなす同時在場の下限(annual.crowd_threshold の既定 8 と同水準)
DEF_RATIO = 3.0             # ベースライン比の開始閾値
DEF_RATIO_END_MUL = 0.6     # 終了閾値 = 開始閾値 × これ(ヒステリシス)
DEF_DUR = 1                 # 何ビン続いたら 1 件とみなすか
DEF_BASELINE_FLOOR = 0.5    # ベースラインの下限(0 除算と「無人の場所は常に無限倍」を防ぐ)
DEF_MATCH_SLACK = 1         # 意図セル ⇄ 集合の突合で許すビンのずれ(到着の分布ぶん)
DEF_FRESH_MIN = 360         # 種の「鮮度」窓(集合の何分前までに届いた item を新着とみなすか)
DEF_TOP = 30                # レポートに載せる件数
DEF_GENEALOGY_TOP = 10      # 種の系譜を作る集合の件数(伝播の走査費を有界にする)
DEF_PARTICIPANT_CAP = 200   # 1 件の集合について保持する参加者 id の上限

#: 在場の復元に使うイベント(``live_viewer.POS_KINDS`` と同じ意味論)。
POS_KINDS = ("arrive", "stay", "route_start", "move_segment",
             "exit_area", "enter_area", "sleep_start", "wake_up")
#: 意図台帳(案B)の素材。
INTENT_KINDS = ("appointment", "plan_created", "event_host")
#: 帰属(なぜ集まったか)の素材。
ATTR_KINDS = ("event_attend", "plan_block_start", "joint_activity", "joint_invite",
              "transit_delay", "env_feedback", "crowd_surge",
              "rumor_born", "place_label_bind", "flyer_post")

PASS1_KINDS = tuple(sorted(set(POS_KINDS) | set(INTENT_KINDS) | set(ATTR_KINDS)))

#: 位置の状態コード(ノードに居ない 3 態を区別する = 「不明」と「外」を混ぜない)。
_TRANSIT = "\x00transit"
_OUTSIDE = "\x00outside"

_WANT_COLS = ["step", "sim_min", "agent_id", "kind", "payload"]


# --------------------------------------------------------------------------- #
# ラン解決 / 小道具
# --------------------------------------------------------------------------- #
def pick_run(arg_run: str | None) -> str:
    """--run 指定 or l1_events.parquet(part 群でも可)を持つ最新ランを返す。"""
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not ls.l1_paths(d):
            raise SystemExit(f"[gather] L1 が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[gather] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in sorted(os.listdir(RUNS_ROOT)):
        d = os.path.join(RUNS_ROOT, name)
        paths = ls.l1_paths(d) if os.path.isdir(d) else []
        if paths:
            cands.append((max(os.path.getmtime(p) for p in paths), d))
    if not cands:
        raise SystemExit("[gather] L1 を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _payload(raw) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _iter(run_dir: str, kinds):
    """L1 を step 昇順で逐次に読む(payload は dict へ展開)。メモリは batch 分で有界。"""
    for d in ls.iter_columns(run_dir, _WANT_COLS, kinds=tuple(kinds)):
        n = len(d["step"])
        step, smin, aid, kind, pays = (d["step"], d["sim_min"], d["agent_id"],
                                       d["kind"], d["payload"])
        for i in range(n):
            yield (int(step[i]), int(smin[i]), int(aid[i]), kind[i],
                   _payload(pays[i]))


def _bin_of(minute_of_day: int, slot_min: int) -> int:
    return max(0, min(1439, int(minute_of_day))) // int(slot_min)


# --------------------------------------------------------------------------- #
# パス1: 在場の復元(占有ピーク)+ 意図/帰属素材の収集
# --------------------------------------------------------------------------- #
class _Occupancy:
    """個体 → 現在ノードの前方補完と、(ノード, 絶対ビン) → 同時在場ピークの集計。

    「同時」の粒度は **step**(= Δt 分)。ビンの値はそのビン内の step 最大値で、
    step をまたいだ延べ人数ではない(延べだと通り過ぎた人が集合に見える)。
    """

    def __init__(self, slot_min: int, dt_min: int):
        self.slot_min = int(slot_min)
        self.dt_min = int(dt_min)
        self.place: dict[int, str] = {}          # agent_id → node / _TRANSIT / _OUTSIDE
        self.asleep: set[int] = set()
        self.peak: dict[tuple, int] = {}         # (node, abs_bin) → 同時在場ピーク
        self.node_slots: dict[str, set] = defaultdict(set)
        self._cur_step: int | None = None
        self._cur_min: int | None = None
        self.n_steps = 0

    # ---- step 境界で「その step の在場」を該当ビンへ反映する ----
    def close_step(self) -> None:
        if self._cur_step is None:
            return
        abs_bin = (int(self._cur_min) // self.slot_min)
        counts: Counter = Counter()
        for aid, node in self.place.items():
            if node in (_TRANSIT, _OUTSIDE) or aid in self.asleep:
                continue
            counts[node] += 1
        for node, n in counts.items():
            key = (node, abs_bin)
            if n > self.peak.get(key, 0):
                self.peak[key] = n
            self.node_slots[node].add(abs_bin)
        self.n_steps += 1

    def feed(self, step: int, sim_min: int, aid: int, kind: str, pay: dict) -> None:
        if self._cur_step is not None and step != self._cur_step:
            self.close_step()
        self._cur_step, self._cur_min = step, sim_min
        if kind in ("arrive", "stay"):
            node = pay.get("node")
            self.place[aid] = str(node) if node else _TRANSIT
        elif kind in ("route_start", "move_segment"):
            self.place[aid] = _TRANSIT
        elif kind == "exit_area":
            self.place[aid] = _OUTSIDE
            self.asleep.discard(aid)
        elif kind == "enter_area":
            self.place[aid] = _TRANSIT       # 帰還直後の居場所は次の arrive まで不明
        elif kind == "sleep_start":
            self.asleep.add(aid)
        elif kind == "wake_up":
            self.asleep.discard(aid)


def scan_pass1(run_dir: str, slot_min: int, dt_min: int) -> dict:
    """L1 を 1 周して (a) 占有ピーク (b) 意図素材 (c) 帰属素材 を作る。"""
    occ = _Occupancy(slot_min, dt_min)
    appointments: list[dict] = []
    plan_blocks: list[dict] = []
    hosted: dict[int, dict] = {}
    attend: dict[tuple, set] = defaultdict(set)      # (node, abs_bin) → event_id 集合
    attend_by_event: dict[int, set] = defaultdict(set)
    plan_starts: dict[tuple, set] = defaultdict(set)  # (node, abs_bin) → agent 集合
    joint_marks: dict[int, set] = defaultdict(set)    # abs_bin → agent 集合
    congestion: dict[int, list] = defaultdict(list)   # abs_bin → [kind, ...]
    seeds_at_node: dict[str, list] = defaultdict(list)  # node → [{item_id, step, kind}]
    kinds_seen: Counter = Counter()

    for step, smin, aid, kind, pay in _iter(run_dir, PASS1_KINDS):
        kinds_seen[kind] += 1
        occ.feed(step, smin, aid, kind, pay)
        abs_bin = smin // slot_min
        if kind == "appointment":
            appointments.append({"step": step, "sim_min": smin, "agent_id": aid,
                                 "day": _as_int(pay.get("day")),
                                 "when": str(pay.get("when", "") or ""),
                                 "what": str(pay.get("what", "") or ""),
                                 "place": str(pay.get("place", "") or ""),
                                 "with": [int(v) for v in (pay.get("with") or [])
                                          if isinstance(v, int)]})
        elif kind == "plan_created":
            day = smin // 1440
            for idx, b in enumerate(pay.get("blocks") or ()):
                if not isinstance(b, dict):
                    continue
                start = _as_int(b.get("start"))
                if start is None:
                    continue
                plan_blocks.append({"agent_id": aid, "day": day + start // 1440,
                                    "start": start % 1440, "block": idx,
                                    "place": str(b.get("place", "") or ""),
                                    "act": str(b.get("act", "") or "")})
        elif kind == "event_host":
            eid = _as_int(pay.get("event_id"))
            if eid is not None:
                hosted[eid] = {"event_id": eid, "host": aid, "step": step,
                               "sim_min": smin,
                               "node": str(pay.get("node", "") or ""),
                               "title": str(pay.get("title", "") or ""),
                               "start_step": _as_int(pay.get("start_step"))}
        elif kind == "event_attend":
            eid = _as_int(pay.get("event_id"))
            node = occ.place.get(aid)
            if eid is not None:
                attend_by_event[eid].add(aid)
                if node and node not in (_TRANSIT, _OUTSIDE):
                    attend[(node, abs_bin)].add(eid)
        elif kind == "plan_block_start":
            node = pay.get("node") or occ.place.get(aid)
            if node and node not in (_TRANSIT, _OUTSIDE):
                plan_starts[(str(node), abs_bin)].add(aid)
        elif kind in ("joint_activity", "joint_invite"):
            joint_marks[abs_bin].add(aid)
            for v in (pay.get("with") or []):
                if isinstance(v, int):
                    joint_marks[abs_bin].add(int(v))
            inv = pay.get("invitee")
            if isinstance(inv, int):
                joint_marks[abs_bin].add(int(inv))
        elif kind in ("transit_delay", "env_feedback", "crowd_surge"):
            congestion[abs_bin].append(kind)
        elif kind in ("rumor_born", "place_label_bind", "flyer_post"):
            node = pay.get("node")
            if node:
                seeds_at_node[str(node)].append(
                    {"item_id": _as_str(pay.get("item_id") or pay.get("word")),
                     "kind": kind, "step": step, "sim_min": smin,
                     "author": aid})
    occ.close_step()
    return {"occ": occ, "appointments": appointments, "plan_blocks": plan_blocks,
            "hosted": hosted, "attend": attend, "attend_by_event": attend_by_event,
            "plan_starts": plan_starts, "joint_marks": joint_marks,
            "congestion": congestion, "seeds_at_node": seeds_at_node,
            "kinds_seen": kinds_seen}


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_str(v) -> str:
    return "" if v is None else str(v)


# --------------------------------------------------------------------------- #
# ベースライン + 検出(案A)
# --------------------------------------------------------------------------- #
def build_baselines(occ: _Occupancy, bins_per_day: int) -> tuple[dict, dict, str]:
    """(ノード, 日内ビン) → ベースライン。leave-one-out の日別平均が第一候補。

    戻り値 (baseline, node_mean, src)。``src`` は "cross_day" か "node_mean" で、
    どちらを使ったかを必ず出力へ書く(**欠測を偽の値で埋めない**の徹底)。
    """
    by_slot: dict[tuple, list] = defaultdict(list)   # (node, slot_of_day) → [値, ...]
    days: set = set()
    for (node, abs_bin), n in occ.peak.items():
        day, slot = divmod(abs_bin, bins_per_day)
        days.add(day)
        by_slot[(node, slot)].append((day, n))
    node_total: dict[str, list] = defaultdict(list)
    for (node, _slot), vals in by_slot.items():
        node_total[node] += [n for _d, n in vals]
    node_mean = {node: (sum(v) / len(v) if v else 0.0)
                 for node, v in node_total.items()}
    src = "cross_day" if len(days) >= 2 else "node_mean"
    baseline: dict[tuple, float] = {}
    for (node, slot), vals in by_slot.items():
        total = sum(n for _d, n in vals)
        for day, n in vals:
            key = (node, day * bins_per_day + slot)
            if src == "cross_day" and len(vals) >= 2:
                # leave-one-out: 「その日を除いた同じ時間帯の平均」= 当日の値が
                # 自分自身を正当化しない(自己参照でベースラインが上がるのを防ぐ)。
                baseline[key] = (total - n) / (len(vals) - 1)
            else:
                baseline[key] = node_mean.get(node, 0.0)
    return baseline, node_mean, src


def detect_gatherings(occ: _Occupancy, baseline: dict, bins_per_day: int, *,
                      min_n: int, ratio: float, ratio_end: float,
                      dur: int, floor: float) -> list[dict]:
    """ヒステリシス付きの連続ビン走査で「1 回の集合 = 1 件」にまとめる。"""
    out: list[dict] = []
    for node in sorted(occ.node_slots):
        bins = sorted(occ.node_slots[node])
        run: list[tuple] = []
        prev_bin: int | None = None

        def _flush(seq):
            if not seq:
                return
            peak = max(n for _b, n, _base in seq)
            if len(seq) < dur or peak < min_n:
                return
            top = max(seq, key=lambda t: (t[1], -t[0]))
            base_at_peak = top[2]
            out.append({
                "node": node,
                "start_bin": seq[0][0], "end_bin": seq[-1][0],
                "dur_bins": len(seq),
                "n_peak": peak,
                "peak_bin": top[0],
                "baseline": round(base_at_peak, 3),
                "ratio": round(peak / max(base_at_peak, floor), 3),
                "series": [n for _b, n, _base in seq],
            })

        for b in bins:
            n = occ.peak[(node, b)]
            base = baseline.get((node, b), 0.0)
            r = n / max(base, floor)
            hot_start = (n >= min_n and r >= ratio)
            hot_cont = (n >= max(1, int(min_n * 0.6)) and r >= ratio_end)
            contiguous = (prev_bin is not None and b == prev_bin + 1)
            if run and contiguous and hot_cont:
                run.append((b, n, base))
            else:
                _flush(run)
                run = [(b, n, base)] if hot_start else []
            prev_bin = b
        _flush(run)
    out.sort(key=lambda g: (-g["n_peak"], g["start_bin"], g["node"]))
    return out


# --------------------------------------------------------------------------- #
# パス2: 検出された窓の参加者を拾う(窓の外は 1 件も持たない = メモリ有界)
# --------------------------------------------------------------------------- #
def scan_participants(run_dir: str, gatherings: list[dict], slot_min: int,
                      cap: int) -> None:
    """検出窓 (node, [start_bin, end_bin]) に居た個体 id を各件へ焼き込む。

    パス1 と**同じ状態機械**で在場を復元し(定義の二重化を避ける)、窓に入っている
    step でだけ id を拾う。したがって費用は L1 の 1 周 + 窓ぶんの集合演算だけ。
    """
    if not gatherings:
        return
    windows: dict[str, list] = defaultdict(list)
    for idx, g in enumerate(gatherings):
        windows[g["node"]].append((g["start_bin"], g["end_bin"], idx))
        g["participants"] = []
    members: list[set] = [set() for _ in gatherings]
    occ = _Occupancy(slot_min, 0)

    def _harvest():
        if occ._cur_min is None:
            return
        abs_bin = int(occ._cur_min) // slot_min
        for aid, node in occ.place.items():
            if node in (_TRANSIT, _OUTSIDE) or aid in occ.asleep:
                continue
            for lo, hi, idx in windows.get(node, ()):
                if lo <= abs_bin <= hi:
                    members[idx].add(aid)

    prev_step = None
    for step, smin, aid, kind, pay in _iter(run_dir, POS_KINDS):
        if prev_step is not None and step != prev_step:
            _harvest()
        occ.feed(step, smin, aid, kind, pay)
        prev_step = step
    _harvest()
    for idx, g in enumerate(gatherings):
        ids = sorted(members[idx])
        g["n_members"] = len(ids)
        g["participants"] = ids[:cap]
        g["participants_truncated"] = len(ids) > cap


# --------------------------------------------------------------------------- #
# 帰属(なぜ集まったか)
# --------------------------------------------------------------------------- #
def attribute(gatherings: list[dict], mats: dict, bins_per_day: int,
              slot_min: int) -> None:
    """優先順に最初に当たった説明を ``via`` に入れる(当たらなければ ``none``)。"""
    attend, plan_starts = mats["attend"], mats["plan_starts"]
    joint_marks, congestion = mats["joint_marks"], mats["congestion"]
    appts = mats["appointments"]
    seeds_at_node = mats["seeds_at_node"]
    # 予定は (対象日, 時間帯) が場所名でしか書かれないので、**時間の一致だけ**で候補にし、
    # 場所は「予定に場所名が書かれていたか」を refs へ添えるにとどめる(推測しない)。
    appt_by_cell: dict[tuple, set] = defaultdict(set)
    for a in appts:
        if a["day"] is None:
            continue
        mins = _when_minutes(a["when"])
        if mins is None:
            continue
        key = (a["day"] * bins_per_day + _bin_of(mins, slot_min))
        appt_by_cell[key].add(a["agent_id"])
        for w in a["with"]:
            appt_by_cell[key].add(w)

    for g in gatherings:
        node, lo, hi = g["node"], g["start_bin"], g["end_bin"]
        members = set(g.get("participants") or ())
        refs: dict = {}
        via = "none"
        eids = set()
        for b in range(lo, hi + 1):
            eids |= attend.get((node, b), set())
        if eids:
            via, refs["event_ids"] = "event", sorted(eids)
            hosts = [mats["hosted"].get(e, {}).get("host") for e in sorted(eids)]
            refs["hosts"] = [h for h in hosts if h is not None]
        if via == "none":
            hit = set()
            for b in range(lo, hi + 1):
                hit |= appt_by_cell.get(b, set()) & members
            if len(hit) >= 2:
                via, refs["appointment_agents"] = "appointment", sorted(hit)
        if via == "none":
            hit = set()
            for b in range(lo, hi + 1):
                hit |= plan_starts.get((node, b), set())
            if len(hit) >= 2:
                via, refs["plan_agents"] = "plan", sorted(hit)
        if via == "none":
            hit = set()
            for b in range(lo, hi + 1):
                hit |= joint_marks.get(b, set()) & members
            if len(hit) >= 2:
                via, refs["joint_agents"] = "joint", sorted(hit)
        if via == "none":
            marks = []
            for b in range(lo, hi + 1):
                marks += congestion.get(b, [])
            if marks:
                via, refs["congestion_kinds"] = "congestion", sorted(set(marks))
        seeds = seeds_at_node.get(node) or []
        if seeds:                                  # 帰属の判定には使わない(場所の履歴として添える)
            refs["seeds_at_node"] = seeds[:5]
        g["via"] = via
        g["refs"] = refs


def _when_minutes(when: str):
    """予定の ``when``(``"HH:MM"`` / 時間帯語)→ その日の分。不明は None。

    ★語彙表は ``society/schedule.py`` の ``_BAND_MIN`` を**借りる**(写経して二重定義に
      しない)。src を import できない配布形態では時刻表記だけを解し、時間帯語は
      「不明」に落とす(推測しない)。
    """
    text = str(when or "")
    if not text:
        return None
    try:
        sys.path.insert(0, os.path.join(_ROOT, "src"))
        from society import schedule as _schedule
        mins = int(_schedule._when_sort(text))
        return None if mins >= 1440 else mins
    except Exception:                              # noqa: BLE001
        if ":" in text:
            head, _, tail = text.partition(":")
            try:
                return (int(head) % 24) * 60 + int(tail) % 60
            except ValueError:
                return None
        return None


# --------------------------------------------------------------------------- #
# 意図台帳(案B)と突合
# --------------------------------------------------------------------------- #
def build_intent_cells(mats: dict, bins_per_day: int, slot_min: int,
                       spd: int, dt_min: int, start_min: int) -> list[dict]:
    """L1 だけから作る意図セル(場所種別つき)。ノード付きは event 経路のみ。"""
    cells: dict[tuple, dict] = {}

    def _cell(day, wbin, kind, place):
        key = (int(day), int(wbin), str(kind), str(place))
        c = cells.get(key)
        if c is None:
            c = {"day": key[0], "when_bin": key[1], "place_kind": key[2],
                 "place": key[3], "appointment": set(), "plan": set(),
                 "event": set(), "event_ids": set()}
            cells[key] = c
        return c

    for a in mats["appointments"]:
        if a["day"] is None:
            continue
        mins = _when_minutes(a["when"])
        wbin = -1 if mins is None else _bin_of(mins, slot_min)
        c = _cell(a["day"], wbin, "label", a["place"])
        c["appointment"].add(a["agent_id"])
        for w in a["with"]:
            c["appointment"].add(w)
    for b in mats["plan_blocks"]:
        # ★L1 の plan_created は**解決済みノードを射影していない**ので category 止まり。
        c = _cell(b["day"], _bin_of(b["start"], slot_min), "category", b["place"])
        c["plan"].add(b["agent_id"])
    for eid, ev in sorted(mats["hosted"].items()):
        start_step = ev.get("start_step")
        if start_step is None:
            continue
        smin = start_min + int(start_step) * int(dt_min)
        c = _cell(smin // 1440, _bin_of(smin % 1440, slot_min), "node", ev["node"])
        c["event"] |= set(mats["attend_by_event"].get(eid, ()))
        c["event"].add(ev["host"])
        c["event_ids"].add(eid)
    out = []
    for key in sorted(cells):
        c = cells[key]
        ids = c["appointment"] | c["plan"] | c["event"]
        out.append({"day": c["day"], "when_bin": c["when_bin"],
                    "place_kind": c["place_kind"], "place": c["place"],
                    "n_intent": len(ids),
                    "n_appointment": len(c["appointment"]),
                    "n_plan": len(c["plan"]), "n_event": len(c["event"]),
                    "event_ids": sorted(c["event_ids"]),
                    "sample_ids": sorted(ids)[:32], "src": "l1"})
    return out


def load_sidecar_cells(run_dir: str) -> list[dict] | None:
    """走行中サイドカー ``gathering_intent.parquet`` を読む(無ければ None)。

    ★これが在るランでは、L1 からは復元できない 3 欄(計画ブロックの**解決済みノード** /
      予定帳の**相手側の写し** / イベントの**認知集合**)が入っているので、
      突合の主表はこちらを使う。
    """
    path = os.path.join(run_dir, "gathering_intent.parquet")
    if not os.path.isfile(path):
        return None
    import pyarrow.parquet as pq
    tbl = pq.read_table(path)
    d = tbl.to_pydict()
    out = []
    for i in range(tbl.num_rows):
        out.append({"day": int(d["day"][i]), "when_bin": int(d["when_bin"][i]),
                    "place_kind": d["place_kind"][i], "place": d["place"][i],
                    "n_intent": int(d["n_intent"][i]),
                    "n_appointment": int(d["n_appointment"][i]),
                    "n_plan": int(d["n_plan"][i]),
                    "n_event": int(d["n_event"][i]),
                    "event_ids": json.loads(d["event_ids"][i] or "[]"),
                    "sample_ids": json.loads(d["sample_ids"][i] or "[]"),
                    "cap_day": int(d["cap_day"][i]),
                    "cap_min": int(d["cap_min"][i]),
                    "lead_min": int(d["lead_min"][i]), "src": "sidecar"})
    return out


def merge_cells(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """同一セル (day, when_bin, place_kind, place) は **n_intent が最大の行**を採る。

    ★サイドカーは 1 日数回撮るので同じセルが複数回出る(意図が積み上がる過程)。
      ピーク意図が「臨界まであと何人」の分子なので最大を採るのが正しい。
    ★L1 側の cell はサイドカーが撮れなかったイベント(撮影時刻より後に宣言されたもの)を
      拾えるので、**捨てずに併合**する。どちらから来た行かは ``src`` に残る。
    """
    best: dict[tuple, dict] = {}
    for c in list(primary) + list(secondary):
        key = (c["day"], c["when_bin"], c["place_kind"], c["place"])
        cur = best.get(key)
        if cur is None or c["n_intent"] > cur["n_intent"]:
            best[key] = c
    return [best[k] for k in sorted(best)]


def reconcile(cells: list[dict], gatherings: list[dict], occ: _Occupancy,
              baseline: dict, bins_per_day: int, *, min_n: int, min_intent: int,
              ratio: float, floor: float, slack: int = 1) -> list[dict]:
    """意図セル(``place_kind="node"`` のみ)× 検出集合 → 実現 / 不発の対応表。

    ★``slack`` ビンの緩みを持たせる: 「20:00 に集まろう」の集合が 20:30 にピークを
      迎えるのは遅刻ではなく**到着の分布**であって、厳密一致を要求すると実現を
      取りこぼす(過小評価は過大評価と同じくらい悪い)。緩みの幅は出力の
      ``params.match_slack`` に必ず残す。
    ★不発には ``short_by`` = **「検出の臨界まであと何人だったか」**を付ける。臨界は
      2 条件の**両方**なので ``need = max(min_n, ceil(ratio × max(baseline, floor)))``
      で測り(頭数だけを見ると「あと 0 人なのに不発」という読めない行が出る)、
      緩み窓の各ビンで測った不足の**最小値**を採る(= 最も惜しかったビン)。
      ``fail_reason`` に ``headcount`` / ``ratio`` のどちらで落ちたかを残す。
      実在場のピーク ``n_present`` は L1 から測れるので、
      「意図は N 人集まったが、実際にはピーク M 人しか居らず臨界に K 人届かなかった」
      という**現実では観測できない三つ組**がそのまま残る。
    """
    import math
    hit: dict[tuple, list] = defaultdict(list)
    for idx, g in enumerate(gatherings):
        for b in range(g["start_bin"], g["end_bin"] + 1):
            hit[(g["node"], b)].append(idx)
    # ★ランが到達しなかった未来の日(予定は horizon_days=14 先まで書ける)を
    #   「不発」に数えない: 観測されていない時間帯を失敗として計上すると実現率が
    #   構造的に下振れする(欠測を「起きなかった」と読ませない)。
    last_bin = max(b for _n, b in occ.peak) if occ.peak else -1
    out = []
    for c in cells:
        if c["place_kind"] != "node" or c["when_bin"] < 0:
            continue
        if c["n_intent"] < min_intent:
            continue
        abs_bin = c["day"] * bins_per_day + c["when_bin"]
        if abs_bin > last_bin:
            continue
        idxs: list = []
        n_present, best_gap, best_base, reason = 0, None, None, None
        for d in range(-abs(slack), abs(slack) + 1):
            b = abs_bin + d
            idxs += hit.get((c["place"], b)) or []
            n_b = int(occ.peak.get((c["place"], b), 0))
            n_present = max(n_present, n_b)
            base_b = float(baseline.get((c["place"], b), 0.0))
            need_ratio = math.ceil(ratio * max(base_b, floor))
            need = max(int(min_n), int(need_ratio))
            gap = max(0, need - n_b)
            if best_gap is None or gap < best_gap:
                best_gap, best_base = gap, base_b
                reason = "headcount" if n_b < min_n else "ratio"
        row = {"day": c["day"], "when_bin": c["when_bin"], "place": c["place"],
               "n_intent": c["n_intent"], "n_present": int(n_present),
               "baseline": round(best_base or 0.0, 3),
               "src": c.get("src", "l1"), "event_ids": c.get("event_ids") or [],
               "realized": bool(idxs)}
        if idxs:
            g = gatherings[min(idxs)]
            row["gathering"] = {"node": g["node"], "n_peak": g["n_peak"],
                                "via": g.get("via"), "ratio": g["ratio"],
                                "start_bin": g["start_bin"]}
            row["arrival_rate"] = (round(g["n_peak"] / c["n_intent"], 3)
                                   if c["n_intent"] else None)
        else:
            row["short_by"] = int(best_gap or 0)
            row["fail_reason"] = reason
            row["arrival_rate"] = (round(n_present / c["n_intent"], 3)
                                   if c["n_intent"] else None)
        out.append(row)
    out.sort(key=lambda r: (r["realized"], -r["n_intent"], r["day"], r["when_bin"]))
    return out


# --------------------------------------------------------------------------- #
# 種の系譜(flash mob 型。transmission から再構成できる範囲だけ)
# --------------------------------------------------------------------------- #
def scan_genealogy(run_dir: str, gatherings: list[dict], slot_min: int, *,
                   top: int, fresh_min: int = DEF_FRESH_MIN) -> list[dict]:
    """上位 ``top`` 件の集合について、参加者が共有していた item の伝播木を作る。

    ★本文はパースしない。使うのは ``transmission{item_id, from, channel}`` の
      **辺だけ**で、「参加者の何人がその item を持っていたか」「木の深さ・幅」
      「集合の何分前に届いたか」を出す。(場所, 時刻) が本文に載っていたかは
      判定しない = 判らないことを判ったことにしない。
    """
    targets = gatherings[:max(0, int(top))]
    if not targets:
        return []
    want: set = set()
    for g in targets:
        want |= set(g.get("participants") or ())
    if not want:
        return []
    # item_id → [(step, from, to, channel)] を、**参加者に触れる辺だけ**集める。
    edges: dict[str, list] = defaultdict(list)
    for step, smin, aid, _kind, pay in _iter(run_dir, ("transmission",)):
        src = pay.get("from")
        if aid not in want and (not isinstance(src, int) or src not in want):
            continue
        item = _as_str(pay.get("item_id"))
        if not item:
            continue
        edges[item].append((step, smin, src if isinstance(src, int) else None,
                            aid, str(pay.get("channel", "") or "")))
    out = []
    for g in targets:
        members = set(g.get("participants") or ())
        if not members:
            continue
        start_min = g["start_bin"] * slot_min
        ranked = []
        for item, evs in edges.items():
            # ★集合より**後**に届いた辺は種になりえない(因果の向きを守る)。
            before = [e for e in evs if e[1] <= start_min]
            holders = {to for _s, _m, _f, to, _c in before if to in members}
            if len(holders) < 2:
                continue
            parents = {to: frm for _s, _m, frm, to, _c in sorted(before)}
            # ★「何人に届いたか」であって「何本の辺が来たか」ではない: 同じ item を
            #   同じ人が何度も受け取りうるので、**個体ごとに最も新しい到達**へ畳む
            #   (畳まないと fresh が参加者数を超えて意味を失う)。
            latest: dict[int, int] = {}
            for _s, m, _f, to, _c in before:
                if to in members:
                    lag = start_min - m
                    if to not in latest or lag < latest[to]:
                        latest[to] = lag
            lead = sorted(latest.values())
            # 「直前に届いた」ほうが種らしい。全員が持っている語(share 1.0)が並ぶ
            # 小さいランでは、この鮮度だけが順位を分ける唯一の情報になる。
            fresh = sum(1 for v in lead if 0 <= v <= fresh_min)
            ranked.append({
                "item_id": item,
                "holders_in_gathering": len(holders),
                "share": round(len(holders) / len(members), 3),
                "fresh_holders": fresh,
                "fresh_window_min": fresh_min,
                "n_edges_before": len(before),
                "tree_depth": _tree_depth(parents),
                "max_breadth": max(Counter(
                    frm for _s, _m, frm, _t, _c in before
                    if frm is not None).values(), default=0),
                "channels": sorted({c for *_r, c in before if c}),
                "lead_min_median": _median(lead) if lead else None,
            })
        ranked.sort(key=lambda r: (-r["fresh_holders"],
                                   -r["holders_in_gathering"], r["item_id"]))
        if ranked:
            out.append({"node": g["node"], "start_bin": g["start_bin"],
                        "day": g.get("day"), "clock": g.get("clock"),
                        "n_peak": g["n_peak"], "via": g.get("via"),
                        "n_members": g.get("n_members", len(members)),
                        "items": ranked[:5]})
    return out


def _tree_depth(parents: dict) -> int:
    """親写像(子 → 親)から最大の hop 数を返す(閉路は打ち切る)。"""
    best = 0
    for node in parents:
        seen, cur, d = {node}, node, 0
        while True:
            p = parents.get(cur)
            if p is None or p in seen:
                break
            seen.add(p)
            cur, d = p, d + 1
        best = max(best, d)
    return best


def _median(xs: list):
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# --------------------------------------------------------------------------- #
# 解析本体
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, *, slot_min: int = DEF_SLOT_MIN, min_n: int = DEF_MIN_N,
            ratio: float = DEF_RATIO, ratio_end: float | None = None,
            dur: int = DEF_DUR, floor: float = DEF_BASELINE_FLOOR,
            genealogy_top: int = DEF_GENEALOGY_TOP,
            participant_cap: int = DEF_PARTICIPANT_CAP,
            min_intent: int = 2, match_slack: int = DEF_MATCH_SLACK,
            fresh_min: int = DEF_FRESH_MIN) -> dict:
    dt_min = run_dt.dt_min_of(run_dir, notify=False)
    spd = run_dt.steps_per_day(run_dir, dt_min=dt_min, notify=False)
    if 1440 % slot_min:
        raise SystemExit(f"[gather] --slot-min は 1440 の約数にすること: {slot_min}")
    bins_per_day = 1440 // slot_min
    ratio_end = ratio * DEF_RATIO_END_MUL if ratio_end is None else ratio_end

    mats = scan_pass1(run_dir, slot_min, dt_min)
    occ: _Occupancy = mats["occ"]
    baseline, node_mean, base_src = build_baselines(occ, bins_per_day)
    gatherings = detect_gatherings(occ, baseline, bins_per_day, min_n=min_n,
                                   ratio=ratio, ratio_end=ratio_end, dur=dur,
                                   floor=floor)
    scan_participants(run_dir, gatherings, slot_min, participant_cap)
    attribute(gatherings, mats, bins_per_day, slot_min)
    for g in gatherings:
        g["day"] = g["start_bin"] // bins_per_day
        g["slot_of_day"] = g["start_bin"] % bins_per_day
        g["start_min"] = g["start_bin"] * slot_min
        g["clock"] = "%02d:%02d" % divmod(g["start_min"] % 1440, 60)

    start_min = _run_start_min(run_dir)
    cells_l1 = build_intent_cells(mats, bins_per_day, slot_min, spd, dt_min,
                                  start_min)
    cells_sc = load_sidecar_cells(run_dir)
    # サイドカーが在れば主表(L1 では復元できない 3 欄を持つ)。ただし L1 側だけが持つ
    # イベント(撮影時刻より後に宣言されたもの)を捨てないよう**併合**する。
    cells = (merge_cells(cells_sc, cells_l1) if cells_sc is not None else cells_l1)
    table = reconcile(cells, gatherings, occ, baseline, bins_per_day,
                      min_n=min_n, min_intent=min_intent, ratio=ratio,
                      floor=floor, slack=match_slack)
    genealogy = scan_genealogy(run_dir, gatherings, slot_min,
                               top=genealogy_top, fresh_min=fresh_min)

    via_counts = Counter(g["via"] for g in gatherings)
    realized = [r for r in table if r["realized"]]
    return {
        "run": os.path.basename(os.path.abspath(run_dir)),
        "run_dir": os.path.abspath(run_dir),
        "params": {"slot_min": slot_min, "min_n": min_n, "ratio": ratio,
                   "ratio_end": round(ratio_end, 3), "dur_bins": dur,
                   "baseline_floor": floor, "min_intent": min_intent,
                   "match_slack": match_slack, "fresh_min": fresh_min,
                   "dt_min": dt_min, "steps_per_day": spd,
                   "bins_per_day": bins_per_day},
        "coverage": {
            "steps_seen": occ.n_steps,
            "nodes_seen": len(occ.node_slots),
            "occupied_cells": len(occ.peak),
            "baseline_src": base_src,
            "kinds_seen": dict(sorted(mats["kinds_seen"].items())),
            "intent_source": "sidecar" if cells_sc is not None else "l1",
            "sidecar_rows": (len(cells_sc) if cells_sc is not None else 0),
        },
        "gatherings": gatherings,
        "via_counts": dict(sorted(via_counts.items())),
        "intent_cells": cells,
        "intent_outcome": table,
        "intent_summary": {
            "n_cells_total": len(cells),
            "n_cells_node": len(table),
            "n_cells_beyond_run": sum(
                1 for c in cells
                if c["place_kind"] == "node" and c["when_bin"] >= 0
                and c["n_intent"] >= min_intent
                and c["day"] * bins_per_day + c["when_bin"] >
                (max(b for _n, b in occ.peak) if occ.peak else -1)),
            "n_realized": len(realized),
            "n_unrealized": len(table) - len(realized),
            "realize_rate": (round(len(realized) / len(table), 3) if table else None),
            "median_short_by": _median(sorted(
                r["short_by"] for r in table if not r["realized"])),
        },
        "genealogy": genealogy,
    }


def _run_start_min(run_dir: str) -> int:
    """ランの開始 sim_min(``run.start_min``)。読めなければ既定 7:00(make_viewer と同値)。"""
    path = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(path):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(path)
            v = cfg.get("run", {}).get("start_min", None)
            if v is not None:
                return int(v)
        except Exception:                          # noqa: BLE001
            pass
    return 7 * 60


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def build_report(res: dict, top: int = DEF_TOP) -> str:
    p, cov = res["params"], res["coverage"]
    L = [f"# 集合イベント検出レポート — {res['run']}", "",
         f"Δt={p['dt_min']}分 / ビン={p['slot_min']}分({p['bins_per_day']}/日) / "
         f"閾値 n>={p['min_n']} かつ 比>={p['ratio']}(終了 {p['ratio_end']})/ "
         f"継続>={p['dur_bins']}ビン",
         f"走査 step={cov['steps_seen']} / ノード={cov['nodes_seen']} / "
         f"占有セル={cov['occupied_cells']} / ベースライン={cov['baseline_src']}",
         f"意図台帳の出所={cov['intent_source']}"
         + (f"(サイドカー {cov['sidecar_rows']} 行)"
            if cov["intent_source"] == "sidecar" else "(L1 のみ = 計画はカテゴリ止まり)"),
         "", "## 1. 検出された集合", ""]
    if not res["gatherings"]:
        L.append("(閾値を超える集合なし。--min-n / --ratio を下げて再走査できる)")
    else:
        L.append(f"合計 {len(res['gatherings'])} 件 / 帰属内訳: "
                 + ", ".join(f"{k}={v}" for k, v in res["via_counts"].items()))
        L += ["", "| # | 日 | 時刻 | ノード | ピーク | ベース | 比 | 継続 | 帰属 | 参加者 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for i, g in enumerate(res["gatherings"][:top], 1):
            L.append(f"| {i} | {g['day']} | {g['clock']} | {g['node']} | "
                     f"{g['n_peak']} | {g['baseline']} | {g['ratio']} | "
                     f"{g['dur_bins']} | {g['via']} | {g.get('n_members', 0)} |")
        none_g = [g for g in res["gatherings"] if g["via"] == "none"]
        if none_g:
            L += ["", "### ★ via=none(誰も企画していないのに集まった)", ""]
            for g in none_g[:top]:
                L.append(f"- day{g['day']} {g['clock']} `{g['node']}` "
                         f"ピーク{g['n_peak']}人(ベース {g['baseline']} / "
                         f"{g['ratio']}倍・{g['dur_bins']}ビン継続)")
    s = res["intent_summary"]
    L += ["", "## 2. 意図 → 実現 / 不発", "",
          f"ノード付き意図セル {s['n_cells_node']} 件 / 実現 {s['n_realized']} / "
          f"不発 {s['n_unrealized']}"
          + (f" / 実現率 {s['realize_rate']}" if s["realize_rate"] is not None else "")
          + (f"(ラン到達外の未来日 {s['n_cells_beyond_run']} 件は分母から除外)"
             if s["n_cells_beyond_run"] else ""),
          ""]
    unreal = [r for r in res["intent_outcome"] if not r["realized"]]
    if unreal:
        L += [f"★ 不発セルの「臨界まであと何人」中央値 = {s['median_short_by']}", "",
              "| 日 | ビン | ノード | 意図 | 実在場 | ベース | あと何人 | 落ちた条件 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in unreal[:top]:
            L.append(f"| {r['day']} | {r['when_bin']} | {r['place']} | "
                     f"{r['n_intent']} | {r['n_present']} | {r['baseline']} | "
                     f"{r['short_by']} | {r['fail_reason']} |")
    real = [r for r in res["intent_outcome"] if r["realized"]]
    if real:
        L += ["", "| 日 | ビン | ノード | 意図 | 実現ピーク | 到達率 | 帰属 |",
              "|---|---|---|---|---|---|---|"]
        for r in real[:top]:
            g = r["gathering"]
            L.append(f"| {r['day']} | {r['when_bin']} | {r['place']} | "
                     f"{r['n_intent']} | {g['n_peak']} | {r['arrival_rate']} | "
                     f"{g['via']} |")
    L += ["", "## 3. 種の系譜(参加者が共有していた item の伝播)", ""]
    if not res["genealogy"]:
        L.append("(参加者を 2 人以上つなぐ item が無い = 伝播由来の集合は検出されず)")
    else:
        for e in res["genealogy"]:
            L.append(f"- `{e['node']}` day{e['day']} {e['clock']} ピーク{e['n_peak']}人"
                     f"(via={e['via']}・参加 {e['n_members']}人)")
            for it in e["items"]:
                L.append(f"  - item `{it['item_id']}`: 保持 {it['holders_in_gathering']}人"
                         f"(share {it['share']}・直前{it['fresh_window_min']}分の新着 "
                         f"{it['fresh_holders']}人)/ 木の深さ {it['tree_depth']} hop / "
                         f"最大幅 {it['max_breadth']} / 経路 {','.join(it['channels'])}"
                         f" / 到達の中央値 {it['lead_min_median']} 分前")
    L += ["", "---", "",
          "**読み方の注意**: `via=congestion` は滞留(遅延・環境イベント)との時間相関で",
          "付けたラベルで、因果の証明ではない。`place_kind=label/category` の意図セルは",
          "ノードへ束縛していないので突合の対象外(推測で埋めない)。`n_present` は L1 から",
          "復元した同時在場のピークで、`arrive` の前方補完という仮定の上に立つ。",
          f"`short_by=0` でも不発でありうる: 検出は継続 {p['dur_bins']} ビンも要求するので、",
          "人数と比を 1 ビンだけ満たしても 1 件にはならない(その場合 `fail_reason` は",
          "`ratio`/`headcount` のうち最後に測った側になる)。種の系譜は「参加者が共有して",
          "いた item」であって「その item が集合を呼んだ」証明ではない(本文は読まない)。", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="集合イベントの事後検出(案A 集合検知 + 案B 意図突合・読み出し専用)")
    ap.add_argument("--run", default=None, help="ラン名 or パス(既定: 最新ラン)")
    ap.add_argument("--slot-min", type=int, default=DEF_SLOT_MIN,
                    help=f"時間帯ビンの幅(分・1440 の約数。既定 {DEF_SLOT_MIN})")
    ap.add_argument("--min-n", type=int, default=DEF_MIN_N,
                    help=f"集合とみなす同時在場の下限(既定 {DEF_MIN_N})")
    ap.add_argument("--ratio", type=float, default=DEF_RATIO,
                    help=f"ベースライン比の開始閾値(既定 {DEF_RATIO})")
    ap.add_argument("--ratio-end", type=float, default=None,
                    help="終了閾値(既定 = 開始閾値 × 0.6・ヒステリシス)")
    ap.add_argument("--dur", type=int, default=DEF_DUR,
                    help=f"何ビン続いたら 1 件とみなすか(既定 {DEF_DUR})")
    ap.add_argument("--baseline-floor", type=float, default=DEF_BASELINE_FLOOR,
                    help=f"ベースラインの下限(既定 {DEF_BASELINE_FLOOR})")
    ap.add_argument("--min-intent", type=int, default=2,
                    help="突合対象にする意図セルの下限人数(既定 2)")
    ap.add_argument("--match-slack", type=int, default=DEF_MATCH_SLACK,
                    help=f"意図 ⇄ 集合の突合で許すビンのずれ(既定 {DEF_MATCH_SLACK})")
    ap.add_argument("--fresh-min", type=int, default=DEF_FRESH_MIN,
                    help=f"種の鮮度窓(分。既定 {DEF_FRESH_MIN})")
    ap.add_argument("--genealogy-top", type=int, default=DEF_GENEALOGY_TOP,
                    help=f"種の系譜を作る集合の件数(既定 {DEF_GENEALOGY_TOP})")
    ap.add_argument("--top", type=int, default=DEF_TOP, help="レポート掲載件数")
    ap.add_argument("--out", default=None, help="出力先ディレクトリ(既定: ラン直下)")
    ap.add_argument("--quiet", action="store_true", help="コンソール出力を抑える")
    args = ap.parse_args(argv)

    run_dir = pick_run(args.run)
    res = analyze(run_dir, slot_min=args.slot_min, min_n=args.min_n,
                  ratio=args.ratio, ratio_end=args.ratio_end, dur=args.dur,
                  floor=args.baseline_floor, genealogy_top=args.genealogy_top,
                  min_intent=args.min_intent, match_slack=args.match_slack,
                  fresh_min=args.fresh_min)
    out_dir = args.out or run_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "gatherings.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    md_path = os.path.join(out_dir, "gatherings_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(res, args.top))
    if not args.quiet:
        s, cov = res["intent_summary"], res["coverage"]
        print(f"[gather] run={res['run']} nodes={cov['nodes_seen']} "
              f"steps={cov['steps_seen']} baseline={cov['baseline_src']}")
        print(f"[gather] gatherings={len(res['gatherings'])} "
              f"via={res['via_counts']}")
        print(f"[gather] intent(node) cells={s['n_cells_node']} "
              f"realized={s['n_realized']} unrealized={s['n_unrealized']} "
              f"median_short_by={s['median_short_by']} "
              f"src={cov['intent_source']}")
        for g in res["gatherings"][:8]:
            print(f"  day{g['day']} {g['clock']} {g['node']} n={g['n_peak']} "
                  f"base={g['baseline']} x{g['ratio']} via={g['via']} "
                  f"members={g.get('n_members', 0)}")
        print(f"[gather] -> {json_path}")
        print(f"[gather] -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
