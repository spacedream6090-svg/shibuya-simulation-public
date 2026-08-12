#!/usr/bin/env python
"""S-10 MAS 失敗様式(MAST 14 様式)の **L1 判定可能サブセット**(SV2-B・読み取り専用)。

    python scripts/analyze_mas_failures.py runs/<name>
    python scripts/analyze_mas_failures.py runs/<name> --out experiments/mas_failures

■ 正典
  Cemri, M., Pan, M. Z., Yang, S., Agrawal, L. A., Chopra, B., Tiwari, R., Keutzer, K.,
  Parameswaran, A., Klein, D., Ramchandran, K., Zaharia, M., Gonzalez, J. E. & Stoica, I.
  (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657.
  7 フレームワーク・200+ タスク・1,600+ 実行トレースを 6 名の専門アノテータ
  (**アノテータ間一致 κ = 0.88**)で Grounded Theory 分析した 14 失敗様式(FM)。

■ ★★ 適用前に必ず読む読み替え(これを書かずに 14 様式を当てると誤用になる)
  MAST は **タスク遂行型 MAS**(与えられたタスクを協働で解く系)のトレースから帰納された
  分類である。**本シムはタスクを与えていない社会シム**であり、MAST の言う
  「specification(仕様)」に相当するものが**外から与えられていない**。
  → **本シムにおける "仕様" は、エージェントが自分で言い出した目的**に限る:
     `group_found.purpose` / `proposal.text` / `day_plan.plan[].place`。
  この読み替えを書かずに機械適用すると「タスク型の分類を社会シムに機械適用した」と読まれる。

■ ★ 本スクリプトが出すのは「失敗率」ではなく「様式ごとの観測量」である
  MAST の割合(FM-1.3 = 17.14% 等)は**専門アノテータの人手判定**による値であり、
  本スクリプトの機械判定と**同じ量ではない**。並べて比較してはならない。
  実装した様式には `IMPLEMENTED` / 代理指標には `PROXY` を、
  判定できない様式には `NOT_IMPLEMENTED` と**その理由**を必ず付ける
  (リサーチ文書 §5-2:「△ を付けた件は妥当性を実データで確認するまで判定可能と書いてはならない」)。

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。凍結ファイル(`deviation.py` / `truth_ledger.py` /
  `analyze_specialization.py` 等)は**開かない**(既存装置がある様式は二重実装しない)。
  決定論: 走査順・出力キーはすべてソート順。語彙リテラルをコードに置かない
  (文字 n-gram のみ = `analyze_specialization.py` の設計と同じ掟)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import run_dt                      # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: どちらも **ラン依存**(run.dt_min)。ここの値は正準 Δt=10 の既定で、実際の値は
# analyze() が run dir から読む(Δt=10 なら 144 / 6 = 従来と 1 ビットも変わらない)。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY
RESPONSE_WINDOW_MIN = 60         # 「聞いたのに応じない」の猶予 [分](こちらが Δt 非依存の定義)
RESPONSE_WINDOW_STEPS = 6        # ↑を step にしたもの(Δt=10 で 6・Δt=1 で 60)
TURN_CAP = 12                    # 第 87 engaged のターン上限(FM-1.5 の代理閾値)
PREMATURE_DAYS = 3               # 発足から N 日以内に誰も加わらなければ「早すぎる終了」候補
NGRAM_N = 2                      # 文字 2-gram(語彙リテラルを置かない)

USED_KINDS = ("speak", "hear", "dm", "day_plan", "arrive", "enter_building",
              "group_found", "group_join", "joint_invite", "venture_open",
              "venture_close", "proposal")
# W4-D': `l1_stream` へ渡す kind 集合の**唯一の宣言**(中身は USED_KINDS と同一)。
# `tests/test_l1_stream_migration.py` が AST でモジュール内の kind リテラルとの
# 網羅を機械検査する = 様式を 1 つ足して kind を増やしたときの取りこぼしを防ぐ。
WANT_KINDS: frozenset = frozenset(USED_KINDS)

# 14 様式のうち **判定しないもの** と、その理由(理由を書くことが本スクリプトの本体である)
NOT_IMPLEMENTED = {
    "FM-1.1": ("Disobey task specification",
               "`parse_action` が有限語彙 `KNOWN_ACTIONS` で弾くため、"
               "**不正行為の試行そのものが L1 に残らない**。原理的に不可視であり、"
               "代理(`free_action` の逸脱率 / L2 `undefined_action_rate`)は別の量である。"),
    "FM-1.2": ("Disobey role specification",
               "`src/society/observer/deviation.py`(凍結)のペルソナ逸脱率が**既存装置**である。"
               "二重実装しない(凍結ファイルは開かない)。"),
    "FM-1.4": ("Loss of conversation history",
               "発話イベントに対話履歴の刻印が無く、「直近履歴を無視した」ことを"
               "L1 単体では同定できない。`analyze_firing.py` の発火連鎖グラフでの断絶検出が"
               "近いが、断絶と履歴喪失は別の量である。"),
    "FM-2.1": ("Conversation reset",
               "話題トークン列の連続性判定が要る。表層文字列一致では過小にも過大にも振れ、"
               "妥当性を実データで確認していない(リサーチ文書 §5-2 の △)。"),
    "FM-2.2": ("Fail to ask for clarification",
               "疑問形の検出が要り、表層一致は `norm_stage` と同じ弱さを持つ。"
               "最も近い装置は IF-B の `reject_line`(拒否理由の可視化)であって本スクリプトではない。"),
    "FM-2.4": ("Information withholding",
               "`truth_ledger`(凍結)の `beliefs_of` が要る。既定 OFF のランには存在しない。"
               "読むだけなら可だが、**存在しないランで 0 と書くのは捏造**なので判定しない。"),
    "FM-3.2": ("No or incomplete verification",
               "verify モードの `truth_ledger` が要る(既定 OFF)。同上。"),
    "FM-3.3": ("Incorrect verification",
               "`truth_ledger` の `verified=True` と真値の突合が要る(既定 OFF)。同上。"),
}


# --------------------------------------------------------------------------- #
# 純関数(単体テストの対象)
# --------------------------------------------------------------------------- #
def _r(x, nd: int = 6):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def char_ngrams(text: str, n: int = NGRAM_N) -> dict:
    """文字 n-gram のカウント。**語彙リテラルをコードに置かない**ための切り方。"""
    s = "".join(str(text).split())
    if len(s) < n:
        return {}
    out: dict[str, int] = defaultdict(int)
    for i in range(len(s) - n + 1):
        out[s[i:i + n]] += 1
    return dict(out)


def js_divergence(a: dict, b: dict) -> float | None:
    """Jensen–Shannon divergence(底 2・0〜1)。どちらかが空なら None。"""
    if not a or not b:
        return None
    ta, tb = sum(a.values()), sum(b.values())
    if ta <= 0 or tb <= 0:
        return None
    keys = sorted(set(a) | set(b))
    out = 0.0
    for k in keys:
        p = a.get(k, 0) / ta
        q = b.get(k, 0) / tb
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log2(p / m)
        if q > 0:
            out += 0.5 * q * math.log2(q / m)
    return max(0.0, min(1.0, out))


def fm13_step_repetition(events: list[dict]) -> dict:
    """FM-1.3 Step repetition — **同一エージェントが直前と同一の発話を繰り返した割合**。

    MAST 最頻の様式(人手判定 17.14%)。本シムには `observer.echo`(L2 常設 5 列)と
    `reg_ngram_repeat_rate` という既存装置があるので、本欄は**それを置き換えるものではなく
    L1 側からの独立な確認**である。
    """
    last: dict[int, str] = {}
    n_speak = n_rep = 0
    reps_by_agent: dict[int, int] = defaultdict(int)
    for e in events:
        if e["kind"] != "speak":
            continue
        aid = e["agent"]
        txt = str((e["payload"] or {}).get("text", "") or "")
        if not txt:
            continue
        n_speak += 1
        if last.get(aid) == txt:
            n_rep += 1
            reps_by_agent[aid] += 1
        last[aid] = txt
    return {"status": "IMPLEMENTED", "n_speak": n_speak, "n_immediate_repeat": n_rep,
            "rate": _r(n_rep / n_speak, 4) if n_speak else None,
            "n_agents_with_repeat": len(reps_by_agent)}


def fm15_termination_unaware(events: list[dict], turn_cap: int = TURN_CAP,
                             spd: int = STEPS_PER_DAY) -> dict:
    """FM-1.5 Unaware of termination conditions —(**代理**)同日・同一ペアの往復が上限に達した件数。

    根拠は第 87 バッチ `engaged` のターン上限 12(CAMEL の無限 goodbye ループが由来)。
    ★ここで数えているのは「往復が長い」ことであって「終了条件を認識していない」ことではない。
    したがって `PROXY` を外さない。
    """
    turns: dict[tuple, int] = defaultdict(int)
    for e in events:
        if e["kind"] != "speak":
            continue
        aid = e["agent"]
        day = e["step"] // int(spd)
        for h in (e["payload"] or {}).get("hearers", []) or []:
            if isinstance(h, int) and h >= 0 and h != aid:
                key = (day, min(aid, h), max(aid, h))
                turns[key] += 1
    over = sorted(k for k, v in turns.items() if v >= turn_cap)
    return {"status": "PROXY", "turn_cap": int(turn_cap),
            "n_pairs_day": len(turns), "n_over_cap": len(over),
            "rate": _r(len(over) / len(turns), 4) if turns else None,
            "max_turns": (max(turns.values()) if turns else 0)}


def fm25_ignored_input(events: list[dict], window: int = RESPONSE_WINDOW_STEPS) -> dict:
    """FM-2.5 Ignored other agent's input — 聞き手になったのに窓内で何も発話しない割合 +
    `joint_invite` の無応答(verdict == "none")割合。"""
    speak_steps: dict[int, list[int]] = defaultdict(list)
    addressed: list[tuple] = []
    inv_total = inv_none = 0
    for e in events:
        k = e["kind"]
        if k == "speak":
            speak_steps[e["agent"]].append(e["step"])
            for h in (e["payload"] or {}).get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != e["agent"]:
                    addressed.append((h, e["step"]))
        elif k == "joint_invite":
            inv_total += 1
            if str((e["payload"] or {}).get("verdict", "")) == "none":
                inv_none += 1
    for a in speak_steps:
        speak_steps[a].sort()
    import bisect
    ignored = 0
    for (h, st) in addressed:
        arr = speak_steps.get(h, [])
        i = bisect.bisect_left(arr, st)
        if i >= len(arr) or arr[i] > st + window:
            ignored += 1
    return {"status": "IMPLEMENTED", "window_steps": int(window),
            "n_addressed": len(addressed), "n_no_response": int(ignored),
            "rate_no_response": _r(ignored / len(addressed), 4) if addressed else None,
            "n_joint_invite": inv_total, "n_invite_verdict_none": inv_none,
            "rate_invite_none": _r(inv_none / inv_total, 4) if inv_total else None}


def fm26_reasoning_action_mismatch(events: list[dict],
                                   min_vocab_overlap: float = 0.05,
                                   spd: int = STEPS_PER_DAY) -> dict:
    """FM-2.6 Reasoning-action mismatch — 朝の `day_plan` の place にその日到達しなかった割合。

    ★**本シムでの "仕様" は自分で言い出した目的**(冒頭の読み替え)。
    `place` が空文字の計画項目は「場所を宣言していない」= 検証不能なので**分母から外す**
    (0 として数えない)。外した件数も報告する。

    ★★ **名前空間の突合可能性を必ず検算する**(実測で判明した落とし穴):
    `day_plan.plan[].place` は **LLM が書いた自由文**(例「渋谷のカフェ」「自宅」)であり、
    `arrive.name` は **地図ノード名**(例「路上」「渋谷駅ハチ公口」)である。
    両者の語彙がほとんど重ならないランでは、本指標が測っているのは
    **推論と行動の乖離ではなく文字列の不一致**である。
    → 語彙の重なりが `min_vocab_overlap` 未満なら `status` を **`NAME_SPACE_MISMATCH`** にし、
    **率を「失敗率」として読ませない**。この検算を入れないと 97% のような数字を
    そのまま「推論と行動が乖離している」と誤読することになる。
    """
    planned: dict[tuple, set] = defaultdict(set)
    visited: dict[tuple, set] = defaultdict(set)
    place_vocab: set[str] = set()
    visit_vocab: set[str] = set()
    n_items = n_blank = 0
    for e in events:
        k, aid, p = e["kind"], e["agent"], (e["payload"] or {})
        day = e["step"] // int(spd)
        if k == "day_plan":
            for item in p.get("plan", []) or []:
                if not isinstance(item, dict):
                    continue
                n_items += 1
                place = str(item.get("place", "") or "").strip()
                if not place:
                    n_blank += 1
                    continue
                planned[(aid, day)].add(place)
                place_vocab.add(place)
        elif k in ("arrive", "enter_building"):
            nm = str(p.get("name", "") or "").strip()
            if nm:
                visited[(aid, day)].add(nm)
                visit_vocab.add(nm)
    total = miss = 0
    for key in sorted(planned):
        want = planned[key]
        got = visited.get(key, set())
        total += len(want)
        miss += len(want - got)
    overlap = place_vocab & visit_vocab
    frac = (len(overlap) / len(place_vocab)) if place_vocab else None
    status = "IMPLEMENTED"
    note = ""
    if frac is not None and frac < min_vocab_overlap:
        status = "NAME_SPACE_MISMATCH"
        note = ("計画の place(LLM の自由文)と到達名(地図ノード名)の語彙が"
                f"ほとんど重ならない(重なり {len(overlap)}/{len(place_vocab)})。"
                "この率は**推論と行動の乖離ではなく文字列の不一致**を測っている。"
                "失敗率として読んではならない。")
    return {"status": status, "n_plan_items": n_items,
            "n_plan_items_without_place": n_blank,
            "n_verifiable_places": total, "n_not_reached": miss,
            "rate": _r(miss / total, 4) if total else None,
            "n_place_vocab": len(place_vocab), "n_visited_vocab": len(visit_vocab),
            "n_vocab_overlap": len(overlap), "vocab_overlap_frac": _r(frac, 4),
            "min_vocab_overlap": float(min_vocab_overlap),
            "note": note}


def fm23_task_derailment(events: list[dict], n: int = NGRAM_N) -> dict:
    """FM-2.3 Task derailment — 集団の目的語と、加入後のメンバー発話の **JS ダイバージェンス**。

    ★**仕様 = `group_found.purpose`**(自分で言い出した目的)。値が大きいほど「目的から遠い」。
    絶対の閾値は置かない(閾値は事前登録に書いてから使う)。**分布を出すだけ**である。
    """
    purpose: dict[int, str] = {}
    found_step: dict[int, int] = {}
    members: dict[int, set] = defaultdict(set)
    join_step: dict[tuple, int] = {}
    for e in events:
        k, aid, p = e["kind"], e["agent"], (e["payload"] or {})
        if k == "group_found":
            gid = p.get("group_id")
            if gid is None:
                continue
            gid = int(gid)
            purpose[gid] = f"{p.get('purpose','') or ''}{p.get('name','') or ''}"
            found_step.setdefault(gid, e["step"])
            members[gid].add(aid)
            join_step.setdefault((gid, aid), e["step"])
        elif k == "group_join":
            gid = p.get("group_id")
            if gid is None:
                continue
            gid = int(gid)
            members[gid].add(aid)
            join_step.setdefault((gid, aid), e["step"])
    if not purpose:
        return {"status": "NO_GROUPS", "n_groups": 0, "divergences": {}}
    member_of: dict[int, set] = defaultdict(set)
    for gid, ms in members.items():
        for a in ms:
            member_of[a].add(gid)
    said: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    for e in events:
        if e["kind"] != "speak":
            continue
        aid = e["agent"]
        for gid in sorted(member_of.get(aid, ())):
            if gid not in purpose or e["step"] < join_step.get((gid, aid), 0):
                continue
            for g, c in char_ngrams((e["payload"] or {}).get("text", ""), n).items():
                said[gid][g] += c
    divs = {}
    for gid in sorted(purpose):
        d = js_divergence(char_ngrams(purpose[gid], n), dict(said.get(gid, {})))
        if d is not None:
            divs[str(gid)] = _r(d, 4)
    vals = sorted(divs.values())
    med = None
    if vals:
        m = len(vals)
        med = _r(vals[m // 2] if m % 2 else 0.5 * (vals[m // 2 - 1] + vals[m // 2]), 4)
    return {"status": "IMPLEMENTED", "n_groups": len(purpose),
            "n_groups_scored": len(divs), "js_median": med,
            "js_min": (vals[0] if vals else None),
            "js_max": (vals[-1] if vals else None),
            "divergences": divs,
            "note": "閾値は置かない(事前登録に書いてから使う)。分布のみを報告する。"}


def fm31_premature_termination(events: list[dict],
                               premature_days: int = PREMATURE_DAYS,
                               spd: int = STEPS_PER_DAY) -> dict:
    """FM-3.1 Premature termination — 目的未達で終わった集団/事業の件数。

    (a) `group_found` してから premature_days 日以内に **誰も加わらなかった**集団
    (b) `venture_open` から premature_days 日以内の `venture_close`
    ★これは「早く終わった」であって「目的が未達だった」ことの直接の証拠ではない。
    """
    found: dict[int, int] = {}
    joins: dict[int, list[int]] = defaultdict(list)
    v_open: dict[int, int] = {}
    v_close: list[tuple] = []
    for e in events:
        k, aid, p = e["kind"], e["agent"], (e["payload"] or {})
        if k == "group_found" and p.get("group_id") is not None:
            found.setdefault(int(p["group_id"]), e["step"])
        elif k == "group_join" and p.get("group_id") is not None:
            joins[int(p["group_id"])].append(e["step"])
        elif k == "venture_open":
            v_open.setdefault(aid, e["step"])
        elif k == "venture_close":
            v_close.append((aid, e["step"]))
    cutoff = premature_days * int(spd)
    lonely = sorted(g for g, st in found.items()
                    if not any(s - st <= cutoff for s in joins.get(g, [])))
    early_close = sorted(a for (a, st) in v_close
                         if a in v_open and st - v_open[a] <= cutoff)
    return {"status": "IMPLEMENTED", "premature_days": int(premature_days),
            "n_groups_founded": len(found),
            "n_groups_no_join_within_window": len(lonely),
            "rate_groups": _r(len(lonely) / len(found), 4) if found else None,
            "n_ventures_opened": len(v_open),
            "n_ventures_closed_early": len(early_close),
            "rate_ventures": _r(len(early_close) / len(v_open), 4) if v_open else None}


# --------------------------------------------------------------------------- #
# 解析本体
# --------------------------------------------------------------------------- #
def _load_events(run_dir: str) -> list[dict]:
    """`USED_KINDS` だけを読む(W4-D': 共有の `l1_stream` 経由)。

    旧実装は `to_pydict()` で *全行* を Python の list にしてから捨てていた。
    `l1_stream` は **kind を Arrow 上で絞ってから**実体化するので、捨てる行の
    Python オブジェクト生成費が 0 になる。あわせて **part 群の透過連結**と
    **Windows の共有フラグ付き open**(第77バッチの規約)も手に入る。
    実体化ループと**最後の並べ替えは旧実装の逐語**(返り値は 1 バイトも変わらない)。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        return []
    out: list[dict] = []
    for d in ls.iter_columns(run_dir, ["step", "agent_id", "kind", "payload"],
                             kinds=WANT_KINDS):
        for st, aid, kind, raw in zip(d["step"], d["agent_id"], d["kind"], d["payload"]):
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({"step": int(st), "agent": int(aid), "kind": kind,
                        "payload": payload})
    out.sort(key=lambda e: (e["step"], e["agent"], e["kind"]))
    return out


def response_window_steps(run_dir: str) -> int:
    """「聞いたのに応じない」の猶予 [step] = 実時間 RESPONSE_WINDOW_MIN 分ぶん。"""
    return max(1, RESPONSE_WINDOW_MIN // run_dt.min_per_step(run_dir))


def analyze(run_dir: str, *, window: int | None = None,
            turn_cap: int = TURN_CAP, premature_days: int = PREMATURE_DAYS) -> dict:
    """W2-3: `window=None`(既定)はこのランの Δt から実時間 60 分ぶんの step 数を導く
    (Δt=10 なら 6 = 従来の既定と同値)。明示指定があればそちらを尊重する。"""
    # 1 日の step 数もランの Δt に従う(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    if window is None:
        window = max(1, RESPONSE_WINDOW_MIN // run_dt.min_per_step(run_dir))
    events = _load_events(run_dir)
    res: dict = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "spec": {"taxonomy": "MAST (Cemri et al. 2025, arXiv:2503.13657)",
                 "specification_in_this_sim":
                     "エージェントが自分で言い出した目的のみ"
                     "(group_found.purpose / proposal.text / day_plan.plan[].place)",
                 "caveat": "MAST の割合は人手判定。本スクリプトの機械判定と同じ量ではない",
                 "response_window_steps": int(window), "turn_cap": int(turn_cap),
                 "premature_days": int(premature_days)},
        "n_events_used": len(events),
    }
    if not events:
        res["status"] = "NO_DATA"
        res["modes"] = {}
        return res
    res["status"] = "OK"
    res["modes"] = {
        "FM-1.3": {"name": "Step repetition", "fc": "FC1",
                   **fm13_step_repetition(events)},
        "FM-1.5": {"name": "Unaware of termination conditions", "fc": "FC1",
                   **fm15_termination_unaware(events, turn_cap, spd)},
        "FM-2.3": {"name": "Task derailment", "fc": "FC2",
                   **fm23_task_derailment(events)},
        "FM-2.5": {"name": "Ignored other agent's input", "fc": "FC2",
                   **fm25_ignored_input(events, window)},
        "FM-2.6": {"name": "Reasoning-action mismatch", "fc": "FC2",
                   **fm26_reasoning_action_mismatch(events, spd=spd)},
        "FM-3.1": {"name": "Premature termination", "fc": "FC3",
                   **fm31_premature_termination(events, premature_days, spd)},
    }
    res["not_implemented"] = {fm: {"name": nm, "reason": why}
                              for fm, (nm, why) in sorted(NOT_IMPLEMENTED.items())}
    res["coverage"] = {"n_total_modes": 14,
                       "n_scored": len(res["modes"]),
                       "n_not_implemented": len(NOT_IMPLEMENTED)}
    return res


def render(res: dict) -> str:
    L: list[str] = []
    L.append(f"# S-10 MAS 失敗様式(MAST)の L1 判定可能サブセット — `{res.get('run','?')}`")
    L.append("")
    L.append("> **MAST はタスク遂行型 MAS から帰納された分類**であり、本シムはタスクを与えない"
             "社会シムである。本シムでの \"仕様\" は**エージェントが自分で言い出した目的**"
             "(`group_found.purpose` / `proposal.text` / `day_plan.plan[].place`)に限る。")
    L.append("> **MAST の割合(人手判定・κ=0.88)と本スクリプトの機械判定は同じ量ではない。**"
             "並べて比較してはならない。")
    L.append("")
    if res.get("status") != "OK":
        L.append("**NO_DATA** — 判定材料のイベントが無い。捏造しない。")
        return "\n".join(L) + "\n"
    cov = res["coverage"]
    L.append(f"- 使用イベント: {res['n_events_used']}")
    L.append(f"- 判定した様式: **{cov['n_scored']} / {cov['n_total_modes']}**"
             f"(判定しない: {cov['n_not_implemented']} 件・理由は下表)")
    L.append("")
    L.append("## 判定した様式")
    L.append("")
    L.append("| FM | 群 | 名称 | 状態 | 主要量 |")
    L.append("|---|---|---|---|---|")
    for fm in sorted(res["modes"]):
        mm = res["modes"][fm]
        if fm == "FM-1.3":
            v = f"直前反復率 {mm['rate']}({mm['n_immediate_repeat']}/{mm['n_speak']})"
        elif fm == "FM-1.5":
            v = (f"往復 ≥{mm['turn_cap']} のペア日 {mm['n_over_cap']}/{mm['n_pairs_day']} "
                 f"(最大 {mm['max_turns']} 往復)")
        elif fm == "FM-2.3":
            v = (f"目的 vs 発話 JS 中央値 {mm.get('js_median')} "
                 f"[{mm.get('js_min')}, {mm.get('js_max')}] / 集団 {mm.get('n_groups')}")
        elif fm == "FM-2.5":
            v = (f"無応答率 {mm['rate_no_response']}"
                 f"({mm['n_no_response']}/{mm['n_addressed']}) / "
                 f"招待無応答率 {mm['rate_invite_none']}")
        elif fm == "FM-2.6":
            v = (f"計画place未到達率 {mm['rate']}"
                 f"({mm['n_not_reached']}/{mm['n_verifiable_places']}"
                 f"・place空 {mm['n_plan_items_without_place']} 件は分母外)"
                 f"・語彙重なり {mm['n_vocab_overlap']}/{mm['n_place_vocab']}"
                 f"({mm['vocab_overlap_frac']})")
        elif fm == "FM-3.1":
            v = (f"加入0の集団 {mm['n_groups_no_join_within_window']}/"
                 f"{mm['n_groups_founded']}(率 {mm['rate_groups']}) / "
                 f"早期閉店 {mm['n_ventures_closed_early']}/{mm['n_ventures_opened']}")
        else:
            v = "—"
        L.append(f"| **{fm}** | {mm['fc']} | {mm['name']} | `{mm['status']}` | {v} |")
    L.append("")
    L.append("## ★判定しない様式と、その理由(これを書くのが本スクリプトの本体)")
    L.append("")
    L.append("| FM | 名称 | 判定しない理由 |")
    L.append("|---|---|---|")
    for fm, rec in sorted(res["not_implemented"].items()):
        L.append(f"| {fm} | {rec['name']} | {rec['reason']} |")
    L.append("")
    bad = [fm for fm, mm in sorted(res["modes"].items())
           if mm.get("status") == "NAME_SPACE_MISMATCH"]
    if bad:
        L.append("## ★★ 測定の妥当性が壊れている様式(率を失敗率として読んではならない)")
        L.append("")
        for fm in bad:
            L.append(f"- **{fm}** … {res['modes'][fm].get('note')}")
        L.append("")
    L.append("## 読み方の注意")
    L.append("")
    L.append("- `PROXY` の様式(FM-1.5)は**代理指標**である。「往復が長い」ことは観測しているが、")
    L.append("  「終了条件を認識していない」ことは観測していない。")
    L.append("- `NAME_SPACE_MISMATCH` の様式は、**測っている対象が名前空間の不一致にすり替わっている**。")
    L.append("  数値を報告に載せる場合は、必ずこのラベルごと載せる。")
    L.append("- FM-2.3 の JS ダイバージェンスに**閾値を置いていない**。閾値は事前登録に書いてから使う。")
    L.append("- 「失敗が少ない」ことを、本シムの優秀さの証拠に使ってはならない"
             "(判定できない様式が 8 件ある)。")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-10 MAST 失敗様式(読み取り専用)")
    ap.add_argument("run", help="ランディレクトリ")
    ap.add_argument("--window", type=int, default=None,
                    help=f"「応じない」の猶予 [step](既定=実時間 {RESPONSE_WINDOW_MIN} 分ぶん)")
    ap.add_argument("--turn-cap", type=int, default=TURN_CAP)
    ap.add_argument("--premature-days", type=int, default=PREMATURE_DAYS)
    ap.add_argument("--out", default=None, help="出力先(既定 <run>/analysis)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.run):
        print(f"[analyze_mas_failures] ランが無い: {a.run}", file=sys.stderr)
        return 2
    res = analyze(a.run, window=a.window, turn_cap=a.turn_cap,
                  premature_days=a.premature_days)
    out = a.out or os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "mas_failures.json")
    mp = os.path.join(out, "mas_failures.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    print(f"[analyze_mas_failures] {res.get('status')} "
          f"scored={res.get('coverage',{}).get('n_scored')}/14")
    print(f"  -> {jp}")
    print(f"  -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
