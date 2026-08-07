#!/usr/bin/env python
"""第20バッチ: エージェントの主観的世界モデルの後処理・観察 CLI(読み出し専用)。

    python scripts/analyze_worldview.py runs/<name> [--out runs/<name>/worldview_report.md]

ユーザー要望(2026-07-12): 「各エージェントには世界を各々の解釈で捉えた世界があり、
仮説を立て検証しながら世界を知り、行動して変えていく。エージェントが世界をどう捉え、
事象をどう理解するのか(=その世界解釈・考え方)を観察できるようにしてほしい」。

シム側(src/society/worldview.py, 既定 OFF)が日次で L1 "worldview" イベントを出す:
  agent 行 payload = {ctrl, expect_n, err_mean, err_n}
  街 行(agent_id=-1)payload = {norm_rate, pioneer_1d}
本スクリプトは L1(l1_events.parquet)と agents.json だけを読み、以下を出す。

  1. 世界解釈パネル      : agent×日の {ctrl, expect_n, err_mean} + 街 norm_rate
                          → runs/<name>/panel/worldview.parquet(pyarrow・pandas不使用)
  2. 仮説検証ループ      : err_mean 日次系列(個人+全体)・収束(前半vs後半)・学習の速い/遅い
  3. 可制御性の分岐      : 0.5 起点からの軌跡・日次分散(分岐の速度)・最終分布・行動との相関
  4. 解釈の分岐          : 共有事象(悪天/災害/提案成立/広告改定)の前後24hの発話 valence 差
  5. 信念の世界観クラスタ : reflect の belief 書き戻しを文字3-gram コサインで決定論クラスタ化
  6. 世界観カード        : 「この人は世界をこう見ている」を1段落で(上位6/下位3+付録)

方針(誠実性): 値はすべてログ由来。該当イベントが無い指標は「データ不足」と正直に出す。
worldview イベント 0 件のラン(OFF ラン)は「worldview OFF ラン」と明記し主要部をスキップ。
決定論: ソート順固定・乱数ゼロ。感情価は society.lang.sentiment.valence(決定論)を使う。
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                           # l1_stream(共有の逐次読み)用
    sys.path.insert(0, _HERE)

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from society.observer import measure as m            # noqa: E402  (char_ngrams 等を流用)
from society.lang.sentiment import valence           # noqa: E402  (決定論の感情価)

MIN_PER_DAY = 1440

# C6/可制御性の分岐で「開拓的行動」とみなす kind(worldview.py の _PIONEER_KINDS と一致)。
# いずれも agent_id = 行為者。
_PIONEER_KINDS = ("venture_open", "proposal", "event_host", "group_found")

# W4-D: 本解析が読む kind(5 用途の抽出条件を全数列挙したもの)。
#   ① 世界観パネル = worldview
#   ② 開拓的行動   = _PIONEER_KINDS + free_action
#   ③ 共有事象     = weather / disaster / proposal_passed / ad_campaign
#   ④ 発話 valence = speak / sns_post
#   ⑤ 信念テキスト = reflect
# ここに無い kind はどの抽出も `!=` / `not in` で必ず読み飛ばすので、絞っても不変。
# 唯一の全 kind 依存は n_agents のフォールバックで、そちらは agent_id 列走査へ。
# ② は上の定数を **写経せず合成する**。
WANT_KINDS = frozenset(_PIONEER_KINDS) | {
    "worldview", "free_action",
    "weather", "disaster", "proposal_passed", "ad_campaign",
    "speak", "sns_post", "reflect",
}
# 悪天(共有事象)とみなす天気(weather の cond は 晴/曇/雨/雪 の4値)。
_BAD_WEATHER = {"雨", "雪"}
# belief 世界観クラスタの類似閾値(文字3-gram コサイン。>= で同クラスタ)。
_CLUSTER_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# 小道具(決定論の純関数)
# --------------------------------------------------------------------------- #
def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson 相関係数(標準ライブラリのみ)。分散0/n<2 は None。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return round(sxy / math.sqrt(sxx * syy), 4)


def _pvar(values: list[float]) -> float:
    """母分散(1点以下は 0.0)。"""
    vs = [float(v) for v in values if v is not None]
    if len(vs) < 2:
        return 0.0
    return round(st.pvariance(vs), 6)


def _day_of(sim_min: int) -> int:
    return int(sim_min) // MIN_PER_DAY


# --------------------------------------------------------------------------- #
# worldview イベントの分解
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """L1 を `WANT_KINDS` だけ読み、`measure.load_events` と同一形の dict 列で返す。

    残る比例項: 世界観行(agent × 日)と発話テキストの行数。どちらも
    「全期間の系列」「共有事象の前後 24h」を測る定義なので畳めない。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):        # m.load_events と同じく「無ければ落とす」
        raise FileNotFoundError(os.path.join(run_dir, "l1_events.parquet"))
    return list(ls.iter_events(run_dir, kinds=WANT_KINDS))


def split_worldview(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """L1 から worldview イベントを (agent 行, 街 行) に分ける。day = sim_min//1440。"""
    agent_rows: list[dict] = []
    city_rows: list[dict] = []
    for e in events:
        if e["kind"] != "worldview":
            continue
        p = e["payload"]
        day = _day_of(e["sim_min"])
        if e["agent_id"] == -1:
            city_rows.append({"day": day, "sim_min": int(e["sim_min"]),
                              "norm_rate": p.get("norm_rate"),
                              "pioneer_1d": p.get("pioneer_1d")})
        elif e["agent_id"] >= 0:
            agent_rows.append({"agent_id": int(e["agent_id"]), "day": day,
                               "sim_min": int(e["sim_min"]),
                               "ctrl": p.get("ctrl"), "expect_n": p.get("expect_n"),
                               "err_mean": p.get("err_mean"), "err_n": p.get("err_n")})
    return agent_rows, city_rows


# --------------------------------------------------------------------------- #
# 1. 世界解釈パネル(pyarrow。pandas 不使用)
# --------------------------------------------------------------------------- #
_PANEL_COLS = ("run", "agent_id", "day", "ctrl", "expect_n", "err_mean",
               "err_n", "norm_rate")


def _panel_schema():
    import pyarrow as pa
    f = pa.field
    return pa.schema([
        f("run", pa.string()), f("agent_id", pa.int64()), f("day", pa.int64()),
        f("ctrl", pa.float64()), f("expect_n", pa.int64()),
        f("err_mean", pa.float64()), f("err_n", pa.int64()),
        f("norm_rate", pa.float64()),
    ])


def build_panel(run_name: str, agent_rows: list[dict], city_rows: list[dict],
                out_dir: str) -> tuple[str, int]:
    """agent×日 の主観状態 + 街の norm_rate を parquet 化。行数を返す。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    norm_by_day = {r["day"]: r["norm_rate"] for r in city_rows}
    cols: dict[str, list] = {c: [] for c in _PANEL_COLS}
    for r in sorted(agent_rows, key=lambda x: (x["agent_id"], x["day"])):
        cols["run"].append(run_name)
        cols["agent_id"].append(int(r["agent_id"]))
        cols["day"].append(int(r["day"]))
        cols["ctrl"].append(None if r["ctrl"] is None else float(r["ctrl"]))
        cols["expect_n"].append(None if r["expect_n"] is None else int(r["expect_n"]))
        cols["err_mean"].append(None if r["err_mean"] is None else float(r["err_mean"]))
        cols["err_n"].append(None if r["err_n"] is None else int(r["err_n"]))
        nr = norm_by_day.get(r["day"])
        cols["norm_rate"].append(None if nr is None else float(nr))
    path = os.path.join(out_dir, "worldview.parquet")
    pq.write_table(pa.Table.from_pydict(cols, schema=_panel_schema()), path)
    return path, len(cols["day"])


# --------------------------------------------------------------------------- #
# 2. 仮説検証ループ(世界を知る速度)
# --------------------------------------------------------------------------- #
def learning_analysis(agent_rows: list[dict]) -> dict:
    """err_mean の日次系列(個人+全体)と収束(前半 vs 後半)を集計する。"""
    by_agent: dict[int, list[tuple[int, float]]] = defaultdict(list)
    by_day: dict[int, list[float]] = defaultdict(list)
    for r in agent_rows:
        if r["err_mean"] is not None:
            by_agent[r["agent_id"]].append((r["day"], float(r["err_mean"])))
            by_day[r["day"]].append(float(r["err_mean"]))
    overall = [(d, round(sum(v) / len(v), 4)) for d, v in sorted(by_day.items())]

    # 全体の収束: 誤差系列を前半/後半に割り、平均の低下率。
    days = [d for d, _ in overall]
    vals = [v for _, v in overall]
    conv_overall = None
    if len(vals) >= 2:
        half = len(vals) // 2
        first = st.mean(vals[:half]) if half else vals[0]
        second = st.mean(vals[half:])
        conv_overall = round((first - second) / first, 4) if first > 0 else None

    # 個人ごとの低下量(前半平均 - 後半平均)= 世界を学ぶ速さの proxy。
    per_agent: list[dict] = []
    for aid in sorted(by_agent):
        series = sorted(by_agent[aid])
        errs = [v for _, v in series]
        if len(errs) < 2:
            continue
        half = len(errs) // 2
        first = st.mean(errs[:half]) if half else errs[0]
        second = st.mean(errs[half:])
        per_agent.append({
            "agent_id": aid, "err_first": round(first, 4),
            "err_last": round(second, 4), "drop": round(first - second, 4),
            "n": len(errs)})
    # 学習の速い/遅い(低下量 desc / asc)。決定論 tie-break: agent_id。
    fast = sorted(per_agent, key=lambda x: (-x["drop"], x["agent_id"]))
    slow = sorted(per_agent, key=lambda x: (x["drop"], x["agent_id"]))
    return {"overall_series": overall, "conv_overall": conv_overall,
            "per_agent": per_agent, "fast": fast[:5], "slow": slow[:5],
            "n_agents_with_series": len(per_agent)}


# --------------------------------------------------------------------------- #
# 3. 可制御性の分岐(経路依存の定量)
# --------------------------------------------------------------------------- #
def _pioneer_and_free_counts(events: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    """agent ごとの開拓的行動回数(本人件数)と free_action 回数。"""
    pioneer: Counter = Counter()
    free: Counter = Counter()
    for e in events:
        aid = e["agent_id"]
        if not isinstance(aid, int) or aid < 0:
            continue
        if e["kind"] in _PIONEER_KINDS:
            pioneer[aid] += 1
        elif e["kind"] == "free_action":
            free[aid] += 1
    return dict(pioneer), dict(free)


def ctrl_analysis(agent_rows: list[dict], pioneer: dict[int, int],
                  free: dict[int, int]) -> dict:
    """全員 0.5 起点からの ctrl 軌跡・日次分散(分岐の速度)・最終分布・行動相関。"""
    by_day: dict[int, dict[int, float]] = defaultdict(dict)
    for r in agent_rows:
        if r["ctrl"] is not None:
            by_day[r["day"]][r["agent_id"]] = float(r["ctrl"])
    days = sorted(by_day)
    var_series = [(d, _pvar(list(by_day[d].values()))) for d in days]
    if not days:
        return {"days": [], "var_series": [], "final": {}, "top": [], "bottom": [],
                "corr_pioneer": None, "corr_free": None, "n": 0, "table": [],
                "diverged": False}
    final = by_day[days[-1]]                             # {aid: ctrl 終値}
    ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:5]
    bottom = sorted(final.items(), key=lambda kv: (kv[1], kv[0]))[:5]

    # 行動との相関(ctrl 終値 × 開拓的行動 / free_action)。n が小さければ「参考値」。
    ids = sorted(final)
    ctrls = [final[i] for i in ids]
    pio = [float(pioneer.get(i, 0)) for i in ids]
    fre = [float(free.get(i, 0)) for i in ids]
    corr_pioneer = _pearson(ctrls, pio)
    corr_free = _pearson(ctrls, fre)
    table = [{"agent_id": i, "ctrl": round(final[i], 4),
              "pioneer": pioneer.get(i, 0), "free_action": free.get(i, 0)}
             for i in ids]
    diverged = var_series[-1][1] > 1e-6 if var_series else False
    return {"days": days, "var_series": var_series, "final": final,
            "top": top, "bottom": bottom, "corr_pioneer": corr_pioneer,
            "corr_free": corr_free, "n": len(ids), "table": table,
            "diverged": diverged}


# --------------------------------------------------------------------------- #
# 4. 解釈の分岐(同じ世界・違う理解)
# --------------------------------------------------------------------------- #
def _shared_events(events: list[dict]) -> list[dict]:
    """全員が共有する事象(悪天/災害/提案成立/広告改定)を時刻付きで列挙する。"""
    out: list[dict] = []
    for e in events:
        k, p = e["kind"], e["payload"]
        if k == "weather" and (p.get("cond") in _BAD_WEATHER):
            out.append({"sim_min": int(e["sim_min"]), "kind": "悪天",
                        "label": f"悪天({p.get('cond')} {p.get('date', '')})".strip()})
        elif k == "disaster":
            out.append({"sim_min": int(e["sim_min"]), "kind": "災害",
                        "label": f"災害({p.get('kind')}/{p.get('phase')})"})
        elif k == "proposal_passed":
            out.append({"sim_min": int(e["sim_min"]), "kind": "提案成立",
                        "label": f"提案成立(#{p.get('proposal_id')})"})
        elif k == "ad_campaign":
            out.append({"sim_min": int(e["sim_min"]), "kind": "広告改定",
                        "label": f"広告改定({p.get('target')})"})
    return sorted(out, key=lambda x: (x["sim_min"], x["label"]))


def _utterances(events: list[dict]) -> tuple[list[int], list[tuple[int, int, str, float]]]:
    """speak/sns_post の発話を (sim_min, agent_id, text, valence) で時刻順に並べる。"""
    utt: list[tuple[int, int, str, float]] = []
    for e in events:
        if e["kind"] in ("speak", "sns_post"):
            aid = e["agent_id"]
            if not isinstance(aid, int) or aid < 0:
                continue
            text = e["payload"].get("text") or ""
            if text:
                utt.append((int(e["sim_min"]), aid, text, valence(text)))
    utt.sort(key=lambda x: (x[0], x[1]))
    times = [u[0] for u in utt]
    return times, utt


def interpretation_analysis(events: list[dict]) -> dict:
    """共有事象ごとに、前後24hの発話 valence の個体差を集計する。"""
    shared = _shared_events(events)
    if not shared:
        return {"has_events": False, "events": []}
    times, utt = _utterances(events)

    def window(lo: int, hi: int) -> dict[int, list[float]]:
        """[lo, hi) の発話を agent 別に valence リストへ。"""
        i = bisect.bisect_left(times, lo)
        j = bisect.bisect_left(times, hi)
        out: dict[int, list[float]] = defaultdict(list)
        for k in range(i, j):
            _, aid, _text, val = utt[k]
            out[aid].append(val)
        return out

    def window_texts(lo: int, hi: int) -> dict[int, list[tuple[str, float]]]:
        i = bisect.bisect_left(times, lo)
        j = bisect.bisect_left(times, hi)
        out: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for k in range(i, j):
            _, aid, text, val = utt[k]
            out[aid].append((text, val))
        return out

    results: list[dict] = []
    for ev in shared:
        t = ev["sim_min"]
        pre = window(t - MIN_PER_DAY, t)
        post = window(t, t + MIN_PER_DAY)
        post_texts = window_texts(t, t + MIN_PER_DAY)
        # Δvalence(agent 別): 事象後平均 - 事象前平均。前後どちらかしか無い agent は
        # 「事後のみ」を反応値として扱う(事前が無ければ 0 基準)。
        deltas: list[tuple[int, float]] = []
        for aid in sorted(set(pre) | set(post)):
            if aid not in post:
                continue                                # 事象後に発話が無ければ反応不明
            post_mean = st.mean(post[aid])
            pre_mean = st.mean(pre[aid]) if aid in pre else 0.0
            deltas.append((aid, round(post_mean - pre_mean, 4)))
        if not deltas:
            results.append({"label": ev["label"], "kind": ev["kind"],
                            "n_agents": 0, "delta_mean": None, "delta_var": None,
                            "most_pos": None, "most_neg": None})
            continue
        dvals = [d for _, d in deltas]
        dmean = round(st.mean(dvals), 4)
        dvar = round(st.pvariance(dvals), 6) if len(dvals) > 1 else 0.0
        most_pos = max(deltas, key=lambda kv: (kv[1], -kv[0]))
        most_neg = min(deltas, key=lambda kv: (kv[1], kv[0]))

        def example(aid: int, positive: bool) -> str | None:
            texts = post_texts.get(aid) or []
            if not texts:
                return None
            pick = max(texts, key=lambda tv: tv[1]) if positive \
                else min(texts, key=lambda tv: tv[1])
            return pick[0]

        results.append({
            "label": ev["label"], "kind": ev["kind"], "n_agents": len(deltas),
            "delta_mean": dmean, "delta_var": dvar,
            "most_pos": {"agent_id": most_pos[0], "delta": most_pos[1],
                         "text": example(most_pos[0], True)},
            "most_neg": {"agent_id": most_neg[0], "delta": most_neg[1],
                         "text": example(most_neg[0], False)},
        })
    return {"has_events": True, "events": results}


# --------------------------------------------------------------------------- #
# 5. 信念(belief)の世界観クラスタ
# --------------------------------------------------------------------------- #
def collect_beliefs(events: list[dict]) -> list[dict]:
    """reflect の belief 書き戻し(written_back=True かつ belief 非 None)を収集する。

    belief の取得元 = L1 "reflect" イベント(cognition/reflection.py が
    payload={"belief": ..., "written_back": bool} で記録する)。取れない場合の
    L3 snapshots フォールバックは、reflect が正準経路のため本版では不要。
    """
    recs: list[dict] = []
    for e in events:
        if e["kind"] != "reflect":
            continue
        p = e["payload"]
        if p.get("written_back") and p.get("belief"):
            recs.append({"agent_id": int(e["agent_id"]), "day": _day_of(e["sim_min"]),
                         "sim_min": int(e["sim_min"]), "text": str(p["belief"])})
    return recs


def cluster_texts(texts: list[str], threshold: float) -> tuple[dict[str, int], list[str]]:
    """文字3-gram コサイン類似で決定論的な貪欲クラスタリング。

    texts をソートし、先頭から各語を既存クラスタ代表と比較。最大類似が threshold 以上なら
    そのクラスタへ、無ければ新規クラスタ(その語を代表に)。ソート順+strict `>` で
    最早クラスタを選ぶため完全に決定論。measure.char_ngrams/ngram_cosine を流用。
    """
    reps: list[tuple[str, Counter]] = []
    assign: dict[str, int] = {}
    for t in sorted(set(texts)):
        prof = m.char_ngrams(t, 3)
        best_i, best_sim = -1, -1.0
        for i, (_rt, rp) in enumerate(reps):
            sim = m.ngram_cosine(prof, rp)
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_i >= 0 and best_sim >= threshold:
            assign[t] = best_i
        else:
            assign[t] = len(reps)
            reps.append((t, prof))
    return assign, [rt for rt, _ in reps]


def belief_analysis(events: list[dict]) -> dict:
    """belief を世界観クラスタに分け、サイズ・代表・前半/後半の構成変化を出す。"""
    recs = collect_beliefs(events)
    if not recs:
        return {"has_beliefs": False, "n_clusters": 0, "clusters": [],
                "latest_cluster_by_agent": {}, "n_beliefs": 0}
    texts = [r["text"] for r in recs]
    assign, reps = cluster_texts(texts, _CLUSTER_THRESHOLD)

    # クラスタごとの集計: レコード数・ユニーク語・代表(最頻→ソート先頭)。
    members: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        members[assign[r["text"]]].append(r)
    days = [r["day"] for r in recs]
    mid = st.median(days) if days else 0

    clusters: list[dict] = []
    for cid in sorted(members):
        rows = members[cid]
        freq = Counter(r["text"] for r in rows)
        rep = min(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        n_first = sum(1 for r in rows if r["day"] <= mid)
        n_second = len(rows) - n_first
        clusters.append({"cluster_id": cid, "n_records": len(rows),
                         "n_unique": len(freq), "rep": rep,
                         "n_first_half": n_first, "n_second_half": n_second})
    clusters.sort(key=lambda c: (-c["n_records"], c["cluster_id"]))

    # 各 agent の最新 belief が属するクラスタ(カード用)。
    latest: dict[int, dict] = {}
    for r in recs:
        cur = latest.get(r["agent_id"])
        if cur is None or r["sim_min"] > cur["sim_min"]:
            latest[r["agent_id"]] = r
    latest_cluster = {aid: assign[r["text"]] for aid, r in latest.items()}

    # 前半/後半のクラスタ構成(世界観の収斂/分極)。
    first_dist = Counter()
    second_dist = Counter()
    for r in recs:
        (first_dist if r["day"] <= mid else second_dist)[assign[r["text"]]] += 1
    return {"has_beliefs": True, "n_clusters": len(reps), "clusters": clusters,
            "latest_cluster_by_agent": latest_cluster, "n_beliefs": len(recs),
            "latest_by_agent": latest, "mid_day": mid,
            "first_dist": dict(first_dist), "second_dist": dict(second_dist)}


# --------------------------------------------------------------------------- #
# 6. 世界観カード
# --------------------------------------------------------------------------- #
def _ctrl_phrase(c: float | None) -> str:
    if c is None:
        return "手応えは不明(データ不足)"
    if c >= 0.7:
        return f"「動けば世界は応える」という強い手応え(ctrl={c:.2f})"
    if c <= 0.3:
        return f"「何をしても変わらない」という無力感(ctrl={c:.2f})"
    return f"手応えと無力の中間(ctrl={c:.2f})"


def build_agent_summary(agents_meta: list[dict], agent_rows: list[dict],
                        pioneer: dict[int, int], free: dict[int, int],
                        belief: dict) -> dict[int, dict]:
    """カード/付録用に agent ごとの主観状態をまとめる。"""
    name_by_id = {int(a["id"]): a.get("name") for a in agents_meta if "id" in a}
    occ_by_id = {int(a["id"]): a.get("occupation") for a in agents_meta if "id" in a}
    by_agent: dict[int, list[dict]] = defaultdict(list)
    for r in agent_rows:
        by_agent[r["agent_id"]].append(r)
    latest_belief = belief.get("latest_by_agent", {})
    latest_cluster = belief.get("latest_cluster_by_agent", {})

    out: dict[int, dict] = {}
    for aid in sorted(by_agent):
        rows = sorted(by_agent[aid], key=lambda x: x["day"])
        ctrl_vals = [r["ctrl"] for r in rows if r["ctrl"] is not None]
        errs = [r["err_mean"] for r in rows if r["err_mean"] is not None]
        exps = [r["expect_n"] for r in rows if r["expect_n"] is not None]
        lb = latest_belief.get(aid)
        out[aid] = {
            "agent_id": aid, "name": name_by_id.get(aid),
            "occupation": occ_by_id.get(aid),
            "ctrl_final": ctrl_vals[-1] if ctrl_vals else None,
            "err_first": errs[0] if errs else None,
            "err_last": errs[-1] if errs else None,
            "expect_n_final": exps[-1] if exps else None,
            "pioneer": pioneer.get(aid, 0), "free_action": free.get(aid, 0),
            "belief": lb["text"] if lb else None,
            "cluster": latest_cluster.get(aid),
        }
    return out


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3) -> str:
    if v is None:
        return "データ不足"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _card(s: dict) -> list[str]:
    name = s["name"] or f"agent{s['agent_id']}"
    lines = [f"### {name}(id={s['agent_id']}・{s['occupation'] or '職業不明'})"]
    err_note = "データ不足"
    if s["err_first"] is not None and s["err_last"] is not None:
        d = s["err_first"] - s["err_last"]
        trend = "世界に詳しくなりつつある(期待誤差が縮小)" if d > 0 \
            else ("期待誤差は横ばい" if abs(d) < 1e-9 else "期待誤差はむしろ拡大")
        err_note = f"{trend}(誤差 {s['err_first']:.2f}→{s['err_last']:.2f})"
    cl = "未所属" if s["cluster"] is None else f"#{s['cluster']}"
    lines.append(
        f"- この人は世界を: {_ctrl_phrase(s['ctrl_final'])}。{err_note}。"
        f"期待テーブル規模 {_fmt(s['expect_n_final'])}。"
        f"開拓的行動 {s['pioneer']}回 / free_action {s['free_action']}回。"
        f"世界観クラスタ {cl}。")
    if s["belief"]:
        lines.append(f"- 直近 belief: 「{s['belief']}」")
    else:
        lines.append("- 直近 belief: データ不足(書き戻しなし)")
    return lines


def write_report(path: str, run_name: str, off_run: bool, panel_rows: int,
                 learn: dict, ctrl: dict, interp: dict, belief: dict,
                 summary: dict[int, dict], n_agents: int, n_city_days: int) -> str:
    L: list[str] = []
    L.append(f"# 世界解釈レポート: {run_name}\n")
    if off_run:
        L.append("**worldview OFF ラン**: このランには worldview イベントが 0 件です"
                 "(主観的世界モデル層が無効)。主要セクションはスキップします。\n")
        L.append("`worldview.enabled=true` で生成したランを与えると、期待・可制御性・"
                 "規範予期・解釈の分岐・世界観クラスタが観測できます。")
        md = "\n".join(L) + "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return md

    # ---- 0. 概要 ----
    L.append("## 0. 概要")
    L.append(f"- agents: {n_agents} / パネル行数: {panel_rows} / 街の日次観測: {n_city_days}日")
    L.append(f"- パネル: `panel/worldview.parquet`(agent×日 の ctrl/expect_n/err_mean + 街 norm_rate)\n")

    # ---- 2. 仮説検証ループ ----
    L.append("## 1. 仮説検証ループ(世界を知る速度)")
    if learn["overall_series"]:
        series = ", ".join(f"d{d}:{v:.3f}" for d, v in learn["overall_series"])
        L.append(f"- 全体の平均期待誤差(日次): {series}")
        L.append(f"- 全体の収束(前半→後半の低下率): {_fmt(learn['conv_overall'])}"
                 f"{'(前半より後半で誤差が縮んだ=世界を学んだ)' if (learn['conv_overall'] or 0) > 0 else ''}")
        if learn["fast"]:
            L.append("- 早く世界を学ぶ上位(誤差低下量 desc):")
            for a in learn["fast"]:
                L.append(f"  - id={a['agent_id']}: {a['err_first']:.2f}→{a['err_last']:.2f} "
                         f"(低下 {a['drop']:+.2f}, n={a['n']})")
            L.append("- 学びの遅い下位(誤差低下量 asc):")
            for a in learn["slow"]:
                L.append(f"  - id={a['agent_id']}: {a['err_first']:.2f}→{a['err_last']:.2f} "
                         f"(低下 {a['drop']:+.2f}, n={a['n']})")
        else:
            L.append("- 個人系列: データ不足(誤差の日次点が 2 日分に満たない)")
    else:
        L.append("- データ不足(期待誤差の日次系列が無い)")
    L.append("")

    # ---- 3. 可制御性の分岐 ----
    L.append("## 2. 可制御性の分岐(全員 0.5 起点からの経路依存)")
    if ctrl["n"]:
        vs = ", ".join(f"d{d}:{v:.4f}" for d, v in ctrl["var_series"])
        L.append(f"- ctrl の日次分散(分岐の速度): {vs}")
        L.append(f"- 分岐の有無: {'あり(最終日で個体差が生じている)' if ctrl['diverged'] else 'なし(全員ほぼ同値)'}")
        L.append("- ctrl 終値 上位(手応え):")
        for aid, c in ctrl["top"]:
            L.append(f"  - id={aid}: {c:.3f}")
        L.append("- ctrl 終値 下位(無力):")
        for aid, c in ctrl["bottom"]:
            L.append(f"  - id={aid}: {c:.3f}")
        note = " ※n が小さいため参考値" if ctrl["n"] < 10 else ""
        L.append(f"- 行動との相関(ctrl終値 × 開拓的行動): r={_fmt(ctrl['corr_pioneer'], 3)}{note}")
        L.append(f"- 行動との相関(ctrl終値 × free_action): r={_fmt(ctrl['corr_free'], 3)}{note}")
        nz = [t for t in ctrl["table"] if t["pioneer"] or t["free_action"]]
        if nz:
            L.append("- 対応表(開拓的行動 or free_action が非0の個体):")
            for t in sorted(nz, key=lambda x: (-x["ctrl"], x["agent_id"]))[:10]:
                L.append(f"  - id={t['agent_id']}: ctrl={t['ctrl']:.3f} "
                         f"pioneer={t['pioneer']} free={t['free_action']}")
        else:
            L.append("- 対応表: 開拓的行動・free_action ともに全員 0(データ不足)")
    else:
        L.append("- データ不足(ctrl の観測が無い)")
    L.append("")

    # ---- 4. 解釈の分岐 ----
    L.append("## 3. 解釈の分岐(同じ事象・違う理解)")
    if interp["has_events"]:
        L.append("- 共有事象の前後24hの発話感情価(valence)の個体差:")
        for ev in interp["events"]:
            if not ev["n_agents"]:
                L.append(f"  - {ev['label']}: 事象後の発話が無く反応を測れない(データ不足)")
                continue
            L.append(f"  - {ev['label']}: 反応した {ev['n_agents']} 名 / "
                     f"平均Δvalence={_fmt(ev['delta_mean'], 3)} / 分散={_fmt(ev['delta_var'], 4)}")
            mp, mn = ev["most_pos"], ev["most_neg"]
            if mp:
                L.append(f"    - 最ポジ id={mp['agent_id']}(Δ{mp['delta']:+.3f}): "
                         f"「{mp['text'] or '(発話テキストなし)'}」")
            if mn:
                L.append(f"    - 最ネガ id={mn['agent_id']}(Δ{mn['delta']:+.3f}): "
                         f"「{mn['text'] or '(発話テキストなし)'}」")
    else:
        L.append("- データ不足(悪天/災害/提案成立/広告改定 の共有事象がこのランに無い)")
    L.append("")

    # ---- 5. 信念の世界観クラスタ ----
    L.append("## 4. 信念(belief)の世界観クラスタ")
    if belief["has_beliefs"]:
        L.append(f"- belief 書き戻し: {belief['n_beliefs']} 件 → 世界観クラスタ {belief['n_clusters']} 個"
                 f"(文字3-gram コサイン類似 ≥ {_CLUSTER_THRESHOLD} で決定論クラスタ化)")
        for c in belief["clusters"][:8]:
            L.append(f"  - クラスタ#{c['cluster_id']}: {c['n_records']}件"
                     f"(ユニーク{c['n_unique']} / 前半{c['n_first_half']}・後半{c['n_second_half']})"
                     f" 代表「{c['rep']}」")
        fd, sd = belief.get("first_dist", {}), belief.get("second_dist", {})
        L.append(f"- 前半のクラスタ構成: {dict(sorted(fd.items()))}")
        L.append(f"- 後半のクラスタ構成: {dict(sorted(sd.items()))}")
        if belief["n_clusters"] <= 1:
            L.append("- 収斂/分極: クラスタが1個=世界観がほぼ単一(mock の定型内省ではこれが通常)")
    else:
        L.append("- データ不足(belief の書き戻しが無い。writeback=off などの可能性)")
    L.append("")

    # ---- 6. 世界観カード ----
    L.append("## 5. 世界観カード(この人は世界をこう見ている)")
    cards = list(summary.values())
    top6 = sorted(cards, key=lambda s: (-(s["ctrl_final"] if s["ctrl_final"] is not None else -1),
                                        s["agent_id"]))[:6]
    top_ids = {s["agent_id"] for s in top6}
    bottom3 = [s for s in sorted(
        cards, key=lambda s: ((s["ctrl_final"] if s["ctrl_final"] is not None else 2),
                              s["agent_id"])) if s["agent_id"] not in top_ids][:3]
    L.append("### 手応えの上位6人")
    for s in top6:
        L.extend(_card(s))
    L.append("\n### 無力感の下位3人")
    for s in bottom3:
        L.extend(_card(s))
    L.append("")

    # ---- 付録: 全員テーブル ----
    L.append("## 付録: 全 agent の主観状態(ctrl終値 desc)")
    L.append("| id | name | ctrl | err(初→終) | expect_n | pioneer | free | cluster |")
    L.append("|----|------|------|-----------|----------|---------|------|---------|")
    for s in sorted(cards, key=lambda s: (-(s["ctrl_final"] if s["ctrl_final"] is not None else -1),
                                          s["agent_id"])):
        err = "-" if s["err_first"] is None else f"{s['err_first']:.2f}→{s['err_last']:.2f}"
        cl = "-" if s["cluster"] is None else f"#{s['cluster']}"
        L.append(f"| {s['agent_id']} | {s['name'] or '-'} | "
                 f"{_fmt(s['ctrl_final'], 3)} | {err} | {_fmt(s['expect_n_final'])} | "
                 f"{s['pioneer']} | {s['free_action']} | {cl} |")
    L.append("")
    md = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return md


# --------------------------------------------------------------------------- #
# ドライバ
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, out_md: str | None = None,
            events: list[dict] | None = None,
            agents: list[dict] | None = None) -> dict:
    """単一ランの世界解釈分析。events/agents をメモリで渡せば L1 を読まない。"""
    import l1_stream as ls
    run_name = os.path.basename(os.path.normpath(run_dir))
    from_memory = events is not None            # events を渡された経路は逐語温存
    if events is None:
        events = load_events(run_dir)
    if agents is None:
        agents = m.load_agents(run_dir)
    out_md = out_md or os.path.join(run_dir, "worldview_report.md")

    agent_rows, city_rows = split_worldview(events)
    off_run = not agent_rows and not city_rows
    # n_agents は **全 kind** に現れる agent_id の最大値から決まる。events を渡された
    # ときはそれが唯一の情報源なので従来式のまま、L1 から読んだときは列走査へ。
    n_agents = len(agents) or (
        (max((e["agent_id"] for e in events), default=-1) + 1) if from_memory
        else (ls.max_agent_id(run_dir, nonneg=False) + 1))

    if off_run:
        md = write_report(out_md, run_name, True, 0, {}, {}, {}, {}, {}, n_agents, 0)
        return {"run": run_name, "off_run": True, "n_panel_rows": 0,
                "n_clusters": 0, "report": out_md, "md": md}

    out_dir = os.path.join(run_dir, "panel")
    panel_path, panel_rows = build_panel(run_name, agent_rows, city_rows, out_dir)

    pioneer, free = _pioneer_and_free_counts(events)
    learn = learning_analysis(agent_rows)
    ctrl = ctrl_analysis(agent_rows, pioneer, free)
    interp = interpretation_analysis(events)
    belief = belief_analysis(events)
    summary = build_agent_summary(agents, agent_rows, pioneer, free, belief)
    n_city_days = len({r["day"] for r in city_rows})

    md = write_report(out_md, run_name, False, panel_rows, learn, ctrl, interp,
                      belief, summary, n_agents, n_city_days)
    return {"run": run_name, "off_run": False, "n_panel_rows": panel_rows,
            "panel_path": panel_path, "n_clusters": belief["n_clusters"],
            "ctrl_diverged": ctrl["diverged"], "conv_overall": learn["conv_overall"],
            "report": out_md, "md": md,
            "n_shared_events": len(interp.get("events", [])),
            "n_beliefs": belief["n_beliefs"]}


def _print_summary(info: dict) -> None:
    print(f"[analyze_worldview] {info['run']} -> {info['report']}")
    if info["off_run"]:
        print("  worldview OFF ラン(主要セクションはスキップ)")
        return
    print(f"  パネル行数={info['n_panel_rows']}  世界観クラスタ={info['n_clusters']}")
    print(f"  ctrl 分岐={'あり' if info['ctrl_diverged'] else 'なし'}  "
          f"全体収束(前半→後半低下率)={info['conv_overall']}")
    print(f"  belief 書き戻し={info['n_beliefs']}件  共有事象={info['n_shared_events']}件")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第20バッチ: 主観的世界モデルの観察(読み出し専用)")
    ap.add_argument("run_dir", help="例: runs/daily_llm_20a3d")
    ap.add_argument("--out", default=None,
                    help="レポート出力先(既定 runs/<name>/worldview_report.md)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    info = analyze(args.run_dir, out_md=args.out)
    _print_summary(info)


if __name__ == "__main__":
    main()
