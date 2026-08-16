#!/usr/bin/env python
"""第56バッチ タスクB: 社会構造の自然変化の事後分析(後処理・L1+L3 を読むだけ)。

    python scripts/analyze_structure.py runs/<name> [--top-k 10] [--min-days 3]

設計正典: docs/closed-world-daily-observation.md §3 タスクB / devlog Entry 47(Fable 精査)。
目的:「強制トリガーなしで社会構造は変化するのか、固着するのか」を正面から測る。関係性の固定化が
実際に起きるかを **介入せず** 観測する(危機トリガー不採用の日常観察方針)。固着は「観測記録」であり
アラートでも介入でもない。

計算(すべて L1 events + hierarchy ON 時の L3 status から。**新規依存なし**):
  1. edge 組み替え(churn): 日次の新規形成/断絶/風化の件数・率を relation_tier / relation_break から
     再構成(in-sim L2 と同一定義=formed=relation_tier{tier==1} / broken=break{cause!=absence} /
     decayed=break{cause==absence})。母数=当日開始時の活性関係数(tier≥1 の有向台帳を累積再構成)。
  2. 順位固着(Kendall τ): 地位(L3 status)or 評判(L1 reputation_update)ランキングの日次系列の
     前日比・前週比 Kendall τ(τ-b=observer/assets._kendall_tau を再利用=O(N log N)・
     凍結 measure._kendall_tau と同値)。
     τ が高い=順位が入れ替わらない=ヒエラルキーの固着。上位K人の順位推移(レースチャート素材)も出す。
  3. 中心性入れ替わり: 会話グラフ(speak/dm=observer/measure._conversation_adj を再利用)の次数中心性
     上位K の日次入れ替わり率(turnover)。低い=同じ人が中心=固着。
  4. コミュニティ構成の日次変化: **既存の決定論コミュニティ検出**(observer/measure.communities=会話
     グラフの label propagation。Louvain/networkx でなく既存資産を日次窓で回す=新規依存なし)。
     連続2日の「各人の共同メンバー集合」の Jaccard から所属変化率を出す。
  5. 固着検知: 中心性 turnover(低=固着)・edge churn(低=固着)・順位 τ(高=固着)が閾値側で連続
     N 日続いた区間を「構造固着期間」として出力(JSON + テキスト。観測記録=介入しない)。

計算量: O(days × (N log N τ + N log N ソート + E LPA))級(全て純Python/numpy・乱数なし)。
  ★第133 レーンR2: τ だけが O(N²) で残っていた(25万人 = 1 回 312 億ペア × 日 2 回 = 数日級)。
    observer/assets の O(N log N) 実装へ差し替え済み(値は 1 ビットも変わらない)。
決定論: ノード・エッジ・順位のタイは常に id 昇順で解決。出力 JSON は sort_keys=True でバイト同一。

出力:
  runs/<name>/structure.json       … 日次時系列 + 順位 τ + 中心性 turnover + コミュニティ変化 + 固着区間
  runs/<name>/structure_report.md  … 日本語の要約(固着期間の物語つき)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Windows cp932 対策(コンソール出力の文字化け・例外を防ぐ)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                            # noqa: E402  (W2-3: ランの Δt の単一の源)
from society.observer import assets as _assets  # noqa: E402  (τ の O(N log N) 実装)
from society.observer import measure as m  # noqa: E402

# --------------------------------------------------------------------------- #
# 既定パラメータ(決定論のためスクリプト定数。CLI で上書き可)
# --------------------------------------------------------------------------- #
# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 の 144。日の第一手段は sim_min//1440
# なので Δt 非依存で、この定数は sim_min 欠損時の後退経路と L3 の日バケツにだけ効く。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY
TOP_K = 10                   # 中心性/順位の上位K(レースチャート・turnover)
MIN_DAYS = 3                 # 「構造固着」とみなす連続日数の下限 N
CENT_CHURN_LOW = 0.10        # 中心性 turnover がこの値以下 = 上位が入れ替わらない = 固着
EDGE_CHURN_LOW = 0.05        # edge 組み替え率がこの値以下 = 紐帯が組み替わらない = 固着
TAU_HIGH = 0.90              # 前日比 Kendall τ がこの値以上 = 順位が入れ替わらない = 固着
_BREAK_ABSENCE = "absence"   # relation_break の風化(decay)cause(observer/structure と同一定義)


# --------------------------------------------------------------------------- #
# 順位相関 Kendall τ(observer/assets の O(N log N) τ-b を再利用)
# --------------------------------------------------------------------------- #
#: τ-b の実体。`observer/assets._kendall_tau` は「同値ペア数(O(n))+ マージソートの
#: 反転数(O(n log n))」から τ-b を出す純Python 実装で、凍結 `measure._kendall_tau`
#: (明示二重ループ = O(n²))と**有限入力で完全に同値**(float のビットまで一致。
#: tests/test_assets_tau_scale.py が旧 O(n²) 参照実装との一致を機械固定している)。
#:
#: ★なぜ差し替えたか(第133 レーンR2): `rank_stability` は**全ランク住民**の τ を
#:   1 日あたり 2 回(前日比・前週比)呼ぶ。25 万人では 1 回あたり 312 億ペアの
#:   純Python 二重ループ = 数時間、30 日ランで数日級になり事後解析が回らなかった。
#:   値は 1 ビットも変えず、計算量だけを O(N log N) へ落とす差し替え。
#: ★凍結ファイル `observer/measure.py` は 1 バイトも触っていない(m._kendall_tau は
#:   そのまま在る)。ここが選ぶ実装を変えただけ。
_tau_impl = _assets._kendall_tau


def kendall_tau(a, b):
    """2 系列の Kendall τ-b(タイ補正込み)。長さ<2 は None・片側定数は 0.0。

    返り値の丸め(6桁)と None 規約は差し替え前と同一。"""
    if len(a) < 2 or len(b) < 2:
        return None
    t = _tau_impl(a, b)
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    return round(float(t), 6)


# --------------------------------------------------------------------------- #
# 日次バケツ + 補助
# --------------------------------------------------------------------------- #
# W4-D: 本解析が読む kind。
#   ① edge churn      = relation_tier / relation_break
#   ② 順位固着(評判)= reputation_update(L3 status が無いランの後退経路)
#   ③ 中心性・コミュニティ = speak / dm(`measure._conversation_adj` / `communities`)
# ここに無い kind は ①②の if/elif も凍結側の会話グラフも必ず読み飛ばす。
# **全 kind 依存は 2 つだけ**で、どちらも列走査に置き換えてある:
#   n_agents(母数) → `l1_stream.max_agent_id` / 日軸の右端 → `l1_stream.max_day`。
WANT_KINDS = frozenset({"relation_tier", "relation_break", "reputation_update",
                        "speak", "dm"})


def load_events(run_dir: str) -> list[dict]:
    """L1 を `WANT_KINDS` だけ読み、`measure.load_events` と同一形の dict 列で返す。

    残る比例項: 関係イベントと会話イベントの行数(日次窓を全期間ぶん組むので畳めない)。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):        # m.load_events と同じく「無ければ落とす」
        raise FileNotFoundError(os.path.join(run_dir, "l1_events.parquet"))
    return list(ls.iter_events(run_dir, kinds=WANT_KINDS))


def _day_of(e: dict, spd: int = STEPS_PER_DAY) -> int:
    """イベントの暦日。sim_min//1440(欠損時は step//spd へ後退)。

    W2-3: 第一手段は sim_min 基準なので **Δt に一切依存しない**(この形が処方箋の正典)。
    `spd` の既定は正準 144 で、後退経路でも Δt=10 なら従来と完全同値。"""
    sm = e.get("sim_min")
    if sm is None:
        return int(e["step"]) // int(spd)
    return int(sm) // 1440


def bucket_by_day(events: list[dict], spd: int = STEPS_PER_DAY,
                  max_day: int | None = None) -> tuple[dict[int, list], list[int]]:
    """events を暦日へ振り分け、連続する日レンジ [0..max] を返す(空日は空リスト=時系列を連続に)。

    W4-D: `max_day` を渡すと日レンジの右端を **そちら**で決める。events を kind 絞り
    読みにすると「最後の関係イベントの日」までしか日軸が伸びず、系列の長さが変わって
    しまう。日軸は「どの kind でもいいから最後にイベントがあった日」なので、
    呼び出し側が `l1_stream.max_day(run_dir, spd)`(全 kind の 2 列走査)を渡す。
    None なら従来どおり events から決める(= 全件 list を渡す既存経路と逐語同一)。
    """
    by_day: dict[int, list] = defaultdict(list)
    for e in events:
        by_day[_day_of(e, spd)].append(e)
    if max_day is None:
        if not by_day:
            return {}, []
        max_day = max(by_day)
    elif max_day < 0:
        return {}, []
    days = list(range(0, int(max_day) + 1))
    return by_day, days


def _rank_of(values: dict[int, float]) -> dict[int, int]:
    """値の降順順位(1=最上位。同値は id 昇順で安定=決定論)。"""
    order = sorted(values, key=lambda i: (-values[i], i))
    return {i: r for r, i in enumerate(order, start=1)}


def _top_set(degrees: dict[int, int], k: int) -> list[int]:
    """次数の高い順に上位 k(同次数は id 昇順=決定論)。"""
    return [u for u, _ in sorted(degrees.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0                       # 双方が孤立=同一(変化なし)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# --------------------------------------------------------------------------- #
# 1. edge 組み替え(churn)の日次再構成(in-sim L2 と同一定義。母数=当日開始の活性関係数)
# --------------------------------------------------------------------------- #
def reconstruct_churn(by_day: dict[int, list], days: list[int]) -> dict:
    """relation_tier / relation_break から日次 formed/broken/decayed と組み替え率を再構成。

    母数=当日開始時の tier≥1 有向台帳数(events を順に処理して累積 tie_state を持つ=事後層なので前日
    状態を持ってよい。in-sim 側は前日状態を持たない=これは事後の精密版)。relation イベントが 1 件も
    無いラン(relations OFF)では全て 0 = graceful(中心性・コミュニティは会話グラフから別途測れる)。"""
    tie: dict[tuple, int] = {}            # (actor, other) -> 現 tier(有向)
    formed, broken, decayed, active, rate = [], [], [], [], []
    has_rel = False
    for d in days:
        active_start = sum(1 for t in tie.values() if t >= 1)   # 当日開始時の母数
        f = b = c = 0
        for e in by_day.get(d, []):
            k = e["kind"]
            if k == "relation_tier":
                has_rel = True
                p = e["payload"]
                oid = p.get("other")
                tv = int(p.get("tier", 0))
                if isinstance(oid, int):
                    tie[(e["agent_id"], oid)] = tv
                if tv == 1:
                    f += 1                # 0→1=新規紐帯(深化 1→2/2→3 は数えない)
            elif k == "relation_break":
                has_rel = True
                p = e["payload"]
                oid = p.get("other")
                to_t = int(p.get("to_tier", 0))
                if isinstance(oid, int):
                    tie[(e["agent_id"], oid)] = to_t
                if p.get("cause") == _BREAK_ABSENCE:
                    c += 1                # 長期不在の風化
                else:
                    b += 1                # 能動的な負交流の断絶・悪化
        churn = f + b + c
        formed.append(f)
        broken.append(b)
        decayed.append(c)
        active.append(active_start)
        rate.append(round(churn / max(active_start, 1), 6))
    return {"edges_formed": formed, "edges_broken": broken, "edges_decayed": decayed,
            "active_ties": active, "churn_rate": rate, "has_relation_events": has_rel}


# --------------------------------------------------------------------------- #
# 2. 順位固着(Kendall τ + レースチャート素材)
# --------------------------------------------------------------------------- #
def _status_by_day(run_dir: str, days: list[int],
                   spd: int = STEPS_PER_DAY) -> dict[int, dict] | None:
    """L3 スナップから各日の末尾スナップの status を読む(hierarchy ON 時のみ。無ければ None)。"""
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l3_snapshots.parquet")
    if not os.path.exists(path):
        return None
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception:
        return None
    last_by_day: dict[int, dict] = {}
    found = False
    for r in rows:
        d = int(r["step"]) // int(spd)
        prev = last_by_day.get(d)
        if prev is None or int(r["step"]) >= prev[0]:
            last_by_day[d] = (int(r["step"]), r)
    out: dict[int, dict] = {}
    for d, (_step, r) in last_by_day.items():
        try:
            state = json.loads(r["state"])
        except Exception:
            continue
        vals = {}
        for a in state.get("agents", []):
            if "status" in a:
                vals[int(a["id"])] = float(a["status"])
                found = True
        if vals:
            out[d] = vals
    return out if found else None


def _reputation_by_day(by_day: dict[int, list], days: list[int]) -> dict[int, dict]:
    """L1 reputation_update から各日末の累積評判(new を carry-forward)を再構成。"""
    cur: dict[int, float] = {}
    out: dict[int, dict] = {}
    for d in days:
        for e in by_day.get(d, []):
            if e["kind"] == "reputation_update":
                aid = e["agent_id"]
                if aid >= 0:
                    cur[aid] = float(e["payload"].get("new", 0.0))
        if cur:
            out[d] = dict(cur)
    return out


def rank_stability(run_dir: str, by_day: dict[int, list], days: list[int],
                   top_k: int, spd: int = STEPS_PER_DAY) -> dict:
    """地位(L3 status 優先)or 評判(L1)ランキングの前日比・前週比 Kendall τ + 上位K人の順位推移。"""
    vals_by_day = _status_by_day(run_dir, days, spd)
    source = "status"
    if not vals_by_day:
        vals_by_day = _reputation_by_day(by_day, days)
        source = "reputation" if vals_by_day else "none"

    tau_prev_day: list = [None] * len(days)
    tau_prev_week: list = [None] * len(days)
    n_ranked: list = [0] * len(days)
    ranks_by_day: dict[int, dict] = {}

    for idx, d in enumerate(days):
        vd = vals_by_day.get(d)
        if vd:
            ranks_by_day[d] = _rank_of(vd)
            n_ranked[idx] = len(vd)

    def _tau(d_a: int, d_b: int):
        va, vb = vals_by_day.get(d_a), vals_by_day.get(d_b)
        if not va or not vb:
            return None
        common = sorted(set(va) & set(vb))
        if len(common) < 2:
            return None
        return kendall_tau([va[i] for i in common], [vb[i] for i in common])

    for idx, d in enumerate(days):
        if idx >= 1:
            tau_prev_day[idx] = _tau(d, days[idx - 1])
        if idx >= 7:
            tau_prev_week[idx] = _tau(d, days[idx - 7])

    # レースチャート: いずれかの日に上位K入りした住民の日次順位(1=最上位・不在は null)
    ever_top: set[int] = set()
    for d in days:
        rk = ranks_by_day.get(d, {})
        for aid, r in rk.items():
            if r <= top_k:
                ever_top.add(aid)
    agents_meta = {int(a["id"]): a for a in m.load_agents(run_dir) if "id" in a}
    race = []
    for aid in sorted(ever_top,
                      key=lambda i: min((ranks_by_day[d][i] for d in days
                                         if i in ranks_by_day.get(d, {})), default=10**9)):
        ranks = [ranks_by_day.get(d, {}).get(aid) for d in days]
        race.append({"id": aid,
                     "name": agents_meta.get(aid, {}).get("name", f"a{aid}"),
                     "ranks": ranks})
    return {"source": source, "top_k": top_k, "tau_prev_day": tau_prev_day,
            "tau_prev_week": tau_prev_week, "n_ranked": n_ranked, "race": race}


# --------------------------------------------------------------------------- #
# 3. 中心性入れ替わり(会話グラフの次数中心性 上位K turnover)
# --------------------------------------------------------------------------- #
def centrality_churn(by_day: dict[int, list], days: list[int], top_k: int) -> dict:
    """会話グラフ(measure._conversation_adj=speak/dm)の日次次数中心性 上位K の入れ替わり率。"""
    top_agents: list = []
    n_active: list = []
    turnover: list = [None] * len(days)
    prev_top: set = set()
    prev_ok = False
    for idx, d in enumerate(days):
        adj = m._conversation_adj(by_day.get(d, []))
        degrees = {u: len(adj[u]) for u in adj}
        top = _top_set(degrees, top_k)
        top_agents.append(top)
        n_active.append(len(adj))
        if idx >= 1 and prev_ok:
            # 上位K枠のうち「新規に入った」割合(=入れ替わり率。低い=固着)
            turnover[idx] = round(len(set(top) - prev_top) / top_k, 6)
        prev_top = set(top)
        prev_ok = True
    return {"top_k": top_k, "turnover": turnover, "top_agents": top_agents,
            "n_active": n_active}


# --------------------------------------------------------------------------- #
# 4. コミュニティ構成の日次変化(既存の決定論 LPA を日次窓で。Jaccard 所属変化率)
# --------------------------------------------------------------------------- #
def community_change(by_day: dict[int, list], days: list[int], n_agents: int) -> dict:
    """measure.communities(会話グラフの決定論 LPA)を日次窓で回し、共同メンバー集合の Jaccard 変化率。"""
    n_comms: list = []
    change_rate: list = [None] * len(days)
    prev_members: dict[int, set] = {}     # aid -> 同コミュニティの他メンバー集合(前日)
    prev_active: set = set()
    prev_ok = False
    for idx, d in enumerate(days):
        ev = by_day.get(d, [])
        adj = m._conversation_adj(ev)
        active = set(adj)                 # 会話に現れた非孤立ノードのみを対象(孤立 singleton を除く)
        comm = m.communities(ev, n_agents)
        groups: dict[int, set] = defaultdict(set)
        for node in active:
            groups[comm.get(node, node)].add(node)
        n_comms.append(sum(1 for g in groups.values() if len(g) >= 2))
        members: dict[int, set] = {aid: (groups[comm.get(aid, aid)] - {aid})
                                   for aid in active}
        if idx >= 1 and prev_ok:
            common = active & prev_active
            if common:
                sims = [_jaccard(members[a], prev_members[a]) for a in common]
                change_rate[idx] = round(1.0 - sum(sims) / len(sims), 6)
        prev_members, prev_active, prev_ok = members, active, True
    return {"n_communities": n_comms, "change_rate": change_rate}


# --------------------------------------------------------------------------- #
# 5. 固着検知(閾値側で連続 N 日 → 区間)
# --------------------------------------------------------------------------- #
def _stagnant_intervals(flags: list[bool], days: list[int], min_days: int,
                        values: list | None = None) -> list[dict]:
    """flags(日ごとの stagnant 真偽)の連続 True 区間で長さ≥min_days のものを区間化(決定論)。"""
    out: list[dict] = []
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if (j - i + 1) >= min_days:
                seg = {"start_day": days[i], "end_day": days[j], "len": j - i + 1}
                if values is not None:
                    vs = [values[t] for t in range(i, j + 1) if values[t] is not None]
                    seg["mean_value"] = round(sum(vs) / len(vs), 6) if vs else None
                out.append(seg)
            i = j + 1
        else:
            i += 1
    return out


def detect_stagnation(days: list[int], churn: dict, rank: dict, cent: dict,
                      min_days: int, cent_low: float, edge_low: float,
                      tau_high: float) -> dict:
    """3 信号(中心性 turnover 低 / edge churn 低 / 前日比 τ 高)で固着区間を検出する。"""
    cturn = cent["turnover"]
    cflag = [(v is not None and v <= cent_low) for v in cturn]
    crate = churn["churn_rate"]
    # edge churn は relation イベントがある時だけ意味を持つ(無いランでは固着判定に使わない)
    if churn["has_relation_events"]:
        eflag = [(v is not None and v <= edge_low) for v in crate]
    else:
        eflag = [False] * len(days)
    tpd = rank["tau_prev_day"]
    tflag = [(v is not None and v >= tau_high) for v in tpd]

    by_signal = {
        "centrality_churn": _stagnant_intervals(cflag, days, min_days, cturn),
        "edge_churn": _stagnant_intervals(eflag, days, min_days, crate),
        "rank_tau": _stagnant_intervals(tflag, days, min_days, tpd),
    }
    # 見出しの「構造固着期間」= 中心性 turnover が低い(上位が入れ替わらない)連続区間を正典とする
    # (L1 だけから必ず測れる=最も頑健。edge churn/τ は補助信号として by_signal に併記)。
    combined = by_signal["centrality_churn"]
    total = sum(seg["len"] for seg in combined)
    longest = max(combined, key=lambda s: s["len"]) if combined else None
    return {"min_days": min_days,
            "thresholds": {"centrality_churn_low": cent_low,
                           "edge_churn_low": edge_low, "tau_high": tau_high},
            "by_signal": by_signal, "combined": combined,
            "total_stagnant_days": total, "longest": longest}


# --------------------------------------------------------------------------- #
# メイン解析
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, top_k: int = TOP_K, min_days: int = MIN_DAYS,
            cent_low: float = CENT_CHURN_LOW, edge_low: float = EDGE_CHURN_LOW,
            tau_high: float = TAU_HIGH) -> dict:
    """1 ラン分の構造変化解析を実行し、structure.json 相当の dict を返す。"""
    import l1_stream as ls
    events = load_events(run_dir)
    agents = m.load_agents(run_dir)
    # 全 kind 依存の 2 量(母数と日軸)は列走査から取る(WANT_KINDS の解説を参照)。
    n_agents = len(agents) or (ls.max_agent_id(run_dir) + 1)
    # W2-3: 「1 日」はこのランの run.dt_min で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    by_day, days = bucket_by_day(events, spd, ls.max_day(run_dir, spd))
    run_name = os.path.basename(os.path.normpath(run_dir))

    churn = reconstruct_churn(by_day, days)
    rank = rank_stability(run_dir, by_day, days, top_k, spd)
    cent = centrality_churn(by_day, days, top_k)
    comm = community_change(by_day, days, n_agents)
    stag = detect_stagnation(days, churn, rank, cent, min_days,
                             cent_low, edge_low, tau_high)

    return {
        "run": run_name,
        "n_agents": n_agents,
        "n_days": len(days),
        "days": days,
        "params": {"top_k": top_k, "min_days": min_days,
                   "centrality_churn_low": cent_low, "edge_churn_low": edge_low,
                   "tau_high": tau_high, "steps_per_day": spd,
                   "detector": "measure.communities (deterministic LPA on speak/dm graph)",
                   "tau": "assets._kendall_tau (tau-b, O(N log N), "
                          "identical to measure._kendall_tau)"},
        "churn": churn,
        "rank": rank,
        "centrality": cent,
        "community": comm,
        "stagnation": stag,
    }


# --------------------------------------------------------------------------- #
# 日本語レポート
# --------------------------------------------------------------------------- #
def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v}"


def write_report(result: dict, path: str) -> None:
    L: list[str] = []
    L.append(f"# 社会構造の自然変化 観測レポート: {result['run']}\n")
    L.append("「強制トリガーなしで社会構造は変化するのか、固着するのか」を **介入せず** 観測した結果。")
    L.append("固着は観測記録であってアラート・介入ではない(日常観察方針=Entry 47)。\n")
    p = result["params"]
    L.append("## 設定")
    L.append(f"- エージェント数: {result['n_agents']} / 日数: {result['n_days']}")
    L.append(f"- 上位K={p['top_k']} / 固着の連続日数下限 N={p['min_days']}")
    L.append(f"- 閾値: 中心性turnover≤{p['centrality_churn_low']} / "
             f"edge churn≤{p['edge_churn_low']} / 前日比τ≥{p['tau_high']}")
    L.append(f"- コミュニティ検出: {p['detector']}")
    L.append(f"- 順位相関: {p['tau']}\n")

    ch = result["churn"]
    rk = result["rank"]
    ce = result["centrality"]
    co = result["community"]
    L.append("## 日次時系列")
    L.append("| 日 | 形成 | 断絶 | 風化 | 活性関係 | 組み替え率 | 前日比τ | 前週比τ | "
             "中心性turnover | コミュ数 | コミュ変化率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(result["days"]):
        L.append(f"| {d} | {ch['edges_formed'][i]} | {ch['edges_broken'][i]} | "
                 f"{ch['edges_decayed'][i]} | {ch['active_ties'][i]} | "
                 f"{_fmt(ch['churn_rate'][i], pct=True)} | "
                 f"{_fmt(rk['tau_prev_day'][i])} | {_fmt(rk['tau_prev_week'][i])} | "
                 f"{_fmt(ce['turnover'][i], pct=True)} | {co['n_communities'][i]} | "
                 f"{_fmt(co['change_rate'][i], pct=True)} |")
    L.append("")
    L.append(f"順位ソース: **{rk['source']}**"
             "(status=L3 の合成地位 / reputation=L1 の評判 / none=どちらも無し)。"
             "前日比τが高いほど順位が入れ替わらない=ヒエラルキー固着。\n")

    st = result["stagnation"]
    L.append("## 構造固着の検知(観測記録・介入しない)")
    if st["longest"]:
        lg = st["longest"]
        L.append(f"- 見出し(中心性が入れ替わらない期間): 検出 {len(st['combined'])} 区間 / "
                 f"固着延べ {st['total_stagnant_days']} 日 / 最長 Day {lg['start_day']}–{lg['end_day']}"
                 f"({lg['len']}日)")
    else:
        L.append("- 見出し(中心性が入れ替わらない期間): **固着区間なし**"
                 "(上位中心性が N 日以上入れ替わらない期間は検出されなかった=構造は動いている)")
    for sig, label in (("centrality_churn", "中心性 turnover 低(上位が入れ替わらない)"),
                       ("edge_churn", "edge churn 低(紐帯が組み替わらない)"),
                       ("rank_tau", "前日比τ 高(順位が入れ替わらない)")):
        segs = st["by_signal"][sig]
        if segs:
            spans = "、".join(f"Day {s['start_day']}–{s['end_day']}({s['len']}日)" for s in segs)
            L.append(f"  - {label}: {spans}")
        else:
            L.append(f"  - {label}: なし")
    L.append("")

    if rk["race"]:
        L.append("## 上位者の順位推移(レースチャート素材・上位K入りした住民)")
        L.append("| 住民 | " + " | ".join(f"D{d}" for d in result["days"]) + " |")
        L.append("|---" * (len(result["days"]) + 1) + "|")
        for r in rk["race"][:result["rank"]["top_k"]]:
            cells = " | ".join(str(x) if x is not None else "—" for x in r["ranks"])
            L.append(f"| {r['name']} | {cells} |")
        L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第56バッチ タスクB: 社会構造の自然変化の事後分析(L1+L3 を読むだけ)")
    ap.add_argument("run_dir", help="例: runs/day30")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="中心性/順位の上位K(既定 10)")
    ap.add_argument("--min-days", type=int, default=MIN_DAYS,
                    help="構造固着とみなす連続日数の下限 N(既定 3)")
    ap.add_argument("--cent-low", type=float, default=CENT_CHURN_LOW,
                    help="中心性 turnover 固着閾値(既定 0.10)")
    ap.add_argument("--edge-low", type=float, default=EDGE_CHURN_LOW,
                    help="edge churn 固着閾値(既定 0.05)")
    ap.add_argument("--tau-high", type=float, default=TAU_HIGH,
                    help="前日比τ 固着閾値(既定 0.90)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    result = analyze(args.run_dir, top_k=args.top_k, min_days=args.min_days,
                     cent_low=args.cent_low, edge_low=args.edge_low,
                     tau_high=args.tau_high)

    js_path = os.path.join(args.run_dir, "structure.json")
    md_path = os.path.join(args.run_dir, "structure_report.md")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
    write_report(result, md_path)

    st = result["stagnation"]
    print(f"[structure] {args.run_dir}: {result['n_days']} 日 / "
          f"順位ソース {result['rank']['source']} / "
          f"固着区間 {len(st['combined'])}(延べ {st['total_stagnant_days']} 日)")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
