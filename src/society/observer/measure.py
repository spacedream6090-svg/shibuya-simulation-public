"""Phase E: 観測・計測パイプラインの純関数群。

研究の核 = 「世界改変量 Y を traits で回帰した R²(k) の低下・seed 発散・EWS の
三角測量で相転移点 k* を探す」。ここではその素材となる agent/item/network/collective
の各指標と、EWS(Dakos 手順)・OLS R² を計算する。

依存は pyarrow / numpy / 標準ライブラリのみ(scipy / statsmodels / pandas は使わない)。
すべて決定論的(乱数を使う箇所は numpy.default_rng(0) を明示指定)。
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np

# agents.json のメタ列(trait ではない)
_META_KEYS = {
    "id", "name", "age", "gender", "occupation", "has_bicycle", "has_car",
    "home", "home_building", "home_floor", "work_name", "work_building",
    "bedtime_min",
}
# design §11 の因子名(ログには出さない指紋だが、traits ファイルとしては来うる)
_KNOWN_TRAITS = (
    "nfc", "risk_tolerance", "internal_locus", "efficacy", "ownership",
    "world_change",
)
# 個人間の「会話/伝播」とみなすチャネル
_SOCIAL_CHANNELS = {"face", "sns", "dm"}


# --------------------------------------------------------------------------- #
# ローダ
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet を読み、payload(JSON文字列)を dict に展開したイベント列を返す。

    返り値の各要素: {step, sim_min, agent_id, kind, x, y, rng_stream,
    llm_call_id, payload(dict)}。行順(=step 昇順の追記順)を保つ。

    B4-lite(スケール): 全テーブルを一括 to_pydict せず、列射影+RecordBatch 逐次で
    peak メモリを下げる(**返り値・行順は従来と完全同一**)。
    """
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload",
            "rng_stream", "llm_call_id"]
    available = set(pq.read_schema(path).names)
    cols = [c for c in want if c in available]         # 存在する列だけ射影
    pf = pq.ParquetFile(path)
    out: list[dict] = []
    for batch in pf.iter_batches(columns=cols):        # row-group/batch 逐次
        d = batch.to_pydict()
        n = len(d["step"])
        rng = d.get("rng_stream", [None] * n)
        lci = d.get("llm_call_id", [None] * n)
        step, sim_min, agent_id = d["step"], d["sim_min"], d["agent_id"]
        kind, xs, ys, pays = d["kind"], d["x"], d["y"], d["payload"]
        for i in range(n):
            raw = pays[i]
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({
                "step": int(step[i]),
                "sim_min": int(sim_min[i]),
                "agent_id": int(agent_id[i]),
                "kind": kind[i],
                "x": xs[i],
                "y": ys[i],
                "rng_stream": rng[i],
                "llm_call_id": lci[i],
                "payload": payload,
            })
    return out


def load_l2(run_dir: str) -> dict[str, list] | None:
    """l2_metrics.parquet を列 dict で返す(無ければ None)。"""
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l2_metrics.parquet")
    if not os.path.exists(path):
        return None
    return pq.read_table(path).to_pydict()


def load_agents(run_dir: str) -> list[dict]:
    """agents.json を返す(無ければ空)。"""
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_traits(run_dir: str, agents_meta: list[dict] | None = None):
    """traits を (traits_by_id, trait_names, source) で返す。

    優先順: traits.json > agents.json 内の既知因子列 > summary.json。
    見つからなければ ({}, [], None)。R² 部分の入力に使う。
    """
    tpath = os.path.join(run_dir, "traits.json")
    if os.path.exists(tpath):
        with open(tpath, encoding="utf-8") as fh:
            raw = json.load(fh)
        by_id = _normalize_traits(raw)
        names = _infer_trait_names(by_id)
        if names:
            return by_id, names, "traits.json"

    if agents_meta is None:
        agents_meta = load_agents(run_dir)
    if agents_meta:
        present = [t for t in _KNOWN_TRAITS if any(t in a for a in agents_meta)]
        # 既知因子が無くても、メタ以外の数値列があれば trait とみなす
        if not present:
            extra = set()
            for a in agents_meta:
                for k, v in a.items():
                    if k not in _META_KEYS and isinstance(v, (int, float)) \
                            and not isinstance(v, bool):
                        extra.add(k)
            present = sorted(extra)
        if present:
            by_id = {int(a["id"]): {t: a[t] for t in present if t in a}
                     for a in agents_meta if "id" in a}
            names = _infer_trait_names(by_id)
            if names:
                return by_id, names, "agents.json"

    spath = os.path.join(run_dir, "summary.json")
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as fh:
            summ = json.load(fh)
        if isinstance(summ.get("traits"), (list, dict)):
            by_id = _normalize_traits(summ["traits"])
            names = _infer_trait_names(by_id)
            if names:
                return by_id, names, "summary.json"
    return {}, [], None


def _normalize_traits(raw) -> dict[int, dict]:
    """[{id, ...}] でも {id: {...}} でも {id: {...}} 形へ正規化。"""
    by_id: dict[int, dict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                by_id[int(k)] = {a: b for a, b in v.items() if a != "id"}
    elif isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and "id" in row:
                by_id[int(row["id"])] = {a: b for a, b in row.items() if a != "id"}
    return by_id


def _infer_trait_names(by_id: dict[int, dict]) -> list[str]:
    names: set[str] = set()
    for d in by_id.values():
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                names.add(k)
    return sorted(names)


# --------------------------------------------------------------------------- #
# 補助
# --------------------------------------------------------------------------- #
def _item_creators(events: list[dict]) -> tuple[dict[str, int], set[str]]:
    """item_id -> creator(agent_id)、および media 起点 item の集合。"""
    creator: dict[str, int] = {}
    media: set[str] = set()
    for e in events:
        if e["kind"] in ("vocab_coin", "label_coin"):
            iid = e["payload"].get("item_id")
            if iid is None:
                continue
            creator.setdefault(iid, e["agent_id"])
            if e["payload"].get("media"):
                media.add(iid)
    return creator, media


def _ordered(events: list[dict]) -> list[dict]:
    """step 昇順(同 step は元の行順)で安定ソート。"""
    return sorted(events, key=lambda e: e["step"])


# ---- 「世界を変える」ツールの効果(客観カウント。y_external の合成式は変えない) ----
_TOOL_COLS = ("events_hosted", "event_attendees_drawn", "flyer_unique_viewers",
              "groups_founded", "group_members_recruited", "proposals_made",
              "signatures_gathered", "ventures_opened", "venture_revenue")


def _tool_features(events: list[dict]) -> dict[int, dict]:
    """agent ごとのツール使用・効果。0 使用でも壊れない(欠損は 0 として並べる)。"""
    out: dict[int, dict] = defaultdict(lambda: {c: 0 for c in _TOOL_COLS})
    for e in events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if aid is None or aid < 0:
            continue
        if k == "event_host":
            out[aid]["events_hosted"] += 1
        elif k == "event_attend":
            host = p.get("host")
            if isinstance(host, int) and host >= 0:
                out[host]["event_attendees_drawn"] += 1
        elif k == "flyer_view":
            au = p.get("author")
            if isinstance(au, int) and au >= 0:
                out[au]["flyer_unique_viewers"] += 1
        elif k == "group_found":
            out[aid]["groups_founded"] += 1
        elif k == "group_join":
            f = p.get("founder")
            if isinstance(f, int) and f >= 0:
                out[f]["group_members_recruited"] += 1
        elif k == "proposal":
            out[aid]["proposals_made"] += 1
        elif k == "proposal_support":
            au = p.get("author")
            if isinstance(au, int) and au >= 0:
                out[au]["signatures_gathered"] += 1
        elif k == "venture_open":
            out[aid]["ventures_opened"] += 1
        elif k == "venture_sale":
            out[aid]["venture_revenue"] += float(p.get("amount", 0) or 0)
    return out


# --------------------------------------------------------------------------- #
# agent 特徴量
# --------------------------------------------------------------------------- #
def agent_features(events: list[dict], agents_meta: list[dict] | None = None,
                   traits: dict[int, dict] | None = None,
                   y_weights: dict[str, float] | None = None,
                   landmark_score: dict | None = None) -> list[dict]:
    """agent ごとの世界改変量 Y と補助量を返す。

    y_weights (D1 4層Y統合式):
      None なら現行出力と **完全一致**(新列も追加しない=後方互換)。
      dict を渡したときのみ、各 row に Y_composite 列を追加する:
        Y_composite = Y_external + Σ w_col · row[col]
      col は _TOOL_COLS の9列や将来の数値列に対応。row に存在しない列キー
      (数値でない列も含む)は無視する。

    Y_external = 4 成分の和:
      c_transmission  : 自分が創出/送出した item の伝播数(from==自分 or creator==自分)
      c_label_adopt   : 自分の発話・投稿が引き起こした label_adopt 数
                        (採用直前に当該 item を自分がその相手へ伝播していた件数で帰属)
      c_new_relations : face/dm/sns で初めて会話した相手の distinct 数
      c_sns_reach     : 自分が SNS 投稿した item が他者へ届いた sns 伝播数
    Y_internal = reflect(written_back=True) 回数 + belief 文字数合計
    compute    = llm_deliberate + reflect の回数
    n_fires_received = drive_request(granted=True) 数(無ければ 0)
    """
    creator, _media = _item_creators(events)
    ordered = _ordered(events)
    tool_feats = _tool_features(events)

    # Searle 制度化(既定 OFF): institution イベントがあるときだけ列を足す(OFF 出力は不変)。
    inst_created: Counter = Counter()
    for e in events:
        if e["kind"] == "institution":
            cb = e["payload"].get("created_by")
            if isinstance(cb, int) and cb >= 0:
                inst_created[cb] += 1
    has_institutions = bool(inst_created) or any(
        e["kind"] == "institution" for e in events)
    # 制度DSL(既定 ON): institution_rule イベントがあるときだけ rules_enacted 列を足す。
    rules_enacted: Counter = Counter()
    for e in events:
        if e["kind"] == "institution_rule":
            pr = e["payload"].get("proposer")
            if isinstance(pr, int) and pr >= 0:
                rules_enacted[pr] += 1
    has_rules = bool(rules_enacted) or any(
        e["kind"] == "institution_rule" for e in events)
    # Lynch 認知地図(既定 OFF): landmark_score を渡された時だけ観測レンズ列を足す。
    map_breadth: dict[int, int] = {}
    landmark_focus: dict[int, float] = {}
    if landmark_score is not None:
        arrivals: dict[int, Counter] = defaultdict(Counter)
        for e in events:
            if e["kind"] == "arrive":
                node = e["payload"].get("node")
                if node is not None and e["agent_id"] >= 0:
                    arrivals[e["agent_id"]][node] += 1
        for aid, nodes in arrivals.items():
            map_breadth[aid] = len(nodes)          # 訪れた distinct ノード数=地図の広さ
            total = sum(nodes.values())
            if total > 0:                           # landmark への集中度(重み付き平均スコア)
                landmark_focus[aid] = round(sum(
                    cnt * float(landmark_score.get(node, 0.0))
                    for node, cnt in nodes.items()) / total, 4)

    # 自分が SNS 投稿した item(事前パス: post は伝播の前後どちらにも来うる)
    sns_posted: dict[int, set] = defaultdict(set)
    for e in ordered:
        if e["kind"] == "sns_post":
            for it in e["payload"].get("items", []) or []:
                sns_posted[e["agent_id"]].add(it)

    c_trans: Counter = Counter()
    c_adopt: Counter = Counter()
    c_sns: Counter = Counter()
    partners: dict[int, set] = defaultdict(set)
    wb_count: Counter = Counter()
    belief_chars: Counter = Counter()
    compute: Counter = Counter()
    fires: Counter = Counter()
    opinion_moved: Counter = Counter()   # source -> Σ|Δopinion|(FJ #16、y_external外)
    sns_likes: Counter = Counter()       # author -> 受けたいいね数(#14)
    sns_reshares: Counter = Counter()    # author -> 受けたリシェア数(#14)
    last_from: dict[tuple, int] = {}  # (item_id, receiver) -> 直近の送出元

    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if k == "transmission":
            iid, frm, ch = p.get("item_id"), p.get("from"), p.get("channel")
            if isinstance(frm, int) and frm >= 0:
                c_trans[frm] += 1
            crt = creator.get(iid)
            if crt is not None and crt != frm:  # from==creator の二重計上を回避
                c_trans[crt] += 1
            if iid is not None:
                last_from[(iid, aid)] = frm
            if ch == "sns" and isinstance(frm, int) and frm >= 0 \
                    and iid in sns_posted.get(frm, ()):
                c_sns[frm] += 1
        elif k == "label_adopt":
            src = last_from.get((p.get("item_id"), aid))
            if isinstance(src, int) and src >= 0:
                c_adopt[src] += 1
        elif k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    partners[aid].add(h)
                    partners[h].add(aid)
        elif k == "hear":
            sp = p.get("speaker")
            if isinstance(sp, int) and sp >= 0 and sp != aid:
                partners[aid].add(sp)
                partners[sp].add(aid)
        elif k == "dm":
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                partners[aid].add(to)
                partners[to].add(aid)
        elif k == "reflect":
            compute[aid] += 1
            if p.get("written_back"):
                wb_count[aid] += 1
            belief_chars[aid] += len(p.get("belief", "") or "")
        elif k == "llm_deliberate":
            compute[aid] += 1
        elif k == "drive_request":
            if p.get("granted"):
                fires[aid] += 1
        elif k == "opinion_shift":
            src = p.get("source")
            if isinstance(src, int) and src >= 0:
                opinion_moved[src] += abs(float(p.get("new", 0.0))
                                          - float(p.get("old", 0.0)))
        elif k == "sns_like":
            au = p.get("author")
            if isinstance(au, int) and au >= 0:
                sns_likes[au] += 1
        elif k == "sns_reshare":
            au = p.get("author")
            if isinstance(au, int) and au >= 0:
                sns_reshares[au] += 1

    # 対象 agent の集合
    if agents_meta:
        meta_by_id = {int(a["id"]): a for a in agents_meta if "id" in a}
        ids = sorted(meta_by_id)
    else:
        meta_by_id = {}
        seen = {e["agent_id"] for e in events if e["agent_id"] >= 0}
        ids = sorted(seen)

    traits = traits or {}
    out: list[dict] = []
    for aid in ids:
        meta = meta_by_id.get(aid, {})
        c1 = c_trans[aid]
        c2 = c_adopt[aid]
        c3 = len(partners.get(aid, ()))
        c4 = c_sns[aid]
        y_ext = c1 + c2 + c3 + c4
        y_int = wb_count[aid] + belief_chars[aid]
        row = {
            "id": aid,
            "name": meta.get("name"),
            "occupation": meta.get("occupation"),
            "Y_external": y_ext,
            "Y_internal": y_int,
            "c_transmission": c1,
            "c_label_adopt": c2,
            "c_new_relations": c3,
            "c_sns_reach": c4,
            "reflect_writeback": wb_count[aid],
            "belief_chars": belief_chars[aid],
            "compute": compute[aid],
            "n_fires_received": fires[aid],
        }
        # ツール使用・効果を列として並べる(合成は後で研究者が決める。y_external は不変)
        tf = tool_feats.get(aid)
        for col in _TOOL_COLS:
            row[col] = tf[col] if tf else 0
        # 意見操作量(FJ #16)+ SNS 反応(#14)。**y_external には入れない**
        # (B の y_weights 経由でのみ合成可)。
        row["c_opinion_moved"] = round(opinion_moved[aid], 4)
        row["sns_likes_received"] = sns_likes[aid]
        row["sns_reshares_received"] = sns_reshares[aid]
        # 行動心理プラグイン(既定 OFF)の観測列。イベント/引数が無ければ足さない=既定不変。
        if has_institutions:                         # Searle: 発起した制度数(y_external外)
            row["institutions_created"] = inst_created[aid]
        if has_rules:                                # 制度DSL: 実効ルール化した提案数(y_external外)
            row["rules_enacted"] = rules_enacted[aid]
        if landmark_score is not None:               # Lynch: 認知地図の広さ・landmark 集中度
            row["cognitive_map_breadth"] = map_breadth.get(aid, 0)
            row["landmark_focus"] = landmark_focus.get(aid, 0.0)
        for t, v in (traits.get(aid) or {}).items():
            row[t] = v
        # D1: 4層Y統合(既定 None では列を1つも足さない=完全不変)
        if y_weights is not None:
            comp = float(row["Y_external"])
            for col, w in y_weights.items():
                val = row.get(col)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    comp += float(w) * float(val)
            row["Y_composite"] = comp
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# item カスケード
# --------------------------------------------------------------------------- #
def item_cascades(events: list[dict]) -> list[dict]:
    """item ごとに size / adopters / depth / channel_mix / t_half を返す。"""
    creator, media = _item_creators(events)
    trans: dict[str, list] = defaultdict(list)   # iid -> [(step, from, to, channel)]
    edges: dict[str, set] = defaultdict(set)     # iid -> {(from, to)}
    for e in _ordered(events):
        if e["kind"] == "transmission":
            p = e["payload"]
            iid = p.get("item_id")
            if iid is None:
                continue
            trans[iid].append((e["step"], p.get("from"), e["agent_id"],
                               p.get("channel")))
            edges[iid].add((p.get("from"), e["agent_id"]))

    out: list[dict] = []
    for iid in sorted(set(creator) | set(trans)):
        tl = trans.get(iid, [])
        adopters = sorted({to for (_, _, to, _) in tl
                           if isinstance(to, int) and to >= 0})
        out.append({
            "item_id": iid,
            "creator": creator.get(iid),
            "media": iid in media,
            "size": len(tl),
            "n_adopters": len(adopters),
            "adopters": adopters,
            "depth": _tree_depth(edges.get(iid, set())),
            "channel_mix": dict(Counter(ch for (_, _, _, ch) in tl)),
            "t_half": _t_half(tl),
        })
    return out


def _tree_depth(edge_set: set) -> int:
    """from->to 有向辺の最長路(辺数)。media/search/news は from=-1 が自然な根。

    メモ化により O(V+E)。循環がある場合は再帰スタック上の後退辺を無視して
    近似(採用木は実質 DAG なので通常は厳密値)。
    """
    if not edge_set:
        return 0
    adj: dict = defaultdict(list)
    nodes: set = set()
    tos: set = set()
    for a, b in edge_set:
        adj[a].append(b)
        nodes.add(a)
        nodes.add(b)
        tos.add(b)
    roots = [n for n in nodes if n not in tos] or list(nodes)  # 循環時は全ノード
    memo: dict = {}
    onstack: set = set()

    def longest(u: Any) -> int:
        if u in memo:
            return memo[u]
        onstack.add(u)
        best = 0
        for v in adj.get(u, ()):
            if v in onstack:  # 後退辺 = 循環 → スキップ
                continue
            best = max(best, 1 + longest(v))
        onstack.discard(u)
        memo[u] = best
        return best

    return max(longest(r) for r in roots)


def _t_half(tl: list):
    """採用(distinct 受信者)累積が最終値の 50% に達した step。"""
    if not tl:
        return None
    seen: set = set()
    series: list = []
    for step, _frm, to, _ch in sorted(tl):
        if isinstance(to, int) and to >= 0:
            seen.add(to)
        series.append((step, len(seen)))
    final = series[-1][1]
    if final <= 0:
        return None
    half = 0.5 * final
    for step, c in series:
        if c >= half:
            return step
    return series[-1][0]


# --------------------------------------------------------------------------- #
# ネットワーク窓
# --------------------------------------------------------------------------- #
def network_windows(events: list[dict], window: int = 24) -> list[dict]:
    """window step ごとの会話グラフ(speak speaker->hearers, dm)指標。"""
    if not events:
        return []
    ordered = _ordered(events)
    # 窓ごとの全イベント再走査を避け、会話グラフに効く kind だけ先に1回抽出して使い回す
    # (窓数×全件 → 窓数×会話件のみ。step 昇順は維持=break 条件も同値=結果は不変)。
    relevant = [e for e in ordered if e["kind"] in ("speak", "dm")]
    max_step = max(e["step"] for e in events)
    windows: list[dict] = []
    prev_top5: list = []
    w = 0
    start = 0
    while start <= max_step:
        end = start + window
        adj: dict = defaultdict(set)
        for e in relevant:
            s = e["step"]
            if s < start:
                continue
            if s >= end:
                break
            if e["kind"] == "speak":
                sp = e["agent_id"]
                for h in e["payload"].get("hearers", []) or []:
                    if isinstance(h, int) and h >= 0 and h != sp:
                        adj[sp].add(h)
                        adj[h].add(sp)
            elif e["kind"] == "dm":
                fr = e["agent_id"]
                to = e["payload"].get("to")
                if isinstance(to, int) and to >= 0 and to != fr:
                    adj[fr].add(to)
                    adj[to].add(fr)
        nodes = sorted(adj)
        degrees = {u: len(adj[u]) for u in nodes}
        n_edges = sum(degrees.values()) // 2
        n_nodes = len(nodes)
        mean_degree = (2 * n_edges / n_nodes) if n_nodes else 0.0
        top1 = max(nodes, key=lambda u: (degrees[u], -u)) if nodes else None
        top5 = [u for u, _ in sorted(degrees.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:5]]
        churn = len(set(top5) - set(prev_top5)) if prev_top5 else 0
        windows.append({
            "window": w,
            "step_start": start,
            "step_end": end,
            "n_edges": n_edges,
            "n_nodes": n_nodes,
            "mean_degree": mean_degree,
            "clustering": _avg_clustering(adj),
            "top1_agent": top1,
            "centrality_churn": churn,
        })
        prev_top5 = top5
        start = end
        w += 1
    return windows


def _avg_clustering(adj: dict) -> float:
    """平均局所クラスタ係数(degree>=2 のノードで平均)。"""
    nodes = [u for u in adj if len(adj[u]) >= 2]
    if not nodes:
        return 0.0
    total = 0.0
    for u in nodes:
        nb = list(adj[u])
        k = len(nb)
        links = 0
        for i in range(k):
            ai = adj[nb[i]]
            for j in range(i + 1, k):
                if nb[j] in ai:
                    links += 1
        total += (2 * links) / (k * (k - 1))
    return total / len(nodes)


# --------------------------------------------------------------------------- #
# 集団時系列
# --------------------------------------------------------------------------- #
def collective_series(events: list[dict], l2: dict[str, list] | None,
                      n_agents: int) -> dict[str, list]:
    """step ごとの集団指標。l2 にある列はそのまま採用し、event 由来量を足す。"""
    if l2 and l2.get("step"):
        steps = sorted(int(s) for s in l2["step"])
    else:
        mx = max((e["step"] for e in events), default=0)
        steps = list(range(mx + 1))
    index = {s: i for i, s in enumerate(steps)}
    n_steps = len(steps)

    new_items = [0] * n_steps
    adopt_at: dict[int, set] = defaultdict(set)
    vocab_at: dict[int, Counter] = defaultdict(Counter)
    drive_at: dict[int, list] = defaultdict(list)
    for e in events:
        s = e["step"]
        if s not in index:
            continue
        k, p = e["kind"], e["payload"]
        if k == "vocab_coin":
            new_items[index[s]] += 1
        elif k == "label_adopt":
            adopt_at[s].add(e["agent_id"])
        elif k == "vocab_use":
            iid = p.get("item_id")
            if iid is not None:
                vocab_at[s][iid] += 1
        elif k == "drive_request":
            drive_at[s].append(1.0 if p.get("granted") else 0.0)

    cum_adopters: set = set()
    cum_vocab: Counter = Counter()
    adoption_frac: list = []
    vocab_entropy: list = []
    n_new: list = []
    mean_drive_ev: list = []
    has_drive = any(drive_at.values())
    for s in steps:
        cum_adopters |= adopt_at.get(s, set())
        for iid, c in vocab_at.get(s, {}).items():
            cum_vocab[iid] += c
        adoption_frac.append(len(cum_adopters) / n_agents if n_agents else 0.0)
        vocab_entropy.append(_entropy(cum_vocab))
        n_new.append(new_items[index[s]])
        d = drive_at.get(s, [])
        mean_drive_ev.append(sum(d) / len(d) if d else 0.0)

    out: dict[str, list] = {
        "step": steps,
        "adoption_frac": adoption_frac,
        "vocab_entropy": vocab_entropy,
        "n_new_items": n_new,
    }
    # l2 の全数値列を step 順に採用
    if l2 and l2.get("step"):
        row_of = {int(v): i for i, v in enumerate(l2["step"])}
        for col, vals in l2.items():
            if col == "step":
                continue
            out[col] = [vals[row_of[s]] if s in row_of else None for s in steps]
    # mean_drive は drive イベントがあればそれを、無ければ l2 由来を優先
    if has_drive:
        out["mean_drive"] = mean_drive_ev
    elif "mean_drive" not in out:
        out["mean_drive"] = [0.0] * n_steps
    return out


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


# --------------------------------------------------------------------------- #
# 早期警戒シグナル(EWS, Dakos 手順)
# --------------------------------------------------------------------------- #
def ews(series, detrend_bw: float = 0.1, window_frac: float = 0.5) -> dict:
    """ガウシアン detrend -> 移動窓の分散/lag-1自己相関 -> それぞれ Kendall τ。

    detrend_bw: ガウシアンカーネル帯域(系列長に対する割合)。
    window_frac: 移動窓の幅(系列長に対する割合)。
    返り値: {var_tau, ac1_tau, var_series, ac1_series, trend, resid}。
    """
    x = np.asarray(list(series), dtype=float)
    n = x.size
    nan = float("nan")
    if n < 4:
        return {"var_tau": nan, "ac1_tau": nan, "var_series": [],
                "ac1_series": [], "trend": x.tolist(), "resid": [], "n": int(n)}

    # ガウシアンカーネル detrend(自作)
    bw = max(detrend_bw * n, 1.0)
    idx = np.arange(n)
    trend = np.empty(n)
    for i in range(n):
        w = np.exp(-0.5 * ((idx - i) / bw) ** 2)
        trend[i] = float(np.dot(w, x) / np.sum(w))
    resid = x - trend

    win = max(int(round(window_frac * n)), 3)
    win = min(win, n)
    var_series: list = []
    ac1_series: list = []
    for start in range(0, n - win + 1):
        seg = resid[start:start + win]
        var_series.append(float(np.var(seg)))
        ac1_series.append(_lag1_ac(seg))

    t = np.arange(len(var_series), dtype=float)
    var_tau = _kendall_tau(t, np.asarray(var_series))
    ac1_tau = _kendall_tau(t, np.asarray(ac1_series))
    return {
        "var_tau": var_tau,
        "ac1_tau": ac1_tau,
        "var_series": var_series,
        "ac1_series": ac1_series,
        "trend": trend.tolist(),
        "resid": resid.tolist(),
        "n": int(n),
    }


def _lag1_ac(seg) -> float:
    s = np.asarray(seg, dtype=float)
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom <= 0:
        return 0.0
    return float(np.dot(s[:-1], s[1:]) / denom)


def _kendall_tau(a, b) -> float:
    """Kendall τ-b を O(n^2) で自作(tie 補正込み)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n < 2:
        return float("nan")
    concord = discord = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = da * db
            if s > 0:
                concord += 1
            elif s < 0:
                discord += 1
            else:
                if da == 0:
                    tie_a += 1
                if db == 0:
                    tie_b += 1
    n0 = n * (n - 1) / 2.0
    denom = math.sqrt((n0 - tie_a) * (n0 - tie_b))
    if denom == 0:
        return 0.0
    return (concord - discord) / denom


# --------------------------------------------------------------------------- #
# traits 回帰(OLS)
# --------------------------------------------------------------------------- #
def r2_traits(features: list[dict], trait_names, extra_targets=()) -> dict:
    """Y_external / Y_internal ~ traits の OLS を numpy.linalg.lstsq で解く。

    返り値: {"external": {r2, adj_r2, n, coef}, "internal": {...}, "trait_names"}。
    traits が無い / データ不足なら r2=None。

    extra_targets: 追加で R² を出したい被説明変数の列名タプル(例: ("Y_composite",))。
      既定 () では現行と完全に同一出力。指定した列名は返り値のキーとして追加される。
    """
    trait_names = list(trait_names or [])

    def fit(yname: str) -> dict:
        if not trait_names:
            return {"r2": None, "adj_r2": None, "n": 0}
        rows = [f for f in features
                if f.get(yname) is not None
                and all(isinstance(f.get(t), (int, float)) for t in trait_names)]
        n = len(rows)
        p = len(trait_names)
        if n <= p + 1:
            return {"r2": None, "adj_r2": None, "n": n}
        x = np.array([[1.0] + [float(f[t]) for t in trait_names] for f in rows])
        y = np.array([float(f[yname]) for f in rows])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else None
        return {"r2": r2, "adj_r2": adj, "n": n,
                "coef": dict(zip(["intercept"] + trait_names, beta.tolist()))}

    result = {
        "external": fit("Y_external"),
        "internal": fit("Y_internal"),
        "trait_names": trait_names,
    }
    for target in extra_targets:
        result[target] = fit(target)
    return result


# --------------------------------------------------------------------------- #
# semantic drift(M1c: 語形/意味が部分集団間で分岐)
# --------------------------------------------------------------------------- #
# すべて決定論の純関数(乱数不使用・stdlib のみ)。
def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距離(DP、O(len(a)·len(b)) 時間 / O(len(b)) 空間)。"""
    a = a or ""
    b = b or ""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def char_ngrams(text: str, n: int = 3) -> Counter:
    """文字 n-gram の頻度プロファイル(Counter)。

    長さ < n の非空文字列は「その文字列全体」を1つの gram として扱う
    (短い発話でも 0 プロファイルにせず比較可能にする)。空なら空 Counter。
    """
    s = text or ""
    if n <= 0 or not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def ngram_cosine(profile_a: Counter, profile_b: Counter) -> float:
    """2つの n-gram プロファイル間のコサイン**類似度**(0..1)。

    どちらかが空なら 0.0。距離が必要な箇所では 1 - この値 を使う。
    """
    if not profile_a or not profile_b:
        return 0.0
    common = set(profile_a) & set(profile_b)
    dot = sum(profile_a[k] * profile_b[k] for k in common)
    na = math.sqrt(sum(v * v for v in profile_a.values()))
    nb = math.sqrt(sum(v * v for v in profile_b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _conversation_adj(events: list[dict]) -> dict[int, set]:
    """会話グラフ(無向): speak の speaker→hearers と dm の speaker→to。"""
    adj: dict[int, set] = defaultdict(set)
    for e in events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if not isinstance(aid, int) or aid < 0:
            continue
        if k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    adj[aid].add(h)
                    adj[h].add(aid)
        elif k == "dm":
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                adj[aid].add(to)
                adj[to].add(aid)
    return adj


def communities(events: list[dict], n_agents: int,
                iterations: int = 20) -> dict[int, int]:
    """会話グラフ上の決定論的 label propagation。

    - ノード = range(n_agents) ∪ グラフ出現ノード。
    - 各反復で id 昇順に更新。近傍ラベルの最頻値を採用し、同数のときは
      現在ラベルを保持(安定化)、保持できないときは最小ラベルを採る。
    - 固定反復数(既定 20)で打ち切り(早期収束時は break)。
    返り値 {agent_id: community_id}。community_id は所属集団の最小メンバ id。
    """
    adj = _conversation_adj(events)
    nodes: set[int] = set(range(max(int(n_agents), 0)))
    nodes |= set(adj)
    for nb in adj.values():
        nodes |= nb
    labels = {u: u for u in nodes}
    ordered_nodes = sorted(nodes)
    for _ in range(max(int(iterations), 0)):
        changed = False
        for u in ordered_nodes:
            nb = adj.get(u)
            if not nb:
                continue
            cnt = Counter(labels[v] for v in nb)
            maxc = max(cnt.values())
            candidates = [lab for lab, c in cnt.items() if c == maxc]
            if labels[u] in candidates:
                best = labels[u]            # 同数なら現在ラベルを保持
            else:
                best = min(candidates)      # さもなくば最小ラベル
            if best != labels[u]:
                labels[u] = best
                changed = True
        if not changed:
            break
    # community_id = 所属集団の最小メンバ id に正準化
    members: dict[int, list] = defaultdict(list)
    for u, lab in labels.items():
        members[lab].append(u)
    canon: dict[int, int] = {}
    for mem in members.values():
        cid = min(mem)
        for u in mem:
            canon[u] = cid
    return canon


def _cluster_words(words: list[str], max_dist: int = 2) -> list[list[str]]:
    """編集距離 ≤ max_dist で連結される語を union-find でクラスタ化(推移的)。"""
    parent = {w: w for w in words}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    wl = list(words)
    for i in range(len(wl)):
        for j in range(i + 1, len(wl)):
            if edit_distance(wl[i], wl[j]) <= max_dist:
                union(wl[i], wl[j])
    groups: dict[str, list] = defaultdict(list)
    for w in wl:
        groups[find(w)].append(w)
    return [sorted(g) for g in sorted(groups.values(), key=lambda g: min(g))]


def _form_divergence(cluster: list[str], used_comm: dict[str, set]) -> float:
    """クラスタ内で **コミュニティ間で差異的に使われる** 変種ペアの平均編集距離。

    2 変種の使用コミュニティ集合が異なる(= 片方だけが使うコミュニティが存在する)
    ときに、その語形どうしの編集距離を平均する。単一変種/同一分布なら 0.0。
    """
    dists: list[int] = []
    n = len(cluster)
    for i in range(n):
        for j in range(i + 1, n):
            wa, wb = cluster[i], cluster[j]
            ca, cb = used_comm.get(wa, set()), used_comm.get(wb, set())
            if not ca or not cb:
                continue
            if ca != cb:
                dists.append(edit_distance(wa, wb))
    return round(sum(dists) / len(dists), 4) if dists else 0.0


def _mean_pairwise_ngram_distance(profiles: dict[int, Counter]) -> float:
    """コミュニティごとの n-gram プロファイル間の平均コサイン**距離**(1-類似度)。"""
    keys = sorted(profiles)
    if len(keys) < 2:
        return 0.0
    dists: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = ngram_cosine(profiles[keys[i]], profiles[keys[j]])
            dists.append(1.0 - sim)
    return round(sum(dists) / len(dists), 4) if dists else 0.0


def drift_metrics(events: list[dict], n_agents: int) -> dict[str, dict]:
    """語彙アイテム(coin 語の編集距離クラスタ)ごとの semantic drift 指標。

    返り値 {item: {form_divergence, context_divergence, n_communities}}。
      form_divergence   : コミュニティ間で分岐した語形(変種)の平均編集距離。
      context_divergence : 当該語を含む speak/sns_post テキストの文字3-gram
                           プロファイルのコミュニティ間コサイン距離(平均)。
      n_communities     : その語彙を使ったコミュニティ数。
    0 造語・単一コミュニティでも壊れない(空 dict / 0.0 を返す)。
    """
    comm = communities(events, n_agents)

    # coin 語(語形)を収集(media 発は creator=-1 で community を持たない)
    words_set: set[str] = set()
    for e in events:
        if e["kind"] in ("vocab_coin", "label_coin"):
            text = e["payload"].get("text")
            if text:
                words_set.add(text)
    if not words_set:
        return {}

    # 語 -> 使用コミュニティ集合 / 語 -> コミュニティ -> テキスト列
    used_comm: dict[str, set] = defaultdict(set)
    ctx: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        c = comm.get(aid)
        if c is None:
            continue
        if k in ("speak", "sns_post"):
            text = p.get("text") or ""
            items = p.get("items") or []
            hit: set[str] = set()
            for w in items:
                if w in words_set:
                    hit.add(w)
            for w in words_set:                       # 「含む」= 本文に部分一致
                if w and w not in hit and w in text:
                    hit.add(w)
            for w in hit:
                used_comm[w].add(c)
                ctx[w][c].append(text)
        elif k in ("vocab_coin", "label_coin"):
            w = p.get("text")
            if w in words_set and not p.get("media"):
                used_comm[w].add(c)                   # 造語者のコミュニティも計上

    out: dict[str, dict] = {}
    for cluster in _cluster_words(sorted(words_set), max_dist=2):
        comms: set[int] = set()
        for w in cluster:
            comms |= used_comm.get(w, set())
        form = _form_divergence(cluster, used_comm)
        profiles: dict[int, Counter] = {}
        for cm in comms:
            texts: list[str] = []
            for w in cluster:
                texts.extend(ctx.get(w, {}).get(cm, []))
            if texts:
                profiles[cm] = char_ngrams("".join(texts), 3)
        out[cluster[0]] = {
            "form_divergence": form,
            "context_divergence": _mean_pairwise_ngram_distance(profiles),
            "n_communities": len(comms),
        }
    return out
