"""P4: 観測解析のストリーミング化(row-group 逐次・有界メモリ)。

`measure.py` の純関数は events を list[dict] で受け取り、全件を RAM に載せる。本番想定の
~4.68 億件では 100-200GB 必要で破綻する。本モジュールは同じ集計結果を、
`measure.stream_events`(row-group/batch 逐次読み)を回して **有界メモリ** で再現する。

設計の要:
  - 集計状態はエンティティ(agent_id / item_id / step / community)で有界化された
    Counter・set・dict のみ。生イベント列は決してため込まない。
  - l1_events.parquet は step 非減少で追記される(logger の flush/part 結合も step 順)。
    純関数の `_ordered`(step 安定ソート)はファイル順の恒等置換になるため、
    ファイル順の逐次処理は純関数の各パスと同一列になる。
  - 端点計算(木の深さ・Kendall τ・n-gram など)は measure の葉ヘルパを **そのまま再利用**
    するので、出力は既存実装とバイト同一(小ランの新旧突合テストで担保)。

呼び出し側は measure の対応関数と同じ返り値を得る(scripts/analyze.py など)。
"""
from __future__ import annotations

from collections import Counter, defaultdict

from . import measure as m


# --------------------------------------------------------------------------- #
# 逐次読みの薄いラッパ(必要な列だけ射影して batch デコード面積を下げる)
# --------------------------------------------------------------------------- #
def _events(run_dir: str, columns):
    """必要列だけを射影した逐次イベント生成器(payload は要求時に dict 展開)。"""
    return m.stream_events(run_dir, columns=columns)


def max_agent_id(run_dir: str) -> int:
    """agent_id の最大値(agents.json 不在時の n_agents 決定用)。無ければ -1。"""
    mx = -1
    for e in _events(run_dir, ["agent_id"]):
        a = e["agent_id"]
        if a is not None and a > mx:
            mx = a
    return mx


# --------------------------------------------------------------------------- #
# agent 特徴量(measure.agent_features のストリーミング版)
# --------------------------------------------------------------------------- #
def agent_features(run_dir: str, agents_meta=None, traits=None,
                   y_weights=None, landmark_score=None) -> list[dict]:
    """measure.agent_features と同一出力を、2 パスの逐次読みで組み立てる。

    pass1: item 創出者 / media 起点 / SNS 投稿 item / ツール効果 / 制度・ルール /
           landmark 到着 を先に確定(伝播の創出者クレジットに必要)。
    pass2: 伝播・採用・会話・内省・欲求などの本走査。
    """
    cols = ["step", "agent_id", "kind", "payload"]

    # ---- pass1: 事前パス ----
    creator: dict = {}
    media: set = set()
    sns_posted: dict[int, set] = defaultdict(set)
    tool_feats: dict[int, dict] = defaultdict(lambda: {c: 0 for c in m._TOOL_COLS})
    inst_created: Counter = Counter()
    saw_institution = False
    rules_enacted: Counter = Counter()
    saw_rule = False
    arrivals: dict[int, Counter] = defaultdict(Counter)  # landmark 指定時のみ使用

    for e in _events(run_dir, cols):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        # item 創出者・media 起点(setdefault=最初の出現が勝つ。ファイル順=step 順)
        if k in ("vocab_coin", "label_coin"):
            iid = p.get("item_id")
            if iid is not None:
                creator.setdefault(iid, aid)
                if p.get("media"):
                    media.add(iid)
        # SNS 投稿 item
        if k == "sns_post":
            for it in p.get("items", []) or []:
                sns_posted[aid].add(it)
        # ツール効果(measure._tool_features と同一: 実行者 aid<0 のイベントは丸ごと無視)
        if aid is not None and aid >= 0:
            if k == "event_host":
                tool_feats[aid]["events_hosted"] += 1
            elif k == "event_attend":
                host = p.get("host")
                if isinstance(host, int) and host >= 0:
                    tool_feats[host]["event_attendees_drawn"] += 1
            elif k == "flyer_view":
                au = p.get("author")
                if isinstance(au, int) and au >= 0:
                    tool_feats[au]["flyer_unique_viewers"] += 1
            elif k == "group_found":
                tool_feats[aid]["groups_founded"] += 1
            elif k == "group_join":
                f = p.get("founder")
                if isinstance(f, int) and f >= 0:
                    tool_feats[f]["group_members_recruited"] += 1
            elif k == "proposal":
                tool_feats[aid]["proposals_made"] += 1
            elif k == "proposal_support":
                au = p.get("author")
                if isinstance(au, int) and au >= 0:
                    tool_feats[au]["signatures_gathered"] += 1
            elif k == "venture_open":
                tool_feats[aid]["ventures_opened"] += 1
            elif k == "venture_sale":
                tool_feats[aid]["venture_revenue"] += float(p.get("amount", 0) or 0)
        # 制度化(既定 OFF)・制度DSL(既定 ON)
        if k == "institution":
            saw_institution = True
            cb = p.get("created_by")
            if isinstance(cb, int) and cb >= 0:
                inst_created[cb] += 1
        elif k == "institution_rule":
            saw_rule = True
            pr = p.get("proposer")
            if isinstance(pr, int) and pr >= 0:
                rules_enacted[pr] += 1
        # 認知地図(landmark_score 指定時のみ)
        if landmark_score is not None and k == "arrive":
            node = p.get("node")
            if node is not None and aid is not None and aid >= 0:
                arrivals[aid][node] += 1

    has_institutions = bool(inst_created) or saw_institution
    has_rules = bool(rules_enacted) or saw_rule
    map_breadth: dict[int, int] = {}
    landmark_focus: dict[int, float] = {}
    if landmark_score is not None:
        for aid, nodes in arrivals.items():
            map_breadth[aid] = len(nodes)
            total = sum(nodes.values())
            if total > 0:
                landmark_focus[aid] = round(sum(
                    cnt * float(landmark_score.get(node, 0.0))
                    for node, cnt in nodes.items()) / total, 4)

    # ---- pass2: 本走査 ----
    c_trans: Counter = Counter()
    c_adopt: Counter = Counter()
    c_sns: Counter = Counter()
    partners: dict[int, set] = defaultdict(set)
    wb_count: Counter = Counter()
    belief_chars: Counter = Counter()
    compute: Counter = Counter()
    fires: Counter = Counter()
    opinion_moved: Counter = Counter()
    sns_likes: Counter = Counter()
    sns_reshares: Counter = Counter()
    last_from: dict[tuple, int] = {}
    seen: set = set()

    for e in _events(run_dir, cols):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if isinstance(aid, int) and aid >= 0:
            seen.add(aid)
        if k == "transmission":
            iid, frm, ch = p.get("item_id"), p.get("from"), p.get("channel")
            if isinstance(frm, int) and frm >= 0:
                c_trans[frm] += 1
            crt = creator.get(iid)
            if crt is not None and crt != frm:
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

    # ---- 対象 agent 集合 ----
    if agents_meta:
        meta_by_id = {int(a["id"]): a for a in agents_meta if "id" in a}
        ids = sorted(meta_by_id)
    else:
        meta_by_id = {}
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
        tf = tool_feats.get(aid)
        for col in m._TOOL_COLS:
            row[col] = tf[col] if tf else 0
        row["c_opinion_moved"] = round(opinion_moved[aid], 4)
        row["sns_likes_received"] = sns_likes[aid]
        row["sns_reshares_received"] = sns_reshares[aid]
        if has_institutions:
            row["institutions_created"] = inst_created[aid]
        if has_rules:
            row["rules_enacted"] = rules_enacted[aid]
        if landmark_score is not None:
            row["cognitive_map_breadth"] = map_breadth.get(aid, 0)
            row["landmark_focus"] = landmark_focus.get(aid, 0.0)
        for t, v in (traits.get(aid) or {}).items():
            row[t] = v
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
# item カスケード(measure.item_cascades のストリーミング版)
# --------------------------------------------------------------------------- #
def _t_half_from_stepcum(stepcum: list) -> int | None:
    """(step, 累積 distinct 採用者数)列(step 昇順)から t_half を再現する。

    measure._t_half は伝播ごとに (step, len(seen)) を並べるが、閾値到達 step は
    「各 step 終端の累積」だけで決まる(step 内順序に不変)。伝播の受信者は常に
    agent_id>=0 なので、step 単位の終端累積で厳密一致する。
    """
    if not stepcum:
        return None
    final = stepcum[-1][1]
    if final <= 0:
        return None
    half = 0.5 * final
    for step, c in stepcum:
        if c >= half:
            return step
    return stepcum[-1][0]


def item_cascades(run_dir: str) -> list[dict]:
    """measure.item_cascades と同一出力(size/creator/media/adopters/depth/
    channel_mix/t_half)を 1 パスの逐次読みで。伝播列を全保持せず、item ごとに
    有界な集計(Counter/set/step 終端累積)だけを持つ。"""
    cols = ["step", "agent_id", "kind", "payload"]
    creator: dict = {}
    media: set = set()
    size: Counter = Counter()
    chan: dict = defaultdict(Counter)
    edges: dict = defaultdict(set)
    adopters: dict = defaultdict(set)
    stepcum: dict = defaultdict(list)     # iid -> [(step, cum_after_step)]
    cur_step: dict = {}                    # iid -> 集計中の step

    for e in _events(run_dir, cols):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if k in ("vocab_coin", "label_coin"):
            iid = p.get("item_id")
            if iid is not None:
                creator.setdefault(iid, aid)
                if p.get("media"):
                    media.add(iid)
        elif k == "transmission":
            iid = p.get("item_id")
            if iid is None:
                continue
            s = e["step"]
            size[iid] += 1
            chan[iid][p.get("channel")] += 1
            edges[iid].add((p.get("from"), aid))
            # step 終端累積の記録(step が進んだら前 step 分を確定)
            prev = cur_step.get(iid)
            if prev is not None and s != prev:
                stepcum[iid].append((prev, len(adopters[iid])))
            cur_step[iid] = s
            if isinstance(aid, int) and aid >= 0:
                adopters[iid].add(aid)

    # 集計中だった最終 step を確定
    for iid, s in cur_step.items():
        stepcum[iid].append((s, len(adopters[iid])))

    out: list[dict] = []
    for iid in sorted(set(creator) | set(size)):
        ad = sorted(adopters.get(iid, ()))
        out.append({
            "item_id": iid,
            "creator": creator.get(iid),
            "media": iid in media,
            "size": size.get(iid, 0),
            "n_adopters": len(ad),
            "adopters": ad,
            "depth": m._tree_depth(edges.get(iid, set())),
            "channel_mix": dict(chan.get(iid, {})),
            "t_half": _t_half_from_stepcum(stepcum.get(iid, [])),
        })
    return out


# --------------------------------------------------------------------------- #
# ネットワーク窓(measure.network_windows のストリーミング版)
# --------------------------------------------------------------------------- #
def _window_metrics(w: int, window: int, adj: dict, prev_top5: list) -> tuple:
    """1 窓分の指標 dict と、次窓の churn 用 top5 を返す。measure と同一式。"""
    nodes = sorted(adj)
    degrees = {u: len(adj[u]) for u in nodes}
    n_edges = sum(degrees.values()) // 2
    n_nodes = len(nodes)
    mean_degree = (2 * n_edges / n_nodes) if n_nodes else 0.0
    top1 = max(nodes, key=lambda u: (degrees[u], -u)) if nodes else None
    top5 = [u for u, _ in sorted(degrees.items(),
                                 key=lambda kv: (-kv[1], kv[0]))[:5]]
    churn = len(set(top5) - set(prev_top5)) if prev_top5 else 0
    start = w * window
    row = {
        "window": w,
        "step_start": start,
        "step_end": start + window,
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "mean_degree": mean_degree,
        "clustering": m._avg_clustering(adj),
        "top1_agent": top1,
        "centrality_churn": churn,
    }
    return row, top5


def network_windows(run_dir: str, window: int = 24) -> list[dict]:
    """measure.network_windows と同一出力を 1 パスで。会話グラフ(speak/dm)の隣接は
    「集計中の窓」1 つ分だけ RAM に持ち、窓が進むたびに確定・解放する。空窓も measure と
    同じく生成する(step 全体の max で末尾まで埋める)。"""
    cols = ["step", "agent_id", "kind", "payload"]
    windows: list[dict] = []
    prev_top5: list = []
    cw = 0                      # 集計中の窓 index
    adj: dict = defaultdict(set)
    max_step = -1
    started = False

    def emit(w: int, a: dict) -> None:
        nonlocal prev_top5
        row, top5 = _window_metrics(w, window, a, prev_top5)
        windows.append(row)
        prev_top5 = top5

    for e in _events(run_dir, cols):
        s = e["step"]
        if s > max_step:
            max_step = s
        started = True
        k = e["kind"]
        if k not in ("speak", "dm"):
            continue
        we = s // window
        if we > cw:
            emit(cw, adj)                       # 集計中の窓を確定
            for w in range(cw + 1, we):
                emit(w, {})                     # 間の空窓
            cw = we
            adj = defaultdict(set)
        aid = e["agent_id"]
        p = e["payload"]
        if k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    adj[aid].add(h)
                    adj[h].add(aid)
        else:  # dm
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                adj[aid].add(to)
                adj[to].add(aid)

    if not started:
        return []
    emit(cw, adj)                               # 最後の集計窓
    last_w = max_step // window
    for w in range(cw + 1, last_w + 1):
        emit(w, {})                             # 末尾の空窓
    return windows


# --------------------------------------------------------------------------- #
# 集団時系列(measure.collective_series のストリーミング版)
# --------------------------------------------------------------------------- #
def collective_series(run_dir: str, l2, n_agents: int) -> dict[str, list]:
    """measure.collective_series と同一出力。step 窓ごとの部分集計を逐次で畳み込み、
    語彙/採用の累積は有界(cum_vocab≤item 数・cum_adopters≤agent 数)に保つ。"""
    if l2 and l2.get("step"):
        steps = sorted(int(s) for s in l2["step"])
    else:
        mx = -1
        for e in _events(run_dir, ["step"]):
            if e["step"] > mx:
                mx = e["step"]
        steps = list(range(mx + 1))
    index = {s: i for i, s in enumerate(steps)}
    n_steps = len(steps)

    new_items = [0] * n_steps
    adopt_at: dict[int, set] = defaultdict(set)
    vocab_at: dict[int, Counter] = defaultdict(Counter)
    drive_at: dict[int, list] = defaultdict(lambda: [0.0, 0])  # [granted_sum, total]

    for e in _events(run_dir, ["step", "agent_id", "kind", "payload"]):
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
            d = drive_at[s]
            d[0] += 1.0 if p.get("granted") else 0.0
            d[1] += 1

    cum_adopters: set = set()
    cum_vocab: Counter = Counter()
    adoption_frac: list = []
    vocab_entropy: list = []
    n_new: list = []
    mean_drive_ev: list = []
    has_drive = bool(drive_at)
    for s in steps:
        cum_adopters |= adopt_at.get(s, set())
        for iid, c in vocab_at.get(s, {}).items():
            cum_vocab[iid] += c
        adoption_frac.append(len(cum_adopters) / n_agents if n_agents else 0.0)
        vocab_entropy.append(m._entropy(cum_vocab))
        n_new.append(new_items[index[s]])
        d = drive_at.get(s)
        mean_drive_ev.append(d[0] / d[1] if d and d[1] else 0.0)

    out: dict[str, list] = {
        "step": steps,
        "adoption_frac": adoption_frac,
        "vocab_entropy": vocab_entropy,
        "n_new_items": n_new,
    }
    if l2 and l2.get("step"):
        row_of = {int(v): i for i, v in enumerate(l2["step"])}
        for col, vals in l2.items():
            if col == "step":
                continue
            out[col] = [vals[row_of[s]] if s in row_of else None for s in steps]
    if has_drive:
        out["mean_drive"] = mean_drive_ev
    elif "mean_drive" not in out:
        out["mean_drive"] = [0.0] * n_steps
    return out


# --------------------------------------------------------------------------- #
# コミュニティ(measure.communities のストリーミング版)
# --------------------------------------------------------------------------- #
def _conversation_adj(run_dir: str) -> dict[int, set]:
    """会話グラフ(無向): speak の speaker→hearers と dm の speaker→to。逐次構築。"""
    adj: dict[int, set] = defaultdict(set)
    for e in _events(run_dir, ["agent_id", "kind", "payload"]):
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


def _labels_from_adj(adj: dict, n_agents: int, iterations: int = 20) -> dict[int, int]:
    """measure.communities と同一の決定論 label propagation(adj は逐次で構築済み)。"""
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
                best = labels[u]
            else:
                best = min(candidates)
            if best != labels[u]:
                labels[u] = best
                changed = True
        if not changed:
            break
    members: dict[int, list] = defaultdict(list)
    for u, lab in labels.items():
        members[lab].append(u)
    canon: dict[int, int] = {}
    for mem in members.values():
        cid = min(mem)
        for u in mem:
            canon[u] = cid
    return canon


def communities(run_dir: str, n_agents: int, iterations: int = 20) -> dict[int, int]:
    """measure.communities と同一出力(逐次で adj を作り、同じ label 伝播を回す)。"""
    return _labels_from_adj(_conversation_adj(run_dir), n_agents, iterations)


# --------------------------------------------------------------------------- #
# semantic drift(measure.drift_metrics のストリーミング版)
# --------------------------------------------------------------------------- #
def drift_metrics(run_dir: str, n_agents: int) -> dict[str, dict]:
    """measure.drift_metrics と同一出力。会話グラフ→コミュニティ→語彙変種の分岐を、
    語(coin 語)関連のテキストだけ有界に保持して算出する。"""
    # pass1: 会話グラフ(コミュニティ用)+ coin 語の収集
    adj: dict[int, set] = defaultdict(set)
    words_set: set[str] = set()
    for e in _events(run_dir, ["agent_id", "kind", "payload"]):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if isinstance(aid, int) and aid >= 0:
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
        if k in ("vocab_coin", "label_coin"):
            text = p.get("text")
            if text:
                words_set.add(text)
    if not words_set:
        return {}
    comm = _labels_from_adj(adj, n_agents)

    # pass2: 語 -> 使用コミュニティ / 語 -> コミュニティ -> テキスト
    used_comm: dict[str, set] = defaultdict(set)
    ctx: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for e in _events(run_dir, ["agent_id", "kind", "payload"]):
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
            for w in words_set:
                if w and w not in hit and w in text:
                    hit.add(w)
            for w in hit:
                used_comm[w].add(c)
                ctx[w][c].append(text)
        elif k in ("vocab_coin", "label_coin"):
            w = p.get("text")
            if w in words_set and not p.get("media"):
                used_comm[w].add(c)

    out: dict[str, dict] = {}
    for cluster in m._cluster_words(sorted(words_set), max_dist=2):
        comms: set[int] = set()
        for w in cluster:
            comms |= used_comm.get(w, set())
        form = m._form_divergence(cluster, used_comm)
        profiles: dict[int, Counter] = {}
        for cm in comms:
            texts: list[str] = []
            for w in cluster:
                texts.extend(ctx.get(w, {}).get(cm, []))
            if texts:
                profiles[cm] = m.char_ngrams("".join(texts), 3)
        out[cluster[0]] = {
            "form_divergence": form,
            "context_divergence": m._mean_pairwise_ngram_distance(profiles),
            "n_communities": len(comms),
        }
    return out


# --------------------------------------------------------------------------- #
# 造語文脈(scripts/analyze.coinage_contexts のストリーミング版)
# --------------------------------------------------------------------------- #
def coinage_contexts(run_dir: str) -> list[dict]:
    """造語(vocab_coin, media 発を除く)の発生文脈一覧。analyze.coinage_contexts と同一。"""
    out: list[dict] = []
    for e in _events(run_dir, ["step", "agent_id", "kind", "payload"]):
        if e["kind"] != "vocab_coin":
            continue
        p = e["payload"]
        if p.get("media"):
            continue
        out.append({
            "step": e["step"], "agent_id": e["agent_id"],
            "item_id": p.get("item_id"), "text": p.get("text"),
            "fire_reason": p.get("fire_reason"),
            "drive": p.get("drive"),
            "place": p.get("place"),
            "company_ids": p.get("company_ids"),
            "saw_feed": p.get("saw_feed"),
            "adopted_n": p.get("adopted_n"),
            "recent_mem": p.get("recent_mem"),
        })
    return out


# --------------------------------------------------------------------------- #
# ツール使用・効果(scripts/analyze.tool_usage のストリーミング版)
# --------------------------------------------------------------------------- #
_TOOL_KINDS = ("event_host", "event_attend", "flyer_post", "flyer_view",
               "flyer_expire", "group_found", "group_join", "proposal",
               "proposal_support", "proposal_passed", "venture_open",
               "venture_sale", "venture_close")


def tool_usage(run_dir: str) -> dict:
    """analyze.tool_usage と同一出力を 1 パスで。集計は kind/agent/step/エンティティで
    有界(生イベント列を保持しない)。ファイル順は step 昇順なので pure の
    sorted(events, step) と同一列。"""
    totals = {k: 0 for k in _TOOL_KINDS}
    users = defaultdict(lambda: defaultdict(int))
    timeline = defaultdict(lambda: defaultdict(int))
    events_map, groups_map, proposals_map, ventures_map = {}, {}, {}, {}

    for e in _events(run_dir, ["step", "agent_id", "kind", "payload"]):
        k, aid, p, s = e["kind"], e["agent_id"], e["payload"], e["step"]
        if k not in totals:
            continue
        totals[k] += 1
        timeline[s][k] += 1
        if aid is not None and aid >= 0:
            users[k][aid] += 1
        if k == "event_host":
            events_map[p.get("event_id")] = {
                "event_id": p.get("event_id"), "host": aid, "title": p.get("title"),
                "step": s, "attendees": 0}
        elif k == "event_attend":
            ev = events_map.get(p.get("event_id"))
            if ev is not None:
                ev["attendees"] += 1
        elif k == "group_found":
            groups_map[p.get("group_id")] = {
                "group_id": p.get("group_id"), "name": p.get("name"),
                "founder": aid, "step": s, "members_recruited": 0}
        elif k == "group_join":
            g = groups_map.get(p.get("group_id"))
            if g is not None:
                g["members_recruited"] += 1
        elif k == "proposal":
            proposals_map[p.get("proposal_id")] = {
                "proposal_id": p.get("proposal_id"), "author": aid,
                "text": p.get("text"), "step": s, "signatures": 1, "passed": False}
        elif k == "proposal_support":
            pr = proposals_map.get(p.get("proposal_id"))
            if pr is not None:
                pr["signatures"] += 1
        elif k == "proposal_passed":
            pr = proposals_map.get(p.get("proposal_id"))
            if pr is not None:
                pr["passed"] = True
                pr["signatures"] = p.get("supporters", pr["signatures"])
        elif k == "venture_open":
            ventures_map[aid] = {"owner": aid, "name": p.get("name"), "step": s,
                                 "sales": 0, "revenue": 0.0, "closed": False}
        elif k == "venture_sale":
            v = ventures_map.get(aid)
            if v is not None:
                v["sales"] += 1
                v["revenue"] += float(p.get("amount", 0) or 0)
        elif k == "venture_close":
            v = ventures_map.get(aid)
            if v is not None:
                v["closed"] = True

    top_users = {k: sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                 for k, d in users.items() if d}
    return {
        "totals": totals,
        "top_users": top_users,
        "timeline": {str(s): dict(v) for s, v in sorted(timeline.items())},
        "events": list(events_map.values()),
        "groups": list(groups_map.values()),
        "proposals": list(proposals_map.values()),
        "ventures": list(ventures_map.values()),
    }


# --------------------------------------------------------------------------- #
# 物流・在庫の集計(スライス①+②。在庫水準・品切れ率・補充回数・商品別売上)
# --------------------------------------------------------------------------- #
_GOODS_KINDS = ("delivery_trip", "restock", "stock_low", "stock_out")


def goods_usage(run_dir: str) -> dict:
    """物流の実体化(goods)の逐次集計を 1 パスで。集計は kind/cat/item で有界(生イベント列を保持しない)。

    - totals: delivery_trip(配送)/restock(補充到着)/stock_low(在庫僅少)/stock_out(品切れ)件数。
    - restock_qty: 補充で戻した総数量。restock_failed: 発注(配送)されたが到着しなかった数(封鎖=補充失敗の代理)。
    - stockout_rate: 品切れ / (品切れ + 商品つき spend)= 実在庫由来の購入不成立率。
    - by_cat: カテゴリ別の各件数 + 商品つき売上(sold)。
    - top_items: 商品別売上(spend.item)の上位。
    stock_out は実在庫版(payload src=inventory)のみ数える(commerce の在館数代理は除外=意味の混線回避)。"""
    totals = {k: 0 for k in _GOODS_KINDS}
    restock_qty = 0
    by_cat: dict = defaultdict(lambda: {"delivery_trip": 0, "restock": 0,
                                        "stock_low": 0, "stock_out": 0, "sold": 0})
    items_sold: Counter = Counter()
    sold_total = 0

    for e in _events(run_dir, ["agent_id", "kind", "payload"]):
        k, p = e["kind"], e["payload"]
        if k == "stock_out":
            if p.get("src") != "inventory":         # 実在庫版のみ(commerce の在館数代理は除外)
                continue
            totals["stock_out"] += 1
            cat = p.get("cat")
            if cat is not None:
                by_cat[cat]["stock_out"] += 1
        elif k in totals:
            totals[k] += 1
            cat = p.get("cat")
            if cat is not None:
                by_cat[cat][k] += 1
            if k == "restock":
                restock_qty += int(p.get("qty", 0) or 0)
        elif k == "spend":
            item = p.get("item")
            if item is None:
                continue
            items_sold[item] += 1
            sold_total += 1
            cat = p.get("cat")
            if cat is not None:
                by_cat[cat]["sold"] += 1

    denom = totals["stock_out"] + sold_total
    return {
        "totals": totals,
        "restock_qty": restock_qty,
        "restock_failed": max(0, totals["delivery_trip"] - totals["restock"]),
        "stockout_rate": round(totals["stock_out"] / denom, 4) if denom else 0.0,
        "by_cat": {c: dict(v) for c, v in sorted(by_cat.items())},
        "items_sold_total": sold_total,
        "top_items": sorted(items_sold.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }
