#!/usr/bin/env python
"""第18バッチ①: 自然発生コミュニティの観察・分析(後処理・L1 を読むだけ)。

    python scripts/analyze_communities.py runs/<name> [--window-days 7]

ユーザー要望(2026-07-11):エージェント同士を無理に組織にせず、**自然に形成された
コミュニティの動き(誕生・成長・合流・分裂・消滅)と構造**を観察・分析する。

方針(docs/research/community-detection.md に忠実。検収 2026-07-11 の修正込み):
  - 検出グラフのエッジ = 対面会話(speak/hear)・DM・SNS 反応(like/reshare)・
    イベント共同参加(event_attend の二部射影)。反復回数×サリエンス重み α で
    合成した「重み付き無向1層」。**合成重みが min_weight(--min-weight、既定 2.0)
    未満のエッジは足切り**する——一回きりの偶発的接触(すれ違い・同席)は紐帯で
    ない(反復相互作用=紐帯の社会学的定義)。face 1回=1.0 なので、既定では
    「対面2回 or DM1回 or SNS反応+α」で初めて紐帯になる。
    **組織所属(会社・学校)と found_group/group_join は検出に入れない**
    (ユーザーの「無理に組織にしない」の操作化)。参照分割としてのみ使い、
    自然コミュニティとの一致/乖離を NMI・E-I index で測る。
  - 検出の正準器 = networkx.louvain_communities(seed=0, weight="weight")。
    交差確認 = 既存 src/society/observer/measure.communities()(決定論 LPA。
    measure.py は import のみ・一切変更しない)を同一グラフで併走し Q を併記。
    ※ 実測(300体×100日)で LPA は密グラフを全体1コミュニティに潰した
    (研究 §1.3 の既知の退化)ため、重み対応の louvain を正準に昇格した。
  - 足切り後に次数 0 のノードは「無所属」= どのコミュニティにも数えない
    (ビューワー側はグレー表示。communities.json のスキーマは不変)。
  - 時間発展 = 窓ごとに検出 → 窓間 Jaccard マッチング(τ=0.30・タイは最小 id)→
    Palla 2007 のライフサイクル語彙(誕生/成長/縮小/合流/分裂/消滅)へ機械分類。
  - 集団の達成 = host_event の集客・found_group・open_venture/売上・提案/採択(制度化)・
    文化カスケード規模を、その窓のコミュニティメンバーに帰属させて集計。

決定論(絶対条件):ノード・エッジは必ずソート順に追加、seed 固定、community_id は
メンバー最小 id で正準化、Jaccard マッチングのタイは最小 id、成果帰属の多数決タイも
最小 id。出力 JSON は json.dumps(sort_keys=True, ensure_ascii=False) でバイト同一。

出力:
  runs/<name>/communities.json         … 窓→コミュニティ→メンバー/測度/ライフサイクル/達成
  runs/<name>/communities_report.md    … 日本語の要約表 + ライフサイクル物語 + 達成一覧
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

# Windows cp932 対策(コンソール出力の文字化け・例外を防ぐ)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import networkx as nx  # noqa: E402
from society.observer import measure as m  # noqa: E402

# --------------------------------------------------------------------------- #
# 定数(研究 §2・§4 の推奨。決定論のため config ではなくスクリプト定数に固定)
# --------------------------------------------------------------------------- #
STEPS_PER_DAY = 144          # 1 step = 10 sim 分 → 1440/10 = 144(observe.py と同じ)
SEED = 0                     # louvain の乱数シード(決定論の4点セット其の二)
GAMMA = 1.0                  # louvain resolution(既定1)
TAU = 0.30                   # 窓間 Jaccard マッチング閾値(研究 §2.1)
MIN_WEIGHT = 2.0             # 紐帯とみなす最小合成重み(未満のエッジは足切り)
CASCADE_MIN_ADOPTERS = 3     # 文化カスケードを「達成」と数える最小採用者数(1→多)

# チャネル・サリエンス重み(研究 §4.2)。合成重み = Σ_ch α_ch × 反復回数。
# 頻度正規化は行わない(検収 2026-07-11: 正規化は密グラフで一回きりの偶発接触を
# 過大評価した)。代わりに min_weight の足切りで「反復相互作用のみ紐帯」を担保する。
ALPHA = {"face": 1.0, "dm": 2.0, "sns": 1.5, "event": 1.0}
# SNS 反応の承認強度(reshare > like)
SNS_W = {"sns_like": 0.5, "sns_reshare": 1.0}
# 有向グラフ(PageRank=影響力)の辺重み。辺は「影響の起点(話者/著者/主催/DM 送信者)」
# に向けて張る(= 起点ほど PageRank が高い → 創発リーダーの代理)。
DIR_W = {"face": 1.0, "dm": 2.0, "sns_like": 0.5, "sns_reshare": 1.0, "event": 1.0}


# --------------------------------------------------------------------------- #
# 補助(決定論の丸め・Jaccard)
# --------------------------------------------------------------------------- #
def _r(x: float, nd: int = 6) -> float:
    """浮動小数を固定桁で丸める(プラットフォーム差の混入を避ける)。"""
    return round(float(x), nd)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# 参照分割(組織所属)— 検出には入れず、突き合わせにのみ使う
# --------------------------------------------------------------------------- #
def org_labels(agents: list[dict]) -> dict[int, str]:
    """agents.json の work_building を「会社・学校=組織」ラベルとして返す。

    work_building が空のエージェント(無職・来街者など)は参照分割に含めない
    (None は除外)。実在個人名は一切出力しない(id と建物 id のみを扱う)。
    """
    out: dict[int, str] = {}
    for a in agents or []:
        if "id" not in a:
            continue
        wb = a.get("work_building")
        if wb:
            out[int(a["id"])] = str(wb)
    return out


# --------------------------------------------------------------------------- #
# 窓ごとのグラフ構築(重み付き合成・無向1層 + 有向 PageRank 用グラフ)
# --------------------------------------------------------------------------- #
def _add(d: dict, u: int, v: int, w: float) -> None:
    """無向ペア(min,max)へ重みを加算(自己ループは無視)。"""
    if u == v:
        return
    key = (u, v) if u < v else (v, u)
    d[key] += w


def build_window_graph(win_events: list[dict],
                       min_weight: float = MIN_WEIGHT) -> dict:
    """窓内イベント → チャネル別の生重み → 合成・足切りした無向重み + 有向重み。

    合成重み = Σ_ch α_ch × 反復回数。min_weight 未満のエッジは「一回きりの
    偶発的接触=紐帯でない」として除去する(足切り後に次数 0 のノードは
    nodes に含まれない=無所属)。

    返り値:
      W       : {(u,v): weight}  合成・無向・足切り後(検出/測度用)
      adj     : {node: set(node)} 足切り後の隣接(E-I / 密度用の非加重)
      dir_w   : {(u,v): weight}  有向(u→v)。PageRank(影響力)用(足切りなし)
      nodes   : 足切り後のグラフに現れたノード集合(紐帯を1本以上持つ者)
    """
    per: dict[str, dict] = {ch: defaultdict(float) for ch in ALPHA}
    dir_raw: dict[tuple, float] = defaultdict(float)
    coattend: dict[tuple, set] = defaultdict(set)   # (host,event_id) -> 参加者集合

    for e in win_events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if not isinstance(aid, int) or aid < 0:
            continue
        if k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    _add(per["face"], aid, h, 1.0)
                    dir_raw[(h, aid)] += DIR_W["face"]      # 聞き手→話者
        elif k == "hear":
            sp = p.get("speaker")
            if isinstance(sp, int) and sp >= 0 and sp != aid:
                _add(per["face"], aid, sp, 1.0)
                dir_raw[(aid, sp)] += DIR_W["face"]          # 聞き手→話者
        elif k == "dm":
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                _add(per["dm"], aid, to, 1.0)
                dir_raw[(to, aid)] += DIR_W["dm"]            # 受信→送信(起点=送信者)
        elif k in ("sns_like", "sns_reshare"):
            au = p.get("author")
            if isinstance(au, int) and au >= 0 and au != aid:
                _add(per["sns"], aid, au, SNS_W[k])
                dir_raw[(aid, au)] += DIR_W[k]               # 反応者→著者
        elif k == "event_attend":
            eid = p.get("event_id")
            host = p.get("host")
            if eid is not None and isinstance(host, int) and host >= 0:
                coattend[(host, eid)].add(aid)
                coattend[(host, eid)].add(host)              # 主催者も共参加に含める
                dir_raw[(aid, host)] += DIR_W["event"]       # 参加者→主催者
        elif k == "event_host":
            eid = p.get("event_id")
            if eid is not None:
                coattend[(aid, eid)].add(aid)

    # イベント共参加=二部射影(大イベントが完全グラフで支配しないよう 1/(k-1) で割る)
    for (_host, _eid), members in coattend.items():
        ms = sorted(members)
        k = len(ms)
        if k < 2:
            continue
        w = 1.0 / (k - 1)
        for i in range(k):
            for j in range(i + 1, k):
                _add(per["event"], ms[i], ms[j], w)

    # サリエンス α で合成(反復回数×α)→ min_weight 未満のエッジを足切り
    raw: dict[tuple, float] = defaultdict(float)
    for ch, edges in per.items():
        a = ALPHA[ch]
        for key, w in edges.items():
            raw[key] += a * w
    W = {key: w for key, w in raw.items() if w >= min_weight}

    adj: dict[int, set] = defaultdict(set)
    for (u, v) in W:
        adj[u].add(v)
        adj[v].add(u)
    nodes = set(adj)
    return {"W": dict(W), "adj": adj, "dir_w": dict(dir_raw), "nodes": nodes}


# --------------------------------------------------------------------------- #
# 検出(正準=seeded Louvain / 交差確認=自作決定論 LPA)
# --------------------------------------------------------------------------- #
def _synthetic_events(W: dict) -> list[dict]:
    """合成グラフの無向辺を measure.communities() が読める疑似 speak 列に変換。

    measure._conversation_adj は speak(speaker→hearers)から隣接を作るので、
    辺 (a,b) を {"kind":"speak","agent_id":a,"payload":{"hearers":[b]}} で表せば、
    重みを無視した同一の隣接が復元できる(= 合成グラフ上で LPA を回せる)。
    """
    out: list[dict] = []
    for (a, b) in sorted(W):
        out.append({"kind": "speak", "agent_id": a, "payload": {"hearers": [b]}})
    return out


def detect(graph: dict, n_agents: int) -> dict:
    """足切り済み合成グラフを検出。正準=louvain / 交差確認=決定論 LPA。

    決定論の担保: ノード・エッジをソート順で add、seed=SEED 固定、
    community_id はメンバー最小 id に正準化。孤立ノード(足切り後に次数 0)は
    G に含まれない=無所属(part に載らない)。

    返り値:
      part      : {node: community_id}  正準 louvain(community_id=最小メンバ id)
      G         : nx.Graph(重み付き・紐帯を持つノードのみ)
      mod_louv  : 正準(louvain)分割の modularity
      mod_lpa   : 交差確認(決定論 LPA)分割の modularity
      n_lpa     : LPA のコミュニティ数(size>=2)
    """
    W = graph["W"]
    # 決定論の重み付きグラフ(ノードは id 昇順、辺はソート順で add)
    G = nx.Graph()
    G.add_nodes_from(sorted(graph["nodes"]))
    for (u, v) in sorted(W):
        G.add_edge(u, v, weight=_r(W[(u, v)]))

    # 正準: louvain(seed 固定・resolution=GAMMA)→ 最小メンバ id へ正準化
    part: dict[int, int] = {}
    mod_louv = 0.0
    if G.number_of_edges():
        try:
            louv = nx.community.louvain_communities(G, weight="weight",
                                                    resolution=GAMMA, seed=SEED)
            mod_louv = nx.community.modularity(G, louv, weight="weight")
        except Exception:
            louv = []
        for s in louv:
            cid = min(s)
            for node in s:
                part[node] = cid

    # 交差確認: 決定論 LPA(measure.communities を疑似 speak 経由で再利用)
    mod_lpa = 0.0
    n_lpa = 0
    if W:
        lpa = m.communities(_synthetic_events(W), 0)   # n_agents=0: グラフ出現ノードのみ
        groups: dict[int, set] = defaultdict(set)
        for node, c in lpa.items():
            groups[c].add(node)
        lpa_sets = [groups[c] for c in sorted(groups)]
        try:
            mod_lpa = nx.community.modularity(G, lpa_sets, weight="weight")
        except Exception:
            mod_lpa = 0.0
        n_lpa = sum(1 for s in lpa_sets if len(s) >= 2)

    return {"part": part, "G": G, "mod_louv": mod_louv,
            "mod_lpa": mod_lpa, "n_lpa": n_lpa}


# --------------------------------------------------------------------------- #
# 構造測度(E-I index / 内部密度 / PageRank リーダー / NMI・E-I(org))
# --------------------------------------------------------------------------- #
def ei_and_density(W: dict, part: dict[int, int]) -> tuple[dict, dict]:
    """コミュニティ別 E-I index と内部辺数を一括計算(非加重・決定的)。

    E-I = (EL - IL)/(EL + IL)。IL=内部辺数、EL=外向き辺数。-1=結束的/+1=外向き。
    """
    IL: Counter = Counter()
    EL: Counter = Counter()
    for (u, v) in W:
        cu, cv = part.get(u), part.get(v)
        if cu is not None and cu == cv:
            IL[cu] += 1
        else:
            if cu is not None:
                EL[cu] += 1
            if cv is not None:
                EL[cv] += 1
    ei: dict[int, float] = {}
    for c in set(IL) | set(EL):
        tot = IL[c] + EL[c]
        ei[c] = (EL[c] - IL[c]) / tot if tot else 0.0
    return ei, dict(IL)


def ei_index_of_labels(W: dict, labels: dict[int, str]) -> float:
    """参照分割(組織)境界の E-I index(全体1値)。高い=自然な紐帯が組織線を跨ぐ。"""
    il = el = 0
    for (u, v) in W:
        lu, lv = labels.get(u), labels.get(v)
        if lu is None or lv is None:
            continue
        if lu == lv:
            il += 1
        else:
            el += 1
    tot = il + el
    return (el - il) / tot if tot else 0.0


def nmi(la: dict[int, object], lb: dict[int, object]) -> float:
    """2 分割の正規化相互情報量 NMI = 2 I(X;Y)/(H(X)+H(Y))(共通ノード上・決定的)。"""
    nodes = sorted(set(la) & set(lb))
    n = len(nodes)
    if n == 0:
        return 0.0
    ca = Counter(la[x] for x in nodes)
    cb = Counter(lb[x] for x in nodes)
    joint = Counter((la[x], lb[x]) for x in nodes)

    def _h(c: Counter) -> float:
        return -sum((v / n) * math.log(v / n) for v in c.values() if v > 0)

    ha, hb = _h(ca), _h(cb)
    info = 0.0
    for (a, b), v in joint.items():
        pab = v / n
        info += pab * math.log(pab / ((ca[a] / n) * (cb[b] / n)))
    denom = ha + hb
    if denom <= 0:
        return 1.0            # 双方が単一クラスタ = 完全一致とみなす
    return 2 * info / denom


def _pagerank(nodes: set, edges: dict, alpha: float = 0.85,
              max_iter: int = 100, tol: float = 1e-10) -> dict[int, float]:
    """依存フリーの決定論 PageRank(power iteration)。

    networkx 3.6 の nx.pagerank は scipy 必須(本プロジェクトの依存に無い)ため、
    同じ力学(重み付き遷移+dangling 質量の一様再配分)を純 Python で実装する。
    決定論: ノード・辺をソート順に走査、固定反復数+固定許容誤差、乱数なし。
    edges: {(u,v): w} 有向 u→v。
    """
    ns = sorted(nodes)
    n = len(ns)
    if n == 0:
        return {}
    out_w = {u: 0.0 for u in ns}                  # 出次の重み和
    succ: dict[int, list] = {u: [] for u in ns}
    for (u, v) in sorted(edges):
        w = edges[(u, v)]
        if u in out_w and v in out_w and w > 0:
            succ[u].append((v, w))
            out_w[u] += w
    pr = {u: 1.0 / n for u in ns}
    for _ in range(max_iter):
        dangling = sum(pr[u] for u in ns if out_w[u] == 0.0)
        base = (1.0 - alpha) / n + alpha * dangling / n
        nxt = {u: base for u in ns}
        for u in ns:
            if out_w[u] > 0:
                pu = alpha * pr[u] / out_w[u]
                for v, w in succ[u]:
                    nxt[v] += pu * w
        err = sum(abs(nxt[u] - pr[u]) for u in ns)
        pr = nxt
        if err < n * tol:
            break
    return pr


def pagerank_leaders(graph: dict, comm_members: dict[int, set]) -> dict[int, tuple]:
    """有向グラフの PageRank で各コミュニティのリーダー(上位・同点は最小 id)。"""
    dir_w = graph["dir_w"]
    nodes: set = set(graph["nodes"])
    for (u, v) in dir_w:                          # 有向相互作用の端点も母集団に含める
        nodes.add(u)
        nodes.add(v)
    pr = _pagerank(nodes, dir_w) if dir_w else {}
    out: dict[int, tuple] = {}
    for cid, members in comm_members.items():
        # 最大 PageRank(同点は最小 id)
        best = min(sorted(members), key=lambda a: (-pr.get(a, 0.0), a))
        out[cid] = (best, _r(pr.get(best, 0.0)))
    return out


# --------------------------------------------------------------------------- #
# ⑤社会ネットワークの変化(第31バッチ W3)= 窓ごとの大域指標・tie decay・窓遷移
#   既存 build_window_graph の足切り済みグラフ(W/adj/nodes)を再利用する
#   (二重実装しない)。クラスタ係数等は純Python(networkx 等の新規依存を足さない)。
# --------------------------------------------------------------------------- #
def _gini_seq(values) -> float:
    """数列の gini(0=平等..1=集中)。空/総和0 は 0。次数分布の集中度に使う。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    tot = sum(xs)
    if n == 0 or tot <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


def avg_clustering(adj: dict) -> float:
    """平均局所クラスタ係数(Watts-Strogatz)。純Python・非加重・決定論(ソート走査)。
    局所 C_u = 2·(隣接同士の実辺数)/(k(k−1))、k=deg(u)。次数<2 のノードは平均から除外。"""
    csum = 0.0
    cnt = 0
    for u in sorted(adj):
        neigh = adj[u]
        k = len(neigh)
        if k < 2:
            continue
        ns = sorted(neigh)
        links = 0
        for i in range(len(ns)):
            ai = adj.get(ns[i], ())
            for j in range(i + 1, len(ns)):
                if ns[j] in ai:
                    links += 1
        csum += 2.0 * links / (k * (k - 1))
        cnt += 1
    return csum / cnt if cnt else 0.0


def giant_component_frac(nodes: set, adj: dict) -> float:
    """最大連結成分のノード比(BFS・決定論)。孤立ノードは build 側で既に除外済み。"""
    ns = set(nodes)
    if not ns:
        return 0.0
    seen: set = set()
    best = 0
    for start in sorted(ns):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj.get(u, ()):  # adj は無向(両端に登録済み)
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        best = max(best, size)
    return best / len(ns)


def network_metrics(graph: dict) -> dict:
    """1つの窓グラフの大域指標(密度・平均次数・クラスタ係数・次数gini・最大成分比・
    平均紐帯重み)。graph = build_window_graph の返り値(W/adj/nodes)。"""
    nodes, adj, W = graph["nodes"], graph["adj"], graph["W"]
    N = len(nodes)
    E = len(W)
    density = (2.0 * E / (N * (N - 1))) if N > 1 else 0.0
    mean_degree = (2.0 * E / N) if N else 0.0
    degrees = [len(adj[u]) for u in nodes]
    return {
        "n_nodes": N, "n_edges": E,
        "density": _r(density, 6),
        "mean_degree": _r(mean_degree, 4),
        "clustering_coeff": _r(avg_clustering(adj), 6),
        "degree_gini": _r(_gini_seq(degrees), 4),
        "giant_component_frac": _r(giant_component_frac(nodes, adj), 4),
        "mean_tie_weight": _r(sum(W.values()) / E, 4) if E else 0.0,
    }


def tie_decay_hist(graphs_by_win: list[dict]) -> list[dict]:
    """紐帯寿命=各ペアが「最初に現れた窓→最後に現れた窓」の生存窓数の分布(ヒストグラム)。
    ペアは build_window_graph の足切り後の W(=紐帯)を正準とする。"""
    first: dict[tuple, int] = {}
    last: dict[tuple, int] = {}
    for w, g in enumerate(graphs_by_win):
        for e in g["W"]:                    # e=(u,v)・u<v(build 側で正規化済み)
            if e not in first:
                first[e] = w
            last[e] = w
    hist: Counter = Counter()
    for e in first:
        hist[last[e] - first[e] + 1] += 1
    return [{"lifespan_windows": k, "n_ties": hist[k]} for k in sorted(hist)]


def community_flows(comm_by_win: list[dict]) -> list[dict]:
    """隣接窓のコミュニティ間で共有メンバー数を数える(alluvial/帯グラフ用)。
    community_id はメンバー最小 id(既存の正準化)。決定論(ソート走査)。"""
    rows: list[dict] = []
    for w in range(len(comm_by_win) - 1):
        A, B = comm_by_win[w], comm_by_win[w + 1]
        for ca in sorted(A):
            ma = A[ca]
            for cb in sorted(B):
                shared = len(ma & B[cb])
                if shared > 0:
                    rows.append({"window_from": w, "community_from": ca,
                                 "window_to": w + 1, "community_to": cb,
                                 "n_shared_members": shared})
    return rows


# --------------------------------------------------------------------------- #
# 空間フットプリント(メンバーの最頻滞在エリア)
# --------------------------------------------------------------------------- #
def spatial_footprint(win_events: list[dict], members: set) -> list[dict]:
    """コミュニティメンバーの arrive/enter_building から最頻滞在エリア上位3を返す。

    エリア名は payload.name(arrive の地点名・建物名)を優先し、無ければ id を使う。
    """
    area: Counter = Counter()
    for e in win_events:
        aid = e["agent_id"]
        if aid not in members:
            continue
        k, p = e["kind"], e["payload"]
        if k == "arrive":
            nm = p.get("name") or p.get("node") or ""
            if nm and nm != "路上":
                area[nm] += 1
        elif k == "enter_building":
            nm = p.get("name") or p.get("building") or ""
            if nm:
                area[nm] += 1
    top = sorted(area.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return [{"area": a, "count": c} for a, c in top]


# --------------------------------------------------------------------------- #
# 達成の帰属(成果イベント → その窓のコミュニティ)
# --------------------------------------------------------------------------- #
def _empty_ach() -> dict:
    return {"events_hosted": 0, "event_attendees_drawn": 0, "groups_founded": 0,
            "ventures_opened": 0, "venture_revenue": 0.0, "proposals_made": 0,
            "proposals_supported": 0, "rules_enacted": 0, "cascades": []}


def attribute_achievements(win_events: list[dict], part: dict[int, int]) -> dict:
    """窓内の成果イベントを、駆動者の(その窓の)コミュニティに帰属して集計。"""
    ach: dict[int, dict] = defaultdict(_empty_ach)

    def cid_of(a):
        return part.get(a)

    for e in win_events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if not isinstance(aid, int) or aid < 0:
            continue
        if k == "event_host":
            c = cid_of(aid)
            if c is not None:
                ach[c]["events_hosted"] += 1
        elif k == "event_attend":
            host = p.get("host")
            c = cid_of(host) if isinstance(host, int) else None
            if c is not None:
                ach[c]["event_attendees_drawn"] += 1
        elif k == "group_found":
            c = cid_of(aid)
            if c is not None:
                ach[c]["groups_founded"] += 1
        elif k == "venture_open":
            c = cid_of(aid)
            if c is not None:
                ach[c]["ventures_opened"] += 1
        elif k == "venture_sale":
            c = cid_of(aid)
            if c is not None:
                ach[c]["venture_revenue"] += float(p.get("amount", 0) or 0)
        elif k == "proposal":
            c = cid_of(aid)
            if c is not None:
                ach[c]["proposals_made"] += 1
        elif k == "proposal_support":
            au = p.get("author")
            c = cid_of(au) if isinstance(au, int) else cid_of(aid)
            if c is not None:
                ach[c]["proposals_supported"] += 1
        elif k == "institution_rule":
            pr = p.get("proposer")
            c = cid_of(pr) if isinstance(pr, int) else cid_of(aid)
            if c is not None:
                ach[c]["rules_enacted"] += 1
    return ach


# --------------------------------------------------------------------------- #
# ライフサイクル追跡(窓間 Jaccard マッチング → 動的 id / Palla 分類)
# --------------------------------------------------------------------------- #
def track_lifecycle(comm_by_win: list[dict[int, set]]) -> tuple[dict, dict, list]:
    """窓系列のコミュニティ(size>=2)を追跡。

    返り値:
      did_of   : {(w, cid): dynamic_community_id}  一生を1本の線に束ねた id
      role_of  : {(w, cid): {"role","stability"}}  そのスナップショットの誕生/成長/…
      events   : [{from_window,to_window,type,...}]  合流/分裂/誕生/消滅の時系列
    """
    did_of: dict[tuple, int] = {}
    role_of: dict[tuple, dict] = {}
    events: list[dict] = []
    next_did = 0

    for w, parts in enumerate(comm_by_win):
        if w == 0:
            for cid in sorted(parts):
                did_of[(w, cid)] = next_did
                next_did += 1
                role_of[(w, cid)] = {"role": "birth", "stability": 0.0}
            continue
        prev = comm_by_win[w - 1]
        claimed: set = set()          # 当窓で継承済みの動的 id
        # curr 側の分類 + 動的 id 継承(cid 昇順で決定的に)
        for cid in sorted(parts):
            Cj = parts[cid]
            matches = []              # (J, prev_cid)
            for pcid in sorted(prev):
                J = _jaccard(prev[pcid], Cj)
                if J >= TAU:
                    matches.append((J, pcid))
            matches.sort(key=lambda x: (-x[0], x[1]))
            if not matches:
                did_of[(w, cid)] = next_did
                next_did += 1
                role_of[(w, cid)] = {"role": "birth", "stability": 0.0}
            else:
                bestJ, bp = matches[0]
                pdid = did_of[(w - 1, bp)]
                if pdid not in claimed:
                    did_of[(w, cid)] = pdid
                    claimed.add(pdid)
                else:                 # 既に使われた=分裂の枝 → 新規動的 id
                    did_of[(w, cid)] = next_did
                    next_did += 1
                if len(matches) >= 2:
                    role = "merge"
                elif len(Cj) > len(prev[bp]):
                    role = "grow"
                elif len(Cj) < len(prev[bp]):
                    role = "shrink"
                else:
                    role = "stable"
                role_of[(w, cid)] = {"role": role, "stability": _r(bestJ, 4)}

        # 時系列イベント(合流/分裂/誕生/消滅)を prev↔curr の対応から抽出
        # 誕生
        for cid in sorted(parts):
            if role_of[(w, cid)]["role"] == "birth":
                events.append({"from_window": w - 1, "to_window": w,
                               "type": "birth", "community_id": cid,
                               "size": len(parts[cid])})
        # 合流(curr 1個 ← prev 2個以上)
        for cid in sorted(parts):
            Cj = parts[cid]
            srcs = sorted(pcid for pcid in prev if _jaccard(prev[pcid], Cj) >= TAU)
            if len(srcs) >= 2:
                events.append({"from_window": w - 1, "to_window": w,
                               "type": "merge", "into": cid,
                               "from_communities": srcs})
        # 分裂(prev 1個 → curr 2個以上)/ 消滅(prev → curr にマッチ無し)
        for pcid in sorted(prev):
            Ci = prev[pcid]
            dsts = sorted(cid for cid in parts if _jaccard(Ci, parts[cid]) >= TAU)
            if len(dsts) >= 2:
                events.append({"from_window": w - 1, "to_window": w,
                               "type": "split", "from_community": pcid,
                               "into_communities": dsts})
            elif len(dsts) == 0:
                events.append({"from_window": w - 1, "to_window": w,
                               "type": "death", "community_id": pcid,
                               "size": len(Ci)})
    return did_of, role_of, events


# --------------------------------------------------------------------------- #
# メイン解析
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, window_days: int = 7,
            tau: float = TAU, gamma: float = GAMMA,
            min_weight: float = MIN_WEIGHT) -> dict:
    """1 ラン分の解析を実行し、communities.json 相当の dict を返す。"""
    global TAU, GAMMA
    TAU, GAMMA = tau, gamma

    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    n_agents = len(agents) or (max((e["agent_id"] for e in events
                                    if e["agent_id"] >= 0), default=-1) + 1)
    orgs = org_labels(agents)
    run_name = os.path.basename(os.path.normpath(run_dir))

    window_steps = max(int(window_days) * STEPS_PER_DAY, 1)
    max_step = max((e["step"] for e in events), default=0)

    # 窓ごとにイベントを振り分け(step 昇順は保たない=窓内順序は元の行順で十分)
    win_events: list[list[dict]] = []
    n_windows = max_step // window_steps + 1
    buckets: dict[int, list] = defaultdict(list)
    for e in events:
        buckets[e["step"] // window_steps].append(e)
    for w in range(n_windows):
        win_events.append(buckets.get(w, []))

    # 全ラン通してのカスケード(達成の一種)。coin の step で窓へ帰属。
    cascades = m.item_cascades(events)
    coin_step: dict[str, int] = {}
    for e in events:
        if e["kind"] in ("vocab_coin", "label_coin"):
            iid = e["payload"].get("item_id")
            if iid is not None and iid not in coin_step:
                coin_step[iid] = e["step"]

    # ---- 窓ごとの検出・測度 ----
    windows_out: list[dict] = []
    comm_by_win: list[dict[int, set]] = []       # size>=2 のみ(ライフサイクル用)
    part_by_win: list[dict[int, int]] = []        # 全ノード分割(達成帰属・NMI 用)
    ach_by_win: list[dict] = []
    graphs_by_win: list[dict] = []                # 窓グラフ(W3 大域指標・tie decay 用)

    for w in range(n_windows):
        we = win_events[w]
        graph = build_window_graph(we, min_weight)
        graphs_by_win.append(graph)
        det = detect(graph, n_agents)
        part = det["part"]        # 足切り後に紐帯を持つノードのみ(孤立=無所属)
        part_by_win.append(part)

        # size>=2 のコミュニティ(無所属・単独は数えない)
        members: dict[int, list] = defaultdict(list)
        for node, cid in part.items():
            members[cid].append(node)
        comm_members = {cid: set(ms) for cid, ms in members.items()
                        if len(ms) >= 2}
        comm_by_win.append(comm_members)

        ach = attribute_achievements(we, part)
        ach_by_win.append(ach)

        ei, ildeg = ei_and_density(graph["W"], part)
        leaders = pagerank_leaders(graph, comm_members)

        # 参照分割(組織)との一致/乖離
        active_org = {node: orgs[node] for node in graph["nodes"] if node in orgs}
        nmi_org = _r(nmi({node: part[node] for node in active_org}, active_org), 4)
        ei_org = _r(ei_index_of_labels(graph["W"], orgs), 4)

        windows_out.append({
            "window": w,
            "step_start": w * window_steps,
            "step_end": (w + 1) * window_steps,
            "day_start": w * window_days,
            "day_end": (w + 1) * window_days,
            "n_active_nodes": len(graph["nodes"]),
            "n_edges": len(graph["W"]),
            "n_communities": len(comm_members),
            "n_singletons": sum(1 for ms in members.values() if len(ms) == 1),
            "modularity_louvain": _r(det["mod_louv"]),
            "modularity_lpa": _r(det["mod_lpa"]),
            "n_communities_lpa": det["n_lpa"],
            "reference": {"nmi_org": nmi_org, "ei_org": ei_org,
                          "n_org_nodes": len(active_org)},
            "_comm_members": comm_members,   # 後段で使う一時フィールド(出力前に除去)
            "_ei": ei, "_ildeg": ildeg, "_leaders": leaders, "_graph": graph,
        })

    # ---- カスケードを窓・コミュニティへ帰属 ----
    cascade_by: dict[tuple, list] = defaultdict(list)
    for c in cascades:
        if c["n_adopters"] < CASCADE_MIN_ADOPTERS:
            continue
        creator = c.get("creator")
        if not isinstance(creator, int) or creator < 0:
            continue
        cst = coin_step.get(c["item_id"])
        if cst is None:
            continue
        wi = cst // window_steps
        if wi >= len(part_by_win):
            continue
        cid = part_by_win[wi].get(creator)
        if cid is None:
            continue
        cascade_by[(wi, cid)].append({"item_id": c["item_id"], "size": c["size"],
                                      "n_adopters": c["n_adopters"]})

    # ---- ライフサイクル追跡 ----
    did_of, role_of, life_events = track_lifecycle(comm_by_win)

    # ---- コミュニティ単位の出力を組み立て ----
    for w, wout in enumerate(windows_out):
        comm_members = wout.pop("_comm_members")
        ei = wout.pop("_ei")
        ildeg = wout.pop("_ildeg")
        leaders = wout.pop("_leaders")
        wout.pop("_graph")
        ach = ach_by_win[w]

        comms_json: list[dict] = []
        # サイズ降順+最小 id(検収 2026-07-11 のソート指定)
        for cid in sorted(comm_members,
                          key=lambda c: (-len(comm_members[c]), c)):
            ms = sorted(comm_members[cid])
            size = len(ms)
            internal = ildeg.get(cid, 0)
            dens = internal / (size * (size - 1) / 2) if size >= 2 else 0.0
            leader, lpr = leaders.get(cid, (min(ms), 0.0))
            a = dict(_empty_ach())
            a.update(ach.get(cid, {}))
            a["venture_revenue"] = _r(a["venture_revenue"], 2)
            a["cascades"] = sorted(cascade_by.get((w, cid), []),
                                   key=lambda x: (str(x["item_id"])))
            comms_json.append({
                "community_id": cid,
                "dynamic_community_id": did_of.get((w, cid)),
                "size": size,
                "members": ms,
                "internal_density": _r(dens, 4),
                "ei_index": _r(ei.get(cid, 0.0), 4),
                "leader": leader,
                "leader_pagerank": lpr,
                "stability_jaccard": role_of.get((w, cid), {}).get("stability", 0.0),
                "lifecycle": role_of.get((w, cid), {}).get("role", "birth"),
                "spatial_footprint": spatial_footprint(win_events[w], comm_members[cid]),
                "achievements": a,
            })
        wout["communities"] = comms_json

    # ---- ⑤W3: 大域指標の時系列 / tie decay / 窓遷移(communities.json 本体は不変・追加のみ)----
    network_ts = [dict(window=w, day_start=windows_out[w]["day_start"],
                       day_end=windows_out[w]["day_end"],
                       **network_metrics(graphs_by_win[w]))
                  for w in range(n_windows)]

    return {
        "run": run_name,
        "network_ts": network_ts,
        "community_flows": community_flows(comm_by_win),
        "tie_decay": tie_decay_hist(graphs_by_win),
        "params": {
            "window_days": window_days, "window_steps": window_steps,
            "seed": SEED, "resolution": gamma, "tau": tau,
            "min_weight": min_weight,
            "cascade_min_adopters": CASCADE_MIN_ADOPTERS,
            "alpha": ALPHA, "sns_weights": SNS_W,
            "detector": "networkx.louvain_communities(seed=0, weight='weight')",
            "cross_check": "measure.communities (deterministic LPA)",
        },
        "n_agents": n_agents,
        "n_windows": n_windows,
        "n_org_nodes_total": len(orgs),
        "windows": windows_out,
        "lifecycle_events": life_events,
    }


# --------------------------------------------------------------------------- #
# 日本語レポート
# --------------------------------------------------------------------------- #
_LIFE_JP = {"birth": "誕生", "grow": "成長", "shrink": "縮小", "stable": "安定",
            "merge": "合流", "split": "分裂", "death": "消滅"}


def write_report(result: dict, path: str) -> None:
    L: list[str] = []
    L.append(f"# 自然発生コミュニティ観察レポート: {result['run']}\n")
    L.append("エージェントを無理に組織化せず、対面会話・DM・SNS 反応・イベント共参加から")
    L.append("**自然に形成されたコミュニティ**を後処理で検出し、その動き(誕生・成長・")
    L.append("合流・分裂・消滅)と構造・達成を観察した結果。組織所属(会社・学校)と")
    L.append("found_group は検出に入れず、参照分割として一致/乖離のみ測っている。\n")

    p = result["params"]
    L.append("## 設定")
    L.append(f"- エージェント数: {result['n_agents']} / 窓数: {result['n_windows']}"
             f"(窓幅 {p['window_days']}日 = {p['window_steps']} step)")
    L.append(f"- 検出器: {p['detector']}(正準・決定論)")
    L.append(f"- 交差確認: {p['cross_check']} / resolution={p['resolution']}")
    L.append(f"- 紐帯の足切り min_weight={p['min_weight']}"
             "(合成重み未満の偶発的接触はエッジにしない。孤立者=無所属)")
    L.append(f"- Jaccard マッチング閾値 τ={p['tau']} / カスケード最小採用者="
             f"{p['cascade_min_adopters']}")
    L.append(f"- チャネル・サリエンス重み α={p['alpha']}\n")

    # ---- 窓別サマリ表 ----
    L.append("## 窓別サマリ")
    L.append("| 窓 | 日 | 紐帯人数 | コミュ数 | 最大サイズ | 最大コミュのリーダー | "
             "Q(正準louvain/LPA交差) | NMI(組織) | E-I(組織) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for w in result["windows"]:
        comms = w["communities"]
        if comms:
            biggest = max(comms, key=lambda c: (c["size"], -c["community_id"]))
            bsize = biggest["size"]
            leader = biggest["leader"]
        else:
            bsize, leader = 0, "—"
        L.append(f"| {w['window']} | {w['day_start']}–{w['day_end']} | "
                 f"{w['n_active_nodes']} | {w['n_communities']} | {bsize} | "
                 f"agent {leader} | {w['modularity_louvain']}/{w['modularity_lpa']} | "
                 f"{w['reference']['nmi_org']} | {w['reference']['ei_org']} |")
    L.append("")
    L.append("※ NMI(組織)が低く E-I(組織)が正に大きいほど、自然コミュニティは"
             "会社・学校の線から乖離している(= 組織を強制しなくても別の集団が育つ)。\n")

    # ---- 各窓のコミュニティ詳細 ----
    L.append("## コミュニティ構造(窓ごと)")
    for w in result["windows"]:
        if not w["communities"]:
            continue
        L.append(f"### 窓 {w['window']}(Day {w['day_start']}–{w['day_end']})")
        L.append("| id | 動的id | サイズ | 内部密度 | E-I | リーダー | 安定性 | 状態 | 主な居場所 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in sorted(w["communities"], key=lambda c: (-c["size"], c["community_id"])):
            foot = "、".join(f"{f['area']}({f['count']})"
                            for f in c["spatial_footprint"]) or "—"
            L.append(f"| {c['community_id']} | {c['dynamic_community_id']} | "
                     f"{c['size']} | {c['internal_density']} | {c['ei_index']} | "
                     f"agent {c['leader']} | {c['stability_jaccard']} | "
                     f"{_LIFE_JP.get(c['lifecycle'], c['lifecycle'])} | {foot} |")
        L.append("")

    # ---- ライフサイクル物語 ----
    L.append("## ライフサイクル物語(誕生・成長・合流・分裂・消滅)")
    ev = result["lifecycle_events"]
    if not ev:
        L.append("(窓が1つ以下、または追跡対象のコミュニティが無い)\n")
    else:
        by_trans: dict[tuple, list] = defaultdict(list)
        for e in ev:
            by_trans[(e["from_window"], e["to_window"])].append(e)
        for (fw, tw) in sorted(by_trans):
            L.append(f"- **窓 {fw} → {tw}**")
            for e in by_trans[(fw, tw)]:
                t = e["type"]
                if t == "birth":
                    L.append(f"  - 誕生: コミュニティ {e['community_id']}"
                             f"(サイズ {e['size']})が新たに現れた")
                elif t == "death":
                    L.append(f"  - 消滅: コミュニティ {e['community_id']}"
                             f"(サイズ {e['size']})が消えた")
                elif t == "merge":
                    L.append(f"  - 合流: {e['from_communities']} が "
                             f"{e['into']} へまとまった")
                elif t == "split":
                    L.append(f"  - 分裂: {e['from_community']} が "
                             f"{e['into_communities']} へ分かれた")
        L.append("")

    # ---- 達成一覧 ----
    L.append("## 集団が成し遂げたこと(コミュニティ×窓)")
    rows = []
    for w in result["windows"]:
        for c in w["communities"]:
            a = c["achievements"]
            total = (a["events_hosted"] + a["groups_founded"] + a["ventures_opened"]
                     + a["proposals_made"] + a["rules_enacted"] + len(a["cascades"]))
            if total == 0 and a["venture_revenue"] == 0 \
                    and a["event_attendees_drawn"] == 0 and a["proposals_supported"] == 0:
                continue
            rows.append((w["window"], c))
    if not rows:
        L.append("(このランでは検出コミュニティに帰属する成果イベントが無い)\n")
    else:
        L.append("| 窓 | コミュ | 主催(集客) | 設立G | 開業(売上) | 提案(採択) | 制度化 | カスケード |")
        L.append("|---|---|---|---|---|---|---|---|")
        for wi, c in rows:
            a = c["achievements"]
            casc = "、".join(f"{x['item_id']}(×{x['n_adopters']})" for x in a["cascades"]) or "—"
            L.append(f"| {wi} | {c['community_id']} | "
                     f"{a['events_hosted']}({a['event_attendees_drawn']}) | "
                     f"{a['groups_founded']} | "
                     f"{a['ventures_opened']}({a['venture_revenue']}) | "
                     f"{a['proposals_made']}({a['proposals_supported']}) | "
                     f"{a['rules_enacted']} | {casc} |")
        L.append("")

    _write_report_w3(L, result)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _write_report_w3(L: list, result: dict) -> None:
    """⑤W3: ネットワーク大域指標の時系列・tie decay・窓遷移の節(追加のみ)。"""
    L.append("## ネットワーク大域指標の時系列(第31バッチ W3)")
    net = result.get("network_ts") or []
    if not net:
        L.append("(窓グラフが空=データ不足)\n")
    else:
        L.append("| 窓 | 日 | ノード | エッジ | 密度 | 平均次数 | クラスタ係数 | "
                 "次数gini | 最大成分比 | 平均紐帯重み |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in net:
            L.append(f"| {r['window']} | {r['day_start']}–{r['day_end']} | "
                     f"{r['n_nodes']} | {r['n_edges']} | {r['density']} | "
                     f"{r['mean_degree']} | {r['clustering_coeff']} | "
                     f"{r['degree_gini']} | {r['giant_component_frac']} | "
                     f"{r['mean_tie_weight']} |")
        L.append("")
        L.append("※ 全量は panel/network_ts.parquet。密度・クラスタ係数は足切り後"
                 "(min_weight)の重み付き無向グラフの非加重指標。\n")

    L.append("## 紐帯寿命の分布(tie decay)")
    dec = result.get("tie_decay") or []
    if not dec:
        L.append("(観測された紐帯が無い=データ不足)\n")
    else:
        total = sum(d["n_ties"] for d in dec)
        ephem = sum(d["n_ties"] for d in dec if d["lifespan_windows"] == 1)
        L.append(f"- ユニーク紐帯(窓横断のペア)総数: {total} / うち1窓限りの一過性: "
                 f"{ephem}({_r(ephem / total, 3) if total else 0.0})")
        L.append("| 寿命(窓数) | 紐帯数 |")
        L.append("|---|---|")
        for d in dec:
            L.append(f"| {d['lifespan_windows']} | {d['n_ties']} |")
        L.append("")

    L.append("## コミュニティの窓遷移(alluvial 用・共有メンバー)")
    fl = result.get("community_flows") or []
    if not fl:
        L.append("(2窓以上に跨る帯が無い=データ不足)\n")
    else:
        L.append(f"- 遷移エッジ数: {len(fl)}(全量 panel/community_flows.parquet)"
                 "。合流=同一 to へ複数 from が集まる帯・分裂=同一 from から複数 to。")
        L.append("| 窓 | コミュ元 | → 窓 | コミュ先 | 共有メンバー |")
        L.append("|---|---|---|---|---|")
        for r in sorted(fl, key=lambda x: (x["window_from"], -x["n_shared_members"],
                                           x["community_from"], x["community_to"]))[:20]:
            L.append(f"| {r['window_from']} | {r['community_from']} | "
                     f"{r['window_to']} | {r['community_to']} | "
                     f"{r['n_shared_members']} |")
        L.append("")


# --------------------------------------------------------------------------- #
def _strip_private(result: dict) -> dict:
    """出力前に一時フィールド(_ 始まり)が残っていないことを保証する。"""
    for w in result.get("windows", []):
        for k in list(w):
            if k.startswith("_"):
                w.pop(k)
    return result


# W3 追加出力の parquet 化(communities.json/レポートは非破壊で別ファイルに置く)
_W3_KEYS = ("network_ts", "community_flows", "tie_decay")


def _write_w3_panels(run_dir: str, result: dict) -> dict:
    """network_ts / community_flows / tie_decay を runs/<name>/panel へ書く(追加のみ)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = os.path.join(run_dir, "panel")
    os.makedirs(out_dir, exist_ok=True)
    f = pa.field

    def _write(name, keys, schema, rows):
        cols = {c: [r.get(c) for r in rows] for c in keys}
        pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                       os.path.join(out_dir, name))

    _write("network_ts.parquet",
           ["window", "day_start", "day_end", "n_nodes", "n_edges", "density",
            "mean_degree", "clustering_coeff", "degree_gini",
            "giant_component_frac", "mean_tie_weight"],
           pa.schema([
               f("window", pa.int64()), f("day_start", pa.int64()),
               f("day_end", pa.int64()), f("n_nodes", pa.int64()),
               f("n_edges", pa.int64()), f("density", pa.float64()),
               f("mean_degree", pa.float64()), f("clustering_coeff", pa.float64()),
               f("degree_gini", pa.float64()),
               f("giant_component_frac", pa.float64()),
               f("mean_tie_weight", pa.float64())]),
           result.get("network_ts", []))
    _write("community_flows.parquet",
           ["window_from", "community_from", "window_to", "community_to",
            "n_shared_members"],
           pa.schema([
               f("window_from", pa.int64()), f("community_from", pa.int64()),
               f("window_to", pa.int64()), f("community_to", pa.int64()),
               f("n_shared_members", pa.int64())]),
           result.get("community_flows", []))
    _write("tie_decay.parquet", ["lifespan_windows", "n_ties"],
           pa.schema([f("lifespan_windows", pa.int64()),
                      f("n_ties", pa.int64())]),
           result.get("tie_decay", []))
    return {"network_ts": len(result.get("network_ts", [])),
            "community_flows": len(result.get("community_flows", [])),
            "tie_decay": len(result.get("tie_decay", []))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第18バッチ①: 自然発生コミュニティの観察・分析(L1 を読むだけ)")
    ap.add_argument("run_dir", help="例: runs/eco80_3day")
    ap.add_argument("--window-days", type=int, default=7, help="時間窓の日数(既定 7)")
    ap.add_argument("--tau", type=float, default=TAU, help="窓間 Jaccard 閾値(既定 0.30)")
    ap.add_argument("--resolution", type=float, default=GAMMA,
                    help="louvain resolution(既定 1.0)")
    ap.add_argument("--min-weight", type=float, default=MIN_WEIGHT,
                    help="紐帯とみなす最小合成重み(既定 2.0。未満のエッジは足切り)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    result = analyze(args.run_dir, window_days=args.window_days,
                     tau=args.tau, gamma=args.resolution,
                     min_weight=args.min_weight)
    result = _strip_private(result)

    # W3 追加出力(panel/*.parquet)。communities.json 本体は W3 キーを除いて従来通り。
    w3 = _write_w3_panels(args.run_dir, result)

    js_path = os.path.join(args.run_dir, "communities.json")
    md_path = os.path.join(args.run_dir, "communities_report.md")
    core = {k: v for k, v in result.items() if k not in _W3_KEYS}
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(core, sort_keys=True, ensure_ascii=False))
    write_report(result, md_path)

    n_comm = sum(w["n_communities"] for w in result["windows"])
    print(f"[communities] {args.run_dir}: {result['n_windows']} 窓 / "
          f"検出コミュニティ延べ {n_comm} / ライフサイクル事象 "
          f"{len(result['lifecycle_events'])}")
    print(f"  W3: network_ts {w3['network_ts']} 行 / community_flows "
          f"{w3['community_flows']} 行 / tie_decay {w3['tie_decay']} 区分")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")
    print(f"  -> {os.path.join(args.run_dir, 'panel')}/(network_ts, "
          "community_flows, tie_decay).parquet")


if __name__ == "__main__":
    main()
