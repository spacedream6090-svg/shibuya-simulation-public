#!/usr/bin/env python
"""S-09 組織形態の観測(mode × structure × role)(SV2-B・読み取り専用)。

    python scripts/analyze_org_form.py runs/<name>
    python scripts/analyze_org_form.py runs/<name> --window-days 7 --out experiments/org_form

■ 正典
  採る指標と**採らない指標の理由**は事前登録 §3-I(承認対象)で固定済み。
  「どの中心性を使うか」を後から選ぶのは査読で最も突かれるので、本スクリプトは
  **事前登録に書いた指標しか出さない**。

| 軸(サーベイ §4.1.2–4.1.3) | 採る指標 | 一次文献 |
|---|---|---|
| structure: centralized ↔ decentralized | **Freeman の群レベル次数中心化指数 C_D** + `degree_gini` を併記 | Freeman (1979) Social Networks 1:215–239 |
| structure: layered(層の有無) | **Global Reaching Centrality (GRC)**(有向グラフ) | Mones, Vicsek & Vicsek (2012) PLOS ONE 7(3):e33799 |
| role: communicator / worker / director | **Guimerà–Amaral の (z, P) 役割カルトグラフィ** | Guimerà & Amaral (2005) Nature 433:895–900 |
| mode: static ↔ dynamic | **窓間 Jaccard 安定度 + Palla のライフサイクル語彙** | Palla, Barabási & Vicsek (2007) Nature 446:664–667 |
| 仲介 | **betweenness**(`analyze_founders.betweenness_bfs` を再利用) | Freeman (1979) |

■ ★採らないもの(採らない理由を書くのが S-04 の境界宣言の作法)
  - **eigenvector 中心性は採らない**。(a) repo 内に実装が無く新規に書く唯一の中心性になる、
    (b) `nx.eigenvector_centrality` は冪乗法で**収束しないグラフがある**(決定論性の主張と
    相性が悪い)、(c)「誰が影響力を持つか」は S-11(**本選後**)の主題であって
    S-09(組織の**形**)の主題ではない。
  - **Gini の 6 個目・中心性の 3 個目を実装しない**(既存実装を import / 定義を揃える)。

■ ★Guimerà–Amaral の閾値について(事前登録 §3-I の宣言)
  原典の閾値 **z = 2.5 / P = 0.62 は代謝ネットワークで較正された値**であり、社会ネットワークに
  そのまま使える保証は無い。したがって **既定は閾値を使わず (z, P) の分位点を報告**し、
  閾値による 4 分類の件数は**参考値としてのみ併記**する(**主判定には用いない**)。

■ 既存資産の再利用(再実装しない)
  - `scripts/analyze_communities.py` … `build_window_graph` / `detect`(seed 固定 Louvain)/
    `track_lifecycle`(Palla 語彙)/ `_jaccard` / `_gini_seq`
  - `scripts/analyze_founders.py`   … `betweenness_bfs`(Brandes・純 Python)
  - `networkx`                       … `global_reaching_centrality`(GRC の標準実装)

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。**`scripts/analyze_specialization.py`(凍結 13 番)は
  開かない**。決定論: ノード・辺・コミュニティはソート順、Louvain は seed 固定。
  大きすぎるグラフでは GRC / betweenness を**打ち切って打ち切ったと書く**(黙って近似しない)。
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
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx                                   # noqa: E402

import analyze_communities as ac                        # noqa: E402  (凍結外・再利用)
from analyze_founders import betweenness_bfs            # noqa: E402  (再実装しない)

import run_dt                                         # noqa: E402  (W2-3: Δt の単一の源)

STEPS_PER_DAY = ac.STEPS_PER_DAY                        # 正準 Δt=10 の 144(実値は run 依存)
GA_Z_THRESHOLD = 2.5        # Guimerà–Amaral 原典(代謝網で較正)。★参考値専用
GA_P_THRESHOLD = 0.62       # 同上。★主判定には用いない
GRC_MAX_NODES = 1500        # これを超えたら GRC を打ち切る(O(N(N+E)))
BTW_MAX_NODES = 1500        # これを超えたら betweenness を打ち切る
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# build_window_graph が読む kind だけをパースする(全 payload を触らない)
GRAPH_KINDS = ("speak", "hear", "dm", "sns_like", "sns_reshare",
               "event_attend", "event_host")


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


def freeman_degree_centralization(adj: dict) -> float | None:
    """Freeman (1979) の群レベル次数中心化指数 C_D。スター型 = 1・完全グラフ = 0。

        C_D = Σ_i (d_max − d_i) / ((n−1)(n−2))

    ★`degree_gini` とは**別の量**である(Gini は分布の不平等、C_D はスター型からの距離)。
    n < 3 では正規化子が 0 になるので None を返す(0 と偽らない)。
    """
    nodes = sorted(adj)
    n = len(nodes)
    if n < 3:
        return None
    degs = [len(adj[u]) for u in nodes]
    dmax = max(degs)
    num = sum(dmax - d for d in degs)
    den = (n - 1) * (n - 2)
    return num / den if den else None


def global_reaching_centrality(dir_w: dict, max_nodes: int = GRC_MAX_NODES) -> dict:
    """Mones et al. (2012) の GRC。フラット網 ≈ 0・木に近いほど 1。有向グラフに当てる。

    `networkx.algorithms.centrality.global_reaching_centrality` を使う(純 Python・scipy 不要)。
    ノード数が max_nodes を超えたら **打ち切って打ち切ったと書く**(黙って近似しない)。

    ★★ **強連結のときの GRC = 0 を「フラットな組織」と読ませない**(実測で判明した落とし穴):
    有向グラフが強連結なら全ノードの局所到達中心性が 1 になり、
    **GRC は構造的に必ず 0 になる**(最大 − 平均 = 0)。これは「階層が無い」ことの観測ではなく
    **「この窓の粒度では階層を判別できない」**ことの表明である。
    実測(`runs/night_llm_100a3d`・100 ノード / 1,936 有向辺)で強連結成分が 1 個 = GRC 0.0 に
    なることを確認した。よって強連結のときは status を
    **`DEGENERATE_STRONGLY_CONNECTED`** にして区別する。
    """
    nodes = sorted({u for (u, _v) in dir_w} | {v for (_u, v) in dir_w})
    if len(nodes) < 2:
        return {"grc": None, "n_nodes": len(nodes), "n_edges": len(dir_w),
                "status": "TOO_SMALL"}
    if len(nodes) > max_nodes:
        return {"grc": None, "n_nodes": len(nodes), "n_edges": len(dir_w),
                "status": "SKIPPED_TOO_LARGE", "max_nodes": int(max_nodes)}
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (u, v) in sorted(dir_w):
        G.add_edge(u, v)
    try:
        val = float(nx.global_reaching_centrality(G))
        n_scc = int(nx.number_strongly_connected_components(G))
    except Exception as exc:                              # 収束しない/未定義は正直に出す
        return {"grc": None, "n_nodes": len(nodes), "n_edges": len(dir_w),
                "status": f"ERROR:{type(exc).__name__}"}
    status = "OK"
    note = ""
    if n_scc == 1:
        status = "DEGENERATE_STRONGLY_CONNECTED"
        note = ("有向グラフが強連結(SCC=1)なので GRC は構造的に 0 になる。"
                "**「階層が無い」ではなく「この粒度では階層を判別できない」**と読む。")
    return {"grc": _r(val, 4), "n_nodes": len(nodes), "n_edges": len(dir_w),
            "n_scc": n_scc, "status": status, "note": note}


def guimera_amaral(adj: dict, part: dict) -> dict:
    """Guimerà–Amaral (2005) の (z, P)。z = モジュール内次数の z スコア、P = 参加係数。

        z_i = (k_i^{s_i} − mean_{j∈s_i} k_j^{s_i}) / sd_{j∈s_i} k_j^{s_i}
        P_i = 1 − Σ_s (k_{i,s} / k_i)²

    読み替え(事前登録 §3-I): **高 P = communicator(橋)/ 高 z = director(内ハブ)/
    低 z 低 P = worker**。sd = 0 のモジュール(全員同次数)は z = 0 とする。
    """
    nodes = sorted(n for n in adj if n in part)
    if not nodes:
        return {"z": {}, "P": {}, "n": 0}
    # モジュール別の内部次数
    k_in: dict[int, int] = {}
    k_by_mod: dict[int, dict[int, int]] = {}
    for u in nodes:
        cnt: dict[int, int] = defaultdict(int)
        for v in adj[u]:
            c = part.get(v)
            if c is not None:
                cnt[c] += 1
        k_by_mod[u] = dict(cnt)
        k_in[u] = cnt.get(part[u], 0)
    by_mod: dict[int, list] = defaultdict(list)
    for u in nodes:
        by_mod[part[u]].append(u)
    z: dict[int, float] = {}
    for c in sorted(by_mod):
        members = sorted(by_mod[c])
        ks = [k_in[u] for u in members]
        n = len(ks)
        mu = sum(ks) / n
        var = (sum((k - mu) ** 2 for k in ks) / n) if n else 0.0
        sd = math.sqrt(var)
        for u in members:
            z[u] = 0.0 if sd <= 1e-12 else (k_in[u] - mu) / sd
    P: dict[int, float] = {}
    for u in nodes:
        tot = sum(k_by_mod[u].values())
        P[u] = 0.0 if tot == 0 else 1.0 - sum((k / tot) ** 2 for k in k_by_mod[u].values())
    return {"z": {u: _r(z[u], 4) for u in nodes},
            "P": {u: _r(P[u], 4) for u in nodes}, "n": len(nodes)}


def quantiles(values, qs=QUANTILES) -> dict:
    """分位点(線形補間なしの最近傍・決定論)。★主判定はこれで行う(閾値ではなく分布)。"""
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {f"q{int(q*100)}": None for q in qs}
    out = {}
    for q in qs:
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        out[f"q{int(q*100)}"] = _r(xs[i], 4)
    return out


def ga_role_counts(z: dict, P: dict, z_thr: float = GA_Z_THRESHOLD,
                   p_thr: float = GA_P_THRESHOLD) -> dict:
    """★参考値のみ。原典閾値は代謝網で較正された値であり主判定には使わない。"""
    out = {"director_hub": 0, "communicator_bridge": 0,
           "hub_and_bridge": 0, "worker": 0}
    for u in sorted(z):
        hz = z[u] is not None and z[u] >= z_thr
        hp = P.get(u) is not None and P[u] >= p_thr
        if hz and hp:
            out["hub_and_bridge"] += 1
        elif hz:
            out["director_hub"] += 1
        elif hp:
            out["communicator_bridge"] += 1
        else:
            out["worker"] += 1
    return out


# --------------------------------------------------------------------------- #
# 解析本体
# --------------------------------------------------------------------------- #
def _load_events(run_dir: str) -> list[dict]:
    """`analyze_communities.build_window_graph` が読める形で、必要 kind だけを読む。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.exists(path):
        return []
    want = set(GRAPH_KINDS)
    pf = pq.ParquetFile(path)
    avail = set(pf.schema_arrow.names)
    cols = [c for c in ("step", "agent_id", "kind", "payload") if c in avail]
    out: list[dict] = []
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        for st, aid, kind, raw in zip(d["step"], d["agent_id"], d["kind"], d["payload"]):
            if kind not in want:
                continue
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({"step": int(st), "agent_id": int(aid), "kind": kind,
                        "payload": payload})
    return out


def analyze(run_dir: str, *, window_days: int = 7,
            min_weight: float = ac.MIN_WEIGHT) -> dict:
    events = _load_events(run_dir)
    res: dict = {"run": os.path.basename(os.path.normpath(run_dir)),
                 "spec": {"window_days": int(window_days),
                          "min_weight": float(min_weight),
                          "louvain_seed": ac.SEED, "tau": ac.TAU,
                          "ga_thresholds_reference_only": {"z": GA_Z_THRESHOLD,
                                                           "P": GA_P_THRESHOLD},
                          "eigenvector_centrality": "NOT_USED (事前登録 §3-I の 3 理由)",
                          "preregistration":
                              "docs/plans/stationarity-preregistration.md §3-I"},
                 "windows": []}
    if not events:
        res["status"] = "NO_DATA"
        return res
    res["status"] = "OK"

    # W2-3: 窓幅は「日数 × このランの 1 日あたり step 数」(Δt=10 で 144 = 従来と同値)。
    window_steps = max(int(window_days) * run_dt.steps_per_day(run_dir), 1)
    max_step = max(e["step"] for e in events)
    buckets: dict[int, list] = defaultdict(list)
    for e in events:
        buckets[e["step"] // window_steps].append(e)
    n_windows = max_step // window_steps + 1

    comm_by_win: list[dict[int, set]] = []
    for w in range(n_windows):
        we = buckets.get(w, [])
        graph = ac.build_window_graph(we, min_weight)
        det = ac.detect(graph, 0)
        part = det["part"]
        adj = {u: set(vs) for u, vs in graph["adj"].items()}

        members: dict[int, list] = defaultdict(list)
        for node, cid in part.items():
            members[cid].append(node)
        comm_members = {cid: set(ms) for cid, ms in members.items() if len(ms) >= 2}
        comm_by_win.append(comm_members)

        degs = sorted((len(adj[u]) for u in sorted(adj)), reverse=True)
        ga = guimera_amaral(adj, part)
        btw = None
        if adj and len(adj) <= BTW_MAX_NODES:
            cb = betweenness_bfs(adj)
            btw = quantiles(list(cb.values()))
            btw["max"] = _r(max(cb.values()), 4) if cb else None
            btw["status"] = "OK"
        elif adj:
            btw = {"status": "SKIPPED_TOO_LARGE", "max_nodes": BTW_MAX_NODES}
        res["windows"].append({
            "window": w,
            "step_start": w * window_steps,
            "n_nodes": len(adj),
            "n_edges": len(graph["W"]),
            "n_communities": len(comm_members),
            "modularity_louvain": _r(det["mod_louv"], 4),
            # structure: centralized <-> decentralized
            "freeman_C_D": _r(freeman_degree_centralization(adj), 4),
            "degree_gini": _r(ac._gini_seq(degs), 4) if degs else None,
            "max_degree": (degs[0] if degs else 0),
            # structure: layered
            "grc": global_reaching_centrality(graph["dir_w"]),
            # role: communicator / worker / director
            "ga_z_quantiles": quantiles(list(ga["z"].values())),
            "ga_P_quantiles": quantiles(list(ga["P"].values())),
            "ga_role_counts_reference_only": ga_role_counts(ga["z"], ga["P"]),
            "ga_n": ga["n"],
            "betweenness_quantiles": btw,
        })

    # mode: static <-> dynamic(Palla のライフサイクル語彙・窓間 Jaccard 安定度)
    _did, role_of, life_events = ac.track_lifecycle(comm_by_win)
    role_counts: dict[str, int] = defaultdict(int)
    stabilities: list[float] = []
    for (_w, _cid), rec in sorted(role_of.items()):
        role_counts[rec["role"]] += 1
        if rec.get("stability") is not None:
            stabilities.append(float(rec["stability"]))
    ev_counts: dict[str, int] = defaultdict(int)
    for ev in life_events:
        ev_counts[ev["type"]] += 1
    stable_frac = None
    tot_roles = sum(role_counts.values())
    if tot_roles:
        stable_frac = (role_counts.get("stable", 0) + role_counts.get("grow", 0)
                       + role_counts.get("shrink", 0)) / tot_roles
    res["mode"] = {
        "role_counts": dict(sorted(role_counts.items())),
        "lifecycle_event_counts": dict(sorted(ev_counts.items())),
        "stability_jaccard_mean": _r(sum(stabilities) / len(stabilities), 4)
        if stabilities else None,
        "continuity_frac": _r(stable_frac, 4),
        "note": "continuity_frac が高いほど static 寄り・birth/death が多いほど dynamic 寄り",
    }
    # 全窓の要約(structure 軸を 1 行で読むための中央値)
    def _med(key, sub=None):
        vals = []
        for w in res["windows"]:
            v = w[key] if sub is None else (w[key] or {}).get(sub)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        return _r(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]), 4)
    res["summary"] = {
        "n_windows": n_windows,
        "freeman_C_D_median": _med("freeman_C_D"),
        "degree_gini_median": _med("degree_gini"),
        "grc_median": _med("grc", "grc"),
        "modularity_median": _med("modularity_louvain"),
    }
    return res


def render(res: dict) -> str:
    L: list[str] = []
    L.append(f"# S-09 組織形態(mode × structure × role) — `{res.get('run','?')}`")
    L.append("")
    L.append("> 採る指標と**採らない指標の理由**は事前登録 §3-I で固定済み。"
             "**eigenvector 中心性は採らない**(同節の 3 理由)。")
    L.append("")
    if res.get("status") != "OK":
        L.append("**NO_DATA** — 組織グラフを組めるイベントが無い。捏造しない。")
        return "\n".join(L) + "\n"
    s = res["summary"]
    L.append(f"- 窓数: {s['n_windows']}(窓幅 {res['spec']['window_days']} 日)")
    L.append(f"- **Freeman C_D 中央値: {s['freeman_C_D_median']}** "
             f"(スター型=1 / 完全グラフ=0)  degree_gini 中央値: {s['degree_gini_median']}")
    L.append(f"- **GRC 中央値: {s['grc_median']}** (フラット網≈0 / 木に近いほど 1)")
    L.append(f"- modularity(Louvain seed={res['spec']['louvain_seed']})中央値: "
             f"{s['modularity_median']}")
    L.append("")
    L.append("## structure(窓ごと)")
    L.append("")
    L.append("| 窓 | ノード | 辺 | コミュニティ | Freeman C_D | degree_gini | GRC | SCC | GRC状態 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for w in res["windows"]:
        g = w["grc"]
        L.append(f"| {w['window']} | {w['n_nodes']} | {w['n_edges']} "
                 f"| {w['n_communities']} | {w['freeman_C_D']} | {w['degree_gini']} "
                 f"| {g['grc']} | {g.get('n_scc','—')} | {g['status']} |")
    L.append("")
    degen = [w for w in res["windows"]
             if (w["grc"] or {}).get("status") == "DEGENERATE_STRONGLY_CONNECTED"]
    if degen:
        L.append("> **★ GRC の退化(層の判別不能)**: "
                 f"{len(degen)}/{len(res['windows'])} 窓で有向グラフが**強連結**(SCC=1)であり、")
        L.append("> **GRC は構造的に 0 になる**(全ノードの局所到達中心性が 1 = 最大 − 平均 = 0)。")
        L.append("> これは「階層が無い(フラットな組織)」という観測**ではなく**、"
                 "**この窓の粒度では層を判別できない**という表明である。")
        L.append("> 層を見たいなら窓を短くするか、辺の足切り(`--min-weight`)を上げて"
                 "疎なグラフで測り直す必要がある。")
        L.append("")
    L.append("## role — Guimerà–Amaral (z, P)")
    L.append("")
    L.append("> **主判定は分位点**(下表)。閾値 z=2.5 / P=0.62 は**代謝ネットワークで較正された"
             "値**であり社会網にそのまま使える保証が無いため、4 分類の件数は**参考値**である。")
    L.append("")
    L.append("| 窓 | 対象 | z q50 | z q90 | P q50 | P q90 | (参考)director | (参考)communicator "
             "| (参考)両方 | (参考)worker |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in res["windows"]:
        zq, pq, rc = w["ga_z_quantiles"], w["ga_P_quantiles"], \
            w["ga_role_counts_reference_only"]
        L.append(f"| {w['window']} | {w['ga_n']} | {zq['q50']} | {zq['q90']} "
                 f"| {pq['q50']} | {pq['q90']} | {rc['director_hub']} "
                 f"| {rc['communicator_bridge']} | {rc['hub_and_bridge']} "
                 f"| {rc['worker']} |")
    L.append("")
    L.append("## mode — Palla のライフサイクル語彙 + 窓間 Jaccard 安定度")
    L.append("")
    md = res["mode"]
    L.append(f"- 役割件数: {json.dumps(md['role_counts'], ensure_ascii=False)}")
    L.append(f"- ライフサイクル事象: "
             f"{json.dumps(md['lifecycle_event_counts'], ensure_ascii=False)}")
    L.append(f"- 窓間 Jaccard 安定度(平均): **{md['stability_jaccard_mean']}**")
    L.append(f"- 継続率(stable+grow+shrink の割合): **{md['continuity_frac']}** "
             f"— {md['note']}")
    L.append("")
    L.append("## 仲介(betweenness・既存実装 `analyze_founders.betweenness_bfs` を再利用)")
    L.append("")
    L.append("| 窓 | q50 | q90 | 最大 | 状態 |")
    L.append("|---|---|---|---|---|")
    for w in res["windows"]:
        b = w["betweenness_quantiles"] or {}
        L.append(f"| {w['window']} | {b.get('q50')} | {b.get('q90')} | {b.get('max')} "
                 f"| {b.get('status','—')} |")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-09 組織形態の観測(読み取り専用)")
    ap.add_argument("run", help="ランディレクトリ")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--min-weight", type=float, default=ac.MIN_WEIGHT)
    ap.add_argument("--out", default=None, help="出力先(既定 <run>/analysis)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.run):
        print(f"[analyze_org_form] ランが無い: {a.run}", file=sys.stderr)
        return 2
    res = analyze(a.run, window_days=a.window_days, min_weight=a.min_weight)
    out = a.out or os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "org_form.json")
    mp = os.path.join(out, "org_form.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    sm = res.get("summary", {})
    print(f"[analyze_org_form] {res.get('status')} windows={sm.get('n_windows')} "
          f"C_D={sm.get('freeman_C_D_median')} GRC={sm.get('grc_median')}")
    print(f"  -> {jp}")
    print(f"  -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
