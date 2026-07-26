#!/usr/bin/env python
"""第59バッチ スライス(b): 弱い紐帯レンズ(後処理・L1 を読むだけ)。

    python scripts/analyze_weak_ties.py runs/<name> [--min-strength 1]

設計正典: devlog Entry 50(3スライス精査)/ 日常観察方針(Entry 47。介入せず観測する)。
Granovetter「弱い紐帯の強さ」を **介入せず** 観測する。弱い紐帯(=コミュニティをまたぐ橋辺)が
新規情報(語彙の新規採用)の運び手になっているかを、既存資産だけで事後再構成する。

計算(すべて L1 events から。**新規依存なし**=analyze_structure と同じ measure 資産を再利用):
  1. 接触グラフ(無向・重み=接触頻度): speak(speaker→hearers)+ dm(sender→to)の共起回数を辺重みに
     する(measure._conversation_adj と同じ入力チャネル=コミュニティ検出と整合)。辺強度=接触頻度が
     Granovetter の「紐帯の強さ」の代理。
  2. コミュニティ: 既存の決定論 LPA(observer/measure.communities=会話グラフの label propagation。
     Louvain/networkx でなく既存資産)を全ラン窓で1回回す(analyze_structure と同じ検出器)。
  3. bridge 辺の抽出: 両端が別コミュニティの辺 = bridge(弱い紐帯の構造的代理)。同コミュニティ = internal。
     ★観測A(弱い紐帯の強さ): bridge 辺の平均接触頻度 vs internal 辺の平均接触頻度。bridge の方が弱い
       (接触頻度が低い)なら Granovetter の構造的含意と整合。
  4. per-agent brokerage: 各人の接する辺のうち bridge が占める割合(=別コミュニティを橋渡しする度合い)。
  5. Granovetter 検証(観測B): 語彙の **新規採用**(label_adopt)が、どの辺種経由で到来したかを突合。
     採用者 B が語 X を得た出所 A = 「B 宛の直近 transmission(payload.from)」(measure.agent_features の
     last_from 帰属規則の鏡)。A と B が別コミュニティ = bridge 経由 / 同 = internal 経由。bridge 経由の
     採用割合が母集団の bridge 辺割合より高いなら、橋辺が新規情報を運んでいる構造的関連。

計算量: O(E + adopts + N² LPA 近傍)級=長期ランでも軽い(全て純Python・乱数なし)。
決定論: ノード・辺・タイは常に id 昇順で解決。出力 JSON は sort_keys=True でバイト同一。

★正直さ注記(出力にも明記): これは **因果ではなく構造上の関連** の観測記録である。橋辺「経由で」到来した
  という時系列・経路の突合であって、橋辺が採用を「引き起こした」ことの証明ではない(交絡・逆因果を排除しない)。
  コミュニティはラン全体の集約構造(採用時点の逐次コミュニティではない)= 近似である。

出力:
  runs/<name>/weak_ties.json       … コミュニティ/辺種別/brokerage/Granovetter 突合
  runs/<name>/weak_ties_report.md  … 日本語の要約
"""
from __future__ import annotations

import argparse
import json
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

from society.observer import measure as m  # noqa: E402

MIN_STRENGTH = 1        # この接触頻度未満の辺は無視(既定 1=1回でも接触した辺を採る)


# --------------------------------------------------------------------------- #
# 1. 接触グラフ(無向・重み=接触頻度): speak(speaker→hearers)+ dm(sender→to)
# --------------------------------------------------------------------------- #
def build_weighted_graph(events: list[dict]) -> dict[tuple, int]:
    """無向辺 (min(u,v), max(u,v)) -> 接触回数。measure._conversation_adj と同じ入力チャネル。"""
    w: dict[tuple, int] = defaultdict(int)
    for e in events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if not isinstance(aid, int) or aid < 0:
            continue
        if k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    w[(min(aid, h), max(aid, h))] += 1
        elif k == "dm":
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                w[(min(aid, to), max(aid, to))] += 1
    return dict(w)


# --------------------------------------------------------------------------- #
# 1b. クラスタ係数・直径(第64バッチ フェーズ3の観測指標=計画書§4「クラスタ係数/直径」。
#     既存 analyze 系に事後算出が無かったため本スクリプトへ最小追加: 接触グラフは本スクリプトが
#     既に組んでおり、コミュニティ/橋辺と同じグラフ上で読むのが解釈上も一貫するため)
# --------------------------------------------------------------------------- #
def clustering_and_diameter(graph: dict[tuple, int]) -> dict:
    """無向・非重みの平均局所クラスタ係数(Watts-Strogatz 1998)と最大連結成分の直径。

    - clustering_coeff: 次数≥2 のノードの局所係数(隣接間の実辺/可能辺)の平均。母数 0 なら 0.0。
    - diameter_lcc: 最大連結成分(タイは最小ノード id)内の全点対 BFS 最遠距離。N≤数百で O(N·E)。
    全て決定論(ノードは id 昇順・純 Python・乱数なし)。"""
    adj: dict[int, set] = defaultdict(set)
    for (u, v) in graph:
        adj[u].add(v)
        adj[v].add(u)
    nodes = sorted(adj)
    if not nodes:
        return {"clustering_coeff": 0.0, "diameter_lcc": 0, "lcc_size": 0,
                "clustering_n_nodes": 0}
    # 平均局所クラスタ係数(次数 ≥2 のノードのみ=定義できるノードの平均。正直な母数も返す)
    coeffs = []
    for n in nodes:
        nb = sorted(adj[n])
        k = len(nb)
        if k < 2:
            continue
        links = sum(1 for i in range(k) for j in range(i + 1, k)
                    if nb[j] in adj[nb[i]])
        coeffs.append(2.0 * links / (k * (k - 1)))
    cc = round(sum(coeffs) / len(coeffs), 6) if coeffs else 0.0
    # 連結成分(id 昇順の決定論走査)→ 最大成分(タイは最小 id)
    seen: set = set()
    best: list = []
    for start in nodes:
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        i = 0
        while i < len(comp):
            for m in sorted(adj[comp[i]]):
                if m not in seen:
                    seen.add(m)
                    comp.append(m)
            i += 1
        if len(comp) > len(best) or (len(comp) == len(best)
                                     and comp and best and min(comp) < min(best)):
            best = comp
    # 直径 = LCC 内の全ノードからの BFS 最遠距離の最大
    lcc = set(best)
    diam = 0
    for src in sorted(lcc):
        dist = {src: 0}
        q = [src]
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            for m in sorted(adj[u]):
                if m in lcc and m not in dist:
                    dist[m] = dist[u] + 1
                    q.append(m)
        if dist:
            diam = max(diam, max(dist.values()))
    return {"clustering_coeff": cc, "diameter_lcc": int(diam),
            "lcc_size": len(best), "clustering_n_nodes": len(coeffs)}


# --------------------------------------------------------------------------- #
# 5. Granovetter: 語彙の新規採用の出所 A を last_from 規則で帰属(measure.agent_features の鏡)
# --------------------------------------------------------------------------- #
def attribute_adoptions(events: list[dict]) -> list[tuple]:
    """[(source_A, adopter_B, item_id, step), ...]。B 宛の直近 transmission の from を出所 A とする。"""
    ordered = sorted(events, key=lambda e: e["step"])
    last_from: dict[tuple, int] = {}     # (item_id, receiver) -> from
    out: list[tuple] = []
    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if k == "transmission":
            iid, frm = p.get("item_id"), p.get("from")
            if iid is not None:
                last_from[(iid, aid)] = frm            # aid=受信者(provenance の仕様)
        elif k == "label_adopt":
            src = last_from.get((p.get("item_id"), aid))
            if isinstance(src, int) and src >= 0 and src != aid:
                out.append((src, aid, p.get("item_id"), e["step"]))
    return out


# --------------------------------------------------------------------------- #
# メイン解析
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, min_strength: int = MIN_STRENGTH, top_k: int = 15) -> dict:
    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    n_agents = len(agents) or (max((e["agent_id"] for e in events
                                    if e["agent_id"] >= 0), default=-1) + 1)
    name_of = {int(a["id"]): a.get("name", f"a{a['id']}") for a in agents if "id" in a}
    run_name = os.path.basename(os.path.normpath(run_dir))

    graph = {e: c for e, c in build_weighted_graph(events).items() if c >= min_strength}
    comm = m.communities(events, n_agents)                # 決定論 LPA(既存資産)
    comm_sizes = defaultdict(int)
    for _node, c in comm.items():
        comm_sizes[c] += 1
    # 2 人以上のコミュニティのみ「実体あるコミュニティ」と数える(孤立 singleton を除く)
    real_comms = sorted((c for c, n in comm_sizes.items() if n >= 2))

    # ---- 辺種別(bridge / internal)+ 弱い紐帯の強さ(観測A)----
    bridge_edges: list[tuple] = []
    internal_edges: list[tuple] = []
    incident_total: dict[int, int] = defaultdict(int)
    incident_bridge: dict[int, int] = defaultdict(int)
    for (u, v), c in graph.items():
        is_bridge = comm.get(u) != comm.get(v)
        (bridge_edges if is_bridge else internal_edges).append((u, v, c))
        for node in (u, v):
            incident_total[node] += 1
            if is_bridge:
                incident_bridge[node] += 1
    n_edges = len(graph)
    n_bridge = len(bridge_edges)
    n_internal = len(internal_edges)

    def _mean_strength(edges):
        return round(sum(c for *_e, c in edges) / len(edges), 4) if edges else 0.0

    bridge_strength = _mean_strength(bridge_edges)
    internal_strength = _mean_strength(internal_edges)
    topo = clustering_and_diameter(graph)      # 第64: クラスタ係数・直径(フェーズ3観測指標)

    # ---- per-agent brokerage(接する辺の bridge 率)----
    brokers = []
    for aid in sorted(incident_total):
        tot = incident_total[aid]
        br = incident_bridge.get(aid, 0)
        brokers.append({"id": aid, "name": name_of.get(aid, f"a{aid}"),
                        "brokerage": round(br / tot, 4) if tot else 0.0,
                        "n_bridge": br, "n_edges": tot})
    top_brokers = sorted(brokers, key=lambda b: (-b["brokerage"], -b["n_edges"], b["id"]))[:top_k]

    # ---- Granovetter 検証(観測B): 採用の到来辺種別 ----
    adoptions = attribute_adoptions(events)
    adopt_bridge = adopt_internal = 0
    for src, dst, _iid, _step in adoptions:
        if comm.get(src) != comm.get(dst):
            adopt_bridge += 1
        else:
            adopt_internal += 1
    n_adopt = adopt_bridge + adopt_internal
    bridge_edge_frac = round(n_bridge / n_edges, 6) if n_edges else 0.0
    bridge_adopt_frac = round(adopt_bridge / n_adopt, 6) if n_adopt else None

    return {
        "run": run_name,
        "n_agents": n_agents,
        "params": {"min_strength": min_strength, "top_k": top_k,
                   "detector": "measure.communities (deterministic LPA on speak/dm graph)",
                   "channels": "speak(speaker->hearers) + dm(sender->to)"},
        "graph": {"n_edges": n_edges, "n_bridge_edges": n_bridge,
                  "n_internal_edges": n_internal, "bridge_edge_fraction": bridge_edge_frac,
                  "bridge_mean_strength": bridge_strength,
                  "internal_mean_strength": internal_strength,
                  "weak_tie_signal": bool(bridge_edges and internal_edges
                                          and bridge_strength < internal_strength),
                  # 第64バッチ フェーズ3: 計画書§4 の観測指標(クラスタ係数/直径)。
                  # endogenous_invite の弱い紐帯枠が構造(局所閉鎖性・到達距離)を動かすかの事後窓。
                  "clustering_coeff": topo["clustering_coeff"],
                  "clustering_n_nodes": topo["clustering_n_nodes"],
                  "diameter_lcc": topo["diameter_lcc"],
                  "lcc_size": topo["lcc_size"]},
        "community": {"n_communities": len(real_comms),
                      "sizes": sorted((comm_sizes[c] for c in real_comms), reverse=True)},
        "brokerage": top_brokers,
        "granovetter": {
            "n_adoptions_attributed": n_adopt,
            "via_bridge": adopt_bridge, "via_internal": adopt_internal,
            "bridge_adopt_fraction": bridge_adopt_frac,
            "bridge_edge_fraction": bridge_edge_frac,
            # 採用の bridge 率 > 辺の bridge 率 なら「橋辺が新規情報を運ぶ」構造的関連(因果ではない)
            "over_representation": (round(bridge_adopt_frac - bridge_edge_frac, 6)
                                    if bridge_adopt_frac is not None else None),
        },
        "note": ("因果ではなく構造上の関連の観測記録。橋辺『経由で』到来したという時系列・経路の突合であり、"
                 "橋辺が採用を引き起こした証明ではない(交絡・逆因果を排除しない)。コミュニティはラン全体の"
                 "集約構造=近似。日常観察方針(介入しない)。"),
    }


# --------------------------------------------------------------------------- #
# 日本語レポート
# --------------------------------------------------------------------------- #
def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def write_report(result: dict, path: str) -> None:
    g = result["graph"]
    gr = result["granovetter"]
    co = result["community"]
    L: list[str] = []
    L.append(f"# 弱い紐帯レンズ 観測レポート: {result['run']}\n")
    L.append("Granovetter「弱い紐帯の強さ」を **介入せず** 観測した結果(日常観察方針)。")
    L.append("弱い紐帯=コミュニティをまたぐ橋辺が、新規情報(語彙の新規採用)の運び手かを構造から観る。\n")
    L.append("## 設定")
    L.append(f"- エージェント数: {result['n_agents']} / コミュニティ数(2人以上): {co['n_communities']}"
             f"(規模 {co['sizes']})")
    L.append(f"- コミュニティ検出: {result['params']['detector']}")
    L.append(f"- 接触チャネル: {result['params']['channels']} / 最小接触頻度: "
             f"{result['params']['min_strength']}\n")
    L.append("## 辺の種別と『弱い紐帯の強さ』(観測A)")
    L.append(f"- 総辺数 {g['n_edges']} / うち bridge(別コミュ)={g['n_bridge_edges']}"
             f"({_pct(g['bridge_edge_fraction'])})/ internal(同コミュ)={g['n_internal_edges']}")
    L.append(f"- 平均接触頻度: bridge={g['bridge_mean_strength']} / internal={g['internal_mean_strength']}")
    L.append(f"- クラスタ係数(平均局所・次数≥2 の {g.get('clustering_n_nodes', 0)} ノード)= "
             f"{g.get('clustering_coeff', 0.0)} / 直径(最大連結成分 {g.get('lcc_size', 0)} 人)= "
             f"{g.get('diameter_lcc', 0)}(第64: フェーズ3観測指標=計画書§4)")
    if g["weak_tie_signal"]:
        L.append("  → bridge の方が接触頻度が低い = **弱い紐帯が橋になっている**"
                 "(Granovetter の構造的含意と整合)")
    else:
        L.append("  → bridge が internal より弱いとは言えない(この観測では弱い紐帯=橋の含意は出ていない)")
    L.append("")
    L.append("## 語彙新規採用の到来辺種別(観測B: Granovetter 検証)")
    L.append(f"- 帰属できた採用 {gr['n_adoptions_attributed']} 件 / "
             f"bridge 経由 {gr['via_bridge']} ・ internal 経由 {gr['via_internal']}")
    L.append(f"- 採用の bridge 経由率 = {_pct(gr['bridge_adopt_fraction'])} "
             f"(母集団の bridge 辺割合 = {_pct(gr['bridge_edge_fraction'])})")
    if gr["over_representation"] is not None and gr["over_representation"] > 0:
        L.append(f"  → 採用は辺の割合より **bridge 経由に偏っている**(+{_pct(gr['over_representation'])})"
                 "= 橋辺が新規情報を運ぶ構造的関連(因果ではない)")
    elif gr["over_representation"] is not None:
        L.append(f"  → 採用は bridge 経由に偏っていない({_pct(gr['over_representation'])})")
    else:
        L.append("  → 帰属できた採用が無い(vocab/transmission が疎 or relations/labeling OFF)")
    L.append("")
    L.append("## 橋渡し度の高い住民(brokerage=接する辺の bridge 率)")
    L.append("| 住民 | brokerage | bridge辺 | 総辺 |")
    L.append("|---|---|---|---|")
    for b in result["brokerage"]:
        L.append(f"| {b['name']} | {_pct(b['brokerage'])} | {b['n_bridge']} | {b['n_edges']} |")
    L.append("")
    L.append(f"> {result['note']}\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第59バッチ スライスb: 弱い紐帯レンズの事後分析(L1 を読むだけ)")
    ap.add_argument("run_dir", help="例: runs/day30")
    ap.add_argument("--min-strength", type=int, default=MIN_STRENGTH,
                    help="この接触頻度未満の辺を無視(既定 1)")
    ap.add_argument("--top-k", type=int, default=15, help="brokerage 上位K(既定 15)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    result = analyze(args.run_dir, min_strength=args.min_strength, top_k=args.top_k)
    js_path = os.path.join(args.run_dir, "weak_ties.json")
    md_path = os.path.join(args.run_dir, "weak_ties_report.md")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
    write_report(result, md_path)

    g, gr = result["graph"], result["granovetter"]
    print(f"[weak_ties] {args.run_dir}: 辺 {g['n_edges']}(bridge {g['n_bridge_edges']})/ "
          f"採用帰属 {gr['n_adoptions_attributed']}(bridge経由率 {gr['bridge_adopt_fraction']})")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
