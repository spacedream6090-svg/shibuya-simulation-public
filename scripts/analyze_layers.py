#!/usr/bin/env python
"""S-12 伝播チャネルの三層(offline / online / broadcast)別グラフ(SV2-B・読み取り専用)。

    python scripts/analyze_layers.py runs/<name>
    python scripts/analyze_layers.py runs/<name> --out experiments/layers

■ 正典
  層の定義・辺の張り方・broadcast の扱い・F7 への影響は
  **事前登録 `docs/plans/stationarity-preregistration.md` §1.5**(承認対象)。
  本スクリプトはその機械実装であり、**層をここで定義し直さない**。

■ ★ 既存資産を壊さない配置(なぜ analyze_weak_ties.py を改修しないか)
  §1.2 の合成グラフ(speak + dm を 1 枚に混ぜたもの)は**ノイズ床 F の算出に使われている**。
  そこへ層を持ち込むと、既に取った診断値の意味が変わる。したがって
  **`analyze_weak_ties.py` は 1 バイトも触らず、層別は本スクリプトに新設する**。
  両者は別の量であり、報告ではそう書く。

■ 2 系統のグラフ(混ぜない)
  (A) **相互作用グラフ** … 誰と誰が接触したか。`transmission` の有無に依存しない。
      offline   : speak(speaker → hearers)
      online    : dm(sender → to)+ sns_read(reader → authors)
                  + sns_like / sns_reshare(反応者 → author)
      broadcast : 辺を張らない。相手が **-1(媒体)** の受信と news_read を**到達件数**で数える
  (B) **伝播グラフ** … 何が実際に伝わったか。`transmission.payload.channel` で層を決める。
      辺 = (payload.from → agent_id)。from == -1 は辺にせず broadcast の到達件数へ。

■ ★ 実測で判明したこと(リサーチ文書の 5 種表の訂正 = 事前登録 §1.5 に反映済み)
  `channel` の実値は **6 種**である(`event` が抜けていた)。`event` は
  `src/society/tools.py` L1085–1087 の「催しのホストが参加者へ語を教える」経路で、
  **送り手は実在の人**なので **offline**(同席が要る)。
  また **`dm` チャネルは実測でごく僅か**(既存ラン全体で 0.03% 台)であり、
  「online = DM 主体」という直感は実データでは成立しない。
  → 本スクリプトは層の写像を**宣言だけで済ませず**、チャネルごとの `from == -1` 率を
  出力して、offline/online に分類した channel に `-1` が出たら **警告を立てる**。
  **未知の channel は `unmapped` として別掲し、勝手にどれかの層へ入れない。**

■ 出す量
  - 層ごと: ノード数 / 辺数 / 平均次数 / degree_gini / 上位 10% シェア / 最大次数
            / CCDF(F7 の「重い裾」を層別に見るための素材)/ コミュニティ数(決定論 LPA)
  - **層間 edge overlap**(層 α に辺があるとき層 α' にも辺がある条件付き確率。
    Szell et al. 2010 の multiplex の標準量)
  - broadcast: 到達件数・到達人数(辺を張らない)
  - (A) と (B) の**辺集合の重なり**(冗長なら冗長と報告する)

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。凍結ファイル `society.observer.measure` は
  **import するだけ**(決定論 LPA を再実装しない)。決定論: ノード・辺はソート順。
  データが無いときは捏造せず明示する。
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

# ★ 事前登録 §1.5 の写像(ここが唯一の源。変更は事前登録の改訂を伴う)
CHANNEL_LAYER = {
    "face": "offline",
    "event": "offline",      # ★2026-08-06 実測で追加(tools.py の催しホスト経路)
    "dm": "online",
    "sns": "online",
    "search": "broadcast",
    "news": "broadcast",
}
PERSON_LAYERS = ("offline", "online")     # 対人グラフとして辺を張る層
BROADCAST_LAYER = "broadcast"
UNMAPPED = "unmapped"

# (A) 相互作用グラフの材料(payload をパースする kind はこれだけ)
INTERACTION_KINDS = ("speak", "dm", "sns_read", "sns_like", "sns_reshare",
                     "news_read", "transmission")
CCDF_POINTS = (1, 2, 5, 10, 20, 50, 100)


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


def _key(u: int, v: int) -> tuple:
    return (u, v) if u < v else (v, u)


def gini(values) -> float | None:
    """Gini 係数(既存 5 実装と同じ定義。6 個目を書かないため定義を揃えてある)。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return None
    s = sum(xs)
    if s <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def build_layer_graphs(events: list[dict]) -> dict:
    """(A) 相互作用グラフを層別に組む。返り値 {layer: {(u,v): weight}} + broadcast 集計。

    無向・重み = 共起回数。自己ループと負 id(= 媒体)は対人の辺にしない。
    """
    W: dict[str, dict] = {ly: defaultdict(float) for ly in PERSON_LAYERS}
    bcast_hits = 0                                  # 媒体から受け取った回数
    bcast_receivers: set[int] = set()

    def add(layer: str, u, v) -> None:
        if not isinstance(u, int) or not isinstance(v, int):
            return
        if u < 0 or v < 0 or u == v:
            return
        W[layer][_key(u, v)] += 1.0

    for e in events:
        k, aid, p = e["kind"], e.get("agent"), e.get("payload") or {}
        if not isinstance(aid, int):
            continue
        if k == "speak":
            for h in p.get("hearers", []) or []:
                add("offline", aid, h)
        elif k == "dm":
            add("online", aid, p.get("to"))
        elif k == "sns_read":
            for au in p.get("authors", []) or []:
                if isinstance(au, int) and au < 0:
                    bcast_hits += 1
                    if aid >= 0:
                        bcast_receivers.add(aid)
                else:
                    add("online", aid, au)
        elif k in ("sns_like", "sns_reshare"):
            au = p.get("author")
            if isinstance(au, int) and au < 0:
                bcast_hits += 1
                if aid >= 0:
                    bcast_receivers.add(aid)
            else:
                add("online", aid, au)
        elif k == "news_read":
            n = len(p.get("titles", []) or []) or 1
            bcast_hits += n
            if aid >= 0:
                bcast_receivers.add(aid)
    return {"W": {ly: dict(W[ly]) for ly in PERSON_LAYERS},
            "broadcast": {"reach_events": int(bcast_hits),
                          "reach_agents": len(bcast_receivers)}}


def build_transmission_graphs(events: list[dict]) -> dict:
    """(B) 伝播グラフを channel → 層の写像で組む。層の宣言の**検算**も同時に行う。

    返り値 {"W": {layer: {(u,v): w}}, "channel_stats": {...}, "warnings": [...]}
    """
    W: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    stats: dict[str, dict] = {}
    bcast_hits = 0
    bcast_receivers: set[int] = set()
    warnings: list[str] = []

    for e in events:
        if e["kind"] != "transmission":
            continue
        p = e.get("payload") or {}
        ch = p.get("channel")
        to = e.get("agent")
        frm = p.get("from")
        ch_key = str(ch) if ch is not None else "_missing"
        st = stats.setdefault(ch_key, {"n": 0, "n_from_minus1": 0, "layer": None})
        st["n"] += 1
        layer = CHANNEL_LAYER.get(ch_key, UNMAPPED)
        st["layer"] = layer
        if not isinstance(frm, int) or frm < 0:
            st["n_from_minus1"] += 1
            bcast_hits += 1
            if isinstance(to, int) and to >= 0:
                bcast_receivers.add(to)
            continue
        if layer in PERSON_LAYERS and isinstance(to, int) and to >= 0 and to != frm:
            W[layer][_key(frm, to)] += 1.0
        elif layer == BROADCAST_LAYER:
            bcast_hits += 1
            if isinstance(to, int) and to >= 0:
                bcast_receivers.add(to)
        elif layer == UNMAPPED and isinstance(to, int) and to >= 0 and to != frm:
            W[UNMAPPED][_key(frm, to)] += 1.0

    # 検算: 対人層に分類した channel から -1 が出たら、写像の前提が壊れている
    for ch_key, st in sorted(stats.items()):
        if st["layer"] in PERSON_LAYERS and st["n_from_minus1"] > 0:
            warnings.append(
                f"channel `{ch_key}` は {st['layer']} に分類しているが "
                f"from == -1 が {st['n_from_minus1']}/{st['n']} 件ある"
                "(事前登録 §1.5 の前提が壊れている)")
        if st["layer"] == UNMAPPED:
            warnings.append(
                f"未知の channel `{ch_key}`({st['n']} 件)。"
                "勝手に層へ入れず unmapped として別掲する(事前登録 §1.5 の改訂が要る)")
    return {"W": {ly: dict(w) for ly, w in sorted(W.items())},
            "channel_stats": {k: {**v, "from_minus1_frac": _r(
                v["n_from_minus1"] / v["n"] if v["n"] else None)}
                for k, v in sorted(stats.items())},
            "broadcast": {"reach_events": int(bcast_hits),
                          "reach_agents": len(bcast_receivers)},
            "warnings": warnings}


def layer_metrics(W: dict) -> dict:
    """1 層のグラフ指標。F7(次数分布の重い裾)を層別に見るための素材を出す。"""
    if not W:
        return {"n_nodes": 0, "n_edges": 0, "mean_degree": None, "max_degree": None,
                "degree_gini": None, "top10pct_share": None, "ccdf": {},
                "n_communities": 0, "largest_community": 0}
    deg: dict[int, int] = defaultdict(int)
    for (u, v) in W:
        deg[u] += 1
        deg[v] += 1
    degs = sorted(deg.values(), reverse=True)
    n = len(degs)
    total = sum(degs)
    k10 = max(1, int(round(n * 0.10)))
    ccdf = {str(x): _r(sum(1 for d in degs if d >= x) / n, 4) for x in CCDF_POINTS}
    # 決定論 LPA(凍結 measure を import。再実装しない)
    n_comm = largest = 0
    try:
        from society.observer import measure as m
        synth = [{"kind": "speak", "agent_id": a, "payload": {"hearers": [b]}}
                 for (a, b) in sorted(W)]
        lab = m.communities(synth, 0)
        groups: dict[int, int] = defaultdict(int)
        for _node, c in lab.items():
            groups[c] += 1
        sizes = [s for s in groups.values() if s >= 2]
        n_comm, largest = len(sizes), (max(sizes) if sizes else 0)
    except Exception:
        n_comm = largest = 0
    return {"n_nodes": n, "n_edges": len(W),
            "mean_degree": _r(total / n), "max_degree": int(degs[0]),
            "degree_gini": _r(gini(degs), 4),
            "top10pct_share": _r(sum(degs[:k10]) / total, 4) if total else None,
            "ccdf": ccdf, "n_communities": int(n_comm),
            "largest_community": int(largest)}


def edge_overlap(Wa: dict, Wb: dict) -> dict:
    """層間 edge overlap。P(b に辺 | a に辺) と Jaccard を返す(Szell et al. 2010)。"""
    A, B = set(Wa), set(Wb)
    inter = len(A & B)
    union = len(A | B)
    return {"n_a": len(A), "n_b": len(B), "n_both": inter,
            "p_b_given_a": _r(inter / len(A), 4) if A else None,
            "p_a_given_b": _r(inter / len(B), 4) if B else None,
            "jaccard": _r(inter / union, 4) if union else None}


# --------------------------------------------------------------------------- #
# 解析本体
# --------------------------------------------------------------------------- #
def _load_events(run_dir: str) -> list[dict]:
    """必要 kind の payload だけをパースするストリーム読み(全件 RAM 展開しない)。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.exists(path):
        return []
    want = set(INTERACTION_KINDS)
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
            out.append({"step": int(st), "agent": int(aid), "kind": kind,
                        "payload": payload})
    return out


def analyze(run_dir: str) -> dict:
    events = _load_events(run_dir)
    inter = build_layer_graphs(events)
    trans = build_transmission_graphs(events)

    res: dict = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "spec": {"channel_layer_map": dict(sorted(CHANNEL_LAYER.items())),
                 "preregistration": "docs/plans/stationarity-preregistration.md §1.5",
                 "note": "§1.2 の合成グラフ(analyze_weak_ties)とは別の量である"},
        "n_events_used": len(events),
        "interaction": {"layers": {}, "broadcast": inter["broadcast"]},
        "transmission": {"layers": {}, "broadcast": trans["broadcast"],
                         "channel_stats": trans["channel_stats"],
                         "warnings": trans["warnings"]},
    }
    if not events:
        res["status"] = "NO_DATA"
        return res
    res["status"] = "OK"

    for ly in PERSON_LAYERS:
        res["interaction"]["layers"][ly] = layer_metrics(inter["W"].get(ly, {}))
    for ly, W in sorted(trans["W"].items()):
        res["transmission"]["layers"][ly] = layer_metrics(W)

    res["interaction"]["edge_overlap_offline_online"] = edge_overlap(
        inter["W"].get("offline", {}), inter["W"].get("online", {}))
    res["transmission"]["edge_overlap_offline_online"] = edge_overlap(
        trans["W"].get("offline", {}), trans["W"].get("online", {}))
    # (A) と (B) の重なり — 「話しかけた」と「語が乗った」がどれだけ同じ辺か
    res["cross_system_overlap"] = {
        ly: edge_overlap(inter["W"].get(ly, {}), trans["W"].get(ly, {}))
        for ly in PERSON_LAYERS}
    if not trans["channel_stats"]:
        res["transmission"]["warnings"].append(
            "transmission イベントが 0 件。(B) 伝播グラフは空である"
            "(語彙/ラベルの伝播を記録していないラン)。捏造しない。")
    return res


def render(res: dict) -> str:
    L: list[str] = []
    L.append(f"# S-12 伝播チャネル三層 — `{res.get('run','?')}`")
    L.append("")
    L.append("> 層の定義の正典は **事前登録 §1.5**。`analyze_weak_ties.py` の合成グラフ(§1.2)"
             "とは**別の量**である。")
    L.append("")
    if res.get("status") != "OK":
        L.append("**NO_DATA** — 層を組める相互作用イベントが無い。捏造しない。")
        return "\n".join(L) + "\n"
    L.append(f"- 使用イベント: {res['n_events_used']}")
    L.append("")
    L.append("## channel の実測と層の写像(宣言の検算)")
    L.append("")
    cs = res["transmission"]["channel_stats"]
    if cs:
        L.append("| channel | 層 | 件数 | from == -1 | 比率 |")
        L.append("|---|---|---|---|---|")
        for ch, st in sorted(cs.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
            L.append(f"| `{ch}` | {st['layer']} | {st['n']} | {st['n_from_minus1']} "
                     f"| {st['from_minus1_frac']} |")
    else:
        L.append("- `transmission` イベントなし。")
    L.append("")
    if res["transmission"]["warnings"]:
        L.append("### ★警告(層の前提が壊れている / 未知の channel)")
        L.append("")
        for w in res["transmission"]["warnings"]:
            L.append(f"- {w}")
        L.append("")

    for sysname, title in (("interaction", "(A) 相互作用グラフ(誰と誰が接触したか)"),
                           ("transmission", "(B) 伝播グラフ(何が実際に伝わったか)")):
        L.append(f"## {title}")
        L.append("")
        layers = res[sysname]["layers"]
        if layers:
            L.append("| 層 | ノード | 辺 | 平均次数 | 最大次数 | degree_gini "
                     "| 上位10%シェア | コミュニティ数 |")
            L.append("|---|---|---|---|---|---|---|---|")
            for ly, mm in sorted(layers.items()):
                L.append(f"| **{ly}** | {mm['n_nodes']} | {mm['n_edges']} "
                         f"| {mm['mean_degree']} | {mm['max_degree']} "
                         f"| {mm['degree_gini']} | {mm['top10pct_share']} "
                         f"| {mm['n_communities']} |")
        else:
            L.append("- (層が空)")
        b = res[sysname]["broadcast"]
        L.append("")
        L.append(f"- **broadcast 層**(辺を張らない): 到達 {b['reach_events']} 件 / "
                 f"到達人数 {b['reach_agents']} 名")
        ov = res[sysname].get("edge_overlap_offline_online")
        if ov:
            L.append(f"- 層間 edge overlap(offline↔online): "
                     f"P(online|offline)={ov['p_b_given_a']} / "
                     f"P(offline|online)={ov['p_a_given_b']} / Jaccard={ov['jaccard']}")
        L.append("")
    L.append("## (A) と (B) の辺の重なり(「話しかけた」と「語が乗った」)")
    L.append("")
    L.append("| 層 | (A)辺 | (B)辺 | 共通 | Jaccard |")
    L.append("|---|---|---|---|---|")
    for ly, ov in sorted(res.get("cross_system_overlap", {}).items()):
        L.append(f"| {ly} | {ov['n_a']} | {ov['n_b']} | {ov['n_both']} | {ov['jaccard']} |")
    L.append("")
    L.append("## F7(次数分布の重い裾)を層別に読むための CCDF")
    L.append("")
    for sysname in ("interaction", "transmission"):
        for ly, mm in sorted(res[sysname]["layers"].items()):
            if not mm["ccdf"]:
                continue
            cells = " / ".join(f"≥{k}: {v}" for k, v in
                               sorted(mm["ccdf"].items(), key=lambda kv: int(kv[0])))
            L.append(f"- {sysname}.{ly} … {cells}")
    L.append("")
    L.append("> **層ごとに合否を出して併記する**(両層で裾が出た場合にのみ「再現できた」"
             "と数えるのではない)。層で結果が割れること自体が multiplex の知見である"
             "(事前登録 §1.5)。")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-12 伝播チャネル三層(読み取り専用)")
    ap.add_argument("run", help="ランディレクトリ")
    ap.add_argument("--out", default=None, help="出力先(既定 <run>/analysis)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.run):
        print(f"[analyze_layers] ランが無い: {a.run}", file=sys.stderr)
        return 2
    res = analyze(a.run)
    out = a.out or os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "layers.json")
    mp = os.path.join(out, "layers.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    print(f"[analyze_layers] {res.get('status')} events={res.get('n_events_used')} "
          f"warnings={len(res['transmission']['warnings'])}")
    print(f"  -> {jp}")
    print(f"  -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
