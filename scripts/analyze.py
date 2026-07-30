#!/usr/bin/env python
"""Phase E: 単一 run の観測・計測 CLI。

    python scripts/analyze.py runs/<name>

runs/<name>/analysis/ に JSON(agent_features / items / network_windows /
collective)・図 PNG・report.md を出力する。traits が無くても動く(R² は skip)。
図中テキストは日本語フォント文字化け回避のためすべて英語。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from society.observer import measure as m  # noqa: E402
from society.observer import stream as st  # noqa: E402

# Okabe-Ito カラーブラインドセーフ配色
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "font.size": 10,
})


# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _dump(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, default=_jsonable)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def coinage_contexts(events):
    """造語(vocab_coin)の発生文脈一覧。促進はしない=発生過程ときっかけの自然観察。

    scheduler が vocab_coin の payload に載せた文脈(fire_reason/drive/直近記憶/同席者/
    feed有無/場所/採用済み語数)を素直に取り出す。メディア発(media=True)は除く。
    造語0件でも空リストを返して壊れない。
    """
    out = []
    for e in events:
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


def coinage_trigger_dist(coinages):
    """造語のきっかけ(fire_reason)分布。0件なら空 dict。"""
    dist = defaultdict(int)
    for c in coinages:
        dist[c.get("fire_reason") or "unknown"] += 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------- #
_TOOL_KINDS = ("event_host", "event_attend", "flyer_post", "flyer_view",
               "flyer_expire", "group_found", "group_join", "proposal",
               "proposal_support", "proposal_passed", "venture_open",
               "venture_sale", "venture_close")


def tool_usage(events):
    """「世界を変える」ツールの使用・使用者・効果の一覧(0 使用でも壊れない)。

    R4: 効果は客観カウント(参加人数・署名数・売上)。LLM 審判は関与しない。
    """
    totals = {k: 0 for k in _TOOL_KINDS}
    users = defaultdict(lambda: defaultdict(int))     # kind -> {agent_id: count}
    timeline = defaultdict(lambda: defaultdict(int))  # step -> {kind: count}
    events_map, groups_map, proposals_map, ventures_map = {}, {}, {}, {}

    for e in sorted(events, key=lambda x: x["step"]):
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
def _adoption_curves(events, item_ids, steps):
    """item ごとに step 別の累積 distinct 採用者数を返す。"""
    want = set(item_ids)
    by_item = {iid: defaultdict(set) for iid in want}
    for e in events:
        if e["kind"] == "transmission":
            iid = e["payload"].get("item_id")
            if iid in want:
                to = e["agent_id"]
                if isinstance(to, int) and to >= 0:
                    by_item[iid][e["step"]].add(to)
    curves = {}
    for iid in item_ids:
        run: set = set()
        cum = []
        for s in steps:
            run |= by_item[iid].get(s, set())
            cum.append(len(run))
        curves[iid] = cum
    return curves


def fig_adoption(events, cascades, steps, path):
    top = sorted(cascades, key=lambda c: c["size"], reverse=True)[:8]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if top:
        curves = _adoption_curves(events, [c["item_id"] for c in top], steps)
        for i, c in enumerate(top):
            ax.plot(steps, curves[c["item_id"]], color=PALETTE[i % len(PALETTE)],
                    lw=1.8, label=f"{c['item_id']} (n={c['size']})")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative adopters")
    ax.set_title("Adoption S-curves (top 8 items)")
    _save(fig, path)


def fig_cascade(events, cascades, path):
    big = max(cascades, key=lambda c: c["size"], default=None)
    fig, ax = plt.subplots(figsize=(7, 5))
    if big and big["size"] > 0:
        iid = big["item_id"]
        edges = set()
        for e in events:
            if e["kind"] == "transmission" and e["payload"].get("item_id") == iid:
                edges.add((e["payload"].get("from"), e["agent_id"]))
        adj = defaultdict(list)
        parents = defaultdict(list)
        nodes = set()
        for a, b in edges:
            adj[a].append(b)
            parents[b].append(a)
            nodes.add(a)
            nodes.add(b)
        # 最長路レベル付け(depth 指標に一致)。メモ化 + 循環スキップで O(V+E)。
        memo, onstack = {}, set()

        def _depth_to(u):
            if u in memo:
                return memo[u]
            onstack.add(u)
            best = 0
            for p in parents.get(u, ()):
                if p in onstack:
                    continue
                best = max(best, 1 + _depth_to(p))
            onstack.discard(u)
            memo[u] = best
            return best

        level = {n: _depth_to(n) for n in nodes}
        by_level = defaultdict(list)
        for n in sorted(nodes):
            by_level[level[n]].append(n)
        pos = {}
        for lv, ns in by_level.items():
            span = max(len(ns) - 1, 1)
            for j, n in enumerate(sorted(ns)):
                pos[n] = ((j - (len(ns) - 1) / 2) / span, -lv)
        for a, b in edges:
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="#BBBBBB", lw=0.6, zorder=1)
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        cols = [PALETTE[3] if n < 0 else PALETTE[0] for n in nodes]
        ax.scatter(xs, ys, s=45, c=cols, zorder=2, edgecolors="white", linewidths=0.5)
        if len(nodes) <= 40:
            for n in nodes:
                ax.annotate(str(n), pos[n], fontsize=6, ha="center", va="center")
        ax.set_title(f"Largest cascade: {iid} "
                     f"(size={big['size']}, depth={big['depth']}, "
                     f"media={big['media']})")
        ax.set_xlabel("(-1 = media/search/news origin)")
        ax.set_ylabel("depth (root at top)")
        ax.set_xticks([])
    else:
        ax.text(0.5, 0.5, "no cascade", ha="center", va="center")
    _save(fig, path)


def fig_network(windows, path):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    xs = [w["step_start"] for w in windows]
    md = [w["mean_degree"] for w in windows]
    ch = [w["centrality_churn"] for w in windows]
    ax1.plot(xs, md, color=PALETTE[0], marker="o", lw=1.8, label="mean degree")
    ax1.set_xlabel("step (window start)")
    ax1.set_ylabel("mean degree", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax2 = ax1.twinx()
    ax2.plot(xs, ch, color=PALETTE[3], marker="s", lw=1.8, label="centrality churn")
    ax2.set_ylabel("centrality churn (top-5 turnover)", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])
    ax1.set_title("Conversation network over time")
    _save(fig, path)


def fig_rhythm(collective, path):
    steps = collective["step"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for i, key in enumerate(["n_fires", "n_sleeping", "n_working", "n_moving"]):
        vals = collective.get(key)
        if vals and any(v is not None for v in vals):
            ax.plot(steps, [v if v is not None else 0 for v in vals],
                    color=PALETTE[i % len(PALETTE)], lw=1.6, label=key)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no rhythm columns in l2", ha="center", va="center")
    else:
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("step")
    ax.set_ylabel("count of agents")
    ax.set_title("City rhythm (l2 state counts)")
    _save(fig, path)


def fig_drive(events, path):
    granted = defaultdict(int)
    rejected = defaultdict(int)
    for e in events:
        if e["kind"] == "drive_request":
            if e["payload"].get("granted"):
                granted[e["step"]] += 1
            else:
                rejected[e["step"]] += 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if granted or rejected:
        steps = sorted(set(granted) | set(rejected))
        g = [granted[s] for s in steps]
        r = [rejected[s] for s in steps]
        ax.bar(steps, g, color=PALETTE[2], label="granted")
        ax.bar(steps, r, bottom=g, color=PALETTE[3], label="rejected")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "no drive_request events in this run",
                ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_ylabel("drive_request count")
    ax.set_title("Drive-firing histogram (granted / rejected)")
    _save(fig, path)


# --------------------------------------------------------------------------- #
def _fmt_r2(block):
    r2 = block.get("r2")
    if r2 is None:
        return f"n/a (n={block.get('n', 0)})"
    adj = block.get("adj_r2")
    adj_s = f"{adj:.3f}" if adj is not None else "n/a"
    return f"{r2:.3f} (adj {adj_s}, n={block['n']})"


def _load_y_weights(run_dir, cli_json):
    """--y-weights JSON を優先。無ければ run の config.yaml の observer.y_weights。

    どちらも無ければ None(= 従来どおり Y_composite を出さない)。
    依存を増やさないため config.yaml は既存の omegaconf(プロジェクト依存)で
    読むが、失敗しても静かに None を返す。
    """
    if cli_json:
        return json.loads(cli_json)
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        yw = OmegaConf.select(cfg, "observer.y_weights")
        if yw:
            return {str(k): float(v)
                    for k, v in OmegaConf.to_container(yw, resolve=True).items()}
    except Exception:
        return None
    return None


def write_report(path, run_dir, feats, cascades, windows, collective,
                 r2, ews_adopt, ews_entropy, tsrc, tnames,
                 coinages=None, trigger_dist=None, tools=None, drift=None,
                 echo=None):
    big = max(cascades, key=lambda c: c["size"], default=None)
    top_ext = max(feats, key=lambda f: f["Y_external"], default=None)
    top_int = max(feats, key=lambda f: f["Y_internal"], default=None)
    md = [w["mean_degree"] for w in windows] or [0]
    lines = []
    lines.append(f"# Analysis report: {os.path.basename(os.path.normpath(run_dir))}\n")
    lines.append("## Scale")
    lines.append(f"- agents: {len(feats)}")
    lines.append(f"- items (coined/transmitted): {len(cascades)}")
    lines.append(f"- network windows: {len(windows)}")
    lines.append(f"- collective series length: {len(collective['step'])} steps\n")
    lines.append("## World-change (Y)")
    if top_ext:
        lines.append(f"- max Y_external: agent {top_ext['id']} "
                     f"({top_ext.get('name')}) = {top_ext['Y_external']} "
                     f"[trans {top_ext['c_transmission']}, adopt "
                     f"{top_ext['c_label_adopt']}, rel "
                     f"{top_ext['c_new_relations']}, sns "
                     f"{top_ext['c_sns_reach']}]")
    if top_int:
        lines.append(f"- max Y_internal: agent {top_int['id']} "
                     f"= {top_int['Y_internal']}")
    lines.append("")
    lines.append("## Largest cascade")
    if big:
        lines.append(f"- item {big['item_id']}: size {big['size']}, "
                     f"adopters {big['n_adopters']}, depth {big['depth']}, "
                     f"t_half {big['t_half']}, media {big['media']}")
        lines.append(f"- channel mix: {big['channel_mix']}")
        # 第70バッチ IDEA①: 同一話者の再送出を除いた「新規伝播」を並記(既存 size は不変)
        lines.append(f"- size_novel (echo excluded): {big.get('size_novel')} "
                     f"/ echo_share {big.get('echo_share')}")
    lines.append("")
    # ---- エコー/自己反復(第70バッチ IDEA①)。伝播数値が LLM の反復癖の関数でないかの検査 ----
    lines.append("## Echo / self-repetition (IDEA-1; observation only, world unchanged)")
    if echo:
        lines.append(f"- utterances: {echo['n_utterances']} "
                     f"(echo {echo['n_echo_utterances']} = "
                     f"{echo['echo_utterance_rate']})")
        lines.append(f"- self_similarity_mean: {echo['self_similarity_mean']} "
                     f"(char {echo['ngram']}-gram Jaccard, "
                     f"threshold {echo['sim_threshold']})")
        lines.append(f"- echo_max_run: {echo['echo_max_run']} "
                     f"(1.000 = one agent repeated a single sentence throughout; "
                     f"such runs must be excluded or handled separately in k* estimation)")
        lines.append(f"- transmission: {echo['n_transmission']} -> novel "
                     f"{echo['n_transmission_novel']} "
                     f"(rate {echo['transmission_novel_rate']}; "
                     f"same speaker re-sending the same word within "
                     f"{echo['window_steps']} steps is excluded)")
        lines.append(f"- label_adopt: {echo['n_label_adopt']} -> novel "
                     f"{echo['n_label_adopt_novel']} "
                     f"(rate {echo['adopt_novel_rate']}; adoption backed by "
                     f"2+ distinct senders)")
    else:
        lines.append("- (not computed)")
    lines.append("")
    lines.append("## Network")
    lines.append(f"- mean degree range: {min(md):.2f} .. {max(md):.2f}")
    lines.append(f"- max centrality churn: "
                 f"{max((w['centrality_churn'] for w in windows), default=0)}\n")
    lines.append("## R^2(Y ~ traits)")
    lines.append(f"- traits source: {tsrc}  names: {tnames}")
    lines.append(f"- R^2 external: {_fmt_r2(r2['external'])}")
    lines.append(f"- R^2 internal: {_fmt_r2(r2['internal'])}")
    if "Y_composite" in r2:
        lines.append(f"- R^2 Y_composite (D1 4-layer): {_fmt_r2(r2['Y_composite'])}")
    if tsrc is None:
        lines.append("- (traits not found -> R^2 skipped; body can add traits.json later)")
    lines.append("")
    lines.append("## EWS (Kendall tau; rising var/ac1 => approaching transition)")
    lines.append(f"- adoption_frac: var_tau {ews_adopt['var_tau']:.3f}, "
                 f"ac1_tau {ews_adopt['ac1_tau']:.3f}")
    lines.append(f"- vocab_entropy: var_tau {ews_entropy['var_tau']:.3f}, "
                 f"ac1_tau {ews_entropy['ac1_tau']:.3f}")
    lines.append("")
    # semantic drift(M1c: 語形/意味がコミュニティ間で分岐したか)
    drift = drift or {}
    lines.append("## Semantic drift (M1c: form / context divergence across communities)")
    if drift:
        lines.append(f"- vocab clusters analyzed: {len(drift)}")
        ranked = sorted(
            drift.items(),
            key=lambda kv: (-(kv[1]["form_divergence"] + kv[1]["context_divergence"]),
                            kv[0]))
        for item, d in ranked[:10]:
            lines.append(
                f"  - \"{item}\": form_div={d['form_divergence']}, "
                f"context_div={d['context_divergence']}, "
                f"communities={d['n_communities']}")
    else:
        lines.append("- (no coined vocabulary -> drift not computed)")
    lines.append("")
    # 造語の自然観察(発生過程ときっかけ。促進はしない)
    coinages = coinages or []
    trigger_dist = trigger_dist or {}
    lines.append("## Coinage (natural observation: how/why new words emerge)")
    lines.append(f"- coinages: {len(coinages)}")
    if coinages:
        lines.append(f"- trigger distribution (fire_reason): {trigger_dist}")
        lines.append("- contexts (first 10):")
        for c in coinages[:10]:
            mem = " | ".join((c.get("recent_mem") or [])[-2:])
            lines.append(
                f"  - step {c['step']} agent {c['agent_id']} "
                f"\"{c.get('text')}\" [reason={c.get('fire_reason')}, "
                f"drive={c.get('drive')}, place={c.get('place')}, "
                f"with={c.get('company_ids')}, saw_feed={c.get('saw_feed')}, "
                f"adopted_n={c.get('adopted_n')}] recent: {mem}")
    else:
        lines.append("- (no coinage in this run)")
    lines.append("")
    # 「世界を変える」ツールの使用・効果(客観カウント。0 使用でも壊れない)
    tools = tools or {"totals": {}, "events": [], "groups": [], "proposals": [],
                      "ventures": [], "top_users": {}}
    totals = tools.get("totals", {})
    lines.append("## World-change tools (neutral affordance; objective effect counts)")
    used = {k: v for k, v in totals.items() if v}
    lines.append(f"- usage totals: {used or '(none used)'}")
    if tools.get("events"):
        lines.append("- events (host, title, attendees):")
        for ev in sorted(tools["events"],
                         key=lambda e: -e.get("attendees", 0))[:5]:
            lines.append(f"  - #{ev['event_id']} agent {ev['host']} "
                         f"\"{ev.get('title')}\" -> {ev['attendees']} attendees")
    if tools.get("groups"):
        lines.append("- groups (founder, name, recruited):")
        for g in sorted(tools["groups"],
                        key=lambda x: -x.get("members_recruited", 0))[:5]:
            lines.append(f"  - #{g['group_id']} agent {g['founder']} "
                         f"\"{g.get('name')}\" -> {g['members_recruited']} joined")
    if tools.get("proposals"):
        lines.append("- proposals (author, signatures, passed):")
        for pr in sorted(tools["proposals"],
                         key=lambda x: -x.get("signatures", 0))[:5]:
            lines.append(f"  - #{pr['proposal_id']} agent {pr['author']} "
                         f"sig={pr['signatures']} passed={pr['passed']} "
                         f"\"{(pr.get('text') or '')[:24]}\"")
    if tools.get("ventures"):
        lines.append("- ventures (owner, name, sales, revenue):")
        for v in sorted(tools["ventures"],
                        key=lambda x: -x.get("revenue", 0))[:5]:
            lines.append(f"  - agent {v['owner']} \"{v.get('name')}\" "
                         f"sales={v['sales']} rev={v['revenue']:.0f} "
                         f"closed={v['closed']}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def analyze(run_dir, y_weights=None):
    outdir = os.path.join(run_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    # P4: 観測ストリーミング。全件を RAM に載せず row-group 逐次で同一結果を得る。
    agents = m.load_agents(run_dir)
    l2 = m.load_l2(run_dir)
    traits, tnames, tsrc = m.load_traits(run_dir, agents)
    n_agents = len(agents) or (st.max_agent_id(run_dir) + 1)

    feats = st.agent_features(run_dir, agents, traits, y_weights=y_weights)
    cascades = st.item_cascades(run_dir)
    windows = st.network_windows(run_dir)
    collective = st.collective_series(run_dir, l2, n_agents)
    # D1: y_weights 指定時のみ Y_composite の R² も出す(既定は従来と同一)
    extra = ("Y_composite",) if (feats and "Y_composite" in feats[0]) else ()
    r2 = m.r2_traits(feats, tnames, extra_targets=extra)
    ews_adopt = m.ews(collective["adoption_frac"])
    ews_entropy = m.ews(collective["vocab_entropy"])
    coinages = st.coinage_contexts(run_dir)
    trigger_dist = coinage_trigger_dist(coinages)
    tools = st.tool_usage(run_dir)
    drift = st.drift_metrics(run_dir, n_agents)
    # エコー/自己反復(第70バッチ IDEA①)。既存の伝播/採用 KPI は 1 バイトも変えず、
    # 「エコー除外後」の件数・率を **新しい成果物** として並記する(ID-U3)。
    echo = st.echo_novelty(run_dir)

    _dump(os.path.join(outdir, "agent_features.json"), feats)
    _dump(os.path.join(outdir, "items.json"), cascades)
    _dump(os.path.join(outdir, "tools.json"), tools)
    _dump(os.path.join(outdir, "coinage.json"),
          {"n_coinages": len(coinages), "trigger_dist": trigger_dist,
           "contexts": coinages})
    _dump(os.path.join(outdir, "network_windows.json"), windows)
    _dump(os.path.join(outdir, "collective.json"), collective)
    _dump(os.path.join(outdir, "drift.json"), drift)
    _dump(os.path.join(outdir, "echo.json"), echo)
    _dump(os.path.join(outdir, "ews.json"),
          {"adoption_frac": {k: ews_adopt[k] for k in ("var_tau", "ac1_tau")},
           "vocab_entropy": {k: ews_entropy[k] for k in ("var_tau", "ac1_tau")}})
    _dump(os.path.join(outdir, "r2.json"), r2)

    steps = collective["step"]
    # 図も逐次読み(1枚につき使い捨ての生成器を渡す=全件 materialize しない)
    fig_adoption(m.stream_events(run_dir), cascades, steps,
                 os.path.join(outdir, "fig_adoption.png"))
    fig_cascade(m.stream_events(run_dir), cascades,
                os.path.join(outdir, "fig_cascade.png"))
    fig_network(windows, os.path.join(outdir, "fig_network.png"))
    fig_rhythm(collective, os.path.join(outdir, "fig_rhythm.png"))
    fig_drive(m.stream_events(run_dir), os.path.join(outdir, "fig_drive.png"))

    write_report(os.path.join(outdir, "report.md"), run_dir, feats, cascades,
                 windows, collective, r2, ews_adopt, ews_entropy, tsrc, tnames,
                 coinages, trigger_dist, tools, drift, echo)
    return outdir, m.count_events(run_dir), len(cascades)


def main():
    ap = argparse.ArgumentParser(description="Phase E single-run analysis")
    ap.add_argument("run_dir", help="e.g. runs/demo5")
    ap.add_argument("--y-weights", default=None,
                    help='D1 4層Y統合の重み JSON。例: '
                         '\'{"venture_revenue":0.001,"groups_founded":1.0}\'。'
                         '省略時は run の config.yaml の observer.y_weights を読む。')
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    y_weights = _load_y_weights(args.run_dir, args.y_weights)
    outdir, n_events, n_items = analyze(args.run_dir, y_weights=y_weights)
    print(f"[analyze] {args.run_dir}: {n_events} events, {n_items} items "
          f"-> {outdir}")


if __name__ == "__main__":
    main()
