#!/usr/bin/env python
"""お金の移動 + 注意ネットワークの観測 CLI(読み出し専用: L1 ログを読むだけ)。

    python scripts/observe_flows.py runs/<name> [--out runs/<name>/observe_flows]
                                                 [--window-days N]

ユーザー要望(2026-07-08):エージェントの「**お金の移動**」と「**誰に注意を向けて
いるのか**」を観測できたら面白い。scripts/observe.py の流儀を踏襲する:
  - src/society/observer/measure.py の load_events / load_agents を再利用
    (schema.py・既存ファイルは一切変更しない)。
  - 該当イベントが無い項目は "0件" と正直に明記し、値の捏造はしない。
  - payload のフィールド名はランの実データで確認済み(推測で書かない。試験台:
    runs/calib_final_100d)。

出力(--out 既定 = runs/<name>/observe_flows):
  money_flows.json      … ノード(agent/セクター)とエッジ(総額・件数)、日次金流、
                          内部移動(withdraw)別掲、供託金(deposit)の正直な注記
  money_flows_links.csv … Sankey 用の links(source,target,value,count,...)
  flows_summary.md      … 誰が金を集めたか上位・セクター収支・日次金流の推移
  attention.json        … エージェント別(誰に最も注意を向けたか top5 / 誰から集めたか)
  attention_summary.md  … 注意の集中度 gini・被注意 top10・相互注意ペア

--- L1 の金額イベント payload(実データで確認)---
  wage   {amount, gross, tax, balance, account, source, to}   source→agent(収入)
         source: salary(会社給与)/civil(公務員給与)/gig(自営収入)/home_refill(財布補充)/
                 severance(退職金)/None。to は account|cash(エージェント内の入金先)。
  spend  {amount, balance, account, cat, src}                 agent→cat 別シンク(消費)
         cat: food/shop/nightlife/taxi/lodging/medical/fixed_cost。src は cash|card(支払手段)。
         ★売り手フィールドは payload に無い=agent→agent エッジは spend では作れない。
  venture_sale {amount, balance, buyer}                       buyer→seller(agent→agent)
  rent   {amount, paid, carry, account, phase}                agent→大家(paid=実際に動いた額)
  tax    {tax, amount, base, to}                              agent→行政(to: nation|ward|metro)
  civic_service {service, level, amount, detail}              行政(level)→agent(給付。amount 有時)
  reward {amount, balance}                                    報酬→agent
  rule_bonus {rule_id, behavior, amount, balance}             制度DSL 支給→agent
  withdraw {amount, cash, account}                            agent 内部: 口座→現金(別掲)
  deposit  {amount, phase, balance}                           供託金(政治供託。※口座入金ではない)

--- 注意イベント payload(実データで確認)---
  hear         {speaker, items}         聞き手(agent_id)→話し手(speaker)
  dm           {to, text}               送り手(agent_id)→宛先(to)
  sns_read     {authors, n_posts}       読者(agent_id)→各 author
  sns_like     {author, post_id}        読者→author
  sns_reshare  {author, post_id}        読者→author
  event_attend {event_id, host, title}  参加者(agent_id)→主催(host)
  flyer_view   {author, node}           閲覧者(agent_id)→掲示者(author)
  ※ author/host/speaker が -1 のもの(メディア/公式/ニュース)は「メディア/公式」ノードへ集約。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

try:                                # 文字化け対策(日本語 payload の標準出力)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                   # 一部環境(パイプ等)では reconfigure 不可=無害に無視
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                # l1_stream(W2-2 の共有逐次読み)用
    sys.path.insert(0, _HERE)

from society.observer import measure as m  # noqa: E402

MIN_PER_DAY = 1440

# wage の source(=支払いセクター)→ 人が読めるセクター名。実データで確認した値のみ写像し、
# 未知の source は "組織/その他(<値>)" として素通し(捏造せず raw も残す)。
_WAGE_SRC_LABEL = {
    "salary": "組織(会社給与)",
    "civil": "行政(公務員給与)",
    "gig": "市場(自営収入)",
    "home_refill": "自宅(財布補充)",
    "severance": "組織(退職金)",
    None: "組織(賃金)",
    "": "組織(賃金)",
}
# 消費 cat(spend)→ シンク表示名。未知 cat も素通し。
_CAT_LABEL = {
    "food": "飲食", "shop": "買い物", "nightlife": "夜遊び", "taxi": "タクシー",
    "bus": "バス", "lodging": "宿泊", "medical": "医療", "fixed_cost": "固定費",
}
_GOV_LABEL = {"nation": "国", "ward": "区", "metro": "都"}


def _day(sim_min: int) -> int:
    return sim_min // MIN_PER_DAY


# W4-D: 本観測が読む kind(2 つの観測器の if/elif を全数列挙したもの)。
#   observe_money     = wage / spend / venture_sale / rent / tax / civic_service /
#                       reward / rule_bonus / withdraw / deposit
#   observe_attention = hear / dm / sns_read / sns_like / sns_reshare /
#                       event_attend / flyer_view
# どちらの `seen` Counter も **全 kind を数えるが読み出しは自分の種だけ**
# (money: by_event_kind は kind_amt のキー = 上の 10 種 / 明示 10 行も同じ。
#  attention: seen[k] は if/elif の中でしか加算されない)なので、絞っても不変。
# 全 kind 依存は「注意グラフの末尾窓を決める max_day」と「summary の総件数」の 2 つ。
WANT_KINDS = frozenset({
    "wage", "spend", "venture_sale", "rent", "tax", "civic_service",
    "reward", "rule_bonus", "withdraw", "deposit",
    "hear", "dm", "sns_read", "sns_like", "sns_reshare",
    "event_attend", "flyer_view",
})


def load_events(run_dir: str) -> list[dict]:
    """L1 を `WANT_KINDS` だけ読み、`measure.load_events` と同一形の dict 列で返す。

    残る比例項: 金流イベントと注意イベントの行数(どちらも全期間の集計)。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):        # m.load_events と同じく「無ければ落とす」
        raise FileNotFoundError(os.path.join(run_dir, "l1_events.parquet"))
    return list(ls.iter_events(run_dir, kinds=WANT_KINDS))


# --------------------------------------------------------------------------- #
# ノード/エッジの器
# --------------------------------------------------------------------------- #
class FlowGraph:
    """有向・重み付きの金流グラフ。ノード種別(agent/sector)と収支を保持する。"""

    def __init__(self) -> None:
        self.node_kind: dict[str, str] = {}
        self.node_label: dict[str, str] = {}
        self.inflow: dict[str, float] = defaultdict(float)
        self.outflow: dict[str, float] = defaultdict(float)
        self.in_count: dict[str, int] = defaultdict(int)
        self.out_count: dict[str, int] = defaultdict(int)
        self.edge_amt: dict[tuple, float] = defaultdict(float)
        self.edge_cnt: dict[tuple, int] = defaultdict(int)

    def node(self, nid: str, kind: str, label: str) -> str:
        self.node_kind.setdefault(nid, kind)
        self.node_label.setdefault(nid, label)
        return nid

    def add(self, src: str, dst: str, amount: float) -> None:
        amount = float(amount or 0.0)
        self.edge_amt[(src, dst)] += amount
        self.edge_cnt[(src, dst)] += 1
        self.outflow[src] += amount
        self.out_count[src] += 1
        self.inflow[dst] += amount
        self.in_count[dst] += 1


def _agent_node(g: FlowGraph, aid: int, name_of: dict) -> str:
    nid = f"agent:{aid}"
    return g.node(nid, "agent", name_of.get(aid) or f"agent{aid}")


# --------------------------------------------------------------------------- #
# 金流観測
# --------------------------------------------------------------------------- #
def observe_money(events: list[dict], name_of: dict) -> dict:
    g = FlowGraph()
    seen: Counter = Counter()
    kind_amt: dict[str, float] = defaultdict(float)   # イベント種別ごとの総額(監査)
    daily: dict[int, float] = defaultdict(float)      # 日次の外部金流(内部移動は除く)

    # 内部移動(withdraw: 口座→現金)は主グラフに入れず別掲する
    internal = {"withdraw": {"count": 0, "amount": 0.0, "by_agent": Counter()}}
    # 供託金(deposit)は「口座入金」ではなく政治供託。phase 別に正直に集計する
    deposit = {"count": 0, "by_phase": Counter(), "amount_moved": 0.0}

    for e in sorted(events, key=lambda ev: ev["step"]):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        seen[k] += 1
        d = _day(e["sim_min"])

        if k == "wage":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            src = p.get("source")
            label = _WAGE_SRC_LABEL.get(src, f"組織/その他({src})")
            snid = g.node(f"src:wage:{src}", "sector", label)
            g.add(snid, _agent_node(g, aid, name_of), amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "spend":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            cat = p.get("cat") or "unknown"
            snid = g.node(f"sink:cat:{cat}", "sector", _CAT_LABEL.get(cat, cat))
            g.add(_agent_node(g, aid, name_of), snid, amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "venture_sale":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            buyer = p.get("buyer")
            seller = _agent_node(g, aid, name_of)
            if isinstance(buyer, int) and buyer >= 0:
                g.add(_agent_node(g, buyer, name_of), seller, amt)  # 買い手→売り手
            else:                       # 買い手不明(通行人集計等)は市場ノードから
                src = g.node("src:venture_buyers", "sector", "屋台の客(不明)")
                g.add(src, seller, amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "rent":
            if aid is None or aid < 0:
                continue
            paid = float(p.get("paid", 0) or 0)   # 実際に動いた額(carry=繰越は未払い)
            if paid <= 0:
                continue
            snid = g.node("sink:rent", "sector", "大家(家賃)")
            g.add(_agent_node(g, aid, name_of), snid, paid)
            kind_amt[k] += paid
            daily[d] += paid

        elif k == "tax":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            to = p.get("to") or "gov"
            snid = g.node(f"gov:{to}", "sector", f"行政:{_GOV_LABEL.get(to, to)}")
            g.add(_agent_node(g, aid, name_of), snid, amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "civic_service":
            # 給付(welfare 等)= 行政→agent。amount がある場合のみ金流とみなす
            amt = float(p.get("amount", 0) or 0)
            if aid is None or aid < 0 or amt <= 0:
                continue
            lvl = p.get("level") or "gov"
            snid = g.node(f"gov:{lvl}", "sector", f"行政:{_GOV_LABEL.get(lvl, lvl)}")
            g.add(snid, _agent_node(g, aid, name_of), amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "reward":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            snid = g.node("src:reward", "sector", "採用報酬")
            g.add(snid, _agent_node(g, aid, name_of), amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "rule_bonus":
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            snid = g.node("src:rule_bonus", "sector", "制度DSL 支給")
            g.add(snid, _agent_node(g, aid, name_of), amt)
            kind_amt[k] += amt
            daily[d] += amt

        elif k == "withdraw":           # 内部移動(口座→現金)= 別掲。主グラフに入れない
            if aid is None or aid < 0:
                continue
            amt = float(p.get("amount", 0) or 0)
            internal["withdraw"]["count"] += 1
            internal["withdraw"]["amount"] += amt
            internal["withdraw"]["by_agent"][aid] += amt

        elif k == "deposit":            # 供託金(政治供託)= 口座入金ではない。正直に別掲
            deposit["count"] += 1
            phase = str(p.get("phase") or "unknown")
            deposit["by_phase"][phase] += 1
            # 供託金が実際に動くのは受理拠出/没収/返還のとき(拠出不能=insufficient は不動)
            if phase not in ("insufficient", "unknown"):
                deposit["amount_moved"] += float(p.get("amount", 0) or 0)

    # ---- ノード表 ----
    nodes = []
    for nid in sorted(g.node_kind, key=lambda n: (g.node_kind[n], n)):
        inf = round(g.inflow.get(nid, 0.0), 1)
        outf = round(g.outflow.get(nid, 0.0), 1)
        nodes.append({
            "id": nid,
            "kind": g.node_kind[nid],
            "label": g.node_label[nid],
            "inflow": inf,
            "outflow": outf,
            "net": round(inf - outf, 1),
            "in_count": g.in_count.get(nid, 0),
            "out_count": g.out_count.get(nid, 0),
        })

    # ---- エッジ表(総額降順)----
    edges = [{"source": s, "target": t,
              "source_label": g.node_label.get(s, s),
              "target_label": g.node_label.get(t, t),
              "amount": round(a, 1), "count": g.edge_cnt[(s, t)]}
             for (s, t), a in g.edge_amt.items()]
    edges.sort(key=lambda r: -r["amount"])

    # ---- 誰が金を集めたか(agent の inflow/net ランキング)----
    agent_nodes = [n for n in nodes if n["kind"] == "agent"]
    top_collectors = sorted(agent_nodes, key=lambda n: -n["inflow"])[:20]
    top_net = sorted(agent_nodes, key=lambda n: -n["net"])[:20]

    # ---- セクター収支 ----
    sectors = [n for n in nodes if n["kind"] == "sector"]
    sectors.sort(key=lambda n: -(n["inflow"] + n["outflow"]))

    total_external = round(sum(kind_amt.values()), 1)
    return {
        "totals": {
            "n_nodes": len(nodes),
            "n_agent_nodes": len(agent_nodes),
            "n_sector_nodes": len(sectors),
            "n_edges": len(edges),
            "total_external_amount": total_external,
        },
        "by_event_kind": {k: {"count": seen.get(k, 0), "amount": round(v, 1)}
                          for k, v in sorted(kind_amt.items(),
                                             key=lambda kv: -kv[1])},
        "nodes": nodes,
        "edges": edges,
        "top_money_collectors": [{"agent_id": _aid(n["id"]), "label": n["label"],
                                  "inflow": n["inflow"], "net": n["net"]}
                                 for n in top_collectors],
        "top_net_gainers": [{"agent_id": _aid(n["id"]), "label": n["label"],
                             "net": n["net"], "inflow": n["inflow"],
                             "outflow": n["outflow"]} for n in top_net],
        "sector_balance": [{"id": n["id"], "label": n["label"],
                            "inflow": n["inflow"], "outflow": n["outflow"],
                            "net": n["net"]} for n in sectors],
        "daily_flow": {str(d): round(v, 1) for d, v in sorted(daily.items())},
        "internal_moves": {
            "withdraw": {
                "note": "口座→現金の自動引き出し(ATM近似)。エージェント内部の移動="
                        "外部金流には含めない。",
                "count": internal["withdraw"]["count"],
                "amount": round(internal["withdraw"]["amount"], 1),
                "top_agents": [{"agent_id": a, "amount": round(v, 1)} for a, v in
                               internal["withdraw"]["by_agent"].most_common(10)],
            }
        },
        "deposit_political": {
            "note": "L1 の 'deposit' イベントは口座入金ではなく政治供託(供託金)。"
                    "phase=insufficient(拠出不能)は金銭が動かない。実際に動いた額のみ集計。",
            "count": deposit["count"],
            "by_phase": dict(deposit["by_phase"].most_common()),
            "amount_moved": round(deposit["amount_moved"], 1),
        },
        "event_counts": {
            "wage": seen.get("wage", 0),
            "spend": seen.get("spend", 0),
            "rent": seen.get("rent", 0),
            "tax": seen.get("tax", 0),
            "civic_service": seen.get("civic_service", 0),
            "venture_sale": seen.get("venture_sale", 0),
            "reward": seen.get("reward", 0),
            "rule_bonus": seen.get("rule_bonus", 0),
            "withdraw": seen.get("withdraw", 0),
            "deposit": seen.get("deposit", 0),
        },
    }


def _aid(node_id: str):
    """'agent:<n>' から int の agent_id を取り出す(それ以外は None)。"""
    if node_id.startswith("agent:"):
        try:
            return int(node_id.split(":", 1)[1])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# 注意ネットワーク観測
# --------------------------------------------------------------------------- #
_MEDIA_ID = -1
_MEDIA_LABEL = "メディア/公式"


def observe_attention(events: list[dict], name_of: dict,
                      window_days: int | None = None,
                      max_day: int | None = None) -> dict:
    """有向グラフ(注意を向ける人→向けられる人)。重み=回数。

    window_days>0 なら「ラン末尾の N 日」だけに限定する(直近の注意を見る)。

    W4-D: `max_day`(末尾窓の右端)は **全 kind** の最終日で決まる。events を kind
    絞り読みにすると窓がずれるので、呼び出し側が `l1_stream.max_day` の値を渡す。
    None なら従来どおり events から決める(= 全件 list を渡す既存経路と逐語同一)。
    """
    ordered = sorted(events, key=lambda ev: ev["step"])
    if max_day is None:
        max_day = max((_day(e["sim_min"]) for e in ordered), default=0)
    lo_day = (max_day - window_days + 1) if window_days else None

    # (attender, target) -> 回数 / チャネル内訳
    edge_w: dict[tuple, int] = defaultdict(int)
    edge_ch: dict[tuple, Counter] = defaultdict(Counter)
    seen: Counter = Counter()

    def _label(aid: int) -> str:
        if aid == _MEDIA_ID:
            return _MEDIA_LABEL
        return name_of.get(aid) or f"agent{aid}"

    def link(attender, target, channel):
        if not isinstance(attender, int) or attender < 0:
            return                       # 注意の主体は実在エージェントのみ
        if not isinstance(target, int):
            return
        if target == attender:
            return                       # 自己ループは除外
        edge_w[(attender, target)] += 1
        edge_ch[(attender, target)][channel] += 1

    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if lo_day is not None and _day(e["sim_min"]) < lo_day:
            continue
        if k == "hear":
            seen[k] += 1
            link(aid, p.get("speaker"), "hear")           # 聞き手→話し手
        elif k == "dm":
            seen[k] += 1
            link(aid, p.get("to"), "dm")                  # 送り手→宛先
        elif k == "sns_read":
            seen[k] += 1
            for au in p.get("authors", []) or []:
                link(aid, au, "sns_read")                 # 読者→投稿者
        elif k == "sns_like":
            seen[k] += 1
            link(aid, p.get("author"), "sns_like")
        elif k == "sns_reshare":
            seen[k] += 1
            link(aid, p.get("author"), "sns_reshare")
        elif k == "event_attend":
            seen[k] += 1
            link(aid, p.get("host"), "event_attend")      # 参加者→主催
        elif k == "flyer_view":
            seen[k] += 1
            link(aid, p.get("author"), "flyer_view")      # 閲覧者→掲示者

    # ---- 集計 ----
    out_w: dict[int, int] = defaultdict(int)     # 向けた注意の総量
    in_w: dict[int, int] = defaultdict(int)      # 集めた注意の総量
    attends_to: dict[int, Counter] = defaultdict(Counter)
    attended_by: dict[int, Counter] = defaultdict(Counter)
    for (a, t), w in edge_w.items():
        out_w[a] += w
        in_w[t] += w
        attends_to[a][t] += w
        attended_by[t][a] += w

    # 相互注意ペア(A→B と B→A の双方が存在。重みは小さい方=相互に成立した回数)
    mutual = []
    done: set = set()
    for (a, t), w in edge_w.items():
        if a >= t:
            continue
        rev = edge_w.get((t, a))
        if rev and (a, t) not in done:
            done.add((a, t))
            mutual.append({"a": a, "a_label": _label(a),
                           "b": t, "b_label": _label(t),
                           "a_to_b": w, "b_to_a": rev,
                           "mutual": min(w, rev)})
    mutual.sort(key=lambda r: -r["mutual"])

    # per-agent ビュー(実在エージェントのみ。メディアは被注意側で集計に出る)
    agent_ids = sorted({a for a, _ in edge_w} | {t for _, t in edge_w if t >= 0})
    per_agent = []
    for aid in agent_ids:
        top_to = [{"target": t, "label": _label(t), "weight": w}
                  for t, w in attends_to.get(aid, Counter()).most_common(5)]
        top_by = [{"source": s, "label": _label(s), "weight": w}
                  for s, w in attended_by.get(aid, Counter()).most_common(5)]
        per_agent.append({
            "agent_id": aid,
            "name": name_of.get(aid),
            "attention_out": out_w.get(aid, 0),
            "attention_in": in_w.get(aid, 0),
            "n_distinct_attended_to": len(attends_to.get(aid, ())),
            "n_distinct_attended_by": len(attended_by.get(aid, ())),
            "most_attended_to": top_to,
            "most_attended_by": top_by,
        })

    # 被注意の集中度(gini)と top10。メディア(-1)も被注意ランキングには載せる
    in_pairs = sorted(in_w.items(), key=lambda kv: -kv[1])
    gini_vals = [w for aid, w in in_w.items() if aid >= 0]   # gini は実在 agent のみ
    edges = [{"attender": a, "attender_label": _label(a),
              "target": t, "target_label": _label(t),
              "weight": w, "channels": dict(edge_ch[(a, t)])}
             for (a, t), w in edge_w.items()]
    edges.sort(key=lambda r: -r["weight"])

    return {
        "window_days": window_days,
        "day_range": {"max_day": max_day,
                      "included_from_day": lo_day if lo_day is not None else 0},
        "totals": {
            "n_edges": len(edge_w),
            "n_attenders": len(out_w),
            "n_targets": len(in_w),
            "total_attention": int(sum(edge_w.values())),
            "attention_in_gini": round(_gini(gini_vals), 4),
        },
        "channel_counts": dict(seen.most_common()),
        "most_attended": [{"agent_id": aid, "label": _label(aid), "attention_in": w}
                          for aid, w in in_pairs[:10]],
        "mutual_pairs": mutual[:20],
        "per_agent": per_agent,
        "edges": edges,
        "event_counts": {k: seen.get(k, 0) for k in
                         ("hear", "dm", "sns_read", "sns_like", "sns_reshare",
                          "event_attend", "flyer_view")},
    }


def _gini(values: list) -> float:
    """gini 係数(0=完全平等 .. 1=完全集中)。空/総和0 は 0.0。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def write_links_csv(money: dict, path: str) -> None:
    """Sankey 用 links CSV(source,target,value,count + ラベル)。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "target", "source_label",
                                           "target_label", "value", "count"])
        w.writeheader()
        for e in money["edges"]:
            w.writerow({"source": e["source"], "target": e["target"],
                        "source_label": e["source_label"],
                        "target_label": e["target_label"],
                        "value": e["amount"], "count": e["count"]})


def _fmt_yen(v) -> str:
    try:
        return f"{int(round(float(v))):,}円"
    except (TypeError, ValueError):
        return str(v)


def write_money_md(money: dict, run_name: str, n_events: int, path: str) -> None:
    t = money["totals"]
    L: list[str] = []
    L.append(f"# お金の移動 観測サマリ: {run_name}\n")
    L.append("L1 ログの金額イベントから「誰が誰に・どのセクターへお金を動かしたか」を集計した。")
    L.append("値はすべてログ由来の実測。該当イベントが無い項目は「0件」と明記(捏造なし)。\n")

    L.append("## 規模")
    L.append(f"- 総イベント数: {n_events}")
    L.append(f"- ノード数: {t['n_nodes']}(agent {t['n_agent_nodes']} / "
             f"セクター {t['n_sector_nodes']})")
    L.append(f"- エッジ数: {t['n_edges']}")
    L.append(f"- 外部金流の総額: {_fmt_yen(t['total_external_amount'])}\n")

    L.append("## イベント種別ごとの金流(総額)")
    if money["by_event_kind"]:
        L.append("| 種別 | 件数 | 総額 |")
        L.append("|---|---|---|")
        for k, v in money["by_event_kind"].items():
            L.append(f"| {k} | {v['count']} | {_fmt_yen(v['amount'])} |")
    else:
        L.append("(金額イベント 0件)")
    L.append("")

    L.append("## 誰が金を集めたか(agent 別・流入額 top10)")
    if money["top_money_collectors"]:
        L.append("| 順 | agent | 名前 | 流入 | 収支(net) |")
        L.append("|---|---|---|---|---|")
        for i, a in enumerate(money["top_money_collectors"][:10], 1):
            L.append(f"| {i} | {a['agent_id']} | {a['label']} | "
                     f"{_fmt_yen(a['inflow'])} | {_fmt_yen(a['net'])} |")
    else:
        L.append("(収入イベント 0件)")
    L.append("")

    L.append("## 収支(net)で得をした人 top10")
    if money["top_net_gainers"]:
        L.append("| 順 | agent | 名前 | net | 流入 | 流出 |")
        L.append("|---|---|---|---|---|---|")
        for i, a in enumerate(money["top_net_gainers"][:10], 1):
            L.append(f"| {i} | {a['agent_id']} | {a['label']} | {_fmt_yen(a['net'])} "
                     f"| {_fmt_yen(a['inflow'])} | {_fmt_yen(a['outflow'])} |")
    else:
        L.append("(金流 0件)")
    L.append("")

    L.append("## セクター収支(組織・行政・消費先・大家 等)")
    if money["sector_balance"]:
        L.append("| セクター | 流入 | 流出 | net |")
        L.append("|---|---|---|---|")
        for s in money["sector_balance"]:
            L.append(f"| {s['label']} | {_fmt_yen(s['inflow'])} | "
                     f"{_fmt_yen(s['outflow'])} | {_fmt_yen(s['net'])} |")
    else:
        L.append("(セクター 0件)")
    L.append("")

    L.append("## 日次金流の推移(外部金流の総額 / 日)")
    df = money["daily_flow"]
    if df:
        L.append("- " + "、".join(f"Day{d}: {_fmt_yen(v)}" for d, v in df.items()))
    else:
        L.append("(金流 0件)")
    L.append("")

    im = money["internal_moves"]["withdraw"]
    L.append("## 内部移動(別掲: 外部金流に含めない)")
    L.append(f"- withdraw(口座→現金): {im['count']}件 / {_fmt_yen(im['amount'])}")
    dep = money["deposit_political"]
    L.append(f"- deposit(供託金・政治供託): {dep['count']}件 / "
             f"実際に動いた額 {_fmt_yen(dep['amount_moved'])} / "
             f"phase内訳 {dep['by_phase'] or '（なし）'}")
    L.append(f"  - 注: {dep['note']}")
    L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_attention_md(att: dict, run_name: str, path: str) -> None:
    t = att["totals"]
    L: list[str] = []
    L.append(f"# 注意ネットワーク 観測サマリ: {run_name}\n")
    L.append("「誰が誰に注意を向けたか」を有向グラフ化(重み=回数)。聞く・DM・SNS 閲覧/"
             "いいね/リシェア・イベント参加・ビラ閲覧を注意の矢印として合算した。\n")
    if att["window_days"]:
        L.append(f"※ 時間窓: ラン末尾の {att['window_days']} 日"
                 f"(Day{att['day_range']['included_from_day']}〜"
                 f"{att['day_range']['max_day']})に限定。\n")

    L.append("## 規模")
    L.append(f"- 注意エッジ数: {t['n_edges']}")
    L.append(f"- 注意を向けた人数: {t['n_attenders']} / 向けられた対象数: {t['n_targets']}")
    L.append(f"- 総注意量(回数): {t['total_attention']}")
    L.append(f"- 被注意の集中度(gini・実在agentのみ): {t['attention_in_gini']}\n")

    L.append("## チャネル別の注意量")
    if att["channel_counts"]:
        L.append("- " + "、".join(f"{k}: {v}" for k, v in
                                  att["channel_counts"].items()))
    else:
        L.append("(注意イベント 0件)")
    L.append("")

    L.append("## 最も注意を集めた対象 top10")
    if att["most_attended"]:
        L.append("| 順 | id | 名前 | 被注意(回数) |")
        L.append("|---|---|---|---|")
        for i, a in enumerate(att["most_attended"], 1):
            L.append(f"| {i} | {a['agent_id']} | {a['label']} | {a['attention_in']} |")
    else:
        L.append("(注意イベント 0件)")
    L.append("")

    L.append("## 相互注意ペア(双方向に注意を向け合った関係)top10")
    if att["mutual_pairs"]:
        L.append("| A | B | A→B | B→A | 相互(min) |")
        L.append("|---|---|---|---|---|")
        for pr in att["mutual_pairs"][:10]:
            L.append(f"| {pr['a_label']} | {pr['b_label']} | {pr['a_to_b']} | "
                     f"{pr['b_to_a']} | {pr['mutual']} |")
    else:
        L.append("(相互の注意ペア 0件)")
    L.append("")

    L.append("## 観測項目ごとのイベント件数(0件=このランに該当なし)")
    for k, cnt in att["event_counts"].items():
        mark = "" if cnt else "  ← 0件(このランでは観測不可)"
        L.append(f"- {k}: {cnt}件{mark}")
    L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
def _dump(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def observe(run_dir: str, out_dir: str | None = None,
            window_days: int | None = None) -> str:
    out_dir = out_dir or os.path.join(run_dir, "observe_flows")
    os.makedirs(out_dir, exist_ok=True)

    import l1_stream as ls
    events = load_events(run_dir)
    agents = m.load_agents(run_dir)
    name_of = {int(a["id"]): a.get("name") for a in (agents or []) if "id" in a}

    # 全 kind 依存の 2 量。max_day は `_day` と同じ規則(sim_min のみ・step 後退なし)。
    n_events = ls.row_count(run_dir)
    max_day = max(ls.max_day(run_dir, 1, min_per_day=MIN_PER_DAY,
                             step_fallback=False), 0)  # 旧: default=0

    money = observe_money(events, name_of)
    attention = observe_attention(events, name_of, window_days=window_days,
                                  max_day=max_day)

    run_name = os.path.basename(os.path.normpath(run_dir))
    _dump(os.path.join(out_dir, "money_flows.json"), money)
    _dump(os.path.join(out_dir, "attention.json"), attention)
    write_links_csv(money, os.path.join(out_dir, "money_flows_links.csv"))
    write_money_md(money, run_name, n_events,
                   os.path.join(out_dir, "flows_summary.md"))
    write_attention_md(attention, run_name,
                       os.path.join(out_dir, "attention_summary.md"))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="お金の移動 + 注意ネットワークの観測(読み出し専用)")
    ap.add_argument("run_dir", help="例: runs/calib_final_100d")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 runs/<name>/observe_flows)")
    ap.add_argument("--window-days", type=int, default=None,
                    help="注意グラフをラン末尾の N 日に限定(既定=全期間)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    out = observe(args.run_dir, args.out, window_days=args.window_days)
    print(f"[observe_flows] {args.run_dir} -> {out}")


if __name__ == "__main__":
    main()
