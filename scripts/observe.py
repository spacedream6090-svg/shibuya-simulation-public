#!/usr/bin/env python
"""Wave O: 興味・訪問の観測 CLI(sim⇄観測は疎結合: L1 ログを読むだけ)。

    python scripts/observe.py runs/<name> [--out runs/<name>/observe]

ユーザー要望(2026-07-07):エージェントの思考・行動だけでなく「**どのようなものに
興味を持ったのか / 実際にどこへ行ったのか**」を観測する。

出力(--out 既定 = runs/<name>/observe):
  visits.json      … POI/建物/ノード別の訪問・滞在・ユニーク訪問者、行動圏、カテゴリ集客
  interests.json   … エージェント別の関心プロファイル + 全体集計
  poi_ranking.csv  … 訪問先ランキング(建物/ノード横断・訪問数降順)
  summary.md       … 人が読める日本語サマリ(上位POI・関心の偏り・日次/時間帯トレンド)

方針: 既存の src/society/observer/measure.py の load_events / load_agents を再利用する
(schema.py・既存ファイルは一切変更しない)。ランに該当イベントが無い項目は "0件" と
明記し、値の捏造はしない。地図(POI/建物名)は config.yaml の world.map から解決するが、
見つからなくても raw な id で動作を続ける(壊れない)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                       # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

from society.observer import measure as m  # noqa: E402

# 1 step = 10 sim 分。sim_min は 07:00(=420)起点の絶対分(step0→420)。
import run_dt                                 # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: MIN_PER_STEP は **ラン依存**(run.dt_min)。ここの値は正準 Δt=10 の既定で、
# 実際の値は observe() が run dir から読んで observe_visits へ渡す(Δt=10 なら同値)。
MIN_PER_STEP = run_dt.CANON_DT_MIN          # 10
MIN_PER_DAY = 1440                          # 分/日は Δt 非依存(実時間。_day は sim_min 基準)

# 建物 kind → 大まかなカテゴリ(POI が無い建物のフォールバック)
_KIND_TO_CAT = {
    "retail": "shop", "office": "office", "hotel": "hotel", "station": "station",
    "residential": "residential", "house?": "residential", "public": "public",
    "generic": "その他", "school": "education",
}


# --------------------------------------------------------------------------- #
# 地図(POI/建物/ノード名)の解決
# --------------------------------------------------------------------------- #
def load_map(run_dir: str) -> dict:
    """config.yaml の world.map を読み、POI/建物/ノードの索引を返す。

    見つからない/読めない場合は空の索引を返す(観測は raw id で続行)。
    """
    empty = {"building_meta": {}, "poi_by_building": {}, "poi_by_node": {},
             "node_name": {}, "map_path": None, "n_pois": 0, "n_buildings": 0}
    cfg_path = os.path.join(run_dir, "config.yaml")
    map_rel = None
    if os.path.exists(cfg_path):
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
            map_rel = (cfg.get("world", {}) or {}).get("map")
        except Exception:
            map_rel = None
    if not map_rel:
        return empty

    # 解決順: 絶対 → repo root 相対 → run_dir 相対 → そのまま
    cands = []
    if os.path.isabs(map_rel):
        cands = [map_rel]
    else:
        cands = [os.path.join(_ROOT, map_rel), os.path.join(run_dir, map_rel), map_rel]
    map_path = next((p for p in cands if os.path.exists(p)), None)
    if map_path is None:
        return empty
    try:
        city = json.loads(open(map_path, encoding="utf-8").read())
    except Exception:
        return empty

    building_meta = {b["id"]: {"name": b.get("name", ""), "kind": b.get("kind", ""),
                               "levels": b.get("levels", 0)}
                     for b in city.get("buildings", [])}
    poi_by_building: dict[str, list] = defaultdict(list)
    poi_by_node: dict[str, list] = defaultdict(list)
    for p in city.get("pois", []):
        rec = {"name": p.get("name", ""), "cat": p.get("cat", "")}
        if p.get("building"):
            poi_by_building[p["building"]].append(rec)
        if p.get("node"):
            poi_by_node[p["node"]].append(rec)
    node_name = {n["id"]: n["name"] for n in city.get("nodes", []) if n.get("name")}
    return {
        "building_meta": building_meta,
        "poi_by_building": dict(poi_by_building),
        "poi_by_node": dict(poi_by_node),
        "node_name": node_name,
        "map_path": map_path,
        "n_pois": len(city.get("pois", [])),
        "n_buildings": len(city.get("buildings", [])),
    }


def _building_category(bid: str, mp: dict) -> str:
    pois = mp["poi_by_building"].get(bid)
    if pois:
        return Counter(p["cat"] for p in pois if p["cat"]).most_common(1)[0][0]
    kind = mp["building_meta"].get(bid, {}).get("kind", "")
    return _KIND_TO_CAT.get(kind, kind or "不明")


def _building_name(bid: str, mp: dict) -> str:
    meta = mp["building_meta"].get(bid, {})
    nm = meta.get("name")
    if nm:
        return nm
    kind = meta.get("kind") or "建物"
    return f"{kind}#{bid}"


def _node_category(nid: str, mp: dict) -> str:
    pois = mp["poi_by_node"].get(nid)
    if pois:
        return Counter(p["cat"] for p in pois if p["cat"]).most_common(1)[0][0]
    return "街路"


def _node_name(nid: str, mp: dict, payload_name: str = "") -> str:
    if nid in mp["node_name"]:
        return mp["node_name"][nid]
    pois = mp["poi_by_node"].get(nid)
    if pois:
        names = [p["name"] for p in pois if p["name"]]
        if names:
            return "・".join(names[:3])
    if payload_name and payload_name != "路上":
        return payload_name
    return nid


def _hour(sim_min: int) -> int:
    return (sim_min // 60) % 24


def _day(sim_min: int) -> int:
    return sim_min // MIN_PER_DAY


# --------------------------------------------------------------------------- #
# 訪問観測
# --------------------------------------------------------------------------- #
def observe_visits(events: list[dict], mp: dict,
                   mps: int = MIN_PER_STEP) -> dict:
    """POI/建物/ノード別の訪問・滞在・ユニーク訪問者、カテゴリ集客、行動圏。

    W2-3: `mps`(1 step の分数)は滞在 step → 分の換算だけに使う。既定は正準 10。"""
    # ---- 建物訪問(enter_building)+ 滞在時間(enter→exit ペアリング)----
    bld_visits: dict[str, int] = Counter()
    bld_visitors: dict[str, set] = defaultdict(set)
    bld_hour: dict[str, Counter] = defaultdict(Counter)
    bld_day: dict[str, Counter] = defaultdict(Counter)
    bld_stay_steps: dict[str, list] = defaultdict(list)   # 完了滞在のみ
    open_enter: dict[tuple, int] = {}                     # (agent, building) -> enter step
    n_open = 0

    # ---- ノード到着(arrive)----
    node_visits: dict[str, int] = Counter()
    node_visitors: dict[str, set] = defaultdict(set)
    node_hour: dict[str, Counter] = defaultdict(Counter)
    node_day: dict[str, Counter] = defaultdict(Counter)
    node_first: dict[str, int] = Counter()
    node_pay_name: dict[str, str] = {}

    # ---- 行動圏(agent 別)----
    agent_places: dict[int, Counter] = defaultdict(Counter)      # 場所ラベル -> 訪問回数
    agent_nodes: dict[int, set] = defaultdict(set)
    agent_buildings: dict[int, set] = defaultdict(set)

    # ---- カテゴリ集客(訪問イベントを解決カテゴリへ集約)----
    cat_visits: Counter = Counter()
    cat_visitors: dict[str, set] = defaultdict(set)

    ordered = sorted(events, key=lambda e: e["step"])
    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if aid is None or aid < 0:
            continue
        sm = e["sim_min"]
        if k == "arrive":
            nid = p.get("node")
            if nid is None:
                continue
            node_visits[nid] += 1
            node_visitors[nid].add(aid)
            node_hour[nid][_hour(sm)] += 1
            node_day[nid][_day(sm)] += 1
            if p.get("first_visit"):
                node_first[nid] += 1
            node_pay_name.setdefault(nid, p.get("name", "") or "")
            agent_nodes[aid].add(nid)
            label = f"node:{_node_name(nid, mp, p.get('name',''))}"
            agent_places[aid][label] += 1
            cat = _node_category(nid, mp)
            cat_visits[cat] += 1
            cat_visitors[cat].add(aid)
        elif k == "enter_building":
            bid = p.get("building")
            if bid is None:
                continue
            bld_visits[bid] += 1
            bld_visitors[bid].add(aid)
            bld_hour[bid][_hour(sm)] += 1
            bld_day[bid][_day(sm)] += 1
            agent_buildings[aid].add(bid)
            agent_places[aid][f"bld:{_building_name(bid, mp)}"] += 1
            cat = _building_category(bid, mp)
            cat_visits[cat] += 1
            cat_visitors[cat].add(aid)
            key = (aid, bid)
            if key not in open_enter:               # 同建物の再入は最初の enter を保持
                open_enter[key] = e["step"]
        elif k == "exit_building":
            bid = p.get("building")
            key = (aid, bid)
            st = open_enter.pop(key, None)
            if st is not None:
                bld_stay_steps[bid].append(e["step"] - st)
    n_open = len(open_enter)   # 退出ログが無いまま run 終了した滞在(滞在統計から除外)

    # ---- 建物ランキング ----
    def _stay_stats(steps: list) -> dict:
        if not steps:
            return {"n_completed": 0, "mean_stay_min": None, "median_stay_min": None,
                    "max_stay_min": None}
        mins = sorted(s * int(mps) for s in steps)
        n = len(mins)
        med = mins[n // 2] if n % 2 else (mins[n // 2 - 1] + mins[n // 2]) / 2
        return {"n_completed": n,
                "mean_stay_min": round(sum(mins) / n, 1),
                "median_stay_min": round(med, 1),
                "max_stay_min": mins[-1]}

    building_ranking = []
    for bid, n in bld_visits.most_common():
        building_ranking.append({
            "building_id": bid,
            "name": _building_name(bid, mp),
            "kind": mp["building_meta"].get(bid, {}).get("kind", ""),
            "category": _building_category(bid, mp),
            "visits": n,
            "unique_visitors": len(bld_visitors[bid]),
            "stay": _stay_stats(bld_stay_steps.get(bid, [])),
            "by_hour": [bld_hour[bid].get(h, 0) for h in range(24)],
            "by_day": dict(sorted(bld_day[bid].items())),
        })

    node_ranking = []
    for nid, n in node_visits.most_common():
        node_ranking.append({
            "node_id": nid,
            "name": _node_name(nid, mp, node_pay_name.get(nid, "")),
            "category": _node_category(nid, mp),
            "arrivals": n,
            "unique_visitors": len(node_visitors[nid]),
            "first_visits": node_first.get(nid, 0),
            "by_hour": [node_hour[nid].get(h, 0) for h in range(24)],
            "by_day": dict(sorted(node_day[nid].items())),
        })

    # ---- カテゴリ集客ランキング ----
    category_ranking = [{"category": c, "visits": v,
                         "unique_visitors": len(cat_visitors[c])}
                        for c, v in cat_visits.most_common()]

    # ---- 行動圏(agent 別)----
    activity_space = []
    for aid in sorted(agent_places):
        places = agent_places[aid]
        top = places.most_common(5)
        activity_space.append({
            "agent_id": aid,
            "n_distinct_places": len(places),
            "n_distinct_nodes": len(agent_nodes[aid]),
            "n_buildings_entered": len(agent_buildings[aid]),
            "total_visits": sum(places.values()),
            "top_places": [{"place": pl, "visits": c} for pl, c in top],
            "most_frequent_place": top[0][0] if top else None,
        })

    # ---- 全体トレンド(日次・時間帯)----
    day_trend: Counter = Counter()
    hour_trend: Counter = Counter()
    for bid, dd in bld_day.items():
        for d, c in dd.items():
            day_trend[d] += c
    for nid, dd in node_day.items():
        for d, c in dd.items():
            day_trend[d] += c
    for bid, hh in bld_hour.items():
        for h, c in hh.items():
            hour_trend[h] += c
    for nid, hh in node_hour.items():
        for h, c in hh.items():
            hour_trend[h] += c

    all_stays = [s for steps in bld_stay_steps.values() for s in steps]
    return {
        "totals": {
            "n_building_visits": int(sum(bld_visits.values())),
            "n_distinct_buildings": len(bld_visits),
            "n_arrivals": int(sum(node_visits.values())),
            "n_distinct_nodes": len(node_visits),
            "n_open_stays_at_end": n_open,
            "map_resolved": mp["map_path"] is not None,
        },
        "stay_overall": _stay_stats(all_stays),
        "building_ranking": building_ranking,
        "node_ranking": node_ranking,
        "category_ranking": category_ranking,
        "activity_space": activity_space,
        "day_trend": dict(sorted(day_trend.items())),
        "hour_trend": [hour_trend.get(h, 0) for h in range(24)],
    }


# --------------------------------------------------------------------------- #
# 興味観測
# --------------------------------------------------------------------------- #
def observe_interests(events: list[dict], agents: list[dict]) -> dict:
    """agent 別の関心プロファイル + 全体集計。"""
    name_of = {int(a["id"]): a.get("name") for a in (agents or []) if "id" in a}

    drive_reason: dict[int, Counter] = defaultdict(Counter)      # 起動全体
    drive_reason_granted: dict[int, Counter] = defaultdict(Counter)
    affect_cause: dict[int, Counter] = defaultdict(Counter)
    adopted: dict[int, list] = defaultdict(list)                 # (item_id,text)
    vocab_use: dict[int, Counter] = defaultdict(Counter)
    searches: dict[int, list] = defaultdict(list)
    liked_author: dict[int, Counter] = defaultdict(Counter)
    reshared_author: dict[int, Counter] = defaultdict(Counter)
    spend_cat: dict[int, Counter] = defaultdict(Counter)
    spend_amt: dict[int, Counter] = defaultdict(Counter)
    emotion_hist: dict[int, list] = defaultdict(list)
    long_goals: dict[int, list] = defaultdict(list)
    media_titles: dict[int, list] = defaultdict(list)
    plan_what: dict[int, Counter] = defaultdict(Counter)
    plan_places: dict[int, list] = defaultdict(list)

    # 該当イベントの有無(0件を正直に報告する)
    seen_kinds: Counter = Counter()

    for e in sorted(events, key=lambda ev: ev["step"]):
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        seen_kinds[k] += 1
        if aid is None or aid < 0:
            continue
        if k == "drive_request":
            r = p.get("reason") or "unknown"
            drive_reason[aid][r] += 1
            if p.get("granted"):
                drive_reason_granted[aid][r] += 1
        elif k == "affect_update":
            affect_cause[aid][p.get("cause") or "unknown"] += 1
        elif k == "label_adopt":
            adopted[aid].append({"item_id": p.get("item_id"), "text": p.get("text")})
        elif k == "vocab_use":
            iid = p.get("item_id")
            if iid is not None:
                vocab_use[aid][iid] += 1
        elif k == "search":
            q = p.get("query")
            if q:
                searches[aid].append(q)
        elif k == "sns_like":
            au = p.get("author")
            if isinstance(au, int):
                liked_author[aid][au] += 1
        elif k == "sns_reshare":
            au = p.get("author")
            if isinstance(au, int):
                reshared_author[aid][au] += 1
        elif k == "spend":
            cat = p.get("cat") or "unknown"
            spend_cat[aid][cat] += 1
            spend_amt[aid][cat] += float(p.get("amount", 0) or 0)
        elif k == "emotion_label":
            lab = p.get("label")
            if lab:
                emotion_hist[aid].append({"step": e["step"], "label": lab})
        elif k == "long_goal":
            g = p.get("goal")
            if g:
                long_goals[aid].append({"step": e["step"], "goal": g})
        elif k == "media_use":
            t = p.get("title")
            if t:
                media_titles[aid].append(t)
        elif k == "day_plan":
            for item in p.get("plan", []) or []:
                if isinstance(item, dict):
                    plan_what[aid][item.get("what") or "?"] += 1
                    if item.get("place"):
                        plan_places[aid].append(item["place"])

    ids = sorted(set(name_of) | set(drive_reason) | set(spend_cat) | set(adopted)
                 | set(searches) | set(liked_author) | set(plan_what))
    per_agent = []
    for aid in ids:
        per_agent.append({
            "agent_id": aid,
            "name": name_of.get(aid),
            "drive_reason_dist": dict(drive_reason[aid].most_common()),
            "drive_reason_granted_dist": dict(drive_reason_granted[aid].most_common()),
            "affect_cause_dist": dict(affect_cause[aid].most_common()),
            "adopted_words": adopted[aid],
            "vocab_use_dist": dict(vocab_use[aid].most_common()),
            "search_terms": searches[aid],
            "liked_authors": dict(liked_author[aid].most_common(10)),
            "reshared_authors": dict(reshared_author[aid].most_common(10)),
            "spend_category_dist": dict(spend_cat[aid].most_common()),
            "spend_amount_by_cat": {c: round(v, 1)
                                    for c, v in spend_amt[aid].most_common()},
            "emotion_label_history": emotion_hist[aid],
            "long_goals": long_goals[aid],
            "media_titles": media_titles[aid],
            "plan_activity_dist": dict(plan_what[aid].most_common()),
            "plan_place_mentions": plan_places[aid],
        })

    # ---- 全体集計 ----
    def _merge(d: dict) -> Counter:
        out: Counter = Counter()
        for c in d.values():
            out.update(c)
        return out

    all_adopted: Counter = Counter()
    for lst in adopted.values():
        for w in lst:
            if w.get("text"):
                all_adopted[w["text"]] += 1
    all_search: Counter = Counter()
    for lst in searches.values():
        all_search.update(lst)
    all_liked: Counter = Counter()
    for c in liked_author.values():
        all_liked.update(c)
    all_places: Counter = Counter()
    for lst in plan_places.values():
        all_places.update(lst)

    glob = {
        "drive_reason_dist": dict(_merge(drive_reason).most_common()),
        "drive_reason_granted_dist": dict(_merge(drive_reason_granted).most_common()),
        "affect_cause_dist": dict(_merge(affect_cause).most_common()),
        "spend_category_dist": dict(_merge(spend_cat).most_common()),
        "top_adopted_words": _merge_amount(all_adopted, 20),
        "top_search_terms": _merge_amount(all_search, 20),
        "top_liked_authors": {str(a): c for a, c in all_liked.most_common(15)},
        "plan_activity_dist": dict(_merge(plan_what).most_common()),
        "top_planned_places": _merge_amount(all_places, 20),
    }

    # 0件チェック(捏造禁止: 該当イベントが無い観測項目を明示)
    absent = {
        "affect_update(覚醒cause)": seen_kinds.get("affect_update", 0),
        "emotion_label(離散感情)": seen_kinds.get("emotion_label", 0),
        "long_goal(長期目標)": seen_kinds.get("long_goal", 0),
        "search(検索語)": seen_kinds.get("search", 0),
        "label_adopt(語の採用)": seen_kinds.get("label_adopt", 0),
        "spend(消費)": seen_kinds.get("spend", 0),
        "media_use(メディア)": seen_kinds.get("media_use", 0),
    }
    return {"per_agent": per_agent, "global": glob, "event_counts": absent}


def _merge_amount(counter: Counter, n: int) -> dict:
    return {str(k): c for k, c in counter.most_common(n)}


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def write_poi_csv(visits: dict, path: str) -> None:
    """建物・ノードを横断した訪問先ランキング CSV(訪問数降順)。"""
    rows = []
    for b in visits["building_ranking"]:
        rows.append({
            "place_type": "building", "place_id": b["building_id"],
            "name": b["name"], "category": b["category"],
            "visits": b["visits"], "unique_visitors": b["unique_visitors"],
            "mean_stay_min": b["stay"]["mean_stay_min"] or "",
        })
    for nd in visits["node_ranking"]:
        rows.append({
            "place_type": "node", "place_id": nd["node_id"],
            "name": nd["name"], "category": nd["category"],
            "visits": nd["arrivals"], "unique_visitors": nd["unique_visitors"],
            "mean_stay_min": "",
        })
    rows.sort(key=lambda r: (-r["visits"], r["place_id"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "place_type", "place_id", "name",
                                           "category", "visits", "unique_visitors",
                                           "mean_stay_min"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            r["rank"] = i
            w.writerow(r)


def _fmt_dist(d: dict, n: int = 8) -> str:
    if not d:
        return "(0件)"
    items = list(d.items())[:n]
    return "、".join(f"{k}: {v}" for k, v in items)


def write_summary_md(visits: dict, interests: dict, run_name: str,
                     n_events: int, n_agents: int, path: str) -> None:
    t = visits["totals"]
    L = []
    L.append(f"# 興味・訪問 観測サマリ: {run_name}\n")
    L.append("エージェントが「どこへ行き / 何に興味を持ったか」を L1 ログから観測した結果。")
    L.append("値はすべてログ由来の実測。該当イベントが無い項目は「0件」と明記する(捏造なし)。\n")

    L.append("## 規模")
    L.append(f"- エージェント数: {n_agents}")
    L.append(f"- 総イベント数: {n_events}")
    L.append(f"- 地図解決(POI/建物名): {'あり' if t['map_resolved'] else 'なし(raw id 表示)'}\n")

    # ---- 訪問 ----
    L.append("## 訪問観測")
    L.append(f"- 建物入館(enter_building): {t['n_building_visits']}件 / "
             f"{t['n_distinct_buildings']} 棟")
    L.append(f"- ノード到着(arrive): {t['n_arrivals']}件 / "
             f"{t['n_distinct_nodes']} 地点")
    so = visits["stay_overall"]
    if so["n_completed"]:
        L.append(f"- 滞在時間(建物・退出まで観測できた {so['n_completed']}件): "
                 f"平均 {so['mean_stay_min']}分 / 中央 {so['median_stay_min']}分 / "
                 f"最長 {so['max_stay_min']}分"
                 f"(未退出のまま終了: {t['n_open_stays_at_end']}件)")
    else:
        L.append("- 滞在時間: 退出まで観測できた滞在が 0件")
    L.append("")

    L.append("### 上位の訪問先(建物 top 10)")
    if visits["building_ranking"]:
        L.append("| 順 | 建物 | 種別 | カテゴリ | 訪問 | ユニーク | 平均滞在(分) |")
        L.append("|---|---|---|---|---|---|---|")
        for i, b in enumerate(visits["building_ranking"][:10], 1):
            ms = b["stay"]["mean_stay_min"]
            L.append(f"| {i} | {b['name']} | {b['kind']} | {b['category']} | "
                     f"{b['visits']} | {b['unique_visitors']} | "
                     f"{ms if ms is not None else '—'} |")
    else:
        L.append("(建物入館イベント 0件)")
    L.append("")

    L.append("### 上位の到着ノード(top 10)")
    if visits["node_ranking"]:
        L.append("| 順 | 地点 | カテゴリ | 到着 | ユニーク | 初訪問 |")
        L.append("|---|---|---|---|---|---|")
        for i, nd in enumerate(visits["node_ranking"][:10], 1):
            L.append(f"| {i} | {nd['name']} | {nd['category']} | {nd['arrivals']} | "
                     f"{nd['unique_visitors']} | {nd['first_visits']} |")
    else:
        L.append("(到着イベント 0件)")
    L.append("")

    L.append("### カテゴリ別 集客ランキング")
    if visits["category_ranking"]:
        L.append("| カテゴリ | 訪問 | ユニーク訪問者 |")
        L.append("|---|---|---|")
        for c in visits["category_ranking"][:12]:
            L.append(f"| {c['category']} | {c['visits']} | {c['unique_visitors']} |")
    else:
        L.append("(訪問イベント 0件)")
    L.append("")

    # ---- 時間帯 / 日次 ----
    L.append("### 時間帯トレンド(訪問数 / 時)")
    hr = visits["hour_trend"]
    if sum(hr):
        peak = sorted(range(24), key=lambda h: -hr[h])[:3]
        L.append("- ピーク時間帯: " + "、".join(f"{h}時({hr[h]}件)" for h in peak))
        L.append("- 時別: " + " ".join(f"{h:02d}={hr[h]}" for h in range(24) if hr[h]))
    else:
        L.append("(訪問イベント 0件)")
    L.append("")
    L.append("### 日次トレンド(訪問数 / 日)")
    dt = visits["day_trend"]
    if dt:
        L.append("- " + "、".join(f"Day{d}: {c}件" for d, c in dt.items()))
    else:
        L.append("(訪問イベント 0件)")
    L.append("")

    # ---- 行動圏 ----
    L.append("### 行動圏(エージェント別・上位5例)")
    asp = visits["activity_space"]
    if asp:
        avg_places = round(sum(a["n_distinct_places"] for a in asp) / len(asp), 1)
        L.append(f"- 1人あたり平均訪問ユニーク地点数: {avg_places}")
        for a in sorted(asp, key=lambda x: -x["total_visits"])[:5]:
            mf = a["most_frequent_place"]
            L.append(f"  - agent {a['agent_id']}: distinct {a['n_distinct_places']} / "
                     f"入館 {a['n_buildings_entered']}棟 / 最頻 {mf}")
    else:
        L.append("(訪問イベント 0件)")
    L.append("")

    # ---- 興味 ----
    g = interests["global"]
    L.append("## 興味観測(全体)")
    L.append(f"- 思考を起動したきっかけ(drive reason 全体): {_fmt_dist(g['drive_reason_dist'])}")
    L.append(f"- うち発火(granted): {_fmt_dist(g['drive_reason_granted_dist'])}")
    L.append(f"- 消費カテゴリ(spend cat): {_fmt_dist(g['spend_category_dist'])}")
    L.append(f"- 採用された語(top): {_fmt_dist(g['top_adopted_words'])}")
    L.append(f"- 検索語(top): {_fmt_dist(g['top_search_terms'])}")
    L.append(f"- いいねを集めた著者(agent_id: 件数): {_fmt_dist(g['top_liked_authors'])}")
    L.append(f"- 一日計画の活動種別(day_plan what): {_fmt_dist(g['plan_activity_dist'])}")
    L.append(f"- 計画で言及された場所(top): {_fmt_dist(g['top_planned_places'])}")
    L.append(f"- 覚醒を上げた原因(affect cause): {_fmt_dist(g['affect_cause_dist'])}")
    L.append("")

    L.append("### 観測項目ごとのイベント件数(0件=このランに該当なし)")
    for name, cnt in interests["event_counts"].items():
        mark = "" if cnt else "  ← 0件(このランでは観測不可)"
        L.append(f"- {name}: {cnt}件{mark}")
    L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
def _dump(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def observe(run_dir: str, out_dir: str | None = None) -> str:
    out_dir = out_dir or os.path.join(run_dir, "observe")
    os.makedirs(out_dir, exist_ok=True)

    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    n_agents = len(agents) or (max((e["agent_id"] for e in events), default=-1) + 1)
    mp = load_map(run_dir)

    visits = observe_visits(events, mp, run_dt.min_per_step(run_dir))
    interests = observe_interests(events, agents)

    run_name = os.path.basename(os.path.normpath(run_dir))
    _dump(os.path.join(out_dir, "visits.json"), visits)
    _dump(os.path.join(out_dir, "interests.json"), interests)
    write_poi_csv(visits, os.path.join(out_dir, "poi_ranking.csv"))
    write_summary_md(visits, interests, run_name, len(events), n_agents,
                     os.path.join(out_dir, "summary.md"))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Wave O: 興味・訪問の観測")
    ap.add_argument("run_dir", help="例: runs/eco80_3day")
    ap.add_argument("--out", default=None, help="出力先(既定 runs/<name>/observe)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    out = observe(args.run_dir, args.out)
    print(f"[observe] {args.run_dir} -> {out}")


if __name__ == "__main__":
    main()
