#!/usr/bin/env python
"""商業4観点レポート(第18バッチ②・ユーザー承認: 人流商圏/広告ファネル/消費売上/
トレンドイベントROI)。

    python scripts/commercial_report.py runs/<name> [--out runs/<name>/commercial_report.md]

docs/research/commercial-analytics.md の「各KPIを L1 イベントから計算する式」に忠実な
**読み出し専用** レポート。runs/<name> の L1・agents.json・config.yaml(地図解決)を
読むだけで、sim 本体・schema・calibrate_report は一切変更しない。

観点(research doc の §):
  1. 人流・商圏 : footfall(enter_building/通過)・dwell(exit−enter の sim_min 差)・
                 回遊(enter_building 系列の遷移行列)・商圏(home 距離分布 + Huff 来訪シェア)
  2. 広告ファネル: ad_exposure イベント(第18バッチ③で新設予定)から接触リーチ・頻度・
                 target 来店転換(3日以内)・spend。★イベント0件のランは「広告なしラン」
                 と明記してスキップ(古いランは必ず0件)
  3. 消費・売上 : カテゴリ×時間帯×客層の売上・店舗ランキング(spend を在館建物へ帰属)・
                 客単価(在館1回あたり spend 合計)・価格係数(price_change)との対応
  4. トレンド・ROI: 語の採用曲線(label_adopt/transmission の累積)・SNSカスケード規模
                 (sns_reshare 連鎖)・イベント開催→周辺売上リフト(開催日 vs 非開催日。
                 ★リフトは正負両方=渋谷ハロウィン知見)

方針(誠実性): 値はすべてログ由来。該当イベントが無い指標は「データ不足」と正直に出す
(捏造禁止)。conversion/売上帰属は spend に売り手が無いため「在館(enter〜exit)中の
spend を建物へ帰属」する seam を使う(research doc §6-1)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import run_dt                                       # noqa: E402  (W2-3: Δt の単一の源)
from society.observer import measure as m           # noqa: E402  (import のみ)

MIN_PER_DAY = 1440                                  # 分/日は Δt 非依存(実時間)
# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 の 144 = 従来と 1 ビットも変わらない。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY
_CONV_WINDOW_MIN = 3 * MIN_PER_DAY                    # 広告→来店の帰属窓(3日)= research §2.3
_HUFF_BETA = 2.0                                     # Huff 距離減衰 b(research §1 KPI#4)

_SPEND_CATS = ["food", "shop", "nightlife", "taxi", "bus", "lodging",
               "medical", "fixed_cost", "other"]
_CAT_LABEL = {"food": "飲食", "shop": "買い物", "nightlife": "夜遊び",
              "taxi": "タクシー", "bus": "バス", "lodging": "宿泊",
              "medical": "医療", "fixed_cost": "固定費", "other": "その他"}
# 住宅の建物カテゴリ(footfall 上位「施設」表からは既定で除外=自宅はノイズ)
_RESIDENTIAL_CATS = {"house?", "residential", "自宅"}


def _cat(c) -> str:
    c = c or "other"
    return c if c in _SPEND_CATS else "other"


def _hour(sim_min: int) -> int:
    return (sim_min // 60) % 24


def _day(step: int, spd: int = STEPS_PER_DAY) -> int:
    """step 基準の活動日(W2-3: `spd` の既定は正準 144 = 従来と完全同値)。"""
    return step // int(spd)


def _age_band(age) -> str:
    if not isinstance(age, (int, float)):
        return "不明"
    a = int(age)
    if a < 20:
        return "〜19"
    if a < 30:
        return "20代"
    if a < 40:
        return "30代"
    if a < 50:
        return "40代"
    if a < 65:
        return "50-64"
    return "65+"


def _quantiles(xs: list) -> dict:
    """中央値・平均・p25・p75・最大・件数(空なら None)。"""
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return {"n": 0, "median": None, "mean": None, "p25": None,
                "p75": None, "max": None}

    def q(p):
        if n == 1:
            return xs[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

    return {"n": n, "median": round(q(0.5), 1), "mean": round(sum(xs) / n, 1),
            "p25": round(q(0.25), 1), "p75": round(q(0.75), 1),
            "max": round(xs[-1], 1)}


# --------------------------------------------------------------------------- #
# 地図(建物名・カテゴリ・座標)の解決。observe.py の流儀を最小移植(出典: scripts/observe.py)。
# --------------------------------------------------------------------------- #
def load_map(run_dir: str) -> dict:
    empty = {"bname": {}, "bcat": {}, "node_xy": {}, "poi_building": {},
             "resolved": False}
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
    cands = ([map_rel] if os.path.isabs(map_rel)
             else [os.path.join(_ROOT, map_rel), os.path.join(run_dir, map_rel),
                   map_rel])
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        return empty
    try:
        city = json.load(open(path, encoding="utf-8"))
    except Exception:
        return empty
    bname, bcat = {}, {}
    poi_by_b: dict = defaultdict(list)
    poi_building: dict = defaultdict(set)         # POI名 -> 建物id 集合(広告 target 解決用)
    for p in city.get("pois", []):
        if p.get("building"):
            poi_by_b[p["building"]].append(p.get("cat", ""))
            if p.get("name"):
                poi_building[p["name"]].add(p["building"])
    for b in city.get("buildings", []):
        bid = b["id"]
        bname[bid] = b.get("name", "") or ""
        cats = [c for c in poi_by_b.get(bid, []) if c]
        bcat[bid] = (Counter(cats).most_common(1)[0][0] if cats
                     else (b.get("kind", "") or "generic"))
    node_xy = {n["id"]: (n.get("x"), n.get("y")) for n in city.get("nodes", [])
               if n.get("x") is not None}
    return {"bname": bname, "bcat": bcat, "node_xy": node_xy,
            "poi_building": {k: set(v) for k, v in poi_building.items()},
            "resolved": True}


def _bldg_label(bid: str, mp: dict) -> str:
    nm = mp["bname"].get(bid, "")
    if nm:
        return nm
    cat = mp["bcat"].get(bid, "")
    return f"{cat or '建物'}#{bid}"


# --------------------------------------------------------------------------- #
# 在館セッション(enter〜exit)の抽出。売上帰属・dwell・客単価・回遊の共通土台。
# --------------------------------------------------------------------------- #
def build_sessions(events: list[dict], spd: int = STEPS_PER_DAY):
    """agent の建物在館セッション列を返す。

    各セッション: {agent, building, enter_min, exit_min, enter_step, day, hour,
                   dwell_min(None=未退出), spend(在館中 spend 合計), spend_cat}。
    enter〜exit の間(exit 不明なら enter 以降・次の enter まで)に出た spend を帰属する。
    """
    ordered = sorted(events, key=lambda e: e["step"])
    # agent -> 現在開いているセッション
    open_sess: dict[int, dict] = {}
    sessions: list[dict] = []

    def close(aid, exit_min=None):
        s = open_sess.pop(aid, None)
        if s is None:
            return
        if exit_min is not None:
            s["exit_min"] = exit_min
            s["dwell_min"] = max(0, exit_min - s["enter_min"])
        sessions.append(s)

    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if aid is None or aid < 0:
            continue
        if k == "enter_building":
            if aid in open_sess:                 # 退出ログ無しで別建物へ=前を閉じる
                close(aid, exit_min=e["sim_min"])
            bid = p.get("building")
            open_sess[aid] = {
                "agent": aid, "building": bid, "enter_min": e["sim_min"],
                "exit_min": None, "enter_step": e["step"], "day": _day(e["step"], spd),
                "hour": _hour(e["sim_min"]), "dwell_min": None,
                "spend": 0.0, "spend_cat": Counter()}
        elif k == "exit_building":
            close(aid, exit_min=e["sim_min"])
        elif k == "spend":
            s = open_sess.get(aid)
            if s is not None:                    # 在館中の spend を建物へ帰属(§6-1 seam)
                amt = float(p.get("amount") or 0)
                s["spend"] += amt
                s["spend_cat"][_cat(p.get("cat"))] += amt
    for aid in list(open_sess):                  # run 末で未退出のセッションも残す
        close(aid, exit_min=None)
    return sessions


# --------------------------------------------------------------------------- #
# 観点1: 人流・商圏
# --------------------------------------------------------------------------- #
def analyze_footfall(events: list[dict], sessions: list[dict], agents: list[dict],
                     mp: dict, weekday_of: dict) -> dict:
    # footfall: 建物×(時間帯・曜日)。ユニーク訪問者と延べ入館。
    b_visits: Counter = Counter()
    b_uniq: dict = defaultdict(set)
    b_hour: dict = defaultdict(Counter)
    b_wd: dict = defaultdict(Counter)
    b_xy: dict = defaultdict(lambda: [0.0, 0.0, 0])          # 建物座標(enter平均)
    hour_wd_total: dict = defaultdict(int)                    # (hour, weekday)->件数
    street_pass = 0                                          # move_segment 通過(街路 footfall)
    for e in events:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if k == "move_segment" and isinstance(aid, int) and aid >= 0:
            street_pass += 1
    for s in sessions:
        bid = s["building"]
        if bid is None:
            continue
        b_visits[bid] += 1
        b_uniq[bid].add(s["agent"])
        b_hour[bid][s["hour"]] += 1
        wd = weekday_of.get(s["day"], f"D{s['day']}")
        b_wd[bid][wd] += 1
        hour_wd_total[(s["hour"], wd)] += 1

    # 建物座標: enter_building の x,y 平均
    for e in events:
        if e["kind"] == "enter_building":
            bid = e["payload"].get("building")
            if bid is not None and e["x"] is not None:
                acc = b_xy[bid]
                acc[0] += e["x"]
                acc[1] += e["y"]
                acc[2] += 1
    b_centroid = {bid: (a[0] / a[2], a[1] / a[2]) for bid, a in b_xy.items()
                  if a[2] > 0}

    footfall_rows = []
    for bid, n in b_visits.most_common():
        hh = b_hour[bid]
        peak_h = max(hh, key=lambda h: hh[h]) if hh else None
        footfall_rows.append({
            "building": bid, "label": _bldg_label(bid, mp),
            "cat": mp["bcat"].get(bid, ""), "visits": n,
            "unique": len(b_uniq[bid]), "peak_hour": peak_h,
            "by_weekday": dict(b_wd[bid].most_common())})

    # dwell(退出まで観測できたセッション)
    dwell_all = [s["dwell_min"] for s in sessions if s["dwell_min"] is not None]
    dwell_stats = _quantiles(dwell_all)
    n_open = sum(1 for s in sessions if s["dwell_min"] is None)
    # bounce(<3分)率
    bounce = (sum(1 for d in dwell_all if d < 3) / len(dwell_all)
              if dwell_all else None)

    # 商圏: home 座標と建物 centroid の距離分布 + Huff 来訪シェア
    home_xy = {}
    for a in agents:
        h = a.get("home")
        xy = mp["node_xy"].get(h) if h else None
        if xy and xy[0] is not None:
            home_xy[int(a["id"])] = (float(xy[0]), float(xy[1]))
    catchment = _catchment(sessions, b_visits, b_centroid, home_xy, mp)

    return {
        "street_pass_moves": street_pass,
        "footfall": footfall_rows,
        "dwell": dwell_stats, "n_open_sessions": n_open, "bounce_rate": bounce,
        "hour_weekday_total": {f"{h}|{wd}": c
                               for (h, wd), c in sorted(hour_wd_total.items())},
        "catchment": catchment,
        "home_resolved": len(home_xy),
    }


def _catchment(sessions, b_visits, b_centroid, home_xy, mp) -> list:
    """建物ごとに来訪者の自宅距離分布 + Huff 期待シェア(§1 KPI#4)。

    Huff: P_ij = A_j/d_ij^b ÷ Σ_k A_k/d_ik^b(A=魅力度=総来訪数、d=home→建物距離、
    b=距離減衰)。各建物の期待来訪シェアを、住民全員の来訪確率の平均として近似する。
    """
    # home 距離分布(建物ごと)
    b_dist: dict = defaultdict(list)
    for s in sessions:
        bid, aid = s["building"], s["agent"]
        if bid in b_centroid and aid in home_xy:
            cx, cy = b_centroid[bid]
            hx, hy = home_xy[aid]
            b_dist[bid].append(math.hypot(cx - hx, cy - hy))

    # Huff: 上位建物(魅力度=総来訪)に絞って期待シェアを計算
    top_b = [bid for bid, _ in b_visits.most_common(20) if bid in b_centroid]
    attract = {bid: float(b_visits[bid]) for bid in top_b}
    homes = list(home_xy.values())
    huff_share = {bid: 0.0 for bid in top_b}
    if homes and attract:
        for hx, hy in homes:
            weights = {}
            for bid in top_b:
                cx, cy = b_centroid[bid]
                d = max(1.0, math.hypot(cx - hx, cy - hy))
                weights[bid] = attract[bid] / (d ** _HUFF_BETA)
            tot = sum(weights.values())
            if tot > 0:
                for bid in top_b:
                    huff_share[bid] += weights[bid] / tot / len(homes)

    rows = []
    total_visits_top = sum(b_visits[bid] for bid in top_b) or 1
    for bid in top_b:
        dq = _quantiles(b_dist.get(bid, []))
        rows.append({
            "building": bid, "label": _bldg_label(bid, mp),
            "visits": b_visits[bid],
            "actual_share": round(b_visits[bid] / total_visits_top, 4),
            "huff_share": round(huff_share[bid], 4),
            "home_dist_median": dq["median"], "home_dist_n": dq["n"]})
    return rows


def analyze_circulation(sessions: list[dict], mp: dict) -> dict:
    """回遊: 同一 agent×日 の enter_building 系列 → 建物間遷移行列・上位フロー。"""
    by_ad: dict = defaultdict(list)
    for s in sorted(sessions, key=lambda x: (x["agent"], x["enter_min"])):
        if s["building"] is not None:
            by_ad[(s["agent"], s["day"])].append(s["building"])
    trans: Counter = Counter()
    n_chains = 0
    for seq in by_ad.values():
        if len(seq) >= 2:
            n_chains += 1
            for a, b in zip(seq, seq[1:]):
                if a != b:
                    trans[(a, b)] += 1
    flows = [{"from": _bldg_label(a, mp), "to": _bldg_label(b, mp), "count": c}
             for (a, b), c in trans.most_common(15)]
    return {"n_multi_stop_agent_days": n_chains, "n_transitions": sum(trans.values()),
            "top_flows": flows}


# --------------------------------------------------------------------------- #
# 観点2: 広告ファネル(ad_exposure。0件=広告なしランでスキップ)
# --------------------------------------------------------------------------- #
def _resolve_target(tgt: str, mp: dict) -> set:
    """広告 target 名 → 建物 id 集合。POI名の完全一致(POI の building フィールド)→
    建物名の完全一致 → 建物名の部分一致 の順で解決。空集合=解決不可。"""
    bids = mp.get("poi_building", {}).get(tgt)
    if bids:
        return set(bids)
    exact = {bid for bid, nm in mp["bname"].items() if nm and nm == tgt}
    if exact:
        return exact
    return {bid for bid, nm in mp["bname"].items()
            if nm and (tgt in nm or nm in tgt)}


def analyze_ads(events: list[dict], sessions: list[dict], mp: dict,
                agents: list[dict]) -> dict:
    """接触リーチ・頻度・target 来店転換(3日窓・**target 建物への enter のみ**)+
    非接触対照(campaign 非接触者の期間中 target 来店率)とリフト。

    転換の判定は「その agent が最初の接触 t0 から3日以内に **target に解決された建物**へ
    enter_building した」場合のみ(無関係な建物への入館は数えない)。target が地図で
    解決できない場合は分母から除外し件数を明記する(捏造禁止)。
    リフト = 接触側転換率 − 非接触対照率。窓は非対称(接触側=3日窓/対照=全期間)で、
    その旨をレポートに注記する。効果の「盛りすぎ」回避=research doc §3 の最重要点。"""
    exp = [e for e in events if e["kind"] == "ad_exposure"]
    if not exp:
        return {"has_ads": False}

    all_ids = sorted({int(a["id"]) for a in agents if "id" in a}) or sorted(
        {e["agent_id"] for e in events
         if isinstance(e["agent_id"], int) and e["agent_id"] >= 0})

    # ---- リーチ・頻度・最初接触(agent×campaign×target)----
    by_agent: Counter = Counter()
    campaigns: Counter = Counter()
    camp_exposed: dict = defaultdict(set)          # campaign -> 接触 agent 集合
    first_exp: dict = {}                           # (agent, campaign, target) -> 最初接触min
    for e in sorted(exp, key=lambda x: x["step"]):
        aid, p = e["agent_id"], e["payload"]
        if not isinstance(aid, int) or aid < 0:
            continue
        by_agent[aid] += 1
        camp = str(p.get("campaign") or "")
        campaigns[camp] += 1
        camp_exposed[camp].add(aid)
        tgt = p.get("target") or ""
        if tgt:
            first_exp.setdefault((aid, camp, tgt), e["sim_min"])

    reach = len(by_agent)
    freq_dist = Counter(by_agent.values())

    # ---- target → 建物集合の解決 ----
    targets = sorted({tgt for (_a, _c, tgt) in first_exp})
    target_bids = {tgt: _resolve_target(tgt, mp) for tgt in targets}
    unresolved = sorted(t for t, b in target_bids.items() if not b)

    # agent ごとの在館セッション索引(target 照合用)
    sess_by_agent: dict = defaultdict(list)
    for s in sessions:
        if s["building"] is not None:
            sess_by_agent[s["agent"]].append(s)

    # ---- 転換: (agent, campaign, target) 単位。target 建物への enter のみ数える ----
    conv_den = 0
    conv_hit = 0
    excluded_unresolved = 0
    post_spend = []                                # 転換した target 建物在館の spend のみ
    unit: dict = defaultdict(lambda: {"den": 0, "hit": 0})   # (camp,tgt) -> 転換集計
    for (aid, camp, tgt), t0 in sorted(first_exp.items()):
        bids = target_bids.get(tgt) or set()
        if not bids:
            excluded_unresolved += 1               # target 解決不可=分母から除外
            continue
        conv_den += 1
        unit[(camp, tgt)]["den"] += 1
        matched = [s for s in sess_by_agent.get(aid, ())
                   if s["building"] in bids
                   and t0 <= s["enter_min"] <= t0 + _CONV_WINDOW_MIN]
        if matched:
            conv_hit += 1
            unit[(camp, tgt)]["hit"] += 1
            post_spend.append(sum(s["spend"] for s in matched))
    conv_rate = conv_hit / conv_den if conv_den else None

    # ---- 非接触対照: campaign×target ごとに、その campaign に接触が無い agent のうち
    #      期間中(全期間)に target 建物へ入館した人の割合。窓は接触側と非対称=注記必須。
    unit_rows = []
    ctrl_vis_total = 0
    ctrl_n_total = 0
    for (camp, tgt) in sorted(unit):
        bids = target_bids[tgt]
        control = [aid for aid in all_ids if aid not in camp_exposed.get(camp, ())]
        visited = sum(1 for aid in control
                      if any(s["building"] in bids for s in sess_by_agent.get(aid, ())))
        ctrl_rate = visited / len(control) if control else None
        u = unit[(camp, tgt)]
        u_rate = u["hit"] / u["den"] if u["den"] else None
        lift = (u_rate - ctrl_rate) if (u_rate is not None
                                        and ctrl_rate is not None) else None
        ctrl_vis_total += visited
        ctrl_n_total += len(control)
        unit_rows.append({
            "campaign": camp, "target": tgt, "n_target_buildings": len(bids),
            "exposed_den": u["den"], "exposed_hit": u["hit"],
            "exposed_rate": u_rate, "control_n": len(control),
            "control_visited": visited, "control_rate": ctrl_rate, "lift": lift})
    ctrl_rate_all = ctrl_vis_total / ctrl_n_total if ctrl_n_total else None
    lift_all = (conv_rate - ctrl_rate_all) if (conv_rate is not None
                                               and ctrl_rate_all is not None) else None

    return {
        "has_ads": True, "n_exposures": len(exp), "reach": reach,
        "frequency_dist": dict(sorted(freq_dist.items())),
        "campaigns": dict(campaigns.most_common()),
        "targets_resolved": len(targets) - len(unresolved),
        "targets_unresolved": unresolved,
        "excluded_unresolved_pairs": excluded_unresolved,
        "conversion_targets": conv_den, "conversions": conv_hit,
        "conversion_rate": conv_rate,
        "control_rate": ctrl_rate_all, "lift": lift_all,
        "unit_rows": unit_rows,
        "post_visit_spend": _quantiles(post_spend),
        "n_exposed_agents": len(by_agent),
    }


# --------------------------------------------------------------------------- #
# 観点3: 消費・売上
# --------------------------------------------------------------------------- #
def analyze_sales(events: list[dict], sessions: list[dict], agents: list[dict],
                  mp: dict, weekday_of: dict) -> dict:
    meta = {int(a["id"]): a for a in agents if "id" in a}
    # カテゴリ×時間帯×客層
    cat_total: Counter = Counter()
    cat_hour: dict = defaultdict(Counter)
    cat_ageband: dict = defaultdict(Counter)
    cat_occ: dict = defaultdict(Counter)
    for e in events:
        if e["kind"] != "spend":
            continue
        aid, p = e["agent_id"], e["payload"]
        amt = float(p.get("amount") or 0)
        c = _cat(p.get("cat"))
        cat_total[c] += amt
        cat_hour[c][_hour(e["sim_min"])] += amt
        a = meta.get(aid, {})
        cat_ageband[c][_age_band(a.get("age"))] += amt
        cat_occ[c][a.get("occupation") or "不明"] += amt

    # 店舗(building)ランキング: 在館中 spend を建物へ帰属
    store_sales: Counter = Counter()
    store_visits: Counter = Counter()
    baskets = []                                            # 在館1回あたり spend(>0のみ)
    for s in sessions:
        if s["building"] is None:
            continue
        store_visits[s["building"]] += 1
        if s["spend"] > 0:
            store_sales[s["building"]] += s["spend"]
            baskets.append(s["spend"])
    store_rows = []
    for bid, amt in store_sales.most_common(15):
        v = store_visits[bid]
        store_rows.append({
            "building": bid, "label": _bldg_label(bid, mp),
            "cat": mp["bcat"].get(bid, ""), "sales": round(amt, 1),
            "visits": v, "per_visit": round(amt / v, 1) if v else None})

    # 価格感応: price_change(cat, ratio)と当該 cat の総消費の対応(相関は弱い注記つき)
    price_by_cat: dict = defaultdict(list)
    for e in events:
        if e["kind"] == "price_change":
            p = e["payload"]
            price_by_cat[_cat(p.get("cat"))].append(float(p.get("ratio") or 1.0))
    price_rows = []
    for c in _SPEND_CATS:
        ratios = price_by_cat.get(c, [])
        if ratios:
            price_rows.append({
                "cat": c, "label": _CAT_LABEL[c], "n_changes": len(ratios),
                "mean_ratio": round(sum(ratios) / len(ratios), 3),
                "total_spend": round(cat_total.get(c, 0.0), 1)})

    return {
        "cat_total": {c: round(v, 1) for c, v in cat_total.most_common()},
        "cat_hour": {c: [round(cat_hour[c].get(h, 0.0), 1) for h in range(24)]
                     for c in cat_total},
        "cat_ageband": {c: dict(cat_ageband[c].most_common()) for c in cat_total},
        "cat_occupation": {c: dict(cat_occ[c].most_common(8)) for c in cat_total},
        "store_ranking": store_rows,
        "basket": _quantiles(baskets),
        "price_sensitivity": price_rows,
        "n_spend_events": sum(1 for e in events if e["kind"] == "spend"),
    }


# --------------------------------------------------------------------------- #
# 観点4: トレンド・イベントROI
# --------------------------------------------------------------------------- #
def analyze_trends(events: list[dict], sessions: list[dict], mp: dict,
                   n_days: int, spd: int = STEPS_PER_DAY) -> dict:
    # 語の採用曲線(label_adopt / transmission の日次累積)
    adopt_day: Counter = Counter()
    trans_day: Counter = Counter()
    for e in events:
        d = _day(e["step"], spd)
        if e["kind"] == "label_adopt":
            adopt_day[d] += 1
        elif e["kind"] == "transmission":
            trans_day[d] += 1
    adopt_cum, trans_cum = [], []
    ca = ctv = 0
    for d in range(n_days):
        ca += adopt_day.get(d, 0)
        ctv += trans_day.get(d, 0)
        adopt_cum.append(ca)
        trans_cum.append(ctv)

    # SNS カスケード規模(sns_reshare を (author,post_id) で束ねる)
    casc: Counter = Counter()
    for e in events:
        if e["kind"] == "sns_reshare":
            p = e["payload"]
            casc[(p.get("author"), p.get("post_id"))] += 1
    casc_sizes = sorted(casc.values(), reverse=True)
    casc_hist = dict(sorted(Counter(casc_sizes).items()))

    # イベント開催 → 会場周辺売上リフト(開催日 vs 非開催日・同時間帯)
    event_lift = _event_lift(events, sessions, spd)

    return {
        "adopt_cumulative": adopt_cum, "transmission_cumulative": trans_cum,
        "total_adopt": adopt_cum[-1] if adopt_cum else 0,
        "cascade_sizes_top": casc_sizes[:10], "cascade_hist": casc_hist,
        "n_cascades": len(casc),
        "event_lift": event_lift,
    }


def _event_lift(events: list[dict], sessions: list[dict],
                spd: int = STEPS_PER_DAY) -> dict:
    """event_host / annual_event / crowd_surge の開催日を拾い、会場周辺(近接建物)の
    売上を開催日 vs 非開催日で比較(同時間帯窓)。リフトは正負両方を許す(§4 ハロウィン知見)。

    会場 node の座標が要るため、簡易化として「開催日全体の総 spend」を非開催日平均と比較する
    (近接ノード限定は座標依存で脆いので、日次総額の増減=イベント効果の粗い代理とする)。
    """
    host_days = set()
    hosts = []
    for e in events:
        if e["kind"] in ("event_host", "annual_event"):
            d = _day(e["step"], spd)
            host_days.add(d)
            hosts.append({"kind": e["kind"], "day": d,
                          "title": e["payload"].get("title")
                          or e["payload"].get("name") or "",
                          "node": e["payload"].get("node")})
    if not host_days:
        return {"has_events": False}

    # 日次総 spend
    spend_day: Counter = Counter()
    for e in events:
        if e["kind"] == "spend":
            spend_day[_day(e["step"], spd)] += float(e["payload"].get("amount") or 0)
    all_days = sorted(set(range(max(spend_day.keys(), default=0) + 1)))
    on = [spend_day.get(d, 0.0) for d in all_days if d in host_days]
    off = [spend_day.get(d, 0.0) for d in all_days if d not in host_days]
    on_mean = st.mean(on) if on else None
    off_mean = st.mean(off) if off else None
    lift = ((on_mean - off_mean) / off_mean) if (on_mean is not None
            and off_mean not in (None, 0)) else None
    return {"has_events": True, "n_events": len(hosts), "host_days": sorted(host_days),
            "hosts": hosts[:10], "spend_on_event_days_mean": on_mean,
            "spend_off_days_mean": off_mean, "lift_ratio": lift}


# --------------------------------------------------------------------------- #
# レポート描画
# --------------------------------------------------------------------------- #
def _fmt_yen(v) -> str:
    if v is None:
        return "―"
    try:
        return f"{int(round(float(v))):,}円"
    except (TypeError, ValueError):
        return str(v)


def render(run_name, n_events, n_agents, n_days, mp, foot, ads, sales, trends,
           circ) -> str:
    L: list[str] = []
    L.append(f"# 商業4観点レポート: {run_name}\n")
    L.append("L1 ログから商業KPIを算出(docs/research/commercial-analytics.md の計算式)。"
             "値はすべてログ由来の what-if 実験値であり渋谷の実測ではない。該当イベントが"
             "無い指標は「データ不足」と明記(捏造なし)。\n")
    L.append(f"- 規模: agents {n_agents} / events {n_events} / {n_days} 日"
             f" / 地図解決 {'あり' if mp['resolved'] else 'なし(raw id)'}\n")

    # ---- 1. 人流・商圏 ----
    L.append("## 1. 人流・商圏")
    d = foot["dwell"]
    L.append(f"- 街路 footfall(move_segment 通過): {foot['street_pass_moves']:,} 件")
    if d["n"]:
        L.append(f"- dwell 滞在時間(退出観測 {d['n']}件): 中央 {d['median']}分 / "
                 f"平均 {d['mean']}分 / p25 {d['p25']} / p75 {d['p75']} / "
                 f"最長 {d['max']}分(未退出 {foot['n_open_sessions']}件)")
        L.append(f"- bounce(<3分退店)率: "
                 f"{round(foot['bounce_rate'] * 100, 1) if foot['bounce_rate'] is not None else '―'}%")
    else:
        L.append("- dwell 滞在時間: 退出まで観測できたセッションが 0件(データ不足)")
    hw = foot["hour_weekday_total"]
    if hw:
        top = sorted(hw.items(), key=lambda kv: -kv[1])[:3]
        L.append("- 混雑ピーク(時間帯×曜日・入館): "
                 + "、".join(f"{k.replace('|', '時/')}({v}件)" for k, v in top))
    L.append("")
    L.append("### footfall 上位施設(建物・入館ユニーク訪問者順・住宅は除外)")
    commercial_rows = [r for r in foot["footfall"]
                       if r["cat"] not in _RESIDENTIAL_CATS]
    if commercial_rows:
        rows = sorted(commercial_rows, key=lambda r: -r["unique"])[:15]
        L.append("| 順 | 施設 | カテゴリ | 入館 | ユニーク | ピーク時 | 曜日内訳 |")
        L.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows, 1):
            wd = "、".join(f"{k}:{v}" for k, v in list(r["by_weekday"].items())[:4])
            L.append(f"| {i} | {r['label']} | {r['cat']} | {r['visits']} | "
                     f"{r['unique']} | {r['peak_hour']}時 | {wd} |")
        n_res = len(foot["footfall"]) - len(commercial_rows)
        if n_res:
            L.append(f"(住宅 {n_res} 棟を表から除外。街路 footfall・dwell には含む)")
    elif foot["footfall"]:
        L.append("(商業施設への入館 0件=住宅のみ。データ不足)")
    else:
        L.append("(enter_building 0件=データ不足)")
    L.append("")
    L.append("### 回遊(同一 agent×日 の建物間遷移・上位フロー)")
    if circ["top_flows"]:
        L.append(f"- 複数店回遊した agent×日: {circ['n_multi_stop_agent_days']} / "
                 f"総遷移 {circ['n_transitions']}")
        L.append("| from | → | to | 回数 |")
        L.append("|---|---|---|---|")
        for f in circ["top_flows"][:10]:
            L.append(f"| {f['from']} | → | {f['to']} | {f['count']} |")
    else:
        L.append("(回遊: 複数店訪問が 0件=データ不足)")
    L.append("")
    L.append("### 商圏(来訪者の自宅距離 + Huff 期待来訪シェア)")
    if foot["catchment"] and foot["home_resolved"]:
        L.append(f"- 自宅座標を解決できた agent: {foot['home_resolved']} 人"
                 f"(Huff 距離減衰 b={_HUFF_BETA})")
        L.append("| 施設 | 来訪 | 実シェア | Huff期待 | 自宅距離中央値 |")
        L.append("|---|---|---|---|---|")
        for r in foot["catchment"][:10]:
            L.append(f"| {r['label']} | {r['visits']} | {r['actual_share']} | "
                     f"{r['huff_share']} | {r['home_dist_median']} |")
    else:
        L.append("(自宅座標を解決できず=データ不足。地図 world.map が必要)")
    L.append("")

    # ---- 2. 広告ファネル ----
    L.append("## 2. 広告ファネル(OOH ad_exposure)")
    if not ads["has_ads"]:
        L.append("**広告なしラン**: ad_exposure イベントが 0件(第18バッチ③の OOH 実装前の"
                 "ランでは必ず0件)。本セクションはスキップする。\n")
    else:
        L.append(f"- 接触(exposure): {ads['n_exposures']}件 / リーチ(接触した人): "
                 f"{ads['reach']}人")
        L.append(f"- 頻度分布(接触回数→人数): {ads['frequency_dist']}")
        L.append(f"- キャンペーン別接触: {ads['campaigns']}")
        if ads["targets_unresolved"]:
            L.append(f"- target 解決不可: {ads['targets_unresolved']}"
                     f"(接触ペア {ads['excluded_unresolved_pairs']}件を分母から除外)")
        cr = ads["conversion_rate"]
        ctrl = ads["control_rate"]
        lift = ads["lift"]
        L.append(f"- target 来店転換(3日以内に **target 建物**へ enter_building): "
                 f"{ads['conversions']}/{ads['conversion_targets']} = "
                 f"{round(cr * 100, 1) if cr is not None else '―'}%")
        L.append(f"- 非接触対照(campaign 非接触者の期間中 target 来店率): "
                 f"{round(ctrl * 100, 1) if ctrl is not None else '―'}%"
                 f" / **リフト(接触−対照): "
                 f"{round(lift * 100, 1) if lift is not None else '―'}pt**")
        if ads["unit_rows"]:
            L.append("")
            L.append("| キャンペーン | target | 接触側転換(3日窓) | 対照来店率(全期間) | リフト |")
            L.append("|---|---|---|---|---|")
            for r in ads["unit_rows"]:
                er = (f"{r['exposed_hit']}/{r['exposed_den']} = "
                      f"{round(r['exposed_rate'] * 100, 1)}%"
                      if r["exposed_rate"] is not None else "―")
                cr2 = (f"{r['control_visited']}/{r['control_n']} = "
                       f"{round(r['control_rate'] * 100, 1)}%"
                       if r["control_rate"] is not None else "―")
                lf = (f"{round(r['lift'] * 100, 1)}pt" if r["lift"] is not None
                      else "―")
                L.append(f"| {r['campaign']} | {r['target']} | {er} | {cr2} | {lf} |")
        ps = ads["post_visit_spend"]
        if ps["n"]:
            L.append(f"- 転換来店時の spend(target 建物在館のみ): 中央 "
                     f"{_fmt_yen(ps['median'])} / 平均 {_fmt_yen(ps['mean'])}"
                     f"(n={ps['n']})")
        else:
            L.append("- 転換来店時の spend(target 建物在館のみ): 0件=データ不足")
        L.append("> 注: 窓は非対称(接触側=最初接触から3日窓/対照=ラン全期間の来店率)。"
                 "対照が有利に出やすい比較であり、リフトは保守的な見積りになる。")
        L.append("")

    # ---- 3. 消費・売上 ----
    L.append("## 3. 消費・売上")
    if sales["n_spend_events"]:
        L.append(f"- spend イベント: {sales['n_spend_events']}件")
        L.append("### カテゴリ別 総売上")
        L.append("| カテゴリ | 総額 | 主要客層(年代) | 主要客層(職業) |")
        L.append("|---|---|---|---|")
        for c, amt in list(sales["cat_total"].items()):
            ab = "、".join(f"{k}:{_fmt_yen(v)}"
                          for k, v in list(sales["cat_ageband"].get(c, {}).items())[:3])
            oc = "、".join(f"{k}:{_fmt_yen(v)}"
                          for k, v in list(sales["cat_occupation"].get(c, {}).items())[:2])
            L.append(f"| {_CAT_LABEL.get(c, c)} | {_fmt_yen(amt)} | {ab} | {oc} |")
        L.append("")
        b = sales["basket"]
        L.append(f"### 客単価(在館1回あたり spend 合計・>0 のみ)")
        L.append(f"- 中央 {_fmt_yen(b['median'])} / 平均 {_fmt_yen(b['mean'])} / "
                 f"p25 {_fmt_yen(b['p25'])} / p75 {_fmt_yen(b['p75'])}(n={b['n']})")
        L.append("")
        L.append("### 店舗(建物)売上ランキング(在館中 spend を帰属)")
        if sales["store_ranking"]:
            L.append("| 順 | 店舗 | カテゴリ | 売上 | 来店 | 客単価 |")
            L.append("|---|---|---|---|---|---|")
            for i, r in enumerate(sales["store_ranking"][:10], 1):
                L.append(f"| {i} | {r['label']} | {r['cat']} | {_fmt_yen(r['sales'])} "
                         f"| {r['visits']} | {_fmt_yen(r['per_visit'])} |")
        else:
            L.append("(在館中 spend が 0件=売上帰属できず。データ不足)")
        L.append("")
        if sales["price_sensitivity"]:
            L.append("### 価格感応(price_change の cat×ratio と総消費)")
            L.append("| カテゴリ | 変更回数 | 平均比率 | 総消費 |")
            L.append("|---|---|---|---|")
            for r in sales["price_sensitivity"]:
                L.append(f"| {r['label']} | {r['n_changes']} | {r['mean_ratio']} | "
                         f"{_fmt_yen(r['total_spend'])} |")
            L.append("> 注: 価格弾力性の厳密推定には同一財の時系列が要る。ここは cat 集計の"
                     "対応表(粗い代理)。")
        else:
            L.append("### 価格感応: price_change イベント 0件(データ不足)")
        L.append("")
    else:
        L.append("(spend イベント 0件=データ不足)\n")

    # ---- 4. トレンド・イベントROI ----
    L.append("## 4. トレンド・イベントROI")
    L.append(f"- 語の採用 累積(日次): {trends['adopt_cumulative']}(総 "
             f"{trends['total_adopt']})")
    L.append(f"- 伝播 累積(日次): {trends['transmission_cumulative']}")
    if trends["total_adopt"] == 0 and not trends["transmission_cumulative"]:
        L.append("  ← 採用/伝播 0件=S字曲線はデータ不足")
    L.append("")
    L.append("### SNS カスケード規模(sns_reshare 連鎖)")
    if trends["n_cascades"]:
        L.append(f"- カスケード数: {trends['n_cascades']} / 規模上位: "
                 f"{trends['cascade_sizes_top']} / 規模ヒスト(規模:本数) "
                 f"{trends['cascade_hist']}")
    else:
        L.append("(sns_reshare 0件=データ不足)")
    L.append("")
    L.append("### イベント開催→周辺売上リフト(正負両方あり得る=渋谷ハロウィン知見)")
    el = trends["event_lift"]
    if el.get("has_events"):
        lift = el["lift_ratio"]
        L.append(f"- 開催イベント: {el['n_events']}件 / 開催日 {el['host_days']}")
        L.append(f"- 開催日 日次売上 平均 {_fmt_yen(el['spend_on_event_days_mean'])} "
                 f"vs 非開催日 {_fmt_yen(el['spend_off_days_mean'])}")
        if lift is not None:
            sign = "+" if lift >= 0 else ""
            L.append(f"- 売上リフト: {sign}{round(lift * 100, 1)}%"
                     f"(正=集客増収 / 負=ハロウィン型の混雑減収)")
        else:
            L.append("- 売上リフト: 比較対象(非開催日)が不足=データ不足")
    else:
        L.append("(event_host / annual_event 0件=データ不足)")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# parquet 補助出力(panel/commercial_*.parquet)
# --------------------------------------------------------------------------- #
def _write_parquet(path: str, cols: dict, schema=None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = (pa.Table.from_pydict(cols, schema=schema) if schema
             else pa.table(cols))
    pq.write_table(table, path)


def _dump_parquets(out_panel: str, foot, sales, trends) -> None:
    """商業 parquet を常に出力する(データ0件でも schema 付き空テーブルを書く=
    下流ツールが必ずファイルを見つけられるようにする)。"""
    import pyarrow as pa
    os.makedirs(out_panel, exist_ok=True)
    # footfall
    ff = foot["footfall"]
    _write_parquet(os.path.join(out_panel, "commercial_footfall.parquet"), {
        "building": [r["building"] for r in ff],
        "label": [r["label"] for r in ff],
        "cat": [r["cat"] for r in ff],
        "visits": [int(r["visits"]) for r in ff],
        "unique": [int(r["unique"]) for r in ff],
        "peak_hour": [r["peak_hour"] if r["peak_hour"] is not None else -1
                      for r in ff]},
        schema=pa.schema([
            pa.field("building", pa.string()), pa.field("label", pa.string()),
            pa.field("cat", pa.string()), pa.field("visits", pa.int64()),
            pa.field("unique", pa.int64()), pa.field("peak_hour", pa.int64())]))
    # store sales
    sr = sales["store_ranking"]
    _write_parquet(os.path.join(out_panel, "commercial_sales.parquet"), {
        "building": [r["building"] for r in sr],
        "label": [r["label"] for r in sr],
        "cat": [r["cat"] for r in sr],
        "sales": [float(r["sales"]) for r in sr],
        "visits": [int(r["visits"]) for r in sr],
        "per_visit": [float(r["per_visit"]) if r["per_visit"] is not None
                      else None for r in sr]},
        schema=pa.schema([
            pa.field("building", pa.string()), pa.field("label", pa.string()),
            pa.field("cat", pa.string()), pa.field("sales", pa.float64()),
            pa.field("visits", pa.int64()), pa.field("per_visit", pa.float64())]))
    # adoption curve
    ac = trends["adopt_cumulative"]
    _write_parquet(os.path.join(out_panel, "commercial_adoption.parquet"), {
        "day": list(range(len(ac))),
        "adopt_cumulative": [int(x) for x in ac],
        "transmission_cumulative": [int(x) for x in
                                    trends["transmission_cumulative"]]},
        schema=pa.schema([
            pa.field("day", pa.int64()),
            pa.field("adopt_cumulative", pa.int64()),
            pa.field("transmission_cumulative", pa.int64())]))


# --------------------------------------------------------------------------- #
def report(run_dir: str, out_md: str | None = None) -> dict:
    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    mp = load_map(run_dir)
    run_name = os.path.basename(os.path.normpath(run_dir))
    n_agents = len(agents) or (max((e["agent_id"] for e in events), default=-1) + 1)
    n_steps = max((e["step"] for e in events), default=-1) + 1
    # W2-3: 「1 日」はこのランの run.dt_min で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    n_days = -(-n_steps // spd) if n_steps else 0

    # weekday マップ(build_panel と同じ流儀: 暦日 sim_min//1440 で first-wins)
    weekday_of: dict = {}
    for e in sorted(events, key=lambda ev: ev["step"]):
        if e["kind"] == "weather":
            wd = e["payload"].get("weekday")
            if wd:
                weekday_of.setdefault(e["sim_min"] // MIN_PER_DAY, wd)

    sessions = build_sessions(events, spd)
    foot = analyze_footfall(events, sessions, agents, mp, weekday_of)
    circ = analyze_circulation(sessions, mp)
    ads = analyze_ads(events, sessions, mp, agents)
    sales = analyze_sales(events, sessions, agents, mp, weekday_of)
    trends = analyze_trends(events, sessions, mp, n_days, spd)

    md = render(run_name, len(events), n_agents, n_days, mp, foot, ads, sales,
                trends, circ)
    out_md = out_md or os.path.join(run_dir, "commercial_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    _dump_parquets(os.path.join(run_dir, "panel"), foot, sales, trends)
    return {"run": run_name, "out_md": out_md, "has_ads": ads["has_ads"],
            "md": md, "foot": foot, "sales": sales, "trends": trends}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="商業4観点レポート(読み出し専用)")
    ap.add_argument("run_dir", help="例: runs/daily_llm_20a7d")
    ap.add_argument("--out", default=None,
                    help="出力 md(既定 runs/<name>/commercial_report.md)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    info = report(args.run_dir, args.out)
    print(info["md"])
    print(f"\n[commercial_report] {args.run_dir} -> {info['out_md']}")


if __name__ == "__main__":
    main()
