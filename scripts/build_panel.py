#!/usr/bin/env python
"""stage0-1 データパイプライン: L1 検証 + tidy パネル化(第18バッチ②)。

    python scripts/build_panel.py runs/<name> [--warmup-days 7]

docs/plans/data-strategy.md の正式方針(生=正準・最小/指標=後処理で派生)に沿う
**読み出し専用**スクリプト。runs/<name> の L1(l1_events.parquet)・L2・agents.json・
summary.json を読むだけで、sim 本体・schema・calibrate_report は一切変更しない。

段階:
  stage0 検証  : schema 登録 kind との照合(未知 kind 検出)・step 欠落・重複の検査。
                 結果を panel/validation.json に出力し、コンソールにも要約を印字する。
  stage1 整形  : tidy(1行=1観測単位)の2枚のパネルを parquet で出力する。
    (a) panel/agent_day.parquet : 1行 = agent × 日。睡眠h・労働h・賃金収入・カテゴリ別
        支出・発話数・被傾聴数・SNS投稿/反応数・移動距離・訪問POI数・期末資産・warmup。
    (b) panel/run_day.parquet   : 1行 = run × 日。較正指標(calibrate_report と同じ分母の
        流儀)+ 多様性・ジニ等の日次値。

方針(誠実性): 値はすべてログ由来。計算できない指標は None(=データ不足)として正直に
出す。warmup は最初の warmup-days 日を True フラグにするだけで**除外はしない**(除外は
解析側=panel_stats で選択する。data-strategy.md の warm-up 条項)。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from society.observer import measure as m           # noqa: E402  (import のみ)
from society.observer.schema import EVENT_KINDS      # noqa: E402  (正準 kind 一覧)

MIN_PER_DAY = 1440
MIN_PER_STEP = 10
STEPS_PER_DAY = 144

# spend の cat は既知8種に固定し、未知は "other" へ寄せる(tidy: 列を固定して安定化)。
_SPEND_CATS = ["food", "shop", "nightlife", "taxi", "bus", "lodging",
               "medical", "fixed_cost", "other"]


def _cat(c) -> str:
    c = c or "other"
    return c if c in _SPEND_CATS else "other"


def _day(step: int) -> int:
    """step 基準の「活動日」(step // 144)。sim は 07:00 起点なので step 基準だと
    07:00〜翌07:00 が1日=夜間睡眠が同一日に収まり、n_days=n_steps/144 と一致する
    (calibrate_report の _daily_series が s//144 で日を切るのと同じ流儀)。"""
    return step // STEPS_PER_DAY


def _gini(values) -> float | None:
    """gini 係数(0=完全平等 .. 1=完全集中)。空/総和0 は None(データ不足)。"""
    xs = sorted(float(v) for v in values if v is not None)
    n = len(xs)
    if n == 0:
        return None
    total = sum(xs)
    if total <= 0:
        return None
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return round((2.0 * cum) / (n * total) - (n + 1.0) / n, 4)


# --------------------------------------------------------------------------- #
# stage0: 検証
# --------------------------------------------------------------------------- #
def validate_stage0(events: list[dict], n_steps: int) -> dict:
    """schema 照合・step 欠落・重複を検査した検証サマリを返す。"""
    kinds = Counter(e["kind"] for e in events)
    registered = set(EVENT_KINDS)
    unknown = sorted(k for k in kinds if k not in registered)

    steps_present = {e["step"] for e in events}
    expected = set(range(int(n_steps))) if n_steps else set()
    missing = sorted(expected - steps_present) if expected else []

    # 完全重複(同一 step・sim_min・agent・kind・payload)。決定論 sim では 0 が期待値。
    seen: set = set()
    dup = 0
    for e in events:
        key = (e["step"], e["sim_min"], e["agent_id"], e["kind"],
               json.dumps(e["payload"], ensure_ascii=False, sort_keys=True))
        if key in seen:
            dup += 1
        else:
            seen.add(key)

    return {
        "n_events": len(events),
        "n_kinds": len(kinds),
        "unknown_kinds": unknown,             # schema 未登録の kind(あれば要注意)
        "n_unknown_kind_events": sum(kinds[k] for k in unknown),
        "n_steps_expected": int(n_steps),
        "n_steps_present": len(steps_present),
        "n_missing_steps": len(missing),
        "missing_steps_head": missing[:20],   # 先頭のみ(全量は多すぎるため)
        "n_duplicate_events": dup,
        "ok": (not unknown) and (not missing) and dup == 0,
    }


# --------------------------------------------------------------------------- #
# stage1(a): agent × 日 パネル
# --------------------------------------------------------------------------- #
def build_agent_day(events: list[dict], agents: list[dict], n_days: int,
                    warmup_days: int, run_name: str,
                    weekday_of: dict[int, str]) -> dict[str, list]:
    ordered = sorted(events, key=lambda e: e["step"])
    meta_by_id = {int(a["id"]): a for a in agents if "id" in a}
    ids = sorted(meta_by_id) if meta_by_id else sorted(
        {e["agent_id"] for e in events if e["agent_id"] >= 0})

    # (aid, day) をキーにした集計器
    sleep_h: dict = defaultdict(float)
    wage_inc: dict = defaultdict(float)
    spend: dict = defaultdict(lambda: defaultdict(float))   # (aid,day)->cat->amt
    n_speak: dict = defaultdict(int)
    n_heard: dict = defaultdict(int)
    n_post: dict = defaultdict(int)
    n_react: dict = defaultdict(int)
    move_dist: dict = defaultdict(float)
    n_enter: dict = defaultdict(int)
    poi_set: dict = defaultdict(set)                        # 訪問した distinct 場所
    work_h: dict = defaultdict(float)

    # 労働h = 自分の勤務先建物での滞在時間(enter→exit の sim_min 差)。勤務先が不明な
    # agent は 0(=出勤を観測できない)。calibrate_report が n_working ピークで近似する
    # のと同様、per-agent では勤務先建物滞在を労働の proxy とする(出典コメント)。
    workplace = {}
    for aid, a in meta_by_id.items():
        wp = a.get("work_building") or a.get("part_time") or ""
        workplace[aid] = wp if isinstance(wp, str) and wp else None
    open_enter: dict = {}                                   # (aid,bid)->enter sim_min

    # 資産(現金 balance + 口座 account)の最新値を追跡し日末をスナップ。
    last_cash = {aid: float(meta_by_id.get(aid, {}).get("money") or 0.0)
                 for aid in ids}
    last_acct = {aid: 0.0 for aid in ids}
    wealth_day: dict = defaultdict(dict)                    # aid -> day -> wealth

    for e in ordered:
        k, aid, p = e["kind"], e["agent_id"], e["payload"]
        if aid is None or aid < 0:
            continue
        d = _day(e["step"])
        key = (aid, d)
        if k == "wake_up":
            ss = p.get("slept_steps")
            if ss:
                sleep_h[key] += float(ss) * MIN_PER_STEP / 60.0
        elif k == "wage":
            wage_inc[key] += float(p.get("amount") or 0)
        elif k == "spend":
            spend[key][_cat(p.get("cat"))] += float(p.get("amount") or 0)
        elif k == "speak":
            n_speak[key] += 1
        elif k == "hear":
            sp = p.get("speaker")                           # 被傾聴 = 話者側に計上
            if isinstance(sp, int) and sp >= 0:
                n_heard[(sp, d)] += 1
        elif k == "sns_post":
            n_post[key] += 1
        elif k in ("sns_like", "sns_reshare"):
            au = p.get("author")                            # 反応は投稿者(author)に計上
            if isinstance(au, int) and au >= 0:
                n_react[(au, d)] += 1
        elif k == "move_segment":
            move_dist[key] += float(p.get("dist_m") or 0)
        elif k == "enter_building":
            bid = p.get("building")
            n_enter[key] += 1
            if bid is not None:
                poi_set[key].add(("b", bid))
                open_enter.setdefault((aid, bid), e["sim_min"])
        elif k == "exit_building":
            bid = p.get("building")
            ent = open_enter.pop((aid, bid), None)
            if ent is not None and workplace.get(aid) == bid:
                dwell = max(0, e["sim_min"] - ent)         # 滞在分は sim_min 差で算定
                work_h[(aid, _day(e["step"]))] += dwell / 60.0
        elif k == "arrive":
            nid = p.get("node")
            if nid is not None:
                poi_set[key].add(("n", nid))

        # 資産追跡: balance(現金)/account(口座)を持つイベントで最新値を更新
        if "balance" in p and isinstance(p.get("balance"), (int, float)):
            last_cash[aid] = float(p["balance"])
        if "account" in p and isinstance(p.get("account"), (int, float)):
            last_acct[aid] = float(p["account"])
        if aid in last_cash:
            wealth_day[aid][d] = last_cash[aid] + last_acct.get(aid, 0.0)

    # 期末資産の日次充填(イベントが無い日は前日から carry-forward)
    wealth_end: dict = {}
    for aid in ids:
        init = float(meta_by_id.get(aid, {}).get("money") or 0.0)
        cur = init
        for d in range(n_days):
            if d in wealth_day.get(aid, {}):
                cur = wealth_day[aid][d]
            wealth_end[(aid, d)] = cur

    # ---- 行の生成(agent × day を全網羅=行数は必ず len(ids)×n_days)----
    cols: dict[str, list] = {c: [] for c in _AGENT_DAY_COLS}
    for aid in ids:
        a = meta_by_id.get(aid, {})
        age = a.get("age")
        for d in range(n_days):
            key = (aid, d)
            sp = spend.get(key, {})
            cols["run"].append(run_name)
            cols["agent_id"].append(int(aid))
            cols["name"].append(a.get("name"))
            cols["age"].append(int(age) if isinstance(age, (int, float)) else None)
            cols["occupation"].append(a.get("occupation"))
            cols["visitor"].append(bool(a.get("visitor")))
            cols["day"].append(int(d))
            cols["weekday"].append(weekday_of.get(d, ""))
            cols["warmup"].append(d < warmup_days)
            cols["sleep_h"].append(round(sleep_h.get(key, 0.0), 3))
            cols["work_h"].append(round(work_h.get(key, 0.0), 3))
            cols["wage_income"].append(round(wage_inc.get(key, 0.0), 2))
            cols["spend_total"].append(round(sum(sp.values()), 2))
            for c in _SPEND_CATS:
                cols[f"spend_{c}"].append(round(sp.get(c, 0.0), 2))
            cols["n_speak"].append(n_speak.get(key, 0))
            cols["n_heard"].append(n_heard.get(key, 0))
            cols["n_sns_post"].append(n_post.get(key, 0))
            cols["n_sns_react_recv"].append(n_react.get(key, 0))
            cols["move_dist_m"].append(round(move_dist.get(key, 0.0), 1))
            cols["n_enter_building"].append(n_enter.get(key, 0))
            cols["n_poi_visit"].append(len(poi_set.get(key, ())))
            cols["wealth_end"].append(round(wealth_end.get(key, 0.0), 2))
    return cols


_AGENT_DAY_COLS = (
    ["run", "agent_id", "name", "age", "occupation", "visitor", "day",
     "weekday", "warmup", "sleep_h", "work_h", "wage_income", "spend_total"]
    + [f"spend_{c}" for c in _SPEND_CATS]
    + ["n_speak", "n_heard", "n_sns_post", "n_sns_react_recv", "move_dist_m",
       "n_enter_building", "n_poi_visit", "wealth_end"]
)


def _agent_day_schema():
    import pyarrow as pa
    f = pa.field
    fields = [
        f("run", pa.string()), f("agent_id", pa.int64()), f("name", pa.string()),
        f("age", pa.int64()), f("occupation", pa.string()),
        f("visitor", pa.bool_()), f("day", pa.int64()), f("weekday", pa.string()),
        f("warmup", pa.bool_()), f("sleep_h", pa.float64()),
        f("work_h", pa.float64()), f("wage_income", pa.float64()),
        f("spend_total", pa.float64()),
    ]
    fields += [f(f"spend_{c}", pa.float64()) for c in _SPEND_CATS]
    fields += [
        f("n_speak", pa.int64()), f("n_heard", pa.int64()),
        f("n_sns_post", pa.int64()), f("n_sns_react_recv", pa.int64()),
        f("move_dist_m", pa.float64()), f("n_enter_building", pa.int64()),
        f("n_poi_visit", pa.int64()), f("wealth_end", pa.float64()),
    ]
    return pa.schema(fields)


# --------------------------------------------------------------------------- #
# stage1(b): run × 日 パネル
# --------------------------------------------------------------------------- #
_RUN_DAY_COLS = [
    "run", "day", "weekday", "warmup", "n_agents", "n_residents",
    "sleep_h_median", "spend_total", "spend_pp", "spend_food", "engel",
    "wage_total", "wage_pp", "n_speak", "speak_pp", "n_sns_post", "sns_post_pp",
    "n_crime", "crime_pp", "n_enter_building", "n_label_adopt", "n_transmission",
    "n_distinct_items_cum", "status_gini", "mean_grievance", "wealth_gini",
]


def _run_day_schema():
    import pyarrow as pa
    f = pa.field
    return pa.schema([
        f("run", pa.string()), f("day", pa.int64()), f("weekday", pa.string()),
        f("warmup", pa.bool_()), f("n_agents", pa.int64()),
        f("n_residents", pa.int64()), f("sleep_h_median", pa.float64()),
        f("spend_total", pa.float64()), f("spend_pp", pa.float64()),
        f("spend_food", pa.float64()), f("engel", pa.float64()),
        f("wage_total", pa.float64()), f("wage_pp", pa.float64()),
        f("n_speak", pa.int64()), f("speak_pp", pa.float64()),
        f("n_sns_post", pa.int64()), f("sns_post_pp", pa.float64()),
        f("n_crime", pa.int64()), f("crime_pp", pa.float64()),
        f("n_enter_building", pa.int64()), f("n_label_adopt", pa.int64()),
        f("n_transmission", pa.int64()), f("n_distinct_items_cum", pa.int64()),
        f("status_gini", pa.float64()), f("mean_grievance", pa.float64()),
        f("wealth_gini", pa.float64()),
    ])


def build_run_day(events: list[dict], agents: list[dict], l2: dict | None,
                  agent_day: dict[str, list], n_days: int, warmup_days: int,
                  run_name: str, weekday_of: dict[int, str]) -> dict[str, list]:
    n_agents = len(agents) or len({e["agent_id"] for e in events
                                   if e["agent_id"] >= 0})
    n_res = sum(1 for a in agents if not a.get("visitor")) or n_agents

    # 日次イベント集計
    spend_day: dict = defaultdict(float)
    spend_food_day: dict = defaultdict(float)
    wage_day: dict = defaultdict(float)
    speak_day: dict = defaultdict(int)
    post_day: dict = defaultdict(int)
    crime_day: dict = defaultdict(int)
    enter_day: dict = defaultdict(int)
    adopt_day: dict = defaultdict(int)
    trans_day: dict = defaultdict(int)
    items_by_day: dict = defaultdict(set)                  # その日までの distinct item
    for e in events:
        k, p = e["kind"], e["payload"]
        d = _day(e["step"])
        if k == "spend":
            amt = float(p.get("amount") or 0)
            spend_day[d] += amt
            if _cat(p.get("cat")) == "food":
                spend_food_day[d] += amt
        elif k == "wage":
            wage_day[d] += float(p.get("amount") or 0)
        elif k == "speak":
            speak_day[d] += 1
        elif k == "sns_post":
            post_day[d] += 1
        elif k == "crime":
            if p.get("kind") == "theft":
                crime_day[d] += 1
        elif k == "enter_building":
            enter_day[d] += 1
        elif k == "label_adopt":
            adopt_day[d] += 1
            iid = p.get("item_id")
            if iid is not None:
                items_by_day[d].add(iid)
        elif k == "transmission":
            trans_day[d] += 1
            iid = p.get("item_id")
            if iid is not None:
                items_by_day[d].add(iid)

    # 期末資産(agent_day.wealth_end)を日ごとに集めて wealth_gini を計算
    wealth_by_day: dict = defaultdict(list)
    sleep_by_day: dict = defaultdict(list)
    ad = agent_day
    for i in range(len(ad["day"])):
        d = ad["day"][i]
        wealth_by_day[d].append(ad["wealth_end"][i])
        if ad["sleep_h"][i] > 0:
            sleep_by_day[d].append(ad["sleep_h"][i])

    # L2 の日次末値(status_gini / mean_grievance)。無ければ None(データ不足)。
    l2_last: dict = defaultdict(dict)
    if l2 and l2.get("step"):
        steps = l2["step"]
        sg = l2.get("status_gini")
        mg = l2.get("mean_grievance")
        for i, s in enumerate(steps):
            d = int(s) // STEPS_PER_DAY
            if sg is not None and sg[i] is not None:
                l2_last[d]["status_gini"] = sg[i]
            if mg is not None and mg[i] is not None:
                l2_last[d]["mean_grievance"] = mg[i]

    cum_items: set = set()
    cols: dict[str, list] = {c: [] for c in _RUN_DAY_COLS}
    for d in range(n_days):
        cum_items |= items_by_day.get(d, set())
        stot = spend_day.get(d, 0.0)
        food = spend_food_day.get(d, 0.0)
        wtot = wage_day.get(d, 0.0)
        sm = sleep_by_day.get(d, [])
        cols["run"].append(run_name)
        cols["day"].append(int(d))
        cols["weekday"].append(weekday_of.get(d, ""))
        cols["warmup"].append(d < warmup_days)
        cols["n_agents"].append(int(n_agents))
        cols["n_residents"].append(int(n_res))
        cols["sleep_h_median"].append(round(st.median(sm), 3) if sm else None)
        cols["spend_total"].append(round(stot, 2))
        cols["spend_pp"].append(round(stot / n_agents, 2) if n_agents else None)
        cols["spend_food"].append(round(food, 2))
        cols["engel"].append(round(food / stot, 4) if stot > 0 else None)
        cols["wage_total"].append(round(wtot, 2))
        cols["wage_pp"].append(round(wtot / n_agents, 2) if n_agents else None)
        cols["n_speak"].append(speak_day.get(d, 0))
        cols["speak_pp"].append(round(speak_day.get(d, 0) / n_agents, 4)
                                if n_agents else None)
        cols["n_sns_post"].append(post_day.get(d, 0))
        cols["sns_post_pp"].append(round(post_day.get(d, 0) / n_agents, 4)
                                   if n_agents else None)
        cols["n_crime"].append(crime_day.get(d, 0))
        cols["crime_pp"].append(round(crime_day.get(d, 0) / n_agents, 6)
                                if n_agents else None)
        cols["n_enter_building"].append(enter_day.get(d, 0))
        cols["n_label_adopt"].append(adopt_day.get(d, 0))
        cols["n_transmission"].append(trans_day.get(d, 0))
        cols["n_distinct_items_cum"].append(len(cum_items))
        cols["status_gini"].append(l2_last.get(d, {}).get("status_gini"))
        cols["mean_grievance"].append(l2_last.get(d, {}).get("mean_grievance"))
        cols["wealth_gini"].append(_gini(wealth_by_day.get(d, [])))
    return cols


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def _write_parquet(path: str, schema, cols: dict[str, list]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pydict(cols, schema=schema)
    pq.write_table(table, path)
    return table


def _weekday_map(events: list[dict]) -> dict[int, str]:
    """weather イベントから 活動日 index -> 曜日 を作る(無ければ空。mock では空)。

    weather は暦日の 0:00(sim_min%1440==0)+ sim 開始時に発火する。活動日 d
    (step//144)は暦日 d の途中で始まる(開始 sim_min=420+d*1440 → 暦日 d)ので、
    暦日(sim_min//1440)で曜日を引けば活動日 d の始まりの曜日と一致する。最初に観測
    した曜日を採用(setdefault)する。"""
    out: dict[int, str] = {}
    for e in sorted(events, key=lambda ev: ev["step"]):
        if e["kind"] == "weather":
            wd = e["payload"].get("weekday")
            if wd:
                out.setdefault(e["sim_min"] // MIN_PER_DAY, wd)
    return out


def build(run_dir: str, warmup_days: int = 7, out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.path.join(run_dir, "panel")
    os.makedirs(out_dir, exist_ok=True)

    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    l2 = m.load_l2(run_dir)
    summary = {}
    spath = os.path.join(run_dir, "summary.json")
    if os.path.exists(spath):
        summary = json.load(open(spath, encoding="utf-8"))
    n_steps = int(summary.get("n_steps") or
                  (max((e["step"] for e in events), default=-1) + 1))
    max_day = max((_day(e["step"]) for e in events), default=-1)
    n_days = max(max_day + 1, -(-n_steps // STEPS_PER_DAY))  # ceil(n_steps/144)
    run_name = os.path.basename(os.path.normpath(run_dir))
    weekday_of = _weekday_map(events)

    validation = validate_stage0(events, n_steps)
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(validation, fh, ensure_ascii=False, indent=1)

    agent_day = build_agent_day(events, agents, n_days, warmup_days, run_name,
                                weekday_of)
    run_day = build_run_day(events, agents, l2, agent_day, n_days, warmup_days,
                            run_name, weekday_of)
    _write_parquet(os.path.join(out_dir, "agent_day.parquet"),
                   _agent_day_schema(), agent_day)
    _write_parquet(os.path.join(out_dir, "run_day.parquet"),
                   _run_day_schema(), run_day)

    return {
        "run": run_name, "out_dir": out_dir, "n_days": n_days,
        "n_agents": len(agents), "warmup_days": warmup_days,
        "n_agent_day_rows": len(agent_day["day"]),
        "n_run_day_rows": len(run_day["day"]),
        "validation": validation,
    }


def _print_report(info: dict) -> None:
    v = info["validation"]
    print(f"[build_panel] {info['run']} -> {info['out_dir']}")
    print(f"  agents={info['n_agents']} days={info['n_days']} "
          f"warmup_days={info['warmup_days']}")
    print(f"  agent_day 行数={info['n_agent_day_rows']}"
          f"(=agents×days={info['n_agents']}×{info['n_days']})"
          f" / run_day 行数={info['n_run_day_rows']}")
    print("  --- stage0 検証 ---")
    print(f"  events={v['n_events']} kinds={v['n_kinds']} "
          f"判定={'OK' if v['ok'] else '要確認'}")
    print(f"  未知 kind: {v['unknown_kinds'] or 'なし'}"
          f"({v['n_unknown_kind_events']}件)")
    print(f"  step 欠落: {v['n_missing_steps']}件"
          f"(期待 {v['n_steps_expected']} / 実在 {v['n_steps_present']})")
    print(f"  完全重複イベント: {v['n_duplicate_events']}件")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="stage0-1: L1 検証 + tidy パネル化(読み出し専用)")
    ap.add_argument("run_dir", help="例: runs/daily_llm_20a7d")
    ap.add_argument("--warmup-days", type=int, default=7,
                    help="最初の N 日を warmup=True にする(除外はしない。既定7)")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 runs/<name>/panel)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    info = build(args.run_dir, warmup_days=args.warmup_days, out_dir=args.out)
    _print_report(info)


if __name__ == "__main__":
    main()
