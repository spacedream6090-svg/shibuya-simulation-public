#!/usr/bin/env python
"""火種介入トレーサ(spark treatment 第53バッチ・読み出し専用)。

    python scripts/trace_spark.py --run smoke_spark_on
    python scripts/trace_spark.py --run runs/exp_spark_on --out panel/

「立ち上げには初期介入が不可欠か」を実験で見るための事後トレーサ。run dir の l1_events.parquet /
agents.json だけを読み(sim 本体 / schema は一切変更しない完全な後処理)、t=0 の spark_roster を起点に、
sparked 群が非 sparked 群より「速く・遠く」立ち上がったかを比較する。断定はしない=すべて観測量。

追跡する4つのこと(いずれも sparked 起点かを L1 から辿る=語彙可視化=transmission 資産の再利用):
  1. 活動量比: sparked / 非 sparked の1人あたり行動イベント数(speak/dm/主催/出店/提案/語彙 coin 等)。
  2. transmission 波及: sparked が送り手(from)になった item が何 hop 先まで・何人へ届いたか(時間順 BFS)。
  3. 関係形成の起点: sparked が関与した relation_tier(関係深化)の件数。
  4. 組織形成の起点: sparked が結成した group_found / sparked 結成グループへの group_join。

依存は標準ライブラリ + pyarrow のみ(pandas/duckdb 不使用)。spark_roster が無い(spark OFF)ランでも
例外なく動く(n_sparked=0 で全指標 0・空レポート)。1step = 10分・1日 = 144step。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")

# W2-3(Δt 対応の棚卸し): この定数は **未使用**(死んだ定数)。
# 本 module 内で step→日 の換算は行っていない(step をそのまま表示する)ため未使用。
# 将来ここで日を切るときは 144 を書かず scripts/run_dt.steps_per_day(run_dir) を通すこと
# (1 日の step 数は run.dt_min 依存: Δt=10 で 144・Δt=1 で 1440)。
STEPS_PER_DAY = 144          # 正準 Δt=10 の値(未使用)

# 1人あたりの「活動量」を構成する行動イベント種(世界に働きかける社会的行為。agent_id>=0 のもの)。
_ACTIVITY_KINDS = (
    "speak", "dm", "sns_post", "free_action",
    "event_host", "flyer_post", "group_found", "proposal", "candidacy",
    "venture_open", "vocab_coin", "label_coin",
)


# --------------------------------------------------------------------------- #
# ローダ(analyze_founders.py と同じ流儀。numpy を引き込まない最小実装)
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet を列射影 + RecordBatch 逐次で読み、payload を dict 展開して返す。"""
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "payload"]
    available = set(pq.read_schema(path).names)
    cols = [c for c in want if c in available]
    pf = pq.ParquetFile(path)
    out: list[dict] = []
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        n = len(d["step"])
        step, agent, kind, pays = d["step"], d["agent_id"], d["kind"], d["payload"]
        for i in range(n):
            raw = pays[i]
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({"step": int(step[i]), "agent": int(agent[i]),
                        "kind": kind[i], "payload": payload})
    return out


def load_agents(run_dir: str) -> dict[int, dict]:
    """agents.json を {agent_id: {...}} で返す(無ければ空)。"""
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[int, dict] = {}
    for a in data if isinstance(data, list) else []:
        try:
            out[int(a["id"])] = a
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# 0. 名簿(spark_roster)
# --------------------------------------------------------------------------- #
def find_roster(events: list[dict]) -> dict | None:
    """t=0 の spark_roster イベント payload を返す(無ければ None=spark OFF ラン)。"""
    for e in events:
        if e["kind"] == "spark_roster":
            return e["payload"]
    return None


# --------------------------------------------------------------------------- #
# 1. 活動量比(sparked / 非 sparked の1人あたり行動数)
# --------------------------------------------------------------------------- #
def activity_by_agent(events: list[dict]) -> dict[int, int]:
    """agent_id → 行動イベント総数(_ACTIVITY_KINDS)。agent_id<0(世界イベント)は除外。"""
    out: dict[int, int] = defaultdict(int)
    kinds = set(_ACTIVITY_KINDS)
    for e in events:
        aid = e["agent"]
        if aid is None or aid < 0:
            continue
        if e["kind"] in kinds:
            out[aid] += 1
    return dict(out)


def group_activity(activity: dict[int, int], sparked: set[int],
                   residents: list[int]) -> dict:
    """sparked 群 / 非 sparked 群の1人あたり平均行動数と比を返す。"""
    others = [a for a in residents if a not in sparked]
    sp = [a for a in residents if a in sparked]
    sum_sp = sum(activity.get(a, 0) for a in sp)
    sum_ot = sum(activity.get(a, 0) for a in others)
    mean_sp = sum_sp / len(sp) if sp else 0.0
    mean_ot = sum_ot / len(others) if others else 0.0
    ratio = round(mean_sp / mean_ot, 4) if mean_ot > 0 else None
    return {"n_sparked": len(sp), "n_others": len(others),
            "total_sparked": sum_sp, "total_others": sum_ot,
            "mean_per_sparked": round(mean_sp, 4),
            "mean_per_other": round(mean_ot, 4),
            "activity_ratio": ratio}


# --------------------------------------------------------------------------- #
# 2. transmission 波及(sparked が送り手になった item の時間順 BFS)
# --------------------------------------------------------------------------- #
def spark_transmission_reach(events: list[dict], sparked: set[int]) -> dict:
    """sparked が from(送り手)として関与した item ごとに、時間順 BFS で波及範囲を測る。

    各 item: エッジ(step, from, to)を step 昇順に1パスし、seed=sparked 送り手を hop0 として
    到達集合を成長させる。非 seed 到達者数の総和(重複は item 内でユニーク)と最大 hop を集計。"""
    # item_id -> [(step, from, to)]
    by_item: dict[str, list] = defaultdict(list)
    for e in events:
        if e["kind"] != "transmission":
            continue
        to = e["agent"]
        p = e["payload"]
        frm = p.get("from")
        item = p.get("item_id")
        if item is None or frm is None or to is None:
            continue
        try:
            by_item[str(item)].append((e["step"], int(frm), int(to)))
        except (TypeError, ValueError):
            continue

    n_items_seeded = 0
    total_reached = 0
    max_hops = 0
    per_item: list[dict] = []
    for item in sorted(by_item):
        edges = sorted(by_item[item])
        seeds = {frm for _s, frm, _t in edges if frm in sparked}
        if not seeds:
            continue                                     # sparked 発でない item は対象外
        n_items_seeded += 1
        hop = {a: 0 for a in seeds}
        for _s, frm, to in edges:                        # 時間順 1 パス BFS
            if frm in hop and to not in hop:
                hop[to] = hop[frm] + 1
        reached = [a for a, h in hop.items() if h >= 1]
        item_max = max((hop[a] for a in reached), default=0)
        total_reached += len(reached)
        max_hops = max(max_hops, item_max)
        per_item.append({"item": item, "seeds": sorted(seeds),
                         "n_reached": len(reached), "max_hop": item_max})
    per_item.sort(key=lambda r: (-r["n_reached"], -r["max_hop"], r["item"]))
    return {"n_items_seeded_by_sparked": n_items_seeded,
            "total_reached": total_reached, "max_hops": max_hops,
            "top_items": per_item[:10]}


# --------------------------------------------------------------------------- #
# 3/4. 関係形成 / 組織形成の起点
# --------------------------------------------------------------------------- #
def spark_relation_origin(events: list[dict], sparked: set[int]) -> dict:
    """sparked が関与した relation_tier(関係深化)の件数と、sparked↔sparked の内訳。"""
    involving = 0
    both = 0
    for e in events:
        if e["kind"] != "relation_tier":
            continue
        aid = e["agent"]
        other = e["payload"].get("other")
        try:
            other = int(other) if other is not None else None
        except (TypeError, ValueError):
            other = None
        a_in = aid in sparked
        o_in = other in sparked if other is not None else False
        if a_in or o_in:
            involving += 1
        if a_in and o_in:
            both += 1
    return {"relation_tier_involving_sparked": involving,
            "relation_tier_within_sparked": both}


def spark_org_origin(events: list[dict], sparked: set[int]) -> dict:
    """sparked が結成した group_found と、sparked 結成グループへの group_join を数える。"""
    founded_by_sparked: set[str] = set()
    n_found = 0
    for e in events:
        if e["kind"] != "group_found":
            continue
        aid = e["agent"]
        gid = e["payload"].get("group_id")
        if aid in sparked and gid is not None:
            founded_by_sparked.add(str(gid))
            n_found += 1
    n_join_to_sparked = 0
    for e in events:
        if e["kind"] != "group_join":
            continue
        p = e["payload"]
        gid = p.get("group_id")
        founder = p.get("founder")
        if (gid is not None and str(gid) in founded_by_sparked) \
                or (founder is not None and _as_int(founder) in sparked):
            n_join_to_sparked += 1
    return {"group_found_by_sparked": n_found,
            "group_join_to_sparked_groups": n_join_to_sparked}


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 実行・統合
# --------------------------------------------------------------------------- #
def trace(run_dir: str, out_dir: str) -> dict:
    """ラン1本をトレースし、panel/spark_trace.json を書いて集計 dict を返す。"""
    run_name = os.path.basename(os.path.normpath(run_dir))
    events = load_events(run_dir)
    agents = load_agents(run_dir)
    roster = find_roster(events)

    residents = [aid for aid, a in agents.items() if not a.get("visitor")]
    if not residents:                                    # agents.json に visitor 列が無い等の後退
        residents = sorted({e["agent"] for e in events if e["agent"] is not None
                            and e["agent"] >= 0})

    if roster is None:
        result = {"run": run_name, "spark_enabled": False, "n_sparked": 0,
                  "note": "spark_roster 不在=spark OFF ラン(比較対象なし)"}
    else:
        sparked = set(int(x) for x in (roster.get("ids") or []))
        activity = activity_by_agent(events)
        result = {
            "run": run_name, "spark_enabled": True,
            "n_sparked": len(sparked), "sparked_ids": sorted(sparked),
            "roster_params": roster.get("params", {}),
            "roster_menus": roster.get("menus", {}),
            "activity": group_activity(activity, sparked, residents),
            "transmission": spark_transmission_reach(events, sparked),
            "relations": spark_relation_origin(events, sparked),
            "organizations": spark_org_origin(events, sparked),
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "spark_trace.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    result["_path"] = out_path
    return result


def render_text(res: dict) -> str:
    """人が読むテキストレポート(コンソール出力用)。"""
    lines = [f"[spark] run={res['run']}"]
    if not res.get("spark_enabled"):
        lines.append("[spark] spark_roster 不在=spark OFF ラン(sparked 0 名・比較なし)")
        return "\n".join(lines)
    act = res["activity"]
    tr = res["transmission"]
    rel = res["relations"]
    org = res["organizations"]
    lines.append(f"[spark] sparked {res['n_sparked']} 名 / 非 sparked "
                 f"{act['n_others']} 名(居住者)")
    lines.append(f"[spark] 活動量: sparked {act['mean_per_sparked']}/人 vs 非 sparked "
                 f"{act['mean_per_other']}/人 → 比 {act['activity_ratio']}")
    lines.append(f"[spark] transmission 波及: sparked 発 item {tr['n_items_seeded_by_sparked']} 件 / "
                 f"到達 {tr['total_reached']} 名 / 最大 {tr['max_hops']} hop")
    lines.append(f"[spark] 関係形成起点: sparked 関与 relation_tier {rel['relation_tier_involving_sparked']} 件"
                 f"(うち sparked 相互 {rel['relation_tier_within_sparked']})")
    lines.append(f"[spark] 組織形成起点: sparked 結成グループ {org['group_found_by_sparked']} 件 / "
                 f"そこへの加入 {org['group_join_to_sparked_groups']} 件")
    if tr["top_items"]:
        lines.append("\n[spark top items(波及)]")
        for it in tr["top_items"][:5]:
            lines.append(f"  {it['item']}: 到達 {it['n_reached']} 名 / {it['max_hop']} hop "
                         f"(seeds={it['seeds']})")
    return "\n".join(lines)


def _pick_run(arg_run: str | None) -> str:
    """--run 指定 or l1_events.parquet を持つ最新ランを返す(analyze_founders と同流儀)。"""
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isfile(os.path.join(d, "l1_events.parquet")):
            raise SystemExit(f"[spark] l1_events.parquet が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[spark] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[spark] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _resolve_out(run_dir: str, arg_out: str | None) -> str:
    """--out。既定=run_dir/panel。相対指定は run_dir 起点、絶対はそのまま。"""
    if not arg_out:
        return os.path.join(run_dir, "panel")
    return arg_out if os.path.isabs(arg_out) else os.path.join(run_dir, arg_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="火種介入トレーサ(spark treatment・読み出し専用)")
    ap.add_argument("--run", default=None,
                    help="ラン名(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None,
                    help="出力ディレクトリ(既定: <run>/panel)。相対はラン基準")
    args = ap.parse_args()

    run_dir = _pick_run(args.run)
    out_dir = _resolve_out(run_dir, args.out)
    res = trace(run_dir, out_dir)
    print(render_text(res))
    print(f"\n[spark] -> {res['_path']}")


if __name__ == "__main__":
    main()
