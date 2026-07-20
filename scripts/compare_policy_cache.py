#!/usr/bin/env python
"""行動方針キャッシュ(P2 S7)の OFF/ON 比較ハーネス(mock・ブラインド A/B の土台)。

    python scripts/compare_policy_cache.py [--steps 432] [--agents 12] [--seed 42]

同 seed で **policy_cache OFF / ON の 2 ラン(mock)** を回し、次を並べて出力する:
  (a) LLM 呼数の削減率(朝計画 plan 呼・全呼数)+ 再利用率と緩和段の内訳
  (b) 行動レパートリー指標(distinct 行き先数・移動数・計画 what 分布)= ポリシー汚染
      (誤った再利用で行き先が痩せる)の監視
  (c) L1 イベント種別の差分(OFF vs ON)

用途は「本番採否を比較実験で決める」ための素材づくり。**実 LLM 比較(ブラインド A/B)は
本選前に主計画者が実施**する(本スクリプトは mock で機構の効き=呼数削減・レパートリー保存を測る)。
sim 本体・schema・conf は一切変更しない(読み出し=集計のみ・in-process 実行)。

注意: 再利用は「2 日目以降で同じ物理骨格を再訪」して初めて起きるため、既定 --steps は約 3 日
(432 step)。mock は決定論・GPU 不要。実 LLM ランではない。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from society.config import load_config             # noqa: E402
from society.engine.simulation import Simulation   # noqa: E402


def _run(enabled: bool, *, steps: int, agents: int, seed: int,
         reuse_cap: float, out_root: str, extra: list[str]) -> dict:
    tag = "on" if enabled else "off"
    ov = [f"run.seed={seed}", f"run.n_agents={agents}", f"run.n_steps={steps}",
          f"run.name=pc_{tag}", "observer.snapshot_every=100000",
          f"cognition.policy_cache.enabled={'true' if enabled else 'false'}",
          f"cognition.policy_cache.reuse_rate_cap={reuse_cap}", *extra]
    sim = Simulation(load_config(ov), out_dir=os.path.join(out_root, f"pc_{tag}"))
    sim.run()
    return _metrics(sim)


def _metrics(sim) -> dict:
    ev = sim.logger.events
    calls = sim.logger.llm_calls
    kinds = Counter(e.kind for e in ev)
    purposes = Counter(c.get("purpose") for c in calls)

    # 行き先(route_start.dest / arrive.node)と計画 what 分布
    dests: set = set()
    n_moves = 0
    what_dist: Counter = Counter()
    relax: Counter = Counter()
    for e in ev:
        p = e.payload or {}
        if e.kind == "route_start":
            n_moves += 1
            if p.get("dest") is not None:
                dests.add(p["dest"])
        elif e.kind == "arrive" and p.get("node") is not None:
            dests.add(p["node"])
        elif e.kind == "day_plan":
            for it in p.get("plan", []) or []:
                if it.get("what"):
                    what_dist[it["what"]] += 1
        elif e.kind == "policy_reuse":
            relax[p.get("relax")] += 1

    day_plans = kinds.get("day_plan", 0)
    reuses = kinds.get("policy_reuse", 0)
    return {
        "n_events": len(ev),
        "kinds": kinds,
        "n_llm_calls": len(calls),
        "plan_calls": purposes.get("plan", 0),
        "purposes": purposes,
        "day_plans": day_plans,
        "reuses": reuses,
        "relax": relax,
        "distinct_dests": len(dests),
        "n_moves": n_moves,
        "what_dist": what_dist,
    }


def _pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "―"


def render(off: dict, on: dict, *, steps: int, agents: int, seed: int,
           reuse_cap: float) -> str:
    L: list[str] = []
    L.append("# 行動方針キャッシュ(P2 S7)OFF/ON 比較 — mock")
    L.append(f"seed={seed}・agents={agents}・steps={steps}・reuse_rate_cap={reuse_cap}")
    L.append("> 数値は mock ラン由来(渋谷実測でも実 LLM でもない)。機構の効き"
             "(呼数削減・レパートリー保存)を測る素材。実 LLM ブラインド A/B は主計画者が別途実施。\n")

    # (a) LLM 呼数の削減
    L.append("## (a) LLM 呼数の削減")
    plan_saved = off["plan_calls"] - on["plan_calls"]
    L.append("| 指標 | OFF | ON | 削減 |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| 朝計画 plan 呼 | {off['plan_calls']} | {on['plan_calls']} | "
             f"{plan_saved}({_pct(plan_saved, off['plan_calls'])}) |")
    tot_saved = off["n_llm_calls"] - on["n_llm_calls"]
    L.append(f"| 全 LLM 呼 | {off['n_llm_calls']} | {on['n_llm_calls']} | "
             f"{tot_saved}({_pct(tot_saved, off['n_llm_calls'])}) |")
    L.append(f"| day_plan 機会 | {off['day_plans']} | {on['day_plans']} | ― |")
    L.append("")
    opp = on["day_plans"]
    L.append(f"- 再利用(policy_reuse)= **{on['reuses']}** 件 / 計画機会 {opp} "
             f"= 再利用率 **{_pct(on['reuses'], opp)}**(上限 {reuse_cap})")
    if on["relax"]:
        seg = "・".join(f"stage{k}:{v}" for k, v in sorted(on["relax"].items(),
                                                          key=lambda kv: (kv[0] is None, kv[0])))
        L.append(f"- 緩和段の内訳(0=完全一致/1=場所種別を落とす/2=時間帯を落とす): {seg}")
    L.append("")

    # (b) 行動レパートリー
    L.append("## (b) 行動レパートリー(ポリシー汚染の監視)")
    L.append("| 指標 | OFF | ON |")
    L.append("|---|---:|---:|")
    L.append(f"| distinct 行き先ノード数 | {off['distinct_dests']} | {on['distinct_dests']} |")
    L.append(f"| 移動(route_start)数 | {off['n_moves']} | {on['n_moves']} |")
    L.append(f"| 計画 what の種類数 | {len(off['what_dist'])} | {len(on['what_dist'])} |")
    L.append("")
    whats = sorted(set(off["what_dist"]) | set(on["what_dist"]))
    if whats:
        L.append("計画 what 分布:")
        L.append("| what | OFF | ON |")
        L.append("|---|---:|---:|")
        for w in whats:
            L.append(f"| {w} | {off['what_dist'].get(w, 0)} | {on['what_dist'].get(w, 0)} |")
        L.append("")
    L.append("- distinct 行き先数・what 種類数が ON で大きく縮むなら **ポリシー汚染**"
             "(行動レパートリーが痩せる)の兆候。採用条件の一つ。")
    L.append("")

    # (c) L1 イベント差分
    L.append("## (c) L1 イベント種別の差分(ON − OFF)")
    L.append("| kind | OFF | ON | 差 |")
    L.append("|---|---:|---:|---:|")
    for k in sorted(set(off["kinds"]) | set(on["kinds"])):
        o, n = off["kinds"].get(k, 0), on["kinds"].get(k, 0)
        if o != n or k in ("day_plan", "policy_reuse"):
            L.append(f"| {k} | {o} | {n} | {n - o:+d} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="行動方針キャッシュ OFF/ON 比較(mock)")
    ap.add_argument("--steps", type=int, default=432, help="ステップ数(既定432≒3日)")
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cap", type=float, default=0.6, help="再利用率の上限")
    ap.add_argument("--out", default=None, help="レポート出力先 .md(既定=標準出力のみ)")
    ap.add_argument("--extra", nargs="*", default=[], help="共通 dotlist 上書き(OFF/ON 双方に適用)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="pc_compare_") as tmp:
        off = _run(False, steps=args.steps, agents=args.agents, seed=args.seed,
                   reuse_cap=args.reuse_cap, out_root=tmp, extra=list(args.extra))
        on = _run(True, steps=args.steps, agents=args.agents, seed=args.seed,
                  reuse_cap=args.reuse_cap, out_root=tmp, extra=list(args.extra))
    md = render(off, on, steps=args.steps, agents=args.agents, seed=args.seed,
                reuse_cap=args.reuse_cap)
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"\n[compare_policy_cache] -> {args.out}")


if __name__ == "__main__":
    main()
