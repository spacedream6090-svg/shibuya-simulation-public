"""W2 P0 — 非LLMエンジン単価 c(秒/agent-step)の実測ベースライン。

本番プロファイル(conf/production.yaml)+ mock バックエンドで、現行エンジンの
非LLM単価 c を規模別(既定 N=300/1000/3000)に実測し、cProfile でホットスポットを
特定する(docs/plans/w2-execution-plan.md §4 P0 / 個別検収基準)。これが P1(背景SoA化)の
投資判断の土台になる。既知の参考値: 本番プロファイル full≈0.060 / リーン mock≈0.00183
(docs/research/scale-feasibility.md §4)。

────────────────────────────────────────────────────────────────────────
掟(このスクリプトが守ること):
  - **エンジン本体・conf・既定挙動は一切変更しない**。規模拡張は既存名簿(100体)の巡回複製に
    委ねる = simulation.py が既に行う `roster[agent_id % len(roster)]`(および
    organizations.attach の `agent.id % n_roster`)で、run.n_agents を増やすだけで N 体に
    循環拡張される。追加の合成名簿ファイルは作らない(engine の既定機構をそのまま使う)。
  - **mock バックエンドのみ**(model.backend=mock を明示上書き。実LLM呼び出しは一切しない)。
  - 出力は runs/_profile/ 配下(gitignore 済み)。
  - c は cProfile を **外した** 素の wall で測る(プロファイラは別ラン)。

測り方:
  - Simulation を構築(=1回限りのセットアップ。c には含めない)。
  - ウォームアップ warmup step を実行(計時しない = キャッシュ充填・初日境界の一過性を除外)。
  - 残り step を1 step ずつ perf_counter で計時し、総和 / (N × 計時 step 数) = c。
  - c は「非LLM」= mock で LLM をスタブ化した Python step 処理の単価(全チャネル + イベント
    記録の RAM バッファ込み。ディスク flush は finalize のみ = 計時外)。

使い方:
  python scripts/profile_engine.py
  python scripts/profile_engine.py --sizes 300 1000 --steps 24 --warmup 2
  python scripts/profile_engine.py --steps 12            # 遅い場合は step を落として外挿
  python scripts/profile_engine.py --profile-n 1000 --profile-steps 24 --topk 30
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import pstats
import sys
import time
from pathlib import Path

# Windows コンソール(cp932)対策: 進捗 print の en-dash 等で死なない(既存スクリプトの流儀)。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from society.config import load_config          # noqa: E402
from society.engine import scheduler            # noqa: E402
from society.engine.simulation import Simulation  # noqa: E402

_PROFILE = str(_ROOT / "conf" / "production.yaml")


def _build_sim(n: int, steps: int, seed: int) -> Simulation:
    """本番プロファイル + mock で N 体・steps step の Simulation を構築する。

    engine/config は無改変。run.n_agents / run.n_steps / run.seed / run.name と
    model.backend=mock のみ dotlist で上書き(production は backend を触らないので基底の
    mock のままだが、実LLMへの取り違え事故を防ぐため明示する)。名簿(100体)は engine が
    agent_id 巡回で N 体へ拡張する = 計測専用の合成注入は不要。
    """
    overrides = [
        f"run.n_agents={n}",
        f"run.n_steps={steps}",
        f"run.seed={seed}",
        f"run.name=_profile/N{n}_s{steps}_seed{seed}",
        "model.backend=mock",
    ]
    cfg = load_config(overrides=overrides, profile=_PROFILE)
    return Simulation(cfg)


def measure_scale(n: int, steps: int, warmup: int, seed: int) -> dict:
    """N 体の非LLM単価 c を実測する(cProfile なし・素の wall)。"""
    gc.collect()
    t0 = time.perf_counter()
    sim = _build_sim(n, steps, seed)
    build_s = time.perf_counter() - t0

    # ウォームアップ(計時しない)
    for step in range(min(warmup, steps)):
        scheduler.run_step(sim, step)

    per_step: list[float] = []
    for step in range(warmup, steps):
        t = time.perf_counter()
        scheduler.run_step(sim, step)
        per_step.append(time.perf_counter() - t)

    measured_steps = len(per_step)
    total = sum(per_step)
    c = total / (n * measured_steps) if (n and measured_steps) else 0.0
    n_events = sim.logger.total_n_events()
    events_per_agent_step = (n_events / (n * steps)) if (n and steps) else 0.0

    # 後片付け(finalize の重い flush は呼ばない = 計時外・ディスク書かない)。RAM 解放。
    sim.logger.events.clear()
    del sim
    gc.collect()

    per_sorted = sorted(per_step)
    median = per_sorted[len(per_sorted) // 2] if per_sorted else 0.0
    return {
        "n_agents": n,
        "n_steps": steps,
        "warmup": warmup,
        "measured_steps": measured_steps,
        "build_s": round(build_s, 3),
        "total_step_wall_s": round(total, 4),
        "c_s_per_agent_step": c,
        "wall_per_step_s_mean": round(total / measured_steps, 4) if measured_steps else 0.0,
        "wall_per_step_s_median": round(median, 4),
        "wall_per_step_s_min": round(min(per_step), 4) if per_step else 0.0,
        "wall_per_step_s_max": round(max(per_step), 4) if per_step else 0.0,
        "n_events_buffered": n_events,
        "events_per_agent_step": round(events_per_agent_step, 3),
    }


def profile_hotspots(n: int, steps: int, warmup: int, seed: int, topk: int) -> list[dict]:
    """N 体 × steps step を cProfile し、cumtime 上位 topk 関数を返す。

    warmup step はプロファイラ外で回し、その後の step 群をプロファイルする(初回一過性を除く)。
    """
    gc.collect()
    sim = _build_sim(n, steps, seed)
    for step in range(min(warmup, steps)):
        scheduler.run_step(sim, step)

    prof = cProfile.Profile()
    prof.enable()
    for step in range(warmup, steps):
        scheduler.run_step(sim, step)
    prof.disable()

    stats = pstats.Stats(prof)
    rows: list[dict] = []
    # stats.stats: (filename, lineno, funcname) -> (cc, nc, tt, ct, callers)
    for (filename, lineno, funcname), (cc, nc, tt, ct, _callers) in stats.stats.items():
        short = os.path.basename(filename) if filename else "~"
        rows.append({
            "func": f"{short}:{lineno}({funcname})",
            "ncalls": nc,
            "primcalls": cc,
            "tottime_s": round(tt, 4),
            "cumtime_s": round(ct, 4),
            "percall_cum_s": round(ct / nc, 6) if nc else 0.0,
        })
    rows.sort(key=lambda r: r["cumtime_s"], reverse=True)
    sim.logger.events.clear()
    del sim
    gc.collect()
    return rows[:topk]


def _fmt_c(c: float) -> str:
    return f"{c:.6f}"


def print_scale_table(results: list[dict]) -> None:
    print("\n=== 規模別 非LLM単価 c(秒/agent-step)===")
    hdr = (f"{'N':>6} {'step':>5} {'計時':>5} {'build_s':>8} "
           f"{'総step_s':>9} {'c(s/a·step)':>13} {'ms/step':>9} {'ev/a·step':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['n_agents']:>6} {r['n_steps']:>5} {r['measured_steps']:>5} "
              f"{r['build_s']:>8.2f} {r['total_step_wall_s']:>9.3f} "
              f"{_fmt_c(r['c_s_per_agent_step']):>13} "
              f"{r['wall_per_step_s_mean'] * 1000:>9.1f} "
              f"{r['events_per_agent_step']:>10.2f}")
    # スケーリング所見(N に対する c の比)
    if len(results) >= 2:
        print("\n--- スケーリング(基準=最小 N の c)---")
        base = results[0]
        for r in results:
            ratio_n = r["n_agents"] / base["n_agents"]
            ratio_c = (r["c_s_per_agent_step"] / base["c_s_per_agent_step"]
                       if base["c_s_per_agent_step"] else 0.0)
            print(f"  N={r['n_agents']:>6}  N/N0={ratio_n:>5.1f}x  "
                  f"c/c0={ratio_c:>5.2f}x  "
                  f"({'超線形' if ratio_c > 1.15 else '線形前後' if ratio_c > 0.85 else '劣線形'})")


def print_hotspots(rows: list[dict]) -> None:
    print("\n=== cProfile ホットスポット(cumtime 上位)===")
    hdr = f"{'#':>3} {'cumtime_s':>10} {'tottime_s':>10} {'ncalls':>10}  func"
    print(hdr)
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        print(f"{i:>3} {r['cumtime_s']:>10.4f} {r['tottime_s']:>10.4f} "
              f"{r['ncalls']:>10}  {r['func']}")


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description="W2 P0 非LLMエンジン単価 c の実測ベースライン")
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 3000],
                    help="規模 N のリスト(既定 300 1000 3000)")
    ap.add_argument("--steps", type=int, default=24, help="1 ラン当たり step 数(既定 24)")
    ap.add_argument("--warmup", type=int, default=2, help="計時除外する先頭 step 数(既定 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile-n", type=int, default=1000, help="cProfile 対象の N(既定 1000)")
    ap.add_argument("--profile-steps", type=int, default=24, help="cProfile 対象の step(既定 24)")
    ap.add_argument("--topk", type=int, default=30, help="ホットスポット上位数(既定 30)")
    ap.add_argument("--no-profile", action="store_true", help="cProfile を省く(c 実測のみ)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON 出力先(既定 runs/_profile/engine_baseline.json)")
    args = ap.parse_args(argv)

    out_path = args.out or (_ROOT / "runs" / "_profile" / "engine_baseline.json")
    out_path = out_path if out_path.is_absolute() else (_ROOT / out_path)

    print(f"profile: {_PROFILE}  backend=mock  "
          f"sizes={args.sizes}  steps={args.steps}  warmup={args.warmup}")

    results: list[dict] = []
    for n in args.sizes:
        print(f"\n[measure] N={n} × {args.steps}step (warmup {args.warmup}) ...", flush=True)
        r = measure_scale(n, args.steps, args.warmup, args.seed)
        results.append(r)
        print(f"  -> c={_fmt_c(r['c_s_per_agent_step'])} s/agent-step  "
              f"build={r['build_s']}s  総step={r['total_step_wall_s']}s  "
              f"ev/a·step={r['events_per_agent_step']}", flush=True)

    hotspots: list[dict] = []
    if not args.no_profile:
        print(f"\n[cProfile] N={args.profile_n} × {args.profile_steps}step ...", flush=True)
        hotspots = profile_hotspots(args.profile_n, args.profile_steps,
                                    args.warmup, args.seed, args.topk)

    print_scale_table(results)
    if hotspots:
        print_hotspots(hotspots)

    payload = {
        "meta": {
            "profile": "conf/production.yaml",
            "backend": "mock",
            "steps": args.steps,
            "warmup": args.warmup,
            "seed": args.seed,
            "sizes": args.sizes,
            "profile_n": args.profile_n,
            "profile_steps": args.profile_steps,
            "note": ("非LLM単価 c=秒/(N×計時step)。名簿100体を engine の agent_id 巡回で N 体へ"
                     "拡張(合成注入なし)。cProfile は c 計測とは別ラン。"),
        },
        "scales": results,
        "hotspots_cumtime": hotspots,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n-> JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
