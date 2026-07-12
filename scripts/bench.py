#!/usr/bin/env python
"""スケーリング・ベンチ(mock ラン)。

    python scripts/bench.py                               # --agents 10 40 80 --steps 60 --seed 42
    python scripts/bench.py --agents 10 50 --steps 30

各規模で **Mock LLM** の決定論ランを回し、以下を計測して
runs/_bench/bench.json + コンソールに markdown 表を出す:
  * step あたり壁時計時間(time.perf_counter)
  * LLM 呼び出し数(summary["llm_calls"] 由来。mock も CachedLLM 経路を通る)
  * ピークメモリ(tracemalloc の Python ヒープ峰値。Windows なので resource は使わない)

src/ には一切変更を加えない — Simulation を import して config 上書きで回すだけ。
tracemalloc は Python 割り当てのみ計測し、pyarrow の C++ バッファは含まない点に注意
(相対比較・オーダー把握用。厳密 RSS は別途 OS ツールで)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Windows コンソール(cp932)で日本語パスや記号を print しても落ちないように
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from society.config import load_config              # noqa: E402
from society.engine.simulation import Simulation    # noqa: E402


def bench_one(n_agents: int, steps: int, seed: int, out_root: Path,
              backend: str = "mock", servers: list[str] | None = None) -> dict:
    """1 規模を計測。既定は mock backend・キャッシュ無効(純計算時間を測る)。

    backend='vllm' + servers=[...] を渡すと本選バックエンドへの疎通確認にも使える
    (例: --backend vllm --servers http://localhost:8000 --agents 2 --steps 2)。
    """
    overrides = [
        f"run.n_agents={n_agents}",
        f"run.n_steps={steps}",
        f"run.seed={seed}",
        f"run.name=n{n_agents}",
        f"run.out_dir={out_root.as_posix()}",
        f"model.backend={backend}",
        "model.cache=false",          # 純粋な生成時間を測る(リプレイ加速を除外)
    ]
    if servers:
        overrides.append(f"model.servers={json.dumps(list(servers))}")
    cfg = load_config(overrides=overrides)
    sim = Simulation(cfg)

    tracemalloc.start()
    t0 = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    llm_calls = int(summary.get("llm_calls", 0))
    n_events = int(summary.get("n_events", 0))
    return {
        "agents": n_agents,
        "steps": steps,
        "seed": seed,
        "wall_s": round(wall, 4),
        "ms_per_step": round(1000.0 * wall / steps, 3) if steps else None,
        "ms_per_agent_step": (round(1000.0 * wall / (steps * n_agents), 4)
                              if steps and n_agents else None),
        "llm_calls": llm_calls,
        "llm_per_step": round(llm_calls / steps, 2) if steps else None,
        "peak_mem_mb": round(peak / (1024 * 1024), 2),
        "n_events": n_events,
        "events_per_step": round(n_events / steps, 1) if steps else None,
    }


def _markdown_table(rows: list[dict]) -> str:
    cols = [
        ("agents", "agents"), ("steps", "steps"), ("wall_s", "wall(s)"),
        ("ms_per_step", "ms/step"), ("ms_per_agent_step", "ms/agent-step"),
        ("llm_calls", "llm_calls"), ("llm_per_step", "llm/step"),
        ("peak_mem_mb", "peak_mem(MB)"), ("n_events", "events"),
        ("events_per_step", "events/step"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaling bench (mock runs)")
    ap.add_argument("--agents", type=int, nargs="+", default=[10, 40, 80])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/_bench")
    ap.add_argument("--backend", default="mock",
                    help="mock | ollama | vllm(本選)。vllm は --servers 必須。")
    ap.add_argument("--servers", nargs="*", default=None,
                    help="vLLM の base_url(複数可)。疎通確認: --backend vllm --servers http://localhost:8000")
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in args.agents:
        print(f"[bench] running n_agents={n}, steps={args.steps}, "
              f"backend={args.backend} ...", file=sys.stderr, flush=True)
        rows.append(bench_one(n, args.steps, args.seed, out_root,
                              backend=args.backend, servers=args.servers))

    table = _markdown_table(rows)
    payload = {
        "meta": {
            "backend": args.backend, "cache": False, "steps": args.steps,
            "seed": args.seed, "agents": args.agents,
            "note": ("wall = time.perf_counter; peak_mem = tracemalloc Python heap "
                     "peak (excl. pyarrow C++ buffers); Windows なので resource 不使用。"),
        },
        "rows": rows,
    }
    (out_root / "bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + table + "\n")
    print(f"[bench] wrote {out_root / 'bench.json'}")


if __name__ == "__main__":
    main()
