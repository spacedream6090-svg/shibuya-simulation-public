"""k 掃引ランナー(パイロット/本番共用、直列実行)。

単一 N(従来):
    python scripts/run_sweep.py --modes off sham free --seeds 1 2 3 4 5 \
        --agents 60 --steps 144 --prefix pilot \
        --extra agents.personas_file=data/personas_60.json
→ runs/pilot_<mode>_s<seed>/ が modes×seeds 本でき、analyze_sweep.py で横断分析する。

複数 N(有限サイズスケーリング, FSS):
    python scripts/run_sweep.py --modes off free --seeds 1 2 --agents 10 20 40 \
        --alphas 0.25 0.5 0.75 --steps 144 --prefix fss
→ runs/fss_N<n>_<mode>_s<seed>/。--agents が 2 水準以上なら N トークンを名前に挟む
  (単一 N のときは従来名のまま=後方互換)。data/personas_<N>.json と
  data/icebreak_<N>.json が存在すれば自動注入(--no-auto-data で無効)。

k の擬似連続軸:
  --alphas 0.25 0.5 0.75 を渡すと k.writeback=degraded × degraded_alpha=α のランを
  追加する(モード名 dg0.25 等)。off=0.0 / free=1.0 と合わせて連続的な k 軸を作る。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.config import load_config  # noqa: E402
from society.engine.simulation import Simulation  # noqa: E402


def _fmt_alpha(a: float) -> str:
    """モード名/ラン名に使う短いα表現(0.5→"0.5", 0.25→"0.25")。"""
    return "%g" % float(a)


def _build_conditions(modes, alphas):
    """(mode_label, writeback, alpha) の実行条件リストを組む。

    modes は off/sham/free/degraded の生モード。alphas は degraded の擬似連続点で、
    それぞれ (dg<α>, "degraded", α) の条件を追加する。
    """
    conds = [(m, m, None) for m in modes]
    for a in alphas or []:
        conds.append((f"dg{_fmt_alpha(a)}", "degraded", float(a)))
    return conds


def _auto_data_overrides(n: int, disabled: bool) -> list[str]:
    """data/personas_<N>.json / data/icebreak_<N>.json が在れば注入する上書き列。"""
    if disabled:
        return []
    ov: list[str] = []
    if (REPO_ROOT / "data" / f"personas_{n}.json").is_file():
        ov.append(f"agents.personas_file=data/personas_{n}.json")
    if (REPO_ROOT / "data" / f"icebreak_{n}.json").is_file():
        ov.append(f"agents.icebreak_file=data/icebreak_{n}.json")
    return ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["off", "sham", "free"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--agents", nargs="+", type=int, default=[60],
                    help="1つなら従来名、複数なら FSS(名前に N トークン)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[],
                    help="degraded の擬似連続点(k の中間軸)。例: 0.25 0.5 0.75")
    ap.add_argument("--steps", type=int, default=144)
    ap.add_argument("--prefix", default="pilot")
    ap.add_argument("--no-auto-data", action="store_true",
                    help="data/personas_<N>.json 等の自動注入を無効化")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    conditions = _build_conditions(args.modes, args.alphas)
    multi_n = len(args.agents) > 1
    total = len(args.agents) * len(conditions) * len(args.seeds)
    done = 0
    for n in args.agents:
        auto = _auto_data_overrides(n, args.no_auto_data)
        for label, writeback, alpha in conditions:
            for seed in args.seeds:
                # 単一 N は従来名(後方互換)、複数 N は N トークンを挟む
                name = (f"{args.prefix}_N{n}_{label}_s{seed}" if multi_n
                        else f"{args.prefix}_{label}_s{seed}")
                # 掃引は常にクリーン実行(旧 llm_cache がコード変更前の応答を再生し
                # 条件差を消してしまう事故を防ぐ)
                shutil.rmtree(REPO_ROOT / "runs" / name, ignore_errors=True)
                t0 = time.time()
                overrides = [
                    f"k.writeback={writeback}", f"run.seed={seed}",
                    f"run.n_agents={n}", f"run.n_steps={args.steps}",
                    f"run.name={name}"]
                if alpha is not None:
                    overrides.append(f"k.degraded_alpha={alpha}")
                # 自動注入 → --extra の順(--extra が最終権限=手動指定が勝つ)
                overrides += auto + list(args.extra)
                cfg = load_config(overrides)
                summary = Simulation(cfg).run()
                done += 1
                print(f"[{done}/{total}] {name}: {summary['n_events']}ev "
                      f"llm={summary['llm_calls']} "
                      f"adopt={summary['total_adoptions']} "
                      f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
