#!/usr/bin/env python
"""第57バッチ タスクC: 長期日常ラン(30日級)の見積もりベンチ。

短い mock ラン(既定 1日・3日)の実測から線形モデル(y = a + b·steps)を作り、7日・30日の
**実行時間 / L1 行数 / L1 parquet サイズ / ピーク RAM** を外挿する。さらに **mock 7日ラン1本**
(人数控えめ=既定 60人)を実際に回して外挿の精度を検証し、結果を docs/research/longrun-estimate.md に
記録する。30日フルはここでは回さない(必要なら --run-30d で任意)。

    PYTHONIOENCODING=utf-8 python scripts/bench_longrun.py
    PYTHONIOENCODING=utf-8 python scripts/bench_longrun.py --agents 60 --profile conf/longrun30.yaml

鉄則(R1): 実 LLM 禁止(model.backend=mock 固定)・pandas 不使用・UTF-8・フォアグラウンド完結。
測定は決定論(seed 固定)。各ランは**フレッシュな子プロセス**で回し、ピーク RAM = そのプロセスの
生涯ピーク working set(Windows GetProcessMemoryInfo / POSIX getrusage)。tracemalloc は毎アロケーションを
追跡して実時間を数倍に歪めるため**使わない**(過去バグ=第57バッチで是正)。RAM は pyarrow/native を含む
実 RSS のピーク。checkpoint_every / flush_every_steps で上限を切れる量(イベントバッファが支配項)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STEPS_PER_DAY = 144


def _peak_working_set() -> int:
    """このプロセスの生涯ピーク working set(bytes)を安価に取得する。
    Windows=GetProcessMemoryInfo(PeakWorkingSetSize)/ POSIX=resource.getrusage(ru_maxrss)。
    毎アロケーションを追跡する tracemalloc と違いオーバーヘッドゼロ=実時間を歪めない。取得不可は 0。"""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k = ctypes.windll.kernel32
            ps = ctypes.windll.psapi
            k.GetCurrentProcess.restype = wintypes.HANDLE            # 64bit ハンドル切り詰め回避
            ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                ctypes.POINTER(PMC), wintypes.DWORD]
            ps.GetProcessMemoryInfo.restype = wintypes.BOOL
            c = PMC()
            c.cb = ctypes.sizeof(c)
            if ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return int(c.PeakWorkingSetSize)
        else:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux=KB, macOS=bytes(近似のため KB 系を bytes へ)
            return int(ru) * (1024 if ru < 10**9 else 1)
    except Exception:
        return 0
    return 0


def _single_run(profile: str, agents: int, steps: int, seed: int,
                out_dir: Path) -> dict:
    """フレッシュな**子プロセス内**で 1 本回すための本体(ピーク working set=この run の峰)。"""
    from society.config import load_config
    from society.engine.simulation import Simulation
    shutil.rmtree(out_dir, ignore_errors=True)
    ov = [f"run.seed={seed}", f"run.n_agents={agents}", f"run.n_steps={steps}",
          "model.backend=mock", "run.seed_auto=false",
          "observer.checkpoint_every=0", f"run.name={out_dir.name}"]
    cfg = load_config(overrides=ov, profile=profile)
    t0 = time.perf_counter()
    summary = Simulation(cfg, out_dir=out_dir).run()
    dt = time.perf_counter() - t0
    peak = _peak_working_set()
    l1 = out_dir / "l1_events.parquet"
    l1_bytes = l1.stat().st_size if l1.exists() else 0
    total_bytes = sum(p.stat().st_size for p in out_dir.glob("*.parquet"))
    return {"agents": agents, "steps": steps, "days": steps / STEPS_PER_DAY,
            "sec": dt, "events": summary["n_events"], "l1_bytes": l1_bytes,
            "all_parquet_bytes": total_bytes, "peak_ram_bytes": peak}


def _run_once(profile: str, agents: int, steps: int, seed: int,
              out_dir: Path) -> dict:
    """1 本の mock ランを**フレッシュな子プロセス**で回し実測を返す(親のピーク RAM 累積を避ける=
    各 run のピーク working set を独立に測る。tracemalloc 不使用=実時間を歪めない)。"""
    cmd = [sys.executable, os.path.abspath(__file__), "--_single",
           "--profile", profile, "--agents", str(agents), "--steps", str(steps),
           "--seed", str(seed), "--outdir", str(out_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"child run failed (agents={agents} steps={steps}):\n{r.stderr[-2000:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("__BENCH_JSON__")]
    if not line:
        raise RuntimeError(f"child produced no metrics:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}")
    return json.loads(line[-1][len("__BENCH_JSON__"):])


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乗の直線 y = a + b·x(2点なら厳密・3点以上なら回帰)。"""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def _extrap(a: float, b: float, steps: int) -> float:
    return a + b * steps


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _fmt_sec(s: float) -> str:
    if s < 90:
        return f"{s:.1f}s"
    if s < 5400:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def _dispatch_single() -> bool:
    """--_single(子プロセス)モードなら 1 本回して __BENCH_JSON__ を出力し True を返す。"""
    if "--_single" not in sys.argv:
        return False
    ap = argparse.ArgumentParser()
    ap.add_argument("--_single", action="store_true", dest="single")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    res = _single_run(a.profile, a.agents, a.steps, a.seed, Path(a.outdir))
    print("__BENCH_JSON__" + json.dumps(res, ensure_ascii=False))
    return True


def main() -> None:
    if _dispatch_single():
        return
    ap = argparse.ArgumentParser(description="長期ラン見積もりベンチ(第57バッチ タスクC)")
    ap.add_argument("--profile", default="conf/longrun30.yaml")
    ap.add_argument("--agents", type=int, default=60, help="probe/検証の人数(控えめ)")
    ap.add_argument("--probe-days", default="1,3", help="外挿の素にする probe 日数(カンマ区切り)")
    ap.add_argument("--validate-days", type=int, default=7, help="外挿を検証する実測ラン日数")
    ap.add_argument("--target-days", type=int, default=30, help="外挿先の日数")
    ap.add_argument("--target-agents", type=int, default=100,
                    help="外挿先の人数(= profile 既定 n_agents)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/research/longrun-estimate.md")
    ap.add_argument("--run-30d", action="store_true",
                    help="(任意・重い)30日フルも実測して外挿と比較する")
    ap.add_argument("--keep", action="store_true", help="ラン出力を消さない")
    args = ap.parse_args()

    profile = args.profile if os.path.isabs(args.profile) \
        else os.path.join(_ROOT, args.profile)
    bench_root = Path(_ROOT) / "runs" / "_bench_longrun"
    bench_root.mkdir(parents=True, exist_ok=True)

    probe_days = [int(x) for x in args.probe_days.split(",") if x.strip()]
    print(f"[bench] profile={args.profile} agents={args.agents} "
          f"probes={probe_days}日 validate={args.validate_days}日 "
          f"target={args.target_days}日×{args.target_agents}人")

    # ---- 1) probe ラン(外挿の素)----
    probes = []
    for d in probe_days:
        steps = d * STEPS_PER_DAY
        r = _run_once(profile, args.agents, steps, args.seed,
                      bench_root / f"probe_{d}d")
        probes.append(r)
        print(f"  probe {d}日: {_fmt_sec(r['sec'])} / {r['events']:,}ev / "
              f"L1 {_fmt_bytes(r['l1_bytes'])} / peak {_fmt_bytes(r['peak_ram_bytes'])}")

    xs = [p["steps"] for p in probes]
    fits = {k: _linfit(xs, [p[k] for p in probes])
            for k in ("sec", "events", "l1_bytes", "all_parquet_bytes", "peak_ram_bytes")}

    # ---- 2) 検証ラン(7日)実測 vs 外挿 ----
    vsteps = args.validate_days * STEPS_PER_DAY
    val = _run_once(profile, args.agents, vsteps, args.seed,
                    bench_root / f"validate_{args.validate_days}d")
    print(f"  validate {args.validate_days}日(実測): {_fmt_sec(val['sec'])} / "
          f"{val['events']:,}ev / L1 {_fmt_bytes(val['l1_bytes'])} / "
          f"peak {_fmt_bytes(val['peak_ram_bytes'])}")

    def _err(key: str) -> tuple[float, float]:
        pred = _extrap(*fits[key], vsteps)
        act = val[key]
        e = (pred - act) / act * 100 if act else 0.0
        return pred, e

    # ---- 3) 30日への外挿(人数スケール込み)----
    # probe/検証は args.agents 人。イベント量・時間・バイトは人数にほぼ比例するので、
    # 検証ラン(最長=最も安定)の per-agent-step 単価で target-agents へスケールし直す。
    va = val["agents"]; vst = val["steps"]
    per_as_sec = val["sec"] / (va * vst)
    per_ev_agentday = val["events"] / (va * val["days"])
    bytes_per_ev = val["l1_bytes"] / val["events"] if val["events"] else 0
    allbytes_per_ev = val["all_parquet_bytes"] / val["events"] if val["events"] else 0

    tgt_steps = args.target_days * STEPS_PER_DAY
    tgt_agents = args.target_agents
    est_events = per_ev_agentday * tgt_agents * args.target_days
    est = {
        "sec": per_as_sec * tgt_agents * tgt_steps,
        "events": est_events,
        "l1_bytes": bytes_per_ev * est_events,
        "all_parquet_bytes": allbytes_per_ev * est_events,
        # ピーク RAM: checkpoint_every=1440(10日) で ~10日分のイベントが上限。全量保持の下限見積り。
        "peak_ram_full": (val["peak_ram_bytes"] / (va * vst)) * tgt_agents * tgt_steps,
    }
    est["peak_ram_ckpt10d"] = (val["peak_ram_bytes"] / (va * vst)) * tgt_agents * (10 * STEPS_PER_DAY)

    run30 = None
    if args.run_30d:
        print("  [--run-30d] 30日フル実測(重い)…")
        run30 = _run_once(profile, tgt_agents, tgt_steps, args.seed,
                          bench_root / "full_30d")
        print(f"  30日(実測): {_fmt_sec(run30['sec'])} / {run30['events']:,}ev / "
              f"L1 {_fmt_bytes(run30['l1_bytes'])}")

    # ---- 4) レポート出力 ----
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    _write_report(out_path, args, probes, fits, val, _err, est, run30,
                  per_as_sec, per_ev_agentday, bytes_per_ev)
    print(f"[bench] -> {out_path}")

    if not args.keep:
        shutil.rmtree(bench_root, ignore_errors=True)


def _write_report(path, args, probes, fits, val, err_fn, est, run30,
                  per_as_sec, per_ev_agentday, bytes_per_ev) -> None:
    import datetime
    L: list[str] = []
    L.append("# 長期日常ラン(30日級)の実行見積もり — 第57バッチ タスクC\n")
    L.append(f"> 生成: `scripts/bench_longrun.py`(mock・seed={args.seed}・"
             f"profile={args.profile})。{datetime.date.today().isoformat()}。")
    L.append("> 方針正典: docs/closed-world-daily-observation.md §3 タスクC。実 LLM 非使用="
             "**mock エンジン単価**の見積り(実 LLM 時は per-step が LLM レイテンシ律速へ変わる=下記 §5)。\n")

    L.append("## 1. 実測(mock・人数 {}人)\n".format(args.agents))
    L.append("| ラン | 日数 | step | 実時間 | L1行数(events) | L1 parquet | 全parquet | ピークRAM(working set) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in probes + [val]:
        tag = "**検証**" if p is val else "probe"
        L.append(f"| {tag} | {p['days']:.0f} | {p['steps']} | {_fmt_sec(p['sec'])} | "
                 f"{p['events']:,} | {_fmt_bytes(p['l1_bytes'])} | "
                 f"{_fmt_bytes(p['all_parquet_bytes'])} | {_fmt_bytes(p['peak_ram_bytes'])} |")
    L.append("")

    L.append("## 2. 外挿式(probe {} からの線形モデル y = a + b·step)\n".format(
        "/".join(f"{p['days']:.0f}日" for p in probes)))
    labels = {"sec": "実時間[s]", "events": "L1行数", "l1_bytes": "L1 parquet[B]",
              "all_parquet_bytes": "全parquet[B]", "peak_ram_bytes": "ピークRAM[B]"}
    L.append("| 量 | a(切片) | b(step単価) |")
    L.append("|---|---|---|")
    for k, lab in labels.items():
        a, b = fits[k]
        L.append(f"| {lab} | {a:.4g} | {b:.4g} |")
    L.append("")

    L.append("## 3. 外挿の精度検証(7日 予測 vs 実測)\n")
    L.append("| 量 | 線形外挿の予測 | 実測 | 誤差 |")
    L.append("|---|---|---|---|")
    for k, lab in labels.items():
        pred, e = err_fn(k)
        act = val[k]
        pv = _fmt_sec(pred) if k == "sec" else (f"{pred:,.0f}" if k == "events" else _fmt_bytes(pred))
        av = _fmt_sec(act) if k == "sec" else (f"{act:,}" if k == "events" else _fmt_bytes(act))
        L.append(f"| {lab} | {pv} | {av} | {e:+.1f}% |")
    L.append("\n> 線形外挿は probe(短ラン)の固定コスト(init/1日目コールドスタート)を含むため、"
             "長い日数ほど誤差が縮む。以下の 30日外挿は**検証ラン(最長=最安定)の per-agent-step 単価**を用いる。\n")

    L.append("## 4. 30日ランの見積り(人数 {}人・{}step)\n".format(
        args.target_agents, args.target_days * STEPS_PER_DAY))
    L.append(f"- 単価(検証ラン由来): {per_as_sec * 1000:.4f} ms/agent-step ・ "
             f"{per_ev_agentday:.1f} events/agent-day ・ {bytes_per_ev:.1f} B/event(L1)")
    L.append("")
    L.append("| 量 | 30日×{}人 の外挿 |".format(args.target_agents))
    L.append("|---|---|")
    L.append(f"| 実時間(mock) | **{_fmt_sec(est['sec'])}** |")
    L.append(f"| L1行数(events) | {est['events']:,.0f} |")
    L.append(f"| L1 parquet | **{_fmt_bytes(est['l1_bytes'])}** |")
    L.append(f"| 全parquet(L1+L2+L3+L1b) | {_fmt_bytes(est['all_parquet_bytes'])} |")
    L.append(f"| ピークRAM(全量保持=checkpoint OFF の下限) | {_fmt_bytes(est['peak_ram_full'])} |")
    L.append(f"| ピークRAM(checkpoint_every=1440=10日ごと flush) | {_fmt_bytes(est['peak_ram_ckpt10d'])} |")
    L.append("")
    if run30 is not None:
        L.append("### 4b. 30日フル実測(--run-30d)\n")
        L.append(f"- 実時間 {_fmt_sec(run30['sec'])} / L1 {run30['events']:,}行 / "
                 f"{_fmt_bytes(run30['l1_bytes'])} / peak {_fmt_bytes(run30['peak_ram_bytes'])}")
        se = (est['sec'] - run30['sec']) / run30['sec'] * 100 if run30['sec'] else 0
        L.append(f"- 外挿 実時間 誤差: {se:+.1f}%\n")

    L.append("## 5. 本選 GPU 予算への含意\n")
    L.append("- 上表は **mock エンジン単価**。本選=実 LLM ではエンジン計算は LLM 呼(発火/計画/内省)の"
             "レイテンシに隠れるため、per-step 実時間は **req/s と在場数** で決まる(Day-0 ベンチ=finals-hardware-plan §2)。"
             "本ベンチは「エンジン+ロギング+観測レンズが 30日で破綻しないか(RAM/ストレージ/事後分析)」の確認に用いる。")
    L.append("- **ストレージ**: L1 parquet の 30日サイズが上表。lens ON(value/motive/trust/deviation/structure の"
             "L2 スカラー)は L2 に数列足すだけ=支配項は L1。全 parquet 見積りが本選ディスク予算(D8)の一材料。")
    L.append("- **RAM**: checkpoint_every=1440(10日)で L1 バッファは ~10日分に上限化(上表の 2 行目)。"
             "RAM が厳しければ flush_every_steps=288(2日)で更に圧縮できる(既定 0=従来と完全同一)。")
    L.append("- **分割実行**: 30日は 10日×3 に割って回せる(conf/longrun30.yaml のコマンド列)。resume は"
             "バイト一致(tests/test_resume + test_longrun)。夜間予算に合わせて区切っても straight と同一出力。")
    L.append("- **k\\* 測定との棲み分け**: 本 30日ランは構造創発の観察 1 条件(人数を絞る)。k\\* は 7–14日・"
             "人数多めで別に回す(二段構え)。D15(finals-day1-decisions)で本選確保を判断する。\n")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
