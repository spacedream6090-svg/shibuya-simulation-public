#!/usr/bin/env python
"""思考リソースLOD 構成別 比較ハーネス(計算リソース × シミュ時間)。

ユーザー方針(2026-07-20): 「LOD は 1step ごとに層を移り変わる流動的なもの。背景に思考を
割かず前景に割きすぎるのは非現実的。全員にまんべんなくリソースを割く」。本ハーネスは
**思考タイミング系の各実装**(全員思考の drive ゲート + step 予算 + 行間 + N比例)を構成別に
mock で走らせ、壁時間・LLM 呼数(purpose 別)・**エージェント別 熟慮回数の分布(Gini + ヒスト)**
= まんべんなさの定量・イベント総数・**ピーク RSS(外部 PowerShell Get-Process WorkingSet64)**を
1 表に揃える。src/ は一切編集しない(Simulation を import して config 上書きで回すだけ)。

なぜ外部 PowerShell か:
  Windows では psutil 不使用・ctypes(GetProcessMemoryInfo)が黙って 0 を返す個体があるため、
  ピーク RSS は「子プロセスで 1 構成を走らせ、親が Get-Process -Id <pid> の WorkingSet64 を
  ポーリングして最大値を採る」方式で測る。子=シミュ本体、親=サンプラ(別コア)。

使い方:
  # 既定マトリクス(8 構成)を n=1000 × 144step で直列比較(RSS つき)
  python scripts/bench_lod.py --agents 1000 --steps 144
  # 軽量スモーク(RSS なし・in-process)
  python scripts/bench_lod.py --agents 40 --steps 24 --configs cap300 full --no-rss
  # 構成の限定
  python scripts/bench_lod.py --configs cap300 full dens0.30 s7

出力: runs/_bench_lod/comparison.json + コンソール markdown 表。
計測の約束:
  * 直列実行のみ(並列にしない=計測が汚れる)。同一 seed を全構成で共有。
  * 子プロセスは PYTHONIOENCODING=utf-8 固定・ログは encoding="utf-8", errors="replace"。
  * wall はシミュ本体(Simulation 構築 + run)= time.perf_counter。python 起動/import は除く。
  * 熟慮回数 = L1b(llm_calls)の agent_id 別件数(= その個体が実際に上位 LOD=思考へ入った回数)。
    分布は在場全個体(思考 0 回を含む)で採り、Gini↓=まんべんなさ↑。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:   # 同ディレクトリの run_dt を import
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_dt                                    # noqa: E402  (W2-3: Δt の単一の源)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# 構成マトリクス(思考タイミング系のみ。実験用 LOD は s7 参考のみ)
# =============================================================================
# 各構成 = ベース(mock・cache off・同一 seed)への config 上書きリスト。
# ラベルは日本語(報告書表の行名)。density 系は行間 OFF にして密度単独の効果を切り分ける。
def build_matrix() -> dict[str, dict]:
    return {
        # (b) 旧来ベースライン: 固定 cap 300 のみ(N 比例なし・行間なし)= 既定 config 相当
        "cap300": {
            "label": "固定cap300(旧来)",
            "overrides": ["lod.max_llm_per_step=300"],
        },
        # (a) 現行フル: 全員思考 + 行間(interstitial) + N 比例 density0.15
        "full": {
            "label": "現行フル(全員思考+行間+N比例0.15)",
            "overrides": [
                "lod.n_proportional.enabled=true",
                "lod.n_proportional.density=0.15",
                "prompts.interstitial.enabled=true",
            ],
        },
        # (c) 密度掃引(行間 OFF で密度単独)
        "dens0.05": {
            "label": "N比例 density0.05",
            "overrides": ["lod.n_proportional.enabled=true",
                          "lod.n_proportional.density=0.05"],
        },
        "dens0.15": {
            "label": "N比例 density0.15",
            "overrides": ["lod.n_proportional.enabled=true",
                          "lod.n_proportional.density=0.15"],
        },
        "dens0.30": {
            "label": "N比例 density0.30",
            "overrides": ["lod.n_proportional.enabled=true",
                          "lod.n_proportional.density=0.30"],
        },
        # (d) drive ゲート感度(decay で申請プールを増減。budget は cap300 固定)。
        #     drive_hi=減衰遅い→ゲージ高止まり→申請多い / drive_lo=減衰速い→申請少ない。
        #     ★閾値そのもの(drive_threshold)は個体の trait 由来で R1 上グローバル・ノブに
        #       しない(生得性の裏口回避)。ここは状態量=ゲージ側の decay を振る代理。
        "drive_hi": {
            "label": "drive高感度(decay0.01)",
            "overrides": ["lod.max_llm_per_step=300", "drive.decay=0.01"],
        },
        "drive_lo": {
            "label": "drive低感度(decay0.05)",
            "overrides": ["lod.max_llm_per_step=300", "drive.decay=0.05"],
        },
        # (e) S7 方針キャッシュ ON(参考=実験用 LOD。本番は既定 OFF 運用)。full と同条件 + cache。
        "s7": {
            "label": "S7方針キャッシュON(参考)",
            "overrides": [
                "lod.n_proportional.enabled=true",
                "lod.n_proportional.density=0.15",
                "prompts.interstitial.enabled=true",
                "cognition.policy_cache.enabled=true",
            ],
        },
    }


# =============================================================================
# 純関数(テスト対象): 分布 → Gini / ヒスト / 要約
# =============================================================================
def _gini(values: list[float]) -> float:
    """非負値列(思考 0 回を含む)の Gini 係数。全 0 or 全等値 → 0.0、極端不均等 → (n-1)/n。

    ソート済みの O(n log n) 閉形式: G = (2·Σ i·x_i)/(n·Σx) − (n+1)/n(x 昇順・i は 1..n)。
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total <= 0.0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _percentile(xs: list[float], p: float):
    """最近傍順位法(pandas 不使用・純 Python)。空なら None。"""
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


# 熟慮回数のヒスト・バケット(境界は含む側=左閉右閉の整数域)
_HIST_BUCKETS = [
    ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
    ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, math.inf),
]


def _histogram(counts: list[int]) -> dict[str, int]:
    """個体別 熟慮回数 → バケット別 人数。ユーザーの「まんべんなく」を目で見るための粗ヒスト。"""
    out = {name: 0 for name, _lo, _hi in _HIST_BUCKETS}
    for c in counts:
        for name, lo, hi in _HIST_BUCKETS:
            if lo <= c <= hi:
                out[name] += 1
                break
    return out


def _summ_dist(counts: list[int]) -> dict:
    """個体別 熟慮回数(在場全個体・0 を含む)の分布要約。gini↓=まんべんなさ↑。"""
    n = len(counts)
    total = sum(counts)
    nonzero = sum(1 for c in counts if c > 0)
    return {
        "n_agents": n,
        "total_thinks": total,
        "mean": round(total / n, 3) if n else None,
        "median": _percentile(counts, 50),
        "p90": _percentile(counts, 90),
        "p99": _percentile(counts, 99),
        "max": max(counts) if counts else None,
        # coverage=1度でも思考した個体の割合(「背景の誰か」が完全に無視されていないか)
        "coverage": round(nonzero / n, 4) if n else None,
        "zeros": n - nonzero,
        "gini": round(_gini(counts), 4),
        "histogram": _histogram(counts),
    }


# =============================================================================
# 1 構成の実走(in-process。子プロセスからも親の in-process 経路からも呼ぶ)
# =============================================================================
def run_one(name: str, overrides: list[str], *, agents: int, steps: int,
            seed: int, out_root: Path) -> dict:
    """1 構成を mock で実走し、計測 dict を返す(RSS は含まない=親が付ける)。

    - wall_build = Simulation(cfg) 構築秒 / wall_run = sim.run() 秒 / wall_s = 合算。
    - 熟慮回数分布 = L1b(sim.logger.llm_calls)の agent_id 別件数を在場全個体で採る。
    - LLM 呼数(purpose 別)= 同 L1b を purpose で集計。
    """
    from society.config import load_config
    from society.engine.simulation import Simulation

    run_dir = out_root / "runs" / name
    ov = [
        f"run.n_agents={agents}", f"run.n_steps={steps}", f"run.seed={seed}",
        f"run.name={name}", f"run.out_dir={run_dir.as_posix()}",
        "model.backend=mock", "model.cache=false",
    ] + list(overrides)
    cfg = load_config(overrides=ov)

    t0 = time.perf_counter()
    sim = Simulation(cfg)
    wall_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    summary = sim.run()
    wall_run = time.perf_counter() - t1

    # --- 個体別 熟慮回数(= 上位 LOD へ入った回数)。在場全個体(0 を含む)で分布を採る ---
    per_agent = collections.Counter()
    by_purpose = collections.Counter()
    for row in sim.logger.llm_calls:
        aid = row.get("agent_id")
        if aid is not None and aid >= 0:
            per_agent[aid] += 1
        by_purpose[row.get("purpose")] += 1
    universe = [a.id for a in sim.agents]           # pool OFF=在場全個体=全 N
    counts = [per_agent.get(aid, 0) for aid in universe]

    # drive ゲートの申請/発火(ゲート単独の観察。planning/reflect を除いた「熟慮」の芯)
    granted = fires = requests = 0
    for e in sim.logger.events:
        if e.kind == "drive_request":
            requests += 1
            if e.payload.get("granted"):
                granted += 1
    fires = granted

    return {
        "name": name,
        "agents": agents,
        "steps": steps,
        "seed": seed,
        "wall_build_s": round(wall_build, 4),
        "wall_run_s": round(wall_run, 4),
        "wall_s": round(wall_build + wall_run, 4),
        "ms_per_step": round(1000.0 * wall_run / steps, 3) if steps else None,
        "ms_per_agent_step": (round(1000.0 * wall_run / (steps * agents), 5)
                              if steps and agents else None),
        "llm_calls": int(summary.get("llm_calls", 0)),
        "llm_per_step": round(int(summary.get("llm_calls", 0)) / steps, 2) if steps else None,
        "llm_by_purpose": dict(sorted(by_purpose.items(),
                                      key=lambda kv: -kv[1])),
        "budget_max_per_step": int(getattr(sim.budget, "max_per_step", -1)),
        "drive_requests": requests,
        "drive_fires": fires,
        "n_events": int(summary.get("n_events", 0)),
        "events_per_step": round(int(summary.get("n_events", 0)) / steps, 1) if steps else None,
        "think_dist": _summ_dist(counts),
    }


def run_matrix_inproc(names: list[str], *, agents: int, steps: int, seed: int,
                      out_root: Path) -> dict:
    """複数構成を in-process で直列実走(RSS なし)。テスト・軽量比較用。"""
    matrix = build_matrix()
    rows = []
    for nm in names:
        spec = matrix[nm]
        r = run_one(nm, spec["overrides"], agents=agents, steps=steps,
                    seed=seed, out_root=out_root)
        r["label"] = spec["label"]
        rows.append(r)
    return _payload(rows, agents=agents, steps=steps, seed=seed, rss="none")


def _payload(rows: list[dict], *, agents: int, steps: int, seed: int,
             rss: str) -> dict:
    return {
        "meta": {
            "harness": "bench_lod",
            "backend": "mock", "cache": False,
            "agents": agents, "steps": steps, "seed": seed,
            "rss_method": rss,
            "note": ("同一 seed 全構成共有・直列実行。熟慮回数=L1b(llm_calls) agent_id 別件数を"
                     "在場全個体(思考0回含む)で分布化。gini↓=まんべんなさ↑。"
                     "peak_rss=子プロセスの WorkingSet64 を親が Get-Process でポーリングした最大値。"
                     "wall=Simulation 構築+run(python 起動/import は除外)。"),
        },
        "rows": rows,
    }


# =============================================================================
# 2 markdown 整形
# =============================================================================
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def comparison_markdown(rows: list[dict]) -> str:
    cols = [
        ("label", "構成"), ("budget_max_per_step", "budget/step"),
        ("wall_run_s", "run(s)"), ("ms_per_step", "ms/step"),
        ("llm_calls", "LLM呼"), ("llm_per_step", "LLM/step"),
        ("peak_rss_mb", "peakRSS(MB)"), ("n_events", "events"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def distribution_markdown(rows: list[dict]) -> str:
    cols = [
        ("label", "構成"), ("mean", "平均"), ("median", "中央"),
        ("p90", "p90"), ("p99", "p99"), ("max", "最大"),
        ("coverage", "coverage"), ("zeros", "思考0の人数"), ("gini", "Gini"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = []
    for r in rows:
        d = r.get("think_dist", {})
        vals = {"label": r.get("label")}
        vals.update({k: d.get(k) for k, _ in cols if k != "label"})
        body.append("| " + " | ".join(_fmt(vals.get(k)) for k, _ in cols) + " |")
    return "\n".join([head, sep, *body])


# =============================================================================
# 3 子プロセス実行 + 外部 PowerShell による ピーク RSS サンプリング
# =============================================================================
def _ps_workingset(pid: int) -> int | None:
    """PowerShell Get-Process -Id <pid> の WorkingSet64(バイト)。取得不可なら None。"""
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=15)
    except Exception:
        return None
    s = (p.stdout or "").strip()
    if not s:
        return None
    try:
        return int(s.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def run_config_subprocess(name: str, spec: dict, *, agents: int, steps: int,
                          seed: int, out_root: Path, sample_s: float) -> dict:
    """1 構成を子プロセスで実走し、親が WorkingSet64 をポーリングしてピーク RSS を付ける。"""
    spec_path = out_root / f"_spec_{name}.json"
    res_path = out_root / f"_result_{name}.json"
    log_path = out_root / f"_log_{name}.txt"
    if res_path.exists():
        res_path.unlink()
    spec_path.write_text(json.dumps({
        "name": name, "overrides": spec["overrides"], "label": spec["label"],
        "agents": agents, "steps": steps, "seed": seed,
        "out_root": str(out_root), "result": str(res_path),
    }, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)]
    print(f"[bench_lod] {name}: 子プロセス起動(n={agents} steps={steps}) ...",
          file=sys.stderr, flush=True)
    peak = 0
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf)
        while proc.poll() is None:
            ws = _ps_workingset(proc.pid)
            if ws:
                peak = max(peak, ws)
            time.sleep(sample_s)
        # 終了直前/直後の最後の 1 サンプル(単調増加のピークは終盤にある)
        ws = _ps_workingset(proc.pid)
        if ws:
            peak = max(peak, ws)
    child_wall = time.perf_counter() - t0

    if not res_path.exists():
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            pass
        raise RuntimeError(f"構成 {name} の子プロセスが結果を残さず終了(rc={proc.returncode})。"
                           f"\n--- log tail ---\n{tail}")
    r = json.loads(res_path.read_text(encoding="utf-8"))
    r["label"] = spec["label"]
    r["peak_rss_mb"] = round(peak / (1024 * 1024), 2) if peak else None
    r["child_wall_s"] = round(child_wall, 3)     # 参考: python 起動/import 込みの子プロセス寿命
    return r


def _worker_main(spec_path: str) -> None:
    """子プロセス本体: spec を読み、1 構成を実走し、結果 JSON を書く。"""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out_root = Path(spec["out_root"])
    r = run_one(spec["name"], spec["overrides"], agents=int(spec["agents"]),
                steps=int(spec["steps"]), seed=int(spec["seed"]), out_root=out_root)
    Path(spec["result"]).write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[worker] {spec['name']} done: wall_run={r['wall_run_s']}s "
          f"llm={r['llm_calls']} gini={r['think_dist']['gini']}", flush=True)


# =============================================================================
# エントリポイント
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="思考リソースLOD 構成別 比較ハーネス")
    ap.add_argument("--worker", default=None,
                    help="(内部)子プロセス実行: 指定 spec JSON の 1 構成を走らせ結果を書く")
    ap.add_argument("--agents", type=int, default=1000)
    # W2-3: 既定は「1 シミュ日」= 基底 conf の run.dt_min 依存(Δt=10 で 144)。
    ap.add_argument("--steps", type=int, default=None,
                    help="1 構成の step 数(既定=1 シミュ日ぶん。Δt=10 で 144)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="対象構成名(既定=全マトリクス)。例: --configs cap300 full dens0.30")
    ap.add_argument("--out", default="runs/_bench_lod")
    ap.add_argument("--sample-s", type=float, default=0.5,
                    help="ピーク RSS の PowerShell ポーリング間隔秒(既定0.5)")
    ap.add_argument("--no-rss", action="store_true",
                    help="子プロセス/RSS を使わず in-process で走らせる(軽量・RSS なし)")
    args = ap.parse_args()
    if args.steps is None:
        args.steps = run_dt.steps_per_day(REPO_ROOT / "conf")

    if args.worker:
        _worker_main(args.worker)
        return

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    names = args.configs or list(matrix.keys())
    unknown = [n for n in names if n not in matrix]
    if unknown:
        raise SystemExit(f"未知の構成: {unknown}。既定: {list(matrix.keys())}")

    rows: list[dict] = []
    for nm in names:
        if args.no_rss:
            r = run_one(nm, matrix[nm]["overrides"], agents=args.agents,
                        steps=args.steps, seed=args.seed, out_root=out_root)
            r["label"] = matrix[nm]["label"]
            r["peak_rss_mb"] = None
            print(f"[bench_lod] {nm}: run={r['wall_run_s']}s llm={r['llm_calls']} "
                  f"gini={r['think_dist']['gini']}", file=sys.stderr, flush=True)
        else:
            r = run_config_subprocess(nm, matrix[nm], agents=args.agents,
                                      steps=args.steps, seed=args.seed,
                                      out_root=out_root, sample_s=args.sample_s)
            print(f"[bench_lod] {nm}: run={r['wall_run_s']}s llm={r['llm_calls']} "
                  f"peakRSS={r.get('peak_rss_mb')}MB gini={r['think_dist']['gini']}",
                  file=sys.stderr, flush=True)
        rows.append(r)

    payload = _payload(rows, agents=args.agents, steps=args.steps, seed=args.seed,
                       rss=("none" if args.no_rss else "powershell_workingset64"))
    (out_root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n### 計算リソース × シミュ時間\n")
    print(comparison_markdown(rows))
    print("\n### 熟慮回数の分布(まんべんなさ = Gini↓)\n")
    print(distribution_markdown(rows))
    print(f"\n[bench_lod] wrote {out_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
