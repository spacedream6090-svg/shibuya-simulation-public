#!/usr/bin/env python
"""LLM 健全性の事後点検(P0バッチ 2026-07-29)。**L1/L1b/L2 を読むだけ**。

scripts/watchdog.py がラン**中**の警告(--fallback-warn)を担うのに対し、こちらは
終わった(あるいは走っている)ランを後から点検する。シミュ本体を import しないので
runs/ に残っている過去ランすべてに適用できる。

    python scripts/watchdog_llm.py runs/prod1
    python scripts/watchdog_llm.py "runs/endo8w_*" --fallback-warn 0.2 --json

出すもの
--------
- llm_calls        : l1b_llm.parquet の行数(= summary.json の llm_calls と一致するはず)
- cache_hit_rate   : l1b の cached=True の割合
- fallback_rate    : L1 kind="fallback" 件数 ÷ llm_calls
- purpose 内訳     : purpose ごとの呼数・キャッシュ率(plan_retry の多さ=計画の失敗の階段の指標)
- L2 由来の系列    : observer.llm_health.enabled=true のランは l2_metrics の最終行も併記して
                     突合できる(両者が食い違ったら計装のバグ)

正直な限界(observer/aggregate.py と同じ):
  L1 `fallback` イベントを出すのは発話系のパース失敗だけ。朝の計画は再試行
  (purpose="plan_retry")→前日流用→既定骨格の階段で沈黙して退行し、夜の内省も空応答で
  黙って退行する。したがって fallback_rate は**真のパース失敗率の下限**。plan_retry_rate を
  併記するのはその穴を部分的に埋めるため。

終了コード: 閾値超過のランが 1 つでもあれば 1(CI/運用スクリプトから使えるように)。
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _s in (sys.stdout, sys.stderr):              # Windows コンソール(cp932)対策
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _read_table(path: Path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def _l1b_stats(run_dir: Path) -> dict:
    """l1b_llm.parquet から呼数・キャッシュ率・purpose 内訳。無ければ空。"""
    path = run_dir / "l1b_llm.parquet"
    if not path.exists():
        return {"llm_calls": None, "cache_hits": None, "by_purpose": {}}
    tbl = _read_table(path).to_pylist()
    by_purpose: Counter = Counter()
    cached_by_purpose: Counter = Counter()
    cache_hits = 0
    for r in tbl:
        purpose = str(r.get("purpose") or "?")
        by_purpose[purpose] += 1
        if r.get("cached"):
            cache_hits += 1
            cached_by_purpose[purpose] += 1
    return {
        "llm_calls": len(tbl),
        "cache_hits": cache_hits,
        "by_purpose": {p: {"calls": n,
                           "cache_hit_rate": round(cached_by_purpose[p] / n, 4)}
                       for p, n in sorted(by_purpose.items(),
                                          key=lambda kv: -kv[1])},
    }


def _l1_fallbacks(run_dir: Path) -> int | None:
    """L1 の kind='fallback' 件数。summary.json の event_kinds があればそれを使う
    (巨大な L1 を読まずに済む)。無ければ parquet の kind 列だけを読む。"""
    summ = run_dir / "summary.json"
    if summ.exists():
        try:
            s = json.loads(summ.read_text(encoding="utf-8"))
            kinds = s.get("event_kinds")
            if isinstance(kinds, dict):
                return int(kinds.get("fallback", 0))
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    path = run_dir / "l1_events.parquet"
    if not path.exists():
        return None
    kinds = _read_table(path, columns=["kind"]).column("kind").to_pylist()
    return sum(1 for k in kinds if k == "fallback")


def _l2_last(run_dir: Path) -> dict | None:
    """observer.llm_health.enabled=true のランのみ: L2 最終行の 3 列。"""
    path = run_dir / "l2_metrics.parquet"
    if not path.exists():
        return None
    tbl = _read_table(path)
    cols = set(tbl.column_names)
    if not {"llm_calls_total", "llm_fallback_rate"} <= cols or tbl.num_rows == 0:
        return None
    last = tbl.slice(tbl.num_rows - 1, 1).to_pylist()[0]
    return {k: last.get(k) for k in
            ("step", "llm_calls_total", "llm_fallback_rate", "llm_cache_hit_rate")}


def check_run(run_dir: Path) -> dict:
    """1 ランの健全性レポート(純粋な読み取り。副作用なし)。"""
    stats = _l1b_stats(run_dir)
    calls = stats["llm_calls"]
    fallbacks = _l1_fallbacks(run_dir)
    plan_calls = stats["by_purpose"].get("plan", {}).get("calls", 0)
    plan_retry = stats["by_purpose"].get("plan_retry", {}).get("calls", 0)
    return {
        "run": run_dir.name,
        "llm_calls": calls,
        "fallbacks": fallbacks,
        "fallback_rate": (round(fallbacks / calls, 6)
                          if calls and fallbacks is not None else None),
        "cache_hit_rate": (round(stats["cache_hits"] / calls, 6)
                           if calls else None),
        # 朝の計画の「失敗の階段」の第1段(再試行)の割合 = fallback が拾えない失敗の代理
        "plan_retry_rate": (round(plan_retry / plan_calls, 6)
                            if plan_calls else None),
        "by_purpose": stats["by_purpose"],
        "l2_llm_health": _l2_last(run_dir),
    }


def _fmt(report: dict) -> str:
    def _p(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"
    lines = [
        f"## {report['run']}",
        f"  llm_calls      : {report['llm_calls']}",
        f"  fallback       : {report['fallbacks']} "
        f"(rate {_p(report['fallback_rate'])})",
        f"  cache_hit_rate : {_p(report['cache_hit_rate'])}",
        f"  plan_retry_rate: {_p(report['plan_retry_rate'])}",
    ]
    if report["l2_llm_health"]:
        h = report["l2_llm_health"]
        lines.append(f"  L2(最終step {h['step']}): calls={h['llm_calls_total']} "
                     f"fallback={_p(h['llm_fallback_rate'])} "
                     f"cache={_p(h['llm_cache_hit_rate'])}")
    top = list(report["by_purpose"].items())[:8]
    if top:
        lines.append("  purpose: " + ", ".join(
            f"{p}={d['calls']}" for p, d in top))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 健全性の事後点検(読み取り専用)")
    ap.add_argument("runs", nargs="+", help="run dir または glob(例 \"runs/endo8w_*\")")
    ap.add_argument("--fallback-warn", type=float, default=0.20,
                    help="この fallback_rate を超えたら警告して終了コード1(既定0.20)")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    args = ap.parse_args()

    dirs: list[Path] = []
    for pat in args.runs:
        p = Path(pat)
        if not p.is_absolute():
            p = REPO_ROOT / pat
        for hit in sorted(_glob.glob(str(p))):
            hp = Path(hit)
            if hp.is_dir():
                dirs.append(hp)
    if not dirs:
        print(f"[watchdog_llm] 対象 run が見つからない: {args.runs}", file=sys.stderr)
        raise SystemExit(2)

    reports = [check_run(d) for d in dirs]
    bad = [r for r in reports
           if r["fallback_rate"] is not None
           and r["fallback_rate"] > args.fallback_warn]
    if args.json:
        print(json.dumps({"reports": reports,
                          "fallback_warn": args.fallback_warn,
                          "over_threshold": [r["run"] for r in bad]},
                         ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(_fmt(r))
        for r in bad:
            print(f"[WARN] {r['run']}: fallback_rate="
                  f"{r['fallback_rate']:.4f} > {args.fallback_warn}", file=sys.stderr)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
