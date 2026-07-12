#!/usr/bin/env python
"""SNS/DM 架橋距離の集計(第22バッチ P2・旧 shibuya-sim simulation.py:880-922 の思想を移植)。

    python scripts/analyze_bridging.py runs/<name> [--far-m 500]

sns_geo.enabled=true で回したランの transmission イベント(payload.dist_m)から、
「インターネット層(SNS/DM)が物理的な隔たりをどれだけ越えて情報を運んだか」を測る。
SNS 有無の A/B 対照で被説明変数に使う(docs/plans/legacy-adoption.md P2)。

**読み出し専用**: runs/<name>/l1_events.parquet を読むだけ。sim 本体・schema は変更しない。
dist_m は「読んだ瞬間の両者の物理距離」(地図座標系=m 相当)。対面(face)は共在が前提で
dist_m を持たない設計のため、距離統計は n/a と正直に出す。

出力: runs/<name>/analysis/bridging.json + コンソール要約
  - チャネル別: 件数・dist_m 記録率・平均/中央値/p90
  - far_frac: dist_m >= far-m(既定 500m)の割合(sns/dm 別)
  - 日次系列: 日(step//144)ごとの sns/dm 伝播数・far_frac
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEPS_PER_DAY = 144


def load_transmissions(run_dir: str) -> list[dict]:
    """l1_events.parquet から transmission 行のみ(payload は dict 化して返す)。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    rows = pq.read_table(path).to_pylist()
    out = []
    for e in rows:
        if e["kind"] != "transmission":
            continue
        p = e["payload"]
        p = json.loads(p) if isinstance(p, str) else (p or {})
        out.append({"step": int(e["step"]), "to": int(e["agent_id"]),
                    "from": p.get("from"), "channel": p.get("channel"),
                    "dist_m": p.get("dist_m")})
    return out


def _stats(dists: list[float]) -> dict:
    """平均/中央値/p90(空なら None=データ不足を正直に)。"""
    if not dists:
        return {"mean": None, "median": None, "p90": None}
    xs = sorted(dists)
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return {"mean": round(st.fmean(xs), 1), "median": round(st.median(xs), 1),
            "p90": round(p90, 1)}


def summarize(trans: list[dict], far_m: float) -> dict:
    by_ch: dict[str, list[dict]] = defaultdict(list)
    for t in trans:
        by_ch[str(t["channel"])].append(t)

    channels = {}
    for ch, rows in sorted(by_ch.items()):
        dists = [float(t["dist_m"]) for t in rows if t["dist_m"] is not None]
        entry = {"n": len(rows), "n_with_dist": len(dists), **_stats(dists)}
        if ch in ("sns", "dm"):
            entry["far_frac"] = (round(sum(d >= far_m for d in dists) / len(dists), 4)
                                 if dists else None)
        channels[ch] = entry

    daily: dict[int, dict] = {}
    for t in trans:
        if t["channel"] not in ("sns", "dm"):
            continue
        d = daily.setdefault(t["step"] // STEPS_PER_DAY,
                             {"n": 0, "n_far": 0, "n_with_dist": 0})
        d["n"] += 1
        if t["dist_m"] is not None:
            d["n_with_dist"] += 1
            d["n_far"] += int(float(t["dist_m"]) >= far_m)
    daily_series = [{"day": day, **v,
                     "far_frac": (round(v["n_far"] / v["n_with_dist"], 4)
                                  if v["n_with_dist"] else None)}
                    for day, v in sorted(daily.items())]

    return {"far_m": far_m, "n_transmissions": len(trans),
            "channels": channels, "daily": daily_series}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--far-m", type=float, default=500.0,
                    help="「遠距離」とみなす閾値(m 相当・既定 500)")
    args = ap.parse_args()

    trans = load_transmissions(args.run_dir)
    result = summarize(trans, args.far_m)

    out_dir = os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bridging.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[bridging] transmissions={result['n_transmissions']} far_m={args.far_m}")
    for ch, s in result["channels"].items():
        far = f" far_frac={s['far_frac']}" if "far_frac" in s else ""
        dist = (f" dist(mean/med/p90)={s['mean']}/{s['median']}/{s['p90']}"
                if s["n_with_dist"] else " dist=n/a")
        print(f"  {ch:8s} n={s['n']:6d} with_dist={s['n_with_dist']:6d}{dist}{far}")
    if not any(s["n_with_dist"] for s in result["channels"].values()):
        print("  ※ dist_m が 1 件もない: sns_geo.enabled=true で回したランか確認のこと")
    print(f"[bridging] -> {out_path}")


if __name__ == "__main__":
    main()
