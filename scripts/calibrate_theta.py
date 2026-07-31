#!/usr/bin/env python3
"""θ 較正パイロット(第83バッチ)= 発火閾値の**全体スケールだけ**を目標発火率に合わせる。

正典: docs/plans/source/cognition-design-record.md **§2.6**(共通スケールの較正)・
      **§3.5**(パイロット 1 日で発火数分布を実測し 10 日ぶんの総推論量を見積もる)・
      §7-7(θ 較正パイロット)/ physics-instructions.md Part P5。

設計 §2.6 の要求(逐語)
------------------------
「**共通スケールの較正**: パイロットランで『1 日あたり平均発火数が目標値』になるよう
  θ の**全体スケールのみ**を合わせる。**個体差は θ_i の分散として残す。**」

したがって本スクリプトが触るレバーは `cognition.fire.theta_scale`(**全文脈・全個体に
一様に掛かる 1 個のスカラ**)だけである。文脈別の水準比(較正テーブル `salience.<ctx>.theta`)
にも個体倍率(ペルソナ由来 θ₀ + 恒常性)にも**一切触らない**。個体差は潰さない。

    θ_i(ctx) = table.salience[ctx].theta × **theta_scale** × 個体倍率(ペルソナ/恒常性)
                                            ↑ ここだけを掃引する

制御量は「**驚き(salience)発火**の 1 日 1 人あたり件数」
-------------------------------------------------------
θ が門番をしているのは salience だけである(周期発火は較正テーブルの周期が決めており
θ では下げられない・social は既存機構が担う・internal はニーズ閾値の横断が決める)。
`plasticity.theta_target_per_day` = f* も同じ定義なので、恒常性と較正が同じ量を見る。

恒常性は較正中だけ切る(`theta_mu=0`)
--------------------------------------
θ の恒常性 `θ ← θ + μ(f̄ − f*)` は**負のフィードバック**なので、入れたまま掃引すると
scale を動かしても個体倍率が打ち消しに回り、**開ループ利得が測れない**。較正パイロットは
μ=0(開ループ)で回し、恒常性は本番ランで個体のドリフト対策として働かせる。
これは制御工学でも生理学でも標準的な分離である(set point と loop gain を別々に決める)。

実装前リサーチ(発火率較正の慣行)
----------------------------------
- **firing rate homeostasis**(Turrigiano ら)= 神経細胞は平均発火率を set point の周りに
  負のフィードバックで維持する。重要なのは「**set point の値そのものは恒常性機構からは
  導けない**」ことで、文献でも set point は**外から与える所与**として扱われる
  (Styr & Slutsky 2021, *Mitochondria: new players in homeostatic regulation of firing
  rate set points*, Trends Neurosci 44(6) / O'Leary et al. 2014 の恒常性モデル)。
  → 本リポジトリでも同じ立場を取る: **f\\* は測定で決まる量ではなく宣言する量**。
     本スクリプトは「f\\* を決める」のではなく「**宣言した f\\* に scale を合わせる**」だけ。
     f\\* 自体の根拠づけは conf の値と事前登録の領分であり、ここでは仮値として扱う。
- 人間側の接地は現状ない。経験サンプリング法(experience sampling)の思考プローブ研究は
  「1 日 N 回の思考」を数える設計になっておらず(プローブ提示のたびに注意状態を分類する
  設計。例: Smith et al. 2018, *Mind-wandering rates fluctuate across the day*,
  Conscious Cogn 66:11-16)、**1 日あたりの思考の切り替わり回数の実測値は本スクリプトの
  根拠にできない**。よって conf の `theta_target_per_day` は **provisional** のまま。

出力(2 つ)
-----------
1. `data/calib/theta_scale.json` — 凍結。scale・実測 f・生成条件・sha256・探索の全履歴。
2. **10 日ラン総推論量の見積表**(§3.5)。人数スケール別に LLM 呼数を外挿する。
   併せて「1 人あたり呼数が人数に対して不変か」を **N を変えた 2 本目のパイロット**で
   経験的に確かめる(密度依存のチャンネルがあるので、不変性は仮定してはいけない)。

★ 凍結値を conf へ書き戻さない(設計判断)
-----------------------------------------
`conf/cognition/calib_default.yaml` は **provisional 宣言のまま据え置く**。本スクリプトの
結果は独立した凍結ファイルに置き、**適用は明示的な conf 上書き**で行う:

    python -m society.run ... cognition.fire.theta_scale=<凍結値>

理由: (a) 較正テーブルは「人間の実測データで差し替える」ための器(P0(3))であって、
本装置の自己較正結果を混ぜると出自が判らなくなる。(b) 凍結ファイルをランタイムで
自動適用すると「同じ conf・同じ seed でも、ファイルの有無で世界が変わる」= ゴールデンと
再現性の前提が崩れる。(c) 適用値は既に `run_manifest.json` の `cognition.fire.theta_scale`
に載るので、事後にどの scale で走ったかは常に判る。→ **src の差分ゼロで来歴は完全**。

使い方
------
    python scripts/calibrate_theta.py                       # 既定: 100 体 × 1 日
    python scripts/calibrate_theta.py --target 4.0          # f* を変える
    python scripts/calibrate_theta.py --watch               # 監視仕様 ON で較正する
    python scripts/calibrate_theta.py --print-only          # 凍結せず表示だけ

**決定論**: 同じ引数なら**バイト同一**の JSON を出す(壁時計を書かない。`--stamp` のときだけ書く)。

正直な限界
----------
- **mock バックエンドのパイロット**である。実 LLM では発話量・watch 仕様の質が変わるので
  `ext.heard` 系の偏差が動き、同じ scale でも f はずれうる。
- `--watch`(監視仕様 ON)は f を**桁で**動かす。mock は毎回トリガを出すので
  Σ_j w_ij·[trigger_j] が S を支配し、驚き項の較正としては代表性がない。既定は OFF。
- f(scale) は単調減少**のはず**だが厳密には保証されない(scale を変えると誰が発火するかが
  変わり、世界そのものが分岐する)。グリッドの単調性を検査して結果に載せる。
- 見積表は「1 人あたり呼数が人数に対して不変」を仮定した外挿。仮定の当否を 2 点で
  実測して併記するが、2 点は 2 点でしかない(1 万体の外挿は 100 倍の外挿である)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.cognition import calib as CALIB          # noqa: E402
from society.cognition import fire as FIRE            # noqa: E402
from society.config import load_config                # noqa: E402
from society.engine.simulation import Simulation      # noqa: E402

SCHEMA = 1
KIND = "theta_scale"
DEFAULT_OUT = REPO_ROOT / "data" / "calib" / "theta_scale.json"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "_theta_pilot"
MINUTES_PER_DAY = 1440


# --------------------------------------------------------------------------- #
# パイロットラン(1 点の評価)
# --------------------------------------------------------------------------- #
def _clean(run_dir: Path, keep: bool) -> None:
    """安全弁つきの掃除(measure_sigma.py と同じ流儀)。"""
    if not run_dir.exists() or keep:
        return
    if any(run_dir.iterdir()) and not any(run_dir.glob("*/config.yaml")):
        raise SystemExit(
            f"--run-dir がラン出力ディレクトリに見えない: {run_dir}\n"
            "  別のパスを指すか --keep-run-dir を付けて再利用すること。")
    shutil.rmtree(run_dir)


def pilot(scale: float, *, agents: int, steps: int, seed: int, run_dir: Path,
          extra: list[str], watch: bool, tag: str) -> dict:
    """`theta_scale=scale` で mock パイロットを 1 本回し、発火の実測値を返す。

    ★ 掃引するのは `cognition.fire.theta_scale` **だけ**。文脈別の θ 水準比も
      個体倍率(ペルソナ由来 θ₀)も触らない = 設計 §2.6「個体差は分散として残す」。
    ★ `theta_mu=0` で恒常性を切る(開ループ利得を測るため。本文の説明を参照)。
    """
    out = Path(run_dir) / tag
    if out.exists():
        shutil.rmtree(out)
    dotlist = [
        f"run.seed={seed}", f"run.n_agents={agents}", f"run.n_steps={steps}",
        f"run.name=_theta_{tag}", "model.backend=mock",
        "cognition.fire.enabled=true",
        # g/θ 更新則は ON(個体倍率と可塑性は本番と同じ)だが恒常性だけ切る
        "cognition.g_update.enabled=true",
        "cognition.g_update.theta_mu=0.0",
        "cognition.g_update.log_every_steps=0",
        f"cognition.fire.theta_scale={scale!r}",
        *(["cognition.watch.enabled=true"] if watch else []),
        *extra,
    ]
    sim = Simulation(load_config(dotlist), out_dir=out)
    sim.run()
    return measure(out, agents=agents, steps=steps)


def measure(out: Path, *, agents: int, steps: int) -> dict:
    """ラン出力から発火の実測値を取り出す(**parquet だけを読む**= streaming でも同じ)。"""
    table = pq.read_table(Path(out) / "l1_events.parquet",
                          columns=["step", "sim_min", "agent_id", "kind", "payload"])
    kinds = table.column("kind").to_pylist()
    aids = table.column("agent_id").to_pylist()
    mins = table.column("sim_min").to_pylist()
    loads = table.column("payload").to_pylist()

    by_reason: dict[str, int] = {}
    by_ctx: dict[str, int] = {}
    sal_by_agent: dict[int, int] = {}
    all_by_agent: dict[int, int] = {}
    n_cog_event = 0
    span_min = 0
    for kind, aid, smin, blob in zip(kinds, aids, mins, loads):
        if kind == "cog_event":
            n_cog_event += 1
            continue
        if kind != "cog_fire":
            continue
        rec = json.loads(blob) if blob else {}
        reason = str(rec.get("reason", ""))
        by_reason[reason] = by_reason.get(reason, 0) + 1
        ctx = str(rec.get("ctx", ""))
        by_ctx[ctx] = by_ctx.get(ctx, 0) + 1
        all_by_agent[int(aid)] = all_by_agent.get(int(aid), 0) + 1
        if reason == FIRE.SALIENCE:
            sal_by_agent[int(aid)] = sal_by_agent.get(int(aid), 0) + 1
        span_min = max(span_min, int(smin))

    n_seen = len(all_by_agent) or int(agents)
    # 日数はシム内時間で数える(Δt を変えても同じ意味になる)。
    days = max(1e-9, (span_min + 1) / float(MINUTES_PER_DAY))
    summary = json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))
    n_sal = by_reason.get(FIRE.SALIENCE, 0)
    sal_counts = [sal_by_agent.get(a, 0) for a in sorted(all_by_agent)]
    return {
        "n_agents": n_seen, "n_steps": int(steps), "days": days,
        "n_cog_fire": sum(all_by_agent.values()), "n_cog_event": n_cog_event,
        "by_reason": dict(sorted(by_reason.items())),
        "by_context": dict(sorted(by_ctx.items())),
        "f_salience_per_agent_day": n_sal / (n_seen * days),
        "f_total_per_agent_day": sum(all_by_agent.values()) / (n_seen * days),
        "salience_per_agent": _dist(sal_counts),
        "total_per_agent": _dist([all_by_agent[a] for a in sorted(all_by_agent)]),
        "llm_calls": int(summary.get("llm_calls", 0)),
        "llm_cache_hits": int(summary.get("llm_cache_hits", 0)),
        "calls_per_agent_day": int(summary.get("llm_calls", 0)) / (n_seen * days),
    }


def _dist(values: list[int]) -> dict:
    """発火数分布の要約(§3.5「発火数分布を実測」)。素の Python = 桁の再現性を自分で握る。"""
    if not values:
        return {"n": 0}
    xs = sorted(float(v) for v in values)
    n = len(xs)
    mean = math.fsum(xs) / n
    var = math.fsum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "sd": math.sqrt(max(0.0, var)),
        "min": xs[0], "p10": _q(xs, 0.10), "median": _q(xs, 0.50),
        "p90": _q(xs, 0.90), "max": xs[-1],
        "frac_zero": sum(1 for x in xs if x == 0.0) / n,
    }


def _q(sorted_xs: list[float], q: float) -> float:
    """線形補間の分位点(numpy 既定と同じ定義)。"""
    if not sorted_xs:
        return 0.0
    pos = (len(sorted_xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


# --------------------------------------------------------------------------- #
# 探索(粗グリッド → 対数空間の二分)
# --------------------------------------------------------------------------- #
def _round(scale: float, digits: int) -> float:
    r = round(float(scale), digits)
    return 0.0 if r == 0.0 else r


def search(evaluate, *, target: float, tol: float, start: float, span: int,
           max_iter: int, digits: int, max_extend: int = 6) -> dict:
    """f(scale) = 目標発火率 になる scale を求める。

    1. 粗グリッド `start × 2^k`(k = −span..span)を評価してブラケットを探す
    2. グリッド全体が目標の片側にしか無ければ、**倍々でグリッドを外へ伸ばす**
       (θ の絶対水準は較正前なので、初期グリッドが外れているのが普通)
    3. ブラケットを**対数空間で**二分する(scale は正のスケール量なので幾何中点が自然)
    4. 相対誤差が tol 以内になったら終了。全評価点を trace に残す

    ★ 評価はすべて**丸めた scale** で行う(凍結する値と実際に検証した値が一致する)。
    """
    cache: dict[float, float] = {}
    trace: list[dict] = []
    phase = {"name": "grid"}

    def f(scale: float) -> float:
        s = _round(scale, digits)
        if s in cache:
            return cache[s]
        res = evaluate(s)
        value = float(res["f_salience_per_agent_day"])
        cache[s] = value
        trace.append({"theta_scale": s, "f": value,
                      "rel_error": _rel(value, target),
                      "n_salience": res["by_reason"].get(FIRE.SALIENCE, 0),
                      "phase": phase["name"]})
        return value

    grid = [_round(start * (2.0 ** k), digits) for k in range(-span, span + 1)]
    values = [f(s) for s in grid]
    monotone = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    # ---- グリッドが目標を跨いでいなければ外側へ伸ばす(単調減少を仮定した外挿)----
    phase["name"] = "extend"
    n_extend = 0
    while n_extend < max_extend and values[0] < target:      # 全部が目標より低い → 下へ
        nxt = _round(grid[0] / 2.0, digits)
        if nxt <= 0.0 or nxt >= grid[0]:
            break
        grid.insert(0, nxt)
        values.insert(0, f(nxt))
        n_extend += 1
    while n_extend < max_extend and values[-1] > target:      # 全部が目標より高い → 上へ
        nxt = _round(grid[-1] * 2.0, digits)
        if nxt <= grid[-1]:
            break
        grid.append(nxt)
        values.append(f(nxt))
        n_extend += 1
    phase["name"] = "bisect"

    lo = hi = None
    for i in range(len(grid) - 1):
        if values[i] >= target >= values[i + 1]:
            lo, hi = grid[i], grid[i + 1]
            break

    n_bisect = 0
    if lo is not None:
        for _ in range(max_iter):
            if _rel(cache[_round(lo, digits)], target) <= tol:
                break
            if _rel(cache[_round(hi, digits)], target) <= tol:
                break
            mid = _round(math.sqrt(lo * hi), digits)
            if mid <= lo or mid >= hi:
                break                                # 丸めの解像度に到達した
            n_bisect += 1
            value = f(mid)
            if value >= target:
                lo = mid
            else:
                hi = mid

    best = min(trace, key=lambda r: (r["rel_error"], r["theta_scale"]))
    return {
        "target": float(target), "tol": float(tol),
        "grid": grid, "monotone_on_grid": bool(monotone), "n_extend": n_extend,
        "bracket": ([lo, hi] if lo is not None else None),
        "n_evaluations": len(trace), "n_bisect": n_bisect,
        "trace": trace,
        "theta_scale": best["theta_scale"], "f_observed": best["f"],
        "rel_error": best["rel_error"],
        "converged": bool(best["rel_error"] <= tol),
    }


def _rel(value: float, target: float) -> float:
    if target <= 0.0:
        return abs(float(value))
    return abs(float(value) - float(target)) / float(target)


# --------------------------------------------------------------------------- #
# 総推論量の見積(§3.5)
# --------------------------------------------------------------------------- #
def estimate(pilot_res: dict, scale_check: dict | None, *,
             agent_scales: list[int], days: float) -> dict:
    """人数スケール別の 10 日ラン総推論量。

    ★ 「1 人あたり呼数が人数に不変」という**仮定**の外挿である。仮定の当否を
      2 点(既定の人数とその半分)で実測して併記する。密度依存のチャンネル
      (ext.crowd_local / ext.encounter)があるので不変性は自明ではない。
    """
    per = float(pilot_res["calls_per_agent_day"])
    rows = []
    for n in sorted(set(int(x) for x in agent_scales)):
        rows.append({
            "n_agents": n, "days": float(days),
            "llm_calls": per * n * float(days),
            "cog_fire": float(pilot_res["f_total_per_agent_day"]) * n * float(days),
            "salience": float(pilot_res["f_salience_per_agent_day"]) * n * float(days),
        })
    out = {
        "basis": {
            "calls_per_agent_day": per,
            "fires_per_agent_day": float(pilot_res["f_total_per_agent_day"]),
            "salience_per_agent_day": float(pilot_res["f_salience_per_agent_day"]),
            "measured_at_n_agents": int(pilot_res["n_agents"]),
            "backend": "mock",
        },
        "rows": rows,
        "assumption": "1 人あたり LLM 呼数が人数に対して不変(密度依存チャンネルがあるので自明ではない)",
    }
    if scale_check:
        a, b = scale_check["small"], scale_check["large"]
        ratio = (b["calls_per_agent_day"] / a["calls_per_agent_day"]
                 if a["calls_per_agent_day"] > 0 else None)
        out["scale_check"] = {
            "small": {"n_agents": a["n_agents"],
                      "calls_per_agent_day": a["calls_per_agent_day"],
                      "salience_per_agent_day": a["f_salience_per_agent_day"]},
            "large": {"n_agents": b["n_agents"],
                      "calls_per_agent_day": b["calls_per_agent_day"],
                      "salience_per_agent_day": b["f_salience_per_agent_day"]},
            "calls_per_agent_ratio_large_over_small": ratio,
            "reading": ("比が 1 から離れるほど『1 人あたり呼数は人数に依存する』= "
                        "上の外挿は人数が離れるほど外れる。2 点は 2 点でしかない。"),
        }
    return out


# --------------------------------------------------------------------------- #
# 凍結ファイル
# --------------------------------------------------------------------------- #
def build_doc(result: dict, pilot_res: dict, est: dict, *, run_cfg: dict,
              calib: dict, sigma: dict, stamp: bool) -> dict:
    doc = {
        "meta": {
            "schema": SCHEMA, "kind": KIND,
            "generator": "scripts/calibrate_theta.py",
            "design": "cognition-design-record.md 2.6 / 3.5",
            "policy": {
                "lever": "cognition.fire.theta_scale (全文脈・全個体に一様な 1 スカラ)",
                "preserved": "文脈別 θ 水準比 / ペルソナ由来の個体倍率 = **触らない**"
                             "(設計 2.6「個体差は θ_i の分散として残す」)",
                "controlled_quantity": "salience 発火 [件/日/人]"
                                       "(θ が門番をしているのはこの発火源だけ)",
                "homeostasis": "較正中は theta_mu=0 で切る(開ループ利得を測るため)",
                "apply": "conf へ書き戻さない。ランの dotlist で "
                         "cognition.fire.theta_scale=<値> と明示的に上書きする",
                "f_target_status": "provisional(f* は測定で決まる量ではなく宣言する量。"
                                   "人間側の実測にはまだ接地していない)",
            },
            "inputs": {
                "calib_table": {"file": CALIB.rel_path(calib["path"]),
                                "sha256": calib["sha256"],
                                "version": calib["table"]["version"],
                                "status": calib["table"]["status"]},
                "sigma_c": {"file": CALIB.rel_path(sigma["path"]),
                            "status": sigma["status"], "sha256": sigma["sha256"]},
            },
            "run": run_cfg,
        },
        "result": {
            "theta_scale": result["theta_scale"],
            "f_target_per_agent_day": result["target"],
            "f_observed_per_agent_day": result["f_observed"],
            "rel_error": result["rel_error"],
            "tol": result["tol"],
            "converged": result["converged"],
            "apply_override": f"cognition.fire.theta_scale={result['theta_scale']}",
        },
        "search": {k: result[k] for k in
                   ("grid", "monotone_on_grid", "n_extend", "bracket",
                    "n_evaluations", "n_bisect", "trace")},
        "pilot_at_frozen_scale": pilot_res,
        "inference_estimate": est,
    }
    if stamp:
        doc["meta"]["generated_at"] = (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    doc["meta"]["payload_sha256"] = CALIB.payload_sha256(doc)
    return doc


def write_doc(doc: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    return out


def render(doc: dict) -> str:
    """人が読む要約(凍結ファイルと同じ内容を表にするだけ)。"""
    res, est = doc["result"], doc["inference_estimate"]
    src = doc["search"]
    out = ["# θ 較正パイロット(設計 §2.6 / §3.5)", "",
           f"目標 f* = {res['f_target_per_agent_day']:.4g} 件/日/人"
           f"(salience 発火のみ・**provisional**)", "",
           f"- **凍結する全体スケール**: `cognition.fire.theta_scale="
           f"{res['theta_scale']}`",
           f"- 実測 f = {res['f_observed_per_agent_day']:.4g} 件/日/人"
           f"(相対誤差 {res['rel_error'] * 100:.2f}% / 許容 {res['tol'] * 100:.0f}%)",
           f"- 収束: {'はい' if res['converged'] else '**いいえ**'} / "
           f"グリッド単調性: {'単調' if src['monotone_on_grid'] else '**非単調**'} / "
           f"評価 {src['n_evaluations']} 点", "",
           "## 探索の全履歴", "",
           "| theta_scale | f [件/日/人] | 相対誤差 | salience 件数 | 段階 |",
           "|---|---|---|---|---|"]
    for row in src["trace"]:
        out.append(f"| {row['theta_scale']} | {row['f']:.4g} | "
                   f"{row['rel_error'] * 100:.1f}% | {row['n_salience']} | "
                   f"{row['phase']} |")
    pilot_res = doc["pilot_at_frozen_scale"]
    dist = pilot_res["salience_per_agent"]
    out += ["", "## 凍結スケールでの発火数分布(§3.5)", "",
            f"- 発火源内訳: {pilot_res['by_reason']}",
            f"- 文脈内訳: {pilot_res['by_context']}",
            f"- salience/人/日: mean={dist['mean']:.3g} sd={dist['sd']:.3g} "
            f"median={dist['median']:.3g} p90={dist['p90']:.3g} "
            f"max={dist['max']:.3g} 無発火の割合={dist['frac_zero'] * 100:.1f}%",
            f"- LLM 呼数: {pilot_res['llm_calls']}(1 人 1 日あたり "
            f"{pilot_res['calls_per_agent_day']:.3g})", "",
            "## 10 日ラン総推論量の見積(§3.5)", "",
            "| 人数 | 日数 | LLM 呼数(推定) | 認知イベント | うち salience |",
            "|---|---|---|---|---|"]
    for row in est["rows"]:
        out.append(f"| {row['n_agents']:,} | {row['days']:.0f} | "
                   f"{row['llm_calls']:,.0f} | {row['cog_fire']:,.0f} | "
                   f"{row['salience']:,.0f} |")
    if est.get("scale_check"):
        sc = est["scale_check"]
        ratio = sc["calls_per_agent_ratio_large_over_small"]
        out += ["", f"人数不変性の確認: {sc['small']['n_agents']} 体 = "
                    f"{sc['small']['calls_per_agent_day']:.4g} 呼/人/日 vs "
                    f"{sc['large']['n_agents']} 体 = "
                    f"{sc['large']['calls_per_agent_day']:.4g} 呼/人/日 "
                    f"(比 {ratio:.3f})",
                f"  {sc['reading']}"]
    out += ["", "## 限界(必ず併記する)", "",
            "- **mock バックエンド**のパイロット。実 LLM では発話量が変わり f はずれうる。",
            "- f\\* は**宣言する量**であって本スクリプトが決めた量ではない"
            "(神経科学の firing rate homeostasis でも set point は所与)。",
            "- 見積表は「1 人あたり呼数が人数に不変」の外挿。1 万体は 100 倍の外挿である。",
            "- 凍結値は conf へ書き戻さない。適用は "
            f"`{res['apply_override']}` の明示上書き。", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=100, help="パイロットの人数(既定 100)")
    ap.add_argument("--days", type=float, default=1.0, help="パイロットの日数(既定 1)")
    ap.add_argument("--steps", type=int, default=0, help="step 数を直接指定(>0 で優先)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=float, default=None,
                    help="目標発火率 f*[件/日/人]。既定 = conf の "
                         "cognition.g_update.theta_target_per_day")
    ap.add_argument("--tol", type=float, default=0.10, help="許容相対誤差(既定 10%%)")
    ap.add_argument("--start", type=float, default=1.0, help="グリッド中心の scale")
    ap.add_argument("--span", type=int, default=3,
                    help="グリッドの片側段数(start×2^±span。既定 3 = 0.125..8)")
    ap.add_argument("--max-iter", type=int, default=8, help="二分の最大反復")
    ap.add_argument("--max-extend", type=int, default=6,
                    help="グリッドが目標を跨がないときに外へ伸ばす最大回数(0=伸ばさない)")
    ap.add_argument("--digits", type=int, default=6, help="scale の丸め桁")
    ap.add_argument("--watch", action="store_true",
                    help="監視仕様(cognition.watch)ON で較正する(既定 OFF)")
    ap.add_argument("--project-days", type=float, default=10.0, help="見積の日数")
    ap.add_argument("--project-agents", default="100,1000,10000",
                    help="見積の人数スケール(カンマ区切り)")
    ap.add_argument("--no-scale-check", action="store_true",
                    help="人数不変性の確認パイロット(人数半分)を回さない")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--keep-run-dir", action="store_true")
    ap.add_argument("--set", dest="extra", action="append", default=[],
                    help="追加の conf 上書き(dotlist。複数可)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stamp", action="store_true",
                    help="生成時刻を書く(**非決定になる**。既定は書かない)")
    ap.add_argument("--print-only", action="store_true", help="凍結せず表示だけ")
    args = ap.parse_args(argv)

    base = load_config([])
    target = (float(args.target) if args.target is not None
              else float(base.cognition.g_update.theta_target_per_day))
    steps_per_day = MINUTES_PER_DAY // int(base.run.dt_min)
    steps = int(args.steps) if args.steps > 0 else int(round(args.days * steps_per_day))
    run_dir = Path(args.run_dir)
    _clean(run_dir, args.keep_run_dir)

    calib = CALIB.load_calib(None)
    sigma = CALIB.load_sigma(None)

    counter = {"n": 0}

    def evaluate(scale: float) -> dict:
        counter["n"] += 1
        return pilot(scale, agents=args.agents, steps=steps, seed=args.seed,
                     run_dir=run_dir, extra=list(args.extra), watch=bool(args.watch),
                     tag=f"e{counter['n']:03d}")

    print(f"[θ 較正] f* = {target} 件/日/人 / パイロット {args.agents} 体 × "
          f"{steps} step(mock, seed {args.seed}, watch="
          f"{'ON' if args.watch else 'OFF'})")
    result = search(evaluate, target=target, tol=float(args.tol),
                    start=float(args.start), span=int(args.span),
                    max_iter=int(args.max_iter), digits=int(args.digits),
                    max_extend=max(0, int(args.max_extend)))
    print(f"[θ 較正] 凍結 scale = {result['theta_scale']} / "
          f"f = {result['f_observed']:.4g}(相対誤差 {result['rel_error'] * 100:.2f}%)")

    # 凍結スケールそのもので 1 本(探索キャッシュではなく**凍結値で回した実測**を載せる)
    counter["n"] += 1
    pilot_res = pilot(result["theta_scale"], agents=args.agents, steps=steps,
                      seed=args.seed, run_dir=run_dir, extra=list(args.extra),
                      watch=bool(args.watch), tag="frozen")

    scale_check = None
    if not args.no_scale_check and args.agents >= 4:
        half = max(2, int(args.agents) // 2)
        small = pilot(result["theta_scale"], agents=half, steps=steps,
                      seed=args.seed, run_dir=run_dir, extra=list(args.extra),
                      watch=bool(args.watch), tag="half")
        scale_check = {"small": small, "large": pilot_res}

    est = estimate(pilot_res, scale_check,
                   agent_scales=[int(x) for x in str(args.project_agents).split(",")
                                 if str(x).strip()],
                   days=float(args.project_days))
    run_cfg = {
        "source": "pilot", "backend": "mock", "n_agents": int(args.agents),
        "n_steps": steps, "seed": int(args.seed), "dt_min": int(base.run.dt_min),
        "watch_enabled": bool(args.watch),
        "g_update_enabled": True, "theta_mu": 0.0,
        "overrides": sorted(str(x) for x in args.extra),
    }
    doc = build_doc(result, pilot_res, est, run_cfg=run_cfg, calib=calib,
                    sigma=sigma, stamp=bool(args.stamp))
    print()
    print(render(doc))
    if args.print_only:
        return 0
    out = write_doc(doc, Path(args.out))
    print(f"[凍結] {out}")
    print(f"payload_sha256: {doc['meta']['payload_sha256']}")
    print(f"[適用] {doc['result']['apply_override']}  ← conf には書き戻さない")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
