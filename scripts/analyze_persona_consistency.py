#!/usr/bin/env python
"""S-06 個体レベルの長期一貫性(SV2-B・読み取り専用・シム本体ゼロタッチ)。

    python scripts/analyze_persona_consistency.py runs/<name>
    python scripts/analyze_persona_consistency.py runs/<name> --out experiments/persona_consistency

■ 問い
  「同じ個体が 10 日(= 1,440 step)を通じて自分らしくあり続けるか」。
  Li, K. et al. (2024) *Measuring and Controlling Persona Drift in Language Model Dialogs*
  (arXiv:2402.10962)は **8 ターン以内で有意な persona drift** を示した。本シムの時間尺度は
  それを桁で超える。サーベイ §3.5 のとおり **lifelong evaluation はほぼ存在しない**ので、
  本指標は long-horizon persona consistency の測定として分野で新しい。

■ 出す 1 数値
    consistency_ratio = mean_i(個体 *内* 日間 TVD) / mean_d(同日の個体 *間* TVD)
  **小さいほど「個体が一貫している」**(自分の昨日との差 < 他人との差)。
  1 を超えたら「個体差より日替わりの方が大きい」= 一貫していない。

■ ★交換単位が変わることの明示(この 1 行を書かないと検定の意味が変わる)
  `scripts/diagnose_stationarity.py` は D を **「エージェント方向の標本」**として扱う
  (交換単位 = エージェント。同 docstring L44–45 が Farine & Carter 2022 を引く)。
  **本スクリプトが個体別に検定するとき、交換単位は「日」に変わる。**
  したがって帰無仮説も変わる:
    - diagnose_stationarity … H0「**各エージェントについて**日ラベルは情報を持たない」
                              (交換 = 各エージェントの差ベクトルの符号反転)
    - 本スクリプト(個体別) … H0「**この個体について**日ラベルは情報を持たない」
                              (交換 = この個体の日ペア差ベクトルの符号反転。標本 = 日ペア)
  検定量(TVD)と手続き(sign-flip permutation)は同一なので **関数は import して共有する**
  (複製しない)。変わるのは **D の行が何であるか**だけである。
  ★個体ごとに検定を回すので**多重比較になる**。個体単位の主張はせず、
  BH-FDR 後の有意率(= 何割の個体で日ラベルが情報を持つか)だけを報告する。

■ 特徴ベクトル
  個体 × 日 の**イベント種別ヒストグラム**を各日で和 1 に正規化した確率ベクトル。
  TVD = 0.5 · Σ_j |p_j − q_j| ∈ [0,1] = 「1 日のプロフィールのうち何割が入れ替わったか」。
  diagnose_stationarity の主判定と同じ読み方であり、あちらは *平均* プロフィールの差、
  こちらは *個体ごと* のプロフィールの差を見る(**別の量**である)。

■ 正直な限界
  1. 「一貫している」の中身は**行動配分の一貫性**であり、**発話内容の一貫性ではない**。
     persona drift の本来の定義(自己記述の維持)とは別の量である。
  2. イベント種別は世界側の機会に強く依存する(雨の日は外出が減る)。したがって
     **一貫性が高くても「個体が強い」ことの証明にはならない**。`flat_traits`(ゼロ対照)
     ランとの差分を見て初めて「一貫性が特性由来か環境由来か」が言える(--baseline)。
  3. 両日に現れない個体は除外する。除外数を必ず報告する(黙って落とさない)。

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。凍結ファイル `diagnose_stationarity.py` は
  **import するだけ**(内容を変えないのでハッシュは不変。逆方向の依存は作らない)。
  決定論: 個体・日・kind はソート順、部分抽出は等間隔、乱数は default_rng(0) 固定。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Iterable, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                                        # noqa: E402

# ★凍結ファイルからの import(内容は 1 バイトも変えない。複製もしない)
from diagnose_stationarity import (                       # noqa: E402
    _r, paired_dz, paired_signflip_tvd,
)

import run_dt                                          # noqa: E402  (W2-3: Δt の単一の源)

# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 の 144(= 従来と 1 ビットも変わらない)。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY
MAX_PAIR_AGENTS = 200          # 個体間 TVD の全ペア計算に使う個体数の上限(等間隔サブサンプル)
MAX_KINDS = 96                 # 特徴次元の上限(超過分は "_other" へ集約。凍結側と同じ流儀)
MIN_DAYS = 2                   # 個体内比較に必要な最小在籍日数
RNG_SEED = 0

# --------------------------------------------------------------------------- #
# W4-D': ★**この解析だけは kind で絞れない**(絞らないことの宣言)
#
# 特徴ベクトルが「個体 × 日 の **kind ヒストグラム**」そのものなので、どの kind を
# 落としても次元が 1 本消えて TVD が動く。すなわち「WANT_KINDS 以外の行は下流が
# 必ず読み飛ばす」という絞り読みの前提が、本解析では**原理的に成立しない**
# (`build_panel` の資産追跡が kind 非依存だったのと同族の事情)。
# したがって `None` = 全 kind を宣言し、`l1_stream` へ `kinds=` を渡さない。
# 本解析で効くのは代わりに次の 4 点である:
#   (a) **3 列射影**(payload を 1 バイトも読まない)  (b) **逐次生成**(全件 list を
#   作らない)  (c) **part 群の透過連結**  (d) **Windows の共有フラグ付き open**
# ★ここを「EVENT_KINDS 全部」と書いてはならない。未登録 kind を持つランで
#   その行が黙って落ち、次元が減って出力が変わる(欠測を偽の値で埋めない)。
# --------------------------------------------------------------------------- #
WANT_KINDS = None


# --------------------------------------------------------------------------- #
# 純関数(単体テストの対象)
# --------------------------------------------------------------------------- #
def tvd(p, q) -> float:
    """total variation distance = 0.5 · Σ|p−q|。p, q は和 1 の確率ベクトル。"""
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    return 0.5 * float(np.abs(a - b).sum())


def _even_subsample(items: list, cap: int) -> list:
    """等間隔サブサンプル(決定論)。cap 以下ならそのまま返す。"""
    items = sorted(items)
    n = len(items)
    if n <= cap or cap <= 0:
        return items
    idx = [int(round(i * (n - 1) / (cap - 1))) for i in range(cap)]
    out, seen = [], set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def build_profiles(events: Iterable[dict], steps_per_day: int = STEPS_PER_DAY,
                   max_kinds: int = MAX_KINDS) -> dict:
    """イベント列から個体 × 日 × kind の確率ベクトルを作る。

    `events` は **1 回だけ舐める**(list でも generator でも同じ結果になる)。

    返り値 {"P": ndarray(agent, day, dim), "agents": [id...], "days": [d...],
            "dims": [kind...], "present": ndarray(agent, day) bool}
    その日 1 件もイベントが無い (個体, 日) は present=False(ゼロベクトルにしない)。
    """
    counts: dict[tuple[int, int, str], int] = defaultdict(int)
    kind_total: dict[str, int] = defaultdict(int)
    agents_set, days_set = set(), set()
    for e in events:
        aid = e.get("agent")
        if aid is None or int(aid) < 0:
            continue
        aid = int(aid)
        day = int(e["step"]) // int(steps_per_day)
        kind = str(e["kind"])
        counts[(aid, day, kind)] += 1
        kind_total[kind] += 1
        agents_set.add(aid)
        days_set.add(day)
    if not counts:
        return {"P": np.zeros((0, 0, 0)), "agents": [], "days": [], "dims": [],
                "present": np.zeros((0, 0), dtype=bool)}
    # 次元: 出現数の多い順に max_kinds 個。タイは kind 名の昇順(決定論)。残りは "_other"。
    ranked = sorted(kind_total, key=lambda k: (-kind_total[k], k))
    keep = sorted(ranked[:max_kinds])
    dims = keep + (["_other"] if len(ranked) > max_kinds else [])
    dim_of = {k: i for i, k in enumerate(dims)}
    other = dim_of.get("_other")

    agents = sorted(agents_set)
    days = sorted(days_set)
    ai = {a: i for i, a in enumerate(agents)}
    di = {d: i for i, d in enumerate(days)}
    P = np.zeros((len(agents), len(days), len(dims)), dtype=np.float64)
    for (aid, day, kind), c in counts.items():
        j = dim_of.get(kind, other)
        if j is None:
            continue
        P[ai[aid], di[day], j] += c
    tot = P.sum(axis=2)
    present = tot > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(present[:, :, None], P / np.where(tot[:, :, None] == 0, 1.0,
                                                       tot[:, :, None]), 0.0)
    return {"P": P, "agents": agents, "days": days, "dims": dims, "present": present}


def within_agent_tvd(prof: dict) -> dict:
    """個体ごとに、在籍した連続日ペアの TVD 平均を返す。{agent: mean_tvd}。"""
    P, present = prof["P"], prof["present"]
    out: dict[int, float] = {}
    for i, aid in enumerate(prof["agents"]):
        idx = [j for j in range(P.shape[1]) if present[i, j]]
        if len(idx) < MIN_DAYS:
            continue
        vals = [tvd(P[i, idx[k]], P[i, idx[k + 1]]) for k in range(len(idx) - 1)]
        if vals:
            out[aid] = float(sum(vals) / len(vals))
    return out


def between_agent_tvd(prof: dict, cap: int = MAX_PAIR_AGENTS) -> dict:
    """同じ日に在籍した個体どうしの TVD 平均を日ごとに返す。{day: mean_tvd}。

    全ペア O(A²) を避けるため、**等間隔サブサンプル**(決定論)で個体を cap 件に絞る。
    """
    P, present = prof["P"], prof["present"]
    agents = prof["agents"]
    pick = set(_even_subsample(list(range(len(agents))), cap))
    out: dict[int, float] = {}
    for j, day in enumerate(prof["days"]):
        rows = [i for i in range(P.shape[0]) if i in pick and present[i, j]]
        if len(rows) < 2:
            continue
        M = P[rows, j, :]
        # 0.5·Σ|p−q| をベクトル化(行数 <= cap なので密行列で足りる)
        d = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2) * 0.5
        iu = np.triu_indices(len(rows), k=1)
        out[day] = float(d[iu].mean())
    return out


def per_agent_signflip(prof: dict, mc: int = 2000, seed: int = RNG_SEED) -> list[dict]:
    """個体ごとに「日ラベルは情報を持たない」を sign-flip permutation で検定する。

    ★交換単位 = **日ペア**(diagnose_stationarity はエージェント)。
    D の行 = その個体の連続日ペア差ベクトル。検定関数は凍結ファイルから import して共有。
    """
    P, present = prof["P"], prof["present"]
    rows: list[dict] = []
    for i, aid in enumerate(prof["agents"]):
        idx = [j for j in range(P.shape[1]) if present[i, j]]
        if len(idx) < 3:                    # 日ペアが 2 本未満では符号反転に意味が無い
            continue
        D = np.asarray([P[i, idx[k]] - P[i, idx[k + 1]] for k in range(len(idx) - 1)],
                       dtype=np.float64)
        t = paired_signflip_tvd(D, n_mc=mc, rng_seed=seed)
        dz = paired_dz(D)
        rows.append({"agent": int(aid), "n_day_pairs": int(D.shape[0]),
                     "tvd": t.get("tvd"), "p": t.get("p"), "method": t.get("method"),
                     "dz_mean_abs": dz.get("dz_mean_abs")})
    rows.sort(key=lambda r: r["agent"])
    return rows


# --------------------------------------------------------------------------- #
# 解析本体
# --------------------------------------------------------------------------- #
def _load_events(run_dir: str) -> Iterator[dict]:
    """l1_events を [step, agent_id, kind] のみで読む(payload に触らない = 最安)。

    W4-D': 共有の `l1_stream` 経由へ移した。**kind では絞らない**(理由は上の
    `WANT_KINDS = None` の宣言)。旧実装との差は
    (a) 全件 list ではなく**逐次 yield**(`build_profiles` は 1 回舐めるだけなので
    ここが唯一の全件 RAM 展開だった)、(b) part 群の透過連結、(c) Windows の
    共有フラグ付き open、の 3 点。**yield する dict は旧実装の逐語**である。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        return
    for d in ls.iter_columns(run_dir, ["step", "agent_id", "kind"]):
        for st, aid, kind in zip(d["step"], d["agent_id"], d["kind"]):
            yield {"step": int(st), "agent": int(aid), "kind": kind}


def analyze(run_dir: str, *, cap: int = MAX_PAIR_AGENTS, mc: int = 2000,
            seed: int = RNG_SEED, steps_per_day: int | None = None) -> dict:
    """W2-3: `steps_per_day=None`(既定)はこのランの run.dt_min から導く
    (Δt=10 なら 144 = 従来の既定と同値)。明示指定があればそちらを尊重する。"""
    if steps_per_day is None:
        steps_per_day = run_dt.steps_per_day(run_dir)
    events = _load_events(run_dir)
    prof = build_profiles(events, steps_per_day=steps_per_day)
    if not prof["agents"]:
        return {"run": os.path.basename(os.path.normpath(run_dir)),
                "status": "NO_DATA", "n_agents": 0, "n_days": 0}
    within = within_agent_tvd(prof)
    between = between_agent_tvd(prof, cap=cap)
    n_excluded = len(prof["agents"]) - len(within)
    w_mean = (sum(within.values()) / len(within)) if within else None
    b_mean = (sum(between.values()) / len(between)) if between else None
    ratio = (w_mean / b_mean) if (w_mean is not None and b_mean not in (None, 0)) else None

    rows = per_agent_signflip(prof, mc=mc, seed=seed)
    pvals = [r["p"] for r in rows if r["p"] is not None]
    sig_raw = sum(1 for p in pvals if p < 0.05)
    sig_fdr = None
    if pvals:
        from panel_stats import bh_fdr        # 遅延 import(凍結外・BH-FDR を再実装しない)
        q = bh_fdr(pvals)
        sig_fdr = sum(1 for v in q if v is not None and v < 0.05)

    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "status": "OK",
        "spec": {"steps_per_day": int(steps_per_day), "max_pair_agents": int(cap),
                 "mc": int(mc), "rng_seed": int(seed),
                 "exchange_unit": "day_pair (NOT agent — see docstring)",
                 "null_hypothesis": "この個体について日ラベルは情報を持たない"},
        "n_agents": len(prof["agents"]),
        "n_days": len(prof["days"]),
        "n_dims": len(prof["dims"]),
        "dims": prof["dims"],
        "n_agents_with_within": len(within),
        "n_agents_excluded_lt_min_days": int(n_excluded),
        "within_agent_tvd_mean": _r(w_mean),
        "between_agent_tvd_mean": _r(b_mean),
        "consistency_ratio": _r(ratio),
        "within_by_agent": {str(a): _r(v) for a, v in sorted(within.items())},
        "between_by_day": {str(d): _r(v) for d, v in sorted(between.items())},
        "per_agent_tests": rows,
        "n_tested": len(rows),
        "n_sig_raw_p05": int(sig_raw),
        "n_sig_bh_fdr_q05": (int(sig_fdr) if sig_fdr is not None else None),
        "frac_sig_bh_fdr_q05": (_r(sig_fdr / len(rows)) if (sig_fdr is not None and rows)
                                else None),
    }


def render(res: dict) -> str:
    L: list[str] = []
    L.append(f"# S-06 個体レベル長期一貫性 — `{res.get('run','?')}`")
    L.append("")
    if res.get("status") != "OK":
        L.append(f"**{res.get('status')}** — 個体別プロフィールを作れるイベントが無い。"
                 "捏造しない。")
        return "\n".join(L) + "\n"
    L.append("> `consistency_ratio = 個体内 日間 TVD / 個体間 TVD`。"
             "**小さいほど個体が一貫している**(1 を超えたら個体差より日替わりが大きい)。")
    L.append("> ★交換単位は **日ペア**であって `diagnose_stationarity` の**エージェント**ではない"
             "(帰無仮説が違う。docstring 参照)。")
    L.append("")
    L.append(f"- 個体数: {res['n_agents']}  日数: {res['n_days']}  特徴次元: {res['n_dims']}")
    L.append(f"- 個体内比較ができた個体: {res['n_agents_with_within']} "
             f"(在籍日数 < {MIN_DAYS} で除外: {res['n_agents_excluded_lt_min_days']})")
    L.append("")
    L.append("| 量 | 値 |")
    L.append("|---|---|")
    L.append(f"| 個体 **内** 日間 TVD の平均 | {res['within_agent_tvd_mean']} |")
    L.append(f"| 個体 **間** TVD の平均 | {res['between_agent_tvd_mean']} |")
    L.append(f"| **consistency_ratio** | **{res['consistency_ratio']}** |")
    L.append("")
    r = res["consistency_ratio"]
    if r is not None:
        if r < 1.0:
            L.append("→ 個体は「他人との差」より「自分の昨日との差」の方が小さい "
                     "= **一貫している側**。")
        else:
            L.append("→ 個体内の日替わりが個体間差を上回っている "
                     "= **一貫していない側**(persona drift の疑い)。")
    L.append("")
    L.append("## 個体別 sign-flip 検定(多重比較。個体単位の主張はしない)")
    L.append("")
    L.append(f"- 検定した個体: {res['n_tested']}(日ペアが 2 本以上ある個体のみ)")
    L.append(f"- 生の p < 0.05: {res['n_sig_raw_p05']}")
    L.append(f"- **BH-FDR q < 0.05: {res['n_sig_bh_fdr_q05']} "
             f"(割合 {res['frac_sig_bh_fdr_q05']})**")
    L.append("")
    L.append("## 正直な限界")
    L.append("")
    L.append("- 測っているのは**行動配分の一貫性**であり、**発話内容の一貫性ではない**。")
    L.append("- イベント種別は世界側の機会に依存する。`flat_traits` 対照との差分を見るまで、")
    L.append("  一貫性が**特性由来か環境由来か**は言えない。")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-06 個体レベル長期一貫性(読み取り専用)")
    ap.add_argument("run", help="ランディレクトリ")
    ap.add_argument("--out", default=None, help="出力先(既定 <run>/analysis)")
    ap.add_argument("--cap", type=int, default=MAX_PAIR_AGENTS)
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--baseline", default=None,
                    help="ゼロ対照ラン(flat_traits 等)。指定すると ratio の差分を併記")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.run):
        print(f"[analyze_persona_consistency] ランが無い: {a.run}", file=sys.stderr)
        return 2
    res = analyze(a.run, cap=a.cap, mc=a.mc, seed=a.seed)
    if a.baseline:
        if os.path.isdir(a.baseline):
            base = analyze(a.baseline, cap=a.cap, mc=a.mc, seed=a.seed)
            res["baseline"] = {"run": base.get("run"),
                               "consistency_ratio": base.get("consistency_ratio")}
            if res.get("consistency_ratio") is not None and \
                    base.get("consistency_ratio") is not None:
                res["baseline"]["delta"] = _r(res["consistency_ratio"]
                                              - base["consistency_ratio"])
        else:
            res["baseline"] = {"run": a.baseline, "status": "NOT_FOUND"}
    out = a.out or os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "persona_consistency.json")
    mp = os.path.join(out, "persona_consistency.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    print(f"[analyze_persona_consistency] {res.get('status')} "
          f"ratio={res.get('consistency_ratio')} agents={res.get('n_agents')}")
    print(f"  -> {jp}")
    print(f"  -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
