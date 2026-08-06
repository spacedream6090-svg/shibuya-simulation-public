#!/usr/bin/env python
"""S-05 seed 間分散の分散分解(SV2-B・読み取り専用・シム本体ゼロタッチ)。

    python scripts/analyze_seed_variance.py runs/A_s1 runs/A_s2 runs/B_s1 runs/B_s2
    python scripts/analyze_seed_variance.py "runs/exp_*_s*" --out experiments/seed_variance

■ 何のための装置か(「あると良い」ではなく「約束済み」)
  事前登録 `docs/plans/stationarity-preregistration.md` 前文 ④ 反証条件 (a)
  「**条件間差が seed 間差を上回らない**」は既に承認対象として書かれている。
  **この装置が無いと、その反証条件そのものが判定不能**である。
  定義・判定規則・退化ラベルの正典は同文書 **§3-G**。本スクリプトはその機械実装であり、
  **規則を勝手に足さない**(足したくなったら先に事前登録を直す)。

■ 分解の定義(§3-G.1 と同一。ここで独自定義しない)
    mean_c      = (1/|S_c|) Σ_s y_{c,s}                 # 条件 c の seed 平均
    V_condition = Var_c( mean_c )            (ddof=1)   # 決定論的分散 V_d
    V_seed      = mean_c[ Var_s( y_{c,s} ) ] (ddof=1)   # 確率的分散 E_s
    V_total     = V_condition + V_seed
    ratio       = V_condition / V_seed                  ← 判定に使う量
    share_condition = V_condition / V_total, share_seed = V_seed / V_total

  根拠: Carmona-Cabrero, Á., Muñoz-Carpena, R., Oh, W. S. & Muneepeerakul, R. (2024).
  *Decomposing Variance Decomposition for Stochastic Models.* JASSS 27(1):16.
  DOI 10.18564/jasss.5174 の "Approach IV"。**乱数 seed をモデル入力として扱ってはならない**
  とし、総分散を決定論的分散 V_d と確率的分散 E_s に分けて両者の寄与率を報告することを推奨する。
  反復数の目安は Lorscheid, Heine & Meyer (2012)(出力 CV が安定する点)。

■ 判定規則(§3-G.2。**ラン後に変えない**)
    G1: ratio > 1.0
    G2: ratio の seed 階層ブートストラップ 95%CI 下限 > 1.0   (B=2000, default_rng(0))
    G3: 全条件で条件内 seed 本数 |S_c| >= 3
  → **主張してよいのは G1 ∧ G2 ∧ G3 を満たした指標に限る**。
  退化ラベル:
    INSUFFICIENT_SEEDS               … G3 不成立(有意でも主張しない)
    DEGENERATE_ZERO_SEED_VARIANCE    … V_seed == 0(mock / キャッシュ完全命中で起こりうる。
                                        これを「条件差が圧倒的」と読んではならない)
    INSUFFICIENT_CONDITIONS          … 条件が 1 つしかない(V_condition が定義できない)
    NO_DATA                          … 当該指標が全ランで欠損

■ 既存資産の再利用(新規実装を増やさない)
  - `scripts/panel_stats.py` … `_parse_cond_seed`(run 名 → (条件, seed))/ `_load_run_day` /
    `_mean_over_days` / `_cfg_seed` / `t_ci` / `bh_fdr` / `_METRICS`(指標表とラベル)
  - `scripts/analyze_sweep.py` … seed 階層ブートストラップ(B=2000)と同型の手続き
  - **`scripts/analyze_g.py` の分解とは別物**。あちらは単一ラン内の *個体* 横断分散を
    g0 と Δg に分けるもので、seed でも条件でもない(拡張ではなく新設が正しい)。

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。`metrics_spec_hash` の SPEC_FILES 14 本に触れない。
  決定論: 条件・seed はソート順、ブートストラップは `numpy.random.default_rng(0)` 固定、
  JSON は `sort_keys=True, ensure_ascii=False`。データが無いときは捏造せず明示終了。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                                    # noqa: E402

import panel_stats as ps                              # noqa: E402  (純関数の再利用)

# --- 事前登録 §3-G.2 の閾値(ここが唯一の源。変更は事前登録の改訂を伴う) ---
G1_RATIO_MIN = 1.0        # G1: ratio がこれを超えること
G2_CI_LOWER_MIN = 1.0     # G2: ブートストラップ CI 下限がこれを超えること
G3_MIN_SEEDS = 3          # G3: 全条件でこの本数以上の seed
BOOTSTRAP_B = 2000        # ブートストラップ反復(analyze_sweep.py と同値)
BOOTSTRAP_SEED = 0        # numpy.random.default_rng(0) 固定

LABEL_OK = "OK"
LABEL_FAIL = "CONDITION_NOT_ABOVE_SEED"
LABEL_INSUFFICIENT_SEEDS = "INSUFFICIENT_SEEDS"
LABEL_DEGENERATE = "DEGENERATE_ZERO_SEED_VARIANCE"
LABEL_INSUFFICIENT_CONDITIONS = "INSUFFICIENT_CONDITIONS"
LABEL_NO_DATA = "NO_DATA"


# --------------------------------------------------------------------------- #
# 純関数(単体テストの対象)
# --------------------------------------------------------------------------- #
def _r(x, nd: int = 6):
    """None 安全な round(JSON をバイト安定にする)。diagnose_stationarity._r と同じ作法。"""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _var(xs, ddof: int = 1):
    """標本分散(ddof 既定 1)。n <= ddof は None(0 と偽らない)。"""
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n <= ddof:
        return None
    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / (n - ddof)


def decompose(by_cond: dict) -> dict:
    """{condition: {seed: value}} を V_condition / V_seed / ratio に分解する。

    事前登録 §3-G.1 の定義そのもの。判定はここでは下さない(下すのは verdict())。
    """
    conds = sorted(by_cond)
    means, within, n_seeds = {}, {}, {}
    for c in conds:
        vals = [by_cond[c][s] for s in sorted(by_cond[c]) if by_cond[c][s] is not None]
        n_seeds[c] = len(vals)
        if not vals:
            continue
        means[c] = sum(vals) / len(vals)
        w = _var(vals, ddof=1)
        if w is not None:
            within[c] = w
    used = sorted(means)
    v_cond = _var([means[c] for c in used], ddof=1) if len(used) >= 2 else None
    v_seed = (sum(within[c] for c in sorted(within)) / len(within)) if within else None
    v_total = None
    ratio = share_c = share_s = None
    if v_cond is not None and v_seed is not None:
        v_total = v_cond + v_seed
        if v_seed > 0:
            ratio = v_cond / v_seed
        if v_total > 0:
            share_c, share_s = v_cond / v_total, v_seed / v_total
    return {
        "conditions": used,
        "n_conditions": len(used),
        "n_seeds_by_condition": {c: n_seeds[c] for c in used},
        "cond_means": {c: _r(means[c]) for c in used},
        "V_condition": _r(v_cond),
        "V_seed": _r(v_seed),
        "V_total": _r(v_total),
        "ratio": _r(ratio),
        "share_condition": _r(share_c),
        "share_seed": _r(share_s),
        # 条件内分散が 1 自由度未満で推定できなかった条件(G3 の実体)
        "conditions_without_within_var": [c for c in used if c not in within],
    }


def bootstrap_ratio_ci(by_cond: dict, b: int = BOOTSTRAP_B,
                       seed: int = BOOTSTRAP_SEED) -> dict:
    """seed 階層ブートストラップで ratio の 95%CI を出す(G2 の実体)。

    条件は固定し、**条件内の seed を復元抽出**して ratio を再計算する
    (`analyze_sweep.py::_bootstrap_ci` と同型。乱数は default_rng(seed) 固定)。
    ratio が定義できない反復(V_seed == 0 等)は棄却せず **除外して n_valid を報告する**。
    """
    conds = sorted(by_cond)
    pools = {c: [by_cond[c][s] for s in sorted(by_cond[c]) if by_cond[c][s] is not None]
             for c in conds}
    pools = {c: v for c, v in pools.items() if v}
    if len(pools) < 2:
        return {"lo": None, "hi": None, "b": 0, "n_valid": 0}
    rng = np.random.default_rng(seed)
    order = sorted(pools)
    ratios = []
    for _ in range(int(b)):
        means, within = [], []
        for c in order:
            arr = pools[c]
            idx = rng.integers(0, len(arr), size=len(arr))
            samp = [arr[int(i)] for i in idx]
            means.append(sum(samp) / len(samp))
            w = _var(samp, ddof=1)
            if w is not None:
                within.append(w)
        vc = _var(means, ddof=1)
        vs = (sum(within) / len(within)) if within else None
        if vc is None or vs is None or vs <= 0:
            continue
        ratios.append(vc / vs)
    if not ratios:
        return {"lo": None, "hi": None, "b": int(b), "n_valid": 0}
    a = np.sort(np.asarray(ratios, dtype=np.float64))
    return {"lo": _r(float(np.percentile(a, 2.5))),
            "hi": _r(float(np.percentile(a, 97.5))),
            "b": int(b), "n_valid": int(a.size)}


def verdict(dec: dict, ci: dict) -> dict:
    """事前登録 §3-G.2 の G1/G2/G3 を機械判定してラベルを返す。

    ★ 退化を「合格」にも「不合格」にも丸めない。丸めると読者が区別できなくなる。
    """
    if dec["n_conditions"] == 0:
        return {"label": LABEL_NO_DATA, "G1": None, "G2": None, "G3": None,
                "claimable": False}
    if dec["n_conditions"] < 2:
        return {"label": LABEL_INSUFFICIENT_CONDITIONS, "G1": None, "G2": None,
                "G3": None, "claimable": False}
    g3 = (min(dec["n_seeds_by_condition"].values()) >= G3_MIN_SEEDS
          and not dec["conditions_without_within_var"])
    if dec["V_seed"] is not None and dec["V_seed"] == 0:
        return {"label": LABEL_DEGENERATE, "G1": None, "G2": None, "G3": bool(g3),
                "claimable": False}
    if dec["ratio"] is None:
        return {"label": LABEL_DEGENERATE, "G1": None, "G2": None, "G3": bool(g3),
                "claimable": False}
    g1 = dec["ratio"] > G1_RATIO_MIN
    g2 = (ci.get("lo") is not None) and (ci["lo"] > G2_CI_LOWER_MIN)
    if not g3:
        # ★ G3 不成立は「有意でも主張しない」。G1/G2 の値は記録するが claimable=False。
        return {"label": LABEL_INSUFFICIENT_SEEDS, "G1": bool(g1), "G2": bool(g2),
                "G3": False, "claimable": False}
    ok = bool(g1 and g2)
    return {"label": LABEL_OK if ok else LABEL_FAIL, "G1": bool(g1), "G2": bool(g2),
            "G3": True, "claimable": ok}


def spread(dec: dict) -> dict:
    """条件間差を S-16 FormatSpread と同形式(中央値 [min, max])で返す。単一値で書かない。"""
    vals = sorted(float(v) for v in dec["cond_means"].values())
    if not vals:
        return {"median": None, "min": None, "max": None, "range": None}
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"median": _r(med), "min": _r(vals[0]), "max": _r(vals[-1]),
            "range": _r(vals[-1] - vals[0])}


# --------------------------------------------------------------------------- #
# ラン読み取り
# --------------------------------------------------------------------------- #
def collect(run_dirs: list[str], use_warmup: bool = False) -> dict:
    """各ランの run_day から指標を集め {metric: {cond: {seed: value}}} を返す。"""
    per_metric: dict[str, dict] = {k: {} for k in ps._METRIC_KEYS}
    runs_meta = []
    for run_dir in sorted(run_dirs):
        name = os.path.basename(os.path.normpath(run_dir))
        cond, seed = ps._parse_cond_seed(name, ps._cfg_seed(run_dir))
        if seed is None:
            seed = name                       # seed 不明: run 名そのものを seed 鍵にする
        rd = ps._load_run_day(run_dir)
        runs_meta.append({"run": name, "condition": cond, "seed": seed})
        for key in ps._METRIC_KEYS:
            if key == "r2_traits_Yext":
                v = ps._r2_yext(run_dir)
            else:
                v = ps._mean_over_days(rd, key, use_warmup)
            per_metric[key].setdefault(cond, {})[seed] = v
    return {"per_metric": per_metric, "runs": runs_meta}


def analyze(run_dirs: list[str], *, use_warmup: bool = False,
            b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    got = collect(run_dirs, use_warmup=use_warmup)
    rows = []
    for key in ps._METRIC_KEYS:
        by_cond = {c: sv for c, sv in got["per_metric"][key].items()
                   if any(v is not None for v in sv.values())}
        dec = decompose(by_cond)
        ci = bootstrap_ratio_ci(by_cond, b=b, seed=seed) if dec["n_conditions"] >= 2 \
            else {"lo": None, "hi": None, "b": 0, "n_valid": 0}
        vd = verdict(dec, ci)
        rows.append({"metric": key, "label_jp": ps._LABEL.get(key, key),
                     **dec, "ci95": ci, **vd, "spread": spread(dec)})
    claimable = sorted(r["metric"] for r in rows if r["claimable"])
    withheld = sorted(r["metric"] for r in rows if not r["claimable"]
                      and r["label"] != LABEL_NO_DATA)
    return {
        "spec": {"G1_ratio_min": G1_RATIO_MIN, "G2_ci_lower_min": G2_CI_LOWER_MIN,
                 "G3_min_seeds": G3_MIN_SEEDS, "bootstrap_b": int(b),
                 "bootstrap_seed": int(seed), "use_warmup": bool(use_warmup),
                 "preregistration": "docs/plans/stationarity-preregistration.md §3-G"},
        "runs": got["runs"],
        "n_runs": len(got["runs"]),
        "metrics": rows,
        "claimable_metrics": claimable,
        "withheld_metrics": withheld,
    }


# --------------------------------------------------------------------------- #
# レンダ
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 4) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def render(res: dict) -> str:
    L: list[str] = []
    L.append("# S-05 seed 間分散の分散分解")
    L.append("")
    L.append("> 定義・判定規則の正典は **事前登録 §3-G**。本レポートはその機械判定の出力である。")
    L.append("> `ratio = V_condition / V_seed`。**主張してよいのは G1 ∧ G2 ∧ G3 を満たした指標に限る**。")
    L.append("> 根拠: Carmona-Cabrero et al. (2024) JASSS 27(1):16(seed を入力として扱わない分解)。")
    L.append("")
    sp = res["spec"]
    L.append(f"- ラン数: {res['n_runs']}  条件数: "
             f"{len(sorted({r['condition'] for r in res['runs']}))}")
    L.append(f"- 閾値: G1 ratio > {sp['G1_ratio_min']} / G2 CI下限 > {sp['G2_ci_lower_min']} / "
             f"G3 seed本数 ≥ {sp['G3_min_seeds']}")
    L.append(f"- ブートストラップ: B={sp['bootstrap_b']} / `default_rng({sp['bootstrap_seed']})`")
    L.append("")
    L.append("## 条件 × seed の内訳")
    L.append("")
    L.append("| run | 条件 | seed |")
    L.append("|---|---|---|")
    for r in res["runs"]:
        L.append(f"| `{r['run']}` | {r['condition']} | {r['seed']} |")
    L.append("")
    L.append("## 指標ごとの分解と判定")
    L.append("")
    L.append("| 指標 | V_condition | V_seed | ratio | 95%CI | share_cond | share_seed "
             "| G1 | G2 | G3 | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tick = {True: "○", False: "×", None: "—"}
    for m in res["metrics"]:
        ci = m["ci95"]
        cis = "—" if ci["lo"] is None else f"[{_f(ci['lo'],3)}, {_f(ci['hi'],3)}]"
        L.append(f"| {m['label_jp']} | {_f(m['V_condition'],6)} | {_f(m['V_seed'],6)} "
                 f"| {_f(m['ratio'],3)} | {cis} | {_f(m['share_condition'],3)} "
                 f"| {_f(m['share_seed'],3)} | {tick[m['G1']]} | {tick[m['G2']]} "
                 f"| {tick[m['G3']]} | **{m['label']}** |")
    L.append("")
    L.append("## 条件間差(S-16 FormatSpread 形式 = 中央値 [min, max]。単一値で書かない)")
    L.append("")
    L.append("| 指標 | 条件平均の中央値 [min, max] | レンジ | 条件別 seed 本数 |")
    L.append("|---|---|---|---|")
    for m in res["metrics"]:
        s = m["spread"]
        if s["median"] is None:
            continue
        ns = ", ".join(f"{c}={n}" for c, n in sorted(m["n_seeds_by_condition"].items()))
        L.append(f"| {m['label_jp']} | {_f(s['median'])} [{_f(s['min'])}, {_f(s['max'])}] "
                 f"| {_f(s['range'])} | {ns} |")
    L.append("")
    L.append("## ★この規則で「出さないことにした結論」(報告書 §7.1 へ転記する)")
    L.append("")
    if res["withheld_metrics"]:
        for m in res["metrics"]:
            if m["metric"] in res["withheld_metrics"]:
                L.append(f"- **{m['label_jp']}** … {m['label']}"
                         f"(ratio={_f(m['ratio'],3)} / 条件数={m['n_conditions']})")
    else:
        L.append("- (なし)")
    L.append("")
    L.append("## 主張してよい指標(G1 ∧ G2 ∧ G3)")
    L.append("")
    if res["claimable_metrics"]:
        for m in res["metrics"]:
            if m["claimable"]:
                L.append(f"- **{m['label_jp']}** … ratio={_f(m['ratio'],3)} "
                         f"CI下限={_f(m['ci95']['lo'],3)}")
    else:
        L.append("- **(なし)** — 条件間差が seed 間差を上回った指標は 1 つも無い。"
                 "事前登録 前文 ④(a) により、**この母集団からは条件差の結論を出さない**。")
    L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        for h in hits:
            if os.path.isdir(h):
                out.append(h)
    return sorted(set(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-05 seed 間分散の分散分解(読み取り専用)")
    ap.add_argument("runs", nargs="+", help="ランディレクトリ(glob 可)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "seed_variance"))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--use-warmup", action="store_true",
                    help="warmup 日も平均に含める(既定は除外)")
    a = ap.parse_args(argv)

    run_dirs = _expand(a.runs)
    if not run_dirs:
        print("[analyze_seed_variance] ランが 1 つも見つからない。捏造せず終了する。",
              file=sys.stderr)
        return 2
    res = analyze(run_dirs, use_warmup=a.use_warmup, b=a.bootstrap, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "seed_variance.json")
    mpath = os.path.join(a.out, "seed_variance.md")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    print(f"[analyze_seed_variance] runs={res['n_runs']} "
          f"claimable={len(res['claimable_metrics'])} "
          f"withheld={len(res['withheld_metrics'])}")
    print(f"  -> {jpath}")
    print(f"  -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
