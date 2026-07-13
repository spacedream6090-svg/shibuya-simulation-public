#!/usr/bin/env python
"""stage2 統計: 複数ランのパネルを縦結合し、シード横断の平均±95%CI +
条件×同seed のペア比較(第18バッチ②)。

    python scripts/panel_stats.py runs/<A> runs/<B> ... [--out experiments/<exp>]

data-strategy.md の stage2-3(複数シードの平均±信頼区間・条件間は同seedでペア比較=
分散削減・研究量 R²)を実装する **読み出し専用** スクリプト。各ランの
panel/run_day.parquet(build_panel の出力)を読む。無ければ build_panel を内部で
呼んで生成する(sim 本体・schema は変更しない)。

出力(--out 既定 = experiments/panel_stats):
  panel_combined.parquet … 全ランの run_day を縦結合(run 列つき)
  panel_stats.parquet    … 指標×条件の 平均・95%CI・n・sd の要約表
  panel_stats.md         … 人が読める日本語サマリ(CI 表 + ペア比較表)

CI は正規近似(mean ± 1.96·sd/√n)。n が小さいと粗い近似である旨を明記する。
R²(traits→Y_external)は society.observer.measure を read-only で再利用して各ランで算出。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)                     # 同ディレクトリの build_panel を import

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import build_panel as bp                       # noqa: E402  (パネル生成の再利用)
from society.observer import measure as m       # noqa: E402  (R² 等の read-only 再利用)

# CI 表に載せる指標(run_day の数値列 + 研究量)。ラベルは日本語。
_METRICS = [
    ("spend_pp", "消費/人日(円)"),
    ("wage_pp", "賃金/人日(円)"),
    ("sleep_h_median", "睡眠中央値(h)"),
    ("engel", "エンゲル係数"),
    ("speak_pp", "発話/人日"),
    ("sns_post_pp", "SNS投稿/人日"),
    ("crime_pp", "窃盗/人日"),
    ("n_enter_building", "建物入館(日次)"),
    ("n_label_adopt", "語の採用(日次)"),
    ("n_transmission", "伝播(日次)"),
    ("status_gini", "地位ジニ"),
    ("wealth_gini", "資産ジニ"),
    ("r2_traits_Yext", "R²(traits→Y_external)"),
]
_METRIC_KEYS = [k for k, _ in _METRICS]
_LABEL = dict(_METRICS)


# --------------------------------------------------------------------------- #
# 補助
# --------------------------------------------------------------------------- #
def _load_run_day(run_dir: str) -> dict[str, list]:
    """panel/run_day.parquet を読む。無ければ build_panel で生成してから読む。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "panel", "run_day.parquet")
    if not os.path.exists(path):
        bp.build(run_dir)                      # 副作用: panel/ を生成(読み出し専用の派生)
    return pq.read_table(path).to_pydict()


def _cfg_seed(run_dir: str):
    """config.yaml から run.seed を読む(無ければ None)。"""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return None
    try:
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return (cfg.get("run", {}) or {}).get("seed")
    except Exception:
        return None


def _parse_cond_seed(run_name: str, cfg_seed):
    """run 名の末尾 _s<数字> を seed とみなし、残りを condition とする。
    末尾が無ければ condition=run 名全体・seed=config の seed。"""
    mt = re.search(r"_s(\d+)$", run_name)
    if mt:
        return run_name[:mt.start()], int(mt.group(1))
    return run_name, cfg_seed


def _r2_yext(run_dir: str):
    """measure を再利用して R²(traits→Y_external)を返す(算出不能なら None)。"""
    try:
        events = m.load_events(run_dir)
        agents = m.load_agents(run_dir)
        traits, names, _src = m.load_traits(run_dir, agents)
        feats = m.agent_features(events, agents, traits)
        res = m.r2_traits(feats, names)
        return res.get("external", {}).get("r2")
    except Exception:
        return None


def _mean_over_days(rd: dict[str, list], key: str, use_warmup: bool):
    """run_day の1指標を(非warmup)日で平均。全日 warmup のときは全日で平均。"""
    vals, vals_all = [], []
    for i in range(len(rd["day"])):
        v = rd.get(key, [None] * len(rd["day"]))[i]
        if v is None:
            continue
        vals_all.append(float(v))
        if use_warmup or not rd["warmup"][i]:
            vals.append(float(v))
    use = vals if vals else vals_all
    return sum(use) / len(use) if use else None


def _mean_ci(xs: list):
    """平均・95%CI(正規近似)・n・sd。n<2 は CI=None。"""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, None, 0, None
    mean = sum(xs) / n
    if n < 2:
        return mean, None, None, 1, None
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    half = 1.96 * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n, sd


# --------------------------------------------------------------------------- #
# ⑦ 統計的有意性の拡張(第30バッチ)= 効果量・置換検定・BH-FDR・t分布CI
# すべて純Python(scipy 禁止)・読み出し専用。docs/plans/analytics-suite.md(W1)・
# docs/research/analytics-methods.md(⑦)に従う。テストしやすいよう純関数で分離。
# --------------------------------------------------------------------------- #

# 両側95%(片側0.975)t 分位点。df=1..30 の内蔵表(scipy 非依存)。df≥31 は正規近似。
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t_crit(df: int) -> float:
    """両側95% CI 用の t 分位点(df=1..30 は表・それ以上は 1.96 の正規近似)。"""
    if df < 1:
        return None                                # type: ignore
    return _T_CRIT_95.get(df, 1.96)


def t_ci(xs: list):
    """平均・95%CI(t 分布・n<30 で正規近似より広い)・n・sd。n<2 は CI=None。
    n≥31 は t_crit が 1.96 を返すので _mean_ci(正規近似)と一致する。"""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, None, 0, None
    mean = sum(xs) / n
    if n < 2:
        return mean, None, None, 1, None
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    half = t_crit(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n, sd


def paired_cohens_d(diffs: list):
    """対応差の Cohen's d = mean(diff) / sd(diff)(ddof=1)。
    n<2 または sd=0(差が一定)は None(効果量が定義できない/発散)。"""
    ds = [d for d in diffs if d is not None]
    n = len(ds)
    if n < 2:
        return None
    mean = sum(ds) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in ds) / (n - 1))
    if sd == 0:
        return None
    return mean / sd


def cliffs_delta(xs: list, ys: list):
    """Cliff's δ = (#(y>x) − #(y<x)) / (n·m)。ys が大きいほど正(= cond B − cond A と整合)。
    範囲 [-1,1]。空群は None。"""
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if not xs or not ys:
        return None
    gt = lt = 0
    for x in xs:
        for y in ys:
            if y > x:
                gt += 1
            elif y < x:
                lt += 1
    return (gt - lt) / (len(xs) * len(ys))


def perm_test_paired(diffs: list, seed: int = 0, max_exact: int = 12,
                     n_random: int = 10000):
    """対応差の符号反転による両側置換検定 p。統計量 = |平均差|。
    n≤max_exact(既定12)は 2^n を全列挙、超えたら乱択 n_random 回(seed 固定・add-one)。
    n=0 は None。"""
    ds = [d for d in diffs if d is not None]
    n = len(ds)
    if n == 0:
        return None
    obs = abs(sum(ds) / n)
    eps = 1e-12
    if n <= max_exact:
        import itertools
        cnt = tot = 0
        for signs in itertools.product((1, -1), repeat=n):
            m = abs(sum(s * d for s, d in zip(signs, ds)) / n)
            tot += 1
            if m >= obs - eps:
                cnt += 1
        return cnt / tot                            # 観測(all +)を含むので add-one 不要
    import random
    rng = random.Random(seed)
    cnt = 1                                          # add-one: 観測自身を帰無側に含める(保守的)
    for _ in range(n_random):
        m = abs(sum((d if rng.random() < 0.5 else -d) for d in ds) / n)
        if m >= obs - eps:
            cnt += 1
    return cnt / (n_random + 1)


def bh_fdr(pvals: list):
    """Benjamini-Hochberg FDR 補正後 q(入力順に整列)。None はそのまま None を返す。
    q_(i) = min_{k≥i} ( p_(k)·m / k ) を単調化して 1 で頭打ち。"""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    q: list = [None] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])     # p 昇順
    prev = 1.0
    for rank in range(m, 0, -1):                     # 大きい順位から単調化
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q


def _fmt(v) -> str:
    if v is None:
        return "―"
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:
            return f"{v:.2e}"
        return f"{v:,.4g}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def analyze(run_dirs: list[str]) -> dict:
    runs = []
    combined: dict[str, list] = None            # type: ignore
    for rd_path in run_dirs:
        rd = _load_run_day(rd_path)
        name = os.path.basename(os.path.normpath(rd_path))
        cond, seed = _parse_cond_seed(name, _cfg_seed(rd_path))
        # 各ランの1指標=非warmup日平均。R² はラン全体で1値。
        per = {k: _mean_over_days(rd, k, use_warmup=False) for k in _METRIC_KEYS
               if k != "r2_traits_Yext"}
        per["r2_traits_Yext"] = _r2_yext(rd_path)
        runs.append({"run": name, "condition": cond, "seed": seed, "metrics": per})
        # 縦結合(run_day をそのまま積む)
        if combined is None:
            combined = {c: list(rd[c]) for c in rd}
        else:
            for c in combined:
                combined[c].extend(rd.get(c, [None] * len(rd["day"])))

    # ---- 条件ごとの CI(全ラン=条件 __all__ も出す)----
    conditions = sorted({r["condition"] for r in runs})
    ci_rows = []                                 # (metric, condition, mean, lo, hi, n, sd)
    groups = {"__all__": runs}
    for c in conditions:
        groups[c] = [r for r in runs if r["condition"] == c]
    for gk in ["__all__"] + conditions:
        grp = groups[gk]
        for k in _METRIC_KEYS:
            xs = [r["metrics"].get(k) for r in grp]
            mean, lo, hi, n, sd = _mean_ci(xs)          # 既存: 正規近似CI(後方互換・不変)
            _, tlo, thi, _, _ = t_ci(xs)                # 追加列: t分布CI(n<30 で正規より広い)
            ci_rows.append({"metric": k, "condition": gk, "mean": mean,
                            "ci_lo": lo, "ci_hi": hi, "n_runs": n, "sd": sd,
                            "ci_lo_t": tlo, "ci_hi_t": thi})

    # ---- 条件×同seed ペア比較(2条件が同一 seed を共有する場合)----
    pairs = []
    if len(conditions) == 2:
        ca, cb = conditions
        seeds_a = {r["seed"]: r for r in groups[ca] if r["seed"] is not None}
        seeds_b = {r["seed"]: r for r in groups[cb] if r["seed"] is not None}
        shared = sorted(set(seeds_a) & set(seeds_b))
        for k in _METRIC_KEYS:
            diffs, va_list, vb_list = [], [], []
            for s in shared:
                va = seeds_a[s]["metrics"].get(k)
                vb = seeds_b[s]["metrics"].get(k)
                if va is not None and vb is not None:
                    diffs.append(vb - va)          # cond B − cond A(同seed差)
                    va_list.append(va)
                    vb_list.append(vb)
            mean, lo, hi, n, sd = _mean_ci(diffs)          # 既存: 正規近似CI(不変)
            _, tlo, thi, _, _ = t_ci(diffs)                # 追加: t分布CI
            pairs.append({"metric": k, "cond_a": ca, "cond_b": cb,
                          "n_shared_seeds": n, "mean_diff": mean,
                          "ci_lo": lo, "ci_hi": hi,
                          # --- 第30バッチ追加(効果量・順位・置換p)---
                          "ci_lo_t": tlo, "ci_hi_t": thi,
                          "cohen_d": paired_cohens_d(diffs),
                          "cliff_delta": cliffs_delta(va_list, vb_list),
                          "perm_p": perm_test_paired(diffs, seed=0),
                          "fdr_q": None})            # 併記指標群に対し後段で BH-FDR を埋める
        # ペア比較で併記する指標群(perm_p)に BH-FDR 補正 → q を書き戻す
        qs = bh_fdr([p["perm_p"] for p in pairs])
        for p, q in zip(pairs, qs):
            p["fdr_q"] = q

    return {"runs": runs, "conditions": conditions, "ci_rows": ci_rows,
            "pairs": pairs, "combined": combined}


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def _write_combined(path: str, combined: dict[str, list]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pydict(combined, schema=bp._run_day_schema()),
                   path)


def _write_stats_parquet(path: str, ci_rows: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    # 既存列は不変・末尾に t分布CI(ci_lo_t/ci_hi_t)を追加(後方互換=新列の追加のみ)。
    cols = {c: [r.get(c) for r in ci_rows] for c in
            ("metric", "condition", "mean", "ci_lo", "ci_hi", "n_runs", "sd",
             "ci_lo_t", "ci_hi_t")}
    schema = pa.schema([
        pa.field("metric", pa.string()), pa.field("condition", pa.string()),
        pa.field("mean", pa.float64()), pa.field("ci_lo", pa.float64()),
        pa.field("ci_hi", pa.float64()), pa.field("n_runs", pa.int64()),
        pa.field("sd", pa.float64()),
        pa.field("ci_lo_t", pa.float64()), pa.field("ci_hi_t", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)


def _write_pairs_parquet(path: str, pairs: list) -> None:
    """第30バッチ新規: ペア比較の効果量・置換p・FDR を parquet 化(新ファイル=非破壊)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = ("metric", "cond_a", "cond_b", "n_shared_seeds", "mean_diff",
            "ci_lo", "ci_hi", "ci_lo_t", "ci_hi_t", "cohen_d", "cliff_delta",
            "perm_p", "fdr_q")
    cols = {c: [p.get(c) for p in pairs] for c in keys}
    schema = pa.schema([
        pa.field("metric", pa.string()), pa.field("cond_a", pa.string()),
        pa.field("cond_b", pa.string()), pa.field("n_shared_seeds", pa.int64()),
        pa.field("mean_diff", pa.float64()), pa.field("ci_lo", pa.float64()),
        pa.field("ci_hi", pa.float64()), pa.field("ci_lo_t", pa.float64()),
        pa.field("ci_hi_t", pa.float64()), pa.field("cohen_d", pa.float64()),
        pa.field("cliff_delta", pa.float64()), pa.field("perm_p", pa.float64()),
        pa.field("fdr_q", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)


def render(res: dict) -> str:
    L: list[str] = []
    L.append("# パネル統計(シード横断 平均±95%CI)\n")
    L.append(f"- 対象ラン: {len(res['runs'])} 本 / 条件: "
             f"{', '.join(res['conditions'])}")
    L.append("- CI は正規近似(mean ± 1.96·sd/√n)。n が小さいと粗い近似(≥5 seed が主張の"
             "最低ライン=data-strategy.md)。指標は各ランの非warmup日平均を1標本とする。")
    L.append("")
    L.append("> **p 値の限界(必読)**: シミュレーションの p 値はラン数を増やせば任意に小さくできる"
             "(些少な効果でも有意になり得る)。**効果量(Cohen's d・Cliff's δ)と CI 幅を主・p を従**"
             "として、実質的有意性で判断する(JASSS の ABM 出力解析標準)。多重比較は BH-FDR で補正済み。\n")

    L.append("## ラン一覧")
    L.append("| run | 条件 | seed |")
    L.append("|---|---|---|")
    for r in res["runs"]:
        L.append(f"| {r['run']} | {r['condition']} | {r['seed']} |")
    L.append("")

    # 全ラン CI
    L.append("## 全ラン横断の平均±95%CI")
    L.append("| 指標 | 平均 | 95%CI下 | 95%CI上 | n | sd |")
    L.append("|---|---|---|---|---|---|")
    by = {(r["metric"], r["condition"]): r for r in res["ci_rows"]}
    for k in _METRIC_KEYS:
        r = by.get((k, "__all__"))
        if not r:
            continue
        L.append(f"| {_LABEL[k]} | {_fmt(r['mean'])} | {_fmt(r['ci_lo'])} | "
                 f"{_fmt(r['ci_hi'])} | {r['n_runs']} | {_fmt(r['sd'])} |")
    L.append("")

    # 条件別 CI(条件が2つ以上のとき)
    if len(res["conditions"]) >= 2:
        L.append("## 条件別の平均(±95%CI)")
        L.append("| 指標 | " + " | ".join(res["conditions"]) + " |")
        L.append("|---|" + "---|" * len(res["conditions"]))
        for k in _METRIC_KEYS:
            cells = []
            for c in res["conditions"]:
                r = by.get((k, c))
                if r and r["mean"] is not None:
                    ci = (f" (±{_fmt((r['ci_hi'] - r['mean']))})"
                          if r["ci_hi"] is not None else "")
                    cells.append(f"{_fmt(r['mean'])}{ci}")
                else:
                    cells.append("―")
            L.append(f"| {_LABEL[k]} | " + " | ".join(cells) + " |")
        L.append("")

    # ペア比較
    if res["pairs"]:
        ca, cb = res["conditions"]
        L.append(f"## 条件×同seed ペア比較(差 = {cb} − {ca}・分散削減)")
        L.append("| 指標 | 平均差 | 95%CI下 | 95%CI上 | 共有seed数 |")
        L.append("|---|---|---|---|---|")
        for p in res["pairs"]:
            L.append(f"| {_LABEL[p['metric']]} | {_fmt(p['mean_diff'])} | "
                     f"{_fmt(p['ci_lo'])} | {_fmt(p['ci_hi'])} | "
                     f"{p['n_shared_seeds']} |")
        L.append("")

        # 効果量・順位・置換p・FDR(⑦拡張・第30バッチ)
        L.append(f"## 効果量・置換検定・FDR(差 = {cb} − {ca})")
        L.append("- **主指標**: paired Cohen's d(差の標準化)/ Cliff's δ(順位ベース・[-1,1])。"
                 "**従指標**: 置換 p(符号反転・n≤12 は全列挙)/ BH-FDR 補正 q。t分布CI は n<30 用。")
        L.append("| 指標 | Cohen's d | Cliff's δ | 置換p | FDR q | 95%CI(t)下 | 上 |")
        L.append("|---|---|---|---|---|---|---|")
        for p in res["pairs"]:
            L.append(f"| {_LABEL[p['metric']]} | {_fmt(p.get('cohen_d'))} | "
                     f"{_fmt(p.get('cliff_delta'))} | {_fmt(p.get('perm_p'))} | "
                     f"{_fmt(p.get('fdr_q'))} | {_fmt(p.get('ci_lo_t'))} | "
                     f"{_fmt(p.get('ci_hi_t'))} |")
        L.append("")

    return "\n".join(L)


def run_stats(run_dirs: list[str], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    res = analyze(run_dirs)
    _write_combined(os.path.join(out_dir, "panel_combined.parquet"),
                    res["combined"])
    _write_stats_parquet(os.path.join(out_dir, "panel_stats.parquet"),
                         res["ci_rows"])
    if res["pairs"]:                                # ペアがある時だけ新ファイルを出す(非破壊)
        _write_pairs_parquet(os.path.join(out_dir, "panel_pairs.parquet"),
                             res["pairs"])
    md = render(res)
    with open(os.path.join(out_dir, "panel_stats.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    return {"out_dir": out_dir, "n_runs": len(res["runs"]),
            "conditions": res["conditions"], "md": md}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="stage2: 複数ランのパネル統計(シード横断 CI + ペア比較)")
    ap.add_argument("run_dirs", nargs="+", help="例: runs/A runs/B ...")
    ap.add_argument("--out", default=os.path.join("experiments", "panel_stats"),
                    help="出力先(既定 experiments/panel_stats)")
    args = ap.parse_args()
    for d in args.run_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"not a directory: {d}")
    info = run_stats(args.run_dirs, args.out)
    print(info["md"])
    print(f"\n[panel_stats] {info['n_runs']} runs -> {info['out_dir']}")


if __name__ == "__main__":
    main()
