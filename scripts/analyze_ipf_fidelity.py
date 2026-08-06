#!/usr/bin/env python
"""S-14 IPF 合成人口の周辺分布再現誤差(SRMSE / TAE / CE)(SV2-B・読み取り専用)。

    python scripts/analyze_ipf_fidelity.py
    python scripts/analyze_ipf_fidelity.py --pool data/persona_pool/L1
    python scripts/analyze_ipf_fidelity.py --run runs/<name>       # agents.json を実績に使う

■ 何を測るか
  IPF の**目標周辺分布**(`data/shibuya_population.json` の `age_bands` / `gender` /
  `occupations`)と、**生成された合成人口の実績**(`data/persona_pool/L1/part-*.jsonl`、
  または任意ランの `agents.json`)を突合し、合成人口の標準的な適合度指標を出す。

■ 指標(Voas, D. & Williamson, P. (2001). *Evaluating Goodness-of-Fit Measures for
  Synthetic Microdata.* Geographical and Environmental Modelling 5(2):177–200)
    TAE   = Σ_cells |obs − exp|                     総絶対誤差
    CE    = TAE / 2                                  分類誤差(誤分類された個体数)
    SRMSE = sqrt( Σ (obs − exp)² / n_cells ) / (Σ exp / n_cells)
            標準化二乗平均平方根誤差。**0 = 完全一致**。セル平均で割るので表間で比較できる
  ★同論文は**疎な表ではゼロセルが指標を壊す**と注意している。本スクリプトは
  ゼロセル数を必ず併記する(隠さない)。

■ ★実装状況の実査結果(なぜ後付けで作れるか)
  - IPF 本体は `scripts/build_persona_pool.py` の `_ipf_joint(pop, iters=60)`(決定論)。
  - **誤差を出す処理はどこにも無い。** `meta.json` に入るのは層別件数(`layer_targets` /
    `layer_counts`)だけで、年齢・性別・職業の周辺分布誤差は記録されていない。
  - 近いものは `tests/test_persona_pool.py` の周辺分布 assert(許容誤差 0.06)だが、
    **合否だけ出して数値を出さない**。
  → 本スクリプトは `part-*.jsonl` の各行が持つ `age` / `gender` / `occupation` と
    目標 share の突合だけで完全に後付けできる。

■ ★`run_manifest` に足さない(設計判断)
  `src/society/observer/manifest.py` に載せると `src/` を触る = **本選フリーズの対象**になる。
  受け皿は `docs/plans/observation-report-template.md` §4.4(既に空欄として存在)である。

■ 正直な注記
  - 目標 share 自体が **暫定値**である(`data/shibuya_population.json` の `meta.note`:
    年齢 5 歳階級 × 性別と職業大分類の区実数は未確認)。
    **したがって「誤差が小さい = 現実に近い」ではなく「誤差が小さい = 指定した目標に忠実」**
    である。この区別を報告に必ず書く。本スクリプトは目標側の `status` / `note` を出力に含める。
  - 目標に無いカテゴリ(例: 議員)と、帯の外の年齢は **`not_in_target` として別掲**する。
    黙って捨てると分母がずれる。

■ R1 ドクトリン
  `src/` と `conf/` に 1 バイトも触らない。決定論: カテゴリはソート順で出力する。
  データが無いときは捏造せず明示終了する。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_POP = os.path.join(_ROOT, "data", "shibuya_population.json")
DEFAULT_POOL = os.path.join(_ROOT, "data", "persona_pool", "L1")
NOT_IN_TARGET = "_not_in_target"


# --------------------------------------------------------------------------- #
# 純関数(単体テストの対象)
# --------------------------------------------------------------------------- #
def _r(x, nd: int = 6):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def fit_measures(observed: dict, target_share: dict, n_total: int) -> dict:
    """Voas & Williamson (2001) の TAE / CE / SRMSE をセル別誤差つきで返す。

    observed      : {category: count}(目標に無いカテゴリを含んでよい)
    target_share  : {category: share}(和が 1 でなくても正規化する)
    n_total       : 実績の総数(期待度数の基準)
    """
    cats = sorted(set(target_share))
    ssum = sum(float(v) for v in target_share.values())
    if ssum <= 0 or n_total <= 0:
        return {"cells": [], "TAE": None, "CE": None, "SRMSE": None,
                "n_cells": 0, "n_zero_cells": 0,
                "n_not_in_target": int(sum(c for k, c in observed.items()
                                           if k not in target_share))}
    rows = []
    sq = 0.0
    tae = 0.0
    exp_sum = 0.0
    zero_cells = 0
    for c in cats:
        exp = n_total * (float(target_share[c]) / ssum)
        obs = float(observed.get(c, 0))
        diff = obs - exp
        tae += abs(diff)
        sq += diff * diff
        exp_sum += exp
        if obs == 0 or exp == 0:
            zero_cells += 1
        rows.append({"category": c,
                     "target_share": _r(float(target_share[c]) / ssum, 6),
                     "observed_share": _r(obs / n_total, 6),
                     "target_count": _r(exp, 3), "observed_count": int(obs),
                     "diff": _r(diff, 3),
                     "pct_error": _r(100.0 * diff / exp, 3) if exp > 0 else None})
    n_cells = len(cats)
    rmse = math.sqrt(sq / n_cells) if n_cells else None
    mean_exp = exp_sum / n_cells if n_cells else None
    srmse = (rmse / mean_exp) if (rmse is not None and mean_exp) else None
    extra = sorted(k for k in observed if k not in target_share)
    return {"cells": rows, "TAE": _r(tae, 3), "CE": _r(tae / 2.0, 3),
            "SRMSE": _r(srmse, 6), "n_cells": n_cells,
            "n_zero_cells": int(zero_cells),
            "n_not_in_target": int(sum(observed[k] for k in extra)),
            "not_in_target_categories": extra}


def age_band_label(band) -> str:
    return f"{int(band[0])}-{int(band[1])}"


def assign_age_band(age, bands) -> str:
    """年齢 → 帯ラベル。どの帯にも入らなければ `_not_in_target`(黙って捨てない)。"""
    try:
        a = int(age)
    except (TypeError, ValueError):
        return NOT_IN_TARGET
    for b in bands:
        lo, hi = int(b["band"][0]), int(b["band"][1])
        if lo <= a <= hi:
            return age_band_label(b["band"])
    return NOT_IN_TARGET


def tally(records: list[dict], pop: dict) -> dict:
    """レコード列(age / gender / occupation を持つ dict)を 3 属性で集計する。"""
    bands = pop.get("age_bands", []) or []
    occ_names = {str(o["name"]) for o in (pop.get("occupations", []) or [])}
    genders = set(pop.get("gender", {}) or {})
    age_c: dict[str, int] = defaultdict(int)
    gen_c: dict[str, int] = defaultdict(int)
    occ_c: dict[str, int] = defaultdict(int)
    for rec in records:
        age_c[assign_age_band(rec.get("age"), bands)] += 1
        g = str(rec.get("gender", "") or "")
        gen_c[g if g in genders else NOT_IN_TARGET] += 1
        o = str(rec.get("occupation", "") or "")
        occ_c[o if o in occ_names else NOT_IN_TARGET] += 1
    return {"age_band": dict(age_c), "gender": dict(gen_c), "occupation": dict(occ_c)}


def target_shares(pop: dict) -> dict:
    return {
        "age_band": {age_band_label(b["band"]): float(b["share"])
                     for b in (pop.get("age_bands", []) or [])},
        "gender": {str(k): float(v) for k, v in (pop.get("gender", {}) or {}).items()},
        "occupation": {str(o["name"]): float(o["share"])
                       for o in (pop.get("occupations", []) or [])},
    }


# --------------------------------------------------------------------------- #
# ローダ
# --------------------------------------------------------------------------- #
def load_pool(pool_dir: str) -> list[dict]:
    """persona_pool の part-*.jsonl を読み、age/gender/occupation だけを取り出す。"""
    out: list[dict] = []
    for path in sorted(glob.glob(os.path.join(pool_dir, "part-*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append({"age": rec.get("age"), "gender": rec.get("gender"),
                            "occupation": rec.get("occupation")})
    return out


def load_run_agents(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [{"age": a.get("age"), "gender": a.get("gender"),
             "occupation": a.get("occupation")} for a in data
            if isinstance(a, dict)]


def analyze(records: list[dict], pop: dict, source: str) -> dict:
    n = len(records)
    obs = tally(records, pop)
    tgt = target_shares(pop)
    attrs = {}
    for key in ("age_band", "gender", "occupation"):
        attrs[key] = fit_measures(obs[key], tgt[key], n)
    return {
        "source": source,
        "n_records": n,
        "status": "OK" if n else "NO_DATA",
        "target_meta": {"status": (pop.get("meta", {}) or {}).get("status"),
                        "note": (pop.get("meta", {}) or {}).get("note")},
        "attributes": attrs,
        "caveat": ("目標 share 自体が暫定値である。誤差が小さいことは"
                   "「現実に近い」ではなく「指定した目標に忠実」を意味する。"),
    }


def render(res: dict) -> str:
    L: list[str] = []
    L.append("# S-14 IPF 合成人口の周辺分布再現誤差")
    L.append("")
    L.append("> 指標は Voas & Williamson (2001):"
             " **TAE**(総絶対誤差)/ **CE = TAE/2**(誤分類個体数)/"
             " **SRMSE**(0 = 完全一致)。")
    L.append(f"> ★{res['caveat']}")
    L.append("")
    L.append(f"- 実績の出所: `{res['source']}`  件数: **{res['n_records']}**")
    tm = res.get("target_meta", {})
    if tm.get("status"):
        L.append(f"- 目標側の status: `{tm['status']}`")
    L.append("")
    if res.get("status") != "OK":
        L.append("**NO_DATA** — 実績レコードが 0 件。捏造しない。")
        return "\n".join(L) + "\n"
    L.append("## 属性ごとの適合度")
    L.append("")
    L.append("| 属性 | セル数 | ゼロセル | TAE | CE | **SRMSE** | 目標外の件数 |")
    L.append("|---|---|---|---|---|---|---|")
    jp = {"age_band": "年齢帯", "gender": "性別", "occupation": "職業"}
    for key in ("age_band", "gender", "occupation"):
        a = res["attributes"][key]
        L.append(f"| {jp[key]} | {a['n_cells']} | {a['n_zero_cells']} | {a['TAE']} "
                 f"| {a['CE']} | **{a['SRMSE']}** | {a['n_not_in_target']} |")
    L.append("")
    for key in ("age_band", "gender", "occupation"):
        a = res["attributes"][key]
        L.append(f"### {jp[key]}")
        L.append("")
        L.append("| カテゴリ | 目標 share | 実績 share | 目標件数 | 実績件数 | 差 | %誤差 |")
        L.append("|---|---|---|---|---|---|---|")
        for c in a["cells"]:
            L.append(f"| {c['category']} | {c['target_share']} | {c['observed_share']} "
                     f"| {c['target_count']} | {c['observed_count']} | {c['diff']} "
                     f"| {c['pct_error']} |")
        if a["not_in_target_categories"]:
            L.append("")
            L.append(f"- **目標に無いカテゴリ**(別掲・黙って捨てない): "
                     f"{', '.join(a['not_in_target_categories'])} "
                     f"= {a['n_not_in_target']} 件")
        L.append("")
    L.append("> **ゼロセルの注意**(Voas & Williamson 2001): 疎な表ではゼロセルが指標を壊す。")
    L.append("> 上表の「ゼロセル」列が 0 でないときは SRMSE を単独で読まない。")
    L.append("")
    L.append("> **貼り付け先**: `docs/plans/observation-report-template.md` §4.4"
             "(Initialization / ODD ⑤)。`run_manifest` には足さない(`src/` を触るため)。")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S-14 IPF 再現誤差(読み取り専用)")
    ap.add_argument("--pop", default=DEFAULT_POP, help="目標周辺分布 JSON")
    ap.add_argument("--pool", default=None, help="persona_pool の層ディレクトリ")
    ap.add_argument("--run", default=None, help="ランの agents.json を実績に使う")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 experiments/ipf_fidelity または <run>/analysis)")
    a = ap.parse_args(argv)

    if not os.path.exists(a.pop):
        print(f"[analyze_ipf_fidelity] 目標分布が無い: {a.pop}", file=sys.stderr)
        return 2
    with open(a.pop, encoding="utf-8") as fh:
        pop = json.load(fh)

    if a.run:
        if not os.path.isdir(a.run):
            print(f"[analyze_ipf_fidelity] ランが無い: {a.run}", file=sys.stderr)
            return 2
        records = load_run_agents(a.run)
        source = os.path.join(a.run, "agents.json")
        out = a.out or os.path.join(a.run, "analysis")
    else:
        pool = a.pool or DEFAULT_POOL
        if not os.path.isdir(pool):
            print(f"[analyze_ipf_fidelity] プールが無い: {pool}", file=sys.stderr)
            return 2
        records = load_pool(pool)
        source = pool
        out = a.out or os.path.join(_ROOT, "experiments", "ipf_fidelity")
    if not records:
        print(f"[analyze_ipf_fidelity] レコードが 0 件: {source}。捏造せず終了する。",
              file=sys.stderr)
        return 3

    res = analyze(records, pop, os.path.relpath(source, _ROOT).replace("\\", "/"))
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, "ipf_fidelity.json")
    mp = os.path.join(out, "ipf_fidelity.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, sort_keys=True, indent=2)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    s = {k: res["attributes"][k]["SRMSE"] for k in sorted(res["attributes"])}
    print(f"[analyze_ipf_fidelity] n={res['n_records']} SRMSE={s}")
    print(f"  -> {jp}")
    print(f"  -> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
