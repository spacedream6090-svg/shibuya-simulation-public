#!/usr/bin/env python
"""運/実力分解(第51バッチ・読み出し専用)。

    python scripts/analyze_luck.py --run u51_audit_80a1d
    python scripts/analyze_luck.py --run u51_audit_80a1d --out luck/

外部指摘「100%予測可能な世界ではダイナミズムが失われる」への回答その2=分解。
住民の成果(収入・関係数・評判)を **実力由来(本人属性=traits+安定属性)** と
**運由来(偶発イベントへの曝露)** に分散分解して観測する。断定はしない=すべて「観測」。

研究の核 = 「成果 Y を traits で回帰した R²」(src/society/observer/measure.py の
load_traits / r2_traits と同じ流儀)。ここではそれを 3 段の階層回帰へ拡張する:
  skill モデル : Y ~ traits + age + occupation(実力側)              → R²_skill
  full モデル  : Y ~ traits + age + occupation + luck_exposure(+運) → R²_full
  luck 単独    : Y ~ luck_exposure                                    → R²_luck_only
分散分解: 実力 R²_skill / 運の追加 ΔR²_luck = R²_full − R²_skill / 残差 1 − R²_full。

運側の説明変数 = **audit_uncertainty.LUCK_KINDS の対応表を共有 import** した偶発イベント曝露の
per-agent 件数。**将来 kind を足すのは LUCK_KINDS に 1 行**(第54バッチ=chance_event)。

OLS は numpy.linalg.lstsq(measure.r2_traits と同流儀)。多重共線・小標本の限界は出力と docs に
正直に注記する。依存は 標準ライブラリ + pyarrow + numpy。pandas / duckdb 不使用・シム本体は不変。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")
sys.path.insert(0, _HERE)                       # scripts/ は package ではない
sys.path.insert(0, os.path.join(_ROOT, "src"))  # society.observer.measure(研究資産)

import audit_uncertainty as au                  # noqa: E402  共有: LUCK_KINDS / attribute_chances
from society.observer import measure            # noqa: E402  研究資産: load_traits(OLS 流儀)

STEPS_PER_DAY = 144

# 実力側 trait の許可リスト(traits.json には money/wage/visitor/part_time も入るため、
# 成果への漏れ=リークを避けて安定な人格/傾性因子のみ採用する)。
SKILL_TRAITS = (
    "internal_locus", "nfc", "risk_tolerance", "drive_threshold", "fire_weight",
    "efficacy", "ownership", "world_change", "need_for_cognition",
)
# 成果側に混ぜてはいけない列(実力側から明示排除=リーク防止の安全弁)。
_TRAIT_BLOCKLIST = {"money", "wage", "visitor", "part_time", "account", "balance", "id"}


# --------------------------------------------------------------------------- #
# 成果(被説明変数)の集計
# --------------------------------------------------------------------------- #
# 収入源: kind -> payload の金額キー(本人 agent_id へ帰属)。
_INCOME_SRC = {"wage": "amount", "venture_sale": "amount", "deliver": "fare"}


def agent_outcomes(events: list[dict], residents: set[int]) -> dict[int, dict]:
    """住民ごとの成果 {income, relations, reputation} を L1 から集計する。

      income     : wage(給与/バイト/gig/退職金)+ venture_sale(出店売上)+ deliver(配達 gig)の総額。
      relations  : 会話で結びついた distinct 相手数(speak hearers / hear speaker / dm to。
                   measure.agent_features の c_new_relations と同定義)+ relation_tier 到達相手も合流。
      reputation : reputation_update の正の増分(new-old)合計 + 受けた SNS いいね数(評判/地位の代理)。
    """
    income: Counter = Counter()
    partners: dict[int, set] = defaultdict(set)
    rep_gain: dict[int, float] = defaultdict(float)
    likes: Counter = Counter()

    for e in events:
        aid, k, p = e["agent"], e["kind"], e["payload"]
        if aid is None or aid < 0:
            # deliver は agent_id=-1(店側)になりうる→ courier へ回す(下で処理)
            if k == "deliver":
                cr = p.get("courier")
                if isinstance(cr, int) and cr >= 0:
                    income[cr] += _f(p.get("fare"))
            continue
        if k in _INCOME_SRC:
            income[aid] += _f(p.get(_INCOME_SRC[k]))
        elif k == "speak":
            for h in p.get("hearers", []) or []:
                if isinstance(h, int) and h >= 0 and h != aid:
                    partners[aid].add(h)
                    partners[h].add(aid)
        elif k == "hear":
            sp = p.get("speaker")
            if isinstance(sp, int) and sp >= 0 and sp != aid:
                partners[aid].add(sp)
                partners[sp].add(aid)
        elif k == "dm":
            to = p.get("to")
            if isinstance(to, int) and to >= 0 and to != aid:
                partners[aid].add(to)
                partners[to].add(aid)
        elif k == "relation_tier":
            other = p.get("other")
            if isinstance(other, int) and other >= 0 and other != aid:
                partners[aid].add(other)
        elif k == "reputation_update":
            rep_gain[aid] += max(0.0, _f(p.get("new")) - _f(p.get("old")))
        elif k == "sns_like":
            au_ = p.get("author")
            if isinstance(au_, int) and au_ >= 0:
                likes[au_] += 1

    out: dict[int, dict] = {}
    for aid in sorted(residents):
        out[aid] = {
            "income": round(income.get(aid, 0.0), 2),
            "relations": len(partners.get(aid, ())),
            "reputation": round(rep_gain.get(aid, 0.0) + likes.get(aid, 0), 4),
        }
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# 実力側の説明変数(traits + age + occupation)
# --------------------------------------------------------------------------- #
def skill_features(agents: dict[int, dict], traits_by_id: dict[int, dict],
                   residents: set[int], *, occ_min_count: int = 3):
    """住民ごとの実力側特徴 {agent: {col: value}} と列名リストを返す。

    traits(許可リスト ∩ 実在) + age + occupation ダミー(頻度 occ_min_count 以上の職業のみ・
    最頻を基準に落とす=決定論)。小標本での過剰パラメータを避けるための刈り込み。
    """
    present_traits = [t for t in SKILL_TRAITS
                      if t not in _TRAIT_BLOCKLIST
                      and any(t in (traits_by_id.get(a) or {}) for a in residents)]

    occ_counts: Counter = Counter()
    for aid in residents:
        occ = (agents.get(aid, {}) or {}).get("occupation")
        if occ:
            occ_counts[occ] += 1
    frequent = [o for o, c in occ_counts.items() if c >= occ_min_count]
    frequent.sort(key=lambda o: (-occ_counts[o], o))
    baseline = frequent[0] if frequent else None            # 最頻=基準(ダミー化しない)
    occ_dummies = [o for o in frequent if o != baseline]
    occ_cols = [f"occ={o}" for o in occ_dummies]

    cols = list(present_traits) + ["age"] + occ_cols
    feats: dict[int, dict] = {}
    for aid in sorted(residents):
        tr = traits_by_id.get(aid) or {}
        meta = agents.get(aid, {}) or {}
        row: dict[str, float] = {}
        ok = True
        for t in present_traits:
            v = tr.get(t)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                ok = False
                break
            row[t] = float(v)
        if not ok:
            continue                                        # trait 欠損の個体は回帰から外す
        age = meta.get("age")
        row["age"] = float(age) if isinstance(age, (int, float)) else 0.0
        occ = meta.get("occupation")
        for o, col in zip(occ_dummies, occ_cols):
            row[col] = 1.0 if occ == o else 0.0
        feats[aid] = row
    return feats, cols, {"traits": present_traits, "occ_baseline": baseline,
                         "occ_dummies": occ_dummies}


# --------------------------------------------------------------------------- #
# OLS(measure.r2_traits と同じ流儀: numpy.linalg.lstsq・切片あり・adj R²)
# --------------------------------------------------------------------------- #
def ols(rows: list[dict], xcols: list[str], ycol: str) -> dict:
    """Y ~ xcols の OLS。返り値 {r2, adj_r2, n, p, coef}。n<=p+1 は r2=None。"""
    xcols = list(xcols)
    good = [r for r in rows
            if isinstance(r.get(ycol), (int, float))
            and all(isinstance(r.get(c), (int, float)) for c in xcols)]
    n = len(good)
    p = len(xcols)
    if n <= p + 1:
        return {"r2": None, "adj_r2": None, "n": n, "p": p}
    x = np.array([[1.0] + [float(r[c]) for c in xcols] for r in good])
    y = np.array([float(r[ycol]) for r in good])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:                                       # Y が定数=分解不能
        return {"r2": None, "adj_r2": None, "n": n, "p": p, "note": "Y は定数(分散0)"}
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else None
    return {"r2": round(r2, 4), "adj_r2": (round(adj, 4) if adj is not None else None),
            "n": n, "p": p,
            "coef": {c: round(b, 6) for c, b in zip(["intercept"] + xcols, beta.tolist())}}


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson 相関(多重共線の目安。決定論・stdlib のみ)。"""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return round(num / den, 4) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# 分解本体
# --------------------------------------------------------------------------- #
def decompose(events: list[dict], agents: dict[int, dict],
              traits_by_id: dict[int, dict]) -> dict:
    """成果ごとの 実力/運 分散分解 + 住民別散布データ + 共線・小標本の診断を返す。"""
    residents = {aid for aid, m in agents.items() if not m.get("visitor")}
    if not residents:                                       # visitor 情報が無い旧ラン等
        residents = {aid for aid in traits_by_id if aid >= 0}

    outcomes = agent_outcomes(events, residents)
    sfeat, scols, sinfo = skill_features(agents, traits_by_id, residents)
    attr = au.attribute_chances(events, agents)
    per_agent = attr["per_agent"]                           # {agent: Counter(category->count)}

    # 統合行(実力側特徴が揃った住民のみ)。運側 = 偶発曝露の総件数。
    rows: list[dict] = []
    for aid in sorted(sfeat):
        row = dict(sfeat[aid])
        row["id"] = aid
        exposure = per_agent.get(aid, Counter())
        row["luck_exposure"] = int(sum(exposure.values()))
        for cat in sorted(au.LUCK_KINDS[k]["category"] for k in au.LUCK_KINDS):
            row[f"luck_{cat}"] = int(exposure.get(cat, 0))
        for oc, ov in outcomes.get(aid, {}).items():
            row[oc] = ov
        rows.append(row)

    luck_cols = ["luck_exposure"]
    results: dict[str, dict] = {}
    for outcome in ("income", "relations", "reputation"):
        skill = ols(rows, scols, outcome)
        full = ols(rows, scols + luck_cols, outcome)
        luck_only = ols(rows, luck_cols, outcome)
        delta = None
        if skill.get("r2") is not None and full.get("r2") is not None:
            delta = round(full["r2"] - skill["r2"], 4)
        residual = (round(1.0 - full["r2"], 4)
                    if full.get("r2") is not None else None)
        results[outcome] = {
            "skill": skill, "full": full, "luck_only": luck_only,
            "delta_r2_luck": delta, "residual_after_full": residual,
        }

    # 共線診断: luck_exposure と各実力列の Pearson |r| の最大値。
    collin = {}
    if rows:
        lx = [float(r["luck_exposure"]) for r in rows]
        for c in scols:
            collin[c] = _pearson(lx, [float(r[c]) for r in rows])
    max_abs_collin = max((abs(v) for v in collin.values()), default=0.0)

    scatter = [{
        "id": r["id"],
        "occupation": (agents.get(r["id"], {}) or {}).get("occupation"),
        "age": (agents.get(r["id"], {}) or {}).get("age"),
        "income": r.get("income"), "relations": r.get("relations"),
        "reputation": r.get("reputation"),
        "luck_exposure": r["luck_exposure"],
        "luck_by_category": {c.split("luck_", 1)[1]: r[c] for c in r
                             if c.startswith("luck_") and c != "luck_exposure"},
        "traits": {t: r.get(t) for t in sinfo["traits"]},
    } for r in rows]

    return {
        "n_residents": len(residents),
        "n_modeled": len(rows),
        "skill_columns": scols,
        "skill_info": sinfo,
        "luck_columns": luck_cols,
        "outcomes": results,
        "collinearity_luck_vs_skill": collin,
        "max_abs_collinearity": round(max_abs_collin, 4),
        "scatter": scatter,
        "limitations": _limitations(len(rows), scols, max_abs_collin),
    }


def _limitations(n: int, scols: list[str], max_collin: float) -> list[str]:
    notes = [
        "観測であって因果ではない: R² は説明分散であり『実力/運が成果を生む』因果効果ではない。",
        f"小標本: モデル対象 n={n}・説明変数 p={len(scols)}。n が p に近いほど R² は過大・"
        "adj_R² を優先し、単一ランの数値を一般化しない(seed 反復が必要)。",
        "運の内生性: 寄り道/中断/退屈探索など agent スコープの曝露は本人の行動選択と相関しうる"
        "(=純外生ショックではない)。純外生は世界ショック(disaster/transit_delay/infra_outage)側。",
        f"多重共線: luck_exposure と実力列の |Pearson| 最大={round(max_collin, 4)}。"
        "高いほど ΔR²_luck の帰属が不安定(実力と運の寄与が分離しにくい)。",
        "mock LLM ラン: 発話内容は定型で relations/reputation の変動が実 LLM より小さい="
        "R² の水準はバックエンド依存。傾向の読み取りに留める。",
        "1 日ラン: 収入/関係は日次で頭打ち。世界ショックは日次発火のため単日では 0 になりうる"
        "(別途 shock probe ランで world スコープの帰属経路を確認)。",
    ]
    return notes


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, out_dir: str) -> dict:
    """ラン1本を分解し、luck/decomposition.json を書く。"""
    run_name = os.path.basename(os.path.normpath(run_dir))
    events = au.load_events(run_dir)
    agents = au.load_agents(run_dir)
    traits_by_id, trait_names, tsource = measure.load_traits(run_dir)
    res = decompose(events, agents, traits_by_id)
    res["run"] = run_name
    res["traits_source"] = tsource
    res["traits_available"] = trait_names

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "decomposition.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, sort_keys=True)
    res["paths"] = {"json": json_path}
    return res


def render_text(res: dict) -> str:
    lines: list[str] = []
    lines.append(f"# 運/実力分解: {res['run']}")
    lines.append(f"住民 {res['n_residents']} / モデル対象 {res['n_modeled']} / "
                 f"traits 源 {res['traits_source']} {res['traits_available']}")
    lines.append(f"実力列({len(res['skill_columns'])}): {res['skill_columns']}")
    lines.append(f"最大 |共線|(運×実力): {res['max_abs_collinearity']}")
    lines.append("")
    for oc, r in res["outcomes"].items():
        sk = r["skill"].get("r2")
        fu = r["full"].get("r2")
        lo = r["luck_only"].get("r2")
        lines.append(f"## {oc}")
        lines.append(f"  実力 R²={sk}  full R²={fu}  運単独 R²={lo}")
        lines.append(f"  運の追加 ΔR²={r['delta_r2_luck']}  残差={r['residual_after_full']}  "
                     f"(n={r['skill'].get('n')})")
    lines.append("")
    lines.append("## 限界(要約)")
    for note in res["limitations"]:
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def _pick_run(arg_run: str | None) -> str:
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isfile(os.path.join(d, "l1_events.parquet")):
            raise SystemExit(f"[luck] l1_events.parquet が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[luck] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[luck] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _resolve_out(run_dir: str, arg_out: str | None) -> str:
    if not arg_out:
        return os.path.join(run_dir, "luck")
    return arg_out if os.path.isabs(arg_out) else os.path.join(run_dir, arg_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="運/実力分解(読み出し専用)")
    ap.add_argument("--run", default=None,
                    help="ラン名(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None,
                    help="出力ディレクトリ(既定: <run>/luck)。相対はラン基準")
    args = ap.parse_args()

    run_dir = _pick_run(args.run)
    out_dir = _resolve_out(run_dir, args.out)
    res = analyze(run_dir, out_dir)
    print(render_text(res))
    print(f"[luck] -> {res['paths']['json']}")


if __name__ == "__main__":
    main()
