#!/usr/bin/env python
"""⑥介入前後の比較(between-run CRN + within-run DiD・分析スイート W2・第31バッチ)。

    python scripts/compare_runs.py --a runs/X1 runs/X2 ... --b runs/Y1 runs/Y2 ... [--out ...]

docs/plans/analytics-suite.md(W2)・docs/research/analytics-methods.md(⑥)に沿う
**L1/派生 読み出し専用** スクリプト。sim 本体・schema・conf は変更しない。統計は
scripts/panel_stats.py の純関数(paired Cohen's d・Cliff's δ・置換検定・BH-FDR・t分布CI)を
**import して流用**(再実装しない)。

やること:
  1. between-run(CRN ペア比較): 同 seed のペア(--a[i] と --b[i]・seed 順で対応)ごとに、
     各ランのスカラ指標(l2_metrics.parquet 各列の step 平均・panel/run_day.parquet 各列の
     非warmup日平均)を突き合わせ、差 = B − A の分布を置換検定・効果量で検定する。
     両側にそろって存在する指標だけを対象とし、欠ける指標は「データ不足」と明記する。
  2. CRN 健全性チェック: ペアの L1 イベント列を介入 step まで(介入が無ければ先頭から
     --crn-max-steps まで)突合し、最初の分岐 step・一致率を報告。実行パス破れ(=ペア比較の
     前提崩壊)を警告する(本シミュは rng_stream 分離設計だが、条件差そのものが分岐を生む)。
  3. within-run DiD: config の scenario shock(scenario_shock イベントの介入 step)がある
     ランで (after−before)_処置 − (after−before)_対照 を出す。介入 step が特定できない
     ランは**明示的にスキップ**(黙って 0 を出さない)。

出力(--out 既定 runs/_compare/<A条件>_vs_<B条件>):
  compare_report.md    … 免責 + ペア比較(効果量・p・FDR)+ CRN 健全性 + DiD
  compare_stats.parquet … 指標×(平均差・Cohen's d・Cliff's δ・置換p・FDR q・t分布CI)

誠実性(既存レポート群から継承):
  - レポート冒頭に免責: 「p はラン数で任意に小さくできる。効果量と CI 幅を主・p を従」。
  - 欠測指標は「データ不足」。同一ランを --a と --b に与えたら全差ゼロ・p≈1(セルフチェック)。
  - 数値はログ由来の what-if 実験値(渋谷の実測ではない)。決定論(ソート固定・wall-clock 非依存)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)                         # 同ディレクトリの panel_stats を import

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import panel_stats as ps                           # noqa: E402  (統計純関数の再利用)

STEPS_PER_DAY = 144

# panel/run_day のうち指標として扱わない列(ID・メタ)。
_PANEL_NONMETRIC = {"run", "day", "weekday", "warmup", "n_agents", "n_residents"}


# --------------------------------------------------------------------------- #
# 1) ラン単位のスカラ指標
# --------------------------------------------------------------------------- #
def _read_pydict(path: str) -> dict[str, list] | None:
    import pyarrow.parquet as pq
    if not os.path.exists(path):
        return None
    return pq.read_table(path).to_pydict()


def _mean_numeric(vals) -> float | None:
    nums = []
    for v in vals:
        if v is None:
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            return None                            # 非数値列 → 指標にしない
    return sum(nums) / len(nums) if nums else None


def run_scalars(run_dir: str) -> dict[str, float]:
    """ラン 1 本のスカラ指標 dict を返す(読み出し専用)。
      l2.<col>    : l2_metrics.parquet の各数値列の step 平均(step 列自体は除外)
      panel.<col> : panel/run_day.parquet の各数値列の非warmup日平均(存在すれば)
    値が None の指標はキーごと落とす(= その指標は「データ不足」として下流で扱う)。"""
    out: dict[str, float] = {}
    l2 = _read_pydict(os.path.join(run_dir, "l2_metrics.parquet"))
    if l2:
        for col, vals in l2.items():
            if col == "step":
                continue
            mv = _mean_numeric(vals)
            if mv is not None:
                out[f"l2.{col}"] = mv
    rd = _read_pydict(os.path.join(run_dir, "panel", "run_day.parquet"))
    if rd and "day" in rd:
        warm = rd.get("warmup", [False] * len(rd["day"]))
        for col, vals in rd.items():
            if col in _PANEL_NONMETRIC:
                continue
            keep = [vals[i] for i in range(len(vals)) if not warm[i]]
            if not keep:
                keep = vals                        # 全 warmup のときは全日
            mv = _mean_numeric(keep)
            if mv is not None:
                out[f"panel.{col}"] = mv
    return out


# --------------------------------------------------------------------------- #
# 2) seed ペアリング
# --------------------------------------------------------------------------- #
def _seed_of(run_dir: str):
    name = os.path.basename(os.path.normpath(run_dir))
    _cond, seed = ps._parse_cond_seed(name, ps._cfg_seed(run_dir))
    return seed


def _cond_of(run_dir: str) -> str:
    name = os.path.basename(os.path.normpath(run_dir))
    cond, _seed = ps._parse_cond_seed(name, ps._cfg_seed(run_dir))
    return cond


def pair_runs(a_dirs: list[str], b_dirs: list[str]) -> tuple[list[tuple], list[str]]:
    """A/B のラン集合を seed で対応付ける。全ランの seed が取れれば seed 一致で組み、
    取れない/重複するときは与えられた順(位置)で組む。返り値 = (pairs, notes)。
    pairs = [(a_dir, b_dir, seed_label), ...]。"""
    notes: list[str] = []
    sa = [_seed_of(d) for d in a_dirs]
    sb = [_seed_of(d) for d in b_dirs]
    can_seed = (all(s is not None for s in sa) and all(s is not None for s in sb)
                and len(set(sa)) == len(sa) and len(set(sb)) == len(sb))
    pairs: list[tuple] = []
    if can_seed:
        mb = {s: d for s, d in zip(sb, b_dirs)}
        for s, d in sorted(zip(sa, a_dirs)):
            if s in mb:
                pairs.append((d, mb[s], f"s{s}"))
            else:
                notes.append(f"A の seed={s} に対応する B が無い(不計上)")
        for s, d in zip(sb, b_dirs):
            if s not in set(sa):
                notes.append(f"B の seed={s} に対応する A が無い(不計上)")
        notes.insert(0, "対応付け: seed 一致")
    else:
        n = min(len(a_dirs), len(b_dirs))
        for i in range(n):
            pairs.append((a_dirs[i], b_dirs[i], f"pos{i}"))
        if len(a_dirs) != len(b_dirs):
            notes.append(f"A={len(a_dirs)} 本・B={len(b_dirs)} 本で数が不一致 → 先頭 {n} 組のみ")
        notes.insert(0, "対応付け: 位置順(seed を一意に取得できず)")
    return pairs, notes


# --------------------------------------------------------------------------- #
# 3) 指標ごとの比較(panel_stats の効果量・置換検定・BH-FDR を流用)
# --------------------------------------------------------------------------- #
def compare_metrics(pairs: list[tuple]) -> tuple[list[dict], dict]:
    """各ペアのスカラ指標を読み、指標ごとに差 = B − A を検定する。
    両側そろって値のあるペアだけを用い、そのペア数を n_pairs に記録する。"""
    a_scalars = [run_scalars(a) for (a, b, _s) in pairs]
    b_scalars = [run_scalars(b) for (a, b, _s) in pairs]
    all_keys = sorted(set().union(*a_scalars, *b_scalars)) if pairs else []

    rows: list[dict] = []
    perm_ps: list = []
    for k in all_keys:
        diffs, av, bv = [], [], []
        n_missing = 0
        for i in range(len(pairs)):
            a = a_scalars[i].get(k)
            b = b_scalars[i].get(k)
            if a is None or b is None:
                n_missing += 1
                continue
            diffs.append(b - a)
            av.append(a)
            bv.append(b)
        n = len(diffs)
        if n == 0:
            rows.append({"metric": k, "n_pairs": 0, "n_missing": n_missing,
                         "mean_diff": None, "ci_lo_t": None, "ci_hi_t": None,
                         "cohen_d": None, "cliff_delta": None,
                         "perm_p": None, "fdr_q": None, "data_short": True})
            perm_ps.append(None)
            continue
        mean, _lo, _hi, _n, _sd = ps._mean_ci(diffs)
        _, tlo, thi, _, _ = ps.t_ci(diffs)
        p = ps.perm_test_paired(diffs, seed=0)
        rows.append({"metric": k, "n_pairs": n, "n_missing": n_missing,
                     "mean_diff": mean, "ci_lo_t": tlo, "ci_hi_t": thi,
                     "cohen_d": ps.paired_cohens_d(diffs),
                     "cliff_delta": ps.cliffs_delta(av, bv),
                     "perm_p": p, "fdr_q": None, "data_short": False})
        perm_ps.append(p)
    # BH-FDR(併記指標群に対する多重比較補正)
    qs = ps.bh_fdr(perm_ps)
    for r, q in zip(rows, qs):
        r["fdr_q"] = q
    meta = {"a_scalars": a_scalars, "b_scalars": b_scalars, "keys": all_keys}
    return rows, meta


# --------------------------------------------------------------------------- #
# 4) CRN 健全性チェック(介入 step まで L1 を突合)
# --------------------------------------------------------------------------- #
def intervention_step(run_dir: str):
    """scenario_shock(phase=start / event)の介入 step を返す。無ければ None。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.exists(path):
        return None
    avail = set(pq.read_schema(path).names)
    cols = [c for c in ("step", "kind", "payload") if c in avail]
    pf = pq.ParquetFile(path)
    at = None
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["step"])):
            if d["kind"][i] == "scenario_shock":
                try:
                    p = json.loads(d["payload"][i]) if d["payload"][i] else {}
                except (json.JSONDecodeError, TypeError):
                    p = {}
                if p.get("phase") in ("start", "event"):
                    a = p.get("at", d["step"][i])
                    at = a if at is None else min(at, a)
    return at


def _step_sigs(run_dir: str, upto_step: int) -> dict[int, Counter]:
    """step 別のイベント署名 multiset を作る(step<=upto_step)。
    署名 = (agent_id, kind, round(x,1), round(y,1))。実行パスの差を検出する proxy。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    avail = set(pq.read_schema(path).names)
    want = [c for c in ("step", "agent_id", "kind", "x", "y") if c in avail]
    pf = pq.ParquetFile(path)
    sigs: dict[int, Counter] = defaultdict(Counter)
    for batch in pf.iter_batches(columns=want):
        d = batch.to_pydict()
        steps = d["step"]
        # このバッチが全て範囲外(step>upto)なら以降も昇順なので打ち切り
        if steps and steps[0] > upto_step:
            break
        for i in range(len(steps)):
            s = steps[i]
            if s > upto_step:
                continue
            x = d["x"][i]
            y = d["y"][i]
            sig = (d["agent_id"][i], d["kind"][i],
                   round(float(x), 1) if x is not None else None,
                   round(float(y), 1) if y is not None else None)
            sigs[s][sig] += 1
    return sigs


def crn_health(a_dir: str, b_dir: str, crn_max_steps: int) -> dict:
    """ペア (a,b) の L1 を介入 step まで(無ければ crn_max_steps まで)突合。
    最初の分岐 step・一致率・突合 step 数を返す。"""
    at_a = intervention_step(a_dir)
    at_b = intervention_step(b_dir)
    ats = [a for a in (at_a, at_b) if a is not None]
    if ats:
        upto = min(ats)                            # 介入前だけを比較(CRN 前提の検証)
        basis = f"介入 step={upto} まで"
    else:
        upto = int(crn_max_steps)
        basis = f"介入なし → 先頭 {upto} step まで"
    sa = _step_sigs(a_dir, upto)
    sb = _step_sigs(b_dir, upto)
    steps = sorted(set(sa) | set(sb))
    first_div = None
    matched = 0
    for s in steps:
        if sa.get(s, Counter()) == sb.get(s, Counter()):
            matched += 1
        elif first_div is None:
            first_div = s
    compared = len(steps)
    rate = (matched / compared) if compared else None
    return {"basis": basis, "upto": upto, "compared_steps": compared,
            "matched_steps": matched, "match_rate": rate,
            "first_divergence": first_div,
            "intact": (first_div is None and compared > 0)}


# --------------------------------------------------------------------------- #
# 5) within-run DiD(介入 step の前後 × 処置/対照)
# --------------------------------------------------------------------------- #
def did_effect(pre_treat: list, post_treat: list,
               pre_ctrl: list, post_ctrl: list):
    """差の差(DiD)= (mean(post_treat)−mean(pre_treat)) − (mean(post_ctrl)−mean(pre_ctrl))。
    いずれかの群が空なら None(算出不能)。純関数=テスト対象。"""
    def _m(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None
    mt_pre, mt_post = _m(pre_treat), _m(post_treat)
    mc_pre, mc_post = _m(pre_ctrl), _m(post_ctrl)
    if None in (mt_pre, mt_post, mc_pre, mc_post):
        return None
    return (mt_post - mt_pre) - (mc_post - mc_pre)


def _scenario_shock_params(run_dir: str):
    """scenario_shock start の payload(center/radius_m 等)を返す。無ければ None。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.exists(path):
        return None
    avail = set(pq.read_schema(path).names)
    cols = [c for c in ("step", "kind", "payload") if c in avail]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["step"])):
            if d["kind"][i] == "scenario_shock":
                try:
                    p = json.loads(d["payload"][i]) if d["payload"][i] else {}
                except (json.JSONDecodeError, TypeError):
                    p = {}
                if p.get("phase") == "start":
                    return p
    return None


def did_within(run_dir: str, metric_kind: str = "move_segment") -> dict:
    """ラン内 DiD。scenario_shock(shock_closure の start)がある場合のみ算出。
    処置群 = 介入中心から radius_m 内で観測されたことのある agent、対照群 = それ以外。
    指標 = agent ごとの metric_kind イベント数(介入 step 前後で比較)。
    介入 step / center / radius が取れないランは明示的にスキップ(reason つき)。"""
    at = intervention_step(run_dir)
    if at is None:
        return {"ok": False, "reason": "scenario_shock 無し(介入 step を特定できずスキップ)"}
    params = _scenario_shock_params(run_dir)
    center = None
    radius = None
    if params is not None:
        c = params.get("center")
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            center = (float(c[0]), float(c[1]))
        if params.get("radius_m") is not None:
            radius = float(params["radius_m"])
    if center is None or radius is None:
        return {"ok": False, "reason": f"介入 step={at} だが center/radius 不明"
                "(shock_closure 以外)→ 空間処置群を定義できずスキップ"}

    # L1 を 1 パス: agent ごとに (介入前 metric 数・介入後 metric 数・中心近傍観測フラグ)
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    avail = set(pq.read_schema(path).names)
    want = [c for c in ("step", "agent_id", "kind", "x", "y") if c in avail]
    pf = pq.ParquetFile(path)
    pre: dict = defaultdict(int)
    post: dict = defaultdict(int)
    near: set = set()
    for batch in pf.iter_batches(columns=want):
        d = batch.to_pydict()
        for i in range(len(d["step"])):
            aid = d["agent_id"][i]
            if aid is None or aid < 0:
                continue
            x, y = d["x"][i], d["y"][i]
            if x is not None and y is not None:
                if math.hypot(float(x) - center[0], float(y) - center[1]) <= radius:
                    near.add(aid)
            if d["kind"][i] == metric_kind:
                if d["step"][i] < at:
                    pre[aid] += 1
                else:
                    post[aid] += 1
    agents = set(pre) | set(post) | near
    pre_t, post_t, pre_c, post_c = [], [], [], []
    for aid in agents:
        if aid in near:
            pre_t.append(pre.get(aid, 0))
            post_t.append(post.get(aid, 0))
        else:
            pre_c.append(pre.get(aid, 0))
            post_c.append(post.get(aid, 0))
    did = did_effect(pre_t, post_t, pre_c, post_c)
    return {"ok": did is not None, "at": at, "center": center, "radius": radius,
            "metric_kind": metric_kind, "n_treat": len(pre_t), "n_ctrl": len(pre_c),
            "did": did,
            "reason": None if did is not None else "処置/対照群のいずれかが空"}


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def _write_stats_parquet(path: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = ("metric", "n_pairs", "n_missing", "mean_diff", "ci_lo_t", "ci_hi_t",
            "cohen_d", "cliff_delta", "perm_p", "fdr_q")
    cols = {c: [r.get(c) for r in rows] for c in keys}
    schema = pa.schema([
        pa.field("metric", pa.string()), pa.field("n_pairs", pa.int64()),
        pa.field("n_missing", pa.int64()), pa.field("mean_diff", pa.float64()),
        pa.field("ci_lo_t", pa.float64()), pa.field("ci_hi_t", pa.float64()),
        pa.field("cohen_d", pa.float64()), pa.field("cliff_delta", pa.float64()),
        pa.field("perm_p", pa.float64()), pa.field("fdr_q", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)


def render_report(cond_a: str, cond_b: str, pairs: list[tuple],
                  pair_notes: list[str], rows: list[dict],
                  crn: list[dict], did: list[dict]) -> str:
    L: list[str] = []
    L.append(f"# 介入前後の比較 — {cond_b} vs {cond_a}(差 = {cond_b} − {cond_a})\n")
    L.append("> **p 値の限界(必読)**: シミュレーションの p 値はラン数を増やせば任意に小さくできる"
             "(些少な効果でも有意になり得る)。**効果量(Cohen's d・Cliff's δ)と CI 幅を主・p を従**"
             "として、実質的有意性で判断する(JASSS の ABM 出力解析標準)。多重比較は BH-FDR で補正済み。")
    L.append("> 数値はログ由来の what-if 実験値(渋谷の実測ではない)。\n")

    # ---- ペア ----
    L.append("## CRN ペア")
    for note in pair_notes:
        L.append(f"- {note}")
    L.append("")
    L.append("| ペア | A | B |")
    L.append("|---|---|---|")
    for (a, b, s) in pairs:
        L.append(f"| {s} | {os.path.basename(os.path.normpath(a))} | "
                 f"{os.path.basename(os.path.normpath(b))} |")
    L.append("")

    # ---- CRN 健全性 ----
    L.append("## CRN 健全性チェック(実行パスの突合)")
    L.append("- 目的: 同 seed のペアが介入前に同一の実行パス(= L1 の (agent,kind,x,y) 署名)を"
             "たどるかを確認する。**分岐が早いほどペア比較の分散削減が効かない**(前提が弱い)。")
    L.append("")
    L.append("| ペア | 突合範囲 | 突合step | 一致step | 一致率 | 最初の分岐step | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for (pair, h) in zip(pairs, crn):
        rate = f"{100.0 * h['match_rate']:.1f}%" if h["match_rate"] is not None else "―"
        verdict = "一致(CRN 保持)" if h["intact"] else (
            "警告: 実行パス分岐あり" if h["compared_steps"] else "データ不足")
        fd = h["first_divergence"] if h["first_divergence"] is not None else "―"
        L.append(f"| {pair[2]} | {h['basis']} | {h['compared_steps']} | "
                 f"{h['matched_steps']} | {rate} | {fd} | {verdict} |")
    L.append("")
    if any(not h["intact"] and h["compared_steps"] for h in crn):
        L.append("- ⚠ 分岐が検出されたペアがある。条件差そのものが実行パスを変える場合は想定内だが、"
                 "介入前から分岐していると **CRN による分散削減の前提が崩れる**。効果量・CI は"
                 "「独立2群の差」に近い解釈で読むこと。")
        L.append("")

    # ---- 指標比較 ----
    L.append("## 指標ごとの差(効果量・置換検定・FDR)")
    have = [r for r in rows if not r["data_short"]]
    short = [r for r in rows if r["data_short"]]
    if have:
        L.append("- **主指標**: paired Cohen's d(差の標準化)・Cliff's δ(順位・[-1,1])。"
                 "**従指標**: 置換 p(符号反転・n≤12 全列挙)・BH-FDR q。CI は t 分布(n<30)。")
        L.append("| 指標 | n | 平均差 | Cohen's d | Cliff's δ | 置換p | FDR q | 95%CI(t)下 | 上 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in have:
            L.append(f"| {r['metric']} | {r['n_pairs']} | {ps._fmt(r['mean_diff'])} | "
                     f"{ps._fmt(r['cohen_d'])} | {ps._fmt(r['cliff_delta'])} | "
                     f"{ps._fmt(r['perm_p'])} | {ps._fmt(r['fdr_q'])} | "
                     f"{ps._fmt(r['ci_lo_t'])} | {ps._fmt(r['ci_hi_t'])} |")
        L.append("")
    else:
        L.append("- 両側そろって値のある指標が **0**(データ不足)。ペアのスカラ指標を比較できない。")
        L.append("")
    if short:
        L.append("### データ不足(片側または両側で欠測 → 比較不能)")
        L.append("| 指標 | 欠測ペア数 |")
        L.append("|---|---|")
        for r in short:
            L.append(f"| {r['metric']} | {r['n_missing']} |")
        L.append("")

    # ---- within-run DiD ----
    L.append("## within-run DiD(介入 step 前後 × 処置/対照)")
    any_did = False
    L.append("| ラン | 結果 |")
    L.append("|---|---|")
    for (rd, dd) in did:
        name = os.path.basename(os.path.normpath(rd))
        if dd.get("ok"):
            any_did = True
            L.append(f"| {name} | DiD={ps._fmt(dd['did'])}(介入step={dd['at']}・"
                     f"処置{dd['n_treat']}体/対照{dd['n_ctrl']}体・指標={dd['metric_kind']}) |")
        else:
            L.append(f"| {name} | スキップ: {dd['reason']} |")
    L.append("")
    if not any_did:
        L.append("- 対象ランに介入(scenario_shock)が無いため DiD は全てスキップ。"
                 "**0 を捏造しない**(データ不足)。介入ランを与えれば算出する。")
        L.append("")
    return "\n".join(L)


def run_compare(a_dirs: list[str], b_dirs: list[str], out_dir: str,
                crn_max_steps: int = 288) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pairs, pair_notes = pair_runs(a_dirs, b_dirs)
    cond_a = _cond_of(a_dirs[0]) if a_dirs else "A"
    cond_b = _cond_of(b_dirs[0]) if b_dirs else "B"
    rows, _meta = compare_metrics(pairs)
    crn = [crn_health(a, b, crn_max_steps) for (a, b, _s) in pairs]
    did = [(a, did_within(a)) for (a, b, _s) in pairs]

    _write_stats_parquet(os.path.join(out_dir, "compare_stats.parquet"), rows)
    md = render_report(cond_a, cond_b, pairs, pair_notes, rows, crn, did)
    with open(os.path.join(out_dir, "compare_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    return {"out_dir": out_dir, "n_pairs": len(pairs), "cond_a": cond_a,
            "cond_b": cond_b, "rows": rows, "crn": crn, "did": did, "md": md,
            "n_have": sum(1 for r in rows if not r["data_short"])}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="⑥介入前後の比較(between-run CRN + within-run DiD・L1/派生 読み出し専用)")
    ap.add_argument("--a", nargs="+", required=True, help="対照群のラン(seed 順)")
    ap.add_argument("--b", nargs="+", required=True, help="処置群のラン(seed 順)")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 runs/_compare/<A条件>_vs_<B条件>)")
    ap.add_argument("--crn-max-steps", type=int, default=288,
                    help="介入が無いとき CRN 突合する先頭 step 数(既定288=2日)")
    args = ap.parse_args()
    for d in list(args.a) + list(args.b):
        if not os.path.isdir(d):
            raise SystemExit(f"not a directory: {d}")
    out = args.out
    if out is None:
        ca = _cond_of(args.a[0])
        cb = _cond_of(args.b[0])
        name = f"{cb}_vs_{ca}" if cb != ca else f"{ca}_vs_self"
        out = os.path.join("runs", "_compare", name)
    info = run_compare(args.a, args.b, out, crn_max_steps=args.crn_max_steps)
    print(info["md"])
    print(f"\n[compare_runs] pairs={info['n_pairs']}  "
          f"比較可能指標={info['n_have']}  -> {info['out_dir']}")


if __name__ == "__main__":
    main()
