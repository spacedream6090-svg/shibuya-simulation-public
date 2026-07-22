#!/usr/bin/env python
"""不確実性の監査(第51バッチ・読み出し専用)。

    python scripts/audit_uncertainty.py --run u51_audit_80a1d
    python scripts/audit_uncertainty.py --run u51_audit_80a1d --out audit/

外部指摘「100%予測可能な世界ではダイナミズムが失われる」への回答その1=監査。
現状シムに **既にどれだけの揺らぎが入っているか** を、完成済みラン(l1_events.parquet)から
後処理で測る。**シム本体・schema・config には一切触れない**(完全な読み取り)。

このスクリプトが測るのは 1 点:
  「住民 1 人の 1 日に、揺らぎ由来の『想定外イベント』が何件入るか」
= 確率的実行(寄り道 detour / 中断 interrupt)・退屈/好奇心の内発探索(boredom_explore)・
  品切れ遭遇(stock_out)・病気発症(illness)・犯罪被害(crime)・ライフイベント(life_event)・
  街全体の交通遅延/災害/インフラ障害(transit_delay / disaster / infra_outage)への曝露。

対応表 = `LUCK_KINDS`(下記)。**将来の kind 追加(第54バッチ=不確実性モードの chance_event)は
LUCK_KINDS に 1 行足すだけ** で監査にも運/実力分解(analyze_luck.py が import)にも反映される。
各 kind が schema 登録済みであることは tests/test_audit_uncertainty.py が検証する。

依存は標準ライブラリ + pyarrow のみ(pandas / duckdb 不使用・numpy は任意)。1 step=10分・1日=144step。
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")

STEPS_PER_DAY = 144


# --------------------------------------------------------------------------- #
# 揺らぎ由来イベント対応表(運/実力分解の analyze_luck.py と共有 import)
# --------------------------------------------------------------------------- #
# ★この 1 表がバッチ51とバッチ54の接点。1 kind = 1 行 で足せる(下の chance_event 例参照)。
#   フィールド:
#     category  短い分類ラベル(集計の粒度)。
#     scope     "agent" = 当該 agent_id 本人への発生 / "world" = agent_id=-1 の街全体ショック
#               (曝露は「その日 街に居た住民全員」に帰属=有界で決定論)。
#     stream    その揺らぎを駆動する RngHub の named stream(監査文書用の出所メモ。"-"=内生)。
#     valence   運の符号: "neg"(不運)/ "pos"(幸運)/ "mixed"(どちらにもなりうる)。
#     attr_field payload のこのキーの id に帰属し直す(None=agent_id 本人)。crime の被害者など。
#     phase     ("field","value"): payload[field]==value の行だけ数える(disaster の onset のみ 等)。
#     note      説明。
def _lk(category, *, scope="agent", stream="-", valence="neg",
        attr_field=None, phase=None, note=""):
    return {"category": category, "scope": scope, "stream": stream,
            "valence": valence, "attr_field": attr_field, "phase": phase,
            "note": note}


LUCK_KINDS: dict[str, dict] = {
    # ---- 確率的実行 L3(routine.stochastic。専用 stream)----
    "detour":          _lk("detour",    stream="detour",    valence="mixed",
                           note="寄り道=目的地手前で近傍POIに立ち寄り(per-move 抽選)"),
    "interrupt":       _lk("interrupt", stream="interrupt", valence="mixed",
                           note="活動途中の中断=離脱(per-step 抽選)"),
    # ---- 好奇心/退屈の内発ドライブ(drive.boredom。専用 stream)----
    "boredom_explore": _lk("curiosity", stream="boredom",   valence="pos",
                           note="退屈ゲージ閾値超え→近傍の未訪問POIへ内発的探索"),
    # ---- 商業ダイナミクス(commerce.inventory / 在館数)----
    "stock_out":       _lk("scarcity",  stream="(内生:在庫/在館数)", valence="neg",
                           note="品切れ/行列に遭遇し購入不成立"),
    # ---- 健康(health。専用 stream)----
    "illness":         _lk("health",    stream="health", valence="neg",
                           phase=("state", "onset"),
                           note="病気の発症=欠勤/外出抑制(onset のみ計上)"),
    # ---- 治安(society_diversity。専用 stream。被害者へ帰属)----
    "crime":           _lk("crime",     stream="crime", valence="neg",
                           attr_field="victim",
                           note="窃盗など犯罪被害(被害者の不運。offender は加害側=別)"),
    # ---- 世帯・恋愛(household。date/dinner 系 stream)----
    "life_event":      _lk("life",      stream="date/dinner/household", valence="mixed",
                           note="ライフイベント(出会い partner / 別れ breakup)"),
    # ---- 都市・環境ショック(disaster。世界レベル stream。街全体へ曝露)----
    "disaster":        _lk("shock",     scope="world", stream="disaster", valence="neg",
                           phase=("phase", "onset"),
                           note="災害ショックの発生(台風/地震/大雪。onset のみ計上)"),
    "transit_delay":   _lk("delay",     scope="world", stream="disaster", valence="neg",
                           note="交通の遅延・運休(街全体)"),
    "infra_outage":    _lk("shock",     scope="world", stream="disaster", valence="neg",
                           phase=("phase", "onset"),
                           note="インフラ障害=停電/通信断/断水(onset のみ計上)"),
    # ---- 第54バッチ(不確実性モード)で接続(chance.enabled ON のみ・既定 OFF=0 件)。1 行で監査+ΔR² に反映 ----
    "chance_event":    _lk("chance",    stream="chance", valence="mixed",
                           note="臨時収入 windfall/財布紛失 loss/偶然の出会い encounter の偶発事(src/society/chance.py)"),
}


def _phase_ok(spec: dict, payload: dict) -> bool:
    """LUCK_KINDS の phase フィルタ(("field","value"))を評価。None なら常に True。"""
    ph = spec.get("phase")
    if not ph:
        return True
    field, value = ph
    return payload.get(field) == value


def _attr_id(spec: dict, agent_id: int, payload: dict):
    """揺らぎイベントを帰属させる agent id を返す(attr_field 指定時は payload から)。"""
    af = spec.get("attr_field")
    if af is not None:
        v = payload.get(af)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return agent_id


# --------------------------------------------------------------------------- #
# ローダ(scripts/analyze_founders.py と同じ house style: pyarrow 列射影 + batch 逐次)
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet を読み、{step, sim_min, agent, kind, payload(dict)} 列で返す。"""
    import pyarrow.parquet as pq

    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "payload"]
    available = set(pq.read_schema(path).names)
    cols = [c for c in want if c in available]
    pf = pq.ParquetFile(path)
    out: list[dict] = []
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        n = len(d["step"])
        smin = d.get("sim_min")
        pays = d["payload"]
        for i in range(n):
            raw = pays[i]
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({
                "step": int(d["step"][i]),
                "sim_min": (int(smin[i]) if smin is not None and smin[i] is not None
                            else int(d["step"][i]) * 10),
                "agent": int(d["agent_id"][i]),
                "kind": d["kind"][i],
                "payload": payload,
            })
    return out


def load_agents(run_dir: str) -> dict[int, dict]:
    """agents.json を {agent_id: meta} で返す(無ければ空)。visitor 判定に使う。"""
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[int, dict] = {}
    for a in data:
        try:
            out[int(a["id"])] = a
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# 帰属の中核(監査 audit と 運/実力分解 analyze_luck が共有する 1 箇所)
# --------------------------------------------------------------------------- #
def _active_residents_by_day(events: list[dict], residents: set[int] | None):
    """day -> その日 街に居た住民集合(= agent_id>=0 のイベントを持つ・residents 指定時は住民のみ)。"""
    active: dict[int, set] = defaultdict(set)
    for e in events:
        aid = e["agent"]
        if aid >= 0 and (residents is None or aid in residents):
            active[e["step"] // STEPS_PER_DAY].add(aid)
    return active


def attribute_chances(events: list[dict],
                      agents: dict[int, dict] | None = None) -> dict:
    """揺らぎ由来イベントを住民へ帰属した中核結果を返す(監査・運分解の共通土台)。

    LUCK_KINDS の意味づけ(scope / attr_field / phase)を適用する **唯一の場所**。
      - scope="agent": 本人(attr_field 指定時は payload の当該 id=犯罪被害者など)へ帰属。
      - scope="world": agent_id=-1 のショックを「その日 街に居た住民全員」へ 1 件ずつ曝露として帰属
        (= 有界・決定論。街に居なければ曝露しない)。
    返り値:
      per_ad        {(agent, day): Counter(category->count)}
      per_agent     {agent: Counter(category->count)}(全日合算=analyze_luck が使う運側変数)
      by_kind       {kind: 生件数}(phase フィルタ後)
      by_category   {category: 帰属済み件数}
      world_by_day  {day: Counter(kind->count)}
      n_world_exposures  world 曝露の総帰属件数
    """
    agents = agents or {}
    residents = {aid for aid, m in agents.items() if not m.get("visitor")} or None
    active_by_day = _active_residents_by_day(events, residents)

    per_ad: dict[tuple, Counter] = defaultdict(Counter)
    by_kind: Counter = Counter()
    by_category: Counter = Counter()
    world_by_day: dict[int, Counter] = defaultdict(Counter)

    for e in events:
        spec = LUCK_KINDS.get(e["kind"])
        if spec is None or not _phase_ok(spec, e["payload"]):
            continue
        by_kind[e["kind"]] += 1
        day = e["step"] // STEPS_PER_DAY
        if spec["scope"] == "world":
            world_by_day[day][e["kind"]] += 1
        else:
            tid = _attr_id(spec, e["agent"], e["payload"])
            if tid is None or tid < 0:
                continue
            per_ad[(tid, day)][spec["category"]] += 1
            by_category[spec["category"]] += 1

    n_world_exposures = 0
    for day, kc in world_by_day.items():
        pool = active_by_day.get(day) or {aid for (aid, d) in per_ad if d == day}
        for aid in pool:
            for kind, c in kc.items():
                cat = LUCK_KINDS[kind]["category"]
                per_ad[(aid, day)][cat] += c
                by_category[cat] += c
                n_world_exposures += c

    per_agent: dict[int, Counter] = defaultdict(Counter)
    for (aid, _day), cc in per_ad.items():
        per_agent[aid].update(cc)

    return {"per_ad": per_ad, "per_agent": per_agent, "by_kind": by_kind,
            "by_category": by_category, "world_by_day": world_by_day,
            "n_world_exposures": n_world_exposures,
            "active_by_day": active_by_day, "residents": residents}


# --------------------------------------------------------------------------- #
# 監査本体
# --------------------------------------------------------------------------- #
def audit(events: list[dict], agents: dict[int, dict] | None = None) -> dict:
    """揺らぎ由来イベントの per-agent-day 件数分布と全イベント比率を返す。"""
    agents = agents or {}
    residents = {aid for aid, m in agents.items() if not m.get("visitor")}

    total_events = len(events)
    total_agent_events = sum(1 for e in events if e["agent"] >= 0)   # 比率の分母

    attr = attribute_chances(events, agents)
    per_ad = attr["per_ad"]
    by_kind = attr["by_kind"]
    by_category = attr["by_category"]
    world_by_day = attr["world_by_day"]
    active_by_day = attr["active_by_day"]
    n_world_exposures = attr["n_world_exposures"]

    # per-agent-day の「想定外イベント総数」の分布
    counts = [sum(cc.values()) for cc in per_ad.values()]
    # 揺らぎに 1 件も遭遇しなかった (resident, day) も母集団に含める(0 を数える)。
    if agents:
        all_days = sorted({e["step"] // STEPS_PER_DAY for e in events})
        zero_cells = 0
        seen = set(per_ad)
        for aid in residents:
            for day in all_days:
                if (aid, day) not in seen and aid in active_by_day.get(day, ()):  # 在街日のみ
                    zero_cells += 1
        counts = counts + [0] * zero_cells

    dist = _distribution(counts)
    chance_total = sum(by_category.values())

    return {
        "n_events_total": total_events,
        "n_agent_events": total_agent_events,
        "n_residents": len(residents) if agents else None,
        "chance_events_attributed": chance_total,
        "chance_fraction_of_agent_events": (round(chance_total / total_agent_events, 4)
                                            if total_agent_events else 0.0),
        "world_exposures": n_world_exposures,
        "per_agent_day": dist,
        "by_kind": dict(sorted(by_kind.items())),
        "by_category": dict(sorted(by_category.items())),
        "world_shock_days": {str(d): dict(kc) for d, kc in sorted(world_by_day.items())},
        "luck_kinds_registered": sorted(LUCK_KINDS),
    }


def _distribution(counts: list[int]) -> dict:
    """件数リストの要約統計 + ヒストグラム(決定論。空でも壊れない)。"""
    n = len(counts)
    if n == 0:
        return {"n_cells": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0,
                "nonzero_cells": 0, "nonzero_fraction": 0.0, "histogram": {}}
    s = sorted(counts)
    total = sum(s)
    mean = total / n
    median = _quantile(s, 0.5)
    p90 = _quantile(s, 0.9)
    nonzero = sum(1 for c in s if c > 0)
    hist: Counter = Counter(s)
    return {
        "n_cells": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "p90": round(p90, 4),
        "max": s[-1],
        "nonzero_cells": nonzero,
        "nonzero_fraction": round(nonzero / n, 4),
        "histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def _quantile(sorted_vals: list[int], q: float) -> float:
    """線形補間分位点(numpy を引き込まない最小実装)。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #
def render_text(run_name: str, res: dict) -> str:
    d = res["per_agent_day"]
    lines: list[str] = []
    lines.append(f"# 不確実性監査: {run_name}")
    lines.append("")
    lines.append(f"総イベント数           : {res['n_events_total']}")
    lines.append(f"agent 帰属イベント数   : {res['n_agent_events']}")
    lines.append(f"住民数(非 visitor)   : {res['n_residents']}")
    lines.append(f"揺らぎ由来(帰属済み)  : {res['chance_events_attributed']} "
                 f"件 = 全 agent イベントの {res['chance_fraction_of_agent_events']:.1%}")
    lines.append(f"世界ショック曝露       : {res['world_exposures']} 件")
    lines.append("")
    lines.append("## 想定外イベント / 住民1人1日(在街セルのみ)")
    lines.append(f"  セル数 {d['n_cells']} / 平均 {d['mean']} / 中央 {d['median']} / "
                 f"p90 {d['p90']} / 最大 {d['max']}")
    lines.append(f"  1件以上遭遇したセル: {d['nonzero_cells']} "
                 f"({d['nonzero_fraction']:.1%})")
    lines.append(f"  ヒストグラム(件数→セル数): {d['histogram']}")
    lines.append("")
    lines.append("## kind 別の生件数")
    for k, v in res["by_kind"].items():
        spec = LUCK_KINDS[k]
        lines.append(f"  {k:16s} {v:5d}  [{spec['category']}/{spec['valence']}] "
                     f"stream={spec['stream']}")
    lines.append("")
    lines.append("## category 別の帰属件数")
    for k, v in res["by_category"].items():
        lines.append(f"  {k:12s} {v:5d}")
    if res["world_shock_days"]:
        lines.append("")
        lines.append("## 世界ショック発生日")
        for day, kc in res["world_shock_days"].items():
            lines.append(f"  day {day}: {kc}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, out_dir: str) -> dict:
    """ラン1本を監査し、audit/uncertainty.json と uncertainty.txt を書く。"""
    run_name = os.path.basename(os.path.normpath(run_dir))
    events = load_events(run_dir)
    agents = load_agents(run_dir)
    res = audit(events, agents)
    res["run"] = run_name

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "uncertainty.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, sort_keys=True)
    txt_path = os.path.join(out_dir, "uncertainty.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(render_text(run_name, res))
    res["paths"] = {"json": json_path, "txt": txt_path}
    return res


def _pick_run(arg_run: str | None) -> str:
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isfile(os.path.join(d, "l1_events.parquet")):
            raise SystemExit(f"[audit] l1_events.parquet が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[audit] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[audit] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _resolve_out(run_dir: str, arg_out: str | None) -> str:
    if not arg_out:
        return os.path.join(run_dir, "audit")
    return arg_out if os.path.isabs(arg_out) else os.path.join(run_dir, arg_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="不確実性の監査(読み出し専用)")
    ap.add_argument("--run", default=None,
                    help="ラン名(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None,
                    help="出力ディレクトリ(既定: <run>/audit)。相対はラン基準")
    args = ap.parse_args()

    run_dir = _pick_run(args.run)
    out_dir = _resolve_out(run_dir, args.out)
    res = analyze(run_dir, out_dir)
    print(render_text(res["run"], res))
    print(f"[audit] -> {res['paths']['json']}")
    print(f"[audit] -> {res['paths']['txt']}")


if __name__ == "__main__":
    main()
