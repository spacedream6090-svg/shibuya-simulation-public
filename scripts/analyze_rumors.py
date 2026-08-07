#!/usr/bin/env python
"""噂(情報オブジェクト)の伝播カスケード解析 — IF-C(第95バッチ)。

    python scripts/analyze_rumors.py runs/<name> [--out runs/<name>/analysis]

**読み出し専用**: `runs/<name>/l1_events.parquet` を読むだけ。シム本体・schema・
既存の解析スクリプトは 1 バイトも触らない(IF-B の reject.note_verify と同じ
「外から出力を読む」流儀)。`observer/metrics_spec.py` の**凍結 14 ファイル**にも
本スクリプトは**含まれない**(= metrics_spec_hash は無風)。

何を出すか
----------
**(A) OASIS 流の 3 指標**(docs/research/if-lane-research.md §2-1)。
     OASIS (2024, 100万エージェント) が実データとの整合を測るのに使った指標に揃える:
       scale       … その噂に到達した**人数**(根 = 初期 knower を含む)
       depth       … 伝播木の**最大深さ**(根 = 深さ 0)
       max_breadth … 深さごとの参加者数の**最大**
     ★OASIS は「自分たちの depth は実データより**顕著に浅い**」と自認している。
       本シムの depth を並べれば直接比較の主張が立つ。

**(B) Daley-Kendall / Maki-Thompson の理論値との比較**(§2-1)。
     古典モデルの極限値は
       最終 ignorant 比 ≈ **0.203**(x = e^{-2x} の解 = 「2割は永久に知らないまま」)
       spreader ピーク比 = **1 − ln2 ≈ 0.3069**
     である。本シムは「よく混ざった無限母集団で全員が全員に出会いうる」という
     古典モデルの前提を**満たさない**(物理的な同席・生活時間・関係の偏りがある)ので、
     ★**一致しないことが期待される**。本表は「どちらが正しいか」ではなく
     「**LLM エージェントの街が古典噂モデルからどう逸脱するか**」を数値で示すためにある
     (§2-3: それ自体が論文の主張になりうる)。

正直な限界(4 件)
------------------
1. **母集団の取り方**: 理論の「ignorant 比」は母集団が固定の閉じた系を前提にする。
   本シムは来街者の出入りがあるので、分母は `--population`(明示)か、
   L1 に現れた**相異なる agent_id 数**(既定)を使う。どちらも近似であり出力に明記する。
2. **spreader ピークは観測できない**: L1 には「今 spreader である人数」の時系列が無い
   (`rumor_stifle` は stifler 化の**瞬間**しか記録しない)。そこで
   `rumor_born` / `transmission` / `rumor_stifle` から**再構成した**時系列を使う。
   再構成は「知った時点で spreader、stifle した時点で脱落」の決定論再生であり、
   忘却 TTL(`forget_steps`)を ON にしたランでは**過大**に出る(忘却は L1 に出ない)。
3. **凍結指標との混線**: `observer/measure.py` / `stream.py` / `echo.py` の
   `c_transmission` / `n_transmission` / `transmission_novel_rate` は item の kind を
   見ないので、噂 ON のランではそれらに噂が混ざる。本スクリプトは item_id の接頭辞
   (`rumor-`)で**噂だけ**を切り出すので、混線した凍結指標から噂ぶんを引き算できる
   (出力の `vocab_transmissions` がその引き算の分母)。
4. **プール退場**: `world/pool.py` の dehydrate は `_rumors` を運ばないので、街を出て
   戻った者は噂を忘れる。pool ON のランでは scale が過小に出る(rumors.py の docstring)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:                     # l1_stream(W2-2 の共有逐次読み)用
    sys.path.insert(0, _SCRIPTS)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = 1
RUMOR_PREFIX = "rumor-"

# Daley-Kendall / Maki-Thompson の極限値(if-lane-research §2-1)。
# 最終 ignorant 比 = x* where x = exp(-2(1-x)) … 数値解 ≈ 0.2032
DK_FINAL_IGNORANT = 0.2032
DK_PEAK_SPREADER = 1.0 - math.log(2.0)          # ≈ 0.30685


# --------------------------------------------------------------------------- #
# 読み込み(読み取り専用)
# --------------------------------------------------------------------------- #
WANT_KINDS = ("rumor_born", "rumor_stifle", "transmission", "speak", "hear", "dm")


def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet から本解析に要る種だけを dict 化して返す(**小ラン専用**)。

    W2-2: 旧実装は `read_table().to_pylist()` で L1 を丸ごと Python 化していた。
    ここは `l1_stream.iter_events(kinds=...)` の逐次読みに置き換えてあり、
    **返り値・行順は 1 バイトも変わらない**(不要な種の dict を作らなくなっただけ)。

    ただし `speak`/`hear`/`dm` は L1 の大半を占めるので、**戻り値の list 自体が
    在場 25万 × 10 日では依然として数十 GB になる**。本選規模では `collect()` を使う
    こと(`main()` はそちらを呼ぶ)。この関数は既存テスト・小ランの互換のために残す。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        raise SystemExit(f"[rumors] l1_events.parquet が無い: {run_dir}")
    out: list[dict] = [
        {"step": e["step"], "agent_id": e["agent_id"], "kind": e["kind"],
         "payload": e["payload"]}
        for e in ls.iter_events(run_dir,
                                columns=["step", "agent_id", "kind", "payload"],
                                kinds=WANT_KINDS)
    ]
    out.sort(key=lambda r: (r["step"], r["kind"], r["agent_id"]))
    return out


def is_rumor(item_id) -> bool:
    return isinstance(item_id, str) and item_id.startswith(RUMOR_PREFIX)


# --------------------------------------------------------------------------- #
# W2-2: チャンク集約(在場 25万対応)
# --------------------------------------------------------------------------- #
def _new_bundle() -> dict:
    return {"events": [], "population_seen": set(),
            "n_all_transmissions": 0, "n_rumor_transmissions": 0}


def _feed(b: dict, step: int, aid: int, kind: str, p: dict) -> None:
    """1 イベントを集約に畳む。**木に効く行だけ** `events` に残す。

    - 母集団は集合(順序非依存・上限 = 体数)。
    - `transmission` の件数は 2 本のカウンタ(全件 / 噂ぶん)。
    - `speak`/`hear`/`dm` と **噂でない** `transmission` は母集団と件数にしか効かない
      (`build_cascades` はこれらを必ず読み飛ばす)ので、ここで捨てる。これが
      「40.6 億件 → 噂カスケードの実サイズ」に落ちる唯一の理由。
    """
    if aid >= 0:
        b["population_seen"].add(aid)
    for key in ("from", "to", "speaker"):
        v = p.get(key)
        if isinstance(v, int) and v >= 0:
            b["population_seen"].add(v)
    for key in ("hearers", "knowers"):
        for v in (p.get(key) or []):
            if isinstance(v, int) and v >= 0:
                b["population_seen"].add(v)
    if kind == "transmission":
        b["n_all_transmissions"] += 1
        if not is_rumor(p.get("item_id")):
            return
        b["n_rumor_transmissions"] += 1
    elif kind not in ("rumor_born", "rumor_stifle"):
        return
    b["events"].append({"step": step, "agent_id": aid, "kind": kind, "payload": p})


def collect(run_dir: str) -> dict:
    """L1 を **1 パスの逐次読み**で畳み、噂解析に要る集約だけを返す。

    ピークメモリは **O(噂カスケードに効く行数 + 体数)**。L1 の総行数には比例しない。
    `summarize()` に直接渡せる(`summarize(load_events(rd), None)` と同じ結果を返す
    ことを `tests/test_l1_stream.py` がバイト同値で固定している)。

    残る比例項(正直な注記):
      - `events` は「噂の誕生・stifle・**噂の**伝播」の行数に比例する。噂が街全体に
        広がった場合はこれ自体が大きくなりうる(= O(1) にはならない)。
      - `population_seen` は distinct agent_id に比例(在場 25万なら 25万要素の set)。
    """
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        raise SystemExit(f"[rumors] l1_events.parquet が無い: {run_dir}")
    b = _new_bundle()
    for e in ls.iter_events(run_dir,
                            columns=["step", "agent_id", "kind", "payload"],
                            kinds=WANT_KINDS):
        _feed(b, e["step"], e["agent_id"], e["kind"], e["payload"])
    # 旧 load_events と同じ全体ソート。対象は「木に効く行」だけに縮んでいるが、
    # 安定ソートなので生存行どうしの相対順序は全件ソートしたときと同一になる。
    b["events"].sort(key=lambda r: (r["step"], r["kind"], r["agent_id"]))
    return b


def _bundle_from_events(events: list[dict]) -> dict:
    """既存の list[dict] 経路(テスト・小ラン)から同じ集約を作る互換シム。"""
    b = _new_bundle()
    for e in events:
        _feed(b, e["step"], e["agent_id"], e["kind"], e["payload"])
    b["events"].sort(key=lambda r: (r["step"], r["kind"], r["agent_id"]))
    return b


# --------------------------------------------------------------------------- #
# カスケード木の再構成(§2-2: transmissions は IC の realization そのもの)
# --------------------------------------------------------------------------- #
def build_cascades(events: list[dict]) -> dict[str, dict]:
    """item_id → {src_kind, node, born_step, roots, edges, stifles}。

    edges = (step, from_agent, to_agent) の**時刻順**。同じ受け手に複数の親が来る場合は
    **最初に届いた辺だけ**を木の辺として採る(= IC の「その受け手を活性化した辺」)。
    後続の重複到達は `redundant` に数える(Maki-Thompson の stifler 化トリガーの実測量)。
    """
    cas: dict[str, dict] = {}
    for e in events:
        p = e["payload"]
        if e["kind"] == "rumor_born":
            iid = p.get("item_id")
            if not is_rumor(iid):
                continue
            roots = [int(a) for a in (p.get("knowers") or [])]
            cas[iid] = {"item_id": iid, "src_kind": str(p.get("src_kind", "")),
                        "node": str(p.get("node", "")), "born_step": e["step"],
                        "creator": e["agent_id"], "roots": roots,
                        "parent": {int(a): None for a in roots},
                        "when": {int(a): e["step"] for a in roots},
                        "edges": [], "redundant": 0, "stifles": []}
        elif e["kind"] == "transmission":
            iid = p.get("item_id")
            c = cas.get(iid) if is_rumor(iid) else None
            if c is None:
                continue
            to, frm = e["agent_id"], p.get("from")
            if not isinstance(frm, int):
                continue
            if to in c["parent"]:
                c["redundant"] += 1
                continue
            c["parent"][to] = frm
            c["when"][to] = e["step"]
            c["edges"].append((e["step"], frm, to))
        elif e["kind"] == "rumor_stifle":
            iid = p.get("item_id")
            c = cas.get(iid) if is_rumor(iid) else None
            if c is not None:
                c["stifles"].append((e["step"], e["agent_id"]))
    return cas


def _depths(c: dict) -> dict[int, int]:
    """各 knower の深さ(根 = 0)。親が未解決(木の外から来た辺)は根扱いにする。"""
    depth: dict[int, int] = {a: 0 for a in c["roots"]}
    for _step, frm, to in c["edges"]:            # edges は時刻順 = 親が先に確定している
        depth[to] = depth.get(frm, 0) + 1
    for a in c["parent"]:
        depth.setdefault(a, 0)
    return depth


def cascade_metrics(c: dict) -> dict:
    """OASIS の 3 指標 + 補助量。"""
    depth = _depths(c)
    by_level: dict[int, int] = defaultdict(int)
    for d in depth.values():
        by_level[d] += 1
    steps = [s for s in c["when"].values()]
    return {
        "item_id": c["item_id"], "src_kind": c["src_kind"], "node": c["node"],
        "born_step": c["born_step"], "creator": c["creator"],
        "n_roots": len(c["roots"]),
        "scale": len(depth),                       # OASIS: 参加者数
        "depth": (max(depth.values()) if depth else 0),   # OASIS: 木の最大深さ
        "max_breadth": (max(by_level.values()) if by_level else 0),  # OASIS: 深さ別最大幅
        "breadth_by_level": {str(k): by_level[k] for k in sorted(by_level)},
        "n_edges": len(c["edges"]),
        "n_redundant": c["redundant"],             # 既知の相手への再到達(MT のトリガー)
        "n_stifle": len(c["stifles"]),
        "life_steps": (max(steps) - c["born_step"]) if steps else 0,
    }


# --------------------------------------------------------------------------- #
# stifler 化と古典理論値の比較
# --------------------------------------------------------------------------- #
def population_of(events: list[dict], explicit: int | None) -> tuple[int, str]:
    """母集団 N(= ignorant 比の分母)。**近似であることを出力に明記する**。"""
    if explicit and explicit > 0:
        return int(explicit), "explicit(--population)"
    seen: set[int] = set()
    for e in events:
        if e["agent_id"] >= 0:
            seen.add(e["agent_id"])
        for key in ("from", "to", "speaker"):
            v = e["payload"].get(key)
            if isinstance(v, int) and v >= 0:
                seen.add(v)
        for v in (e["payload"].get("hearers") or []):
            if isinstance(v, int) and v >= 0:
                seen.add(v)
        for v in (e["payload"].get("knowers") or []):
            if isinstance(v, int) and v >= 0:
                seen.add(v)
    return len(seen), "distinct agent_id in L1(近似)"


def spreader_series(c: dict) -> list[tuple[int, int]]:
    """(step, その時点の spreader 人数) の時系列を**再構成**する。

    「知った時点で spreader / stifle した時点で脱落」の決定論再生。
    忘却 TTL ON のランでは過大に出る(忘却は L1 に出ない)= docstring の限界 2。
    """
    marks: list[tuple[int, int]] = []
    for a, s in c["when"].items():
        marks.append((int(s), +1))
    for s, _a in c["stifles"]:
        marks.append((int(s), -1))
    marks.sort()
    out: list[tuple[int, int]] = []
    cur = 0
    for s, d in marks:
        cur += d
        if out and out[-1][0] == s:
            out[-1] = (s, cur)
        else:
            out.append((s, cur))
    return out


def theory_table(metrics: list[dict], cascades: dict, n_pop: int) -> dict:
    """DK/MT の理論値との比較表(**一致を期待しない**比較。docstring 参照)。"""
    rows = []
    for m in metrics:
        c = cascades[m["item_id"]]
        series = spreader_series(c)
        peak = max((n for _s, n in series), default=0)
        known = m["scale"]
        rows.append({
            "item_id": m["item_id"], "src_kind": m["src_kind"],
            "scale": known,
            "final_ignorant_frac": (round(1.0 - known / n_pop, 4) if n_pop else None),
            "peak_spreader_frac": (round(peak / n_pop, 4) if n_pop else None),
            "n_stifle": m["n_stifle"], "n_redundant": m["n_redundant"],
        })
    fi = [r["final_ignorant_frac"] for r in rows if r["final_ignorant_frac"] is not None]
    ps = [r["peak_spreader_frac"] for r in rows if r["peak_spreader_frac"] is not None]
    return {
        "population": n_pop,
        "theory": {"final_ignorant": DK_FINAL_IGNORANT,
                   "peak_spreader": round(DK_PEAK_SPREADER, 4)},
        "observed_mean": {
            "final_ignorant": (round(st.fmean(fi), 4) if fi else None),
            "peak_spreader": (round(st.fmean(ps), 4) if ps else None)},
        "deviation": {
            "final_ignorant": (round(st.fmean(fi) - DK_FINAL_IGNORANT, 4) if fi else None),
            "peak_spreader": (round(st.fmean(ps) - DK_PEAK_SPREADER, 4) if ps else None)},
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# まとめ + レポート
# --------------------------------------------------------------------------- #
def _agg(xs: list[int]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {"n": len(xs), "mean": round(st.fmean(xs), 4),
            "median": round(st.median(xs), 4), "max": max(xs)}


def summarize(source, population: int | None) -> dict:
    """`collect()` の集約(dict)でも、旧来の events list でも同じ結果を返す。

    W2-2 で入口を 2 系統にした。dict を渡す経路は L1 の総行数に比例するメモリを
    持たない(`collect` を参照)。list を渡す旧経路は既存テストとの互換のため維持。
    """
    bundle = source if isinstance(source, dict) else _bundle_from_events(source)
    events = bundle["events"]
    cascades = build_cascades(events)
    metrics = [cascade_metrics(c) for c in
               sorted(cascades.values(), key=lambda c: c["item_id"])]
    if population and population > 0:
        n_pop, pop_src = int(population), "explicit(--population)"
    else:
        n_pop = len(bundle["population_seen"])
        pop_src = "distinct agent_id in L1(近似)"
    n_rumor_tr = bundle["n_rumor_transmissions"]
    n_all_tr = bundle["n_all_transmissions"]
    by_src: dict[str, int] = defaultdict(int)
    for m in metrics:
        by_src[m["src_kind"]] += 1
    return {
        "schema": SCHEMA,
        "population": {"n": n_pop, "source": pop_src},
        "counts": {
            "rumors_born": len(metrics),
            "rumor_transmissions": n_rumor_tr,
            "vocab_transmissions": n_all_tr - n_rumor_tr,  # 凍結指標との引き算用
            "stifles": sum(m["n_stifle"] for m in metrics),
            "by_src_kind": {k: by_src[k] for k in sorted(by_src)},
        },
        "oasis": {                                  # OASIS 流の 3 指標(§2-1)
            "scale": _agg([m["scale"] for m in metrics]),
            "depth": _agg([m["depth"] for m in metrics]),
            "max_breadth": _agg([m["max_breadth"] for m in metrics]),
        },
        "theory": theory_table(metrics, cascades, n_pop),
        "cascades": metrics,
    }


def render(res: dict) -> str:
    c, o, t = res["counts"], res["oasis"], res["theory"]
    L = ["# 噂の伝播カスケード(IF-C)", "",
         f"- 噂の誕生: **{c['rumors_born']}** 件"
         f"(内訳: {json.dumps(c['by_src_kind'], ensure_ascii=False)})",
         f"- 噂の伝播: **{c['rumor_transmissions']}** 件"
         f"(同ランの語彙伝播 {c['vocab_transmissions']} 件とは item_id 接頭辞で分離済み)",
         f"- 語り手が黙った(stifler 化): **{c['stifles']}** 件",
         f"- 母集団 N = {res['population']['n']}({res['population']['source']})", "",
         "## OASIS 流の 3 指標", "",
         "| 指標 | n | 平均 | 中央値 | 最大 |", "|---|---|---|---|---|"]
    for key, label in (("scale", "scale(到達人数)"), ("depth", "depth(伝播木の深さ)"),
                       ("max_breadth", "max_breadth(世代最大幅)")):
        a = o[key]
        L.append(f"| {label} | {a['n']} | {a['mean']} | {a['median']} | {a['max']} |")
    L += ["", "> OASIS (2024) は自分たちの depth が実データより**顕著に浅い**と自認している。",
          "> 上表の depth を並べれば直接比較の主張が立つ。", "",
          "## Daley-Kendall / Maki-Thompson の理論値との比較", "",
          "| 量 | 理論値 | 本ラン(平均) | 差 |", "|---|---|---|---|",
          f"| 最終 ignorant 比 | {t['theory']['final_ignorant']} | "
          f"{t['observed_mean']['final_ignorant']} | {t['deviation']['final_ignorant']} |",
          f"| spreader ピーク比 | {t['theory']['peak_spreader']} | "
          f"{t['observed_mean']['peak_spreader']} | {t['deviation']['peak_spreader']} |", "",
          "> ★**一致を期待する表ではない**。古典モデルは「よく混ざった無限母集団」を前提に",
          "> するが、本シムは物理的な同席・生活時間・関係の偏りを持つ。この表は",
          "> 「LLM エージェントの街が古典噂モデルからどう逸脱するか」を数値で示すためにある。",
          "> spreader ピークは L1 からの**再構成**であり、忘却 TTL ON のランでは過大に出る。"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="噂の伝播カスケード解析(読み取り専用)")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(l1_events.parquet を含む)")
    ap.add_argument("--population", type=int, default=None,
                    help="ignorant 比の分母 N(既定: L1 に現れた相異なる agent_id 数=近似)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run_dir>/analysis)")
    args = ap.parse_args()

    # W2-2: 全件 list ではなくチャンク集約(在場 25万 × 10 日でも RAM が総行数に比例しない)
    res = summarize(collect(args.run_dir), args.population)
    out_dir = args.out or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "rumors.json")
    mpath = os.path.join(out_dir, "rumors_report.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=True)
    md = render(res)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[written] {jpath}")
    print(f"[written] {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
