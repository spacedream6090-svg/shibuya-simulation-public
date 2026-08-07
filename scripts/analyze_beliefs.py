#!/usr/bin/env python
"""第73バッチ Part B: 真偽台帳 = 信念・伝播木・検証行動の事後分析(読み取り専用)。

    python scripts/analyze_beliefs.py runs/<name> [--top-k 15] [--allow-tier-mismatch]

設計正典: docs/plans/source/dual-mode-instruments.md Part B /
          docs/plans/dual-mode-observe-verify-plan.md §2 第73行。実装 src/society/truth_ledger.py。

入力は 2 つだけ(**シムには一切触れない**):
  1. runs/<name>/l1_events.parquet … belief_update / belief_transmit / belief_verify
     (= エージェントが持っている信念とその由来。真値は入っていない)
  2. runs/<name>/beliefs_ledger.json … 真偽台帳(ground truth)。**真値はここにしかない**
     ので、信念距離を出すにはこの 1 ファイルが要る。L1 と分離してあるのは、真値が L1 の
     消費経路(行間ダイジェスト等)へ紛れ込む余地を構造的に断つため。

出す指標(すべて決定論・乱数ゼロ・純 Python):
  A. 信念距離 … |信念の値 − 真値| の時系列(step ビン)と agent 別。直接目撃=0、伝聞は
     ホップとともに増える。検証で 0 へ戻る。
  B. 検証率 … fact 別 / agent 別。分母は「その fact の信念を持った者」。
     ★ match=="recent"(= about の文字列から対象を特定できず、直近の未検証信念へ後退した分)を
       別建てで出す。水増しを疑う読者がそこを引けるようにするため。
  C. 訂正の伝播効率 … 検証済みエージェントと接触(= belief_transmit の受け手になる)した後、
     相手の信念が真値へどれだけ動いたか(接触前後の距離差の平均)。
  D. 伝播木 … fact ごとの (親→子, step, hop) 辺リスト。JSON に丸ごと出す。

出力:
  runs/<name>/beliefs.json         … 上記すべて(sort_keys=True でバイト同一)
  runs/<name>/beliefs_report.md    … 日本語の要約

★正直さ注記(出力にも明記):
  - 真値は [0,1] の 1 次元スカラー(ミニマル版)。truth_status 5 分類はフル版の担当。
  - 「訂正の伝播効率」は接触前後の**関連**であって因果の証明ではない(交絡・逆因果を排除しない)。
  - 検証行動を取るかはエージェントの判断に委ねてある(誘導文言なし)。したがって検証数が 0 の
    ランはありうる。0 を「機構が壊れている」と読まないこと(誰が確かめるかが観測対象そのもの)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Windows cp932 対策(コンソール出力の文字化け・例外を防ぐ)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # noqa: BLE001
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                # 同ディレクトリの l1_stream を import
    sys.path.insert(0, _HERE)

import l1_stream as ls                                    # noqa: E402  (W2-2)
from society import registry as R                         # noqa: E402

LEDGER_NAME = "beliefs_ledger.json"
# W2-2: L1 から読む種。`belief_events` が絞る集合と**同一**でなければならない
# (observer/schema.py が登録する belief_* はこの 3 種で全部)。ここで Arrow レベルに
# 絞ることで、解析が使わない種の Python dict を 1 個も作らずに済む。
BELIEF_KINDS = ("belief_update", "belief_transmit", "belief_verify")
NOTE = ("観測記録であって因果の証明ではない(訂正の伝播効率は接触前後の関連)。"
        "真値は [0,1] の1次元スカラー(ミニマル版)。検証行動は誘導していないので"
        "検証数 0 のランはありうる。")


# --------------------------------------------------------------------------- #
# 入力
# --------------------------------------------------------------------------- #
def load_ledger(run_dir: str) -> dict:
    path = os.path.join(run_dir, LEDGER_NAME)
    if not os.path.exists(path):
        return {"n_facts": 0, "facts": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def belief_events(events: list[dict]) -> list[dict]:
    """belief_* の 3 種だけを step→(元の並び)で取り出す(L1 の並びは既に決定論)。"""
    return [e for e in events if str(e.get("kind", "")).startswith("belief_")]


def load_belief_events(run_dir: str) -> list[dict]:
    """L1 から belief_* だけを**逐次読み**で取り出す(W2-2)。

    旧実装は `measure.load_events(run_dir)` で L1 を全件 RAM 展開してから
    `belief_events` で絞っていた。在場 25万 × 10 日の L1 は 40.6 億件なので
    (`docs/plans/proposal-dp-u3-observe-250k.md` §2-4)、解析が使うのが数万件でも
    40.6 億個の Python dict を作って捨てることになり、確実に破綻する。

    **返り値は 1 バイトも変わらない**: `l1_stream.iter_events` の要素は
    `load_events` の要素とキー順まで同一で、`kinds=` は「その条件で filter した
    部分列」を順序を保って返すことが `tests/test_l1_stream.py` で固定されている。
    さらに絞り込みの結果へ `belief_events` を**もう一度**掛けているので、
    絞る集合は旧実装の述語そのものが最終決定する(二重掛けは冪等)。
    """
    if not ls.l1_paths(run_dir):
        # 欠測を偽の値で埋めない(旧実装も L1 が無ければ例外で止まっていた)。
        raise SystemExit(f"[beliefs] l1_events.parquet が無い: {run_dir}")
    return belief_events(list(ls.iter_events(run_dir, kinds=BELIEF_KINDS)))


# --------------------------------------------------------------------------- #
# A. 信念距離(時系列 + agent 別) / D. 伝播木
# --------------------------------------------------------------------------- #
def trace(bevents: list[dict], truth: dict[str, float], bin_steps: int) -> dict:
    """belief_update を時系列に再生し、距離・伝播木・保持者を組み立てる。

    返り値:
      series   … [{step_bin, n_beliefs, distance_mean, hop_mean, verified_rate}]
      by_agent … agent_id -> {n, distance_mean, verified, hops}
      tree     … fact_id -> [{parent, child, step, hop, cause}]
      holders  … fact_id -> [agent_id]
      updates  … 再生に使った (step, agent, fact, dist, verified) の列(C で再利用)
    """
    state: dict[tuple[int, str], dict] = {}     # (agent, fact) -> {"dist", "hop", "ver"}
    tree: dict[str, list] = defaultdict(list)
    holders: dict[str, set] = defaultdict(set)
    updates: list[dict] = []
    bins: dict[int, list] = defaultdict(list)

    for e in bevents:
        if e["kind"] != "belief_update":
            continue
        p = e["payload"]
        fid = p.get("fact")
        if fid is None or fid not in truth:
            continue
        aid = int(e["agent_id"])
        step = int(e["step"])
        dist = abs(float(p.get("value", 0.0)) - float(truth[fid]))
        ver = p.get("verified")
        state[(aid, fid)] = {"dist": dist, "hop": int(p.get("hop", 0)),
                             "ver": ver, "step": step}
        holders[fid].add(aid)
        parent = int(p.get("from", -1))
        if p.get("cause") == "transmit" and parent >= 0:
            tree[fid].append({"parent": parent, "child": aid, "step": step,
                              "hop": int(p.get("hop", 0)), "cause": "transmit"})
        updates.append({"step": step, "agent": aid, "fact": fid, "dist": dist,
                        "verified": ver, "cause": p.get("cause")})
        # その時点の全体状態を step ビンへ積む(rolling ではなく「更新のたびの断面」)
        b = step - (step % max(1, bin_steps))
        bins[b].append((sum(s["dist"] for s in state.values()) / len(state),
                        sum(s["hop"] for s in state.values()) / len(state),
                        sum(1 for s in state.values() if s["ver"]) / len(state),
                        len(state)))

    series = []
    for b in sorted(bins):
        rows = bins[b]
        series.append({
            "step_bin": b,
            "n_beliefs": rows[-1][3],
            "distance_mean": round(sum(r[0] for r in rows) / len(rows), 6),
            "hop_mean": round(sum(r[1] for r in rows) / len(rows), 6),
            "verified_rate": round(sum(r[2] for r in rows) / len(rows), 6),
        })

    by_agent: dict[int, dict] = {}
    for (aid, _fid), s in sorted(state.items()):
        rec = by_agent.setdefault(aid, {"n": 0, "dist_sum": 0.0, "verified": 0,
                                        "hop_sum": 0})
        rec["n"] += 1
        rec["dist_sum"] += s["dist"]
        rec["hop_sum"] += s["hop"]
        rec["verified"] += 1 if s["ver"] else 0
    for rec in by_agent.values():
        n = max(1, rec["n"])
        rec["distance_mean"] = round(rec.pop("dist_sum") / n, 6)
        rec["hop_mean"] = round(rec.pop("hop_sum") / n, 6)
        rec["verified_rate"] = round(rec["verified"] / n, 6)
    return {"series": series, "by_agent": by_agent,
            "tree": {k: v for k, v in sorted(tree.items())},
            "holders": {k: sorted(v) for k, v in sorted(holders.items())},
            "updates": updates, "final": state}


def tree_stats(tree: dict) -> dict:
    """伝播木の形(枝数・最大深さ・分岐)。決定論。"""
    n_edges = sum(len(v) for v in tree.values())
    depth = 0
    fanout: list[int] = []
    for edges in tree.values():
        depth = max([depth] + [int(x["hop"]) for x in edges])
        by_parent: dict[int, int] = defaultdict(int)
        for x in edges:
            by_parent[x["parent"]] += 1
        fanout.extend(by_parent.values())
    return {"n_trees": len(tree), "n_edges": n_edges, "max_hop": depth,
            "fanout_mean": round(sum(fanout) / len(fanout), 4) if fanout else 0.0}


# --------------------------------------------------------------------------- #
# B. 検証率(fact 別 / agent 別)
# --------------------------------------------------------------------------- #
def verification(bevents: list[dict], final: dict) -> dict:
    """検証行動の内訳(how × outcome)と、fact 別・agent 別の検証率。"""
    tries = [e for e in bevents if e["kind"] == "belief_verify"]
    by_how: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    by_match: dict[str, int] = defaultdict(int)
    actors: dict[int, int] = defaultdict(int)
    for e in tries:
        p = e["payload"]
        by_how[str(p.get("how"))][str(p.get("outcome"))] += 1
        by_match[str(p.get("match"))] += 1
        if p.get("outcome") == "ok":
            actors[int(e["agent_id"])] += 1

    per_fact: dict[str, dict] = defaultdict(lambda: {"holders": 0, "verified": 0})
    per_agent: dict[int, dict] = defaultdict(lambda: {"beliefs": 0, "verified": 0})
    for (aid, fid), s in final.items():
        per_fact[fid]["holders"] += 1
        per_agent[aid]["beliefs"] += 1
        if s["ver"]:
            per_fact[fid]["verified"] += 1
            per_agent[aid]["verified"] += 1
    for rec in list(per_fact.values()) + list(per_agent.values()):
        base = rec.get("holders") or rec.get("beliefs") or 1
        rec["rate"] = round(rec["verified"] / base, 6)
    return {
        "n_actions": len(tries),
        "by_how": {k: dict(sorted(v.items())) for k, v in sorted(by_how.items())},
        "by_match": dict(sorted(by_match.items())),
        "verifiers": dict(sorted(actors.items())),
        "per_fact": {k: v for k, v in sorted(per_fact.items())},
        "per_agent": {k: v for k, v in sorted(per_agent.items())},
    }


# --------------------------------------------------------------------------- #
# C. 訂正の伝播効率(検証済みの相手から聞いた後、信念は真値へ動いたか)
# --------------------------------------------------------------------------- #
def correction(bevents: list[dict], truth: dict[str, float]) -> dict:
    """belief_transmit を時系列に再生し、送り手が検証済みだった場合の受け手の距離変化を測る。

    ★ 関連であって因果ではない(送り手が検証済みであることと、受け手が動くことの共起)。"""
    dist: dict[tuple[int, str], float] = {}
    ver: dict[tuple[int, str], object] = {}
    gains_ver: list[float] = []
    gains_unver: list[float] = []
    pending: dict[tuple[int, str], tuple[float, bool]] = {}

    for e in bevents:
        p = e["payload"]
        fid = p.get("fact")
        if fid is None or fid not in truth:
            continue
        if e["kind"] == "belief_transmit":
            src = int(e["agent_id"])
            src_ver = bool(ver.get((src, fid)))
            for rid in (p.get("to") or []):
                key = (int(rid), fid)
                if key in dist:                    # 接触前の距離を控える(初回獲得は対象外)
                    pending[key] = (dist[key], src_ver)
            continue
        if e["kind"] != "belief_update":
            continue
        key = (int(e["agent_id"]), fid)
        new = abs(float(p.get("value", 0.0)) - float(truth[fid]))
        prev = pending.pop(key, None)
        if prev is not None:
            before, src_ver = prev
            (gains_ver if src_ver else gains_unver).append(before - new)
        dist[key] = new
        ver[key] = p.get("verified")

    def _mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    return {"n_from_verified": len(gains_ver), "n_from_unverified": len(gains_unver),
            "gain_from_verified": _mean(gains_ver),
            "gain_from_unverified": _mean(gains_unver),
            "delta": round(_mean(gains_ver) - _mean(gains_unver), 6)}


# --------------------------------------------------------------------------- #
# 統合
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, *, bin_steps: int = 24, top_k: int = 15) -> dict:
    ledger = load_ledger(run_dir)
    facts = {f["id"]: f for f in ledger.get("facts", [])}
    truth = {fid: float(f["value"]) for fid, f in facts.items()}
    bevents = load_belief_events(run_dir)

    tr = trace(bevents, truth, bin_steps)
    ver = verification(bevents, tr["final"])
    corr = correction(bevents, truth)
    stats = tree_stats(tr["tree"])

    kinds: dict[str, int] = defaultdict(int)
    for f in facts.values():
        kinds[str(f.get("kind"))] += 1
    n_bel = len(tr["final"])
    dist_final = (round(sum(s["dist"] for s in tr["final"].values()) / n_bel, 6)
                  if n_bel else 0.0)
    top = sorted(tr["by_agent"].items(),
                 key=lambda kv: (-kv[1]["n"], kv[0]))[:top_k]
    return {
        "run_dir": run_dir,
        "ledger": {"n_facts": len(facts), "by_kind": dict(sorted(kinds.items()))},
        "beliefs": {"n_agent_fact_pairs": n_bel,
                    "distance_mean_final": dist_final,
                    "n_updates": len(tr["updates"]),
                    "n_transmits": sum(1 for e in bevents
                                       if e["kind"] == "belief_transmit")},
        "series": tr["series"],
        "by_agent_top": [{"agent": a, **rec} for a, rec in top],
        "verification": ver,
        "correction": corr,
        "tree_stats": stats,
        "tree": tr["tree"],
        "holders": tr["holders"],
        "note": NOTE,
    }


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    return f"{100.0 * float(x):.1f}%"


def write_report(result: dict, path: str) -> None:
    L = ["# 真偽台帳レンズ(信念・伝播木・検証行動)", ""]
    led, bel = result["ledger"], result["beliefs"]
    L.append(f"- 対象: `{result['run_dir']}`")
    L.append(f"- 台帳の fact: {led['n_facts']} 件 "
             f"({', '.join(f'{k}×{v}' for k, v in led['by_kind'].items()) or '-'})")
    L.append(f"- 信念(agent×fact): {bel['n_agent_fact_pairs']} 対 / "
             f"更新 {bel['n_updates']} 件 / 伝聞送出 {bel['n_transmits']} 件")
    L.append(f"- 最終の信念距離(平均 |信念 − 真値|): **{bel['distance_mean_final']}**")
    L.append("")
    L.append("## 信念距離の推移(step ビン)")
    L.append("| step | 信念数 | 距離 | 平均ホップ | 検証率 |")
    L.append("|---|---|---|---|---|")
    for row in result["series"]:
        L.append(f"| {row['step_bin']} | {row['n_beliefs']} | {row['distance_mean']} "
                 f"| {row['hop_mean']} | {_pct(row['verified_rate'])} |")
    L.append("")
    ts = result["tree_stats"]
    L.append("## 伝播木")
    L.append(f"- 木の数(= 伝わった fact 数): {ts['n_trees']} / 枝 {ts['n_edges']} / "
             f"最大ホップ {ts['max_hop']} / 平均分岐 {ts['fanout_mean']}")
    L.append("")
    v = result["verification"]
    L.append("## 検証行動(誰が確かめに行ったか)")
    L.append(f"- 実行 {v['n_actions']} 件 / 対象特定の内訳 {v['by_match']}")
    if v["by_how"]:
        L.append("")
        L.append("| 手段 | 結果の内訳 |")
        L.append("|---|---|")
        for how, out in v["by_how"].items():
            L.append(f"| {how} | {out} |")
    else:
        L.append("- 検証行動は 1 件も起きなかった(**誘導していないので 0 はありうる**)。")
    L.append("")
    c = result["correction"]
    L.append("## 訂正の伝播効率(検証済みの相手から聞いた後、真値へ動いたか)")
    L.append(f"- 検証済み発 {c['n_from_verified']} 件: 平均 {c['gain_from_verified']} だけ真値へ接近")
    L.append(f"- 未検証発 {c['n_from_unverified']} 件: 平均 {c['gain_from_unverified']}")
    L.append(f"- 差 = **{c['delta']}**(正なら検証済み発の方が訂正力が強い)")
    L.append("")
    L.append(f"> {result['note']}")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第73バッチ Part B: 信念・伝播木・検証行動の事後分析(読み取り専用)")
    ap.add_argument("run_dir", help="例: runs/day30")
    ap.add_argument("--bin-steps", type=int, default=24,
                    help="信念距離の時系列ビン幅 [step](既定 24=4時間)")
    ap.add_argument("--top-k", type=int, default=15, help="agent 別の上位K(既定 15)")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="再現性等級の異なるランを混ぜてよい(第72 のラン間比較ガード)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    R.guard_or_die([args.run_dir], allow_mismatch=args.allow_tier_mismatch)

    result = analyze(args.run_dir, bin_steps=args.bin_steps, top_k=args.top_k)
    js_path = os.path.join(args.run_dir, "beliefs.json")
    md_path = os.path.join(args.run_dir, "beliefs_report.md")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
    write_report(result, md_path)

    print(f"[beliefs] {args.run_dir}: fact {result['ledger']['n_facts']} / "
          f"信念 {result['beliefs']['n_agent_fact_pairs']} 対 / "
          f"距離 {result['beliefs']['distance_mean_final']} / "
          f"伝播木 {result['tree_stats']['n_trees']}本(枝 {result['tree_stats']['n_edges']})/ "
          f"検証 {result['verification']['n_actions']} 件")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
