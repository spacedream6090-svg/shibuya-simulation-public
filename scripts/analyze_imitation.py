#!/usr/bin/env python
"""第59バッチ スライス(c): 模倣連鎖検出レンズ(後処理・L1 を読むだけ)。

    python scripts/analyze_imitation.py runs/<name> [--window 7] [--lag 3]

設計正典: devlog Entry 50(3スライス精査)/ natural-coinage 方針(自然発生の観測のみ=模倣を促進しない)。
「B が行動 X を初めて(過去 W 日未実行)した直前 T 日以内に、X 実行済みの A と接触していたか」を **純事後**で
検出する。模倣を促す介入は一切しない(観測のみ)。時間順序 + 曝露の相関であって因果断定はしない。

行動語彙(機械的にカテゴリ化。固有名詞はスクリプト定数=BEHAVIOR_FIELDS に集約=コードに直書きしない):
  POI カテゴリ訪問(spend.cat / service_use.service)・出店(venture_open)・グループ(group_found/join)・
  自助系(service_use.service=lesson/gym は上の service_use で被覆)・イベント(event_host/attend)・
  娯楽(media_use.medium)・学び(study.subject)。交通/固定費など「場所選択でない」spend は除外。

検出規則(決定論):
  1. 行動 X の実行日 exec[X][agent] を全て収集。B の day d の実行が **初実行** = [d-W, d-1] に X 未実行。
  2. 曝露: 初実行 day d の直前 [d-T, d] に、B が「その接触日までに X 実行済み」の A と接触していた
     = 模倣候補 (A→B, X, lag=d-接触日)。接触 = speak/hear/dm + joint_activity(with) + event_attend(host)
     + group_join(founder) + chance_event(encounter, other)。
  3. 非曝露ベースライン比(相対リスク): **at-risk 個体日**(その日 X が初実行になり得る=過去 W 日未実行)を
     曝露あり/なしに層別し、各層で「初実行が起きた率」を出す。exposed_rate / unexposed_rate = 相対リスク。
     >1 なら曝露と初実行の関連(=模倣の兆候)。<=1 なら関連なし。曝露が無くても初実行は起きる(=ベースライン)。
  4. 模倣連鎖 A→B→C: 候補辺 (A→B, X) と (B→C, X) が同一 X で連なる(A の実行→B が模倣→C が B を模倣)三つ組。

計算量: O(behaviors × roster × days × T)級。実行者2人以上の行動のみ解析=長期ランでも軽い(純Python・乱数なし)。
決定論: 全ての集合・タイは id/day 昇順で解決。出力 JSON は sort_keys=True でバイト同一。

★正直さ注記(出力にも明記): 時間順序 + 曝露の **相関** であり **因果ではない**。曝露なしでも初実行は起きる
  (相対リスクのベースライン)。接触は同席・会話の観測であって「A を真似た」という主観の証明ではない。

出力:
  runs/<name>/imitation.json       … 行動別 模倣候補率・相対リスク・連鎖・全体集計
  runs/<name>/imitation_report.md  … 日本語の要約
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Windows cp932 対策
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                            # noqa: E402  (W2-3: ランの Δt の単一の源)
from society.observer import measure as m  # noqa: E402

# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 の 144。日の第一手段は sim_min//1440
# なので Δt 非依存で、この定数は sim_min 欠損時の後退経路にだけ効く。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY
WINDOW = 7          # 初実行の判定窓 W(過去 W 日 X 未実行なら「初実行」)
LAG = 3             # 曝露窓 T(初実行の直前 T 日以内の接触を候補にする)
MIN_DOERS = 2       # この人数未満しか実行していない行動は解析しない(模倣が定義できない)

# 行動 X の抽出規則(kind -> payload の下位種別フィールド。None=kind 全体が1行動)。
#   固有名詞はここに集約=検出ロジックに直書きしない(natural-coinage 方針の作法)。
BEHAVIOR_FIELDS: dict[str, str | None] = {
    "spend": "cat",            # 消費カテゴリ(food/nightlife/shop/...=場所選択)
    "service_use": "service",  # サービス(grooming/lesson/gym/clinic/laundry。自助系=lesson/gym を含む)
    "media_use": "medium",     # 娯楽メディア
    "study": "subject",        # 学び(教科)
    "venture_open": None,      # 出店
    "group_found": None,       # グループ結成
    "group_join": None,        # グループ加入
    "event_host": None,        # イベント主催
    "event_attend": None,      # イベント参加
}
# 「場所選択でない」spend カテゴリ(交通・固定費・医療・宿泊)は行動語彙から除外。
SPEND_EXCLUDE = {"taxi", "bus", "fixed_cost", "medical", "lodging"}


# 接触(同席・会話)を作る kind。build_contacts の if/elif の全数。
_CONTACT_KINDS = frozenset({"speak", "hear", "dm", "joint_activity",
                            "event_attend", "group_join", "chance_event"})

# W4-D: 本解析が読む kind = 行動語彙(BEHAVIOR_FIELDS)+ 接触(_CONTACT_KINDS)。
# `behavior_label` は `kind not in BEHAVIOR_FIELDS` で必ず None を返し、
# `build_contacts` も if/elif で他種を必ず読み飛ばすので、絞っても出力は不変。
# 全 kind 依存は 2 つ(roster のフォールバック・max_day)で、どちらも列走査へ。
# 定義の源は上の 2 表なので **写経せず合成する**。
WANT_KINDS = frozenset(BEHAVIOR_FIELDS) | _CONTACT_KINDS


def _load_events(run_dir: str) -> list[dict]:
    """L1 を `WANT_KINDS` だけ読み、`measure.load_events` と同一形の dict 列で返す。

    残る比例項: 行動イベントと接触イベントの行数(初実行判定が全期間の日次集合を
    要求するので、行そのものは畳めない)。"""
    import l1_stream as ls
    if not ls.l1_paths(run_dir):        # m.load_events と同じく「無ければ落とす」
        raise FileNotFoundError(os.path.join(run_dir, "l1_events.parquet"))
    return list(ls.iter_events(run_dir, kinds=WANT_KINDS))


def _day_of(e: dict, spd: int = STEPS_PER_DAY) -> int:
    sm = e.get("sim_min")
    if sm is None:
        return int(e["step"]) // int(spd)
    return int(sm) // 1440


def behavior_label(kind: str, payload: dict) -> str | None:
    """1 イベント → 行動 X ラベル(対象外は None)。"""
    if kind not in BEHAVIOR_FIELDS:
        return None
    field = BEHAVIOR_FIELDS[kind]
    if field is None:
        return kind
    sub = payload.get(field) if isinstance(payload, dict) else None
    if sub in (None, ""):
        return None
    if kind == "spend" and str(sub) in SPEND_EXCLUDE:
        return None
    return f"{kind}:{sub}"


# --------------------------------------------------------------------------- #
# 接触(同席・会話)グラフ(日次・対称)
# --------------------------------------------------------------------------- #
def build_contacts(events: list[dict], spd: int = STEPS_PER_DAY) -> dict[int, dict[int, set]]:
    """contacts[a][day] = その日 a が接触した相手集合(対称)。"""
    contacts: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))

    def _add(a, b, d):
        if isinstance(a, int) and isinstance(b, int) and a >= 0 and b >= 0 and a != b:
            contacts[a][d].add(b)
            contacts[b][d].add(a)

    for e in events:
        k, aid, p, d = e["kind"], e["agent_id"], e["payload"], _day_of(e, spd)
        if not isinstance(aid, int) or aid < 0:
            continue
        if k == "speak":
            for h in p.get("hearers", []) or []:
                _add(aid, h, d)
        elif k == "hear":
            _add(aid, p.get("speaker"), d)
        elif k == "dm":
            _add(aid, p.get("to"), d)
        elif k == "joint_activity":
            members = [x for x in (p.get("with") or []) if isinstance(x, int)]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    _add(members[i], members[j], d)
        elif k == "event_attend":
            _add(aid, p.get("host"), d)
        elif k == "group_join":
            _add(aid, p.get("founder"), d)
        elif k == "chance_event" and p.get("type") == "encounter":
            _add(aid, p.get("other"), d)
    return contacts


# --------------------------------------------------------------------------- #
# メイン解析
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, window: int = WINDOW, lag: int = LAG,
            min_doers: int = MIN_DOERS, top_k: int = 20) -> dict:
    import l1_stream as ls
    events = _load_events(run_dir)
    agents = m.load_agents(run_dir)
    name_of = {int(a["id"]): a.get("name", f"a{a['id']}") for a in agents if "id" in a}
    # roster / max_day は **全 kind** から決まる(街に出た全員・最後にイベントがあった日)。
    # 絞った events から作ると人と日が落ちるので、専用の列走査に置き換える。
    roster = sorted(name_of) or sorted(ls.distinct_agent_ids(run_dir))
    run_name = os.path.basename(os.path.normpath(run_dir))
    # W2-3: sim_min 欠損時の後退経路で使う 1 日あたり step 数(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    max_day = ls.max_day(run_dir, spd)

    # 行動 X の実行日 exec[X][agent] = set(days) / first_exec[X][agent] = 最小日
    exec_days: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for e in events:
        aid = e["agent_id"]
        if not isinstance(aid, int) or aid < 0:
            continue
        x = behavior_label(e["kind"], e["payload"])
        if x is not None:
            exec_days[x][aid].add(_day_of(e, spd))
    first_exec = {x: {a: min(ds) for a, ds in by.items()} for x, by in exec_days.items()}

    contacts = build_contacts(events, spd)

    # 解析対象の行動 = 実行者が min_doers 人以上(模倣が定義できる)
    behaviors = sorted(x for x, by in exec_days.items() if len(by) >= min_doers)

    per_behavior: list[dict] = []
    cand_edges: list[tuple] = []          # (A, B, X, lag, day_B)
    tot_exposed_start = tot_unexposed_start = 0
    tot_exposed_risk = tot_unexposed_risk = 0
    lags: list[int] = []

    for x in behaviors:
        fe = first_exec[x]
        ex = exec_days[x]
        exposed_risk = unexposed_risk = exposed_start = unexposed_start = 0
        for a in roster:
            a_days = ex.get(a, set())
            for d in range(0, max_day + 1):
                # at-risk = 過去 W 日 X 未実行(その日 X が初実行になり得る)
                if any((d - w) in a_days for w in range(1, window + 1)):
                    continue
                started = d in a_days               # この日 X を実行(at-risk 前提=初実行)
                # 曝露: [d-lag, d] に「接触日までに X 実行済み」の A と接触(最小 lag を採る)
                hit_a = None
                hit_lag = None
                for dc in range(d, max(-1, d - lag) - 1, -1):
                    partners = contacts.get(a, {}).get(dc)
                    if not partners:
                        continue
                    for other in sorted(partners):
                        f = fe.get(other)
                        if f is not None and f <= dc:
                            hit_a, hit_lag = other, d - dc
                            break
                    if hit_a is not None:
                        break
                exposed = hit_a is not None
                if exposed:
                    exposed_risk += 1
                    if started:
                        exposed_start += 1
                        cand_edges.append((hit_a, a, x, hit_lag, d))
                        lags.append(hit_lag)
                else:
                    unexposed_risk += 1
                    if started:
                        unexposed_start += 1
        n_fresh = exposed_start + unexposed_start
        if n_fresh == 0:
            continue
        exp_rate = exposed_start / exposed_risk if exposed_risk else 0.0
        unexp_rate = unexposed_start / unexposed_risk if unexposed_risk else 0.0
        rr = (exp_rate / unexp_rate) if unexp_rate > 0 else None
        per_behavior.append({
            "behavior": x, "n_fresh_starts": n_fresh,
            "exposed_starts": exposed_start, "unexposed_starts": unexposed_start,
            "imitation_candidate_rate": round(exposed_start / n_fresh, 4),
            "exposed_start_rate": round(exp_rate, 6),
            "unexposed_start_rate": round(unexp_rate, 6),
            "relative_risk": round(rr, 4) if rr is not None else None,
        })
        tot_exposed_start += exposed_start
        tot_unexposed_start += unexposed_start
        tot_exposed_risk += exposed_risk
        tot_unexposed_risk += unexposed_risk

    per_behavior.sort(key=lambda r: (-r["n_fresh_starts"], r["behavior"]))

    # 模倣連鎖 A→B→C(同一 X・候補辺が連なる。B が被模倣者かつ模倣者)
    edges_by_x: dict[str, list] = defaultdict(list)
    for a, b, x, lg, day in cand_edges:
        edges_by_x[x].append((a, b, day))
    chains: list[dict] = []
    for x, edges in edges_by_x.items():
        by_target = defaultdict(list)     # b -> [(a, day_b)]
        by_source = defaultdict(list)     # b -> [(c, day_c)]
        for a, b, day in edges:
            by_target[b].append((a, day))
            by_source[a].append((b, day))
        for b in sorted(set(by_target) & set(by_source)):
            for a, day_b in by_target[b]:
                for c, day_c in by_source[b]:
                    if c != a and c != b and day_c >= day_b:  # 時間順序 A→B→C
                        chains.append({"behavior": x, "chain": [a, b, c],
                                       "names": [name_of.get(i, f"a{i}") for i in (a, b, c)]})
    chains.sort(key=lambda ch: (ch["behavior"], ch["chain"]))

    tot_fresh = tot_exposed_start + tot_unexposed_start
    pooled_exp = tot_exposed_start / tot_exposed_risk if tot_exposed_risk else 0.0
    pooled_unexp = tot_unexposed_start / tot_unexposed_risk if tot_unexposed_risk else 0.0
    pooled_rr = (pooled_exp / pooled_unexp) if pooled_unexp > 0 else None

    return {
        "run": run_name,
        "n_agents": len(roster),
        "n_days": max_day + 1,
        "params": {"window_W": window, "lag_T": lag, "min_doers": min_doers,
                   "steps_per_day": spd,
                   "contact_channels": "speak/hear/dm + joint_activity + event_attend(host) "
                                       "+ group_join(founder) + chance_event(encounter)",
                   "behaviors_fields": {k: v for k, v in BEHAVIOR_FIELDS.items()}},
        "n_behaviors_analyzed": len(behaviors),
        "overall": {
            "n_fresh_starts": tot_fresh,
            "exposed_starts": tot_exposed_start, "unexposed_starts": tot_unexposed_start,
            "imitation_candidate_rate": round(tot_exposed_start / tot_fresh, 4) if tot_fresh else None,
            "exposed_start_rate": round(pooled_exp, 6),
            "unexposed_start_rate": round(pooled_unexp, 6),
            "relative_risk": round(pooled_rr, 4) if pooled_rr is not None else None,
            "n_candidate_edges": len(cand_edges),
            "mean_lag_days": round(sum(lags) / len(lags), 3) if lags else None,
            "n_chains": len(chains),
        },
        "per_behavior": per_behavior[:top_k],
        "chains": chains[:top_k],
        "note": ("時間順序+曝露の相関であり因果ではない。曝露なしでも初実行は起きる(相対リスクのベースライン)。"
                 "接触は同席・会話の観測であって『A を真似た』という主観の証明ではない。模倣を促す介入はしない"
                 "(自然発生の観測のみ=natural-coinage 方針)。"),
    }


# --------------------------------------------------------------------------- #
# 日本語レポート
# --------------------------------------------------------------------------- #
def write_report(result: dict, path: str) -> None:
    o = result["overall"]
    L: list[str] = []
    L.append(f"# 模倣連鎖検出レンズ 観測レポート: {result['run']}\n")
    L.append("「初実行の直前に、その行動をすでにしていた人と接触していたか」を **純事後**で検出した結果。")
    L.append("模倣を促す介入はしない(自然発生の観測のみ)。時間順序+曝露の相関であり因果ではない。\n")
    p = result["params"]
    L.append("## 設定")
    L.append(f"- エージェント {result['n_agents']} / 日数 {result['n_days']} / "
             f"解析対象行動 {result['n_behaviors_analyzed']}(実行者≥{p['min_doers']}人)")
    L.append(f"- 初実行窓 W={p['window_W']}日 / 曝露窓 T={p['lag_T']}日")
    L.append(f"- 接触チャネル: {p['contact_channels']}\n")
    L.append("## 全体集計")
    L.append(f"- 初実行 {o['n_fresh_starts']} 件 / 曝露あり {o['exposed_starts']} ・ 曝露なし {o['unexposed_starts']}")
    imr = o["imitation_candidate_rate"]
    L.append(f"- 模倣候補率(初実行のうち曝露ありの割合)= "
             f"{'—' if imr is None else f'{imr * 100:.1f}%'}")
    L.append(f"- **非曝露ベースライン比(相対リスク)** = "
             f"{o['relative_risk'] if o['relative_risk'] is not None else '—'}"
             f"(曝露層の初実行率 {o['exposed_start_rate']} / 非曝露層 {o['unexposed_start_rate']})")
    if o["relative_risk"] is not None:
        if o["relative_risk"] > 1.0:
            L.append(f"  → 相対リスク>1 = 曝露と初実行の**関連あり**(模倣の兆候。因果ではない)")
        else:
            L.append(f"  → 相対リスク≤1 = 曝露と初実行の関連は出ていない(曝露なしでも同程度に起きる)")
    L.append(f"- 模倣候補辺 {o['n_candidate_edges']} / 平均 lag "
             f"{o['mean_lag_days'] if o['mean_lag_days'] is not None else '—'} 日 / 連鎖(A→B→C){o['n_chains']} 件\n")
    L.append("## 行動別(初実行の多い順)")
    L.append("| 行動 X | 初実行 | 模倣候補率 | 相対リスク(曝露/非曝露) |")
    L.append("|---|---|---|---|")
    for r in result["per_behavior"]:
        rr = r["relative_risk"]
        L.append(f"| {r['behavior']} | {r['n_fresh_starts']} | "
                 f"{r['imitation_candidate_rate'] * 100:.1f}% | {rr if rr is not None else '—'} |")
    L.append("")
    if result["chains"]:
        L.append("## 模倣連鎖 A→B→C(同一行動が伝播した三つ組)")
        for ch in result["chains"]:
            L.append(f"- [{ch['behavior']}] {' → '.join(ch['names'])}")
        L.append("")
    L.append(f"> {result['note']}\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第59バッチ スライスc: 模倣連鎖検出レンズの事後分析(L1 を読むだけ・純事後)")
    ap.add_argument("run_dir", help="例: runs/day30")
    ap.add_argument("--window", type=int, default=WINDOW, help="初実行の判定窓 W 日(既定 7)")
    ap.add_argument("--lag", type=int, default=LAG, help="曝露窓 T 日(既定 3)")
    ap.add_argument("--min-doers", type=int, default=MIN_DOERS,
                    help="解析対象にする最小実行者数(既定 2)")
    ap.add_argument("--top-k", type=int, default=20, help="出力する行動/連鎖の上限(既定 20)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    result = analyze(args.run_dir, window=args.window, lag=args.lag,
                     min_doers=args.min_doers, top_k=args.top_k)
    js_path = os.path.join(args.run_dir, "imitation.json")
    md_path = os.path.join(args.run_dir, "imitation_report.md")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
    write_report(result, md_path)

    o = result["overall"]
    print(f"[imitation] {args.run_dir}: 初実行 {o['n_fresh_starts']} / 模倣候補率 "
          f"{o['imitation_candidate_rate']} / 相対リスク {o['relative_risk']} / 連鎖 {o['n_chains']}")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
