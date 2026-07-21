#!/usr/bin/env python
"""群別の行動統計(群のオントロジー S-c: 文化圏×経験の世界観共有群の効果分離)。

ランの l1 イベント + agents.json から、ontology 群(agents.json の ontology_group)ごとに
行動統計を出す。行動差はルールで書かず LLM に委ねる設計(nature-like)なので、ここは**観測**の
後処理として「群の割合を振った A/B」で群ごとの反応差を測るための集計器である。

使い方(群割合を振った A/B):
    # 例) ontology.enabled=true で composition(群構成比)を変えた2ラン(mock/実LLM)を出し、
    #     それぞれに本スクリプトを当てて群別サマリを比較する。
    #     A: stochastic 層に欧米圏旅行者を多めに / B: 少なめに、等。
    python scripts/analyze_groups.py runs/quakeA --shock-post 6 --shock-pre 6
    python scripts/analyze_groups.py runs/quakeB --shock-post 6 --shock-pre 6
    # → 各 runs/<name>/groups/group_summary.json と group_report.md を突き合わせて、
    #   「地震慣れ層(quake_exp=high)の割合 → 災害窓での退避率・移動潜時」の差を読む。

出力(<out>/groups/、既定は <run_dir>/groups/):
  ① 災害/ショック窓(disaster/world_event/scenario_shock の onset 前後)の反応:
       移動開始までの潜時(step)・窓内の移動距離(m)・退避(屋内→屋外=exit_building /
       圏外=exit_area)率。窓 = [onset, onset+shock_post]。
  ② 平常時(いずれのショック窓 [onset-shock_pre, onset+shock_post] にも属さない step):
       訪問先エントロピー(arrive ノード分布の bits)・会話数(speak+構造化会話 C2)・発話量(文字数)。
  ③ 群別サマリ表(group_summary.json + group_report.md)。

実装方針(既存 scripts の流儀):
  - ストリーミング読み(society.observer.measure.stream_events)。全件 materialize しない有界メモリ。
  - pyarrow + numpy + 純 Python のみ(pandas 禁止)。
  - ontology 未記録(OFF ラン=agents.json に ontology_group なし)は明示終了(捏造回避)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from society.observer.measure import stream_events        # noqa: E402

# ショックの起点(onset)とみなすイベント種(いずれも世界イベント agent_id=-1)。
_SHOCK_KINDS = frozenset({"disaster", "world_event", "scenario_shock"})
# onset ではない=窓を開かない phase(解除・終息・復旧)。
_NON_ONSET_PHASE = frozenset({"clear", "end", "recovery", "over", "resolved"})
# 移動開始(潜時の起点)とみなす種。
_MOVE_KINDS = frozenset({"route_start", "move_segment"})
# 退避=屋内→屋外(exit_building)/ 圏外(exit_area)。
_EVAC_KINDS = frozenset({"exit_building", "exit_area"})


# --------------------------------------------------------------------------- #
# 群 id ↔ agent
# --------------------------------------------------------------------------- #
# 文化圏軸(第1軸=従来)を指す --axis の別名。これ以外は axes.<name> の第2軸以降。
_CULTURE_AXIS = frozenset({None, "", "culture", "default", "ontology_group"})


def load_group_map(run_dir: str, axis: str | None = None
                   ) -> tuple[dict[int, str], dict[str, dict]]:
    """agents.json から {agent_id: group} と {group: meta(label 等)} を返す。

    axis=None/culture は従来どおり ontology_group(文化圏軸)。それ以外は ontology_axes[axis]
    (直交する第2軸)で群別集計する。指定軸の群が1体も無い(OFF/未設定)なら空(呼び出し側で明示終了)。"""
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    by_id: dict[int, str] = {}
    is_culture = axis in _CULTURE_AXIS
    for r in rows:
        if "id" not in r:
            continue
        g = (r.get("ontology_group") if is_culture
             else (r.get("ontology_axes") or {}).get(axis))
        if g is not None:
            by_id[int(r["id"])] = str(g)
    return by_id, {}


def load_group_defs(run_dir: str, axis: str | None = None) -> dict[str, dict]:
    """conf の ontology.groups(または axes.<axis>.groups)からラベル等の定義を読む(無ければ空=id 表示)。

    ラン再現の config.yaml は runs/<name>/config.yaml に落ちる(なければ conf/config.yaml)。"""
    is_culture = axis in _CULTURE_AXIS
    for rel in ("config.yaml", os.path.join("..", "..", "conf", "config.yaml")):
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            try:
                import yaml
                with open(path, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh) or {}
                onto = (doc.get("ontology") or {})
                groups = (onto.get("groups") if is_culture
                          else ((onto.get("axes") or {}).get(axis) or {}).get("groups")) or {}
                if groups:
                    return {str(k): dict(v or {}) for k, v in groups.items()}
            except Exception:
                pass
    return {}


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #
def entropy_bits(counter: Counter) -> float:
    """カウント分布のシャノンエントロピー(bits)。空/1点=0.0。"""
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def find_onsets(run_dir: str) -> list[int]:
    """ショック onset の step を昇順・重複排除で返す(pass 1: 種が少ない=軽い)。"""
    onsets: set[int] = set()
    for ev in stream_events(run_dir, columns=["step", "agent_id", "kind", "payload"]):
        if ev["kind"] in _SHOCK_KINDS:
            phase = str((ev.get("payload") or {}).get("phase", "")).lower()
            if phase not in _NON_ONSET_PHASE:
                onsets.add(int(ev["step"]))
    return sorted(onsets)


def _in_reaction_window(step: int, onsets: list[int], post: int) -> int | None:
    """step が属する反応窓 [onset, onset+post] の onset を返す(最も近い先行 onset)。無ければ None。"""
    hit = None
    for s in onsets:
        if s <= step <= s + post:
            hit = s                                    # 昇順なので最後に一致した=直近の onset
    return hit


def _in_any_full_window(step: int, onsets: list[int], pre: int, post: int) -> bool:
    """step がいずれかのショック全窓 [onset-pre, onset+post] に属するか(平常時の除外用)。"""
    for s in onsets:
        if s - pre <= step <= s + post:
            return True
    return False


def collect(run_dir: str, group_by_id: dict[int, str],
            onsets: list[int], pre: int, post: int) -> dict:
    """pass 2: ショック窓反応 + 平常時統計を1ストリームで集計(有界メモリ)。"""
    # --- 平常時(窓外)の per-agent 蓄積 ---
    visit_counts: dict[int, Counter] = defaultdict(Counter)   # arrive ノード分布
    speak_events: Counter = Counter()                          # speak 件数
    speak_chars: Counter = Counter()                           # 発話文字数
    conv_events: Counter = Counter()                           # 構造化会話 C2 件数
    # --- ショック窓反応 per (agent, onset) ---
    react: dict[tuple[int, int], dict] = {}

    def _ensure(aid: int, s: int) -> dict:
        key = (aid, s)
        r = react.get(key)
        if r is None:
            r = {"first_move": None, "dist": 0.0, "evac": False, "active": False}
            react[key] = r
        return r

    for ev in stream_events(run_dir,
                            columns=["step", "agent_id", "kind", "payload"]):
        aid = int(ev["agent_id"])
        if aid < 0 or aid not in group_by_id:
            continue
        step = int(ev["step"])
        kind = ev["kind"]
        pay = ev.get("payload") or {}

        # ショック反応窓(反応窓=[onset, onset+post])
        s = _in_reaction_window(step, onsets, post) if onsets else None
        if s is not None:
            r = _ensure(aid, s)
            r["active"] = True
            if kind in _MOVE_KINDS:
                if r["first_move"] is None:
                    r["first_move"] = step
                if kind == "move_segment":
                    r["dist"] += float(pay.get("dist_m", 0.0) or 0.0)
            if kind in _EVAC_KINDS:
                r["evac"] = True

        # 平常時(全窓の外)の統計
        if not onsets or not _in_any_full_window(step, onsets, pre, post):
            if kind == "arrive":
                node = pay.get("node") or pay.get("name")
                if node is not None:
                    visit_counts[aid][str(node)] += 1
            elif kind == "speak":
                speak_events[aid] += 1
                speak_chars[aid] += len(str(pay.get("text", "")))
            elif kind == "conversation":
                conv_events[aid] += 1

    return {"visit_counts": visit_counts, "speak_events": speak_events,
            "speak_chars": speak_chars, "conv_events": conv_events, "react": react}


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def summarize(group_by_id: dict[int, str], agg: dict, post: int) -> dict:
    """群別サマリ(①災害窓反応 ②平常時)を組み立てる。"""
    groups = sorted(set(group_by_id.values()))
    members: dict[str, list[int]] = defaultdict(list)
    for aid, g in group_by_id.items():
        members[g].append(aid)

    # 反応記録を群でまとめる
    react_by_group: dict[str, list[dict]] = defaultdict(list)
    for (aid, _s), r in agg["react"].items():
        if r["active"]:
            react_by_group[group_by_id[aid]].append(r)

    out: dict[str, dict] = {}
    for g in groups:
        ids = members[g]
        # ② 平常時
        ent = [entropy_bits(agg["visit_counts"][a]) for a in ids
               if sum(agg["visit_counts"][a].values()) > 0]
        spk = [agg["speak_events"][a] for a in ids]
        chars = [agg["speak_chars"][a] for a in ids]
        conv = [agg["conv_events"][a] for a in ids]
        # ① 災害窓反応
        rs = react_by_group[g]
        lat = [r["first_move"] for r in rs if r["first_move"] is not None]
        # 潜時 = 移動開始 step − onset step。onset は key に無いので後段で引く必要があるため、
        # ここでは react に onset を畳んで潜時を直接持たせる(collect 側で first_move は絶対 step)。
        out[g] = {
            "n_agents": len(ids),
            # ① 災害/ショック窓
            "shock_n_active": len(rs),
            "shock_move_rate": (round(len(lat) / len(rs), 4) if rs else None),
            "shock_latency_mean": (round(_mean(_latencies(rs)), 3)
                                   if _latencies(rs) else None),
            "shock_dist_mean": (round(_mean([r["dist"] for r in rs]), 2)
                                if rs else None),
            "shock_evac_rate": (round(sum(1 for r in rs if r["evac"]) / len(rs), 4)
                                if rs else None),
            # ② 平常時
            "visit_entropy_mean": (round(_mean(ent), 4) if ent else None),
            "speak_count_mean": round(_mean(spk), 3) if spk else None,
            "speak_chars_mean": round(_mean(chars), 2) if chars else None,
            "conversation_mean": round(_mean(conv), 3) if conv else None,
        }
    return out


def _latencies(records: list[dict]) -> list[float]:
    """反応記録から潜時(=移動開始 step − onset step)を取り出す。"""
    return [r["latency"] for r in records if r.get("latency") is not None]


# --------------------------------------------------------------------------- #
def analyze(run_dir: str, pre: int, post: int, axis: str | None = None) -> dict:
    group_by_id, _ = load_group_map(run_dir, axis)
    if not group_by_id:
        label = axis if axis not in _CULTURE_AXIS else "ontology_group"
        raise SystemExit(
            f"対象外: agents.json に {label} が無い(ontology OFF ランか、その軸が未設定)。"
            "ontology.enabled=true(第2軸は axes に該当軸)で走らせたランを渡すこと(捏造回避)。")
    onsets = find_onsets(run_dir)
    agg = collect(run_dir, group_by_id, onsets, pre, post)
    # 潜時(相対)を各反応記録へ畳む: first_move(絶対step)− 直近 onset。
    for (aid, s), r in agg["react"].items():
        r["latency"] = (r["first_move"] - s) if r["first_move"] is not None else None
    summ = summarize(group_by_id, agg, post)
    defs = load_group_defs(run_dir, axis)
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "axis": (axis if axis not in _CULTURE_AXIS else "culture"),
        "shock_pre": pre, "shock_post": post,
        "n_onsets": len(onsets), "onsets": onsets[:50],
        "groups": summ,
        "group_labels": {g: (defs.get(g, {}).get("label", g)) for g in summ},
    }


def _fmt(v) -> str:
    return "-" if v is None else (f"{v}")


def to_markdown(report: dict) -> str:
    lines = [f"# 群別行動統計: {report['run']}", ""]
    lines.append(f"- 集計軸: {report.get('axis', 'culture')}")
    lines.append(f"- ショック窓: onset 前 {report['shock_pre']} / 後 {report['shock_post']} step")
    lines.append(f"- 検出 onset 数: {report['n_onsets']}"
                 + (f"(step {report['onsets']})" if report["onsets"] else ""))
    lines.append("")
    lines.append("## ① 災害/ショック窓の反応")
    lines.append("| 群 | ラベル | 活動体数 | 移動率 | 潜時(step) | 移動距離(m) | 退避率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g, s in sorted(report["groups"].items()):
        lab = report["group_labels"].get(g, g)
        lines.append(f"| {g} | {lab} | {_fmt(s['shock_n_active'])} | "
                     f"{_fmt(s['shock_move_rate'])} | {_fmt(s['shock_latency_mean'])} | "
                     f"{_fmt(s['shock_dist_mean'])} | {_fmt(s['shock_evac_rate'])} |")
    lines.append("")
    lines.append("## ② 平常時")
    lines.append("| 群 | ラベル | 人数 | 訪問エントロピー(bits) | 会話数 | 発話量(字) | 構造化会話 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g, s in sorted(report["groups"].items()):
        lab = report["group_labels"].get(g, g)
        lines.append(f"| {g} | {lab} | {_fmt(s['n_agents'])} | "
                     f"{_fmt(s['visit_entropy_mean'])} | {_fmt(s['speak_count_mean'])} | "
                     f"{_fmt(s['speak_chars_mean'])} | {_fmt(s['conversation_mean'])} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(l1_events.parquet + agents.json)")
    ap.add_argument("--shock-pre", type=int, default=6,
                    help="平常時から除外する onset 前の step 数(既定6=1時間)")
    ap.add_argument("--shock-post", type=int, default=6,
                    help="反応窓 [onset, onset+post] の後幅 step 数(既定6=1時間)")
    ap.add_argument("--axis", default=None,
                    help="群別集計の軸(既定=文化圏 ontology_group。第2軸は info_behavior 等)")
    ap.add_argument("--out", default=None, help="出力先(既定 <run_dir>/groups)")
    args = ap.parse_args(argv)

    report = analyze(args.run_dir, args.shock_pre, args.shock_post, args.axis)
    out_dir = args.out or os.path.join(args.run_dir, "groups")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "group_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    md = to_markdown(report)
    with open(os.path.join(out_dir, "group_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    sys.stdout.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
