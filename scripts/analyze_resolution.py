"""入力解像度LOD(lod.input_res)の効果分離分析 — I3(第31バッチ 2026-07-14)。

計画: docs/plans/input-resolution-lod.md §3 I3・§4(取り下げ条件 K1-K4 の事前登録)。
R_input 水準 {narrow, mid, wide} 別に「技術的飢餓」チャネルと「低注意らしさ」チャネルを
分離集計する。K1 の判定材料 = 劣化が fallback/空応答(飢餓)に現れるのか、
話題の狭さ・行動レパートリー(低注意者らしさ)に現れるのか。

使い方:
    python scripts/analyze_resolution.py runs/<name> [--out DIR]

前提: ランが lod.input_res.enabled=true で実行され agents.json に input_res が記録
されていること(OFF ランは解析対象外として明示終了する。捏造回避)。
注意: mock はプロンプト内容に機械的に反応するだけなので、解像度効果の解釈は実LLMランのみ。
distinct-n は標本量に敏感なため、水準集計は「プール値」と「個体平均」を併記する。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:                    # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                                # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 の 144(= 従来と 1 ビットも変わらない)。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY

LEVELS = ("narrow", "mid", "wide")


# --------------------------------------------------------------------- 純関数
def char_ngrams(text: str, n: int) -> list[str]:
    s = "".join(text.split())
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """distinct-n = 文字 n-gram の種類数/延べ数(0件は 0.0)。"""
    grams: Counter = Counter()
    for t in texts:
        grams.update(char_ngrams(t, n))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def entropy(counts: Counter) -> float:
    """Shannon エントロピー(bit)。訪問先分布の広さの代理。"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# --------------------------------------------------------------------- 集計
def collect(run_dir: Path) -> tuple[dict[int, dict], dict[int, str]]:
    # W2-3: 「1 日」はこのランの run.dt_min で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    agents = json.loads((run_dir / "agents.json").read_text(encoding="utf-8"))
    level_of = {a["id"]: a.get("input_res") for a in agents}
    if not any(level_of.values()):
        sys.exit("このランは agents.json に input_res が無い(lod.input_res OFF)"
                 "=解像度分析の対象外。ON のランを指定すること。")

    cols = ["step", "agent_id", "kind", "payload"]
    t = pq.read_table(run_dir / "l1_events.parquet", columns=cols).to_pydict()

    per: dict[int, dict] = defaultdict(lambda: {
        "n_deliberate": 0, "n_fallback": 0, "n_speak": 0, "speak_empty": 0,
        "speak_error": 0, "texts": [], "places": Counter(),
        "place_days": defaultdict(set),  # day -> set(place)
        "n_sns_read": 0, "n_news_read": 0,
        "n_reflect": 0, "written_back": 0,
    })
    for step, aid, kind, payload in zip(t["step"], t["agent_id"], t["kind"],
                                        t["payload"]):
        aid = int(aid)
        if aid not in level_of:
            continue
        m = per[aid]
        if kind == "llm_deliberate":
            m["n_deliberate"] += 1
        elif kind == "fallback":
            m["n_fallback"] += 1
        elif kind == "speak":
            try:
                text = json.loads(payload).get("text") or ""
            except Exception:
                text = ""
            m["n_speak"] += 1
            if not text.strip():
                m["speak_empty"] += 1
            elif "__" in text:
                m["speak_error"] += 1
            else:
                m["texts"].append(text)
        elif kind == "arrive":
            try:
                name = json.loads(payload).get("name") or ""
            except Exception:
                name = ""
            if name:
                m["places"][name] += 1
                m["place_days"][int(step) // spd].add(name)
        elif kind == "sns_read":
            m["n_sns_read"] += 1
        elif kind == "news_read":
            m["n_news_read"] += 1
        elif kind == "reflect":
            m["n_reflect"] += 1
            try:
                if json.loads(payload).get("written_back"):
                    m["written_back"] += 1
            except Exception:
                pass
    return per, level_of


def agent_rows(per: dict[int, dict], level_of: dict[int, str]) -> list[dict]:
    rows = []
    for aid in sorted(per):
        m = per[aid]
        attempts = m["n_deliberate"] + m["n_fallback"]
        # レパートリー曲線: 日ごとの「累積ユニーク訪問先数」
        seen: set[str] = set()
        curve = {}
        for day in sorted(m["place_days"]):
            seen |= m["place_days"][day]
            curve[day] = len(seen)
        rows.append({
            "agent_id": aid,
            "level": level_of.get(aid) or "",
            "n_deliberate": m["n_deliberate"],
            "n_fallback": m["n_fallback"],
            "fallback_rate": m["n_fallback"] / attempts if attempts else 0.0,
            "n_speak": m["n_speak"],
            "speak_empty": m["speak_empty"],
            "speak_error": m["speak_error"],
            "distinct2": distinct_n(m["texts"], 2),
            "distinct3": distinct_n(m["texts"], 3),
            "n_places_unique": len(m["places"]),
            "place_entropy_bit": entropy(m["places"]),
            "n_sns_read": m["n_sns_read"],
            "n_news_read": m["n_news_read"],
            "n_reflect": m["n_reflect"],
            "written_back": m["written_back"],
            "repertoire_curve": json.dumps(curve, sort_keys=True),
        })
    return rows


def level_table(rows: list[dict], per: dict[int, dict],
                level_of: dict[int, str]) -> list[dict]:
    out = []
    for lv in LEVELS:
        sub = [r for r in rows if r["level"] == lv]
        if not sub:
            out.append({"level": lv, "n_agents": 0})
            continue
        pooled = [t for r in sub for t in per[r["agent_id"]]["texts"]]
        att = sum(r["n_deliberate"] + r["n_fallback"] for r in sub)
        fb = sum(r["n_fallback"] for r in sub)

        def mean(key: str) -> float:
            return sum(r[key] for r in sub) / len(sub)

        out.append({
            "level": lv,
            "n_agents": len(sub),
            # --- 飢餓チャネル(K1: ここに劣化が出たら「解像度」でなく技術的飢餓)---
            "fallback_rate": fb / att if att else 0.0,
            "speak_empty": sum(r["speak_empty"] for r in sub),
            "speak_error": sum(r["speak_error"] for r in sub),
            # --- 低注意らしさチャネル(仮説はこちらに単調な差)---
            "distinct2_pooled": distinct_n(pooled, 2),
            "distinct2_agent_mean": mean("distinct2"),
            "distinct3_pooled": distinct_n(pooled, 3),
            "places_unique_mean": mean("n_places_unique"),
            "place_entropy_mean": mean("place_entropy_bit"),
            # --- 参考(入力チャネル使用量・不変域の健全性)---
            "sns_read_mean": mean("n_sns_read"),
            "news_read_mean": mean("n_news_read"),
            "n_speak_mean": mean("n_speak"),
            "written_back_rate": (sum(r["written_back"] for r in sub)
                                  / max(1, sum(r["n_reflect"] for r in sub))),
        })
    return out


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_report(out_dir: Path, run_name: str, rows: list[dict],
                 levels: list[dict]) -> None:
    lines = [
        f"# 入力解像度の効果分離レポート — {run_name}",
        "",
        "計画: docs/plans/input-resolution-lod.md §3 I3(取り下げ条件 K1-K4 は §4 で事前登録済み)。",
        "解釈の順序: まず**飢餓チャネル**(fallback率・空応答・エラー)に水準差が無いことを確認し、",
        "その上で**低注意らしさチャネル**(distinct-n・訪問先の狭さ)の単調性を見る。",
        "**K1**: 劣化が飢餓チャネルに現れた場合「入力解像度=個体差の再現」の主張は取り下げ。",
        "注意: distinct-n は標本量に敏感(発話数が水準間で違うとプール値が歪む)。個体平均を併記。",
        "mock ランはプロンプト内容への機械反応のみ=効果の解釈は実LLMランに限る。",
        "",
        "## 水準別集計",
        "",
    ]
    keys = [k for k in levels[0].keys()] if levels else []
    for tbl_keys in [keys]:
        lines.append("| " + " | ".join(tbl_keys) + " |")
        lines.append("|" + "---|" * len(tbl_keys))
        for row in levels:
            lines.append("| " + " | ".join(fmt(row.get(k, "-"))
                                           for k in tbl_keys) + " |")
    empty_levels = [r["level"] for r in levels if r.get("n_agents", 0) == 0]
    if empty_levels:
        lines += ["", f"※ データ不足: 水準 {', '.join(empty_levels)} は割当0体。"]
    n_speak_total = sum(r["n_speak"] for r in rows)
    if n_speak_total == 0:
        lines += ["", "※ データ不足: 発話イベント0件のため distinct-n は無意味。"]
    lines += [
        "",
        "## 判定材料の読み方(事前登録の再掲)",
        "- 飢餓チャネルに水準差 → **K1 発動**(解像度でなく技術的飢餓。設計やり直し)。",
        "- narrow で distinct-n・訪問先エントロピーが低い → 低注意者らしさとして仮説と整合。",
        "- written_back_rate は不変域(beliefs)の健全性チェック(水準差が出てはいけない)。",
        "",
        f"個体別の詳細: resolution_agents.parquet(n={len(rows)})。",
    ]
    (out_dir / "resolution_report.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="入力解像度LODの効果分離分析(I3)")
    ap.add_argument("run_dir", help="例: runs/ires_llm_1d")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 runs/<name>/panel)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    per, level_of = collect(run_dir)
    rows = agent_rows(per, level_of)
    levels = level_table(rows, per, level_of)

    if rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_dir / "resolution_agents.parquet")
    write_report(out_dir, run_dir.name, rows, levels)

    print(f"水準別集計({run_dir.name}):")
    for row in levels:
        print("  " + json.dumps(row, ensure_ascii=False,
                                default=str)[:200])
    print(f"出力: {out_dir / 'resolution_report.md'} / resolution_agents.parquet")


if __name__ == "__main__":
    main()
