"""決定モードの内訳 — 「その決定を下したのは LLM か・ルールか・再利用か」(V3)。

正典: docs/plans/external-audit-triage.md §3.2 V3 /
      実装 src/society/observer/decision_mode.py

何を出すのか
------------
本シムの「決定」は 3 レーンあり、本スクリプトはその 3 本を **1 枚の表**にまとめる。
出所はレーンごとに違い、**新しい記録を足したのは 1 本だけ**である:

  | レーン | 決定モードの出所 | 新規記録 |
  |---|---|---|
  | **朝の計画** | L1 `plan_created.payload.src` ∈ {llm, prev_day, skeleton}
                   + `plan_skipped{reason}` | なし(既存) |
  | **夜の内省** | `l1b_llm.purpose == "reflect"` + L1 `reflect_dropped` | なし(既存) |
  | **日中熟慮** | `summary.json` の `decision_mode` ブロック | **V3 で追加** |

日中熟慮だけが新規なのは、**分母がどこにも無かった**から。`_decide` は毎 step・在場
覚醒の全個体へ必ず 1 行動を返すのに、ルール層(routine.decide)が決めた分は L1 にも
l1b にも 1 バイトも残らない。したがって「LLM 被覆率 = LLM が決めた決定 / 全決定」は
原理的に計算できなかった(監査の 0.173 回/人日は**呼数 ÷ 人日**であって割合ではない)。

読み取り専用
------------
runs/<name>/ を読むだけ。シム本体を 1 つも変えず、乱数を引かず、LLM も呼ばない。
メモリは O(日数 × 語彙) で L1 の行数にも part 数にも比例しない。

使い方
------
    python scripts/decision_modes.py runs/<name>
    python scripts/decision_modes.py runs/<name> --json runs/<name>/decision_modes.json
    python scripts/decision_modes.py runs/<name> --out runs/<name>/decision_modes.md

終了コード: `decision_mode` の不変式 `points == llm + reuse + rule` が破れていたら 1
(= 記録側のバグ。黙って合わせずここで落とす)。summary に `decision_mode` が無い
(= observer.decision_mode OFF で回したラン)ときは 0 のまま、計画・内省の 2 レーンだけを出す。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:      # `python scripts/...` 以外の経路から import
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import l1_stream as ls                                              # noqa: E402
import run_dt                                                       # noqa: E402

MIN_PER_DAY = 1440

#: 計画レーンで読む L1 の kind(payload の src / reason だけを見る)。
PLAN_KINDS: tuple[str, ...] = ("plan_created", "plan_skipped")
#: 内省レーンで読む L1 の kind(撃った分は l1b から数える)。
REFLECT_KINDS: tuple[str, ...] = ("reflect_dropped",)
#: 方針キャッシュ(既定 OFF)。summary が無いランでも再利用だけは L1 から拾える。
REUSE_KINDS: tuple[str, ...] = ("policy_reuse",)


# --------------------------------------------------------------------------- #
# 0. run-dir の読み出し
# --------------------------------------------------------------------------- #
def _read_summary(run_dir) -> dict:
    p = Path(run_dir) / "summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _start_min(run_dir) -> int:
    """`run.start_tod`(既定 07:00)を分 of day で返す。l1b は sim_min を持たないので要る。"""
    p = Path(run_dir) / "config.yaml"
    if not p.is_file():
        return 420
    try:
        import yaml                                            # noqa: PLC0415
        v = ((yaml.safe_load(p.read_text(encoding="utf-8")) or {})
             .get("run", {}).get("start_tod"))
    except Exception:                                          # noqa: BLE001
        return 420
    if v is None:
        return 420
    s = str(v).strip()
    if ":" in s:
        try:
            hh, mm = s.split(":", 1)
            return (int(hh) * 60 + int(mm)) % MIN_PER_DAY
        except ValueError:
            return 420
    try:
        return int(float(s)) % MIN_PER_DAY
    except ValueError:
        return 420


# --------------------------------------------------------------------------- #
# 1. 計画レーン・内省レーン(既存 L1 / l1b だけで完全に可視)
# --------------------------------------------------------------------------- #
def scan_l1(run_dir) -> dict:
    """L1 を 1 パス走査して計画/内省/再利用の日別内訳を作る。

    `plan_created.payload.src` がそのまま「その日の計画を誰が書いたか」である
    (llm = 朝の LLM / prev_day = 前日の再利用 / skeleton = ルールの骨格)。
    """
    plan: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reflect: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reuse: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if not ls.l1_paths(run_dir):
        return {"plan": {}, "reflect": {}, "reuse": {}, "have_l1": False}

    spd = run_dt.steps_per_day(run_dir, notify=False)
    kinds = PLAN_KINDS + REFLECT_KINDS + REUSE_KINDS
    for batch in ls.iter_record_batches(
            run_dir, columns=["step", "sim_min", "kind", "payload"], kinds=kinds):
        d = batch.to_pydict()
        for i, kind in enumerate(d.get("kind") or []):
            day = run_dt.day_of({"sim_min": (d.get("sim_min") or [None])[i],
                                 "step": d["step"][i]}, spd)
            raw = (d.get("payload") or [None] * len(d["step"]))[i]
            try:
                pay = json.loads(raw) if raw else {}
            except ValueError:
                pay = {}
            if kind == "plan_created":
                plan[day][str(pay.get("src") or "?")] += 1
            elif kind == "plan_skipped":
                plan[day]["skipped:" + str(pay.get("reason") or "?")] += 1
            elif kind == "reflect_dropped":
                reflect[day]["dropped"] += 1
            elif kind == "policy_reuse":
                reuse[day][str(pay.get("kind") or "?")] += 1
    return {"plan": {k: dict(v) for k, v in plan.items()},
            "reflect": {k: dict(v) for k, v in reflect.items()},
            "reuse": {k: dict(v) for k, v in reuse.items()},
            "have_l1": True}


def scan_l1b(run_dir) -> dict:
    """l1b_llm を 1 パス走査して purpose 別 × 日別の呼数を作る。

    l1b は `sim_min` を持たないので day は `start_tod + step*Δt` から作る
    (L1 側の `sim_min // 1440` と同じ暦日境界に落ちる)。
    """
    path = Path(run_dir) / "l1b_llm.parquet"
    parts = ls.l1_paths(run_dir, "l1b_llm")
    src = [path] if path.is_file() else parts
    if not src:
        return {}
    mps = run_dt.min_per_step(run_dir, notify=False)
    start = _start_min(run_dir)
    out: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for one in src:
        for d in ls.iter_table_columns(one, ["step", "purpose"]):
            steps = d.get("step") or []
            purposes = d.get("purpose") or [None] * len(steps)
            for st, pur in zip(steps, purposes):
                day = (start + int(st) * mps) // MIN_PER_DAY
                out[day][str(pur or "?")] += 1
    return {k: dict(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 2. 3 レーンを 1 つのレポートへ
# --------------------------------------------------------------------------- #
def _lane_row(label: str, cells: dict) -> dict:
    """1 レーン 1 日ぶんの行(モード → 件数)+ シェア。"""
    total = sum(int(v) for v in cells.values())
    return {"lane": label, "total": total,
            "modes": {k: int(v) for k, v in sorted(cells.items())},
            "share": {k: (round(int(v) / total, 6) if total else 0.0)
                      for k, v in sorted(cells.items())}}


def analyze(run_dir) -> dict:
    summary = _read_summary(run_dir)
    dm = summary.get("decision_mode") or {}
    l1 = scan_l1(run_dir)
    l1b = scan_l1b(run_dir)

    days = set()
    days |= {int(d) for d in (dm.get("by_day") or {})}
    days |= set(l1["plan"]) | set(l1["reflect"]) | set(l1b)

    rows: list[dict] = []
    residual_total = 0
    for day in sorted(days):
        lanes: list[dict] = []
        # ---- 日中熟慮(V3。summary が唯一の出所)----
        cell = (dm.get("by_day") or {}).get(str(day))
        if cell:
            residual_total += abs(int(cell.get("residual", 0)))
            lanes.append({"lane": "deliberate",
                          "total": int(cell["points"]),
                          "modes": {"llm": int(cell["llm"]),
                                    "reuse": int(cell["reuse"]),
                                    "rule": int(cell["rule"])},
                          "share": dict(cell["share"]),
                          "residual": int(cell.get("residual", 0)),
                          "llm_calls": dict(cell.get("llm_calls") or {}),
                          "llm_unparsed": dict(cell.get("llm_unparsed") or {}),
                          "rule_by_reason": dict(cell.get("rule_by_reason") or {}),
                          "rule_by_src": dict(cell.get("rule_by_src") or {})})
        # ---- 朝の計画(既存 L1 だけで可視)----
        pl = dict(l1["plan"].get(day) or {})
        if pl:
            modes: dict[str, int] = {}
            for k, v in pl.items():
                # src=llm → LLM / prev_day → 再利用 / skeleton → ルール / skipped:* → 落ちた
                modes[{"llm": "llm", "prev_day": "reuse",
                       "skeleton": "rule"}.get(k, k)] = \
                    modes.get({"llm": "llm", "prev_day": "reuse",
                               "skeleton": "rule"}.get(k, k), 0) + int(v)
            lanes.append(_lane_row("plan", modes))
        # ---- 夜の内省(既存 l1b + L1 だけで可視。ルールの退路が無いレーン)----
        n_reflect = int((l1b.get(day) or {}).get("reflect", 0))
        n_dropped = int((l1["reflect"].get(day) or {}).get("dropped", 0))
        if n_reflect or n_dropped:
            lanes.append(_lane_row("reflect", {"llm": n_reflect,
                                               "dropped": n_dropped}))
        rows.append({"day": day, "lanes": lanes,
                     "llm_calls_by_purpose": dict(l1b.get(day) or {})})

    return {"schema": 1,
            "run_dir": str(Path(run_dir).resolve()),
            "have_decision_mode": bool(dm),
            "decision_mode_total": dm.get("total"),
            "residual_total": residual_total,
            "days": rows}


# --------------------------------------------------------------------------- #
# 3. Markdown
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    return f"{100.0 * float(x):.2f}%"


def render_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append("# 決定モードの内訳(V3)")
    L.append("")
    L.append(f"- run: `{rep['run_dir']}`")
    if not rep["have_decision_mode"]:
        L.append("- ⚠ `summary.decision_mode` が無い(observer.decision_mode OFF のラン)。"
                 "**日中熟慮レーンの分母は出せない** — 計画・内省の 2 レーンだけを出す。")
    if rep["residual_total"]:
        L.append(f"- ★不変式の破れ residual = {rep['residual_total']}"
                 "(points == llm + reuse + rule が成り立っていない = 記録側のバグ)")
    L.append("")

    tot = rep.get("decision_mode_total")
    if tot:
        L.append("## 日中熟慮レーン(全期間)")
        L.append("")
        L.append("| 決定点 | LLM | 再利用 | ルール | LLM シェア |")
        L.append("|---:|---:|---:|---:|---:|")
        L.append(f"| {tot['points']:,} | {tot['llm']:,} | {tot['reuse']:,} | "
                 f"{tot['rule']:,} | {_pct(tot['share']['llm'])} |")
        L.append("")
        L.append("ルール決定の理由 × 出所:")
        L.append("")
        L.append("| 理由 | 件数 | | 出所 | 件数 |")
        L.append("|---|---:|---|---|---:|")
        reasons = sorted((tot.get("rule_by_reason") or {}).items())
        srcs = sorted((tot.get("rule_by_src") or {}).items())
        for i in range(max(len(reasons), len(srcs))):
            a = f"`{reasons[i][0]}` | {reasons[i][1]:,}" if i < len(reasons) else " | "
            b = f"`{srcs[i][0]}` | {srcs[i][1]:,}" if i < len(srcs) else " | "
            L.append(f"| {a} | | {b} |")
        L.append("")
        calls = tot.get("llm_calls") or {}
        fails = tot.get("llm_unparsed") or {}
        if calls:
            L.append("用途(l1b の purpose)別の熟慮呼 — `fallback{parse_error}` には "
                     "trigger が載らないので、**不成立の内訳はここにしか無い**:")
            L.append("")
            L.append("| purpose | 撃った | 不成立 | 決めた |")
            L.append("|---|---:|---:|---:|")
            for k in sorted(calls):
                n, f = int(calls[k]), int(fails.get(k, 0))
                L.append(f"| `{k}` | {n:,} | {f:,} | {n - f:,} |")
            L.append("")

    L.append("## 日別 × レーン")
    L.append("")
    L.append("| 日 | レーン | 決定 | LLM | 再利用 | ルール | その他 | LLM シェア |")
    L.append("|---:|---|---:|---:|---:|---:|---|---:|")
    for row in rep["days"]:
        for lane in row["lanes"]:
            m = lane["modes"]
            other = {k: v for k, v in m.items()
                     if k not in ("llm", "reuse", "rule")}
            share = lane["share"].get("llm", 0.0)
            L.append(f"| {row['day']} | {lane['lane']} | {lane['total']:,} | "
                     f"{m.get('llm', 0):,} | {m.get('reuse', 0):,} | "
                     f"{m.get('rule', 0):,} | "
                     f"{', '.join(f'{k}={v}' for k, v in sorted(other.items())) or '-'} | "
                     f"{_pct(share)} |")
    L.append("")
    L.append("> レーンの読み方: **deliberate** = その step の行動を決めたのは誰か"
             "(分母 = 在場覚醒の個体数 × step)。**plan** = その日の計画を書いたのは誰か"
             "(src=llm / prev_day=前日の再利用 / skeleton=ルール骨格 / skipped:*=立たなかった)。"
             "**reflect** = 夜の内省(ルールの退路が無いので撃つか諦めるかの 2 択)。")
    L.append("")
    L.append("*本スクリプトは読み取り専用(シム本体ゼロタッチ・乱数0・LLM呼0)。*")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--out", help="Markdown の出力先(省略時は標準出力)")
    ap.add_argument("--json", dest="json_out", help="JSON の出力先")
    args = ap.parse_args(argv)

    rep = analyze(Path(args.run_dir))
    md = render_markdown(rep)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:                      # noqa: BLE001 (古い stdout)
            pass
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 1 if rep["residual_total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
