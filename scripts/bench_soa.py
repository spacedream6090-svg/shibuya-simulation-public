#!/usr/bin/env python
"""SoA アクタ基盤(`src/society/engine/soa.py`)のマイクロベンチ。

測るもの(既定 250,000 行 = 本選規模の 1 桁上):
  (a) 全列更新   — needs 減衰 `need *= 0.99`(float32・全行)
  (b) マスク更新 — 在場 10% だけを更新(`col[active_mask] *= k`)
  (c) 乱数一括   — `PhiloxDraws.uniform()` を 250k id ぶん
  (d) 書き込み流 — `PendingWrites` 10,000 件を flush(競合込み)
  (e) 昇降格     — `hydrate` + `dehydrate` の往復 1,000 回

出力は markdown 表(標準出力)と `runs/_bench/bench_soa.json`。
**runs/_bench/ の外には何も書かない**(runs/ は .gitignore 済み)。

計測の約束:
  * 直列実行のみ。各ケースはウォームアップ 1 回のあと `--repeat` 回まわし、
    **最小値**(= 外乱の少ない下限)と中央値の両方を出す。
  * `time.perf_counter_ns` で測る。テーブル確保・マスク生成などの準備は
    計測区間の外に出す(`reserve()` で容量成長も先に済ませる = ビュー再取得の
    ぶれを排除)。
  * ns/agent の分母は各ケースの「基準単位」列に明記した数(全行 / 更新行 /
    書き込み件数 / 往復回数)。

使い方:
  python scripts/bench_soa.py                     # 既定 250k × repeat 7
  python scripts/bench_soa.py --rows 50000 --repeat 3
  python scripts/bench_soa.py --no-json           # 標準出力のみ
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np                                        # noqa: E402

from society.engine.soa import (                          # noqa: E402
    ActorTable, PendingWrites, PhiloxDraws,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# hot 列スキーマ(エンジン監査の「40 個前後の hot スカラー」を模したもの)
# =============================================================================
# 座標・場所コード・スケジュールポインタ・欲求/ドライブ・所持金・フラグ。
# cold(MemoryStore / persona テキスト / route リスト)は**対象外**。
HOT_SCHEMA = [
    ("x", "float32", 0.0), ("y", "float32", 0.0),
    ("ind_x", "float32", 0.0), ("ind_y", "float32", 0.0),
    ("node", "int32", -1), ("dest", "int32", -1),
    ("building", "int32", -1), ("floor", "int16", 0),
    ("loc_code", "uint8", 0), ("activity_code", "uint8", 0),
    ("trip_mode", "uint8", 0), ("ind_zone", "uint8", 0),
    ("edge_offset", "float32", 0.0),
    ("plan_ptr", "int32", 0), ("plan_step", "int32", -1), ("plan_day", "int32", -1),
    ("stay_until", "int32", -1), ("sleep_until", "int32", -1),
    ("refractory_until", "int32", -1), ("conv_cooldown_until", "int32", -1),
    ("detained_until", "int32", -1), ("bankrupt_until", "int32", -1),
    ("need", "float32", 0.5), ("drive", "float32", 0.0),
    ("arousal", "float32", 0.0), ("fatigue", "float32", 0.0),
    ("opinion", "float32", 0.0), ("fire_weight", "float32", 0.0),
    ("status", "float32", 0.0), ("congestion", "float32", 0.0),
    ("money", "float64", 0.0), ("account", "float64", 0.0),
    ("period_income", "float64", 0.0), ("rent_due", "float64", 0.0),
    ("wage", "float32", 0.0),
    ("work_node", "int32", -1), ("org_id", "int32", -1),
    ("work_start_min", "int16", 0), ("work_end_min", "int16", 0),
    ("arrears_days", "int16", 0), ("work_days", "int16", 0),
    ("sleeping", "bool", False), ("visitor", "bool", False),
    ("part_time", "bool", False), ("evicted", "bool", False),
    ("homing", "bool", False), ("exit_intent", "bool", False),
]


# =============================================================================
# 計測ヘルパ
# =============================================================================
def _time(fn, repeat: int) -> tuple[float, float]:
    """(最小 ns, 中央値 ns)。ウォームアップ 1 回は捨てる。"""
    fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return float(min(samples)), float(statistics.median(samples))


def _case(name: str, basis: str, denom: int, fn, repeat: int) -> dict:
    lo, med = _time(fn, repeat)
    return {
        "case": name,
        "basis": basis,
        "denom": denom,
        "min_ms": lo / 1e6,
        "median_ms": med / 1e6,
        "ns_per_unit": lo / denom,
        "ns_per_unit_median": med / denom,
    }


# =============================================================================
# 本体
# =============================================================================
def run(rows: int, repeat: int, seed: int) -> dict:
    t = ActorTable(HOT_SCHEMA, capacity=rows)
    ids = t.alloc(rows)
    rng = np.random.default_rng(seed)
    t.col("need")[:] = rng.random(rows, dtype=np.float32)
    t.col("money")[:] = rng.random(rows) * 10000.0
    t.col("node")[:] = rng.integers(0, 20000, rows)

    draws = PhiloxDraws(seed)
    results = []

    # --- (a) 全列更新: needs 減衰(float32・全行・in-place) --------------------
    need = t.col("need")

    def case_a():
        np.multiply(need, np.float32(0.99), out=need)

    results.append(_case("(a) 全列更新 need*=0.99 (float32)", "全行", rows,
                         case_a, repeat))

    # --- (b) マスク更新: 在場 10% のみ ----------------------------------------
    active10 = np.zeros(rows, dtype=bool)
    active10[rng.choice(rows, size=rows // 10, replace=False)] = True
    n_active = int(active10.sum())
    drive = t.col("drive")

    def case_b():
        drive[active10] += np.float32(0.01)      # gather → 加算 → scatter(現実的な型)

    results.append(_case("(b) マスク更新 drive+=0.01 (在場10%)", "更新行", n_active,
                         case_b, repeat))
    results[-1]["ns_per_table_row"] = results[-1]["min_ms"] * 1e6 / rows

    # --- (c) 乱数一括: PhiloxDraws.uniform ------------------------------------
    def case_c():
        draws.uniform("decide", 42, ids)

    results.append(_case("(c) PhiloxDraws.uniform 一括", "全行", rows,
                         case_c, repeat))

    # --- (c') 参考: uniform4(1 回の全単射から 4 draw = 償却 1/4) ---------------
    def case_c4():
        draws.uniform4("decide", 42, ids)

    results.append(_case("(c') uniform4(4 draw 同時)", "draw", rows * 4,
                         case_c4, repeat))

    # --- (d) PendingWrites 10,000 件の flush ----------------------------------
    n_w = 10_000
    w_ids = ids[rng.integers(0, rows, n_w)]               # 重複あり = 競合込み
    w_cols = ["money", "node", "drive", "stay_until", "need"]
    w_col_pick = rng.integers(0, len(w_cols), n_w)
    w_vals = rng.random(n_w) * 100.0
    plan = [(int(w_ids[i]), w_cols[int(w_col_pick[i])], float(w_vals[i]))
            for i in range(n_w)]

    def case_d():
        pw = PendingWrites()
        for i, c, v in plan:
            pw.push(i, c, v)
        pw.flush(t)

    results.append(_case("(d) PendingWrites 1万件 push+flush", "書き込み", n_w,
                         case_d, repeat))

    # --- (d') flush 単体(push のループを除いた正味の適用コスト) ---------------
    pw_pre = PendingWrites()
    for i, c, v in plan:
        pw_pre.push(i, c, v)
    snap = (list(pw_pre._ids), list(pw_pre._cols), list(pw_pre._vals),
            list(pw_pre._seqs), pw_pre._next)

    def case_d2():
        pw_pre._ids[:] = snap[0]
        pw_pre._cols[:] = snap[1]
        pw_pre._vals[:] = snap[2]
        pw_pre._seqs[:] = snap[3]
        pw_pre._next = snap[4]
        pw_pre.flush(t)

    results.append(_case("(d') flush 単体(push 除く)", "書き込み", n_w,
                         case_d2, repeat))

    # --- (e) hydrate + dehydrate 往復 1,000 回 --------------------------------
    n_p = 1_000
    p_ids = [int(x) for x in ids[rng.choice(rows, size=n_p, replace=False)]]

    def case_e():
        for i in p_ids:
            t.dehydrate(i, t.hydrate(i))

    results.append(_case("(e) hydrate+dehydrate 往復", "往復", n_p,
                         case_e, repeat))

    # --- (e') 一括版(要求集合セマンティクスで 1000 件まとめて) ----------------
    p_arr = np.asarray(p_ids, dtype=np.int64)

    def case_e2():
        t.dehydrate_many(p_arr, t.hydrate_many(p_arr))

    results.append(_case("(e') hydrate_many+dehydrate_many", "往復", n_p,
                         case_e2, repeat))

    return {
        "meta": {
            "harness": "bench_soa",
            "rows": rows,
            "repeat": repeat,
            "seed": seed,
            "n_columns": len(HOT_SCHEMA),
            "bytes_per_row": int(sum(np.dtype(d).itemsize for _n, d, _v in HOT_SCHEMA)),
            "table_mb": round(rows * sum(np.dtype(d).itemsize
                                         for _n, d, _v in HOT_SCHEMA) / 1024 / 1024, 2),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "processor": platform.processor(),
            "note": ("直列・ウォームアップ1回捨て・min/median の 2 本立て。"
                     "ns/unit の分母は basis 列。"),
        },
        "rows_result": results,
        "projection": _projection(results),
    }


def _projection(results: list[dict]) -> dict:
    """「ホットループが (a)(b)(c) 相当の演算 5 本」だったときの c(s/agent-step)。"""
    def pick(tag: str) -> dict:
        for r in results:
            if r["case"].startswith(tag):
                return r
        raise KeyError(tag)

    array_ops = [pick("(a)"), pick("(b)"), pick("(c)")]
    # 全行基準に揃える(マスク更新は「テーブル 1 行あたり」に換算)
    per_agent_ns = [r.get("ns_per_table_row", r["ns_per_unit"]) for r in array_ops]
    mean_ns = sum(per_agent_ns) / len(per_agent_ns)
    c5 = 5.0 * mean_ns / 1e9
    # 付帯コスト: 1 step あたり「全体の 4% に書き込み」「0.4% を昇降格」と仮定
    w_ns = pick("(d')")["ns_per_unit"]
    p_ns = pick("(e)")["ns_per_unit"]
    return {
        "per_agent_ns_by_op": {r["case"]: n for r, n in zip(array_ops, per_agent_ns)},
        "mean_op_ns_per_agent": mean_ns,
        "c_5ops_s_per_agent_step": c5,
        "c_5ops_ms_per_step_at_1k": c5 * 1000 * 1e3,
        "c_5ops_ms_per_step_at_250k": c5 * 250_000 * 1e3,
        "plus_writes_4pct_s_per_agent_step": 0.04 * w_ns / 1e9,
        "plus_promote_0p4pct_s_per_agent_step": 0.004 * p_ns / 1e9,
        "total_s_per_agent_step": (c5 + 0.04 * w_ns / 1e9 + 0.004 * p_ns / 1e9),
    }


# =============================================================================
# 整形
# =============================================================================
def markdown(payload: dict) -> str:
    m = payload["meta"]
    head = ("| ケース | 基準単位 | 件数 | min (ms) | median (ms) | **ns/単位** |")
    sep = "|---|---|---:|---:|---:|---:|"
    body = []
    for r in payload["rows_result"]:
        body.append("| {c} | {b} | {d:,} | {lo:.3f} | {me:.3f} | **{ns:.2f}** |".format(
            c=r["case"], b=r["basis"], d=r["denom"],
            lo=r["min_ms"], me=r["median_ms"], ns=r["ns_per_unit"]))
    p = payload["projection"]
    tail = [
        "",
        f"表: rows={m['rows']:,} / 列 {m['n_columns']} 本 / "
        f"{m['bytes_per_row']} B/行 = {m['table_mb']} MB / "
        f"numpy {m['numpy']} / Python {m['python']} / {m['platform']}",
        "",
        "### 投影: ホットループが「(a)(b)(c) 相当の演算 5 本」だったときの c",
        "",
        "| 項目 | 値 |",
        "|---|---:|",
        f"| 1 演算あたり(全行基準) | {p['mean_op_ns_per_agent']:.2f} ns/agent |",
        f"| **c = 5 演算** | **{p['c_5ops_s_per_agent_step']:.3e} s/agent-step** |",
        f"| 同 N=1,000 での 1 step | {p['c_5ops_ms_per_step_at_1k']:.3f} ms |",
        f"| 同 N=250,000 での 1 step | {p['c_5ops_ms_per_step_at_250k']:.3f} ms |",
        f"| + 書き込み(全体の4%) | {p['plus_writes_4pct_s_per_agent_step']:.3e} s/agent-step |",
        f"| + 昇降格(全体の0.4%) | {p['plus_promote_0p4pct_s_per_agent_step']:.3e} s/agent-step |",
        f"| **合計** | **{p['total_s_per_agent_step']:.3e} s/agent-step** |",
    ]
    return "\n".join([head, sep, *body, *tail])


def main() -> None:
    ap = argparse.ArgumentParser(description="SoA アクタ基盤のマイクロベンチ")
    ap.add_argument("--rows", type=int, default=250_000)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/_bench")
    ap.add_argument("--no-json", action="store_true", help="標準出力のみ(JSON を書かない)")
    args = ap.parse_args()

    payload = run(args.rows, args.repeat, args.seed)
    print("\n### SoA アクタ基盤 マイクロベンチ\n")
    print(markdown(payload))

    if not args.no_json:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        path = out / "bench_soa.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"\n[bench_soa] wrote {path}")


if __name__ == "__main__":
    main()
