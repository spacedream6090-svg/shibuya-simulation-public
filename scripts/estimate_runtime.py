#!/usr/bin/env python
"""実行時間試算ツール — 体数×日数から LLM 実行の所要時間を予測する。

背景: 会話数(=LLM 呼び出し数)は日を追って増え、体数に対しては超線形に伸びる
(社会的発火)。較正ラン(既存 run dir)の l1b_llm.parquet と agents.json から
モデルの係数を測り、任意の体数・日数へ外挿する。

モデル:
    calls(day d, N体) = C1 × (N / N_calib)^α × (1 + g)^(d-1)
    所要秒          = Σ_d calls(d, N) × sec_per_call × (1 + overhead)

  C1      : 較正ランの day1 実呼数(cached 除外)
  N_calib : 較正ランの体数
  α       : 体数スケール指数(較正ラン≥2本なら log-log 回帰、1本なら
            15体アンカー実測との対で自動計算、無しなら既定 1.15)
  g       : 日成長率(日別呼数の幾何平均成長 day1→last)

使い方:
    python scripts/estimate_runtime.py --agents 300 --days 7 \
        [--calib runs/demo_event_200a3d [runs/...]] \
        [--sec-per-call 3.1] [--overhead 0.15] [--start "09:00"]

依存: numpy + pyarrow + 標準ライブラリのみ(pandas/duckdb 不使用)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]

# ── 実測ベースの既定定数(較正ラン無し時 / アンカー) ──────────────────────
# 1 日 = 144 ステップ(10 分刻み・24h)。全較正ランが n_steps=432(3日)。
DEFAULT_STEPS_PER_DAY = 144
DEFAULT_SEC_PER_CALL = 3.1   # qwen3:4b 実測の 1 呼あたり秒
DEFAULT_OVERHEAD = 0.15      # スケジューラ/IO 等のオーバーヘッド率
DEFAULT_ALPHA = 1.15         # 社会的発火の超線形(較正 1 本以下のときの既定)

# 15体アンカー: 実測「15体・2日で375呼」= 187.5 呼/日(α 自動計算の対点に使う)
ANCHOR_N = 15
ANCHOR_CALLS_PER_DAY = 187.5

# 較正ラン無し時の内蔵既定(200体・実測ベース)。個別ランのばらつきで
# demo_event_200a3d の実測は day1≈4293 / g≈0.01 だが、既定はまるめ値を採用。
BUILTIN_N = 200
BUILTIN_DAY1 = 4105
BUILTIN_G = 0.05

# 実測でカバーしている範囲(これを超える外挿には警告を出す)
MEASURED_AGENT_RANGE = (15, 200)
MEASURED_DAY_RANGE = (1, 3)


# ── 較正データ読み取り ────────────────────────────────────────────────────
def read_daily_calls(run_dir: Path, steps_per_day: int = DEFAULT_STEPS_PER_DAY) -> list[int]:
    """l1b_llm.parquet の step 列を日別の実 LLM 呼数系列に集計する。

    cached 列があれば cache hit(True)を除外して実呼数のみ数える。
    戻り値は [day1, day2, ...](間の欠日は 0 で埋める)。
    """
    path = run_dir / "l1b_llm.parquet"
    if not path.exists():
        raise FileNotFoundError(f"l1b_llm.parquet が見つからない: {path}")
    cols = ["step"]
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    if "cached" in schema_names:
        cols.append("cached")
    tbl = pq.read_table(path, columns=cols).to_pydict()
    steps = np.asarray(tbl["step"], dtype=np.int64)
    if "cached" in tbl:
        cached = np.asarray(tbl["cached"], dtype=bool)
        steps = steps[~cached]  # cache hit を除外した実呼数
    if steps.size == 0:
        return []
    # step は 1 始まり。day = (step-1) // steps_per_day + 1。
    day = (steps - 1) // steps_per_day + 1
    n_days = int(day.max())
    counts = np.bincount(day, minlength=n_days + 1)[1:]  # index0(day0)を捨てる
    return [int(c) for c in counts]


def read_agent_count(run_dir: Path) -> int:
    """agents.json から体数を読む(list なら len、dict なら n_agents/agents)。"""
    path = run_dir / "agents.json"
    if not path.exists():
        # summary.json 側にも n_agents はある(フォールバック)
        sp = run_dir / "summary.json"
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                return int(json.load(f).get("n_agents", 0))
        raise FileNotFoundError(f"agents.json が見つからない: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        if isinstance(data.get("agents"), list):
            return len(data["agents"])
        if "n_agents" in data:
            return int(data["n_agents"])
    raise ValueError(f"agents.json の形式が不明: {path}")


# ── 数値カーネル(純関数・テスト対象) ────────────────────────────────────
def geom_growth(daily: list[int] | list[float]) -> float | None:
    """日別呼数系列の幾何平均成長率 g(day1→last)。

    g = (daily[-1] / daily[0])^(1/(len-1)) - 1。
    2 日未満、または day1 が 0 のときは None(算定不能)。
    """
    if daily is None or len(daily) < 2:
        return None
    first, last = float(daily[0]), float(daily[-1])
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / (len(daily) - 1)) - 1.0


def fit_alpha(points: list[tuple[float, float]]) -> float:
    """(N, calls_day1) の点群から体数スケール指数 α を log-log 回帰で推定。

    ln(calls) = α·ln(N) + b の最小二乗傾き。2 点なら (log比/log比) に一致。
    N が全て同一(縮退)なら ValueError。
    """
    pts = [(float(n), float(c)) for n, c in points if n > 0 and c > 0]
    if len(pts) < 2:
        raise ValueError("α 推定には異なる体数の 2 点以上が必要")
    xs = np.log(np.array([n for n, _ in pts]))
    ys = np.log(np.array([c for _, c in pts]))
    if float(np.ptp(xs)) == 0.0:
        raise ValueError("体数が全て同一で α を推定できない")
    # 傾き = cov(x,y)/var(x)(polyfit 次数1と同値)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope


def predict_calls(c1: float, n_calib: float, alpha: float, g: float,
                  agents: float, days: int) -> list[float]:
    """日別の予測呼数系列を返す。calls(d) = C1×(N/N_calib)^α×(1+g)^(d-1)。"""
    scale = (float(agents) / float(n_calib)) ** float(alpha)
    return [float(c1) * scale * (1.0 + float(g)) ** (d - 1) for d in range(1, days + 1)]


def format_hms(seconds: float) -> str:
    """秒を「X時間Y分」(1 時間未満は「Y分」)へ。四捨五入分。"""
    total_min = int(round(seconds / 60.0))
    h, m = divmod(total_min, 60)
    if h > 0:
        return f"{h}時間{m:02d}分"
    return f"{m}分"


# ── 較正の組み立て ────────────────────────────────────────────────────────
@dataclass
class Calibration:
    n_calib: float          # 基準体数
    c1: float               # 基準ランの day1 実呼数
    g: float                # 日成長率
    alpha: float            # 体数スケール指数
    alpha_method: str       # α の算定方法(表示用)
    g_method: str           # g の算定方法(表示用)
    n_runs: int             # 較正ラン本数(0=内蔵既定)
    points: list = field(default_factory=list)  # [(N, day1), ...]


def build_calibration(calib_dirs: list[Path], *, steps_per_day: int = DEFAULT_STEPS_PER_DAY,
                      alpha_override: float | None = None,
                      g_override: float | None = None) -> Calibration:
    """較正ラン群(0本以上)から Calibration を組む。"""
    runs = []  # (N, daily_calls)
    for d in calib_dirs:
        daily = read_daily_calls(d, steps_per_day)
        n = read_agent_count(d)
        if not daily:
            continue
        runs.append((n, daily))

    if not runs:
        # 較正無し: 内蔵既定
        alpha = alpha_override if alpha_override is not None else DEFAULT_ALPHA
        g = g_override if g_override is not None else BUILTIN_G
        return Calibration(
            n_calib=BUILTIN_N, c1=BUILTIN_DAY1, g=g, alpha=alpha,
            alpha_method="指定(--alpha)" if alpha_override is not None else "内蔵既定(α=1.15)",
            g_method="指定(--g)" if g_override is not None else f"内蔵既定(g={BUILTIN_G})",
            n_runs=0, points=[(BUILTIN_N, BUILTIN_DAY1)],
        )

    points = [(float(n), float(daily[0])) for n, daily in runs]
    # 基準ラン = 体数最大のもの(最も外挿元に近い)
    ref_n, ref_daily = max(runs, key=lambda r: r[0])
    c1 = float(ref_daily[0])

    # g: 各ラン(>=2日)の幾何成長を平均
    if g_override is not None:
        g, g_method = g_override, "指定(--g)"
    else:
        gs = [gg for gg in (geom_growth(daily) for _, daily in runs) if gg is not None]
        if gs:
            g = float(np.mean(gs))
            g_method = f"較正{len(gs)}本の幾何成長平均"
        else:
            g, g_method = BUILTIN_G, f"内蔵既定(g={BUILTIN_G}・較正が単日のみ)"

    # α: --alpha 指定 > 較正≥2本の回帰 > 較正1本+15体アンカー > 既定
    if alpha_override is not None:
        alpha, alpha_method = alpha_override, "指定(--alpha)"
    else:
        distinct_n = {n for n, _ in points}
        if len(distinct_n) >= 2:
            alpha = fit_alpha(points)
            alpha_method = f"較正{len(points)}本の log-log 回帰"
        elif len(points) == 1 and points[0][0] != ANCHOR_N:
            alpha = fit_alpha([(ANCHOR_N, ANCHOR_CALLS_PER_DAY)] + points)
            alpha_method = "15体アンカー実測+較正1本の自動計算"
        else:
            alpha, alpha_method = DEFAULT_ALPHA, "内蔵既定(α=1.15)"

    return Calibration(n_calib=float(ref_n), c1=c1, g=g, alpha=alpha,
                       alpha_method=alpha_method, g_method=g_method,
                       n_runs=len(runs), points=points)


# ── 外挿警告 ──────────────────────────────────────────────────────────────
def extrapolation_warnings(agents: int, days: int, calib: Calibration) -> list[str]:
    """実測範囲(15〜200体・1〜3日)を超える外挿への警告リスト。"""
    warns = []
    lo_a, hi_a = MEASURED_AGENT_RANGE
    lo_d, hi_d = MEASURED_DAY_RANGE
    if agents > hi_a:
        warns.append(f"体数 {agents} は実測上限 {hi_a} 体を超過(超線形の外挿・過大評価に注意)")
    if agents < lo_a:
        warns.append(f"体数 {agents} は実測下限 {lo_a} 体を下回る(小規模外挿)")
    if days > hi_d:
        warns.append(f"日数 {days} は実測上限 {hi_d} 日を超過(日成長 g の外挿は不確実)")
    if days < lo_d:
        warns.append(f"日数 {days} は 1 日未満")
    if calib.n_runs == 0:
        warns.append("較正ラン未指定(内蔵既定値で試算・精度低)")
    return warns


# ── レポート生成 ──────────────────────────────────────────────────────────
def build_report(agents: int, days: int, calib: Calibration, *,
                 sec_per_call: float = DEFAULT_SEC_PER_CALL,
                 overhead: float = DEFAULT_OVERHEAD,
                 start: datetime | None = None) -> dict:
    """予測の全数値をまとめた dict を返す(表示にも試験にも使う)。"""
    calls = predict_calls(calib.c1, calib.n_calib, calib.alpha, calib.g, agents, days)
    per_call = sec_per_call * (1.0 + overhead)
    rows = []
    cum = 0.0
    for d, cd in enumerate(calls, start=1):
        day_sec = cd * per_call
        cum += day_sec
        eta = (start + timedelta(seconds=cum)) if start is not None else None
        rows.append({"day": d, "calls": cd, "day_sec": day_sec,
                     "cum_sec": cum, "eta": eta})
    total_calls = float(sum(calls))
    total_sec = cum
    return {
        "agents": agents, "days": days, "rows": rows,
        "total_calls": total_calls, "total_sec": total_sec,
        "sec_per_call": sec_per_call, "overhead": overhead,
        "calib": calib,
        "warnings": extrapolation_warnings(agents, days, calib),
    }


def render_report(rep: dict) -> str:
    """レポート dict を人間可読のテキスト表へ整形。"""
    calib: Calibration = rep["calib"]
    out = []
    out.append("=" * 64)
    out.append(f" 実行時間試算  {rep['agents']}体 × {rep['days']}日")
    out.append("=" * 64)
    out.append(f" 較正: ラン{calib.n_runs}本  基準={int(calib.n_calib)}体"
               f"  C1(day1呼数)={calib.c1:.0f}")
    out.append(f"   α = {calib.alpha:.3f}  [{calib.alpha_method}]")
    out.append(f"   g = {calib.g*100:.2f}%/日  [{calib.g_method}]")
    out.append(f"   sec/call = {rep['sec_per_call']:.2f}  overhead = {rep['overhead']*100:.0f}%"
               f"  (実効 {rep['sec_per_call']*(1+rep['overhead']):.2f} 秒/呼)")
    out.append("-" * 64)
    has_eta = rep["rows"] and rep["rows"][0]["eta"] is not None
    header = f" {'日':>3} | {'予測呼数':>9} | {'日所要':>10} | {'累積所要':>11}"
    if has_eta:
        header += f" | {'完了予定':>12}"
    out.append(header)
    out.append("-" * 64)
    for r in rep["rows"]:
        line = (f" {r['day']:>3} | {r['calls']:>9.0f} | {format_hms(r['day_sec']):>10}"
                f" | {format_hms(r['cum_sec']):>11}")
        if has_eta:
            line += f" | {r['eta'].strftime('%m/%d %H:%M'):>12}"
        out.append(line)
    out.append("-" * 64)
    out.append(f" 総呼数 {rep['total_calls']:.0f}  総所要 {format_hms(rep['total_sec'])}"
               f"  ({rep['total_sec']/3600:.1f} 時間)")
    if rep["warnings"]:
        out.append("-" * 64)
        out.append(" [外挿警告]")
        for w in rep["warnings"]:
            out.append(f"   ! {w}")
    out.append("=" * 64)
    # 1 行サマリ
    conf = f"較正ラン{calib.n_runs}本"
    caution = "・外挿注意" if rep["warnings"] else ""
    out.append(f" {rep['agents']}体 × {rep['days']}日 ≈ {format_hms(rep['total_sec'])}"
               f"(信頼: {conf}{caution})")
    out.append("=" * 64)
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────
def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_ROOT / path)


def main(argv: list[str]) -> int:
    # Windows コンソール(cp932)対策: 進捗 print の非 cp932 文字で死なない。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="体数×日数から LLM 実行の所要時間を試算する(較正ラン任意)")
    ap.add_argument("--agents", type=int, required=True, help="試算したい体数")
    ap.add_argument("--days", type=int, required=True, help="試算したい日数")
    ap.add_argument("--calib", type=str, nargs="*", default=[],
                    help="較正ラン dir(複数可)。未指定なら内蔵既定値で試算")
    ap.add_argument("--sec-per-call", type=float, default=DEFAULT_SEC_PER_CALL,
                    help=f"1 呼あたり秒(既定 {DEFAULT_SEC_PER_CALL}・qwen3:4b 実測)")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                    help=f"オーバーヘッド率(既定 {DEFAULT_OVERHEAD})")
    ap.add_argument("--alpha", type=float, default=None, help="体数スケール指数 α を明示指定")
    ap.add_argument("--g", type=float, default=None, help="日成長率 g を明示指定(0.05 等)")
    ap.add_argument("--steps-per-day", type=int, default=DEFAULT_STEPS_PER_DAY,
                    help=f"1 日のステップ数(既定 {DEFAULT_STEPS_PER_DAY})")
    ap.add_argument("--start", type=str, default=None,
                    help='完走予定の開始時刻 "HH:MM"(既定=現在時刻を使わず ETA 非表示)')
    args = ap.parse_args(argv)

    calib_dirs = [_resolve(c) for c in args.calib]
    for d in calib_dirs:
        if not d.exists():
            print(f"[警告] 較正ランが見つからない: {d}", file=sys.stderr)
    calib_dirs = [d for d in calib_dirs if d.exists()]

    calib = build_calibration(calib_dirs, steps_per_day=args.steps_per_day,
                              alpha_override=args.alpha, g_override=args.g)

    start = None
    if args.start:
        try:
            hh, mm = args.start.split(":")
            now = datetime.now()
            start = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        except ValueError:
            print(f"[警告] --start の形式が不正(HH:MM): {args.start}", file=sys.stderr)

    rep = build_report(args.agents, args.days, calib,
                       sec_per_call=args.sec_per_call, overhead=args.overhead,
                       start=start)
    print(render_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
