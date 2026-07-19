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

    # 本選フリート(7GPU vLLM)+ 予算ゲート + LOD + 非LLM を織り込んだ試算:
    python scripts/estimate_runtime.py --agents 10000 --days 1 \
        --calib runs/demo_event_200a3d --fleet vllm-7gpu-a100 \
        [--speculative] [--prefix-cache] [--lod-fg-ratio 0.1] [--nonllm-profile full]

フリート(--fleet)は「フリートの実効並列 req/s」から sec/呼を導く(呼数/日 ÷ req/s =
秒/シミュ日)。予算ゲート(--cap-per-step)は deliberate を step 上限で飽和させ、素の α を
大 N へ外挿したときの桁違いの過大評価を防ぐ。LOD(--lod-fg-ratio)は前景=フルLLM・背景=
軽量として実効 LLM 呼数を割り引く(非LLM 物理 step は割引かない)。数値の根拠と不確実性は
docs/research/scale-feasibility.md。

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

# ── 予算ゲート(deliberate 飽和)と呼種内訳 ────────────────────────────────
# conf/config.yaml lod.max_llm_per_step。deliberate 系(reply/social/…)は step
# あたりこの上限で頭打ち。plan/reflect は budget を通らず N にほぼ線形に伸びる。
DEFAULT_CAP_PER_STEP = 300
# runs/demo_event_200a3d の purpose 内訳: 予算ゲート対象(deliberate 系合計=
# llm_deliberate=12229)/ 非ゲート(plan 419 + reflect 513 = 932)/ 総 13161。
DELIB_FRACTION = 12229 / 13161        # ≈ 0.929(実測)= 予算ゲートで飽和しうる割合
UNCAPPED_ALPHA = 1.0                  # plan+reflect は体あたり概ね一定=線形近似

# ── 非LLM(Python 側 step 処理)コスト ────────────────────────────────────
# LLM 呼と別に、全 agent の物理 step(知覚/欲求/移動/イベント記録)が毎 step 走る。
# LOD(前景比率)では減らない(背景個体も物理 step は回る)。2 つの実測アンカー:
#   LEAN: mock 300体×100日=132分(compute-efficiency.md)→ 0.00183 秒/agent-step。
#         LLM をスタブ化した軽量ラン=Python 下限。
#   FULL: runs/demo_event_200a3d(200体・全チャネル)の実測 wall からの分離推定。
#         ckpt144→ckpt432=515.7分(2シミュ日)/ LLM逐次=(4493+4375)×3.1=458.2分 →
#         非LLM ≈ 28.7分/シミュ日 = 0.060 秒/agent-step(全チャネル+イベント記録IO)。
# 両者は 33 倍開く。真値は有効チャネル数と観測IOの最適化度に依存(scale-feasibility.md §4)。
NONLLM_SEC_PER_AGENT_STEP_LEAN = 0.00183
NONLLM_SEC_PER_AGENT_STEP_FULL = 0.060
NONLLM_PROFILES = {
    "none": 0.0,
    "lean": NONLLM_SEC_PER_AGENT_STEP_LEAN,
    "full": NONLLM_SEC_PER_AGENT_STEP_FULL,
}

# ── LOD(agent-tier)既定 ─────────────────────────────────────────────────
DEFAULT_LOD_BG_FACTOR = 0.1           # 背景個体の呼数=前景の 1/10(agent-lod-deepdive.md)

# ── フリート(推論バックエンド)プリセット ────────────────────────────────
# req/s は「プロンプト ~1,300tok 入力 + ~320tok 生成」という本シミュ呼形状での
# 1GPU あたり実効リクエスト毎秒。qwen3:4b 級(dense, bf16)を仮定。
# ★重要(捏造回避): 4B×1300/320 の *measured* 公開値は存在しない。A100/H100 の
#   req/s は近傍実測(Gemma-3-4B ≈ 3,385 out tok/s @ A100-40GB・100/600・databasemart)
#   + FLOP/帯域モデルからの **推定**。本選前に必ず実測で置換すること:
#     vllm bench serve --random-input-len 1300 --random-output-len 320 で数点。
#   出典と不確実性の詳細は docs/research/scale-feasibility.md §2。
# spec_mult: 高並列 regime の現実値 1.0〜1.3(vLLM spec-decode blog は QPS≈1 で最大2.8x
#   だが本シミュの飽和運転は逆 regime=減速もあり得る)。apc_mult: ペルソナ+履歴の
#   共有接頭辞キャッシュ 1.8〜3.0(SqueezeBits・共有が無いと逆に −37%)。
FLEET_PRESETS = {
    "ollama-local": {
        "kind": "sequential", "n_gpus": 1, "sec_per_call": 3.1,
        "spec_mult": 1.0, "apc_mult": 1.0,
        "note": "ローカル ollama 逐次(実測 3.1 秒/呼・並列なし)",
    },
    "vllm-7gpu-a100": {
        "kind": "throughput", "n_gpus": 7, "req_per_s_per_gpu": 8.0,  # 推定 range 6–12
        "spec_mult": 1.15, "apc_mult": 2.2,
        "note": "A100 80GB×7・4B bf16・1300in/320out 推定 8 req/s/GPU(range 6–12)",
    },
    "vllm-7gpu-h100": {
        "kind": "throughput", "n_gpus": 7, "req_per_s_per_gpu": 14.0,  # 推定 range 10–20
        "spec_mult": 1.15, "apc_mult": 2.2,
        "note": "H100 80GB×7・A100比~1.7x(HBM3)推定 14 req/s/GPU(range 10–20)",
    },
    "vllm-7gpu-a5000": {
        "kind": "throughput", "n_gpus": 7, "req_per_s_per_gpu": 3.0,  # 推定 range 2–5
        "spec_mult": 1.15, "apc_mult": 2.2,
        "note": "A5000 24GB×7(本選想定機)・4bit・A100比~0.35x 推定 3 req/s/GPU(range 2–5)",
    },
}


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


def predict_calls_capped(c1: float, n_calib: float, alpha: float, g: float,
                         agents: float, days: int, *,
                         steps_per_day: int = DEFAULT_STEPS_PER_DAY,
                         cap_per_step: int = DEFAULT_CAP_PER_STEP,
                         delib_fraction: float = DELIB_FRACTION,
                         uncapped_alpha: float = UNCAPPED_ALPHA,
                         lod_mult: float = 1.0) -> list[float]:
    """予算ゲート(deliberate 飽和)を織り込んだ日別予測呼数。

    呼を 2 成分に分ける(内訳は runs/demo_event_200a3d 実測):
      ・deliberate 系(delib_fraction)= budget ゲート。N^α で伸びるが 1 step あたり
        cap_per_step で頭打ち → 大 N では飽和(N 非依存)。
      ・plan/reflect 系(1 − delib_fraction)= 非ゲート。N^uncapped_alpha(≈線形)。
    lod_mult は両成分の *需要* を割り引く(背景個体の呼数減)。cap は割引後の需要に掛かる。
    cap_per_step<=0 なら上限なし = 素の α 外挿(過大評価=上限値)。

    ★これが素の predict_calls との決定的差: α=1.209 を 1万体超へ素で外挿すると
      deliberate を桁で過大評価する(200体で ~28/step → cap 300 は ~1,500体で binding)。
    """
    scale_d = (float(agents) / float(n_calib)) ** float(alpha)
    scale_u = (float(agents) / float(n_calib)) ** float(uncapped_alpha)
    delib1 = float(c1) * float(delib_fraction) * scale_d * float(lod_mult)
    uncap1 = float(c1) * (1.0 - float(delib_fraction)) * scale_u * float(lod_mult)
    cap_day = (float(cap_per_step) * float(steps_per_day)
               if cap_per_step and cap_per_step > 0 else math.inf)
    out = []
    for d in range(1, days + 1):
        gm = (1.0 + float(g)) ** (d - 1)
        delib = min(delib1 * gm, cap_day)
        uncap = uncap1 * gm
        out.append(delib + uncap)
    return out


def lod_call_multiplier(fg_ratio: float, bg_factor: float = DEFAULT_LOD_BG_FACTOR) -> float:
    """前景比率と背景係数から実効呼数倍率を返す。

    前景 = フル LLM(1.0)、背景 = 呼数 bg_factor(既定 1/10)。
    mult = fg + (1−fg)×bg_factor。fg=1 で 1.0(=LOD なし)、fg=0 で bg_factor。
    """
    fg = min(max(float(fg_ratio), 0.0), 1.0)
    return fg + (1.0 - fg) * float(bg_factor)


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


# ── フリート(推論バックエンド)解決 ──────────────────────────────────────
@dataclass
class Fleet:
    name: str
    kind: str            # "sequential" | "throughput"
    n_gpus: int
    sec_per_call: float  # 実効 秒/呼(throughput は 1/実効 req_s)
    req_per_s: float     # 実効 集約 req/s(sequential は 1/sec_per_call)
    speculative: bool
    prefix_cache: bool
    note: str


def resolve_fleet(name: str, *, speculative: bool = False,
                  prefix_cache: bool = False) -> Fleet:
    """プリセット名から実効 sec/呼(=1/実効 req_s)を導く。

    throughput: 実効 req_s = req_per_s_per_gpu × n_gpus ×(spec)×(apc)。
    sequential: 実測 sec_per_call をそのまま(並列なし)。
    """
    if name not in FLEET_PRESETS:
        raise ValueError(f"未知のフリート: {name}(選択肢: {', '.join(FLEET_PRESETS)})")
    p = FLEET_PRESETS[name]
    if p["kind"] == "sequential":
        spc = float(p["sec_per_call"])
        return Fleet(name, "sequential", int(p["n_gpus"]), spc, 1.0 / spc,
                     False, False, p["note"])
    base = float(p["req_per_s_per_gpu"]) * int(p["n_gpus"])
    mult = 1.0
    if speculative:
        mult *= float(p["spec_mult"])
    if prefix_cache:
        mult *= float(p["apc_mult"])
    req_s = base * mult
    return Fleet(name, "throughput", int(p["n_gpus"]), 1.0 / req_s, req_s,
                 bool(speculative), bool(prefix_cache), p["note"])


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
                 start: datetime | None = None,
                 steps_per_day: int = DEFAULT_STEPS_PER_DAY,
                 cap_per_step: int | None = None,
                 lod_call_mult: float = 1.0,
                 delib_fraction: float = DELIB_FRACTION,
                 uncapped_alpha: float = UNCAPPED_ALPHA,
                 nonllm_sec_per_agent_step: float = 0.0,
                 fleet: Fleet | None = None) -> dict:
    """予測の全数値をまとめた dict を返す(表示にも試験にも使う)。

    cap_per_step が真: 予算ゲート付き(predict_calls_capped)。None/0: 素の α 外挿。
    lod_call_mult: LOD 実効呼数倍率(LLM 呼のみ割引・非LLM 物理 step は割引かない)。
    nonllm_sec_per_agent_step: 1 agent・1 step あたりの非LLM(Python)壁時間。
    既定(cap=None, lod=1.0, nonllm=0.0)は従来と同一挙動(後方互換)。
    """
    if cap_per_step and cap_per_step > 0:
        calls = predict_calls_capped(
            calib.c1, calib.n_calib, calib.alpha, calib.g, agents, days,
            steps_per_day=steps_per_day, cap_per_step=cap_per_step,
            delib_fraction=delib_fraction, uncapped_alpha=uncapped_alpha,
            lod_mult=lod_call_mult)
    else:
        base = predict_calls(calib.c1, calib.n_calib, calib.alpha, calib.g, agents, days)
        calls = [c * float(lod_call_mult) for c in base]
    per_call = sec_per_call * (1.0 + overhead)
    nonllm_day = float(steps_per_day) * float(agents) * float(nonllm_sec_per_agent_step)
    rows = []
    cum = 0.0
    llm_total = 0.0
    nonllm_total = 0.0
    for d, cd in enumerate(calls, start=1):
        llm_sec = cd * per_call
        day_sec = llm_sec + nonllm_day
        llm_total += llm_sec
        nonllm_total += nonllm_day
        cum += day_sec
        eta = (start + timedelta(seconds=cum)) if start is not None else None
        rows.append({"day": d, "calls": cd, "llm_sec": llm_sec, "nonllm_sec": nonllm_day,
                     "day_sec": day_sec, "cum_sec": cum, "eta": eta})
    total_calls = float(sum(calls))
    total_sec = cum
    # 1 シミュ日あたり代表値(day1)と「24h で回せるシミュ日数」。
    per_simday_sec = rows[0]["day_sec"] if rows else 0.0
    sim_days_per_24h = (86400.0 / per_simday_sec) if per_simday_sec > 0 else float("inf")
    return {
        "agents": agents, "days": days, "rows": rows,
        "total_calls": total_calls, "total_sec": total_sec,
        "llm_sec": llm_total, "nonllm_sec": nonllm_total,
        "per_simday_sec": per_simday_sec, "sim_days_per_24h": sim_days_per_24h,
        "sec_per_call": sec_per_call, "overhead": overhead,
        "cap_per_step": cap_per_step, "lod_call_mult": float(lod_call_mult),
        "nonllm_sec_per_agent_step": float(nonllm_sec_per_agent_step),
        "fleet": fleet, "calib": calib,
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
    fleet: Fleet | None = rep.get("fleet")
    if fleet is not None and fleet.kind == "throughput":
        levers = []
        if fleet.speculative:
            levers.append("spec")
        if fleet.prefix_cache:
            levers.append("apc")
        lv = ("+" + "+".join(levers)) if levers else ""
        out.append(f"   fleet = {fleet.name}{lv}  {fleet.n_gpus}GPU"
                   f"  実効 {fleet.req_per_s:.1f} req/s"
                   f"  (= {rep['sec_per_call']:.4f} 秒/呼)")
    else:
        out.append(f"   sec/call = {rep['sec_per_call']:.2f}  overhead = {rep['overhead']*100:.0f}%"
                   f"  (実効 {rep['sec_per_call']*(1+rep['overhead']):.3f} 秒/呼)")
    if rep.get("cap_per_step"):
        out.append(f"   予算ゲート = {rep['cap_per_step']}/step  "
                   f"(deliberate {DELIB_FRACTION*100:.0f}% は飽和・plan+reflect は α={UNCAPPED_ALPHA} 線形)")
    if rep.get("lod_call_mult", 1.0) != 1.0:
        out.append(f"   LOD 実効呼数倍率 = {rep['lod_call_mult']:.3f}  (前景=フルLLM・背景=1/10 等)")
    if rep.get("nonllm_sec_per_agent_step", 0.0) > 0:
        out.append(f"   非LLM = {rep['nonllm_sec_per_agent_step']:.5f} 秒/agent-step"
                   f"  (全 {rep['agents']} 体の物理 step・LOD で減らない)")
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
    if rep.get("nonllm_sec", 0.0) > 0:
        out.append(f"   内訳: LLM {format_hms(rep['llm_sec'])}"
                   f"  +  非LLM {format_hms(rep['nonllm_sec'])}"
                   f"  (非LLM {rep['nonllm_sec']/max(rep['total_sec'],1e-9)*100:.0f}%)")
    sd = rep.get("sim_days_per_24h", float("inf"))
    sd_str = "∞" if sd == float("inf") else f"{sd:.1f}"
    out.append(f"   1シミュ日 ≈ {format_hms(rep['per_simday_sec'])}"
               f"  → 24h で {sd_str} シミュ日/日")
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
    ap.add_argument("--fleet", type=str, default=None, choices=list(FLEET_PRESETS),
                    help="推論バックエンドのプリセット(sec/呼を実効 req/s から導出)")
    ap.add_argument("--speculative", action="store_true",
                    help="speculative decoding を適用(フリートの spec_mult)")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="prefix cache(APC)を適用(フリートの apc_mult・共有接頭辞前提)")
    ap.add_argument("--sec-per-call", type=float, default=None,
                    help=f"1 呼あたり秒を明示(既定: --fleet か {DEFAULT_SEC_PER_CALL}・qwen3:4b 実測)")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                    help=f"オーバーヘッド率(既定 {DEFAULT_OVERHEAD})")
    ap.add_argument("--alpha", type=float, default=None, help="体数スケール指数 α を明示指定")
    ap.add_argument("--g", type=float, default=None, help="日成長率 g を明示指定(0.05 等)")
    ap.add_argument("--cap-per-step", type=int, default=DEFAULT_CAP_PER_STEP,
                    help=f"deliberate の step 上限(既定 {DEFAULT_CAP_PER_STEP}=config.yaml。0 で上限なし=素の α 外挿)")
    ap.add_argument("--lod-fg-ratio", type=float, default=1.0,
                    help="前景エージェント比率(1.0=全員フルLLM。0.1 等で背景を割引)")
    ap.add_argument("--lod-bg-factor", type=float, default=DEFAULT_LOD_BG_FACTOR,
                    help=f"背景個体の呼数係数(既定 {DEFAULT_LOD_BG_FACTOR}=前景の1/10)")
    ap.add_argument("--nonllm-profile", type=str, default="lean", choices=list(NONLLM_PROFILES),
                    help="非LLM(Python step)コスト: none/lean(mock 0.00183)/full(実測 0.060 秒/agent-step)")
    ap.add_argument("--nonllm-sec-per-agent-step", type=float, default=None,
                    help="非LLM秒/agent-step を明示(--nonllm-profile より優先)")
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

    # フリート解決 → 実効 sec/呼(明示 --sec-per-call が最優先)。
    fleet = None
    if args.fleet:
        fleet = resolve_fleet(args.fleet, speculative=args.speculative,
                              prefix_cache=args.prefix_cache)
    if args.sec_per_call is not None:
        sec_per_call = args.sec_per_call
    elif fleet is not None:
        sec_per_call = fleet.sec_per_call
    else:
        sec_per_call = DEFAULT_SEC_PER_CALL

    # 非LLM 秒/agent-step(明示 > プロファイル)。
    if args.nonllm_sec_per_agent_step is not None:
        nonllm = args.nonllm_sec_per_agent_step
    else:
        nonllm = NONLLM_PROFILES[args.nonllm_profile]

    lod_mult = lod_call_multiplier(args.lod_fg_ratio, args.lod_bg_factor)
    cap = args.cap_per_step if args.cap_per_step and args.cap_per_step > 0 else None

    rep = build_report(args.agents, args.days, calib,
                       sec_per_call=sec_per_call, overhead=args.overhead,
                       start=start, steps_per_day=args.steps_per_day,
                       cap_per_step=cap, lod_call_mult=lod_mult,
                       nonllm_sec_per_agent_step=nonllm, fleet=fleet)
    print(render_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
