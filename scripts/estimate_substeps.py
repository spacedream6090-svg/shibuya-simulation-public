#!/usr/bin/env python
"""物理サブステップの事前見積 — 竹-4 残⑦「10日ラン ON 時の総サブステップ事前見積」。

何を数えるか
------------
`physics.zones_enabled: true` のとき、世界 tick(既定 10 分)の中を
`src/society/physics.py::_run_zone` が **dt_sub 刻みで通しで積分する**。その 1 刻みが
「サブステップ」。本ツールはランを回す前に **総サブステップ数**(と、その内訳である
agent·サブステップ = 実際の演算量)を見積もる。

1 step あたりのサブステップ数がどう決まるか(実装からの写し)
-------------------------------------------------------------
  n_sub_max(zone) = min(max_sub_steps, max(1, round(step_seconds / dt_sub)))
        既定は min(12000, 600/0.05) = 12000。
  実際に回る数 sub_done(zone, step) は **ゾーンが空になったら打ち切られる**:
    - step の頭で members も waiting も居なければ `_run_zone` は即 return(= 0)。
    - 誰か居る間は毎刻み回る。**入場待ち(waiting)だけでも回る**(信号赤で縁石に
      溜まっている間も時間は進む)。
    - 全員が抜けた瞬間に break。
  → sub_done(zone, step) = min( n_sub_max , ceil( T_clear / dt_sub ) )
     T_clear = step の頭から「そのゾーンが空になるまで」の秒数(空にならなければ step 全長)。
  流入候補は **step の頭で一括に集める**(`_run_zone` の (2))ので、step 途中で新規に
  ゾーンが「再点火」することはない。これが上の式が成り立つ理由。

  総サブステップ = Σ_zone Σ_step sub_done(zone, step)
  総 agent·サブステップ = Σ_zone Σ_step Σ_刻み (その刻みの在圏人数)
        ← 実際の計算コストはこちら。1 刻みの費用は在圏人数 n に対して概ね O(n²)
          (physics.`_accumulate` が毎刻み n×n の距離行列を作る。sfm_core の近傍 cap は
           対人力の**合算相手**を絞るだけで、距離行列自体は全ペア)。

2 つのモード
------------
  ① 実測外挿モード(--calib runs/<dir>): 既存ランの L1 から `zone_gate` イベントを拾い、
     ゾーンごとの在圏区間を復元して **実際に回ったサブステップ数**を数え、目標体数・日数へ
     外挿する。物理 ON のランが要る(zone_gate が 0 件なら「測れない」と明記して降りる)。
  ② 理論見積モード(--conf + --agents): conf のゾーン宣言と体数から、待ち行列
     M/G/∞ の空き確率で duty(ゾーンが塞がっている時間割合)を出して積む。

  どちらのモードも **仮定を数式ごとレポートに印字する**(隠さない)。

使い方
------
    # 理論見積(本選 10 日・1万体)。ゾーン宣言を持つ conf を指す。
    python scripts/estimate_substeps.py --days 10 --agents 10000 --conf conf/observe.yaml

    # 実測外挿(物理 ON の較正ランがあるとき)
    python scripts/estimate_substeps.py --days 10 --agents 10000 \
        --calib runs/zone_smoke --conf runs/zone_smoke/config.yaml

    # 診断ラン用の実測プロファイル(在街者の日内変動)を既存ランから借りる
    python scripts/estimate_substeps.py --days 10 --agents 10000 --conf conf/observe.yaml \
        --profile-run runs/demo_event_200a3d

    # 壁時計もこのマシンの実費用で(数秒のマイクロベンチ。シムは動かさない)
    python scripts/estimate_substeps.py --days 10 --agents 10000 --conf conf/observe.yaml \
        --measure-cost --json out/substeps.json

依存: numpy + pyarrow + omegaconf/yaml(既存 scripts と同じ)。src/ は **読むだけ**。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))            # society.world.zones / observer.measure

# ── 実測アンカー ───────────────────────────────────────────────────────────
# reference/physics_bench/out/results.json の throughput 節(200 体・dt=0.05 相当の
# agent·step/s)。**このリポジトリの自前 SFM/ORCA コアの実測**であって外部ベンチではない。
#   counterflow_w3.sfm  38111.7 / crossing.sfm  37585.5  → 保守側の小さい方を採る
#   counterflow_w3.orca 57344.4 / crossing.orca 66998.9  → 同上
BENCH_N_REF = 200                       # 実測時の在圏人数(この人数での 1 体·1刻みの費用)
BENCH_AGENT_SUBSTEPS_PER_S = {"sfm": 37585.5, "orca": 57344.4}
BENCH_SOURCE = "reference/physics_bench/out/results.json (throughput 節・200体)"

# 歩行速度(dwell = 通過距離 / 速度)。Weidmann 1993 の自由歩行 1.34 m/s。
DEFAULT_WALK_SPEED = 1.34
# 1 体が 1 日にそのゾーンを通る回数(往復 = 2)。★根拠のない既定。必ず自分の数字へ。
DEFAULT_TRAVERSALS_PER_DAY = 2.0
# 全体のうちそのゾーンを経路に含む割合。★根拠のない既定。必ず自分の数字へ。
DEFAULT_ZONE_SHARE = 0.5
DEFAULT_DT_MIN = 10


# =========================================================================== #
# 数値カーネル(純関数・テスト対象)
# =========================================================================== #
def n_sub_per_step(dt_sub: float, max_sub_steps: int, step_seconds: float) -> int:
    """1 世界 step で回りうるサブステップ数の上限(physics.`_run_zone` と同一式)。"""
    return min(int(max_sub_steps), max(1, int(round(float(step_seconds) / float(dt_sub)))))


def substeps_from_busy(busy_s: float, dt_sub: float, n_sub_max: int) -> int:
    """「step の頭からゾーンが空になるまで busy_s 秒」→ 実際に回るサブステップ数。

    physics 側は 1 刻みずつ回して「空になった」判定を刻みの**後**に行うので、
    busy_s に食い込んだ刻みは 1 つ数える = ceil。上限 n_sub_max で必ず止まる。
    """
    if busy_s <= 0.0:
        return 0
    return min(int(n_sub_max), int(math.ceil(float(busy_s) / float(dt_sub) - 1e-12)))


def mg_inf_duty(arrival_per_s: float, dwell_s: float) -> float:
    """M/G/∞ の「1 人以上居る確率」= 1 − exp(−λτ)。

    仮定: 到着はポアソン(λ = arrival_per_s)・滞在時間の分布は任意(平均 τ = dwell_s)・
    サーバ数無限(= ゾーンは詰まらない)。詰まると τ が伸びるので、これは **下限側**の duty。
    """
    lam, tau = float(arrival_per_s), float(dwell_s)
    if lam <= 0.0 or tau <= 0.0:
        return 0.0
    return 1.0 - math.exp(-min(lam * tau, 700.0))


def duty_scale(duty: float, ratio: float) -> float:
    """duty(塞がり率)を体数比 ratio へ外挿する。

    M/G/∞ では空き確率 = exp(−λτ) で、λ ∝ 体数。よって
        1 − duty' = exp(−λ·ratio·τ) = (1 − duty)^ratio
    体数を r 倍すると「空いている確率が r 乗で潰れる」= duty は 1 へ張り付く。
    ★これが本ツールのいちばん重要な帰結: 総サブステップは体数に比例せず **飽和する**。
      伸びるのは agent·サブステップ(演算量)の方。
    """
    idle = max(0.0, min(1.0, 1.0 - float(duty)))
    r = max(0.0, float(ratio))
    if idle <= 0.0:
        return 1.0
    return 1.0 - idle ** r


def cost_coefficients(engine: str, cost_model: str = "quadratic",
                      measured: dict | None = None,
                      anchor: dict | None = None,
                      n_ref: int = BENCH_N_REF) -> dict:
    """1 サブステップの費用 sec(n) = a + b·n² の係数を決める(n = 在圏人数)。

    - measured(--measure-cost の実測)があればそれを使う = **このマシンの実費用**。
    - 無ければベンチ点(n_ref 体での agent·step/s)から b だけを起こす:
        n_ref 体 1 刻みの実測費用 = n_ref / A  →  b = 1/(A·n_ref)、a = 0。
      a=0 は「小さい n での numpy 呼び出し固定費を無視する」という **過小評価**なので、
      実測(--measure-cost)を強く勧める。
    - cost_model="linear" は費用 ∝ n(= b·n の形)。近傍 cap だけが効く楽観側。
    """
    if measured and engine in measured:
        m = dict(measured[engine])
        m.setdefault("model", "measured")
        return m
    a = (anchor or BENCH_AGENT_SUBSTEPS_PER_S)
    per_s = float(a.get(engine, a.get("sfm", 1.0)))
    if per_s <= 0.0:
        return {"a": 0.0, "b": 0.0, "c1": 0.0, "model": "none"}
    if cost_model == "linear":
        return {"a": 0.0, "b": 0.0, "c1": 1.0 / per_s, "model": "bench-linear"}
    return {"a": 0.0, "b": 1.0 / (per_s * float(n_ref)), "c1": 0.0,
            "model": "bench-quadratic"}


def wall_seconds(substeps: float, mean_occupancy: float, coef: dict) -> float:
    """サブステップ数 × 平均在圏人数 → 壁時計秒。sec = subs × (a + c1·n + b·n²)。

    ★ Jensen の不等式ぶんの偏り: n の分散を無視して平均在圏 n̄ で代表させているので、
      2 次項は真値より小さく出る(混雑の山が均されるため)。
    """
    n = max(0.0, float(mean_occupancy))
    per = (float(coef.get("a", 0.0)) + float(coef.get("c1", 0.0)) * n
           + float(coef.get("b", 0.0)) * n * n)
    return float(substeps) * per


def measure_engine_cost(engine: str = "sfm", ns=(1, 4, 16, 48, 128, 200),
                        dt: float = 0.05, budget_s: float = 0.05) -> dict:
    """このマシンで **1 サブステップの実費用** を測り、sec(n) = a + b·n² を当てる。

    測るのは `_run_zone` が 1 刻みで確実に踏む部分:
      engine.step(dt)              … SFM/ORCA の積分本体
      n×n 距離行列 + min_gap        … physics.`_accumulate` の重なり計測
    含めていないもの(= この見積は**下限側**): `_accumulate` / `_advance_and_collect`
    の O(n) Python ループ、L1 への zone_gate 書き出し、レコード保存。
    乱数もゾーン外の状態も触らない純粋なマイクロベンチ(シムを 1 バイトも動かさない)。
    """
    import time

    import numpy as np

    from society.world import orca_core as _orca
    from society.world import sfm_core as _sfm

    rows = []
    for n in ns:
        n = int(n)
        if n < 1:
            continue
        side = max(2.0, math.sqrt(n / 1.0))          # 密度 ~1 人/m² の正方形へ並べる
        g = int(math.ceil(math.sqrt(n)))
        pts = [((i % g) * side / g, (i // g) * side / g) for i in range(n)]
        pos = np.array(pts, dtype=np.float64)
        vel = np.zeros_like(pos)
        goal = pos + np.array([100.0, 0.0])
        v0 = np.full(n, 1.34)
        radius = np.full(n, 0.3)
        if engine == "orca":
            eng = _orca.OrcaCrowd(pos, vel, goal, v0, radius, neighbor_cap=12)
        else:
            eng = _sfm.Crowd(pos, vel, goal, v0, radius=radius, neighbor_cap=12)
        # 予備実行(JIT ではないが allocator を温める)
        eng.step(dt)
        reps = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(reps):
                eng.step(dt)
                p = eng.pos
                d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                _orca.min_gap(p, radius)
            el = time.perf_counter() - t0
            if el >= budget_s or reps >= 4096:
                break
            reps = max(reps * 2, int(reps * budget_s / max(el, 1e-9)))
        rows.append((float(n), el / reps))
    if len(rows) < 2:
        return {"a": rows[0][1] if rows else 0.0, "b": 0.0, "c1": 0.0,
                "model": "measured", "points": rows}
    # a = 固定費(= 最小 n での実測。numpy 呼び出しのオーバーヘッド)。
    #   ここを 2 変数最小二乗に任せると大 n の点に引かれて a<0 → 0 へ潰れ、
    #   小 n(= ゾーンに数人しか居ない大半の step)を系統的に過小評価する。
    a = min(sec for _n, sec in rows)
    # b = 原点通過の最小二乗(残差 / n²)。b = Σ (sec−a)·n² / Σ n⁴。
    num = sum(max(sec - a, 0.0) * (n ** 2) for n, sec in rows)
    den = sum((n ** 4) for n, _sec in rows)
    b = (num / den) if den > 0 else 0.0
    return {"a": max(a, 0.0), "b": max(b, 0.0), "c1": 0.0, "model": "measured",
            "points": [(int(n), sec) for n, sec in rows]}


def diurnal_weights(n_steps_per_day: int, active_hours: float, night_factor: float,
                    step_minutes: int, start_min: int = 420) -> list[float]:
    """1 日ぶんの相対到着重み w(step)(平均が 1 になるよう正規化)。

    2 帯モデル: 起床帯(active_hours 時間)は 1、それ以外は night_factor。
    ★これは形だけのプレースホルダ。実測プロファイル(--profile-run)を渡せば置き換わる。
    """
    n = max(1, int(n_steps_per_day))
    raw = []
    for s in range(n):
        tod = (start_min + s * step_minutes) % (24 * 60)
        # 起床帯は 06:00 から active_hours 時間
        hi = 6 * 60 + active_hours * 60
        raw.append(1.0 if 6 * 60 <= tod < hi else float(night_factor))
    m = sum(raw) / n
    return [r / m for r in raw] if m > 0 else [1.0] * n


# =========================================================================== #
# ゾーン宣言の読み取り(conf。src/ は読むだけ)
# =========================================================================== #
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def _shoelace_area(poly) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def load_zones(conf_path: Path) -> tuple[list[dict], list[str]]:
    """conf の `physics.zones` を見積用の軽い dict 列にする。

    `zones_enabled: false`(既定)でも **宣言があれば見積もる**(「ON にしたらどうなるか」が
    本ツールの用途なので)。その場合は注記を返す。
    src の `world.zones.build_cfg` を第一候補にして既定値の単一源を保つが、壁/歩行可能面の
    データファイルが無い環境では落ちるので、そのときは ZONE_DEFAULTS だけで組み直す。
    """
    notes: list[str] = []
    cfg = _load_yaml(conf_path)
    raw = dict(cfg.get("physics") or {})
    specs = list(raw.get("zones") or ())
    if not specs:
        return [], [f"{conf_path} に physics.zones の宣言が無い(= ゾーン 0 件)"]
    if not raw.get("zones_enabled", False):
        notes.append("conf の physics.zones_enabled は false"
                     "(宣言だけを読んで「ON にしたら」を見積もる)")
    forced = dict(raw)
    forced["zones_enabled"] = True
    try:
        from society.world import zones as _zones
        built = _zones.build_cfg(forced, _ROOT)["zones"]
        out = [{"id": z.id, "engine": z.engine, "dt_sub": z.dt_sub,
                "max_sub_steps": int(z.max_sub_steps),
                "area_m2": float(z.walkable_area_m2 or z.area_m2() or 0.0),
                "polygon_area_m2": float(z.area_m2() or 0.0),
                "has_signal": bool(z.signal),
                "source": "society.world.zones.build_cfg"} for z in built]
        return out, notes
    except Exception as exc:                       # データ不在・壁ファイル欠損など
        notes.append(f"build_cfg が使えないので既定値だけで組む({type(exc).__name__}: {exc})")
    try:
        from society.world.zones import ZONE_DEFAULTS as _ZD
        d = dict(_ZD)
    except Exception:
        d = {"engine": "sfm", "dt_sub": 0.05, "max_sub_steps": 12000}
    out = []
    for spec in specs:
        s = dict(spec)
        poly = [(float(p[0]), float(p[1])) for p in (s.get("polygon") or ())]
        out.append({
            "id": str(s.get("id", "?")),
            "engine": str(s.get("engine", d.get("engine", "sfm"))),
            "dt_sub": float(s.get("dt_sub", d.get("dt_sub", 0.05))),
            "max_sub_steps": int(s.get("max_sub_steps", d.get("max_sub_steps", 12000))),
            "area_m2": _shoelace_area(poly) if len(poly) >= 3 else 0.0,
            "polygon_area_m2": _shoelace_area(poly) if len(poly) >= 3 else 0.0,
            "has_signal": bool(s.get("signal")),
            "source": "conf 直読み(既定値マージ)",
        })
    return out, notes


def dt_min_of(conf_path: Path | None, override: int | None) -> int:
    if override:
        return int(override)
    if conf_path and conf_path.exists():
        try:
            return int((_load_yaml(conf_path).get("run") or {}).get("dt_min", DEFAULT_DT_MIN))
        except Exception:
            pass
    return DEFAULT_DT_MIN


# =========================================================================== #
# 実測モード: L1 の zone_gate から在圏区間を復元する
# =========================================================================== #
def scan_zone_gates(run_dir: Path, step_seconds: float) -> dict:
    """L1 を逐次読みして `zone_gate` から (zone, agent) の在圏区間 [t0, t1] 秒を組む。

    - enter: payload の `wait_s`(入場までの待ち秒)と `waited_steps` から
      **待ち行列に並び始めた step** を割り出す。ループは waiting だけでも回るので、
      サブステップ会計の起点は「並び始めた step の頭」である。
    - exit : payload の `dwell_s`(積分した滞在秒)。t1 = 入場時刻 + dwell_s。
    - 出ないまま終わった個体はランの最終 step 末まで在圏とみなす。
    メモリは (zone, agent) の同時在圏数で有界(全イベントを溜めない)。
    """
    from society.observer import measure as _m

    open_rec: dict[tuple, dict] = {}
    intervals: dict[str, list] = {}
    n_enter = n_exit = 0
    max_step = -1
    for e in _m.stream_events(str(run_dir), columns=["step", "agent_id", "kind", "payload"]):
        s = int(e["step"])
        if s > max_step:
            max_step = s
        if e["kind"] != "zone_gate":
            continue
        p = e["payload"] or {}
        z = str(p.get("zone", "?"))
        key = (z, int(e["agent_id"]))
        if p.get("dir") == "enter":
            n_enter += 1
            wait_s = float(p.get("wait_s", 0.0) or 0.0)
            waited = int(p.get("waited_steps", 0) or 0)
            t_in = s * step_seconds + max(0.0, wait_s - waited * step_seconds)
            queue_start = (s - waited) * step_seconds        # ループが回り始めた時刻
            open_rec[key] = {"t_in": t_in, "q0": queue_start}
        else:
            n_exit += 1
            rec = open_rec.pop(key, None)
            dwell = float(p.get("dwell_s", 0.0) or 0.0)
            if rec is None:                                   # enter を見ていない(resume 等)
                t0 = s * step_seconds
                t1 = t0 + dwell
            else:
                t0 = rec["q0"]
                t1 = rec["t_in"] + dwell
            intervals.setdefault(z, []).append((t0, max(t1, t0)))
    end_t = (max_step + 1) * step_seconds
    for (z, _aid), rec in open_rec.items():                   # 出ないまま終端
        intervals.setdefault(z, []).append((rec["q0"], end_t))
    return {"intervals": intervals, "n_enter": n_enter, "n_exit": n_exit,
            "n_steps": max_step + 1, "n_unclosed": len(open_rec)}


def measure_substeps(intervals: list, n_steps: int, dt_sub: float, n_sub_max: int,
                     step_seconds: float) -> dict:
    """在圏区間の集合 → step ごとの実サブステップ数と agent·サブステップ数。

    sub_done(step) = ceil( (そのゾーンが空になるまでの秒数) / dt_sub ) 上限 n_sub_max。
    agent·サブステップ(step) = Σ_個体 (その step 内の在圏秒) / dt_sub。
    """
    per_step = [0] * max(0, int(n_steps))
    agent_sub = [0.0] * max(0, int(n_steps))
    for t0, t1 in intervals:
        s0 = max(0, int(t0 // step_seconds))
        s1 = min(int(n_steps) - 1, int((t1 - 1e-9) // step_seconds))
        for s in range(s0, s1 + 1):
            a, b = s * step_seconds, (s + 1) * step_seconds
            ov = min(t1, b) - max(t0, a)
            if ov <= 0:
                continue
            agent_sub[s] += ov / dt_sub
            # ループの終端は「step の頭から数えて最後に誰かが居た瞬間」まで
            busy = min(t1, b) - a
            per_step[s] = max(per_step[s], substeps_from_busy(busy, dt_sub, n_sub_max))
    total = sum(per_step)
    return {"per_step": per_step, "agent_substeps_per_step": agent_sub,
            "substeps_total": int(total),
            "agent_substeps_total": float(sum(agent_sub)),
            "busy_steps": sum(1 for v in per_step if v > 0),
            "saturated_steps": sum(1 for v in per_step if v >= n_sub_max)}


def street_profile(run_dir: Path, n_steps_per_day: int) -> list[float] | None:
    """既存ランの L1 から「日内の移動量プロファイル」を作る(平均 1 に正規化)。

    move_segment の step 別件数 = その step に街路を動いていた個体数。ゾーンへの到着率は
    これに比例するとみなす(= 形だけ借りて水準は体数で決める)。物理 OFF のランでも作れる。
    """
    from society.observer import measure as _m

    counts = [0] * int(n_steps_per_day)
    seen = 0
    for e in _m.stream_events(str(run_dir), columns=["step", "kind"]):
        if e["kind"] != "move_segment":
            continue
        counts[int(e["step"]) % int(n_steps_per_day)] += 1
        seen += 1
    if seen == 0:
        return None
    m = sum(counts) / len(counts)
    return [c / m for c in counts] if m > 0 else None


# =========================================================================== #
# 見積の組み立て
# =========================================================================== #
def theory_estimate(zones: list, agents: int, days: int, *, dt_min: int,
                    weights: list, traversals_per_day: float, zone_share: float,
                    walk_speed: float, span_m: float | None,
                    cost_model: str, measured_cost: dict | None = None) -> dict:
    """理論見積モード。conf のゾーン宣言 + 体数 → duty → サブステップ。"""
    step_seconds = dt_min * 60.0
    n_per_day = max(1, (24 * 60) // dt_min)
    n_steps = n_per_day * int(days)
    out_zones = []
    for z in zones:
        n_sub_max = n_sub_per_step(z["dt_sub"], z["max_sub_steps"], step_seconds)
        span = float(span_m) if span_m else math.sqrt(max(z["polygon_area_m2"], 1.0))
        tau = span / max(walk_speed, 1e-6)
        arrivals_day = float(agents) * float(traversals_per_day) * float(zone_share)
        subs = 0.0
        agent_subs = 0.0
        duties = []
        occs = []
        for d in range(int(days)):
            for s in range(n_per_day):
                w = weights[s % len(weights)]
                lam = arrivals_day * w / 86400.0
                duty = mg_inf_duty(lam, tau)
                occ = lam * tau                        # Little の法則(期待在圏人数)
                subs += n_sub_max * duty
                agent_subs += n_sub_max * occ
                if d == 0:
                    duties.append(duty)
                    occs.append(occ)
        mean_occ = (agent_subs / subs) if subs > 0 else 0.0
        coef = cost_coefficients(z["engine"], cost_model, measured_cost)
        out_zones.append({
            "id": z["id"], "engine": z["engine"], "dt_sub": z["dt_sub"],
            "n_sub_max": n_sub_max, "span_m": round(span, 1), "dwell_s": round(tau, 1),
            "arrivals_per_day": arrivals_day,
            "duty_mean": round(sum(duties) / len(duties), 4) if duties else 0.0,
            "duty_peak": round(max(duties), 4) if duties else 0.0,
            "occupancy_peak": round(max(occs), 1) if occs else 0.0,
            "substeps": subs, "agent_substeps": agent_subs,
            "mean_occupancy": round(mean_occ, 2),
            "cost_model": coef["model"],
            "wall_s": wall_seconds(subs, mean_occ, coef),
            "upper_bound_substeps": n_sub_max * n_steps,
        })
    return {
        "mode": "theory", "agents": int(agents), "days": int(days), "dt_min": dt_min,
        "n_steps": n_steps, "zones": out_zones,
        "substeps_total": sum(z["substeps"] for z in out_zones),
        "agent_substeps_total": sum(z["agent_substeps"] for z in out_zones),
        "wall_s_total": sum(z["wall_s"] for z in out_zones),
        "upper_bound_substeps": sum(z["upper_bound_substeps"] for z in out_zones),
        "assumptions": [
            f"到着 = ポアソン過程。λ(step) = 体数 × 通過回数/日({traversals_per_day})"
            f" × ゾーン通過率({zone_share}) × 日内重み w(step) / 86400",
            f"滞在 τ = 通過距離 / 歩行速度({walk_speed} m/s・Weidmann 1993 自由歩行)。"
            "通過距離の既定は √(ポリゴン面積)",
            "duty(塞がり率) = 1 − exp(−λτ)(M/G/∞ の空き確率。詰まりで τ が伸びる分は"
            "織り込んでいない = duty は下限側)",
            "期待在圏人数 = λτ(Little の法則)。step 内で定常とみなす",
            (f"費用 sec(n) = a + b·n²(このマシンで実測: --measure-cost)"
             if measured_cost else
             f"費用モデル = {cost_model}"
             f"(ベンチ実測 {BENCH_SOURCE} を {BENCH_N_REF} 体点として外挿。"
             "小 n の固定費 a=0 とみなすので**過小評価**。--measure-cost 推奨)"),
            "壁時計は平均在圏人数で代表させている(n の分散ぶん 2 次項が過小 = Jensen)",
            "★通過回数/日 と ゾーン通過率 は**根拠のない既定**。自分の数字に置き換えること",
        ],
    }


def measured_estimate(calib_dir: Path, zones: list, agents: int, days: int, *,
                      dt_min: int, calib_agents: int, cost_model: str,
                      measured_cost: dict | None = None) -> dict:
    """実測外挿モード。物理 ON の既存ランの L1 から数え、目標体数・日数へ伸ばす。"""
    step_seconds = dt_min * 60.0
    n_per_day = max(1, (24 * 60) // dt_min)
    scan = scan_zone_gates(calib_dir, step_seconds)
    zmap = {z["id"]: z for z in zones}
    out_zones = []
    ratio_n = (float(agents) / float(calib_agents)) if calib_agents > 0 else 1.0
    calib_days = max(1e-9, scan["n_steps"] / n_per_day)
    for zid, iv in sorted(scan["intervals"].items()):
        z = zmap.get(zid) or {"id": zid, "engine": "sfm", "dt_sub": 0.05,
                              "max_sub_steps": 12000}
        n_sub_max = n_sub_per_step(z["dt_sub"], z["max_sub_steps"], step_seconds)
        mm = measure_substeps(iv, scan["n_steps"], z["dt_sub"], n_sub_max, step_seconds)
        # 体数外挿: duty は (1−duty)^ratio 則、agent·サブステップは体数に比例。
        pred_sub_day = 0.0
        for s, v in enumerate(mm["per_step"]):
            duty = v / n_sub_max if n_sub_max else 0.0
            pred_sub_day += n_sub_max * duty_scale(duty, ratio_n)
        pred_sub_day /= calib_days
        pred_agent_day = mm["agent_substeps_total"] / calib_days * ratio_n
        subs = pred_sub_day * days
        agent_subs = pred_agent_day * days
        mean_occ = (agent_subs / subs) if subs > 0 else 0.0
        coef = cost_coefficients(z["engine"], cost_model, measured_cost)
        out_zones.append({
            "id": zid, "engine": z["engine"], "dt_sub": z["dt_sub"],
            "n_sub_max": n_sub_max, "cost_model": coef["model"],
            "measured_substeps": mm["substeps_total"],
            "measured_agent_substeps": round(mm["agent_substeps_total"], 1),
            "measured_busy_steps": mm["busy_steps"],
            "measured_saturated_steps": mm["saturated_steps"],
            "measured_steps": scan["n_steps"],
            "substeps": subs, "agent_substeps": agent_subs,
            "mean_occupancy": round(mean_occ, 2),
            "wall_s": wall_seconds(subs, mean_occ, coef),
            "upper_bound_substeps": n_sub_max * n_per_day * days,
        })
    return {
        "mode": "measured", "agents": int(agents), "days": int(days), "dt_min": dt_min,
        "calib_dir": str(calib_dir), "calib_agents": int(calib_agents),
        "calib_days": round(calib_days, 3), "calib_steps": scan["n_steps"],
        "n_enter": scan["n_enter"], "n_exit": scan["n_exit"],
        "n_unclosed": scan["n_unclosed"],
        "n_steps": n_per_day * days, "zones": out_zones,
        "substeps_total": sum(z["substeps"] for z in out_zones),
        "agent_substeps_total": sum(z["agent_substeps"] for z in out_zones),
        "wall_s_total": sum(z["wall_s"] for z in out_zones),
        "upper_bound_substeps": sum(z["upper_bound_substeps"] for z in out_zones),
        "assumptions": [
            "在圏区間は zone_gate(enter の wait_s/waited_steps・exit の dwell_s)から復元。"
            "ループは入場待ちだけでも回るので、起点は**待ち行列に並んだ step の頭**",
            f"体数外挿: 1 − duty' = (1 − duty)^({agents}/{calib_agents})"
            "(M/G/∞ の空き確率 exp(−λτ) と λ ∝ 体数から)",
            "agent·サブステップは体数に比例(Little の法則。混雑で τ が伸びる分は未計上)",
            f"日数外挿: 較正ラン {round(calib_days, 2)} 日の 1 日平均 × {days} 日"
            "(日ごとに定常 = 曜日差・イベント日を織り込まない)",
            ("費用 sec(n) = a + b·n²(このマシンで実測: --measure-cost)"
             if measured_cost else
             f"費用モデル = {cost_model}(ベンチ実測 {BENCH_SOURCE}・小 n の固定費を"
             "無視するので過小評価。--measure-cost 推奨)"),
        ],
    }


# =========================================================================== #
# レポート
# =========================================================================== #
def extrapolation_warnings(rep: dict) -> list[str]:
    """外挿の効いている方向を明示する(黙って伸ばさない)。"""
    warns: list = []
    if rep["mode"] == "measured":
        cd = float(rep.get("calib_days", 0.0))
        if cd < 1.0:
            warns.append(f"較正ランが {cd:.2f} 日 = 1 日未満。日内変動を 1 周期も見ていない"
                         "(日数外挿は形だけ)")
        r = (rep["agents"] / rep["calib_agents"]) if rep["calib_agents"] else 0
        if r > 10:
            warns.append(f"体数外挿が {r:.0f} 倍。duty はほぼ 1 に張り付くので"
                         "総サブステップは上限で頭打ち(= この見積の主張は上限そのもの)")
        if rep.get("n_unclosed"):
            warns.append(f"exit を見ていない在圏 {rep['n_unclosed']} 件はラン終端まで在圏と"
                         "みなした(過大側)")
    for z in rep["zones"]:
        if z["mean_occupancy"] > BENCH_N_REF:
            warns.append(f"zone {z['id']}: 平均在圏 {z['mean_occupancy']:.0f} 人は"
                         f"費用実測点({BENCH_N_REF} 体)の外。壁時計は 2 次外挿の値")
        ub = z.get("upper_bound_substeps") or 0
        if ub and z["substeps"] >= ub * 0.999:
            warns.append(f"zone {z['id']}: サブステップが上限に飽和"
                         "(= 体数を増やしてもこれ以上増えない。増えるのは agent·サブステップ)")
    return warns


def _hms(seconds: float) -> str:
    total_min = int(round(float(seconds) / 60.0))
    h, m = divmod(total_min, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}日{h}時間{m:02d}分"
    if h:
        return f"{h}時間{m:02d}分"
    return f"{m}分"


def _si(x: float) -> str:
    x = float(x)
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"{x/div:.2f}{unit}"
    return f"{x:.0f}"


def render_report(rep: dict, notes: list | None = None) -> str:
    out = []
    title = {"theory": "理論見積", "measured": "実測外挿"}.get(rep["mode"], rep["mode"])
    out.append("=" * 72)
    out.append(f" 物理サブステップ事前見積 [{title}]"
               f"  {rep['agents']}体 × {rep['days']}日  (Δt={rep['dt_min']}分"
               f" / {rep['n_steps']} step)")
    out.append("=" * 72)
    if rep["mode"] == "measured":
        out.append(f" 較正ラン: {rep['calib_dir']}  {rep['calib_agents']}体"
                   f" × {rep['calib_days']}日({rep['calib_steps']} step)")
        out.append(f"   zone_gate: enter {rep['n_enter']} / exit {rep['n_exit']}"
                   f" / 未閉 {rep['n_unclosed']}")
    if not rep["zones"]:
        out.append(" ゾーンが 0 件 = サブステップは 1 つも回らない(物理 OFF と同じ)")
        for n in (notes or []):
            out.append(f"   ! {n}")
        out.append("=" * 72)
        return "\n".join(out)
    out.append("-" * 72)
    hdr = (f" {'zone':<12} {'eng':<5} {'dt':>5} {'n_sub/step':>10} {'総sub':>9}"
           f" {'飽和%':>6} {'平均在圏':>8} {'agent·sub':>10} {'壁時計':>12}")
    out.append(hdr)
    out.append("-" * 72)
    for z in rep["zones"]:
        sat = (z["substeps"] / z["upper_bound_substeps"] * 100.0
               if z["upper_bound_substeps"] else 0.0)
        out.append(f" {z['id']:<12} {z['engine']:<5} {z['dt_sub']:>5.3f}"
                   f" {z['n_sub_max']:>10,} {_si(z['substeps']):>9} {sat:>5.1f}%"
                   f" {z['mean_occupancy']:>8.1f} {_si(z['agent_substeps']):>10}"
                   f" {_hms(z['wall_s']):>12}")
        if rep["mode"] == "measured":
            out.append(f"   └ 実測: sub {z['measured_substeps']:,}"
                       f" / agent·sub {_si(z['measured_agent_substeps'])}"
                       f" / 稼働 {z['measured_busy_steps']}step"
                       f"(うち上限飽和 {z['measured_saturated_steps']})")
        if rep["mode"] == "theory":
            out.append(f"   └ 通過距離 {z['span_m']}m / 滞在 {z['dwell_s']}s"
                       f" / 到着 {_si(z['arrivals_per_day'])}人日"
                       f" / duty 平均 {z['duty_mean']:.3f}・ピーク {z['duty_peak']:.3f}"
                       f" / ピーク在圏 {z['occupancy_peak']}人")
    out.append("-" * 72)
    ub = rep["upper_bound_substeps"]
    out.append(f" 総サブステップ  {rep['substeps_total']:,.0f}"
               f"   (理論上限 {ub:,} = 全 step で上限まで回した場合"
               f" / 充足率 {rep['substeps_total']/ub*100 if ub else 0:.1f}%)")
    out.append(f" 総 agent·サブステップ  {rep['agent_substeps_total']:,.0f}"
               f"   ← 実際の演算量はこちら")
    out.append(f" 推定 壁時計          {_hms(rep['wall_s_total'])}"
               f"  ({rep['wall_s_total']/3600:.1f} 時間・物理パートのみ・1 プロセス直列)")
    out.append("-" * 72)
    out.append(" [仮定と式(隠さない)]")
    for a in rep["assumptions"]:
        out.append(f"   ・{a}")
    warns = rep.get("warnings") or extrapolation_warnings(rep)
    if warns:
        out.append(" [外挿警告]")
        for w in warns:
            out.append(f"   ! {w}")
    for n in (notes or []):
        out.append(f"   ! {n}")
    out.append("=" * 72)
    return "\n".join(out)


# =========================================================================== #
# CLI
# =========================================================================== #
def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_ROOT / path)


def _agent_count(run_dir: Path) -> int:
    p = run_dir / "agents.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                if isinstance(data.get("agents"), list):
                    return len(data["agents"])
                return int(data.get("n_agents", 0))
        except (ValueError, OSError):
            pass
    sp = run_dir / "summary.json"
    if sp.exists():
        try:
            return int(json.loads(sp.read_text(encoding="utf-8")).get("n_agents", 0))
        except (ValueError, OSError):
            pass
    return 0


def main(argv: list) -> int:
    for _s in (sys.stdout, sys.stderr):            # Windows cp932 対策(既存 scripts と同じ)
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="物理(SFM/ORCA)ゾーンの総サブステップを事前見積する(竹-4 残⑦)")
    ap.add_argument("--days", type=int, default=10, help="見積る日数(既定 10=本選)")
    ap.add_argument("--agents", type=int, default=10000, help="見積る体数")
    ap.add_argument("--conf", type=str, default="conf/config.yaml",
                    help="ゾーン宣言を読む conf(既定 conf/config.yaml)")
    ap.add_argument("--dt-min", type=int, default=None,
                    help="1 step の分数(既定=conf の run.dt_min。無ければ 10)")
    ap.add_argument("--calib", type=str, default=None,
                    help="実測外挿の較正ラン dir(物理 ON=zone_gate を含むもの)")
    ap.add_argument("--calib-agents", type=int, default=None,
                    help="較正ランの体数(既定=agents.json から読む)")
    ap.add_argument("--profile-run", type=str, default=None,
                    help="日内プロファイルを借りるラン dir(move_segment の step 別件数)")
    ap.add_argument("--traversals-per-agent-day", type=float,
                    default=DEFAULT_TRAVERSALS_PER_DAY,
                    help=f"1 体が 1 日にそのゾーンを通る回数(既定 {DEFAULT_TRAVERSALS_PER_DAY})")
    ap.add_argument("--zone-share", type=float, default=DEFAULT_ZONE_SHARE,
                    help=f"そのゾーンを経路に含む個体の割合(既定 {DEFAULT_ZONE_SHARE})")
    ap.add_argument("--walk-speed", type=float, default=DEFAULT_WALK_SPEED,
                    help=f"歩行速度 [m/s](既定 {DEFAULT_WALK_SPEED}=Weidmann 1993)")
    ap.add_argument("--span-m", type=float, default=None,
                    help="ゾーン通過距離 [m](既定=√ポリゴン面積)")
    ap.add_argument("--active-hours", type=float, default=18.0,
                    help="到着がある時間帯の長さ [h](既定 18・06:00 起点)")
    ap.add_argument("--night-factor", type=float, default=0.1,
                    help="夜間の相対到着率(既定 0.1)")
    ap.add_argument("--cost-model", choices=["quadratic", "linear"], default="quadratic",
                    help="1 刻みの費用の在圏人数依存(既定 quadratic=n×n 距離行列)")
    ap.add_argument("--measure-cost", action="store_true",
                    help="このマシンで 1 サブステップの実費用を測って sec(n)=a+b·n² を当てる"
                         "(数秒。シムは 1 バイトも動かさない純粋なマイクロベンチ)")
    ap.add_argument("--json", type=str, default=None, help="レポート dict の JSON 出力先")
    args = ap.parse_args(argv)

    conf_path = _resolve(args.conf)
    notes: list = []
    if not conf_path.exists():
        notes.append(f"conf が見つからない: {conf_path}(ゾーン 0 件として扱う)")
        zones, znotes = [], []
    else:
        zones, znotes = load_zones(conf_path)
    notes.extend(znotes)
    dt_min = dt_min_of(conf_path if conf_path.exists() else None, args.dt_min)
    n_per_day = max(1, (24 * 60) // dt_min)

    weights = diurnal_weights(n_per_day, args.active_hours, args.night_factor, dt_min)
    if args.profile_run:
        pr = _resolve(args.profile_run)
        prof = street_profile(pr, n_per_day) if pr.exists() else None
        if prof:
            weights = prof
            notes.append(f"日内プロファイルを {pr} の move_segment 実測で置換")
        else:
            notes.append(f"--profile-run {pr} からプロファイルを作れず 2 帯モデルのまま")

    measured_cost = None
    if args.measure_cost:
        measured_cost = {}
        for eng in sorted({z["engine"] for z in zones} or {"sfm"}):
            try:
                measured_cost[eng] = measure_engine_cost(eng)
            except Exception as exc:               # 依存不足・API 変更など
                notes.append(f"--measure-cost({eng})に失敗: {type(exc).__name__}: {exc}")
        if not measured_cost:
            measured_cost = None
        else:
            for eng, c in sorted(measured_cost.items()):
                notes.append(f"実測費用[{eng}] sec(n) = {c['a']:.3e} + {c['b']:.3e}·n²"
                             f"(点: {c.get('points')})")

    reports = []
    if args.calib:
        cd = _resolve(args.calib)
        if not (cd / "l1_events.parquet").exists():
            notes.append(f"較正ランに l1_events.parquet が無い: {cd}(実測モードを飛ばす)")
        else:
            ca = args.calib_agents or _agent_count(cd)
            if ca <= 0:
                ca = 1
                notes.append(f"較正ランの体数が読めないので 1 とみなす: {cd}")
            rep = measured_estimate(cd, zones, args.agents, args.days, dt_min=dt_min,
                                    calib_agents=ca, cost_model=args.cost_model,
                                    measured_cost=measured_cost)
            if not rep["zones"]:
                notes.append(f"{cd} に zone_gate が 1 件も無い = 物理 OFF のラン。"
                             "実測外挿はできない(理論見積だけを出す)")
            else:
                reports.append(rep)
    reports.append(theory_estimate(
        zones, args.agents, args.days, dt_min=dt_min, weights=weights,
        traversals_per_day=args.traversals_per_agent_day, zone_share=args.zone_share,
        walk_speed=args.walk_speed, span_m=args.span_m, cost_model=args.cost_model,
        measured_cost=measured_cost))

    for rep in reports:
        rep["warnings"] = extrapolation_warnings(rep)
        print(render_report(rep, notes))
    if args.json:
        jp = _resolve(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({"reports": reports, "notes": notes},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON: {os.path.relpath(jp, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
