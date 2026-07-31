#!/usr/bin/env python3
"""σ_c 実測ハーネス(第80バッチ 2026-08-01)= 発火判定式の**分母**を測って凍結する。

正典: docs/plans/source/cognition-design-record.md §2.3 / §7-2「σ_c 実測 — フラット固定 g で
      1 ラン、チャンネル分散を収集して固定」/ docs/plans/cognition-physics-plan.md §4 第80行。

なぜ必要か
----------
    S_i(t) = Σ_c  g_ic · |o_c(t) − ô_ic| / σ_c  +  Σ_j  w_ij · [trigger_j 成立]

設計 §2.3:「σ_c はそのチャンネルの実測標準偏差。**これで割ることが必須。** 割らないと LLM が
出す期待値に単位がなく、エージェント間・チャンネル間で g が比較できなくなる」。そして
「σ_c は**事前のパイロットランで実測して固定する**(ラン中に更新すると決定論が壊れる)」。

本スクリプトがやること
----------------------
  1. mock バックエンドのパイロットラン(既定 100 体 × 2 日相当 = 288 step)を
     `cognition.channels.enabled=true` で回し、`channels.parquet` を作る
  2. それを読んでチャンネル別の標準偏差・分布要約を算出する
  3. `data/calib/sigma_c.json` へ凍結する(sha256・生成条件・ラン設定を同梱)

**決定論**: 同じ引数なら**バイト同一**の JSON を出す(壁時計時刻を一切書かない。
`--stamp` を付けたときだけ生成時刻を書く = 意図的に非決定にする)。

σ = 0(定数チャンネル)の扱い — **除外**
-----------------------------------------
σ ≤ `--sigma-floor`(既定 1e-9)または有効標本 < `--min-samples` のチャンネルは
`usable: false` + `reason` を立てて凍結する。消費側(第81)は `usable: false` を
**S から外す**(`cognition/calib.py: sigma_of()` が既にそう実装してある)。

最小床(σ←ε)を敷く案は採らない: 定数チャンネルに床を敷くと 1/ε が巨大な利得になり、
浮動小数の最下位ビットの揺れが発火を駆動する装置になる(測定器が現象に反応してしまう)。
床の値そのものはファイルに残すので、後から方針を変えたい人が根拠つきで切り替えられる。

2 種類の σ を出す(使うのは pooled)
------------------------------------
  sigma          … 全 (agent, step) 標本をプールした標準偏差(ddof=1)。**これを凍結値にする**
  sigma_within   … 個体ごとに標準偏差を出して個体平均したもの(参考値)
プールした σ には**個体間の差**が入る。|o − ô| は個体内の偏差なので within のほうが
概念的には近いが、(a) g の比較可能性は「チャンネル間で同じ物差し」であることが本質で
あり、(b) within は個体ごとの標本数が少ないと不安定になる。よって凍結値は pooled とし、
within は診断値として併記する(両方が桁違いなら個体間異質性が支配的だと判る)。

使い方
------
    # 既定のパイロット(100 体 × 2 日 = 288 step, mock, seed 42)
    python scripts/measure_sigma.py
    # 規模を変える / 別ファイルへ出す
    python scripts/measure_sigma.py --agents 40 --days 1 --out /tmp/sigma.json
    # 既にあるランの channels.parquet から測り直すだけ(ランを回さない)
    python scripts/measure_sigma.py --from-parquet runs/pilot/channels.parquet
    # 追加の conf 上書き(観測対象の機構を ON にして σ を測りたいとき)
    python scripts/measure_sigma.py --set weather.enabled=true --set disaster.enabled=true

正直な限界
----------
- **mock バックエンドのパイロットで測った σ である**。実 LLM のランでは発話量が変わるので
  `ext.heard` の σ は動きうる(本選前に実 LLM の診断ランで測り直すのが正しい)。
- 既定パイロットでは天候・災害・退屈ゲージなどが OFF なので、それらのチャンネルは
  σ=0 または欠測になる。**それが判るように凍結する**のが本スクリプトの役割で、
  「とりあえず 1 を入れておく」ことはしない。
- 2 日は日周期を 2 周しか回っていない。日周期の振幅が σ に十分に入っているかは
  `--days` を変えた感度確認でしか判らない(生成条件をファイルに残してある)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.cognition import calib as CALIB          # noqa: E402
from society.cognition import channels as CH          # noqa: E402
from society.config import load_config                # noqa: E402
from society.engine.simulation import Simulation      # noqa: E402

SCHEMA = CALIB.SCHEMA
DEFAULT_OUT = REPO_ROOT / CALIB.SIGMA_DEFAULT_REL
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "_sigma_pilot"
STEPS_PER_DAY = 144                                    # Δt=10 分(正準)


# --------------------------------------------------------------------------- #
# パイロットラン
# --------------------------------------------------------------------------- #
def run_pilot(*, agents: int, steps: int, seed: int, run_dir: Path,
              extra: list[str], keep: bool) -> Path:
    """mock パイロットを回して channels.parquet のパスを返す。"""
    if run_dir.exists() and not keep:
        # 安全弁: 「前回のラン出力」以外は消さない(--run-dir の打ち間違いで任意の
        # ディレクトリを消さないため)。空 dir か config.yaml のある dir だけを再利用する。
        if any(run_dir.iterdir()) and not (run_dir / "config.yaml").exists():
            raise SystemExit(
                f"--run-dir がラン出力ディレクトリに見えない(config.yaml が無い): {run_dir}\n"
                "  別のパスを指すか、--keep-run-dir を付けて再利用すること。")
        shutil.rmtree(run_dir)
    dotlist = [
        f"run.seed={seed}", f"run.n_agents={agents}", f"run.n_steps={steps}",
        "run.name=_sigma_pilot", "model.backend=mock",
        "cognition.channels.enabled=true", "cognition.channels.every_steps=1",
        *extra,
    ]
    sim = Simulation(load_config(dotlist), out_dir=run_dir)
    sim.run()
    path = run_dir / "channels.parquet"
    if not path.exists():
        raise RuntimeError(f"パイロットが channels.parquet を出さなかった: {path}")
    return path


# --------------------------------------------------------------------------- #
# 統計(numpy に頼らず素の Python で。桁の再現性を自分で握るため)
# --------------------------------------------------------------------------- #
def _std(values: list[float]) -> float:
    """標本標準偏差(ddof=1)。n<2 は 0.0。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


def _round(value: float, digits: int) -> float:
    """凍結値の丸め(プラットフォーム間で表記が揺れないように桁を固定する)。"""
    r = round(float(value), digits)
    return 0.0 if r == 0.0 else r        # -0.0 を 0.0 に潰す


def summarize(path: Path, *, sigma_floor: float, min_samples: int,
              digits: int) -> tuple[dict, dict]:
    """channels.parquet → (チャンネル別要約, 標本メタ)。"""
    table = pq.read_table(path)
    names = set(table.column_names)
    for key in CH.KEY_COLUMNS:
        if key not in names:
            raise ValueError(f"{path} に固定列 {key} が無い")
    agent_ids = table.column("agent_id").to_pylist()
    steps = table.column("step").to_pylist()
    n_rows = len(agent_ids)

    out: dict[str, dict] = {}
    for ch in CH.CHANNELS:
        col = ch.column
        if col not in names:
            out[ch.id] = {"sigma": 0.0, "usable": False, "reason": "column_absent",
                          "n": 0, "n_missing": n_rows}
            continue
        raw = table.column(col).to_pylist()
        pairs = [(aid, v) for aid, v in zip(agent_ids, raw) if v is not None]
        vals = [float(v) for _aid, v in pairs]
        n = len(vals)
        n_missing = n_rows - n
        entry: dict = {"n": n, "n_missing": n_missing,
                       "source": ch.source, "unit": ch.unit,
                       "implemented": bool(ch.implemented)}
        if not ch.implemented:
            entry.update({"sigma": 0.0, "usable": False, "reason": "not_implemented"})
            out[ch.id] = entry
            continue
        if n < max(2, min_samples):
            entry.update({"sigma": 0.0, "usable": False,
                          "reason": ("no_samples" if n == 0 else "too_few_samples")})
            out[ch.id] = entry
            continue
        sigma = _std(vals)
        # 個体内 σ の個体平均(診断値)
        by_agent: dict[int, list[float]] = {}
        for aid, v in pairs:
            by_agent.setdefault(int(aid), []).append(float(v))
        within = [_std(vs) for vs in by_agent.values() if len(vs) >= 2]
        mean = math.fsum(vals) / n
        entry.update({
            "sigma": _round(sigma, digits),
            "sigma_within": _round(
                (math.fsum(within) / len(within)) if within else 0.0, digits),
            "mean": _round(mean, digits),
            "min": _round(min(vals), digits),
            "max": _round(max(vals), digits),
            "n_unique": len(set(vals)),
            "n_agents": len(by_agent),
            "usable": bool(sigma > sigma_floor),
        })
        if not entry["usable"]:
            entry["reason"] = "constant_channel"
        out[ch.id] = entry

    meta = {"n_rows": n_rows,
            "n_agents": len(set(agent_ids)),
            "n_steps": len(set(steps)),
            "step_min": (min(steps) if steps else -1),
            "step_max": (max(steps) if steps else -1)}
    return out, meta


# --------------------------------------------------------------------------- #
# 凍結ファイル
# --------------------------------------------------------------------------- #
def build_doc(channels: dict, sample: dict, *, run_cfg: dict,
              sigma_floor: float, min_samples: int, digits: int,
              stamp: bool) -> dict:
    doc = {
        "meta": {
            "schema": SCHEMA,
            "kind": "sigma_c",
            "generator": "scripts/measure_sigma.py",
            # チャンネル定義そのもののハッシュ。定義が 1 本でも変われば σ の凍結は無効になる。
            "channel_spec_sha256": CH.spec_sha256(),
            "n_channels": len(CH.CHANNELS),
            "policy": {
                "sigma": "pooled over all (agent, step) samples, ddof=1",
                "zero_sigma": "excluded (usable=false). no floor is applied to sigma.",
                "sigma_floor": sigma_floor,
                "min_samples": min_samples,
                "round_digits": digits,
                "missing": "missing values are dropped, never imputed with 0",
            },
            "run": run_cfg,
            "sample": sample,
        },
        "channels": channels,
    }
    if stamp:                                          # 明示指定時のみ(=決定論を捨てる)
        doc["meta"]["generated_at"] = (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    doc["meta"]["payload_sha256"] = CALIB.payload_sha256(doc)
    return doc


def write_doc(doc: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=100, help="パイロットの人数(既定 100)")
    ap.add_argument("--days", type=float, default=2.0, help="パイロットの日数(既定 2)")
    ap.add_argument("--steps", type=int, default=0,
                    help="step 数を直接指定(>0 のとき --days より優先)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--keep-run-dir", action="store_true",
                    help="既存のラン出力を消さずに再利用する")
    ap.add_argument("--from-parquet", type=Path, default=None,
                    help="既存の channels.parquet から測り直す(ランを回さない)")
    ap.add_argument("--set", dest="extra", action="append", default=[],
                    help="追加の conf 上書き(dotlist。複数可)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sigma-floor", type=float, default=1e-9,
                    help="これ以下の σ は定数チャンネルとして usable=false にする")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--digits", type=int, default=10, help="凍結値の丸め桁")
    ap.add_argument("--stamp", action="store_true",
                    help="生成時刻を書く(**非決定になる**。既定は書かない)")
    ap.add_argument("--print-only", action="store_true", help="書き出さずに要約だけ表示")
    args = ap.parse_args(argv)

    steps = int(args.steps) if args.steps > 0 else int(round(args.days * STEPS_PER_DAY))
    if args.from_parquet is not None:
        path = Path(args.from_parquet)
        run_cfg = {"source": "existing_parquet", "path": CALIB.rel_path(path)}
    else:
        path = run_pilot(agents=args.agents, steps=steps, seed=args.seed,
                         run_dir=Path(args.run_dir), extra=list(args.extra),
                         keep=args.keep_run_dir)
        run_cfg = {"source": "pilot", "backend": "mock", "n_agents": int(args.agents),
                   "n_steps": steps, "seed": int(args.seed),
                   "overrides": sorted(str(x) for x in args.extra)}

    channels, sample = summarize(path, sigma_floor=args.sigma_floor,
                                 min_samples=args.min_samples, digits=args.digits)
    doc = build_doc(channels, sample, run_cfg=run_cfg, sigma_floor=args.sigma_floor,
                    min_samples=args.min_samples, digits=args.digits,
                    stamp=bool(args.stamp))

    print(f"標本: {sample['n_rows']} 行 "
          f"({sample['n_agents']} 体 × {sample['n_steps']} step)")
    print(f"{'channel':24s} {'source':10s} {'sigma':>12s} {'within':>12s} "
          f"{'mean':>10s} {'n':>7s} {'miss':>6s}  usable")
    for ch in CH.CHANNELS:
        row = channels[ch.id]
        print(f"{ch.id:24s} {ch.source:10s} {row.get('sigma', 0.0):12.6f} "
              f"{row.get('sigma_within', 0.0):12.6f} {row.get('mean', 0.0):10.4f} "
              f"{row.get('n', 0):7d} {row.get('n_missing', 0):6d}  "
              f"{'yes' if row.get('usable') else 'NO (' + str(row.get('reason', '')) + ')'}")
    n_usable = sum(1 for r in channels.values() if r.get("usable"))
    print(f"usable: {n_usable}/{len(channels)}")

    if args.print_only:
        return 0
    out = write_doc(doc, Path(args.out))
    print(f"凍結: {out}")
    print(f"payload_sha256: {doc['meta']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
