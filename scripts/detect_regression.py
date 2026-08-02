#!/usr/bin/env python
"""退行シグナルの判定(第91バッチ・後処理・**L1/L2 を読むだけ**・シム本体に非依存)。

    # 通常(ラン終了後)
    python scripts/detect_regression.py runs/smoke_tall --alpha 0.05 --min-rel-slope 0.02

    # 実行中のランを覗く(完結した part だけ読む。--quick は 1 行 JSON のみ)
    python scripts/detect_regression.py runs/smoke_tall --quick --alpha 0.05 --min-rel-slope 0.02

    # 分散崩壊診断(N を変えた複数ラン。統計的平滑化とモデル由来の均質化を切り分ける)
    python scripts/detect_regression.py --variance-collapse runs/n50 runs/n100 runs/n200 \
        --alpha 0.05 --min-rel-slope 0.02 --out runs/n200

設計正典: docs/plans/source/design-discussion-20260802.md **§3** /
          判定基準の根拠と較正手順: docs/research/regression-signals.md。
列の定義(単一の源): src/society/observer/regression.py。

何を判定するか
--------------
§3 が名指しする 4 群を、**L2 の監視列の時系列**として読む:

  ① 行動分散の単調**減少**        reg_act_between_var / reg_act_entropy_mean
  ② 訪問地点エントロピーの**低下** reg_visit_entropy
  ③ 語彙エントロピー**低下** + n-gram 重複率**上昇**
                                  reg_vocab_entropy / reg_ngram_repeat_rate
  ④ 発火率の 0 / 飽和への**張り付き** reg_fire_zero_frac / reg_fire_sat_frac

「どちらへ動いたら退行か」は **スクリプトが持たない**。方向は
`society.observer.regression.REGRESSION_DIRECTION` が単一の源で、列を足したときに
判定方向を書き忘れる経路が構造上存在しない。

★閾値は**引数必須**(`--alpha` / `--min-rel-slope`)。事前登録(U-10)・
`analyze_specialization.py` と同じ流儀で、コードに既定値を埋め込まない。

検定(決定論・乱数ゼロ)
------------------------
**Mann-Kendall 傾向検定**(Mann 1945 / Kendall 1975。同順位補正つき正規近似)+
**Theil-Sen 傾向の大きさ**(中央値勾配)。ノンパラメトリックで、外れ値と非正規性に強い。

  S = Σ_{i<j} sgn(x_j − x_i)、Var(S) = [n(n−1)(2n+5) − Σ_t t(t−1)(2t+5)] / 18
  z = (S − sgn(S)) / √Var(S)、p = 2(1 − Φ(|z|))(両側)

★**自己相関の扱い(最重要)**: L2 の監視列は rolling 窓(既定 144 step)の集計なので、
隣接行は窓の 99% 以上を共有する。この系列をそのまま MK にかけると有効標本数が桁で
過大になり、**どんな微小な傾きでも p<0.001 になる**。したがって既定で
**窓幅ぶん間引いてから**検定する(`--stride 0` = ランの `window_steps` を manifest /
config から自動採用 = 非重複標本)。`--stride` で明示的に上書きできるが、
窓幅未満にすると有意性は信用できない(その旨をレポートに必ず出す)。

判定式(3 条件の連言。どれか 1 つでも欠けたら「退行なし」)
----------------------------------------------------------
  1. p < `--alpha`
  2. 傾きの符号が `REGRESSION_DIRECTION` と一致する
  3. |相対傾き| ≥ `--min-rel-slope`
     相対傾き = Theil-Sen 勾配(1 標本 = 1 窓あたり)÷ 系列の平均の絶対値
     (= 「1 シミュ日あたり平均の何割ぶん動いたか」。単位に依存しない効果量)

★n が大きいと微小な傾きでも p は容易に小さくなるので、**効果量との連言**にしている
(`diagnose_stationarity.py` / `analyze_specialization.py` と同じ作法)。

分散崩壊診断(§3 の「統計的平滑化(1/√N)とモデル由来の均質化を区別する」)
-----------------------------------------------------------------------------
**個体間分散の不偏推定量(n−1)は、iid 標本なら期待値が N に依存しない。**
したがって N を増やして下がるなら、それは「サンプルが増えて平均が滑らかになった」
のとは別の現象である。区別は 2 本の曲線を重ねて行う:

  (a) **標本曲線(帰無)** … *ひとつの*ラン(最大 N)の個体集合から m 体を無作為抽出して
      分散を計算し、m を掃く。母集団は同一なので **期待値は平坦**で、幅だけが
      m とともに縮む(≈ V·√(2/(m−1)))。これが「統計的平滑化で説明できる範囲」。
  (b) **実測曲線** … N の違うラン(それぞれ全個体)の分散。

  (b) が (a) の帯を**下に**外れたら、母集団そのものが均質化している
  = **モデル由来の均質化**。帯の中なら統計的平滑化(推定ノイズ)で説明がつく。

参考の傾き線として log-log 上に **b=0(崩壊なし)** と **b=−0.5(1/√N)** を引く。

出力
----
  <out>/regression_report.json  … 機械可読(sort_keys=True でバイト同一)
  <out>/regression_report.md    … 日本語の判定表
  <out>/regression_variance.svg … 分散崩壊診断の図(--variance-collapse のときだけ)

依存: pyarrow / numpy / stdlib のみ(pandas・duckdb 禁止=リポジトリ不変則。
matplotlib も使わない=素の SVG。`make_endo_report.py` と同じ流儀)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

try:                                            # Windows cp932 対策
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from society.observer import regression as RG          # noqa: E402

# 第77バッチの教訓(Windows: 素の open() は FILE_SHARE_DELETE を立てないので、
# 読んでいる最中の part をランが unlink できず **シム本体が finalize で落ちる**)を
# そのまま流用する。実装の単一の源は live_viewer.py。
from live_viewer import (                              # noqa: E402
    _open_shared, is_complete_parquet, list_parts,
)

STEPS_PER_DAY = 144


# --------------------------------------------------------------------------- #
# 統計(すべて決定論・stdlib + numpy)
# --------------------------------------------------------------------------- #
def _phi(x: float) -> float:
    """標準正規の累積分布(math.erf のみ)。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall(xs) -> dict:
    """Mann-Kendall 傾向検定(同順位補正つき正規近似)+ Theil-Sen 勾配。

    n < 4 は検定しない(`method="skipped(n<4)"`・p=1.0)。等間隔標本を前提にする。
    """
    vals = [float(v) for v in xs]
    n = len(vals)
    out = {"n": n, "S": 0, "z": 0.0, "p": 1.0, "sen_slope": 0.0,
           "method": "mann_kendall(normal_approx,tie_corrected)"}
    if n < 4:
        out["method"] = f"skipped(n<4: n={n})"
        return out
    s = 0
    for i in range(n - 1):
        vi = vals[i]
        for j in range(i + 1, n):
            d = vals[j] - vi
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    ties = Counter(vals)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in ties.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var <= 0.0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    slopes = [(vals[j] - vals[i]) / (j - i)
              for i in range(n - 1) for j in range(i + 1, n)]
    slopes.sort()
    m = len(slopes)
    sen = (slopes[m // 2] if m % 2 else 0.5 * (slopes[m // 2 - 1] + slopes[m // 2]))
    out.update({"S": int(s), "z": round(z, 6),
                "p": round(2.0 * (1.0 - _phi(abs(z))), 8),
                "sen_slope": round(sen, 9)})
    return out


def unbiased_var(values) -> float:
    """不偏分散(n−1)。observer/regression.unbiased_var と同一定義(そちらを呼ぶ)。"""
    return RG.unbiased_var(values)


# --------------------------------------------------------------------------- #
# L2 の読み込み(canonical / 実行中の part の両対応。part は完結したものだけ)
# --------------------------------------------------------------------------- #
def _read_parquet_shared(path: Path, columns=None):
    """`_open_shared` 経由で parquet を読む(読むだけでランを壊さないため)。"""
    with _open_shared(path) as f:
        return pq.read_table(f, columns=columns)


def load_l2(run_dir: Path) -> tuple[list, dict, dict]:
    """(steps, {col: [値]}, meta) を返す。canonical が無ければ完結 part を index 順に連結。"""
    meta = {"source": None, "parts": 0, "skipped_parts": []}
    canonical = run_dir / "l2_metrics.parquet"
    tables = []
    if canonical.exists():
        meta["source"] = "canonical"
        tables.append(_read_parquet_shared(canonical))
    else:
        meta["source"] = "parts"
        for idx, path in list_parts(run_dir, "l2_metrics"):
            if not is_complete_parquet(path):
                meta["skipped_parts"].append(path.name)
                continue
            tables.append(_read_parquet_shared(path))
            meta["parts"] += 1
    if not tables:
        return [], {}, meta
    names = [c for c in tables[0].column_names]
    cols: dict = {c: [] for c in names}
    steps: list = []
    for t in tables:
        d = t.to_pydict()
        n_rows = t.num_rows
        for c in names:
            cols[c].extend(d.get(c, [None] * n_rows))
        if "step" in d:
            steps.extend(d["step"])
        else:                                   # step 列が無いランは行番号で代用
            base = len(steps)
            steps.extend(range(base, base + n_rows))
    return list(steps), cols, meta


def run_window_steps(run_dir: Path) -> int:
    """そのランの監視窓幅(manifest → config.yaml の順に探す。見つからねば 1 日)。"""
    man = run_dir / "run_manifest.json"
    if man.exists():
        try:
            blob = json.loads(man.read_text(encoding="utf-8"))
            w = ((blob.get("regression") or {}).get("window_steps"))
            if w:
                return int(w)
        except Exception:                       # noqa: BLE001
            pass
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:                                    # 素朴な行走査(omegaconf を呼ばない)
            in_reg = False
            for line in cfg.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("regression:"):
                    in_reg = True
                    continue
                if in_reg:
                    if s.startswith("window_steps:"):
                        return int(s.split(":", 1)[1].strip())
                    if s and not line.startswith((" ", "\t")):
                        in_reg = False
        except Exception:                       # noqa: BLE001
            pass
    return STEPS_PER_DAY


def downsample(steps, vals, stride: int, warmup: int = 0,
               first_step: int | None = None) -> tuple[list, list]:
    """stride step ごとに 1 点だけ拾う(非重複標本にするための間引き)。

    `warmup` step は捨てる。rolling 窓は**ラン冒頭では満ちていない**(窓幅に満たない
    step 数で集計される)ので、そこを混ぜると「窓が満ちていく過程」が単調増加として
    検定に載る = 退行の逆向きの偽陽性になる。既定は窓幅ぶん(= 最初の 1 窓を捨てる)。
    """
    out_s, out_v, last = [], [], None
    if first_step is None:
        for s, v in zip(steps, vals):
            if v is not None:
                first_step = int(s)
                break
    floor = (int(first_step) + int(warmup)) if first_step is not None else 0
    for s, v in zip(steps, vals):
        if v is None:
            continue
        s = int(s)
        if s < floor:
            continue
        if last is None or s - last >= stride:
            out_s.append(s)
            out_v.append(float(v))
            last = s
    return out_s, out_v


# --------------------------------------------------------------------------- #
# ① 時系列トレンド検定
# --------------------------------------------------------------------------- #
def trend_report(run_dir: Path, alpha: float, min_rel_slope: float,
                 stride: int, warmup: int = -1) -> dict:
    steps, cols, meta = load_l2(run_dir)
    present = [c for c in RG.COLUMNS if c in cols]
    win = run_window_steps(run_dir)
    eff_stride = int(stride) if stride > 0 else int(win)
    eff_warmup = int(win) if warmup < 0 else int(warmup)
    res = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "l2": meta,
        "n_rows": len(steps),
        "window_steps": int(win),
        "stride": eff_stride,
        "stride_source": "arg" if stride > 0 else "run(window_steps)",
        "warmup": eff_warmup,
        "warmup_source": "run(window_steps)" if warmup < 0 else "arg",
        "stride_below_window": bool(eff_stride < win),
        "columns_present": present,
        "columns_missing": [c for c in RG.COLUMNS if c not in cols],
        "thresholds": {"alpha": alpha, "min_rel_slope": min_rel_slope},
        "signals": {},
        "flagged": [],
    }
    if not present:
        res["verdict"] = "NO_COLUMNS"
        res["why"] = ("L2 に退行シグナル列が 1 つも無い。"
                      "`observer.regression.enabled=true` で回したランを渡すこと。")
        return res
    for col in present:
        direction = RG.REGRESSION_DIRECTION.get(col)
        ds_steps, ds_vals = downsample(steps, cols[col], eff_stride, eff_warmup)
        mk = mann_kendall(ds_vals)
        mean = (sum(ds_vals) / len(ds_vals)) if ds_vals else 0.0
        denom = abs(mean)
        rel = (mk["sen_slope"] / denom) if denom > 1e-12 else 0.0
        row = {
            "direction_of_regression": direction,
            "n_samples": len(ds_vals),
            "first": round(ds_vals[0], 6) if ds_vals else None,
            "last": round(ds_vals[-1], 6) if ds_vals else None,
            "mean": round(mean, 6),
            "rel_slope": round(rel, 6),
            **mk,
        }
        if direction is None:
            row["verdict"] = "CONTEXT_ONLY"     # 分母・母数の列(判定方向を持たない)
        elif mk["method"].startswith("skipped"):
            row["verdict"] = "INSUFFICIENT"
        else:
            sign_ok = (mk["sen_slope"] * direction) > 0.0
            row["verdict"] = ("REGRESSION"
                              if (mk["p"] < alpha and sign_ok
                                  and abs(rel) >= min_rel_slope)
                              else "OK")
        res["signals"][col] = row
        if row["verdict"] == "REGRESSION":
            res["flagged"].append(col)
    n_test = sum(1 for r in res["signals"].values()
                 if r["verdict"] in ("REGRESSION", "OK"))
    if n_test == 0:
        res["verdict"] = "INSUFFICIENT"
        res["why"] = (f"間引き後の標本が 4 点未満。窓 {res['window_steps']} step の"
                      f"ウォームアップ {eff_warmup} step を捨てたうえで "
                      f"{eff_stride} step ごとに 1 点拾うので、判定には最低でも "
                      f"{eff_warmup + 4 * eff_stride} step のランが要る。")
    elif res["flagged"]:
        res["verdict"] = "REGRESSION"
        res["why"] = ("退行方向の単調傾向が閾値を超えた列がある: "
                      + ", ".join(res["flagged"]))
    else:
        res["verdict"] = "OK"
        res["why"] = "全シグナルで、退行方向の単調傾向は閾値に達していない。"
    return res


# --------------------------------------------------------------------------- #
# ② 分散崩壊診断(L1 から個体別 act 分布を組む)
# --------------------------------------------------------------------------- #
def _l1_paths(run_dir: Path) -> list[Path]:
    canonical = run_dir / "l1_events.parquet"
    if canonical.exists():
        return [canonical]
    return [p for _i, p in list_parts(run_dir, "l1_events")
            if is_complete_parquet(p)]


def act_counts_from_l1(run_dir: Path, window_steps: int,
                       batch_rows: int = 262_144) -> tuple[dict, int, int]:
    """ランの**末尾 window_steps**の個体別 act 件数 {aid: {act_index: n}} を作る。

    L2 の最終行と同じ窓を、L1 から**独立に**再計算する経路(= 列の独立検算にもなる)。
    payload 列は 1 バイトも読まない(kind と agent_id と step だけ)。
    """
    paths = _l1_paths(run_dir)
    if not paths:
        return {}, -1, 0
    max_step = -1
    for path in paths:                          # パス 1: 最終 step を確定(step 列だけ)
        with _open_shared(path) as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=["step"]):
                col = batch.column(0).to_pylist()
                if col:
                    m = max(col)
                    if m > max_step:
                        max_step = int(m)
    if max_step < 0:
        return {}, -1, 0
    floor = max_step - int(window_steps) + 1
    acts: dict = {}
    for path in paths:                          # パス 2: 窓内の act だけ畳む
        with _open_shared(path) as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(
                    batch_size=batch_rows,
                    columns=["step", "agent_id", "kind"]):
                d = batch.to_pydict()
                for s, aid, kind in zip(d["step"], d["agent_id"], d["kind"]):
                    if s < floor:
                        continue
                    idx = RG._ACT_INDEX.get(kind)
                    if idx is None or aid is None or int(aid) < 0:
                        continue
                    row = acts.setdefault(int(aid), {})
                    row[idx] = row.get(idx, 0) + 1
    return acts, max_step, max(1, max_step - floor + 1)


def _var_of_subset(acts: dict, ids) -> float:
    sub = {aid: acts[aid] for aid in ids}
    return RG.act_dispersion(sub)[0]


def subsample_band(acts: dict, sizes, reps: int, seed: int = 0) -> list:
    """同一母集団から m 体を無作為抽出したときの分散の**帰無帯**(平均と 5/95 分位)。

    母集団が同じなので不偏推定量の期待値は m に依存しない(平坦)。幅だけが縮む。
    乱数は numpy.random.default_rng(seed) 固定 = 決定論。
    """
    ids = sorted(acts)
    n = len(ids)
    rng = np.random.default_rng(seed)
    out = []
    for m in sizes:
        m = int(m)
        if m < 2 or m > n:
            continue
        vals = []
        for _ in range(int(reps)):
            pick = rng.choice(n, size=m, replace=False)
            vals.append(_var_of_subset(acts, [ids[int(i)] for i in pick]))
        vals.sort()
        lo_i = max(0, int(0.05 * (len(vals) - 1)))
        hi_i = min(len(vals) - 1, int(math.ceil(0.95 * (len(vals) - 1))))
        out.append({
            "m": m,
            "mean": round(float(sum(vals) / len(vals)), 9),
            "p05": round(float(vals[lo_i]), 9),
            "p95": round(float(vals[hi_i]), 9),
            "reps": int(reps),
        })
    return out


def loglog_slope(points) -> dict:
    """log10 V = a + b·log10 N の最小二乗(b が傾き)。点が 2 未満なら None。"""
    xs = [math.log10(p["n"]) for p in points if p["n"] > 0 and p["var"] > 0]
    ys = [math.log10(p["var"]) for p in points if p["n"] > 0 and p["var"] > 0]
    if len(xs) < 2:
        return {"b": None, "a": None, "n_points": len(xs)}
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return {"b": None, "a": None, "n_points": len(xs)}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return {"b": round(b, 6), "a": round(my - b * mx, 6), "n_points": len(xs)}


def variance_collapse(run_dirs, window_steps: int, reps: int,
                      sizes_arg) -> dict:
    """N の違うランの実測曲線 + 最大 N ランの部分抽出帯 → 崩壊の切り分け。"""
    points = []
    for rd in run_dirs:
        rd = Path(rd)
        win = window_steps if window_steps > 0 else run_window_steps(rd)
        acts, max_step, span = act_counts_from_l1(rd, win)
        var, ent, n = RG.act_dispersion(acts)
        points.append({"run": rd.name, "run_dir": str(rd), "n": int(n),
                       "var": round(var, 9), "entropy_mean": round(ent, 6),
                       "window_steps": int(win), "last_step": int(max_step),
                       "span_steps": int(span)})
    points.sort(key=lambda p: p["n"])
    res = {"points": points, "fit": loglog_slope(points), "band": [],
           "band_from": None, "reps": int(reps)}
    if not points:
        res["verdict"] = "NO_DATA"
        res["why"] = "L1 が読めるランが 1 つも無い。"
        return res
    big = max(points, key=lambda p: p["n"])
    res["band_from"] = big["run"]
    acts, _ms, _sp = act_counts_from_l1(Path(big["run_dir"]),
                                        big["window_steps"])
    sizes = ([int(s) for s in sizes_arg] if sizes_arg
             else sorted({p["n"] for p in points if 2 <= p["n"] <= big["n"]}))
    res["band"] = subsample_band(acts, sizes, reps)
    # ★帯の向きの読み方(ここが診断の心臓部)
    #   帯は **最大 N のラン**の個体集合から m 体を無作為抽出したときの分散の分布である。
    #   帰無仮説は「V は N に依存しない」なので:
    #     ・小 N の実測が帯の **上** … 小さい集団のほうが多様だった = 大きい集団のほうが
    #       抽出ノイズで説明できないほど均質 = **モデル由来の均質化**(§3 が探しているもの)
    #     ・小 N の実測が帯の **下** … N が大きいほど多様になった = 崩壊ではない。
    #       通常は「N ごとに母集団構成が違う」(名簿の巡回複製・presence 層の切れ方など)。
    band_by_m = {b["m"]: b for b in res["band"]}
    above, below = [], []
    for p in points:
        b = band_by_m.get(p["n"])
        if b is None:
            continue
        p["band_p05"] = b["p05"]
        p["band_p95"] = b["p95"]
        p["inside_band"] = bool(b["p05"] <= p["var"] <= b["p95"])
        if p["var"] > b["p95"]:
            above.append(p["run"])
        elif p["var"] < b["p05"]:
            below.append(p["run"])
    res["above_band"] = above
    res["below_band"] = below
    fit_b = res["fit"]["b"]
    if len(points) < 2:
        res["verdict"] = "SINGLE_RUN"
        res["why"] = ("ランが 1 本しかない。N 依存は測れない"
                      "(部分抽出帯だけを帰無分布として出してある)。")
    elif above:
        res["verdict"] = "MODEL_HOMOGENIZATION_SUSPECTED"
        res["why"] = (f"小さい N の実測分散が、最大 N ラン(`{res['band_from']}`)から"
                      f"同数を抽出した帯の**上**に出た: {', '.join(above)}。"
                      f"= N が大きいほど集団が均質になっており、統計的平滑化"
                      f"(推定ノイズ)では説明できない。log-log 傾き b={fit_b}。")
    elif below:
        res["verdict"] = "POPULATION_MISMATCH_SUSPECTED"
        res["why"] = (f"小さい N の実測分散が帯の**下**に出た: {', '.join(below)}。"
                      f"これは分散**崩壊**の向きではない(N が大きいほど多様)。"
                      f"N 水準ごとに母集団構成が違う疑いが強い"
                      f"(平坦名簿の巡回複製・presence 層の切れ方など)。"
                      f"人口構成を揃えてから再判定すること。log-log 傾き b={fit_b}。")
    elif fit_b is not None and fit_b <= -0.25:
        res["verdict"] = "AMBIGUOUS"
        res["why"] = (f"帯の中には収まっているが log-log 傾き b={fit_b} が "
                      f"0 から離れている(1/√N の目安は −0.5)。"
                      f"N 水準を増やして再判定すること。")
    else:
        res["verdict"] = "NO_COLLAPSE"
        res["why"] = (f"実測分散は同一母集団の部分抽出帯の中にあり、log-log 傾き "
                      f"b={fit_b} も 0 近傍。N 依存の均質化は検出されない。")
    return res


# --------------------------------------------------------------------------- #
# SVG(素の文字列。matplotlib 非依存 = make_endo_report.py と同じ流儀)
# --------------------------------------------------------------------------- #
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def variance_svg(vc: dict, width: int = 720, height: int = 420) -> str:
    """log10 N × log10 V の散布図 + 部分抽出帯 + 参照傾き(b=0 / b=−0.5)。"""
    pts = [p for p in vc.get("points", []) if p["n"] > 0 and p["var"] > 0]
    band = [b for b in vc.get("band", []) if b["m"] > 0 and b["p95"] > 0]
    if not pts:
        return ('<svg width="200" height="40"><text x="4" y="24" '
                'font-size="12">データなし</text></svg>')
    xs = [math.log10(p["n"]) for p in pts] + [math.log10(b["m"]) for b in band]
    ys = ([math.log10(p["var"]) for p in pts]
          + [math.log10(max(b["p05"], 1e-12)) for b in band]
          + [math.log10(b["p95"]) for b in band])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    padx, pady = 0.08 * (x1 - x0), 0.15 * (y1 - y0)
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    L, R, T, B = 66, 16, 34, 46
    def px(x): return L + (x - x0) / (x1 - x0) * (width - L - R)
    def py(y): return T + (1.0 - (y - y0) / (y1 - y0)) * (height - T - B)

    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
          f'height="{height}" viewBox="0 0 {width} {height}" '
          f'font-family="sans-serif" font-size="11">',
          f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
          f'<text x="{L}" y="20" font-size="13" font-weight="bold">'
          f'分散崩壊診断: 個体間分散 V(N)(両対数)</text>']
    # 枠と軸ラベル
    el.append(f'<rect x="{L}" y="{T}" width="{width-L-R}" height="{height-T-B}" '
              f'fill="none" stroke="#888"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = x0 + frac * (x1 - x0)
        gy = y0 + frac * (y1 - y0)
        el.append(f'<line x1="{px(gx):.1f}" y1="{T}" x2="{px(gx):.1f}" '
                  f'y2="{height-B}" stroke="#eee"/>')
        el.append(f'<line x1="{L}" y1="{py(gy):.1f}" x2="{width-R}" '
                  f'y2="{py(gy):.1f}" stroke="#eee"/>')
        el.append(f'<text x="{px(gx):.1f}" y="{height-B+14}" '
                  f'text-anchor="middle" fill="#444">{10**gx:.0f}</text>')
        el.append(f'<text x="{L-6}" y="{py(gy)+4:.1f}" text-anchor="end" '
                  f'fill="#444">{10**gy:.2e}</text>')
    el.append(f'<text x="{(L+width-R)/2:.0f}" y="{height-8}" '
              f'text-anchor="middle" fill="#444">N(個体数)</text>')
    el.append(f'<text x="14" y="{(T+height-B)/2:.0f}" fill="#444" '
              f'transform="rotate(-90 14 {(T+height-B)/2:.0f})" '
              f'text-anchor="middle">V = 個体間分散(不偏)</text>')
    # 帰無帯(同一母集団からの部分抽出)
    if len(band) >= 2:
        top = " ".join(f"{px(math.log10(b['m'])):.1f},"
                       f"{py(math.log10(b['p95'])):.1f}" for b in band)
        bot = " ".join(f"{px(math.log10(b['m'])):.1f},"
                       f"{py(math.log10(max(b['p05'], 1e-12))):.1f}"
                       for b in reversed(band))
        el.append(f'<polygon points="{top} {bot}" fill="#9ecae1" '
                  f'fill-opacity="0.45" stroke="none"><title>'
                  f'同一母集団({_esc(vc.get("band_from"))})からの部分抽出 '
                  f'5–95%</title></polygon>')
        mid = " ".join(f"{px(math.log10(b['m'])):.1f},"
                       f"{py(math.log10(max(b['mean'], 1e-12))):.1f}"
                       for b in band)
        el.append(f'<polyline points="{mid}" fill="none" stroke="#3182bd" '
                  f'stroke-width="1.5" stroke-dasharray="5 3"/>')
    # 参照傾き(最大 N の実測点を通す)
    anchor = max(pts, key=lambda p: p["n"])
    ax, ay = math.log10(anchor["n"]), math.log10(anchor["var"])
    for b_ref, color, dash, label in ((0.0, "#31a354", "1 0", "b=0(崩壊なし)"),
                                      (-0.5, "#e6550d", "2 3", "b=-0.5(1/√N)")):
        p1 = (x0, ay + b_ref * (x0 - ax))
        p2 = (x1, ay + b_ref * (x1 - ax))
        el.append(f'<line x1="{px(p1[0]):.1f}" y1="{py(p1[1]):.1f}" '
                  f'x2="{px(p2[0]):.1f}" y2="{py(p2[1]):.1f}" stroke="{color}" '
                  f'stroke-width="1" stroke-dasharray="{dash}"><title>'
                  f'{_esc(label)}</title></line>')
    # 実測点
    for p in pts:
        cx, cy = px(math.log10(p["n"])), py(math.log10(p["var"]))
        fill = "#e6550d" if p.get("inside_band") is False else "#111"
        el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{fill}">'
                  f'<title>{_esc(p["run"])}: N={p["n"]} V={p["var"]:.3e}</title>'
                  f'</circle>')
        el.append(f'<text x="{cx+7:.1f}" y="{cy-6:.1f}" fill="#111">'
                  f'{_esc(p["run"])}</text>')
    # 凡例(色だけに頼らない=形も変える。make_endo_report と同じ配慮)
    ly = T + 14
    for color, dash, label in (("#3182bd", "5 3", "部分抽出の中央(帰無=平坦)"),
                               ("#31a354", "1 0", "b=0(崩壊なし)"),
                               ("#e6550d", "2 3", "b=-0.5(1/√N)")):
        el.append(f'<line x1="{width-R-190}" y1="{ly}" x2="{width-R-160}" '
                  f'y2="{ly}" stroke="{color}" stroke-width="1.5" '
                  f'stroke-dasharray="{dash}"/>')
        el.append(f'<text x="{width-R-155}" y="{ly+4}" fill="#333">'
                  f'{_esc(label)}</text>')
        ly += 15
    el.append("</svg>")
    return "".join(el)


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
_VERDICT_MARK = {"OK": "✅", "REGRESSION": "🚨", "INSUFFICIENT": "—",
                 "CONTEXT_ONLY": "·", "NO_COLUMNS": "—"}


def render(res: dict) -> str:
    tr = res.get("trend")
    L = ["# 退行シグナル判定レポート", "",
         f"- 対象: `{res.get('run_dir', '-')}`",
         f"- 閾値(引数必須): `--alpha {res['thresholds']['alpha']}` / "
         f"`--min-rel-slope {res['thresholds']['min_rel_slope']}`",
         f"- 正典: docs/plans/source/design-discussion-20260802.md §3 / "
         f"判定基準: docs/research/regression-signals.md", ""]
    if tr:
        L += ["## §A 時系列トレンド検定(Mann-Kendall + Theil-Sen)", "",
              f"- L2 の読み元: {tr['l2']['source']}"
              + (f"(完結 part {tr['l2']['parts']} 枚"
                 + (f"・書きかけスキップ {len(tr['l2']['skipped_parts'])} 枚)"
                    if tr['l2']['skipped_parts'] else ")")
                 if tr['l2']['source'] == 'parts' else ""),
              f"- L2 行数 {tr['n_rows']} / 監視窓 {tr['window_steps']} step / "
              f"間引き幅 {tr['stride']} step({tr['stride_source']}) / "
              f"ウォームアップ棄却 {tr['warmup']} step({tr['warmup_source']})", ""]
        if tr["stride_below_window"]:
            L += ["> ⚠ **間引き幅が監視窓より狭い**。隣接標本が窓を共有するので "
                  "p 値は過大に有意側へ寄る。有意性を語らないこと。", ""]
        L += ["| シグナル | 退行方向 | n | 初 | 終 | 相対傾き/窓 | z | p | 判定 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for col in tr["columns_present"]:
            r = tr["signals"][col]
            d = {1: "↑", -1: "↓", None: "—"}[r["direction_of_regression"]]
            L.append(f"| `{col}` | {d} | {r['n_samples']} | {r['first']} | "
                     f"{r['last']} | {r['rel_slope']} | {r['z']} | {r['p']} | "
                     f"{_VERDICT_MARK.get(r['verdict'], '')} {r['verdict']} |")
        L += ["", f"### **{tr['verdict']}**", "", tr["why"], ""]
        if tr["columns_missing"]:
            L += [f"- 不在の列: {', '.join('`' + c + '`' for c in tr['columns_missing'])}"
                  " (`cognition.fire` OFF のランでは発火率 5 列が構造上出ない)", ""]
    vc = res.get("variance_collapse")
    if vc:
        L += ["## §B 分散崩壊診断(統計的平滑化 vs モデル由来の均質化)", "",
              "個体間分散の**不偏推定量**は iid 標本なら期待値が N に依存しない。"
              "帯は**最大 N のラン**から同数を無作為抽出したときの分布(= 帰無)なので、"
              "**小さい N の実測が帯の上に出たとき**だけ「N が大きいほど均質」"
              "= モデル由来の均質化と読む。帯の下に出たのは崩壊の向きではなく、"
              "たいてい母集団構成の不一致である。", "",
              "| ラン | N | V(個体間分散) | 帯 5% | 帯 95% | 帯の中 |",
              "|---|---|---|---|---|---|"]
        for p in vc["points"]:
            L.append(f"| `{p['run']}` | {p['n']} | {p['var']:.6e} | "
                     f"{p.get('band_p05', '—')} | {p.get('band_p95', '—')} | "
                     f"{'✅' if p.get('inside_band') else ('🚨' if 'inside_band' in p else '—')} |")
        fit = vc["fit"]
        L += ["", f"- log-log 傾き **b = {fit['b']}**"
                  f"(参照: b=0 崩壊なし / b=−0.5 が 1/√N の形。点数 {fit['n_points']})",
              f"- 帰無帯の出所: `{vc['band_from']}` から {vc['reps']} 回の無作為抽出"
              f"(numpy default_rng(0) 固定 = 決定論)",
              "", f"![分散崩壊診断](regression_variance.svg)", "",
              f"### **{vc['verdict']}**", "", vc["why"], ""]
    L += ["## §C 正直な限界", "",
          "1. L2 の監視列は rolling 窓なので**隣接行は独立ではない**。既定では窓幅ぶん"
          "間引いて非重複標本にしているが、それでも完全な独立ではない(窓の境界で"
          "同じイベントが両側に入りうる)。",
          "2. Mann-Kendall は**単調**傾向だけを見る。「途中で一度崩れて戻った」型の"
          "退行は検出しない。",
          "3. 母集団は「窓内に L1 イベントを出した個体」。睡眠中で無イベントの個体は"
          "分母から落ちる(observer/regression.py の限界 1 と同じ)。",
          "4. 語彙は文字 n-gram であって語ではない。第87 の定型応答は除外済みだが、"
          "**他の機構由来の固定文があれば同じ汚染が起きる**(`exclude_template` は"
          "engaged の TEMPLATE 1 種のみ)。",
          "5. 分散崩壊診断の帰無帯は「**そのラン内の**個体は交換可能」という仮定に"
          "立つ。ペルソナ層(L1/L2/L3…)が N によって変わるランでは、帯の外れが"
          "モデル由来ではなく**母集団構成の差**でも起きる。N を変えるときは"
          "人口構成を揃えること。", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="退行シグナルの判定(第91バッチ。設計 §3)。閾値は引数必須。")
    ap.add_argument("run_dirs", nargs="*", help="ランのディレクトリ(1 つ以上)")
    # ---- 判定閾値(**必須**。既定値をコードに埋め込まない)----
    ap.add_argument("--alpha", type=float, required=True,
                    help="[必須] Mann-Kendall の有意水準")
    ap.add_argument("--min-rel-slope", type=float, required=True,
                    help="[必須] 相対傾き(Theil-Sen ÷ 平均)の下限=効果量")
    # ---- 計算パラメータ(既定あり。値は出力に必ず記録する)----
    ap.add_argument("--stride", type=int, default=0,
                    help="間引き幅 step(既定 0=ランの監視窓幅=非重複標本)")
    ap.add_argument("--warmup", type=int, default=-1,
                    help="先頭で捨てる step 数(既定 -1=監視窓幅。窓が満ちるまでの"
                         "立ち上がりを検定に混ぜないため)")
    ap.add_argument("--quick", action="store_true",
                    help="1 行 JSON だけ出す(watchdog から呼ぶ用。ファイルを書かない)")
    ap.add_argument("--variance-collapse", action="store_true",
                    help="分散崩壊診断も行う(run_dirs を N の水準として使う)")
    ap.add_argument("--var-window", type=int, default=0,
                    help="分散診断で使う窓幅 step(既定 0=各ランの監視窓幅)")
    ap.add_argument("--var-reps", type=int, default=200,
                    help="部分抽出の反復回数(既定 200・決定論 seed=0)")
    ap.add_argument("--var-sizes", nargs="*", default=None,
                    help="部分抽出のサイズ列(既定=実測ランの N をそのまま使う)")
    ap.add_argument("--out", default=None, help="出力先(既定: 先頭ラン)")
    args = ap.parse_args(argv)

    dirs = [Path(d) for d in args.run_dirs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if not dirs or missing:
        raise SystemExit(f"ランが見つからない: {missing or args.run_dirs}")

    trend = trend_report(dirs[0], args.alpha, args.min_rel_slope, args.stride,
                         args.warmup)

    if args.quick:
        # watchdog から呼ぶ経路。**ファイルを 1 つも書かず・絶対に非ゼロ終了しない**
        # (監視がランを殺してはならない。P0 の llm_health 監視と同じ方針)。
        out = {"run": trend["run"], "verdict": trend["verdict"],
               "flagged": trend["flagged"], "n_rows": trend["n_rows"],
               "stride": trend["stride"],
               "signals": {k: {"p": v["p"], "rel_slope": v["rel_slope"],
                               "verdict": v["verdict"]}
                           for k, v in trend["signals"].items()}}
        print(json.dumps(out, ensure_ascii=False, sort_keys=True))
        if trend["verdict"] == "REGRESSION":
            print(f"[regression-warn] {trend['why']}", file=sys.stderr)
        return 0

    res = {
        "schema": 1,
        "run_dir": str(dirs[0]),
        "thresholds": {"alpha": args.alpha, "min_rel_slope": args.min_rel_slope},
        "params": {"stride": args.stride, "var_window": args.var_window,
                   "var_reps": args.var_reps, "var_sizes": args.var_sizes},
        "trend": trend,
        "variance_collapse": (variance_collapse(dirs, args.var_window,
                                                args.var_reps, args.var_sizes)
                              if args.variance_collapse else None),
    }
    out_dir = Path(args.out or dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regression_report.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    if res["variance_collapse"]:
        (out_dir / "regression_variance.svg").write_text(
            variance_svg(res["variance_collapse"]), encoding="utf-8")
    md = render(res)
    (out_dir / "regression_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[written] {out_dir / 'regression_report.json'}")
    print(f"[written] {out_dir / 'regression_report.md'}")
    if res["variance_collapse"]:
        print(f"[written] {out_dir / 'regression_variance.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
