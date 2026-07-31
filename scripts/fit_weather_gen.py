#!/usr/bin/env python3
"""確率的天候生成器(stochastic weather generator)の較正(天候の生成型実データ化 前半)。

入力 = `data/snapshot/weather_tokyo_aug.json`(scripts/fetch_weather_history.py が凍結した実測)。
出力 = `data/snapshot/weather_gen_params.json`(生成器パラメータ + 実測要約 + 自己検証結果)。

**決定論**: 同じ入力・同じ引数なら**バイト同一**の JSON を出す(既定では壁時計時刻を書かない。
provenance は入力の payload_sha256 で取る。`--stamp` を付けたときだけ時刻を書く=非決定論になる)。

--------------------------------------------------------------------------- モデル
WGEN 系(Richardson 1981 / Richardson & Wright 1984)の標準構成に、低周波(年々)変動の
補正(Chen, Brissette & Leconte 2010)を足したもの。詳細と出典は
docs/research/weather-generator-design.md §1「リサーチ結果」。

  (1) 天気カテゴリ: **3状態1次マルコフ連鎖** {晴, 曇, 雨}(+雪は8月は確率0で保持)。
      WGEN 原型の「2状態(乾/湿)」も P(wet|wet)/P(wet|dry) として併記・出力する
      (現行 weather.py が 雨/曇/晴 の3分割なので、生成器側は3状態を本体に使う)。
  (2) 気温: 4段の分解 —
        T(d) = μ_state(d) + β_month·(d − d_ref) + Y + σ_state(d)·χ(d)
        μ_state / σ_state : 天気状態(晴/曇/雨)別の平均・標準偏差(=WGEN の wet/dry 条件付け)
        β_month           : 月内の線形トレンド(8月は下旬ほど涼しい。実測 −0.083℃/日)
        Y                 : **その年(=そのラン)ごとに1回だけ引く年効果**(hi/lo 相関つき2変量正規)
        χ(d)              : 標準化残差の1次自己回帰(Matalas 1967 の2変量形)
                            χ(d) = A·χ(d−1) + B·ε(d),  A = M1·M0⁻¹,  B·Bᵀ = M0 − M1·M0⁻¹·M1ᵀ
                            (M0 = lag-0 相関行列, M1 = lag-1 交差相関行列, ε ~ N(0, I))
  (3) 降水量: 湿日(≥ 閾値)の量をガンマ分布(モーメント法)で。
  (4) 湿度: 状態別平均 + 気温偏差への線形回帰 + 残差ノイズ(WBGT 推定に要る)。
  (5) WBGT: 小野ら(2014)の推定式の係数を保持(環境省 熱中症予防情報サイト掲載式)。

**なぜ連続猛暑(persistence)が現行モデルで出ないか / 何が要るか**:
  現行 `src/society/weather.py:68-69` は `hi_c + rng.integers(-3, 4)` の**独立同分布**。
  日間の自己相関はゼロ・年々変動もゼロなので、
    ・35℃の出現は 1/7 = 14.29% で固定、**36℃以上は確率0**(上限が 32+3 = 35)
    ・猛暑の連続は「独立試行がたまたま続く」確率(0.1429^n)しか出ない
  実測(東京・8月・2015-2025)は lag-1 自己相関 hi=0.61 / lo=0.68、年平均の年々SD ≈ 1.4℃。
  → 連続猛暑を出すのに要るのは **(a) 日間 AR(1) と (b) 年効果(低周波成分)の2つ**。
    (a) だけでは「暑い年/涼しい年」が消えて長い連続が不足する(Chen et al. 2010 の指摘)。

--------------------------------------------------------------------------- 使い方
  # 既定(移転後 2015-2025 の窓で較正 + モンテカルロ自己検証)
  python scripts/fit_weather_gen.py
  # 全期間で較正(感度分析)
  python scripts/fit_weather_gen.py --fit-years 1996-2025 --out /tmp/params_all.json
  # 検証だけ(既存 params の再現性チェック)
  python scripts/fit_weather_gen.py --verify --out data/snapshot/weather_gen_params.json

--------------------------------------------------------------------------- 限界(正直に)
  - 観測点は「東京」(北の丸公園)。**渋谷の値ではない**。都市キャノピー補正は入れていない。
  - 既定窓 2015-2025 は **11年 = 341日**。年効果 SD は 11 標本からの推定=不確かさが大きい。
  - 2015-2025 の中にも強い上昇トレンドがある(2015年 猛暑日8日 → 2025年 18日)。
    生成器はこれを「年効果の分散」として吸収する=**2026年の予報ではない**。
  - WBGT は**実測ではなく近似**。全天日射量は日照時間からの代理、日最高 WBGT の代理として
    Ta=日最高気温・RH=日最小湿度を入れている(§WBGT_NOTE)。環境省の実測 WBGT(東京 44132)
    との突合は本バッチでは未実施。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
DEFAULT_IN = REPO_ROOT / "data" / "snapshot" / "weather_tokyo_aug.json"
DEFAULT_OUT = REPO_ROOT / "data" / "snapshot" / "weather_gen_params.json"

# ★第80バッチ(W2)で **生成器の本体は src/society/weather_gen.py へ移した**。
#   実装の二重化(同じ AR(1) を fit 側と sim 側に 2 本書く)を避けるため、本スクリプトは
#   src の実装を import して使う=「fit が出す系列」と「シムが出す系列」は原理的に同一。
#   本スクリプトの公開名(STATES / RAIN / wbgt_ono2014 / label_state / payload_sha256 …)は
#   後方互換のためそのまま残す(既存テストは無改修で緑)。
sys.path.insert(0, str(REPO_ROOT / "src"))
from society import weather_gen as WG                          # noqa: E402

# 天気状態(現行 weather.py の cond 4種と同じ語。雪は8月は確率0で保持)。
STATES = list(WG.STATES)
ACTIVE_STATES = ["晴", "曇", "雨"]          # 8月に実際に出る3状態
RAIN = WG.RAIN

# 小野ら(2014)「通常観測気象要素を用いた WBGT の推定」日生気誌 50(4), 147-157。
# 環境省 熱中症予防情報サイト(https://www.wbgt.env.go.jp/wbgt_detail.php)掲載の推定式:
#   WBGT = 0.735*Ta + 0.0374*RH + 0.00292*Ta*RH + 7.619*SR − 4.557*SR² − 0.0572*WS − 4.064
#   Ta[℃] 気温 / RH[%] 相対湿度 / SR[kW/m²] 全天日射量 / WS[m/s] 平均風速
WBGT_ONO2014 = WG.WBGT_ONO2014
# 実測 WBGT の定義式(参考。本スクリプトでは使わない=湿球/黒球温度を持っていない)。
WBGT_MEASURED = WG.WBGT_MEASURED
# 日射量の代理に使う仮定(★仮定であることを必ず添えて出す)。
SR_CLEAR_KW = WG.SR_CLEAR_KW      # 快晴日の日中平均 全天日射量[kW/m²](東京・8月の目安)
DAYLENGTH_H = WG.DAYLENGTH_H      # 8月中旬の東京の可照時間[h]の目安

wbgt_ono2014 = WG.wbgt_ono2014
sr_proxy = WG.sr_proxy
label_state = WG.label_state
payload_sha256 = WG.payload_sha256


# ------------------------------------------------------------------ 実測の読み込みとラベル付け
def load_history(path: Path, years: tuple[int, int], month: int, wet_mm: float):
    doc = json.loads(path.read_text(encoding="utf-8"))
    days = [d for d in doc["days"]
            if d["month"] == month and years[0] <= d["year"] <= years[1]]
    days.sort(key=lambda d: d["date"])
    for d in days:
        d["_state"] = label_state(d, wet_mm)
    return doc, days


# ------------------------------------------------------------------ 統計ユーティリティ(numpy のみ)
def _r6(x):
    """出力の丸め(バイト同一性のため全数値をここに通す)。"""
    if isinstance(x, (list, tuple)):
        return [_r6(v) for v in x]
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, 6)


def ols_slope(x, y):
    """単回帰の傾きと切片(numpy.polyfit と同値・欠測なし前提)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0, float(ym)
    slope = float(((x - xm) * (y - ym)).sum() / denom)
    return slope, float(ym - slope * xm)


def year_effect_variance(groups: list[np.ndarray]) -> tuple[float, float]:
    """一元配置ランダム効果モデルで (年効果の分散, 年内分散) を分離する。

    σ²_year = (MSB − MSW) / n̄ ,  σ²_within = MSW  (標本数が群間で等しい古典形)。
    負になったら 0 に切る(標本が小さいとき起きる=正直に 0 とする)。
    """
    k = len(groups)
    ns = np.array([len(g) for g in groups], float)
    grand = float(np.concatenate(groups).mean())
    ssb = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    dfb, dfw = k - 1, int(ns.sum()) - k
    if dfb <= 0 or dfw <= 0:
        return 0.0, float(np.concatenate(groups).var(ddof=1))
    msb, msw = ssb / dfb, ssw / dfw
    n_bar = float(ns.mean())
    return max(0.0, (msb - msw) / n_bar), msw


def lag_matrices(res: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """標準化残差(年ごとに区切ったリスト・各要素は shape (n_days, 2))から M0, M1 を作る。

    M0[i][j] = corr(χ_i(t), χ_j(t)),  M1[i][j] = corr(χ_i(t), χ_j(t−1))。
    年をまたぐペアは使わない(8/31 → 翌年8/1 は連続していない)。
    """
    cur = np.concatenate([r[1:] for r in res], axis=0)
    prev = np.concatenate([r[:-1] for r in res], axis=0)
    allv = np.concatenate(res, axis=0)
    m0 = np.corrcoef(allv.T)
    m1 = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            m1[i, j] = np.corrcoef(cur[:, i], prev[:, j])[0, 1]
    return m0, m1


def matalas_ab(m0: np.ndarray, m1: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Matalas(1967)の2変量 AR(1): A = M1 M0⁻¹, B Bᵀ = M0 − A M1ᵀ, B = chol(BBᵀ)。"""
    a = m1 @ np.linalg.inv(m0)
    bbt = m0 - a @ m1.T
    bbt = 0.5 * (bbt + bbt.T)            # 数値誤差の非対称を消す
    note = "cholesky"
    try:
        b = np.linalg.cholesky(bbt)
    except np.linalg.LinAlgError:        # 非正定値なら固有値を切り上げて対称平方根へ
        w, v = np.linalg.eigh(bbt)
        w = np.clip(w, 1e-12, None)
        b = v @ np.diag(np.sqrt(w)) @ v.T
        note = "eigh_clipped(非正定値のため固有値を1e-12で下切り)"
    return a, b, note


def gamma_mom(x: np.ndarray) -> tuple[float, float]:
    """ガンマ分布のモーメント法。shape k = m²/v, scale θ = v/m。"""
    m = float(x.mean())
    v = float(x.var(ddof=1)) if len(x) > 1 else 0.0
    if m <= 0 or v <= 0:
        return 1.0, max(m, 1e-6)
    return m * m / v, v / m


def run_lengths(mask: np.ndarray) -> list[int]:
    """True の連の長さの一覧(mask は 1次元 bool)。"""
    out, run = [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def spell_stats(hi: np.ndarray, seg_len: int, thr: float) -> dict:
    """猛暑連長の要約。hi は (n_seg*seg_len,) で年ごとに区切られている前提。"""
    segs = hi.reshape(-1, seg_len)
    lens: list[int] = []
    for s in segs:
        lens.extend(run_lengths(s >= thr))
    lens_arr = np.array(lens, float) if lens else np.array([0.0])
    per_year_max = np.array([max(run_lengths(s >= thr) or [0]) for s in segs], float)
    return {
        "threshold_c": _r6(thr),
        "n_spells": len(lens),
        "spells_per_year": _r6(len(lens) / len(segs)),
        "mean_len": _r6(lens_arr.mean()) if lens else 0.0,
        "max_len": int(lens_arr.max()) if lens else 0,
        "share_len_ge_3": _r6(float((lens_arr >= 3).mean()) if lens else 0.0),
        "share_len_ge_5": _r6(float((lens_arr >= 5).mean()) if lens else 0.0),
        "share_len_ge_7": _r6(float((lens_arr >= 7).mean()) if lens else 0.0),
        "year_max_mean": _r6(per_year_max.mean()),
        "p_year_has_len_ge_5": _r6(float((per_year_max >= 5).mean())),
        "p_year_has_len_ge_10": _r6(float((per_year_max >= 10).mean())),
    }


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """2標本 Kolmogorov-Smirnov 統計量 D と漸近 p 値(scipy 非依存)。"""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, allv, side="right") / n1
    cdf2 = np.searchsorted(b, allv, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    ne = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (ne + 0.12 + 0.11 / ne) * d
    if lam < 1e-8:                       # D=0(完全一致)は級数が発散するので明示的に返す
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


# ------------------------------------------------------------------ 較正
def fit_month(days: list[dict], month: int, wet_mm: float, hot_c: float) -> dict:
    years = sorted({d["year"] for d in days})
    hi = np.array([d["temp_hi"] for d in days], float)
    lo = np.array([d["temp_lo"] for d in days], float)
    dom = np.array([d["day"] for d in days], float)
    state = np.array([d["_state"] for d in days], int)
    yr = np.array([d["year"] for d in days], int)
    precip = np.array([(d["precip_mm"] or 0.0) for d in days], float)
    rh = np.array([(d["humid_mean"] if d["humid_mean"] is not None else np.nan)
                   for d in days], float)
    rh_min = np.array([(d["humid_min"] if d["humid_min"] is not None else np.nan)
                       for d in days], float)
    sun = np.array([(d["sunshine_h"] or 0.0) for d in days], float)
    wind = np.array([(d["wind_mean"] if d["wind_mean"] is not None else np.nan)
                     for d in days], float)
    d_ref = 16.0                                  # 月内トレンドの基準日(8/16 = 本選ランの開始日)

    # --- (a) 月内線形トレンド ---------------------------------------------------
    b_hi, _ = ols_slope(dom - d_ref, hi)
    b_lo, _ = ols_slope(dom - d_ref, lo)
    hi_dt = hi - b_hi * (dom - d_ref)
    lo_dt = lo - b_lo * (dom - d_ref)

    # --- (b) 年効果(低周波成分)の分離 -----------------------------------------
    g_hi = [hi_dt[yr == y] for y in years]
    g_lo = [lo_dt[yr == y] for y in years]
    var_y_hi, var_w_hi = year_effect_variance(g_hi)
    var_y_lo, var_w_lo = year_effect_variance(g_lo)
    ym_hi = np.array([g.mean() for g in g_hi])
    ym_lo = np.array([g.mean() for g in g_lo])
    corr_year = float(np.corrcoef(ym_hi, ym_lo)[0, 1]) if len(years) > 2 else 0.0
    # 年平均を除いた系列(以降の条件付き平均/分散・AR(1) はこの上で測る)
    hi_a = np.concatenate([g - g.mean() for g in g_hi])
    lo_a = np.concatenate([g - g.mean() for g in g_lo])
    order = np.concatenate([np.where(yr == y)[0] for y in years])
    hi_anom = np.empty_like(hi_a)
    lo_anom = np.empty_like(lo_a)
    hi_anom[order] = hi_a
    lo_anom[order] = lo_a
    grand_hi = float(hi_dt.mean())
    grand_lo = float(lo_dt.mean())

    # --- (c) 天気状態の1次マルコフ連鎖(3状態)+ 2状態(乾/湿)の併記 -----------
    n_state = len(STATES)
    trans = np.zeros((n_state, n_state))
    for y in years:
        s = state[yr == y]
        for t in range(1, len(s)):
            trans[s[t - 1], s[t]] += 1
    counts = trans.copy()
    marg = np.array([float((state == k).mean()) for k in range(n_state)])
    with np.errstate(invalid="ignore", divide="ignore"):
        rowsum = trans.sum(axis=1, keepdims=True)
        prob = np.where(rowsum > 0, trans / np.where(rowsum == 0, 1, rowsum), 0.0)
    for k in range(n_state):                      # 出現しない状態(=雪)は周辺分布で埋める
        if counts[k].sum() == 0:
            prob[k] = marg
    wet = state == RAIN
    n_dd = n_dw = n_wd = n_ww = 0
    for y in years:
        w = wet[yr == y]
        for t in range(1, len(w)):
            if w[t - 1] and w[t]:
                n_ww += 1
            elif w[t - 1] and not w[t]:
                n_wd += 1
            elif (not w[t - 1]) and w[t]:
                n_dw += 1
            else:
                n_dd += 1
    p01 = n_dw / (n_dw + n_dd) if (n_dw + n_dd) else 0.0      # P(wet|dry)
    p11 = n_ww / (n_ww + n_wd) if (n_ww + n_wd) else 0.0      # P(wet|wet)
    pi_wet = p01 / (1.0 + p01 - p11) if (1.0 + p01 - p11) > 0 else float(wet.mean())

    # --- (d) 状態別の平均・SD(年効果を除いた偏差の上で) ------------------------
    per_state = {}
    for k, name in enumerate(STATES):
        m = state == k
        if m.sum() >= 3:
            per_state[name] = {
                "n": int(m.sum()),
                "mean_hi": _r6(grand_hi + hi_anom[m].mean()),
                "sd_hi": _r6(hi_anom[m].std(ddof=1)),
                "mean_lo": _r6(grand_lo + lo_anom[m].mean()),
                "sd_lo": _r6(lo_anom[m].std(ddof=1)),
            }
        else:                                     # 標本不足(=雪)は全体値で代用し明示
            per_state[name] = {
                "n": int(m.sum()),
                "mean_hi": _r6(grand_hi), "sd_hi": _r6(hi_anom.std(ddof=1)),
                "mean_lo": _r6(grand_lo), "sd_lo": _r6(lo_anom.std(ddof=1)),
                "note": "標本不足のため全状態プールの値で代用",
            }

    # --- (e) 標準化残差の AR(1)(Matalas 2変量)---------------------------------
    zh = np.empty_like(hi_anom)
    zl = np.empty_like(lo_anom)
    for k, name in enumerate(STATES):
        m = state == k
        if not m.any():
            continue
        ps = per_state[name]
        zh[m] = (grand_hi + hi_anom[m] - ps["mean_hi"]) / max(ps["sd_hi"], 1e-9)
        zl[m] = (grand_lo + lo_anom[m] - ps["mean_lo"]) / max(ps["sd_lo"], 1e-9)
    res_by_year = [np.stack([zh[yr == y], zl[yr == y]], axis=1) for y in years]
    m0, m1 = lag_matrices(res_by_year)
    a_mat, b_mat, chol_note = matalas_ab(m0, m1)

    # --- (f) 湿日の降水量(ガンマ)------------------------------------------------
    wet_amounts = precip[precip >= wet_mm]
    g_shape, g_scale = gamma_mom(wet_amounts) if len(wet_amounts) > 1 else (1.0, wet_mm)

    # --- (g) 湿度(WBGT 用)--------------------------------------------------------
    ok = ~np.isnan(rh)
    rh_state = {}
    for k, name in enumerate(STATES):
        m = ok & (state == k)
        if m.sum() >= 3:
            rh_state[name] = {"n": int(m.sum()), "mean": _r6(rh[m].mean()),
                              "sd": _r6(rh[m].std(ddof=1))}
        else:
            rh_state[name] = {"n": int(m.sum()), "mean": _r6(rh[ok].mean()),
                              "sd": _r6(rh[ok].std(ddof=1)),
                              "note": "標本不足のため全体値で代用"}
    rh_dem = rh[ok] - np.array([rh_state[STATES[k]]["mean"] for k in state[ok]])
    hi_dem = hi_anom[ok]
    beta_rh, _ = ols_slope(hi_dem, rh_dem)
    resid = rh_dem - beta_rh * hi_dem
    okm = ~np.isnan(rh_min)
    okw = ~np.isnan(wind)
    d_rh_min, _ = ols_slope(rh[okm & ok], rh_min[okm & ok])

    return {
        "n_days": len(days), "n_years": len(years),
        "years": [years[0], years[-1]],
        "d_ref": _r6(d_ref),
        "states": list(STATES),
        "markov3": {
            "transition": [[_r6(v) for v in row] for row in prob],
            "counts": [[int(v) for v in row] for row in counts],
            "marginal": [_r6(v) for v in marg],
            "note": "行 = 前日の状態, 列 = 当日の状態(順序は states)",
        },
        "markov2_wet_dry": {
            "wet_threshold_mm": _r6(wet_mm),
            "p_wet_given_dry": _r6(p01), "p_wet_given_wet": _r6(p11),
            "p_wet_stationary": _r6(pi_wet),
            "p_wet_empirical": _r6(float(wet.mean())),
            "counts": {"dd": n_dd, "dw": n_dw, "wd": n_wd, "ww": n_ww},
            "note": "WGEN 原型の2状態表現(生成器本体は markov3 を使う)",
        },
        "temp": {
            "grand_mean_hi": _r6(grand_hi), "grand_mean_lo": _r6(grand_lo),
            "month_trend_hi_per_day": _r6(b_hi), "month_trend_lo_per_day": _r6(b_lo),
            "per_state": per_state,
            "year_effect": {
                "sd_hi": _r6(math.sqrt(var_y_hi)), "sd_lo": _r6(math.sqrt(var_y_lo)),
                "corr": _r6(corr_year),
                "within_sd_hi": _r6(math.sqrt(var_w_hi)),
                "within_sd_lo": _r6(math.sqrt(var_w_lo)),
                "n_years": len(years),
                "note": ("一元配置ランダム効果 σ²_year=(MSB−MSW)/n̄。"
                         "ラン開始時に1回だけ引く(=そのランは暑い年/涼しい年のどちらかになる)。"
                         "標本 %d 年からの推定=不確かさ大。" % len(years)),
            },
            "ar1": {
                "M0": [[_r6(v) for v in row] for row in m0],
                "M1": [[_r6(v) for v in row] for row in m1],
                "A": [[_r6(v) for v in row] for row in a_mat],
                "B": [[_r6(v) for v in row] for row in b_mat],
                "lag1_hi": _r6(m1[0, 0]), "lag1_lo": _r6(m1[1, 1]),
                "corr_hi_lo": _r6(m0[0, 1]),
                "decomposition": chol_note,
                "note": "χ(d) = A·χ(d−1) + B·ε(d), ε ~ N(0, I) (Matalas 1967)",
            },
        },
        "precip": {
            "gamma_shape": _r6(g_shape), "gamma_scale": _r6(g_scale),
            "n_wet_days": int(len(wet_amounts)),
            "mean_wet_mm": _r6(float(wet_amounts.mean()) if len(wet_amounts) else 0.0),
            "max_observed_mm": _r6(float(precip.max())),
            "note": "モーメント法。極端豪雨の裾は再現しない(ガンマの既知の限界)",
        },
        "humidity": {
            "per_state": rh_state,
            "beta_on_temp_hi_anom": _r6(beta_rh),
            "resid_sd": _r6(float(resid.std(ddof=1))),
            "humid_min_from_mean_slope": _r6(d_rh_min),
            "humid_min_mean": _r6(float(rh_min[okm].mean())),
            "wind_mean_mean": _r6(float(wind[okw].mean())),
            "wind_mean_sd": _r6(float(wind[okw].std(ddof=1))),
            "note": "RH = mean_state + beta·(T_hi 年内偏差) + N(0, resid_sd)",
        },
        "_obs_arrays": {"hi": hi, "lo": lo, "state": state, "sun": sun,
                        "rh": rh, "rh_min": rh_min, "wind": wind, "precip": precip,
                        "yr": yr, "dom": dom},
    }


# ------------------------------------------------------------------ 生成(オフライン検証用)
def simulate(pm: dict, n_years: int, seg_len: int, seed: int,
             day_start: int = 1, year_effect: bool = True) -> dict:
    """較正パラメータから日別系列を生成する(決定論・RngHub と同じ PCG64/SeedSequence)。

    返り値の各配列は shape (n_years*seg_len,)。年ごとに区切られている。
    `year_effect=False` は「年効果を外した」比較用(低周波成分の寄与を測るため)。

    ★本体は `src/society/weather_gen.generate_segment`(シムが使うのと同じ実装)。
      ここは「1本の rng を年ごとに連続消費する」ループだけを持つ薄い殻。
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed)])))
    prep = WG.prepare(pm)
    segs = [WG.generate_segment(rng, prep, seg_len, day_start=day_start,
                                year_effect=year_effect) for _ in range(n_years)]
    n = n_years * seg_len
    n_clipped = sum(s["n_clipped"] for s in segs)
    cat = {k: np.concatenate([s[k] for s in segs])
           for k in ("state", "temp_hi", "temp_lo", "humid_mean", "precip_mm")}
    cat["share_clipped"] = n_clipped / max(1, n)
    return cat


def simulate_synthetic_current(n_years: int, seg_len: int, seed: int, month: int = 8) -> dict:
    """**現行 src/society/weather.py の合成モデルの鏡**(比較表のため)。

    weather.py:22-27 の 8月 = (32, 25, 0.30, 0.0) と weather.py:55-72 の _sample() を
    そのまま写したもの(2026-08-01 時点)。weather.py が変わったらここも見直すこと
    (tests/test_weather_gen_offline.py が src との一致を機械検査する)。
    """
    table = {8: (32, 25, 0.30, 0.0), 7: (30, 23, 0.42, 0.0)}
    hi_c, lo_c, rain_w, snow_w = table[month]
    n = n_years * seg_len
    hi = np.zeros(n)
    lo = np.zeros(n)
    st = np.zeros(n, int)
    for i in range(n):
        # weather.py は hub.stream("weather", day_index) から日ごとに独立ストリームを引く。
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), i])))
        r = float(rng.random())
        if r < snow_w:
            cond = 3
        elif r < snow_w + rain_w:
            cond = RAIN
        else:
            span = 1.0 - snow_w - rain_w
            rem = r - (snow_w + rain_w)
            cond = 1 if rem < span * 0.4 else 0
        t_hi = int(hi_c + int(rng.integers(-3, 4)))
        t_lo = int(lo_c + int(rng.integers(-3, 4)))
        if t_lo > t_hi:
            t_lo = t_hi
        hi[i], lo[i], st[i] = t_hi, t_lo, cond
    return {"state": st, "temp_hi": hi, "temp_lo": lo}


# ------------------------------------------------------------------ 検証
def _series_stats(hi: np.ndarray, lo: np.ndarray, state: np.ndarray,
                  seg_len: int, hot_c: float) -> dict:
    segs = hi.reshape(-1, seg_len)
    lag1 = []
    for s in segs:
        if len(s) > 2 and s.std() > 0:
            lag1.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    year_means = segs.mean(axis=1)

    def _skew(v):
        m, s = v.mean(), v.std(ddof=1)
        return float((((v - m) / s) ** 3).mean()) if s > 0 else 0.0

    return {
        "n_days": int(len(hi)),
        "mean_hi": _r6(hi.mean()), "sd_hi": _r6(hi.std(ddof=1)),
        "skew_hi": _r6(_skew(hi)), "skew_lo": _r6(_skew(lo)),
        "mean_lo": _r6(lo.mean()), "sd_lo": _r6(lo.std(ddof=1)),
        "max_hi": _r6(hi.max()), "min_hi": _r6(hi.min()),
        "p_hi_ge_35": _r6(float((hi >= 35.0).mean())),
        "p_hi_ge_36": _r6(float((hi >= 36.0).mean())),
        "p_hi_ge_37": _r6(float((hi >= 37.0).mean())),
        "p_hi_ge_33": _r6(float((hi >= 33.0).mean())),
        "p_lo_ge_25": _r6(float((lo >= 25.0).mean())),   # 熱帯夜
        "lag1_hi": _r6(float(np.mean(lag1)) if lag1 else 0.0),
        "interannual_sd_of_mean_hi": _r6(float(year_means.std(ddof=1))),
        "p_rain": _r6(float((state == RAIN).mean())),
        "spell": spell_stats(hi, seg_len, hot_c),
    }


def build_validation(pm: dict, obs: dict, seg_len: int, hot_c: float,
                     n_years: int, seed: int) -> dict:
    hi_o, lo_o, st_o = obs["hi"], obs["lo"], obs["state"]
    n_obs_years = len(hi_o) // seg_len
    o = _series_stats(hi_o, lo_o, st_o, seg_len, hot_c)
    sim = simulate(pm, n_years, seg_len, seed)
    g = _series_stats(sim["temp_hi"], sim["temp_lo"], sim["state"], seg_len, hot_c)
    g["share_clipped"] = _r6(sim["share_clipped"])
    sim_noy = simulate(pm, n_years, seg_len, seed, year_effect=False)
    g_noy = _series_stats(sim_noy["temp_hi"], sim_noy["temp_lo"], sim_noy["state"],
                          seg_len, hot_c)
    cur = simulate_synthetic_current(n_years, seg_len, seed)
    c = _series_stats(cur["temp_hi"], cur["temp_lo"], cur["state"], seg_len, hot_c)

    d_hi, p_hi = ks_two_sample(hi_o, sim["temp_hi"])
    d_lo, p_lo = ks_two_sample(lo_o, sim["temp_lo"])
    d_cur, p_cur = ks_two_sample(hi_o, cur["temp_hi"])
    # 猛暑連長分布の KS(連の長さの標本同士)
    lens_o = [x for s in hi_o.reshape(-1, seg_len) for x in run_lengths(s >= hot_c)]
    lens_g = [x for s in sim["temp_hi"].reshape(-1, seg_len) for x in run_lengths(s >= hot_c)]
    lens_c = [x for s in cur["temp_hi"].reshape(-1, seg_len) for x in run_lengths(s >= hot_c)]
    d_sp, p_sp = (ks_two_sample(np.array(lens_o, float), np.array(lens_g, float))
                  if lens_o and lens_g else (1.0, 0.0))
    d_spc, p_spc = (ks_two_sample(np.array(lens_o, float), np.array(lens_c, float))
                    if lens_o and lens_c else (1.0, 0.0))
    return {
        "monte_carlo": {"n_years": n_years, "seg_len": seg_len, "seed": seed,
                        "rng": "numpy PCG64 / SeedSequence([seed]) = RngHub と同型"},
        "observed": o,
        "generated": g,
        "generated_no_year_effect": g_noy,
        "current_synthetic_weather_py": c,
        "ks": {
            "temp_hi_observed_vs_generated": {"D": _r6(d_hi), "p": _r6(p_hi)},
            "temp_lo_observed_vs_generated": {"D": _r6(d_lo), "p": _r6(p_lo)},
            "temp_hi_observed_vs_current": {"D": _r6(d_cur), "p": _r6(p_cur)},
            "spell_len_observed_vs_generated": {"D": _r6(d_sp), "p": _r6(p_sp),
                                                "n_obs": len(lens_o), "n_gen": len(lens_g)},
            "spell_len_observed_vs_current": {"D": _r6(d_spc), "p": _r6(p_spc),
                                              "n_obs": len(lens_o), "n_cur": len(lens_c)},
            "note": ("p は漸近近似(scipy 非依存の自前実装)。生成側 n が大きいので"
                     "p は小さく出やすい=D の大小で比べること。"),
        },
        "observed_years": n_obs_years,
        "known_gaps": [
            {
                "metric": "lag-1 autocorrelation of daily Tmax (raw series)",
                "observed": o["lag1_hi"], "generated": g["lag1_hi"],
                "current_synthetic": c["lag1_hi"],
                "diagnosis": (
                    "標準化残差 χ の lag-1 は設計どおり再現している(実測 0.517 / 生成 0.509 を実測)。"
                    "生の系列で差が出るのは、WGEN 系が『天気状態の連鎖』と『気温残差の AR(1)』を"
                    "**独立**に置くため Cov(状態_t, 残差_{t-1}) が 0 になるから。実際には"
                    "『今日が平年より暑い → 明日も晴れやすい』という結合がある。"
                    "気温の条件付けを wet/dry 2状態(WGEN 原型)にしても 0.374 で改善しない"
                    "(状態別 SD をプールしても 0.409)ので、本バッチでは3状態条件付けのまま採用した。"),
                "citation": ("Weather generators generally underestimate persistence "
                             "(Wilks & Wilby 1999 系の既知の指摘)。"),
                "impact": ("連続猛暑の再現(=本件の目的)には効いていない: 猛暑連長の KS 距離は"
                           "生成 D=0.09 に対し現行合成 D=0.39。年内最長連の平均も 実測 3.55 / "
                           "生成 3.13 / 現行合成 1.48。"),
            },
            {
                "metric": "skewness of daily Tmax",
                "observed": o["skew_hi"], "generated": g["skew_hi"],
                "current_synthetic": c["skew_hi"],
                "diagnosis": ("実測は左に歪む(歪度 −1.2 = 上限が詰まり下側に長い裾)。"
                              "正規 AR(1) は対称なので上側を出しすぎる。"
                              "temp.clip(凍結実測の全期間レンジ ±1℃)で物理ガードを掛けているが、"
                              "P(Tmax>=36℃) は依然として実測より高めに出る。"),
                "impact": "36℃以上の日数を過大評価する。猛暑の『量』を論じるときは注意。",
            },
        ],
    }


def build_wbgt_block(obs: dict) -> dict:
    hi, rh_min, sun, wind = obs["hi"], obs["rh_min"], obs["sun"], obs["wind"]
    ok = ~(np.isnan(rh_min) | np.isnan(wind))
    sr = sr_proxy(sun[ok])
    w = wbgt_ono2014(hi[ok], rh_min[ok], sr, wind[ok])
    return {
        "formula": ("WBGT = 0.735*Ta + 0.0374*RH + 0.00292*Ta*RH "
                    "+ 7.619*SR - 4.557*SR^2 - 0.0572*WS - 4.064"),
        "coef": WBGT_ONO2014,
        "citation": ("小野雅司ら(2014)『通常観測気象要素を用いた WBGT の推定』日生気誌 50(4), 147-157。"
                     "式は環境省 熱中症予防情報サイト掲載のもの "
                     "https://www.wbgt.env.go.jp/wbgt_detail.php"),
        "measured_definition": {
            "formula": "WBGT = 0.7*Tw + 0.2*Tg + 0.1*Ta",
            "coef": WBGT_MEASURED,
            "note": "実測地点(全国47点)の定義。湿球 Tw・黒球 Tg を持たないので本バッチでは使わない",
        },
        "inputs_used": {
            "Ta": "日最高気温[℃](日最高 WBGT の代理)",
            "RH": "日最小相対湿度[%](日最高気温と同時刻に近いため)",
            "SR": f"日照時間から代理: SR = {SR_CLEAR_KW} * min(1, 日照時間 / {DAYLENGTH_H})",
            "WS": "日平均風速[m/s]",
        },
        "assumptions": [
            f"快晴日の日中平均全天日射量 = {SR_CLEAR_KW} kW/m^2(★仮定・実測ではない)",
            f"可照時間 = {DAYLENGTH_H} h(8月中旬の東京の目安・★仮定)",
            "日最高気温と日最小湿度が同時刻に起きるとみなす(★近似)",
            "SR 項の寄与は SR=0.5 で +2.7℃、SR=0.75 で +3.2℃ ある=仮定の感度は小さくない",
        ],
        "alert_thresholds": {
            "heat_alert_wbgt": 33.0,
            "note": ("熱中症警戒アラートは『翌日・当日の日最高暑さ指数(WBGT)が33(予測値)に"
                     "達する場合に発表』(https://www.wbgt.env.go.jp/about_alert.php)。"
                     "下の集計は本代理式の値であり、公式の暑さ指数ではない。"),
        },
        "observed_proxy_distribution": {
            "n": int(ok.sum()),
            "mean": _r6(float(w.mean())), "sd": _r6(float(w.std(ddof=1))),
            "p50": _r6(float(np.percentile(w, 50))),
            "p90": _r6(float(np.percentile(w, 90))),
            "max": _r6(float(w.max())),
            "share_ge_28": _r6(float((w >= 28).mean())),
            "share_ge_31": _r6(float((w >= 31).mean())),
            "share_ge_33": _r6(float((w >= 33).mean())),
        },
        "validation_status": ("★未検証。環境省の実測 WBGT(東京 44132・実況2010年〜)との突合は"
                             "本バッチでは未実施(CSV は POST フォーム経由、WebAPI は "
                             "https://www.wbgt.env.go.jp/api/v1/getSurveyData が HTTP 400 を返し"
                             "引数仕様を確定できなかった)。次バッチの課題。"),
    }


# ------------------------------------------------------------------ CLI
def _parse_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="実測から確率的天候生成器のパラメータを較正する")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--month", type=int, default=8)
    ap.add_argument("--fit-years", default="2015-2025",
                    help="較正窓(既定 2015-2025 = 観測点移転後。全期間は 1996-2025)")
    ap.add_argument("--alt-years", default="1996-2025",
                    help="感度分析として併記する窓(既定 1996-2025)")
    ap.add_argument("--wet-threshold-mm", type=float, default=1.0,
                    help="降水日の閾値[mm](気象庁の降水日 = 1.0mm 以上)")
    ap.add_argument("--hot-c", type=float, default=35.0, help="猛暑日の閾値[℃]")
    ap.add_argument("--validate-years", type=int, default=3000, help="自己検証のモンテカルロ年数")
    ap.add_argument("--validate-seed", type=int, default=20260801)
    ap.add_argument("--stamp", action="store_true",
                    help="壁時計時刻を meta に書く(★決定論=バイト同一が壊れる。既定 OFF)")
    ap.add_argument("--verify", action="store_true", help="既存 --out の payload_sha256 を検証")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.verify:
        if not out.exists():
            print(f"[NG] ファイルが無い: {out}", file=sys.stderr)
            return 2
        doc = json.loads(out.read_text(encoding="utf-8"))
        got = payload_sha256(doc)
        if got != doc["meta"].get("payload_sha256"):
            print(f"[NG] payload_sha256 不一致: 記載 {doc['meta'].get('payload_sha256')} / 実算 {got}",
                  file=sys.stderr)
            return 1
        print(f"[OK] {out} payload_sha256={got}")
        return 0

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[NG] 入力が無い: {inp}\n"
              f"    先に `python scripts/fetch_weather_history.py` を走らせること。", file=sys.stderr)
        return 2
    fit_years = _parse_range(args.fit_years)
    alt_years = _parse_range(args.alt_years)
    doc_src, days = load_history(inp, fit_years, args.month, args.wet_threshold_mm)
    if not days:
        print(f"[NG] 該当日が0件(month={args.month} years={fit_years})", file=sys.stderr)
        return 1
    seg_len = max(len([d for d in days if d["year"] == y])
                  for y in {d["year"] for d in days})
    n_per_year = {y: len([d for d in days if d["year"] == y]) for y in {d["year"] for d in days}}
    if len(set(n_per_year.values())) != 1:
        print(f"[NG] 年ごとの日数が揃っていない: {sorted(n_per_year.items())}", file=sys.stderr)
        return 1

    pm = fit_month(days, args.month, args.wet_threshold_mm, args.hot_c)
    obs = pm.pop("_obs_arrays")
    # 生成値の物理的な箍。★モデルではなくガード。凍結実測の**全期間**(較正窓ではない)の
    # レンジ ± margin で切る。正規 AR(1) は上側の裾を出しすぎる(実測 hi の歪度 −1.2)ため。
    all_month = [d for d in doc_src["days"] if d["month"] == args.month]
    a_hi = np.array([d["temp_hi"] for d in all_month if d["temp_hi"] is not None], float)
    a_lo = np.array([d["temp_lo"] for d in all_month if d["temp_lo"] is not None], float)
    margin = 1.0
    pm["temp"]["clip"] = {
        "hi_min": _r6(a_hi.min() - margin), "hi_max": _r6(a_hi.max() + margin),
        "lo_min": _r6(a_lo.min() - margin), "lo_max": _r6(a_lo.max() + margin),
        "margin_c": _r6(margin),
        "source_years": [int(min(d["year"] for d in all_month)),
                         int(max(d["year"] for d in all_month))],
        "note": ("凍結実測の全期間レンジ ± margin。正規 AR(1) の上側の裾の出しすぎを止める"
                 "物理ガードであってモデルではない(切られた割合は validation.generated."
                 "share_clipped に出す)。"),
    }
    validation = build_validation(pm, obs, seg_len, args.hot_c,
                                  args.validate_years, args.validate_seed)
    wbgt = build_wbgt_block(obs)

    # 感度分析: 別窓での主要量だけ(パラメータ全体は出さない=ファイルを膨らませない)
    _, days_alt = load_history(inp, alt_years, args.month, args.wet_threshold_mm)
    alt_block = None
    if days_alt:
        pm_alt = fit_month(days_alt, args.month, args.wet_threshold_mm, args.hot_c)
        obs_alt = pm_alt.pop("_obs_arrays")
        alt_block = {
            "years": [alt_years[0], alt_years[1]], "n_days": pm_alt["n_days"],
            "observed": _series_stats(obs_alt["hi"], obs_alt["lo"], obs_alt["state"],
                                      seg_len, args.hot_c),
            "p_wet_given_wet": pm_alt["markov2_wet_dry"]["p_wet_given_wet"],
            "p_wet_given_dry": pm_alt["markov2_wet_dry"]["p_wet_given_dry"],
            "grand_mean_hi": pm_alt["temp"]["grand_mean_hi"],
            "year_effect_sd_hi": pm_alt["temp"]["year_effect"]["sd_hi"],
            "lag1_hi": pm_alt["temp"]["ar1"]["lag1_hi"],
            "note": "感度分析用。観測点移転(2014-12-02)を跨ぐので系列に不連続がある",
        }

    result = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "scripts/fit_weather_gen.py",
            "source_file": str(inp.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_payload_sha256": doc_src["meta"]["payload_sha256"],
            "source_fetched_at_utc": doc_src["meta"]["fetched_at_utc"],
            "source_station": doc_src["meta"]["station"]["name_ja"],
            "source_attribution": doc_src["meta"]["source"]["attribution"],
            "fit_window": {"years": [fit_years[0], fit_years[1]],
                           "month": args.month, "seg_len": seg_len},
            "wet_threshold_mm": _r6(args.wet_threshold_mm),
            "hot_day_c": _r6(args.hot_c),
            "model": {
                "name": "WGEN 系 3状態マルコフ + 状態別気温 + Matalas 2変量 AR(1) + 年効果",
                "references": [
                    "Richardson, C.W. (1981) Stochastic simulation of daily precipitation, "
                    "temperature, and solar radiation. Water Resources Research 17(1):182-190.",
                    "Richardson, C.W. & Wright, D.A. (1984) WGEN: A model for generating daily "
                    "weather variables. USDA-ARS ARS-8.",
                    "Matalas, N.C. (1967) Mathematical assessment of synthetic hydrology. "
                    "Water Resources Research 3(4):937-945.",
                    "Wilks, D.S. & Wilby, R.L. (1999) The weather generation game: a review of "
                    "stochastic weather models. Progress in Physical Geography 23(3):329-357.",
                    "Chen, J., Brissette, F.P. & Leconte, R. (2010) A daily stochastic weather "
                    "generator for preserving low-frequency of climate variability. "
                    "Journal of Hydrology 388(3-4):480-490.",
                ],
                "note": "設計の根拠と出典 URL は docs/research/weather-generator-design.md §1",
            },
            "caveats": [
                "観測点は『東京』(北の丸公園)。渋谷の値ではない。都市キャノピー補正なし。",
                "既定窓 2015-2025 は 11年 = %d 日。年効果 SD の推定は 11 標本=不確かさ大。" % len(days),
                "窓内にも強い上昇トレンドがある(2015年 猛暑日8日 → 2025年 18日)。"
                "生成器はこれを年効果の分散として吸収する=2026年の予報ではない。",
                "WBGT は近似(全天日射量は日照時間からの代理)。公式の暑さ指数ではない。",
                "8月のみ較正済み。他の月は未較正=生成モードでは使えない(合成へフォールバックすること)。",
            ],
            "determinism": ("同じ入力・同じ引数ならバイト同一の JSON を出す"
                            "(--stamp を付けない限り壁時計時刻を書かない)。"),
        },
        "months": {str(args.month): pm},
        "wbgt": wbgt,
        "validation": validation,
        "sensitivity_alt_window": alt_block,
    }
    if args.stamp:
        import datetime as _dt
        result["meta"]["fitted_at_utc"] = (_dt.datetime.now(_dt.timezone.utc)
                                           .replace(microsecond=0).isoformat()
                                           .replace("+00:00", "Z"))
    result["meta"]["payload_sha256"] = payload_sha256(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    o, g, c = validation["observed"], validation["generated"], validation["current_synthetic_weather_py"]
    print(f"[OK] {out}")
    print(f"  窓 {fit_years[0]}-{fit_years[1]} / {pm['n_days']}日 / {pm['n_years']}年")
    print(f"  P(wet|wet)={pm['markov2_wet_dry']['p_wet_given_wet']}  "
          f"P(wet|dry)={pm['markov2_wet_dry']['p_wet_given_dry']}")
    print(f"  lag1 hi={pm['temp']['ar1']['lag1_hi']}  年効果SD hi={pm['temp']['year_effect']['sd_hi']}")
    print("  {:<30}{:>10}{:>12}{:>12}".format("metric", "observed", "generated", "current"))
    for key, lab in (("mean_hi", "mean Tmax [C]"), ("sd_hi", "sd Tmax [C]"),
                     ("skew_hi", "skew Tmax"), ("max_hi", "max Tmax [C]"),
                     ("p_hi_ge_35", "P(Tmax>=35C)"), ("p_hi_ge_36", "P(Tmax>=36C)"),
                     ("lag1_hi", "lag-1 autocorr"),
                     ("interannual_sd_of_mean_hi", "sd of yearly mean [C]"),
                     ("p_lo_ge_25", "P(Tmin>=25C) tropical night"),
                     ("p_rain", "P(rain)")):
        print("  {:<30}{:>10}{:>12}{:>12}".format(lab, str(o[key]), str(g[key]), str(c[key])))
    for key, lab in (("mean_len", "hot spell mean len"), ("max_len", "hot spell max len"),
                     ("year_max_mean", "yearly longest (mean)"),
                     ("p_year_has_len_ge_5", "P(year has >=5 in a row)"),
                     ("p_year_has_len_ge_10", "P(year has >=10 in a row)")):
        print("  {:<30}{:>10}{:>12}{:>12}".format(
            lab, str(o["spell"][key]), str(g["spell"][key]), str(c["spell"][key])))
    print(f"  share_clipped(generated) = {g.get('share_clipped')}")
    ks = validation["ks"]
    print(f"  KS(温度 実測vs生成) D={ks['temp_hi_observed_vs_generated']['D']}  "
          f"KS(実測vs現行) D={ks['temp_hi_observed_vs_current']['D']}")
    print(f"  KS(連長 実測vs生成) D={ks['spell_len_observed_vs_generated']['D']}  "
          f"KS(連長 実測vs現行) D={ks['spell_len_observed_vs_current']['D']}")
    print(f"  payload_sha256={result['meta']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
