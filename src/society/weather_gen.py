"""確率的天候生成器の本体(天候 W2 第80バッチ 2026-08-01)。

正典: docs/research/weather-generator-design.md(W1 の較正結果と統合設計)。

何のための module か
--------------------
`weather.py` の既定モード ``synthetic``(月別テーブル + ±3℃ 一様)は
**上限が 35℃・36℃以上は確率0・日間自己相関ゼロ**という構造的欠陥を持つ
(設計書 §0-2 に数値で確定済み)。本 module は較正済みパラメータ
(``data/snapshot/weather_gen_params.json``)から

  (1) 天気カテゴリ = 3状態1次マルコフ連鎖 {晴, 曇, 雨}(+雪)
  (2) 気温 = μ_state + β_month·(dom − d_ref) + Y(年効果) + σ_state·χ
      χ(d) = A·χ(d−1) + B·ε(d)   … Matalas(1967)の2変量 AR(1)
  (3) 降水量 = 湿日のガンマ分布 / 湿度 = 状態別平均 + 気温偏差への回帰 + 残差

で日別系列を生成する。**これは W1 で `scripts/fit_weather_gen.py` に書かれていた
生成ロジックそのもの**で、実装の二重化を避けるため src 側へ本体を置き、
scripts 側(較正・自己検証)が本 module を import する向きに整理してある
(= fit 側が出す系列と src 側が出す系列は原理的に同一)。

設計上の約束
------------
- **場所の知識を持たない**: 気温の値・地点・パラメータファイルのパスは1つも書かない。
  パスは envpack(``envpack.climate.gen_params`` / ``.table``)から降ってくる。
- **決定論**: 乱数は呼び出し側が渡す ``numpy.random.Generator`` からのみ引く。
  同じ (rng, params, n_days, day_start) なら必ず同じ系列。
- **prefix 安定**: 系列は逐次生成なので、n_days を伸ばしても先頭 n 日は不変。
  これが resume(= ラン再開時に系列を作り直す)で同一系列になる根拠。
- **AR(1) の状態を sim に持たせない**: 呼び出し側はラン開始時に全日数を一括生成して
  メモ化するだけ(``checkpoint.py`` は無改修)。

限界(隠さない・設計書 §3.4 / §6 と同じもの)
--------------------------------------------
- 生の日別系列の lag-1 自己相関は実測 0.52 に対し生成器 0.39(WGEN 系の既知の過小評価)。
- 正規 AR(1) は対称なので P(Tmax≥36℃) を実測より高めに出す(clip は物理ガードであって
  分布形を直さない)。
- WBGT は**公式の暑さ指数ではない**代理値。生成モードでは全天日射量を持たないので
  天気状態から仮定した ``SR_BY_STATE`` を使う(表引きモードでは実測の日照時間から出す)。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# 天気状態(weather.py の cond 4種と同じ語。順序は較正パラメータの states と一致すること)。
STATES = ("晴", "曇", "雨", "雪")
RAIN = 2
SNOW = 3

# --------------------------------------------------------------------------- #
# WBGT(暑さ指数)の推定式 — 小野雅司ら(2014)『通常観測気象要素を用いた WBGT の推定』
# 日生気誌 50(4), 147-157。式は環境省 熱中症予防情報サイト掲載のもの
# https://www.wbgt.env.go.jp/wbgt_detail.php
#   WBGT = 0.735*Ta + 0.0374*RH + 0.00292*Ta*RH + 7.619*SR − 4.557*SR² − 0.0572*WS − 4.064
#   Ta[℃] 気温 / RH[%] 相対湿度 / SR[kW/m²] 全天日射量 / WS[m/s] 平均風速
# --------------------------------------------------------------------------- #
WBGT_ONO2014 = {
    "ta": 0.735, "rh": 0.0374, "ta_rh": 0.00292,
    "sr": 7.619, "sr2": -4.557, "ws": -0.0572, "const": -4.064,
}
# 実測 WBGT の定義式(参考。湿球/黒球温度を持たないので使わない)。
WBGT_MEASURED = {"tw": 0.7, "tg": 0.2, "ta": 0.1}
SR_CLEAR_KW = 0.75      # 快晴日の日中平均 全天日射量[kW/m²](★仮定・実測ではない)
DAYLENGTH_H = 13.3      # 盛夏の可照時間[h]の目安(★仮定)
# ★生成モード専用の仮定: 較正パラメータは日照時間を持たないので、天気状態から日射を置く。
#   表引きモード(実測)では使わない(実測の日照時間から sr_proxy で出す)。
SR_BY_STATE = (0.75, 0.35, 0.10, 0.10)     # 晴 / 曇 / 雨 / 雪 [kW/m²]

HOT_DAY_C = 35.0        # 猛暑日(気象庁の定義: 日最高気温 35℃以上)


def wbgt_ono2014(ta, rh, sr, ws):
    """小野ら(2014)推定式。ta[℃], rh[%], sr[kW/m²], ws[m/s] → WBGT[℃]。"""
    c = WBGT_ONO2014
    return (c["ta"] * ta + c["rh"] * rh + c["ta_rh"] * ta * rh
            + c["sr"] * sr + c["sr2"] * sr * sr + c["ws"] * ws + c["const"])


def sr_proxy(sunshine_h):
    """日照時間[h] → 全天日射量[kW/m²] の**代理**(★実測ではない・線形近似)。"""
    return SR_CLEAR_KW * np.clip(np.asarray(sunshine_h, dtype=float) / DAYLENGTH_H,
                                 0.0, 1.0)


def label_state(day: dict, wet_mm: float) -> int:
    """実測1日 → 天気状態の添字。雨 = 日降水量 ≥ wet_mm(気象庁の降水日の定義 1.0mm)。

    晴/曇 は気象庁自身の「天気概況(昼)」の先頭語で分ける(晴・快晴 → 晴、それ以外 → 曇)。
    日照時間の閾値で切るより、気象庁の判定をそのまま使うほうが恣意性が少ない。
    """
    if (day.get("snow_cm") or 0.0) > 0.0:
        return SNOW
    if (day.get("precip_mm") or 0.0) >= wet_mm:
        return RAIN
    head = (day.get("cond_day") or "")
    return 0 if head.startswith(("晴", "快晴")) else 1


# --------------------------------------------------------------------------- #
# ハッシュ(来歴の連鎖)— scripts/fit_weather_gen.py と同一定義(fit 側が import する)
# --------------------------------------------------------------------------- #
def _canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_sha256(doc: dict) -> str:
    """較正パラメータ本体のハッシュ(壁時計時刻を除く=同入力→同値)。"""
    payload = {k: v for k, v in doc.items() if k != "meta"}
    meta = {k: v for k, v in doc["meta"].items()
            if k not in ("payload_sha256", "fitted_at_utc")}
    payload["meta"] = meta
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def history_payload_sha256(doc: dict) -> str:
    """凍結実測のハッシュ(取得日時を含めない)。fetch_weather_history.py と同一定義。"""
    payload = {"station": doc["meta"]["station"],
               "columns": doc["meta"]["columns"],
               "days": doc["days"]}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """ファイルの生バイト列の SHA-256(改行・整形まで含めた同一性)。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path_str: str) -> Path:
    """リポジトリルート相対(既存 config の規約)または絶対パスを解決する。"""
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- #
# 読み込み(改竄検知つき)
# --------------------------------------------------------------------------- #
def load_params(path_str: str, verify: bool = True) -> dict:
    """較正パラメータ JSON を読む。payload_sha256 が合わなければ即エラー(黙って使わない)。"""
    path = resolve_path(path_str)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        declared = (doc.get("meta") or {}).get("payload_sha256")
        got = payload_sha256(doc)
        if declared != got:
            raise ValueError(
                f"天候パラメータの payload_sha256 不一致(改竄またはファイル破損): {path}\n"
                f"  記載 {declared} / 実算 {got}")
    if not (doc.get("months") or {}):
        raise ValueError(f"天候パラメータに較正済みの月が1つも無い: {path}")
    return doc


def load_table(path_str: str, verify: bool = True) -> dict:
    """凍結実測(日別)JSON を読む。payload_sha256 が合わなければ即エラー。"""
    path = resolve_path(path_str)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        declared = (doc.get("meta") or {}).get("payload_sha256")
        got = history_payload_sha256(doc)
        if declared != got:
            raise ValueError(
                f"天候実測データの payload_sha256 不一致(改竄またはファイル破損): {path}\n"
                f"  記載 {declared} / 実算 {got}")
    if not doc.get("days"):
        raise ValueError(f"天候実測データが空: {path}")
    return doc


def table_by_date(doc: dict) -> dict:
    """凍結実測 → {"YYYY-MM-DD": 日レコード}。"""
    return {str(d["date"]): d for d in doc["days"]}


def table_latest_by_monthday(doc: dict) -> dict:
    """凍結実測 → {"MM-DD": その月日の**最新年**のレコード}。

    シムの暦(例: 2026年)が凍結データの年(1996–2025)の外にあるとき、
    「同じ時期の直近の現実」へ回すための索引。年を差し替えたことは呼び出し側が
    必ず来歴へ記録する(黙って別の年の値を使わない)。
    """
    out: dict[str, dict] = {}
    for d in doc["days"]:
        key = f"{int(d['month']):02d}-{int(d['day']):02d}"
        cur = out.get(key)
        if cur is None or int(d["year"]) > int(cur["year"]):
            out[key] = d
    return out


# --------------------------------------------------------------------------- #
# 生成器
# --------------------------------------------------------------------------- #
def prepare(pm: dict) -> dict:
    """1つの月の較正パラメータ → 生成に使う行列・ベクトルの前計算(乱数を引かない純関数)。"""
    trans = np.array(pm["markov3"]["transition"], float)
    trans = trans / trans.sum(axis=1, keepdims=True)   # 丸め(6桁)で 1.0 を割るのを直す
    cum = np.cumsum(trans, axis=1)
    marg = np.array(pm["markov3"]["marginal"], float)
    marg_cum = np.cumsum(marg / marg.sum())
    ye = pm["temp"]["year_effect"]
    sd_y = np.array([ye["sd_hi"], ye["sd_lo"]], float)
    corr_y = float(ye["corr"])
    cov_y = np.array([[sd_y[0] ** 2, corr_y * sd_y[0] * sd_y[1]],
                      [corr_y * sd_y[0] * sd_y[1], sd_y[1] ** 2]])
    try:
        chol_y = np.linalg.cholesky(cov_y)
    except np.linalg.LinAlgError:
        chol_y = np.diag(sd_y)
    states = list(pm.get("states") or STATES)
    clip = pm["temp"].get("clip") or {}
    hum = pm["humidity"]
    rh_ps = hum["per_state"]
    # 実測 RH の総平均(状態別平均の標本数加重)= 日最小湿度の回帰の基準点。
    n_tot = sum(float(rh_ps[s]["n"]) for s in states) or 1.0
    rh_grand = sum(float(rh_ps[s]["n"]) * float(rh_ps[s]["mean"]) for s in states) / n_tot
    return {
        "states": states,
        "cum": cum, "marg_cum": marg_cum,
        "A": np.array(pm["temp"]["ar1"]["A"], float),
        "B": np.array(pm["temp"]["ar1"]["B"], float),
        "chol_y": chol_y,
        "mean_hi": np.array([pm["temp"]["per_state"][s]["mean_hi"] for s in states]),
        "sd_hi": np.array([pm["temp"]["per_state"][s]["sd_hi"] for s in states]),
        "mean_lo": np.array([pm["temp"]["per_state"][s]["mean_lo"] for s in states]),
        "sd_lo": np.array([pm["temp"]["per_state"][s]["sd_lo"] for s in states]),
        "b_hi": float(pm["temp"]["month_trend_hi_per_day"]),
        "b_lo": float(pm["temp"]["month_trend_lo_per_day"]),
        "d_ref": float(pm["d_ref"]),
        "rh_mean": np.array([float(rh_ps[s]["mean"]) for s in states]),
        "rh_grand": float(rh_grand),
        "beta_rh": float(hum["beta_on_temp_hi_anom"]),
        "rh_sd": float(hum["resid_sd"]),
        "rh_min_slope": float(hum.get("humid_min_from_mean_slope", 1.0)),
        "rh_min_mean": float(hum.get("humid_min_mean", rh_grand)),
        "wind_mean": float(hum.get("wind_mean_mean", 2.0)),
        "g_shape": float(pm["precip"]["gamma_shape"]),
        "g_scale": float(pm["precip"]["gamma_scale"]),
        "hi_min": float(clip.get("hi_min", -1e9)), "hi_max": float(clip.get("hi_max", 1e9)),
        "lo_min": float(clip.get("lo_min", -1e9)), "lo_max": float(clip.get("lo_max", 1e9)),
    }


def generate_segment(rng, prep: dict, n_days: int, day_start: int = 1,
                     year_effect: bool = True) -> dict:
    """1 本の連続系列(= 1 年ぶん / 1 ラン ぶん)を生成する。

    **乱数の消費順**(この順序が W1 の `scripts/fit_weather_gen.py` と一致することが
    実装二重化を避ける唯一の要件。テストが同一系列を機械検査する):
      年効果(year_effect=True のときだけ 2 draw)→ χ0(2 draw)→ 初日の状態(1 draw)
      → 各日: [t>0 なら 状態遷移(1)+ χ 更新(2)] → 湿度残差(1) → 雨なら降水量(1)

    返り値は長さ n_days の numpy 配列群 + 箍(clip)に触れた日数。
    """
    n_states = len(prep["states"])
    out_state = np.zeros(n_days, int)
    out_hi = np.zeros(n_days)
    out_lo = np.zeros(n_days)
    out_rh = np.zeros(n_days)
    out_pr = np.zeros(n_days)
    n_clipped = 0
    a_mat, b_mat = prep["A"], prep["B"]
    mean_hi, sd_hi = prep["mean_hi"], prep["sd_hi"]
    mean_lo, sd_lo = prep["mean_lo"], prep["sd_lo"]
    hi_min, hi_max = prep["hi_min"], prep["hi_max"]
    lo_min, lo_max = prep["lo_min"], prep["lo_max"]

    yeff = prep["chol_y"] @ rng.standard_normal(2) if year_effect else np.zeros(2)
    chi = rng.standard_normal(2)
    s = min(int(np.searchsorted(prep["marg_cum"], rng.random())), n_states - 1)
    for t in range(n_days):
        if t > 0:
            s = min(int(np.searchsorted(prep["cum"][s], rng.random())), n_states - 1)
            chi = a_mat @ chi + b_mat @ rng.standard_normal(2)
        dom = day_start + t
        hi = mean_hi[s] + prep["b_hi"] * (dom - prep["d_ref"]) + yeff[0] + sd_hi[s] * chi[0]
        lo = mean_lo[s] + prep["b_lo"] * (dom - prep["d_ref"]) + yeff[1] + sd_lo[s] * chi[1]
        # ★正規 AR(1) は上側の裾を出しすぎる(実測 hi の歪度 −1.2 = 左に歪んだ分布)。
        #   物理的な箍として凍結実測の全期間レンジ ± margin で切る(モデルではなくガード)。
        if hi > hi_max or hi < hi_min or lo > lo_max or lo < lo_min:
            n_clipped += 1
        hi = min(hi_max, max(hi_min, hi))
        lo = min(lo_max, max(lo_min, lo))
        if lo > hi:                       # 現行 weather.py と同じ後段ガード
            lo = hi
        rh = (prep["rh_mean"][s] + prep["beta_rh"] * (hi - mean_hi[s] - yeff[0])
              + prep["rh_sd"] * rng.standard_normal())
        out_state[t] = s
        out_hi[t] = hi
        out_lo[t] = lo
        out_rh[t] = min(100.0, max(10.0, rh))
        out_pr[t] = rng.gamma(prep["g_shape"], prep["g_scale"]) if s == RAIN else 0.0
    return {"state": out_state, "temp_hi": out_hi, "temp_lo": out_lo,
            "humid_mean": out_rh, "precip_mm": out_pr, "n_clipped": n_clipped}


def day_records(seg: dict, prep: dict) -> list[dict]:
    """生成系列 → 1日1件の dict(weather.py が消費する形)。

    ``cond`` は現行 synthetic と同じ4語彙。``temp_hi`` / ``temp_lo`` は int に丸める
    (プロンプト1行の見た目を synthetic と揃えるため)。生の実数は ``temp_hi_c`` に残す。
    ``wbgt`` は**公式の暑さ指数ではない代理値**(日射は天気状態からの仮定 SR_BY_STATE)。
    """
    states = prep["states"]
    out: list[dict] = []
    for i in range(len(seg["state"])):
        s = int(seg["state"][i])
        hi = float(seg["temp_hi"][i])
        lo = float(seg["temp_lo"][i])
        rh = float(seg["humid_mean"][i])
        # 日最小湿度 = 総平均を基準にした回帰(較正時の OLS の切片と同値の復元)。
        rh_min = prep["rh_min_mean"] + prep["rh_min_slope"] * (rh - prep["rh_grand"])
        rh_min = min(100.0, max(5.0, rh_min))
        sr = SR_BY_STATE[s] if s < len(SR_BY_STATE) else SR_BY_STATE[-1]
        out.append({
            "cond": states[s],
            "temp_hi": int(round(hi)),
            "temp_lo": int(round(lo)),
            "temp_hi_c": round(hi, 1),
            "temp_lo_c": round(lo, 1),
            "precip_mm": round(float(seg["precip_mm"][i]), 1),
            "humid": int(round(rh)),
            "wbgt": round(float(wbgt_ono2014(hi, rh_min, sr, prep["wind_mean"])), 1),
            "source": "generated",
        })
    return out


def record_from_observation(day: dict, wet_mm: float = 1.0) -> dict:
    """凍結実測の1日 → weather.py が消費する形(表引きモード)。

    WBGT は W1 と同じ代理(Ta=日最高気温 / RH=日最小湿度 / SR=日照時間からの代理 /
    WS=日平均風速)。**公式の暑さ指数ではない**。
    """
    s = label_state(day, wet_mm)
    hi = float(day["temp_hi"])
    lo = float(day["temp_lo"])
    rh = day.get("humid_mean")
    rh_min = day.get("humid_min")
    ws = day.get("wind_mean")
    sun = day.get("sunshine_h") or 0.0
    wbgt = None
    if rh_min is not None and ws is not None:
        wbgt = round(float(wbgt_ono2014(hi, float(rh_min), float(sr_proxy(sun)),
                                        float(ws))), 1)
    return {
        "cond": STATES[s],
        "temp_hi": int(round(hi)),
        "temp_lo": int(round(lo)),
        "temp_hi_c": round(hi, 1),
        "temp_lo_c": round(lo, 1),
        "precip_mm": round(float(day.get("precip_mm") or 0.0), 1),
        "humid": (int(round(float(rh))) if rh is not None else None),
        "wbgt": wbgt,
        "date": str(day["date"]),
        "source": "table",
    }


# --------------------------------------------------------------------------- #
# 系列の要約(検収・summary 用。乱数を引かない)
# --------------------------------------------------------------------------- #
def series_stats(records: list[dict], hot_c: float = HOT_DAY_C) -> dict:
    """生成/表引き系列の要約統計(猛暑日率・連長・平均など)。"""
    if not records:
        return {"n_days": 0}
    hi = [float(r.get("temp_hi_c", r["temp_hi"])) for r in records]
    runs: list[int] = []
    cur = 0
    for v in hi:
        if v >= hot_c:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    n = len(hi)
    mean = sum(hi) / n
    var = sum((v - mean) ** 2 for v in hi) / max(1, n - 1)
    return {
        "n_days": n,
        "mean_temp_hi": round(mean, 3),
        "sd_temp_hi": round(math.sqrt(var), 3),
        "max_temp_hi": round(max(hi), 1),
        "min_temp_hi": round(min(hi), 1),
        "p_hot_day": round(sum(1 for v in hi if v >= hot_c) / n, 4),
        "hot_spells": runs,
        "max_hot_spell": max(runs) if runs else 0,
        "n_rain_days": sum(1 for r in records if r["cond"] in ("雨", "雪")),
    }
