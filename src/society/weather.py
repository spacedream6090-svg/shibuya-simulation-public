"""天気(第7バッチ 2026-07-07。ユーザー要望)。

その日の天気を日次で決定論生成し、エージェント行動・気分に影響させる(グローバル
文脈・全員共通・k 非依存)。天気は新 stream ``sim.hub.stream("weather", day_index)``
から**1日1回だけ**引く(既存 stream の draw 順に影響しない=決定論・ゴールデン不変)。

天気→不快感(grievance)は factors 係数レイヤー経由(R9)。このモジュールは grievance の
係数(rain_grievance)を保持し、雨天など不快な天気の日に不透明な magnitude を返すだけ
(scheduler がそれを factors.update.on_weather に渡す)。乱数は増やさない(天気は既に確定
=決定論的加算)。

★ このファイルは src/society 直下(engine/cognition/world 検査対象外)なので、因子語を
  含む係数ロジックはここに置いてよい(no-fingerprint 契約に触れない)。
既定 OFF(enabled=false)= 一切引かず・イベントも出さず・プロンプトにも載らない(不変)。
"""
from __future__ import annotations

from .world import calendar as _calendar

# 月(1-12)→ (最高気温中心, 最低気温中心, 雨の重み, 雪の重み)。
# 梅雨(6-7)は雨増、盛夏(7-8)は高温、冬(12-2)は寒く低確率で雪。現実の東京の月別気候の近似。
_MONTH_CLIMATE = {
    1: (10, 2, 0.15, 0.08), 2: (11, 2, 0.15, 0.06), 3: (15, 6, 0.22, 0.0),
    4: (20, 10, 0.25, 0.0), 5: (24, 14, 0.25, 0.0), 6: (26, 18, 0.45, 0.0),
    7: (30, 23, 0.42, 0.0), 8: (32, 25, 0.30, 0.0), 9: (27, 20, 0.35, 0.0),
    10: (22, 14, 0.25, 0.0), 11: (17, 9, 0.20, 0.0), 12: (12, 4, 0.15, 0.05),
}
# 季節なし(calendar OFF)の既定気候。ほぼ一様寄りの穏やかな分布。
_NEUTRAL_CLIMATE = (21, 12, 0.22, 0.0)

# 悪天候(不快感の源)とみなす天気。
_BAD_CONDS = {"雨", "雪"}


def build_cfg(raw: dict | None) -> dict:
    """conf の weather ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        # 雨天の日に grievance を上げる係数(factors 経由の加算量)。★ON 推奨: 0.01。
        # 既定 0.0 = 天気 ON でも grievance は動かさない(天気イベント・文脈行のみ)。
        "rain_grievance": float(raw.get("rain_grievance", 0.0)),
    }


def _sample(rng, month: int | None) -> dict:
    """1日ぶんの天気を決定論生成(rng は当日専用 stream)。"""
    hi_c, lo_c, rain_w, snow_w = (
        _MONTH_CLIMATE[month] if month is not None else _NEUTRAL_CLIMATE)
    r = float(rng.random())
    if r < snow_w:
        cond = "雪"
    elif r < snow_w + rain_w:
        cond = "雨"
    else:                                    # 残りを 曇(4割)/ 晴(6割)に割る
        span = 1.0 - snow_w - rain_w
        rem = r - (snow_w + rain_w)
        cond = "曇" if rem < span * 0.4 else "晴"
    temp_hi = int(hi_c + int(rng.integers(-3, 4)))     # ±3℃ の日毎ゆらぎ
    temp_lo = int(lo_c + int(rng.integers(-3, 4)))
    if temp_lo > temp_hi:                    # 最低が最高を超えない
        temp_lo = temp_hi
    return {"cond": cond, "temp_hi": temp_hi, "temp_lo": temp_lo}


def weather_for(sim, day_index: int) -> dict:
    """当日の天気を決定論生成して返す。dict = {"cond", "temp_hi", "temp_lo"}。

    calendar が enabled のときはその日の月から季節バイアスを掛ける(夏は高温・梅雨は雨増)。
    calendar OFF なら季節なしの穏やかな分布。乱数は新 stream "weather" から1日1回だけ引く。
    """
    rng = sim.hub.stream("weather", day_index)
    cal = getattr(sim, "calendarcfg", None)
    month = None
    if cal and cal.get("enabled"):
        month = _calendar.date_of(cal, day_index * 1440).month
    return _sample(rng, month)


def weather_line(weather_dict: dict | None) -> str | None:
    """プロンプトへ注入する天気1行。無ければ None(=注入しない=不変)。

    例: "今日の天気: 雨、最高14℃。"
    """
    if not weather_dict:
        return None
    return f"今日の天気: {weather_dict['cond']}、最高{weather_dict['temp_hi']}℃。"


def discomfort_delta(weather_dict: dict | None, cfg: dict) -> float:
    """悪天候(雨・雪)の日の不快感の量(不透明 magnitude)。scheduler が factors へ渡す。

    晴・曇や rain_grievance=0.0(既定)のときは 0.0(=加算なし)。乱数は引かない(決定論)。
    """
    if weather_dict and weather_dict.get("cond") in _BAD_CONDS:
        return float(cfg.get("rain_grievance", 0.0))
    return 0.0
