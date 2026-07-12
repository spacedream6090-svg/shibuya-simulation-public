"""日付・カレンダー(第7バッチ 2026-07-07。ユーザー要望)。

sim_min から暦(日付・曜日・平日/休日)を導出する純関数群。エージェント行動の
暦的文脈(グローバル・全員共通・k 非依存)。すべて乱数を一切引かない=決定論。

★ このモジュールは world/ 配下(tests/test_contracts の走査対象)なので、
  因子語(構成概念名)を一切書かない。ここが扱うのは「日付」だけ。
既定 OFF(enabled=false)= date_line は None を返し、既存挙動に一切影響しない。
"""
from __future__ import annotations

import datetime

# 曜日の日本語(0=月 … 6=日)。datetime.date.weekday() と同じ並び。
_WEEKDAY_JP = "月火水木金土日"


def build_cfg(raw: dict | None) -> dict:
    """conf の world.calendar ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "start_date": str(raw.get("start_date", "2026-04-01")),
        "weekday_work": bool(raw.get("weekday_work", False)),
        "holidays": [str(h) for h in (raw.get("holidays") or [])],
    }


def day_index(sim_min: int) -> int:
    """経過日インデックス(0=初日)。1日=1440分。"""
    return sim_min // 1440


def _start_date(cfg: dict) -> datetime.date:
    return datetime.date.fromisoformat(cfg["start_date"])


def date_of(cfg: dict, sim_min: int) -> datetime.date:
    """その sim_min の実日付 = start_date + 経過日。"""
    return _start_date(cfg) + datetime.timedelta(days=day_index(sim_min))


def weekday_of(cfg: dict, sim_min: int) -> int:
    """曜日(0=月 … 6=日)。"""
    return date_of(cfg, sim_min).weekday()


def weekday_jp(cfg: dict, sim_min: int) -> str:
    """曜日の日本語1文字(月火水木金土日)。"""
    return _WEEKDAY_JP[weekday_of(cfg, sim_min)]


def is_holiday(cfg: dict, sim_min: int) -> bool:
    """休日か(土日、または holidays に含まれる YYYY-MM-DD)。"""
    d = date_of(cfg, sim_min)
    if d.weekday() >= 5:                      # 土(5)・日(6)
        return True
    return d.isoformat() in cfg["holidays"]


def is_workday(cfg: dict, sim_min: int) -> bool:
    """平日(=非休日)か。weekday_work ゲートの判定に使う。"""
    return not is_holiday(cfg, sim_min)


def date_line(cfg: dict, sim_min: int) -> str | None:
    """プロンプトへ注入する日付1行。無効なら None(=注入しない=不変)。

    例: "今日は2026年4月8日(火)です。"
    """
    if not cfg["enabled"]:
        return None
    d = date_of(cfg, sim_min)
    return f"今日は{d.year}年{d.month}月{d.day}日({weekday_jp(cfg, sim_min)})です。"
