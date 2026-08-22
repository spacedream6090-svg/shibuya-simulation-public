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

# =========================================================================== #
# 曜日仕様の語彙とパーサ(第144: `world/presence.py` から**移設**して 1 本化)
#
# なぜここか: 「曜日仕様の文字列 → 曜日集合」は暦そのものの語彙であり、
# 名簿の在場判定(presence)と勤務ゲート(cognition/routine・engine/scheduler)の
# **両方**が同じ規則で読まなければならない。写しを 2 つ持つと片方だけが直る
# (実際 `mon-sat` の解釈は presence 側だけが直っていた)。presence 側は本 module から
# import するだけになり、公開名(`_days_match` / `_parse_days` / `_WEEKDAYS`)は
# 再輸出で保つ = あちらの既存テストは 1 行も変わらない。
# ★乱数も日付も引かない純関数(spec と weekday の 2 引数だけ)= 決定論。
# =========================================================================== #
#: Mon..Fri(暦の weekday() と同じ並び)。
WEEKDAYS = frozenset({0, 1, 2, 3, 4})
#: 曜日名 → index(0=Mon..6=Sun)。範囲記法 "a-b" の両端に使う。
DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
#: 範囲記法でない固定語彙(名簿・台帳・conf が実際に書いている語)。
DAYS_ALIAS: dict[str, frozenset[int]] = {
    "all": frozenset(range(7)),
    "everyday": frozenset(range(7)),
    "daily": frozenset(range(7)),
    "weekday": WEEKDAYS,           # 役割の当番表が使う語
    "weekdays": WEEKDAYS,
    "school_day": WEEKDAYS,        # 通学日
    "weekend": frozenset({5, 6}),
    "weekends": frozenset({5, 6}),
    "holiday": frozenset({5, 6}),
}


def parse_days(spec: str) -> frozenset[int] | None:
    """曜日仕様 → 出勤曜日の集合。**解釈できない仕様は None**(呼び出し側が後退する)。

    受理する形(名簿・台帳・conf が実際に書いている形だけ。推測で語彙を増やさない):
      - 固定語 `DAYS_ALIAS`("all" / "weekday" / "school_day" / "weekend" …)
      - 範囲記法 `mon-fri` / `mon-sat` / `tue-sun` / `sat-sun`(両端含む・折り返し可)
      - カンマ列 `mon,wed,fri`(範囲と混在可: `mon-fri,sun`)
    """
    key = str(spec or "").strip().lower()
    if not key:
        return None
    got: set[int] = set()
    for token in key.split(","):
        token = token.strip()
        if not token:
            continue
        alias = DAYS_ALIAS.get(token)
        if alias is not None:
            got |= set(alias)
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            lo, hi = DAY_INDEX.get(lo_s.strip()), DAY_INDEX.get(hi_s.strip())
            if lo is None or hi is None:
                return None
            # 折り返し(例 "sat-sun" は 5,6 / "sun-tue" は 6,0,1)も素直に展開する。
            got |= {(lo + i) % 7 for i in range((hi - lo) % 7 + 1)}
            continue
        one = DAY_INDEX.get(token)
        if one is None:
            return None
        got.add(one)
    return frozenset(got) or None


def days_match(spec: str, weekday: int) -> bool:
    """曜日仕様と当日 weekday の一致(**解釈できない仕様だけ**平日扱いに後退)。

    後退規則は残す(未知の語を黙って「毎日出勤」にしない方が安全側)。
    """
    if not spec or spec == "all":
        return True
    days = parse_days(spec)
    if days is None:
        return weekday in WEEKDAYS            # 解釈できない仕様 = 従来どおり平日扱い
    return int(weekday) in days


def build_cfg(raw: dict | None) -> dict:
    """conf の world.calendar ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "start_date": str(raw.get("start_date", "2026-04-01")),
        "weekday_work": bool(raw.get("weekday_work", False)),
        "holidays": [str(h) for h in (raw.get("holidays") or [])],
        # ★第144(既定 false = 現行と 1 バイト同一)。
        #  respect_work_days: weekday_work のゲートを「個体が宣言した営業曜日
        #    (agent.work_dow ← 台帳 shift_pattern.days)」で解く。宣言の無い個体は
        #    従来どおり is_workday(平日 + holidays)へ後退する。
        #  calendar_weekday: 在場(presence)とバイトの曜日時計を day%7 から**暦の曜日**へ
        #    差し替える。false = 従来の day%7(day0=月曜)。
        "respect_work_days": bool(raw.get("respect_work_days", False)),
        "calendar_weekday": bool(raw.get("calendar_weekday", False)),
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


def days_in_month(cfg: dict, sim_min: int) -> int:
    """その日が属する月の日数(28..31)。「月末払い」の給料日判定に使う。"""
    d = date_of(cfg, sim_min)
    nxt = (d.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    return (nxt - datetime.timedelta(days=1)).day


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
