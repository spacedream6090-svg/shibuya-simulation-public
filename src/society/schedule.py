"""長期予定・スケジュール帳(第7バッチ 2026-07-07。ユーザー要望)。

会話(speak / DM)に含まれる未来の日付・時刻を**決定論パーサ**(正規表現)で抽出し、
話者と聞き手のスケジュール帳へ自動記入する。近い予定はプロンプトに1行注入して行動・
発話の継続性(人間らしさ)を出す。すべて乱数を一切引かない=決定論。**追加 LLM 呼び出し
ゼロ**(R1): 解析対象は「既に生成済みの発話テキスト」だけで、生成を1回も増やさない。

★ このファイルは src/society 直下(engine/cognition/world 検査対象外)なので、日本語の
  時間表現ロジックをここに置く。因子語(構成概念名)は一切書かない。
既定 OFF(enabled=false)= 抽出も注入もせず・イベントも出さない(現行挙動と完全同一)。

データ構造(1件の予定):
  {"day": int,        # 絶対 day_index(calendar で解決済み。1日=1440分)
   "when": str,       # "朝|昼|午後|夕方|夜" または "HH:MM"(取れなければ "")
   "what": str,       # 行動("会う|食事|遊び|買い物|勉強会|イベント" 等。取れなければ "")
   "place": str,      # 場所名(取れなければ "")
   "with": list[int], # 相手のエージェント id(会話相手)
   "src_step": int}   # 記入元 step(観測用)
"""
from __future__ import annotations

import datetime
import re

from .world import calendar as _cal

# 全角数字 → 半角(会話テキストの正準化)。
_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")

# 曜日文字(0=月 .. 6=日。datetime.date.weekday() と同じ並び)。
_WEEKDAY_CH = "月火水木金土日"

# 時間帯語 → 代表分(近さ比較・並べ替え用)。"" は末尾に寄せる。
_BAND_MIN = {"朝": 8 * 60, "昼": 12 * 60, "午後": 14 * 60,
             "夕方": 18 * 60, "夜": 20 * 60, "": 24 * 60}

# 行動(what)の近傍語 → 正規化ラベル(先頭一致優先)。
_WHAT_PATTERNS = (
    (re.compile(r"会お|会う|会い|会え|集ま|集合|待ち合わ"), "会う"),
    (re.compile(r"ご飯|ごはん|食事|ランチ|ディナー|飲み|飲も|お茶"), "食事"),
    (re.compile(r"買い物|ショッピング"), "買い物"),
    (re.compile(r"勉強会|勉強|ゼミ|講義"), "勉強会"),
    (re.compile(r"遊ぼ|遊び|遊ぶ"), "遊び"),
    (re.compile(r"イベント|ライブ|コンサート|花火|祭"), "イベント"),
)

# 場所(place)のヒント語のうち **generic な施設語**(場所非依存=基盤に置いてよい)。
# 地名固有のヒント(スクランブル交差点・渋谷駅 等)は envpack.lexicon.place_hints から受ける
# (基盤に地名を残さない)。抽出時は place_hints(地名固有)を先に、この施設語を後に連結する
# (元の「長い語=地名を先に」順を保つ=バイト一致)。
_GENERIC_PLACE_HINTS = (
    "図書館", "美術館", "映画館", "居酒屋", "レストラン", "喫茶店",
    "カフェ", "公園", "駅", "学校", "会社", "家",
)
# 「◯◯で(会う/集まる/…)」= 場所の助詞パターン(ヒントに無い固有地名の後退用)。
# 平仮名は含めない(「あとで会う」等の副詞を場所に誤認しない)。
_DE_RE = re.compile(r"([一-龥ァ-ヴーA-Za-z0-9]{1,10})で"
                    r"(?:会|集ま|集合|待ち合わ|遊|食|飲)")


def build_cfg(raw: dict | None) -> dict:
    """conf の schedule ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_items": int(raw.get("max_items", 8)),
        "horizon_days": int(raw.get("horizon_days", 14)),
        "inject_prompt": bool(raw.get("inject_prompt", True)),
    }


# ---------------------------------------------------------------- 日付・時刻の解決
def _start_date(cal: dict) -> datetime.date:
    return _cal.date_of(cal, 0)                      # start_date(day_index=0 の日)


def _base_date(cal: dict, base_day: int) -> datetime.date:
    return _cal.date_of(cal, base_day * 1440)        # start_date + base_day


def _weekday_of(cal: dict, day: int) -> int:
    return _cal.weekday_of(cal, day * 1440)          # 0=月 .. 6=日


def _next_weekday(cal: dict, base_day: int, target_wd: int) -> int:
    """base_day 以降で次に target_wd 曜になる絶対 day_index(当日は除く=常に未来)。"""
    base_wd = _weekday_of(cal, base_day)
    offset = ((target_wd - base_wd - 1) % 7) + 1     # 1..7
    return base_day + offset


def _abs_month_day(cal: dict, base_day: int, month: int, day: int) -> int | None:
    """「◯月◯日」→ 現在日以降で直近の該当日の絶対 day_index(無効日は None)。"""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    start = _start_date(cal)
    base = _base_date(cal, base_day)
    for yr in (base.year, base.year + 1):
        try:
            target = datetime.date(yr, month, day)
        except ValueError:
            return None
        if target >= base:
            return (target - start).days
    return None


def _next_dom(cal: dict, base_day: int, dom: int) -> int | None:
    """「◯日」(日のみ)→ 現在日以降で直近の該当日の絶対 day_index。"""
    if not (1 <= dom <= 31):
        return None
    start = _start_date(cal)
    d = _base_date(cal, base_day)
    for _ in range(62):                              # 高々2か月先まで探せば足りる
        if d.day == dom:
            return (d - start).days
        d += datetime.timedelta(days=1)
    return None


def _match_day(text: str, base_day: int, cal: dict) -> int | None:
    """未来の「日」を1つ解決して絶対 day_index を返す(無ければ None)。優先順に判定。"""
    if "明後日" in text or "あさって" in text:
        return base_day + 2
    if "明日" in text or "あした" in text or "あす" in text:
        return base_day + 1
    m = re.search(r"(\d{1,2})日後", text)
    if m:
        return base_day + int(m.group(1))
    if "今夜" in text or "今晩" in text or "本日" in text or "今日" in text:
        return base_day
    if "再来週" in text:
        return base_day + 14
    if "来週" in text:
        return base_day + 7
    if "週末" in text:
        return _next_weekday(cal, base_day, 5)       # 次の土曜
    m = re.search(r"(?:今度の|次の|この)?([月火水木金土日])曜", text)
    if m:
        return _next_weekday(cal, base_day, _WEEKDAY_CH.index(m.group(1)))
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text)
    if m:
        return _abs_month_day(cal, base_day, int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d{1,2})日(?!後)", text)
    if m:
        return _next_dom(cal, base_day, int(m.group(1)))
    return None


def _fmt_clock(h: int, mm: int) -> str:
    return f"{h:02d}:{mm:02d}"


def _match_time(text: str) -> str:
    """時刻(HH:MM)または時間帯語を返す(無ければ "")。"""
    m = re.search(r"午前(\d{1,2})時(半)?", text)
    if m:
        return _fmt_clock(int(m.group(1)) % 12, 30 if m.group(2) else 0)
    m = re.search(r"午後(\d{1,2})時(半)?", text)
    if m:
        return _fmt_clock(int(m.group(1)) % 12 + 12, 30 if m.group(2) else 0)
    m = re.search(r"(\d{1,2})時(半)?", text)
    if m and 0 <= int(m.group(1)) <= 23:
        return _fmt_clock(int(m.group(1)), 30 if m.group(2) else 0)
    if "今夜" in text or "今晩" in text:
        return "夜"
    for band in ("夕方", "朝", "昼", "夜"):
        if band in text:
            return band
    if "午後" in text:
        return "午後"
    return ""


def _is_clock(when: str) -> bool:
    return ":" in when


def _clock_minutes(when: str) -> int | None:
    if not _is_clock(when):
        return None
    h, mm = when.split(":")
    return int(h) * 60 + int(mm)


def _match_what(text: str) -> str:
    for rx, label in _WHAT_PATTERNS:
        if rx.search(text):
            return label
    return ""


def _match_place(text: str, place_hints: tuple | list = ()) -> str:
    # 地名固有ヒント(envpack)を先に、generic 施設語を後に(元の順を保つ)。
    for hint in tuple(place_hints) + _GENERIC_PLACE_HINTS:
        if hint in text:
            return hint
    m = _DE_RE.search(text)
    if m:
        return m.group(1)
    return ""


def extract(text: str, *, base_day: int, base_min: int, cal: dict | None = None,
            horizon_days: int = 14, place_hints: tuple | list = ()) -> list[dict]:
    """発話テキストから未来の予定を抽出する(決定論・乱数不使用)。

    未来の日時が1つも無ければ空リスト。相対日は calendar で絶対 day_index に解決する
    (calendar OFF でも build_cfg の既定 start_date を基準に経過日で解決=祝日概念なし)。
    返す各要素は {"day","when","what","place"}(with / src_step は記入側が付与する)。
    """
    if not text:
        return []
    cal = cal or _cal.build_cfg(None)
    t = str(text).translate(_ZEN)
    day = _match_day(t, base_day, cal)
    when = _match_time(t)
    if day is None:
        if _is_clock(when):                          # 時刻だけ=今日の予定
            day = base_day
        else:
            return []                                # 未来の日時なし
    if day < base_day or day > base_day + horizon_days:
        return []                                    # 過去 / 期間外は記入しない
    if day == base_day:                              # 当日は「時間指定あり」のみ予定化
        if not when:                                 # 今日で時刻・時間帯なし=予定ではない
            return []
        if _is_clock(when):                          # 今日かつ既に過ぎた時刻は記入しない
            mins = _clock_minutes(when)
            if mins is not None and mins <= base_min % 1440:
                return []
    return [{"day": int(day), "when": when,
             "what": _match_what(t), "place": _match_place(t, place_hints)}]


# ---------------------------------------------------------------- 帳簿操作
def _dedup_key(appt: dict) -> tuple:
    return (int(appt["day"]), str(appt["when"]), str(appt.get("what", "")),
            tuple(sorted(int(i) for i in appt.get("with", []))))


def record(agent, appt: dict, *, step: int, max_items: int = 8) -> bool:
    """予定を agent.schedule に追加する(重複は追加しない)。追加したら True。

    同じ day/when/what/相手 は重複とみなす。件数が上限を超えたら過去側(day, src_step
    の小さい順)から間引く。乱数は引かない(決定論)。
    """
    sched = agent.schedule
    key = _dedup_key(appt)
    for e in sched:
        if _dedup_key(e) == key:
            return False
    sched.append({
        "day": int(appt["day"]), "when": str(appt["when"]),
        "what": str(appt.get("what", "")), "place": str(appt.get("place", "")),
        "with": [int(i) for i in appt.get("with", [])], "src_step": int(step)})
    if len(sched) > max_items:                        # 上限超過=古い順に間引く
        sched.sort(key=lambda e: (e["day"], e["src_step"]))
        del sched[:len(sched) - max_items]
    return True


def gc(agent, current_day: int) -> None:
    """過去の予定(day < 当日)を失効させる(当日経過で自動 GC)。"""
    sched = getattr(agent, "schedule", None)
    if sched:
        agent.schedule[:] = [e for e in sched if int(e["day"]) >= current_day]


# ---------------------------------------------------------------- プロンプト1行
def _when_sort(when: str) -> int:
    if _is_clock(when):
        return _clock_minutes(when)
    return _BAND_MIN.get(when, _BAND_MIN[""])


def _day_label(dd: int) -> str:
    return ("今日" if dd == 0 else "明日" if dd == 1 else "明後日" if dd == 2
            else f"{dd}日後")


def _sentence(e: dict, today: int) -> str:
    """1件の予定を自然な句にする(例: 明日15:00に駅前で会う約束)。"""
    label = _day_label(int(e["day"]) - today)
    when = e["when"]
    where = f"{e['place']}で" if e["place"] else ""
    what = e["what"]
    body = f"{what}約束" if what else "予定がある"
    return f"{label}{when}に{where}{body}"


def next_line(agent, *, today: int, horizon_days: int = 14) -> str | None:
    """近い将来(今日〜horizon_days 先)の直近1件を1行にする。無ければ None。"""
    cands = [e for e in getattr(agent, "schedule", [])
             if today <= int(e["day"]) <= today + horizon_days]
    if not cands:
        return None
    cands.sort(key=lambda e: (int(e["day"]), _when_sort(e["when"]), e["src_step"]))
    return "予定: " + _sentence(cands[0], today) + "。"


def today_line(agent, *, today: int) -> str | None:
    """当日の予定(最大3件)を1行にする(朝の計画プロンプト用)。無ければ None。"""
    todays = [e for e in getattr(agent, "schedule", []) if int(e["day"]) == today]
    if not todays:
        return None
    todays.sort(key=lambda e: (_when_sort(e["when"]), e["src_step"]))
    segs = []
    for e in todays[:3]:
        where = f"{e['place']}で" if e["place"] else ""
        what = e["what"] or "予定"
        segs.append(f"{e['when']}{where}{what}".strip())
    return "今日の予定: " + "、".join(segs)
