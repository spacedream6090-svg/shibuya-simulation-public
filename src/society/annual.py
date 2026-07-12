"""年中行事・群集(現実ギャップ Wave G4 2026-07-07。ユーザー要望)。

暦(world/calendar.py)の上に文化的リズムを載せる層。暦の特定日に命名された年中行事
(正月・ハロウィン・年末など)を発生させ、プロンプトに文脈を注入する(「今日はハロウィン
です。」)。群集フラグ(crowd)のある行事日には自由時間の行き先をスクランブル交差点付近の
集会ノードへ確率的に寄せ(渋谷固有のハロウィン型群集)、集中が閾値を超えたら crowd_surge を
記録する(=世界改変・規制の実舞台)。

★ このファイルは src/society 直下(engine/cognition/world の静的検査対象外)なので、行事名・
  群集ロジックはここに置いてよい(no-fingerprint 契約に触れない)。ただし本モジュールは因子語
  (構成概念名)を扱わない=群集の混雑は発火系(grievance/drive)に一切接続しない(雰囲気密度のみ)。

R1 呼数不変: 群集は新しい LLM エージェントを足さず、generate() を1回も追加しない。実装は
  (a) プロンプト文脈注入(内容のみ=event_line)、
  (b) routine 自由行動の行き先バイアス(専用 stream "crowd"=既存 draw 順を汚さない=決定論)、
  (c) 非LLM のアンビエント群集密度(可視化/雰囲気=発火に食わせない)の3つに限る。

既定 OFF(enabled=false)= event_line は None、群集バイアスも surge も無し(現行挙動と完全同一・
乱数消費不変=ゴールデン golden_baseline_l1.json を守る)。年中行事の発生は暦から純関数導出(乱数不要)。
"""
from __future__ import annotations

from .observer.schema import Event
from .world import calendar as _calendar

# 既定の年中行事(名前, 月, 日, 群集フラグ)。ON かつ world.calendar 有効時のみ効く。
#   正月(1/1)・年末(12/31)は文脈注入のみ、ハロウィン(10/31)は渋谷固有の群集を伴う。
_DEFAULT_EVENTS: list[dict] = [
    {"name": "正月", "month": 1, "day": 1, "crowd": False},
    {"name": "ハロウィン", "month": 10, "day": 31, "crowd": True},
    {"name": "年末", "month": 12, "day": 31, "crowd": False},
]


def build_cfg(raw: dict | None) -> dict:
    """conf の annual_events ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    events_raw = raw.get("events", None)
    if events_raw is None:
        events = [dict(e) for e in _DEFAULT_EVENTS]
    else:
        events = []
        for e in (events_raw or []):
            e = dict(e)
            events.append({
                "name": str(e.get("name", "")),
                "month": int(e.get("month", 1)),
                "day": int(e.get("day", 1)),
                "crowd": bool(e.get("crowd", False)),
            })
    return {
        "enabled": bool(raw.get("enabled", False)),
        "events": events,
        "crowd_threshold": int(raw.get("crowd_threshold", 8)),
        "crowd_bias": float(raw.get("crowd_bias", 0.7)),
        "crowd_radius_m": float(raw.get("crowd_radius_m", 60.0)),
        "ambient": bool(raw.get("ambient", False)),
        "ambient_per_surge": int(raw.get("ambient_per_surge", 3000)),
    }


def gathering_node(city, dests: list[str]) -> str | None:
    """集会ノード = スクランブル交差点(地図ローカル原点 (0,0))に最も近い主要ノード(destination)。

    渋谷ハロウィン型群集の集約先。決定論(距離→ノード名の順で一意)。候補が無ければ None。
    地図の原点はスクランブル交差点(conf の座標系注記)なので原点最近傍=スクランブル付近。
    """
    if not dests:
        return None

    def _d2(node: str) -> float:
        x, y = city.node_xy(node)
        return float(x) * float(x) + float(y) * float(y)

    return min(sorted(dests), key=lambda n: (_d2(n), n))


def event_today(cfg: dict, cal: dict | None, sim_min: int) -> dict | None:
    """その sim_min の当日に該当する年中行事(最初の1件)。無ければ None。

    annual OFF / calendar OFF(日付が定まらない)なら None(=年中行事は暦の上に乗る)。純関数・乱数不要。
    """
    if not cfg["enabled"] or not cal or not cal.get("enabled"):
        return None
    d = _calendar.date_of(cal, sim_min)
    for e in cfg["events"]:
        if e["month"] == d.month and e["day"] == d.day:
            return e
    return None


def event_line(cfg: dict, cal: dict | None, sim_min: int) -> str | None:
    """プロンプトへ注入する年中行事1行。当日でなければ None(=注入しない=不変)。

    例: "今日はハロウィンです。"
    """
    e = event_today(cfg, cal, sim_min)
    if e is None:
        return None
    return f"今日は{e['name']}です。"


def check_surge(sim, step: int, sim_min: int) -> None:
    """群集(渋谷ハロウィン型)の集中を観測し crowd_surge を1日1回記録する(非LLM・決定論)。

    集会ノード周辺(crowd_radius_m 内)に居る街内エージェント数が閾値超で crowd_surge を記録。
    発火系(grievance/drive)には一切接続しない(混雑=雰囲気密度のみ=観測/可視化)。乱数は引かない。
    群集日でない / annual OFF / 集会ノード無し / その日は記録済み なら完全 no-op(不変)。
    """
    event = getattr(sim, "today_crowd_event", None)
    node = getattr(sim, "crowd_node", None)
    if not event or node is None:
        return
    day = sim_min // 1440
    if getattr(sim, "_crowd_surge_day", -1) == day:
        return                                    # その日は記録済み(1日1回=最初の閾値超で確定)
    cfg = sim.annualcfg
    cx, cy = sim.city.node_xy(node)
    r = float(cfg["crowd_radius_m"])
    r2 = r * r
    count = sum(1 for a in sim.agents
                if a.loc != "outside" and not a.sleeping
                and (float(a.x) - cx) ** 2 + (float(a.y) - cy) ** 2 <= r2)
    if count < int(cfg["crowd_threshold"]):
        return
    sim._crowd_surge_day = day
    payload = {"node": node, "level": int(count), "event": event}
    if cfg["ambient"]:                            # 非LLM の雰囲気密度(可視化用。発火に食わせない)
        payload["ambient"] = int(count) * int(cfg["ambient_per_surge"])
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="crowd_surge", x=float(cx), y=float(cy),
                         payload=payload))
