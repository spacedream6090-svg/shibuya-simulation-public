"""日次プレゼンス純関数(W2 P3 / ローテーション機構)。

正典: docs/plans/w2-execution-plan.md §4 P3 / docs/plans/persona-pool.md §5「presence 純関数」・§9。

100万ペルソナプール(P5)から「今日この街にいる人(present)」を**決定論**で選ぶ。
選択は `hub.stream("presence", persona_id, day)` ベースの層別純関数であり、

  - **k 非依存**: 経験→内部状態のゲイン k を一切読まない。
  - **trait 非依存**: 個体の trait/beliefs を読まない。入力は暦(day/weekday)+ ペルソナ固有の
    presence 属性(層・visit_rate・cadence・duty)+ 専用 stream のみ。
  - **resume 不変**: day のみが可変入力。他者の実行順・生存状態に依存しない(RngHub がステートレス)。

→ k 掃引の全条件で「同じ人が同じ日に present」(共通乱数)= presence が k* を交絡しない(R1)。

日内の出入り(朝流入→夕退出)は既存の visitor 入退場機構が担う。ここは**日次選択のみ**を行う。
在場実測曲線(data/jinryu/shibuya_concurrent_144step_curve.csv)は v1 では使わない(将来拡張点)。

層別規則(P5 の presence キー):
  - resident       (L1 住民・L5 議員): 毎日 present。
  - duty           (L5 役割=駅員/運転士/警察/配信者): 役割で常時(当番。duty_pattern.days)。
  - workday_shift  (L2 域内従業者): 平日 + シフト暦。
  - cadence        (L3 定期来街=学生/常連): 通学日・週次常連の暦。
  - stochastic     (L4 非定期来街): visit_rate で抽選(re-visit 性 = 同一人物が周期で再訪)。

日次の総在場数は `present_cap` を上限に**層優先**で充足する:
  resident/duty/workday > cadence > stochastic。上限で溢れた層のみ、当日固有の決定論ランキングで
  部分集合を採る(→ 日ごとに顔ぶれが入れ替わる=ローテーション)。上位層は溢れない限り毎日不変。
"""
from __future__ import annotations

from dataclasses import dataclass

# presence キー → 優先度 tier(小さいほど先に充足=cap で落とされにくい)。
# resident/duty/workday は「毎日概ね同じ顔」(回転が薄い)。cadence/stochastic が回転の主層。
_TIER = {"resident": 0, "duty": 1, "workday_shift": 2, "cadence": 3, "stochastic": 4}
_N_TIERS = 5

_WEEKDAYS = frozenset({0, 1, 2, 3, 4})     # Mon..Fri(day % 7 で day0=Monday とする)


@dataclass
class PresenceRec:
    """presence 純関数が読む**スリムな**ペルソナ記述子(full record は読まない)。

    PoolStore が P5 シャードから抽出する(pool.py::_slim)。ここに beliefs/traits/k は入れない。
    """
    pid: str                    # ペルソナ id(例 "L4_00012345")= agent.id の安定源
    key: str                    # presence キー(resident/duty/workday_shift/cadence/stochastic)
    work_days: str = ""         # workday_shift: "mon-fri" | "all" | ...
    cadence: str = ""           # cadence: "school_day" | "weekly_N"
    visit_rate: float = 0.0     # stochastic: 来訪確率/日(0.003〜0.06)
    revisit: bool = False       # stochastic: 再訪する個体か(周期で同一人物が戻る)
    duty_days: str = "all"      # duty: duty_pattern.days("all" | ...)


def _days_match(spec: str, weekday: int) -> bool:
    """暦の曜日仕様と当日 weekday の一致(未知仕様は平日扱いに後退)。"""
    if not spec or spec == "all":
        return True
    if spec in ("mon-fri", "school_day"):
        return weekday in _WEEKDAYS
    return weekday in _WEEKDAYS


def _weekly_days(hub, pid: str, week: int, n: int) -> set[int]:
    """常連(weekly_N): その週に来る n 曜日を決定論で確定(週固有 stream)。"""
    n = max(1, min(int(n), 7))
    g = hub.stream("presence_cadence", pid, int(week))
    return {int(x) for x in g.choice(7, size=n, replace=False)}


def _revisit_present(hub, pid: str, day: int) -> bool:
    """再訪個体の周期プレゼンス。period/phase は**day 非依存**の固定値(同一人物が規則的に戻る)。

    → 「二度目の来街で前回の場所を思い出す観光客」「行きつけの常連」の観測が自然に立ち上がる。
    """
    g = hub.stream("presence_revisit", pid)      # day を渡さない=個体固定の周期
    period = 3 + int(g.integers(0, 8))           # 3..10 日周期
    phase = int(g.integers(0, period))
    return (day - phase) % period == 0


def _eligible(rec: PresenceRec, day: int, weekday: int, hub) -> bool:
    """当日 present 資格があるか(層別・純関数・k/trait 非参照)。"""
    key = rec.key
    if key == "resident":
        return True
    if key == "duty":
        return _days_match(rec.duty_days, weekday)
    if key == "workday_shift":
        return _days_match(rec.work_days, weekday)
    if key == "cadence":
        cad = rec.cadence or "school_day"
        if cad.startswith("weekly_"):
            try:
                n = int(cad.split("_", 1)[1])
            except (ValueError, IndexError):
                n = 2
            return weekday in _weekly_days(hub, rec.pid, day // 7, n)
        return _days_match(cad, weekday)
    if key == "stochastic":
        if rec.revisit and _revisit_present(hub, rec.pid, day):
            return True
        return float(hub.stream("presence", rec.pid, int(day)).random()) < rec.visit_rate
    return False


def present_for_day(recs, day: int, present_cap: int, hub, weekday: int) -> list[str]:
    """day の present ペルソナ id を層優先で present_cap まで選ぶ(決定論・純関数)。

    引数:
      recs: PresenceRec の反復可能(PoolStore.presence_records())。
      day: 日インデックス(0 起点)。
      present_cap: 日次総在場の上限。
      hub: RngHub(stream 専用。k/step 非参照)。
      weekday: 当日の曜日(0=Mon..6=Sun)。呼び出し側が暦から計算して渡す。

    返り: present の pid の**ソート済み**リスト(同 (seed, day) で常に同一集合・同一順)。
    """
    cap = int(present_cap)
    tiers: list[list[str]] = [[] for _ in range(_N_TIERS)]
    for rec in recs:
        if _eligible(rec, day, weekday, hub):
            tiers[_TIER.get(rec.key, _N_TIERS - 1)].append(rec.pid)

    present: list[str] = []
    for tier in range(_N_TIERS):
        ids = tiers[tier]
        if not ids:
            continue
        room = cap - len(present)
        if room <= 0:
            break
        if len(ids) <= room:
            present.extend(ids)
        else:
            # cap で溢れる層: 当日固有の決定論ランキングで部分集合を採る(=日ごとに入れ替わる)。
            ranked = sorted(
                ids,
                key=lambda pid: (float(hub.stream("presence_rank", pid, int(day)).random()), pid))
            present.extend(ranked[:room])
            break
    present.sort()
    return present
