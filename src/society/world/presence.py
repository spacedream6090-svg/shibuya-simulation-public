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

層別クォータ(DP-U3 案A・conf `pool.tier_quota.enabled`・**既定 OFF = 上の層優先と完全一致**)
------------------------------------------------------------------------------------------
正典: docs/plans/proposal-dp-u3-observe-250k.md §1.2「案 A の中身」。

層優先は「優先順に埋めて途中で break」なので、上位層が cap を食い切ると**下位層が構造的に
全滅**する。第91 の実測(名簿 100万・cap 25万・平日)では resident 30,034 + duty 986 で残
218,980 を workday_shift(253,702)が埋め切り、**cadence / stochastic が 1 人も入らない**
= 25万人の街に来街者がゼロ、という破綻が起きた。

ON にすると cap を層ごとのクォータに分配する:

    quota[t] = round(present_cap × eligible[t] / Σ eligible)     # 当日資格者の比率を保存
    present  = ∪_t rank_t(eligible[t])[:quota[t]]                # 溢れた層だけ当日ランキングで切る

- **乱数追加ゼロ**: 端数処理は整数演算の最大剰余法(同値タイブレーク = 層の固定順)。
  層内の並び順・選抜規則は層優先のときと同一(既存 `presence_rank` stream)。
- k 非依存・trait 非依存・resume 不変は不変(入力は当日の資格者数と層だけ)。
- ★ON では resident も比率で切られる(cap < 資格者総数のとき)。「上位層は毎日不変」という
  層優先の性質は ON では成り立たない。これは構成比を保存することの代償であり設計どおり。
- 週末の総在場が現実の 28% しかない問題(提案書 §1.1)は**名簿側の較正問題**であって
  クォータでは直らない(クォータは比率を保存するだけで人を増やさない)。
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


def _ranked(ids: list[str], day: int, hub) -> list[str]:
    """層内の当日ランキング(既存規則。溢れた層の部分集合を採る順序)。"""
    return sorted(
        ids,
        key=lambda pid: (float(hub.stream("presence_rank", pid, int(day)).random()), pid))


def quota_by_ratio(sizes: list[int], cap: int) -> list[int]:
    """層別クォータ(DP-U3 案A)。**整数演算のみ・乱数ゼロ・同入力同出力**。

    quota[t] = round(cap × sizes[t] / Σ sizes) を**最大剰余法**で合計 = min(cap, Σ sizes) に
    正規化する。剰余が同値なら層 index(= 層名の固定順)の小さい方を優先=タイブレークも決定論。
    資格者数がクォータ未満になる層は**全員だけ**入れ、余った枠は残りの層へ同じ規則で再配分する
    (毎周 1 人以上配るので必ず停止する)。cap ≧ Σ sizes なら全層まるごと(= 切らない)。
    """
    n = len(sizes)
    quota = [0] * n
    room = [max(0, int(s)) for s in sizes]          # 各層の未割当の資格者
    remaining = max(0, int(cap))
    while remaining > 0:
        active = [t for t in range(n) if room[t] > 0]
        if not active:
            break                                    # 資格者を配り切った(cap > Σ eligible)
        total = sum(room[t] for t in active)
        if remaining >= total:                       # 全層とも取り分 ≧ 資格者 → 全員入れて終了
            for t in active:
                quota[t] += room[t]
                room[t] = 0
            break
        base = [(remaining * room[t]) // total for t in active]
        order = sorted(range(len(active)),           # 端数の大きい層から 1 ずつ(同値は層順)
                       key=lambda i: (-((remaining * room[active[i]]) % total), active[i]))
        for i in order[:remaining - sum(base)]:
            base[i] += 1
        moved = 0
        for i, t in enumerate(active):
            give = min(base[i], room[t])             # 資格者不足の層は全員だけ(残枠は次周で再配分)
            quota[t] += give
            room[t] -= give
            moved += give
        if moved <= 0:
            break                                    # 前進しない(到達しない)= 無限ループ防止
        remaining -= moved
    return quota


def present_for_day(recs, day: int, present_cap: int, hub, weekday: int,
                    tier_quota: bool = False) -> list[str]:
    """day の present ペルソナ id を present_cap まで選ぶ(決定論・純関数)。

    引数:
      recs: PresenceRec の反復可能(PoolStore.presence_records())。
      day: 日インデックス(0 起点)。
      present_cap: 日次総在場の上限。
      hub: RngHub(stream 専用。k/step 非参照)。
      weekday: 当日の曜日(0=Mon..6=Sun)。呼び出し側が暦から計算して渡す。
      tier_quota: False(既定)= 従来の層優先 break。True = 層別クォータ(module docstring)。

    返り: present の pid の**ソート済み**リスト(同 (seed, day) で常に同一集合・同一順)。
    """
    cap = int(present_cap)
    tiers: list[list[str]] = [[] for _ in range(_N_TIERS)]
    for rec in recs:
        if _eligible(rec, day, weekday, hub):
            tiers[_TIER.get(rec.key, _N_TIERS - 1)].append(rec.pid)

    present: list[str] = []
    if tier_quota:
        # 層別クォータ: cap を当日資格者の比率で分配し、溢れた層だけ当日ランキングで切る。
        quota = quota_by_ratio([len(ids) for ids in tiers], cap)
        for tier in range(_N_TIERS):
            ids, q = tiers[tier], quota[tier]
            if not ids or q <= 0:
                continue
            present.extend(ids if len(ids) <= q else _ranked(ids, day, hub)[:q])
        present.sort()
        return present

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
            present.extend(_ranked(ids, day, hub)[:room])
            break
    present.sort()
    return present
