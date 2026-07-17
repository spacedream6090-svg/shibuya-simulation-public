"""routine 層(非LLM の既定行動)= 移動・滞在・日課(通勤/食事/帰宅/就寝)のみ。

★ 発話・思考は必ず LLM。routine は定型文を一切話さない。
   会話・投稿・DM の起動は欲求駆動発火(cognition/drive.py + scheduler の
   _phase_drive)に一本化(Phase A、2026-07-04)。routine は身体の日課だけを返す。
日課(v3): 職場(実在 POI)への通勤・勤務、食事時間帯の飲食店行き、
          実在の住宅建物への帰宅・就寝(個体別時刻。記憶整理 LLM の分散が目的の一つ)。
EPR(Song 2010)は自由時間の行き先選択。
すべて乱数と物理情報のみで決める(因子は見ない = no-fingerprint)。
"""
from __future__ import annotations

import math

import numpy as np

from .. import commerce as _commerce
from .. import disaster as _disaster
from .. import diversity as _diversity
from .. import household as _household
from .. import inner_life as _inner
from .. import media
from ..factors import update as factor_update
from ..observer.schema import Event
from ..world import calendar as _calendar

RHO = 0.6
GAMMA = 0.21

MEAL_WINDOWS = [(11 * 60 + 30, 13 * 60 + 30), (18 * 60, 21 * 60)]


def _minutes_of_day(sim_min: int) -> int:
    return sim_min % 1440


def _cal(sim) -> dict | None:
    """当日の暦 cfg(world.calendar)。未設定/既定 OFF なら weekday ゲートは無効=不変。"""
    return getattr(sim, "calendarcfg", None)


def bedtime_reached(agent, sim_min: int) -> bool:
    """就寝時刻(個体差あり)を過ぎたか。窓は就寝時刻から4時間。"""
    m = _minutes_of_day(sim_min)
    start = agent.bedtime_min
    return ((m - start) % 1440) < 240


def in_work_window(agent, sim_min: int, cal: dict | None = None) -> bool:
    """本業・登校の勤務時間帯か。cal(world.calendar cfg)を渡し weekday_work=true のときは
    平日(非休日)だけに絞る。cal=None / 無効 / weekday_work=false なら現行=毎日勤務(不変)。
    バイト(in_part_time_window)は平日ゲートの対象外(既存の曜日挙動を変えない)。
    病気(健康 H1、既定 OFF=agent.sick 立たず不変)のときは欠勤=勤務時間外扱い(False)。"""
    if getattr(agent, "sick", False):              # 病気=欠勤(健康 H1。既定 OFF は False)
        return False
    if agent.work_start_min < 0:
        return False
    if cal is not None and cal.get("enabled") and cal.get("weekday_work") \
            and not _calendar.is_workday(cal, sim_min):
        return False
    m = _minutes_of_day(sim_min)
    return agent.work_start_min <= m < agent.work_end_min


def in_part_time_window(agent, sim_min: int) -> bool:
    """バイトのシフト時間帯か(曜日+時間帯)。経済 v0: 学生・フリーター等のみ part_time を持つ。

    病気(健康 H1、既定 OFF=agent.sick 立たず不変)のときは欠勤=シフト時間外扱い(False)。"""
    if getattr(agent, "sick", False):              # 病気=欠勤(健康 H1。既定 OFF は False)
        return False
    pt = agent.part_time
    if not pt:
        return False
    day = (sim_min // 1440) % 7
    if day not in pt["days"]:
        return False
    m = _minutes_of_day(sim_min)
    return pt["start_min"] <= m < pt["end_min"]


def in_meal_window(sim_min: int) -> bool:
    m = _minutes_of_day(sim_min)
    return any(a <= m < b for a, b in MEAL_WINDOWS)


def _lynch_destination(agent, sim, rng: np.random.Generator) -> str:
    """Lynch 都市イメージ プラグイン(既定 OFF、ON 時のみ・新 stream で呼ばれる)。

    自由時間の行き先を landmark_score(engine が事前構築した「イメージアビリティ」の
    不透明スコア)で重み付けサンプルする。この関数は score 値を見るだけで因子名は知らない。
    候補が無ければ現在地(=stay)。既定 decide の draw 順は一切汚さない(専用 rng)。
    """
    score = sim.landmark_score
    cands = [d for d in sim.dests if d != agent.node and d in score]
    if not cands:
        return agent.node
    weights = np.array([max(0.0, float(score[d])) for d in cands], dtype=float)
    total = weights.sum()
    if total <= 0.0:
        return cands[int(rng.integers(len(cands)))]
    weights /= total
    return cands[int(rng.choice(len(cands), p=weights))]


def choose_destination(agent, dests: list[str], rng: np.random.Generator) -> str:
    visited = [d for d in dests if agent.visits[d] > 0 and d != agent.node]
    unvisited = [d for d in dests if agent.visits[d] == 0 and d != agent.node]
    s = max(1, len(visited))
    if unvisited and (not visited or rng.random() < RHO * s ** (-GAMMA)):
        return unvisited[int(rng.integers(len(unvisited)))]
    if visited:
        weights = np.array([agent.visits[d] for d in visited], dtype=float)
        weights /= weights.sum()
        return visited[int(rng.choice(len(visited), p=weights))]
    others = [d for d in dests if d != agent.node]
    return others[int(rng.integers(len(others)))] if others else agent.node


def _choose_mode(agent, sim, dest: str, rng: np.random.Generator) -> str:
    ax, ay = sim.city.node_xy(agent.node)
    bx, by = sim.city.node_xy(dest)
    dist = math.hypot(ax - bx, ay - by)
    if dist > 400:
        if agent.has_car and rng.random() < 0.7:
            return "car"
        if agent.has_bicycle and rng.random() < 0.8:
            return "bicycle"
    return "walk"


# ---- 健康・疲労・病気(後続波 H1、既定 OFF=agent.sick/withdrawn 立たず不変)----
def _sick_home(agent, sim, sim_min: int, step: int, rng: np.random.Generator) -> dict:
    """病気で在宅療養: 自宅へ帰り留まる(欠勤=在宅)。呼び出し側が agent.sick を確認済みのときだけ来る。

    夜は上流の bedtime 分岐が睡眠を処理するので、ここは日中の「自宅に居る/帰る」だけを担う。
    自宅の建物があれば入って在宅、無ければ自宅前の路上で静養する。乱数は帰路の mode 選択のみ
    (health ON のときだけ引かれる=既定 OFF の draw 順は不変)。"""
    if agent.building:
        if agent.building == agent.home_building:
            return {"type": "stay"}                # 自宅で療養(在宅)
        return {"type": "exit_building"}           # 他所の建物 → 出て帰路へ
    if agent.route:
        return {"type": "continue"}                # 帰宅中
    if agent.home_node and agent.node != agent.home_node:
        return {"type": "move_to", "dest": agent.home_node, "stay_steps": 0,
                "mode": _choose_mode(agent, sim, agent.home_node, rng),
                "homing": True}
    if agent.home_building and sim.city.has_building(agent.home_building):
        return {"type": "enter_building", "building": agent.home_building,
                "floor": agent.home_floor, "stay_steps": 3}   # 自宅前 → 入って在宅
    return {"type": "stay"}                        # 自宅建物なし → 路上で静養


# ---- 朝の一日計画(routine の行き先の土台。ユーザー要望 2026-07-06)----
# meetup = 待ち合わせ(像・名所 等の landmark)。会えたかは既存の対面機構が拾う。
_PLAN_CAT = {"meal": "food", "shop": "shop", "leisure": "leisure",
             "park": "leisure", "work": "office", "meetup": "landmark",
             "visit": "landmark"}
_PLAN_ACTIVITY = {"meal": "eating", "shop": "shopping"}


def _time_band(sim_min: int) -> str:
    """計画の時間帯ラベル(朝/昼/午後/夕方/夜)。深夜は就寝帯なので夜に畳む。"""
    h = (sim_min % 1440) // 60
    if 5 <= h < 11:
        return "朝"
    if 11 <= h < 14:
        return "昼"
    if 14 <= h < 17:
        return "午後"
    if 17 <= h < 19:
        return "夕方"
    return "夜"


def _resolve_plan_dest(agent, sim, item, rng, sim_min: int):
    """予定の place/what から行き先ノードを解く。place は POI 名の部分一致で優先解決し、
    無ければ what のカテゴリから既存ロジックでサンプル、それも無ければ EPR の行き先。
    有料カテゴリは残高で除外(食事の行き先選択と同じ扱い)。商業 H3 ON 時は閉店中 POI も除外。"""
    place = str(item.get("place") or "").strip()
    if place:
        matches = [p for p in sim.city.poi_list
                   if place in p["name"] and p["node"] != agent.node
                   and agent.money >= _poi_price(sim, p)]
        matches = _commerce.filter_open(sim, matches, sim_min)   # 閉店中 POI を除外(H3、OFF=不変)
        if matches:
            return matches[int(rng.integers(len(matches)))]["node"]
    what = str(item.get("what") or "").strip()
    if what == "home":
        return agent.home_node or None
    cat = _PLAN_CAT.get(what)
    if cat:
        pool = [p for p in sim.city.pois_by_cat(cat)
                if p["node"] != agent.node and agent.money >= _poi_price(sim, p)]
        pool = _commerce.filter_open(sim, pool, sim_min)         # 閉店中 POI を除外(H3、OFF=不変)
        if pool:
            return pool[int(rng.integers(len(pool)))]["node"]
    return choose_destination(agent, sim.dests, rng)


def _plan_move(agent, sim, sim_min: int, step: int, rng) -> dict | None:
    """現在の時間帯に一致する未消化の予定があれば、その行き先への move を組む。

    day_plan が空(=計画なし。planning 無効時も含む)のときは乱数を一切引かずに None
    を返す(既定挙動の再現性を保つ)。1項目は1回だけ消化する(解決可否に関わらず)。"""
    plan = getattr(agent, "day_plan", None)
    if not plan:
        return None
    band = _time_band(sim_min)
    item = next((it for it in plan
                 if not it.get("done") and it.get("when") == band), None)
    if item is None:
        return None
    item["done"] = True
    dest = _resolve_plan_dest(agent, sim, item, rng, sim_min)
    if not dest or dest == agent.node:
        return None
    action = {"type": "move_to", "dest": dest,
              "stay_steps": int(rng.integers(2, 5)),
              "mode": _choose_mode(agent, sim, dest, rng)}
    activity = _PLAN_ACTIVITY.get(str(item.get("what") or ""), "")
    if activity:
        action["activity"] = activity
    return _augment_ride(agent, sim, action, step, sim_min)


# ---- 交通機関: タクシー+簡易バス(ユーザー要望 2026-07-06)----
def _augment_ride(agent, sim, action: dict, step: int, sim_min: int) -> dict:
    """行き先 move に交通機関(タクシー/バス)を重ねる。乗るときは車速で走り、到着時に
    運賃を払う(scheduler が action["ride"] を読んで課金)。乗らなければ action は不変。"""
    extra = _ride_extra(agent, sim, action["dest"], step, sim_min)
    if extra:
        action["mode"] = extra[0]
        action["ride"] = extra[1]
    return action


def _ride_extra(agent, sim, dest, step: int, sim_min: int):
    """(mode, ride) を返す。ride = {mode, fare, from, to}。乗らなければ None。

    無効時は乱数を一切引かずに None(既定=旧挙動の再現性)。バス→タクシーの順に判定する
    (バスは既定 OFF。有効な合成/簡易路線で停留所が近いときだけ乗れる)。自家用車保有者は
    タクシーを使わない。来街者も使える。"""
    ride = getattr(sim, "ridecfg", None)
    if not ride:
        return None
    rb = getattr(sim, "rulebook", None)          # 制度DSL: taxi/bus 運賃への fee ルール
    buses = getattr(sim, "buses", None)
    bus = ride["bus"]
    if bus["enabled"] and buses is not None and buses.serves(sim_min):
        leg = buses.find_ride(agent.node, dest, sim.city)
        if leg is not None:
            fare = float(bus["fare"])
            if rb is not None:
                fare = rb.fee_price("bus", fare)
            return ("car", {"mode": "bus", "fare": fare,
                            "from": leg["from"], "to": leg["to"]})
    taxi = ride["taxi"]
    if not taxi["enabled"] or agent.has_car:
        return None
    ax, ay = sim.city.node_xy(agent.node)
    bx, by = sim.city.node_xy(dest)
    dist = math.hypot(ax - bx, ay - by)
    if dist < float(taxi["min_dist_m"]):
        return None
    fare = float(taxi["base_fare"]) + float(taxi["per_km"]) * (dist / 1000.0)
    if rb is not None:
        fare = rb.fee_price("taxi", fare)
    if agent.money < fare:
        return None
    if sim.hub.stream("taxi", agent.id, step).random() >= float(taxi["prob"]):
        return None
    return ("car", {"mode": "taxi", "fare": round(fare, 1),
                    "from": agent.node, "to": dest})


# ---- 制度DSL(ホワイトリスト型ルール。ユーザー構想 2026-07-06)----
def _curfew_suppressed(agent, sim, dest_node: str, sim_min: int, step: int) -> bool:
    """curfew ルールが今この時刻×行き先カテゴリを抑制するか(禁止でなく確率的抑制)。

    抑制対象の curfew が無い/時間外/カテゴリ非該当なら乱数を一切引かず False(既定不変)。
    係数 w(0..1)= 通過確率。w>=1.0 は抑制なし(恒等)。
    """
    rb = getattr(sim, "rulebook", None)
    if rb is None or not rb.has_curfew():
        return False
    hour = (sim_min % 1440) // 60
    dest_cats = {p.get("cat") for p in sim.city.pois_at_node(dest_node)}
    w = rb.curfew_weight(hour, dest_cats)
    if w >= 1.0:
        return False
    rng = sim.hub.stream("curfew", agent.id, step)
    return rng.random() >= w                      # (1-w) の確率で行き先を取りやめる


def _crowd_dest(agent, sim, step: int) -> str | None:
    """群集日(大規模行事型): 自由時間の行き先を集会ノード(地図原点付近)へ確率的に寄せる。

    群集無効 / 非群集日(today_crowd_event なし)/ 集会ノードが現在地 / 寄せない抽選 なら、乱数を
    一切引かず None(=既定不変・通常の行き先選択へ)。専用 stream 'crowd' のみ使用=既存 draw 順を
    汚さない(決定論・ゴールデン保護)。行き先(移動)だけを変え、発火判断・LLM 呼数は増やさない。
    勤務・睡眠・帰宅など既定の拘束は上流で処理済み=ここは自由行動の行き先のみを対象にする。
    """
    node = getattr(sim, "crowd_node", None)
    if node is None or not getattr(sim, "today_crowd_event", None):
        return None
    if node == agent.node:
        return None
    crng = sim.hub.stream("crowd", agent.id, step)
    if crng.random() >= float(sim.annualcfg["crowd_bias"]):
        return None
    return node


def _weekly_boost_dest(agent, sim, step: int) -> str | None:
    """weekly_event の発火日に、その場所を余暇の行き先候補としてブーストする。

    今日ブーストするノードが無ければ乱数を引かず None(既定不変)= 通常の行き先選択へ。
    """
    rb = getattr(sim, "rulebook", None)
    if rb is None or not rb.today_boost:
        return None
    cands = [n for n in rb.today_boost if n != agent.node]
    if not cands:
        return None
    brng = sim.hub.stream("rule_boost", agent.id, step)
    if brng.random() >= rb.cfg["boost_prob"]:
        return None
    return cands[int(brng.integers(len(cands)))]


# ---- 娯楽メディア(TV・動画・ゲーム。在宅の余暇。既定 OFF。バッチD)----
# 設計は docs/lit/media__entertainment-effects.md。効果は最小限:
#   (a)気分修復: セッション終了時に factors seam(cause="media")で grievance を小さく下げる。
#   (b)時間置換: セッション中は在宅で外出・余暇外出・SNS閲覧をしない(機会費用のみ)。
#   (c)prompt_context ON 時のみ: 直近視聴タイトル1行を記憶へ→発火プロンプト文脈に載る。
# 乱数は新 stream "media" だけ。OFF 時は draw を一切引かず既定挙動バイト一致。
# no-fingerprint: このモジュールは性格特性を読まず media.py の precompute 値だけを使う。
def _media_settings(sim) -> dict:
    """media 設定を sim に一度だけキャッシュ(以後は再解析しない)。"""
    mcfg = getattr(sim, "_media_settings", None)
    if mcfg is None:
        mcfg = media.media_cfg(sim.cfg)
        sim._media_settings = mcfg
    return mcfg


def _at_home(agent) -> bool:
    """在宅(自宅の建物の中、または自宅前の路上に静止)か。"""
    if agent.building is not None:
        return agent.building == agent.home_building
    return not agent.route and agent.node == agent.home_node


def _in_media_band(sim_min: int, mcfg: dict) -> bool:
    """現在の時刻が娯楽メディアの時間帯(朝/夜)か。"""
    h = (sim_min % 1440) // 60
    mh, nh = mcfg["morning_hours"], mcfg["night_hours"]
    return (mh[0] <= h < mh[1]) or (nh[0] <= h < nh[1])


def _end_media(agent) -> None:
    agent._media_end = -1
    if getattr(agent, "activity", "") == "media":
        agent.activity = ""


def _media_action(agent, sim, sim_min: int, step: int) -> dict | None:
    """在宅の娯楽メディア視聴セッション。dict(=media_stay で在宅占有)か None(通常行動へ)。

    media_stay は _apply では no-op(その場に留まる)、かつ type!="stay" なので scheduler は
    スマホ閲覧(SNS/ニュース)を挟まない=時間置換(外出も余暇外出もしない)。"""
    mcfg = _media_settings(sim)
    if not mcfg["enabled"] or agent.sleeping or agent.visitor \
            or agent.loc == "outside":
        return None
    # インフラ障害(停電・通信断=H4、既定 OFF は infra_out=False で不変): 在宅娯楽メディアを抑制する
    # (停電で TV/ゲームが使えない)。既に視聴中でも中断。障害が無ければ従来どおり(バイト一致)。
    if _disaster.infra_out(sim):
        if getattr(agent, "_media_end", -1) >= 0:
            _end_media(agent)
        return None
    # 勤務・バイトの時間帯は仕事優先(視聴を始めない・進行中でも中断)。
    if in_work_window(agent, sim_min, _cal(sim)) \
            or in_part_time_window(agent, sim_min):
        if getattr(agent, "_media_end", -1) >= 0:
            _end_media(agent)
        return None
    end = getattr(agent, "_media_end", -1)
    at_home = _at_home(agent)
    # --- セッション終了 → 気分修復(1回)→ 通常行動へ ---
    if end >= 0 and step >= end:
        _end_media(agent)
        factor_update.on_media(agent, mags=sim.mags, step=step,
                               sim_min=sim_min, logger=sim.logger)
        return None
    # --- 継続中 → 在宅なら視聴を続ける(外出・SNS閲覧をしない)---
    if end >= 0:
        if at_home:
            agent.activity = "media"
            return {"type": "media_stay"}
        _end_media(agent)                     # 在宅でなくなった=中断
        return None
    # --- セッションなし → 開始判定(在宅×朝/夜の帯のみ)---
    if not at_home or agent.route or not _in_media_band(sim_min, mcfg):
        return None
    day = sim_min // 1440
    if getattr(agent, "_media_day", -1) != day:
        agent._media_day = day
        agent._media_count = 0
    if agent._media_count >= mcfg["max_sessions_per_day"]:
        return None
    rng = sim.hub.stream("media", agent.id, step)
    prof = media.profile_for(agent, mcfg)
    if rng.random() >= media.start_prob(prof, mcfg):
        return None
    steps = media.session_length(prof, rng, mcfg)
    medium = media.pick_medium(prof, rng)
    agent._media_end = step + steps
    agent._media_count += 1
    agent.activity = "media"
    payload = {"medium": medium, "steps": int(steps), "at": _time_band(sim_min)}
    if mcfg["prompt_context"]:                # 独立フラグ(LLM キャッシュキーに影響)
        title = media.pick_title(medium, rng,
                                 titles=sim.envpackcfg["media"])
        payload["title"] = title
        agent.remember(f"「{title}」({media.medium_jp(medium)})を楽しんだ", kind="media")
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="media_use", x=agent.x, y=agent.y, payload=payload))
    return {"type": "media_stay"}


def decide(agent, step: int, sim, place: str, rng: np.random.Generator,
           has_company: bool) -> dict:
    """routine の行動決定。発話はしない(socialize で LLM へ委譲)。"""
    sim_min = sim.clock.sim_min(step)
    m = _minutes_of_day(sim_min)
    cal = _cal(sim)                         # 暦(weekday_work ゲート用。既定 OFF=不変)

    # ---- 娯楽メディア(在宅の TV/動画/ゲーム。既定 OFF)。就寝より優先し在宅を占有 ----
    media_action = _media_action(agent, sim, sim_min, step)
    if media_action is not None:
        return media_action

    # ---- 就寝時刻: 居住者は自宅(実在の住宅建物)へ、来街者は街の外の家へ帰る ----
    if bedtime_reached(agent, sim_min) and not agent.sleeping:
        if agent.visitor and agent.loc == "street":
            if agent.building:
                return {"type": "exit_building"}
            if agent.route or agent.exit_intent:
                return {"type": "continue"} if agent.route else {"type": "stay"}
            # 帰路: 終電があるうちは駅、終電後は縁のゲートウェイまで歩く
            if sim.transit.has_service(sim_min) and sim.city.station_node:
                dest = sim.city.station_node
            else:
                gates = sim.city.gateways or [agent.home_node]
                dest = gates[agent.id % len(gates)]
            return {"type": "move_to", "dest": dest, "stay_steps": 0,
                    "mode": _choose_mode(agent, sim, dest, rng),
                    "exit": True, "homing": True}
        if agent.building and agent.building == agent.home_building:
            return {"type": "sleep"}               # すでに自宅の中 → 就寝
        if agent.building:
            return {"type": "exit_building"}
        if agent.node == agent.home_node and not agent.route:
            return {"type": "go_to_bed"}           # 自宅前 → 入って就寝
        if not agent.homing and not agent.route:
            return {"type": "move_to", "dest": agent.home_node, "stay_steps": 0,
                    "mode": _choose_mode(agent, sim, agent.home_node, rng),
                    "homing": True}
        return {"type": "continue"} if agent.route else {"type": "stay"}

    # ---- 病気(健康 H1、既定 OFF=agent.sick 立たず不変): 在宅療養へ寄せる(欠勤は work_window 側)----
    #  夜間は上の bedtime 分岐が先に睡眠を処理するので、ここは日中の在宅だけを担う。
    if getattr(agent, "sick", False):
        return _sick_home(agent, sim, sim_min, step, rng)

    # ---- 災害の外出抑制(都市・環境ショック H4、既定 OFF=_disaster_homebound 立たず不変): 台風/地震/大雪
    #  など発生中に在宅へ寄せる(交通麻痺+外出抑制)。在宅ロジックは病気(_sick_home)と同型=自宅へ帰り
    #  留まる。夜間は上の bedtime 分岐が睡眠を処理済み。乱数は帰路の mode 選択のみ(OFF は呼ばれない=不変)。
    if _disaster.is_homebound(agent):
        return _sick_home(agent, sim, sim_min, step, rng)

    # ---- 屋内 ----
    if agent.building:
        at_work = (agent.building == agent.work_building
                   and in_work_window(agent, sim_min, cal))
        at_pt = (agent.part_time and agent.building == agent.part_time["building"]
                 and in_part_time_window(agent, sim_min))
        if at_work or at_pt:
            return {"type": "stay"}                # 勤務・バイト中はその場に留まる
        # バイトのシフトが始まったら、今いる建物を出てバイト先へ向かう
        if agent.part_time and in_part_time_window(agent, sim_min) \
                and agent.building != agent.part_time["building"]:
            return {"type": "exit_building"}
        if agent.building == agent.home_building and not agent.route \
                and step < agent.stay_until:
            return {"type": "stay"}                # 自宅でくつろぐ(起床直後など)
        if step >= agent.stay_until:
            return {"type": "exit_building"}
        levels = int(sim.city.building(agent.building)["levels"])
        if rng.random() < 0.30 and levels > 1:
            floors = [f for f in range(1, levels + 1) if f != agent.floor]
            return {"type": "floor_move",
                    "floor": floors[int(rng.integers(len(floors)))]}
        return {"type": "stay"}

    # ---- 移動中 ----
    if agent.route:
        return {"type": "continue"}

    # ---- 勤務時間帯: 職場(実在 POI)へ ----
    if in_work_window(agent, sim_min, cal):
        if agent.node == agent.work_node:
            if agent.work_building \
                    and sim.city.has_building(agent.work_building):
                left_steps = max(1, (agent.work_end_min - m) // 10)
                return {"type": "enter_building", "building": agent.work_building,
                        "floor": agent.work_floor, "stay_steps": left_steps,
                        "activity": "working"}
            # 路面の職場(屋台・配達拠点など): その場で勤務
            return {"type": "stay"}
        if agent.work_node:
            return {"type": "move_to", "dest": agent.work_node, "stay_steps": 1,
                    "mode": _choose_mode(agent, sim, agent.work_node, rng),
                    "activity": "commuting"}

    # ---- バイトのシフト時間帯: バイト先(実在 POI)へ ----
    if agent.part_time and in_part_time_window(agent, sim_min):
        pt = agent.part_time
        if agent.node == pt["node"]:
            if pt["building"] and sim.city.has_building(pt["building"]):
                left_steps = max(1, (pt["end_min"] - m) // 10)
                return {"type": "enter_building", "building": pt["building"],
                        "floor": pt["floor"], "stay_steps": left_steps,
                        "activity": "working"}
            return {"type": "stay"}                # 路面のバイト先: その場で勤務
        return {"type": "move_to", "dest": pt["node"], "stay_steps": 1,
                "mode": _choose_mode(agent, sim, pt["node"], rng),
                "activity": "commuting"}

    # ---- 滞在中 ----
    if step < agent.stay_until:
        return {"type": "stay"}

    # ---- 食事時間帯: 実在の飲食店へ(残高が価格に満たない店は避ける=経済 v0)----
    if in_meal_window(sim_min) \
            and rng.random() < float(sim.cfg.world.meal_prob):
        foods = sim.city.pois_by_cat("food") + sim.city.pois_by_cat("nightlife")
        foods = [p for p in foods if p["node"] != agent.node
                 and agent.money >= _poi_price(sim, p)]
        foods = _commerce.filter_open(sim, foods, sim_min)   # 閉店中の飲食店/夜遊び先を除外(H3、OFF=不変)
        if foods:
            p = foods[int(rng.integers(len(foods)))]
            if _curfew_suppressed(agent, sim, p["node"], sim_min, step):
                return {"type": "stay"}              # 制度DSL: 時間帯×カテゴリの抑制
            return _augment_ride(agent, sim, {
                "type": "move_to", "dest": p["node"],
                "stay_steps": int(rng.integers(2, 5)),   # 2-4 step(短縮)
                "mode": _choose_mode(agent, sim, p["node"], rng),
                "activity": "eating"}, step, sim_min)

    # ---- 朝の一日計画(土台): 現在の時間帯に一致する未消化の予定へ向かう ----
    plan_move = _plan_move(agent, sim, sim_min, step, rng)
    if plan_move is not None:
        return plan_move

    # ---- メンタル(健康 H1、既定 OFF=withdrawn 立たず不変): 慢性高 grievance の持続 →
    #  引きこもり=自由時間の外出(街の外への exit・気まぐれな行き先)を控え在宅寄りにする。
    #  勤務・食事は上流で処理済み(引きこもりは自由行動のみを抑制)。乱数は帰路の mode 選択のみ。
    if getattr(agent, "withdrawn", False):
        if agent.home_node and agent.node != agent.home_node and not agent.route:
            return {"type": "move_to", "dest": agent.home_node, "stay_steps": 0,
                    "mode": _choose_mode(agent, sim, agent.home_node, rng),
                    "homing": True}
        return {"type": "stay"}

    # ---- 次の行動 ----
    r = rng.random()
    if r < float(sim.cfg.world.exit_prob) and sim.exit_points:
        gate = sim.exit_points[int(rng.integers(len(sim.exit_points)))]
        return {"type": "move_to", "dest": gate, "stay_steps": 0,
                "mode": _choose_mode(agent, sim, gate, rng), "exit": True}
    blds = sim.city.buildings_at(agent.node)
    blds = [b for b in blds if b["kind"] not in ("residential", "house?")
            or b["id"] == agent.home_building]      # 他人の家には入らない
    if blds and r < float(sim.cfg.world.building_enter_prob):
        bld = blds[int(rng.integers(len(blds)))]
        return {"type": "enter_building", "building": bld["id"],
                "stay_steps": int(rng.integers(2, 7))}   # 2-6 step(短縮)
    # 自由時間の行き先。群集(大規模行事型)> 観光回遊(後続波 H5)> デート(後続波 H2)>
    # 趣味(後続波 H6)> 制度DSL の weekly_event ブースト > Lynch landmark > EPR の順。いずれも OFF・非該当では乱数を
    # 引かず None=以降は従来通り(不変)。観光回遊は観光客のランドマーク巡り(tourist_visit)へ寄せる。
    # EPR の行き先候補は危険地帯(犯罪履歴ノード=治安回避 H5)を除いたリストにする(OFF=全候補で不変)。
    crowd = _crowd_dest(agent, sim, step)
    if crowd is not None:
        dest = crowd                                 # 群集日: 地図原点付近の集会ノードへ寄せる
    elif (tour := _diversity.tourist_dest(agent, sim, step, sim_min)) is not None:
        dest = tour                                  # 観光客: ランドマークへ回遊(後続波 H5)
    elif (date := _household.date_dest(agent, sim, step, sim_min)) is not None:
        dest = date                                  # デート: パートナーとの共有目的地へ(後続波 H2)
    elif (hobby := _inner.hobby_dest(agent, sim, step, sim_min)) is not None:
        dest = hobby                                 # 趣味: 余暇の行き先を趣味の場所へ寄せる(後続波 H6)
    elif (boost := _weekly_boost_dest(agent, sim, step)) is not None:
        dest = boost                                 # 制度DSL: 定期イベントの場所へ寄せる
    elif sim.psychcfg["lynch"]["enabled"] and getattr(sim, "landmark_score", None):
        dest = _lynch_destination(agent, sim, sim.hub.stream("lynch", agent.id, step))
    else:
        dest = choose_destination(agent, _diversity.safe_dests(sim, agent, sim.dests), rng)
    if dest == agent.node:
        return {"type": "stay"}
    if _curfew_suppressed(agent, sim, dest, sim_min, step):
        return {"type": "stay"}                      # 制度DSL: 時間帯×カテゴリの抑制
    return _augment_ride(agent, sim, {
        "type": "move_to", "dest": dest,
        "stay_steps": int(rng.integers(1, 4)),       # 1-3 step(短縮)
        "mode": _choose_mode(agent, sim, dest, rng)}, step, sim_min)


def _poi_price(sim, poi: dict) -> float:
    """消費カテゴリ(食事/nightlife 等)の価格。経済が無効なら 0(常に払える)。

    制度DSL の fee ルールがあれば残高判定にも反映する(rulebook 無し/無ルールで不変)。
    """
    economy = getattr(sim, "economy", None)
    if not economy or not economy.get("enabled"):
        return 0.0
    base = float(economy["prices"].get(poi.get("cat", ""), 0.0))
    rb = getattr(sim, "rulebook", None)
    return rb.fee_price(poi.get("cat", ""), base) if rb is not None else base
