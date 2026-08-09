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

from . import day_plan as _day_plan
from . import drive as _drive
from .. import commerce as _commerce
from .. import delivery as _delivery
from .. import disaster as _disaster
from .. import diversity as _diversity
from .. import household as _household
from .. import inner_life as _inner
from .. import joint as _joint
from .. import media
from .. import party as _party
from .. import services as _services
from .. import spark as _spark
from ..factors import update as factor_update
from ..observer.schema import Event
from ..world import calendar as _calendar

RHO = 0.6
GAMMA = 0.21

MEAL_WINDOWS = [(11 * 60 + 30, 13 * 60 + 30), (18 * 60, 21 * 60)]


# ============================================================
# 確率的実行(行動のゆらぎ)= P2 スライス S4。既定 OFF=新 stream を一切引かず
# ゴールデン L1 バイト一致。実証3層(interstitial-life §4.1):
#   L1 骨格 motif(stream "motif"): 朝1回、その日の行動骨格を確率抽選(routine=高確率・
#     「いつもと違う一日」=低確率)。novelty 日は L2/L3 の強度を増幅する。
#   L2 時刻ジッター(stream "jitter_time"): flexible 活動の開始/継続を一様 ±jitter_min ずらす
#     (MATSim mutationRange=1800s=±30分・一様・affectsDuration)。fixed アンカーは対象外。
#   L3 逸脱(stream "detour"/"interrupt"): 寄り道=目的地手前で近傍 POI に立ち寄り(per-move・
#     活動種別で可変)/ 中断=活動途中で離脱。
#   Gumbel(stream "gumbel"): 行き先 argmax を U=V+ε(ε=-log(-log(uniform)))の確率選択へ置換
#     (ON 時のみ)。温度は活動3分類で固定(mandatory=低温≒決定論・discretionary=高温)。
# 摂動は専用 stream にだけ載せ、既存 routine の乱数(choose_destination 等)は既定 stream の
# まま触らない(crowd/curfew/media の作法)。入力は物理量・時刻・活動種別のみ(因子・k を見ない)。
_CATS3 = ("mandatory", "maintenance", "discretionary")
_FIXED_ACTS = frozenset({"working", "commuting"})   # fixed アンカー由来(ゆらぎ・中断の対象外)
# 実行時 activity ラベル → 3分類。未知/自由行動は discretionary(最も揺らぐ)。
_ACT_CAT = {"working": "mandatory", "commuting": "mandatory",
            "eating": "maintenance", "shopping": "maintenance"}
# 計画語彙 what → 3分類(plan item の cat 推定用。plan_schema と同一マップ)。
_WHAT_CAT = {"work": "mandatory", "study": "mandatory",
             "meal": "maintenance", "shop": "maintenance", "personal": "maintenance",
             "escort": "maintenance", "home": "maintenance",
             "leisure": "discretionary", "park": "discretionary",
             "walk": "discretionary", "visit": "discretionary", "meetup": "discretionary"}
_STOCH_UNSET = object()


def _catmap(raw, default: dict) -> dict:
    raw = raw or {}
    return {c: float(raw.get(c, default[c])) for c in _CATS3}


def _build_stochastic_cfg(raw) -> dict:
    """routine.stochastic の生 cfg を数値へ正規化する(enabled 済み前提)。"""
    gum = raw.get("gumbel", {}) or {}
    return {
        "novelty_prob": float(raw.get("novelty_prob", 0.15)),
        "novelty_scale": float(raw.get("novelty_scale", 1.6)),
        "jitter_min": float(raw.get("jitter_min", 30.0)),
        "detour_prob": _catmap(raw.get("detour_prob"),
                               {"mandatory": 0.05, "maintenance": 0.12,
                                "discretionary": 0.18}),
        "detour_radius_m": float(raw.get("detour_radius_m", 300.0)),
        "interrupt_prob": _catmap(raw.get("interrupt_prob"),
                                  {"mandatory": 0.0, "maintenance": 0.06,
                                   "discretionary": 0.10}),
        "gumbel_enabled": bool(gum.get("enabled", False)),
        "gumbel_temp": _catmap(gum.get("temperature"),
                               {"mandatory": 0.3, "maintenance": 0.8,
                                "discretionary": 1.5}),
    }


def _stochastic_cfg(sim):
    """routine.stochastic 設定を sim に一度だけキャッシュ。無効/未設定なら None(=既定挙動で
    新 stream を一切引かない)。既存 _media_settings と同じ遅延キャッシュ作法。"""
    cached = getattr(sim, "_stoch_cfg", _STOCH_UNSET)
    if cached is not _STOCH_UNSET:
        return cached
    raw = (sim.cfg.get("routine", {}) or {}).get("stochastic", {}) or {}
    cfg = _build_stochastic_cfg(raw) if raw.get("enabled", False) else None
    sim._stoch_cfg = cfg
    return cfg


def _world_prob(sim, key: str) -> float:
    """`world.<key>`(確率定数)を sim に一度だけキャッシュして返す。

    config は走行開始前に確定する(`load_config` の Δt 変換 `timeconv.apply_dt` と
    `registry.apply_mode` が唯一の書き換え点で、どちらも `Simulation.__init__` より前 /
    その中の1回きり)。step ループ中に `cfg.world.*` へ書く箇所は存在しないので、
    値は毎 step 読むのと同一で、消えるのは OmegaConf のノード解決コストだけ。
    **キーごとに遅延**して読む(=そのキーを使う分岐に初めて入った時に初めて解決する)ため、
    読まれる config キーの集合も現行と完全に同じ。`_stoch_cfg` / `_media_settings` と同じ作法。
    """
    cache = getattr(sim, "_routine_world_probs", None)
    if cache is None:
        cache = {}
        sim._routine_world_probs = cache
    value = cache.get(key)
    if value is None:
        value = float(sim.cfg.world[key])
        cache[key] = value
    return value


def _cat_of(activity: str = "", what: str = "") -> str:
    """activity ラベル or 計画語彙 what を3分類へ写す(既定=discretionary)。"""
    if what:
        return _WHAT_CAT.get(what, "discretionary")
    return _ACT_CAT.get(activity or "", "discretionary")


def _motif_scale(agent, scfg) -> float:
    """novelty 日は L2/L3 の強度を novelty_scale 倍(routine 日は等倍)。"""
    return scfg["novelty_scale"] \
        if getattr(agent, "_motif", "routine") == "novelty" else 1.0


def maybe_roll_motif(sim, agent, step: int) -> None:
    """L1: その日の行動骨格 motif を専用 stream で1日1回抽選する(冪等)。既定 OFF は
    何も引かず何も生やさない。朝の計画確定後(planning)/ その日の初回 routine 決定のいずれか
    早い方で1回だけ確定する。motif は (agent, day) で決まるので確定タイミングに依存しない。"""
    scfg = _stochastic_cfg(sim)
    if scfg is None:
        return
    day = sim.clock.sim_min(step) // 1440
    if getattr(agent, "_motif_day", -1) == day:
        return
    agent._motif_day = day
    r = float(sim.hub.stream("motif", agent.id, day).random())
    agent._motif = "novelty" if r < scfg["novelty_prob"] else "routine"


def _jitter_stay(agent, sim, base_steps: int, step: int, scfg,
                 activity: str = "", what: str = "") -> int:
    """L2: flexible 活動の stay_steps(継続時間)を一様 ±jitter_min ずらす。fixed アンカー
    (勤務・通勤=mandatory)は揺らさない。1 step=10 分。専用 stream 'jitter_time'。"""
    if activity in _FIXED_ACTS or _cat_of(activity, what) == "mandatory":
        return base_steps
    span = (scfg["jitter_min"] * _motif_scale(agent, scfg)
            / float(sim.clock.step_minutes))                        # 分→step
    if span <= 0.0:
        return base_steps
    rng = sim.hub.stream("jitter_time", agent.id, step)
    return max(0, base_steps + int(round(float(rng.uniform(-span, span)))))


def _jitter_band_min(agent, sim, sim_min: int, step: int, scfg) -> int:
    """L2: 予定消化の判定時刻(開始時刻)を一様 ±jitter_min ずらす。専用 stream 'jitter_time'
    の別キー(既存 stay ジッターと独立)。"""
    span = scfg["jitter_min"] * _motif_scale(agent, scfg)
    if span <= 0.0:
        return sim_min
    rng = sim.hub.stream("jitter_time", agent.id, step, "band")
    return int(sim_min + round(float(rng.uniform(-span, span))))


def _nearby_pois(agent, sim, dest, radius_m: float) -> list:
    """現在地の半径内 POI ノード(現在地・目的地は除く)。周辺スポット密度がそのまま重み
    (同一ノードに複数 POI があれば重複=魅力度スケール)。"""
    ax, ay = sim.city.node_xy(agent.node)
    out = []
    for p in sim.city.poi_list:
        node = p["node"]
        if node == agent.node or node == dest:
            continue
        px, py = sim.city.node_xy(node)
        if math.hypot(ax - px, ay - py) <= radius_m:
            out.append(node)
    return out


def _maybe_detour(agent, sim, dest, step: int, sim_min: int, scfg,
                  activity: str = "", what: str = ""):
    """L3 寄り道: 目的地手前で近傍 POI に立ち寄る寄り先ノードを返す(しなければ None)。
    per-move 確率は活動種別で可変(mandatory 低・discretionary 高)・motif で変調。専用 stream
    'detour'。発火時は軽量 detour イベントを1件記録する(種別=元活動・寄り先・元目的地)。"""
    p = scfg["detour_prob"][_cat_of(activity, what)] * _motif_scale(agent, scfg)
    if p <= 0.0:
        return None
    rng = sim.hub.stream("detour", agent.id, step)
    if float(rng.random()) >= p:
        return None
    cands = _nearby_pois(agent, sim, dest, scfg["detour_radius_m"])
    if not cands:
        return None
    to = cands[int(rng.integers(len(cands)))]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="detour", x=agent.x, y=agent.y,
                         payload={"act": activity or what or "free",
                                  "to": to, "from_dest": dest}))
    return to


def _maybe_interrupt(agent, sim, step: int, sim_min: int, scfg) -> bool:
    """L3 中断: 滞在中の活動を途中で離脱するか。fixed アンカー由来(勤務・通勤)は中断しない。
    活動種別で可変・motif で変調。専用 stream 'interrupt'。発火時は interrupt を1件記録する。"""
    act = getattr(agent, "activity", "") or ""
    if act in _FIXED_ACTS:
        return False
    p = scfg["interrupt_prob"][_cat_of(act)] * _motif_scale(agent, scfg)
    if p <= 0.0:
        return False
    if float(sim.hub.stream("interrupt", agent.id, step).random()) >= p:
        return False
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="interrupt", x=agent.x, y=agent.y,
                         payload={"act": act or "free"}))
    return True


def _gumbel_destination(agent, dests: list, sim, step: int, scfg,
                        activity: str = "", what: str = "") -> str:
    """Gumbel-max: argmax_d(V_d/T + ε_d)、ε=-log(-log(uniform))、V=log1p(訪問回数)。温度 T は
    活動3分類で固定(低温=いつもの行き先が argmax を支配・高温=ε 支配で多様化)。専用 stream
    'gumbel'。候補が無ければ現在地(=stay)。"""
    cands = [d for d in dests if d != agent.node]
    if not cands:
        return agent.node
    temp = max(1e-6, scfg["gumbel_temp"][_cat_of(activity, what)])
    rng = sim.hub.stream("gumbel", agent.id, step)
    best, best_u = agent.node, -math.inf
    for d in cands:
        v = math.log1p(float(agent.visits[d]))
        eps = -math.log(-math.log(max(1e-12, float(rng.random()))))
        u = v / temp + eps
        if u > best_u:
            best_u, best = u, d
    return best


# ============================================================
# 好奇心/退屈の内発ドライブ = P2 スライス S5(interstitial-life §4.4-a)。ゲージ本体は
# cognition/drive.py(boredom_tick/ready/fire。silence と同じ作法)。ここは「閾値超えで LLM
# なしの内発的探索を発火する」出力側。行き先=近傍の未訪問/低頻度 POI(=知らない店/ふらっと
# 歩く)。S4 Gumbel 機構があれば再利用(温度は discretionary)し、なければ決定論近傍選択。
# 新 stream は "boredom" 1本のみ(Gumbel 乱数もこの stream から引く=stream を増やさない)。
# 既定 OFF(drivecfg['boredom'] 無効)は stream を一切引かず boredom_explore 0 件=バイト一致。
# 入力は物理量(現在地・近傍 POI・訪問回数)のみ=因子・k を見ない(no-fingerprint / R1)。
def _dest_kind(sim, node) -> str:
    """探索先ノードの種別(先頭 POI のカテゴリ、無ければ 'street')。boredom_explore payload 用。"""
    pois = sim.city.pois_at_node(node)
    if pois:
        return pois[0].get("cat", "poi") or "poi"
    return "street"


def _boredom_destination(agent, sim, scfg, bc, rng):
    """退屈探索の行き先ノード: 近傍(radius_m)の未訪問/低頻度 POI。未訪問があれば最優先。
    S4 Gumbel(scfg あり & 有効)なら V=-log1p(訪問回数)(未訪問/低頻度ほど高)+ ε の確率選択を
    boredom stream で行い、なければ決定論選択(訪問回数↑距離↑node の辞書順)。候補なしは None。"""
    cands = _nearby_pois(agent, sim, agent.node, bc["radius_m"])   # 現在地・重複ノードは _nearby_pois 側で除外
    if not cands:
        return None
    unseen = [n for n in cands if agent.visits[n] == 0]
    pool = unseen if unseen else cands
    if scfg is not None and scfg["gumbel_enabled"]:               # S4 Gumbel 機構の再利用(温度=discretionary)
        temp = max(1e-6, scfg["gumbel_temp"]["discretionary"])
        best, best_u = None, -math.inf
        for n in pool:
            v = -math.log1p(float(agent.visits[n]))               # 未訪問/低頻度ほど高スコア(novelty 志向)
            eps = -math.log(-math.log(max(1e-12, float(rng.random()))))
            u = v / temp + eps
            if u > best_u:
                best_u, best = u, n
        return best
    ax, ay = sim.city.node_xy(agent.node)                        # 決定論近傍選択(乱数を引かない)
    return min(pool, key=lambda n: (int(agent.visits[n]),
                                    math.hypot(ax - sim.city.node_xy(n)[0],
                                               ay - sim.city.node_xy(n)[1]),
                                    str(n)))


def _maybe_boredom_explore(agent, sim, step: int, sim_min: int, from_act: str = ""):
    """退屈ゲージが閾値超え+cooldown 明けなら内発的探索(近傍の未訪問/低頻度 POI への move)を
    LLM なしで発火する。発火時は boredom_explore を1件記録しゲージをリセット+cooldown を張る。
    既定 OFF・非候補・非発火・行き先なし はいずれも None=既定挙動(stream を引かない)。"""
    bc = sim.drivecfg["boredom"]
    if not bc["enabled"] or not _drive.boredom_ready(agent, sim.drivecfg, step):
        return None
    rng = sim.hub.stream("boredom", agent.id, step)              # 新 stream(発火抽選 + Gumbel 乱数を兼ねる)
    if float(rng.random()) >= bc["fire_prob"]:
        return None
    dest = _boredom_destination(agent, sim, _stochastic_cfg(sim), bc, rng)
    if dest is None or dest == agent.node:
        return None
    gauge = round(float(getattr(agent, "_boredom", 0.0)), 3)
    _drive.boredom_fire(agent, sim.drivecfg, step)              # ゲージリセット + cooldown
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="boredom_explore", x=agent.x, y=agent.y,
                         payload={"from": from_act or "free",
                                  "to_kind": _dest_kind(sim, dest), "gauge": gauge}))
    # 近傍(radius_m<既定の car/bicycle しきい 400m)なので徒歩固定=乱数を引かず _choose_mode と同結果。
    return {"type": "move_to", "dest": dest, "stay_steps": bc["stay_steps"],
            "mode": "walk"}


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


def _lynch_pool(sim):
    """Lynch の候補ノード列と重み配列を1回だけ作ってキャッシュする(値は毎回作るのと同一)。

    `sim.dests` は Simulation 構築時に1度だけ組まれて以後不変(engine 側に再代入も追記も
    無い)、`sim.landmark_score` も ON のとき構築時に1度だけ作られる。よって
    「score を持つ dests」とその重みは走行中ずっと同じ = 呼び出しごとに作り直す必要が無い。
    列の順序も重みの値も元の内包表記と同一で、呼び出し側は現在地1点を抜くだけになる。
    保険として `len(sim.dests)` が変わったら作り直す。
    戻り値 (候補列, 重み配列, ノード→添字)。"""
    n = len(sim.dests)
    cached = getattr(sim, "_lynch_pool_cache", None)
    if cached is None or cached[3] != n:
        score = sim.landmark_score
        pool = [d for d in sim.dests if d in score]
        pool_w = np.array([max(0.0, float(score[d])) for d in pool], dtype=float)
        cached = (pool, pool_w, {d: i for i, d in enumerate(pool)}, n)
        sim._lynch_pool_cache = cached
    return cached[0], cached[1], cached[2]


def _lynch_destination(agent, sim, rng: np.random.Generator) -> str:
    """Lynch 都市イメージ プラグイン(既定 OFF、ON 時のみ・新 stream で呼ばれる)。

    自由時間の行き先を landmark_score(engine が事前構築した「イメージアビリティ」の
    不透明スコア)で重み付けサンプルする。この関数は score 値を見るだけで因子名は知らない。
    候補が無ければ現在地(=stay)。既定 decide の draw 順は一切汚さない(専用 rng)。
    """
    pool, pool_w, index = _lynch_pool(sim)
    hit = index.get(agent.node)
    if hit is None:                        # 現在地は候補に含まれない = 全部が候補
        cands, weights = pool, pool_w.copy()   # copy: 下で /= により破壊するため
    else:                                  # 現在地の1点だけを抜く(順序・値は元の内包表記と同一)
        cands = pool[:hit] + pool[hit + 1:]
        weights = np.delete(pool_w, hit)
    if not cands:
        return agent.node
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
                "floor": agent.home_floor,
                "stay_steps": sim.clock.dur_steps(3)}   # 自宅前 → 入って在宅(30分)
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


def _plan_move(agent, sim, sim_min: int, step: int, rng, scfg=None) -> dict | None:
    """現在の時間帯に一致する未消化の予定があれば、その行き先への move を組む。

    day_plan が空(=計画なし。planning 無効時も含む)のときは乱数を一切引かずに None
    を返す(既定挙動の再現性を保つ)。1項目は1回だけ消化する(解決可否に関わらず)。
    scfg(P2 S4)= None のときは新 stream を一切引かず既定挙動とバイト一致。ON のときのみ
    L2 開始/継続ジッター・L3 寄り道を専用 stream で重ねる(既存の rng draw 順は不変)。"""
    plan = getattr(agent, "day_plan", None)
    if not plan:
        return None
    band_min = sim_min if scfg is None \
        else _jitter_band_min(agent, sim, sim_min, step, scfg)   # L2 開始ジッター
    band = _time_band(band_min)
    item = next((it for it in plan
                 if not it.get("done") and it.get("when") == band), None)
    if item is None:
        return None
    item["done"] = True
    dest = _resolve_plan_dest(agent, sim, item, rng, sim_min)
    if not dest or dest == agent.node:
        return None
    what = str(item.get("what") or "")
    base_stay = sim.clock.dur_steps(int(rng.integers(2, 5)))   # 20-40分
    if scfg is not None:
        detour = _maybe_detour(agent, sim, dest, step, sim_min, scfg, what=what)  # L3 寄り道
        if detour is not None:
            dest = detour
        base_stay = _jitter_stay(agent, sim, base_stay, step, scfg, what=what)    # L2 継続ジッター
    action = {"type": "move_to", "dest": dest, "stay_steps": base_stay,
              "mode": _choose_mode(agent, sim, dest, rng)}
    activity = _PLAN_ACTIVITY.get(what, "")
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
        # 実バスダイヤの静的表 v-Ride-2(既定 OFF=bus_table is None=従来近似のまま=バイト一致)。
        # 表があれば実ダイヤ近似(次便待ち+区間所要)へ格上げし、ride に wait_s/ride_s を追記する。
        bt = getattr(sim, "bus_table", None)
        wait_s = ride_s = None
        if bt is not None:
            rt = bt.find_ride(agent.node, dest, sim.city, sim_min)
            leg = None if rt is None else {"from": rt["from"], "to": rt["to"]}
            if rt is not None:
                wait_s, ride_s = rt["wait_s"], rt["ride_s"]
        else:
            leg = buses.find_ride(agent.node, dest, sim.city)
        if leg is not None:
            fare = float(bus["fare"])
            if rb is not None:
                fare = rb.fee_price("bus", fare)
            ride_out = {"mode": "bus", "fare": fare,
                        "from": leg["from"], "to": leg["to"]}
            if wait_s is not None:                   # 実ダイヤ近似のときだけ観測を追記(OFF=不変)
                ride_out["wait_s"] = round(float(wait_s), 1)
                ride_out["ride_s"] = round(float(ride_s), 1)
            return ("car", ride_out)
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
    # ---- 宅配④: 配達 gig 中の配達員は配送を最優先(通常 routine に割り込ませない。既定OFF=フラグ立たず不変)----
    gig = _delivery.courier_action(agent, sim, step, sim_min)
    if gig is not None:
        return gig
    cal = _cal(sim)                         # 暦(weekday_work ゲート用。既定 OFF=不変)
    scfg = _stochastic_cfg(sim)             # 確率的実行 P2 S4(None=既定=新 stream を一切引かない)
    if scfg is not None:
        maybe_roll_motif(sim, agent, step)  # L1 骨格 motif(1日1回・冪等)

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
                left_steps = max(1, sim.clock.min_to_steps(agent.work_end_min - m))
                return {"type": "enter_building", "building": agent.work_building,
                        "floor": agent.work_floor, "stay_steps": left_steps,
                        "activity": "working"}
            # 路面の職場(屋台・配達拠点など): その場で勤務
            return {"type": "stay"}
        if agent.work_node:
            return {"type": "move_to", "dest": agent.work_node,
                    "stay_steps": sim.clock.dur_steps(1),
                    "mode": _choose_mode(agent, sim, agent.work_node, rng),
                    "activity": "commuting"}

    # ---- バイトのシフト時間帯: バイト先(実在 POI)へ ----
    if agent.part_time and in_part_time_window(agent, sim_min):
        pt = agent.part_time
        if agent.node == pt["node"]:
            if pt["building"] and sim.city.has_building(pt["building"]):
                left_steps = max(1, sim.clock.min_to_steps(pt["end_min"] - m))
                return {"type": "enter_building", "building": pt["building"],
                        "floor": pt["floor"], "stay_steps": left_steps,
                        "activity": "working"}
            return {"type": "stay"}                # 路面のバイト先: その場で勤務
        return {"type": "move_to", "dest": pt["node"],
                "stay_steps": sim.clock.dur_steps(1),
                "mode": _choose_mode(agent, sim, pt["node"], rng),
                "activity": "commuting"}

    # ---- 滞在中(退屈探索 S5 + L3 中断 S4: flexible 活動のみ。既定 OFF=常に stay)----
    if step < agent.stay_until:
        # 好奇心/退屈ドライブ(P2 S5): 同じ場所への長居でゲージが閾値超え → 内発的探索へ(LLM なし)。
        explore = _maybe_boredom_explore(agent, sim, step, sim_min,
                                         from_act=getattr(agent, "activity", "") or "stay")
        if explore is not None:
            agent.stay_until = step                  # 滞在を切り上げて近傍の未訪問先へ
            agent.activity = ""
            return explore
        if scfg is not None and _maybe_interrupt(agent, sim, step, sim_min, scfg):
            agent.stay_until = step                  # 滞在打ち切り=離脱 → 以降の分岐(次の行動)へ
            agent.activity = ""
        else:
            return {"type": "stay"}

    # ---- 宅配④: 食事帯の在宅/職場滞在は外食の代替として宅配を注文しうる(既定OFF=None=不変)。当たれば
    #  stay を返して外食に出ない(二重課金しない)。乱数は専用 stream "delivery"(main routine draw 順に不干渉)----
    order = _delivery.maybe_order(agent, sim, step, sim_min)
    if order is not None:
        return order

    # ---- 食事時間帯: 実在の飲食店へ(残高が価格に満たない店は避ける=経済 v0)----
    if in_meal_window(sim_min) \
            and rng.random() < _world_prob(sim, "meal_prob"):
        foods = sim.city.pois_by_cat("food") + sim.city.pois_by_cat("nightlife")
        foods = [p for p in foods if p["node"] != agent.node
                 and agent.money >= _poi_price(sim, p)]
        foods = _commerce.filter_open(sim, foods, sim_min)   # 閉店中の飲食店/夜遊び先を除外(H3、OFF=不変)
        if foods:
            p = foods[int(rng.integers(len(foods)))]
            if _curfew_suppressed(agent, sim, p["node"], sim_min, step):
                return {"type": "stay"}              # 制度DSL: 時間帯×カテゴリの抑制
            dest = p["node"]
            base_stay = sim.clock.dur_steps(int(rng.integers(2, 5)))  # 20-40分
            if scfg is not None:                     # L3 寄り道 + L2 継続ジッター(maintenance)
                detour = _maybe_detour(agent, sim, dest, step, sim_min, scfg,
                                       activity="eating")
                if detour is not None:
                    dest = detour
                base_stay = _jitter_stay(agent, sim, base_stay, step, scfg,
                                         activity="eating")
            return _augment_ride(agent, sim, {
                "type": "move_to", "dest": dest, "stay_steps": base_stay,
                "mode": _choose_mode(agent, sim, dest, rng),
                "activity": "eating"}, step, sim_min)

    # ---- 朝の一日計画(土台): 現在の時間帯に一致する未消化の予定へ向かう ----
    #  第86 day_plan v1(既定 OFF): ON のときだけ構造化ブロックの実行系へ差し替える
    #  (**単一の作用点**)。OFF は従来の _plan_move をそのまま呼ぶ=バイト一致。
    #  どちらも None = 空き時間 → 以降の既存の習慣ポリシーが埋める。
    plan_move = (_day_plan.plan_action(agent, sim, sim_min, step, rng, scfg)
                 if _day_plan.enabled(sim)
                 else _plan_move(agent, sim, sim_min, step, rng, scfg))
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
    if r < _world_prob(sim, "exit_prob") and sim.exit_points:
        gate = sim.exit_points[int(rng.integers(len(sim.exit_points)))]
        return {"type": "move_to", "dest": gate, "stay_steps": 0,
                "mode": _choose_mode(agent, sim, gate, rng), "exit": True}
    blds = sim.city.buildings_at(agent.node)
    blds = [b for b in blds if b["kind"] not in ("residential", "house?")
            or b["id"] == agent.home_building]      # 他人の家には入らない
    if blds and r < _world_prob(sim, "building_enter_prob"):
        bld = blds[int(rng.integers(len(blds)))]
        return {"type": "enter_building", "building": bld["id"],
                "stay_steps": sim.clock.dur_steps(int(rng.integers(2, 7)))}  # 20-60分
    # 自由時間の行き先。群集(大規模行事型)> 観光回遊(後続波 H5)> デート(後続波 H2)>
    # 趣味(後続波 H6)> 制度DSL の weekly_event ブースト > Lynch landmark > EPR の順。いずれも OFF・非該当では乱数を
    # 引かず None=以降は従来通り(不変)。観光回遊は観光客のランドマーク巡り(tourist_visit)へ寄せる。
    # EPR の行き先候補は危険地帯(犯罪履歴ノード=治安回避 H5)を除いたリストにする(OFF=全候補で不変)。
    crowd = _crowd_dest(agent, sim, step)
    joint_act = ""                                   # 共同行動の活動タグ(cost 用。既定 ""=既定不変)
    service_act = ""                                 # サービス来店タグ(到着時 charge_service 用。既定 ""=不変)
    if crowd is not None:
        dest = crowd                                 # 群集日: 地図原点付近の集会ノードへ寄せる
    elif (tour := _diversity.tourist_dest(agent, sim, step, sim_min)) is not None:
        dest = tour                                  # 観光客: ランドマークへ回遊(後続波 H5)
    elif (date := _household.date_dest(agent, sim, step, sim_min)) is not None:
        dest = date                                  # デート: パートナーとの共有目的地へ(後続波 H2)
    elif (dinner := _household.dinner_dest(agent, sim, step, sim_min)) is not None:
        dest = dinner                                # 夕食の共食: 夜帯に世帯 home へ収束(第44バッチ S-R1)
    elif (rendez := _joint.joint_dest(agent, sim, step, sim_min)) is not None:
        dest = rendez                                # 共同行動: 同伴グループの共有 POI へ収束(第44バッチ S-R3)
        joint_act = _joint.route_activity(agent)     # meal_cafe→eating(既存 _charge_meal で課金)/他→""
    elif (party := _party.party_dest(agent, sim, step, sim_min)) is not None:
        dest = party                                 # 来街者 party: 連れと共有の回遊 POI へ収束(第45バッチ S-R5)
    elif (spk := _spark.spark_dest(agent, sim, step, sim_min)) is not None:
        dest = spk                                   # 火種介入: sparked の集会アンカー POI へ収束(第53バッチ・bias 減衰)
    elif (svc := _services.service_dest(agent, sim, step, sim_min)) is not None:
        dest = svc                                   # サービス来店: 近傍の service/education POI へ寄せる(第46バッチ ③)
        service_act = "service"                      # 到着時 activity=="service" で受給(charge_service)
    elif (hobby := _inner.hobby_dest(agent, sim, step, sim_min)) is not None:
        dest = hobby                                 # 趣味: 余暇の行き先を趣味の場所へ寄せる(後続波 H6)
    elif (boost := _weekly_boost_dest(agent, sim, step)) is not None:
        dest = boost                                 # 制度DSL: 定期イベントの場所へ寄せる
    elif sim.psychcfg["lynch"]["enabled"] and getattr(sim, "landmark_score", None):
        dest = _lynch_destination(agent, sim, sim.hub.stream("lynch", agent.id, step))
    elif scfg is not None and scfg["gumbel_enabled"]:
        # Gumbel ロジット化(ON 時のみ): 行き先 argmax を U=V+ε の確率選択へ置換(discretionary)。
        dest = _gumbel_destination(
            agent, _diversity.safe_dests(sim, agent, sim.dests), sim, step, scfg)
    else:
        dest = choose_destination(agent, _diversity.safe_dests(sim, agent, sim.dests), rng)
    if dest == agent.node:
        return {"type": "stay"}
    if _curfew_suppressed(agent, sim, dest, sim_min, step):
        return {"type": "stay"}                      # 制度DSL: 時間帯×カテゴリの抑制
    base_stay = sim.clock.dur_steps(int(rng.integers(1, 4)))  # 10-30分
    if scfg is not None:                             # L3 寄り道 + L2 継続ジッター(discretionary)
        detour = _maybe_detour(agent, sim, dest, step, sim_min, scfg)
        if detour is not None:
            dest = detour
        base_stay = _jitter_stay(agent, sim, base_stay, step, scfg)
    if service_act:                                  # サービス来店: 定義の滞在 step を使う(到着で受給)
        base_stay = int(getattr(agent, "_service_stay", base_stay) or base_stay)
    action = {"type": "move_to", "dest": dest, "stay_steps": base_stay,
              "mode": _choose_mode(agent, sim, dest, rng)}
    if joint_act:                                    # 共同行動の活動タグ(cost 用。既定 ""=未設定=不変)
        action["activity"] = joint_act
    elif service_act:                                # サービス来店タグ(到着時 charge_service。既定 ""=不変)
        action["activity"] = service_act
    return _augment_ride(agent, sim, action, step, sim_min)


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
