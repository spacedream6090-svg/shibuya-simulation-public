"""駅員・車掌 = **持ち場を持つアクター**(actor model P3a・``transit_staff``・**既定 OFF**)。

正典
----
- ``docs/plans/actor-model-migration-plan.md`` の
  「**3a 駅員・車掌**: 車掌 = ドア閉判断エージェント(tier0)。**ダイヤ結合ループの内生化**:
   ホーム密度 → 乗降時間(+15 人/車両 ≒ +1 秒・東京実証則)→ 続行遅延 → 滞留。
   ダイヤ自体は凍結制度・遅延と運休は行為に。」(ユーザー承認済みの設計文書 §9 優先度 1)
- 前例: ``src/society/devices.py``(P2 = 目標を持たない装置アクター)。本 module は
  その **目標と持ち場を持つ人間側**にあたる層で、装置層の「細粒度が粗い近似に勝つ」
  という優先順位の作法をそのまま引き継ぐ。

何を解く問題か
--------------
監査で判ったこと(実測):

1. **電車は実体ではない**。``world/transit.py`` の ``has_service()`` は時刻表の述語で、
   列車オブジェクトも運転士も車掌も世界に存在しない。
2. それでも**遅延は起きていた**。作っていたのは
   (a) ``envfeedback`` 規則1(ホーム負荷 → 停車時間延長 → ``delay_min``)と
   (b) ``disaster`` の日次サイコロ。どちらも **主体が居ない**。
   つまり「誰も決めていないのに世界が遅れる」= P2 で装置に与えた
   「世界に作用するものは全てアクターである」という原則の穴だった。
3. ペルソナには **駅員 / 車掌 / 電車運転士 が実在する**のに、地図に対応する POI
   カテゴリが無いので ``work.bind_workplace`` が ``no_category`` で束ねられず、
   **一度も持ち場に立たなかった**(職場束ね統計の未解決職業に出ている)。

本 module はこの 3 つを 1 本のループで閉じる:

    ホーム負荷 → **当直の車掌のドア閉判断**(L1 ``dwell_decision``)→ ``delay_min``
    → 既存の消費者(退出/帰還の待ち・``ext.transit_delay``・計画の作り直し)

★**変えたのは「誰が決めるか」であって「何が起きるか」ではない**(既定サブフラグ):
  停車時間の式・閾値・上限・回復運転項は ``envfeedback`` の規則1 と **同一の関数**
  (``envfeedback.dwell_step`` / ``advance_delay`` / ``commit_delay``)を呼ぶ。
  定数は本 module に 1 つも無い。tests/test_transit_staff.py が「同じ負荷なら
  規則1 が出したはずの ``delay_min`` と 1 ビットも違わない」を機械固定する。

較正(**東京の実測則**)
------------------------
- Palmqvist / Tomii / Ochiai (2020) ほかの東京の実測: **+15 人/車両 ≒ +1 秒の停車時間**、
  通勤列車の総遅延の **約 9 割は 5 分以下の停車時間超過**。
  → conf ``transit_staff.dwell.dwell_s_per_15pax``(既定 1.0 秒)に置いた。
- ★既定では**この較正値を使わない**(``dwell.calibrated: false``)。規則1 の既定係数
  ``env.feedback.transit.dwell_sec_per_pax`` をそのまま使う = **legacy の数値を 1 も
  動かさない**。較正版は opt-in のサブフラグで、ON のとき係数だけが
  ``dwell_s_per_15pax / 15`` に差し替わる(式・上限・回復項は共有のまま)。

本波の範囲(**正直なスコープ宣言**)
------------------------------------
ループが閉じるのは **ホーム負荷 → 車掌のドア閉判断 → ``delay_min`` → 既存の消費者**
までである。以下は **本波では作らない**(列車が実体でないため原理的に作れない):

- **列車エンティティ**(編成・車両・乗車率)。したがって「車両あたりの乗車人数」は
  持てず、負荷はホーム 1 ノードの人数で代理する(``envfeedback`` の限界 1 と同じ)。
- **続行間隔(headway)/ 閉塞信号の伝播**。下流列車への遅延伝播は時間方向のみ
  (同一駅の連続する発車列に沿った γ の持ち越し)= 規則1 の限界 2 をそのまま継承。
- **運休の宣言**。運休は今も ``disaster`` 層と ``transit.suspended`` の担当で、
  本 module は ``has_service()`` を**読むだけ**(運転中止を決める運行指令アクターは
  後続波)。運行が無い step ではドアを閉める判断そのものが存在しない。

R1 ドクトリン
-------------
- 既定 ``transit_staff.enabled: false`` では ``phase`` が即 return し、**束ねも起きず・
  L1 に 1 件も出ず・sim にも agent にも属性が生えず・プロンプトが 1 バイトも変わらない**
  (= ゴールデン L1 バイト一致)。
- **乱数を 1 本も引かない**(当直の選定も当直窓も名簿順の純関数)。``generate()`` の
  呼び出しサイトを 1 つも作らない = LLM 追加呼ゼロ = k 非依存。
- no-fingerprint: **プロンプトへ 1 バイトも足さない**。遅延は従来どおり「動けない」という
  世界の事実としてしか個体に届かない(語彙も欄も増やさない)。
- 状態は ``envfeedback`` の ``_envfb``(checkpoint 中央管理済み)に**間借りする**だけで、
  本 module 固有の可変状態は「束ね済みフラグ」1 個しか持たない。束ねは冪等なので
  resume で 2 度走っても結果は同じ(``work_node`` は checkpoint に載っている)。

正直な限界(4 件)
------------------
1. **駅は 1 ノード**。改札 / ホーム / コンコースの区別が無いので、駅員の「持ち場」
   (改札・ホーム・案内・事務室)は表現できない。全員が同じノードに立つ。
2. **当直表は合成**。ペルソナ pool の ``duty_pattern``(``rotates: true`` /
   ``shift_hours: 8``)は record 側の情報で、agent には届いていない。本 module は
   名簿順(agent id 昇順)の index から ``n_shifts`` 直を回す**決定論の合成当直表**を
   作る。既に勤務窓を持つ個体の窓は**上書きしない**。
3. **当直帯の外は無人になる**。既定 2 直(05:00〜13:00 / 13:00〜21:00)なので終電帯には
   乗務員が居ない。そこでは ``dwell_decision`` が ``agent_id=-1`` かつ
   ``unstaffed=true`` で出る(= **黙って進めない**。自動運転/指令所に相当する
   「運行装置」が決めたことを正直に記録する)。
4. **運行時間外は注入しない**。``dwell.require_service: true``(既定)では
   ``has_service()`` が False の step で停車時間を**足さない**(閉めるドアが無いから)。
   規則1 は運行の有無を見ずに注入していたので、ここだけは意図的な差である
   (``require_service: false`` にすれば規則1 と全 step で完全同値になる = テストで固定)。
"""
from __future__ import annotations

from . import envfeedback as envfb
from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種の**材料側**登録(devices.py / traces.py / rumors.py と同じ流儀。
# observer/schema.py には 1 バイトも書かない)。
# --------------------------------------------------------------------------- #
register_event_kind(
    "dwell_decision",
    "当直の車掌によるドア閉判断(= 停車時間延長 → 遅延)。agent_id = 乗務員"
    "(不在なら -1 かつ payload.unstaffed=true・payload.operator=運行装置 id)"
    " {node, line, n_lines, platform_load, standing, exchange, threshold, excess,"
    " delay_s, delay_min, prev_min, calibrated, unstaffed}")
register_event_kind(
    "transit_staff_bound",
    "駅員・乗務員を駅ノードの持ち場へ束ねた統計。**起動時 1 回**(step 0 のみ)"
    " {node, n_station, n_crew, n_kept}")

# 乗務員が 1 人も居ない回の記録主体(= 自動運転 / 運行指令に相当する装置)。
# ★**装置の実体は作らない**: 本波は列車エンティティを持たないので、置けるのは
#   「誰が決めたことになっているか」を辿れる安定 id だけである(捏造しない)。
# ★接頭辞を ``operator`` にしないのは、``world.devices.transit_operator``(IF-F W2 =
#   **運休を決める**運行事業者装置 ``operator:transit``)と別物だからである。あちらは
#   「今日は止める」を決める主体、こちらは「乗務員が居ないまま閉まったドア」の記録主体。
#   後続波で両者を 1 台に畳むなら、その判断は装置レジストリ側で行う。
OPERATOR_PREFIX = "train_op"


def operator_device_id(node: str) -> str:
    """駅ノード → 運行装置 id(**安定 id**。ランを跨いでも同じ名前になる)。"""
    return f"{OPERATOR_PREFIX}:{node}"


# =========================================================================== #
# cfg 正準化(devices.build_cfg と同型: dict / OmegaConf 両対応・未知キーは捨てる)
# =========================================================================== #
DEFAULTS = {
    "enabled": False,
    "bind": {
        "enabled": True,
        "station_occupations": ("駅員",),
        "crew_occupations": ("車掌", "電車運転士"),
        "first_open": "05:00",           # 最初の直の始まり(初電の帯)
        "shift_hours": 8,                # 1 直の長さ[時間](pool の duty_pattern と同値)
        "n_shifts": 2,                   # 1 日に回す直の数(名簿順の index で割り振る)
    },
    "dwell": {
        "enabled": True,
        "require_service": True,         # 運行が無い step は停車時間を足さない
        "calibrated": False,             # 東京実証則の係数を使うか(既定 = 規則1 の係数)
        "dwell_s_per_15pax": 1.0,        # +15 人 ≒ +1 秒(Palmqvist/Tomii/Ochiai 2020)
        "log_every_steps": 1,            # dwell_decision の記録間引き(観測装置の設定)
    },
}


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def _occ_tuple(raw, fallback) -> tuple:
    """職業名リストの正準化(空白除去・空要素落とし・**宣言順を保つ**)。空なら既定へ戻す。"""
    got = _to_plain(raw)
    if isinstance(got, str):
        got = [got]
    out = tuple(str(x).strip() for x in (got or ()) if str(x).strip())
    return out or tuple(fallback)


def build_cfg(raw) -> dict:
    """conf の ``transit_staff`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``devices.build_cfg`` と同じ作法)。
    未知のキーは黙って捨てる(捏造しない)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = {"enabled": bool(raw.get("enabled", False)),
           "bind": dict(DEFAULTS["bind"]), "dwell": dict(DEFAULTS["dwell"])}
    got = dict(_to_plain(raw.get("bind")) or {})
    bind = cfg["bind"]
    if "enabled" in got:
        bind["enabled"] = bool(got["enabled"])
    if "station_occupations" in got:
        bind["station_occupations"] = _occ_tuple(
            got["station_occupations"], DEFAULTS["bind"]["station_occupations"])
    if "crew_occupations" in got:
        bind["crew_occupations"] = _occ_tuple(
            got["crew_occupations"], DEFAULTS["bind"]["crew_occupations"])
    if "first_open" in got:
        bind["first_open"] = str(got["first_open"])
    if "shift_hours" in got:
        bind["shift_hours"] = max(1, int(got["shift_hours"]))
    if "n_shifts" in got:
        bind["n_shifts"] = max(1, int(got["n_shifts"]))
    got = dict(_to_plain(raw.get("dwell")) or {})
    dwell = cfg["dwell"]
    if "enabled" in got:
        dwell["enabled"] = bool(got["enabled"])
    if "require_service" in got:
        dwell["require_service"] = bool(got["require_service"])
    if "calibrated" in got:
        dwell["calibrated"] = bool(got["calibrated"])
    if "dwell_s_per_15pax" in got:
        dwell["dwell_s_per_15pax"] = max(0.0, float(got["dwell_s_per_15pax"]))
    if "log_every_steps" in got:
        dwell["log_every_steps"] = max(1, int(got["log_every_steps"]))
    return cfg


def cfg_of(sim) -> dict:
    """駅員・車掌の設定(初回のみ ``sim.cfg.transit_staff`` から遅延構築してキャッシュ)。

    ``simulation.py`` は読み取り専用なので cfg は本 module が遅延構築する
    (``devices.cfg_of`` / ``traces.cfg_of`` と同型)。キャッシュ属性
    ``sim.transitstaffcfg`` は L1/L2/L3/乱数に一切現れない。
    """
    got = getattr(sim, "transitstaffcfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("transit_staff", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.transitstaffcfg = got
    return got


def enabled(sim) -> bool:
    """駅員・車掌アクター層が有効か。既定 OFF=新経路を一切通らない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


def dwell_loop_active(sim) -> bool:
    """停車時間ループを**本 module が所有しているか**。

    ★``envfeedback.update`` は**この 1 関数だけ**を見て規則1(ホーム密度 → 停車時間)を
      評価するか決める(主体なしの近似と主体つきモデルの二重適用の防止)。

    ★``env.feedback`` 側が OFF のときは **False を返す**(所有しない)。理由: 遅延の
      下流の消費者(``hold_exit`` / ``hold_return`` / ``delay_flag``)は全部 envfeedback に
      あり、あちらが OFF ならどれも即 return する。そこで本 module だけ ON にしても
      「誰も読まない数字」を書くだけで**ループが閉じない**ので、黙って走らせない。
    """
    cfg = cfg_of(sim)
    if not (cfg["enabled"] and cfg["dwell"]["enabled"]):
        return False
    ecfg = getattr(sim, "envfbcfg", None)
    return bool(ecfg and ecfg["enabled"] and ecfg["transit"]["enabled"])


# =========================================================================== #
# (1) 持ち場への束ね(駅員 = 駅ノード / 乗務員 = 駅ノード)
# =========================================================================== #
def _station(sim):
    city = getattr(sim, "city", None)
    return getattr(city, "station_node", None) if city is not None else None


def _hhmm(text, default: int) -> int:
    """"HH:MM" → 分 of day(不正なら default)。work.py の同名補助と同じ規約。"""
    try:
        h, m = str(text).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError, TypeError):
        return int(default)


def duty_window(bcfg: dict, index: int) -> tuple[int, int]:
    """当直の窓 ``(open, close)``[分 of day]。**名簿順 index の純関数**(乱数ゼロ)。

    ``first_open`` から ``shift_hours`` ごとに ``n_shifts`` 直を並べ、名簿順(agent id
    昇順)の index を ``% n_shifts`` で割り振る。日を跨ぐ直は現行の勤務窓判定
    (``routine.in_work_window`` は ``start <= 分 of day < end``)が表現できないので
    24:00 で切る(= 限界 3 に明記した「当直帯の外は無人」)。
    """
    n = max(1, int(bcfg["n_shifts"]))
    hours = max(1, int(bcfg["shift_hours"]))
    first = _hhmm(bcfg["first_open"], 5 * 60)
    o = first + (int(index) % n) * hours * 60
    if o >= 1440:                                  # 直が日を跨いだら最初の直へ畳む
        o = first
    return o, min(1440, o + hours * 60)


def _roles(cfg) -> tuple[frozenset, frozenset]:
    b = cfg["bind"]
    return frozenset(b["station_occupations"]), frozenset(b["crew_occupations"])


def is_crew(sim, agent) -> bool:
    """乗務員(車掌・電車運転士)か。OFF でも純粋な述語として使える(状態を触らない)。"""
    _, crew = _roles(cfg_of(sim))
    return str(getattr(agent, "occupation", "")) in crew


def bind(sim) -> dict:
    """駅員・乗務員を駅ノードの持ち場へ束ねる(決定論・乱数ゼロ・**冪等**・resume 安全)。

    ``work.bind_workplace``(別レーン所有)は地図に対応 POI カテゴリが無い職業を
    ``no_category`` として**正直に束ねられないまま**残す。本関数はその後に走り、
    ゲートが ON のときだけ「駅という持ち場」を与える(= あちらの統計の意味は変えない)。

    - 並びは **agent id 昇順**(当直表の index もこの並びで決まる = 乱数ゼロ)。
    - 既に駅ノードに勤務窓つきで立っている個体は **1 バイトも触らない**(冪等)。
    - 勤務窓を持たない個体にだけ合成当直表の窓を与える(既存の窓は上書きしない)。
    - 駅は路面ノード扱い(建物内勤務にしない)= ``work_building`` は空に揃える。
    """
    cfg = cfg_of(sim)
    out = {"node": "", "n_station": 0, "n_crew": 0, "n_kept": 0}
    if not (cfg["enabled"] and cfg["bind"]["enabled"]):
        return out
    station = _station(sim)
    if not station:
        return out
    out["node"] = str(station)
    st_occ, crew_occ = _roles(cfg)
    both = st_occ | crew_occ
    targets = [a for a in sorted(sim.agents, key=lambda a: int(a.id))
               if str(getattr(a, "occupation", "")) in both]
    for index, agent in enumerate(targets):
        placed = (str(getattr(agent, "work_node", "")) == str(station)
                  and int(getattr(agent, "work_start_min", -1)) >= 0)
        if placed:
            out["n_kept"] += 1
        else:
            agent.work_node = str(station)
            agent.work_building = ""               # 駅は路面ノード(建物内勤務にしない)
            agent.work_floor = 0
            if int(getattr(agent, "work_start_min", -1)) < 0:
                o, c = duty_window(cfg["bind"], index)
                agent.work_start_min = o
                agent.work_end_min = c
        if str(getattr(agent, "occupation", "")) in crew_occ:
            out["n_crew"] += 1
        else:
            out["n_station"] += 1
    return out


# =========================================================================== #
# (2) 当直の乗務員(**決定論の選定**: 最小 agent id)
# =========================================================================== #
def on_duty_crew(sim, sim_min: int):
    """この step にドア閉を判断する乗務員。居なければ ``None``。

    条件(すべて観測可能な状態の純関数・乱数ゼロ):
      ① 乗務員職(``crew_occupations``)である
      ② **駅ノードに居る**(街の中・起きている)
      ③ **勤務窓の中**(``routine.in_work_window`` = 既存の勤務判定をそのまま使う。
         病気・平日ゲートもあちらの規約に従う)
    複数居るときは **最小 agent id**(名簿の先頭 = 到着順にも乱数にも依存しない)。
    """
    cfg = cfg_of(sim)
    station = _station(sim)
    if not station:
        return None
    _, crew_occ = _roles(cfg)
    if not crew_occ:
        return None
    from .cognition import routine as _routine     # 遅延 import(循環と重い import の回避)
    cal = getattr(sim, "calendarcfg", None)
    best = None
    for agent in sim.agents:
        if str(getattr(agent, "occupation", "")) not in crew_occ:
            continue
        if agent.loc == "outside" or agent.sleeping or agent.node != station:
            continue
        if not _routine.in_work_window(agent, sim_min, cal):
            continue
        if best is None or int(agent.id) < int(best.id):
            best = agent
    return best


# =========================================================================== #
# (3) 単一作用点(step 末)
# =========================================================================== #
def _lines(sim, sim_min: int) -> list:
    transit = getattr(sim, "transit", None)
    if transit is None:
        return []
    return list(transit.lines_in_service(int(sim_min)))


def _in_service(sim, sim_min: int) -> bool:
    transit = getattr(sim, "transit", None)
    return True if transit is None else bool(transit.has_service(int(sim_min)))


def _sec_per_pax(dcfg: dict):
    """停車時間の係数[秒/人]。既定は None = 規則1 の係数をそのまま使う。

    ``calibrated: true`` のときだけ東京実証則(+15 人 ≒ +1 秒)へ差し替える。
    ★係数を返すだけ = 式・上限・回復運転項は ``envfeedback`` 側の共有関数のまま。
    """
    if not dcfg["calibrated"]:
        return None
    return float(dcfg["dwell_s_per_15pax"]) / 15.0


def _dwell(sim, step: int, sim_min: int, since_idx: int) -> None:
    """ホーム負荷 → **当直の車掌のドア閉判断** → ``delay_min``(規則1 と同じ状態)。"""
    if not dwell_loop_active(sim):
        return
    station = _station(sim)
    if station is None:
        return
    dcfg = cfg_of(sim)["dwell"]
    tcfg = sim.envfbcfg["transit"]
    st = envfb.state(sim)
    prev = float(st["delay_min"])
    # 集約は envfeedback と**同じ関数**(規則1 が見た負荷と 1 も違わないための唯一の担保)
    at_node, _inflow, exchange = envfb.aggregate(sim, station, since_idx)
    load = envfb.platform_load(at_node, station, exchange)
    excess, inject, new = envfb.dwell_step(tcfg, load, prev, _sec_per_pax(dcfg))
    door = _in_service(sim, sim_min) or not dcfg["require_service"]
    if not door:
        # 電車が居ない = 閉めるドアが無い。停車時間は足さず回復運転項だけを回す。
        inject, new = 0.0, envfb.advance_delay(tcfg, prev, 0.0)
    envfb.commit_delay(st, new, excess if door else 0)
    if not door or int(step) % int(dcfg["log_every_steps"]) != 0:
        return
    if not (excess > 0 or (prev > 0.0 and new == 0.0)):
        return                                     # 規則1 と同じ記録条件(L1 の量を抑える)
    crew = on_duty_crew(sim, sim_min)
    lines = _lines(sim, sim_min)
    payload = {"node": str(station), "line": lines[0] if lines else "",
               "n_lines": len(lines), "platform_load": int(load),
               "standing": int(at_node.get(station, 0)), "exchange": int(exchange),
               "threshold": int(tcfg["platform_threshold"]), "excess": int(excess),
               "delay_s": round(inject * 60.0, 3), "delay_min": round(new, 3),
               "prev_min": round(prev, 3), "calibrated": bool(dcfg["calibrated"]),
               "unstaffed": crew is None}
    if crew is None:
        payload["operator"] = operator_device_id(str(station))
        x, y = sim.city.node_xy(station)
        agent_id = -1
    else:
        x, y = float(crew.x), float(crew.y)
        agent_id = int(crew.id)
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=agent_id,
                         kind="dwell_decision", x=float(x), y=float(y),
                         payload=payload))


def phase(sim, step: int, sim_min: int, since_idx: int = -1) -> None:
    """駅員・車掌アクターの**単一作用点**。既定 OFF は即 return(バイト一致)。

    置き場所は ``envfeedback.update`` の**直後**(scheduler の step 末)。理由:
      - 停車時間は「その step のホーム負荷」から決まる **step 末の集約量**なので、
        step 途中で計算すると処理順が結果に混入する(第81 同期バリアの規約)。
      - 規則1 は ``update`` の中で本 module へ譲っている(``_staff_dwell_active``)ので、
        譲られた側は同じ step のうちに ``delay_min`` を確定させなければならない
        (下流の観測チャンネル ``ext.transit_delay`` はこの直後に読む)。
    ★束ねは最初の呼び出しで 1 回だけ(冪等なので resume で 2 度走っても同じ)。L1 の
      ``transit_staff_bound`` は **step 0 のときだけ**出す(``workplace_bound`` が
      ``step != 0`` ガードで resume の二重記録を防いでいるのと同型)。
    ★世界状態(所持金・位置・関係・drive・opinion)は 1 つも動かさない。
    """
    if not enabled(sim):
        return
    if not getattr(sim, "_transit_staff_bound", False):
        stat = bind(sim)
        sim._transit_staff_bound = True
        if stat["node"] and int(step) == 0:
            sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=-1,
                                 kind="transit_staff_bound", x=0.0, y=0.0,
                                 payload=dict(stat)))
    _dwell(sim, step, sim_min, since_idx)
