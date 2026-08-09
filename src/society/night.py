"""夜(00:00-05:00)を表現可能にする層(第101バッチ Wave4 III-1「夜間開放」。既定 OFF)。

監査の指摘: 現行の街は **構造的に夜が無い**。
  (1) 日跨ぎシフトが表現できない  … ``work._window`` が「閉<=開」を open+8h へ丸めるので
      22:00→06:00 の夜勤が原理的に作れず、組織台帳のシフトは 4 種・最遅 close=23:00 しか無い。
  (2) 24 時間営業が表現できない    … ``commerce`` の営業時間表はカテゴリ 3 種
      (food 11-23 / shop 10-21 / nightlife 18-05)だけで、コンビニは汎用 ``shop`` として
      21:00-10:00 に強制閉店する。
  (3) 夜勤者が 1 人も居ない        … バイトは <=22:00・住民は 22:00-01:50 に就寝・
      来街者は ~24:00 までに全員退去。
  (4) 終電後に街に居られない      … 終電を逃した homing 個体は縁のゲートウェイまで歩いて
      **世界から出る**(scheduler の放棄経路)。実際の渋谷では終電後にネットカフェ・サウナ・
      カラオケが受け皿になる(終電 ~24:40 / 始発 04:34)。

本モジュールは (1)(2)(4) の**判断**と (2) の**時間表**を 1 か所に閉じ、既定 OFF
(``world.night_economy.enabled=false``)では 1 バイトも動かさない。(3) は生成器側
(scripts/build_orgs.py の夜勤シフト語彙 + scripts/build_persona_pool.py の L2 継承)で解く。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world/factors の禁則
  ディレクトリ外)。夜間経済の語彙(24h 営業・夜勤・避難先)をここに閉じる。

R1(呼数不変・決定論):
  - generate() を 1 本も足さない。避難先の選定は (距離, node id) の純関数 = **乱数ゼロ**
    (新 stream を 1 本も引かない)。営業時間は時刻からの純関数。
  - 日跨ぎ勤務窓は agent の属性 ``work_wraps`` で表現する。既定 OFF では**属性自体が生えない**
    ので ``routine.in_work_window`` の分岐は getattr の既定値で従来式へ落ちる(= バイト一致)。
  - 避難は物理位置 = 対面 co-location を変えうる(commerce H3 / lodging Wave L と同型)が、
    判断材料は 時刻・config・物理位置・運行の有無 のみ = k 非依存。

OFF が保つ不変条件: night_refuge 0 件・agent に属性が生えない・営業時間表が 1 エントリも
  変わらない・勤務窓の丸めが従来コードのまま(L1 バイト一致 golden_baseline_l1.json)。
"""
from __future__ import annotations

from .observer.schema import Event, register_event_kind

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: devices.py / transit_staff.py / street_life.py と
# 同じ流儀 = observer/schema.py 本体を触らずに自分の種を自分で登録する)。
# ★1 個体 1 晩あたり 1 件だけ(退出は既存の exit_area で観測できるので種を足さない)。
# ★payload に自由文は 1 つも入らない(poi は地図の実名 = 既存 lodging_checkin と同格、
#   node は世界の識別子、残りは時刻と真偽値)。
# --------------------------------------------------------------------------- #
register_event_kind("night_refuge", "終電を逃した個体が夜の避難先(ネットカフェ/サウナ/"
                                    "カラオケ等)へ入り、始発まで街に留まった"
                                    "{poi, node, until_min, first_train}")

# ---- カテゴリ/サブカテゴリ別の夜間営業時間 [開店時, 閉店時](24h。開==閉 = 24 時間営業)----
# ★キーは POI の ``cat`` **または** ``subcat`` のどちらにも当たる(subcat が優先)。
#   v7 地図(data/shibuya_osm_wide_v7.json)の POI は cat しか持たない(build_map.poi_category が
#   OSM の ``shop=convenience`` を汎用 ``shop`` へ潰しており、生タグは保存されていない)。
#   したがって **既定の subcat 系エントリは v7 では 1 件も当たらない**(= 完全な no-op)。
#   これは仕様であって取りこぼしではない: 地図 v8 が ``subcat`` を通すか、運用者が
#   ``subcat_keywords`` を書いた時点で、この表を書き換えずにそのまま効きはじめる。
DEFAULT_HOURS = {
    # --- v8(または subcat_keywords)で活きるサブカテゴリ ---
    "convenience": [0, 0],     # コンビニ = 24 時間営業(開==閉)
    "net_cafe": [0, 0],        # ネットカフェ・複合カフェ = 24 時間(終電後の受け皿)
    "sauna": [0, 0],           # サウナ・カプセル = 24 時間
    "karaoke": [11, 5],        # カラオケ 11:00〜翌 05:00
    "club": [22, 5],           # クラブ 22:00〜翌 05:00
    # --- v7 でも当たるカテゴリ(commerce の既定表を上書きする)---
    "nightlife": [18, 5],      # 既定と同値(明示。ここを触れば夜遊びの窓が一括で動く)
}

DEFAULT_REFUGE_CATS = ["nightlife"]                     # v7 で実在するカテゴリ
DEFAULT_REFUGE_SUBCATS = ["net_cafe", "sauna", "karaoke"]   # v8 で活きるサブカテゴリ

DEFAULTS = {
    "enabled": False,
    # 日跨ぎシフト(close<open を「翌朝まで」と読む)。OFF のときは work._window の
    # 従来コード(閉<=開 → open+8h へ丸める)を**文字どおり**通す。
    "midnight_shift": True,
    "hours": DEFAULT_HOURS,
    # サブカテゴリ判定の名称キーワード(subcat → キーワード列)。**既定は空**。
    # ★ETHICS §2: 実在企業名(ブランド名)をソース/config へ書かない。v7 の POI 名で
    #   一般名詞マッチ(「ストア」「マート」等)を試すと Apple Store / ABC マート等を
    #   コンビニと誤判定するので(実測: 一般名詞ヒット 21 件のうちコンビニは 4 件)、
    #   **既定では 1 件も分類しない**。運用者が明示的に書いたときだけ効く。
    "subcat_keywords": {},
    "refuge": {
        "enabled": True,
        "cats": DEFAULT_REFUGE_CATS,
        "subcats": DEFAULT_REFUGE_SUBCATS,
        "max_dist_m": 800.0,       # 駅からこの距離までの避難先だけを使う(徒歩圏)
        "max_stay_min": 480,       # 始発が見つからないとき(運休等)の滞在上限 [分]
    },
}


def build_cfg(raw) -> dict:
    """conf の ``world.night_economy`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(commerce / lodging / health と同型)。
    hours は cat|subcat → [開店時, 閉店時](24h)で受け、内部では (open_min, close_min) 分に
    変換して保持する(commerce.build_cfg と同じ規約)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    hours_raw = raw.get("hours", DEFAULT_HOURS)
    hours: dict[str, tuple[int, int]] = {}
    for key, hc in dict(hours_raw or {}).items():
        hours[str(key)] = (int(hc[0]) * 60, int(hc[1]) * 60)
    kw_raw = dict(raw.get("subcat_keywords", {}) or {})
    ref = dict(raw.get("refuge", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "midnight_shift": bool(raw.get("midnight_shift", True)),
        "hours": hours,
        "subcat_keywords": {str(k): [str(w) for w in (v or [])]
                            for k, v in kw_raw.items()},
        "refuge": {
            "enabled": bool(ref.get("enabled", True)),
            "cats": [str(c) for c in (ref.get("cats") or DEFAULT_REFUGE_CATS)],
            "subcats": [str(c) for c in (ref.get("subcats")
                                         or DEFAULT_REFUGE_SUBCATS)],
            "max_dist_m": float(ref.get("max_dist_m", 800.0)),
            "max_stay_min": max(1, int(ref.get("max_stay_min", 480))),
        },
    }


def enabled(sim) -> bool:
    """夜間経済(第101 III-1)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "nightcfg", None)
    return bool(cfg and cfg["enabled"])


def wire_work(ncfg: dict | None, workcfg: dict | None) -> None:
    """夜間ゲートを work 側(勤務窓の丸め)へ 1 フラグだけ渡す(配線は simulation が 1 回だけ)。

    ``work.build_cfg`` は conf の ``work`` ブロックしか見えないので、``world.night_economy``
    の可否をここで注入する。OFF(または未設定)では False = ``work._window`` は従来の
    「閉<=開 → open+8h」の丸めをそのまま通す。"""
    if not workcfg:
        return
    on = bool(ncfg and ncfg["enabled"] and ncfg["midnight_shift"])
    workcfg["bind_workplace"]["midnight_shift"] = on


# ---------------------------------------------------------------- 営業時間(時刻の純関数)
def poi_subcat(cfg: dict, poi: dict) -> str | None:
    """POI のサブカテゴリ(コンビニ/ネカフェ/サウナ…)。判らなければ None。

    解決順(**推測しない**):
      ① POI 自身の ``subcat`` フィールド(地図 v8 が OSM の ``shop=convenience`` 等を
         そのまま通したときに現れる。現行 v7 には存在しない)
      ② config の ``subcat_keywords``(運用者が明示的に書いたキーワード。既定は空)
    どちらも無ければ None = カテゴリ既定の営業時間で扱う。"""
    sub = poi.get("subcat")
    if sub:
        return str(sub)
    kw = cfg["subcat_keywords"]
    if not kw:
        return None
    name = str(poi.get("name", ""))
    if not name:
        return None
    for sc in sorted(kw):                       # 決定論(キー昇順で最初に当たったもの)
        for word in kw[sc]:
            if word and word in name:
                return sc
    return None


def cat_hours(ccfg: dict, ncfg: dict | None) -> dict:
    """カテゴリ単位の実効営業時間表(commerce の既定表 + 夜間の上書き)。

    夜間 OFF(または None)なら **commerce の表そのもの**(同一オブジェクト)を返す
    = 呼び出し側の走査順・内容とも従来どおり(バイト一致)。"""
    if not (ncfg and ncfg["enabled"]):
        return ccfg["hours"]
    merged = dict(ccfg["hours"])
    for key, hc in ncfg["hours"].items():
        if key in merged:                       # cat に当たるキーだけを上書きする
            merged[key] = hc
    return merged


def poi_hours(ccfg: dict, ncfg: dict | None, poi: dict):
    """その POI の実効営業時間 (open_min, close_min)。ゲート無し(常時営業)なら None。

    優先順: subcat の上書き → cat の上書き → commerce のカテゴリ既定表。"""
    if ncfg and ncfg["enabled"]:
        sub = poi_subcat(ncfg, poi)
        if sub is not None and sub in ncfg["hours"]:
            return ncfg["hours"][sub]
        cat = str(poi.get("cat", ""))
        if cat in ncfg["hours"] and cat in ccfg["hours"]:
            return ncfg["hours"][cat]
    return ccfg["hours"].get(str(poi.get("cat", "")))


# ---------------------------------------------------------------- 終電後の避難先(refuge)
def refuge_on(sim) -> bool:
    cfg = getattr(sim, "nightcfg", None)
    return bool(cfg and cfg["enabled"] and cfg["refuge"]["enabled"])


def _poi_is_refuge(ncfg: dict, poi: dict) -> bool:
    sub = poi_subcat(ncfg, poi)
    if sub is not None and sub in ncfg["refuge"]["subcats"]:
        return True
    return str(poi.get("cat", "")) in ncfg["refuge"]["cats"]


def refuge_pois(sim) -> list:
    """避難先候補 POI の一覧(決定論の安定順・sim に 1 度だけキャッシュ)。

    **駅ノードの POI は候補にしない**: 避難は「駅から歩いて移る」行為で、駅に留まるのは
    避難ではない。加えて、駅を除いておくと始発時に必ず「避難先 → 駅」の経路が要る形になり、
    帰路の経路を張り忘れる分岐が構造的に消える(下の ``check_in`` の前提)。"""
    cache = getattr(sim, "_night_refuge_pois", None)
    if cache is None:
        ncfg = sim.nightcfg
        station = str(getattr(sim.city, "station_node", "") or "")
        seen: dict[str, dict] = {}
        for cat in sorted(set(ncfg["refuge"]["cats"])):
            for p in sim.city.pois_by_cat(cat):
                seen[str(p.get("id", ""))] = p
        if ncfg["refuge"]["subcats"]:            # subcat 経路(v8 / keywords)は全 POI を走査
            for p in sim.city.poi_list:
                if _poi_is_refuge(ncfg, p):
                    seen[str(p.get("id", ""))] = p
        cache = sorted((p for p in seen.values()
                        if str(p.get("node", "")) != station),
                       key=lambda p: (str(p.get("node", "")), str(p.get("id", ""))))
        sim._night_refuge_pois = cache
    return cache


def poi_open(sim, poi: dict, sim_min: int) -> bool:
    """夜間層から見たその POI の開閉(commerce.enabled とは独立に評価する)。

    避難先が開いているかは「街の側の事実」であって商業ダイナミクスの ON/OFF とは別問題
    なので、commerce が OFF のランでも同じ表で判断する(判断材料を 1 か所に保つ)。"""
    from . import commerce as _commerce
    hc = poi_hours(sim.commercecfg, getattr(sim, "nightcfg", None), poi)
    if hc is None:
        return True
    return _commerce.is_open_window(hc, sim_min)


def nearest_refuge(sim, agent, sim_min: int):
    """(x, y) から最も近い**営業中の**避難先 POI。決定論(距離 → node id → poi id の順)。

    半径 ``max_dist_m`` の外は見ない(徒歩圏の外まで歩くのは終電後の行動として不自然)。"""
    ncfg = sim.nightcfg
    lim = float(ncfg["refuge"]["max_dist_m"]) ** 2
    best = None
    best_key = None
    for p in refuge_pois(sim):
        if not poi_open(sim, p, sim_min):
            continue
        px, py = sim.city.node_xy(p["node"])
        d = (px - agent.x) ** 2 + (py - agent.y) ** 2
        if d > lim:
            continue
        key = (d, str(p.get("node", "")), str(p.get("id", "")))
        if best_key is None or key < best_key:
            best_key, best = key, p
    return best


def next_service_min(sim, sim_min: int, max_stay_min: int) -> tuple[int, bool]:
    """次に運行が始まる(= 始発)までの待ち分数。見つからなければ (max_stay_min, False)。

    判定は **既存の運行述語 ``transit.has_service``** をそのまま使う(始発時刻の定義を
    この層で作り直さない)。走査は Δt 刻み = 呼び出し 1 回あたり高々 max_stay_min/Δt 回。
    運休(災害)で 1 日中サービスが無い日でも上限で必ず打ち切る(街に固まらない安全弁)。"""
    dt = max(1, int(sim.clock.step_minutes))
    limit = max(1, int(max_stay_min))
    m = dt
    while m <= limit:
        if sim.transit.has_service(int(sim_min) + m):
            return m, True
        m += dt
    return limit, False


def take_refuge(sim, agent, step: int, sim_min: int) -> bool:
    """終電後の homing 個体を、縁のゲートウェイではなく夜間の避難先へ向かわせる。

    True を返したら呼び出し側(scheduler._try_exit)は退出処理をしない。前提が 1 つでも
    欠ければ **何も触らずに False**(= 従来の「縁まで歩いて退出」へ落ちる)。
    乱数を 1 本も引かない(選定は距離と識別子の純関数)。"""
    if not refuge_on(sim):
        return False
    if not getattr(agent, "homing", False):        # 夜の帰宅の瞬間のみ(日中の退出は対象外)
        return False
    if getattr(agent, "_night_refuge_node", ""):   # 既に避難先へ向かっている
        return False
    poi = nearest_refuge(sim, agent, sim_min)
    if poi is None:
        return False
    path, used_mode = sim.router.route(agent.node, poi["node"], "walk")
    if len(path) < 2:                              # 到達不能 → 従来の退出へ後退
        return False
    agent.exit_intent = False       # 到着時に _try_exit を呼ばせない(縁でないノードで退出しない)
    agent.homing = False
    agent._night_refuge_node = str(poi["node"])
    agent._night_refuge_poi = str(poi.get("name") or "")
    agent.route = path[1:]
    agent.edge_offset = 0.0
    agent.trip_mode = used_mode
    agent._pending_stay = 0
    agent._pending_activity = ""
    agent._ride_pending = None
    return True


def on_arrive(sim, agent, step: int, sim_min: int) -> None:
    """避難先に着いた step の 1 点(scheduler の到着処理から呼ばれる)。OFF/非該当は no-op。"""
    node = getattr(agent, "_night_refuge_node", "")
    if not node or str(agent.node) != str(node):
        return
    _check_in(sim, agent, step, sim_min)


def _check_in(sim, agent, step: int, sim_min: int) -> None:
    """避難先で始発まで過ごす(既存の睡眠機構を使う)+ 始発で駅へ戻る経路を張っておく。

    ★内省(reflect_step)は**立てない**: 立てると LLM 呼が 1 本増えて R1(呼数不変)を破る。
    ★建物には入らない(路面の POI ノードに留まる): 入館すると起床時に退館 → 経路再張りの
      分岐が増える。夜の街に「居る」ことが目的なので路面で足りる。"""
    ncfg = sim.nightcfg
    wait_min, found = next_service_min(sim, sim_min, ncfg["refuge"]["max_stay_min"])
    agent._night_refuge_node = ""
    poi_name = getattr(agent, "_night_refuge_poi", "")
    agent._night_refuge_poi = ""
    agent.sleeping = True
    agent.activity = ""
    agent.stay_until = step
    agent.sleep_until = step + max(1, sim.clock.min_to_steps(wait_min))
    # 起床(= 始発)後は駅へ戻って通常の退出経路に乗る。経路は今のうちに張っておく
    # (睡眠中は _phase_move が飛ばすので消費されない = 起床した step から歩き出す)。
    agent.exit_intent = True
    agent.homing = True
    station = getattr(sim.city, "station_node", "")
    if station and str(agent.node) != str(station):
        path, used_mode = sim.router.route(agent.node, station, "walk")
        if len(path) >= 2:
            agent.route = path[1:]
            agent.edge_offset = 0.0
            agent.trip_mode = used_mode
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="night_refuge", x=agent.x, y=agent.y,
                         payload={"poi": poi_name,
                                  "node": str(agent.node),
                                  "until_min": int(sim_min) + int(wait_min),
                                  "first_train": bool(found)}))
