"""都市・環境インフラのショック(現実ギャップ 後続波 H4 2026-07-07。ユーザー要望)。

生活を麻痺させる外生ショック=集団的 grievance の源。最小・非LLM・決定論で載せる層。
既存の scenario/transit の最小拡張として3機構(都市再開発=地図が時間で変わる機構は本波では
作らない=注記に留める):

  1. 災害ショック(disaster): 台風・地震・大雪など。発生日/確率で発動し、発生中は
     交通麻痺(電車運休=transit.has_service を止める)+ 外出抑制(在宅=routine が自宅へ寄せる)+
     強い grievance の集中(factors 経由)+ プロンプト文脈(「台風で交通が麻痺…」)。発生/解除を
     disaster(agent_id=-1, phase=onset/clear)で記録する。
  2. 交通の遅延・遅刻・運休(transit_delay): 電車は現状 定刻(乱れなし)。日次に遅延/運休を
     注入する(新 stream "disaster" で乱す=既存の定刻ダイヤの draw を汚さない)。運休の日は
     transit.has_service を止める。乗れない/遅れる→grievance(factors 経由)。transit_delay を記録。
  3. インフラ障害(infra_outage): 停電・通信断・断水=生活麻痺。軽実装=在宅娯楽(media)・
     スマホ(SNS/ニュース/検索)の一部を抑制 + grievance。発生/解除を infra_outage で記録。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  災害・遅延・障害のロジック(grievance の factors hook 呼び出しを含む)をここに閉じる=
  engine/cognition/world に構成概念名を書かない(no-fingerprint 契約に触れない)。grievance は
  factors 層 hook(on_disaster)へ**不透明な magnitude** だけを渡す。

R1 呼数不変: どの機構も generate() を1本も足さない。発生/遅延/障害の確率判定は**新 stream
  "disaster"**(既存 draw 順に挿入しない=決定論・ゴールデン保護)、発生日指定は暦不要の day-index
  からの純関数(乱数なし)。災害(在宅)・運休は物理位置・移動を変え対面 co-location が変化しうる
  (=FixedLLM で ON!=OFF になりうる=career G5 / crowd G4 / 健康 H1 / 世帯 H2 / 商業 H3 と同型)が、
  機構は k・内面状態(構成概念)を発火判断に食わせず day-index・config・新 stream・物理位置のみ参照する
  =compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。grievance は drive(発火系)には
  接続しない(呼数を1本も動かさない)。

既定 OFF(enabled=false)= 発生/遅延/障害なし・在宅抑制なし・transit 運休なし・"disaster" stream も
  引かない・イベント 0 件・乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。既存の摂動シナリオ
  (shock_closure/shock_event)の挙動・イベント列も一切変えない(本波は独立の phase として動く)。
  新イベント種は disaster / transit_delay / infra_outage(schema.py 登録済み=編集しない)。
"""
from __future__ import annotations

from . import devices as devices_mod
from .factors import update as factor_update
from .observer.schema import Event


def build_cfg(raw) -> dict:
    """conf の disaster ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(career/health/household/commerce と同型)。すべて既定 OFF
    (enabled=false)で scheduler の disaster 日次フェーズ・routine の在宅抑制・transit 運休・media/phone の
    障害抑制が完全 no-op。

    パラメータ(ON 時のみ効く):
      災害: days             発生日の day-index リスト(run 開始=0。決定論・乱数不要)
            onset_prob       days 非該当日の発症確率(新 stream "disaster"。既定 0=確率発生なし)
            duration_days    1回の災害の継続日数
            kind             災害の種類(台風/地震/大雪。プロンプト・イベントの語)
            grievance        災害中の不満増分(不透明 magnitude。1日1回。0.0=観測のみ)
            stay_home_bias   災害中に住民が在宅(外出抑制)になる確率(新 stream "disaster"。0=抑制なし)
            suspend_transit  災害中に電車を運休にするか(交通麻痺。既定 true)
      交通遅延: delay_prob         日次の遅延/運休発生確率(新 stream "disaster"。既定 0=なし)
                suspend_prob       遅延日のうち運休(has_service 停止)になる確率(残りは遅延=grievanceのみ)
                delay_grievance    遅延/運休に遭遇した街内の人の不満増分(不透明 magnitude。0.0=観測のみ)
      インフラ: outage_prob        日次の障害発生確率(新 stream "disaster"。既定 0=なし)
                outage_kind        障害の種類(停電/通信断/断水)
                outage_duration_days 障害の継続日数
                outage_grievance   障害中の不満増分(不透明 magnitude。1日1回。0.0=観測のみ)
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    days = [int(d) for d in (raw.get("days", []) or [])]
    return {
        "enabled": bool(raw.get("enabled", False)),
        # ---- 災害ショック ----
        "days": days,
        "onset_prob": float(raw.get("onset_prob", 0.0)),
        "duration_days": int(raw.get("duration_days", 1)),
        "kind": str(raw.get("kind", "台風")),
        "grievance": float(raw.get("grievance", 0.0)),
        "stay_home_bias": float(raw.get("stay_home_bias", 0.0)),
        "suspend_transit": bool(raw.get("suspend_transit", True)),
        # ---- 交通の遅延・遅刻・運休 ----
        "delay_prob": float(raw.get("delay_prob", 0.0)),
        "suspend_prob": float(raw.get("suspend_prob", 0.0)),
        "delay_grievance": float(raw.get("delay_grievance", 0.0)),
        # ---- インフラ障害 ----
        "outage_prob": float(raw.get("outage_prob", 0.0)),
        "outage_kind": str(raw.get("outage_kind", "停電")),
        "outage_duration_days": int(raw.get("outage_duration_days", 1)),
        "outage_grievance": float(raw.get("outage_grievance", 0.0)),
    }


def enabled(sim) -> bool:
    """都市・環境ショック(H4)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "disastercfg", None)
    return bool(cfg and cfg["enabled"])


def infra_out(sim) -> bool:
    """インフラ障害(停電・通信断)が発生中か(media/phone の在宅抑制ゲート用)。

    disaster OFF では常に False(enabled ゲートで media/phone がバイト一致)。ON でも障害が
    起きていなければ False。"""
    return enabled(sim) and bool(getattr(sim, "_infra_active", False))


def is_homebound(agent) -> bool:
    """災害で在宅(外出抑制)中か(routine の在宅ゲート用)。既定 OFF では常に False。"""
    return bool(getattr(agent, "_disaster_homebound", False))


def _log(sim, step: int, sim_min: int, kind: str, payload: dict) -> None:
    """世界レベルのショック記録(agent_id=-1。disaster/transit_delay/infra_outage)。"""
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1, kind=kind,
                         x=0.0, y=0.0, payload=payload))


# ---------------------------------------------------------------- 災害ショック(日次)
def _should_onset(cfg: dict, day: int, rng) -> bool:
    """今日 災害が新たに発生するか。発生日指定(days)は決定論(乱数なし)、非該当日は onset_prob。"""
    if day in cfg["days"]:
        return True
    p = float(cfg["onset_prob"])
    return p > 0.0 and bool(rng.random() < p)


def _update_disaster(sim, cfg: dict, day: int, step: int, sim_min: int, rng) -> bool:
    """災害の発生/継続/解除を日次で更新し、当日アクティブかを返す(副作用=状態・イベント・grievance)。

    継続中は grievance(1日1回・街内)+ 住民の在宅抽選(新 stream "disaster")+ プロンプト文脈を更新する。
    住民の在宅は agent._disaster_homebound(routine が読む)に設定=routine は新規 rng を引かない(既定 OFF は
    getattr 既定 False で不変)。返り値=当日アクティブ(交通麻痺=運休の要否判定に使う)。"""
    active = bool(getattr(sim, "_disaster_active", False))
    if active and day >= int(getattr(sim, "_disaster_until_day", -1)):
        _log(sim, step, sim_min, "disaster",
             {"kind": getattr(sim, "_disaster_kind", None), "phase": "clear"})
        sim._disaster_active = False
        sim._disaster_kind = None
        active = False
    if not active and _should_onset(cfg, day, rng):
        sim._disaster_active = True
        sim._disaster_kind = cfg["kind"]
        sim._disaster_until_day = day + max(1, int(cfg["duration_days"]))
        _log(sim, step, sim_min, "disaster", {"kind": cfg["kind"], "phase": "onset"})
        active = True
    if active:
        kind = getattr(sim, "_disaster_kind", cfg["kind"])
        sim.today_disaster_line = f"今日は{kind}で交通が麻痺し、外出が難しい状況です。"
        mag = float(cfg["grievance"])
        bias = float(cfg["stay_home_bias"])
        for agent in sim.agents:                       # id 昇順=決定論
            if not agent.visitor:                      # 在宅抑制は住民のみ(来街者は街の外に生活基盤)
                r = sim.hub.stream("disaster", agent.id, step).random()
                agent._disaster_homebound = bool(r < bias)
            if agent.loc != "outside":                 # grievance は街の外(不在)には効かせない
                factor_update.on_disaster(agent, mag, cause="disaster", step=step,
                                          sim_min=sim_min, logger=sim.logger)
    else:
        sim.today_disaster_line = None
        for agent in sim.agents:
            agent._disaster_homebound = False
    return active


# ---------------------------------------------------------------- 交通の遅延・運休(日次)
def _roll_transit_delay(sim, cfg: dict, step: int, sim_min: int, rng) -> bool:
    """日次に電車の遅延/運休を注入する。返り値=今日運休(has_service 停止)か。

    delay_prob で遅延日を決め、うち suspend_prob で運休(残りは遅延=grievance のみで運行は継続)。
    遅延/運休に遭遇した街内の人へ grievance(factors 経由・不透明 magnitude)。既存の定刻ダイヤの
    draw は一切触らず、新 stream "disaster" でのみ乱す(決定論・ゴールデン保護)。"""
    p = float(cfg["delay_prob"])
    if p <= 0.0 or rng.random() >= p:
        return False
    suspend = float(cfg["suspend_prob"]) > 0.0 and rng.random() < float(cfg["suspend_prob"])
    _log(sim, step, sim_min, "transit_delay",
         {"kind": "運休" if suspend else "遅延", "line": None})
    mag = float(cfg["delay_grievance"])
    if mag != 0.0:
        for agent in sim.agents:
            if agent.loc != "outside":
                factor_update.on_disaster(agent, mag, cause="transit_delay", step=step,
                                          sim_min=sim_min, logger=sim.logger)
    return suspend


# ---------------------------------------------------------------- インフラ障害(日次)
def _update_infra(sim, cfg: dict, day: int, step: int, sim_min: int, rng) -> None:
    """インフラ障害(停電・通信断・断水)の発生/継続/解除を日次で更新する(軽実装)。

    継続中は grievance(1日1回・街内)。障害中フラグ(sim._infra_active)は media/phone の在宅抑制が読む。
    発生/解除を infra_outage(agent_id=-1, phase=onset/clear)で記録する。"""
    active = bool(getattr(sim, "_infra_active", False))
    if active and day >= int(getattr(sim, "_infra_until_day", -1)):
        _log(sim, step, sim_min, "infra_outage",
             {"kind": getattr(sim, "_infra_kind", None), "phase": "clear"})
        sim._infra_active = False
        sim._infra_kind = None
        active = False
    p = float(cfg["outage_prob"])
    if not active and p > 0.0 and rng.random() < p:
        sim._infra_active = True
        sim._infra_kind = cfg["outage_kind"]
        sim._infra_until_day = day + max(1, int(cfg["outage_duration_days"]))
        _log(sim, step, sim_min, "infra_outage",
             {"kind": cfg["outage_kind"], "phase": "onset"})
        active = True
    if active:
        mag = float(cfg["outage_grievance"])
        if mag != 0.0:
            for agent in sim.agents:
                if agent.loc != "outside":
                    factor_update.on_disaster(agent, mag, cause="infra_outage", step=step,
                                              sim_min=sim_min, logger=sim.logger)


# ---------------------------------------------------------------- 日次フェーズ(1日1回)
def tick_day(sim, step: int, sim_min: int) -> None:
    """都市・環境ショックの日境界更新(災害→交通遅延→インフラ障害)。既定 OFF=完全 no-op(バイト一致)。

    確率判定はすべて世界レベルの新 stream "disaster"(agent_id=-1)から固定順に引く(決定論)。1日の
    交通麻痺(運休)= 災害の suspend_transit または 交通遅延の運休 のどちらか。transit.suspended を
    毎日この時点で更新する(disaster OFF では触れない=has_service バイト一致)。

    ★因果の切り分け(IF-F W2): **自然事象は災害そのものだけ**である。運休・遅延・障害の告知は
    それを見た**事業者(装置)の判断** = DEVS の δ_ext であって自然の続きではない。混ぜたままだと
    台帳上「運休は自然が起こした」ことになり、**止めると決めた主体が世界から消える**。そこで
      - 災害の発生/継続/解除(disaster)  … 装置スコープを開かない = natural のまま
      - 遅延/運休(transit_delay)        … operator:transit の窓の中で出す
      - インフラ障害(infra_outage)      … operator:infra の窓の中で出す
      - 運休フラグの書き込み              … 運行事業者装置の δ_ext を**素通し**で通す
    とする。**計算は 1 演算も変えていない**(下の suspend 式は従来と同一)ので、causality も
    装置層も、どちらの ON/OFF でも運休する日は完全に同じである(テストで機械固定)。"""
    if not enabled(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_disaster_day", -1):
        return
    sim._disaster_day = day
    cfg = sim.disastercfg
    rng = sim.hub.stream("disaster", "world", step)     # 世界レベルの日次判定(固定順=決定論)
    disaster_active = _update_disaster(sim, cfg, day, step, sim_min, rng)
    with devices_mod.cause_scope(sim, devices_mod.DEV_OPERATOR_TRANSIT):
        delay_suspend = _roll_transit_delay(sim, cfg, step, sim_min, rng)
    with devices_mod.cause_scope(sim, devices_mod.DEV_OPERATOR_INFRA):
        _update_infra(sim, cfg, day, step, sim_min, rng)
    # 交通麻痺(運休): 災害中(suspend_transit)または 交通遅延の運休 の日は電車を止める。
    suspend = bool((disaster_active and cfg["suspend_transit"]) or delay_suspend)
    operator = devices_mod.transit_operator(sim)        # 既定 OFF = None(下 2 行は通らない)
    if operator is not None:
        suspend = bool(operator.on_input(-1, {
            "day": day, "disaster_active": disaster_active,
            "suspend_transit": bool(cfg["suspend_transit"]),
            "delay_suspend": delay_suspend}, sim_min)["suspended"])
    sim.transit.suspended = suspend
