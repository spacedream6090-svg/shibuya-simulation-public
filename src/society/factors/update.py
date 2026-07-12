"""state 更新則レジストリ(OPEN#2 の因果構造実装、生態系フェーズ Phase D)。

原則(design §12 / risk-register):
- **因果構造のみ文献接地、magnitude は conf の factors ブロックで可変**(B段で調律。
  決め打ち禁止)。
- **R9**: 更新関数はイベント(出来事)だけを受け取り、traits を一切見ない。
- **R7**: 恒常性は注入しない(飽和は affordance 制約に委ねる)。clip [0,1] のみ。
- 全変更は state_update イベントに記録(cause=ルール名)→ 事後の因果分解が可能。

ルールと文献接地(lit/state-update__open2-overview, factors__personality-motivation):
  congestion       混雑遅延 → grievance+          (Epstein 2002 hardship)
  heard_valence    ネガ/ポジ発話・投稿・報道を聞く → grievance±(SIMCA: affective が強い)
  being_heard      発話に聞き手がいた → efficacy+  (Bandura: 社会的説得・被承認)
  ignored          誰にも聞かれない発話 → efficacy− 微(説得の失敗)
  work_done        勤務を完遂 → efficacy+          (Bandura: 達成経験=最強の源泉)
  vicarious        他者の言葉・成功を目撃 → efficacy+ 小(Bandura: 代理経験)
  park             公園・緑地に滞在 → grievance−   (環境心理: 回復環境)
  own_adopted      自分の造語を他者が採用 → ownership+ efficacy+(影響の実感=当事者化)
  money_pressure   手持ちが逼迫している間 → grievance+(経済的困難=hardship。1日1回)
  media            在宅の娯楽メディア視聴 → grievance−(気分管理 Zillmann / ゲーム気分改善
                   Vuorre 2024。既定 magnitude 0.0=OFF 時完全不変。バッチD)
"""
from __future__ import annotations

from . import affect
from ..observer.logger import ObserverLogger
from ..observer.schema import Event

# magnitude の既定値(conf の factors: で上書き可能。B段調律対象)
DEFAULT_MAGNITUDES = {
    "congestion_grievance": 0.010,
    "heard_valence_grievance": 0.020,   # ×|valence|(ネガで+、ポジで−×0.5)
    "heard_valence_efficacy": 0.006,    # ポジ発話を聞く→わずかに前向き
    "being_heard_efficacy": 0.006,
    "ignored_efficacy": -0.004,
    "work_done_efficacy": 0.012,
    "vicarious_efficacy": 0.004,
    "park_grievance": -0.008,
    "own_adopted_ownership": 0.030,
    "own_adopted_efficacy": 0.010,
    "money_pressure_grievance": 0.015,
    # 娯楽メディア(バッチD)。既定 0.0 = seam のみ実装で OFF 時完全不変。
    # 推奨 ON 値 -0.03(Vuorre 2024 のゲーム気分改善 +0.034/15分 を逆符号で写す)。
    "media_grievance": 0.0,
    # 相対的剥奪(Wave G1 2026-07-07)。参照集団(現在この街に居る他者)の所持金中央値を
    # 下回る量に応じた不満。engine が正規化差 d∈(0,1] を掛けた mag を渡すのでこの係数が上限。
    # 個体の相対的地位で grievance に個体差が復活=efficacy/grievance 飽和(retrospective §2 懸念)
    # を破るのが狙い。既定 0.0 = OFF(バイト一致)。★ON 推奨: 例 0.012。
    "relative_deprivation_grievance": 0.0,
}


def build_magnitudes(raw: dict | None) -> dict:
    mags = dict(DEFAULT_MAGNITUDES)
    if raw:
        mags.update({k: float(v) for k, v in dict(raw).items()
                     if k in DEFAULT_MAGNITUDES})
    return mags


def _bump(agent, name: str, delta: float, cause: str, *, step: int,
          sim_min: int, logger: ObserverLogger) -> float:
    """state を delta だけ動かし clip・記録。戻り値=実際の変化量。"""
    if delta == 0.0 or name not in agent.states:
        return 0.0
    old = agent.states[name]
    new = min(1.0, max(0.0, old + delta))
    if new == old:
        return 0.0
    agent.states[name] = new
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="state_update", x=agent.x, y=agent.y,
                     payload={"name": name, "old": round(old, 4),
                              "new": round(new, 4), "cause": cause}))
    _impact_note(agent, name, new - old, step=step, sim_min=sim_min, logger=logger)
    return new - old


def _impact_note(agent, name: str, delta: float, *, step: int, sim_min: int,
                 logger: ObserverLogger) -> None:
    """日内衝撃ゲージ(第12バッチ: 出来事誘発の深い内省)。既定 OFF=即 return=バイト一致。

    全 state 変化の単一通過点(_bump)から呼ぶ: |Δstate| は「信念・期待との乖離」の
    k 非依存な近似(docs/research/deep-reflection-triggers.md §3。強度でなく自己関連の
    乖離が誘因、の作業仮説的操作化)。ネガ(grievance↑/efficacy↓)は非対称に重く、
    ポジ(逆向き)もゼロにしない(畏敬・転機の反例)。個人閾値(reflect_p)超えで
    深い内省を incubation_days 後に予約(侵入的段階=「頭から離れない」記憶を残す)。
    cooldown 中は再予約しない(反芻の害の抑制)。"""
    track = bool(getattr(agent, "_impact_on", False))
    p = getattr(agent, "reflect_p", None)
    if not track and p is None:
        return
    if name == "grievance":
        neg, pos = (delta, 0.0) if delta > 0 else (0.0, -delta)
    elif name == "efficacy":
        neg, pos = (-delta, 0.0) if delta < 0 else (0.0, delta)
    else:
        return
    agent.impact_neg_today += neg
    agent.impact_pos_today += pos
    if p is None:
        return
    agent.impact_today += p["neg_weight"] * neg + p["pos_weight"] * pos
    day = sim_min // 1440
    if (agent.impact_today >= p["threshold"] and agent.deep_due_day < 0
            and day >= agent.deep_cooldown_until_day):
        agent.deep_due_day = day + int(p["incubation_days"])
        logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="reflection_trigger", x=agent.x, y=agent.y,
                         payload={"gauge": round(agent.impact_today, 4),
                                  "day": day, "due_day": agent.deep_due_day}))
        agent.remember("今日の出来事が頭から離れない", importance_bonus=0.5)


# ---------------------------------------------------------------- ルール群
def on_congestion(agent, factor: float, *, mags: dict, step: int, sim_min: int,
                  logger: ObserverLogger) -> float:
    if factor >= 1.0:
        return 0.0
    return _bump(agent, "grievance",
                 mags["congestion_grievance"] * (1.0 - factor),
                 "congestion", step=step, sim_min=sim_min, logger=logger)


def on_heard_valence(agent, v: float, *, mags: dict, step: int, sim_min: int,
                     logger: ObserverLogger, scale: float = 1.0) -> float:
    """聞いた/読んだ言葉の感情価。ネガ→grievance+、ポジ→grievance−(半分)+efficacy+。"""
    if v == 0.0:
        return 0.0
    total = 0.0
    if v < 0:
        total += _bump(agent, "grievance",
                       mags["heard_valence_grievance"] * (-v) * scale,
                       "heard_negative", step=step, sim_min=sim_min, logger=logger)
    else:
        total += _bump(agent, "grievance",
                       -0.5 * mags["heard_valence_grievance"] * v * scale,
                       "heard_positive", step=step, sim_min=sim_min, logger=logger)
        total += _bump(agent, "efficacy",
                       mags["heard_valence_efficacy"] * v * scale,
                       "heard_positive", step=step, sim_min=sim_min, logger=logger)
    return total


def on_arousal(agent, cfg: dict, *, valence_abs: float = 0.0, novelty: float = 0.0,
               addressed: float = 0.0, congestion: float = 0.0, cause: str = "",
               step: int, sim_min: int, logger: ObserverLogger) -> float:
    """観測イベントで覚醒度 arousal を動かす hook(感情・興味・注意ハブ。on_heard_valence の隣)。

    非LLM・決定論・RNG不要。入力は**観測量のみ**(|valence|・novelty・addressed・congestion)=
    R1(k 非依存: belief 書き戻し・k.writeback を一切見ない)。arousal 更新則(漏れ積分器)は
    factors/affect が持ち、ここは agent.arousal を動かして affect_update を記録する薄い hook。

    既定 gain=0(または enabled=false)なら **no-op**: arousal を触らず・ログも出さない=バイト一致。
    affect_update は enabled かつ gain>0 のときだけ出る(既定は一切出さない)。arousal は drive の
    state_change reason にフィードバックしない(自己増幅ループ防止、§6.3)。
    """
    if not cfg.get("enabled") or cfg["arousal_gain"] == 0.0:
        return 0.0
    signal = affect.arousal_signal(valence_abs=valence_abs, novelty=novelty,
                                   addressed=addressed, congestion=congestion)
    if signal <= 0.0:
        return 0.0
    old = getattr(agent, "arousal", cfg["arousal_baseline"])
    new = affect.bump(old, signal, cfg)
    if new == old:
        return 0.0
    agent.arousal = new
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="affect_update", x=agent.x, y=agent.y,
                     payload={"old": round(old, 4), "new": round(new, 4),
                              "cause": cause}))
    return new - old


def on_spoke(agent, n_hearers: int, *, mags: dict, step: int, sim_min: int,
             logger: ObserverLogger) -> float:
    if n_hearers > 0:
        return _bump(agent, "efficacy", mags["being_heard_efficacy"],
                     "being_heard", step=step, sim_min=sim_min, logger=logger)
    return _bump(agent, "efficacy", mags["ignored_efficacy"],
                 "ignored", step=step, sim_min=sim_min, logger=logger)


def on_work_done(agent, *, mags: dict, step: int, sim_min: int,
                 logger: ObserverLogger) -> float:
    return _bump(agent, "efficacy", mags["work_done_efficacy"],
                 "work_done", step=step, sim_min=sim_min, logger=logger)


def on_vicarious(agent, *, mags: dict, step: int, sim_min: int,
                 logger: ObserverLogger) -> float:
    return _bump(agent, "efficacy", mags["vicarious_efficacy"],
                 "vicarious", step=step, sim_min=sim_min, logger=logger)


def on_park(agent, *, mags: dict, step: int, sim_min: int,
            logger: ObserverLogger) -> float:
    return _bump(agent, "grievance", mags["park_grievance"],
                 "park", step=step, sim_min=sim_min, logger=logger)


def on_own_adopted(agent, *, mags: dict, step: int, sim_min: int,
                   logger: ObserverLogger) -> float:
    total = _bump(agent, "ownership", mags["own_adopted_ownership"],
                  "own_adopted", step=step, sim_min=sim_min, logger=logger)
    total += _bump(agent, "efficacy", mags["own_adopted_efficacy"],
                   "own_adopted", step=step, sim_min=sim_min, logger=logger)
    return total


def on_money_pressure(agent, *, mags: dict, step: int, sim_min: int,
                      logger: ObserverLogger) -> float:
    """手持ちが逼迫している間の不満。呼び出し側が「残高<閾値」の判定と頻度(1日1回)を
    担う。この関数はイベント(逼迫している事実)だけを受け取り traits を見ない(R9)。"""
    return _bump(agent, "grievance", mags["money_pressure_grievance"],
                 "money_pressure", step=step, sim_min=sim_min, logger=logger)


def relative_deprivation_coef(mags: dict) -> float:
    """相対的剥奪の係数(Wave G1)。engine が「中央値算出の要否」を判定するための不透明な float。

    engine(scheduler)が因子名 grievance を書かずに coef>0 を判定できるよう、factors が
    キー名を隠して係数だけ返す(no-fingerprint)。既定 0.0=OFF。"""
    return float(mags.get("relative_deprivation_grievance", 0.0))


def on_relative_deprivation(agent, magnitude: float, *, step: int, sim_min: int,
                            logger: ObserverLogger) -> float:
    """参照集団(他者)の所持金中央値を下回る相対的剥奪 → 不満(grievance+)。1日1回(Wave G1)。

    R9: 呼び出し側(scheduler)は参照中央値と自分の所持金の差から**不透明な magnitude**
    (係数 × 正規化差)だけを渡し、この関数は traits も参照集団の中身も見ない(on_money_pressure
    と同型)。magnitude=0.0(係数 relative_deprivation_grievance=0.0=OFF、または中央値以上)なら
    _bump が state を触らず記録もしない=バイト一致。係数は factors config で調律する。

    相対的剥奪(relative deprivation, Runciman/Gurr): 絶対的窮乏でなく他者比較の格差が不満の源。
    個体の相対的地位で grievance に個体差が復活し、飽和(retrospective §2 懸念)を破る。"""
    return _bump(agent, "grievance", magnitude, "relative_deprivation",
                 step=step, sim_min=sim_min, logger=logger)


def on_evicted(agent, magnitude: float, *, step: int, sim_min: int,
               logger: ObserverLogger) -> float:
    """家賃滞納による立退き(強制退去)の生活不安(制度深化3 2026-07-08)。

    R9: 呼び出し側(scheduler の口座 E5 日次境界)は滞納の中身を渡さず**不透明な
    magnitude**(economy.accounts.eviction_grievance)だけを渡す(on_weather と同型)。
    magnitude=0.0 なら _bump が state を触らず記録もしない=バイト一致。
    住居喪失は最も強いストレス因の一つ(住宅不安定性→精神的健康の悪化)。"""
    return _bump(agent, "grievance", magnitude, "eviction",
                 step=step, sim_min=sim_min, logger=logger)


def on_bankrupt(agent, magnitude: float, *, step: int, sim_min: int,
                logger: ObserverLogger) -> float:
    """自己破産(免責+資産圧縮+活動制限)の生活不安(制度深化3 2026-07-08)。

    R9: on_evicted と同型(不透明 magnitude=economy.accounts.bankruptcy_grievance)。
    免責で借金は消えるが、社会的スティグマと再出発の重さを心理側で表す。"""
    return _bump(agent, "grievance", magnitude, "bankruptcy",
                 step=step, sim_min=sim_min, logger=logger)


def on_weather(agent, magnitude: float, *, step: int, sim_min: int,
               logger: ObserverLogger) -> float:
    """悪天候(雨・雪)の日の不快感(第7バッチ 2026-07-07)。1日1回・全員共通の暦的文脈。

    呼び出し側(scheduler)は「その日の天気の不快度」= 不透明な magnitude だけを渡し、
    この関数は traits を見ない(R9)。magnitude=0.0(既定 rain_grievance=0.0)なら _bump が
    state を触らず記録もしない=天気 ON でも grievance は不変(天気イベント・文脈行のみ)。
    係数(rain_grievance)は weather.py が保持する(conf.weather.rain_grievance で調律)。
    """
    return _bump(agent, "grievance", magnitude, "weather",
                 step=step, sim_min=sim_min, logger=logger)


def on_enforcement(agent, magnitude: float, *, step: int, sim_min: int,
                   logger: ObserverLogger) -> float:
    """条例・制度DSL の prohibit ルール執行(公務員=警察官)を受けた違反者の不満(grievance+)。Wave G3。

    執行ルート(現実の第3ルート): 制度が実効ルールとして違反者に罰を科す。R9: 呼び出し側
    (government/scheduler の執行フェーズ)は罰則の重み=**不透明な magnitude** だけを渡し、この関数は
    traits も rule も違反内容も見ない(on_weather/on_relative_deprivation と同型)。magnitude=0.0
    (係数 enforcement.grievance=0.0)なら _bump が state を触らず記録もしない=バイト一致。"""
    return _bump(agent, "grievance", magnitude, "enforcement",
                 step=step, sim_min=sim_min, logger=logger)


def on_job_loss(agent, magnitude: float, *, step: int, sim_min: int,
                logger: ObserverLogger) -> float:
    """失業(職の喪失)による生活基盤の不安 → 不満(grievance+)。Wave G5(キャリア転換)。

    キャリア転換ルート(現実ギャップ): 職を失うと収入が絶たれ生活が不安定化する。R9: 呼び出し側
    (scheduler の career 日次フェーズ)は生活不安の重み=**不透明な magnitude** だけを渡し、この関数は
    traits も職業も org も見ない(on_weather/on_enforcement と同型)。magnitude=0.0(係数
    career.unemployment_grievance=0.0)なら _bump が state を触らず記録もしない=バイト一致。係数は
    career config が保持する(この関数は不透明な量だけ受け取る=no-fingerprint)。"""
    return _bump(agent, "grievance", magnitude, "job_loss",
                 step=step, sim_min=sim_min, logger=logger)


def on_misinfo(agent, magnitude: float, *, step: int, sim_min: int,
               logger: ObserverLogger) -> float:
    """誤情報投稿への炎上(ネガ反応の殺到)を受けた発信者の不満(grievance+)。Wave G6(情報環境の非対称)。

    情報環境の非対称ルート: フェイク/誤情報が炎上(flame)すると発信者に社会的圧が集中する。R9: 呼び出し側
    (net/infoenv の炎上処理)は炎上の重み=**不透明な magnitude** だけを渡し、この関数は traits も投稿内容も
    見ない(on_enforcement/on_job_loss と同型)。magnitude=0.0(係数 info_env.misinfo.flame_grievance=0.0)なら
    _bump が state を触らず記録もしない=バイト一致。係数は info_env config が保持する。発火系(drive)には
    接続しない(呼び出し側は返り値で drive.add しない)=R1: 炎上は generate 呼数を1本も動かさない。"""
    return _bump(agent, "grievance", magnitude, "misinfo_flame",
                 step=step, sim_min=sim_min, logger=logger)


def on_scarcity(agent, magnitude: float, *, step: int, sim_min: int,
                logger: ObserverLogger) -> float:
    """品切れ・行列(需要集中の資源希少化)に遭遇した消費者の不満(grievance+)。後続波 H3(商業ダイナミクス)。

    商業ルート(現実ギャップ): 需要が集中した店で在庫が尽きる/行列ができると消費者に不快が生じる。R9:
    呼び出し側(commerce)は希少化の重み=**不透明な magnitude** だけを渡し、この関数は traits も店も
    在館数も見ない(on_enforcement/on_job_loss/on_misinfo と同型)。magnitude=0.0(係数
    commerce.stock_grievance=0.0)なら _bump が state を触らず記録もしない=バイト一致。係数は commerce
    config が保持する(この関数は不透明な量だけ受け取る=no-fingerprint)。発火系(drive)には接続しない
    (呼び出し側は返り値で drive.add しない)=R1: 品切れは generate 呼数を1本も動かさない。"""
    return _bump(agent, "grievance", magnitude, "scarcity",
                 step=step, sim_min=sim_min, logger=logger)


def on_disaster(agent, magnitude: float, *, cause: str = "disaster", step: int,
                sim_min: int, logger: ObserverLogger) -> float:
    """都市・環境ショック(災害・交通運休/遅延・インフラ障害)を受けた人の不満(grievance+)。後続波 H4。

    生活を麻痺させる外生ショック=集団的 grievance の源。R9: 呼び出し側(src/society/disaster.py)は
    ショックの重み=**不透明な magnitude** だけを渡し、この関数は traits も災害種別も物理位置も見ない
    (on_weather/on_enforcement/on_scarcity と同型)。cause はショックの別(disaster/transit_delay/
    infra_outage)を state_update に刻むラベルにすぎない(構成概念ではない)。magnitude=0.0(各係数=0.0)
    なら _bump が state を触らず記録もしない=バイト一致(ショックのイベントは出るが grievance は不変)。
    係数は disaster config が保持する(この関数は不透明な量だけ受け取る=no-fingerprint)。発火系(drive)には
    接続しない(呼び出し側は返り値で drive.add しない)=R1: ショックは generate 呼数を1本も動かさない。"""
    return _bump(agent, "grievance", magnitude, cause,
                 step=step, sim_min=sim_min, logger=logger)


def on_crime(agent, magnitude: float, *, cause: str = "crime", step: int,
             sim_min: int, logger: ObserverLogger) -> float:
    """犯罪・迷惑行為の被害(窃盗の被害者 / 迷惑行為の周囲の人)→ 不満(grievance+)。後続波 H5。

    街の負の相互作用=個体の被害体験に根ざす grievance の源。R9: 呼び出し側(src/society/diversity.py)は
    被害の重み=**不透明な magnitude** だけを渡し、この関数は traits も犯罪種別も物理位置も見ない
    (on_disaster/on_scarcity と同型)。cause は被害の別(crime=窃盗被害者 / nuisance=迷惑の周囲)を
    state_update に刻むラベルにすぎない(構成概念ではない)。窃盗の money− は diversity 側で扱う客観量
    (現金)で、この hook は grievance だけを不透明に受け取る。magnitude=0.0(各係数=0.0)なら _bump が
    state を触らず記録もしない=バイト一致(crime/nuisance のイベントは出るが grievance は不変)。係数は
    diversity config が保持する。発火系(drive)には接続しない(呼び出し側は返り値で drive.add しない)
    =R1: 犯罪被害は generate 呼数を1本も動かさない。"""
    return _bump(agent, "grievance", magnitude, cause,
                 step=step, sim_min=sim_min, logger=logger)


def on_media(agent, *, mags: dict, step: int, sim_min: int,
             logger: ObserverLogger) -> float:
    """在宅の娯楽メディア(TV/動画/ゲーム)セッション終了時の気分修復(バッチD)。

    気分管理理論(Zillmann): 娯楽視聴はネガ気分を和らげる。実証アンカーは Vuorre 2024
    のゲーム気分改善(+0.034 / 最初の15分で飽和)。ここは park_grievance を前例に
    grievance を小さく下げる seam。呼び出し側(routine)は「視聴した事実」だけを渡し、
    この関数は traits を見ない(R9)。magnitude 既定 0.0 = OFF 時完全不変(delta 0 なら
    _bump が state を触らず記録もしない)。conf.factors.media_grievance で調律する。"""
    return _bump(agent, "grievance", mags["media_grievance"], "media",
                 step=step, sim_min=sim_min, logger=logger)


# ---------------------------------------------------------------- 集団効力感(SIMCA/Bandura。psych.collective プラグイン、既定 OFF)
def on_group_success(agent, magnitude: float, *, step: int, sim_min: int,
                     logger: ObserverLogger) -> float:
    """所属グループ由来の成功(提案成立・グループイベントへの十分な参加)→ 集団効力感+。

    R9: 呼び出し側(tools)は「成功が起きた事実」と magnitude(不透明な量)だけを渡す。
    collective_efficacy state を持たない agent(=プラグイン OFF)には _bump が no-op を返す
    ので二重に安全。因子名を知ってよいのは factors だけ。
    """
    return _bump(agent, "collective_efficacy", magnitude, "group_success",
                 step=step, sim_min=sim_min, logger=logger)


def on_ingroup_heard(agent, v: float, boost: float, *, mags: dict, step: int,
                     sim_min: int, logger: ObserverLogger) -> float:
    """同じグループのメンバーの発話を聞く(社会的アイデンティティ)= heard_valence を boost 倍。

    boost はエンジンが渡す**不透明な倍率**(1.0=通常/域外、>1=in-group。in-group 判定は
    engine 側のグループ照合で決まり、この関数は因子名も判定条件も持たない)。boost=1.0 なら
    on_heard_valence と完全同一の効果=OFF 時のバイト一致を保証する。
    """
    return on_heard_valence(agent, v, mags=mags, step=step, sim_min=sim_min,
                            logger=logger, scale=boost)
