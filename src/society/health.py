"""健康・疲労・病気・メンタル(現実ギャップ 後続波 H1 2026-07-07。ユーザー要望)。

生活の身体的基盤を最小・非LLM・決定論で載せる層。3機構:
  1. 疲労ゲージ(fatigue): 起きて活動(勤務/移動)している間に蓄積、睡眠で回復。高疲労で
     発火 effective_threshold を上げる(休息へ寄る)。drive ゲージ / arousal と並列の内部
     transient(states 監査集合=efficacy/grievance/ownership には入れない=R²(k) を汚さない)。
  2. 病気(illness): 日次に新 stream "health" で確率発症 → 欠勤(勤務スキップ)・外出抑制
     (在宅)・受診で医療消費(medical_visit=spend)。数日で回復(illness state=onset/recover)。
  3. メンタル(mental): 慢性的な高 grievance の持続 → 引きこもり(withdrawn = 自由時間の
     外出を控え社会的活動が低下)。既存 grievance を参照する(この module=src/society 直下=
     engine/cognition/world の静的検査対象外なので構成概念名をここに閉じてよい=no-fingerprint 契約に触れない)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。

R1 呼数不変: どの機構も generate() を1本も足さない。疲労は決定論(RNG不要)、病気の発症/回復は
  新 stream "health"(既存 draw 順に挿入しない)、メンタルは grievance の読み取りのみ。病気(在宅)・
  疲労(休息)は物理位置を変え対面 co-location が変化しうる(=FixedLLM で ON!=OFF になりうる=career
  G5 / crowd G4 と同型)が、機構は k・内面状態(構成概念)を発火判断に食わせず暦・config・新 stream・
  物理位置・grievance(k 非依存の state)のみ参照する=compute_matched 下の k 不変性で呼数一致を担保する。

既定 OFF(enabled=false)= 疲労更新なし・病気/受診なし・引きこもりなし・health stream も引かない・
  イベント 0 件・乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。既定 ON でも各 gain/prob が
  0(=恒等)なら実質 no-op(fatigue_gain=0 で疲労不変・onset_prob=0 で発症なし)。
"""
from __future__ import annotations

from .observer.schema import Event


def build_cfg(raw) -> dict:
    """conf の health ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(career/annual と同型)。すべて既定 OFF(enabled=false)で
    scheduler の health 日次/毎step フェーズと routine の在宅/欠勤/引きこもりが完全 no-op。

    パラメータ(ON 時のみ効く):
      疲労: fatigue_gain_work/move  活動1stepあたりの疲労蓄積(既定 0=恒等)
            fatigue_recovery         睡眠1stepあたりの回復量
            fatigue_high             「高疲労」とみなす閾値(健康 update / 行動鈍化の境界)
            fatigue_threshold_gain   高疲労→発火 effective_threshold の上げ幅(既定 0=恒等)
      病気: onset_prob               日次の発症確率(新 stream "health"。既定 0=発症なし)
            illness_days             発症してから回復までの日数
            medical_prob             発症直後に受診する確率(→ medical_visit + spend)
            medical_cost             受診1回の医療費(spend cat="medical")
      メンタル: mental_grievance_threshold  「高 grievance」とみなす閾値
                mental_withdraw_days         高 grievance がこの日数続いたら引きこもり(0=メンタル無効)
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        # ---- 疲労(内部 transient。既定 gain=0=恒等)----
        "fatigue_gain_work": float(raw.get("fatigue_gain_work", 0.0)),
        "fatigue_gain_move": float(raw.get("fatigue_gain_move", 0.0)),
        "fatigue_recovery": float(raw.get("fatigue_recovery", 0.15)),
        "fatigue_high": float(raw.get("fatigue_high", 0.6)),
        "fatigue_threshold_gain": float(raw.get("fatigue_threshold_gain", 0.0)),
        # ---- 病気(新 stream "health"。既定 onset_prob=0=発症なし)----
        "onset_prob": float(raw.get("onset_prob", 0.0)),
        "illness_days": int(raw.get("illness_days", 3)),
        "medical_prob": float(raw.get("medical_prob", 0.0)),
        "medical_cost": float(raw.get("medical_cost", 3000.0)),
        # ---- メンタル(慢性高 grievance→引きこもり。既定 need>0 だが閾値高で通常不発)----
        "mental_grievance_threshold": float(raw.get("mental_grievance_threshold", 0.7)),
        "mental_withdraw_days": int(raw.get("mental_withdraw_days", 3)),
    }


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------- 疲労ゲージ
def tick_fatigue(agent, cfg: dict, step: int, sim_min: int, logger) -> None:
    """疲労ゲージの毎step更新(内部 transient。RNG不要・決定論)。

    起きて活動(勤務=activity working / 移動=route or commuting)している間に fatigue が
    蓄積、睡眠(sleeping)で回復。範囲外(loc=outside)は不在=変化なし。arousal.decay と同型の
    静かな transient 更新で、高疲労境界(fatigue_high)を跨いだ step だけ health_update を1件
    記録する(sparse・「効いた時」の観測)。gain 既定 0 のときは fatigue 不変=ログも出ない。
    """
    if getattr(agent, "loc", "street") == "outside":
        return
    old = float(getattr(agent, "fatigue", 0.0))
    if getattr(agent, "sleeping", False):
        new = old - float(cfg["fatigue_recovery"])
    else:
        activity = getattr(agent, "activity", "")
        if activity == "working":
            gain = float(cfg["fatigue_gain_work"])
        elif activity == "commuting" or agent.route:
            gain = float(cfg["fatigue_gain_move"])
        else:
            gain = 0.0
        new = old + gain
    new = _clip01(new)
    if new == old:
        return
    agent.fatigue = new
    hi = float(cfg["fatigue_high"])
    if (old >= hi) != (new >= hi):                 # 高疲労境界を跨いだときだけ記録
        logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="health_update", x=agent.x, y=agent.y,
                         payload={"name": "fatigue", "old": round(old, 4),
                                  "new": round(new, 4),
                                  "cause": "tired" if new >= hi else "rested"}))


def fatigue_threshold_delta(agent, cfg: dict) -> float:
    """高疲労 → 発火 effective_threshold への上げ幅(不透明な delta。休息へ寄る)。

    drive.effective_threshold 側の clip[0.30, 0.85] に合流する(affect.threshold_delta と同型)。
    gain=0(または fatigue が閾値以下)なら 0(恒等=バイト一致)。engine には float だけ渡す。
    """
    g = float(cfg["fatigue_threshold_gain"])
    if g == 0.0:
        return 0.0
    f = float(getattr(agent, "fatigue", 0.0))
    hi = float(cfg["fatigue_high"])
    if f <= hi:
        return 0.0
    span = 1.0 - hi
    return g if span <= 0.0 else g * (f - hi) / span


# ---------------------------------------------------------------- 病気(日次・新 stream "health")
def roll_illness(agent, cfg: dict, rng, step: int) -> dict | None:
    """病気の日次遷移(発症/回復)。副作用=agent.sick/agent.sick_until を更新。

    rng は呼び出し側(scheduler)が引く新 stream "health"(既存 draw 順に挿入しない=決定論・
    ゴールデン保護)。戻り値: None(変化なし)/ {"state":"onset","days":d} / {"state":"recover"}。
    発症中(sick)は sick_until に達したら回復。非発症は onset_prob で発症(数日間 sick)。
    """
    if getattr(agent, "sick", False):
        if step >= int(getattr(agent, "sick_until", -1)):
            agent.sick = False
            agent.sick_until = -1
            return {"state": "recover"}
        return None
    p = float(cfg["onset_prob"])
    if p > 0.0 and rng.random() < p:
        days = int(cfg["illness_days"])
        agent.sick = True
        agent.sick_until = step + days * 144
        return {"state": "onset", "days": days}
    return None


def roll_medical(cfg: dict, rng) -> bool:
    """発症直後に受診するか(medical_visit)。roll_illness と同じ "health" stream を続けて引く。"""
    p = float(cfg["medical_prob"])
    return p > 0.0 and bool(rng.random() < p)


def is_sick(agent) -> bool:
    """病気で在宅療養中か(routine の欠勤/在宅ゲート用)。既定 OFF では常に False。"""
    return bool(getattr(agent, "sick", False))


# ---------------------------------------------------------------- メンタル(日次・grievance 参照)
def update_mental(agent, cfg: dict, step: int, sim_min: int, logger) -> None:
    """慢性的な高 grievance の持続 → 引きこもり(withdrawn)を日次更新。

    grievance の参照はこの module(src/society 直下=CHECKED_DIRS 外)に閉じる=no-fingerprint 契約に
    触れない。withdrawn は drive/arousal と並列の内部 transient(states 監査集合に入れない=R²(k) を
    汚さない)。高 grievance が mental_withdraw_days 連続で引きこもりに入り、grievance が閾値を下回れば
    解除。遷移した step だけ health_update(name="mental")を記録する。mental_withdraw_days<=0 でメンタル無効。
    """
    need = int(cfg["mental_withdraw_days"])
    if need <= 0:
        return
    thr = float(cfg["mental_grievance_threshold"])
    g = float(agent.states.get("grievance", 0.0))
    if g >= thr:
        agent.chronic_days = int(getattr(agent, "chronic_days", 0)) + 1
    else:
        agent.chronic_days = 0
    was = bool(getattr(agent, "withdrawn", False))
    if not was and agent.chronic_days >= need:
        agent.withdrawn = True
        _log_mental(agent, step, sim_min, logger, 0, 1, "withdraw")
    elif was and g < thr:
        agent.withdrawn = False
        _log_mental(agent, step, sim_min, logger, 1, 0, "recover")


def is_withdrawn(agent) -> bool:
    """引きこもり傾向(自由時間の外出抑制)か。既定 OFF では常に False。"""
    return bool(getattr(agent, "withdrawn", False))


def _log_mental(agent, step: int, sim_min: int, logger, old: int, new: int,
                cause: str) -> None:
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="health_update", x=agent.x, y=agent.y,
                     payload={"name": "mental", "old": old, "new": new,
                              "cause": cause}))
