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

★ 世代交代: 重症度の状態機械(レーン H1・2026-08-10。正典 docs/plans/body-incident-layer-plan.md §1)
--------------------------------------------------------------------------------------------------
上の「病気」は **真偽値ひとつ**(sick)しか持っておらず、「軽い風邪で出勤する」も「路上で倒れて
救急が来る」も同じ 1 ビットで表現されていた。``health.severity.enabled``(既定 false)で、その
1 ビットを **S0 健康 → S1 軽症 → S2 中等症 → S3 重症 → S4 心停止 → {回復, 死亡}** の状態機械へ
置き換える。既定 OFF では下の severity 経路を **1 行も通さない**(=現行と完全同値)。

  発症チャネル(各 = 独立ハザード × 共変量。すべて新 stream "health_onset"・実数較正)
    急病       … 現行 onset_prob の後継。S1-S2 帯・回復 3〜7 日
    熱中症     … λ(WBGT) の閾値関数(WBGT 32 以上で急峻)。``sim.today_weather["wbgt"]`` を読む
                  (この値は weather_gen が既に作っていて **誰も読んでいなかった**)。屋外・高齢で増幅
    急性アルコール … 夜間店舗に居るときだけ活性・20 代ピーク・金土で増幅・74% が軽症
    外傷(転倒)… 高齢重み・混雑/夜間で増幅
    心停止/脳卒中 … 極低率。**年齢帯レート必須**(全国平均の直適用は禁忌 = 渋谷の街頭人口は 20-40 代)

  frailty … 年齢帯の通院者率(65+ ≒ 70%)から導く体力係数。**全ハザードを乗算**する。
            (seed, agent.id) の安定ハッシュで個体差を付ける **純関数**(新しい乱数を引かない)。
  sick-role … S1 は予定を守って出勤しうる(日本の実測 presenteeism ≒ 45%)+ 発火閾値のデバフ。
              S2 で受診 / 停留(既存 ``_sick_home`` プリミティブを再利用)。S3 以上は行動不能。
  互換写像 … 既存の消費者(routine.in_work_window / _sick_home / city_ops.ems)は今までどおり
             ``agent.sick`` だけを読む。severity ON のときは本 module が
             **S2 以上 → sick=True / S1 は presenteeism の裏返し / S0 → False** を書き込む。
  死 … ユーザー決定(計画書 §6-1)。H1 トグル配下(別オプトインにしない)。**境界 despawn と同型**の
       永続退場(loc="outside" + 帰還予定を実質無限へ)+ L1 1 行 + causality(cause_type=physics)。
       中立語彙・演出ゼロ。量は較正で担保する(OHCA 生存退院 ≒ 10%)。

  R1: 本節も generate() を 1 本も足さない(身体は物理・選択は決定論の純関数)。新しい乱数は
      すべて **新 named stream**("health_onset" / "health_outcome")で、既存 stream の消費順を
      1 バイトも動かさない。
"""
from __future__ import annotations

import hashlib

from .observer.schema import Event, register_event_kind


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
        # ---- 重症度の状態機械(H1 世代交代。既定 OFF=上の sick ブールがそのまま現行動作)----
        "severity": build_severity_cfg(raw.get("severity")),
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
def roll_illness(agent, cfg: dict, rng, step: int,
                 steps_per_day: int = 144) -> dict | None:
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
        agent.sick_until = step + days * int(steps_per_day)
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


# =========================================================================== #
#                    重症度の状態機械(H1 世代交代・既定 OFF)
# =========================================================================== #
# ★L1 イベント種は**材料側 registration**(city_ops.py:167 / devices.py と同じ流儀 =
#   observer/schema.py 本体を触らずに自分の種を自分で登録する)。``illness`` /
#   ``medical_visit`` / ``health_update`` は schema.py に既登録なのでここでは足さない
#   (``illness`` の契約には最初から未使用の ``kind?`` が予約されており、発症チャネル名を
#   そこへ載せる = スキーマ無風)。
register_event_kind(
    "death", "死(身体の出来事。境界 despawn と同型の永続退場)。agent_id = 本人"
             "{cause, sev, age_band, frailty, day}")

# 重症度(S0 健康 → S1 軽症 → S2 中等症 → S3 重症 → S4 心停止)
S_HEALTHY, S_MILD, S_MODERATE, S_SEVERE, S_ARREST = 0, 1, 2, 3, 4
SEVERITY_NAMES: tuple[str, ...] = ("healthy", "mild", "moderate", "severe", "arrest")

# 発症チャネル(``illness`` payload の kind 欄に載る語。自由文ではない固定語彙)
CH_ILLNESS = "illness"        # 急病(風邪・胃腸炎系)= 現行 onset_prob の後継
CH_HEAT = "heatstroke"        # 熱中症
CH_ALCOHOL = "alcohol"        # 急性アルコール
CH_TRAUMA = "trauma"          # 外傷(転倒)
CH_CARDIAC = "cardiac"        # 心停止 / 脳卒中
CHANNELS: tuple[str, ...] = (CH_ILLNESS, CH_HEAT, CH_ALCOHOL, CH_TRAUMA, CH_CARDIAC)

#: 死者の帰還予定 step(境界 despawn と同じ機構を使い、実質無限で永久に戻らない)
NEVER_RETURN = 10 ** 9

# --------------------------------------------------------------------------- #
# 記憶へ入る定型 1 行(**チャネルだけの純関数**。数字・金額・config・実験条件・機構語ゼロ)
# = city_ops.COLLAPSE_TEXT / traces.sentence と同じ no-fingerprint 規約。
# --------------------------------------------------------------------------- #
ONSET_TEXT: dict[str, str] = {
    CH_ILLNESS: "体調を崩した",
    CH_HEAT: "暑さで気分が悪くなった",
    CH_ALCOHOL: "酔いが回って気分が悪くなった",
    CH_TRAUMA: "転んで体を痛めた",
    CH_CARDIAC: "急に胸が苦しくなった",
}
RECOVER_TEXT = "体調が回復した"
CARE_TEXT = "医療機関にかかった"
WORSEN_TEXT = "容体が悪くなった"

# --------------------------------------------------------------------------- #
# 年齢帯テーブル(**全国レートの直適用は禁忌**という計画書 §7 の要請への回答)
# --------------------------------------------------------------------------- #
#: 通院者率(国民生活基礎調査の年齢階級別・人口千対を割合へ直したもの)。frailty の素。
#  65 歳以上でおよそ 0.70 = 計画書 §1 が挙げた「65+=70%」のアンカー。
_TSUUIN = ((20, 0.15), (30, 0.20), (40, 0.25), (50, 0.32),
           (60, 0.43), (65, 0.56), (75, 0.69), (999, 0.73))

#: 年齢帯ラベル(L1 payload の age_band 欄。解析側で内生性を機械検証するための前兆状態)
_AGE_BANDS = ((20, "u20"), (30, "20s"), (40, "30s"), (50, "40s"),
              (65, "50-64"), (75, "65-74"), (999, "75+"))

#: 院外心停止(OHCA)の年齢帯別レート [件/人日]。全国の年間発生率(人口 10 万対)を
#  365 で割った値。20-40 代が主体の街頭人口へ全国平均(≒100/10万/年)を直に当てると
#  脳卒中過剰になるため、必ず年齢帯で引く(計画書 §7「年齢構造」)。
_OHCA_BY_AGE = ((20, 5.5e-8), (40, 1.9e-7), (65, 1.1e-6),
                (75, 4.9e-6), (999, 1.9e-5))

#: 急性アルコールの年齢重み(20 代ピーク)。搬送実績の年齢分布から。
_ALCOHOL_BY_AGE = ((20, 0.20), (30, 1.00), (40, 0.60), (60, 0.30), (999, 0.15))

#: 転倒・外傷の年齢重み(高齢で急増 = 一般負傷搬送の年齢分布)。
_TRAUMA_BY_AGE = ((65, 0.50), (75, 2.00), (999, 5.00))

#: チャネルごとの「その日いつ起こりうるか」[分 of day]。複数区間可(アルコールは日跨ぎ)。
_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    CH_ILLNESS: ((0, 1440),),
    CH_HEAT: ((540, 1080),),                 # 09:00〜18:00(日中の暑熱曝露)
    CH_ALCOHOL: ((0, 300), (1080, 1440)),    # 18:00〜翌 05:00(nightlife の営業窓と同型)
    CH_TRAUMA: ((0, 1440),),
    CH_CARDIAC: ((0, 1440),),
}

SEVERITY_DEFAULTS: dict = {
    "enabled": False,
    # ---- frailty(年齢帯の通院者率 → 全ハザードの乗数)----
    "frailty_ref": 0.40,       # 基準の通院者率(この値で frailty=1.0)
    "frailty_jitter": 0.25,    # 個体差の幅(seed 純関数。新しい乱数は引かない)
    "frailty_min": 0.20,
    "frailty_max": 3.00,
    # ---- ① 急病(現行 onset_prob の後継)----
    #  成人の感冒・胃腸炎などの発症は年 2〜3 回 ≒ 2.5/365 = 6.8e-3 /人日。
    "acute_illness_daily": 0.0068,
    "acute_illness_days_lo": 3,      # 回復まで(日)
    "acute_illness_days_hi": 7,
    "acute_illness_mild": 0.85,      # 残りが S2(中等症)
    # ---- ② 熱中症: λ(WBGT) の閾値関数 ----
    #  都の日次搬送(23 区)から。WBGT 28 で ≒3e-6 /人日、WBGT 33 で ≒2.6e-5 /人日を
    #  通る指数曲線 λ = base·exp(slope·(WBGT − ref))。WBGT 25 未満は 0(発生しない)。
    "heat_base": 3.0e-6,
    "heat_ref_wbgt": 28.0,
    "heat_slope": 0.432,
    "heat_floor_wbgt": 25.0,
    "heat_outdoor_mult": 2.0,        # 屋外(路上)に居るあいだの増幅
    "heat_elderly_mult": 3.0,        # 65 歳以上の増幅
    "heat_mild": 0.55,               # S1 / 残りは S2:S3 = heat_moderate で按分
    "heat_moderate": 0.40,
    "heat_days_lo": 1,
    "heat_days_hi": 2,
    # ---- ③ 急性アルコール(夜間店舗に居るときだけ活性)----
    #  夜に飲食・夜間店舗へ居る個体 1 人日あたり。都の急性アルコール搬送(≒1.8 万件/年)を
    #  夜間の飲酒機会数で割り戻した水準。搬送の 74% が軽症(実測)。
    "alcohol_nightlife_daily": 3.0e-4,
    "alcohol_weekend_mult": 1.6,     # 金・土の増幅
    "alcohol_mild": 0.74,
    "alcohol_moderate": 0.24,
    "alcohol_days_lo": 1,
    "alcohol_days_hi": 1,
    # ---- ④ 外傷(転倒)----
    #  一般負傷は搬送のおよそ 2 割。区の 65 件/日 × 0.2 ÷ 23 万人 ≒ 5.7e-5 /人日 を
    #  年齢重み(_TRAUMA_BY_AGE)込みの平均として置いた基準値。
    "trauma_daily": 3.0e-5,
    "trauma_night_mult": 1.5,
    "trauma_crowd_mult": 1.4,
    "trauma_crowd_n": 6,             # 近傍にこの人数以上居れば「混雑」とみなす
    "trauma_mild": 0.62,
    "trauma_moderate": 0.33,
    "trauma_days_lo": 2,
    "trauma_days_hi": 10,
    # ---- ⑤ 心停止 / 脳卒中(極低率・年齢帯レート必須)----
    "cardiac_scale": 1.0,            # 年齢帯レート(_OHCA_BY_AGE)への一括較正係数
    "cardiac_arrest_share": 0.45,    # 発症のうち S4(心停止)の割合。残りは S3(脳卒中等)
    "cardiac_days_lo": 7,
    "cardiac_days_hi": 21,
    "ward_ohca_per_day": 0.6,        # ★参照値のみ(渋谷実測 0.5〜0.7 件/日)。判定に使わない
    # ---- 進行(S1→S2→S3。状態機械が単なる「跳躍」に退化しないための項)----
    "worsen_daily": 0.02,            # 発症中の 1 日あたり悪化確率
    # ---- sick-role ----
    "presenteeism": 0.45,            # S1 で予定どおり出勤する割合(日本の実測)
    "seek_care": 0.62,               # S2 で受診する割合(残りは自宅停留)
    "care_cost": 3000.0,             # 受診 1 回の自己負担(既存 spend cat="medical")
    "mild_threshold_gain": 0.06,     # S1 の性能デバフ(発火 effective_threshold の上げ幅)
    # ---- 搬送先での確定(通報時点の「見かけ」と分離する)----
    #  東京消防庁の搬送人員の程度別構成: 軽症 52.8% / 中等症 38.6% / 重症 7.3% / 死亡 1.3%。
    "confirm_mild": 0.528,
    "confirm_moderate": 0.386,
    "confirm_severe": 0.073,
    # ---- 死(計画書 §6-1 = ユーザー承認済み。H1 トグル配下・別オプトインにしない)----
    "ohca_survival": 0.10,           # OHCA の生存退院率 ≒ 10% → P(死|S4) = 0.90
    "severe_fatality": 0.005,        # ★S3 の致死率。搬送全体の死亡 1.3%(東京)から OHCA 寄与を
                                     #   差し引いた**残差として置いた較正値**で、単独の実測ではない
    "arrest_outcome_steps": 6,       # S4 の転帰が確定するまで
    "severe_outcome_steps": 144,     # S3 の致死転帰が確定するまで
}

_SEV_INTS = ("acute_illness_days_lo", "acute_illness_days_hi", "heat_days_lo",
             "heat_days_hi", "alcohol_days_lo", "alcohol_days_hi",
             "trauma_days_lo", "trauma_days_hi", "trauma_crowd_n",
             "cardiac_days_lo", "cardiac_days_hi", "arrest_outcome_steps",
             "severe_outcome_steps")


def build_severity_cfg(raw) -> dict:
    """``health.severity`` ブロックの正準化(既定 OFF = 現行動作と完全同一)。

    dict / OmegaConf / dotlist の文字列どれでも受け、**未知キーは黙って捨てる**
    (city_ops.build_cfg / traces.build_cfg と同じ流儀)。
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    out = dict(SEVERITY_DEFAULTS)
    out["enabled"] = bool(raw.get("enabled", False))
    for key, default in SEVERITY_DEFAULTS.items():
        if key == "enabled" or key not in raw:
            continue
        value = raw[key]
        if key in _SEV_INTS:
            out[key] = max(0, int(value))
        else:
            out[key] = max(0.0, float(value))
    for lo, hi in (("acute_illness_days_lo", "acute_illness_days_hi"),
                   ("heat_days_lo", "heat_days_hi"),
                   ("alcohol_days_lo", "alcohol_days_hi"),
                   ("trauma_days_lo", "trauma_days_hi"),
                   ("cardiac_days_lo", "cardiac_days_hi")):
        out[lo] = max(1, int(out[lo]))
        out[hi] = max(int(out[lo]), int(out[hi]))
    return out


def severity_cfg(cfg: dict) -> dict:
    """健康 cfg から severity ブロックを取り出す(旧 cfg にも耐える)。"""
    return dict(cfg or {}).get("severity") or SEVERITY_DEFAULTS


def severity_on(sim) -> bool:
    """重症度の状態機械が有効か。**健康そのものが OFF なら常に False**(正直な依存関係)。"""
    cfg = getattr(sim, "healthcfg", None)
    if not cfg or not cfg.get("enabled"):
        return False
    return bool(severity_cfg(cfg)["enabled"])


# --------------------------------------------------------------------------- #
# 純関数(乱数ゼロ・世界を 1 バイトも触らない)
# --------------------------------------------------------------------------- #
def _stable_hash(value: str) -> int:
    """プロセス間で安定なハッシュ(rng._stable_hash / city_ops._stable_hash と同式)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _band(table, age: int):
    for edge, value in table:
        if int(age) < edge:
            return value
    return table[-1][1]


def age_band(age: int) -> str:
    """年齢帯ラベル(L1 payload の前兆状態欄)。"""
    return str(_band(_AGE_BANDS, int(age)))


def frailty_of(agent, scfg: dict, seed: int) -> float:
    """体力係数 = 年齢帯の通院者率 / 基準 × 個体差。**全ハザードを乗算する**。

    ★健康が ``age`` を読むのは本 module で初めて(監査でグリーンフィールド確認済み)。
    ★個体差は (seed, agent.id) の安定ハッシュ = **新しい乱数を 1 本も引かない純関数**。
      同じ seed・同じ id なら resume を跨いでもプール回転を跨いでも同じ値になる。
    """
    ref = float(scfg["frailty_ref"]) or 1.0
    base = float(_band(_TSUUIN, int(getattr(agent, "age", 30)))) / ref
    jit = float(scfg["frailty_jitter"])
    if jit > 0.0:
        # [-jit, +jit] の一様相当(安定ハッシュの下位 4 桁を写像)
        u = (_stable_hash(f"frailty/{int(seed)}/{int(agent.id)}") % 10000) / 10000.0
        base *= 1.0 + jit * (2.0 * u - 1.0)
    lo, hi = float(scfg["frailty_min"]), float(scfg["frailty_max"])
    return lo if base < lo else hi if base > hi else base


def severity_of(agent) -> int:
    """いまの重症度(S0..S4)。severity OFF / 健康な個体では常に 0。"""
    return int(getattr(agent, "severity", 0) or 0)


def apparent_severity(agent) -> int:
    """**通報時点の見かけの重症度**(居合わせた人が観察できる量)。

    確定重症度(``confirm_severity``)とは別物である = 「通報は不確実性から生まれる」の
    再現(東京の実測では搬送の 52.8% が軽症)。
    """
    return severity_of(agent)


def confirm_severity(agent, scfg: dict, seed: int, step: int) -> int:
    """搬送先で**確定する**重症度(乱数ゼロの純関数)。

    見かけ(``apparent_severity``)とは独立に、東京消防庁の搬送人員の程度別構成
    (軽症 52.8 / 中等症 38.6 / 重症 7.3 / 死亡 1.3 %)から引く。心停止だけは
    見誤りようがないので、見かけが S4 ならそのまま S4 で確定する。
    ★H1 の範囲では**この値は身体状態へ戻さない**(搬送・入院・会計は H2 レーンの領分)。
      本波では観測量として ``ems_dispatch`` の payload と ``agent.sev_confirmed`` に置くだけ。
    """
    if severity_of(agent) >= S_ARREST:
        return S_ARREST
    u = (_stable_hash(f"confirm/{int(seed)}/{int(agent.id)}/{int(step)}") % 10000) / 10000.0
    acc = float(scfg["confirm_mild"])
    if u < acc:
        return S_MILD
    acc += float(scfg["confirm_moderate"])
    if u < acc:
        return S_MODERATE
    acc += float(scfg["confirm_severe"])
    return S_SEVERE if u < acc else S_ARREST


def note_confirmed(sim, agent, step: int) -> int:
    """搬送先での確定重症度を**健康側で**決めて刻み、値を返す(city_ops からの唯一の口)。

    ★city_ops は健康状態へ 1 バイトも書かない(あちらの AST 検査がそれを機械固定している)。
      確定は身体の意味論なので書き手は本 module に閉じる。
    """
    scfg = severity_cfg(sim.healthcfg)
    got = confirm_severity(agent, scfg, int(getattr(sim.hub, "master_seed", 0)),
                           int(step))
    agent.sev_confirmed = int(got)
    return int(got)


def collapse_source(agent, step: int) -> str:
    """**S3/S4 への遷移そのもの**が救急の引き金(city_ops の暫定ハッシュゲートの世代交代)。

    city_ops._collapse_gate から読まれる唯一の口。戻り値は発症チャネル名(倒れないなら "")。
    ★遷移した **その step だけ** 名前を返す = 同じ発症で二度倒れない(city_ops 側に
      「消費した」印を書かせないための設計 = あちらは健康状態へ 1 バイトも書かない)。
    """
    if int(getattr(agent, "sev_collapse_step", -1)) != int(step):
        return ""
    return str(getattr(agent, "sev_channel", "") or CH_ILLNESS)


def severity_threshold_delta(agent, cfg: dict) -> float:
    """軽症で出勤している個体の**性能デバフ**(発火 effective_threshold の上げ幅)。

    ``fatigue_threshold_delta`` と同じ合流点(drive.effective_threshold の clip)へ渡す
    不透明な float。severity OFF / 健康な個体では 0.0 = 恒等(バイト一致)。
    """
    scfg = severity_cfg(cfg)
    if not scfg["enabled"]:
        return 0.0
    return float(scfg["mild_threshold_gain"]) if severity_of(agent) == S_MILD else 0.0


def is_dead(agent) -> bool:
    """永続退場(死)しているか。既定 OFF では常に False。"""
    return bool(getattr(agent, "dead", False))


# --------------------------------------------------------------------------- #
# ハザード(すべて「その日その個体が発症する確率」= 日次・Δt 非依存)
# --------------------------------------------------------------------------- #
def heat_hazard(agent, scfg: dict, wbgt) -> float:
    """熱中症の日次ハザード λ(WBGT)。WBGT が無い(天気 OFF)ランでは 0 = 発生しない。"""
    if wbgt is None:
        return 0.0
    w = float(wbgt)
    if w < float(scfg["heat_floor_wbgt"]):
        return 0.0
    import math
    lam = float(scfg["heat_base"]) * math.exp(
        float(scfg["heat_slope"]) * (w - float(scfg["heat_ref_wbgt"])))
    if str(getattr(agent, "loc", "street")) == "street" and not agent.building:
        lam *= float(scfg["heat_outdoor_mult"])
    if int(getattr(agent, "age", 30)) >= 65:
        lam *= float(scfg["heat_elderly_mult"])
    return lam


def alcohol_hazard(agent, scfg: dict, weekend: bool) -> float:
    """急性アルコールの日次ハザード。**発症の実現は夜間店舗に居ることが条件**(発火時に判定)。"""
    lam = float(scfg["alcohol_nightlife_daily"])
    lam *= float(_band(_ALCOHOL_BY_AGE, int(getattr(agent, "age", 30))))
    if weekend:
        lam *= float(scfg["alcohol_weekend_mult"])
    return lam


def trauma_hazard(agent, scfg: dict) -> float:
    """外傷(転倒)の日次ハザード。年齢重みが主項・混雑/夜間の増幅は発火時に効かせる。"""
    return float(scfg["trauma_daily"]) * float(
        _band(_TRAUMA_BY_AGE, int(getattr(agent, "age", 30))))


def cardiac_hazard(agent, scfg: dict) -> float:
    """心停止 / 脳卒中の日次ハザード(**年齢帯レート**。全国平均の直適用はしない)。"""
    return float(_band(_OHCA_BY_AGE, int(getattr(agent, "age", 30)))) \
        * float(scfg["cardiac_scale"])


# --------------------------------------------------------------------------- #
# 世界の読み取り(前兆状態。すべて観測=世界を 1 バイトも書き換えない)
# --------------------------------------------------------------------------- #
def today_wbgt(sim):
    """当日の暑さ指数(``sim.today_weather["wbgt"]``)。天気 OFF / 欠測では None。

    ★この値は weather_gen が最初から作っていて **本 module 以前は誰も読んでいなかった**
      (計画書 §1「シーム敷設済み」)。欠測を 0 と偽らない = None のまま返す。
    """
    w = getattr(sim, "today_weather", None)
    if not isinstance(w, dict):
        return None
    got = w.get("wbgt")
    return None if got is None else float(got)


def _is_weekend_night(sim, sim_min: int) -> bool:
    """金・土か(飲酒の増幅日)。暦 OFF のランは経過日の剰余で 0=月として数える。"""
    cal = getattr(sim, "calendarcfg", None)
    if cal and cal.get("enabled"):
        from .world import calendar as _cal
        return _cal.weekday_of(cal, int(sim_min)) in (4, 5)
    return ((int(sim_min) // 1440) % 7) in (4, 5)


def _nightlife_nodes(sim) -> frozenset:
    """夜間店舗のあるノード集合(地図からの決定論。初回のみ構築してキャッシュ)。"""
    got = getattr(sim, "_h1_night_nodes", None)
    if got is None:
        try:
            got = frozenset(str(p["node"]) for p in sim.city.pois_by_cat("nightlife"))
        except Exception:                          # noqa: BLE001(地図の形が違う保険)
            got = frozenset()
        sim._h1_night_nodes = got
    return got


def _at_nightlife(sim, agent) -> bool:
    """いま飲酒活動の現場(夜間店舗のノード)に居るか = アルコール発症の必要条件。"""
    if agent.loc == "outside" or agent.sleeping:
        return False
    return str(agent.node) in _nightlife_nodes(sim)


def _present(agent) -> bool:
    """世界の中に居るか(範囲外へ出ている個体には身体の出来事を起こさない)。"""
    return agent.loc != "outside" and not is_dead(agent)


def _near_count(sim, agent) -> int:
    """近傍の在場人数(混雑の代理)。

    ★共在判定は**唯一の索引持ち** ``sim.percept_index`` に相乗りする(計画書 §3
      「共在判定の一本化」= O(n²) の独立スキャンを新設しない)。索引はこの step の
      位置確定後に張り直されるので、身体フェーズが読むのは**直前 step の索引**である
      (正直な既知の粗さ。混雑の代理量としては 1 step のずれは無視できる)。
      索引がまだ無い(初 step)ランでは 0 = 捏造しない。
    """
    idx = getattr(sim, "percept_index", None)
    if idx is None:
        return 0
    try:
        return len(idx.hearers(agent))
    except Exception:                              # noqa: BLE001(索引 API 差の保険)
        return 0


# --------------------------------------------------------------------------- #
# 観測タリー(OFF では None = 何も出さない。city_ops.provenance と同じ作法)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_h1_state", None)
    if st is None:
        st = {"onsets": {c: 0 for c in CHANNELS}, "worsen": 0, "recover": 0,
              "collapses": 0, "deaths": 0, "deaths_by_channel": {}, "care": 0,
              "presentee": 0, "dropped": {c: 0 for c in CHANNELS}, "days": 0,
              # 外部の出来事(喧嘩・火災・交通)から入った発症の件数(``on_injury``)。
              # 内訳は onsets[trauma] に合流するので、ここは**外から来た分**の内数。
              "injuries": 0}
        sim._h1_state = st
    return st


def provenance(sim) -> dict | None:
    """重症度層の観測タリー(既定 OFF は None)。★較正の突き合わせをここで正直に出す。

    第109バッチ D2 で ``summary.json`` の ``health`` キーへ配線した(OFF = キー自体を
    出さないので既存ラン・golden は無風)。★None を返す条件は ``severity_on`` なので、
    ``health.enabled=true`` でも ``health.severity.enabled=false`` のランはキーが出ない
    (重症度層が回っていない = 出す観測量が無い、という正直な区別)。
    """
    if not severity_on(sim):
        return None
    st = getattr(sim, "_h1_state", None)
    scfg = severity_cfg(sim.healthcfg)
    out: dict = {"schema": 1, "severity_names": list(SEVERITY_NAMES),
                 "channels": list(CHANNELS)}
    if st is None:
        out.update({"onsets": {c: 0 for c in CHANNELS}, "deaths": 0, "days": 0})
        return out
    days = max(1, int(st["days"]))
    out.update({"onsets": {k: int(v) for k, v in sorted(st["onsets"].items())},
                "dropped": {k: int(v) for k, v in sorted(st["dropped"].items())},
                "worsen": int(st["worsen"]), "recover": int(st["recover"]),
                "collapses": int(st["collapses"]), "deaths": int(st["deaths"]),
                "deaths_by_channel": {k: int(v) for k, v
                                      in sorted(st["deaths_by_channel"].items())},
                "care": int(st["care"]), "presentee": int(st["presentee"]),
                "injuries": int(st.get("injuries", 0)),
                "days": days,
                "deaths_per_day": round(float(st["deaths"]) / days, 4),
                "collapses_per_day": round(float(st["collapses"]) / days, 3)})
    n_agents = len(getattr(sim, "agents", ()) or ())
    if n_agents > 0:
        # 渋谷の実測 OHCA 0.5〜0.7 件/日 は区の人口(約 23 万人)に対する数字。本ランの
        # 規模へ線形で割り戻した**物差し**であって、目標値でも検定でもない。
        out["ohca_reference_per_day"] = round(
            float(scfg["ward_ohca_per_day"]) * n_agents / 230000.0, 4)
    return out


# --------------------------------------------------------------------------- #
# 状態の書き込み(**互換写像を 1 箇所に閉じる**)
# --------------------------------------------------------------------------- #
def _apply_sick_role(agent, scfg: dict, presentee: bool) -> None:
    """重症度 → 既存 ``sick`` ブールへの互換写像(唯一の書き手)。

      S0 健康        … sick=False
      S1 軽症        … presenteeism なら sick=False(予定を守って出勤・性能デバフのみ)
                       そうでなければ sick=True(在宅療養 = 現行の欠勤経路へ合流)
      S2 中等症以上  … sick=True(受診/停留。S3 以上は事実上の行動不能)

    ★既存の消費者(routine.in_work_window / routine._sick_home / city_ops.ems.require_sick)は
      今までどおり ``sick`` だけを読めばよい = 呼び出し側を 1 行も書き換えない世代交代。
    """
    sev = severity_of(agent)
    if sev >= S_MODERATE:
        agent.sick = True
    elif sev == S_MILD:
        agent.sick = not bool(presentee)
    else:
        agent.sick = False
    agent.sick_until = int(getattr(agent, "sev_until", -1)) if agent.sick else -1


def _clear_severity(agent) -> None:
    agent.severity = S_HEALTHY
    agent.sev_channel = ""
    agent.sev_until = -1
    agent.sev_outcome_step = -1
    agent.sev_fatal = False
    agent.sev_presentee = False
    agent.sev_collapse_step = -1
    agent.sev_confirmed = -1
    agent.sick = False
    agent.sick_until = -1


def _payload(agent, scfg: dict, state: str, extra: dict) -> dict:
    """L1 の 1 行 + **前兆状態**(WBGT・飲酒・年齢帯・frailty…)を同梱した payload。

    ★内生性を解析側で機械検証できるようにするための同梱(計画書 §3 の L1 規約)。
      自由文・地名・実験条件は 1 つも入らない(固定語彙と数量だけ)。
    """
    out = {"state": state, "kind": str(getattr(agent, "sev_channel", "") or ""),
           "sev": severity_of(agent),
           "age_band": age_band(int(getattr(agent, "age", 30))),
           "frailty": round(float(getattr(agent, "sev_frailty", 1.0)), 3)}
    out.update(extra)
    return out


def _log(logger, agent, kind: str, payload: dict, step: int, sim_min: int) -> None:
    logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id), kind=kind,
                     x=agent.x, y=agent.y, payload=payload))


# =========================================================================== #
# ① 日次: ハザード抽選 + 進行 + sick-role(新 stream "health_onset")
# =========================================================================== #
#: 1 個体 1 日あたりに引く一様乱数の本数。**分岐で消費順が変わらないよう固定長で引く**
#  (どの枝を通っても同じ本数 = resume・ON/ON 再現の決定論を構造で担保する)。
_U = 12
_U_WORSEN, _U_SEVDIST, _U_DAYS, _U_SLOT = 0, 6, 7, 8
_U_FATAL, _U_PRESENTEE, _U_CARE = 9, 10, 11


def _draw_slot_min(scfg: dict, channel: str, u: float) -> int:
    """そのチャネルが起こりうる時間帯[分 of day]の中から発症時刻を引く(決定論)。"""
    spans = _WINDOWS[channel]
    total = sum(hi - lo for lo, hi in spans)
    pos = float(u) * total
    for lo, hi in spans:
        width = hi - lo
        if pos < width:
            return int(lo + pos)
        pos -= width
    return int(spans[-1][1] - 1)


def _draw_days(scfg: dict, channel: str, u: float) -> int:
    lo = int(scfg[f"{channel_key(channel)}_days_lo"])
    hi = int(scfg[f"{channel_key(channel)}_days_hi"])
    return lo + int(float(u) * (hi - lo + 1)) if hi > lo else lo


def channel_key(channel: str) -> str:
    """チャネル名 → conf のキー接頭辞(固定表。自由文ではない)。"""
    return {CH_ILLNESS: "acute_illness", CH_HEAT: "heat", CH_ALCOHOL: "alcohol",
            CH_TRAUMA: "trauma", CH_CARDIAC: "cardiac"}[channel]


def _draw_initial_severity(scfg: dict, channel: str, u: float) -> int:
    """発症時の重症度(チャネル別の実測分布から)。"""
    u = float(u)
    if channel == CH_CARDIAC:
        return S_ARREST if u < float(scfg["cardiac_arrest_share"]) else S_SEVERE
    if channel == CH_ILLNESS:
        return S_MILD if u < float(scfg["acute_illness_mild"]) else S_MODERATE
    mild = float(scfg[f"{channel_key(channel)}_mild"])
    mod = float(scfg[f"{channel_key(channel)}_moderate"])
    if u < mild:
        return S_MILD
    return S_MODERATE if u < mild + mod else S_SEVERE


def severity_day(sim, step: int, sim_min: int, spend=None) -> None:
    """日次境界: 発症チャネルの抽選・進行・回復予定・sick-role の確定。

    ★乱数はすべて **新 stream "health_onset"**((agent, step) キー)= 既存 draw 順へ
      1 本も割り込まない。1 個体につき固定長 12 本を引くので、どのチャネルが当たっても
      次の個体の消費位置は動かない(決定論を分岐に依存させない)。
    ★``spend`` は呼び出し側(scheduler)が渡す受診の支払い口(既存 ``_spend`` の薄い包み)。
      None なら受診の会計だけを行わない(単体テストからの直接呼び出し用)。
    """
    scfg = severity_cfg(sim.healthcfg)
    logger = sim.logger
    st = _state(sim)
    st["days"] += 1
    seed = int(getattr(sim.hub, "master_seed", 0))
    day = int(sim_min) // 1440
    wbgt = today_wbgt(sim)
    weekend = _is_weekend_night(sim, sim_min)
    for agent in sim.agents:                       # id 昇順 = 決定論
        if is_dead(agent):
            continue
        u = sim.hub.stream("health_onset", int(agent.id), int(step)).random(_U)
        if severity_of(agent) > 0:
            _day_progress(agent, scfg, u, sim, st, step, sim_min, logger, spend)
            continue
        if getattr(agent, "sev_pending", None):
            continue                               # 当日ぶんの発症は予約済み
        _day_onset_roll(agent, scfg, u, sim, st, step, sim_min, day, wbgt,
                        weekend, seed, logger)


def _day_progress(agent, scfg: dict, u, sim, st: dict, step: int, sim_min: int,
                  logger, spend) -> None:
    """発症中の 1 日: 悪化判定 + sick-role の再確定(受診はここで 1 回だけ)。"""
    sev = severity_of(agent)
    if sev < S_SEVERE and float(u[_U_WORSEN]) < float(scfg["worsen_daily"]):
        agent.severity = sev + 1
        st["worsen"] += 1
        _presentee_and_care(agent, scfg, u, sim, st, step, sim_min, logger, spend)
        if agent.severity >= S_SEVERE:
            _mark_collapse(agent, scfg, float(u[_U_FATAL]), st, step)
        agent.remember(WORSEN_TEXT)
        _log(logger, agent, "illness",
             _payload(agent, scfg, "worsen", {"from": sev}), step, sim_min)
    else:
        _apply_sick_role(agent, scfg, bool(getattr(agent, "sev_presentee", False)))
        # 発症日に停留を選んだ中等症が、翌日以降に受診へ切り替わる余地(1 回まで)
        maybe_care(agent, scfg, float(u[_U_CARE]), st, step, sim_min, logger, spend,
                   sim)


def _day_onset_roll(agent, scfg: dict, u, sim, st: dict, step: int, sim_min: int,
                    day: int, wbgt, weekend: bool, seed: int, logger) -> None:
    """5 チャネルの独立ハザード × frailty。当たったら**その日の発症時刻を予約する**。"""
    fr = frailty_of(agent, scfg, seed)
    agent.sev_frailty = fr
    hazards = ((CH_ILLNESS, float(scfg["acute_illness_daily"])),
               (CH_HEAT, heat_hazard(agent, scfg, wbgt)),
               (CH_ALCOHOL, alcohol_hazard(agent, scfg, weekend)),
               (CH_TRAUMA, trauma_hazard(agent, scfg)),
               (CH_CARDIAC, cardiac_hazard(agent, scfg)))
    fired = ""
    for i, (channel, lam) in enumerate(hazards):
        if lam > 0.0 and float(u[1 + i]) < lam * fr:
            fired = channel
            break                                  # 同日 2 チャネルは作らない(先勝ち)
    if not fired:
        return
    sev = _draw_initial_severity(scfg, fired, float(u[_U_SEVDIST]))
    agent.sev_pending = {"day": int(day), "slot_min": _draw_slot_min(scfg, fired,
                                                                    float(u[_U_SLOT])),
                         "ch": fired, "sev": int(sev),
                         "days": _draw_days(scfg, fired, float(u[_U_DAYS])),
                         "u_fatal": float(u[_U_FATAL]),
                         "u_presentee": float(u[_U_PRESENTEE]),
                         "u_care": float(u[_U_CARE]), "wbgt": wbgt}


def _presentee_and_care(agent, scfg: dict, u, sim, st: dict, step: int,
                        sim_min: int, logger, spend) -> None:
    """sick-role の決定論選択: S1=出勤するか / S2=受診するか(乱数は既に引いた分を使う)。"""
    sev = severity_of(agent)
    presentee = sev == S_MILD and float(u[_U_PRESENTEE]) < float(scfg["presenteeism"])
    agent.sev_presentee = bool(presentee)
    if presentee:
        st["presentee"] += 1
    _apply_sick_role(agent, scfg, presentee)
    maybe_care(agent, scfg, float(u[_U_CARE]), st, step, sim_min, logger, spend, sim)


def _medical_care(sim, agent, cost: float, spend, step: int, sim_min: int) -> bool:
    """受診の支払いを **H2 レーン(``medical.py``)へ委ねる soft seam**。

    あちらが ON で受診先(クリニック)を解決できたときだけ True を返し、支払い(受け手を
    是正した ``spend``)と保険給付の記帳を済ませている。既定 OFF / 受診先が無い /
    ``spend`` が None のときは False = 呼び出し側が**今までどおり**支払う(完全同値)。
    ★H1 は「いつ・誰が受診するか」だけを決める。受け手が誰かは医療の意味論なので H2 の領分。
    """
    if sim is None:
        return False
    try:
        from . import medical as _medical
    except Exception:                              # noqa: BLE001(あちらが無くても壊さない)
        return False
    hook = getattr(_medical, "on_care", None)
    if not callable(hook):
        return False
    try:
        return bool(hook(sim, agent, float(cost), spend, int(step), int(sim_min)))
    except Exception:                              # noqa: BLE001(契約が違っても壊さない)
        return False


def maybe_care(agent, scfg: dict, u_care: float, st: dict, step: int,
               sim_min: int, logger, spend, sim=None) -> None:
    """中等症の**受診 / 停留**の決定論選択(1 発症につき 1 回まで)。

    受診は既存の消費経路(``spend`` cat="medical" + ``medical_visit``)をそのまま使う。
    ``spend`` が None のとき(単体テスト・会計を持たない呼び出し)は会計だけ行わない。
    ★``sim`` は H2 レーン(搬送・医療費)への soft seam のためだけに受ける任意引数で、
      None(既定)のときは本関数の挙動は従来と 1 バイトも変わらない。
    """
    if severity_of(agent) != S_MODERATE or bool(getattr(agent, "sev_cared", False)):
        return
    if float(u_care) >= float(scfg["seek_care"]):
        return                                     # 停留(自宅で様子を見る)
    agent.sev_cared = True
    st["care"] += 1
    cost = float(scfg["care_cost"])
    if not _medical_care(sim, agent, cost, spend, step, sim_min):
        if spend is not None:
            spend(agent, cost, "medical")
    agent.remember(CARE_TEXT)
    _log(logger, agent, "medical_visit",
         {"node": str(agent.node), "cost": round(cost, 1),
          "sev": S_MODERATE, "kind": str(getattr(agent, "sev_channel", "") or "")},
         step, sim_min)


def _mark_collapse(agent, scfg: dict, u_fatal: float, st: dict, step: int) -> None:
    """S3/S4 への遷移そのものを刻む = **救急の引き金**(city_ops がこの step だけ読む)。

    転帰(死ぬか回復するか)も**ここで一度に決める**。治療の有無を転帰へ接続するのは
    搬送・入院を作る H2 レーンの領分なので、H1 では正直に「発症時に決まる」と宣言する。
    量の較正: P(死|S4) = 1 − OHCA 生存退院率(≒10%)。P(死|S3) は搬送全体の死亡 1.3%
    (東京)から OHCA 寄与を差し引いた**残差**として置いた値(単独の実測ではない)。
    """
    agent.sev_collapse_step = int(step)
    st["collapses"] += 1
    sev = severity_of(agent)
    fatal_p = (1.0 - float(scfg["ohca_survival"])) if sev >= S_ARREST \
        else float(scfg["severe_fatality"])
    agent.sev_fatal = bool(float(u_fatal) < fatal_p)
    if sev >= S_ARREST:
        agent.sev_outcome_step = int(step) + int(scfg["arrest_outcome_steps"])
    elif agent.sev_fatal:
        agent.sev_outcome_step = int(step) + int(scfg["severe_outcome_steps"])


# =========================================================================== #
# ② 毎 step: 予約された発症の発火 / 回復 / 転帰(**乱数ゼロの純粋な状態機械**)
# =========================================================================== #
def _can_realize(sim, agent, channel: str) -> bool:
    """発症が実現する条件。満たさなければ**その日の発症は起きなかった**ことにする。

    (= 飲酒していない個体が急性アルコールにはならない)。落ちた件数は
    provenance の ``dropped`` に出るので、黙って消えることはない。
    """
    if not _present(agent):
        return False
    if channel == CH_ALCOHOL:
        return _at_nightlife(sim, agent)
    if channel in (CH_HEAT, CH_TRAUMA):
        return not agent.sleeping
    return True                                    # 急病・心停止は就寝中にも起こる


def severity_tick(sim, step: int, sim_min: int, spend=None) -> None:
    """毎 step: 転帰の確定 → 回復 → 予約された発症の発火。**乱数を 1 本も引かない**。

    ★``city_ops.phase`` の**直前**に置く(scheduler)。S3/S4 への遷移が刻む
      ``sev_collapse_step`` を、同じ step のうちに救急の連鎖が読むため。
    ★``spend`` は受診の支払い口(既存 ``_spend`` の薄い包み)。発症したその場で中等症の
      受診/停留を決めるために要る(日境界まで待たせると「発症の翌日に受診」になる)。
    """
    scfg = severity_cfg(sim.healthcfg)
    st = _state(sim)
    logger = sim.logger
    minute = int(sim_min) % 1440
    day = int(sim_min) // 1440
    for agent in sim.agents:                       # id 昇順 = 決定論
        if is_dead(agent):
            # ★プール回転で同じペルソナが再実体化されると build_pool_agent が街の中へ
            #   立たせてしまう(presence は (pid, day) の純関数で死を知らない)。退場の
            #   状態を冪等に貼り直すだけ = L1 は 1 件も出さない(死は既に 1 行記録済み)。
            if agent.loc != "outside":
                _exit_world(agent)
            continue
        if severity_of(agent) > 0:
            out = int(getattr(agent, "sev_outcome_step", -1))
            if out >= 0 and int(step) >= out:
                _resolve_outcome(agent, scfg, st, step, sim_min, logger)
                continue
            until = int(getattr(agent, "sev_until", -1))
            if until >= 0 and int(step) >= until:
                _recover(agent, scfg, st, step, sim_min, logger)
        pending = getattr(agent, "sev_pending", None)
        if not pending or int(pending["day"]) != day:
            continue
        if minute < int(pending["slot_min"]):
            continue                               # まだその時刻に達していない
        agent.sev_pending = None
        if not _can_realize(sim, agent, str(pending["ch"])):
            st["dropped"][str(pending["ch"])] += 1
            continue
        _fire(sim, agent, scfg, pending, st, step, sim_min, logger, spend)


def _fire(sim, agent, scfg: dict, pending: dict, st: dict, step: int,
          sim_min: int, logger, spend=None) -> None:
    """発症(S0 → S1/S2/S3/S4)。L1 は 1 行 + **前兆状態**を同梱する。"""
    channel = str(pending["ch"])
    sev = int(pending["sev"])
    # 発火時の共変量(当日の抽選では判らない量)で軽く増幅する = 夜間・混雑の転倒。
    near = _near_count(sim, agent)
    night = not (6 * 60 <= int(sim_min) % 1440 < 22 * 60)
    if channel == CH_TRAUMA and sev < S_SEVERE:
        boost = float(scfg["trauma_night_mult"]) if night else 1.0
        if near >= int(scfg["trauma_crowd_n"]):
            boost *= float(scfg["trauma_crowd_mult"])
        if boost >= 2.0:                           # 悪条件が重なった転倒は 1 段重い
            sev += 1
    agent.severity = int(sev)
    agent.sev_channel = channel
    agent.sev_cared = False
    agent.sev_confirmed = -1
    agent.sev_outcome_step = -1
    agent.sev_fatal = False
    agent.sev_until = int(step) + int(pending["days"]) * max(
        1, int(sim.clock.steps_per_day))
    presentee = (agent.severity == S_MILD
                 and float(pending["u_presentee"]) < float(scfg["presenteeism"]))
    agent.sev_presentee = bool(presentee)
    if presentee:
        st["presentee"] += 1
    _apply_sick_role(agent, scfg, presentee)
    if agent.severity >= S_SEVERE:
        _mark_collapse(agent, scfg, float(pending["u_fatal"]), st, step)
    st["onsets"][channel] += 1
    agent.remember(ONSET_TEXT[channel])
    _log(logger, agent, "illness",
         _payload(agent, scfg, "onset",
                  {"days": int(pending["days"]),
                   "wbgt": pending.get("wbgt"),
                   "drinking": channel == CH_ALCOHOL,
                   "outdoor": agent.loc == "street" and not agent.building,
                   "near": int(near), "night": bool(night),
                   "presentee": bool(presentee)}),
         step, sim_min)
    # 中等症は発症したその場で受診 / 停留を決める(既存 spend cat="medical" を再利用)。
    # ★発症の記録より**後**に置く = L1 が「倒れる前に医者にかかった」順序にならない。
    maybe_care(agent, scfg, float(pending["u_care"]), st, step, sim_min, logger,
               spend, sim)


def _recover(agent, scfg: dict, st: dict, step: int, sim_min: int, logger) -> None:
    """回復(→ S0)。既存の ``illness`` state=recover 契約をそのまま使う。"""
    was = severity_of(agent)
    payload = _payload(agent, scfg, "recover", {"from": was})
    payload["sev"] = S_HEALTHY
    _clear_severity(agent)
    st["recover"] += 1
    agent.remember(RECOVER_TEXT)
    _log(logger, agent, "illness", payload, step, sim_min)


def _resolve_outcome(agent, scfg: dict, st: dict, step: int, sim_min: int,
                     logger) -> None:
    """S4(心停止)/ 致死性 S3 の転帰確定 = **死** または蘇生後の中等症へ戻る。"""
    if bool(getattr(agent, "sev_fatal", False)):
        _die(agent, scfg, st, step, sim_min, logger)
        return
    agent.sev_outcome_step = -1
    if severity_of(agent) >= S_ARREST:             # 蘇生 → 中等症へ落ちて療養が続く
        agent.severity = S_MODERATE
        _apply_sick_role(agent, scfg, False)
        _log(logger, agent, "illness",
             _payload(agent, scfg, "improve", {"from": S_ARREST}), step, sim_min)


def _die(agent, scfg: dict, st: dict, step: int, sim_min: int, logger) -> None:
    """死 = **境界 despawn と同型の永続退場**(ユーザー決定 §6-1)。

    新しい退場機構は作らない: 既存の「範囲外へ出た個体」の表現(``loc="outside"`` +
    帰還予定 ``return_at``)をそのまま使い、帰還予定を実質無限へ置くだけで永続になる。
    したがって全フェーズの「範囲外は飛ばす」判定がそのまま効き、``sim.agents`` からの
    抜き取り(= 他レーンが持つ反復の前提を壊す操作)を 1 度も行わない。
    L1 は **1 行だけ**(演出ゼロ・中立語彙)。因果台帳では cause_type=physics。
    """
    channel = str(getattr(agent, "sev_channel", "") or CH_CARDIAC)
    payload = {"cause": channel, "sev": severity_of(agent),
               "age_band": age_band(int(getattr(agent, "age", 30))),
               "frailty": round(float(getattr(agent, "sev_frailty", 1.0)), 3),
               "day": int(sim_min) // 1440}
    _log(logger, agent, "death", payload, step, sim_min)
    st["deaths"] += 1
    st["deaths_by_channel"][channel] = st["deaths_by_channel"].get(channel, 0) + 1
    _clear_severity(agent)
    agent.dead = True
    agent.sev_channel = channel                    # 死因は残す(観測用・行動には効かない)
    _exit_world(agent)


def _exit_world(agent) -> None:
    """世界から退く物理状態(**境界 despawn と同じ表現**)。冪等・L1 を 1 件も出さない。"""
    agent.sleeping = False
    agent.route = []
    agent.dest = None
    agent.homing = False
    agent.exit_intent = False
    agent.building = None
    agent.loc = "outside"
    agent.return_at = NEVER_RETURN
    agent.stay_until = 0
    agent.day_plan = []
    agent.plan_step = -1
    # ★DPH-B(`lod.budget.tiers`)の FIFO 繰り越し予約の後始末(第117 レーンB3)。
    #   `plan_due_step` は「最初に予約された step」の印で、**`plan_step` とセットでしか
    #   意味を持たない**。上で `plan_step` を落としながらこの印を残すと、印だけが宙に浮き、
    #   同じ実体が次に予約を受けた日に `scheduler._defer_first` が**何日も前の step** を
    #   返す = `waited` が巨大 = 初回から `max_defer_steps` 超過扱いで、LLM 計画を一度も
    #   撃たないまま骨格へ落ちる(「今日はじめて予約された人」が「何日も待たされた人」に
    #   化ける)。この関数はプール回転で再実体化された退場者へ**毎 step 冪等に貼り直される**
    #   (上の `phase` 参照)ので、印の寿命は実体の寿命より長くなりうる。
    #   ★既定(tiers OFF)では**誰もこの属性を生やさない**ので、下の分岐は必ず False =
    #     1 ビットも書かない(ゴールデン L1・checkpoint の pickle バイトとも不変)。
    #   ★内省側(`reflect_due_step`)は対象外: ここは `reflect_step` を落とさないので
    #     予約は生きたまま `_reflect_due` にドレインされる = 宙に浮かない。
    if int(getattr(agent, "plan_due_step", -1)) >= 0:
        agent.plan_due_step = -1


# =========================================================================== #
# ③ 外部からの負傷(**他レーンからの唯一の公開 API**)
# =========================================================================== #
#: 外部の出来事から渡される「見立て」を重症度へ写す表。**S1/S2 に閉じる**:
#:  倒れて意識を失う(S3/S4)は救急の引き金そのものなので、身体側の発症チャネルか
#:  H1 の進行(worsen)からしか到達させない(事件側から直接 S3 を作ると、同じ現象を
#:  2 つのレーンが書くことになる)。
_INJURY_MIN, _INJURY_MAX = S_MILD, S_MODERATE


def injury_days(agent, scfg: dict, seed: int, step: int) -> int:
    """負傷の療養日数(**乱数を 1 本も引かない**安定ハッシュの純関数)。

    ★``health_onset`` stream は 1 人日 12 本の固定長で、そこへ 1 本でも割り込むと
      既存ランの消費列がずれる。したがって外部からの負傷は**新しい乱数を引かない**
      (``frailty_of`` / ``confirm_severity`` と同じ安定ハッシュの流儀)。"""
    lo = int(scfg["trauma_days_lo"])
    hi = max(lo, int(scfg["trauma_days_hi"]))
    if hi <= lo:
        return lo
    u = (_stable_hash(f"injury/{int(seed)}/{int(agent.id)}/{int(step)}") % 10000) \
        / 10000.0
    return lo + int(u * (hi - lo + 1))


def on_injury(sim, agent, severity_hint: int, source: str = "",
              *, step: int = -1, sim_min: int = 0) -> int:
    """**事件で負傷した**個体を身体の状態機械へ載せる(他レーンからの唯一の公開 API)。

    呼び出し側(喧嘩・火災・交通)は「どのくらいの怪我か」の見立て(1=軽症 / 2=中等症)
    だけを渡す。重症度・療養日数・sick-role・記憶・L1 は**すべて本 module が決める**
    (身体の意味論は健康側に閉じる = ``city_ops`` が健康状態を 1 バイトも書かないのと同じ規約)。

    - severity OFF / 死亡 / 範囲外 / 既にもっと重い状態 のときは **何もしない**(戻り値 0)
    - 新しい乱数を 1 本も引かない(``health_onset`` の固定長 12 本を 1 バイトも動かさない)
    - LLM を 1 度も呼ばない・プロンプトの欄を 1 つも増やさない

    戻り値 = 適用後の重症度(0 = 何もしなかった)。"""
    if not severity_on(sim) or agent is None:
        return 0
    if is_dead(agent) or not _present(agent):
        return 0
    scfg = severity_cfg(sim.healthcfg)
    want = max(_INJURY_MIN, min(_INJURY_MAX, int(severity_hint)))
    if severity_of(agent) >= want:
        return severity_of(agent)                  # 既にもっと重い = 上書きしない
    seed = int(getattr(sim.hub, "master_seed", 0))
    days = injury_days(agent, scfg, seed, int(step))
    agent.severity = int(want)
    agent.sev_channel = CH_TRAUMA
    agent.sev_cared = False
    agent.sev_confirmed = -1
    agent.sev_outcome_step = -1
    agent.sev_fatal = False
    agent.sev_until = int(step) + days * max(1, int(sim.clock.steps_per_day))
    if not hasattr(agent, "sev_frailty"):
        agent.sev_frailty = frailty_of(agent, scfg, seed)
    presentee = (want == S_MILD
                 and (_stable_hash(f"injury_presentee/{seed}/{int(agent.id)}")
                      % 10000) / 10000.0 < float(scfg["presenteeism"]))
    agent.sev_presentee = bool(presentee)
    _apply_sick_role(agent, scfg, presentee)
    st = _state(sim)
    st["onsets"][CH_TRAUMA] += 1
    st["injuries"] = int(st.get("injuries", 0)) + 1
    agent.remember(ONSET_TEXT[CH_TRAUMA])
    _log(sim.logger, agent, "illness",
         _payload(agent, scfg, "onset",
                  {"days": int(days), "external": True,
                   "source": str(source or "")}),
         int(step), int(sim_min))
    return int(want)


# =========================================================================== #
# ④ プール回転の往復(**監査で見つかったバグの同梱修正**)
# =========================================================================== #
#: 退避に載せる身体の欄。★``sick`` / ``sick_until`` は**これまで 1 つも載っていなかった**ため、
#  住民がプール回転(退場 → 再来街)のたびに病気を忘れていた(監査発見)。severity 一式と塞ぐ。
#  ★レーン D1(2026-08-12)の総点検で、身体の欄なのに載っていなかった 4 つを足した:
#    - ``sev_collapse_step`` … sev_* 族で唯一の非搭載だった(救急の引き金の刻印)。
#    - ``chronic_days`` / ``withdrawn`` … 慢性高 grievance → 引きこもり(自由時間の外出抑制)。
#      **行動を変える持続状態**なのに回転のたびに「治って」いた。
#    - ``fatigue`` … 疲労ゲージ(発火閾値の乗数 fatigue_threshold_delta の入力)。
#  既定値と等しい欄はキーを作らないので、健康な個体の退避 dict は 1 バイトも変わらない。
_SLIM_FIELDS = (("sick", bool, False), ("sick_until", int, -1),
                ("severity", int, 0), ("sev_channel", str, ""),
                ("sev_until", int, -1), ("sev_outcome_step", int, -1),
                ("sev_fatal", bool, False), ("sev_presentee", bool, False),
                ("sev_cared", bool, False), ("sev_confirmed", int, -1),
                ("sev_frailty", float, 1.0), ("dead", bool, False),
                ("sev_collapse_step", int, -1), ("chronic_days", int, 0),
                ("withdrawn", bool, False), ("fatigue", float, 0.0))


def slim_state(agent) -> dict:
    """身体状態の退避辞書(**既定値しか持たない個体では空**= 退避 dict のバイト列不変)。"""
    out: dict = {}
    for name, cast, default in _SLIM_FIELDS:
        got = getattr(agent, name, default)
        if cast is bool:
            got = bool(got)
        elif cast is int:
            got = int(got)
        elif cast is float:
            got = float(got)
        else:
            got = str(got or "")
        if got != default:
            out[name] = got
    pending = getattr(agent, "sev_pending", None)
    if pending:
        out["sev_pending"] = {"day": int(pending["day"]),
                              "slot_min": int(pending["slot_min"]),
                              "ch": str(pending["ch"]), "sev": int(pending["sev"]),
                              "days": int(pending["days"]),
                              "u_fatal": float(pending["u_fatal"]),
                              "u_presentee": float(pending["u_presentee"]),
                              "u_care": float(pending["u_care"]),
                              "wbgt": (None if pending.get("wbgt") is None
                                       else float(pending["wbgt"]))}
    return out


def apply_slim(agent, state: dict) -> None:
    """退避辞書 → 再来街エージェント(**キー欠落を許容**= 旧 退避辞書からは何も生やさない)。"""
    if not state:
        return
    for name, cast, _default in _SLIM_FIELDS:
        if name in state:
            setattr(agent, name, cast(state[name]))
    pending = state.get("sev_pending")
    if pending:
        agent.sev_pending = dict(pending)
