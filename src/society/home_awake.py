"""在宅覚醒(HOME_AWAKE・β9)。既定 OFF。

正典: docs/plans/beta-implementation-plan.md §2 / docs/plans/reflection-leisure-plan.md
§5.2「家に入る = 寝る」・§4「余暇・家内行動の現実(統計)」。

現状(OFF)の構造:
  `routine.decide` の就寝分岐が `go_to_bed` を返し、`scheduler._apply` が
  **入館と就寝を同じ 1 step の中で連続実行する**(`kind = "sleep"` に落とす)。
  実測で `enter_building{home:true}` → `sleep_start` の分数は n=1,512 の **100.0% が 0 分**、
  つまり「帰宅してから寝るまで」という時間帯そのものが世界に存在しない。
  現実は帰宅 19:15 / 就寝 23:39 = **4 時間 24 分**(社会生活基本調査 令和3年・東京・有業者)。

本モジュール(ON)が足すもの — **2 つだけ**:
  ① 就寝ハザード: 帰宅しても即座には寝ない。各 step で
       p_sleep = sigmoid(b0 + b1*概日 + b2*疲労 + b3*翌日早出 − b4*在宅活動に没入中)
     を引いて就寝を抽選する。概日の中心は**既存の個体差 `bedtime_min`**(新しい生理状態を
     1 つも足さない)。係数 b0..b4 は conf(`daily.home_awake.hazard`)= 量は場所の知識。
  ② 在宅活動ラベル 8 種(meal/bath/housework/family_talk/media/hobby/study/rest)の
     ルールベース抽選。年齢帯 × 就業有無 × 時刻帯の重み表(§4 の統計水準)から選ぶ。

**LLM 呼はこのモジュールから 1 本も出ない**(ルールベースのみ)。さらに在宅覚醒中の個体は
`muted()` により**睡眠中と同じ扱い**にして、
  (a) 欲求発火 `_phase_drive` の active リスト、
  (b) 発話の返答権の付与(`_apply` の speak → `partner._reply_to`)
の 2 箇所から外す = 「起きている時間が伸びたぶんだけ呼数が増える」という副作用を
構造的に断つ(R1: この機能は LLM を 1 呼も足さない)。
★正直な代償: この 2 箇所を閉じると「同居人どうしが夜の自宅で話す」発話は成立しない。
  現行 OFF ではその時間帯に両者とも就寝しているので**失うものはゼロ**だが、
  開けるかどうかは呼数と引き換えの判断であって、conf ではなくコードの 1 行に置いてある。
★閉じていないもの(意図的): RFX-A の内省的瞬間(`reflect_timing.arm_moments`)は
  `sim.agents` を直接走査し、かつ「予約の満期」方式で**総呼数が保存**されるので、
  在宅覚醒はそのまま `_low_load_context` の "home" 文脈として効く(= 内省の発火文脈が
  歩行中心から在宅中心へ移るという本レーンの狙いは、呼数を 1 本も足さずに実現する)。

乱数は新 stream **"home_awake"** だけ(既存 stream の draw 順に一切触らない)。
OFF では `enabled()` が False を返した時点で draw もイベントも 0 = ゴールデン L1 バイト一致。

no-fingerprint: 本モジュールは traits/因子を 1 つも読まない(年齢・職業・世帯の
人口統計属性だけ)。活動ラベルはプロンプトに機構語として現れない(`_ACTIVITY_JP` に
無い値なので `build_prompt` は「いま:」行を 1 行も足さない)。
"""
from __future__ import annotations

import hashlib
import math

from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# 既定値(conf.daily.home_awake で上書き可)。既定 OFF = 挙動は完全に旧来どおり。
# --------------------------------------------------------------------------- #
_DEFAULTS: dict = {
    "enabled": False,
    # 帰宅トリガの前倒し[分]。0 = 既存の帰宅トリガ(bedtime_reached)のまま = スコープ最小。
    # >0 にすると就寝時刻の lead_min 前から帰路につく(= 帰宅 19:15 / 就寝 23:39 の
    # 現実へ寄せる LSR-H 相当。既定 0 なので ON にしても帰宅時刻は動かない)。
    # ★lead.mode="per_agent" ではこの一律値は使わず、個体別の前倒しを組む(下記 "lead")。
    "lead_min": 0,
    # ---- 帰宅前倒しの個体分布(ユーザー決定 2026-08-16)----------------------- #
    "lead": {
        "mode": "fixed",        # fixed(既定=lead_min の一律値) | per_agent
        "jitter_min": 30,       # 日次ジッター ±[分](既存 stream "home_awake")
        # 就業状態 → 前倒しの基準[分]。詳細な出典は _SEG_SOURCE(下)に書く。
        "segment_base": {"worker": 195, "student": 225,
                         "non_working": 255, "night_worker": 90},
        # 年齢帯 → 基準への加算[分]。★uncalibrated parameter(下記 _SEG_SOURCE 参照)。
        "age_delta": {"youth": 0, "mid": -25, "senior": 10, "elder": 40},
        # 個体差[分]の分位テーブル(blake2b の分位で引く = 乱数 stream を 1 本も消費しない)。
        # 右裾の長い形(少数が大きく早く帰る)。★合計 0 = セグメント基準が母平均になる。
        # ★uncalibrated parameter(分布の形の公表値は無い)。
        "spread_quantiles": [-78, -56, -40, -26, -13, 1, 16, 34, 60, 102],
    },
    # ---- 同居人どうしの夜の自宅会話(ユーザー決定 2026-08-16)。既定 OFF ------- #
    "evening_talk": {
        "enabled": False,       # ON で「同一世帯 かつ 両者 HOME_AWAKE」のペアだけ発話を開く
    },
    # 帰宅後に起きていられる上限[分]。到達したら必ず就寝する(ハザードの裾を切る安全弁)。
    "max_awake_min": 200,
    # 「翌日早出」の判定: 勤務開始がこの分 of day より早ければ早出(既定 480 = 08:00)。
    "early_start_min": 480,
    # ハザード係数(behavior_params)。符号は式のとおり(b4 だけ引く)。
    #  ★切片だけは **確率** `p0` として持つ: 切片は「1 step あたり」の量なので Δt を
    #    変えたら変換が要る唯一の係数で、確率で持てば timeconv の PROB(べき変換
    #    p' = 1−(1−p)^(Δt/10))がそのまま効く。式に入れるときに logit(p0) へ直す。
    #    b1..b4 は対数オッズの**寄与**(= ハザード比の対数)なので Δt 非依存。
    "hazard": {
        "p0": 0.032,  # 切片: 就寝時刻ちょうど・他項ゼロのときの 1 step あたり就寝確率
        "b1": 0.95,   # 概日: 就寝時刻からの経過[時間]あたりの対数オッズ
        "b2": 1.20,   # 疲労(既存 agent.fatigue 0..1。health OFF では常に 0 = 無効)
        "b3": 0.80,   # 翌日早出(0/1)
        "b4": 1.00,   # 在宅活動に没入中(0/1)= 就寝を**遅らせる**(式では引く)
    },
    # 没入とみなす在宅活動(b4 の対象)。
    "engaged_acts": ["media", "hobby", "study", "family_talk"],
    # 世帯シナジー(最小): 同居人も在宅覚醒中なら family_talk の重みを何倍にするか。
    "family_talk_boost": 3.0,
}

# 在宅活動ラベル(8 種)。順序は決定論(抽選の正準順序)。
ACTS: tuple[str, ...] = ("meal", "bath", "housework", "family_talk",
                         "media", "hobby", "study", "rest")

# 帰宅前倒しの決め方。fixed = 全体一律(既定・後方互換)。per_agent = 個体別。
LEAD_MODES: tuple[str, ...] = ("fixed", "per_agent")

# --------------------------------------------------------------------------- #
# 活動の 1 セッション長[step](正準 Δt=10 分基準)。
#   出典: 社会生活基本調査 令和3年(総務省統計局)・平均時刻編 —
#     夕食開始 19:17(東京)で食事は 1 日計 96 分 → 夕食 1 回 30〜50 分。
#     身の回りの用事(入浴を含む)1 日計 83〜84 分 → 1 回 20〜40 分。
#     休養・くつろぎ 117 分/日(独身期35歳未満 143 分)= 在宅の既定状態で長め。
#     趣味・娯楽は「まれだが長い」(行動者率 25.2% × 行動者平均 189 分)ので上限を長く取る。
#     テレビ等 128 分/日 + 自宅モバイルネット 89 分(総務省 IICP 令和7年度)。
# --------------------------------------------------------------------------- #
_SESSION_STEPS: dict[str, tuple[int, int]] = {
    "meal":        (3, 5),
    "bath":        (2, 4),
    "housework":   (2, 4),
    "family_talk": (2, 4),
    "media":       (3, 8),
    "hobby":       (3, 9),
    "study":       (3, 6),
    "rest":        (2, 6),
}

# --------------------------------------------------------------------------- #
# 時刻帯 → 在宅活動の基礎重み。
#   出典: 社会生活基本調査 令和3年 表1-1〜1-4(全国・10歳以上・週全体・分/日)
#     1 次: 睡眠 474 / 身の回りの用事 84 / 食事 99
#     2 次: 家事 87 / 買い物 26
#     3 次: テレビ等 128 / 休養・くつろぎ 117 / 趣味・娯楽 48 /
#           学習・自己啓発 13 / 交際・付き合い 10
#   + 平均時刻編: 夕食開始 19:17(東京)・就寝 23:39(東京)。
#   → 夕方帯は食事が主役、夜帯は入浴のあと 3 次活動(メディア・休養)へ移る、
#     朝帯は朝食と身支度、日中帯は家事・休養、という形に落とす。
#   重みは相対値(合計を 1 にする必要はない。抽選時に正規化する)。
# --------------------------------------------------------------------------- #
_BAND_WEIGHTS: dict[str, dict[str, float]] = {
    # 朝(05-11 時): 起床後。朝食+身支度が中核(食事 99 分 / 身の回り 84 分)。
    "morning": {"meal": 0.30, "bath": 0.24, "housework": 0.16, "family_talk": 0.04,
                "media": 0.10, "hobby": 0.04, "study": 0.02, "rest": 0.10},
    # 日中(11-17 時): 在宅している側(無業・休日・在宅勤務)の帯。家事と休養。
    "day":     {"meal": 0.14, "bath": 0.04, "housework": 0.24, "family_talk": 0.05,
                "media": 0.17, "hobby": 0.10, "study": 0.06, "rest": 0.20},
    # 夕方(17-22 時): 帰宅直後。夕食開始 19:17 = ほぼ即座に食事が始まる。
    "evening": {"meal": 0.28, "bath": 0.16, "housework": 0.14, "family_talk": 0.07,
                "media": 0.16, "hobby": 0.06, "study": 0.03, "rest": 0.10},
    # 夜(22-05 時): 入浴を済ませたあとの 3 次活動。休養・くつろぎが既定状態。
    "night":   {"meal": 0.06, "bath": 0.16, "housework": 0.08, "family_talk": 0.06,
                "media": 0.24, "hobby": 0.10, "study": 0.04, "rest": 0.26},
}

# --------------------------------------------------------------------------- #
# 年齢帯の補正(乗数)。
#   出典: 社会生活基本調査 令和3年 表5-1(ライフステージ別)+ 年齢別の年齢勾配。
#     独身期35歳未満: 家事 21 分(全国 87 の 0.24 倍)/ テレビ等 50 分(128 の 0.39 倍)/
#       休養・くつろぎ 143 分(117 の 1.22 倍)/ 趣味・娯楽 90 分(48 の 1.88 倍)/
#       交際・付き合い 17 分(10 の 1.7 倍)。
#     テレビは 15-19 歳 26 分 → 85 歳以上 262 分の 10 倍勾配。
#     趣味・娯楽は 20-24 歳 78 分がピーク、40 代 26 分で底、退職後に回復。
#   ※ media ラベルは「テレビ+自宅ネット」の合算なので、若年の TV 離れ(0.39 倍)を
#     そのまま当てず、自宅モバイルネット(20 代のネット 266.9 分/日・IICP 令和7年度)で
#     相殺した 1.0 前後に置く。
# --------------------------------------------------------------------------- #
_AGE_MULT: dict[str, dict[str, float]] = {
    "youth":  {"housework": 0.25, "media": 1.00, "rest": 1.20, "hobby": 1.90,
               "family_talk": 0.60, "study": 1.60},
    "mid":    {"housework": 1.10, "media": 1.00, "rest": 1.00, "hobby": 0.70,
               "family_talk": 1.20, "study": 0.60},
    "senior": {"housework": 1.20, "media": 1.20, "rest": 1.05, "hobby": 0.90,
               "family_talk": 1.10, "study": 0.50},
    "elder":  {"housework": 1.10, "media": 1.90, "rest": 1.25, "hobby": 1.10,
               "family_talk": 1.00, "study": 0.40},
}

# 就業有無の補正(乗数)。
#   出典: 表5-1 — 有業者は仕事 332 分(独身期35歳未満)を抱えるぶん 3 次活動が短く、
#   学生は学習・自己啓発が桁で長い。無業(退職・無職)は 3 次活動が最長。
_WORK_MULT: dict[str, dict[str, float]] = {
    "worker":  {"study": 0.50, "housework": 0.90, "hobby": 0.90, "rest": 1.00},
    "student": {"study": 4.00, "housework": 0.60, "hobby": 1.20, "rest": 0.95},
    "idle":    {"study": 0.80, "housework": 1.30, "hobby": 1.20, "rest": 1.15},
}

# 学生とみなす職業語(名簿の職業語彙。地名は含めない)。
_STUDENT_OCCS = ("大学生", "学生", "高校生", "中学生", "小学生", "専門学校生", "院生")

# --------------------------------------------------------------------------- #
# 帰宅前倒し(lead)の出典と、どこまでが較正済みでどこからが未較正か
# --------------------------------------------------------------------------- #
_SEG_SOURCE = """
segment_base / age_delta / spread_quantiles の来歴(捏造しないための明示):

★較正済み(直接アンカーが在る)= worker
  data/ground_truth/registry.yaml の `tu_home_to_sleep_gap_min` = **264 分**
  (社会生活基本調査 令和3年・東京都・有業者・平日: 帰宅 19:15 → 就寝 23:39)。
  本シムの就寝は「帰宅 = bedtime_min − lead」から**ハザードの期待遅延 D** だけ後ろに
  出る。既定係数での D は 60 体 x 3 日 mock の実測で **中央値 100 分 / 平均 100 分**
  (lead=0 のときの帰宅→就寝 gap がそのまま D)。第一近似は
      base_worker ≈ 264 − D = 165 分
  だが、前倒しを入れると**就寝時刻より前の窓でもハザードが小さく効く**ので実測 gap は
  lead + D より短くなる(worker セグメント実測 **222.2 分** @ base=165)。そこで
  **較正を 1 回だけ回した**: 局所感度 Δgap/Δlead ≈ 0.81(= (222.2−100)/150)から
      base_worker = 165 + (264 − 222.2)/0.81 ≈ 195 分
  として全セグメントへ同じ +30 を平行移動した(相対順序は設計どおり保つ)。
  ★これは「264 という公表値」と「本シムの実測」だけから引いた値で、新しい数字を
    発明していない(derived + 1 回の較正反復)。60 体 mock での較正なので、
    本選規模での再確認は V1(reality_score)レーンの仕事。

★uncalibrated parameter(公表値が無い。方向だけ文献に接地し、値は仮置き)
  - student(195): 社基調に**学生の帰宅時刻**の公表値は無い。通学の終わりは退勤より
    早く、学習・自己啓発が長い(9.1% × 長時間)ことから worker より長めに置いた仮置き。
  - non_working(225): 通勤・通学の行動者率は 39.7% しかない(= 6 割は「帰宅」という
    事象を持たない)。3 次活動は退職世帯で最長(表5-1)。よって最も長く在宅と置いたが、
    「何分前に家に居るか」の公表値は無い。
  - night_worker(60): 夜勤は位相が反転しており、夕方に出勤する。前倒しは小さいはず、
    という定性のみ。公表値なし。
  - age_delta: **社基調の公表表に年齢帯別の帰宅時刻は無い**(帰宅時刻は有業者平均のみ)。
    ここでは 3 次活動(自由行動)のライフステージ差 —— 独身期35歳未満 363 分 /
    全国総数 376 分 / 末子就学前の子育て世帯 202 分(表5-1)—— を
    「自由時間が短い帯は帰宅が遅い」の**代理**として使った。代理仮定そのものが未検証。
    ★年齢帯別の就寝時刻(20-24 歳 24:10 等)は **bedtime_min 側(AGE-B の U 字)が
      既に担っている**ので、ここで二重に効かせない(age_delta は帰宅側だけの小さな差)。
  - spread_quantiles: 個体差の分布形の公表値は無い。右裾の長い形(少数が大きく早く
    帰る = 短時間勤務・在宅勤務)を仮置きし、合計 0 = セグメント基準を母平均に保った。
"""


# --------------------------------------------------------------------------- #
# conf
# --------------------------------------------------------------------------- #
def cfg_of(cfg) -> dict:
    """conf(OmegaConf)→ 実行時の home_awake 設定 dict。

    `daily.home_awake` ブロックが無ければ既定(OFF)= 挙動バイト一致。"""
    raw = OmegaConf.select(cfg, "daily.home_awake")
    raw = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    raw = raw or {}
    out = dict(_DEFAULTS)
    out["hazard"] = dict(_DEFAULTS["hazard"])
    out["engaged_acts"] = list(_DEFAULTS["engaged_acts"])
    out["lead"] = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list)
                       else v) for k, v in _DEFAULTS["lead"].items()}
    out["evening_talk"] = dict(_DEFAULTS["evening_talk"])
    for k, v in raw.items():
        if k == "hazard" and isinstance(v, dict):
            for hk, hv in v.items():
                if hk in out["hazard"]:
                    out["hazard"][hk] = float(hv)
        elif k == "lead" and isinstance(v, dict):
            for lk, lv in v.items():
                if lk not in out["lead"]:
                    continue
                if isinstance(out["lead"][lk], dict) and isinstance(lv, dict):
                    out["lead"][lk] = {**out["lead"][lk], **lv}
                else:
                    out["lead"][lk] = lv
        elif k == "evening_talk" and isinstance(v, dict):
            for tk, tv in v.items():
                if tk in out["evening_talk"]:
                    out["evening_talk"][tk] = tv
        elif k in out and k not in ("hazard", "lead", "evening_talk"):
            out[k] = v
    out["enabled"] = bool(out["enabled"])
    out["lead_min"] = max(0, int(out["lead_min"]))
    out["max_awake_min"] = max(0, int(out["max_awake_min"]))
    out["early_start_min"] = int(out["early_start_min"])
    out["engaged_acts"] = tuple(str(a) for a in out["engaged_acts"])
    out["family_talk_boost"] = float(out["family_talk_boost"])
    lead = out["lead"]
    lead["mode"] = str(lead["mode"]).strip().lower()
    if lead["mode"] not in LEAD_MODES:
        raise ValueError(f"daily.home_awake.lead.mode='{lead['mode']}' は未知"
                         f"(有効値: {', '.join(LEAD_MODES)})。既定 fixed = 一律 lead_min。")
    lead["jitter_min"] = max(0, int(lead["jitter_min"]))
    lead["segment_base"] = {str(k): int(v) for k, v in lead["segment_base"].items()}
    lead["age_delta"] = {str(k): int(v) for k, v in lead["age_delta"].items()}
    lead["spread_quantiles"] = tuple(int(x) for x in lead["spread_quantiles"]) \
        or (0,)
    out["evening_talk"]["enabled"] = bool(out["evening_talk"]["enabled"])
    return out


def settings(sim) -> dict:
    """sim へ一度だけキャッシュした home_awake 設定(以後は再解析しない。media と同型)。"""
    hcfg = getattr(sim, "_home_awake_settings", None)
    if hcfg is None:
        hcfg = cfg_of(sim.cfg)
        sim._home_awake_settings = hcfg
    return hcfg


# --------------------------------------------------------------------------- #
# プロファイル(人口統計属性のみ。traits/因子は読まない = R9)
# --------------------------------------------------------------------------- #
def _age_band(age: int) -> str:
    if age < 30:
        return "youth"
    if age < 50:
        return "mid"
    if age < 65:
        return "senior"
    return "elder"


def _work_band(agent) -> str:
    if str(getattr(agent, "occupation", "")) in _STUDENT_OCCS:
        return "student"
    if int(getattr(agent, "work_start_min", -1)) >= 0:
        return "worker"
    return "idle"


def profile_for(agent) -> dict:
    """age/occupation から活動重みの個人プロファイルを導く(乱数を引かない・純関数)。

    ★media.profile_for と違い **agent へキャッシュしない**: 年齢は AGE-F の誕生日で、
      就業は career(離職・転職)で**ラン中に変わる**ので、キャッシュすると古い帯に
      貼りついたまま更新されない。呼ばれるのは活動セッションの開始時だけ
      (1 個体あたり 1 晩に 2〜3 回)なので、毎回引き直しても費用は無視できる。"""
    return {"age_band": _age_band(int(getattr(agent, "age", 40))),
            "work_band": _work_band(agent)}


def time_band(sim_min: int) -> str:
    """時刻 → 在宅活動の時刻帯(morning / day / evening / night)。"""
    h = (sim_min % 1440) // 60
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "day"
    if 17 <= h < 22:
        return "evening"
    return "night"


def act_weights(agent, sim_min: int, hcfg: dict, *, mates_awake: bool) -> dict[str, float]:
    """年齢帯 × 就業有無 × 時刻帯 の重み表(+ 世帯シナジー)。決定論・乱数ゼロ。"""
    prof = profile_for(agent)
    base = _BAND_WEIGHTS[time_band(sim_min)]
    age_m = _AGE_MULT[prof["age_band"]]
    work_m = _WORK_MULT[prof["work_band"]]
    out: dict[str, float] = {}
    for act in ACTS:
        w = float(base[act]) * float(age_m.get(act, 1.0)) * float(work_m.get(act, 1.0))
        out[act] = w
    # 世帯シナジー(最小): 同居人も在宅覚醒中なら団らんの重みを上げる。
    # 共同意思決定・家計連動はやらない(本選後)。
    if mates_awake:
        out["family_talk"] *= float(hcfg["family_talk_boost"])
    else:
        out["family_talk"] = 0.0        # 独居 / 同居人が寝ている = 団らんは起きない
    return out


def pick_act(weights: dict[str, float], rng) -> str:
    """重み表から在宅活動ラベルを 1 つ抽選(決定論的な正準順序で累積)。"""
    total = sum(max(0.0, weights.get(a, 0.0)) for a in ACTS)
    if total <= 0.0:
        return "rest"
    r = float(rng.random()) * total
    acc = 0.0
    for act in ACTS:
        acc += max(0.0, weights.get(act, 0.0))
        if r < acc:
            return act
    return "rest"


def session_steps(act: str, rng, clock=None) -> int:
    """在宅活動 1 セッションの長さ[step]。Δt 非依存(clock.dur_steps で読み替える)。"""
    lo, hi = _SESSION_STEPS.get(act, (2, 4))
    n = int(rng.integers(lo, hi + 1))
    return int(clock.dur_steps(n)) if clock is not None else n


# --------------------------------------------------------------------------- #
# 就寝ハザード
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 帰宅前倒し(lead)= 個体分布(ユーザー決定 2026-08-16)
#   lead_i = segment_base[就業状態] + age_delta[年齢帯] + 個体差(blake2b の分位)
#            + 日次ジッター(stream "home_awake")
#   ★個体差は **乱数 stream を 1 本も消費しない**(blake2b の安定ハッシュ。economy.stable_unit /
#     aging の誕生日と同流儀 = 別ラン・resume でも同じ人が同じ位置)。
# --------------------------------------------------------------------------- #
def _stable_unit(salt: str, key: str) -> float:
    """(用途 salt, 安定キー)→ [0,1)。乱数 stream 無風・プロセス跨ぎ安定。"""
    h = hashlib.blake2b(f"{salt}\x1f{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def is_night_worker(agent) -> bool:
    """夜勤か。既存の勤務時刻だけで判定する(日跨ぎ勤務 or 夕方以降の始業)。"""
    start = int(getattr(agent, "work_start_min", -1))
    if start < 0:
        return False
    end = int(getattr(agent, "work_end_min", 0))
    return end < start or start >= 18 * 60


def lead_segment(agent) -> str:
    """就業状態のセグメント(segment_base のキー)。夜勤は就業状態より優先する。"""
    if is_night_worker(agent):
        return "night_worker"
    band = _work_band(agent)
    return {"worker": "worker", "student": "student"}.get(band, "non_working")


def base_lead_min(agent, hcfg: dict) -> int:
    """個体固定の帰宅前倒し[分](日次ジッター前)。決定論・乱数 stream ゼロ。"""
    lead = hcfg["lead"]
    seg = lead_segment(agent)
    base = int(lead["segment_base"].get(seg, 0))
    base += int(lead["age_delta"].get(_age_band(int(getattr(agent, "age", 40))), 0))
    q = lead["spread_quantiles"]
    idx = min(len(q) - 1,
              int(_stable_unit("home_awake_lead", str(int(getattr(agent, "id", 0))))
                  * len(q)))
    return max(0, base + int(q[idx]))


def lead_min_for(agent, sim, sim_min: int, hcfg: dict) -> int:
    """今日の帰宅前倒し[分]。mode=fixed は一律 lead_min(= 既定 0 で従来どおり)。

    per_agent では 1 日 1 回だけ決めて agent へ持たせる(= 毎 step 引き直さない・
    checkpoint には agent の属性として自然に載る)。ジッターの stream キーは
    ("home_awake", agent.id, "lead", day) で、毎 step のハザード draw と衝突しない。"""
    lead = hcfg["lead"]
    if lead["mode"] != "per_agent":
        return int(hcfg["lead_min"])
    day = int(sim_min) // 1440
    if int(getattr(agent, "_home_lead_day", -1)) == day:
        return int(getattr(agent, "_home_lead_min", 0))
    minutes = base_lead_min(agent, hcfg)
    jit = int(lead["jitter_min"])
    if jit > 0:
        rng = sim.hub.stream("home_awake", int(agent.id), "lead", day)
        minutes += int(rng.integers(-jit, jit + 1))
    minutes = max(0, minutes)
    agent._home_lead_day = day
    agent._home_lead_min = minutes
    return minutes


# --------------------------------------------------------------------------- #
# LLM 経路のゲート(muted / evening_talk)
# --------------------------------------------------------------------------- #
def home_awake_now(agent) -> bool:
    """いま自宅の中で覚醒中か。

    OFF では `_home_awake_since` が誰にも生えないので **常に False**(getattr 既定 -1)。"""
    if int(getattr(agent, "_home_awake_since", -1)) < 0:
        return False
    home = getattr(agent, "home_building", "")
    return bool(home) and getattr(agent, "building", None) == home


def _housemate_ids(agent) -> frozenset[int]:
    return frozenset(int(o) for o in (getattr(agent, "housemates", None) or ()))


def has_awake_housemate(agent, sim) -> bool:
    """同一世帯の誰かが「同じ自宅の中で在宅覚醒中」か(在場述語 present_agent 経由)。"""
    for oid in _housemate_ids(agent):
        other = sim.present_agent(oid)
        if other is None or int(other.id) == int(agent.id):
            continue
        if home_awake_now(other) and not other.sleeping:
            return True
    return False


def talk_enabled(sim) -> bool:
    hcfg = settings(sim)
    return bool(hcfg["enabled"] and hcfg["evening_talk"]["enabled"])


def muted(agent, sim) -> bool:
    """在宅覚醒中で、かつ evening_talk の条件を満たさない = 睡眠中と同じ扱い。

    OFF では `home_awake_now` が常に False なので **常に False** = 既存経路と 1 ビットも
    変わらない。evening_talk ON かつ**在宅覚醒中の同居人が居る**個体だけは、夜の自宅の
    団らんのために発火の対象へ戻す(相手が同居人であることは `pair_open` が別途固定する)。"""
    if not home_awake_now(agent):
        return False
    if not talk_enabled(sim):
        return True
    return not has_awake_housemate(agent, sim)


def pair_open(speaker, partner, sim) -> bool:
    """speaker → partner に返答権を渡してよいか(在宅覚醒がからむときの追加条件)。

    partner が在宅覚醒中でなければ常に True = 従来経路。在宅覚醒中なら
    **evening_talk ON かつ「同一世帯 かつ 両者 HOME_AWAKE」**のときだけ開く
    (= 同居人以外・来客・外部への経路は閉じたまま)。"""
    if not home_awake_now(partner):
        return True
    if not talk_enabled(sim):
        return False
    return home_awake_now(speaker) and int(speaker.id) in _housemate_ids(partner)


def reply_open(agent, sim) -> bool:
    """保留中の返答を **いま** 消費してよいか。

    在宅覚醒中でなければ従来どおり True。在宅覚醒中は、その返答をくれた相手が
    「在宅覚醒中の同居人」であるときだけ開く(路上で受け取った返答権は預かったまま
    起床後へ持ち越す = 返事を落とさない)。"""
    if not home_awake_now(agent):
        return True
    rt = getattr(agent, "_reply_to", None)
    if rt is None:
        return False
    speaker = sim.present_agent(int(rt[0]))
    return speaker is not None and pair_open(speaker, agent, sim)


def hours_since_bedtime(agent, sim_min: int) -> float:
    """就寝時刻(個体差 bedtime_min)からの符号付き経過[時間]。円環 ±12 時間で表す。"""
    d = (int(sim_min) - int(getattr(agent, "bedtime_min", 1320))) % 1440
    if d >= 720:
        d -= 1440
    return d / 60.0


def early_tomorrow(agent, sim, sim_min: int, hcfg: dict) -> bool:
    """翌日が早出か。既存の勤務開始時刻(+ 暦があれば翌日が勤務日か)だけから導く。

    ★第144 `respect_work_days`(既定 false = 下の分岐が旧コードと 1 バイト同一): 暦ゲートは
      `routine.in_work_window` / `scheduler._wage_worked_today` と**同一の式**で判定する
      (「勤務日か」を答える窓口が 3 つあるので、3 つとも同じ規則で答えなければならない)。
    """
    start = int(getattr(agent, "work_start_min", -1))
    if start < 0 or start >= int(hcfg["early_start_min"]):
        return False
    cal = getattr(sim, "calendarcfg", None)
    if cal is not None and cal.get("enabled") and cal.get("weekday_work"):
        from .world import calendar as _calendar
        spec = str(getattr(agent, "work_dow", "") or "") if cal.get("respect_work_days") else ""
        tomorrow = int(sim_min) + 1440
        if spec:
            if not _calendar.days_match(spec, _calendar.weekday_of(cal, tomorrow)):
                return False
        elif not _calendar.is_workday(cal, tomorrow):
            return False
    return True


def _logit(p: float) -> float:
    """確率 → 対数オッズ。0/1 は有限な巨大値へ丸める(ハザードの実質 OFF / 即発火)。"""
    p = float(p)
    if p <= 0.0:
        return -60.0
    if p >= 1.0:
        return 60.0
    return math.log(p / (1.0 - p))


def sleep_prob(agent, sim, sim_min: int, hcfg: dict, *, engaged: bool) -> float:
    """1 step あたりの就寝確率 p_sleep。

        p = sigmoid(logit(p0) + b1*概日 + b2*疲労 + b3*翌日早出 − b4*在宅活動に没入中)

    全て**既存の state** から組む(新しい生理状態を 1 つも足さない):
      概日     = 個体の bedtime_min からの経過[時間](= ハザードの中心が個体差)
      疲労     = 既存 agent.fatigue(health OFF では 0 = この項が無効)
      翌日早出 = 既存 work_start_min(+ 暦)
      没入     = いま選んでいる在宅活動が engaged_acts に入るか
    """
    b = hcfg["hazard"]
    z = (_logit(b["p0"])
         + float(b["b1"]) * hours_since_bedtime(agent, sim_min)
         + float(b["b2"]) * float(getattr(agent, "fatigue", 0.0) or 0.0)
         + float(b["b3"]) * (1.0 if early_tomorrow(agent, sim, sim_min, hcfg) else 0.0)
         - float(b["b4"]) * (1.0 if engaged else 0.0))
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
