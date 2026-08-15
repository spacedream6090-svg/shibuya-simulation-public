"""因子レジストリ(OPEN#1 決定: D5 最小セット、registry で拡張可)。

原則(design §11): engine はここの因子名を一切名指ししない。
因子は DATA であり、測定は observer が事後に行う(no-fingerprint)。
"""
from __future__ import annotations

import math

import numpy as np

# trait: 初期固定(生まれつき側)。値域 [0,1]。
TRAITS: dict[str, str] = {
    "nfc":             "Need for Cognition(思考頻度の傾向)",
    "risk_tolerance":  "リスク許容",
    "internal_locus":  "内的統制(結果は自分次第という信念)",
}

# state: 経験で動く(創発側)。初期値。OPEN#2: 更新則は seam(update.py)。
STATE_INIT: dict[str, float] = {
    "efficacy":  0.5,   # 自己効力感
    "grievance": 0.1,   # 不満
    "ownership": 0.1,   # 当事者意識
}


def drive_params(traits: dict[str, float],
                 rng: np.random.Generator,
                 threshold_dist: str = "normal") -> tuple[float, float]:
    """traits → 欲求発火の個体差 (drive_threshold, fire_weight)。

    因子名を知ってよいのは factors/ だけ(no-fingerprint)。cognition/engine は
    戻り値のパラメータしか見ない。ヒューリスティック v1(RQ1 検収後に調整):
    - NFC(思考頻度の傾向)が高い → 閾値が低い(考え事が起動しやすい)
    - 内的統制が高い → 申請が通りやすい(自分の思考を行動に移す重み)

    threshold_dist("normal"既定 | "lognormal"): 閾値の分布形の seam(B段)。
    lognormal はモーメントマッチング(μ,σ を平均 m・標準偏差 s から算出)で
    現行 normal と平均・散らばりを一致させる。乱数の消費本数は両者で同一
    (閾値=1 draw)なので既定(normal)の再現性は不変。
    """
    nfc = traits.get("nfc", 0.5)
    locus = traits.get("internal_locus", 0.5)
    mean_t, std_t = 0.60 - 0.20 * (nfc - 0.5) * 2, 0.08
    if threshold_dist == "lognormal":
        # moment matching: mean=mean_t, std=std_t の対数正規に一致させる
        sigma2 = math.log(1.0 + (std_t * std_t) / (mean_t * mean_t))
        mu = math.log(mean_t) - 0.5 * sigma2
        threshold = float(np.clip(math.exp(rng.normal(mu, math.sqrt(sigma2))),
                                  0.30, 0.85))
    else:
        threshold = float(np.clip(rng.normal(mean_t, std_t), 0.30, 0.85))
    fire_weight = float(np.clip(rng.normal(0.50 + 0.30 * (locus - 0.5) * 2, 0.12),
                                0.15, 0.90))
    return threshold, fire_weight


def flat_traits(names=None, value: float = 0.5) -> dict[str, float]:
    """全 trait を同一の定数にした traits(第74バッチ IDEA④ の初期個体差ゼロ対照)。

    キー集合は `sample_traits` と同じ(sorted(TRAITS))ので、下流の写像関数は
    ひとつも欠損キーを見ない。乱数を引かない = 呼び出し位置に関係なく draw 順不変。
    """
    keys = sorted(TRAITS) if names is None else sorted(names)
    v = float(value)
    return {k: v for k in keys}


def flat_drive_params(traits: dict[str, float]) -> tuple[float, float]:
    """`drive_params` の**乱数抜き中心値**(= 個体差ゼロ版の (threshold, fire_weight))。

    drive_params は traits から分布の中心を決め、そこへ 2 draw のノイズを載せる。
    初期個体差ゼロ対照ではノイズも「生まれつきのばらつき」なので中心値へ潰す。
    threshold_dist は normal / lognormal ともモーメントマッチングで平均が一致する設計
    (drive_params の docstring)なので、中心値はどちらでも同じ = 引数に取らない。
    **乱数を 1 draw も引かない**ので、呼び出し側は従来どおり drive_params を呼んで
    draw を消費したうえで値だけ差し替えれば、ON/OFF で draw 順が完全に一致する。
    """
    nfc = traits.get("nfc", 0.5)
    locus = traits.get("internal_locus", 0.5)
    threshold = float(np.clip(0.60 - 0.20 * (nfc - 0.5) * 2, 0.30, 0.85))
    fire_weight = float(np.clip(0.50 + 0.30 * (locus - 0.5) * 2, 0.15, 0.90))
    return threshold, fire_weight


def drift_params(traits: dict[str, float], rng: np.random.Generator,
                 cfg: dict) -> dict[str, float]:
    """traits → 内省しやすさの時間ドリフト率(E2)。drive_params と同格(no-fingerprint)。

    因子名を知ってよいのは factors/ だけ。cognition/drive は返り値の数値しか見ない。
    設計(docs/research/reflection-drift.md §2.3)の写像:
    - NFC(思考頻度の傾向)が高い → **鋭敏化型(sensitizer)**: 使うほど内省が加速し閾値↓
      (sens 強め・馴化弱め)。
    - NFC 低 / 内的統制低 → **馴化型(habituator)**: 刺激に慣れて内省が減る(馴化強め)。
    direction_bias>0 = sensitizer 寄り(sens↑・habit↓)。cfg は conf.drive.drift の基準率。
    速さの散らばりは LogNormal(σ/μ≈0.3)を **1 draw**。build_agent 末尾で引くので
    既存ストリームの draw 位置を動かさない(既定 OFF では一切引かない=再現性維持)。
    """
    nfc = traits.get("nfc", 0.5)
    locus = traits.get("internal_locus", 0.5)
    db = float(np.clip(0.6 * (nfc - 0.5) * 2.0 + 0.2 * (locus - 0.5) * 2.0,
                       -0.6, 0.6))
    sigma = 0.3
    speed = float(np.clip(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma),
                          0.3, 3.0))         # 中心 1・σ/μ≈0.3(モーメント調整)
    base_h = float(cfg.get("habit_rate", 0.02))
    base_s = float(cfg.get("sens_rate", 0.02))
    base_r = float(cfg.get("recovery_rate", 0.01))
    return {
        "direction_bias": round(db, 4),
        "habit_rate": round(base_h * (1.0 - db) * speed, 6),
        "sens_rate": round(base_s * (1.0 + db) * speed, 6),
        "recovery_rate": round(base_r, 6),
    }


def reflect_trigger_params(traits: dict[str, float], rng: np.random.Generator,
                           cfg: dict) -> dict:
    """traits → 出来事誘発の深い内省(第12バッチ)の個人パラメータ。drift_params と同格。

    根拠(docs/research/deep-reflection-triggers.md §3: 特性反芻・出来事中心性の個人差):
    - NFC 高 → 閾値低め(熟慮に入りやすい)。
    - 内的統制 低 → ネガ重み増(出来事に揺さぶられやすい=外的統制)。
    - ばらつきは LogNormal を **1 draw**(build_agent 末尾=既存 draw 位置を動かさない。
      既定 OFF では一切引かない=再現性維持)。遅い個体は侵入的→熟慮的の間が1晩長い。
    """
    nfc = traits.get("nfc", 0.5)
    locus = traits.get("internal_locus", 0.5)
    sigma = 0.25
    speed = float(np.clip(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma),
                          0.4, 2.5))
    thr = float(np.clip(float(cfg["threshold"])
                        * (1.0 - 0.3 * (nfc - 0.5) * 2.0) * speed, 0.10, 5.0))
    negw = float(cfg["neg_weight"]) * (1.0 + 0.3 * (0.5 - locus) * 2.0)
    return {
        "threshold": round(thr, 4),
        "neg_weight": round(negw, 4),
        "pos_weight": float(cfg["pos_weight"]),
        "incubation_days": int(cfg["incubation_days"]) + (1 if speed > 1.15 else 0),
        "cooldown_days": int(cfg["cooldown_days"]),
    }


# ---------------------------------------------------------------- 欲求プロファイル(needs)
# 潜在価値の 5 次元(いずれも中心 0.5=倍率 1.0 の連続量)。文献接地は
# docs/lit/needs__individual-differences.md(SDT 基本欲求 / Schwartz 基本価値 /
# Reiss 16 欲求 / Zuckerman 刺激欲求・Aron 感覚処理感受性)。engine には次元名を渡さない
# (no-fingerprint: 生成はこの factors 層のみ、変換 needs.py は不透明な倍率だけ扱う)。
NEEDS_DIMS: tuple[str, ...] = (
    "stimulation",   # 刺激・開放(Schwartz stimulation/self-direction, Zuckerman SS, Reiss 好奇心)
    "security",      # 安全・平穏(Schwartz security, Reiss tranquility/order, 高 SPS 側)
    "relatedness",   # 関係・慈愛(SDT relatedness, Schwartz benevolence, Reiss social contact)
    "competence",    # 有能・達成(SDT competence, Schwartz achievement, Reiss 好奇心)
    "autonomy",      # 自律・自己志向(SDT autonomy, Schwartz self-direction, Reiss independence)
)

# 職業 → 価値次元の微調整(向きのみ。大きさは cfg.occupation_weight で調律)。
_OCC_VALUE_NUDGE: dict[str, dict[str, float]] = {
    "デザイナー":   {"stimulation": 0.3, "autonomy": 0.2},
    "バンドマン":   {"stimulation": 0.4, "autonomy": 0.2, "relatedness": 0.1},
    "写真家":       {"stimulation": 0.3, "autonomy": 0.3},
    "フリーランス": {"autonomy": 0.3, "stimulation": 0.1},
    "エンジニア":   {"competence": 0.3, "security": 0.1},
    "会社員":       {"security": 0.2, "competence": 0.1},
    "カフェ店員":   {"relatedness": 0.2},
    "アパレル店員": {"relatedness": 0.2, "stimulation": 0.1},
    "美容師":       {"relatedness": 0.2},
    "配達員":       {"autonomy": 0.1},
    "大学生":       {"stimulation": 0.2, "competence": 0.1},
    "無職":         {},
}


def needs_profile(traits: dict[str, float], age: int, occupation: str,
                  rng: np.random.Generator, cfg: dict) -> dict[str, float]:
    """traits + 年齢 + 職業 + needs 専用乱数 → 潜在価値プロファイル(5 次元、[0,1])。

    因子名(trait/価値次元)を知ってよいのは factors/ だけ(no-fingerprint)。cognition/drive も
    engine も、needs.py が返す**不透明な倍率 dict** しか見ない。設計:
    docs/lit/needs__individual-differences.md の写像表(向きのみ、大きさは cfg で調律):
    - stimulation: NFC↑・リスク許容↑ で高、**年齢↑で低**(刺激欲求は加齢で低下)、創造系職業で高。
    - security:    リスク許容↓ で高、年齢↑で高。
    - relatedness: (1−内的統制) 寄り + 個体固有ゆらぎ、対人系職業で高。
    - competence:  NFC↑・内的統制↑ で高、専門系職業で高。
    - autonomy:    内的統制↑ で高、フリー系職業で高。

    R1(k 非依存): 入力は trait/年齢/職業 と needs 専用乱数のみ。belief 書き戻し・k.writeback を
    一切参照しない → 同じ seed/agent なら k を変えても同一プロファイル。乱数は **5 draw**
    (次元ごとの個体固有ゆらぎ)を needs 専用ストリームからのみ引く(既存ストリームは不変)。
    """
    def c(name: str) -> float:                     # trait を中心 0 の [-1,1] へ
        return (float(traits.get(name, 0.5)) - 0.5) * 2.0

    nfc, locus, risk = c("nfc"), c("internal_locus"), c("risk_tolerance")
    age_c = float(np.clip((int(age) - 18) / 52.0, 0.0, 1.0) - 0.5) * 2.0  # 若→負, 老→正
    occ = _OCC_VALUE_NUDGE.get(str(occupation), {})
    ow = float(cfg.get("occupation_weight", 1.0))
    sigma = float(cfg.get("profile_sigma", 0.10))

    center = {
        "stimulation": 0.5 + 0.22 * nfc + 0.22 * risk - 0.20 * age_c,
        "security":    0.5 - 0.28 * risk + 0.18 * age_c,
        "relatedness": 0.5 - 0.15 * locus,
        "competence":  0.5 + 0.22 * nfc + 0.18 * locus,
        "autonomy":    0.5 + 0.28 * locus,
    }
    profile: dict[str, float] = {}
    for dim in NEEDS_DIMS:                          # 固定順で 5 draw(決定論)
        val = center[dim] + ow * occ.get(dim, 0.0) + float(rng.normal(0.0, sigma))
        profile[dim] = round(float(np.clip(val, 0.0, 1.0)), 4)
    return profile


# --------------------------------------------------------------------------- #
# 年齢 → 思考/推論の「量」の個体係数(AGE-C。第116バッチ 2026-08-15・**既定 OFF**)
#
# 正典: docs/plans/age-diversity-plan.md §4-4(AGE-C)。
#
# 越えてはいけない線(§4-1): **年齢は「誰が撃てるか」を決めてはならない**。
#   `LodBudget.take()` と `requesters.sort(key=(-drive, id))` は 1 行も触らない。
#   年齢が変えてよいのは「本人がどれくらい撃ちたくなるか」= 個体の drive 分布だけで、
#   量は行動から創発する(高齢は閾値が僅かに高く周期が僅かに長い → ゲージの立ち上がりが
#   遅い → 上位に来る頻度が下がる → **結果として**呼が減る。誰も除外していない)。
#
# 文献が課す 3 制約(§4-4):
#   1. **「単一の年齢係数」は誤り**(Hartshorne & Germine 2015・N=48,537: 処理速度 19-22 /
#      感覚探求 19 / 自己制御 23-26 / 結晶性 50-65 とピークがばらける)。
#      → 周期と閾値を**別々の曲線**にする(1 本の age_factor を作らない)。
#   2. 加齢は「1 つの選択肢に時間がかかる」方向に効き、「選択肢の数を減らす」方向には
#      一貫しない(Queen 2012: 1 セルあたり処理時間 1,084→1,418→1,773 ms = +63% だが
#      開いたセル割合は .78/.85/.85 で落ちない)。→ **周期を伸ばすのは可・閲覧幅は削らない**。
#   3. 大きさは conf で調律する(`base_period` 自体が「暫定・人間データ未較正」と自己申告
#      している層なので、符号+桁は文献・微調整は conf = `needs_profile` と同じ作法)。
#
# **乱数を 1 draw も引かない純関数**(media.profile_for と同じ零 draw 型)= 呼び出し位置に
# 関係なく draw 順不変 = RNG バイト一致が自明に保たれる。
# --------------------------------------------------------------------------- #
def _piecewise(age: float, knots) -> float:
    """折れ線(x 昇順の [x, y] 列)を age で評価する。範囲外は端の値で平ら(外挿しない)。"""
    pts = [(float(x), float(y)) for x, y in knots]
    if not pts:
        return 0.0
    if age <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if age <= x1:
            span = x1 - x0
            if span <= 0.0:
                return y1
            return y0 + (y1 - y0) * (age - x0) / span
    return pts[-1][1]


def age_cognition_params(age: int, cfg: dict) -> dict[str, float]:
    """年齢 → 思考の**量**に関する個体係数(**決定論・乱数ゼロ**)。

    返す量(cognition 側はこのキー名しか知らない = no-fingerprint):
      period_mult … 定期発火の基本周期にかける個体倍率(>1 = 思考の間隔が長い)
      theta_delta … 発火申請の実効閾値に足す不透明な delta(>0 = 撃ちたくなりにくい)

    どちらも `ref_age` で恒等(1.0 / 0.0)になるよう正規化してから gain を掛ける。
    ⇒ **人口の中心年齢では現行較正どおり**で、若年/高齢だけが両側へ振れる(対称)。
    これが「小規模 = cap 非拘束のランでも呼数が概ね相殺する」根拠(§4-2)。
    """
    a = float(int(age))
    ref = float(cfg.get("ref_age", 38.0))
    # ---- 周期(熟慮時間の年齢曲線。Queen 2012)。ref_age で 1.0 になるよう規格化 ----
    pk = cfg.get("period_knots") or []
    base = _piecewise(a, pk)
    base_ref = _piecewise(ref, pk) or 1.0
    gain = float(cfg.get("period_gain", 1.0))
    mult = 1.0 + gain * (base / base_ref - 1.0)
    mult = min(float(cfg.get("period_max", 1.8)),
               max(float(cfg.get("period_min", 0.7)), mult))
    # ---- 閾値(感覚探求 19 歳ピーク / 自己制御 23-26 台地。Steinberg 2018)----
    tk = cfg.get("theta_knots") or []
    delta = (_piecewise(a, tk) - _piecewise(ref, tk)) * float(cfg.get("theta_gain", 1.0))
    clip = abs(float(cfg.get("theta_clip", 0.08)))
    delta = min(clip, max(-clip, delta))
    return {"period_mult": round(float(mult), 6),
            "theta_delta": round(float(delta), 6)}


def opinion_params(traits: dict[str, float],
                   rng: np.random.Generator) -> float:
    """traits → Friedkin-Johnsen 感受性 s(社会的影響の受けやすさ)。drive_params と同格。

    因子名を知ってよいのは factors/ だけ(no-fingerprint)。意見更新則(opinion.py)も
    engine も、この関数が返す s の値しか見ない。

    根拠(ヒューリスティック v1、RQ1 検収後に調整):
    - 内的統制(internal_locus)が高い人ほど「結果は自分次第」という信念が強く、
      他者の意見に流されにくい → s を下げる(FJ の anchor 引き戻しが相対的に効く)。
    - Need for Cognition が高い人は情報を批判的に精査し、単純な同調が弱まる傾向 →
      s を弱めに下げる。
    s ∈ [0.05, 0.95]。**1 draw のみ**で、build_agent の末尾に追加するため既存
    ストリームの draw 位置を動かさない(既定の再現性に影響しない)。
    """
    locus = traits.get("internal_locus", 0.5)
    nfc = traits.get("nfc", 0.5)
    mean_s = 0.55 - 0.30 * (locus - 0.5) * 2 - 0.10 * (nfc - 0.5) * 2
    return float(np.clip(rng.normal(mean_s, 0.12), 0.05, 0.95))


def sdt_params(traits: dict[str, float], rng: np.random.Generator,
               cfg: dict) -> dict[str, float]:
    """traits → 欲求ゲージ入力の個人別倍率 dict(SDT 動機プラグイン、既定 OFF)。

    自己決定理論(Ryan&Deci)の内発/外発の2軸を trait から近似し、出来事の種類ごとの
    倍率(reason → multiplier)を返す。cognition/drive.add はこの dict を **reason 文字列で
    引くだけ**(no-fingerprint: drive.py も engine も因子名・軸名を知らない)。

    - 内発(autonomy/competence)= NFC(認知の欲求)+ 内的統制(自律)→ 探索的出来事を押し上げる
      (初めての場所 novel_place / 知らない言葉 unknown_word)。
    - 外発(controlled/social)= 外的統制傾向 + リスク許容(社会的刺激志向)→ 社会・媒体由来を
      押し上げる(sns / 話しかけられた addressed / 同席 company / DM dm_received)。
    中立(0.5)= 倍率 1.0 の連続量。boost>1 で高特性ほど >1、低特性ほど <1。
    文献接地は**軸の因果構造だけ**(magnitude は conf の psych.sdt で調律。決め打ち禁止)。
    """
    nfc = traits.get("nfc", 0.5)
    locus = traits.get("internal_locus", 0.5)
    risk = traits.get("risk_tolerance", 0.5)
    intrinsic = float(np.clip(0.5 * nfc + 0.5 * locus + rng.normal(0.0, 0.05),
                              0.0, 1.0))
    extrinsic = float(np.clip(0.5 * (1.0 - locus) + 0.5 * risk
                              + rng.normal(0.0, 0.05), 0.0, 1.0))
    ib = float(cfg.get("intrinsic_boost", 1.4))
    eb = float(cfg.get("extrinsic_boost", 1.4))

    def _mult(score: float, boost: float) -> float:
        return round(float(1.0 + (score - 0.5) * 2.0 * (boost - 1.0)), 4)

    mods: dict[str, float] = {}
    for reason in ("novel_place", "unknown_word"):
        mods[reason] = _mult(intrinsic, ib)
    for reason in ("sns", "addressed", "company", "dm_received"):
        mods[reason] = _mult(extrinsic, eb)
    return mods


def collective_init(traits: dict[str, float],
                    rng: np.random.Generator) -> float:
    """集団効力感 state の初期値(集団プラグイン ON 時のみ registry から与える)。

    弱い接地: 内的統制が高いほど「集団で成せる」初期期待が高め(Bandura の collective
    efficacy は個人効力感と相関)。中心 0.35、std 0.08、clip[0,1]。1 draw のみ。
    """
    locus = traits.get("internal_locus", 0.5)
    base = 0.35 + 0.20 * (locus - 0.5) * 2.0
    return float(np.clip(rng.normal(base, 0.08), 0.0, 1.0))


def sample_traits(rng: np.random.Generator, tail_frac: float = 0.1) -> dict[str, float]:
    """trait を正規分布(clip)からサンプル。tail_frac の確率で上位 tail を明示確保(D6)。

    tail 確保 = silicon sampling の tail 過小(risk-register R2)への防御。
    どの trait が tail 化するかも乱数(設計者が「改変者の型」を決めない)。
    """
    traits: dict[str, float] = {}
    tail_trait = None
    if rng.random() < tail_frac:
        tail_trait = sorted(TRAITS)[int(rng.integers(len(TRAITS)))]
    for name in sorted(TRAITS):
        if name == tail_trait:
            traits[name] = float(rng.uniform(0.9, 1.0))
        else:
            traits[name] = float(np.clip(rng.normal(0.5, 0.18), 0.0, 1.0))
    return traits


# --------------------------------------------------------------------------- #
# 認知感度 g の個体パラメータ(第82バッチ 2026-08-01)
#
# 正典: docs/plans/source/cognition-design-record.md §2.5
#   「**LLM は g を設定しない。** ペルソナが初期値と動き方を決め、履歴が動かす」
#   「**ペルソナが決めるのは `g⁰`、`η`、`λ` の 3 つ。** g そのものではなく
#     『g の動き方』をペルソナが決める、という構造にする」
#
# 本 module に置く理由(no-fingerprint 契約 design §11): 因子名を綴ってよいのは
# factors/ だけ。cognition/ は**戻り値の数値しか見ない**(drive_params と同格)。
# 乱数を 1 draw も引かない純関数 = 呼び出し位置に関係なく draw 順不変。
# --------------------------------------------------------------------------- #
def cognition_params(traits: dict[str, float]) -> dict[str, float]:
    """traits → 感度 g の「動き方」+ 閾値 θ の個体成分(**すべて決定論・乱数ゼロ**)。

    返す量(cognition 側はこのキー名しか知らない):
      g0     … 感度の基準値 g⁰ の全体スケール(引き戻し先)
      eta    … 可塑性 η(動きやすさ)
      lam    … 引き戻し λ(性格の永続性。**この項がないと履歴が性格を上書きして全員収束する**)
      theta0 … 閾値 θ の個体倍率(1.0 = 較正テーブルどおり)
      bias_external / bias_body / bias_prediction
             … 観測チャンネルの**源**ごとの初期配分(条件 P = ペルソナ整合的な異質性)。
               源の名前は cognition/channels.py が持つ分類で、因子名ではない。

    ヒューリスティック v1(§2.5 の定性記述からの素朴な写像。較正は第83パイロット):
      - 思考頻度の傾性が高い → 感度が高く(g0↑)、可塑性も高い(eta↑)、閾値は低め(theta0↓)
      - リスク許容が高い → 外界のほうへ注意が寄る(bias_external↑ / bias_body↓)
      - 内的統制が高い → 自分の基準へ戻る力が強い(lam↑ = 履歴に流されにくい)
    """
    nfc = float(traits.get("nfc", 0.5))
    risk = float(traits.get("risk_tolerance", 0.5))
    locus = float(traits.get("internal_locus", 0.5))
    return {
        "g0": float(np.clip(1.0 + 0.60 * (nfc - 0.5) * 2.0, 0.20, 2.50)),
        "eta": float(np.clip(0.06 + 0.04 * (nfc - 0.5) * 2.0, 0.005, 0.30)),
        "lam": float(np.clip(0.05 + 0.03 * (locus - 0.5) * 2.0, 0.005, 0.30)),
        "theta0": float(np.clip(1.0 - 0.25 * (nfc - 0.5) * 2.0, 0.40, 1.80)),
        "bias_external": float(np.clip(1.0 + 0.50 * (risk - 0.5) * 2.0, 0.30, 2.00)),
        "bias_body": float(np.clip(1.0 - 0.50 * (risk - 0.5) * 2.0, 0.30, 2.00)),
        "bias_prediction": 1.0,
    }


def flat_cognition_params(value: float = 0.5) -> dict[str, float]:
    """`cognition_params` の**個体差ゼロ版**(条件 F/N の土台)。

    trait を全部 `value` にしたときの戻り値と厳密に一致する(= フラット条件は
    「全員が平均的なペルソナ」ではなく「ペルソナ由来の個体差が無い」状態)。
    """
    return cognition_params({k: float(value) for k in sorted(TRAITS)})


def outcome_scalar(states: dict[str, float]) -> float:
    """state ベクトル → 「ニーズ充足・ゴール進捗がどれだけ進んでいるか」[0,1]。

    設計 §2.5 の `r_ic`(感作項)の材料。**発火後の行動によるニーズ充足・ゴール進捗の
    改善分**を測るために、cognition 側が「差分を取れる 1 本のスカラ」を必要とする。
    その中身(どの state をどの向きで数えるか)は因子の意味を知る本 module の責務で、
    cognition/ は差分の数値しか見ない。

    向き: 自己効力感・当事者意識は正、不満は負。欠測キーは中立値(0.5 / 0.0)で扱う。
    """
    pos = (float(states.get("efficacy", 0.5))
           + float(states.get("ownership", 0.1))) * 0.5
    neg = float(states.get("grievance", 0.0))
    return float(np.clip(0.5 + 0.5 * (pos - neg), 0.0, 1.0))
