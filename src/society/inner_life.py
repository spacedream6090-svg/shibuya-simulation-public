"""内面の本格版(現実ギャップ 後続波 H6 2026-07-07。ユーザー要望)。

エージェントの内面を3機構で豊穣化する。設計正典:
  docs/design-candidates/gap-implementation-plan.md(§後続波)
  docs/lit/neuroscience__emotion-interest-attention.md §5.2(本格版)

  1. 離散感情ラベル(emotion): 既存 core affect(mood_valence × arousal)から Ekman 系の
     離散感情ラベルを**決定論の純関数**で写像し、「今は不安/高揚している」等をプロンプトに注入する。
     神経科学 §1.2 の二層設計 = ゲート計算は次元(valence×arousal)・プロンプトは離散ラベル
     (構成主義: 離散ラベルは core affect の上に構成される言語ラベル)。core affect の arousal は
     観測イベントのみ由来(affect.py の設計=k 非依存)。ラベルもその arousal と valence の純関数
     =k 非依存。ラベル変化のときだけ emotion_label を記録(sparse)。affect ON が前提(arousal が
     動かないと無効)。
  2. 長期目標・人生設計(goals)★keystone 駆動源: エージェントが数日〜数週の目標/野心を持つ
     (現状は1日計画のみ)。目標は価値プロファイル(needs)/persona/traits から**決定論導出**し、
     プロンプトに「長期的に◯◯したい」を注入して行動を長期的に方向づける。long_goal を記録。
  3. 趣味・関心・サブカルチャー(hobbies): persona(職業)+ 価値プロファイルから趣味を**決定論付与**
     し、プロンプト文脈 + 余暇の行き先バイアス(任意・専用 stream "inner")の核にする。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  価値次元名(stimulation/relatedness 等)を知ってよいのは factors と本モジュールだけ(no-fingerprint)。
  engine/cognition/world へは**不透明な文字列(感情ラベル・目標文・趣味文・行き先ノード)**だけを渡す。

R1 呼数不変: どの機構も generate() を1本も足さない。感情ラベルの更新・目標/趣味の付与は決定論
  (乱数不要)。プロンプト注入は**内容のみ**=呼数を変えない(FixedLLM で ON==OFF)。趣味の余暇行き先
  バイアスだけは物理位置=対面 co-location を変えうる(FixedLLM で ON!=OFF になりうる=crowd G4 /
  観光 H5 と同型)が、機構は k・内面状態(構成概念)を発火判断に食わせず名簿・config・専用 stream
  "inner"・物理位置のみ参照する=compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。
  行き先バイアスは既定 leisure_bias=0.0=無効(=注入だけの純粋 ON==OFF を既定で保つ)。

既定 OFF(inner_life.enabled=false)= 感情ラベル更新なし・目標/趣味を付けず・プロンプト注入なし・
  行き先バイアスなし・"inner" stream も引かない=イベント 0 件・乱数消費不変(ゴールデン
  golden_baseline_l1.json を守る)。新イベント種は emotion_label / long_goal(schema.py 登録済み=
  編集しない)。趣味は文脈で表現(新種不要)。
"""
from __future__ import annotations

from .observer.schema import Event

# 潜在価値の 5 次元(factors/registry.NEEDS_DIMS と同一。needs OFF 時は traits から近似する)。
_DIMS: tuple[str, ...] = (
    "stimulation", "security", "relatedness", "competence", "autonomy",
)

# ---- 離散感情ラベル(§1.2 Ekman 系。core affect の象限 → ラベル・プロンプト句)----
# 高覚醒×負=不安 / 高覚醒×正=高揚 / 高覚醒×中立=高ぶり(驚き)/ 低覚醒×負=しょんぼり /
# 低覚醒×正=穏やか / 低覚醒×中立=平静(落ち着き)。arousal 帯 × valence 符号の純関数。
_EMOTION: dict[tuple[str, str], tuple[str, str]] = {
    ("high", "neg"): ("不安", "不安で落ち着かない気分だ"),
    ("high", "pos"): ("高揚", "気分が高揚してわくわくしている"),
    ("high", "neu"): ("高ぶり", "何かが気になって落ち着かない"),
    ("low", "neg"): ("しょんぼり", "少し沈んで元気が出ない"),
    ("low", "pos"): ("穏やか", "穏やかで満ち足りた気分だ"),
    ("low", "neu"): ("平静", "落ち着いている"),
}

# ---- 長期目標(価値プロファイルの最優位次元 → 目標文)。決定論・keystone 駆動源 ----
_GOAL_BY_DIM: dict[str, str] = {
    "stimulation": "新しいことに挑戦して、刺激のある毎日を送りたい",
    "security":    "落ち着いた安定した暮らしを築きたい",
    "relatedness": "気の合う仲間を増やして、人とのつながりを深めたい",
    "competence":  "自分のスキルを磨いて、何かを成し遂げたい",
    "autonomy":    "自分のやり方で、自由に生きていきたい",
}

# ---- 趣味・関心(サブカルチャー)。ラベル → 余暇の行き先 POI カテゴリ(既知 cat のみ)----
_HOBBY_CAT: dict[str, str] = {
    "音楽・ライブ":       "nightlife",
    "ファッション・買い物": "shop",
    "アート・写真":       "leisure",
    "カフェ・食べ歩き":   "food",
    "散歩・公園めぐり":   "leisure",
}
# 職業 → 主たる趣味(persona 由来)。無ければ最優位価値次元から補う。
_HOBBY_BY_OCC: dict[str, str] = {
    "バンドマン":   "音楽・ライブ",
    "大学生":       "音楽・ライブ",
    "デザイナー":   "アート・写真",
    "写真家":       "アート・写真",
    "フリーランス": "アート・写真",
    "アパレル店員": "ファッション・買い物",
    "美容師":       "ファッション・買い物",
    "カフェ店員":   "カフェ・食べ歩き",
    "会社員":       "カフェ・食べ歩き",
    "エンジニア":   "カフェ・食べ歩き",
    "配達員":       "散歩・公園めぐり",
    "無職":         "散歩・公園めぐり",
}
_HOBBY_BY_DIM: dict[str, str] = {
    "stimulation": "音楽・ライブ",
    "security":    "カフェ・食べ歩き",
    "relatedness": "カフェ・食べ歩き",
    "competence":  "アート・写真",
    "autonomy":    "散歩・公園めぐり",
}


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------- config
def build_cfg(raw) -> dict:
    """conf の inner_life ブロックを型強制つきで正準化する(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(diversity/health/household と同型)。master enabled=false
    のとき全機構(感情ラベル・目標・趣味・行き先バイアス)が完全 no-op=バイト一致。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    emo = dict(raw.get("emotion", {}) or {})
    goals = dict(raw.get("goals", {}) or {})
    hob = dict(raw.get("hobbies", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),      # ★master ゲート
        "emotion": {
            "enabled": bool(emo.get("enabled", True)),
            "arousal_high": float(emo.get("arousal_high", 0.35)),      # 覚醒 ≥ これ=高覚醒帯
            "valence_deadzone": float(emo.get("valence_deadzone", 0.15)),  # |mood| < これ=中立
        },
        "goals": {
            "enabled": bool(goals.get("enabled", True)),
            "inject_prompt": bool(goals.get("inject_prompt", True)),
            "leisure_bias": float(goals.get("leisure_bias", 0.0)),     # 現状は注入のみ(拡張余地)
        },
        "hobbies": {
            "enabled": bool(hob.get("enabled", True)),
            "inject_prompt": bool(hob.get("inject_prompt", True)),
            "leisure_bias": float(hob.get("leisure_bias", 0.0)),       # ★>0 で余暇行き先を趣味へ
        },
    }


def cfg_of(sim) -> dict:
    """inner_life 設定を sim に一度だけキャッシュして返す(government._gov と同型の遅延構築)。

    simulation.py を編集せずに配線するための据え付け。sim.cfg.inner_life を型強制して保持する。"""
    cfg = getattr(sim, "innerlifecfg", None)
    if cfg is None:
        cfg = build_cfg(sim.cfg.get("inner_life", None))
        sim.innerlifecfg = cfg
    return cfg


def enabled(sim) -> bool:
    """内面本格版(H6)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


# ---------------------------------------------------------------- 価値プロファイル
def _value_profile(agent) -> dict[str, float]:
    """5 次元の潜在価値プロファイル(needs ON なら _needs_profile、無ければ traits から近似)。

    needs.build_mods が付す agent._needs_profile(価値名を持つ層)があればそれを使う。無ければ
    traits(nfc/internal_locus/risk_tolerance)から registry.needs_profile と同じ向きで近似する
    (乱数なし=決定論)。no-fingerprint: この写像は本モジュール(CHECKED_DIRS 外)に閉じる。"""
    prof = getattr(agent, "_needs_profile", None)
    if prof:
        return {d: float(prof.get(d, 0.5)) for d in _DIMS}
    t = getattr(agent, "traits", None) or {}
    nfc = float(t.get("nfc", 0.5))
    locus = float(t.get("internal_locus", 0.5))
    risk = float(t.get("risk_tolerance", 0.5))
    return {
        "stimulation": _clip01(0.5 + 0.3 * (nfc - 0.5) + 0.3 * (risk - 0.5)),
        "security":    _clip01(0.5 - 0.3 * (risk - 0.5)),
        "relatedness": _clip01(0.5 - 0.2 * (locus - 0.5)),
        "competence":  _clip01(0.5 + 0.3 * (nfc - 0.5) + 0.2 * (locus - 0.5)),
        "autonomy":    _clip01(0.5 + 0.3 * (locus - 0.5)),
    }


def _dominant_dim(agent) -> str:
    """最優位の価値次元(同点は _DIMS の固定順で先勝ち=決定論)。"""
    prof = _value_profile(agent)
    return max(_DIMS, key=lambda d: (prof[d], -_DIMS.index(d)))


# ---------------------------------------------------------------- 離散感情ラベル(§1.2)
def _mood_valence(states: dict[str, float]) -> float:
    """core affect の valence 軸 = efficacy − grievance を [-1,1] に(§6.1)。新規変数ゼロ。"""
    e = float(states.get("efficacy", 0.5))
    g = float(states.get("grievance", 0.0))
    v = e - g
    return -1.0 if v < -1.0 else 1.0 if v > 1.0 else v


def emotion_label(valence: float, arousal: float, ecfg: dict) -> tuple[str, str]:
    """core affect(mood_valence × arousal)→ 離散感情(ラベル, プロンプト句)の**純関数**。

    決定論・RNG不要・k 非依存(valence は factors 層 state、arousal は観測イベントのみ由来)。
    arousal 帯(high/low)× valence 符号(neg/neu/pos)の 6 象限を Ekman 系ラベルへ写す
    (構成主義: 離散ラベル = core affect 上に構成される言語ラベル)。"""
    band = "high" if float(arousal) >= float(ecfg["arousal_high"]) else "low"
    dz = float(ecfg["valence_deadzone"])
    v = float(valence)
    sign = "neg" if v <= -dz else "pos" if v >= dz else "neu"
    return _EMOTION[(band, sign)]


def update_emotion(agent, cfg: dict, step: int, sim_min: int, logger) -> None:
    """毎step(affect の decay と同居): 現在の core affect から離散感情ラベルを再計算する。

    ラベルが変わったときだけ emotion_label を記録(sparse)+ プロンプト句を agent に保持する。
    arousal/valence の純関数=乱数を引かず・drive(発火系)に一切フィードバックしない(§6.3 自己増幅
    ループ防止)。affect OFF / emotion 無効のとき呼び出し側が来ない(=no-op)。"""
    ecfg = cfg["emotion"]
    mv = _mood_valence(getattr(agent, "states", {}) or {})
    a = float(getattr(agent, "arousal", 0.2))
    label, phrase = emotion_label(mv, a, ecfg)
    agent._emotion_phrase = phrase
    if getattr(agent, "_emotion_label", None) != label:
        agent._emotion_label = label
        logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="emotion_label", x=agent.x, y=agent.y,
                         payload={"label": label}))


# ---------------------------------------------------------------- 長期目標・趣味の付与
def long_goal(agent) -> str:
    """価値プロファイルの最優位次元 → 長期目標文(決定論)。keystone 行動の駆動源。"""
    return _GOAL_BY_DIM[_dominant_dim(agent)]


def hobbies(agent) -> list[str]:
    """persona(職業)+ 価値プロファイル → 趣味・関心(1〜2件、決定論)。サブカルチャーの核。

    主たる趣味は職業から(無い職業は最優位価値次元から)。副次の趣味は最優位次元由来(主と異なれば
    追加)。下位集団形成の核 + プロンプト文脈 + 余暇の行き先バイアスに使う。"""
    dim = _dominant_dim(agent)
    primary = _HOBBY_BY_OCC.get(str(getattr(agent, "occupation", "")),
                                _HOBBY_BY_DIM[dim])
    out = [primary]
    secondary = _HOBBY_BY_DIM[dim]
    if secondary != primary:
        out.append(secondary)
    return out


def precompute(sim, step: int, sim_min: int) -> None:
    """起動後1回: 各エージェントに長期目標・趣味を付与する(決定論・id 昇順・乱数なし)。

    goals ON なら agent.life_goal を設定し long_goal を1件記録(設定=変化なので1回)。hobbies ON なら
    agent.hobbies を設定(趣味は文脈で表現=専用イベント種なし)。OFF の機構は付与しない(getattr 既定で
    バイト一致)。emotion は毎step の update_emotion が担うのでここでは扱わない。"""
    cfg = cfg_of(sim)
    goals_on = cfg["goals"]["enabled"]
    hob_on = cfg["hobbies"]["enabled"]
    for a in sorted(sim.agents, key=lambda a: a.id):
        if hob_on:
            a.hobbies = hobbies(a)
        if goals_on:
            g = long_goal(a)
            a.life_goal = g
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                 kind="long_goal", x=a.x, y=a.y,
                                 payload={"goal": g}))


def assign_for_entry(sim, agent) -> None:
    """1 個体ぶんの長期目標・趣味の付与(入場駆動・冪等・決定論・乱数ゼロ。レーン乙 A11)。

    ★``precompute`` は**起動時 1 回**しか走らないので、プール回転で day1 以降に入場する
      個体は長期目標も趣味も持たないまま暮らしていた(``_inner_life_init`` の意味を
      「全員済み」から「入場時に済ませる」へ広げるのが本関数)。
    ★L1 は 1 件も増やさない: ``precompute`` が出す ``long_goal`` は「起動時セグメント」の
      構成要素(resume の ``_init_event_mark`` が本数を数えている)で、入場のたびに足すと
      その契約と ON/OFF のイベント本数の宣言が崩れる。値は pool の退避で運ばれるので、
      再来街でも同じ目標を持ち続ける。OFF の機構は付与しない(getattr 既定でバイト一致)。"""
    cfg = cfg_of(sim)
    if cfg["hobbies"]["enabled"] and not getattr(agent, "hobbies", None):
        agent.hobbies = hobbies(agent)
    if cfg["goals"]["enabled"] and not getattr(agent, "life_goal", ""):
        agent.life_goal = long_goal(agent)


# ---------------------------------------------------------------- プロンプト注入(内容のみ=R1 呼数不変)
def emotion_line(agent, cfg: dict, *, affect_on: bool) -> str | None:
    """発火プロンプト用: 今の離散感情1行(affect ON が前提)。既定 OFF/非該当は None=不変。"""
    if not (affect_on and cfg["enabled"] and cfg["emotion"]["enabled"]):
        return None
    phrase = getattr(agent, "_emotion_phrase", None)
    if not phrase:
        return None
    return f"今の気持ち: {phrase}。"


def goal_line(agent, cfg: dict) -> str | None:
    """発火プロンプト用: 長期的な目標1行(keystone の方向づけ)。既定 OFF/未設定は None=不変。"""
    if not (cfg["enabled"] and cfg["goals"]["enabled"] and cfg["goals"]["inject_prompt"]):
        return None
    g = getattr(agent, "life_goal", None)
    if not g:
        return None
    return f"長期的な目標: {g}。"


def hobby_line(agent, cfg: dict) -> str | None:
    """発火プロンプト用: 趣味・関心1行(サブカルチャーの文脈)。既定 OFF/未設定は None=不変。"""
    if not (cfg["enabled"] and cfg["hobbies"]["enabled"] and cfg["hobbies"]["inject_prompt"]):
        return None
    hobs = getattr(agent, "hobbies", None)
    if not hobs:
        return None
    return f"趣味・関心: {'、'.join(hobs)}。"


# ---------------------------------------------------------------- 余暇の行き先バイアス(任意)
def hobby_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """余暇の行き先を趣味の POI カテゴリへ寄せる(任意・専用 stream "inner")。決定論・非LLM。

    OFF / 趣味無効 / leisure_bias=0 / 趣味未設定 / 抽選外 / 候補なし なら None(既定不変)。bias<=0 なら
    乱数を一切引かず None(=注入だけの純粋 ON==OFF を既定で保つ)。行き先(移動)だけを変え、発火判断・
    LLM 呼数は増やさない(R1: 呼数不変は compute_matched 下の k 不変性で担保)。専用 stream "inner" は
    既存 draw 順を汚さない(ゴールデン/決定論の保護)。趣味は文脈で表現=専用イベント種を出さない。"""
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    hcfg = cfg["hobbies"]
    if not hcfg["enabled"]:
        return None
    bias = float(hcfg["leisure_bias"])
    if bias <= 0.0:
        return None
    hobs = getattr(agent, "hobbies", None)
    if not hobs:
        return None
    cat = _HOBBY_CAT.get(hobs[0])
    if not cat:
        return None
    rng = sim.hub.stream("inner", agent.id, step)
    if rng.random() >= bias:
        return None
    cands = [p for p in sim.city.pois_by_cat(cat) if p["node"] != agent.node]
    if not cands:
        return None
    return cands[int(rng.integers(len(cands)))]["node"]
