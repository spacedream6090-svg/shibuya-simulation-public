"""開放行動の価値レイヤ(第17バッチ 2026-07-10。既定 OFF)。

ユーザー構想: 物理・現実の制約以外は何でもできる開放行動("do")を LLM の選択に委ね、
その意味づけを「価値の4軸」で観測する。軸=ユーザーの3分類(実用/感情/社会。『世界2.0』)
+文献の多理論一致で追加した認識的価値(好奇心・新奇・学び)。
学術接地: docs/research/desire-value-theory.md(SNG 消費価値・Holbrook・SDT・Alderfer 飽和/
退行・warm-glow・経験財)。

レイヤ分担(no-fingerprint / needs.py と同じ流儀):
- 本 module が「価値名を知る層」(CHECKED_DIRS=engine/cognition/world の外)。
- 判定は決定論の語彙辞書(基盤)+LLM の中立自己申告(同じ1呼の JSON 内 value キー=呼数不変)
  のブレンド。LLM judge は使わない(R1)。
- 充足(satiation)は「感度=可・目標注入=不可」の線内: 飢えた価値に紐づく出来事ほど
  ゲージに響く倍率 sat_mods(reason→mult)だけを外へ出す。プロンプトに価値名は出さない。

既定 OFF = ヘッダ不変・sat 属性 None・sat_mods None → 全経路バイト一致。
乱数は一切使わない(完全決定論。価格も中央値固定)。
"""
from __future__ import annotations

TAGS = ("utility", "emotion", "social", "epistemic")

# LLM の自己申告(日本語/英語ゆらぎ)→ 正準タグ
_REPORT_ALIASES = {
    "実用": "utility", "実用的": "utility", "役に立つ": "utility", "utility": "utility",
    "functional": "utility", "practical": "utility",
    "感情": "emotion", "感情的": "emotion", "楽しい": "emotion", "emotion": "emotion",
    "emotional": "emotion", "hedonic": "emotion",
    "社会": "social", "社会的": "social", "つながり": "social", "social": "social",
    "認識": "epistemic", "認識的": "epistemic", "好奇心": "epistemic", "学び": "epistemic",
    "新奇": "epistemic", "epistemic": "epistemic", "curiosity": "epistemic",
}

# 行動カテゴリ辞書(決定論。先勝ち=リスト順に部分一致)。tags は重み(合計~1)。
# cost=(円) は消費を伴うカテゴリの中央値(乱数なし)。文献の生活行動分類(社会生活基本調査
# 3次活動)を骨格に、渋谷固有語を避けた一般語彙のみ(基盤層=場所非依存)。
_CATEGORIES: tuple[tuple[str, dict], ...] = (
    ("創作", {"kw": ("絵を描", "描く", "作曲", "曲を作", "小説", "詩を書", "書き物",
                     "創作", "工作", "編み物", "料理を作", "写真を撮", "動画を作",
                     "デザイン", "ものづくり", "handmade", "DIY"),
              "tags": {"emotion": 0.35, "epistemic": 0.35, "utility": 0.2, "social": 0.1},
              "cost": 0}),
    ("学習", {"kw": ("勉強", "学ぶ", "学習", "練習", "調べ", "資格", "講座", "本を読",
                     "読書", "語学"),
              "tags": {"epistemic": 0.6, "utility": 0.3, "emotion": 0.1},
              "cost": 0}),
    ("鑑賞", {"kw": ("テレビ", "映画", "アニメ", "ドラマ", "動画を見", "配信を見",
                     "音楽を聴", "ラジオ", "美術館", "展示", "鑑賞", "ライブ", "コンサート",
                     "観劇", "舞台"),
              "tags": {"emotion": 0.6, "epistemic": 0.25, "social": 0.15},
              "cost": 1500}),
    ("ゲーム", {"kw": ("ゲーム", "ボードゲーム", "対戦"),
                "tags": {"emotion": 0.55, "epistemic": 0.25, "social": 0.2},
                "cost": 0}),
    ("運動", {"kw": ("運動", "ランニング", "走る", "ジョギング", "筋トレ", "ジム", "ヨガ",
                     "ストレッチ", "体操", "泳", "サッカー", "野球", "バスケ", "ダンス"),
              "tags": {"emotion": 0.4, "utility": 0.4, "social": 0.2},
              "cost": 0}),
    ("散策", {"kw": ("散歩", "散策", "ぶらぶら", "街を歩", "歩き回", "公園", "自然",
                     "景色", "空を見", "夜景"),
              "tags": {"emotion": 0.5, "epistemic": 0.3, "utility": 0.2},
              "cost": 0}),
    ("交流", {"kw": ("友達と", "友人と", "話す", "おしゃべり", "お茶", "飲みに", "飲み会",
                     "遊びに", "会いに", "集まり", "交流"),
              "tags": {"social": 0.6, "emotion": 0.4},
              "cost": 1200}),
    ("飲食", {"kw": ("食べ", "ご飯", "ランチ", "カフェ", "コーヒー", "スイーツ", "甘いもの",
                     "ラーメン", "居酒屋"),
              "tags": {"emotion": 0.6, "utility": 0.3, "social": 0.1},
              "cost": 900}),
    ("買い物", {"kw": ("買い物", "買う", "ショッピング", "服を見", "雑貨", "古着"),
                "tags": {"utility": 0.4, "emotion": 0.45, "social": 0.15},
                "cost": 2500}),
    ("向社会", {"kw": ("募金", "寄付", "ボランティア", "手伝う", "助け", "ゴミ拾い",
                       "献血", "地域の"),
                "tags": {"social": 0.65, "emotion": 0.25, "utility": 0.1},
                "cost": 500}),
    ("家事", {"kw": ("掃除", "洗濯", "片付け", "整理", "家事", "修理", "手入れ"),
              "tags": {"utility": 0.75, "emotion": 0.15, "epistemic": 0.1},
              "cost": 0}),
    ("休養", {"kw": ("休む", "昼寝", "ぼーっと", "のんびり", "ゆっくり", "何もしない",
                     "リラックス", "瞑想", "温泉", "サウナ", "風呂"),
              "tags": {"emotion": 0.7, "utility": 0.3},
              "cost": 0}),
    ("探索", {"kw": ("行ったことのない", "新しい場所", "探検", "探索", "冒険", "初めての"),
              "tags": {"epistemic": 0.6, "emotion": 0.4},
              "cost": 0}),
)
_OTHER_TAGS = {"utility": 0.25, "emotion": 0.25, "social": 0.25, "epistemic": 0.25}

# 5次元潜在プロファイル(factors/registry.needs_profile)→ 4タグ(desire-value-theory §6.3)
_DIM_TO_TAG = {
    "utility":   {"competence": 1.0},
    "emotion":   {"stimulation": 0.5, "security": 0.5},
    "social":    {"relatedness": 1.0},
    "epistemic": {"stimulation": 0.5, "competence": 0.5},
}

# drive の reason → タグ橋渡し(needs._REASON_SENSITIVITY の次元を _DIM_TO_TAG で写像した近似。
# ここに価値名を閉じ込め、drive.add は sat_mods を引くだけ=名前を知らない)。
_REASON_TO_TAG = {
    "novel_place":  {"epistemic": 0.8, "emotion": 0.2},
    "unknown_word": {"epistemic": 1.0},
    "addressed":    {"social": 1.0},
    "dm_received":  {"social": 1.0},
    "company":      {"social": 1.0},
    "silence":      {"emotion": 0.5, "social": 0.5},
    "news":         {"epistemic": 0.6, "utility": 0.4},
    "sns":          {"social": 0.6, "emotion": 0.4},
}


# ---- 生活の自己決定 P2(D3 棚卸し。docs/plans/agent-freedom-plan.md #6-#10。既定 全 OFF)----
# 5項目(move_home/buy/study/partnership/deviance)を個別 bool で切る。全 false=現行とバイト一致。
# メニューは中立提示(客観条件のみ)・裁定は決定論(新 stream のみ)=R1 の呼数不変。
_P2_DEFAULTS = {
    "move_home": False,     # #6 住居移転(空き住戸へ転居。敷金=現金障壁)
    "buy": False,           # #7 消費の意思(発火時の buy。非発火の buy 抽選はフォールバックで残す)
    "study": False,         # #8 学び直し(学校/図書館で聴講=記録のみ。賃金経路は Skill 討議後)
    "partnership": False,   # #9 家族の主体性(交際の申込/別れ。相互応答は将来拡張)
    "deviance": False,      # #10 軽微な逸脱(無許可出店→既存 enforcement が摘発)
    "deposit": 50000.0,     # #6 引っ越しの敷金(月家賃1ヶ月分相当の現金障壁。建物別家賃の内生化はしない)
    "deviance_fine": 5000.0,  # #10 無許可出店の摘発時の罰金(既存 enforcement 機構に接続)
    "partner_closeness": 15.0,  # #9 交際成立の closeness 閾値(household 既定と同値。household OFF 時の後退値)
}
_P2_BOOL = ("move_home", "buy", "study", "partnership", "deviance")


def _build_p2(raw) -> dict:
    """freedom.p2 サブブロックの型強制(既定 全 false=完全 no-op)。OmegaConf/dict の両方を受ける。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = dict(_P2_DEFAULTS)
    for k, v in raw.items():
        if k not in _P2_DEFAULTS:
            continue
        cfg[k] = bool(v) if k in _P2_BOOL else float(v)
    return cfg


def build_cfg(raw: dict | None) -> dict:
    """conf の freedom ブロックを型強制つきで正準化(既定 OFF=現行と完全同一)。"""
    raw = dict(raw or {})
    return {
        "open_actions": bool(raw.get("open_actions", False)),
        "satiation_gain": float(raw.get("satiation_gain", 0.3)),  # 0=純観測(k掃引時は0推奨)
        "sat_step": float(raw.get("sat_step", 0.15)),      # 自由行動1回の充足量
        "sat_decay": float(raw.get("sat_decay", 0.15)),    # 日次の中立(0.5)への回帰率
        "max_minutes": int(raw.get("max_minutes", 240)),
        "mods_lo": float(raw.get("mods_lo", 0.7)),
        "mods_hi": float(raw.get("mods_hi", 1.5)),
        # 第70バッチ IDEA②(既定 全 False=ヘッダ・イベント・L2 ともバイト一致)。
        # undefined_register: enum 外の行動主張を `undefined_action` として記録する(パース後の
        #   振り分けのみ=プロンプト不変・LLM 呼数不変・乱数ゼロ)。
        # explicit_nothing: 「何もしない」をヘッダに 1 行だけ足す(open_actions の "do" と同型)+
        #   `stay{reason:"chosen_nothing"}` の記録を開く。
        "undefined_register": bool(raw.get("undefined_register", False)),
        "explicit_nothing": bool(raw.get("explicit_nothing", False)),
        "p2": _build_p2(raw.get("p2")),
    }


def classify(what: str) -> tuple[str, dict[str, float]]:
    """行動の自由文 → (カテゴリ, タグ重み)。決定論(先勝ち)。該当なし=「その他」均等。"""
    for name, spec in _CATEGORIES:
        for kw in spec["kw"]:
            if kw in what:
                return name, dict(spec["tags"])
    return "その他", dict(_OTHER_TAGS)


def category_cost(category: str) -> float:
    for name, spec in _CATEGORIES:
        if name == category:
            return float(spec.get("cost", 0))
    return 0.0


def parse_report(raw) -> list[str]:
    """LLM の自己申告 value(list/str)→ 正準タグ列(未知語は落とす=中立)。"""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip()
        tag = _REPORT_ALIASES.get(key)
        if tag is None:            # 「感情的価値」のような揺らぎを部分一致で拾う
            for alias, t in _REPORT_ALIASES.items():
                if alias and alias in key:
                    tag = t
                    break
        if tag and tag not in out:
            out.append(tag)
    return out


def blend_tags(lex_tags: dict[str, float], report: list[str]) -> dict[str, float]:
    """辞書タグ(基盤)と自己申告(本人の意味づけ)を 1:1 でブレンド。申告なし=辞書のまま。"""
    if not report:
        return lex_tags
    share = 1.0 / len(report)
    out = {t: 0.5 * lex_tags.get(t, 0.0) for t in TAGS}
    for t in report:
        out[t] = out.get(t, 0.0) + 0.5 * share
    return {t: round(v, 4) for t, v in out.items() if v > 0}


def profile4(agent) -> dict[str, float]:
    """5次元潜在プロファイル(needs ON 時のみ存在)→ 4タグの素質(0-1、中立0.5)。"""
    p5 = getattr(agent, "_needs_profile", None)
    if not p5:
        return {t: 0.5 for t in TAGS}
    out = {}
    for tag, coeffs in _DIM_TO_TAG.items():
        acc = sum(c * float(p5.get(dim, 0.5)) for dim, c in coeffs.items())
        out[tag] = round(min(1.0, max(0.0, acc)), 4)
    return out


def value_match(agent, tags: dict[str, float]) -> float:
    """行動のタグ×本人の素質の一致度(-1..1、中立素質なら0)。観測用。"""
    prof = profile4(agent)
    return round(sum(w * (prof.get(t, 0.5) - 0.5) * 2.0 for t, w in tags.items()), 4)


def satisfy(agent, tags: dict[str, float], cfg: dict) -> dict[str, float]:
    """自由行動による充足(限界効用逓減)。sat 属性が無ければ no-op(OFF)。"""
    sat = getattr(agent, "sat", None)
    if sat is None:
        return {}
    step = cfg["sat_step"]
    for t, w in tags.items():
        cur = sat.get(t, 0.5)
        sat[t] = round(min(1.0, cur + step * w * (1.0 - cur)), 4)
    _refresh_mods(agent, cfg)
    return dict(sat)


def decay_daily(agent, cfg: dict) -> None:
    """日次: sat を中立0.5へ回帰(需要の再蓄積)+昨日の経験(behav_today 由来は drive 側で
    カウント済み)の軽い充足。OFF(sat None)なら no-op。"""
    sat = getattr(agent, "sat", None)
    if sat is None:
        return
    d = cfg["sat_decay"]
    for t in TAGS:
        cur = sat.get(t, 0.5)
        sat[t] = round(cur + d * (0.5 - cur), 4)
    _refresh_mods(agent, cfg)


def _refresh_mods(agent, cfg: dict) -> None:
    """sat(充足)→ sat_mods(reason→倍率)。飢えた価値に紐づく出来事ほどゲージに響く
    (感度のみ=目標注入なし)。satiation_gain=0 なら常に1.0(=純観測)。"""
    gain = cfg["satiation_gain"]
    sat = agent.sat
    lo, hi = cfg["mods_lo"], cfg["mods_hi"]
    mods: dict[str, float] = {}
    for reason, coeffs in _REASON_TO_TAG.items():
        hunger = sum(c * (0.5 - sat.get(t, 0.5)) * 2.0 for t, c in coeffs.items())
        mods[reason] = round(min(hi, max(lo, 1.0 + gain * hunger)), 4)
    agent.sat_mods = mods
