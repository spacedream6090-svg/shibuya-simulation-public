"""欲求駆動発火(生態系フェーズ Phase A、ユーザー仕様 2026-07-04)。

仕組み: 出来事が欲求ゲージ(drive, 0-1)を溜める → 個人別閾値で「申請」→
申請時に個人の重み(fire_weight)で確率的に発火判定 → 不発ならゲージを
数十%減衰(fail_decay)→ 後のstepで再蓄積・再申請。

設計原則:
- 入力は**観測可能な出来事のみ**(信念・k条件に依存しない)= R1 計算量交絡対策。
  発火回数は k 条件間で監査可能(drive_request イベント+L2)。
- 個体差(drive_threshold / fire_weight)は traits から factors/ 内で写像
  (no-fingerprint: このモジュールも engine も因子名を知らない)。
- 学術接地: Park 2023 reflection(重要度累積閾値)/ Fogg B=MAP / OASIS 確率的
  活性化 / metacontrol(認知努力は stake の高い時に配分)。係数は RQ1 検収後に調整。
"""
from __future__ import annotations

# 出来事の種類 → ゲージ増分の既定重み(conf の drive.weights で上書き可)
DEFAULT_WEIGHTS = {
    "novel_place": 0.35,    # 初めての場所に着いた
    "congestion": 0.30,     # ひどい混雑に巻き込まれた
    "unknown_word": 0.40,   # 知らない言葉を聞いた
    "addressed": 0.25,      # 対面/DMで話しかけられた・聞いた
    "dm_received": 0.35,    # DM を受け取った
    "news": 0.15,           # ニュースを見た
    "sns": 0.08,            # SNS タイムラインを見た
    "company": 0.10,        # 人と一緒にいる(社交圧)
    "silence": 0.015,       # 沈黙の継続(最後の思考からの時間経過)
    "state_change": 0.50,   # 内部状態の変化量 |Δ| に掛ける係数
}


def _boredom_cfg(raw: dict | None) -> dict:
    """好奇心/退屈の内発ドライブ(P2 S5)の生 cfg を数値へ正規化。既定 enabled=False=OFF で
    現行と完全同一(ゲージを一切触らない=バイト一致)。設計: docs/plans/p2-interstitial-design.md §1 S5。"""
    b = dict(raw or {})
    return {
        # 既定 OFF = ゲージ機構が完全 no-op(boredom_* が agent に一切触れない=バイト一致)。
        "enabled": bool(b.get("enabled", False)),
        "accrual": float(b.get("accrual", 0.06)),          # 同じ場所への長居 1step の蓄積
        "decay": float(b.get("decay", 0.10)),              # 移動・新奇での減衰率
        "novelty_relief": float(b.get("novelty_relief", 0.20)),  # 新しい場所到達の追加減衰(脱馴化の入口)
        "threshold": float(b.get("threshold", 0.60)),      # 探索発火の閾値(ゲージがこれ以上で候補)
        "fire_prob": float(b.get("fire_prob", 0.5)),       # 閾値超え時に実際に探索へ動く確率(stream "boredom")
        "cooldown_steps": int(b.get("cooldown_steps", 6)), # 探索発火後の不応期(step)
        "radius_m": float(b.get("radius_m", 300.0)),       # 探索先を探す近傍半径(m)
        "stay_steps": int(b.get("stay_steps", 2)),         # 探索先での滞在 step 数
    }


def build_cfg(raw: dict | None) -> dict:
    raw = dict(raw or {})
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in dict(raw.get("weights", {})).items()})
    d = dict(raw.get("drift", {}) or {})
    drift = {
        # 既定 OFF = 現行 fixed と完全同一(theta_drift 恒等 0=バイト一致)。
        "enabled": bool(d.get("enabled", False)),
        "habit_rate": float(d.get("habit_rate", 0.02)),      # 馴化(低顕著入力→閾値↑)
        "sens_rate": float(d.get("sens_rate", 0.02)),        # 鋭敏化(発火・新奇→閾値↓)
        "recovery_rate": float(d.get("recovery_rate", 0.01)),  # 自発回復(0へ緩やか)
        "max_abs": float(d.get("max_abs", 0.15)),            # |theta_drift| の可動域
    }
    return {
        "decay": float(raw.get("decay", 0.02)),
        "fail_decay": float(raw.get("fail_decay", 0.30)),
        "fire_reset": float(raw.get("fire_reset", 0.20)),
        "refractory_steps": int(raw.get("refractory_steps", 3)),
        "conv_max_turns": int(raw.get("conv_max_turns", 3)),
        "conv_cooldown_steps": int(raw.get("conv_cooldown_steps", 6)),
        # 発火の関数形の seam(B段)。既定 fixed=現行(閾値ゲート+個人重み抽選)。
        # logistic=p=σ(slope·(drive−threshold)) に一本化(既定挙動には触れない)。
        "firing": str(raw.get("firing", "fixed")),
        "slope": float(raw.get("slope", 8.0)),
        "weights": weights,
        "drift": drift,
        "boredom": _boredom_cfg(raw.get("boredom", {})),
    }


# ---------------------------------------------------------------- 発火閾値ドリフト E2
# 実効閾値 = clip(drive_threshold + theta_drift, 0.30, 0.85)。設計:
# docs/research/reflection-drift.md。R1: 入力は出来事量・発火経験のみ(k 非依存)。
_HABIT_REASONS = frozenset({"silence", "sns", "company", "news"})   # 馴化(閾値↑)
_SENS_REASONS = frozenset({"novel_place", "unknown_word", "state_change"})  # 鋭敏化(↓)
_TH_LO, _TH_HI = 0.30, 0.85


def effective_threshold(agent, affect_delta: float = 0.0) -> float:
    """発火申請の実効閾値。theta_drift(E2 ドリフト)と affect_delta(覚醒の逆U字変調、
    factors/affect が算出する不透明な delta)の合算が 0(=OFF・中立)なら drive_threshold を
    厳密に返す(バイト一致)。それ以外は clip(drive_threshold + Σdelta, 0.30, 0.85)。

    affect_delta 既定 0.0 = 感情・興味・注意ハブ OFF(threshold_gain=0)で恒等。既存呼び出し
    (単一引数)は default 0.0 でそのまま=現行と完全同一。"""
    total = getattr(agent, "theta_drift", 0.0) + float(affect_delta)
    if total == 0.0:
        return agent.drive_threshold
    return min(_TH_HI, max(_TH_LO, agent.drive_threshold + total))


def _drift_add(agent, delta: float, cfg: dict) -> None:
    cap = cfg["drift"]["max_abs"]
    agent.theta_drift = min(cap, max(-cap, agent.theta_drift + delta))


def _drift_event(agent, reason: str, base_w: float, scale: float, cfg: dict) -> None:
    """出来事1件ぶんの馴化/鋭敏化。drift_p が None(=OFF)なら何もしない=バイト一致。

    馴化: 低顕著の単調入力(沈黙/SNS/同席/ニュース)を受けるたび閾値↑(頻度依存=毎回)。
    鋭敏化(脱馴化): 新奇/強イベント(初訪問/未知語/大 |Δstate|)で閾値↓。強さは
    base_w(語の顕著さ)or scale(|Δstate|)に比例。すべて k 非依存量(R1)。"""
    p = getattr(agent, "drift_p", None)
    if p is None:
        return
    if reason in _HABIT_REASONS:
        _drift_add(agent, p["habit_rate"] * base_w, cfg)
    elif reason == "state_change":
        _drift_add(agent, -p["sens_rate"] * scale, cfg)
    elif reason in _SENS_REASONS:
        _drift_add(agent, -p["sens_rate"] * base_w, cfg)


def _drift_recover(agent, cfg: dict) -> None:
    """自発回復: 毎 step theta_drift を 0 へ緩やかに引く。OFF なら no-op。"""
    p = getattr(agent, "drift_p", None)
    if p is None:
        return
    agent.theta_drift -= p["recovery_rate"] * agent.theta_drift


def _drift_fire(agent, cfg: dict) -> None:
    """発火経験そのものが次の発火を促す(練習効果=使用依存の鋭敏化)。OFF なら no-op。"""
    p = getattr(agent, "drift_p", None)
    if p is None:
        return
    _drift_add(agent, -p["sens_rate"], cfg)


def add(agent, reason: str, cfg: dict, scale: float = 1.0) -> None:
    """出来事によるゲージ加算。reason ごとの寄与も記録(発火時の文脈選択用)。

    個人別倍率プラグイン(いずれも既定 OFF、この関数は reason 文字列しか見ない=指紋なし):
    - SDT 動機(agent.drive_mods): 内発/外発の2軸倍率。
    - 欲求プロファイル(agent.needs_mods): 価値プロファイル由来の感度倍率(何を欲し・何を不快に感じるか)。
    両者は別属性で**乗算合成**。OFF 時は各属性が None(未設定)なので現行と完全同一(バイト一致)。
    """
    mods = getattr(agent, "drive_mods", None)
    if mods is not None:
        scale *= mods.get(reason, 1.0)
    nmods = getattr(agent, "needs_mods", None)
    if nmods is not None:
        scale *= nmods.get(reason, 1.0)
    # 価値の充足・飢餓(第17バッチ freedom・既定 OFF=None で不変): 飢えた価値に紐づく
    # 出来事ほど響く時変の感度倍率(values.py が算出。この関数は名前を知らない)。
    smods = getattr(agent, "sat_mods", None)
    if smods is not None:
        scale *= smods.get(reason, 1.0)
    base_w = cfg["weights"].get(reason, 0.0)
    w = base_w * scale
    if w <= 0.0:
        return
    agent.drive = min(1.0, agent.drive + w)
    agent._drive_reasons[reason] = agent._drive_reasons.get(reason, 0.0) + w
    # 無意識層(第12バッチ・既定 OFF=behav_today None で no-op): 当日の行動・経験カウント。
    bt = getattr(agent, "behav_today", None)
    if bt is not None:
        bt[reason] = bt.get(reason, 0) + 1
    # 内省ドリフト E2(既定 OFF=drift_p None で no-op)。出来事量のみ入力(R1)。
    _drift_event(agent, reason, base_w, scale, cfg)


def step_tick(agent, cfg: dict, step: int) -> None:
    """毎stepの自然減衰+沈黙の継続(考えていない時間がゆっくり欲求を押し上げる)。

    会話クールダウンが明けたら返答枠(conv_turns_left)を戻す=次の会話に備える。
    """
    agent.drive = max(0.0, agent.drive * (1.0 - cfg["decay"]))
    _drift_recover(agent, cfg)          # 自発回復(0へ)。OFF なら no-op=バイト一致
    add(agent, "silence", cfg)          # 沈黙=低顕著の単調入力 → 馴化(add 内で処理)
    if agent.conv_turns_left <= 0 and step >= agent.conv_cooldown_until:
        agent.conv_turns_left = cfg["conv_max_turns"]
    boredom_tick(agent, cfg, step)      # P2 S5: 退屈/好奇心ゲージ(既定 OFF=no-op=バイト一致)


def top_reason(agent) -> str:
    """ゲージへの寄与が最大の出来事(発火時の思考の「きっかけ」)。"""
    if not agent._drive_reasons:
        return "solo"
    return max(sorted(agent._drive_reasons), key=lambda k: agent._drive_reasons[k])


def on_fire(agent, step: int, cfg: dict) -> None:
    agent.drive *= cfg["fire_reset"]
    agent.refractory_until = step + cfg["refractory_steps"]
    agent._drive_reasons = {}
    _drift_fire(agent, cfg)             # 発火経験=練習効果(鋭敏化)。OFF なら no-op


def on_reject(agent, cfg: dict) -> None:
    """抽選に外れた: ゲージを数十%減衰(ユーザー仕様)。理由は保持(再申請に備える)。"""
    agent.drive *= (1.0 - cfg["fail_decay"])


# ---------------------------------------------------------------- 好奇心/退屈ドライブ P2 S5
# 内発的探索の種(interstitial-life.md §4.4-a)。silence ゲージと同じ「毎 step の物理入力で
# 溜まり・減衰する」作法。入力は物理量のみ(現在ノード・移動中か)=beliefs/k を一切見ない(R1)。
# 発火(探索行動)は routine 側(cognition/routine.py の _maybe_boredom_explore)が担い、
# ここはゲージの蓄積・閾値判定・リセットだけを持つ(silence/drive と drive.py に同居)。
# 既定 OFF(cfg['boredom']['enabled']=False)は agent に一切触れない=ゴールデン L1 バイト一致。
_BORE_UNSET = object()


def boredom_tick(agent, cfg: dict, step: int) -> None:
    """毎 step のゲージ更新: 同じ場所への長居・単調入力で蓄積、移動・新奇な場所で減衰。

    - 移動中(route あり)= 単調でない → 緩やかに減衰。
    - 同じノードに留まる(長居)→ accrual だけ蓄積(clip 1.0)。
    - 新しいノードに着いた → 減衰 + novelty_relief(脱馴化の入口。到達先で novel_place が
      起きれば既存 drive の鋭敏化=顕著性に自然接続する)。
    OFF or 初回観測は agent フィールドを触るだけで蓄積せず、OFF は完全 no-op(バイト一致)。"""
    bc = cfg["boredom"]
    if not bc["enabled"]:
        return
    node = getattr(agent, "node", None)
    last = getattr(agent, "_bore_node", _BORE_UNSET)
    if last is _BORE_UNSET:                     # 初回: ゲージ0で観測開始(まだ蓄積しない)
        agent._boredom = 0.0
        agent._bore_node = node
        agent._bore_cooldown = 0
        return
    cur = agent._boredom
    if getattr(agent, "route", None):          # 移動中=単調でない → 緩やかに減衰
        cur *= (1.0 - bc["decay"])
    elif node == last:                         # 同じ場所に留まる(長居)→ 蓄積
        cur = min(1.0, cur + bc["accrual"])
    else:                                      # 新しい場所に着いた → 新奇で減衰(脱馴化)
        cur = max(0.0, cur * (1.0 - bc["decay"]) - bc["novelty_relief"])
    agent._boredom = cur
    agent._bore_node = node


def boredom_ready(agent, cfg: dict, step: int) -> bool:
    """探索発火の候補か: ゲージが閾値超え & cooldown 明け。OFF/未観測なら False(=探索しない)。"""
    bc = cfg["boredom"]
    if not bc["enabled"]:
        return False
    return getattr(agent, "_boredom", 0.0) >= bc["threshold"] \
        and step >= getattr(agent, "_bore_cooldown", 0)


def boredom_fire(agent, cfg: dict, step: int) -> None:
    """探索へ動いた後の後処理: ゲージをリセット + cooldown を張る(連続探索の抑制)。"""
    bc = cfg["boredom"]
    agent._boredom = 0.0
    agent._bore_node = getattr(agent, "node", None)
    agent._bore_cooldown = step + bc["cooldown_steps"]
