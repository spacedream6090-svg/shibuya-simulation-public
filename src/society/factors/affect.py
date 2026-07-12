"""感情・興味・注意の統合ハブ(最小版)。設計正典: docs/lit/neuroscience__emotion-interest-attention.md §5.1/§6。

脳科学の核(Russell 1980 core affect / Menon&Uddin 2010 salience network): 感情・興味・注意は
独立モジュールではなく、**覚醒度 arousal(1スカラ)+ 顕著さ salience(1関数)** を共有した1回路系。
本 module は非LLM・決定論・RNG不要の factors 層に、その計算を集約する:
  1. arousal 更新則(観測イベント→相動性上昇 / 毎step baseline へ漏れ減衰。§1・§5.1)
  2. novelty/arousal → memory.importance 加点(§1.6 LaBar&Cabeza / §2.3 Gruber+ 2014)
  3. salience スコア(bottom-up + top-down。§3 Corbetta&Shulman / Cowan / Vuilleumier)
  4. 発火 effective_threshold の逆U字変調(§6.3 Yerkes-Dodson 1908)

no-fingerprint: この module だけが valence/novelty/needs 次元を参照する。engine/cognition/world へは
**不透明な数値**(arousal スカラ・salience の float・gain・K)だけを渡す。
既定 OFF(enabled=false・全 gain 0・K 0)で現行挙動とバイト一致(engine 側が enabled で分岐し新経路を
一切通さない)。最小版は全て決定論=乱数を引かない(新 stream すら不要)。
"""
from __future__ import annotations

# salience スコアの内部重み(bottom-up + top-down。向き=因果構造のみ固定、K/gain は conf で調律)。
# bottom-up: novelty(Berlyne collative / Loewenstein gap)+ |valence|(Vuilleumier: 脅威が注意を奪う)
#          + addressed(社会的顕著)+ arousal 利得(salience network の島皮質→自律=arousal 結合)。
# top-down: needs 関連(不透明倍率)+ 当日目標語(Corbetta&Shulman の背側 goal-driven)。
_SAL_W_NOVELTY = 1.0
_SAL_W_VALENCE = 1.0
_SAL_W_ADDRESSED = 1.0
_SAL_W_AROUSAL = 0.5
_SAL_W_TOPDOWN = 0.5


def build_cfg(raw: dict | None) -> dict:
    """conf の affect ブロックを型強制つきで正準化する(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "arousal_baseline": float(raw.get("arousal_baseline", 0.2)),
        "arousal_gain": float(raw.get("arousal_gain", 0.0)),      # 観測イベント→覚醒の上がりやすさ
        "arousal_decay": float(raw.get("arousal_decay", 0.1)),    # 毎step baseline へ戻る率
        "importance_arousal_w": float(raw.get("importance_arousal_w", 0.0)),
        "importance_novelty_w": float(raw.get("importance_novelty_w", 0.0)),
        "salience_k": int(raw.get("salience_k", 0)),              # 0=無効(全通し)。>0 で上位K
        "threshold_gain": float(raw.get("threshold_gain", 0.0)),  # 逆U字の効き幅(0=恒等)
    }


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------- arousal(§1・§5.1)
def arousal_signal(*, valence_abs: float = 0.0, novelty: float = 0.0,
                   addressed: float = 0.0, congestion: float = 0.0) -> float:
    """観測イベント1件の覚醒信号(0..1 目安)。|valence|・novelty・被話しかけ・混雑の和。

    すべて**観測量**(R1 安全: belief/k を参照しない)。個々の寄与は [0,1] 想定で、
    合算後に呼び出し側が gain を掛ける(漏れ積分器の相動性入力)。
    """
    return (abs(float(valence_abs)) + float(novelty)
            + float(addressed) + float(congestion))


def bump(arousal: float, signal: float, cfg: dict) -> float:
    """漏れ積分器の相動性上昇: arousal += gain·signal(clip[0,1])。gain=0 で恒等(バイト一致)。"""
    return _clip01(arousal + cfg["arousal_gain"] * float(signal))


def decay(agent, cfg: dict) -> None:
    """毎 step、baseline へ緩やかに戻す(漏れ)。decay=0 or 既に baseline なら実質不変。

    drive ゲージの step_tick と同型の減衰。arousal は drive と**並列の内部 transient**として持ち、
    states 監査集合(efficacy/grievance/ownership)には載せない(§6.3、R²(k) を汚さない)。
    """
    a0 = cfg["arousal_baseline"]
    a = getattr(agent, "arousal", a0)
    agent.arousal = _clip01(a + cfg["arousal_decay"] * (a0 - a))


# ---------------------------------------------------------------- 記憶符号化(§1.6・§2.3)
def importance_bonus(agent, cfg: dict, *, novelty: float = 0.0) -> float:
    """観測時の memory.importance 加点 = w_a·arousal + w_n·novelty(clip は memory 側で 1–10)。

    高覚醒・高好奇心の出来事を顕著記憶として底上げする(LaBar&Cabeza 2006 / Gruber+ 2014)。
    既定 w=0 で 0(=固定 importance 3.0 のまま=現行と一致)。
    """
    a = getattr(agent, "arousal", cfg["arousal_baseline"])
    return cfg["importance_arousal_w"] * a + cfg["importance_novelty_w"] * float(novelty)


# ---------------------------------------------------------------- 注意 salience(§3)
def _needs_relevance(agent) -> float:
    """top-down 関連度を needs_mods(**不透明倍率**)経由で拾う(価値名は見ない)。

    needs OFF(needs_mods 不在)なら 0(top-down 寄与なし)。倍率 1.0 中心の平均偏差を関連度の
    近似に使う(注意の構え=needs_profile 由来。no-fingerprint: 次元名を factors の外に出さない)。
    """
    mods = getattr(agent, "needs_mods", None)
    if not mods:
        return 0.0
    vals = list(mods.values())
    return (sum(vals) / len(vals)) - 1.0 if vals else 0.0


def salience_score(agent, cfg: dict, *, novelty: float = 0.0,
                   valence_abs: float = 0.0, addressed: float = 0.0,
                   goal_match: float = 0.0) -> float:
    """知覚 item の顕著さ(**不透明 float**)。bottom-up(novelty+|valence|+addressed+arousal 利得)
    + top-down(needs 関連 + 当日目標語)。決定論・RNG 不要。engine には返り値の float だけ渡す。"""
    a = getattr(agent, "arousal", cfg["arousal_baseline"])
    return (_SAL_W_NOVELTY * float(novelty)
            + _SAL_W_VALENCE * abs(float(valence_abs))
            + _SAL_W_ADDRESSED * float(addressed)
            + _SAL_W_AROUSAL * a
            + _SAL_W_TOPDOWN * (_needs_relevance(agent) + float(goal_match)))


# ---------------------------------------------------------------- 発火閾値の逆U字(§6.3)
def threshold_delta(agent, cfg: dict) -> float:
    """発火 effective_threshold への逆U字変調量(Yerkes-Dodson 1908)。

    中覚醒で閾値↓(行動しやすい=最適覚醒)・低/過覚醒で↑(退屈/狭窄)。gain=0 で 0(恒等=バイト一致)。
    値は drive.effective_threshold 側の clip[0.30, 0.85] に合流する(不透明な delta のみ返す)。
    """
    g = cfg["threshold_gain"]
    if g == 0.0:
        return 0.0
    a = getattr(agent, "arousal", cfg["arousal_baseline"])
    inverted_u = 4.0 * a * (1.0 - a)          # [0,1]、a=0.5 で peak 1
    return g * (0.5 - inverted_u)             # 中覚醒→負(閾値↓)、両端→正(閾値↑)
