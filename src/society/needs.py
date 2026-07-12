"""欲求プロファイルの個人差プラグイン(config 着脱式・既定 OFF。Wave2-2)。

目的: エージェントごとに「**何を欲し・何を不快と感じるか**」を分散させる。既存の欲求駆動発火
(cognition/drive)のゲージ入力(novel_place / congestion / unknown_word / ...)に対する
**個人別の感度倍率**(reason → multiplier)を、確立した個人差理論から決める。学術接地:
docs/lit/needs__individual-differences.md(SDT 基本欲求 / Schwartz 基本価値 / Reiss 16 欲求 /
Zuckerman 刺激欲求・Aron 感覚処理感受性)。

レイヤ分担(no-fingerprint):
- factors/registry.needs_profile: trait/年齢/職業 → 潜在価値プロファイル(5 次元)を生成(価値名を知る層)。
- 本 module: プロファイル → **不透明な倍率 dict**(reason 文字列だけを鍵に持つ)へ変換。
- cognition/drive.add: agent.needs_mods を reason 文字列で引くだけ(価値名・次元名を知らない)。

原則:
- **既定 OFF = OFF 時は needs_mods 属性が無い(=None)→ drive の挙動はバイト一致で不変**。
- **R1(k 非依存)**: プロファイルは trait/年齢/職業 + needs 専用乱数のみ由来。belief 書き戻し・
  k.writeback を参照しない → 同じ seed/agent なら k を変えても同一プロファイル。
- **psych.sdt との非衝突**: SDT は drive_mods(内発/外発の2軸)、needs は needs_mods(価値プロファイル)。
  別属性なので drive.add で**乗算合成**でき、既定は needs 単独(sdt 既定 OFF)。

steward への統合メモ(engine/simulation.py への 3 行、psych.sdt ブロックと同型):
    if self.needscfg["enabled"]:
        from ..needs import precompute as needs_precompute
        needs_precompute(self.agents, self.hub, self.needscfg)
"""
from __future__ import annotations

from .factors import registry

# reason(ゲージ入力)ごとの、潜在価値次元に対する感度係数(向き=符号)。
# 大きさは cfg.sensitivity_gain で一括調律(決め打ちしない)。負の入力(congestion / silence)は
# **倍率↑=より速くゲージが溜まる=より不快に感じる**(=「何を不快と感じるか」の個体差)。
# 次元キーは registry.NEEDS_DIMS。この module は reason 文字列を鍵に持つ倍率 dict しか外へ出さない。
_REASON_SENSITIVITY: dict[str, dict[str, float]] = {
    "novel_place":  {"stimulation": 1.0, "autonomy": 0.5, "security": -0.6},
    "unknown_word": {"stimulation": 0.7, "competence": 0.8},
    "congestion":   {"security": 1.0, "stimulation": -0.7},   # 不快入力(安全/高感受性で強い)
    "addressed":    {"relatedness": 1.0},
    "dm_received":  {"relatedness": 0.8},
    "company":      {"relatedness": 0.7, "autonomy": -0.4},
    "silence":      {"stimulation": 0.6, "relatedness": 0.5}, # 不快入力(退屈/孤独の感受性)
    "news":         {"competence": 0.4, "security": 0.5},
    "sns":          {"relatedness": 0.5, "stimulation": 0.3},
    "state_change": {"competence": 0.4},
}


def build_cfg(raw: dict | None) -> dict:
    """conf の needs ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "sensitivity_gain": float(raw.get("sensitivity_gain", 0.6)),  # 倍率の効き幅
        "profile_sigma": float(raw.get("profile_sigma", 0.10)),       # 個体固有ゆらぎ
        "occupation_weight": float(raw.get("occupation_weight", 1.0)),# 職業ニュアンスの強さ
        "clip_lo": float(raw.get("clip_lo", 0.5)),                    # 倍率下限
        "clip_hi": float(raw.get("clip_hi", 2.0)),                    # 倍率上限
    }


def mods_from_profile(profile: dict[str, float], cfg: dict) -> dict[str, float]:
    """潜在価値プロファイル(5 次元)→ reason 別の**個人別ゲージ倍率**(不透明 dict)。

    mult(reason) = clip(1 + gain · Σ_dim coef[reason][dim] · (profile[dim]−0.5)·2, [lo, hi])。
    profile[dim] が中心 0.5 のとき全 reason で倍率 1.0(=現行と一致)。
    """
    gain = cfg["sensitivity_gain"]
    lo, hi = cfg["clip_lo"], cfg["clip_hi"]
    mods: dict[str, float] = {}
    for reason, coeffs in _REASON_SENSITIVITY.items():
        acc = 0.0
        for dim, coef in coeffs.items():
            acc += coef * (float(profile.get(dim, 0.5)) - 0.5) * 2.0
        mult = 1.0 + gain * acc
        mods[reason] = round(float(min(hi, max(lo, mult))), 4)
    return mods


def build_mods(agent, rng, cfg: dict) -> dict[str, float]:
    """1 エージェント分: プロファイル生成(factors)→ 倍率変換。プロファイルも観測用に付す。"""
    profile = registry.needs_profile(agent.traits, agent.age, agent.occupation,
                                     rng, cfg)
    agent._needs_profile = profile          # 観測用(価値プロファイルの分散確認)。engine は読まない
    return mods_from_profile(profile, cfg)


def precompute(agents, hub, cfg: dict) -> None:
    """有効時、各エージェントに needs_mods(reason→倍率)を付与する。専用 stream "needs" で
    既存 draw 順を汚さない(psych.sdt が "sdt" stream を使うのと同型)。OFF 時は呼ばれない。"""
    for a in agents:
        a.needs_mods = build_mods(a, hub.stream("needs", a.id), cfg)
