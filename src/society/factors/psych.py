"""行動心理プラグイン(config 着脱式・既定 OFF。ユーザー方針 2026-07-06)。

文献の数値を固定実装せず、**回しながら柔軟に変える**ための seam をまとめる:
  - SDT 動機(psych.sdt)        : 内発/外発の欲求ゲージ倍率(registry.sdt_params)
  - 集団効力感(psych.collective): 新 state collective_efficacy(enable_collective)
  - Lynch 都市イメージ(psych.lynch): landmark 重み・観測レンズ(engine 側で構築)
  - Searle 制度化(psych.searle) : 成立提案の制度化・プロンプト注入(tools 側)

原則(全プラグイン共通):
- **既定 OFF = OFF 時はイベント列がバイト一致で不変**(engine/cognition は enabled で分岐)。
- no-fingerprint: factors だけが trait/state 名を知る。engine には不透明な倍率・スコアだけ渡す。
- R9: update 則はイベントのみ参照。R1: 非 LLM(呼数を変えない)。
"""
from __future__ import annotations

from . import registry


def build_cfg(raw: dict | None) -> dict:
    """conf の psych ブロックを型強制つきで正準化する(全プラグイン既定 OFF)。"""
    raw = dict(raw or {})
    sdt = dict(raw.get("sdt", {}) or {})
    col = dict(raw.get("collective", {}) or {})
    lyn = dict(raw.get("lynch", {}) or {})
    sea = dict(raw.get("searle", {}) or {})
    return {
        "sdt": {
            "enabled": bool(sdt.get("enabled", False)),
            "intrinsic_boost": float(sdt.get("intrinsic_boost", 1.4)),
            "extrinsic_boost": float(sdt.get("extrinsic_boost", 1.4)),
        },
        "collective": {
            "enabled": bool(col.get("enabled", False)),
            "group_success_magnitude": float(
                col.get("group_success_magnitude", 0.05)),
            "ingroup_boost": float(col.get("ingroup_boost", 1.2)),
            "attend_threshold": int(col.get("attend_threshold", 2)),
        },
        "lynch": {
            "enabled": bool(lyn.get("enabled", False)),
            "landmark_weight": float(lyn.get("landmark_weight", 1.0)),
        },
        "searle": {
            "enabled": bool(sea.get("enabled", False)),
            "max_recent": int(sea.get("max_recent", 2)),
        },
    }


def ensure_collective(agent, rng) -> bool:
    """集団効力感の state を**まだ持っていなければ**据える(冪等。レーン乙 A5)。

    ★因子名を知ってよいのは factors 層だけなので、「もう据わっているか」の判定も
      engine ではなくここに置く(engine 側は真偽しか受け取らない = 指紋の禁止を守る)。
      プール回転で途中入場する個体にも配るために要る(起動時 1 回では届かない)。
    戻り値 = 新たに据えたか。"""
    if "collective_efficacy" in agent.states:
        return False
    enable_collective(agent, rng)
    return True


def enable_collective(agent, rng) -> None:
    """集団効力感プラグイン ON 時のみ、agent に新 state を1つ追加する。

    ★ states dict への追加は ON 時だけ(OFF 時の snapshot・L2 を不変に保つため)。
    因子名(collective_efficacy)を知ってよいのは factors だけ = engine は本関数を呼ぶだけ。
    """
    agent.states["collective_efficacy"] = registry.collective_init(agent.traits, rng)
