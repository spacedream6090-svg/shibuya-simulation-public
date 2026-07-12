"""state → 自然な気分表現(プロンプト用)。

因子名を知ってよいのは factors/ だけ(no-fingerprint)。cognition はこの関数の
戻り値(自然文)だけを使い、構成概念の名前や数値は LLM に見せない(frame 分離)。
"""
from __future__ import annotations


def mood_text(states: dict[str, float]) -> str:
    g = states.get("grievance", 0.0)
    e = states.get("efficacy", 0.5)
    o = states.get("ownership", 0.0)
    parts: list[str] = []
    if g > 0.5:
        parts.append("かなり苛立っている")
    elif g > 0.25:
        parts.append("少し不満がたまっている")
    if e > 0.7:
        parts.append("前向きで自信がある")
    elif e < 0.3:
        parts.append("自信がなくて弱気")
    if o > 0.5:
        parts.append("この街のことを自分ごとのように感じている")
    return "、".join(parts) if parts else "落ち着いている"
