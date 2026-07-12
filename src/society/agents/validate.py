"""ペルソナ文の注入/評価分離バリデータ(D6)。

構成概念(D5 の trait/state)の名前がペルソナ文に混入すると、LLM の
ロールプレイ増幅(R4)と評価の循環を招く。生成時に必ず検査する。
"""
from __future__ import annotations

CONSTRUCT_BLACKLIST = [
    # D5 state
    "効力感", "自己効力", "不満がたまりやすい", "当事者意識",
    # D5 trait
    "認知欲求", "リスク許容", "内的統制", "統制の所在",
    # 心理尺度の匂いが強い語(控えめ翻訳の原則)
    "外向性", "神経症傾向", "開放性", "誠実性", "協調性",
    "efficacy", "grievance", "ownership", "locus of control",
]


def construct_violations(text: str) -> list[str]:
    """ペルソナ文に混入した構成概念語を返す(空なら合格)。"""
    return [w for w in CONSTRUCT_BLACKLIST if w.lower() in text.lower()]
