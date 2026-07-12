"""日本語感情価スコアラ(society.lang.sentiment.valence)のユニットテスト。"""
from __future__ import annotations

from society.lang.sentiment import valence


def test_positive_sentence():
    # ポジ文は正
    assert valence("今日は最高に楽しい") > 0.0


def test_negative_sentence():
    # ネガ文は負
    assert valence("最悪だ、うんざりする") < 0.0


def test_negation_flips_polarity():
    # 否定反転: 肯定語+否定辞 → 負
    assert valence("楽しくない") < 0.0


def test_neutral_sentence_is_zero():
    # 感情語を含まない中立文は 0.0
    assert valence("駅に着いた") == 0.0


def test_deterministic():
    # 決定論: 同一入力を2回評価しても同値
    t = "渋谷の混雑にうんざりしたが、宮下公園はとても気持ちよかった😊"
    assert valence(t) == valence(t)


def test_range_within_bounds():
    # 出力は常に [-1, 1]
    samples = [
        "今日は最高に楽しい",
        "最悪だ、うんざりする",
        "楽しくない",
        "駅に着いた",
        "",
        "すごく嬉しい😄👍",
        "もう疲れた、悲しい😢💢",
        "彼はとても親切で優しい素晴らしい人だ",
        "ぐぬぬ",
    ]
    for t in samples:
        v = valence(t)
        assert -1.0 <= v <= 1.0


def test_empty_and_whitespace_are_zero():
    # 空文字・空白のみは 0.0(例外を出さない)
    assert valence("") == 0.0
    assert valence("　　") == 0.0


def test_emphasis_and_emoji_positive():
    # 強調語+ポジ絵文字は正になり、素の肯定文以上に振れる
    assert valence("すごく嬉しい😊") > 0.0


def test_negation_recovers_negative_word():
    # ネガ語+否定 → 肯定側(「問題はない」は好ましい状況)
    assert valence("特に問題はない") > 0.0
