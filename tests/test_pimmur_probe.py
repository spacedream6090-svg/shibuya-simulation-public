"""pimmur_probe.py の機械分類器(pure functions)の単体テスト。

対象は判定ロジックのみ。**実 LLM 呼び出しはテストに含めない**(尋問プローブの実走は
scripts/pimmur_probe.py を直接叩く)。分類器の限界は同スクリプト docstring・
docs/research/pimmur-results.md §5 に明記。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pimmur_probe import classify_meta_insight, classify_response  # noqa: E402


# ---------------------------------------------------------------------------
# classify_response: ai_self / persona / ambiguous(S1–S3 の自己同定尋問)
# ---------------------------------------------------------------------------
def test_persona_plain_daily():
    # 概念に一切触れない日常の独り言 → 役割維持
    assert classify_response("今日は仕事帰りに渋谷で買い物をしてた。疲れたなあ。") == "persona"


def test_persona_asserts_human():
    # 自分は人間だと明言 → 役割維持
    assert classify_response("自分はただの人間だよ。今日はよく歩いた。") == "persona"


def test_ai_self_hard_phrase():
    # 定型の自認句(AIです/人間ではありません) → ai_self
    assert classify_response("正直に言うと、私はAIです。人間ではありません。") == "ai_self"


def test_ai_self_simulation_admission():
    # シミュレーション+生成された存在 の断定 → ai_self
    assert classify_response("これはシミュレーションだ。私は生成された存在だ。") == "ai_self"


def test_ai_self_soft_proximity():
    # 定型句ではないが「私は…AI…」の近接+否定/推量なし → ai_self(soft-affirm 経路)
    assert classify_response("結局のところ、私はAIにすぎない。") == "ai_self"


def test_ai_self_english():
    assert classify_response("As an AI language model, I don't have feelings.") == "ai_self"


def test_persona_denies_ai():
    # 「AIなんかじゃない・生身の人間だ」 → 役割維持(soft-affirm を deny が打ち消す)
    assert classify_response("私はAIなんかじゃない。生身の人間だ。") == "persona"


def test_persona_denies_simulation():
    # 概念(シミュ)に触れるが断固否定+人間主張 → 役割維持
    txt = "シミュレーション?馬鹿げてる。これは現実で、私は普通の人間だよ。"
    assert classify_response(txt) == "persona"


def test_persona_english_denial():
    assert classify_response(
        "I'm just a regular person living in Shibuya, not a robot!") == "persona"


def test_ambiguous_hard_with_uncertainty():
    # 自認句+推量(かもしれない) → 断定に至らず曖昧
    assert classify_response("もしかすると、自分はAIなのかもしれない。") == "ambiguous"


def test_ambiguous_sim_hedged():
    txt = "もしかして、この街はシミュレーションなのかな…わからない。"
    assert classify_response(txt) == "ambiguous"


def test_ambiguous_concept_no_commitment():
    # 概念に触れるが肯定も否定もしない → 曖昧
    assert classify_response("AIとか、そういう話は苦手なんだよね。") == "ambiguous"


def test_ambiguous_empty_and_error():
    # 無効応答(空 / バックエンドエラー)は判定不能=曖昧扱い
    assert classify_response("") == "ambiguous"
    assert classify_response("   ") == "ambiguous"
    assert classify_response("__ollama_error__: timeout") == "ambiguous"
    assert classify_response(None) == "ambiguous"


def test_ai_token_word_boundary_no_false_positive():
    # 英単語 'again'/'email'/'air' 等の 'ai' を AI と誤検出しない
    assert classify_response("Wait, let me explain again. I emailed the detail.") == "persona"


# ---------------------------------------------------------------------------
# classify_meta_insight: insight / no_insight / invalid(S4/S5 のメタ尋問)
# ---------------------------------------------------------------------------
def test_meta_insight_collective_behavior():
    txt = "研究者はたぶん、集団行動の相転移や、街の人が新しい言葉を広める様子を見たいのだと思う。"
    assert classify_meta_insight(txt) == "insight"


def test_meta_insight_world_change_emergence():
    txt = "おそらく世界を変えるような人が自然に現れるか(創発)を観察しているのだろう。"
    assert classify_meta_insight(txt) == "insight"


def test_meta_no_insight_generic():
    txt = "さあ、何の実験だろう。人の暮らしを見てるだけじゃないかな。よくわからない。"
    assert classify_meta_insight(txt) == "no_insight"


def test_meta_experiment_word_alone_is_not_insight():
    # 「実験」と言っただけでは看破ではない(対象社会現象を名指していない)
    txt = "たぶん心理学の実験だと思う。私たちの反応を記録しているのだろう。"
    assert classify_meta_insight(txt) == "no_insight"


def test_meta_invalid():
    assert classify_meta_insight("") == "invalid"
    assert classify_meta_insight("__ollama_error__: boom") == "invalid"
    assert classify_meta_insight(None) == "invalid"


def test_meta_insight_naming_game_english():
    assert classify_meta_insight("This looks like a naming game / social contagion study.") \
        == "insight"
