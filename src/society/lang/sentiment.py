"""日本語短文の感情価(valence)を決定論・純標準ライブラリで採点するモジュール。

辞書: 東北大学 乾・岡崎研究室「日本語評価極性辞書」(用言編+名詞編)を変換した
``REPO_ROOT/data/sentiment_lexicon.json`` を使用する。

  - 用言編: 小林のぞみ, 乾健太郎, 松本裕治, 立石健二, 福島俊一.
    意見抽出のための評価表現の収集. 自然言語処理, Vol.12, No.3, pp.203-222, 2005.
  - 名詞編: 東山昌彦, 乾健太郎, 松本裕治.
    述語の選択選好性に着目した名詞の評価極性の獲得.
    言語処理学会第14回年次大会論文集, pp.584-587, 2008.
  - 配布元: https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html
  - ライセンス: 上記クレジットを明記すれば商用利用・再配布可(公式配布条件)。

手法(LLM 不使用・形態素解析不使用・純 stdlib):
  1. 辞書語の最長一致・非重複な部分文字列マッチ(左から走査)。
  2. 否定反転: 一致語の直後〜6文字以内に否定辞(ない/ぬ/ません/なかった/
     じゃない/くない)があれば、その語の極性を反転。
  3. 強調: 一致語の直前が強調語(すごく/めっちゃ/最高に 等)なら重み 1.5 倍。
  4. 絵文字の簡易対応(😊😄👍=+ / 😢😡💢=−)。
  5. score = (Σpos − Σneg) / max(1, Σpos + Σneg) を [-1, 1] にクリップ。
     一致 0 件なら 0.0。

注意(既知の限界): 形態素解析を行わないため、活用語は表層の部分一致に依存する。
い形容詞は基本形に加え語幹(末尾「い」を除いた形)でもマッチさせて活用・否定に
備えるが(例:「楽しくない」→「楽し」に一致し否定反転)、稀に無関係語への
誤一致を生じ得る。皮肉・多義語(「ヤバい」等)は非対応。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

# --- 辞書ファイルの場所(REPO_ROOT/data/sentiment_lexicon.json)------------------
# src/society/lang/sentiment.py -> parents[3] がリポジトリルート。
REPO_ROOT = Path(__file__).resolve().parents[3]
LEXICON_PATH = REPO_ROOT / "data" / "sentiment_lexicon.json"

# --- 否定辞・強調語・絵文字(定数)------------------------------------------------
# 長い綴りを先に並べる(検出は in 判定なので順序非依存だが可読性のため)。
_NEGATIONS = ("じゃない", "くない", "なかった", "ません", "ない", "ぬ")
_NEG_WINDOW = 6  # 一致語の直後、何文字先までを否定辞の探索範囲にするか

_EMPHASIS = (
    "すっごく", "すごく", "すごい", "とっても", "とても",
    "めっちゃくちゃ", "めっちゃ", "めちゃくちゃ", "めちゃ",
    "最高に", "本当に", "ほんとに", "かなり", "非常に",
    "マジで", "まじで", "マジ", "まじ", "超", "激",
)
_EMPHASIS_WEIGHT = 1.5

_POS_EMOJI = frozenset("😊😄👍🙂😁🎉😍😆")
_NEG_EMOJI = frozenset("😢😡💢😠😭😞😔👎")

# --- 遅延ロードされる辞書状態(モジュール内で1回だけ構築)------------------------
_LEXICON: Dict[str, int] | None = None  # 語 -> +1(pos) / -1(neg)
_MAX_KEY_LEN: int = 0


def _load() -> Dict[str, int]:
    """辞書を1回だけロードして {語: 極性(+1/-1)} を返す。失敗時は空辞書。"""
    global _LEXICON, _MAX_KEY_LEN
    if _LEXICON is not None:
        return _LEXICON

    lex: Dict[str, int] = {}
    try:
        with open(LEXICON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for w in data.get("pos", ()):
            if isinstance(w, str) and len(w) >= 2:
                lex[w] = 1
        for w in data.get("neg", ()):
            if isinstance(w, str) and len(w) >= 2:
                lex.setdefault(w, -1)
        # い形容詞の語幹展開: 「Xい」(3文字以上)を「X」でもマッチ可能にし、
        # 活用形・否定形(楽しく/楽しかっ/楽しくない …)を部分一致で拾えるようにする。
        # 基本形が既にあるので、最長一致では基本形が優先される。
        for w, pol in list(lex.items()):
            if len(w) >= 3 and w.endswith("い"):
                lex.setdefault(w[:-1], pol)
    except Exception:
        # ロード失敗(ファイル無し/壊れ 等)は握りつぶし、空辞書で 0.0 を返す。
        lex = {}

    _LEXICON = lex
    _MAX_KEY_LEN = max((len(k) for k in lex), default=0)
    return _LEXICON


@lru_cache(maxsize=65536)
def valence(text: str) -> float:
    """日本語短文の感情価を [-1, 1] で返す(決定論・純標準ライブラリ)。

    正なら肯定的、負なら否定的、0.0 は中立(または感情語なし)。
    同一入力に対し常に同一値を返す。例外は送出しない。
    同一入力に冪等なため有界メモ化している(タイムライン順位付けが同一投稿を
    繰り返し採点する構造で呼数が桁で減る。辞書はプロセス内で1回ロード=不変)。
    """
    if not text:
        return 0.0
    lex = _load()
    if not lex:
        return 0.0

    max_len = _MAX_KEY_LEN
    n = len(text)
    sum_pos = 0.0
    sum_neg = 0.0

    i = 0
    while i < n:
        # 位置 i から始まる最長一致語を探す(2文字以上のみ; 1文字語は誤爆源)。
        matched = None
        hi = max_len if max_len < n - i else n - i
        for length in range(hi, 1, -1):
            sub = text[i:i + length]
            pol = lex.get(sub)
            if pol is not None:
                matched = (length, pol)
                break
        if matched is None:
            i += 1
            continue

        length, pol = matched
        end = i + length

        # 強調: 直前が強調語なら重み 1.5。
        weight = 1.0
        prefix = text[:i]
        for emp in _EMPHASIS:
            if prefix.endswith(emp):
                weight = _EMPHASIS_WEIGHT
                break

        # 否定反転: 直後〜6文字以内に否定辞があれば極性反転。
        tail = text[end:end + _NEG_WINDOW]
        for neg in _NEGATIONS:
            if neg in tail:
                pol = -pol
                break

        if pol > 0:
            sum_pos += weight
        else:
            sum_neg += weight

        i = end  # 非重複走査(一致語の後ろから続行)

    # 絵文字の簡易加点。
    for ch in text:
        if ch in _POS_EMOJI:
            sum_pos += 1.0
        elif ch in _NEG_EMOJI:
            sum_neg += 1.0

    total = sum_pos + sum_neg
    if total == 0.0:
        return 0.0

    score = (sum_pos - sum_neg) / (total if total > 1.0 else 1.0)
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score
