"""バッテリー指標(純関数のみ・I/O なし・乱数なし)。

方針:
  - すべて決定論。同じ入力なら同じ値(テストで既知値と突き合わせる)。
  - 形態素解析器に依存しない。日本語の語彙指標は**文字 n-gram**で測る
    (MeCab 等を足すと環境依存が増え、モデル比較の再現性が落ちるため)。
  - オラクルが無い量は「絶対評価しない」。対照(参照統計・プラセボ・他モデル)との
    差分でのみ主張する。この方針は scripts/analyze_specialization.py と同じ。
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------- 行動カテゴリ
# 社会生活基本調査「行動の種類(20分類)」に合わせる。参照統計と刺激プロンプトの
# 双方がこの並びを正典として使う(順序も固定=ベクトル比較の前提)。
ACTIVITY_CATEGORIES: tuple[str, ...] = (
    "睡眠",
    "身の回りの用事",
    "食事",
    "通勤・通学",
    "仕事",
    "学業",
    "家事",
    "介護・看護",
    "育児",
    "買い物",
    "移動",
    "テレビ・ラジオ・新聞・雑誌",
    "休養・くつろぎ",
    "学習・自己啓発・訓練",
    "趣味・娯楽",
    "スポーツ",
    "ボランティア活動・社会参加活動",
    "交際・付き合い",
    "受診・療養",
    "その他",
)
CATEGORY_INDEX = {c: i for i, c in enumerate(ACTIVITY_CATEGORIES)}

# 屋内で完結するカテゴリ(B層・雨摂動の方向妥当性判定に使う)
INDOOR_CATEGORIES: frozenset[str] = frozenset({
    "睡眠", "身の回りの用事", "食事", "家事", "介護・看護", "育児",
    "テレビ・ラジオ・新聞・雑誌", "休養・くつろぎ", "学習・自己啓発・訓練",
    "趣味・娯楽", "受診・療養",
})
OUTDOOR_CATEGORIES: frozenset[str] = frozenset({
    "スポーツ", "移動", "買い物", "ボランティア活動・社会参加活動",
})

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[。、!?…\.\,\!\?~ー―\-「」『』()\(\)]+")

# 表記ゆれ吸収(モデルは「通勤」「移動(通勤・通学以外)」等を平気で返す)。
# ★捨てずに寄せる: 未知語を "その他" に落とすと被覆率が正しく測れない。
ACTIVITY_ALIASES: dict[str, str] = {
    "通勤": "通勤・通学", "通学": "通勤・通学", "通勤通学": "通勤・通学",
    "外出": "移動", "移動・外出": "移動", "電車移動": "移動",
    "テレビ": "テレビ・ラジオ・新聞・雑誌", "ラジオ": "テレビ・ラジオ・新聞・雑誌",
    "新聞": "テレビ・ラジオ・新聞・雑誌", "読書": "趣味・娯楽",
    "テレビ・ラジオ": "テレビ・ラジオ・新聞・雑誌",
    "休養": "休養・くつろぎ", "くつろぎ": "休養・くつろぎ", "休憩": "休養・くつろぎ",
    "学習": "学習・自己啓発・訓練", "自己啓発": "学習・自己啓発・訓練",
    "勉強": "学習・自己啓発・訓練", "訓練": "学習・自己啓発・訓練",
    "趣味": "趣味・娯楽", "娯楽": "趣味・娯楽",
    "ボランティア": "ボランティア活動・社会参加活動",
    "社会参加": "ボランティア活動・社会参加活動",
    "交際": "交際・付き合い", "付き合い": "交際・付き合い", "友人と会う": "交際・付き合い",
    "受診": "受診・療養", "療養": "受診・療養", "通院": "受診・療養",
    "介護": "介護・看護", "看護": "介護・看護",
    "身の回り": "身の回りの用事", "身支度": "身の回りの用事", "入浴": "身の回りの用事",
    "洗面": "身の回りの用事", "支度": "身の回りの用事",
    "仕事": "仕事", "労働": "仕事", "業務": "仕事", "勤務": "仕事",
    "学校": "学業", "授業": "学業",
    "家事": "家事", "炊事": "家事", "掃除": "家事", "洗濯": "家事",
    "育児": "育児", "子育て": "育児",
    "買い物": "買い物", "買物": "買い物", "ショッピング": "買い物",
    "睡眠": "睡眠", "就寝": "睡眠", "仮眠": "睡眠", "昼寝": "睡眠",
    "食事": "食事", "朝食": "食事", "昼食": "食事", "夕食": "食事",
    "スポーツ": "スポーツ", "運動": "スポーツ", "散歩": "スポーツ",
}
_PAREN = re.compile(r"[（(][^）)]*[）)]")


def normalize_activity(raw: str) -> str:
    """モデルが返した行動名を ACTIVITY_CATEGORIES のいずれかに寄せる。

    順に: 完全一致 → 括弧書き除去して完全一致 → 別名表 → カテゴリの前方/部分一致
    → "その他"。決定論(辞書順に依存しない)。
    """
    s = norm_text(str(raw))
    if s in CATEGORY_INDEX:
        return s
    s2 = _WS.sub("", _PAREN.sub("", s))
    if s2 in CATEGORY_INDEX:
        return s2
    if s2 in ACTIVITY_ALIASES:
        return ACTIVITY_ALIASES[s2]
    for cat in ACTIVITY_CATEGORIES:          # 正典の並び順で走査=決定論
        if cat != "その他" and (s2.startswith(cat) or cat in s2):
            return cat
    for alias, cat in ACTIVITY_ALIASES.items():
        if alias and alias in s2:
            return cat
    return "その他"


def norm_text(text: str) -> str:
    """NFKC 正規化 + 前後空白除去。指標の入口を1本にする。"""
    return unicodedata.normalize("NFKC", text or "").strip()


# ---------------------------------------------------------------- 情報量系
def _probs(counts: Sequence[float]) -> list[float]:
    total = float(sum(counts))
    if total <= 0:
        return [0.0] * len(counts)
    return [max(0.0, float(c)) / total for c in counts]


def entropy(counts: Sequence[float], *, base: float = 2.0) -> float:
    """シャノンエントロピー(既定 bit)。"""
    h = 0.0
    for p in _probs(counts):
        if p > 0:
            h -= p * math.log(p, base)
    return h


def normalized_entropy(counts: Sequence[float], *, k: int | None = None) -> float:
    """H / log(k)。k 既定 = カテゴリ数(counts の長さ)。k<=1 なら 0。"""
    k = len(counts) if k is None else int(k)
    if k <= 1:
        return 0.0
    return entropy(counts) / math.log(k, 2.0)


def gini_simpson(counts: Sequence[float]) -> float:
    """1 - Σp^2。1つに集中で 0、均等分散で 1-1/k。"""
    return 1.0 - sum(p * p for p in _probs(counts))


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence(底2)。値域 [0,1]。完全一致=0、素性素で1。

    どちらも和で正規化してから計算する(生の度数を渡してよい)。
    """
    if len(p) != len(q):
        raise ValueError(f"長さ不一致: {len(p)} vs {len(q)}")
    pp, qq = _probs(p), _probs(q)
    if sum(pp) <= 0 or sum(qq) <= 0:
        raise ValueError("片側が全ゼロ。JSD は定義できない。")
    out = 0.0
    for a, b in zip(pp, qq):
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log(a / m, 2.0)
        if b > 0:
            out += 0.5 * b * math.log(b / m, 2.0)
    # 数値誤差で微小に [0,1] を外れることがあるのでクランプ
    return min(1.0, max(0.0, out))


# ---------------------------------------------------------------- 文字 n-gram
def char_ngrams(text: str, n: int = 2) -> list[str]:
    """空白を除いた正規化文字列の n-gram 列。"""
    s = _WS.sub("", norm_text(text))
    if n <= 0 or len(s) < n:
        return []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """ユニーク n-gram / 全 n-gram。1.0=全く重複なし。空なら 0.0。"""
    grams: list[str] = []
    for t in texts:
        grams.extend(char_ngrams(t, n))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_rate(texts: Iterable[str], n: int = 3) -> float:
    """n-gram 重複率 = 1 - distinct_n。退行(同じ言い回しの再生産)の指標。"""
    texts = list(texts)
    grams: list[str] = []
    for t in texts:
        grams.extend(char_ngrams(t, n))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def vocab_entropy(texts: Iterable[str], n: int = 2) -> float:
    """文字 n-gram 分布のエントロピー(bit)。語彙収縮の直接指標。"""
    c: Counter[str] = Counter()
    for t in texts:
        c.update(char_ngrams(t, n))
    if not c:
        return 0.0
    return entropy(list(c.values()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def mean_pairwise_distance(texts: Sequence[str], n: int = 2) -> float:
    """全ペアの (1 - Jaccard) 平均。応答の**多様性**。同一文だけなら 0。"""
    sets = [set(char_ngrams(t, n)) for t in texts]
    if len(sets) < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            tot += 1.0 - jaccard(sets[i], sets[j])
            cnt += 1
    return tot / cnt


# ---------------------------------------------------------------- 分位・分散
def quantile(values: Sequence[float], q: float) -> float:
    """線形補間の分位点(numpy の method='linear' と同一)。空はエラー。"""
    if not values:
        raise ValueError("空列の分位点は定義できない")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q は [0,1]: {q}")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def cv(values: Sequence[float]) -> float:
    """変動係数 sd/mean(母集団 sd)。mean<=0 なら 0.0。"""
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    if m <= 0:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / m


def variance_ratio(model_var: float, ref_var: float) -> float:
    """分散比 = モデル分散 / 参照分散。参照<=0 なら例外(黙って 0 を返さない)。"""
    if ref_var <= 0:
        raise ValueError("参照分散が 0 以下。分散比は定義できない。")
    return float(model_var) / float(ref_var)


# ---------------------------------------------------------------- 会話統計
# あいづち: 応答的で内容語をほぼ持たない短い発話。厳密な言語学的定義ではなく
# **粗い代理指標**であることを明記する(モデル間の相対比較にのみ使う)。
BACKCHANNEL_PATTERNS: tuple[str, ...] = (
    r"うん+", r"ええ", r"はい", r"へえ+", r"ふーん", r"ふうん", r"ほう",
    r"そう(だ|です)?ね", r"そうなんだ", r"そっか", r"そうか",
    r"なるほど", r"確かに", r"たしかに", r"わかる", r"だよね", r"ですね",
    r"あー", r"あぁ", r"おお+",
)
_BACKCHANNEL_RE = re.compile(
    r"^(?:" + "|".join(BACKCHANNEL_PATTERNS) + r")+$")
BACKCHANNEL_MAX_CHARS = 12

DISAGREE_PATTERNS: tuple[str, ...] = (
    r"^いや", r"いや、", r"でも", r"しかし", r"とはいえ", r"そうかな",
    r"違うと思", r"そうは思わない", r"賛成でき", r"反対", r"どうかな",
    r"ちょっと違", r"それは違", r"納得でき",
)
_DISAGREE_RE = re.compile("|".join(DISAGREE_PATTERNS))


def is_backchannel(utterance: str) -> bool:
    s = _PUNCT.sub("", _WS.sub("", norm_text(utterance)))
    if not s or len(s) > BACKCHANNEL_MAX_CHARS:
        return False
    return bool(_BACKCHANNEL_RE.match(s))


def is_disagreement(utterance: str) -> bool:
    return bool(_DISAGREE_RE.search(norm_text(utterance)))


def backchannel_rate(utterances: Sequence[str]) -> float:
    if not utterances:
        return 0.0
    return sum(1 for u in utterances if is_backchannel(u)) / len(utterances)


def disagreement_rate(utterances: Sequence[str]) -> float:
    if not utterances:
        return 0.0
    return sum(1 for u in utterances if is_disagreement(u)) / len(utterances)


_LATIN = re.compile(r"[A-Za-z]")


def latin_ratio(texts: Iterable[str]) -> float:
    """空白を除いた全文字に占めるラテン文字の割合。

    用途: **日本語で発話できているかの妥当性ゲート**。思考モデルが英語の前置き
    ("Okay, I need to ...")を本文に書くと、会話統計(発話長・語彙多様度)は
    見かけ上よく見えるのに測っているものが会話ではなくなる。実測で踏んだ。
    """
    total = 0
    latin = 0
    for t in texts:
        s = _WS.sub("", norm_text(t))
        total += len(s)
        latin += len(_LATIN.findall(s))
    return latin / total if total else 0.0


def utterance_lengths(utterances: Sequence[str]) -> list[int]:
    return [len(_WS.sub("", norm_text(u))) for u in utterances]


def topic_persistence(utterances: Sequence[str], n: int = 2) -> float:
    """隣接発話の文字 n-gram Jaccard 類似の平均。話題の持続。

    1.0 に張り付く=同じことを言い続けている(退行)、0 付近=毎回話題が飛ぶ。
    人間会話は中間にある、という仮説の検証用であって、高い方が良い指標ではない。
    """
    if len(utterances) < 2:
        return 0.0
    sets = [set(char_ngrams(u, n)) for u in utterances]
    vals = [jaccard(sets[i], sets[i + 1]) for i in range(len(sets) - 1)]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- スケジュール
def parse_hhmm(s: str) -> int:
    """'HH:MM' → 0時からの分。'24:00' は 1440 として受ける。"""
    s = norm_text(str(s)).replace("：", ":")
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        raise ValueError(f"時刻形式が不正: {s!r}")
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 24 and 0 <= mi < 60):
        raise ValueError(f"時刻が範囲外: {s!r}")
    return h * 60 + mi


def schedule_to_slots(blocks: Sequence[dict], *, slot_minutes: int = 15
                      ) -> tuple[list[str | None], int]:
    """ブロック列 → スロット配列(1日 1440/slot_minutes 個)と衝突スロット数。

    blocks: [{"start":"07:00","end":"08:00","activity":"食事"}, ...]
    - end < start は日跨ぎとみなし、24:00 で切って残りを 0:00 から埋める。
    - 先着優先(先のブロックが取ったスロットは上書きしない)。上書きしようとした
      回数を衝突数として返す = スケジュールの整合性の指標。
    - 未知カテゴリは "その他" に落とす(捨てない: 被覆率を正しく測るため)。
    """
    if 1440 % slot_minutes != 0:
        raise ValueError("slot_minutes は 1440 の約数であること")
    n = 1440 // slot_minutes
    slots: list[str | None] = [None] * n
    conflicts = 0
    for b in blocks:
        act = normalize_activity(b.get("activity", ""))
        try:
            st, en = parse_hhmm(b.get("start", "")), parse_hhmm(b.get("end", ""))
        except (ValueError, TypeError):
            continue
        spans = [(st, en)] if en > st else [(st, 1440), (0, en)]
        for a, z in spans:
            for i in range(a // slot_minutes, min(n, (z + slot_minutes - 1) // slot_minutes)):
                if slots[i] is None:
                    slots[i] = act
                elif slots[i] != act:
                    conflicts += 1
    return slots, conflicts


def slot_coverage(slots: Sequence[str | None]) -> float:
    if not slots:
        return 0.0
    return sum(1 for s in slots if s is not None) / len(slots)


def time_budget(slots: Sequence[str | None], *, slot_minutes: int = 15
                ) -> list[float]:
    """スロット配列 → カテゴリ別の分数ベクトル(ACTIVITY_CATEGORIES 順)。"""
    out = [0.0] * len(ACTIVITY_CATEGORIES)
    for s in slots:
        if s is None:
            continue
        out[CATEGORY_INDEX[s]] += slot_minutes
    return out


def wake_time(slots: Sequence[str | None], *, slot_minutes: int = 15,
              earliest_min: int = 180) -> float | None:
    """起床時刻(分)= earliest_min(既定 03:00)以降で最初に「睡眠でない」スロット。"""
    start = earliest_min // slot_minutes
    for i in range(start, len(slots)):
        if slots[i] is not None and slots[i] != "睡眠":
            return i * slot_minutes
    return None


def sleep_time(slots: Sequence[str | None], *, slot_minutes: int = 15,
               earliest_min: int = 1080) -> float | None:
    """就寝時刻(分)= earliest_min(既定 18:00)以降で最初の連続「睡眠」の開始。

    見つからない場合、翌日 0 時以降の先頭睡眠を 1440+ で返す(日跨ぎの素直な表現)。
    """
    start = earliest_min // slot_minutes
    for i in range(start, len(slots)):
        if slots[i] == "睡眠":
            return i * slot_minutes
    for i in range(0, start):
        if slots[i] == "睡眠":
            return 1440.0 + i * slot_minutes
    return None


def first_activity_time(slots: Sequence[str | None], activity: str, *,
                        slot_minutes: int = 15) -> float | None:
    for i, s in enumerate(slots):
        if s == activity:
            return i * slot_minutes
    return None


def quantile_gap(model_values: Sequence[float], ref_values: Sequence[float],
                 qs: Sequence[float] = (0.25, 0.5, 0.75)) -> dict:
    """分位差(分)。ref は参照統計の分位点そのもの(既に分位化済み)を渡す。

    返り値: {"q25": diff, "q50": diff, "q75": diff, "mean_abs": ...}
    ref_values は qs と同じ長さの「参照分位点」列。
    """
    if len(ref_values) != len(qs):
        raise ValueError("ref_values は qs と同じ長さで渡すこと")
    out: dict = {}
    diffs = []
    for q, rv in zip(qs, ref_values):
        d = quantile(model_values, q) - float(rv)
        out[f"q{int(round(q * 100))}"] = d
        diffs.append(abs(d))
    out["mean_abs"] = sum(diffs) / len(diffs)
    return out


# ---------------------------------------------------------------- JSON 抽出
def extract_json(text: str) -> dict | None:
    """モデル応答から最初の JSON オブジェクトを取り出す(前後の散文に耐える)。

    src/society/cognition/deliberate.py の寛容パースと同じ思想:
    途中で切れた JSON には閉じ括弧を足して1度だけ再試行する。失敗は None。
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    i = s.find("{")
    if i < 0:
        return None
    depth, in_str, esc, end = 0, False, False, -1
    for j in range(i, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    frag = s[i:end] if end > 0 else s[i:]
    for cand in (frag, frag + '"}', frag + "}", frag + "]}"):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------- 自己矛盾(簡易)
# 「私は X が好き/嫌い」型の素朴な抽出。E層の記憶整合の**簡易ルール**であり、
# 見逃しは多い(再現率は低い)。検出したものだけを報告する。
_PREF_RE = re.compile(
    r"(?:私|僕|自分|俺)(?:は)?[、,]?\s*([^。、,\.\n]{1,20}?)\s*(?:が|は)\s*"
    r"(好き|嫌い|苦手|得意)")
_OPPOSITE = {"好き": {"嫌い", "苦手"}, "嫌い": {"好き"},
             "苦手": {"好き", "得意"}, "得意": {"苦手"}}


def extract_self_claims(texts: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for t in texts:
        for m in _PREF_RE.finditer(norm_text(t)):
            out.append((m.group(1).strip(), m.group(2)))
    return out


def contradiction_count(texts: Iterable[str]) -> int:
    """同一対象に対する反対極性の主張の組数(簡易)。"""
    claims = extract_self_claims(texts)
    seen: dict[str, set[str]] = {}
    n = 0
    for obj, pol in claims:
        prev = seen.setdefault(obj, set())
        n += sum(1 for p in prev if pol in _OPPOSITE.get(p, set()))
        prev.add(pol)
    return n


# ---------------------------------------------------------------- 傾き
def slope(values: Sequence[float]) -> float:
    """等間隔系列の最小二乗の傾き(1 ステップあたりの変化)。長さ<2 は 0。"""
    xs = list(range(len(values)))
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(values) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (v - my) for x, v in zip(xs, values)) / den


def windows(seq: Sequence, k: int) -> list[list]:
    """列を k 個の連続窓に分ける(端数は末尾の窓へ)。k<=0 or 空は []。"""
    seq = list(seq)
    if k <= 0 or not seq:
        return []
    k = min(k, len(seq))
    size = len(seq) // k
    out = []
    for i in range(k):
        a = i * size
        z = len(seq) if i == k - 1 else (i + 1) * size
        out.append(seq[a:z])
    return out


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))
