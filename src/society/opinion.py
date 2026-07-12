"""意見力学(Friedkin-Johnsen)= cheap tier の非LLM 意見更新則(#16、ユーザー採用)。

文献(docs/lit): 複数報告で **FJ(初期意見 anchor 付き重み付き平均)が LLM の
意見更新挙動と最も一致**する。ここでは engine/cognition から呼べる純関数だけを置く。

★ このモジュールを src/society/ 直下(tests/test_contracts の CHECKED_DIRS 外)に
  置く理由: registry の写像根拠に触れず、engine からは因子名を一切見せない設計だが、
  意見更新自体は observer 指標(他者の認知を操作した量)の素材であり、engine の
  発話・DM・SNS 閲覧の 3 接点から呼ばれる。因子語を含めないため契約検査対象外でよい。

更新則(決定論・乱数不使用):
    o ← (1−s)·anchor + s·((1−w)·o + w·v)
  s = agent.opinion_susceptibility(社会的影響の受けやすさ)
  anchor = agent.opinion_anchor(FJ の stubbornness=初期意見への引き戻し)
  w = チャネル別の説得重み(対面 > DM > SNS 閲覧)
  v = expressed_value(聞いた/読んだテキストの感情価 valence)
  結果は [−1, 1] にクリップ。
"""
from __future__ import annotations


def expose(agent, expressed_value: float, w: float) -> float:
    """1 回の曝露に対する新しい意見値(純関数・agent は変更しない)。

    o ← (1−s)·anchor + s·((1−w)·o + w·v)、[−1,1] clip。
    """
    s = float(agent.opinion_susceptibility)
    anchor = float(agent.opinion_anchor)
    o = float(agent.opinion)
    v = float(expressed_value)
    social = (1.0 - w) * o + w * v
    new = (1.0 - s) * anchor + s * social
    if new < -1.0:
        return -1.0
    if new > 1.0:
        return 1.0
    return new


def alignment(a: float, b: float) -> float:
    """2つの意見値(ともに [−1, 1])の整合度。1=完全一致 / 0=正反対(決定論・純関数)。

    ★ 情報環境の非対称(Wave G6)の推薦=エコーチェンバーのランキング素材。閲覧者の意見 a と
      投稿の価 b の近さを不透明スコアに変換する。ここに集約する理由(no-fingerprint 方針):
      推薦バイアスは「意見の近さ」を参照するが、その参照は opinion.py 経由(この純関数)で
      不透明化し、engine/net には生の意見差でなく [0,1] の整合スコアだけを見せる。"""
    d = abs(float(a) - float(b))
    if d > 2.0:
        d = 2.0
    return 1.0 - d / 2.0
