"""V-P1: プロンプトの一貫性修正(purpose 別ヘッダ + 規律 3 行)。**既定 OFF = バイト一致**。

正典: docs/research/dialogue-coherence-and-model.md
      §1.3(指示矛盾の実ジャーナル解剖)/ §3(参照実装との差分表)/ §6.1(改修候補 P1)/
      §9-1(段階案の第 1 段 =「即日・低リスク」)。

何が壊れていたか(実ジャーナルの一次データ)
--------------------------------------------
`deliberate.build_prompt` は**発話・計画・内省・recall の全経路で共有**され、冒頭に必ず
行動メニューを置く:

    出力は次のいずれかの JSON 1個のみ(キー名は厳守):
      {"action": "speak", ...} / {"action": "post", ...} / {"action": "wander"} …

reflect / plan / recall はその後ろに**別の**「出力は次の JSON 1個のみ」を足す
(`reflection._REFLECT_TASK` / `planning._PLAN_TASK` / `day_plan.schema_prompt`)。つまり
**1 プロンプトに相互排他の出力指示が 2 つ**あり、モデルは「後勝ち」を期待されている
(runs/night_llm_100a3d のジャーナル現物で確認済み)。8B 級では概ね後勝ちするが、
  (a) 指示追従の予算を無駄に食う、
  (b) 行動メニューの語彙(speak / post …)が plan / reflect の自由文欄へ混入する誘因になる。
加えて、参照実装(Generative Agents / ai-town / AgentSociety / TinyTroupe)がいずれも持つ
「ペルソナ優先・知らないことは知らない・直前と同じことを繰り返さない」型の**規律文言**が
本リポには無い(反復注意は `said` が在るときの 1 行だけ)。

本 module が足すもの(2 つだけ)
--------------------------------
① **purpose 別ヘッダ** … reflect / plan / recall のヘッダから行動メニューを外し、
   「あなたは〈街〉の街で暮らす一人の人間です。…」の 1 行だけにする。結果、その purpose の
   出力指示は**ちょうど 1 つ**になる。**deliberate のヘッダは 1 バイトも変えない**
   (行動メニューは deliberate の本体そのものだから)。
② **規律 3 行** … ペルソナ直後に固定注入する中立の 3 行(TinyTroupe の
   `tiny_person.mustache` が最も明示的に持つ一貫性文言の、日本語・最小版)。
   位置がペルソナ直後である根拠は Lost in the Middle(arXiv:2307.03172)=
   守らせたい指示は文脈の**先頭か末尾**に置く。全 purpose 共通。

R1 ドクトリン / ゴールデンとの関係
----------------------------------
- 既定 `prompts.p1.enabled: false` では `purpose_for()` が **None** を返し、
  `build_prompt` は分岐に 1 度も入らない(= 出力はバイト一致 = ゴールデン
  `tests/data/golden_baseline_l1.json` 不変)。
- ON で変わるのは**プロンプト文字列だけ**である: `generate()` の呼び出し点は 1 つも増減せず、
  乱数を 1 本も引かず、発火判断にも一切触れない(= 呼数・rng 消費・発火列は ON/OFF で完全一致)。
  応答固定バックエンドで L1 がバイト一致することを tests/test_prompt_p1.py が機械固定する。
- no-fingerprint: 機構語(発火・閾値・驚き・条件・モデル・シミュレーション)も因子名も
  実験条件語も 1 語も書かない。書くのは「人物として書くときの構え」だけ。

正直な限界
----------
- ヘッダを変えると、その purpose の APC(prefix cache)prefix は現行と別物になる。効くのは
  reflect / plan / recall(呼数の約 7%)だけで、deliberate(約 93%)の prefix は不変。
- `ablate.prompt_paraphrase`(第92)の凍結置換表は**現行ヘッダ定数**を置換元に持つので、
  P1 ON の reflect / plan / recall ヘッダは置換対象に入らない(言い換えが薄くなる)。
  両者は「本選 ON」と「アブレーション専用」で用途が排他なので実害はないが、併用するなら
  置換表側に P1 ヘッダを足す必要がある(未実施 = 既知の限界として明記する)。
- 本 module は「入力の作り」だけを直す。§9 の中心命題どおり、逐語記憶の再注入ループ(P2)や
  会話スレッド化は**別の段**であって、ここでは 1 バイトも触らない。
"""
from __future__ import annotations

import re

SCHEMA = 1

# 受理する purpose(= プロンプトを組む 4 経路)。ここに無い値は組み立て時に即エラーにする
# (綴り違いが「黙って OFF」になるのを防ぐ)。
PURPOSES: tuple[str, ...] = ("deliberate", "plan", "reflect", "recall")

# --------------------------------------------------------------------------- #
# ① purpose 別ヘッダ
#
# 1 文目は現行 `deliberate._HEADER_INTRO` の 1 文目と**一字一句同一**にする(街の名前は
# envpack 由来のまま = 基盤に地名リテラルを残さない)。変えるのは 2 文目だけ。
# --------------------------------------------------------------------------- #
_INTRO = "あなたは{city}の街で暮らす一人の人間です。"
_TAIL: dict[str, str] = {
    "plan": "これから書くのは、あなた自身の今日の過ごし方です。",
    "reflect": "これから書くのは、あなた自身の振り返りです。",
    "recall": "これから書くのは、あなた自身が思い出したいことです。",
}

# --------------------------------------------------------------------------- #
# ② 規律 3 行(ペルソナ直後に固定注入。全 purpose 共通)
#
# 文言の出所は TinyTroupe `tinytroupe/agent/prompts/tiny_person.mustache`:
#   「persona characteristics ALWAYS OVERRIDE ANY BUILT-IN CHARACTERISTICS」
#   「ペルソナが知らないはずのことは知らないふりをする(捏造禁止)」
#   「直前と同一の行動を繰り返さない」
# ★ 1 行目は「**自分の地の傾向より**優先する」と書く(「他のどの指示より優先」と書くと
#   直後の出力書式の指示と衝突しうる = 本 module が消しに来た当の矛盾を作ってしまう)。
# ★ 3 行目は現行の `said` 注意行(発話経路のみ・条件つき)の常設・中立版。
DISCIPLINE: tuple[str, ...] = (
    "人物設定(すぐ上の自己紹介)を、いつでも自分の地の傾向より優先する。",
    "覚えていないこと・ここに書かれていないことは作らない。"
    "知らないことは知らないままでよい。",
    "直前に自分が言った・書いたことと同じ話題や言い回しをなぞらない。",
)


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `prompts.p1` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {"enabled": bool(raw.get("enabled", False))}


def _sub(cfg, *path):
    """OmegaConf / dict どちらでも辿れる安全な取り出し(欠けていれば None)。"""
    node = cfg
    for key in path:
        if node is None:
            return None
        try:
            node = node.get(key, None)          # DictConfig も Mapping も .get を持つ
        except (AttributeError, TypeError):
            return None
    return node


def cfg_of(sim) -> dict:
    """`sim.cfg.prompts.p1` を正準化して返す(未宣言・sim スタブでも安全に OFF)。"""
    return build_cfg(_sub(getattr(sim, "cfg", None), "prompts", "p1"))


def enabled(sim) -> bool:
    """P1 が有効か。既定 OFF。"""
    return bool(cfg_of(sim)["enabled"])


def purpose_for(sim, purpose: str) -> str | None:
    """`build_prompt` へ渡す `p1_purpose`。**OFF は None**(= 呼び出し側が分岐しない)。"""
    if purpose not in PURPOSES:
        raise ValueError(f"未知の purpose: {purpose!r}(既知: {PURPOSES})")
    return purpose if enabled(sim) else None


# --------------------------------------------------------------------------- #
# 出力上限・温度の seam(purpose 別。既定 None/0 = 現行値へフォールバック = 不変)
#
# ★これらは P1 トグルとは**独立**に効く(既存 `model.plan_max_tokens` と同じ流儀)。
#   トグルはプロンプト**文字列**の話、こちらはサンプリング**パラメータ**の話なので、
#   1 本のトグルに束ねると A/B のときに何が効いたか分離できなくなる。
# ★温度の purpose 分化の根拠: Generative Agents の実コード(run_gpt_prompt.py)は
#   「判定・分解・採点 = 0 / 生成系 = 0.5〜1」を使い分けている。plan は構造化出力、
#   recall は検索クエリなので、発話と同じ 0.7 は参照実装の流儀に反する(§3 温度の行)。
# --------------------------------------------------------------------------- #
def _temperature(sim, key: str) -> float | None:
    """`model.<key>` を float で返す。未設定(null)は None = 呼び出し側が現行値を使う。"""
    value = _sub(getattr(sim, "cfg", None), "model", key)
    return None if value is None else float(value)


def plan_temperature(sim) -> float | None:
    """朝の計画の温度(`model.plan_temperature`)。None = `model.temperature`。"""
    return _temperature(sim, "plan_temperature")


def recall_temperature(sim) -> float | None:
    """内省第 1 段 recall の温度(`model.recall_temperature`)。None = 現行の 0.7。"""
    return _temperature(sim, "recall_temperature")


# --------------------------------------------------------------------------- #
# ヘッダ・規律の組み立て(純関数。副作用・乱数ゼロ)
#
# ★第117 SNC v2 の判定 2 行(relate / follow)は**ここには無い**。理由: 本 module が
#   ヘッダを差し替えるのは reflect / plan / recall だけで、発話・返答(deliberate)の
#   ヘッダは `header()` が `base` をそのまま返す = 1 バイトも変えないからである。
#   したがって「返答のときだけ 2 行足す」は `deliberate.build_prompt` の reply 枝
#   (`snc_section`)に置いてある。注入点がそこ 1 つで P1 の ON/OFF 両方を覆う。
#   文面の正典は src/society/net/contact_formation.py の JUDGMENT_LINES。
# --------------------------------------------------------------------------- #
def header(purpose: str, *, base: str, city_name: str = "") -> str:
    """purpose 別のヘッダ 1 行を返す。

    `base` は現行の共有ヘッダ(`deliberate._header()` の出力)。**deliberate はそれを
    そのまま返す**(= 行動メニューは deliberate の本体なので 1 バイトも変えない)。
    それ以外の purpose は行動メニューを落とし、intro 1 行だけにする。
    """
    if purpose not in PURPOSES:
        raise ValueError(f"未知の purpose: {purpose!r}(既知: {PURPOSES})")
    if purpose == "deliberate":
        return base
    return _INTRO.format(city=city_name) + _TAIL[purpose]


def discipline_lines() -> list[str]:
    """ペルソナ直後に足す規律 3 行(新しいリストを返す = 呼び出し側で壊せない)。"""
    return list(DISCIPLINE)


# --------------------------------------------------------------------------- #
# 機械検査(検収 (b)。**判定はここでせず、数字と一覧を返す**)
# --------------------------------------------------------------------------- #
# 「出力は次の(いずれかの) JSON 1個のみ」= 出力形式の指示。1 プロンプトに 2 つ在るのが
# §1.3 の矛盾そのものなので、本数を数えられるようにしておく。
OUTPUT_DIRECTIVE = re.compile(r"出力は次の(?:いずれかの )?\s*JSON 1個のみ")

# 行動メニュー由来のトークン(deliberate 専用の語彙)。plan / reflect / recall の
# プロンプトに 1 つでも現れたら混入している。
MENU_TOKENS: tuple[str, ...] = (
    '"action": "speak"',
    '"action": "post"',
    '"action": "dm"',
    '"action": "wander"',
    '"action": "coin_label"',
    '"action": "do"',
    '"action": "nothing"',
    '"action": "verify"',
)


# 発話専用の書式指示。共有経路のせいで plan / reflect / recall にも載っていた
# (「…新しい内容を**話す**」を朝の計画のプロンプトが要求している状態)。
SPEECH_ONLY_LINES: tuple[str, ...] = (
    "今の時刻・場所・気分・出来事に根ざした新しい内容を話す。",
)


def audit(prompt: str, purpose: str) -> dict:
    """1 本のプロンプトの機械検査。**合否は返さない**(数字と一覧だけを返す)。

    返り値:
      {"purpose": …, "directives": 出力形式指示の本数,
       "menu_tokens": 混入している行動メニュー語彙(昇順),
       "speech_only": 混入している発話専用の書式指示(昇順),
       "has_discipline": 規律 3 行が全部あるか}
    """
    if purpose not in PURPOSES:
        raise ValueError(f"未知の purpose: {purpose!r}(既知: {PURPOSES})")
    tokens = sorted({t for t in MENU_TOKENS if t in prompt})
    speech = sorted({t for t in SPEECH_ONLY_LINES if t in prompt})
    return {"purpose": purpose,
            "directives": len(OUTPUT_DIRECTIVE.findall(prompt)),
            "menu_tokens": tokens,
            "speech_only": speech,
            "has_discipline": all(line in prompt for line in DISCIPLINE)}


# --------------------------------------------------------------------------- #
# 新旧サンプルの書き出し(検収 (d) = A8 の目視素材)
#
# ★ここは**観測専用**。シムも世界も 1 ビットも触らない(読むだけ・組むだけ)。
# ★実 LLM は 1 本も呼ばない。
# --------------------------------------------------------------------------- #
class _SampleMem:
    """サンプル用の記憶スタブ(`build_prompt` が触る最小の口だけを持つ)。"""

    day_summaries = ["昨日は人に会って、そのあと長く歩いた"]

    def recent(self, n):
        return ["路上で知人と立ち話をした", "店の前で行列を見た"][:n]

    def retrieve(self, *a, **k):
        return ["前にもこの時間はここが混んでいた"]

    def query_ex(self, *a, **k):
        class _R:
            hits = ["同じ場所で似た話をした"]
            failed = False
            cue = "手掛かり"
        return _R()

    def relation_line(self, ids):
        return None


class _SampleAgent:
    """サンプル用のエージェントスタブ(名簿にも世界にも触らない純粋な入れ物)。"""

    id = 0
    name = "見本"
    persona = "私は見本の人間です。近所の店で働いていて、休みの日はよく歩きます。"
    activity = "working"
    states: dict = {}
    beliefs = ["人は思ったより話しかけてくれる"]
    adopted = {"見本語"}
    said = ["さっきの一言"]
    money = 120000
    self_model = {"self": "よく歩く人", "ties": "同じ店の人"}
    implicit_self = "人と過ごす時間がいつもより増えている"
    mem = _SampleMem()


def _samples(city: str = "見本町") -> list[tuple[str, str, str]]:
    """[(purpose, 旧プロンプト, 新プロンプト)] を組む(遅延 import = 循環回避)。"""
    from . import planning as _planning
    from . import reflection as _reflection
    from .deliberate import build_prompt

    agent = _SampleAgent()
    common = dict(place_name="見本通り", sim_min=8 * 60 + 30, step=51,
                  city_name=city)
    dp_cfg = {"min_blocks": 4, "max_blocks": 8, "max_conting": 3}
    out: list[tuple[str, str, str]] = []

    def _deliberate(flag):
        return build_prompt(agent, place_name=common["place_name"],
                            surprise="social", nearby_names=["甲", "乙"],
                            nearby_ids=[1, 2], nearby_pois=["店A", "店B"],
                            sim_min=common["sim_min"], step=common["step"],
                            city_name=city, p1_purpose=flag)

    def _plan(flag):
        return _planning.build_plan_prompt(
            agent, place_name=common["place_name"], sim_min=common["sim_min"],
            step=common["step"], city_name=city, day_plan=dp_cfg,
            p1=bool(flag))

    def _reflect(flag):
        return _reflection.build_reflect_request(
            agent, step=common["step"], sim_min=common["sim_min"],
            place_name="自宅", date_line=None, weather_line=None,
            reflect_cfg=None, reflect_variety=False,
            interstitial_digest=None, interstitial=False, city_name=city,
            max_tokens=896, think=False, recalled=[], recall_fail=None,
            p1=bool(flag))["prompt"]

    def _recall(flag):
        return _reflection.build_recall_request(
            agent, step=common["step"], place_name="自宅", city_name=city,
            p1=bool(flag))["prompt"]

    out.append(("deliberate", _deliberate(None), _deliberate("deliberate")))
    out.append(("plan", _plan(False), _plan(True)))
    out.append(("reflect", _reflect(False), _reflect(True)))
    out.append(("recall", _recall(False), _recall(True)))
    return out


def dump_samples(out_path) -> str:
    """新旧プロンプトのサンプルを 1 枚の Markdown へ書き出す(A8 の目視素材)。

    `out_path` はファイルパス(親ディレクトリは作る)。書いたパスを返す。
    """
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# V-P1 プロンプト新旧サンプル(目視素材)",
        "",
        "- 生成: `society.cognition.prompt_p1.dump_samples`(実 LLM 呼びゼロ・決定論)。",
        "- 旧 = `prompts.p1.enabled: false`(現行)/ 新 = `true`。",
        "- 見るところ: (1) plan / reflect / recall の冒頭から行動メニューが消えたか、",
        "  (2)「出力は次の…JSON 1個のみ」が各 purpose で 1 本だけか、",
        "  (3) ペルソナ直後に規律 3 行が入ったか。",
        "",
    ]
    for purpose, old, new in _samples():
        a_old, a_new = audit(old, purpose), audit(new, purpose)
        parts += [
            f"## {purpose}",
            "",
            f"- 旧: 出力形式指示 {a_old['directives']} 本 / "
            f"行動メニュー語彙 {len(a_old['menu_tokens'])} 種 / "
            f"発話専用の指示 {len(a_old['speech_only'])} 本 / "
            f"規律行 {'有' if a_old['has_discipline'] else '無'} / {len(old)} 字",
            f"- 新: 出力形式指示 {a_new['directives']} 本 / "
            f"行動メニュー語彙 {len(a_new['menu_tokens'])} 種 / "
            f"発話専用の指示 {len(a_new['speech_only'])} 本 / "
            f"規律行 {'有' if a_new['has_discipline'] else '無'} / {len(new)} 字",
            "",
            "### 旧",
            "",
            "```text",
            old,
            "```",
            "",
            "### 新",
            "",
            "```text",
            new,
            "```",
            "",
        ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)
