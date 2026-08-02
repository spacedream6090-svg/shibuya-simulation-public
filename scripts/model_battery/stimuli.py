"""A〜E 5層の刺激生成(決定論)。

掟:
  - **全モデルに同一プロンプト・同一シード・同一温度**を流す(正典 §4)。
    したがってプロンプト文字列にモデル名・バックエンド名は一切入れない。
  - 刺激は seed と scale だけの純関数。時刻・乱数・環境変数を読まない。
  - 会話層(C)と長期層(E)は前ターンの応答に依存するので、静的なリストではなく
    `drive(cfg, ask)` の形で駆動する。ask(Call) -> 応答文字列 を呼び手が与える。
    → 偽 ask を渡せば LLM 無しでプロンプト列の決定論をテストできる。
  - 応答の**長さを指定しない**(発話長分布そのものが C 層の観測量なので、
    「20〜60字」等の指示は測りたいものを潰す)。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import metrics as M

LAYER_IDS = ("A", "B", "C", "D", "E")


# ---------------------------------------------------------------- 設定
@dataclass(frozen=True)
class BatteryConfig:
    """バッテリー1回分の設定。scale=1.0 が正規、0.2 が縮小版(各層 1/5)。"""
    seed: int = 20260803
    temperature: float = 0.8
    scale: float = 1.0

    def _n(self, full: int, floor: int) -> int:
        return max(floor, int(round(full * self.scale)))

    @property
    def n_personas(self) -> int:      # A/B 層のペルソナ数(最大5)
        return min(len(PERSONAS), self._n(len(PERSONAS), 2))

    @property
    def n_topics(self) -> int:        # C 層の話題数(最大10)
        return min(len(TOPICS), self._n(len(TOPICS), 2))

    @property
    def n_dialog_turns(self) -> int:  # C 層 1 対話あたりの総ターン数
        return self._n(8, 4)

    @property
    def n_d_samples(self) -> int:     # D 層のサンプル数(生命線: 正規 30)
        return self._n(30, 3)

    @property
    def n_e_turns(self) -> int:       # E 層の自己対話ターン数(正規 50)
        return self._n(50, 6)

    @property
    def n_e_days(self) -> int:        # E 層の繰り返し計画日数(シミュ3日相当・固定)
        return 3


# ---------------------------------------------------------------- Call
@dataclass(frozen=True)
class Call:
    layer: str
    test: str
    case_id: str
    turn: int
    prompt: str
    max_tokens: int
    json_mode: bool
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    """(base, 層, テスト, ケース, ターン) から 31bit の決定論シードを作る。

    ollama / OpenAI 互換の seed は 32bit 整数を想定。base を変えると全体が動く。
    """
    h = hashlib.sha256(("|".join([str(base)] + [str(p) for p in parts]))
                       .encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


# ---------------------------------------------------------------- ペルソナ
@dataclass(frozen=True)
class Persona:
    pid: str
    name: str
    text: str
    baseline: str          # B 層で提示する「今の予定」
    outdoor_plan: str      # 雨摂動の対象になる屋外の予定


PERSONAS: tuple[Persona, ...] = (
    Persona(
        "office_worker", "田村",
        "34歳の会社員。渋谷のオフィスで働いている。世田谷から電車で通勤。"
        "平日は9時半出社・19時前後退社。妻と二人暮らし。運動不足を気にしている。",
        "10:00に社内会議。12:15に同僚と昼食。19:30に退社して帰宅。",
        "18:00から会社の近くを30分ジョギングする予定"),
    Persona(
        "student", "小峰",
        "20歳の大学生。渋谷の大学に通っている。実家住まい。"
        "授業は週4日。バイトはカフェで週2回。夜更かしが多い。",
        "13:00から3限の授業。16:30にサークル。19:00からバイト。",
        "15:00に友人と公園で待ち合わせる予定"),
    Persona(
        "retiree", "小野",
        "72歳。数年前に定年退職した。渋谷区内の集合住宅に一人で住んでいる。"
        "朝が早い。近所の散歩と将棋の会が楽しみ。膝が少し悪い。",
        "9:00から散歩。11:00に病院。15:00に将棋の会。",
        "9:00から近所を1時間散歩する予定"),
    Persona(
        "parent", "西野",
        "41歳。小学生の子どもが二人いる。週3日パートで働いている。"
        "夕方は子どもの送り迎えと食事の支度で忙しい。自分の時間がほとんどない。",
        "9:30からパート勤務。15:00に学童へ迎え。18:00に夕食の支度。",
        "16:00に子どもと近所の公園へ行く予定"),
    Persona(
        "night_shift", "梶",
        "27歳。渋谷の飲食店で働いている。シフトは夕方から深夜。"
        "起きるのは昼過ぎ。休みの日も生活リズムが夜型のまま。",
        "16:00に出勤。24:00に閉店作業。26:00頃に帰宅。",
        "14:00に自転車で買い出しに行く予定"),
)
PERSONA_BY_ID = {p.pid: p for p in PERSONAS}

DAY_TYPES = ("平日", "休日")


# ---------------------------------------------------------------- 共通の断片
def _activity_menu() -> str:
    """行動20分類の一覧(プロンプトに載せる正典の並び)。"""
    return " / ".join(M.ACTIVITY_CATEGORIES)


_JSON_ONLY = "出力は JSON だけ。前置き・説明・コードブロック記号は書かない。"


# ---------------------------------------------------------------- A 層
A_TEST = "A_day_schedule"


def a_prompt(p: Persona, daytype: str) -> str:
    return (
        f"あなたは次の人物です。\n"
        f"名前: {p.name}\n{p.text}\n\n"
        f"今日は{daytype}です。今日一日(0:00 から 24:00 まで)を、"
        f"あなたが実際に送るであろう一日として時間割にしてください。"
        f"理想の一日ではなく、いつも通りの一日です。\n\n"
        f"行動の名前は次の20語から必ず選びます:\n{_activity_menu()}\n\n"
        f"形式:\n"
        f'{{"blocks":[{{"start":"00:00","end":"06:30","activity":"睡眠"}},'
        f'{{"start":"06:30","end":"07:00","activity":"身の回りの用事"}}]}}\n\n'
        f"- 0:00 から 24:00 まで隙間なく埋めること(睡眠も必ず書く)。\n"
        f"- start/end は24時間表記の HH:MM。ブロックは時刻順。\n"
        f"- {_JSON_ONLY}"
    )


def drive_a(cfg: BatteryConfig, ask: Callable[[Call], str]) -> None:
    for p in PERSONAS[:cfg.n_personas]:
        for dt in DAY_TYPES:
            case = f"{p.pid}/{dt}"
            ask(Call("A", A_TEST, case, 0, a_prompt(p, dt), 900, True,
                     stable_seed(cfg.seed, "A", A_TEST, case, 0),
                     {"persona": p.pid, "daytype": dt}))


# ---------------------------------------------------------------- B 層
B_TEST = "B_perturbation"


@dataclass(frozen=True)
class Perturbation:
    kid: str
    label: str
    event: str
    # 方向妥当性の判定種別。invite は「正解の向き」が無いので None を返す。
    check: str


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("rain", "雨",
                 "出かける直前に、強い雨が降り出した。当分やみそうにない。", "rain"),
    Perturbation("delay", "遅延",
                 "出かけようとしたら、いつも使っている路線が30分の遅延だと表示が出ている。",
                 "delay"),
    Perturbation("invite", "誘い",
                 "友人から「今夜どこかでごはんでもどう?」と連絡が来た。", "invite"),
    Perturbation("closure", "閉店",
                 "行こうとしていた店の前まで来たら、臨時休業の貼り紙が出ていた。",
                 "closure"),
)
PERTURBATION_BY_ID = {x.kid: x for x in PERTURBATIONS}


def b_prompt(p: Persona, pert: Perturbation) -> str:
    plan = p.outdoor_plan if pert.kid == "rain" else p.baseline
    return (
        f"あなたは次の人物です。\n"
        f"名前: {p.name}\n{p.text}\n\n"
        f"今日の予定: {plan}\n\n"
        f"ところが、{pert.event}\n\n"
        f"あなたはどうしますか。次の JSON で答えてください。\n"
        f'{{"change": true, "new_activity": "<行動20語のどれか>", '
        f'"delta_minutes": 0, "reason": "<理由を一文で>"}}\n\n'
        f"- change: 予定を変えるなら true、変えないなら false。\n"
        f"- new_activity: 変えた後にすることを次の20語から選ぶ"
        f"(変えないなら元の行動を選ぶ)。\n{_activity_menu()}\n"
        f"- delta_minutes: 予定の時刻をずらす分数。早めるなら負、遅らせるなら正、"
        f"ずらさないなら 0。\n"
        f"- {_JSON_ONLY}"
    )


def drive_b(cfg: BatteryConfig, ask: Callable[[Call], str]) -> None:
    for p in PERSONAS[:cfg.n_personas]:
        for pert in PERTURBATIONS:
            case = f"{p.pid}/{pert.kid}"
            ask(Call("B", B_TEST, case, 0, b_prompt(p, pert), 300, True,
                     stable_seed(cfg.seed, "B", B_TEST, case, 0),
                     {"persona": p.pid, "perturbation": pert.kid,
                      "check": pert.check}))


# ---------------------------------------------------------------- C 層
C_TEST = "C_dialogue"

TOPICS: tuple[str, ...] = (
    "今日の天気",
    "渋谷駅の人の多さ",
    "最近見たドラマや動画",
    "仕事や学校でうんざりしたこと",
    "週末の予定",
    "食べ物の値上がり",
    "睡眠と体調",
    "近所の店が入れ替わったこと",
    "家族や親戚のこと",
    "電車が遅れたときの話",
)


def c_prompt(speaker: Persona, other: Persona, topic: str,
             transcript: list[tuple[str, str]]) -> str:
    if transcript:
        lines = "\n".join(f"{n}: {t}" for n, t in transcript)
        hist = f"ここまでの会話:\n{lines}\n\n"
        task = "続けて、あなたの次の一言を書いてください。"
    else:
        hist = ""
        task = f"「{topic}」について、あなたから最初の一言を言ってください。"
    return (
        f"あなたは次の人物です。\n"
        f"名前: {speaker.name}\n{speaker.text}\n\n"
        f"{other.name}さんと立ち話をしています。\n\n"
        f"{hist}{task}\n\n"
        f"- 話し言葉で、あなたが実際に言うとおりに書く。\n"
        f"- 同意しても、しなくてもよい。話を変えてもよい。\n"
        f"- 発話の中に名前・かぎかっこ・ト書き・説明を書かない。\n"
        f'形式: {{"say": "<発話>"}}\n'
        f"- {_JSON_ONLY}"
    )


def drive_c(cfg: BatteryConfig, ask: Callable[[Call], str]) -> None:
    for ti, topic in enumerate(TOPICS[:cfg.n_topics]):
        a = PERSONAS[ti % len(PERSONAS)]
        b = PERSONAS[(ti + 1) % len(PERSONAS)]
        case = f"t{ti:02d}"
        transcript: list[tuple[str, str]] = []
        for turn in range(cfg.n_dialog_turns):
            spk, oth = (a, b) if turn % 2 == 0 else (b, a)
            call = Call("C", C_TEST, case, turn,
                        c_prompt(spk, oth, topic, transcript), 220, True,
                        stable_seed(cfg.seed, "C", C_TEST, case, turn),
                        {"topic": topic, "speaker": spk.pid, "listener": oth.pid})
            resp = ask(call)
            transcript.append((spk.name, _one_line(say_of(resp))))


SAY_KEY = "say"


def say_of(text: str) -> str:
    """応答から発話本文を取り出す({"say": "..."} 前提。壊れていたら生文字列)。

    ★C/E を JSON 封筒にした理由(第90バッチの実測):
      自由文で聞くと qwen3:4b は think=False でも英語の推敲を延々と書き、
      max_tokens を 220→800→1400 と上げても発話に到達しなかった。一方、
      本シミュ本体は会話も `format=json` の行動 JSON で受け取っている
      (src/society/cognition/deliberate.py の parse_action)。つまり自由文設定は
      **本番より厳しい別条件**を測ってしまう。封筒だけ本番に合わせ、中身(発話の
      長さ・語彙・あいづち・不同意)は一切指定しないことで、測りたいものを保つ。
    """
    obj = M.extract_json(text or "")
    if isinstance(obj, dict):
        v = obj.get(SAY_KEY)
        if isinstance(v, str) and v.strip():
            return v
    return text or ""


def _one_line(text: str) -> str:
    """会話履歴に積む用に応答を1行へ潰す(履歴のプロンプト形が壊れないように)。"""
    s = M.norm_text(text).replace("\r", " ").replace("\n", " ")
    return " ".join(s.split())


# ---------------------------------------------------------------- D 層
D_TEST = "D_variance"

# 選択肢は「どれも普通にありうる」8択。正解が無いので分散だけを測れる。
D_CHOICES: tuple[str, ...] = (
    "カフェに入る", "本屋をのぞく", "公園のベンチで座る", "散歩する",
    "友人に連絡する", "買い物をする", "家に帰る", "何もしないでぼんやりする",
)
D_PERSONA = PERSONAS[0]


def d_prompt() -> str:
    menu = "\n".join(f"- {c}" for c in D_CHOICES)
    return (
        f"あなたは次の人物です。\n"
        f"名前: {D_PERSONA.name}\n{D_PERSONA.text}\n\n"
        f"平日の昼、渋谷で予定が急に無くなって1時間だけ空きました。"
        f"あなたはどうしますか。\n\n"
        f"次から一つ選びます:\n{menu}\n\n"
        f'形式: {{"choice": "<選んだもの>", "why": "<理由を一文で>"}}\n'
        f"- {_JSON_ONLY}"
    )


def drive_d(cfg: BatteryConfig, ask: Callable[[Call], str]) -> None:
    prompt = d_prompt()      # ★全サンプルで完全に同一のプロンプト(シードだけ変える)
    for i in range(cfg.n_d_samples):
        case = f"s{i:03d}"
        ask(Call("D", D_TEST, case, 0, prompt, 200, True,
                 stable_seed(cfg.seed, "D", D_TEST, case, 0), {"sample": i}))


# ---------------------------------------------------------------- E 層
E_MONO_TEST = "E_monologue"
E_PLAN_TEST = "E_repeat_plan"
E_HISTORY_WINDOW = 6        # 自己対話の履歴窓(全ターン積むと文脈長が交絡するため)
E_PERSONA = PERSONAS[1]


def e_mono_prompt(history: list[str]) -> str:
    if history:
        recent = "\n".join(f"- {h}" for h in history[-E_HISTORY_WINDOW:])
        hist = f"さっきまで考えていたこと:\n{recent}\n\n"
    else:
        hist = ""
    return (
        f"あなたは次の人物です。\n"
        f"名前: {E_PERSONA.name}\n{E_PERSONA.text}\n\n"
        f"{hist}"
        f"いま一人で考えごとをしています。次に頭に浮かんだことを、"
        f"独り言としてそのまま書いてください。\n"
        f"- 一人称の話し言葉。説明・見出し・箇条書きは書かない。\n"
        f'形式: {{"say": "<独り言>"}}\n'
        f"- {_JSON_ONLY}"
    )


def e_plan_prompt(day: int, prev: list[str]) -> str:
    if prev:
        lines = "\n".join(f"{i + 1}日目: {p}" for i, p in enumerate(prev))
        hist = f"これまでの日の過ごし方:\n{lines}\n\n"
    else:
        hist = ""
    return (
        f"あなたは次の人物です。\n"
        f"名前: {E_PERSONA.name}\n{E_PERSONA.text}\n\n"
        f"{hist}"
        f"今日は{day}日目です。今日一日をどう過ごすか、"
        f"次の JSON で3つから5つの予定にまとめてください。\n"
        f'{{"day": {day}, "plan": [{{"when":"HH:MM","what":"<すること>"}}], '
        f'"mood": "<今日の気分を一言>"}}\n'
        f"- {_JSON_ONLY}"
    )


def drive_e(cfg: BatteryConfig, ask: Callable[[Call], str]) -> None:
    history: list[str] = []
    for turn in range(cfg.n_e_turns):
        call = Call("E", E_MONO_TEST, "mono", turn, e_mono_prompt(history),
                    220, True,
                    stable_seed(cfg.seed, "E", E_MONO_TEST, "mono", turn), {})
        history.append(_one_line(say_of(ask(call))))

    prev: list[str] = []
    for day in range(1, cfg.n_e_days + 1):
        case = f"day{day}"
        call = Call("E", E_PLAN_TEST, case, day - 1, e_plan_prompt(day, prev),
                    400, True,
                    stable_seed(cfg.seed, "E", E_PLAN_TEST, case, day - 1),
                    {"day": day})
        prev.append(_one_line(ask(call)))


# ---------------------------------------------------------------- 束ね
DRIVERS: dict[str, Callable[[BatteryConfig, Callable[[Call], str]], None]] = {
    "A": drive_a, "B": drive_b, "C": drive_c, "D": drive_d, "E": drive_e,
}

TESTS_BY_LAYER: dict[str, tuple[str, ...]] = {
    "A": (A_TEST,), "B": (B_TEST,), "C": (C_TEST,), "D": (D_TEST,),
    "E": (E_MONO_TEST, E_PLAN_TEST),
}


def collect_calls(cfg: BatteryConfig, layers: tuple[str, ...],
                  responder: Callable[[Call], str] | None = None) -> list[Call]:
    """刺激列を収集する(テスト・件数見積り用)。

    responder 未指定なら固定文字列を返す偽 ask を使う = **プロンプト決定論の検査**。
    C/E は前応答に依存するので、responder が変われば以降のプロンプトも変わる。
    """
    resp = responder or (lambda c: '{"ok":1}')
    out: list[Call] = []

    def ask(call: Call) -> str:
        out.append(call)
        return resp(call)

    for layer in layers:
        if layer not in DRIVERS:
            raise ValueError(f"未知の層: {layer}(有効: {LAYER_IDS})")
        DRIVERS[layer](cfg, ask)
    return out


def iter_layers(spec: str) -> tuple[str, ...]:
    """'A,B,C' / 'all' → 層タプル。順序は正典 LAYER_IDS 順に正規化。"""
    s = (spec or "").strip().lower()
    if s in ("", "all", "*"):
        return LAYER_IDS
    want = {x.strip().upper() for x in spec.split(",") if x.strip()}
    bad = want - set(LAYER_IDS)
    if bad:
        raise ValueError(f"未知の層: {sorted(bad)}(有効: {LAYER_IDS})")
    return tuple(x for x in LAYER_IDS if x in want)
