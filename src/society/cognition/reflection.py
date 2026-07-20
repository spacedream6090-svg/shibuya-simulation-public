"""ソロ内省 = k の実装部位(D7)。就寝直後に1回発火(記憶整理)。

★ 個別の就寝時刻に紐づくため、内省 LLM の呼び出しは自然に時間分散する
  (2026-07-04 ユーザー指示の意図そのもの: 一斉夜間バッチを避ける)。

k.writeback:
  free     — 内省の結論(belief)をそのまま自分に書き戻す(結合 最大)
  degraded — 確率 alpha でのみ書き戻す(結合 中間)
  sham     — 内省は実行するが結果を捨てる(計算量同一・結合ゼロ)★R1 対照
  off      — 内省自体を行わない

controls(D7 対照。R1 = 計算量交絡の排除):
  none            — 現状動作(off は完全に何もしない)
  compute_matched — off でも内省 LLM を実行し全結果を破棄(belief も consolidate も
                    しない)。これで off/sham/degraded/free の呼数・トークンが一致。
  null_series     — 内省側は none と同じ(発火側でダミー呼び出しを足す。scheduler)

agentic_pull(ユーザー採用決定): true のとき内省を固定2段にする。
  第1段=「何を思い出したいか」を出させ(recall)→ mem.query → 第2段=検索結果を
  注入した本内省。off 以外の全条件で常に+1呼(compute_matched の off も2呼)。
"""
from __future__ import annotations

from ..observer.logger import ObserverLogger
from ..observer.schema import Event
from ..rng import _stable_hash
from .deliberate import build_prompt, parse_action

_REFLECT_TASK = (
    "\n眠りにつく前に、今日一日を振り返って内省してください。"
    "\nまず今日の出来事を順に思い出し、なぜ印象に残ったのか・自分の"
    "気持ちがどう動いたか・明日の自分にどう影響するかまでじっくり"
    "考えてから、結論だけを JSON で出力してください。"
    "\n出力は次の JSON 1個のみ(キー名は厳守):"
    '\n{"action": "reflect", "summary": "今日一日の要約を一文",'
    ' "salient": [{"text": "印象に残った出来事", "importance": 1〜10の数}],'
    ' "belief": "今日の経験からの考え方の変化・結論を一文"}')

def build_reflection_cfg(raw) -> dict:
    """反射=自己モデル(第11バッチ)+出来事誘発の深い内省(第12バッチ)の設定。

    第12バッチ(2026-07-08、ユーザー仮説→文献検証 docs/research/deep-reflection-triggers.md
    / self-concept-identity.md を反映):
    - deep: 深い内省は固定周期でなく**出来事に誘発**される。日内の衝撃ゲージ(信念との
      乖離の近似=|Δstate| をネガ非対称で加重。Baumeister の負の優位)が個人閾値を超えた日
      →「頭から離れない」侵入的段階(記憶の重要度↑)→ **incubation_days(1晩以上)おいて**
      その夜の内省を深い内省に格上げ(PTG の 侵入的→熟慮的 の二段構え)。cooldown で
      過剰発火(反芻の害)を抑制。
    - implicit_self: 無意識層=自分の行動・経験の客観カウントから決定論で組む
      「最近の自分」1行(Bem 自己知覚・working self-concept の近似)。揮発的な作動自己
      であり、安定した核自己(self_model=深い内省の産物)とはフィールドを分ける
      (二層は共有状態でつながる相互作用する2経路)。
    - self_model_days: 旧固定周期(レガシー・実験対照用)。0=無効。
    すべて既定 OFF=従来と完全同一(バイト一致)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    deep = dict(raw.get("deep", {}) or {})
    imp = dict(raw.get("implicit_self", {}) or {})
    return {
        "self_model_days": int(raw.get("self_model_days", 0)),
        "deep": {
            "enabled": bool(deep.get("enabled", False)),
            "threshold": float(deep.get("threshold", 0.60)),       # 日内衝撃ゲージの基準閾値
            "neg_weight": float(deep.get("neg_weight", 2.0)),      # 負の非対称(Baumeister)
            "pos_weight": float(deep.get("pos_weight", 1.0)),      # 正も閾値超え可(畏敬・転機)
            "incubation_days": int(deep.get("incubation_days", 1)),  # 侵入的→熟慮的の間(日)
            "cooldown_days": int(deep.get("cooldown_days", 3)),    # 深い内省後の不応期(反芻抑制)
        },
        "implicit_self": {
            "enabled": bool(imp.get("enabled", False)),
            "ema": float(imp.get("ema", 0.7)),                     # 行動ベースラインの平滑率
        },
    }


# 反射=自己モデル(第11バッチ 2026-07-08。Generative Agents の reflection に倣う):
# 深い内省の夜、同じ1回の内省呼び出しを格上げし、蓄積記憶から
# より抽象的な自己像(性格・最近の変化)と大事な関係の要約を生成する。
# 前回の自己像は build_prompt が「自分の理解(内省より)」として注入済みなので、
# LLM はそれを踏まえて更新する=自己認識の再帰。呼数は不変(R1)。
_DEEP_REFLECT_TASK = (
    "\n眠りにつく前に、今日一日とこれまでの日々を振り返って深く内省してください。"
    "\n今日の出来事に加えて、蓄積した記憶を振り返り、自分がどんな人間か・最近"
    "どう変わったか(自己像)と、自分にとって大事な人間関係もまとめ直してください。"
    "以前の自己像(「自分の理解」)があれば、それを引き写すのではなく今の自分に"
    "合わせて更新してください。"
    "\n出力は次の JSON 1個のみ(キー名は厳守):"
    '\n{"action": "reflect", "summary": "今日一日の要約を一文",'
    ' "salient": [{"text": "印象に残った出来事", "importance": 1〜10の数}],'
    ' "belief": "今日の経験からの考え方の変化・結論を一文",'
    ' "self": "自分がどんな人間か・最近の変化の要約を一文",'
    ' "ties": "自分にとって大事な人間関係の要約を一文"}')


# ---------------------------------------------------------------- 内省プロンプト改善 A4(第20バッチ検収)
# 実LLM 検収で belief の約25%が雛形の復唱=「今日の経験から〜」で始まる定型文だった問題への対策。
# prompts.reflect_variety=true のとき、belief 説明(JSON 例)の言い回しを個体×日で 4 バリアントの
# **決定論ローテーション**にし、「決まり文句で belief を始めない」1行を足す。狙いは、常に同じ
# 例文「今日の経験からの…」を見せることで生じる冒頭句のアンカー効果を外すこと。
#   - バリアントはすべて意味等価(問うことは同じ=「今日をふまえ自分の中で動いた考え・結論を一文」。
#     言い回しだけを変える)。JSON のキー(summary/salient/belief/self/ties)は一切変えない。
#   - 選択は乱数を使わず _stable_hash(agent_id, day)(rng.py 流儀=プロセス非依存)で決める=
#     呼数・乱数構造・発火は不変(R1)。既定 OFF は _REFLECT_TASK/_DEEP_REFLECT_TASK とバイト一致
#     (ゴールデン tests/test_scenario.py を守る)。
_ORIG_BELIEF_DESC = "今日の経験からの考え方の変化・結論を一文"
_JSON_HEAD = "\n出力は次の JSON 1個のみ(キー名は厳守):"
# ★既知の限界(実LLM検収 a4_reflect_variety_on_s42・qwen3:4b・2026-07-15):
#   4B モデルは説明句を belief 値へ**そのまま丸写し**することがある(15件中5件)。
#   文言の言い換え(体言句化)や「説明の文言を写さない」注意でも防げないことを
#   ペアプローブ(同一文脈15本×新旧文言)で確認済み=語調でなく確率的な失敗。
#   → 文言は検収ランで検証済みのこのセットに固定。丸写しの機械ガード(完全一致 belief の
#   棄却)や reflect 温度の調整は writeback 率・R1 に触れるためユーザー判断待ち。
_BELIEF_VARIANTS = (
    "考えたことの結論・自分の中で変わった見方を一文",
    "心に残った気づきや、これからに活きる結論を一文",
    "自分なりにたどり着いた結論・変化した考えを一文",
    "いまの実感に近い言葉で、得た気づき・結論を一文",
)
_VARIETY_NOTE = ("\n※ belief は「今日の経験から」「今日の出来事」などの決まり文句で"
                 "書き始めず、自分の言葉で書いてください。")
# 差し替えの足場(冒頭句アンカー)が両タスク文にちょうど1回ずつ在ることを import 時に担保。
# ここが壊れると variety の note/変異が黙って落ちるので、崩れたら即座に import を失敗させる。
assert _ORIG_BELIEF_DESC in _REFLECT_TASK and _ORIG_BELIEF_DESC in _DEEP_REFLECT_TASK
assert _JSON_HEAD in _REFLECT_TASK and _JSON_HEAD in _DEEP_REFLECT_TASK


# ---------------------------------------------------------------- ナラティブ補間の物語化(P2 S2)
# 機械ダイジェスト(前回発火以降の客観列挙。build_prompt が interstitial_digest として注入)は
# 「意味づけをしない」。意味づけ=一人称の物語化は夜内省の LLM の仕事という二段分離
# (docs/research/interstitial-life.md §4.2)。prompts.interstitial.enabled=true のときだけ
# 内省タスク文の末尾に「今日一日を一人称の短い物語として振り返る」1文を足す(追加 LLM 呼なし・
# JSON 契約キー不変=既存 summary に綴る)。既定 OFF は _REFLECT_TASK とバイト一致(ゴールデン不変)。
_STORY_NOTE = (
    "\nまた、今日一日を振り返り、実際にあった出来事の流れ(行った場所・会った人・"
    "起きたこと)を、順を追って一人称の短い物語として summary に綴ってください。")


def _reflect_task(*, deep: bool, variety: bool, agent_id: int, day: int) -> str:
    """内省タスク文を返す。variety=False は従来定数とバイト一致(既定 OFF=ゴールデン不変)。

    variety=True のときだけ belief 説明を個体×日で決定論ローテーションし、定型句回避の1行を
    足す。乱数は使わず _stable_hash で選ぶ(R1: 呼数・乱数消費・発火は不変)。JSON 契約キーは不変。
    """
    base = _DEEP_REFLECT_TASK if deep else _REFLECT_TASK
    if not variety:
        return base
    idx = _stable_hash(f"reflect_variety/{agent_id}/{day}") % len(_BELIEF_VARIANTS)
    task = base.replace(_ORIG_BELIEF_DESC, _BELIEF_VARIANTS[idx])
    return task.replace(_JSON_HEAD, _VARIETY_NOTE + _JSON_HEAD)


# ---------------------------------------------------------------- 無意識層(第12バッチ)
# 自己認識の更新は明示的な内省なしでも裏で回る(Bem 自己知覚・working self-concept・
# 反射的評価。docs/research/self-concept-identity.md §1=二層は条件付き支持)。
# ここは LLM を使わない決定論の系: 昨日の行動・経験カウントのベースラインからの逸脱
# (予測誤差の近似)と感情価バランスから「最近の自分」1行を組み、build_prompt が注入する。
# 揮発的な作動自己=安定した核自己(self_model)とはフィールドを分ける。
_REASON_JP = {"company": "人と過ごす時間", "sns": "SNSを見る時間",
              "news": "ニュースを追う時間", "silence": "ひとりの時間",
              "novel_place": "新しい場所に行くこと", "unknown_word": "知らない言葉との出会い"}


def update_implicit_self(agent, ema: float) -> None:
    """日次: 行動ベースライン(EMA)を更新し、逸脱と感情価から「最近の自分」を組み直す。

    behav_today が None(=implicit_self OFF)なら何もしない(バイト一致)。乱数なし。"""
    bt = getattr(agent, "behav_today", None)
    if bt is None:
        return
    be = agent.behav_ema if agent.behav_ema is not None else {}
    agent.behav_ema = be
    keys = set(bt) | {k for k in be if not k.startswith("_")}
    dev_up = dev_down = None
    best_up = best_down = 0.0
    for k in sorted(keys):
        today = float(bt.get(k, 0.0))
        base = float(be.get(k, today))
        d = today - base
        signif = max(2.0, 0.5 * base)              # 有意な逸脱=絶対2回 or ベースの5割
        if d > best_up and d >= signif:
            best_up, dev_up = d, k
        if d < best_down and -d >= signif:
            best_down, dev_down = d, k
        be[k] = ema * base + (1.0 - ema) * today
    be["_neg"] = ema * float(be.get("_neg", 0.0)) \
        + (1.0 - ema) * float(getattr(agent, "impact_neg_today", 0.0))
    be["_pos"] = ema * float(be.get("_pos", 0.0)) \
        + (1.0 - ema) * float(getattr(agent, "impact_pos_today", 0.0))
    parts = []
    if dev_up is not None and _REASON_JP.get(dev_up):
        parts.append(f"{_REASON_JP[dev_up]}がいつもより増えている")
    elif dev_down is not None and _REASON_JP.get(dev_down):
        parts.append(f"{_REASON_JP[dev_down]}がいつもより減っている")
    if be["_neg"] > 1.5 * be["_pos"] and be["_neg"] > 0.05:
        parts.append("気持ちはすこし重い")
    elif be["_pos"] > 1.5 * be["_neg"] and be["_pos"] > 0.05:
        parts.append("気持ちは上向き")
    agent.implicit_self = "、".join(parts)


class DirectSink:
    """観測出力の直結先(逐次経路)。log_llm_call / log をその場で呼ぶ。"""
    def __init__(self, logger: ObserverLogger):
        self.logger = logger

    def call(self, d: dict) -> None:
        self.logger.log_llm_call(d)

    def event(self, e: Event) -> None:
        self.logger.log(e)


class BufferSink:
    """観測出力の遅延バッファ(P2 S6b 一括発行用)。

    バッチ経路では recall の解決を全個体まとめて行うが、イベント列は逐次実行と
    同一の並び(個体ごとに [recall 系 → reflect 系])を保ちたい。ここに溜めて
    最終 apply ループの各個体の先頭で flush する。
    """
    def __init__(self):
        self.items: list[tuple[str, object]] = []

    def call(self, d: dict) -> None:
        self.items.append(("call", d))

    def event(self, e: Event) -> None:
        self.items.append(("event", e))

    def flush(self, logger: ObserverLogger) -> None:
        for kind, x in self.items:
            if kind == "call":
                logger.log_llm_call(x)
            else:
                logger.log(x)
        self.items.clear()


def build_recall_request(agent, *, step: int, place_name: str,
                         date_line: str | None = None,
                         weather_line: str | None = None,
                         city_name: str = "") -> dict:
    """agentic pull 第1段の LLM 要求(P2 S6b: build/resolve 分割)。generate_many 互換。"""
    prompt = (build_prompt(agent, place_name=place_name, surprise=None,
                           nearby_names=[], step=step, city_name=city_name,
                           date_line=date_line, weather_line=weather_line)
              + "\n今日の内省を始める前に、まず今日のことで思い出したいことを一つ挙げてください。"
              + "\n出力は次の JSON 1個のみ:"
              + '\n{"action": "recall", "query": "思い出したい事柄を短く"}')
    return {"prompt": prompt, "rng_key": f"recall/{agent.id}/{step}",
            "temperature": 0.7, "max_tokens": 100, "think": False}


def resolve_recall(agent, *, step: int, sim_min: int, response: str,
                   call_id: str | None, cached: bool,
                   sink) -> tuple[list[str], str | None]:
    """recall 応答の解決(想起は決定論)。観測出力は sink 経由(直結/遅延を差し替え可)。

    戻り = (hits, fail_line)。ACT-R 有効かつ「手掛かりはあるが全候補が閾値未達」なら
    memory_fail イベントを発火し「思い出そうとして失敗」の1行を fail_line で返す(ACT-R OFF
    では query_ex.failed が常に False=fail_line None=memory_recall のみ=バイト一致)。
    """
    sink.call({"llm_call_id": call_id, "agent_id": agent.id,
               "purpose": "recall", "step": step, "cached": cached})
    action = parse_action(response)
    query_text = action.get("query", "") if action and action.get("type") == "recall" else ""
    fail_line: str | None = None
    if query_text:
        res = agent.mem.query_ex(step, query_text, n=3, agent_id=agent.id)
        hits = res.hits
        if res.failed:                   # 手掛かりはあるが全候補が閾値未達=思い出そうとして失敗
            fail_line = f"({res.cue}のことを思い出そうとしたが、はっきりしない…)"
            best = res.best_activation
            sink.event(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="memory_fail", x=agent.x, y=agent.y,
                             llm_call_id=call_id,
                             payload={"query": res.cue,
                                      "activation": (round(best, 3)
                                                     if best is not None else None),
                                      "tau": res.tau}))
    else:
        hits = []
    sink.event(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="memory_recall", x=agent.x, y=agent.y,
                     llm_call_id=call_id,
                     payload={"query": query_text, "n_hits": len(hits)}))
    return hits, fail_line


def _recall_query(agent, *, step: int, sim_min: int, llm, place_name: str,
                  logger: ObserverLogger, date_line: str | None = None,
                  weather_line: str | None = None,
                  city_name: str = "") -> tuple[list[str], str | None]:
    """agentic pull 第1段: 何を思い出したいかを LLM に出させ、mem.query で想起する。

    LLM 呼び出しを1本足す(R1: writeback 条件に依らず一定)。想起自体は決定論。
    date_line/weather_line は当日の暦・天気(既定 None=注入せず従来と完全一致)。
    city_name はプロンプト冒頭の街名(envpack。基盤に地名を残さない)。
    build_recall_request + generate + resolve_recall の合成(処理順は分割前と完全同一)。
    """
    req = build_recall_request(agent, step=step, place_name=place_name,
                               date_line=date_line, weather_line=weather_line,
                               city_name=city_name)
    response, call_id, cached = llm.generate(
        req["prompt"], rng_key=req["rng_key"], temperature=req["temperature"],
        max_tokens=req["max_tokens"], think=req["think"])
    return resolve_recall(agent, step=step, sim_min=sim_min, response=response,
                          call_id=call_id, cached=cached,
                          sink=DirectSink(logger))


def maybe_reflect(agent, *, step: int, sim_min: int, writeback: str, alpha: float,
                  llm, place_name: str, rng, logger: ObserverLogger,
                  max_tokens: int = 280, think: bool = False,
                  controls: str = "none", agentic_pull: bool = False,
                  date_line: str | None = None,
                  weather_line: str | None = None,
                  reflect_cfg: dict | None = None,
                  reflect_variety: bool = False,
                  interstitial_digest: str | None = None,
                  interstitial: bool = False,
                  city_name: str = "") -> None:
    """就寝中で reflect_step に達していれば内省(1睡眠につき1回)。

    v2(Phase B): 同じ1回の呼び出しで**記憶の統合**(日次要約+顕著エピソード
    重要度採点)も行う。統合は k の全条件(off以外)で常に適用=計算量・記憶量を
    条件間で同一に保つ。**k のゲートは belief の書き戻しのみ**(D7)。

    v3(第11バッチ): 深い内省の夜だけ _DEEP_REFLECT_TASK に格上げし、自己像 self/ties を
    生成して agent.self_model を更新する(反射=自己モデル)。呼数・rng 消費は不変
    (同じ1回の呼び出しのプロンプト差のみ)。自己モデルの書き込みも belief と同じ
    k ゲート(written)に従う=k の作用チャネルを1本に保つ。

    v4(第12バッチ): 深い内省の誘発を**出来事駆動**に(文献検証済み。build_reflection_cfg
    参照)。日内衝撃ゲージの閾値超え(factors._bump 側)が agent.deep_due_day を予約し、
    その日以降の夜に深い内省が起こる(侵入的→熟慮的の遅延)。実行後は cooldown。
    レガシーの固定周期(self_model_days)も併存(実験対照用)。既定は全OFF=従来と完全同一。

    v5(A4 第20バッチ検収): reflect_variety=True のとき内省タスク文の belief 説明を個体×日で
    決定論ローテーション(_reflect_task)+定型句回避の1行を足す。belief の雛形復唱対策。
    既定 OFF はプロンプト・呼数・乱数消費とも従来と完全同一(R1・ゴールデン不変)。
    """
    st = begin_reflect(agent, step=step, writeback=writeback, controls=controls)
    if st is None:
        return
    discard = st["discard"]

    # ---- agentic pull 第1段(常に+1呼。R1: writeback 条件に依らず一定)----
    recalled: list[str] = []
    recall_fail: str | None = None       # ACT-R 有効時のみ「思い出そうとして失敗」の1行
    if agentic_pull:
        recalled, recall_fail = _recall_query(
            agent, step=step, sim_min=sim_min, llm=llm,
            place_name=place_name, logger=logger,
            date_line=date_line, weather_line=weather_line,
            city_name=city_name)

    req = build_reflect_request(
        agent, step=step, sim_min=sim_min, place_name=place_name,
        date_line=date_line, weather_line=weather_line,
        reflect_cfg=reflect_cfg, reflect_variety=reflect_variety,
        interstitial_digest=interstitial_digest, interstitial=interstitial,
        city_name=city_name, max_tokens=max_tokens, think=think,
        recalled=recalled, recall_fail=recall_fail)
    response, call_id, cached = llm.generate(
        req["prompt"], rng_key=req["rng_key"], temperature=req["temperature"],
        max_tokens=req["max_tokens"], think=req["think"])
    apply_reflect_response(agent, step=step, sim_min=sim_min,
                           writeback=writeback, alpha=alpha, rng=rng,
                           logger=logger, controls=controls, req=req,
                           response=response, call_id=call_id, cached=cached,
                           discard=discard)


def begin_reflect(agent, *, step: int, writeback: str,
                  controls: str = "none") -> dict | None:
    """内省の発火ゲート(P2 S6b: 分割)。発火しないなら None。発火するなら
    reflect_step を消費し {"discard": bool} を返す(処理順は分割前と完全同一)。"""
    if step != agent.reflect_step:
        return None
    compute_matched = (controls == "compute_matched")
    # off かつ compute_matched 以外 = 現状動作(内省自体を行わない)
    if writeback == "off" and not compute_matched:
        return None
    agent.reflect_step = -1                        # 1睡眠(1帰宅)につき1回
    return {"discard": (writeback == "off")}       # compute_matched の off: 計算のみ・全破棄


def build_reflect_request(agent, *, step: int, sim_min: int, place_name: str,
                          date_line: str | None, weather_line: str | None,
                          reflect_cfg: dict | None, reflect_variety: bool,
                          interstitial_digest: str | None, interstitial: bool,
                          city_name: str, max_tokens: int, think: bool,
                          recalled: list[str],
                          recall_fail: str | None) -> dict:
    """内省本体の LLM 要求を組み立てる(P2 S6b)。generate_many 互換+apply 用メタ。"""
    # 深い内省の夜か。(1) 出来事誘発(第12バッチ・主経路): 衝撃ゲージの閾値超えが予約した
    # deep_due_day 以降の最初の夜(侵入的→熟慮的の遅延)。(2) レガシー固定周期(対照用)。
    day = sim_min // 1440
    rcfg = reflect_cfg or {}
    period = int(rcfg.get("self_model_days", 0))
    due = int(getattr(agent, "deep_due_day", -1))
    deep_event = bool(due >= 0 and day >= due)
    deep = deep_event or bool(period > 0 and day >= 1 and day % period == 0)
    task = _reflect_task(deep=deep, variety=reflect_variety,
                         agent_id=agent.id, day=day)   # A4: 既定 OFF=従来定数と同一
    if interstitial:                     # P2 S2: 物語化の1文を足す(ON時のみ=OFFは従来定数と同一)
        task = task + _STORY_NOTE
    prompt = (build_prompt(agent, place_name=place_name, surprise=None,
                           nearby_names=[], step=step, city_name=city_name,
                           date_line=date_line, weather_line=weather_line,
                           interstitial_digest=interstitial_digest)
              + (f"\n思い出したこと: {' / '.join(recalled)}" if recalled else "")
              + (f"\n{recall_fail}" if recall_fail else "")   # ACT-R OFF は None=1行も足さない
              + task)
    return {"prompt": prompt, "rng_key": f"reflect/{agent.id}/{step}",
            "temperature": 0.7, "max_tokens": max_tokens, "think": think,
            "deep": deep, "deep_event": deep_event, "day": day, "rcfg": rcfg}


def apply_reflect_response(agent, *, step: int, sim_min: int, writeback: str,
                           alpha: float, rng, logger: ObserverLogger,
                           controls: str, req: dict, response: str,
                           call_id: str | None, cached: bool,
                           discard: bool) -> None:
    """内省応答の適用(build_reflect_request と対・P2 S6b)。"""
    deep, deep_event = req["deep"], req["deep_event"]
    day, rcfg = req["day"], req["rcfg"]
    logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                         "purpose": "reflect", "step": step, "cached": cached})
    action = parse_action(response)
    if action is not None and action["type"] != "reflect":
        action = None
    belief = action.get("belief") if action else None

    # ---- 記憶の統合(off以外で実行 = 計算量同一。R1)。compute_matched の off は破棄 ----
    if not discard:
        agent.mem.consolidate(step,
                              action.get("summary") if action else None,
                              action.get("salient") if action else None,
                              agent_id=agent.id)

    # ---- belief の書き戻し(k のゲート)。discard は書き戻さない ----
    written = False
    if belief and not discard:
        if writeback == "free":
            written = True
        elif writeback == "degraded":
            written = bool(rng.random() < alpha)
        elif writeback == "sham":
            written = False   # 計算はした・結果は捨てる(R1 対照の要)
    if written:
        agent.beliefs.append(belief)

    # ---- 自己モデルの更新(第11バッチ。deep の夜のみ・k ゲートは belief と同一)----
    sm_updated = False
    if deep and written and action is not None:
        self_txt = str(action.get("self") or "").strip()[:80]
        ties_txt = str(action.get("ties") or "").strip()[:80]
        if self_txt:
            agent.self_model = {"self": self_txt, "ties": ties_txt, "day": day}
            sm_updated = True
    # ---- 出来事誘発の消費と不応期(第12バッチ)。深い内省を実行した夜に予約を解消 ----
    if deep_event:
        agent.deep_due_day = -1
        p = getattr(agent, "reflect_p", None)
        cool = int((p or {}).get("cooldown_days",
                                 rcfg.get("deep", {}).get("cooldown_days", 3)))
        agent.deep_cooldown_until_day = day + cool

    payload = {"mode": writeback, "written_back": written,
               "belief": belief if written else None,
               "summary": (action or {}).get("summary"),
               "n_salient": len((action or {}).get("salient") or [])}
    if controls != "none":                         # none は現状のイベント列を厳守(ゴールデン)
        payload["controls"] = controls
    if deep:                                       # 深い内省の夜のみ追記(OFF=キーなし=不変)
        payload["deep"] = True
        payload["cause"] = "event" if deep_event else "period"
        payload["self_model_updated"] = sm_updated
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id, kind="reflect",
                     x=agent.x, y=agent.y, llm_call_id=call_id, payload=payload))
