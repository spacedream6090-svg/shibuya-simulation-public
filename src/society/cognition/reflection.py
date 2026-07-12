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


def _recall_query(agent, *, step: int, sim_min: int, llm, place_name: str,
                  logger: ObserverLogger, date_line: str | None = None,
                  weather_line: str | None = None) -> list[str]:
    """agentic pull 第1段: 何を思い出したいかを LLM に出させ、mem.query で想起する。

    LLM 呼び出しを1本足す(R1: writeback 条件に依らず一定)。想起自体は決定論。
    date_line/weather_line は当日の暦・天気(既定 None=注入せず従来と完全一致)。
    """
    prompt = (build_prompt(agent, place_name=place_name, surprise=None,
                           nearby_names=[], step=step,
                           date_line=date_line, weather_line=weather_line)
              + "\n今日の内省を始める前に、まず今日のことで思い出したいことを一つ挙げてください。"
              + "\n出力は次の JSON 1個のみ:"
              + '\n{"action": "recall", "query": "思い出したい事柄を短く"}')
    response, call_id, cached = llm.generate(
        prompt, rng_key=f"recall/{agent.id}/{step}", temperature=0.7,
        max_tokens=100, think=False)
    logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                         "purpose": "recall", "step": step, "cached": cached})
    action = parse_action(response)
    query_text = action.get("query", "") if action and action.get("type") == "recall" else ""
    hits = agent.mem.query(step, query_text, n=3) if query_text else []
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="memory_recall", x=agent.x, y=agent.y,
                     llm_call_id=call_id,
                     payload={"query": query_text, "n_hits": len(hits)}))
    return hits


def maybe_reflect(agent, *, step: int, sim_min: int, writeback: str, alpha: float,
                  llm, place_name: str, rng, logger: ObserverLogger,
                  max_tokens: int = 280, think: bool = False,
                  controls: str = "none", agentic_pull: bool = False,
                  date_line: str | None = None,
                  weather_line: str | None = None,
                  reflect_cfg: dict | None = None) -> None:
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
    """
    if step != agent.reflect_step:
        return
    compute_matched = (controls == "compute_matched")
    # off かつ compute_matched 以外 = 現状動作(内省自体を行わない)
    if writeback == "off" and not compute_matched:
        return
    agent.reflect_step = -1                        # 1睡眠(1帰宅)につき1回
    discard = (writeback == "off")                 # compute_matched の off: 計算のみ・全破棄

    # ---- agentic pull 第1段(常に+1呼。R1: writeback 条件に依らず一定)----
    recalled: list[str] = []
    if agentic_pull:
        recalled = _recall_query(agent, step=step, sim_min=sim_min, llm=llm,
                                 place_name=place_name, logger=logger,
                                 date_line=date_line, weather_line=weather_line)

    # 深い内省の夜か。(1) 出来事誘発(第12バッチ・主経路): 衝撃ゲージの閾値超えが予約した
    # deep_due_day 以降の最初の夜(侵入的→熟慮的の遅延)。(2) レガシー固定周期(対照用)。
    day = sim_min // 1440
    rcfg = reflect_cfg or {}
    period = int(rcfg.get("self_model_days", 0))
    due = int(getattr(agent, "deep_due_day", -1))
    deep_event = bool(due >= 0 and day >= due)
    deep = deep_event or bool(period > 0 and day >= 1 and day % period == 0)
    prompt = (build_prompt(agent, place_name=place_name, surprise=None,
                           nearby_names=[], step=step,
                           date_line=date_line, weather_line=weather_line)
              + (f"\n思い出したこと: {' / '.join(recalled)}" if recalled else "")
              + (_DEEP_REFLECT_TASK if deep else _REFLECT_TASK))
    rng_key = f"reflect/{agent.id}/{step}"
    response, call_id, cached = llm.generate(
        prompt, rng_key=rng_key, temperature=0.7, max_tokens=max_tokens,
        think=think)
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
                              action.get("salient") if action else None)

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
