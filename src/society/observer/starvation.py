"""DPH-O = 「削ったことを必ず記録する」観測 4 点(既定 OFF・**世界を 1 バイトも動かさない**)。

正典: docs/plans/dayplan-horizon-plan.md §4.2(歪めないための保証設計)・§6 レーン DPH-O

何を解く問題か
--------------
本シムには「静かに消える」経路が 4 本あり、そのどれもが **L1 にも L2 にも痕跡を残さない**。
事後解析からは「起きなかった」と区別がつかないので、本選 10 日ランの結果が歪んでいても
それを検出する手段が無い。

  ① **返事の枯渇**(`reply_dropped`)
     `scheduler._decide` は返答保証の直前で `agent._reply_to` を None にクリアしてから
     予算を取りに行く。予算が切れていると**その返事は黙って消える**(`conv_turns_left` も
     減らない)。実測: cap 60 の 3 日ランで reply が −96%、08-23 時は 1 件も返らない。
     = 「話しかけたのに誰も返さない街」が観測不能なまま成立していた。

  ② **日跨ぎブロックの圧潰**(`wrap_clipped`)
     「23:00-02:00 に飲む」と書かれた計画は `repair` の `round` 演算子に
     23:00-23:10 へ潰される。`round` は「ほぼ全計画で起きる正規化」として
     `conflict_repair_rate` からも除外されているので、**1 つの指標にも出ない**。

  ③ **計画の欠落**(`plan_skipped`)
     予約された step に `sleeping` / `outside` だった個体は `plan_step` を消費済みのまま
     スキップされ、その暦日の計画を永久に失う。L1 に 1 件も残らない。

  ④ **予算枯渇の内訳**(`budget` の purpose 別カウンタ)
     L1 に残るのは `drive_request{granted:false}` だけで、**どの用途が落ちたか**は判らない。
     L2 へ列を足すのは `observer/aggregate.py` が凍結対象(metrics_spec.SPEC_FILES)なので
     不可 → summary.json の provenance へ出す(第86 day_plan / 第94 IF-B と同じ作法)。

  ⑤ **予算がどれだけ拘束していたか**(`llm_budget` の used / cap。第117 レーンB3)
     ④ は「落ちた呼」を数えるが、**cap そのものが binding だったのか**は判らない。
     1 step の実使用 `LodBudget.used` と上限 `max_per_step` を step ごとに拾い、
     `used_per_step`(平均 / p95)と `used_over_cap_mean`(= 予算の充填率)を出す。
     ★ p95 のために step 列は持たず**ヒストグラム**(used → 件数)で持つ:
       10 日ラン(1,440 step)でも cap+1 個の整数で足り、checkpoint も膨らまない。

契約
----
- **記録専用**: 分岐を 1 つも作らず、乱数 stream を 1 本も引かず、LLM 呼数を 1 も変えず、
  プロンプトを 1 バイトも変えない。ここが数えた値を読んで動く行はシム側に 1 行も無い。
- 既定 OFF では `enabled(sim)` が False = state が生えない = 新 kind 0 件 = L1 バイト一致。
- 累積タリーは `sim._starvation_state`(int と素の dict のみ)。checkpoint が中央管理する
  (保存しないと mid-day resume で summary の累積値が straight と食い違う。
   第86 day_plan / 第94 IF-B と同じ型のギャップ)。
"""
from __future__ import annotations

from .schema import Event, register_event_kind

SCHEMA = 1

register_event_kind(
    "reply_dropped",
    "★話しかけられたのに予算切れで返事が落ちた(DPH-O ①。既定 OFF では 0 件)。"
    "現行は agent._reply_to を消してから予算を取りに行くので、この 1 件が無いと"
    "「返さなかった」ことが原理的に観測できない {speaker, turns_left}")
register_event_kind(
    "plan_skipped",
    "★予約された朝の計画が立たなかった(DPH-O ③。既定 OFF では 0 件)。"
    "reason=sleeping / outside(予約 step に眠っていた・街の外に居た)/ "
    "defer_cap(二層予算の繰り越し上限を超えて骨格へ落ちた){reason, waited}")
register_event_kind(
    "reflect_dropped",
    "★夜の内省が二層予算の繰り越し上限を超えて諦められた(DPH-O ③ の内省版。"
    "既定 OFF・かつ lod.budget.tiers OFF では 0 件){waited}")


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
def enabled(sim) -> bool:
    """observer.starvation.enabled(既定 False)。"""
    flag = getattr(sim, "_starvation_on", None)
    if flag is None:
        try:
            raw = (sim.cfg.get("observer", None) or {}).get("starvation", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        flag = bool((raw or {}).get("enabled", False))
        sim._starvation_on = flag
    return flag


def _step_budget_zero() -> dict:
    """⑤ step 予算タリーの初期値(int と素の dict のみ。over_cap_sum だけ float)。"""
    return {"steps": 0, "used_sum": 0, "cap_sum": 0,
            "over_cap_sum": 0.0, "over_cap_steps": 0, "used_hist": {}}


def state(sim) -> dict:
    """累積タリー(checkpoint.py が中央管理する。ON のときだけ生える)。"""
    st = getattr(sim, "_starvation_state", None)
    if st is None:
        st = {"schema": SCHEMA, "reply_dropped": 0, "reflect_dropped": 0,
              "wrap_clipped": 0, "plan_expired_awake_steps": 0,
              "plan_expired_awake_agent_days": 0, "wrap_tail_lost": 0,
              "plan_skipped": {}, "budget": {},
              "step_budget": _step_budget_zero(),
              # ★B9: 「個体×日」の重複排除の印(**observer 側**に持つ。下記 `_expired_seen`)。
              #   当日ぶんだけを持ち、日が変わったら捨てる = 保持は 1 日分で有界。
              "plan_expired_seen_day": -1, "plan_expired_seen": {}}
        sim._starvation_state = st
    return st


def budget_counters(sim) -> dict | None:
    """LodBudget へ渡す purpose 別カウンタ(OFF は None = budget は dict も作らない)。"""
    return state(sim)["budget"] if enabled(sim) else None


# --------------------------------------------------------------------------- #
# 記録(全て「OFF なら即 return」)
# --------------------------------------------------------------------------- #
def note_reply_dropped(sim, agent, step: int, sim_min: int,
                       speaker_id) -> None:
    """① 予算切れで返事が落ちた。**予算を取り直さない**(呼数は 1 も動かない)。"""
    if not enabled(sim):
        return
    st = state(sim)
    st["reply_dropped"] += 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="reply_dropped", x=agent.x, y=agent.y,
                         payload={"speaker": int(speaker_id),
                                  "turns_left": int(getattr(agent, "conv_turns_left", 0))}))


def note_plan_skipped(sim, agent, step: int, sim_min: int, reason: str,
                      waited: int = 0) -> None:
    """③ 朝の計画が立たなかった(眠っていた / 街の外 / 繰り越し上限)。"""
    if not enabled(sim):
        return
    st = state(sim)
    st["plan_skipped"][reason] = st["plan_skipped"].get(reason, 0) + 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="plan_skipped", x=agent.x, y=agent.y,
                         payload={"reason": str(reason), "waited": int(waited)}))


def note_reflect_dropped(sim, agent, step: int, sim_min: int,
                         waited: int = 0) -> None:
    """③' 夜の内省が繰り越し上限を超えて諦められた(骨格に相当する退路が無いため)。"""
    if not enabled(sim):
        return
    st = state(sim)
    st["reflect_dropped"] += 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="reflect_dropped", x=agent.x, y=agent.y,
                         payload={"waited": int(waited)}))


def note_wrap_clipped(sim, n: int = 1) -> None:
    """② 日跨ぎ表記(end < start)が最小継続へ潰された件数。**イベントは出さない**。

    1 計画に複数件出うるうえ、潰れた事実は `plan_repair` の payload にも載る
    (day_plan 側が観測 ON のときだけ `wrap_clipped` キーを足す)ので、
    ここは累積タリーだけを持つ = L1 の行数を増やさない。
    """
    if not enabled(sim) or n <= 0:
        return
    state(sim)["wrap_clipped"] += int(n)


def _expired_seen(sim, day: int) -> dict:
    """★B9(第117 レーンE): 「今日この個体を既に数えたか」の印を **observer 側**で持つ。

    旧実装は `agent._plan_expired_day` を**個体へ書いて**いた。これは
    `checkpoint.save` の `agents` pickle に自然同梱されるので、**観測 ON/OFF で
    checkpoint のバイト列が変わる**(R1「観測は世界を 1 バイトも変えない」の文言違反)。
    エンジンは 1 行もこの属性を読まないので動力学は無傷だったが、
    「観測を入れたら保存物が変わる」は約束の破れそのものである。

    そこで印を `sim._starvation_state`(= observer 名前空間・checkpoint が
    `runtime["starvation_state"]` として**既に**中央管理している)へ移す。
    保持は**当日ぶんだけ**で、日が変わったら丸ごと捨てる:
      - `note_plan_expired_awake` の呼び出しは `sim_min` 単調増加の日境界処理なので、
        「前日の印」を残す必要がない = 旧属性方式と**厳密に同じ重複排除**になる。
      - したがってメモリ/checkpoint は「1 日に 1 度でも失効覚醒した個体数」で上限が付く
        (旧コメントが恐れた 25 万体ぶんの恒久集合にはならない)。
    値は set ではなく素の dict(`{agent_id: 1}`)で持つ = 本 state の
    「int と素の dict のみ」規約(module docstring)を保ち、pickle の集合反復順
    非保存の影響も受けない(membership しか使わない)。
    """
    st = state(sim)
    if int(st.get("plan_expired_seen_day", -1)) != int(day):
        st["plan_expired_seen_day"] = int(day)
        st["plan_expired_seen"] = {}
    seen = st.get("plan_expired_seen")
    if seen is None:                               # 旧 checkpoint から復元した state
        seen = {}
        st["plan_expired_seen"] = seen
    return seen


def note_plan_expired_awake(sim, agent, sim_min: int) -> None:
    """★深夜 0 時の無条件失効で「計画なしのまま覚醒していた」個体×step を数える。

    `個体×step` と `個体×日`(初回だけ数える)の 2 本を持つ。後者の重複排除は
    **observer 側の日別の印**(`_expired_seen`)で行う = agent には 1 バイトも書かない。
    """
    if not enabled(sim):
        return
    st = state(sim)
    st["plan_expired_awake_steps"] += 1
    day = int(sim_min) // 1440
    seen = _expired_seen(sim, day)
    aid = int(getattr(agent, "id", -1))
    if aid not in seen:
        seen[aid] = 1
        st["plan_expired_awake_agent_days"] += 1


def note_step_budget(sim, budget) -> None:
    """⑤ この step の LLM 予算の使用量(used / cap)を 1 件積む。**記録専用**。

    `scheduler.run_step` の末尾から 1 step につき 1 度だけ呼ぶ(`budget.reset()` が
    次 step の頭で `used` を 0 に戻すので、ここが読める最後の位置)。読むだけで
    `budget` にも `sim` にも書かない = 呼数・乱数・L1 のいずれも動かさない。
    """
    if not enabled(sim):
        return
    sb = state(sim).setdefault("step_budget", _step_budget_zero())
    used = int(getattr(budget, "used", 0))
    cap = int(getattr(budget, "max_per_step", 0))
    sb["steps"] = int(sb.get("steps", 0)) + 1
    sb["used_sum"] = int(sb.get("used_sum", 0)) + used
    sb["cap_sum"] = int(sb.get("cap_sum", 0)) + cap
    hist = sb.setdefault("used_hist", {})
    hist[used] = int(hist.get(used, 0)) + 1
    if cap > 0:                                    # cap=0 の step は比を作れない(0 と偽らない)
        sb["over_cap_sum"] = float(sb.get("over_cap_sum", 0.0)) + used / cap
        sb["over_cap_steps"] = int(sb.get("over_cap_steps", 0)) + 1


def _p95_from_hist(hist: dict) -> int:
    """ヒストグラム(値 → 件数)から p95 を最近傍順位法で取る(numpy 非依存・整数)。"""
    total = sum(int(v) for v in hist.values())
    if total <= 0:
        return 0
    need = -(-total * 95 // 100)                   # ceil(0.95 * N)
    acc = 0
    for value in sorted(int(k) for k in hist):
        acc += int(hist[value])
        if acc >= need:
            return int(value)
    return int(max(int(k) for k in hist))


def note_wrap_tail_lost(sim, n: int = 1) -> None:
    """日跨ぎの尾(まだ実行されていない翌日側のブロック)が新しい計画で上書きされた件数。"""
    if not enabled(sim) or n <= 0:
        return
    state(sim)["wrap_tail_lost"] += int(n)


# --------------------------------------------------------------------------- #
# summary.json(OFF はキー自体を出さない)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    if not enabled(sim):
        return None
    st = getattr(sim, "_starvation_state", None)
    if st is None:                                 # ON だが 1 件も起きていない(短いラン)
        st = state(sim)
    budget = {k: {"granted": int(v["granted"]), "denied": int(v["denied"])}
              for k, v in sorted(st["budget"].items())}
    total_g = sum(v["granted"] for v in budget.values())
    total_d = sum(v["denied"] for v in budget.values())
    # ⑤ step あたりの used / cap(旧 checkpoint から復元した state にはキーが無いので .get)
    sb = st.get("step_budget") or _step_budget_zero()
    n_steps = int(sb.get("steps", 0))
    n_ratio = int(sb.get("over_cap_steps", 0))
    used_hist = sb.get("used_hist") or {}
    llm_budget = {
        "steps": n_steps,
        "used_total": int(sb.get("used_sum", 0)),
        "used_per_step": {
            "mean": (round(int(sb.get("used_sum", 0)) / n_steps, 4)
                     if n_steps else 0.0),
            "p95": _p95_from_hist(used_hist)},
        "cap_per_step_mean": (round(int(sb.get("cap_sum", 0)) / n_steps, 4)
                              if n_steps else 0.0),
        # 予算の充填率(1.0 = 毎 step 使い切っていた = cap が完全に binding)
        "used_over_cap_mean": (round(float(sb.get("over_cap_sum", 0.0)) / n_ratio, 4)
                               if n_ratio else 0.0)}
    return {"schema": SCHEMA,
            "reply_dropped": int(st["reply_dropped"]),
            "reflect_dropped": int(st["reflect_dropped"]),
            "wrap_clipped": int(st["wrap_clipped"]),
            "wrap_tail_lost": int(st["wrap_tail_lost"]),
            "plan_expired_awake_steps": int(st["plan_expired_awake_steps"]),
            "plan_expired_awake_agent_days":
                int(st["plan_expired_awake_agent_days"]),
            "plan_skipped": {k: int(v) for k, v in sorted(st["plan_skipped"].items())},
            "llm_budget_by_purpose": budget,
            "llm_budget_granted_total": int(total_g),
            "llm_budget_denied_total": int(total_d),
            "llm_budget": llm_budget}
