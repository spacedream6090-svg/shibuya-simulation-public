"""朝の一日計画(ユーザー要望 2026-07-06)。

起床(来街者は帰還)の直後の step に1回だけ LLM を呼び、その人の「今日一日の予定」を
2〜5個立てる。予定は routine(LLM 非駆動の既定行動)の行き先選択の**土台**になる
(強制ではない: 既存の確率的行動はそのまま)。

呼び出しは個別の起床時刻に紐づくため自然に時間分散する。呼数は k 条件と無関係で
全員・毎朝1回(R1: 計画プロンプトには build_prompt と同じ文脈=beliefs も入るので、
k は「プロンプト内容」という正当な経路でのみ計画に作用する)。

出力 JSON(parse は deliberate.parse_action の "plan" 分岐):
  {"action": "plan", "items": [{"when": "朝|昼|午後|夕方|夜",
                                "what": "work|meal|shop|leisure|park|walk|home|visit",
                                "place": "任意の場所名"}, ...]}
壊れた/空なら計画なし(day_plan=[])= 従来ルーチンへフォールバック(D16)。
"""
from __future__ import annotations

from .. import schedule as _schedule
from ..observer.schema import Event
from . import plan_schema as _pf
from .deliberate import build_prompt, parse_action


def _today_schedule_line(sim, agent, sim_min: int) -> str | None:
    """当日の予定を計画プロンプトへ差し込む1行(schedule 有効時のみ。既定 None=不変)。"""
    scfg = getattr(sim, "schedulecfg", None)
    if not (scfg and scfg["enabled"] and scfg["inject_prompt"]):
        return None
    return _schedule.today_line(agent, today=sim_min // 1440)

# routine 側と共有する時間帯・行動カテゴリの語彙(自然文で提示する)。
_BANDS = "朝・昼・午後・夕方・夜"
_WHATS = "work(仕事)・meal(食事)・shop(買い物)・leisure(遊び)・park(公園)・" \
         "walk(散歩)・home(帰宅)・visit(訪問)"

_PLAN_TASK = (
    "\n今日一日の計画を、これからの時間帯ごとに 2〜5 個の予定として立ててください。"
    f"\n時間帯(when)は {_BANDS} のいずれか。内容(what)は {_WHATS} のいずれか。"
    "\n場所(place)は行きたい店・場所の名前(思いつかなければ空でよい)。"
    "\n出力は次の JSON 1個のみ(キー名は厳守):"
    '\n{"action": "plan", "items": ['
    '{"when": "朝", "what": "shop", "place": "場所名"}, ...]}')


def build_plan_prompt(agent, *, place_name: str, sim_min: int, step: int,
                      date_line: str | None = None,
                      weather_line: str | None = None,
                      schedule_line: str | None = None,
                      interstitial_digest: str | None = None,
                      city_name: str = "",
                      framework: dict | None = None) -> str:
    """build_prompt の文脈(ペルソナ・記憶・日記・信念)+ 所持金 + 計画タスク。

    ★ build_prompt の出力は既存の呼び出し(発話・内省)と共有するため一切変えない。
      計画用の追記(所持金・タスク)はここで後ろに足す(APC の prefix 一致も保たれる)。
    date_line/weather_line は当日の暦・天気(既定 None=注入せず従来と完全一致)。
    schedule_line は当日の予定(schedule 有効時のみ。None=注入せず不変)。
    interstitial_digest は前回発火以降の客観ダイジェスト(P2 S2。既定 None=注入せず不変)。
    framework は日課計画フレームワーク設定(P2 S1)。None=既定 OFF=スキーマ指示を足さない
    =従来のプロンプトとバイト一致。有効時のみ末尾にスキーマ指示を1ブロック足す。
    """
    base = build_prompt(agent, place_name=place_name, surprise=None,
                        nearby_names=[], sim_min=sim_min, step=step,
                        city_name=city_name,
                        date_line=date_line, weather_line=weather_line,
                        schedule_line=schedule_line,
                        interstitial_digest=interstitial_digest)
    money_line = f"\n今の所持金: 約{int(getattr(agent, 'money', 0.0))}円"
    task = _PLAN_TASK if framework is None else _PLAN_TASK + _pf.schema_prompt(framework)
    return base + money_line + task


def _normalize(items: list, max_items: int) -> list:
    """LLM 出力を計画リストに整える。各項目に done フラグを付け、件数を丸める。"""
    out: list = []
    for it in items[:max_items]:
        if not isinstance(it, dict):
            continue
        out.append({"when": str(it.get("when") or "").strip(),
                    "what": str(it.get("what") or "").strip(),
                    "place": str(it.get("place") or "").strip(),
                    "done": False})
    return out


def make_plan(sim, agent, step: int, sim_min: int, place_name: str,
              interstitial_digest: str | None = None) -> None:
    """1回の LLM 呼び出しで今日の計画を生成し、agent.day_plan に格納する。

    失敗・空でも day_plan=[](従来ルーチン)にし、day_plan イベントは必ず1件出す
    (1人1日1回の観測・R1 監査のため)。翌朝はこの関数がまた呼ばれて上書きする。
    interstitial_digest は前回発火以降の客観ダイジェスト(P2 S2。既定 None=注入せず不変)。
    """
    max_items = int(sim.planningcfg.get("max_items", 5))
    framework = _pf.framework_cfg(sim)               # None=既定 OFF=現行経路(バイト一致)
    prompt = build_plan_prompt(agent, place_name=place_name, sim_min=sim_min,
                              step=step,
                              city_name=getattr(sim, "place_name", ""),
                              date_line=getattr(sim, "today_date_line", None),
                              weather_line=getattr(sim, "today_weather_line", None),
                              schedule_line=_today_schedule_line(sim, agent, sim_min),
                              interstitial_digest=interstitial_digest,
                              framework=framework)
    # 朝の計画だけ上限を分けられる seam(model.plan_max_tokens)。
    # 未設定 or 0/null は model.max_tokens にフォールバック=既定挙動は完全不変。
    _plan_mt = int(sim.cfg.model.get("plan_max_tokens", 0) or 0)
    plan_max_tokens = _plan_mt if _plan_mt > 0 else int(sim.cfg.model.max_tokens)
    response, call_id, cached = sim.llm.generate(
        prompt, rng_key=f"plan/{agent.id}/{step}",
        temperature=float(sim.cfg.model.temperature),
        max_tokens=plan_max_tokens)
    sim.logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                             "purpose": "plan", "step": step, "cached": cached})
    if framework is not None:                        # 日課計画フレームワーク経路(P2 S1)
        _make_plan_framework(sim, agent, step, sim_min, prompt, response, call_id,
                             framework, max_items, plan_max_tokens)
        return
    action = parse_action(response)
    items = _normalize(action["items"], max_items) \
        if action is not None and action.get("type") == "plan" else []
    agent.day_plan = items
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="day_plan", x=agent.x, y=agent.y,
                         llm_call_id=call_id,
                         payload={"n": len(items),
                                  "plan": [{"when": it["when"], "what": it["what"],
                                            "place": it["place"]} for it in items]}))


def _parse_plan_items(response: str, framework: dict, max_items: int):
    """応答 → 正規化済みスキーマ items(不成立なら None)。parse_action が決定論修復
    (_loads_lenient)を内包するので、ここは F3 第1段(修復)を通した結果を受ける。"""
    action = parse_action(response)
    if action is None or action.get("type") != "plan":
        return None
    items = _pf.normalize_items(action["items"], framework, max_items)
    return items or None


def _make_plan_framework(sim, agent, step: int, sim_min: int, prompt: str,
                         response: str, call_id: str | None, framework: dict,
                         max_items: int, plan_max_tokens: int) -> None:
    """フレームワーク経路(ON)。失敗の階段(F3): 決定論修復(_parse_plan_items 内)→
    再試行1回 → 前日計画流用 → 初日は職業別デフォルト骨格。コンパイル結果は day_plan へ
    降格写像し、要約を day_plan_compiled(ON のみ・1日1件)へ記録する。"""
    items, path = _parse_plan_items(response, framework, max_items), "llm"
    if items is None:                                # 修復しても不成立 → 再試行1回
        retry, rcall_id, rcached = sim.llm.generate(
            prompt, rng_key=f"plan/{agent.id}/{step}/retry",
            temperature=float(sim.cfg.model.temperature),
            max_tokens=plan_max_tokens)
        sim.logger.log_llm_call({"llm_call_id": rcall_id, "agent_id": agent.id,
                                 "purpose": "plan_retry", "step": step,
                                 "cached": rcached})
        items, path, call_id = _parse_plan_items(retry, framework, max_items), \
            "retry", rcall_id
    if items is None:                                # 再試行も不成立 → 前日計画流用
        prev = getattr(agent, "_plan_prev_day", None)
        if prev:
            items, path = _pf.normalize_items(prev, framework, max_items), "prev_day"
        else:                                        # 初日: 職業別デフォルト骨格
            items, path = _pf.default_skeleton(agent, framework), "default"
    schedule, day_plan, summary = _pf.compile_plan(agent, sim, items, sim_min,
                                                    framework)
    agent.day_plan = day_plan
    agent.day_schedule = schedule                    # 観測・S4/S6 実行接続の受け皿
    agent._plan_prev_day = [it for it in items if not it.get("anchor")]  # 翌日の流用元
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="day_plan", x=agent.x, y=agent.y,
                         llm_call_id=call_id,
                         payload={"n": len(day_plan),
                                  "plan": [{"when": it["when"], "what": it["what"],
                                            "place": it["place"]} for it in day_plan]}))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="day_plan_compiled", x=agent.x, y=agent.y,
                         llm_call_id=call_id,
                         payload={"n": summary["n"], "cats": summary["cats"],
                                  "anchors": summary["anchors"], "path": path}))
