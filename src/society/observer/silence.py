"""沈黙の第一級化 + 未定義行動レジスタ の観測(第70バッチ IDEA②。既定 OFF)。

狙い(docs/research/hackathon1-analysis/ideas-shortlist.md §2 ①⑥)
----------------------------------------------------------------
- **未定義行動**: 現行は enum 外の `action` 値を `parse_action` が None で捨て、scheduler が
  `fallback{reason:"parse_error"}` = **JSON 崩れと同じ箱**に入れて内容を失っていた。
  「与えられた行動空間で目的が達成できないとき空間そのものを拡張しようとする個体」は
  『世界を変えようとする個体』の操作的定義の最有力候補なので、**捨てずに記録**する。
- **沈黙**: 「関わらない自由」が選べないシムは現実から乖離する。`wander→stay` は「ぶらつく」で
  あって「誰とも関わらない」の意思表示ではない。`{"action":"nothing"}` を受理動詞にする。

L2 列(すべて既定 OFF = 列そのものが出ない = L2 バイト不変)
-----------------------------------------------------------
freedom.undefined_register = true のとき:
  undefined_action_total … L1 `undefined_action` の**ラン累積**件数
  undefined_action_rate  … 同 ÷ LLM 総呼数(llm_health の `llm_fallback_rate` と同じ分母)
      ★保存則: 振り分けなので `llm_fallback_rate + undefined_action_rate` が
        旧 `llm_fallback_rate` に一致する(同一ランで比較したときの検収条件)。
freedom.explicit_nothing = true のとき:
  silent_agent_rate   … 当日 speak/sns_post/dm が 1 件も無い個体の割合
  chosen_nothing_rate … 当日の `stay{reason:"chosen_nothing"}` ÷ 当日の `llm_deliberate`

resume の限界(正直註記)
------------------------
`undefined_action_*` は **プロセス開始からの累積**(llm_health 3 列と同じ性質)なので
resume すると 0 から数え直す。`silent_agent_rate` / `chosen_nothing_rate` は暦日タリーで、
state を checkpoint に載せていない(lens/structure/deviation と同流儀)ため mid-day resume
では当日分が途中から数え直しになる。いずれも **既定 OFF** なので既存の resume バイト一致
テストとは衝突しない。本選ランで ON にする場合は日境界での checkpoint を推奨。
"""
from __future__ import annotations

_UTTERANCE_KINDS = ("speak", "sns_post", "dm")
_CHOSEN_NOTHING = "chosen_nothing"


# --------------------------------------------------------------------------- #
# トグル取得(values.build_cfg で正準化済みの freedomcfg を読む。無ければ全て OFF)
# --------------------------------------------------------------------------- #
def _fcfg(sim) -> dict:
    return getattr(sim, "freedomcfg", None) or {}


def register_on(sim) -> bool:
    """未定義行動レジスタ(freedom.undefined_register)。"""
    return bool(_fcfg(sim).get("undefined_register", False))


def nothing_on(sim) -> bool:
    """沈黙の第一級化(freedom.explicit_nothing)。"""
    return bool(_fcfg(sim).get("explicit_nothing", False))


# --------------------------------------------------------------------------- #
# 未定義行動(累積カウンタ由来。走査ゼロ)
# --------------------------------------------------------------------------- #
def _llm_calls(sim) -> int | None:
    """LLM 総呼数(aggregate._llm_totals と同じ源)。取得不能なら None。"""
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    llm = getattr(sim, "llm", None)
    calls = getattr(llm, "calls", None)
    if calls is None:                     # 呼数カウンタを持たない backend(素の LLMBackend)
        calls = getattr(logger, "n_llm_rows", 0)
    return int(calls)


def undefined_scalars(sim) -> dict | None:
    """undefined_action の累積件数と率(OFF は None=列なし)。"""
    if not register_on(sim):
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    total = int(getattr(logger, "n_undefined_action_events", 0))
    calls = _llm_calls(sim)
    return {
        "undefined_action_total": total,
        "undefined_action_rate": round(total / calls, 6) if calls else 0.0,
    }


# --------------------------------------------------------------------------- #
# 沈黙(暦日タリー。lens/structure.observe と同型の増分走査)
# --------------------------------------------------------------------------- #
def _fresh() -> dict:
    return {"day": -1, "speakers": set(), "n_deliberate": 0, "n_nothing": 0}


def observe(sim) -> dict | None:
    """当日の発話者集合・熟慮件数・沈黙選択件数を積んで state を返す(OFF は None)。

    collect(sim) が step 末に 1 回だけ呼ぶ経路に載る。idempotent・決定論・読むだけ。
    `speakers` は membership と len のみ使用(集合反復順に依存しない)。
    """
    if not nothing_on(sim):
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # 単調増加
    processed = int(getattr(sim, "_silence_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):
        new = len(events)
    st = getattr(sim, "_silence_state", None)
    if st is None:
        st = _fresh()
        sim._silence_state = st
    for e in events[len(events) - new:]:
        d = int(e.sim_min) // 1440
        if d != st["day"]:                 # 暦日が進んだ → 当日タリーをリセット
            st["day"] = d
            st["speakers"] = set()
            st["n_deliberate"] = 0
            st["n_nothing"] = 0
        kind = e.kind
        if kind in _UTTERANCE_KINDS:
            if int(e.agent_id) >= 0:
                st["speakers"].add(int(e.agent_id))
        elif kind == "llm_deliberate":
            st["n_deliberate"] += 1
        elif kind == "stay" and e.payload.get("reason") == _CHOSEN_NOTHING:
            st["n_nothing"] += 1
    sim._silence_processed = total
    return st


def silence_scalars(sim) -> dict | None:
    """沈黙 2 列(OFF は None=列なし=L2 バイト不変)。"""
    st = observe(sim)
    if st is None:
        return None
    n_agents = len(getattr(sim, "agents", ()) or ())
    speakers = len(st["speakers"])
    silent = max(0, n_agents - speakers)
    n_del = int(st["n_deliberate"])
    return {
        "silent_agent_rate": round(silent / n_agents, 6) if n_agents else 0.0,
        "chosen_nothing_rate": (round(st["n_nothing"] / n_del, 6) if n_del else 0.0),
    }


COLUMNS = ("undefined_action_total", "undefined_action_rate",
           "silent_agent_rate", "chosen_nothing_rate")
