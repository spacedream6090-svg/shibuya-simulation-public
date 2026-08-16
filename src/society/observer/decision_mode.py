"""V3 = 決定モード印字(observer.decision_mode・**既定 OFF**)。

正典: docs/plans/external-audit-triage.md §3.2 V3(「決定モード印字(habit/rule/LLM を
      provenance へ)= 既存 causality/L1 で大半可視・**差分のみ**」)

何を解く問題か — 既に見えているもの / 見えていなかったもの
--------------------------------------------------------
本シムの「決定」は 3 レーンあり、**そのうち 2 本は既に完全に可視**である:

  - **朝の計画**: `plan_created.payload.src` ∈ {llm, prev_day, skeleton} が
    「その日の計画を誰が書いたか」をそのまま持つ。落ちた分は `plan_skipped{reason}`
    (DPH-O ③)。→ **新しい記録は要らない**。
  - **夜の内省**: `l1b_llm.purpose == "reflect"` が撃った分、`reflect_dropped`
    (DPH-O ③')が諦めた分。→ **新しい記録は要らない**。

見えていなかったのは **日中熟慮レーンの分母**である。`_decide` は毎 step・在場覚醒の
全個体について**必ず 1 つの行動を返す**が、そのうち L1/l1b に痕跡が残るのは

  - LLM が決めた分(`llm_deliberate` / `l1b_llm`)
  - 方針キャッシュが決めた分(`policy_reuse`。既定 OFF)

だけで、**ルール層(routine.decide)が決めた分は 1 バイトも残らない**。つまり

    「LLM 被覆率 = LLM が決めた決定 / 全決定」

の**分母が原理的に計算できない**状態だった(監査 F1 の 0.173 回/人/日は
「呼数 ÷ 人日」であって「決定に占める割合」ではない)。加えて

  - LLM を撃ったが JSON が読めず rule へ後退した分は `fallback{reason:"parse_error"}`
    に**trigger が載らない**(どの用途の熟慮が壊れたか判らない)
  - 予算切れで LLM に到達しなかった分と、そもそも発火権を得なかった分が混ざる
  - 「朝の計画のブロックを実行した」行動と「純粋な習慣」が rule の中で混ざる
    (前者は**間接的に LLM 由来**なので、被覆率の意味がまるで違う)

が判らなかった。本 module はこの 4 点**だけ**を埋める。

数え方(**唯一の不変式**)
------------------------
    points == llm + reuse + rule                     … 日ごとに厳密に成立

  - `points` … その step に `_decide` を通る個体数(= `len(active)`)を 1 step に
    1 度だけ足す。**決定 1 回につき 1**。
  - `llm`    … `llm_calls − llm_unparsed`。`_llm_speak_g` は 1 決定につき最大 2 回
    (返事 → 発火)撃つが、**成功した呼はその決定を必ず返す**(呼び出し側が
    `if action is not None: return action`)ので、差は「LLM が決めた決定の数」に
    一致する(2 回撃って 2 回とも壊れた決定は差 0 = rule 側へ落ちる)。
  - `reuse`  … 方針キャッシュ(kind="deliberate")の再利用。
  - `rule`   … ルール層が決めた決定を `note_rule` が**その場で 1 件**数える
    (残差ではない)。理由 × 出所の 2 軸で持つ。

不変式が破れたら summary の `residual` に非ゼロで出る(黙って合わせない)。

契約
----
- **記録専用**: 分岐を 1 つも作らず、乱数 stream を 1 本も引かず、LLM 呼数を 1 も変えず、
  プロンプトを 1 バイトも変えない。ここが数えた値を読んで動く行はシム側に 1 行も無い。
- 既定 OFF では `enabled(sim)` が False = state が生えない = **新 L1 kind 0 件・
  L1 の行も payload も 1 バイト不変**(出口は summary.json のキー 1 つだけ)。
- 累積タリーは `sim._decision_mode_state`(int と素の dict のみ)。checkpoint が
  中央管理する(保存しないと mid-day resume で summary の累積値が straight と
  食い違う。第86 day_plan / DPH-O starvation と同じ型のギャップ)。
- L2 へ列は足さない(`observer/aggregate.py` は metrics_spec.SPEC_FILES の凍結対象)。

正直な限界
----------
- `llm_calls`/`llm_unparsed` は**呼**の数、`rule` は**決定**の数。上の不変式は
  「llm = calls − unparsed」という**恒等式が成り立つ経路構造**に依存しており、
  それをテストで機械固定している(`_llm_speak_g` の成功は必ずその決定を返す)。
- 移動中(route を辿っている最中)の `continue` は rule に数える。計画ブロックへ
  向かって歩いている step でも「その step の決定はルールが下した」と読む。
  ブロックが**開始した** step だけが `plan:*` に立つ。
- `reply_starved` は「返事の予算が取れなかった」決定だが、その後に発火権があれば
  LLM が決めうる。理由は**最終的に rule で終わった決定にだけ**付く。
"""
from __future__ import annotations

SCHEMA = 1

# rule 決定の理由(= LLM が決めなかった理由)。**この 6 語から増やさない**。
RULE_REASONS: tuple[str, ...] = (
    "no_fire",          # 発火権も返事も無かった(= 純粋な習慣の step。圧倒的多数)
    "reply_starved",    # 話しかけられたが二層予算が取れず返事が落ちた(DPH-O ①)
    "reply_unparsed",   # 返事の LLM を撃ったが行動にならなかった(壊れた JSON / ablate.llm_off)
    "fire_unparsed",    # 発火の LLM を撃ったが行動にならなかった(同上)
    "template",         # 第87 engaged の定型返答(LLM を 1 本も呼ばない・既定 OFF)
    "detained",         # 勾留中(制度深化2・既定 0 では立たない)
)

# rule 決定の出所(その step の行動を実際に組み立てた層)。`plan:*` は朝の計画の
# ブロックが開始した step で、`*` にはその計画の来歴(llm / skeleton / prev_day)が入る。
RULE_SRC_HABIT = "habit"


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
def enabled(sim) -> bool:
    """observer.decision_mode.enabled(既定 False)。初回だけ conf を読んでキャッシュ。"""
    flag = getattr(sim, "_decision_mode_on", None)
    if flag is None:
        try:
            raw = (sim.cfg.get("observer", None) or {}).get("decision_mode", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        flag = bool((raw or {}).get("enabled", False))
        sim._decision_mode_on = flag
    return flag


def _zero_day() -> dict:
    """1 日ぶんのセル(int と素の dict のみ = checkpoint の pickle が自明)。"""
    return {"points": 0,
            "llm_calls": {},        # trigger → 撃った呼(l1b の deliberate 系と一致)
            "llm_unparsed": {},     # trigger → そのうち行動にならなかった呼
            "reuse": {},            # policy_cache の kind(deliberate / plan)→ 件数
            "rule": {}}             # "<reason>|<src>" → 件数(2 軸の平坦キー)


def state(sim) -> dict:
    """累積タリー(checkpoint.py が中央管理する。ON のときだけ生える)。"""
    st = getattr(sim, "_decision_mode_state", None)
    if st is None:
        st = {"schema": SCHEMA, "days": {}}
        sim._decision_mode_state = st
    return st


def _cell(sim, sim_min: int) -> dict:
    """その暦日のセル。日境界の定義は全 module 共通(sim_min // 1440)。"""
    day = str(int(sim_min) // 1440)
    days = state(sim)["days"]
    cell = days.get(day)
    if cell is None:
        cell = days[day] = _zero_day()
    return cell


def _bump(d: dict, key: str, n: int = 1) -> None:
    d[key] = int(d.get(key, 0)) + int(n)


# --------------------------------------------------------------------------- #
# 記録(全て「OFF なら即 return」)
# --------------------------------------------------------------------------- #
def note_points(sim, sim_min: int, n: int) -> None:
    """この step の決定点の数(= `_decide` を通る個体数)。**1 step に 1 度だけ**。

    逐次経路と一括発行経路(engine.batch_llm)は同じ `active` を回すので、
    どちらでも同じ値が立つ(= batch ON/OFF で分母が動かない)。
    """
    if not enabled(sim):
        return
    clear_pending(sim)                             # step 境界で一時スロットを必ず空にする
    if n > 0:
        _cell(sim, sim_min)["points"] += int(n)


def note_llm_call(sim, sim_min: int, purpose: str) -> None:
    """熟慮の LLM を 1 本撃った(`log_llm_call` と同じ位置 = l1b の行と 1:1)。"""
    if not enabled(sim):
        return
    _bump(_cell(sim, sim_min)["llm_calls"], str(purpose))


def note_llm_unparsed(sim, sim_min: int, purpose: str) -> None:
    """撃ったが行動にならなかった(壊れた JSON。`_log_reject` と同じ位置)。

    ★`fallback{reason:"parse_error"}` には trigger が載らないので、**用途別の
      内訳はここにしか無い**(既存 L1 からは復元できない)。
    """
    if not enabled(sim):
        return
    _bump(_cell(sim, sim_min)["llm_unparsed"], str(purpose))


def note_reuse(sim, sim_min: int, kind: str) -> None:
    """方針キャッシュの再利用(`policy_reuse` イベントと同じ位置)。kind=deliberate/plan。"""
    if not enabled(sim):
        return
    _bump(_cell(sim, sim_min)["reuse"], str(kind))


def note_plan_driven(sim, src: str) -> None:
    """朝の計画のブロックがこの step の行動を決めた(day_plan.plan_action の作用点)。

    ここでは**数えず**、直後に必ず来る `note_rule` が消費する一時スロットへ置くだけ
    (= 理由 × 出所の 2 軸を 1 セルで持つため)。スロットは `sim` の属性 1 つで、
    checkpoint にも L1 にも state にも出ない。`plan_action` は `routine.decide` の
    中の**同期呼び出し**(yield を挟まない)なので、一括発行で個体が入れ替わっても
    他個体のスロットを踏むことは構造的に起きない。
    """
    if not enabled(sim):
        return
    sim._decision_mode_pending = "plan:" + str(src or "")


def note_rule(sim, sim_min: int, reason: str) -> None:
    """ルール層がこの決定を下した(**決定 1 回につきちょうど 1 度**呼ばれる)。"""
    if not enabled(sim):
        return
    src = getattr(sim, "_decision_mode_pending", None) or RULE_SRC_HABIT
    sim._decision_mode_pending = None
    _bump(_cell(sim, sim_min)["rule"], f"{reason}|{src}")


def clear_pending(sim) -> None:
    """一時スロットを捨てる(`note_points` が step 境界で呼ぶ保険)。

    通常は `note_plan_driven` → `note_rule` が同じ 1 決定の中で必ず対になる
    (`plan_action` は `routine.decide` の中の同期呼び出しで、その戻り先が `note_rule`)。
    例外などで対が崩れても、**step を跨いで別の決定へ出所が漏れない**ことをここで保証する。
    """
    if getattr(sim, "_decision_mode_pending", None) is not None:
        sim._decision_mode_pending = None


# --------------------------------------------------------------------------- #
# summary.json(OFF はキー自体を出さない)
# --------------------------------------------------------------------------- #
def _roll(cell: dict) -> dict:
    """1 日ぶんのセル → 決定モードの内訳(share つき)。"""
    calls = sum(int(v) for v in cell["llm_calls"].values())
    unparsed = sum(int(v) for v in cell["llm_unparsed"].values())
    llm = calls - unparsed
    reuse = int(cell["reuse"].get("deliberate", 0))
    rule = sum(int(v) for v in cell["rule"].values())
    points = int(cell["points"])
    by_reason: dict = {}
    by_src: dict = {}
    for k, v in cell["rule"].items():
        reason, _, src = str(k).partition("|")
        _bump(by_reason, reason, int(v))
        _bump(by_src, src, int(v))

    def _sh(x: int) -> float:
        return round(x / points, 6) if points else 0.0

    return {
        "points": points,
        "llm": llm, "reuse": reuse, "rule": rule,
        "share": {"llm": _sh(llm), "reuse": _sh(reuse), "rule": _sh(rule)},
        # ★不変式 points == llm + reuse + rule の破れ。0 でないなら記録側のバグ
        #   (黙って合わせない = 発見できるようにする)。
        "residual": points - llm - reuse - rule,
        "llm_calls": {k: int(v) for k, v in sorted(cell["llm_calls"].items())},
        "llm_unparsed": {k: int(v) for k, v in sorted(cell["llm_unparsed"].items())},
        "reuse_by_kind": {k: int(v) for k, v in sorted(cell["reuse"].items())},
        "rule_by_reason": {k: int(v) for k, v in sorted(by_reason.items())},
        "rule_by_src": {k: int(v) for k, v in sorted(by_src.items())},
    }


def provenance(sim) -> dict | None:
    """summary.json の `decision_mode` ブロック(OFF は None = キー自体を出さない)。"""
    if not enabled(sim):
        return None
    st = getattr(sim, "_decision_mode_state", None)
    if st is None:                                 # ON だが 1 件も起きていない(短いラン)
        st = state(sim)
    days = st.get("days") or {}
    total = _zero_day()
    for cell in days.values():
        total["points"] += int(cell.get("points", 0))
        for name in ("llm_calls", "llm_unparsed", "reuse", "rule"):
            for k, v in (cell.get(name) or {}).items():
                _bump(total[name], str(k), int(v))
    return {"schema": SCHEMA,
            "lane": "deliberate",
            "total": _roll(total),
            "by_day": {d: _roll(days[d]) for d in sorted(days, key=lambda x: int(x))}}
