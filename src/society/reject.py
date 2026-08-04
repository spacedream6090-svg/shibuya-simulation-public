"""拒否通知の段階 conf 化 IF-B(``cognition.rejection_notify``・**既定 "silent"**)。

正典
----
- docs/research/llm-world-interface-audit.md **§2-C**(拒否3層と通知の現状 = 無音拒否の全リスト)
- docs/research/if-lane-research.md **§3**(工学 = Inner Monologue の feedback 3 種 /
  心理 = **Ovsiankina 効果**(中断課題の再開率 67%)+ Masicampo & Baumeister 2011
  「計画を立てるだけで侵入思考が消える」)・§6 の反映表 IF-2
- docs/plans/if-sv-p4-plan.md 「IF-B」行

★**Zeigarnik 効果は 2025 年のメタ分析(59 文献・中断/完了の再生比 0.99)で否定されている**。
  本 module の根拠として書いてはならない(if-lane-research §3-1 / §6)。生き残ったのは
  Ovsiankina 効果(中断は**再開/再計画**を誘発する)であり、本 module が測るのは
  「拒否後に記憶が強くなるか」ではなく「**拒否後に再計画が起きるか**」である。

何を解く問題か
--------------
監査 §2-C の層(c)「裁定拒否」には、通知される拒否と**まったく通知されない拒否**が
混在している。通知あり = 出店許可却下・破産直後出店・交際不成立・無許可摘発。
通知なし = 所持金不足の出店 / 敷金不足の転居 / 空き住戸なし / 交際申込の相手不在 /
経路が張れない / verify の no_target・no_witness・no_channel。

後者は SayCan が自ら述べた限界(「価値関数が弾いたことは分かるが、**なぜ弾かれたかが
返らない**」)と同じ位置にある。Inner Monologue はその直接の回答で、失敗の**記述**を
自然言語で戻すとループが閉じる。本 module はその 3 段階(なし / 記述 / 介入)を
conf の 3 水準として実装する:

    silent  … 現行と完全同一(無音。既定)
    memory  … 定型文 1 行を当人の記憶へ + L1 ``action_reject``
    engaged … memory に加えて認知イベントを「今」へ前倒し(= 既存 ``note_plan_exception``
              と同じ流儀。``cognition.engaged`` が OFF のランでは **memory へ縮退**)

no-fingerprint との両立(R1)
----------------------------
規約の本質は「**実験条件がプロンプトから漏れないこと**」であって「拒否理由を返さない
こと」ではない(if-lane-research §3-2)。そこで通知文を

    (行為種, 理由コード) の **純関数**

に閉じる。行為種も理由コードも有限集合(``KINDS`` / ``REASONS``)で、
**config・実験条件・k・エージェント特性・時刻・数値を 1 つも読まない**。
同じ拒否は全実験条件で同一バイト列になる(tests/test_rejection_notify.py が機械固定)。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``envfeedback.py`` は 1 バイトも変更しない。同 module の「新しい記憶文・欄・理由キーを
   足さない」は**意図的設計**であり、本バッチの対象外である。
2. ``truth_ledger.py`` も 1 バイトも変更しない。同 module は
   ``observer/metrics_spec.py`` の**凍結 14 ファイル**に入っており、1 文字でも触れば
   ``metrics_spec_hash`` が動いて過去ランとの指標比較が切れる。
   verify の空振り(no_target / no_witness / no_channel)の通知は、engine の
   ディスパッチャ側で「その呼び出しが出した L1 ``belief_verify`` の **outcome 1 語**」だけを
   読む ``note_verify`` で実現する(fact の ID も値も canary も読まない)。

本バッチの対象外(3 件・理由つき)
----------------------------------
1. **envfeedback 系(改札規制・遅延・poi_hold)の通知化** — envfeedback.py が
   「プロンプトに 1 バイトも足さない」を明文規約として持つ層。**規約の対象であって
   実装漏れではない**ので、通知化は本バッチの守備範囲に入れない。
2. **閉店が候補から静かに消える件** — これは「選択**前**のフィルタ」であって拒否ではない。
   本人は「行こうとして断られた」のではなく「最初からその候補を見ていない」。
   ここに通知を足すと、世界に存在しない出来事(申し込んで断られた経験)を捏造することになる。
3. **config ゲート層(b)の静かな無視** — 機能が OFF の世界に「できなかった」を通知するのは
   矛盾する。その世界では**その行為が存在しない**のであって、拒否されたのではない
   (day_plan の contingency が「前提機構 OFF の条件は常に不成立」としたのと同じ片側倒し)。

R1 ドクトリン
-------------
- 既定 ``silent`` では ``notify`` / ``note_why`` / ``why_line`` が即 return し、
  **L1 に 1 件も出ず・記憶に 1 行も入らず・agent に属性が生えず・プロンプトが 1 バイトも
  変わらない**(= ゴールデン L1 バイト一致。第75 dormant / 第93 provlink と同じ
  「OFF ではキー自体を作らない」流儀)。
- **乱数 stream を 1 本も引かない**(通知は完全決定論)。
- **generate() の呼び出しサイトを 1 つも作らない**。``engaged`` 水準の前倒しは既存の
  ``day_plan.note_plan_exception`` = 認知イベントキューの ``advance`` であり、
  新しい呼びではない(第87 engaged が「既存発火を束ねるだけ」としたのと同じ構造)。

正直な限界
----------
- 通知は**当人の記憶 1 行**であって、行動を強制しない。LLM が読んで何をするかは
  観測対象であり、本 module は「読める状態にする」までしかやらない。
- ``engaged`` 水準の前倒しは ``cognition.engaged``(→ ``cognition.fire``)が ON の
  ランでしか効かない。OFF のランでは **memory と完全に同一挙動**になる
  (縮退を隠さずテストで固定する)。
- 交際申込の「相手が既に交際中」は本バッチの対象に入れていない(監査 §2-C が挙げたのは
  「相手不在(近傍にいない)」1 件)。理由コードを増やす前に、対象一覧を監査と 1 対 1 に
  保つことを優先した。
"""
from __future__ import annotations

from .observer.schema import Event

SCHEMA = 1

# ---- 3 水準(Inner Monologue の no feedback / scene description / intervention)---- #
SILENT = "silent"
MEMORY = "memory"
ENGAGED = "engaged"
LEVELS: tuple[str, ...] = (SILENT, MEMORY, ENGAGED)

# --------------------------------------------------------------------------- #
# 有限語彙(SayCan の affordance が「有限スキル集合上の値関数」であることと同型)
#
# ★ここに無い (kind, reason) の組は通知できない = 自由文が通知経路へ入る余地が構造的に無い。
# --------------------------------------------------------------------------- #
KINDS: tuple[str, ...] = (
    "open_venture",          # 出店(tools._open_venture)
    "move_home",             # 転居(scheduler._apply_move_home)
    "propose_partnership",   # 交際の申込(scheduler._apply_partnership)
    "move_to",               # 移動(scheduler._apply の move_to)
    "verify",                # 確かめる(truth_ledger.apply_verify)
)
REASONS: tuple[str, ...] = (
    "no_money",     # 手持ち(+口座+融資)で足りない
    "no_room",      # 空いている住まいが無い
    "absent",       # 相手がその場にいない
    "unreachable",  # 経路が張れない
    "no_target",    # 確かめる対象が見当たらない
    "no_witness",   # 知っている人が近くにいない
    "no_channel",   # 調べる手立てが無い
)

# 「何をしようとして」= 行為種 → 定型の意志表現(**数値・時刻・固有名を入れない**)。
ACT_JP: dict[str, str] = {
    "open_venture": "店を出そう",
    "move_home": "住まいを移そう",
    "propose_partnership": "交際を申し込もう",
    "move_to": "その場所へ行こう",
    "verify": "本当かどうか確かめよう",
}
# 「何が障害だったか」= 理由コード → 定型の述語(同上)。
REASON_JP: dict[str, str] = {
    "no_money": "資金が足りなかった",
    "no_room": "空いている住まいが見つからなかった",
    "absent": "相手がその場にいなかった",
    "unreachable": "そこへ行く道が見つからなかった",
    "no_target": "確かめたいことが見当たらなかった",
    "no_witness": "知っている人が近くにいなかった",
    "no_channel": "調べる手立てがなかった",
}

# --------------------------------------------------------------------------- #
# 計画の失敗理由(day_plan の plan_exception。§3 = Ovsiankina / Masicampo & Baumeister)
#
# _replan が鳴るのは「must が開始予定を過ぎても始まっていない」ときだけなので、
# 理由は**時間の 2 種類しかない**(捏造して増やさない)。
# --------------------------------------------------------------------------- #
PLAN_REASONS: tuple[str, ...] = (
    "late",     # 開始予定を過ぎても始められなかった(ずらして立て直す)
    "missed",   # 動かせない予定の時間帯そのものが過ぎた(組めなかった)
)
PLAN_REASON_JP: dict[str, str] = {"late": "時間の都合", "missed": "時間切れ"}

# 計画の活動語 → 日本語(day_plan.ACTS と 1 対 1。テストが同期を機械固定する)。
PLAN_ACT_JP: dict[str, str] = {
    "work": "仕事", "study": "学び", "meal": "食事", "shop": "買い物",
    "leisure": "遊び", "park": "公園で過ごすこと", "walk": "散歩",
    "home": "家で過ごすこと", "visit": "人を訪ねること", "personal": "用事",
    "escort": "送り迎え", "meetup": "待ち合わせ",
}

# 一時属性名(**silent では 1 度も生えない**)。checkpoint は agents pickle に自然同梱。
WHY_KEY = "_plan_exception_why"


# --------------------------------------------------------------------------- #
# 定型規則(純関数。sim も agent も config も見ない = 全実験条件で同一バイト列)
# --------------------------------------------------------------------------- #
def sentence(kind: str, reason: str) -> str:
    """拒否の定型文。**(行為種, 理由コード) だけの純関数**。

    2 要素(何をしようとして・何が障害だったか)しか持たない。時刻・数値・固有名・
    実験条件は 1 つも入らない(tests が文面へ数字が混ざらないことを機械検査する)。
    """
    act, why = ACT_JP.get(str(kind)), REASON_JP.get(str(reason))
    if act is None or why is None:                 # 語彙外は通知しない(捏造しない)
        return ""
    return f"{act}としたが、{why}。"


def plan_sentence(act: str, reason: str) -> str:
    """計画の失敗理由の定型文(engaged REPLAN の状況行へ **1 行だけ**入る)。

    Inner Monologue の scene description に対応する 1 行で、同じく純関数。
    """
    word, why = PLAN_ACT_JP.get(str(act)), PLAN_REASON_JP.get(str(reason))
    if word is None or why is None:
        return ""
    return f"予定していた{word}が{why}でできなかった。"


# --------------------------------------------------------------------------- #
# 水準の解決(初回のみ読んでキャッシュ。キャッシュ属性は L1/L2/乱数/checkpoint に出ない)
# --------------------------------------------------------------------------- #
def level_of(sim) -> str:
    """``cognition.rejection_notify`` の実効水準(未知値・旧 config は silent へ倒す)。"""
    lv = getattr(sim, "_reject_level", None)
    if lv is None:
        try:
            raw = (sim.cfg.get("cognition", None) or {}).get("rejection_notify",
                                                             SILENT)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = SILENT
        lv = str(raw or SILENT)
        if lv not in LEVELS:
            lv = SILENT
        sim._reject_level = lv
    return lv


def enabled(sim) -> bool:
    """通知が実効か(= silent でない)。"""
    return level_of(sim) != SILENT


def _engaged_on(sim) -> bool:
    """``engaged`` 水準が実際に効くか。OFF なら memory へ縮退する(隠さない)。"""
    from .cognition import engaged as _engaged
    return _engaged.enabled(sim)


# --------------------------------------------------------------------------- #
# 集計(summary.json。checkpoint.py が runtime で中央管理する)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_reject_state", None)
    if st is None:
        st = {"schema": SCHEMA, "notified": 0, "advanced": 0, "degraded": 0,
              "why_set": 0, "why_cleared": 0, "why_injected": 0,
              "by_kind": {}, "by_reason": {}}
        sim._reject_state = st
    return st


def _bump(table: dict, key: str) -> None:
    table[key] = int(table.get(key, 0)) + 1


def provenance(sim) -> dict | None:
    """summary.json の ``rejection_notify`` キー(silent は None = キー自体を出さない)。"""
    if not enabled(sim):
        return None
    out: dict = {"schema": SCHEMA, "level": level_of(sim),
                 "effective_level": (ENGAGED if (level_of(sim) == ENGAGED
                                                 and _engaged_on(sim)) else MEMORY),
                 "n_kinds": len(KINDS), "n_reasons": len(REASONS)}
    st = getattr(sim, "_reject_state", None)
    if st is None:                                 # ON だが 1 件も拒否が起きていない
        out.update({"notified": 0, "advanced": 0, "degraded": 0, "why_set": 0,
                    "why_cleared": 0, "why_injected": 0,
                    "by_kind": {}, "by_reason": {}})
        return out
    out.update({
        "notified": int(st["notified"]), "advanced": int(st["advanced"]),
        "degraded": int(st["degraded"]), "why_set": int(st["why_set"]),
        "why_cleared": int(st["why_cleared"]),
        "why_injected": int(st["why_injected"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())},
        "by_reason": {k: int(v) for k, v in sorted(st["by_reason"].items())}})
    return out


# --------------------------------------------------------------------------- #
# 通知(裁定側からの**唯一の入口**)
# --------------------------------------------------------------------------- #
def notify(sim, agent, kind: str, reason: str, step: int, sim_min: int) -> bool:
    """拒否を当人へ通知する。**silent では完全 no-op**(False を返して 1 バイトも動かさない)。

    memory  … 定型文 1 行を ``agent.remember`` へ + L1 ``action_reject{kind, reason}``
    engaged … 上記 + 認知イベントの前倒し(既存 ``day_plan.note_plan_exception``)。
              ``cognition.engaged`` OFF のランでは前倒しを行わない = memory へ縮退。

    乱数ゼロ・LLM 呼ゼロ・完全決定論。世界状態(所持金・位置・関係)は 1 つも動かさない
    = 「通知」であって「救済」ではない(拒否そのものは従来どおり成立する)。
    """
    level = level_of(sim)
    if level == SILENT:
        return False
    text = sentence(kind, reason)
    if not text:                                   # 語彙外 = 通知しない(捏造しない)
        return False
    agent.remember(text, kind="reject")
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                         kind="action_reject", x=agent.x, y=agent.y,
                         payload={"kind": str(kind), "reason": str(reason)}))
    st = _state(sim)
    st["notified"] += 1
    _bump(st["by_kind"], str(kind))
    _bump(st["by_reason"], str(reason))
    if level == ENGAGED:
        if _engaged_on(sim):
            from .cognition import day_plan as _day_plan
            _day_plan.note_plan_exception(sim, agent, int(sim_min))
            # 前倒しは「fire の内部発火源が生きているとき」だけ実際に起きる。
            # 起きたかどうかは印で判定する(起きていないのに数えない)。
            if int(getattr(agent, "_plan_exception", -1)) == int(sim_min):
                st["advanced"] += 1
            else:
                st["degraded"] += 1
        else:                                      # 縮退を数える(隠さない)
            st["degraded"] += 1
    return True


def note_verify(sim, agent, events, step: int, sim_min: int) -> bool:
    """検証行動(verify)の空振りを当人へ通知する。**silent では呼ばれても no-op**。

    ★裁定 module(``truth_ledger.py``)を **1 バイトも変えない**ための作法。
      同 module は ``observer/metrics_spec.py`` の凍結 14 ファイルに入っており、
      1 文字でも触れば ``metrics_spec_hash`` が動いて過去ランとの比較が切れる。
      そこで engine 側で「その呼び出しが出した L1 ``belief_verify``」の **outcome だけ**を
      読み、有限語彙(``REASONS``)に載っていれば通知する。
    ★読むのは outcome 1 語だけで、fact の ID も値も canary も**一切触らない**
      (台帳の中身は engine へも当人へも渡らない = 既存の漏洩契約は無風)。
    """
    if not enabled(sim):
        return False
    for e in events:
        if e.kind != "belief_verify" or int(e.agent_id) != int(agent.id):
            continue
        reason = str((e.payload or {}).get("outcome", ""))
        if reason in REASONS:                      # ok / pending 系は拒否ではない
            return notify(sim, agent, "verify", reason, step, sim_min)
    return False


# --------------------------------------------------------------------------- #
# 計画の失敗理由(§3: 伝達 → 注入 → **降格**)
#
# Masicampo & Baumeister 2011 の直接の設計要求:「実行意図を形成するだけで侵入思考は
# 急減する」。したがって失敗理由は記憶に残しっぱなしにせず、**再計画が実った時点で消す**。
# 25万体スケールで「全員のプロンプトが失敗談で埋まる」劣化も同時に防ぐ。
# --------------------------------------------------------------------------- #
def note_why(sim, agent, act: str, reason: str) -> None:
    """計画の失敗理由を当人へ 1 件だけ控える(**silent では属性すら生えない**)。"""
    if not enabled(sim):
        return
    if str(act) not in PLAN_ACT_JP or str(reason) not in PLAN_REASON_JP:
        return                                     # 語彙外は控えない(捏造しない)
    setattr(agent, WHY_KEY, (str(act), str(reason)))
    _state(sim)["why_set"] += 1


def clear_why(sim, agent) -> None:
    """**降格**: 再計画が実った(代替ブロックが動き出した)= 侵入は止まる。

    ``silent`` では属性が生えないので常に no-op。日を跨いだ持ち越しは朝の計画生成
    (``day_plan.apply``)が同じ経路で消す = 「未解決のまま日を跨いだら消す」。
    """
    if getattr(agent, WHY_KEY, None) is None:
        return
    setattr(agent, WHY_KEY, None)
    if enabled(sim):
        _state(sim)["why_cleared"] += 1


def why_of(agent) -> tuple[str, str] | None:
    """控えてある失敗理由(無ければ None)。"""
    why = getattr(agent, WHY_KEY, None)
    if not why:
        return None
    return str(why[0]), str(why[1])


def why_line(sim, agent) -> str | None:
    """engaged の **REPLAN エピソード中だけ**プロンプトへ足す 1 行(他は None)。

    ★ここが本 module で唯一プロンプトに触れる場所。書くのは「予定していた〈act〉が
      〈reason〉でできなかった。」の定型 1 行だけで、機構語(発火・驚き・閾値・
      エピソード・再計画・例外)も実験条件語も因子名も数値も 1 文字も書かない。
    ★注入は**状況行の 1 行だけ**(§3「1 行だけ注入」)。記憶側には何も足さない
      (memory 水準の定型文とは別経路 = 二重に書かない)。
    """
    if not enabled(sim):
        return None
    why = why_of(agent)
    if why is None:
        return None
    from .cognition import engaged as _engaged
    if not _engaged.enabled(sim):
        return None
    ep = _engaged.episode_of(agent)
    if ep is None or ep["kind"] != _engaged.REPLAN:
        return None
    line = plan_sentence(why[0], why[1])
    if not line:
        return None
    _state(sim)["why_injected"] += 1
    return line
