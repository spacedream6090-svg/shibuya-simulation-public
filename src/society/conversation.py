"""会話3層 C2/C3(P2 スライス S3)= 構造化会話(LLMなし)+ すれ違い集計。

設計: docs/plans/p2-interstitial-design.md §1 S3 / docs/research/interstitial-life.md §4.3。

- **C1**(既存): フル LLM 発話(engine の _phase_drive → _llm_speak)。本モジュールは触らない。
- **C2**(本体): 空間索引で「近くに居て双方会話可能(睡眠でない・cooldown 明け・上限内)」の対を
  決定論列挙し、専用 stream `c2_meet` で会話成立を抽選する。成立したら Dialogue Act 列
  (挨拶→話題提示→スタンス表明→同意/反論→別れ)を**決定論遷移**で組む(実文は作らない)。
  話題・スタンス・トーン・帰結は両者の状態(関係値・opinion・共有語彙)からの決定論写像。
  機械的効果は既存資産を**呼ぶだけ**: 関係更新(relations/record_contact)・意見更新(opinion の
  FJ・k 非依存)・語彙接触(labels の complex contagion)・drive ゲージ加算(addressed)。
  1 会話 1 件の `conversation` イベント(実文なし=軽量)を記録する。
- **C2→C1 昇格**: 帰結が「強い意見差・関係の転機・未知語接触」を含むとき drive を余分に押し上げ、
  次以降の step の既存 _phase_drive がフル LLM 発話(C1)を自然に発火する。C2 自身は LLM を呼ばない。
- **C3**: すれ違いは会話イベントを作らず**日次カウンタ**(今日すれ違った人・挨拶した人)のみを
  agent の軽量フィールド(_c3_pass / _c3_greet)に置く。S2 のダイジェストが読める場所。

配置(src/society 直下・test_contracts の CHECKED_DIRS 外): opinion/relations/labels/drive の
不透明値だけを扱い、因子名・地名リテラルを一切持たない(no-fingerprint)。engine からは
scheduler._phase_c2 が本モジュールの run_phase を呼ぶだけ(engine への逆依存なし)。

R1(既定 OFF=バイト一致): enabled=false のとき run_phase は即 return し、新 stream `c2_meet` を
一度も引かず・`conversation` イベントを 0 件・agent に新フィールドを一切生やさない。ON でも
追加 LLM 呼はゼロ(呼数 k 非依存)。乱数は会話成立の抽選 `c2_meet` 1 本のみ。
"""
from __future__ import annotations

from .cognition import drive as drive_mod
from . import opinion as opinion_mod
from . import relations as relations_mod
from .observer.schema import Event
from .world import scene_desc as scene_desc_mod
from .world.perception import hearers_of

# ---- 既定値(conf/config.yaml の conversation ブロックが上書き)----
# meet_prob / cooldown_steps / daily_cap は「ON 時 ~30 会話/人日」を狙う初期値。
# 実測較正(mock 24step・n=60)の根拠は最終報告に記す(コードにマジックナンバーの由来を残さない方針)。
DEFAULTS = {
    "enabled": False,
    "c2": {
        "meet_prob": 0.8,        # 同席・会話可能な 1 対あたり毎 step の会話成立確率(c2_meet 抽選)
        "cooldown_steps": 2,     # 会話後にその個体が次の C2 に入れるまでの step 数(連発の抑制)
        "daily_cap": 40,         # 1 人 1 日あたりの C2 会話の上限(暴走防止・現実レンジ上限側)
        "align_agree": 0.72,     # opinion 整合度がこれ以上=同意スタンス
        "align_oppose": 0.38,    # opinion 整合度がこれ以下=反論スタンス(強い意見差=昇格材料)
        "tier_friendly": 1,      # 関係 tier がこれ以上(または count>=2)で友好トーン寄り
    },
}

# 話題・スタンス・トーン・帰結の語彙(payload 値。地名でも因子名でもない中立キー)。
_TOPIC_SMALLTALK = "smalltalk"   # 世間話(共有文脈が薄い)
_TOPIC_CATCHUP = "catchup"       # 近況(既知の間柄)
_TOPIC_OPINION = "opinion"       # 意見交換(意見差が主題)
_TOPIC_NOVELTY = "novelty"       # 新しい語・話題(未知語の紹介)


def build_cfg(raw) -> dict:
    """conf の conversation ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(relations.build_cfg と同じ流儀)。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    raw = dict(raw or {})
    cfg = {"enabled": bool(raw.get("enabled", DEFAULTS["enabled"]))}
    c2_raw = dict(raw.get("c2", {}) or {})
    c2 = dict(DEFAULTS["c2"])
    for k, v in c2_raw.items():
        if k not in c2:
            continue
        if k in ("cooldown_steps", "daily_cap", "tier_friendly"):
            c2[k] = int(v)
        else:
            c2[k] = float(v)
    cfg["c2"] = c2
    return cfg


def _cfg(sim) -> dict:
    """conversation 設定を遅延構築して sim にキャッシュ(simulation.py を編集せずに済ませる)。"""
    cfg = getattr(sim, "_c2cfg", None)
    if cfg is None:
        cfg = build_cfg(sim.cfg.get("conversation", None))
        sim._c2cfg = cfg
    return cfg


def enabled(sim) -> bool:
    return bool(_cfg(sim)["enabled"])


# ---------------------------------------------------------------- 会話内容(決定論写像)
def _rel_strength(a, b) -> tuple[int, float]:
    """(tier, count) を両者の関係台帳から取り出す(強い方を採用)。台帳が無ければ (0, 0)。"""
    tier = 0
    count = 0.0
    for x, y in ((a, b), (b, a)):
        rel = x.mem.relations.get(y.id)
        if rel:
            tier = max(tier, int(rel.get("tier", 0)))
            count = max(count, float(rel.get("count", 0)))
    return tier, count


def _novel_word(speaker, hearer) -> str | None:
    """speaker が採用済みで hearer が未採用の語のうち辞書順最小(決定論)。無ければ None。"""
    diff = speaker.adopted - hearer.adopted
    if not diff:
        return None
    return min(diff)


def plan_conversation(a, b, opinioncfg, c2cfg) -> dict:
    """両者の状態(関係・opinion・共有語彙)から会話の型・帰結を決定論写像する(乱数不使用)。

    返り(すべて決定論): topic/tone/stance/outcome/acts と、機械効果に使う中間量
    (opinion 整合度・関係 tier・a→b/b→a の新語)。実文テキストは一切作らない。
    """
    align = opinion_mod.alignment(a.opinion, b.opinion)   # 1=一致 .. 0=正反対(不透明スコア)
    tier, count = _rel_strength(a, b)
    acquainted = tier >= c2cfg["tier_friendly"] or count >= 2.0
    word_ab = _novel_word(a, b)                            # a が b に紹介し得る未知語
    word_ba = _novel_word(b, a)                            # b が a に紹介し得る未知語
    has_novel = bool(word_ab or word_ba)

    # スタンス(同意/中立/反論)= opinion 整合度の決定論しきい
    if align >= c2cfg["align_agree"]:
        stance = "agree"
    elif align <= c2cfg["align_oppose"]:
        stance = "oppose"
    else:
        stance = "neutral"

    # 話題(種別)= 共有文脈の決定論写像(新語 > 意見差 > 近況 > 世間話)
    if has_novel:
        topic = _TOPIC_NOVELTY
    elif stance == "oppose":
        topic = _TOPIC_OPINION
    elif acquainted:
        topic = _TOPIC_CATCHUP
    else:
        topic = _TOPIC_SMALLTALK

    # トーン = 関係 + 整合度の決定論写像
    if acquainted and stance != "oppose":
        tone = "friendly"
    elif stance == "oppose":
        tone = "cool"
    else:
        tone = "neutral"

    # 帰結(昇格材料)= 未知語接触 > 強い意見差 > 関係の転機 > 平穏
    if has_novel:
        outcome = "new_word"
    elif stance == "oppose":
        outcome = "opinion_gap"
    elif acquainted and stance == "agree" and tier >= 2:
        outcome = "rapport_shift"
    else:
        outcome = "uneventful"

    # Dialogue Act 列(挨拶→話題提示→スタンス表明→同意/反論→別れ)を決定論遷移で組む。
    # 実文は作らず、遷移の「数」だけを記録に残す(軽量)。中立の世間話は応答句を省く。
    acts = ["greet", f"topic:{topic}", f"stance:{stance}"]
    if stance in ("agree", "oppose"):
        acts.append("respond")
    acts.append("farewell")

    return {"topic": topic, "tone": tone, "stance": stance, "outcome": outcome,
            "acts": len(acts), "align": align, "tier": tier,
            "word_ab": word_ab, "word_ba": word_ba}


# ---------------------------------------------------------------- 機械的効果(既存資産を呼ぶだけ)
def _contact(sim, actor, other, valence: float, step: int, sim_min: int) -> None:
    """交流 1 件を actor→other の関係台帳へ反映(scheduler._contact と同型)。

    relations ON なら符号つき closeness 更新+tier 変化ログ、OFF なら従来の record_contact のみ。"""
    if sim.relationscfg["enabled"]:
        relations_mod.note_contact(actor, other.id, other.name, "", valence,
                                   sim.relationscfg, step, sim_min, sim.logger)
    else:
        actor.mem.record_contact(other.id, other.name, step, "")


def _shift_opinion(sim, listener, expressed_value: float, source_id: int,
                   step: int, sim_min: int) -> None:
    """聞いた相手の意見値へ FJ で寄せる(scheduler._shift_opinion と同型・決定論・k 非依存)。"""
    ocfg = sim.opinioncfg
    if not ocfg.get("enabled", True):
        return
    old = listener.opinion
    new = opinion_mod.expose(listener, expressed_value, ocfg["w_face"])
    if abs(new - old) < 1e-4:
        return
    listener.opinion = new
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=listener.id,
                         kind="opinion_shift", x=listener.x, y=listener.y,
                         payload={"source": source_id, "old": round(old, 4),
                                  "new": round(new, 4)}))


def _hear_word(sim, hearer, word: str | None, speaker_id: int,
               step: int, sim_min: int) -> None:
    """語 1 語の聴取(complex contagion 経路=labels.on_hear を呼ぶだけ)。未知語なら drive+。"""
    if not word:
        return
    got_unknown = sim.labels.on_hear(hearer, [word], speaker_id, step=step,
                                     sim_min=sim_min, channel="face",
                                     logger=sim.logger)
    if got_unknown:
        drive_mod.add(hearer, "unknown_word", sim.drivecfg)


def _apply_effects(sim, a, b, plan: dict, step: int, sim_min: int) -> None:
    """会話 1 件の機械効果を両者へ適用(すべて既存資産を呼ぶだけ・追加 LLM 呼ゼロ)。"""
    drivecfg = sim.drivecfg
    # (1) drive ゲージ: 双方「話しかけられた」= addressed 加算(会話は必ず相手に向く)。
    drive_mod.add(a, "addressed", drivecfg)
    drive_mod.add(b, "addressed", drivecfg)
    # (2) 関係更新: スタンスの符号を交流の感情価に(同意/中立=+、反論=−)。両方向に記録。
    valence = -0.5 if plan["stance"] == "oppose" else 0.4
    _contact(sim, a, b, valence, step, sim_min)
    _contact(sim, b, a, valence, step, sim_min)
    # (3) 意見更新(FJ・k 非依存): 互いに相手の意見値へ寄る(対面 w_face)。
    _shift_opinion(sim, a, b.opinion, b.id, step, sim_min)
    _shift_opinion(sim, b, a.opinion, a.id, step, sim_min)
    # (4) 語彙接触(complex contagion): 片方が知る未知語をもう片方が聞く。
    _hear_word(sim, b, plan["word_ab"], a.id, step, sim_min)
    _hear_word(sim, a, plan["word_ba"], b.id, step, sim_min)
    # (5) C2→C1 昇格: 帰結が顕著(強い意見差・関係の転機)なら drive を余分に押し上げる
    #     (未知語接触は (4) の unknown_word で既に押し上げ済み)。次 step の _phase_drive が
    #     フル LLM 発話(C1)を自然に発火する接続点。scale は不透明量(整合度/tier)から。
    if plan["outcome"] == "opinion_gap":
        gap = 1.0 - plan["align"]                         # 強い意見差ほど大きく押す
        drive_mod.add(a, "state_change", drivecfg, scale=gap)
        drive_mod.add(b, "state_change", drivecfg, scale=gap)
    elif plan["outcome"] == "rapport_shift":
        drive_mod.add(a, "state_change", drivecfg, scale=0.5)
        drive_mod.add(b, "state_change", drivecfg, scale=0.5)


# ---------------------------------------------------------------- C3(すれ違い集計)
def _roll_day(agent, day: int) -> None:
    """日境界で C2 日次カウンタ(会話上限)と C3 すれ違い集計をリセット(決定論)。"""
    if getattr(agent, "_c2_day", None) != day:
        agent._c2_day = day
        agent._c2_count = 0
        agent._c3_pass = set()        # 今日すれ違った人の id(N人=len)
        agent._c3_greet = set()       # 今日会話(挨拶)した人の id(M人=len)


# ---------------------------------------------------------------- フェーズ本体(オーケストレータ)
def run_phase(sim, step: int, sim_min: int) -> None:
    """C2/C3 フェーズ。既定 OFF=即 return(新 stream を引かず・イベント 0・新フィールド無し)。

    位置が確定した後(_phase_drive と同じく sim.percept_index を利用)に呼ばれる。会話成立は
    専用 stream `c2_meet` の抽選のみが乱数を消費する。会話内容・機械効果は決定論。
    """
    if not enabled(sim):
        return
    cfg = _cfg(sim)["c2"]
    idx = getattr(sim, "percept_index", None)
    if idx is None:
        return
    radius = float(sim.cfg.world.perception_radius_m)
    day = sim_min // 1440
    meet_prob = cfg["meet_prob"]
    cooldown = cfg["cooldown_steps"]
    cap = cfg["daily_cap"]

    # 会話可能な個体(在圏・非睡眠・非勾留)を id 昇順で。索引が既に睡眠/範囲外を除いている。
    active = sorted((a for a in sim.agents
                     if a.loc != "outside" and not a.sleeping
                     and step >= getattr(a, "detained_until", 0)),
                    key=lambda x: x.id)
    for a in active:
        _roll_day(a, day)
    conversed: set[int] = set()      # この step で既に会話した個体(1 step 1 会話に上限)

    for a in active:
        if a.id in conversed:
            continue
        for b in hearers_of(a, idx, radius):
            if b.id <= a.id or b.id in conversed:
                continue
            # 双方すれ違い(C3): 近くに居た=会話有無に関わらず「すれ違った」1 人に数える。
            a._c3_pass.add(b.id)
            b._c3_pass.add(a.id)
            # 双方会話可能か(cooldown 明け・日次上限内)。片方でも不可なら会話は成立しない。
            if step < getattr(a, "_c2_cooldown_until", 0) \
                    or step < getattr(b, "_c2_cooldown_until", 0):
                continue
            if a._c2_count >= cap or b._c2_count >= cap:
                continue
            # 会話成立の抽選(唯一の乱数消費 = 新 stream c2_meet)。対キー (小id, 大id, step)。
            rng = sim.hub.stream("c2_meet", a.id, b.id, step)
            if float(rng.random()) >= meet_prob:
                continue
            # --- 成立 ---: 会話内容を決定論写像し、機械効果を適用、conversation を 1 件記録。
            plan = plan_conversation(a, b, sim.opinioncfg, cfg)
            _apply_effects(sim, a, b, plan, step, sim_min)
            payload = {"with": b.id, "topic": plan["topic"],
                       "tone": plan["tone"], "outcome": plan["outcome"],
                       "acts": plan["acts"]}
            # 行間レイヤ配線(v0-2 S3): scene_desc ON なら「場所×注視対象」の中立要約を
            # payload メタに反映(実文は作らない)。OFF は key を足さない=従来 payload と同一。
            _scenecfg = getattr(sim, "scenecfg", None)
            if scene_desc_mod.enabled(_scenecfg):
                _meta = scene_desc_mod.conversation_meta(a, city=sim.city,
                                                         cfg=_scenecfg)
                if _meta:
                    payload["scene"] = _meta
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                 kind="conversation", x=a.x, y=a.y,
                                 payload=payload))
            a._c3_greet.add(b.id)
            b._c3_greet.add(a.id)
            a._c2_count += 1
            b._c2_count += 1
            a._c2_cooldown_until = step + cooldown
            b._c2_cooldown_until = step + cooldown
            conversed.add(a.id)
            conversed.add(b.id)
            break            # a はこの step もう会話しない(1 step 1 会話)
