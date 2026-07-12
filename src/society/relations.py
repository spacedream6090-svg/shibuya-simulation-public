"""社会関係の質(現実ギャップ Wave G2 2026-07-07)。

keystone(世界改変者)は社会資本の上に立つ。現状の関係台帳は「会話回数(count)」だけで
質的段階が無い。本モジュールは決定論・RNG不要で 4 機構を実装する:

  1. 関係の深化段階(tier): 交流の蓄積(closeness weight)から質的段階を導出
     (0=見知らぬ / 1=知人 / 2=友人 / 3=親友)。プロンプトに「○○とは友人」と載せる。
  2. 断絶/悪化(decay): ネガティブな交流(−neg_weight)・長期不在(日次減衰)で親密度が
     下がり、段階が落ちる(relation_break)。現状の「蓄積のみ」を破る。
  3. 評判/信頼(reputation): 各 agent の評判スコア(語の採用・被傾聴で伝播する社会的地位)。
     プロンプトに軽く反映(「街でそこそこ名前が知られている」)。
  4. 派閥/内外集団(軽量): 既存グループ(found_group/group_join)の共有からの「同じ仲間」
     プロンプト文脈のみ(重い分極力学は後続波)。

配置(R9 / no-fingerprint): tier/reputation のロジックは src/society 直下=検査対象外に置く。
tier/reputation は efficacy/grievance 等の因子 state に結合しない(独立の観測スコア+プロンプト
文脈に留める。G1 が grievance→発火の結合を保留したのと同様、結合可否は保留)。

決定論: すべて既生成の交流イベント(speak/hear/dm)の集計と日次減衰。乱数を一切引かない
(既存 stream の draw 順に無影響)。LLM 呼び出しも増やさない(R1: FixedLLM で ON==OFF 呼数一致)。

既定 OFF(enabled=false): scheduler がこのモジュールを一切呼ばず、record_contact も
closeness_delta を渡さない=関係台帳・プロンプト・イベント列がバイト一致で不変
(ゴールデン tests/data/golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

from .observer.schema import Event

# tier ラベル(プロンプト用)。0=見知らぬ(ラベルなし)/1=知人/2=友人/3=親友。
TIER_LABELS = {1: "知人", 2: "友人", 3: "親友"}

DEFAULTS = {
    "enabled": False,
    # ---- tier 導出: closeness weight(親密度の累積)の閾値。count 相当のスケール。----
    "tier_acquaintance": 2.0,     # これ以上で知人(tier 1)
    "tier_friend": 5.0,           # これ以上で友人(tier 2)
    "tier_close": 12.0,           # これ以上で親友(tier 3)
    # ---- closeness の増減(交流の符号)。ポジ交流で +pos、ネガ交流で −neg(悪化は速い)。----
    "pos_weight": 1.0,
    "neg_weight": 2.0,
    # ---- 断絶/悪化: 最終接触日から decay_after_days を超えて不在の相手を日次減衰。----
    "decay_per_day": 1.0,         # 不在1日ごとに closeness をこれだけ減らす
    "decay_after_days": 0,        # 最終接触からこの日数を超えた不在で減衰(0=翌日から)
    # ---- 評判/信頼スコア(語の採用・被傾聴で伝播する社会的地位)。----
    "rep_adopted": 1.0,           # 自分の造語が他者に採用されるたび +(影響の可視化)
    "rep_mention": 0.2,           # 聞き手のいる発話をするたび +(社会的可視性=名が広まる)
    "rep_decay_per_day": 0.5,     # 日次で評判は緩やかに風化
    "rep_known_threshold": 3.0,   # これ以上で「街で名が知られている」文脈をプロンプトに
    # ---- 派閥/内外集団(軽量): 同じグループ所属を「同じ仲間」1行に(重い力学は後続波)。----
    "faction": True,
}

_BOOL_KEYS = ("enabled", "faction")
_INT_KEYS = ("decay_after_days",)


def build_cfg(raw) -> dict:
    """conf の relations ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(config.py の作法を局所化)。"""
    raw = dict(raw or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        else:
            cfg[k] = float(v)
    return cfg


# ---------------------------------------------------------------- tier 導出(純関数)
def tier_of(closeness: float, cfg: dict) -> int:
    """closeness weight から質的段階を決定論算出(純関数)。0=見知らぬ..3=親友。"""
    if closeness >= cfg["tier_close"]:
        return 3
    if closeness >= cfg["tier_friend"]:
        return 2
    if closeness >= cfg["tier_acquaintance"]:
        return 1
    return 0


# ---------------------------------------------------------------- tier 変化の記録
def _log_tier(actor, other_id: int, old_tier: int, new_tier: int, count: int,
              cause: str, step: int, sim_min: int, logger) -> None:
    """tier が上がれば relation_tier、下がれば relation_break を actor 視点で記録(決定論)。"""
    if new_tier > old_tier:
        logger.log(Event(step=step, sim_min=sim_min, agent_id=actor.id,
                         kind="relation_tier", x=actor.x, y=actor.y,
                         payload={"other": int(other_id), "tier": int(new_tier),
                                  "count": int(count)}))
    else:
        logger.log(Event(step=step, sim_min=sim_min, agent_id=actor.id,
                         kind="relation_break", x=actor.x, y=actor.y,
                         payload={"other": int(other_id), "from_tier": int(old_tier),
                                  "to_tier": int(new_tier), "cause": cause}))


# ---------------------------------------------------------------- 交流(1件)の反映
def note_contact(actor, other_id: int, other_name: str, text: str,
                 valence: float, cfg: dict, step: int, sim_min: int,
                 logger) -> None:
    """1件の交流を actor→other の関係台帳へ反映する(closeness 更新 + tier 変化ログ)。

    record_contact を closeness_delta 付きで呼び(count/last_step/last も従来通り更新)、
    交流の符号で closeness を動かす。valence>=0=ポジ交流(+pos_weight)、<0=ネガ交流
    (−neg_weight)。closeness から tier を導出し、変化していれば relation_tier/relation_break
    を記録する。乱数・LLM を一切使わない。"""
    delta = cfg["pos_weight"] if valence >= 0.0 else -cfg["neg_weight"]
    rel = actor.mem.record_contact(other_id, other_name, step, text,
                                   closeness_delta=delta)
    old_tier = int(rel.get("tier", 0))
    new_tier = tier_of(float(rel["closeness"]), cfg)
    if new_tier != old_tier:
        rel["tier"] = new_tier
        _log_tier(actor, other_id, old_tier, new_tier, int(rel.get("count", 0)),
                  "contact", step, sim_min, logger)


# ---------------------------------------------------------------- 評判/信頼
def gain_reputation(agent, amount: float, cause: str, step: int, sim_min: int,
                    logger) -> None:
    """評判スコアを amount だけ上げて reputation_update を記録(観測量からの集計)。

    スコアは agent._reputation に動的属性で保持(既定 OFF 時は誰も触れない=バイト一致)。
    amount=0 なら記録もしない。cause=own_adopted(語が採用)/ mention(被傾聴で名が広まる)。"""
    if amount == 0.0:
        return
    old = float(getattr(agent, "_reputation", 0.0))
    new = old + float(amount)
    if new == old:
        return
    agent._reputation = new
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="reputation_update", x=agent.x, y=agent.y,
                     payload={"old": round(old, 4), "new": round(new, 4),
                              "cause": cause}))


def reputation_decay(agent, cfg: dict, step: int, sim_min: int, logger) -> None:
    """評判の日次風化(0 で床止め)。既に 0 なら何もしない(記録もしない)。"""
    old = float(getattr(agent, "_reputation", 0.0))
    if old <= 0.0:
        return
    new = max(0.0, old - cfg["rep_decay_per_day"])
    if new == old:
        return
    agent._reputation = new
    logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                     kind="reputation_update", x=agent.x, y=agent.y,
                     payload={"old": round(old, 4), "new": round(new, 4),
                              "cause": "decay"}))


# ---------------------------------------------------------------- 日境界(断絶/風化)
def decay_day(sim, cfg: dict, step: int, sim_min: int) -> None:
    """日境界: 最終接触が前日以前の相手の closeness を減衰(長期不在の断絶)+ 評判の風化。

    決定論(乱数なし・agent は id 昇順)。tier が下がれば relation_break を記録。closeness を
    持たない台帳(初期の顔なじみ・icebreak など生きた交流の無い相手)は減衰対象外
    (=それらの count ベース挙動は不変)。"""
    today = sim_min // 1440
    logger = sim.logger
    grace = int(cfg["decay_after_days"])
    for agent in sim.agents:
        reputation_decay(agent, cfg, step, sim_min, logger)
        for other_id, rel in agent.mem.relations.items():
            if "closeness" not in rel:
                continue
            last_day = sim.clock.day(int(rel.get("last_step", 0)))
            if (today - last_day) <= grace:        # 今日(や猶予内)接触した相手は減衰しない
                continue
            old_tier = int(rel.get("tier", 0))
            rel["closeness"] = float(rel["closeness"]) - cfg["decay_per_day"]
            new_tier = tier_of(float(rel["closeness"]), cfg)
            if new_tier != old_tier:
                rel["tier"] = new_tier
                _log_tier(agent, other_id, old_tier, new_tier,
                          int(rel.get("count", 0)), "absence", step, sim_min, logger)


# ---------------------------------------------------------------- プロンプト文
def social_lines(actor, nearby_ids: list[int], cfg: dict,
                 tools=None) -> tuple[str | None, str | None]:
    """発火プロンプト用の (間柄行, 評判行) を組み立てる(決定論)。

    間柄行: 同席者ごとに tier があれば「○○とは友人」等、無ければ従来の「N回話した仲」。
    派閥(faction)ON かつ同じグループ所属なら「(同じ仲間)」を添える。最大2件(既存踏襲)。
    評判行: 自分の評判が rep_known_threshold 以上のとき「街で名が知られている」1行。
    どちらも該当なしなら None(既定 OFF 経路は build_prompt が従来の relation_line に後退)。"""
    mem = actor.mem
    parts: list[str] = []
    for oid in nearby_ids:
        rel = mem.relations.get(oid)
        if not rel:
            continue
        tier = int(rel.get("tier", 0))
        shares = bool(cfg["faction"] and tools is not None
                      and tools.share_group(actor.id, oid))
        if tier >= 1:
            part = f"{rel['name']}とは{TIER_LABELS[tier]}"
        elif rel["count"] >= 2:
            part = f"{rel['name']}とは{rel['count']}回話した仲"
        elif shares:
            parts.append(f"{rel['name']}とは同じ仲間")
            continue
        else:
            continue
        if shares:
            part += "(同じ仲間)"
        parts.append(part)
    relation_line = "、".join(parts[:2]) if parts else None

    rep = float(getattr(actor, "_reputation", 0.0))
    reputation_line = ("あなたは街でそこそこ名前が知られている"
                       if rep >= cfg["rep_known_threshold"] else None)
    return relation_line, reputation_line
