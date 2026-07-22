"""生活の偶発イベント層(第54バッチ 2026-07-23。ユーザー要望: 純観察=不確実性許容モード)。

外部方針(2026-07-23)「対照実験ができないスケジュールなので不確実性の高いシミュレーションにしても
いい。再現性を求めるシミュレーションと、純粋な観察なので不確実性を許容できるシミュレーションの選択が
できるようにしてほしい」への回答の柱の一つ。**システムの決定論(同 seed 同結果)は絶対に壊さない**。
不確実性は「エージェント視点の不確実な世界」= 人生に効く偶発イベント層として与える。

設計(uncertainty-audit.md §4 の採点表。採択第1候補=臨時収入/財布紛失・第2=偶然の出会いを単一
chance_event に束ねる):日次境界に専用 stream "chance"((agent, day) 個体別キー=編成順非依存)で
1人1日 daily_rate の確率で偶発事に遭い、重み付きカタログから1件を引いて効果を適用する:
  - windfall: 臨時収入(拾得・還付・当選小口)= money+。
  - loss:     財布紛失・急な出費             = money−(手持ちの範囲=外生流出)。
  - encounter: 偶然の再会/出会い             = 近傍 or 関係台帳の既知相手と closeness+(双方)。

効果は **money / relations / 記憶のみ**(既存の間接経路で自然に波及)。ドライブ/発火には一切流さない
=k の同定を汚さない。windfall/loss で「収入は運にほぼ無感応(第51実測 ΔR²≈0.001)」の穴に外生分散を
狙って注入でき、encounter で「関係は運が主経路(ΔR²≈0.19)」を最小コストで強める(=運/実力分解 §3)。

★このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  偶発ロジックはここと conf にのみ閉じる(no-fingerprint 契約=engine/scheduler には chance_mod.tick_day
  の呼び出しと chancecfg の組み立てだけを置き、因子名/構成概念は一切書かない)。

R1(バイト一致): 新 stream は "chance" 1本のみ・追加 LLM 呼ゼロ=呼数 k 非依存。判定は物理量
  (現在地・所持金・関係台帳の既知相手)・config・新 stream のみ参照し、k・内面状態(構成概念)を発火判断に
  食わせない。既定 OFF(enabled=false)= 抽選せず・chance_event 0 件・"chance" stream も引かない=
  乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。効果(money±/closeness+)は物理位置・対面
  co-location を変えうる(=FixedLLM で ON!=OFF になりうる=career G5 / crowd G4 / 健康 H1 と同型)が、
  機構は k を読まない=compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。
"""
from __future__ import annotations

from .observer.schema import Event

# カタログのイベント種別(重み付き抽選の固定順=決定論)。money 系=windfall/loss、関係系=encounter。
_KINDS = ("windfall", "loss", "encounter")


def build_cfg(raw) -> dict:
    """conf の chance ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(disaster/health/career と同型)。すべて既定 OFF
    (enabled=false)で scheduler の chance 日次フェーズが完全 no-op。

    パラメータ(ON 時のみ効く):
      daily_rate  1人1日あたり何かの偶発事に遭う確率(現実の桁で較正。下の較正根拠を参照)。
      events      データ駆動カタログ(重み付き):
        windfall  {weight, money_lo, money_hi}  臨時収入(拾得・還付・当選小口)= money+
        loss      {weight, money_lo, money_hi}  財布紛失・急な出費             = money−
        encounter {weight, closeness}           偶然の再会/出会い(近傍or既知相手)= closeness+(双方)
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    ev = dict(raw.get("events", {}) or {})
    wf = dict(ev.get("windfall", {}) or {})
    ls = dict(ev.get("loss", {}) or {})
    en = dict(ev.get("encounter", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "daily_rate": float(raw.get("daily_rate", 0.03)),
        "events": {
            "windfall": {"weight": float(wf.get("weight", 1.0)),
                         "money_lo": float(wf.get("money_lo", 1000.0)),
                         "money_hi": float(wf.get("money_hi", 20000.0))},
            "loss": {"weight": float(ls.get("weight", 1.0)),
                     "money_lo": float(ls.get("money_lo", 1000.0)),
                     "money_hi": float(ls.get("money_hi", 20000.0))},
            "encounter": {"weight": float(en.get("weight", 2.0)),
                          "closeness": float(en.get("closeness", 1.5))},
        },
    }


def enabled(sim) -> bool:
    """偶発イベント層(第54バッチ)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "chancecfg", None)
    return bool(cfg and cfg["enabled"])


# ---------------------------------------------------------------- 候補プール(encounter 用)
def _candidate_pools(sim) -> dict[int, list[int]]:
    """各 agent の encounter 相手候補を **抽選前に一括スナップショット** する(=編成順非依存)。

    候補 = その日 街に居る(loc!='outside')agent の { 関係台帳の既知相手 ∪ 同ノードの近傍 }。自分以外・
    現存(sim.agents に居る)のみ。**抽選より前に全員分を確定**することで、抽選中に発生する台帳変化
    (ある個体の encounter が別個体の台帳に自分を書き込む)が他個体の候補集合を動かさない=id 昇順の
    走査でも順序非依存(closeness の加算は可換・イベントは発火個体ごとに決定論)。sorted で決定論。"""
    active = {a.id for a in sim.agents}
    by_node: dict[str, list[int]] = {}
    for a in sim.agents:
        if a.loc != "outside":
            by_node.setdefault(a.node, []).append(a.id)
    pools: dict[int, list[int]] = {}
    for a in sim.agents:
        if a.loc == "outside":              # 街の外に居る個体は偶然の出会いなし(相手不在=不発)
            pools[a.id] = []
            continue
        cands: set[int] = set()
        for oid in a.mem.relations:         # 既知相手(再会)
            if oid in active and oid != a.id:
                cands.add(oid)
        for oid in by_node.get(a.node, ()):  # 近傍(同ノード=物理的な出会い)
            if oid != a.id:
                cands.add(oid)
        pools[a.id] = sorted(cands)
    return pools


# ---------------------------------------------------------------- 効果の適用
def _apply_money(sim, agent, ecfg: dict, sign: int, rng, step: int, sim_min: int) -> None:
    """windfall(sign>0)/ loss(sign<0)= 現金の外生の授受を正直に会計する(会計註記=balance を payload に)。

    金額は同 stream の uniform 抽選。windfall は money+、loss は money−(0 未満にしない=手持ちの範囲での
    現金流出)。記録する amount は **実際の増減の絶対値**(loss が床で頭打ちなら実損=payload と収支が一致)。
    経済会計は既存の severance/home_refill(現金への外生入金)と同じ流儀で money を直接動かす。"""
    lo = float(ecfg["money_lo"])
    hi = float(ecfg["money_hi"])
    if hi < lo:
        lo, hi = hi, lo
    draw = round(float(rng.uniform(lo, hi)))
    if draw <= 0:
        return
    old = float(agent.money)
    if sign > 0:
        agent.money = old + draw
        kind, note = "windfall", "臨時収入があった"
    else:
        agent.money = max(0.0, old - draw)
        kind, note = "loss", "お金を失った"
    amount = round(abs(agent.money - old), 1)
    if amount <= 0.0:                        # 実質変化なし(loss で手持ち 0)=記録しない
        return
    agent.remember(note)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="chance_event", x=agent.x, y=agent.y,
                         payload={"type": kind, "amount": amount,
                                  "balance": round(agent.money, 1)}))


def _apply_encounter(sim, agent, ecfg: dict, pools: dict, rng, step: int,
                     sim_min: int) -> None:
    """偶然の再会/出会い: 近傍 or 既知相手を決定論選定し、双方の closeness+ と remember 1行。

    相手選択は事前スナップショット pools からの一様抽選(同 stream。sorted 済み=決定論)。closeness は
    record_contact 流儀(closeness_delta)で **双方の台帳** に加算(相手が未知なら台帳エントリを作る=
    新しい出会い)。両者に「偶然○○に会った」を1件ずつ記憶させ、chance_event(type=encounter, other)を
    発火個体側に1件記録する。候補が居なければ不発(相手不在=何も起きない)。"""
    cands = pools.get(agent.id, ())
    if not cands:
        return
    other_id = cands[int(rng.integers(len(cands)))]
    other = sim.agent_by_id.get(other_id)
    if other is None:
        return
    delta = float(ecfg["closeness"])
    text = "偶然会った"
    agent.mem.record_contact(other_id, other.name, step, text, closeness_delta=delta)
    other.mem.record_contact(agent.id, agent.name, step, text, closeness_delta=delta)
    agent.remember(f"偶然{other.name}に会った")
    other.remember(f"偶然{agent.name}に会った")
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="chance_event", x=agent.x, y=agent.y,
                         payload={"type": "encounter", "other": int(other_id)}))


# ---------------------------------------------------------------- 日次フェーズ(1日1回)
def tick_day(sim, step: int, sim_min: int) -> None:
    """偶発イベントの日境界抽選。既定 OFF=完全 no-op(chance_event 0 件・"chance" stream も引かない)。

    各 agent は専用 stream "chance"((agent.id, day) 個体別キー=編成順非依存)から固定順に引く:
      ① rate ゲート(random() < daily_rate)② カタログの重み付き選択 ③ 効果側の抽選(money uniform or
      encounter の相手 index)。効果は money±/closeness+/記憶のみ=ドライブ/発火に流さない(k 非汚染)。
    encounter の候補は抽選前に全員分をスナップショット(_candidate_pools)して順序非依存を保つ。"""
    if not enabled(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_chance_day", -1):
        return
    sim._chance_day = day
    cfg = sim.chancecfg
    rate = float(cfg["daily_rate"])
    if rate <= 0.0:
        return
    events_cfg = cfg["events"]
    catalog = [(k, float(events_cfg[k]["weight"])) for k in _KINDS]  # 固定順=決定論
    total_w = sum(w for _k, w in catalog)
    if total_w <= 0.0:
        return
    pools = _candidate_pools(sim)                     # encounter 候補を抽選前に確定(順序非依存)
    for agent in sim.agents:                          # id 昇順=決定論
        rng = sim.hub.stream("chance", agent.id, day)  # 新 stream(既存 draw 順に影響しない)
        if float(rng.random()) >= rate:
            continue
        pick = float(rng.random()) * total_w          # 重み付き選択
        acc = 0.0
        chosen = catalog[-1][0]
        for name, w in catalog:
            acc += w
            if pick < acc:
                chosen = name
                break
        if chosen == "windfall":
            _apply_money(sim, agent, events_cfg["windfall"], 1, rng, step, sim_min)
        elif chosen == "loss":
            _apply_money(sim, agent, events_cfg["loss"], -1, rng, step, sim_min)
        else:
            _apply_encounter(sim, agent, events_cfg["encounter"], pools, rng,
                             step, sim_min)
