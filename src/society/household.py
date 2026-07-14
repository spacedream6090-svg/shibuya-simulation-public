"""世帯・家族・恋愛(現実ギャップ 後続波 H2 2026-07-07。ユーザー要望)。

現状のエージェントは全員独立個人。生活の紐帯(世帯・家族・同居・恋愛)を最小・非LLM・
決定論で載せる層。3機構(childbirth/死別=エージェント数が変わる重い機構は本波では作らない):

  1. 世帯・同居(build_households): 起動時に居住者を決定論で世帯(夫婦・家族・ルームシェア)へ
     グループ化する。同一世帯は住居(home_building/home_node/home_floor)を共有=夜間の家庭内
     co-location(会話の場)が生まれる。各メンバに household_id / 同居者 id(housemates)を付す。
  2. 家族関係・同居者との関係(context_line): 同居者・恋人が同席したとき、プロンプトへ
     「同居する家族/同居人の○○」「恋人の○○」を添える(既存 relation_line の隣、G2 と協調)。
  3. 恋愛・パートナー形成(form_partners): G2 relations の親密度 closeness が相互に非常に高い2者を
     決定論でパートナー(恋人)にする。パートナーは特別な間柄(親友の上位)+デート(date_dest)が
     移動理由に。partner_formed / life_event(kind="partner")を記録する。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  世帯グループ化・パートナー形成のロジックをここに閉じる=no-fingerprint 契約に触れない
  (パートナー形成は relations の closeness=不透明な数値のみを読み、構成概念名を engine 側に書かない)。

R1 呼数不変: どの機構も generate() を1本も足さない。世帯グループ化は決定論(startup。新 stream
  "household" のみ=既存 draw 順に挿入しない)、パートナー形成は relations closeness(観測イベント由来=
  k 非依存)からの決定論導出、デートは新 stream "date" のみ。住居共有・デートは物理位置を変え対面
  co-location が変化しうる(=FixedLLM で ON!=OFF になりうる=career G5 / crowd G4 / 健康 H1 と同型)が、
  機構は k・内面状態(構成概念)を発火判断に食わせず、名簿・config・relations の closeness・物理位置の
  み参照する=compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。

既定 OFF(enabled=false)= 世帯を組まず(home 割当も不変)・パートナー形成なし・context_line/date とも
  無し・"household"/"date" stream も引かない=イベント 0 件・乱数消費不変(ゴールデン
  golden_baseline_l1.json を守る)。新イベント種は partner_formed / life_event(schema.py 登録済み)。
"""
from __future__ import annotations

from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    # ---- 世帯グループ化(startup。新 stream "household")----
    "sizes": [1, 2, 3, 4],                 # 取りうる世帯サイズ(1=単身は世帯を組まない)
    "size_weights": [0.35, 0.30, 0.22, 0.13],  # サイズの相対重み(合計は正規化される)
    "family_ratio": 0.7,                   # 2人以上世帯が「家族」の割合(残りはルームシェア=同居人)
    # ---- 恋愛・パートナー形成(日次。G2 closeness から決定論)----
    "partner_closeness": 15.0,             # 相互 closeness がこれ以上でパートナー成立(親友 tier_close の上位)
    "date_bias": 0.5,                      # 自由時間にパートナーとの共有デート先へ寄せる確率/step(stream "date")
}

_BOOL_KEYS = ("enabled",)
_LIST_INT_KEYS = ("sizes",)
_LIST_FLOAT_KEYS = ("size_weights",)


def build_cfg(raw) -> dict:
    """conf の household ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(career/health/relations と同型)。すべて既定 OFF
    (enabled=false)で simulation の世帯構築・scheduler のパートナー形成・routine のデートが
    完全 no-op(イベント 0 件・新 stream も引かない=ゴールデンを守る)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _LIST_INT_KEYS:
            cfg[k] = [int(x) for x in v]
        elif k in _LIST_FLOAT_KEYS:
            cfg[k] = [float(x) for x in v]
        else:
            cfg[k] = float(v)
    return cfg


# ---------------------------------------------------------------- 世帯グループ化(startup)
def _pick_size(cfg: dict, rng) -> int:
    """世帯サイズを重み付きサンプル(新 stream "household"。決定論)。"""
    sizes = cfg["sizes"]
    weights = cfg["size_weights"]
    if not sizes:
        return 1
    total = float(sum(weights)) if weights else 0.0
    if total <= 0.0:
        return int(sizes[int(rng.integers(len(sizes)))])
    r = float(rng.random()) * total
    acc = 0.0
    for s, w in zip(sizes, weights):
        acc += float(w)
        if r < acc:
            return int(s)
    return int(sizes[-1])


def build_households(sim) -> None:
    """起動時1回: 居住者を決定論で世帯にまとめ、同一世帯で住居(home)を共有する。

    決定論: 居住者を id 昇順に並べ、新 stream "household" からサイズを引いて先頭から詰める
    (既存 draw 順に一切影響しない=OFF は完全 no-op)。2人以上の世帯にだけ household_id・
    同居者 id を付し、home_building/home_node/home_floor を代表(最小 id)へ寄せる(夜間の家庭内
    co-location)。simulation では本呼び出しの後に「顔なじみ」ブロックが home_building 共有から
    初期関係を張る=housemate が自然に顔なじみになる。来街者(街の外に家)は世帯を組まない。"""
    cfg = sim.householdcfg
    if not cfg["enabled"]:
        return
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    rng = sim.hub.stream("household")
    i = 0
    idx = 0
    while i < len(residents):
        size = _pick_size(cfg, rng)
        members = residents[i:i + size]
        i += len(members)
        if len(members) < 2:                       # 単身世帯は個人のまま(何も付けない)
            continue
        is_family = bool(rng.random() < float(cfg["family_ratio"]))
        rep = members[0]                           # 代表(最小 id)の住居を世帯で共有
        hh_id = f"hh{idx}"
        idx += 1
        for m in members:
            m.household_id = hh_id
            m.household_kind = "family" if is_family else "roommate"
            m.housemates = [o.id for o in members if o.id != m.id]
            m.home_building = rep.home_building
            m.home_node = rep.home_node
            m.home_floor = rep.home_floor


# ---------------------------------------------------------------- 家族/同居/恋人の文脈
def context_line(actor, nearby_ids, agent_by_id) -> str | None:
    """発火プロンプト用: 同席する同居者・恋人の間柄行を組み立てる(決定論・k 非依存)。

    同席者(nearby_ids)のうち恋人=「恋人の○○」、同居者=「同居する家族の○○」/「同居人の○○」。
    該当なしなら None(build_prompt は household 行を1行も足さない=OFF/非該当はバイト一致)。
    最大3件。名前は名簿(agent_by_id)から引く。"""
    housemates = set(getattr(actor, "housemates", None) or [])
    pid = getattr(actor, "partner_id", None)
    label = "同居する家族" if getattr(actor, "household_kind", "") == "family" else "同居人"
    parts: list[str] = []
    for oid in nearby_ids:
        meta = agent_by_id.get(oid)
        if meta is None:
            continue
        if pid is not None and oid == pid:
            parts.append(f"恋人の{meta.name}")
        elif oid in housemates:
            parts.append(f"{label}の{meta.name}")
    return "、".join(parts[:3]) if parts else None


# ---------------------------------------------------------------- デート(自由行動の行き先)
def date_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """パートナーとの共有デート先(自由時間の行き先バイアス)。決定論・非LLM。

    household OFF / パートナー無し / 抽選外 なら乱数を一切引かず None(既定不変)。専用 stream
    "date"(既存 draw 順を汚さない)で date_bias 抽選。行き先は (ペア, 当日) から純関数導出=両者が
    同一ノードを選ぶ=デートで合流(co-location)。移動(行き先)だけを変え、発火判断・LLM 呼数は
    増やさない(R1)。現在地と同じなら None(=stay)。"""
    cfg = sim.householdcfg
    if not cfg["enabled"]:
        return None
    pid = getattr(agent, "partner_id", None)
    if pid is None:
        return None
    partner = sim.agent_by_id.get(pid)
    if partner is None or partner.visitor:
        return None
    rng = sim.hub.stream("date", agent.id, step)
    if rng.random() >= float(cfg["date_bias"]):
        return None
    dests = sim.dests
    if not dests:
        return None
    day = sim_min // 1440
    lo, hi = (agent.id, pid) if agent.id < pid else (pid, agent.id)
    node = dests[(lo * 1000003 + hi * 97 + day) % len(dests)]
    return node if node != agent.node else None


# ---------------------------------------------------------------- 恋愛・パートナー形成(日次)
def _log_partner(a, b, step: int, sim_min: int, logger) -> None:
    """パートナー成立を a 視点で記録(partner_formed + life_event(kind="partner"))。"""
    logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                     kind="partner_formed", x=a.x, y=a.y,
                     payload={"other": int(b.id)}))
    logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                     kind="life_event", x=a.x, y=a.y,
                     payload={"kind": "partner", "other": int(b.id)}))


def bond(sim, a, b, step: int, sim_min: int) -> None:
    """2者をパートナーにする(partner_id 相互設定 + partner_formed/life_event + 記憶)。

    form_partners(日次の自動成立)と P2 の propose_partnership(#9)で共用する「世帯結合処理」。
    決定論・LLM 非増(呼び出し側が成立可否=closeness 閾値を判定してから呼ぶ)。"""
    a.partner_id = int(b.id)
    b.partner_id = int(a.id)
    _log_partner(a, b, step, sim_min, sim.logger)
    a.remember(f"{b.name}と付き合い始めた")
    b.remember(f"{a.name}と付き合い始めた")


def unbond(sim, a, step: int, sim_min: int) -> bool:
    """パートナー関係を解消する(P2 break_up #9)。解消したら True、独身なら False。

    既存 relation_break 流儀 + life_event(kind="breakup")を a 視点で記録し、双方の partner_id を
    外す(=世帯的な紐帯の分離)。相手側の応答は将来拡張(今回は起点側の決定で解消=正直な簡略化)。"""
    pid = getattr(a, "partner_id", None)
    if pid is None:
        return False
    b = sim.agent_by_id.get(pid)
    a.partner_id = None
    if b is not None and getattr(b, "partner_id", None) == a.id:
        b.partner_id = None
    other_name = b.name if b is not None else ""
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                         kind="relation_break", x=a.x, y=a.y,
                         payload={"other": int(pid), "from_tier": 3, "to_tier": 0,
                                  "cause": "breakup"}))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                         kind="life_event", x=a.x, y=a.y,
                         payload={"kind": "breakup", "other": int(pid)}))
    a.remember(f"{other_name}と別れた" if other_name else "恋人と別れた")
    if b is not None:
        b.remember(f"{a.name}と別れた")
    return True


def form_partners(sim, step: int, sim_min: int) -> None:
    """日次: G2 relations の親密度 closeness が相互に閾値超の2者を決定論でパートナーにする。

    closeness は relations(Wave G2)が交流の符号から積む観測量(k 非依存。relations OFF では
    台帳に closeness が無く=0=誰も成立しない)。id 昇順で走査し、独身の agent ごとに相互 closeness
    が partner_closeness 以上の相手(独身)を候補にし、相互 min closeness 最大(同点は id 最小)を選ぶ。
    成立で双方の partner_id を張り、partner_formed / life_event を記録する。決定論・LLM 非増(R1)。"""
    cfg = sim.householdcfg
    if not cfg["enabled"]:
        return
    thr = float(cfg["partner_closeness"])
    for a in sim.agents:                                   # id 昇順=決定論
        if a.visitor or getattr(a, "partner_id", None) is not None:
            continue
        cands = []
        for oid, rel in a.mem.relations.items():
            b = sim.agent_by_id.get(oid)
            if b is None or b.visitor or getattr(b, "partner_id", None) is not None:
                continue
            ca = float(rel.get("closeness", 0.0))
            if ca < thr:
                continue
            rb = b.mem.relations.get(a.id)
            cb = float(rb.get("closeness", 0.0)) if rb else 0.0
            if cb < thr:
                continue
            cands.append((min(ca, cb), -int(oid), b))       # (相互 closeness, -id, 相手)
        if not cands:
            continue
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)  # 相互最大→id 最小
        b = cands[0][2]
        bond(sim, a, b, step, sim_min)                        # 世帯結合処理(P2 と共用)
