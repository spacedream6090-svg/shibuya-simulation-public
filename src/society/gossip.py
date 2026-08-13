"""負の評判(悪評)の内生伝播 第61バッチ スライス(c)(2026-07-25。既定 OFF)。

現状の評判(relations.gain_reputation/reputation_decay)は「正の地位が上がり日次風化する」のみ
= 対人レベルの負の評判(ゴシップによる制裁)が無い=「いい人ばかりの世界」。語彙伝播で実証済みの
complex contagion(閾値2)を負方向に再利用し、悪評の内生伝播を実装する。**外部からの介入・危機注入
ではない**(悪評の種は既存の負イベントのみ=内生)。設計正典: devlog Entry 50 の 3 スライス精査。

  1. 種(seed): 既存の負イベント(破産・立退き・関係断絶・誤情報。種になるイベント種は conf の
     マップ gossip.seed_events=データ駆動でコードに固有名詞を書かない)の当事者が、(agent,day) 確率
     seed_prob×weight で「悪評タグ」化しうる。悪評タグは**内容を持たない匿名タグ**(「何をしたか」は
     イベント参照で観測側が L1 から再構成)=LLM に悪評文を生成させない(呼数不変・指紋回避)。
     種化した対象の**直近の会話相手**(既存の会話接触=k 非依存の観測量)が初期の「知る者」になる。
  2. 伝播(spread): complex contagion の負方向。悪評タグは既存の会話接触に乗り、**独立した複数の
     知人**(adopt_threshold=2。labels と同じ閾値設計だが「延べ回数」でなく「相異なる知人数」で数える
     =true complex contagion)から聞くと採用する。判定は (listener,day) キー決定論(RNG を引かない)。
  3. 忘却(fade): 日次で decay_prob で忘れる=永続の烙印にしない(現実の噂の風化)。

制裁(効果=内生・軽量。既定は効果最小・上限つき安全弁 max_penalty で孤立への滑り台にしない):
  - 会話相手選択の優先度低下(demote_partners。B3b の遭遇優先と同じ相手選択層のみ=会話は必ず起き相手
    だけ変わる=会話数・LLM 呼を作らない)。
  - 共同行動(joint)への誘い確率低下(joint_penalty。既存の誘い→承諾の確率式を下げるだけ=新 draw なし)。
  - status への効き(status_penalty。悪評の被知覚数/N を負項として小さく反映。status.py の合成に載る)。

配置(R9 / no-fingerprint): このファイルは src/society 直下=engine/cognition/actions/labeling/world の
  CHECKED_DIRS 外=no-fingerprint 契約に触れない(relations.py / mobility.py / status.py と同じ層)。
  悪評タグは efficacy/grievance 等の因子 state に結合しない(独立の観測スコア+既存確率の変調のみ)。

R1(既定 OFF=バイト一致): enabled=false のとき phase() は即 return し、専用 named stream "gossip" を
  一度も引かず・gossip_seed/spread/fade を 0 件・agent/sim に新フィールドを一切生やさない。効果フック
  (demote_partners/joint_penalty/status_penalty)も enabled=false なら agent 属性を読まず 0/素通り。
  ON でも LLM を 1 本も足さない(新 generate なし・プロンプト不変=呼数 k 非依存)。RNG は種と忘却の
  みが named stream "gossip"(伝播は閾値=RNG 不使用)。resume==straight: agent 側の状態(_gossip_known/
  _gossip_heard)は agents pickle に自然同梱、sim 側の日カウンタ/pending/state は checkpoint.py が中央管理
  (watermark はプロセス内 logger カウンタ由来なので保存せず load で 0 に戻す=第59 assets と同流儀)。
"""
from __future__ import annotations

from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    # ---- 種(seed): 既存の負イベント → 悪評タグ化 ----
    "seed_prob": 0.5,            # 負イベント当事者が悪評タグ化する (agent,day) 確率(× event weight)
    # 種になる既存イベント種 → 重み(コードに固有名詞を書かない=データ駆動の写像)。
    "seed_events": {
        "bankruptcy": 1.0,       # 自己破産(常に負)
        "eviction": 0.7,         # 立退き(回復 phase=rehoused は seed_exclude_payload で除外)
        "relation_break": 0.3,   # 関係断絶(原因側が特定できない=台帳所有者=正直に低確率)
        "misinfo": 0.6,          # 誤情報の発信者(訂正済みも含む=発信者は author 一貫)
    },
    # payload の値にこれが含まれるイベントは種にしない(eviction の回復 rehoused 等=データ駆動)。
    "seed_exclude_payload": ["rehoused"],
    "seed_contact_days": 1,      # 種化時、対象の直近この日数内に会話した相手を初期の知る者にする
    # ---- 伝播(spread): complex contagion ----
    "adopt_threshold": 2,        # 相異なる知人が何人で悪評を採用するか(labels と同じ閾値設計)
    # ---- 忘却(fade) ----
    "decay_prob": 0.15,          # 悪評を 1 日で忘れる確率(烙印にしない=噂の風化)
    # ---- 制裁(効果)。既定は効果最小・上限つき ----
    "max_penalty": 0.2,          # 各制裁効果の上限(孤立への滑り台にしない安全弁)
    "joint_penalty": 0.15,       # 悪評を知る相手を共同行動に誘う確率の低下幅
    "status_weight": 0.1,        # 被知覚数(/N)を status の負項に反映する重み(既定=効果最小)
}

_INT_KEYS = ("seed_contact_days", "adopt_threshold")
_FLOAT_KEYS = ("seed_prob", "decay_prob", "max_penalty", "joint_penalty", "status_weight")


# --------------------------------------------------------------------------- #
# cfg 正準化(assets.build_cfg / relations.build_cfg と同型: dict/OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の gossip ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(relations.build_cfg と同じ作法)。seed_events は
    「上書き」= 与えたキーだけ既定へ重ねる(値を書けば重なる)。"""
    raw = _to_plain(raw) or {}
    raw = dict(raw)
    cfg = dict(DEFAULTS)
    cfg["seed_events"] = dict(DEFAULTS["seed_events"])
    cfg["seed_exclude_payload"] = list(DEFAULTS["seed_exclude_payload"])
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "seed_events":
            m = dict(_to_plain(v) or {})
            cfg["seed_events"] = {str(kk): float(vv) for kk, vv in m.items()}
        elif k == "seed_exclude_payload":
            cfg["seed_exclude_payload"] = [str(x) for x in (list(v) or [])]
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
    return cfg


def cfg_of(sim) -> dict:
    """gossip 設定を返す(初回のみ sim.cfg.gossip から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(assets/status.cfg_of と同型)。
    キャッシュ属性 sim.gossipcfg は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "gossipcfg", None)
    if c is None:
        raw = None
        try:
            raw = sim.cfg.get("gossip", None)
        except Exception:
            raw = None
        c = build_cfg(raw)
        sim.gossipcfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 状態アクセサ(ON 経路でのみ agent 属性を生やす=OFF はバイト一致)
# --------------------------------------------------------------------------- #
def _known(agent) -> dict:
    """agent が悪評を知っている対象 target_id → 学習日。ON 経路でのみ生やす。"""
    d = getattr(agent, "_gossip_known", None)
    if d is None:
        d = {}
        agent._gossip_known = d
    return d


def _heard(agent) -> dict:
    """agent が対象 target_id について聞いた**相異なる知人**の集合(complex contagion の蓄積器)。"""
    d = getattr(agent, "_gossip_heard", None)
    if d is None:
        d = {}
        agent._gossip_heard = d
    return d


def perceived_counts(sim) -> dict:
    """target_id → 何人が悪評を知っているか(被知覚数)。決定論・読むだけ(agent 属性は生やさない)。"""
    counts: dict = {}
    for a in sim.agents:
        known = getattr(a, "_gossip_known", None)
        if not known:
            continue
        for t in known:
            counts[t] = counts.get(t, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 種のスキャン(毎 step・読むだけ・RNG/L1 なし): 新規 L1 から負イベント当事者を pending に積む
# --------------------------------------------------------------------------- #
def _scan_seeds(sim, cfg: dict) -> None:
    """前回処理済み総数 sim._gossip_watermark 以降の新規 L1 を走査し、種候補を pending に積む。

    assets.observe と同型の watermark(total=_n_flushed+len(events))で「新規イベントだけ」を見る
    (idempotent・監査済みの flush 対応)。RNG も L1 も出さない=当日のロールは _roll_day(日境界)が担う。
    pending は agent_id → (weight, cause_kind)。同一日に複数の負イベントがあれば最大 weight を採る。"""
    logger = getattr(sim, "logger", None)
    if logger is None:
        return
    seed_events = cfg["seed_events"]
    if not seed_events:
        sim._gossip_watermark = int(getattr(logger, "_n_flushed", 0)) + len(logger.events)
        return
    exclude = cfg["seed_exclude_payload"]
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_gossip_watermark", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):
        new = len(events)
    pending = getattr(sim, "_gossip_pending", None)
    if pending is None:
        pending = {}
        sim._gossip_pending = pending
    for e in events[len(events) - new:]:
        w = seed_events.get(e.kind)
        if w is None or e.agent_id < 0:
            continue
        if exclude and any(v in exclude for v in e.payload.values()):
            continue
        cur = pending.get(e.agent_id)
        if cur is None or w > cur[0]:
            pending[e.agent_id] = (float(w), e.kind)
    sim._gossip_watermark = total


# --------------------------------------------------------------------------- #
# 日境界(暦日1回): 種ロール + 伝播(complex contagion) + 忘却
# --------------------------------------------------------------------------- #
def _seed_knowers(sim, target, day: int, contact_days: int) -> list:
    """種化した対象の初期の知る者=対象の直近 contact_days 日内に会話した相手(既存の会話接触に乗る)。

    決定論(id 昇順)。来街者は街の外へ去る=知る者に含めない(伝播が続かない)。乱数不使用。"""
    out = []
    rels = getattr(target.mem, "relations", {}) or {}
    for oid in sorted(rels):
        rel = rels[oid]
        last = int(rel.get("last_step", 0))
        if (day - sim.clock.day(last)) <= contact_days:
            kn = sim.present_agent(oid)            # 街に居ない相手は知る者に含めない(幽霊)
            if kn is not None and not getattr(kn, "visitor", False) and kn.id != target.id:
                out.append(kn)
    return out


def _spread_for(sim, L, prev_day: int, thr: int, step: int, sim_min: int, st: dict) -> None:
    """listener L の complex contagion 伝播(前日 prev_day に会話した相手のうち悪評を知る者を数える)。

    L が前日会話した相手(relations の last_step が prev_day)を id 昇順に見て、その相手が知る各対象を
    L の「聞いた相異なる知人」集合に加える。相異なる知人が thr 人に達したら採用(gossip_spread)。
    判定は RNG を引かない=(listener,day) キー決定論。L 自身が対象の悪評は数えない。"""
    known_L = getattr(L, "_gossip_known", None)
    rels = getattr(L.mem, "relations", {}) or {}
    contacts = [oid for oid in sorted(rels)
                if sim.clock.day(int(rels[oid].get("last_step", 0))) == prev_day]
    if not contacts:
        return
    for oid in contacts:
        # ★伝播源も在場者のみ。退場者の ``_gossip_known`` は脱水時点の古い集合で、
        #   街に居ない人から「昨日聞いた」ことにはできない。
        src = sim.present_agent(oid)
        if src is None:
            continue
        src_known = getattr(src, "_gossip_known", None)
        if not src_known:
            continue
        for t in sorted(src_known):
            if t == L.id or (known_L is not None and t in known_L):
                continue
            heard = _heard(L)
            s = heard.get(t)
            if s is None:
                s = set()
                heard[t] = s
            s.add(oid)
            if len(s) >= thr:                      # 相異なる知人が閾値に到達=採用
                _known(L)[t] = sim_min // 1440
                del heard[t]
                st["spread"] += 1
                sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=L.id,
                                     kind="gossip_spread", x=L.x, y=L.y,
                                     payload={"target": int(t), "sources": thr}))


def _roll_day(sim, cfg: dict, day: int, prev_day: int, step: int, sim_min: int) -> None:
    """日境界の本体: 種ロール(pending)→ 伝播(前日接触)→ 忘却。RNG は named stream "gossip" のみ。"""
    logger = sim.logger
    st = getattr(sim, "_gossip_state", None)
    if st is None or st.get("day") != day:
        st = {"day": day, "seed": 0, "spread": 0, "fade": 0}   # L2 の当日カウンタ(日境界でリセット)
        sim._gossip_state = st
    thr = int(cfg["adopt_threshold"])
    seed_prob = float(cfg["seed_prob"])
    decay_prob = float(cfg["decay_prob"])
    contact_days = int(cfg["seed_contact_days"])
    counts = perceived_counts(sim)                 # target → 既知の知る者数(再種化回避に使う)

    # ---- 種(seed): pending の当事者を (target,day) 確率で悪評タグ化。RNG=stream("gossip","seed",..) ----
    pending = getattr(sim, "_gossip_pending", None) or {}
    for aid in sorted(pending):
        weight, kind = pending[aid]
        target = sim.agent_by_id.get(aid)
        if target is None or getattr(target, "visitor", False):
            continue
        if aid in counts:                          # 既に悪評が知られている=再種化しない
            continue
        rng = sim.hub.stream("gossip", "seed", int(aid), int(step))
        if rng.random() >= min(1.0, seed_prob * weight):
            continue
        seeded = _seed_knowers(sim, target, day, contact_days)
        for kn in seeded:
            _known(kn)[aid] = day
        if seeded:
            counts[aid] = counts.get(aid, 0) + len(seeded)
        st["seed"] += 1
        logger.log(Event(step=step, sim_min=sim_min, agent_id=int(aid),
                         kind="gossip_seed", x=target.x, y=target.y,
                         payload={"n": len(seeded), "cause": kind}))
    sim._gossip_pending = {}                        # 当日分を消費(次の蓄積は空から)

    # ---- 伝播(spread): complex contagion(前日接触・RNG 不使用=閾値判定)----
    if prev_day >= 0:
        for L in sim.agents:                        # id 昇順=決定論
            if getattr(L, "visitor", False):
                continue
            _spread_for(sim, L, prev_day, thr, step, sim_min, st)

    # ---- 忘却(fade): 知る各対象を decay_prob で忘れる。RNG=stream("gossip","fade",..)(seed と独立)----
    if decay_prob > 0.0:
        for a in sim.agents:                        # id 昇順=決定論
            # ---- レーン乙 F3: 「不在中は忘れない」バイアスを塞ぐ ----------------------
            # 忘却は日境界の Bernoulli 1 回。プール回転で街を出ていた個体はその日境界を
            # 一度も通らないので、悪評の生存確率が (1-p)^在場日数 になっていた(正しくは
            # (1-p)^経過日数)。またいだ本数 n を個体側の印から数え、**1 回の draw の
            # 閾値**を 1-(1-p)^n に上げる。★draw の本数は 1 本も増減しない = 消費列不変。
            # ★毎日在場(既定プロファイル)では n=1 = 閾値も decay_prob そのもの
            #   = 浮動小数のビットまで同一(下の n<=1 分岐)。
            last_tick = int(getattr(a, "_gossip_fade_day", -1))
            n = 1 if last_tick < 0 else max(1, int(day) - last_tick)
            a._gossip_fade_day = int(day)
            known = getattr(a, "_gossip_known", None)
            if not known:
                continue
            p_eff = decay_prob if n <= 1 else 1.0 - (1.0 - decay_prob) ** n
            rng = sim.hub.stream("gossip", "fade", int(a.id), int(step))
            for t in sorted(known):                 # sorted=pickle の dict 反復順非保存に非依存
                if rng.random() < p_eff:
                    del known[t]
                    st["fade"] += 1
                    logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                     kind="gossip_fade", x=a.x, y=a.y,
                                     payload={"target": int(t)}))


def phase(sim, step: int, sim_min: int) -> None:
    """毎 step の gossip フェーズ。種スキャン(毎 step)+ 種/伝播/忘却のロール(日境界のみ)。

    既定 OFF=即 return(stream を引かず・イベント 0 件・新フィールド無し=ゴールデン L1 バイト一致)。
    scheduler の run_step から infoenv(誤情報)確定後・collect(L2)前に呼ばれる=当日の負イベントを
    同 step のうちにスキャンし、L2 スカラーへ即反映する。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    _scan_seeds(sim, cfg)                            # 毎 step: 種候補の蓄積(RNG/L1 なし)
    day = sim_min // 1440
    prev_day = int(getattr(sim, "_gossip_day", -1))
    if day == prev_day:
        return
    sim._gossip_day = day
    _roll_day(sim, cfg, day, prev_day, step, sim_min)


# --------------------------------------------------------------------------- #
# 効果フック(制裁=既存確率/選択の変調のみ。OFF は 0/素通り=agent 属性を読まない)
# --------------------------------------------------------------------------- #
def demote_partners(sim, agent, hearers):
    """悪評を知る相手を返答相手選択から後退させる(B3b と同じ相手選択層)。全員が悪評対象なら
    そのまま返す(会話は必ず起きる=相手だけ変わる=会話数・LLM 呼を作らない)。呼び手(_select_partner)が
    enabled ゲート済み。読むだけ(agent 属性を生やさない)。"""
    known = getattr(agent, "_gossip_known", None)
    if not known:
        return hearers
    clean = [h for h in hearers if h.id not in known]
    return clean if clean else hearers


def joint_penalty(sim, inviter, cand_id) -> float:
    """共同行動の誘い→承諾確率の低下幅(誘い手が候補の悪評を知るとき)。OFF/非該当は 0.0=素通り。

    既存の誘い確率式(joint.accept_prob)を下げるだけ=新しい draw を足さない(joint stream の消費不変)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 0.0
    known = getattr(inviter, "_gossip_known", None)
    if not known or cand_id not in known:
        return 0.0
    return min(float(cfg["max_penalty"]), float(cfg["joint_penalty"]))


def status_penalty(sim, agent_id, perceived: dict) -> float:
    """悪評の被知覚数(/N)を status の負項に反映する幅(既定=効果最小・max_penalty で上限)。

    status.py の合成後スコアから減じる小さな負項。OFF は 0.0=status 不変。読むだけ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 0.0
    n = len(getattr(sim, "agents", []) or []) or 1
    frac = float(perceived.get(agent_id, 0)) / n
    return min(float(cfg["max_penalty"]), float(cfg["status_weight"]) * frac)


# --------------------------------------------------------------------------- #
# L2 全体スカラー 3 列(ON 時のみ・OFF は None=列なし=L2 不変。assets/structure と同型)
# --------------------------------------------------------------------------- #
def scalars(sim) -> dict | None:
    """L2 全体スカラー(OFF は None)。住民別は L2 に出さない(伝播経路は L1 から事後再構成)。

      gossip_active_count = 現在悪評を知られている人数(≥1 人に知られる対象の数)
      gossip_spread_total = 当日の伝播採用数(gossip_spread の当日件数)
      gossip_reach_mean   = 悪評 1 件あたり平均到達数(総知る者数 / 現アクティブ対象数)"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    counts = perceived_counts(sim)
    active = len(counts)
    total_knowers = sum(counts.values())
    reach = (total_knowers / active) if active else 0.0
    st = getattr(sim, "_gossip_state", None) or {}
    return {
        "gossip_active_count": int(active),
        "gossip_spread_total": int(st.get("spread", 0)),
        "gossip_reach_mean": round(float(reach), 4),
    }
