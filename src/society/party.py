"""来街者 party の実体化(関係性の再現 第45バッチ S-R5。ユーザー要望 2026-07-21)。

プールの L4 来街者 record に `party_size` 1-5(build_persona_pool.py=同行人数)が実在するが
sim 内で未使用=来街グループが「同席する連れ」として実体化されていなかった。本モジュールは
その日 present な来街者を party_size に従ってグループ化し「同席の連れ」にする:

  - **同一入街時刻・同一初期ノード**: グループの代表(最小 id)のノードに連れを寄せる(present
    な来街者はローテーション/起動時の同一日境界で実体化=同時入街。node を揃えて同席させる)。
  - **相互 relations 注入**: 連れ同士を record_contact で相互に知人〜友人へ(closeness 直接注入)。
  - **回遊の共有**: 連れの回遊先=代表と共有の POI(party_today)。回遊帯に共有 POI へ寄せる
    (party_dest。新 stream "party")=連れが一緒に動く。
  - **観測**: 実体化の時点で連れが同一初期ノードに同席=joint_activity{type:"party"} を1件記録
    (既存 kind 再利用=過剰な新 kind を作らない)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  較正値・グループ化ロジックはここ(と conf)にのみ書く=no-fingerprint 契約に触れない。

R1・presence 整合: presence(P3 純関数=in world/presence.py)の draw 順を一切変えない。party は
  present 判定の**後**に、実体化済みの sim.agents(present な来街者)だけをグループ化する=presence
  純関数(pid, day のみ依存)に触れない=rotation の決定論・resume バイト一致(test_pool_rotation)を
  保つ。乱数は新 stream "party" のみ(回遊バイアス抽選)。同一初期ノードへの寄せ・relations 注入は
  乱数ゼロ。連れの同席は物理位置=対面 co-location を変えうる(household/joint/career と同型で許容)→
  呼数不変は compute_matched 下の k 不変性で担保する。

既定 OFF(enabled=false)= 何もグループ化せず・party_today も付かず・"party" stream も引かず・
  joint_activity 0 件=乱数消費・イベント列ともバイト一致(ゴールデン golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    "max_party": 5,            # party_size の上限(プールは1-5)
    "closeness": 5.0,          # 連れ相互に注入する closeness(relations の友人 tier2 相当)
    "tier": 2,                 # 注入する relations tier(2=友人)
    "roam_bias": 0.5,          # 回遊帯に共有 POI へ寄せる確率/step(新 stream "party")
    "band_start_min": 600,     # 回遊帯の開始(分 of day)= 10:00
    "band_end_min": 1320,      # 回遊帯の終了(分 of day)= 22:00
}

_BOOL_KEYS = ("enabled",)
_INT_KEYS = ("max_party", "tier", "band_start_min", "band_end_min")
_FLOAT_KEYS = ("closeness", "roam_bias")


def build_cfg(raw) -> dict:
    """conf の party ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
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
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
    return cfg


# ---------------------------------------------------------------- 純関数ヘルパ
def _party_size(agent) -> int:
    """来街者 record 由来の同行人数(build_pool_agent が record["party_size"] を保持)。無ければ 1。"""
    try:
        return int(getattr(agent, "party_size", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _shared_poi(sim, leader_id: int, day: int) -> str | None:
    """(代表 id, day) から回遊の共有 POI を1つ純関数導出(全員が同じ先を選ぶ=合流)。"""
    dests = sim.dests
    if not dests:
        return None
    return dests[(leader_id * 1000003 + day * 97) % len(dests)]


def _inject(a, b, tier: int, closeness: float) -> None:
    """a→b の連れ関係を tier/closeness へ確定注入(直接代入=二重辺でも膨らませない)。"""
    rel = a.mem.record_contact(b.id, b.name, 0, "連れ")
    rel["closeness"] = float(closeness)
    rel["tier"] = int(tier)


def _fold_trip(agent) -> None:
    """移動中の旅を畳む(= 位置を据え替える前の整合クリア)。冪等・L1 を 1 件も出さない。

    ★第109バッチで判明した**本選ブロッカーの本体**: 寄せるときに ``node`` だけ書き換えて
      ``route`` を残すと、次の ``_phase_move`` が (新 node, 旧 route[0]) という**存在しない辺**を
      引いて ``KeyError: The edge (...) is not in the graph`` で死ぬ。起動時(day0)の実体化直後は
      全員 route が空なので露見せず、**日次ローテーション**(``_phase_pool_rotation``)で
      「前日から在場していて今まさに移動中の来街者」が連れになった瞬間に落ちる(実測 step 144=day 1)。

    表現は既存の「旅を畳む」前例に揃える(``health._exit_world`` の despawn / 執行の勾留
    ``route/dest/exit_intent/homing/activity`` クリア / ``_maybe_lodge`` の
    ``_pending_stay/_pending_activity/_ride_pending`` クリア)。新しい語彙は 1 つも作らない。
    ``edge_offset`` を 0 に戻すのは「ノードちょうどに立っている」ことの表明(経路再付与の
    各サイトが 0 を書く既存作法と同じ)。"""
    agent.route = []
    agent.dest = None
    agent.edge_offset = 0.0
    agent.trip_mode = "walk"
    agent.exit_intent = False          # 退出途中の旅も畳む(移動をやめた個体が意図だけ持ち続けない)
    agent.homing = False
    agent.activity = ""
    agent._pending_activity = ""       # 到着時に消費するはずだった予約(到着はもう来ない)
    agent._pending_stay = 0
    agent._ride_pending = None         # 未到着の乗車=運賃を取り損ねない(到着課金の前提を残さない)


# ---------------------------------------------------------------- 実体化(present 判定の後)
def form_parties(sim, step: int, sim_min: int) -> None:
    """その日 present な来街者を party_size でグループ化し「同席の連れ」にする(既定 OFF=no-op)。

    起動時(day0)とローテーション(日境界)で present 実体化の**後**に呼ばれる。present な来街者を
    id 昇順に走査し、party_size>=2 の代表ごとに未割当の来街者を連れに束ね、同一初期ノードへ寄せ・
    相互 relations を注入・共有 POI(party_today)を据え・joint_activity{type:"party"} を1件記録する。
    presence 純関数には一切触れない(sim.agents の並べ替え・relations 注入のみ)=rotation 不変。"""
    cfg = getattr(sim, "partycfg", None)
    if not cfg or not cfg["enabled"]:
        return
    for a in sim.agents:                              # 前日分をクリア(日境界=当日で上書き)
        a.party_today = None
    day = sim_min // 1440
    visitors = sorted((a for a in sim.agents if a.visitor), key=lambda a: a.id)
    assigned: set = set()
    max_p = int(cfg["max_party"])
    tier = int(cfg["tier"])
    clo = float(cfg["closeness"])
    band = [int(cfg["band_start_min"]), int(cfg["band_end_min"])]
    n_v = len(visitors)
    # ★連れ探索の開始位置(第153 の性能修正)。`assigned` は増える一方なので「未割当が
    #   最初に現れる位置」は**単調に前へ進む**。従来は代表 1 人ごとに visitors を先頭から
    #   舐め直していた(O(V²)= 250k の py-spy で form_parties が step 時間の 10.4%・
    #   ホット行は :146 の `c.id in assigned`)。head より手前は**全員 assigned = どのみち
    #   continue で捨てられる要素**なので、そこを飛ばしても選ばれる連れ・その順序は同一。
    head = 0
    for leader in visitors:
        if leader.id in assigned:
            continue
        want = min(_party_size(leader), max_p)
        if want < 2:
            continue
        members = [leader]
        while head < n_v and visitors[head].id in assigned:
            head += 1
        for j in range(head, n_v):                    # 連れ = 次の未割当来街者(id 昇順=決定論)
            if len(members) >= want:
                break
            c = visitors[j]
            if c.id == leader.id or c.id in assigned:
                continue
            members.append(c)
        if len(members) < 2:                          # 連れが確保できなければ不成立
            continue
        node = leader.node
        # 寄せ先の座標。旅を畳んだ以上「ノードちょうどに立っている」のが正直な位置なので
        # node_xy を採る(day0 の実体化直後は leader.x/y と一致=従来と同値)。地図に無いノード
        # (テストが node だけ差し替える等)は代表の座標へフォールバック(medical と同じ保険)。
        try:
            lx, ly = sim.city.node_xy(node)
        except Exception:                             # noqa: BLE001(未知ノードの保険)
            lx, ly = leader.x, leader.y
        poi = _shared_poi(sim, leader.id, day)
        for m in members:                             # 同一初期ノードへ寄せる(同席の連れ)
            _fold_trip(m)                             # ★寄せる前に旅を畳む(存在しない辺を残さない)
            m.node = node
            m.x = lx
            m.y = ly
            m.party_today = {"poi": poi, "band": band}
            assigned.add(m.id)
        for i, m in enumerate(members):               # 連れ相互に relations 注入
            for o in members[i + 1:]:
                _inject(m, o, tier, clo)
                _inject(o, m, tier, clo)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(leader.id),
                             kind="joint_activity", x=leader.x, y=leader.y,
                             payload={"type": "party",
                                      "with": sorted(int(m.id) for m in members),
                                      "place": node, "tier": int(tier)}))
        sim._joint_total = getattr(sim, "_joint_total", 0) + 1


# ---------------------------------------------------------------- 回遊(自由行動の行き先)
def party_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """連れの回遊先=代表と共有の POI(回遊帯のみ)。決定論バイアス・非LLM。

    party OFF / 連れ予定なし / 帯外 / 抽選外 なら乱数を一切引かず None(既定不変)。専用 stream
    "party"(既存 draw 順を汚さない)で roam_bias 抽選。現在地と同じなら None(=到着済み)。"""
    cfg = getattr(sim, "partycfg", None)
    if not cfg or not cfg["enabled"]:
        return None
    pt = getattr(agent, "party_today", None)
    if not pt:
        return None
    lo, hi = pt["band"]
    m = sim_min % 1440
    if not (lo <= m < hi):
        return None
    poi = pt.get("poi")
    if not poi:
        return None
    rng = sim.hub.stream("party", agent.id, step)
    if rng.random() >= float(cfg["roam_bias"]):
        return None
    return poi if poi != agent.node else None
