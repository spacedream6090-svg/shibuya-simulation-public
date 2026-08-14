"""内部可動性 第60バッチ(b): 閉じた世界の構造固着を運営者介入なしに内生的に動かす転居機構。

devlog Entry 50 で確定した「エージェント側の状態・選択に由来する内生変化=火種介入とは別物」の
3機構のうち ①転居 と ②bond→同棲/unbond→転出、③career選択由来化の求職マッチをここに閉じる
(このファイルは src/society 直下=engine/cognition/actions/labeling/world の CHECKED_DIRS 外=
no-fingerprint 契約に触れない。household.py / freedom_p2.py / friends.py と同じ層)。

R1 呼数不変: いずれの機構も generate() を1本も足さず、k(信念書き戻し)・内面状態(構成概念)を
発火判断に食わせず、物理位置(home/work=k 非依存の観測量)・config・relations の closeness・
組織台帳・新 stream "housing" のみ参照する(呼数は k 非依存)。転居は夜間の home co-location を
変えうる(FixedLLM で ON!=OFF になりうる=career G5 / household H2 と同型)が、呼数不変は
compute_matched 下の k 不変性で担保する。

既定 OFF(housing.relocation.enabled=false / household.cohabit.enabled=false)= 新経路を一切
通さず relocate / move_in / move_out を1件も出さず "housing" stream も引かない(乱数消費不変=
ゴールデン golden_baseline_l1.json を守る)。住居の書換は home_building/home_node/home_floor の
3属性のみ(routine が毎回ライブ参照=移動先へ自然反映。友人グラフは起動時1回=再計算経路なし=
実査済み)。世帯は同一 home を共有(_share_home 整合)=転居/同棲は世帯単位で3属性を書換える。
新規状態(通勤既知値・同棲経過日・同棲フラグ)は agent 属性=checkpoint に自然同梱。
"""
from __future__ import annotations

import hashlib
import math

from .observer.schema import Event

# ---------------------------------------------------------------- 転居 config(housing.relocation)
RELOCATION_DEFAULTS = {
    "enabled": False,
    "job_prob": 0.5,               # 職場変更後・通勤閾値超の日次転居確率(職場近接へ)
    "rent_prob": 0.3,              # 家賃逼迫(E5 滞納)時の日次転居確率
    "commute_threshold_m": 1500.0, # この通勤距離(m)超で職場近接転居を検討
    "candidate_k": 5,              # 近接上位 K 件から (agent,day) 乱数で1件(過度な遠距離を避ける)
}

_R_BOOL = ("enabled",)
_R_INT = ("candidate_k",)
_R_FLOAT = ("job_prob", "rent_prob", "commute_threshold_m")


def build_relocation_cfg(raw) -> dict:
    """conf の housing.relocation ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    reloc = dict(raw.get("relocation", raw) or {})   # housing.relocation.* / 直下どちらも受ける
    cfg = dict(RELOCATION_DEFAULTS)
    for k, v in reloc.items():
        if k not in RELOCATION_DEFAULTS:
            continue
        if k in _R_BOOL:
            cfg[k] = bool(v)
        elif k in _R_INT:
            cfg[k] = int(v)
        else:
            cfg[k] = float(v)
    return cfg


def build_cohabit_cfg(raw) -> dict:
    """household.cohabit ブロックを正準化(既定 OFF=現行挙動と完全同一)。household.build_cfg から呼ぶ。

    closeness の既定 0 は「household.partner_closeness を採用」を意味する(_cohabit_threshold で解決)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "days": int(raw.get("days", 14)),          # bond 継続日数(同棲判定の最短)
        "closeness": float(raw.get("closeness", 0.0)),  # 維持すべき相互 closeness(0=partner_closeness)
    }


# ---------------------------------------------------------------- 純関数ヘルパ
def _stable_uniform(key: str) -> float:
    """(key) から一様値 [0,1)(hashlib=決定論・RngHub 無風=順位タイ破りに使う。friends と同方式)。"""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _node_xy(sim, node: str):
    if not node:
        return None
    try:
        return sim.city.node_xy(node)
    except Exception:
        return None


def _dist(a, b) -> float:
    if a is None or b is None:
        return 1.0e12
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _accounts_on(sim) -> bool:
    acc = (getattr(sim, "economy", None) or {}).get("accounts")
    return bool(acc and acc.get("enabled"))


def _household_members(sim, agent) -> list:
    """agent の世帯メンバ(自分を先頭に含む)。housemates を名簿から解決(来街者は除外)。"""
    members = [agent]
    for oid in getattr(agent, "housemates", None) or []:
        # ★レーン乙 ブロック3(c): 在場述語。街に居ない同居人へ home を書いても次の
        #   hydrate で捨てられる(= 転居が片側だけ成立して世帯の住居が割れる)。
        m = sim.present_agent(oid)
        if m is not None and not m.visitor and m.id != agent.id:
            members.append(m)
    return members


def _set_home(members, bld, floor: int) -> None:
    """世帯メンバ全員の home(建物/ノード/階)を bld へ寄せる(_share_home 整合=世帯単位で共有)。

    ★レーン乙 ブロック3: ここは「シミュレーション中に**実際に引っ越した**」唯一の経路なので、
      印(``_home_moved``)を残す。pool の退避はこの印が立っている個体だけ home を運ぶ
      (印が無い個体の home は名簿 + 世帯から決定論で組み直されるので運ぶ必要がなく、
      運ぶと二重の真実源になる)。印が無ければ属性も生えない=既定ランはバイト一致。"""
    node = bld["entrance"]
    levels = int(bld.get("levels", 1) or 1)
    fl = max(1, min(int(floor), levels))
    for m in members:
        m.home_building = bld["id"]
        m.home_node = node
        m.home_floor = fl
        m._home_moved = True


def _pick_building_near(sim, target_xy, exclude_bld: str, k: int, rng):
    """target_xy に近い residential 建物を近接上位 K 件から stream で1件(現在の建物は除外)。

    母集団=persona 生成が使う city.residential_buildings(住居割当ロジックと同一)。target_xy が
    無ければ id 昇順で後退。候補が無ければ None。決定論(近接ソート+ (agent,day) 乱数)。"""
    res = getattr(sim.city, "residential_buildings", None) or []
    cands = [b for b in res if b["id"] != exclude_bld]
    if not cands:
        return None
    if target_xy is not None:
        cands = sorted(cands, key=lambda b: (_dist(b.get("centroid"), target_xy), b["id"]))
    else:
        cands = sorted(cands, key=lambda b: b["id"])
    pool = cands[: max(1, int(k))]
    return pool[int(rng.integers(len(pool)))]


def _mutual_closeness(a, b) -> float:
    """相互 closeness の小さい方(household.form_partners と同じ両側閾値の流儀)。"""
    ra = a.mem.relations.get(b.id)
    rb = b.mem.relations.get(a.id)
    ca = float(ra.get("closeness", 0.0)) if ra else 0.0
    cb = float(rb.get("closeness", 0.0)) if rb else 0.0
    return min(ca, cb)


def _cohabit_threshold(sim, ccfg: dict) -> float:
    """同棲判定の closeness 閾値(cohabit.closeness>0 を優先、0 なら household.partner_closeness)。"""
    thr = float(ccfg.get("closeness", 0.0))
    if thr > 0.0:
        return thr
    hh = getattr(sim, "householdcfg", None) or {}
    return float(hh.get("partner_closeness", 15.0))


# ---------------------------------------------------------------- ① 転居(日次・stream "housing")
def relocate_day(sim, step: int, sim_min: int) -> None:
    """日次境界: 職場変更後の通勤逼迫 or 家賃逼迫(E5 滞納)から内生的に世帯単位で転居する。

    トリガーは (i) 職場変更(work_node の差分=switch_org/rehire/venture本業化/求職 いずれも)後、
    通勤距離が閾値超 → 日次 job_prob (ii) accounts ON かつ rent_due>0(滞納)→ 日次 rent_prob。
    行き先は職場(job)/現住地(rent)近接の residential 建物を上位 K から (agent,day) 乱数で決定論
    選択し、世帯全員の home 3属性を書換える(_share_home 整合)。外生イベントでの強制転居は作らない。
    立退き中(evicted)は住居を持たない=対象外。既定 OFF は即 return(stream も引かない=バイト一致)。"""
    cfg = getattr(sim, "housingcfg", None)
    if not (cfg and cfg["enabled"]):
        return
    day = sim_min // 1440
    accounts_on = _accounts_on(sim)
    thr_m = float(cfg["commute_threshold_m"])
    kK = int(cfg["candidate_k"])
    moved: set[int] = set()
    for agent in sim.agents:                              # id 昇順=決定論
        if agent.visitor or agent.id in moved or getattr(agent, "evicted", False):
            continue
        # 職場変更検知(初観測は基準化のみ=発火しない。以後 work_node が変わったら検討フラグ)
        seen = getattr(agent, "_reloc_seen_work", None)
        cur = getattr(agent, "work_node", "") or ""
        if seen is None:
            agent._reloc_seen_work = cur
        elif cur != seen:
            agent._reloc_seen_work = cur
            if cur:
                agent._reloc_consider = True
        reason = None
        target_xy = None
        rng = None
        # (i) 職場近接転居: 職場変更後、通勤距離が閾値超のとき日次確率で
        if getattr(agent, "_reloc_consider", False) and cur:
            wxy = _node_xy(sim, cur)
            hxy = _node_xy(sim, getattr(agent, "home_node", "") or "")
            if wxy is not None and hxy is not None and _dist(wxy, hxy) > thr_m:
                rng = sim.hub.stream("housing", agent.id, step)
                if rng.random() < float(cfg["job_prob"]):
                    reason, target_xy = "job", wxy
            else:                                        # 通勤が閾値以下=検討終了(移動済み等)
                agent._reloc_consider = False
        # (ii) 家賃逼迫転居(E5 滞納が現に発生している個体)
        if reason is None and accounts_on and float(getattr(agent, "rent_due", 0.0)) > 0.0:
            if rng is None:
                rng = sim.hub.stream("housing", agent.id, step)
            if rng.random() < float(cfg["rent_prob"]):
                reason = "rent"
                target_xy = _node_xy(sim, getattr(agent, "home_node", "") or "")
        if reason is None:
            continue
        # 行き先(同 rng から続けて引く=(agent,day) 内の逐次 draw)。世帯全員で移る。
        bld = _pick_building_near(sim, target_xy, agent.home_building, kK, rng)
        if bld is None:
            continue
        levels = int(bld.get("levels", 1) or 1)
        floor = 1 + int(rng.integers(max(1, levels)))
        members = _household_members(sim, agent)
        old = agent.home_building
        _set_home(members, bld, floor)
        for m in members:
            moved.add(m.id)
        if reason == "job":
            agent._reloc_consider = False                # 転居で通勤逼迫は解消
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="relocate", x=agent.x, y=agent.y,
                             payload={"from": old, "to": bld["id"], "reason": reason,
                                      "n": len(members)}))
        agent.remember("生活の都合で引っ越した")


# ---------------------------------------------------------------- ② bond→同棲(日次・stream "housing")
def cohabit_day(sim, step: int, sim_min: int) -> None:
    """日次境界: パートナー成立から N 日継続 + closeness 維持で同棲へ(世帯の物理再編。move_in)。

    id 昇順に走査し、独身でないペアを (lo=id 小, hi=id 大) の順で1回だけ見る。同棲開始日を初観測日
    から数え(_cohabit_since)、N 日以上経過し相互 closeness が閾値以上なら、hi(後から来た側)の
    世帯が lo 宅へ転居して1世帯に併合する(_share_home 整合)。bond/unbond の既存イベントは不変で、
    新イベント move_in のみを出す。既定 OFF は即 return(stream も引かない=バイト一致)。決定論・
    generate ゼロ増(closeness は relations=k 非依存の観測量、住居は物理位置)。"""
    hh = getattr(sim, "householdcfg", None)
    ccfg = (hh or {}).get("cohabit")
    if not (hh and hh.get("enabled") and ccfg and ccfg.get("enabled")):
        return
    day = sim_min // 1440
    ndays = int(ccfg["days"])
    thr = _cohabit_threshold(sim, ccfg)
    seen_pairs: set = set()
    for a in sim.agents:                                 # id 昇順=決定論
        if a.visitor or getattr(a, "evicted", False):
            continue
        pid = getattr(a, "partner_id", None)
        if pid is None:
            continue
        b = sim.agent_by_id.get(pid)
        if b is None or b.visitor or getattr(b, "evicted", False):
            continue
        lo, hi = (a, b) if a.id < b.id else (b, a)
        key = (lo.id, hi.id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if getattr(lo, "_cohabiting", False) or getattr(hi, "_cohabiting", False):
            continue                                     # 既に同棲中(あるいは他方と)
        since = getattr(lo, "_cohabit_since", None)
        if since is None:                                # 初観測=起算日を記録して翌日以降に判定
            lo._cohabit_since = hi._cohabit_since = day
            continue
        if day - int(since) < ndays:
            continue
        if _mutual_closeness(lo, hi) < thr:
            continue
        _cohabit_move_in(sim, lo, hi, step, sim_min)


def _cohabit_move_in(sim, keeper, mover, step: int, sim_min: int) -> None:
    """mover(後から来た側=id 大)の世帯を keeper 宅へ移し、1世帯に併合する(世帯単位・_share_home)。"""
    if not (keeper.home_building and sim.city.has_building(keeper.home_building)):
        return                                           # keeper に実在住宅が無い=同棲不成立(正直な縮退)
    kb = sim.city.building(keeper.home_building)
    old = mover.home_building
    mover_members = _household_members(sim, mover)
    _set_home(mover_members, kb, int(getattr(keeper, "home_floor", 1) or 1))
    _merge_households(sim, keeper, mover)
    keeper._cohabiting = mover._cohabiting = True
    mover._cohabit_mover = True                          # 後から来た側=別離時の転出候補
    keeper._cohabit_mover = False
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=keeper.id,
                         kind="move_in", x=keeper.x, y=keeper.y,
                         payload={"pair": [int(keeper.id), int(mover.id)],
                                  "from": old, "to": kb["id"], "reason": "cohabit"}))
    keeper.remember(f"{mover.name}と一緒に暮らし始めた")
    mover.remember(f"{keeper.name}の家へ引っ越して同棲を始めた")


def _merge_households(sim, keeper, mover) -> None:
    """keeper 側 + mover 側を1世帯へ併合(共通 household_id・housemates 更新・kind=family)。"""
    ids = {keeper.id, mover.id}
    members = [keeper, mover]
    for x in (keeper, mover):
        for oid in getattr(x, "housemates", None) or []:
            m = sim.present_agent(oid)     # ★レーン乙 ブロック3(c): 在場述語(幽霊へ書かない)
            if m is not None and m.id not in ids:
                ids.add(m.id)
                members.append(m)
    hh_id = keeper.household_id or f"cohab{keeper.id}_{mover.id}"
    for m in members:
        m.household_id = hh_id
        m.household_kind = "family"
        m.housemates = sorted(i for i in ids if i != m.id)


def on_breakup(sim, a, b, step: int, sim_min: int) -> None:
    """unbond(別離)フック: 同棲していた場合、後から来た側(mover)を新居へ転出させ世帯を分離する。

    household.unbond から遅延 import で呼ぶ(既存 bond/unbond のイベントは不変=move_out のみ新規)。
    cohabit OFF / 非同棲なら完全 no-op(stream も引かない)。新居は転居 picker を再利用(現住地近接)。"""
    hh = getattr(sim, "householdcfg", None)
    ccfg = (hh or {}).get("cohabit")
    if not (hh and hh.get("enabled") and ccfg and ccfg.get("enabled")):
        return
    mover = keeper = None
    for x, y in ((a, b), (b, a)):
        if x is not None and getattr(x, "_cohabiting", False) \
                and getattr(x, "_cohabit_mover", False):
            mover, keeper = x, y
            break
    if mover is None:
        return                                           # 同棲していなかった=何もしない
    kK = int((getattr(sim, "housingcfg", None) or {}).get("candidate_k", 5))
    rng = sim.hub.stream("housing", mover.id, step)
    txy = _node_xy(sim, getattr(mover, "home_node", "") or "")
    old = mover.home_building
    bld = _pick_building_near(sim, txy, mover.home_building, kK, rng)
    if bld is not None:
        levels = int(bld.get("levels", 1) or 1)
        floor = 1 + int(rng.integers(max(1, levels)))
        _set_home([mover], bld, floor)                   # mover 個人が転出(後から来た側)
        to_bld = bld["id"]
    else:
        to_bld = old                                     # 空き住宅が無い=形式上のみ分離(正直な縮退)
    _split_household(sim, mover)
    _clear_cohabit(mover, keeper)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=mover.id,
                         kind="move_out", x=mover.x, y=mover.y,
                         payload={"pair": [int(keeper.id) if keeper is not None else -1,
                                           int(mover.id)],
                                  "from": old, "to": to_bld, "reason": "breakup"}))
    mover.remember("別れて家を出た")


def _split_household(sim, mover) -> None:
    """mover を世帯から抜く(残りメンバの housemates から除去し、mover は単身に戻す)。

    正直な簡略化: 併合前に mover が連れていた同居人の再分割はしない(残留メンバ扱い)。同棲は主に
    独身パートナー2人=このケースは実務上ほぼ発生しない。"""
    for oid in list(getattr(mover, "housemates", None) or []):
        m = sim.present_agent(oid)         # ★レーン乙 ブロック3(c): 在場述語(幽霊へ書かない)
        if m is None:
            continue
        m.housemates = [i for i in (m.housemates or []) if i != mover.id]
        if not m.housemates:
            m.household_id = None
            m.household_kind = ""
            m._hh_detached = True
    mover.household_id = None
    mover.household_kind = ""
    mover.housemates = []
    # ★レーン乙 ブロック3: 「世帯を抜けた」という事実の印。pool の再入場で名簿由来の世帯へ
    #   自動で座り直されてしまう(分離が忘れられる)のを防ぐ唯一の手段。既定ランでは
    #   この関数自体が呼ばれない(cohabit OFF)ので属性は 1 つも生えない=バイト一致。
    mover._hh_detached = True


def leave_household(sim, agent) -> None:
    """**他レーンからの公開 API**: agent を世帯から抜く(残るメンバの housemates も更新)。

    存在の内生化 POP(``population.py``)の恒久転出が使う。転出は「街から出る」であって
    「別れて家を出る」(``on_breakup`` の move_out)ではないので、**新居を探さず・
    move_out も出さず**、世帯の紐帯だけを外す(``_split_household`` そのもの)。
    ★``_hh_detached`` の印が立つので、pool の再入場で名簿由来の世帯へ座り直されない。
    """
    _split_household(sim, agent)


def _clear_cohabit(mover, keeper) -> None:
    for x in (mover, keeper):
        if x is None:
            continue
        x._cohabiting = False
        x._cohabit_mover = False
        x._cohabit_since = None


# ---------------------------------------------------------------- ③ 求職マッチ(career.by_choice)
def match_job(sim, agent):
    """求職 tool 発火時の決定論マッチング: 定員に空きがある会社 org を1つ返す(空き無し=None)。

    定員近似=台帳 size.employees(正直な註記: 現従業員数は org_id 別の実カウント、上限は台帳の
    公称人員)。空き(現員 < 定員)org のうち、現職の wage_tier に整合する org を優先し、home_node からの
    workplace_poi 近接距離昇順、同点は安定ハッシュ(agent,org)で決定論タイブレーク。学校・現職は除外。
    乱数を1本も引かない(純台帳+安定ハッシュ)=switch_org/rehire の実体は既存機構を再利用する。"""
    book = getattr(sim, "orgs", None)
    if not book:
        return None
    counts: dict = {}
    for a in sim.agents:
        oid = getattr(a, "org_id", None)
        if oid:
            counts[str(oid)] = counts.get(str(oid), 0) + 1
    cur_org = getattr(agent, "org_id", None)
    cur_tier = ""
    if cur_org is not None and str(cur_org) in book:
        cur_tier = str(book[str(cur_org)].get("wage_tier", ""))
    hxy = _node_xy(sim, getattr(agent, "home_node", "") or "")
    cands = []
    for oid, org in book.items():
        if org.get("school_type"):                       # 学校は求職対象外(被雇用の職場のみ)
            continue
        if cur_org is not None and str(oid) == str(cur_org):
            continue                                     # 現職は除外(必ず別 org へ)
        cap = int((org.get("size") or {}).get("employees", 0) or 0)
        if cap and counts.get(str(oid), 0) >= cap:
            continue                                     # 定員満=空き無し(近似)
        tier_rank = 0 if (not cur_tier or str(org.get("wage_tier", "")) == cur_tier) else 1
        wp = org.get("workplace_poi") or {}
        wxy = (wp.get("x"), wp.get("y")) if wp.get("x") is not None else None
        dist = _dist(hxy, wxy)
        h = _stable_uniform(f"{int(agent.id)}\x1f{oid}")
        cands.append((tier_rank, dist, h, str(oid), org))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return cands[0][4]
