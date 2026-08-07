"""街路の環境情報(第18バッチ 2026-07-11。ユーザー要望): 街頭広告(OOH)+群衆の視覚情報。

現実の渋谷は無数の広告・視覚・聴覚情報に満ちるが、シミュの街路は「店名と人」しか
見えない。本モジュールは「シミュレーションに大きな変化を与え、メリット>リソースの
情報のみ」というユーザー基準で採択された2チャネルを実装する(音環境は見送り):

  1. 街頭広告(OOH): 掲出地点(大型ビジョン/看板=POI 名で指定)の近くを通った
     エージェントに視認判定を行い、ad_exposure を L1 に記録+発火時のプロンプトに
     「広告を見かけた」1行を注入する。広告主はシミュ内の実在 POI(=接触→来店→購買の
     ファネルが世界内で閉じ、商業レポートの広告枠価値算定に直結する)。
     効果量は docs/research/commercial-analytics.md §7 の現実バンドに従う:
     視認率 大型0.3-0.6/通常0.1-0.3、有効フリークエンシー≈3回、adstock は
     想起窓(recall_steps)で近似。**行動への転換は決定論で押し込まない**
     (中立提示のみ=接触の大半が無効果という現実の再現は LLM の自然な無視に委ねる)。
  2. 群衆の視覚情報: 同じ知覚文脈にいる実在エージェントの人数・年齢構成を
     決定論で1行に要約して注入する(記述的規範=「みんながしていること」への同調・
     流行伝播の経路)。集計のみ=捏造なし・乱数なし・イベントなし(L1 位置から再構成可)。

★ 配置: src/society 直下(engine/cognition/world の静的検査対象外)。
決定論: 乱数は新 stream "ads" のみ(視認判定=agent×step×slot キー・キャンペーン改定=
  period×slot キーのステートレス draw)。OFF は draw ゼロ・イベントゼロ・プロンプト不変
  (ゴールデン tests/data/golden_baseline_l1.json を守る)。crowd は乱数を一切引かない。
R1 呼数不変: どちらも generate() を1本も足さない(発火者のプロンプト内容のみ変わる=
  weather/disaster_line と同型)。視認判定は物理位置と config のみ参照=k 非依存。
"""
from __future__ import annotations

from .observer.schema import Event

ADS_DEFAULTS = {
    "enabled": False,
    # 掲出地点(POI 名で指定。完全一致→部分一致で解決)。基盤 config は空=機構のみ
    # (EnvPack 原則: 機構=基盤・場所の値=環境プロファイル側)。
    "large": [],                 # 大型ビジョン(視認率 p_see_large)
    "slots": [],                 # 通常看板(視認率 p_see_normal)
    "radius_m": 60.0,            # 掲出地点からの視認距離(大型ビジョンは知覚40mより届く)
    "p_see_large": 0.45,         # 較正: 大型 OOH 視認率 0.3〜0.6(commercial-analytics §7)
    "p_see_normal": 0.20,        # 較正: 通常看板 0.1〜0.3
    "cooldown_steps": 6,         # 同一枠の再接触の最短間隔(6step=1時間。滞在の重複記録防止)
    "daily_cap": 4,              # 1人1日の接触記録上限(枠横断。反復記録の暴走防止)
    "recall_steps": 144,         # プロンプト注入の想起窓(144step=1日。adstock λ≈0.7-0.9/日の近似)
    "eff_frequency": 3,          # 有効フリークエンシー(累計これ以上で「何度も見かけた」表現)
    "campaign_days": 7,          # キャンペーン改定周期(日)。枠ごとに広告主 POI を決定論再抽選
    "target_cats": ["food", "cafe", "shop", "leisure", "nightlife"],  # 広告主になれる POI カテゴリ
}

CROWD_DEFAULTS = {
    "enabled": False,
}


def build_ads_cfg(raw: dict | None) -> dict:
    """conf の ads ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    cfg = dict(ADS_DEFAULTS)
    cfg.update({k: raw[k] for k in raw if k in ADS_DEFAULTS})
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["large"] = [str(s) for s in (cfg["large"] or [])]
    cfg["slots"] = [str(s) for s in (cfg["slots"] or [])]
    cfg["target_cats"] = [str(c) for c in (cfg["target_cats"] or [])]
    for k in ("radius_m", "p_see_large", "p_see_normal"):
        cfg[k] = float(cfg[k])
    for k in ("cooldown_steps", "daily_cap", "recall_steps", "eff_frequency",
              "campaign_days"):
        cfg[k] = int(cfg[k])
    return cfg


def build_crowd_cfg(raw: dict | None) -> dict:
    raw = dict(raw or {})
    return {"enabled": bool(raw.get("enabled", CROWD_DEFAULTS["enabled"]))}


# ------------------------------------------------------------------ 広告: 掲出枠
def _name_index(items, key_of) -> dict:
    """名前→座標キーの決定論索引(名前昇順・先勝ち)。"""
    idx: dict = {}
    for it in items:
        nm, val = key_of(it)
        if nm and nm not in idx:
            idx[nm] = val
    return idx


def _lookup(idx: dict, name: str):
    """完全一致→部分一致(ソート順=決定論)。無ければ None。"""
    if name in idx:
        return idx[name]
    for nm in sorted(idx):
        if name in nm or nm in name:
            return idx[nm]
    return None


def resolve_slots(city, cfg: dict) -> list[dict]:
    """掲出地点名 → 座標の決定論解決。POI → 建物(centroid)→ 名前付きノード の順に
    フォールバック(現実の掲出場所は POI とは限らない: 大型ビジョン=建物・駅前広場=ノード)。

    各名前空間内は 完全一致→部分一致(ソート順=決定論)。解決できない名前は黙って落とす
    (環境プロファイル側の設定ミスは接触0件として現れる)。返り値は名前昇順の
    [{"name", "x", "y", "large"}]。
    """
    poi_idx = _name_index(
        sorted(city.poi_list, key=lambda q: str(q.get("name") or "")),
        lambda p: (str(p.get("name") or "").strip(), p["node"]))
    bld_idx = _name_index(
        sorted(getattr(city, "buildings", []) or [],
               key=lambda b: str(b.get("name") or "")),
        lambda b: (str(b.get("name") or "").strip(), b.get("centroid")))
    node_idx = {}
    graph = getattr(city, "graph", None)
    if graph is not None:
        for nid in sorted(graph.nodes):
            nm = str(graph.nodes[nid].get("name") or "").strip()
            if nm and nm not in node_idx:
                node_idx[nm] = nid

    def _find_xy(name: str):
        node = _lookup(poi_idx, name)
        if node is not None:
            return city.node_xy(node)
        cxy = _lookup(bld_idx, name)
        if cxy is not None:
            return cxy
        nid = _lookup(node_idx, name)
        if nid is not None:
            return city.node_xy(nid)
        return None

    slots = []
    for name, large in ([(s, True) for s in cfg["large"]]
                        + [(s, False) for s in cfg["slots"]]):
        xy = _find_xy(name)
        if xy is None:
            continue
        slots.append({"name": name, "x": float(xy[0]), "y": float(xy[1]),
                      "large": large})
    slots.sort(key=lambda s: s["name"])
    return slots


def pick_campaigns(sim, cfg: dict, slots: list[dict], period: int) -> dict:
    """period(campaign_days 刻み)ごとの広告主 POI を枠ごとに決定論抽選する。

    広告主=target_cats の実在 POI(名前あり)。乱数は stream("ads","campaign",period,枠名)
    から枠ごとに1draw のみ(ステートレス=順序非依存)。候補ゼロなら枠は休眠(接触なし)。
    """
    eligible = sorted({str(p.get("name") or "").strip()
                       for p in sim.city.poi_list
                       if p.get("cat") in cfg["target_cats"] and p.get("name")})
    campaigns: dict[str, dict] = {}
    cat_of = {}
    for p in sim.city.poi_list:
        nm = str(p.get("name") or "").strip()
        if nm and nm not in cat_of:
            cat_of[nm] = str(p.get("cat") or "")
    for slot in slots:
        pool = [nm for nm in eligible if nm != slot["name"]]
        if not pool:
            continue
        rng = sim.hub.stream("ads", "campaign", period, slot["name"])
        target = pool[int(rng.integers(len(pool)))]
        campaigns[slot["name"]] = {
            "campaign": f"{slot['name']}#p{period}",
            "target": target,
            "cat": cat_of.get(target, ""),
        }
    return campaigns


# ------------------------------------------------------------------ 広告: 接触
def phase(sim, step: int, sim_min: int) -> None:
    """毎step: 掲出地点の近くにいる路上エージェントの視認判定(scheduler が位置確定後に呼ぶ)。

    OFF は即 return(draw・イベント・状態ともゼロ=バイト一致)。判定は id 昇順×枠名順で
    決定論。屋内・睡眠中・街の外は見ない(OOH=街路の広告)。
    """
    cfg = getattr(sim, "adscfg", None)
    if not cfg or not cfg["enabled"]:
        return
    slots = getattr(sim, "_ads_slots", None)
    if slots is None:
        slots = resolve_slots(sim.city, cfg)
        sim._ads_slots = slots
        # 第98バッチ 小粒A: 枠(slots)は city からの決定論導出なので毎ラン組み直すが、
        # **キャンペーン期の進行は日を跨ぐ状態**で checkpoint が運ぶ。ここで無条件に -1 へ
        # 潰すと resume 直後に同じ期を引き直して ad_campaign を二重記録するので、既存値が
        # あれば残す(fresh ランでは属性が生えていない → -1 / {} = 従来と完全同一)。
        sim._ads_period = getattr(sim, "_ads_period", -1)
        sim._ads_campaigns = getattr(sim, "_ads_campaigns", None) or {}
    if not slots:
        return
    day = sim_min // 1440
    period = day // max(1, cfg["campaign_days"])
    if period != sim._ads_period:                    # キャンペーン改定(枠ごとに1draw)
        sim._ads_period = period
        sim._ads_campaigns = pick_campaigns(sim, cfg, slots, period)
        for slot in slots:
            c = sim._ads_campaigns.get(slot["name"])
            if c:
                sim.logger.log(Event(
                    step=step, sim_min=sim_min, agent_id=-1, kind="ad_campaign",
                    x=slot["x"], y=slot["y"],
                    payload={"slot": slot["name"], "campaign": c["campaign"],
                             "target": c["target"], "cat": c["cat"],
                             "period": period}))
    r2 = cfg["radius_m"] * cfg["radius_m"]
    for agent in sim.agents:                         # id 昇順=決定論
        if agent.loc == "outside" or agent.sleeping or agent.building:
            continue
        if getattr(agent, "_ad_day", None) != day:   # 日次の接触上限カウンタ
            agent._ad_day = day
            agent._ad_today = 0
        if agent._ad_today >= cfg["daily_cap"]:
            continue
        for slot in slots:
            c = sim._ads_campaigns.get(slot["name"])
            if c is None:
                continue
            dx = agent.x - slot["x"]
            dy = agent.y - slot["y"]
            if dx * dx + dy * dy > r2:
                continue
            last = getattr(agent, "_ad_last", None) or {}
            if step - last.get(slot["name"], -10**9) < cfg["cooldown_steps"]:
                continue
            p = cfg["p_see_large"] if slot["large"] else cfg["p_see_normal"]
            rng = sim.hub.stream("ads", agent.id, step, slot["name"])
            if float(rng.random()) >= p:
                continue
            last[slot["name"]] = step
            agent._ad_last = last
            seen = getattr(agent, "ad_seen", None)
            if seen is None:
                seen = {}
                agent.ad_seen = seen
            seen[c["campaign"]] = seen.get(c["campaign"], 0) + 1
            agent.ad_recent = (step, c["campaign"], c["target"])
            agent._ad_today += 1
            sim.logger.log(Event(
                step=step, sim_min=sim_min, agent_id=agent.id,
                kind="ad_exposure", x=agent.x, y=agent.y,
                payload={"campaign": c["campaign"], "target": c["target"],
                         "slot": slot["name"], "cat": c["cat"],
                         "n_seen": seen[c["campaign"]]}))
            if agent._ad_today >= cfg["daily_cap"]:
                break


def ads_line(agent, cfg: dict | None, step: int) -> str | None:
    """発火時のプロンプト1行(中立提示・行動誘導の語は書かない)。

    直近接触が想起窓(recall_steps)内のときだけ。有効フリークエンシー(累計3回〜)で
    「何度も」の一語だけ変える(反復接触の心理的現実)。OFF/接触なしは None=不変。
    """
    if not cfg or not cfg["enabled"]:
        return None
    recent = getattr(agent, "ad_recent", None)
    if not recent:
        return None
    seen_step, campaign, target = recent
    if step - seen_step > cfg["recall_steps"]:
        return None
    n = (getattr(agent, "ad_seen", None) or {}).get(campaign, 1)
    rep = "何度も" if n >= cfg["eff_frequency"] else ""
    return f"街頭の広告: 「{target}」の広告を{rep}見かけた。"


# ------------------------------------------------------------------ 群衆の視覚情報
_BANDS = ((30, "若い人"), (50, "働き盛りの人"), (200, "年配の人"))


def crowd_line(company, cfg: dict | None) -> str | None:
    """同じ知覚文脈にいる実在エージェントから群衆の見た目を決定論要約(記述的規範)。

    人数の質感+年齢構成(過半数の帯があれば「〜が目立つ」)。集計のみ=捏造なし・
    乱数なし。OFF は None=1行も足さない=バイト一致。
    """
    if not cfg or not cfg["enabled"]:
        return None
    n = len(company)
    if n == 0:
        return "まわりの様子: あたりに人影は少ない。"
    size = "賑わっている" if n >= 8 else "人通りがある" if n >= 3 else "人はまばらだ"
    counts = [0, 0, 0]
    for a in company:
        age = int(getattr(a, "age", 35))
        for i, (hi, _) in enumerate(_BANDS):
            if age < hi:
                counts[i] += 1
                break
    desc = ""
    for i, (_, label) in enumerate(_BANDS):
        if counts[i] * 2 >= n and counts[i] > 0:
            desc = f"。{label}が目立つ"
            break
    return f"まわりの様子: {size}{desc}。"
