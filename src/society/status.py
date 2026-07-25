"""社会的ヒエラルキー(地位・信用・名声)= 合成地位スコアの背骨(第11バッチ 2026-07-08)。

ユーザー仮説: ヒエラルキーは (a) 組織・コミュニティを存続させ、(b) 世界を変える動機にもなる。
「信用がある人にリソースと人員が集まる」= マタイ効果/優先的選択(Barabási)/威信(prestige)。
本モジュールは調査文書 docs/research/hierarchy.md の採択候補(#2 地位の観測層 + #1 評判→動員 +
#3 優先的選択)を、決定論・非LLM・既定 OFF で実装する。

  合成地位スコア status(0..1): 各 agent の「客観カウントのみ」を材料に日次(暦日1回)で再計算する。
    - 評判       relations の _reputation(relations OFF なら 0)
    - フォロワー  internet の被フォロー数(net 無しなら 0)
    - 資産       money + account の母集団内順位
    - 制度実績   自分の提案の可決数(proposal_passed の author 集計)
    - 商い       venture の累計売上 sales_total
    - 主催       host_event の主催回数
  各材料を母集団内の百分位(rank/N。同値は id 昇順で安定=スケール不変・決定論)に直してから
  weights で混合する。乱数なし・LLM なし・k(思考頻度)非参照。★禁止(injection): プロンプトへ
  「あなたは地位が高い/低い」等を注入したり、高地位個体を変革行動へ直接誘導しない。ここは
  「地位が可視化・変換できる」affordance までに留める(hierarchy.md §3 の線引き)。

配置(R9 / no-fingerprint): status のロジックは src/society 直下=静的検査(CHECKED_DIRS)対象外に
  置く。status は efficacy/grievance 等の因子 state に結合しない(独立の観測スコア + 既存確率の変調
  のみ)。engine(scheduler)からは因子名を見せない。

決定論・R1: 材料はすべて既に確定した客観カウントの集計。乱数を一切引かず、LLM を1本も足さない。
  動員フック(tools/net)は既存の確率(rng.random() の閾値)を変調するだけ=新しい draw・stream を
  足さない。優先的選択(feed)は既存の候補選択を露出重みで並べ替えるだけ=乱数不使用。

既定 OFF(hierarchy.enabled=false): scheduler の _phase_status が即 return、動員フックは加算 0/倍率 1、
  feed は従来 TL、L3 に status キーを足さず、L2 集計は None=列なし。= 関係台帳・プロンプト・イベント列・
  スナップショット・乱数消費がバイト一致で不変(ゴールデン golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

# 合成地位スコアの既定(economy.build_career_cfg と同型: dict/OmegaConf 両対応)。
DEFAULTS = {
    "enabled": False,          # ★ON 推奨(本番): true
    "attract_gain": 0.15,      # イベント参加/グループ加入の確率に + gain × 主催者(創設者)status
    "feed_gain": 0.3,          # フィード露出重みに ×(1 + gain × 投稿者 status)
    "buy_gain": 0.2,           # 屋台購買確率に ×(1 + gain × 店主 status)
    # 合成重み(材料の混合。母集団内百分位に掛ける相対重み。合計は内部で正規化=スケール不変)。
    "weights": {
        "rep": 0.25,           # 評判(prestige: 語採用・被傾聴で伝播する社会的名声)
        "followers": 0.25,     # 被フォロー数(注意経済・優先的選択のハブ性)
        "wealth": 0.20,        # 資産(階級: money + account)
        "inst": 0.10,          # 制度実績(党派: 提案の可決数)
        "biz": 0.10,           # 商い(累計売上)
        "host": 0.10,          # 主催(イベント主催回数)
    },
}

_WEIGHT_KEYS = ("rep", "followers", "wealth", "inst", "biz", "host")


def build_status_cfg(raw) -> dict:
    """conf の hierarchy ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dict / OmegaConf 両対応。dotlist 上書きは文字列で入り得るため型強制する
    (economy.build_career_cfg / relations.build_cfg と同じ作法)。"""
    try:                                       # OmegaConf は plain dict へ落として扱う
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    raw = dict(raw or {})
    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        "attract_gain": float(raw.get("attract_gain", DEFAULTS["attract_gain"])),
        "feed_gain": float(raw.get("feed_gain", DEFAULTS["feed_gain"])),
        "buy_gain": float(raw.get("buy_gain", DEFAULTS["buy_gain"])),
        "weights": dict(DEFAULTS["weights"]),
    }
    w = dict(raw.get("weights", {}) or {})
    for k, v in w.items():
        if k in cfg["weights"]:
            cfg["weights"][k] = float(v)
    return cfg


# ---------------------------------------------------------------- ゲート / cfg
def cfg_of(sim) -> dict:
    """hierarchy 設定を返す(初回のみ sim.cfg から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本モジュールが遅延構築する(_selfmodel_days と同型)。
    キャッシュ属性 sim.statuscfg の設定は L1/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "statuscfg", None)
    if c is None:
        raw = None
        try:
            raw = sim.cfg.get("hierarchy", None)
        except Exception:
            raw = None
        c = build_status_cfg(raw)
        sim.statuscfg = c
    return c


def enabled(sim) -> bool:
    """社会的ヒエラルキー(地位機構)が有効か。既定 OFF=新経路を一切通さない。"""
    return bool(cfg_of(sim)["enabled"])


# ---------------------------------------------------------------- 百分位・順位(純関数)
def _percentiles(values: dict, ids: list) -> dict:
    """母集団内の百分位(rank/N。同値は id 昇順で安定)。返り値は id -> (0,1] の percentile。

    値の昇順(同値は id 昇順)に並べ、1-indexed の順位 / N を割り当てる=スケール不変・決定論。"""
    order = sorted(ids, key=lambda i: (values[i], i))
    n = len(order) or 1
    return {i: (pos + 1) / n for pos, i in enumerate(order)}


def _ranks_by_status(agents) -> dict:
    """status 降順の整数順位(1=最上位。同値は id 昇順で安定)。移動性(rank mobility)用。"""
    order = sorted(agents, key=lambda a: (-float(getattr(a, "status", 0.0)), a.id))
    return {a.id: rank for rank, a in enumerate(order, start=1)}


# ---------------------------------------------------------------- 日次: 合成地位の再計算
def _material_counts(sim):
    """tools から制度実績(可決数)・商い(売上)・主催(回数)を1パスで集計する(決定論)。"""
    passed_by: dict = {}
    host_by: dict = {}
    sales_by: dict = {}
    tools = getattr(sim, "tools", None)
    if tools is not None:
        for pr in getattr(tools, "proposals", {}).values():
            if pr.get("passed"):
                a = pr.get("author")
                passed_by[a] = passed_by.get(a, 0) + 1
        for ev in getattr(tools, "events", {}).values():
            h = ev.get("host")
            host_by[h] = host_by.get(h, 0) + 1
        for v in getattr(tools, "ventures", {}).values():
            o = v.get("owner")
            sales_by[o] = sales_by.get(o, 0.0) + float(v.get("sales_total", 0.0))
    return passed_by, host_by, sales_by


def _recompute(sim, cfg: dict) -> None:
    """全 agent の status(0..1)を客観カウントの母集団内百分位の重み混合で決定論算出する。"""
    agents = sim.agents
    if not agents:
        return
    net = getattr(sim, "net", None)
    passed_by, host_by, sales_by = _material_counts(sim)
    ids = [a.id for a in agents]
    rep = {a.id: float(getattr(a, "_reputation", 0.0)) for a in agents}
    fol = {a.id: (float(net.follower_count(a.id)) if net is not None else 0.0)
           for a in agents}
    wealth = {a.id: float(getattr(a, "money", 0.0)) + float(getattr(a, "account", 0.0))
              for a in agents}
    inst = {a.id: float(passed_by.get(a.id, 0)) for a in agents}
    biz = {a.id: float(sales_by.get(a.id, 0.0)) for a in agents}
    host = {a.id: float(host_by.get(a.id, 0)) for a in agents}

    pr_rep = _percentiles(rep, ids)
    pr_fol = _percentiles(fol, ids)
    pr_wealth = _percentiles(wealth, ids)
    pr_inst = _percentiles(inst, ids)
    pr_biz = _percentiles(biz, ids)
    pr_host = _percentiles(host, ids)

    w = cfg["weights"]
    wsum = sum(w.values()) or 1.0
    for a in agents:
        i = a.id
        s = (w["rep"] * pr_rep[i] + w["followers"] * pr_fol[i]
             + w["wealth"] * pr_wealth[i] + w["inst"] * pr_inst[i]
             + w["biz"] * pr_biz[i] + w["host"] * pr_host[i]) / wsum
        a.status = round(s, 6)

    # 負の評判(第61バッチ c): 悪評の被知覚数(/N)を status の負項として小さく反映する(max_penalty で
    # 上限=孤立への滑り台にしない)。gossip OFF は penalty=0=status 不変(gossip.py に閉じる=no-fingerprint)。
    from . import gossip as _gossip
    if _gossip.enabled(sim):
        perceived = _gossip.perceived_counts(sim)
        for a in agents:
            pen = _gossip.status_penalty(sim, a.id, perceived)
            if pen:
                a.status = round(max(0.0, a.status - pen), 6)

    # 移動性(前日 rank との平均絶対差)を sim に保持(L2 aggregator が参照)。
    ranks = _ranks_by_status(agents)
    prev = getattr(sim, "_status_prev_rank", None)
    if prev:
        common = [i for i in ranks if i in prev]
        sim._status_rank_mobility = (
            sum(abs(ranks[i] - prev[i]) for i in common) / len(common)
            if common else 0.0)
    else:
        sim._status_rank_mobility = 0.0
    sim._status_prev_rank = ranks


def material_breakdown(sim, cfg: dict | None = None) -> dict:
    """合成地位 status の材料別内訳(各材料の母集団内百分位 + weights 適用後の寄与)を返す純関数。

    第50バッチ T6(信用内訳レンズ)の正準実装。_recompute と同じ客観カウント・同じ百分位規則で
    決定論再現する(乱数なし・LLM なし・挙動不変=読むだけ)。status.enabled を問わない(呼び手が判断)。

    返り値:
      {"materials": (…6材料…),
       "weights": {mat: 正規化後の重み},
       "agents": [{"id", "status", "raw":{mat:実値}, "pct":{mat:0-1}, "contrib":{mat:0-1寄与}}…]}
      contrib は sum が status に一致する(weights を正規化し百分位に掛けた分解)。"""
    if cfg is None:
        cfg = cfg_of(sim)
    agents = sim.agents
    materials = _WEIGHT_KEYS
    w = cfg["weights"]
    wsum = sum(w.values()) or 1.0
    wn = {k: w[k] / wsum for k in materials}
    if not agents:
        return {"materials": materials, "weights": wn, "agents": []}
    net = getattr(sim, "net", None)
    passed_by, host_by, sales_by = _material_counts(sim)
    ids = [a.id for a in agents]
    raw = {
        "rep": {a.id: float(getattr(a, "_reputation", 0.0)) for a in agents},
        "followers": {a.id: (float(net.follower_count(a.id)) if net is not None else 0.0)
                      for a in agents},
        "wealth": {a.id: float(getattr(a, "money", 0.0)) + float(getattr(a, "account", 0.0))
                   for a in agents},
        "inst": {a.id: float(passed_by.get(a.id, 0)) for a in agents},
        "biz": {a.id: float(sales_by.get(a.id, 0.0)) for a in agents},
        "host": {a.id: float(host_by.get(a.id, 0)) for a in agents},
    }
    pct = {m: _percentiles(raw[m], ids) for m in materials}
    out = []
    for a in agents:
        i = a.id
        # status は未丸めの和を丸める(=_recompute の round(sum(wn*pct),6) と一字一致)。
        # contrib は表示用に各項を丸める(和は status と丸め誤差内で一致=表示の分解)。
        status_i = round(sum(wn[m] * pct[m][i] for m in materials), 6)
        contrib = {m: round(wn[m] * pct[m][i], 6) for m in materials}
        out.append({
            "id": i,
            "status": status_i,
            "raw": {m: round(raw[m][i], 4) for m in materials},
            "pct": {m: round(pct[m][i], 6) for m in materials},
            "contrib": contrib,
        })
    return {"materials": materials, "weights": {k: round(v, 6) for k, v in wn.items()},
            "agents": out}


def phase_day(sim, step: int, sim_min: int) -> None:
    """日次(暦日1回)に全 agent の status を再計算する。OFF なら完全 no-op(バイト一致)。

    決定論(乱数なし・LLM なし・イベント 0 件)。暦日境界のゲートは本関数が自前で持つ
    (_phase_relations_day と同型。simulation.py を触らず sim._status_day を遅延管理)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_status_day", -1):
        return
    sim._status_day = day
    _recompute(sim, cfg)


# ---------------------------------------------------------------- 動員・引力フック(既存確率の変調)
def attract_bonus(sim, host_id) -> float:
    """イベント参加/グループ加入の確率への地位加算 = attract_gain × 主催者(創設者)status。

    OFF or 主催者不明なら 0.0(= 従来の確率と完全同一)。新しい draw を足さない(閾値だけ変える)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 0.0
    who = sim.agent_by_id.get(host_id)
    s = float(getattr(who, "status", 0.0)) if who is not None else 0.0
    return cfg["attract_gain"] * s


def buy_multiplier(sim, owner_id) -> float:
    """屋台購買確率の倍率 = 1 + buy_gain × 店主 status。OFF or 店主不明なら 1.0(従来と完全同一)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return 1.0
    who = sim.agent_by_id.get(owner_id)
    s = float(getattr(who, "status", 0.0)) if who is not None else 0.0
    return 1.0 + cfg["buy_gain"] * s


def feed_exposure_weight(sim, author_id) -> float:
    """フィード露出重み = (1 + feed_gain × 投稿者 status) ×(1 + 被フォロー数)。Barabási 優先的選択。

    被フォロー数に単調増加(優先的選択)・status に単調増加(威信)。決定論・乱数不使用。
    hierarchy OFF ではそもそも呼ばれない(infoenv.timeline が従来 TL に後退)。"""
    cfg = cfg_of(sim)
    net = getattr(sim, "net", None)
    who = sim.agent_by_id.get(author_id)
    s = float(getattr(who, "status", 0.0)) if who is not None else 0.0
    fol = float(net.follower_count(author_id)) if net is not None else 0.0
    return (1.0 + cfg["feed_gain"] * s) * (1.0 + fol)
