"""共同行動エンジン(関係性の再現 第44バッチ S-R3。ユーザー要望 2026-07-21)。

関係(友人・同居人)を起点に「N人が同一(POI,時刻)へ独立に収束+合流は決定論」で共同行動
(外食・カフェ/映画/カラオケ/買い物)を発生させる層。先行例(Generative Agents の創発、
AgentSociety の gravity model)の知見どおり、誘い→承諾→合流は**全決定論**で LLM 呼を増やさない。
既存 household.date_dest(ペア決定論収束)の一般化=任意の関係グループへ拡張したもの。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  活動カタログの較正値・活動名リテラルはここ(と conf)にのみ書く=no-fingerprint 契約に触れない
  (engine/cognition 側の配線は date_dest 分岐と同型の中立な1分岐のみ)。

R1 呼数不変: generate() を1本も足さない。日次編成(_phase_joint=日境界。新 stream "joint"
  のみ)・同伴者選択(relations closeness/housemates=k 非依存の観測量から決定論)・誘い→承諾
  (新 stream "joint")・ランデブー POI(グループ最小 id/day/活動タイプの純関数=乱数ゼロ)。
  合流(同一 POI 収束)は物理位置=対面 co-location を変えうる(FixedLLM で ON!=OFF になりうる=
  career/health/household と同型で許容)が、機構は k・内面状態(構成概念)を発火判断に食わせず、
  名簿・config・relations の closeness・物理位置のみ参照する=compute_matched 下の k 不変性
  (k=free==k=off の呼数一致)で担保する。

既定 OFF(enabled=false)= 何も編成せず(joint_today も付かず)・"joint" stream を引かず・
  joint_activity 0 件=イベント/乱数消費とも不変(ゴールデン golden_baseline_l1.json を守る)。

第62バッチ(2026-07-27): 承諾判断の内生化(relations.endogenous_accept。既定 OFF)。ON 時のみ
  承諾抽選を「較正確率を事前分布に残した内生判定の合成」に置き換え(relations_endo.decide_accept
  =全決定論・LLM呼ゼロ)、joint_invite を記録する。抽選 draw は always-draw(veto 時も draw して
  結果を破棄)= "joint" stream の decision 単位の消費はON/OFF不変。OFF はこの経路に一切入らない。

第64バッチ(2026-07-27): 誘う相手の内生選抜 フェーズ3(relations.endogenous_invite。既定 OFF。
  **実装のみ=実験投入はフェーズ2 phase3_go ゲート**)。ON 時のみ _companions の friend 経路の
  候補集合を並べ替え・拡張する(枠組み・min/max_group・daily_rate 不変): 前日計画 with(従来=
  最優先)→前日発話の明示キュー→closeness 降順(較正事前分布の維持)→弱い紐帯探索枠(tier=1・
  安定ハッシュ=乱数ゼロ・末尾)。joint_invite に source を付し L2 に invite_* 2列を出す。
  OFF は候補順・"joint" stream 消費・イベント・L2 とも従来と完全同一(relations_endo 冒頭参照)。
"""
from __future__ import annotations

from . import dunbar as _dunbar
from . import gossip as _gossip
from . import relations as _relations
from . import relations_endo as _endo
from .observer.schema import Event

# ---- 活動カタログ(渋谷較正=§3.3 参加率。友人系4種)。POI カテゴリは routine._PLAN_CAT の語彙
#      (food/leisure/shop)と整合。band は時間帯(下の bands の名前)。activity_tag は routine の
#      活動タグ(cost 用: eating で既存 _charge_meal が飲食費を課金。他は "" で追加課金なし)。----
_DEFAULT_ACTIVITIES = {
    "meal_cafe": {"weight": 0.39, "poi_cat": "food", "band": "night",
                  "activity_tag": "eating"},   # 外食・カフェ=参加率最高(§2.6 外食39.2%)
    "cinema": {"weight": 0.34, "poi_cat": "leisure", "band": "day",
               "activity_tag": ""},            # 映画(§2.6 映画館33.7%)
    "karaoke": {"weight": 0.20, "poi_cat": "leisure", "band": "night",
                "activity_tag": ""},           # カラオケ(§2.6 20.2%・集団性高)
    "shopping": {"weight": 0.15, "poi_cat": "shop", "band": "day",
                 "activity_tag": ""},          # ショッピング(§2.6 9位・渋谷若年で高頻度)
}
# 時間帯(分 of day の [開始, 終了))。day=昼〜夕方の外出、night=夕〜夜の外食/カラオケ。
_DEFAULT_BANDS = {"day": [660, 1020], "night": [1020, 1320]}  # 11:00-17:00 / 17:00-22:00
# 年齢条件付き重み倍率(ontology の info_behavior bands と同型=キーは年代下限の文字列)。
# 渋谷=若年層が友人遊びの主役(§2.7 女友達54%)=若年でカラオケ/買い物/映画を高く較正する。
_DEFAULT_AGE_WEIGHTS = {
    "0":  {"karaoke": 1.6, "shopping": 1.4, "cinema": 1.2, "meal_cafe": 1.0},
    "35": {"karaoke": 0.8, "shopping": 0.9, "cinema": 1.0, "meal_cafe": 1.0},
    "55": {"karaoke": 0.4, "shopping": 0.7, "cinema": 0.8, "meal_cafe": 1.0},
}

DEFAULTS = {
    "enabled": False,
    # 1人1日あたりの共同外出の発生確率。§2.3(学生時代友人と会うのは月1回が最頻=~0.03/日/特定相手)と
    # 桁が合う控えめな既定。複数の友人サークルにまたがる「何らかの共同外出」の集約率として ~週1(0.15/日)。
    "daily_rate": 0.15,
    "accept_base": 0.5,       # 誘いの基礎承諾確率(host_event の attend_base 相当)
    "tier_bonus": 0.3,        # 親友(tier 3)への承諾加算(関係の強さで参加率が上がる)
    "min_group": 2,           # 成立に必要な最小人数(自分含む)
    "max_group": 4,           # グループ最大人数(自分含む。2-4人)
    "friend_tier": 2,         # 同伴候補の最小 relations tier(2=友人)
    # ---- 職場の会食・飲み会の階層依存(S-R4。§2.5: 飲みニケーション忌避は"上司同席・義務的"に集中。
    #      SHIBUYA109 lab.: 上司ありの飲み会「苦手」66.7% だが同世代なら「好き」50.8%)。組織台帳に
    #      役職(rank/seniority/founder)データは無い(org_role は職種名=デザイナー/エンジニア等で階層で
    #      ない)ため、上司/同世代の代理指標は **年齢差>boss_age_gap 歳**(正直な近似)。hierarchical=true
    #      の活動(colleague_dinner)でのみ、年齢差の大きい相手の承諾に hierarchy_penalty を課す。----
    "hierarchy_penalty": 0.35,  # 上司同席(年齢差>boss_age_gap)の飲み会の承諾ペナルティ
    "boss_age_gap": 10,         # これを超える年齢差を上司/部下(階層)の代理指標にする(役職データ不在の近似)
    "activities": _DEFAULT_ACTIVITIES,
    "bands": _DEFAULT_BANDS,
    "age_weights": _DEFAULT_AGE_WEIGHTS,
}

_BOOL_KEYS = ("enabled",)
_INT_KEYS = ("min_group", "max_group", "friend_tier", "boss_age_gap")
_FLOAT_KEYS = ("daily_rate", "accept_base", "tier_bonus", "hierarchy_penalty")


def build_cfg(raw) -> dict:
    """conf の joint ブロックを型強制つきで正準化(既定 OFF=何も編成しない=バイト一致)。

    dotlist / OmegaConf どちらでも受ける(household/relations と同型)。マップ(activities/
    bands/age_weights)はキーを文字列に固定する(config_hash の json.dumps(sort_keys) 対策)。"""
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
        elif k == "activities":
            cfg[k] = _norm_activities(v)
        elif k == "bands":
            cfg[k] = {str(bk): [int(x) for x in bv] for bk, bv in (v or {}).items()}
        elif k == "age_weights":
            cfg[k] = {str(bk): {str(a): float(w) for a, w in (bv or {}).items()}
                      for bk, bv in (v or {}).items()}
    return cfg


def _norm_activities(raw) -> dict:
    out: dict = {}
    for name, spec in (raw or {}).items():
        spec = dict(spec or {})
        out[str(name)] = {
            "weight": float(spec.get("weight", 0.0)),
            "poi_cat": str(spec.get("poi_cat", "")),
            "band": str(spec.get("band", "day")),
            "activity_tag": str(spec.get("activity_tag", "")),
            # S-R4: 同伴者の関係タイプ(friend=友人 tier≥friend_tier / housemate=同居人 /
            #   colleague=同 org_id の同僚)。未指定は従来どおり友人経路。
            "companion_type": str(spec.get("companion_type", "friend")),
            # S-R4: 階層依存(true=上司同席で承諾↓。colleague_dinner=飲み会に付す)。
            "hierarchical": bool(spec.get("hierarchical", False)),
        }
    return out


# ---------------------------------------------------------------- 純関数ヘルパ
def _act_names(cfg: dict) -> list:
    """活動名の決定論順(POI ハッシュ・重み抽選の安定順)。"""
    return sorted(cfg["activities"].keys())


def _age_weight_row(cfg: dict, age) -> dict:
    """年齢に対応する年代 band(下限キーが age 以下で最大)の重み倍率 dict(該当なしは空)。"""
    bands = cfg.get("age_weights") or {}
    if not bands:
        return {}
    try:
        a = int(age)
    except (TypeError, ValueError):
        return {}
    chosen: dict = {}
    for lo in sorted(bands, key=lambda s: int(s)):
        if a >= int(lo):
            chosen = bands[lo]
        else:
            break
    return chosen


def _pick_activity(cfg: dict, age, rng) -> str | None:
    """活動タイプを重み(較正 weight × 年齢倍率)付きサンプル(新 stream "joint")。"""
    names = _act_names(cfg)
    if not names:
        return None
    row = _age_weight_row(cfg, age)
    weights = [float(cfg["activities"][n]["weight"]) * float(row.get(n, 1.0))
               for n in names]
    total = sum(weights)
    if total <= 0.0:
        return names[int(rng.integers(len(names)))]
    r = float(rng.random()) * total
    acc = 0.0
    for n, w in zip(names, weights):
        acc += w
        if r < acc:
            return n
    return names[-1]


def _band_minutes(cfg: dict, band_name: str):
    """band 名 → [開始分, 終了分]。未定義なら丸1日(常時)。"""
    b = (cfg.get("bands") or {}).get(band_name)
    if not b or len(b) < 2:
        return [0, 1440]
    return [int(b[0]), int(b[1])]


def _tier(sim, actor, other_id: int) -> int:
    """actor→other の relations tier(closeness 由来)。relations 無効/未接触は 0。"""
    rel = getattr(actor.mem, "relations", {}).get(other_id)
    if not rel:
        return 0
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    return _relations.tier_of(float(rel.get("closeness", 0.0)), rc)


def _resolve_with(sim, agent) -> list:
    """planning.framework の同伴者フィールド `with`(計画スキーマの名前)を relations 既知相手へ
    決定論解決する(名前一致→closeness 最大)。planning OFF/未解決は空(追加 LLM 呼ゼロ)。"""
    sched = getattr(agent, "day_schedule", None)
    if not sched:
        return []
    names: list = []
    for it in sched:
        for w in (it.get("with") or []):
            w = str(w).strip()
            if w and w not in names:
                names.append(w)
    if not names:
        return []
    rels = getattr(agent.mem, "relations", {}) or {}
    out: list = []
    for nm in names:
        best = None
        best_clo = -1e18
        for oid, rel in rels.items():
            rn = str(rel.get("name", ""))
            if rn == nm or (nm and nm in rn):
                clo = float(rel.get("closeness", 0.0))
                if clo > best_clo or (clo == best_clo and (best is None or oid < best)):
                    best_clo = clo
                    best = oid
        if best is not None and best not in out:
            out.append(best)
    return out


def _colleagues(sim, agent, assigned: set) -> list:
    """同 org_id の同僚 id を id 昇順で返す(S-R4)。org_id 無し/来街者/自分/既割当は除外。"""
    self_org = getattr(agent, "org_id", None)
    if not self_org:
        return []
    out: list = []
    for o in sorted(sim.agents, key=lambda a: a.id):
        if o.id == agent.id or o.id in assigned or o.visitor:
            continue
        if getattr(o, "org_id", None) == self_org:
            out.append(o.id)
    return out


def _companions(sim, agent, cfg: dict, assigned: set,
                companion_type: str = "friend",
                icfg: dict | None = None, sources: dict | None = None) -> list:
    """同伴候補を決定論で順序付ける。関係タイプ(companion_type)で経路を分ける(S-R4):
      friend    = planning の `with` 解決を先頭 → 友人 tier≥friend_tier を closeness 降順→id 昇順
                  → 友人ゼロなら housemates(既存 S-R3 挙動。未指定の既定)。
      housemate = 同居人(housemates)のみ。
      colleague = 同 org_id の同僚のみ(会食・飲み会)。
    来街者・既割当・自分は除外。

    第64バッチ フェーズ3(icfg=relations.endogenous_invite。既定 OFF=従来と候補順まで完全同一):
    ON 時のみ friend 経路を並べ替え・拡張する(選抜の枠組みは不変): `with` 解決(従来=最優先)の
    直後に前日発話の明示キュー相手を挿入し、末尾に弱い紐帯探索枠(tier=1・安定ハッシュ=乱数ゼロ)
    を付す。残りは従来の closeness 降順=較正分布を事前分布として維持(二段構えの invite 版)。
    sources(呼び手が渡す dict)に {id: 選抜経路} を書き戻す(plan_with/dialog_cue/closeness/
    weak_tie/housemate/colleague)。候補の総数は従来どおり呼び手の max_group 制約下=Dunbar 上限は
    不変。**関係の総量上限(friend_graph の層次数=close 3-5/friend 7-12/acq +20)も変更しない**:
    誘いは一時的接触であり tier 遷移は relations の closeness 蓄積経由のみ=誘い候補の拡張が層上限を
    新規超過させる経路は構造上ない(relations_endo 冒頭 docstring)。"""
    out: list = []
    seen: set = set()

    def _add(oid, src: str = ""):
        if oid == agent.id or oid in assigned or oid in seen:
            return
        # ★候補は**在場者に限る**(``agent_by_id`` は退場者も返す = 幽霊)。幽霊が
        #   max_group の枠を食うと、実在の友人が誘われず joint_fulfill_rate が機構的に
        #   単調低下する(= 観測の汚染)。
        o = sim.present_agent(oid)
        if o is None or o.visitor:
            return
        out.append(oid)
        seen.add(oid)
        if sources is not None:
            sources[oid] = src

    if companion_type == "colleague":
        for oid in _colleagues(sim, agent, assigned):
            _add(oid, "colleague")
        return out
    if companion_type == "housemate":
        for oid in (getattr(agent, "housemates", None) or []):
            _add(oid, "housemate")
        return out
    # friend(既定)
    inv_on = bool(icfg and icfg.get("enabled"))    # 第64: 誘い先の内生選抜(既定 OFF)
    for cid in _resolve_with(sim, agent):        # S-R3(7): `with` 名の解決相手を先頭に
        _add(cid, "plan_with")
    if inv_on:
        # 第64(2): 前日発話の明示キュー相手を次点へ挿入(語彙は accept ブロックと共用=単一の源)
        for cid in _endo.invite_cue_partners(sim, agent):
            _add(cid, "dialog_cue")
    ft = int(cfg["friend_tier"])
    friends: list = []
    for oid, rel in (getattr(agent.mem, "relations", {}) or {}).items():
        if _tier(sim, agent, oid) >= ft:
            friends.append((float(rel.get("closeness", 0.0)), oid))
    for _clo, oid in sorted(friends, key=lambda t: (-t[0], t[1])):
        _add(oid, "closeness")
    if not out:                                  # 友人がいなければ housemates(同居人)
        for oid in (getattr(agent, "housemates", None) or []):
            _add(oid, "housemate")
    if inv_on:
        # 第64(3): 弱い紐帯の探索枠(tier=1 知人・(agent,day) 安定ハッシュ=乱数ゼロ・候補末尾。
        # 理論根拠: Granovetter 1973 弱い紐帯=新規情報の橋/現実の交流量は強い紐帯が大半
        # (Onnela 2007 PNAS)=末尾枠は保守的既定。relations_endo.INVITE_DEFAULTS 参照)
        day = int(getattr(sim, "_joint_day", 0))
        for oid in _endo.weak_tie_candidates(
                sim, agent, day, int(icfg.get("weak_tie_slots", 1)),
                assigned | seen | {agent.id}):
            _add(oid, "weak_tie")
    return out


def accept_prob(cfg: dict, tier: int, hierarchical: bool, age_gap: int) -> float:
    """誘いの承諾確率(決定論の式)= accept_base + tier_bonus(親友) − hierarchy_penalty(上司同席)。

    §2.5 の階層依存: hierarchical=true(飲み会)かつ相手との年齢差が boss_age_gap を超える(=上司/
    部下の代理指標。組織台帳に役職データが無いための正直な近似)とき承諾を hierarchy_penalty 下げる。
    同世代(年齢差小)・非階層活動(ランチ)ではペナルティ無し=許容。"""
    p = float(cfg["accept_base"]) + (float(cfg["tier_bonus"]) if tier >= 3 else 0.0)
    if hierarchical and int(age_gap) > int(cfg["boss_age_gap"]):
        p -= float(cfg["hierarchy_penalty"])
    return p


def _rendezvous_poi(sim, group: list, day: int, act: str, cfg: dict) -> str | None:
    """(グループ最小 id, day, 活動タイプ) から活動カテゴリ内 POI を1つ純関数導出=全員が同じ POI を
    選ぶ(date_dest の一般化)。カテゴリに POI が無ければ dests から後退(それも無ければ None)。"""
    cat = cfg["activities"][act]["poi_cat"]
    lo = min(group)
    ai = _act_names(cfg).index(act) if act in cfg["activities"] else 0
    key = lo * 1000003 + day * 97 + ai * 7
    pool = sim.city.pois_by_cat(cat)
    if pool:
        return pool[key % len(pool)]["node"]
    dests = sim.dests
    if dests:
        return dests[key % len(dests)]
    return None


# ---------------------------------------------------------------- 日次編成(日境界)
def plan_day(sim, step: int, sim_min: int) -> None:
    """日境界で当日の共同活動を編成する(id 昇順・居住者のみ・新 stream "joint")。

    各未割当の居住者につき daily_rate で発生判定→活動タイプ抽選→同伴候補選択→誘い→承諾
    (承諾者1人以上で成立・2-4人)→ランデブー POI 導出、を全決定論で行い、成立グループの各
    メンバに joint_today(当日の共同予定)を付す。翌日の plan_day 冒頭で全員クリアする。"""
    cfg = getattr(sim, "jointcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_joint_day", -1):
        return
    sim._joint_day = day
    # 第62バッチ: 承諾判断の内生化(relations.endogenous_accept。既定 OFF=判定・記録経路に
    # 一切入らず sim._endo_state も生えない=バイト一致)。設計正典: docs/plans/
    # endogenous-relations-plan.md §2。判定は relations_endo.decide_accept(全決定論・LLM呼ゼロ)。
    ecfg = _endo.cfg_of(sim)
    endo_on = bool(ecfg["enabled"])
    est = _endo.day_state(sim, day) if endo_on else None
    # 第64バッチ フェーズ3: 誘う相手の内生選抜(relations.endogenous_invite。既定 OFF=候補順・
    # 記録経路とも従来と完全同一で sim._invite_state も生えない=バイト一致)。実装のみ=実験投入は
    # phase3_go ゲート(計画書§4)。選抜は _companions 内(全決定論・乱数/LLM 呼ゼロ)。
    icfg = _endo.invite_cfg_of(sim)
    inv_on = bool(icfg["enabled"])
    ist = _endo.invite_day_state(sim, day) if inv_on else None
    for a in sim.agents:                              # 前日分をクリア(日境界=当日で上書き)
        a.joint_today = None
    sim._joint_groups = []
    assigned: set = set()
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    for a in residents:
        if a.id in assigned:
            continue
        rng = sim.hub.stream("joint", a.id, day)      # 個体別=編成順に非依存
        if float(rng.random()) >= float(cfg["daily_rate"]):
            continue
        act = _pick_activity(cfg, getattr(a, "age", 0), rng)
        if act is None:
            continue
        spec = cfg["activities"][act]
        ctype = spec.get("companion_type", "friend")   # S-R4: 友人/同居人/同僚
        hier = bool(spec.get("hierarchical", False))   # S-R4: 上司同席で承諾↓(飲み会)
        srcs: dict | None = {} if inv_on else None     # 第64: {id: 選抜経路}(OFF は None=素通り)
        cands = _companions(sim, a, cfg, assigned, ctype, icfg, srcs)
        if not cands:
            continue
        group = [a.id]
        max_g = int(cfg["max_group"])
        a_age = int(getattr(a, "age", 0) or 0)
        act_band = _band_minutes(cfg, spec["band"])    # 活動の時間帯(第62: 衝突判定にも使う)
        for cid in cands:
            if len(group) >= max_g:
                break
            other = sim.present_agent(cid)         # 在場者のみ(_companions で濾過済み・二重の帯)
            age_gap = abs(a_age - int(getattr(other, "age", 0) or 0)) if other else 0
            p_calib = accept_prob(cfg, _tier(sim, a, cid), hier, age_gap)  # S-R4: 階層依存
            gp = _gossip.joint_penalty(sim, a, cid)   # 負の評判(第61 c): 悪評を知る相手は誘いにくい(既定 OFF=0)
            r = float(rng.random())                   # 誘い→承諾の抽選(always-draw: ON/OFF で消費数・順序不変)
            if inv_on:                                # 第64: 誘い経路の当日タリー(L2 invite_* 2列の材料)
                _endo.tally_invite_source(ist, srcs.get(cid, "closeness"))
            if endo_on and other is not None:
                # 第62: 内生判定(構造化・決定論)→ 合成 p=clamp(w·p_calib+(1−w)·p_endo)−gossip
                # (gossip は常に最後の減算=第61 の不変則)。conflict_veto は確率でなく確定拒否
                # = 直前の draw r を破棄(conditionally-use。draw 列はズレない)。
                verdict, basis = _endo.decide_accept(sim, a, other, act, act_band, ecfg)
                p_mix, forced = _endo.compose(p_calib, verdict, basis, ecfg)
                p_final = 0.0 if forced else p_mix - gp
                ok = (not forced) and (r < p_final)
                _endo.tally_invite(est, cid, verdict, p_calib, ok)
                payload = {"invitee": int(cid),
                           "verdict": verdict or "none",
                           "basis": basis,
                           "p_calib": round(float(p_calib), 4),
                           "p_final": round(float(p_final), 4),
                           "accepted": bool(ok)}
                if inv_on:
                    # 第64: 誘い先の選抜経路(invite ON 時のみ=accept 単独 ON の payload は
                    # 従来とバイト一致のまま)。
                    payload["source"] = srcs.get(cid, "closeness")
                sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=a.id,
                                     kind="joint_invite", x=a.x, y=a.y,
                                     payload=payload))
            else:
                ok = r < (p_calib - gp)               # 従来の較正抽選(既定=バイト一致)
            if ok:
                group.append(cid)
        if len(group) < int(cfg["min_group"]):
            continue
        poi = _rendezvous_poi(sim, group, day, act, cfg)
        if poi is None:
            continue
        band = act_band
        tag = cfg["activities"][act]["activity_tag"]
        grp_tier = max((_tier(sim, a, gid) for gid in group[1:]), default=0)
        for gid in group:
            g = sim.present_agent(gid)             # 幽霊に joint_today を書いても hydrate で消える
            if g is not None:
                g.joint_today = {"poi": poi, "band": band, "activity": act,
                                 "activity_tag": tag, "tier": grp_tier}
            assigned.add(gid)
        sim._joint_groups.append({"members": list(group), "poi": poi,
                                  "activity": act, "band": band,
                                  "tier": grp_tier, "logged": False})


# ---------------------------------------------------------------- 収束(自由行動の行き先)
def joint_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """当日の共同予定 POI への行き先バイアス(活動 band 内のみ)。決定論・乱数ゼロ・非LLM。

    joint OFF / 当日予定なし / band 外 なら乱数を一切引かず None(既定不変)。band 内なら
    予定 POI を返す(現在地と同じなら None=到着済み)。編成時に全メンバ同一 POI を格納済み
    なので合流(co-location)が起きる。date 分岐より優先度は下(routine の elif 順で date が勝つ)。"""
    cfg = getattr(sim, "jointcfg", None)
    if not cfg or not cfg["enabled"]:
        return None
    jt = getattr(agent, "joint_today", None)
    if not jt:
        return None
    lo, hi = jt["band"]
    m = sim_min % 1440
    if not (lo <= m < hi):
        return None
    poi = jt["poi"]
    return poi if poi and poi != agent.node else None


def route_activity(agent) -> str:
    """共同行動 move に載せる routine 活動タグ(cost 用)。meal_cafe=eating(既存 _charge_meal が
    飲食費を課金)、他は ""(追加課金なし)。joint_dest が非 None を返す band 内でのみ呼ばれる。"""
    jt = getattr(agent, "joint_today", None)
    return jt.get("activity_tag", "") if jt else ""


# ---------------------------------------------------------------- 観測(同席で1件)
def observe(sim, step: int, sim_min: int) -> None:
    """毎 step: 成立グループが実際に POI で2人以上同席した最初の step で joint_activity を1件記録
    (1グループ1日1回)。joint OFF / band 外 / グループ無しは即 return(=バイト一致)。"""
    cfg = getattr(sim, "jointcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    groups = getattr(sim, "_joint_groups", None)
    if not groups:
        return
    m = sim_min % 1440
    est = _endo.active_state(sim)   # 第62: 履行観測(ON かつ当日タリーがある時のみ。OFF=None=素通り)
    for grp in groups:
        lo, hi = grp["band"]
        if not (lo <= m < hi):
            continue
        if est is not None:
            _endo.mark_fulfilled(sim, grp, est)      # 承諾者の実同席を記録(読むだけ・乱数ゼロ)
        if grp["logged"]:
            continue
        poi = grp["poi"]
        # ★同席は**在場者だけ**で数える。退場者は脱水時点の古い node を持ったままなので、
        #   ``agent_by_id`` で引くと「街に居ないのに待ち合わせ場所に居る」幽霊の同席になる。
        present = [gid for gid in grp["members"]
                   if (o := sim.present_agent(gid)) is not None and o.node == poi]
        if len(present) >= 2:
            grp["logged"] = True
            # 第75バッチ(ダンバー認知枠。既定 OFF=即 return=バイト一致): 同席は関係の**維持**
            # 行為=活性関係の last_step を進め、休眠中の相手なら再会させる(relation_rekindle)。
            # 1グループ1日1回のこの地点だけで呼ぶ(毎 step の全ペア走査をしない)。
            _dunbar.touch_group(sim, present, step, sim_min)
            leader = sim.present_agent(min(present))
            lx = leader.x if leader is not None else 0.0
            ly = leader.y if leader is not None else 0.0
            sim.logger.log(Event(step=step, sim_min=sim_min,
                                 agent_id=int(min(present)), kind="joint_activity",
                                 x=lx, y=ly,
                                 payload={"type": grp["activity"],
                                          "with": sorted(int(g) for g in present),
                                          "place": poi, "tier": int(grp["tier"])}))
            sim._joint_total = getattr(sim, "_joint_total", 0) + 1
