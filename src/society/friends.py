"""友人グラフ生成(関係性の再現 第45バッチ S-R2。ユーザー要望 2026-07-21)。

現状の初期関係は「同建物の顔なじみ最大3人」(simulation.py の顔なじみブロック)のみ=地理的共在
だけが源泉で、homophily(年齢/職業の類似)・学校/職場の紐帯が無い。本モジュールは起動時1回、
居住者の現実的な友人ネットワークを決定論で張る:

  - homophily(McPherson 2001 "Birds of a Feather"): 類似が紐帯を生む。強さ順に 年齢>職業。
    → 年齢近接を最も強く、職業一致を次に重み付ける。
  - 共有所属: 同 work org_id(社会人後の新規友人は職場34.0%)・学生同士かつ同 school org_id
    (現在の友人は学生時代中心31.6%)。
  - 同地区近接(同一住宅建物=近所)。
  - 次数は Dunbar の入れ子層で較正: 親友~3-5(tier3=支援クリーク)・友人~10-15(tier2)・知人
    (tier1=弱い紐帯を薄く)。relations.py の tier 閾値(2.0/5.0/12.0)へ closeness を直接注入して
    その層に載せる(顔なじみ経路と同じ record_contact を使い、同一ペアの二重辺は closeness を
    加算せず直接代入で避ける)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  homophily/Dunbar の較正値・所属ロジックはここ(と conf)にのみ書く=no-fingerprint 契約に触れない。

決定論・run.seed 非依存(比較実験の要): 辺は (persona id ペア, friend_graph.seed) の安定ハッシュ
  (hashlib=RngHub 無風=乱数 stream を1本も引かない。ontology._stable_uniform と同方式)+ ペルソナ
  属性(age/occupation/org_id/home)の純関数。同一設定なら別ランでも同一人物ペアは同一の間柄。
  プールの pool_pid(build_persona_pool.py 生成物=run.seed 非依存で固定)がある個体はそれを、直接
  ランは agent.id を pid とする。来街者は対象外(家族・友人は圏外=household と整合)。既存の顔なじみ
  経路(simulation.py)はそのまま(friend_graph ON でも closeness を直接代入=二重辺を作らない)。

既定 OFF(enabled=false)= 何も張らず・relations 台帳に一切触れず・friend_graph_built も出さない
  =乱数消費・イベント列・プロンプトともバイト一致(ゴールデン golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

import hashlib

from . import relations as _relations
from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    "seed": 20260722,          # 安定ハッシュ種(run.seed と独立=別ランで同一人物ペアは同一辺)
    # ---- homophily の重み(McPherson: 年齢>職業)----
    "w_age": 1.0,              # 年齢近接(最強)
    "w_occ": 0.5,              # 職業一致(次点)
    "w_same_work": 1.2,        # 共有所属: 同 work org_id(職場友人34.0%)
    "w_same_school": 1.0,      # 共有所属: 学生同士かつ同 school org_id(学生時代友人31.6%)
    "w_same_area": 0.4,        # 同地区近接(同一住宅建物=近所)
    "age_scale": 15.0,         # 年齢差の正規化スケール(歳。この差で年齢類似度が 0 になる)
    "noise": 0.3,              # スコアの決定論ノイズ幅(タイ破り・多様性=run.seed 非依存)
    # ---- Dunbar 入れ子層の次数較正(親友~3-5 / 友人~10-15 / 知人=弱い紐帯を薄く)----
    "close_min": 3,            # 親友層(tier3)の下限次数
    "close_max": 5,            # 親友層の上限次数
    "friend_min": 7,           # 友人層(tier2)の追加次数下限(親友+友人で~10-15)
    "friend_max": 12,          # 友人層の追加次数上限
    "acq_extra": 20,           # 知人層(tier1)の追加次数(弱い紐帯を薄く)
    "margin": 0.5,             # 注入 closeness の上乗せ(閾値+margin=その tier に確実に載る)
    # ---- β4 初期関係の減衰整合(第117 レーンB3・**既定 None = 上の margin と完全に同一**)----
    # 正典: docs/research/initial-relations-improvement.md §0 R2 / §1.3 / §4 機構4 (4-d) / §8.2。
    # 何が壊れていたか: 注入値は「tier 閾値 + 0.5」ちょうどなのに、`relations.decay_day` は
    #   closeness を持つ台帳を **毎日 1.0 減らす**(decay_per_day=1.0 / decay_after_days=0)。
    #   → 接触が無ければ **親友は翌日に tier2・知人は 3 日で関係消滅**、親友も 13 日で
    #   closeness 0 = **初期友人グラフは 2 週間で蒸発する過渡現象**だった(§1.3 の表)。
    #   しかも顔なじみ・icebreak は closeness を持たないので減衰の対象外 =
    #   **構造化された初期関係だけが真っ先に腐る**という逆転が起きていた。
    # 一次データ: 日本人の対面交際頻度の**中央値は「月1回〜2週に1回」**(§8.2)。
    #   「知人は 2 日会わないと消える」は現実と 1 桁以上ずれている。
    # 直し方(§4 R2): margin を層別に分け、減衰 1.0/日 の下で
    #   **親友 ≈ 2 週間 / 友人 ≈ 1 週間 / 知人 2〜3 日**は接触ゼロでも層に留まるようにする。
    # ★上限の制約: `relations.tier_of` は closeness から tier を引き直すので、注入値が
    #   **上の層の閾値に届くと昇格してしまう**(margin_friend は tier_close−tier_friend 未満、
    #   margin_acq は tier_friend−tier_acquaintance 未満でなければならない)。
    # None = このキーを書かない = 従来どおり全層 `margin` = ゴールデン L1 バイト一致。
    "margin_close": None,      # 親友(tier3)の上乗せ。既定 None → margin
    "margin_friend": None,     # 友人(tier2)の上乗せ。既定 None → margin
    "margin_acq": None,        # 知人(tier1)の上乗せ。既定 None → margin
    # ---- AGE-D: 次数の年齢曲線(第116バッチ 2026-08-15・**既定 OFF**)----
    # 正典: docs/plans/age-diversity-plan.md §4-6。
    # 現状の穴: `w_age` で「誰と繋がるか」は年齢に依るのに、**次数(何人と繋がるか)は
    #   年齢非依存**だった(15 歳も 75 歳も同じ 3-5 + 7-12 + 20)。
    # 較正(Bhattacharya, Ghosh, Monsivais, Dunbar & Kaski 2016 *R. Soc. Open Sci.* 3:160097。
    #   携帯 CDR **660 万ユーザー・年齢性別既知 320 万**): 月間 alter 数は **25 歳前後で最大の
    #   15-20 人** → 45 歳まで減少 → **45-55 は台地** → 55 以降また減少。
    #   Dunbar 2020 *Proc. R. Soc. A* 476:20200446 も「**年齢の逆 J 字関数・20-30 代ピーク**」。
    # ★重要な形の制約(4 つの独立ソースが一致): **加齢が削るのは外層(周辺・同僚)であって
    #   内核ではない**。親友は年齢不変(Bruine de Bruin: 周辺 r=−.13 / **親友 r=.01**)、
    #   内核はむしろ増える(English & Carstensen 2014: 内核 6.21→7.75 / 周辺 7.35→7.06)、
    #   家族ネットワークは規模が安定(Wrzus 2013 メタ分析 277 研究 177,635 人)。
    #   ⇒ 実装は「目標次数を年齢で縮める」ではなく **「弱紐帯(tier2 友人 / tier1 知人)の
    #   生成数だけを年齢で縮め、強紐帯(tier3 親友)は 1 人も触らない」**。
    "age_degree": False,       # ★これが AGE-D の唯一のトグル(既定 OFF=現行と完全同一)
    "age_degree_ref": 25.0,    # この年齢で倍率 1.0(= 現行較正値がピーク年齢に対応する)
    # 月間 alter 数 / ピーク(25 歳 ≈ 17.5 人)。45-55 の台地と 55 以降の再減少を折れ線で。
    "age_degree_knots": [[15.0, 0.75], [25.0, 1.00], [40.0, 0.66],
                         [45.0, 0.60], [55.0, 0.60], [70.0, 0.45], [85.0, 0.35]],
    "age_degree_min": 0.20,
    "age_degree_max": 1.20,
}

_BOOL_KEYS = ("enabled", "age_degree")
_INT_KEYS = ("seed", "close_min", "close_max", "friend_min", "friend_max", "acq_extra")
_FLOAT_KEYS = ("w_age", "w_occ", "w_same_work", "w_same_school", "w_same_area",
               "age_scale", "noise", "margin", "age_degree_ref",
               "age_degree_min", "age_degree_max")
# 層別 margin は「書かない = None = margin へ後退」を表せる必要があるので float 強制しない
_OPT_FLOAT_KEYS = ("margin_close", "margin_friend", "margin_acq")
_KNOT_KEYS = ("age_degree_knots",)
# 層別 margin の tier 対応(3=親友 / 2=友人 / 1=知人)。build_friend_graph が引く。
_MARGIN_KEY_OF_TIER = {3: "margin_close", 2: "margin_friend", 1: "margin_acq"}


def build_cfg(raw) -> dict:
    """conf の friend_graph ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in _OPT_FLOAT_KEYS:
            cfg[k] = None if v is None else float(v)
        elif k in _KNOT_KEYS:
            cfg[k] = sorted(([float(x), float(y)] for x, y in (v or [])),
                            key=lambda p: p[0])
    return cfg


# ---------------------------------------------------------------- 純関数ヘルパ
def _stable_uniform(seed: int, key: str) -> float:
    """(seed, key) から run.seed 非依存の一様値 [0,1)(hashlib=決定論・RngHub 無風)。

    ontology._stable_uniform と同方式(blake2b はプロセス跨ぎで安定=別ラン/resume でも同一)。"""
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _pid(agent) -> str:
    """安定 persona id(プールは pool_pid=run.seed 非依存で固定、直接ランは agent.id)。"""
    pid = getattr(agent, "pool_pid", None)
    return str(pid) if pid is not None else str(agent.id)


def _pair_key(pa: str, pb: str) -> str:
    """順序に依らないペアキー(a,b と b,a で同一=対称なノイズ)。"""
    return f"{pa}\x1e{pb}" if pa <= pb else f"{pb}\x1e{pa}"


def _score(a, b, cfg: dict) -> float:
    """居住者ペアの親和スコア(homophily+共有所属+近接+決定論ノイズ)。決定論・乱数ゼロ。"""
    s = 0.0
    # homophily 年齢近接(McPherson で最強クラス)。
    aa = int(getattr(a, "age", 0) or 0)
    ab = int(getattr(b, "age", 0) or 0)
    s += float(cfg["w_age"]) * max(0.0, 1.0 - abs(aa - ab) / float(cfg["age_scale"]))
    # homophily 職業一致。
    occ = getattr(a, "occupation", "")
    if occ and occ == getattr(b, "occupation", ""):
        s += float(cfg["w_occ"])
    # 共有所属: 同 org_id(学生同士=学生時代友人 / それ以外=職場友人)。
    oa = getattr(a, "org_id", None)
    ob = getattr(b, "org_id", None)
    if oa and ob and oa == ob:
        if (getattr(a, "org_role", "") == "学生"
                and getattr(b, "org_role", "") == "学生"):
            s += float(cfg["w_same_school"])
        else:
            s += float(cfg["w_same_work"])
    # 同地区近接(同一住宅建物=近所)。
    hb = getattr(a, "home_building", "")
    if hb and hb == getattr(b, "home_building", ""):
        s += float(cfg["w_same_area"])
    # 決定論ノイズ(タイ破り・多様性)= run.seed 非依存。
    s += float(cfg["noise"]) * _stable_uniform(int(cfg["seed"]), _pair_key(_pid(a), _pid(b)))
    return s


def _degree(cfg: dict, pid: str, lo_key: str, hi_key: str, salt: str) -> int:
    """個体別の層内次数を [lo, hi] から安定ハッシュで決める(run.seed 非依存)。"""
    lo = int(cfg[lo_key])
    hi = int(cfg[hi_key])
    if hi <= lo:
        return max(0, lo)
    u = _stable_uniform(int(cfg["seed"]) + 1, f"{salt}\x1f{pid}")
    return lo + int(u * (hi - lo + 1))


def age_degree_mult(age, cfg: dict) -> float:
    """年齢 → **弱紐帯の次数**にかける倍率(AGE-D。決定論・乱数ゼロ・OFF は常に 1.0)。

    折れ線(`age_degree_knots`)を `age_degree_ref` で規格化する。範囲外は端の値で平ら
    (外挿しない = 6 歳や 82 歳へ曲線を伸ばして偽の精度を作らない)。"""
    if not cfg.get("age_degree", False):
        return 1.0
    knots = cfg.get("age_degree_knots") or []
    if not knots:
        return 1.0

    def at(x: float) -> float:
        if x <= knots[0][0]:
            return knots[0][1]
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x <= x1:
                span = x1 - x0
                return y1 if span <= 0.0 else y0 + (y1 - y0) * (x - x0) / span
        return knots[-1][1]

    ref = at(float(cfg["age_degree_ref"])) or 1.0
    m = at(float(int(age or 0))) / ref
    return min(float(cfg["age_degree_max"]), max(float(cfg["age_degree_min"]), m))


def _inject(a, b, tier: int, closeness: float) -> None:
    """a→b の関係を tier/closeness へ確定注入(直接代入=二重辺でも closeness を膨らませない)。

    record_contact で台帳エントリを確保(顔なじみ経路と同じ)し、closeness/tier を直接据える。
    tier は relations の消費側(social_lines は rel['tier'] を、joint は closeness を tier_of で
    読む)双方が整合するよう両方を据える。"""
    rel = a.mem.record_contact(b.id, b.name, 0, "友人")
    rel["closeness"] = float(closeness)
    rel["tier"] = int(tier)


# ---------------------------------------------------------------- 起動時1回
def build_friend_graph(sim) -> None:
    """居住者の友人ネットワークを起動時に決定論で張る(既定 OFF=no-op=バイト一致)。

    顔なじみブロックの直後に1呼び出しで呼ばれる。各居住者の相手を親和スコア降順に並べ、Dunbar 層
    (親友/友人/知人)で desired tier を割り(有向)、ペアの max tier で対称化して closeness/tier を
    注入する。乱数 stream を1本も引かない(全 hashlib)=既存 draw 順に無影響=run.seed 非依存。"""
    cfg = getattr(sim, "friendcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    if len(residents) < 2:
        return
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    thr = {1: float(rc["tier_acquaintance"]), 2: float(rc["tier_friend"]),
           3: float(rc["tier_close"])}
    margin = float(cfg["margin"])
    # β4: 層別 margin(既定は 3 層とも None = margin = 従来と 1 バイトも変わらない)。
    # 減衰 1.0/日 の下で「その層が接触ゼロで何日もつか」を決める唯一の数値。
    margin_of = {tier: (margin if cfg.get(key) is None else float(cfg[key]))
                 for tier, key in _MARGIN_KEY_OF_TIER.items()}
    # 各居住者の相手を親和スコア降順に並べ、Dunbar 層で desired tier を割る(有向)。
    desired: dict = {}
    for a in residents:
        pid = _pid(a)
        n_close = _degree(cfg, pid, "close_min", "close_max", "close")
        n_friend = _degree(cfg, pid, "friend_min", "friend_max", "friend")
        n_acq = int(cfg["acq_extra"])
        # AGE-D: 年齢曲線は**弱紐帯(友人 tier2 / 知人 tier1)だけ**を伸縮させ、
        # 親友(tier3=内核)は 1 人も触らない(加齢が削るのは外層という 4 ソース一致の所見)。
        # 既定 OFF では mult が厳密に 1.0 = 下の 2 行は恒等 = バイト一致。
        mult = age_degree_mult(getattr(a, "age", 0), cfg)
        if mult != 1.0:
            n_friend = max(0, int(round(n_friend * mult)))
            n_acq = max(0, int(round(n_acq * mult)))
        ranked = sorted((b for b in residents if b.id != a.id),
                        key=lambda b: (-_score(a, b, cfg), b.id))
        for rank, b in enumerate(ranked):
            if rank < n_close:
                desired[(a.id, b.id)] = 3
            elif rank < n_close + n_friend:
                desired[(a.id, b.id)] = 2
            elif rank < n_close + n_friend + n_acq:
                desired[(a.id, b.id)] = 1
            else:
                break
    # 対称化(ペアの max tier)+ closeness/tier を注入。
    n_edges = 0
    for i, a in enumerate(residents):
        for b in residents[i + 1:]:
            tier = max(desired.get((a.id, b.id), 0), desired.get((b.id, a.id), 0))
            if tier <= 0:
                continue
            clo = thr[tier] + margin_of[tier]
            _inject(a, b, tier, clo)
            _inject(b, a, tier, clo)
            n_edges += 1
    mean_deg = round(2.0 * n_edges / len(residents), 4) if residents else 0.0
    sim.logger.log(Event(step=0, sim_min=0, agent_id=-1, kind="friend_graph_built",
                         x=0.0, y=0.0,
                         payload={"n_edges": int(n_edges), "mean_degree": mean_deg}))
