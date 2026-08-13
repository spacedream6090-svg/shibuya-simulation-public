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
    # ---- 世帯の現実化(S-R1。既定 false=現行の id 順機械束ねと完全同一)----
    "realistic": False,                    # true で人口統計整合の束ね(地理×年齢)+続柄付与+渋谷 size_weights
    # 渋谷区の現実(令和2国勢調査 §2.1: 単身64.5%・2人以上35.5%)。2人以上は全国分布(2/3/4=28.1/16.6/11.9)で按分。
    "size_weights_realistic": [0.645, 0.176, 0.104, 0.075],
    # ---- 恋愛・パートナー形成(日次。G2 closeness から決定論)----
    "partner_closeness": 15.0,             # 相互 closeness がこれ以上でパートナー成立(親友 tier_close の上位)
    "date_bias": 0.5,                      # 自由時間にパートナーとの共有デート先へ寄せる確率/step(stream "date")
    # ---- 夕食の共食(S-R1。夜帯に世帯メンバの自由時間行き先を home へ寄せ、2人以上同席で joint_activity 記録)----
    #  prob=農水省 食育調査 令和5(§2.4: 家族と夕食「ほぼ毎日」68.7%)。band=18:00-21:00(分 int)。
    "family_dinner": {"enabled": False, "prob": 0.69,
                      "band_start_min": 1080, "band_end_min": 1260},
    # ---- bond→同棲/unbond→転出(内部可動性 第60バッチ b。既定 OFF=世帯の物理再編なし=バイト一致)----
    #  days=同棲判定の最短 bond 継続日数、closeness=維持すべき相互 closeness(0=partner_closeness)。
    #  ロジックは mobility.py(cohabit_day / on_breakup)。ここは config のみ。
    "cohabit": {"enabled": False, "days": 14, "closeness": 0.0},
    # ---- プール名簿からの世帯再構築(レーン乙 ブロック3。既定 OFF=完全 no-op=バイト一致)----
    #  build_households は**起動時 1 回**しか走らないので、プール回転で day1 以降に入場する
    #  個体(finals 構成では 20.9 万人)は世帯を一切持てなかった。さらに housemates /
    #  household_id は退避に載っておらず、build_pool_agent が home_* を素の record から
    #  作り直すため「同居しているという事実」そのものが回転のたびに壊れていた。
    #  ON にすると世帯を **pool 名簿の決定論的な分割**として持ち、入場のたびに冪等に着席する
    #  (org_id 方式と同型: 世界の乱数ではなく名簿が世帯を決める)。乱数 stream は 1 本も引かない
    #  (すべて安定ハッシュ)。seed は run.seed と独立=別ランでも同じ人が同じ世帯。
    "pool_bind": {"enabled": False, "seed": 20260813},
}

_BOOL_KEYS = ("enabled", "realistic")
_LIST_INT_KEYS = ("sizes",)
_LIST_FLOAT_KEYS = ("size_weights", "size_weights_realistic")

# 続柄ラベル(現実化 S-R1)。中立な家族語のみ(構成概念名・地名を含まない)。
_ROLE_SPOUSE = {"男": "夫", "女": "妻"}


def _fd_cfg(raw) -> dict:
    """family_dinner サブブロックを型強制つきで正準化(band は分 int)。"""
    base = dict(DEFAULTS["family_dinner"])
    raw = dict(raw or {})
    for k, v in raw.items():
        if k not in base:
            continue
        if k == "enabled":
            base[k] = bool(v)
        elif k == "prob":
            base[k] = float(v)
        else:
            base[k] = int(v)
    return base


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
        elif k == "family_dinner":
            cfg[k] = _fd_cfg(v)
        elif k == "cohabit":
            from . import mobility as _mobility
            cfg[k] = _mobility.build_cohabit_cfg(v)
        elif k == "pool_bind":                     # レーン乙 ブロック3(既定 OFF=完全 no-op)
            pb = dict(DEFAULTS["pool_bind"])
            for kk, vv in dict(v or {}).items():
                if kk in pb:
                    pb[kk] = bool(vv) if kk == "enabled" else int(vv)
            cfg[k] = pb
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


def _assign_household(members: list, is_family: bool, hh_id: str,
                      roles: list | None = None) -> None:
    """世帯メンバに household_id/kind/housemates/共有 home を付す(代表=最小 id の住居を共有)。"""
    rep = members[0]
    for i, m in enumerate(members):
        m.household_id = hh_id
        m.household_kind = "family" if is_family else "roommate"
        m.household_role = roles[i] if roles else ""
        m.housemates = [o.id for o in members if o.id != m.id]
        m.home_building = rep.home_building
        m.home_node = rep.home_node
        m.home_floor = rep.home_floor


def _family_roles(members_by_age: list) -> list:
    """年齢昇順のメンバ列に続柄を割り当てる(決定論・現実化 S-R1)。

    2人=年齢近接ペア(夫婦): 性別で夫/妻(同性・不明は同居人)。3-4人=親子: 年長の2人を親
    (夫婦=夫/妻)、年少を子。年齢差は「年長=親・年少=子」で必ず現れる(束ねは年齢昇順)。"""
    n = len(members_by_age)
    if n == 2:
        return [_ROLE_SPOUSE.get(getattr(m, "gender", ""), "同居人")
                for m in members_by_age]
    # 3-4人: 年長2人(末尾2)=親(夫婦の続柄)、残り(年少)=子
    roles = ["子"] * n
    for j in range(n - 2, n):                       # 末尾2 = 年長 = 親
        roles[j] = _ROLE_SPOUSE.get(getattr(members_by_age[j], "gender", ""), "親")
    return roles


def _geo_key(a) -> str:
    """束ねの地理キー(同じ住居建物→近接)。建物 id が無ければ home ノードで後退。"""
    return getattr(a, "home_building", "") or getattr(a, "home_node", "") or ""


def build_households(sim) -> None:
    """起動時1回: 居住者を決定論で世帯にまとめ、同一世帯で住居(home)を共有する。

    既定(realistic=false): 居住者を id 昇順に並べ、新 stream "household" からサイズを引いて
    先頭から詰める(既存 draw 順に一切影響しない=OFF は完全 no-op)。realistic=true: 居住者を
    (地理キー, 年齢) でソートしてから束ね(size_weights_realistic=渋谷実数)、2人=年齢近接の夫婦・
    3-4人=年長の親+年少の子を決定論で割り当て household_role を付す。乱数は "household" のみ。
    2人以上の世帯にだけ household_id・同居者 id を付し、home を代表(最小 id)へ寄せる(夜間の家庭内
    co-location)。simulation では本呼び出しの後に「顔なじみ」ブロックが home_building 共有から
    初期関係を張る。来街者(街の外に家)は世帯を組まない。"""
    cfg = sim.householdcfg
    if not cfg["enabled"]:
        return
    if pool_bind_on(sim):
        # ★レーン乙 ブロック3: pool 名簿由来の世帯が有効なら、**day0 の着席もそちらに従う**。
        #   でないと「day0 に居た人は起動時束ね(hh…)/ 後から来た人は名簿束ね(hp…)」の
        #   2 つの分割が並立し、同じ世帯のはずの 2 人が別々の世帯に入る。
        #   乱数 stream "household" は 1 度も引かない(RngHub.stream は毎回新しい Generator を
        #   派生するので、引かないことで他の列は 1 粒も動かない)。
        for a in sorted(sim.agents, key=lambda a: a.id):
            bind_pool_household(sim, a)
        return
    rng = sim.hub.stream("household")
    realistic = bool(cfg.get("realistic"))
    if realistic:                                  # 人口統計整合の束ね(地理×年齢)
        residents = sorted(
            (a for a in sim.agents if not a.visitor),
            key=lambda a: (_geo_key(a), int(getattr(a, "age", 0)), a.id))
        weights = cfg.get("size_weights_realistic") or cfg["size_weights"]
    else:                                          # 現行=id 昇順の機械束ね(バイト一致)
        residents = sorted((a for a in sim.agents if not a.visitor),
                           key=lambda a: a.id)
        weights = cfg["size_weights"]
    scfg = dict(cfg)
    scfg["size_weights"] = weights
    i = 0
    idx = 0
    while i < len(residents):
        size = _pick_size(scfg, rng)
        members = residents[i:i + size]
        i += len(members)
        if len(members) < 2:                       # 単身世帯は個人のまま(何も付けない)
            continue
        is_family = bool(rng.random() < float(cfg["family_ratio"]))
        hh_id = f"hh{idx}"
        idx += 1
        roles = None
        if realistic and is_family:                # 続柄割り当て(年齢昇順で親子/夫婦)
            by_age = sorted(members, key=lambda m: (int(getattr(m, "age", 0)), m.id))
            role_of = dict(zip((m.id for m in by_age), _family_roles(by_age)))
            roles = [role_of[m.id] for m in members]
        elif realistic:                            # ルームシェア=同居人
            roles = ["同居人"] * len(members)
        _assign_household(members, is_family, hh_id, roles)


# ================================================ プール名簿からの世帯再構築(レーン乙 ブロック3)
# なぜ要るか(現物の穴):
#   ① ``build_households`` は **起動時 1 回** しか走らない = day1 以降にプール回転で入場する
#      個体(finals 構成で 20.9 万人)は世帯を永久に持てない。
#   ② ``housemates`` / ``household_id`` / ``household_role`` は退避辞書に載っておらず、
#      ``build_pool_agent`` は home_* を **素の record から作り直す** = 再来街のたびに
#      「同居しているという事実」そのものが消える(住居も別の建物へ飛ぶ)。
#   ③ その結果 ``mobility._merge_households`` / ``_split_household`` /
#      ``household.unbond`` が **幽霊の世帯台帳**を書き換えていた。
# 方針(org_id 方式と同型 = 世界の乱数ではなく名簿が世帯を決める):
#   世帯を「pool 名簿の**決定論的な分割**」として定義し、入場のたびに冪等に着席させる。
#   乱数 stream を 1 本も引かない(すべて安定ハッシュ)= 既存 draw 順は 1 粒も動かない。
#   住居は世帯 id から同じ母集団(city.residential_buildings)を決定論で引く
#   (``_assign_household`` が「代表の住居を全員で共有」しているのと同じ意味論を、
#    代表が街に居ない日でも成立する形へ置き換えたもの)。
_HH_PREFIX = "hp"          # プール由来の世帯 id 接頭辞(起動時束ねの "hh" と衝突させない)


def _u01(seed: int, *parts) -> float:
    """安定ハッシュ → [0,1) の一様値(run.seed 非依存・プロセス非依存・乱数 stream ゼロ)。"""
    import hashlib
    key = ":".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2.0 ** 64


def _size_from_u(cfg: dict, u: float) -> int:
    """``_pick_size`` と**同じ重み・同じ CDF 走査**をハッシュ値 u で行う(乱数を引かない)。"""
    sizes = cfg["sizes"]
    weights = (cfg.get("size_weights_realistic") or cfg["size_weights"]) \
        if cfg.get("realistic") else cfg["size_weights"]
    if not sizes:
        return 1
    total = float(sum(weights)) if weights else 0.0
    if total <= 0.0:
        return int(sizes[int(u * len(sizes)) % len(sizes)])
    r = u * total
    acc = 0.0
    for s, w in zip(sizes, weights):
        acc += float(w)
        if r < acc:
            return int(s)
    return int(sizes[-1])


def pool_bind_on(sim) -> bool:
    """プール世帯の再構築が有効か(既定 OFF=完全 no-op=バイト一致)。"""
    cfg = getattr(sim, "householdcfg", None)
    return bool(cfg and cfg["enabled"] and cfg["pool_bind"]["enabled"]
                and getattr(sim, "_pool", None) is not None)


def _pool_partition(sim) -> dict:
    """pool 名簿の居住者を世帯へ分割する索引(遅延構築・1 回だけ・決定論)。

    ``{"of": pid -> 世帯番号, "mem": 世帯番号 -> [pid...]}``。母集団は presence 層が
    ``resident`` の record(= 街に家がある層。来街者は世帯を組まないという既存規約と同じ)。
    列挙順 = シャード順 = ``PoolStore.id_of`` を定める順序そのものなので、
    ``build_households`` の「id 昇順に並べて先頭から詰める」を **プール全体へ**広げた形になる。
    サイズは世帯番号の安定ハッシュから既存の重み表で引く(乱数 stream ゼロ)。"""
    idx = getattr(sim, "_pool_hh_index", None)
    if idx is not None:
        return idx
    cfg = sim.householdcfg
    seed = int(cfg["pool_bind"]["seed"])
    residents = [r.pid for r in sim._pool.presence_records() if r.key == "resident"]
    of: dict[str, int] = {}
    mem: list[list[str]] = []
    i = 0
    while i < len(residents):
        j = len(mem)
        size = _size_from_u(cfg, _u01(seed, "size", j))
        group = residents[i:i + size]
        i += len(group)
        if len(group) < 2:                         # 単身世帯は個人のまま(既存規約と同じ)
            mem.append([])                         # 番号は詰めない(ハッシュ入力の安定性)
            continue
        for pid in group:
            of[pid] = j
        mem.append(group)
    idx = {"of": of, "mem": mem}
    sim._pool_hh_index = idx
    return idx


def _pool_hh_facts(sim, j: int) -> dict:
    """世帯 j の確定事実(住居・続柄・メンバ agent id)を 1 度だけ組んで控える。

    住居は世帯 id の安定ハッシュで ``city.residential_buildings`` から引く(persona の住居割当と
    **同じ母集団**)。続柄は realistic のときだけ、メンバの record(年齢・性別)を読んで
    ``_family_roles`` と同一の規則で決める(record は pool から遅延読み=世帯あたり最大 4 行)。"""
    cache = getattr(sim, "_pool_hh_facts", None)
    if cache is None:
        cache = {}
        sim._pool_hh_facts = cache
    got = cache.get(j)
    if got is not None:
        return got
    cfg = sim.householdcfg
    seed = int(cfg["pool_bind"]["seed"])
    pids = _pool_partition(sim)["mem"][j]
    res = getattr(sim.city, "residential_buildings", None) or []
    bld = None
    if res:
        bld = res[int(_u01(seed, "bld", j) * len(res)) % len(res)]
    levels = max(1, int((bld or {}).get("levels", 1) or 1))
    facts = {
        "hh_id": f"{_HH_PREFIX}{j}",
        "kind": "family" if _u01(seed, "fam", j) < float(cfg["family_ratio"]) else "roommate",
        "ids": {pid: sim._pool.id_of(pid) for pid in pids},
        "building": (bld or {}).get("id", "") if bld else "",
        "node": (bld or {}).get("entrance", "") if bld else "",
        "floor": 1 + int(_u01(seed, "flr", j) * levels) % levels,
        "roles": {},
    }
    if cfg.get("realistic") and facts["kind"] == "family":
        recs = []
        for pid in pids:                           # 年齢・性別は名簿にしかない(1 世帯 = 最大 4 行)
            try:
                r = sim._pool.get(pid)
            except (KeyError, OSError, ValueError):
                r = {}
            recs.append((int(r.get("age", 0) or 0), pid, str(r.get("gender", "") or "")))
        recs.sort(key=lambda t: (t[0], facts["ids"][t[1]]))    # 年齢昇順 → 同齢は id 昇順
        n = len(recs)
        if n == 2:
            roles = [_ROLE_SPOUSE.get(g, "同居人") for _a, _p, g in recs]
        else:
            roles = ["子"] * n
            for k in range(n - 2, n):              # 末尾2 = 年長 = 親
                roles[k] = _ROLE_SPOUSE.get(recs[k][2], "親")
        facts["roles"] = {p: roles[k] for k, (_a, p, _g) in enumerate(recs)}
    elif cfg.get("realistic"):
        facts["roles"] = {pid: "同居人" for pid in pids}
    cache[j] = facts
    return facts


def bind_pool_household(sim, agent, record: dict | None = None) -> None:
    """入場した個体を pool 名簿由来の世帯へ**冪等に**着席させる(build_pool_agent から呼ぶ)。

    既定 OFF は 1 行も通らない(``pool_bind_on`` が False)= home 割当も属性も完全に不変。
    ON でも「居住者(record の presence が resident)かつ 2 人以上の世帯に属する」個体だけが
    対象で、来街者・単身世帯は素通り(= 起動時束ねと同じ規約)。乱数を一切引かない。
    day0 の着席でも日境界の回転でも hydrate 再入でも**同じ世帯・同じ住居**へ着く。"""
    if not pool_bind_on(sim) or getattr(agent, "visitor", False):
        return
    pid = getattr(agent, "pool_pid", None)
    if pid is None:
        return
    j = _pool_partition(sim)["of"].get(pid)
    if j is None:                                  # 単身 or 非居住層 = 世帯を組まない
        return
    f = _pool_hh_facts(sim, j)
    agent.household_id = f["hh_id"]
    agent.household_kind = f["kind"]
    agent.household_role = f["roles"].get(pid, "")
    aid = int(agent.id)
    agent.housemates = sorted(v for k, v in f["ids"].items() if k != pid and int(v) != aid)
    if f["building"]:                              # 世帯が住居を持つ=メンバ全員が同じ家に住む
        agent.home_building = f["building"]
        agent.home_node = f["node"]
        agent.home_floor = int(f["floor"])


def reconcile_partner(sim, agent) -> bool:
    """入場時の**片側パートナーの清算**(レーン丙 5。決定論・冪等・乱数ゼロ)。戻り値 = 消したか。

    塞ぐ穴(レーン乙の発見): :func:`unbond` は「相手が街に居ないときは相手側の
    ``partner_id`` を消せない」(退場者へ書いても ``hydrate`` が捨てる)。その相手が後日
    再来街すると、退避辞書に載っていた ``partner_id`` がそのまま戻り

        A.partner_id = None / B.partner_id = A

    という**片側だけ既婚**の非対称が残る。この状態の B は ``form_partners`` の
    「独身のみ」フィルタに永久に掛からず(= 二度と恋愛できない)、``date_dest`` は
    別れた相手をデート先の基準に使い続け、``mobility.on_breakup`` の同棲解消も届かない。

    やること: 自分の ``partner_id`` が指す相手が**在場していて**、その相手の
    ``partner_id`` が自分でないなら自分の側を外す。相手が街に居ない回は**何もしない**
    (判定材料が無い状態で消すと、単に同じ非対称を反対向きに作るだけになる = 正直な限界)。

    ★呼ぶ位置は :func:`bind_pool_household` の中ではなく **hydrate の後**でなければ
      ならない: ``build_pool_agent``(= 世帯着席)は ``pool.hydrate`` の**前**に走るので、
      その時点の ``partner_id`` はまだ ``None`` に戻っている(``_hh_detached`` の打ち消しが
      hydrate 側に置かれているのと同じ順序の問題)。
    ★``pool_bind`` トグル配下(既定 OFF は 1 行も通らない = 現行のまま)。
    """
    if not pool_bind_on(sim):
        return False
    pid = getattr(agent, "partner_id", None)
    if pid is None:
        return False
    other = sim.present_agent(int(pid))
    if other is None:                              # 相手が街に居ない = 判定できない(保留)
        return False
    op = getattr(other, "partner_id", None)
    if op is not None and int(op) == int(agent.id):
        return False                               # 相互に張れている = 正常
    agent.partner_id = None
    return True


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
            # 現実化 S-R1: 続柄(夫/妻/親/子)があれば具体的な間柄に(無ければ現行文言)。
            role = getattr(meta, "household_role", "")
            parts.append(f"{role}の{meta.name}" if role else f"{label}の{meta.name}")
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


# ---------------------------------------------------------------- 夕食の共食(S-R1)
def _at_home(agent) -> bool:
    """在宅(自宅の建物の中、または自宅前の路上に静止)か(routine._at_home と同定義)。"""
    if getattr(agent, "building", None) is not None:
        return agent.building == agent.home_building
    return not agent.route and agent.node == agent.home_node


def dinner_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """夕食帯に世帯メンバの自由時間行き先を home へ寄せる(共食。date_dest と同型)。決定論・非LLM。

    household OFF / family_dinner OFF / 同居者なし / 帯外 / 抽選外 なら乱数を一切引かず None
    (既定不変)。専用 stream "dinner"(既存 draw 順を汚さない)で prob 抽選。行き先は自宅ノード=
    同一世帯が home へ収束(夜間の家庭内 co-location)。現在地と同じなら None(=到着済み)。"""
    cfg = sim.householdcfg
    fd = cfg["family_dinner"]
    if not cfg["enabled"] or not fd["enabled"]:
        return None
    if not (getattr(agent, "housemates", None) or []):
        return None
    m = sim_min % 1440
    if not (int(fd["band_start_min"]) <= m < int(fd["band_end_min"])):
        return None
    rng = sim.hub.stream("dinner", agent.id, step)
    if rng.random() >= float(fd["prob"]):
        return None
    home = getattr(agent, "home_node", "")
    return home if home and home != agent.node else None


def observe_dinner(sim, step: int, sim_min: int) -> None:
    """毎 step: 夕食帯に同一世帯の2人以上が home で同席した最初の step で joint_activity
    (type="family_dinner")を1件記録(1世帯1日1回)。OFF/帯外/非該当は即 return(=バイト一致)。"""
    cfg = sim.householdcfg
    fd = cfg["family_dinner"]
    if not cfg["enabled"] or not fd["enabled"]:
        return
    m = sim_min % 1440
    if not (int(fd["band_start_min"]) <= m < int(fd["band_end_min"])):
        return
    day = sim_min // 1440
    logged = sim._dinner_logged
    by_hh: dict = {}
    for a in sim.agents:
        hid = getattr(a, "household_id", None)
        if hid is None or a.visitor:
            continue
        by_hh.setdefault(hid, []).append(a)
    for hid, members in by_hh.items():
        if (hid, day) in logged:
            continue
        present = [x for x in members if _at_home(x)]
        if len(present) >= 2:
            logged.add((hid, day))
            leader = min(present, key=lambda x: x.id)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=leader.id,
                                 kind="joint_activity", x=leader.x, y=leader.y,
                                 payload={"type": "family_dinner",
                                          "with": sorted(int(x.id) for x in present),
                                          "place": leader.home_node, "tier": 0}))
            sim._joint_total = getattr(sim, "_joint_total", 0) + 1


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
    # ★レーン乙 ブロック3(c): 在場述語。街に居ない相手の partner_id を消しても次の
    #   hydrate で捨てられる(既存の「相手側の応答は将来拡張」という正直な簡略化と同じ地平)。
    b = sim.present_agent(pid)
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
    # 内部可動性 第60バッチ b: 同棲していたら後から来た側を転出させる(cohabit OFF/非同棲は no-op)。
    from . import mobility as _mobility
    _mobility.on_breakup(sim, a, b, step, sim_min)
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
            # ★相手も**在場者に限る**(``agent_by_id`` は退場者も返す = 幽霊)。退場者を選ぶと
            #   ``b.partner_id = a.id`` が hydrate で捨てられ、**片側だけ既婚**の非対称が残る。
            b = sim.present_agent(oid)
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
