"""サービスの実体化 スライス③(理美容/クリニック/塾/ジム/クリーニング等。既定 OFF)。

正典: docs/research/economy-goods-services.md §7 ③(サービスの実体=滞在+効果+予約)。現状は
service/education カテゴリの POI が経済的に眠っている(§2.4)。ここに「滞在して効果を受ける」最小形を
非LLM・決定論(+隔離 stream)で載せる層。物流①②(goods.py)と対になる需要側の体験実体。

3要素(いずれも決定論・LLM 呼ゼロ・乱数は新 stream "service" のみ):

  1. サービス定義テーブル(データ駆動=本モジュール既定 + config で上書き): 理美容/クリニック(任意
     受診=既存 medical_visit と非重複=病気起点でない健康維持)/塾・習い事/ジム/クリーニング/汎用対人
     サービス。各定義={cats(ホスト POI カテゴリ), hints(POI 名の部分一致=空なら当該カテゴリの任意 POI),
     price, stay, band(効果の帯), magnitude(効果の不透明量), daily_rate(日次利用率=現実頻度に桁を合わせた
     較正), age(該当年齢帯), remember(記憶1行)}。

  2. 来店発火(service_dest): 自由時間の行き先バイアス(routine の dest チェーンに中立1分岐=date/joint と
     同型)。営業時間・年齢帯・所持金でゲートし、専用 stream "service"(既存 draw 順に不干渉)から利用率で
     抽選。当たれば近傍の該当 POI ノードへ寄せ、来店予定 (key, node) を本人へ退避する。判定は物理位置・時刻・
     config・観測量(所持金・年齢)のみ=k・内面構成概念を一切食わせない(co-location 変化=career/health/joint
     と同型で許容=compute_matched 下の k 不変性で担保)。

  3. 受給(charge_service は scheduler が到着時に呼ぶ): 到着で _spend 課金(会計不変)+ 効果=factors への
     不透明 magnitude hook(on_service。散髪=気分/ジム=活力/塾=学習素地。既存 on_work_done/on_scarcity と
     同型。発火系 drive には接続しない)+ 本人の記憶1行 + service_use を記録。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  サービス定義・効果テーブル・較正値をここ(本モジュール既定)/config に閉じる=engine/cognition/world に
  地名・因子語・サービス名を書かない(no-fingerprint 契約に触れない)。効果は factors 層 hook(on_service)へ
  band と不透明 magnitude だけを渡す(drive=発火系には接続しない)。

★ 供給側 serve(work.service=L2 業務実体)との seam: 受給の _spend は cat=service_cat(既定 "service")で出る。
  work.serve_by_cat に "service" を写像すれば、同一 work_node の勤務中スタッフに serve が帰属し、需要側
  service_use と供給側 serve が同一接点で対になる(本モジュールは供給側を作らない=重複しない)。

R1 呼数不変: generate() を1本も足さない。来店は既存の routine 発火に相乗り(新 generate なし)。乱数は新
  stream "service" 1本のみ(既存 draw 順に不干渉)。既定 OFF(services.enabled=false)= service_dest が即 None
  =来店せず・"service" stream も引かず・service_use/state_update(効果)とも 0 件=ゴールデン
  golden_baseline_l1.json をバイト一致で守る。新イベント種は service_use(schema.py 登録)。
"""
from __future__ import annotations

from .observer.schema import Event

# サービス定義テーブル(config 未指定時の既定。値は data-driven=config で上書き可)。
#   cats     : ホストとなる POI カテゴリ(service/education/leisure)。
#   hints    : POI 名の部分一致ヒント(空=そのカテゴリの任意 POI に一致=汎用フォールバック)。
#   price    : 1回あたりの料金(円)。stay: 滞在 step 数(1 step=10分)。
#   band     : 効果の帯(mood=気分/vitality=活力/learning=学習)= factors.on_service の対象 state を決める。
#   magnitude: 効果の不透明 magnitude(band で state が決まる。0.0=効果なし=観測のみ)。
#   daily_rate: 1日あたりの利用率(現実頻度に桁を合わせた較正。下記 free_steps_ref で per-step 化)。
#   age      : [min, max] 該当年齢帯(この範囲の個体のみ来店)。
#   remember : 受給時に本人の記憶へ残す1行(客観記述)。
# 較正根拠(§3.4/現実の生活頻度): 理美容=月1-2回≈0.05/日、クリーニング=週1回≈0.14/日、ジム=週1-2回≈0.2/日、
#   塾・習い事=週1-2回(該当層)≈0.2/日、任意受診(健診)=年数回≈0.012/日。桁を合わせた初期値(B段で調律)。
_DEFAULT_SERVICES: dict[str, dict] = {
    # 理美容(散髪・美容): 気分/外見が上がる。月1-2回。
    "grooming": {"cats": ["service"], "hints": ["美容", "理容", "床屋", "サロン", "ヘア", "ネイル"],
                 "price": 4000, "stay": 3, "band": "mood", "magnitude": 0.03,
                 "daily_rate": 0.05, "age": [12, 90], "remember": "髪を整えて気分が上がった"},
    # 任意受診・健診(病気起点でない=既存 medical_visit と非重複): 体調管理=活力。年数回。
    #   ★exclude_subcats(H2 レーン 2026-08-10 の是正): 地図 v8 は総合病院を subcat=hospital で
    #     区別しているが cat は service に潰れているため、名前ヒントだけで引くと**総合病院が
    #     健診の行き先として選ばれていた**(v8 の hospital 7 件には名称が「〜クリニック」の
    #     施設が実在する)。病院は救急の**搬送先**(src/society/medical.py)であって歩いて
    #     行く場所ではないので、ここから除く。★subcat を持たない地図(v7 以前)では
    #     何も落ちない=完全同値(lodging の love_hotel フィルタと同型)。
    "clinic": {"cats": ["service"], "hints": ["クリニック", "医院", "診療", "歯科", "内科", "眼科"],
               "exclude_subcats": ["hospital"],
               "price": 3000, "stay": 2, "band": "vitality", "magnitude": 0.02,
               "daily_rate": 0.012, "age": [0, 120], "remember": "健康のために受診してきた"},
    # 塾・習い事・スクール: 学習素地=効力感。該当層(若年)。週1-2回。
    #   dev_axis/dev_gain(T4 自助努力): 反復通学で技能ストック skill が逓減つき累積する(唯一の経済経路)。
    "lesson": {"cats": ["education", "service"], "hints": ["塾", "スクール", "教室", "予備校", "アカデミー"],
               "price": 5000, "stay": 3, "band": "learning", "magnitude": 0.02,
               "daily_rate": 0.20, "age": [8, 35], "remember": "習い事で新しいことを学んだ",
               "dev_axis": "skill", "dev_gain": 0.08, "dev_remember": "学びの積み重ねが力になってきた"},
    # ジム・フィットネス: 疲労回復・活力。週1-2回。
    #   dev_axis/dev_gain(T4 自助努力): 反復通いで体力ストック fitness が累積する(経済経路は持たない=観測のみ)。
    "gym": {"cats": ["service", "leisure"], "hints": ["ジム", "フィットネス", "スポーツ", "ヨガ", "プール"],
            "price": 2000, "stay": 4, "band": "vitality", "magnitude": 0.025,
            "daily_rate": 0.20, "age": [15, 75], "remember": "ジムで汗を流してすっきりした",
            "dev_axis": "fitness", "dev_gain": 0.08, "dev_remember": "ジム通いの成果を感じる"},
    # クリーニング・ランドリー: 生活を整える=気分。週1回。
    "laundry": {"cats": ["service"], "hints": ["クリーニング", "ランドリー", "洗濯"],
                "price": 1200, "stay": 1, "band": "mood", "magnitude": 0.015,
                "daily_rate": 0.14, "age": [18, 90], "remember": "衣類を整えてもらった"},
    # 汎用の対人サービス(名ヒント無し=当該カテゴリの任意 service POI に一致=器を起こすフォールバック)。
    "service": {"cats": ["service"], "hints": [], "price": 2000, "stay": 2,
                "band": "mood", "magnitude": 0.015, "daily_rate": 0.03, "age": [12, 100],
                "remember": "用事を済ませてきた"},
}

_SERVICE_ORDER = tuple(sorted(_DEFAULT_SERVICES))   # 決定論の巡回/選択順(キー昇順)


def build_cfg(raw) -> dict:
    """conf の services ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(commerce/goods/work と同型)。既定 OFF(enabled=false)で
    service_dest が即 None=来店せず・"service" stream を引かず・service_use 0 件=ゴールデンをバイト一致で守る。
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    src = dict(raw.get("services", {}) or {})
    services: dict[str, dict] = {}
    base = {str(k): dict(v) for k, v in _DEFAULT_SERVICES.items()}
    for k, v in (src or base).items():          # config があればそれを、無ければ既定を採る
        d = dict(v)
        age = d.get("age", [0, 120]) or [0, 120]
        services[str(k)] = {
            "cats": [str(c) for c in (d.get("cats") or [])],
            "hints": [str(h) for h in (d.get("hints") or [])],
            # 除外するサブカテゴリ(既定 空=何も落とさない=従来と完全同値)。
            "exclude_subcats": [str(s) for s in (d.get("exclude_subcats") or [])],
            "price": float(d.get("price", 0.0)),
            "stay": max(1, int(d.get("stay", 2))),
            "band": str(d.get("band", "mood")),
            "magnitude": float(d.get("magnitude", 0.0)),
            "daily_rate": float(d.get("daily_rate", 0.0)),
            "age": [int(age[0]), int(age[1])],
            "remember": str(d.get("remember", "サービスを受けた")),
            # T4 自助努力(第52バッチ): 反復利用の累積軸。dev_axis="" or dev_gain<=0 なら累積なし
            #   (単発受給のみ=既存挙動)。self_dev.enabled=false ではそもそも読まれない。
            "dev_axis": str(d.get("dev_axis", "") or ""),
            "dev_gain": float(d.get("dev_gain", 0.0)),
            "dev_remember": str(d.get("dev_remember", "") or ""),
        }
    sd = dict(raw.get("self_dev", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "services": services,
        "spend_cat": str(raw.get("spend_cat", "service")),   # _spend の cat(work.serve_by_cat と対の seam)
        "open_from": int(raw.get("open_from", 9)),           # サービス営業帯(時)
        "open_to": int(raw.get("open_to", 20)),
        "radius_m": float(raw.get("radius_m", 700.0)),       # 来店先を探す近傍半径(m)
        "free_steps_ref": max(1, int(raw.get("free_steps_ref", 48))),  # 日次利用率→per-step 化の基準自由 step 数
        # ---- 自助努力 affordance(T4 第52バッチ)。すべて既定 OFF/中立=会計もイベントも不変(バイト一致)----
        #  enabled: 累積(accrue)・賃金乗数・日次減衰を有効化する主スイッチ(false=誰も self_dev を書かない)。
        #  wage_coef: 賃金乗数 =1 + wage_coef × skill の係数(既定 0.0=ON でも会計不変。較正は conf)。
        #  decay: skill/fitness の日次乗算減衰(既定 0.0=減衰なし。>0 で「維持しないと薄れる」有界均衡)。
        #  remember_threshold: 累積軸がこの値を初めて越えた時だけ記憶を1行残す(毎回は騒音)。
        "self_dev": {
            "enabled": bool(sd.get("enabled", False)),
            "wage_coef": float(sd.get("wage_coef", 0.0)),
            "decay": float(sd.get("decay", 0.0)),
            "remember_threshold": float(sd.get("remember_threshold", 0.5)),
        },
    }


def enabled(sim) -> bool:
    """サービスの実体(来店+効果)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "servicescfg", None)
    return bool(cfg and cfg["enabled"])


# ================================================================ 自助努力 affordance(T4 第52バッチ)
# 「自力で成長・改善できる」可能性を用意する層。成長は強制せず選べる状態にする(affordance)。既存の
# サービス受給(charge_service)に、反復利用の**累積効果**(agent.self_dev)と、経済成果への**1本だけの
# 間接経路**(skill → 賃金乗数)を足す。既存の on_service 単発効果は不変(独立)。
#
# R1 の要点:
#  - 発火判定に累積状態を読ませない。経路は経済成果(賃金)のみ。money_pressure 等の心理は money を見る
#    ので self_dev→wage→money→(既存 economy↔psych seam) が唯一の間接経路=「1本だけ」。
#  - 累積(accrue)・減衰(daily)は決定論=RNG を1つも引かない(新 stream も足さない)。
#  - self_dev.enabled=false / wage_coef=0.0 / decay=0.0 は完全 no-op(state もイベントも会計も不変=バイト一致)。
#  - 累積ロジックは本モジュール(src/society 直下=CHECKED_DIRS 外)に閉じる。engine/factors には既存の
#    hook 流儀以外を足さない(no-fingerprint)。skill/fitness は経験由来のストックで k(構成概念)ではない。

_WAGE_AXIS = "skill"   # 賃金へ結ぶ唯一の累積軸(fitness 等は累積・観測するが経済経路を持たない=「1本だけ」)


def accrue_self_dev(sim, agent, key: str, step: int, sim_min: int):
    """サービス反復受給の累積効果(T4)。サービス定義の dev_axis 軸へ逓減つきで加算する。

    戻り値 = (axis, 累積後の値) or None(self_dev OFF / このサービスに dev_axis 無し / dev_gain<=0)。
    累積式 = old + dev_gain/(1+old)。増分は現在値に反比例=単調増だが増分は 0 へ漸近する「飽和形」。
    根拠: 練習の冪乗則/学習曲線(熟達が高いほど1回の伸びは小さい)。日次 decay(既定 0)と併せると
    有界な均衡へ収束=無限成長を防ぐ。決定論(RNG ゼロ)・既存 on_service 単発効果からは独立。
    remember は累積軸が remember_threshold を**初めて越えた時だけ**1行(毎回は騒音)。"""
    cfg = getattr(sim, "servicescfg", None)
    sd = cfg.get("self_dev") if cfg else None
    if not (sd and sd["enabled"]):
        return None
    svc = cfg["services"].get(key)
    if svc is None:
        return None
    axis = svc.get("dev_axis") or ""
    gain = float(svc.get("dev_gain", 0.0))
    if not axis or gain <= 0.0:
        return None
    store = getattr(agent, "self_dev", None)
    if store is None:                                # 旧 checkpoint 等で欠落していても安全に起こす
        store = {}
        agent.self_dev = store
    old = float(store.get(axis, 0.0))
    new = old + gain / (1.0 + old)                   # 逓減(飽和形)。無限成長を防ぐ
    store[axis] = new
    thr = float(sd["remember_threshold"])
    if thr > 0.0 and old < thr <= new:               # 閾値の初到達だけ記憶(反復の毎回は残さない)
        agent.remember(str(svc.get("dev_remember", "") or "続けてきた努力が実を結びはじめた"))
    return (axis, new)


def self_dev_daily(sim, step: int, sim_min: int) -> None:
    """自助努力ストックの日次自然減衰(T4)。維持しないと薄れる(技能の忘却・detraining)。

    乗算減衰 x*=(1-decay): 蓄積 gain と釣り合う有界均衡へ収束し、負値化しない。既定 decay=0.0 /
    self_dev.enabled=false は完全 no-op(state を触らず・イベントも出さない=バイト一致)。決定論(RNG ゼロ)。
    減衰は静かに(イベント無し)—累積の観測は受給時の service_use payload と L2 集計が担う。"""
    cfg = getattr(sim, "servicescfg", None)
    sd = cfg.get("self_dev") if cfg else None
    if not (sd and sd["enabled"]):
        return
    decay = float(sd["decay"])
    if decay <= 0.0:
        return
    keep = 1.0 - decay
    for agent in sim.agents:
        store = getattr(agent, "self_dev", None)
        if not store:
            continue
        for axis in list(store):
            store[axis] = float(store[axis]) * keep


def self_dev_wage_mult(sim, agent) -> float:
    """自助努力(skill)による賃金乗数 = 1 + wage_coef × skill(T4 経済成果への1本だけの経路)。

    scheduler の賃金支給点(_pay_wage)が1箇所だけ呼ぶ。self_dev OFF / wage_coef=0.0 は 1.0(会計不変=
    バイト一致)。skill=累積した技能ストック(経験由来・traits 非依存)で、これは**経済成果**の乗数であり
    発火判定には一切使われない(R1)。読むのは _WAGE_AXIS="skill" のみ(fitness 等は賃金へ結ばない)。"""
    cfg = getattr(sim, "servicescfg", None)
    sd = cfg.get("self_dev") if cfg else None
    if not (sd and sd["enabled"]):
        return 1.0
    coef = float(sd["wage_coef"])
    if coef == 0.0:
        return 1.0
    skill = float((getattr(agent, "self_dev", None) or {}).get(_WAGE_AXIS, 0.0))
    return 1.0 + coef * skill


# ---------------------------------------------------------------- POI 索引(遅延・軽量)
def _index(sim) -> dict:
    """関係カテゴリの service/education POI を cat -> [(node, name, x, y)] で遅延構築(1回)。

    決定論・乱数なし。全 POI を1度だけ走査(有界)。サービス定義に現れるカテゴリのみ索引する。"""
    idx = getattr(sim, "_service_index", None)
    if idx is not None:
        return idx
    cats: set[str] = set()
    for svc in sim.servicescfg["services"].values():
        cats.update(svc["cats"])
    idx: dict = {}
    for c in cats:
        entries = []
        for p in sim.city.pois_by_cat(c):
            node = p.get("node")
            if not node:
                continue
            x, y = sim.city.node_xy(node)
            # subcat は地図 v8 以降だけが持つ(v7 以前は常に "" = 除外が何も落とさない)。
            entries.append((node, str(p.get("name", "")), float(x), float(y),
                            str(p.get("subcat") or "")))
        idx[c] = entries
    sim._service_index = idx
    return idx


def _matches(svc: dict, name: str, subcat: str = "") -> bool:
    """POI 名 name がサービス定義に該当するか(hints 空=汎用=常に一致)。

    ★exclude_subcats に載るサブカテゴリは**先に落とす**(既定 空=何も落とさない=従来同値)。
    """
    if subcat and subcat in (svc.get("exclude_subcats") or ()):
        return False
    hints = svc["hints"]
    if not hints:
        return True
    return any(h in name for h in hints)


def _nearest_poi(sim, agent, svc: dict, radius2: float):
    """agent の現在地から半径内で最も近い該当 POI ノードを返す(決定論・同距離はノード昇順)。無ければ None。"""
    idx = _index(sim)
    ax, ay = float(agent.x), float(agent.y)
    best_node = None
    best_key = None
    for cat in svc["cats"]:
        for node, name, x, y, subcat in idx.get(cat, ()):
            if not _matches(svc, name, subcat):
                continue
            d2 = (x - ax) ** 2 + (y - ay) ** 2
            if d2 > radius2:
                continue
            key = (d2, node)
            if best_key is None or key < best_key:
                best_key = key
                best_node = node
    return best_node


def _applicable(sim, agent) -> list:
    """この agent が来店しうるサービス [(key, svc, node, per_step_prob)] を列挙(年齢・所持金・近傍でゲート)。

    決定論・乱数なし(抽選は service_dest が1 draw で行う)。age 帯外/近傍に POI 無し/所持金不足は除外。"""
    cfg = sim.servicescfg
    age = int(getattr(agent, "age", 30) or 30)
    money = float(getattr(agent, "money", 0.0))
    radius2 = float(cfg["radius_m"]) ** 2
    ref = float(cfg["free_steps_ref"])
    out = []
    for key in _SERVICE_ORDER if set(_SERVICE_ORDER) <= set(cfg["services"]) else sorted(cfg["services"]):
        svc = cfg["services"].get(key)
        if svc is None:
            continue
        lo, hi = svc["age"]
        if not (lo <= age <= hi):
            continue
        if svc["price"] > money:                 # 所持金不足=来店しない(無駄足の回避)
            continue
        rate = svc["daily_rate"]
        if rate <= 0.0:
            continue
        node = _nearest_poi(sim, agent, svc, radius2)
        if node is None or node == agent.node:   # 近傍に無い/既に同ノード=この step は対象外
            continue
        out.append((key, svc, node, rate / ref))
    return out


def service_dest(agent, sim, step: int, sim_min: int):
    """自由時間の来店バイアス(routine の dest チェーンの中立1分岐=date/joint と同型)。

    戻り値=来店先ノード(または None=来店せず=以降は従来通り)。OFF/営業時間外/該当なしは None を返し
    "service" stream を一切引かない(バイト一致)。ON かつ候補ありなら stream "service" を1 draw:
    総来店確率(候補の per-step 確率の総和)を下回れば利用率で重み付けしたサービスを1つ選び、来店予定
    (key, node)を本人へ退避して node を返す。判定は年齢・所持金・時刻・物理位置・config のみ=k 非依存。"""
    cfg = getattr(sim, "servicescfg", None)
    if not (cfg and cfg["enabled"]):
        return None
    m = int(sim_min) % 1440
    if not (int(cfg["open_from"]) * 60 <= m < int(cfg["open_to"]) * 60):
        return None
    cand = _applicable(sim, agent)
    if not cand:
        return None
    total = sum(p for (_k, _s, _n, p) in cand)
    if total <= 0.0:
        return None
    r = float(sim.hub.stream("service", agent.id, step).random())   # 専用 stream(既存 draw 順に不干渉)
    if r >= total:
        return None                                  # 来店せず(この step は miss)
    acc = 0.0
    for key, svc, node, p in cand:                   # r∈[0,total) を利用率で重み付け選択(1 draw)
        acc += p
        if r < acc:
            agent._service_pending = (key, node)     # 到着時に charge_service が読む
            agent._service_stay = int(svc["stay"])
            return node
    return None


# ---------------------------------------------------------------- 受給(到着時。scheduler が呼ぶ)
def _poi_name(sim, node: str, svc: dict) -> str | None:
    """node 上でサービス定義に一致する POI 名(観測用。無ければ None)。"""
    for p in sim.city.pois_at_node(node):
        if p.get("cat") in svc["cats"] and _matches(svc, str(p.get("name", "")),
                                                    str(p.get("subcat") or "")):
            return p.get("name")
    return None


def charge_service(sim, agent, step: int, sim_min: int, spend) -> bool:
    """来店先へ到着したサービス受給: 課金(spend 経由)+ 効果(factors on_service)+ 記憶 + service_use。

    scheduler が到着時(agent.activity=="service")に呼ぶ。予定 (key, node) が現在ノードと一致し所持金が
    足りるときだけ成立(不一致/不足は no-op で予定をクリア)。spend は scheduler._spend(会計は既存経路に
    閉じる)。効果は不透明 magnitude を band つきで on_service へ渡す(drive=発火系には接続しない=R1)。
    戻り値=受給が成立したか。RNG は一切引かない(来店抽選は service_dest で済み)=決定論。"""
    pend = getattr(agent, "_service_pending", None)
    agent._service_pending = None
    if pend is None:
        return False
    key, node = pend
    if agent.node != node:                           # 予定ノードでない(寄り道/中断等)=受給しない
        return False
    svc = sim.servicescfg["services"].get(key)
    if svc is None:
        return False
    price = float(svc["price"])
    if float(getattr(agent, "money", 0.0)) < price:  # 到着時点で不足=受給しない
        return False
    spend(sim, agent, price, sim.servicescfg["spend_cat"], step, sim_min)   # 課金(会計不変)
    apply_effect(sim, agent, key, step, sim_min)     # 効果(不透明 magnitude・drive 非接続)
    agent.remember(str(svc["remember"]))
    dev = accrue_self_dev(sim, agent, key, step, sim_min)   # T4 自助努力: 反復利用の累積(既定 OFF=None)
    payload = {"node": node, "service": key,
               "cost": round(price, 1),
               "poi": _poi_name(sim, node, svc) or node}
    if dev is not None:                              # 累積後の軸値を観測に追記(既存キー不変・追加のみ)
        payload["dev_axis"] = dev[0]
        payload["dev_value"] = round(dev[1], 4)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="service_use", x=agent.x, y=agent.y,
                         payload=payload))
    sim._service_total = int(getattr(sim, "_service_total", 0)) + 1
    return True


def apply_effect(sim, agent, key: str, step: int, sim_min: int) -> float:
    """サービスの効果を factors 層 hook(on_service)へ band と不透明 magnitude で渡す。

    band(mood/vitality/learning)は本モジュール/config が持つ帯ラベルにすぎず、対象 state の選択は factors が
    行う(no-fingerprint: society 側は state 名を書かない)。magnitude=0.0 は完全 no-op。発火系 drive には
    接続しない(呼び出し側は返り値で drive.add しない)=R1: 効果は generate 呼数を1本も動かさない。"""
    svc = sim.servicescfg["services"].get(key)
    if svc is None or float(svc["magnitude"]) == 0.0:
        return 0.0
    from .factors import update as factor_update
    return factor_update.on_service(agent, svc["band"], float(svc["magnitude"]),
                                    step=step, sim_min=sim_min, logger=sim.logger)
