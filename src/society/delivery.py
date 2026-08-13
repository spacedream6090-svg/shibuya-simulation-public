"""宅配・フードデリバリー スライス④(delivery。既定 OFF)。

正典: docs/research/economy-goods-services.md §7 ④(宅配・フードデリバリー=ラストマイル ABM 構造)。
物流①(goods.py=店舗在庫・(s,S)補充)・サービス③(services.py)の上に、買い手が **来店せず注文** →
配達員が **店で受取** → **配送トリップ**(店→注文者)→ **到着で受給+課金** の4段を、最小・非LLM・
決定論(乱数は新 stream "delivery" 1本のみ)で載せる層。§3.3 のラストマイル ABM(注文生成→トリップ→
到着)をそのまま踏襲する。

4段(いずれも決定論・LLM 呼ゼロ・乱数は "delivery" stream のみ):

  1. 注文生成(maybe_order。routine が食事帯に呼ぶ): 在宅(または職場滞在)の agent が食事帯に、
     既存の食事行動(外食)の **代替** として宅配を選ぶ。判定=在宅/職場×食事帯×所持金×近傍に在庫あり
     の食 POI、の観測量のみ+専用 stream "delivery" の1抽選(order_rate)。当たれば **最寄りの在庫あり
     食 POI**(店)を選び、① の実在庫を1単位引き(店で受取=品切れ POI は注文不可)、注文予定を
     sim._delivery_pending へ積み、order を記録して agent は **その step 外食に出ない**(stay を返す=
     既存の食事課金経路と二重にならない)。1日1回上限(現実頻度=月数回/世帯の桁に合わせる)。

  2. 配達員(_select_courier): economy.py の gig/自営「配達員」職の agent に **業務実体** を与える。
     注文時に **最寄りの勤務可能な配達員**(sleeping/範囲外/別配達中を除く=id 昇順タイブレーク)を決定論
     選定して注文へ紐づける。名簿に専任配達員が居なければ **agent_id=-1 の抽象トリップ**(配達員役)へ
     graceful degradation=注文は必ず配送される(配達員収入は発生しない=正直な穴)。

  3. 配送トリップ(dispatch + deliver_arrivals): 到着 eta は距離から決定論(goods.delivery_trip と同型)。
     配達員実体が居れば dispatch がその agent を **物理的に配車**(配達員→店→注文者 の経路を route_to 再利用で
     張り、_delivery_gig を立てて通常 routine に割り込ませない)。到着(arrive_step)で受給を確定=注文者に
     課金(item_price=食事 + fee=配達手数料。_spend 経由=会計は既存経路)+ 配達員に gig 収入
     (_pay_wage source="gig"=既存 wage/gig 会計流儀)。deliver を記録。物理トリップの走行時間と eta は
     厳密一致しない(eta 決定論を正典どおり採り、配達員の物理移動は並行の best-effort 可視化=正直な近似)。

  4. 観測: order{poi, item, fee, cat, to_node}(agent_id=注文者)/ deliver{courier, from_poi, to_node,
     item, fare}(agent_id=配達員 or -1)。schema 登録。aggregate に配達件数列 n_delivery(OFF=None)。
     注文者「頼んだ○○が届いた」/ 配達員「配達で稼いだ」を記憶へ。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。注文・配車・
  配送・会計・メニュー(テキスト)・配達員職名をここ/config に閉じる=engine/cognition/world に地名・因子語・
  商品名を書かない(no-fingerprint 契約に触れない)。① の実在庫は goods の共有台帳(sim._goods_stock)を
  goods の遅延初期化流儀のまま1店だけ引く(goods の挙動を汚さない)。

R1 呼数不変: generate() を1本も足さない。注文は既存の routine 食事分岐に相乗り(新 generate なし)。乱数は
  新 stream "delivery" 1本のみ(既存 draw 順に不干渉)。既定 OFF(delivery.enabled=false)= maybe_order が
  即 None=注文せず・"delivery" stream も引かず・order/deliver とも 0 件・配達員フラグも注文台帳も生えない=
  ゴールデン golden_baseline_l1.json をバイト一致で守る。判定は物理位置・時刻・所持金・config・観測量のみ=
  k・内面構成概念を一切食わせない(注文で在宅に留まる=co-location 変化は services/goods と同型=compute_matched
  下の k 不変性で担保)。新イベント種は order / deliver(schema.py 登録)。
"""
from __future__ import annotations

import hashlib

from .observer.schema import Event

# 食事帯(宅配注文の受付。分 of day)。既定=昼 11:30-13:30 / 夜 18:00-21:00(routine の MEAL_WINDOWS と同帯)。
_DEFAULT_WINDOWS = [[11 * 60 + 30, 13 * 60 + 30], [18 * 60, 21 * 60]]

# 宅配メニュー(① goods カタログが無い/OFF のときの既定。日替わり巡回で決定論に選ぶ)。config で上書き可。
_DEFAULT_MENU = ["弁当", "ラーメン", "カレー", "ピザ", "寿司"]


def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py / goods.py と同流儀=決定論リプレイの土台)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def build_cfg(raw) -> dict:
    """conf の delivery ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(commerce/goods/services と同型)。既定 OFF(enabled=false)で
    maybe_order が即 None=注文せず・"delivery" stream を引かず・order/deliver 0 件=ゴールデンをバイト一致で守る。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    wins = raw.get("windows") or _DEFAULT_WINDOWS
    windows = [[int(w[0]), int(w[1])] for w in wins]
    menu = [str(x) for x in (raw.get("menu") or _DEFAULT_MENU)]
    return {
        "enabled": bool(raw.get("enabled", False)),
        # 食事帯の在宅/職場滞在 step あたりの注文確率(較正: 出前/宅配=月数回/世帯 の桁。§10 較正)。
        "order_rate": float(raw.get("order_rate", 0.02)),
        "windows": windows,                                  # 注文受付=食事帯(分 of day)
        "cat": str(raw.get("cat", "food")),                  # 宅配対象の POI カテゴリ(食=food)
        "radius_m": float(raw.get("radius_m", 1500.0)),      # 注文先(店)を探す近傍半径(m)
        # ---- 手数料(Uber Eats 相当の簡略化=定額+距離比例まで。過剰設計禁止)----
        "base_fee": float(raw.get("base_fee", 350.0)),       # 定額配達手数料(円)
        "per_km_fee": float(raw.get("per_km_fee", 100.0)),   # 距離比例(円/km)
        "courier_share": float(raw.get("courier_share", 0.7)),  # 手数料のうち配達員の取り分(残=プラットフォーム)
        # ---- 配送トリップ eta(距離から決定論。goods.delivery_trip と同型)----
        "eta_m_per_step": float(raw.get("eta_m_per_step", 500.0)),  # 走行速度換算(m/step。1 step=10分)
        "min_eta_steps": max(1, int(raw.get("min_eta_steps", 1))),
        "max_eta_steps": max(1, int(raw.get("max_eta_steps", 6))),
        "courier_occupation": str(raw.get("courier_occupation", "配達員")),  # 配達員職(economy の gig/自営)
        "max_per_day": max(1, int(raw.get("max_per_day", 1))),   # 1日1個体あたりの注文上限(月数回較正)
        "menu": menu,
    }


def enabled(sim) -> bool:
    """宅配④(注文・配車・配送)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "deliverycfg", None)
    return bool(cfg and cfg["enabled"])


# ---------------------------------------------------------------- 幾何・在庫の補助
def _dist_m(sim, na: str, nb: str) -> float:
    ax, ay = sim.city.node_xy(na)
    bx, by = sim.city.node_xy(nb)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _store_in_stock(sim, node: str, cat: str) -> bool:
    """① goods ON なら店の実在庫が1以上か(品切れ=注文先にしない)。goods OFF は常に True(在庫制約なし)。

    読み取り専用(遅延初期化しない=全 POI の在庫台帳を汚さない)。未実体化 POI は満在庫扱い=在庫あり。"""
    from . import goods as _goods
    if not _goods.enabled(sim):
        return True
    cur = sim._goods_stock.get((str(node), str(cat)))
    return cur is None or cur >= 1


def _reserve_stock(sim, node: str, cat: str) -> bool:
    """① goods ON なら注文確定時に店の実在庫を1単位引く(配達員が店で受取)。goods OFF は常に True。

    選定済みの1店だけを goods の遅延初期化流儀(未実体化=上限 S から)で引く=goods.on_purchase と同型・
    goods の挙動を汚さない。在庫切れ(<1)なら False=注文不成立(呼び出し側は order を出さない)。"""
    from . import goods as _goods
    if not _goods.enabled(sim):
        return True
    key = (str(node), str(cat))
    cur = sim._goods_stock.get(key)
    if cur is None:
        cur = _goods._capacity(sim.goodscfg, cat)        # 遅延初期化=購入が起きた店だけ実体化
    if cur < 1:
        sim._goods_stock[key] = cur
        return False
    sim._goods_stock[key] = cur - 1
    return True


def _pick_item(sim, cat: str, node: str, sim_min: int) -> str:
    """注文の品を決定論の日替わり巡回で選ぶ。① goods ON ならそのカタログ(商品実体②)を再利用し、
    無ければ本モジュール/config のメニューから選ぶ(同一 (店, 日) は同じ品=決定論)。"""
    from . import goods as _goods
    if _goods.enabled(sim):
        it = _goods.pick_item(sim.goodscfg, cat, node, sim_min)
        if it is not None:
            return it
    menu = sim.deliverycfg["menu"] or _DEFAULT_MENU
    day = int(sim_min) // 1440
    return menu[(day + _stable_hash(f"{node}/{cat}")) % len(menu)]


def in_window(cfg: dict, sim_min: int) -> bool:
    """宅配注文の受付帯(食事帯)か。"""
    m = int(sim_min) % 1440
    return any(a <= m < b for a, b in cfg["windows"])


def _order_location(agent) -> str | None:
    """在宅(または職場滞在)で停止中なら配送先ノード(home_node / work_node)、そうでなければ None。

    停止中(not route)=腰を据えて頼める状態。路上の自宅前(node 一致)/ 自宅・職場の建物内(building 一致)の
    どちらでも受ける(routine の食事分岐は路上=building 不在で呼ぶが、直接呼び出しにも堅牢に対応)。"""
    if agent.route:
        return None
    home = getattr(agent, "home_node", "") or ""
    if home and (agent.node == home
                 or (agent.building and agent.building == getattr(agent, "home_building", ""))):
        return home
    work = getattr(agent, "work_node", "") or ""
    if work and (agent.node == work
                 or (agent.building and agent.building == getattr(agent, "work_building", ""))):
        return work
    return None


def _nearest_store(sim, agent, cfg: dict):
    """注文者の現在地から半径内で最も近い **在庫ありの食 POI** を返す(決定論・同距離はノード昇順)。無ければ None。"""
    cat = cfg["cat"]
    radius2 = float(cfg["radius_m"]) ** 2
    ax, ay = float(agent.x), float(agent.y)
    best = None
    best_key = None
    for p in sim.city.pois_by_cat(cat):
        node = p.get("node")
        if not node:
            continue
        x, y = sim.city.node_xy(node)
        d2 = (x - ax) ** 2 + (y - ay) ** 2
        if d2 > radius2:
            continue
        if not _store_in_stock(sim, node, cat):          # ① 品切れの店は注文先にしない
            continue
        key = (d2, node)
        if best_key is None or key < best_key:
            best_key = key
            best = (node, str(p.get("name", "")))
    return best


def _fee_and_eta(sim, store_node: str, to_node: str, cfg: dict):
    """配達手数料(定額+距離比例)と配送トリップの eta(距離から決定論)を返す。(fee, eta_steps, dist_m)。"""
    dist = _dist_m(sim, store_node, to_node)
    fee = float(cfg["base_fee"]) + float(cfg["per_km_fee"]) * (dist / 1000.0)
    eta = int(round(dist / max(1.0, float(cfg["eta_m_per_step"]))))
    eta = max(int(cfg["min_eta_steps"]), min(int(cfg["max_eta_steps"]), eta))
    return round(fee, 1), eta, dist


def _select_courier(sim, store_node: str, orderer_id: int, cfg: dict):
    """最寄りの勤務可能な配達員(occupation=courier_occupation)を決定論選定(同距離は id 昇順)。

    sleeping / 範囲外(outside)/ 既に別の配達 gig 中 / 注文者本人 は除外。居なければ None(=抽象トリップへ
    graceful)。距離は店(受取地)との2乗距離。判定は物理位置・occupation・状態のみ=k 非依存。"""
    occ = cfg["courier_occupation"]
    sx, sy = sim.city.node_xy(store_node)
    best = None
    best_key = None
    for a in sim.agents:
        if a.id == orderer_id:
            continue
        if str(getattr(a, "occupation", "")) != occ:
            continue
        if a.sleeping or a.loc == "outside":
            continue
        if getattr(a, "_delivery_gig", None) is not None:  # 既に配達中=二重配車しない
            continue
        d2 = (float(a.x) - sx) ** 2 + (float(a.y) - sy) ** 2
        key = (d2, a.id)
        if best_key is None or key < best_key:
            best_key = key
            best = a
    return best


# ---------------------------------------------------------------- 注文生成(routine が食事帯に呼ぶ)
def maybe_order(agent, sim, step: int, sim_min: int):
    """食事帯の在宅/職場滞在 agent が宅配を注文するか(外食の代替)。注文したら routine アクション
    {"type":"stay"} を返す(= その step 外食に出ない=二重課金しない)。注文しない/OFF は None。

    OFF・食事帯外・非在宅・所持金不足・近傍に在庫ありの店なし・当日既に注文済み では None を返し、かつ
    OFF・非該当では "delivery" stream を一切引かない(バイト一致)。当たり(order_rate)時のみ最寄り在庫店を
    選び、① 実在庫を1引き(配達員が店で受取)、配達員を決定論選定し、eta で到着予定を積み、order を記録する。
    RNG は "delivery" stream の1 draw のみ(既存 draw 順に不干渉)=決定論・k 非依存。"""
    cfg = getattr(sim, "deliverycfg", None)
    if not (cfg and cfg["enabled"]):
        return None
    if getattr(agent, "visitor", False):                 # 来街者は街の外に住居=宅配対象外
        return None
    if not in_window(cfg, sim_min):
        return None
    to_node = _order_location(agent)                     # 在宅/職場滞在で停止中か(=配送先)
    if to_node is None:
        return None
    day = int(sim_min) // 1440
    if int(getattr(agent, "_delivery_day", -1)) == day:  # 1日1回上限(月数回較正)
        return None
    if getattr(agent, "_delivery_await", False):         # 配送中は再注文しない
        return None
    # ---- 抽選(専用 stream "delivery"。既存 draw 順に不干渉)----
    r = float(sim.hub.stream("delivery", agent.id, step).random())
    if r >= float(cfg["order_rate"]):
        return None                                      # この step は注文しない(miss)
    cat = cfg["cat"]
    store = _nearest_store(sim, agent, cfg)               # 最寄りの在庫ありの食 POI
    if store is None:
        return None                                      # 近傍に(在庫ありの)店なし=注文不可
    store_node, store_name = store
    item_price = float(_price_of(sim, cat))
    fee, eta, _dist = _fee_and_eta(sim, store_node, to_node, cfg)
    if float(getattr(agent, "money", 0.0)) < item_price + fee:  # 支払い不能=注文しない(無駄足回避)
        return None
    if not _reserve_stock(sim, store_node, cat):         # ① 品切れ(念のため)=注文不成立
        return None
    item = _pick_item(sim, cat, store_node, sim_min)
    courier = _select_courier(sim, store_node, agent.id, cfg)
    courier_income = round(fee * float(cfg["courier_share"]), 1) if courier is not None else 0.0
    _pending(sim).append({
        "orderer": int(agent.id), "store_node": store_node, "store_name": store_name,
        "cat": cat, "item": item, "to_node": to_node, "arrive": int(step) + int(eta),
        "courier": int(courier.id) if courier is not None else -1,
        "item_price": float(item_price), "fee": float(fee),
        "courier_income": float(courier_income), "dispatched": False,
    })
    agent._delivery_day = day
    agent._delivery_await = True
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="order", x=agent.x, y=agent.y,
                         payload={"poi": store_name or store_node, "item": item,
                                  "fee": round(fee, 1), "cat": str(cat), "to_node": to_node}))
    return {"type": "stay"}                               # 外食に出ない(在宅で受け取る)


def _price_of(sim, cat: str) -> float:
    """宅配の品の価格(=外食と同じ食事価格。会計は既存 economy 価格に閉じる)。"""
    from .economy import price_of
    eco = getattr(sim, "economy", None)
    if not eco:
        return 0.0
    return float(price_of(cat, eco, getattr(sim, "rulebook", None)))


def _pending(sim) -> list:
    """宅配注文の到着待ち台帳(遅延構築)。"""
    p = getattr(sim, "_delivery_pending", None)
    if p is None:
        p = []
        sim._delivery_pending = p
    return p


# ---------------------------------------------------------------- 配達員の gig(routine が最優先で拾う)
def courier_action(agent, sim, step: int, sim_min: int):
    """配達 gig 中の配達員の行動(routine の最上流で拾う)。配送を最優先し通常 routine に割り込ませない。

    OFF・gig 不在では None(=通常 routine へ=バイト一致)。gig 中は経路が残っていれば continue、着いていれば
    stay(到着=eta の deliver_arrivals が gig を解除して復帰させる)。"""
    if not enabled(sim):
        return None
    if getattr(agent, "_delivery_gig", None) is None:
        return None
    return {"type": "continue"} if agent.route else {"type": "stay"}


# ---------------------------------------------------------------- 配車(物理移動)+ 到着処理
def dispatch(sim, step: int, sim_min: int) -> None:
    """配達員実体を物理的に配車する(_phase_delivery が毎 step 呼ぶ)。配達員→店→注文者 の経路を張り、
    _delivery_gig を立てて通常 routine に割り込ませない(route_to 再利用=新移動機構は作らない)。

    抽象トリップ(courier=-1)は物理移動なし。配達員が屋内/睡眠/範囲外で配車できない場合も物理移動は
    見送るが、到着(eta)での受給・課金・収入は必ず成立する(配送は必ず届く)。配車は1注文1回だけ試みる
    (dispatched フラグ)。決定論・乱数なし・LLM 呼ゼロ。"""
    for o in _pending(sim):
        if o["dispatched"]:
            continue
        o["dispatched"] = True
        cid = int(o["courier"])
        if cid < 0:
            continue                                      # 抽象トリップ=物理配車なし
        # ★レーン乙 ブロック7: 在場述語で引く。``agent_by_id`` は「これまで実体化した全個体」の
        #   索引なので ``is None`` は**決して真にならない**(プール回転で街を出た個体も残る)。
        #   脱水済みの実体へ route/gig を書いても次の hydrate で捨てられ、配達員は永久に
        #   幽霊 gig を抱えたまま戻ってくる(_select_courier が二度と選ばない)。
        courier = sim.present_agent(cid)
        if courier is None or courier.sleeping or courier.loc == "outside" or courier.building:
            continue                                      # 配車不能=物理移動なし(受給は eta で成立)
        store_node, to_node = o["store_node"], o["to_node"]
        leg1, _m1 = sim.router.route(courier.node, store_node, "walk")   # 配達員→店(受取)
        route = list(leg1[1:]) if len(leg1) >= 2 else []
        leg2, _m2 = sim.router.route(store_node, to_node, "walk")        # 店→注文者(配送)
        if len(leg2) >= 2:
            route = route + list(leg2[1:])
        if not route:
            continue                                      # 経路なし=物理移動なし(受給は eta で成立)
        courier.route = route                             # 物理移動を開始(_phase_move が進める)
        courier.edge_offset = 0.0
        courier.dest = to_node
        courier.trip_mode = "walk"
        courier.exit_intent = False
        courier.homing = False
        courier._pending_activity = ""                    # 到着時に外食課金等を誘発させない
        courier._pending_stay = 0
        courier._ride_pending = None
        courier._delivery_gig = {"arrive": int(o["arrive"]), "to_node": to_node}


def deliver_arrivals(sim, step: int, sim_min: int, spend, pay_wage) -> None:
    """到着処理(_phase_delivery が毎 step 呼ぶ): eta 到達の注文を配送完了させる。

    到着で: 注文者へ課金(item_price=食事[cat]+ fee=配達手数料[cat=delivery]。_spend 経由=会計は既存経路・
    二重課金なし)+ 配達員へ gig 収入(courier_income=手数料×courier_share。_pay_wage source="gig"=既存 wage/gig
    会計)+ 記憶 + deliver 記録。抽象トリップ(courier=-1)は収入なし(配達員が居ない=正直な穴)。金の帳尻:
    注文者支出(item_price+fee)= 店売上(item_price)+ 配達員収入(courier_income)+ プラットフォーム手数料
    (fee−courier_income)。spend/pay_wage は scheduler の _spend/_pay_wage(engine 側で注入=会計を1箇所に閉じる)。
    決定論・乱数なし・LLM 呼ゼロ。"""
    pend = _pending(sim)
    due = [o for o in pend if int(o["arrive"]) <= step]
    if not due:
        return
    due.sort(key=lambda o: (int(o["arrive"]), int(o["orderer"]), str(o["store_node"])))
    for o in due:
        pend.remove(o)
        # ★レーン乙 ブロック7(**保存則系・最優先**): 在場述語で引く。脱水済みの注文者へ
        #   spend すると「家計の現金は次の hydrate で戻るのに店の売上・SFC 行は実在する」
        #   = 現金が湧く。配達員側の pay_wage は逆に現金が消える。どちらも保存則の破れ。
        #   注文は最大 max_eta_steps 先に着地するので日境界(=回転)を跨ぐことが実際にある。
        orderer = sim.present_agent(int(o["orderer"]))
        if orderer is None:                               # 注文者が街に居ない=課金せず破棄
            continue
        cat = str(o["cat"])
        spend(sim, orderer, float(o["item_price"]), cat, step, sim_min, item=str(o["item"]))  # 食事
        spend(sim, orderer, float(o["fee"]), "delivery", step, sim_min)                        # 配達手数料
        orderer._delivery_await = False
        orderer.remember(f"頼んだ{o['item']}が届いた")
        cid = int(o["courier"])
        courier = sim.present_agent(cid) if cid >= 0 else None
        fare = 0.0
        cx, cy = orderer.x, orderer.y
        if courier is not None:
            fare = float(o["courier_income"])
            if fare > 0.0:
                pay_wage(sim, courier, fare, step, sim_min, source="gig")   # 配達員=gig 収入
            courier.remember("配達で稼いだ")
            courier._delivery_gig = None                  # gig 終了=通常の routine へ復帰
            cx, cy = courier.x, courier.y
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=cid, kind="deliver",
                             x=cx, y=cy,
                             payload={"courier": cid, "from_poi": o["store_name"] or o["store_node"],
                                      "to_node": o["to_node"], "item": str(o["item"]),
                                      "fare": round(fare, 1), "cat": cat}))
        sim._delivery_total = int(getattr(sim, "_delivery_total", 0)) + 1


def tick(sim, step: int, sim_min: int, spend, pay_wage) -> None:
    """宅配④の毎 step 更新(既定 OFF=即 return=バイト一致)。物理配車(dispatch)+ 到着処理。

    _phase_move の前に呼ぶ(配車した配達員をその step のうちに動かす)。決定論・乱数なし=新 stream を引かない。"""
    if not enabled(sim):
        return
    dispatch(sim, step, sim_min)
    deliver_arrivals(sim, step, sim_min, spend, pay_wage)
