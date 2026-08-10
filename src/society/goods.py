"""物流の実体化 スライス①+②(店舗在庫・日次補充トリップ・商品実体。既定 OFF)。

正典: docs/research/economy-goods-services.md §7 ①(店舗在庫+日次補充=物流トリップ)+
②(商品実体=何を買ったか)。現状は「金は動くが物の実体が無い」(spend は残高移転のみ・
commerce.stock_out は在館数=混雑の代理であって実在庫ではない)。ここに **実在庫**(場所に紐づく
カテゴリ別在庫数)と **(s,S) 補充方策の日次物流トリップ**、および **買った物(商品実体)** を、
最小・非LLM・決定論で載せる層。

3機構(いずれも決定論・LLM 呼ゼロ・乱数ゼロ=新 stream 不要):

  1. 店舗在庫: 消費が起きる POI(node×カテゴリ)に在庫を持たせる(遅延初期化=購入が起きた
     POI だけ在庫が実体化=有界)。購入(_charge_meal / 建物内消費)で consume_units だけ減らし、
     在庫が尽きたら **購入不成立**(spend を出さない=実在庫由来の品切れ体験)。品切れは既存 kind
     `stock_out`(意味を「在館数の代理」から「実在庫の枯渇」へ拡張)+ 不満(factors on_scarcity)。

  2. 日次補充トリップ: 日次レビューで在庫が発注点 s(reorder_point)以下の POI へ、最寄り
     ゲートウェイ(街の外の物流拠点=depot)から **配送トリップ**(delivery_trip)を発つ。
     lead_time_steps 後に **到着で在庫を上限 S(capacity=order-up-to)へ回復**(restock)。
     経路が封鎖(災害の運休 sim.transit.suspended / 摂動シナリオ shock_closure)されていると
     **補充失敗=在庫回復なし**(発注は落とし、翌レビューで再発注=欠品が波及する構造)。

  3. 商品実体: 購入時に「何を買ったか」(カテゴリ→具体商品カタログ=config 由来テキスト)を
     決定論の日替わり巡回で選び、spend payload に `item` として載せる(会計不変=金額は変えない)。
     買った物は本人の所持(直近購入の有界リスト)+ 行間ダイジェスト(S2)に乗る。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  在庫・(s,S)補充・配送・商品カタログ(テキスト)をここに閉じる=engine/cognition/world に
  地名・因子語・商品名を書かない(no-fingerprint 契約に触れない)。品切れ→grievance は factors 層
  hook(on_scarcity)へ **不透明な magnitude** だけを渡す(drive=発火系には接続しない)。

R1 呼数不変: generate() を1本も足さない。在庫 decrement・(s,S)判定・配送トリップ・到着回復・
  商品選択は **すべて決定論**(乱数を一切引かない=新 stream 不要=既存 draw 順に不干渉)。発火・在庫
  判定に k・内面状態(構成概念)を一切食わせず、在庫量・時刻(sim_min)・config・物理位置のみ参照する。
  品切れは spend を抑制するだけで移動体・co-location を変えない(補充トリップは agent_id=-1 の
  世界イベント=個体の位置を動かさない)=compute_matched 下の k 不変性で呼数一致を担保する。
  ★Wave 4 III-4(city_ops)ON のときだけ、delivery_trip の agent_id が **当直の納品ドライバー**に
  なる(payload に driver / 不在なら unstaffed を足す)。それでも本 module は**誰の位置も動かさない**
  (ドライバーの持ち場は city_ops.bind が起動時に与えるもので、ここでは帰属を記録するだけ)。
  city_ops OFF では分岐をどちらも通らず agent_id=-1 と payload 5 キーが従来と完全同一。

既定 OFF(commerce.inventory.enabled=false)= 在庫実体なし・品切れなし・delivery_trip/restock/
  stock_low/stock_out(実在庫版)とも 0 件・所持/ダイジェストも生えない・乱数消費不変(ゴールデン
  golden_baseline_l1.json を守る)。新イベント種は delivery_trip / restock / stock_low(schema.py 登録)。
  stock_out は commerce 既存 kind の意味拡張として再利用する。
"""
from __future__ import annotations

import hashlib

from . import city_ops as _city_ops
from .observer.schema import Event

# カテゴリ別の既定 (s,S) 較正(正典 §3.4: 生鮮 food は回転が速い=在庫厚め・発注点高め、日用 shop は
# 日1回相当)。ここに無いカテゴリは default_* を使う。値は config で上書き可(データ駆動)。
_DEFAULT_CAPACITY = {"food": 40, "cafe": 40, "shop": 30, "nightlife": 30}
_DEFAULT_REORDER = {"food": 12, "cafe": 12, "shop": 8, "nightlife": 8}

# 商品カタログ(カテゴリ→具体商品テキスト。config 未指定時の既定。家計調査費目と整合する粒度)。
# ★このテキストは config 由来に閉じる(engine には書かない)。日替わり巡回で決定論に選ぶ。
_DEFAULT_CATALOG = {
    "food": ["ラーメン", "定食", "カレー", "寿司"],
    "cafe": ["コーヒー", "紅茶", "ケーキ", "サンドイッチ"],
    "shop": ["Tシャツ", "雑貨", "文房具", "日用品"],
    "nightlife": ["カクテル", "ビール", "ハイボール"],
}

# 所持/ダイジェストで「たくさん買った」と言い換える件数の閾値(客観記述の粒度)。
_MANY_THRESHOLD = 6


def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py と同流儀=決定論リプレイの土台)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def build_cfg(raw_commerce) -> dict:
    """conf の commerce.inventory ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(commerce/disaster と同型)。commerce ブロック全体を受け取り、
    その下の `inventory` サブブロックを読む(トップレベル重複を避け、conf は in-place)。すべて既定 OFF
    (enabled=false)で在庫実体・品切れ・補充トリップ・商品実体が完全 no-op(イベント 0 件・所持も
    生えない=ゴールデンを守る)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw_commerce):
        raw_commerce = OmegaConf.to_container(raw_commerce, resolve=True)
    rc = dict(raw_commerce or {})
    raw = dict(rc.get("inventory", {}) or {})

    def _int_map(key, default_map):
        src = raw.get(key)
        m = dict(src) if src else dict(default_map)
        return {str(k): int(v) for k, v in m.items()}

    def _cat_map(key, default_map):
        src = raw.get(key)
        m = dict(src) if src else dict(default_map)
        return {str(k): [str(x) for x in (v or [])] for k, v in m.items()}

    return {
        "enabled": bool(raw.get("enabled", False)),
        # ---- (s,S) 在庫方策 ----
        "capacity": _int_map("capacity", _DEFAULT_CAPACITY),          # 上限 S = 初期在庫 = order-up-to
        "reorder_point": _int_map("reorder_point", _DEFAULT_REORDER),  # 下限 s = 発注点
        "default_capacity": int(raw.get("default_capacity", 30)),
        "default_reorder_point": int(raw.get("default_reorder_point", 8)),
        "consume_units": max(1, int(raw.get("consume_units", 1))),     # 1購入あたりの在庫消費
        # ---- 補充トリップ(物流) ----
        "lead_time_steps": max(0, int(raw.get("lead_time_steps", 3))),  # 発注→到着(配送走行時間)
        "restock_hour": int(raw.get("restock_hour", 4)),                # 日次補充レビューの時刻(時)
        # ---- 品切れ体験 ----
        "stockout_grievance": float(raw.get("stockout_grievance", 0.02)),  # 不透明 magnitude(0.0=観測のみ)
        # ---- 商品実体(②) ----
        "catalog": _cat_map("catalog", _DEFAULT_CATALOG),
        "belongings_max": max(1, int(raw.get("belongings_max", 8))),    # 所持(直近購入)の有界リスト長
    }


def enabled(sim) -> bool:
    """物流スライス(在庫・補充・商品実体)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "goodscfg", None)
    return bool(cfg and cfg["enabled"])


# ---------------------------------------------------------------- (s,S) パラメータ
def _capacity(cfg: dict, cat: str) -> int:
    return int(cfg["capacity"].get(str(cat), cfg["default_capacity"]))


def _reorder(cfg: dict, cat: str) -> int:
    return int(cfg["reorder_point"].get(str(cat), cfg["default_reorder_point"]))


def _poi_name(sim, node: str, cat: str) -> str | None:
    for p in sim.city.pois_at_node(node):
        if p.get("cat") == cat:
            return p.get("name")
    return None


def stock_of(sim, node: str, cat: str) -> int:
    """(node, cat) の現在在庫(遅延初期化=未購入 POI は上限 S から開始)。決定論・副作用は初期化のみ。"""
    key = (str(node), str(cat))
    st = sim._goods_stock
    cur = st.get(key)
    if cur is None:
        cur = _capacity(sim.goodscfg, cat)
        st[key] = cur
    return cur


# ---------------------------------------------------------------- 商品実体(②)
def pick_item(cfg: dict, cat: str, node: str, sim_min: int) -> str | None:
    """購入時に「何を買ったか」を決定論の日替わり巡回で選ぶ(organizations.daily_output と同型)。

    カタログ空/未掲載カテゴリは None(spend payload に item を載せない=会計不変)。day と POI/cat の
    安定ハッシュから index を決める=同一 POI は日ごとに巡回、POI ごとに別の品を推す(決定論)。"""
    catalog = cfg["catalog"].get(str(cat))
    if not catalog:
        return None
    day = int(sim_min) // 1440
    idx = (day + _stable_hash(f"{node}/{cat}")) % len(catalog)
    return catalog[idx]


def _note_belonging(agent, item: str, cfg: dict) -> None:
    """買った物を本人の所持(直近購入の有界リスト)+ 当日ダイジェスト材料に加える。

    属性は必要時にのみ生やす(OFF/非該当は属性不在=実行時状態を汚さない=バイト一致)。"""
    bel = getattr(agent, "_goods_belongings", None)
    if bel is None:
        bel = []
        agent._goods_belongings = bel
    bel.append(item)
    cap = int(cfg["belongings_max"])
    if len(bel) > cap:                                   # 有界: 最古を捨てる
        del bel[0:len(bel) - cap]
    acc = getattr(agent, "_goods_bought", None)
    if acc is None:
        acc = {}
        agent._goods_bought = acc
    acc[item] = acc.get(item, 0) + 1


def belongings(agent) -> list:
    """本人の所持(直近購入の有界リスト)。属性不在=空。"""
    return list(getattr(agent, "_goods_belongings", None) or [])


def clear_digest(agent) -> None:
    """発火(ダイジェスト取得)時に当日購入アキュムレータを空にする(前回発火以降の仕切り直し)。"""
    if getattr(agent, "_goods_bought", None):
        agent._goods_bought = {}


def digest_line(agent) -> str | None:
    """当日の購入を interstitial(S2)ダイジェストの1事実に整形(客観記述・意味づけしない)。

    アキュムレータ不在/空なら None(=1行も足さない=バイト一致)。テキストは本モジュール/config に閉じる。"""
    acc = getattr(agent, "_goods_bought", None)
    if not acc:
        return None
    total = sum(acc.values())
    top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if total >= _MANY_THRESHOLD:
        return f"今日は買い物が多かった({top}など{total}点)"
    return f"今日は{top}などを買った({total}点)"


# ---------------------------------------------------------------- 購入(在庫 decrement + 品切れ)
def on_purchase(sim, agent, cat: str, step: int, sim_min: int):
    """消費(食事/買物/夜遊び)の購入点で在庫を1単位減らす。戻り値 (ok, item)。

    ok=False は **在庫切れ=購入不成立**(呼び出し側は spend を出さない)。副作用: 実在庫由来の
    stock_out(kind 再利用・payload src="inventory")+ 不満(factors on_scarcity へ不透明 magnitude)+
    本人の記憶。ok=True は在庫があり decrement 済み・item は買った品(spend payload/所持/ダイジェスト用)。
    RNG は一切引かない=決定論・既存 draw 順を汚さない。"""
    cfg = sim.goodscfg
    node = agent.node
    key = (str(node), str(cat))
    st = sim._goods_stock
    cur = st.get(key)
    if cur is None:
        cur = _capacity(cfg, cat)                        # 遅延初期化=購入が起きた POI だけ実体化
    units = int(cfg["consume_units"])
    if cur < units:                                      # 在庫切れ → 購入不成立(実在庫由来の品切れ体験)
        st[key] = cur
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="stock_out", x=agent.x, y=agent.y,
                             payload={"poi": _poi_name(sim, node, cat) or node,
                                      "cat": str(cat), "src": "inventory"}))
        mag = float(cfg["stockout_grievance"])
        if mag != 0.0:
            from .factors import update as factor_update
            factor_update.on_scarcity(agent, mag, step=step, sim_min=sim_min,
                                      logger=sim.logger)
        agent.remember("欲しい物が品切れで買えなかった")
        return (False, None)
    st[key] = cur - units                                # 在庫を減らす
    item = pick_item(cfg, cat, node, sim_min)
    if item is not None:
        _note_belonging(agent, item, cfg)
    return (True, item)


# ---------------------------------------------------------------- 補充トリップ(物流)
def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _nearest_gateway(sim, node: str) -> str | None:
    """補充トリップの起点=最寄りゲートウェイ(街の外の物流拠点=depot)。決定論(同距離はノード昇順)。

    ゲートウェイが無ければ None(配送は「外から来る」抽象=起点不明でも在庫回復は成立)。"""
    gws = getattr(sim.city, "gateways", None) or []
    if not gws:
        return None
    p = sim.city.node_xy(node)
    return min(gws, key=lambda g: (_dist2(sim.city.node_xy(g), p), g))


def _delivery_origin(sim, node: str, cat: str) -> str | None:
    """補充トリップの出発地。B2B ⑤ ON なら卸 org の所在ノード(内生化)、それ以外は最寄り depot(従来)。

    b2b OFF/卸不在では従来どおり最寄りゲートウェイ=外生 depot(挙動不変=バイト一致)。"""
    from . import b2b as _b2b
    if _b2b.enabled(sim):
        org_node = _b2b.origin_node(sim, node, cat)
        if org_node is not None:
            return org_node
    return _nearest_gateway(sim, node)


def _b2b_fulfill(sim, node: str, cat: str, qty: int, step: int, sim_min: int) -> bool:
    """B2B ⑤ ON なら小売の仕入れを卸 org 在庫から満たす(不足=補充失敗=欠品波及)。OFF/卸不在は常に True。"""
    from . import b2b as _b2b
    if not _b2b.enabled(sim):
        return True
    return _b2b.fulfill(sim, node, cat, qty, step, sim_min)


def _delivery_blocked(sim, node: str) -> bool:
    """補充トリップの経路が封鎖されているか(災害の運休 / 摂動シナリオ shock_closure)。

    最小形: (a) 災害・運休で物流が止まる(sim.transit.suspended)/ (b) shock_closure が発動中で
    目的地に接する道路エッジが封鎖(sim.scenario.closed に node を端点に含む)。どちらも観測量・config・
    物理位置のみ参照=k 非依存。封鎖時は補充失敗→在庫回復なし(欠品が波及する災害接続の seam)。"""
    transit = getattr(sim, "transit", None)
    if transit is not None and bool(getattr(transit, "suspended", False)):
        return True
    scen = getattr(sim, "scenario", None)
    if scen is not None and bool(getattr(scen, "active", False)):
        closed = getattr(scen, "closed", None) or ()
        for edge in closed:
            if node in edge:
                return True
    return False


def review_and_order(sim, step: int, sim_min: int) -> None:
    """日次 (s,S) レビュー: 在庫が発注点 s 以下の POI へ最寄り depot から配送トリップを発つ。

    在庫状態のある POI(=購入が起きた POI)だけを走査(有界・軽量)。既に在庫が s 以下で発注が空いて
    いなければ、stock_low(在庫僅少)+ delivery_trip(配送トリップ・agent_id=-1 の世界イベント)を出し、
    到着 step を pending に積む(1 POI×cat あたり多重発注しない)。決定論・乱数なし・LLM 呼ゼロ。"""
    cfg = sim.goodscfg
    st = sim._goods_stock
    pending = sim._goods_pending
    lead = int(cfg["lead_time_steps"])
    for key in sorted(st):                               # ノード×カテゴリ昇順=決定論
        node, cat = key
        if key in pending:                               # 発注済み(在庫が届くまで多重発注しない)
            continue
        level = st[key]
        if level > _reorder(cfg, cat):
            continue
        frm = _delivery_origin(sim, node, cat)       # b2b ON=卸 org の所在ノード / OFF=最寄り depot
        qty = max(0, _capacity(cfg, cat) - level)
        arrive = step + lead
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="stock_low", x=0.0, y=0.0,
                             payload={"poi": node, "cat": str(cat), "level": int(level)}))
        # ---- 納品の運転手化(Wave 4 III-4 city_ops。**既定 OFF は 1 バイトも変わらない**)----
        # ★変えたのは「誰が運んだか」だけ: 発注判定・数量 qty・到着 eta・(s,S) は 1 も動かない。
        #   city_ops OFF(または運転手が当直に居ない)では下の 2 分岐をどちらも通らないので、
        #   agent_id=-1 と payload の 5 キーが従来と完全に同一 = ゴールデン L1 バイト一致。
        #   運転手が居ない ON のランでは unstaffed=true を出す(= 黙って無人で運ばない。
        #   transit_staff の dwell_decision と同じ「正直な無人マーカー」の作法)。
        trip_payload = {"from": frm, "to": node, "cat": str(cat),
                        "qty": int(qty), "eta": int(arrive)}
        trip_agent_id, trip_x, trip_y = -1, 0.0, 0.0
        driver = _city_ops.assign_delivery_driver(sim, node, step, sim_min)
        if driver is not None:
            trip_agent_id = int(driver.id)
            trip_x, trip_y = float(driver.x), float(driver.y)
            trip_payload["driver"] = int(driver.id)
            _city_ops.note_delivery_trip(sim, True)
        elif _city_ops.enabled(sim):
            trip_payload["unstaffed"] = True
            _city_ops.note_delivery_trip(sim, False)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=trip_agent_id,
                             kind="delivery_trip", x=trip_x, y=trip_y,
                             payload=trip_payload))
        pending[key] = arrive


def deliver_arrivals(sim, step: int, sim_min: int) -> None:
    """到着処理(毎step): 到着予定の補充トリップを在庫回復させる(restock)。

    経路が封鎖(_delivery_blocked)されていれば **補充失敗=在庫回復なし**(発注は落とす→翌レビューで
    再発注=欠品が波及)。開通していれば在庫を上限 S(capacity=order-up-to)へ戻し restock を記録する。
    決定論・乱数なし・LLM 呼ゼロ・個体の位置は動かさない(agent_id=-1 の世界イベント)。"""
    cfg = sim.goodscfg
    st = sim._goods_stock
    pending = sim._goods_pending
    due = sorted(k for k, arr in pending.items() if arr <= step)
    for key in due:
        pending.pop(key, None)
        node, cat = key
        if _delivery_blocked(sim, node):                 # 封鎖→補充失敗(在庫回復なし・翌レビューで再発注)
            continue
        cap = _capacity(cfg, cat)
        level = st.get(key, cap)
        qty = max(0, cap - level)
        if not _b2b_fulfill(sim, node, cat, qty, step, sim_min):   # ⑤ 卸在庫不足→仕入れ失敗=欠品波及
            continue                                      # (b2b OFF/卸不在は常に True=挙動不変)
        st[key] = cap
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="restock", x=0.0, y=0.0,
                             payload={"poi": node, "cat": str(cat), "qty": int(qty),
                                      "from": _nearest_gateway(sim, node)}))


def tick(sim, step: int, sim_min: int) -> None:
    """物流の毎step更新(既定 OFF=即 return=バイト一致)。到着処理(毎step)+ 日次補充レビュー。

    到着は毎step処理(封鎖の解除で自然復帰)。レビューは1日1回、restock_hour 以降の最初の step で発火
    (物流が街を走る時間帯=正典 §3.4 較正)。既存の scenario/disaster が確定した後に呼ぶ(その日の封鎖を
    補充が読む)。決定論・乱数なし=新 stream を引かない(ゴールデン保護)。"""
    if not enabled(sim):
        return
    deliver_arrivals(sim, step, sim_min)
    day = int(sim_min) // 1440
    minute = int(sim_min) % 1440
    if day > int(getattr(sim, "_goods_review_day", -1)) \
            and minute >= int(sim.goodscfg["restock_hour"]) * 60:
        sim._goods_review_day = day
        review_and_order(sim, step, sim_min)
