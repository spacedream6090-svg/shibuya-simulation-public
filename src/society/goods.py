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

    # ---- 2層在庫 INV-A/B(既定 OFF=1 バイトも変えない)----------------------------
    tt = dict(raw.get("two_tier", {}) or {})

    def _tt_float_map(key):
        src = tt.get(key)
        return {str(k): float(v) for k, v in dict(src or {}).items()}

    def _tt_int_map(key):
        src = tt.get(key)
        return {str(k): int(v) for k, v in dict(src or {}).items()}

    two_tier = {
        "enabled": bool(tt.get("enabled", False)),
        # BY 容量 = 棚容量 × back_ratio(業態別。未掲載は default_back_ratio)
        "back_ratio": _tt_float_map("back_ratio"),
        "default_back_ratio": float(tt.get("default_back_ratio", 0.3)),
        # 棚の補充点(未掲載は 棚容量 × default_restock_ratio)。これを割ると店員の知覚が立つ
        "restock_point": _tt_int_map("restock_point"),
        "default_restock_ratio": float(tt.get("default_restock_ratio", 0.5)),
        "restock_batch": max(0, int(tt.get("restock_batch", 0))),   # 0=棚を満たすまで運ぶ
        "staff_restock": bool(tt.get("staff_restock", True)),       # 補充を店員の行動にする
        "staff_order": bool(tt.get("staff_order", True)),           # 発注を店主の行動にする
        "unstaffed_fallback": bool(tt.get("unstaffed_fallback", True)),
        "percept": bool(tt.get("percept", True)),                   # 来店時の棚知覚 1 行
    }

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
        # INV-C: レビュー間隔(0=現行の日次と完全同値。>0 で step 周期=CVS の 1 日 2-3 便 /
        #        SM の隔日を表現する。値は「何 step ごとに (s,S) を見るか」)
        "review_every_steps": max(0, int(raw.get("review_every_steps", 0))),
        # ---- 品切れ体験 ----
        "stockout_grievance": float(raw.get("stockout_grievance", 0.02)),  # 不透明 magnitude(0.0=観測のみ)
        # ---- 商品実体(②) ----
        "catalog": _cat_map("catalog", _DEFAULT_CATALOG),
        "belongings_max": max(1, int(raw.get("belongings_max", 8))),    # 所持(直近購入)の有界リスト長
        # ---- 2層在庫 + 店員/店主の行動化(INV-A/B)----
        "two_tier": two_tier,
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


# ---------------------------------------------------------------- 2層在庫(INV-A)
def two_tier_on(sim) -> bool:
    """店頭(棚)とバックヤードの 2 層在庫が有効か。既定 OFF=在庫は 1 層のまま=バイト一致。"""
    cfg = getattr(sim, "goodscfg", None)
    return bool(cfg and cfg["enabled"] and cfg["two_tier"]["enabled"])


def _back_capacity(cfg: dict, cat: str) -> int:
    """バックヤード容量 = 棚容量 × back_ratio(業態別。CVS 系 0.3 / SM 系 1.5 が較正目安)。"""
    tt = cfg["two_tier"]
    ratio = tt["back_ratio"].get(str(cat))
    if ratio is None:
        ratio = tt["default_back_ratio"]
    return max(0, int(_capacity(cfg, cat) * float(ratio)))


def _shelf_restock_point(cfg: dict, cat: str) -> int:
    """棚の補充点。ここを割ると「棚が薄い」が在店店員の知覚になる(購入時に O(1) で判定)。"""
    tt = cfg["two_tier"]
    v = tt["restock_point"].get(str(cat))
    if v is None:
        v = int(_capacity(cfg, cat) * float(tt["default_restock_ratio"]))
    return max(0, int(v))


def back_of(sim, node: str, cat: str) -> int:
    """(node, cat) のバックヤード在庫(未実体化=0=**空の倉庫から始まる**)。副作用なし。"""
    return int(getattr(sim, "_goods_back", {}).get((str(node), str(cat)), 0))


def _set_low(sim, node: str, cat: str) -> None:
    """棚薄フラグを立てる(node → {cat} の索引。O(1)・後で「その店だけ」を引ける形)。"""
    sim._goods_low.setdefault(str(node), set()).add(str(cat))


def _clear_low(sim, node: str, cat: str) -> None:
    """棚薄フラグを落とす(空になった店はキーごと消す=索引を有界に保つ)。"""
    s = sim._goods_low.get(str(node))
    if s is not None:
        s.discard(str(cat))
        if not s:
            sim._goods_low.pop(str(node), None)


def _note_shelf(sim, node: str, cat: str, level: int) -> None:
    """棚の水準が動いた直後に呼ぶ eager 更新(遅延評価はしない=resume 二重適用のバグ源)。"""
    if int(level) < _shelf_restock_point(sim.goodscfg, cat):
        _set_low(sim, node, cat)
    else:
        _clear_low(sim, node, cat)


def note_shelf(sim, node: str, cat: str, level: int) -> None:
    """棚の水準が**本 module の外**(宅配の取り置き等)で動いたときの eager 更新。2層 OFF は no-op。"""
    if two_tier_on(sim):
        _note_shelf(sim, node, cat, level)


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
    st[key] = cur - units                                # 在庫を減らす(2層 ON では **棚のみ**)
    if cfg["two_tier"]["enabled"]:
        # INV-B トリガ: 棚が補充点を割った瞬間に O(1) でフラグを立てる。これが「棚が減った」
        # という世界の状態であり、**変化(棚が埋まる)は在店店員の行動の結果**として起きる。
        _note_shelf(sim, node, cat, cur - units)
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


def _b2b_fulfill(sim, node: str, cat: str, qty: int, step: int, sim_min: int) -> int:
    """B2B ⑤ ON なら小売の仕入れを卸 org 在庫から満たし、**実際に納品できた数量**を返す。

    b2b OFF / 卸不在では常に qty(=満額。従来の外生 depot 補充と完全に同じ)。
    b2b ON でサブトグル(partial_fulfillment / multi_source)が両方 OFF なら、返り値は
    **qty か 0 のどちらか**しか取らない = 従来の all-or-nothing とバイト一致で同値。
    """
    from . import b2b as _b2b
    if not _b2b.enabled(sim):
        return int(qty)
    return _b2b.fulfill_qty(sim, node, cat, qty, step, sim_min)


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


def _order_one(sim, node: str, cat: str, step: int, sim_min: int) -> bool:
    """(node, cat) 1 件の (s,S) 判定と発注(**唯一の発注点**)。発注したら True。

    2層 OFF では従来と 1 バイトも変わらない(在庫水準=棚だけ・order-up-to=棚容量 S)。
    2層 ON では在庫水準を **棚+BY の合計**、order-up-to を **棚容量+BY容量** にする
    ((R,s,S) の教科書どおりの 2 段拡張。リード・b2b・多重発注抑止は不変)。
    """
    cfg = sim.goodscfg
    st = sim._goods_stock
    pending = sim._goods_pending
    key = (str(node), str(cat))
    if key in pending:                               # 発注済み(在庫が届くまで多重発注しない)
        return False
    level = st.get(key)
    if level is None:                                # 未実体化(一度も売れていない棚)=従来どおり対象外
        return False
    target = _capacity(cfg, cat)
    if cfg["two_tier"]["enabled"]:
        level = int(level) + back_of(sim, node, cat)
        target += _back_capacity(cfg, cat)
    if level > _reorder(cfg, cat):
        return False
    frm = _delivery_origin(sim, node, cat)       # b2b ON=卸 org の所在ノード / OFF=最寄り depot
    qty = max(0, target - level)
    arrive = step + int(cfg["lead_time_steps"])
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
    return True


def review_and_order(sim, step: int, sim_min: int) -> None:
    """定期 (s,S) レビュー: 在庫が発注点 s 以下の POI へ最寄り depot から配送トリップを発つ。

    在庫状態のある POI(=購入が起きた POI)だけを走査(有界・軽量)。既に在庫が s 以下で発注が空いて
    いなければ、stock_low(在庫僅少)+ delivery_trip(配送トリップ・agent_id=-1 の世界イベント)を出し、
    到着 step を pending に積む(1 POI×cat あたり多重発注しない)。決定論・乱数なし・LLM 呼ゼロ。

    ★INV-B(発注の行動化)ON のときは、**店主/店長が居る店はここで発注しない**(当人の出勤中の
      行動 `stock_order` が発注する=欠勤すれば発注されないのが世界の因果として正しい)。
      ここが受け持つのは「働き手が 1 人も割り当てられていない POI」だけで、それも
      `unstaffed_fallback` を宣言したときに限る(カバー率は provenance が正直に出す)。"""
    cfg = sim.goodscfg
    skip = None
    if cfg["two_tier"]["enabled"] and cfg["two_tier"]["staff_order"]:
        skip = staffed_nodes(sim, sim_min)
        if not cfg["two_tier"]["unstaffed_fallback"]:
            return                                       # 代替再現を宣言していない=誰も発注しない
    for key in sorted(sim._goods_stock):                 # ノード×カテゴリ昇順=決定論
        node, cat = key
        if skip is not None and node in skip:
            continue                                     # 担い手が居る店は当人の行動が発注する
        if _order_one(sim, node, cat, step, sim_min) and skip is not None:
            sim._goods_ops["order_unstaffed"] += 1


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
        shelf = st.get(key, cap)
        two_tier = cfg["two_tier"]["enabled"]
        if two_tier:
            bk = int(sim._goods_back.get(key, 0))
            level = int(shelf) + bk
            target = cap + _back_capacity(cfg, cat)
        else:
            bk, level, target = 0, shelf, cap
        qty = max(0, target - level)
        got = _b2b_fulfill(sim, node, cat, qty, step, sim_min)   # ⑤ 卸在庫から仕入れる
        if got <= 0:                                      # 全く納品できない=欠品が川上へ波及
            continue                                      # (b2b OFF/卸不在は常に満額=挙動不変)
        # ★部分納品(b2b.partial_fulfillment)ON のときだけ got < qty がありうる。その場合は
        #   届いた分だけ在庫が戻り、水準は発注点を下回ったままなので**翌日の (s,S) レビューが
        #   自動的に残りを再発注する**(= バックオーダーを既存機構が担う。新しい待ち行列を
        #   1 本も作らない)。OFF では got は必ず qty なので下の 1 行は st[key] = cap と同値。
        payload = {"poi": node, "cat": str(cat), "qty": int(got),
                   "from": _nearest_gateway(sim, node)}
        if two_tier:
            # INV-A: 納品は **バックヤードへ入る**(棚には 1 個も乗らない)。棚を埋めるのは
            # 店員の行動 shelf_restock だけ = 「店内に在るのに棚に無い」(Gruen)が構造で出る。
            sim._goods_back[key] = bk + got
            st.setdefault(key, int(shelf))               # 遅延初期化を確定(棚の水準は不変)
            payload["tier"] = "back"
            _note_shelf(sim, node, cat, st[key])         # 届いた=店員が棚出しすべき状態
        else:
            st[key] = level + got
        # PRICE-B2(閉店前見切り)が読む「**当日の入荷バッチ**」の標。厳密な消費期限は実装
        # しない(当日バッチ概念のみ=正典 §1.5 B2)ので、その (node,cat) に今日納品が
        # あったかだけを日番号で残す。dict 1 代入・イベントは 1 件も増えず L1 に 1 バイトも
        # 出ない(見切り OFF でも残すのは、ON/OFF で在庫台帳の履歴が食い違わないため)。
        sim._goods_delivered[key] = int(sim_min) // 1440
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="restock", x=0.0, y=0.0, payload=payload))


def tick(sim, step: int, sim_min: int) -> None:
    """物流の毎step更新(既定 OFF=即 return=バイト一致)。到着処理(毎step)+ 日次補充レビュー。

    到着は毎step処理(封鎖の解除で自然復帰)。レビューは既定で1日1回、restock_hour 以降の最初の step で
    発火(物流が街を走る時間帯=正典 §3.4 較正)。`review_every_steps` > 0 なら step 周期(便数の表現=
    INV-C)。既存の scenario/disaster が確定した後に呼ぶ(その日の封鎖を補充が読む)。決定論・乱数なし=
    新 stream を引かない(ゴールデン保護)。"""
    if not enabled(sim):
        return
    deliver_arrivals(sim, step, sim_min)
    cfg = sim.goodscfg
    every = int(cfg["review_every_steps"])
    if every > 0:                                        # INV-C: step 周期のレビュー(便数の表現)
        if int(step) % every == 0:                       # 決定論・状態を持たない=resume 安全
            review_and_order(sim, step, sim_min)
        return
    day = int(sim_min) // 1440
    minute = int(sim_min) % 1440
    if day > int(getattr(sim, "_goods_review_day", -1)) \
            and minute >= int(cfg["restock_hour"]) * 60:
        sim._goods_review_day = day
        review_and_order(sim, step, sim_min)


# =========================================================================== #
# INV-B: 補充・発注を「店員/店主の行動」として実装する層(2層 ON かつ本トグル ON のみ)
# --------------------------------------------------------------------------- #
# 設計原則(ユーザードクトリン): エンジンが世界を規定するのではなく、**世界の状態
# (棚が減った)はエージェントの行動トリガーであり、変化(棚が埋まる・発注される)は
# 当人の行動の結果**として起きる。エンジン規則による代替再現は「店員/店主が 1 人も
# 割り当てられていない POI」に限り、`unstaffed_fallback` で明示的に宣言したときだけ
# 通す(カバー率は provenance が正直に出す=誇張しない)。
#
# 費用: LLM 呼ゼロ・乱数ゼロ・k 非依存。棚薄フラグは購入時に O(1) で立ち、この層は
#   **フラグの立った店だけ**を回す(イベント駆動)= エンジン規則版と同じ計算量。
#   在勤索引だけが 1 step 1 パス(scheduler._phase_work_service と同型・同費用)。
# =========================================================================== #
#: 補充・発注の記憶(定型文。機構語・実験条件語・因子名を 1 文字も含まない)。
_RESTOCK_TEXT = "品薄の棚に商品を出した"
_ORDER_TEXT = "店の在庫を確かめて、足りない分を発注した"

#: 棚の様子(来店時の知覚)のカテゴリ語。config 由来の商品カタログと同じ層に閉じる。
_CAT_LABEL = {"food": "食べ物", "cafe": "飲み物", "shop": "商品", "nightlife": "酒"}


def staff_actions_on(sim) -> bool:
    """補充/発注のどちらかが「当人の行動」になっているか(2層 OFF は常に False)。"""
    if not two_tier_on(sim):
        return False
    tt = sim.goodscfg["two_tier"]
    return bool(tt["staff_restock"] or tt["staff_order"])


def _markdown_on(sim, sim_min: int) -> bool:
    """閉店前見切り(PRICE-B2)が **この step に** staff_phase へ相乗りするか。既定 OFF=False。

    見切りは棚(在庫の実体)と当日の入荷バッチを読むので、**在庫 ON が前提**である
    (在庫 OFF の世界には見切る棚が無い)。2層(two_tier)は前提にしない=単層でも効く。
    価格の判定式・イベント・時刻窓は commerce 側が持つ(本 module は価格を 1 文字も知らない)。
    ★時刻窓の先読みを含める: 見切りが起きうるのは 1 日のうち数十 step だけなので、窓の外では
      在勤者索引の O(N) 走査を 1 度も走らせない(INV-B が別の用事で組む step は相乗りする)。"""
    cfg = getattr(sim, "goodscfg", None)
    if not (cfg and cfg["enabled"]):
        return False
    from . import commerce as _commerce
    return _commerce.markdown_window_open(sim, sim_min)


def staffed_nodes(sim, sim_min: int) -> set:
    """**働き手が割り当てられている職場ノード**の集合(日次で 1 回だけ組む)。

    判定は既存の職場割当そのもの(``work_node`` があり ``work_start_min >= 0``
    = ``work.has_workplace`` と同じ述語)。ここに入らない POI が「職場ノード縮退の
    未カバー分」= エンジン規則で代替再現してよい唯一の範囲。
    転職・レイオフ・city_ops の再バインドで日々変わるので **日境界で組み直す**。"""
    day = int(sim_min) // 1440
    got = getattr(sim, "_goods_staffed", None)
    if got is not None and int(getattr(sim, "_goods_staffed_day", -1)) == day:
        return got
    out = set()
    for a in sim.agents:
        if getattr(a, "dead", False):
            continue
        wn = str(getattr(a, "work_node", "") or "")
        if wn and int(getattr(a, "work_start_min", -1)) >= 0:
            out.add(wn)
    sim._goods_staffed = out
    sim._goods_staffed_day = day
    return out


def store_cats(sim, node: str) -> list:
    """その店(ノード)で**在庫が実体化しているカテゴリ**(昇順=決定論)。

    POI 台帳を 1 周するだけ(1 ノードあたり数件)。棚が未実体化(一度も売れていない)の
    カテゴリは対象外 = 既存の (s,S) 走査と同じ有界性を保つ。"""
    st = sim._goods_stock
    out = set()
    for p in (sim.city.pois_at_node(node) or []):
        cat = str(p.get("cat") or "")
        if cat and (str(node), cat) in st:
            out.add(cat)
    return sorted(out)


def _restock_store(sim, node: str, agent, step: int, sim_min: int) -> int:
    """棚薄フラグの立った 1 店を BY から補充する(**1 店 1 補充 = L1 の洪水を作らない**)。

    agent=None は unstaffed フォールバック(担い手が 1 人も居ない POI のみ・宣言つき)。
    BY が空なら 1 個も動かない(= 棚は空のまま = 正しい欠品)。整数のみ・eager 更新。"""
    cats = sorted(sim._goods_low.get(str(node)) or ())
    if not cats:
        return 0
    cfg = sim.goodscfg
    batch = int(cfg["two_tier"]["restock_batch"])
    st, back = sim._goods_stock, sim._goods_back
    moved, total = [], 0
    for cat in cats:
        key = (str(node), cat)
        bk = int(back.get(key, 0))
        cap = _capacity(cfg, cat)
        shelf = int(st.get(key, cap))
        room = cap - shelf
        if room <= 0:                                    # 棚が満杯=補充不要(フラグだけ落とす)
            _clear_low(sim, node, cat)
            continue
        if bk <= 0:                                      # BY 空=運ぶ物が無い(フラグは立てたまま)
            continue
        move = min(bk, room)
        if batch > 0:
            move = min(move, batch)
        if move <= 0:
            continue
        st[key] = shelf + move                           # 棚 ← BY(総量は保存=どこにも湧かない)
        if bk - move <= 0:
            back.pop(key, None)
        else:
            back[key] = bk - move
        total += move
        moved.append(cat)
        _note_shelf(sim, node, cat, shelf + move)
    if not moved:
        return 0
    payload = {"poi": str(node), "cats": moved, "qty": int(total), "n": len(moved)}
    if agent is None:
        x, y = sim.city.node_xy(node)
        payload["unstaffed"] = True                      # 黙って無人で棚を埋めない(正直な標)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="shelf_restock", x=float(x), y=float(y),
                             payload=payload))
        sim._goods_ops["restock_unstaffed"] += 1
    else:
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                             kind="shelf_restock", x=float(agent.x), y=float(agent.y),
                             payload=payload))
        sim._goods_ops["restock_staffed"] += 1
        agent.remember(_RESTOCK_TEXT)                    # 当人の記憶=既存の work 文脈へ自然に載る
    sim._goods_ops["restock_units"] += int(total)
    return total


def _order_window(cfg: dict, step: int, sim_min: int) -> int:
    """店主のレビュー窓 id(既定=日。`review_every_steps` > 0 なら step 周期=便数)。"""
    every = int(cfg["review_every_steps"])
    if every > 0:
        return int(step) // every
    return int(sim_min) // 1440


def _order_store(sim, node: str, agent, step: int, sim_min: int) -> int:
    """店主/店長が自店の在庫を確かめて発注する(当人の行動。発注できた件数を返す)。"""
    n = 0
    for cat in store_cats(sim, node):
        if _order_one(sim, node, cat, step, sim_min):
            n += 1
    if n <= 0:
        return 0
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind="stock_order", x=float(agent.x), y=float(agent.y),
                         payload={"poi": str(node), "n": int(n)}))
    sim._goods_ops["order_staffed"] += n
    agent.remember(_ORDER_TEXT)
    return n


def staff_phase(sim, step: int, sim_min: int) -> None:
    """INV-B の単一作用点(既定 OFF=即 return=バイト一致)。位置確定後に呼ぶ。

    ①在勤索引: 「自分の職場ノードに**居て**、勤務時間帯にある」個体を work_node で索引する
      (scheduler._phase_work_service / city_ops._on_duty_crew と**同じ述語**=新しい在勤概念を
      発明しない)。②補充: 棚薄フラグの立った店に在勤者が居れば、**id 最小の当人の行動**として
      棚を埋める。③発注: 在勤者の居る店で、その窓でまだ発注していなければ当人が発注する。
      ④担い手が 1 人も割り当てられていない POI に限り、宣言つきでエンジンが代替再現する。
      ⑤閉店前見切り(PRICE-B2。既定 OFF): 同じ在勤索引に**相乗り**して、在店店員が値札を
        替える(markdown)。見切り専用の勤務概念を発明しないための相乗りで、判定式と
        イベントは commerce 側(commerce.markdown_phase)が持つ=本 module は価格を知らない。"""
    md_on = _markdown_on(sim, sim_min)
    staff_on = staff_actions_on(sim)
    if not (staff_on or md_on):
        return
    from .cognition import routine as _routine
    tt = sim.goodscfg["two_tier"]
    low = sim._goods_low
    if not (tt["staff_order"] or low or md_on):
        return                                           # 補充だけの構成で棚薄が 0 件=走査しない
    cal = getattr(sim, "calendarcfg", None)
    # 店の職場だけを見る(事務所等は素通り=毎 step の POI 走査を作らない)。日次索引に載って
    # いなくても**その日に棚薄フラグが立った店**は必ず拾う(索引の更新遅れで取りこぼさない)。
    stores = _store_nodes(sim, int(sim_min) // 1440)
    on_duty: dict[str, list] = {}
    for a in sim.agents:
        if a.loc == "outside" or getattr(a, "dead", False) or getattr(a, "sleeping", False):
            continue
        wn = str(getattr(a, "work_node", "") or "")
        if not wn or str(a.node) != wn:                  # 職場に**居る**ことが条件(在場)
            continue
        if wn not in stores and wn not in low:
            continue
        if not _routine.in_work_window(a, int(sim_min), cal):
            continue
        on_duty.setdefault(wn, []).append(a)
    for lst in on_duty.values():
        lst.sort(key=lambda z: z.id)                     # id 昇順=決定論(当番の唯一の入力)
    # ---- ② 補充(フラグの立った店だけ=イベント駆動)----
    if staff_on and tt["staff_restock"] and sim._goods_low:
        assigned = staffed_nodes(sim, sim_min) if tt["unstaffed_fallback"] else None
        for node in sorted(sim._goods_low):              # list コピー=走査中の削除に安全
            crew = on_duty.get(node)
            if crew:
                _restock_store(sim, node, crew[0], step, sim_min)
            elif assigned is not None and node not in assigned:
                _restock_store(sim, node, None, step, sim_min)   # 未カバー POI のみ代替再現
    # ---- ③ 発注(店主/店長の出勤中の行動。欠勤なら発注されない)----
    if staff_on and tt["staff_order"]:
        win = _order_window(sim.goodscfg, step, sim_min)
        seen = sim._goods_order_win
        for node in sorted(on_duty):
            if seen.get(node) == win:                    # この窓ではもう見た(1 店 1 レビュー)
                continue
            if not store_cats(sim, node):                # 店でない職場(事務所等)は素通り
                continue
            seen[node] = win
            _order_store(sim, node, on_duty[node][0], step, sim_min)
    # ---- ⑤ 閉店前見切り(PRICE-B2。既定 OFF=即 return=バイト一致)----
    if md_on:
        from . import commerce as _commerce
        _commerce.markdown_phase(
            sim, on_duty,
            staffed_nodes(sim, sim_min) if tt["unstaffed_fallback"] else None,
            step=step, sim_min=sim_min)


# ---------------------------------------------------------------- 来店時の棚知覚(客側)
def shelf_note(sim, agent) -> str | None:
    """いま居る店の棚の様子(3値のうち **非潤沢のときだけ** 1 行)。既定 OFF は None。

    購入判定が店の実在庫台帳を直接読む(pull 型)のと同じ現場主義の完成形で、
    「棚を見た」ことが LLM の判断・会話・記憶に自然に効く。O(1)/来店・LLM 呼数不変・
    プロンプト +1 行のみ。潤沢な棚は 1 行も足さない(= 平常時はバイト一致に近い)。"""
    cfg = getattr(sim, "goodscfg", None)
    if not (cfg and cfg["enabled"]):
        return None
    tt = cfg["two_tier"]
    if not (tt["enabled"] and tt["percept"]):
        return None
    node = str(getattr(agent, "node", "") or "")
    if not node:
        return None
    st = sim._goods_stock
    seen, parts = set(), []
    for p in (sim.city.pois_at_node(node) or []):
        cat = str(p.get("cat") or "")
        if not cat or cat in seen:
            continue
        seen.add(cat)
        level = st.get((node, cat))
        if level is None:                                # 一度も売れていない棚=見た目に変化なし
            continue
        if int(level) <= 0:
            state = "空だった"
        elif int(level) < _shelf_restock_point(cfg, cat):
            state = "残りわずかだった"
        else:
            continue                                     # 潤沢=1 行も足さない
        parts.append(f"{_CAT_LABEL.get(cat, '商品')}の棚は{state}")
    if not parts:
        return None
    return "棚の様子: " + "、".join(sorted(parts)[:2])


# ---------------------------------------------------------------- 観測(較正アンカー照合用)
def _store_nodes(sim, day: int | None = None) -> set:
    """在庫カテゴリの POI を持つノード(=店)。**日次で 1 回だけ**組む。

    ventures(個体が開いた店)で POI は日々増えうるので日境界で組み直す(staffed_nodes と同型)。
    day=None は「既にあるものを使う・無ければ組む」(観測=finalize からの呼び出し)。"""
    got = getattr(sim, "_goods_store_nodes", None)
    if got is not None and (day is None or int(getattr(sim, "_goods_store_day", -1)) == day):
        return got
    cats = set(sim.goodscfg["capacity"]) | set(_DEFAULT_CAPACITY)
    out = {str(k[0]) for k in sim._goods_stock}       # 棚が実体化した=売った実績のある店
    for node in sim.city.graph:
        for p in (sim.city.pois_at_node(node) or []):
            if str(p.get("cat") or "") in cats:
                out.add(str(node))
                break
    sim._goods_store_nodes = out
    sim._goods_store_day = -1 if day is None else int(day)
    return out


def provenance(sim) -> dict | None:
    """2層在庫の観測タリー(summary.json の goods キー)。2層 OFF は None=キーを出さない。

    較正アンカー(docs/plans/inventory-two-tier-plan.md §3)と照合できる**生の観測量**だけを
    出す(誇張も推計もしない): 欠品棚率・BY 在庫比・「BY に在るのに棚に無い」件数・
    補充/発注の担い手内訳・**担い手カバー率**(エンジン代替がどれだけ効いたか)。"""
    if not two_tier_on(sim):
        return None
    st = sim._goods_stock
    back = sim._goods_back
    ops = getattr(sim, "_goods_ops", None) or {}
    n_keys = len(st)
    empty = sum(1 for v in st.values() if int(v) <= 0)
    hidden = sum(1 for k, v in st.items() if int(v) <= 0 and int(back.get(k, 0)) > 0)
    shelf_units = sum(int(v) for v in st.values())
    back_units = sum(int(v) for v in back.values())
    units = shelf_units + back_units
    stores = _store_nodes(sim)
    assigned = getattr(sim, "_goods_staffed", None) or set()
    covered = len(stores & assigned)
    return {
        "shelf_keys": n_keys,
        "shelf_units": shelf_units,
        "back_units": back_units,
        "back_share": round(back_units / units, 4) if units else 0.0,
        "empty_shelves": empty,
        "empty_shelf_rate": round(empty / n_keys, 4) if n_keys else 0.0,
        # 「店内に在るのに棚に無い」= Gruen 25% 帯の機構的な対応物(INV-B が下げる量)
        "in_back_not_on_shelf": hidden,
        "in_back_not_on_shelf_rate": round(hidden / empty, 4) if empty else 0.0,
        "restock_staffed": int(ops.get("restock_staffed", 0)),
        "restock_unstaffed": int(ops.get("restock_unstaffed", 0)),
        "restock_units": int(ops.get("restock_units", 0)),
        "order_staffed": int(ops.get("order_staffed", 0)),
        "order_unstaffed": int(ops.get("order_unstaffed", 0)),
        # 担い手カバー率(最終日の職場割当が分子。未カバー分だけがエンジン代替の範囲)
        "store_nodes": len(stores),
        "staffed_store_nodes": covered,
        "staff_coverage": round(covered / len(stores), 4) if stores else 0.0,
        "unstaffed_fallback": bool(sim.goodscfg["two_tier"]["unstaffed_fallback"]),
    }
