"""B2B 卸→小売の仕入れ スライス⑤(commerce.inventory.b2b。既定 OFF)。

正典: docs/research/economy-goods-services.md §7 ⑤(会社間取引 B2B=卸→小売)。物流①(goods.py)の
店舗補充(restock)の供給元を、外生 depot(街の外のゲートウェイ)から **卸 org** へ内生化する層。
EVE の「入力→出力の物質収支」を1段だけ写す(§5.1)。組織の自然形成/供給網の創発(MEMORY
org-emergence-goal)と災害時の川上断絶の伝播(§3.1 の bankruptcy avalanche)に効く。

機構(全決定論・LLM 呼ゼロ・乱数ゼロ=新 stream 不要):

  1. 卸 org の日次生産: 既存の production(勤務完遂→_log_org_output)に **在庫増** を接続する。素材/製品を
     産出する org(output_kinds に goods/food 等を持つ=卸/製造)の在庫が勤務完遂ごとに production_units 増える。

  2. 小売の仕入れ(fulfill): goods.py の補充(restock)到着時に、補充先(小売 POI=node×cat)の supply_kind に
     一致する **最寄りの卸 org** の在庫から qty を引く。**org 間で金+物が移転**(卸=売上 revenue↑・在庫↓、
     小売=仕入費 procurement↑。org_ledger の会計流儀=買い側支出=売り側売上で保存)。b2b_trade を1件記録。

  3. 卸在庫不足=補充失敗: 卸の在庫が qty に満たなければ **仕入れ失敗**(goods.py の封鎖時失敗と同じ経路=
     在庫回復なし・翌レビューで再発注)=欠品が川上へ波及(供給網の雪崩が観測可能に)。

  4. 物流トリップ: goods.py の delivery_trip を再利用(出発地=卸 org の所在ノード)。新トリップ機構は作らない。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。卸 org の
  選定規則・会計・供給写像をここ/config に閉じる。生産量・仕入・清算は全決定論(乱数を一切引かない)。

R1 呼数不変: generate() を1本も足さない。既定 OFF(commerce.inventory.b2b.enabled=false、または inventory
  OFF)= 補充供給元は従来の外生 depot のまま・b2b_trade 0 件・在庫台帳も生えない=ゴールデンをバイト一致で守る。
  新イベント種は b2b_trade(schema.py 登録)。organizations が OFF(sim.orgs 不在)なら ON でも卸が居ない=
  外生 depot 扱いで graceful(b2b_trade 0 件)=b2b は organizations ON で初めて内生化する。
"""
from __future__ import annotations

from . import economy_sfc as sfc_mod
from .observer.schema import Event

# 小売カテゴリ → 卸が供給する output_kind(素材/製品)。実データ実査(organizations_shibuya.json):
#   food/cafe/nightlife は "food" を産出する org(CA 系: コーヒー豆焙煎・卸/ベーカリー 等)から、
#   shop は "goods" を産出する org(AP/WH 系: 雑貨卸・アパレル企画製造 等)から仕入れる。
# 卸=「output_kinds に素材/製品(goods/food)を持つ org」を選定規則とする(§7-⑤: output_kinds が素材/製品)。
_DEFAULT_SUPPLY_KIND = {"food": "food", "cafe": "food", "nightlife": "food", "shop": "goods"}
_SUPPLY_KINDS = ("food", "goods")   # 卸=これらの output_kind を持つ org(在庫が増える対象)


def build_cfg(raw_commerce) -> dict:
    """conf の commerce.inventory.b2b ブロックを型強制つきで正準化(既定 OFF=現行 goods と完全同一)。

    commerce ブロック全体を受け取り(goods.build_cfg と同型)、その下の inventory.b2b を読む。既定 OFF
    (enabled=false)で goods の補充供給元は従来の外生 depot のまま=b2b_trade 0 件=ゴールデンを守る。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw_commerce):
        raw_commerce = OmegaConf.to_container(raw_commerce, resolve=True)
    rc = dict(raw_commerce or {})
    inv = dict(rc.get("inventory", {}) or {})
    raw = dict(inv.get("b2b", {}) or {})
    sk = raw.get("supply_kind")
    supply = {str(k): str(v) for k, v in (dict(sk) if sk else dict(_DEFAULT_SUPPLY_KIND)).items()}
    wp = raw.get("wholesale_price")
    wprice = {str(k): float(v) for k, v in (dict(wp) if wp else {}).items()}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "supply_kind": supply,                                       # 小売 cat → 卸 output_kind
        "production_units": max(1, int(raw.get("production_units", 8))),  # 勤務完遂1回で卸在庫が増える量
        "initial_stock": max(0, int(raw.get("initial_stock", 0))),  # 卸の起動時在庫(0=生産で立ち上げ)
        # レーン乙 D2: 起動時在庫を「その卸が供給を受け持つ小売の**上限在庫 S の合計 × 日数**」で
        # 与える(0=OFF=現行と完全同一)。initial_stock(定額)より上位で、指定されたらこちらを使う。
        "initial_stock_days": max(0, int(raw.get("initial_stock_days", 0))),
        # 卸値(1単位あたりの仕入単価。cat 別に上書き可。既定=小売の下代近似)。
        "wholesale_price": wprice,
        "default_wholesale_price": float(raw.get("default_wholesale_price", 300.0)),
    }


def enabled(sim) -> bool:
    """B2B 内生化が有効か。inventory(goods)ON かつ b2b.enabled で成立。既定 OFF=外生 depot のまま。"""
    gcfg = getattr(sim, "goodscfg", None)
    bcfg = getattr(sim, "b2bcfg", None)
    return bool(gcfg and gcfg["enabled"] and bcfg and bcfg["enabled"])


def _state(sim) -> dict:
    """B2B の会計・在庫台帳(遅延構築)。stock=卸在庫 / revenue=卸売上 / procurement=小売仕入費合計 /
    sold_qty=卸出荷数 / trades=取引件数。org_ledger の会計流儀=買い側支出=売り側売上で保存する。"""
    b = getattr(sim, "_b2b", None)
    if not b:                                        # None または未構築の空 dict(simulation の初期値)
        b = {"stock": {}, "revenue": {}, "procurement": 0.0, "sold_qty": {}, "trades": 0}
        sim._b2b = b
    return b


def is_wholesale(org: dict) -> bool:
    """org が卸/製造(素材/製品 output_kind を持つ=在庫が増える供給元)か。"""
    oks = org.get("output_kinds") or []
    return any(k in _SUPPLY_KINDS for k in oks)


def _wholesale_price(cfg: dict, cat: str) -> float:
    return float(cfg["wholesale_price"].get(str(cat), cfg["default_wholesale_price"]))


def wholesale_for(sim, node: str, cat: str) -> dict | None:
    """小売(node, cat)の supply_kind に一致する最寄りの卸 org を返す(決定論・同距離は org id 昇順)。

    卸が居ない/organizations OFF(sim.orgs 不在)なら None(=外生 depot 扱いで graceful)。org の所在=
    workplace_poi.node(地図に無い org は候補外)。距離は小売ノードとの2乗距離。"""
    book = getattr(sim, "orgs", None)
    if not book:
        return None
    kind = sim.b2bcfg["supply_kind"].get(str(cat))
    if not kind:
        return None
    # ★レーン乙 D2(純粋な最適化。選ばれる org は 1 件も変わらない): この関数は 1 注文につき
    #   2 回(発注時の origin_node と到着時の fulfill)呼ばれ、そのたびに台帳 9,872 社を
    #   全走査していた。写像は (node, cat) → org の**静的な純関数**(台帳も地図も走行中に
    #   変わらない)なので 1 度だけ引いて控える。
    memo = getattr(sim, "_b2b_supplier_memo", None)
    if memo is None:
        memo = {}
        sim._b2b_supplier_memo = memo
    ck = (str(node), str(cat))
    if ck in memo:
        return memo[ck]
    p = sim.city.node_xy(node)
    best = None
    best_key = None
    for oid in sorted(book):
        org = book[oid]
        if kind not in (org.get("output_kinds") or []):
            continue
        wn = (org.get("workplace_poi") or {}).get("node")
        if not wn or wn not in sim.city.graph:
            continue
        q = sim.city.node_xy(wn)
        d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
        key = (d2, oid)
        if best_key is None or key < best_key:
            best_key = key
            best = org
    memo[ck] = best
    return best


def origin_node(sim, node: str, cat: str) -> str | None:
    """補充トリップ(delivery_trip)の出発地=卸 org の所在ノード(b2b ON 時)。卸不在なら None(depot へ委譲)。"""
    org = wholesale_for(sim, node, cat)
    if org is None:
        return None
    return (org.get("workplace_poi") or {}).get("node")


def on_production(sim, org: dict, step: int, sim_min: int) -> None:
    """卸 org の勤務完遂1回ぶんの生産=在庫増(scheduler._log_org_output から呼ぶ)。決定論・乱数なし。

    卸/製造(is_wholesale)でない org は在庫を持たない(no-op)。b2b OFF は完全 no-op。"""
    if not enabled(sim) or not is_wholesale(org):
        return
    oid = str(org["id"])
    b = _state(sim)
    b["stock"][oid] = b["stock"].get(oid, 0) + int(sim.b2bcfg["production_units"])


def _served_capacity(sim) -> dict:
    """卸 org → 「供給を受け持つ小売の上限在庫 S の合計」(1 回だけ構築・決定論・乱数ゼロ)。

    レーン乙 D2: ``wholesale_for`` は小売 1 件につき**最寄り 1 社**を指名するので、供給の
    受け持ちは台帳と地図だけで決まる静的な写像である。その「1 日で起こりうる最大の補充要求」
    (= Σ 上限在庫 S)が分かれば、起動時在庫を**日数**で指定できる(初期条件として正当な形)。
    地図上の小売 POI を 1 周するだけ(メモ化された ``wholesale_for`` を使う)。"""
    got = getattr(sim, "_b2b_served_cap", None)
    if got is not None:
        return got
    from . import goods as _goods
    gcfg = sim.goodscfg
    out: dict[str, int] = {}
    cats = tuple(sorted(sim.b2bcfg["supply_kind"]))
    for node in sorted(sim.city.graph):
        for poi in (sim.city.pois_at_node(node) or []):
            cat = str(poi.get("cat") or "")
            if cat not in cats:
                continue
            org = wholesale_for(sim, node, cat)
            if org is None:
                continue
            out[str(org["id"])] = out.get(str(org["id"]), 0) + _goods._capacity(gcfg, cat)
    sim._b2b_served_cap = out
    return out


def _initial_stock_for(sim, oid: str) -> int:
    """卸 1 社ぶんの起動時在庫(既定 0 = 現行と完全同一 = キーも生えない)。"""
    days = int(sim.b2bcfg["initial_stock_days"])
    if days > 0:
        return days * int(_served_capacity(sim).get(str(oid), 0))
    return int(sim.b2bcfg["initial_stock"])


def fulfill(sim, node: str, cat: str, qty: int, step: int, sim_min: int) -> bool:
    """小売(node, cat)への qty の仕入れを卸 org の在庫から満たす(goods.deliver_arrivals から呼ぶ)。

    戻り値=仕入れ成立か。卸不在(外生 depot 扱い)なら True(従来の外生補充=b2b_trade を出さない)。卸在庫が
    qty 未満なら False(=補充失敗=在庫回復なし=goods 側で欠品が波及)。成立時は org 間で金+物を移転:
    卸 在庫 −qty・売上 +amount / 小売 仕入費 +amount(会計保存=買い側支出=売り側売上)。b2b_trade を1件記録。
    RNG は一切引かない=決定論。amount = qty × 卸値(cat 別)。

    IF-E2 UNCOVERED(第98バッチ・``economy.org_accounting``。既定 OFF=完全 no-op=payload バイト一致):
    従来この帳簿 dict は**どの主体の残高とも接続していなかった**(金は 1 円も動かない)。ON では
    ``economy_sfc.on_b2b_trade`` が **買い手 org の預金 → 売り手 org の預金**へ実際に移す
    (買い手の小売 POI を台帳で特定できなければ域外資本の店とみなし RoW が払う)。
    **在庫・仕入れ成否(戻り値)・トリップ・帳簿 dict は 1 バイトも変わらない**=b2b の動力学は不変で、
    payload に ``payer`` / ``payee`` が増えるだけ。"""
    if qty <= 0:
        return True
    org = wholesale_for(sim, node, cat)
    if org is None:                                   # 卸不在=外生 depot 扱い(従来補充。b2b_trade なし)
        return True
    oid = str(org["id"])
    b = _state(sim)
    if oid not in b["stock"]:
        seed = _initial_stock_for(sim, oid)
        if seed:
            b["stock"][oid] = int(seed)                # 起動時在庫(遅延初期化)
    have = int(b["stock"].get(oid, 0))
    if have < qty:                                    # 卸在庫不足=仕入れ失敗(欠品波及・翌レビューで再発注)
        return False
    price = _wholesale_price(sim.b2bcfg, cat)
    amount = float(qty) * price
    b["stock"][oid] = have - qty                      # 物: 卸在庫が減る
    b["revenue"][oid] = b["revenue"].get(oid, 0.0) + amount   # 金: 売り側=売上
    b["procurement"] += amount                        # 金: 買い側=仕入費(合計)
    b["sold_qty"][oid] = b["sold_qty"].get(oid, 0) + int(qty)
    b["trades"] += 1
    sim._b2b_total = b["trades"]
    org_node = (org.get("workplace_poi") or {}).get("node")
    payload = {"from_org": oid, "to_poi": node, "cat": str(cat),
               "qty": int(qty), "amount": round(amount, 1),
               "from_node": org_node}
    parties = sfc_mod.on_b2b_trade(sim, node, cat, oid, amount, step, sim_min)
    if parties is not None:                           # IF-E2(既定 OFF=None=キーなし)
        payload["payer"], payload["payee"] = parties
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="b2b_trade", x=0.0, y=0.0, payload=payload))
    return True
