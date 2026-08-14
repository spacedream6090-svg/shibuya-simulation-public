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
        # ---- 第114 レーン乙(供給の詰まりを解く 2 件。既定 OFF=現行と 1 バイト一致)----
        # ① 部分納品: 在庫の範囲で納め、残りは翌日以降へ持ち越す(バックオーダー)。
        # ② 分散調達: 指名卸に在庫が無いとき近傍の同業卸へ決定論でフォールバックする。
        "partial_fulfillment": bool(raw.get("partial_fulfillment", False)),
        "multi_source": bool(raw.get("multi_source", False)),
        # 分散調達で当たる卸の総数(指名卸を含む)。距離昇順・同距離は org id 昇順。
        "max_sources": max(1, int(raw.get("max_sources", 3))),
    }


def enabled(sim) -> bool:
    """B2B 内生化が有効か。inventory(goods)ON かつ b2b.enabled で成立。既定 OFF=外生 depot のまま。"""
    gcfg = getattr(sim, "goodscfg", None)
    bcfg = getattr(sim, "b2bcfg", None)
    return bool(gcfg and gcfg["enabled"] and bcfg and bcfg["enabled"])


def _state(sim) -> dict:
    """B2B の会計・在庫台帳(遅延構築)。stock=卸在庫 / revenue=卸売上 / procurement=小売仕入費合計 /
    sold_qty=卸出荷数 / trades=取引件数。org_ledger の会計流儀=買い側支出=売り側売上で保存する。

    ★部分納品 ON のときだけ 3 欄が増える(``req_qty`` = 要求総量 / ``short_qty`` = 納めきれ
      なかった総量 / ``partial`` = 部分納品になった件数)。**遅延構築のときにキーを作らない**
      ので、OFF のランでは台帳 dict のキー集合が現行と 1 バイトも変わらない。"""
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


def suppliers_for(sim, node: str, cat: str) -> tuple:
    """小売(node, cat)から見た**近い順の卸 org 列**(先頭 = ``wholesale_for`` と同一社)。

    分散調達(``multi_source``)ON のときだけ呼ばれる。順序は (2乗距離, org id) の昇順で
    完全に決定論(乱数を 1 粒も引かない)。長さは ``max_sources`` まで。
    ``wholesale_for`` と同じく (node, cat) の**静的な純関数**(台帳も地図も走行中に
    変わらない)なので 1 度だけ引いて控える。
    """
    book = getattr(sim, "orgs", None)
    if not book:
        return ()
    kind = sim.b2bcfg["supply_kind"].get(str(cat))
    if not kind:
        return ()
    memo = getattr(sim, "_b2b_suppliers_memo", None)
    if memo is None:
        memo = {}
        sim._b2b_suppliers_memo = memo
    ck = (str(node), str(cat))
    if ck in memo:
        return memo[ck]
    p = sim.city.node_xy(node)
    ranked = []
    for oid in sorted(book):
        org = book[oid]
        if kind not in (org.get("output_kinds") or []):
            continue
        wn = (org.get("workplace_poi") or {}).get("node")
        if not wn or wn not in sim.city.graph:
            continue
        q = sim.city.node_xy(wn)
        ranked.append((((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2), oid, org))
    ranked.sort(key=lambda t: (t[0], t[1]))
    out = tuple(org for _d2, _oid, org in ranked[:int(sim.b2bcfg["max_sources"])])
    memo[ck] = out
    return out


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
    _ship(sim, org, node, cat, qty, step, sim_min)
    return True


def _stock_of(sim, b: dict, org: dict) -> int:
    """卸 1 社の現在庫(起動時在庫の遅延初期化つき。``fulfill`` の該当部分と同一の規則)。"""
    oid = str(org["id"])
    if oid not in b["stock"]:
        seed = _initial_stock_for(sim, oid)
        if seed:
            b["stock"][oid] = int(seed)
    return int(b["stock"].get(oid, 0))


def _ship(sim, org: dict, node: str, cat: str, qty: int, step: int, sim_min: int,
          req: int = 0) -> None:
    """卸 org → 小売(node, cat)へ qty 単位を出荷する(金と物の移転 + b2b_trade 1 件)。

    ``fulfill`` の成立枝をそのまま関数へ括り出したもので、**既定経路では payload も
    台帳の更新順も 1 バイト変わらない**(``req`` を渡すのは部分納品 ON のときだけ)。
    """
    oid = str(org["id"])
    b = _state(sim)
    price = _wholesale_price(sim.b2bcfg, cat)
    amount = float(qty) * price
    b["stock"][oid] = int(b["stock"].get(oid, 0)) - int(qty)   # 物: 卸在庫が減る
    b["revenue"][oid] = b["revenue"].get(oid, 0.0) + amount    # 金: 売り側=売上
    b["procurement"] += amount                        # 金: 買い側=仕入費(合計)
    b["sold_qty"][oid] = b["sold_qty"].get(oid, 0) + int(qty)
    b["trades"] += 1
    sim._b2b_total = b["trades"]
    org_node = (org.get("workplace_poi") or {}).get("node")
    payload = {"from_org": oid, "to_poi": node, "cat": str(cat),
               "qty": int(qty), "amount": round(amount, 1),
               "from_node": org_node}
    if req:                                           # 部分納品 ON のときだけ生える 1 キー
        payload["req"] = int(req)
    parties = sfc_mod.on_b2b_trade(sim, node, cat, oid, amount, step, sim_min)
    if parties is not None:                           # IF-E2(既定 OFF=None=キーなし)
        payload["payer"], payload["payee"] = parties
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="b2b_trade", x=0.0, y=0.0, payload=payload))


def fulfill_qty(sim, node: str, cat: str, qty: int, step: int, sim_min: int) -> int:
    """**実際に納品できた数量**を返す(第114 レーン乙: 部分納品 + 分散調達)。

    塞ぐ問題(第113 実測): 生産 42,408 / 需要 44,604 という**わずかな**恒常ギャップに対して、
    従来の ``fulfill`` は all-or-nothing(``have < qty`` なら 0 本も納めず False)だったため、
    「あと 1 個足りない」だけで補充が丸ごと失敗し、翌日も同じ量を要求してまた失敗する
    ——という自己再生産する詰まりを起こしていた(全店品切れの波及)。人為的に生産量を
    増やすのは**実勢の改竄**なので、直すのは配分の規則の側である。

      ① 部分納品(``partial_fulfillment``): 在庫の範囲で納める。残りは在庫水準が発注点を
         下回ったままになるので、**翌日の (s,S) レビューが自動的に再発注する**
         = バックオーダーは既存機構がそのまま担う(新しい待ち行列を作らない)。
      ② 分散調達(``multi_source``): 指名卸の在庫が尽きたら、近い順に最大
         ``max_sources`` 社まで当たる(決定論・乱数ゼロ)。

    **数量保存**: 納品量は各卸の在庫からしか引かず、在庫は ``on_production`` の生産と
    起動時在庫からしか増えない = Σ納品 ≤ Σ生産 + Σ起動時在庫 が構造的に成立する
    (「作った以上に納める」経路が 1 本も無い)。

    どちらも OFF のときは ``fulfill`` と**完全に同値**(成立なら qty・不成立なら 0)。
    """
    if qty <= 0:
        return 0
    cfg = sim.b2bcfg
    partial = bool(cfg["partial_fulfillment"])
    multi = bool(cfg["multi_source"])
    if not (partial or multi):                        # 既定 = 従来経路(1 行も新しく通らない)
        return int(qty) if fulfill(sim, node, cat, qty, step, sim_min) else 0
    orgs = suppliers_for(sim, node, cat) if multi else ()
    if not orgs:
        org = wholesale_for(sim, node, cat)
        if org is None:                               # 卸不在=外生 depot 扱い(従来補充)
            return int(qty)
        orgs = (org,)
    b = _state(sim)
    want = int(qty)
    got = 0
    for org in orgs:
        if want <= 0:
            break
        have = _stock_of(sim, b, org)
        take = have if partial else (want if have >= want else 0)
        if take > want:
            take = want
        if take <= 0:
            continue
        _ship(sim, org, node, cat, take, step, sim_min,
              req=(int(qty) if partial else 0))
        got += take
        want -= take
    if partial:                                       # 観測: 要求 / 不足 / 部分納品の件数
        b["req_qty"] = int(b.get("req_qty", 0)) + int(qty)
        b["short_qty"] = int(b.get("short_qty", 0)) + int(qty - got)
        if 0 < got < int(qty):
            b["partial"] = int(b.get("partial", 0)) + 1
    return int(got)
