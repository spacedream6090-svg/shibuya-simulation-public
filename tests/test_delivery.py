"""宅配・フードデリバリー スライス④(delivery。既定 OFF)のテスト。

正典: docs/research/economy-goods-services.md §7 ④。R1 の鉄則を継承:
- OFF(既定): L1 が純粋既定と完全一致(バイト一致)・order/deliver 0 件・注文台帳も配達員フラグも生えない・
  "delivery" stream も引かない。
- ON: (a)在宅/職場滞在の食事帯注文=maybe_order が最寄り在庫店を選び ① 在庫を1引き order を出して stay を返す
      (外食に出ない=二重課金しない)。(b)品切れ店には注文が立たない(_nearest_store がスキップ・
      _reserve_stock が False)。(c)到着(eta)で受給=注文者に課金(食事+配達手数料)+ 配達員に gig 収入。
      (d)金と物の保存(注文者支出=店売上+配達員収入+手数料の帳尻・在庫が1減る)。(e)配達員実体ありなら
      dispatch が courier を物理配車(route が張られる)。(f)配達員不在は agent_id=-1 の抽象トリップへ graceful
      (注文は必ず配送される)。(g)決定論(同 seed 2回で L1 一致)。(h)k∈{free,off}で LLM 呼数一致(発火判断に
      k を食わせない=k 不変性)。(i)schema 登録。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤48 step)。追加 LLM 呼ゼロ・乱数は "delivery" stream のみ。
"""
from __future__ import annotations

import json

from society import delivery as delivery_mod
from society import goods as goods_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

_ON = {"delivery.enabled": "true"}
_OFF = {"delivery.enabled": "false"}
_GOODS = {"commerce.inventory.enabled": "true"}
_ISL = {"prompts.interstitial.enabled": "true"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _food_node(sim):
    return sim.city.pois_by_cat("food")[0]["node"]


def _neutralize_couriers(sim):
    """名簿中の既存「配達員」を全て非配達員にする(テストの配達員選定を決定論にする)。"""
    for a in sim.agents:
        if getattr(a, "occupation", "") == "配達員":
            a.occupation = "会社員"


def _orderer_at_home(sim):
    """agents[0] を食 POI ノードに「在宅・停止中」として据える(maybe_order の発火条件を満たす)。"""
    node = _food_node(sim)
    o = sim.agents[0]
    o.visitor = False
    o.home_node = node
    o.node = node
    o.route = []
    o.building = None
    o.x, o.y = sim.city.node_xy(node)
    o.money = 100000.0
    return o, node


def _set_courier(sim, at_node):
    """agents[1] を唯一の「配達員」として at_node に据える(勤務可能=起きて路上に居る)。"""
    _neutralize_couriers(sim)
    c = sim.agents[1]
    c.occupation = "配達員"
    c.node = at_node
    c.x, c.y = sim.city.node_xy(at_node)
    c.sleeping = False
    c.loc = "street"
    c.building = None
    c.route = []
    return c


# ------------------------------------------------------ (i) schema 登録
def test_schema_registered():
    for k in ("order", "deliver"):
        assert k in EVENT_KINDS, f"{k} が schema 未登録"


# ------------------------------------------------------ OFF: 純粋既定と L1 完全一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。OFF では宅配イベントも注文台帳も配達員フラグも生えない。"""
    pure = _sim(tmp_path, "pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=48, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(delivery seam が no-op でない)"
    for k in ("order", "deliver"):
        assert not _kind(off, k), f"OFF で {k} が出ている"
    assert not off._delivery_pending, "OFF なのに注文台帳が実体化している"
    for a in pure.agents:
        assert not hasattr(a, "_delivery_await"), "OFF なのに注文フラグが生えている"
        assert not hasattr(a, "_delivery_gig"), "OFF なのに配達 gig フラグが生えている"


def test_maybe_order_off_returns_none(tmp_path):
    """OFF では maybe_order が即 None を返す(注文せず・"delivery" stream も引かない)。"""
    sim = _sim(tmp_path, "moff", **_OFF)
    o, _node = _orderer_at_home(sim)
    assert delivery_mod.maybe_order(o, sim, 5, 12 * 60) is None


# ------------------------------------------------------ (a) 注文生成(外食の代替)
def test_maybe_order_places_order_and_stays(tmp_path):
    """ON で在宅×食事帯×当たりなら最寄り店へ注文を積み order を出し、stay を返す(外食に出ない)。"""
    sim = _sim(tmp_path, "order", **{**_ON, **_GOODS})
    sim.deliverycfg["order_rate"] = 1.0                  # 強制ヒット(決定論の抽選確認)
    orderer, node = _orderer_at_home(sim)
    _set_courier(sim, node)
    cap = goods_mod._capacity(sim.goodscfg, "food")
    action = delivery_mod.maybe_order(orderer, sim, 5, 12 * 60)
    assert action == {"type": "stay"}, "注文したのに外食に出ている(二重課金の危険)"
    assert len(sim._delivery_pending) == 1
    orders = _kind(sim, "order")
    assert len(orders) == 1 and orders[0].agent_id == orderer.id
    assert orders[0].payload["cat"] == "food" and "item" in orders[0].payload
    # ① 在庫が店で受取ぶん1減る(遅延初期化=上限 S から)
    assert sim._goods_stock[(node, "food")] == cap - 1
    # 当日は再注文しない(1日1回上限)
    assert delivery_mod.maybe_order(orderer, sim, 6, 12 * 60 + 10) is None
    assert len(sim._delivery_pending) == 1


def test_maybe_order_off_meal_window_none(tmp_path):
    """食事帯外では注文しない(注文受付=食事帯のみ)。"""
    sim = _sim(tmp_path, "win", **_ON)
    sim.deliverycfg["order_rate"] = 1.0
    orderer, _node = _orderer_at_home(sim)
    assert delivery_mod.maybe_order(orderer, sim, 5, 3 * 60) is None    # 3:00=食事帯外


# ------------------------------------------------------ (b) 品切れ店には注文が立たない
def test_no_order_at_stocked_out_poi(tmp_path):
    """① 品切れの最寄り店には注文を立てない(_nearest_store がスキップ・_reserve_stock=False・在庫は動かない)。"""
    sim = _sim(tmp_path, "sold_out", **{**_ON, **_GOODS})
    sim.deliverycfg["order_rate"] = 1.0
    _neutralize_couriers(sim)
    orderer, _node = _orderer_at_home(sim)
    store = delivery_mod._nearest_store(sim, orderer, sim.deliverycfg)
    assert store is not None
    snode = store[0]
    sim._goods_stock[(snode, "food")] = 0                # 最寄り店を品切れにする
    # _nearest_store は品切れ店をスキップ(別の在庫店 or None)
    store2 = delivery_mod._nearest_store(sim, orderer, sim.deliverycfg)
    assert store2 is None or store2[0] != snode
    # _reserve_stock は品切れ店で False(在庫は動かさない)
    assert delivery_mod._reserve_stock(sim, snode, "food") is False
    assert sim._goods_stock[(snode, "food")] == 0
    # maybe_order は品切れ店から注文を取らない(その店の在庫は 0 のまま)
    delivery_mod.maybe_order(orderer, sim, 5, 12 * 60)
    assert sim._goods_stock[(snode, "food")] == 0, "品切れ店から注文が立った(在庫が動いた)"


# ------------------------------------------------------ (c)(d) 到着で受給+課金+収入・金と物の保存
def test_order_to_delivery_conservation(tmp_path):
    """注文→在庫減→配送→到着課金の金と物の保存(注文者支出=店売上+配達員収入+手数料の帳尻)。"""
    sim = _sim(tmp_path, "cons", **{**_ON, **_GOODS})
    sim.deliverycfg["order_rate"] = 1.0
    orderer, node = _orderer_at_home(sim)
    courier = _set_courier(sim, node)                    # 店(=注文者ノード)に配達員=決定論選定・dist0
    cap = goods_mod._capacity(sim.goodscfg, "food")
    m0o, m0c = orderer.money, courier.money
    action = delivery_mod.maybe_order(orderer, sim, 5, 12 * 60)
    assert action == {"type": "stay"}
    rec = sim._delivery_pending[0]
    assert rec["courier"] == courier.id
    assert sim._goods_stock[(node, "food")] == cap - 1   # 物: 店で受取=在庫1減
    item_price, fee, income = rec["item_price"], rec["fee"], rec["courier_income"]
    # order の観測(手数料)
    assert _kind(sim, "order")[0].payload["fee"] == round(fee, 1)
    # 到着=配送完了(scheduler の会計経路を注入)
    delivery_mod.deliver_arrivals(sim, rec["arrive"], rec["arrive"] * 10,
                                  scheduler._spend, scheduler._pay_wage)
    # 金の保存: 注文者支出 = 食事 + 配達手数料
    assert abs((m0o - orderer.money) - (item_price + fee)) < 1e-6, "注文者支出が食事+手数料と不一致"
    # 配達員収入 = 手数料 × courier_share(gig 会計で入金)
    assert abs((courier.money - m0c) - income) < 1e-6, "配達員の gig 収入が手数料取り分と不一致"
    assert income == round(fee * sim.deliverycfg["courier_share"], 1)
    # 帳尻: 手数料 = 配達員収入 + プラットフォーム手数料(>=0)=二重計上なし
    platform = round(fee - income, 1)
    assert platform >= 0.0
    # 観測: deliver=1件・spend=2件(食事+delivery)・二重課金なし
    dev = _kind(sim, "deliver")
    assert len(dev) == 1 and dev[0].agent_id == courier.id
    assert dev[0].payload["courier"] == courier.id
    assert abs(dev[0].payload["fare"] - income) < 1e-6
    spends = _kind(sim, "spend")
    assert len(spends) == 2 and sorted(s.payload["cat"] for s in spends) == ["delivery", "food"]
    assert sim._delivery_total == 1
    assert orderer._delivery_await is False              # 受給後に注文フラグが解除される


# ------------------------------------------------------ (e) 配達員実体ありで物理移動が起きる
def test_courier_physical_movement(tmp_path):
    """配達員実体ありのケースで dispatch が courier を物理配車する(経路 route が張られ gig が立つ)。"""
    sim = _sim(tmp_path, "courier", **_ON)
    sim.deliverycfg["order_rate"] = 1.0
    orderer, node = _orderer_at_home(sim)
    foods = sim.city.pois_by_cat("food")
    other = next((p["node"] for p in foods if p["node"] != node), None)
    if other is None:                                    # 全食 POI が同一ノード=別の路上ノードへ
        other = next(n for n in sim.city.graph.nodes if n != node)
    courier = _set_courier(sim, other)                   # 店から離れた場所に配達員(移動が必要)
    action = delivery_mod.maybe_order(orderer, sim, 5, 12 * 60)
    assert action == {"type": "stay"}
    assert sim._delivery_pending[0]["courier"] == courier.id
    assert not courier.route                             # 配車前は経路なし
    delivery_mod.dispatch(sim, 6, 12 * 60 + 10)          # 物理配車(配達員→店→注文者)
    assert courier.route, "配達員実体が居るのに物理移動(route)が張られていない"
    assert getattr(courier, "_delivery_gig", None) is not None, "配達 gig フラグが立っていない"
    assert sim._delivery_pending[0]["dispatched"] is True
    # gig 中の配達員は routine の最上流で配送を最優先(continue=経路を進む)
    from society.cognition import routine
    act = routine._delivery.courier_action(courier, sim, 6, 12 * 60 + 10)
    assert act == {"type": "continue"}


# ------------------------------------------------------ (f) 配達員不在=抽象トリップへ graceful
def test_graceful_without_courier(tmp_path):
    """専任配達員が居ない名簿では agent_id=-1 の抽象トリップへ graceful=注文は必ず配送される(収入なし)。"""
    sim = _sim(tmp_path, "nocourier", **_ON)
    sim.deliverycfg["order_rate"] = 1.0
    _neutralize_couriers(sim)                            # 配達員を1人も居なくする
    orderer, _node = _orderer_at_home(sim)
    action = delivery_mod.maybe_order(orderer, sim, 5, 12 * 60)
    assert action == {"type": "stay"}
    rec = sim._delivery_pending[0]
    assert rec["courier"] == -1, "配達員不在なのに実体が紐づいた"
    assert rec["courier_income"] == 0.0
    delivery_mod.deliver_arrivals(sim, rec["arrive"], rec["arrive"] * 10,
                                  scheduler._spend, scheduler._pay_wage)
    dev = _kind(sim, "deliver")
    assert len(dev) == 1 and dev[0].agent_id == -1        # 世界イベント(抽象トリップ)
    assert dev[0].payload["courier"] == -1 and dev[0].payload["fare"] == 0.0
    assert not _kind(sim, "wage"), "配達員不在なのに gig 収入が支給された"
    assert sim._delivery_total == 1                       # 注文は必ず配送される


# ------------------------------------------------------ (g) 決定論(同 seed 2回で完全一致)
def test_on_deterministic(tmp_path):
    """delivery ON 同士 2 回で L1 完全一致(決定論・"delivery" stream のみ・mock)。"""
    a = _sim(tmp_path, "det_a", steps=30, **{**_ON, **_GOODS, **_ISL})
    a.run()
    b = _sim(tmp_path, "det_b", steps=30, **{**_ON, **_GOODS, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# ------------------------------------------------------ (h) k 不変性(発火判断に k を食わせない)
class _FixedLLM:
    """内容非依存の固定応答 backend。呼数だけを数える。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, steps=24,
               **{**ov, "prompts.interstitial.enabled": "true"})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_k_invariant(tmp_path):
    """delivery ON で k=free と k=off の LLM 呼数が一致(注文で在宅に留まる=co-location 変化だが判定に k を食わせない)。"""
    free = _run_fixed(tmp_path, "k_free",
                      **{**_ON, "delivery.order_rate": "1.0", "k.writeback": "free"})
    off = _run_fixed(tmp_path, "k_off",
                     **{**_ON, "delivery.order_rate": "1.0", "k.writeback": "off"})
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"k∈{{free,off}} で呼数が変化(k 不変性 違反): free={free.llm.calls} off={off.llm.calls}"
