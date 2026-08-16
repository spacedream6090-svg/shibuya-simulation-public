"""物流の実体化 スライス①+②(店舗在庫・日次補充トリップ・商品実体。既定 OFF)のテスト。

正典: docs/research/economy-goods-services.md §7 ①②。R1 の鉄則を継承:
- OFF(既定): L1 が純粋既定と完全一致(バイト一致)・delivery_trip/restock/stock_low/stock_out(実在庫版)
  0 件・在庫状態も所持も生えない。
- ON: (a)在庫が spend で減る。(b)在庫切れで spend 不成立+実在庫版 stock_out を記録。
      (c)発注点 s 以下で補充トリップ(delivery_trip+stock_low)→到着で在庫が上限 S へ回復(restock)。
      (d)封鎖(運休/shock_closure)で補充失敗=在庫回復なし。(e)商品カテゴリ(item)が spend に付与。
      (f)決定論(同 seed 2回で L1 一致)。(g)LLM 呼数が OFF と同一・k∈{free,off}一致(追加呼ゼロ=R1)。
      (h)schema 登録。(i)streaming 集計(在庫水準・品切れ率・補充回数・商品別売上)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤48 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from society import b2b as b2b_mod
from society import goods as goods_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer import stream as st
from society.observer.schema import EVENT_KINDS, Event

_ON = {"commerce.inventory.enabled": "true"}
_OFF = {"commerce.inventory.enabled": "false"}
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


# ------------------------------------------------------ (h) schema 登録
def test_schema_registered():
    for k in ("delivery_trip", "restock", "stock_low"):
        assert k in EVENT_KINDS, f"{k} が schema 未登録"
    assert "stock_out" in EVENT_KINDS         # 既存 kind を実在庫へ意味拡張して再利用


# ------------------------------------------------------ OFF: 純粋既定と L1 完全一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。OFF では物流イベントも在庫状態も所持も生えない。"""
    pure = _sim(tmp_path, "pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=48, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(goods seam が no-op でない)"
    for k in ("delivery_trip", "restock", "stock_low"):
        assert not _kind(off, k), f"OFF で {k} が出ている"
    assert not [e for e in off.logger.events
                if e.kind == "stock_out" and (e.payload or {}).get("src") == "inventory"]
    assert not pure._goods_stock, "OFF なのに在庫状態が実体化している"
    for a in pure.agents:
        assert not hasattr(a, "_goods_belongings"), "OFF なのに所持が生えている"


# ------------------------------------------------------ (a) 在庫が spend で減る
def test_stock_decrements_on_purchase(tmp_path):
    """on_purchase で在庫が consume_units 減り、(True, item) を返す(遅延初期化=上限 S から開始)。"""
    sim = _sim(tmp_path, "dec", **_ON)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    cap = goods_mod._capacity(sim.goodscfg, "food")
    ok, item = goods_mod.on_purchase(sim, agent, "food", 3, 700)
    assert ok is True and item is not None
    assert sim._goods_stock[(node, "food")] == cap - sim.goodscfg["consume_units"]
    # もう一度買うとさらに減る(decrement の累積)
    goods_mod.on_purchase(sim, agent, "food", 4, 710)
    assert sim._goods_stock[(node, "food")] == cap - 2 * sim.goodscfg["consume_units"]
    # 買った物が本人の所持に乗る
    assert item in goods_mod.belongings(agent)


def test_stock_decrements_via_charge_meal(tmp_path):
    """実ラン経路(_charge_meal)でも在庫が1単位減り、spend が1件出る(commerce OFF=価格不変)。"""
    sim = _sim(tmp_path, "meal", **_ON)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    agent.x, agent.y = sim.city.node_xy(node)
    agent.money = 100000.0
    cap = goods_mod._capacity(sim.goodscfg, "food")
    scheduler._charge_meal(sim, agent, 5, 700)
    assert sim._goods_stock[(node, "food")] == cap - 1
    assert len(_kind(sim, "spend")) == 1


# ------------------------------------------------------ (b) 在庫切れで購入不成立
def test_stockout_suppresses_purchase(tmp_path):
    """在庫 0 の POI で on_purchase は (False, None)=購入不成立、実在庫版 stock_out を記録+grievance。"""
    sim = _sim(tmp_path, "out", **_ON)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    sim._goods_stock[(node, "food")] = 0            # 在庫を尽きさせる
    g0 = agent.states.get("grievance", 0.0)
    ok, item = goods_mod.on_purchase(sim, agent, "food", 3, 700)
    assert ok is False and item is None
    outs = [e for e in _kind(sim, "stock_out") if e.payload.get("src") == "inventory"]
    assert len(outs) == 1 and outs[0].payload["cat"] == "food"
    assert agent.states.get("grievance", 0.0) > g0, "品切れで grievance が動いていない"
    assert sim._goods_stock[(node, "food")] == 0, "品切れなのに在庫が動いた"


def test_stockout_suppresses_spend_via_charge_meal(tmp_path):
    """_charge_meal 経路でも在庫切れなら spend を1件も出さない(実在庫由来の品切れ体験)。"""
    sim = _sim(tmp_path, "outmeal", **_ON)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    agent.x, agent.y = sim.city.node_xy(node)
    agent.money = 100000.0
    sim._goods_stock[(node, "food")] = 0
    scheduler._charge_meal(sim, agent, 5, 700)
    assert not _kind(sim, "spend"), "在庫切れなのに spend が出た(購入不成立でない)"
    assert [e for e in _kind(sim, "stock_out") if e.payload.get("src") == "inventory"]


# ------------------------------------------------------ (c) 発注点→補充トリップ→到着で回復
def test_reorder_dispatches_delivery_trip(tmp_path):
    """在庫が発注点 s 以下なら review で delivery_trip + stock_low を出し、pending に到着 step を積む。"""
    sim = _sim(tmp_path, "order", **_ON)
    node = _food_node(sim)
    s = goods_mod._reorder(sim.goodscfg, "food")
    sim._goods_stock[(node, "food")] = s            # ちょうど発注点(<=s で発注)
    goods_mod.review_and_order(sim, 10, 700)
    trips = _kind(sim, "delivery_trip")
    lows = _kind(sim, "stock_low")
    assert len(trips) == 1 and trips[0].payload["to"] == node
    assert trips[0].agent_id == -1 and trips[0].payload["cat"] == "food"
    assert len(lows) == 1 and lows[0].payload["level"] == s
    assert (node, "food") in sim._goods_pending
    assert sim._goods_pending[(node, "food")] == 10 + sim.goodscfg["lead_time_steps"]
    # 発注済みは多重発注しない
    goods_mod.review_and_order(sim, 11, 710)
    assert len(_kind(sim, "delivery_trip")) == 1, "在庫が届く前に多重発注している"


def test_no_order_when_above_reorder(tmp_path):
    """在庫が発注点より多ければ review は何も出さない(補充トリップ 0 件)。"""
    sim = _sim(tmp_path, "noorder", **_ON)
    node = _food_node(sim)
    sim._goods_stock[(node, "food")] = goods_mod._reorder(sim.goodscfg, "food") + 1
    goods_mod.review_and_order(sim, 10, 700)
    assert not _kind(sim, "delivery_trip") and not _kind(sim, "stock_low")


def test_arrival_restocks_to_capacity(tmp_path):
    """到着処理で在庫が上限 S(capacity)へ回復し、restock を1件記録する。"""
    sim = _sim(tmp_path, "arrive", **_ON)
    node = _food_node(sim)
    cap = goods_mod._capacity(sim.goodscfg, "food")
    sim._goods_stock[(node, "food")] = 1
    sim._goods_pending[(node, "food")] = 12         # step 12 到着
    goods_mod.deliver_arrivals(sim, 12, 800)
    restocks = _kind(sim, "restock")
    assert len(restocks) == 1 and restocks[0].payload["poi"] == node
    assert restocks[0].payload["qty"] == cap - 1 and restocks[0].agent_id == -1
    assert sim._goods_stock[(node, "food")] == cap, "到着したのに在庫が S へ戻っていない"
    assert (node, "food") not in sim._goods_pending, "到着したのに pending が残っている"


def test_full_loop_review_to_restock(tmp_path):
    """発注点→delivery_trip→lead_time 後の到着→restock→在庫回復 の全経路が閉じる。"""
    sim = _sim(tmp_path, "loop", **_ON)
    node = _food_node(sim)
    cap = goods_mod._capacity(sim.goodscfg, "food")
    lead = sim.goodscfg["lead_time_steps"]
    sim._goods_stock[(node, "food")] = 0
    goods_mod.review_and_order(sim, 20, 700)
    assert _kind(sim, "delivery_trip")
    goods_mod.deliver_arrivals(sim, 20 + lead, 730)
    assert _kind(sim, "restock")
    assert sim._goods_stock[(node, "food")] == cap


# ------------------------------------------------------ (d) 封鎖で補充失敗(災害接続の seam)
def test_closure_blocks_restock(tmp_path):
    """運休(sim.transit.suspended)中は到着しても補充失敗=在庫回復なし・restock 0 件・翌レビューで再発注。"""
    sim = _sim(tmp_path, "block", **_ON)
    node = _food_node(sim)
    sim._goods_stock[(node, "food")] = 0
    sim._goods_pending[(node, "food")] = 12
    sim.transit.suspended = True                    # 災害の運休=物流が止まる
    goods_mod.deliver_arrivals(sim, 12, 800)
    assert not _kind(sim, "restock"), "封鎖中なのに補充が成立した"
    assert sim._goods_stock[(node, "food")] == 0, "封鎖中なのに在庫が回復した"
    assert (node, "food") not in sim._goods_pending, "失敗した発注が残っている(再発注できない)"
    # 開通すれば次の発注→到着で回復する(欠品が波及した後の復帰)
    sim.transit.suspended = False
    goods_mod.review_and_order(sim, 13, 810)
    assert _kind(sim, "delivery_trip"), "開通後に再発注されていない"


def test_scenario_closure_blocks_restock(tmp_path):
    """摂動シナリオ shock_closure(目的地に接する道路封鎖)でも補充失敗=在庫回復なし。"""
    sim = _sim(tmp_path, "scen", **_ON)
    node = _food_node(sim)
    sim._goods_stock[(node, "food")] = 0
    sim._goods_pending[(node, "food")] = 12
    # 目的地ノードを端点に含む封鎖エッジを直に据える(shock_closure の最小再現)
    sim.scenario.active = True
    sim.scenario.closed = {(node, "zzz_other_node")}
    goods_mod.deliver_arrivals(sim, 12, 800)
    assert not _kind(sim, "restock")
    assert sim._goods_stock[(node, "food")] == 0


# ------------------------------------------------------ (e) 商品カテゴリが spend に付与
def test_item_attached_to_spend(tmp_path):
    """_charge_meal 経由の spend payload に item(買った物)が載る(会計不変=金額は変えない)。"""
    sim = _sim(tmp_path, "item", **_ON)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    agent.x, agent.y = sim.city.node_xy(node)
    agent.money = 100000.0
    scheduler._charge_meal(sim, agent, 5, 700)
    spends = _kind(sim, "spend")
    assert len(spends) == 1 and "item" in spends[0].payload
    assert spends[0].payload["item"] in sim.goodscfg["catalog"]["food"]
    assert spends[0].payload["cat"] == "food"


def test_item_absent_when_off(tmp_path):
    """goods OFF では spend payload に item を足さない(バイト一致の担保)。"""
    sim = _sim(tmp_path, "noitem", **_OFF)
    node = _food_node(sim)
    agent = sim.agents[0]
    agent.node = node
    agent.x, agent.y = sim.city.node_xy(node)
    agent.money = 100000.0
    scheduler._charge_meal(sim, agent, 5, 700)
    spends = _kind(sim, "spend")
    assert len(spends) == 1 and "item" not in spends[0].payload


def test_pick_item_deterministic():
    """商品選択は決定論(同一 (cat, node, day) は同じ品。カタログ空/未掲載は None)。"""
    cfg = goods_mod.build_cfg({"inventory": {"enabled": True}})
    a = goods_mod.pick_item(cfg, "food", "nodeA", 700)
    b = goods_mod.pick_item(cfg, "food", "nodeA", 700)
    assert a == b and a in cfg["catalog"]["food"]
    assert goods_mod.pick_item(cfg, "unknown_cat", "nodeA", 700) is None


# ------------------------------------------------------ (f) 決定論(同 seed 2回で完全一致)
def test_on_deterministic(tmp_path):
    """goods ON 同士 2 回で L1 完全一致(決定論・乱数不使用・mock)。"""
    a = _sim(tmp_path, "det_a", steps=30, **{**_ON, **_ISL})
    a.run()
    b = _sim(tmp_path, "det_b", steps=30, **{**_ON, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# ------------------------------------------------------ (g) LLM 呼数が OFF と同一・k 不変
class _FixedLLM:
    """内容非依存の固定応答 backend。呼数だけを数える(下流分岐を無効化して呼数の純測定)。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, steps=24, **{**ov, "prompts.interstitial.enabled": "true"})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_invariant(tmp_path):
    """goods ON/OFF で LLM 呼数が完全一致(追加 LLM 呼ゼロ=R1)。"""
    on = _run_fixed(tmp_path, "cc_on", **_ON)
    off = _run_fixed(tmp_path, "cc_off", **_OFF)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"goods で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


def test_llm_call_count_k_invariant(tmp_path):
    """goods ON で k=free と k=off の LLM 呼数が一致(発火判断に k を食わせない=k 不変性)。"""
    free = _run_fixed(tmp_path, "k_free", **{**_ON, "k.writeback": "free"})
    off = _run_fixed(tmp_path, "k_off", **{**_ON, "k.writeback": "off"})
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"k∈{{free,off}} で呼数が変化(k 不変性 違反): free={free.llm.calls} off={off.llm.calls}"


# ------------------------------------------------------ (i) streaming 集計
def _write_run(rd, events):
    """events(dict 列)を l1_events.parquet + summary.json に書く(test_streaming_analyze と同流儀)。"""
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e.get("sim_min", 0)) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0 for _ in events], "y": [0.0 for _ in events],
        "payload": [json.dumps(e["payload"], ensure_ascii=False, sort_keys=True)
                    for e in events],
    }
    pq.write_table(pa.table(cols), rd / "l1_events.parquet")
    (rd / "summary.json").write_text(json.dumps({"n_steps": 1}), encoding="utf-8")


def _ev(step, aid, kind, **payload):
    return {"step": step, "sim_min": step * 10, "agent_id": aid, "kind": kind,
            "payload": payload}


def test_goods_usage_aggregation(tmp_path):
    """streaming 集計 goods_usage が在庫水準・品切れ率・補充回数・商品別売上を有界1パスで出す。"""
    ev = [
        _ev(0, 1, "spend", cat="food", amount=900, item="ラーメン"),
        _ev(0, 2, "spend", cat="food", amount=900, item="ラーメン"),
        _ev(0, 3, "spend", cat="shop", amount=2500, item="雑貨"),
        _ev(1, 4, "stock_out", cat="food", poi="n1", src="inventory"),
        _ev(1, -1, "stock_out", cat="food", poi="n9"),        # 在館数代理(src なし)=除外
        _ev(2, -1, "stock_low", cat="food", poi="n1", level=3),
        _ev(2, -1, "delivery_trip", cat="food", to="n1", frm="g1", qty=39, eta=5),
        _ev(2, -1, "delivery_trip", cat="shop", to="n2", frm="g1", qty=20, eta=5),
        _ev(5, -1, "restock", cat="food", poi="n1", qty=39),  # food は到着
        # shop の delivery_trip は到着せず=封鎖の代理(restock_failed=1)
    ]
    rd = tmp_path / "agg"
    _write_run(rd, ev)
    out = st.goods_usage(str(rd))
    t = out["totals"]
    assert t["delivery_trip"] == 2 and t["restock"] == 1
    assert t["stock_low"] == 1 and t["stock_out"] == 1     # 実在庫版のみ(src なしは除外)
    assert out["restock_qty"] == 39
    assert out["restock_failed"] == 1                       # 2 配送 - 1 到着
    # 品切れ率 = 1 / (1 品切れ + 3 商品つき spend) = 0.25
    assert out["stockout_rate"] == 0.25 and out["items_sold_total"] == 3
    assert out["by_cat"]["food"]["sold"] == 2 and out["by_cat"]["shop"]["sold"] == 1
    assert out["top_items"][0] == ("ラーメン", 2)


# ==================================================================== スライス⑤ B2B(卸→小売)
# 正典 §7 ⑤。①の補充供給元を外生 depot から卸 org へ内生化(org 間で金+物が移転・保存)。卸在庫不足=
# 補充失敗(封鎖時と同じ経路=欠品波及)。全決定論・乱数ゼロ・LLM 呼ゼロ。既定 OFF=b2b_trade 0 件=バイト一致。
_B2B_ON = {"commerce.inventory.enabled": "true",
           "commerce.inventory.b2b.enabled": "true",
           "organizations.enabled": "true"}


def _shop_node(sim):
    return sim.city.pois_by_cat("shop")[0]["node"]


def test_b2b_schema_registered():
    assert "b2b_trade" in EVENT_KINDS, "b2b_trade が schema 未登録"


def test_b2b_off_uses_gateway_no_trade(tmp_path):
    """inventory ON・b2b OFF では補充供給元は外生 depot のまま=b2b_trade 0 件・restock は成立(挙動不変)。"""
    sim = _sim(tmp_path, "b2boff", **_ON)
    node = _food_node(sim)
    sim._goods_stock[(node, "food")] = goods_mod._reorder(sim.goodscfg, "food")
    goods_mod.review_and_order(sim, 10, 700)
    assert _kind(sim, "delivery_trip")
    goods_mod.deliver_arrivals(sim, 10 + sim.goodscfg["lead_time_steps"], 730)
    assert _kind(sim, "restock") and not _kind(sim, "b2b_trade")


def test_b2b_wholesale_selection(tmp_path):
    """卸 org 選定: 小売 cat の supply_kind(food系→food・shop→goods)に一致する org を返す(実データ実査)。"""
    sim = _sim(tmp_path, "b2bsel", **_B2B_ON)
    scheduler._ensure_orgs(sim)
    org_f = b2b_mod.wholesale_for(sim, _food_node(sim), "food")
    assert org_f is not None and "food" in (org_f.get("output_kinds") or [])
    org_s = b2b_mod.wholesale_for(sim, _shop_node(sim), "shop")
    assert org_s is not None and "goods" in (org_s.get("output_kinds") or [])


def test_b2b_production_grows_stock(tmp_path):
    """卸/製造 org(素材/製品 産出)は生産で在庫が増え、非卸 org は増えない(is_wholesale)。"""
    sim = _sim(tmp_path, "b2bprod", **_B2B_ON)
    scheduler._ensure_orgs(sim)
    wh = next(o for o in sim.orgs.values() if b2b_mod.is_wholesale(o))
    nonwh = next(o for o in sim.orgs.values() if not b2b_mod.is_wholesale(o))
    b = b2b_mod._state(sim)
    u = sim.b2bcfg["production_units"]
    b2b_mod.on_production(sim, wh, 5, 700)
    b2b_mod.on_production(sim, wh, 6, 710)
    assert b["stock"][str(wh["id"])] == 2 * u
    b2b_mod.on_production(sim, nonwh, 5, 700)
    assert str(nonwh["id"]) not in b["stock"], "非卸 org の在庫が増えている"


def test_b2b_fulfill_conservation(tmp_path):
    """小売補充が卸在庫から引かれ、org 間で金+物が保存する(買い側仕入費=売り側売上・卸在庫が qty 減)。"""
    sim = _sim(tmp_path, "b2bcons", **_B2B_ON)
    scheduler._ensure_orgs(sim)
    node = _shop_node(sim)
    org = b2b_mod.wholesale_for(sim, node, "shop")
    oid = str(org["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 100
    cap = goods_mod._capacity(sim.goodscfg, "shop")
    price = b2b_mod._wholesale_price(sim.b2bcfg, "shop")
    sim._goods_stock[(node, "shop")] = 0
    sim._goods_pending[(node, "shop")] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    trades = _kind(sim, "b2b_trade")
    assert len(trades) == 1
    tr = trades[0].payload
    assert tr["from_org"] == oid and tr["to_poi"] == node and tr["cat"] == "shop"
    assert tr["qty"] == cap and trades[0].agent_id == -1
    assert abs(tr["amount"] - cap * price) < 1e-6
    # 物の保存: 卸在庫が qty だけ減った / 小売は上限 S へ補充された
    assert b["stock"][oid] == 100 - cap
    assert sim._goods_stock[(node, "shop")] == cap and _kind(sim, "restock")
    # 金の保存: 売り側=売上 == 買い側=仕入費 == amount(二重計上なし)
    assert abs(b["revenue"][oid] - cap * price) < 1e-6
    assert abs(b["procurement"] - cap * price) < 1e-6
    assert b["sold_qty"][oid] == cap and b["trades"] == 1


def test_b2b_depletion_blocks_restock(tmp_path):
    """卸在庫が仕入れ量に満たない → 補充失敗=在庫回復なし・b2b_trade 0 件(欠品が川上へ波及)。"""
    sim = _sim(tmp_path, "b2bdeplete", **_B2B_ON)
    scheduler._ensure_orgs(sim)
    node = _shop_node(sim)
    org = b2b_mod.wholesale_for(sim, node, "shop")
    oid = str(org["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 5                                   # < 補充量(cap=30)=卸在庫不足
    sim._goods_stock[(node, "shop")] = 0
    sim._goods_pending[(node, "shop")] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    assert not _kind(sim, "b2b_trade") and not _kind(sim, "restock")
    assert sim._goods_stock[(node, "shop")] == 0, "卸在庫不足なのに補充が成立した(欠品波及せず)"
    assert b["stock"][oid] == 5, "取引不成立なのに卸在庫が減っている"


def test_b2b_graceful_without_orgs(tmp_path):
    """b2b ON でも organizations OFF(卸 org 不在)なら外生 depot 扱いで graceful=restock 成立・b2b_trade 0 件。"""
    sim = _sim(tmp_path, "b2bnoorg",
               **{"commerce.inventory.enabled": "true",
                  "commerce.inventory.b2b.enabled": "true"})   # organizations は OFF
    node = _shop_node(sim)
    cap = goods_mod._capacity(sim.goodscfg, "shop")
    sim._goods_stock[(node, "shop")] = 0
    sim._goods_pending[(node, "shop")] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    assert not _kind(sim, "b2b_trade")
    assert _kind(sim, "restock") and sim._goods_stock[(node, "shop")] == cap


def test_b2b_deterministic(tmp_path):
    """b2b ON 同士 2 回で L1 完全一致(生産・仕入・清算は全決定論・乱数ゼロ)。"""
    a = _sim(tmp_path, "b2bdet_a", steps=30, **{**_B2B_ON, **_ISL})
    a.run()
    b = _sim(tmp_path, "b2bdet_b", steps=30, **{**_B2B_ON, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "b2b ON の決定論が崩れている"


def test_b2b_flow_aggregation(tmp_path):
    """streaming 集計 b2b_flow が取引件数・数量・金額・卸別を有界1パスで出す。"""
    ev = [
        _ev(0, -1, "b2b_trade", from_org="co_ap_01", to_poi="n1", cat="shop", qty=30, amount=9000),
        _ev(1, -1, "b2b_trade", from_org="co_ap_01", to_poi="n2", cat="shop", qty=10, amount=3000),
        _ev(2, -1, "b2b_trade", from_org="co_ca_06", to_poi="n3", cat="food", qty=20, amount=5000),
    ]
    rd = tmp_path / "b2bagg"
    _write_run(rd, ev)
    out = st.b2b_flow(str(rd))
    assert out["n_trade"] == 3 and out["qty_total"] == 60
    assert out["amount_total"] == 17000.0
    assert out["by_org"]["co_ap_01"] == {"qty": 40, "amount": 12000.0}
    assert out["by_org"]["co_ca_06"] == {"qty": 20, "amount": 5000.0}


# =========================================================================== #
# B3(第118 レーンC): 棚と到着待ちの発注を checkpoint で運ぶ
#
#   卸台帳 `_b2b` は保存されているのに、**小売の棚**(`_goods_stock`)と**到着待ちの
#   発注**(`_goods_pending`)だけが未保存だった(非対称な既存欠陥)。goods.py は
#   「棚が無ければ容量 S から始める」遅延初期化なので、保存しないと resume 直後に
#   全 POI の在庫が満杯へ回復し、品切れも進行中の発注も丸ごと消える。
# =========================================================================== #
#: 棚を薄く(S=3 / s=2)して品切れを起こしやすくし、リードタイムを長く(12 step)して
#  「到着待ちの発注が checkpoint を跨ぐ」状態を必ず作る。日次レビューは 1 日 1 回なので
#  split=127 は「レビュー直後 = 61 件が到着待ち」のど真ん中(下で空回り防止を検査する)。
_INV_THIN = {"commerce.inventory.enabled": "true",
             "commerce.inventory.capacity": "{food: 3, cafe: 3, shop: 3, nightlife: 3}",
             "commerce.inventory.reorder_point": "{food: 2, cafe: 2, shop: 2, nightlife: 2}",
             "commerce.inventory.default_capacity": "3",
             "commerce.inventory.default_reorder_point": "2",
             "commerce.inventory.lead_time_steps": "12"}
_INV_SPLIT, _INV_TOTAL = 127, 144


def _cfg_of(name, n_steps, every=0):
    ov = dict(_INV_THIN)
    if every:
        ov["observer.checkpoint_every"] = str(every)
    dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _inv_rows(run_dir):
    return pq.read_table(run_dir / "l1_events.parquet").to_pylist()


def _purse(sim):
    return [round(float(a.money), 6) for a in sim.agents]


def test_resume_carries_the_shelf_and_the_pending_orders(tmp_path):
    """★分割走行 == 一気通し: 減耗中・品切れ中・到着待ち中の 3 状態を跨いで一致する。

    L1(全行)・棚・到着待ち・全員の所持金まで一致させる。保存が無ければ resume 直後に
    棚が満杯へ戻り、品切れ(stock_out)が消え、発注が宙に消えて翌日もう一度発注される。
    """
    from society.engine import checkpoint

    st_dir = tmp_path / "inv_st"
    straight = Simulation(_cfg_of("inv_st", _INV_TOTAL), out_dir=st_dir)
    straight.run()

    d = tmp_path / "inv_rs"
    s1 = Simulation(_cfg_of("inv_rs", _INV_SPLIT, every=_INV_SPLIT), out_dir=d)
    for step in range(_INV_SPLIT):
        scheduler.run_step(s1, step)
    # ---- 空回り防止: 3 状態が **同時に** 立っていること ----
    cap = goods_mod._capacity(s1.goodscfg, "food")
    assert any(v == 0 for v in s1._goods_stock.values()), "品切れ中の棚が 1 つも無い"
    assert any(0 < v < cap for v in s1._goods_stock.values()), "減耗中の棚が 1 つも無い"
    assert s1._goods_pending, "到着待ちの発注が 1 件も無い(split 位置の再調整が要る)"
    checkpoint.save(s1, _INV_SPLIT,
                    d / "checkpoint" / f"ckpt-{_INV_SPLIT:06d}.pkl.gz")
    s1.logger.flush_segment()

    s2 = Simulation(_cfg_of("inv_rs", _INV_TOTAL, every=_INV_SPLIT), out_dir=d)
    s2.run(resume_from=d)

    assert _inv_rows(d) == _inv_rows(st_dir), "inventory ON の resume が straight と不一致"
    assert s2._goods_stock == straight._goods_stock, "棚が一気通しと食い違う"
    assert s2._goods_pending == straight._goods_pending, "到着待ちが一気通しと食い違う"
    assert _purse(s2) == _purse(straight), "所持金が一気通しと食い違う"


def test_checkpoint_roundtrips_the_shelf_and_pending(tmp_path):
    """棚と発注が往復で **1 件残らず** 戻る(タプルキーのまま = sorted の決定論を保つ)。"""
    from society.engine import checkpoint

    src = _sim(tmp_path, "shelf_src", **{"commerce.inventory.enabled": "true"})
    node = _food_node(src)
    src._goods_stock = {(node, "food"): 0, (node, "cafe"): 7}
    src._goods_pending = {(node, "food"): 41}
    p = checkpoint.save(src, 5, tmp_path / "shelf_src" / "checkpoint" / "ckpt-000005.pkl.gz")

    dst = _sim(tmp_path, "shelf_dst", **{"commerce.inventory.enabled": "true"})
    assert checkpoint.load(dst, p) == 5
    assert dst._goods_stock == {(node, "food"): 0, (node, "cafe"): 7}
    assert dst._goods_pending == {(node, "food"): 41}
    assert all(isinstance(k, tuple) for k in dst._goods_stock), "キーがタプルで戻っていない"


def test_old_checkpoint_without_the_shelf_keys_still_loads(tmp_path):
    """新キーを持たない旧 checkpoint でも落ちず、従来どおり遅延初期化へ落ちる。"""
    import gzip
    import pickle

    from society.engine import checkpoint

    src = _sim(tmp_path, "shelf_old_src", **{"commerce.inventory.enabled": "true"})
    src._goods_stock = {(_food_node(src), "food"): 1}
    p = checkpoint.save(src, 3,
                        tmp_path / "shelf_old_src" / "checkpoint" / "ckpt-000003.pkl.gz")
    with gzip.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    for k in ("goods_stock", "goods_pending"):
        blob["runtime"].pop(k, None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _sim(tmp_path, "shelf_old_dst", **{"commerce.inventory.enabled": "true"})
    assert checkpoint.load(dst, p) == 3
    assert dst._goods_stock == {} and dst._goods_pending == {}   # 従来の既定
