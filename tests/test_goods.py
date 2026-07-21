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
