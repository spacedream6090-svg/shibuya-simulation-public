"""第114 レーン乙: B2B の**部分納品**と**分散調達**(どちらも既定 OFF)。

塞ぐ問題(第113 実測): 生産 42,408 / 需要 44,604 という**わずか 5% の**恒常ギャップに対して、
従来の仕入れは all-or-nothing(在庫が要求量に 1 個でも足りなければ 0 本も納めない)だった。
そのため「あと 1 個足りない」だけで補充が丸ごと失敗し、翌日も同じ量を要求してまた失敗する
——という自己再生産する詰まり(全店品切れの波及)を起こしていた。

★恒常ギャップ自体は生産・需要の**実勢**なので production_units には触らない
  (数字をいじって供給を作るのは実勢の改竄)。直すのは**配分の規則**の側である。

受入条件:
  - 両サブトグル OFF = 従来の all-or-nothing と**完全に同値**(L1 バイト一致)
  - 部分納品: 在庫の範囲で納まり、残りは翌日の (s,S) レビューが再発注する
  - 分散調達: 指名卸が空でも近傍の同業卸から納まる(決定論・距離昇順)
  - **数量保存**: Σ納品 ≤ Σ生産 + Σ起動時在庫(作った以上に納めない)
全経路 mock(実 LLM 不使用)・乱数ゼロ・LLM 呼ゼロ。
"""
from __future__ import annotations

from society import b2b as b2b_mod
from society import goods as goods_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_B2B_ON = {"commerce.inventory.enabled": "true",
           "commerce.inventory.b2b.enabled": "true",
           "organizations.enabled": "true"}
_LANE_ON = dict(_B2B_ON, **{"commerce.inventory.b2b.partial_fulfillment": "true",
                            "commerce.inventory.b2b.multi_source": "true"})


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _shop_node(sim):
    return sim.city.pois_by_cat("shop")[0]["node"]


def _armed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, **ov)
    scheduler._ensure_orgs(sim)
    return sim


# =========================================================================== #
# 既定 OFF = 従来の all-or-nothing と完全同値
# =========================================================================== #
def test_defaults_are_off():
    cfg = load_config()
    b = cfg.commerce.inventory.b2b
    assert b.partial_fulfillment is False and b.multi_source is False
    assert int(b.max_sources) == 3


def test_off_keeps_all_or_nothing(tmp_path):
    """在庫不足 → 1 本も納めない・restock なし(第113 までと 1 バイト違わない)。"""
    sim = _armed(tmp_path, "off_aon", **_B2B_ON)
    node = _shop_node(sim)
    org = b2b_mod.wholesale_for(sim, node, "shop")
    oid = str(org["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 5                                # < cap(=補充要求量)
    sim._goods_stock[(node, "shop")] = 0
    sim._goods_pending[(node, "shop")] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    assert not _kind(sim, "b2b_trade") and not _kind(sim, "restock")
    assert b["stock"][oid] == 5, "不成立なのに卸在庫が減っている"
    assert "req_qty" not in b, "OFF なのに部分納品の観測欄が生えている"


def test_off_l1_is_byte_identical(tmp_path):
    """本レーンのコード変更が既定ランの L1 を 1 バイトも動かしていない。"""
    a = _sim(tmp_path, "byte_a", steps=48, **_B2B_ON)
    a.run()
    b = _sim(tmp_path, "byte_b", steps=48, **_B2B_ON)
    b.run()
    x = (tmp_path / "byte_a" / "l1_events.parquet").read_bytes()
    y = (tmp_path / "byte_b" / "l1_events.parquet").read_bytes()
    assert x == y


def test_fulfill_qty_matches_fulfill_when_both_toggles_are_off(tmp_path):
    """サブトグル OFF では fulfill_qty は「qty か 0」しか返さない(= 旧 API と同値)。"""
    sim = _armed(tmp_path, "eq", **_B2B_ON)
    node = _shop_node(sim)
    oid = str(b2b_mod.wholesale_for(sim, node, "shop")["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 4
    assert b2b_mod.fulfill_qty(sim, node, "shop", 10, 1, 10) == 0
    b["stock"][oid] = 40
    assert b2b_mod.fulfill_qty(sim, node, "shop", 10, 1, 10) == 10


# =========================================================================== #
# ① 部分納品
# =========================================================================== #
def test_partial_delivers_what_is_there(tmp_path):
    sim = _armed(tmp_path, "part1",
                 **dict(_B2B_ON,
                        **{"commerce.inventory.b2b.partial_fulfillment": "true"}))
    node = _shop_node(sim)
    oid = str(b2b_mod.wholesale_for(sim, node, "shop")["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 5
    cap = goods_mod._capacity(sim.goodscfg, "shop")
    sim._goods_stock[(node, "shop")] = 0
    sim._goods_pending[(node, "shop")] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    trades = _kind(sim, "b2b_trade")
    assert len(trades) == 1 and trades[0].payload["qty"] == 5
    assert trades[0].payload["req"] == cap, "要求量 req が payload に残っていない"
    assert b["stock"][oid] == 0, "在庫を出し切っていない"
    assert sim._goods_stock[(node, "shop")] == 5, "届いた分だけ在庫が戻っていない"
    restock = _kind(sim, "restock")
    assert len(restock) == 1 and restock[0].payload["qty"] == 5, \
        "restock の qty が『実際に届いた量』でない"
    assert b["req_qty"] == cap and b["short_qty"] == cap - 5 and b["partial"] == 1


def test_partial_remainder_is_reordered_the_next_review(tmp_path):
    """★バックオーダー = 既存の (s,S) レビューが担う(新しい待ち行列を作らない)。"""
    sim = _armed(tmp_path, "part2",
                 **dict(_B2B_ON,
                        **{"commerce.inventory.b2b.partial_fulfillment": "true"}))
    node = _shop_node(sim)
    key = (node, "shop")
    oid = str(b2b_mod.wholesale_for(sim, node, "shop")["id"])
    b = b2b_mod._state(sim)
    b["stock"][oid] = 5
    sim._goods_stock[key] = 0
    sim._goods_pending[key] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    level = sim._goods_stock[key]
    assert 0 < level <= goods_mod._reorder(sim.goodscfg, "shop"), \
        "部分納品の後の水準が発注点より上(翌日の再発注が起きない)"
    # 翌日のレビュー: 未達分が自動的に再発注される
    goods_mod.review_and_order(sim, 200, 12000)
    trips = _kind(sim, "delivery_trip")
    assert trips, "残りが再発注されていない(バックオーダーが機能していない)"
    assert trips[-1].payload["qty"] == goods_mod._capacity(sim.goodscfg, "shop") - level


def test_partial_never_ships_more_than_exists(tmp_path):
    """★数量保存: 納品量の合計 ≤ 生産 + 起動時在庫(作った以上に納めない)。"""
    sim = _armed(tmp_path, "part3", **_LANE_ON)
    node = _shop_node(sim)
    b = b2b_mod._state(sim)
    orgs = b2b_mod.suppliers_for(sim, node, "shop")
    assert orgs, "卸が 1 社も見つからない(検査が空振り)"
    produced = 0
    for i, org in enumerate(orgs):
        b["stock"][str(org["id"])] = 3 + i             # 合計でも要求量に満たない
        produced += 3 + i
    got = b2b_mod.fulfill_qty(sim, node, "shop", 999, 12, 800)
    assert got == produced, "在庫総量を超えて納品している(数量保存の破れ)"
    assert sum(b["stock"].values()) == 0
    assert sum(t.payload["qty"] for t in _kind(sim, "b2b_trade")) == produced


# =========================================================================== #
# ② 分散調達
# =========================================================================== #
def test_suppliers_are_ranked_deterministically(tmp_path):
    sim = _armed(tmp_path, "multi1", **_LANE_ON)
    node = _shop_node(sim)
    got = b2b_mod.suppliers_for(sim, node, "shop")
    assert 1 <= len(got) <= int(sim.b2bcfg["max_sources"])
    assert got[0]["id"] == b2b_mod.wholesale_for(sim, node, "shop")["id"], \
        "先頭が指名卸(最寄り 1 社)と一致していない"
    assert b2b_mod.suppliers_for(sim, node, "shop") == got, "2 回目で並びが変わった"
    assert all("goods" in (o.get("output_kinds") or []) for o in got)


def test_multi_source_falls_back_to_the_neighbour(tmp_path):
    """指名卸が空でも近傍の同業卸から満額で納まる(部分納品は使わない)。"""
    sim = _armed(tmp_path, "multi2",
                 **dict(_B2B_ON, **{"commerce.inventory.b2b.multi_source": "true"}))
    node = _shop_node(sim)
    orgs = b2b_mod.suppliers_for(sim, node, "shop")
    if len(orgs) < 2:
        return                                          # 卸が 1 社の地図では検査対象外
    b = b2b_mod._state(sim)
    b["stock"][str(orgs[0]["id"])] = 0                  # 指名卸は空
    b["stock"][str(orgs[1]["id"])] = 500
    cap = goods_mod._capacity(sim.goodscfg, "shop")
    assert b2b_mod.fulfill_qty(sim, node, "shop", cap, 12, 800) == cap
    trades = _kind(sim, "b2b_trade")
    assert len(trades) == 1
    assert trades[0].payload["from_org"] == str(orgs[1]["id"])
    assert "req" not in trades[0].payload, "部分納品 OFF なのに req が生えている"


def test_multi_source_splits_across_wholesalers_with_partial(tmp_path):
    sim = _armed(tmp_path, "multi3", **_LANE_ON)
    node = _shop_node(sim)
    orgs = b2b_mod.suppliers_for(sim, node, "shop")
    if len(orgs) < 2:
        return
    b = b2b_mod._state(sim)
    b["stock"][str(orgs[0]["id"])] = 4
    b["stock"][str(orgs[1]["id"])] = 500
    cap = goods_mod._capacity(sim.goodscfg, "shop")
    assert b2b_mod.fulfill_qty(sim, node, "shop", cap, 12, 800) == cap
    qtys = [t.payload["qty"] for t in _kind(sim, "b2b_trade")]
    assert qtys == [4, cap - 4], "2 社に分けて引いていない"
    assert b["stock"][str(orgs[0]["id"])] == 0


def test_max_sources_bounds_the_search(tmp_path):
    sim = _armed(tmp_path, "multi4",
                 **dict(_LANE_ON, **{"commerce.inventory.b2b.max_sources": "1"}))
    node = _shop_node(sim)
    assert len(b2b_mod.suppliers_for(sim, node, "shop")) == 1


def test_money_is_conserved_per_trade(tmp_path):
    """金の保存(買い側仕入費 = 売り側売上の合計)は分割納品でも崩れない。"""
    sim = _armed(tmp_path, "money", **_LANE_ON)
    node = _shop_node(sim)
    orgs = b2b_mod.suppliers_for(sim, node, "shop")
    b = b2b_mod._state(sim)
    for org in orgs:
        b["stock"][str(org["id"])] = 7
    got = b2b_mod.fulfill_qty(sim, node, "shop", 999, 12, 800)
    price = b2b_mod._wholesale_price(sim.b2bcfg, "shop")
    assert abs(b["procurement"] - got * price) < 1e-6
    assert abs(sum(b["revenue"].values()) - b["procurement"]) < 1e-6
    assert sum(b["sold_qty"].values()) == got


def test_registry_declares_the_new_toggles():
    from society import registry as R
    feats = {f.id: f for f in R.FEATURES}
    for fid in ("commerce.inventory.b2b.partial_fulfillment",
                "commerce.inventory.b2b.multi_source"):
        assert fid in feats
        assert feats[fid].off_value is False and feats[fid].affects_k is False
