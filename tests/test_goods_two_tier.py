"""2層在庫 INV-A/B/C(棚+バックヤード・店員の棚出し・店主の発注・便数conf・来店時の棚知覚)のテスト。

正典: docs/plans/inventory-two-tier-plan.md §1(INV-A/B/C + 来店時の台帳参照)・§3(較正アンカー)・§4(検収)。
リサーチ: docs/research/retail-inventory-empirics.md。

設計原則(ユーザードクトリン): エンジンが世界を規定するのではなく、**世界の状態(棚が減った)は
エージェントの行動トリガーであり、変化(棚が埋まる・発注される)は当人の行動の結果**として起きる。
エンジン規則による代替再現は「店員/店主が 1 人も割り当てられていない POI」に限り、conf で宣言した
ときだけ通す(カバー率を観測する)。

R1 の鉄則を継承:
- OFF(既定): L1 が **inventory ON の従来挙動とバイト一致**(BY 台帳も棚薄フラグも生えない)。
- ON: (a)購入は棚だけを削る・納品は BY へ入る。(b)**棚⇄BY 保存則**(Σ=棚+BY+累積消費−累積納品が不変)。
      (c)補充境界(店員不在 / BY 空 / 棚満杯)。(d)店主の発注(欠勤なら発注されない)。
      (e)未カバー POI のみエンジン代替(unstaffed=true)。(f)INV-C 便数 conf。
      (g)resume == straight(BY・棚薄フラグ・レビュー窓の搬送)。(h)同 seed 2 ランで L1 一致。
      (i)LLM 呼数不変(k 非依存)。(j)来店時の棚知覚は非潤沢のときだけ 1 行。(k)L1 は 1 店 1 補充。
検証は mock のみ(実LLM 禁止・≤48 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

from society import goods as goods_mod
from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer import stream as st
from society.observer.causality import CAUSE_OF_KIND, AGENT
from society.observer.schema import EVENT_KINDS

_INV = {"commerce.inventory.enabled": "true"}
_TT = {**_INV, "commerce.inventory.two_tier.enabled": "true"}
_ISL = {"prompts.interstitial.enabled": "true"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
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


def _clear_all_workplaces(sim):
    """全員の職場割当を外す(= どの POI も『担い手ゼロ』= フォールバックの検査条件)。"""
    for a in sim.agents:
        a.work_node = ""
        a.work_start_min = -1
    sim._goods_staffed = None
    sim._goods_staffed_day = -1


def _staff(sim, node, agent, sim_min=700):
    """agent をその店の在勤者にする(既存の職場割当そのもの=新しい在勤概念を作らない)。"""
    _clear_all_workplaces(sim)
    agent.work_node = node
    agent.work_building = ""
    agent.work_start_min = 0
    agent.work_end_min = 1439
    agent.node = node
    agent.x, agent.y = sim.city.node_xy(node)
    agent.loc = "in"
    agent.sleeping = False
    sim._goods_staffed = None
    sim._goods_staffed_day = -1
    return agent


# ------------------------------------------------------ 登録(schema + 因果台帳の 2 箇所)
def test_schema_and_causality_registered():
    for k in ("shelf_restock", "stock_order"):
        assert k in EVENT_KINDS, f"{k} が schema 未登録"
        assert CAUSE_OF_KIND[k] == AGENT, f"{k} は**当人の行為**として分類されていない"


# ------------------------------------------------------ OFF: 従来挙動とバイト一致
def test_two_tier_off_matches_inventory_only(tmp_path):
    """明示 OFF と「inventory ON だけ」が L1 完全一致(2層 seam が no-op)。"""
    base = _sim(tmp_path, "tt_base", steps=48, **_INV)
    base.run()
    off = _sim(tmp_path, "tt_off", steps=48,
               **{**_INV, "commerce.inventory.two_tier.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "two_tier OFF が従来挙動と不一致(seam が no-op でない)"
    for k in ("shelf_restock", "stock_order"):
        assert not _kind(base, k) and not _kind(off, k)
    assert not base._goods_back, "OFF なのに BY 台帳が実体化している"
    assert not base._goods_low, "OFF なのに棚薄フラグが立っている"
    assert not base._goods_order_win, "OFF なのに店主のレビュー窓が生えている"


def test_two_tier_off_matches_pure_default(tmp_path):
    """在庫ごと OFF(純粋既定)とも L1 完全一致(ゴールデンの世界を 1 バイトも動かさない)。"""
    pure = _sim(tmp_path, "tt_pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "tt_expl", steps=48,
               **{"commerce.inventory.enabled": "false",
                  "commerce.inventory.two_tier.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off)


def test_review_every_steps_default_is_the_current_daily(tmp_path):
    """INV-C の既定 0 は現行の日次レビューと完全同値(L1 バイト一致)。"""
    a = _sim(tmp_path, "rev_a", steps=48, **_INV)
    a.run()
    b = _sim(tmp_path, "rev_b", steps=48,
             **{**_INV, "commerce.inventory.review_every_steps": "0"})
    b.run()
    assert _l1(a) == _l1(b)


# ------------------------------------------------------ (a) 購入は棚のみ・納品は BY へ
def test_purchase_only_touches_the_shelf(tmp_path):
    """購入は棚だけを削る(BY は 1 個も動かない)+ 補充点を割ると棚薄フラグが立つ。"""
    sim = _sim(tmp_path, "tt_buy", **_TT)
    node = _food_node(sim)
    a = sim.agents[0]
    a.node = node
    cap = goods_mod._capacity(sim.goodscfg, "food")
    rp = goods_mod._shelf_restock_point(sim.goodscfg, "food")
    sim._goods_stock[(node, "food")] = rp + 1
    sim._goods_back[(node, "food")] = 5
    ok, _item = goods_mod.on_purchase(sim, a, "food", 3, 700)
    assert ok is True
    assert sim._goods_stock[(node, "food")] == rp          # 棚が 1 減った
    assert sim._goods_back[(node, "food")] == 5, "購入で BY が動いた(2層になっていない)"
    assert node not in sim._goods_low, "補充点ちょうどでフラグが立った(閾値は『未満』)"
    ok, _item = goods_mod.on_purchase(sim, a, "food", 4, 710)   # ここで補充点を割る
    assert ok is True and sim._goods_stock[(node, "food")] == rp - 1
    assert sim._goods_low.get(node) == {"food"}, "補充点を割ったのにフラグが立たない"
    assert sim._goods_back[(node, "food")] == 5
    assert cap > rp


def test_stockout_even_though_the_backyard_is_full(tmp_path):
    """★2層の核: BY が満杯でも棚が空なら購入不成立(Gruen「店内に在るが棚に無い」)。"""
    sim = _sim(tmp_path, "tt_gruen", **_TT)
    node = _food_node(sim)
    a = sim.agents[0]
    a.node = node
    sim._goods_stock[(node, "food")] = 0
    sim._goods_back[(node, "food")] = 999
    ok, item = goods_mod.on_purchase(sim, a, "food", 3, 700)
    assert ok is False and item is None, "BY 在庫で棚の品切れが埋まってしまっている"
    assert [e for e in _kind(sim, "stock_out") if e.payload.get("src") == "inventory"]


def test_delivery_lands_in_the_backyard_not_on_the_shelf(tmp_path):
    """納品は BY へ入る(棚は 1 個も増えない)。restock payload に tier=back が付く。"""
    sim = _sim(tmp_path, "tt_arrive", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    cap = goods_mod._capacity(sim.goodscfg, "food")
    bcap = goods_mod._back_capacity(sim.goodscfg, "food")
    sim._goods_stock[key] = 0
    sim._goods_pending[key] = 12
    goods_mod.deliver_arrivals(sim, 12, 800)
    rs = _kind(sim, "restock")
    assert len(rs) == 1 and rs[0].payload["tier"] == "back"
    assert sim._goods_stock[key] == 0, "納品が棚へ直に乗った(2層になっていない)"
    assert sim._goods_back[key] == cap + bcap, "BY へ order-up-to 分が入っていない"


def test_back_capacity_and_restock_point_are_integers(tmp_path):
    """BY 容量=棚容量×back_ratio・棚の補充点=棚容量×比。どちらも整数のみ(端数の持ち越しなし)。"""
    sim = _sim(tmp_path, "tt_int", **{**_TT,
                                      "commerce.inventory.two_tier.back_ratio":
                                          "{food: 0.3, shop: 1.5}"})
    cfg = sim.goodscfg
    assert goods_mod._back_capacity(cfg, "food") == int(40 * 0.3) == 12
    assert goods_mod._back_capacity(cfg, "shop") == int(30 * 1.5) == 45
    assert goods_mod._back_capacity(cfg, "cafe") == int(40 * 0.3)      # 未掲載=既定比
    for cat in ("food", "shop", "cafe", "nightlife"):
        assert isinstance(goods_mod._back_capacity(cfg, cat), int)
        assert isinstance(goods_mod._shelf_restock_point(cfg, cat), int)


# ------------------------------------------------------ (b) 棚⇄BY 保存則
def _total(sim, key):
    return int(sim._goods_stock.get(key, 0)) + int(sim._goods_back.get(key, 0))


def test_shelf_and_backyard_conserve_units(tmp_path):
    """★Σ = 棚 + BY + 累積消費 − 累積納品 が動かない(棚出しはどこにも湧かせない・失わせない)。"""
    sim = _sim(tmp_path, "tt_cons", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    sim._goods_stock[key] = 4
    sim._goods_back[key] = 0
    consumed, delivered = 0, 0
    base = _total(sim, key) + consumed - delivered
    buyer = sim.agents[1]
    buyer.node = node
    # 買う(消費)
    for i in range(3):
        ok, _ = goods_mod.on_purchase(sim, buyer, "food", i, 700 + i)
        if ok:
            consumed += sim.goodscfg["consume_units"]
        assert _total(sim, key) + consumed - delivered == base
    # 納品(BY へ)
    sim._goods_pending[key] = 10
    before = _total(sim, key)
    goods_mod.deliver_arrivals(sim, 10, 720)
    delivered += _total(sim, key) - before
    assert _total(sim, key) + consumed - delivered == base
    # 棚出し(BY → 棚。総量は動かない)
    moved = goods_mod._restock_store(sim, node, a, 11, 730)
    assert moved > 0, "テスト前提が崩れた(棚出しが 1 個も起きていない)"
    assert _total(sim, key) + consumed - delivered == base, "棚出しで総量が変わった"


def test_restock_moves_exactly_between_the_two_tiers(tmp_path):
    """棚出しは BY から引いた分だけ棚へ積む(1 対 1)。"""
    sim = _sim(tmp_path, "tt_move", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    sim._goods_stock[key] = 2
    sim._goods_back[key] = 7
    goods_mod._note_shelf(sim, node, "food", 2)
    moved = goods_mod._restock_store(sim, node, a, 5, 700)
    assert moved == 7                                       # BY 全量が棚へ(棚に余地あり)
    assert sim._goods_stock[key] == 9 and key not in sim._goods_back


# ------------------------------------------------------ (c) 補充境界(不在 / BY 空 / 棚満杯)
def test_no_staff_no_restock(tmp_path):
    """★店員が居なければ棚は埋まらない(BY に在っても)= 世界の状態が当人の事情に依存する。"""
    sim = _sim(tmp_path, "tt_nostaff", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = sim.agents[0]
    _staff(sim, node, a)
    a.work_start_min, a.work_end_min = 600, 700             # いまは勤務時間外
    sim._goods_stock[key] = 0
    sim._goods_back[key] = 20
    goods_mod._note_shelf(sim, node, "food", 0)
    goods_mod.staff_phase(sim, 5, 60)                       # 01:00 = 誰も出勤していない
    assert not _kind(sim, "shelf_restock"), "在勤者が居ないのに棚が埋まった"
    assert sim._goods_stock[key] == 0 and sim._goods_back[key] == 20
    assert sim._goods_low.get(node) == {"food"}, "フラグは立ったまま(次の出勤で埋まる)"


def test_empty_backyard_cannot_restock(tmp_path):
    """BY が空なら店員が居ても 1 個も動かない(= 正しい欠品。イベントも出さない)。"""
    sim = _sim(tmp_path, "tt_byempty", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    sim._goods_stock[key] = 0
    sim._goods_back.pop(key, None)
    goods_mod._note_shelf(sim, node, "food", 0)
    assert goods_mod._restock_store(sim, node, a, 5, 700) == 0
    assert not _kind(sim, "shelf_restock")
    assert sim._goods_stock[key] == 0


def test_full_shelf_clears_the_flag_without_an_event(tmp_path):
    """棚が満杯なら補充不要=フラグだけ落ちてイベントは出ない(L1 の洪水を作らない)。"""
    sim = _sim(tmp_path, "tt_full", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    cap = goods_mod._capacity(sim.goodscfg, "food")
    sim._goods_stock[key] = cap
    sim._goods_back[key] = 10
    goods_mod._set_low(sim, node, "food")                   # 古いフラグが残っていた状況
    assert goods_mod._restock_store(sim, node, a, 5, 700) == 0
    assert not _kind(sim, "shelf_restock")
    assert node not in sim._goods_low
    assert sim._goods_back[key] == 10


def test_restock_batch_bounds_one_action(tmp_path):
    """restock_batch は 1 回の棚出しで運ぶ量を上限する(0=棚を満たすまで)。"""
    sim = _sim(tmp_path, "tt_batch",
               **{**_TT, "commerce.inventory.two_tier.restock_batch": "3"})
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    sim._goods_stock[key] = 0
    sim._goods_back[key] = 20
    goods_mod._note_shelf(sim, node, "food", 0)
    assert goods_mod._restock_store(sim, node, a, 5, 700) == 3
    assert sim._goods_stock[key] == 3 and sim._goods_back[key] == 17


# ------------------------------------------------------ INV-B(a) 補充は**当人の行動**
def test_restock_is_the_staff_members_own_action(tmp_path):
    """在勤の店員が居る店では、その個体の L1 行動として shelf_restock が出る + 本人の記憶に残る。"""
    sim = _sim(tmp_path, "tt_act", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    rp = goods_mod._shelf_restock_point(sim.goodscfg, "food")
    sim._goods_stock[key] = 0
    sim._goods_back[key] = rp + 5
    goods_mod._note_shelf(sim, node, "food", 0)
    goods_mod.staff_phase(sim, 7, 700)
    ev = _kind(sim, "shelf_restock")
    assert len(ev) == 1
    assert ev[0].agent_id == a.id, "棚出しが当人の行為として記録されていない"
    assert "unstaffed" not in ev[0].payload
    assert ev[0].payload["poi"] == node and ev[0].payload["cats"] == ["food"]
    assert ev[0].payload["qty"] == rp + 5 and ev[0].payload["n"] == 1
    assert ev[0].x == a.x and ev[0].y == a.y
    assert goods_mod._RESTOCK_TEXT in a.mem.recent(4), \
        "当人の記憶に残っていない(既存の work 文脈へ載らない)"
    assert sim._goods_stock[key] == rp + 5
    assert node not in sim._goods_low, "棚が補充点を越えたのにフラグが残っている"
    assert key not in sim._goods_back, "BY を使い切ったのにキーが残っている"


def test_one_restock_event_per_store_per_trigger(tmp_path):
    """★L1 体積: 1 店の複数カテゴリをまとめて 1 件・フラグが落ちれば次 step は 0 件。"""
    sim = _sim(tmp_path, "tt_vol", **_TT)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    cats = [str(p.get("cat")) for p in sim.city.pois_at_node(node)]
    for cat in cats:
        sim._goods_stock[(node, cat)] = 0
        sim._goods_back[(node, cat)] = 9
        goods_mod._note_shelf(sim, node, cat, 0)
    goods_mod.staff_phase(sim, 7, 700)
    ev = _kind(sim, "shelf_restock")
    assert len(ev) == 1, "1 店 1 補充になっていない(イベント洪水)"
    assert ev[0].payload["n"] == len(set(cats))
    goods_mod.staff_phase(sim, 8, 710)                      # フラグは落ちている
    assert len(_kind(sim, "shelf_restock")) == 1, "フラグが落ちたのに毎 step 補充している"


# ------------------------------------------------------ INV-B(c) 未カバー POI のみ代替再現
def test_unstaffed_fallback_only_for_poi_without_any_worker(tmp_path):
    """担い手が 1 人も割り当てられていない POI だけエンジンが代替再現(unstaffed=true)。"""
    sim = _sim(tmp_path, "tt_unstaffed", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    _clear_all_workplaces(sim)                              # どの POI にも担い手が居ない
    sim._goods_stock[key] = 0
    sim._goods_back[key] = 6
    goods_mod._note_shelf(sim, node, "food", 0)
    goods_mod.staff_phase(sim, 7, 700)
    ev = _kind(sim, "shelf_restock")
    assert len(ev) == 1 and ev[0].agent_id == -1
    assert ev[0].payload["unstaffed"] is True, "無人の補充が黙って個体の行為に化けている"
    assert sim._goods_stock[key] == 6
    assert sim._goods_ops["restock_unstaffed"] == 1 and sim._goods_ops["restock_staffed"] == 0


def test_assigned_but_absent_staff_gets_no_fallback(tmp_path):
    """★担い手が居る店は、その人が休んでいてもエンジンが肩代わりしない(欠品が本人の事情の帰結)。"""
    sim = _sim(tmp_path, "tt_absent", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = sim.agents[0]
    _staff(sim, node, a)
    a.loc = "outside"                                       # 割当はあるが街に居ない
    sim._goods_stock[key] = 0
    sim._goods_back[key] = 6
    goods_mod._note_shelf(sim, node, "food", 0)
    goods_mod.staff_phase(sim, 7, 700)
    assert not _kind(sim, "shelf_restock"), "担い手の欠勤をエンジンが肩代わりした"
    assert sim._goods_stock[key] == 0


def test_fallback_can_be_declined(tmp_path):
    """unstaffed_fallback=false なら未カバー POI も補充されない(黙って無人で動かさない)。"""
    sim = _sim(tmp_path, "tt_nofb",
               **{**_TT, "commerce.inventory.two_tier.unstaffed_fallback": "false"})
    node = _food_node(sim)
    key = (node, "food")
    _clear_all_workplaces(sim)
    sim._goods_stock[key] = 0
    sim._goods_back[key] = 6
    goods_mod._note_shelf(sim, node, "food", 0)
    goods_mod.staff_phase(sim, 7, 700)
    assert not _kind(sim, "shelf_restock")
    assert sim._goods_stock[key] == 0


# ------------------------------------------------------ INV-B(b) 発注は**店主の行動**
def test_owner_places_the_order_while_on_duty(tmp_path):
    """在勤の店主が自店の在庫を確かめて発注する(stock_order = 当人の行為 + 既存の発注経路)。"""
    sim = _sim(tmp_path, "tt_order", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = _staff(sim, node, sim.agents[0])
    sim._goods_stock[key] = 0
    goods_mod.staff_phase(sim, 7, 700)
    orders = _kind(sim, "stock_order")
    assert len(orders) == 1 and orders[0].agent_id == a.id
    assert orders[0].payload["poi"] == node and orders[0].payload["n"] >= 1
    assert _kind(sim, "delivery_trip") and _kind(sim, "stock_low")
    assert key in sim._goods_pending
    # 同じ窓(同じ日)では二度見ない
    goods_mod.staff_phase(sim, 8, 710)
    assert len(_kind(sim, "stock_order")) == 1


def test_absent_owner_means_no_order(tmp_path):
    """★店主が出勤しなければ発注されない(品切れが本人の事情の帰結になる)。"""
    sim = _sim(tmp_path, "tt_noorder", **_TT)
    node = _food_node(sim)
    key = (node, "food")
    a = sim.agents[0]
    _staff(sim, node, a)
    a.sleeping = True                                       # 起きていない=在勤でない
    sim._goods_stock[key] = 0
    goods_mod.staff_phase(sim, 7, 700)
    assert not _kind(sim, "stock_order") and not _kind(sim, "delivery_trip")
    assert key not in sim._goods_pending


def test_engine_review_skips_the_stores_that_have_an_owner(tmp_path):
    """エンジンの定期レビューは担い手の居る店を走査しない(発注は当人の行動へ移った)。"""
    sim = _sim(tmp_path, "tt_skip", **_TT)
    node = _food_node(sim)
    other = sim.city.pois_by_cat("shop")[0]["node"]
    _staff(sim, node, sim.agents[0])                        # node だけ担い手あり
    sim._goods_stock[(node, "food")] = 0
    sim._goods_stock[(other, "shop")] = 0
    goods_mod.review_and_order(sim, 9, 700)
    tos = {e.payload["to"] for e in _kind(sim, "delivery_trip")}
    assert other in tos, "未カバー POI の代替発注が出ていない"
    assert node not in tos, "担い手の居る店をエンジンが発注した(行動化されていない)"
    assert sim._goods_ops["order_unstaffed"] >= 1


def test_engine_review_still_runs_when_staff_order_is_off(tmp_path):
    """staff_order=false なら従来どおりエンジンが一括レビューする(2層だけ使う構成)。"""
    sim = _sim(tmp_path, "tt_engorder",
               **{**_TT, "commerce.inventory.two_tier.staff_order": "false"})
    node = _food_node(sim)
    _staff(sim, node, sim.agents[0])
    sim._goods_stock[(node, "food")] = 0
    goods_mod.review_and_order(sim, 9, 700)
    assert {e.payload["to"] for e in _kind(sim, "delivery_trip")} == {node}
    assert not _kind(sim, "stock_order")


# ------------------------------------------------------ INV-C 便数(レビュー間隔)
def test_review_every_steps_fires_on_the_step_cycle(tmp_path):
    """review_every_steps>0 は step 周期で (s,S) を見る(CVS の 1 日 2-3 便 / SM の隔日)。"""
    sim = _sim(tmp_path, "tt_cadence",
               **{**_INV, "commerce.inventory.review_every_steps": "6"})
    node = _food_node(sim)
    sim._goods_stock[(node, "food")] = 0
    goods_mod.tick(sim, 5, 50)                              # 5 % 6 != 0 = 便が無い
    assert not _kind(sim, "delivery_trip")
    goods_mod.tick(sim, 6, 60)                              # 6 % 6 == 0 = 便が走る
    assert len(_kind(sim, "delivery_trip")) == 1
    assert int(getattr(sim, "_goods_review_day", -1)) == -1, "日次カウンタを汚している"


def test_order_window_follows_review_every_steps(tmp_path):
    """店主のレビュー窓も便数 conf に従う(既定=日 / >0 なら step 周期)。"""
    cfg_daily = goods_mod.build_cfg({"inventory": {"enabled": True}})
    assert goods_mod._order_window(cfg_daily, 5, 700) == 0
    assert goods_mod._order_window(cfg_daily, 5, 1500) == 1
    cfg_step = goods_mod.build_cfg({"inventory": {"enabled": True,
                                                  "review_every_steps": 6}})
    assert goods_mod._order_window(cfg_step, 5, 700) == 0
    assert goods_mod._order_window(cfg_step, 6, 700) == 1


# ------------------------------------------------------ (j) 来店時の棚知覚
def test_shelf_note_only_when_the_shelf_is_not_plentiful(tmp_path):
    """潤沢=1 行も足さない / 残りわずか・空のときだけ 1 行(3値)。"""
    sim = _sim(tmp_path, "tt_percept", **_TT)
    node = _food_node(sim)
    a = sim.agents[0]
    a.node = node
    cap = goods_mod._capacity(sim.goodscfg, "food")
    rp = goods_mod._shelf_restock_point(sim.goodscfg, "food")
    assert goods_mod.shelf_note(sim, a) is None, "未実体化の棚で行が出た"
    sim._goods_stock[(node, "food")] = cap
    assert goods_mod.shelf_note(sim, a) is None, "潤沢なのに行が出た"
    sim._goods_stock[(node, "food")] = rp - 1
    low = goods_mod.shelf_note(sim, a)
    assert low is not None and "残りわずか" in low
    sim._goods_stock[(node, "food")] = 0
    empty = goods_mod.shelf_note(sim, a)
    assert empty is not None and "空" in empty
    assert "\n" not in empty and len(empty) < 120, "1 行に収まっていない"


def test_shelf_note_is_off_by_default(tmp_path):
    """2層 OFF / percept=false では常に None(既定 OFF=1 行も足さない=バイト一致)。"""
    off = _sim(tmp_path, "tt_p_off", **_INV)
    node = _food_node(off)
    off.agents[0].node = node
    off._goods_stock[(node, "food")] = 0
    assert goods_mod.shelf_note(off, off.agents[0]) is None
    nop = _sim(tmp_path, "tt_p_nop",
               **{**_TT, "commerce.inventory.two_tier.percept": "false"})
    nop.agents[0].node = node
    nop._goods_stock[(node, "food")] = 0
    assert goods_mod.shelf_note(nop, nop.agents[0]) is None


def test_material_and_prompt_carry_exactly_one_shelf_line(tmp_path):
    """_gather_material が shelf_line を集め、build_prompt がちょうど 1 行だけ足す。"""
    sim = _sim(tmp_path, "tt_prompt", **_TT)
    node = _food_node(sim)
    a = sim.agents[0]
    a.node = node
    sim._goods_stock[(node, "food")] = 0
    kwargs = scheduler._gather_material(sim, a, "solo", 0, 0)
    assert kwargs["shelf_line"], "material に棚の様子が入っていない"
    line = kwargs.pop("shelf_line")
    base = deliberate.build_prompt(a, **kwargs, shelf_line=None)
    withline = deliberate.build_prompt(a, **kwargs, shelf_line=line)
    assert deliberate.build_prompt(a, **kwargs) == base      # 既定は 1 行も足さない
    assert withline.count("\n") == base.count("\n") + 1
    assert line in withline


def test_perception_contract_round_trips_the_shelf_line(tmp_path):
    """契約経路(cognition.contract)でも無損失で往復する。"""
    sim = _sim(tmp_path, "tt_contract",
               **{**_TT, "cognition.contract.enabled": "true"})
    node = _food_node(sim)
    a = sim.agents[0]
    a.node = node
    sim._goods_stock[(node, "food")] = 0
    material = scheduler._gather_material(sim, a, "solo", 1, 10)
    percept = scheduler.build_perception(sim, a, material)
    assert percept.shelf_line == material["shelf_line"]
    kw = percept.prompt_kwargs()
    assert {k: v for k, v in kw.items() if k in material} == material


# ------------------------------------------------------ (h)(i) 決定論・呼数不変
def test_two_tier_is_deterministic(tmp_path):
    """2層 ON 同士 2 回で L1 完全一致(決定論・乱数不使用)。"""
    a = _sim(tmp_path, "tt_det_a", steps=36, **{**_TT, **_ISL})
    a.run()
    b = _sim(tmp_path, "tt_det_b", steps=36, **{**_TT, **_ISL})
    b.run()
    assert _l1(a) == _l1(b), "2層 ON の決定論が崩れている"
    assert a._goods_back == b._goods_back and a._goods_low == b._goods_low


class _FixedLLM:
    """内容非依存の固定応答 backend(呼数だけを数える)。"""

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
    """2層 ON/OFF で LLM 呼数が完全一致(棚知覚はプロンプト 1 行だけ=追加呼ゼロ=R1)。"""
    on = _run_fixed(tmp_path, "tt_cc_on", **_TT)
    off = _run_fixed(tmp_path, "tt_cc_off", **_INV)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"2層で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


def test_llm_call_count_k_invariant(tmp_path):
    """2層 ON で k=free と k=off の呼数が一致(発火判断に k を食わせない)。"""
    free = _run_fixed(tmp_path, "tt_k_free", **{**_TT, "k.writeback": "free"})
    off = _run_fixed(tmp_path, "tt_k_off", **{**_TT, "k.writeback": "off"})
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0


# ------------------------------------------------------ (g) resume == straight
#: 棚を薄く(S=4 / s=3 / 補充点 2)して棚薄と品切れを起こしやすくし、BY 比を大きく(1.0)して
#  「BY に在庫が残ったまま checkpoint を跨ぐ」状態を必ず作る。便数(INV-C)を 6 step ごとに
#  上げてリードを 3 step にし、127 step のあいだに発注→到着→棚出しが何度も回るようにする。
_TT_THIN = {
    "commerce.inventory.enabled": "true",
    "commerce.inventory.two_tier.enabled": "true",
    "commerce.inventory.two_tier.default_back_ratio": "1.0",
    "commerce.inventory.capacity": "{food: 4, cafe: 4, shop: 4, nightlife: 4}",
    "commerce.inventory.reorder_point": "{food: 3, cafe: 3, shop: 3, nightlife: 3}",
    "commerce.inventory.default_capacity": "4",
    "commerce.inventory.default_reorder_point": "3",
    "commerce.inventory.lead_time_steps": "3",
    "commerce.inventory.review_every_steps": "6",
}
_SPLIT, _TOTAL = 127, 144


def _cfg_of(name, n_steps, every=0):
    ov = dict(_TT_THIN)
    if every:
        ov["observer.checkpoint_every"] = str(every)
    dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir):
    import pyarrow.parquet as pq
    return pq.read_table(run_dir / "l1_events.parquet").to_pylist()


def test_resume_carries_the_backyard_and_the_low_flags(tmp_path):
    """★分割走行 == 一気通し: BY 在庫・棚薄フラグ・店主のレビュー窓を跨いで一致する。"""
    from society.engine import checkpoint

    st_dir = tmp_path / "tt_st"
    straight = Simulation(_cfg_of("tt_st", _TOTAL), out_dir=st_dir)
    straight.run()

    d = tmp_path / "tt_rs"
    s1 = Simulation(_cfg_of("tt_rs", _SPLIT, every=_SPLIT), out_dir=d)
    for step in range(_SPLIT):
        scheduler.run_step(s1, step)
    # ---- 空回り防止: 2層特有の状態が **同時に** 立っていること ----
    assert s1._goods_back, "BY に在庫が 1 件も無い(split 位置の再調整が要る)"
    assert s1._goods_low, "棚薄フラグが 1 件も立っていない"
    assert s1._goods_order_win, "店主のレビュー窓が 1 件も無い"
    cap = goods_mod._capacity(s1.goodscfg, "food")
    assert any(v < cap for v in s1._goods_stock.values()), "減耗中の棚が 1 つも無い"
    checkpoint.save(s1, _SPLIT, d / "checkpoint" / f"ckpt-{_SPLIT:06d}.pkl.gz")
    s1.logger.flush_segment()

    s2 = Simulation(_cfg_of("tt_rs", _TOTAL, every=_SPLIT), out_dir=d)
    s2.run(resume_from=d)

    assert _rows(d) == _rows(st_dir), "2層 ON の resume が straight と不一致"
    assert s2._goods_stock == straight._goods_stock, "棚が一気通しと食い違う"
    assert s2._goods_back == straight._goods_back, "BY が一気通しと食い違う"
    assert s2._goods_low == straight._goods_low, "棚薄フラグが一気通しと食い違う"
    assert s2._goods_order_win == straight._goods_order_win, "レビュー窓が食い違う"
    assert s2._goods_ops == straight._goods_ops, "観測タリーが食い違う"


def test_checkpoint_roundtrips_the_two_tier_ledgers(tmp_path):
    """BY・棚薄フラグ・レビュー窓が往復で 1 件残らず戻る(タプル/集合のまま)。"""
    from society.engine import checkpoint

    src = _sim(tmp_path, "tt_ck_src", **_TT)
    node = _food_node(src)
    src._goods_back = {(node, "food"): 5}
    src._goods_low = {node: {"food", "cafe"}}
    src._goods_order_win = {node: 3}
    src._goods_ops["restock_staffed"] = 7
    src._goods_staffed = {node}                             # 日次の派生索引も運ぶ
    src._goods_staffed_day = 2
    src._goods_store_nodes = {node, "nZ"}
    src._goods_store_day = 2
    p = checkpoint.save(src, 5, tmp_path / "tt_ck_src" / "checkpoint" / "ckpt-000005.pkl.gz")

    dst = _sim(tmp_path, "tt_ck_dst", **_TT)
    assert checkpoint.load(dst, p) == 5
    assert dst._goods_back == {(node, "food"): 5}
    assert dst._goods_low == {node: {"food", "cafe"}}
    assert dst._goods_order_win == {node: 3}
    assert dst._goods_ops["restock_staffed"] == 7
    # 派生索引は**空でも**運ぶ(resume 側だけが step 途中で組み直して分岐しないため)
    assert dst._goods_staffed == {node} and dst._goods_staffed_day == 2
    assert dst._goods_store_nodes == {node, "nZ"} and dst._goods_store_day == 2


def test_resumed_daily_index_is_not_rebuilt_mid_day(tmp_path):
    """★resume 後、その日の担い手集合は**組み直さない**(step 途中の世界を読んで分岐しない)。"""
    sim = _sim(tmp_path, "tt_idx", **_TT)
    node = _food_node(sim)
    for a in sim.agents:                                    # 実際の割当は全員なし
        a.work_node = ""
        a.work_start_min = -1
    sim._goods_staffed = {node}                             # checkpoint から戻った状態を模す
    sim._goods_staffed_day = 0
    assert goods_mod.staffed_nodes(sim, 700) == {node}, "同じ日なのに組み直した(分岐の種)"
    assert goods_mod.staffed_nodes(sim, 1500) == set(), "翌日に組み直していない"
    assert sim._goods_staffed_day == 1


def test_old_checkpoint_without_the_two_tier_keys_still_loads(tmp_path):
    """新キーを持たない旧 checkpoint でも落ちず、従来どおり空から始まる。"""
    import gzip
    import pickle

    from society.engine import checkpoint

    src = _sim(tmp_path, "tt_old_src", **_TT)
    src._goods_back = {(_food_node(src), "food"): 3}
    p = checkpoint.save(src, 3, tmp_path / "tt_old_src" / "checkpoint" / "ckpt-000003.pkl.gz")
    with gzip.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    for k in ("goods_back", "goods_low", "goods_order_win", "goods_ops",
              "goods_staffed", "goods_staffed_day",
              "goods_store_nodes", "goods_store_day"):
        blob["runtime"].pop(k, None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _sim(tmp_path, "tt_old_dst", **_TT)
    assert checkpoint.load(dst, p) == 3
    assert dst._goods_back == {} and dst._goods_low == {}


# ------------------------------------------------------ 観測(較正アンカーとの照合材料)
def test_provenance_is_none_when_off(tmp_path):
    assert goods_mod.provenance(_sim(tmp_path, "tt_prov_off", **_INV)) is None


def test_provenance_reports_the_calibration_anchors(tmp_path):
    """欠品棚率・BY 在庫比・「BY に在るのに棚に無い」比・担い手カバー率が出る(誇張しない生値)。"""
    sim = _sim(tmp_path, "tt_prov", **_TT)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    other = sim.city.pois_by_cat("shop")[0]["node"]
    sim._goods_stock = {(node, "food"): 0, (other, "shop"): 10}
    sim._goods_back = {(node, "food"): 12}
    goods_mod.staffed_nodes(sim, 700)
    prov = goods_mod.provenance(sim)
    assert prov["shelf_keys"] == 2 and prov["empty_shelves"] == 1
    assert prov["empty_shelf_rate"] == 0.5
    assert prov["in_back_not_on_shelf"] == 1                # BY に在るのに棚に無い(Gruen)
    assert prov["in_back_not_on_shelf_rate"] == 1.0
    assert prov["shelf_units"] == 10 and prov["back_units"] == 12
    assert 0.0 < prov["back_share"] < 1.0
    assert prov["store_nodes"] >= 1
    assert prov["staffed_store_nodes"] == 1                 # a の 1 店だけが担い手つき
    assert 0.0 < prov["staff_coverage"] <= 1.0
    assert prov["unstaffed_fallback"] is True
    assert a.work_node == node


def test_summary_carries_the_goods_provenance(tmp_path):
    """summary.json に goods キーが載る(OFF ではキー自体が出ない)。"""
    on = _sim(tmp_path, "tt_sum_on", steps=12, **_TT)
    on.run()
    summary = json.loads((on.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["goods"] == goods_mod.provenance(on)
    off = _sim(tmp_path, "tt_sum_off", steps=12, **_INV)
    off.run()
    assert "goods" not in json.loads(
        (off.out_dir / "summary.json").read_text(encoding="utf-8"))


def test_frozen_streaming_spec_is_untouched_by_this_lane(tmp_path):
    """★凍結ファイル(observer/stream.py = metrics_spec の SPEC_FILES)を触っていない。

    2層在庫の観測は **summary.json の goods キー**(goods.provenance)に閉じる。既存の
    goods_usage は在庫スライス①②の定義のまま = 指標コードの凍結ハッシュが動かない。
    新 kind(shelf_restock / stock_order)は goods_usage のどの分岐にも入らない
    (kind 名が一致せず、payload に "cat" も "item" も無い)= 集計値が 1 つも動かない。"""
    from society.observer import metrics_spec as ms

    assert "src/society/observer/stream.py" in ms.SPEC_FILES

    import pyarrow as pa
    import pyarrow.parquet as pq

    ev = [
        {"step": 0, "agent_id": 4, "kind": "shelf_restock",
         "payload": {"poi": "n1", "cats": ["food"], "qty": 12, "n": 1}},
        {"step": 1, "agent_id": -1, "kind": "shelf_restock",
         "payload": {"poi": "n2", "cats": ["shop"], "qty": 5, "n": 1, "unstaffed": True}},
        {"step": 2, "agent_id": 4, "kind": "stock_order", "payload": {"poi": "n1", "n": 2}},
        {"step": 3, "agent_id": 5, "kind": "spend",
         "payload": {"cat": "food", "amount": 900, "item": "ラーメン"}},
    ]
    rd = tmp_path / "tt_agg"
    rd.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "step": [e["step"] for e in ev], "sim_min": [e["step"] * 10 for e in ev],
        "agent_id": [e["agent_id"] for e in ev], "kind": [e["kind"] for e in ev],
        "x": [0.0] * len(ev), "y": [0.0] * len(ev),
        "payload": [json.dumps(e["payload"], ensure_ascii=False, sort_keys=True)
                    for e in ev],
    }), rd / "l1_events.parquet")
    (rd / "summary.json").write_text(json.dumps({"n_steps": 1}), encoding="utf-8")
    out = st.goods_usage(str(rd))
    assert out["totals"] == {"delivery_trip": 0, "restock": 0,
                             "stock_low": 0, "stock_out": 0}
    assert out["items_sold_total"] == 1, "既存の集計だけが動く(新 kind は素通り)"
    assert "shelf_restock" not in out and "stock_order" not in out


# ------------------------------------------------------ 実ラン(フル経路が閉じる)
def test_full_run_closes_the_loop(tmp_path):
    """実ランで 購入→棚薄→(納品→BY)→店員の棚出し が起き、BY 台帳が実体化する。"""
    sim = Simulation(_cfg_of("tt_loop", 96), out_dir=tmp_path / "tt_loop")
    sim.run()
    assert sim._goods_back or _kind(sim, "shelf_restock"), \
        "2層の経路が 1 度も通っていない(テスト前提が崩れた)"
    for e in _kind(sim, "shelf_restock"):
        assert e.payload["qty"] > 0 and e.payload["n"] >= 1
        if e.agent_id == -1:
            assert e.payload.get("unstaffed") is True
        else:
            assert "unstaffed" not in e.payload
