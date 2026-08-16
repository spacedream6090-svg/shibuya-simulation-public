"""在館数(commerce.occupancy)の走査量削減の**同値性**と計算量(レーンP A4)。

塞ぐ穴: `commerce.occupancy` は呼ぶたびに `sim.agents` の全走査(25 万体 = 1 回 25 万比較)で、
  (1) 購入 1 件ごと(commerce.py on_purchase)
  (2) VC 審査の venture 1 件ごと(tools.py _vc_review)
に呼ばれるため O(購入数×人口) / O(店舗数×人口) になっていた。

本バッチの手当ては **2 つとも「同値変換」**(値・イベント・L1 バイトが 1 ビットも変わらない):
  A. `node_counts(sim)` = 1 走査で「ノード→在場・覚醒人数」表を作る。**同一時点の在館数を
     複数ノードぶん要る場所**(= VC 審査ループ。誰の node/loc/sleeping も動かさない)でだけ使う。
  B. `demand_cap` + `count_at_node` = 在館数の使い道(品切れ判定・価格係数)がどちらも頭打ちに
     なる点で走査を打ち切る。打ち切り点以上では on_purchase の結論が同一。

★実装しなかったこと(正直な報告): 「step 単位のキャッシュ(step が変わったら 1 回だけ全走査)」は
  **同値にならない**。scheduler._phase_move は同一 step の中で agent ごとに `agent.node` を書き換え
  (scheduler.py:1330)、その**同じループの中で**到着した個体の購入 `_charge_meal`(scheduler.py:1368
  → commerce.on_purchase)を呼ぶ。つまり同一 step 内の 2 回の購入の間で在館数は動く。
  `test_occupancy_changes_within_a_step` がこの事実を機械固定する(将来 step キャッシュを入れたく
  なった人が、まずこのテストで止まるように)。O(1) 逆引き索引にするには node/loc/sleeping の
  **書き込み側にフック**が要る(scheduler.py / agent 側 = 別レーン所有)。
"""
from __future__ import annotations

import json
import math
import random

from society import commerce, economy
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_ON = {"commerce.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# 参照実装 = 置き換え前の全走査(1 行も変えずに保持する)。
def _occ_scan(sim, node) -> int:
    return sum(1 for a in sim.agents
               if a.node == node and a.loc != "outside" and not a.sleeping)


def _shuffle_world(sim, rnd, nodes):
    """ランダムな配置・出入り(loc=outside)・睡眠遷移を作る。"""
    for a in sim.agents:
        a.node = rnd.choice(nodes)
        a.loc = rnd.choice(["street", "building", "outside", "outside"])
        a.sleeping = rnd.random() < 0.3


# --------------------------------------------------------------------- A. 表(node_counts)
def test_node_counts_matches_full_scan(tmp_path):
    """ランダム配置・出入り・睡眠遷移を跨いで node_counts == 旧全走査(全ノード)。"""
    sim = _sim(tmp_path, "counts", n=40)
    rnd = random.Random(3)
    nodes = sorted({a.node for a in sim.agents})[:6] or [sim.agents[0].node]
    for rounds in range(12):
        _shuffle_world(sim, rnd, nodes)
        counts = commerce.node_counts(sim)
        for node in nodes:
            assert commerce.occupancy(sim, node, counts) == _occ_scan(sim, node), \
                f"round={rounds} node={node} で表と全走査が食い違う"
        assert commerce.occupancy(sim, "no_such_node", counts) == 0, "未知ノードが 0 でない"
        # 表を使わない呼び出し(既定の引数)は従来どおり全走査で同値
        for node in nodes:
            assert commerce.occupancy(sim, node) == _occ_scan(sim, node)


def test_node_counts_all_outside_or_sleeping_is_empty(tmp_path):
    """全員 outside / 全員睡眠なら表は空(旧走査の 0 と一致)。"""
    sim = _sim(tmp_path, "counts_edge", n=12)
    node = sim.agents[0].node
    for a in sim.agents:
        a.node, a.loc, a.sleeping = node, "outside", False
    assert commerce.node_counts(sim) == {} and _occ_scan(sim, node) == 0
    for a in sim.agents:
        a.loc, a.sleeping = "street", True
    assert commerce.node_counts(sim) == {} and _occ_scan(sim, node) == 0
    for a in sim.agents:
        a.sleeping = False
    assert commerce.node_counts(sim)[node] == len(sim.agents) == _occ_scan(sim, node)


# --------------------------------------------------------------------- B. 打ち切り(cap)
def test_count_at_node_equals_min_of_scan_and_cap(tmp_path):
    """count_at_node(cap) == min(旧全走査, cap)(cap<=0 は 0・巨大 cap は全走査)。"""
    sim = _sim(tmp_path, "cap", n=40)
    rnd = random.Random(5)
    nodes = sorted({a.node for a in sim.agents})[:5] or [sim.agents[0].node]
    for _ in range(10):
        _shuffle_world(sim, rnd, nodes)
        for node in nodes:
            true_n = _occ_scan(sim, node)
            for cap in (0, 1, 2, 3, 6, 20, 10 ** 6):
                assert commerce.count_at_node(sim, node, cap) == min(true_n, cap), \
                    f"node={node} cap={cap} true={true_n}"
            assert commerce.count_at_node(sim, node, commerce._NO_CAP) == true_n


def _decide(cfg, occ):
    """on_purchase が occ から出す**結論**(品切れなら None、そうでなければ価格係数)。"""
    if commerce.is_stock_out(cfg, occ):
        return None
    return commerce.price_coef(cfg, occ)


def _same(a, b) -> bool:
    """NaN 同士は「同じ結論」とみなす(病的 config の比較のため)。"""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_demand_cap_sweep_never_changes_the_decision():
    """感度/閾値/基準/clip 幅の総当たり(1,296 通り)で、cap 打ち切りが結論を変えない。

    病的な設定(sens=0/負/極小/inf/NaN・clip の上下逆転・基準が負)も含める。打ち切れない
    設定では `demand_cap` は `_NO_CAP`(=全走査)を返して安全側に倒れる。"""
    n_capped = 0
    for sens in (0.0, 0.15, 0.3, -0.2, -1.0, 1e-9, 1e-320, float("inf"), float("nan")):
        for thr in (-5, 0, 1, 2, 6, 100):
            for ref in (-50, 0, 2, 5):
                for lo, hi in ((0.7, 1.6), (0.5, 0.5), (2.0, 0.5), (1.0, 1.0),
                               (0.0, 10.0), (float("nan"), 1.0)):
                    cfg = commerce.build_cfg({
                        "enabled": True, "price_sensitivity": sens,
                        "stock_threshold": thr, "demand_ref": ref,
                        "price_min": lo, "price_max": hi})
                    cap = commerce.demand_cap(cfg)
                    if cap >= commerce._NO_CAP:
                        continue                      # 打ち切らない = 従来どおり全走査
                    n_capped += 1
                    for occ in range(0, min(cap, 300) + 300):
                        assert _same(_decide(cfg, min(occ, cap)), _decide(cfg, occ)), \
                            f"sens={sens} thr={thr} ref={ref} clip=({lo},{hi}) cap={cap} occ={occ}"
    assert n_capped > 500, f"打ち切りが効いた設定が少なすぎる({n_capped})"


def test_demand_cap_never_changes_the_decision(tmp_path):
    """打ち切り点 cap 以上では結論(品切れ/価格係数)が真の在館数と完全一致する。

    在庫 ON(閾値>0)/在庫 OFF(価格のみ)/係数 0(恒等)/負の感度(混雑でセール)を総当たり。"""
    variants = [
        {},                                                     # 既定(thr=6, sens=0.15)
        {"commerce.stock_threshold": "0"},                      # 在庫 OFF = 価格の頭打ちで打ち切る
        {"commerce.stock_threshold": "0", "commerce.price_sensitivity": "0.0"},
        {"commerce.stock_threshold": "0", "commerce.price_sensitivity": "-0.2"},
        {"commerce.stock_threshold": "0", "commerce.price_sensitivity": "0.01"},
        {"commerce.stock_threshold": "3", "commerce.price_sensitivity": "0.3"},
        {"commerce.stock_threshold": "1"},
    ]
    for i, ov in enumerate(variants):
        sim = _sim(tmp_path, f"cap_cfg{i}", n=4, **{**_ON, **ov})
        cfg = sim.commercecfg
        cap = commerce.demand_cap(cfg)
        assert cap >= 0
        for occ in range(0, min(cap, 500) + 400):
            assert _decide(cfg, min(occ, cap)) == _decide(cfg, occ), \
                f"variant={ov} cap={cap} occ={occ} で結論が変わった"
    # 既定 conf(stock_threshold=6)の打ち切り点は 6(= 閾値)であることを機械固定
    sim = _sim(tmp_path, "cap_default", n=4, **_ON)
    assert commerce.demand_cap(sim.commercecfg) == 6
    # price_sensitivity=0 かつ在庫 OFF は「1 人も数えなくてよい」(cap=0)
    sim0 = _sim(tmp_path, "cap_zero", n=4, **{**_ON, "commerce.stock_threshold": "0",
                                              "commerce.price_sensitivity": "0.0"})
    assert commerce.demand_cap(sim0.commercecfg) == 0


def test_on_purchase_identical_with_and_without_cap(tmp_path, monkeypatch):
    """on_purchase の戻り値・イベントが「打ち切りあり/なし」で完全一致(ランダム混雑 × 設定)。"""
    for i, ov in enumerate([{}, {"commerce.stock_threshold": "3"},
                            {"commerce.stock_threshold": "0"},
                            {"commerce.price_sensitivity": "0.4",
                             "commerce.stock_threshold": "12"}]):
        results = {}
        for capped in (True, False):
            sim = _sim(tmp_path, f"pur{i}_{capped}", n=30, **{**_ON, **ov})
            r = random.Random(100 + i)               # 同じ世界を 2 度作る(決定論)
            nodes = sorted({a.node for a in sim.agents})[:4] or [sim.agents[0].node]
            _shuffle_world(sim, r, nodes)
            buyer = sim.agents[0]
            buyer.loc, buyer.sleeping = "street", False
            amounts = []
            for node in nodes:
                buyer.node = node
                if not capped:                       # 打ち切りを殺す = 常に全走査(旧経路)
                    monkeypatch.setattr(commerce, "demand_cap",
                                        lambda cfg: commerce._NO_CAP)
                else:
                    monkeypatch.undo()
                amounts.append(commerce.on_purchase(sim, buyer, "food", 900.0, 0, 0))
            results[capped] = (amounts, _l1(sim))
            monkeypatch.undo()
        assert results[True][0] == results[False][0], f"変種{i}: 戻り値が食い違う"
        assert results[True][1] == results[False][1], f"変種{i}: イベント列が食い違う"
        assert any(a is not None for a in results[True][0]), "テストが無風(全部品切れ)"


def test_full_run_l1_identical_with_and_without_cap(tmp_path, monkeypatch):
    """mock フルラン(144 step・商業 ON)の L1 が打ち切りの有無で **1 バイトも変わらない**。"""
    on = {**_ON, "commerce.demand_ref": "2", "commerce.price_sensitivity": "0.3",
          "commerce.stock_threshold": "2", "commerce.stock_grievance": "0.03"}
    a = _sim(tmp_path, "cap_run_a", n=30, steps=144, **on)
    a.run()
    monkeypatch.setattr(commerce, "demand_cap", lambda cfg: commerce._NO_CAP)
    b = _sim(tmp_path, "cap_run_b", n=30, steps=144, **on)
    b.run()
    monkeypatch.undo()
    assert _l1(a) == _l1(b), "打ち切りの有無で L1 が変わった(同値変換になっていない)"
    kinds = {e.kind for e in a.logger.events}
    assert "price_change" in kinds or "stock_out" in kinds, "テストが無風(商業イベント 0 件)"


# --------------------------------------------------------------------- C. VC 審査(1 審査 1 走査)
def _mk_ventures(sim, tools, owners, nodes):
    for owner, node in zip(owners, nodes):
        v = {"owner": owner.id, "node": node, "name": f"屋台{owner.id}", "offer": "",
             "price": 800.0, "opened_step": 0, "last_sale_step": 0, "open_at": 0,
             "permitted": True, "sales_total": 8000.0, "fulltime": False}
        tools.ventures[owner.id] = v
        tools.ventures_by_node[node].append(v)


def test_vc_review_scans_once_and_values_match_scan(tmp_path, monkeypatch):
    """VC 審査: 在館数の全走査は **1 審査につき 1 回**(店舗数に比例しない)。値は旧走査と同一。"""
    sim = _sim(tmp_path, "vc", n=24)
    sim.economy["accounts"] = economy.build_accounts_cfg({"enabled": True})
    sim.economy["vc"] = economy.build_vc_cfg({"enabled": True, "review_period_days": 1})
    tools = sim.tools
    nodes = sorted({a.node for a in sim.agents})[:5]
    assert len(nodes) >= 2, "地図が単一ノード(テストが無風)"
    rnd = random.Random(17)
    _shuffle_world(sim, rnd, nodes)
    owners = sim.agents[:len(nodes)]
    for o in owners:                                  # 店主は在場(present_agent を通す)
        o.loc, o.sleeping, o.account, o.money = "street", False, 0.0, 0.0
    _mk_ventures(sim, tools, owners, nodes)

    n_tables = {"n": 0}
    real_counts = commerce.node_counts
    monkeypatch.setattr(commerce, "node_counts",
                        lambda s: (n_tables.__setitem__("n", n_tables["n"] + 1),
                                   real_counts(s))[1])
    seen = []
    real_score = commerce.vc_score
    monkeypatch.setattr(commerce, "vc_score",
                        lambda sales, occ, deg, cfg: (seen.append((sales, occ, deg)),
                                                      real_score(sales, occ, deg, cfg))[1])
    sim._vc_review_day = -1
    tools._vc_review(sim, step=0, sim_min=0)
    monkeypatch.undo()

    assert n_tables["n"] == 1, f"1 審査で在館数の走査が {n_tables['n']} 回(店舗数に比例している)"
    assert len(seen) == len(owners), "審査された venture 数が合わない"
    order = [tools.ventures[oid]["node"] for oid in sorted(tools.ventures)]
    for (sales, occ, deg), node in zip(seen, order):
        assert occ == _occ_scan(sim, node), f"node={node} の在館数が旧全走査と食い違う"
    assert any(occ > 0 for _, occ, _ in seen), "テストが無風(在館数が全部 0)"


def test_vc_review_loop_does_not_move_anyone(tmp_path):
    """表を使ってよい前提の証明: 審査ループは誰の node/loc/sleeping も動かさない(読むだけ)。"""
    sim = _sim(tmp_path, "vc_static", n=24)
    sim.economy["accounts"] = economy.build_accounts_cfg({"enabled": True})
    sim.economy["vc"] = economy.build_vc_cfg({"enabled": True, "review_period_days": 1})
    tools = sim.tools
    nodes = sorted({a.node for a in sim.agents})[:4]
    _shuffle_world(sim, random.Random(23), nodes)
    owners = sim.agents[:len(nodes)]
    for o in owners:
        o.loc, o.sleeping, o.account, o.money = "street", False, 0.0, 0.0
    _mk_ventures(sim, tools, owners, nodes)
    before = [(a.id, a.node, a.loc, a.sleeping) for a in sim.agents]
    sim._vc_review_day = -1
    tools._vc_review(sim, step=0, sim_min=0)
    after = [(a.id, a.node, a.loc, a.sleeping) for a in sim.agents]
    assert before == after, "審査ループが在場状態を動かしている(表の使用前提が崩れる)"


# --------------------------------------------------------------------- D. 反証(step キャッシュは不可)
def test_occupancy_changes_within_a_step(tmp_path):
    """★同一 step 内で在館数は動く = 「step 単位キャッシュ」は同値変換にならない、の反証。

    scheduler._phase_move は 1 つのループの中で agent.node を書き換え(移動)、その同じループで
    到着した個体の購入(_charge_meal → commerce.on_purchase → occupancy)を呼ぶ。つまり同一 step の
    2 回の購入の間に在館数が変わりうる。ここでは「移動を挟むと同 step でも値が変わる」ことだけを
    直接固定する(将来 step キャッシュを入れると、このテストが赤になる)。"""
    sim = _sim(tmp_path, "instep", n=12, **_ON)
    node_a = sim.agents[0].node
    node_b = next((a.node for a in sim.agents if a.node != node_a), None)
    assert node_b is not None, "地図が単一ノード(テストが無風)"
    for a in sim.agents:
        a.node, a.loc, a.sleeping = node_a, "street", False
    first = commerce.occupancy(sim, node_a)
    sim.agents[-1].node = node_b                      # _phase_move の 1 個体ぶんの移動を模す
    second = commerce.occupancy(sim, node_a)
    assert second == first - 1, "同一 step 内の移動が在館数に反映されていない"
    # 実エンジンの seam がまだ「移動ループの中で購入が走る」形であること(この形である限り
    # step キャッシュは同値にならない)。_phase_move が _charge_meal を呼んでいるかで見る。
    import inspect
    src = inspect.getsource(scheduler._phase_move)
    assert "_charge_meal" in src, \
        "移動ループから購入が消えた(step キャッシュの可否を再検討してよい)"


def test_commerce_off_never_counts(tmp_path):
    """commerce OFF(既定)では on_purchase 経路に入らない = 在館数を 1 度も数えない。"""
    sim = _sim(tmp_path, "off", n=20, steps=144)
    assert not commerce.enabled(sim)
    sim.run()
    for k in ("shop_state", "price_change", "stock_out"):
        assert not [e for e in sim.logger.events if e.kind == k], f"OFF で {k} が出ている"
