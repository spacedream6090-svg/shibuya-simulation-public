"""商業・店舗のダイナミクス(現実ギャップ 後続波 H3 2026-07-07)= 営業時間/動的価格/在庫 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): shop_state/price_change/stock_out が 0 件・営業時間ゲートも価格係数もかからず・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- 営業時間 ON: カテゴリの開閉が時刻の純関数で決まり、閉店中の POI が行き先候補から除外され、開閉遷移で
  shop_state が出る(単体・決定論・RNG不要)。
- 動的価格 ON: 在館数(需要)が基準を超えると価格係数>1(surge)で消費額が上がり、下回るとセール(<1)。
  price_change(cat, ratio)。price_sensitivity=0 で恒等(不変)。
- 在庫 ON: 需要集中(在館数≥閾値)で stock_out=購入抑制 + 不満(factors 経由 on_scarcity)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 閉店による行き先除外は物理位置=対面 co-location を変えうる(career G5 / crowd G4 / 健康 H1 と同型)。
  よって呼数不変は compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import commerce
from society.config import load_config
from society.engine.simulation import Simulation

_COMMERCE = {"commerce.enabled": "true"}
# 各機構を確実に効かせた ON 設定(決定論/k 不変性/スモーク用)。demand_ref=2 で単身客(occ=1)は
# セール(coef<1=price_change)、2 人以上の集中で stock_out(threshold=2)。
_ON = {**_COMMERCE, "commerce.demand_ref": "2", "commerce.price_sensitivity": "0.3",
       "commerce.stock_threshold": "2", "commerce.stock_grievance": "0.03"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。H3 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"commerce.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H3 seam が no-op でない)"
    for k in ("shop_state", "price_change", "stock_out"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- 営業時間
def test_is_open_by_category_and_hours(tmp_path):
    """営業時間は時刻の純関数(RNG不要・決定論)。夜間営業(open>close)と非ゲート常時営業を確認。"""
    sim = _sim(tmp_path, "hours", steps=1, **_COMMERCE)
    cfg = sim.commercecfg
    # food 11:00-23:00
    assert not commerce.is_open(cfg, "food", 9 * 60), "09:00 に food が開いている"
    assert commerce.is_open(cfg, "food", 12 * 60), "12:00 に food が閉じている"
    assert not commerce.is_open(cfg, "food", 23 * 60 + 30), "23:30 に food が開いている"
    # nightlife 18:00-翌05:00(夜間営業)
    assert not commerce.is_open(cfg, "nightlife", 12 * 60), "正午に nightlife が開いている"
    assert commerce.is_open(cfg, "nightlife", 20 * 60), "20:00 に nightlife が閉じている"
    assert commerce.is_open(cfg, "nightlife", 2 * 60), "02:00(翌朝)に nightlife が閉じている"
    # 非ゲートカテゴリ(office 等)は常時営業
    assert commerce.is_open(cfg, "office", 3 * 60), "非ゲートカテゴリが常時営業でない"


def test_closed_pois_excluded_from_candidates(tmp_path):
    """閉店中の POI は行き先候補から除外される(filter_open)。OFF なら除外せず不変。"""
    sim = _sim(tmp_path, "closed", steps=1, **_COMMERCE)
    foods = sim.city.pois_by_cat("food")[:5]
    assert foods, "food POI が存在しない(地図の想定外)"
    assert commerce.filter_open(sim, foods, 3 * 60) == [], "深夜に food が候補に残っている"
    assert commerce.filter_open(sim, foods, 12 * 60) == foods, "昼に food が除外されている"
    # OFF は素通し(バイト一致の根拠)
    sim.commercecfg["enabled"] = False
    assert commerce.filter_open(sim, foods, 3 * 60) is foods, "OFF で候補が変化している"


def test_shop_state_transitions_logged(tmp_path):
    """開閉遷移で shop_state を記録(初回はベースライン=ログなし、遷移した step だけ 1 件)。"""
    sim = _sim(tmp_path, "ss", steps=1, **_COMMERCE)
    commerce.tick_shop_state(sim, 0, 9 * 60)         # 09:00 全カテゴリ閉=ベースライン(ログなし)
    assert not _kind(sim, "shop_state"), "初回観測でログが出ている(ベースラインのはず)"
    commerce.tick_shop_state(sim, 1, 10 * 60 + 30)   # 10:30 shop が開店
    ss = _kind(sim, "shop_state")
    assert any(e.payload["poi"] == "shop" and e.payload["state"] == "open"
               for e in ss), "shop の開店遷移(shop_state)が出ていない"
    assert all(e.agent_id == -1 for e in ss), "shop_state が世界イベント(agent_id=-1)でない"


# --------------------------------------------------------------------- 動的価格
def test_price_coef_surge_and_sale(tmp_path):
    """在館数(需要)で価格係数が決まる: ref 超で surge(>1)、未満でセール(<1)、clip 内。"""
    sim = _sim(tmp_path, "coef", steps=1,
               **{**_COMMERCE, "commerce.price_sensitivity": "0.2",
                  "commerce.demand_ref": "3"})
    cfg = sim.commercecfg
    assert commerce.price_coef(cfg, 3) == 1.0, "基準需要で係数が 1.0 でない"
    assert commerce.price_coef(cfg, 6) > 1.0, "混雑(需要超)で surge にならない"
    assert commerce.price_coef(cfg, 1) < 1.0, "閑散(需要未満)でセールにならない"
    assert commerce.price_coef(cfg, 999) == cfg["price_max"], "係数上限で clip していない"
    assert commerce.price_coef(cfg, -999) == cfg["price_min"], "係数下限で clip していない"


def test_price_sensitivity_zero_is_identity(tmp_path):
    """price_sensitivity=0 なら係数は常に 1.0(恒等=消費額不変)。"""
    sim = _sim(tmp_path, "id", steps=1,
               **{**_COMMERCE, "commerce.price_sensitivity": "0.0"})
    cfg = sim.commercecfg
    assert commerce.price_coef(cfg, 0) == 1.0 and commerce.price_coef(cfg, 50) == 1.0


def test_on_purchase_surge_raises_amount_and_logs(tmp_path):
    """混雑時 on_purchase は消費額を係数>1 で上げ price_change(surge)を記録する。"""
    sim = _sim(tmp_path, "surge", steps=1,
               **{**_COMMERCE, "commerce.price_sensitivity": "0.2",
                  "commerce.demand_ref": "2", "commerce.stock_threshold": "100"})
    a = sim.agents[0]
    for o in sim.agents[1:5]:                        # a を含め node に 5 人=需要集中
        o.node, o.loc, o.sleeping = a.node, "street", False
    amount = commerce.on_purchase(sim, a, "food", 900.0, 0, 0)
    assert amount is not None and amount > 900.0, "混雑なのに消費額が上がっていない(surge)"
    pc = _kind(sim, "price_change")
    assert pc and pc[-1].payload["cat"] == "food" and pc[-1].payload["ratio"] > 1.0, \
        "price_change(surge)が出ていない"


# --------------------------------------------------------------------- 在庫・品切れ
def test_stock_out_suppresses_purchase_and_raises_grievance(tmp_path):
    """需要集中(在館数≥閾値)で stock_out=購入抑制(None)+ 不満(factors 経由 on_scarcity)。"""
    sim = _sim(tmp_path, "so", steps=1,
               **{**_COMMERCE, "commerce.stock_threshold": "3",
                  "commerce.stock_grievance": "0.05"})
    a = sim.agents[0]
    for o in sim.agents[1:5]:                        # a を含め node に 5 人 ≥ 閾値 3
        o.node, o.loc, o.sleeping = a.node, "street", False
    g0 = a.states["grievance"]
    amount = commerce.on_purchase(sim, a, "food", 900.0, 0, 0)
    assert amount is None, "品切れ/行列なのに購入が抑制されていない"
    assert _kind(sim, "stock_out"), "stock_out が出ていない"
    assert a.states["grievance"] > g0, "品切れ→不満(grievance+)が入っていない"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "scarcity"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "品切れ→grievance(scarcity)が factors 経由で入っていない"


def test_stock_grievance_zero_observes_only(tmp_path):
    """stock_grievance=0.0 なら stock_out は出るが grievance は不変(観測のみ)。"""
    sim = _sim(tmp_path, "so0", steps=1,
               **{**_COMMERCE, "commerce.stock_threshold": "3",
                  "commerce.stock_grievance": "0.0"})
    a = sim.agents[0]
    for o in sim.agents[1:5]:
        o.node, o.loc, o.sleeping = a.node, "street", False
    g0 = a.states["grievance"]
    assert commerce.on_purchase(sim, a, "food", 900.0, 0, 0) is None
    assert _kind(sim, "stock_out"), "stock_out が出ていない"
    assert a.states["grievance"] == g0, "grievance=0 指定なのに不満が動いた"


# --------------------------------------------------------------------- 統合(消費経路)
def test_run_emits_commerce_events(tmp_path):
    """フル run(30 agents×144 step)で shop_state/price_change/stock_out が出て落ちない。"""
    sim = _sim(tmp_path, "run", n=30, steps=144, **_ON)
    sim.run()
    assert _kind(sim, "shop_state"), "shop_state が 1 件も出ていない(営業時間遷移)"
    assert _kind(sim, "price_change") or _kind(sim, "stock_out"), \
        "動的価格/在庫のイベントが 1 件も出ていない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """商業 ON(営業時間+動的価格+在庫)同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 k 不変性
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    """商業 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_commerce_call_count_k_invariant(tmp_path):
    """閉店による行き先除外は対面 co-location を変え ON!=OFF になりうる(実在する営業時間の必然)。だが
    R1 の本旨=「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。commerce の
    機構は時刻+物理位置(在館数)+config のみ参照し、発火判断に k・内面状態(構成概念)を食わせないため、
    k=free と k=off で generate 呼数が完全一致する(career G5 / 健康 H1 と同型)。"""
    free = _run_k(tmp_path, "cm_free", writeback="free")
    off = _run_k(tmp_path, "cm_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"commerce の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "shop_state"), "営業時間 ON なのに shop_state が出ていない(機構が不発)"
