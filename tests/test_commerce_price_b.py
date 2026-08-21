"""PRICE-B 時間帯価格(B1)+ 閉店前見切り(B2)のテスト。第147。

正典: docs/plans/inventory-two-tier-plan.md §1.5 PRICE-B。
リサーチ: docs/research/retail-pricing-empirics.md(定価改定は年 0.8 回=数日ランでは価格は実質不変。
  動く現実の主経路=事前公表の時間帯料金表 と 閉店前見切り。後者は公取委2009 で「店側の裁量行動」
  として法的に確定した類型)。

何を機械固定するか:
- OFF/空(既定): L1 が現行とバイト一致・markdown 0 件・価格係数は恒等 1.0。
- B1: 帯の引き当て(並び順に最初が勝つ / 日跨ぎ / 帯の外は 1.0)。★price_change を 1 件も出さない。
- B2: 段階の時刻窓(0→1→2)/ 棚が空 or 当日入荷なしでは貼らない / **1 店 1 step に 1 件** /
      段階は単調(戻らない)/ 翌朝リセット / 閉店でリセット / unstaffed 分岐 / checkpoint 搬送。
- 合成: 時間帯係数 × 見切り係数(順序を式で固定)。
- 決定論(同 seed 2 ラン一致)・resume == straight。
検証は mock のみ(実LLM 禁止・≤144 step)。乱数は 1 本も引かない。
"""
from __future__ import annotations

import json

from society import commerce, goods as goods_mod
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.causality import AGENT, CAUSE_OF_KIND
from society.observer.schema import EVENT_KINDS

_INV = {"commerce.inventory.enabled": "true"}
_MD = {**_INV, "commerce.markdown.enabled": "true"}
#: 18:00-23:00 のディナー帯を 1.67 倍(900→1,500 円)にする事前公表の料金表。
_SCHED = {"commerce.price_schedule": "{food: [[1080, 1380, 1.67]]}"}


def _sim(tmp_path, name, n=25, steps=1, **ov):
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


def _stock(sim, node, cat, units, day):
    """その (店, 業態) に棚在庫と**当日の入荷バッチ**を置く(見切りのトリガ条件を満たす)。"""
    sim._goods_stock[(node, cat)] = int(units)
    sim._goods_delivered[(node, cat)] = int(day)


def _staff(sim, node, agent):
    """agent をその店の在勤者にする(INV-B と同じ在勤述語=新しい勤務概念を作らない)。"""
    for a in sim.agents:
        a.work_node = ""
        a.work_start_min = -1
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


# ------------------------------------------------------ 登録(schema + 因果台帳)
def test_schema_and_causality_registered():
    assert "markdown" in EVENT_KINDS, "markdown が schema 未登録"
    assert CAUSE_OF_KIND["markdown"] == AGENT, "markdown が**当人の行為**として分類されていない"


def test_registry_declares_the_toggles():
    from society import registry as R
    for fid in ("commerce.price_schedule", "commerce.markdown.enabled"):
        feat = R.BY_ID.get(fid)
        assert feat is not None, f"{fid} が registry に未宣言"
        assert feat.repro_tier == "strict" and feat.affects_k is False


# ------------------------------------------------------ OFF/空: 現行とバイト一致
def test_defaults_match_pure_default(tmp_path):
    """明示 OFF/空と純粋既定が L1 完全一致(B1/B2 の seam が no-op)。"""
    pure = _sim(tmp_path, "pb_pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "pb_off", steps=48,
               **{"commerce.markdown.enabled": "false", "commerce.price_schedule": "{}"})
    off.run()
    assert _l1(pure) == _l1(off), "B1/B2 既定が純粋既定と不一致(seam が no-op でない)"
    assert not _kind(pure, "markdown")


def test_markdown_off_matches_inventory_only(tmp_path):
    """在庫 ON のままの見切り OFF が、在庫 ON だけの世界と L1 完全一致。"""
    base = _sim(tmp_path, "pb_inv", steps=48, **_INV)
    base.run()
    off = _sim(tmp_path, "pb_inv_off", steps=48,
               **{**_INV, "commerce.markdown.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    assert not _kind(base, "markdown") and not base._md_stage


def test_empty_schedule_is_identity(tmp_path):
    """既定 {} は恒等 1.0 で、金額オブジェクトをそのまま返す(浮動小数の乗算も通さない)。"""
    sim = _sim(tmp_path, "pb_id", steps=1)
    assert commerce.price_schedule_coef(sim.commercecfg, "food", 20 * 60) == 1.0
    amount = 912.5
    assert commerce.apply_price(sim, amount, "food", _food_node(sim), 20 * 60) is amount


def test_apply_price_passes_through_a_suppressed_purchase(tmp_path):
    """購入抑制(None)はそのまま素通しする(係数が None に掛かって落ちない)。"""
    sim = _sim(tmp_path, "pb_none", steps=1, **{**_MD, **_SCHED})
    assert commerce.apply_price(sim, None, "food", _food_node(sim), 20 * 60) is None


# ------------------------------------------------------ B1 時間帯料金表
def test_price_schedule_band_lookup(tmp_path):
    """帯の中は係数・帯の外は 1.0(時刻の純関数・O(帯数))。"""
    sim = _sim(tmp_path, "pb_band", steps=1, **_SCHED)
    cfg = sim.commercecfg
    assert commerce.price_schedule_coef(cfg, "food", 12 * 60) == 1.0, "昼に夜の係数が出ている"
    assert commerce.price_schedule_coef(cfg, "food", 18 * 60) == 1.67, "帯の開始が含まれていない"
    assert commerce.price_schedule_coef(cfg, "food", 22 * 60) == 1.67
    assert commerce.price_schedule_coef(cfg, "food", 23 * 60) == 1.0, "帯の終了が含まれている(半開区間でない)"
    assert commerce.price_schedule_coef(cfg, "shop", 20 * 60) == 1.0, "表に無い業態に係数が出ている"


def test_price_schedule_wraps_midnight_and_first_match_wins(tmp_path):
    """開始>終了は日跨ぎ・帯が重なったら**並び順に最初**が勝つ(決定論)。"""
    sim = _sim(tmp_path, "pb_wrap", steps=1,
               **{"commerce.price_schedule":
                  "{nightlife: [[1320, 120, 0.8], [0, 1440, 1.5]]}"})
    cfg = sim.commercecfg
    assert commerce.price_schedule_coef(cfg, "nightlife", 23 * 60) == 0.8, "日跨ぎ帯に入っていない"
    assert commerce.price_schedule_coef(cfg, "nightlife", 1 * 60) == 0.8, "翌朝側が帯に入っていない"
    assert commerce.price_schedule_coef(cfg, "nightlife", 12 * 60) == 1.5, "後続の帯が拾えていない"


def test_price_schedule_emits_no_price_change(tmp_path):
    """★掲示済みの料金表どおりに払うのは「価格の**変化**」ではない=price_change を 1 件も出さない。"""
    sim = _sim(tmp_path, "pb_nopc", n=25, steps=144, **_SCHED)
    sim.run()
    assert not _kind(sim, "price_change"), \
        "時間帯料金表が price_change を出している(イベント洪水の再発)"


def test_price_schedule_raises_the_spend(tmp_path):
    """ディナー帯の消費額が昼より高い(係数が実際に spend へ乗る)。"""
    sim = _sim(tmp_path, "pb_amt", steps=1, **_SCHED)
    node = _food_node(sim)
    assert commerce.apply_price(sim, 900.0, "food", node, 12 * 60) == 900.0
    assert commerce.apply_price(sim, 900.0, "food", node, 19 * 60) == 900.0 * 1.67


# ------------------------------------------------------ B2 見切りの時刻窓
def test_markdown_stage_window(tmp_path):
    """段階は閉店までの残り step から決まる(0=まだ / 1 / 2)。閉店時刻は hours から導出。"""
    sim = _sim(tmp_path, "pb_win", steps=1, **_MD)
    cfg = sim.commercecfg                            # food 11:00-23:00・Δt=10 分
    st = commerce.markdown_stage_now
    assert st(cfg, "food", 19 * 60, 10) == 0, "3 時間より前に見切っている"
    assert st(cfg, "food", 20 * 60, 10) == 1, "閉店 3 時間前(18 step)で 1 段階目にならない"
    assert st(cfg, "food", 21 * 60, 10) == 1
    assert st(cfg, "food", 21 * 60 + 30, 10) == 2, "閉店 1.5 時間前(9 step)で 2 段階目にならない"
    assert st(cfg, "food", 22 * 60 + 50, 10) == 2
    assert st(cfg, "food", 23 * 60, 10) == 0, "閉店後に見切っている"
    assert st(cfg, "food", 3 * 60, 10) == 0, "閉店中に見切っている"
    assert st(cfg, "office", 20 * 60, 10) == 0, "時刻表の無い業態(常時営業)を見切っている"


def test_markdown_reads_the_effective_night_hours(tmp_path):
    """★閉店時刻は**実効表**(夜間経済の上書き込み)から引く(見切り専用の時刻表を作らない)。"""
    sim = _sim(tmp_path, "pb_night", steps=1,
               **{**_MD, "world.night_economy.enabled": "true",
                  "world.night_economy.hours.food": "[11, 20]"})
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    # 20:00 閉店へ前倒しされたので 17:00(=3 時間前)が 1 段階目。20:00 以降は閉店=0。
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=17 * 60) == 1
    assert commerce.markdown_coef(sim, node, "food", 17 * 60) == 0.8
    assert commerce.markdown_coef(sim, node, "food", 20 * 60 + 30) == 1.0, \
        "前倒しされた閉店時刻を見ていない"


def test_markdown_stage_keeps_wall_clock_across_dt(tmp_path):
    """★Δt を変えても**同じ時刻**で値札が替わる(timeconv の STEPS 変換 × step_minutes)。

    段階の閾値は step 数なので、Δt を細かくしたら conf 側の step 数が逆比例で増えなければ
    見切りの開始時刻がずれる。conf ロードの変換(commerce.markdown.stage_steps=STEPS)と
    実行時の step_minutes が噛み合っていることを、実時刻で機械固定する。"""
    for dt in (10, 5):            # ★Δt=10 の整数分の 1 は丸めが起きない(9→18 step 等)
        sim = _sim(tmp_path, f"pb_dt{dt}", steps=1, **{**_MD, "run.dt_min": str(dt)})
        cfg, smin = sim.commercecfg, sim.clock.step_minutes
        assert smin == dt, "Δt が clock へ届いていない(テスト前提)"
        st = commerce.markdown_stage_now
        assert st(cfg, "food", 19 * 60 + 50, smin) == 0, f"Δt={dt}: 3 時間より前に見切っている"
        assert st(cfg, "food", 20 * 60, smin) == 1, f"Δt={dt}: 閉店 3 時間前が 1 段階目でない"
        assert st(cfg, "food", 21 * 60 + 30, smin) == 2, f"Δt={dt}: 閉店 1.5 時間前が 2 段階目でない"
    # 粗い Δt では step 格子への丸めが入る(9 step=90分 → Δt=20 では 4 step=80分)。
    # 実時刻は **±1 step 以内**にとどまる、が STEPS 変換の正直な約束(消える/倍になるはしない)。
    coarse = _sim(tmp_path, "pb_dt20", steps=1, **{**_MD, "run.dt_min": "20"})
    st, smin = commerce.markdown_stage_now, coarse.clock.step_minutes
    assert st(coarse.commercecfg, "food", 20 * 60, smin) == 1, "粗い Δt で 1 段階目が消えた"
    assert st(coarse.commercecfg, "food", 22 * 60, smin) == 2, "粗い Δt で 2 段階目が消えた"


def test_timeconv_classifies_the_new_time_keys():
    """★新 conf キーのうち時間・step 次元を持つものが timeconv の棚卸し契約に載っている。"""
    from society import timeconv as T
    assert T.covers("commerce.markdown.stage_steps")
    assert T.covers("commerce.markdown.stage_coefs")
    assert T.covers("commerce.price_schedule")
    assert T.covers("commerce.crowding.band_hours")
    cls = {k: c for k, c, _w in T.TABLE}
    assert cls["commerce.markdown.stage_steps"] == T.STEPS, \
        "見切りの段階は実時間の商習慣=STEPS(逆比例)でなければならない"
    assert cls["commerce.markdown.stage_coefs"] == T.INVARIANT
    assert cls["commerce.price_schedule"] == T.INVARIANT
    assert cls["commerce.crowding.band_hours"] == T.INVARIANT


# ------------------------------------------------------ B2 行動としての見切り
def test_markdown_is_the_staff_action_and_sets_the_coefficient(tmp_path):
    """在店店員の行動として markdown が出て、以後その店の価格に段階係数が乗る。"""
    sim = _sim(tmp_path, "pb_act", steps=1, **_MD)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    assert commerce.markdown_coef(sim, node, "food", 20 * 60) == 1.0, "貼る前から係数が効いている"
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 1
    ev = _kind(sim, "markdown")
    assert len(ev) == 1 and ev[0].agent_id == a.id, "当人の行為として出ていない"
    assert ev[0].payload["poi"] == node and ev[0].payload["stage"] == 1
    assert ev[0].payload["coef"] == 0.8 and "unstaffed" not in ev[0].payload
    assert commerce.markdown_coef(sim, node, "food", 20 * 60) == 0.8, "1 段階目の係数が効いていない"
    assert commerce.apply_price(sim, 1000.0, "food", node, 20 * 60) == 800.0
    assert any(commerce._MARKDOWN_TEXT in str(m.text) for m in a.mem.buffer), \
        "値札を替えたことが当人の記憶に残っていない"


def test_markdown_stage_advances_then_stops(tmp_path):
    """段階は 0.8 → 0.5 と**単調に**進み、同じ段階では二度と貼らない(1 店 1 step 1 件の担保)。"""
    sim = _sim(tmp_path, "pb_stage", steps=1, **_MD)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 1
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=2, sim_min=20 * 60 + 10) == 0, \
        "同じ段階でもう一度貼っている(L1 の洪水)"
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=3, sim_min=21 * 60 + 40) == 1
    ev = _kind(sim, "markdown")
    assert [e.payload["stage"] for e in ev] == [1, 2], "段階遷移が 1→2 になっていない"
    assert ev[-1].payload["coef"] == 0.5, "2 段階目が半額帯(0.5)になっていない"
    assert commerce.markdown_coef(sim, node, "food", 21 * 60 + 40) == 0.5


def test_markdown_is_one_event_per_store_per_step(tmp_path):
    """複数業態が同時に窓へ入っても、1 店 1 step に markdown は **1 件**へまとまる。"""
    sim = _sim(tmp_path, "pb_one", steps=1,
               **{**_MD, "commerce.markdown.cats": "[food, cafe]",
                  "commerce.hours.cafe": "[10, 23]"})
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    _stock(sim, node, "cafe", 10, day=0)
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 1
    ev = _kind(sim, "markdown")
    assert len(ev) == 1, f"1 店 1 step に markdown が {len(ev)} 件出ている"
    assert ev[0].payload["cats"] == ["cafe", "food"], "複数業態が 1 件へまとまっていない"
    assert commerce.markdown_coef(sim, node, "cafe", 20 * 60) == 0.8


def test_markdown_requires_shelf_and_todays_batch(tmp_path):
    """棚が空 / 当日の入荷バッチが無い店では値札を替えない(売り切るものが無い)。"""
    sim = _sim(tmp_path, "pb_cond", steps=1, **_MD)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 0, day=0)               # 棚が空
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 0
    _stock(sim, node, "food", 10, day=-1)             # 当日の入荷バッチが無い(昨日の納品)
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 0
    assert not _kind(sim, "markdown")


def test_markdown_resets_next_morning_and_at_closing(tmp_path):
    """翌朝(日鍵)と閉店で係数は 1.0 へ戻り、翌日はまた 1 段階目から貼り直す。"""
    sim = _sim(tmp_path, "pb_reset", steps=1, **_MD)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60)
    assert commerce.markdown_coef(sim, node, "food", 20 * 60) == 0.8
    assert commerce.markdown_coef(sim, node, "food", 23 * 60 + 30) == 1.0, "閉店後も見切りが効いている"
    nxt = 20 * 60 + 1440                              # 翌日の同じ時刻
    assert commerce.markdown_coef(sim, node, "food", nxt) == 1.0, "翌朝リセットされていない"
    _stock(sim, node, "food", 10, day=1)              # 翌日ぶんの入荷
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=200, sim_min=nxt) == 1
    assert _kind(sim, "markdown")[-1].payload["stage"] == 1, "翌日が 1 段階目から始まっていない"


def test_markdown_unstaffed_fallback(tmp_path):
    """担い手が 1 人も居ない POI だけ agent_id=-1 + unstaffed=true(INV-B と同一規約)。"""
    sim = _sim(tmp_path, "pb_uns", steps=1, **_MD)
    node = _food_node(sim)
    _stock(sim, node, "food", 10, day=0)
    # 担い手が「割り当てられているのに不在」= 誰も値札を替えない
    assert commerce.markdown_phase(sim, {}, {node}, step=1, sim_min=20 * 60) == 0
    assert not _kind(sim, "markdown")
    # 担い手ゼロの POI だけ、宣言つきでエンジンが代替再現する
    assert commerce.markdown_phase(sim, {}, set(), step=1, sim_min=20 * 60) == 1
    ev = _kind(sim, "markdown")[-1]
    assert ev.agent_id == -1 and ev.payload["unstaffed"] is True, "無人の標が立っていない"


def test_markdown_phase_is_noop_when_off(tmp_path):
    """OFF では窓の中でも 1 件も出ず、_md_stage も生えない。"""
    sim = _sim(tmp_path, "pb_mdoff", steps=1, **_INV)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    assert commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60) == 0
    assert not sim._md_stage and not _kind(sim, "markdown")


def test_markdown_rides_the_inventory_staff_phase(tmp_path):
    """★見切りは INV-B の staff_phase に**相乗り**する(見切り専用の勤務概念を作らない)。"""
    sim = _sim(tmp_path, "pb_ride", steps=1, **_MD)
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    goods_mod.staff_phase(sim, 1, 20 * 60)            # 2層 OFF でも見切りだけは回る
    ev = _kind(sim, "markdown")
    assert ev and ev[0].agent_id == a.id, "staff_phase 経由で当人の行動として出ていない"


# ------------------------------------------------------ 合成(B1 × B2)
def test_multiplier_composes_schedule_then_markdown(tmp_path):
    """合成は **時間帯係数 × 見切り係数**(式で順序を固定)。"""
    sim = _sim(tmp_path, "pb_mix", steps=1, **{**_MD, **_SCHED})
    node = _food_node(sim)
    a = _staff(sim, node, sim.agents[0])
    _stock(sim, node, "food", 10, day=0)
    commerce.markdown_phase(sim, {node: [a]}, None, step=1, sim_min=20 * 60)
    sched = commerce.price_schedule_coef(sim.commercecfg, "food", 20 * 60)
    mark = commerce.markdown_coef(sim, node, "food", 20 * 60)
    assert (sched, mark) == (1.67, 0.8), "前提(両方が効いている)が崩れている"
    assert commerce.price_multiplier(sim, "food", node, 20 * 60) == sched * mark
    assert commerce.apply_price(sim, 900.0, "food", node, 20 * 60) == 900.0 * (sched * mark)


# ------------------------------------------------------ 統合・決定論・搬送
#: 棚を薄く・便数を上げて「当日の入荷が棚に残ったまま夕方を迎える」状態を作る。
_THIN = {
    "commerce.inventory.enabled": "true",
    "commerce.markdown.enabled": "true",
    "commerce.inventory.capacity": "{food: 4, cafe: 4, shop: 4, nightlife: 4}",
    "commerce.inventory.reorder_point": "{food: 3, cafe: 3, shop: 3, nightlife: 3}",
    "commerce.inventory.default_capacity": "4",
    "commerce.inventory.default_reorder_point": "3",
    "commerce.inventory.lead_time_steps": "3",
    "commerce.inventory.review_every_steps": "6",
}
_SPLIT, _TOTAL = 82, 144


def _cfg_of(name, n_steps, every=0):
    ov = dict(_THIN)
    if every:
        ov["observer.checkpoint_every"] = str(every)
    dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir):
    import pyarrow.parquet as pq
    return pq.read_table(run_dir / "l1_events.parquet").to_pylist()


def test_run_emits_markdown_and_is_deterministic(tmp_path):
    """フル run で markdown が出て、同 seed 2 ランが L1 完全一致(決定論)。"""
    a = Simulation(_cfg_of("pb_det_a", _TOTAL), out_dir=tmp_path / "pb_det_a")
    a.run()
    assert _kind(a, "markdown"), "見切り ON なのに markdown が 1 件も出ていない"
    b = Simulation(_cfg_of("pb_det_b", _TOTAL), out_dir=tmp_path / "pb_det_b")
    b.run()
    assert _l1(a) == _l1(b), "PRICE-B ON の決定論が崩れている"


def test_resume_carries_the_markdown_stage(tmp_path):
    """★分割走行 == 一気通し: 見切りの段階と当日の入荷バッチが checkpoint を跨いで一致する。"""
    from society.engine import checkpoint

    st_dir = tmp_path / "pb_st"
    straight = Simulation(_cfg_of("pb_st", _TOTAL), out_dir=st_dir)
    straight.run()
    assert straight._md_stage, "空回り(見切りの段階が 1 件も立っていない)"

    d = tmp_path / "pb_rs"
    s1 = Simulation(_cfg_of("pb_rs", _SPLIT, every=_SPLIT), out_dir=d)
    for step in range(_SPLIT):
        scheduler.run_step(s1, step)
    assert s1._md_stage, "split 位置で見切りの段階が立っていない(位置の再調整が要る)"
    assert s1._goods_delivered, "split 位置で当日の入荷バッチが無い"
    checkpoint.save(s1, _SPLIT, d / "checkpoint" / f"ckpt-{_SPLIT:06d}.pkl.gz")
    s1.logger.flush_segment()

    s2 = Simulation(_cfg_of("pb_rs", _TOTAL, every=_SPLIT), out_dir=d)
    s2.run(resume_from=d)
    assert _rows(d) == _rows(st_dir), "PRICE-B ON の resume が straight と不一致"
    assert s2._md_stage == straight._md_stage, "見切りの段階が一気通しと食い違う"
    assert s2._goods_delivered == straight._goods_delivered, "当日の入荷バッチが食い違う"


def test_checkpoint_roundtrips_the_markdown_state(tmp_path):
    """段階と当日バッチが往復で 1 件残らず戻る。"""
    from society.engine import checkpoint

    src = _sim(tmp_path, "pb_ck_src", **_MD)
    node = _food_node(src)
    src._md_stage = {(node, "food"): [3, 2]}
    src._goods_delivered = {(node, "food"): 3}
    p = checkpoint.save(src, 5, tmp_path / "pb_ck_src" / "checkpoint" / "ckpt-000005.pkl.gz")

    dst = _sim(tmp_path, "pb_ck_dst", **_MD)
    assert checkpoint.load(dst, p) == 5
    assert dst._md_stage == {(node, "food"): [3, 2]}
    assert dst._goods_delivered == {(node, "food"): 3}


def test_old_checkpoint_without_the_new_keys_still_loads(tmp_path):
    """新キーを持たない旧 checkpoint でも落ちず、従来どおり空から始まる。"""
    import gzip
    import pickle

    from society.engine import checkpoint

    src = _sim(tmp_path, "pb_old_src", **_MD)
    src._md_stage = {(_food_node(src), "food"): [1, 1]}
    p = checkpoint.save(src, 3, tmp_path / "pb_old_src" / "checkpoint" / "ckpt-000003.pkl.gz")
    with gzip.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    for k in ("md_stage", "goods_delivered"):
        blob["runtime"].pop(k, None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _sim(tmp_path, "pb_old_dst", **_MD)
    assert checkpoint.load(dst, p) == 3
    assert dst._md_stage == {} and dst._goods_delivered == {}
