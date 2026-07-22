"""火種介入の実験条件化 = spark treatment(第53バッチ)のテスト。

設計: docs/plans/finals-day1-decisions.md D13 / src/society/spark.py。介入は初期条件・環境の差
(初期関係の束/初期資本・在庫/定期集会アンカー)としてのみ与える ON/OFF treatment。プロンプトへの
台本注入は一切しない。選抜は trait-blind(pid, spark.seed の安定ハッシュ純関数=無作為)。

方針(既存 R1 の鉄則を継承・mock のみ・実LLM 禁止):
- OFF(既定): 純粋既定と L1 完全一致(144step。ゴールデンは test_scenario が別途担保)。spark_roster 0 件・
  spark_anchor 属性なし・_spark_ids なし・"spark" stream を1本も引かない。
- ON: (1)決定論(同 seed 2 ラン一致)。(2)選抜が run.seed 非依存(spark.seed 固定で run.seed 変更→同一名簿)。
  (3)選抜が traits を読まない(traits 改変で名簿不変)。(a)closeness 注入・(b)money 上乗せ・(c)アンカー収束+
  bias 減衰。spark_roster 1件。k=free/off で LLM 呼数一致。事後トレーサ scripts/trace_spark.py が mock ランで動く。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from society import spark as _spark
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
import trace_spark as ts                              # noqa: E402

_ON = {"spark.enabled": "true"}
_OFF = {"spark.enabled": "false"}


def _sim(tmp_path, name, n=100, steps=1, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _roster(sim):
    ev = _kind(sim, "spark_roster")
    return ev[0].payload if ev else None


# --------------------------------------------------------------- schema 登録
def test_spark_roster_kind_registered():
    assert "spark_roster" in EVENT_KINDS


# --------------------------------------------------------------- OFF 既定一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144step)。spark_roster/spark_anchor/_spark_ids とも生えない。"""
    pure = _sim(tmp_path, "pure", n=20, steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", n=20, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(spark seam が no-op でない)"
    assert not _kind(pure, "spark_roster"), "OFF で spark_roster が出ている"
    assert not hasattr(pure, "_spark_ids"), "OFF なのに _spark_ids が生えている"
    assert all(not hasattr(a, "spark_anchor") for a in pure.agents), \
        "OFF なのに spark_anchor が生えている"


# --------------------------------------------------------------- ON 決定論
def test_on_deterministic(tmp_path):
    """spark ON 同士 2 回で L1 完全一致(決定論・mock 60step)。"""
    a = _sim(tmp_path, "det_a", n=60, steps=60, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=60, steps=60, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "spark ON の決定論が崩れている(専用 stream 以外の乱数漏れ?)"


def test_on_differs_from_off(tmp_path):
    """ON は OFF と L1 が異なる(=介入が実際に注入され行動が変わっている)。"""
    on = _sim(tmp_path, "diff_on", n=60, steps=60, **_ON)
    on.run()
    off = _sim(tmp_path, "diff_off", n=60, steps=60, **_OFF)
    off.run()
    assert _l1(on) != _l1(off)


# --------------------------------------------------------------- 選抜が run.seed 非依存
def test_selection_run_seed_independent(tmp_path):
    """spark.seed 固定で run.seed を変えても選抜名簿が不変(pid, spark.seed の純関数=比較実験の要)。"""
    a = _sim(tmp_path, "seed_a", n=100, **{**_ON, "run.seed": "1"})
    b = _sim(tmp_path, "seed_b", n=100, **{**_ON, "run.seed": "999999"})
    ra, rb = _roster(a), _roster(b)
    assert ra and rb, "spark_roster が出ていない"
    assert ra["ids"] == rb["ids"] and ra["pids"] == rb["pids"], \
        "run.seed を変えると選抜名簿が変わる(run.seed 非依存が崩れている)"
    assert ra["n"] == a.sparkcfg["n"] > 0


# --------------------------------------------------------------- 選抜が traits を読まない
def test_selection_is_trait_blind(tmp_path):
    """traits を改変しても _select の名簿が不変(選抜は pid のみの純関数=生得/創発の交絡を排除)。"""
    sim = _sim(tmp_path, "blind", n=100, **_ON)
    before = [a.id for a in _spark._select(sim, sim.sparkcfg)]
    for a in sim.agents:                              # traits を全面改変(選抜に影響してはならない)
        a.traits = {"nfc": 0.99, "risk_tolerance": 0.99, "internal_locus": 0.99,
                    "efficacy": 0.99, "ownership": 0.99, "world_change": 0.99}
    after = [a.id for a in _spark._select(sim, sim.sparkcfg)]
    assert before == after and before, "traits 改変で名簿が変わった(trait-blind 違反)"


# --------------------------------------------------------------- (a) closeness 注入
def test_relations_bundle_injected(tmp_path):
    """(a) sparked 相互に friend closeness/tier が注入される(t=0 の初期条件・減衰対象外)。"""
    sim = _sim(tmp_path, "rel", n=100, **_ON)
    ros = _roster(sim)
    ids = ros["ids"]
    clo = sim.sparkcfg["menus"]["relations"]["closeness"]
    tier = _spark._tier_for(sim, clo)
    a0 = sim.agent_by_id[ids[0]]
    a1 = sim.agent_by_id[ids[1]]
    rel = a0.mem.relations.get(a1.id)
    assert rel is not None and "closeness" in rel, "sparked 相互に関係が注入されていない"
    assert abs(float(rel["closeness"]) - clo) < 1e-9, "注入 closeness が config と不一致"
    assert int(rel["tier"]) == tier, "注入 tier が closeness と不整合"
    # 各 sparked に注入辺(closeness 付き)が複数ある(相互 + 近傍束)。
    for i in ids:
        injected = [r for r in sim.agent_by_id[i].mem.relations.values()
                    if "closeness" in r]
        assert len(injected) >= len(ids) - 1, "sparked の初期関係の束が薄すぎる"


# --------------------------------------------------------------- (b) money 上乗せ
def test_capital_money_overlay(tmp_path):
    """(b) sparked の money が OFF ベースに config の money を上乗せした値になる(同 run.seed で base 一致)。"""
    on = _sim(tmp_path, "cap_on", n=100, **_ON)      # 構築のみ(apply は __init__ で走る)
    off = _sim(tmp_path, "cap_off", n=100, **_OFF)
    ros = _roster(on)
    ids = set(ros["ids"])
    add = on.sparkcfg["menus"]["capital"]["money"]
    for aid, a_on in on.agent_by_id.items():
        a_off = off.agent_by_id[aid]
        if aid in ids:
            assert abs(a_on.money - (a_off.money + add)) < 1e-6, \
                f"sparked {aid} の money 上乗せが config と不一致"
        else:
            assert abs(a_on.money - a_off.money) < 1e-6, \
                f"非 sparked {aid} の money が動いている(介入が漏れている)"


# --------------------------------------------------------------- (c) アンカー収束 + bias 減衰
def test_anchor_common_poi_and_bias_decay(tmp_path):
    """(c) sparked 全員が共通の集会 POI を持ち、band 内で bias 抽選が step 経過で減衰する。"""
    # band を丸1日に広げ、bias=1.0・decay=0.05 に固定して早い step と遅い step の寄せを比較する。
    ov = {**_ON, "spark.menus.anchor.band": "[0,1440]",
          "spark.menus.anchor.bias": "1.0", "spark.menus.anchor.decay": "0.05"}
    sim = _sim(tmp_path, "anchor", n=100, **ov)
    ids = _roster(sim)["ids"]
    anchors = {sim.agent_by_id[i].spark_anchor["poi"] for i in ids}
    assert len(anchors) == 1 and next(iter(anchors)) is not None, \
        "sparked 全員が共通の集会 POI を持っていない"
    # 早い step(bias≈1)は寄せ多数、遅い step(bias≈e^-10≈0)は寄せほぼゼロ=指数減衰。
    early = sum(1 for i in ids
                if _spark.spark_dest(sim.agent_by_id[i], sim, 1, 10) is not None)
    late = sum(1 for i in ids
               if _spark.spark_dest(sim.agent_by_id[i], sim, 200, 2000) is not None)
    assert early > 0, "早い step でアンカーへ寄っていない(bias が効いていない)"
    assert early > late, f"bias が step で減衰していない: early={early} late={late}"


def test_spark_dest_no_draw_when_out_of_band(tmp_path):
    """band 外・非 sparked では spark_dest が None(乱数を引かない=既定 draw 順に不干渉)。"""
    ov = {**_ON, "spark.menus.anchor.band": "[600,660]"}
    sim = _sim(tmp_path, "band", n=60, **ov)
    ids = set(_roster(sim)["ids"])
    a = sim.agent_by_id[ids.pop()]
    assert _spark.spark_dest(a, sim, 5, 0) is None, "band 外で None を返していない"
    other = next(x for x in sim.agents if x.id not in set(_roster(sim)["ids"]))
    assert _spark.spark_dest(other, sim, 5, 630) is None, "非 sparked に spark_anchor が付いている"


# --------------------------------------------------------------- spark_roster 1件
def test_spark_roster_single_event(tmp_path):
    """ON で spark_roster が正確に1件・n>0・payload の型が固定。"""
    sim = _sim(tmp_path, "roster", n=100, **_ON)
    ev = _kind(sim, "spark_roster")
    assert len(ev) == 1, "spark_roster が1件でない"
    p = ev[0].payload
    assert ev[0].agent_id == -1 and ev[0].step == 0
    assert set(p) == {"ids", "pids", "n", "menus", "params"}
    assert p["n"] == len(p["ids"]) > 0
    assert set(p["menus"]) == {"relations", "capital", "anchor"}


# --------------------------------------------------------------- k=free/off 呼数一致
def test_call_count_k_invariant(tmp_path):
    """真の R1「呼数 k 非依存」: spark ON のまま k(writeback)を free/off に振っても LLM 呼数が不変
    (spark は物理量[現在地・時刻・step]と config・名簿のみを読み k 由来量を一切見ない)。"""
    ov = {"lod.max_llm_per_step": 2}
    free = _sim(tmp_path, "k_free", n=100, steps=24, **{**_ON, **ov, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", n=100, steps=24, **{**_ON, **ov, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"spark ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ================================================================= 事後トレーサ
def _ev(step, agent, kind, payload=None):
    return {"step": step, "agent": agent, "kind": kind, "payload": payload or {}}


def test_tracer_activity_ratio_pure():
    """活動量比: sparked の1人あたり行動数 / 非 sparked の比を正しく算出する(合成イベント)。"""
    events = [_ev(10, 1, "speak"), _ev(20, 1, "speak"), _ev(30, 1, "venture_open"),
              _ev(40, 2, "proposal"),                      # sparked=2 も1件
              _ev(50, 3, "speak")]                          # 非 sparked=3 が1件
    activity = ts.activity_by_agent(events)
    out = ts.group_activity(activity, sparked={1, 2}, residents=[1, 2, 3])
    assert out["total_sparked"] == 4 and out["total_others"] == 1
    assert out["mean_per_sparked"] == 2.0 and out["mean_per_other"] == 1.0
    assert out["activity_ratio"] == 2.0


def test_tracer_transmission_reach_pure():
    """transmission 波及: sparked 発 item の時間順 BFS で到達人数・hop を数える(合成イベント)。"""
    # item X: 1(sparked)→2→3 の2 hop 連鎖。1 は sparked。
    events = [_ev(10, 2, "transmission", {"item_id": "X", "from": 1, "channel": "face"}),
              _ev(20, 3, "transmission", {"item_id": "X", "from": 2, "channel": "face"}),
              # item Y: 非 sparked 発(5→6)=対象外。
              _ev(15, 6, "transmission", {"item_id": "Y", "from": 5, "channel": "sns"})]
    out = ts.spark_transmission_reach(events, sparked={1})
    assert out["n_items_seeded_by_sparked"] == 1
    assert out["total_reached"] == 2 and out["max_hops"] == 2


def test_tracer_end_to_end_mock_run(tmp_path):
    """トレーサが mock の spark ON ランで動く(spark_trace.json を書き・sparked を検出)。"""
    name = "trace_run"
    sim = _sim(tmp_path, name, n=60, steps=24, **_ON)
    sim.run()
    run_dir = tmp_path / name
    res = ts.trace(str(run_dir), str(run_dir / "panel"))
    assert res["spark_enabled"] is True and res["n_sparked"] > 0
    assert (run_dir / "panel" / "spark_trace.json").exists()
    assert set(res["activity"]) >= {"activity_ratio", "mean_per_sparked", "n_sparked"}
    # OFF ラン(spark_roster 不在)では graceful に spark_enabled=False。
    off_name = "trace_off"
    off = _sim(tmp_path, off_name, n=60, steps=24, **_OFF)
    off.run()
    off_res = ts.trace(str(tmp_path / off_name), str(tmp_path / off_name / "panel"))
    assert off_res["spark_enabled"] is False and off_res["n_sparked"] == 0
