"""摂動シナリオ(#8: baseline / shock_closure / shock_event)のテスト。

(a) baseline が「変更前」の L1 イベント列と完全一致(自前ゴールデン)。
    ゴールデンは scenario 層導入より前に採取した固定データ(tests/data/golden_baseline_l1.json)。
    自営の日銭(gig)は意図的な既定挙動追加なので economy.wages.自営=0 で中立化し、
    scenario/labeling/config スチュワードの各 seam が完全な no-op であることを固定する。
    加えて baseline == 発火しない shock(自己完結)で、シナリオ機構が不発時に何も足さないことを担保。
(b) shock_closure 発動中は封鎖半径内エッジを新規経路が避け、解除後に経路が復帰する。
(c) shock_event で世界ニュースが配信される。
(d) 発動・解除が scenario_shock イベントとして記録される。
"""
from __future__ import annotations

import json
from pathlib import Path

from society.config import load_config
from society.engine.simulation import Simulation
from society.world.scenario import _edge_key

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name, **ov):
    dot = ["run.seed=42", "run.n_agents=15", "run.n_steps=144", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


# --------------------------------------------------------------------------- #
def test_baseline_matches_prechange_golden(tmp_path):
    """既定(baseline)は scenario 層導入前の L1 と一字一句一致(gig 中立化で seam を隔離)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    # 意図的な既定挙動追加(gig / 朝の一日計画 / タクシー・バス)は中立化して、各 seam が
    # 変更前 L1 に対し完全な no-op であることを固定する(planning/taxi/bus off = 旧挙動)。
    sim = _run(tmp_path, "gbase", **{"economy.wages.自営": 0,
                                     "planning.enabled": "false",
                                     "transit_ride.taxi.enabled": "false",
                                     "transit_ride.bus.enabled": "false",
                                     "rules.enabled": "false"})
    assert _l1(sim) == golden, "baseline が変更前ゴールデンと不一致(seam が no-op でない)"


def test_baseline_equals_inactive_shock(tmp_path):
    """発火しない shock(at_step が run 末尾より後)は baseline と完全同一 = 機構の no-op。"""
    a = _run(tmp_path, "b_base")
    b = _run(tmp_path, "b_late", **{
        "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 9999, duration_steps: 3, radius_m: 150}"})
    assert _l1(a) == _l1(b)


def test_shock_closure_reroutes_and_restores(tmp_path):
    """封鎖中は新規経路が封鎖エッジを避け、解除後は封鎖前と同一経路に復帰する。"""
    sim = _run(tmp_path, "closure", **{
        "run.n_steps": 1, "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 2, duration_steps: 3, center: [0,0], radius_m: 150}"})
    sc = sim.scenario
    closed_preview = sc._edges_in_radius((0.0, 0.0), 150.0)
    assert closed_preview, "封鎖対象エッジがゼロ(半径/中心が不適切)"

    # 封鎖区間をまたぐ徒歩経路(gateway ペア)を探す
    gws = sim.city.gateways
    src = dst = pre = None
    for i in range(len(gws)):
        for j in range(len(gws)):
            if i == j:
                continue
            p = sim.router.route(gws[i], gws[j], "walk")[0]
            if len(p) >= 2 and any(_edge_key(u, v) in closed_preview
                                   for u, v in zip(p, p[1:])):
                src, dst, pre = gws[i], gws[j], p
                break
        if src:
            break
    assert src is not None, "封鎖区間をまたぐ経路が見つからない"

    # 発動 → 新規経路は封鎖エッジを1本も含まない
    sc.on_step(sim, 2)
    assert sc.active and sc.closed
    post = sim.router.route(src, dst, "walk")[0]
    assert not any(_edge_key(u, v) in sc.closed for u, v in zip(post, post[1:])), \
        "封鎖中の新規経路が封鎖エッジを通っている"
    assert post != pre, "封鎖で経路が変わっていない(避けていない)"

    # 解除 → 経路が封鎖前と一致(復帰)
    sc.on_step(sim, 5)
    assert not sc.active and not sc.closed
    restored = sim.router.route(src, dst, "walk")[0]
    assert restored == pre, "解除後に経路が封鎖前へ復帰していない"


def test_shock_closure_edges_flag_lifecycle(tmp_path):
    """発動で closed フラグが立ち、解除で全撤去される(グラフに残さない)。"""
    sim = _run(tmp_path, "flag", **{
        "run.n_steps": 1, "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 0, duration_steps: 2, center: [0,0], radius_m: 120}"})
    sc = sim.scenario
    sc.on_step(sim, 0)
    n_closed = sum(1 for _u, _v, d in sim.city.graph.edges(data=True) if d.get("closed"))
    assert n_closed == len(sc.closed) > 0
    sc.on_step(sim, 2)                                     # 解除
    left = sum(1 for _u, _v, d in sim.city.graph.edges(data=True) if d.get("closed"))
    assert left == 0, "解除後も closed フラグが残っている"


def test_shock_event_publishes_news(tmp_path):
    """shock_event で世界ニュース(world_event)が配信される。"""
    sim = _run(tmp_path, "sevent", **{
        "run.n_steps": 6, "world.scenario": "shock_event",
        "world.scenario_params":
            "{at_step: 1, title: 駅前で封鎖騒ぎ, text: 混雑に注意, word: 封鎖祭り}"})
    world_evs = [e for e in sim.logger.events if e.kind == "world_event"]
    assert any(e.payload.get("title") == "駅前で封鎖騒ぎ" for e in world_evs), \
        "shock_event のニュースが出ていない"
    # 注入した語がメディア発の item として登録される
    assert "封鎖祭り" in sim.labels.text_to_item


def test_scenario_shock_logged(tmp_path):
    """発動・解除が scenario_shock イベントとして記録される(closure=start/end)。"""
    sim = _run(tmp_path, "log", **{
        "run.n_steps": 8, "world.scenario": "shock_closure",
        "world.scenario_params": "{at_step: 1, duration_steps: 2, center: [0,0], radius_m: 120}"})
    shocks = [e for e in sim.logger.events if e.kind == "scenario_shock"]
    phases = [e.payload["phase"] for e in shocks]
    assert "start" in phases and "end" in phases, f"start/end が揃っていない: {phases}"
    start = next(e for e in shocks if e.payload["phase"] == "start")
    assert start.payload["kind"] == "shock_closure"
    assert start.payload["n_edges"] > 0
