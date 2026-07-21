"""職場束ね直し(work.bind_workplace。既定 OFF)のテスト(第49バッチ)。

設計: docs/research/l2-work-reality.md §4「残課題(coverage)」。R1 の鉄則を継承:
- OFF(既定): 直接ラン L1 が純粋既定と完全一致(バイト一致)・pool 経路も work_node 付与状況/
  イベント列とも既定と一致・workplace_bound 0 件。
- ON: (a) work_node の付かない L2/L3(occupation が _WORK_CAT 非該当)が台帳 workplace_poi へ束なる。
      (b) coverage が実際に上がる(n_unbound_after < n_unbound_before・workplace_bound 1 件)。
      (c) 決定論=同 seed 2 ラン + hydrate 再入(build_pool_agent 再実行)で同一 work_node。
      (d) run.seed を変えても束ね(新規束ね分)は不変(hashlib 純関数=RngHub 非依存)。
      (e) serve 帰属が増える(束ねで生まれた勤務中スタッフに接客が帰属)。
検証は mock のみ(実LLM 禁止・≤24 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society import work as work_mod                   # noqa: E402
from society.agents.persona import _WORK_CAT           # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine import scheduler                    # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402
from society.observer.schema import EVENT_KINDS, Event  # noqa: E402


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。"""
    out = tmp_path_factory.mktemp("pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_sim(name, pool_dir, out_dir, n_steps=1, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=out_dir / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# ------------------------------------------------------ (schema) 種別登録
def test_schema_registered():
    assert "workplace_bound" in EVENT_KINDS


# ------------------------------------------------------ OFF: 直接ラン L1 が純粋既定と一致
def test_off_direct_run_matches_pure_default(tmp_path):
    """bind_workplace は pool 専用=直接ランでは明示 OFF が純粋既定と L1 完全一致(バイト一致)。"""
    def mk(name, **ov):
        dot = ["run.seed=42", "run.n_agents=25", "run.n_steps=24", f"run.name={name}"]
        dot += [f"{k}={v}" for k, v in ov.items()]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    pure = mk("pure"); pure.run()
    off = mk("off", **{"work.bind_workplace.enabled": "false"}); off.run()
    assert _l1(pure) == _l1(off)
    assert not [e for e in off.logger.events if e.kind == "workplace_bound"]


# ------------------------------------------------------ OFF: pool 経路も既定と完全一致
def test_off_pool_matches_default(small_pool, tmp_path):
    """pool ON + bind 明示 OFF が pool ON(bind キー無し)と L1 完全一致=work_node 付与状況も不変。"""
    base = _pool_sim("base", small_pool, tmp_path, n_steps=6)
    base.run()
    off = _pool_sim("off", small_pool, tmp_path, n_steps=6,
                    **{"work.bind_workplace.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    assert off._pool is not None
    assert not [e for e in off.logger.events if e.kind == "workplace_bound"]
    # work_node 付与状況(pid→bool)も完全一致
    wa = {a.pool_pid: bool(a.work_node) for a in base.agents}
    wb = {a.pool_pid: bool(a.work_node) for a in off.agents}
    assert wa == wb


# ------------------------------------------------------ ON: coverage が実際に上がる
def test_on_coverage_rises(small_pool, tmp_path):
    """ON で work_node の付かない L2/L3 が束なり coverage が上がる(after < before・event 1 件)。"""
    off = _pool_sim("cov_off", small_pool, tmp_path)
    on = _pool_sim("cov_on", small_pool, tmp_path,
                   **{"work.bind_workplace.enabled": "true"})
    on.run()
    stat = on._workbind_stat
    assert stat is not None
    assert stat["n_unbound_after"] < stat["n_unbound_before"], stat
    assert stat["n_bound"] > 0
    assert stat["n_unbound_before"] - stat["n_unbound_after"] == stat["n_bound"]
    # 起動時 1 件の世界イベント(agent_id=-1)
    wb = [e for e in on.logger.events if e.kind == "workplace_bound"]
    assert len(wb) == 1 and wb[0].agent_id == -1 and wb[0].step == 0
    assert wb[0].payload == stat
    # 束ねで present L2 の work_node 保有が実数として増える
    def bound_l2(sim):
        return sum(1 for a in sim.agents
                   if str(getattr(a, "pool_pid", "")).startswith("L2") and a.work_node)
    assert bound_l2(on) > bound_l2(off)
    # 束ね先は必ず現行地図のノード(経路探索の安全)
    for a in on.agents:
        if a.work_node:
            assert a.work_node in on.city.graph


# ------------------------------------------------------ ON: 決定論(同 seed 2 ラン一致)
def test_on_deterministic_same_seed(small_pool, tmp_path):
    """同 seed 2 ランで L1 完全一致 + 全 present の work_node 一致(決定論・乱数不使用)。"""
    a = _pool_sim("det_a", small_pool, tmp_path, n_steps=150,
                  **{"work.bind_workplace.enabled": "true"})
    a.run()
    b = _pool_sim("det_b", small_pool, tmp_path, n_steps=150,
                  **{"work.bind_workplace.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b)
    assert {x.pool_pid: x.work_node for x in a.agents} \
        == {x.pool_pid: x.work_node for x in b.agents}


# ------------------------------------------------------ ON: hydrate 再入で同一 work_node
def test_on_rebuild_same_work_node(small_pool, tmp_path):
    """build_pool_agent を同一 pid で再実行(=日境界ローテーション/hydrate 再入)しても同一 work_node。"""
    sim = _pool_sim("rebuild", small_pool, tmp_path,
                    **{"work.bind_workplace.enabled": "true"})
    bound = [a for a in sim.agents if a.work_node][:20]
    assert bound
    for a in bound:
        rec = sim._pool.get(a.pool_pid)
        again = sim.build_pool_agent(a.pool_pid, rec)
        assert again.work_node == a.work_node
        assert again.work_start_min == a.work_start_min


# ------------------------------------------------------ ON: run.seed を変えても束ね不変
def test_on_bind_seed_independent(small_pool, tmp_path):
    """新規に束ねた個体(occupation が _WORK_CAT 非該当)の work_node は run.seed に依らず不変。"""
    a = _pool_sim("si_a", small_pool, tmp_path,
                  **{"work.bind_workplace.enabled": "true"})
    b = _pool_sim("si_b", small_pool, tmp_path,
                  **{"work.bind_workplace.enabled": "true", "run.seed": "999"})
    da = {x.pool_pid: (x.work_node, x.occupation) for x in a.agents}
    db = {x.pool_pid: x.work_node for x in b.agents}
    common = [p for p in (set(da) & set(db)) if da[p][1] not in _WORK_CAT and da[p][0]]
    assert common, "新規束ねの共通 present pid が見つからない"
    for p in common:
        assert da[p][0] == db[p], f"束ねが run.seed 依存: {p}"


# ------------------------------------------------------ ON: serve 帰属が増える
def test_on_bind_increases_serve(small_pool, tmp_path):
    """束ねで生まれた勤務中スタッフに接客(serve)が帰属する(束ね無しなら不在=unstaffed)。"""
    sim = _pool_sim("serve", small_pool, tmp_path,
                    **{"work.bind_workplace.enabled": "true",
                       "work.service.enabled": "true"})
    # 束ねで food カテゴリの職場に付いた勤務中スタッフを1人選ぶ(取り違え排除に他は非スタッフ化)
    staff = None
    for a in sim.agents:
        if not a.work_node:
            continue
        if any(p.get("cat") == "food" for p in sim.city.pois_at_node(a.work_node)):
            staff = a
            break
    assert staff is not None, "food 職場に束ねたスタッフが居ない"
    for a in sim.agents:
        a.work_start_min = -1
    node = staff.work_node
    x, y = sim.city.node_xy(node)
    staff.node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440   # 常時勤務窓
    customer = next(a for a in sim.agents if a.id != staff.id)
    customer.node = node
    customer.x, customer.y = x, y

    def fire():
        since = len(sim.logger.events)
        sim.logger.log(Event(step=5, sim_min=700, agent_id=customer.id, kind="spend",
                             x=x, y=y, payload={"cat": "food"}))
        scheduler._phase_work_service(sim, 5, 700, since)
        return [e for e in sim.logger.events[since:] if e.kind == "serve"]

    staffed = fire()
    assert len(staffed) == 1 and staffed[0].agent_id == staff.id
    assert staffed[0].payload["customer"] == customer.id
    # 束ね無し(work_node を外す)なら同じ消費が unstaffed=帰属が消える(=束ねが serve を増やした)
    staff.work_node = ""
    unst = fire()
    assert len(unst) == 1 and unst[0].agent_id == -1
    assert unst[0].payload.get("unstaffed") is True


# ------------------------------------------------------ 束ね関数の単体(POI 決定論マッチ)
def test_bind_fallback_deterministic_pure_function():
    """台帳ノードが地図に無い時の POI 決定論マッチが (seed, key) の純関数=乱数ゼロ。"""
    u1 = work_mod._stable_uniform(20260722, "food\x1fL2_00000001")
    u2 = work_mod._stable_uniform(20260722, "food\x1fL2_00000001")
    assert u1 == u2 and 0.0 <= u1 < 1.0
    assert work_mod._stable_uniform(20260722, "food\x1fL2_00000002") != u1
