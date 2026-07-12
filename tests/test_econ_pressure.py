"""経済的生活圧(Wave G1 2026-07-07)= 相対的剥奪 + 固定費 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): relative_deprivation の state_update / fixed_cost の spend が 0 件・
  イベント列は純粋既定と L1 完全一致(既存挙動に無影響=ゴールデンを守る)。
- 相対的剥奪 ON: 参照集団(街に居る他者)の所持金中央値を下回る agent の grievance が
  上がり、中央値以上は上がらない。より低い所持金ほど大きく上がる(正規化差に比例)。
- 固定費 ON: 日次で spend(cat=fixed_cost)が出て money が減る。既定 0 で出ない。
- 決定論: ON 同士2回で L1 完全一致。
- R1 呼数不変: 応答固定 backend(FixedLLM)で ON/OFF の generate 呼数が一致(非LLM機構)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 20, steps: int = 1, *,
         rd: float | None = None, fixed: float | None = None, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if rd is not None:
        dot.append(f"factors.relative_deprivation_grievance={rd}")
    if fixed is not None:
        dot.append(f"economy.fixed_cost_daily={fixed}")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _rd_ups(sim):
    return [e for e in sim.logger.events if e.kind == "state_update"
            and e.payload.get("cause") == "relative_deprivation"]


def _fc_spends(sim, agent_id=None):
    return [e for e in sim.logger.events if e.kind == "spend"
            and e.payload.get("cat") == "fixed_cost"
            and (agent_id is None or e.agent_id == agent_id)]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(係数0.0)と純粋既定(config.yaml の既定0.0)が L1 完全一致(144 step)。

    OFF 時は relative_deprivation / fixed_cost のイベントが1件も出ない(seam が完全 no-op)。
    """
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, rd=0.0, fixed=0.0)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G1 seam が no-op でない)"
    for s in (pure, off):
        assert not _rd_ups(s), "OFF で relative_deprivation の state_update が出ている"
        assert not _fc_spends(s), "OFF で fixed_cost の spend が出ている"


# --------------------------------------------------------------- 相対的剥奪
def test_relative_deprivation_grievance_by_rank(tmp_path):
    """相対的剥奪 ON: 中央値未満の agent は grievance↑、中央値以上は不変。より低い所持金ほど大きい。

    money_pressure(閾値2000)の交絡を避けるため全員を閾値より上に置き、gig を中立化(自営=0)。
    """
    sim = _sim(tmp_path, "rd", n=20, rd=0.012, **{"economy.wages.自営": 0})
    for i, a in enumerate(sim.agents):
        a.loc = "street"
        a.money = 3000.0 + i * 1000.0        # 3000..22000(全員 >2000)。中央値=12500
    before = {a.id: a.states["grievance"] for a in sim.agents}
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)

    low = sim.agents[0]                       # 3000 << 中央値 → grievance↑
    mid_below = sim.agents[9]                 # 12000 < 中央値(すぐ下)→ 小さく↑
    high = sim.agents[-1]                     # 22000 ≥ 中央値 → 不変
    d_low = low.states["grievance"] - before[low.id]
    d_near = mid_below.states["grievance"] - before[mid_below.id]
    assert d_low > 0.0, "中央値未満の agent の grievance が上がっていない"
    assert high.states["grievance"] == before[high.id], "中央値以上の agent の grievance が動いた"
    assert d_low > d_near > 0.0, "所持金が低いほど大きく上がっていない(正規化差に比例していない)"
    assert _rd_ups(sim), "relative_deprivation の state_update が出ていない"


def test_relative_deprivation_magnitude_formula(tmp_path):
    """単体: 実際の grievance 増分が mag = coef × min(1, (median-money)/max(median,1)) に一致。"""
    sim = _sim(tmp_path, "rdmag", n=20, rd=0.012, **{"economy.wages.自営": 0})
    for i, a in enumerate(sim.agents):
        a.loc = "street"
        a.money = 3000.0 + i * 1000.0
    a0 = sim.agents[0]
    g0 = a0.states["grievance"]
    ref = scheduler._money_median(sim)        # gig 中立化・固定費OFF=phase 後も money 不変
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    exp = 0.012 * min(1.0, (ref - a0.money) / max(ref, 1.0))
    assert abs((a0.states["grievance"] - g0) - exp) < 1e-9, \
        f"magnitude 不一致: 実{a0.states['grievance'] - g0} 期待{exp}"


# --------------------------------------------------------------- 固定費
def test_fixed_cost_daily_spend(tmp_path):
    """固定費 ON: 日次で spend(cat=fixed_cost)が出て money が減る。既定 0 では出ない。"""
    sim = _sim(tmp_path, "fc", n=20, fixed=300, **{"economy.wages.自営": 0})
    a = sim.agents[0]
    a.loc, a.visitor, a.money = "street", False, 5000.0
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    sp = _fc_spends(sim, a.id)
    assert sp, "fixed_cost の spend が出ていない"
    assert sp[-1].payload["amount"] == 300.0
    assert a.money == 5000.0 - 300.0, "固定費が money から控除されていない"

    off = _sim(tmp_path, "fc_off", n=20, fixed=0.0)
    off.agents[0].loc = "street"
    off._econ_day = -1
    scheduler._phase_daily(off, step=0, sim_min=0)
    assert not _fc_spends(off), "既定0.0 で fixed_cost の spend が出ている"


def test_fixed_cost_skips_visitor(tmp_path):
    """固定費は来街者(街の外に住居)には掛からない(既存 rent と同じ扱い)。"""
    sim = _sim(tmp_path, "fcvis", n=20, fixed=300, **{"economy.wages.自営": 0})
    vis = next((a for a in sim.agents if a.visitor), None)
    if vis is None:                           # 手続き生成に来街者が居なければ人工的に設定
        vis = sim.agents[0]
        vis.visitor = True
    vis.loc = "street"
    m0 = vis.money
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    assert not _fc_spends(sim, vis.id), "来街者に fixed_cost が掛かっている"
    assert vis.money == m0, "来街者の money が固定費で減っている"


# --------------------------------------------------------------- 決定論 / R1 呼数
def test_on_deterministic(tmp_path):
    """相対的剥奪+固定費 ON 同士2回で L1 完全一致。ON で新イベントが実際に出る(配線が効く)。"""
    a = _sim(tmp_path, "on_a", n=20, steps=144, rd=0.012, fixed=300)
    a.run()
    b = _sim(tmp_path, "on_b", n=20, steps=144, rd=0.012, fixed=300)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    assert _rd_ups(a), "ON で relative_deprivation の state_update が出ていない"
    assert _fc_spends(a), "ON で fixed_cost の spend が出ていない"


class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, *, rd, fixed, n=20, steps=72):
    # money_pressure(閾値2000)の発火動態を無効化(閾値0)して G1 機構を隔離する。
    # 相対的剥奪は drive.add を呼ばず、固定費は非LLM の spend=どちらも generate を増やさない。
    sim = _sim(tmp_path, name, n=n, steps=steps, rd=rd, fixed=fixed,
               **{"economy.money_pressure_threshold": 0, "economy.wages.自営": 0})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend なら ON/OFF で generate 呼数が完全一致(R1: 非LLM機構=呼数不変)。"""
    on = _run_fixed(tmp_path, "cc_on", rd=0.012, fixed=300)
    off = _run_fixed(tmp_path, "cc_off", rd=0.0, fixed=0.0)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert _rd_ups(on), "ON で relative_deprivation が出ていない(配線未接続)"
    assert not _rd_ups(off), "OFF で relative_deprivation が出ている"
