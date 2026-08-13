"""レーン丙(第113): 痩せ修正の仕上げ小粒 5 件。

  1  A12 車両資産の**日次追随**(保存則安全版 = 境界流入 rot_in を恒等式の流入側に立てる)
  2  A13 **日次入場者名簿サイドカー** roster.parquet(既定 OFF / finals ON)
  3  E5 職場束ね **coverage の観測**(累計ではなく最終日の在場者を分母にした保持率)
  4  回転を跨いだ在院者の**退院位置の整合**(node と x/y の食い違い)
  5  相手不在 unbond が残す**片側 partner_id の清算**(入場時整合)

**共通の受入条件**: 当該機構 OFF / 回転の無いランでは 1 バイトも変わらないこと。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from society import assets as A                       # noqa: E402
from society import household as HH                   # noqa: E402
from society import medical as M                      # noqa: E402
from society import work as W                         # noqa: E402
from society.agents.agent import Agent                # noqa: E402
from society.agents.memory import MemoryStore         # noqa: E402
from society.config import load_config                # noqa: E402
from society.engine import scheduler                  # noqa: E402
from society.engine.simulation import Simulation      # noqa: E402
from society.observer import roster as RO             # noqa: E402

ASSETS_ON = {"world.assets.enabled": "true", "organizations.enabled": "true",
             "household.enabled": "true"}


def _cfg(name, n_steps=24, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _armed(tmp_path, name, n_agents=20, **ov):
    """``arm`` だけを走らせた登記簿(test_assets._armed と同型 = 速い)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents, **dict(ASSETS_ON, **ov))
    scheduler._ensure_orgs(sim)
    A.arm(sim, 0, 0)
    return sim


def _newcomer(sim, has_car=True):
    """途中入場者の最小形(プール回転で実体化される個体と同じ「新しい id + has_car」)。"""
    aid = max(int(a.id) for a in sim.agents) + 1
    a = Agent(id=aid, name=f"新{aid}", age=33, occupation="会社員", persona="p",
              traits={"openness": 0.5}, states={}, mem=MemoryStore())
    a.has_car = bool(has_car)
    a.home_node = str(sorted(sim.city.graph.nodes)[0])
    a.x, a.y = sim.city.node_xy(a.home_node)
    return a


# =========================================================================== #
# 1  A12 車両資産の日次追随(保存則安全版)
# =========================================================================== #
def test_hei1_late_entrant_vehicle_is_registered_at_the_day_boundary(tmp_path):
    """途中入場者の車が台帳に載る(第109 は day0 在場者の車しか登記していなかった)。"""
    sim = _armed(tmp_path, "hei1_reg")
    st = A.state_of(sim)
    stock0 = dict(st["stock0"])
    live0 = A.n_assets(sim, A.VEHICLE)

    newbie = _newcomer(sim)
    sim.agents.append(newbie)
    assert A.owner_of(sim, A.vehicle_id(newbie.id)) is None, "前提が崩れている"

    A.phase(sim, 144, 1440)                            # 翌日の日境界
    assert A.owner_of(sim, A.vehicle_id(newbie.id)) == A.agent_party(newbie.id)
    assert A.n_assets(sim, A.VEHICLE) == live0 + 1
    # ★保存則の**右辺**(初期ストック)は 1 も動かない = 検査が無内容にならない
    assert st["stock0"] == stock0, "stock0(保存則の右辺)が走行中に動いた"


def test_hei1_conservation_stays_green_with_boundary_inflow(tmp_path):
    """境界流入は恒等式の**流入側**に立つ: live + k5 − born − rot_in = stock0。"""
    sim = _armed(tmp_path, "hei1_cons")
    sim.agents.append(_newcomer(sim))
    sim.agents.append(_newcomer(sim))
    A.phase(sim, 144, 1440)
    cons = A.conservation(sim)
    assert cons[A.VEHICLE]["rot_in"] == 2
    assert cons[A.VEHICLE]["born"] == 0 and cons[A.VEHICLE]["k5"] == 0
    for cat, c in cons.items():
        assert c["residual"] == 0, (cat, c)
    prov = A.provenance(sim)
    assert prov["rot_in"] == {A.VEHICLE: 2}
    assert prov["conservation"][A.VEHICLE]["residual"] == 0


def test_hei1_daily_followup_is_idempotent_and_zero_random(tmp_path):
    """同じ日に 2 度・翌日にもう一度通しても登記は増えない(冪等・乱数ゼロ)。"""
    sim = _armed(tmp_path, "hei1_idem")
    sim.agents.append(_newcomer(sim))
    A.phase(sim, 144, 1440)
    n1 = A.n_assets(sim, A.VEHICLE)
    A.phase(sim, 145, 1450)                            # 同じ日 = 走査ごとスキップ
    A.phase(sim, 288, 2880)                            # 翌日 = 走査するが既に登記済み
    assert A.n_assets(sim, A.VEHICLE) == n1
    assert A.conservation(sim)[A.VEHICLE]["rot_in"] == 1


def test_hei1_is_a_no_op_without_new_entrants(tmp_path):
    """新規入場が無いラン(既定プロファイル = pool OFF)では 1 行も増えない=バイト不変。"""
    sim = _armed(tmp_path, "hei1_noop")
    before = dict(A.conservation(sim))
    for step, sim_min in ((144, 1440), (288, 2880), (432, 4320)):
        A.phase(sim, step, sim_min)
    assert A.conservation(sim) == before
    assert A.provenance(sim)["rot_in"] == {}


def test_hei1_dead_entrants_do_not_get_a_vehicle_row(tmp_path):
    """死者の車は登記しない(初期配賦と同じ述語)。"""
    sim = _armed(tmp_path, "hei1_dead")
    ghost = _newcomer(sim)
    ghost.dead = True
    sim.agents.append(ghost)
    A.phase(sim, 144, 1440)
    assert A.owner_of(sim, A.vehicle_id(ghost.id)) is None
    assert A.conservation(sim)[A.VEHICLE]["rot_in"] == 0


def test_hei1_analyze_script_agrees_with_the_new_identity(tmp_path):
    """★解析(scripts/analyze_assets.py)も**独立に**数え直して残差 0 に達する。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import analyze_assets as AA

    sim = _armed(tmp_path, "hei1_an")
    sim.agents.append(_newcomer(sim))
    A.phase(sim, 144, 1440)
    A.dump(sim)
    ledger = AA.load_ledger(sim.out_dir)
    assert ledger["rot_in"] == {A.VEHICLE: 1}, "サイドカーに帳尻項が載っていない"
    res = AA.summarize(ledger, AA.load_transfers(sim.out_dir),
                       AA.load_last_cash(sim.out_dir), last_step=144)
    for cat, c in res["conservation"].items():
        assert c["residual"] == 0, (cat, c)
    assert res["conservation_agrees"] is True
    assert "PASS" in AA.render(res)


def test_hei1_analyze_script_is_unchanged_for_old_ledgers(tmp_path):
    """帳尻項の欄が無い**旧ラン**の台帳では式が live − stock0 へ退化する(後方互換)。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import analyze_assets as AA

    ledger = {"schema": A.SCHEMA, "n_rows": 2, "stock0": {A.VEHICLE: 2},
              "landlords": [], "org_source": "none",
              "rows": [{"asset": "vh:1", "cat": A.VEHICLE, "right": A.OWN,
                        "party": A.agent_party(1), "since": 0,
                        "building": "", "floor": 0, "node": ""},
                       {"asset": "vh:2", "cat": A.VEHICLE, "right": A.OWN,
                        "party": A.agent_party(2), "since": 0,
                        "building": "", "floor": 0, "node": ""}]}
    res = AA.summarize(ledger, [], {}, last_step=10)
    assert res["conservation"][A.VEHICLE] == {
        "live": 2, "born": 0, "k5": 0, "rot_in": 0, "stock0": 2, "residual": 0}


# =========================================================================== #
# 2  A13 日次入場者名簿サイドカー
# =========================================================================== #
class _RosterSim:
    """``RosterDaily`` が読む面だけを持つ最小のシム(観測側が世界に触らないことの裏返し)。"""

    def __init__(self, agents):
        self.agents = list(agents)


def _fake_agent(aid: int, **ov):
    a = Agent(id=aid, name=f"人{aid}", age=20 + aid, occupation="会社員",
              persona="p", traits={"openness": 0.25, "extraversion": 0.75},
              states={}, mem=MemoryStore())
    for k, v in ov.items():
        setattr(a, k, v)
    return a


def test_hei2_roster_is_off_by_default(tmp_path):
    """既定 OFF = オブジェクトを作らず 1 ファイルも書かない(既存ラン無風)。"""
    sim = _sim(tmp_path, "hei2_off", n_steps=2, n_agents=6)
    assert sim.roster_sc is None
    sim.run()
    assert not (sim.out_dir / "roster.parquet").exists()
    assert not list(sim.out_dir.glob("roster.part-*.parquet"))


def test_hei2_each_agent_is_written_once_with_its_entry_day(tmp_path):
    """1 個体 1 行(入場した日)。同じ日に 2 度撮っても 2 行目は作らない。"""
    import pyarrow.parquet as pq

    sc = RO.RosterDaily(tmp_path / "r1")
    sim = _RosterSim([_fake_agent(1), _fake_agent(2)])
    assert sc.on_step(sim, 0, 0) == 2
    assert sc.on_step(sim, 1, 10) == 0, "同じ日に 2 度撮った"
    sim.agents.append(_fake_agent(3))                  # 翌日の入場者
    assert sc.on_step(sim, 144, 1440) == 1
    assert sc.on_step(sim, 288, 2880) == 0, "再来街で 2 行目を作っている"
    sc.finalize()
    tbl = pq.read_table(tmp_path / "r1" / "roster.parquet")
    got = {int(r["agent_id"]): int(r["day"]) for r in tbl.to_pylist()}
    assert got == {1: 0, 2: 0, 3: 1}


def test_hei2_row_carries_the_agents_json_and_traits_columns(tmp_path):
    """agents.json 同等の欄 + traits 同等列(trait 名はコードに書かず JSON 1 列)。"""
    import pyarrow.parquet as pq

    sc = RO.RosterDaily(tmp_path / "r2")
    a = _fake_agent(7, has_car=True, home_building="b1", home_floor=3,
                    work_node="n9", org_id="o5", household_id="hp3",
                    wage=12345.0, money=678.9)
    sc.on_step(_RosterSim([a]), 0, 0)
    sc.finalize()
    row = pq.read_table(tmp_path / "r2" / "roster.parquet").to_pylist()[0]
    for col in ("day", "agent_id", "name", "age", "occupation", "visitor",
                "has_car", "home_building", "home_floor", "work_node",
                "org_id", "household_id", "money", "wage", "traits_json"):
        assert col in row, col
    assert row["has_car"] is True and row["home_floor"] == 3
    assert row["org_id"] == "o5" and row["household_id"] == "hp3"
    assert json.loads(row["traits_json"]) == a.traits


def test_hei2_seen_set_survives_a_resume(tmp_path):
    """★resume==straight: 既存 part から ``_seen`` を復元し、2 行目を作らない。"""
    sc = RO.RosterDaily(tmp_path / "r3")
    sim = _RosterSim([_fake_agent(1), _fake_agent(2)])
    sc.on_step(sim, 0, 0)
    sc.flush_segment()                                 # checkpoint 相当
    again = RO.RosterDaily(tmp_path / "r3")            # resume 相当(新プロセス)
    again._resumed = True                              # Simulation.run() の resume 分岐が立てる印
    assert again.capture(sim, 1) == 0, "resume 後に在場者が丸ごと 2 行目になった"
    sim.agents.append(_fake_agent(3))
    assert again.capture(sim, 1) == 1, "resume 後の新規入場者が拾えていない"


def test_hei2_fresh_run_into_a_dirty_dir_still_writes(tmp_path):
    """resume でない(``_resumed`` が立たない)ランは旧ファイルを既出扱いしない。"""
    sc = RO.RosterDaily(tmp_path / "r5")
    sim = _RosterSim([_fake_agent(1), _fake_agent(2)])
    sc.on_step(sim, 0, 0)
    sc.finalize()
    fresh = RO.RosterDaily(tmp_path / "r5")            # 同じ out_dir へ fresh に書く
    assert fresh.capture(sim, 0) == 2, "旧 canonical のせいで 1 行も書けなくなっている"


def test_hei2_capture_does_not_touch_the_world(tmp_path):
    """観測側だけで閉じる: 個体の属性を 1 つも書き換えない。"""
    sc = RO.RosterDaily(tmp_path / "r4")
    a = _fake_agent(1)
    before = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    sc.on_step(_RosterSim([a]), 0, 0)
    after = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    assert set(before) == set(after) and before["node"] == after["node"]


def test_hei2_toggle_is_declared_in_the_registry():
    from society import registry as R
    feat = {f.id: f for f in R.FEATURES}["observer.roster_daily.enabled"]
    assert feat.off_value is False and feat.affects_k is False


def test_hei2_roster_module_is_not_in_the_frozen_spec():
    """凍結 14 ファイル(metrics_spec)に触れていない = spec hash 無風。"""
    from society.observer import metrics_spec as MS
    assert "src/society/observer/roster.py" not in MS.SPEC_FILES


# =========================================================================== #
# 3  E5 職場束ね coverage の観測
# =========================================================================== #
def test_hei3_coverage_key_is_absent_without_a_pool(tmp_path):
    """pool OFF(既定・名簿ラン)では provenance が None = summary にキーが生えない。"""
    sim = _sim(tmp_path, "hei3_off", n_steps=2, n_agents=6)
    assert W.provenance(sim) is None
    summary = sim.run()
    assert "workplace_coverage" not in summary


def test_hei3_has_workplace_is_the_same_predicate_as_the_binder():
    """coverage の述語が ``bind_workplace`` の had/has と 1 文字も違わない。"""
    a = _fake_agent(1)
    assert W.has_workplace(a) is False
    a.work_node = "n1"
    assert W.has_workplace(a) is False, "勤務窓の無い work_node を数えている"
    a.work_start_min = 540
    assert W.has_workplace(a) is True


# =========================================================================== #
# 4  回転を跨いだ在院者の退院位置
# =========================================================================== #
def _admit(sim, patient, node):
    """在院中の状態を手で据える(``_arrive`` が置くのと同じ欄だけ)。"""
    patient.med_admitted = True
    patient.med_node = str(node)
    patient.med_poi = "hospital"
    patient.med_confirmed = 2
    patient.med_since = 0
    patient.med_until = 1
    patient.sleeping = True
    patient.building = None
    patient.floor = 0
    patient.node = str(node)
    patient.x, patient.y = sim.city.node_xy(node)


def _discharge_now(sim, patient, step=1, sim_min=10):
    cfg = M.cfg_of(sim)
    st = M._state(sim)
    bud = M._Budget(sim, st, int(cfg["max_events_per_step"]))
    M._discharge(sim, cfg, st, bud, patient, step, sim_min, None)


def test_hei4_discharge_across_a_rotation_lands_on_the_hospital_node(tmp_path):
    """★回転で再入場した在院者の退院: node と x/y が病院で一致する(食い違いの解消)。"""
    sim = _sim(tmp_path, "hei4_rot", n_steps=1, n_agents=8, **{"medical.enabled": "true"})
    nodes = sorted(sim.city.graph.nodes)
    hosp, elsewhere = nodes[0], nodes[-1]
    patient = sorted(sim.agents, key=lambda a: int(a.id))[0]
    _admit(sim, patient, hosp)
    # プール回転の再入場 = record から作り直された状態(building は落ち、位置は入場地点)
    patient.building = None
    patient.node = str(elsewhere)
    patient.x, patient.y = sim.city.node_xy(elsewhere)

    _discharge_now(sim, patient)
    out = [e for e in sim.logger.events if e.kind == "hospital_discharge"]
    assert len(out) == 1
    assert str(patient.node) == str(hosp), "退院しても再入場地点に居る"
    assert (patient.x, patient.y) == sim.city.node_xy(hosp)
    assert (round(out[0].x, 6), round(out[0].y, 6)) == \
        (round(patient.x, 6), round(patient.y, 6))
    assert str(out[0].payload["node"]) == str(hosp)


def test_hei4_discharge_is_unchanged_when_the_patient_never_left(tmp_path):
    """毎日在場のラン(既定プロファイル)では 1 行も通らない=バイト不変。"""
    sim = _sim(tmp_path, "hei4_same", n_steps=1, n_agents=8,
               **{"medical.enabled": "true"})
    hosp = sorted(sim.city.graph.nodes)[0]
    patient = sorted(sim.agents, key=lambda a: int(a.id))[0]
    _admit(sim, patient, hosp)
    xy = (patient.x, patient.y)
    _discharge_now(sim, patient)
    assert str(patient.node) == str(hosp) and (patient.x, patient.y) == xy


# =========================================================================== #
# 5  片側 partner_id の清算
# =========================================================================== #
class _PartnerSim:
    """``reconcile_partner`` が読む面だけを持つ最小のシム。"""

    def __init__(self, agents, pool_bind=True):
        self._present = {int(a.id): a for a in agents}
        self._pool = object() if pool_bind else None
        self.householdcfg = HH.build_cfg(
            {"enabled": True, "pool_bind": {"enabled": bool(pool_bind)}})

    def present_agent(self, aid):
        return self._present.get(int(aid))


def test_hei5_one_sided_partner_is_cleared_on_entry():
    """相手不在の unbond が残した**片側だけ既婚**を入場時に清算する。"""
    a, b = _fake_agent(1), _fake_agent(2)
    a.partner_id = None                                # 別れた側(unbond を通った)
    b.partner_id = int(a.id)                           # 街に居なかったので消せなかった側
    sim = _PartnerSim([a, b])
    assert HH.reconcile_partner(sim, b) is True
    assert b.partner_id is None
    assert HH.reconcile_partner(sim, b) is False, "冪等でない"


def test_hei5_a_healthy_couple_is_left_alone():
    a, b = _fake_agent(1), _fake_agent(2)
    a.partner_id, b.partner_id = int(b.id), int(a.id)
    sim = _PartnerSim([a, b])
    assert HH.reconcile_partner(sim, b) is False
    assert b.partner_id == int(a.id) and a.partner_id == int(b.id)


def test_hei5_absent_partner_is_left_pending():
    """相手が街に居ない回は**何もしない**(判定材料が無い = 正直な限界)。"""
    b = _fake_agent(2)
    b.partner_id = 1                                   # 相手は退場中
    sim = _PartnerSim([b])
    assert HH.reconcile_partner(sim, b) is False
    assert b.partner_id == 1


def test_hei5_is_a_no_op_when_pool_bind_is_off():
    """トグル配下(既定 OFF)= 1 行も通らない=現行のまま。"""
    a, b = _fake_agent(1), _fake_agent(2)
    a.partner_id = None
    b.partner_id = int(a.id)
    sim = _PartnerSim([a, b], pool_bind=False)
    assert HH.reconcile_partner(sim, b) is False
    assert b.partner_id == int(a.id)


# =========================================================================== #
# ON スモーク(プール回転あり・mock)= 2 / 3 の実測
# =========================================================================== #
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない。test_lane_otsu と同型)。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("pool_hei")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


@pytest.fixture(scope="module")
def rot_run(small_pool, tmp_path_factory):
    """回転を跨ぐ mock ラン(roster ON / org 配属 ON / 職場束ね ON / 登記簿 ON / 世帯 ON)。"""
    out = tmp_path_factory.mktemp("hei_on")
    dot = ["run.seed=42", "run.n_agents=10", "run.n_steps=200", "run.name=hei_on",
           "model.backend=mock", "observer.snapshot_every=144",
           "pool.enabled=true", f"pool.dir={small_pool}", "pool.present_cap=400",
           "observer.roster_daily.enabled=true",
           "organizations.enabled=true",
           "work.bind_workplace.enabled=true",
           "world.assets.enabled=true",
           "household.enabled=true", "household.pool_bind.enabled=true"]
    sim = Simulation(load_config(dot), out_dir=out / "on")
    summary = sim.run()
    return sim, summary


def test_hei1_smoke_rotation_registers_vehicles_and_keeps_the_residual_zero(rot_run):
    """★ON スモーク: 回転で入った車が登記され、それでも Σ 残差が 0 のまま。"""
    sim, summary = rot_run
    prov = summary["assets"]
    assert prov["rot_in"].get(A.VEHICLE, 0) > 0, \
        "回転で入場した個体の車が 1 台も登記されていない(日次追随が効いていない)"
    for cat, c in prov["conservation"].items():
        assert c["residual"] == 0, (cat, c)
    # 台帳の生存行 = 初期ストック + 境界流入(帳尻項が実際にその差を説明している)
    live = int(prov["conservation"][A.VEHICLE]["live"])
    assert live == int(prov["stock0"].get(A.VEHICLE, 0)) \
        + int(prov["rot_in"][A.VEHICLE])


def test_hei1_smoke_analyze_script_stays_green_on_the_rotation_run(rot_run):
    """★同じランで scripts/analyze_assets.py も独立に PASS する。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import analyze_assets as AA

    sim, _summary = rot_run
    ledger = AA.load_ledger(sim.out_dir)
    assert ledger is not None and ledger["rot_in"]
    res = AA.summarize(ledger, AA.load_transfers(sim.out_dir),
                       AA.load_last_cash(sim.out_dir), last_step=199)
    for cat, c in res["conservation"].items():
        assert c["residual"] == 0, (cat, c)
    assert res["conservation_agrees"] is True
    assert "**PASS**" in AA.render(res)


def test_hei2_smoke_roster_covers_the_late_entrants(rot_run):
    """★中間日入場者が roster.parquet に出る(agents.json/traits.json には居ない人)。"""
    import pyarrow.parquet as pq

    sim, _summary = rot_run
    path = sim.out_dir / "roster.parquet"
    assert path.exists(), "roster.parquet が書かれていない"
    rows = pq.read_table(path).to_pylist()
    assert rows, "1 行も書かれていない"
    ids = [int(r["agent_id"]) for r in rows]
    assert len(ids) == len(set(ids)), "同じ個体が 2 行ある"
    late = [r for r in rows if int(r["day"]) > 0]
    assert late, "中間日入場者が 1 行も出ていない(スモークとして成立していない)"
    day0_json = {int(r["id"]) for r in
                 json.loads((sim.out_dir / "agents.json").read_text(encoding="utf-8"))}
    traits_json = {int(k) for k in
                   json.loads((sim.out_dir / "traits.json").read_text(encoding="utf-8"))}
    for r in late:
        aid = int(r["agent_id"])
        assert aid not in day0_json and aid not in traits_json, \
            "day0 名簿に居る個体を「中間日入場」と数えている"
        assert json.loads(r["traits_json"]), "traits が空(名簿として使えない)"


def test_hei3_smoke_summary_carries_the_coverage_section(rot_run):
    """★summary に coverage 節が出る。累計(builds)と最終日の保持率が別物であること。"""
    sim, summary = rot_run
    cov = summary.get("workplace_coverage")
    assert cov is not None, "summary に workplace_coverage が無い(配線が外れている)"
    assert cov == W.provenance(sim), "summary 側で観測量を作り直している"
    assert cov["n_present"] == len(sim.agents)
    assert 0.0 <= cov["org_id_rate"] <= 1.0
    assert 0.0 <= cov["work_node_rate"] <= 1.0
    assert cov["n_org_id"] == sum(1 for a in sim.agents if getattr(a, "org_id", None))
    assert cov["n_work_node"] == sum(1 for a in sim.agents if W.has_workplace(a))
    # 累計は「実体化を通った回数」なので在場者数以上になる(= 痩せを見えなくしていた量)
    assert cov["builds"]["org_seen"] >= cov["n_present"]


def test_hei2_smoke_roster_is_the_only_new_output(rot_run):
    """A13 が足したファイルは roster.parquet 1 枚だけ(part は finalize で畳まれている)。"""
    sim, _summary = rot_run
    assert not list(sim.out_dir.glob("roster.part-*.parquet"))


def _roster_dot(small_pool, name, n_steps, every):
    return ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
            f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144",
            "pool.enabled=true", f"pool.dir={small_pool}", "pool.present_cap=300",
            "observer.roster_daily.enabled=true",
            f"observer.checkpoint_every={every}"]


def test_hei2_split_execution_matches_the_straight_run(small_pool, tmp_path):
    """★resume==straight: 分割実行(80 + resume)の roster が一気通しと完全一致する。"""
    import pyarrow.parquet as pq

    straight = tmp_path / "rs_straight"
    Simulation(load_config(_roster_dot(small_pool, "rs_straight", 200, 0)),
               out_dir=straight).run()
    split = tmp_path / "rs_split"
    Simulation(load_config(_roster_dot(small_pool, "rs_split", 80, 80)),
               out_dir=split).run()                    # 前チャンク(clean finalize)
    Simulation(load_config(_roster_dot(small_pool, "rs_split", 200, 80)),
               out_dir=split).run(resume_from=split)   # 後チャンク(途中再開)

    a = sorted(pq.read_table(straight / "roster.parquet").to_pylist(),
               key=lambda r: (int(r["day"]), int(r["agent_id"])))
    b = sorted(pq.read_table(split / "roster.parquet").to_pylist(),
               key=lambda r: (int(r["day"]), int(r["agent_id"])))
    assert [int(r["agent_id"]) for r in a] == [int(r["agent_id"]) for r in b], \
        "分割実行で名簿の顔ぶれが変わった(_seen の読み戻しが効いていない)"
    assert a == b, "分割実行で名簿の中身(入場日・素性)が変わった"
