"""設備 = 摩耗する装置(昇降設備の DEVS。``world.facilities``)のテスト。

正典
  - docs/plans/body-incident-layer-plan.md §3「設備」行(wear clock + 月1保守 → 閉じ込め →
    インターホン → 保守会社エージェント(30-80分) → 復旧。**停電・漏水は skip**)
  - src/society/devices.py(DEVS 契約: 状態を変えてよいのは δ_int と δ_ext だけ・
    **装置は乱数を 1 本も引かない決定論的な応答関数**)

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・名簿を組まない・state も属性も生えない・
     stream "incident_facility" を 1 本も引かない
  ② 名簿は**地図の純関数**(階数と用途)で、id は devices.py の閉じた接頭辞名簿に載る
  ③ DEVS 契約: δ_int(保守・待機摩耗・復旧)と δ_ext(利用・故障)だけが状態を動かす。
     **δ の外の代入は監査証跡に illegal として残る**(契約が機械検査できる)
  ④ 摩耗は**エージェントの利用から内生する**(既存 L1 の floor_move / enter_building)
  ⑤ 故障 → 閉じ込め → **インターホン通報(行為)** → 対応者の出動(行為) → 復旧
  ⑥ エスカレーターは**閉じ込めない**(構造上の事実)
  ⑦ 較正: EV 閉じ込め 1 万件/年 ÷ 78 万台 = 3.5e-5 件/台/日 が既定で立っている
  ⑧ ON 同 seed 2 ラン一致 / resume == straight(**wear が checkpoint に載る**)
  ⑨ ★静的検査: generate() を呼ばない・stream は 1 本だけ・**装置クラスは乱数を持たない**
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from society import devices as D
from society import facility_devices as FD
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS, Event

MODULE = Path(FD.__file__)
REPO = Path(__file__).resolve().parents[1]

NEW_KINDS = ("facility_fault", "facility_call", "facility_dispatch",
             "facility_restore", "facility_maintenance")

OFF = {"world.facilities.enabled": "false"}
ON = {"world.facilities.enabled": "true"}


def _cfg(name, n_steps=48, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=48, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _place(sim, agent, node, *, building=""):
    agent.node = str(node)
    agent.x, agent.y = sim.city.node_xy(node)
    agent.loc = "street"
    agent.building = str(building)
    agent.sleeping = False
    agent.route = []
    return agent


def _exile(sim, keep=()):
    ids = {int(a.id) for a in keep}
    for a in sim.agents:
        if int(a.id) not in ids:
            a.loc, a.building, a.route = "outside", "", []


def _lone_device(sim, facility=FD.ELEVATOR):
    """名簿から 1 台選ぶ(device_id 昇順の先頭 = 決定論)。"""
    for dev in FD.registry_of(sim):
        if dev.facility == facility:
            return dev
    raise AssertionError(f"{facility} が名簿に 1 台も居ない")


# =========================================================================== #
# (A) 出荷既定・宣言・較正アンカー(検収基準 ①の前段・⑦)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.world.facilities.enabled) is False
    assert bool(cfg.world.facilities.elevator.enabled) is True
    assert bool(cfg.world.facilities.escalator.enabled) is True
    # ★閉じ込めはエレベーターだけ(エスカレーターは構造上起きない)
    assert bool(cfg.world.facilities.elevator.traps) is True
    assert bool(cfg.world.facilities.escalator.traps) is False


def test_calibration_anchor_is_the_national_entrapment_rate():
    """★EV 閉じ込め 全国およそ 1 万件/年 ÷ 全国設置台数 ≒ 78 万台 = 0.0128 件/台/年。"""
    cfg = FD.build_cfg(load_config().world.facilities)
    per_day = cfg[FD.ELEVATOR]["fault_per_day"]
    assert abs(per_day - 3.5e-5) < 1e-9
    per_year = per_day * 365.0
    assert abs(per_year - 10_000.0 / 780_000.0) < 2.0e-4, per_year
    # 復旧は 30〜80 分(計画 §3)
    assert cfg[FD.ELEVATOR]["repair_min_lo"] == 30.0
    assert cfg[FD.ELEVATOR]["repair_min_hi"] == 80.0
    assert cfg["maintenance_days"] == 30           # 月 1 回の保守


def test_registry_declares_every_toggle():
    feats = {f.id: f for f in R.FEATURES}
    parent = feats["world.facilities.enabled"]
    assert parent.repro_tier == "strict"
    assert parent.affects_k is False
    assert parent.fingerprint_risk == "possible"
    assert parent.off_value is False
    assert feats["world.facilities.elevator.enabled"].off_value is True
    assert feats["world.facilities.elevator.traps"].off_value is True
    assert feats["world.facilities.escalator.traps"].off_value is False


def test_every_new_kind_is_registered_and_classified():
    for kind in NEW_KINDS:
        assert kind in EVENT_KINDS, kind
        assert C.cause_of(kind) in C.CAUSE_TYPES, kind
    assert C.cause_of("facility_call") == C.AGENT        # 通報は**行為**
    assert C.cause_of("facility_dispatch") == C.AGENT    # 出動も**行為**
    assert C.cause_of("facility_fault") == C.DEVICE      # 故障は装置の出来事
    assert C.cause_of("facility_maintenance") == C.SCHEDULE   # 暦だけで決まる制度
    # 装置 id を刻める語であること(名簿に載る id を出す種だから)
    for kind in ("facility_fault", "facility_restore", "facility_maintenance"):
        assert C.cause_of(kind) in C.DEVICE_STAMPABLE


def test_device_ids_are_in_the_closed_catalogue():
    """★装置 id の名簿は 1 箇所(devices.py)。新しい接頭辞もそこに登録されている。"""
    assert D.LIFT_PREFIX == "lift"
    assert D.LIFT_PREFIX in D.DYNAMIC_DEVICE_PREFIXES
    did = FD.facility_device_id("b123", FD.ELEVATOR)
    assert did == "lift:b123-ev"
    assert D.device_id_is_known(did)
    assert D.device_id_is_known(FD.facility_device_id("b123", FD.ESCALATOR))


# =========================================================================== #
# (B) OFF = 完全 no-op(検収基準 ①)
# =========================================================================== #
def test_off_is_a_pure_noop(tmp_path):
    plain = _sim(tmp_path, "fd_plain", n_steps=24)
    plain.run()
    off = _sim(tmp_path, "fd_off", n_steps=24, **OFF)
    off.run()
    assert _l1(plain) == _l1(off)
    for kind in NEW_KINDS:
        assert _kind(off, kind) == [], kind
    assert getattr(off, "facilities", None) is None      # 名簿を組まない
    assert getattr(off, "_facility_state", None) is None
    assert FD.provenance(off) is None
    assert FD.state_of(off) is None                      # checkpoint も無風
    for agent in off.agents:
        assert not hasattr(agent, "facility_call_until")


def test_off_never_touches_the_new_stream(tmp_path, monkeypatch):
    sim = _sim(tmp_path, "fd_stream", n_steps=24, **OFF)
    seen: list[str] = []
    original = sim.hub.stream

    def spy(*key):
        seen.append(str(key[0]))
        return original(*key)

    monkeypatch.setattr(sim.hub, "stream", spy)
    sim.run()
    assert "incident_facility" not in seen


# =========================================================================== #
# (C) 名簿 = 地図の純関数(検収基準 ②)
# =========================================================================== #
def test_registry_is_a_pure_function_of_the_map(tmp_path):
    sim = _sim(tmp_path, "fd_reg", n_steps=1, **ON)
    reg = FD.registry_of(sim)
    cfg = FD.cfg_of(sim)
    want_ev = {str(b["id"]) for b in sim.city.buildings
               if int(b.get("levels") or 1) >= cfg[FD.ELEVATOR]["min_levels"]
               and b.get("entrance")}
    want_es = {str(b["id"]) for b in sim.city.buildings
               if int(b.get("levels") or 1) >= cfg[FD.ESCALATOR]["min_levels"]
               and str(b.get("kind") or "") in FD.ESCALATOR_BUILDING_KINDS
               and b.get("entrance")}
    got_ev = {d.building for d in reg if d.facility == FD.ELEVATOR}
    got_es = {d.building for d in reg if d.facility == FD.ESCALATOR}
    assert got_ev == want_ev and got_es == want_es
    assert want_ev and want_es, "テストが空振り(設備が 1 台も立たない地図)"
    assert list(reg.ids()) == sorted(reg.ids())            # 反復は id 昇順(決定論)
    assert all(D.device_id_is_known(i) for i in reg.ids())


def test_maintenance_days_are_spread_across_devices(tmp_path):
    """保守日は台ごとにばらす(全台が同じ日に一斉点検する退化を避ける)。"""
    sim = _sim(tmp_path, "fd_spread", n_steps=1, **ON)
    days = {d.next_maint_min // 1440 for d in FD.registry_of(sim)}
    assert len(days) > 5, days


# =========================================================================== #
# (D) DEVS 契約(検収基準 ③)
# =========================================================================== #
def _device(**over):
    cfg = FD.build_cfg({"enabled": True, **over})
    return FD.LiftDevice("lift:btest-ev", FD.ELEVATOR, "btest", "n1",
                         cfg=cfg, sub=cfg[FD.ELEVATOR], start_min=0), cfg


def test_delta_ext_use_wears_the_device():
    dev, _cfg_ = _device()
    assert dev.wear == 0.0 and dev.uses == 0
    out = dev.on_input(7, {"kind": "use"}, 0)
    assert out["ok"] is True
    assert dev.wear == 1.0 and dev.uses == 1
    assert dev.audit()["ext"] == 1 and dev.audit()["illegal"] == 0


def test_delta_int_maintenance_resets_wear():
    dev, _cfg_ = _device()
    for _ in range(5):
        dev.on_input(1, {"kind": "use"}, 0)
    assert dev.wear == 5.0
    out = dev.on_schedule(dev.next_maint_min)
    assert "maintained" in out and out["maintained"]["uses"] == 5
    assert dev.wear == 0.0 and dev.uses == 0
    assert dev.last_maint_min == dev.next_maint_min - dev.maint_min


def test_delta_int_restores_after_the_repair_time():
    dev, _cfg_ = _device()
    assert 30.0 <= dev.repair_min <= 80.0
    out = dev.on_input(-1, {"kind": "fault"}, 1000)     # ★世界からの外部入力
    assert out["faulted"] is True and out["traps"] is True
    assert dev.down is True
    assert dev.on_schedule(1000 + int(dev.repair_min) - 1) is False   # まだ止まっている
    got = dev.on_schedule(1000 + int(round(dev.repair_min)))
    assert "restored" in got and dev.down is False
    assert got["restored"]["down_min"] == int(round(dev.repair_min))


def test_use_is_refused_while_down():
    dev, _cfg_ = _device()
    dev.on_input(-1, {"kind": "fault"}, 0)
    out = dev.on_input(3, {"kind": "use"}, 0)
    assert out["ok"] is False and out["down"] is True
    assert dev.uses == 0                              # 止まっている設備は摩耗しない


def test_assignment_outside_delta_is_recorded_as_illegal():
    """★契約が**機械検査できる**: δ の窓の外の代入は監査証跡に残る。"""
    dev, _cfg_ = _device()
    dev.wear = 99.0                                   # 自発的遷移(契約違反)
    assert dev.audit()["illegal"] == 1


def test_hazard_grows_with_wear_and_is_zero_while_down():
    dev, cfg = _device()
    base = dev.hazard_per_step(144)
    assert base > 0.0
    for _ in range(int(cfg["wear_full"])):
        dev.on_input(1, {"kind": "use"}, 0)
    worn = dev.hazard_per_step(144)
    assert abs(worn - base * (1.0 + cfg["wear_gain"])) < 1e-15
    dev.on_input(-1, {"kind": "fault"}, 0)
    assert dev.hazard_per_step(144) == 0.0


def test_a_full_on_run_keeps_the_devs_contract(tmp_path):
    """ラン全体でも自発的遷移が 1 件も起きない(illegal == 0)。"""
    sim = _sim(tmp_path, "fd_run", n_steps=48, **ON)
    sim.run()
    audit = FD.audit_report(sim)
    assert audit, "テストが空振り(名簿が空)"
    assert sum(v["illegal"] for v in audit.values()) == 0
    assert sum(v["int"] for v in audit.values()) > 0     # δ_int は実際に走っている


# =========================================================================== #
# (E) 摩耗はエージェントの利用から内生する(検収基準 ④)
# =========================================================================== #
def test_wear_comes_from_agent_floor_moves(tmp_path):
    sim = _sim(tmp_path, "fd_use", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ELEVATOR)
    before = dev.wear
    sim.logger.log(Event(step=0, sim_min=0, agent_id=int(sim.agents[0].id),
                         kind="floor_move", x=0.0, y=0.0,
                         payload={"building": dev.building, "floor": 9}))
    FD.phase(sim, 0, 0)
    assert dev.uses == 1
    assert dev.wear > before
    assert sim._facility_state["uses"] == 1


def test_low_floors_ride_the_escalator_when_one_exists(tmp_path):
    """低層はエスカレーター・高層はエレベーター(**明示の仮定**を固定する)。"""
    sim = _sim(tmp_path, "fd_pick", n_steps=1, **ON)
    index = FD._by_building(sim)
    both = next(b for b, devs in sorted(index.items())
                if FD.ELEVATOR in devs and FD.ESCALATOR in devs)
    devs = index[both]
    assert FD._device_for_floor(devs, 1, 4) is None          # 1 階 = 昇降しない
    assert FD._device_for_floor(devs, 3, 4) is devs[FD.ESCALATOR]
    assert FD._device_for_floor(devs, 9, 4) is devs[FD.ELEVATOR]


# =========================================================================== #
# (F) 故障 → 閉じ込め → 通報 → 出動 → 復旧(検収基準 ⑤⑥)
# =========================================================================== #
def _fault(sim, dev, step=0, sim_min=0):
    cfg = FD.cfg_of(sim)
    st = FD._state(sim)
    bud = FD._Budget(sim, st, int(cfg["max_events_per_step"]))
    FD._fault_one(sim, cfg, st, bud, dev, step, sim_min)
    return st


def test_entrapment_chain_calls_and_dispatches(tmp_path):
    sim = _sim(tmp_path, "fd_chain", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ELEVATOR)
    trapped, guard = sim.agents[0], sim.agents[1]
    _exile(sim, keep=(trapped, guard))
    _place(sim, trapped, dev.node, building=dev.building)
    _place(sim, guard, dev.node)
    guard.occupation = "警備員"
    st = _fault(sim, dev)
    faults = _kind(sim, "facility_fault")
    assert len(faults) == 1
    payload = faults[0].payload
    assert payload["device_id"] == dev.device_id
    assert payload["trapped"] == 1
    # ★前兆状態(摩耗・利用回数・前回保守からの日数)が同梱される = 内生性の機械検証
    assert set(("wear", "uses", "days_since_maint")) <= set(payload)
    calls = _kind(sim, "facility_call")
    assert len(calls) == 1 and calls[0].agent_id == trapped.id
    assert calls[0].payload["self_call"] is True
    dispatch = _kind(sim, "facility_dispatch")
    assert len(dispatch) == 1 and dispatch[0].agent_id == guard.id
    assert dispatch[0].payload["unstaffed"] is False
    assert guard.work_node == dev.node
    assert st["trapped"] == 1 and st["calls"] == 1 and st["dispatches"] == 1
    # 復旧は δ_int(修理時間の経過)で起きる
    FD.phase(sim, 1, int(round(dev.repair_min)) + 1)
    assert len(_kind(sim, "facility_restore")) == 1
    assert dev.down is False


def test_escalator_never_traps(tmp_path):
    """★エスカレーターは構造上**閉じ込めない**(止まるだけ)。"""
    sim = _sim(tmp_path, "fd_es", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ESCALATOR)
    rider = sim.agents[0]
    _exile(sim, keep=(rider,))
    _place(sim, rider, dev.node, building=dev.building)
    st = _fault(sim, dev)
    assert _kind(sim, "facility_fault")[0].payload["trapped"] == 0
    assert st["trapped"] == 0
    # 居合わせた人は「止まった」ことを知らせる(閉じ込めではない通報)
    calls = _kind(sim, "facility_call")
    assert len(calls) == 1 and calls[0].payload["self_call"] is False


def test_no_responder_is_an_honest_unstaffed_marker(tmp_path):
    """対応者が名簿に居なければ**黙って落とさず** unstaffed=true を残す。"""
    sim = _sim(tmp_path, "fd_unstaffed", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ELEVATOR)
    trapped = sim.agents[0]
    _exile(sim, keep=(trapped,))
    _place(sim, trapped, dev.node, building=dev.building)
    trapped.occupation = "無職"
    st = _fault(sim, dev)
    dispatch = _kind(sim, "facility_dispatch")
    assert len(dispatch) == 1
    assert dispatch[0].agent_id == -1
    assert dispatch[0].payload["unstaffed"] is True
    assert st["unstaffed"] == 1


def test_empty_building_is_never_reported(tmp_path):
    """誰も居なければ通報も対応も起きない(行為が無ければ応答も無い)。"""
    sim = _sim(tmp_path, "fd_empty", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ELEVATOR)
    _exile(sim)
    _fault(sim, dev)
    assert len(_kind(sim, "facility_fault")) == 1
    assert _kind(sim, "facility_call") == []
    assert _kind(sim, "facility_dispatch") == []


def test_a_second_fault_does_not_double_count(tmp_path):
    sim = _sim(tmp_path, "fd_twice", n_steps=1, **ON)
    dev = _lone_device(sim, FD.ELEVATOR)
    _exile(sim)
    _fault(sim, dev)
    _fault(sim, dev)
    assert len(_kind(sim, "facility_fault")) == 1


# =========================================================================== #
# (G) 決定論・resume(検収基準 ⑧)
# =========================================================================== #
FAST = {**ON, "world.facilities.elevator.fault_per_day": "2.0",
        "world.facilities.escalator.fault_per_day": "2.0"}


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "fd_det_a", n_steps=48, n_agents=20, **FAST)
    a.run()
    b = _sim(tmp_path, "fd_det_b", n_steps=48, n_agents=20, **FAST)
    b.run()
    assert _l1(a) == _l1(b)
    assert _kind(a, "facility_fault"), "テストが空振り(故障 0 件)"
    assert FD.state_of(a)["devices"] == FD.state_of(b)["devices"]


def test_resume_matches_straight(tmp_path):
    """★wear / 故障中 / 次の保守が checkpoint に載る(resume で摩耗が 0 に戻らない)。"""
    ov = {**FAST, "run.start_tod": "00:00"}
    split, total = 36, 72
    straight_dir = tmp_path / "fd_straight"
    straight = Simulation(_cfg("fd_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "facility_fault"), "テスト前提が崩れた(故障ゼロ)"

    d = tmp_path / "fd_resumed"
    sim1 = Simulation(_cfg("fd_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("fd_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(facility resume)"
    assert FD.state_of(straight)["devices"] == FD.state_of(sim2)["devices"]
    assert straight._facility_state["faults"] == sim2._facility_state["faults"]
    assert max(v["wear"] for v in FD.state_of(sim2)["devices"].values()) > 0.0


def test_provenance_reports_measurement_against_the_anchor(tmp_path):
    sim = _sim(tmp_path, "fd_prov", n_steps=48, n_agents=20, **FAST)
    sim.run()
    prov = FD.provenance(sim)
    assert prov["elevators"] > 0 and prov["escalators"] > 0
    assert prov["faults"] == len(_kind(sim, "facility_fault"))
    assert prov["fault_reference_per_day"] > 0.0
    assert set(prov["faults_by_kind"]) <= set(FD.FACILITY_KINDS)


# =========================================================================== #
# (H) 静的検査(検収基準 ⑨)
# =========================================================================== #
def test_module_calls_no_llm_and_only_one_new_stream():
    text = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "generate" not in (attrs | names)
    assert "llm" not in attrs
    streams = set(re.findall(r'hub\.stream\(\s*"([a-z_]+)"', text))
    assert streams == {"incident_facility"}, streams


def test_the_device_class_itself_draws_no_randomness():
    """★devices.py の設計契約: **装置は決定論的な応答関数**(乱数を 1 本も引かない)。

    抽選は module の ``phase`` 側(named stream)で行い、結果だけを δ_ext の
    外部入力として渡す = 装置クラスの中に乱数の識別子が 1 つも無い。
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "LiftDevice")
    attrs = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(cls) if isinstance(n, ast.Name)}
    idents = attrs | names
    for bad in ("rng", "stream", "hub", "random", "poisson", "uniform"):
        assert bad not in idents, f"装置クラスが乱数に触れている: {bad}"


def test_power_and_water_outage_are_deliberately_absent():
    """★停電・漏水は**作らない**(SAIDI 年 13 分 = 起きないが正解)。理由が明記されている。"""
    text = MODULE.read_text(encoding="utf-8")
    assert "SAIDI" in text and "停電" in text and "漏水" in text
    tree = ast.parse(text)
    idents = ({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
              | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
              | {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
    for bad in ("outage", "blackout", "power_cut", "leak", "water_leak"):
        assert bad not in idents, f"停電/漏水の実装が入っている: {bad}"


def test_frozen_metric_spec_files_are_untouched():
    from society.observer import metrics_spec as MS
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for word in ("facility_devices", "facility_fault", "world.facilities"):
            assert word not in text, f"凍結ファイル {rel} に H5 の痕跡がある"
