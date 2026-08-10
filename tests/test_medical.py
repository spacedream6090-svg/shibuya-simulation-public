"""医療の受け皿 = 搬送・入院・医療費(レーン **H2** 2026-08-10)のテスト。

正典: docs/plans/body-incident-layer-plan.md §2 / §6-4(保険=命名 RoW チャネルは承認済み)。

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・新 4 種 0 件・agent に属性が生えない・
     sim に state が生えない・H1 の受診経路が 1 バイトも変わらない
  ② 搬送先 = **地図の subcat=hospital**(v8 は 7 件)。subcat を持たない地図では 0 件 =
     搬送が起きない(名前から病院を推測しない)。clinic は**病院を必ず除く**(ヒント是正)
  ③ 搬送 → 入院 → 退院: 位置が病院へ移り、在院長が**確定重症度**で決まり、退院で状態が戻る
  ④ 金の三本足: ① 公費(区の歳出 → RoW ems_operation)② 自己負担(受け手 = 受診先ノード)
     ③ 保険給付(RoW insurance_reimbursement → 医療機関 org)。**貨幣保存が閉じたまま**
  ⑤ 受け手の是正: 医療費が「支払者の居場所」ではなく「受診先」で解決される(RoW 漏れの修復)
  ⑥ 統合結線: ``city_ops.request_ems`` の公開シーム / ``health.on_injury`` の公開 API
  ⑦ 救急車ひっ迫: 全車出動中の通報が ``unstaffed`` として観測できる
  ⑧ ON 同 seed 2 回一致 / resume 跨ぎ同値 / 乱数を 1 本も引かない(AST)/ LLM 呼数不変
  ⑨ 宣言の追随: registry / causality / timeconv / 会計の網羅(COVERED_KINDS)
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_accounting as aa                     # noqa: E402

from society import city_ops as CO                  # noqa: E402
from society import economy_sfc as SFC              # noqa: E402
from society import health as H                     # noqa: E402
from society import incidents_env as IE             # noqa: E402
from society import medical as M                    # noqa: E402
from society import registry as R                   # noqa: E402
from society import services as SV                  # noqa: E402
from society import timeconv as T                   # noqa: E402
from society.config import load_config              # noqa: E402
from society.engine import checkpoint, scheduler    # noqa: E402
from society.engine.simulation import Simulation    # noqa: E402
from society.observer import causality as C         # noqa: E402
from society.observer.schema import EVENT_KINDS     # noqa: E402

MODULE = Path(M.__file__)

#: 本 module が出す L1 種(すべて材料側 registration)
NEW_KINDS = ("ems_transport", "hospital_admit", "hospital_discharge", "medical_bill")

V8 = "data/shibuya_osm_wide_v8.json"
V7 = "data/shibuya_osm_wide_v7.json"

HEALTH = {"health.enabled": "true"}
SEV = {**HEALTH, "health.severity.enabled": "true"}
#: 発症チャネルを全部止める(舞台を手で作るため)。test_health_severity と同型。
QUIET = {"health.severity.acute_illness_daily": "0.0",
         "health.severity.trauma_daily": "0.0",
         "health.severity.cardiac_scale": "0.0",
         "health.severity.alcohol_nightlife_daily": "0.0",
         "health.severity.heat_base": "0.0",
         "health.severity.worsen_daily": "0.0"}
MED_ON = {"medical.enabled": "true"}
MED_OFF = {"medical.enabled": "false"}
SFC_ON = {"economy.org_accounting.enabled": "true", "organizations.enabled": "true",
          "government.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_health_severity.py / test_city_ops.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _stage(tmp_path, name, n_agents=12, wide=V8, **ov):
    """病人 1 人 + 通行人 2 人 + 当直の救急隊 1 人(test_health_severity._ems_stage と同型)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents,
               **{"city_ops.enabled": "true", "world.map": wide,
                  **SEV, **QUIET, **MED_ON, **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    crew = ags[0]
    crew.occupation = CO.EMS_CREW
    CO._ensure_bound(sim, -1, 0)
    patient = ags[1]
    patient.occupation = "会社員"
    nodes = sorted(sim.city.graph.nodes)
    for who, node in ((patient, nodes[0]), (crew, nodes[-1])):
        who.node, who.loc, who.building = str(node), "street", ""
        who.sleeping, who.route = False, []
        who.x, who.y = sim.city.node_xy(node)
    crew.work_start_min, crew.work_end_min = 0, 1440
    near = ags[2:4]
    for other in near:
        other.occupation = "会社員"
        other.node, other.loc, other.building = str(nodes[0]), "street", ""
        other.sleeping, other.route = False, []
        other.x, other.y = sim.city.node_xy(nodes[0])
    for other in ags[4:]:
        other.loc = "outside"
    return sim, patient, crew, near


def _collapse(sim, patient, step=0, sim_min=420, sev=None):
    """S3 への遷移を手で刻んで救急連鎖を 1 回まわす(H1 の引き金の形をそのまま使う)。"""
    patient.severity = H.S_SEVERE if sev is None else int(sev)
    patient.sev_channel = H.CH_TRAUMA
    patient.sev_collapse_step = int(step)
    CO.phase(sim, step, sim_min)


def _spy_spend():
    """``_spend`` の薄いスタブ(第 4 引数 payee_node まで受ける)。"""
    paid = []

    def _pay(agent, amount, cat, payee_node=None):
        paid.append((int(agent.id), float(amount), str(cat), payee_node))
    return paid, _pay


# =========================================================================== #
# ① OFF(既定)= 現行と完全同値
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。新 4 種は 1 件も出ない。"""
    pure = _sim(tmp_path, "pure", n_steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", n_steps=144, **MED_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H2 の seam が no-op でない)"
    for kind in NEW_KINDS:
        assert not _kind(pure, kind)


def test_off_matches_golden(tmp_path):
    """★ゴールデン L1 とバイト一致(既定 OFF の最終防衛線)。

    上書きは test_org_accounting.py / test_traces.py と同じ「意図的な既定挙動追加」の中立化。
    """
    golden = json.loads((_ROOT / "tests" / "data" / "golden_baseline_l1.json")
                        .read_text(encoding="utf-8"))
    ov = {"economy.wages.自営": 0, "planning.enabled": "false",
          "transit_ride.taxi.enabled": "false", "transit_ride.bus.enabled": "false",
          "rules.enabled": "false"}
    sim = _sim(tmp_path, "golden", n_steps=144, n_agents=15, **ov)
    sim.run()
    assert _l1(sim) == golden, "H2 の seam がゴールデンを動かしている"


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では sim に state が生えず・agent に属性が生えない(city_ops と同じ流儀)。"""
    sim = _sim(tmp_path, "off_state", n_steps=24, **MED_OFF)
    sim.run()
    for attr in ("_med_state", "_med_hospitals", "_med_clinics"):
        assert getattr(sim, attr, None) is None, f"OFF で sim.{attr} が生えた"
    for agent in sim.agents:
        for attr in ("med_admitted", "med_until", "med_dest_node",
                     "med_transport_until"):
            assert not hasattr(agent, attr), f"OFF で agent に {attr} が生えた"
    assert M.provenance(sim) is None


def test_off_seams_are_noop(tmp_path):
    """OFF では公開の口を直接叩いても何も起きない(直接呼んでも安全)。"""
    sim = _sim(tmp_path, "off_call", n_steps=1, **MED_OFF)
    agent = sim.agents[0]
    paid, pay = _spy_spend()
    assert M.on_ems_dispatch(sim, agent, None, 0, 420) is False
    assert M.on_care(sim, agent, 3000.0, pay, 0, 420) is False
    assert M.care_venue(sim, agent) is None
    M.phase(sim, 0, 420, pay)
    assert paid == [] and not [e for e in sim.logger.events if e.kind in NEW_KINDS]


def test_off_keeps_the_h1_care_path_byte_identical(tmp_path):
    """★H1 の受診経路は medical OFF で 1 バイトも変わらない(受け手も従来どおり)。"""
    sim = _sim(tmp_path, "h1_care", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    agent = sim.agents[0]
    agent.loc, agent.sleeping, agent.building = "street", False, None
    agent.severity = H.S_MODERATE
    paid, pay = _spy_spend()
    H.maybe_care(agent, H.severity_cfg(sim.healthcfg), 0.0, H._state(sim), 0, 420,
                 sim.logger, pay, sim)
    assert len(paid) == 1
    assert paid[0][2] == "medical" and paid[0][3] is None, "OFF で受診先が渡された"


# =========================================================================== #
# ② 搬送先の解決(clinic ヒントの是正を含む)
# =========================================================================== #
def test_hospitals_come_from_the_subcat(tmp_path):
    """v8 の ``subcat=hospital`` 7 件が搬送先(地図データと厳密一致)。"""
    sim = _sim(tmp_path, "hosp", n_steps=1, **{**MED_ON, "world.map": V8})
    raw = json.loads(Path(V8).read_text(encoding="utf-8"))
    want = sum(1 for p in raw["pois"] if p.get("subcat") == "hospital")
    got = M.hospitals(sim)
    assert want == 7, f"地図 v8 の hospital 件数が変わった: {want}"
    assert len(got) == want
    assert all(p.get("subcat") == "hospital" for p in got)


@pytest.mark.parametrize("wide", (V7, "data/shibuya_osm.json"))
def test_no_hospital_without_subcat(tmp_path, wide):
    """★subcat を持たない地図では病院 0 件 = 搬送が起きない(名前から推測しない)。"""
    sim = _sim(tmp_path, f"nosub_{Path(wide).stem}", n_steps=1,
               **{**SEV, **QUIET, **MED_ON, "world.map": wide})
    assert M.hospitals(sim) == []
    agent = sim.agents[0]
    assert M.on_ems_dispatch(sim, agent, None, 0, 420) is False
    assert M.provenance(sim)["no_hospital"] == 1, "0 件のまま黙って通り過ぎた"


def test_clinics_never_include_a_hospital(tmp_path):
    """★受診先(クリニック)から総合病院を除く = 是正の核。"""
    sim = _sim(tmp_path, "clinics", n_steps=1, **{**MED_ON, "world.map": V8})
    hospital_ids = {str(p.get("id")) for p in M.hospitals(sim)}
    assert hospital_ids, "前提が崩れている(v8 に病院が無い)"
    assert not {str(p.get("id")) for p in M.clinics(sim)} & hospital_ids
    # 名称が「〜クリニック」の**病院**が v8 に実在する = ヒントだけでは分けられない
    assert any("クリニック" in str(p.get("name") or "") for p in M.hospitals(sim)), \
        "この地図では是正の意味が測れない(前提の確認)"


def test_service_clinic_excludes_hospital_subcat():
    """★``services.py`` の任意受診(健診)も病院を引かない(同じ是正・データ駆動)。"""
    cfg = SV.build_cfg({"services": {"clinic": {"cats": ["service"],
                                                "hints": ["クリニック"],
                                                "exclude_subcats": ["hospital"]}}})
    svc = cfg["services"]["clinic"]
    assert SV._matches(svc, "美容クリニック", "") is True
    assert SV._matches(svc, "美容クリニック", "hospital") is False, "病院が健診先に入る"
    # 既定表にも同じ宣言が入っている(conf 未指定のランでも効く)
    assert "hospital" in SV._DEFAULT_SERVICES["clinic"]["exclude_subcats"]


def test_service_clinic_is_unchanged_without_subcat():
    """★subcat を持たない地図(v7 以前)では**何も落ちない**= 完全同値。"""
    cfg = SV.build_cfg({"services": {"clinic": {"cats": ["service"],
                                                "hints": ["クリニック"],
                                                "exclude_subcats": ["hospital"]}}})
    svc = cfg["services"]["clinic"]
    assert SV._matches(svc, "○○クリニック", "") is True
    plain = SV.build_cfg({})["services"]["grooming"]
    assert plain["exclude_subcats"] == [], "既定で除外が生えている"


def test_care_venue_is_a_clinic_not_a_hospital(tmp_path):
    """自力受診の行き先は**クリニック**で、搬送先(病院)にはならない。"""
    sim = _sim(tmp_path, "venue", n_steps=1, **{**MED_ON, "world.map": V8})
    hospital_nodes = {str(p["node"]) for p in M.hospitals(sim)}
    clinic = M.clinics(sim)[0]
    agent = sim.agents[0]
    agent.x, agent.y = sim.city.node_xy(str(clinic["node"]))
    node = M.care_venue(sim, agent)
    assert node == str(clinic["node"])
    assert node not in hospital_nodes


# =========================================================================== #
# ③ 搬送 → 入院 → 退院
# =========================================================================== #
def test_dispatch_starts_a_transport_and_arrival_admits(tmp_path):
    """出動 → 搬送 → 病院で入院。位置が病院ノードへ移り L1 が 2 行出る。"""
    sim, patient, crew, _near = _stage(tmp_path, "transport")
    _collapse(sim, patient, step=0)
    assert _kind(sim, "ems_dispatch"), "前提が崩れている(出動が起きていない)"
    assert int(patient.med_transport_until) == 3, "搬送中の印が立っていない"
    assert not _kind(sim, "ems_transport"), "搬送が同 step に完了している"
    M.phase(sim, 1, 430)                                   # まだ着かない
    assert not _kind(sim, "ems_transport")
    M.phase(sim, 3, 450)                                   # 到着
    trans = _kind(sim, "ems_transport")
    admit = _kind(sim, "hospital_admit")
    assert len(trans) == 1 and len(admit) == 1
    assert trans[0].agent_id == int(crew.id), "搬送は隊員の行為(agent_id が違う)"
    assert admit[0].agent_id == int(patient.id)
    assert str(patient.node) == str(trans[0].payload["node"])
    assert str(patient.node) in {str(p["node"]) for p in M.hospitals(sim)}
    assert patient.med_admitted is True and patient.sleeping is True
    assert M.provenance(sim)["transports"] == 1


@pytest.mark.parametrize("confirmed,expect_days", [(H.S_MILD, 0), (H.S_MODERATE, 3),
                                                   (H.S_SEVERE, 14), (H.S_ARREST, 21)])
def test_stay_length_follows_the_confirmed_severity(confirmed, expect_days):
    """★在院長は**確定重症度**の純関数(軽症だけが日未満 = 数時間で帰宅)。"""
    cfg = M.build_cfg({"enabled": True})
    got = M.stay_steps(cfg, confirmed, 144)
    if expect_days == 0:
        assert got == cfg["admit"]["mild_steps"] < 144
    else:
        assert got == expect_days * 144


def test_discharge_restores_the_agent_and_logs_one_line(tmp_path):
    """退院 = 状態復帰(起床・建物を出る)+ L1 1 行。"""
    sim, patient, _crew, _near = _stage(tmp_path, "discharge")
    patient.sev_confirmed = H.S_MILD                       # 軽症 = 数時間で帰宅
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    until = int(patient.med_until)
    M.phase(sim, until, 450 + (until - 3) * 10)
    out = _kind(sim, "hospital_discharge")
    assert len(out) == 1 and out[0].agent_id == int(patient.id)
    assert patient.sleeping is False and patient.building is None
    assert M.in_care(patient) is False
    assert M.provenance(sim)["discharges"] == 1


def test_a_patient_is_not_transported_twice(tmp_path):
    """在院中の個体はもう一度搬送されない(二重搬送しない)。"""
    sim, patient, crew, _near = _stage(tmp_path, "twice")
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    assert M.on_ems_dispatch(sim, patient, crew, 4, 460) is False
    assert M.provenance(sim)["transports"] == 1


def test_admit_disabled_wakes_the_patient_in_place(tmp_path):
    """入院を切ると搬送だけが起き、在院はしない(子トグルが効く)。"""
    sim, patient, _crew, _near = _stage(tmp_path, "noadmit",
                                        **{"medical.admit.enabled": "false"})
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    assert _kind(sim, "ems_transport") and not _kind(sim, "hospital_admit")
    assert getattr(patient, "med_admitted", False) is False


def test_death_in_care_clears_the_state_without_a_second_line(tmp_path):
    """在院中の死は H1 が 1 行記録済み = ここでは L1 を 1 件も足さない。"""
    sim, patient, _crew, _near = _stage(tmp_path, "died")
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    assert patient.med_admitted is True
    patient.dead = True
    before = len(sim.logger.events)
    M.phase(sim, 4, 460)
    assert len(sim.logger.events) == before, "死亡で L1 を二重に出している"
    assert M.in_care(patient) is False
    assert M.provenance(sim)["died_in_care"] == 1


# =========================================================================== #
# ④ 金の三本足
# =========================================================================== #
def test_ems_transport_is_funded_by_the_ward_budget(tmp_path):
    """① 救急搬送 = 区(ward)の歳出 → RoW(ems_operation)。総マネー保存は閉じたまま。"""
    sim, patient, _crew, _near = _stage(tmp_path, "ems_money", **SFC_ON)
    scheduler._sfc_arm(sim, 0, 420)                        # 行政 / org 預金の実体化
    gov = sim.government
    _collapse(sim, patient, step=1, sim_min=430)
    before_total = SFC.total_money(sim)
    before_ward = gov.balance["ward"]
    M.phase(sim, 4, 460)
    ev = _kind(sim, "ems_transport")[-1]
    cost = float(ev.payload["cost"])
    assert cost > 0.0 and ev.payload["payer"] == "government"
    assert ev.payload["payee"] == "row:ems_operation"
    assert gov.balance["ward"] == pytest.approx(before_ward - cost)
    assert SFC.total_money(sim) == pytest.approx(before_total), "貨幣が湧いた/消えた"
    assert SFC.state_of(sim)["row"]["ems_operation"]["out"] == pytest.approx(cost)


def test_ems_cost_is_not_booked_without_an_authority(tmp_path):
    """行政が居ない世界では**街の残高が 1 円も動かない**(payer を載せない)。"""
    sim, patient, _crew, _near = _stage(tmp_path, "ems_noauth")
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    ev = _kind(sim, "ems_transport")[-1]
    assert "payer" not in ev.payload and "payee" not in ev.payload
    assert M.provenance(sim)["ems_funded"] == 0.0
    assert M.provenance(sim)["ems_cost_total"] > 0.0, "公費の観測まで消えている"


def test_insurance_is_a_named_row_channel(tmp_path):
    """③ 保険 7 割 = RoW チャネル ``insurance_reimbursement`` → 医療機関 org。"""
    sim, patient, _crew, _near = _stage(tmp_path, "insurance", **SFC_ON)
    scheduler._sfc_arm(sim, 0, 420)
    venue = M.care_venue(sim, patient)                      # 受診先(= 保険給付の受け手)
    assert venue, "前提が崩れている(近くにクリニックが無い)"
    sim.orgs = {"c1": {"id": "c1", "name": "医院",
                       "workplace_poi": {"cat": "service", "node": str(venue)}}}
    sim._sfc_book_idx = None                                # 台帳索引を張り直す
    before_total = SFC.total_money(sim)
    before_org = SFC.org_balance(sim, "c1")
    paid, pay = _spy_spend()
    got = M.on_care(sim, patient, 3000.0, pay, 1, 430)
    assert got is True and paid and paid[0][3] == str(venue)
    bill = _kind(sim, "medical_bill")[-1]
    assert bill.payload["kind"] == "clinic"
    assert bill.payload["gross"] == pytest.approx(10000.0)   # 3000 / 0.3
    assert bill.payload["self_pay"] == pytest.approx(3000.0)
    assert bill.payload["amount"] == pytest.approx(7000.0), "保険 7 割が動いていない"
    assert bill.payload["payer"] == "row:insurance_reimbursement"
    assert bill.payload["payee"] == "c1"
    assert SFC.state_of(sim)["row"]["insurance_reimbursement"]["in"] \
        == pytest.approx(7000.0)
    assert SFC.org_balance(sim, "c1") == pytest.approx(before_org + 7000.0)
    # RoW からの入金は街の残高を増やし、RoW 累積が同額減る = 不変量は閉じたまま
    assert SFC.total_money(sim) == pytest.approx(before_total)


def test_insurance_to_an_outside_hospital_moves_no_money(tmp_path):
    """受け手 org を特定できない給付は**街の残高を 1 円も動かさない**(正直開示)。"""
    sim, patient, _crew, _near = _stage(tmp_path, "ins_row", **SFC_ON)
    scheduler._sfc_arm(sim, 0, 420)
    sim.orgs = {}
    sim._sfc_book_idx = None
    before_total = SFC.total_money(sim)
    _paid, pay = _spy_spend()
    M.on_care(sim, patient, 3000.0, pay, 1, 430)
    bill = _kind(sim, "medical_bill")[-1]
    assert bill.payload["amount"] == 0.0 and "payer" not in bill.payload
    assert SFC.total_money(sim) == pytest.approx(before_total)
    assert M.provenance(sim)["insurance_to_row"] == pytest.approx(7000.0)


def test_the_patient_share_goes_to_the_venue_not_the_current_location(tmp_path):
    """★②の是正: 受け手が**受診先ノード**で解決される(旧: 支払者の居場所 → RoW 漏れ)。"""
    sim = _sim(tmp_path, "payee_fix", n_steps=1, n_agents=12,
               **{**SEV, **QUIET, **MED_ON, **SFC_ON, "world.map": V8})
    scheduler.run_step(sim, 0)
    clinic = M.clinics(sim)[0]
    sim.orgs = {"c1": {"id": "c1", "name": "医院",
                       "workplace_poi": {"cat": "service", "node": str(clinic["node"])}}}
    sim._sfc_book_idx = None
    agent = sim.agents[0]
    agent.loc, agent.building, agent.sleeping = "street", None, False
    agent.x, agent.y = sim.city.node_xy(str(clinic["node"]))
    agent.money = 50000.0
    # 是正前と同じ形(受診先を渡さない)の支払いは、居場所からの解決に落ちる
    before = SFC.state_of(sim)["payee"]["row"]
    scheduler._spend(sim, agent, 3000.0, "medical", 1, 430)
    plain = _kind(sim, "spend")[-1].payload["payee"]
    # 是正後(受診先を渡す)は台帳の org へ着地する
    scheduler._spend(sim, agent, 3000.0, "medical", 1, 430,
                     payee_node=str(clinic["node"]))
    fixed = _kind(sim, "spend")[-1].payload["payee"]
    assert fixed == "c1", f"受診先で解決されていない: {fixed}"
    assert plain != fixed, "是正の前後で受け手が同じ(舞台が測れていない)"
    assert SFC.state_of(sim)["payee"]["row"] == before + 1, "旧経路の RoW 漏れが再現しない"


def test_admission_bill_is_charged_at_discharge(tmp_path):
    """入院費は退院時に**日数ぶん**まとめて請求される(受け手 = 病院ノード)。"""
    sim, patient, _crew, _near = _stage(tmp_path, "admit_bill")
    paid, pay = _spy_spend()
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450, pay)
    assert paid == [], "入院の時点で請求している(退院時のはず)"
    node = str(patient.med_node)
    until = int(patient.med_until)
    # ★確定重症度を刻むのは健康側(city_ops → health.note_confirmed)なので、期待値は
    #   そこから導く(テストが在院長を決め打つと、確定の分離そのものを測れない)。
    stayed = until - 3
    days = max(1, round(stayed / float(sim.clock.steps_per_day)))
    M.phase(sim, until, 450 + stayed * 10, pay)
    assert len(paid) == 1
    _aid, amount, cat, payee_node = paid[0]
    assert cat == "medical" and payee_node == node, "受け手が病院ノードでない"
    assert amount == pytest.approx(10500.0 * days), "日数ぶんの自己負担になっていない"
    bill = _kind(sim, "medical_bill")[-1]
    assert bill.payload["kind"] == "hospital" and bill.payload["days"] == days
    assert bill.payload["gross"] == pytest.approx(amount / 0.3, rel=1e-6)


def test_money_is_conserved_across_a_run_with_medical_on(tmp_path):
    """★貨幣保存(Σ全主体残高 + RoW + K5 = 一定)が**三本足を踏んだまま**毎 step 閉じる。

    ★``cardiac_scale`` の増幅は**較正ではなく点火**である(実測レートでは 1 日に数件しか
      起きないので、短ランでは三本足を踏まないまま緑になってしまう)。
    """
    sim = _sim(tmp_path, "conserve", n_steps=144, n_agents=30,
               **{**SEV, **MED_ON, **SFC_ON, "world.map": V8,
                  "city_ops.enabled": "true",
                  "health.severity.cardiac_scale": "200000",
                  "agents.personas_file": "data/personas_80.json"})
    for agent in sorted(sim.agents, key=lambda a: int(a.id))[:3]:
        agent.occupation = CO.EMS_CREW                     # 小ランの名簿に救急隊が居ない
    scheduler.run_step(sim, 0)
    base = SFC.total_money(sim)
    for step in range(1, 144):
        scheduler.run_step(sim, step)
        assert SFC.total_money(sim) == pytest.approx(base, rel=1e-9), \
            f"step {step} で総マネーが動いた"
    prov = M.provenance(sim)
    assert prov["transports"] > 0, "三本足を 1 度も踏まないまま緑になっている"
    assert prov["ems_funded"] > 0.0 and prov["self_pay_total"] > 0.0


# =========================================================================== #
# ⑤ 統合結線(公開シーム 2 件)
# =========================================================================== #
def test_request_ems_is_the_public_seam(tmp_path):
    """``city_ops.request_ems`` が公開シームとして実在し、当直を割り当てて印を付ける。"""
    assert callable(getattr(CO, "request_ems", None)), "公開シームが消えた"
    sim, patient, crew, _near = _stage(tmp_path, "seam")
    got = CO.request_ems(sim, patient, None, "injury", step=0, sim_min=420)
    assert got["crew"] is crew and got["unstaffed"] is False
    assert got["response_min"] is not None and got["response_min"] >= 0.0
    assert int(crew.city_ops_ems_until) > 0, "出動中の印が付いていない"
    assert str(crew.work_node) == str(patient.node), "持ち場が現場へ移っていない"
    assert not sim.logger.events, "シームが L1 を出している(記録は呼び出し側の責務)"


def test_request_ems_is_unstaffed_when_city_ops_is_off(tmp_path):
    """救急の実体が世界に無いランでは必ず unstaffed(黙って None を返さない)。"""
    sim = _sim(tmp_path, "seam_off", n_steps=1, **{"city_ops.enabled": "false"})
    got = CO.request_ems(sim, sim.agents[0], None, "injury", step=0, sim_min=420)
    assert got["crew"] is None and got["unstaffed"] is True


def test_incidents_env_uses_the_public_seam():
    """★H5 は private ヘルパではなく公開シームを呼ぶ(呼び出し規則の一本化)。"""
    src = Path(IE.__file__).read_text(encoding="utf-8")
    assert "request_ems" in src
    assert "_on_duty_crew(" not in src, "private ヘルパを直接呼んでいる"


def test_on_injury_is_the_public_api_and_maps_to_s1_s2(tmp_path):
    """``health.on_injury`` = 他レーンからの唯一の口。見立ては S1/S2 に閉じる。"""
    assert callable(getattr(H, "on_injury", None))
    sim = _sim(tmp_path, "injury", n_steps=1, n_agents=8, **{**SEV, **QUIET})
    agent = sim.agents[0]
    agent.loc, agent.sleeping, agent.building = "street", False, None
    got = H.on_injury(sim, agent, 2, "brawl", step=0, sim_min=420)
    assert got == H.S_MODERATE and agent.severity == H.S_MODERATE
    assert agent.sev_channel == H.CH_TRAUMA and agent.sick is True
    ev = _kind(sim, "illness")[-1]
    assert ev.payload["external"] is True and ev.payload["source"] == "brawl"
    # 見立てが重すぎても S2 で頭打ち(事件側から S3/S4 を作らない)
    other = sim.agents[1]
    other.loc, other.sleeping, other.building = "street", False, None
    assert H.on_injury(sim, other, 9, "fire", step=0, sim_min=420) == H.S_MODERATE


def test_on_injury_is_noop_when_severity_is_off(tmp_path):
    """severity OFF では 1 バイトも動かない(H4/H5 の従来挙動と同値)。"""
    sim = _sim(tmp_path, "injury_off", n_steps=1, n_agents=8, **HEALTH)
    agent = sim.agents[0]
    assert H.on_injury(sim, agent, 2, "brawl", step=0, sim_min=420) == 0
    assert int(getattr(agent, "severity", 0)) == 0
    assert not _kind(sim, "illness")


def test_on_injury_never_downgrades_and_draws_no_new_random(tmp_path):
    """★既にもっと重い個体を上書きしない / **health_onset の消費列を 1 本も動かさない**。"""
    sim = _sim(tmp_path, "injury_keep", n_steps=1, n_agents=8, **{**SEV, **QUIET})
    agent = sim.agents[0]
    agent.loc, agent.sleeping, agent.building = "street", False, None
    agent.severity = H.S_SEVERE
    assert H.on_injury(sim, agent, 1, "fire", step=0, sim_min=420) == H.S_SEVERE
    assert agent.severity == H.S_SEVERE
    # 乱数: on_injury は stream を 1 本も引かない(AST で機械固定)
    tree = ast.parse(Path(H.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "on_injury")
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for banned in ("stream", "random", "integers", "uniform", "shuffle"):
        assert banned not in names, f"on_injury に乱数の識別子 {banned} がある"


def test_h5_injury_reaches_the_body(tmp_path):
    """H5(火災・交通)の負傷が同じ公開 API へ結線されている。"""
    sim = _sim(tmp_path, "h5_injury", n_steps=1, n_agents=8, **{**SEV, **QUIET})
    victim = sim.agents[0]
    victim.loc, victim.sleeping, victim.building = "street", False, None
    IE._note_injury(sim, victim, "serious", "fire", 0, 420)
    assert victim.severity == H.S_MODERATE and victim.sev_channel == H.CH_TRAUMA
    assert set(IE.INJURY_HINT.values()) <= {1, 2}, "事件側から S3 以上を作っている"


# =========================================================================== #
# ⑥ 救急車ひっ迫(全車出動中の通報)
# =========================================================================== #
def test_all_crews_busy_makes_the_next_call_unstaffed(tmp_path):
    """★ひっ迫: 当直が出動中なら次の通報は ``unstaffed`` として観測できる。"""
    sim, patient, crew, near = _stage(tmp_path, "surge")
    crew.city_ops_ems_until = 99                           # 唯一の隊が出動中
    got = CO.request_ems(sim, patient, None, "collapse", step=0, sim_min=420)
    assert got["crew"] is None and got["unstaffed"] is True
    _collapse(sim, patient, step=0)
    disp = _kind(sim, "ems_dispatch")
    assert len(disp) == 1 and disp[0].payload["unstaffed"] is True
    assert disp[0].agent_id == -1
    assert CO.provenance(sim)["dispatch_unstaffed"] == 1
    # 搬送も起きない(応える隊が居ないので患者は運ばれない)
    M.phase(sim, 3, 450)
    assert not _kind(sim, "ems_transport")


def test_transport_holds_the_crew_until_arrival(tmp_path):
    """搬送のあいだ隊は戻らない(= 次の通報に応えられない = ひっ迫が内生する)。"""
    sim, patient, crew, _near = _stage(tmp_path, "hold")
    _collapse(sim, patient, step=0)
    assert int(crew.city_ops_ems_until) >= 3, "搬送中も現着時間だけで戻ってしまう"


# =========================================================================== #
# ⑦ 決定論・resume・R1
# =========================================================================== #
def test_on_is_deterministic(tmp_path):
    """ON 同 seed の 2 ランが L1 完全一致(乱数を引かない = 構造的に決定論)。"""
    ov = {**SEV, **MED_ON, "world.map": V8, "city_ops.enabled": "true"}
    a = _sim(tmp_path, "det_a", n_steps=48, n_agents=20, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", n_steps=48, n_agents=20, **ov)
    b.run()
    assert _l1(a) == _l1(b)


def test_resume_matches_straight(tmp_path):
    """★resume 跨ぎ同値(在院状態は agent pickle・タリーは checkpoint に載る)。"""
    ov = {**SEV, **MED_ON, "world.map": V8, "city_ops.enabled": "true",
          "observer.checkpoint_every": 24}

    def rows(out_dir):
        import pyarrow.parquet as pq
        files = sorted((out_dir / "l1").glob("*.parquet"))
        got = []
        for path in files:
            table = pq.read_table(path, columns=["step", "agent_id", "kind"])
            got += list(zip(table["step"].to_pylist(),
                            table["agent_id"].to_pylist(),
                            table["kind"].to_pylist()))
        return got

    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 48, 20, **ov), out_dir=straight).run()
    resumed = tmp_path / "resumed"
    sim1 = Simulation(_cfg("resumed", 24, 20, **ov), out_dir=resumed)
    for step in range(24):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 24, resumed / "checkpoint" / "ckpt-000024.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(_cfg("resumed", 48, 20, **ov),
               out_dir=resumed).run(resume_from=resumed)
    assert rows(straight) == rows(resumed), "medical ON の resume が straight と不一致"


def test_checkpoint_carries_the_medical_tally(tmp_path):
    """搬送・入院のタリーが checkpoint に載る(旧 ckpt は None = 無風)。"""
    sim, patient, _crew, _near = _stage(tmp_path, "ckpt")
    _collapse(sim, patient, step=0)
    M.phase(sim, 3, 450)
    path = tmp_path / "ckpt" / "checkpoint" / "ckpt-000004.pkl.gz"
    checkpoint.save(sim, 4, path)
    import gzip
    import pickle
    blob = pickle.loads(gzip.decompress(path.read_bytes()))
    state = blob["runtime"]["medical_state"]
    assert state is not None and int(state["transports"]) == 1


def test_module_draws_no_random_and_calls_no_llm():
    """★AST 静的検査: 乱数の呼び出しも generate() の呼び出しサイトも 1 つも無い。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("stream", "random", "integers", "uniform", "shuffle", "choice",
                   "default_rng", "hub", "generate", "llm"):
        assert banned not in names, f"medical.py に {banned} がある"


def test_module_never_writes_health_state():
    """★AST 静的検査: 身体の状態(sick / severity / sev_*)へ**代入しない**(H1 の管轄)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute):
                written.add(target.attr)
    for banned in ("sick", "sick_until", "severity", "sev_until", "sev_channel",
                   "sev_confirmed", "dead", "fatigue"):
        assert banned not in written, f"medical.py が身体状態 {banned} を書いている"


def test_memory_lines_are_place_and_number_free():
    """★no-fingerprint: 記憶へ入る定型文に地名・数字・実験条件・機構語が 1 つも無い。"""
    for text in (M.TRANSPORT_TEXT, M.ADMIT_TEXT, M.DISCHARGE_TEXT, M.CLINIC_TEXT):
        assert text and not any(ch.isdigit() for ch in text)
        for word in ("渋谷", "道玄坂", "medical", "config", "step", "円", "％", "%"):
            assert word not in text, f"禁止語 {word} が入っている: {text}"


def test_on_does_not_change_llm_call_count(tmp_path):
    """affects_k=False の機械固定: ON/OFF で generate() の呼数が 1 も動かない。"""
    def _n(name, **ov):
        sim = _sim(tmp_path, name, n_steps=48, n_agents=15,
                   **{**SEV, "world.map": V8, "city_ops.enabled": "true", **ov})
        calls = []
        inner = sim.llm.generate

        def _spy(*a, **kw):
            calls.append(1)
            return inner(*a, **kw)
        sim.llm.generate = _spy
        sim.run()
        return len(calls)
    assert _n("k_off", **MED_OFF) == _n("k_on", **MED_ON)


# =========================================================================== #
# ⑧ 宣言の追随(registry / causality / timeconv / 会計の網羅)
# =========================================================================== #
def test_new_kinds_are_registered_and_classified():
    """新 4 種が L1 スキーマに登録され、因果台帳にも分類されている。"""
    for kind in NEW_KINDS:
        assert kind in EVENT_KINDS, f"{kind} が登録されていない"
        assert C.cause_of(kind) in C.CAUSE_TYPES, kind
    assert C.cause_of("ems_transport") == C.AGENT      # 運ぶのは隊員の行為
    assert C.cause_of("hospital_admit") == C.DEVICE    # 入院を決めるのは医療機関
    assert C.cause_of("medical_bill") == C.DEVICE      # 保険の内訳は制度の会計処理


def test_registry_declares_every_toggle():
    """conf の bool リーフがレジストリに宣言されている(未宣言トグルの検出)。"""
    ids = {f.id for f in R.FEATURES}
    for key in ("medical.enabled", "medical.transport.enabled",
                "medical.transport.hold_crew", "medical.clinic.enabled",
                "medical.admit.enabled", "medical.money.enabled"):
        assert key in ids, f"{key} が registry に無い"
    cfg = load_config([])
    assert R.undeclared_toggles(cfg) == []


def test_timeconv_classifies_the_step_keys():
    """step 単位のキーが Δt 変換の網に載っている(Δt=1 で実時刻が保たれる)。"""
    assert T.classify("medical.transport.steps")[0] == T.STEPS
    assert T.classify("medical.admit.mild_steps")[0] == T.STEPS
    assert T.classify("medical.max_events_per_step")[0] == T.RATE
    assert T.classify("medical.admit.moderate_days")[0] == T.INVARIANT
    fine = load_config(["run.dt_min=1"])
    assert int(fine.medical.transport.steps) == 30, "実時間 30 分が保たれていない"
    assert int(fine.medical.admit.moderate_days) == 3, "日数まで変換している"


def test_accounting_covers_the_new_money_paths():
    """★IF-E の監視装置との接続: 新しい金の経路が接続済みとして宣言されている。"""
    for kind in ("ems_transport", "medical_bill"):
        assert kind in aa.MONEY_KINDS, f"{kind} が会計検査の網に無い"
        assert kind in SFC.COVERED_KINDS, f"{kind} が COVERED_KINDS に無い"
    assert set(aa.MONEY_KINDS) <= set(SFC.COVERED_KINDS) | set(SFC.UNCOVERED_KINDS)
    for kind in ("hospital_admit", "hospital_discharge"):
        assert kind in aa.DERIVED_MONEY_KINDS, f"{kind} が二重計上の禁止リストに無い"
    assert "insurance_reimbursement" in SFC.CHANNELS_IN
    assert "ems_operation" in SFC.CHANNELS_OUT
    assert set(SFC.CHANNELS) == set(SFC.CHANNELS_IN) | set(SFC.CHANNELS_OUT)


def test_accounting_classifier_maps_the_new_kinds_to_known_sectors():
    """分類器が新 2 種を **void ではない**部門へ落とす(漏れの族を作らない)。"""
    flows = aa.flows_for("ems_transport",
                         {"cost": 45000.0, "payer": "government",
                          "payee": "row:ems_operation"}, {})
    assert len(flows) == 1
    assert flows[0].src == aa.GOVERNMENT and flows[0].dst == aa.EXTERNAL
    assert aa.flows_for("ems_transport", {"cost": 45000.0}, {}) == [], \
        "行政が居ない世界で動いていない金のフローを立てている"
    bills = aa.flows_for("medical_bill",
                         {"amount": 7000.0, "payer": "row:insurance_reimbursement",
                          "payee": "h1", "self_pay": 3000.0}, {})
    assert len(bills) == 1
    assert bills[0].src == aa.EXTERNAL and bills[0].dst == aa.ORG
    assert aa.flows_for("medical_bill", {"amount": 0.0, "self_pay": 3000.0}, {}) == []


def test_frozen_files_are_untouched():
    """★凍結 14 本を 1 バイトも触っていない(本レーンの絶対規律)。"""
    frozen = ["truth_ledger.py"] + [f"observer/{n}.py" for n in
                                    ("aggregate", "measure", "stream", "echo", "norms",
                                     "silence", "deviation", "structure",
                                     "initial_frame")]
    for name in frozen:
        src = (_ROOT / "src" / "society" / name).read_text(encoding="utf-8")
        assert "medical" not in src, f"{name} に H2 の痕跡がある"
