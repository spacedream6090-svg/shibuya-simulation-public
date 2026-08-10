"""身体の階層化 = **重症度の状態機械**(レーン H1 2026-08-10)のテスト。

正典: docs/plans/body-incident-layer-plan.md §1 / §6-1(死の表現はユーザー承認済み)。

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・``death`` 0 件・agent の身体欄が中立値のまま・
     ``health.enabled=true`` でも severity OFF なら**単一真偽値の現行経路**がそのまま通る
  ② frailty = 年齢帯の通院者率由来の純関数(seed と id だけで決まる・resume 不変・clip)
  ③ 発症チャネル 5 種のハザード形(WBGT 閾値関数・年齢帯 OHCA・20 代ピークの飲酒・高齢の転倒)
  ④ 互換写像: S2 以上 → sick=True / S1 は presenteeism の裏返し / S0 → False。
     既存の消費者(in_work_window / _sick_home)を 1 行も書き換えずに世代交代できている
  ⑤ 実現条件: 飲酒していない個体は急性アルコールにならない(落ちた件は dropped に出る)
  ⑥ **救急の引き金の世代交代**: city_ops の暫定ハッシュゲートではなく「S3/S4 への遷移そのもの」
  ⑦ 傍観者モデルの較正: 集団 P が 0.85〜0.95 帯・危険事態は責任分散なし・関係者で上がる・
     心停止で誰も居合わせなければ通報が起きない(目撃されなかった倒れ)
  ⑧ 重症度は搬送先で確定(見かけと確定の分離。実測 52.8% が軽症)
  ⑨ 死 = 境界 despawn と同型の永続退場 + L1 1 行 + causality=physics。量は較正で担保
  ⑩ プール回転の往復(**監査で見つかったバグの同梱修正**: sick/sick_until が載っていなかった)
  ⑪ resume 跨ぎ同値 / ON 同 seed 2 回一致 / LLM 呼数を 1 本も足さない(AST)
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from society import city_ops as CO
from society import health as H
from society import registry as R
from society.agents.agent import Agent
from society.agents.memory import MemoryStore
from society.cognition import drive, routine
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS
from society.world import pool as pool_mod

MODULE = Path(H.__file__)

HEALTH = {"health.enabled": "true"}
SEV = {**HEALTH, "health.severity.enabled": "true"}
#: チャネルを 1 本ずつ切り分けるための「他は全部止める」上書き。
QUIET = {"health.severity.acute_illness_daily": "0.0",
         "health.severity.trauma_daily": "0.0",
         "health.severity.cardiac_scale": "0.0",
         "health.severity.alcohol_nightlife_daily": "0.0",
         "health.severity.heat_base": "0.0",
         "health.severity.worsen_daily": "0.0"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=48, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=48, n_agents=20, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _scfg(sim=None):
    return H.severity_cfg(sim.healthcfg) if sim is not None else dict(H.SEVERITY_DEFAULTS)


def _agent(aid=1, age=30, **kw):
    a = Agent(id=aid, name="甲", age=age, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class _FixedLLM:
    """**プロンプト非依存**の応答スタブ(呼数だけ数える)。"""

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


# =========================================================================== #
# ① OFF(既定)= 現行と完全同値
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。``death`` は 1 件も出ない。"""
    pure = _sim(tmp_path, "pure", n_steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", n_steps=144,
               **{"health.severity.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(severity seam が no-op でない)"
    assert not _kind(pure, "death")


def test_off_leaves_the_body_fields_neutral(tmp_path):
    """OFF では身体の欄が 1 つも動かない(agent の中立値のまま)。"""
    sim = _sim(tmp_path, "off_fields", n_steps=24)
    sim.run()
    for a in sim.agents:
        assert a.severity == 0 and a.sev_channel == "" and a.sev_until == -1
        assert a.sev_pending is None and a.dead is False
        assert a.sev_collapse_step == -1 and a.sev_confirmed == -1


def test_severity_off_keeps_the_legacy_boolean_path(tmp_path):
    """health ON + severity OFF は**単一真偽値の現行経路**をそのまま通る(payload 契約不変)。"""
    sim = _sim(tmp_path, "legacy", n_steps=1, n_agents=15,
               **{**HEALTH, "health.onset_prob": "1.0", "health.medical_prob": "1.0"})
    sim.run()
    onset = [e for e in _kind(sim, "illness") if e.payload["state"] == "onset"]
    assert onset, "対照が空回り(現行経路で発症していない)"
    for e in onset:
        assert set(e.payload) == {"state", "days"}, \
            f"severity OFF なのに payload が増えている: {e.payload}"
    assert not _kind(sim, "death")


def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.health.severity.enabled) is False
    assert bool(cfg.health.enabled) is False


def test_severity_needs_the_parent_health_toggle(tmp_path):
    """★正直な依存関係: 親(health.enabled)が OFF なら severity だけ ON にしても動かない。"""
    sim = _sim(tmp_path, "orphan", n_steps=1,
               **{"health.severity.enabled": "true"})
    assert H.severity_on(sim) is False


def test_registry_declares_the_toggle():
    ids = {f.id for f in R.FEATURES}
    assert "health.severity.enabled" in ids
    f = next(x for x in R.FEATURES if x.id == "health.severity.enabled")
    assert f.repro_tier == "strict" and f.affects_k is False


def test_death_kind_is_registered_and_classified_as_physics():
    """新種 ``death`` は材料側 registration + 因果台帳に載っている(cause_type=physics)。"""
    assert "death" in EVENT_KINDS, "death が schema に未登録"
    assert C.CAUSE_OF_KIND["death"] == C.PHYSICS
    assert "death" not in C.DEVICE_STAMPABLE, "死に装置 id を与えてはいけない"


# =========================================================================== #
# ② frailty(年齢帯の通院者率 → 全ハザードの乗数)
# =========================================================================== #
def test_frailty_rises_with_age():
    """通院者率の年齢勾配がそのまま体力係数の勾配になる(65+ が 20 代の 3 倍以上)。"""
    s = _scfg()
    young = H.frailty_of(_agent(1, age=25), s, seed=7)
    old = H.frailty_of(_agent(1, age=70), s, seed=7)
    assert old > young * 3.0, f"年齢勾配が弱すぎる: {young} → {old}"


def test_frailty_is_a_pure_function_of_seed_and_id():
    """**新しい乱数を 1 本も引かない**: (seed, id, age) だけで値が決まり、何度呼んでも同じ。"""
    s = _scfg()
    a = _agent(11, age=40)
    b = _agent(11, age=40)
    assert H.frailty_of(a, s, 7) == H.frailty_of(b, s, 7) == H.frailty_of(a, s, 7)
    assert H.frailty_of(_agent(12, age=40), s, 7) != H.frailty_of(a, s, 7)
    assert H.frailty_of(a, s, 8) != H.frailty_of(a, s, 7), "seed が効いていない"


def test_frailty_is_clipped():
    s = dict(_scfg(), frailty_min=0.9, frailty_max=1.1)
    for age in (5, 30, 90):
        got = H.frailty_of(_agent(3, age=age), s, 7)
        assert 0.9 <= got <= 1.1


def test_age_band_labels_cover_every_age():
    for age, want in ((10, "u20"), (25, "20s"), (35, "30s"), (45, "40s"),
                      (60, "50-64"), (70, "65-74"), (85, "75+")):
        assert H.age_band(age) == want


# =========================================================================== #
# ③ 発症チャネル(ハザードの形)
# =========================================================================== #
def test_heat_hazard_is_zero_below_the_floor_and_steep_above_32():
    """λ(WBGT): 閾値未満で 0・欠測(天気 OFF)でも 0・32 以上で急峻に立ち上がる。"""
    s = _scfg()
    a = _agent(1, age=30, loc="street", building=None)
    assert H.heat_hazard(a, s, None) == 0.0, "WBGT 欠測を 0 と偽らず発生もさせない"
    assert H.heat_hazard(a, s, 20.0) == 0.0
    at28 = H.heat_hazard(a, s, 28.0)
    at32 = H.heat_hazard(a, s, 32.0)
    at35 = H.heat_hazard(a, s, 35.0)
    assert 0.0 < at28 < at32 < at35
    assert at32 / at28 > 4.0, f"32 度で急峻になっていない: {at28} → {at32}"


def test_heat_hazard_amplifies_outdoors_and_for_the_elderly():
    s = _scfg()
    indoor = _agent(1, age=30, loc="street", building="b1")
    outdoor = _agent(1, age=30, loc="street", building=None)
    elderly = _agent(1, age=75, loc="street", building=None)
    assert H.heat_hazard(outdoor, s, 33.0) > H.heat_hazard(indoor, s, 33.0)
    assert H.heat_hazard(elderly, s, 33.0) > H.heat_hazard(outdoor, s, 33.0)


def test_cardiac_hazard_uses_age_bands_not_a_flat_national_rate():
    """★計画書 §7「年齢構造」: 全国平均の直適用は禁忌 = 年齢帯で 2 桁の差が出る。"""
    s = _scfg()
    young = H.cardiac_hazard(_agent(1, age=25), s)
    old = H.cardiac_hazard(_agent(1, age=80), s)
    assert old / young > 50.0, f"年齢帯レートになっていない: {young} vs {old}"
    # 街頭人口の主体(20-40 代)では極低率であること(1 人日あたり 1e-6 未満)
    assert H.cardiac_hazard(_agent(1, age=30), s) < 1e-6


def test_alcohol_hazard_peaks_in_the_twenties_and_rises_on_weekend_nights():
    s = _scfg()
    twenties = H.alcohol_hazard(_agent(1, age=25), s, weekend=False)
    forties = H.alcohol_hazard(_agent(1, age=45), s, weekend=False)
    teen = H.alcohol_hazard(_agent(1, age=17), s, weekend=False)
    assert twenties > forties > teen
    assert H.alcohol_hazard(_agent(1, age=25), s, weekend=True) > twenties


def test_trauma_hazard_rises_with_age():
    s = _scfg()
    assert H.trauma_hazard(_agent(1, age=80), s) \
        > H.trauma_hazard(_agent(1, age=70), s) \
        > H.trauma_hazard(_agent(1, age=30), s)


def test_initial_severity_distributions_match_the_measured_shares():
    """74% が軽症(急性アルコール)・心停止の割合など、分布の境界が conf どおり。"""
    s = _scfg()
    assert H._draw_initial_severity(s, H.CH_ALCOHOL, 0.70) == H.S_MILD
    assert H._draw_initial_severity(s, H.CH_ALCOHOL, 0.80) == H.S_MODERATE
    assert H._draw_initial_severity(s, H.CH_ALCOHOL, 0.99) == H.S_SEVERE
    assert H._draw_initial_severity(s, H.CH_CARDIAC, 0.10) == H.S_ARREST
    assert H._draw_initial_severity(s, H.CH_CARDIAC, 0.90) == H.S_SEVERE
    assert H._draw_initial_severity(s, H.CH_ILLNESS, 0.10) == H.S_MILD


# =========================================================================== #
# ④ 互換写像(既存の消費者を 1 行も書き換えない世代交代)
# =========================================================================== #
def test_compat_mapping_to_the_legacy_boolean():
    s = _scfg()
    a = _agent(1)
    for sev, presentee, want in ((H.S_HEALTHY, False, False),
                                 (H.S_MILD, True, False),     # 出勤する軽症
                                 (H.S_MILD, False, True),     # 休む軽症
                                 (H.S_MODERATE, True, True),  # 中等症は必ず sick
                                 (H.S_SEVERE, True, True),
                                 (H.S_ARREST, True, True)):
        a.severity = sev
        a.sev_until = 999
        H._apply_sick_role(a, s, presentee)
        assert a.sick is want, f"S{sev}/presentee={presentee} の写像が {a.sick}"
        assert (a.sick_until == 999) if want else (a.sick_until == -1)
    assert H.is_sick(a) == a.sick


def test_mild_presentee_keeps_the_work_window_and_is_dulled(tmp_path):
    """S1 で出勤する個体は勤務時間帯扱いのまま(欠勤しない)+ 発火閾値のデバフを受ける。"""
    sim = _sim(tmp_path, "presentee", n_steps=1, **SEV)
    a = sim.agents[0]
    a.work_start_min, a.work_end_min = 8 * 60, 17 * 60
    a.severity, a.sev_presentee = H.S_MILD, True
    H._apply_sick_role(a, H.severity_cfg(sim.healthcfg), True)
    assert routine.in_work_window(a, 10 * 60), "presenteeism なのに欠勤扱い"
    d = H.severity_threshold_delta(a, sim.healthcfg)
    assert d > 0.0, "軽症の性能デバフが効いていない"
    a.drive_threshold = 0.55
    assert drive.effective_threshold(a, d) > drive.effective_threshold(a, 0.0)


def test_moderate_is_absent_and_heads_home(tmp_path):
    """S2 は欠勤 + 既存の ``_sick_home`` プリミティブで自宅へ寄る(新機構を作らない)。"""
    sim = _sim(tmp_path, "moderate", n_steps=1, **SEV)
    a = next(x for x in sim.agents if not x.visitor and x.home_node)
    a.work_start_min, a.work_end_min = 8 * 60, 17 * 60
    a.severity = H.S_MODERATE
    a.sev_until = 10_000
    H._apply_sick_role(a, H.severity_cfg(sim.healthcfg), False)
    assert not routine.in_work_window(a, 10 * 60)
    a.building, a.route, a.sleeping, a.loc = None, [], False, "street"
    a.node = next(d for d in sim.dests if d != a.home_node)
    a.bedtime_min = 1320
    action = routine.decide(a, 48, sim, "路上",
                            sim.hub.stream("decide", a.id, 48), has_company=False)
    assert action["type"] == "move_to" and action["dest"] == a.home_node


def test_moderate_seeks_care_at_onset_and_only_once(tmp_path):
    """S2 は**発症したその場で**受診/停留を決め、1 発症につき受診は 1 回まで。"""
    sim = _sim(tmp_path, "care", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.loc, a.sleeping, a.building = "street", False, None
    _stage_pending(sim, a, H.CH_ILLNESS, sev=H.S_MODERATE)
    a.sev_pending["u_care"] = 0.0                          # 必ず受診する側
    paid = []
    H.severity_tick(sim, 0, 420, lambda ag, amt, cat: paid.append((ag.id, amt, cat)))
    visits = _kind(sim, "medical_visit")
    assert len(visits) == 1 and visits[0].payload["sev"] == H.S_MODERATE
    assert paid and paid[0][2] == "medical"
    assert a.sev_cared is True
    # ★受診の記録は発症の記録より後(倒れる前に医者にかかった順序にしない)
    order = [e.kind for e in sim.logger.events
             if e.kind in ("illness", "medical_visit")]
    assert order == ["illness", "medical_visit"], order
    H.maybe_care(a, H.severity_cfg(sim.healthcfg), 0.0, H._state(sim), 1, 430,
                 sim.logger, None)
    assert len(_kind(sim, "medical_visit")) == 1, "同じ発症で二度受診している"


def test_moderate_can_choose_to_stay_home(tmp_path):
    """停留(受診しない)を選ぶ枝がある = 決定論の二択であって全員受診ではない。"""
    sim = _sim(tmp_path, "stay", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.loc, a.sleeping = "street", False
    _stage_pending(sim, a, H.CH_ILLNESS, sev=H.S_MODERATE)
    a.sev_pending["u_care"] = 0.999
    H.severity_tick(sim, 0, 420)
    assert not _kind(sim, "medical_visit")
    assert a.severity == H.S_MODERATE and a.sick is True


def test_severity_threshold_delta_is_zero_when_off(tmp_path):
    """severity OFF では軽症でもデバフ 0(恒等 = バイト一致)。"""
    sim = _sim(tmp_path, "nodebuff", n_steps=1, **HEALTH)
    a = sim.agents[0]
    a.severity = H.S_MILD
    assert H.severity_threshold_delta(a, sim.healthcfg) == 0.0


# =========================================================================== #
# ⑤ 実現条件(発症は「起こりうる場所と時刻」でしか実現しない)
# =========================================================================== #
def _stage_pending(sim, agent, channel, sev=H.S_MILD, slot_min=0, day=0):
    agent.sev_pending = {"day": day, "slot_min": slot_min, "ch": channel,
                         "sev": sev, "days": 1, "u_fatal": 0.99,
                         "u_presentee": 0.99, "u_care": 0.99, "wbgt": None}


def test_alcohol_only_fires_where_people_drink(tmp_path):
    """飲酒していない個体は急性アルコールにならない(落ちた件は dropped に出る)。"""
    sim = _sim(tmp_path, "alc", n_steps=1, n_agents=12, **SEV)
    a = sim.agents[0]
    a.loc, a.sleeping = "street", False
    a.node = sorted(sim.city.graph.nodes)[0]              # 夜間店舗ではないノード
    _stage_pending(sim, a, H.CH_ALCOHOL)
    H.severity_tick(sim, 0, 420)
    assert a.severity == H.S_HEALTHY
    assert H.provenance(sim)["dropped"][H.CH_ALCOHOL] == 1
    nightlife = sim.city.pois_by_cat("nightlife")
    if not nightlife:
        pytest.skip("この地図には nightlife POI が無い")
    a.node = str(nightlife[0]["node"])
    _stage_pending(sim, a, H.CH_ALCOHOL)
    H.severity_tick(sim, 1, 430)
    assert a.severity == H.S_MILD and a.sev_channel == H.CH_ALCOHOL


def test_a_pending_onset_waits_for_its_time_of_day(tmp_path):
    """予約された発症はその時刻[分 of day]に達するまで起きない。"""
    sim = _sim(tmp_path, "slot", n_steps=1, n_agents=12, **SEV)
    a = sim.agents[0]
    a.loc, a.sleeping = "street", False
    _stage_pending(sim, a, H.CH_ILLNESS, slot_min=900)
    H.severity_tick(sim, 0, 600)                           # 10:00 = まだ
    assert a.severity == H.S_HEALTHY and a.sev_pending is not None
    H.severity_tick(sim, 1, 910)                           # 15:10 = 発症
    assert a.severity == H.S_MILD


def test_onset_payload_carries_the_antecedent_state(tmp_path):
    """★L1 は 1 行 + **前兆状態**(WBGT・飲酒・年齢帯・frailty…)を同梱する。"""
    sim = _sim(tmp_path, "payload", n_steps=1, n_agents=12, **SEV)
    a = sim.agents[0]
    a.loc, a.sleeping, a.building = "street", False, None
    a.sev_frailty = 1.5
    _stage_pending(sim, a, H.CH_HEAT, sev=H.S_MODERATE)
    a.sev_pending["wbgt"] = 33.4
    H.severity_tick(sim, 0, 700)
    ev = _kind(sim, "illness")
    assert len(ev) == 1
    got = ev[0].payload
    for key in ("state", "kind", "sev", "age_band", "frailty", "days", "wbgt",
                "drinking", "outdoor", "near", "night", "presentee"):
        assert key in got, f"前兆状態 {key} が payload に無い"
    assert got["state"] == "onset" and got["kind"] == H.CH_HEAT
    assert got["wbgt"] == 33.4 and got["frailty"] == 1.5


def test_weather_layer_is_the_only_source_of_wbgt(tmp_path):
    """``sim.today_weather["wbgt"]`` を読む(天気 OFF では None = 熱中症は起きない)。"""
    sim = _sim(tmp_path, "wbgt", n_steps=1, **SEV)
    assert H.today_wbgt(sim) is None
    sim.today_weather = {"wbgt": 31.7}
    assert H.today_wbgt(sim) == 31.7


# =========================================================================== #
# ⑥ 救急の引き金の世代交代
# =========================================================================== #
def test_collapse_source_fires_only_on_the_transition_step():
    """S3/S4 への**遷移そのもの**が引き金 = 同じ発症で二度倒れない。"""
    a = _agent(1)
    assert H.collapse_source(a, 5) == ""
    a.sev_collapse_step, a.sev_channel = 5, H.CH_CARDIAC
    assert H.collapse_source(a, 5) == H.CH_CARDIAC
    assert H.collapse_source(a, 6) == "", "遷移した step 以外でも倒れている"


def _ems_stage(tmp_path, name, n_agents=12, crowd=2, **ov):
    """病人 1 人 + 近くの通行人 + 当直の救急隊 1 人を立たせた舞台(test_city_ops と同型)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents,
               **{"city_ops.enabled": "true", **SEV, **QUIET, **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    crew = ags[0]
    crew.occupation = CO.EMS_CREW
    CO._ensure_bound(sim, -1, 0)
    patient = ags[1]
    patient.occupation = "会社員"
    nodes = sorted(sim.city.graph.nodes)
    for a, node in ((patient, nodes[0]), (crew, nodes[-1])):
        a.node, a.loc, a.building, a.sleeping, a.route = str(node), "street", "", False, []
        a.x, a.y = sim.city.node_xy(node)
    crew.work_start_min, crew.work_end_min = 0, 1440
    near = ags[2:2 + crowd]
    for other in near:
        other.occupation = "会社員"
        other.node, other.loc, other.building = str(nodes[0]), "street", ""
        other.sleeping, other.route = False, []
        other.x, other.y = sim.city.node_xy(nodes[0])
    for other in ags[2 + crowd:]:
        other.loc = "outside"
    return sim, patient, crew, near


def test_the_chain_is_driven_by_the_severity_transition(tmp_path):
    """★倒れる → 通報 → 出動が、**重症度が S3 になったこと**から起きる(ハッシュゲートではない)。"""
    sim, patient, crew, near = _ems_stage(tmp_path, "sev_chain")
    patient.severity, patient.sev_channel = H.S_SEVERE, H.CH_TRAUMA
    patient.sev_collapse_step = 7
    CO.phase(sim, 7, 420 + 7 * 10)
    downs, calls, disp = (_kind(sim, "collapse"), _kind(sim, "ems_call"),
                          _kind(sim, "ems_dispatch"))
    assert len(downs) == 1 and len(calls) == 1 and len(disp) == 1
    assert downs[0].payload["source"] == H.CH_TRAUMA
    assert downs[0].payload["sev"] == H.S_SEVERE
    assert calls[0].agent_id in {int(a.id) for a in near}
    assert disp[0].agent_id == int(crew.id)
    # 同じ発症で次の step にもう一度倒れることはない
    CO.phase(sim, 8, 420 + 8 * 10)
    assert len(_kind(sim, "collapse")) == 1


def test_the_hash_gate_is_not_used_when_severity_is_on(tmp_path):
    """severity ON では ``sick`` だけの個体は倒れない(引き金が本物に替わっている)。"""
    sim, patient, _crew, _near = _ems_stage(tmp_path, "no_hash",
                                            **{"city_ops.ems.collapse_per_10k": "10000"})
    patient.sick = True                                    # 旧ゲートなら当たる状態
    for step in range(int(sim.clock.steps_per_day)):
        CO.phase(sim, step, 420 + step * 10)
    assert not _kind(sim, "collapse"), "暫定ハッシュゲートがまだ生きている"


# =========================================================================== #
# ⑦ 傍観者モデルの較正
# =========================================================================== #
def _group_prob(n, alpha, solo, cap=0.98):
    return min(cap, 1.0 - (1.0 - solo) ** (float(n) ** (1.0 - alpha)))


def test_group_probability_lands_in_the_measured_band():
    """★集団としての P(誰かが動く) が実測帯(0.85〜0.95)に載る(曖昧・軽微な状況)。"""
    e = CO.DEFAULTS["ems"]
    solo = float(e["bystander_solo_prob"])
    alpha = float(e["bystander_alpha_ambiguous"])
    for n in (3, 4, 5):
        got = _group_prob(n, alpha, solo)
        assert 0.85 <= got <= 0.95, f"n={n} で {got:.3f}(実測帯 0.85〜0.95 の外)"


def test_diffusion_of_responsibility_only_applies_to_ambiguous_situations():
    """危険が明白なら人数減衰なし(alpha=0)= 人が増えるほど誰かが動く(メタ分析の知見)。"""
    e = CO.DEFAULTS["ems"]
    solo = float(e["bystander_solo_prob"])
    amb = _group_prob(8, float(e["bystander_alpha_ambiguous"]), solo)
    danger = _group_prob(8, float(e["bystander_alpha_danger"]), solo)
    assert danger > amb, "危険事態で責任分散が消えていない"
    assert float(e["bystander_alpha_danger"]) == 0.0


def test_related_bystander_raises_the_probability(tmp_path):
    """関係者(同居人・恋人・同僚)が居合わせると介入率に倍率がかかる。"""
    sim, patient, _crew, near = _ems_stage(tmp_path, "related", crowd=1)
    patient.severity, patient.sev_channel = H.S_MODERATE, H.CH_ALCOHOL
    patient.work_node = "___none___"
    ecfg = CO.cfg_of(sim)["ems"]
    _c, _n, plain = CO._bystander_caller(sim, ecfg, patient, 0, 3)
    patient.housemates = [int(near[0].id)]
    _c2, _n2, tied = CO._bystander_caller(sim, ecfg, patient, 0, 3)
    assert tied > plain, f"関係者が居ても確率が上がらない: {plain} → {tied}"


def test_an_unwitnessed_arrest_produces_no_call(tmp_path):
    """★心停止で誰も居合わせなければ通報そのものが起きない(自分では呼べない)。

    黙って通り過ぎず ``no_call`` として数える(正直な稼働率の観測)。
    """
    sim, patient, _crew, near = _ems_stage(tmp_path, "unwitnessed")
    for other in near:
        other.loc = "outside"
    patient.severity, patient.sev_channel = H.S_ARREST, H.CH_CARDIAC
    patient.sev_collapse_step = 3
    CO.phase(sim, 3, 450)
    assert len(_kind(sim, "collapse")) == 1
    assert not _kind(sim, "ems_call") and not _kind(sim, "ems_dispatch")
    assert CO.provenance(sim)["no_call"] == 1


def test_a_conscious_patient_calls_for_themselves(tmp_path):
    """心停止でなければ誰も動かなくても本人が呼べる(出動が主体なしにならない)。"""
    sim, patient, _crew, near = _ems_stage(tmp_path, "selfcall")
    for other in near:
        other.loc = "outside"
    patient.severity, patient.sev_channel = H.S_SEVERE, H.CH_TRAUMA
    patient.sev_collapse_step = 3
    CO.phase(sim, 3, 450)
    calls = _kind(sim, "ems_call")
    assert len(calls) == 1 and calls[0].agent_id == int(patient.id)
    assert calls[0].payload["self_call"] is True


def test_call_delay_follows_the_measured_distribution():
    """通報までの所要時間が実測の 3 ビン(46/29/25%)から引かれる(観測量)。"""
    e = CO.build_cfg({"enabled": True})["ems"]
    patient = _agent(1)
    got = {}
    for step in range(3000):
        got[CO._call_delay_min(e, patient, 0, step)] = \
            got.get(CO._call_delay_min(e, patient, 0, step), 0) + 1
    assert set(got) == {e["call_delay_fast_min"], e["call_delay_mid_min"],
                        e["call_delay_slow_min"]}
    fast = got[e["call_delay_fast_min"]] / 3000.0
    assert abs(fast - e["call_delay_fast_share"]) < 0.05, f"1分未満の割合が {fast:.3f}"


def test_bystander_model_is_off_when_severity_is_off(tmp_path):
    """severity OFF では従来の「最近傍が必ず通報する」= ``no_call`` が構造的に 0。"""
    sim = _sim(tmp_path, "byoff", n_steps=1, n_agents=12,
               **{"city_ops.enabled": "true", **HEALTH,
                  "city_ops.ems.collapse_per_10k": "10000"})
    assert CO._severity_on(sim) is False


# =========================================================================== #
# ⑧ 重症度は搬送先で確定する(見かけとの分離)
# =========================================================================== #
def test_confirmed_severity_follows_the_measured_transport_mix():
    """★東京実測: 搬送の 52.8% が軽症 = 「全通報が重病ではない」の再現。"""
    s = _scfg()
    got = {}
    for aid in range(4000):
        a = _agent(aid, age=30)
        a.severity = H.S_SEVERE                            # 見かけは重症でも…
        v = H.confirm_severity(a, s, seed=1, step=aid)
        got[v] = got.get(v, 0) + 1
    mild = got.get(H.S_MILD, 0) / 4000.0
    assert abs(mild - s["confirm_mild"]) < 0.03, f"軽症の割合が {mild:.3f}"
    assert got.get(H.S_MODERATE, 0) > got.get(H.S_SEVERE, 0)


def test_arrest_is_never_downgraded_at_the_hospital():
    """心停止は見誤りようがない = 確定でも S4 のまま。"""
    s = _scfg()
    a = _agent(1)
    a.severity = H.S_ARREST
    for step in range(50):
        assert H.confirm_severity(a, s, 1, step) == H.S_ARREST


def test_dispatch_separates_apparent_from_confirmed(tmp_path):
    """出動記録は**見かけ**と**確定**の両方を運ぶ(通報は不確実性から生まれる)。"""
    sim, patient, _crew, _near = _ems_stage(tmp_path, "confirm")
    patient.severity, patient.sev_channel = H.S_SEVERE, H.CH_TRAUMA
    patient.sev_collapse_step = 4
    CO.phase(sim, 4, 460)
    disp = _kind(sim, "ems_dispatch")
    assert len(disp) == 1
    assert disp[0].payload["apparent"] == H.S_SEVERE
    assert "confirmed" in disp[0].payload
    assert patient.sev_confirmed == disp[0].payload["confirmed"]


# =========================================================================== #
# ⑨ 死(ユーザー決定 §6-1。H1 トグル配下・現実量較正)
# =========================================================================== #
def test_arrest_fatality_matches_the_ohca_survival_calibration():
    """P(死|心停止) = 1 − OHCA 生存退院率(既定 10% → 0.90)。"""
    s = _scfg()
    st = {"collapses": 0}
    a = _agent(1)
    a.severity = H.S_ARREST
    H._mark_collapse(a, s, 0.85, st, 3)                    # 0.85 < 0.90 = 死
    assert a.sev_fatal is True
    H._mark_collapse(a, s, 0.95, st, 3)                    # 0.95 >= 0.90 = 生存
    assert a.sev_fatal is False
    a.severity = H.S_SEVERE                                 # 重症の致死率は残差の較正値
    H._mark_collapse(a, s, 0.004, st, 3)
    assert a.sev_fatal is True
    H._mark_collapse(a, s, 0.10, st, 3)
    assert a.sev_fatal is False


def test_death_is_a_permanent_exit_like_the_boundary_despawn(tmp_path):
    """死 = 境界 despawn と同型の永続退場 + L1 1 行(演出ゼロ)。"""
    sim = _sim(tmp_path, "death", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.loc, a.sleeping, a.building = "street", False, None
    a.severity, a.sev_channel, a.sev_fatal = H.S_ARREST, H.CH_CARDIAC, True
    a.sev_outcome_step = 2
    H.severity_tick(sim, 2, 500)
    ev = _kind(sim, "death")
    assert len(ev) == 1 and ev[0].agent_id == int(a.id)
    assert set(ev[0].payload) == {"cause", "sev", "age_band", "frailty", "day"}
    assert ev[0].payload["cause"] == H.CH_CARDIAC
    assert a.dead is True and H.is_dead(a)
    assert a.loc == "outside" and a.return_at == H.NEVER_RETURN
    assert a.sleeping is False and a.route == [] and a.sick is False
    # 二度と動き出さない(次の tick で何も起きない)
    H.severity_tick(sim, 3, 510)
    assert len(_kind(sim, "death")) == 1


def test_a_dead_persona_reentering_via_the_pool_is_put_back_out(tmp_path):
    """★プール回転で死者のペルソナが再実体化されても街を歩き出さない(冪等な貼り直し)。

    presence は (pid, day) の純関数で死を知らないので、退場の状態を毎 step 貼り直す。
    L1 は 1 件も増えない(死は既に 1 行記録済み)。
    """
    sim = _sim(tmp_path, "revive", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.dead = True
    a.loc, a.sleeping = "street", False             # 再実体化された直後の状態を模す
    before = len(sim.logger.events)
    H.severity_tick(sim, 5, 470)
    assert a.loc == "outside" and a.return_at == H.NEVER_RETURN
    assert len(sim.logger.events) == before, "貼り直しで L1 が増えている"


def test_a_resuscitated_patient_falls_back_to_moderate(tmp_path):
    """蘇生した心停止は中等症へ落ちて療養が続く(死は必然ではない)。"""
    sim = _sim(tmp_path, "rosc", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.loc, a.sleeping = "street", False
    a.severity, a.sev_channel, a.sev_fatal = H.S_ARREST, H.CH_CARDIAC, False
    a.sev_outcome_step, a.sev_until = 2, 5000
    H.severity_tick(sim, 2, 500)
    assert a.dead is False and a.severity == H.S_MODERATE and a.sick is True


def test_default_rates_keep_deaths_rare(tmp_path):
    """★災害映画化の防止: **既定レートのまま**なら小規模ランで死は 0 件が正しい。"""
    sim = _sim(tmp_path, "rare", n_steps=144, n_agents=40, **SEV)
    sim.run()
    assert not _kind(sim, "death"), "既定レートで死が出ている(較正が壊れている)"
    prov = H.provenance(sim)
    assert prov["deaths"] == 0 and prov["days"] >= 1
    assert prov["ohca_reference_per_day"] < 0.01, "参照値の割り戻しが壊れている"


def test_recovery_returns_to_healthy(tmp_path):
    sim = _sim(tmp_path, "recover", n_steps=1, n_agents=12, **{**SEV, **QUIET})
    a = sim.agents[0]
    a.loc, a.sleeping = "street", False
    a.severity, a.sev_channel, a.sev_until = H.S_MODERATE, H.CH_ILLNESS, 4
    H._apply_sick_role(a, H.severity_cfg(sim.healthcfg), False)
    H.severity_tick(sim, 4, 460)
    rec = [e for e in _kind(sim, "illness") if e.payload["state"] == "recover"]
    assert len(rec) == 1 and rec[0].payload["from"] == H.S_MODERATE
    assert a.severity == H.S_HEALTHY and a.sick is False and a.sick_until == -1


# =========================================================================== #
# ⑩ プール回転の往復(**監査で見つかったバグの同梱修正**)
# =========================================================================== #
def test_pool_field_list_mirrors_the_health_module():
    """world/ 側の写しが society/health の正典と一致する(層の逆流を避けた写しの固定)。"""
    assert pool_mod._HEALTH_FIELDS == H._SLIM_FIELDS
    assert pool_mod._HEALTH_PENDING == "sev_pending"


def test_dehydrate_of_a_healthy_agent_is_unchanged():
    """健康な個体の退避 dict に新キーが生えない(既定ランのバイト列不変)。"""
    a = _agent(11)
    assert "health" not in pool_mod.dehydrate(a)


def test_pool_rotation_no_longer_forgets_the_illness():
    """★バグ修正: ``sick`` / ``sick_until`` が退避に載っていなかった(回転で病気を忘れた)。"""
    a = _agent(11)
    a.sick, a.sick_until = True, 1234
    a.severity, a.sev_channel, a.sev_until = H.S_MODERATE, H.CH_HEAT, 1234
    a.sev_frailty = 1.75
    a.sev_pending = {"day": 2, "slot_min": 700, "ch": H.CH_TRAUMA, "sev": 1,
                     "days": 3, "u_fatal": 0.5, "u_presentee": 0.1,
                     "u_care": 0.2, "wbgt": 31.0}
    state = pool_mod.dehydrate(a)
    json.dumps(state)                                      # JSON 安全
    b = _agent(12)
    pool_mod.hydrate(b, state)
    assert b.sick is True and b.sick_until == 1234
    assert b.severity == H.S_MODERATE and b.sev_channel == H.CH_HEAT
    assert b.sev_frailty == 1.75 and b.sev_pending["ch"] == H.CH_TRAUMA
    state["health"]["sev_pending"]["sev"] = 99             # 実体を共有しない
    assert b.sev_pending["sev"] == 1


def test_hydrate_tolerates_states_without_the_health_key():
    """旧 退避辞書(新キー無し)からの復元で身体の欄を 1 つも書き換えない(前方互換)。"""
    a = _agent(11)
    old = pool_mod.dehydrate(a)
    b = _agent(12)
    b.sick = True
    pool_mod.hydrate(b, old)
    assert b.sick is True, "キーが無いのに既定値で上書きしている"


def test_dead_agents_survive_the_round_trip():
    a = _agent(11)
    a.dead = True
    state = pool_mod.dehydrate(a)
    b = _agent(12)
    pool_mod.hydrate(b, state)
    assert b.dead is True


# =========================================================================== #
# ⑪ 決定論 / resume / R1
# =========================================================================== #
_LOUD = {**SEV, "health.severity.acute_illness_daily": "0.25",
         "health.severity.cardiac_scale": "5000",
         "health.severity.trauma_daily": "0.02",
         "city_ops.enabled": "true"}


def test_on_is_deterministic_across_two_runs(tmp_path):
    outs = []
    for i in (1, 2):
        sim = _sim(tmp_path, f"det{i}", n_steps=144, n_agents=30, **_LOUD)
        sim.run()
        outs.append(_l1(sim))
    assert outs[0] == outs[1], "ON の決定論が崩れている"
    assert [r for r in outs[0] if r[2] == "illness"], "対照が空回り(発症 0 件)"


def test_resume_matches_the_straight_run(tmp_path):
    """★resume 跨ぎ同値: 身体の欄は agents pickle に自然同梱される。"""
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 48, 24, **_LOUD), out_dir=straight).run()

    d = tmp_path / "resumed"
    every = {"observer.checkpoint_every": 24}
    sim1 = Simulation(_cfg("resumed", 24, 24, **_LOUD, **every), out_dir=d)
    for step in range(24):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 24, d / "checkpoint" / "ckpt-000024.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(_cfg("resumed", 48, 24, **_LOUD, **every),
               out_dir=d).run(resume_from=d)
    assert rows(straight) == rows(d), "severity ON の resume が straight と不一致"


def test_module_makes_no_llm_call():
    """★AST 静的検査: health.py に generate() の呼び出しサイトが 1 つも無い(k 非依存)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "generate" not in names
    assert "llm" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_the_new_streams_are_named_and_new():
    """★新乱数は新 named stream のみ(既存 stream の消費順に割り込まない)。"""
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "stream" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant):
                used.add(str(first.value))
    assert used == {"health_onset"}, f"想定外の stream 名: {sorted(used)}"


def test_severity_tick_draws_no_random_number():
    """★毎 step の状態機械は**乱数を 1 本も引かない**(発火・回復・転帰は決定論)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "severity_tick")
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for banned in ("stream", "random", "integers", "uniform", "shuffle", "hub"):
        assert banned not in names, f"severity_tick に乱数の識別子 {banned} がある"


def test_memory_lines_are_place_and_number_free():
    """★no-fingerprint: 記憶へ入る定型文に地名・数字・実験条件・機構語が 1 つも無い。"""
    texts = list(H.ONSET_TEXT.values()) + [H.RECOVER_TEXT, H.CARE_TEXT, H.WORSEN_TEXT]
    banned = ("渋谷", "道玄坂", "宇田川", "ハチ公", "センター街", "severity",
              "config", "step", "%", "円", "S1", "S2")
    for t in texts:
        assert t and not any(ch.isdigit() for ch in t), f"数字が入っている: {t}"
        for w in banned:
            assert w not in t, f"禁止語 {w} が入っている: {t}"


def test_call_count_is_k_invariant(tmp_path):
    """R1: 呼数が k(writeback)に依存しない(compute_matched 下で厳密に一致)。"""
    calls = []
    for wb in ("free", "off"):
        sim = _sim(tmp_path, f"k_{wb}", n_steps=144, n_agents=30,
                   **{**_LOUD, "controls.mode": "compute_matched", "k.writeback": wb})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        calls.append(sim.llm.calls)
    assert calls[0] == calls[1] > 0, f"severity の呼数が k に依存(R1 違反): {calls}"


def test_provenance_is_none_when_off(tmp_path):
    sim = _sim(tmp_path, "prov_off", n_steps=1, **HEALTH)
    assert H.provenance(sim) is None
