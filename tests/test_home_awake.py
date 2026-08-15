"""在宅覚醒 HOME_AWAKE(β9)のテスト。

正典: docs/plans/beta-implementation-plan.md §2 / docs/plans/reflection-leisure-plan.md §5.2。

検証:
  (1) OFF(既定)= home_activity 0 件・ハザード係数を何に変えてもイベント列がバイト一致
      (= 機構が完全な no-op)+ 2 ラン決定論。
  (2) 帰宅 → 就寝の分離: OFF は gap が全て 0 分(現行構造の固定)/ ON は非ゼロが出る。
  (3) 新 kind `home_activity` の **2 箇所登録**(schema.EVENT_KINDS + causality.CAUSE_OF_KIND)
      = 第115 の「因果台帳未登録で finals ラン即死」の再発防止。
  (4) LLM 呼を 1 本も足さない: 在宅覚醒中は `_phase_drive` の対象外(muted)・
      home_stay は type!="stay" なのでスマホ閲覧を挟まない・モジュールに LLM 呼び出しが無い。
  (5) 就寝ハザードの単調性(概日・疲労・翌日早出で上がり、在宅活動への没入で下がる)。
  (6) 在宅活動ラベル 8 種の重み表(年齢帯 × 就業有無 × 時刻帯)+ 世帯シナジー。
  (7) max_awake_min の安全弁(上限に達したら必ず就寝 = 4 時間窓を出て夜の街へ歩き出さない)。
  (8) レジストリ宣言・未宣言トグル 0。
  (9) ON で resume == straight(L1/L2/L3 完全一致)。
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from society import home_awake as HA
from society import registry as R
from society import timeconv as T
from society.cognition import routine
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer import schema as S


# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, steps=1, n=10, extra=None):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "government.enabled=false"]
    dot += list(extra or [])
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _events(sim):
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


def _rng(sim, agent, step):
    return sim.hub.stream("decide", agent.id, step)


def _home_at_night(sim, a, step, bedtime=None):
    """agent を「自宅前の路上・覚醒・就寝時刻ちょうど」の状態に置く(media の _home_at と同型)。

    bedtime 既定 = その step の時刻 = `bedtime_reached` が真かつハザードの概日項が 0。"""
    bedtime = sim.clock.sim_min(step) % 1440 if bedtime is None else bedtime
    a.visitor = False
    a.loc = "street"
    a.building = None
    a.route = []
    a.sleeping = False
    a.home_node = a.home_node or sim.city.gateways[0]
    a.node = a.home_node
    a.x, a.y = sim.city.node_xy(a.node)
    a.work_start_min = -1
    a.part_time = None
    a.bedtime_min = bedtime
    a.now_step = step
    a.housemates = []
    a._home_awake_since = -1
    a._home_act = ""
    a._home_act_end = -1


def _house(**kw):
    """ハザード/重み表用の最小 agent スタブ(traits を 1 つも持たない = R9)。"""
    class _A:
        pass
    a = _A()
    a.age = kw.pop("age", 40)
    a.occupation = kw.pop("occupation", "会社員")
    a.work_start_min = kw.pop("work_start_min", 540)
    a.bedtime_min = kw.pop("bedtime_min", 1320)
    a.fatigue = kw.pop("fatigue", 0.0)
    a.housemates = kw.pop("housemates", [])
    a._home_act = kw.pop("_home_act", "")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class _NoCal:
    calendarcfg = None


# ---------------------------------------------------- (1) OFF = 完全な no-op
def test_off_is_byte_identical_and_ignores_all_knobs(tmp_path):
    """既定 OFF: home_activity 0 件 + ハザード係数を極端に変えてもイベント列がバイト一致。"""
    a = _sim(tmp_path, "ha_off_a", steps=200)
    a.run()
    b = _sim(tmp_path, "ha_off_b", steps=200,
             extra=["daily.home_awake.enabled=false",
                    "daily.home_awake.hazard.p0=1e-12",
                    "daily.home_awake.lead_min=600",
                    "daily.home_awake.max_awake_min=1"])
    b.run()
    ea, eb = _events(a), _events(b)
    assert ea == eb, "OFF なのにノブでイベント列が動いた(no-op でない)"
    assert not any(k == "home_activity" for (_s, _i, k, _p) in ea), \
        "OFF なのに home_activity が出ている"


def test_off_never_touches_home_awake_state(tmp_path):
    """OFF では _home_awake_since が誰にも生えない = muted が全員 False(active リスト不変)。"""
    sim = _sim(tmp_path, "ha_off_state", steps=200)
    sim.run()
    assert all(int(getattr(a, "_home_awake_since", -1)) < 0 for a in sim.agents)
    assert not any(HA.muted(a, sim) for a in sim.agents)


# ---------------------------------------------- (2) 帰宅 → 就寝の分離(本体)
def test_off_go_to_bed_sleeps_in_the_same_step(tmp_path):
    """現行構造の固定: OFF は go_to_bed が入館と就寝を同じ step で連続実行する(gap=0 分)。"""
    sim = _sim(tmp_path, "ha_gap_off")
    a = sim.agents[0]
    _home_at_night(sim, a, 0)
    act = routine.decide(a, 0, sim, "home", _rng(sim, a, 0), has_company=False)
    assert act["type"] == "go_to_bed" and "awake" not in act
    scheduler._apply(sim, a, act, 0, sim.clock.sim_min(0))
    assert a.sleeping, "OFF で go_to_bed が就寝に落ちていない"
    kinds = [e.kind for e in sim.logger.events if e.agent_id == a.id]
    assert "enter_building" in kinds and "sleep_start" in kinds
    e_in = next(e for e in sim.logger.events if e.kind == "enter_building")
    e_sl = next(e for e in sim.logger.events if e.kind == "sleep_start")
    assert e_sl.step - e_in.step == 0, "OFF なのに帰宅と就寝の step が離れている"


def test_on_enters_home_without_sleeping(tmp_path):
    """ON: go_to_bed{awake:true} で入館するが sleeping にならない(状態分離)。"""
    sim = _sim(tmp_path, "ha_enter",
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])   # 事実上ハザード発火なし
    a = next(x for x in sim.agents
             if x.home_building and sim.city.has_building(x.home_building))
    _home_at_night(sim, a, 0)
    act = routine.decide(a, 0, sim, "home", _rng(sim, a, 0), has_company=False)
    assert act == {"type": "go_to_bed", "awake": True}
    scheduler._apply(sim, a, act, 0, sim.clock.sim_min(0))
    assert not a.sleeping, "ON なのに入館と同時に寝てしまった"
    assert a.building == a.home_building, "自宅建物に入っていない"
    assert [e.kind for e in sim.logger.events if e.agent_id == a.id] == ["enter_building"]
    assert HA.muted(a, sim), "在宅覚醒中なのに muted でない(発火対象から外れていない)"


def test_on_home_activity_sessions_then_sleep(tmp_path):
    """ON: 在宅覚醒の各 step が home_stay を返し home_activity を記録し、やがて就寝する。"""
    sim = _sim(tmp_path, "ha_sess",
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12",
                      "daily.home_awake.max_awake_min=60"])
    a = next(x for x in sim.agents
             if x.home_building and sim.city.has_building(x.home_building))
    _home_at_night(sim, a, 0)
    act = routine.decide(a, 0, sim, "home", _rng(sim, a, 0), has_company=False)
    scheduler._apply(sim, a, act, 0, sim.clock.sim_min(0))
    seen = []
    for s in range(1, 12):
        act = routine.decide(a, s, sim, "home", _rng(sim, a, s), has_company=False)
        seen.append(act["type"])
        scheduler._apply(sim, a, act, s, sim.clock.sim_min(s))
        if a.sleeping:
            break
    assert "home_stay" in seen, "在宅覚醒の step が 1 つも無い"
    assert a.sleeping, "max_awake_min=60 を過ぎても寝ていない(安全弁が効いていない)"
    acts = [e for e in sim.logger.events if e.kind == "home_activity"]
    assert acts, "home_activity が 1 件も出ていない"
    for e in acts:
        assert e.payload["act"] in HA.ACTS
        assert e.payload["steps"] >= 1
        assert e.payload["awake_min"] >= 0
    e_in = next(e for e in sim.logger.events if e.kind == "enter_building")
    e_sl = next(e for e in sim.logger.events if e.kind == "sleep_start")
    assert e_sl.step > e_in.step, "ON なのに帰宅 → 就寝の gap が 0 のまま"


def test_home_stay_is_not_stay(tmp_path):
    """home_stay は type!='stay' = scheduler がスマホ閲覧(SNS/ニュース)を挟まない。

    これが「LLM 呼を 1 本も足さない」の 2 本目の柱(1 本目は muted)。"""
    sim = _sim(tmp_path, "ha_notstay",
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a = next(x for x in sim.agents
             if x.home_building and sim.city.has_building(x.home_building))
    _home_at_night(sim, a, 0)
    scheduler._apply(sim, a, routine.decide(a, 0, sim, "home", _rng(sim, a, 0),
                                            has_company=False),
                     0, sim.clock.sim_min(0))
    nxt = routine.decide(a, 1, sim, "home", _rng(sim, a, 1), has_company=False)
    assert nxt["type"] == "home_stay" and nxt["type"] != "stay"


# ------------------------------------------- (3) 新 kind の 2 箇所登録(必須)
def test_home_activity_kind_registered_in_both_places():
    """新 kind は schema と causality の **2 箇所**に登録する(第115 の即死教訓)。"""
    assert "home_activity" in S.EVENT_KINDS, \
        "home_activity が observer/schema.py に未登録"
    assert "home_activity" in C.CAUSE_OF_KIND, \
        "home_activity が observer/causality.py の CAUSE_OF_KIND に未登録"
    assert C.CAUSE_OF_KIND["home_activity"] in C.CAUSE_TYPES
    assert C.CAUSE_OF_KIND["home_activity"] == C.AGENT


def test_every_registered_kind_has_a_cause_and_vice_versa():
    """登録網羅: schema に在って causality に無い kind(およびその逆)を 1 件も許さない。"""
    import society  # noqa: F401  (材料側の register も読み込む)
    missing = sorted(set(S.EVENT_KINDS) - set(C.CAUSE_OF_KIND))
    stale = sorted(set(C.CAUSE_OF_KIND) - set(S.EVENT_KINDS))
    assert not missing, f"因果台帳に未分類の kind: {missing}"
    assert not stale, f"schema に無いのに因果台帳に残っている kind: {stale}"


# ------------------------------------- (4) LLM 呼を 1 本も足さないことの機械固定
def test_module_has_no_llm_call_site():
    """home_awake.py は LLM を一切呼ばない(ルールベースのみ)= 静的検査。"""
    src = inspect.getsource(HA)
    for token in ("from .llm", "import llm", "sim.llm", ".generate(",
                  "build_prompt(", "backend("):
        assert token not in src, f"home_awake.py に LLM 経路 '{token}' が混入している"


def test_llm_call_volume_stays_within_noise(tmp_path):
    """(c) 呼数: ON で LLM 呼の総量が動かない(±15% 以内)。

    ★**完全一致は成立しない**(正直な記録)。この機能は LLM の**呼び出し点**を 1 つも
      足さない(それは上の 2 本 = span テストと muted テストが直接証明している)が、
      就寝時刻が後ろへずれるぶん**起床・朝計画・夜の内省の時刻**が動き、有限長のランの
      窓に入る睡眠サイクル数と遭遇の並びが変わる。実測(mock):
        40 体 x 3 日 … OFF 1,553 → ON 1,559(**+0.4%**)
        60 体 x 3 日 … OFF 2,448 → ON 2,255(**−7.9%**・sleep_start 150→130 =
                       窓の切り口が変わったぶん)
      差の符号は一定しない = **系統的な増加ではなく再配置**である、というのがここで
      固定したい性質。呼び出し点が増えれば span テストが先に落ちる。"""
    off = _sim(tmp_path, "ha_llm_off", steps=432, n=40)
    off.run()
    on = _sim(tmp_path, "ha_llm_on", steps=432, n=40,
              extra=["daily.home_awake.enabled=true"])
    on.run()
    n_off = pq.read_table(Path(off.out_dir) / "l1b_llm.parquet").num_rows
    n_on = pq.read_table(Path(on.out_dir) / "l1b_llm.parquet").num_rows
    assert abs(n_on - n_off) <= 0.15 * n_off, \
        f"LLM 呼の総量が大きく動いた(OFF={n_off} ON={n_on})"


def test_no_llm_call_inside_home_awake_spans(tmp_path):
    """在宅覚醒中の (agent, step) には LLM 呼が **1 本も**記録されない(機構の直接証明)。"""
    sim = _sim(tmp_path, "ha_span", steps=432, n=40,
               extra=["daily.home_awake.enabled=true"])
    sim.run()
    spans: dict[int, list[tuple[int, int]]] = {}
    open_at: dict[int, int] = {}
    has_act: set[int] = set()
    for e in sim.logger.events:
        if e.kind == "enter_building" and e.payload.get("home"):
            open_at[e.agent_id] = e.step
        elif e.kind == "home_activity":
            has_act.add(e.agent_id)
        elif e.kind == "sleep_start":
            s = open_at.pop(e.agent_id, None)
            if s is not None and e.agent_id in has_act and e.step > s:
                spans.setdefault(e.agent_id, []).append((s, e.step))
            has_act.discard(e.agent_id)
    assert spans, "在宅覚醒の区間が 1 つも観測されなかった"
    rows = pq.read_table(Path(sim.out_dir) / "l1b_llm.parquet").to_pylist()
    bad = [r for r in rows
           if any(lo < int(r["step"]) < hi
                  for lo, hi in spans.get(int(r["agent_id"]), ()))]
    assert not bad, f"在宅覚醒中に LLM 呼が出ている: {bad[:5]}"


def test_muted_excludes_home_awake_from_drive_active(tmp_path):
    """在宅覚醒中の個体は _phase_drive の active から外れる(= 発火機会が増えない)。"""
    sim = _sim(tmp_path, "ha_mute",
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a = next(x for x in sim.agents
             if x.home_building and sim.city.has_building(x.home_building))
    _home_at_night(sim, a, 0)
    assert not HA.muted(a, sim)                       # まだ路上 = 通常どおり発火対象
    scheduler._apply(sim, a, routine.decide(a, 0, sim, "home", _rng(sim, a, 0),
                                            has_company=False),
                     0, sim.clock.sim_min(0))
    assert HA.muted(a, sim)                           # 在宅覚醒 = 睡眠中と同じ扱い
    active = [x for x in sim.agents if x.loc != "outside" and not x.sleeping
              and not HA.muted(x, sim)]
    assert a not in active


# -------------------------------------------------- (5) 就寝ハザードの単調性
def test_hazard_monotonic_in_every_term():
    """p_sleep が概日・疲労・翌日早出で上がり、在宅活動への没入で下がる。"""
    cfg = HA.cfg_of(load_config([]))
    sim = _NoCal()
    base = _house(bedtime_min=1320, work_start_min=-1)
    p0 = HA.sleep_prob(base, sim, 1320, cfg, engaged=False)
    p1 = HA.sleep_prob(base, sim, 1320 + 120, cfg, engaged=False)
    assert p1 > p0, "概日(就寝時刻からの経過)で就寝確率が上がらない"
    tired = _house(bedtime_min=1320, work_start_min=-1, fatigue=1.0)
    assert HA.sleep_prob(tired, sim, 1320, cfg, engaged=False) > p0, "疲労で上がらない"
    early = _house(bedtime_min=1320, work_start_min=420)      # 07:00 出勤 = 早出
    assert HA.sleep_prob(early, sim, 1320, cfg, engaged=False) > p0, "翌日早出で上がらない"
    assert HA.sleep_prob(base, sim, 1320, cfg, engaged=True) < p0, \
        "在宅活動への没入で就寝確率が下がらない"
    assert 0.0 < p0 < 1.0


def test_hazard_center_is_individual_bedtime():
    """ハザードの中心は既存の個体差 bedtime_min(新しい生理状態を足していない)。"""
    cfg = HA.cfg_of(load_config([]))
    sim = _NoCal()
    a = _house(bedtime_min=1320, work_start_min=-1)
    b = _house(bedtime_min=1380, work_start_min=-1)
    assert HA.sleep_prob(a, sim, 1380, cfg, engaged=False) \
        > HA.sleep_prob(b, sim, 1380, cfg, engaged=False)
    assert abs(HA.hours_since_bedtime(a, 1320)) < 1e-9
    assert abs(HA.hours_since_bedtime(a, 1320 + 60) - 1.0) < 1e-9
    assert abs(HA.hours_since_bedtime(a, 1320 - 60) + 1.0) < 1e-9   # 円環で符号付き


def test_hazard_coefficients_come_from_conf():
    """係数はハードコードでなく conf(daily.home_awake.hazard)から来る。"""
    cfg = HA.cfg_of(load_config(["daily.home_awake.hazard.p0=0.995"]))
    assert cfg["hazard"]["p0"] == 0.995
    p = HA.sleep_prob(_house(work_start_min=-1), _NoCal(), 1320, cfg, engaged=False)
    assert p > 0.99, "conf の p0 がハザードに効いていない"


# ------------------------------------------- (6) 在宅活動ラベルと重み表
def test_act_labels_are_the_eight_kinds():
    assert set(HA.ACTS) == {"meal", "bath", "housework", "family_talk",
                            "media", "hobby", "study", "rest"}
    assert len(HA.ACTS) == 8


def test_weights_reflect_survey_levels():
    """社会生活基本調査の水準が重みの向きに写っている(若年単身は家事が少なく趣味が多い)。"""
    cfg = HA.cfg_of(load_config([]))
    evening = 19 * 60
    young = HA.act_weights(_house(age=24, occupation="会社員"), evening, cfg,
                           mates_awake=False)
    mid = HA.act_weights(_house(age=45, occupation="会社員"), evening, cfg,
                         mates_awake=False)
    assert young["housework"] < mid["housework"], "独身期の家事 21 分/日 が反映されていない"
    assert young["hobby"] > mid["hobby"], "20 代の趣味・娯楽ピークが反映されていない"
    elder = HA.act_weights(_house(age=70, occupation="無職"), evening, cfg,
                           mates_awake=False)
    assert elder["media"] > young["media"], "テレビの年齢勾配(10 倍)が反映されていない"
    student = HA.act_weights(_house(age=20, occupation="大学生", work_start_min=-1),
                             evening, cfg, mates_awake=False)
    assert student["study"] > young["study"], "学生の学習が反映されていない"


def test_weights_depend_on_time_band():
    """時刻帯で重みが変わる(夕方は食事が主役・夜は休養とメディア)。"""
    cfg = HA.cfg_of(load_config([]))
    a = _house(age=40)
    ev = HA.act_weights(a, 19 * 60, cfg, mates_awake=False)
    ni = HA.act_weights(a, 23 * 60, cfg, mates_awake=False)
    assert ev["meal"] > ni["meal"], "夕食開始 19:17 が反映されていない"
    assert ni["rest"] > ev["rest"], "夜の休養・くつろぎが反映されていない"
    assert HA.time_band(8 * 60) == "morning" and HA.time_band(13 * 60) == "day"
    assert HA.time_band(19 * 60) == "evening" and HA.time_band(23 * 60) == "night"


def test_household_synergy_raises_family_talk():
    """世帯シナジー(最小): 同居人も在宅覚醒中なら family_talk の重みが上がる。"""
    cfg = HA.cfg_of(load_config([]))
    a = _house(age=40)
    alone = HA.act_weights(a, 20 * 60, cfg, mates_awake=False)
    together = HA.act_weights(a, 20 * 60, cfg, mates_awake=True)
    assert alone["family_talk"] == 0.0, "独居なのに団らんの重みが立っている"
    assert together["family_talk"] > 0.0
    assert together["family_talk"] > alone["family_talk"]
    # 他のラベルは 1 つも動かない(シナジーは family_talk に閉じている)
    for act in HA.ACTS:
        if act != "family_talk":
            assert alone[act] == together[act]


def test_pick_act_is_deterministic_and_in_range():
    cfg = HA.cfg_of(load_config([]))
    w = HA.act_weights(_house(age=30), 20 * 60, cfg, mates_awake=True)
    for seed in range(20):
        r1 = np.random.Generator(np.random.PCG64(seed))
        r2 = np.random.Generator(np.random.PCG64(seed))
        assert HA.pick_act(w, r1) == HA.pick_act(w, r2)
    r = np.random.Generator(np.random.PCG64(7))
    assert all(HA.pick_act(w, r) in HA.ACTS for _ in range(200))


def test_mates_synergy_uses_household_index(tmp_path):
    """同居人判定は household の housemates + 在場述語 present_agent を流用する。"""
    sim = _sim(tmp_path, "ha_mates",
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a, b = sim.agents[0], sim.agents[1]
    _home_at_night(sim, a, 0)
    _home_at_night(sim, b, 0)
    a.housemates = [b.id]
    assert not routine._mates_home_awake(a, sim)      # b はまだ路上
    b.building = b.home_building = a.home_building or "x"
    assert routine._mates_home_awake(a, sim)
    b.sleeping = True
    assert not routine._mates_home_awake(a, sim)      # 寝ている同居人は団らんの相手でない


# ------------------------------------------------------- (8) レジストリ規律
def test_registry_declares_the_toggle_and_no_undeclared():
    assert "daily.home_awake.enabled" in R.BY_ID, "レジストリ未宣言"
    f = R.BY_ID["daily.home_awake.enabled"]
    assert f.repro_tier == "strict"
    assert f.affects_k is False, "LLM の呼び出し点を足さない機能なので affects_k=False"
    assert R.undeclared_toggles(load_config()) == []


def test_conf_declares_default_off():
    cfg = HA.cfg_of(load_config([]))
    assert cfg["enabled"] is False
    assert cfg["lead_min"] == 0, "既定 lead_min は 0(帰宅時刻を動かさない)"


# ---------------------------------------------------- (9) resume == straight
def test_resume_matches_straight_with_home_awake(tmp_path):
    """ON で「一気 160step」と「80+resume」の L1/L2/L3 が完全一致(在宅覚醒が跨いで復元)。"""
    ov = {"run.seed": 42, "run.n_agents": 20, "model.backend": "mock",
          "daily.home_awake.enabled": "true", "government.enabled": "false"}

    def cfg(name, n_steps, **extra):
        dot = [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        dot += [f"run.n_steps={n_steps}", f"run.name={name}"]
        return load_config(dot)

    def rows(run_dir, stem):
        return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()

    straight = tmp_path / "ha_straight"
    Simulation(cfg("ha_straight", 160), out_dir=straight).run()

    resumed = tmp_path / "ha_resume"
    every = {"observer.checkpoint_every": 80}
    sim1 = Simulation(cfg("ha_resume", 80, **every), out_dir=resumed)
    for step in range(80):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 80, resumed / "checkpoint" / "ckpt-000080.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(cfg("ha_resume", 160, **every), out_dir=resumed).run(resume_from=resumed)

    assert rows(straight, "l1_events") == rows(resumed, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert rows(straight, stem) == rows(resumed, stem), f"{stem} 不一致"


# --------------------------------------------------------------------------- #
# (10) 帰宅前倒しの個体分布(lead.mode=per_agent。ユーザー決定 2026-08-16)
# --------------------------------------------------------------------------- #
def test_lead_mode_default_is_fixed_and_zero():
    """既定は fixed かつ lead_min=0 = 帰宅時刻を 1 分も動かさない(後方互換)。"""
    cfg = HA.cfg_of(load_config([]))
    assert cfg["lead"]["mode"] == "fixed"
    assert cfg["lead_min"] == 0
    assert set(cfg["lead"]["segment_base"]) == {"worker", "student",
                                                "non_working", "night_worker"}
    assert sum(cfg["lead"]["spread_quantiles"]) == 0, \
        "分位テーブルの合計が 0 でない = セグメント基準が母平均にならない"


def test_lead_mode_unknown_raises():
    with pytest.raises(ValueError):
        HA.cfg_of(load_config(["daily.home_awake.lead.mode=weekly"]))


def test_lead_segment_covers_worker_student_nonworking_nightshift():
    """セグメントは 就業状態 × 夜勤有無。夜勤は就業状態より優先する。"""
    assert HA.lead_segment(_house(occupation="会社員", work_start_min=540,
                                  work_end_min=1080)) == "worker"
    assert HA.lead_segment(_house(occupation="大学生", work_start_min=-1)) == "student"
    assert HA.lead_segment(_house(occupation="無職", work_start_min=-1)) == "non_working"
    # 日跨ぎ勤務(22:00→06:00)も夕方始業(18:00→)も夜勤
    assert HA.lead_segment(_house(occupation="警備員", work_start_min=1320,
                                  work_end_min=360)) == "night_worker"
    assert HA.lead_segment(_house(occupation="会社員", work_start_min=1080,
                                  work_end_min=1380)) == "night_worker"


def test_lead_individual_offset_is_stable_and_consumes_no_stream():
    """個体差は blake2b の分位 = 決定論・プロセス跨ぎ安定・乱数 stream をゼロ消費。"""
    cfg = HA.cfg_of(load_config(["daily.home_awake.lead.mode=per_agent"]))
    vals = {i: HA.base_lead_min(_house(id=i, occupation="会社員",
                                       work_start_min=540, work_end_min=1080), cfg)
            for i in range(200)}
    assert vals == {i: HA.base_lead_min(_house(id=i, occupation="会社員",
                                               work_start_min=540, work_end_min=1080), cfg)
                    for i in range(200)}, "同じ個体で値が揺れる(決定論でない)"
    assert len(set(vals.values())) >= 8, "個体差が効いていない(分位が 1 つに潰れている)"
    mean = sum(vals.values()) / len(vals)
    base = cfg["lead"]["segment_base"]["worker"]
    assert base - 55 <= mean <= base + 25, \
        f"worker セグメントの平均が基準 {base} から外れすぎ: {mean}"


def test_lead_segments_are_ordered_as_designed():
    """非就業 > 学生 > 有業 > 夜勤 の順(= 出典コメントの向きどおり)。"""
    cfg = HA.cfg_of(load_config(["daily.home_awake.lead.mode=per_agent"]))
    b = cfg["lead"]["segment_base"]
    assert b["non_working"] > b["student"] > b["worker"] > b["night_worker"]


def test_lead_per_agent_moves_home_time_but_not_bedtime(tmp_path):
    """per_agent は帰宅時刻だけを前倒しし、bedtime_min(AGE-B の U 字)は 1 分も触らない。"""
    sim = _sim(tmp_path, "ha_lead", steps=1, n=20,
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.lead.mode=per_agent"])
    before = {a.id: a.bedtime_min for a in sim.agents}
    hcfg = HA.settings(sim)
    leads = [HA.lead_min_for(a, sim, sim.clock.sim_min(0), hcfg) for a in sim.agents]
    assert {a.id: a.bedtime_min for a in sim.agents} == before, \
        "lead の計算が bedtime_min を書き換えている"
    assert max(leads) > 0 and len(set(leads)) > 1, "個体別になっていない"


def test_lead_is_decided_once_per_day(tmp_path):
    """lead は 1 日 1 回だけ決めて持ち回す(毎 step 引き直さない = 日中に帰宅時刻が揺れない)。"""
    sim = _sim(tmp_path, "ha_lead1", steps=1, n=10,
               extra=["daily.home_awake.enabled=true",
                      "daily.home_awake.lead.mode=per_agent"])
    hcfg = HA.settings(sim)
    a = sim.agents[0]
    day0 = [HA.lead_min_for(a, sim, 600 + i, hcfg) for i in range(5)]
    assert len(set(day0)) == 1, "同じ日のうちに lead が変わる"
    day1 = HA.lead_min_for(a, sim, 600 + 1440, hcfg)
    assert isinstance(day1, int)               # 翌日は引き直す(値は同じでも構わない)


def test_fixed_mode_ignores_per_agent_table(tmp_path):
    """mode=fixed では segment_base を何に変えてもイベント列が動かない(回帰アンカー)。"""
    a = _sim(tmp_path, "ha_fx_a", steps=200,
             extra=["daily.home_awake.enabled=true"])
    a.run()
    b = _sim(tmp_path, "ha_fx_b", steps=200,
             extra=["daily.home_awake.enabled=true",
                    "daily.home_awake.lead.segment_base.worker=999",
                    "daily.home_awake.lead.jitter_min=120"])
    b.run()
    assert _events(a) == _events(b), "fixed なのに per_agent 用のノブが効いている"


# --------------------------------------------------------------------------- #
# (11) 同居人どうしの夜の自宅会話(evening_talk。ユーザー決定 2026-08-16)
# --------------------------------------------------------------------------- #
def test_evening_talk_default_off_and_regression_anchor(tmp_path):
    """既定 OFF。かつ evening_talk を明示 false にしても ON 挙動が一致(回帰アンカー)。"""
    assert HA.cfg_of(load_config([]))["evening_talk"]["enabled"] is False
    a = _sim(tmp_path, "ha_et_a", steps=200, n=20,
             extra=["household.enabled=true", "daily.home_awake.enabled=true"])
    a.run()
    b = _sim(tmp_path, "ha_et_b", steps=200, n=20,
             extra=["household.enabled=true", "daily.home_awake.enabled=true",
                    "daily.home_awake.evening_talk.enabled=false"])
    b.run()
    assert _events(a) == _events(b)


def test_evening_talk_opens_only_household_pairs(tmp_path):
    """ON: 同一世帯 かつ 両者 HOME_AWAKE のペアだけ開き、非世帯ペアは muted のまま。"""
    sim = _sim(tmp_path, "ha_et_pair",
               extra=["household.enabled=true",
                      "daily.home_awake.enabled=true",
                      "daily.home_awake.evening_talk.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a, mate, stranger = sim.agents[0], sim.agents[1], sim.agents[2]
    for x in (a, mate, stranger):
        _home_at_night(sim, x, 0)
    a.housemates, mate.housemates = [mate.id], [a.id]
    stranger.housemates = []
    # 3 人とも同じ建物の中で在宅覚醒にする(位置は同一 = hearers_of の文脈が揃う)
    bld = a.home_building or "b0"
    for x in (a, mate, stranger):
        x.home_building = bld
        x.building = bld
        x._home_awake_since = 0
    assert HA.home_awake_now(a) and HA.home_awake_now(mate)
    # 世帯ペア: 双方 muted を解かれ、返答権も開く
    assert not HA.muted(a, sim) and not HA.muted(mate, sim)
    assert HA.pair_open(a, mate, sim) and HA.pair_open(mate, a, sim)
    # 非世帯: 在宅覚醒中の同居人が居ないので muted のまま・返答権も閉じたまま
    assert HA.muted(stranger, sim)
    assert not HA.pair_open(a, stranger, sim)
    assert not HA.pair_open(stranger, a, sim)
    # 同居人が寝たら閉じる
    mate.sleeping = True
    assert HA.muted(a, sim)


def test_evening_talk_off_keeps_household_pairs_muted(tmp_path):
    """evening_talk OFF なら同一世帯でも閉じたまま(前回実装と同一の性質)。"""
    sim = _sim(tmp_path, "ha_et_off",
               extra=["household.enabled=true", "daily.home_awake.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a, mate = sim.agents[0], sim.agents[1]
    for x in (a, mate):
        _home_at_night(sim, x, 0)
        x._home_awake_since = 0
        x.building = x.home_building = a.home_building or "b0"
    a.housemates, mate.housemates = [mate.id], [a.id]
    assert HA.muted(a, sim) and HA.muted(mate, sim)
    assert not HA.pair_open(a, mate, sim)


def test_reply_open_holds_street_replies_but_passes_housemate_replies(tmp_path):
    """保留中の返答は「在宅覚醒中の同居人から」のときだけ消費される(外部は預かったまま)。"""
    sim = _sim(tmp_path, "ha_et_reply",
               extra=["household.enabled=true", "daily.home_awake.enabled=true",
                      "daily.home_awake.evening_talk.enabled=true",
                      "daily.home_awake.hazard.p0=1e-12"])
    a, mate, stranger = sim.agents[0], sim.agents[1], sim.agents[2]
    for x in (a, mate, stranger):
        _home_at_night(sim, x, 0)
    a.housemates, mate.housemates = [mate.id], [a.id]
    bld = a.home_building or "b0"
    for x in (a, mate):
        x.home_building = bld
        x.building = bld
        x._home_awake_since = 0
    a._reply_to = (stranger.id, "路上でかけられた声")
    assert not HA.reply_open(a, sim), "路上で受けた返答権が在宅覚醒中に消費されている"
    a._reply_to = (mate.id, "同居人の声")
    assert HA.reply_open(a, sim)
    a._reply_to = None
    assert not HA.reply_open(a, sim)


def test_evening_talk_registry_and_timeconv_declared():
    """新 conf キーは registry(第72)と timeconv(Δt 分類)の両方に載る。"""
    for fid in ("daily.home_awake.lead.mode",
                "daily.home_awake.evening_talk.enabled"):
        assert fid in R.BY_ID, f"{fid} が registry に未宣言"
    assert R.BY_ID["daily.home_awake.lead.mode"].off_value == "fixed"
    assert R.BY_ID["daily.home_awake.evening_talk.enabled"].affects_k is True, \
        "evening_talk は generate() の呼び出し点を増やすので affects_k=True"
    assert R.undeclared_toggles(load_config()) == []
    for key in ("daily.home_awake.lead.mode",
                "daily.home_awake.lead.jitter_min",
                "daily.home_awake.lead.segment_base.worker",
                "daily.home_awake.lead.age_delta.youth",
                "daily.home_awake.lead.spread_quantiles",
                "daily.home_awake.evening_talk.enabled"):
        assert T.covers(key), f"{key} が timeconv.TABLE に未分類"


def test_home_activity_label_is_not_a_conversation_record():
    """二重計上の確認: home_activity は時間配分の記録で、発話は既存の会話系 kind。

    両者は別の層(ラベル抽選はルールベース・発話は drive 由来)で、同じ出来事を
    2 回数えることはない = home_activity は会話 kind の語彙に 1 つも含まれない。"""
    assert "home_activity" in S.EVENT_KINDS
    for talk_kind in ("speak", "hear", "conversation", "dm"):
        assert talk_kind in S.EVENT_KINDS
        assert talk_kind != "home_activity"
    # family_talk はあくまで在宅活動ラベルの 1 つ(会話イベントの発生条件ではない)
    assert "family_talk" in HA.ACTS


def test_resume_matches_straight_with_lead_and_talk(tmp_path):
    """per_agent + evening_talk で resume==straight(日次 lead と在宅覚醒が跨いで復元)。"""
    ov = {"run.seed": 42, "run.n_agents": 20, "model.backend": "mock",
          "household.enabled": "true", "government.enabled": "false",
          "daily.home_awake.enabled": "true",
          "daily.home_awake.lead.mode": "per_agent",
          "daily.home_awake.evening_talk.enabled": "true"}

    def cfg(name, n_steps, **extra):
        dot = [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        dot += [f"run.n_steps={n_steps}", f"run.name={name}"]
        return load_config(dot)

    def rows(run_dir, stem):
        return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()

    straight = tmp_path / "ha2_straight"
    Simulation(cfg("ha2_straight", 160), out_dir=straight).run()

    resumed = tmp_path / "ha2_resume"
    every = {"observer.checkpoint_every": 80}
    sim1 = Simulation(cfg("ha2_resume", 80, **every), out_dir=resumed)
    for step in range(80):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 80, resumed / "checkpoint" / "ckpt-000080.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(cfg("ha2_resume", 160, **every), out_dir=resumed).run(resume_from=resumed)

    assert rows(straight, "l1_events") == rows(resumed, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert rows(straight, stem) == rows(resumed, stem), f"{stem} 不一致"


def test_on_is_deterministic(tmp_path):
    """ON: 同 seed 2 ランでイベント列が完全一致(新 stream の決定論)。"""
    a = _sim(tmp_path, "ha_det_a", steps=200,
             extra=["daily.home_awake.enabled=true"])
    a.run()
    b = _sim(tmp_path, "ha_det_b", steps=200,
             extra=["daily.home_awake.enabled=true"])
    b.run()
    assert _events(a) == _events(b)
