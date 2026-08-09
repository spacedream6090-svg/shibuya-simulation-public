"""駅員・車掌 = 持ち場を持つアクター(``transit_staff``・actor model P3a)のテスト。

正典
  - ``docs/plans/actor-model-migration-plan.md``「3a 駅員・車掌」= 車掌 = ドア閉判断
    エージェント。ダイヤ結合ループの内生化(ホーム密度 → 乗降時間 → 続行遅延 → 滞留)。
  - 較正: 東京の実測則(Palmqvist / Tomii / Ochiai 2020 ほか)= **+15 人/車両 ≒ +1 秒**、
    通勤列車の総遅延の**約 9 割が 5 分以下の停車時間超過**。

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・イベント 0 件・sim にも agent にも属性が生えない
  ② 束ね: 決定論(agent id 昇順)・**冪等**・resume 安全・既存の勤務窓を壊さない
  ③ 当直の選定: **最小 agent id**・勤務窓の外/不在は選ばれない(決定論)
  ④ ★**等価性**: 同じ負荷なら ``delay_min`` は規則1 が出したはずの値と 1 ビットも違わない
     (= アクター化で変わったのは **誰が決めるか**であって **何が起きるか**ではない)
  ⑤ 規則1 の非二重適用(両方向: OFF 対照では規則1 が確かに発火する)
  ⑥ 較正サブフラグ: ``dwell_s_per_15pax / 15`` [秒/人] に**係数だけ**差し替わる
  ⑦ 同 seed 2 ラン → L1 完全一致 / ★乱数がコードとして存在しない(AST の静的検査)
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from society import envfeedback as ENV
from society import registry as R
from society import transit_staff as TS
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden_baseline_l1.json"

# test_devices.py:45 / test_traces.py:45 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"transit_staff.enabled": "false"}
# 20 体の mock でも規則1 / 車掌ループが実際に発火する縮約(既定閾値は本番規模向け)。
_ENV_ON = {"env.feedback.enabled": "true",
           "env.feedback.log_every_steps": "1",
           "env.feedback.transit.platform_threshold": "1"}
ON = {**_ENV_ON, "transit_staff.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_devices.py / test_envfeedback.py と同型)
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n_steps=1, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _run(tmp_path, name, **kw):
    sim = _sim(tmp_path, name, **kw)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _station_crowd(sim, n=None):
    """合成高負荷: 指定人数を駅ノードへ物理的に置く(tests/test_envfeedback.py と同型)。"""
    station = sim.city.station_node
    for a in (sim.agents if n is None else sim.agents[:n]):
        a.loc, a.node, a.sleeping, a.route = "street", station, False, []
        a.x, a.y = sim.city.node_xy(station)
    return station


def _make_staff(sim, station_ids=(), crew_ids=(), at_station=True,
                window=(0, 1440)):
    """指定 id の個体を駅員 / 乗務員にする(職業を書くだけ。束ねは bind が行う)。"""
    by_id = {int(a.id): a for a in sim.agents}
    for aid in station_ids:
        by_id[int(aid)].occupation = "駅員"
    for aid in crew_ids:
        by_id[int(aid)].occupation = "車掌"
    for aid in list(station_ids) + list(crew_ids):
        a = by_id[int(aid)]
        a.work_start_min, a.work_end_min = int(window[0]), int(window[1])
        a.sick = False
        if at_station:
            a.loc, a.node, a.sleeping, a.route = "street", sim.city.station_node, False, []
            a.x, a.y = sim.city.node_xy(a.node)
    return by_id


def _service_min(sim, day=0):
    """運行時間帯に入る sim 分(始発〜終電の中心。1 駅 1 本でも必ず在線する時刻)。"""
    for m in range(day * 1440, day * 1440 + 1440, 10):
        if sim.transit.has_service(m):
            return m
    raise AssertionError("運行時間帯が 1 つも無いダイヤ")


def _no_service_min(sim, day=0):
    for m in range(day * 1440, day * 1440 + 1440, 10):
        if not sim.transit.has_service(m):
            return m
    raise AssertionError("終電後の時間帯が無いダイヤ")


# =========================================================================== #
# (A) 出荷既定・宣言
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.transit_staff.enabled) is False
    # サブフラグは「親が ON になったときの既定形」= true(親 OFF では 1 バイトも効かない)
    assert bool(cfg.transit_staff.bind.enabled) is True
    assert bool(cfg.transit_staff.dwell.enabled) is True
    assert bool(cfg.transit_staff.dwell.require_service) is True
    # ★較正値は東京の実測則(+15 人/車両 ≒ +1 秒)。既定では**使わない**
    assert bool(cfg.transit_staff.dwell.calibrated) is False
    assert float(cfg.transit_staff.dwell.dwell_s_per_15pax) == 1.0
    assert list(cfg.transit_staff.bind.station_occupations) == ["駅員"]
    assert list(cfg.transit_staff.bind.crew_occupations) == ["車掌", "電車運転士"]


def test_registry_and_schema_declared():
    feats = {f.id: f for f in R.FEATURES}
    for fid in ("transit_staff.enabled", "transit_staff.bind.enabled",
                "transit_staff.dwell.enabled",
                "transit_staff.dwell.require_service",
                "transit_staff.dwell.calibrated"):
        feat = feats[fid]
        assert feat.repro_tier == "strict"        # 乱数を 1 本も引かない
        assert feat.affects_k is False            # generate() の呼び出しサイト不変
        assert feat.fingerprint_risk == "none"    # プロンプトへ 1 バイトも足さない
    # L1 種は**材料側**(transit_staff.py の import)で登録している
    for kind in ("dwell_decision", "transit_staff_bound"):
        assert kind in EVENT_KINDS


def test_no_undeclared_toggles_from_this_block():
    """本ブロックの bool リーフはすべてレジストリ宣言済み(CI ゲートの自分ぶん)。"""
    mine = [k for k in R.undeclared_toggles(load_config())
            if k.startswith("transit_staff.")]
    assert mine == []


def test_config_degrades_unknown_and_forces_types():
    cfg = TS.build_cfg({"enabled": "true", "未知": 3,
                        "bind": {"enabled": 1, "n_shifts": "0", "shift_hours": -4,
                                 "station_occupations": "駅員", "未知": 1},
                        "dwell": {"calibrated": 1, "dwell_s_per_15pax": "-2",
                                  "log_every_steps": 0, "未知": 1}})
    assert cfg["enabled"] is True and "未知" not in cfg
    assert cfg["bind"]["enabled"] is True and "未知" not in cfg["bind"]
    assert cfg["bind"]["n_shifts"] == 1 and cfg["bind"]["shift_hours"] == 1
    assert cfg["bind"]["station_occupations"] == ("駅員",)       # 文字列 1 本も受ける
    assert cfg["dwell"]["calibrated"] is True
    assert cfg["dwell"]["dwell_s_per_15pax"] == 0.0
    assert cfg["dwell"]["log_every_steps"] == 1
    # 空リストは既定へ戻る(「乗務員が 1 人も存在しない世界」を事故で作らない)
    assert TS.build_cfg({"bind": {"crew_occupations": []}})["bind"][
        "crew_occupations"] == ("車掌", "電車運転士")


def test_module_has_no_rng_no_llm():
    """★アクターの判断は**観測された状態の純関数**(設計契約)= 乱数を 1 本も引かない。

    散文(docstring / コメント)は対象外 — 実際に評価される識別子だけを見る
    (tests/test_devices.py::test_module_has_no_rng_no_llm と同流儀)。
    """
    tree = ast.parse(Path(TS.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    idents = attrs | names | funcs
    for bad in ("rng", "random", "shuffle", "choice", "seed", "uniform",
                "gauss", "poisson", "jitter"):
        hit = sorted(i for i in idents if bad in i.lower())
        assert hit == [], f"アクターに乱数相当の識別子がある(決定論契約の違反): {hit}"
    assert "hub" not in idents and "stream" not in idents
    assert "generate" not in idents and "llm" not in attrs


def test_no_dwell_constants_are_duplicated():
    """★停車時間の定数・式は ``envfeedback`` にしか無い(写経の禁止を機械固定)。

    本 module に ``recovery`` / ``dwell_cap_min`` / ``delay_cap_min`` /
    ``dwell_sec_per_pax`` の**リテラル文字列も 60 での除算も**現れないことを AST で見る
    (散文は対象外)。この 4 つが無ければ停車時間の式は**再現できない**。
    ★``platform_threshold`` だけは例外的に許す: 演算(超過人数)には使わず、
      ``dwell_decision`` の payload に「どの閾値と比べたか」を残すためだけに読む。
      **1 箇所しか読んでいない**ことを下で数える(演算へ持ち込まれていない証拠)。
    """
    tree = ast.parse(Path(TS.__file__).read_text(encoding="utf-8"))
    docs = {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)}          # docstring は散文=対象外
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docs}
    for key in ("recovery", "dwell_cap_min", "delay_cap_min", "dwell_sec_per_pax"):
        assert key not in strings, f"規則1 の定数キーが写経されている: {key}"
    src = Path(TS.__file__).read_text(encoding="utf-8")
    assert src.count('"platform_threshold"') == 1, "閾値が記録以外の用途で読まれている"
    # 唯一許される算術は較正係数の 15 分割(秒/15人 → 秒/人)。60 での換算は共有関数の側。
    divisors = {n.right.value for n in ast.walk(tree)
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                and isinstance(n.right, ast.Constant)}
    assert divisors == {15.0}, f"想定外の除算定数がある: {divisors}"


# =========================================================================== #
# (B) OFF = 現行と 1 バイトも変わらない(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    """既定 OFF のゴールデン一致(``env.feedback`` の共有関数抽出が恒等であることの本丸)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    dot = ["run.seed=42", "run.n_agents=15", "run.n_steps=144",
           "run.name=ts_golden"]                          # ゴールデン採取時と同一条件
    dot += [f"{k}={v}" for k, v in _GOLDEN_NEUTRAL.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "ts_golden")
    sim.run()
    assert _l1(sim) == golden, "駅員・車掌レーンの seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _run(tmp_path, "ts_pure", n_steps=24)
    off = _run(tmp_path, "ts_off", n_steps=24, **OFF)
    assert _l1(pure) == _l1(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    sim = _run(tmp_path, "ts_off_noop", n_steps=24)
    for kind in ("dwell_decision", "transit_staff_bound"):
        assert _kind(sim, kind) == []
    assert getattr(sim, "_transit_staff_bound", None) is None
    assert TS.enabled(sim) is False
    assert TS.dwell_loop_active(sim) is False
    assert TS.bind(sim) == {"node": "", "n_station": 0, "n_crew": 0, "n_kept": 0}


def test_off_leaves_envfeedback_rule1_intact(tmp_path):
    """★駅員 OFF では envfeedback 単独ランの L1 が 1 バイトも変わらない(共有関数の抽出が恒等)。"""
    a = _run(tmp_path, "ts_envonly_a", n_steps=24, **_ENV_ON)
    b = _run(tmp_path, "ts_envonly_b", n_steps=24, **_ENV_ON,
             **{"transit_staff.enabled": "false"})
    assert _l1(a) == _l1(b)


# =========================================================================== #
# (C) 束ね = 持ち場(検収基準 ②)
# =========================================================================== #
def test_bind_is_deterministic_and_idempotent(tmp_path):
    sim = _sim(tmp_path, "ts_bind", **ON)
    _make_staff(sim, station_ids=(3, 1), crew_ids=(2,), at_station=False,
                window=(-1, 0))
    for a in sim.agents:                                  # 勤務窓を持たない状態に戻す
        if a.id in (1, 2, 3):
            a.work_start_min, a.work_end_min, a.work_node = -1, 0, ""
    first = TS.bind(sim)
    snap = [(a.id, a.work_node, a.work_start_min, a.work_end_min,
             a.work_building, a.work_floor) for a in sim.agents]
    second = TS.bind(sim)
    snap2 = [(a.id, a.work_node, a.work_start_min, a.work_end_min,
              a.work_building, a.work_floor) for a in sim.agents]
    assert snap == snap2, "冪等でない(2 度目の束ねが状態を動かした)"
    assert first["n_station"] == 2 and first["n_crew"] == 1
    assert first["n_kept"] == 0 and second["n_kept"] == 3
    assert first["node"] == sim.city.station_node
    for aid in (1, 2, 3):
        a = next(x for x in sim.agents if x.id == aid)
        assert a.work_node == sim.city.station_node
        assert a.work_start_min >= 0 and a.work_end_min > a.work_start_min
        assert a.work_building == "" and a.work_floor == 0


def test_bind_shift_table_is_a_pure_function_of_roster_order(tmp_path):
    """当直表は名簿順(agent id 昇順)index の純関数 = 乱数ゼロ・ラン跨ぎで同一。"""
    bcfg = TS.build_cfg({"bind": {"first_open": "05:00", "shift_hours": 8,
                                  "n_shifts": 2}})["bind"]
    assert TS.duty_window(bcfg, 0) == (300, 780)          # 05:00-13:00
    assert TS.duty_window(bcfg, 1) == (780, 1260)         # 13:00-21:00
    assert TS.duty_window(bcfg, 2) == TS.duty_window(bcfg, 0)   # 直は循環する
    three = TS.build_cfg({"bind": {"n_shifts": 3}})["bind"]
    assert TS.duty_window(three, 2) == (1260, 1440)       # 日跨ぎは 24:00 で切る


def test_bind_does_not_overwrite_an_existing_work_window(tmp_path):
    sim = _sim(tmp_path, "ts_bind_keep", **ON)
    _make_staff(sim, crew_ids=(5,), at_station=False, window=(9 * 60, 17 * 60))
    a = next(x for x in sim.agents if x.id == 5)
    a.work_node = ""                                      # 窓はあるが持ち場が無い
    TS.bind(sim)
    assert a.work_node == sim.city.station_node           # 持ち場は与える
    assert (a.work_start_min, a.work_end_min) == (9 * 60, 17 * 60)   # 窓は触らない


def test_bind_off_subflag_leaves_agents_untouched(tmp_path):
    sim = _sim(tmp_path, "ts_bind_off", **ON,
               **{"transit_staff.bind.enabled": "false"})
    _make_staff(sim, crew_ids=(4,), at_station=False, window=(-1, 0))
    a = next(x for x in sim.agents if x.id == 4)
    a.work_node, a.work_start_min = "", -1
    assert TS.bind(sim)["node"] == ""
    assert a.work_node == "" and a.work_start_min == -1


# =========================================================================== #
# (D) 当直の選定(検収基準 ③)
# =========================================================================== #
def test_crew_pick_is_lowest_agent_id_present_and_on_duty(tmp_path):
    sim = _sim(tmp_path, "ts_crew", **ON)
    _make_staff(sim, crew_ids=(7, 2, 5))
    sim_min = _service_min(sim)
    assert TS.on_duty_crew(sim, sim_min).id == 2          # 最小 id
    # 勤務窓の外は選ばれない
    next(x for x in sim.agents if x.id == 2).work_end_min = 0
    assert TS.on_duty_crew(sim, sim_min).id == 5
    # 駅に居ない者も選ばれない
    other = next(n for n in sim.city.graph.nodes if n != sim.city.station_node)
    next(x for x in sim.agents if x.id == 5).node = other
    assert TS.on_duty_crew(sim, sim_min).id == 7
    # 眠っている者も選ばれない → 全員外れると None
    next(x for x in sim.agents if x.id == 7).sleeping = True
    assert TS.on_duty_crew(sim, sim_min) is None


def test_is_crew_reads_only_configured_occupations(tmp_path):
    sim = _sim(tmp_path, "ts_iscrew", **ON)
    a, b = sim.agents[0], sim.agents[1]
    a.occupation, b.occupation = "車掌", "駅員"
    assert TS.is_crew(sim, a) is True
    assert TS.is_crew(sim, b) is False                    # 駅員はドアを閉めない


# =========================================================================== #
# (E) ★等価性 = 変わったのは「誰が決めるか」だけ(検収基準 ④)
# =========================================================================== #
def _drive_one_step(sim, sim_min, crowd=None):
    """1 step ぶんの停車時間ループを**実コード**で回す(集約 → 判断 → 状態)。"""
    _station_crowd(sim, crowd)
    TS.phase(sim, 0, sim_min, len(sim.logger.events))
    return ENV.state(sim)["delay_min"]


def test_delay_equals_what_rule1_would_have_computed(tmp_path):
    """★同じ負荷 → ``delay_min`` は規則1 の値と **1 ビットも違わない**(既定サブフラグ)。

    片方は駅員レーン(車掌が決める)、もう片方は envfeedback 規則1(主体なしの近似)。
    ``float`` の等値比較で固定する(丸めの逃げ道を残さない)。
    """
    staff = _sim(tmp_path, "ts_eq_staff", **ON)
    rule1 = _sim(tmp_path, "ts_eq_rule1", **_ENV_ON)
    sim_min = _service_min(staff)
    for sim in (staff, rule1):
        _make_staff(sim, crew_ids=(0,))
        _station_crowd(sim)
    got = []
    for step in range(6):
        TS.phase(staff, step, sim_min, len(staff.logger.events))
        ENV.update(rule1, step, sim_min, len(rule1.logger.events))
        got.append((ENV.state(staff)["delay_min"], ENV.state(rule1)["delay_min"]))
    assert all(a == b for a, b in got), f"アクター化で数値が動いた: {got}"
    assert got[-1][0] > 0.0, "検収が空回りしている(遅延が 1 度も出ていない)"
    # 累積カウンタ・上限も同じ状態を指す
    assert ENV.state(staff)["n_transit"] == ENV.state(rule1)["n_transit"]
    assert ENV.state(staff)["delay_max"] == ENV.state(rule1)["delay_max"]


def test_delay_is_written_to_the_same_state_downstream_reads(tmp_path):
    """遅延の書き先は規則1 と同じ ``st['delay_min']`` = 下流の消費者が無改造で読む。"""
    sim = _sim(tmp_path, "ts_downstream", **ON,
               **{"env.feedback.transit.flag_min": "0.001"})
    _make_staff(sim, crew_ids=(0,))
    assert _drive_one_step(sim, _service_min(sim)) > 0.0
    assert ENV.delay_min(sim) > 0.0
    assert ENV.delay_flag(sim) == 1.0                      # 第80 チャンネルが動く
    assert ENV.hold_exit(sim, sim.agents[1], 0, _service_min(sim)) is True


def test_no_service_step_injects_nothing_but_still_decays(tmp_path):
    """運行が無い step は停車時間を足さない(閉めるドアが無い)= 回復運転項だけ回る。"""
    sim = _sim(tmp_path, "ts_nosvc", **ON)
    _make_staff(sim, crew_ids=(0,))
    svc = _service_min(sim)
    peak = _drive_one_step(sim, svc)
    assert peak > 0.0
    dead = _no_service_min(sim)
    after = _drive_one_step(sim, dead)
    gamma = float(sim.envfbcfg["transit"]["recovery"])
    assert after == ENV.advance_delay(sim.envfbcfg["transit"], peak, 0.0) == \
        min(float(sim.envfbcfg["transit"]["delay_cap_min"]), gamma * peak)
    assert _kind(sim, "dwell_decision") == [] or all(
        e.sim_min != dead for e in _kind(sim, "dwell_decision")), \
        "電車が居ない step にドア閉判断が出ている"


def test_require_service_false_is_byte_equal_to_rule1_on_every_step(tmp_path):
    """``require_service: false`` なら運行時間外も含め規則1 と全 step 完全同値。"""
    staff = _sim(tmp_path, "ts_rs_staff", **ON,
                 **{"transit_staff.dwell.require_service": "false"})
    rule1 = _sim(tmp_path, "ts_rs_rule1", **_ENV_ON)
    for sim in (staff, rule1):
        _make_staff(sim, crew_ids=(0,))
        _station_crowd(sim)
    dead = _no_service_min(staff)
    for step in range(4):
        TS.phase(staff, step, dead, len(staff.logger.events))
        ENV.update(rule1, step, dead, len(rule1.logger.events))
        assert ENV.state(staff)["delay_min"] == ENV.state(rule1)["delay_min"]
    assert ENV.state(staff)["delay_min"] > 0.0


# =========================================================================== #
# (F) L1 の帰属(誰が決めたか)
# =========================================================================== #
def test_dwell_decision_is_attributed_to_the_on_duty_crew(tmp_path):
    sim = _sim(tmp_path, "ts_attr", **ON)
    _make_staff(sim, crew_ids=(6, 3))
    sim_min = _service_min(sim)
    _drive_one_step(sim, sim_min)
    got = _kind(sim, "dwell_decision")
    assert len(got) == 1
    ev = got[0]
    assert ev.agent_id == 3                                # 最小 id の乗務員
    assert ev.payload["unstaffed"] is False
    assert "operator" not in ev.payload
    assert ev.payload["delay_s"] > 0.0
    assert ev.payload["platform_load"] == ev.payload["standing"] + ev.payload["exchange"]
    assert ev.payload["excess"] == max(
        0, ev.payload["platform_load"] - ev.payload["threshold"])
    assert ev.payload["node"] == sim.city.station_node
    assert ev.payload["n_lines"] >= 1 and ev.payload["line"]
    crew = next(a for a in sim.agents if a.id == 3)
    assert (ev.x, ev.y) == (crew.x, crew.y)


def test_dwell_decision_marks_unstaffed_when_no_crew_is_present(tmp_path):
    """★乗務員が居ない回は**黙って進めない**: agent_id=-1 + unstaffed マーカーで残す。"""
    sim = _sim(tmp_path, "ts_unstaffed", **ON)          # 乗務員を 1 人も作らない
    sim_min = _service_min(sim)
    assert TS.on_duty_crew(sim, sim_min) is None
    delay = _drive_one_step(sim, sim_min)
    assert delay > 0.0, "無人でも物理は進む(遅延は起きる)"
    got = _kind(sim, "dwell_decision")
    assert len(got) == 1 and got[0].agent_id == -1
    assert got[0].payload["unstaffed"] is True
    assert got[0].payload["operator"] == TS.operator_device_id(sim.city.station_node)
    assert (got[0].x, got[0].y) == sim.city.node_xy(sim.city.station_node)


def test_bound_event_is_emitted_once_at_step0(tmp_path):
    sim = _sim(tmp_path, "ts_bound_ev", **ON)
    _make_staff(sim, station_ids=(1,), crew_ids=(2,))
    TS.phase(sim, 0, _service_min(sim), len(sim.logger.events))
    TS.phase(sim, 1, _service_min(sim), len(sim.logger.events))
    got = _kind(sim, "transit_staff_bound")
    assert len(got) == 1 and got[0].step == 0 and got[0].agent_id == -1
    assert got[0].payload["n_station"] == 1 and got[0].payload["n_crew"] == 1
    assert got[0].payload["node"] == sim.city.station_node


def test_bound_event_is_not_re_emitted_after_resume(tmp_path):
    """resume(束ね済みフラグが checkpoint に無い)でも L1 に 2 件並ばない。"""
    sim = _sim(tmp_path, "ts_bound_resume", **ON)
    _make_staff(sim, crew_ids=(2,))
    TS.phase(sim, 0, _service_min(sim), len(sim.logger.events))
    sim._transit_staff_bound = False                    # resume 相当(フラグだけ消える)
    TS.phase(sim, 7, _service_min(sim), len(sim.logger.events))
    assert len(_kind(sim, "transit_staff_bound")) == 1


# =========================================================================== #
# (G) 規則1 の非二重適用(検収基準 ⑤ = 両方向)
# =========================================================================== #
def _fire_rule1_only(sim, step, sim_min):
    """envfeedback 側だけを直接叩いて規則1 の状態がどう動いたかを見る。"""
    since = len(sim.logger.events)
    ENV.update(sim, step, sim_min, since)
    return ENV.state(sim)


def test_rule1_fires_when_staff_layer_is_off(tmp_path):
    """(対照)駅員 OFF では規則1 が**確かに**発火する = 下のテストが空回りしない。"""
    sim = _sim(tmp_path, "ts_dbl_off", **_ENV_ON)
    _station_crowd(sim)
    st = _fire_rule1_only(sim, 0, _service_min(sim))
    assert st["delay_min"] > 0.0 and st["n_transit"] == 1
    assert [e for e in sim.logger.events
            if e.kind == ENV.EVENT_KIND
            and e.payload.get("rule") == ENV.RULE_TRANSIT], "規則1 の L1 が無い"


def test_rule1_is_not_evaluated_when_staff_layer_is_on(tmp_path):
    """★駅員 ON では規則1 を**評価しない**(状態も L1 も 1 つも動かない)。"""
    sim = _sim(tmp_path, "ts_dbl_on", **ON)
    _station_crowd(sim)
    assert TS.dwell_loop_active(sim) is True
    st = _fire_rule1_only(sim, 0, _service_min(sim))
    assert st["delay_min"] == 0.0 and st["n_transit"] == 0
    assert [e for e in sim.logger.events
            if e.kind == ENV.EVENT_KIND
            and e.payload.get("rule") == ENV.RULE_TRANSIT] == []
    # 規則3(POI 占有)は別の物理なので touch していない
    assert "poi_hold" in st


def _load_now(sim, since_idx=-1):
    """いまのホーム負荷(共有集約器から。テストが式を写経しないための 1 行)。"""
    station = sim.city.station_node
    at_node, _inflow, exchange = ENV.aggregate(sim, station, since_idx)
    return ENV.platform_load(at_node, station, exchange)


def test_no_double_application_in_a_full_step(tmp_path):
    """1 step 通しでも遅延は 1 回しか積まれない(update → phase の順序込み)。"""
    sim = _sim(tmp_path, "ts_dbl_full", **ON)
    _make_staff(sim, crew_ids=(0,))
    _station_crowd(sim)
    sim_min = _service_min(sim)
    since = len(sim.logger.events)
    ENV.update(sim, 0, sim_min, since)
    TS.phase(sim, 0, sim_min, since)
    tcfg = sim.envfbcfg["transit"]
    assert ENV.state(sim)["delay_min"] == ENV.dwell_step(tcfg, _load_now(sim), 0.0)[2]
    assert ENV.state(sim)["n_transit"] == 1


def test_dwell_loop_stays_inactive_without_envfeedback(tmp_path):
    """env.feedback が OFF なら駅員 ON でもループを所有しない(誰も読まない数字を書かない)。"""
    sim = _sim(tmp_path, "ts_noenv", **{"transit_staff.enabled": "true"})
    _make_staff(sim, crew_ids=(0,))
    assert TS.enabled(sim) is True
    assert TS.dwell_loop_active(sim) is False
    _station_crowd(sim)
    TS.phase(sim, 0, _service_min(sim), len(sim.logger.events))
    assert _kind(sim, "dwell_decision") == []
    assert getattr(sim, "_envfb", None) is None            # 状態すら作らない
    # 束ねだけは走る(持ち場の実体化は env.feedback と独立の観測価値がある)
    assert len(_kind(sim, "transit_staff_bound")) == 1


def test_dwell_subflag_off_leaves_rule1_in_charge(tmp_path):
    sim = _sim(tmp_path, "ts_dwell_off", **ON,
               **{"transit_staff.dwell.enabled": "false"})
    _station_crowd(sim)
    assert TS.dwell_loop_active(sim) is False
    st = _fire_rule1_only(sim, 0, _service_min(sim))
    assert st["delay_min"] > 0.0                           # 規則1 が担当に戻る


# =========================================================================== #
# (H) 較正サブフラグ(検収基準 ⑥)
# =========================================================================== #
def test_calibrated_subflag_swaps_only_the_coefficient(tmp_path):
    """``calibrated: true`` → 係数が ``dwell_s_per_15pax / 15`` [秒/人] に差し替わる。

    東京実証則(+15 人 ≒ +1 秒)= 1/15 秒/人。式・上限・回復運転項は共有のまま。
    """
    legacy = _sim(tmp_path, "ts_cal_off", **ON)
    calib = _sim(tmp_path, "ts_cal_on", **ON,
                 **{"transit_staff.dwell.calibrated": "true"})
    sim_min = _service_min(legacy)
    for sim in (legacy, calib):
        _make_staff(sim, crew_ids=(0,))
    a = _drive_one_step(legacy, sim_min)
    b = _drive_one_step(calib, sim_min)
    tcfg = legacy.envfbcfg["transit"]
    load = _load_now(legacy)
    assert load == _load_now(calib) > int(tcfg["platform_threshold"])
    assert a == ENV.dwell_step(tcfg, load, 0.0)[2]                    # legacy 係数
    assert b == ENV.dwell_step(tcfg, load, 0.0, 1.0 / 15.0)[2]        # 東京較正係数
    assert b < a, "既定 0.8 秒/人 より東京較正 (1/15 秒/人) の方が小さいはず"
    ev = _kind(calib, "dwell_decision")[0]
    assert ev.payload["calibrated"] is True
    assert _kind(legacy, "dwell_decision")[0].payload["calibrated"] is False


def test_calibration_never_touches_rule1_numbers(tmp_path):
    """較正サブフラグは規則1(駅員 OFF)の数値に一切影響しない。"""
    base = _sim(tmp_path, "ts_cal_base", **_ENV_ON)
    with_flag = _sim(tmp_path, "ts_cal_flag", **_ENV_ON,
                     **{"transit_staff.dwell.calibrated": "true"})
    for sim in (base, with_flag):
        _station_crowd(sim)
        _fire_rule1_only(sim, 0, _service_min(sim))
    assert ENV.state(base)["delay_min"] == ENV.state(with_flag)["delay_min"] > 0.0


# =========================================================================== #
# (I) 再現性(検収基準 ⑦)
# =========================================================================== #
def test_two_identical_on_runs_match_byte_for_byte(tmp_path):
    a = _run(tmp_path, "ts_rep_a", n_steps=24, **ON)
    b = _run(tmp_path, "ts_rep_b", n_steps=24, **ON)
    assert _l1(a) == _l1(b)
    assert len(a.logger.llm_calls) == len(b.logger.llm_calls) > 0


def test_on_smoke_run_completes_and_binds(tmp_path):
    """mock 24 step の通しラン(ON)。束ねが 1 度だけ起き、L1 が壊れない。"""
    sim = _sim(tmp_path, "ts_smoke", n_steps=24, **ON)
    for aid in (1, 2):
        next(a for a in sim.agents if a.id == aid).occupation = "車掌"
    next(a for a in sim.agents if a.id == 3).occupation = "駅員"
    sim.run()
    bound = _kind(sim, "transit_staff_bound")
    assert len(bound) == 1
    assert bound[0].payload["n_crew"] == 2 and bound[0].payload["n_station"] == 1
    for aid in (1, 2, 3):
        assert next(a for a in sim.agents if a.id == aid).work_node == \
            sim.city.station_node
    # 全 dwell_decision は駅ノードで起き、agent_id は乗務員か -1 のどちらか
    crew_ids = {1, 2}
    for e in _kind(sim, "dwell_decision"):
        assert e.payload["node"] == sim.city.station_node
        assert e.agent_id in crew_ids or e.agent_id == -1
        assert (e.agent_id == -1) == e.payload["unstaffed"]


def test_scheduler_seam_is_a_single_gated_line():
    """scheduler の seam は **gated な 1 行**だけ(他のフェーズには 1 バイトも触れない)。"""
    src = (REPO / "src" / "society" / "engine" / "scheduler.py").read_text(
        encoding="utf-8")
    assert src.count("transit_staff_mod.") == 1
    assert "from .. import transit_staff as transit_staff_mod" in src
    assert "transit_staff_mod.phase(sim, step, sim_min, _env_idx)" in src


def test_envfeedback_defers_through_a_single_predicate():
    """規則1 が譲る判断は 1 関数(``_staff_dwell_active``)にしか書かれていない。"""
    src = (REPO / "src" / "society" / "envfeedback.py").read_text(encoding="utf-8")
    assert src.count("_staff_dwell_active(sim)") == 2      # 定義 1 + 参照 1
    assert "from . import transit_staff as staff_mod" in src
