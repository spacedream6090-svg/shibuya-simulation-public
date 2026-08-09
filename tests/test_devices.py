"""装置(device)層 = 目標を持たないアクター(``world.devices``)のテスト。

正典
  - docs/plans/actor-model-migration.md「P2: 世界に作用するものは全てアクターにする」
    (装置は目標を持たないアクター。状態を変えてよいのは
     (a) 宣言済みのスケジュール/物理[DEVS δ_int] か
     (b) エージェントの行為への応答[DEVS δ_ext] のときだけ = **自発的遷移の禁止**)
  - 較正の公表値: 日本の駅の自動改札 ≈ 60 人/分/通路(IC タッチ 0.2 秒)/
    エスカレーター実測 ≈ 80 人/分(理論値・公称値は実測より 35〜60% 高い → **使わない**)

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・イベント 0 件・sim にも agent にも属性が生えない
  ② DEVS 契約: δ の**外**での状態変化は監査証跡に illegal として残る
     (わざと契約を破る派生クラスで検出を機械固定する)
  ③ 改札の算術: 120 人 / 8 通路 / 60 人分 → 待ち最大 14.0 秒・掃け 15.0 秒 /
     到着順は agent_id 昇順の決定論 / 同 seed 2 ラン L1 一致
  ④ 改札 ON のスモーク(mock ≤24 step): gate_pass は **wait_s>0 のときだけ** /
     envfeedback 規則2 が**二重に適用されない**(状態が 1 も動かない)
  ⑤ 信号: 既存の tests/test_crossing_signals.py は**無修正で緑**(時刻計算不変)/
     交差点表 → device_id の対応 / signal_events ON でも **1 交差点 1 日 1 件まで**
  ⑥ ★乱数がコードとして存在しない(AST の静的検査。tests/test_traces.py と同流儀)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from society import devices as D
from society import envfeedback as ENV
from society import registry as R
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS
from society.world import sfm_core as _sfm
from society.world import zones as Z

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden_baseline_l1.json"

# test_traces.py:45 / test_rumors.py:45 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"world.devices.enabled": "false"}
# 通路 1 本・6 人/分(= 10 秒/人)= 1 step(600 秒)の定員 60 人。小規模ランでも
# 待ちが出る条件(既定の 8 通路 × 60 人/分 = 4800 人/step では誰も待たない)。
ON = {"world.devices.enabled": "true",
      "world.devices.faregate.enabled": "true",
      "world.devices.faregate.n_gates": 1,
      "world.devices.faregate.service_rate_per_min": 6}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_traces.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=15, zone_specs=None, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
    return cfg


def _sim(tmp_path, name, n_steps=24, n_agents=15, zone_specs=None, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, zone_specs, **ov),
                      out_dir=tmp_path / name)


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


# =========================================================================== #
# (A) 出荷既定・宣言
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.world.devices.enabled) is False
    assert bool(cfg.world.devices.faregate.enabled) is False
    assert bool(cfg.world.devices.signal_events) is False
    # ★較正値は実測系の公表値(公称値ではない): 60 人/分/通路 = 1.0 秒/人
    assert float(cfg.world.devices.faregate.service_rate_per_min) == 60.0
    assert int(cfg.world.devices.faregate.n_gates) == 8


def test_registry_and_schema_declared():
    feats = {f.id: f for f in R.FEATURES}
    for fid in ("world.devices.enabled", "world.devices.faregate.enabled",
                "world.devices.signal_events"):
        feat = feats[fid]
        assert feat.repro_tier == "strict"        # 乱数を 1 本も引かない
        assert feat.affects_k is False            # generate() の呼び出しサイト不変
        assert feat.fingerprint_risk == "none"    # プロンプトへ 1 バイトも足さない
        assert feat.off_value is False
    # L1 種は**材料側**(devices.py の import)で登録している
    for kind in ("gate_pass", "device_load", "signal_summary"):
        assert kind in EVENT_KINDS


def test_config_degrades_unknown_and_forces_types():
    cfg = D.build_cfg({"enabled": "true", "signal_events": 1, "未知": 3,
                       "faregate": {"enabled": "true", "n_gates": "0",
                                    "service_rate_per_min": "12",
                                    "max_hold_steps": -4, "未知": 1}})
    assert cfg["enabled"] is True and cfg["signal_events"] is True
    assert "未知" not in cfg and "未知" not in cfg["faregate"]
    assert cfg["faregate"]["n_gates"] == 1            # 0 本の改札は作らない
    assert cfg["faregate"]["service_rate_per_min"] == 12.0
    assert cfg["faregate"]["max_hold_steps"] == 0


# --------------------------------------------------------------------------- #
# (A-2) ★静的検査: 乱数がコードとして存在しない(検収基準 ⑥)
# --------------------------------------------------------------------------- #
def test_module_has_no_rng_no_llm():
    """★装置は**決定論的な応答関数**(設計契約)= 乱数を 1 本も引かない。

    散文(docstring / コメント)は対象外 — **実際に評価される識別子**だけを見る
    (tests/test_traces.py::test_module_has_no_propagation_no_llm_no_rng と同流儀)。
    """
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    idents = attrs | names | funcs
    for bad in ("rng", "random", "shuffle", "choice", "seed", "uniform",
                "gauss", "poisson", "jitter"):
        hit = sorted(i for i in idents if bad in i.lower())
        assert hit == [], f"装置に乱数相当の識別子がある(決定論契約の違反): {hit}"
    # 乱数ハブそのものにも触らない / LLM も呼ばない
    assert "hub" not in idents and "stream" not in idents
    assert "generate" not in idents and "llm" not in attrs


# =========================================================================== #
# (B) OFF = 現行と 1 バイトも変わらない(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "dev_golden", n_steps=144, **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "装置層の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _run(tmp_path, "dev_pure")
    off = _run(tmp_path, "dev_off", **OFF)
    assert _l1(pure) == _l1(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    sim = _run(tmp_path, "dev_off_noop")
    for kind in ("gate_pass", "device_load", "signal_summary"):
        assert _kind(sim, kind) == []
    assert getattr(sim, "devices", None) is None       # 名簿すら組まない
    assert D.faregate_active(sim) is False
    assert D.pending_exits(sim, 0) == ()
    assert D.audit_report(sim) == {}
    for agent in sim.agents:                           # agent に属性が生えない
        assert not hasattr(agent, "_dev_gate_wait")
        assert not hasattr(agent, "_dev_gate_held")


def test_off_hold_exit_is_a_no_op(tmp_path):
    sim = _sim(tmp_path, "dev_off_hold", n_steps=1)
    agent = sim.agents[0]
    D.phase(sim, 0, 0)
    assert D.hold_exit(sim, agent, 0, 0) is False
    assert not hasattr(agent, "_dev_gate_wait")
    assert getattr(sim, "devices", None) is None


# =========================================================================== #
# (C) DEVS 契約: δ の外での状態変化を監査で捕まえる(検収基準 ②)
# =========================================================================== #
class _Rogue(D.Device):
    """**わざと契約を破る**装置(自発的に状態を変える)。検出の機械固定用。"""

    kind = "rogue"
    STATE = ("level",)

    def __init__(self):
        super().__init__("rogue:1")
        self.level = 0                      # 構築(まだ seal していない = 遷移ではない)
        self.seal()

    def _delta_int(self, sim_min: int) -> bool:
        self.level += 1                     # 正当: 宣言済みスケジュール
        return True

    def _delta_ext(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        self.level += 10                    # 正当: 行為への応答
        return {"level": self.level}

    def misbehave(self):
        self.level += 100                   # ★契約違反: δ の外での自発的遷移


def test_devs_contract_allows_only_two_paths():
    dev = _Rogue()
    assert dev.audit() == {"int": 0, "ext": 0, "illegal": 0}
    dev.on_schedule(60)
    dev.on_input(3, {"kind": "x"}, 60)
    assert dev.level == 11
    assert dev.audit()["int"] == 1 and dev.audit()["ext"] == 1
    assert dev.audit()["illegal"] == 0, "正当な 2 経路が illegal に数えられている"
    assert [t["path"] for t in dev.transitions] == ["int", "ext"]


def test_devs_contract_detects_spontaneous_transition():
    """★δ の**外**で状態を変えたら監査証跡に残る(= 自発的遷移の検出)。"""
    dev = _Rogue()
    dev.on_schedule(60)
    dev.misbehave()
    assert dev.audit()["illegal"] == 1, "自発的遷移が検出されていない(契約が形骸)"
    assert dev.transitions[-1] == {"path": "illegal", "sim_min": -1,
                                   "attr": "level"}
    # 監査は「禁止」ではなく「記録」= 代入自体は通る(ランを止めない)
    assert dev.level == 101


def test_devs_audit_trail_is_bounded():
    """証跡の明細は AUDIT_MAX で打ち切る(件数のタリーは無制限に数える)。"""
    dev = _Rogue()
    for i in range(D.AUDIT_MAX + 20):
        dev.misbehave()
    assert len(dev.transitions) == D.AUDIT_MAX
    assert dev.audit()["illegal"] == D.AUDIT_MAX + 20


def test_faregate_and_signal_declare_their_state():
    """実装済み装置の ``STATE`` 宣言(信号は**記憶を持たない** = 空)。"""
    assert D.FaregateArray.STATE == ("window", "served")
    assert D.SignalDevice.STATE == ()


# =========================================================================== #
# (D) 改札の算術(検収基準 ③)
# =========================================================================== #
def _array(n_gates=8, rate=60.0, window_min=10, window_seconds=600.0):
    return D.FaregateArray("faregate:t", n_gates=n_gates,
                           service_rate_per_min=rate, window_min=window_min,
                           window_seconds=window_seconds)


def test_faregate_drain_of_120_passengers_on_8_gates():
    """120 人 / 8 通路 / 60 人分(= 1.0 秒/人)→ 最後の 1 人の待ち 14.0 秒・掃け 15.0 秒。"""
    dev = _array()
    dev.on_schedule(0)
    waits = [dev.on_input(i, {"kind": "pass"}, 0)["wait_s"] for i in range(120)]
    assert dev.headway_s == 1.0                      # 60 人/分/通路 = 1.0 秒/人
    assert all(w == float(i // 8) for i, w in enumerate(waits))
    assert waits[:8] == [0.0] * 8                    # 最初の 8 人は待たない
    assert max(waits) == 14.0                        # 120/8 = 15 巡目 = 14 人ぶん待つ
    assert max(waits) + dev.headway_s == 15.0        # 掃けきるまで 15.0 秒
    assert dev.served == 120


def test_faregate_capacity_and_window_reset():
    """整備窓(= 1 step)の定員を超えたら通れない。窓が変われば δ_int で定員が戻る。"""
    dev = _array(n_gates=1, rate=1.0)                # 60 秒/人 → 定員 10 人/600 秒
    dev.on_schedule(0)
    assert dev.capacity == 10
    got = [dev.on_input(i, {"kind": "pass"}, 0) for i in range(12)]
    assert [g["passed"] for g in got] == [True] * 10 + [False, False]
    assert dev.served == 10, "定員超過の要求で状態が動いている(δ_ext の契約違反)"
    dev.on_schedule(10)                              # 次の step = 新しい整備窓
    assert dev.window == 1 and dev.served == 0
    assert dev.on_input(99, {"kind": "pass"}, 10)["passed"] is True


def test_faregate_order_is_arrival_then_agent_id():
    """到着順(= step → agent_id 昇順)で待ちが決まる。乱数ゼロ・同入力同出力。"""
    a, b = _array(n_gates=2, rate=60.0), _array(n_gates=2, rate=60.0)
    a.on_schedule(0)
    b.on_schedule(0)
    ids = [0, 1, 2, 3, 4, 5]
    wa = [a.on_input(i, {"kind": "pass"}, 0)["wait_s"] for i in ids]
    wb = [b.on_input(i, {"kind": "pass"}, 0)["wait_s"] for i in ids]
    assert wa == wb == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert [a.on_input(i, {"kind": "pass"}, 0)["queue_len"] for i in (9, 8)] \
        == [6, 7], "queue_len が到着順で単調に増えていない"


def test_faregate_no_transition_outside_devs_paths():
    """改札の DEVS 状態は δ_int / δ_ext でしか動かない(監査 illegal ゼロ)。"""
    dev = _array()
    dev.on_schedule(0)
    for i in range(50):
        dev.on_input(i, {"kind": "pass"}, 0)
    dev.on_schedule(10)
    assert dev.audit()["illegal"] == 0
    assert dev.audit()["ext"] == 50 and dev.audit()["int"] == 2


# =========================================================================== #
# (E) 改札 ON のスモーク(mock ≤24 step。検収基準 ④)
# =========================================================================== #
def test_on_two_identical_runs_match(tmp_path):
    a = _run(tmp_path, "dev_on_a", **ON)
    b = _run(tmp_path, "dev_on_b", **ON)
    assert _l1(a) == _l1(b), "装置 ON で同 seed 2 ランが一致しない(決定論の破れ)"


def test_on_registry_holds_the_station_faregate(tmp_path):
    sim = _sim(tmp_path, "dev_reg", n_steps=1, **ON)
    D.phase(sim, 0, 0)
    reg = sim.devices
    assert len(reg) == 1
    dev = reg.of_kind("faregate")[0]
    assert dev.device_id == D.faregate_device_id(sim.city.station_node)
    assert reg.get(dev.device_id) is dev
    assert reg.ids() == (dev.device_id,)
    assert dev.n_gates == 1 and dev.headway_s == 10.0     # 6 人/分 = 10 秒/人


def _station_crowd(sim, n=None):
    """合成高負荷: 指定人数を駅ノードへ物理的に置く(tests/test_envfeedback.py と同型)。"""
    station = sim.city.station_node
    for a in (sim.agents if n is None else sim.agents[:n]):
        a.loc, a.node, a.sleeping, a.route = "street", station, False, []
        a.x, a.y = sim.city.node_xy(station)
        a.exit_intent, a.homing, a.stay_until = True, False, 0
    return station


def _exit_all(sim, agents, step=0):
    """**実コード**(scheduler._try_exit)で駅の改札を通らせる(agent_id 昇順)。"""
    sim_min = sim.clock.sim_min(step)
    D.phase(sim, step, sim_min)
    for a in sorted(agents, key=lambda x: int(x.id)):
        scheduler._try_exit(sim, a, step, sim_min)
    return sim_min


def test_on_gate_pass_only_when_waiting(tmp_path):
    """``gate_pass`` は **wait_s>0 の通過だけ**(L1 の量を原理的に抑える)。

    合成高負荷(駅に 20 人)を **scheduler._try_exit の seam 経由**で通す。
    通路 1 本 × 6 人/分 = 10 秒/人 なので、待ちは 0, 10, 20, … 秒。
    """
    sim = _sim(tmp_path, "dev_on_pass", n_steps=1, n_agents=20, **ON)
    _station_crowd(sim, 20)
    _exit_all(sim, sim.agents[:20])
    assert all(a.loc == "outside" for a in sim.agents[:20]), "誰も改札を通れていない"
    passes = _kind(sim, "gate_pass")
    assert len(passes) == 19, "先頭 1 人(待ちゼロ)以外の 19 件が出ていない"
    for e in passes:
        assert e.payload["wait_s"] > 0.0, "待ちゼロの通過が L1 に出ている"
        assert e.payload["device_id"] == D.faregate_device_id(sim.city.station_node)
        assert e.payload["dir"] == "exit"
        assert e.agent_id >= 0                       # 通った本人が主語
    waits = [e.payload["wait_s"] for e in passes]
    assert waits == [10.0 * i for i in range(1, 20)]
    assert [e.agent_id for e in passes] == sorted(e.agent_id for e in passes), \
        "到着順が agent_id 昇順になっていない(決定論の破れ)"
    # 監査: 実ラン中も自発的遷移はゼロ
    assert D.audit_report(sim) and \
        all(a["illegal"] == 0 for a in D.audit_report(sim).values())


def test_on_device_load_is_at_most_one_per_device_per_hour(tmp_path):
    sim = _sim(tmp_path, "dev_on_load", n_steps=24, n_agents=20, **ON)
    _station_crowd(sim, 20)
    _exit_all(sim, sim.agents[:20])
    for step in range(1, 24):                        # 4 時間ぶん時計を進める
        D.phase(sim, step, sim.clock.sim_min(step))
    loads = _kind(sim, "device_load")
    assert loads, "負荷要約が 1 件も出ていない(検収の空回り)"
    seen = set()
    for e in loads:
        key = (e.payload["device_id"], e.sim_min // 60)
        assert key not in seen, f"device_load が 1 台 1 時間に 2 件以上出た: {key}"
        seen.add(key)
        assert e.agent_id == -1
        assert e.payload["kind"] == "faregate"
    assert loads[0].payload["passed"] == 20
    assert loads[0].payload["wait_s_max"] == 190.0   # 20 人目 = 19 × 10 秒
    assert len(loads) == 1, "通過が無い時間帯にも要約が出ている(L1 の無駄)"


def test_on_hold_and_retry_across_steps(tmp_path):
    """定員超過の個体は駅に留まり、次の整備窓(次 step)で通る = 再試行の口がある。"""
    sim = _sim(tmp_path, "dev_hold", n_steps=2, n_agents=15,
               **{**ON, "world.devices.faregate.service_rate_per_min": 1})
    station = sim.city.station_node
    D.phase(sim, 0, 0)
    dev = D.faregate_of(sim)
    assert dev.capacity == 10
    agents = sim.agents[:12]
    held = [a for a in agents if D.hold_exit(sim, a, 0, 0)]
    assert len(held) == 2, "定員 10 人を超えた 2 人が留め置かれていない"
    for a in held:
        assert a._dev_gate_wait == 1 and a._dev_gate_held == 1
    # 留め置かれた個体を駅に立たせると、次 step の再試行対象として拾われる
    for a in held:
        a.loc, a.node, a.route, a.exit_intent, a.sleeping = \
            "street", station, [], True, False
    assert set(id(a) for a in D.pending_exits(sim, 1)) == set(id(a) for a in held)
    D.phase(sim, 1, 10)                              # 新しい整備窓
    assert all(D.hold_exit(sim, a, 1, 10) is False for a in held)


def test_on_envfeedback_gate_rule_is_not_double_applied(tmp_path):
    """★装置 ON では envfeedback 規則2 を**評価しない**(状態が 1 も動かない)。

    同じ物理(改札の処理能力の飽和)を粗い近似と細粒度モデルの両方で数えない、という
    優先順位の機械固定。規則1(遅延)・規則3(POI)は別の物理なので touch しない。
    """
    envon = {"env.feedback.enabled": "true",
             "env.feedback.gate.enabled": "true",
             "env.feedback.gate.capacity_per_min": 0.001}   # 必ず飽和する閾値

    def _fire(name, **ov):
        sim = _sim(tmp_path, name, n_steps=1, **{**envon, **ov})
        station = sim.city.station_node
        since = len(sim.logger.events)
        from society.observer.schema import Event
        for aid in range(5):                          # 駅への流入 5 件(> 容量)
            sim.logger.log(Event(step=0, sim_min=0, agent_id=aid, kind="arrive",
                                 x=0.0, y=0.0, payload={"node": station,
                                                        "name": "駅"}))
        ENV.update(sim, 0, 0, since)
        return sim, ENV.state(sim)

    # (1) 装置 OFF: 従来どおり規則2 が発動する(検収の空回り防止)
    _sim_off, st_off = _fire("dev_env_off")
    assert st_off["gate_until"] > 0 and st_off["n_gate"] == 1
    # (2) 装置 ON: 規則2 の状態が 1 も動かない
    sim_on, st_on = _fire("dev_env_on", **ON)
    assert D.faregate_active(sim_on) is True
    assert st_on["gate_until"] == -1 and st_on["gate_cool_until"] == -1
    assert st_on["n_gate"] == 0 and st_on["gate_flag"] is False
    assert ENV.gate_active(sim_on, 0) is False
    # 規則1(遅延)は別の物理なので生きている
    assert st_on["n_transit"] >= 0 and "delay_min" in st_on


def test_scheduler_seams_are_gated_and_minimal():
    """scheduler の seam は「gated な 3 行」だけ(_apply には 1 バイトも触れない)。"""
    src = (REPO / "src" / "society" / "engine" / "scheduler.py").read_text(
        encoding="utf-8")
    assert src.count("devices_mod.") == 3        # phase / hold_exit / pending_exits の 3 箇所だけ
    assert "from .. import devices as devices_mod" in src
    assert "devices_mod.phase(sim, step, sim_min)" in src
    assert "if via_station and devices_mod.hold_exit(sim, agent, step, sim_min):" in src
    assert "for _held in devices_mod.pending_exits(sim, step):" in src


# =========================================================================== #
# (F) 歩行者信号 = 記憶を持たない固定スケジュール装置(検収基準 ⑤)
# =========================================================================== #
def test_signal_gate_timing_math_is_untouched():
    """``crossing_id`` を付けても位相・現示は 1 ビットも変わらない(既存テストの補強)。"""
    plain = _sfm.SignalGate(140, 37, 10, offset_s=13.0)
    named = _sfm.SignalGate(140, 37, 10, offset_s=13.0, crossing_id=7)
    for t in (0.0, 12.3, 36.9, 47.0, 139.9, 1000.5, 12345.678):
        assert named.phase(t) == plain.phase(t)
        assert named.is_green(t) == plain.is_green(t)
        assert named.is_flashing(t) == plain.is_flashing(t)
        assert named.can_cross(t) == plain.can_cross(t)


def test_signal_gate_device_id_mapping():
    """交差点表の id → 安定 device_id。既存呼び出し(id 無し)では None のまま。"""
    assert _sfm.SignalGate(140, 37, 10).device_id is None
    assert _sfm.SignalGate(140, 37, 10).crossing_id is None
    got = _sfm.SignalGate(140, 37, 10, crossing_id=7)
    assert got.device_id == D.signal_device_id(7) == "signal:7"
    # ★接頭辞は 2 箇所に書かれている(sfm_core は society 層を import しない)ので
    #   一致を機械固定する
    assert D.SIGNAL_PREFIX == "signal"
    assert _sfm.SignalGate.scramble(140, 37, 10, crossing_id=3).device_id == \
        D.signal_device_id(3)
    assert _sfm.SignalGate.scramble(140, 37, 10).device_id is None


def test_zones_signal_spec_carries_crossing_id():
    """zones テーブルが crossing_id を通す(= SignalGate(**zone.signal) で id が付く)。"""
    got = Z.load_signal({"mode": "explicit", "cycle_s": 140.0, "green_s": 37.0,
                         "flash_s": 10.0, "offset_s": 0.0, "crossing_id": 11},
                        REPO)
    assert got["crossing_id"] == 11
    gate = _sfm.SignalGate(**got)
    assert gate.device_id == D.signal_device_id(11)
    assert gate.cycle_s == 140.0 and gate.green_s == 37.0   # 周期は表のまま
    assert Z.load_signal({"mode": "none"}, REPO) is None


_SIG_ZONE = {"id": "zsig", "engine": "sfm", "dt_sub": 0.05,
             "polygon": [[-25.0, -25.0], [25.0, -25.0], [25.0, 25.0], [-25.0, 25.0]],
             "signal": {"mode": "explicit", "cycle_s": 140.0, "green_s": 37.0,
                        "flash_s": 10.0, "offset_s": 0.0, "crossing_id": 42}}


def test_signal_device_registered_and_is_stateless(tmp_path):
    sim = _sim(tmp_path, "dev_sig", n_steps=1, zone_specs=[_SIG_ZONE],
               **{"world.devices.enabled": "true"})
    D.phase(sim, 0, 0)
    sigs = sim.devices.of_kind("signal")
    assert [s.device_id for s in sigs] == [D.signal_device_id(42)]
    dev = sigs[0]
    assert dev.crossing_id == 42
    assert dev.green_ratio() == round(47.0 / 140.0, 4)
    # 記憶を持たない = δ_int は状態を残さず、応答は絶対時刻の純関数
    assert dev.on_schedule(0) is False
    assert dev.audit()["illegal"] == 0
    assert dev.on_input(1, {"sim_sec": 0.0}, 0)["can_cross"] is True
    assert dev.on_input(1, {"sim_sec": 100.0}, 0)["can_cross"] is False
    assert dev.gate.can_cross(0.0) is True


def test_signal_events_off_emits_nothing(tmp_path):
    sim = _run(tmp_path, "dev_sig_off", n_steps=24, n_agents=10,
               zone_specs=[_SIG_ZONE], **{"world.devices.enabled": "true"})
    assert _kind(sim, "signal_summary") == []


def test_signal_events_on_is_at_most_one_per_crossing_per_day(tmp_path):
    """★1 交差点 1 日に最大 1 件(サイクル 140 秒 < 1 step 600 秒 なので周期毎には出さない)。"""
    sim = _run(tmp_path, "dev_sig_on", n_steps=300, n_agents=8,
               zone_specs=[_SIG_ZONE],
               **{"world.devices.enabled": "true",
                  "world.devices.signal_events": "true"})
    evs = _kind(sim, "signal_summary")
    assert evs, "signal_events ON なのに 1 件も出ていない(検収の空回り)"
    seen = set()
    for e in evs:
        key = (e.payload["device_id"], e.sim_min // 1440)
        assert key not in seen, f"signal_summary が 1 交差点 1 日に 2 件以上: {key}"
        seen.add(key)
        assert e.agent_id == -1
        assert e.payload["crossing_id"] == 42
        assert e.payload["cycle_s"] == 140.0 and e.payload["green_s"] == 37.0
    # 300 step = 2 日境界を跨ぐが、周期(140s)ごとには絶対に出ない
    assert len(evs) <= 3
    assert len(evs) * 140.0 < 300 * 600.0


def test_frozen_metric_spec_files_are_untouched():
    """凍結ファイル群に本バッチの痕跡が 1 つも無い(metrics_spec_hash が動かない)。"""
    from society.observer import metrics_spec as MS
    for rel in MS.SPEC_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "devices" not in text and "gate_pass" not in text, \
            f"凍結ファイル {rel} に装置層の痕跡がある"


assert sys.version_info >= (3, 9)      # dict|None 記法(実装側)の下限
