"""感情・興味・注意ハブ(affect、既定 OFF)のテスト。

設計正典: docs/lit/neuroscience__emotion-interest-attention.md §5.1(最小版=3改造)/§6(統合)。
検証の柱(既存の鉄則を継承):
- OFF 既定 == 純粋既定の L1 完全一致(144 step)。**最重要**(バイト一致)。
- arousal 単体: ネガ/新奇イベントで↑、無入力で baseline へ減衰。
- importance 加点単体 / salience 上位 K ゲート単体 / 発火閾値の逆U字単体。
- 決定論(同 ON 設定2回=L1 一致)。ON で L1 が変わる(=配線が効いている)。
- R1 呼数不変: FixedLLM で ON(threshold_gain=0)/OFF の generate 呼数が一致(salience ゲートは
  発火数を変えない)。
- mock 全 ON 完走 + arousal が動く。実LLM 禁止(mock のみ)。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from society.agents.memory import MemoryStore
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors import affect
from society.factors import update as factor_update
from society.world.perception import salience_gate


# --------------------------------------------------------------------------- helpers
def _sim(tmp_path, name: str, n: int = 20, steps: int = 24, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=12"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim) -> list:
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _on_ov(threshold_gain: float = 0.05) -> dict:
    """全 ON の dotlist(推奨 ON 値ベース)。"""
    return {"affect.enabled": "true", "affect.arousal_gain": 0.15,
            "affect.importance_arousal_w": 2.0, "affect.importance_novelty_w": 1.5,
            "affect.salience_k": 4, "affect.threshold_gain": threshold_gain}


# ============================================================ 1. cfg / 恒等の核
def test_affect_cfg_default_off():
    assert affect.build_cfg({})["enabled"] is False
    assert affect.build_cfg(None)["enabled"] is False
    c = affect.build_cfg({"enabled": True})
    assert c["enabled"] is True
    assert c["arousal_baseline"] == 0.2 and c["arousal_gain"] == 0.0
    assert c["salience_k"] == 0 and c["threshold_gain"] == 0.0


# ============================================================ 2. arousal 単体(§1/§5.1)
def test_arousal_signal_and_bump():
    cfg = affect.build_cfg({"enabled": True, "arousal_gain": 0.2})
    sig = affect.arousal_signal(valence_abs=0.8, novelty=1.0, addressed=1.0)
    assert abs(sig - 2.8) < 1e-9
    assert affect.bump(0.2, sig, cfg) > 0.2          # 相動性上昇
    # gain=0 は恒等(バイト一致)
    z = affect.build_cfg({"enabled": True, "arousal_gain": 0.0})
    assert affect.bump(0.2, sig, z) == 0.2
    # clip [0,1]
    assert affect.bump(0.95, 10.0, cfg) == 1.0


def test_arousal_decays_to_baseline():
    cfg = affect.build_cfg({"enabled": True, "arousal_decay": 0.3,
                            "arousal_baseline": 0.2})
    ag = SimpleNamespace(arousal=0.9)
    affect.decay(ag, cfg)
    assert 0.2 < ag.arousal < 0.9                    # baseline へ寄る
    for _ in range(80):                              # 無入力の連続 step
        affect.decay(ag, cfg)
    assert abs(ag.arousal - 0.2) < 1e-3              # baseline へ収束


def test_on_arousal_noop_when_gain_zero():
    """gain=0(または enabled=false)で on_arousal は no-op(arousal 不変・ログなし)=バイト一致。"""
    logs = []
    logger = SimpleNamespace(log=lambda e: logs.append(e))
    ag = SimpleNamespace(arousal=0.2, x=0.0, y=0.0, id=1)
    cfg = affect.build_cfg({"enabled": True, "arousal_gain": 0.0})
    d = factor_update.on_arousal(ag, cfg, valence_abs=1.0, cause="x", step=0,
                                 sim_min=0, logger=logger)
    assert d == 0.0 and ag.arousal == 0.2 and not logs
    # enabled=false でも同様
    off = affect.build_cfg({"enabled": False, "arousal_gain": 0.5})
    factor_update.on_arousal(ag, off, valence_abs=1.0, cause="x", step=0,
                             sim_min=0, logger=logger)
    assert ag.arousal == 0.2 and not logs


def test_on_arousal_rises_and_logs():
    logs = []
    logger = SimpleNamespace(log=lambda e: logs.append(e))
    ag = SimpleNamespace(arousal=0.2, x=1.0, y=2.0, id=3)
    cfg = affect.build_cfg({"enabled": True, "arousal_gain": 0.2})
    d = factor_update.on_arousal(ag, cfg, valence_abs=0.8, novelty=1.0,
                                 cause="heard", step=5, sim_min=50, logger=logger)
    assert d > 0.0 and ag.arousal > 0.2
    assert len(logs) == 1 and logs[0].kind == "affect_update"
    assert logs[0].payload["cause"] == "heard"


# ============================================================ 3. importance 加点(§1.6/§2.3)
def test_memory_importance_bonus():
    m = MemoryStore()
    m.observe(0, "既定")                              # bonus=0 → 3.0(現行と一致)
    assert m.buffer[-1].importance == 3.0
    m.observe(0, "高覚醒", importance_bonus=2.5)
    assert m.buffer[-1].importance == 5.5
    m.observe(0, "clip 上限", importance_bonus=100.0)  # clip 1–10
    assert m.buffer[-1].importance == 10.0


def test_importance_bonus_formula():
    cfg = affect.build_cfg({"enabled": True, "importance_arousal_w": 2.0,
                            "importance_novelty_w": 1.5})
    ag = SimpleNamespace(arousal=0.5)
    got = affect.importance_bonus(ag, cfg, novelty=1.0)
    assert abs(got - (2.0 * 0.5 + 1.5 * 1.0)) < 1e-9
    zero = affect.build_cfg({"enabled": True})       # w=0 → 0(固定 3.0 のまま)
    assert affect.importance_bonus(ag, zero, novelty=1.0) == 0.0


# ============================================================ 4. salience 上位 K ゲート(§3)
def test_salience_gate_topk_and_order():
    items = ["a", "b", "c", "d"]
    scores = [0.1, 0.9, 0.5, 0.2]
    assert salience_gate(items, scores, 0) == items          # K=0(∞)=全通し・順序保持
    assert salience_gate(items, scores, 10) == items         # K>=N=全通し
    assert salience_gate(items, scores, 2) == ["b", "c"]     # 上位2(元順序を保つ)
    # 同点は index 昇順(決定論)
    assert salience_gate(["x", "y", "z"], [0.5, 0.5, 0.1], 1) == ["x"]


def test_salience_score_bottomup():
    cfg = affect.build_cfg({"enabled": True})
    ag = SimpleNamespace(arousal=0.2)
    strong = affect.salience_score(ag, cfg, valence_abs=0.9, addressed=1.0)
    weak = affect.salience_score(ag, cfg, valence_abs=0.1)
    assert strong > weak                              # 強 valence + 自分宛て=顕著


# ============================================================ 5. 発火閾値の逆U字(§6.3)
def test_threshold_delta_inverted_u():
    z = affect.build_cfg({"enabled": True, "threshold_gain": 0.0})
    assert affect.threshold_delta(SimpleNamespace(arousal=0.5), z) == 0.0  # gain=0 恒等
    cfg = affect.build_cfg({"enabled": True, "threshold_gain": 0.1})
    mid = affect.threshold_delta(SimpleNamespace(arousal=0.5), cfg)
    hi = affect.threshold_delta(SimpleNamespace(arousal=1.0), cfg)
    lo = affect.threshold_delta(SimpleNamespace(arousal=0.0), cfg)
    assert mid < 0.0                                  # 中覚醒=閾値↓(行動しやすい)
    assert hi > 0.0 and lo > 0.0                      # 過/低覚醒=閾値↑(狭窄/退屈)


# ============================================================ 6. OFF: バイト一致(最重要)
def test_off_matches_pure_default(tmp_path):
    """affect.enabled=false なら、他パラメータが非ゼロでも純粋既定と L1 完全一致(144 step)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "off_params", steps=144,
               **{"affect.enabled": "false", "affect.arousal_gain": 0.15,
                  "affect.importance_arousal_w": 2.0,
                  "affect.importance_novelty_w": 1.5,
                  "affect.salience_k": 4, "affect.threshold_gain": 0.05})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(affect seam が no-op でない)"
    assert not [e for e in pure.logger.events if e.kind == "affect_update"]


# ============================================================ 7. 決定論 / ON で L1 が変わる
def test_affect_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", steps=48, **_on_ov()); a.run()
    b = _sim(tmp_path, "det_b", steps=48, **_on_ov()); b.run()
    assert _l1(a) == _l1(b), "affect ON が同 seed で非決定論"


def test_affect_on_changes_l1(tmp_path):
    off = _sim(tmp_path, "chg_off", steps=48); off.run()
    on = _sim(tmp_path, "chg_on", steps=48, **_on_ov()); on.run()
    assert _l1(off) != _l1(on), "affect ON で L1 が変わらない(効いていない)"
    assert [e for e in on.logger.events if e.kind == "affect_update"], \
        "affect_update が 1 件も出ていない(arousal が動いていない)"


# ============================================================ 8. R1: 呼数不変(salience ゲート)
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, ov, steps=30, n=20, seed=42) -> int:
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"] + [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim.llm.calls


def test_salience_gate_does_not_change_call_count(tmp_path):
    """FixedLLM: affect ON(threshold_gain=0)と OFF で generate 呼数が完全一致。

    salience ゲートは「知覚する item の集合(記憶書込・プロンプト材料)」だけを変え、
    発火判断(drive ゲージ)には触れない → 呼数は不変(R1)。threshold_gain=0 に固定して
    覚醒の逆U字が発火を動かす経路を切り、ゲートの呼数中立性だけを検証する。"""
    off = _run_fixed(tmp_path, "cc_off", {})
    on = _run_fixed(tmp_path, "cc_on", _on_ov(threshold_gain=0.0))
    assert off == on and off > 0, (off, on)


# ============================================================ 9. mock 全 ON 完走 + arousal 可動
def test_mock_full_on_smoke(tmp_path):
    sim = _sim(tmp_path, "smoke", n=30, steps=144, **_on_ov())
    summary = sim.run()
    assert summary["n_steps"] == 144
    au = [e for e in sim.logger.events if e.kind == "affect_update"]
    assert au, "affect_update が出ていない(arousal が動いていない)"
    # 覚醒が上下する(new>old と new<old の両方=相動性上昇と漏れ減衰の両立)
    ups = [e for e in au if e.payload["new"] > e.payload["old"]]
    assert ups, "arousal の上昇イベントが無い"
