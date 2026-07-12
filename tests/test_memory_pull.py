"""agentic pull 記憶(ユーザー採用決定)。「日常=push / 内省のみ pull」に忠実。

- 内省を固定2段(recall→本内省)にし、内省1回につき recall 呼び出しを1本足す。
  writeback 条件に依らず一定(R1: 呼数を k と不変に保つ)。
- 発火時(deliberate)は LLM を増やさず build_prompt 内で決定論 pull を1行注入するのみ。
- mem.query は決定論・非LLM(retrieve と同じ採点式)。
"""
from __future__ import annotations

from society.agents.memory import MemoryStore
from society.config import load_config
from society.engine.simulation import Simulation


def _run(tmp_path, name, n_agents=12, n_steps=144, **overrides):
    dot = [f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "observer.snapshot_every=60"]
    dot += [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _purpose_count(sim, purpose):
    return sum(1 for r in sim.logger.llm_calls if r.get("purpose") == purpose)


def test_agentic_pull_adds_one_recall_call_per_reflect(tmp_path):
    """agentic_pull=true: 内省1回につき recall 呼び出しが1本(全 writeback で等量)。"""
    for wb in ("free", "sham", "degraded"):
        sim = _run(tmp_path, f"ap_{wb}", **{"memory.agentic_pull": "true",
                                            "k.writeback": wb})
        n_reflect = _purpose_count(sim, "reflect")
        n_recall = _purpose_count(sim, "recall")
        assert n_reflect > 0, f"{wb}: 内省が発生していない"
        assert n_recall == n_reflect, \
            f"{wb}: recall({n_recall}) != reflect({n_reflect})"
    # compute_matched の off も 2 呼(recall+reflect)になる
    cm = _run(tmp_path, "ap_cmoff", **{"memory.agentic_pull": "true",
                                       "controls.mode": "compute_matched",
                                       "k.writeback": "off"})
    assert _purpose_count(cm, "recall") == _purpose_count(cm, "reflect") > 0


def test_agentic_pull_logs_memory_recall_events(tmp_path):
    sim = _run(tmp_path, "ap_ev", **{"memory.agentic_pull": "true"})
    recalls = [e for e in sim.logger.events if e.kind == "memory_recall"]
    assert recalls, "memory_recall イベントが出ていない"
    for e in recalls:
        assert "query" in e.payload and "n_hits" in e.payload
    # pull 無効時は memory_recall が一切出ない(現状不変)
    base = _run(tmp_path, "ap_base", **{"memory.agentic_pull": "false"})
    assert not any(e.kind == "memory_recall" for e in base.logger.events)


def test_query_returns_relevant_memory():
    """mem.query が自由文クエリから関連記憶を返す(決定論・非LLM)。"""
    m = MemoryStore()
    m.observe(0, "駅前でコーヒーを飲んだ")
    m.consolidate(10, "初日", [("宮下公園で花音と長話した", 8.0),
                               ("道玄坂の混雑にうんざりした", 6.0)])
    got = m.query(20, "宮下公園 花音 のこと", n=2)
    assert got and "宮下公園" in got[0]
    got2 = m.query(20, "道玄坂 混雑", n=1)
    assert got2 and "道玄坂" in got2[0]
    # クエリが空なら空(想起の暴発を防ぐ)
    assert m.query(20, "", n=2) == []


def test_fire_count_stable_across_k_with_pull(tmp_path):
    """pull 有効時も k∈{free,off} で発火数が ±20%(既存 test_drive の流儀)。"""
    fires = {}
    for mode in ("free", "off"):
        sim = _run(tmp_path, f"apk_{mode}", n_agents=20, n_steps=60,
                   **{"memory.agentic_pull": "true", "k.writeback": mode})
        fires[mode] = sum(1 for e in sim.logger.events
                          if e.kind == "drive_request" and e.payload["granted"])
    lo, hi = sorted(fires.values())
    assert lo > 0
    assert hi <= lo * 1.2 + 3, f"pull 有効で k 条件間の発火が乖離: {fires}"
