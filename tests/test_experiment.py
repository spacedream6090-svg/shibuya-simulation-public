"""対照実験ランナー(scripts/run_experiment.py。第9バッチ)のテスト。

- マニフェストの展開(条件×シード、上書きの合成順、bool の dotlist 化)。
- 小規模 mock 実行で run dir と索引ができる(実LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expand_matrix(tmp_path):
    m = _mod()
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(
        "name: exp\n"
        "seeds: [1, 2]\n"
        "common: { run.n_agents: 5 }\n"
        "conditions:\n"
        "  - name: a\n"
        "    overrides: { recursion.enabled: true }\n"
        "  - name: b\n"
        "    overrides: { k.writeback: 'off' }\n",
        encoding="utf-8")
    manifest = m.load_manifest(manifest_path)
    jobs = m.expand_matrix(manifest)
    assert [j["run"] for j in jobs] == ["exp_a_s1", "exp_a_s2",
                                       "exp_b_s1", "exp_b_s2"]
    assert jobs[0]["overrides"] == ["run.n_agents=5", "recursion.enabled=true",
                                    "run.seed=1", "run.name=exp_a_s1"]
    assert "k.writeback=off" in jobs[2]["overrides"]


def test_small_matrix_runs(tmp_path):
    """2条件×1シードの極小 mock 行列が完走し、条件が config に効いている。"""
    m = _mod()
    sys.path.insert(0, str(REPO / "src"))
    from society.config import load_config
    from society.engine.simulation import Simulation

    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(
        "name: exsmoke\n"
        "seeds: [7]\n"
        "common: { run.n_agents: 8, run.n_steps: 6 }\n"
        "conditions:\n"
        "  - name: base\n"
        "    overrides: {}\n"
        "  - name: rec\n"
        "    overrides: { recursion.enabled: true }\n",
        encoding="utf-8")
    jobs = m.expand_matrix(m.load_manifest(manifest_path))
    results = {}
    for j in jobs:
        cfg = load_config(overrides=j["overrides"])
        sim = Simulation(cfg, out_dir=tmp_path / j["run"])
        sim.run()
        results[j["condition"]] = sim
    assert results["base"].recursion.cfg["enabled"] is False
    assert results["rec"].recursion.cfg["enabled"] is True
    assert (tmp_path / "exsmoke_base_s7").is_dir()
    assert (tmp_path / "exsmoke_rec_s7").is_dir()
    # 実験の索引と同じ形が組めること(ランナー本体の main は CLI なのでここまで)
    index = {"runs": [{k: j[k] for k in ("run", "condition", "seed")}
                      for j in jobs]}
    assert json.dumps(index, ensure_ascii=False)
