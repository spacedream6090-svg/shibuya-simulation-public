"""興味・訪問の観測 CLI(scripts/observe.py。第9バッチ Wave O)のテスト。

小さい mock ラン(20体×48step)を回して observe() がクラッシュせず、visits/interests に
非自明な内容(訪問記録・関心プロファイル)が入ることを確認する。実LLM 禁止。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _observe_mod():
    spec = importlib.util.spec_from_file_location(
        "observe", REPO / "scripts" / "observe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_observe_on_small_run(tmp_path):
    from society.config import load_config
    from society.engine.simulation import Simulation

    run_dir = tmp_path / "obs_run"
    cfg = load_config(overrides=["run.n_agents=20", "run.n_steps=48",
                                 "run.name=obs_run"])
    sim = Simulation(cfg, out_dir=run_dir)
    sim.run()

    mod = _observe_mod()
    out = Path(mod.observe(str(run_dir)))
    for fname in ("visits.json", "interests.json", "poi_ranking.csv", "summary.md"):
        assert (out / fname).exists(), f"{fname} が生成されていない"

    visits = json.loads((out / "visits.json").read_text(encoding="utf-8"))
    interests = json.loads((out / "interests.json").read_text(encoding="utf-8"))
    # 訪問: 何らかの場所記録がある(mock 48step でも移動・建物入りは起きる)
    assert visits, "visits.json が空"
    blob = json.dumps(visits, ensure_ascii=False)
    assert ("arrive" in blob) or ("visit" in blob) or ("node" in blob), \
        "visits に訪問らしき内容が無い"
    # 関心: エージェント別プロファイルが人数ぶんある(0件項目は0と明記される設計)
    assert interests, "interests.json が空"
    iblob = json.dumps(interests, ensure_ascii=False)
    assert "drive" in iblob or "reason" in iblob or "interest" in iblob or \
        "profile" in iblob, "interests に関心プロファイルらしき内容が無い"

    md = (out / "summary.md").read_text(encoding="utf-8")
    assert len(md) > 200 and "訪問" in md or "関心" in md, "summary.md が薄すぎる"
