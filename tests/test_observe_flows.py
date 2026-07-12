"""お金の移動 + 注意ネットワークの観測 CLI(scripts/observe_flows.py。タスク3)のテスト。

小さい mock ラン(10体×144step)を回して observe() がクラッシュせず、money_flows.json /
attention.json が生成され、金流(wage 総額)・注意(エッジ)に非自明な内容が入ることを確認する。
実LLM 禁止(mock backend)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _flows_mod():
    spec = importlib.util.spec_from_file_location(
        "observe_flows", REPO / "scripts" / "observe_flows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_run(tmp_path):
    from society.config import load_config
    from society.engine.simulation import Simulation

    run_dir = tmp_path / "flows_run"
    cfg = load_config(overrides=["run.n_agents=10", "run.n_steps=144",
                                 "run.name=flows_run"])
    sim = Simulation(cfg, out_dir=run_dir)
    sim.run()
    return run_dir


def test_observe_flows_on_small_run(tmp_path):
    run_dir = _small_run(tmp_path)
    mod = _flows_mod()
    out = Path(mod.observe(str(run_dir)))

    for fname in ("money_flows.json", "attention.json", "money_flows_links.csv",
                  "flows_summary.md", "attention_summary.md"):
        assert (out / fname).exists(), f"{fname} が生成されていない"

    # ---- 金流の骨格 ----
    money = json.loads((out / "money_flows.json").read_text(encoding="utf-8"))
    assert money["by_event_kind"].get("wage", {}).get("amount", 0) > 0, \
        "wage 総額が 0(mock 144step でも勤務完遂の賃金は出るはず)"
    assert money["totals"]["n_edges"] > 0, "金流エッジが 0"
    assert money["totals"]["n_agent_nodes"] > 0, "agent ノードが 0"
    # 誰が金を集めたかランキングが agent_id を持つ
    assert any(a.get("agent_id") is not None
               for a in money["top_money_collectors"])
    # 内部移動(withdraw)と供託金(deposit)の別掲が構造として存在
    assert "withdraw" in money["internal_moves"]
    assert "deposit_political" in money

    # ---- 注意ネットワークの骨格 ----
    att = json.loads((out / "attention.json").read_text(encoding="utf-8"))
    assert att["totals"]["n_edges"] > 0, "注意エッジが 0(hear/sns 等があるはず)"
    assert att["totals"]["total_attention"] > 0
    assert 0.0 <= att["totals"]["attention_in_gini"] <= 1.0
    assert att["per_agent"], "per_agent が空"
    # チャネル別カウントに実データがある
    assert sum(att["channel_counts"].values()) > 0

    # ---- サマリ md が薄すぎない ----
    md = (out / "flows_summary.md").read_text(encoding="utf-8")
    assert len(md) > 200 and "金" in md
    amd = (out / "attention_summary.md").read_text(encoding="utf-8")
    assert len(amd) > 200 and "注意" in amd


def test_observe_flows_window_days(tmp_path):
    """--window-days 相当(末尾N日限定)を渡しても壊れず、範囲情報が出る。"""
    run_dir = _small_run(tmp_path)
    mod = _flows_mod()
    out = Path(mod.observe(str(run_dir), out_dir=str(tmp_path / "win"),
                           window_days=1))
    att = json.loads((out / "attention.json").read_text(encoding="utf-8"))
    assert att["window_days"] == 1
    assert att["day_range"]["max_day"] >= 0
