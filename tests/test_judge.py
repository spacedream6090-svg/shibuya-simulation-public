"""scripts/judge.py の最小テスト(mock バックエンド・決定論)。

* mock で judge_run が judge.json を出し、Fleiss κ が [-1,1] か nan に収まる。
* ドシエ構築が空ラン(イベント0)でも安全(例外なし・空 dict)。
* Fleiss κ の数値が既知例と一致する(numpy 自作の健全性)。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO = Path(__file__).resolve().parents[1]

# scripts/judge.py をモジュールとして読み込む(scripts はパッケージではない)
_spec = importlib.util.spec_from_file_location("judge", _REPO / "scripts" / "judge.py")
judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge)


def _write_run(dir_: Path, events: list[dict], agents: list[dict]) -> None:
    """最小の l1_events.parquet + agents.json を書く(logger と同じ列)。"""
    dir_.mkdir(parents=True, exist_ok=True)
    n = len(events)
    tbl = pa.table({
        "step": pa.array([e["step"] for e in events], pa.int32()),
        "sim_min": pa.array([0] * n, pa.int32()),
        "agent_id": pa.array([e["agent_id"] for e in events], pa.int32()),
        "kind": pa.array([e["kind"] for e in events], pa.string()),
        "x": pa.array([0.0] * n, pa.float32()),
        "y": pa.array([0.0] * n, pa.float32()),
        "payload": pa.array([json.dumps(e.get("payload", {}), ensure_ascii=False)
                             for e in events], pa.string()),
        "rng_stream": pa.array([""] * n, pa.string()),
        "llm_call_id": pa.array([None] * n, pa.string()),
    })
    pq.write_table(tbl, dir_ / "l1_events.parquet", compression="zstd")
    (dir_ / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                      encoding="utf-8")


def _sample_events() -> list[dict]:
    # agent 0: 世界改変っぽい(提案・結成・造語・SNS)/ agent 1: 雑談のみ / agent 2: 無活動
    return [
        {"step": 1, "agent_id": 0, "kind": "speak", "payload": {"text": "この街を変えたい", "hearers": [1]}},
        {"step": 3, "agent_id": 0, "kind": "vocab_coin", "payload": {"item_id": "v1", "text": "モヤり現象"}},
        {"step": 5, "agent_id": 0, "kind": "proposal", "payload": {"proposal_id": "p1", "text": "混雑を直そう"}},
        {"step": 6, "agent_id": 0, "kind": "group_found", "payload": {"group_id": "g1", "name": "渋谷の会", "purpose": "つながる"}},
        {"step": 7, "agent_id": 0, "kind": "sns_post", "payload": {"text": "みんな集まろう", "items": ["v1"]}},
        {"step": 2, "agent_id": 1, "kind": "speak", "payload": {"text": "今日は暑いね", "hearers": [0]}},
        {"step": 4, "agent_id": 1, "kind": "dm", "payload": {"to": 0, "text": "どこにいる?"}},
    ]


def test_judge_run_mock_writes_json_and_kappa_in_range(tmp_path):
    run = tmp_path / "run"
    agents = [{"id": 0, "name": "A", "occupation": "会社員"},
              {"id": 1, "name": "B", "occupation": "大学生"},
              {"id": 2, "name": "C", "occupation": "無職"}]
    _write_run(run, _sample_events(), agents)

    result = judge.judge_run(str(run), backend="mock", judges=3)

    jpath = run / "analysis" / "judge.json"
    assert jpath.exists()
    assert (run / "analysis" / "judge_report.md").exists()
    saved = json.loads(jpath.read_text(encoding="utf-8"))

    assert saved["n_agents"] == 3
    assert saved["backend"] == "mock"
    k = saved["fleiss_kappa"]
    assert k is None or (-1.0 - 1e-9 <= k <= 1.0 + 1e-9)
    assert saved["kappa_adopt_threshold"] == 0.7
    # 各 agent は judges 回だけ投票し votes が bool
    for a in result["agents"]:
        assert len(a["votes"]) == 3
        assert all(isinstance(v, bool) for v in a["votes"])


def test_judge_run_is_deterministic(tmp_path):
    run = tmp_path / "run"
    agents = [{"id": i, "name": f"A{i}", "occupation": "x"} for i in range(4)]
    _write_run(run, _sample_events(), agents)
    r1 = judge.judge_run(str(run), backend="mock", judges=3)
    r2 = judge.judge_run(str(run), backend="mock", judges=3)
    assert [a["votes"] for a in r1["agents"]] == [a["votes"] for a in r2["agents"]]
    assert r1["fleiss_kappa"] == r2["fleiss_kappa"]


def test_build_dossiers_empty_run_is_safe(tmp_path):
    run = tmp_path / "empty"
    agents = [{"id": 0, "name": "A", "occupation": "x"}]
    _write_run(run, [], agents)  # イベント0
    import society.observer.measure as m
    events = m.load_events(str(run))
    dossiers = judge.build_dossiers(events, agents)
    assert dossiers == {0: ""}
    # judge_run 全体も空ランで壊れない
    result = judge.judge_run(str(run), backend="mock", judges=3)
    assert result["n_agents"] == 1
    k = result["fleiss_kappa"]
    assert k is None or (-1.0 <= k <= 1.0)


def test_build_dossiers_truncates(tmp_path):
    events = [{"step": i, "agent_id": 0, "kind": "speak",
               "payload": {"text": "あ" * 50, "hearers": [1]}} for i in range(100)]
    dossiers = judge.build_dossiers(events, [{"id": 0}], max_chars=300)
    assert len(dossiers[0]) <= 320  # 上限 + 省略記号の余白
    assert "以下略" in dossiers[0]


def test_fleiss_kappa_known_values():
    # 完全一致(全評者が同じ item を category1、別 item を category0)→ κ = 1
    perfect = np.array([[0, 3], [3, 0], [0, 3], [3, 0]], dtype=float)
    assert judge.fleiss_kappa(perfect) == pytest.approx(1.0, abs=1e-9)
    # 全 item・全評者が同一 category → 期待一致=1 → 退化 nan
    degen = np.array([[0, 3], [0, 3], [0, 3]], dtype=float)
    assert math.isnan(judge.fleiss_kappa(degen))
    # 評者<2 → nan
    assert math.isnan(judge.fleiss_kappa(np.array([[1, 0], [0, 1]], dtype=float)))
    # 一般値は [-1, 1]
    mix = np.array([[2, 1], [1, 2], [3, 0], [0, 3], [2, 1]], dtype=float)
    k = judge.fleiss_kappa(mix)
    assert -1.0 <= k <= 1.0
