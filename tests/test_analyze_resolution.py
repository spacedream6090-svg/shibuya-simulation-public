"""analyze_resolution.py(I3・入力解像度の効果分離)のテスト。"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_resolution import distinct_n, entropy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_distinct_n_known_values():
    # "ああああ" → bigram 延べ3・種類1
    assert distinct_n(["ああああ"], 2) == pytest.approx(1 / 3)
    # 全て異なる bigram
    assert distinct_n(["あいう"], 2) == pytest.approx(1.0)
    assert distinct_n([], 2) == 0.0
    assert distinct_n([""], 3) == 0.0


def test_entropy_known_values():
    assert entropy(Counter()) == 0.0
    assert entropy(Counter({"a": 4})) == 0.0                    # 1点分布
    assert entropy(Counter({"a": 1, "b": 1})) == pytest.approx(1.0)  # 一様2点=1bit


def _write_run(tmp_path: Path, with_level: bool) -> Path:
    run = tmp_path / "r"
    run.mkdir()
    agents = [{"id": 0, "name": "A", **({"input_res": "narrow"} if with_level else {})},
              {"id": 1, "name": "B", **({"input_res": "wide"} if with_level else {})}]
    (run / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                     encoding="utf-8")
    rows = {
        "step": [1, 2, 3, 150, 4],
        "agent_id": [0, 0, 1, 1, 1],
        "kind": ["speak", "fallback", "arrive", "arrive", "llm_deliberate"],
        "payload": [json.dumps({"text": "こんにちは"}),
                    json.dumps({"reason": "parse"}),
                    json.dumps({"name": "ハチ公前広場"}),
                    json.dumps({"name": "駅前カフェ"}),
                    json.dumps({"trigger": "novel_place"})],
    }
    pq.write_table(pa.Table.from_pydict(rows), run / "l1_events.parquet")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    return run


def test_off_run_is_rejected(tmp_path):
    """input_res 未記録(OFFラン)は明示終了=捏造回避。"""
    run = _write_run(tmp_path, with_level=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts/analyze_resolution.py"),
                        str(run)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "対象外" in (r.stdout + r.stderr)


def test_level_separation_and_outputs(tmp_path):
    run = _write_run(tmp_path, with_level=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts/analyze_resolution.py"),
                        str(run)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    tbl = pq.read_table(run / "panel" / "resolution_agents.parquet").to_pylist()
    by_id = {row["agent_id"]: row for row in tbl}
    assert by_id[0]["level"] == "narrow" and by_id[1]["level"] == "wide"
    # agent0: deliberate 0 + fallback 1 → fallback_rate 1.0 / agent1: 1+0 → 0.0
    assert by_id[0]["fallback_rate"] == pytest.approx(1.0)
    assert by_id[1]["fallback_rate"] == pytest.approx(0.0)
    # agent1 の訪問先2種(2日にまたがる)→ レパートリー曲線 {0:1, 1:2}
    assert json.loads(by_id[1]["repertoire_curve"]) == {"0": 1, "1": 2}
    report = (run / "panel" / "resolution_report.md").read_text(encoding="utf-8")
    assert "K1" in report and "データ不足: 水準 mid" in report
