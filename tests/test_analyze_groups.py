"""analyze_groups.py(群のオントロジー S-c: 群別行動統計)のテスト。

合成イベントで ①災害/ショック窓の反応(潜時・距離・退避)②平常時(訪問エントロピー・会話・発話量)の
集計が正しいことを固定する。ontology 未記録(OFF ラン)は明示終了=捏造回避。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_groups as ag                               # noqa: E402

REPO = _ROOT
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def test_entropy_known_values():
    assert ag.entropy_bits(Counter()) == 0.0
    assert ag.entropy_bits(Counter({"a": 4})) == 0.0                 # 1点分布
    assert ag.entropy_bits(Counter({"a": 1, "b": 1})) == pytest.approx(1.0)  # 一様2点=1bit
    assert ag.entropy_bits(Counter({"c": 2, "d": 1})) == pytest.approx(0.9182958, abs=1e-6)


def _write_run(tmp_path: Path, with_group: bool = True) -> Path:
    run = tmp_path / "r"
    run.mkdir()
    agents = [
        {"id": 0, "name": "A0", **({"ontology_group": "jp_metro"} if with_group else {})},
        {"id": 1, "name": "A1", **({"ontology_group": "jp_metro"} if with_group else {})},
        {"id": 2, "name": "A2", **({"ontology_group": "west_visit"} if with_group else {})},
        {"id": 3, "name": "A3", **({"ontology_group": "west_visit"} if with_group else {})},
    ]
    (run / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                     encoding="utf-8")
    # (step, agent_id, kind, payload)
    rows: list[tuple[int, int, str, dict]] = [
        # --- ショック onset(step10)+ 誤って窓を開かない clear(step30)---
        (10, -1, "disaster", {"kind": "quake", "phase": "onset"}),
        (30, -1, "disaster", {"kind": "quake", "phase": "clear"}),
        # --- 反応窓 [10,15] ---
        # agent0(jp_metro): 潜時2・距離150・圏外退避
        (12, 0, "route_start", {"dest": "x"}),
        (12, 0, "move_segment", {"dist_m": 100.0}),
        (13, 0, "move_segment", {"dist_m": 50.0}),
        (14, 0, "exit_area", {"gateway": "station"}),
        # agent1(jp_metro): 潜時1・距離20・退避なし
        (11, 1, "route_start", {"dest": "y"}),
        (11, 1, "move_segment", {"dist_m": 20.0}),
        # agent2(west_visit): 窓内で活動あるが移動なし(移動率の分母)
        (12, 2, "stay", {"node": "n"}),
        # agent3(west_visit): 潜時4・距離200・屋内→屋外退避
        (14, 3, "move_segment", {"dist_m": 200.0}),
        (14, 3, "exit_building", {"building": "b"}),
        # --- 平常時(窓 [5,15] の外)---
        (20, 0, "arrive", {"node": "A"}),
        (21, 0, "speak", {"text": "こんにちは"}),          # 5字
        (22, 0, "arrive", {"node": "B"}),
        (20, 1, "arrive", {"node": "A"}),                  # 1点=エントロピー0
        (25, 2, "arrive", {"node": "C"}),
        (26, 2, "arrive", {"node": "C"}),
        (27, 2, "arrive", {"node": "D"}),                  # {C:2,D:1}
        (25, 2, "speak", {"text": "やあ"}),                # 2字
        (26, 2, "conversation", {"with": 0}),
    ]
    tbl = pa.table({
        "step": [r[0] for r in rows],
        "agent_id": [r[1] for r in rows],
        "kind": [r[2] for r in rows],
        "payload": [json.dumps(r[3], ensure_ascii=False) for r in rows],
    })
    pq.write_table(tbl, run / "l1_events.parquet")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    return run


def test_shock_window_and_normal_stats(tmp_path):
    run = _write_run(tmp_path)
    rep = ag.analyze(str(run), pre=5, post=5)
    assert rep["n_onsets"] == 1 and rep["onsets"] == [10]       # clear は窓を開かない
    g = rep["groups"]

    # ---- ① 災害窓 ----
    jm = g["jp_metro"]
    assert jm["shock_n_active"] == 2
    assert jm["shock_move_rate"] == pytest.approx(1.0)
    assert jm["shock_latency_mean"] == pytest.approx(1.5)       # (2+1)/2
    assert jm["shock_dist_mean"] == pytest.approx(85.0)         # (150+20)/2
    assert jm["shock_evac_rate"] == pytest.approx(0.5)          # agent0 のみ

    wv = g["west_visit"]
    assert wv["shock_n_active"] == 2                            # agent2(stay)+agent3
    assert wv["shock_move_rate"] == pytest.approx(0.5)          # agent3 のみ移動
    assert wv["shock_latency_mean"] == pytest.approx(4.0)       # agent3 のみ
    assert wv["shock_dist_mean"] == pytest.approx(100.0)        # (0+200)/2
    assert wv["shock_evac_rate"] == pytest.approx(0.5)          # agent3

    # ---- ② 平常時 ----
    assert jm["visit_entropy_mean"] == pytest.approx(0.5)       # (1.0+0.0)/2
    assert jm["speak_count_mean"] == pytest.approx(0.5)         # (1+0)/2
    assert jm["speak_chars_mean"] == pytest.approx(2.5)         # (5+0)/2
    assert wv["visit_entropy_mean"] == pytest.approx(0.9183, abs=1e-4)  # agent2 のみ visit(4dp丸め)
    assert wv["conversation_mean"] == pytest.approx(0.5)        # agent2 の C2 1件


def test_off_run_rejected(tmp_path):
    """ontology 未記録(OFF ラン)は非ゼロ終了(捏造回避)。"""
    run = _write_run(tmp_path, with_group=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts/analyze_groups.py"),
                        str(run)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_ENV)
    assert r.returncode != 0
    assert "対象外" in (r.stdout + r.stderr)


def test_cli_writes_outputs(tmp_path):
    run = _write_run(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "scripts/analyze_groups.py"),
                        str(run), "--shock-pre", "5", "--shock-post", "5"],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_ENV)
    assert r.returncode == 0, r.stderr
    summ = json.loads((run / "groups" / "group_summary.json").read_text(encoding="utf-8"))
    assert set(summ["groups"]) == {"jp_metro", "west_visit"}
    report = (run / "groups" / "group_report.md").read_text(encoding="utf-8")
    assert "災害/ショック窓の反応" in report and "平常時" in report
