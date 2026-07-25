"""第59バッチ スライス(c): 模倣連鎖検出レンズ(scripts/analyze_imitation.py)のテスト。

検証項目:
  (label)  behavior_label が kind+payload を行動 X へ機械分類(spend.cat / venture_open / 除外)。
  (detect) 既知の合成イベント列で 模倣候補(曝露あり初実行)・非曝露初実行を正しく検出。
  (rate)   模倣候補率 + 非曝露ベースライン比(相対リスク>1)の検算。
  (chain)  A→B→C の模倣連鎖を検出。
  (det)    2 回実行で JSON バイト一致(決定論)。
  (cli)    CLI が imitation.json / imitation_report.md を生成。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_imitation as ai                               # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "analyze_imitation.py"


def _write_run(tmp_path: Path, name: str, events: list, n_agents: int = 6) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    agents = [{"id": i, "name": f"住民{i}", "occupation": "会社員"} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["sim_min"]) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0 for _ in events], "y": [0.0 for _ in events],
        "payload": [json.dumps(e.get("payload", {}), ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), run_dir / "l1_events.parquet")
    return run_dir


def _e(day, a, kind, **payload):
    return {"step": day * 144, "agent_id": a, "kind": kind,
            "sim_min": day * 1440 + 100, "payload": payload}


def _spend(day, a):
    return _e(day, a, "spend", cat="food", amount=500)


def _speak(day, a, hearers):
    return _e(day, a, "speak", text="x", hearers=list(hearers))


def test_behavior_label():
    assert ai.behavior_label("spend", {"cat": "food"}) == "spend:food"
    assert ai.behavior_label("spend", {"cat": "taxi"}) is None        # 交通=除外
    assert ai.behavior_label("service_use", {"service": "gym"}) == "service_use:gym"
    assert ai.behavior_label("venture_open", {"name": "x"}) == "venture_open"
    assert ai.behavior_label("arrive", {"node": "n"}) is None         # 対象外 kind


def _imitation_events():
    """0 が spend:food(day0)→ 1 へ接触(day1)→ 1 が初実行(day1)→ 2 へ接触(day2)→ 2 初実行(day2)。
    3 は誰とも接触せず day5 に初実行(非曝露ベースライン)。"""
    return [
        _spend(0, 0),                       # 先行実行者
        _speak(1, 0, [1]),                  # 0→1 接触
        _spend(1, 1),                       # 1 初実行(曝露=0 が day0 に実行済み)
        _speak(2, 1, [2]),                  # 1→2 接触
        _spend(2, 2),                       # 2 初実行(曝露=1)。連鎖 0→1→2
        _spend(5, 3),                       # 3 初実行(非曝露=誰とも接触なし)
    ]


def test_detect_and_rate(tmp_path):
    rd = _write_run(tmp_path, "im", _imitation_events())
    res = ai.analyze(str(rd))
    o = res["overall"]
    # 初実行4件(0,1,2,3)・曝露あり2件(1,2)・曝露なし2件(0,3)
    assert o["n_fresh_starts"] == 4
    assert o["exposed_starts"] == 2 and o["unexposed_starts"] == 2
    assert abs(o["imitation_candidate_rate"] - 0.5) < 1e-9
    # 曝露層の初実行率(2/2=1.0)≫ 非曝露層 → 相対リスク>1
    assert o["relative_risk"] is not None and o["relative_risk"] > 1.0
    assert o["exposed_start_rate"] == 1.0
    # 候補辺に (0→1) と (1→2)
    # 行動別: spend:food が対象
    beh = {r["behavior"]: r for r in res["per_behavior"]}
    assert "spend:food" in beh
    assert beh["spend:food"]["exposed_starts"] == 2


def test_chain_detected(tmp_path):
    rd = _write_run(tmp_path, "imchain", _imitation_events())
    res = ai.analyze(str(rd))
    assert res["overall"]["n_chains"] >= 1
    chains = [tuple(c["chain"]) for c in res["chains"] if c["behavior"] == "spend:food"]
    assert (0, 1, 2) in chains


def test_unexposed_baseline_no_contact(tmp_path):
    """接触が全く無いランは曝露0=模倣候補率0(初実行は起きるがベースラインのみ)。"""
    ev = [_spend(0, 0), _spend(3, 1), _spend(6, 2)]      # 接触イベントなし
    rd = _write_run(tmp_path, "imbase", ev)
    res = ai.analyze(str(rd))
    o = res["overall"]
    assert o["exposed_starts"] == 0
    assert o["imitation_candidate_rate"] == 0.0
    assert o["n_chains"] == 0


def test_deterministic(tmp_path):
    rd = _write_run(tmp_path, "imdet", _imitation_events())
    a = json.dumps(ai.analyze(str(rd)), sort_keys=True, ensure_ascii=False)
    b = json.dumps(ai.analyze(str(rd)), sort_keys=True, ensure_ascii=False)
    assert a == b, "imitation 解析が決定論でない(2 回で不一致)"


def test_cli(tmp_path):
    rd = _write_run(tmp_path, "imcli", _imitation_events())
    r = subprocess.run([sys.executable, str(SCRIPT), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert (rd / "imitation.json").exists() and (rd / "imitation_report.md").exists()
    js = json.loads((rd / "imitation.json").read_text(encoding="utf-8"))
    assert js["overall"]["n_chains"] >= 1
