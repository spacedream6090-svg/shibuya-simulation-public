"""第59バッチ スライス(b): 弱い紐帯レンズ(scripts/analyze_weak_ties.py)のテスト。

検証項目:
  (graph)  合成 2 コミュニティ + 既知の橋辺で bridge/internal 辺数・弱い紐帯の強さが期待どおり。
  (comm)   既存の決定論 LPA(measure.communities)で 2 コミュニティに分かれる。
  (grano)  語彙の新規採用の到来辺種別(bridge/internal)が last_from 帰属で正しく突合。
  (broker) per-agent brokerage(接する辺の bridge 率)。
  (det)    2 回実行で JSON バイト一致(決定論)。
  (cli)    CLI が weak_ties.json / weak_ties_report.md を生成。
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

import analyze_weak_ties as awt                              # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "analyze_weak_ties.py"


def _write_run(tmp_path: Path, name: str, events: list, n_agents: int = 6) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    agents = [{"id": i, "name": f"住民{i}", "occupation": "会社員"} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e.get("sim_min", 420 + int(e["step"]) * 10)) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0 for _ in events], "y": [0.0 for _ in events],
        "payload": [json.dumps(e.get("payload", {}), ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), run_dir / "l1_events.parquet")
    return run_dir


def _speak(step, a, hearers, sim_min=None):
    return {"step": step, "agent_id": a, "kind": "speak",
            "sim_min": sim_min if sim_min is not None else 420 + step * 10,
            "payload": {"text": "x", "hearers": list(hearers)}}


def _two_cluster_events():
    """コミュニティ A={0,1,2}・B={3,4,5}(三角で密)+ 弱い橋辺 (0,4)=1回だけ。

    橋辺の両端はこの決定論 LPA が 2 コミュニティに分離する配置(0,4 を採用=探索で確認)。"""
    ev = []
    s = 0
    # 密な内部辺(各クラスタ内を複数回=接触頻度=3)
    for _rep in range(3):
        for a, hs in ((0, [1, 2]), (1, [0, 2]), (2, [0, 1]),
                      (3, [4, 5]), (4, [3, 5]), (5, [3, 4])):
            ev.append(_speak(s, a, hs)); s += 1
    # 弱い橋辺(0,4)を1回だけ(接触頻度=1<内部3)
    ev.append(_speak(s, 0, [4])); s += 1
    return ev, s


def test_communities_and_bridge_edges(tmp_path):
    ev, s = _two_cluster_events()
    # 語彙: 0 が造語 → 1(同コミュ=internal)と 4(別コミュ=bridge)へ伝播 → 双方が採用
    ev += [
        {"step": s, "agent_id": 0, "kind": "vocab_coin", "sim_min": 420 + s * 10,
         "payload": {"item_id": "vocab-1", "text": "ミーム"}},
        {"step": s + 1, "agent_id": 1, "kind": "transmission", "sim_min": 420 + (s + 1) * 10,
         "payload": {"item_id": "vocab-1", "from": 0, "channel": "sns"}},
        {"step": s + 1, "agent_id": 1, "kind": "label_adopt", "sim_min": 420 + (s + 1) * 10,
         "payload": {"item_id": "vocab-1", "text": "ミーム"}},
        {"step": s + 2, "agent_id": 4, "kind": "transmission", "sim_min": 420 + (s + 2) * 10,
         "payload": {"item_id": "vocab-1", "from": 0, "channel": "sns"}},
        {"step": s + 2, "agent_id": 4, "kind": "label_adopt", "sim_min": 420 + (s + 2) * 10,
         "payload": {"item_id": "vocab-1", "text": "ミーム"}},
    ]
    rd = _write_run(tmp_path, "wt", ev)
    res = awt.analyze(str(rd))
    # 2 コミュニティ(各3人)
    assert res["community"]["n_communities"] == 2
    assert res["community"]["sizes"] == [3, 3]
    # 辺: 内部6 + 橋1 = 7
    g = res["graph"]
    assert g["n_edges"] == 7 and g["n_bridge_edges"] == 1 and g["n_internal_edges"] == 6
    # 弱い紐帯の強さ: 橋(頻度1)< 内部(頻度3) → weak_tie_signal True
    assert g["bridge_mean_strength"] < g["internal_mean_strength"]
    assert g["weak_tie_signal"] is True
    # Granovetter: 採用2件・bridge経由1(0→3)・internal経由1(0→1)
    gr = res["granovetter"]
    assert gr["n_adoptions_attributed"] == 2
    assert gr["via_bridge"] == 1 and gr["via_internal"] == 1
    assert abs(gr["bridge_adopt_fraction"] - 0.5) < 1e-9
    # brokerage: agent0 と agent4 が橋渡し(接する辺の bridge 率>0)。橋辺 (0,4)。
    bmap = {b["id"]: b for b in res["brokerage"]}
    assert bmap[0]["n_bridge"] == 1 and bmap[4]["n_bridge"] == 1
    assert bmap[0]["brokerage"] > 0 and bmap[1]["brokerage"] == 0


def test_deterministic(tmp_path):
    ev, _s = _two_cluster_events()
    rd = _write_run(tmp_path, "wtdet", ev)
    a = json.dumps(awt.analyze(str(rd)), sort_keys=True, ensure_ascii=False)
    b = json.dumps(awt.analyze(str(rd)), sort_keys=True, ensure_ascii=False)
    assert a == b, "weak_ties 解析が決定論でない(2 回で不一致)"


def test_graceful_without_conversation(tmp_path):
    ev = [{"step": 0, "agent_id": 0, "kind": "arrive", "sim_min": 420,
           "payload": {"node": "n1"}}]
    rd = _write_run(tmp_path, "wtempty", ev)
    res = awt.analyze(str(rd))
    assert res["graph"]["n_edges"] == 0
    assert res["granovetter"]["n_adoptions_attributed"] == 0
    assert res["granovetter"]["bridge_adopt_fraction"] is None


def test_cli(tmp_path):
    ev, _s = _two_cluster_events()
    rd = _write_run(tmp_path, "wtcli", ev)
    r = subprocess.run([sys.executable, str(SCRIPT), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert (rd / "weak_ties.json").exists() and (rd / "weak_ties_report.md").exists()
    js = json.loads((rd / "weak_ties.json").read_text(encoding="utf-8"))
    assert js["community"]["n_communities"] == 2
