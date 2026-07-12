"""第18バッチ①: 自然発生コミュニティの観察・分析(scripts/analyze_communities.py)と
viz/make_viewer.py のコミュニティ色分けのテスト。

検証項目(タスク検収+修正指示 2026-07-11):
  (a) 同一入力で 2 回実行 → communities.json がバイト一致(決定論)。
  (b) 植え付けた 2 クリークを、正準検出器(louvain seed=0)が 2 コミュニティに
      分離する。弱いエッジ 1 本の橋(合成重み < min_weight)は足切りされ、
      橋があっても 2 つに割れる。min_weight 未満の偶発エッジ(1回きりの hear)は
      無視され、その相手は無所属になる。
  (c) communities.json が無いラン dir ではビューワーが従来と同一 HTML(トークン残らず・
      コミュニティ選択肢なし)。有→無で再生成するとバイト一致=後方互換。
  (d) ライフサイクル分類の単体(合流・分裂の人工ケース)。

合成 L1(parquet)を tmp_path に書いて後処理スクリプトを回す(house style は
Simulation+load_config だが、本機能は「runs/ を読むだけ」の純後処理なので合成 L1 で足りる)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_communities.py"
VIEWER = ROOT / "viz" / "make_viewer.py"


def _load_ac():
    """analyze_communities.py をモジュールとして読み込む(単体関数を直接叩くため)。"""
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("analyze_communities", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ac = _load_ac()


# --------------------------------------------------------------------------- #
# 合成 L1 の生成
# --------------------------------------------------------------------------- #
def _write_run(run_dir: Path, events: list[dict], agents: list[dict] | None = None):
    """events(dict列)を l1_events.parquet として書く。payload は JSON 文字列化。

    load_events が直接参照する列(step/sim_min/agent_id/kind/x/y/payload)を必ず含める。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e.get("sim_min", 420 + int(e["step"]) * 10)) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [float(e.get("x", 0.0)) for e in events],
        "y": [float(e.get("y", 0.0)) for e in events],
        "payload": [json.dumps(e.get("payload", {}), ensure_ascii=False) for e in events],
    }
    table = pa.table(cols)
    pq.write_table(table, run_dir / "l1_events.parquet")
    if agents is not None:
        (run_dir / "agents.json").write_text(
            json.dumps(agents, ensure_ascii=False), encoding="utf-8")


def _clique_speaks(members, step):
    """members 全員が互いに speak する(その step の完全グラフ会話)。"""
    out = []
    for a in members:
        hearers = [h for h in members if h != a]
        out.append({"step": step, "agent_id": a, "kind": "speak",
                    "payload": {"hearers": hearers, "text": "x", "items": []}})
    return out


def _two_clique_run_events(n_windows=2, cross=False):
    """2 クリーク {0,1,2,3} と {4,5,6,7} を各窓で会話させる合成 L1。"""
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    ev: list[dict] = []
    for w in range(n_windows):
        base = w * 144 + 10
        for rep in range(3):
            ev += _clique_speaks(A, base + rep)
            ev += _clique_speaks(B, base + rep)
        if cross:                       # 弱い橋(疎結合)を 1 本だけ
            ev.append({"step": base, "agent_id": 0, "kind": "speak",
                       "payload": {"hearers": [4], "text": "x", "items": []}})
    return ev


_AGENTS8 = [{"id": i, "name": f"住民{i}", "occupation": "会社員",
             "work_building": "bA" if i < 4 else "bB"} for i in range(8)]


# --------------------------------------------------------------------------- #
# (a) 決定論: 2 回実行で communities.json がバイト一致
# --------------------------------------------------------------------------- #
def test_json_byte_identical(tmp_path):
    rd = tmp_path / "run_det"
    _write_run(rd, _two_clique_run_events(n_windows=2), _AGENTS8)

    def run_once() -> bytes:
        subprocess.run([sys.executable, str(SCRIPT), str(rd), "--window-days", "1"],
                       check=True, capture_output=True)
        return (rd / "communities.json").read_bytes()

    b1 = run_once()
    b2 = run_once()
    assert b1 == b2, "同一入力で communities.json がバイト一致しない(決定論が崩れている)"
    obj = json.loads(b1)
    assert obj["run"] == "run_det"
    assert obj["n_windows"] == 2
    # 各窓で 2 クリークが 2 コミュニティとして出ている
    assert all(w["n_communities"] == 2 for w in obj["windows"]), \
        "植え付けた 2 クリークが 2 コミュニティにならない"
    # リーダーの PageRank が実際に計算されている(scipy 不在で 0 に退化しない)
    assert all(c["leader_pagerank"] > 0
               for w in obj["windows"] for c in w["communities"]), \
        "leader_pagerank が 0(PageRank が計算されていない)"


# --------------------------------------------------------------------------- #
# (b) 2 クリーク → 2 コミュニティ(正準 louvain)+ 弱エッジの足切り
# --------------------------------------------------------------------------- #
def test_two_cliques_split():
    # 橋なし: 正準検出器(louvain seed=0)がちょうど 2 コミュニティに分ける
    ev = []
    for g in ([0, 1, 2, 3], [4, 5, 6, 7]):
        for _ in range(3):
            ev += _clique_speaks(g, 0)
    graph = ac.build_window_graph(ev)
    det = ac.detect(graph, 8)
    part = det["part"]
    labels = {part[i] for i in range(8)}
    assert len(labels) == 2, f"2 クリークが 2 コミュニティにならない: {part}"
    assert part[0] == part[1] == part[2] == part[3]
    assert part[4] == part[5] == part[6] == part[7]
    assert part[0] != part[4]
    # community_id はメンバ最小 id に正準化されている
    assert part[0] == 0 and part[4] == 4

    # 弱い橋 1 本(speak 1回 = 合成重み 1.0 < min_weight 2.0)は足切りされ、
    # 橋があっても正準検出器は 2 クリークに割る
    ev_bridge = ev + [{"step": 0, "agent_id": 0, "kind": "speak",
                       "payload": {"hearers": [4]}}]
    g2 = ac.build_window_graph(ev_bridge)
    assert (0, 4) not in g2["W"], "min_weight 未満の橋エッジが足切りされていない"
    p2 = ac.detect(g2, 8)["part"]
    assert p2[0] == p2[1] == p2[2] == p2[3] == 0
    assert p2[4] == p2[5] == p2[6] == p2[7] == 4
    assert p2[0] != p2[4], "弱い橋を足切りしても 2 コミュニティに割れない"


def test_min_weight_prunes_incidental_contact():
    """一回きりの偶発接触(hear 1回 = 1.0 < 2.0)は紐帯にならず、相手は無所属。"""
    ev = []
    for _ in range(3):
        ev += _clique_speaks([0, 1, 2, 3], 0)
    # agent 8 が agent 0 の発話を 1 回だけ耳にした(すれ違い)
    ev.append({"step": 0, "agent_id": 8, "kind": "hear",
               "payload": {"speaker": 0, "items": []}})
    graph = ac.build_window_graph(ev)
    assert (0, 8) not in graph["W"], "偶発エッジが足切りされていない"
    assert 8 not in graph["nodes"], "紐帯ゼロの agent 8 がグラフに残っている"
    part = ac.detect(graph, 9)["part"]
    assert 8 not in part, "無所属のはずの agent 8 がコミュニティに入っている"
    assert part[0] == part[1] == part[2] == part[3] == 0

    # min_weight を下げれば同じ接触が紐帯になる(オプションの動作確認)
    g_low = ac.build_window_graph(ev, min_weight=1.0)
    assert (0, 8) in g_low["W"], "--min-weight 1.0 で偶発エッジが紐帯にならない"


# --------------------------------------------------------------------------- #
# (c) communities.json 無し → ビューワー従来同一 / 有無トグルでバイト一致(後方互換)
# --------------------------------------------------------------------------- #
def _minimal_map(path: Path):
    city = {
        "buildings": [{"id": "b1", "name": "テストビル", "kind": "generic",
                       "levels": 3, "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 10, "y": 10}],
        "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [10, 10]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def test_viewer_backward_compat(tmp_path):
    rd = tmp_path / "run_viz"
    _write_run(rd, _two_clique_run_events(n_windows=2), _AGENTS8)
    map_path = tmp_path / "map.json"
    _minimal_map(map_path)
    # transit.file は存在しないパスにして外部依存を切る(build_data は skip する)
    (rd / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n",
        encoding="utf-8")

    def gen_viewer() -> str:
        subprocess.run([sys.executable, str(VIEWER), str(rd)],
                       check=True, capture_output=True)
        return hashlib.sha256((rd / "viewer.html").read_bytes()).hexdigest()

    # communities.json 無し: 従来同一(トークン残らず・コミュニティ選択肢なし)
    h_no = gen_viewer()
    html_no = (rd / "viewer.html").read_text(encoding="utf-8")
    assert "__COMMUNITY_" not in html_no, "置換されなかったトークンが残っている"
    assert 'value="community"' not in html_no, "communities.json 無しなのに選択肢が出た"
    assert "function communityColor" not in html_no

    # communities.json 有り: 選択肢と色分け JS が入る
    subprocess.run([sys.executable, str(SCRIPT), str(rd), "--window-days", "1"],
                   check=True, capture_output=True)
    gen_viewer()
    html_yes = (rd / "viewer.html").read_text(encoding="utf-8")
    assert 'value="community"' in html_yes, "communities.json 有りなのに選択肢が出ない"
    assert "function communityColor" in html_yes
    assert '"communities":' in html_yes, "communities データが埋め込まれていない"

    # 再び communities.json を消す → 従来出力にバイト一致(後方互換の合格条件)
    (rd / "communities.json").unlink()
    h_no2 = gen_viewer()
    assert h_no2 == h_no, "communities.json 除去後にビューワーが元に戻らない(後方互換違反)"


# --------------------------------------------------------------------------- #
# (d) ライフサイクル分類の単体(合流・分裂)
# --------------------------------------------------------------------------- #
def test_lifecycle_merge_and_split():
    # 合流: 窓0 の 2 コミュニティ → 窓1 の 1 コミュニティ
    did, role, events = ac.track_lifecycle(
        [{0: {0, 1, 2}, 3: {3, 4, 5}}, {0: {0, 1, 2, 3, 4, 5}}])
    assert role[(1, 0)]["role"] == "merge", "合流が merge に分類されない"
    merges = [e for e in events if e["type"] == "merge"]
    assert merges and sorted(merges[0]["from_communities"]) == [0, 3], \
        f"合流イベントの元コミュニティが不正: {merges}"

    # 分裂: 窓0 の 1 コミュニティ → 窓1 の 2 コミュニティ
    did2, role2, events2 = ac.track_lifecycle(
        [{0: {0, 1, 2, 3, 4, 5}}, {0: {0, 1, 2}, 3: {3, 4, 5}}])
    splits = [e for e in events2 if e["type"] == "split"]
    assert splits, "分裂が split イベントとして検出されない"
    assert splits[0]["from_community"] == 0
    assert sorted(splits[0]["into_communities"]) == [0, 3]

    # 誕生/消滅: マッチしない窓遷移
    _, role3, events3 = ac.track_lifecycle([{0: {0, 1, 2}}, {9: {9, 10, 11}}])
    types = sorted(e["type"] for e in events3)
    assert "birth" in types and "death" in types, f"誕生/消滅が出ない: {types}"
