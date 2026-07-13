"""分析スイート W3(拡張2本)の純関数テスト(第31バッチ)。

対象(すべて読み出し専用の後処理・シミュ実行不要):
  scripts/analyze_communities.py … ⑤社会ネットワークの変化(大域指標・tie decay・窓遷移)
  scripts/build_panel.py          … ④活動時間バジェット + tempogram(活動復元)
  scripts/calibrate_report.py     … 分布一致(KS/EMD・1次元自作)

合成データは pyarrow.Table.from_pydict / 素の dict 列で作る(pandas/duckdb は使わない)。
既知の小グラフ・小分布で手計算値との一致を検証し、0件・空データの縮退も1本置く。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import analyze_communities as ac   # noqa: E402
import build_panel as bp           # noqa: E402
import calibrate_report as cr      # noqa: E402


# --------------------------------------------------------------------------- #
# 補助: 合成グラフ / 合成イベント
# --------------------------------------------------------------------------- #
def _graph(edges, w=3.0):
    """(u,v) 列 → build_window_graph 互換の窓グラフ dict(W/adj/nodes)。"""
    W: dict = {}
    adj: dict = defaultdict(set)
    for (u, v) in edges:
        key = (u, v) if u < v else (v, u)
        W[key] = w
        adj[u].add(v)
        adj[v].add(u)
    return {"W": W, "adj": dict(adj), "nodes": set(adj)}


def _ev(kind, aid, step, payload=None):
    return {"kind": kind, "agent_id": aid, "step": step,
            "sim_min": 420 + step * 10, "payload": payload or {}}


# --------------------------------------------------------------------------- #
# ⑤ ネットワーク大域指標(既知の小グラフの手計算一致)
# --------------------------------------------------------------------------- #
def test_network_metrics_known_graph():
    # 三角形 {0,1,2} + 独立辺 {3,4}: N=5, E=4
    g = _graph([(0, 1), (1, 2), (0, 2), (3, 4)])
    mt = ac.network_metrics(g)
    assert mt["n_nodes"] == 5 and mt["n_edges"] == 4
    assert mt["density"] == 0.4                      # 2E/(N(N-1)) = 8/20
    assert mt["mean_degree"] == 1.6                  # 2E/N = 8/5
    assert mt["clustering_coeff"] == 1.0             # 三角の3点=1・次数1は除外
    assert mt["degree_gini"] == 0.15                 # 次数[2,2,2,1,1] の gini
    assert mt["giant_component_frac"] == 0.6         # 最大成分3/5
    assert mt["mean_tie_weight"] == 3.0

    # 空グラフの縮退
    e = ac.network_metrics({"W": {}, "adj": {}, "nodes": set()})
    assert e["n_nodes"] == 0 and e["n_edges"] == 0
    assert e["density"] == 0.0 and e["mean_degree"] == 0.0
    assert e["clustering_coeff"] == 0.0 and e["giant_component_frac"] == 0.0
    assert e["mean_tie_weight"] == 0.0


def test_tie_decay_survival():
    # (0,1): 窓0..2 で連続=寿命3 / (1,2): 窓0のみ=1 / (3,4): 窓2のみ=1
    graphs = [_graph([(0, 1), (1, 2)]), _graph([(0, 1)]),
              _graph([(0, 1), (3, 4)])]
    hist = ac.tie_decay_hist(graphs)
    assert hist == [{"lifespan_windows": 1, "n_ties": 2},
                    {"lifespan_windows": 3, "n_ties": 1}]
    assert ac.tie_decay_hist([]) == []               # 縮退


def test_community_flows_shared_members():
    comm = [{0: {0, 1, 2}, 3: {3, 4, 5}}, {0: {0, 1, 2, 3}, 5: {4, 5}}]
    rows = ac.community_flows(comm)
    got = {(r["window_from"], r["community_from"], r["window_to"],
            r["community_to"], r["n_shared_members"]) for r in rows}
    assert got == {(0, 0, 1, 0, 3),      # {0,1,2}∩{0,1,2,3}=3
                   (0, 3, 1, 0, 1),      # {3,4,5}∩{0,1,2,3}=1
                   (0, 3, 1, 5, 2)}      # {3,4,5}∩{4,5}=2
    assert ac.community_flows([]) == []              # 窓が1つ以下=縮退
    assert ac.community_flows([{0: {0, 1}}]) == []


# --------------------------------------------------------------------------- #
# ④ 活動時間バジェット / tempogram(合成イベントの復元)
# --------------------------------------------------------------------------- #
def _activity_events():
    """agent0(睡眠・移動・飲食)と agent1(自宅・買物)の合成 L1。"""
    ev = []
    # agent0: 就寝 step96→138(sleep 42step=420分)
    ev.append(_ev("sleep_start", 0, 96, {"building": "bh0", "until_step": 138}))
    # agent0: 移動 step0,1,2 + step100(step100 は睡眠区間に入るので sleep が上書き)
    for s in (0, 1, 2, 100):
        ev.append(_ev("move_segment", 0, s, {"dist_m": 100.0}))
    # agent0: 飲食(eating)建物 step10→16(6step=60分)
    ev.append(_ev("enter_building", 0, 10, {"building": "bf", "activity": "eating"}))
    ev.append(_ev("exit_building", 0, 16, {"building": "bf"}))
    # agent1: 自宅(activity 空・home_building 一致)step5→11(6step=60分)
    ev.append(_ev("enter_building", 1, 5, {"building": "bh1", "activity": ""}))
    ev.append(_ev("exit_building", 1, 11, {"building": "bh1"}))
    # agent1: 買物(shopping)step20→23(3step=30分)
    ev.append(_ev("enter_building", 1, 20, {"building": "bs", "activity": "shopping"}))
    ev.append(_ev("exit_building", 1, 23, {"building": "bs"}))
    return ev


def test_time_budget_categories_and_1440_cap():
    agents = [{"id": 0}, {"id": 1, "home_building": "bh1"}]
    recon = bp.reconstruct_activity(_activity_events(), agents)
    tb = bp.build_time_budget(recon, "syn")
    # (agent_id, category) -> minutes_per_day
    mpd = {(tb["agent_id"][i], tb["category"][i]): tb["minutes_per_day"][i]
           for i in range(len(tb["run"]))}
    assert mpd[(0, "sleep")] == 420.0                # 42 step × 10
    assert mpd[(0, "food")] == 60.0                  # 6 step
    assert mpd[(0, "move")] == 30.0                  # 3 step(step100 は sleep に吸収)
    assert mpd[(1, "home")] == 60.0                  # activity 空 × home_building 一致
    assert mpd[(1, "shop")] == 30.0
    # 各エージェントの 分/日 合計は 1440 を超えない(step 単位復元の不変条件)
    per_agent: dict = defaultdict(float)
    for i in range(len(tb["run"])):
        per_agent[tb["agent_id"][i]] += tb["minutes_per_day"][i]
    for aid, tot in per_agent.items():
        assert tot <= 1440.0 + 1e-9, f"agent {aid} の 分/日 合計 {tot} が 1440 超"


def test_tempogram_shares_sum_to_one():
    agents = [{"id": 0}, {"id": 1, "home_building": "bh1"}]
    recon = bp.reconstruct_activity(_activity_events(), agents)
    tg = bp.build_tempogram(recon, "syn")
    by_hour: dict = defaultdict(float)
    for i in range(len(tg["run"])):
        by_hour[tg["hour"][i]] += tg["share"][i]
    assert by_hour, "tempogram が空"
    for h, s in by_hour.items():
        assert abs(s - 1.0) < 1e-9, f"時刻 {h} のシェア合計 {s} が 1 でない"


def test_reconstruct_empty_is_degenerate():
    recon = bp.reconstruct_activity([], [])
    assert recon["adc"] == {} and recon["tempo"] == {}
    assert bp.build_time_budget(recon, "e")["run"] == []
    assert bp.build_tempogram(recon, "e")["run"] == []
    # calibrate 側の要約も「データ不足」に縮退
    summ = cr.activity_summary(recon)
    assert summ["n_all"] == 0 and summ["has_split"] is False


# --------------------------------------------------------------------------- #
# 分布一致(KS / EMD・1次元自作)の既知分布ペア
# --------------------------------------------------------------------------- #
def test_ks_and_emd_known_pairs():
    # 同一分布 → 0
    assert cr.ks_stat([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    assert cr.emd_1d([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    # 完全分離 → KS=1
    assert cr.ks_stat([1, 2, 3, 4], [5, 6, 7, 8]) == 1.0
    # EMD は各質量を 4 だけ動かす=4
    assert abs(cr.emd_1d([1, 2, 3, 4], [5, 6, 7, 8]) - 4.0) < 1e-9
    assert abs(cr.emd_1d([0], [4]) - 4.0) < 1e-9
    # 部分的なズレ: a=[1,1,2], b=[1,2,2] → KS = |2/3−1/3| = 1/3
    assert abs(cr.ks_stat([1, 1, 2], [1, 2, 2]) - 1.0 / 3.0) < 1e-9
    # 空群は None
    assert cr.ks_stat([], [1]) is None
    assert cr.emd_1d([1], []) is None


# --------------------------------------------------------------------------- #
# build_panel.build() の統合(合成 L1 → 2枚の新 parquet が出る + 1440 不変条件)
# --------------------------------------------------------------------------- #
def _write_run(rd: Path, events, agents, n_steps):
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["sim_min"]) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0 for _ in events],
        "y": [0.0 for _ in events],
        "payload": [json.dumps(e["payload"], ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), rd / "l1_events.parquet")
    (rd / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                    encoding="utf-8")
    (rd / "summary.json").write_text(json.dumps({"n_steps": n_steps}),
                                     encoding="utf-8")


def test_build_writes_time_budget_and_tempogram(tmp_path):
    rd = tmp_path / "run_w3"
    _write_run(rd, _activity_events(), [{"id": 0}, {"id": 1, "home_building": "bh1"}],
               n_steps=144)
    info = bp.build(str(rd), warmup_days=0)
    assert info["n_time_budget_rows"] > 0 and info["n_tempogram_rows"] > 0

    tb = pq.read_table(rd / "panel" / "time_budget.parquet")
    assert set(tb.column_names) == {"run", "agent_id", "category",
                                    "minutes_total", "n_days_active",
                                    "minutes_per_day"}
    tg = pq.read_table(rd / "panel" / "tempogram.parquet")
    assert set(tg.column_names) == {"run", "hour", "category",
                                    "present_count", "share"}
    # 既存の agent_day / run_day も従来どおり出ている(後方互換=追加のみ)
    assert (rd / "panel" / "agent_day.parquet").exists()
    assert (rd / "panel" / "run_day.parquet").exists()
