"""D0 非定常性診断 CLI(scripts/diagnose_stationarity.py)の単体テスト。

方針(既存の analyze 系テストと同流儀 = シミュ実行不要の合成データ・全決定論):
- 合成 L1 parquet を tmp_path に書き、指標・検定・判定を手計算/構成的期待値と突き合わせる。
- 「反復する世界」(毎日まったく同じ行動)と「ドリフトする世界」(訪問先が日ごとにずれる)を
  作り分け、主判定ラベルが STATIONARY_LIKE / NONSTATIONARY に分かれることを固定する。
- クラスタ係数・直径は numpy 高速路と(analyze_weak_ties からコピーした)純 Python 版が
  一致することを固定する。
- 出力 JSON のバイト同一性(決定論)と、シム本体を import していないこと(再生専用の掟)を固定。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import diagnose_stationarity as ds   # noqa: E402

STEPS_PER_DAY = 144


# --------------------------------------------------------------------------- #
# 合成 L1 の生成
# --------------------------------------------------------------------------- #
def _write_l1(path: Path, rows: list) -> None:
    """rows = [(step, agent_id, kind, payload_dict), ...] を L1 スキーマの parquet に書く。"""
    rows = sorted(rows, key=lambda r: r[0])
    tbl = pa.table({
        "step": pa.array([int(r[0]) for r in rows], pa.int32()),
        "sim_min": pa.array([int(r[0]) * 10 for r in rows], pa.int32()),
        "agent_id": pa.array([int(r[1]) for r in rows], pa.int32()),
        "kind": pa.array([str(r[2]) for r in rows], pa.string()),
        "x": pa.array([0.0] * len(rows), pa.float32()),
        "y": pa.array([0.0] * len(rows), pa.float32()),
        "payload": pa.array([json.dumps(r[3], ensure_ascii=False, sort_keys=True)
                             for r in rows], pa.string()),
        "rng_stream": pa.array([""] * len(rows), pa.string()),
        "llm_call_id": pa.array([""] * len(rows), pa.string()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path)


def _make_run(run_dir: Path, *, n_agents: int, n_days: int, nodes_of,
              n_speak=lambda d: 1, truncate_last_day: bool = False,
              coin_of=lambda a, d: None) -> Path:
    """合成ラン(1 エージェント 1 日 = 到着 6 件 + 発話 n_speak 件 + state_update 1 件)。

    nodes_of(a, d) -> その日にそのエージェントが到着するノード名のリスト。"""
    rows: list = []
    visited: dict = {}
    for d in range(n_days):
        base = d * STEPS_PER_DAY
        rows.append((base, -1, "traffic_flow", {}))            # 日の頭(span 用・agent 非依存)
        for a in range(n_agents):
            rows.append((base + 1, a, "state_update", {"name": "x"}))
            for j, node in enumerate(nodes_of(a, d)):
                first = node not in visited.setdefault(a, set())
                visited[a].add(node)
                rows.append((base + 5 + j, a, "arrive",
                             {"node": node, "first_visit": first, "name": node}))
            for s in range(n_speak(d)):
                hearer = (a + 1 + s) % n_agents
                if hearer != a:
                    rows.append((base + 100 + s, a, "speak", {"hearers": [hearer],
                                                              "text": "t"}))
            item = coin_of(a, d)
            if item is not None:
                rows.append((base + 110, a, "label_coin", {"item_id": item, "text": item}))
                rows.append((base + 110, a, "vocab_coin", {"item_id": item, "text": item}))
                rows.append((base + 111, a, "vocab_use", {"item_id": item}))
        if not (truncate_last_day and d == n_days - 1):
            rows.append((base + STEPS_PER_DAY - 1, -1, "traffic_flow", {}))   # 日の尻
    _write_l1(run_dir / "l1_events.parquet", rows)
    return run_dir


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #
def test_entropy_and_jaccard():
    assert ds._entropy_bits([]) == 0.0
    assert ds._entropy_bits([5]) == 0.0
    assert ds._entropy_bits([1, 1]) == pytest.approx(1.0)
    assert ds._entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)
    assert ds._jaccard(set(), set()) == 1.0
    assert ds._jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert ds._jaccard({1, 2}, {1, 2}) == 1.0


# --------------------------------------------------------------------------- #
# グラフ位相: numpy 高速路 == 純 Python(analyze_weak_ties からのコピー)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_clustering_and_diameter_numpy_matches_python(seed):
    rng = np.random.default_rng(seed)
    n = 18
    edges = {}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.22:
                edges[(u, v)] = 1
    if not edges:
        pytest.skip("空グラフ")
    nodes = sorted({x for e in edges for x in e})
    idx = {u: i for i, u in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for (u, v) in edges:
        A[idx[u], idx[v]] = True
        A[idx[v], idx[u]] = True
    py = ds.clustering_and_diameter_py(edges)
    npv = ds.clustering_and_diameter_np(A)
    assert py["clustering_coeff"] == pytest.approx(npv["clustering_coeff"], abs=1e-6)
    assert py["clustering_n_nodes"] == npv["clustering_n_nodes"]
    assert py["lcc_size"] == npv["lcc_size"]
    assert py["diameter_lcc"] == npv["diameter_lcc"]


def test_clustering_triangle_and_path():
    tri = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
    r = ds.clustering_and_diameter_py(tri)
    assert r["clustering_coeff"] == pytest.approx(1.0)
    assert r["diameter_lcc"] == 1 and r["lcc_size"] == 3
    path = {(0, 1): 1, (1, 2): 1, (2, 3): 1}
    r2 = ds.clustering_and_diameter_py(path)
    assert r2["clustering_coeff"] == pytest.approx(0.0)
    assert r2["diameter_lcc"] == 3 and r2["lcc_size"] == 4


# --------------------------------------------------------------------------- #
# 置換検定
# --------------------------------------------------------------------------- #
def test_signflip_exhaustive_small_n():
    D = np.array([[0.4, -0.4], [0.4, -0.4], [0.4, -0.4], [0.4, -0.4], [0.4, -0.4]])
    r = ds.paired_signflip_tvd(D)
    assert r["method"] == "exhaustive"
    assert r["n_perm"] == 2 ** 5
    assert r["p"] == pytest.approx(2.0 / 32)          # 全符号一致 = 理論下限
    assert r["min_p"] == pytest.approx(2.0 / 32)
    assert r["tvd"] == pytest.approx(0.4)


def test_signflip_degenerate_zero():
    r = ds.paired_signflip_tvd(np.zeros((30, 4)))
    assert r["method"] == "degenerate_zero" and r["p"] == 1.0 and r["tvd"] == 0.0


def test_signflip_monte_carlo_deterministic():
    rng = np.random.default_rng(7)
    D = rng.normal(0.0, 1.0, size=(40, 5)) + 0.9
    a = ds.paired_signflip_tvd(D, n_mc=2000)
    b = ds.paired_signflip_tvd(D, n_mc=2000)
    assert a == b                                     # default_rng(0) 固定 = 再現
    assert a["method"] == "monte_carlo" and a["n_perm"] == 2000
    assert a["p"] < 0.05                              # 明確な平均シフトは検出される
    null = ds.paired_signflip_tvd(rng.normal(0.0, 1.0, size=(40, 5)) * 0.0 + 0.0,
                                  n_mc=500)
    assert null["p"] == 1.0                           # 差ゼロ = 有意にならない


# --------------------------------------------------------------------------- #
# 指標(語彙・グラフ・行動)
# --------------------------------------------------------------------------- #
def test_daily_metrics_from_synthetic_payloads(tmp_path):
    run = _make_run(tmp_path / "r_metrics", n_agents=6, n_days=4,
                    nodes_of=lambda a, d: [f"n{(a + j) % 5}" for j in range(3)],
                    coin_of=lambda a, d: f"w{d}" if a == 0 else None)
    res = ds.analyze(str(run), mc=200)
    daily = res["daily"]
    assert res["meta"]["n_days"] == 4
    assert res["meta"]["n_agents_total"] == 6
    # 語彙: 1 日 1 語ずつ造語 → 新語 1 / 新規性 1.0(使われる語はその日の 1 語のみ)
    for d in range(4):
        assert daily[d]["vocab"]["new_items_coined"] == 1
        assert daily[d]["vocab"]["items_used"] == 1
        assert daily[d]["vocab"]["novelty_share"] == 1.0
    # 行動: day0 は全部初訪問 = 回帰率 0、以降は既訪問のみ = 回帰率 1
    assert daily[0]["behavior"]["revisit_rate"] == pytest.approx(0.0)
    for d in range(1, 4):
        assert daily[d]["behavior"]["revisit_rate"] == pytest.approx(1.0)
        assert daily[d]["behavior"]["jaccard_prev_day"] == pytest.approx(1.0)
    # 接触グラフ: speak は毎日 (a -> a+1) の同じ 6 辺 → day0 に 6 本、以降 0 本
    assert daily[0]["graph"]["new_edges"] == 6
    assert daily[0]["graph"]["cum_edges"] == 6
    for d in range(1, 4):
        assert daily[d]["graph"]["new_edges"] == 0
        assert daily[d]["graph"]["new_edge_rate"] == pytest.approx(0.0)
    # 6 人のリング = 平均次数 2 / クラスタ係数 0 / 直径 3
    assert daily[3]["graph"]["mean_degree"] == pytest.approx(2.0)
    assert daily[3]["graph"]["clustering_coeff"] == pytest.approx(0.0)
    assert daily[3]["graph"]["diameter_lcc"] == 3


def test_relation_events_counted(tmp_path):
    rows = []
    for d in range(3):
        base = d * STEPS_PER_DAY
        rows.append((base, -1, "traffic_flow", {}))
        for a in range(4):
            rows.append((base + 1, a, "state_update", {}))
            rows.append((base + 2, a, "relation_tier",
                         {"other": (a + 1) % 4, "tier": 1, "count": 2}))
            if d == 2:
                rows.append((base + 3, a, "relation_break",
                             {"other": (a + 1) % 4, "from_tier": 1, "to_tier": 0,
                              "cause": "contact"}))
        rows.append((base + STEPS_PER_DAY - 1, -1, "traffic_flow", {}))
    run = tmp_path / "r_rel"
    _write_l1(run / "l1_events.parquet", rows)
    res = ds.analyze(str(run), mc=100)
    d0, d1, d2 = (r["graph"] for r in res["daily"])
    assert d0["relation_tier_events"] == 4 and d1["relation_tier_events"] == 4
    assert d0["new_relation_pairs"] == 4        # (0,1)(1,2)(2,3)(0,3) が day0 に初出
    assert d1["new_relation_pairs"] == 0
    assert d2["relation_break_events"] == 4


# --------------------------------------------------------------------------- #
# 主判定: 反復する世界 vs ドリフトする世界
# --------------------------------------------------------------------------- #
def _stationary_run(tmp_path):
    return _make_run(tmp_path / "r_stat", n_agents=20, n_days=8,
                     nodes_of=lambda a, d: [f"n{(a * 7 + j) % 30}" for j in range(6)])


def _drifting_run(tmp_path):
    """半分は個人固定・半分は全員共通で日ごとにずれるノード群(= 集団として一貫したドリフト)。

    1 日 6 ノード均等なので visit ブロックの TVD は lag1 で 1/6、lag≥3 で 3/6。
    kind ヒストグラムは不変なので combined(= 0.5·kind + 0.5·visit)はその半分になる。"""
    return _make_run(
        tmp_path / "r_drift", n_agents=20, n_days=8,
        nodes_of=lambda a, d: [f"a{a}_{j}" for j in range(3)]
        + [f"g{(d + j) % 30}" for j in range(3)])


def test_stationary_world_is_not_distinguishable(tmp_path):
    res = ds.analyze(str(_stationary_run(tmp_path)), mc=2000)
    prim = [t for t in res["self_similarity"]["tests"]
            if t["block"] == "combined" and (t["day_a"], t["day_b"]) == (2, 5)][0]
    assert prim["tvd"] == pytest.approx(0.0, abs=1e-9)
    assert prim["p"] == 1.0
    assert res["verdict"]["label"] == "STATIONARY_LIKE"
    assert res["verdict"]["primary_pass"] is False
    # 毎日同じ場所 → 前日 Jaccard 1.0 / 回帰率 1.0(day0 以外)
    assert res["daily"][5]["behavior"]["jaccard_prev_day"] == pytest.approx(1.0)
    assert res["daily"][5]["behavior"]["revisit_rate"] == pytest.approx(1.0)


def test_drifting_world_is_distinguishable(tmp_path):
    res = ds.analyze(str(_drifting_run(tmp_path)), mc=2000)
    prim = [t for t in res["self_similarity"]["tests"]
            if t["block"] == "combined" and (t["day_a"], t["day_b"]) == (2, 5)][0]
    assert prim["p"] < 0.05
    assert prim["tvd"] >= ds.DRAFT_THRESHOLDS["tvd_min"]
    assert prim["tvd"] == pytest.approx(0.25, abs=1e-6)     # 0.5 · (3/6)
    assert res["verdict"]["label"] == "SUSTAINED_NONSTATIONARY"
    assert res["verdict"]["primary_pass"] is True
    assert res["verdict"]["sustained_pass"] is True
    # lag が伸びるほど TVD が増える(ドリフトの指紋)
    tvd = res["self_similarity"]["tvd"]["combined"]
    assert tvd["by_lag"]["3"]["mean"] > tvd["by_lag"]["1"]["mean"]
    assert tvd["adjacent_median"] == pytest.approx(0.5 / 6, abs=1e-6)
    assert tvd["lag_slope"] > 0
    # kind ヒストグラムは日によらず一定 = kind ブロックだけなら差なし
    kind = [t for t in res["self_similarity"]["tests"]
            if t["block"] == "kind" and (t["day_a"], t["day_b"]) == (2, 5)][0]
    assert kind["tvd"] == pytest.approx(0.0, abs=1e-9)


def test_transient_only_is_separated_from_sustained(tmp_path):
    """立ち上がりだけドリフトし、以降は隣接日程度しか動かない世界。

    daily300_100d で実際に観測された形(burn-in 後は定常)の最小再現。主判定 (2,5) は
    通るが、同じ lag を後期で取り直した (16,19) は通らない → TRANSIENT_ONLY。"""
    def _off(d):
        return 3 * d if d < 6 else 15 + (d % 2)

    run = _make_run(
        tmp_path / "r_trans", n_agents=20, n_days=20,
        nodes_of=lambda a, d: [f"a{a}_{j}" for j in range(3)]
        + [f"g{_off(d) + j}" for j in range(3)])
    res = ds.analyze(str(run), mc=2000)
    vd = res["verdict"]
    assert vd["primary_pass"] is True
    assert vd["sustained_pass"] is False
    assert vd["label"] == "TRANSIENT_ONLY"
    assert vd["late_same_lag"]["late_same_lag_pair"] == [16, 19]
    prim = [t for t in res["self_similarity"]["tests"]
            if t["block"] == "combined" and (t["day_a"], t["day_b"]) == (2, 5)][0]
    assert prim["tvd"] == pytest.approx(0.25, abs=1e-6)      # 完全に別の場所へ
    assert vd["late_same_lag"]["tvd"] == pytest.approx(0.5 / 6, abs=1e-6)   # 隣接日と同程度


def test_late_same_lag_pair_reported(tmp_path):
    res = ds.analyze(str(_drifting_run(tmp_path)), mc=500)
    lsl = res["verdict"]["late_same_lag"]
    a, b = lsl["late_same_lag_pair"]
    assert b - a == 3                       # 主判定 (2,5) と同じ lag
    assert a >= 2 and b <= 7
    assert lsl["tvd"] is not None and lsl["p"] is not None


# --------------------------------------------------------------------------- #
# 部分日の除外
# --------------------------------------------------------------------------- #
def test_partial_last_day_flagged_and_excluded(tmp_path):
    run = _make_run(tmp_path / "r_partial", n_agents=10, n_days=6,
                    nodes_of=lambda a, d: [f"n{(a + j) % 9}" for j in range(4)],
                    truncate_last_day=True)
    res = ds.analyze(str(run), mc=200)
    flags = [r["full_day"] for r in res["daily"]]
    assert flags[:5] == [True] * 5
    assert flags[5] is False
    assert res["meta"]["n_full_days"] == 5
    assert res["meta"]["full_days"] == [0, 1, 2, 3, 4]
    # ノイズ床(隣接ペア)は完全日どうしの 4 ペアのみ
    assert len(res["self_similarity"]["tvd"]["combined"]["adjacent"]) == 4
    # 既定ペアに部分日は入らない
    for p in res["params"]["pairs"]:
        assert 5 not in p
    assert any("部分日" in n for n in res["notes"])


# --------------------------------------------------------------------------- #
# 決定論・出力・疎結合の掟
# --------------------------------------------------------------------------- #
def test_output_is_byte_identical_across_calls(tmp_path):
    run = _drifting_run(tmp_path)
    a = json.dumps(ds.analyze(str(run), mc=1000), sort_keys=True, ensure_ascii=False)
    b = json.dumps(ds.analyze(str(run), mc=1000), sort_keys=True, ensure_ascii=False)
    assert a == b


def test_cli_writes_json_and_md(tmp_path):
    run = _drifting_run(tmp_path)
    rc = ds.main([str(run), "--mc", "500"])
    assert rc == 0
    js = run / "diagnose_stationarity.json"
    md = run / "diagnose_stationarity.md"
    assert js.is_file() and md.is_file()
    res = json.loads(js.read_text(encoding="utf-8"))
    assert res["run"] == run.name
    assert res["verdict"]["status"].startswith("DRAFT")
    text = md.read_text(encoding="utf-8")
    assert "ステータス: ドラフト判定" in text
    assert "日次自己相似性" in text
    assert "lag 構造" in text


def test_custom_pairs_and_batching_do_not_change_metrics(tmp_path):
    run = _drifting_run(tmp_path)
    a = ds.analyze(str(run), mc=200, batch_rows=17)      # 極小バッチでストリーム境界を跨がせる
    b = ds.analyze(str(run), mc=200, batch_rows=1 << 20)  # 1 バッチ
    assert a["daily"] == b["daily"]
    assert a["meta"] == b["meta"]
    c = ds.analyze(str(run), mc=200, pairs=[(1, 4)])
    assert c["params"]["pairs"] == [[1, 4]]
    assert {(t["day_a"], t["day_b"]) for t in c["self_similarity"]["tests"]} == {(1, 4)}


def test_script_does_not_import_simulation_core():
    """再生専用の掟(実査 §8.4): L1 を読むだけ = シム本体(society.*)を import しない。"""
    src = (_ROOT / "scripts" / "diagnose_stationarity.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        assert not s.startswith("import society")
        assert not s.startswith("from society")
    assert "load_events(" not in src           # 全件 RAM 展開の measure 資産は呼ばない
    assert "iter_batches" in src               # ストリーム走査であること


def test_missing_parquet_raises(tmp_path):
    empty = tmp_path / "nope"
    empty.mkdir()
    with pytest.raises(SystemExit):
        ds.analyze(str(empty))
