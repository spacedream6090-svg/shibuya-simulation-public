"""⑥介入前後の比較の純関数テスト(第31バッチ・分析スイート W2)。

scripts/compare_runs.py の between-run ペア比較(セルフチェック=差ゼロ・p≈1 / 作為ペア=
検出)・CRN 突合の分岐検出・DiD・データ不足の縮退を、合成 parquet で検証する。
統計は panel_stats の純関数を流用するので、ここでは compare_runs の組み立てを検証する。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import pyarrow as pa           # noqa: E402
import pyarrow.parquet as pq   # noqa: E402

import compare_runs as cr      # noqa: E402


def _write_l2(run_dir: str, metrics: dict[str, list]) -> None:
    """l2_metrics.parquet を合成(step 列 + 指定の数値列)。各列の step 平均が run_scalars。"""
    os.makedirs(run_dir, exist_ok=True)
    n = len(next(iter(metrics.values())))
    cols = {"step": list(range(n))}
    cols.update({k: [float(v) for v in vs] for k, vs in metrics.items()})
    fields = [pa.field("step", pa.int64())] + \
             [pa.field(k, pa.float64()) for k in metrics]
    pq.write_table(pa.Table.from_pydict(cols, schema=pa.schema(fields)),
                   os.path.join(run_dir, "l2_metrics.parquet"))


def _write_l1(run_dir: str, rows: list[tuple]) -> None:
    """l1_events.parquet を合成。rows = (step, agent_id, kind, x, y, payload_dict)。"""
    os.makedirs(run_dir, exist_ok=True)
    cols = {
        "step": [r[0] for r in rows],
        "sim_min": [r[0] * 10 for r in rows],
        "agent_id": [r[1] for r in rows],
        "kind": [r[2] for r in rows],
        "x": [r[3] for r in rows],
        "y": [r[4] for r in rows],
        "payload": [json.dumps(r[5]) if r[5] is not None else None for r in rows],
    }
    schema = pa.schema([
        pa.field("step", pa.int64()), pa.field("sim_min", pa.int64()),
        pa.field("agent_id", pa.int64()), pa.field("kind", pa.string()),
        pa.field("x", pa.float64()), pa.field("y", pa.float64()),
        pa.field("payload", pa.string()),
    ])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                   os.path.join(run_dir, "l1_events.parquet"))


# --------------------------------------------------------------- run_scalars
def test_run_scalars_mean_over_steps(tmp_path):
    d = str(tmp_path / "r_s1")
    _write_l2(d, {"n_working": [10, 12, 14], "mean_grievance": [0.2, 0.2, 0.2]})
    sc = cr.run_scalars(d)
    assert sc["l2.n_working"] == 12.0          # (10+12+14)/3
    assert abs(sc["l2.mean_grievance"] - 0.2) < 1e-9
    assert "l2.step" not in sc                  # step 列は指標にしない


# --------------------------------------------------------------- セルフチェック
def test_self_compare_zero_diff_p1(tmp_path):
    """同一ランを --a と --b(3 seed)に与えたら全差ゼロ・置換 p≈1。"""
    dirs = []
    for s in (1, 2, 3):
        d = str(tmp_path / f"cond_s{s}")
        _write_l2(d, {"n_working": [10 + s, 10 + s]})   # seed で値は変えるが A=B
        dirs.append(d)
    pairs, _notes = cr.pair_runs(dirs, dirs)
    rows, _meta = cr.compare_metrics(pairs)
    row = next(r for r in rows if r["metric"] == "l2.n_working")
    assert row["n_pairs"] == 3
    assert row["mean_diff"] == 0.0
    assert row["cohen_d"] is None              # sd=0 → 効果量は定義されない
    assert abs(row["perm_p"] - 1.0) < 1e-12    # 差ゼロ → p=1


# --------------------------------------------------------------- 検出
def test_contrived_difference_detected(tmp_path):
    """作為的に差を入れたペア(diffs=[2,3,4])→ 平均差・効果量・置換 p を検出。"""
    a_dirs, b_dirs = [], []
    for s, (a, b) in zip((1, 2, 3), [(10, 12), (10, 13), (10, 14)]):
        da = str(tmp_path / f"off_s{s}")
        db = str(tmp_path / f"free_s{s}")
        _write_l2(da, {"n_working": [a, a]})
        _write_l2(db, {"n_working": [b, b]})
        a_dirs.append(da)
        b_dirs.append(db)
    pairs, notes = cr.pair_runs(a_dirs, b_dirs)
    assert any("seed 一致" in n for n in notes)
    rows, _meta = cr.compare_metrics(pairs)
    row = next(r for r in rows if r["metric"] == "l2.n_working")
    assert row["mean_diff"] == 3.0             # (2+3+4)/3
    assert row["cohen_d"] is not None
    # 符号反転の全列挙(2^3=8): |合計/3|=3 は all+ と all- の 2 通り → 0.25
    assert abs(row["perm_p"] - 0.25) < 1e-12


# --------------------------------------------------------------- データ不足
def test_missing_metric_marked_data_short(tmp_path):
    """片側にしか無い指標は「データ不足(比較不能)」として記録される。"""
    da = str(tmp_path / "off_s1")
    db = str(tmp_path / "free_s1")
    _write_l2(da, {"n_working": [10, 10], "only_in_a": [1, 1]})
    _write_l2(db, {"n_working": [12, 12]})
    pairs, _notes = cr.pair_runs([da], [db])
    rows, _meta = cr.compare_metrics(pairs)
    short = next(r for r in rows if r["metric"] == "l2.only_in_a")
    assert short["data_short"] is True
    assert short["n_pairs"] == 0
    assert short["n_missing"] == 1


# --------------------------------------------------------------- CRN 突合
def test_crn_divergence_detected(tmp_path):
    """介入なしの2ランを突合: step0,1 一致・step2 で分岐 → first_divergence=2。"""
    common = [
        (0, 1, "arrive", 10.0, 10.0, {"node": "n1"}),
        (1, 1, "arrive", 20.0, 20.0, {"node": "n2"}),
    ]
    da = str(tmp_path / "a")
    db = str(tmp_path / "b")
    _write_l1(da, common + [(2, 1, "arrive", 30.0, 30.0, {"node": "n3"})])
    _write_l1(db, common + [(2, 1, "arrive", 99.0, 99.0, {"node": "nX"})])  # 分岐
    h = cr.crn_health(da, db, crn_max_steps=100)
    assert h["compared_steps"] == 3
    assert h["matched_steps"] == 2
    assert h["first_divergence"] == 2
    assert h["intact"] is False
    assert abs(h["match_rate"] - 2 / 3) < 1e-12


def test_crn_intact_identical_runs(tmp_path):
    """同一 L1 を突合 → 分岐なし・一致率100%(CRN 保持)。"""
    rows = [(0, 1, "arrive", 10.0, 10.0, {"node": "n1"}),
            (1, 2, "move_segment", 20.0, 20.0, {})]
    da = str(tmp_path / "a")
    db = str(tmp_path / "b")
    _write_l1(da, rows)
    _write_l1(db, rows)
    h = cr.crn_health(da, db, crn_max_steps=100)
    assert h["first_divergence"] is None
    assert h["intact"] is True
    assert h["match_rate"] == 1.0


# --------------------------------------------------------------- DiD
def test_did_effect_pure():
    # (post_t - pre_t) - (post_c - pre_c) = (3-1) - (2-1) = 1
    assert cr.did_effect([1, 1], [3, 3], [1, 1], [2, 2]) == 1.0
    # 群が空 → None(算出不能)
    assert cr.did_effect([], [3], [1], [2]) is None


def test_did_within_skips_without_shock(tmp_path):
    """scenario_shock が無いランは DiD を明示スキップ(0 を捏造しない)。"""
    d = str(tmp_path / "noshock")
    _write_l1(d, [(0, 1, "arrive", 10.0, 10.0, {"node": "n1"})])
    res = cr.did_within(d)
    assert res["ok"] is False
    assert "scenario_shock" in res["reason"]


def test_did_within_computes_with_shock(tmp_path):
    """shock_closure(center/radius)のあるランで空間処置群を作り DiD を算出。"""
    d = str(tmp_path / "shock")
    rows = [
        # 介入宣言(step 5・中心(0,0)・半径 50m)
        (5, -1, "scenario_shock", 0.0, 0.0,
         {"kind": "shock_closure", "at": 5, "phase": "start",
          "center": [0.0, 0.0], "radius_m": 50.0}),
        # 処置 agent 1(中心近傍): 介入前 2 件・介入後 0 件(移動が減る)
        (0, 1, "move_segment", 5.0, 5.0, {}),
        (1, 1, "move_segment", 5.0, 5.0, {}),
        # 対照 agent 2(遠方 500m): 介入前 1 件・介入後 1 件(不変)
        (0, 2, "move_segment", 500.0, 500.0, {}),
        (6, 2, "move_segment", 500.0, 500.0, {}),
    ]
    _write_l1(d, rows)
    res = cr.did_within(d, metric_kind="move_segment")
    assert res["ok"] is True
    assert res["at"] == 5
    assert res["n_treat"] == 1 and res["n_ctrl"] == 1
    # 処置(0-2=-2) - 対照(1-1=0) = -2
    assert res["did"] == -2.0
