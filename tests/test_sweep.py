"""バッチC: 掃引/FSS 解析(scripts/analyze_sweep.py)の軽量テスト。

合成ランdir を tmp に作り、analyze_sweep が
  - (N, k) 表(sweep_table.json)
  - fss.json(k*(N) 雛形・N 水準数の妥当性)
  - EWS/FSS の図・report
を壊れず出すこと、旧形式ラン名(N トークン無し)でも回帰しないことを確かめる。
"""
from __future__ import annotations

import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import analyze_sweep as A  # noqa: E402


# --------------------------------------------------------------------------- #
def _events(n_agents, n_steps, seed):
    """採用が seed 依存の順で進む合成イベント列(adoption_frac が step で増える)。"""
    rows = []

    def add(step, aid, kind, **payload):
        rows.append({"step": step, "sim_min": step, "agent_id": aid,
                     "kind": kind, "x": 0.0, "y": 0.0, "rng_stream": "",
                     "llm_call_id": None, "payload": json.dumps(payload)})

    add(0, 0, "vocab_coin", item_id="L", text="foo")
    adopters = list(range(1, n_agents))
    shift = seed % max(len(adopters), 1)
    order = adopters[shift:] + adopters[:shift]
    for i, aid in enumerate(order):
        s = 1 + int(i * (n_steps - 2) / max(len(order), 1))
        add(s, aid, "transmission", item_id="L", **{"from": 0}, channel="face")
        add(s, aid, "label_adopt", item_id="L", text="foo")
    for s in range(n_steps):
        add(s, 0, "vocab_use", item_id="L")
    add(n_steps - 1, 0, "reflect", mode="free", written_back=True, belief="x")
    return rows


def _write_run(base, name, n_agents, seed, writeback, alpha=0.5, n_steps=12):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    cfg = (
        "run:\n"
        f"  seed: {seed}\n"
        f"  n_agents: {n_agents}\n"
        f"  n_steps: {n_steps}\n"
        f"  name: {name}\n"
        "k:\n"
        f"  writeback: {writeback}\n"
        f"  degraded_alpha: {alpha}\n"
        "  reflect_period_days: 1\n"
    )
    with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(cfg)
    rows = _events(n_agents, n_steps, seed)
    tbl = pa.table({
        "step": pa.array([r["step"] for r in rows], pa.int32()),
        "sim_min": pa.array([r["sim_min"] for r in rows], pa.int32()),
        "agent_id": pa.array([r["agent_id"] for r in rows], pa.int32()),
        "kind": pa.array([r["kind"] for r in rows]),
        "x": pa.array([r["x"] for r in rows], pa.float64()),
        "y": pa.array([r["y"] for r in rows], pa.float64()),
        "rng_stream": pa.array([r["rng_stream"] for r in rows]),
        "llm_call_id": pa.array([r["llm_call_id"] for r in rows], pa.int32()),
        "payload": pa.array([r["payload"] for r in rows]),
    })
    pq.write_table(tbl, os.path.join(d, "l1_events.parquet"))
    # traits(R^2 の入力。無くても解析は動くが列を出す)
    agents = [{"id": a, "name": f"a{a}",
               "t1": round((a * 7 % 10) / 10.0, 2),
               "t2": round((a * 3 % 10) / 10.0, 2)} for a in range(n_agents)]
    with open(os.path.join(d, "agents.json"), "w", encoding="utf-8") as fh:
        json.dump(agents, fh)
    return d


def _grid(base, Ns, conds, seeds, prefix, multi_n=True):
    dirs = []
    for n in Ns:
        for label, wb, alpha in conds:
            for s in seeds:
                name = (f"{prefix}_N{n}_{label}_s{s}" if multi_n
                        else f"{prefix}_{label}_s{s}")
                dirs.append(_write_run(base, name, n, s, wb, alpha or 0.5))
    return dirs


# --------------------------------------------------------------------------- #
def test_analyze_sweep_nk_table_and_fss(tmp_path):
    base = str(tmp_path / "runs")
    conds = [("off", "off", None), ("free", "free", None)]
    dirs = _grid(base, [10, 20], conds, [1, 2], "c_fss")
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))

    for fn in ("sweep_table.json", "seed_divergence.json", "fss.json",
               "report_sweep.md", "fig_r2_by_k.png", "fig_ews_by_k.png",
               "fig_divergence.png", "fig_fires_by_k.png", "fig_fss.png"):
        assert os.path.isfile(os.path.join(out, fn)), fn

    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    assert st["N_levels"] == [10, 20]
    # 各 N に off/free の (N,k) セルがある
    for N in ("10", "20"):
        cells = st["table"][N]
        assert set(cells) == {"off", "free"}
        assert cells["off"]["k_eff"] == 0.0
        assert cells["free"]["k_eff"] == 1.0
        # EWS 列が入っている(三角測量の3本目)
        assert "EWS_var" in cells["off"]
        assert cells["off"]["n_seeds"] == 2

    with open(os.path.join(out, "fss.json"), encoding="utf-8") as fh:
        fss = json.load(fh)
    assert fss["n_levels"] == 2
    assert fss["sufficient"] is False              # 2<3 → 外挿しない
    assert set(fss["by_N"]) == {"10", "20"}
    for N in ("10", "20"):
        f = fss["by_N"][N]
        for key in ("kstar_r2_gradient", "kstar_divergence_jump",
                    "kstar_ews_peak", "k_grid"):
            assert key in f
        assert f["k_grid"] == [0.0, 1.0]           # off, free の連続 k 軸


def test_degraded_alpha_becomes_continuous_axis(tmp_path):
    base = str(tmp_path / "runs")
    conds = [("off", "off", None), ("dg0.5", "degraded", 0.5),
             ("free", "free", None), ("sham", "sham", None)]
    dirs = _grid(base, [10], conds, [1, 2], "c_fss", multi_n=False)
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))
    with open(os.path.join(out, "fss.json"), encoding="utf-8") as fh:
        fss = json.load(fh)
    # 連続 k 軸は off=0 / dg=0.5 / free=1(sham は対照で除外)
    assert fss["by_N"]["10"]["k_grid"] == [0.0, 0.5, 1.0]
    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    assert st["table"]["10"]["sham"]["is_control"] is True
    assert st["table"]["10"]["dg0.5"]["k_eff"] == 0.5


def test_three_N_levels_enables_extrapolation(tmp_path):
    base = str(tmp_path / "runs")
    conds = [("off", "off", None), ("free", "free", None)]
    dirs = _grid(base, [10, 20, 40], conds, [1, 2], "c_fss")
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))
    with open(os.path.join(out, "fss.json"), encoding="utf-8") as fh:
        fss = json.load(fh)
    assert fss["n_levels"] == 3
    assert fss["sufficient"] is True
    assert "extrapolation" in fss


def test_legacy_run_name_no_regression(tmp_path):
    """旧形式(N トークン無し)ラン名でも config.yaml の n_agents で group できる。"""
    base = str(tmp_path / "runs")
    conds = [("off", "off", None), ("free", "free", None)]
    dirs = _grid(base, [60], conds, [1, 2], "pilot", multi_n=False)
    # ラン名に N トークンが無いこと(=従来の pilot_<mode>_s<seed>)
    assert all("_N" not in os.path.basename(d) for d in dirs)
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))
    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    assert st["N_levels"] == [60]                  # config から N を復元
    assert set(st["table"]["60"]) == {"off", "free"}
