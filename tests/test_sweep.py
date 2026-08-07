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


def _write_l2(run_dir, n_steps, llm_health):
    """l2_metrics.parquet を書く。llm_health=None なら **3 列を出さない**

    (= observer.llm_health.enabled=false のラン。列そのものが存在しない)。
    llm_health=(calls_final, hit_rate, fallback_rate) を渡すと 3 列が載る。
    calls は**累積列**なので step ごとに増やし、最終行が calls_final になる。
    """
    cols = {"step": pa.array(list(range(n_steps)), pa.int32()),
            "mean_drive": pa.array([0.5] * n_steps, pa.float64())}
    if llm_health is not None:
        calls_final, hit, fb = llm_health
        cols["llm_calls_total"] = pa.array(
            [int(round(calls_final * (s + 1) / n_steps)) for s in range(n_steps)],
            pa.int64())
        cols["llm_cache_hit_rate"] = pa.array([hit] * n_steps, pa.float64())
        cols["llm_fallback_rate"] = pa.array([fb] * n_steps, pa.float64())
    pq.write_table(pa.table(cols), os.path.join(run_dir, "l2_metrics.parquet"))


def _write_run(base, name, n_agents, seed, writeback, alpha=0.5, n_steps=12,
               llm_health=False):
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
    # llm_health=False(既定)は L2 そのものを書かない = 従来のテストと完全同一
    if llm_health is not False:
        _write_l2(d, n_steps, llm_health)
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


# --------------------------------------------------------------------------- #
# 小粒C タスク1: llm_health 3 列(累積列のラン最終値)の接続
# --------------------------------------------------------------------------- #
def test_last_valid_skips_trailing_missing():
    """累積列の要約は「最後の**有効**値」。列なし/全欠測は None(0 で埋めない)。"""
    assert A._last_valid([1.0, 2.0, 3.0]) == 3.0
    assert A._last_valid([1.0, 2.0, None]) == 2.0          # 末尾欠測は飛ばす
    assert A._last_valid([1.0, None, float("nan")]) == 1.0  # nan も欠測扱い
    assert A._last_valid([None, None]) is None
    assert A._last_valid([]) is None
    assert A._last_valid(None) is None                      # 列そのものが無いラン


def test_llm_health_three_columns_reach_the_sweep_table(tmp_path):
    """3 列が (N,k) セル・runs 行・report に出る。calls は合計、率は CI セル。"""
    base = str(tmp_path / "runs")
    dirs = []
    for wb, lab in (("off", "off"), ("free", "free")):
        for s in (1, 2):
            dirs.append(_write_run(base, f"h_N10_{lab}_s{s}", 10, s, wb,
                                   llm_health=(1000, 0.25, 0.05)))
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))

    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    for lab in ("off", "free"):
        c = st["table"]["10"][lab]
        assert c["llm_health_runs"] == 2 == c["n_runs"]
        # 累積列の最終値 1000 が 2 ラン分 → セルでは合計
        assert c["llm_calls_total"] == 2000.0
        # 率 2 列は既にラン全体の率なのでレート化せずそのまま平均
        assert c["llm_cache_hit_rate"]["mean"] == 0.25
        assert c["llm_fallback_rate"]["mean"] == 0.05
    # runs 行にもラン単位の値が載る
    assert all(r["llm_calls_total"] == 1000.0 for r in st["runs"])
    assert all(r["llm_fallback_rate"] == 0.05 for r in st["runs"])

    with open(os.path.join(out, "report_sweep.md"), encoding="utf-8") as fh:
        md = fh.read()
    assert "LLM health" in md
    assert "cache_hit_rate" in md and "fallback_rate" in md
    assert "2/2" in md                              # runs w/ llm_health 列


def test_llm_health_off_runs_are_na_not_dropped(tmp_path):
    """llm_health OFF(列なし)のランでも解析は完走し、当該 3 列だけ n/a になる。"""
    base = str(tmp_path / "runs")
    dirs = []
    for wb, lab in (("off", "off"), ("free", "free")):
        for s in (1, 2):
            # L2 は在るが llm_health 3 列は無い = observer.llm_health.enabled=false
            dirs.append(_write_run(base, f"n_N10_{lab}_s{s}", 10, s, wb,
                                   llm_health=None))
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))

    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    for lab in ("off", "free"):
        c = st["table"]["10"][lab]
        assert c["llm_health_runs"] == 0
        assert c["llm_calls_total"] is None         # 欠測を 0 で埋めない
        assert c["llm_cache_hit_rate"]["mean"] is None
        assert c["llm_fallback_rate"]["mean"] is None
        # ★他の指標は落ちていない(ラン自体は生きている)
        assert c["n_runs"] == 2 and c["n_seeds"] == 2
        assert "EWS_var" in c and c["compute_total"] >= 0
    with open(os.path.join(out, "report_sweep.md"), encoding="utf-8") as fh:
        md = fh.read()
    assert "LLM health" in md and "0/2" in md


def test_llm_health_mixed_cell_counts_only_present_runs(tmp_path):
    """同一セル内に列ありランと列なしランが混在しても合計/平均は列ありぶんだけ。"""
    base = str(tmp_path / "runs")
    dirs = [_write_run(base, "m_N10_off_s1", 10, 1, "off",
                       llm_health=(300, 0.5, 0.1)),
            _write_run(base, "m_N10_off_s2", 10, 2, "off", llm_health=None)]
    out = A.analyze_sweep(dirs, out=str(tmp_path / "out"))
    with open(os.path.join(out, "sweep_table.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    c = st["table"]["10"]["off"]
    assert c["n_runs"] == 2 and c["llm_health_runs"] == 1
    assert c["llm_calls_total"] == 300.0            # 列なしランを 0 として足さない
    assert c["llm_fallback_rate"]["mean"] == 0.1
    assert c["llm_fallback_rate"]["n"] == 1         # seed ブロックも 1 本だけ
