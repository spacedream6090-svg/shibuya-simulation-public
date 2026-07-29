"""LLM 健全性 KPI(observer.llm_health)+ summary の性能キー + bench.py 出力先のテスト。

P0バッチ 2026-07-29。

方針(第62/64/65 の鉄則を継承):
- OFF(既定): L2 に llm_* 列が 1 つも出ない・L1 バイト一致・LLM 呼数一致・
  新イベントゼロ(= 既存ランと完全に同じ観測)。
- ON: 3 列(llm_calls_total / llm_cache_hit_rate / llm_fallback_rate)が毎 step 出て、
  値が summary.json / CachedLLM のカウンタ / L1 の `fallback` イベント数と厳密に一致する。
- ObserverLogger の内部カウンタは flush_segment(part 化)を挟んでも失われない。
- summary.json の追加キー(elapsed_sec / peak_rss_mb)は **既存キーを一切変えない**。
- scripts/bench.py: 既存出力を上書きしない安全弁と、既存ランの再集計の純関数。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import aggregate as agg
from society.observer.logger import ObserverLogger
from society.observer.schema import Event

REPO_ROOT = Path(__file__).resolve().parents[1]

_HEALTH_ON = {"observer.llm_health.enabled": "true"}


def _sim(tmp_path, name, n=8, steps=6, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144",
           f"run.out_dir={(tmp_path / 'runs').as_posix()}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l2_cols(sim):
    return pq.read_table(sim.out_dir / "l2_metrics.parquet").column_names


# --------------------------------------------------------------------------- #
# 1) 既定 OFF = 完全な no-op
# --------------------------------------------------------------------------- #
def test_off_has_no_llm_columns_and_identical_l1(tmp_path):
    base = _sim(tmp_path, "h_base")
    base.run()
    on_off = _sim(tmp_path, "h_off", **{"observer.llm_health.enabled": "false"})
    on_off.run()

    assert [c for c in _l2_cols(base) if c.startswith("llm_")] == []
    assert _l1(base) == _l1(on_off)
    assert base.llm.calls == on_off.llm.calls


def test_on_does_not_change_l1_or_call_count(tmp_path):
    """ON は「読むだけ」= L1 バイト一致・LLM 呼数一致(観測だけが増える)。"""
    off = _sim(tmp_path, "h2_off")
    off.run()
    on = _sim(tmp_path, "h2_on", **_HEALTH_ON)
    on.run()

    assert _l1(off) == _l1(on)
    assert off.llm.calls == on.llm.calls
    assert sorted(c for c in _l2_cols(on) if c.startswith("llm_")) == [
        "llm_cache_hit_rate", "llm_calls_total", "llm_fallback_rate"]


# --------------------------------------------------------------------------- #
# 2) ON の値の検算
# --------------------------------------------------------------------------- #
def test_on_values_match_counters(tmp_path):
    sim = _sim(tmp_path, "h3", n=12, steps=12, **_HEALTH_ON)
    summary = sim.run()
    rows = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pylist()
    assert rows, "L2 が空"

    last = rows[-1]
    # 最終 step の累積 = summary.json の llm_calls(同じ源=CachedLLM.calls)
    assert last["llm_calls_total"] == summary["llm_calls"]
    calls = summary["llm_calls"]
    if calls:
        assert abs(last["llm_cache_hit_rate"]
                   - summary["llm_cache_hits"] / calls) < 1e-9
        n_fb = sum(1 for e in sim.logger.events if e.kind == "fallback")
        # flush していないランでは logger.events が全件 = 内部カウンタと一致
        assert sim.logger.n_fallback_events == n_fb
        assert abs(last["llm_fallback_rate"] - n_fb / calls) < 1e-9

    # 累積列は単調非減少(per-step 差分ではないことの表明)
    totals = [r["llm_calls_total"] for r in rows]
    assert totals == sorted(totals)
    for r in rows:
        assert 0.0 <= r["llm_cache_hit_rate"] <= 1.0
        assert 0.0 <= r["llm_fallback_rate"] <= 1.0


def test_aggregators_are_pure_arithmetic():
    """集計器そのものの検算(合成 sim。乱数もシミュも要らない)。"""
    class _Cfg:
        observer = {"llm_health": {"enabled": True}}

    class _LLM:
        calls = 200
        hits = 50

    class _Logger:
        n_fallback_events = 4
        n_llm_rows = 200
        n_llm_cached = 50

    class _Sim:
        cfg = _Cfg()
        llm = _LLM()
        logger = _Logger()

    sim = _Sim()
    assert agg.AGGREGATORS["llm_calls_total"](sim) == 200
    assert agg.AGGREGATORS["llm_cache_hit_rate"](sim) == 0.25
    assert agg.AGGREGATORS["llm_fallback_rate"](sim) == 0.02

    # 呼数 0 のときはゼロ除算せず 0.0
    _LLM.calls = 0
    _LLM.hits = 0
    assert agg.AGGREGATORS["llm_calls_total"](sim) == 0
    assert agg.AGGREGATORS["llm_cache_hit_rate"](sim) == 0.0
    assert agg.AGGREGATORS["llm_fallback_rate"](sim) == 0.0

    # OFF なら全て None(=列を出さない)
    _Cfg.observer = {"llm_health": {"enabled": False}}
    for key in ("llm_calls_total", "llm_cache_hit_rate", "llm_fallback_rate"):
        assert agg.AGGREGATORS[key](sim) is None
    # ブロックごと無い古い config でも None(後方互換)
    _Cfg.observer = {}
    for key in ("llm_calls_total", "llm_cache_hit_rate", "llm_fallback_rate"):
        assert agg.AGGREGATORS[key](sim) is None


def test_logger_counters_survive_flush_segment(tmp_path):
    """part 化(flush_segment)でバッファが空になっても累積カウンタは保たれる。"""
    lg = ObserverLogger(tmp_path / "lg")
    for i in range(3):
        lg.log(Event(step=i, sim_min=i, agent_id=0, kind="fallback",
                     x=0.0, y=0.0, payload={"reason": "parse_error"}))
    lg.log_llm_call({"llm_call_id": "a", "agent_id": 0, "purpose": "social",
                     "step": 0, "cached": False})
    lg.log_llm_call({"llm_call_id": "a", "agent_id": 0, "purpose": "social",
                     "step": 1, "cached": True})
    assert (lg.n_fallback_events, lg.n_llm_rows, lg.n_llm_cached) == (3, 2, 1)
    lg.flush_segment()
    assert lg.events == [] and lg.llm_calls == []
    assert (lg.n_fallback_events, lg.n_llm_rows, lg.n_llm_cached) == (3, 2, 1)


# --------------------------------------------------------------------------- #
# 3) summary.json の追加キー
# --------------------------------------------------------------------------- #
def test_summary_has_perf_keys_without_touching_existing(tmp_path):
    sim = _sim(tmp_path, "h4")
    summary = sim.run()
    # 既存キーは全部そのまま在る
    for key in ("n_agents", "n_steps", "n_events", "event_kinds", "llm_calls",
                "llm_cache_hits", "n_items", "n_transmissions",
                "total_adoptions", "out_dir", "files"):
        assert key in summary
    assert isinstance(summary["elapsed_sec"], float)
    assert summary["elapsed_sec"] >= 0.0
    # peak_rss_mb は取得できた環境でのみ出す(欠測を 0 と偽らない契約)
    if "peak_rss_mb" in summary:
        assert isinstance(summary["peak_rss_mb"], float)
        assert summary["peak_rss_mb"] > 0.0
    on_disk = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["elapsed_sec"] == summary["elapsed_sec"]


# --------------------------------------------------------------------------- #
# 4) scripts/bench.py(出力の喪失防止 + 既存ラン再集計)
# --------------------------------------------------------------------------- #
def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "_bench_p0", REPO_ROOT / "scripts" / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench_p0"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bench_does_not_clobber_existing_output(tmp_path):
    bench = _load_bench()
    js, md = bench.resolve_out_paths(tmp_path, "bench_scaling", overwrite=False)
    assert js.name == "bench_scaling.json" and md.name == "bench_scaling.md"
    js.write_text("{}", encoding="utf-8")
    js2, md2 = bench.resolve_out_paths(tmp_path, "bench_scaling", overwrite=False)
    assert js2 != js and js2.name.startswith("bench_scaling-")
    assert js2.suffix == ".json" and md2.suffix == ".md"
    assert json.loads(js.read_text(encoding="utf-8")) == {}, "既存実測が壊されている"
    # --overwrite を明示したときだけ従来どおり同名を返す
    js3, _ = bench.resolve_out_paths(tmp_path, "bench_scaling", overwrite=True)
    assert js3 == js


def test_bench_reaggregates_existing_runs(tmp_path):
    bench = _load_bench()
    run = tmp_path / "n40"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(
        {"n_agents": 40, "n_steps": 144, "n_events": 8640, "llm_calls": 720}),
        encoding="utf-8")
    empty = tmp_path / "nosummary"
    empty.mkdir()

    rows = bench.rows_from_existing_runs([run, empty])
    assert len(rows) == 1, "summary.json の無い dir は捏造せずスキップ"
    r = rows[0]
    assert r["agents"] == 40 and r["steps"] == 144
    assert r["wall_s"] is None, "wall 実測の無い世代のランを 0 と偽ってはならない"
    assert r["llm_per_agent_day"] == 18.0        # 720 / (40 × 1日)
    assert r["events_per_agent_day"] == 216.0    # 8640 / 40
    assert r["source"] == "existing:n40"


def test_bench_per_agent_day_pure():
    bench = _load_bench()
    assert bench._per_agent_day(1000, 100, 144) == 10.0
    assert bench._per_agent_day(1000, 100, 288) == 5.0
    assert bench._per_agent_day(10, 0, 144) is None
    assert bench._per_agent_day(10, 100, 0) is None


# --------------------------------------------------------------------------- #
# 5) watchdog(ラン中の警告)/ watchdog_llm(事後点検)
# --------------------------------------------------------------------------- #
def _load_script(stem: str):
    spec = importlib.util.spec_from_file_location(
        f"_p0_{stem}", REPO_ROOT / "scripts" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_p0_{stem}"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_watchdog_reads_llm_health_and_warns(tmp_path):
    """watchdog は L2 の llm_fallback_rate を読んで警告する(kill はしない)。"""
    wd_mod = _load_script("watchdog")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = wd_mod.parse_args(["--run-dir", str(run_dir), "--fallback-warn", "0.1"])
    wd = wd_mod.Watchdog(args)

    # まだ L2 が無い → 監視不能(None)。例外を投げない。
    assert wd.read_llm_health() is None
    wd._maybe_check_llm_health(now=10.0 ** 9)
    assert wd.last_health is None
    assert "監視できない" in wd.log_path.read_text(encoding="utf-8")

    # llm_health ON のランを模した L2 を置く
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"step": [0, 1],
                             "llm_calls_total": [10, 20],
                             "llm_fallback_rate": [0.0, 0.5],
                             "llm_cache_hit_rate": [0.0, 0.25]}),
                   run_dir / "l2_metrics.parquet")
    health = wd.read_llm_health()
    assert health["llm_fallback_rate"] == 0.5 and health["llm_calls_total"] == 20

    wd._health_next_t = 0.0
    wd._maybe_check_llm_health(now=10.0 ** 9)
    log = wd.log_path.read_text(encoding="utf-8")
    assert "WARN llm_fallback_rate=0.5000" in log
    wd.write_status("running")
    status = json.loads(wd.status_path.read_text(encoding="utf-8"))
    assert status["llm_health"]["llm_fallback_rate"] == 0.5


def test_watchdog_llm_posthoc_report(tmp_path):
    """watchdog_llm は llm_health OFF のランでも L1b + summary から健全性を出せる。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    wl = _load_script("watchdog_llm")
    run = tmp_path / "r1"
    run.mkdir()
    pq.write_table(pa.table({
        "llm_call_id": ["a", "b", "c", "d"],
        "agent_id": [1, 1, 2, 2],
        "purpose": ["social", "plan", "plan_retry", "reflect"],
        "step": [0, 1, 1, 2],
        "cached": [False, True, False, False]}), run / "l1b_llm.parquet")
    (run / "summary.json").write_text(json.dumps(
        {"n_agents": 2, "n_steps": 3, "llm_calls": 4,
         "event_kinds": {"fallback": 1, "arrive": 9}}), encoding="utf-8")

    rep = wl.check_run(run)
    assert rep["llm_calls"] == 4
    assert rep["fallbacks"] == 1 and rep["fallback_rate"] == 0.25
    assert rep["cache_hit_rate"] == 0.25
    assert rep["plan_retry_rate"] == 1.0          # plan 1 呼に対し plan_retry 1 呼
    assert rep["l2_llm_health"] is None           # llm_health OFF のランは L2 列なし
    assert rep["by_purpose"]["social"]["calls"] == 1
