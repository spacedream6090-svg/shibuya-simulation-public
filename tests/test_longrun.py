"""第57バッチ タスクC: 長期日常ラン(30日級)プロトコルのテスト。

検証項目:
  (A) conf/longrun30.yaml が load_config で読める + 長期観察の要フラグが立つ
      (観察揺らぎ chance/stochastic/boredom・生活機構・lens 3系・checkpoint・n_steps=4320)。
  (B) --daily-rollup: build_rollup_data の日次集計が正しい(sim_min 由来の日境界・日次平均・
      structure.json の束ね)。main() は rollup.html だけを追加生成し、既存 viewer/dashboard の
      生成経路には入らない(= --daily-rollup 未指定=従来出力・rollup.html を作らない)。
  (C) scripts/bench_longrun の外挿計算(_linfit / _extrap)の単体。
  (D) 分割 resume の一致スモーク: 「一気 30step」==「10+10+10 の 3チャンク(各 finalize)」の
      l1/l2/l3 がバイト一致(第57バッチの分割実行=前チャンク canonical 結合の補修を固定)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from society.config import load_config          # noqa: E402
from society.engine.simulation import Simulation  # noqa: E402

import make_viewer as mv                          # noqa: E402


# =========================================================== (A) profile 読み込み
def test_longrun_profile_loads_with_expected_flags():
    cfg = load_config(overrides=["run.n_agents=20", "run.n_steps=8", "model.backend=mock"],
                      profile="conf/longrun30.yaml")
    # 期間・checkpoint(分割前提)
    # ↑ n_steps は override したが、素の profile の 4320 も確認する:
    raw = load_config(profile="conf/longrun30.yaml")
    assert raw.run.n_steps == 4320, "30日=4320step が profile 既定でない"
    assert int(cfg.observer.checkpoint_every) == 1440, "checkpoint_every=1440(10日)でない"
    assert bool(cfg.run.get("seed_auto")) is True, "seed_auto(純観察)が立っていない"
    # 観察の揺らぎ層(observe 姿勢)
    assert cfg.chance.enabled and cfg.routine.stochastic.enabled and cfg.drive.boredom.enabled
    assert cfg.weather.enabled and cfg.drive.drift.enabled
    # 生活機構(daily 本番構成と整合)
    assert cfg.relations.enabled and cfg.hierarchy.enabled and cfg.household.enabled
    assert cfg.organizations.enabled and cfg.economy.accounts.enabled
    # 観測レンズ 3系(構造創発の測る道具)
    assert cfg.lens.enabled and cfg.lens.structure.enabled and cfg.lens.deviation.enabled
    assert cfg.lens.trust.enabled and cfg.lens.value4.enabled and cfg.lens.motives.enabled


# =========================================================== (B) 日次ロールアップ
def _synth_rollup_run(run_dir: Path, n_days: int = 2, start_min: int = 0,
                      with_structure: bool = True) -> Path:
    """合成ラン: l1(sim_min つき)+ l2(step毎)+ summary(+structure)を書く。
    start_min=0(深夜0時起点)にして 1日=144step の綺麗な境界にする。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    spd = mv.ROLLUP_STEPS_PER_DAY                      # 144
    n_steps = n_days * spd
    # --- L1(build_rollup の start_min 復元用に sim_min 列を持たせる)---
    rows = []
    for step in range(n_steps):
        rows.append({"step": step, "sim_min": start_min + step * mv.STEP_MINUTES,
                     "agent_id": 0, "kind": "arrive", "x": 0.0, "y": 0.0,
                     "payload": "{}", "rng_stream": "", "llm_call_id": ""})
    l1_fields = [("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
                 ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
                 ("payload", pa.string()), ("rng_stream", pa.string()),
                 ("llm_call_id", pa.string())]
    pq.write_table(pa.table({k: [r[k] for r in rows] for k, _ in l1_fields},
                            schema=pa.schema(l1_fields)),
                   run_dir / "l1_events.parquet")
    # --- L2(step毎1行。mean_grievance=日ごとの定数=日次平均がその定数に一致)---
    l2 = []
    for step in range(n_steps):
        d = (start_min + step * mv.STEP_MINUTES) // 1440
        l2.append({"step": step, "mean_grievance": 0.1 * (d + 1),
                   "edge_churn_rate": 0.02 * (d + 1), "n_sns_posts": float(step)})
    l2_keys = sorted(l2[0].keys())
    pq.write_table(pa.table({k: [r[k] for r in l2] for k in l2_keys}),
                   run_dir / "l2_metrics.parquet")
    (run_dir / "summary.json").write_text(
        json.dumps({"n_agents": 12, "n_steps": n_steps}, ensure_ascii=False),
        encoding="utf-8")
    if with_structure:
        struct = {"run": run_dir.name, "n_days": n_days, "days": list(range(n_days)),
                  "churn": {"churn_rate": [0.03 * (d + 1) for d in range(n_days)]},
                  "rank": {"tau_prev_day": [None] + [0.5] * (n_days - 1)},
                  "centrality": {"turnover": [None] + [0.1] * (n_days - 1)},
                  "community": {"change_rate": [None] + [0.2] * (n_days - 1)},
                  "stagnation": {"combined": [], "total_stagnant_days": 0,
                                 "longest": None, "by_signal": {
                                     "centrality_churn": [], "edge_churn": [],
                                     "rank_tau": []}}}
        (run_dir / "structure.json").write_text(
            json.dumps(struct, ensure_ascii=False), encoding="utf-8")
    return run_dir


def test_rollup_daily_aggregation(tmp_path):
    rd = _synth_rollup_run(tmp_path / "roll", n_days=3, start_min=0)
    data = mv.build_rollup_data(rd)
    assert data["startMin"] == 0
    assert data["days"] == [0, 1, 2], "3日ぶんの日インデックスが揃わない"
    # 日次平均=日ごと定数どおり(0.1, 0.2, 0.3)
    assert data["metrics"]["mean_grievance"] == [0.1, 0.2, 0.3]
    assert data["metrics"]["edge_churn_rate"] == [0.02, 0.04, 0.06]
    # n_sns_posts の日次平均は step の日内平均(day0=0..143 の平均=71.5)
    assert abs(data["metrics"]["n_sns_posts"][0] - 71.5) < 1e-6
    assert "mean_grievance" in data["metricKeys"] and data["hasL2"] is True
    assert data["structure"] is not None and data["nAgents"] == 12


def test_rollup_startmin_shifts_day_boundary(tmp_path):
    """start_min=420(07:00)なら sim_min//1440 の日境界がずれ、日数が増える(構造と同定義)。"""
    rd = _synth_rollup_run(tmp_path / "roll7", n_days=2, start_min=420,
                           with_structure=False)
    data = mv.build_rollup_data(rd)
    assert data["startMin"] == 420
    # 07:00 起点で 288step は 2暦日 → 3日(day2 は 07:00 まで=partial)にまたがる
    assert data["days"][-1] == 2 and data["structure"] is None


def test_rollup_graceful_without_l2(tmp_path):
    rd = tmp_path / "empty"
    rd.mkdir()
    # L1 だけ(sim_min 無し)→ start_min 既定・L2 空・structure 無し=骨格を返す(例外なし)
    pq.write_table(pa.table({"step": pa.array([0], pa.int32()),
                             "agent_id": pa.array([0], pa.int32()),
                             "kind": pa.array(["arrive"], pa.string())}),
                   rd / "l1_events.parquet")
    data = mv.build_rollup_data(rd)
    assert data["hasL2"] is False and data["metricKeys"] == []
    assert data["structure"] is None


def test_daily_rollup_cli_only_writes_rollup(tmp_path):
    """--daily-rollup は rollup.html だけを追加生成し、viewer/dashboard を作らない
    (= 既存生成経路に入らない=未指定時の従来出力はバイト同一の担保)。"""
    rd = _synth_rollup_run(tmp_path / "cli", n_days=2, start_min=0)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"),
                        str(rd), "--daily-rollup"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert (rd / "rollup.html").exists(), "rollup.html が生成されていない"
    assert not (rd / "viewer.html").exists(), "--daily-rollup で viewer.html を作ってはいけない"
    assert not (rd / "dashboard.html").exists(), "--daily-rollup で dashboard.html を作ってはいけない"
    html = (rd / "rollup.html").read_text(encoding="utf-8")
    assert "日次ロールアップ" in html, "画面に日次ロールアップの明示が無い(silent 禁止)"
    assert "Day 0" in html or "days" in html
    assert len(html) < 400_000, "ロールアップが軽量でない(positions を埋め込んでいる疑い)"


# =========================================================== (C) bench 外挿単体
def test_bench_linfit_and_extrap():
    import bench_longrun as bl
    a, b = bl._linfit([144.0, 432.0], [30.0, 90.0])   # 完全比例(切片0・傾き0.2083…)
    assert abs(b - (90 - 30) / (432 - 144)) < 1e-9
    assert abs(a - (30 - b * 144)) < 1e-6
    # 外挿: 1008step の予測
    y = bl._extrap(a, b, 1008)
    assert abs(y - (a + b * 1008)) < 1e-9 and y > 90


# =========================================================== (D) 分割 resume 一致
def _cfg(name: str, n_steps: int, every: int, tmp_path: Path):
    return load_config(["run.seed=42", "run.n_agents=12", f"run.n_steps={n_steps}",
                        f"run.name={name}", "model.backend=mock",
                        f"observer.checkpoint_every={every}"])


def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def test_split_execution_matches_straight(tmp_path):
    """一気 30step == 10+10+10 の 3チャンク(各チャンクが clean に finalize)で l1/l2/l3 が一致。
    分割実行(夜間予算で区切って回す)= conf/longrun30.yaml の運用手順を固定する。"""
    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 30, 10, tmp_path), out_dir=straight).run()

    chunk = tmp_path / "chunk"
    Simulation(_cfg("chunk", 10, 10, tmp_path), out_dir=chunk).run()                 # Day0-1
    Simulation(_cfg("chunk", 20, 10, tmp_path), out_dir=chunk).run(resume_from=chunk)  # -> 20
    Simulation(_cfg("chunk", 30, 10, tmp_path), out_dir=chunk).run(resume_from=chunk)  # -> 30

    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(chunk, stem), f"{stem} が分割で不一致"
    # 分割後に part が残っていない(finalize で結合・削除)
    assert not list(chunk.glob("l1_events.part-*.parquet"))
