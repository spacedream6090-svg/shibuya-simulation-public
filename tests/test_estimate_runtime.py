"""estimate_runtime.py と make_hub.py の純関数テスト(実データ非依存)。

合成の l1b_llm.parquet(既知の日別呼数)・agents.json・空 HTML を tmp に作り、
較正モデル(g / α / 総時間)と外挿警告、ハブのタブ検出・md 変換を検証する。
シミュ実行は不要。
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
sys.path.insert(0, str(_ROOT / "viz"))
sys.path.insert(0, str(_ROOT / "scripts"))

import estimate_runtime as er  # noqa: E402
import make_hub as mh  # noqa: E402

SPD = er.DEFAULT_STEPS_PER_DAY  # 144


# ── 合成ラン生成ヘルパ ────────────────────────────────────────────────────
def _write_run(run_dir: Path, n_agents: int, daily_calls: list[int],
               cached_per_day: list[int] | None = None) -> Path:
    """指定の日別呼数を持つ合成 l1b_llm.parquet と agents.json を書く。

    day d の呼を step=(d-1)*SPD+1 に集中させる(集計上の day 割当は同じ)。
    cached_per_day を渡すと各日その本数だけ cached=True の行を追加する。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    steps: list[int] = []
    cached: list[bool] = []
    for d, ncalls in enumerate(daily_calls, start=1):
        base_step = (d - 1) * SPD + 1
        steps += [base_step] * ncalls
        cached += [False] * ncalls
        if cached_per_day:
            nc = cached_per_day[d - 1]
            steps += [base_step] * nc
            cached += [True] * nc
    tbl = pa.table({
        "agent_id": pa.array([0] * len(steps), pa.int64()),
        "cached": pa.array(cached, pa.bool_()),
        "step": pa.array(steps, pa.int64()),
    })
    pq.write_table(tbl, run_dir / "l1b_llm.parquet")
    agents = [{"id": i, "name": f"a{i}"} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents), encoding="utf-8")
    return run_dir


# ── read_daily_calls ──────────────────────────────────────────────────────
def test_read_daily_calls_basic(tmp_path):
    rd = _write_run(tmp_path / "r", 50, [100, 110, 121])
    assert er.read_daily_calls(rd) == [100, 110, 121]
    assert er.read_agent_count(rd) == 50


def test_read_daily_calls_excludes_cached(tmp_path):
    # 実呼 [100, 200]・cached [10, 20] → cached を除いた実呼のみ数える
    rd = _write_run(tmp_path / "r", 30, [100, 200], cached_per_day=[10, 20])
    assert er.read_daily_calls(rd) == [100, 200]


# ── geom_growth ───────────────────────────────────────────────────────────
def test_geom_growth_exact():
    assert er.geom_growth([100, 110, 121]) == pytest.approx(0.10, rel=1e-9)
    assert er.geom_growth([200, 200, 200]) == pytest.approx(0.0, abs=1e-12)


def test_geom_growth_degenerate():
    assert er.geom_growth([100]) is None
    assert er.geom_growth([0, 100]) is None
    assert er.geom_growth([]) is None


# ── fit_alpha(log-log 回帰) ──────────────────────────────────────────────
def test_fit_alpha_linear():
    # calls ∝ N なら α = 1
    assert er.fit_alpha([(10, 100), (100, 1000)]) == pytest.approx(1.0, rel=1e-9)


def test_fit_alpha_superlinear():
    # (15, 187.5) と (200, 4105) の対 → 1.19 付近
    a = er.fit_alpha([(er.ANCHOR_N, er.ANCHOR_CALLS_PER_DAY), (200, 4105)])
    expected = np.log(4105 / 187.5) / np.log(200 / 15)
    assert a == pytest.approx(expected, rel=1e-9)
    assert 1.15 < a < 1.25


def test_fit_alpha_needs_two_points():
    with pytest.raises(ValueError):
        er.fit_alpha([(100, 500)])
    with pytest.raises(ValueError):
        er.fit_alpha([(100, 500), (100, 600)])  # 体数同一


# ── predict_calls / 総時間 ────────────────────────────────────────────────
def test_predict_calls_and_total_time():
    # C1=1000, N_calib=100, α=1, g=0 → N=200 で常に 2000 呼/日
    calls = er.predict_calls(1000, 100, 1.0, 0.0, 200, 3)
    assert calls == pytest.approx([2000, 2000, 2000])
    # 総時間: Σ calls × sec × (1+overhead)
    sec, ov = 3.1, 0.15
    total = sum(calls) * sec * (1 + ov)
    assert total == pytest.approx(6000 * 3.1 * 1.15)


def test_predict_calls_growth():
    # g=0.10 → 幾何級数。C1=1000,N=N_calib,α任意で日 [1000,1100,1210]
    calls = er.predict_calls(1000, 100, 1.5, 0.10, 100, 3)
    assert calls == pytest.approx([1000, 1100, 1210])


def test_build_report_total_matches_manual(tmp_path):
    rd = _write_run(tmp_path / "r", 100, [1000, 1100, 1210])
    calib = er.build_calibration([rd])
    # 単一較正: N_calib=100, C1=1000, g=(1210/1000)^0.5-1=0.10
    assert calib.n_calib == 100
    assert calib.c1 == pytest.approx(1000)
    assert calib.g == pytest.approx(0.10, rel=1e-9)
    rep = er.build_report(100, 3, calib, sec_per_call=3.0, overhead=0.0)
    # N=N_calib, α 無関係(倍率1) → 日別 [1000,1100,1210], 総 3310 呼
    assert rep["total_calls"] == pytest.approx(3310)
    assert rep["total_sec"] == pytest.approx(3310 * 3.0)


def test_format_hms():
    assert er.format_hms(3600) == "1時間00分"
    assert er.format_hms(3660) == "1時間01分"
    assert er.format_hms(1800) == "30分"


# ── 較正の α 選択ロジック ─────────────────────────────────────────────────
def test_build_calibration_two_runs_regression(tmp_path):
    r1 = _write_run(tmp_path / "r15", 15, [100, 120])
    r2 = _write_run(tmp_path / "r200", 200, [4000, 4400])
    calib = er.build_calibration([r1, r2])
    assert calib.n_runs == 2
    assert calib.n_calib == 200  # 基準=体数最大
    expected_alpha = np.log(4000 / 100) / np.log(200 / 15)
    assert calib.alpha == pytest.approx(expected_alpha, rel=1e-9)
    assert "回帰" in calib.alpha_method


def test_build_calibration_single_run_uses_anchor(tmp_path):
    r = _write_run(tmp_path / "r200", 200, [4105, 4200])
    calib = er.build_calibration([r])
    # 較正1本 → 15体アンカーとの対で α 自動計算
    expected = np.log(4105 / er.ANCHOR_CALLS_PER_DAY) / np.log(200 / er.ANCHOR_N)
    assert calib.alpha == pytest.approx(expected, rel=1e-9)
    assert "アンカー" in calib.alpha_method


def test_build_calibration_no_runs_defaults():
    calib = er.build_calibration([])
    assert calib.n_runs == 0
    assert calib.alpha == er.DEFAULT_ALPHA
    assert calib.n_calib == er.BUILTIN_N
    assert calib.c1 == er.BUILTIN_DAY1


def test_alpha_override(tmp_path):
    r = _write_run(tmp_path / "r", 200, [4000, 4400])
    calib = er.build_calibration([r], alpha_override=1.42)
    assert calib.alpha == 1.42
    assert "指定" in calib.alpha_method


# ── 外挿警告 ──────────────────────────────────────────────────────────────
def test_extrapolation_warnings_fire():
    calib = er.build_calibration([])  # n_runs=0 でも警告に含まれる
    # 体数 300 (>200) & 日数 7 (>3) の両方で警告
    warns = er.extrapolation_warnings(300, 7, calib)
    joined = " ".join(warns)
    assert "300" in joined and "体" in joined
    assert "7" in joined and "日" in joined


def test_no_extrapolation_warning_in_range(tmp_path):
    r = _write_run(tmp_path / "r", 100, [1000, 1100, 1210])
    calib = er.build_calibration([r])  # n_runs=1
    warns = er.extrapolation_warnings(100, 2, calib)
    # 範囲内(15..200体・1..3日)かつ較正あり → 警告なし
    assert warns == []


# ══════════════════════════════ make_hub ══════════════════════════════════
def test_md_to_html():
    md = "# 見出し\n\n- 項目**強調**\n\n---\n本文 `code` です"
    out = mh.md_to_html(md)
    assert "<h1>見出し</h1>" in out
    assert "<li>項目<strong>強調</strong></li>" in out
    assert "<hr>" in out
    assert "<code>code</code>" in out


def test_md_to_html_escapes():
    out = mh.md_to_html("plain <script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def _make_synth_run(run_dir: Path, files: list[str], summary: dict | None = None):
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".md"):
            p.write_text("# ラン要約\n\n- 体数 200\n", encoding="utf-8")
        else:
            p.write_text("<html><body>x</body></html>", encoding="utf-8")
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_hub_detects_present_tabs_only(tmp_path):
    run = tmp_path / "run_a"
    _make_synth_run(
        run,
        ["viewer3d.html", "dashboard.html", "panel/crowd_demo.html", "panel/summary_ja.md"],
        summary={"n_agents": 200, "n_steps": 432, "n_events": 238993,
                 "event_kinds": {"a": 1, "b": 2}, "llm_calls": 13161},
    )
    out_path, artifacts = mh.make_hub(run)
    labels = {a["label"] for a in artifacts}
    # 存在するもの
    assert "3Dビュー" in labels
    assert "ダッシュボード" in labels
    assert "群集デモ" in labels
    assert "要約" in labels
    # 存在しないもの(タブを出さない)
    assert "2Dビュー" not in labels
    assert "ヒートマップ" not in labels
    assert "OD人流" not in labels

    doc = out_path.read_text(encoding="utf-8")
    # iframe の相対 src / 遅延 src が入っている
    assert 'src="viewer3d.html"' in doc  # 先頭タブは即 src
    assert 'data-src="panel/crowd_demo.html"' in doc  # 後続は遅延
    # ヒートマップのタブは出ない
    assert "ヒートマップ</button>" not in doc
    # md はインライン埋め込み(iframe ではなく <div class="md">)
    assert "ラン要約" in doc
    # ヘッダのラン概要
    assert "200" in doc and "238,993" in doc


def test_hub_3d_lite_fallback(tmp_path):
    run = tmp_path / "run_b"
    _make_synth_run(run, ["viewer3d_lite.html"], summary={"n_agents": 10})
    _out, artifacts = mh.make_hub(run)
    three = next(a for a in artifacts if a["label"] == "3Dビュー")
    assert three["rel"] == "viewer3d_lite.html"


def test_hub_no_artifacts(tmp_path):
    run = tmp_path / "run_c"
    run.mkdir()
    out_path, artifacts = mh.make_hub(run)
    assert artifacts == []
    assert out_path.exists()  # 空でも hub.html は書く
    assert "成果物がありません" in out_path.read_text(encoding="utf-8")
