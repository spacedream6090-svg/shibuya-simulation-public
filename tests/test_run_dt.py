"""W2-3: 解析スクリプトの Δt(run.dt_min)対応。

検収の柱は 3 本(タスク指示の①②③に対応):

  ① **Δt=10 の出力は 1 ビットも変わらない**。移行の絶対条件なので、変換式そのものを
     「W2-3 以前の直書き式」でテスト内に温存し、新実装と**厳密一致**させる
     (`_LEGACY_*` の各関数がその温存版。ここを緩めると移行の意味が消える)。
     さらに run dir 単位でも「dt_min: 10 を書いたラン」と「dt_min を持たない旧ラン」で
     **parquet がバイト一致**することを固定する(= 仮定経路も同値)。
  ② **Δt=1 のランで換算が正しい**。1 日 = 1440 step / 1 step = 1 分 / 1 時間 = 60 step。
     レート量(睡眠時間・滞在分・活動分)は step 数 ×1/10 で**実時間が保たれる**。
  ③ **dt 未記載ランは 10 を仮定し、その旨を告知する**(黙って仮定しない)。

正典: docs/research/obs-u2-dt1min-design.md §1.3(C 級)/ §4 B4。
実装: scripts/run_dt.py(解析側の Δt 単一の源)。
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import run_dt  # noqa: E402


# --------------------------------------------------------------------------- #
# W2-3 以前の直書き式(**温存**)。新実装はこれと Δt=10 で厳密一致しなければならない。
# --------------------------------------------------------------------------- #
_LEGACY_STEPS_PER_DAY = 144
_LEGACY_MIN_PER_STEP = 10
_LEGACY_STEPS_PER_HOUR = 6


def _legacy_day(step):                       # build_panel._day / commercial_report._day
    return step // _LEGACY_STEPS_PER_DAY


def _legacy_hour_of_step(step):              # build_panel._hour_of_step
    return ((420 + step * _LEGACY_MIN_PER_STEP) % 1440) // 60


def _legacy_hour_bin_of_step(step):          # analyze_od.hour_bin_of_step
    return (step % _LEGACY_STEPS_PER_DAY) // _LEGACY_STEPS_PER_HOUR


def _legacy_day_of(ev):                      # analyze_structure._day_of / analyze_imitation
    sm = ev.get("sim_min")
    if sm is None:
        return int(ev["step"]) // _LEGACY_STEPS_PER_DAY
    return int(sm) // 1440


def _legacy_sim_min(ev):                     # analyze_founders / audit_uncertainty の復元
    sm = ev.get("sim_min")
    return int(sm) if sm is not None else int(ev["step"]) * _LEGACY_MIN_PER_STEP


# --------------------------------------------------------------------------- #
# 合成ラン(run dir)
# --------------------------------------------------------------------------- #
def _write_run(tmp: Path, name: str, *, dt_min: int | None, n_steps: int,
               events: list[dict], agents: list[dict] | None = None,
               manifest_dt: int | None = None) -> Path:
    """最小のラン出力ディレクトリを作る(config.yaml / summary.json / agents.json / L1)。

    `dt_min=None` は **第79バッチ以前のラン**(config.yaml に run.dt_min が無い)を模す。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["run:", "  seed: 42", f"  n_agents: {len(agents or [])}",
             f"  n_steps: {n_steps}"]
    if dt_min is not None:
        lines.append(f"  dt_min: {dt_min}")
    lines += ["  start_tod: 07:00", "observer:", "  snapshot_every: 12", ""]
    (d / "config.yaml").write_text("\n".join(lines), encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps({"n_agents": len(agents or []), "n_steps": n_steps,
                    "n_events": len(events)}, ensure_ascii=False),
        encoding="utf-8")
    (d / "agents.json").write_text(json.dumps(agents or [], ensure_ascii=False),
                                   encoding="utf-8")
    if manifest_dt is not None:
        (d / "run_manifest.json").write_text(
            json.dumps({"run": {"dt_min": manifest_dt}}), encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["sim_min"]) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [float(e.get("x", 0.0)) for e in events],
        "y": [float(e.get("y", 0.0)) for e in events],
        "payload": [json.dumps(e.get("payload") or {}, ensure_ascii=False)
                    for e in events],
        "rng_stream": [None] * len(events),
        "llm_call_id": [None] * len(events),
    }
    schema = pa.schema([("step", pa.int64()), ("sim_min", pa.int64()),
                        ("agent_id", pa.int64()), ("kind", pa.string()),
                        ("x", pa.float64()), ("y", pa.float64()),
                        ("payload", pa.string()), ("rng_stream", pa.string()),
                        ("llm_call_id", pa.string())])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                   d / "l1_events.parquet")
    return d


def _life_events(dt_min: int) -> list[dict]:
    """同じ**実時間**の 1 日ぶんの生活イベント(Δt に応じて step/sim_min を作り直す)。

    07:00 起点 = sim_min 420。睡眠 8 時間(480 分)・在館 2 時間(120 分)。
    """
    spd = 1440 // dt_min
    per_h = 60 // dt_min

    def ev(kind, step, payload=None, aid=1):
        return {"step": step, "sim_min": 420 + step * dt_min, "agent_id": aid,
                "kind": kind, "x": 0.0, "y": 0.0, "payload": payload or {}}

    out = [
        # 起床(前夜の睡眠 8 時間 = 480 分ぶんの step 数)
        ev("wake_up", 1 * per_h, {"slept_steps": 8 * per_h}),
        ev("speak", 2 * per_h, {"text": "おはよう", "hearers": [2]}),
        ev("enter_building", 3 * per_h, {"building": "b1", "activity": "working"}),
        ev("exit_building", 5 * per_h, {"building": "b1"}),      # 在館 2 時間
        ev("spend", 6 * per_h, {"cat": "food", "amount": 800}),
        ev("wage", 7 * per_h, {"amount": 12000}),
    ]
    # 移動は **1 step ぶんの前進**を表す点イベントなので、同じ実時間(10 分)を表すには
    # Δt=1 では 10 件必要になる(距離の合計は同じ 500m)。ここを 1 件のままにすると
    # 「Δt=1 のほうが移動時間が 1/10」という **フィクスチャ側の人工物**を測ってしまう。
    n_move = max(1, 10 // dt_min)
    for j in range(n_move):
        out.append(ev("move_segment", 8 * per_h + j, {"dist_m": 500.0 / n_move}))
    # 翌日(= 1 日ぶん先)にも 1 件置いて日境界を跨がせる
    out.append(ev("speak", spd + 2 * per_h, {"text": "また明日", "hearers": [2]}))
    return out


_AGENTS = [{"id": 1, "name": "a1", "money": 1000, "work_building": "b1"},
           {"id": 2, "name": "a2", "money": 1000}]


# =========================================================================== #
# 0. run_dt そのもの
# =========================================================================== #
def test_canonical_constants_match_src():
    """正準値は src/society/timeconv.py と一致していなければならない(二重定義の腐敗検知)。"""
    from society import timeconv as tc
    assert run_dt.CANON_DT_MIN == tc.CANON_DT_MIN
    assert run_dt.MINUTES_PER_DAY == tc.MINUTES_PER_DAY
    assert run_dt.CANON_STEPS_PER_DAY == tc.steps_per_day(tc.CANON_DT_MIN) == 144
    assert run_dt.CANON_STEPS_PER_HOUR == tc.steps_per_hour(tc.CANON_DT_MIN) == 6


@pytest.mark.parametrize("dt", [1, 2, 5, 10, 30, 60])
def test_derived_quantities_match_timeconv(dt):
    """派生量の定義は timeconv / clock と同一(解析側だけ別の式にしない)。"""
    from society import timeconv as tc
    assert run_dt.steps_per_day(dt_min=dt) == tc.steps_per_day(dt)
    assert run_dt.steps_per_hour(dt_min=dt) == tc.steps_per_hour(dt)
    assert run_dt.step_seconds(dt_min=dt) == tc.step_seconds(dt)
    assert run_dt.min_per_step(dt_min=dt) == dt


def test_read_order_config_then_manifest_then_assumed(tmp_path):
    """config.yaml → run_manifest.json → 仮定 の順に読む。"""
    ev = _life_events(10)
    d1 = _write_run(tmp_path, "cfg", dt_min=5, n_steps=288, events=ev, agents=_AGENTS)
    assert run_dt.read_dt_min(d1) == (5, run_dt.SRC_CONFIG)

    # config に dt_min が無く manifest にある → manifest
    d2 = _write_run(tmp_path, "man", dt_min=None, n_steps=288, events=ev,
                    agents=_AGENTS, manifest_dt=2)
    assert run_dt.read_dt_min(d2) == (2, run_dt.SRC_MANIFEST)

    # どちらにも無い → 正準 10 の仮定
    d3 = _write_run(tmp_path, "none", dt_min=None, n_steps=288, events=ev,
                    agents=_AGENTS)
    assert run_dt.read_dt_min(d3) == (10, run_dt.SRC_ASSUMED)

    # ディレクトリ自体が無くても落ちない
    assert run_dt.read_dt_min(tmp_path / "missing") == (10, run_dt.SRC_ASSUMED)
    assert run_dt.read_dt_min(None) == (10, run_dt.SRC_ASSUMED)


def test_assumption_is_announced_not_silent(tmp_path):
    """③ dt 未記載ランでは **10 を仮定した旨を必ず告知**する(黙って仮定しない)。"""
    d = _write_run(tmp_path, "legacy_notify", dt_min=None, n_steps=144,
                   events=_life_events(10), agents=_AGENTS)
    run_dt._NOTIFIED.clear()
    buf = io.StringIO()
    assert run_dt.dt_min_of(d, stream=buf) == 10
    msg = buf.getvalue()
    assert msg, "仮定したのに何も告知していない"
    assert "run.dt_min" in msg and str(d.resolve()) in msg
    # 同じランについては 1 回だけ(解析 1 本が何度引いても静か)
    buf2 = io.StringIO()
    run_dt.dt_min_of(d, stream=buf2)
    assert buf2.getvalue() == ""


def test_no_announcement_when_dt_is_declared(tmp_path):
    """① 宣言済みのランでは 1 バイトも出さない(既存ランの標準出力を汚さない)。"""
    d = _write_run(tmp_path, "declared", dt_min=10, n_steps=144,
                   events=_life_events(10), agents=_AGENTS)
    run_dt._NOTIFIED.clear()
    buf = io.StringIO()
    assert run_dt.dt_min_of(d, stream=buf) == 10
    assert buf.getvalue() == ""


@pytest.mark.parametrize("bad", [0, -1, 7, 11, 100, "abc", None, True])
def test_invalid_dt_is_rejected(bad):
    """1440 の約数の正整数以外は採用しない(timeconv.dt_of と同じ規則)。"""
    assert run_dt.valid_dt_min(bad) is None


def test_dt_coercion_matches_timeconv():
    """int() への丸め方も timeconv.dt_of と同じ(解析側だけ別解釈にしない)。

    ★conf 側で run.dt_min は int 強制される(config._INT_KEYS)ので実運用では起きないが、
    「シムは 1 で回ったのに解析は 10 と読んだ」という取り違えを構造的に防ぐ。
    """
    from omegaconf import OmegaConf
    from society import timeconv as tc
    for raw in (1.5, "5", 30):
        cfg = OmegaConf.create({"run": {"dt_min": raw}})
        assert run_dt.valid_dt_min(raw) == tc.dt_of(cfg)


def test_dotlist_and_yaml_sources(tmp_path):
    assert run_dt.dt_min_from_dotlist(["run.seed=1", "run.dt_min=1"]) == 1
    assert run_dt.dt_min_from_dotlist(["run.dt_min=5", "run.dt_min=2"]) == 2  # 後勝ち
    assert run_dt.dt_min_from_dotlist(["run.dt_min=7"]) is None               # 不正は無視
    assert run_dt.dt_min_from_dotlist([]) is None
    # 基底 conf / プロファイルを直に読む
    assert run_dt.dt_min_from_yaml(_ROOT / "conf" / "config.yaml") == 10
    assert run_dt.dt_min_from_yaml(_ROOT / "conf" / "smoke_dt1.yaml") == 1
    assert run_dt.dt_min_from_yaml(tmp_path / "nope.yaml") is None


def test_day_helpers_prefer_sim_min():
    """day_of は sim_min//1440 を第一手段にする(= Δt に一切依存しない処方箋)。"""
    # sim_min があれば spd を何にしても同じ日
    ev = {"step": 999, "sim_min": 1440 + 30}
    assert run_dt.day_of(ev, 144) == run_dt.day_of(ev, 1440) == 1
    # sim_min が無いときだけ spd が効く
    assert run_dt.day_of({"step": 288}, 144) == 2
    assert run_dt.day_of({"step": 288}, 1440) == 0
    # activity_day は step 基準(開始時刻起点の活動日)
    assert run_dt.activity_day_of(287, 144) == 1
    assert run_dt.activity_day_of(2879, 1440) == 1
    # sim_min の復元
    assert run_dt.sim_min_of({"step": 5}, 10) == 50
    assert run_dt.sim_min_of({"step": 5}, 1) == 5
    assert run_dt.sim_min_of({"step": 5, "sim_min": 77}, 1) == 77


# =========================================================================== #
# 1. ① Δt=10 で「W2-3 以前の直書き式」と厳密一致(純関数)
# =========================================================================== #
@pytest.mark.parametrize("step", [0, 1, 5, 6, 47, 143, 144, 145, 287, 1000, 4321])
def test_pure_functions_identical_to_legacy_at_dt10(step):
    import analyze_od
    import analyze_structure
    import build_panel
    import commercial_report

    assert build_panel._day(step) == _legacy_day(step)
    assert build_panel._hour_of_step(step) == _legacy_hour_of_step(step)
    assert commercial_report._day(step) == _legacy_day(step)
    assert analyze_od.hour_bin_of_step(step) == _legacy_hour_bin_of_step(step)
    for ev in ({"step": step}, {"step": step, "sim_min": step * 10}):
        assert analyze_structure._day_of(ev) == _legacy_day_of(ev)
    assert run_dt.activity_day_of(step) == _legacy_day(step)


def test_module_constants_still_canonical_at_dt10():
    """モジュール定数は正準 Δt=10 の値のまま(既存 import 互換 = 従来と同じ数)。"""
    import analyze_bridging
    import analyze_communities
    import analyze_flows_grid
    import analyze_founders
    import analyze_imitation
    import analyze_mas_failures
    import analyze_od
    import analyze_structure
    import audit_uncertainty
    import build_panel
    import calibrate_report
    import commercial_report
    import detect_emergence
    import detect_regression
    import summarize_run

    for mod in (analyze_bridging, analyze_communities, analyze_flows_grid,
                analyze_founders, analyze_imitation, analyze_mas_failures,
                analyze_od, analyze_structure, audit_uncertainty, build_panel,
                calibrate_report, commercial_report, detect_emergence,
                detect_regression, summarize_run):
        assert mod.STEPS_PER_DAY == 144, mod.__name__
    assert build_panel.MIN_PER_STEP == 10
    assert calibrate_report.MIN_PER_STEP == 10
    assert analyze_od.STEPS_PER_HOUR == 6
    assert detect_emergence.WINDOW_STEPS == 24
    assert analyze_mas_failures.RESPONSE_WINDOW_STEPS == 6


def test_summarize_run_days_row_identical_to_legacy():
    import summarize_run
    summary = {"n_agents": 12, "n_steps": 432, "n_events": 100}
    rows = summarize_run._overview_rows(summary)
    legacy = [r for r in rows if r["label"].startswith("シミュレーション日数")]
    assert legacy and legacy[0]["num"] == 432 // _LEGACY_STEPS_PER_DAY == 3


def test_detect_regression_window_fallback_is_one_day(tmp_path):
    import detect_regression
    d = _write_run(tmp_path, "reg10", dt_min=10, n_steps=288,
                   events=_life_events(10), agents=_AGENTS)
    assert detect_regression.run_window_steps(d) == 144            # ① 従来値
    d1 = _write_run(tmp_path, "reg1", dt_min=1, n_steps=2880,
                    events=_life_events(1), agents=_AGENTS)
    assert detect_regression.run_window_steps(d1) == 1440          # ② 実時間 1 日を保つ


# =========================================================================== #
# 2. ① run dir 単位のバイト同値(宣言済み Δt=10 と「dt 未記載の旧ラン」)
# =========================================================================== #
def _panel_bytes(run_dir: Path) -> dict[str, bytes]:
    import build_panel
    build_panel.build(str(run_dir))
    out = run_dir / "panel"
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def test_build_panel_bytes_identical_declared_vs_legacy_run(tmp_path):
    """① dt_min: 10 を書いたランと、dt_min を持たない旧ランで **panel がバイト一致**。

    = 「読めたら使う / 読めなければ 10 を仮定」の両経路が Δt=10 で完全に同じ結果を出す。
    """
    # ★ラン名は parquet の run 列に入るので、両者で **同じ basename** にする
    #   (比べたいのは Δt の読み取り経路の差だけ)。
    ev = _life_events(10)
    a = _write_run(tmp_path / "declared", "same_run", dt_min=10, n_steps=288,
                   events=ev, agents=_AGENTS)
    b = _write_run(tmp_path / "legacy", "same_run", dt_min=None, n_steps=288,
                   events=ev, agents=_AGENTS)
    ba, bb = _panel_bytes(a), _panel_bytes(b)
    assert set(ba) == set(bb)
    for name in ba:
        assert ba[name] == bb[name], f"{name} が両経路でバイト一致しない"


def test_build_panel_dt10_values_match_legacy_formula(tmp_path):
    """① 具体値も直書き式どおり(睡眠 8h・労働 2h・活動日の切り方)。"""
    import pyarrow.parquet as pq

    d = _write_run(tmp_path, "vals10", dt_min=10, n_steps=288,
                   events=_life_events(10), agents=_AGENTS)
    import build_panel
    build_panel.build(str(d))
    t = pq.read_table(d / "panel" / "agent_day.parquet").to_pydict()
    rows = {(t["agent_id"][i], t["day"][i]): i for i in range(len(t["day"]))}
    i = rows[(1, 0)]
    assert t["sleep_h"][i] == pytest.approx(8.0)      # slept_steps 48 × 10 分
    assert t["work_h"][i] == pytest.approx(2.0)       # 在館 120 分
    assert t["wage_income"][i] == pytest.approx(12000.0)
    assert (1, 1) in rows                             # 翌日の発話が day=1 に落ちる


# =========================================================================== #
# 3. ② Δt=1 の合成ランで換算が正しい
# =========================================================================== #
def test_dt1_run_derived_quantities(tmp_path):
    d = _write_run(tmp_path, "dt1", dt_min=1, n_steps=1440,
                   events=_life_events(1), agents=_AGENTS)
    assert run_dt.dt_min_of(d) == 1
    assert run_dt.steps_per_day(d) == 1440
    assert run_dt.steps_per_hour(d) == 60
    assert run_dt.min_per_step(d) == 1
    assert run_dt.step_seconds(d) == 60.0
    assert run_dt.days_of(2880, d) == 2


def test_dt1_build_panel_preserves_real_time(tmp_path):
    """② 実時間が保たれる: Δt=1 でも睡眠 8h・労働 2h・同じ活動日に落ちる。

    step 数は 10 倍だが、レート量(分/時間)は ×1/10 されるので **値は同じ**。
    これが「Δt=10 ランとの比較が成立する」ことの意味そのもの。
    """
    import pyarrow.parquet as pq
    import build_panel

    d10 = _write_run(tmp_path, "cmp10", dt_min=10, n_steps=288,
                     events=_life_events(10), agents=_AGENTS)
    d1 = _write_run(tmp_path, "cmp1", dt_min=1, n_steps=2880,
                    events=_life_events(1), agents=_AGENTS)
    build_panel.build(str(d10))
    build_panel.build(str(d1))

    def _cols(d):
        t = pq.read_table(d / "panel" / "agent_day.parquet").to_pydict()
        return {(t["agent_id"][i], t["day"][i]):
                (t["sleep_h"][i], t["work_h"][i], t["wage_income"][i],
                 t["n_speak"][i])
                for i in range(len(t["day"]))}

    c10, c1 = _cols(d10), _cols(d1)
    assert set(c10) == set(c1), "活動日の切り方が Δt で変わってしまっている"
    for key in c10:
        assert c10[key] == pytest.approx(c1[key]), key


def test_dt1_time_budget_minutes_are_real_minutes(tmp_path):
    """② 時間バジェット(分/日)は step 数ではなく **実時間の分**で一致する。"""
    import pyarrow.parquet as pq
    import build_panel

    d10 = _write_run(tmp_path, "tb10", dt_min=10, n_steps=288,
                     events=_life_events(10), agents=_AGENTS)
    d1 = _write_run(tmp_path, "tb1", dt_min=1, n_steps=2880,
                    events=_life_events(1), agents=_AGENTS)
    build_panel.build(str(d10))
    build_panel.build(str(d1))

    def _mins(d):
        t = pq.read_table(d / "panel" / "time_budget.parquet").to_pydict()
        return {(t["agent_id"][i], t["category"][i]): t["minutes_total"][i]
                for i in range(len(t["category"]))}

    m10, m1 = _mins(d10), _mins(d1)
    assert m10, "Δt=10 側で活動が 1 件も復元できていない(テストの前提が壊れている)"
    assert set(m10) == set(m1)
    for k in m10:
        assert m10[k] == pytest.approx(m1[k]), k


def test_dt1_windows_scale_with_real_time(tmp_path):
    """② 「4 時間の窓」「60 分の猶予」「1 日の窓」は実時間で保たれる。"""
    import analyze_mas_failures
    import detect_emergence

    d10 = _write_run(tmp_path, "win10", dt_min=10, n_steps=288,
                     events=_life_events(10), agents=_AGENTS)
    d1 = _write_run(tmp_path, "win1", dt_min=1, n_steps=2880,
                    events=_life_events(1), agents=_AGENTS)
    assert detect_emergence.window_steps_of(d10) == 24            # ① 従来値
    assert detect_emergence.window_steps_of(d1) == 240            # ② ×10
    assert analyze_mas_failures.response_window_steps(d10) == 6   # ① 従来値
    assert analyze_mas_failures.response_window_steps(d1) == 60   # ② ×10


def test_dt1_analysis_entry_points_use_run_dt(tmp_path):
    """② 代表的な解析の入口が Δt=1 のランで 1440 step/日を使う。"""
    import analyze_bridging
    import analyze_od
    import analyze_structure

    d1 = _write_run(tmp_path, "ep1", dt_min=1, n_steps=2880,
                    events=_life_events(1), agents=_AGENTS)
    spd = run_dt.steps_per_day(d1)
    sph = run_dt.steps_per_hour(d1)
    assert (spd, sph) == (1440, 60)
    # hour_bin: Δt=1 では 60 step で 1 時間ビンが進む
    assert analyze_od.hour_bin_of_step(0, spd, sph) == 0
    assert analyze_od.hour_bin_of_step(60, spd, sph) == 1
    assert analyze_od.hour_bin_of_step(1440, spd, sph) == 0
    # 日の切り方(sim_min が無い後退経路)
    assert analyze_structure._day_of({"step": 1440}, spd) == 1
    assert analyze_structure._day_of({"step": 1439}, spd) == 0
    # transmission の日次集計
    trans = [{"step": 0, "channel": "sns", "dist_m": 10.0},
             {"step": 1440, "channel": "sns", "dist_m": 10.0}]
    res = analyze_bridging.summarize(trans, 500.0, spd)
    assert [r["day"] for r in res["daily"]] == [0, 1]
    res10 = analyze_bridging.summarize(trans, 500.0, 144)
    assert [r["day"] for r in res10["daily"]] == [0, 10]     # ① 従来(144 step/日)


def test_observe_stay_minutes_scale(tmp_path):
    """② 滞在 step → 分の換算が Δt に従う(実時間の滞在は同じ)。"""
    import observe

    mp = {"building_meta": {}, "poi_by_building": {}, "poi_by_node": {},
          "node_name": {}, "map_path": None, "n_pois": 0, "n_buildings": 0}

    def _ev(kind, step, dt, payload):
        return {"step": step, "sim_min": 420 + step * dt, "agent_id": 1,
                "kind": kind, "x": 0.0, "y": 0.0, "payload": payload}

    # 2 時間の在館
    e10 = [_ev("enter_building", 0, 10, {"building": "b1"}),
           _ev("exit_building", 12, 10, {"building": "b1"})]
    e1 = [_ev("enter_building", 0, 1, {"building": "b1"}),
          _ev("exit_building", 120, 1, {"building": "b1"})]
    r10 = observe.observe_visits(e10, mp, 10)
    r1 = observe.observe_visits(e1, mp, 1)
    b10 = r10["building_ranking"][0]["stay"]
    b1 = r1["building_ranking"][0]["stay"]
    assert b10["mean_stay_min"] == b1["mean_stay_min"] == 120.0


def test_calibrate_report_activity_minutes_scale():
    """② 生活時間配分(分/日)も 1 step の分数に従う。"""
    import calibrate_report as cr
    recon = {"adc": {(1, 0, "sleep"): 48}, "holiday_by_day": {0: False},
             "cats": ["sleep"]}
    a10 = cr.activity_summary(recon, 10)
    assert a10["overall"]["sleep"] == pytest.approx(480.0)     # ① 48 step × 10 分
    recon1 = {"adc": {(1, 0, "sleep"): 480}, "holiday_by_day": {0: False},
              "cats": ["sleep"]}
    a1 = cr.activity_summary(recon1, 1)
    assert a1["overall"]["sleep"] == pytest.approx(480.0)      # ② 480 step × 1 分


# =========================================================================== #
# 4. 直書きが戻ってこないことの構造的な歯止め
# =========================================================================== #
_MIGRATED = [
    "analyze_bridging.py", "analyze_communities.py", "analyze_endo_treatment.py",
    "analyze_flows_grid.py", "analyze_founders.py", "analyze_imitation.py",
    "analyze_luck.py", "analyze_mas_failures.py", "analyze_od.py",
    "analyze_org_form.py", "analyze_persona_consistency.py",
    "analyze_resolution.py", "analyze_structure.py", "audit_uncertainty.py",
    "build_panel.py", "calibrate_report.py", "commercial_report.py",
    "detect_emergence.py", "detect_regression.py", "measure_sigma.py",
    "observe.py", "summarize_run.py",
]


def test_migrated_scripts_import_run_dt():
    """移行済みスクリプトは必ず run_dt を経由する(144 を再び直書きさせない)。"""
    for name in _MIGRATED:
        src = (_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "import run_dt" in src, f"{name} が run_dt を import していない"


def test_export_3d_loads_run_dt_without_touching_sys_path():
    """export_3d は「scripts/ を sys.path に載せない」設計なので importlib 経由で借りる。

    (tracks_bin と同じ流儀。ここを `import run_dt` に変えると export_3d の
     ロード規約が壊れるので、経路そのものを固定しておく。)
    """
    src = (_ROOT / "scripts" / "export_3d.py").read_text(encoding="utf-8")
    assert "def _load_run_dt(" in src
    assert "import run_dt" not in src, "export_3d が素の import に変わっている(設計逸脱)"
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "sys.path.insert" in ln], \
        "export_3d が sys.path を触り始めている(設計逸脱)"


def test_frozen_spec_files_untouched_by_this_batch():
    """凍結 SPEC_FILES(解析側 4 本)は W2-3 では **1 バイトも触っていない**。

    触った瞬間 run_manifest の metrics_spec_hash が変わる = 事後の指標いじりと区別できない。
    Δt 直書きが残っているのは既知(報告済み)で、解除はユーザー判断(8/15)。
    """
    from society.observer import metrics_spec
    frozen = [f for f in metrics_spec.SPEC_FILES if f.startswith("scripts/")]
    assert frozen, "凍結リストに解析側ファイルが無い(前提が変わった)"
    for rel in frozen:
        src = (_ROOT / rel).read_text(encoding="utf-8")
        assert "run_dt" not in src, f"凍結ファイル {rel} に W2-3 の変更が入っている"


def test_run_dt_cli_reports_provenance(tmp_path):
    """CLI がラン 1 本の Δt と来歴を出せる(運用で「どっちの世界か」を即座に確かめる口)。"""
    d = _write_run(tmp_path, "cli", dt_min=1, n_steps=1440,
                   events=_life_events(1), agents=_AGENTS)
    r = subprocess.run([sys.executable, str(_ROOT / "scripts" / "run_dt.py"), str(d)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc == {"dt_min": 1, "source": "config.yaml", "assumed": False,
                   "steps_per_day": 1440, "steps_per_hour": 60, "min_per_step": 1}
