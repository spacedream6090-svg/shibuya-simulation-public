"""scripts/estimate_substeps.py の検証(竹-4 残⑦ = 物理サブステップの事前見積)。

方針:
- 純関数(n_sub_per_step / substeps_from_busy / mg_inf_duty / duty_scale / wall_seconds)は
  physics.py の式そのものと**同じ値**になることを固定する。
- 実測モードは **合成 L1** で検算し、さらに**物理 ON の mock スモークを実際に回して**
  `physics.continuity(sim)["sub_steps_total"]`(= 真値)と突き合わせる。
  これが「payload の読み方(wait_s / waited_steps / dwell_s)が正しい」ことの唯一の証拠。
- 実 LLM は使わない(model.backend=mock・8 step)。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "estimate_substeps", REPO_ROOT / "scripts" / "estimate_substeps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ES = _load()
STEP_S = 600.0


# =========================================================================== #
# (1) 純関数 — physics.py の式と一致する
# =========================================================================== #
def test_n_sub_per_step_matches_physics_formula():
    """n_sub = min(max_sub_steps, max(1, round(step_seconds/dt_sub)))(physics.py:331)。"""
    from society.world.zones import ZONE_DEFAULTS
    dt = float(ZONE_DEFAULTS["dt_sub"])
    cap = int(ZONE_DEFAULTS["max_sub_steps"])
    assert ES.n_sub_per_step(dt, cap, 600.0) == 12000          # 既定 = 600s / 0.05s
    # 上限が binding するのは Δt > 10 分のとき(zones.py の警告コメントと同じ現象)
    assert ES.n_sub_per_step(0.05, 12000, 1200.0) == 12000     # 24000 要るのに 12000 で止まる
    assert ES.n_sub_per_step(0.05, 12000, 60.0) == 1200        # Δt=1分 なら上限に届かない
    assert ES.n_sub_per_step(0.1, 12000, 600.0) == 6000
    assert ES.n_sub_per_step(0.05, 3, 600.0) == 3
    assert ES.n_sub_per_step(1000.0, 12000, 600.0) == 1        # 最低 1 刻み


def test_substeps_from_busy_is_ceiling_and_capped():
    """食い込んだ刻みは 1 つ数える(ceil)。上限で必ず止まる。空なら 0。"""
    assert ES.substeps_from_busy(0.0, 0.05, 12000) == 0
    assert ES.substeps_from_busy(-1.0, 0.05, 12000) == 0
    assert ES.substeps_from_busy(0.05, 0.05, 12000) == 1        # ちょうどは 1(丸め誤差に強い)
    assert ES.substeps_from_busy(0.06, 0.05, 12000) == 2
    assert ES.substeps_from_busy(600.0, 0.05, 12000) == 12000
    assert ES.substeps_from_busy(9999.0, 0.05, 12000) == 12000  # 上限で頭打ち


def test_duty_scale_is_the_mg_inf_law():
    """duty_scale(duty, r) は「到着率を r 倍したときの duty」と厳密に一致する。"""
    lam, tau = 0.02, 40.0
    base = ES.mg_inf_duty(lam, tau)
    for r in (0.5, 1.0, 2.0, 7.0, 100.0):
        assert ES.duty_scale(base, r) == pytest.approx(ES.mg_inf_duty(lam * r, tau), rel=1e-9)
    assert ES.duty_scale(base, 1.0) == pytest.approx(base)
    assert ES.duty_scale(0.0, 5.0) == 0.0                      # 誰も来ないゾーンは増えない
    assert ES.duty_scale(1.0, 0.5) == 1.0                      # 飽和は飽和のまま
    # 体数を増やすほど 1 へ張り付く(= 総サブステップは飽和する)
    assert ES.duty_scale(base, 1000.0) > 0.999


def test_mg_inf_duty_edges():
    assert ES.mg_inf_duty(0.0, 100.0) == 0.0
    assert ES.mg_inf_duty(1.0, 0.0) == 0.0
    assert ES.mg_inf_duty(1e6, 1e6) == pytest.approx(1.0)      # overflow しない


def test_bench_anchor_reproduces_measured_throughput():
    """ベンチ由来の係数は「n_ref 体での実測 agent·step/s」を再現する(定義の自己整合)。"""
    for eng, per_s in ES.BENCH_AGENT_SUBSTEPS_PER_S.items():
        coef = ES.cost_coefficients(eng, "quadratic")
        sec = ES.wall_seconds(1.0, ES.BENCH_N_REF, coef)       # 1 刻み・n_ref 体の秒
        assert ES.BENCH_N_REF / sec == pytest.approx(per_s, rel=1e-9)
    lin = ES.cost_coefficients("sfm", "linear")
    assert ES.wall_seconds(1.0, 10.0, lin) == pytest.approx(
        10.0 / ES.BENCH_AGENT_SUBSTEPS_PER_S["sfm"])


def test_measured_cost_wins_over_bench_anchor():
    m = {"sfm": {"a": 1e-5, "b": 2e-7, "c1": 0.0, "model": "measured"}}
    coef = ES.cost_coefficients("sfm", "quadratic", measured=m)
    assert coef["model"] == "measured"
    assert ES.wall_seconds(100.0, 10.0, coef) == pytest.approx(100 * (1e-5 + 2e-7 * 100))


def test_diurnal_weights_average_to_one():
    w = ES.diurnal_weights(144, 18.0, 0.1, 10)
    assert len(w) == 144
    assert sum(w) / len(w) == pytest.approx(1.0)
    assert max(w) > min(w)                                     # 夜と昼で差がある


# =========================================================================== #
# (2) 実測モード — 合成 L1 で検算
# =========================================================================== #
def _write_l1(run_dir: Path, rows: list) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")
    return run_dir


def _ev(step, aid, kind, payload):
    return {"step": step, "sim_min": 420 + step * 10, "agent_id": aid, "kind": kind,
            "x": 0.0, "y": 0.0, "payload": json.dumps(payload, ensure_ascii=False),
            "rng_stream": "", "llm_call_id": ""}


def test_measure_substeps_on_hand_computed_intervals():
    """在圏区間 → step ごとのサブステップ数を手計算と突き合わせる。"""
    dt, n_sub_max = 0.05, 12000
    # step0 の頭から 30 秒だけ在圏 / step1 は丸ごと在圏 / step2 は 0.06 秒だけ
    iv = [(0.0, 30.0), (600.0, 1200.0), (1200.0, 1200.06)]
    mm = ES.measure_substeps(iv, 3, dt, n_sub_max, STEP_S)
    assert mm["per_step"] == [600, 12000, 2]
    assert mm["substeps_total"] == 12602
    assert mm["busy_steps"] == 3
    assert mm["saturated_steps"] == 1
    # agent·サブステップ = 在圏秒 / dt(重なりは足し合わせる)
    assert mm["agent_substeps_total"] == pytest.approx((30.0 + 600.0 + 0.06) / dt)


def test_measure_substeps_takes_the_union_not_the_sum():
    """2 人が同時に居ても「ループが回る刻み」は 1 回ぶん(= 最後に抜けた人まで)。"""
    dt, n_sub_max = 0.05, 12000
    mm = ES.measure_substeps([(0.0, 10.0), (0.0, 25.0)], 1, dt, n_sub_max, STEP_S)
    assert mm["per_step"] == [500]                              # 25s / 0.05
    assert mm["agent_substeps_total"] == pytest.approx((10.0 + 25.0) / dt)


def test_scan_zone_gates_uses_queue_start_and_dwell(tmp_path):
    """起点 = 待ち行列に並んだ step の頭 / 終点 = 入場時刻 + dwell_s。"""
    rows = [
        # step2 に入場。1 step 待った(= step1 の頭からループは回っている)。
        _ev(2, 7, "zone_gate", {"zone": "z1", "dir": "enter", "wait_s": 612.5,
                                "waited_steps": 1}),
        _ev(3, 7, "zone_gate", {"zone": "z1", "dir": "exit", "dwell_s": 40.0,
                                "reason": "gate"}),
        _ev(3, 9, "spend", {"amount": 1}),                      # 無関係な kind
    ]
    _write_l1(tmp_path / "r", rows)
    scan = ES.scan_zone_gates(tmp_path / "r", STEP_S)
    assert scan["n_enter"] == 1 and scan["n_exit"] == 1 and scan["n_unclosed"] == 0
    (t0, t1), = scan["intervals"]["z1"]
    assert t0 == pytest.approx(1 * STEP_S)                      # 並び始めた step1 の頭
    assert t1 == pytest.approx(2 * STEP_S + 12.5 + 40.0)        # 入場 + dwell


def test_scan_zone_gates_keeps_unclosed_until_run_end(tmp_path):
    """exit を見ないまま終わった個体はラン終端まで在圏とみなす(捏造せず件数を返す)。"""
    rows = [_ev(1, 3, "zone_gate", {"zone": "z1", "dir": "enter", "wait_s": 0.0,
                                    "waited_steps": 0}),
            _ev(5, 3, "arrive", {})]
    _write_l1(tmp_path / "r", rows)
    scan = ES.scan_zone_gates(tmp_path / "r", STEP_S)
    assert scan["n_unclosed"] == 1
    (t0, t1), = scan["intervals"]["z1"]
    assert (t0, t1) == (1 * STEP_S, 6 * STEP_S)                 # max_step=5 → 終端 6*600


def test_run_without_zone_gate_is_reported_not_faked(tmp_path):
    """物理 OFF のラン(zone_gate 0 件)では実測モードが空を返す(ゼロを捏造しない)。"""
    _write_l1(tmp_path / "r", [_ev(0, 1, "arrive", {"node": "n1"})])
    rep = ES.measured_estimate(tmp_path / "r", [], agents=100, days=10, dt_min=10,
                               calib_agents=10, cost_model="quadratic")
    assert rep["zones"] == []
    assert rep["substeps_total"] == 0
    assert "measured" == rep["mode"]
    assert ES.render_report(rep)                                # 落ちずに描ける


# =========================================================================== #
# (3) 理論モード
# =========================================================================== #
def _zone(zid="z1", engine="sfm", side=90.0):
    h = side / 2.0
    return {"id": zid, "engine": engine, "dt_sub": 0.05, "max_sub_steps": 12000,
            "area_m2": side * side, "polygon_area_m2": side * side,
            "has_signal": False, "source": "test"}


def test_theory_estimate_never_exceeds_the_upper_bound():
    w = ES.diurnal_weights(144, 18.0, 0.1, 10)
    rep = ES.theory_estimate([_zone()], agents=10000, days=10, dt_min=10, weights=w,
                             traversals_per_day=2.0, zone_share=0.5, walk_speed=1.34,
                             span_m=None, cost_model="quadratic")
    z = rep["zones"][0]
    assert z["upper_bound_substeps"] == 12000 * 1440
    assert 0 < rep["substeps_total"] <= rep["upper_bound_substeps"]
    assert 0.0 <= z["duty_mean"] <= z["duty_peak"] <= 1.0


def test_theory_substeps_saturate_but_work_keeps_growing():
    """★本ツールの主張: 体数を増やすと総サブステップは飽和し、agent·サブステップだけ伸びる。"""
    w = ES.diurnal_weights(144, 18.0, 0.1, 10)
    kw = dict(days=1, dt_min=10, weights=w, traversals_per_day=2.0, zone_share=0.5,
              walk_speed=1.34, span_m=None, cost_model="quadratic")
    small = ES.theory_estimate([_zone()], agents=2000, **kw)
    big = ES.theory_estimate([_zone()], agents=200000, **kw)
    ub = small["upper_bound_substeps"]
    assert big["substeps_total"] / ub > 0.99                     # 飽和した
    assert big["substeps_total"] / small["substeps_total"] < 3.0  # サブステップは伸びない
    ratio = big["agent_substeps_total"] / small["agent_substeps_total"]
    assert ratio == pytest.approx(100.0, rel=1e-6)               # 演算量は体数に比例
    assert big["wall_s_total"] > small["wall_s_total"]


def test_zero_zones_means_zero_substeps():
    rep = ES.theory_estimate([], agents=250000, days=10, dt_min=10,
                             weights=[1.0] * 144, traversals_per_day=2.0,
                             zone_share=0.5, walk_speed=1.34, span_m=None,
                             cost_model="quadratic")
    assert rep["substeps_total"] == 0 and rep["zones"] == []
    assert "ゾーンが 0 件" in ES.render_report(rep)


def test_report_always_prints_the_assumptions():
    """仮定を隠さない = レポートに式が必ず出る(指示の受入条件)。"""
    w = ES.diurnal_weights(144, 18.0, 0.1, 10)
    rep = ES.theory_estimate([_zone()], agents=10000, days=10, dt_min=10, weights=w,
                             traversals_per_day=2.0, zone_share=0.5, walk_speed=1.34,
                             span_m=None, cost_model="quadratic")
    text = ES.render_report(rep)
    assert "[仮定と式" in text
    assert "M/G/∞" in text
    assert "根拠のない既定" in text
    assert all(a in text for a in rep["assumptions"])


def test_extrapolation_warnings_flag_the_directions():
    rep = {"mode": "measured", "agents": 10000, "calib_agents": 30, "calib_days": 0.17,
           "n_unclosed": 2,
           "zones": [{"id": "z1", "mean_occupancy": 480.0,
                      "substeps": 100.0, "upper_bound_substeps": 100}]}
    warns = " / ".join(ES.extrapolation_warnings(rep))
    assert "1 日未満" in warns
    assert "体数外挿" in warns
    assert "exit を見ていない" in warns
    assert "飽和" in warns


# =========================================================================== #
# (4) conf 読み・CLI
# =========================================================================== #
def test_load_zones_reads_declaration_even_when_disabled(tmp_path):
    """zones_enabled=false でも宣言があれば「ON にしたら」を見積もる(注記つき)。"""
    p = tmp_path / "c.yaml"
    p.write_text("run:\n  dt_min: 10\nphysics:\n  zones_enabled: false\n  zones:\n"
                 "    - id: zz\n      engine: sfm\n      dt_sub: 0.04\n"
                 "      polygon: [[0,0],[10,0],[10,10],[0,10]]\n", encoding="utf-8")
    zones, notes = ES.load_zones(p)
    assert [z["id"] for z in zones] == ["zz"]
    assert zones[0]["dt_sub"] == 0.04
    assert zones[0]["polygon_area_m2"] == pytest.approx(100.0)
    assert any("zones_enabled" in n for n in notes)
    assert ES.dt_min_of(p, None) == 10


def test_load_zones_empty_declaration(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("physics:\n  zones_enabled: false\n  zones: []\n", encoding="utf-8")
    zones, notes = ES.load_zones(p)
    assert zones == [] and notes


def test_cli_runs_with_repo_default_conf(capsys):
    """既定 conf(ゾーン 0 件)でも落ちない。"""
    assert ES.main(["--days", "1", "--agents", "100"]) == 0
    out = capsys.readouterr().out
    assert "物理サブステップ事前見積" in out


def test_cli_writes_json(tmp_path):
    out = tmp_path / "sub.json"
    assert ES.main(["--days", "2", "--agents", "500", "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["reports"] and data["reports"][0]["mode"] == "theory"


def test_cli_missing_calib_is_a_note_not_a_crash(tmp_path, capsys):
    assert ES.main(["--days", "1", "--agents", "10", "--calib", str(tmp_path / "nope")]) == 0
    assert "l1_events.parquet が無い" in capsys.readouterr().out


# =========================================================================== #
# (5) ★真値突合: 物理 ON の mock スモークを実際に回す
# =========================================================================== #
@pytest.mark.xdist_group("physics_zone_smoke")
def test_measured_substeps_match_the_simulator_ground_truth(tmp_path):
    """L1 から復元したサブステップ数が `physics.continuity` の実カウンタと一致する。

    これが payload の読み方(enter の wait_s/waited_steps・exit の dwell_s)が
    正しいことの唯一の証拠。誤差は ceil の丸めぶん(稼働 step 数)までしか許さない。
    実 LLM は使わない(model.backend=mock・20体×8step)。
    """
    from omegaconf import OmegaConf

    from society import physics as P
    from society.config import load_config
    from society.engine.simulation import Simulation

    r = 25.0
    cfg = load_config(["run.seed=42", "run.n_agents=20", "run.n_steps=8",
                       "run.name=zone_smoke", "model.backend=mock",
                       "observer.snapshot_every=144"])
    OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
    cfg.physics.zones = [{"id": "z1", "engine": "orca", "dt_sub": 0.05,
                          "polygon": [[-r, -r], [r, -r], [r, r], [-r, r]]}]
    out = tmp_path / "zone_smoke"
    sim = Simulation(cfg, out_dir=out)
    sim.run()
    truth = int(P.continuity(sim)["sub_steps_total"])
    assert truth > 0, "テスト前提が崩れた(ゾーンを誰も通らない)"

    scan = ES.scan_zone_gates(out, sim.clock.step_seconds)
    assert scan["intervals"], "L1 に zone_gate が無い"
    total = 0
    slack = 0
    for zid, iv in scan["intervals"].items():
        mm = ES.measure_substeps(iv, scan["n_steps"], 0.05,
                                 ES.n_sub_per_step(0.05, 12000, sim.clock.step_seconds),
                                 sim.clock.step_seconds)
        total += mm["substeps_total"]
        slack += mm["busy_steps"]                # ceil の丸めは稼働 step あたり最大 1
    assert abs(total - truth) <= slack, (total, truth, slack)
    assert abs(total - truth) / truth < 0.01     # 実用上は 1% 未満で一致する


@pytest.mark.xdist_group("physics_zone_smoke")
def test_measure_engine_cost_fits_a_positive_quadratic():
    """このマシンでの費用実測は sec(n) = a + b·n²(a,b ≥ 0)に当たり、n で単調に増える。"""
    c = ES.measure_engine_cost("sfm", ns=(1, 16, 64), budget_s=0.01)
    assert c["model"] == "measured"
    assert c["a"] >= 0.0 and c["b"] > 0.0
    assert len(c["points"]) == 3
    coef = ES.cost_coefficients("sfm", "quadratic", measured={"sfm": c})
    assert ES.wall_seconds(1.0, 64.0, coef) > ES.wall_seconds(1.0, 1.0, coef)
    # ベンチ既定より小 n の固定費を持つ(= 過小評価にならない)
    bench = ES.cost_coefficients("sfm", "quadratic")
    assert ES.wall_seconds(1.0, 1.0, coef) > ES.wall_seconds(1.0, 1.0, bench)


def test_math_import_is_used_for_ceiling():
    """substeps_from_busy の ceil は浮動小数の 1 刻み手前で誤発火しない。"""
    for k in (1, 7, 120, 12000):
        assert ES.substeps_from_busy(k * 0.05, 0.05, 12000) == min(k, 12000)
    assert math.ceil(0.05 / 0.05) == 1
