"""定数の毎分レート化(Δt 不変化)= 第79バッチのテスト。

正典: docs/plans/source/cognition-design-record.md §5.2-5.3 / cognition-physics-plan.md §4。

守るもの(重要度順):
  (1) **既定 run.dt_min=10 は 1 バイトも変えない**。config は同一オブジェクトが素通しされ、
      L1 イベント列は明示 dt_min=10 と純粋既定で完全一致(golden 本体は test_scenario.py)。
  (2) 分類の数学: レート=線形 / 確率=べき変換(ハザード保存) / step 長=逆比例 / 不変。
      **確率を線形スケールしない**ことを合成テストで固定する。
  (3) 網羅性: conf/config.yaml の「1 step 前提」を示唆する行のキーが全て
      timeconv.TABLE に載っている(変換済み or 不変+理由)。載っていなければ落ちる。
  (4) Δt≠10 のランが完走し、1 日あたりの主要統計が同オーダーに収まる(T3 の緩い版)。
      **厳密一致は要求しない**(乱数消費列が変わる = 別世界として正当)。
"""
from __future__ import annotations

import json
import re

import pytest
from omegaconf import OmegaConf

from society import registry as R
from society import timeconv as T
from society.config import DEFAULT_CONFIG, load_config
from society.engine.simulation import Simulation
from society.world.clock import Clock, STEP_MINUTES, STEPS_PER_DAY


# --------------------------------------------------------------------------- #
def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name, n_steps=24, n_agents=8, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    summary = sim.run()
    return sim, summary


# --------------------------------------------------------------------------- #
# (1) 分類の数学
# --------------------------------------------------------------------------- #
def test_helpers_are_identity_at_canonical_dt():
    """Δt=10 では全ヘルパが恒等(浮動小数演算を 1 回も通さない)。"""
    assert T.CANON_DT_MIN == 10
    for v in (0.02, 0.35, 0.9983, 800.0, 0.0, 1.0):
        assert T.scale_rate(v, 10) is v
        assert T.scale_prob(v, 10) is v
        assert T.scale_keep(v, 10) is v
    for n in (0, 1, 3, 144, 288):
        assert T.scale_steps(n, 10) == n
    assert T.steps_per_day(10) == 144 and T.steps_per_hour(10) == 6
    assert T.step_seconds(10) == 600.0


def test_rate_is_linear():
    assert T.scale_rate(800.0, 5) == pytest.approx(400.0)
    assert T.scale_rate(800.0, 1) == pytest.approx(80.0)
    assert T.scale_rate(800.0, 20) == pytest.approx(1600.0)


def test_prob_is_power_transformed_not_linear():
    """p=0.35 を Δt=1 にすると 0.035(線形)ではない。線形は 10 step 合成で 0.30 に落ちる。"""
    p10 = 0.35
    p1 = T.scale_prob(p10, 1)
    assert p1 != pytest.approx(p10 / 10.0), "確率を線形スケールしている"
    assert p1 == pytest.approx(1.0 - (1.0 - p10) ** 0.1)
    linear_composed = 1.0 - (1.0 - p10 / 10.0) ** 10
    assert linear_composed == pytest.approx(0.2997, abs=1e-3)   # 0.35 に戻らない
    assert 1.0 - (1.0 - p1) ** 10 == pytest.approx(p10, abs=1e-12)  # べき変換は戻る


@pytest.mark.parametrize("dt", [1, 2, 5, 20])
@pytest.mark.parametrize("p", [0.02, 0.35, 0.8, 0.69])
def test_prob_composes_back_to_the_canonical_probability(dt, p):
    """細/粗どちらへ刻み直しても『10 分あたり少なくとも1回』の確率が保存される。"""
    q = T.scale_prob(p, dt)
    n = 10.0 / dt
    assert 1.0 - (1.0 - q) ** n == pytest.approx(p, abs=1e-12)


def test_prob_never_exceeds_one_when_coarsening():
    """Δt を粗くしても確率が 1 を超えない(線形なら 0.35×6=2.1 で破綻する)。"""
    assert 0.0 <= T.scale_prob(0.35, 60) <= 1.0
    assert T.scale_prob(0.0, 60) == 0.0 and T.scale_prob(1.0, 60) == 1.0


def test_keep_is_power_transformed():
    k = 0.9983
    assert T.scale_keep(k, 5) == pytest.approx(k ** 0.5)
    assert T.scale_keep(k, 5) ** 2 == pytest.approx(k, abs=1e-12)


def test_steps_are_inverse_and_preserve_zero():
    assert T.scale_steps(6, 5) == 12 and T.scale_steps(6, 1) == 60
    assert T.scale_steps(0, 1) == 0, "0(=無効)は 1 に繰り上げない"
    assert T.scale_steps(1, 20) == 1, "丸めで 0 にせず最低 1 step を残す"


# --------------------------------------------------------------------------- #
# (2) 中央 Δt の受け口
# --------------------------------------------------------------------------- #
def test_shipped_config_ships_canonical_dt():
    cfg = load_config()
    assert int(cfg.run.dt_min) == 10, "出荷既定が正準 Δt でない"
    assert T.dt_of(cfg) == 10


def test_dt_of_defaults_to_canonical_for_legacy_config():
    """dt_min キーを持たない旧 config でも正準 10 に落ちる(後方互換)。"""
    assert T.dt_of(OmegaConf.create({"run": {"seed": 1}})) == 10
    assert T.dt_of(OmegaConf.create({})) == 10


@pytest.mark.parametrize("bad", [0, -5, 7, 11, 13])
def test_dt_must_be_a_positive_divisor_of_1440(bad):
    """日境界が step 境界に落ちない Δt は拒否する(日次機構が壊れるため)。"""
    with pytest.raises(ValueError):
        T.dt_of(OmegaConf.create({"run": {"dt_min": bad}}))


def test_registry_declares_dt_min_as_non_bool_toggle():
    f = R.BY_ID["run.dt_min"]
    assert f.off_value == 10, "off 値(=正準・golden の世界)が 10 でない"
    assert f.repro_tier == "strict" and f.affects_k is False
    assert not R.is_enabled(f, 10) and R.is_enabled(f, 5)


# --------------------------------------------------------------------------- #
# (3) 既定 Δt=10 は 1 バイトも変えない(最重要)
# --------------------------------------------------------------------------- #
def test_apply_dt_is_a_pure_noop_at_canonical_dt():
    """同一オブジェクトを返し、中身も完全一致(恒等式でも浮動小数を通さない)。"""
    cfg = load_config()
    before = OmegaConf.to_container(cfg, resolve=True)
    out = T.apply_dt(cfg)
    assert out is cfg
    assert OmegaConf.to_container(out, resolve=True) == before


def test_explicit_dt10_is_byte_identical_to_pure_default(tmp_path):
    """run.dt_min=10 を明示的に書いても L1 バイト一致(= golden を守る)。"""
    a, _ = _run(tmp_path, "dt_base")
    b, _ = _run(tmp_path, "dt_ten", **{"run.dt_min": 10})
    assert _l1(a) == _l1(b)


def test_dt10_config_snapshot_equals_default_snapshot(tmp_path):
    """保存 config も一致する(= 過去ランとの config 突合が壊れない)。"""
    a = OmegaConf.to_container(load_config(["run.seed=7"]), resolve=True)
    b = OmegaConf.to_container(load_config(["run.seed=7", "run.dt_min=10"]),
                               resolve=True)
    assert a == b


def test_clock_defaults_match_the_module_constants():
    c = Clock(start_min=420)
    assert c.step_minutes == STEP_MINUTES
    assert c.steps_per_day == STEPS_PER_DAY == 144
    assert c.steps_per_hour == 6 and c.step_seconds == 600.0
    assert [c.sim_min(s) for s in (0, 1, 144)] == [420, 430, 1860]
    assert c.dur_steps(3) == 3 and c.min_to_steps(45) == 4


def test_clock_follows_dt():
    c = Clock(start_min=420, step_minutes=5)
    assert c.steps_per_day == 288 and c.steps_per_hour == 12
    assert c.step_seconds == 300.0
    assert c.sim_min(2) == 430, "2 step × 5 分 = 10 分後"
    assert c.dur_steps(3) == 6, "30 分の滞在は Δt=5 では 6 step"
    assert c.min_to_steps(45) == 9


# --------------------------------------------------------------------------- #
# (4) 分類テーブルの健全性と網羅性
# --------------------------------------------------------------------------- #
def test_table_is_well_formed():
    keys = [k for k, _c, _w in T.TABLE]
    assert len(keys) == len(set(keys)), "テーブルにキーの重複がある"
    for key, cls, why in T.TABLE:
        assert cls in T.CLASSES, f"{key}: 未知の分類 {cls}"
        assert why.strip(), f"{key}: 理由が空(不変と判断した理由は必須)"


def test_every_table_key_exists_in_shipped_config():
    """死んだ宣言の検出(conf に無いキーを分類しても意味がない)。"""
    cfg = load_config()
    dead = [pat for pat, _c, _w in T.TABLE
            if not list(T._iter_targets(cfg, pat))]
    assert dead == [], f"conf に存在しないキーを分類している: {dead}"


# 「1 step = 10 分」を前提にしていることを示唆する行(= 棚卸し grep)
_INVENTORY_RE = re.compile(
    r"per_step|/step|毎 ?step|step ?あたり|1 ?step|1step|_steps\b|_step\b|10分|10 分|per-step")


def _inventory_keys() -> list[str]:
    """conf/config.yaml を棚卸し正規表現で走査し、ヒットした行のドットパスを返す。"""
    lines = DEFAULT_CONFIG.read_text(encoding="utf-8").split("\n")
    stack: list[tuple[int, str]] = []
    hits: list[str] = []
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w.]*)\s*:(.*)$", ln)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _i, k in stack] + [key])
        if rest.strip() and not rest.strip().startswith("#"):
            if _INVENTORY_RE.search(ln):
                hits.append(dotted)
        else:
            stack.append((indent, key))
    return hits


def test_inventory_grep_is_not_empty():
    """走査そのものが壊れていないことの自己点検(0 件なら網羅テストが空回りする)。"""
    assert len(_inventory_keys()) >= 50


def test_every_inventory_hit_is_classified():
    """★検収基準(3): 棚卸し grep の全ヒットが『変換済み / 不変(理由つき)』に載る。"""
    missing = sorted({k for k in _inventory_keys() if not T.covers(k)})
    assert missing == [], (
        "timeconv.TABLE に分類が無いキーがある(変換するか、理由つきで不変と宣言する): "
        + ", ".join(missing))


def test_wildcard_classification_resolves():
    assert T.classify("world.modes.speeds.walk") == (T.RATE, "移動距離 m/step")
    assert T.classify("services.services.gym.stay")[0] == T.STEPS
    assert T.classify("routine.stochastic.interrupt_prob.mandatory")[0] == T.PROB
    assert T.classify("no.such.key") is None


# --------------------------------------------------------------------------- #
# (5) Δt≠10 の変換結果
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt", [5, 1])
def test_config_transform_follows_the_table(dt):
    base = load_config()
    cfg = load_config([f"run.dt_min={dt}"])
    r = dt / 10.0
    # RATE: 線形
    assert float(cfg.world.modes.speeds.walk) == pytest.approx(
        float(base.world.modes.speeds.walk) * r)
    # PROB: べき変換(線形ではない)
    assert float(cfg.world.meal_prob) == pytest.approx(
        1.0 - (1.0 - float(base.world.meal_prob)) ** r)
    assert float(cfg.world.meal_prob) != pytest.approx(float(base.world.meal_prob) * r)
    # STEPS: 逆比例
    assert int(cfg.drive.refractory_steps) == int(base.drive.refractory_steps) * (10 // dt)
    assert list(cfg.world.outside_steps) == [v * (10 // dt) for v in base.world.outside_steps]
    # INVARIANT: 触らない
    assert int(cfg.run.n_steps) == int(base.run.n_steps)
    assert float(cfg.chance.daily_rate) == float(base.chance.daily_rate)
    assert float(cfg.career.layoff_prob) == float(base.career.layoff_prob)


def test_int_keys_stay_int_after_transform():
    """int で書かれた予算・速度は int のまま(下流の int() 前提を壊さない)。"""
    cfg = load_config(["run.dt_min=5"])
    for key in ("lod.max_llm_per_step", "world.traffic.free_speed_min",
                "delivery.eta_m_per_step"):
        v = OmegaConf.select(cfg, key)
        assert isinstance(v, int) and not isinstance(v, bool), f"{key} が int でない: {v!r}"


# --------------------------------------------------------------------------- #
# (6) Δt≠10 のスモーク(完走 + 統計が同オーダー)
# --------------------------------------------------------------------------- #
def _per_day(sim, dt: int, n_steps: int) -> dict:
    days = n_steps * dt / 1440.0
    kinds: dict[str, int] = {}
    for e in sim.logger.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {
        "move": (kinds.get("arrive", 0) + kinds.get("route_start", 0)) / days,
        "arrive": kinds.get("arrive", 0) / days,
        "fires": (kinds.get("llm_deliberate", 0)
                  + kinds.get("drive_request", 0)) / days,
    }


@pytest.mark.parametrize("dt", [5, 1])
def test_finer_dt_runs_to_completion(tmp_path, dt):
    """240 分ぶん(= 既定 24 step 相当)を Δt=5/1 で回し切る。"""
    n_steps = 240 // dt
    sim, summary = _run(tmp_path, f"dt_smoke{dt}", n_steps=n_steps,
                        **{"run.dt_min": dt})
    assert summary["n_events"] > 0
    assert len(sim.agents) == 8
    assert all(a.node is not None for a in sim.agents)
    assert all(a.money == a.money for a in sim.agents)      # NaN 混入なし
    assert sim.clock.step_minutes == dt
    assert sim.clock.sim_min(n_steps) == sim.clock.start_min + 240


@pytest.mark.parametrize("dt", [5, 1])
def test_daily_statistics_stay_within_the_same_order(tmp_path, dt):
    """★検収基準(2): 1 日あたりの移動・発火が Δt=10 と同オーダー(厳密一致は要求しない)。

    許容は 1/5〜5 倍(= 同オーダー)。乱数消費列が変わるので一致は原理的に望めない。
    """
    base, _ = _run(tmp_path, f"dt_stat_base{dt}", n_steps=144, n_agents=8,
                   **{"run.dt_min": 10})
    n_steps = 1440 // dt
    fine, _ = _run(tmp_path, f"dt_stat_fine{dt}", n_steps=n_steps, n_agents=8,
                   **{"run.dt_min": dt})
    b = _per_day(base, 10, 144)
    f = _per_day(fine, dt, n_steps)
    for key in ("move", "arrive", "fires"):
        assert b[key] > 0, f"基準ランで {key} が 0(比較が成立しない)"
        ratio = f[key] / b[key]
        assert 0.2 <= ratio <= 5.0, (
            f"Δt={dt} の 1 日あたり {key} が桁違い: {f[key]:.1f} vs {b[key]:.1f} "
            f"(×{ratio:.2f})")


def test_resume_is_consistent_under_finer_dt(tmp_path):
    """Δt≠10 でも「一気」と「途中再開」が一致する(保存 config 経由で Δt が復元される)。"""
    import pyarrow.parquet as pq

    from society.engine import checkpoint, scheduler

    def cfg(name, n_steps):
        return load_config(["run.seed=42", "run.n_agents=12", "run.dt_min=5",
                            f"run.n_steps={n_steps}", f"run.name={name}",
                            "model.backend=mock", "observer.checkpoint_every=20"])

    def rows(d, stem="l1_events"):
        return pq.read_table(d / f"{stem}.parquet").to_pylist()

    straight = tmp_path / "dt5_straight"
    Simulation(cfg("dt5_straight", 40), out_dir=straight).run()

    resumed = tmp_path / "dt5_resumed"
    sim1 = Simulation(cfg("dt5_resumed", 20), out_dir=resumed)
    assert sim1.clock.step_minutes == 5
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, resumed / "checkpoint" / "ckpt-000020.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(cfg("dt5_resumed", 40), out_dir=resumed)
    sim2.run(resume_from=resumed)
    assert sim2.clock.step_minutes == 5

    assert rows(straight) == rows(resumed), "Δt=5 で resume が一致しない"


def test_finer_dt_keeps_sleep_and_stay_in_wall_clock_minutes(tmp_path):
    """step 数で書かれた持続長が『分』として保存される(Δt=5 で睡眠が半分にならない)。"""
    base, _ = _run(tmp_path, "dt_sleep10", n_steps=1, **{"run.dt_min": 10})
    fine, _ = _run(tmp_path, "dt_sleep5", n_steps=1, **{"run.dt_min": 5})
    b = sorted(a.sleep_steps for a in base.agents)
    f = sorted(a.sleep_steps for a in fine.agents)
    assert f == [v * 2 for v in b], "sleep_steps が Δt に追従していない"
    assert all(390 <= v * 10 <= 490 for v in b)        # 6.5〜8h の帯
