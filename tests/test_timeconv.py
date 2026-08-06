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

import ast
import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society import registry as R
from society import timeconv as T
from society.config import DEFAULT_CONFIG, load_config, save_config
from society.engine.simulation import Simulation
from society.world.clock import Clock, STEP_MINUTES, STEPS_PER_DAY

_REPO = Path(__file__).resolve().parents[1]


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
    assert f.repro_tier == "strict"
    assert not R.is_enabled(f, 10) and R.is_enabled(f, 5)


def test_registry_declares_dt_min_as_affects_k():
    """★第94バッチ OBS-U2 の訂正: run.dt_min は affects_k=True。

    affects_k の定義は registry.py:24-30 の「ON/OFF で LLM 呼び出しの発生箇所・本数が
    直接変わるか(= **予算を変える**か)」。Δt は RATE 類の lod.max_llm_per_step を
    Δt/10 倍する(Δt=1 で 300→30)ので、定義上 True でなければならない。
    実測でも呼数は ×2.2〜2.4(docs/research/obs-u2-dt1min-design.md §2(b))。"""
    f = R.BY_ID["run.dt_min"]
    assert f.affects_k is True, "Δt は LLM 予算(max_llm_per_step)を変える=affects_k"
    base = load_config()
    fine = load_config(["run.dt_min=1"])
    assert int(fine.lod.max_llm_per_step) * 10 == int(base.lod.max_llm_per_step), \
        "affects_k の根拠(予算が Δt に比例する)が失われている"


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


# 基底 conf に**実在しないことが仕様**の宣言(= 死んだ宣言ではない)。理由を必ず添える。
_ABSENT_BY_DESIGN: dict[str, str] = {
    # §1.2 B1(第94バッチ OBS-U2): world.scenario_params は基底が `{}`(baseline では読まれない)。
    # ラン側が dotlist/profile で渡した瞬間に実在キーになるので、宣言だけ先に置く。
    "world.scenario_params.at_step": "scenario 引数は基底 `{}`(ラン側が渡したときだけ実在)",
    "world.scenario_params.duration_steps": "同上",
}


def test_every_table_key_exists_in_shipped_config():
    """死んだ宣言の検出(conf に無いキーを分類しても意味がない)。

    例外は _ABSENT_BY_DESIGN のみ(理由つき)。scenario 引数のように「基底では空だが
    ラン側が渡す」キーは宣言しないと永久に変換の網から漏れるため、ここだけ許す。"""
    cfg = load_config()
    dead = [pat for pat, _c, _w in T.TABLE
            if not list(T._iter_targets(cfg, pat)) and pat not in _ABSENT_BY_DESIGN]
    assert dead == [], f"conf に存在しないキーを分類している: {dead}"


def test_absent_by_design_keys_are_declared_and_convert_when_present():
    """★§1.2 B1: 例外リストは (a) TABLE に載っている (b) 渡された瞬間に変換される。"""
    declared = {pat for pat, _c, _w in T.TABLE}
    for key in _ABSENT_BY_DESIGN:
        assert key in declared, f"{key} が TABLE から消えている(例外リストが陳腐化)"
        assert T.covers(key)
    # 実際にランが渡す形(dotlist)で入れると、Δt=1 では実時刻を保つよう逆比例する
    ov = ["world.scenario=shock_closure",
          "world.scenario_params={at_step: 36, duration_steps: 18, "
          "center: [0,0], radius_m: 150}"]
    base = load_config(ov)
    fine = load_config([*ov, "run.dt_min=1"])
    assert int(base.world.scenario_params.at_step) == 36          # Δt=10 は素通し
    assert int(base.world.scenario_params.duration_steps) == 18
    assert int(fine.world.scenario_params.at_step) == 360, "発動時点の実時刻が保たれていない"
    assert int(fine.world.scenario_params.duration_steps) == 180
    assert float(fine.world.scenario_params.radius_m) == 150.0, "距離を Δt で触ってはいけない"


# 「1 step = 10 分」を前提にしていることを示唆する行(= 棚卸し grep)
_INVENTORY_RE = re.compile(
    r"per_step|/step|毎 ?step|step ?あたり|1 ?step|1step|_steps\b|_step\b|10分|10 分|per-step")


def _scanned_conf_files() -> list[Path]:
    """棚卸し走査の対象ファイル(★§1.2 B2 = 第94バッチ OBS-U2 で拡張)。

    従来は conf/config.yaml だけだった。プロファイル固有の step 単位キー(観測間引き・
    窓・cooldown)は基底に無くても apply_dt の網に載らなければ Δt=1 で静かに壊れるので、
    **実際にランへ渡す全プロファイル**を走査対象にする。
    conf/experiments/ は除外 = `common:` / `arms:` の**実験行列スキーマ**であり config の
    ドットパスではない(common.run.n_steps は config キーではない)。"""
    conf = DEFAULT_CONFIG.parent
    return ([DEFAULT_CONFIG]
            + sorted(p for p in conf.glob("*.yaml") if p != DEFAULT_CONFIG)
            + sorted(conf.glob("profiles/*.yaml")))


def _inventory_keys(path: Path | None = None) -> list[str]:
    """conf の YAML を棚卸し正規表現で走査し、ヒットした行のドットパスを返す。"""
    lines = (path or DEFAULT_CONFIG).read_text(encoding="utf-8").split("\n")
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


def test_every_profile_inventory_hit_is_classified():
    """★§1.2 B2(第94バッチ OBS-U2): 走査を **全プロファイル**へ広げる。

    conf/production.yaml / daily.yaml / observe.yaml / smoke_*.yaml / profiles/*.yaml が
    上書きする step 単位キーも宣言漏れなら落とす(= 本番プロファイルで Δt=1 を回した
    ときだけ壊れる、という最も見つけにくい事故の構造的な防止)。"""
    files = _scanned_conf_files()
    assert len(files) >= 6, f"走査対象が少なすぎる(glob が壊れた): {files}"
    missing: dict[str, list[str]] = {}
    for path in files:
        bad = sorted({k for k in _inventory_keys(path) if not T.covers(k)})
        if bad:
            missing[path.name] = bad
    assert missing == {}, f"プロファイル側に未分類の step 単位キーがある: {missing}"


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


# --------------------------------------------------------------------------- #
# (7) 第94バッチ OBS-U2 — resume の Δt 二重適用(確定ブロッカー)の根治
#
#     症状: `scripts/run.py --resume` が Δt≠10 で**必ず** ValueError で落ちる。
#     機序: run dir の config.yaml は save_config が **apply_dt 済みの姿**で書いた
#           スナップショットなのに run.dt_min を保持しているため、読み直しで
#           apply_dt が二重適用される(apply_dt は冪等ではない)。
#     根治: load_config(apply_dt=False) を resume 経路だけに渡す(config のバイト列は
#           一切変えない = Δt=10 は完全に無風)。
#     正典: docs/research/obs-u2-dt1min-design.md §2(e) / §4 B1 / §5 R9。
# --------------------------------------------------------------------------- #
_DOUBLE_APPLY_WITNESS = ("world.modes.speeds.walk",      # RATE 800 → 80 →(誤)8
                         "drive.refractory_steps",       # STEPS  3 → 30 →(誤)300
                         "lod.max_llm_per_step")         # RATE 300 → 30 →(誤)3


def test_saved_config_is_already_converted_and_reload_must_not_convert_again(tmp_path):
    """★根治の中核: 保存済み config の読み直しは apply_dt=False でなければ壊れる。

    (a) apply_dt=False の再読込は保存時と**完全一致**(全キー)。
    (b) 既定(True)の再読込は 3 キーが実際に二重変換される = このフラグが必要な理由の実証。
    """
    cfg = load_config(["run.dt_min=1"])
    save_config(cfg, tmp_path)
    again = load_config(path=tmp_path / "config.yaml", apply_dt=False)
    assert (OmegaConf.to_container(again, resolve=True)
            == OmegaConf.to_container(cfg, resolve=True)), \
        "apply_dt=False の再読込が保存時と一致しない"
    twice = load_config(path=tmp_path / "config.yaml")          # 既定 = 二重適用
    for key in _DOUBLE_APPLY_WITNESS:
        once, dup = OmegaConf.select(cfg, key), OmegaConf.select(twice, key)
        # RATE は線形(×Δt/10)・STEPS は逆比例(×10/Δt)なので、二重適用の向きは分類で決まる
        factor = 10.0 if T.classify(key)[0] == T.STEPS else 1 / 10.0
        assert dup != once, f"{key} が二重変換されていない(テストの前提が崩れた)"
        assert float(dup) == pytest.approx(float(once) * factor, rel=1e-9), \
            f"{key} の二重変換の形が想定と違う: {once} → {dup}"


def test_reload_with_apply_dt_false_is_a_noop_at_canonical_dt(tmp_path):
    """Δt=10 では apply_dt の有無で 1 バイトも変わらない(= 既定ランは完全に無風)。"""
    cfg = load_config(["run.seed=7"])
    save_config(cfg, tmp_path)
    a = load_config(path=tmp_path / "config.yaml")               # 既定(従来の経路)
    b = load_config(path=tmp_path / "config.yaml", apply_dt=False)
    assert (OmegaConf.to_container(a, resolve=True)
            == OmegaConf.to_container(b, resolve=True))


def _load_run_cli():
    """scripts/run.py を module として読み込む(tests/test_chance.py と同じイディオム)。"""
    spec = importlib.util.spec_from_file_location("run_entry", _REPO / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli(run_mod, monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run_mod.main()


@pytest.mark.parametrize("dt", [1, 10])
def test_run_cli_resume_matches_straight(tmp_path, monkeypatch, dt):
    """★受入条件: `scripts/run.py --resume` の一致(Δt=1 = 新規救済 / Δt=10 = 回帰)。

    Δt=10 は従来から通っていた経路なので**バイト同一のまま**でなければならず、
    Δt=1 は本修正が入るまで config_hash 不一致の ValueError で必ず落ちていた。
    どちらも「一気 40 step」==「20 step で打ち切り → --resume で 40 step まで」。"""
    import pyarrow.parquet as pq

    run_mod = _load_run_cli()
    base = ["model.backend=mock", "run.seed=42", "run.n_agents=8",
            f"run.dt_min={dt}", f"run.out_dir={tmp_path.as_posix()}",
            "observer.checkpoint_every=20"]
    _cli(run_mod, monkeypatch, [*base, "run.name=straight", "run.n_steps=40"])
    _cli(run_mod, monkeypatch, [*base, "run.name=split", "run.n_steps=20"])
    _cli(run_mod, monkeypatch, ["--resume", str(tmp_path / "split"), "run.n_steps=40"])

    def rows(name, stem="l1_events"):
        return pq.read_table(str(tmp_path / name / f"{stem}.parquet")).to_pylist()

    a, b = rows("straight"), rows("split")
    assert len(a) == len(b) > 0, f"Δt={dt}: 行数不一致 {len(a)} vs {len(b)}"
    assert a == b, f"Δt={dt}: --resume の l1_events が一気通しと不一致"
    assert rows("straight", "l2_metrics") == rows("split", "l2_metrics")
    # 復元された config が二重変換されていないことを直接確認(症状の根に触る)
    saved = load_config(path=tmp_path / "split" / "config.yaml", apply_dt=False)
    fresh = load_config([f"run.dt_min={dt}", "run.seed=42", "run.n_agents=8",
                         "run.n_steps=40", "run.name=split", "model.backend=mock",
                         f"run.out_dir={tmp_path.as_posix()}",
                         "observer.checkpoint_every=20"])
    for key in _DOUBLE_APPLY_WITNESS:
        assert OmegaConf.select(saved, key) == OmegaConf.select(fresh, key), \
            f"Δt={dt}: 保存 config の {key} が新規ロードと食い違う"


def test_every_saved_config_reader_disables_apply_dt():
    """★R9(再発防止): 保存済み config を読む呼び出しは全て apply_dt=False を渡す。

    `load_config(path=…)` はリポジトリ全体で scripts/run.py の resume 経路だけだが、
    将来この経路が増えたときに黙って二重変換へ戻らないよう AST で固定する
    (docstring の注意書きだけでは守られない)。tests/ は既定挙動そのものを検査する
    ので対象外。"""
    targets = sorted(list((_REPO / "scripts").glob("*.py"))
                     + list((_REPO / "src").rglob("*.py")))
    checked = 0
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "load_config":
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if "path" not in kw:
                continue                      # 基底 config からの新規ロード = 変換して正しい
            checked += 1
            flag = kw.get("apply_dt")
            assert isinstance(flag, ast.Constant) and flag.value is False, (
                f"{path.relative_to(_REPO)}:{node.lineno} が保存済み config を "
                "apply_dt=False なしで読んでいる(Δt≠10 で二重変換になる)")
    assert checked >= 1, "保存済み config を読む経路が 1 つも見つからない(検査が空回り)"


def test_checkpoint_mismatch_error_points_at_the_dt_double_conversion():
    """副次(§4 B1): config_hash 不一致の文言が真因(Δt 二重変換)を指す。"""
    src = (_REPO / "src" / "society" / "engine" / "checkpoint.py").read_text(encoding="utf-8")
    assert "apply_dt=False" in src and "run.dt_min" in src, \
        "checkpoint の不整合メッセージが Δt 二重変換に言及していない"


# --------------------------------------------------------------------------- #
# (8) 第94バッチ OBS-U2 — A 級直書き(Clock を経由していない Δt 依存量)の是正
#
#     各件について 2 本ずつ固定する:
#       ① Δt=10 で**従来の直書きと数値恒等**(golden の世界を 1 ビットも動かさない)
#       ② Δt=1 で**実時間が保存**される(= 是正の目的)
#     正典: docs/research/obs-u2-dt1min-design.md §1.1(A1-A7)。
#     ★A8/A9 は本バッチの実装中に新規発見した同型の穴(設計調査の A 級表には無い)。
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, dt=10, n_agents=8, **ov):
    """走らせない(構築だけの)Simulation。Clock 依存の関数を直接叩くための足場。"""
    dot = ["run.seed=42", f"run.n_agents={n_agents}", "run.n_steps=1",
           f"run.name={name}", "model.backend=mock", f"run.dt_min={dt}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_a1_steps_until_tod_converts_minutes_with_the_clock():
    """A1 scheduler._steps_until_tod: 分 → step が Δt に追随する。"""
    from society.engine import scheduler as S

    for cur, target in ((420, 480), (1400, 60), (720, 720), (0, 1439), (615, 90)):
        delta = (int(target) - cur % 1440) % 1440 or 1440
        assert S._steps_until_tod(cur, target) == delta // 10, "Δt=10 の既定が動いた"
        assert S._steps_until_tod(cur, target, 10) == delta // 10
        fine = S._steps_until_tod(cur, target, 1)
        assert fine == delta, "Δt=1 で待ち時間が実時間にならない"
        assert fine * 1 == pytest.approx(delta), "実時間[分]が保存されていない"
        # 是正前の直書き(delta // 10)なら Δt=1 では 1/10 の実時間になっていた
        assert fine != delta // 10 or delta <= 1


def test_a2_account_day_boundary_uses_steps_per_day_not_clock_day(tmp_path):
    """A2 _phase_accounts_day: 会計日境界 = step // steps_per_day。

    ★clock.day() にすると start_tod=07:00 で境界が step 102 へ動き、**Δt=10 でも**
    会計日がずれる(golden 破壊)。その退行をここで直接踏み抜く。"""
    from society.engine import scheduler as S

    base = _sim(tmp_path, "a2_base", dt=10)
    assert base.clock.steps_per_day == 144 and base.clock.start_min == 420
    for step in (0, 102, 143):                       # 102 = clock.day() の境界(踏んではいけない)
        S._phase_accounts_day(base, step, base.clock.sim_min(step))
        assert base._acct_day == 0, f"step={step} で会計日が進んだ(clock.day() 化の退行)"
    S._phase_accounts_day(base, 144, base.clock.sim_min(144))
    assert base._acct_day == 1, "step 144 で会計日が進まない(Δt=10 の従来挙動が壊れた)"

    fine = _sim(tmp_path, "a2_fine", dt=1)
    assert fine.clock.steps_per_day == 1440
    for step in (0, 1020, 1439):
        S._phase_accounts_day(fine, step, fine.clock.sim_min(step))
        assert fine._acct_day == 0, f"Δt=1 の step={step} で会計日が進んだ(1日=1440step)"
    S._phase_accounts_day(fine, 1440, fine.clock.sim_min(1440))
    assert fine._acct_day == 1, "Δt=1 で 1 日 = 1440 step になっていない"


def test_a3_bus_headway_is_measured_in_wall_clock():
    """A3 BusNetwork.serves: 便判定の分 → step が Δt に追随する(headway ×10 の解消)。"""
    from society.world.transit import BusNetwork

    class _City:
        pass

    coarse = BusNetwork([], _City(), headway_steps=3)                    # Δt=10 既定
    assert coarse.step_minutes == 10
    assert [m for m in range(0, 120, 10) if coarse.serves(m)] == [0, 30, 60, 90]
    # Δt=1: headway_steps は STEPS 類なので config 側で ×10(3 step → 30 step)になる
    fine = BusNetwork([], _City(), headway_steps=30, step_minutes=1)
    assert [m for m in range(0, 120) if fine.serves(m)] == [0, 30, 60, 90], \
        "Δt=1 で便間隔(実時間 30 分)が保たれていない"
    # 是正前(sim_min // 10 の直書き)なら同じ便判定が 10 分連続していた
    naive = [m for m in range(0, 120) if ((m // 10) % 30) == 0]
    assert naive[:10] == list(range(10)), "テストの前提(旧実装の壊れ方)が崩れた"


def test_a4_indoor_cell_offset_requires_step_seconds():
    """A4 _indoor_cell_offset: step_seconds は必須引数(既定 600.0 の取りこぼしを構造的に封じる)。"""
    from society.engine import scheduler as S

    par = inspect.signature(S._indoor_cell_offset).parameters["step_seconds"]
    assert par.default is inspect.Parameter.empty, \
        "step_seconds に既定値が復活している(片方の呼び出しが 600.0 に落ちる)"
    for step in range(8):
        assert 0.0 <= S._indoor_cell_offset("b1", 2, step, 600.0) < 600.0
        off1 = S._indoor_cell_offset("b1", 2, step, 60.0)
        assert 0.0 <= off1 < 60.0, "Δt=1 で step 内オフセットが step 長を超えている"
    # 同じセル・同じ step なら比だけが Δt で変わる(決定論・seed 非依存の性質は不変)
    assert (S._indoor_cell_offset("b1", 2, 3, 60.0) * 10.0
            == pytest.approx(S._indoor_cell_offset("b1", 2, 3, 600.0)))
    src = inspect.getsource(S._phase_indoor)
    assert src.count("_indoor_cell_offset(") == 2 and "step_seconds" not in src.split(
        "_indoor_cell_offset(")[0][-200:], "呼び出し形が変わった(要追随)"
    assert src.count("sim.clock.step_seconds") >= 2, \
        "_phase_indoor の 2 箇所とも clock.step_seconds を渡していない"


@pytest.mark.parametrize("dt,per_day", [(10, 144), (1, 1440)])
def test_a5_venture_fulltime_days_use_steps_per_day(tmp_path, dt, per_day):
    """A5 tools._ventures_fulltime: 開業経過日数の 144 直書き → Clock。"""
    sim = _sim(tmp_path, f"a5_{dt}", dt=dt, **{"career.enabled": "true"})
    owner = next(a for a in sim.agents if not a.visitor)
    min_days = int(sim.careercfg["venture_fulltime_days"])
    v = {"name": "露店", "node": owner.node, "opened_step": 0,
         "sales_total": float(sim.careercfg["venture_fulltime_sales"]) + 1.0}
    sim.tools.ventures = {owner.id: v}
    sim.tools.ventures_by_node = {owner.node: [v]}
    just_before = min_days * per_day - 1
    sim.tools._ventures_fulltime(sim, just_before, sim.clock.sim_min(just_before))
    assert not v.get("fulltime"), (
        f"Δt={dt}: 経過 {min_days} 日の手前(step {just_before})で本業化した")
    sim.tools._ventures_fulltime(sim, min_days * per_day,
                                 sim.clock.sim_min(min_days * per_day))
    assert v.get("fulltime"), f"Δt={dt}: 経過 {min_days} 日ちょうどで本業化しない"


@pytest.mark.parametrize("dt,per_day", [(10, 144), (1, 1440)])
def test_a6_bankruptcy_restriction_lasts_the_same_wall_clock(tmp_path, dt, per_day):
    """A6 _eviction_bankruptcy_day: 破産制限期間 [日→step] の 144 直書き → Clock。"""
    from society.engine import scheduler as S

    sim = _sim(tmp_path, f"a6_{dt}", dt=dt)
    acc = dict(sim.economy["accounts"])
    acc.update({"eviction_days": 0, "bankruptcy_days": 1, "bankruptcy_keep": 0.0,
                "bankruptcy_restrict_days": 30})
    agent = next(a for a in sim.agents if not a.visitor)
    agent.rent_due, agent.arrears_days = 100.0, 0
    step = 5 * per_day
    S._eviction_bankruptcy_day(sim, agent, acc, step, sim.clock.sim_min(step))
    assert agent.bankrupt_until == step + 30 * per_day, \
        f"Δt={dt}: 制限期間が 30 日ぶんの step になっていない"
    restrict_min = (agent.bankrupt_until - step) * dt
    assert restrict_min == 30 * 1440, "実時間の制限期間(30日)が保存されていない"


def test_a7_actr_base_activation_is_a_function_of_wall_clock_time():
    """A7 MemoryStore._base_activation: 冪乗則忘却の時間単位を正準 step へ読み替える。"""
    from society.agents.memory import Episode, MemoryStore

    assert MemoryStore().dt_min == 10, "既定が正準でない(既定ランが恒等パスを外れる)"
    ep = Episode(step=0, text="出来事")
    coarse, fine = MemoryStore(dt_min=10), MemoryStore(dt_min=1)
    for elapsed_min in (10, 100, 550, 1440):
        b10 = coarse._base_activation(ep, elapsed_min // 10, 0.5)
        b1 = fine._base_activation(ep, elapsed_min, 0.5)
        assert b1 == pytest.approx(b10, abs=1e-12), \
            f"経過 {elapsed_min} 分の活性が Δt で変わる(実時間の関数になっていない)"
    # 既定パスは従来式そのもの(math.log(max(1, Δstep)**-d))= バイト一致の根拠
    assert coarse._base_activation(ep, 55, 0.5) == pytest.approx(math.log(55 ** -0.5))
    # 是正前は「step 差をそのまま時間」に使っていた = Δt=1 で 3.2 倍速に忘れていた
    assert fine._base_activation(ep, 55, 0.5) != pytest.approx(
        coarse._base_activation(ep, 55, 0.5))


@pytest.mark.parametrize("dt,per_day", [(10, 144), (1, 1440)])
def test_a8_venture_auto_close_lasts_the_same_wall_clock(tmp_path, dt, per_day):
    """A8(新規発見)tools._ventures_close: 売上ゼロ日数の 144 直書き → Clock。"""
    sim = _sim(tmp_path, f"a8_{dt}", dt=dt)
    owner = next(a for a in sim.agents if not a.visitor)
    days = int(sim.tools.cfg["venture_close_days"])
    v = {"name": "露店", "node": owner.node, "opened_step": 0, "last_sale_step": 0}
    sim.tools.ventures = {owner.id: v}
    sim.tools.ventures_by_node = {owner.node: [v]}
    just_before = days * per_day - 1
    sim.tools._ventures_close(sim, just_before, sim.clock.sim_min(just_before))
    assert owner.id in sim.tools.ventures, f"Δt={dt}: {days} 日の手前で閉店した"
    sim.tools._ventures_close(sim, days * per_day, sim.clock.sim_min(days * per_day))
    assert owner.id not in sim.tools.ventures, f"Δt={dt}: {days} 日で閉店しない"


@pytest.mark.parametrize("dt", [10, 1])
def test_a9_inflow_commuter_arrival_keeps_wall_clock(tmp_path, dt):
    """A9(新規発見)_init_inflow_commuters: 初日の到着 step の `// 10` 直書き → Clock。"""
    sim = _sim(tmp_path, f"a9_{dt}", dt=dt)
    agent = sim.agents[0]
    agent.commute, agent.commute_gateway, agent.arrival_min = True, "edge", 9 * 60
    sim._init_inflow_commuters()
    expected_min = 9 * 60 - sim.clock.start_min          # 07:00 開始 → 120 分後
    assert agent.return_at == expected_min // dt
    assert agent.return_at * dt == expected_min, "初日の到着が実時刻からずれた"
    assert sim.clock.sim_min(agent.return_at) == 9 * 60


# --------------------------------------------------------------------------- #
# (9) 第94バッチ OBS-U2 — Δt=1 並行小ラン用プロファイル(§4 B5)
# --------------------------------------------------------------------------- #
_DT1_PROFILE = _REPO / "conf" / "smoke_dt1.yaml"


def _leaf_keys(node, prefix=""):
    """YAML ツリーの葉のドットパスを列挙する。"""
    try:
        items = list(node.items())
    except AttributeError:
        yield prefix
        return
    if not items:
        yield prefix
        return
    for k, v in items:
        yield from _leaf_keys(v, f"{prefix}.{k}" if prefix else str(k))


def test_dt1_profile_writes_only_invariant_keys():
    """★Δt プロファイルの鉄則: **変換される分類のキーを書かない**。

    プロファイルは apply_dt の**前**に重なるので、RATE/PROB/KEEP/STEPS のキーを
    「Δt=1 での値」で書くと更に変換されて二重になる(例: max_llm_per_step=30 → 3)。
    書いてよいのは INVARIANT 分類(= Δt が触らない量)と未分類の非時間キーだけ。"""
    prof = OmegaConf.load(_DT1_PROFILE)
    bad = []
    for key in _leaf_keys(prof):
        if key == "run.dt_min":
            continue                       # Δt の宣言点そのもの
        hit = T.classify(key)
        if hit is not None and hit[0] != T.INVARIANT:
            bad.append(f"{key}({hit[0]})")
    assert bad == [], (
        "Δt=1 プロファイルが『変換される分類』のキーを書いている(二重変換になる): "
        + ", ".join(bad))


def test_dt1_profile_yields_the_intended_world():
    """プロファイルを通したときの最終 config が設計どおりか(自動換算と手動 ×10 の分担)。"""
    cfg = load_config(["run.n_steps=24", "run.n_agents=8"], profile=str(_DT1_PROFILE))
    assert int(cfg.run.dt_min) == 1
    base = load_config()
    # (a) 自動: RATE の予算は 300 → 30(1 日 cap の総量 43,200 は保存)
    assert int(cfg.lod.max_llm_per_step) * 10 == int(base.lod.max_llm_per_step)
    assert int(cfg.lod.max_llm_per_step) * 1440 == int(base.lod.max_llm_per_step) * 144
    # (b) 自動: STEPS の L3 間隔は逆比例(実時間 2 時間の粒度を保つ)
    assert int(cfg.observer.snapshot_every) == int(base.observer.snapshot_every) * 10
    # (c) 手動: INVARIANT な観測間引きは実験者が ×10 する(§2(f) の設計判断)
    assert int(cfg.cognition.channels.every_steps) == 10
    assert int(cfg.cognition.g_update.log_every_steps) == 10
    assert int(cfg.observer.state_hash.interval) == 10
    assert int(cfg.env.feedback.log_every_steps) == 40
    assert int(cfg.observer.flush_every_steps) == 1440       # 1 シミュ日ごと
    assert int(cfg.observer.checkpoint_every) == 1440        # 観察ランの運用規約=毎日
    # (d) 驚き駆動を観るためのプロファイルなので発火層が ON
    assert bool(cfg.cognition.fire.enabled) is True


def test_dt1_profile_runs_and_fires_on_salience(tmp_path):
    """★受入スモーク: Δt=1 の小ランがクラッシュせず、**salience 発火が出る**。

    Δt=10 では CogQueue の構造上 salience が 0 件になる(設計調査 §2(b) 実測)。
    Δt=1 で 1 件でも出ることが OBS-U2 の目的そのものなので、ここで固定する。"""
    cfg = load_config(["run.n_steps=24", "run.n_agents=12", "run.seed=42",
                       "run.name=dt1_smoke", "model.backend=mock"],
                      profile=str(_DT1_PROFILE))
    sim = Simulation(cfg, out_dir=tmp_path / "dt1_smoke")
    summary = sim.run()
    assert summary["n_events"] > 0
    assert sim.clock.step_minutes == 1
    reasons = [e.payload.get("reason") for e in sim.logger.events
               if e.kind == "cog_fire"]
    assert reasons, "cog_fire が 1 件も出ていない(認知層が動いていない)"
    assert reasons.count("salience") > 0, (
        "Δt=1 でも salience 発火が 0 件(驚き駆動が観測できない = OBS-U2 の動機が消える)")


def test_b6_enforcement_memory_text_states_real_minutes(tmp_path):
    """§1.2 B6: 留置の記憶テキスト(プロンプト入力)の分数が Δt に追随する。

    Δt=10 では文字列が 1 バイトも変わらないこと(= 認知の入力が不変)を固定する。"""
    from society.engine import scheduler as S

    src = inspect.getsource(S._phase_enforcement)
    assert "det * sim.clock.step_minutes" in src and "det * 10" not in src, \
        "留置時間の分数が step_minutes を経由していない(Δt=1 で 10 倍の嘘になる)"
    coarse = _sim(tmp_path, "b6_coarse", dt=10)
    fine = _sim(tmp_path, "b6_fine", dt=1)
    det_coarse, det_fine = 3, 30                    # STEPS 類 = config 側で ×10 される
    assert (f"取り締まりで{det_coarse * coarse.clock.step_minutes}分間その場に留め置かれた"
            == "取り締まりで30分間その場に留め置かれた"), "Δt=10 の文面が変わった"
    assert (det_fine * fine.clock.step_minutes
            == det_coarse * coarse.clock.step_minutes), "Δt=1 で拘束時間の実分数がずれる"
