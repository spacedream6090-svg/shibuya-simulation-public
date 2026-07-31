"""天候 W2(較正済み生成器の weather.py 統合)のテスト — 第80バッチ 2026-08-01。

正典: docs/research/weather-generator-design.md §4(統合設計)。
W1(第79バッチ・scripts と data/snapshot だけ)で較正した確率生成器を、
``weather.mode: synthetic(既定) / generated / table`` の3モードとしてシムへ入れる。

鉄則の検証:
- **既定 synthetic は 1 バイトも変わらない**: 明示 synthetic == キー未指定 == 純粋既定で
  L1 一致・weather payload は従来どおり {cond, temp_hi} の2キー・プロンプト1行の文言も不変
  (extra_prompt_fields=true にしても synthetic では無視する)。
  OFF(enabled=false)では mode=generated でも静的ファイルを 1 バイトも読まない。
- **実装の二重化が無い**: `scripts/fit_weather_gen.simulate`(較正・自己検証側)と
  `society.weather_gen.generate_segment`(シム側)が **同じ入力→同じ系列**を出す。
- **generated**: 新 stream "weather_gen" のみ使用(既存 "weather" は引かない)・同 seed 2ラン一致・
  prefix 安定(= resume で系列を作り直しても同一)・resume==straight(日境界跨ぎ)・
  較正済みでない月/暦 OFF/月跨ぎは **エラーにせず synthetic へフォールバック**して件数と理由を残す。
  シード掃引の統計が W1 の較正値と整合(P(≥35℃)・連続猛暑の到達可能性・36℃以上が出ること)。
- **table**: 乱数を1つも引かない・凍結実測と値が一致・暦の年が凍結範囲外なら同月日の最新年へ
  回して**その事実を記録**・8月外は synthetic フォールバック+警告。
- **来歴**: generated/table のとき summary.json と run_manifest.json に weather_params_sha256 /
  weather_source_sha256 とファイルの sha256 が載る。改竄した params は読み込み時に弾く。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import weather as W
from society import weather_gen as WG
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.rng import RngHub

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fit_weather_gen as FG                      # noqa: E402

PARAMS = REPO_ROOT / "data" / "snapshot" / "weather_gen_params.json"
HISTORY = REPO_ROOT / "data" / "snapshot" / "weather_tokyo_aug.json"
needs_data = pytest.mark.skipif(
    not (PARAMS.exists() and HISTORY.exists()),
    reason="data/snapshot/weather_*.json が未生成(第79バッチ W1 の成果物が要る)")

# 本選ランの想定(8/16 起点・10日)。1 day = 144 step。
_AUG = {"world.calendar.enabled": "true", "world.calendar.start_date": "2026-08-16",
        "weather.enabled": "true"}


# --------------------------------------------------------------------------- #
# ヘルパ
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, steps=144, n=10, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _run(tmp_path, name, steps=144, n=10, **ov):
    sim = _sim(tmp_path, name, steps, n, **ov)
    sim.run()
    return sim, tmp_path / name


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _wev(sim):
    return [e for e in sim.logger.events if e.kind == "weather"]


def _summary(out_dir):
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def _manifest(out_dir):
    return json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))


class _CountingHub:
    """stream キーごとの派生回数を数えるプロキシ(乱数経路の実査用)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _params():
    return WG.load_params(str(PARAMS))


# ============================================================ A. 既定 synthetic の不変性
def test_shipped_default_mode_is_synthetic():
    cfg = load_config()
    assert str(cfg.weather.mode) == "synthetic"
    assert cfg.weather.extra_prompt_fields is False
    # 生成器/実測表のパスは envpack(場所の値)側にある = src には東京固有のパスが無い
    assert str(cfg.envpack.climate.gen_params).endswith("weather_gen_params.json")
    assert str(cfg.envpack.climate.table).endswith("weather_tokyo_aug.json")


def test_explicit_synthetic_matches_unset_l1(tmp_path):
    """mode を明示しても既定ランと L1 が一字一句一致する(新キーが no-op)。"""
    a, _ = _run(tmp_path, "w_unset", steps=144, **_AUG)
    b, _ = _run(tmp_path, "w_syn", steps=144, **{**_AUG, "weather.mode": "synthetic"})
    assert _l1(a) == _l1(b)


def test_weather_off_never_reads_files(tmp_path):
    """enabled=false なら mode=generated でも静的ファイルを 1 バイトも読まない=純粋既定と一致。"""
    sim = _sim(tmp_path, "w_off", steps=12,
               **{"weather.enabled": "false", "weather.mode": "generated"})
    assert sim.weathercfg["params"] is None and sim.weathercfg["provenance"] is None
    assert W.provenance(sim) is None
    sim.run()
    pure, _ = _run(tmp_path, "w_pure", steps=12)
    assert _l1(sim) == _l1(pure)
    assert "weather" not in _summary(tmp_path / "w_off")
    assert "weather" not in _manifest(tmp_path / "w_off")


def test_synthetic_payload_keeps_two_keys(tmp_path):
    """synthetic の weather payload は従来どおり(観測列を足さない=L1 バイト一致の実体)。"""
    sim, _ = _run(tmp_path, "w_pay", steps=144, **_AUG)
    ev = _wev(sim)
    assert ev
    for e in ev:
        assert set(e.payload) == {"cond", "temp_hi", "date", "weekday", "holiday"}


def test_synthetic_prompt_line_is_never_extended():
    """extra_prompt_fields=true でも synthetic の文言は 1 バイトも変わらない。"""
    w = {"cond": "雨", "temp_hi": 14, "temp_lo": 9, "humid": 88, "wbgt": 22.2}
    base = "今日の天気: 雨、最高14℃。"
    assert W.weather_line(w) == base
    assert W.weather_line(w, W.build_cfg({"enabled": True})) == base
    assert W.weather_line(
        w, W.build_cfg({"enabled": True, "extra_prompt_fields": True})) == base
    assert W.weather_line(None) is None


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="weather.mode"):
        W.build_cfg({"enabled": True, "mode": "turbo"})


# ============================================================ B. 実装の二重化が無いこと
@needs_data
def test_src_generator_and_fit_script_produce_the_same_series():
    """★検収の核心: 較正スクリプト側と シム側が **同じ入力→同じ系列**(実装が1本)。"""
    pm = _params()["months"]["8"]
    prep = WG.prepare(pm)
    for seed in (1, 20260801):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed])))
        mine = WG.generate_segment(rng, prep, 31, day_start=1)
        theirs = FG.simulate(pm, 1, 31, seed)
        for key in ("state", "temp_hi", "temp_lo", "humid_mean", "precip_mm"):
            assert np.array_equal(mine[key], theirs[key]), (seed, key)
    # 多年ループ(rng を年間で連続消費する)でも一致する
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7])))
    segs = [WG.generate_segment(rng, prep, 31, day_start=1) for _ in range(5)]
    ref = FG.simulate(pm, 5, 31, 7)
    assert np.array_equal(np.concatenate([s["temp_hi"] for s in segs]), ref["temp_hi"])


@needs_data
def test_generated_series_is_prefix_stable():
    """長さを伸ばしても先頭 n 日は不変(= resume で作り直しても同一系列になる根拠)。"""
    pm = _params()["months"]["8"]
    prep = WG.prepare(pm)
    hub = RngHub(42)
    short = WG.generate_segment(hub.stream("weather_gen", 0), prep, 5, day_start=16)
    long = WG.generate_segment(hub.stream("weather_gen", 0), prep, 40, day_start=16)
    assert np.array_equal(short["temp_hi"], long["temp_hi"][:5])
    assert np.array_equal(short["state"], long["state"][:5])


# ============================================================ C. generated モード
@needs_data
def test_generated_run_is_well_formed(tmp_path):
    """10日ラン: 日数ぶんのイベント・語彙4種・temp_lo<=temp_hi・観測列が載る。"""
    sim, out = _run(tmp_path, "g_ok", steps=1440, **{**_AUG, "weather.mode": "generated"})
    ev = _wev(sim)
    assert len(ev) == 11                       # 8/16〜8/26(1440 step = 10 日 + 起点)
    for e in ev:
        p = e.payload
        assert e.agent_id == -1
        assert p["cond"] in ("晴", "曇", "雨", "雪")
        assert isinstance(p["temp_hi"], int) and isinstance(p["temp_lo"], int)
        assert p["temp_lo"] <= p["temp_hi"]
        assert p["source"] == "generated"
        assert p["humid"] is not None and p["wbgt"] is not None
        assert p["precip_mm"] >= 0.0
        assert (p["precip_mm"] > 0.0) == (p["cond"] in ("雨", "雪"))
    assert sim.today_weather_line.startswith("今日の天気: ")
    prov = _summary(out)["weather"]
    assert prov["days"] == {"generated": 11, "table": 0, "table_year_remapped": 0,
                            "synthetic_fallback": 0, "fallback_reasons": {}}


@needs_data
def test_generated_two_runs_same_seed_are_identical(tmp_path):
    a, _ = _run(tmp_path, "g_d1", steps=1440, **{**_AUG, "weather.mode": "generated"})
    b, _ = _run(tmp_path, "g_d2", steps=1440, **{**_AUG, "weather.mode": "generated"})
    assert _l1(a) == _l1(b)


@needs_data
def test_generated_uses_only_the_new_stream(tmp_path):
    """★設計の要: 新キー "weather_gen" のみ。既存 "weather" stream は1度も引かない。"""
    sim = _sim(tmp_path, "g_stream", steps=1440, **{**_AUG, "weather.mode": "generated"})
    hub = _CountingHub(sim.hub)
    sim.hub = hub
    sim.run()
    assert hub.counts.get("weather_gen") == 1, "系列は1ランで1回だけ生成する(メモ化)"
    assert "weather" not in hub.counts, "generated なのに合成用 stream を引いている"
    # 対照: synthetic は "weather" を日数ぶん引き "weather_gen" を引かない
    sim2 = _sim(tmp_path, "g_stream_syn", steps=1440, **_AUG)
    hub2 = _CountingHub(sim2.hub)
    sim2.hub = hub2
    sim2.run()
    assert hub2.counts.get("weather") == 11 and "weather_gen" not in hub2.counts


@needs_data
def test_generated_matches_the_offline_generator(tmp_path):
    """ランの系列が「同 seed の RngHub("weather_gen",0) + 8/16 起点」の生成と一致する。"""
    sim, _ = _run(tmp_path, "g_ref", steps=1440, **{**_AUG, "weather.mode": "generated"})
    prep = WG.prepare(_params()["months"]["8"])
    seg = WG.generate_segment(RngHub(42).stream("weather_gen", 0), prep,
                              len(sim._weather_series), day_start=16)
    ref = WG.day_records(seg, prep)
    got = [{k: v for k, v in r.items() if k != "date"} for r in sim._weather_series]
    assert got == ref


@needs_data
def test_generated_resume_matches_straight_across_day_boundary(tmp_path):
    """resume==straight(split 200 > 日境界 144)。系列は再構築されるが prefix 安定で同一。"""
    ov = {**_AUG, "weather.mode": "generated", "weather.rain_grievance": "0.01"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=12", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "g_straight"
    straight = Simulation(_cfg("g_straight", 300), out_dir=straight_dir)
    straight.run()
    assert len(_wev(straight)) >= 3, "テスト前提が崩れた(日境界を跨いでいない)"

    d = tmp_path / "g_resumed"
    split, total = 200, 300
    sim1 = Simulation(_cfg("g_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("g_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(generated resume)"


def test_synthetic_resume_no_longer_duplicates_the_weather_event(tmp_path):
    """★W2 の検収で顕在化した**既存バグ**の回帰固定(生成モードとは独立)。

    `_cal_day`(日付・天気の日境界進行)が checkpoint に載っていなかったため、
    weather.enabled=true の mid-day resume は再開直後の 1 step で同じ日を再処理し、
    weather イベントを二重記録していた(rain_grievance>0 なら不快感も二重加算)。
    """
    ov = {"world.calendar.enabled": "true", "world.calendar.start_date": "2026-08-16",
          "weather.enabled": "true", "weather.rain_grievance": "0.01"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=12", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "s_straight"
    straight = Simulation(_cfg("s_straight", 300), out_dir=straight_dir)
    straight.run()
    d = tmp_path / "s_resumed"
    split, total = 200, 300                    # split=200 は日境界(102 / 246)の途中
    sim1 = Simulation(_cfg("s_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    assert sim1._cal_day == 1
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    # 直接検証: 日境界進行と当日確定値が round-trip する(空回り防止)
    sim3 = Simulation(_cfg("s_inspect", split), out_dir=tmp_path / "s_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._cal_day == 1
    assert sim3.today_weather == sim1.today_weather
    assert sim3.today_weather_line == sim1.today_weather_line
    assert sim3.today_date_line == sim1.today_date_line

    sim2 = Simulation(_cfg("s_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    sa = pq.read_table(straight_dir / "l1_events.parquet").to_pylist()
    sb = pq.read_table(d / "l1_events.parquet").to_pylist()
    assert [e for e in sa if e["kind"] == "weather"] == \
           [e for e in sb if e["kind"] == "weather"], "天気イベントが二重記録されている"
    assert sa == sb


@needs_data
def test_generated_falls_back_when_month_not_calibrated(tmp_path, caplog):
    """較正済みでない月(3月)は **エラーにせず** synthetic へ落ち、件数と理由が残る。"""
    with caplog.at_level(logging.WARNING, logger="society.weather"):
        sim, out = _run(tmp_path, "g_mar", steps=432,
                        **{**_AUG, "weather.mode": "generated",
                           "world.calendar.start_date": "2026-03-01"})
    ev = _wev(sim)
    assert ev and all(set(e.payload) >= {"cond", "temp_hi"} for e in ev)
    assert all(e.payload["source"] == "synthetic" for e in ev)
    days = _summary(out)["weather"]["days"]
    assert days["generated"] == 0 and days["synthetic_fallback"] == len(ev)
    assert days["fallback_reasons"] == {"month_3_not_calibrated": len(ev)}
    assert any("フォールバック" in r.message for r in caplog.records)


@needs_data
def test_generated_requires_calendar(tmp_path):
    """暦 OFF では『何月何日か』が決まらない=合成へ落ちる(黙って別の月を使わない)。"""
    sim, out = _run(tmp_path, "g_nocal", steps=288,
                    **{"weather.enabled": "true", "weather.mode": "generated"})
    days = _summary(out)["weather"]["days"]
    assert days["generated"] == 0
    assert days["fallback_reasons"] == {"calendar_off": days["synthetic_fallback"]}


@needs_data
def test_generated_does_not_extrapolate_across_month_boundary(tmp_path):
    """8月の較正を9月へ当てない(月をまたいだ日は合成へ落ちる)。"""
    sim, out = _run(tmp_path, "g_cross", steps=864,     # 6日: 8/28〜9/2
                    **{**_AUG, "weather.mode": "generated",
                       "world.calendar.start_date": "2026-08-28"})
    ev = _wev(sim)
    by_date = {e.payload["date"]: e.payload["source"] for e in ev}
    assert by_date["2026-08-28"] == "generated"
    assert by_date["2026-08-31"] == "generated"
    assert by_date["2026-09-01"] == "synthetic"
    d = _summary(out)["weather"]["days"]
    assert d["generated"] == 4 and d["fallback_reasons"] == {"outside_anchor_month": 3}


@needs_data
def test_generated_extra_prompt_fields(tmp_path):
    """extra_prompt_fields=true: 基底1行は不変のまま末尾に中立な事実だけを足す。"""
    sim, _ = _run(tmp_path, "g_extra", steps=144,
                  **{**_AUG, "weather.mode": "generated",
                     "weather.extra_prompt_fields": "true"})
    line = sim.today_weather_line
    w = sim.today_weather
    assert line.startswith(f"今日の天気: {w['cond']}、最高{w['temp_hi']}℃。")
    assert f"最低{w['temp_lo']}℃" in line and f"湿度{w['humid']}%" in line
    assert f"暑さ指数(推定){w['wbgt']}。" in line
    assert ("猛暑日(最高気温35℃以上)。" in line) == (w["temp_hi_c"] >= 35.0)


# ------------------------------------------------------------ C'. 統計が W1 の較正と整合
@needs_data
class TestGeneratedStatisticsMatchCalibration:
    """シード掃引: シム側の生成器が W1 の較正値(設計書 §3.2)を再現していること。

    ★数値は tests/test_weather_gen_offline.py の C 群(較正スクリプト側)と同じ性質を、
      **シムが実際に使う経路**(RngHub の "weather_gen" stream)で確認するもの。
    """

    N_SEEDS = 400

    def _sweep(self, n_days: int, day_start: int):
        prep = WG.prepare(_params()["months"]["8"])
        hi: list[float] = []
        max_spell: list[int] = []
        for seed in range(self.N_SEEDS):
            seg = WG.generate_segment(RngHub(seed).stream("weather_gen", 0), prep,
                                      n_days, day_start=day_start)
            recs = WG.day_records(seg, prep)
            hi += [r["temp_hi_c"] for r in recs]
            max_spell.append(WG.series_stats(recs)["max_hot_spell"])
        return np.array(hi), np.array(max_spell)

    def test_hot_day_rate_matches_w1(self):
        """8月まるごと(31日)の P(Tmax≥35℃) が W1 の 22% 近傍にある。"""
        hi, _ = self._sweep(31, 1)
        p35 = float((hi >= 35.0).mean())
        assert 0.17 < p35 < 0.28, p35
        # 現行合成では構造的に不可能な 36℃以上が出る(実測にも 7.3% ある)
        assert float((hi >= 36.0).mean()) > 0.03
        assert 31.5 < hi.mean() < 33.5, hi.mean()

    def test_ten_day_window_is_cooler_by_the_month_trend(self):
        """本選窓(8/16 起点 10日)は月内トレンド(−0.083℃/日)ぶん下旬寄りで少し涼しい。"""
        hi31, _ = self._sweep(31, 1)
        hi10, _ = self._sweep(11, 16)
        assert hi10.mean() < hi31.mean()
        assert 0.10 < float((hi10 >= 35.0).mean()) < 0.25

    def test_long_heat_waves_are_reachable_in_a_ten_day_run(self):
        """★本件の目的: 10日ランの窓でも連続猛暑が出現しうる(現行合成では届かない)。"""
        _hi, spells = self._sweep(11, 16)
        assert spells.max() >= 7, spells.max()
        assert float((spells >= 5).mean()) > 0.01
        # 対照: 現行合成(32±3 の独立同分布)は 5 連ですら滅多に出ない
        cur = FG.simulate_synthetic_current(self.N_SEEDS, 11, 99)["temp_hi"]
        cur_max = np.array([max(FG.run_lengths(s >= 35.0) or [0])
                            for s in cur.reshape(-1, 11)])
        assert cur_max.max() < 7
        assert float((cur_max >= 5).mean()) < float((spells >= 5).mean())

    def test_interannual_spread_survives(self):
        """年効果(低周波): ランごとの平均のばらつきが現行合成より大きい。"""
        prep = WG.prepare(_params()["months"]["8"])
        means = []
        for seed in range(self.N_SEEDS):
            seg = WG.generate_segment(RngHub(seed).stream("weather_gen", 0), prep,
                                      31, day_start=1)
            means.append(float(seg["temp_hi"].mean()))
        assert float(np.std(means, ddof=1)) > 1.0


# ============================================================ D. table モード
@needs_data
def test_table_matches_frozen_observations(tmp_path):
    """実日付が凍結範囲にある年(2025)では実測値そのものが出る。"""
    sim, out = _run(tmp_path, "t_2025", steps=1440,
                    **{**_AUG, "weather.mode": "table",
                       "world.calendar.start_date": "2025-08-17"})
    frozen = {d["date"]: d for d in
              json.loads(HISTORY.read_text(encoding="utf-8"))["days"]}
    ev = _wev(sim)
    assert len(ev) == 11
    for e in ev:
        obs = frozen[e.payload["date"]]
        assert e.payload["source"] == "table"
        assert e.payload["obs_date"] == e.payload["date"]
        assert e.payload["temp_hi_c"] == round(float(obs["temp_hi"]), 1)
        assert e.payload["temp_lo_c"] == round(float(obs["temp_lo"]), 1)
        assert e.payload["precip_mm"] == round(float(obs["precip_mm"] or 0.0), 1)
    days = _summary(out)["weather"]["days"]
    assert days["table"] == 11 and days["table_year_remapped"] == 0


@needs_data
def test_table_draws_no_random_numbers(tmp_path):
    """完全決定論: 天候用の stream を 1 本も引かない。"""
    sim = _sim(tmp_path, "t_norng", steps=1440,
               **{**_AUG, "weather.mode": "table",
                  "world.calendar.start_date": "2025-08-17"})
    hub = _CountingHub(sim.hub)
    sim.hub = hub
    sim.run()
    assert "weather" not in hub.counts and "weather_gen" not in hub.counts


@needs_data
def test_table_remaps_year_outside_frozen_range_and_records_it(tmp_path, caplog):
    """暦の年(2026)が凍結範囲外なら同月日の**最新年**(2025)へ回し、その事実を残す。"""
    with caplog.at_level(logging.WARNING, logger="society.weather"):
        sim, out = _run(tmp_path, "t_2026", steps=1440,
                        **{**_AUG, "weather.mode": "table"})
    ev = _wev(sim)
    assert len(ev) == 11
    for e in ev:
        assert e.payload["source"] == "table_year_remapped"
        assert e.payload["obs_date"].startswith("2025-")
        assert e.payload["obs_date"][5:] == e.payload["date"][5:]   # 月日は同じ
    days = _summary(out)["weather"]["days"]
    assert days["table"] == 11 and days["table_year_remapped"] == 11
    assert any("凍結データの範囲外" in r.message for r in caplog.records)


@needs_data
def test_table_outside_august_falls_back_with_warning(tmp_path, caplog):
    """凍結していない月(8月以外)は **エラーでなく** synthetic フォールバック + 警告。"""
    with caplog.at_level(logging.WARNING, logger="society.weather"):
        sim, out = _run(tmp_path, "t_mar", steps=432,
                        **{**_AUG, "weather.mode": "table",
                           "world.calendar.start_date": "2026-03-01"})
    ev = _wev(sim)
    assert ev and all(e.payload["source"] == "synthetic" for e in ev)
    days = _summary(out)["weather"]["days"]
    assert days["fallback_reasons"] == {"date_not_in_frozen_table": len(ev)}
    assert any("フォールバック" in r.message for r in caplog.records)


@needs_data
def test_table_run_reproduces_the_real_2025_heat_wave(tmp_path):
    """2025年8月に実際に起きた連続猛暑が table モードのランに現れる(設計書 §2.1)。"""
    _sim_, out = _run(tmp_path, "t_wave", steps=1440,
                      **{**_AUG, "weather.mode": "table",
                         "world.calendar.start_date": "2025-08-17"})
    series = _summary(out)["weather"]["series"]
    assert series["max_hot_spell"] >= 8, series
    assert series["p_hot_day"] > 0.6, series


# ============================================================ E. 来歴(ハッシュ)
@needs_data
def test_generated_records_provenance_hashes(tmp_path):
    """summary.json と run_manifest.json に weather_params_sha256 と実データの sha256 が載る。"""
    _s, out = _run(tmp_path, "p_gen", steps=288, **{**_AUG, "weather.mode": "generated"})
    doc = json.loads(PARAMS.read_text(encoding="utf-8"))
    hist = json.loads(HISTORY.read_text(encoding="utf-8"))
    for block in (_summary(out)["weather"], _manifest(out)["weather"]):
        assert block["mode"] == "generated"
        assert block["weather_params_sha256"] == doc["meta"]["payload_sha256"]
        assert block["weather_source_sha256"] == hist["meta"]["payload_sha256"]
        f = block["files"]["gen_params"]
        assert f["file_sha256"] == WG.file_sha256(PARAMS)
        assert "気象庁" in f["attribution"]          # PDL1.0 の出典明示は必須条件
        assert block["calibrated_months"] == [8]
        assert block["fit_window"]["years"] == [2015, 2025]


@needs_data
def test_table_records_provenance_hashes(tmp_path):
    _s, out = _run(tmp_path, "p_tab", steps=288,
                   **{**_AUG, "weather.mode": "table",
                      "world.calendar.start_date": "2025-08-17"})
    hist = json.loads(HISTORY.read_text(encoding="utf-8"))
    for block in (_summary(out)["weather"], _manifest(out)["weather"]):
        assert block["mode"] == "table"
        assert block["weather_source_sha256"] == hist["meta"]["payload_sha256"]
        assert block["files"]["table"]["file_sha256"] == WG.file_sha256(HISTORY)
        assert block["table_dates"] == ["1996-08-01", "2025-08-31"]


def test_synthetic_run_writes_no_weather_block(tmp_path):
    """既定モードのランは summary/manifest の形も従来どおり(weather キー自体が無い)。"""
    _s, out = _run(tmp_path, "p_syn", steps=144, **_AUG)
    assert "weather" not in _summary(out)
    assert "weather" not in _manifest(out)


@needs_data
def test_tampered_params_are_rejected(tmp_path):
    """payload_sha256 を直さずに値をいじった較正ファイルは読み込み時に弾く。"""
    doc = json.loads(PARAMS.read_text(encoding="utf-8"))
    doc["months"]["8"]["temp"]["grand_mean_hi"] += 1.0
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256 不一致"):
        WG.load_params(str(bad))
    with pytest.raises(ValueError, match="payload_sha256 不一致"):
        W.build_cfg({"enabled": True, "mode": "generated"}, gen_params=str(bad))


def test_missing_params_path_is_a_clear_error():
    with pytest.raises(ValueError, match="gen_params"):
        W.build_cfg({"enabled": True, "mode": "generated"})
    with pytest.raises(ValueError, match="table"):
        W.build_cfg({"enabled": True, "mode": "table"})


# ============================================================ F. レジストリ / モード
def test_registry_declares_the_new_toggles():
    assert R.undeclared_toggles(load_config()) == []
    f = R.BY_ID["weather.mode"]
    assert f.repro_tier == "strict" and f.off_value == "synthetic"
    assert R.BY_ID["weather.extra_prompt_fields"].repro_tier == "strict"
    # 未宣言検出がこの namespace で機能していることの自己点検
    cfg = R._container(load_config())
    cfg["weather"]["some_new_flag"] = True
    assert "weather.some_new_flag" in R.undeclared_toggles(cfg)


def test_generated_survives_verify_mode():
    """strict なので対照実験モード(verify)でも自動 OFF されない。"""
    gated, rep = R.apply_mode(load_config(["run.mode=verify", "weather.enabled=true",
                                           "weather.mode=generated"]))
    assert str(gated.weather.mode) == "generated"
    assert "weather.mode" not in {d["id"] for d in rep["auto_disabled"]}
    assert {"id": "weather.mode", "repro_tier": "strict", "affects_k": False,
            "fingerprint_risk": "possible"} in rep["enabled"]


def test_timeconv_classifies_weather_keys_as_invariant():
    """天候は日次確定(1 日 1 回)なので Δt を変えても意味が変わらない。"""
    from society import timeconv as T
    for key in ("weather.mode", "weather.rain_grievance", "weather.extra_prompt_fields"):
        cls, why = T.classify(key)
        assert cls == T.INVARIANT and why.strip(), key
