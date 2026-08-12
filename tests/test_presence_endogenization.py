"""在場の内生化(第109バッチ レーン PRES-Pool: A1 習慣 / A2 cap 撤去 / B 職業多様性)。

正典: docs/plans/presence-endogenization-plan.md §4(ユーザー承認 2026-08-12)/
      docs/research/presence-endogenization.md §2・§4 / src/society/world/presence.py docstring。

ユーザー原理「範囲内にいる人の数もこちらが指定するものではない…世界のアルゴリズムによって
結果が左右されたり、エージェントの量が決まったりすることはなく」に対して、現行機構に残って
いた違反は 2 点だけだった(リサーチ §2.4):

  (a) present_cap の当日ランキング切り … 落ちる理由が本人の行動に帰属しない  → **A2**
  (b) stochastic 層の Bernoulli 抽選   … 今日来るかを世界のコインが決める      → **A1**

本ファイルが機械固定するもの:
  ① **分布保存の較正証明**(A1①): 習慣カレンダーの 1 日あたり在場確率が
     Bernoulli(visit_rate) と厳密に一致する(丸め誤差ゼロ)。
  ② **週次総量の保存**(A1②): 曜日プロファイルは週平均 1.0 = 週内の配分だけを動かす。
  ③ **天候弾性の厳密さ**(A1②): 雨日の減少量が目的別弾性そのもの。generated 以外は不活性。
  ④ **mon-sat バグの修正**(A1③): 名簿が土曜出勤と書いている個体が土曜に資格を持つ。
  ⑤ **cap 撤去**(A2): 資格者 = 在場者。cap も presence_rank も 1 度も引かない。
  ⑥ **既定 OFF の完全同値**: 新引数を省いた `present_for_day` は現行と 1 バイト一致。
  ⑦ **職業多様性**(B): 台帳の業種×細分類×ロール → 職業名。予約語との衝突が意図どおり。
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society.config import load_config                 # noqa: E402
from society.rng import RngHub                          # noqa: E402
from society.world import pool as pool_mod              # noqa: E402
from society.world import presence as P                 # noqa: E402
from society.world.presence import PresenceRec, present_for_day  # noqa: E402


# =========================================================================== #
# 合成名簿(実プールを触らずに層の性質だけを測る)
# =========================================================================== #
def _stoch(n: int, rates=None, purpose: str = "", seed: int = 7):
    """非 revisit の stochastic 層だけの名簿(revisit=False = 習慣カレンダーの対象)。"""
    rng = np.random.default_rng(seed)
    if rates is None:                     # 実プール L4 と同じ log-uniform [0.003, 0.06]
        rates = np.exp(rng.uniform(np.log(0.003), np.log(0.06), n))
    return [PresenceRec(pid=f"S_{i:07d}", key="stochastic",
                        visit_rate=float(rates[i]), visit_purpose=purpose)
            for i in range(n)]


def _habit(**over) -> dict:
    raw = {"habit": {"enabled": True, "weekday": False, "weather": False}}
    raw["habit"].update(over)
    return P.build_presence_cfg(raw)["habit"]


def _count(recs, day: int, hub, habit=None, rain=None, weekday=None) -> int:
    wd = day % 7 if weekday is None else weekday
    return sum(1 for r in recs if P._eligible(r, day, wd, hub, habit, rain))


# =========================================================================== #
# ① A1① 分布保存の較正証明 — 習慣カレンダーは Bernoulli と同じ期待値を持つ
# =========================================================================== #
def test_habit_daily_probability_equals_visit_rate_exactly():
    """★較正証明の芯: 同一レートの大集団で「今日居る割合」が visit_rate に一致する。

    θ_i ~ U[0,1) より frac(θ_i + r·d) は [0,1) 上一様 → 区間 (r·d, r·(d+1)] に整数が
    入る確率は区間長 r そのもの。丸め(period=round(1/r))を入れていないので**厳密**。
    """
    hub, habit = RngHub(42), _habit()
    n = 40_000
    for rate in (0.02, 0.05, 0.1, 0.25):
        recs = _stoch(n, rates=np.full(n, rate))
        got = [_count(recs, d, hub, habit) / n for d in range(5)]
        sd = math.sqrt(rate * (1 - rate) / n)       # 二項の標準誤差
        for f in got:
            assert abs(f - rate) < 4 * sd, (rate, f, sd)


def test_habit_population_expectation_matches_bernoulli_within_calibration_band():
    """実プール相当のレート分布で、集団の日次期待在場が Bernoulli と一致する(較正不変)。"""
    hub = RngHub(42)
    recs = _stoch(30_000)
    expected = sum(r.visit_rate for r in recs)      # = Σ visit_rate = 現行の期待在場
    habit = _count_mean(recs, hub, _habit())
    bern = _count_mean(recs, hub, None)
    assert abs(habit - expected) / expected < 0.01, (habit, expected)
    assert abs(bern - expected) / expected < 0.01, (bern, expected)
    # 2 つのモードの日次平均どうしも 1% 以内(= 較正を引き継げる)
    assert abs(habit - bern) / expected < 0.015


def _count_mean(recs, hub, habit, days: int = 70, rain=None) -> float:
    return statistics.mean(_count(recs, d, hub, habit, rain) for d in range(days))


def test_habit_preserves_each_individual_long_run_rate():
    """個人の長期来街レートも厳密保存(D 日の回数 = floor(θ+D·r) − floor(θ) ∈ {⌊D·r⌋, ⌈D·r⌉})。"""
    hub, habit = RngHub(42), _habit()
    days = 400
    for rec in _stoch(60, seed=11):
        got = sum(1 for d in range(days)
                  if P._eligible(rec, d, d % 7, hub, habit, None))
        want = days * rec.visit_rate
        assert abs(got - want) <= 1.0, (rec.pid, rec.visit_rate, got, want)


def test_habit_is_a_calendar_not_a_coin():
    """「今日行く日か」が個体固定の位相だけで決まる = 同じ人の来街日が規則的に並ぶ。"""
    hub, habit = RngHub(42), _habit()
    rec = PresenceRec(pid="S_regular", key="stochastic", visit_rate=0.25)
    days = [d for d in range(40) if P._eligible(rec, d, d % 7, hub, habit, None)]
    gaps = {b - a for a, b in zip(days, days[1:])}
    assert gaps <= {4}, gaps            # 1/0.25 = ちょうど 4 日おき(揺らぎゼロ)
    assert len(days) == 10


def test_habit_consumes_no_per_day_random_draw():
    """★乱数は**減る**: 当日 stream("presence", pid, day) を 1 度も引かない。"""
    class _Spy(RngHub):
        def __init__(self, seed):
            super().__init__(seed)
            self.keys: list[tuple] = []

        def stream(self, *key):
            self.keys.append(key)
            return super().stream(*key)

    recs = _stoch(200)
    spy_on, spy_off = _Spy(42), _Spy(42)
    _count(recs, 3, spy_on, _habit())
    _count(recs, 3, spy_off, None)
    assert not any(k[0] == "presence" for k in spy_on.keys)
    assert all(k[0] == "presence_habit" for k in spy_on.keys)
    assert all(k[0] == "presence" for k in spy_off.keys)
    # 個体固定 stream = day を鍵に含めない(= resume/日跨ぎで同じ習慣)
    assert all(len(k) == 2 for k in spy_on.keys)


def test_habit_is_deterministic_and_day_pure():
    """同 seed・同 day で完全一致(k 非依存・resume 不変の芯)。"""
    recs = _stoch(3000)
    habit = _habit()
    a = present_for_day(recs, 5, 10 ** 9, RngHub(42), 5, habit=habit)
    b = present_for_day(recs, 5, 10 ** 9, RngHub(42), 5, habit=habit)
    assert a == b == sorted(a)
    # 別の day では別集合(習慣カレンダーが動いている)
    c = present_for_day(recs, 6, 10 ** 9, RngHub(42), 6, habit=habit)
    assert set(a) != set(c)


# =========================================================================== #
# ② A1② 曜日プロファイル — 週次総量は不変・週内の配分だけが動く
# =========================================================================== #
def test_weekday_profiles_are_normalized_to_weekly_mean_one():
    """全プロファイルの 7 日和がちょうど 7.0(= 週平均 1.0)。"""
    habit = P.build_presence_cfg({"habit": {"enabled": True}})["habit"]
    assert set(habit["profile"]) == set(P._WEEKDAY_PROFILE_RAW)
    for name, prof in habit["profile"].items():
        assert len(prof) == 7
        assert abs(sum(prof) - 7.0) < 1e-9, (name, sum(prof))


def test_weekday_profile_preserves_weekly_total():
    """★週次の総来街量は曜日 ON/OFF で不変(既存 visit_rate 較正を 1 人も動かさない)。"""
    hub = RngHub(42)
    recs = []
    for i, purpose in enumerate(P._WEEKDAY_PROFILE_RAW):
        recs += [PresenceRec(pid=f"{i}_{j:06d}", key="stochastic",
                             visit_rate=0.05, visit_purpose=purpose)
                 for j in range(4000)]
    flat = sum(_count(recs, d, hub, _habit(weekday=False)) for d in range(28))
    prof = sum(_count(recs, d, hub, _habit(weekday=True)) for d in range(28))
    assert abs(prof - flat) / flat < 0.02, (prof, flat)


def test_weekday_profile_moves_leisure_to_the_weekend():
    """娯楽目的は週末に寄り、業務来訪は平日に寄る(向きの機械固定)。"""
    hub = RngHub(42)
    for purpose, weekend_heavier in (("観光・見物", True), ("エンタメ・イベント", True),
                                     ("ビジネス来訪", False), ("通院・用事", False)):
        recs = _stoch(8000, rates=np.full(8000, 0.05), purpose=purpose)
        habit = _habit(weekday=True)
        wk = statistics.mean(_count(recs, d, hub, habit) for d in (0, 1, 2, 3))
        we = statistics.mean(_count(recs, d, hub, habit) for d in (5, 6))
        assert (we > wk) is weekend_heavier, (purpose, wk, we)


def test_unknown_purpose_gets_no_covariate():
    """目的が無い/未知の個体には共変量を掛けない(推測で埋めない=正直側)。"""
    hub = RngHub(42)
    recs = _stoch(6000, rates=np.full(6000, 0.05), purpose="")
    a = [_count(recs, d, hub, _habit(weekday=True)) for d in range(7)]
    b = [_count(recs, d, hub, _habit(weekday=False)) for d in range(7)]
    assert a == b


# =========================================================================== #
# ③ A1② 天候 — 減少量は目的別弾性そのもの・generated 以外は不活性
# =========================================================================== #
def test_rain_reduces_by_the_declared_elasticity():
    """雨日の減少割合が `_RAIN_ELASTICITY` と一致する(母集団の減少量 = e·N)。"""
    hub = RngHub(42)
    habit = _habit(weekday=False, weather=True)
    for purpose, e in P._RAIN_ELASTICITY.items():
        recs = _stoch(20_000, rates=np.full(20_000, 0.1), purpose=purpose)
        dry = sum(_count(recs, d, hub, habit, rain=False) for d in range(6))
        wet = sum(_count(recs, d, hub, habit, rain=True) for d in range(6))
        got = 1.0 - wet / dry
        assert abs(got - e) < 0.02, (purpose, got, e)


def test_rain_elasticity_is_larger_on_weekends_and_capped():
    """週末は弾性が大きく(実測 −29% > 平日)、実測上限で頭打ちになる。"""
    habit = _habit(weather=True)
    for purpose in ("観光・見物", "買い物"):
        wk = P._rain_elasticity(purpose, 2, habit)
        we = P._rain_elasticity(purpose, 6, habit)
        assert we > wk
        assert we <= P._RAIN_MAX + 1e-12
    # 必要用事(業務・通院)は天候で減らない
    assert P._rain_elasticity("ビジネス来訪", 6, habit) == 0.0
    assert P._rain_elasticity("通院・用事", 6, habit) == 0.0


def test_rain_none_is_inactive():
    """rain=None(= generated 以外・weather OFF)では天候の経路を 1 度も通らない。"""
    hub = RngHub(42)
    recs = _stoch(4000, rates=np.full(4000, 0.1), purpose="観光・見物")
    habit = _habit(weather=True)
    assert [_count(recs, d, hub, habit, rain=None) for d in range(5)] == \
        [_count(recs, d, hub, habit, rain=False) for d in range(5)]


def test_weather_peek_is_inactive_outside_generated():
    """`weather.peek_bad_day` は synthetic / OFF では **None**(H1 熱中症と同型の宣言)。"""
    from society import weather as W

    class _Sim:
        weathercfg = {"enabled": True, "mode": "synthetic"}
        calendarcfg = {"enabled": True}
    assert W.peek_bad_day(_Sim(), 0) is None
    _Sim.weathercfg = {"enabled": False, "mode": "generated"}
    assert W.peek_bad_day(_Sim(), 0) is None
    _Sim.weathercfg = {"enabled": True, "mode": "generated", "params": {"months": {}}}
    assert W.peek_bad_day(_Sim(), 0) is None          # 較正月が無い = 不活性


def test_weather_peek_has_no_side_effects_on_provenance():
    """覗いても summary の generated 件数・使用系列を汚さない(観測の非汚染)。"""
    from society import weather as W

    class _Sim:
        pass
    sim = _Sim()
    sim.weathercfg = {"enabled": True, "mode": "synthetic"}
    sim.calendarcfg = {"enabled": True}
    W.peek_bad_day(sim, 0)
    assert getattr(sim, "_weather_stats", None) is None
    assert getattr(sim, "_weather_used", None) is None


# =========================================================================== #
# ④ A1③ mon-sat バグの修正
# =========================================================================== #
def _legacy_days_match(spec: str, weekday: int) -> bool:
    """修正前の `_days_match`(presence.py:78-84)の**逐語コピー**。"""
    if not spec or spec == "all":
        return True
    if spec in ("mon-fri", "school_day"):
        return weekday in P._WEEKDAYS
    return weekday in P._WEEKDAYS


@pytest.mark.parametrize("spec,days", [
    ("all", {0, 1, 2, 3, 4, 5, 6}),
    ("", {0, 1, 2, 3, 4, 5, 6}),
    ("mon-fri", {0, 1, 2, 3, 4}),
    ("mon-sat", {0, 1, 2, 3, 4, 5}),          # ★これが直った本体
    ("school_day", {0, 1, 2, 3, 4}),
    ("weekday", {0, 1, 2, 3, 4}),             # L5 duty_pattern が使う語
    ("sat-sun", {5, 6}),
    ("weekend", {5, 6}),
    ("tue-sun", {1, 2, 3, 4, 5, 6}),
    ("mon,wed,fri", {0, 2, 4}),
    ("mon-fri,sun", {0, 1, 2, 3, 4, 6}),
    ("zzz", {0, 1, 2, 3, 4}),                 # 解釈できない仕様は従来どおり平日後退
])
def test_days_match_reads_the_roster_intent(spec, days):
    assert {w for w in range(7) if P._days_match(spec, w)} == days


def test_mon_sat_was_the_only_behavioural_change_for_shipped_specs():
    """出荷されている曜日仕様のうち、旧実装と食い違うのは `mon-sat` の土曜だけ。

    (`all` は旧実装でも早期 return で True。`mon-fri` / `school_day` は明示分岐。
     `weekday` は旧実装の後退規則で偶然正しかった。)
    """
    shipped = ["all", "mon-fri", "mon-sat", "school_day", "weekday", ""]
    diff = {(s, w) for s in shipped for w in range(7)
            if P._days_match(s, w) != _legacy_days_match(s, w)}
    assert diff == {("mon-sat", 5)}


def test_mon_sat_fix_adds_saturday_workers_on_the_real_ledger(tmp_path):
    """実台帳(センサス較正)から作った名簿で、土曜の workday_shift が実際に増える。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    bpp.build_pool(tmp_path / "p", 42, 0.01, orgs, pop, total_target=1_000_000)
    recs = [r for r in pool_mod.PoolStore(tmp_path / "p").presence_records()
            if r.key == "workday_shift"]
    assert recs
    now = sum(1 for r in recs if P._days_match(r.work_days, 5))
    before = sum(1 for r in recs if _legacy_days_match(r.work_days, 5))
    assert now > before, (now, before)
    # 増分はちょうど mon-sat 個体の数(= 名簿の意図どおり)
    assert now - before == sum(1 for r in recs if r.work_days == "mon-sat") > 0


# =========================================================================== #
# ⑤ A2 cap 撤去(emergent)
# =========================================================================== #
def _mixed(resident=0, work=0, stochastic=0):
    recs = [PresenceRec(pid=f"RES_{i:06d}", key="resident") for i in range(resident)]
    recs += [PresenceRec(pid=f"WRK_{i:06d}", key="workday_shift", work_days="mon-fri")
             for i in range(work)]
    recs += [PresenceRec(pid=f"STO_{i:06d}", key="stochastic", visit_rate=1.0)
             for i in range(stochastic)]
    return recs


def test_emergent_admits_every_eligible_person():
    """★資格者 = 在場者(cap を一切見ない)。総在場は個人の予定の合算として創発する。"""
    recs = _mixed(resident=30, work=254, stochastic=25)
    got = present_for_day(recs, 0, 250, RngHub(42), 1, True, emergent=True)
    assert len(got) == 309                      # 30 + 254 + 25(cap 250 を超えて全員)
    eligible = [r.pid for r in recs if P._eligible(r, 0, 1, RngHub(42))]
    assert got == sorted(eligible)


def test_emergent_ignores_cap_completely():
    """cap を変えても結果が 1 バイトも動かない(cap が世界から消えている)。"""
    recs = _mixed(resident=30, work=254, stochastic=25)
    base = present_for_day(recs, 0, 250, RngHub(42), 1, True, emergent=True)
    for cap in (0, 1, 10, 10 ** 9):
        assert present_for_day(recs, 0, cap, RngHub(42), 1, True,
                               emergent=True) == base


def test_emergent_still_respects_the_calendar():
    """cap は消すが**予定は消さない**: 平日勤務者は週末に居ない(内生であって無条件ではない)。"""
    recs = _mixed(resident=30, work=254, stochastic=25)
    we = present_for_day(recs, 0, 250, RngHub(42), 6, emergent=True)
    assert not any(p.startswith("WRK_") for p in we)
    assert sum(1 for p in we if p.startswith("RES_")) == 30


def test_emergent_draws_no_ranking_randomness():
    """溢れた層を切らない = `presence_rank` を 1 度も引かない(乱数も減る)。"""
    class _Spy(RngHub):
        def __init__(self, seed):
            super().__init__(seed)
            self.keys: list[tuple] = []

        def stream(self, *key):
            self.keys.append(key)
            return super().stream(*key)

    recs = _mixed(resident=30, work=254, stochastic=25)
    spy = _Spy(42)
    present_for_day(recs, 0, 10, spy, 1, True, emergent=True)
    assert not any(k[0] == "presence_rank" for k in spy.keys)


def test_emergent_is_deterministic_across_two_identical_runs():
    recs = _mixed(resident=10, work=40, stochastic=200)
    habit = _habit()
    a = present_for_day(recs, 3, 5, RngHub(42), 3, emergent=True, habit=habit)
    b = present_for_day(recs, 3, 5, RngHub(42), 3, emergent=True, habit=habit)
    assert a == b == sorted(a)


# =========================================================================== #
# ⑥ 既定 OFF = 現行と完全同値
# =========================================================================== #
def test_defaults_are_byte_identical_to_the_current_path():
    """新引数を省いた呼び出しが、明示 OFF とも旧経路とも一致する。"""
    recs = _mixed(resident=30, work=254, stochastic=25) + _stoch(500)
    for tq in (False, True):
        base = present_for_day(recs, 2, 200, RngHub(42), 2, tq)
        assert base == present_for_day(recs, 2, 200, RngHub(42), 2, tq,
                                       habit=None, emergent=False, rain=None)
        off = P.build_presence_cfg(None)
        assert base == present_for_day(recs, 2, 200, RngHub(42), 2, tq,
                                       habit=off["habit"],
                                       emergent=off["emergent"])


def test_presence_cfg_defaults_and_validation():
    cfg = P.build_presence_cfg(None)
    assert cfg["mode"] == "quota" and cfg["emergent"] is False
    assert cfg["habit"]["enabled"] is False
    assert P.habit_on(None) is False and P.habit_on(cfg["habit"]) is False
    with pytest.raises(ValueError):
        P.build_presence_cfg({"mode": "turbo"})


def test_conf_defaults_and_registry_declaration():
    """出荷 conf の既定が現行経路で、新トグルがレジストリ/allowlist に宣言済み。"""
    from society import registry as R
    cfg = load_config([])
    assert str(cfg.pool.presence.mode) == "quota"
    assert cfg.pool.presence.habit.enabled is False
    on = load_config(["pool.presence.mode=emergent",
                      "pool.presence.habit.enabled=true"])
    assert str(on.pool.presence.mode) == "emergent"
    ids = {f.id for f in R.FEATURES}
    assert {"pool.presence.mode", "pool.presence.habit.enabled"} <= ids
    assert {"pool.presence.habit.weekday", "pool.presence.habit.weather"} <= set(R.ALLOWLIST)
    assert R.undeclared_toggles(load_config()) == []


def test_finals_conf_turns_habit_on_and_leaves_cap_removal_off():
    """本選 conf: A1 は ON・A2 は quota のまま(8/15 実測後に 1 行で ON できる形)。"""
    import yaml
    doc = yaml.safe_load((_ROOT / "conf" / "finals_observe.yaml").read_text(encoding="utf-8"))
    pres = doc["pool"]["presence"]
    assert pres["habit"] == {"enabled": True, "weekday": True, "weather": True}
    assert pres["mode"] == "quota"
    assert doc["weather"]["mode"] == "generated"      # 天候共変量が実働する前提


def test_slim_carries_visit_purpose(tmp_path):
    """PoolStore のスリム記述子が visit_purpose を運ぶ(A1② の共変量キー)。"""
    rec = {"id": "L4_1", "presence": "stochastic", "visit_rate": 0.02,
           "visit_purpose": "買い物"}
    slim = pool_mod._slim(rec)
    assert slim.visit_purpose == "買い物"
    assert slim.visit_purpose is sys.intern("買い物")     # 100万件で 7 個へ畳む
    assert pool_mod._slim({"id": "x"}).visit_purpose == ""


# =========================================================================== #
# ⑦ B 職業多様性(build_persona_pool の職業名対応表)
# =========================================================================== #
def _new_occupations() -> set[str]:
    out: set[str] = set()
    for d in bpp._OCC_BY_INDUSTRY.values():
        out |= set(d.values())
    for d in bpp._OCC_BY_SECTOR.values():
        out |= set(d.values())
    return out | set(bpp._OCC_SCHOOL.values())


@pytest.mark.parametrize("ikey,sector,role,want", [
    ("LS", "美容室", "スタイリスト", "美容師"),
    ("MW", "調剤薬局", "医療スタッフ", "薬剤師"),
    ("MW", "調剤薬局", "介護スタッフ", "登録販売者"),
    ("MW", "介護/福祉", "介護スタッフ", "介護士"),
    ("MW", "歯科", "医療スタッフ", "歯科衛生士"),
    ("WR", "アパレル小売", "販売スタッフ", "アパレル店員"),
    ("WR", "食品小売", "販売スタッフ", "食品販売員"),
    ("WR", "雑貨・ライフスタイル", "販売スタッフ", "雑貨販売員"),
    ("FB", "ダイニング", "キッチン", "調理師"),
    ("FB", "ベーカリー", "キッチン", "パン職人"),
    ("RE", "賃貸仲介/管理", "営業", "不動産営業"),
    ("SV", "警備", "スタッフ", "警備員"),
    ("SV", "施設管理/メンテ", "スタッフ", "設備保守員"),
    ("TR", "宅配拠点", "ドライバー", "宅配ドライバー"),
    ("ED", "音楽教室", "講師", "音楽講師"),
    ("PS", "税務・会計", "コンサルタント", "税理士"),
    ("CS", "郵便局窓口", "窓口", "郵便窓口係"),
])
def test_occupation_map_reads_the_ledger(ikey, sector, role, want):
    assert bpp.occupation_for(ikey, sector, role) == want


def test_office_five_roles_are_kept_for_it():
    """規律 ③: 情報通信業の現行 5 種は置き換えない(実在の職種名)。"""
    for role in ("エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート"):
        assert bpp.occupation_for("IT", "業務SaaS", role) == role


def test_unmapped_pairs_fall_back_to_the_role_name():
    """規律 ⑤: 写像に無い (業種, ロール) は推測せずロール名のまま。"""
    assert bpp.occupation_for("ZZ", "無い業種", "謎ロール") == "謎ロール"
    assert bpp.occupation_for("WR", "アパレル小売", "店長") == "店長"


def test_every_mapped_sector_exists_in_the_census_ledger():
    """規律 ②: 台帳に無い業種の職業を作らない(死んだ写像を残さない)。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    real = {(c["industry_key"], c.get("sector_detail", "")) for c in orgs["companies"]}
    dead = sorted(k for k in bpp._OCC_BY_SECTOR if k not in real)
    assert dead == [], dead
    keys = {c["industry_key"] for c in orgs["companies"]}
    assert set(bpp._OCC_BY_INDUSTRY) <= keys


def test_mapped_roles_exist_in_the_ledger_roles():
    """写像のロール名が台帳の roles(または夜勤 roles)に実在する。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    by_key: dict[str, set[str]] = {}
    for c in orgs["companies"]:
        by_key.setdefault(c["industry_key"], set()).update(c.get("roles") or [])
    bad = []
    for ikey, table in bpp._OCC_BY_INDUSTRY.items():
        bad += [(ikey, r) for r in table if r not in by_key.get(ikey, set())]
    for (ikey, _sector), table in bpp._OCC_BY_SECTOR.items():
        bad += [(ikey, r) for r in table if r not in by_key.get(ikey, set())]
    assert bad == [], bad


def test_night_shift_roles_are_never_remapped():
    """規律 ④: conf が名指しする夜勤ロール名は写像を通さない(設定と名簿を切らない)。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    night = {r for c in orgs["companies"]
             for r in ((c.get("night_shift") or {}).get("roles") or [])}
    assert {"常駐警備", "設備巡回", "夜間清掃"} <= night
    slots = bpp._build_L2_slots(orgs, 1.0)
    got = {occ for (_o, role, occ, _sp, _d) in slots if role in night}
    assert got <= night, got - night


def test_new_occupations_are_plain_nouns_without_brands():
    """規律 ①: 実在ブランド/企業/人名を想起させる語を 1 つも使わない(語彙の素性検査)。"""
    new = _new_occupations()
    assert new
    for occ in new:
        assert occ.strip() == occ and 1 < len(occ) <= 12, occ
        assert not any(ch in occ for ch in "()()株㈱・&@"), occ
        assert occ not in bpp._FAMILY_SET and occ not in bpp._GIVEN_SET


def test_collisions_with_reserved_occupation_words_are_the_declared_ones():
    """★予約語(賃金/職場/執行/警備/消防/街頭)との衝突が**宣言どおり**であること。

    ここが増えるということは「知らないうちに別レーンの機構へ人を流し込んだ」ということ。
    """
    from society import city_ops, facility_devices, incidents_env, street_life
    from society.agents.persona import _WORK_CAT
    from society.economy import CIVIL_SERVANTS, PART_TIME_OCC, WAGE_CAT
    new = _new_occupations()
    # ① 賃金・職場の既存表に当たる語 = この 3 つだけ(= 職場 POI と日給を持つ)
    assert sorted(new & set(WAGE_CAT)) == ["アパレル店員", "カフェ店員", "美容師"]
    assert sorted(new & set(_WORK_CAT)) == ["アパレル店員", "カフェ店員", "美容師"]
    # ② conf が名指ししていたのに名簿に 0 人だった語がここで生える(第108/109 の監査事実)
    assert "警備員" in new and "設備保守員" in new
    assert set(incidents_env.GUARD_OCCS) & new == {"警備員"}
    assert set(facility_devices.RESPONDER_OCCS) & new == {"警備員", "設備保守員"}
    # ③ 公務員・消防・街頭の生業・city_ops 役割へは 1 人も流し込まない
    assert not (new & set(CIVIL_SERVANTS))
    assert not (new & set(incidents_env.FIRE_OCCS))
    assert not (new & set(city_ops.CITY_OPS_OCCS))
    assert not (new & set(city_ops.POLICE_OCCS))
    assert not (new & set(street_life.STREET_OCCS))
    assert not (new & PART_TIME_OCC)
    # ④ 「写真家」= 自営(gig 収入)には**しない**(雇われの撮影者はカメラマン)
    assert "写真家" not in new and "カメラマン" in new


def test_occupation_map_increases_diversity_and_can_be_switched_off(tmp_path):
    """職業の種類が増え、`--no-occupation-map` で第108 までの形に戻せる。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    on, _c = bpp.build_pool(tmp_path / "on", 42, 0.02, orgs, pop, 1_000_000)
    off, _c2 = bpp.build_pool(tmp_path / "off", 42, 0.02, orgs, pop, 1_000_000,
                              occupations=False)
    assert on["occupation_map"] is True and off["occupation_map"] is False
    assert on["occupations_distinct"] > off["occupations_distinct"]
    assert sum(on["occupations"].values()) == sum(off["occupations"].values())
    # 層別件数・議員数は写像の有無で 1 件も動かない(人数は台帳が決める)
    assert on["layer_counts"] == off["layer_counts"]
    assert on["councilors"] == off["councilors"] == bpp.N_COUNCILORS


def test_l5_ids_and_councilors_are_untouched_by_the_occupation_map(tmp_path):
    """★L5 既存 id 不変・councilors バイト同一の掟(L5 は写像を 1 度も通らない)。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    _m1, c_on = bpp.build_pool(tmp_path / "on", 42, 0.02, orgs, pop, 1_000_000)
    _m2, c_off = bpp.build_pool(tmp_path / "off", 42, 0.02, orgs, pop, 1_000_000,
                                occupations=False)
    assert c_on == c_off
    shipped = json.loads((_ROOT / "data" / "personas_councilors.json")
                         .read_text(encoding="utf-8"))
    assert [p["id"] for p in shipped["personas"]] == [c["id"] for c in c_on]
    assert [p["name"] for p in shipped["personas"]] == [c["name"] for c in c_on]

    def l5(d: Path):
        out = []
        for f in sorted((d / "L5").glob("part-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line:
                    r = json.loads(line)
                    out.append((r["id"], r["occupation"], r.get("role", "")))
        return out
    assert l5(tmp_path / "on") == l5(tmp_path / "off")


def test_persona_sentence_uses_occupation_and_keeps_the_ledger_role(tmp_path):
    """persona 文が職業名で始まり、社内ロールは括弧で残る(整合の機械固定)。"""
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    bpp.build_pool(tmp_path / "p", 42, 0.02, orgs, pop, 1_000_000)
    seen = Counter()
    for f in sorted((tmp_path / "p" / "L2").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            assert f"歳の{r['occupation']}(" in r["persona"], r["persona"]
            assert f"渋谷({r['role']})に" in r["persona"], r["persona"]
            seen[r["occupation"]] += 1
    assert len(seen) > 40                     # ロール 40 種を超える職業の多様性


# =========================================================================== #
# ⑧ 実ラン統合(ON 縦煙): 同 seed 2 ラン同値 / resume 跨ぎ / 在場 = 資格者
# =========================================================================== #
@pytest.fixture(scope="module")
def census_pool(tmp_path_factory):
    """センサス較正台帳から作る小プール(mon-sat と visit_purpose が実物で入る)。"""
    out = tmp_path_factory.mktemp("pres_pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_census.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _sim_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


_ON = {"pool.presence.habit.enabled": "true", "pool.tier_quota.enabled": "true"}


def test_on_two_identical_runs_match(census_pool, tmp_path):
    """★ON(habit + emergent)の同 seed 2 ランが L1 まで完全一致(決定論)。"""
    from society.engine.simulation import Simulation

    def l1(sim):
        return [(e.step, e.agent_id, e.kind,
                 json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
                for e in sim.logger.events]
    ov = dict(_ON, **{"pool.presence.mode": "emergent"})
    a = Simulation(_sim_cfg("on1", census_pool, 160, **ov), out_dir=tmp_path / "on1")
    a.run()
    b = Simulation(_sim_cfg("on2", census_pool, 160, **ov), out_dir=tmp_path / "on2")
    b.run()
    assert l1(a) == l1(b)
    assert [e.payload for e in a.logger.events if e.kind == "presence_change"] == \
        [e.payload for e in b.logger.events if e.kind == "presence_change"]


def test_emergent_run_present_equals_eligible(census_pool, tmp_path):
    """★縦煙 ④: emergent の実ランで在場数が**資格者数と一致**する(cap が効いていない)。"""
    from society.engine.simulation import Simulation
    ov = dict(_ON, **{"pool.presence.mode": "emergent"})
    sim = Simulation(_sim_cfg("emg", census_pool, 160, cap=50, **ov),
                     out_dir=tmp_path / "emg")     # cap=50 は在場より遥かに小さい
    sim.run()
    recs = sim._pool.presence_records()
    habit = sim._pool_presence["habit"]
    assert sim._pool_presence["emergent"] is True
    for e in sim.logger.events:
        if e.kind != "presence_change":
            continue
        day = int(e.payload["day"])
        want = sum(1 for r in recs
                   if P._eligible(r, day, sim._pool_weekday(day), sim.hub, habit, None))
        assert e.payload["n_present"] == want > 50, (day, e.payload, want)


def test_on_resume_byte_matches_straight(census_pool, tmp_path):
    """★ON(habit + emergent)でも「一気 vs 中断→resume」の l1_events が完全一致。"""
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler
    from society.engine.simulation import Simulation

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()
    ov = dict(_ON, **{"pool.presence.mode": "emergent"})
    st = tmp_path / "st"
    Simulation(_sim_cfg("rst", census_pool, 200, **ov), out_dir=st).run()

    rs = tmp_path / "rs"
    s1 = Simulation(_sim_cfg("rrs", census_pool, 100,
                             **dict(ov, **{"observer.checkpoint_every": 100})), out_dir=rs)
    for step in range(100):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 100, rs / "checkpoint" / "ckpt-000100.pkl.gz")
    s1._save_pool_sidecar(100)
    s1.logger.flush_segment()
    s2 = Simulation(_sim_cfg("rrs", census_pool, 200,
                             **dict(ov, **{"observer.checkpoint_every": 100})), out_dir=rs)
    s2.run(resume_from=rs)
    assert rows(st) == rows(rs), "ON の resume が straight と byte 不一致"


def test_on_run_keeps_lane_109_features_alive(census_pool, tmp_path):
    """★縦煙 ⑤: ON にしても第109 の結線(pool 経路の org 配属)が生きている。"""
    from society.engine.simulation import Simulation
    ov = dict(_ON, **{
        "pool.presence.mode": "emergent",
        "organizations.enabled": "true",
        "organizations.book": "data/organizations_shibuya_census.json"})
    sim = Simulation(_sim_cfg("l109", census_pool, 1, **ov), out_dir=tmp_path / "l109")
    attached = [a for a in sim.agents if getattr(a, "org_id", None)]
    assert attached, "pool 経路の org 配属が 0 件(第109 ORG の結線が死んでいる)"
    assert sim._pool_org_stat["n_attached"] == len(attached)
    # 職業名対応表が実際に届いている(台帳ロール名でない職業を持つ個体が居る)
    occs = {str(getattr(a, "occupation", "")) for a in sim.agents}
    assert occs & {"調理師", "販売員", "接客係", "事務員", "不動産営業", "看護師"}
