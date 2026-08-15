"""AGE-F: 加齢・誕生日(第116バッチ 2026-08-15・既定 OFF)。

正典: docs/plans/age-diversity-plan.md §4-8。実装: src/society/aging.py。

本テストが機械固定すること:
- **既定 OFF はバイト一致**(誕生日イベント 0 件・age 不変)。
- **暦 OFF では完全 no-op**(捏造した暦で誕生日を作らない)。
- **1 個体 1 年 1 回**・366 日ランで全員がちょうど 1 回歳を取る(誕生日の網羅性)。
- **決定論**: 誕生日は (seed, persona id) の純関数でプロセス跨ぎ・別ランで同一。
- **新しい L1 kind を足さない**(既存 life_event の下位 kind)。
- **年齢キャッシュの無効化**: 歳を取ったら AGE-C の思考係数が作り直される。
"""
from __future__ import annotations

import datetime
import json

from society import aging
from society.config import load_config
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _birthdays(sim):
    return [e for e in sim.logger.events
            if e.kind == "life_event" and e.payload.get("kind") == "birthday"]


# --------------------------------------------------------------------- OFF
def test_off_matches_pure_default(tmp_path):
    """暦 ON + aging OFF が純粋な暦 ON と L1 完全一致(seam が no-op)。"""
    cal = {"world.calendar.enabled": "true"}
    base = _sim(tmp_path, "aging_base", steps=144, **cal)
    base.run()
    off = _sim(tmp_path, "aging_off", steps=144, **{**cal, "aging.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    assert not _birthdays(off)


def test_no_calendar_is_noop(tmp_path):
    """暦 OFF では aging ON でも 1 件も起きない(実日付が無いのに誕生日を作らない)。"""
    sim = _sim(tmp_path, "aging_nocal", steps=288, **{"aging.enabled": "true"})
    ages = {a.id: a.age for a in sim.agents}
    sim.run()
    assert not _birthdays(sim), "暦 OFF なのに誕生日が発生した"
    assert {a.id: a.age for a in sim.agents} == ages, "暦 OFF なのに年齢が動いた"


# --------------------------------------------------------------------- 決定論
def test_birthday_is_a_deterministic_pure_function():
    """誕生日は (seed, persona id) の純関数(同じ入力 → 同じ日・seed を変えると変わる)。"""
    class _A:
        id = 42
    cfg = aging.build_cfg({"enabled": True})
    assert aging.birthday_of(_A(), cfg) == aging.birthday_of(_A(), cfg)
    other = aging.build_cfg({"enabled": True, "seed": 999})
    diffs = 0
    for i in range(200):
        class _B:
            id = i
        if aging.birthday_of(_B(), cfg) != aging.birthday_of(_B(), other):
            diffs += 1
    assert diffs > 150, "seed を変えても誕生日がほとんど動かない(ハッシュが効いていない)"


def test_birthdays_cover_the_year_and_skip_feb_29():
    """誕生日は 1〜365 日目に散り、2/29 は選ばれない(平年に消える個体を作らない)。"""
    class _A:
        def __init__(self, i):
            self.id = i
    cfg = aging.build_cfg({"enabled": True})
    days = {aging.birthday_of(_A(i), cfg) for i in range(5000)}
    assert (2, 29) not in days, "2/29 が誕生日に選ばれている"
    assert len(days) == 365, f"365 日を網羅していない: {len(days)}"
    for m, d in days:
        datetime.date(2001, m, d)          # 実在しない日付なら例外


# --------------------------------------------------------------------- ON
def test_everyone_ages_exactly_once_per_year(tmp_path):
    """暦を 1 年進めると、全員がちょうど 1 回だけ歳を取る(1 個体 1 年 1 回)。"""
    sim = _sim(tmp_path, "aging_year", n=12, steps=1,
               **{"aging.enabled": "true", "world.calendar.enabled": "true"})
    before = {a.id: a.age for a in sim.agents}
    cal = sim.calendarcfg
    seen = {}
    for day in range(365):                 # 日境界を 365 回だけ直接叩く(ラン全長は不要)
        sim._aging_day = -1                # 日ガードを外して当該日を処理させる
        aging.phase_day(sim, step=day * 144, sim_min=day * 1440)
    for e in _birthdays(sim):
        seen[e.agent_id] = seen.get(e.agent_id, 0) + 1
    assert set(seen) == set(before), f"歳を取らなかった個体が居る: {set(before) - set(seen)}"
    assert set(seen.values()) == {1}, f"1 年に 2 回以上歳を取った個体が居る: {seen}"
    assert all(a.age == before[a.id] + 1 for a in sim.agents)
    assert cal["enabled"]


def test_day_guard_prevents_double_count(tmp_path):
    """同じ日を二度叩いても 2 回歳を取らない(resume 安全の日境界ガード)。"""
    sim = _sim(tmp_path, "aging_guard", n=40, steps=1,
               **{"aging.enabled": "true", "world.calendar.enabled": "true"})
    cfg = aging.cfg_of(sim)
    from society.world import calendar as _cal
    # 誰か 1 人の誕生日にあたる日を探して、その日を 2 回叩く。
    target = sim.agents[0]
    m, d = aging.birthday_of(target, cfg)
    start = datetime.date.fromisoformat(sim.calendarcfg["start_date"])
    for off in range(0, 400):
        if (start + datetime.timedelta(days=off)).month == m \
                and (start + datetime.timedelta(days=off)).day == d:
            break
    sim_min = off * 1440
    assert _cal.date_of(sim.calendarcfg, sim_min) == start + datetime.timedelta(days=off)
    age0 = target.age
    aging.phase_day(sim, step=0, sim_min=sim_min)
    aging.phase_day(sim, step=1, sim_min=sim_min)      # 同じ日を再度
    assert target.age == age0 + 1, "同じ日で二重に加齢した"


def test_age_derived_caches_are_invalidated(tmp_path):
    """歳を取ったら AGE-C の思考係数キャッシュが作り直される(古い年齢が残らない)。"""
    from society.cognition import age_cog
    sim = _sim(tmp_path, "aging_cache", n=8, steps=1,
               **{"aging.enabled": "true", "world.calendar.enabled": "true",
                  "cognition.age.enabled": "true"})
    a = sim.agents[0]
    a.age = 44
    a._age_cog = None
    before = age_cog.period_mult(sim, a)
    cfg = aging.cfg_of(sim)
    m, d = aging.birthday_of(a, cfg)
    start = datetime.date.fromisoformat(sim.calendarcfg["start_date"])
    off = next(o for o in range(400)
               if (start + datetime.timedelta(days=o)).month == m
               and (start + datetime.timedelta(days=o)).day == d)
    aging.phase_day(sim, step=0, sim_min=off * 1440)
    assert a.age == 45
    assert age_cog.period_mult(sim, a) != before, "年齢キャッシュが古いまま残っている"


def test_uses_existing_life_event_kind(tmp_path):
    """新しい L1 kind を足していない(既存 life_event の下位 kind に相乗り)。"""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "src/society/aging.py").read_text(encoding="utf-8")
    assert "register_event_kind" not in src, "aging が新しい L1 kind を登録している"
    assert 'kind="life_event"' in src
