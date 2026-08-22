"""第144: 台帳の営業曜日を勤務ゲートへ / 曜日時計の暦整合(既定 OFF=現行と 1 バイト同一)。

何を解く問題か(デバッグ済み・根本原因確定)
--------------------------------------------
本選 conf は `start_date: "2026-08-22"`(**土曜**)× `weekday_work: true` で走る。
`routine.in_work_window` の暦ゲートは「土日は全員休み」しか表現できないので、初日の
勤務が 144 step すべて全滅していた(在勤 0 人 → staffed 供給 / serve / 賃金が丸ごとゼロ)。
ところが組織台帳は `shift_pattern.days` で営業曜日を**既に宣言している**
(センサス較正台帳: mon-fri 3,971 社 / all 3,929 社 / mon-sat 1,972 社 = 土曜営業 50.4%)。
職場束ね(`work.bind_workplace`)がその days を捨てていたのが直接の原因である。

加えて在場(presence)の曜日時計は `day % 7`(day0=**月曜**)で、暦(8/22=土)と 5 日ぶん
位相がずれていた。「暦の曜日」と「presence の曜日」が別物という割り切りは、暦 ON の
ランでは名簿の曜日宣言と勤務ゲートの食い違いとして表面化する。

検証(すべて mock / 決定論。新しい乱数 stream も LLM 呼も 1 本も足していない)
  (1) 曜日パーサの移設(presence → calendar)が**規則も戻り値も変えない**
  (2) `respect_work_days` OFF = 現行同値(土曜に全滅)/ ON = 宣言どおり
  (3) 賃金ゲート(`scheduler._wage_worked_today`)が勤務ゲートと**同値**
  (4) `work_dow` が pool 退避 ⇄ 再来街 と checkpoint 往復で保存される
  (5) `calendar_weekday` ON で presence とバイトの曜日が暦の曜日になる(OFF は day%7)
  (6) 契約列挙(新 conf キーの存在・既定値・レジストリ宣言)
  (7) 統合ミニ: 土曜始まりの小規模ランで在勤が立ち上がる(OFF ではゼロ)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                        # noqa: E402
from society import home_awake as home_awake_mod        # noqa: E402
from society import registry as R                       # noqa: E402
from society import work as work_mod                    # noqa: E402
from society.agents.agent import Agent                  # noqa: E402
from society.agents.memory import MemoryStore           # noqa: E402
from society.cognition import plan_schema               # noqa: E402
from society.cognition import routine                   # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine import checkpoint, scheduler        # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.world import calendar as cal_mod           # noqa: E402
from society.world import pool as pool_mod              # noqa: E402
from society.world import presence as presence_mod      # noqa: E402

SAT = "2026-08-22"          # 本選初日(土)
SUN = "2026-08-23"          # 日
MON = "2026-08-24"          # 月


def _cal(*, enabled=True, start=SAT, weekday_work=True,
         respect=False, calendar_weekday=False) -> dict:
    return cal_mod.build_cfg({"enabled": enabled, "start_date": start,
                              "weekday_work": weekday_work,
                              "respect_work_days": respect,
                              "calendar_weekday": calendar_weekday})


def _worker(dow, *, start_min: int = 9 * 60, end_min: int = 18 * 60):
    """勤務窓を持つ最小の個体(`in_work_window` / `_wage_worked_today` が読む欄だけ)。"""
    a = SimpleNamespace(work_start_min=start_min, work_end_min=end_min, sick=False)
    if dow is not None:
        a.work_dow = dow
    return a


# =========================================================================== #
# (1) 曜日パーサの移設: presence 側は同じ関数を再輸出しているだけ
# =========================================================================== #
def test_presence_reexports_the_moved_parser():
    """`world/presence.py` の `_days_match` / `_parse_days` / `_WEEKDAYS` は calendar の実体。"""
    assert presence_mod._days_match is cal_mod.days_match
    assert presence_mod._parse_days is cal_mod.parse_days
    assert presence_mod._WEEKDAYS is cal_mod.WEEKDAYS


@pytest.mark.parametrize("spec,days", [
    ("mon-fri", {0, 1, 2, 3, 4}),
    ("mon-sat", {0, 1, 2, 3, 4, 5}),
    ("all", set(range(7))),
    ("", set(range(7))),                       # 無宣言 = 「絞らない」(呼び出し側が後退)
    ("sat-sun", {5, 6}),
    ("sun-tue", {6, 0, 1}),                    # 折り返し
    ("mon,wed,fri", {0, 2, 4}),
    ("mon-fri,sun", {0, 1, 2, 3, 4, 6}),
    ("school_day", {0, 1, 2, 3, 4}),
    ("weekend", {5, 6}),
    ("解釈できない語", {0, 1, 2, 3, 4}),        # 解釈不能 = 平日扱いへ後退(安全側)
])
def test_calendar_days_match_direct_cases(spec, days):
    assert {w for w in range(7) if cal_mod.days_match(spec, w)} == days


def test_calendar_parse_days_returns_none_for_unparsable():
    assert cal_mod.parse_days("") is None
    assert cal_mod.parse_days("解釈できない語") is None
    assert cal_mod.parse_days("mon-sat") == frozenset({0, 1, 2, 3, 4, 5})


# =========================================================================== #
# (2) respect_work_days: OFF = 現行同値 / ON = 宣言どおり
# =========================================================================== #
def test_off_is_identical_to_current_behaviour():
    """OFF では `work_dow` を宣言していても**読まれない** = 土曜は全員が勤務時間外。"""
    off = _cal(respect=False)
    declared = _worker("mon-sat")
    plain = _worker(None)
    for sim_min in (10 * 60, 13 * 60, 17 * 60):
        assert routine.in_work_window(declared, sim_min, off) is False
        assert routine.in_work_window(plain, sim_min, off) is False
        # 宣言の有無で 1 ビットも変わらない(= work_dow が完全に不活性)
        assert routine.in_work_window(declared, sim_min, off) \
            == routine.in_work_window(plain, sim_min, off)


def test_off_weekday_is_unchanged():
    """OFF の平日(月曜始まり)は従来どおり勤務窓が開く(退行が無いことの対照)。"""
    off = _cal(start=MON, respect=False)
    assert routine.in_work_window(_worker("mon-sat"), 10 * 60, off) is True
    assert routine.in_work_window(_worker(None), 10 * 60, off) is True


def test_on_saturday_opens_only_for_saturday_shops():
    """ON: 土曜は mon-sat / all の宣言者だけが勤務窓に入る(mon-fri は入らない)。"""
    on = _cal(start=SAT, respect=True)
    assert routine.in_work_window(_worker("mon-sat"), 10 * 60, on) is True
    assert routine.in_work_window(_worker("all"), 10 * 60, on) is True
    assert routine.in_work_window(_worker("mon-fri"), 10 * 60, on) is False
    # 無宣言(学校など shift_pattern を持たない層)は**従来どおり**平日ゲートへ後退
    assert routine.in_work_window(_worker(None), 10 * 60, on) is False
    assert routine.in_work_window(_worker(""), 10 * 60, on) is False


def test_on_sunday_opens_only_for_all():
    """ON: 日曜は all の宣言者だけ。mon-sat も mon-fri も閉じる。"""
    on = _cal(start=SUN, respect=True)
    assert routine.in_work_window(_worker("all"), 10 * 60, on) is True
    assert routine.in_work_window(_worker("mon-sat"), 10 * 60, on) is False
    assert routine.in_work_window(_worker("mon-fri"), 10 * 60, on) is False
    assert routine.in_work_window(_worker(None), 10 * 60, on) is False


def test_on_weekday_is_unchanged_for_every_spec():
    """ON でも平日は全員が開く(= ON は「土日を開ける」方向にしか効かない)。"""
    on = _cal(start=MON, respect=True)
    for spec in ("mon-fri", "mon-sat", "all", None, ""):
        assert routine.in_work_window(_worker(spec), 10 * 60, on) is True, spec


def test_on_still_respects_time_of_day_and_sickness():
    """ON でも「時刻の窓」と病欠は従来どおり効く(曜日ゲート以外は 1 行も変えていない)。"""
    on = _cal(start=SAT, respect=True)
    assert routine.in_work_window(_worker("mon-sat"), 8 * 60, on) is False   # 始業前
    assert routine.in_work_window(_worker("mon-sat"), 19 * 60, on) is False  # 終業後
    sick = _worker("mon-sat")
    sick.sick = True
    assert routine.in_work_window(sick, 10 * 60, on) is False


def test_calendar_disabled_or_weekday_work_off_is_a_full_noop():
    """暦 OFF / weekday_work OFF なら respect_work_days は完全 no-op(親トグルの契約)。"""
    for cal in (_cal(enabled=False, respect=True), _cal(weekday_work=False, respect=True)):
        for spec in ("mon-fri", "mon-sat", None):
            assert routine.in_work_window(_worker(spec), 10 * 60, cal) is True, spec


def test_second_day_uses_the_second_days_weekday():
    """判定は sim_min の実日付から引く(土曜始まりの翌日 = 日曜で mon-sat が閉じる)。"""
    on = _cal(start=SAT, respect=True)
    assert routine.in_work_window(_worker("mon-sat"), 10 * 60, on) is True          # 8/22 土
    assert routine.in_work_window(_worker("mon-sat"), 1440 + 10 * 60, on) is False  # 8/23 日
    assert routine.in_work_window(_worker("mon-sat"), 2880 + 10 * 60, on) is True   # 8/24 月


# =========================================================================== #
# (3) 賃金ゲートが勤務ゲートと同値(会計の破れを構造的に禁じる)
# =========================================================================== #
@pytest.mark.parametrize("start", [SAT, SUN, MON])
@pytest.mark.parametrize("respect", [False, True])
@pytest.mark.parametrize("spec", ["mon-fri", "mon-sat", "all", None])
def test_wage_gate_matches_work_gate(start, respect, spec):
    """同一個体・同一 sim_min で `_wage_worked_today` と `in_work_window` の真偽が一致する。"""
    cal = _cal(start=start, respect=respect)
    sim = SimpleNamespace(calendarcfg=cal)
    agent = _worker(spec)
    sim_min = 10 * 60                              # 勤務窓の内側(時刻条件は両者で同じ)
    assert scheduler._wage_worked_today(sim, agent, sim_min) \
        == routine.in_work_window(agent, sim_min, cal)


def test_wage_gate_ignores_time_of_day():
    """賃金ゲートは「時刻の条件だけ外す」= 曜日の判定は勤務ゲートと同じ(既存契約の再確認)。"""
    cal = _cal(start=SAT, respect=True)
    sim = SimpleNamespace(calendarcfg=cal)
    assert scheduler._wage_worked_today(sim, _worker("mon-sat"), 3 * 60) is True
    assert scheduler._wage_worked_today(sim, _worker("mon-fri"), 3 * 60) is False


@pytest.mark.parametrize("respect", [False, True])
@pytest.mark.parametrize("spec", ["mon-fri", "mon-sat", "all", None])
def test_home_awake_early_tomorrow_uses_the_same_gate(respect, spec):
    """「翌日は勤務日か」を答える 3 つ目の窓口(在宅覚醒 β9)も同じ規則で答える。

    金曜(8/21)の夜に「明日=土曜は出勤日か」を訊く。ON なら mon-sat / all だけが True。
    """
    cal = _cal(start="2026-08-21", respect=respect)          # 金
    sim = SimpleNamespace(calendarcfg=cal)
    agent = _worker(spec, start_min=7 * 60)                  # early_start_min(480)より早い
    got = home_awake_mod.early_tomorrow(agent, sim, 22 * 60, {"early_start_min": 480})
    want = routine.in_work_window(agent, 1440 + 8 * 60, cal)  # 翌日の勤務窓(08:00)
    assert got is want


def test_wage_gate_unemployed_and_sick_unchanged():
    cal = _cal(start=SAT, respect=True)
    sim = SimpleNamespace(calendarcfg=cal)
    jobless = _worker("all", start_min=-1)
    assert scheduler._wage_worked_today(sim, jobless, 10 * 60) is False
    sick = _worker("all")
    sick.sick = True
    assert scheduler._wage_worked_today(sim, sick, 10 * 60) is False


# =========================================================================== #
# (4) work_dow の生え方と搬送(pool 退避 ⇄ 再来街 / checkpoint 往復)
# =========================================================================== #
class _StubCity:
    """`_resolve_node` / `_resolve_building` が触る最小の市街地図(路面の職場 1 つ)。"""

    graph = {"n_stub": True}

    def pois_at_node(self, node):
        return [{"id": "p_stub", "cat": "shop", "building": None}]

    def pois_by_cat(self, cat):
        return [{"id": "p_stub", "cat": "shop", "node": "n_stub", "building": None}] \
            if cat == "shop" else []

    def has_building(self, bid):
        return False

    def buildings_at(self, node):
        return []


def _bind_cfg(**ov) -> dict:
    raw = {"bind_workplace": {"enabled": True, "poi_match_fallback": False, **ov}}
    return work_mod.build_cfg(raw)["bind_workplace"]


def _fresh_agent(aid: int = 1):
    a = Agent(id=aid, name=f"甲{aid}", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    a.work_start_min = -1
    return a


def test_bind_workplace_copies_days_from_the_ledger():
    """台帳 entry の days が `agent.work_dow` へ写る(勤務窓も従来どおり補われる)。"""
    book = {"o1": {"node": "n_stub", "open": "10:00", "close": "20:00", "days": "mon-sat"}}
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"org_id": "o1"}, _StubCity(), book, _bind_cfg())
    assert a.work_node == "n_stub"
    assert a.work_dow == "mon-sat"
    assert (a.work_start_min, a.work_end_min) == (10 * 60, 20 * 60)


def test_bind_workplace_falls_back_to_the_record_shift_pattern():
    """台帳に days が無ければ record.shift_pattern.days を使う(プール record は両方を持つ)。"""
    book = {"o1": {"node": "n_stub", "open": "10:00", "close": "20:00", "days": None}}
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"org_id": "o1", "shift_pattern": {"days": "all"}},
                            _StubCity(), book, _bind_cfg())
    assert a.work_dow == "all"


def _duty_bind_cfg():
    """org_id を持たない L5 役割職を束ねる構成(occ_cat 写像 + POI マッチ)。"""
    return work_mod.build_cfg({"bind_workplace": {"enabled": True,
                                                  "poi_match_fallback": True,
                                                  "occ_cat": {"当番役": "shop"}}})["bind_workplace"]


@pytest.mark.parametrize("days,opens_on_saturday", [("all", True), ("weekday", False),
                                                    ("mon-fri", False), ("mon-sat", True)])
def test_duty_pattern_days_reach_the_agent(days, opens_on_saturday):
    """L5 役割職(shift_pattern を持たず duty_pattern.days に当番表を書く層)にも曜日が届く。

    ★ユーザー決定「census 宣言曜日の一貫適用」。これが無いと respect_work_days ON でも
      駅員・警察官の類だけが土日に持ち場から消える(presence は duty_days で在場させるので、
      在場しているのに勤務窓が閉じている、という食い違いになる)。
    """
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"role": "当番役", "duty_pattern": {"days": days}},
                            _StubCity(), {}, _duty_bind_cfg())
    assert a.work_node == "n_stub", "束ねが成立していない(前提が崩れている)"
    assert a.work_dow == days
    on = _cal(start=SAT, respect=True)
    assert routine.in_work_window(a, 10 * 60, on) is opens_on_saturday
    # 既定 OFF は従来どおり(土曜は誰も開かない)
    assert routine.in_work_window(a, 10 * 60, _cal(start=SAT, respect=False)) is False
    # 平日は ON / OFF どちらでも開く
    assert routine.in_work_window(a, 2 * 1440 + 10 * 60, on) is True


def test_shift_pattern_wins_over_duty_pattern():
    """後退の順は 台帳 → shift_pattern → duty_pattern(先に当たったものが勝つ)。"""
    book = {"o1": {"node": "n_stub", "open": "10:00", "close": "20:00", "days": "mon-fri"}}
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"org_id": "o1", "shift_pattern": {"days": "mon-sat"},
                                "duty_pattern": {"days": "all"}},
                            _StubCity(), book, _bind_cfg())
    assert a.work_dow == "mon-fri"                       # 台帳が最優先
    b = _fresh_agent(2)
    work_mod.bind_workplace(b, {"org_id": "o1", "shift_pattern": {"days": "mon-sat"},
                                "duty_pattern": {"days": "all"}},
                            _StubCity(),
                            {"o1": {**book["o1"], "days": None}}, _bind_cfg())
    assert b.work_dow == "mon-sat"                       # 台帳に無ければ shift_pattern


def test_duty_pattern_absent_grows_no_attribute():
    """duty_pattern を持たない層(議員など days=None)は従来どおり属性が生えない。"""
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"role": "当番役", "duty_pattern": {"days": None}},
                            _StubCity(), {}, _duty_bind_cfg())
    assert a.work_node == "n_stub"
    assert "work_dow" not in a.__dict__


def test_bind_workplace_grows_no_attribute_without_a_declaration():
    """宣言がどこにも無ければ属性を**生やさない**(getattr の既定 "" へ落ちる = 従来の後退)。"""
    book = {"o1": {"node": "n_stub", "open": "10:00", "close": "20:00", "days": None}}
    a = _fresh_agent()
    work_mod.bind_workplace(a, {"org_id": "o1"}, _StubCity(), book, _bind_cfg())
    assert "work_dow" not in a.__dict__
    assert getattr(a, "work_dow", "") == ""


def test_bind_workplace_does_not_touch_the_int_work_days_counter():
    """`agent.work_days`(給料日までの勤務日数・int)は**別物**なので 1 も動かさない。"""
    book = {"o1": {"node": "n_stub", "open": "10:00", "close": "20:00", "days": "mon-sat"}}
    a = _fresh_agent()
    a.work_days = 7
    work_mod.bind_workplace(a, {"org_id": "o1"}, _StubCity(), book, _bind_cfg())
    assert a.work_days == 7 and isinstance(a.work_days, int)


def test_load_bind_book_carries_days_for_every_org():
    """台帳ローダが days を落とさない(会社は全社が宣言・学校は timetable のみ = None)。"""
    book = work_mod.load_bind_book({"book": "data/organizations_shibuya_census.json"}, _ROOT)
    assert all("days" in e for e in book.values())
    declared = [e["days"] for e in book.values() if e["days"]]
    assert set(declared) == {"mon-fri", "mon-sat", "all"}
    assert len(declared) == 9872                      # 会社 9,872 社は全社が宣言済み


def test_work_dow_survives_the_pool_rotation():
    """退避(dehydrate)→ 再来街(hydrate)で `work_dow` が保存される。"""
    a = _fresh_agent()
    a.work_dow = "mon-sat"
    state = pool_mod.dehydrate(a)
    assert state["misc"]["work_dow"] == "mon-sat"
    back = _fresh_agent()
    pool_mod.hydrate(back, state)
    assert back.work_dow == "mon-sat"


def test_pool_state_is_byte_identical_without_a_declaration():
    """宣言の無い個体(束ね OFF のラン)では退避 dict に 1 バイトも足さない。"""
    state = pool_mod.dehydrate(_fresh_agent())
    assert "work_dow" not in (state.get("misc") or {})


def test_work_dow_survives_a_checkpoint_roundtrip(tmp_path):
    """checkpoint(agents pickle)往復で `work_dow` が戻る = resume で勤務曜日を失わない。"""
    def mk(name, steps):
        dot = ["run.seed=42", "run.n_agents=12", f"run.n_steps={steps}",
               f"run.name={name}", "model.backend=mock"]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    sim = mk("ckpt", 4)
    for step in range(2):
        scheduler.run_step(sim, step)
    sim.agents[0].work_dow = "mon-sat"
    path = checkpoint.save(sim, 2, tmp_path / "ckpt" / "checkpoint" / "ckpt-000002.pkl.gz")
    fresh = mk("ckpt2", 4)
    assert checkpoint.load(fresh, path) == 2
    assert getattr(fresh.agents[0], "work_dow", "") == "mon-sat"
    # 宣言していない個体には属性が生えないまま戻る(生やさない契約が往復でも保たれる)
    assert not any("work_dow" in a.__dict__ for a in fresh.agents[1:])


# =========================================================================== #
# (5) calendar_weekday: presence とバイトの曜日時計を暦へ合わせる
# =========================================================================== #
def _sim(tmp_path, name: str, **ov) -> Simulation:
    dot = ["run.seed=42", "run.n_agents=10", "run.n_steps=1", f"run.name={name}",
           "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_pool_weekday_follows_the_calendar_when_on(tmp_path):
    """start_date=2026-08-22(土)で day0 の presence 曜日 = 5(土)。OFF は従来どおり 0(月)。"""
    common = {"world.calendar.enabled": "true", "world.calendar.start_date": SAT}
    off = _sim(tmp_path, "cw_off", **common)
    on = _sim(tmp_path, "cw_on", **{**common, "world.calendar.calendar_weekday": "true"})
    assert off._pool_weekday(0) == 0 and off._pool_weekday(5) == 5     # day % 7(day0=月)
    assert on._pool_weekday(0) == 5                                    # 8/22 = 土
    assert on._pool_weekday(1) == 6                                    # 8/23 = 日
    assert on._pool_weekday(2) == 0                                    # 8/24 = 月


def test_pool_weekday_is_untouched_when_calendar_is_off(tmp_path):
    """暦そのものが OFF なら calendar_weekday を立てても day % 7 のまま(親トグルの契約)。"""
    sim = _sim(tmp_path, "cw_caloff",
               **{"world.calendar.enabled": "false",
                  "world.calendar.start_date": SAT,
                  "world.calendar.calendar_weekday": "true"})
    assert [sim._pool_weekday(d) for d in range(8)] == [d % 7 for d in range(8)]


def test_part_time_window_weekday_clock():
    """バイトの曜日も ON のときだけ暦へ合わせる(OFF は `(sim_min // 1440) % 7` のまま)。"""
    a = SimpleNamespace(sick=False,
                        part_time={"days": {5}, "start_min": 9 * 60, "end_min": 18 * 60})
    off = _cal(start=SAT, calendar_weekday=False)
    on = _cal(start=SAT, calendar_weekday=True)
    assert routine.in_part_time_window(a, 10 * 60) is False            # 旧時計: day0 = 月(0)
    assert routine.in_part_time_window(a, 10 * 60, off) is False
    assert routine.in_part_time_window(a, 10 * 60, on) is True         # 暦: day0 = 土(5)
    # 旧時計で土曜になる日(day5)は ON では木曜(8/27)なのでシフト外
    assert routine.in_part_time_window(a, 5 * 1440 + 10 * 60) is True
    assert routine.in_part_time_window(a, 5 * 1440 + 10 * 60, on) is False


def test_plan_anchor_uses_the_same_part_time_clock():
    """日計画のシフトアンカーも同じ曜日時計で置く(「計画には載るのに行かない」を防ぐ)。"""
    a = SimpleNamespace(work_start_min=-1, work_end_min=0, sick=False,
                        part_time={"days": {5}, "start_min": 9 * 60,
                                   "end_min": 18 * 60, "node": "n", "building": None,
                                   "floor": 0})
    on = _cal(start=SAT, calendar_weekday=True)
    for sim_min in (10 * 60, 5 * 1440 + 10 * 60):
        placed = bool(plan_schema.anchors_from_shift(a, sim_min, on))
        assert placed is routine.in_part_time_window(a, sim_min, on)
        placed_off = bool(plan_schema.anchors_from_shift(a, sim_min))
        assert placed_off is routine.in_part_time_window(a, sim_min)


# =========================================================================== #
# (6) 契約列挙(新 conf キー・既定値・レジストリ宣言)
# =========================================================================== #
def test_new_calendar_keys_exist_with_conservative_defaults():
    """基底 conf に 2 キーが在り、どちらも既定 false(= 現行と 1 バイト同一)。"""
    cfg = load_config()
    assert cfg.world.calendar.respect_work_days is False
    assert cfg.world.calendar.calendar_weekday is False
    built = cal_mod.build_cfg(None)
    assert built["respect_work_days"] is False and built["calendar_weekday"] is False
    # build_cfg は未知の raw でも欄を必ず作る(呼び出し側の .get 依存を減らす)
    assert set(cal_mod.build_cfg({})) == {"enabled", "start_date", "weekday_work",
                                          "holidays", "respect_work_days",
                                          "calendar_weekday"}


def test_new_calendar_keys_are_declared_in_the_registry():
    """レジストリ宣言済み(未宣言トグル検出の CI ゲートを通る)。"""
    ids = {f.id for f in R.FEATURES}
    for fid in ("world.calendar.respect_work_days", "world.calendar.calendar_weekday"):
        assert fid in ids, fid
    got = {f.id: f for f in R.FEATURES}
    for fid in ("world.calendar.respect_work_days", "world.calendar.calendar_weekday"):
        f = got[fid]
        assert f.repro_tier == "strict"          # 乱数も LLM も引かない決定論ゲート
        assert f.affects_k is False              # generate() の呼び出し点を足さない
        assert f.fingerprint_risk == "none"      # プロンプトへ 1 バイトも足さない
    assert R.undeclared_toggles(load_config()) == []


def test_finals_profile_turns_both_keys_on():
    """本選 conf は 2 キーとも ON(土曜始まりで在勤が全滅しないための必須設定)。"""
    fin = OmegaConf.load(_ROOT / "conf" / "finals_observe.yaml")
    assert fin.world.calendar.start_date == SAT
    assert fin.world.calendar.weekday_work is True
    assert fin.world.calendar.respect_work_days is True
    assert fin.world.calendar.calendar_weekday is True


def test_default_profiles_keep_both_keys_off():
    """既定プロファイルは触っていない(= 既定 OFF のまま golden 不変)。"""
    for name in ("daily.yaml", "production.yaml", "longrun30.yaml",
                 "smoke_tall.yaml", "smoke_wide.yaml", "observe.yaml"):
        raw = OmegaConf.load(_ROOT / "conf" / name)
        cal = (raw.get("world") or {}).get("calendar") or {}
        assert not cal.get("respect_work_days", False), name
        assert not cal.get("calendar_weekday", False), name


# =========================================================================== #
# (7) 統合ミニ: 土曜始まりの小規模ランで在勤が立ち上がる(OFF ではゼロ)
# =========================================================================== #
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プールは触らない。test_workplace_bind と同型)。"""
    out = tmp_path_factory.mktemp("pool_dow")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _saturday_sim(name, pool_dir, out_dir, *, respect: bool, n_steps: int = 1):
    """土曜始まり・暦 ON・束ね ON の小規模ラン(respect_work_days だけを振る)。"""
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", "pool.present_cap=400",
           "world.calendar.enabled=true", f"world.calendar.start_date={SAT}",
           "world.calendar.weekday_work=true",
           "world.calendar.calendar_weekday=true",       # 両腕とも同じ在場集合にする
           f"world.calendar.respect_work_days={'true' if respect else 'false'}",
           "work.bind_workplace.enabled=true",
           "work.bind_workplace.rebind_bound=true"]
    return Simulation(load_config(dot), out_dir=out_dir / name)


def _on_duty(sim, sim_min: int) -> list:
    cal = sim.calendarcfg
    return [a for a in sim.agents if routine.in_work_window(a, sim_min, cal)]


def test_saturday_staffing_is_zero_without_the_fix(small_pool, tmp_path):
    """OFF: 土曜は勤務窓に入る個体が 1 人も居ない(= 報告されたバグの再現)。"""
    sim = _saturday_sim("sat_off", small_pool, tmp_path, respect=False)
    assert sim._pool_weekday(0) == 5, "day0 が土曜になっていない(前提が崩れている)"
    assert sim.agents, "在場者が 0 人(前提が崩れている)"
    assert _on_duty(sim, 11 * 60) == []
    assert [a for a in sim.agents
            if scheduler._wage_worked_today(sim, a, 11 * 60)] == []


def test_saturday_staffing_recovers_with_the_fix(small_pool, tmp_path):
    """ON: 土曜営業を宣言した職場の担い手が実際に勤務窓へ入り、賃金ゲートも同時に開く。"""
    sim = _saturday_sim("sat_on", small_pool, tmp_path, respect=True)
    duty = _on_duty(sim, 11 * 60)
    assert duty, "土曜営業の宣言者が 1 人も勤務窓に入っていない"
    # 勤務ゲートと賃金ゲートが**同じ集合**を返す(会計の食い違いが構造的に起きない)
    wage = [a for a in sim.agents if scheduler._wage_worked_today(sim, a, 11 * 60)]
    assert {a.id for a in wage} >= {a.id for a in duty}
    for a in duty:
        assert cal_mod.days_match(getattr(a, "work_dow", ""), 5), \
            f"土曜に開いた個体の宣言が土曜を含まない: {getattr(a, 'work_dow', '')}"
    # 台帳の宣言が実際に個体へ届いている(mon-fri と土曜営業の両方が居る)
    declared = {getattr(a, "work_dow", "") for a in sim.agents if getattr(a, "work_dow", "")}
    assert declared, "work_dow が 1 個体にも付いていない"
    assert declared & {"mon-sat", "all"}, declared


def test_saturday_run_puts_people_at_work_end_to_end(small_pool, tmp_path):
    """数 step 回して**本業**の「出勤・通勤」ラベルが立つ(OFF では 1 人も立たない)。

    ★バイト(part_time)は暦の平日ゲートの対象外(既存仕様。`in_part_time_window` の
      docstring)なので、OFF でも土曜のシフトに入る個体は居る。ここが測りたいのは
      **本業の勤務**なので、part_time を持たない個体だけを数える。
    """
    def peak_main_job(sim, steps: int) -> int:
        peak = 0
        for step in range(steps):
            scheduler.run_step(sim, step)
            peak = max(peak, sum(1 for a in sim.agents
                                 if getattr(a, "activity", "") in ("working", "commuting")
                                 and not getattr(a, "part_time", None)))
        return peak

    steps = 66                       # 10 分 step で 11:00 過ぎまで(勤務窓の内側へ入る)
    off = _saturday_sim("e2e_off", small_pool, tmp_path, respect=False, n_steps=steps)
    on = _saturday_sim("e2e_on", small_pool, tmp_path, respect=True, n_steps=steps)
    assert peak_main_job(off, steps) == 0, "OFF なのに土曜に本業出勤が起きている"
    assert peak_main_job(on, steps) > 0, "ON でも土曜に本業出勤が 1 人も起きていない"


def test_saturday_run_is_deterministic(small_pool, tmp_path):
    """同 seed 2 ランで在勤集合が完全一致(乱数を 1 粒も足していない)。"""
    a = _saturday_sim("det_a", small_pool, tmp_path, respect=True)
    b = _saturday_sim("det_b", small_pool, tmp_path, respect=True)
    assert {x.id: getattr(x, "work_dow", "") for x in a.agents} \
        == {x.id: getattr(x, "work_dow", "") for x in b.agents}
    assert {x.id for x in _on_duty(a, 11 * 60)} == {x.id for x in _on_duty(b, 11 * 60)}
