"""第7バッチ(ユーザー要望 2026-07-07: 日付・天気・通勤/通学の物理 POI 束縛)のテスト。

鉄則の検証:
- OFF(既定): date_line/weather を注入せず・weather イベント0件・work_node 上書きなし=
  既存挙動と L1 バイト一致(決定論も維持)。
- calendar ON: date_line がプロンプトに載る / 曜日・休日導出が正しい。
- weather ON: 天気イベントが日数分出る / weather_line が sim に載る / ON 同士2回で L1 一致 /
  天気→grievance は factors 経由(cause="weather" の state_update)。
- commute_to_poi ON: 配属者の work_node == org.workplace_poi.node(上書きが起きる)。
- weekday_work ON: 日曜相当の日に本業勤務(production/study)が出ない=平日ゲート機能。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import weather as weather_mod
from society.cognition import deliberate
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors import update as factor_update
from society.world import calendar


def _sim(tmp_path, name: str, n: int = 30, steps: int = 1, orgs: bool = False,
         **ov) -> Simulation:
    dot = [f"run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1",
           "agents.personas_file=data/personas_80.json",
           "government.enabled=false"]
    if orgs:
        dot.append("organizations.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と全キー未指定の純粋既定が L1 完全一致(seam が no-op)。天気イベントも0件。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"world.calendar.enabled": "false", "weather.enabled": "false",
                  "world.calendar.weekday_work": "false",
                  "organizations.commute_to_poi": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(日付・天気 seam が no-op でない)"
    assert not any(e.kind == "weather" for e in pure.logger.events), \
        "OFF なのに weather イベントが出ている"
    assert not any(e.kind == "state_update" and e.payload.get("cause") == "weather"
                   for e in pure.logger.events)
    assert pure.today_date_line is None and pure.today_weather_line is None


def test_off_deterministic(tmp_path):
    """OFF 同士2回でイベント列が一致(決定論を壊していない)。"""
    a = _sim(tmp_path, "offdet_a", steps=144)
    a.run()
    b = _sim(tmp_path, "offdet_b", steps=144)
    b.run()
    assert _l1(a) == _l1(b)


# --------------------------------------------------------------------- calendar
def test_calendar_derivations():
    """曜日・休日・is_workday・date_line の導出(純関数・乱数不使用)。"""
    cfg = calendar.build_cfg({"enabled": True, "start_date": "2026-04-05"})  # 日曜
    assert calendar.weekday_of(cfg, 0) == 6           # 日
    assert calendar.is_holiday(cfg, 0) and not calendar.is_workday(cfg, 0)
    assert calendar.date_line(cfg, 0) == "今日は2026年4月5日(日)です。"
    # 翌日(day_index=1)= 月曜 = 平日
    assert calendar.weekday_of(cfg, 1440) == 0
    assert calendar.is_workday(cfg, 1440)
    assert calendar.date_line(cfg, 1440) == "今日は2026年4月6日(月)です。"
    # 追加休日(平日でも holidays に含めれば休日)
    cfg2 = calendar.build_cfg({"enabled": True, "start_date": "2026-04-06",
                               "holidays": ["2026-04-06"]})
    assert calendar.is_holiday(cfg2, 0)
    # 無効なら date_line は None
    assert calendar.date_line(calendar.build_cfg({"enabled": False}), 0) is None


def test_calendar_prompt_injection_optional(tmp_path):
    """build_prompt: date_line/weather_line を渡した時だけ1行足す(None で不変)。"""
    ag = _sim(tmp_path, "cal_prompt").agents[0]
    on = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                 nearby_names=[], sim_min=0, step=0,
                                 date_line="今日は2026年4月8日(火)です。",
                                 weather_line="今日の天気: 雨、最高14℃。")
    assert "今日は2026年4月8日(火)です。" in on
    assert "今日の天気: 雨、最高14℃。" in on
    off = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                  nearby_names=[], sim_min=0, step=0)
    assert "今日は" not in off and "今日の天気" not in off


def test_calendar_run_sets_today_lines(tmp_path):
    """calendar ON の run 後、sim に当日の date_line が確定して載る。"""
    sim = _sim(tmp_path, "cal_run", steps=3,
               **{"world.calendar.enabled": "true",
                  "world.calendar.start_date": "2026-04-05"})
    sim.run()
    assert sim.today_date_line == "今日は2026年4月5日(日)です。"


# --------------------------------------------------------------------- weather
def test_weather_helpers_and_factor_seam(tmp_path):
    """weather の純関数(discomfort_delta/weather_line)+ factors 経由の grievance 加算。"""
    wcfg = weather_mod.build_cfg({"enabled": True, "rain_grievance": 0.02})
    assert weather_mod.discomfort_delta({"cond": "雨", "temp_hi": 14}, wcfg) == 0.02
    assert weather_mod.discomfort_delta({"cond": "雪", "temp_hi": 1}, wcfg) == 0.02
    assert weather_mod.discomfort_delta({"cond": "晴", "temp_hi": 25}, wcfg) == 0.0
    assert weather_mod.weather_line({"cond": "雨", "temp_hi": 14}) \
        == "今日の天気: 雨、最高14℃。"
    assert weather_mod.weather_line(None) is None
    # factors 経由(R9): magnitude を渡すと grievance が上がり cause="weather" が記録される
    sim = _sim(tmp_path, "wea_seam", steps=1, **{"weather.enabled": "true"})
    sim.run()
    a = sim.agents[0]
    g0 = a.states["grievance"]
    d = factor_update.on_weather(a, 0.05, step=0, sim_min=0, logger=sim.logger)
    assert d > 0 and a.states["grievance"] > g0
    assert any(e.kind == "state_update" and e.payload.get("cause") == "weather"
               for e in sim.logger.events)


def test_weather_events_per_day_and_determinism(tmp_path):
    """weather ON: 天気イベントが日数分・agent_id=-1 で出る / ON 同士2回で L1 一致。"""
    def run(name):
        sim = _sim(tmp_path, name, steps=144,
                   **{"weather.enabled": "true", "weather.rain_grievance": "0.01",
                      "world.calendar.enabled": "true"})
        sim.run()
        return sim

    a = run("wea_a")
    b = run("wea_b")
    assert _l1(a) == _l1(b), "weather ON 同士でイベント列が一致しない(非決定論)"
    n_days = len({a.clock.sim_min(s) // 1440 for s in range(144)})
    wev = [e for e in a.logger.events if e.kind == "weather"]
    assert len(wev) == n_days, f"weather イベント数 {len(wev)} != 日数 {n_days}"
    for e in wev:
        assert e.agent_id == -1
        assert e.payload["cond"] in ("晴", "曇", "雨", "雪")
        assert isinstance(e.payload["temp_hi"], int)
        assert "date" in e.payload and "weekday" in e.payload   # calendar 併記
    assert a.today_weather_line and a.today_weather_line.startswith("今日の天気:")


# --------------------------------------------------------------------- commute
def test_commute_to_poi_binds_work_node(tmp_path):
    """commute_to_poi ON: 配属者の work_node == org.workplace_poi.node(上書きが起きる)。"""
    on = _sim(tmp_path, "com_on", steps=1, orgs=True,
              **{"organizations.commute_to_poi": "true"})
    on.run()
    off = _sim(tmp_path, "com_off", steps=1, orgs=True,
               **{"organizations.commute_to_poi": "false"})
    off.run()
    book = on.orgs
    matched = 0
    for a in on.agents:
        oid = getattr(a, "org_id", None)
        if not oid:
            continue
        node = (book[oid].get("workplace_poi") or {}).get("node")
        if node and node in on.city.graph:
            assert a.work_node == node, "配属者の work_node が workplace_poi.node に一致しない"
            matched += 1
    assert matched > 0, "上書き対象の配属者が1人もいない"
    # commute OFF では上書きされない → 少なくとも1人 work_node が ON と異なる
    off_nodes = {a.id: a.work_node for a in off.agents}
    assert any(a.work_node != off_nodes.get(a.id) for a in on.agents), \
        "commute ON/OFF で work_node に差が無い(上書きが機能していない)"


def test_commute_off_keeps_org_line_byte_identical(tmp_path):
    """organizations ON でも commute_to_poi=false なら既存 org 挙動と L1 一致(不変の保証)。"""
    a = _sim(tmp_path, "com_bytea", steps=144, orgs=True)
    a.run()
    b = _sim(tmp_path, "com_byteb", steps=144, orgs=True,
             **{"organizations.commute_to_poi": "false"})
    b.run()
    assert _l1(a) == _l1(b)


# --------------------------------------------------------------------- weekday_work
def test_weekday_work_gates_full_time(tmp_path):
    """weekday_work ON: 日曜始まりの日は本業勤務(production/study)が出ず、月曜始まりは出る。"""
    common = {"world.calendar.enabled": "true",
              "world.calendar.weekday_work": "true"}
    sun = _sim(tmp_path, "sun", steps=144, orgs=True,
               **{**common, "world.calendar.start_date": "2026-04-05"})  # 日曜
    sun.run()
    mon = _sim(tmp_path, "mon", steps=144, orgs=True,
               **{**common, "world.calendar.start_date": "2026-04-06"})  # 月曜
    mon.run()
    sun_work = [e for e in sun.logger.events if e.kind in ("production", "study")]
    mon_work = [e for e in mon.logger.events if e.kind in ("production", "study")]
    assert not sun_work, "日曜(休日)なのに本業勤務(production/study)が出ている"
    assert mon_work, "月曜(平日)なのに本業勤務(production/study)が出ていない"
