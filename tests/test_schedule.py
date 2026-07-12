"""長期予定・スケジュール帳(第7バッチ 2026-07-07)のテスト。

鉄則の検証:
- OFF(既定): 会話からの抽出・注入なし・appointment 0 件=既存挙動と L1 バイト一致。
- extract 単体: 相対日/絶対日/時刻/行動/場所の各パターン。未来日時なし→空。
- 両者共有: speak は聞き手全員+話者に同一予定(DM は to にも)。
- プロンプト1行注入: schedule_line が build_prompt に載る/予定が無ければ載らない。
- 決定論: ON 同士2回で L1 完全一致(会話由来の予定込み)。
- R1 呼数不変: 応答固定バックエンド(FixedLLM)で ON/OFF の generate 呼数が一致。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import schedule
from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world import calendar


def _sim(tmp_path, name: str, *, n: int = 20, steps: int = 1, sched: bool = True,
         **ov) -> Simulation:
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1"]
    if sched:
        dot.append("schedule.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------- extract 単体
def test_extract_relative_time_place():
    """「明日15時に渋谷で会おう」→ day=+1・when=15:00・what=会う・place=渋谷。"""
    out = schedule.extract("明日15時に渋谷で会おう", base_day=0, base_min=8 * 60)
    assert out == [{"day": 1, "when": "15:00", "what": "会う", "place": "渋谷"}]


def test_extract_various_patterns():
    cal = calendar.build_cfg({"enabled": True, "start_date": "2026-04-01"})
    # 明後日 + 時間帯語 + 場所 + 食事
    a = schedule.extract("明後日の夕方、センター街でご飯食べよう", base_day=0,
                         base_min=8 * 60, cal=cal)[0]
    assert a["day"] == 2 and a["when"] == "夕方"
    assert a["what"] == "食事" and a["place"] == "センター街"
    # ◯日後(時刻なし)
    b = schedule.extract("3日後にまた会おう", base_day=0, base_min=8 * 60, cal=cal)[0]
    assert b["day"] == 3 and b["when"] == "" and b["what"] == "会う"
    # 時刻の「半」+ 午後
    c = schedule.extract("明日午後2時半に図書館で勉強しよう", base_day=0,
                         base_min=8 * 60, cal=cal)[0]
    assert c["day"] == 1 and c["when"] == "14:30" and c["what"] == "勉強会"
    assert c["place"] == "図書館"
    # 絶対日(◯月◯日)= start_date 2026-04-01 起点で 4/8 は day_index=7
    d = schedule.extract("4月8日の10時に会いましょう", base_day=0,
                         base_min=8 * 60, cal=cal)[0]
    assert d["day"] == 7 and d["when"] == "10:00"
    # 週末 → 次の土曜(曜日=5)に解決される
    e = schedule.extract("週末に公園で遊ぼう", base_day=0, base_min=8 * 60, cal=cal)[0]
    assert calendar.weekday_of(cal, e["day"] * 1440) == 5
    assert e["what"] == "遊び" and e["place"] == "公園"
    # 今夜(時間帯語)= 当日・夜
    f = schedule.extract("今夜、渋谷で飲もう", base_day=0, base_min=5 * 60, cal=cal)[0]
    assert f["day"] == 0 and f["when"] == "夜" and f["what"] == "食事"


def test_extract_no_future_returns_empty():
    """未来の日時が無ければ空(今日で時間指定なし・過去時刻も記入しない)。"""
    assert schedule.extract("こんにちは、元気ですか?", base_day=0, base_min=600) == []
    assert schedule.extract("今日はいい天気だね", base_day=0, base_min=600) == []
    # 今日かつ既に過ぎた時刻は記入しない(現在 12:00、7時 は過去)
    assert schedule.extract("今日7時に集合ね", base_day=0, base_min=12 * 60) == []
    # 期間外(horizon 超え)は記入しない
    assert schedule.extract("30日後に会おう", base_day=0, base_min=600,
                            horizon_days=14) == []
    # 空テキスト
    assert schedule.extract("", base_day=0, base_min=600) == []


# --------------------------------------------------------------- 両者共有(記入)
def test_speak_shares_with_speaker_and_hearers(tmp_path):
    """speak: 話者と聞き手全員の帳簿に同一予定(with は相手 id)+ appointment ログ。"""
    sim = _sim(tmp_path, "share", n=6)
    speaker, h1, h2 = sim.agents[0], sim.agents[1], sim.agents[2]
    scheduler._record_appointments(sim, speaker, "明日15時に渋谷で会おう",
                                   [h1.id, h2.id], step=5, sim_min=8 * 60)
    for a in (speaker, h1, h2):
        assert len(a.schedule) == 1, "予定が記入されていない"
        appt = a.schedule[0]
        assert appt["day"] == 1 and appt["when"] == "15:00"
        assert appt["what"] == "会う" and appt["place"] == "渋谷"
    assert speaker.schedule[0]["with"] == sorted([h1.id, h2.id])
    assert h1.schedule[0]["with"] == sorted([speaker.id, h2.id])   # 自分以外
    appts = [e for e in sim.logger.events if e.kind == "appointment"]
    assert len(appts) == 1
    assert appts[0].agent_id == speaker.id
    assert appts[0].payload == {"day": 1, "when": "15:00", "what": "会う",
                                "place": "渋谷", "with": sorted([h1.id, h2.id])}


def test_dm_shares_with_recipient(tmp_path):
    """DM: to にも同一予定が入る。"""
    sim = _sim(tmp_path, "dm_share", n=4)
    speaker, to = sim.agents[0], sim.agents[3]
    scheduler._record_appointments(sim, speaker, "明後日の夜に恵比寿で食事しよう",
                                   [to.id], step=2, sim_min=9 * 60)
    for a in (speaker, to):
        assert len(a.schedule) == 1 and a.schedule[0]["day"] == 2
        assert a.schedule[0]["what"] == "食事" and a.schedule[0]["place"] == "恵比寿"


def test_record_dedup_and_gc(tmp_path):
    """同一予定は重複追加しない / 過去の予定は当日経過で GC。"""
    sim = _sim(tmp_path, "dedup", n=3)
    a = sim.agents[0]
    assert scheduler._schedule_on(sim)
    scheduler._record_appointments(sim, a, "明日15時に渋谷で会おう", [], 1, 8 * 60)
    scheduler._record_appointments(sim, a, "明日15時に渋谷で会おう", [], 2, 8 * 60)
    assert len(a.schedule) == 1, "重複が追加された"
    # GC: 当日=2 に進むと day<2 の予定(day=1)は失効
    schedule.gc(a, 2)
    assert a.schedule == []


def test_off_no_appointments_and_no_injection(tmp_path):
    """OFF(既定): appointment 0 件・帳簿は空・schedule_line は None。"""
    off = _sim(tmp_path, "off", n=10, steps=144, sched=False)
    off.run()
    assert not any(e.kind == "appointment" for e in off.logger.events)
    assert all(not a.schedule for a in off.agents)
    assert scheduler._schedule_line(off, off.agents[0], 600) is None


def test_off_equals_pure_default_l1(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(schedule seam が no-op)。"""
    pure = _sim(tmp_path, "pure", n=12, steps=144, sched=False)
    pure.run()
    off = _sim(tmp_path, "expl_off", n=12, steps=144, sched=False,
               **{"schedule.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off)


# --------------------------------------------------------------- プロンプト注入
def test_prompt_injection_and_absence(tmp_path):
    """近い予定がある agent には schedule_line が載り、無い agent には載らない。"""
    sim = _sim(tmp_path, "inject", n=4)
    ag = sim.agents[0]
    schedule.record(ag, {"day": 1, "when": "15:00", "what": "会う",
                         "place": "渋谷", "with": []}, step=0)
    line = scheduler._schedule_line(sim, ag, 0)          # today=0
    assert line is not None and line.startswith("予定:")
    prompt = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                     nearby_names=[], sim_min=0, step=0,
                                     schedule_line=line)
    assert line in prompt
    # 予定の無い agent には注入されない
    plain = sim.agents[1]
    assert scheduler._schedule_line(sim, plain, 0) is None
    p2 = deliberate.build_prompt(plain, place_name="渋谷", surprise="solo",
                                 nearby_names=[], sim_min=0, step=0)
    assert "予定:" not in p2


def test_schedule_line_none_is_byte_identical(tmp_path):
    """schedule_line=None は従来の build_prompt と完全一致(純追記の後方互換)。"""
    ag = _sim(tmp_path, "byte", n=3).agents[0]
    kw = dict(place_name="渋谷", surprise="social", nearby_names=["A"],
              sim_min=0, step=0)
    assert deliberate.build_prompt(ag, **kw) \
        == deliberate.build_prompt(ag, schedule_line=None, **kw)


# --------------------------------------------------------------- 決定論 / R1 呼数
class _FixedLLM:
    """応答をプロンプトに依存させない backend(未来日時入りの発話を固定で返す)。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


_DATED = json.dumps({"action": "speak", "text": "明日15時に渋谷で会おう"},
                    ensure_ascii=False)


def _run_fixed(tmp_path, name, *, sched, n=20, steps=24):
    sim = _sim(tmp_path, name, n=n, steps=steps, sched=sched)
    sim.llm = _FixedLLM(_DATED)
    sim.run()
    return sim


def test_on_deterministic_with_appointments(tmp_path):
    """ON 同士2回で L1 完全一致(会話由来の予定込み)+ appointment が発生する。"""
    a = _run_fixed(tmp_path, "det_a", sched=True)
    b = _run_fixed(tmp_path, "det_b", sched=True)
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    assert any(e.kind == "appointment" for e in a.logger.events), \
        "appointment が1件も出ていない(抽出→記入が動いていない)"


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend なら ON/OFF で generate 呼数が完全一致(R1: 呼数不変)。"""
    on = _run_fixed(tmp_path, "cc_on", sched=True)
    off = _run_fixed(tmp_path, "cc_off", sched=False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert any(e.kind == "appointment" for e in on.logger.events)
    assert not any(e.kind == "appointment" for e in off.logger.events)
