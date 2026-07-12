"""娯楽メディア(TV・動画・ゲーム。バッチD)のテスト。

検証:
- ON → 夜間・在宅で media_use が発生 / セッション中は外出しない(media_stay)/
  セッション終了時に cause="media" の回復 state_update / 年齢プロファイル差が出る。
- OFF(既定)→ media_use 0件・既存挙動は決定論で不変。
- 決定論(同設定2回=同一のイベント列)。
- mock 24 step 完走。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from society import media
from society.cognition import routine
from society.config import load_config
from society.engine.simulation import Simulation


# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, steps=1, n=10, extra=None):
    # government.enabled=false: 並行バッチ(行政・税)の効果から media を隔離する
    # 明示 OFF(既定 OFF と同義)。media の検証を行政の税・給付から独立させる。
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "observer.snapshot_every=1", "government.enabled=false"]
    dot += list(extra or [])
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _events(sim):
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


def _hour_step(sim, hour):
    """sim_min の hour(0-23)がその値になる最初の step を返す。"""
    for s in range(0, 300):
        if (sim.clock.sim_min(s) % 1440) // 60 == hour:
            return s
    raise AssertionError(f"hour={hour} の step が見つからない")


def _home_at(sim, a, step):
    """agent を在宅・自宅前の路上・覚醒状態にし、勤務窓を外す。"""
    a.visitor = False
    a.loc = "street"
    a.building = None
    a.route = []
    a.sleeping = False
    a.home_node = a.home_node or sim.city.gateways[0]
    a.node = a.home_node
    a.x, a.y = sim.city.node_xy(a.node)
    a.work_start_min = -1
    a.part_time = None
    a.bedtime_min = 1430
    a.now_step = step
    a._media_end = -1
    a._media_day = -1
    a._media_count = 0
    a.states["grievance"] = 0.5


# --------------------------------------------------------------------------- #
def test_off_default_no_media_events_and_deterministic(tmp_path):
    """既定(media 未設定=OFF): media_use は0件、かつ2回のランでイベント列が一致。"""
    a = _sim(tmp_path, "off_a", steps=60)
    a.run()
    b = _sim(tmp_path, "off_b", steps=60)
    b.run()
    ea, eb = _events(a), _events(b)
    assert ea == eb, "OFF 同士でイベント列が一致しない(非決定論)"
    assert not any(k == "media_use" for (_s, _i, k, _p) in ea), \
        "media OFF なのに media_use が出ている"


def test_on_night_home_produces_media_use_and_repair(tmp_path):
    """ON: 夜間・在宅で media_use が発生し、終了時に cause='media' の回復が入る。"""
    sim = _sim(tmp_path, "night", extra=["media.enabled=true",
                                         "media.start_prob=1.0",
                                         "factors.media_grievance=-0.02"])
    step = _hour_step(sim, 22)                       # 22:00 = 夜間帯
    sim_min = sim.clock.sim_min(step)

    started = []
    for a in sim.agents:
        _home_at(sim, a, step)
        act = routine.decide(a, step, sim, "home",
                             sim.hub.stream("decide", a.id, step),
                             has_company=False)
        if act["type"] == "media_stay":
            started.append(a)
    assert started, "夜間・在宅で誰も視聴を始めなかった"

    med = [e for e in sim.logger.events if e.kind == "media_use"]
    assert len(med) == len(started), "media_use の件数が開始者数と一致しない"
    ev = med[0]
    assert ev.payload["medium"] in ("tv", "video", "game")
    assert ev.payload["steps"] >= 1
    assert ev.payload["at"] in ("夜", "深夜", "夕方", "朝", "昼", "午後")

    # 開始者を1人選び、セッション終了まで進めて回復 state_update を確認する。
    a = started[0]
    g0 = a.states["grievance"]
    end = a._media_end
    assert end > step
    for s in range(step + 1, end + 1):
        routine.decide(a, s, sim, "home",
                       sim.hub.stream("decide", a.id, s), has_company=False)
    repairs = [e for e in sim.logger.events
               if e.kind == "state_update" and e.payload.get("cause") == "media"
               and e.agent_id == a.id]
    assert repairs, "セッション終了時に cause='media' の回復が出ていない"
    assert a.states["grievance"] < g0, "grievance が下がっていない(気分修復なし)"


def test_time_displacement_media_stay_not_stay(tmp_path):
    """セッション中は type='media_stay'(!='stay')= scheduler がスマホ閲覧を挟まず外出もしない。"""
    sim = _sim(tmp_path, "disp", extra=["media.enabled=true",
                                        "media.start_prob=1.0"])
    step = _hour_step(sim, 22)
    hits = 0
    for a in sim.agents:
        _home_at(sim, a, step)
        act = routine.decide(a, step, sim, "home",
                             sim.hub.stream("decide", a.id, step),
                             has_company=False)
        if act["type"] == "media_stay":
            hits += 1
            # 継続 step も media_stay を返す(在宅占有)= 外出しない。
            nxt = routine.decide(a, step + 1, sim, "home",
                                 sim.hub.stream("decide", a.id, step + 1),
                                 has_company=False)
            assert nxt["type"] == "media_stay"
            assert nxt["type"] != "stay"
    assert hits, "セッションが1件も始まらなかった"


def test_work_hours_take_priority(tmp_path):
    """勤務時間帯は視聴を始めない(仕事優先)。"""
    sim = _sim(tmp_path, "work", extra=["media.enabled=true",
                                        "media.start_prob=1.0"])
    step = _hour_step(sim, 9)                        # 9:00 = 朝の帯かつ勤務窓に入れる
    a = sim.agents[0]
    _home_at(sim, a, step)
    a.work_start_min = 8 * 60                        # 勤務中に設定
    a.work_end_min = 17 * 60
    a.work_node = a.node
    act = routine.decide(a, step, sim, "home",
                         sim.hub.stream("decide", a.id, step), has_company=False)
    assert act["type"] != "media_stay", "勤務時間帯なのに視聴を始めた"
    assert not any(e.kind == "media_use" and e.agent_id == a.id
                   for e in sim.logger.events)


def test_age_profile_differs():
    """年齢帯で媒体構成が変わる(若年=動画・ゲーム寄り、高齢=TV寄り)。"""
    mcfg = media.media_cfg(load_config(["media.enabled=true"]))
    youth = SimpleNamespace(age=20, occupation="大学生")
    elder = SimpleNamespace(age=72, occupation="無職")
    py = media.profile_for(youth, mcfg)
    pe = media.profile_for(elder, mcfg)
    assert py["weights"] != pe["weights"]

    def frac(prof, target, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        picks = [media.pick_medium(prof, rng) for _ in range(n)]
        return picks.count(target) / n

    assert frac(pe, "tv") > frac(py, "tv"), "高齢の TV 比率が若年以下"
    assert frac(py, "game") > frac(pe, "game"), "若年のゲーム比率が高齢以下"
    assert frac(py, "video") > frac(pe, "video"), "若年の動画比率が高齢以下"
    # 開始確率も年齢帯・職業の自由時間で差が出る。
    assert media.start_prob(py, mcfg) > media.start_prob(
        media.profile_for(SimpleNamespace(age=40, occupation="会社員"), mcfg), mcfg)


def test_prompt_context_flag_controls_title_and_memory(tmp_path):
    """prompt_context ON のときだけ payload に title が入り、視聴の記憶が残る(発火文脈へ)。"""
    def _first_session(pc: bool):
        sim = _sim(tmp_path, f"pc_{pc}", extra=[
            "media.enabled=true", "media.start_prob=1.0",
            f"media.prompt_context={'true' if pc else 'false'}"])
        step = _hour_step(sim, 22)
        for a in sim.agents:
            _home_at(sim, a, step)
            act = routine.decide(a, step, sim, "home",
                                 sim.hub.stream("decide", a.id, step),
                                 has_company=False)
            if act["type"] == "media_stay":
                ev = next(e for e in sim.logger.events
                          if e.kind == "media_use" and e.agent_id == a.id)
                return a, ev
        raise AssertionError("セッションが始まらなかった")

    a_on, ev_on = _first_session(True)
    assert "title" in ev_on.payload
    title = ev_on.payload["title"]
    assert any(title in t for t in a_on.mem.recent(6)), \
        "prompt_context ON なのに視聴タイトルが記憶に残っていない"

    a_off, ev_off = _first_session(False)
    assert "title" not in ev_off.payload
    assert not any("楽しんだ" in t for t in a_off.mem.recent(6)), \
        "prompt_context OFF なのに視聴タイトルが記憶に混入している"


def test_full_run_on_natural_events_and_deterministic(tmp_path):
    """通しラン(168 step)で media_use が自然発生し、ON 同士は決定論で一致する。"""
    def run(name):
        sim = _sim(tmp_path, name, steps=168, n=8,
                   extra=["media.enabled=true"])
        sim.run()
        return sim

    a = run("full_a")
    b = run("full_b")
    assert _events(a) == _events(b), "ON 同士でイベント列が一致しない(非決定論)"
    n_media = sum(1 for e in a.logger.events if e.kind == "media_use")
    assert n_media > 0, "通しランで media_use が自然発生していない"

    # 同じ長さの OFF ランには media_use が一切出ない。
    off = _sim(tmp_path, "full_off", steps=168, n=8)
    off.run()
    assert not any(e.kind == "media_use" for e in off.logger.events)


def test_mock_24step_completes(tmp_path):
    """mock で 24 step 完走(ON)。要約が返り、LLM を直接増やさない(media は非LLM)。"""
    sim = _sim(tmp_path, "smoke24", steps=24, n=10, extra=["media.enabled=true"])
    summary = sim.run()
    assert summary["n_steps"] == 24
    assert summary["n_agents"] == 10
    # media 由来の LLM 呼び出しは存在しない(全て非LLM)= media_use は llm_call を伴わない。
    for e in sim.logger.events:
        if e.kind == "media_use":
            assert e.llm_call_id is None
