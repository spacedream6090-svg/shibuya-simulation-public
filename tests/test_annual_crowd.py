"""文化カレンダー・群集(現実ギャップ Wave G4 2026-07-07)= 年中行事/ハロウィン型群集 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): annual_event/crowd_surge が 0 件・event_line 注入なし・イベント列は純粋既定と
  L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- 年中行事 ON: ハロウィン当日に event_line がプロンプトに載り annual_event が出る。非該当日/暦 OFF は載らない。
- 群集 ON: 群集日(ハロウィン)に自由行動の行き先がスクランブル側(集会ノード)へ偏り、集中が閾値超で
  crowd_surge が出る。非群集日(正月)は today_crowd_event なし=通常の EPR(乱数も引かない)。
- 決定論: ON 同士2回で L1 完全一致。R1 呼数不変(FixedLLM で ON==OFF)=annual 機構は generate を1回も足さない。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_HALLOWEEN = {"world.calendar.enabled": "true",
              "world.calendar.start_date": "2026-10-31",
              "annual_events.enabled": "true"}
_NEWYEAR = {"world.calendar.enabled": "true",
            "world.calendar.start_date": "2026-01-01",
            "annual_events.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。G4 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"annual_events.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G4 seam が no-op でない)"
    for k in ("annual_event", "crowd_surge"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    assert pure.today_event_line is None and pure.today_crowd_event is None
    assert pure.crowd_node is None, "OFF で集会ノードが解決されている"


# --------------------------------------------------------------------- 年中行事
def test_annual_event_line_and_log(tmp_path):
    """ハロウィン当日: event_line がプロンプトに載り annual_event が出る。非該当日/暦 OFF は出ない。"""
    from society.cognition import deliberate
    sim = _sim(tmp_path, "hw", steps=1, **_HALLOWEEN)
    sim.run()
    assert sim.today_event_line == "今日はハロウィンです。"
    ae = _kind(sim, "annual_event")
    assert ae and ae[0].agent_id == -1 and ae[0].payload["name"] == "ハロウィン" \
        and ae[0].payload["date"] == "2026-10-31", "annual_event が正しく記録されていない"

    agent = sim.agents[0]
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3,
                                     event_line=sim.today_event_line)
    assert "今日はハロウィンです。" in prompt, "event_line がプロンプトに載っていない"
    plain = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                    nearby_names=[], sim_min=600, step=3)
    assert "ハロウィン" not in plain, "event_line=None なのに行事語が載っている"

    # 非該当日(通常日): event_line なし・annual_event なし
    day = _sim(tmp_path, "plainday", steps=1,
               **{"world.calendar.enabled": "true",
                  "world.calendar.start_date": "2026-06-15",
                  "annual_events.enabled": "true"})
    day.run()
    assert day.today_event_line is None
    assert not _kind(day, "annual_event"), "非該当日に annual_event が出ている"

    # 暦 OFF なら annual ON でも年中行事は出ない(年中行事は暦の上に乗る)
    nocal = _sim(tmp_path, "nocal", steps=1, **{"annual_events.enabled": "true"})
    nocal.run()
    assert nocal.today_event_line is None
    assert not _kind(nocal, "annual_event"), "暦 OFF で annual_event が出ている"


# --------------------------------------------------------------------- 群集: 行き先バイアス
def test_crowd_bias_targets_gathering_node(tmp_path):
    """群集日(ハロウィン)は自由行動の行き先を集会ノードへ寄せる。非群集日(正月)は寄せない。"""
    from society.cognition import routine
    sim = _sim(tmp_path, "bias", steps=1,
               **{**_HALLOWEEN, "annual_events.crowd_bias": "1.0"})
    sim.run()
    assert sim.today_crowd_event == "ハロウィン"
    assert sim.crowd_node is not None, "群集日なのに集会ノードが解決されていない"
    away = next(a for a in sim.agents if a.node != sim.crowd_node)
    assert routine._crowd_dest(away, sim, step=7) == sim.crowd_node, \
        "群集日に行き先が集会ノードへ寄っていない(bias=1.0)"
    at = next((a for a in sim.agents if a.node == sim.crowd_node), None)
    if at is not None:
        assert routine._crowd_dest(at, sim, step=7) is None, "既に集会ノードに居るのに寄せている"

    # 非群集日(正月): today_crowd_event なし → バイアスなし(乱数も引かない)
    ny = _sim(tmp_path, "ny", steps=1, **{**_NEWYEAR, "annual_events.crowd_bias": "1.0"})
    ny.run()
    assert ny.today_crowd_event is None, "正月(非群集)なのに群集イベントが立っている"
    away2 = next(a for a in ny.agents if a.node != ny.crowd_node)
    assert routine._crowd_dest(away2, ny, step=7) is None, "非群集日にバイアスがかかっている"


# --------------------------------------------------------------------- 群集: crowd_surge
def test_crowd_surge_logged_when_concentrated(tmp_path):
    """群集日: 集会ノード周辺の集中が閾値超で crowd_surge を記録(非LLM・決定論)。"""
    sim = _sim(tmp_path, "surge", n=20, steps=1,
               **{**_HALLOWEEN, "annual_events.crowd_threshold": "4"})
    sim.run()
    node = sim.crowd_node
    cx, cy = sim.city.node_xy(node)
    for a in sim.agents[:5]:                            # 集会ノードへ5人集中(閾値4超)
        a.loc, a.sleeping, a.building = "street", False, None
        a.node, a.x, a.y = node, cx, cy
    sim._crowd_surge_day = -1                           # 記録済みフラグをリセット
    n0 = len(_kind(sim, "crowd_surge"))
    sim_min = sim.clock.sim_min(50)
    scheduler._phase_crowd(sim, step=50, sim_min=sim_min)
    surges = _kind(sim, "crowd_surge")
    assert len(surges) > n0, "集中が閾値超なのに crowd_surge が出ていない"
    p = surges[-1].payload
    assert p["node"] == node and p["level"] >= 5 and p["event"] == "ハロウィン", \
        f"crowd_surge の payload が不正: {p}"
    assert surges[-1].agent_id == -1

    # 1日1回まで: 同日に再度呼んでも増えない
    scheduler._phase_crowd(sim, step=51, sim_min=sim.clock.sim_min(51))
    assert len(_kind(sim, "crowd_surge")) == len(surges), "crowd_surge が1日に複数回出ている"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """年中行事+群集 ON(ハロウィン)同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_HALLOWEEN)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_HALLOWEEN)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, *, annual):
    # 非群集の行事日(正月)で比較する: annual ON は event_line 注入 + annual_event 記録のみで
    # 行き先(移動)を変えない=annual 機構が generate を1回も足さないことを厳密検証する。
    ov = {"world.calendar.enabled": "true", "world.calendar.start_date": "2026-01-01"}
    if annual:
        ov["annual_events.enabled"] = "true"
    sim = _sim(tmp_path, name, n=30, steps=144, **ov)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: 年中行事 ON/OFF(非群集の行事日=正月)で generate 呼数が完全一致。

    群集の行き先バイアスは自由行動の移動のみを変える物理機構(weekly_event/Lynch と同型)であり、
    annual/群集の実装は generate() を1回も追加せず・混雑を発火系に接続しない(R1)。"""
    on = _run_fixed(tmp_path, "cc_on", annual=True)
    off = _run_fixed(tmp_path, "cc_off", annual=False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert _kind(on, "annual_event"), "正月の annual_event が出ていない(ON 側の機構が動いていない)"


def _run_crowd_k(tmp_path, name, *, writeback):
    """群集日(ハロウィン)を compute_matched(k 掃引で実際に使う対照)下で回し呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_HALLOWEEN, "controls.mode": "compute_matched",
                  "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_crowd_call_count_k_invariant(tmp_path):
    """群集日は対面 co-location が増え ON!=OFF になる(実在する群集の必然)。だが R1 の本旨=
    「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。群集の移動は
    暦日付+物理位置+config のみ参照し k・内面状態を一切読まないため、k=free と k=off で呼数が完全一致する。"""
    free = _run_crowd_k(tmp_path, "ck_free", writeback="free")
    off = _run_crowd_k(tmp_path, "ck_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"群集日の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "crowd_surge"), "群集日なのに crowd_surge が出ていない(機構が不発)"
