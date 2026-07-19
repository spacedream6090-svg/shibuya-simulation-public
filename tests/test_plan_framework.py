"""日課計画フレームワーク(P2 S1)のテスト。

設計: docs/research/daily-plan-framework.md(F1 スキーマ / F2 コンパイラ / F3 失敗の階段)。
docs/plans/p2-interstitial-design.md §1 S1。

検収:
  OFF: ゴールデンバイト一致(既定=framework OFF)+ 新イベント(day_plan_compiled)ゼロ。
  ON(mock): (a) スキーマでパース→コンパイル (b) 同 seed 2回一致 (c) LLM 呼数 OFF 同一
            (d) 勤務アンカーが shift どおり配置 (e) パース失敗→再試行→前日流用/初日既定
            (f) day_plan_compiled が出る。
既存 test_planning.py の k 不変・byte 一致は不変(OFF 経路は現行と完全同一)。
"""
from __future__ import annotations

import json

from society.cognition import planning
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 8, steps: int = 1, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _run_stream(tmp_path, name: str, ov: list[str]):
    """168 step = 翌日 11:00 まで(全員が一度は起床する=day_plan が現れる長さ)。"""
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=168", f"run.name={name}",
           *ov]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


def _wake_and_plan(sim, step: int = 5):
    """全員を step で起床させ、直後 step で朝の計画を生成させる(短尺で計画を確実に出す)。"""
    for a in sim.agents:
        a.sleeping, a.sleep_until = True, step
        a.loc, a.building = "street", None
        a.node = a.home_node
        a.x, a.y = sim.city.node_xy(a.node)
        a.plan_day, a.plan_step, a.day_plan = -1, -1, []
    scheduler._phase_wake_and_returns(sim, step, sim.clock.sim_min(step))
    scheduler._phase_planning(sim, step + 1, sim.clock.sim_min(step + 1))


def _events(sim):
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# OFF: ゴールデンバイト一致
# --------------------------------------------------------------------------- #
def test_off_is_golden_byte_identical(tmp_path):
    """framework OFF(既定)は明示 OFF と byte 一致・決定論。新イベントは一切出さない。"""
    base = _run_stream(tmp_path, "off_base", [])                     # 既定=framework OFF
    exp = _run_stream(tmp_path, "off_explicit",
                      ["planning.framework.enabled=false"])
    assert base == exp, "framework OFF が既定と byte 一致しない(後方互換が壊れている)"
    kinds = {k for (_s, _i, k, _p) in base}
    assert "day_plan_compiled" not in kinds, "OFF なのに新イベントが出ている"
    assert "day_plan" in kinds, "OFF で従来の day_plan が消えている"


# --------------------------------------------------------------------------- #
# ON: パース→コンパイル / 決定論 / 新イベント(a, b, f)
# --------------------------------------------------------------------------- #
def test_on_parses_compiles_and_emits_event(tmp_path):
    """ON: 全員がスキーマでパース→コンパイルされ、day_plan_compiled が1人1件出る。"""
    sim = _sim(tmp_path, "on_a", **{"planning.framework.enabled": "true"})
    _wake_and_plan(sim)
    comp = [e for e in sim.logger.events if e.kind == "day_plan_compiled"]
    plans = [e for e in sim.logger.events if e.kind == "day_plan"]
    assert len(comp) == len(sim.agents), "day_plan_compiled が全員に出ていない"
    assert len(plans) == len(sim.agents), "day_plan が全員に出ていない"
    # コンパイル要約の形(cat 内訳・アンカー数・失敗経路)が載っている
    for e in comp:
        p = e.payload
        assert set(p["cats"]) == {"mandatory", "maintenance", "discretionary"}
        assert p["path"] in ("llm", "retry", "prev_day", "default")
        assert p["anchors"] >= 0 and p["n"] >= p["anchors"]
    # 少なくとも一部の agent で降格 day_plan が非空(コンパイラが実際に充填している)
    assert any(a.day_plan for a in sim.agents)
    assert all(hasattr(a, "day_schedule") for a in sim.agents)


def test_on_same_seed_is_deterministic(tmp_path):
    """ON: 同 seed の2ランでイベント列が完全一致(決定論)。"""
    s1 = _sim(tmp_path, "on_det1", **{"planning.framework.enabled": "true"})
    s2 = _sim(tmp_path, "on_det2", **{"planning.framework.enabled": "true"})
    _wake_and_plan(s1)
    _wake_and_plan(s2)
    assert _events(s1) == _events(s2), "同 seed ON でイベント列が一致しない"


# --------------------------------------------------------------------------- #
# ON: LLM 呼数が OFF と同一(c)
# --------------------------------------------------------------------------- #
def test_on_llm_call_count_equals_off(tmp_path):
    """正常経路(mock)では ON でも再試行が起きず、計画 LLM 呼数は OFF と同一(R1)。"""
    def plan_calls(name, ov):
        sim = _sim(tmp_path, name, **ov)
        _wake_and_plan(sim)
        return sum(1 for c in sim.logger.llm_calls if c["purpose"] == "plan"), \
            sum(1 for c in sim.logger.llm_calls if c["purpose"] == "plan_retry")

    off_plan, off_retry = plan_calls("cc_off", {"planning.framework.enabled": "false"})
    on_plan, on_retry = plan_calls("cc_on", {"planning.framework.enabled": "true"})
    assert off_retry == 0 and on_retry == 0, "正常経路で再試行が発生している"
    assert on_plan == off_plan > 0, \
        f"計画 LLM 呼数が OFF と異なる: on={on_plan} off={off_plan}"


# --------------------------------------------------------------------------- #
# ON: 勤務アンカーが shift どおり配置(d)
# --------------------------------------------------------------------------- #
def test_anchor_placed_from_shift(tmp_path):
    """勤務窓(work_start/end_min)が mandatory アンカーとして shift どおり先置きされる。"""
    sim = _sim(tmp_path, "anchor", **{"planning.framework.enabled": "true"})
    a = sim.agents[0]
    a.work_start_min, a.work_end_min = 9 * 60, 17 * 60     # 09:00-17:00(朝バンド始業)
    a.part_time = None
    a.day_plan = []
    step = 6
    planning.make_plan(sim, a, step, sim.clock.sim_min(step), "職場付近")
    anchors = [e for e in a.day_schedule if e["anchor"]]
    assert anchors, "勤務窓があるのにアンカーが置かれていない"
    anc = anchors[0]
    assert anc["cat"] == "mandatory" and anc["flex"] == "fixed"
    assert anc["start_min"] == 9 * 60, "アンカーが shift の始業時刻に置かれていない"
    assert anc["when"] == "朝"
    # 降格 day_plan にアンカー(勤務)は含めない(勤務窓は routine が独立に処理する)
    assert all(it["what"] != "work" for it in a.day_plan)
    comp = [e for e in sim.logger.events
            if e.kind == "day_plan_compiled" and e.agent_id == a.id][-1]
    assert comp.payload["anchors"] >= 1


# --------------------------------------------------------------------------- #
# ON: 失敗の階段(e)= パース失敗→再試行→前日流用 / 初日既定骨格
# --------------------------------------------------------------------------- #
def _break_llm(sim, monkeypatch):
    """LLM 応答を常に修復不能な非 JSON にする(第1回+再試行の両方が壊れる)。"""
    monkeypatch.setattr(sim.llm, "generate",
                        lambda *a, **k: ("壊れた応答(JSONではない", "brk", False))


def test_failure_ladder_prev_day_reuse(tmp_path, monkeypatch):
    """パース失敗→再試行も失敗→前日計画を流用する(path=prev_day)。"""
    sim = _sim(tmp_path, "fail_prev", **{"planning.framework.enabled": "true"})
    a = sim.agents[0]
    a.work_start_min, a.part_time = -1, None              # アンカーなし=流用がそのまま出る
    a._plan_prev_day = [{"intent": "前日の買い物", "cat": "maintenance",
                         "what": "shop", "place": "", "when": "昼", "flex": "flexible"}]
    _break_llm(sim, monkeypatch)
    step = 6
    planning.make_plan(sim, a, step, sim.clock.sim_min(step), "どこか")
    comp = [e for e in sim.logger.events
            if e.kind == "day_plan_compiled" and e.agent_id == a.id][-1]
    assert comp.payload["path"] == "prev_day", "前日流用の経路に入っていない"
    assert [it["what"] for it in a.day_plan] == ["shop"], "前日計画が復元されていない"
    # 再試行が1回起きている(purpose=plan_retry が1件)
    assert sum(1 for c in sim.logger.llm_calls
               if c["purpose"] == "plan_retry" and c["agent_id"] == a.id) == 1


def test_failure_ladder_default_skeleton(tmp_path, monkeypatch):
    """初日(前日計画なし)は職業別デフォルト骨格へ落ちる(path=default)。"""
    sim = _sim(tmp_path, "fail_default", **{"planning.framework.enabled": "true"})
    a = sim.agents[0]
    a.work_start_min, a.part_time = -1, None             # 非勤務=日中外出の骨格
    if hasattr(a, "_plan_prev_day"):
        del a._plan_prev_day
    _break_llm(sim, monkeypatch)
    step = 6
    planning.make_plan(sim, a, step, sim.clock.sim_min(step), "どこか")
    comp = [e for e in sim.logger.events
            if e.kind == "day_plan_compiled" and e.agent_id == a.id][-1]
    assert comp.payload["path"] == "default", "初日のデフォルト骨格に落ちていない"
    assert a.day_plan, "デフォルト骨格から day_plan が構築されていない"
