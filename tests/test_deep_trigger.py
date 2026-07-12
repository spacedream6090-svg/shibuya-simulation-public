"""出来事誘発の深い内省+無意識層(第12バッチ 2026-07-08)のテスト。

文献検証(docs/research/deep-reflection-triggers.md / self-concept-identity.md)を反映した設計:
- 日内衝撃ゲージ: |Δstate| をネガ非対称(neg_weight>pos_weight・ポジもゼロでない)で加重。
- 個人閾値(traits→reflect_trigger_params)超え → reflection_trigger(侵入的段階)→
  incubation_days(1晩以上)おいて夜の内省が深い内省に(cause="event")→ cooldown。
- 無意識層: 行動カウントのベースライン逸脱+感情価から「最近の自分」1行(決定論・非LLM)。
すべて既定 OFF=純粋既定と L1 完全一致。R1: 呼数は不変(深い内省=同じ1呼のプロンプト差)。
"""
from __future__ import annotations

import json

from society.cognition import reflection
from society.cognition.deliberate import build_prompt
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors import update as factor_update


def _sim(tmp_path, name, n=20, steps=144, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


_P = {"threshold": 0.30, "neg_weight": 2.0, "pos_weight": 1.0,
      "incubation_days": 1, "cooldown_days": 3}


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。reflection_trigger は出ない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"reflection.deep.enabled": "false",
                                   "reflection.implicit_self.enabled": "false",
                                   "reflection.self_model_days": "0"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(第12バッチ seam が no-op でない)"
    assert not _kind(pure, "reflection_trigger")
    assert all(a.reflect_p is None and a.behav_today is None for a in pure.agents)


# --------------------------------------------------------------------- ゲージと非対称
def test_gauge_asymmetry_and_trigger(tmp_path):
    """ネガはポジの neg_weight 倍で効き、閾値超えで incubation 後の深い内省を予約する。"""
    sim = _sim(tmp_path, "gauge", steps=1, **{"reflection.deep.enabled": "true"})
    sim.run()
    a, b = sim.agents[0], sim.agents[1]
    for x in (a, b):
        x.reflect_p = dict(_P)
        x.impact_today = x.impact_neg_today = x.impact_pos_today = 0.0
        x.deep_due_day, x.deep_cooldown_until_day = -1, -1
        x.states["grievance"], x.states["efficacy"] = 0.3, 0.3
    # ネガ(grievance +0.1)と ポジ(efficacy +0.1)で同じ |Δ| → ゲージは2倍差(非対称)
    factor_update.on_weather(a, 0.1, step=10, sim_min=100, logger=sim.logger)
    factor_update._bump(b, "efficacy", 0.1, "test", step=10, sim_min=100,
                        logger=sim.logger)
    assert abs(a.impact_today - 0.2) < 1e-9 and abs(b.impact_today - 0.1) < 1e-9, \
        f"非対称加重が崩れている: neg={a.impact_today} pos={b.impact_today}"
    assert a.deep_due_day == -1, "閾値未満で予約している"
    # 2発目で a が閾値(0.30)超え → day0+incubation1=day1 に予約+侵入的段階のイベント
    factor_update.on_weather(a, 0.1, step=11, sim_min=110, logger=sim.logger)
    assert a.deep_due_day == 1, f"予約日が不正: {a.deep_due_day}"
    trig = _kind(sim, "reflection_trigger")
    assert trig and trig[-1].agent_id == a.id and trig[-1].payload["due_day"] == 1
    # ポジも積めば閾値を超えられる(valence ゲートではない)
    factor_update._bump(b, "efficacy", 0.1, "test", step=12, sim_min=120,
                        logger=sim.logger)
    factor_update._bump(b, "efficacy", 0.1, "test", step=13, sim_min=130,
                        logger=sim.logger)
    assert b.deep_due_day == 1, "ポジティブな乖離で予約できていない"


# --------------------------------------------------------------------- 遅延実行と不応期
def test_incubation_deep_night_and_cooldown(tmp_path):
    """予約日以降の夜に深い内省(cause=event)→ 自己モデル更新 → cooldown 中は再予約なし。"""
    sim = _sim(tmp_path, "incub", steps=1, **{"reflection.deep.enabled": "true"})
    sim.run()
    a = sim.agents[0]
    a.reflect_p = dict(_P)
    a.deep_due_day = 1
    # day0 の夜(sim_min<1440)は予約日前=通常の内省のまま
    a.reflect_step = 200
    reflection.maybe_reflect(a, step=200, sim_min=1300, writeback="free", alpha=0.5,
                             llm=sim.llm, place_name="自宅",
                             rng=sim.hub.stream("writeback", a.id, 200),
                             logger=sim.logger, reflect_cfg=sim.reflectcfg)
    evs = [e for e in _kind(sim, "reflect") if e.agent_id == a.id]
    assert evs and not evs[-1].payload.get("deep"), "予約日前に深い内省になっている"
    assert a.deep_due_day == 1, "予約が勝手に消えている"
    # day1 の夜=深い内省(cause=event)。実行で予約解消+cooldown(day1+3=4)
    a.reflect_step = 350
    reflection.maybe_reflect(a, step=350, sim_min=1440 + 1300, writeback="free",
                             alpha=0.5, llm=sim.llm, place_name="自宅",
                             rng=sim.hub.stream("writeback", a.id, 350),
                             logger=sim.logger, reflect_cfg=sim.reflectcfg)
    ev = [e for e in _kind(sim, "reflect") if e.agent_id == a.id][-1]
    assert ev.payload.get("deep") and ev.payload.get("cause") == "event"
    assert a.self_model and a.self_model["day"] == 1, "深い内省で自己モデルが書かれていない"
    assert a.deep_due_day == -1 and a.deep_cooldown_until_day == 4
    # cooldown 中(day2)はゲージが超えても再予約しない
    a.impact_today = 0.0
    a.states["grievance"] = 0.2
    for i in range(4):
        factor_update.on_weather(a, 0.1, step=400 + i, sim_min=2 * 1440 + 100 + i,
                                 logger=sim.logger)
    assert a.deep_due_day == -1, "cooldown 中に再予約している(反芻の抑制が効いていない)"


# --------------------------------------------------------------------- 無意識層
def test_implicit_self_line_and_injection(tmp_path):
    """行動の逸脱+感情価から「最近の自分」を組み、プロンプトへ注入される。"""
    sim = _sim(tmp_path, "imp", steps=1,
               **{"reflection.implicit_self.enabled": "true"})
    sim.run()
    a = sim.agents[0]
    assert a.behav_today is not None, "implicit ON なのに行動カウンタが無い"
    a.behav_today = {"company": 10}
    a.behav_ema = {"company": 2.0}
    a.impact_neg_today, a.impact_pos_today = 1.0, 0.0
    reflection.update_implicit_self(a, ema=0.7)
    assert "人と過ごす時間がいつもより増えている" in a.implicit_self
    assert "気持ちはすこし重い" in a.implicit_self
    p = build_prompt(a, place_name="自宅", surprise=None, nearby_names=[], step=10)
    assert "最近の自分(なんとなく感じていること):" in p and a.implicit_self in p
    # OFF の agent は行なし
    a.implicit_self = ""
    p2 = build_prompt(a, place_name="自宅", surprise=None, nearby_names=[], step=11)
    assert "最近の自分" not in p2


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: deep+implicit ON/OFF で generate 呼数が完全一致(288 step)。"""
    def run(name, on):
        ov = {"reflection.deep.enabled": str(on).lower(),
              "reflection.implicit_self.enabled": str(on).lower()}
        sim = _sim(tmp_path, name, steps=288, **ov)
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    on = run("tr_on", True)
    off = run("tr_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert [e.step for e in _kind(on, "reflect")] == \
        [e.step for e in _kind(off, "reflect")], "内省の回数・時刻が変わっている"


# --------------------------------------------------------------------- 決定論
def test_on_deterministic(tmp_path):
    """deep+implicit ON 同士 2 回で L1 完全一致(mock・決定論・288 step)。"""
    ov = {"reflection.deep.enabled": "true",
          "reflection.implicit_self.enabled": "true",
          "reflection.deep.threshold": "0.2"}      # 起こりやすくして経路も踏む
    a = _sim(tmp_path, "det_a", steps=288, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", steps=288, **ov)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
