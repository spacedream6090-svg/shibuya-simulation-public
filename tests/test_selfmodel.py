"""反射=自己モデル(第11バッチ 2026-07-08)のテスト。

Generative Agents の reflection に倣う: reflection.self_model_days=N で N日ごとの夜、
同じ1回の内省呼び出しを「深い内省」に格上げし、蓄積記憶から自己像(self)と大事な
関係(ties)を生成 → agent.self_model を更新 → 以後の全プロンプトへ
「自分の理解(内省より)」1行を注入=自己認識の再帰的更新。

検証の柱:
- OFF(既定 0)= 純粋既定と L1 完全一致。
- ON: 深い内省の夜に self_model が書かれ、プロンプトに注入される。
- k ゲート: writeback=sham では深い内省が走っても self_model は書かれない(belief と同一ゲート)。
- R1 呼数不変: 応答固定 backend で ON/OFF の generate 呼数が完全一致
  (深い内省は同じ1呼のプロンプト差のみ=呼び出しを1本も足さない)。
"""
from __future__ import annotations

import json

from society.cognition.deliberate import build_prompt
from society.config import load_config
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=20, steps=288, **ov):
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


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF(self_model_days=0)と純粋既定が L1 完全一致。deep キーも出ない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"reflection.self_model_days": "0"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(自己モデル seam が no-op でない)"
    assert all("deep" not in e.payload for e in _kind(pure, "reflect")), \
        "OFF で reflect payload に deep が出ている"
    assert all(a.self_model is None for a in pure.agents)


# --------------------------------------------------------------------- ON: 更新と注入
def test_deep_reflection_updates_and_injects(tmp_path):
    """day1 の夜=深い内省: self_model が書かれ、以後のプロンプトに1行注入される。"""
    sim = _sim(tmp_path, "deep", **{"reflection.self_model_days": "1"})
    sim.run()
    deep_evs = [e for e in _kind(sim, "reflect") if e.payload.get("deep")]
    assert deep_evs, "深い内省が一度も起きていない(288step に day1 の就寝が含まれるはず)"
    updated = [a for a in sim.agents if a.self_model]
    assert updated, "誰の self_model も書かれていない"
    a = updated[0]
    assert a.self_model["self"] and a.self_model["day"] >= 1
    assert any(e.payload.get("self_model_updated") for e in deep_evs)
    # プロンプト注入: persona 直後に「自分の理解(内省より)」行
    p = build_prompt(a, place_name="自宅", surprise=None, nearby_names=[], step=290)
    assert "自分の理解(内省より):" in p and a.self_model["self"] in p
    # day0(sim_min<1440)の通常内省は deep ではない(N日ごと・day>=1 のみ。
    # シミュは 7:00 開始なので step でなく sim_min で日を判定する)
    day0 = [e for e in _kind(sim, "reflect") if e.sim_min < 1440]
    assert day0 and all(not e.payload.get("deep") for e in day0)


# --------------------------------------------------------------------- k ゲート
def test_sham_writes_nothing(tmp_path):
    """writeback=sham: 深い内省は走る(計算量同一)が self_model は書かれない(k ゲート共有)。"""
    sim = _sim(tmp_path, "sham", **{"reflection.self_model_days": "1",
                                    "k.writeback": "sham"})
    sim.run()
    deep_evs = [e for e in _kind(sim, "reflect") if e.payload.get("deep")]
    assert deep_evs, "sham でも深い内省自体は走るはず(計算量同一=R1)"
    assert all(not e.payload.get("self_model_updated") for e in deep_evs)
    assert all(a.self_model is None for a in sim.agents), \
        "sham で self_model が書かれている(k ゲート破り)"


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


def _run_fixed(tmp_path, name, *, days):
    sim = _sim(tmp_path, name, **{"reflection.self_model_days": str(days)})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: 自己モデル ON/OFF で generate 呼数が完全一致。

    深い内省は「同じ1回の内省呼び出しのプロンプト差」だけで、呼び出しを1本も
    足さない(R1)。応答固定なら self も書かれない(fixed 応答に self キーが無い)
    ため、行動系列も同一のまま=呼数の完全一致を厳密に検証できる。"""
    on = _run_fixed(tmp_path, "sm_on", days=1)
    off = _run_fixed(tmp_path, "sm_off", days=0)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
    assert [e.step for e in _kind(on, "reflect")] == \
        [e.step for e in _kind(off, "reflect")], "内省の回数・時刻が変わっている"


# --------------------------------------------------------------------- 決定論
def test_on_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(mock・決定論)。"""
    a = _sim(tmp_path, "det_a", **{"reflection.self_model_days": "1"})
    a.run()
    b = _sim(tmp_path, "det_b", **{"reflection.self_model_days": "1"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
