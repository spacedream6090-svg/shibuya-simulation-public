"""開放行動+価値の4軸(第17バッチ 2026-07-10)のテスト。

設計(docs/research/desire-value-theory.md・ユーザー決定 2026-07-10):
- freedom.open_actions=true のときだけヘッダに "do" 1行(中立提示・例なし)。
- 裁定は決定論(語彙辞書+中立自己申告のブレンド・価格は中央値固定・乱数なし)。
- 充足 sat(4軸)→ sat_mods(reason→感度倍率)のみ外へ(感度=可・目標注入=不可)。
- 既定 OFF=ヘッダ・挙動バイト一致。R1: 呼数不変(メニュー1行の内容差のみ)。
"""
from __future__ import annotations

import json

from society import values
from society.cognition.deliberate import build_prompt, parse_action
from society.config import load_config
from society.engine.simulation import Simulation


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


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。sat 属性なし・free_action 0件。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"freedom.open_actions": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(第17バッチ seam が no-op でない)"
    assert not _kind(pure, "free_action")
    assert all(getattr(a, "sat", None) is None and getattr(a, "sat_mods", None) is None
               for a in pure.agents)
    p = build_prompt(pure.agents[0], place_name="自宅", surprise=None,
                     nearby_names=[], step=1)
    assert '"action": "do"' not in p, "OFF なのにヘッダに do 行がある"


# --------------------------------------------------------------------- 単体
def test_values_unit():
    """辞書分類・自己申告の正準化・ブレンド・充足の力学(純関数)。"""
    cat, tags = values.classify("公園で絵を描く")
    assert cat == "創作" and tags["epistemic"] > 0
    cat2, _ = values.classify("駅前で募金する")
    assert cat2 == "向社会"
    cat3, tags3 = values.classify("意味不明な何か")
    assert cat3 == "その他" and abs(sum(tags3.values()) - 1.0) < 1e-6
    assert values.parse_report(["感情", "epistemic", "未知語"]) == ["emotion", "epistemic"]
    blended = values.blend_tags({"utility": 1.0}, ["social"])
    assert abs(blended["utility"] - 0.5) < 1e-6 and abs(blended["social"] - 0.5) < 1e-6

    class _A:                                  # sat の力学(飽和・回帰)
        pass
    a = _A()
    a.sat = {t: 0.5 for t in values.TAGS}
    cfg = values.build_cfg({"open_actions": True})
    values.satisfy(a, {"emotion": 1.0}, cfg)
    assert a.sat["emotion"] > 0.5, "充足が上がらない"
    assert a.sat_mods["silence"] < 1.0 or a.sat_mods["sns"] < 1.0, \
        "満たされた価値に紐づく感度が下がらない"
    before = a.sat["emotion"]
    values.decay_daily(a, cfg)
    assert 0.5 < a.sat["emotion"] < before, "日次回帰が効かない"


def test_parse_action_do():
    """"do" の解釈: what 必須・minutes クリップ・value 通過。壊れは None。"""
    act = parse_action('{"action": "do", "what": "本を読む", "minutes": 999, '
                       '"value": ["認識"]}')
    assert act == {"type": "free_action", "what": "本を読む", "where": None,
                   "minutes": 240, "value_report": ["認識"]}
    assert parse_action('{"action": "do"}') is None


# --------------------------------------------------------------------- ON 挙動
def test_on_free_actions_and_effects(tmp_path):
    """ON: mock が do を出し、free_action イベント+価値タグ+充足が観測される。"""
    sim = _sim(tmp_path, "on", steps=288, **{"freedom.open_actions": "true"})
    sim.run()
    evs = _kind(sim, "free_action")
    assert evs, "ON なのに free_action が 1 件も出ない(mock 分岐が届いていない)"
    cats = {e.payload["category"] for e in evs}
    assert cats <= {"創作", "学習", "鑑賞", "散策", "家事", "向社会", "その他"}, \
        f"想定外カテゴリ: {cats}"
    assert all("tags" in e.payload and "match" in e.payload for e in evs)
    moved = [a for a in sim.agents
             if a.sat and any(abs(v - 0.5) > 1e-9 for v in a.sat.values())]
    assert moved, "誰の充足も動いていない"
    # 有料カテゴリが出ていれば支出も記録される(向社会500円/鑑賞1500円)
    paid = [e for e in evs if e.payload["cost"] > 0]
    spends = {json.dumps(e.payload, ensure_ascii=False)
              for e in sim.logger.events if e.kind == "spend"}
    if paid:
        assert any("free_" in s for s in spends), "有料の開放行動に spend が伴わない"


def test_on_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(mock・決定論)。"""
    a = _sim(tmp_path, "det_a", steps=288, **{"freedom.open_actions": "true"})
    a.run()
    b = _sim(tmp_path, "det_b", steps=288, **{"freedom.open_actions": "true"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


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
    """応答固定 backend: freedom ON/OFF で generate 呼数が完全一致(288 step)。"""
    def run(name, on):
        sim = _sim(tmp_path, name, steps=288,
                   **{"freedom.open_actions": str(on).lower()})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    on = run("r1_on", True)
    off = run("r1_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
