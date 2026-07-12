"""組織(職場・学校の実態)統合(Wave2-1 2026-07-06)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): production/study が 0 件・org_* 属性なし・イベント列は純粋既定と完全一致。
- ON: 勤務完遂→production(産出は台帳の products_services 由来)/ 学生→study(教科は台帳由来)/
  所属1行がプロンプトに載る / 未配属(無職等)は無変化 / 同一設定2回で決定論。
- H 配線: tools.equip_all=true で発火プロンプトに「所持ツール」節(呼数は不変)。
- 検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import organizations
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name: str, n: int = 30, steps: int = 1,
         orgs: bool = True, **ov) -> Simulation:
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "agents.personas_file=data/personas_80.json",
           "observer.snapshot_every=1"]
    if orgs:
        dot.append("organizations.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    return Simulation(cfg, out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と organizations キー無しの純粋既定が L1 完全一致(既存挙動に無影響)。"""
    pure = _sim(tmp_path, "pure", steps=144, orgs=False)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, orgs=False,
               **{"organizations.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(organizations seam が no-op でない)"
    for k in ("production", "study"):
        assert not [e for e in pure.logger.events if e.kind == k], f"{k} が OFF で出ている"
    assert not any(getattr(a, "org_line", None) for a in pure.agents), \
        "OFF で org_line が付いている"


# --------------------------------------------------------------------- ON: 産出と学習
def test_on_production_and_study(tmp_path):
    """ON: 1日で勤務完遂→production、学生→study が台帳由来の中身で出る。"""
    sim = _sim(tmp_path, "on1", steps=144)
    sim.run()
    prod = [e for e in sim.logger.events if e.kind == "production"]
    study = [e for e in sim.logger.events if e.kind == "study"]
    assert prod, "production が 1 件も出ていない(勤務完遂と接続されていない)"
    assert study, "study が 1 件も出ていない(学生 8, 27 が名簿範囲内のはず)"
    book = sim.orgs
    for e in prod[:10]:
        org = book[e.payload["org"]]
        assert e.payload["output"] in org["products_services"], \
            "産出が台帳の products_services 由来でない"
        assert e.payload["kind"] in org["output_kinds"]
    for e in study[:10]:
        org = book[e.payload["org"]]
        pool = (org.get("subjects") or org.get("faculties")
                or org.get("lecture_fields") or [])
        assert e.payload["subject"] in pool, "教科が台帳由来でない"
        assert e.payload["role"] == "学生"
    # 組織会計は「集計だけ」: production の件数と ledger が整合
    total = sum(v["production_count"] for v in sim.org_ledger.values())
    assert total == len(prod)


def test_on_org_line_and_unassigned(tmp_path):
    """ON: 配属者に org_line(職場:/学校:)が付き、未配属(無職等)には何も付かない。"""
    sim = _sim(tmp_path, "on2", steps=1)
    sim.run()
    lines = [getattr(a, "org_line", None) for a in sim.agents]
    attached = [ln for ln in lines if ln]
    assert attached, "誰にも org_line が付いていない"
    assert any(ln.startswith("職場: ") for ln in attached)
    assert any(ln.startswith("学校: ") for ln in attached)   # 学生 8, 27 が範囲内
    # 配属ファイルの unassigned(無職)は org_* が無い
    asn, n_roster = organizations.assignments_for("data/personas_80.json")
    idx = {int(a["agent_index"]) for a in asn}
    for a in sim.agents:
        if (a.id % n_roster) not in idx:
            assert getattr(a, "org_line", None) is None, "未配属に org_line が付いた"


def test_prompt_injection_and_determinism(tmp_path):
    """ON: build_prompt に所属1行が載る。同一設定2回でイベント列完全一致(決定論)。"""
    from society.cognition import deliberate
    sim = _sim(tmp_path, "on3", steps=1)
    sim.run()
    agent = next(a for a in sim.agents if getattr(a, "org_line", None))
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3)
    assert agent.org_line in prompt, "所属1行がプロンプトに載っていない"
    # OFF 相当(org_line が無い agent)は従来と同一構成
    plain = next(a for a in sim.agents if getattr(a, "org_line", None) is None)
    p2 = deliberate.build_prompt(plain, place_name="路上", surprise="solo",
                                 nearby_names=[], sim_min=600, step=3)
    assert "職場: " not in p2 and "学校: " not in p2

    a1 = _sim(tmp_path, "det_a", steps=144)
    a1.run()
    a2 = _sim(tmp_path, "det_b", steps=144)
    a2.run()
    assert _l1(a1) == _l1(a2), "ON の決定論が崩れている"


# --------------------------------------------------------------------- H 配線
def test_equip_all_wiring(tmp_path):
    """tools.equip_all=true で発火プロンプトに「所持ツール」節が載る(呼数は変えない)。"""
    sim = _sim(tmp_path, "equip", steps=1, orgs=False,
               **{"tools.equip_all": "true"})
    sim.run()
    captured: list[str] = []
    real = sim.llm.generate

    def capture(prompt, **kw):
        captured.append(prompt)
        return real(prompt, **kw)

    sim.llm.generate = capture
    agent = sim.agents[0]
    agent.sleeping = False
    agent.loc = "street"
    scheduler._llm_speak(sim, agent, "solo", step=5, sim_min=600)
    assert captured, "発火プロンプトが生成されていない"
    assert "所持ツール" in captured[0], "equip_all=true なのに所持ツール節が無い"
    for tool in ("propose", "host_event", "post_flyer", "found_group", "open_venture"):
        assert tool in captured[0]

    sim2 = _sim(tmp_path, "equip_off", steps=1, orgs=False)
    sim2.run()
    captured2: list[str] = []
    real2 = sim2.llm.generate

    def capture2(prompt, **kw):
        captured2.append(prompt)
        return real2(prompt, **kw)

    sim2.llm.generate = capture2
    agent2 = sim2.agents[0]
    agent2.sleeping = False
    agent2.loc = "street"
    scheduler._llm_speak(sim2, agent2, "solo", step=5, sim_min=600)
    assert captured2 and "所持ツール" not in captured2[0], \
        "既定 OFF なのに所持ツール節が載っている"
