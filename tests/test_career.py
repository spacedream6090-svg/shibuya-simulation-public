"""キャリア転換(現実ギャップ Wave G5 2026-07-07)= 失業/求職/転職/起業転換 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): unemployment/job_change/venture_fulltime が 0 件・新 stream "career" も引かない・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
  organizations ON の世界でも career OFF は完全 no-op。
- 失業 ON: 被雇用者が確率で失業し unemployment(lost)+job_change が出て、org/勤務窓/賃金がクリアされ、
  生活不安(grievance+)が factors 経由で入る(単体・決定論)。
- 求職 ON: 失業者が新 org へ再就職(unemployment(hired)+job_change)= 勤務窓・収入が復帰。
- 転職 ON: 被雇用者が別 org へ移り job_change(from!=to)が出る。
- 起業転換 ON: 好調 venture の所有者が本業を辞め venture_fulltime になる(org/本業賃金クリア・勤務地=屋台)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 失業は物理位置(勤務窓)を変え対面 co-location が変化しうる(群集 G4 と同型)。よって呼数不変は
  compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import organizations
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_CAREER = {"career.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, orgs=False, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if orgs:
        dot += ["agents.personas_file=data/personas_80.json",
                "organizations.enabled=true"]
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
    """明示 OFF と純粋既定が L1 完全一致(144 step)。G5 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"career.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(G5 seam が no-op でない)"
    for k in ("unemployment", "job_change", "venture_fulltime"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


def test_off_noop_under_organizations(tmp_path):
    """organizations ON の世界でも career OFF は完全 no-op(org 有効ラン基準と L1 一致)。"""
    base = _sim(tmp_path, "orgbase", steps=144, orgs=True)
    base.run()
    off = _sim(tmp_path, "orgoff", steps=144, orgs=True,
               **{"career.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "organizations ON で career seam が no-op でない"


# --------------------------------------------------------------------- 失業
def test_layoff_clears_job_and_raises_grievance(tmp_path):
    """失業 ON: 被雇用者が失業し unemployment(lost)+job_change、org/勤務窓/賃金クリア、grievance+。"""
    sim = _sim(tmp_path, "layoff", steps=1, orgs=True,
               **{**_CAREER, "career.layoff_prob": "1.0",
                  "career.switch_prob": "0.0", "career.rehire_prob": "0.0",
                  "career.unemployment_grievance": "0.05"})
    sim.run()
    lost = [e for e in _kind(sim, "unemployment") if e.payload["state"] == "lost"]
    jc = [e for e in _kind(sim, "job_change") if e.payload["cause"] == "layoff"]
    assert lost, "失業 ON なのに unemployment(lost)が出ていない(被雇用者が居ない?)"
    assert len(jc) == len(lost), "失業 1 件につき job_change 1 件が出ていない"
    for e in lost:
        a = sim.agent_by_id[e.agent_id]
        assert a.org_id is None and a.wage == 0.0 and a.work_start_min == -1, \
            "失業者の org/賃金/勤務窓がクリアされていない"
        assert organizations.is_laid_off(a), "失業フラグが立っていない(求職の対象にならない)"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "job_loss"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "失業→grievance(job_loss)が factors 経由で入っていない"


# --------------------------------------------------------------------- 転職
def test_switch_changes_org(tmp_path):
    """転職 ON: 被雇用者が別 org へ移り job_change(from!=to)が出て org_id が変わる。"""
    sim = _sim(tmp_path, "switch", steps=1, orgs=True,
               **{**_CAREER, "career.layoff_prob": "0.0",
                  "career.switch_prob": "1.0", "career.rehire_prob": "0.0"})
    sim.run()
    jc = [e for e in _kind(sim, "job_change") if e.payload["cause"] == "switch"]
    assert jc, "転職 ON なのに job_change(switch)が出ていない"
    for e in jc:
        assert e.payload["from_org"] and e.payload["to_org"] \
            and e.payload["from_org"] != e.payload["to_org"], \
            f"転職の from/to が不正: {e.payload}"
        a = sim.agent_by_id[e.agent_id]
        assert a.org_id == e.payload["to_org"], "転職後の org_id が to_org と一致しない"
    assert not _kind(sim, "unemployment"), "転職のみ ON なのに失業が出ている"


# --------------------------------------------------------------------- 求職
def test_rehire_restores_job(tmp_path):
    """求職 ON: 失業者(前日 layoff)が新 org へ再就職し unemployment(hired)+job_change=収入復帰。"""
    sim = _sim(tmp_path, "rehire", steps=200, orgs=True,
               **{**_CAREER, "career.layoff_prob": "1.0",
                  "career.switch_prob": "0.0", "career.rehire_prob": "1.0"})
    sim.run()
    hired = [e for e in _kind(sim, "unemployment") if e.payload["state"] == "hired"]
    jc = [e for e in _kind(sim, "job_change") if e.payload["cause"] == "rehire"]
    assert hired, "求職 ON(前日失業)なのに unemployment(hired)が出ていない"
    assert len(jc) == len(hired), "再就職 1 件につき job_change(rehire)1 件が出ていない"
    for e in hired:
        a = sim.agent_by_id[e.agent_id]
        assert a.org_id is not None and not organizations.is_laid_off(a), \
            "再就職者が employed に戻っていない"
    assert any(sim.agent_by_id[e.agent_id].wage > 0.0 for e in hired), \
        "再就職で収入(wage)が誰も復帰していない"


# --------------------------------------------------------------------- 起業転換
def test_venture_fulltime_conversion(tmp_path):
    """起業転換 ON: 好調 venture の所有者が本業を辞め venture_fulltime になる(1店1回)。"""
    sim = _sim(tmp_path, "vf", steps=1, orgs=True,
               **{**_CAREER, "career.layoff_prob": "0.0", "career.switch_prob": "0.0",
                  "career.venture_fulltime_sales": "5000",
                  "career.venture_fulltime_days": "0"})
    sim.run()
    owner = next(a for a in sim.agents
                 if organizations.is_employee(a) and not a.visitor)
    node = owner.node
    sim.tools.ventures[owner.id] = {
        "owner": owner.id, "node": node, "name": "屋台テスト", "offer": "x",
        "price": 800, "opened_step": 0, "last_sale_step": 0,
        "sales_total": 6000.0, "fulltime": False}
    n0 = len(_kind(sim, "venture_fulltime"))
    sim.tools._ventures_fulltime(sim, step=10, sim_min=sim.clock.sim_min(10))
    vf = _kind(sim, "venture_fulltime")
    assert len(vf) > n0 and vf[-1].payload["name"] == "屋台テスト", \
        "好調 venture が本業化(venture_fulltime)していない"
    assert owner.org_id is None and owner.wage == 0.0, "本業 org/賃金がクリアされていない"
    assert owner.work_node == node and owner.work_start_min >= 0, "勤務地が屋台へ移っていない"
    # 1店1回: 同じ venture は再変換しない
    sim.tools._ventures_fulltime(sim, step=11, sim_min=sim.clock.sim_min(11))
    assert len(_kind(sim, "venture_fulltime")) == len(vf), "同じ venture が複数回変換された"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """キャリア転換 ON(既定確率)同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, orgs=True, **_CAREER)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, orgs=True, **_CAREER)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 k 不変性
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    """キャリア転換 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144, orgs=True,
               **{**_CAREER, "career.layoff_prob": "0.05",
                  "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_career_call_count_k_invariant(tmp_path):
    """失業は対面 co-location を変え ON!=OFF になりうる(実在する転換の必然)。だが R1 の本旨=
    「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。career の失業/転職は
    暦日付+物理位置+config+新 stream のみ参照し k・内面状態を一切読まないため、k=free と k=off で
    generate 呼数が完全一致する(annual 群集 G4 と同型)。"""
    free = _run_k(tmp_path, "ck_free", writeback="free")
    off = _run_k(tmp_path, "ck_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"career の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "unemployment"), "失業日なのに unemployment が出ていない(機構が不発)"
