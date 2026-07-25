"""内部可動性 第60バッチ(b): 転居 / bond→同棲・unbond→転出 / career選択由来化(求職)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): relocate/move_in/move_out/job_search が 0 件・新 stream "housing" も引かない・
  イベント列は純粋既定と L1 完全一致(ゴールデン golden_baseline_l1.json を守る)。organizations/
  household ON の世界でも 3 機構 OFF は完全 no-op。
- 転居 ON: 職場変更後の通勤逼迫で世帯全員が一緒に relocate(reason=job)。行き先は決定論(同 seed 同結果)。
- 同棲 ON: bond→N日+closeness で move_in(相手宅へ転居・世帯併合)、unbond で move_out(新居・世帯分離)。
- 求職 ON: job_search tool 発火→ mobility.match_job→ 既存 switch_org で org_id が変わる。空き無し=outcome none。
- 決定論: 3 機構 ON 同士 2 回で L1 完全一致。resume==straight は境界跨ぎでも L1/L2/L3 一致。
- R1: 転居は物理位置(home)を変え対面 co-location が変化しうるが、呼数不変は compute_matched 下の
  k 不変性(by_choice ON/OFF で generate 呼数一致=求職 tool は既存 tool 選択枠内)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society import household as household_mod
from society import mobility, organizations
from society.cognition.deliberate import parse_action
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

# 3 機構 ON(職場変更で work_node が動くよう commute_to_poi=true。career で職場変更を作る)。
_ON = {
    "organizations.enabled": "true", "organizations.commute_to_poi": "true",
    "agents.personas_file": "data/personas_80.json",
    "relations.enabled": "true", "household.enabled": "true",
    "career.enabled": "true", "career.switch_prob": "0.2", "career.layoff_prob": "0.0",
    "housing.relocation.enabled": "true",
    "housing.relocation.commute_threshold_m": "100",
    "housing.relocation.job_prob": "1.0",
    "household.cohabit.enabled": "true", "career.by_choice.enabled": "true",
}


def _sim(tmp_path, name, n=40, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "model.backend=mock", "observer.snapshot_every=1"]
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
    """明示 OFF と純粋既定が L1 完全一致(144 step)。第60バッチの新イベントは 1 件も出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"housing.relocation.enabled": "false",
                  "household.cohabit.enabled": "false",
                  "career.by_choice.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(mobility seam が no-op でない)"
    for k in ("relocate", "move_in", "move_out", "job_search"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


def test_off_noop_under_orgs_household(tmp_path):
    """organizations+household ON の世界でも 3 機構 OFF は完全 no-op(基準ランと L1 一致)。"""
    base = _sim(tmp_path, "base", steps=144, **{
        "organizations.enabled": "true", "agents.personas_file": "data/personas_80.json",
        "relations.enabled": "true", "household.enabled": "true"})
    base.run()
    off = _sim(tmp_path, "orgoff", steps=144, **{
        "organizations.enabled": "true", "agents.personas_file": "data/personas_80.json",
        "relations.enabled": "true", "household.enabled": "true",
        "housing.relocation.enabled": "false", "household.cohabit.enabled": "false",
        "career.by_choice.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "orgs+household ON で mobility seam が no-op でない"


# --------------------------------------------------------------------- ① 転居
def _drive_relocation(sim):
    """世帯持ちの居住者を1人選び、職場変更検討フラグを立てて relocate_day を1回叩く。"""
    target = next(a for a in sim.agents
                  if not a.visitor and a.work_node and a.housemates)
    target._reloc_seen_work = target.work_node    # 差分検知はさせず、検討フラグを直に立てる
    target._reloc_consider = True
    mobility.relocate_day(sim, step=200, sim_min=200 * 10)
    return target


def test_relocation_moves_whole_household(tmp_path):
    """転居 ON: 職場変更→通勤閾値超で relocate(reason=job)。世帯全員の home が一致(一緒に動く)。"""
    sim = _sim(tmp_path, "reloc", **{
        "organizations.enabled": "true", "organizations.commute_to_poi": "true",
        "agents.personas_file": "data/personas_80.json", "household.enabled": "true",
        "housing.relocation.enabled": "true", "housing.relocation.job_prob": "1.0",
        "housing.relocation.commute_threshold_m": "50"})
    target = _drive_relocation(sim)
    rel = [e for e in _kind(sim, "relocate") if e.agent_id == target.id]
    assert rel and rel[-1].payload["reason"] == "job", "職場変更後の relocate(job)が出ていない"
    members = [target] + [sim.agent_by_id[i] for i in target.housemates]
    assert all(m.home_building == target.home_building for m in members), \
        "世帯全員の home_building が一致していない(一緒に動いていない)"
    assert all(m.home_node == target.home_node for m in members), \
        "世帯全員の home_node が一致していない"
    assert rel[-1].payload["n"] == len(members), "relocate.n が世帯人数と一致しない"


def test_relocation_destination_deterministic(tmp_path):
    """行き先は決定論(同 seed・同シナリオで同一の到着建物)。"""
    a = _sim(tmp_path, "rd_a", **{
        "organizations.enabled": "true", "organizations.commute_to_poi": "true",
        "agents.personas_file": "data/personas_80.json", "household.enabled": "true",
        "housing.relocation.enabled": "true", "housing.relocation.job_prob": "1.0",
        "housing.relocation.commute_threshold_m": "50"})
    b = _sim(tmp_path, "rd_b", **{
        "organizations.enabled": "true", "organizations.commute_to_poi": "true",
        "agents.personas_file": "data/personas_80.json", "household.enabled": "true",
        "housing.relocation.enabled": "true", "housing.relocation.job_prob": "1.0",
        "housing.relocation.commute_threshold_m": "50"})
    ta, tb = _drive_relocation(a), _drive_relocation(b)
    assert ta.id == tb.id and ta.home_building == tb.home_building, \
        "同 seed の転居先が非決定的"


# --------------------------------------------------------------------- ② 同棲/転出
def test_cohabit_move_in_and_move_out(tmp_path):
    """同棲 ON: bond→N日+closeness で move_in(世帯併合・home 共有)、unbond で move_out(世帯分離)。"""
    sim = _sim(tmp_path, "cohab", **{
        "relations.enabled": "true", "household.enabled": "true",
        "household.cohabit.enabled": "true", "household.cohabit.days": "2",
        "household.cohabit.closeness": "10"})
    res = [a for a in sim.agents if not a.visitor and a.home_building
           and sim.city.has_building(a.home_building)
           and getattr(a, "partner_id", None) is None]
    a, b = res[0], res[1]
    household_mod.bond(sim, a, b, step=0, sim_min=0)
    a.mem.record_contact(b.id, b.name, 0, "友人")["closeness"] = 20.0
    b.mem.record_contact(a.id, a.name, 0, "友人")["closeness"] = 20.0
    lo, hi = (a, b) if a.id < b.id else (b, a)     # keeper=id 小, mover=id 大
    lo._cohabit_since = hi._cohabit_since = 0
    keeper_home = lo.home_building
    mobility.cohabit_day(sim, step=288, sim_min=2880)   # day=2 >= since(0)+days(2)
    mi = _kind(sim, "move_in")
    assert mi, "同棲成立(move_in)が出ていない"
    assert hi.home_building == keeper_home, "後から来た側(mover)が keeper 宅へ移っていない"
    assert lo.household_id is not None and hi.household_id == lo.household_id, \
        "世帯が併合されていない"
    assert hi.id in lo.housemates and lo.id in hi.housemates, "housemates が相互に入っていない"
    assert getattr(hi, "_cohabit_mover", False), "mover フラグが立っていない"
    # 別離 → 転出(mover が新居へ・世帯分離)
    n0 = len(_kind(sim, "move_out"))
    household_mod.unbond(sim, hi, step=310, sim_min=3100)
    mo = _kind(sim, "move_out")
    assert len(mo) > n0, "別離時の転出(move_out)が出ていない"
    assert hi.household_id is None and hi.housemates == [], "mover が世帯から分離されていない"


# --------------------------------------------------------------------- ③ 求職(career選択由来化)
def test_parse_action_job_search():
    """parse_action が job_search を正規化する。"""
    assert parse_action(json.dumps({"action": "job_search"})) == {"type": "job_search"}


def test_job_search_switches_org(tmp_path):
    """求職 ON: job_search 発火→ switch_org 実行(org_id 変化)+ job_change(cause=job_search)。"""
    sim = _sim(tmp_path, "js", **{
        "organizations.enabled": "true", "agents.personas_file": "data/personas_80.json",
        "career.by_choice.enabled": "true"})
    scheduler._ensure_orgs(sim)                    # 配属を先に確定(run 前は org 未付与)
    agent = next(a for a in sim.agents
                 if organizations.is_employee(a) and not a.visitor)
    old = agent.org_id
    sim.tools.apply(sim, agent, {"type": "job_search"}, step=5, sim_min=50)
    js = _kind(sim, "job_search")
    assert js and js[-1].payload["outcome"] == "hired", "求職の hired が出ていない"
    assert agent.org_id is not None and agent.org_id != old, "org_id が変わっていない"
    jc = [e for e in _kind(sim, "job_change") if e.payload["cause"] == "job_search"]
    assert jc, "job_change(cause=job_search)が併記されていない"


def test_job_search_no_vacancy(tmp_path):
    """空き無し(現職以外に候補が無い)なら job_search{outcome:none} を記録し org は不変。"""
    sim = _sim(tmp_path, "jsn", **{
        "organizations.enabled": "true", "agents.personas_file": "data/personas_80.json",
        "career.by_choice.enabled": "true"})
    scheduler._ensure_orgs(sim)                    # 配属を先に確定(run 前は org 未付与)
    agent = next(a for a in sim.agents
                 if organizations.is_employee(a) and not a.visitor)
    sim.orgs = {str(agent.org_id): sim.orgs[str(agent.org_id)]}   # 現職のみ=空き無し
    old = agent.org_id
    sim.tools.apply(sim, agent, {"type": "job_search"}, step=5, sim_min=50)
    js = _kind(sim, "job_search")
    assert js and js[-1].payload["outcome"] == "none", "空き無しで outcome=none が出ていない"
    assert agent.org_id == old, "マッチ失敗なのに org_id が変わった"


# --------------------------------------------------------------------- R1 呼数 k/ON-OFF 不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_calls(tmp_path, name, by_choice):
    sim = _sim(tmp_path, name, n=30, steps=144, **{
        "organizations.enabled": "true", "agents.personas_file": "data/personas_80.json",
        "career.by_choice.enabled": by_choice, "controls.mode": "compute_matched"})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_job_search_call_count_invariant(tmp_path):
    """求職 tool は既存 tool 選択枠内=メニュー1行を足すだけ(新 generate なし)。compute_matched 下で
    by_choice ON/OFF の generate 呼数が完全一致する(R1)。"""
    on = _run_calls(tmp_path, "jc_on", "true")
    off = _run_calls(tmp_path, "jc_off", "false")
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"求職の呼数が by_choice に依存(R1 違反): on={on.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """3 機構 ON(既定確率)同士 2 回で L1 完全一致(決定論・mock 2 日=288 step)。"""
    a = _sim(tmp_path, "det_a", steps=288, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", steps=288, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- resume==straight
def _rows(run_dir: Path, stem: str = "l1_events") -> list:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _cfg_ov(name, n_steps, **ov):
    dot = ["run.seed=42", "run.n_agents=40", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run_straight(tmp_path, name, n_steps, **ov):
    d = tmp_path / name
    Simulation(_cfg_ov(name, n_steps, **ov), out_dir=d).run()
    return d


def _run_resume(tmp_path, name, split, total, **ov):
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    sim1 = Simulation(_cfg_ov(name, split, **every, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg_ov(name, total, **every, **ov), out_dir=d)
    sim2.run(resume_from=d)
    return d


# resume 用: career 由来の転居を境界跨ぎで発火させる(relations/household の日次フェーズは OFF=
# 未保存カウンタの二重発火を避ける=それらは OFF で完全 no-op)。career 由来転居は day0 で職場変更→
# day1(step144)で通勤逼迫を検知して relocate=境界を跨いで発火する。
_ON_RESUME = {
    "organizations.enabled": "true", "organizations.commute_to_poi": "true",
    "agents.personas_file": "data/personas_80.json",
    "career.enabled": "true", "career.switch_prob": "0.3", "career.layoff_prob": "0.0",
    "housing.relocation.enabled": "true",
    "housing.relocation.commute_threshold_m": "100",
    "housing.relocation.job_prob": "1.0",
    "household.cohabit.enabled": "true", "career.by_choice.enabled": "true",
}


def test_resume_matches_straight_mechanisms_on(tmp_path):
    """3 機構 ON(career 駆動)で 100+resume→200 が straight 200 と L1/L2/L3 全行一致(日境界 step144 跨ぎ)。

    _housing_day / _career_day を checkpoint 中央管理し、resume 時は org を再 attach しない(転職を潰さ
    ない)ことで、mid-day checkpoint(step100=day0)からの resume でも転居/転職の日境界処理が二重発火
    せず straight とバイト一致する(B4/第59 と同型)。"""
    straight = _run_straight(tmp_path, "m_straight", 200, **_ON_RESUME)
    resumed = _run_resume(tmp_path, "m_resume", 100, 200, **_ON_RESUME)
    a, b = _rows(straight, "l1_events"), _rows(resumed, "l1_events")
    assert any(r["kind"] == "relocate" for r in a), \
        "resume テストが relocate を跨いで発火していない(機構未行使=テスト無効)"
    assert len(a) == len(b), f"l1 行数不一致: {len(a)} vs {len(b)}"
    assert a == b, "l1_events が byte 級で不一致(3 機構 ON の resume)"
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"
