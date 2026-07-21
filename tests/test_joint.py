"""共同行動エンジン(関係性の再現 第44バッチ S-R3)= 友人・同居人の関係駆動ランデブーのテスト。

方針(既存の鉄則を継承):
- OFF(既定): joint_activity 0 件・誰も joint_today を持たない・新 stream "joint" も引かない・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- ON 決定論: 同 seed 2 ラン で L1 完全一致。
- ON 機能: mock ラン(household+relations ON)で joint_activity が出て、参加者が同一 POI に同席する。
- R1: 合流は物理位置=対面 co-location を変えうる(career/health/household と同型で許容)。呼数不変は
  compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock / _FixedLLM のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine.simulation import Simulation

# 友人系の同伴が成立する状態(household=housemates、relations=友人 closeness を読む)。
# daily_rate/accept_base を高くして機構を確実に発火させる(単体検証)。
_JOINT_ON = {"joint.enabled": "true", "joint.daily_rate": "1.0",
             "joint.accept_base": "1.0", "household.enabled": "true",
             "relations.enabled": "true"}


def _sim(tmp_path, name, n=20, steps=1, **ov):
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
    """明示 OFF と純粋既定が L1 完全一致(144 step)。joint_activity は 1 件も出ず誰も joint_today を持たない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"joint.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(joint seam が no-op でない)"
    assert not _kind(pure, "joint_activity"), "OFF で joint_activity が出ている"
    assert all(getattr(a, "joint_today", None) is None for a in pure.agents), \
        "OFF なのに joint_today が付いている"


# --------------------------------------------------------------------- 決定論
def test_on_deterministic(tmp_path):
    """joint+household+relations ON 同士 2 回で L1 完全一致(決定論・mock 100 step)。"""
    a = _sim(tmp_path, "det_a", n=20, steps=100, **_JOINT_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=20, steps=100, **_JOINT_ON)
    b.run()
    assert _l1(a) == _l1(b), "joint ON の決定論が崩れている"


# --------------------------------------------------------------------- 機能(合流)
def test_joint_fires_and_colocated(tmp_path):
    """joint ON: joint_activity が1件以上出て、参加者が同一 POI へ収束・同席していることを検証する。

    steps=100 は day 境界(step 102)より前=当日の成立グループが sim._joint_groups に残る。
    観測(observe)は「実際に POI で2人以上同席した step」でのみ記録する=イベントの with は
    その時点で place に居た参加者。ランデブー(同一 POI 収束)を group 台帳と突き合わせて確認する。"""
    sim = _sim(tmp_path, "fire", n=20, steps=100, **_JOINT_ON)
    sim.run()
    ja = _kind(sim, "joint_activity")
    assert ja, "joint ON なのに joint_activity が1件も出ていない"
    groups = {g["poi"]: g for g in sim._joint_groups}
    for e in ja:
        who = e.payload["with"]
        place = e.payload["place"]
        assert len(who) >= 2, "joint_activity の参加者が2人未満(同席=合流でない)"
        assert place, "joint_activity に place(合流 POI)が無い"
        grp = groups.get(place)
        assert grp is not None, "イベントの place に対応する成立グループが無い"
        assert set(who).issubset(set(grp["members"])), \
            "同席者が成立グループのメンバでない"
        assert grp["poi"] == place, "グループのランデブー POI とイベント place が不一致"
        assert e.payload["type"] in sim.jointcfg["activities"], \
            "未知の活動タイプが記録されている"


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


def _run_k(tmp_path, name, *, writeback, joint_on):
    """joint(+household+relations)を compute_matched(k 掃引で使う対照)下で回し呼数を数える。"""
    ov = dict(_JOINT_ON) if joint_on else {}
    sim = _sim(tmp_path, name, n=20, steps=100,
               **{**ov, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_joint_call_count_k_invariant(tmp_path):
    """合流(同一 POI 収束)は対面 co-location を変え ON!=OFF になりうる(実在する紐帯の必然=career/health/
    household と同型で許容。鉄則 R1 3)。R1 の本旨=「呼数が k(writeback)に依存しない」ことは
    compute_matched 下で厳密に保たれる。joint 機構は名簿+物理位置+config+relations closeness
    (観測イベント由来=k 非依存の観測量)+新 stream のみ参照し、発火判断に k・内面状態(構成概念)を
    食わせないため、joint ON のまま k=free と k=off で generate 呼数が完全一致する(household の R1 テストと同型)。"""
    free = _run_k(tmp_path, "jk_free", writeback="free", joint_on=True)
    off = _run_k(tmp_path, "jk_off", writeback="off", joint_on=True)
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"joint の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "joint_activity"), "joint ON なのに joint_activity が出ていない(機構が不発)"


# --------------------------------------------------------------------- 職場の会食(S-R4)
from society import joint as _joint  # noqa: E402


def test_accept_prob_hierarchy_penalty():
    """上司同席(hierarchical かつ年齢差>boss_age_gap)で承諾確率が下がる=同世代/非階層では下がらない。"""
    cfg = _joint.build_cfg({"accept_base": 0.5, "tier_bonus": 0.3,
                            "hierarchy_penalty": 0.35, "boss_age_gap": 10})
    same_gen = _joint.accept_prob(cfg, tier=2, hierarchical=True, age_gap=5)     # 同世代の飲み会
    boss = _joint.accept_prob(cfg, tier=2, hierarchical=True, age_gap=20)        # 上司同席の飲み会
    lunch = _joint.accept_prob(cfg, tier=2, hierarchical=False, age_gap=20)      # 非階層(ランチ)は年齢差無関係
    assert boss < same_gen, "上司同席(年齢差大)でも承諾率が下がっていない(階層依存が効いていない)"
    assert lunch == same_gen, "非階層の活動で年齢差ペナルティが掛かっている(§2.5=ランチは許容)"
    assert abs(boss - (0.5 - 0.35)) < 1e-9, "hierarchy_penalty の適用値が式と一致しない"
    # 親友(tier3)への tier_bonus は上乗せされる(階層とは独立)。
    assert _joint.accept_prob(cfg, tier=3, hierarchical=True, age_gap=5) > same_gen


def test_colleague_companion_selection(tmp_path):
    """companion_type=colleague は同 org_id の同僚(非来街者)だけを同伴候補にする(友人/同居人経路と別)。"""
    sim = _sim(tmp_path, "colle", n=20, steps=1)
    a, b, c, d = sim.agents[0], sim.agents[1], sim.agents[2], sim.agents[3]
    for x in (a, b, c, d):
        x.visitor = False
    a.org_id = b.org_id = "orgX"     # a,b は同僚
    c.org_id = "orgY"                # c は別組織
    d.org_id = None                  # d は未配属
    cands = _joint._companions(sim, a, sim.jointcfg, set(), "colleague")
    assert b.id in cands, "同 org_id の同僚が候補に入っていない"
    assert c.id not in cands and d.id not in cands, "別組織/未配属が同僚候補に混入している"
    # 来街者は除外される
    b.visitor = True
    assert b.id not in _joint._companions(sim, a, sim.jointcfg, set(), "colleague"), \
        "来街者の同僚が候補に残っている"
