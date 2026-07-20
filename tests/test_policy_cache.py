"""行動方針キャッシュ(AGA Lifestyle Policy)= P2 スライス S7 のテスト。

設計: docs/research/interstitial-life.md §4.4(4-b)/§7・docs/plans/p2-interstitial-design.md §1 S7。
既定 OFF 運用(本番採否は比較実験で決める)。R1 の関門:
  ① 呼数 k 非依存(k∈{free,off} で再利用数・LLM 呼数が一致)
  ② 既定 OFF でゴールデン L1 バイト一致(no-op)
  ③ ゲート入力に信念系が含まれない(キー構築関数の入力型を静的に検査)

会話・実 LLM は使わない。再利用は「2 日目以降で同じ物理骨格を再訪したら起きる」構造なので、
mock ≤48step の通しランでは自然発火しない(朝計画は 1 日 1 回・翌朝は step≥144)。よって
再利用機構は planning.make_plan を **2 日ぶん直接駆動** して検証する(通しランは決定論・
バイト一致・k 非依存の確認に使う)。
"""
from __future__ import annotations

import inspect
import json

from society.cognition import deliberate, planning
from society.cognition import policy_cache as pc
from society.config import load_config
from society.engine.simulation import Simulation


# --------------------------------------------------------------------------- #
def _sim(tmp_path, name: str, ov: list[str] | None = None,
         n: int = 8, steps: int = 1) -> Simulation:
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "observer.snapshot_every=1000",
           *(ov or [])]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


_ON = ["cognition.policy_cache.enabled=true"]


def _dump(events):
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in events]


def _wake_at_home(sim, a, sim_min: int) -> None:
    """エージェントを朝・自宅・起床済みの物理状態に置く(朝計画の発火条件)。"""
    a.loc, a.building, a.sleeping, a.route = "street", None, False, []
    a.node = a.home_node
    a.x, a.y = sim.city.node_xy(a.node)
    a.activity = ""


_D0 = 8 * 60                     # day0 08:00(朝)
_D7 = 7 * 1440 + 8 * 60          # day7 08:00(day0 と同一曜日 → weekday_type 一致を保証)


def _plan_calls(sim) -> int:
    return sum(1 for c in sim.logger.llm_calls if c.get("purpose") == "plan")


def _reuses(sim) -> list:
    return [e for e in sim.logger.events if e.kind == "policy_reuse"]


# --------------------------------------------------------------------------- #
# (OFF) ゴールデン=バイト一致 / no-op
# --------------------------------------------------------------------------- #
def test_off_default_is_deterministic_and_no_reuse(tmp_path):
    """既定(policy_cache 未設定)は決定論 + policy_reuse が一切出ない(no-op)。"""
    a = _sim(tmp_path, "off_a", steps=40)
    a.run()
    b = _sim(tmp_path, "off_b", steps=40)
    b.run()
    assert _dump(a.logger.events) == _dump(b.logger.events)
    kinds = {e.kind for e in a.logger.events}
    assert "policy_reuse" not in kinds


def test_off_absent_equals_explicit_false(tmp_path):
    """cognition.policy_cache 節が無い(既定)と enabled=false の明示は L1 バイト一致。

    168 step = 翌日 11:00 まで回して **朝計画(day_plan)を実際に発火**させ、make_plan の
    OFF フックが no-op であることを実ラン上でバイト比較する(test_planning と同じ長さ)。"""
    a = _sim(tmp_path, "absent", steps=168)
    a.run()
    b = _sim(tmp_path, "explicit_false",
             ov=["cognition.policy_cache.enabled=false"], steps=168)
    b.run()
    da, db = _dump(a.logger.events), _dump(b.logger.events)
    assert da == db
    assert any(k == "day_plan" for (_s, _i, k, _p) in da)   # OFF フック経路を確かに通した
    assert not any(k == "policy_reuse" for (_s, _i, k, _p) in da)


def test_off_make_plan_hook_is_noop(tmp_path):
    """OFF では make_plan の cache フックが完全 no-op: 2 日駆動しても毎回 LLM を呼び
    (plan 呼 2 本)policy_reuse は 0(再利用しない)。"""
    sim = _sim(tmp_path, "off_hook")
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    planning.make_plan(sim, a, 3, _D0, "自宅")
    _wake_at_home(sim, a, _D7)
    planning.make_plan(sim, a, 150, _D7, "自宅")
    assert _plan_calls(sim) == 2
    assert _reuses(sim) == []
    assert getattr(sim, "_policy_cache", None) is None    # cache すら作らない


# --------------------------------------------------------------------------- #
# (a) ON: 2 日目で再利用が起き、LLM 呼がスキップされる
# --------------------------------------------------------------------------- #
def test_reuse_fires_and_skips_llm_on_second_day(tmp_path):
    sim = _sim(tmp_path, "reuse", ov=_ON + ["cognition.policy_cache.reuse_rate_cap=1.0"])
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    planning.make_plan(sim, a, 3, _D0, "自宅")           # day0: miss → 生成 + 格納
    assert _plan_calls(sim) == 1
    assert _reuses(sim) == []

    _wake_at_home(sim, a, _D7)
    planning.make_plan(sim, a, 150, _D7, "自宅")          # day7: hit → 再利用(LLM スキップ)
    assert _plan_calls(sim) == 1                          # LLM 呼は増えていない
    reuses = _reuses(sim)
    assert len(reuses) == 1
    assert reuses[0].payload == {"kind": "plan", "relax": 0, "saved": 1}
    day_plans = [e for e in sim.logger.events if e.kind == "day_plan"]
    assert len(day_plans) == 2
    assert day_plans[1].llm_call_id is None              # 再利用は LLM 呼に紐づかない
    assert a.day_plan and all("done" in it for it in a.day_plan)


# --------------------------------------------------------------------------- #
# (b) 同 seed 2 回で完全一致
# --------------------------------------------------------------------------- #
def test_same_seed_two_runs_identical(tmp_path):
    def drive(name):
        sim = _sim(tmp_path, name, ov=_ON + ["cognition.policy_cache.reuse_rate_cap=1.0"])
        a = sim.agents[0]
        _wake_at_home(sim, a, _D0)
        planning.make_plan(sim, a, 3, _D0, "自宅")
        _wake_at_home(sim, a, _D7)
        planning.make_plan(sim, a, 150, _D7, "自宅")
        return _dump(sim.logger.events)
    assert drive("run1") == drive("run2")


# --------------------------------------------------------------------------- #
# (c) k∈{free,off} で LLM 呼数・再利用数が一致(R1 関門①)
# --------------------------------------------------------------------------- #
def test_call_counts_are_k_invariant(tmp_path):
    """信念(k の作用点)が違っても、キーが信念を排除するので再利用判定・呼数が一致する。"""
    def drive(wb: str, beliefs: list[str]):
        sim = _sim(tmp_path, f"kinv_{wb}",
                   ov=_ON + ["cognition.policy_cache.reuse_rate_cap=1.0",
                             f"k.writeback={wb}"])
        a = sim.agents[0]
        a.beliefs = list(beliefs)                        # free だけ信念を持たせる
        _wake_at_home(sim, a, _D0)
        planning.make_plan(sim, a, 3, _D0, "自宅")
        _wake_at_home(sim, a, _D7)
        planning.make_plan(sim, a, 150, _D7, "自宅")
        return _plan_calls(sim), len(_reuses(sim))
    free = drive("free", ["ある考えが強くなってきた", "街は変えられる"])
    off = drive("off", [])
    assert free == off == (1, 1)


# --------------------------------------------------------------------------- #
# (d) 再利用率がハード上限を超えない(多様性の下限を保証)
# --------------------------------------------------------------------------- #
def test_reuse_rate_capped(tmp_path):
    cap = 0.3
    sim = _sim(tmp_path, "cap",
               ov=_ON + [f"cognition.policy_cache.reuse_rate_cap={cap}"])
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    for i in range(24):                                  # 同一物理骨格を 24 回再訪(step だけ変える)
        planning.make_plan(sim, a, 10 + i, _D0, "自宅")
    cache = sim._policy_cache
    assert cache.n_opportunities == 24
    assert cache.n_reuses > 0                            # 上限内では再利用が起きる
    assert cache.n_reuses <= cap * cache.n_opportunities  # ハード上限を超えない
    # 観測(policy_reuse イベント数)= n_reuses と一致
    assert len(_reuses(sim)) == cache.n_reuses


# --------------------------------------------------------------------------- #
# (e) キャッシュが有界(個体毎上限 + 全体 LRU で追い出し)
# --------------------------------------------------------------------------- #
def _key(place="home", band="朝", persona="会社員/30"):
    return pc.build_key(pc.SituationSkeleton(
        persona=persona, weekday_type="workday", time_band=band,
        place_kind=place, weather="na", activity_cat="discretionary"))


def test_cache_bounded_per_agent():
    cache = pc.PolicyCache()
    for i in range(5):                                   # 個体上限 3 に対し 5 件投入
        cache.store(0, _key(place=f"p{i}"), [{"i": i}], per_agent_max=3, global_max=100)
    assert len(cache.by_agent[0]) == 3
    assert cache.size() == 3
    assert cache.lookup(0, _key(place="p0"), 0) is None  # 古い 2 件は追い出し
    assert cache.lookup(0, _key(place="p1"), 0) is None
    assert cache.lookup(0, _key(place="p4"), 0) is not None


def test_cache_bounded_global_lru():
    cache = pc.PolicyCache()
    for aid in range(5):                                 # 全体上限 3 に対し 5 個体×1 件
        cache.store(aid, _key(place=f"a{aid}"), [{"a": aid}],
                    per_agent_max=100, global_max=3)
    assert cache.size() == 3
    assert cache.lookup(0, _key(place="a0"), 0) is None  # 最古の個体は追い出し
    assert cache.lookup(4, _key(place="a4"), 0) is not None


# --------------------------------------------------------------------------- #
# (f) near-match 緩和段(完全一致 → 場所種別 → 時間帯)の単体
# --------------------------------------------------------------------------- #
def test_near_match_relaxation_stages():
    cache = pc.PolicyCache()
    cache.store(0, _key(place="home", band="朝"), ["POLICY"],
                per_agent_max=16, global_max=64)

    # 完全一致 → stage 0
    hit = cache.lookup(0, _key(place="home", band="朝"), 2)
    assert hit == (["POLICY"], 0)
    # 場所種別だけ違う → stage 1(max_relax>=1)。max_relax=0 では不一致
    q1 = _key(place="cafe", band="朝")
    assert cache.lookup(0, q1, 0) is None
    assert cache.lookup(0, q1, 2)[1] == 1
    # 場所種別+時間帯が違う → stage 2(max_relax>=2)。max_relax=1 では不一致
    q2 = _key(place="cafe", band="夜")
    assert cache.lookup(0, q2, 1) is None
    assert cache.lookup(0, q2, 2)[1] == 2
    # ペルソナが違う → どの段でも不一致(persona は緩和しない)
    q3 = _key(place="home", band="朝", persona="大学生/20")
    assert cache.lookup(0, q3, 2) is None


# --------------------------------------------------------------------------- #
# (関門③) キーに信念系が混入しないことの静的チェック
# --------------------------------------------------------------------------- #
def test_key_inputs_exclude_beliefs_static():
    # SituationSkeleton のフィールド名(トークン単位)に信念系の語が無い
    names = pc.skeleton_field_names()
    tokens: set[str] = set()
    for nm in names:
        tokens.update(nm.lower().split("_"))
    assert tokens.isdisjoint(pc.FORBIDDEN_KEY_TERMS), tokens & pc.FORBIDDEN_KEY_TERMS
    # build_key の唯一の入力は SituationSkeleton(信念系を受け取れない型)
    params = list(inspect.signature(pc.build_key).parameters)
    assert params == ["skel"]
    ann = inspect.signature(pc.build_key).parameters["skel"].annotation
    assert ann is pc.SituationSkeleton or ann == "SituationSkeleton"


def test_key_unchanged_by_beliefs(tmp_path):
    """同一物理状況なら信念/意見/drive が変わってもキーは不変(呼数 k 非依存の根拠)。"""
    sim = _sim(tmp_path, "keyeq")
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    k1 = pc.build_key(pc._plan_skeleton(sim, a, _D0))
    a.beliefs = ["世界は変えられると思うようになった"]
    a.opinion = 0.9
    a.drive = 0.8
    k2 = pc.build_key(pc._plan_skeleton(sim, a, _D0))
    assert k1 == k2


# --------------------------------------------------------------------------- #
# 熟慮の cognition 層入口(scheduler 差し込みは主計画者が統合)。ここでは入口が
# 機能することを直接確認する(reuse_action / store_action の往復)。
# --------------------------------------------------------------------------- #
def test_deliberate_entry_reuse_roundtrip(tmp_path):
    sim = _sim(tmp_path, "delib",
               ov=_ON + ["cognition.policy_cache.reuse_rate_cap=1.0"])
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    # 1 回目: 未格納なので None(呼び出し側は通常 LLM 生成へ)
    assert deliberate.maybe_reuse_action(sim, a, 5, _D0, "social") is None
    action = {"type": "speak", "text": "やあ"}
    deliberate.store_action(sim, a, 5, _D0, "social", action)
    # 2 回目(翌週の朝・同トリガ): 再利用が起きて action のコピーが返る
    got = deliberate.maybe_reuse_action(sim, a, 160, _D7, "social")
    assert got == action and got is not action           # コピー(共有参照でない)
    reuses = _reuses(sim)
    assert len(reuses) == 1 and reuses[0].payload["kind"] == "deliberate"


def test_off_deliberate_entry_is_noop(tmp_path):
    sim = _sim(tmp_path, "delib_off")
    a = sim.agents[0]
    _wake_at_home(sim, a, _D0)
    assert deliberate.maybe_reuse_action(sim, a, 5, _D0, "social") is None
    deliberate.store_action(sim, a, 5, _D0, "social", {"type": "speak", "text": "x"})
    assert deliberate.maybe_reuse_action(sim, a, 160, _D7, "social") is None
    assert getattr(sim, "_policy_cache", None) is None
