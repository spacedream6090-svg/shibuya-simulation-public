"""ACT-R 活性化ベースの人間らしい記憶(M-W)のテスト。

設計正典: docs/research/memory-cognitive-research.md 実装スケッチ M-W。
既定 OFF(`actr is None`)は現行 GA 0.5:2:3 とバイト一致=検証は test_scenario/
test_determinism/test_memory(_pull) が担う。ここでは:
  - OFF 不変(refs 動的属性を一切付けない・failed 常に False・忘却しない・mock 短ラン決定論)
  - 基礎活性化 B=ln(Σ Δt^−d) の数値(既知 refs→期待 B±0.01)
  - 再強化(想起成功で refs に現在 step 追記→活性化↑=testing effect)
  - ノイズ付き閾値の想起失敗率が理論値(ロジスティック CDF)と大数一致
  - memory_fail イベントの発火(内省 agentic pull で全候補が閾値未達)
  - fan 効果(類似記憶が多いほど個別の連想後押しが弱る=干渉)
  - 忘却間引きの決定論(同 seed→同じ生存集合・要約に痕跡)
"""
from __future__ import annotations

import math

from society.agents.memory import ACTR_DEFAULTS, Episode, MemoryStore
from society.observer.logger import ObserverLogger


# --------------------------------------------------------------------------- #
# OFF 不変(既定 actr is None)
# --------------------------------------------------------------------------- #
def test_off_is_unchanged_and_never_writes_refs():
    """OFF 時: retrieve/query は現行どおり・refs 動的属性を一切付けない・failed は常に False。"""
    m = MemoryStore()                              # actr=None(既定)
    m.observe(0, "駅前でコーヒーを飲んだ")
    m.consolidate(10, "初日", [("宮下公園で花音と長話した", 8.0),
                               ("道玄坂の混雑にうんざりした", 6.0)])
    got = m.retrieve(20, ["宮下公園", "花音"], n=2)
    assert got and "宮下公園" in got[0]
    # OFF は refs を pickle に混ぜない=Episode に refs 属性が存在しない(checkpoint バイト一致)
    for ep in m.episodes + m.buffer:
        assert not hasattr(ep, "refs"), "OFF なのに refs 属性が付いている(pickle バイト非一致)"
    # OFF の query_ex は failed 常に False(=deliberate/reflection の失敗1行が出ない=バイト一致)
    assert m.query_ex(20, "宮下公園 花音", n=2).failed is False
    assert m.query_ex(20, "存在しない手掛かり語", n=2).failed is False
    assert m.query_ex(20, "", n=2).hits == []      # 手掛かり無しは空(暴発防止)


def test_off_consolidate_never_forgets():
    """OFF の consolidate は間引かない(忘却は ACT-R ON のときだけ動く)。"""
    m = MemoryStore(store_cap=999)
    for i in range(20):
        m.episodes.append(Episode(step=0, text=f"遠い過去の出来事{i}", importance=3.0))
    m.consolidate(100000, "要約", None)            # Δt 巨大でも OFF は誰も消さない
    assert len(m.episodes) == 20


def test_off_mock_run_deterministic_no_memory_fail(tmp_path):
    """mock 短ラン(既定=ACT-R OFF): 同 seed で2回=完全一致・memory_fail は1件も出ない。"""
    from society.config import load_config
    from society.engine.simulation import Simulation

    def _run(name):
        cfg = load_config(["run.seed=5", "run.n_agents=10", "run.n_steps=72",
                           f"run.name={name}"])
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.run()
        return sim

    a, b = _run("off_a"), _run("off_b")
    ea = [(e.step, e.agent_id, e.kind, str(e.payload)) for e in a.logger.events]
    eb = [(e.step, e.agent_id, e.kind, str(e.payload)) for e in b.logger.events]
    assert ea == eb, "既定 OFF の mock 短ランが決定論でない"
    assert not any(e.kind == "memory_fail" for e in a.logger.events)
    # OFF ラン後も Episode に refs 属性が漏れていない(動的属性 S3 方式の担保)
    assert all(not hasattr(ep, "refs")
               for ag in a.agents for ep in ag.mem.episodes + ag.mem.buffer)


# --------------------------------------------------------------------------- #
# 基礎活性化 B の数値(既知 refs→期待 B±0.01)
# --------------------------------------------------------------------------- #
def test_base_activation_matches_power_law():
    """B = ln(Σ_j Δt_j^−d)、d=0.5。単一参照と複数参照の既知値に ±0.01 で一致。"""
    m = MemoryStore(actr=MemoryStore.actr_config())
    d = m.actr["d"]
    ep = Episode(step=0, text="出来事")           # refs 未設定→暗黙 [ep.step]=[0]
    # 単一参照: B = ln(100^−0.5) = ln(0.1) = −2.302585
    b1 = m._base_activation(ep, 100, d)
    assert abs(b1 - (-2.302585)) < 0.01
    assert abs(b1 - math.log(100 ** -0.5)) < 1e-9
    # 複数参照(再想起で refs が増えた状態): refs=[0, 90], step=100
    ep.refs = [0, 90]
    b2 = m._base_activation(ep, 100, d)
    assert abs(b2 - (-0.876500)) < 0.01
    assert abs(b2 - math.log(100 ** -0.5 + 10 ** -0.5)) < 1e-9
    # 参照が増える(再想起)ほど基礎活性化は上がる(頻度効果)
    assert b2 > b1


# --------------------------------------------------------------------------- #
# 再強化(testing effect): 想起成功で refs に現在 step を追記→活性化↑
# --------------------------------------------------------------------------- #
def test_strengthen_raises_activation():
    """_strengthen 直接: refs に step が入り、基礎活性化が上がる。"""
    m = MemoryStore(actr=MemoryStore.actr_config())
    d = m.actr["d"]
    ep = Episode(step=0, text="想起した出来事")
    before = m._base_activation(ep, 100, d)
    m._strengthen(ep, 95)
    assert getattr(ep, "refs", None) == [0, 95]    # 生成 step を種に、想起 step を追記
    after = m._base_activation(ep, 100, d)
    assert after > before


def test_retrieve_strengthens_surfaced_episode():
    """retrieve で想起に成功した episode の refs に想起 step が追記される(end-to-end)。"""
    # s を極小にしてノイズを無効化=決定論的に閾値超えを確定させる
    m = MemoryStore(actr=MemoryStore.actr_config(s=1e-6, seed=1))
    m.episodes.append(Episode(step=90, text="宮下公園で花音と再会", importance=8.0))
    hits = m.retrieve(100, ["宮下公園", "花音"], n=2, agent_id=5)
    assert hits and "宮下公園" in hits[0]
    assert getattr(m.episodes[0], "refs", None) == [90, 100]


# --------------------------------------------------------------------------- #
# ノイズ付き閾値: 想起失敗率が理論値(ロジスティック CDF)と大数一致
# --------------------------------------------------------------------------- #
def test_noise_threshold_failure_rate_matches_theory():
    """A=detA+ε(ε~Logistic(0,s))。fail=A<τ の頻度が 1/(1+e^(-(τ-detA)/s)) に大数一致。"""
    m = MemoryStore(actr=MemoryStore.actr_config(s=0.5, seed=0))
    p = m.actr
    tau, s = p["tau"], p["s"]
    # Δt=55 で B≈τ(単一参照・importance=3 なので detA=B、閾値近傍で fail≈0.5)
    ep = Episode(step=0, text="夕方の駅前", importance=3.0)
    step = 55
    det = m._det_activation(ep, step, [], {}, p["W"], p["d"], p["S"])
    theo_fail = 1.0 / (1.0 + math.exp(-(tau - det) / s))
    n = 6000
    fails = sum(1 for aid in range(n)
                if det + m._recall_noise(aid, step, ep, s) < tau)
    emp = fails / n
    assert abs(emp - theo_fail) < 0.03, f"emp={emp:.3f} theo={theo_fail:.3f}"
    # 閾値を大きく上回る記憶はほぼ想起成功、大きく下回る記憶はほぼ失敗(単調性)
    now = 5000
    strong = Episode(step=now - 1, text="さっきの強い記憶", importance=10.0)  # Δt=1・高重要
    weak = Episode(step=0, text="遠い薄い記憶", importance=1.0)               # Δt=5000・低重要
    fs = sum(1 for aid in range(2000)
             if m._det_activation(strong, now, [], {}, p["W"], p["d"], p["S"])
             + m._recall_noise(aid, now, strong, s) < tau)
    fw = sum(1 for aid in range(2000)
             if m._det_activation(weak, now, [], {}, p["W"], p["d"], p["S"])
             + m._recall_noise(aid, now, weak, s) < tau)
    assert fs < 100 and fw > 1900, f"単調性が崩れている: strong_fail={fs} weak_fail={fw}"


# --------------------------------------------------------------------------- #
# memory_fail イベントの発火(内省 agentic pull で全候補が閾値未達)
# --------------------------------------------------------------------------- #
class _StubLLM:
    """recall 第1段の応答を固定で返す最小 LLM(呼数・rng は問わない)。"""
    def __init__(self, query: str):
        self._q = query

    def generate(self, prompt, *, rng_key=None, temperature=0.0,
                 max_tokens=0, think=False):
        return ('{"action": "recall", "query": "%s"}' % self._q, "call-x", False)


def _make_agent(mem: MemoryStore):
    from society.agents.agent import Agent
    a = Agent(id=1, name="被験者", age=30, occupation="会社員",
              persona="渋谷で暮らす一人の人間。", traits={}, states={})
    a.mem = mem
    return a


def test_memory_fail_event_fires_on_recall_failure(tmp_path):
    """ACT-R ON の内省 pull で、手掛かりはあるが全候補が閾値未達なら memory_fail が出る。"""
    from society.cognition.reflection import _recall_query

    # 極古の記憶1件のみ(Δt 巨大で B が深く負)→ 手掛かりが当たっても閾値に届かない
    mem = MemoryStore(actr=MemoryStore.actr_config(s=0.05, seed=3))
    mem.episodes.append(Episode(step=0, text="宮下公園で長話した", importance=3.0))
    agent = _make_agent(mem)
    logger = ObserverLogger(tmp_path)
    llm = _StubLLM("宮下公園")

    hits, fail_line = _recall_query(agent, step=100000, sim_min=1440, llm=llm,
                                    place_name="自宅", logger=logger)
    assert hits == []                              # 全候補が閾値未達=何も想起できない
    assert fail_line and "はっきりしない" in fail_line
    fails = [e for e in logger.events if e.kind == "memory_fail"]
    assert len(fails) == 1
    ev = fails[0]
    assert ev.payload["query"] == "宮下公園"
    assert ev.payload["tau"] == mem.actr["tau"]
    assert ev.payload["activation"] < ev.payload["tau"]   # 最良候補でも閾値未達
    # memory_recall も出る(n_hits=0)
    assert any(e.kind == "memory_recall" and e.payload["n_hits"] == 0
               for e in logger.events)


def test_recall_success_emits_no_memory_fail(tmp_path):
    """想起に成功する強い記憶では memory_fail は出ない(失敗イベントの偽陽性なし)。"""
    from society.cognition.reflection import _recall_query

    mem = MemoryStore(actr=MemoryStore.actr_config(s=1e-6, seed=3))
    mem.episodes.append(Episode(step=99, text="宮下公園で花音と再会", importance=9.0))
    agent = _make_agent(mem)
    logger = ObserverLogger(tmp_path)
    llm = _StubLLM("宮下公園")

    hits, fail_line = _recall_query(agent, step=100, sim_min=1440, llm=llm,
                                    place_name="自宅", logger=logger)
    assert hits and "宮下公園" in hits[0]
    assert fail_line is None
    assert not any(e.kind == "memory_fail" for e in logger.events)


# --------------------------------------------------------------------------- #
# fan 効果(類似記憶が多いほど個別の連想後押しが弱る=干渉)
# --------------------------------------------------------------------------- #
def test_fan_effect_weakens_common_cue():
    """同じ手掛かり語 j に紐づく記憶が多い(fan_j 大)ほど、その j 経由の後押し S−ln(fan) が減る。"""
    m = MemoryStore(actr=MemoryStore.actr_config())
    p = m.actr
    d, S, W = p["d"], p["S"], p["W"]        # W_j=W/n_cues、ここでは 1 手掛かりなので W
    target = Episode(step=90, text="渋谷で偶然の再会", importance=5.0)
    ctx = ["渋谷"]
    a_low = m._det_activation(target, 100, ctx, {"渋谷": 1}, W, d, S)    # ありふれない=強い
    a_high = m._det_activation(target, 100, ctx, {"渋谷": 12}, W, d, S)  # ありふれる=弱い
    assert a_low > a_high
    # 差は連想強度の差 S−ln(1) − (S−ln(12)) = ln(12) にちょうど一致(理論)
    assert abs((a_low - a_high) - math.log(12)) < 1e-9


# --------------------------------------------------------------------------- #
# 忘却間引きの決定論(同 seed→同じ生存集合・要約に痕跡)
# --------------------------------------------------------------------------- #
def _store_with_old_episodes(seed: int) -> MemoryStore:
    m = MemoryStore(actr=MemoryStore.actr_config(seed=seed,
                                                 forget_floor=-2.0, forget_prob=0.5))
    for i in range(20):                     # Δt 巨大→ B≈−5.7 < floor で全て間引き候補
        m.episodes.append(Episode(step=0, text=f"遠い記憶{i}", importance=3.0))
    return m


def test_forgetting_is_deterministic_and_leaves_trace():
    """ACT-R ON の consolidate: 下限未満の episodes を確率的に間引くが同 seed なら完全再現。"""
    m1 = _store_with_old_episodes(seed=7)
    m2 = _store_with_old_episodes(seed=7)
    for m in (m1, m2):
        m.consolidate(100000, "その日の要約", None, agent_id=3)
    surv1 = [e.text for e in m1.episodes]
    surv2 = [e.text for e in m2.episodes]
    assert surv1 == surv2                    # 同 seed→同じ生存集合(決定論)
    assert 0 < len(surv1) < 20               # 一部が間引かれ、全滅もしない(確率的間引き)
    assert m1.day_summaries[-1] == "その日の要約"   # 要約側に痕跡が残る(完全消去でない)


def test_forgetting_spares_fresh_and_active_episodes():
    """直近で強い(高活性)episodes は下限を割らないので間引かれない。"""
    m = MemoryStore(actr=MemoryStore.actr_config(seed=11, forget_prob=1.0))
    m.episodes.append(Episode(step=99999, text="ついさっきの鮮明な記憶", importance=8.0))
    m.episodes.append(Episode(step=0, text="遠い薄い記憶", importance=1.0))
    m.consolidate(100000, "要約", None, agent_id=1)
    texts = [e.text for e in m.episodes]
    assert "ついさっきの鮮明な記憶" in texts   # 高活性は残る
    assert "遠い薄い記憶" not in texts         # 下限未満は prob=1.0 で確実に間引き


def test_actr_defaults_are_canonical():
    """既定パラメータが研究ドク §f の値(d=0.5, τ=-2, s=0.5, S=2, W=1)であること。"""
    assert ACTR_DEFAULTS["d"] == 0.5
    assert ACTR_DEFAULTS["tau"] == -2.0
    assert ACTR_DEFAULTS["s"] == 0.5
    assert ACTR_DEFAULTS["S"] == 2.0
    assert ACTR_DEFAULTS["W"] == 1.0


# --------------------------------------------------------------------------- #
# config 配線(スチュワード実装: simulation.py memory.actr → agent.mem.actr)
# --------------------------------------------------------------------------- #
def _wiring_sim(tmp_path, name: str, extra: list[str]):
    from society.config import load_config
    from society.engine.simulation import Simulation
    dotlist = ["run.n_agents=4", "run.n_steps=1", f"run.name={name}",
               "model.backend=mock"] + extra
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


def test_actr_config_wiring_off(tmp_path):
    """既定(memory.actr なし)は全エージェント mem.actr=None(バイト一致経路)。"""
    sim = _wiring_sim(tmp_path, "actr_off", [])
    assert all(a.mem.actr is None for a in sim.agents)


def test_actr_config_wiring_on(tmp_path):
    """memory.actr.enabled=true で全員に設定注入・個体別シード・override 反映。"""
    sim = _wiring_sim(tmp_path, "actr_on",
                      ["memory.actr.enabled=true", "memory.actr.tau=-1.5"])
    seeds = set()
    for a in sim.agents:
        assert a.mem.actr is not None
        assert abs(a.mem.actr["tau"] - (-1.5)) < 1e-9   # override 反映
        assert a.mem.actr["d"] == ACTR_DEFAULTS["d"]     # 未指定は既定
        assert "enabled" not in a.mem.actr               # 制御キーは持ち込まない
        seeds.add(a.mem.actr["seed"])
    assert len(seeds) == len(sim.agents)                 # 個体別シードが全て異なる
