"""日次ローテーション/presence 機構のテスト(W2 P3)。

正典: docs/plans/w2-execution-plan.md §4 P3 / docs/plans/persona-pool.md §5・§9。

検証:
  - OFF: pool.enabled=false で presence_change が1件も出ず sim._pool is None(ゴールデンは test_scenario)。
  - presence 単体(純関数): 同 (seed,day) 同集合・cap 遵守・層優先・resident 毎日・stochastic は日で変動。
  - dehydrate/hydrate 往復: 信念・記憶・所持金・関係が持続する。
  - ON mock: (a) 日境界で presence_change が出て入替がある (b) 再来街者の beliefs が保持される
             (c) 同 seed 2 回一致 (d) agent.id がペルソナ id(pid_to_int)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402
from society.rng import RngHub                          # noqa: E402
from society.world import pool as pool_mod              # noqa: E402
from society.world import presence as presence_mod      # noqa: E402
from society.world.presence import PresenceRec, present_for_day  # noqa: E402
from society.agents.agent import Agent                  # noqa: E402
from society.agents.memory import MemoryStore           # noqa: E402


# ============================================================ 小さなテスト用プール
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で数百〜千体の小プールを tmp に生成(実プール 736MB は触らない)。"""
    out = tmp_path_factory.mktemp("pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


# ============================================================ presence 純関数(単体)
def _recs(n_res=0, n_work=0, n_stoch=0, rate=0.5):
    recs = []
    for i in range(n_res):
        recs.append(PresenceRec(pid=f"R_{i:04d}", key="resident"))
    for i in range(n_work):
        recs.append(PresenceRec(pid=f"W_{i:04d}", key="workday_shift", work_days="mon-fri"))
    for i in range(n_stoch):
        recs.append(PresenceRec(pid=f"S_{i:04d}", key="stochastic", visit_rate=rate))
    return recs


def test_presence_deterministic_same_seed_day():
    """同 (seed, day) なら present 集合が完全一致(純関数=リプレイ再現)。"""
    recs = _recs(n_res=5, n_stoch=200, rate=0.5)
    a = present_for_day(recs, day=3, present_cap=999, hub=RngHub(42), weekday=1)
    b = present_for_day(recs, day=3, present_cap=999, hub=RngHub(42), weekday=1)
    assert a == b
    assert a == sorted(a)                                   # 決定論=ソート済み


def test_presence_cap_and_layer_priority():
    """cap を厳守し、溢れたら層優先(resident は落ちない)。"""
    recs = _recs(n_res=10, n_stoch=100, rate=1.0)           # stochastic は全員 eligible
    present = present_for_day(recs, day=0, present_cap=25, hub=RngHub(7), weekday=1)
    assert len(present) == 25                                # cap 遵守
    residents = [p for p in present if p.startswith("R_")]
    assert len(residents) == 10                              # resident(tier0)は全員維持
    assert len([p for p in present if p.startswith("S_")]) == 15   # 残り枠だけ stochastic


def test_presence_resident_every_day_stochastic_varies():
    """resident は毎日 present・stochastic は日により変動(回転の主層)。"""
    recs = _recs(n_res=8, n_stoch=300, rate=0.5)
    hub = RngHub(42)
    sets = [set(present_for_day(recs, day=d, present_cap=9999, hub=hub, weekday=1))
            for d in range(4)]
    res_ids = {f"R_{i:04d}" for i in range(8)}
    for s in sets:
        assert res_ids <= s                                 # resident は毎日全員 present
    stoch = [{p for p in s if p.startswith("S_")} for s in sets]
    assert stoch[0] != stoch[1] and stoch[1] != stoch[2]    # stochastic は日で変わる


def test_presence_workday_weekday_only():
    """workday_shift は平日のみ present(weekday>=5 は不在)。"""
    recs = _recs(n_res=2, n_work=20)
    wk = present_for_day(recs, day=0, present_cap=999, hub=RngHub(1), weekday=2)   # 水
    we = present_for_day(recs, day=0, present_cap=999, hub=RngHub(1), weekday=6)   # 日
    assert len([p for p in wk if p.startswith("W_")]) == 20
    assert len([p for p in we if p.startswith("W_")]) == 0


# ============================================================ dehydrate / hydrate 往復
def test_dehydrate_hydrate_roundtrip():
    """スリム状態の往復で信念・記憶・所持金・関係が保持される。"""
    a = Agent(id=1, name="甲", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    a.beliefs = ["渋谷は落ち着く"]
    a.money = 12345.0
    a.mem.day_summaries = ["いい一日だった"]
    a.mem.observe(5, "スクランブルを歩いた")
    a.mem.consolidate(5, "散歩した", [("スクランブルを歩いた", 6.0)])
    a.mem.record_contact(99, "乙", 5, "はじめまして")
    a.mem.record_contact(99, "乙", 6, "また会った")
    state = pool_mod.dehydrate(a)

    b = Agent(id=2, name="甲", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    pool_mod.hydrate(b, state)
    assert b.beliefs == ["渋谷は落ち着く"]
    assert b.money == 12345.0
    assert "いい一日だった" in b.mem.day_summaries
    assert any("スクランブル" in e.text for e in b.mem.episodes)
    assert 99 in b.mem.relations and b.mem.relations[99]["count"] == 2


# ============================================================ id_of 密・int32 安全・衝突ゼロ
def test_id_of_dense_and_bounded(small_pool):
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    ids = [ps.id_of(r.pid) for r in recs]
    assert all(0 <= i < 2 ** 31 for i in ids)              # 観測 agent_id の int32 列に収まる
    assert len(set(ids)) == len(ids)                       # 密割当=衝突ゼロ(全 id ユニーク)
    assert ps.id_of(recs[0].pid) == ps.id_of(recs[0].pid)  # 安定(同一プールで不変)


# ============================================================ PoolStore 遅延読み
def test_poolstore_lazy_get(small_pool):
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    assert len(recs) > 300
    keys = Counter(r.key for r in recs)
    assert keys["resident"] > 0 and keys["stochastic"] > 0
    # full record は id 指定で1件だけ遅延読みできる(全読みしない)
    pid = recs[0].pid
    full = ps.get(pid)
    assert full["id"] == pid and "persona" in full


# ============================================================ ON mock 統合
def _pool_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _presence_sets(small_pool, cap=400):
    """day0/day1 の present 集合を純関数で先読み(テストで入替対象を選ぶ)。"""
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    hub = RngHub(42)
    s0 = set(present_for_day(recs, 0, cap, hub, 0 % 7))
    s1 = set(present_for_day(recs, 1, cap, hub, 1 % 7))
    return s0, s1


def test_on_s6a_budget_tracks_present(small_pool, tmp_path):
    """S6a: pool ON + n_proportional なら LLM 予算 cap = ceil(density × 在場数)(N=len(sim.agents))。"""
    import math as _m
    cfg = _pool_cfg("s6a", small_pool, n_steps=1, cap=400,
                    **{"lod.n_proportional.enabled": "true",
                       "lod.n_proportional.density": "0.15"})
    sim = Simulation(cfg, out_dir=tmp_path / "s6a")            # __init__ が day0 着席 + 予算更新
    assert sim.agents
    assert sim.budget.max_per_step == max(1, _m.ceil(0.15 * len(sim.agents)))
    assert sim.budget.max_per_step != 300                     # 固定 lod.max_llm_per_step ではない


def test_off_no_presence_change(tmp_path):
    """OFF(既定)は pool 経路を1本も通さない(sim._pool None・presence_change 0 件)。"""
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=6",
           "run.name=off", "model.backend=mock"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "off")
    sim.run()
    assert sim._pool is None
    assert not [e for e in sim.logger.events if e.kind == "presence_change"]


def test_on_rotation_and_ids(small_pool, tmp_path):
    """ON: 日境界で入替(presence_change)・cap 遵守・agent.id=ペルソナ id・在場>0。"""
    sim = Simulation(_pool_cfg("on", small_pool, n_steps=210), out_dir=tmp_path / "on")
    sim.run()
    pchanges = [e for e in sim.logger.events if e.kind == "presence_change"]
    assert pchanges, "presence_change が1件も出ていない(日境界の入替が発火していない)"
    assert any(e.payload["n_enter"] > 0 and e.payload["n_exit"] > 0 for e in pchanges), \
        "入替(enter/exit)が発生していない"
    for e in pchanges:
        assert e.agent_id == -1                             # 世界イベント
        assert e.payload["n_present"] <= 400                # cap 遵守
    # agent.id はペルソナ id 安定(密割当・pool.id_of)
    assert sim.agents
    for a in sim.agents:
        assert a.id == sim._pool.id_of(a.pool_pid)


def test_on_determinism_same_seed(small_pool, tmp_path):
    """同 seed 2 回で presence_change 列と最終在場集合が完全一致(決定論)。"""
    s1 = Simulation(_pool_cfg("d1", small_pool, n_steps=210), out_dir=tmp_path / "d1")
    s1.run()
    s2 = Simulation(_pool_cfg("d2", small_pool, n_steps=210), out_dir=tmp_path / "d2")
    s2.run()
    p1 = [e.payload for e in s1.logger.events if e.kind == "presence_change"]
    p2 = [e.payload for e in s2.logger.events if e.kind == "presence_change"]
    assert p1 == p2
    assert sorted(a.pool_pid for a in s1.agents) == sorted(a.pool_pid for a in s2.agents)


def test_on_revisitor_memory_preserved(small_pool, tmp_path):
    """再来街者(day0 不在 → day1 present)の beliefs/記憶が hydrate で保持される。"""
    s0, s1 = _presence_sets(small_pool)
    entrants = sorted(s1 - s0)
    assert entrants, "day0 不在→day1 present の再来街候補が見つからない"
    x = entrants[0]

    sim = Simulation(_pool_cfg("mem", small_pool, n_steps=210), out_dir=tmp_path / "mem")
    # 事前にドーマント退避ストアへ x のスリム状態(識別可能な belief)を仕込む
    assert x not in {a.pool_pid for a in sim.agents}        # x は day0 不在
    sim._dormant.save(x, {"beliefs": ["__memtest__"], "money": 777.0})
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    assert x in live, "再来街者 x が day1 に present になっていない"
    assert "__memtest__" in live[x].beliefs                 # 記憶(信念)が復元された
    assert live[x].money == 777.0                            # 所持金も持続


def test_on_resume_byte_matches_straight(small_pool, tmp_path):
    """pool ON でも「一気 vs 中断→resume」の l1_events が完全一致(P3 検収基準)。

    dormant 退避ストア + 退場者参照 + 日境界進行を sidecar で復元することで、ローテーションを
    2 回跨いでも(boundary=step102/246)straight run と byte 級一致する。"""
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    st = tmp_path / "st"
    Simulation(_pool_cfg("rst", small_pool, n_steps=300), out_dir=st).run()

    rs = tmp_path / "rs"
    s1 = Simulation(_pool_cfg("rrs", small_pool, n_steps=150,
                              **{"observer.checkpoint_every": 150}), out_dir=rs)
    for step in range(150):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 150, rs / "checkpoint" / "ckpt-000150.pkl.gz")
    s1._save_pool_sidecar(150)
    s1.logger.flush_segment()
    s2 = Simulation(_pool_cfg("rrs", small_pool, n_steps=300,
                              **{"observer.checkpoint_every": 150}), out_dir=rs)
    s2.run(resume_from=rs)
    assert rows(st) == rows(rs), "pool ON の resume が straight と byte 不一致"


def test_on_dehydrate_saves_departed(small_pool, tmp_path):
    """退場者(day0 present → day1 不在)は dehydrate されドーマントストアに残る(記憶保持)。"""
    s0, s1 = _presence_sets(small_pool)
    leavers = sorted(s0 - s1)
    assert leavers, "day0 present→day1 不在の退場候補が見つからない"
    y = leavers[0]

    sim = Simulation(_pool_cfg("dehy", small_pool, n_steps=210), out_dir=tmp_path / "dehy")
    ya = next(a for a in sim.agents if a.pool_pid == y)      # day0 の live 実体
    ya.beliefs.append("__d0mark__")
    sim.run()
    assert y not in {a.pool_pid for a in sim.agents}         # day1 は不在
    assert y in sim._dormant                                 # ドーマントに退避済み
    assert "__d0mark__" in sim._dormant.peek(y)["beliefs"]   # 退場時の記憶が保存された


# ============================================================ M-3: 退避台帳の幅 × dunbar 休眠
# 第75バッチ実測の持ち越し: dunbar(認知枠)で**休眠**した関係は closeness=0 に退避され、
# 接触も止まるので count も伸びない。ところが pool の dehydrate は「接触回数の多い上位
# rel_cap 件」で台帳を切るため、休眠した弱い紐帯は**再会する前に退場で消える**。
# 第86バッチ保守 M-3 で conf キー pool.relations_cap / pool.episodes_cap を新設した
# (既定 = 現行値 20 / 30 = 挙動不変。観察ランでのみ広げる)。
def _rel(count: int, closeness: float, name: str) -> dict:
    return {"name": name, "count": count, "last_step": 0, "last": "",
            "closeness": closeness, "tier": 1}


def _agent_with_ledger(agent, n_active: int, n_weak: int) -> list:
    """活性 n_active 件(接触多)+ 弱い紐帯 n_weak 件(接触 1 回)を仕込む。返り値=弱い側の id。"""
    agent.mem.relations = {}
    for i in range(n_active):
        agent.mem.relations[1000 + i] = _rel(50 + i, 8.0, f"active{i}")
    weak = []
    for i in range(n_weak):
        oid = 2000 + i
        agent.mem.relations[oid] = _rel(1, 1.0, f"weak{i}")
        weak.append(oid)
    return weak


def test_m3_default_relations_cap_is_unchanged():
    """既定は従来どおり 20 件切り(configure を呼ばなければ素値のまま)。"""
    pool_mod.configure(rel_cap=pool_mod._REL_CAP, ep_cap=pool_mod._EP_CAP)
    a = Agent(id=1, name="甲", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    weak = _agent_with_ledger(a, n_active=20, n_weak=5)
    st = pool_mod.dehydrate(a)
    assert len(st["relations"]) == 20
    assert not (set(weak) & set(st["relations"])), "弱い紐帯が既定で残ってしまっている"
    assert pool_mod.caps()["rel"] == 20


def test_m3_widened_cap_keeps_the_weak_ties():
    """configure(rel_cap=…) を広げると弱い紐帯が退避台帳に残る(既定へ必ず戻す)。"""
    a = Agent(id=1, name="甲", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    weak = _agent_with_ledger(a, n_active=20, n_weak=5)
    try:
        pool_mod.configure(rel_cap=40)
        st = pool_mod.dehydrate(a)
    finally:
        pool_mod.configure(rel_cap=pool_mod._REL_CAP)
    assert len(st["relations"]) == 25
    assert set(weak) <= set(st["relations"])
    b = Agent(id=2, name="乙", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    pool_mod.hydrate(b, st)
    assert set(weak) <= set(b.mem.relations)          # 往復しても残る


def test_m3_config_keys_are_ints_and_default_to_current_values():
    cfg = load_config([])
    assert int(cfg.pool.relations_cap) == 20 == pool_mod._REL_CAP
    assert int(cfg.pool.episodes_cap) == 30 == pool_mod._EP_CAP
    over = load_config(["pool.relations_cap=60", "pool.episodes_cap=80"])
    assert isinstance(over.pool.relations_cap, int) and over.pool.relations_cap == 60
    assert isinstance(over.pool.episodes_cap, int) and over.pool.episodes_cap == 80


def test_m3_dunbar_dormant_ties_survive_rotation_only_when_widened(small_pool, tmp_path):
    """★相互作用(dunbar ON + pool ON): 休眠した紐帯は既定 20 件切りで退場時に消え、
    pool.relations_cap を広げると退避台帳に残って**再会の可能性が保たれる**。

    日境界のローテーションは `_phase_pool_rotation` を直接呼んで起こす(210 step 走らせる
    必要はない)。退場者 y の台帳に「活性 20 件(接触多)+ 休眠 5 件(接触 1 回)」を仕込み、
    休眠は dunbar の正規の遷移(_make_dormant)で作る = closeness 0 / dormant フラグ付き。
    """
    from society import dunbar as dunbar_mod
    from society.engine import scheduler

    s0, s1 = _presence_sets(small_pool)
    y = sorted(s0 - s1)[0]                            # day0 present → day1 不在

    def rotate(name: str, **ov) -> dict:
        cfg = _pool_cfg(name, small_pool, n_steps=1,
                        **{"relations.enabled": "true",
                           "relations.dunbar.enabled": "true", **ov})
        sim = Simulation(cfg, out_dir=tmp_path / name)
        assert dunbar_mod.enabled(sim), "dunbar が ON になっていない"
        ya = next(a for a in sim.agents if a.pool_pid == y)
        weak = _agent_with_ledger(ya, n_active=20, n_weak=5)
        for oid in weak:                              # 正規の遷移で休眠にする
            dunbar_mod._make_dormant(sim, ya, oid, step=1, sim_min=10)
        assert dunbar_mod.dormant_ids(ya) == weak
        scheduler._phase_pool_rotation(sim, 144, 1440)   # 日境界(day1)を起こす
        assert y not in {a.pool_pid for a in sim.agents}
        saved = sim._dormant.peek(y)
        assert saved is not None, "退場者が退避されていない"
        return {"weak": weak, "relations": saved["relations"]}

    narrow = rotate("m3n")                                   # 既定 20 件
    assert len(narrow["relations"]) == 20
    assert not (set(narrow["weak"]) & set(narrow["relations"])), \
        "既定で休眠紐帯が残っている(この持ち越しの前提が崩れた)"

    try:
        wide = rotate("m3w", **{"pool.relations_cap": 40})   # 広げたラン
    finally:
        pool_mod.configure(rel_cap=pool_mod._REL_CAP, ep_cap=pool_mod._EP_CAP)
    assert len(wide["relations"]) == 25
    assert set(wide["weak"]) <= set(wide["relations"])

    # 再来街(hydrate)したときに休眠のまま残っている = 再会の余地が保たれている
    b = Agent(id=7, name="乙", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    pool_mod.hydrate(b, {"relations": wide["relations"]})
    assert dunbar_mod.dormant_ids(b) == wide["weak"]
