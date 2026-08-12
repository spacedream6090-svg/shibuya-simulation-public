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
    # 事前にドーマント退避ストアへ**全再来街候補**のスリム状態(識別可能な belief)を仕込む。
    # ★1 人(entrants[0])だけを仕込む形だと、その 1 人がたまたま日跨ぎのあとに買い物を
    #   すると所持金の検査が落ちる(実測: 復元された 777 円で 1800 円の夜遊びを試みて
    #   残高 0 になる個体が entrants[0] に来ることがある)。復元の検査と「復元後は普通に
    #   暮らす」ことは別問題なので、**取引をしなかった個体**で所持金を固定する。
    #   名簿(プール)に役割が 1 つ増えるだけで entrants[0] は入れ替わるので、
    #   特定の 1 個体に依存しない形にしておく。
    assert x not in {a.pool_pid for a in sim.agents}        # x は day0 不在
    for pid in entrants:
        sim._dormant.save(pid, {"beliefs": ["__memtest__"], "money": 777.0})
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    assert x in live, "再来街者 x が day1 に present になっていない"
    assert "__memtest__" in live[x].beliefs                 # 記憶(信念)が復元された
    revisitors = [live[pid] for pid in entrants if pid in live]
    assert revisitors, "day1 に戻った再来街者が 1 人も居ない(検査が空回り)"
    for a in revisitors:
        assert "__memtest__" in a.beliefs                   # 全員の記憶が復元された
    moved = {e.agent_id for e in sim.logger.events
             if e.kind in ("spend", "wage", "chance_event")}
    quiet = [a for a in revisitors if a.id not in moved]
    assert quiet, "取引をしなかった再来街者が 1 人も居ない(所持金の検査が空回り)"
    for a in quiet:
        assert a.money == 777.0                             # 所持金も持続


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


# ============================================================ pool 経路の org 配属(第109 レーン ORG)
_ORGS_BOOK = "data/organizations_shibuya_wide11k.json"     # small_pool fixture の母体と同一


def test_pool_route_attaches_org_from_record(small_pool, tmp_path):
    """★第108 縦煙「organizations.attach 0 件」の解消: pool record の org_id で配属される。

    organizations.attach は「名簿ファイル → 配属ファイル」でしか引けず、pool ラン
    (agents.personas_file=null)では構造的に 0 件だった。プール record は台帳から逆算して
    作られていて org_id/role を持つので、build_pool_agent がそれで配属する。"""
    sim = Simulation(_pool_cfg("orgon", small_pool, n_steps=1,
                               **{"organizations.enabled": "true",
                                  f"organizations.book": _ORGS_BOOK}),
                     out_dir=tmp_path / "orgon")
    attached = [a for a in sim.agents if getattr(a, "org_id", None)]
    assert attached, "pool 経路で org_id が 1 件も付いていない(第108 の 0 件が残っている)"
    assert sim._pool_org_stat["n_attached"] == len(attached)
    for a in attached:                                     # attach と同じ 3 属性が揃う
        assert a.org_id in sim.orgs
        assert isinstance(getattr(a, "org_role", None), str)
        assert getattr(a, "org_line", "")                  # プロンプトの所属行(falsy=注入なし)
    # 配属された org は実在の複数社(n_orgs=3 のような縮退でない)
    assert len({a.org_id for a in attached}) > 1


def test_pool_org_attach_is_off_when_organizations_off(small_pool, tmp_path):
    """organizations OFF(既定)では 1 件も付かない = 従来と完全同一。"""
    sim = Simulation(_pool_cfg("orgoff", small_pool, n_steps=1),
                     out_dir=tmp_path / "orgoff")
    assert sim._pool_org_book is None
    assert not [a for a in sim.agents if getattr(a, "org_id", None)]


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


# ============================================================ DP-U3 案A: 層別クォータ
# 正典: docs/plans/proposal-dp-u3-observe-250k.md §1.2「案 A の中身」/ §5.1 R-1。
# 第91 の破綻: 層優先は「優先順に埋めて途中 break」なので、上位層(workday_shift)が cap を
# 食い切ると来街層(cadence / stochastic)が **構造的に全滅**する(cap 25万・平日で 0 人)。
# conf `pool.tier_quota.enabled`(既定 false)で cap を層別クォータへ分配する。追加乱数ゼロ。
def _legacy_present(recs, day: int, cap: int, hub, weekday: int) -> list[str]:
    """DP-U3 以前の `present_for_day` 本体(層優先 break)の**逐語コピー**。

    既定 OFF が現行と 1 バイトも変わらないことを、この参照実装との突合で固定する。
    """
    tiers = [[] for _ in range(presence_mod._N_TIERS)]
    for rec in recs:
        if presence_mod._eligible(rec, day, weekday, hub):
            tiers[presence_mod._TIER.get(rec.key, presence_mod._N_TIERS - 1)].append(rec.pid)
    present: list[str] = []
    for tier in range(presence_mod._N_TIERS):
        ids = tiers[tier]
        if not ids:
            continue
        room = cap - len(present)
        if room <= 0:
            break
        if len(ids) <= room:
            present.extend(ids)
        else:
            ranked = sorted(
                ids,
                key=lambda pid: (float(hub.stream("presence_rank", pid, int(day)).random()), pid))
            present.extend(ranked[:room])
            break
    present.sort()
    return present


def _tier_recs(resident=0, duty=0, work=0, cadence=0, stochastic=0):
    """5 層すべてを含む合成名簿(当日資格は平日なら全員 True = 比率が読みやすい)。"""
    recs = []
    for i in range(resident):
        recs.append(PresenceRec(pid=f"RES_{i:06d}", key="resident"))
    for i in range(duty):
        recs.append(PresenceRec(pid=f"DUT_{i:06d}", key="duty", duty_days="all"))
    for i in range(work):
        recs.append(PresenceRec(pid=f"WRK_{i:06d}", key="workday_shift", work_days="mon-fri"))
    for i in range(cadence):
        recs.append(PresenceRec(pid=f"CAD_{i:06d}", key="cadence", cadence="school_day"))
    for i in range(stochastic):
        recs.append(PresenceRec(pid=f"STO_{i:06d}", key="stochastic", visit_rate=1.0))
    return recs


def _by_tier(pids) -> dict:
    return Counter(p.split("_", 1)[0] for p in pids)


# 第91 の実測構成(名簿 100万・平日資格者)を 1/1000 に縮めた再現ケース。
# 提案書 §1.1 の表: resident 30,034 / duty 986 / workday 253,702 / cadence 27,357 / stoch 24,883。
_B91 = dict(resident=30, duty=1, work=254, cadence=27, stochastic=25)   # 計 337 人(= 33.7万)
_B91_CAP = 250                                                          # = 在場 25万


# ---------------------------------------------------------------- ① OFF = 現行完全一致
@pytest.mark.parametrize("cap", [10, 25, 100, 250, 337, 1000])
@pytest.mark.parametrize("weekday", [1, 6])
def test_quota_off_matches_legacy_exactly(cap, weekday):
    """既定 OFF は DP-U3 以前の層優先 break と**選抜集合も順序も**完全一致(全 cap 帯)。"""
    recs = _tier_recs(**_B91)
    for day in range(3):
        hub_a, hub_b = RngHub(42), RngHub(42)
        assert present_for_day(recs, day, cap, hub_a, weekday) == \
            _legacy_present(recs, day, cap, hub_b, weekday)


def test_quota_off_matches_legacy_on_real_pool(small_pool):
    """実 P5 プール(小)でも既定 OFF は逐語コピーの参照実装と一致する。"""
    recs = pool_mod.PoolStore(small_pool).presence_records()
    for cap in (50, 300, 100000):
        assert present_for_day(recs, 0, cap, RngHub(42), 0) == \
            _legacy_present(recs, 0, cap, RngHub(42), 0)
    # 明示 False も既定(引数省略)と同一
    assert present_for_day(recs, 1, 300, RngHub(42), 1, False) == \
        present_for_day(recs, 1, 300, RngHub(42), 1)


def test_quota_off_default_is_false_in_conf():
    cfg = load_config([])
    assert cfg.pool.tier_quota.enabled is False
    on = load_config(["pool.tier_quota.enabled=true"])
    assert on.pool.tier_quota.enabled is True


def test_quota_toggle_is_declared_in_registry():
    """新トグルは機能レジストリに宣言済み(未宣言検出 CI ゲートに落ちない)。"""
    from society import registry as R
    ids = {f.id for f in R.FEATURES}
    assert "pool.tier_quota.enabled" in ids
    assert R.undeclared_toggles(load_config()) == []


# ---------------------------------------------------------------- ② ON = 構成比保存(第91 の解消)
def test_quota_on_keeps_visitors_that_legacy_wipes_out():
    """★第91 破綻ケース: OFF では来街層が 0 人、ON では quota どおり残る(構成比保存)。"""
    recs = _tier_recs(**_B91)
    off = _by_tier(present_for_day(recs, 0, _B91_CAP, RngHub(42), 1))
    on = _by_tier(present_for_day(recs, 0, _B91_CAP, RngHub(42), 1, True))

    # OFF: resident 30 + duty 1 で残 219 を workday(254)が埋め切り break = 来街者ゼロ
    assert off["CAD"] == 0 and off["STO"] == 0
    assert off["WRK"] == 219 and off["RES"] == 30 and off["DUT"] == 1

    # ON: 提案書 §1.2 の割当(resident 22,282 / duty 731 / workday 188,228 /
    #     cadence 20,297 / stochastic 18,461)を 1/1000 に縮めた値と一致する
    assert on == Counter({"WRK": 188, "RES": 22, "CAD": 20, "STO": 19, "DUT": 1})
    assert sum(on.values()) == _B91_CAP
    assert on["CAD"] + on["STO"] == 39          # 来街者が 3.9万人ぶん入る(現行 = 0 人)

    # 構成比が資格者比率に保存されている(最大剰余法の丸め以内 = 1 人)
    total = sum(_B91.values())
    for tag, n_elig in (("RES", 30), ("DUT", 1), ("WRK", 254), ("CAD", 27), ("STO", 25)):
        assert abs(on[tag] - _B91_CAP * n_elig / total) <= 1.0


def test_quota_on_overflow_tiers_use_the_existing_daily_ranking():
    """溢れた層の切り方は**既存の当日ランキングそのまま**(新しい順序を発明していない)。"""
    recs = _tier_recs(**_B91)
    hub = RngHub(42)
    got = set(present_for_day(recs, 0, _B91_CAP, hub, 1, True))
    quota = presence_mod.quota_by_ratio([30, 1, 254, 27, 25], _B91_CAP)
    for tag, prefix, tier in (("RES", "RES_", 0), ("WRK", "WRK_", 2),
                              ("CAD", "CAD_", 3), ("STO", "STO_", 4)):
        ids = [r.pid for r in recs if r.pid.startswith(prefix)]
        expect = presence_mod._ranked(ids, 0, hub)[:quota[tier]]
        assert {p for p in got if p.startswith(prefix)} == set(expect), tag


def test_quota_on_weekend_falls_back_to_eligible_total():
    """週末は workday_shift が失格 → クォータは残層に配られるが**総数は資格者数が上限**。

    (提案書 §1.2 の正直な注記: 週末の人口崩れは名簿の較正問題でクォータでは直らない)
    """
    recs = _tier_recs(**_B91)
    on = _by_tier(present_for_day(recs, 0, _B91_CAP, RngHub(42), 6, True))
    assert on["WRK"] == 0
    assert sum(on.values()) == 30 + 1 + 25         # cadence(school_day)も週末は失格
    assert on["STO"] == 25                          # 資格者は全員入る(捏造しない)


# ---------------------------------------------------------------- ③ quota 合計 = cap
@pytest.mark.parametrize("cap", [0, 1, 2, 5, 37, 100, 249, 250, 336, 337, 338, 10**6])
def test_quota_sums_to_cap_or_eligible_total(cap):
    """端数正規化: Σ quota = min(cap, Σ 資格者)。各層は資格者数を超えない。"""
    sizes = [30, 1, 254, 27, 25]
    q = presence_mod.quota_by_ratio(sizes, cap)
    assert sum(q) == min(cap, sum(sizes))
    assert all(0 <= q[t] <= sizes[t] for t in range(len(sizes)))


@pytest.mark.parametrize("sizes", [
    [0, 0, 0, 0, 0],                 # 全層で資格者ゼロ
    [7, 0, 0, 0, 0],                 # 1 層だけ
    [1, 1, 1, 1, 1],                 # 完全同数(タイブレークが全部効く)
    [3, 3, 3, 0, 1],
    [999999, 1, 1, 1, 1],            # 極端な偏り
    [30034, 986, 253702, 27357, 24883],   # 提案書 §1.1 の実測(平日)
])
@pytest.mark.parametrize("cap", [0, 1, 3, 4, 5, 999, 250000, 10**7])
def test_quota_invariants_over_many_shapes(sizes, cap):
    q = presence_mod.quota_by_ratio(sizes, cap)
    assert sum(q) == min(cap, sum(sizes)), (sizes, cap, q)
    assert all(0 <= q[t] <= sizes[t] for t in range(len(sizes))), (sizes, cap, q)


def test_quota_real_scale_matches_the_proposal_table():
    """提案書 §1.2 の割当(cap 25万・平日実測比)を実装が再現する(端数処理の差は ±1 人)。

    提案書の手計算は素の round(合計 249,999 = 1 人足りない)。実装は最大剰余法なので
    合計が厳密に cap になり、その 3 人ぶんが上位剰余の層(resident/duty/cadence)へ入る。
    """
    q = presence_mod.quota_by_ratio([30034, 986, 253702, 27357, 24883], 250000)
    assert sum(q) == 250000
    assert q == [22283, 732, 188227, 20297, 18461]
    for got, doc in zip(q, [22282, 731, 188228, 20297, 18461]):     # 提案書 §1.2 の表
        assert abs(got - doc) <= 1
    assert q[3] + q[4] == 38758                        # 来街者(現行 = 0 人)= 約 3.9 万人


def test_quota_tie_break_is_the_fixed_tier_order():
    """剰余が同値なら層の固定順(resident>duty>workday>cadence>stochastic)で決める。"""
    # 5 層同数 × cap=7 → base=1 ずつ・余り 2 を上位 2 層へ
    assert presence_mod.quota_by_ratio([10, 10, 10, 10, 10], 7) == [2, 2, 1, 1, 1]
    assert presence_mod.quota_by_ratio([10, 10, 10, 10, 10], 6) == [2, 1, 1, 1, 1]


# ---------------------------------------------------------------- ④ 決定論
def test_quota_on_is_deterministic():
    """同入力 2 走で同一選抜(乱数追加ゼロ・整数演算の端数処理)。"""
    recs = _tier_recs(**_B91)
    a = present_for_day(recs, 5, _B91_CAP, RngHub(42), 3, True)
    b = present_for_day(recs, 5, _B91_CAP, RngHub(42), 3, True)
    assert a == b and a == sorted(a)
    # quota 側も純関数(同入力同出力)
    assert presence_mod.quota_by_ratio([30, 1, 254, 27, 25], 250) == \
        presence_mod.quota_by_ratio([30, 1, 254, 27, 25], 250)
    # ON は日ごとに顔ぶれが動く(ローテーションは失われていない)
    d0 = set(present_for_day(recs, 0, _B91_CAP, RngHub(42), 1, True))
    d1 = set(present_for_day(recs, 1, _B91_CAP, RngHub(42), 1, True))
    assert d0 != d1


def test_quota_on_adds_no_new_random_draws():
    """ON でも引く stream は既存の presence 系のみ(新種の乱数キーを増やしていない)。"""
    seen: list[str] = []

    class _Spy(RngHub):
        def stream(self, *key):
            seen.append(str(key[0]))
            return super().stream(*key)

    present_for_day(_tier_recs(**_B91), 0, _B91_CAP, _Spy(42), 1, True)
    assert set(seen) <= {"presence", "presence_rank", "presence_cadence", "presence_revisit"}


# ---------------------------------------------------------------- ⑤ 資格者 < quota の再配分
def test_quota_short_tier_takes_all_and_rest_is_redistributed():
    """資格者数が取り分に満たない層は**全員だけ**入り、余りは残層へ回る(枠を捨てない)。"""
    # cap > Σ 資格者: 全層まるごと(存在しない人を捏造しない)
    assert presence_mod.quota_by_ratio([10, 2, 3, 0, 5], 100) == [10, 2, 3, 0, 5]
    # 資格者ゼロの層には 1 人も配らず、その枠は資格者のいる層へ回る
    q = presence_mod.quota_by_ratio([5, 0, 0, 0, 95], 50)
    assert q[1] == q[2] == q[3] == 0 and sum(q) == 50
    # 小さすぎて比例配分では 0 になる層も、cap が資格者総数を超えれば全員入る
    q2 = presence_mod.quota_by_ratio([1, 1, 100000, 1, 1], 100004)
    assert q2 == [1, 1, 100000, 1, 1]


def test_quota_short_tier_end_to_end():
    """名簿側から見た再配分: 取り分に届かない小さな層は**全員 present**・切られるのは過多の層だけ。"""
    recs = _tier_recs(resident=3, duty=2, work=40, cadence=1, stochastic=4)   # 計 50
    got = _by_tier(present_for_day(recs, 0, 45, RngHub(42), 1, True))
    assert got["RES"] == 3 and got["DUT"] == 2 and got["CAD"] == 1            # 全員入る
    assert got["WRK"] == 36 and got["STO"] == 3                               # 過多の層だけ切られる
    assert sum(got.values()) == 45                                            # 枠の取りこぼしゼロ


# ---------------------------------------------------------------- ON 統合(mock・実プール小)
_KEYS = ("resident", "duty", "workday_shift", "cadence", "stochastic")


def test_quota_on_wiring_and_composition(small_pool, tmp_path):
    """conf → Simulation → presence の配線と、実プールでの構成比保存(day0=月曜)。

    cap は「上位 3 層(resident+duty+workday_shift)の当日資格者数ちょうど」に取る。
    層優先ではここで cap 到達 → break = **来街層が 1 人も入らない**(第91 の破綻の縮小再現)。
    """
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    key_of = {r.pid: r.key for r in recs}
    elig = Counter(r.key for r in recs if presence_mod._eligible(r, 0, 0, RngHub(42)))
    sizes = [elig.get(k, 0) for k in _KEYS]
    cap = sizes[0] + sizes[1] + sizes[2]
    assert sizes[2] > 0 and sizes[3] + sizes[4] > 0, "プールの層構成が前提を満たしていない"

    off = Simulation(_pool_cfg("tq_off", small_pool, n_steps=1, cap=cap),
                     out_dir=tmp_path / "tq_off")
    on = Simulation(_pool_cfg("tq_on", small_pool, n_steps=1, cap=cap,
                              **{"pool.tier_quota.enabled": "true"}),
                    out_dir=tmp_path / "tq_on")
    assert off._pool_tier_quota is False and on._pool_tier_quota is True

    c_off = Counter(key_of[a.pool_pid] for a in off.agents)
    c_on = Counter(key_of[a.pool_pid] for a in on.agents)
    assert len(off.agents) == len(on.agents) == cap           # どちらも cap 遵守
    assert c_off["cadence"] == 0 and c_off["stochastic"] == 0  # 層優先では来街層が全滅
    assert c_on["cadence"] > 0 and c_on["stochastic"] > 0      # クォータでは残る
    assert c_on["workday_shift"] < c_off["workday_shift"]      # 溢れた層が譲る
    # ON の割当は当日資格者の比率(day0=月曜)にぴったり一致する
    quota = presence_mod.quota_by_ratio(sizes, cap)
    for i, k in enumerate(_KEYS):
        assert c_on.get(k, 0) == quota[i], k


def test_quota_on_rotates_at_the_day_boundary(small_pool, tmp_path):
    """ON でも日境界のローテーションが従来どおり起き、cap と層構成が保たれる。

    150 step 走らせる必要はないので M-3 テストと同じく `_phase_pool_rotation` を直接呼ぶ。
    """
    from society.engine import scheduler
    cap = 120
    sim = Simulation(_pool_cfg("tq_rot", small_pool, n_steps=1, cap=cap,
                               **{"pool.tier_quota.enabled": "true"}),
                     out_dir=tmp_path / "tq_rot")
    d0 = {a.pool_pid for a in sim.agents}
    scheduler._phase_pool_rotation(sim, 144, 1440)          # day1(火曜)を起こす
    d1 = {a.pool_pid for a in sim.agents}
    assert len(d1) == cap and d0 != d1                       # cap 遵守 + 顔ぶれの入替
    ev = [e for e in sim.logger.events if e.kind == "presence_change"]
    assert ev and ev[-1].payload["n_present"] == len(d1)
    key_of = {r.pid: r.key for r in sim._pool.presence_records()}
    assert Counter(key_of[p] for p in d1)["stochastic"] > 0  # 来街層が翌日も残る


# ============================================================ 第98バッチ 小粒A: 退避の残課題 2 件
# 台帳 PENDING の「IF-C 残② `_rumors` を運ばない」「竹-4 残③ `_phys_*` を運ばない」を塞いだ
# ことの機械固定。★受入条件の芯は「持たない個体の退避 dict が現行と完全同一」であること。
def _bare_agent(aid: int = 11):
    return Agent(id=aid, name="丙", age=30, occupation="会社員", persona="p",
                 traits={}, states={}, mem=MemoryStore())


def test_rumor_key_matches_rumors_module():
    """pool の `_RUMOR_KEY` は society.rumors.RUMOR_KEY と同一(層の逆流を避けた写しの固定)。"""
    from society import rumors as rumors_mod
    assert pool_mod._RUMOR_KEY == rumors_mod.RUMOR_KEY


def test_dehydrate_without_rumors_or_phys_is_unchanged():
    """rumors / physics を持たない個体の退避 dict に新キーが 1 つも生えない(既定ラン不変)。

    「属性が在って非空のときだけ入れる」設計の受入条件そのもの。空 dict / 空 body でも
    キーを作らない(OFF ランと ON だが未経験のランで退避のバイト列が割れないこと)。
    """
    a = _bare_agent()
    base = pool_mod.dehydrate(a)
    assert "rumors" not in base and "phys_body" not in base

    setattr(a, pool_mod._RUMOR_KEY, {})          # ON だが 1 件も知らない = 空 dict
    a._phys_body = None                          # 物理 ON だが 1 度も積分していない
    assert pool_mod.dehydrate(a) == base, "空の状態で退避 dict が変わっている"


def test_dehydrate_hydrate_carries_rumors_and_phys_body():
    """噂の知識(役割/冗長回数/知った step/源)と群集体感が往復で同値に戻る(挿入順も)。"""
    a = _bare_agent()
    setattr(a, pool_mod._RUMOR_KEY, {
        "rumor-2": {"role": "stifler", "redundant": 3, "step": 40, "src": "enforcement"},
        "rumor-1": {"role": "spreader", "redundant": 0, "step": 12, "src": "event_host"},
    })
    a._phys_body = {"blocked": 0.25, "contact": 1.5, "local_density": 0.75}
    state = pool_mod.dehydrate(a)
    assert list(state["rumors"]) == ["rumor-2", "rumor-1"], "知った順(挿入順)が壊れている"

    import json
    json.dumps(state)                            # JSON 安全(プリミティブ/dict のみ)

    b = _bare_agent(12)
    pool_mod.hydrate(b, state)
    assert getattr(b, pool_mod._RUMOR_KEY) == getattr(a, pool_mod._RUMOR_KEY)
    assert list(getattr(b, pool_mod._RUMOR_KEY)) == ["rumor-2", "rumor-1"]
    assert b._phys_body == a._phys_body
    # 退避辞書を書き換えても復元後の個体に波及しない(実体を共有しない)
    state["rumors"]["rumor-1"]["redundant"] = 99
    assert getattr(b, pool_mod._RUMOR_KEY)["rumor-1"]["redundant"] == 0


def test_hydrate_tolerates_old_state_without_new_keys():
    """旧 退避辞書(新キー無し)からの復元で属性を 1 つも生やさない(前方互換)。"""
    a = _bare_agent()
    old = pool_mod.dehydrate(a)                  # 新キーを持たない辞書
    b = _bare_agent(13)
    pool_mod.hydrate(b, old)
    assert not hasattr(b, pool_mod._RUMOR_KEY)
    assert not hasattr(b, "_phys_body")


def test_dehydrate_does_not_carry_zone_ownership():
    """ゾーン所有(走行レコード)は**意図的に運ばない**(再来街は別経路で組み直されるため)。"""
    a = _bare_agent()
    a._phys_zone = "scramble"
    a._phys_pos = (1.0, 2.0)
    a._phys_path = ["n1", "n2"]
    state = pool_mod.dehydrate(a)
    assert not [k for k in state if k.startswith("phys_") and k != "phys_body"]
    b = _bare_agent(14)
    pool_mod.hydrate(b, state)
    assert getattr(b, "_phys_zone", None) is None


# ================================================ レーン D1(第109): 退避フィールドの総点検
# 第107 の `sick` / `sick_until` 修正(= 回転のたびに病気を忘れていた)と**まったく同じ型**の
# 取りこぼしを、agent 属性の全数走査(AST)と pool の搬送リストの突合で洗い出して塞いだ。
# 判定基準は「① 個体固有 ② 日を跨いで持続 ③ 行動 or L1 の発火可否を変える」の 3 つ全部。
#
#   H2 医療     … 在院/搬送の印(med_*)。運ばないと入院中の個体が回転で即退院する。
#   H4 酩酊     … `_inc_intox_until`。当該 module が「正直な限界 4」として自己申告していた。
#   第61 悪評   … `_gossip_known` / `_gossip_heard`(`_rumors` と同型の「街を出ると忘れる」)。
#   第73 信念   … `_fact_beliefs`(`beliefs` は運ぶのに fact 信念だけ落ちていた)。
#   評判/所持   … `_reputation`(合成地位 T6 の 1 項)/ `_goods_belongings`(有界リスト)。
#   H1 の残り   … `sev_collapse_step` / `chronic_days` / `withdrawn` / `fatigue`
#                 (health._SLIM_FIELDS へ追加 = pool 側の写しと同期)。
def test_med_field_list_mirrors_medical_clear():
    """`_MED_FIELDS` が `medical._clear` の**全数と既定値**に一致する(写しの機械固定)。"""
    from society import medical as medical_mod
    a = _bare_agent(21)
    medical_mod._clear(a)                       # 正典が「印を落とす」= 既定値そのもの
    expect = {k: v for k, v in vars(a).items() if k.startswith("med_")}
    got = {name: default for name, _cast, default in pool_mod._MED_FIELDS}
    assert got == expect, "pool の med 写しが medical._clear と食い違っている"


def test_intox_key_matches_incidents_module():
    """`_INTOX_KEY` は society.incidents_interpersonal.INTOX_KEY と同一(写しの固定)。"""
    from society import incidents_interpersonal as inc_mod
    assert pool_mod._INTOX_KEY == inc_mod.INTOX_KEY


def test_dehydrate_without_the_d1_fields_is_unchanged():
    """D1 で足した族を 1 つも持たない個体の退避 dict に新キーが生えない(既定ラン不変)。

    「属性が在って**非既定**のときだけ入れる」設計の受入条件そのもの。空 dict / 空リスト /
    0 でもキーを作らないこと(機構 ON だが未経験のランで退避のバイト列が割れないこと)。
    """
    a = _bare_agent()
    base = pool_mod.dehydrate(a)
    assert not (set(base) & {"med", "intox_until", "gossip_known", "gossip_heard",
                             "fact_beliefs", "reputation", "belongings",
                             "controllability"})

    from society import medical as medical_mod
    medical_mod._clear(a)                       # 医療 ON だが 1 度も搬送されていない
    setattr(a, pool_mod._INTOX_KEY, 0)          # 酩酊 ON だが素面
    a._gossip_known, a._gossip_heard = {}, {}   # 悪評 ON だが何も知らない
    a._fact_beliefs = {}
    a._reputation = 0.0
    a._goods_belongings = []
    assert pool_mod.dehydrate(a) == base, "既定値だけの状態で退避 dict が変わっている"


def test_dehydrate_hydrate_carries_the_d1_fields():
    """D1 で足した族が往復で同値に戻る(JSON 安全・実体非共有つき)。"""
    import json

    a = _bare_agent()
    a.med_admitted, a.med_until, a.med_since = True, 900, 700
    a.med_node, a.med_poi, a.med_confirmed = "n42", "病院", 3
    setattr(a, pool_mod._INTOX_KEY, 123)
    a._gossip_known = {5: 2, 9: 3}
    a._gossip_heard = {7: {1, 4, 2}}
    a._fact_beliefs = {"f-1": {"value": 3, "conf": 0.8, "verified": False}}
    a._reputation = 2.5
    a._goods_belongings = ["傘", "弁当"]
    a.controllability = 0.73
    state = pool_mod.dehydrate(a)
    json.dumps(state)                            # JSON 安全(集合を残していない)
    assert state["gossip_heard"] == {7: [1, 2, 4]}, "集合を sorted list へ正規化していない"

    b = _bare_agent(22)
    pool_mod.hydrate(b, state)
    assert (b.med_admitted, b.med_until, b.med_since) == (True, 900, 700)
    assert (b.med_node, b.med_poi, b.med_confirmed) == ("n42", "病院", 3)
    assert getattr(b, pool_mod._INTOX_KEY) == 123
    assert b._gossip_known == {5: 2, 9: 3}
    assert b._gossip_heard == {7: {1, 2, 4}}     # 集合へ戻る(membership/len で使われる)
    assert b._fact_beliefs == a._fact_beliefs
    assert b._reputation == 2.5
    assert b._goods_belongings == ["傘", "弁当"]
    assert b.controllability == 0.73             # C2: 再来街で「学習前」に戻らない
    # 退避辞書を書き換えても復元後の個体に波及しない(実体を共有しない)
    state["fact_beliefs"]["f-1"]["conf"] = 0.1
    state["belongings"].append("鍵")
    assert b._fact_beliefs["f-1"]["conf"] == 0.8
    assert b._goods_belongings == ["傘", "弁当"]


def test_d1_hydrate_tolerates_old_state_without_new_keys():
    """旧 退避辞書(D1 の新キー無し)からの復元で属性を 1 つも生やさない(前方互換)。"""
    a = _bare_agent()
    old = pool_mod.dehydrate(a)
    b = _bare_agent(23)
    pool_mod.hydrate(b, old)
    for attr in ("med_admitted", pool_mod._INTOX_KEY, "_gossip_known", "_gossip_heard",
                 "_fact_beliefs", "_reputation", "_goods_belongings", pool_mod._CTRL_KEY):
        assert not hasattr(b, attr), f"{attr} が旧辞書から生えている"


def test_health_slim_fields_now_cover_the_whole_body_family():
    """H1 の残り 4 欄(collapse / chronic / withdrawn / fatigue)が退避に載る。"""
    from society import health as health_mod
    assert pool_mod._HEALTH_FIELDS == health_mod._SLIM_FIELDS
    names = {n for n, _c, _d in pool_mod._HEALTH_FIELDS}
    assert {"sev_collapse_step", "chronic_days", "withdrawn", "fatigue"} <= names
    a = _bare_agent()
    a.withdrawn, a.chronic_days, a.fatigue = True, 7, 0.625
    a.sev_collapse_step = 88
    st = pool_mod.dehydrate(a)
    b = _bare_agent(24)
    pool_mod.hydrate(b, st)
    assert (b.withdrawn, b.chronic_days, b.fatigue, b.sev_collapse_step) \
        == (True, 7, 0.625, 88)


# ---------------------------------------------------------------- 縦煙相当の統合(mock・48step)
# 第108 の全ON縦煙で L1 が +106 行(+0.89%)食い違った構成を**縮小して**固定する。
# relations + friend_graph + joint + party + household + gossip + info_env を pool ON の上で
# 同時に立てるのがこの組み合わせの肝で(第98/第101 の resume 監査はこの同時 ON を通って
# いなかった)、conf/finals_observe.yaml のどのブロックが効いていたかを dotlist で写している
# (プロファイル本体には依存させない = 他レーンの conf 編集でこのテストが揺れない)。
_FINALS_LIKE_ON = {
    # ★層別クォータ ON は必須(finals_observe と同じ)。OFF だと平日は上位層が cap を
    #   食い切って来街層が 0 人になり(第91 破綻)、party が 1 件も成立しない = 縦煙の
    #   主因(party の joint_activity +97)を通らないテストになる。
    "pool.tier_quota.enabled": "true",
    "relations.enabled": "true",
    "relations.dunbar.enabled": "true",
    "relations.endogenous_accept.enabled": "true",
    "relations.endogenous_invite.enabled": "true",
    "relations.endogenous_quality.enabled": "true",
    "friend_graph.enabled": "true",
    "joint.enabled": "true",
    "party.enabled": "true",
    "household.enabled": "true",
    "household.family_dinner.enabled": "true",
    "gossip.enabled": "true",
    "info_env.enabled": "true",
    "info_env.influence.enabled": "true",
    "info_env.misinfo.enabled": "true",
}


def test_finals_like_all_on_resume_matches_straight(small_pool, tmp_path):
    """★第109 縦煙の縮小再現: 全ON相当 48step の「一気 vs 24+24」が L1 バイト一致。

    修正前の実測(本テストと同じ組み合わせ・実プール縮小):
        straight 3,066 行 / resume 3,079 行(+13)
        friend_graph_built +1(起動時 1 回の発火体の二重発火)
        joint_activity     +11(party の実体化が __init__ でもう一度走る)
        signal_summary     +1(1 交差点 1 日 1 件のガードが checkpoint に無い)
    修正後は 3,066 == 3,066(全行一致)。
    """
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d, stem="l1_events"):
        return pq.read_table(Path(d) / f"{stem}.parquet").to_pylist()

    st = tmp_path / "fin_st"
    Simulation(_pool_cfg("fin_st", small_pool, n_steps=48, cap=150, **_FINALS_LIKE_ON),
               out_dir=st).run()

    rs = tmp_path / "fin_rs"
    s1 = Simulation(_pool_cfg("fin_rs", small_pool, n_steps=24, cap=150,
                              **{"observer.checkpoint_every": 24}, **_FINALS_LIKE_ON),
                    out_dir=rs)
    for step in range(24):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 24, rs / "checkpoint" / "ckpt-000024.pkl.gz")
    s1._save_pool_sidecar(24)
    s1.logger.flush_segment()
    s2 = Simulation(_pool_cfg("fin_rs", small_pool, n_steps=48, cap=150,
                              **{"observer.checkpoint_every": 24}, **_FINALS_LIKE_ON),
                    out_dir=rs)
    s2.run(resume_from=rs)

    a, b = rows(st), rows(rs)
    # 空回り防止: 起動時 1 回の発火体が**実在する**構成であること
    kinds = Counter(r["kind"] for r in a)
    assert kinds["friend_graph_built"] == 1, "friend_graph_built が出ていない(前提が崩れた)"
    assert sum(1 for r in a if r["kind"] == "joint_activity"
               and "party" in str(r["payload"])) > 0, \
        "party の joint_activity が出ていない(前提が崩れた=cap/プールの再調整)"
    assert len(a) == len(b), (
        f"L1 行数不一致: straight={len(a)} resume={len(b)} / 差分 kind="
        f"{Counter(r['kind'] for r in b) - kinds}")
    assert a == b, "全ON相当の resume が straight と byte 不一致"
