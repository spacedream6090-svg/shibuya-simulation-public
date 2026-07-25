"""負の評判(悪評)の内生伝播 第61バッチ スライス(c)のテスト。

検証項目(タスク検収 2026-07-25):
  (R1)   OFF(既定)= 純粋既定と L1 完全一致(gossip_seed/spread/fade 0 件・stream "gossip" 不使用)。
         OFF は L2 に gossip_* 列を一切足さない。relations ON でも gossip OFF は完全 no-op。
  (scan) 種スキャン=conf の seed_events マップ(データ駆動)で負イベント当事者を pending 化。
         seed_exclude_payload(rehoused 等の回復サブイベント)は種にしない。
  (seed) 種ロール=(agent,day) 確率 seed_prob で悪評タグ化。対象の直近会話相手が初期の知る者に。
         seed_prob=0 で不発・=1 で必発(決定論)。
  (spread) complex contagion 閾値=2 の直接検証(1人からは採用しない/2人で採用)。
  (fade) 忘却=decay_prob=1 で消える・=0 で残る。
  (sanction) demote_partners(相手選択後退)・joint_penalty・status_penalty と max_penalty 上限。
  (det)  ON は決定論(同一入力 2 回で L1 バイト一致)。ON は L2 に gossip_* 3 列を出す。
  (resume) checkpoint 状態(_gossip_known/_gossip_pending/_gossip_day/_gossip_state)が resume==straight。
  (k)    compute_matched 下で gossip ON の k=free と k=off の generate 呼数が完全一致(R1 の本旨)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society import gossip
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.logger import ObserverLogger
from society.observer.schema import Event
from society.rng import RngHub

GOSSIP_COLS = ("gossip_active_count", "gossip_spread_total", "gossip_reach_mean")


# --------------------------------------------------------------------------- #
# 軽量フェイク(種/伝播/忘却/効果の決定論を精密に単体検証する)
# --------------------------------------------------------------------------- #
class _Mem:
    def __init__(self):
        self.relations: dict = {}


class _Agent:
    def __init__(self, aid, x=0.0, y=0.0, visitor=False):
        self.id = aid
        self.x = x
        self.y = y
        self.visitor = visitor
        self.mem = _Mem()


class _Clock:
    def day(self, step):
        return int(step) // 144


class _Sim:
    """gossip.phase / 効果フックが要る最小面のみ。"""
    def __init__(self, tmp_path, agents, cfg_over=None, seed=7):
        self.agents = agents
        self.agent_by_id = {a.id: a for a in agents}
        self.hub = RngHub(seed)
        self.clock = _Clock()
        self.logger = ObserverLogger(Path(tmp_path))
        self.gossipcfg = gossip.build_cfg({"enabled": True, **(cfg_over or {})})
        self.cfg = {}          # cfg_of は gossipcfg キャッシュを優先=ここは未使用


def _contact(agent, other_id, last_step):
    agent.mem.relations[other_id] = {"name": f"a{other_id}", "count": 1,
                                     "last_step": last_step, "last": ""}


def _events(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------------- #
# (scan) 種スキャン=データ駆動 + 回復サブイベント除外
# --------------------------------------------------------------------------- #
def test_scan_seeds_data_driven(tmp_path):
    sim = _Sim(tmp_path, [_Agent(0), _Agent(1)])
    cfg = sim.gossipcfg
    sim.logger.log(Event(step=1, sim_min=10, agent_id=0, kind="bankruptcy", x=0, y=0,
                         payload={"debt": 100}))
    sim.logger.log(Event(step=1, sim_min=10, agent_id=1, kind="speak", x=0, y=0,
                         payload={"text": "x"}))              # 非種イベント=無視
    gossip._scan_seeds(sim, cfg)
    assert sim._gossip_pending == {0: (1.0, "bankruptcy")}, "種スキャンがデータ駆動でない"


def test_scan_seeds_excludes_recovery_payload(tmp_path):
    sim = _Sim(tmp_path, [_Agent(0)])
    cfg = sim.gossipcfg
    sim.logger.log(Event(step=1, sim_min=10, agent_id=0, kind="eviction", x=0, y=0,
                         payload={"phase": "rehoused"}))       # 回復=種にしない
    gossip._scan_seeds(sim, cfg)
    assert sim._gossip_pending == {}, "回復サブイベント(rehoused)が種化された"


# --------------------------------------------------------------------------- #
# (seed) 種ロール=(agent,day) 確率・対象の直近会話相手が知る者
# --------------------------------------------------------------------------- #
def _seed_run(tmp_path, seed_prob):
    C = _Agent(0)
    k1, k2, other = _Agent(1), _Agent(2), _Agent(3)
    _contact(C, 1, 100)          # C は 1,2 と day0(step100=day0)に会話
    _contact(C, 2, 100)
    sim = _Sim(tmp_path, [C, k1, k2, other],
               cfg_over={"seed_prob": seed_prob, "decay_prob": 0.0, "seed_contact_days": 1})
    sim._gossip_pending = {0: (1.0, "bankruptcy")}
    sim._gossip_day = 0
    gossip.phase(sim, step=144, sim_min=1440)                 # day0→1 の境界でロール
    return sim, k1, k2, other


def test_seed_fires_and_marks_recent_contacts(tmp_path):
    sim, k1, k2, other = _seed_run(tmp_path, 1.0)
    seeds = _events(sim, "gossip_seed")
    assert len(seeds) == 1 and seeds[0].payload == {"n": 2, "cause": "bankruptcy"}
    assert getattr(k1, "_gossip_known") == {0: 1} and getattr(k2, "_gossip_known") == {0: 1}
    assert not getattr(other, "_gossip_known", None), "会話していない者まで知る者になった"


def test_seed_never_fires_at_prob_zero(tmp_path):
    sim, k1, _k2, _o = _seed_run(tmp_path, 0.0)
    assert not _events(sim, "gossip_seed"), "seed_prob=0 で種が出た"
    assert not getattr(k1, "_gossip_known", None)


# --------------------------------------------------------------------------- #
# (spread) complex contagion 閾値=2(1人では採用しない/2人で採用)
# --------------------------------------------------------------------------- #
def _spread_run(tmp_path, n_knower_contacts):
    """listener L(id0)が day0 に n 人の「target 9 を知る知人」と会話した状況で day1 境界に伝播判定。"""
    L = _Agent(0)
    knowers = [_Agent(i) for i in (1, 2)]
    for kn in knowers:
        kn._gossip_known = {9: 0}
    for i in range(n_knower_contacts):
        _contact(L, knowers[i].id, 100)                      # day0 に会話
    sim = _Sim(tmp_path, [L, *knowers], cfg_over={"decay_prob": 0.0, "adopt_threshold": 2})
    sim._gossip_day = 0
    gossip.phase(sim, step=144, sim_min=1440)
    return sim, L


def test_spread_one_source_does_not_adopt(tmp_path):
    sim, L = _spread_run(tmp_path, 1)
    assert not _events(sim, "gossip_spread"), "1人から聞いただけで採用した(閾値2違反)"
    assert 9 not in getattr(L, "_gossip_known", {})


def test_spread_two_sources_adopt(tmp_path):
    sim, L = _spread_run(tmp_path, 2)
    sp = _events(sim, "gossip_spread")
    assert len(sp) == 1 and sp[0].payload == {"target": 9, "sources": 2}
    assert getattr(L, "_gossip_known", {}).get(9) == 1, "2人の独立情報源で採用されない"


# --------------------------------------------------------------------------- #
# (fade) 忘却=decay_prob で消える / 残る
# --------------------------------------------------------------------------- #
def _fade_run(tmp_path, decay_prob):
    a = _Agent(0)
    a._gossip_known = {9: 0}
    sim = _Sim(tmp_path, [a], cfg_over={"decay_prob": decay_prob})
    sim._gossip_day = 0
    gossip.phase(sim, step=144, sim_min=1440)
    return sim, a


def test_fade_forgets_at_prob_one(tmp_path):
    sim, a = _fade_run(tmp_path, 1.0)
    assert _events(sim, "gossip_fade"), "decay_prob=1 で忘却が起きない"
    assert 9 not in a._gossip_known, "忘れたのに悪評が残っている"


def test_fade_keeps_at_prob_zero(tmp_path):
    sim, a = _fade_run(tmp_path, 0.0)
    assert not _events(sim, "gossip_fade") and a._gossip_known == {9: 0}


# --------------------------------------------------------------------------- #
# (sanction) 相手選択後退 / joint 誘い低下 / status 負項 + max_penalty 上限
# --------------------------------------------------------------------------- #
def test_demote_partners_deprioritizes_known_bad(tmp_path):
    sim = _Sim(tmp_path, [_Agent(0)])
    a = sim.agents[0]
    a._gossip_known = {5: 0}
    h5, h6 = _Agent(5), _Agent(6)
    assert gossip.demote_partners(sim, a, [h5, h6]) == [h6], "悪評相手が後退していない"
    # 全員が悪評対象なら素通り(会話は必ず起きる=相手だけ変わる)
    assert gossip.demote_partners(sim, a, [h5]) == [h5], "全員悪評でも会話相手を残すべき"


def test_joint_penalty_and_cap(tmp_path):
    sim = _Sim(tmp_path, [_Agent(0)],
               cfg_over={"joint_penalty": 0.5, "max_penalty": 0.2})
    inv = sim.agents[0]
    inv._gossip_known = {7: 0}
    assert gossip.joint_penalty(sim, inv, 7) == 0.2, "誘い低下が max_penalty で頭打ちにならない"
    assert gossip.joint_penalty(sim, inv, 8) == 0.0, "悪評を知らない相手に誘い低下が効いた"


def test_status_penalty_and_cap(tmp_path):
    sim = _Sim(tmp_path, [_Agent(i) for i in range(10)],
               cfg_over={"status_weight": 0.1, "max_penalty": 0.2})
    assert gossip.status_penalty(sim, 3, {3: 5}) == 0.05     # 0.1 * 5/10
    assert gossip.status_penalty(sim, 3, {3: 100}) == 0.2    # capped(0.1*10=1.0 → 0.2)


def test_effects_noop_when_disabled(tmp_path):
    sim = _Sim(tmp_path, [_Agent(0)])
    sim.gossipcfg = gossip.build_cfg({"enabled": False})
    inv = sim.agents[0]
    inv._gossip_known = {7: 0}
    assert gossip.joint_penalty(sim, inv, 7) == 0.0
    assert gossip.status_penalty(sim, 0, {0: 5}) == 0.0


# --------------------------------------------------------------------------- #
# Simulation 統合(R1 / L2 / 決定論 / resume / k 不変)
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n=14, steps=1, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def test_off_matches_pure_default(tmp_path):
    """OFF(既定)= 純粋既定と L1 完全一致・gossip イベント 0 件・L2 に gossip 列なし(120 step)。"""
    pure = _sim(tmp_path, "pure", steps=120)
    pure.run()
    off = _sim(tmp_path, "off", steps=120, **{"gossip.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と L1 不一致(gossip seam が no-op でない)"
    for k in GOSSIP_COLS:
        assert all(k not in row for row in pure.logger.metrics), f"OFF で L2 に {k} 列がある"
    for k in ("gossip_seed", "gossip_spread", "gossip_fade"):
        assert not [e for e in pure.logger.events if e.kind == k], f"OFF で {k} が出た"


def test_off_noop_with_relations_on(tmp_path):
    """relations ON の世界でも gossip OFF は完全 no-op(relations ON 基準と L1 一致・120 step)。"""
    base = _sim(tmp_path, "relbase", steps=120, **{"relations.enabled": "true"})
    base.run()
    off = _sim(tmp_path, "reloff", steps=120,
               **{"relations.enabled": "true", "gossip.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "relations ON で gossip seam が no-op でない"


def test_on_emits_l2_columns(tmp_path):
    """gossip ON は L2 に gossip_* 3 列を毎行出す(型安定)。"""
    on = _sim(tmp_path, "on_l2", steps=120,
              **{"relations.enabled": "true", "gossip.enabled": "true"})
    on.run()
    assert on.logger.metrics, "L2 が空"
    for row in on.logger.metrics:
        for col in GOSSIP_COLS:
            assert col in row, f"gossip ON なのに L2 に {col} 列が無い"
        assert isinstance(row["gossip_active_count"], int)
        assert isinstance(row["gossip_reach_mean"], float)


def test_on_deterministic_l1(tmp_path):
    a = _sim(tmp_path, "det_a", steps=120,
             **{"relations.enabled": "true", "gossip.enabled": "true"})
    a.run()
    b = _sim(tmp_path, "det_b", steps=120,
             **{"relations.enabled": "true", "gossip.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b), "gossip ON の L1 が決定論でない(2 回で不一致)"


# ---- resume==straight(日境界=step 102 を跨ぐ・gossip ON)----
# ★ relations も ON にすると **pre-existing** の relations resume gap(_rel_day が checkpoint 未保存=
#   reputation_decay が resume 初 step で二重発火)を踏むため、end-to-end resume は gossip 単独 ON で
#   gossip 自身の resume 機構(日境界ゲート/watermark/state/L2)を隔離検証する。悪評の実状態(知る者/
#   pending)の round-trip は下の test_checkpoint_roundtrips_gossip_state が直接検証する。
def _cfg_r(name, n_steps, **ov):
    dot = ["run.seed=42", "run.n_agents=16", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "gossip.enabled=true"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir, stem="l1_events"):
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def test_resume_matches_straight(tmp_path):
    """一気 160step と 80+resume の l1/l2/l3 が全行一致(gossip の日境界ゲート/watermark/state が復元)。"""
    straight = tmp_path / "g_straight"
    Simulation(_cfg_r("g_straight", 160), out_dir=straight).run()
    # phase1: 80 step で ckpt→中断(クラッシュ相当)。phase2: load→160 まで。
    resumed = tmp_path / "g_resume"
    sim1 = Simulation(_cfg_r("g_resume", 80, **{"observer.checkpoint_every": 80}),
                      out_dir=resumed)
    for step in range(80):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 80, resumed / "checkpoint" / "ckpt-000080.pkl.gz")
    sim1.logger.flush_segment()
    Simulation(_cfg_r("g_resume", 160, **{"observer.checkpoint_every": 80}),
               out_dir=resumed).run(resume_from=resumed)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} が resume で不一致"


def test_checkpoint_roundtrips_gossip_state(tmp_path):
    """悪評の状態(agent._gossip_known / sim._gossip_pending / _gossip_day / _gossip_state)が
    save→load で完全復元(assets/org と同じく checkpoint.py の中央管理)。"""
    sim = _sim(tmp_path, "ckpt_gs", steps=1,
               **{"relations.enabled": "true", "gossip.enabled": "true"})
    scheduler.run_step(sim, 0)
    sim.agents[0]._gossip_known = {3: 2, 5: 1}
    sim.agents[1]._gossip_heard = {7: {0, 2}}
    sim._gossip_pending = {4: (0.7, "eviction")}
    sim._gossip_day = 2
    sim._gossip_state = {"day": 2, "seed": 3, "spread": 1, "fade": 2}
    p = checkpoint.save(sim, 1, tmp_path / "ck" / "ckpt-000001.pkl.gz")
    sim2 = _sim(tmp_path, "ckpt_gs2", steps=5,
                **{"relations.enabled": "true", "gossip.enabled": "true"})
    checkpoint.load(sim2, p)
    assert sim2.agents[0]._gossip_known == {3: 2, 5: 1}
    assert sim2.agents[1]._gossip_heard == {7: {0, 2}}
    assert sim2._gossip_pending == {4: (0.7, "eviction")}
    assert sim2._gossip_day == 2
    assert sim2._gossip_state == {"day": 2, "seed": 3, "spread": 1, "fade": 2}
    assert sim2._gossip_watermark == 0, "watermark は load で 0 に戻る(fresh logger)"


# ---- k 不変性(compute_matched: R1 の本旨=呼数が k に依存しない)----
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    sim = _sim(tmp_path, name, n=20, steps=120,
               **{"relations.enabled": "true", "gossip.enabled": "true",
                  "gossip.seed_prob": "1.0", "controls.mode": "compute_matched",
                  "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_gossip_call_count_k_invariant(tmp_path):
    """制裁(相手選択後退/joint 誘い)は対面 co-location を変えうる(career/joint と同型)。だが R1 の本旨=
    「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる(gossip は k・内面状態を
    一切読まず、既存イベント列・会話接触台帳・専用 stream・config のみ参照するため)。"""
    free = _run_k(tmp_path, "gk_free", writeback="free")
    off = _run_k(tmp_path, "gk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"gossip の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
