"""第149 レーン2: 関係台帳の退避梯子(S7)+ 自然削除(REL)。

正典: docs/plans/relations-tier-plan.md §2(機構と文献)/ §3(実装計画)。

守るもの(検収基準の順)
  (1) **既定 lru = 現行と完全一致**(last_step 最古 → 相手id 小)。既定では
      `relations_evict` / `relations_tier_acq` の**インスタンス属性が 1 つも生えない**
      (= pickle/checkpoint バイト一致 + 250k 体でも RAM が増えない)。
  (2) 梯子順のピン: ①hear 洪水(closeness 無し)→ ②tier0 の弱い紐帯 → ③dormant → ④tier1。
  (3) **tier2 以上は退避不可侵**: 候補が尽きたら cap 超過を許容して持ち越す。
  (4) いま接触した相手は常に保護(現行 LRU と同じ不変条件)。
  (5) 自然削除: 既定 OFF は 1 件も削らない / ON の削除条件 / **dormant は削らない**(可逆性)/
      relation_forget イベントの payload。
  (6) 決定論(同 seed 2 ラン一致)・AgentRef の共有経路が同一挙動・契約列挙ピン。
検証は mock のみ(実 LLM 禁止・≤24step)。乱数は 1 本も新設しない。
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from society import registry as R
from society import relations as REL
from society.agents.memory import MemoryStore
from society.agents.ref import AgentRef
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS

_REPO_FINALS = Path(__file__).resolve().parents[1] / "conf" / "finals_observe.yaml"


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _mem(cap: int, mode: str = "tiered", tier_acq: float = 2.0) -> MemoryStore:
    m = MemoryStore(relations_max=cap)
    if mode != "lru":
        m.relations_evict = mode
        m.relations_tier_acq = tier_acq
    return m


def _put(m, oid: int, *, count=1, last_step=0, closeness=None, tier=None,
         dormant=None, dormant_closeness=None) -> None:
    """台帳エントリを手組みする(record_contact を通さずに状態だけ作る)。"""
    rel = {"name": f"n{oid}", "count": int(count), "last_step": int(last_step),
           "last": ""}
    if closeness is not None:
        rel["closeness"] = float(closeness)
    if tier is not None:
        rel["tier"] = int(tier)
    if dormant:
        rel["dormant"] = True
        rel["dormant_closeness"] = float(dormant_closeness or 0.0)
        rel["dormant_step"] = int(last_step)
    m.relations[oid] = rel


def _sim(tmp_path, name, n=12, steps=1, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# (1) 既定 lru = 現行と完全一致 / 属性を生やさない
# --------------------------------------------------------------------------- #
def test_default_is_lru_and_leaves_no_instance_attributes():
    m = MemoryStore(relations_max=3)
    assert m.relations_evict == "lru"                # ClassVar の既定が読まれる
    assert m.relations_tier_acq == 2.0
    assert "relations_evict" not in m.__dict__       # ★instance には生えない
    assert "relations_tier_acq" not in m.__dict__


def test_default_lru_eviction_is_unchanged():
    """tests/test_scale.py の既存ピンと同じ挙動(現行 LRU をそのまま踏襲)。"""
    m = MemoryStore(relations_max=3)
    m.record_contact(1, "a", 1)
    m.record_contact(2, "b", 2)
    m.record_contact(3, "c", 3)
    m.record_contact(1, "a", 4)
    m.record_contact(4, "d", 5)
    assert set(m.relations) == {1, 3, 4}
    m.record_contact(5, "e", 6)
    assert set(m.relations) == {1, 4, 5}


def test_lru_is_value_blind_by_design():
    """★現行 LRU の欠陥そのもの(梯子の存在理由)を明文で固定する。

    直近に声が聞こえただけの他人が、10 日会っていない親友を押し出す。"""
    m = MemoryStore(relations_max=2)
    _put(m, 1, count=50, last_step=10, closeness=30.0, tier=3)   # 親友(古い)
    _put(m, 2, count=1, last_step=900, closeness=None)           # 洪水(新しい)
    m.record_contact(3, "new", 1000)
    assert 1 not in m.relations, "LRU は親友を落とす(これが直すべき欠陥)"


# --------------------------------------------------------------------------- #
# (2) 梯子順のピン(洪水 → tier0 → dormant → tier1)
# --------------------------------------------------------------------------- #
def test_ladder_order_flood_then_weak_then_dormant_then_tier1():
    m = _mem(4)
    _put(m, 1, count=3, last_step=100)                            # ① 洪水
    _put(m, 2, count=9, last_step=100, closeness=0.4, tier=0)     # ② 弱い tier0
    _put(m, 3, count=9, last_step=100, dormant=True,
         dormant_closeness=7.0, closeness=0.0, tier=0)            # ③ dormant
    _put(m, 4, count=9, last_step=100, closeness=3.0, tier=1)     # ④ tier1
    _put(m, 5, count=9, last_step=100, closeness=8.0, tier=2)     # ⑤ 不可侵
    order = []
    for step in range(4):                            # 1 件ずつ超過させて順序を読む
        m.relations_max = len(m.relations) - 1
        before = set(m.relations)
        m.record_contact(99, "keep", 1000 + step)    # 触った相手は保護 + 1 件増える
        m.relations_max = len(m.relations)           # 以後は超過しない状態へ戻す
        gone = before - set(m.relations)
        order.extend(sorted(gone))
    assert order == [1, 2, 3, 4], order


def test_flood_entries_go_by_count_then_last_step_then_id():
    """① の段の順序: count 昇順 → last_step 昇順 → 相手id 昇順。"""
    m = _mem(3)
    _put(m, 10, count=5, last_step=1)
    _put(m, 11, count=2, last_step=9)                # count 最小 = 最初に落ちる
    _put(m, 12, count=5, last_step=0)                # count 同値なら last_step 昇順
    m.record_contact(99, "x", 100)
    assert 11 not in m.relations and set(m.relations) == {10, 12, 99}
    m.relations_max = 2
    m.record_contact(99, "x", 101)
    assert 12 not in m.relations, set(m.relations)


def test_weak_tier0_goes_by_closeness_then_last_step():
    m = _mem(3)
    _put(m, 10, count=9, last_step=5, closeness=1.9, tier=0)
    _put(m, 11, count=9, last_step=5, closeness=0.1, tier=0)     # 最弱
    _put(m, 12, count=9, last_step=1, closeness=1.9, tier=0)     # 同値 → 古い方
    m.record_contact(99, "x", 100)
    assert 11 not in m.relations
    m.relations_max = 2
    m.record_contact(99, "x", 101)
    assert 12 not in m.relations, set(m.relations)


def test_dormant_is_protected_from_flood_and_weak_ties():
    """休眠紐帯は洪水・弱い紐帯より**後**に落ちる(Levin et al. 2011)。"""
    m = _mem(2)
    _put(m, 1, count=1, last_step=999)                            # 洪水(最新)
    _put(m, 2, count=9, last_step=0, dormant=True,
         dormant_closeness=11.0, closeness=0.0, tier=0)           # 休眠(最古)
    m.record_contact(99, "x", 1000)
    assert set(m.relations) == {2, 99}, set(m.relations)


def test_tier_threshold_separates_weak_from_acquaintance():
    """closeness < tier_acquaintance の tier0 は②・それ以外の tier<=1 は④。"""
    m = _mem(2, tier_acq=2.0)
    _put(m, 1, count=9, last_step=0, closeness=1.5, tier=0)       # ② 弱い
    _put(m, 2, count=9, last_step=0, closeness=3.0, tier=1)       # ④ 知人
    m.record_contact(99, "x", 100)
    assert set(m.relations) == {2, 99}


# --------------------------------------------------------------------------- #
# (3) tier2 以上は退避不可侵(候補が無ければ cap 超過を許容)
# --------------------------------------------------------------------------- #
def test_tier2_and_above_are_never_evicted():
    m = _mem(2)
    _put(m, 1, count=9, last_step=0, closeness=8.0, tier=2)       # 友人
    _put(m, 2, count=9, last_step=0, closeness=20.0, tier=3)      # 親友
    m.record_contact(99, "x", 100)                                # 3 件 > cap 2
    assert set(m.relations) == {1, 2, 99}, "cap 超過を許容していない"
    assert len(m.relations) == 3
    # ①-④ の候補が現れたらそちらが落ちる(超過は解消へ向かう)
    _put(m, 3, count=1, last_step=1)
    m.record_contact(99, "x", 101)
    assert 3 not in m.relations and {1, 2}.issubset(m.relations)


def test_recent_contact_is_always_protected():
    """いま接触した相手は、たとえ最弱の洪水エントリでも落とさない。"""
    m = _mem(1)
    _put(m, 1, count=99, last_step=0, closeness=1.0, tier=0)
    m.record_contact(7, "fresh", 5)                  # 7 は count=1 の最弱だが保護される
    assert 7 in m.relations
    assert set(m.relations) == {7}


# --------------------------------------------------------------------------- #
# (4) AgentRef(退場者)の共有経路が同一挙動
# --------------------------------------------------------------------------- #
def test_agent_ref_shares_the_same_ladder(tmp_path):
    sim = _sim(tmp_path, "ref", n=4)
    a = sim.agents[0]
    a.mem.relations_max = 2
    a.mem.relations_evict = "tiered"
    a.mem.relations_tier_acq = 2.0
    _put(a.mem, 1, count=9, last_step=0, closeness=9.0, tier=2)  # 友人
    _put(a.mem, 2, count=1, last_step=900)                       # 洪水
    ref = AgentRef(a)
    assert ref.mem.relations is a.mem.relations      # 同じ dict を共有(写しを作らない)
    assert ref.mem.relations_evict == "tiered"
    assert ref.mem.relations_tier_acq == 2.0
    ref.mem.record_contact(3, "x", 1000)
    assert 2 not in a.mem.relations, "退場者経路で梯子が効いていない"
    assert 1 in a.mem.relations


def test_agent_ref_defaults_to_lru(tmp_path):
    sim = _sim(tmp_path, "ref_lru", n=4)
    a = sim.agents[0]
    a.mem.relations_max = 2
    _put(a.mem, 1, count=9, last_step=0, closeness=9.0, tier=2)
    _put(a.mem, 2, count=1, last_step=900)
    ref = AgentRef(a)
    assert ref.mem.relations_evict == "lru"
    ref.mem.record_contact(3, "x", 1000)
    assert 1 not in a.mem.relations, "既定は現行 LRU(価値盲目)のまま"


# --------------------------------------------------------------------------- #
# (5) 自然削除(REL)
# --------------------------------------------------------------------------- #
def _forget_scene(tmp_path, name, **ov):
    """1 体の台帳へ 5 種のエントリを置き、日境界の decay_day を 1 回だけ回す。"""
    sim = _sim(tmp_path, name, n=4, **{"relations.enabled": "true", **ov})
    a = sim.agents[0]
    a.mem.relations.clear()
    # today = 10 日目。last_step は step 単位(clock.day で日へ変換される)。
    # old = 6 日目(gap 4 > after_days 3)/ fresh = 9 日目(gap 1 ≤ 3)。
    steps_per_day = 1440 // int(sim.cfg.run.dt_min)
    old = 6 * steps_per_day
    fresh = 9 * steps_per_day
    _put(a.mem, 1, count=1, last_step=old, closeness=0.2, tier=0)    # ★削除対象
    _put(a.mem, 2, count=9, last_step=old, closeness=0.2, tier=0)    # count 超過
    _put(a.mem, 3, count=1, last_step=old, closeness=4.0, tier=1)    # closeness 高
    _put(a.mem, 4, count=1, last_step=fresh, closeness=0.2, tier=0)  # 直近すぎる
    _put(a.mem, 5, count=1, last_step=old, closeness=0.0, tier=0,
         dormant=True, dormant_closeness=6.0)                        # 休眠=不可触
    _put(a.mem, 6, count=1, last_step=old)                           # closeness 無し
    sim_min = 10 * 1440
    REL.decay_day(sim, sim.relationscfg, sim_min // int(sim.cfg.run.dt_min),
                  sim_min)
    return sim, a


def test_forget_is_off_by_default(tmp_path):
    sim, a = _forget_scene(tmp_path, "forget_off")
    assert set(a.mem.relations) == {1, 2, 3, 4, 5, 6}
    assert [e for e in sim.logger.events if e.kind == "relation_forget"] == []


def test_forget_on_removes_only_the_faded_ties(tmp_path):
    sim, a = _forget_scene(tmp_path, "forget_on",
                           **{"memory.relations_forget.enabled": "true"})
    assert 1 not in a.mem.relations, "減衰しきった紐帯が残っている"
    # 残るもの: 常連(2)・親密(3)・直近(4)・休眠(5)・closeness 無し(6)
    assert set(a.mem.relations) == {2, 3, 4, 5, 6}, set(a.mem.relations)


def test_forget_never_touches_dormant_ties(tmp_path):
    """休眠は「退避であって削除でない」= 可逆性を守る(floor をどれだけ上げても残る)。"""
    sim, a = _forget_scene(tmp_path, "forget_dorm",
                           **{"memory.relations_forget.enabled": "true",
                              "memory.relations_forget.floor": 99.0,
                              "memory.relations_forget.count_max": 99,
                              "memory.relations_forget.after_days": 0})
    assert 5 in a.mem.relations and a.mem.relations[5]["dormant"] is True


def test_forget_event_payload(tmp_path):
    sim, a = _forget_scene(tmp_path, "forget_ev",
                           **{"memory.relations_forget.enabled": "true"})
    evs = [e for e in sim.logger.events if e.kind == "relation_forget"]
    assert len(evs) == 1
    p = evs[0].payload
    assert p["other"] == 1 and p["count"] == 1
    assert set(p) == {"other", "closeness", "count", "gap_days"}
    assert p["gap_days"] >= 1
    assert evs[0].agent_id == a.id


def test_forget_cfg_defaults_and_coercion():
    cfg = REL.build_forget_cfg(None)
    assert cfg == {"enabled": False, "floor": 0.5, "count_max": 2,
                   "after_days": 3}
    got = REL.build_forget_cfg({"enabled": "1", "count_max": "5",
                                "floor": "1.5", "unknown": 3})
    assert got["enabled"] is True and got["count_max"] == 5
    assert got["floor"] == 1.5 and "unknown" not in got


# --------------------------------------------------------------------------- #
# (6) 決定論・イベント種の登録・契約列挙ピン
# --------------------------------------------------------------------------- #
def test_relation_forget_kind_is_registered_in_both_places():
    """★schema と causality の**両方**に登録する(片方だけだと本選 conf で即死する)。"""
    assert "relation_forget" in EVENT_KINDS
    assert C.CAUSE_OF_KIND["relation_forget"] == C.DEVICE


def test_deterministic_run_with_both_knobs_on(tmp_path):
    ov = {"relations.enabled": "true", "memory.relations_max": 40,
          "memory.relations_evict": "tiered",
          "memory.relations_forget.enabled": "true"}
    a = _sim(tmp_path, "rel_det_a", n=20, steps=12, **ov)
    a.run()
    b = _sim(tmp_path, "rel_det_b", n=20, steps=12, **ov)
    b.run()
    assert _l1(a) == _l1(b)


def test_default_off_run_matches_pure_default(tmp_path):
    base = _sim(tmp_path, "rel_off_base", n=20, steps=12)
    base.run()
    same = _sim(tmp_path, "rel_off_same", n=20, steps=12,
                **{"memory.relations_evict": "lru",
                   "memory.relations_forget.enabled": "false"})
    same.run()
    assert _l1(base) == _l1(same)


def test_unknown_evict_mode_falls_back_to_current_behaviour(tmp_path):
    sim = _sim(tmp_path, "rel_bad", n=4, **{"memory.relations_evict": "whatever"})
    assert sim.relations_evict == "lru"
    assert "relations_evict" not in sim.agents[0].mem.__dict__


def test_simulation_wires_the_ladder_only_when_enabled(tmp_path):
    off = _sim(tmp_path, "wire_off", n=4)
    assert "relations_evict" not in off.agents[0].mem.__dict__
    on = _sim(tmp_path, "wire_on", n=4,
              **{"memory.relations_evict": "tiered", "relations.enabled": "true"})
    mem = on.agents[0].mem
    assert mem.relations_evict == "tiered"
    assert mem.relations_tier_acq == float(on.relationscfg["tier_acquaintance"])


def test_conf_defaults_are_all_off():
    cfg = load_config()
    assert str(cfg.memory.relations_evict) == "lru"
    assert bool(cfg.memory.relations_forget.enabled) is False
    assert float(cfg.memory.relations_forget.floor) == 0.5
    assert int(cfg.memory.relations_forget.count_max) == 2
    assert int(cfg.memory.relations_forget.after_days) == 3
    assert int(cfg.memory.relations_max) == 0


def test_registry_declares_the_new_keys():
    for dotted in ("memory.relations_evict", "memory.relations_forget.enabled"):
        f = R.BY_ID.get(dotted)
        assert f is not None, f"{dotted} がレジストリ未宣言"
        assert f.repro_tier == "strict", dotted      # LLM の自由文を読まない
        assert f.affects_k is False, dotted          # generate() の呼び出し点は不変
        assert f.fingerprint_risk == "possible", dotted
    assert R.BY_ID["memory.relations_evict"].off_value == "lru"
    assert R.undeclared_toggles(load_config()) == []


def test_finals_profile_declares_the_registered_values():
    fin = OmegaConf.load(_REPO_FINALS)
    assert int(fin.memory.relations_max) == 2000
    assert str(fin.memory.relations_evict) == "tiered"
    # ★自然削除は**実装のみ**: 本選は false 据え置き(ON 判断は判定ラウンド)
    assert bool(fin.memory.relations_forget.enabled) is False


def test_touched_files_are_not_frozen():
    """本レーンが触ったファイルが凍結 SPEC_FILES に 1 つも入っていない。"""
    from society.observer import metrics_spec as MS
    touched = (
        "src/society/agents/memory.py",
        "src/society/agents/ref.py",
        "src/society/relations.py",
        "src/society/engine/simulation.py",
        "src/society/observer/schema.py",
        "src/society/observer/causality.py",
        "src/society/registry.py",
    )
    for rel in touched:
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"
