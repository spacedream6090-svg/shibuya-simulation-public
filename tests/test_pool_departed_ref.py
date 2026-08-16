"""退場者の軽量参照 ``AgentRef``(レーン R1 A1)。

正典: src/society/agents/ref.py の docstring。

背景(裏取り済みの事実):
  ``sim.agent_by_id`` は「これまで実体化した**全**個体」の索引で、プール回転で街を出た
  個体を**意図的に消さない**(過去の参照 = 造語の作者名・SNS/DM の著者名・関係台帳が
  退場後も解決できるように)。要件は正しいが、解決に要るのは名前などの数十バイトなのに
  統合エピソード 120 + 未統合 30 + 関係台帳 + 信念 + persona 文を抱えた**フル Agent**を
  累計ぶん掴み続けていた(本選 = 25 万在場 + 途中入場 20.9 万 = 累計 45.9 万体)。

検証:
  (1) 運ぶ欄の表(``READ_BY``)と実装(``CARRIED``)の対応 —— 表に書いた欄は必ず在る。
  (2) 落とす欄は **AttributeError を明示 raise**(静かな既定値を作らない)。
  (3) ``getattr(x, name, default)`` で読む読者は、フル Agent と**同じ答え**になる。
  (4) 書き込みは通る(退場者への書き込みは現行でも世界に残らない = 同値)。
  (5) pool ON の mock ラン: 差し替え あり/なし で **L1 がバイト一致**する。
  (6) 造語の作者名 / SNS の著者名が退場後も同一に解決する。
  (7) 回転 2 周(退場 → 再入場 → 再退場)で壊れない。
  (8) 退場者 pickle が**実測で縮む**。
  (9) 旧サイドカー(departed にフル Agent・vital 無し)を resume でも受ける。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society.agents.agent import Agent                 # noqa: E402
from society.agents.memory import MemoryStore          # noqa: E402
from society.agents.ref import AgentRef, is_ref, to_ref  # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine import scheduler                   # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない。test_pool_rotation と同型)。"""
    out = tmp_path_factory.mktemp("pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _agent(aid=7, **ov):
    a = Agent(id=aid, name=f"甲{aid}", age=30, occupation="会社員",
              persona="長い自己紹介文" * 20, traits={"openness": 0.5},
              states={"grievance": 0.2, "efficacy": 0.4}, mem=MemoryStore())
    a.beliefs = ["渋谷は落ち着く"]
    a.mem.observe(1, "スクランブルを歩いた")
    a.mem.consolidate(1, "散歩した", [("スクランブルを歩いた", 6.0)])
    a.mem.record_contact(99, "乙", 1, "はじめまして")
    a.adopted = {"バズ語"}
    a.money = 12345.0
    for k, v in ov.items():
        setattr(a, k, v)
    return a


# ===================================================== (1) 表と実装の対応
def test_read_by_table_is_covered_by_the_carried_fields():
    """『退場者に対して読まれる欄』の全数列挙表に載った欄は、必ず ref に在る。"""
    carried = set(AgentRef.CARRIED)
    missing = sorted(k for k in AgentRef.READ_BY if k not in carried)
    assert missing == [], f"表に載っているのに運んでいない欄: {missing}"


def test_read_by_table_has_a_reader_for_every_entry():
    """表の各欄には『どこが読むか』が 1 件以上書いてある(空回りの表を作らない)。"""
    for name, sites in AgentRef.READ_BY.items():
        assert sites, f"{name} に読者が書かれていない"


def test_dropped_and_carried_do_not_overlap():
    assert not (set(AgentRef.CARRIED) & set(AgentRef.DROPPED))


# ===================================================== (2) 落とす欄は明示 raise
@pytest.mark.parametrize("name", ["persona", "beliefs", "heard_counts", "visits",
                                  "said", "day_plan", "schedule", "self_dev"])
def test_dropped_attribute_raises_with_an_explicit_reason(name):
    """静かな既定値を返さず、**なぜ落としたか**を述べて AttributeError を投げる。"""
    ref = AgentRef(_agent())
    with pytest.raises(AttributeError) as e:
        getattr(ref, name)
    msg = str(e.value)
    assert name in msg and "AgentRef" in msg
    assert "present_agent" in msg or "_dormant" in msg     # 正しい代替経路を指す


def test_unknown_attribute_raises_and_points_at_the_table():
    ref = AgentRef(_agent())
    with pytest.raises(AttributeError) as e:
        ref.totally_unknown_field
    assert "ref.py" in str(e.value)                        # 足す場所を明示する


@pytest.mark.parametrize("name", ["episodes", "buffer", "day_summaries", "observe"])
def test_ref_mem_only_keeps_the_relation_ledger(name):
    ref = AgentRef(_agent())
    with pytest.raises(AttributeError):
        getattr(ref.mem, name)


# ===================================================== (3) getattr 既定値の読者
#: ``getattr(x, name, default)`` で退場者を読む場所の実物(ref.py の表と対)。
_GETATTR_READERS = ("home_building", "visitor", "evicted", "status", "language",
                    "tourist", "occupation", "org_role", "traits", "partner_id",
                    "housemates", "dead", "loc", "sleeping", "node", "building",
                    "floor", "money", "account", "pool_pid", "adopted", "states")


@pytest.mark.parametrize("name", _GETATTR_READERS)
def test_getattr_with_default_readers_get_the_same_answer(name):
    """フル Agent と ref で ``getattr(x, name, sentinel)`` が同じ値を返す。

    ここが割れると『街を出た人だけ既定値になる』= 気づけない挙動変化になる。
    """
    a = _agent(pool_pid="L2_000001", org_role="staff", language="ja")
    ref = AgentRef(a)
    sentinel = object()
    assert getattr(ref, name, sentinel) == getattr(a, name, sentinel)


def test_optional_attribute_absent_on_the_agent_stays_absent_on_the_ref():
    """動的属性を**持たない**個体には生やさない(既定値が変わってしまうため)。"""
    a = _agent()                                   # controllability を持たない
    assert not hasattr(a, "controllability")
    ref = AgentRef(a)
    assert not hasattr(ref, "controllability")
    assert getattr(ref, "controllability", 0.5) == getattr(a, "controllability", 0.5)


def test_optional_attribute_present_on_the_agent_is_carried():
    a = _agent()
    a.controllability = 0.73
    a._fact_beliefs = {"f1": {"conf": 0.9}}
    a.mind = {"model": "m", "tier": "high"}
    ref = AgentRef(a)
    assert ref.controllability == 0.73
    assert ref._fact_beliefs == {"f1": {"conf": 0.9}}
    assert ref.mind["model"] == "m"


# ===================================================== (4) 書き込み・記憶・関係台帳
def test_writes_pass_through_and_are_readable_again():
    """退場者への書き込みは現行でも世界に残らない = ref でも同値(落として良いが落とさない)。"""
    ref = AgentRef(_agent())
    ref.controllability = 0.9                      # 表に在る欄
    assert ref.controllability == 0.9
    ref._brand_new_flag = 3                        # 表に無い欄 → _extra へ退避
    assert ref._brand_new_flag == 3


def test_relations_are_shared_not_copied():
    """関係台帳は**同じ dict**(コピーすると退場のたびに台帳が二重になる)。"""
    a = _agent()
    ref = AgentRef(a)
    assert ref.mem.relations is a.mem.relations
    ref.mem.record_contact(11, "丙", 2, "やあ")     # MemoryStore と同一実装を借りている
    assert a.mem.relations[11]["name"] == "丙"


def test_remember_is_a_noop_because_the_write_is_discarded_today():
    """退場者への remember は現行でも次の hydrate で捨てられる = no-op が現行と同値。"""
    ref = AgentRef(_agent())
    assert ref.remember("何か", importance_bonus=1.0) is None


def test_pickle_roundtrip_keeps_the_fields():
    a = _agent()
    a.controllability = 0.4
    ref = AgentRef(a)
    ref._extra_thing = "x"
    back = pickle.loads(pickle.dumps(ref, protocol=pickle.HIGHEST_PROTOCOL))
    assert (back.id, back.name, back.money) == (a.id, a.name, a.money)
    assert back.controllability == 0.4
    assert back._extra_thing == "x"
    assert back.mem.relations == a.mem.relations
    with pytest.raises(AttributeError):
        back.beliefs


def test_to_ref_is_idempotent():
    a = _agent()
    r = to_ref(a)
    assert is_ref(r) and to_ref(r) is r


# ===================================================== ON mock 統合
def _pool_cfg(name, pool_dir, n_steps, cap=150, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "pool.enabled=true",
           f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


#: 第109 縦煙と同じ組み合わせ(test_pool_rotation._FINALS_LIKE_ON の写し + 退場者を読む機構)。
_ON = {
    "pool.tier_quota.enabled": "true",
    "relations.enabled": "true", "relations.dunbar.enabled": "true",
    "friend_graph.enabled": "true", "joint.enabled": "true", "party.enabled": "true",
    "household.enabled": "true", "gossip.enabled": "true",
    "info_env.enabled": "true", "hierarchy.enabled": "true",
    "rumors.enabled": "true", "worldview.enabled": "true",
    "mobility.enabled": "true", "economy.accounts.enabled": "true",
    "assembly.enabled": "true", "diversity.enabled": "true",
    "truth_ledger.enabled": "true", "mind.enabled": "true",
}


def _run(tmp_path, small_pool, name, n_steps, use_ref, **ov):
    """AgentRef 差し替え あり/なし で 1 本走らせる(なし = 旧挙動 = フル Agent を残す)。"""
    saved = scheduler.AgentRef
    scheduler.AgentRef = AgentRef if use_ref else (lambda a: a)
    try:
        out = tmp_path / name
        sim = Simulation(_pool_cfg(name, small_pool, n_steps, **_ON, **ov), out_dir=out)
        sim.run()
        return sim, out
    finally:
        scheduler.AgentRef = saved


def _departed(sim):
    present = {a.id for a in sim.agents}
    return {aid: a for aid, a in sim.agent_by_id.items() if aid not in present}


@pytest.fixture(scope="module")
def ab_runs(small_pool, tmp_path_factory):
    """旧挙動 / 新挙動の 2 本(モジュール内で使い回す = mock ランを 2 本に抑える)。"""
    tmp = tmp_path_factory.mktemp("ab")
    old_sim, old_dir = _run(tmp, small_pool, "old", 440, use_ref=False)
    new_sim, new_dir = _run(tmp, small_pool, "new", 440, use_ref=True)
    return old_sim, old_dir, new_sim, new_dir


def test_departed_entries_become_light_refs(ab_runs):
    """回転で街を出た個体は名簿から消えず、**軽量参照**に化けている。"""
    _old_sim, _old_dir, new_sim, _new_dir = ab_runs
    dep = _departed(new_sim)
    assert dep, "退場者が 1 人も居ない(空回りの検査)"
    assert all(is_ref(a) for a in dep.values())
    assert all(not is_ref(a) for a in new_sim.agents)      # 在場はフル Agent のまま


def test_l1_is_byte_identical_to_the_full_agent_behaviour(ab_runs):
    """★A1 の検収本体: 差し替え あり/なし で L1 が 1 行も違わない。"""
    import pyarrow.parquet as pq
    _old_sim, old_dir, _new_sim, new_dir = ab_runs

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    a, b = rows(old_dir), rows(new_dir)
    assert len(a) > 1000, "L1 が小さすぎる(空回りの検査)"
    assert a == b, f"L1 バイト不一致: old={len(a)} new={len(b)}"


def test_coinage_author_and_sns_author_resolve_after_departure(ab_runs):
    """造語の作者名 / SNS の著者名が、作者が街を出た後も**同じ名前**で解決する。"""
    _old_sim, _old_dir, sim, _new_dir = ab_runs
    dep = _departed(sim)
    resolved = 0
    for item in sim.labels.items.items.values():
        who = sim.agent_by_id.get(item.creator)
        if who is not None and int(item.creator) in dep:
            assert isinstance(who.name, str) and who.name       # 名前が引ける
            resolved += 1
    for post in sim.net.posts:
        aid = post["author"]
        if aid in dep:
            assert sim.agent_by_id[aid].name
            resolved += 1
    assert resolved > 0, "退場した作者/著者が 1 件も無い(空回りの検査)"


def test_two_rotations_survive_departure_reentry_and_departure_again(ab_runs):
    """回転 2 周(退場 → 再入場 → 再退場)を通ってもエラーにならず、参照が保たれる。"""
    _old_sim, _old_dir, sim, _new_dir = ab_runs
    changes = [e for e in sim.logger.events if e.kind == "presence_change"]
    assert len(changes) >= 2, "日境界の回転が 2 回起きていない(空回りの検査)"
    assert sum(e.payload["n_exit"] for e in changes) > 0
    assert sum(e.payload["n_enter"] for e in changes) > 0
    # 名簿は「これまで実体化した全個体」= 在場 + 退場のどちらかで全員解決できる
    for aid, obj in sim.agent_by_id.items():
        assert obj.id == aid


def test_departed_pickle_shrinks(ab_runs):
    """★実測: 退場者を丸ごと pickle したバイト数が縮む(サイドカーが縮む理由そのもの)。"""
    old_sim, _old_dir, new_sim, _new_dir = ab_runs
    old_dep, new_dep = _departed(old_sim), _departed(new_sim)
    assert set(old_dep) == set(new_dep), "退場者の集合が食い違う(前提が崩れた)"
    b_old = len(pickle.dumps(old_dep, protocol=pickle.HIGHEST_PROTOCOL))
    b_new = len(pickle.dumps(new_dep, protocol=pickle.HIGHEST_PROTOCOL))
    assert b_new < b_old / 2.0, f"縮んでいない: old={b_old:,} new={b_new:,}"


def test_pool_sidecar_stores_refs(small_pool, tmp_path):
    """pool サイドカーに載る departed も軽量参照(= 保存側も縮む)。"""
    sim, out = _run(tmp_path, small_pool, "sc", 220, use_ref=True)
    sim._save_pool_sidecar(220)
    blob = sim._read_pool_sidecar(220)
    assert blob["departed"], "サイドカーに退場者が居ない(空回りの検査)"
    assert all(is_ref(a) for a in blob["departed"].values())


def test_old_sidecar_with_full_agents_is_converted_on_resume(small_pool, tmp_path):
    """★旧 checkpoint 互換: departed にフル Agent が入ったサイドカーを読んでも ref へ揃える。"""
    import gzip

    sim, out = _run(tmp_path, small_pool, "oldsc", 220, use_ref=False)
    dep = _departed(sim)
    assert dep and all(not is_ref(a) for a in dep.values())
    # 旧世代のサイドカーを手で作る(vital キーも departed の ref 化も無い)
    blob = {"pool_day": int(sim._pool_day), "dormant": sim._dormant._d,
            "dormant_cap": sim._dormant.cap, "departed": dep}
    path = sim._pool_sidecar_path(220)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    fresh = Simulation(_pool_cfg("oldsc2", small_pool, 220, **_ON),
                       out_dir=tmp_path / "oldsc2")
    fresh.__dict__.pop("_pool_sidecar_blob", None)
    fresh.agents = list(sim.agents)
    fresh.agent_by_id = {a.id: a for a in fresh.agents}
    fresh.out_dir = out
    fresh._restore_pool_resume(220)
    restored = {aid: a for aid, a in fresh.agent_by_id.items()
                if aid not in {x.id for x in fresh.agents}}
    assert restored, "退場者参照が 1 件も復元されていない"
    assert all(is_ref(a) for a in restored.values()), "旧サイドカーの ref 化が効いていない"
