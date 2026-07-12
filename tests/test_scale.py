"""スケール修正バッチ E1 のテスト(B1 空間索引 / B2 flush 分離 / B6 relations LRU /
B7 posts エイジアウト / B8 増分カウンタ)。

不変条件: 既定設定ではすべて従来と完全同一(バイト一致は test_scenario / test_resume が
担保)。ここでは (a) 空間索引 hearers_of が全対全 live 走査と内容・順序まで完全一致、
(b) flush_every の L1 一致、(c) LRU / エイジアウトの決定論挙動、を固定する。
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pyarrow.parquet as pq

from society.agents.memory import MemoryStore
from society.config import load_config
from society.engine.simulation import Simulation
from society.net.internet import Internet
from society.world.perception import PerceptIndex, build_index, hearers_of


# --------------------------------------------------------------------------- #
# B1: hearers_of 空間索引 == 全対全 live 走査(テスト内に参照実装をコピー)
# --------------------------------------------------------------------------- #
class _A:
    """hearers_of が読む最小フィールドだけを持つ合成エージェント。"""
    __slots__ = ("id", "loc", "building", "floor", "x", "y", "sleeping")

    def __init__(self, i, loc, building, floor, x, y, sleeping):
        self.id, self.loc, self.building, self.floor = i, loc, building, floor
        self.x, self.y, self.sleeping = x, y, sleeping


def _ref_context(a):
    if a.loc == "outside":
        return ("outside", a.id)
    if a.building:
        return ("bld", a.building, a.floor)
    return ("street",)


def _ref_hearers(speaker, agents, radius):
    """変更前の hearers_of の完全コピー(全対全走査)。"""
    ctx = _ref_context(speaker)
    result = []
    for other in agents:
        if other.id == speaker.id or other.sleeping or _ref_context(other) != ctx:
            continue
        if math.hypot(other.x - speaker.x, other.y - speaker.y) <= radius:
            result.append(other)
    return sorted(result, key=lambda a: a.id)


def _random_agents(n, seed=0):
    rng = random.Random(seed)
    buildings = [None, "b0", "b1", "b2", "b3"]     # None=路上/範囲外
    agents = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.15:
            loc, bld = "outside", None
        elif roll < 0.55:                          # 屋内: 建物+階でクラスタ
            loc = "street"
            bld = buildings[rng.randint(1, 4)]
        else:
            loc, bld = "street", None              # 路上
        floor = rng.randint(1, 3) if bld else 0
        if bld:                                    # 建物中心±60m(半径超も混ぜる)
            cx, cy = (int(bld[1]) * 300, int(bld[1]) * 300)
            x = cx + rng.uniform(-60, 60)
            y = cy + rng.uniform(-60, 60)
        else:
            x = rng.uniform(-500, 500)
            y = rng.uniform(-500, 500)
        sleeping = rng.random() < 0.1
        agents.append(_A(i, loc, bld, floor, x, y, sleeping))
    return agents


def test_hearers_index_matches_bruteforce_all_contexts():
    """1,000体×複数半径で、索引 hearers が全対全 live 走査と内容・順序まで完全一致。"""
    agents = _random_agents(1000, seed=7)
    for radius in (20.0, 40.0, 80.0):
        idx = build_index(agents, radius)
        assert isinstance(idx, PerceptIndex)
        for sp in agents:
            ref = [a.id for a in _ref_hearers(sp, agents, radius)]
            via_index = [a.id for a in hearers_of(sp, idx, radius)]
            via_live = [a.id for a in hearers_of(sp, agents, radius)]
            assert via_index == ref, (sp.id, radius, "index != bruteforce")
            assert via_live == ref, (sp.id, radius, "live path drifted")


def test_hearers_outside_never_hears():
    """範囲外(outside)は誰も聞こえない(索引・live 両方で空)。"""
    agents = [
        _A(0, "outside", None, 0, 0.0, 0.0, False),
        _A(1, "outside", None, 0, 0.0, 0.0, False),   # 同座標でも別 context
        _A(2, "street", None, 0, 0.0, 0.0, False),
    ]
    idx = build_index(agents, 40.0)
    assert hearers_of(agents[0], idx, 40.0) == []
    assert hearers_of(agents[0], agents, 40.0) == []


def test_hearers_sleeping_excluded():
    """睡眠中の agent は聞き手にならない(索引から除外)。"""
    agents = [
        _A(0, "street", None, 0, 0.0, 0.0, False),
        _A(1, "street", None, 0, 5.0, 0.0, True),     # sleeping
        _A(2, "street", None, 0, 5.0, 0.0, False),
    ]
    idx = build_index(agents, 40.0)
    assert [a.id for a in hearers_of(agents[0], idx, 40.0)] == [2]


# --------------------------------------------------------------------------- #
# B2: flush_every_steps は checkpoint と独立に L1 を part 化し、finalize で結合すると
#     一気ラン(flush なし)と完全一致する。
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps, **ov):
    dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir, stem="l1_events"):
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _run(tmp_path, name, n_steps, **ov):
    d = tmp_path / name
    Simulation(_cfg(name, n_steps, **ov), out_dir=d).run()
    return d


def test_flush_every_matches_plain(tmp_path):
    """flush_every_steps=12 は一気ラン(既定 0)と L1/L2/L3 が完全一致(part は結合済み)。"""
    plain = _run(tmp_path, "fe_plain", 40)
    flushed = _run(tmp_path, "fe_flush", 40, **{"observer.flush_every_steps": 12})
    assert _rows(plain, "l1_events") == _rows(flushed, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(plain, stem) == _rows(flushed, stem), f"{stem} 不一致"
    assert not list(Path(flushed).glob("l1_events.part-*.parquet")), \
        "finalize 後も part が残っている"
    assert (Path(flushed) / "l1_events.parquet").exists()


def test_flush_every_zero_is_default(tmp_path):
    """flush_every_steps=0(既定)は part を一切作らない。"""
    d = _run(tmp_path, "fe_zero", 24, **{"observer.flush_every_steps": 0})
    assert not list(Path(d).glob("l1_events.part-*.parquet"))


# --------------------------------------------------------------------------- #
# B6: relations 台帳の LRU(last_step 最古、同点は相手id小)。既定 0 は無制限。
# --------------------------------------------------------------------------- #
def test_relations_unbounded_by_default():
    m = MemoryStore()                              # relations_max=0
    for i in range(200):
        m.record_contact(i, f"n{i}", i)
    assert len(m.relations) == 200                 # 退避されない


def test_relations_max_lru_eviction():
    m = MemoryStore(relations_max=3)
    m.record_contact(1, "a", 1)
    m.record_contact(2, "b", 2)
    m.record_contact(3, "c", 3)
    assert set(m.relations) == {1, 2, 3}
    m.record_contact(1, "a", 4)                    # id1 を最新化(last_step=4)
    m.record_contact(4, "d", 5)                    # 超過 → 最古(id2, last_step=2)を退避
    assert set(m.relations) == {1, 3, 4}, set(m.relations)
    m.record_contact(5, "e", 6)                    # 最古(id3, last_step=3)を退避
    assert set(m.relations) == {1, 4, 5}, set(m.relations)
    assert len(m.relations) == 3


# --------------------------------------------------------------------------- #
# B7/B8: posts エイジアウト(id 不変)+ react/timeline の id ベース安全性、増分カウンタ。
# --------------------------------------------------------------------------- #
def test_posts_unbounded_by_default():
    """posts_max=0(既定)は id==位置・trim なし(従来と完全同一)。"""
    net = Internet(feed_size=3)                    # posts_max=0
    ids = [net.post(0, f"t{i}", [], i) for i in range(10)]
    assert ids == list(range(10))
    assert len(net.posts) == 10
    assert net.posts[7]["id"] == 7                 # 位置==id


def test_posts_max_ages_out_but_ids_stable():
    net = Internet(feed_size=3, posts_max=5)
    ids = [net.post(0, f"t{i}", [], i) for i in range(10)]
    assert ids == list(range(10))                  # id は追記単調(不変)
    assert len(net.posts) == 5                     # 直近5件のみ保持
    assert [p["id"] for p in net.posts] == [5, 6, 7, 8, 9]
    # 退避済み id への react は安全に None(index ずれで別 post を叩かない)
    assert net.react(1, 0, "like", 0) is None
    # 生存 id への react は正しく効き、増分カウンタが進む
    assert net.react(1, 9, "like", 0) == 0
    assert net.n_likes_total == 1
    assert 1 in net.posts[-1]["likes"]
    # 同じ reader の二重いいねはカウントされない(set 意味を維持)
    assert net.react(1, 9, "like", 0) == 0
    assert net.n_likes_total == 1


def test_posts_max_timeline_safe_after_ageout():
    """read_marks が退避境界より古くても timeline_for が index エラーにならない。"""
    net = Internet(feed_size=3, posts_max=5)
    net.follows = {1: set()}                        # reader1 はフォローなし
    for i in range(3):
        net.post(0, f"t{i}", [], i)
    tl = net.timeline_for(1)                        # ids 0,1,2 を読み read_marks=3
    assert [p["id"] for p in tl] == [0, 1, 2]
    for i in range(3, 10):                          # 追加で id0..4 が退避
        net.post(0, f"t{i}", [], i)
    tl2 = net.timeline_for(1)                       # start(3) < offset(5) でも安全
    assert [p["id"] for p in tl2] == [7, 8, 9]      # feed_size=3 の直近


def test_reshare_increments_counter():
    net = Internet(feed_size=3)
    pid = net.post(0, "hello", [], 0)
    net.react(1, pid, "reshare", 0, author_name="A")
    assert net.n_reshares_total == 1
    assert net.posts[pid]["reshares"] == 1
    assert net.posts[-1]["text"].startswith("RT @A:")
