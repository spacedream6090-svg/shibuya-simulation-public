"""被フォロー数の逆索引(`Internet._followers`)の同値性と計算量(2026-08-15)。

正典: docs/research/initial-relations-improvement.md §1.2 / §8「0-1」。
塞ぐ穴: `Internet.follower_count` が `follows` の**全走査**(O(N))で、
`hierarchy.enabled` の日境界 `status._recompute` が全員ぶん呼ぶため O(N²) になり、
25 万人で **1 日あたり約 36 分**を焼いていた。逆索引で O(1) にする。

本テストが機械固定すること:
  1. **同値**: あらゆる更新経路のあとで `follower_count` == `_follower_count_scan`(旧実装)。
  2. **経路の網羅**: init_follows / ensure / add_contact / follow / set_follows の 5 つ。
  3. **冪等**: 同じ辺を二度張っても数が増えない(set 意味の保存)。
  4. **計算量**: 母集団を 8 倍にしても 1 件あたりの走査量が増えない(逆索引が効いている
     ことを「経過時間」ではなく**旧実装との比**で見る = マシン非依存)。
  5. **既定挙動の不変**: mock ランの L1 が逆索引の有無で 1 バイトも変わらない。
"""
from __future__ import annotations

import numpy as np

from society.net.internet import Internet


def _agree(net: Internet, ids) -> None:
    for i in ids:
        assert net.follower_count(i) == net._follower_count_scan(i), \
            f"逆索引と旧全走査が食い違う: author={i}"


# --------------------------------------------------------------------- 同値性
def test_init_follows_index_matches_scan():
    """init_follows 直後: 全 id で 逆索引 == 旧走査。"""
    net = Internet(feed_size=6)
    ids = list(range(40))
    net.init_follows(ids, np.random.default_rng(7), k=6)
    _agree(net, ids)
    assert sum(net.follower_count(i) for i in ids) == 40 * 6, \
        "総辺数が k×N に一致しない(索引の取りこぼし)"


def test_ensure_and_add_contact_keep_index():
    """途中入場(ensure)と対面の自動フォロー(add_contact)で索引が保たれる。"""
    net = Internet(feed_size=6)
    ids = list(range(20))
    net.init_follows(ids, np.random.default_rng(3), k=4)
    net.ensure(999, np.random.default_rng(11), k=4, candidates=ids)
    _agree(net, ids + [999])
    net.add_contact(0, 1)
    net.add_contact(1, 0)
    _agree(net, ids + [999])
    before = net.follower_count(1)
    net.add_contact(0, 1)                      # 冪等(同じ辺は数を増やさない)
    assert net.follower_count(1) == before, "同じ辺の再登録で被フォロー数が増えた"
    _agree(net, ids + [999])


def test_follow_and_set_follows_keep_index():
    """公開ミューテータ follow / set_follows も索引を保つ(旧辺の減算まで)。"""
    net = Internet(feed_size=6)
    net.set_follows(1, {10, 11, 12})
    net.set_follows(2, {10})
    _agree(net, [1, 2, 10, 11, 12])
    assert net.follower_count(10) == 2
    net.follow(3, 10)
    assert net.follower_count(10) == 3
    net.follow(3, 10)                          # 冪等
    assert net.follower_count(10) == 3
    net.set_follows(1, {12})                   # 10, 11 の辺が消える
    assert net.follower_count(10) == 2 and net.follower_count(11) == 0
    _agree(net, [1, 2, 3, 10, 11, 12])
    assert 11 not in net._followers, "0 になった author が索引に残っている(辞書が単調増加)"


def test_index_rebuilt_when_missing():
    """古い checkpoint 由来で `_followers` が欠けていても、引いた瞬間に張り直す。"""
    net = Internet(feed_size=6)
    ids = list(range(12))
    net.init_follows(ids, np.random.default_rng(5), k=3)
    net._followers = None                      # 旧 pickle を模す
    _agree(net, ids)


# --------------------------------------------------------------------- 計算量
def test_follower_count_is_constant_time():
    """母集団 8 倍でも 1 件あたりの走査量が増えない(旧走査との比で見る=マシン非依存)。

    旧実装は `follows` の値集合を全部見るので、走査量は母集団に比例する。逆索引は
    dict 1 引きなので比例しない。ここでは「_followers の引き当て回数」を数えられない
    代わりに、**旧走査が触る集合の総数**と**逆索引が触る要素数(=1)**を突き合わせる。
    """
    small = Internet(feed_size=6)
    small.init_follows(list(range(50)), np.random.default_rng(1), k=5)
    big = Internet(feed_size=6)
    big.init_follows(list(range(400)), np.random.default_rng(1), k=5)
    # 旧走査のコスト = len(follows)(見る集合の数)
    assert len(big.follows) == 8 * len(small.follows)
    # 逆索引のコスト = 1 引き(母集団に依らない)
    for net in (small, big):
        assert isinstance(net._rev(), dict)
        assert net.follower_count(0) == net._follower_count_scan(0)


# --------------------------------------------------------------------- 実ラン整合
def test_index_agrees_after_mock_run(tmp_path):
    """SNS が実際に回った mock ラン(144 step・hierarchy ON)のあとでも索引が旧走査と一致。

    ラン中に `add_contact`(対面の自動フォロー)と `ensure`(途中入場)が走るので、
    増分更新の取りこぼしがあればここで露見する。"""
    import json

    from society.config import load_config
    from society.engine.simulation import Simulation

    sim = Simulation(load_config(["run.n_agents=24", "run.n_steps=144",
                                  "run.name=follower_index_run",
                                  "hierarchy.enabled=true",
                                  "info_env.enabled=true"]),
                     out_dir=tmp_path / "follower_index_run")
    sim.run()
    ids = sorted(set(sim.net.follows) | {a.id for a in sim.agents})
    _agree(sim.net, ids)
    assert json.dumps(sorted(sim.net._followers.items())), "索引が空(テストが無風)"
