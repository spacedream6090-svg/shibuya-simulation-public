"""SNS の無界走査を塞ぐ 2 件(レーンP A5 / C1)の同値性と計算量。

A5 既読 watermark: `Internet.ensure`(途中入場者の配線)が `read_marks` を据えないため、
  read_marks=0 の個体の初回閲覧が **posts 全履歴の走査**になっていた(25 万体・30 日ランでは
  途中入場 20.9 万人 × 数十万投稿)。入場時点(= 次に付く post id)へ据える。意味の面でも
  「街に来る前のタイムラインを遡って読む」は現実の挙動ではない(スマホを開いた時点の新着から)。

C1 初期フォロー: `Internet.init_follows` は 1 人ぶんごとに `others = [x for x in ids if x != aid]`
  を作り直す **O(N²)**(25 万人で実測 0.011 秒 × 25 万 ≒ 45 分の起動コスト)。抽選
  `rng.choice(n_pop, size=n, replace=False)` は母集団の大きさと本数だけから乱数を引くので、
  添字 → 実 id の写像(自分の位置以上なら +1)に置き換えても **乱数消費列も選ばれる集合も不変**。

本テストが機械固定すること:
  1. init_follows: 旧実装(下の参照実装)と follows が完全一致し、**rng の内部状態まで一致**する。
  2. init_follows: 母集団 N を増やしても旧実装との比で明確に速い(マシン非依存の比で見る)。
  3. ensure: 既読 watermark が末尾に据わり、初回閲覧が posts の先頭から舐めない(スライス開始位置)。
  4. ensure: 既に read_marks を持つ個体には触れない(冪等)。trim(posts_max>0)下でも id 基準で正しい。
  5. 既定プロファイル(プール回転 OFF)では ensure が 1 度も呼ばれない = 基底は不変。
"""
from __future__ import annotations

import time

import numpy as np

from society.net.internet import Internet


# --------------------------------------------------------------------------- #
# 参照実装 = 置き換え前の init_follows(1 行も変えずに保持する)。
# --------------------------------------------------------------------------- #
def _init_follows_old(ids, rng, k=6) -> dict:
    follows = {}
    for aid in ids:
        others = [x for x in ids if x != aid]
        n = min(k, len(others))
        picks = rng.choice(len(others), size=n, replace=False) if n else []
        follows[aid] = {others[int(i)] for i in picks}
    return follows


def _state(rng) -> str:
    return repr(rng.bit_generator.state)


def _both(ids, k, seed=0):
    """新旧を同じ seed で走らせ (新 follows, 旧 follows, 新 rng 状態, 旧 rng 状態) を返す。"""
    net = Internet(feed_size=6)
    r_new = np.random.default_rng(seed)
    net.init_follows(list(ids), r_new, k=k)
    r_old = np.random.default_rng(seed)
    old = _init_follows_old(list(ids), r_old, k=k)
    return net.follows, old, _state(r_new), _state(r_old)


# --------------------------------------------------------------------- C1 同値性
def test_init_follows_matches_old_implementation():
    """連番 id・非連番 id・シャッフル済み名簿で follows が完全一致し、乱数消費列も一致。"""
    cases = [
        (list(range(30)), 6),
        (list(range(200)), 6),
        (list(range(50)), 1),
        (list(range(50)), 49),                    # k = N-1(全員をフォロー)
        (list(range(50)), 100),                   # k > N-1(min で頭打ち)
        ([7, 3, 99, 1, 40, 5, 22], 3),            # 非連番・非ソート
        ([1000 + i * 3 for i in range(80)], 6),   # 飛び飛びの id
    ]
    for ids, k in cases:
        new, old, s_new, s_old = _both(ids, k, seed=11)
        assert new == old, f"follows が旧実装と食い違う ids={ids[:5]}... k={k}"
        assert s_new == s_old, f"rng の消費列が変わった ids={ids[:5]}... k={k}"
        for aid, targets in new.items():
            assert aid not in targets, "自分自身をフォローしている"
            assert targets <= set(ids), "名簿外の id をフォローしている"
            assert len(targets) == min(k, len(ids) - 1), "本数が k(または N-1)でない"


def test_init_follows_edge_cases_match():
    """境界(N=0 / N=1 / k=0)でも旧実装と完全一致(例外も同じ)。"""
    for ids, k in [([], 6), ([5], 6), (list(range(10)), 0)]:
        new, old, s_new, s_old = _both(ids, k, seed=3)
        assert new == old and s_new == s_old, f"境界 ids={ids} k={k}"
    net = Internet(feed_size=6)
    net.init_follows([], np.random.default_rng(1), k=6)
    assert net.follows == {} and net._followers == {}
    net.init_follows([5], np.random.default_rng(1), k=6)
    assert net.follows == {5: set()}, "1 人だけの名簿で自己フォローが生えた"


def test_init_follows_duplicate_ids_fall_back_to_old_path():
    """名簿に重複 id があるとき(退避経路)も旧実装と完全一致。"""
    ids = [1, 2, 2, 3, 4, 4, 4, 5]
    new, old, s_new, s_old = _both(ids, 3, seed=9)
    assert new == old, "重複 id の退避経路が旧実装と食い違う"
    assert s_new == s_old, "重複 id の退避経路で乱数消費が変わった"


def test_init_follows_keeps_contacts_and_follower_index():
    """contacts の初期化と逆索引の張り直し(第116)が壊れていない。"""
    net = Internet(feed_size=6)
    ids = list(range(60))
    net.init_follows(ids, np.random.default_rng(2), k=4)
    assert all(net.contacts[i] == set() for i in ids), "contacts が初期化されていない"
    for i in ids:
        assert net.follower_count(i) == net._follower_count_scan(i), "逆索引が旧走査と不一致"
    assert sum(net.follower_count(i) for i in ids) == 60 * 4, "総辺数 k×N が合わない"


# --------------------------------------------------------------------- C1 計算量
def test_init_follows_is_linear_in_population():
    """母集団を増やしたとき、旧実装との**比**で明確に速い(マシン非依存)。

    旧実装は 1 人ぶんごとに N-1 件のリストを作るので O(N²)。新実装は O(N·k)。
    ここでは同じマシン・同じ試行で両者を測り、比だけを見る(絶対時間は見ない)。"""
    ids = list(range(4000))
    t0 = time.perf_counter()
    _init_follows_old(ids, np.random.default_rng(1), k=6)
    old_sec = time.perf_counter() - t0
    net = Internet(feed_size=6)
    t0 = time.perf_counter()
    net.init_follows(ids, np.random.default_rng(1), k=6)
    new_sec = time.perf_counter() - t0
    assert new_sec * 3 < old_sec, \
        f"O(N²) のままに見える(旧 {old_sec:.3f}s / 新 {new_sec:.3f}s)"
    assert len(net.follows) == len(ids)


# --------------------------------------------------------------------- A5 既読 watermark
class _WatchedPosts(list):
    """スライス開始位置を記録する posts(= 初回閲覧がどこから舐めたかの証拠)。"""

    def __init__(self, *a):
        super().__init__(*a)
        self.slice_starts: list = []

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.slice_starts.append(item.start)
        return super().__getitem__(item)


def test_ensure_sets_read_marks_to_tail():
    """ensure 直後の read_marks は「次に付く post id」= 入場時点。初回閲覧は先頭から舐めない。"""
    net = Internet(feed_size=6)
    net.init_follows(list(range(10)), np.random.default_rng(1), k=3)
    for i in range(200):
        net.post(0, f"t{i}", [], i)
    net.ensure(999, np.random.default_rng(5), k=3, candidates=list(range(10)))
    assert net.read_marks[999] == 200, "既読 watermark が入場時点に据わっていない"
    net.posts = _WatchedPosts(net.posts)
    assert net.timeline_for(999) == [], "入場前の投稿がタイムラインに載っている"
    assert net.posts.slice_starts == [200], \
        f"初回閲覧が全履歴を走査している(開始位置 {net.posts.slice_starts})"


def test_first_timeline_after_ensure_sees_only_new_posts():
    """入場後に流れた投稿だけが初回タイムラインに載る。"""
    net = Internet(feed_size=6)
    net.init_follows(list(range(5)), np.random.default_rng(1), k=2)
    for i in range(40):
        net.post(0, f"old{i}", [], i)
    net.ensure(777, np.random.default_rng(5), k=2, candidates=list(range(5)))
    new_ids = [net.post(1, f"new{i}", [], 100 + i) for i in range(3)]
    assert [p["id"] for p in net.timeline_for(777)] == new_ids


def test_ensure_does_not_touch_existing_read_marks():
    """既に read_marks を持つ個体・既に follows を持つ個体には触れない(冪等)。"""
    net = Internet(feed_size=6)
    net.init_follows([1, 2, 3], np.random.default_rng(1), k=2)
    for i in range(10):
        net.post(1, f"t{i}", [], i)
    net.read_marks[2] = 4                       # 途中まで読んだ既存個体
    net.ensure(2, np.random.default_rng(5), k=2, candidates=[1, 3])
    assert net.read_marks[2] == 4, "既存個体の既読位置が書き換えられた"
    net.ensure(888, np.random.default_rng(5), k=2, candidates=[1, 2, 3])
    net.ensure(888, np.random.default_rng(5), k=2, candidates=[1, 2, 3])   # 2 度目=no-op
    assert net.read_marks[888] == 10
    net.post(1, "after", [], 11)
    net.ensure(888, np.random.default_rng(5), k=2, candidates=[1, 2, 3])
    assert net.read_marks[888] == 10, "再入場で既読位置が末尾へ飛んだ(冪等でない)"


def test_ensure_read_marks_under_trim():
    """posts_max>0(エイジアウト)でも watermark は **post id 基準**で正しい(offset を含む)。"""
    net = Internet(feed_size=3, posts_max=5)
    net.init_follows([1], np.random.default_rng(1), k=0)
    for i in range(10):
        net.post(1, f"t{i}", [], i)             # id 0..9 / 生存 5..9 / offset=5
    assert net._post_offset == 5
    net.ensure(555, np.random.default_rng(5), k=1, candidates=[1])
    assert net.read_marks[555] == 10, "trim 下で watermark が位置(len)になっている(id でない)"
    net.posts = _WatchedPosts(net.posts)
    assert net.timeline_for(555) == []
    assert net.posts.slice_starts == [5], "trim 下で先頭から舐めている"


def test_likes_and_news_have_no_bound_documented():
    """★現状の実測メモ(コード変更なし): trim があるのは posts だけ。likes / news は無界。

    - `posts_max>0` で posts はエイジアウトする(tests/test_scale.py の B7/B8 が固定済み)が、
      **生き残っている post の `likes` 集合**は上限が無く、人気投稿では読者数まで膨らむ。
    - `news` は `publish_news` が追記するだけで一度も刈られない(`latest_news` は末尾を見るだけ)。
    news/likes の有界化は挙動(いいね総数・既存の観測量)を動かすので**別判断**とし、ここでは
    「無界である」という事実だけを機械固定する(将来上限を入れたらこのテストが赤になって気づく)。
    """
    net = Internet(feed_size=3, posts_max=5)
    pid = net.post(0, "viral", [], 0)
    for reader in range(500):
        net.react(reader, pid, "like", 0)
    assert len(net.posts[0]["likes"]) == 500, "likes に上限が入った(別判断のはず)"
    assert net.n_likes_total == 500
    for i in range(300):
        net.publish_news(f"n{i}", "text", [], i)
    assert len(net.news) == 300, "news に上限が入った(別判断のはず)"
    assert len(net.posts) == 5, "posts の trim(posts_max)が効いていない"


# --------------------------------------------------------------------- 基底(pool OFF)不変
def test_default_run_never_calls_ensure(tmp_path, monkeypatch):
    """既定プロファイル(プール回転 OFF)では ensure が 1 度も呼ばれない = 基底は不変。"""
    from society.config import load_config
    from society.engine.simulation import Simulation

    calls = {"n": 0}
    real = Internet.ensure

    def counting(self, aid, rng, k=6, candidates=None):
        calls["n"] += 1
        return real(self, aid, rng, k=k, candidates=candidates)

    monkeypatch.setattr(Internet, "ensure", counting)
    sim = Simulation(load_config(["run.n_agents=12", "run.n_steps=72",
                                  "run.name=ensure_off", "info_env.enabled=true"]),
                     out_dir=tmp_path / "ensure_off")
    sim.run()
    assert calls["n"] == 0, f"既定プロファイルで ensure が {calls['n']} 回呼ばれた"
