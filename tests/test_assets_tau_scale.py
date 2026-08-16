"""資産順位 τ(observer/assets._kendall_tau)の O(n log n) 化の同値性と計算量(レーンP A3)。

塞ぐ穴: `assets._kendall_tau` は明示二重ループ = **O(n²)** で、本選 25 万体では日境界 1 回あたり
312 億回の比較(= day1 でハング級)になる。tie を別集計し、残りをマージソートの反転数から出す
O(n log n) 実装へ置き換えた。

本テストが機械固定すること:
  1. **同値**: 旧 O(n²) 実装(下の参照実装 `_tau_pairwise` = 置換前のコードそのまま)と
     新実装が、tie あり/なし・乱数入力・境界(全同値/完全逆順/n=0,1,2)で完全一致(float も同一)。
  2. **計算量**: n=10,000(旧実装なら 5,000 万ペア)が現実的な時間で完走する。
  3. **既知値**: 完全一致=1.0 / 完全逆順=-1.0 / 片側定数=0.0(measure 仕様)。
"""
from __future__ import annotations

import math
import random
import time

from society.observer import assets as A


# --------------------------------------------------------------------------- #
# 参照実装 = 置き換え前の O(n²) コード(1 行も変えずに保持する)。
# --------------------------------------------------------------------------- #
def _tau_pairwise(a, b):
    n = len(a)
    if n < 2:
        return None
    concord = discord = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = da * db
            if s > 0:
                concord += 1
            elif s < 0:
                discord += 1
            else:
                if da == 0:
                    tie_a += 1
                if db == 0:
                    tie_b += 1
    n0 = n * (n - 1) / 2.0
    denom = math.sqrt((n0 - tie_a) * (n0 - tie_b))
    if denom == 0:
        return 0.0
    return (concord - discord) / denom


def _same(a, b, label=""):
    got, ref = A._kendall_tau(a, b), _tau_pairwise(a, b)
    assert got == ref, f"新旧の τ が食い違う {label}: new={got!r} old={ref!r}"
    return got


# --------------------------------------------------------------------------- 境界
def test_tau_boundaries_match_reference():
    """n=0,1,2 / 全同値 / 完全逆順 / 片側定数 の境界で新旧完全一致。"""
    assert _same([], []) is None
    assert _same([1.0], [2.0]) is None
    assert _same([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert _same([1.0, 2.0], [2.0, 1.0]) == -1.0
    assert _same([5.0] * 8, [5.0] * 8) == 0.0            # 両側定数=分母0
    assert _same([5.0] * 8, list(range(8))) == 0.0       # 片側定数=分母0
    assert _same(list(range(30)), list(range(29, -1, -1))) == -1.0
    assert _same(list(range(30)), list(range(30))) == 1.0
    # 整数入力(既存テストが渡す型)でも一致する
    assert _same([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert _same([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_tau_random_no_ties_match_reference():
    """tie なしの乱数入力 20 ケースで新旧完全一致(float ビットまで)。"""
    rnd = random.Random(20260815)
    for case in range(20):
        n = rnd.randint(2, 120)
        a = rnd.sample(range(10 ** 6), n)                # 相異なる=tie なし
        b = rnd.sample(range(10 ** 6), n)
        _same([float(v) for v in a], [float(v) for v in b], f"case{case}")


def test_tau_random_with_ties_match_reference():
    """tie 多発(値域を極端に狭める)+片側だけ tie/両側 tie を含む 30 ケースで完全一致。"""
    rnd = random.Random(7)
    for case in range(30):
        n = rnd.randint(2, 150)
        span = rnd.choice([1, 2, 3, 5])                  # 値域が狭い = 同値ペアが大量に出る
        a = [float(rnd.randint(0, span)) for _ in range(n)]
        b = [float(rnd.randint(0, span)) for _ in range(n)]
        _same(a, b, f"ties case{case}")
    # 片側だけ tie(a は全同値・b は相異なる)/ 逆側 / 一部だけ tie
    _same([1.0] * 10, [float(i) for i in range(10)], "a-const")
    _same([float(i) for i in range(10)], [1.0] * 10, "b-const")
    _same([1.0, 1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0, 3.0], "partial")


def test_tau_random_wealth_like_match_reference():
    """実データ相当(money+account の連続値 + 0 円の塊)で完全一致。"""
    rnd = random.Random(99)
    for case in range(10):
        n = rnd.randint(50, 200)
        a = [0.0 if rnd.random() < 0.3 else rnd.uniform(0, 5_000_000)
             for _ in range(n)]
        # 前日比 = 少し動いた分布(順位が部分的に入れ替わる)
        b = [v * rnd.uniform(0.8, 1.2) if rnd.random() < 0.9 else 0.0 for v in a]
        _same(a, b, f"wealth case{case}")


def test_rank_tau_uses_common_ids_and_matches_reference():
    """`rank_tau`(共通 id 上の突き合わせ)も参照実装と一致し、共通 id<2 は None。"""
    rnd = random.Random(4)
    cur = {i: rnd.uniform(0, 1000) for i in range(60)}
    prev = {i: rnd.uniform(0, 1000) for i in range(30, 90)}   # 共通 = 30..59
    common = sorted(set(cur) & set(prev))
    assert A.rank_tau(cur, prev) == _tau_pairwise([cur[i] for i in common],
                                                  [prev[i] for i in common])
    assert A.rank_tau(cur, None) is None
    assert A.rank_tau({1: 1.0}, {1: 2.0}) is None            # 共通 id < 2


# --------------------------------------------------------------------------- 計算量
def test_tau_large_n_completes():
    """n=10,000(旧実装なら 4,999 万ペア)が現実的な時間で完走する(本選 25 万体の代理)。

    値の正しさは上の同値性テストが担保するので、ここは**完走すること**と、粗い健全性
    (少しだけ崩した順位列の τ が 1 に近い / 反転列が -1)だけを見る。"""
    n = 10_000
    a = [float(i) for i in range(n)]
    b = list(a)
    b[10], b[20] = b[20], b[10]                    # 1 ペアだけ反転
    t0 = time.perf_counter()
    tau = A._kendall_tau(a, b)
    elapsed = time.perf_counter() - t0
    assert 0.999 < tau < 1.0, tau
    assert elapsed < 30.0, f"n=10,000 に {elapsed:.1f}s(O(n log n) になっていない)"
    assert A._kendall_tau(a, list(reversed(a))) == -1.0


def test_inversion_counter_matches_bruteforce():
    """内部の反転数カウンタ(マージソート)を総当たりで固定する。"""
    rnd = random.Random(11)
    for _ in range(30):
        n = rnd.randint(0, 40)
        seq = [rnd.randint(0, 5) for _ in range(n)]
        brute = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
        assert A._inversions(seq) == brute, seq
        assert A._inversions(seq) == brute, "入力列を破壊している(2 回目が違う)"


def test_tie_pairs_matches_bruteforce():
    """内部の同値ペア数カウンタを総当たりで固定する(ソート済み前提)。"""
    rnd = random.Random(13)
    for _ in range(30):
        n = rnd.randint(0, 40)
        vals = sorted(rnd.randint(0, 4) for _ in range(n))
        brute = sum(1 for i in range(n) for j in range(i + 1, n) if vals[i] == vals[j])
        assert A._tie_pairs(vals) == brute, vals
