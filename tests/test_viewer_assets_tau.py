# 第133 ウェーブ2: viewer の資産τ差し替え(O(n²)→observer/assets.py の O(n log n))の同値性。
# 参照実装 = 差し替え前の make_viewer._assets_tau(1バイトも変えず保持)。
import random

from viz.make_viewer import _assets_tau


def _ref_assets_tau(cur: dict, prev: dict | None):
    """差し替え前の viewer 実装(明示二重ループ)を逐語保持した参照実装。"""
    if not prev:
        return None
    common = sorted(set(cur) & set(prev))
    if len(common) < 2:
        return None
    a = [cur[i] for i in common]
    b = [prev[i] for i in common]
    concord = discord = tie_a = tie_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concord += 1
            elif s < 0:
                discord += 1
            else:
                if a[i] == a[j]:
                    tie_a += 1
                if b[i] == b[j]:
                    tie_b += 1
    n0 = len(a) * (len(a) - 1) / 2.0
    denom = ((n0 - tie_a) * (n0 - tie_b)) ** 0.5
    if denom == 0:
        return 0.0
    return round((concord - discord) / denom, 6)


def test_none_conventions_match():
    assert _assets_tau({}, None) is None
    assert _assets_tau({1: 5.0}, {}) is None
    assert _assets_tau({1: 5.0}, {2: 3.0}) is None          # 共通 id < 2
    assert _assets_tau({1: 5.0, 2: 1.0}, {1: 5.0}) is None  # 共通 id < 2


def test_constant_side_is_zero():
    cur = {i: 100.0 for i in range(10)}
    prev = {i: float(i) for i in range(10)}
    assert _assets_tau(cur, prev) == _ref_assets_tau(cur, prev) == 0.0


def test_random_wealth_matches_the_old_double_loop():
    rng = random.Random(133)
    for trial in range(20):
        ids = rng.sample(range(1000), 80)
        cur = {i: round(rng.uniform(0, 5e5), 2) for i in ids}
        prev = {i: round(rng.uniform(0, 5e5), 2) for i in rng.sample(range(1000), 90)}
        assert _assets_tau(cur, prev) == _ref_assets_tau(cur, prev)


def test_tie_heavy_wealth_matches_the_old_double_loop():
    rng = random.Random(134)
    for trial in range(20):
        ids = list(range(60))
        # 0円の塊 + 少数の離散値 = tie 多発(wealth 実データの形)
        cur = {i: rng.choice([0.0, 0.0, 0.0, 100.0, 250.0]) for i in ids}
        prev = {i: rng.choice([0.0, 0.0, 150.0, 250.0]) for i in ids}
        assert _assets_tau(cur, prev) == _ref_assets_tau(cur, prev)
