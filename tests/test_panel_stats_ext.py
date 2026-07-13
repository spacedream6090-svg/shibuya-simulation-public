"""⑦統計的有意性の拡張 純関数テスト(第30バッチ・分析スイート W1)。

scripts/panel_stats.py に追加した 効果量(Cohen's d・Cliff's δ)・置換検定 p・
BH-FDR 補正 q・t分布CI を、既知の小データで手計算値と一致することを検証する。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import panel_stats as ps   # noqa: E402


def test_t_crit_and_t_ci():
    """t 分位点表と t分布CI(n=5・df=4・t=2.776)。"""
    assert ps.t_crit(4) == 2.776
    assert ps.t_crit(1) == 12.706
    assert ps.t_crit(200) == 1.96                  # df>30 は正規近似
    mean, lo, hi, n, sd = ps.t_ci([1, 2, 3, 4, 5])
    assert mean == 3 and n == 5
    assert abs(sd - math.sqrt(2.5)) < 1e-9          # 標本sd(ddof=1)= √2.5
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)    # = 1.9629...
    assert abs(lo - (3 - half)) < 1e-6
    assert abs(hi - (3 + half)) < 1e-6
    # n≥31 では t_ci と 既存 _mean_ci(正規近似)が一致する
    xs = list(range(40))
    assert abs(ps.t_ci(xs)[1] - ps._mean_ci(xs)[1]) < 1e-9


def test_paired_cohens_d():
    """対応差 Cohen's d = mean/sd。差が一定(sd=0)は None。"""
    assert abs(ps.paired_cohens_d([1, 2, 3]) - 2.0) < 1e-12   # mean2, sd1
    assert ps.paired_cohens_d([2, 2, 2, 2]) is None           # sd=0 → 定義不能
    assert ps.paired_cohens_d([5]) is None                    # n<2


def test_cliffs_delta():
    """Cliff's δ:完全分離=±1・同一分布=0。"""
    assert ps.cliffs_delta([1, 2, 3], [4, 5, 6]) == 1.0       # ys 全て大
    assert ps.cliffs_delta([3, 3], [1, 1]) == -1.0            # ys 全て小
    assert ps.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0       # 対称 → 0
    assert ps.cliffs_delta([], [1]) is None


def test_perm_test_paired_exact():
    """符号反転の全列挙(n≤12)。手計算値と一致。"""
    # diffs=[1,2,3]: |mean|>=2 は (+++)と(---) の2/8 = 0.25
    assert abs(ps.perm_test_paired([1, 2, 3]) - 0.25) < 1e-12
    # diffs=[1,1,1,1]: |mean|>=1 は 全+と全− の2/16 = 0.125
    assert abs(ps.perm_test_paired([1, 1, 1, 1]) - 0.125) < 1e-12
    assert ps.perm_test_paired([]) is None


def test_perm_test_random_deterministic():
    """n>12 は乱択・seed 固定で決定論(2回一致)かつ [0,1]。"""
    diffs = [i - 7.0 + 0.3 for i in range(15)]      # n=15 > max_exact
    p1 = ps.perm_test_paired(diffs, seed=0)
    p2 = ps.perm_test_paired(diffs, seed=0)
    assert p1 == p2                                 # 決定論
    assert 0.0 <= p1 <= 1.0


def test_bh_fdr():
    """BH-FDR 補正 q(手計算)と None 透過。"""
    q = ps.bh_fdr([0.005, 0.01, 0.03, 0.05])        # m=4
    exp = [0.02, 0.02, 0.04, 0.05]
    for a, b in zip(q, exp):
        assert abs(a - b) < 1e-9
    # None は None のまま・有効値のみで m を数える
    q2 = ps.bh_fdr([0.01, None, 0.02])              # m=2
    assert q2[1] is None
    assert abs(q2[0] - 0.02) < 1e-9 and abs(q2[2] - 0.02) < 1e-9
    assert ps.bh_fdr([None, None]) == [None, None]
