"""幾何ユーティリティ。ログ用ポリラインの間引き(形状保存)。

旧実装は「等間隔に最大N点」で間引いていたため、曲がり角が落ちて
ビューア上で車・エージェントが道路外をショートカットして見えた。
RDP(Ramer-Douglas-Peucker)は許容誤差内で形状を保ったまま点数を減らす。
"""
from __future__ import annotations

import math


def rdp(points: list, epsilon: float = 6.0, max_points: int = 24) -> list:
    """形状を保つポリライン間引き。epsilon=許容ずれ(m)。"""
    if len(points) <= 2:
        return list(points)

    def _rdp(pts: list) -> list:
        if len(pts) <= 2:
            return list(pts)
        ax, ay = pts[0][0], pts[0][1]
        bx, by = pts[-1][0], pts[-1][1]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best_d, best_i = -1.0, 0
        for i in range(1, len(pts) - 1):
            px, py = pts[i][0], pts[i][1]
            if norm == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dy * px - dx * py + bx * ay - by * ax) / norm
            if d > best_d:
                best_d, best_i = d, i
        if best_d <= epsilon:
            return [pts[0], pts[-1]]
        left = _rdp(pts[:best_i + 1])
        return left[:-1] + _rdp(pts[best_i:])

    out = _rdp(list(points))
    if len(out) > max_points:              # 念のための上限(等間隔に落とす)
        idxs = [round(i * (len(out) - 1) / (max_points - 1))
                for i in range(max_points)]
        out = [out[i] for i in idxs]
    return out
