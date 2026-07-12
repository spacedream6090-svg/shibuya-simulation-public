"""LOD 予算(1step あたりの LLM 発火上限)。

発火の「きっかけ」判定は v1 の surprise_of(即時トリガー)から
欲求駆動発火(drive.py + scheduler._phase_drive)へ移行した(Phase A, 2026-07-04)。
本モジュールは予算(インフラ上限 N≈90-480 decision/step)のみを担う。
"""
from __future__ import annotations


class LodBudget:
    def __init__(self, max_per_step: int):
        self.max_per_step = int(max_per_step)
        self.used = 0

    def reset(self) -> None:
        self.used = 0

    def take(self) -> bool:
        if self.used < self.max_per_step:
            self.used += 1
            return True
        return False
