"""シミュ内時計(D15)。正準 = 整数 step(1 step = 10分、144 step = 1日)。"""
from __future__ import annotations

STEP_MINUTES = 10
STEPS_PER_DAY = 144
NIGHT_START_HOUR = 0   # 0:00-6:00 = 夜(圧縮: routine のみ+個別内省)
NIGHT_END_HOUR = 6


class Clock:
    def __init__(self, start_hour: int = 7):
        self.start_min = start_hour * 60

    def sim_min(self, step: int) -> int:
        return self.start_min + step * STEP_MINUTES

    def hour(self, step: int) -> int:
        return (self.sim_min(step) // 60) % 24

    def day(self, step: int) -> int:
        return self.sim_min(step) // (24 * 60)

    def is_night(self, step: int) -> bool:
        return NIGHT_START_HOUR <= self.hour(step) < NIGHT_END_HOUR

    def night_slot(self, step: int) -> int | None:
        """夜の中の何番目の step か(個別睡眠=内省の分散に使う)。昼は None。"""
        if not self.is_night(step):
            return None
        minutes_into_night = self.sim_min(step) % (24 * 60) - NIGHT_START_HOUR * 60
        return minutes_into_night // STEP_MINUTES
