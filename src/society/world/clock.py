"""シミュ内時計(D15)。正準 = 整数 step(1 step = 10分、144 step = 1日)。"""
from __future__ import annotations

STEP_MINUTES = 10
STEPS_PER_DAY = 144
NIGHT_START_HOUR = 0   # 0:00-6:00 = 夜(圧縮: routine のみ+個別内省)
NIGHT_END_HOUR = 6


class Clock:
    def __init__(self, start_hour: int = 7, start_min: int | None = None):
        # 開始時刻(day0 step0 の分 of day)。既定 07:00=420 分=現行値(バイト一致)。
        # start_min を明示すると分粒度で指定できる(run.start_tod="HH:MM" の配線先)。
        # start_min=None のときのみ start_hour*60 を使う(後方互換: Clock(start_hour=0) 等)。
        self.start_min = int(start_min) if start_min is not None else start_hour * 60

    def sim_min(self, step: int) -> int:
        return self.start_min + step * STEP_MINUTES

    def hour(self, step: int) -> int:
        return (self.sim_min(step) // 60) % 24

    def day(self, step: int) -> int:
        # 日番号 = 開始時刻を含む絶対分 sim_min を 1440 で割った商(= sim クロックの深夜0時境界)。
        # 開始時刻起点の「経過 simulated day」であり、day 境界は常に深夜0時に落ちる。既定 07:00 開始では
        # day0 が 07:00〜24:00 の短い初日になり、day 境界は step 102(sim_min=1440)。00:00 開始なら
        # day = step // STEPS_PER_DAY と一致し、深夜0時=step の 144 の倍数で自然に日替わりする。
        # 全モジュール共通の日境界判定(sim_min // 1440)がこの定義を共有する=start_tod 変更に追従する。
        return self.sim_min(step) // (24 * 60)

    def is_night(self, step: int) -> bool:
        return NIGHT_START_HOUR <= self.hour(step) < NIGHT_END_HOUR

    def night_slot(self, step: int) -> int | None:
        """夜の中の何番目の step か(個別睡眠=内省の分散に使う)。昼は None。"""
        if not self.is_night(step):
            return None
        minutes_into_night = self.sim_min(step) % (24 * 60) - NIGHT_START_HOUR * 60
        return minutes_into_night // STEP_MINUTES
