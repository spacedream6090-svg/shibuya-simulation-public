"""シミュ内時計(D15)。正準 = 整数 step(既定 1 step = 10分、144 step = 1日)。

第79バッチ(毎分レート化)で 1 step の分数が `run.dt_min` で可変になった。
モジュール定数 STEP_MINUTES / STEPS_PER_DAY は **正準値(=既定)** として残し、
実際の Δt は `Clock.step_minutes` / `Clock.steps_per_day` から取る。
既定 Δt=10 では両者は完全に一致する(= 既存コードの挙動は不変)。
"""
from __future__ import annotations

STEP_MINUTES = 10      # 正準 Δt(run.dt_min の既定値)。実行時の値は Clock.step_minutes
STEPS_PER_DAY = 144    # 正準 Δt での 1 日の step 数。実行時の値は Clock.steps_per_day
NIGHT_START_HOUR = 0   # 0:00-6:00 = 夜(圧縮: routine のみ+個別内省)
NIGHT_END_HOUR = 6


class Clock:
    def __init__(self, start_hour: int = 7, start_min: int | None = None,
                 step_minutes: int = STEP_MINUTES):
        # 開始時刻(day0 step0 の分 of day)。既定 07:00=420 分=現行値(バイト一致)。
        # start_min を明示すると分粒度で指定できる(run.start_tod="HH:MM" の配線先)。
        # start_min=None のときのみ start_hour*60 を使う(後方互換: Clock(start_hour=0) 等)。
        self.start_min = int(start_min) if start_min is not None else start_hour * 60
        # 1 step の分数(run.dt_min の配線先)。既定は正準 10 = 既存の全呼び出しと同一。
        self.step_minutes = int(step_minutes)

    # ---- Δt 由来の派生量(呼び出し側が 144 / 6 / 600 を直書きしないための単一源)----
    @property
    def steps_per_day(self) -> int:
        return max(1, (24 * 60) // self.step_minutes)

    @property
    def steps_per_hour(self) -> int:
        return max(1, 60 // self.step_minutes)

    @property
    def step_seconds(self) -> float:
        return float(self.step_minutes) * 60.0

    def dur_steps(self, canon_steps: int) -> int:
        """正準 Δt(10分)基準で書かれた **持続長 [step]** を現在の Δt へ読み替える。

        コードに直書きされた「2-4 step 滞在」「1 step 通勤」等はすべて 10 分刻みを
        前提にした**時間**なので、Δt を細かくしたら step 数を増やさないと滞在が縮む。
        timeconv.scale_steps と同じ規則(0 は 0 のまま、最低 1)。
        既定 Δt=10 では `int(canon_steps)` を素通しする(= 従来の直書きとバイト一致)。"""
        if self.step_minutes == STEP_MINUTES:
            return int(canon_steps)
        from ..timeconv import scale_steps
        return scale_steps(canon_steps, self.step_minutes)

    def min_to_steps(self, minutes: float) -> int:
        """分 → step(切り捨て)。既定 Δt=10 では `minutes // 10` と同値。"""
        return int(minutes) // self.step_minutes

    def sim_min(self, step: int) -> int:
        return self.start_min + step * self.step_minutes

    def hour(self, step: int) -> int:
        return (self.sim_min(step) // 60) % 24

    def day(self, step: int) -> int:
        # 日番号 = 開始時刻を含む絶対分 sim_min を 1440 で割った商(= sim クロックの深夜0時境界)。
        # 開始時刻起点の「経過 simulated day」であり、day 境界は常に深夜0時に落ちる。既定 07:00 開始では
        # day0 が 07:00〜24:00 の短い初日になり、day 境界は step 102(sim_min=1440)。00:00 開始なら
        # day = step // steps_per_day と一致し、深夜0時=step の 144 の倍数で自然に日替わりする。
        # 全モジュール共通の日境界判定(sim_min // 1440)がこの定義を共有する=start_tod 変更に追従する。
        return self.sim_min(step) // (24 * 60)

    def is_night(self, step: int) -> bool:
        return NIGHT_START_HOUR <= self.hour(step) < NIGHT_END_HOUR

    def night_slot(self, step: int) -> int | None:
        """夜の中の何番目の step か(個別睡眠=内省の分散に使う)。昼は None。"""
        if not self.is_night(step):
            return None
        minutes_into_night = self.sim_min(step) % (24 * 60) - NIGHT_START_HOUR * 60
        return minutes_into_night // self.step_minutes
