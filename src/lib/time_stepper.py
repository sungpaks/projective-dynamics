"""물리 시뮬레이션의 step은 고정하고 프레임 렌더링 속도에 따라 가변"""

import time
from collections.abc import Callable


class TimeStepper:
    def __init__(
        self,
        steps_per_second: float,
        max_frame_time: float = 0.25,
        max_substeps: int = 8,
    ) -> None:
        self._time_step = 1.0 / steps_per_second
        self._max_frame_time = max_frame_time
        self._max_substeps = max_substeps

        self._previous_time = time.perf_counter()
        self._accumulator = 0.0

    def reset(self) -> None:
        """누적된 시간을 버리고 현재 시각에서 다시 시작한다."""
        self._previous_time = time.perf_counter()
        self._accumulator = 0.0

    def advance(self, step: Callable[[float], None]) -> int:
        current_time = time.perf_counter()
        frame_time = current_time - self._previous_time
        self._previous_time = current_time

        frame_time = min(frame_time, self._max_frame_time)
        self._accumulator += frame_time

        substep_count = 0
        while (
            self._accumulator >= self._time_step and substep_count < self._max_substeps
        ):
            step(self._time_step)
            self._accumulator -= self._time_step
            substep_count += 1

        return substep_count
