"""Small helper for measuring how long a step takes, for log lines."""

import time


class Stopwatch:
    """Starts timing when created. Read `elapsed_ms` as often as needed."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        elapsed_seconds = time.perf_counter() - self._start
        return int(elapsed_seconds * 1000)
