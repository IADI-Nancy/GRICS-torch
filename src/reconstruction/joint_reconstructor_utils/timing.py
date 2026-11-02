"""Timing of individual reconstruction steps."""

import time


class Timer:
    """Measure a solver step without coupling it to run logging."""

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.started
