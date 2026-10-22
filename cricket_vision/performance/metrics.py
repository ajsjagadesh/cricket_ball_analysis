from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator


class MetricsCollector:
    """Bounded-memory timing aggregates for pipeline stages."""

    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)
        self._total_seconds: dict[str, float] = defaultdict(float)
        self._max_seconds: dict[str, float] = defaultdict(float)
        self._pipeline_started: float | None = None
        self._processed_frames = 0

    def start(self) -> None:
        if self._pipeline_started is None:
            self._pipeline_started = time.perf_counter()

    def record(self, stage: str, elapsed_seconds: float) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        self._counts[stage] += 1
        self._total_seconds[stage] += elapsed
        self._max_seconds[stage] = max(self._max_seconds[stage], elapsed)

    def frame_processed(self) -> None:
        self._processed_frames += 1

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - started)

    def summary(self) -> dict[str, object]:
        elapsed = (
            max(time.perf_counter() - self._pipeline_started, 1e-9)
            if self._pipeline_started is not None
            else 0.0
        )
        stages = {
            name: {
                "calls": self._counts[name],
                "total_seconds": self._total_seconds[name],
                "mean_seconds": self._total_seconds[name] / self._counts[name],
                "max_seconds": self._max_seconds[name],
            }
            for name in sorted(self._counts)
        }
        return {
            "processed_frames": self._processed_frames,
            "end_to_end_seconds": elapsed,
            "processing_fps": self._processed_frames / elapsed if elapsed > 0 else 0.0,
            "stages": stages,
        }


@contextmanager
def timing_context(label: str):
    """Context manager that measures elapsed time for a section of code."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")
