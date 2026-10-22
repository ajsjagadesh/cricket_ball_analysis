from __future__ import annotations

import sys
import time
from typing import TextIO


def format_progress(current: int, total: int | None, elapsed_seconds: float, width: int = 28) -> str:
    """Format a terminal progress line, supporting videos with unknown frame counts."""

    safe_elapsed = max(elapsed_seconds, 0.05)
    fps = current / safe_elapsed
    if total is None or total <= 0:
        return f"Rendering video | frame {current} | {fps:.1f} fps | {safe_elapsed:.1f}s"

    completed = min(current, total)
    ratio = completed / total
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"Rendering video |{bar}| {ratio * 100:5.1f}% ({completed}/{total}) | {fps:.1f} fps"


class ProgressReporter:
    """Rate-limited terminal progress updates for frame-by-frame processing."""

    def __init__(
        self,
        stream: TextIO | None = None,
        update_interval_seconds: float = 0.2,
    ):
        if update_interval_seconds < 0:
            raise ValueError("update_interval_seconds must be non-negative.")
        self.stream = stream if stream is not None else sys.stderr
        self._interactive = self.stream.isatty()
        self.update_interval_seconds = update_interval_seconds
        self._started_at: float | None = None
        self._last_update_at = 0.0
        self._last_line = ""
        self._current = 0
        self._total: int | None = None
        self._finished = False

    def __call__(self, current: int, total: int | None) -> None:
        now = time.monotonic()
        self._current = current
        self._total = total
        if self._started_at is None:
            self._started_at = now
        if current > 0 and now - self._last_update_at < self.update_interval_seconds:
            return

        if not self._interactive:
            return

        elapsed = now - self._started_at
        self._last_line = format_progress(current, total, elapsed)
        self.stream.write("\r\x1b[2K" + self._last_line)
        self.stream.flush()
        self._last_update_at = now

    def finish(self, succeeded: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if self._started_at is None:
            self._started_at = time.monotonic()
        elapsed = max(time.monotonic() - self._started_at, 0.001)
        result = "Complete" if succeeded else "Failed"
        final_line = (
            format_progress(self._current, self._total, elapsed)
            if self._current
            else "No frames processed"
        )
        if self._interactive:
            self.stream.write("\r\x1b[2K")
        self.stream.write(f"{result} | {final_line} | {elapsed:.1f}s\n")
        self.stream.flush()
