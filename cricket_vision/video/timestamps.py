from __future__ import annotations


def seconds_to_milliseconds(seconds: float) -> float:
    return float(seconds) * 1000.0


def milliseconds_to_seconds(milliseconds: float) -> float:
    return float(milliseconds) / 1000.0
