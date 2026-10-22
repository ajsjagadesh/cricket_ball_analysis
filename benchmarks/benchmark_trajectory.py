from __future__ import annotations

import argparse
import json
import statistics
import time

from cricket_vision.cpp import (
    NATIVE_EXTENSION_AVAILABLE,
    NativeConstantVelocityTrajectory,
    PythonConstantVelocityTrajectory,
)


def _measure(trajectory_type: type, repetitions: int, sample_count: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repetitions):
        trajectory = trajectory_type(max_history=sample_count, max_dt=1.0)
        started = time.perf_counter()
        for index in range(sample_count):
            trajectory.add_sample(index / 60.0, index * 2.0, index * 3.0)
        trajectory.predict(0.2, 8)
        durations.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.mean(durations),
        "repetitions": repetitions,
        "sample_count": sample_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the C++ trajectory extension with its Python fallback.")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.samples < 2:
        parser.error("--repetitions must be positive and --samples must be at least 2.")

    python_result = _measure(PythonConstantVelocityTrajectory, args.repetitions, args.samples)
    result = {
        "native_extension_available": NATIVE_EXTENSION_AVAILABLE,
        "python_fallback": python_result,
        "native_cpp": None,
        "speedup_python_over_cpp": None,
        "warning": "Microbenchmark only; does not imply end-to-end video-analysis acceleration.",
    }
    if NATIVE_EXTENSION_AVAILABLE and NativeConstantVelocityTrajectory is not None:
        native_result = _measure(NativeConstantVelocityTrajectory, args.repetitions, args.samples)
        result["native_cpp"] = native_result
        result["speedup_python_over_cpp"] = (
            python_result["median_seconds"] / native_result["median_seconds"]
            if native_result["median_seconds"] > 0
            else None
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
