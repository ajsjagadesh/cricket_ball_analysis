"""C++ trajectory helper bridge."""

from __future__ import annotations

class PythonConstantVelocityTrajectory:
    def __init__(self, max_history: int = 20, max_dt: float = 0.25):
        if max_history <= 0:
            raise ValueError("max_history must be positive.")
        if not (float(max_dt) > 0.0 and float(max_dt) < float("inf")):
            raise ValueError("max_dt must be finite and positive.")
        self.max_history = int(max_history)
        self.max_dt = float(max_dt)
        self.samples: list[tuple[float, float, float]] = []

    def add_sample(self, timestamp: float, x: float, y: float) -> None:
        if any(not (float("-inf") < float(value) < float("inf")) for value in (timestamp, x, y)):
            raise ValueError("timestamps and positions must be finite.")
        if len(self.samples) >= self.max_history:
            self.samples.pop(0)
        self.samples.append((float(timestamp), float(x), float(y)))

    def predict(self, horizon: float, steps: int) -> list[tuple[float, float]]:
        if len(self.samples) < 2:
            raise ValueError("At least two samples are required for trajectory prediction.")
        if not (float("-inf") < float(horizon) < float("inf")) or horizon < 0.0:
            raise ValueError("Horizon must be finite and non-negative.")
        if steps <= 0:
            raise ValueError("Prediction steps must be positive.")
        last_ts, last_x, last_y = self.samples[-1]
        prev_ts, prev_x, prev_y = self.samples[-2]
        dt = last_ts - prev_ts
        if dt <= 0.0 or dt > self.max_dt:
            raise ValueError("Invalid or stretched timestamps for constant-velocity model.")
        recent = self.samples[-5:]
        mean_t = sum(sample[0] for sample in recent) / len(recent)
        mean_x = sum(sample[1] for sample in recent) / len(recent)
        mean_y = sum(sample[2] for sample in recent) / len(recent)
        time_spread = sum((sample[0] - mean_t) ** 2 for sample in recent)
        if time_spread == 0.0:
            raise ValueError("Timestamps must not all be equal.")
        vx = sum((sample[0] - mean_t) * (sample[1] - mean_x) for sample in recent) / time_spread
        vy = sum((sample[0] - mean_t) * (sample[2] - mean_y) for sample in recent) / time_spread
        return [
            (last_x + vx * (horizon * (index + 1) / steps), last_y + vy * (horizon * (index + 1) / steps))
            for index in range(steps)
        ]

    def size(self) -> int:
        return len(self.samples)


try:
    from .cricket_vision_cpp import ConstantVelocityTrajectory as NativeConstantVelocityTrajectory

    NATIVE_EXTENSION_AVAILABLE = True
except ImportError:  # pragma: no cover - native extension may not be built
    NativeConstantVelocityTrajectory = None
    NATIVE_EXTENSION_AVAILABLE = False

ConstantVelocityTrajectory = (
    NativeConstantVelocityTrajectory
    if NativeConstantVelocityTrajectory is not None
    else PythonConstantVelocityTrajectory
)

__all__ = [
    "ConstantVelocityTrajectory",
    "NativeConstantVelocityTrajectory",
    "PythonConstantVelocityTrajectory",
    "NATIVE_EXTENSION_AVAILABLE",
]
