from __future__ import annotations

try:
    from cricket_vision.cpp import (
        ConstantVelocityTrajectory as _NativeConstantVelocityTrajectory,
        NATIVE_EXTENSION_AVAILABLE,
    )
except Exception:  # pragma: no cover - native extension may not be built on this machine
    _NativeConstantVelocityTrajectory = None
    NATIVE_EXTENSION_AVAILABLE = False


class ConstantVelocityTrajectory:
    """Trajectory model with native C++ acceleration when available.

    The implementation gracefully falls back to a pure-Python version if the extension
    has not been built or cannot be imported.
    """

    def __init__(self, max_history: int = 20, max_dt: float = 0.25):
        if _NativeConstantVelocityTrajectory is not None:
            self._impl = _NativeConstantVelocityTrajectory(max_history=max_history, max_dt=max_dt)
        else:
            self._impl = _PythonConstantVelocityTrajectory(max_history=max_history, max_dt=max_dt)

    def add_sample(self, timestamp: float, x: float, y: float) -> None:
        self._impl.add_sample(timestamp, x, y)

    def predict(self, horizon: float, steps: int) -> list[tuple[float, float]]:
        return self._impl.predict(horizon, steps)

    def size(self) -> int:
        return self._impl.size()

    @property
    def using_native(self) -> bool:
        return NATIVE_EXTENSION_AVAILABLE


class _PythonConstantVelocityTrajectory:
    def __init__(self, max_history: int = 20, max_dt: float = 0.25):
        self.max_history = int(max_history)
        self.max_dt = float(max_dt)
        self.samples: list[tuple[float, float, float]] = []

    def add_sample(self, timestamp: float, x: float, y: float) -> None:
        if any(not (float(value) == float(value)) for value in (timestamp, x, y)):
            raise ValueError("timestamps and positions must be finite.")
        if len(self.samples) >= self.max_history:
            self.samples.pop(0)
        self.samples.append((float(timestamp), float(x), float(y)))

    def predict(self, horizon: float, steps: int) -> list[tuple[float, float]]:
        if len(self.samples) < 2:
            raise ValueError("At least two samples are required for trajectory prediction.")
        if horizon < 0.0:
            raise ValueError("Horizon must be non-negative.")
        if steps <= 0:
            raise ValueError("Prediction steps must be positive.")

        last_ts, last_x, last_y = self.samples[-1]
        prev_ts, _, _ = self.samples[-2]
        dt = last_ts - prev_ts
        if dt <= 0.0 or dt > self.max_dt:
            raise ValueError("Invalid or stretched timestamps for constant-velocity model.")

        recent = self.samples[-5:]
        mean_ts = sum(sample[0] for sample in recent) / len(recent)
        mean_x = sum(sample[1] for sample in recent) / len(recent)
        mean_y = sum(sample[2] for sample in recent) / len(recent)
        time_spread = sum((sample[0] - mean_ts) ** 2 for sample in recent)
        if time_spread == 0.0:
            raise ValueError("Timestamps must not all be equal.")
        velocity_x = sum(
            (sample[0] - mean_ts) * (sample[1] - mean_x) for sample in recent
        ) / time_spread
        velocity_y = sum(
            (sample[0] - mean_ts) * (sample[2] - mean_y) for sample in recent
        ) / time_spread
        return [
            (
                last_x + velocity_x * (horizon * (idx + 1) / steps),
                last_y + velocity_y * (horizon * (idx + 1) / steps),
            )
            for idx in range(steps)
        ]

    def size(self) -> int:
        return len(self.samples)


__all__ = ["ConstantVelocityTrajectory"]
