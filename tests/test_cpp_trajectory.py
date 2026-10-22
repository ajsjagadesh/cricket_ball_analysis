from __future__ import annotations

import math

import pytest

from cricket_vision.cpp import (
    ConstantVelocityTrajectory,
    NATIVE_EXTENSION_AVAILABLE,
    NativeConstantVelocityTrajectory,
    PythonConstantVelocityTrajectory,
)


def test_cpp_or_python_trajectory_prediction_matches_constant_velocity() -> None:
    model = ConstantVelocityTrajectory(max_history=10, max_dt=0.5)
    model.add_sample(0.0, 0.0, 0.0)
    model.add_sample(0.1, 1.0, 2.0)
    model.add_sample(0.2, 2.0, 4.0)

    prediction = model.predict(0.2, 2)
    assert len(prediction) == 2
    assert prediction[0][0] == 3.0
    assert prediction[0][1] == 6.0
    assert prediction[1][0] == 4.0
    assert prediction[1][1] == 8.0


def test_trajectory_uses_recent_detections_to_smooth_direction() -> None:
    model = ConstantVelocityTrajectory(max_history=10, max_dt=0.5)
    for sample in (
        (0.0, 0.0, 0.0),
        (0.1, 1.0, 2.0),
        (0.2, 2.0, 4.0),
        (0.3, 3.0, 6.0),
        (0.4, 3.9, 7.8),
    ):
        model.add_sample(*sample)

    prediction = model.predict(0.1, 1)[0]

    assert prediction[0] == pytest.approx(4.88)
    assert prediction[1] == pytest.approx(9.76)


def test_cpp_trajectory_requires_minimum_history() -> None:
    model = ConstantVelocityTrajectory(max_history=10, max_dt=0.5)
    model.add_sample(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="two samples"):
        model.predict(0.2, 2)


@pytest.mark.skipif(not NATIVE_EXTENSION_AVAILABLE, reason="Native extension not built for this Python interpreter.")
def test_native_and_python_trajectory_implementations_are_equivalent() -> None:
    native = NativeConstantVelocityTrajectory(max_history=3, max_dt=0.5)
    python = PythonConstantVelocityTrajectory(max_history=3, max_dt=0.5)
    samples = [(0.0, 0.0, 0.0), (0.1, 1.0, 2.0), (0.25, 2.5, 5.0), (0.4, 4.0, 8.0)]
    for sample in samples:
        native.add_sample(*sample)
        python.add_sample(*sample)

    assert native.size() == python.size() == 3
    assert native.predict(0.2, 4) == pytest.approx(python.predict(0.2, 4))

    for model in (native, python):
        with pytest.raises(ValueError):
            model.add_sample(float("nan"), 0.0, 0.0)
        with pytest.raises(ValueError):
            model.predict(math.inf, 2)


@pytest.mark.skipif(not NATIVE_EXTENSION_AVAILABLE, reason="Native extension not built for this Python interpreter.")
def test_native_and_python_reject_irregular_timestamps_consistently() -> None:
    for trajectory_type in (NativeConstantVelocityTrajectory, PythonConstantVelocityTrajectory):
        model = trajectory_type(max_history=4, max_dt=0.2)
        model.add_sample(0.0, 0.0, 0.0)
        model.add_sample(0.3, 1.0, 1.0)
        with pytest.raises(ValueError, match="timestamps"):
            model.predict(0.1, 2)
