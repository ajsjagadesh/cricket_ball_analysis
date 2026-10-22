from __future__ import annotations

import numpy as np

from cricket_vision.detection.ball_detector import BallDetector


def test_ball_detector_is_disabled_when_no_checkpoint_is_provided() -> None:
    detector = BallDetector(model_path=None)
    assert detector.status == "disabled"
    assert "No cricket-ball checkpoint" in detector.reason


def test_ball_detector_reports_missing_checkpoint_without_importing_model(tmp_path) -> None:
    detector = BallDetector(model_path=tmp_path / "missing.pt")
    assert detector.status == "disabled"
    assert "not found" in detector.reason


class _Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _Tensor([self.values[index]])

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.values[0]

    def tolist(self):
        return self.values


class _Box:
    def __init__(self, class_id: int, confidence: float, coordinates: list[float]):
        self.cls = _Tensor([class_id])
        self.conf = _Tensor([confidence])
        self.xyxy = [_Tensor(coordinates)]


class _Model:
    names = {0: "ball", 1: "person"}

    def predict(self, **_kwargs):
        result = type(
            "Result",
            (),
            {
                "names": self.names,
                "boxes": [
                    _Box(0, 0.95, [20, 20, 40, 40]),
                    _Box(0, 0.99, [100, 100, 500, 500]),
                    _Box(1, 0.99, [50, 50, 70, 70]),
                ],
            },
        )()
        return [result]


def test_ball_detector_filters_oversized_and_non_ball_boxes(tmp_path) -> None:
    checkpoint = tmp_path / "ball.pt"
    checkpoint.touch()
    detector = BallDetector(model_path=checkpoint, model_instance=_Model())

    detections = detector.detect(np.zeros((600, 800, 3), dtype=np.uint8), 1.0, "scene-0")

    assert len(detections) == 1
    assert detections[0].bbox == (20.0, 20.0, 40.0, 40.0)


def test_ball_detector_keeps_a_high_confidence_ball_box_in_portrait_video(tmp_path) -> None:
    checkpoint = tmp_path / "ball.pt"
    checkpoint.touch()
    model = _Model()
    model.predict = lambda **_kwargs: [
        type(
            "Result",
            (),
            {
                "names": model.names,
                "boxes": [_Box(0, 0.735, [330.9, 474.9, 383.9, 518.8])],
            },
        )()
    ]
    detector = BallDetector(model_path=checkpoint, model_instance=model)

    detections = detector.detect(np.zeros((858, 587, 3), dtype=np.uint8), 1.0, "scene-0")

    assert len(detections) == 1
    assert detections[0].confidence == 0.735
