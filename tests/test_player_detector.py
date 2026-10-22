from __future__ import annotations

import numpy as np

from cricket_vision.detection.player_detector import PlayerDetector


class _Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def __getitem__(self, index):
        return _Tensor(self.values[index])

    def item(self):
        if isinstance(self.values, list):
            return self.values[0]
        return self.values

    def tolist(self):
        return self.values


class _Box:
    def __init__(self, class_id: int, confidence: float, coordinates: list[float]):
        self.cls = _Tensor([class_id])
        self.conf = _Tensor([confidence])
        self.xyxy = [_Tensor(coordinates)]


class _Model:
    names = {0: "person", 1: "sports ball"}

    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        result = type(
            "Result",
            (),
            {
                "names": self.names,
                "boxes": [
                    _Box(0, 0.91, [10.2, 20.4, 50.1, 100.2]),
                    _Box(1, 0.99, [80, 40, 90, 50]),
                    _Box(0, 0.95, [100, 20, 130, 100]),
                ],
                "keypoints": type(
                    "Keypoints",
                    (),
                    {
                        "conf": _Tensor(
                            [
                                [0.9] * 17,
                                [0.9] * 17,
                                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0],
                            ]
                        )
                    },
                )(),
            },
        )()
        return [result]


def test_player_detector_does_not_turn_motion_into_fake_player_boxes() -> None:
    detector = PlayerDetector(model_path=None)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    assert detector.detect(frame) == []
    assert detector.status == "disabled"


def test_player_detector_keeps_only_people_with_a_visible_torso(tmp_path) -> None:
    checkpoint = tmp_path / "pose.pt"
    checkpoint.touch()
    model = _Model()
    detector = PlayerDetector(model_path=checkpoint, model_instance=model)

    detections = detector.detect(
        np.zeros((120, 160, 3), dtype=np.uint8),
        source_timestamp=1.25,
        scene_id="scene-2",
    )

    assert len(detections) == 1
    assert detections[0].bbox == (10, 20, 50, 100)
    assert detections[0].confidence == 0.91
    assert detections[0].source_timestamp == 1.25
    assert detections[0].scene_id == "scene-2"
    assert model.kwargs["conf"] == detector.confidence_threshold
