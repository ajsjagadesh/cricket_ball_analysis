from __future__ import annotations

import numpy as np

from cricket_vision.detection.player_detector import PlayerDetection
from cricket_vision.team_classifier import TeamClassifier


def _make_detection(track_id: int, x1: int, y1: int, x2: int, y2: int) -> PlayerDetection:
    detection = PlayerDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, scene_id="scene-0")
    detection.track_id = track_id
    return detection


def test_team_classifier_produces_consistent_labels() -> None:
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    frame[20:80, 20:80] = (30, 70, 50)
    frame[20:80, 120:180] = (90, 30, 30)

    detections = [
        _make_detection(1, 20, 20, 80, 80),
        _make_detection(2, 120, 20, 180, 80),
    ]

    classifier = TeamClassifier()
    assignment_1 = classifier.classify_frame(frame, detections)

    assert assignment_1[1].label in {"A", "B"}
    assert assignment_1[2].label in {"A", "B"}
    assert assignment_1[1].label != assignment_1[2].label

    frame_2 = np.zeros((120, 220, 3), dtype=np.uint8)
    frame_2[20:80, 20:80] = (25, 80, 60)
    frame_2[20:80, 120:180] = (95, 18, 25)

    assignment_2 = classifier.classify_frame(frame_2, detections)
    assert assignment_2[1].label == assignment_1[1].label
    assert assignment_2[2].label == assignment_1[2].label


def test_team_classifier_unknown_for_ambiguous_crop() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:60, 20:60] = (80, 80, 80)

    classifier = TeamClassifier()
    detection = _make_detection(5, 20, 20, 60, 60)
    assignment = classifier.classify_frame(frame, [detection])

    assert assignment[5].label == "unknown"
