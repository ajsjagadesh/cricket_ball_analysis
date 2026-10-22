from __future__ import annotations

import numpy as np

from cricket_vision.scene_detector import SceneDetector


def test_scene_detector_detects_hard_cut() -> None:
    detector = SceneDetector(scene_cut_threshold=0.05)
    base = np.zeros((80, 80, 3), dtype=np.uint8)
    base[:] = 20

    bright = np.zeros((80, 80, 3), dtype=np.uint8)
    bright[:] = 200

    first = detector.process(base)
    second = detector.process(bright)

    assert first.is_cut is False
    assert second.is_cut is True
    assert second.scene_id == "scene-1"


def test_scene_detector_reset_clears_history() -> None:
    detector = SceneDetector(scene_cut_threshold=0.05)
    first_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    second_frame = np.full((40, 40, 3), 128, dtype=np.uint8)

    detector.process(first_frame)
    detector.process(second_frame)
    detector.reset()

    fresh = detector.process(first_frame)
    assert fresh.scene_id == "scene-0"
