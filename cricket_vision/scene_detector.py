from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneChange:
    scene_id: str
    is_cut: bool
    score: float


class SceneDetector:
    """Basic hard-cut detector based on visual change between consecutive frames."""

    def __init__(self, scene_cut_threshold: float = 0.25):
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._previous_frame: np.ndarray | None = None
        self._scene_index = 0

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def process(self, frame: np.ndarray) -> SceneChange:
        if frame is None or frame.size == 0:
            return SceneChange(scene_id=f"scene-{self._scene_index}", is_cut=False, score=0.0)

        if self._previous_frame is None:
            self._previous_frame = self._normalize_frame(frame)
            return SceneChange(scene_id=f"scene-{self._scene_index}", is_cut=False, score=0.0)

        current = self._normalize_frame(frame)
        diff = cv2.absdiff(current, self._previous_frame)
        score = float(np.mean(diff) / 255.0)
        is_cut = score >= self.scene_cut_threshold

        if is_cut:
            self._scene_index += 1
            self._previous_frame = current
            return SceneChange(scene_id=f"scene-{self._scene_index}", is_cut=True, score=score)

        self._previous_frame = current
        return SceneChange(scene_id=f"scene-{self._scene_index}", is_cut=False, score=score)

    def reset(self) -> None:
        self._previous_frame = None
        self._scene_index = 0

    @property
    def current_scene_id(self) -> str:
        return f"scene-{self._scene_index}"
