from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cricket_vision.tracking.player_tracker import TrackedPlayer

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseObservation:
    track_id: int
    source_timestamp: float
    scene_id: str
    bbox: tuple[int, int, int, int]
    keypoints: dict[str, Keypoint]


class PoseDetector:
    """Optional Ultralytics pose-model adapter operating on the selected track crop."""

    def __init__(
        self,
        model_path: str | Path | None,
        confidence_threshold: float = 0.4,
        inference_resolution: int = 640,
        device: str = "auto",
        model_instance: Any | None = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.inference_resolution = inference_resolution
        self.device = device
        self.model = None
        self.status = "disabled"
        self.reason = "No pose model configured."

        if self.model_path is None:
            return
        if not self.model_path.is_file():
            self.reason = f"Pose checkpoint not found: {self.model_path}"
            return
        if model_instance is not None:
            self.model = model_instance
            self.status = "ready"
            self.reason = ""
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            self.reason = "Ultralytics is not installed; install the optional pose-model dependencies."
            return

        self.model = YOLO(str(self.model_path))
        self.status = "ready"
        self.reason = ""

    def detect(
        self,
        frame: np.ndarray,
        track: TrackedPlayer,
        source_timestamp: float,
        scene_id: str,
    ) -> PoseObservation | None:
        if self.model is None or frame is None or frame.size == 0:
            return None

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox
        x1 = max(0, min(frame_width, x1))
        y1 = max(0, min(frame_height, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        infer_device = None if self.device == "auto" else self.device
        results = self.model.predict(
            source=crop,
            conf=self.confidence_threshold,
            imgsz=self.inference_resolution,
            device=infer_device,
            verbose=False,
        )
        if not results or results[0].keypoints is None:
            return None

        raw_keypoints = results[0].keypoints.data
        if len(raw_keypoints) == 0:
            return None
        values = raw_keypoints[0].detach().cpu().tolist()
        keypoints: dict[str, Keypoint] = {}
        for index, name in enumerate(COCO_KEYPOINT_NAMES):
            if index >= len(values) or len(values[index]) < 3:
                continue
            px, py, confidence = (float(value) for value in values[index][:3])
            if confidence < self.confidence_threshold:
                continue
            keypoints[name] = Keypoint(x=px + x1, y=py + y1, confidence=confidence)

        return PoseObservation(
            track_id=track.track_id,
            source_timestamp=float(source_timestamp),
            scene_id=scene_id,
            bbox=(x1, y1, x2, y2),
            keypoints=keypoints,
        )
