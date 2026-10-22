from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TORSO_KEYPOINT_CONFIDENCE = 0.25
MIN_VISIBLE_BODY_KEYPOINTS = 4
SHOULDER_INDICES = (5, 6)
HIP_INDICES = (11, 12)


@dataclass
class PlayerDetection:
    """Single person detection from a video frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    scene_id: str | None = None
    source_timestamp: float | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class PlayerDetector:
    """Ultralytics person detector; never treats generic motion as a player."""

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        model_path: str | Path | None = None,
        inference_resolution: int = 1280,
        device: str = "auto",
        model_instance: Any | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.model_path = Path(model_path) if model_path else None
        self.inference_resolution = inference_resolution
        self.device = device
        self.model = None
        self.status = "disabled"
        self.reason = "No person-detection checkpoint configured."

        if self.model_path is None:
            return
        if not self.model_path.is_file():
            self.reason = f"Person-detection checkpoint not found: {self.model_path}"
            return
        if model_instance is not None:
            self.model = model_instance
            self.status = "ready"
            self.reason = ""
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            self.reason = "Ultralytics is not installed; install the optional model dependencies."
            return

        self.model = YOLO(str(self.model_path))
        self.status = "ready"
        self.reason = ""

    def detect(
        self,
        frame: np.ndarray,
        source_timestamp: float | None = None,
        scene_id: str | None = None,
    ) -> list[PlayerDetection]:
        if self.model is None or frame is None or frame.size == 0:
            return []

        infer_device = None if self.device == "auto" else self.device
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.inference_resolution,
            device=infer_device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        keypoints = getattr(results[0], "keypoints", None)
        keypoint_confidences = getattr(keypoints, "conf", None)
        if keypoint_confidences is None:
            return []

        names = getattr(results[0], "names", getattr(self.model, "names", {}))
        detections: list[PlayerDetection] = []
        for index, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0].detach().cpu().item())
            if isinstance(names, dict):
                class_name = names.get(class_id, "")
            else:
                class_name = names[class_id] if class_id < len(names) else ""
            if str(class_name).lower() != "person":
                continue

            confidences = keypoint_confidences[index].detach().cpu().tolist()
            visible_keypoints = {
                point_index
                for point_index, confidence in enumerate(confidences)
                if confidence >= TORSO_KEYPOINT_CONFIDENCE
            }
            has_shoulders = any(point_index in visible_keypoints for point_index in SHOULDER_INDICES)
            has_hips = any(point_index in visible_keypoints for point_index in HIP_INDICES)
            if len(visible_keypoints) < MIN_VISIBLE_BODY_KEYPOINTS or not (has_shoulders and has_hips):
                continue

            coordinates = box.xyxy[0].detach().cpu().tolist()
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(box.conf[0].detach().cpu().item())
            detections.append(
                PlayerDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    scene_id=scene_id,
                    source_timestamp=source_timestamp,
                )
            )
        return detections

    def __call__(
        self,
        frame: np.ndarray,
        source_timestamp: float | None = None,
        scene_id: str | None = None,
    ) -> list[PlayerDetection]:
        return self.detect(frame, source_timestamp=source_timestamp, scene_id=scene_id)
