from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    source_timestamp: float
    scene_id: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BallDetector:
    """Optional Ultralytics detector for an explicitly supplied cricket-ball model."""

    def __init__(
        self,
        model_path: str | Path | None,
        confidence_threshold: float = 0.30,
        inference_resolution: int = 1280,
        device: str = "auto",
        max_bbox_area_fraction: float = 0.01,
        max_bbox_dimension_fraction: float = 0.10,
        max_aspect_ratio: float = 4.0,
        model_instance: Any | None = None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if not 0.0 < max_bbox_area_fraction <= 1.0:
            raise ValueError("max_bbox_area_fraction must be in (0, 1].")
        if not 0.0 < max_bbox_dimension_fraction <= 1.0:
            raise ValueError("max_bbox_dimension_fraction must be in (0, 1].")
        if max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio must be at least 1.")
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.inference_resolution = inference_resolution
        self.device = device
        self.max_bbox_area_fraction = max_bbox_area_fraction
        self.max_bbox_dimension_fraction = max_bbox_dimension_fraction
        self.max_aspect_ratio = max_aspect_ratio
        self.model = None
        self.status = "disabled"
        self.reason = "No cricket-ball checkpoint configured."

        if self.model_path is None:
            return
        if not self.model_path.is_file():
            self.reason = f"Cricket-ball checkpoint not found: {self.model_path}"
            return
        if model_instance is not None:
            self.model = model_instance
            self.status = "ready"
            self.reason = ""
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            self.reason = "Ultralytics is not installed; install the optional ball-detection dependencies."
            return

        self.model = YOLO(str(self.model_path))
        self.status = "ready"
        self.reason = ""

    def detect(self, frame: np.ndarray, source_timestamp: float, scene_id: str) -> list[BallDetection]:
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
        detections: list[BallDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        max_dimension = min(frame_width, frame_height) * self.max_bbox_dimension_fraction
        for box in boxes:
            class_id = int(box.cls[0].detach().cpu().item())
            names = getattr(results[0], "names", getattr(self.model, "names", {}))
            if isinstance(names, dict):
                class_name = names.get(class_id, "")
            else:
                class_name = names[class_id] if class_id < len(names) else ""
            if str(class_name).lower() not in {"ball", "cricket ball", "cricket_ball"}:
                continue

            coords = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            x1, y1, x2, y2 = (float(value) for value in coords)
            if x2 <= x1 or y2 <= y1:
                continue
            width = x2 - x1
            height = y2 - y1
            if (
                width * height > frame_area * self.max_bbox_area_fraction
                or max(width, height) > max_dimension
                or max(width / height, height / width) > self.max_aspect_ratio
            ):
                continue
            detections.append(
                BallDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    source_timestamp=float(source_timestamp),
                    scene_id=scene_id,
                )
            )
        return detections
