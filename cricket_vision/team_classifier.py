from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TeamAssignment:
    track_id: int
    label: str
    confidence: float
    scene_id: str | None = None


class TeamClassifier:
    """Unsupervised jersey-color grouping with label stability across frames."""

    def __init__(self, manual_assignments: dict[int, str] | None = None):
        self.manual_assignments = dict(manual_assignments or {})
        self._team_centers: dict[str, float] = {}

    @staticmethod
    def _dominant_hue(bgr: np.ndarray) -> float | None:
        if bgr is None or bgr.size == 0:
            return None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (sat > 35) & (val > 25)
        if np.count_nonzero(mask) < 16:
            return None
        hue = hsv[:, :, 0][mask]
        if hue.size == 0:
            return None
        return float(np.median(hue))

    def _cluster_hues(self, hues: list[float]) -> tuple[float, float]:
        if not hues:
            return (0.0, 0.0)
        ordered = sorted(hues)
        if len(ordered) == 1:
            return (ordered[0], ordered[0])
        midpoint = ordered[len(ordered) // 2]
        left = [value for value in ordered if value <= midpoint]
        right = [value for value in ordered if value > midpoint]
        return (float(np.median(left)) if left else ordered[0], float(np.median(right)) if right else ordered[-1])

    def _resolve_labels(self, cluster_means: list[float]) -> tuple[str, str]:
        if self._team_centers:
            ordered = sorted(cluster_means, key=lambda value: abs(value - self._team_centers.get("A", value)))
            first = ordered[0]
            second = ordered[1] if len(ordered) > 1 else first
            return ("A", "B") if abs(first - self._team_centers.get("A", first)) <= abs(second - self._team_centers.get("A", second)) else ("B", "A")

        lower = min(cluster_means)
        upper = max(cluster_means)
        if lower == upper:
            return ("A", "B")
        return ("A", "B")

    def classify_frame(self, frame: np.ndarray, detections: list, scene_id: str | None = None) -> dict[int, TeamAssignment]:
        assignments: dict[int, TeamAssignment] = {}
        if frame is None or frame.size == 0 or not detections:
            return assignments

        hues: list[float] = []
        detection_details: list[tuple[int, float]] = []
        for index, detection in enumerate(detections):
            track_id = getattr(detection, "track_id", index)
            if track_id in self.manual_assignments:
                assignments[track_id] = TeamAssignment(track_id=track_id, label=self.manual_assignments[track_id], confidence=1.0, scene_id=scene_id)
                continue

            x1, y1, x2, y2 = detection.bbox if hasattr(detection, "bbox") else (0, 0, 0, 0)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            hue = self._dominant_hue(crop)
            if hue is None:
                continue
            hues.append(hue)
            detection_details.append((track_id, hue))

        if len(detection_details) < 2:
            for index, detection in enumerate(detections):
                track_id = getattr(detection, "track_id", index)
                if track_id in self.manual_assignments:
                    assignments[track_id] = TeamAssignment(track_id=track_id, label=self.manual_assignments[track_id], confidence=1.0, scene_id=scene_id)
                    continue
                assignments[track_id] = TeamAssignment(track_id=track_id, label="unknown", confidence=0.0, scene_id=scene_id)
            return assignments

        cluster_a, cluster_b = self._cluster_hues(hues)
        if np.isclose(cluster_a, cluster_b):
            for track_id, _ in detection_details:
                assignments[track_id] = TeamAssignment(track_id=track_id, label="unknown", confidence=0.2, scene_id=scene_id)
            return assignments

        labels = self._resolve_labels([cluster_a, cluster_b])
        self._team_centers["A"] = cluster_a
        self._team_centers["B"] = cluster_b

        for track_id, hue in detection_details:
            if self.manual_assignments.get(track_id) is not None:
                continue
            if hue <= min(cluster_a, cluster_b):
                label = labels[0]
            elif hue >= max(cluster_a, cluster_b):
                label = labels[1]
            else:
                label = "unknown"
            confidence = max(0.15, 1.0 - min(abs(hue - cluster_a), abs(hue - cluster_b)) / 180.0)
            assignments[track_id] = TeamAssignment(track_id=track_id, label=label, confidence=float(confidence), scene_id=scene_id)

        return assignments
