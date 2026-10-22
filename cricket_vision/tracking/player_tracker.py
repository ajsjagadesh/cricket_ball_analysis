from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cricket_vision.detection.player_detector import PlayerDetection


@dataclass
class TrackedPlayer:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    source_timestamp: float | None = None
    scene_id: str | None = None
    last_seen: int = 0
    hits: int = 1
    confirmed: bool = False


class PlayerTracker:
    """Minimal track association based on bounding-box IoU."""

    def __init__(self, max_age: int = 8, iou_threshold: float = 0.15, confirmation_hits: int = 2):
        if max_age < 0 or not 0.0 <= iou_threshold <= 1.0 or confirmation_hits <= 0:
            raise ValueError("Invalid player tracker limits.")
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.confirmation_hits = confirmation_hits
        self._tracks: dict[int, TrackedPlayer] = {}
        self._next_id = 1
        self._frame_index = 0

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def update(self, detections: Iterable[PlayerDetection], source_timestamp: float | None = None) -> list[TrackedPlayer]:
        self._frame_index += 1
        detection_list = list(detections)
        candidate_track_ids = list(self._tracks.keys())
        matched_track_ids: set[int] = set()

        for detection in detection_list:
            best_track_id = None
            best_iou = -1.0
            for track_id in candidate_track_ids:
                if track_id in matched_track_ids:
                    continue
                track = self._tracks[track_id]
                iou = self._iou(track.bbox, detection.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is not None:
                track = self._tracks[best_track_id]
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.source_timestamp = detection.source_timestamp if detection.source_timestamp is not None else source_timestamp
                track.scene_id = detection.scene_id
                track.hits = track.hits + 1 if track.last_seen == 0 else 1
                track.confirmed = track.hits >= self.confirmation_hits
                track.last_seen = 0
                matched_track_ids.add(best_track_id)
            else:
                new_track = TrackedPlayer(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    source_timestamp=detection.source_timestamp if detection.source_timestamp is not None else source_timestamp,
                    scene_id=detection.scene_id,
                    hits=1,
                    confirmed=self.confirmation_hits <= 1,
                )
                self._tracks[self._next_id] = new_track
                matched_track_ids.add(self._next_id)
                self._next_id += 1

        for track in self._tracks.values():
            if track.track_id not in matched_track_ids:
                track.last_seen += 1
                track.hits = 0
                track.confirmed = False

        self._prune_inactive_tracks()
        return self._visible_tracks()

    def _visible_tracks(self) -> list[TrackedPlayer]:
        return [
            track
            for track in self._tracks.values()
            if track.confirmed and track.last_seen == 0
        ]

    def _prune_inactive_tracks(self) -> None:
        for track_id in [track_id for track_id, track in self._tracks.items() if track.last_seen > self.max_age]:
            del self._tracks[track_id]

    @property
    def active_tracks(self) -> list[TrackedPlayer]:
        return self._visible_tracks()

    @property
    def visible_count(self) -> int:
        return len(self._visible_tracks())
