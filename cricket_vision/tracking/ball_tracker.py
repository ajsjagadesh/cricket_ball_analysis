from __future__ import annotations

import math
from dataclasses import dataclass

from cricket_vision.detection.ball_detector import BallDetection


@dataclass(frozen=True)
class BallTrackState:
    detection: BallDetection | None
    history: tuple[BallDetection, ...]
    missed_frames: int
    lost: bool
    confidence: float
    confirmed: bool

    @property
    def center(self) -> tuple[float, float] | None:
        return self.detection.center if self.detection is not None else None


class BallTracker:
    """Single-ball association using timestamped image-plane motion."""

    def __init__(
        self,
        max_history: int = 20,
        max_missed_frames: int = 8,
        max_speed_px_per_second: float = 3000.0,
        association_margin_px: float = 12.0,
        confirmation_hits: int = 3,
        single_detection_confidence: float = 0.65,
    ):
        if (
            max_history <= 0
            or max_missed_frames < 0
            or max_speed_px_per_second <= 0
            or association_margin_px < 0
            or confirmation_hits <= 0
            or not 0.0 <= single_detection_confidence <= 1.0
        ):
            raise ValueError("Invalid ball tracker limits.")
        self.max_history = max_history
        self.max_missed_frames = max_missed_frames
        self.max_speed_px_per_second = max_speed_px_per_second
        self.association_margin_px = association_margin_px
        self.confirmation_hits = confirmation_hits
        self.single_detection_confidence = single_detection_confidence
        self._history: list[BallDetection] = []
        self._missed_frames = 0
        self._scene_id: str | None = None
        self._consecutive_hits = 0
        self._confirmed = False

    @property
    def history(self) -> tuple[BallDetection, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._missed_frames = 0
        self._scene_id = None
        self._consecutive_hits = 0
        self._confirmed = False

    def _predicted_motion(self, timestamp: float) -> tuple[float, float, float, float]:
        samples = self._history[-5:]
        last = samples[-1]
        last_x, last_y = last.center
        if len(samples) < 2:
            return last_x, last_y, 0.0, 0.0

        origin = last.source_timestamp
        times = [item.source_timestamp - origin for item in samples]
        elapsed = timestamp - origin
        mean_time = sum(times) / len(times)
        denominator = sum((item - mean_time) ** 2 for item in times)
        if denominator <= 0.0:
            return last_x, last_y, 0.0, 0.0

        centers = [item.center for item in samples]
        mean_x = sum(center[0] for center in centers) / len(centers)
        mean_y = sum(center[1] for center in centers) / len(centers)
        velocity_x = sum(
            (time - mean_time) * (center[0] - mean_x)
            for time, center in zip(times, centers)
        ) / denominator
        velocity_y = sum(
            (time - mean_time) * (center[1] - mean_y)
            for time, center in zip(times, centers)
        ) / denominator
        return (
            last_x + velocity_x * elapsed,
            last_y + velocity_y * elapsed,
            velocity_x,
            velocity_y,
        )

    def update(
        self,
        candidates: list[BallDetection],
        timestamp: float,
        scene_id: str,
    ) -> BallTrackState:
        if self._scene_id is not None and scene_id != self._scene_id:
            self.reset()
        self._scene_id = scene_id
        candidates = [
            candidate
            for candidate in candidates
            if candidate.scene_id == scene_id
            and math.isfinite(candidate.source_timestamp)
            and math.isfinite(candidate.confidence)
            and 0.0 <= candidate.confidence <= 1.0
            and abs(candidate.source_timestamp - timestamp) <= 0.05
        ]
        chosen: BallDetection | None = None

        if not self._history and candidates:
            chosen = max(candidates, key=lambda candidate: candidate.confidence)
        elif self._history and candidates and self._missed_frames <= self.max_missed_frames:
            last = self._history[-1]
            dt = timestamp - last.source_timestamp
            if dt > 0.0:
                predicted_x, predicted_y, velocity_x, velocity_y = self._predicted_motion(timestamp)
                max_distance = self.max_speed_px_per_second * dt + self.association_margin_px
                speed = math.hypot(velocity_x, velocity_y)
                cross_track_limit = max(
                    self.association_margin_px * 2.0,
                    speed * dt * 0.35 + self.association_margin_px,
                )
                valid = []
                for candidate in candidates:
                    cx, cy = candidate.center
                    offset_x = cx - predicted_x
                    offset_y = cy - predicted_y
                    distance = math.hypot(offset_x, offset_y)
                    cross_track_distance = (
                        abs(offset_x * velocity_y - offset_y * velocity_x) / speed
                        if speed > 1e-6
                        else 0.0
                    )
                    if distance <= max_distance and cross_track_distance <= cross_track_limit:
                        area_ratio = (
                            ((candidate.bbox[2] - candidate.bbox[0]) * (candidate.bbox[3] - candidate.bbox[1]))
                            / max(
                                (last.bbox[2] - last.bbox[0]) * (last.bbox[3] - last.bbox[1]),
                                1.0,
                            )
                        )
                        size_penalty = abs(math.log(max(area_ratio, 1e-6)))
                        score = distance / max(max_distance, 1.0) + 0.04 * size_penalty - 0.02 * candidate.confidence
                        valid.append((score, distance, candidate))
                if valid:
                    chosen = min(valid, key=lambda pair: (pair[0], pair[1]))[2]

        if chosen is not None:
            if self._history and chosen.source_timestamp <= self._history[-1].source_timestamp:
                chosen = None
            else:
                self._history.append(chosen)
                self._history = self._history[-self.max_history :]
                self._missed_frames = 0
                self._consecutive_hits += 1
                self._confirmed = (
                    self._consecutive_hits >= self.confirmation_hits
                    or (
                        len(self._history) == 1
                        and chosen.confidence >= self.single_detection_confidence
                    )
                )
        else:
            # Keep tentative observations through short detector dropouts too.
            # Confirmation still requires the configured number of associated
            # detections; a missed frame does not count as a hit.
            self._missed_frames += 1
            if self._missed_frames > self.max_missed_frames:
                self.reset()

        active_detection = (
            self._history[-1]
            if self._confirmed and self._history and self._missed_frames <= self.max_missed_frames
            else None
        )
        confidence = active_detection.confidence if active_detection is not None else 0.0
        return BallTrackState(
            detection=chosen,
            history=tuple(self._history),
            missed_frames=self._missed_frames,
            lost=active_detection is None,
            confidence=confidence,
            confirmed=self._confirmed,
        )
