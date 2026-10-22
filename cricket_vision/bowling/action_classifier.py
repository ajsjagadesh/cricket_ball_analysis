from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .pose_detector import Keypoint, PoseObservation

ActionStateName = Literal[
    "unknown",
    "run-up",
    "delivery_approaching",
    "release_candidate",
    "follow_through",
]


@dataclass(frozen=True)
class ActionEstimate:
    state: ActionStateName
    confidence: float
    experimental: bool
    release_candidate_timestamp: float | None
    features: dict[str, float | None]
    direction_hypothesis: str
    direction_confidence: float


class BowlingActionClassifier:
    """Transparent, untrained pose-sequence heuristic; not a validated action model."""

    def __init__(
        self,
        window_seconds: float = 1.0,
        wrist_speed_threshold: float = 700.0,
        runup_body_speed_threshold: float = 15.0,
        min_keypoint_confidence: float = 0.4,
    ):
        if window_seconds <= 0 or wrist_speed_threshold <= 0 or runup_body_speed_threshold <= 0:
            raise ValueError("Action window and motion thresholds must be positive.")
        self.window_seconds = float(window_seconds)
        self.wrist_speed_threshold = float(wrist_speed_threshold)
        self.runup_body_speed_threshold = float(runup_body_speed_threshold)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self._observations: deque[PoseObservation] = deque(maxlen=256)
        self._scene_id: str | None = None
        self._last_release_candidate: float | None = None
        self._follow_through_until: float | None = None

    def reset(self) -> None:
        self._observations.clear()
        self._scene_id = None
        self._last_release_candidate = None
        self._follow_through_until = None

    def _valid_point(self, observation: PoseObservation, name: str) -> Keypoint | None:
        point = observation.keypoints.get(name)
        if point is None or point.confidence < self.min_keypoint_confidence:
            return None
        return point

    def _wrist_speed(self, previous: PoseObservation, current: PoseObservation) -> float | None:
        dt = current.source_timestamp - previous.source_timestamp
        if dt <= 0:
            return None
        speeds: list[float] = []
        for side in ("left", "right"):
            old = self._valid_point(previous, f"{side}_wrist")
            new = self._valid_point(current, f"{side}_wrist")
            if old is not None and new is not None:
                speeds.append(math.hypot(new.x - old.x, new.y - old.y) / dt)
        return max(speeds) if speeds else None

    def _elbow_angle(self, observation: PoseObservation) -> float | None:
        angles: list[float] = []
        for side in ("left", "right"):
            shoulder = self._valid_point(observation, f"{side}_shoulder")
            elbow = self._valid_point(observation, f"{side}_elbow")
            wrist = self._valid_point(observation, f"{side}_wrist")
            if shoulder is None or elbow is None or wrist is None:
                continue
            vector_a = (shoulder.x - elbow.x, shoulder.y - elbow.y)
            vector_b = (wrist.x - elbow.x, wrist.y - elbow.y)
            magnitude_a = math.hypot(*vector_a)
            magnitude_b = math.hypot(*vector_b)
            if magnitude_a == 0 or magnitude_b == 0:
                continue
            cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (magnitude_a * magnitude_b)
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
        return max(angles) if angles else None

    @staticmethod
    def _group_center(
        observation: PoseObservation,
        names: tuple[str, ...],
        minimum_confidence: float,
    ) -> tuple[float, float] | None:
        points = [
            point
            for name in names
            if (point := observation.keypoints.get(name)) is not None
            and point.confidence >= minimum_confidence
        ]
        if not points:
            return None
        return (
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )

    def _group_speed(
        self,
        previous: PoseObservation,
        current: PoseObservation,
        names: tuple[str, ...],
    ) -> float | None:
        dt = current.source_timestamp - previous.source_timestamp
        if dt <= 0:
            return None
        previous_center = self._group_center(previous, names, self.min_keypoint_confidence)
        current_center = self._group_center(current, names, self.min_keypoint_confidence)
        if previous_center is None or current_center is None:
            return None
        return math.dist(previous_center, current_center) / dt

    def update(self, observation: PoseObservation | None) -> ActionEstimate:
        if observation is None:
            return ActionEstimate(
                state="unknown",
                confidence=0.0,
                experimental=True,
                release_candidate_timestamp=None,
                features={"wrist_speed_px_s": None, "elbow_angle_degrees": None},
                direction_hypothesis="unknown",
                direction_confidence=0.0,
            )

        if self._scene_id is not None and observation.scene_id != self._scene_id:
            self.reset()
        self._scene_id = observation.scene_id
        if self._observations and observation.source_timestamp <= self._observations[-1].source_timestamp:
            return ActionEstimate(
                state="unknown",
                confidence=0.0,
                experimental=True,
                release_candidate_timestamp=None,
                features={"wrist_speed_px_s": None, "elbow_angle_degrees": None},
                direction_hypothesis="unknown",
                direction_confidence=0.0,
            )

        self._observations.append(observation)
        while (
            len(self._observations) > 1
            and observation.source_timestamp - self._observations[0].source_timestamp > self.window_seconds
        ):
            self._observations.popleft()

        if len(self._observations) < 2:
            speed = None
        else:
            speed = self._wrist_speed(self._observations[-2], self._observations[-1])
        elbow_angle = self._elbow_angle(observation)

        center_speed: float | None = None
        shoulder_speed: float | None = None
        hip_speed: float | None = None
        if len(self._observations) >= 2:
            previous = self._observations[-2]
            shoulder_speed = self._group_speed(
                previous,
                observation,
                ("left_shoulder", "right_shoulder"),
            )
            hip_speed = self._group_speed(previous, observation, ("left_hip", "right_hip"))
            center_speed = self._group_speed(
                previous,
                observation,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
            )

        state: ActionStateName = "unknown"
        confidence = 0.0
        release_timestamp = None
        if self._follow_through_until is not None and observation.source_timestamp <= self._follow_through_until:
            state = "follow_through"
            confidence = 0.45
        elif speed is not None and speed >= self.wrist_speed_threshold and (elbow_angle is None or elbow_angle >= 100):
            state = "release_candidate"
            confidence = min(0.75, 0.5 + (speed - self.wrist_speed_threshold) / self.wrist_speed_threshold * 0.1)
            release_timestamp = observation.source_timestamp
            self._last_release_candidate = observation.source_timestamp
            self._follow_through_until = observation.source_timestamp + 0.4
        elif speed is not None and speed >= self.wrist_speed_threshold * 0.35:
            state = "delivery_approaching"
            confidence = min(0.6, speed / self.wrist_speed_threshold * 0.35)
        elif center_speed is not None and center_speed >= self.runup_body_speed_threshold:
            state = "run-up"
            confidence = min(0.55, center_speed / 100.0)

        return ActionEstimate(
            state=state,
            confidence=float(confidence),
            experimental=True,
            release_candidate_timestamp=release_timestamp,
            features={
                "wrist_speed_px_s": speed,
                "elbow_angle_degrees": elbow_angle,
                "body_center_speed_px_s": center_speed,
                "shoulder_motion_px_s": shoulder_speed,
                "hip_motion_px_s": hip_speed,
            },
            direction_hypothesis="unknown",
            direction_confidence=0.0,
        )
