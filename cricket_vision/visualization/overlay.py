from __future__ import annotations

import cv2
import numpy as np

from cricket_vision.tracking.ball_tracker import BallTrackState
from cricket_vision.tracking.player_tracker import TrackedPlayer
from cricket_vision.bowling.action_classifier import ActionEstimate
from cricket_vision.bowling.pose_detector import PoseObservation


class PlayerOverlay:
    """Draws player boxes and count information onto frames."""

    def draw(
        self,
        frame: np.ndarray,
        tracked_players: list[TrackedPlayer],
        visible_player_count: int | None = None,
        team_assignments: dict[int, object] | None = None,
        scene_id: str | None = None,
        ball_state: BallTrackState | None = None,
        ball_status: str | None = None,
        ball_warning: str | None = None,
        pose_observation: PoseObservation | None = None,
        action_estimate: ActionEstimate | None = None,
        bowler_track_id: int | None = None,
        pose_warning: str | None = None,
        trained_action_prediction: tuple[str, float] | None = None,
    ) -> np.ndarray:
        output = frame.copy()
        for person_number, player in enumerate(tracked_players, start=1):
            x1, y1, x2, y2 = player.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"Person {person_number}", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if bowler_track_id == player.track_id:
                cv2.putText(output, "Selected bowler", (x1, max(24, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            if team_assignments is not None:
                assignment = team_assignments.get(player.track_id)
                if assignment is not None:
                    label = getattr(assignment, "label", "unknown")
                    conf = getattr(assignment, "confidence", 0.0)
                    if label != "unknown":
                        cv2.putText(output, f"Team {label} {conf:.2f}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        count = visible_player_count if visible_player_count is not None else len(tracked_players)
        cv2.putText(output, f"Visible players: {count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if scene_id is not None:
            cv2.putText(output, f"Scene: {scene_id}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if ball_status == "ready" and ball_state is not None and ball_state.confirmed:
            points = [tuple(map(int, detection.center)) for detection in ball_state.history]
            for start, end in zip(points, points[1:]):
                cv2.line(output, start, end, (0, 180, 255), 2)
            if ball_state.detection is not None:
                x1, y1, x2, y2 = map(int, ball_state.detection.bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    output,
                    f"Ball {ball_state.confidence:.2f}",
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1,
                )
        elif ball_warning:
            cv2.putText(output, ball_warning, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        if pose_observation is not None:
            for name, point in pose_observation.keypoints.items():
                px, py = int(point.x), int(point.y)
                cv2.circle(output, (px, py), 3, (255, 180, 0), -1)
                cv2.putText(output, name, (px + 3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 180, 0), 1)
        if action_estimate is not None:
            label = f"Action: {action_estimate.state} ({action_estimate.confidence:.2f}) experimental"
            cv2.putText(output, label, (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
            if action_estimate.release_candidate_timestamp is not None:
                cv2.putText(output, "Release candidate (unvalidated)", (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        elif pose_warning:
            cv2.putText(output, pose_warning, (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        if trained_action_prediction is not None:
            label, confidence = trained_action_prediction
            cv2.putText(
                output,
                f"Trained action model: {label} ({confidence:.2f})",
                (12, 154),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 0),
                1,
            )
        return output
