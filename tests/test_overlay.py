from __future__ import annotations

from unittest.mock import patch

import numpy as np

from cricket_vision.detection.ball_detector import BallDetection
from cricket_vision.tracking.ball_tracker import BallTrackState
from cricket_vision.tracking.player_tracker import TrackedPlayer
from cricket_vision.visualization.overlay import PlayerOverlay


def test_ball_overlay_draws_only_the_tracked_ball_and_its_past_path() -> None:
    first = BallDetection((88, 98, 92, 102), 0.9, 0.9, "scene-0")
    ball = BallDetection((98, 98, 102, 102), 0.9, 1.0, "scene-0")
    state = BallTrackState(ball, (first, ball), 0, False, 0.9, True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    output = PlayerOverlay().draw(
        frame,
        [],
        ball_state=state,
        ball_status="ready",
    )

    magenta = np.all(output == (255, 0, 255), axis=2)
    trail = np.all(output == (0, 180, 255), axis=2)
    assert not np.any(magenta)
    assert np.any(trail[99:101, 90:100])


def test_player_overlay_labels_visible_people_in_order() -> None:
    players = [
        TrackedPlayer(track_id=3, bbox=(10, 20, 50, 100), confidence=0.9, confirmed=True),
        TrackedPlayer(track_id=8, bbox=(60, 20, 100, 100), confidence=0.9, confirmed=True),
    ]

    with patch("cricket_vision.visualization.overlay.cv2.putText") as put_text:
        PlayerOverlay().draw(np.zeros((120, 120, 3), dtype=np.uint8), players)

    labels = [call.args[1] for call in put_text.call_args_list]
    assert "Person 1" in labels
    assert "Person 2" in labels
