from __future__ import annotations

from cricket_vision.detection.player_detector import PlayerDetection
from cricket_vision.tracking.player_tracker import PlayerTracker


def _player(x1: int = 10) -> PlayerDetection:
    return PlayerDetection(
        x1=x1,
        y1=20,
        x2=x1 + 40,
        y2=120,
        confidence=0.8,
        scene_id="scene-0",
    )


def test_player_tracker_waits_for_stable_detection_and_hides_stale_boxes() -> None:
    tracker = PlayerTracker(confirmation_hits=2)

    assert tracker.update([_player()]) == []
    visible = tracker.update([_player(x1=11)])
    assert len(visible) == 1
    assert visible[0].confirmed is True

    assert tracker.update([]) == []
    assert tracker.visible_count == 0


def test_player_tracker_reconfirms_after_a_detection_gap() -> None:
    tracker = PlayerTracker(confirmation_hits=2)
    tracker.update([_player()])
    tracker.update([_player()])

    assert tracker.update([]) == []
    assert tracker.update([_player()]) == []
    assert len(tracker.update([_player()])) == 1
