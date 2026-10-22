from __future__ import annotations

from cricket_vision.detection.ball_detector import BallDetection
from cricket_vision.tracking.ball_tracker import BallTracker


def _ball(timestamp: float, x: float, confidence: float = 0.9, scene_id: str = "scene-0") -> BallDetection:
    return BallDetection(
        bbox=(x - 2, 8, x + 2, 12),
        confidence=confidence,
        source_timestamp=timestamp,
        scene_id=scene_id,
    )



def test_ball_tracker_associates_consistent_candidate_over_higher_confidence_outlier() -> None:
    tracker = BallTracker(max_speed_px_per_second=100.0, association_margin_px=5.0)
    first = tracker.update([_ball(0.0, 10)], timestamp=0.0, scene_id="scene-0")
    assert first.detection is not None

    next_state = tracker.update(
        [_ball(0.1, 12, 0.65), _ball(0.1, 90, 0.99)],
        timestamp=0.1,
        scene_id="scene-0",
    )
    assert next_state.detection is not None
    assert next_state.detection.center[0] == 12
    assert len(next_state.history) == 2


def test_ball_tracker_keeps_short_gap_then_marks_lost() -> None:
    tracker = BallTracker(max_missed_frames=1, confirmation_hits=2)
    tracker.update([_ball(0.0, 10, confidence=0.5)], timestamp=0.0, scene_id="scene-0")
    confirmed = tracker.update([_ball(0.04, 12, confidence=0.5)], timestamp=0.04, scene_id="scene-0")
    assert confirmed.confirmed is True

    gap = tracker.update([], timestamp=0.08, scene_id="scene-0")
    assert gap.lost is False
    assert gap.missed_frames == 1

    lost = tracker.update([], timestamp=0.12, scene_id="scene-0")
    assert lost.lost is True
    assert lost.history == ()
    assert lost.confirmed is False


def test_ball_tracker_resets_on_scene_change() -> None:
    tracker = BallTracker(max_speed_px_per_second=10.0)
    tracker.update([_ball(0.0, 10)], timestamp=0.0, scene_id="scene-0")

    new_scene = tracker.update([_ball(0.1, 200, scene_id="scene-1")], timestamp=0.1, scene_id="scene-1")
    assert new_scene.detection is not None
    assert len(new_scene.history) == 1
    assert new_scene.history[0].scene_id == "scene-1"


def test_ball_tracker_requires_two_hits_before_confirming_a_ball() -> None:
    tracker = BallTracker(confirmation_hits=2)
    tentative = tracker.update([_ball(0.0, 10, confidence=0.5)], timestamp=0.0, scene_id="scene-0")
    assert tentative.detection is not None
    assert tentative.confirmed is False
    assert tentative.lost is True

    confirmed = tracker.update([_ball(0.04, 12, confidence=0.5)], timestamp=0.04, scene_id="scene-0")
    assert confirmed.detection is not None
    assert confirmed.confirmed is True
    assert confirmed.lost is False


def test_ball_tracker_shows_a_clear_high_confidence_detection_immediately() -> None:
    tracker = BallTracker(confirmation_hits=3, single_detection_confidence=0.65)

    state = tracker.update(
        [_ball(0.0, 10, confidence=0.735)],
        timestamp=0.0,
        scene_id="scene-0",
    )

    assert state.confirmed is True
    assert state.lost is False
    assert state.detection is not None
    assert len(state.history) == 1


def test_ball_tracker_keeps_weak_first_detection_tentative() -> None:
    tracker = BallTracker(confirmation_hits=3, single_detection_confidence=0.65)

    state = tracker.update(
        [_ball(0.0, 10, confidence=0.55)],
        timestamp=0.0,
        scene_id="scene-0",
    )

    assert state.confirmed is False
    assert state.lost is True


def test_ball_tracker_confirms_associated_detections_across_a_short_gap() -> None:
    tracker = BallTracker(max_missed_frames=4, confirmation_hits=2)
    first = tracker.update([_ball(0.0, 10, confidence=0.5)], timestamp=0.0, scene_id="scene-0")
    assert first.lost is True

    tracker.update([], timestamp=0.04, scene_id="scene-0")
    confirmed = tracker.update([_ball(0.08, 12, confidence=0.5)], timestamp=0.08, scene_id="scene-0")

    assert confirmed.confirmed is True
    assert confirmed.lost is False
    assert len(confirmed.history) == 2


def test_ball_tracker_starts_fresh_after_tentative_track_exceeds_missed_limit() -> None:
    tracker = BallTracker(max_missed_frames=1, confirmation_hits=2)
    tracker.update([_ball(0.0, 10, confidence=0.5)], timestamp=0.0, scene_id="scene-0")
    tracker.update([], timestamp=0.04, scene_id="scene-0")
    expired = tracker.update([], timestamp=0.08, scene_id="scene-0")
    assert expired.history == ()

    fresh = tracker.update([_ball(0.12, 200, confidence=0.5)], timestamp=0.12, scene_id="scene-0")
    assert fresh.confirmed is False
    assert len(fresh.history) == 1


def test_ball_tracker_rejects_a_candidate_that_moves_too_far_between_frames() -> None:
    tracker = BallTracker(max_speed_px_per_second=3000.0, confirmation_hits=2)
    tracker.update([_ball(0.0, 10, confidence=0.5)], timestamp=0.0, scene_id="scene-0")

    jumped = tracker.update([_ball(1 / 30, 200)], timestamp=1 / 30, scene_id="scene-0")

    assert jumped.detection is None
    assert jumped.confirmed is False


def test_ball_tracker_rejects_stale_timestamp_candidates() -> None:
    tracker = BallTracker()
    stale = tracker.update([_ball(0.0, 10)], timestamp=1.0, scene_id="scene-0")
    assert stale.detection is None
    assert stale.history == ()


def test_ball_tracker_rejects_candidate_that_jumps_sideways_from_motion() -> None:
    tracker = BallTracker(max_speed_px_per_second=6000.0, confirmation_hits=1)
    for timestamp, x in ((0.0, 10.0), (0.1, 20.0), (0.2, 30.0)):
        tracker.update([_ball(timestamp, x)], timestamp=timestamp, scene_id="scene-0")

    sideways = BallDetection(
        bbox=(38.0, 48.0, 42.0, 52.0),
        confidence=0.99,
        source_timestamp=0.3,
        scene_id="scene-0",
    )
    state = tracker.update([sideways], timestamp=0.3, scene_id="scene-0")

    assert state.detection is None
    assert state.confirmed is True
    assert len(state.history) == 3
