from __future__ import annotations

from cricket_vision.bowling.action_classifier import BowlingActionClassifier
from cricket_vision.bowling.bowler_selector import BowlerSelector
from cricket_vision.bowling.pose_detector import Keypoint, PoseDetector, PoseObservation
from cricket_vision.tracking.player_tracker import TrackedPlayer


def _pose(timestamp: float, wrist_x: float, scene_id: str = "scene-0") -> PoseObservation:
    points = {
        "left_shoulder": Keypoint(0.0, 0.0, 0.95),
        "left_elbow": Keypoint(1.0, 0.0, 0.95),
        "left_wrist": Keypoint(wrist_x, 0.0, 0.95),
        "left_hip": Keypoint(2.0, 10.0, 0.95),
        "right_hip": Keypoint(4.0, 10.0, 0.95),
        "right_shoulder": Keypoint(6.0, 0.0, 0.95),
    }
    return PoseObservation(
        track_id=3,
        source_timestamp=timestamp,
        scene_id=scene_id,
        bbox=(0, 0, 20, 40),
        keypoints=points,
    )


def test_manual_bowler_selector_requires_selected_track() -> None:
    tracks = [
        TrackedPlayer(track_id=2, bbox=(0, 0, 10, 20), confidence=0.8),
        TrackedPlayer(track_id=3, bbox=(10, 0, 20, 20), confidence=0.9),
    ]
    assert BowlerSelector().select(tracks) is None
    assert BowlerSelector(3).select(tracks) == tracks[1]
    assert BowlerSelector(9).select(tracks) is None


def test_pose_detector_reports_missing_model_without_loading_checkpoint(tmp_path) -> None:
    detector = PoseDetector(tmp_path / "missing-pose.pt")
    assert detector.status == "disabled"
    assert "not found" in detector.reason


def test_action_classifier_marks_fast_wrist_motion_as_experimental_candidate() -> None:
    classifier = BowlingActionClassifier(wrist_speed_threshold=700.0)
    first = classifier.update(_pose(0.0, 10.0))
    second = classifier.update(_pose(0.1, 110.0))

    assert first.state == "unknown"
    assert second.state == "release_candidate"
    assert second.experimental is True
    assert second.release_candidate_timestamp == 0.1
    assert second.direction_hypothesis == "unknown"


def test_action_classifier_resets_temporal_state_on_scene_change() -> None:
    classifier = BowlingActionClassifier(wrist_speed_threshold=700.0)
    classifier.update(_pose(0.0, 10.0))
    changed_scene = classifier.update(_pose(1.0, 200.0, scene_id="scene-1"))
    assert changed_scene.state == "unknown"
