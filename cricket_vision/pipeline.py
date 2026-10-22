from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Callable, Iterator

import numpy as np

from .config import AppConfig
from .bowling.action_classifier import BowlingActionClassifier
from .bowling.bowler_selector import BowlerSelector
from .bowling.pose_detector import PoseDetector
from .bowling.temporal_training import TemporalActionPredictor
from .detection.ball_detector import BallDetector
from .detection.player_detector import PlayerDetector
from .scene_detector import SceneDetector
from .team_classifier import TeamClassifier
from .tracking.ball_tracker import BallTracker
from .tracking.player_tracker import PlayerTracker
from .video.reader import VideoReader
from .visualization.overlay import PlayerOverlay
from .visualization.video_writer import VideoWriter
from .performance.metrics import MetricsCollector
from .video.reader import Frame


def _timed_frames(reader: VideoReader, metrics: MetricsCollector) -> Iterator[Frame]:
    frames = iter(reader)
    while True:
        started = time.perf_counter()
        try:
            frame = next(frames)
        except StopIteration:
            metrics.record("video_decode", time.perf_counter() - started)
            return
        metrics.record("video_decode", time.perf_counter() - started)
        yield frame


def run_analysis(
    video_path: str | Path,
    output_path: str | Path,
    config: AppConfig | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    frame_callback: Callable[[np.ndarray], bool | None] | None = None,
) -> dict:
    """Minimal end-to-end player detection, tracking, and annotation pipeline."""

    if config is None:
        config = AppConfig(input_path=str(video_path), output_path=str(output_path))

    config.validate()
    metrics = MetricsCollector()
    metrics.start()
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    with VideoReader(input_path) as reader:
        metadata = reader.metadata
        width = metadata.width or 640
        height = metadata.height or 480
        fps = metadata.fps if metadata.fps > 0 else 25.0

        detector = PlayerDetector(
            confidence_threshold=config.player_confidence_threshold,
            model_path=config.pose_model_path,
            inference_resolution=config.pose_inference_resolution,
            device=config.device,
        )
        tracker = PlayerTracker()
        overlay = PlayerOverlay()
        bowler_selector = BowlerSelector(config.bowler_track_id)
        pose_detector = PoseDetector(
            model_path=config.pose_model_path,
            confidence_threshold=config.pose_confidence_threshold,
            inference_resolution=config.pose_inference_resolution,
            device=config.device,
            model_instance=detector.model,
        )
        action_classifier = BowlingActionClassifier(
            window_seconds=config.action_window_seconds,
            wrist_speed_threshold=config.release_wrist_speed_threshold,
            runup_body_speed_threshold=config.runup_body_speed_threshold,
            min_keypoint_confidence=config.pose_confidence_threshold,
        )
        action_predictor = (
            TemporalActionPredictor(config.action_model_path)
            if config.action_model_path is not None
            else None
        )
        scene_detector = SceneDetector(scene_cut_threshold=config.scene_cut_threshold)
        team_classifier = TeamClassifier()
        ball_detector = BallDetector(
            model_path=config.ball_model_path,
            confidence_threshold=config.ball_confidence_threshold,
            inference_resolution=config.ball_inference_resolution,
            device=config.device,
            max_bbox_area_fraction=config.ball_max_bbox_area_fraction,
            max_bbox_dimension_fraction=config.ball_max_bbox_dimension_fraction,
            max_aspect_ratio=config.ball_max_aspect_ratio,
        )
        ball_tracker = BallTracker(
            max_history=config.trajectory_history_length,
            max_missed_frames=config.max_missed_frames,
            max_speed_px_per_second=config.ball_max_speed_px_per_second,
            confirmation_hits=config.ball_confirmation_hits,
        )
        output_target = Path(output_path)
        output_target.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(output_target, fps=fps, width=width, height=height)
        sidecar_path = output_target.with_suffix(".jsonl")

        processed_frames = 0
        peak_visible_players = 0
        stopped_early = False
        try:
            with sidecar_path.open("w", encoding="utf-8") as sidecar:
                for frame in _timed_frames(reader, metrics):
                    image = frame.image
                    if image is None:
                        continue

                    stage_started = time.perf_counter()
                    scene_decision = scene_detector.process(image)
                    metrics.record("scene_detection", time.perf_counter() - stage_started)
                    frame.scene_id = scene_decision.scene_id
                    if scene_decision.is_cut:
                        tracker = PlayerTracker()
                        ball_tracker.reset()
                        action_classifier.reset()
                        if action_predictor is not None:
                            action_predictor.reset()

                    stage_started = time.perf_counter()
                    detections = detector.detect(image, source_timestamp=frame.source_timestamp, scene_id=frame.scene_id)
                    metrics.record("player_detection", time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    active_tracks = tracker.update(detections, source_timestamp=frame.source_timestamp)
                    visible_count = len(active_tracks)
                    peak_visible_players = max(peak_visible_players, visible_count)
                    team_assignments = team_classifier.classify_frame(image, active_tracks, scene_id=frame.scene_id)
                    metrics.record("player_tracking_and_teams", time.perf_counter() - stage_started)
                    selected_bowler = bowler_selector.select(active_tracks)
                    pose_observation = None
                    stage_started = time.perf_counter()
                    if pose_detector.status == "ready" and selected_bowler is not None:
                        pose_observation = pose_detector.detect(
                            image,
                            selected_bowler,
                            frame.source_timestamp,
                            frame.scene_id,
                        )
                    metrics.record("pose_inference", time.perf_counter() - stage_started)
                    action_estimate = action_classifier.update(pose_observation)
                    trained_action_prediction = (
                        action_predictor.predict(action_estimate.features)
                        if action_predictor is not None and pose_observation is not None
                        else None
                    )

                    stage_started = time.perf_counter()
                    ball_detections = ball_detector.detect(image, frame.source_timestamp, frame.scene_id)
                    metrics.record("ball_detection", time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    ball_state = ball_tracker.update(
                        ball_detections,
                        frame.source_timestamp,
                        frame.scene_id,
                    )
                    metrics.record("ball_tracking_and_association", time.perf_counter() - stage_started)

                    stage_started = time.perf_counter()
                    annotated = overlay.draw(
                        image,
                        active_tracks,
                        visible_count,
                        team_assignments=team_assignments,
                        scene_id=frame.scene_id,
                        ball_state=ball_state,
                        ball_status=ball_detector.status,
                        ball_warning=ball_detector.reason if ball_detector.status != "ready" else None,
                        pose_observation=pose_observation,
                        action_estimate=action_estimate,
                        bowler_track_id=bowler_selector.selected_track_id,
                        pose_warning=(
                            pose_detector.reason
                            if pose_detector.status != "ready"
                            else "Choose a bowler using --bowler-track-id."
                            if bowler_selector.selected_track_id is None
                            else None
                        ),
                        trained_action_prediction=trained_action_prediction,
                    )
                    metrics.record("visualization", time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    writer.write(annotated)
                    metrics.record("video_encoding", time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    record = {
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": frame.source_timestamp,
                        "scene_id": frame.scene_id,
                        "visible_player_count_estimate": visible_count,
                        "player_detector_status": detector.status,
                        "ball_detector_status": ball_detector.status,
                        "ball_detections": [
                            {
                                "bbox": detection.bbox,
                                "confidence": detection.confidence,
                                "scene_id": detection.scene_id,
                            }
                            for detection in ball_detections
                        ],
                        "ball_tracking": {
                            "observed": ball_state.detection is not None,
                            "confirmed": ball_state.confirmed,
                            "lost": ball_state.lost,
                            "missed_frames": ball_state.missed_frames,
                            "history_observations": len(ball_state.history),
                            "confidence": ball_state.confidence,
                        },
                        "ball_track_points_2d": [
                            detection.center for detection in ball_state.history
                        ],
                        "bowler_track_id": bowler_selector.selected_track_id,
                        "pose_detector_status": pose_detector.status,
                        "pose_keypoints": (
                            {
                                name: {"x": point.x, "y": point.y, "confidence": point.confidence}
                                for name, point in pose_observation.keypoints.items()
                            }
                            if pose_observation is not None
                            else None
                        ),
                        "bowling_action": {
                            "state": action_estimate.state,
                            "confidence": action_estimate.confidence,
                            "experimental": action_estimate.experimental,
                            "release_candidate_timestamp": action_estimate.release_candidate_timestamp,
                            "direction_hypothesis": action_estimate.direction_hypothesis,
                            "direction_confidence": action_estimate.direction_confidence,
                            "features": action_estimate.features,
                        },
                        "trained_action_prediction": (
                            {
                                "label": trained_action_prediction[0],
                                "confidence": trained_action_prediction[1],
                            }
                            if trained_action_prediction is not None
                            else None
                        ),
                    }
                    sidecar.write(json.dumps(record, separators=(",", ":")) + "\n")
                    metrics.record("sidecar_write", time.perf_counter() - stage_started)
                    processed_frames += 1
                    metrics.frame_processed()
                    if progress_callback is not None:
                        progress_callback(processed_frames, metadata.frame_count)
                    if frame_callback is not None and frame_callback(annotated) is False:
                        stopped_early = True
                        break
        finally:
            writer.close()

    return {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(output_target),
        "sidecar_path": str(sidecar_path),
        "device": config.device,
        "ball_detection_status": ball_detector.status,
        "ball_detection_message": ball_detector.reason or "Custom ball model loaded.",
        "player_detection_status": detector.status,
        "player_detection_message": detector.reason or "Person-detection model loaded.",
        "pose_detection_status": pose_detector.status,
        "pose_detection_message": (
            pose_detector.reason
            if pose_detector.status != "ready"
            else "Pose model loaded; action states are experimental."
        ),
        "bowler_track_id": bowler_selector.selected_track_id,
        "temporal_action_model_status": "ready" if action_predictor is not None else "disabled",
        "processed_frames": processed_frames,
        "stopped_early": stopped_early,
        "visible_player_count": peak_visible_players,
        "performance": metrics.summary(),
        "message": (
            "Video analysis completed. Ball boxes and their past track are shown only after "
            f"{config.ball_confirmation_hits} detections are associated into one track."
        ),
    }
