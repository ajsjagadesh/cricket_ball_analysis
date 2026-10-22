from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cricket_vision.config import AppConfig
from cricket_vision.detection.ball_detector import BallDetection
from cricket_vision.pipeline import run_analysis


def _write_moving_player_video(path: Path, frame_count: int = 10, width: int = 320, height: int = 240) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to create synthetic movement video")

    for idx in range(frame_count):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        x = 20 + idx * 12
        y = 50
        cv2.rectangle(frame, (x, y), (x + 30, y + 80), (30, 30, 30), -1)
        cv2.circle(frame, (x + 15, y - 10), 12, (80, 80, 80), -1)
        writer.write(frame)
    writer.release()


def test_run_analysis_creates_annotated_output(tmp_path: Path) -> None:
    input_path = tmp_path / "moving_player.mp4"
    output_path = tmp_path / "output" / "annotated.mp4"
    _write_moving_player_video(input_path)

    progress_events: list[tuple[int, int | None]] = []
    preview_frames: list[np.ndarray] = []
    result = run_analysis(
        input_path,
        output_path,
        progress_callback=lambda current, total: progress_events.append((current, total)),
        frame_callback=lambda image: preview_frames.append(image.copy()),
    )

    assert result["status"] == "success"
    assert output_path.exists()
    assert result["processed_frames"] >= 1
    assert progress_events[-1] == (result["processed_frames"], result["processed_frames"])
    assert len(preview_frames) == result["processed_frames"]
    assert preview_frames[0].shape == (240, 320, 3)
    assert result["visible_player_count"] == 0
    assert result["player_detection_status"] == "disabled"
    assert result["ball_detection_status"] == "disabled"
    assert result["pose_detection_status"] == "disabled"
    assert Path(result["sidecar_path"]).exists()
    first_record = json.loads(Path(result["sidecar_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert "ball_track_points_2d" in first_record
    assert "predicted_ball_points_2d" not in first_record
    assert "prediction_horizon_seconds" not in first_record
    assert result["performance"]["processed_frames"] == result["processed_frames"]
    assert "video_decode" in result["performance"]["stages"]
    assert "video_encoding" in result["performance"]["stages"]

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 1
    finally:
        capture.release()


def test_run_analysis_can_stop_after_a_preview_frame(tmp_path: Path) -> None:
    input_path = tmp_path / "moving_player.mp4"
    output_path = tmp_path / "output" / "partial.mp4"
    _write_moving_player_video(input_path, frame_count=10)
    shown_frames: list[int] = []

    def stop_after_three(_image: np.ndarray) -> bool:
        shown_frames.append(1)
        return len(shown_frames) < 3

    result = run_analysis(input_path, output_path, frame_callback=stop_after_three)

    assert result["status"] == "success"
    assert result["stopped_early"] is True
    assert result["processed_frames"] == 3
    assert output_path.exists()


def test_run_analysis_runs_ball_detector_on_every_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "moving_ball.mp4"
    output_path = tmp_path / "output" / "tracked_ball.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (200, 120),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create synthetic ball video")
    for x in (60, 65, 70, 75, 80):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (x, 60), 10, (20, 20, 180), -1)
        cv2.ellipse(frame, (x, 60), (8, 4), 35, 0, 360, (240, 240, 240), 2)
        cv2.circle(frame, (x - 3, 57), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    class FakeBallDetector:
        status = "ready"
        reason = ""

        def __init__(self, **_kwargs):
            self.calls = 0

        def detect(self, _frame, timestamp, scene_id):
            self.calls += 1
            if self.calls == 1:
                return [
                    BallDetection(
                        bbox=(48, 48, 72, 72),
                        confidence=0.9,
                        source_timestamp=timestamp,
                        scene_id=scene_id,
                    )
                ]
            return []

    created_detectors = []

    def create_detector(**kwargs):
        detector = FakeBallDetector(**kwargs)
        created_detectors.append(detector)
        return detector

    monkeypatch.setattr("cricket_vision.pipeline.BallDetector", create_detector)
    result = run_analysis(
        input_path,
        output_path,
        config=AppConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            ball_confirmation_hits=3,
        ),
    )

    records = [
        json.loads(line)
        for line in Path(result["sidecar_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert result["processed_frames"] == 5
    assert created_detectors[0].calls == 5
    assert [record["ball_tracking"]["observed"] for record in records] == [True, False, False, False, False]
