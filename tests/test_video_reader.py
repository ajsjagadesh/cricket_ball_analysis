from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cricket_vision.video.reader import VideoReader


def _write_test_video(path: Path, frame_count: int = 5, width: int = 64, height: int = 64, fps: int = 12) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to create synthetic video for tests.")
    for idx in range(frame_count):
        frame = np.full((height, width, 3), 50 + idx * 10, dtype=np.uint8)
        cv2.putText(frame, str(idx), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_video_reader_reads_frames_and_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    _write_test_video(video_path)

    with VideoReader(video_path) as reader:
        frames = list(reader)
        assert len(frames) == 5
        assert frames[0].frame_index == 0
        assert frames[1].source_timestamp == pytest.approx(1 / 12, rel=1e-2)
        assert reader.metadata.width == 64
        assert reader.metadata.height == 64
        assert reader.metadata.fps == pytest.approx(12.0)
        assert reader.metadata.frame_count == 5


def test_video_reader_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        VideoReader("videos/definitely_missing.mp4")
