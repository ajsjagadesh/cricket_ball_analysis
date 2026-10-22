from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cricket_vision.bowling.release_labeler import label_release_timestamps


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("Unable to create test video.")
    for _ in range(24):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()


def test_timestamp_labels_write_sorted_jsonl_with_frame_indices(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    output = tmp_path / "labels.jsonl"
    _write_video(video)

    count = label_release_timestamps(video, output, [0.5, 0.25], bowler_track_id=4)
    labels = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert count == 2
    assert [label["frame_index"] for label in labels] == [3, 6]
    assert [label["timestamp_seconds"] for label in labels] == [0.25, 0.5]
    assert all(label["label_source"] == "manual_timestamp" for label in labels)
    assert all(label["bowler_track_id"] == 4 for label in labels)


def test_timestamp_labels_reject_times_outside_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)

    with pytest.raises(ValueError, match="outside video duration"):
        label_release_timestamps(video, tmp_path / "labels.jsonl", [4.0])


def test_timestamp_labels_require_finite_nonnegative_times(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)

    with pytest.raises(ValueError, match="finite and non-negative"):
        label_release_timestamps(video, tmp_path / "labels.jsonl", [float("nan")])
