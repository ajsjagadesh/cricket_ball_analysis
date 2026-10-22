from __future__ import annotations

import json
from pathlib import Path

import cv2


def label_release_timestamps(
    video_path: str | Path,
    output_path: str | Path,
    timestamps: list[float],
    bowler_track_id: int | None = None,
) -> int:
    """Write release labels for manually observed timestamps without requiring a GUI."""

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if bowler_track_id is not None and bowler_track_id <= 0:
        raise ValueError("bowler_track_id must be positive.")
    if not timestamps:
        raise ValueError("At least one release timestamp is required.")
    if any(not (0.0 <= timestamp < float("inf")) for timestamp in timestamps):
        raise ValueError("Release timestamps must be finite and non-negative.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise ValueError("Video FPS is unavailable; cannot map timestamps to frame indices.")
        duration = frame_count / fps if frame_count > 0 else None
        for timestamp in timestamps:
            if duration is not None and timestamp >= duration:
                raise ValueError(f"Release timestamp {timestamp} is outside video duration {duration:.3f}s.")
    finally:
        capture.release()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for timestamp in sorted(timestamps):
            output.write(
                json.dumps(
                    {
                        "frame_index": round(timestamp * fps),
                        "timestamp_seconds": timestamp,
                        "label": "release",
                        "label_source": "manual_timestamp",
                        "bowler_track_id": bowler_track_id,
                        "video": str(source),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return len(timestamps)


def label_release_frames(
    video_path: str | Path,
    output_path: str | Path,
    bowler_track_id: int | None = None,
) -> int:
    """Review video frame-by-frame and record manually confirmed release timestamps.

    Press Space to label the currently displayed frame, N/any other key to continue,
    and Q/Esc to finish. This tool creates labels; it does not run a release detector.
    """

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if bowler_track_id is not None and bowler_track_id <= 0:
        raise ValueError("bowler_track_id must be positive.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {source}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    labels_written = 0
    try:
        with target.open("w", encoding="utf-8") as output:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                shown = frame.copy()
                cv2.putText(
                    shown,
                    "SPACE: mark release | N: next frame | Q/ESC: finish",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Cricket release labeling", shown)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    timestamp = frame_index / fps if fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    label = {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "label": "release",
                        "label_source": "manual",
                        "bowler_track_id": bowler_track_id,
                        "video": str(source),
                    }
                    output.write(json.dumps(label, separators=(",", ":")) + "\n")
                    output.flush()
                    labels_written += 1
                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return labels_written
