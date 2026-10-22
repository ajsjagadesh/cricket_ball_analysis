from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    frame_index: int
    source_timestamp: float
    image: Optional[np.ndarray] = None
    scene_id: Optional[str] = None


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_seconds: float | None
    codec: str | None = None


class VideoReader:
    """Thin wrapper around OpenCV video reading with timestamp metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._capture: cv2.VideoCapture | None = None
        self._opened = False
        self._frame_index = 0
        self._metadata: VideoMetadata | None = None
        self.open()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Input video not found: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {self.path}")

        self._capture = capture
        self._opened = True
        self._metadata = self._read_metadata()

    def _read_metadata(self) -> VideoMetadata:
        if self._capture is None:
            raise ValueError("Video capture is not initialized.")

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self._capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        duration = None
        if fps > 0 and frame_count is not None:
            duration = float(frame_count) / fps

        codec = None
        if self._capture.get(cv2.CAP_PROP_FOURCC) > 0:
            codec_code = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join(chr((codec_code >> 8 * i) & 0xFF) for i in range(4))

        return VideoMetadata(
            path=str(self.path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            codec=codec,
        )

    def read_next(self) -> Frame | None:
        if self._capture is None:
            raise ValueError("Video capture is not initialized.")

        ok, image = self._capture.read()
        if not ok or image is None:
            return None

        fps = self.metadata.fps
        if fps > 0:
            source_timestamp = float(self._frame_index) / fps
        else:
            pos_msec = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            source_timestamp = float(pos_msec) / 1000.0 if pos_msec > 0 else float(self._frame_index)

        frame = Frame(frame_index=self._frame_index, source_timestamp=source_timestamp, image=image, scene_id=None)
        self._frame_index += 1
        return frame

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read_next()
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
