from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps > 0 else 25.0
        self.width = int(width)
        self.height = int(height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Unable to initialize video writer for {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
