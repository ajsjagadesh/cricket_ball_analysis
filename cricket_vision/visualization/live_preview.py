from __future__ import annotations

import cv2
import numpy as np


class DesktopPreview:
    """Show annotated frames in a desktop window. Press Q or Esc to stop."""

    def __init__(self, title: str = "Cricket Vision"):
        if not cv2.currentUIFramework():
            raise RuntimeError(
                "Desktop preview needs a GUI-enabled OpenCV build. "
                "Install opencv-python instead of opencv-python-headless."
            )
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def __call__(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        cv2.destroyAllWindows()
