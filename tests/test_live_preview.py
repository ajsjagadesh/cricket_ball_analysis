from __future__ import annotations

import numpy as np
import pytest

from cricket_vision.visualization import live_preview


def test_desktop_preview_shows_frame_and_closes_on_q(monkeypatch) -> None:
    shown = []
    closed = []
    monkeypatch.setattr(live_preview.cv2, "currentUIFramework", lambda: "WIN32")
    monkeypatch.setattr(live_preview.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(live_preview.cv2, "imshow", lambda _title, frame: shown.append(frame))
    monkeypatch.setattr(live_preview.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(live_preview.cv2, "destroyAllWindows", lambda: closed.append(True))

    preview = live_preview.DesktopPreview()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    assert preview(frame) is False
    assert shown == [frame]
    preview.close()
    assert closed == [True]


def test_desktop_preview_reports_headless_opencv(monkeypatch) -> None:
    monkeypatch.setattr(live_preview.cv2, "currentUIFramework", lambda: "")

    with pytest.raises(RuntimeError, match="GUI-enabled OpenCV"):
        live_preview.DesktopPreview()


def test_streamlit_app_shows_video_and_preview_options() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from pathlib import Path

    app = Path(__file__).parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(app)).run()

    assert not at.exception
    assert at.title[0].value == "Cricket video analysis"
    assert at.button(key="start_analysis").disabled
    assert at.segmented_control(key="preview_mode").value == "Browser"

    at.text_input(key="video_path").set_value("videos/test.mp4").run()
    assert not at.button(key="start_analysis").disabled

    at.segmented_control(key="source_mode").set_value("Upload video").run()
    assert at.button(key="start_analysis").disabled

    at.segmented_control(key="preview_mode").set_value("Desktop window").run()
    assert not at.exception
