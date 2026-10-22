from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cricket_vision.config import load_config
from cricket_vision.pipeline import run_analysis
from cricket_vision.visualization.live_preview import DesktopPreview


class BrowserPreview:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.last_update = 0.0

    def __call__(self, frame: np.ndarray) -> bool:
        now = time.monotonic()
        if now - self.last_update >= 0.1:
            self.placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
                alt="Live annotated cricket video",
            )
            self.last_update = now
        return True


class BrowserProgress:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.last_update = 0.0

    def __call__(self, current: int, total: int | None) -> None:
        now = time.monotonic()
        if total and current < total and now - self.last_update < 0.1:
            return

        if total:
            self.progress_bar.progress(
                min(current / total, 1.0),
                text=f"Processing frame {current:,} of {total:,}",
            )
        else:
            self.progress_bar.progress(0.0, text=f"Processing frame {current:,}")
        self.last_update = now


st.set_page_config(page_title="Cricket Vision", page_icon=":material/sports_cricket:", layout="wide")
st.title("Cricket video analysis")
st.write("Watch the annotated frames as your video is analyzed and save the finished output.")

source_mode = st.segmented_control(
    "Video source",
    ["File path", "Upload video"],
    default="File path",
    key="source_mode",
)
if source_mode == "Upload video":
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mkv", "mov", "avi"],
        key="video_upload",
    )
    video_path = None
else:
    uploaded_video = None
    video_path = st.text_input(
        "Video file path",
        placeholder="C:\\path\\to\\match.mp4",
        key="video_path",
    )

preview_mode = st.segmented_control(
    "Live output",
    ["Browser", "Desktop window"],
    default="Browser",
    key="preview_mode",
)
output_path = st.text_input(
    "Save annotated video as",
    value="output/annotated_live.mp4",
    key="output_path",
)
start_analysis = st.button(
    "Analyze video",
    type="primary",
    disabled=not (uploaded_video is not None or bool(video_path and video_path.strip())),
    key="start_analysis",
)

if start_analysis:
    with tempfile.TemporaryDirectory(prefix="cricket-vision-") as temp_dir:
        if uploaded_video is not None:
            source_path = Path(temp_dir) / f"input{Path(uploaded_video.name).suffix.lower()}"
            source_path.write_bytes(uploaded_video.getvalue())
        else:
            source_path = Path(video_path).expanduser()
            if not source_path.is_file():
                st.error(f"Video file not found: {source_path}")
                st.stop()

        target_path = Path(output_path).expanduser()
        if target_path.suffix.lower() != ".mp4":
            st.error("Choose an output file name ending in .mp4.")
            st.stop()

        config = load_config()
        config.input_path = str(source_path)
        config.output_path = str(target_path)
        progress_bar = st.progress(0.0, text="Preparing video...")
        progress = BrowserProgress(progress_bar)
        preview_area = st.empty()
        status = st.status("Analyzing video", expanded=True)
        desktop_preview = DesktopPreview() if preview_mode == "Desktop window" else None
        frame_preview = desktop_preview or BrowserPreview(preview_area)

        try:
            result = run_analysis(
                source_path,
                target_path,
                config,
                progress_callback=progress,
                frame_callback=frame_preview,
            )
        except Exception:
            status.update(label="Video analysis failed", state="error")
            raise
        finally:
            if desktop_preview is not None:
                desktop_preview.close()

        if result["stopped_early"]:
            status.update(label="Analysis stopped", state="complete")
            st.info("Stopped early. The video processed so far was saved.")
        else:
            status.update(label="Analysis complete", state="complete")
            progress_bar.progress(1.0, text="Analysis complete")

        st.success(f"Annotated video saved to {target_path}")
        st.video(str(target_path), alt="Annotated cricket video")
