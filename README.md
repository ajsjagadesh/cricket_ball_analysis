# Cricket Vision AI

This project is a Windows 11, Python-first cricket video analysis toolkit. It is being developed incrementally according to the specification in this workspace.

## Current phase

Phase 1 is complete and includes the project skeleton, configuration loader, environment diagnostics, video reader, CLI, and automated tests.

Phase 2 includes person detection using the locally configured Ultralytics pose checkpoint, an IoU tracker, visible-count overlay, and output writer. The person detector uses only the model's `person` class; generic frame motion and bright regions are no longer shown as player detections.

Phase 3 and 4 are also implemented with a basic jersey-color team classifier and scene-cut detection, both tested with synthetic fixtures.

Phase 5 adds an optional custom cricket-ball detector interface, image-plane ball association/tracking, short-term 2D trajectory overlays, JSONL metadata, and a user-supplied dataset training workflow.

Phase 6 adds optional pose inference for a manually selected player track, pose-sequence features, clearly experimental delivery-state heuristics, and a manual release-frame labeling command.

Phases 7–9 add delivery-grouped temporal action model training/evaluation, explicit native-vs-Python trajectory benchmarking, end-to-end performance metrics, and regression tests. The workflow is implemented; measured cricket accuracy remains dependent on user-provided labeled footage and checkpoints.

## Project structure

- `cricket_vision/` — Python package
- `config/default.yaml` — application defaults
- `tests/` — pytest suite
- `scripts/` — Windows PowerShell helpers
- `models/` — model checkpoints and weights
- `datasets/` — dataset instructions and labels
- `videos/` — sample inputs
- `output/` — generated annotated output

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional model workflows have separate dependencies:

```powershell
python -m pip install -r requirements-ball.txt
python -m pip install -r requirements-action.txt
```

## Environment check

```powershell
python -m cricket_vision.cli check-env
```

## Build info

```powershell
python -m cricket_vision.cli build-info
```

## Build the optional C++ trajectory extension

The trajectory helper uses the compiled pybind11 module when it is available and otherwise reports/uses the Python fallback.

```powershell
.\scripts\build_cpp.ps1
```

The script discovers Visual Studio C++ Build Tools, configures CMake with the selected Python interpreter and pybind11, then places the `.pyd` inside `cricket_vision/cpp/`. The extension is imported automatically by the Python package. Check `python -m cricket_vision.cli check-env` for `cpp_extension_available`; an analysis result also reports `cpp_acceleration` as `native` or `python_fallback`.

## Analyze a video

```powershell
python -m cricket_vision.cli analyze --video "videos/match.mp4" --output "output/annotated.mp4"
```

The CLI displays a live terminal progress bar while rendering frames. When the video backend reports a frame count it shows percentage complete and frames processed; if the total is unknown it shows the current frame count, processing FPS, and elapsed time. A completion/failure status is printed before the JSON result.

### Watch the annotated video while it runs

Install the optional live-preview dependencies:

```powershell
python -m pip install -r requirements-live.txt
```

To use the browser preview, run:

```powershell
python -m streamlit run streamlit_app.py
```

Choose a video file or enter its path, then choose **Browser** or **Desktop window**.
The finished annotated MP4 is saved at the output path shown in the app.

To open the desktop preview directly from the CLI:

```powershell
python -m cricket_vision.cli analyze --video "videos/match.mp4" --output "output/annotated.mp4" --preview window
```

Press **Q** or **Esc** in the desktop preview to stop early; frames processed so far
are saved. Desktop preview requires GUI-enabled OpenCV. The browser preview stays in
the browser and does not need a camera.

For pose and bowling-action analysis, supply a compatible pose checkpoint and a player track ID:

```powershell
python -m pip install -r requirements-ball.txt
python -m cricket_vision.cli analyze --video "videos/match.mp4" --output "output/pose_analysis.mp4" --bowler-track-id 3
```

The default config uses the locally downloaded `models/yolo11n-pose.pt` checkpoint. The track ID refers to the current lightweight tracker and may change across cuts or identity switches. The pose checkpoint provides generic human keypoints only; it is not a cricket-specific action model. The temporal states (`run-up`, `delivery_approaching`, `release_candidate`, `follow_through`) are untrained experimental heuristics, not a validated bowling-action classifier. A `release_candidate` is not a confirmed release; no time-to-release or ball-direction prediction is claimed. Direction remains `unknown`.

Player boxes use the person class from this pose checkpoint and require multiple
visible body landmarks, including a shoulder and hip, to filter detections that only
cover a hand, leg, or other object. Detection uses a lower confidence threshold and
1280 inference size to help find distant players in high-resolution footage. A box is
shown only after a track is detected in two consecutive frames, and is hidden as soon
as it is missed to prevent stale boxes. Visible people are labeled `Person 1`,
`Person 2`, etc. in display order for each frame. The old motion/brightness fallback could draw boxes around scoreboard
graphics and other scene changes; it has been removed. If the checkpoint or pose
landmarks are unavailable, player boxes are disabled rather than guessed. Ball
detection continues independently.

### Temporal model training and evaluation

Create labels for reviewed pose feature frames and group every frame from a delivery under one `delivery_id`; details and JSONL format are in [datasets/README.md](datasets/README.md). The split is by delivery group, not adjacent frames:

```powershell
python -m cricket_vision.cli prepare-action-data --frames "output/pose_analysis.jsonl" --labels "datasets/action_labels.jsonl" --output "datasets/action_sequences.jsonl"
python -m cricket_vision.cli validate-action-data --data "datasets/action_sequences.jsonl"
python -m cricket_vision.cli train-action --data "datasets/action_sequences.jsonl" --output "models/bowling_action.pt" --epochs 40
python -m cricket_vision.cli evaluate-action --data "datasets/action_test_sequences.jsonl" --model "models/bowling_action.pt"
```

Training produces frame accuracy, macro-F1, per-class metrics, confusion matrix, pre-release anticipation rate, and mean lead time on held-out delivery groups. These are measurements of the supplied labels/data, not proof of performance on other matches. During analysis, provide `--action-model "models/bowling_action.pt"` along with `--pose-model` and `--bowler-track-id` to display the trained prediction separately from the experimental heuristic.

For timestamp-based manual labeling (works with the default headless OpenCV package), review the clip in a video player and enter release timestamps in seconds:

```powershell
python -m cricket_vision.cli label-release --video "videos/match.mp4" --output "datasets/release_labels.jsonl" --timestamps "4.12,9.48,15.03" --bowler-track-id 3
```

Alternatively, on a GUI-capable OpenCV install, omit `--timestamps` to step through the clip: press Space to label the displayed frame, N (or another key) to continue, or Q/Esc to finish. Each label is JSONL containing frame index, timestamp, the manual `release` label, and optional bowler track ID. Timestamp-to-frame conversion uses nominal FPS, so verify frame indices for variable-frame-rate recordings. Group labels by complete delivery when creating train/validation/test splits to avoid adjacent-frame leakage.

The default config uses the locally downloaded `models/cricket-ball-best.pt` checkpoint
for ball analysis. To use a different checkpoint, pass it explicitly:

```powershell
python -m cricket_vision.cli analyze --video "videos/match.mp4" --output "output/ball_analysis.mp4" --ball-model "models/cricket_ball.pt"
```

The downloaded ball checkpoint is a candidate, not an independently benchmarked best-in-class model; evaluate it on held-out footage before relying on it. The detector runs on every frame, and the tracker associates detections with the existing track using recent motion. Tracks reset at scene cuts. No future ball direction or path is predicted or drawn. Confidence and shape limits are configurable in `config/default.yaml`. The dataset and model sources, terms, and local paths are documented in [datasets/README.md](datasets/README.md) and [models/README.md](models/README.md). Without an available checkpoint, person analysis proceeds and the output reports ball detection as disabled. The annotated video has a same-stem `.jsonl` sidecar with per-frame timestamps, scene IDs, ball detections, tracking status, and the observed ball-track points.

Each JSONL row is one decoded frame. Key fields include `frame_index`, `timestamp_seconds`, `scene_id`, `visible_player_count_estimate`, `ball_detector_status`, `ball_detections` (each has `bbox`, `confidence`, and `scene_id`), `ball_tracking` (observation/lost state, missed-frame count, history length, confidence), `ball_track_points_2d` (observed historical ball centers in image coordinates), `bowler_track_id`, `pose_detector_status`, `pose_keypoints`, and `bowling_action` (experimental state/confidence, unvalidated release-candidate timestamp, and temporal features). `bbox` is `[x1, y1, x2, y2]` in source-frame pixels.

At completion the CLI prints run-level processing FPS and end-to-end seconds plus per-stage call counts, total/mean/maximum latency for video decode, player/ball/pose inference, tracking, visualization, encoding, and sidecar output. This is a measurement for that run, not a performance guarantee.

## C++ trajectory benchmark

Build the extension first, then compare the tiny trajectory operation with the Python fallback:

```powershell
.\scripts\build_cpp.ps1
python -m benchmarks.benchmark_trajectory --repetitions 1000 --samples 20
```

This microbenchmark does not establish whole-pipeline acceleration; Python/C++ boundary overhead may outweigh the arithmetic savings. The pipeline reports whether it is using the native module or fallback.

## Train a ball detector

Read [datasets/README.md](datasets/README.md) for YOLO annotation, train/validation split, training, and evaluation instructions. The dataset and model weights are not provided. Ultralytics and base-model weights have license restrictions; verify those licenses for your use.

```powershell
python -m pip install -r requirements-ball.txt
python -m cricket_vision.cli train-ball --data "datasets/ball/data.yaml" --base-model "yolo11n.pt" --epochs 100 --imgsz 1280
```

## Configuration

The default application settings are in `config/default.yaml`. It is loaded by the configuration helper and validated before processing starts.

## Implementation notes

- OpenCV is used for frame decoding and timestamp handling.
- The video reader preserves `frame_index` and source timestamps in seconds.
- No pretrained ball detector is silently substituted: missing optional weights or inference dependencies disable ball analysis with a visible explanation.
- Person and pose inference use `models/yolo11n-pose.pt` when the local checkpoint and optional dependencies are available; the checkpoint is excluded from version control.
- The person detector filters predictions to the model's `person` class and does not guess players from motion or brightness.

## Known limitations

- The ball detector is a candidate model and has not been benchmarked on held-out footage from the intended camera.
- Ball tracking and trajectory outputs are experimental screen-space estimates; there is no camera calibration, physical speed estimate, or validated accuracy claim.
- Pose keypoints can be extracted when a compatible model and manual bowler track are supplied, but the current action-state heuristic and release candidates have not been validated against labeled data.
- No labeled action dataset is bundled, so temporal model accuracy and pre-release lead time are not measured for this project.
- Synthetic test clips verify code paths and output creation only; they do not establish cricket-video detection quality.

## Troubleshooting

- **Ball/person/pose features show disabled:** check that the configured checkpoint exists under `models\` and install optional packages with `requirements-ball.txt` or `requirements-action.txt` as appropriate. Person boxes are deliberately disabled when the person model is unavailable; motion is not used as a fallback.
- **`torch.cuda.is_available()` is false:** the installed PyTorch package/runtime is CPU-only or CUDA is not configured for that package. Having a CUDA Toolkit installed alone does not mean the PyTorch wheel supports CUDA. Check `python -m cricket_vision.cli check-env`.
- **C++ extension import fails:** build it with `.\scripts\build_cpp.ps1` using the same Python interpreter that runs the app. A `.pyd` built for a different Python ABI (for example, `cp314` vs `cp311`) cannot be imported by the other interpreter; Python fallback is used in that case.
- **Video writer cannot initialize:** try an MP4 codec supported by the installed OpenCV/FFmpeg build and verify the output directory is writable.
- **Player counts or bowling states look wrong:** the generic person model can miss distant players or mistake people in the crowd for players; it is not cricket-specific. Action-state heuristics are also untrained. Validate on representative labeled cricket footage before drawing conclusions.
- **A prediction looks like a physical path:** trajectory points are short-horizon 2D screen coordinates. Camera pan, zoom, cuts, perspective, and bounces are not calibrated.

## PowerShell helpers

```powershell
.\scripts\setup.ps1
.\scripts\check_environment.ps1
```
