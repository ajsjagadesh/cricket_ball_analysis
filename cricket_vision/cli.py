from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .config import load_config
from .environment import get_environment_summary
from .pipeline import run_analysis
from .bowling.release_labeler import label_release_frames, label_release_timestamps
from .bowling.temporal_training import (
    evaluate_action_model,
    load_action_sequences,
    train_action_model,
)
from .bowling.action_dataset import assemble_action_dataset
from .performance.progress import ProgressReporter
from .visualization.live_preview import DesktopPreview


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cricket Vision AI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_env = subparsers.add_parser("check-env", help="Display environment diagnostics")
    check_env.set_defaults(handler=_handle_check_env)

    build_info = subparsers.add_parser("build-info", help="Display project metadata")
    build_info.set_defaults(handler=_handle_build_info)

    analyze = subparsers.add_parser("analyze", help="Run a video analysis pass")
    analyze.add_argument("--video", required=True, help="Path to the input video")
    analyze.add_argument("--output", required=True, help="Path to the output video")
    analyze.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    analyze.add_argument("--ball-model", help="Optional trained cricket-ball detector checkpoint")
    analyze.add_argument("--pose-model", help="Optional pose checkpoint")
    analyze.add_argument("--bowler-track-id", type=int, help="Manually selected bowler player track ID")
    analyze.add_argument("--action-model", help="Optional trained temporal action-model checkpoint")
    analyze.add_argument(
        "--preview",
        choices=("none", "window"),
        default="none",
        help="Show annotated frames in a desktop window while analyzing",
    )
    analyze.set_defaults(handler=_handle_analyze)

    train_ball = subparsers.add_parser("train-ball", help="Train a custom one-class cricket-ball detector")
    train_ball.add_argument("--data", required=True, help="Path to training data YAML")
    train_ball.add_argument("--base-model", default="yolo11n.pt", help="Pretrained detector architecture/checkpoint")
    train_ball.add_argument("--epochs", type=int, default=100)
    train_ball.add_argument("--imgsz", type=int, default=1280)
    train_ball.set_defaults(handler=_handle_train_ball)

    label_release = subparsers.add_parser("label-release", help="Manually label release frames in a video")
    label_release.add_argument("--video", required=True, help="Input video to review")
    label_release.add_argument("--output", required=True, help="JSONL output for release labels")
    label_release.add_argument("--bowler-track-id", type=int, help="Optional selected bowler track ID")
    label_release.add_argument(
        "--timestamps",
        help="Comma-separated manually reviewed release timestamps in seconds (works without GUI support)",
    )
    label_release.set_defaults(handler=_handle_label_release)

    train_action = subparsers.add_parser("train-action", help="Train a temporal bowling-action classifier")
    train_action.add_argument("--data", required=True, help="JSONL labeled pose-feature sequences")
    train_action.add_argument("--output", default="models/bowling_action.pt", help="Output checkpoint path")
    train_action.add_argument("--epochs", type=int, default=40)
    train_action.add_argument("--hidden-size", type=int, default=32)
    train_action.set_defaults(handler=_handle_train_action)

    evaluate_action = subparsers.add_parser("evaluate-action", help="Evaluate an action model on labeled sequences")
    evaluate_action.add_argument("--data", required=True, help="Held-out JSONL sequences")
    evaluate_action.add_argument("--model", required=True, help="Trained temporal action-model checkpoint")
    evaluate_action.set_defaults(handler=_handle_evaluate_action)

    validate_action_data = subparsers.add_parser("validate-action-data", help="Validate labeled action sequence JSONL")
    validate_action_data.add_argument("--data", required=True, help="JSONL labeled pose-feature sequences")
    validate_action_data.set_defaults(handler=_handle_validate_action_data)

    prepare_action_data = subparsers.add_parser(
        "prepare-action-data",
        help="Join reviewed frame-state labels with pose feature sidecar data",
    )
    prepare_action_data.add_argument("--frames", required=True, help="Analysis JSONL sidecar")
    prepare_action_data.add_argument("--labels", required=True, help="JSONL delivery/frame/action labels")
    prepare_action_data.add_argument("--output", required=True, help="Output temporal training JSONL")
    prepare_action_data.set_defaults(handler=_handle_prepare_action_data)

    return parser


def _handle_check_env(_args: argparse.Namespace) -> int:
    print(json.dumps(get_environment_summary(), indent=2, sort_keys=True))
    return 0


def _handle_build_info(_args: argparse.Namespace) -> int:
    summary = {
        "name": "cricket_vision",
        "version": __version__,
        "config_path": "config/default.yaml",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.input_path = args.video
    config.output_path = args.output
    if args.ball_model:
        config.ball_model_path = args.ball_model
    if args.pose_model:
        config.pose_model_path = args.pose_model
    if args.bowler_track_id is not None:
        config.bowler_track_id = args.bowler_track_id
    if args.action_model:
        config.action_model_path = args.action_model
    config.validate()
    progress = ProgressReporter()
    print(f"Preparing video analysis: {args.video}", file=progress.stream, flush=True)
    preview = None
    try:
        if args.preview == "window":
            preview = DesktopPreview()
        result = run_analysis(
            args.video,
            args.output,
            config,
            progress_callback=progress,
            frame_callback=preview,
        )
    except Exception:
        progress.finish(succeeded=False)
        raise
    finally:
        if preview is not None:
            preview.close()
    progress.finish(succeeded=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _handle_train_action(args: argparse.Namespace) -> int:
    result = train_action_model(
        args.data,
        args.output,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _handle_evaluate_action(args: argparse.Namespace) -> int:
    result = evaluate_action_model(args.data, args.model)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _handle_validate_action_data(args: argparse.Namespace) -> int:
    sequences = load_action_sequences(args.data)
    print(
        json.dumps(
            {
                "status": "valid",
                "sequences": len(sequences),
                "delivery_groups": len({sequence.delivery_id for sequence in sequences}),
                "frames": sum(len(sequence.labels) for sequence in sequences),
            },
            indent=2,
        )
    )
    return 0


def _handle_prepare_action_data(args: argparse.Namespace) -> int:
    result = assemble_action_dataset(args.frames, args.labels, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _handle_label_release(args: argparse.Namespace) -> int:
    if args.timestamps:
        timestamps = [float(value.strip()) for value in args.timestamps.split(",") if value.strip()]
        count = label_release_timestamps(args.video, args.output, timestamps, args.bowler_track_id)
    else:
        count = label_release_frames(
            video_path=args.video,
            output_path=args.output,
            bowler_track_id=args.bowler_track_id,
        )
    print(json.dumps({"status": "completed", "labels_written": count, "output": args.output}, indent=2))
    return 0


def _handle_train_ball(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset config not found: {data_path}")
    from .training import train_ball_detector

    result = train_ball_detector(
        data_path,
        model=args.base_model,
        epochs=args.epochs,
        image_size=args.imgsz,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.error("No command supplied.")
    try:
        return args.handler(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
