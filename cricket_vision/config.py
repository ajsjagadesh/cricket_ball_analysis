from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    """Application configuration for the cricket vision pipeline."""

    input_path: str = "videos/sample.mp4"
    output_path: str = "output/annotated.mp4"
    model_dir: str = "models"
    device: str = "auto"
    input_resolution: tuple[int, int] | None = None
    ball_model_path: str | None = None
    player_confidence_threshold: float = 0.30
    ball_confidence_threshold: float = 0.30
    ball_inference_resolution: int = 1280
    ball_max_bbox_area_fraction: float = 0.01
    ball_max_bbox_dimension_fraction: float = 0.10
    ball_max_aspect_ratio: float = 4.0
    ball_max_speed_px_per_second: float = 3000.0
    ball_confirmation_hits: int = 3
    pose_model_path: str | None = None
    bowler_track_id: int | None = None
    pose_inference_resolution: int = 1280
    action_window_seconds: float = 1.0
    release_wrist_speed_threshold: float = 700.0
    runup_body_speed_threshold: float = 15.0
    action_model_path: str | None = None
    pose_confidence_threshold: float = 0.4
    scene_cut_threshold: float = 0.25
    trajectory_history_length: int = 20
    prediction_horizon: int = 15
    max_missed_frames: int = 8
    logging_level: str = "INFO"
    use_cpp_extension: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.input_path:
            raise ValueError("input_path must not be empty.")
        if not self.output_path:
            raise ValueError("output_path must not be empty.")
        if self.trajectory_history_length <= 0:
            raise ValueError("trajectory_history_length must be positive.")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative.")
        if self.ball_inference_resolution <= 0:
            raise ValueError("ball_inference_resolution must be positive.")
        if not 0.0 < self.ball_max_bbox_area_fraction <= 1.0:
            raise ValueError("ball_max_bbox_area_fraction must be in (0, 1].")
        if not 0.0 < self.ball_max_bbox_dimension_fraction <= 1.0:
            raise ValueError("ball_max_bbox_dimension_fraction must be in (0, 1].")
        if self.ball_max_aspect_ratio < 1.0:
            raise ValueError("ball_max_aspect_ratio must be at least 1.")
        if self.ball_max_speed_px_per_second <= 0:
            raise ValueError("ball_max_speed_px_per_second must be positive.")
        if self.ball_confirmation_hits <= 0:
            raise ValueError("ball_confirmation_hits must be positive.")
        if self.pose_inference_resolution <= 0:
            raise ValueError("pose_inference_resolution must be positive.")
        if self.action_window_seconds <= 0:
            raise ValueError("action_window_seconds must be positive.")
        if self.release_wrist_speed_threshold <= 0:
            raise ValueError("release_wrist_speed_threshold must be positive.")
        if self.runup_body_speed_threshold <= 0:
            raise ValueError("runup_body_speed_threshold must be positive.")
        if self.bowler_track_id is not None and self.bowler_track_id <= 0:
            raise ValueError("bowler_track_id must be a positive track ID.")
        if self.player_confidence_threshold < 0.0 or self.player_confidence_threshold > 1.0:
            raise ValueError("player_confidence_threshold must be in [0, 1].")
        if self.ball_confidence_threshold < 0.0 or self.ball_confidence_threshold > 1.0:
            raise ValueError("ball_confidence_threshold must be in [0, 1].")
        if self.scene_cut_threshold < 0.0 or self.scene_cut_threshold > 1.0:
            raise ValueError("scene_cut_threshold must be in [0, 1].")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    config_path = Path(path) if path else Path("config/default.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration at {config_path} must be a mapping.")

    if overrides:
        raw.update(overrides)

    config = AppConfig(**raw)
    config.validate()
    return config


def write_default_config(path: str | Path = "config/default.yaml") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    default = {
        "input_path": "videos/sample.mp4",
        "output_path": "output/annotated.mp4",
        "model_dir": "models",
        "device": "auto",
        "input_resolution": None,
        "ball_model_path": "models/cricket-ball-best.pt",
        "player_confidence_threshold": 0.30,
        "ball_confidence_threshold": 0.30,
        "ball_inference_resolution": 1280,
        "ball_max_bbox_area_fraction": 0.01,
        "ball_max_bbox_dimension_fraction": 0.1,
        "ball_max_aspect_ratio": 4.0,
        "ball_max_speed_px_per_second": 3000.0,
        "ball_confirmation_hits": 3,
        "pose_model_path": "models/yolo11n-pose.pt",
        "bowler_track_id": None,
        "pose_inference_resolution": 1280,
        "action_window_seconds": 1.0,
        "release_wrist_speed_threshold": 700.0,
        "runup_body_speed_threshold": 15.0,
        "action_model_path": None,
        "pose_confidence_threshold": 0.4,
        "scene_cut_threshold": 0.25,
        "trajectory_history_length": 20,
        "prediction_horizon": 15,
        "max_missed_frames": 8,
        "logging_level": "INFO",
        "use_cpp_extension": False,
    }
    target.write_text(yaml.safe_dump(default, sort_keys=False), encoding="utf-8")
    return target
