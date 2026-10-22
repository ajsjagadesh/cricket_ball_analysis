from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cricket_vision.config import AppConfig, load_config, write_default_config


def test_default_config_loads() -> None:
    config = load_config("config/default.yaml")
    assert config.device == "auto"
    assert config.player_confidence_threshold == 0.30
    assert config.ball_confidence_threshold == 0.30
    assert config.ball_max_speed_px_per_second == 3000.0
    assert config.ball_confirmation_hits == 3
    assert config.pose_inference_resolution == 1280
    assert config.trajectory_history_length == 20


def test_invalid_config_values_raise() -> None:
    path = Path("config/invalid_test.yaml")
    path.write_text(yaml.safe_dump({"trajectory_history_length": 0, "device": "bad"}), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_config(path)
    finally:
        path.unlink(missing_ok=True)


def test_invalid_bowler_and_pose_settings_raise() -> None:
    with pytest.raises(ValueError, match="bowler_track_id"):
        AppConfig(bowler_track_id=0).validate()
    with pytest.raises(ValueError, match="pose_inference_resolution"):
        AppConfig(pose_inference_resolution=0).validate()


def test_write_default_config_creates_file(tmp_path: Path) -> None:
    file_path = tmp_path / "custom.yaml"
    written = write_default_config(file_path)
    assert written.exists()
    config = load_config(file_path)
    assert isinstance(config, AppConfig)
