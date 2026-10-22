from __future__ import annotations

import json
from pathlib import Path

import pytest

from cricket_vision.bowling.action_dataset import assemble_action_dataset
from cricket_vision.bowling.temporal_training import FEATURE_NAMES, load_action_sequences


def test_assemble_action_dataset_joins_labeled_frames_to_sidecar_features(tmp_path: Path) -> None:
    sidecar = tmp_path / "analysis.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "train.jsonl"
    frame_records = [
        {
            "frame_index": index,
            "timestamp_seconds": index / 10,
            "bowling_action": {
                "features": {name: float(index) for name in FEATURE_NAMES},
            },
        }
        for index in range(3)
    ]
    label_records = [
        {"delivery_id": "delivery-a", "frame_index": 0, "label": "run-up"},
        {"delivery_id": "delivery-a", "frame_index": 2, "label": "release_candidate"},
        {"delivery_id": "delivery-b", "frame_index": 1, "label": "delivery_approaching"},
        {"delivery_id": "delivery-b", "frame_index": 2, "label": "follow_through"},
    ]
    sidecar.write_text("".join(json.dumps(record) + "\n" for record in frame_records), encoding="utf-8")
    labels.write_text("".join(json.dumps(record) + "\n" for record in label_records), encoding="utf-8")

    result = assemble_action_dataset(sidecar, labels, output)
    sequences = load_action_sequences(output)
    assert result["delivery_sequences"] == 2
    assert result["labeled_frames"] == 4
    assert sequences[0].delivery_id == "delivery-a"
    assert sequences[0].timestamps == (0.0, 0.2)
    assert sequences[0].features[1] == (2.0, 2.0, 2.0, 2.0, 2.0)


def test_assemble_action_dataset_rejects_label_without_pose_features(tmp_path: Path) -> None:
    sidecar = tmp_path / "analysis.jsonl"
    labels = tmp_path / "labels.jsonl"
    sidecar.write_text(json.dumps({"frame_index": 0, "timestamp_seconds": 0.0}) + "\n", encoding="utf-8")
    labels.write_text(
        json.dumps({"delivery_id": "delivery-a", "frame_index": 0, "label": "run-up"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no bowling_action.features"):
        assemble_action_dataset(sidecar, labels, tmp_path / "out.jsonl")
