from __future__ import annotations

import json
from pathlib import Path

import pytest

from cricket_vision.bowling.temporal_training import (
    FEATURE_NAMES,
    TemporalActionPredictor,
    evaluate_action_model,
    load_action_sequences,
    train_action_model,
)


def _write_synthetic_sequences(path: Path) -> None:
    records = []
    for delivery_index in range(4):
        timestamps = [0.0, 0.1, 0.2, 0.3]
        features = [
            [10.0, 80.0, 5.0, 4.0, 3.0],
            [40.0, 100.0, 12.0, 10.0, 6.0],
            [900.0, 150.0, 20.0, 15.0, 10.0],
            [100.0, 120.0, 10.0, 8.0, 5.0],
        ]
        labels = ["run-up", "delivery_approaching", "release_candidate", "follow_through"]
        records.append(
            {
                "delivery_id": f"synthetic-test-{delivery_index}",
                "timestamps_seconds": timestamps,
                "features": features,
                "labels": labels,
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_action_sequence_dataset_validates_schema_and_group_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "action.jsonl"
    _write_synthetic_sequences(dataset)
    sequences = load_action_sequences(dataset)
    assert len(sequences) == 4
    assert len({sequence.delivery_id for sequence in sequences}) == 4


def test_action_sequence_dataset_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "delivery_id": "delivery-1",
                "timestamps_seconds": [0.2, 0.1],
                "features": [[0.0] * 5, [0.0] * 5],
                "labels": ["run-up", "release_candidate"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_action_sequences(dataset)


def test_action_model_training_and_held_out_evaluation(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset = tmp_path / "synthetic_action.jsonl"
    checkpoint = tmp_path / "action.pt"
    _write_synthetic_sequences(dataset)

    training_result = train_action_model(dataset, checkpoint, epochs=2, hidden_size=8, seed=3)
    evaluation_result = evaluate_action_model(dataset, checkpoint)

    assert checkpoint.exists()
    assert Path(training_result["metrics_path"]).exists()
    assert training_result["training_delivery_groups"] + training_result["validation_delivery_groups"] == 4
    assert evaluation_result["delivery_groups"] == 4
    assert "mean_lead_time_seconds" in evaluation_result["metrics"]
    predictor = TemporalActionPredictor(checkpoint)
    prediction, confidence = predictor.predict(dict.fromkeys(FEATURE_NAMES, 1.0))
    assert prediction in {"unknown", "run-up", "delivery_approaching", "release_candidate", "follow_through"}
    assert 0.0 <= confidence <= 1.0
