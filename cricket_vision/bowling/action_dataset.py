from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .temporal_training import ACTION_LABELS, FEATURE_NAMES


def assemble_action_dataset(
    frame_sidecar: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Join manually reviewed frame labels with pose features from analysis JSONL."""

    sidecar_file = Path(frame_sidecar)
    labels_file = Path(labels_path)
    if not sidecar_file.is_file():
        raise FileNotFoundError(f"Analysis sidecar not found: {sidecar_file}")
    if not labels_file.is_file():
        raise FileNotFoundError(f"Action labels not found: {labels_file}")

    frames: dict[int, dict[str, Any]] = {}
    with sidecar_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = record.get("frame_index")
            if not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(f"Sidecar line {line_number}: frame_index must be a non-negative integer.")
            if frame_index in frames:
                raise ValueError(f"Duplicate frame_index {frame_index} in analysis sidecar.")
            frames[frame_index] = record

    grouped_labels: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with labels_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                label_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid action label JSON on line {line_number}: {exc}") from exc
            delivery_id = label_record.get("delivery_id")
            frame_index = label_record.get("frame_index")
            label = label_record.get("label")
            if not isinstance(delivery_id, str) or not delivery_id.strip():
                raise ValueError(f"Action label line {line_number}: delivery_id must be non-empty.")
            if not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(f"Action label line {line_number}: frame_index must be non-negative.")
            if label not in ACTION_LABELS:
                raise ValueError(f"Action label line {line_number}: unsupported label {label!r}.")
            if frame_index not in frames:
                raise ValueError(f"Action label line {line_number}: frame {frame_index} is absent from sidecar.")
            grouped_labels[delivery_id].append((frame_index, label))

    sequences: list[dict[str, Any]] = []
    for delivery_id, entries in sorted(grouped_labels.items()):
        entries.sort(key=lambda item: item[0])
        timestamps: list[float] = []
        features: list[list[float]] = []
        labels: list[str] = []
        last_frame_index: int | None = None
        for frame_index, label in entries:
            if frame_index == last_frame_index:
                raise ValueError(f"Delivery {delivery_id} has duplicate labels for frame {frame_index}.")
            last_frame_index = frame_index
            record = frames[frame_index]
            action_features = record.get("bowling_action", {}).get("features")
            if not isinstance(action_features, dict):
                raise ValueError(f"Frame {frame_index} has no bowling_action.features in the sidecar.")
            vector: list[float] = []
            for feature_name in FEATURE_NAMES:
                value = action_features.get(feature_name)
                vector.append(0.0 if value is None else float(value))
            if any(not (-float("inf") < value < float("inf")) for value in vector):
                raise ValueError(f"Frame {frame_index} contains non-finite action features.")
            timestamps.append(float(record["timestamp_seconds"]))
            features.append(vector)
            labels.append(label)
        if len(labels) < 2:
            raise ValueError(f"Delivery {delivery_id} needs at least two labeled frames.")
        sequences.append(
            {
                "delivery_id": delivery_id,
                "timestamps_seconds": timestamps,
                "features": features,
                "labels": labels,
            }
        )

    if not sequences:
        raise ValueError("No labeled delivery sequences found.")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        for sequence in sequences:
            output.write(json.dumps(sequence, separators=(",", ":")) + "\n")
    return {
        "output": str(output_file),
        "delivery_sequences": len(sequences),
        "labeled_frames": sum(len(sequence["labels"]) for sequence in sequences),
    }
