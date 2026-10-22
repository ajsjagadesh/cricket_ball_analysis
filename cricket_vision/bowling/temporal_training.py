from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTION_LABELS = (
    "unknown",
    "run-up",
    "delivery_approaching",
    "release_candidate",
    "follow_through",
)
FEATURE_NAMES = (
    "wrist_speed_px_s",
    "elbow_angle_degrees",
    "body_center_speed_px_s",
    "shoulder_motion_px_s",
    "hip_motion_px_s",
)


@dataclass(frozen=True)
class ActionSequence:
    delivery_id: str
    timestamps: tuple[float, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]


def load_action_sequences(path: str | Path) -> list[ActionSequence]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Action dataset not found: {source}")
    sequences: list[ActionSequence] = []
    with source.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            delivery_id = record.get("delivery_id")
            timestamps = record.get("timestamps_seconds")
            features = record.get("features")
            labels = record.get("labels")
            if not isinstance(delivery_id, str) or not delivery_id.strip():
                raise ValueError(f"Line {line_number}: delivery_id must be a non-empty string.")
            if not isinstance(timestamps, list) or not isinstance(features, list) or not isinstance(labels, list):
                raise ValueError(f"Line {line_number}: timestamps_seconds, features, and labels must be lists.")
            if not timestamps or len(timestamps) != len(features) or len(features) != len(labels):
                raise ValueError(f"Line {line_number}: timestamps, features, and labels need equal non-zero lengths.")

            parsed_timestamps: list[float] = []
            parsed_features: list[tuple[float, ...]] = []
            parsed_labels: list[str] = []
            for index, (timestamp, vector, label) in enumerate(zip(timestamps, features, labels)):
                if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    raise ValueError(f"Line {line_number}: timestamp {index} must be finite.")
                if index and timestamp <= parsed_timestamps[-1]:
                    raise ValueError(f"Line {line_number}: timestamps must be strictly increasing.")
                if not isinstance(vector, list) or len(vector) != len(FEATURE_NAMES):
                    raise ValueError(
                        f"Line {line_number}: each feature vector must contain {len(FEATURE_NAMES)} values."
                    )
                parsed_vector = tuple(float(value) for value in vector)
                if any(not math.isfinite(value) for value in parsed_vector):
                    raise ValueError(f"Line {line_number}: feature values must be finite.")
                if label not in ACTION_LABELS:
                    raise ValueError(f"Line {line_number}: unsupported action label {label!r}.")
                parsed_timestamps.append(float(timestamp))
                parsed_features.append(parsed_vector)
                parsed_labels.append(label)
            sequences.append(
                ActionSequence(
                    delivery_id=delivery_id,
                    timestamps=tuple(parsed_timestamps),
                    features=tuple(parsed_features),
                    labels=tuple(parsed_labels),
                )
            )
    if not sequences:
        raise ValueError(f"Action dataset contains no sequence records: {source}")
    return sequences


def _new_model(torch: Any, input_size: int, hidden_size: int, class_count: int) -> Any:
    class TemporalActionNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.classifier = torch.nn.Linear(hidden_size, class_count)

        def forward(self, values: Any) -> Any:
            encoded, _ = self.gru(values)
            return self.classifier(encoded)

    return TemporalActionNetwork()


def _grouped_split(sequences: list[ActionSequence], seed: int) -> tuple[list[ActionSequence], list[ActionSequence]]:
    delivery_ids = sorted({sequence.delivery_id for sequence in sequences})
    if len(delivery_ids) < 2:
        raise ValueError("At least two distinct delivery_id groups are required for held-out evaluation.")
    random.Random(seed).shuffle(delivery_ids)
    validation_count = max(1, round(len(delivery_ids) * 0.2))
    validation_ids = set(delivery_ids[:validation_count])
    training = [sequence for sequence in sequences if sequence.delivery_id not in validation_ids]
    validation = [sequence for sequence in sequences if sequence.delivery_id in validation_ids]
    return training, validation


def _class_metrics(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = list(ACTION_LABELS)
    confusion = {actual: {guess: 0 for guess in labels} for actual in labels}
    per_class: dict[str, dict[str, float]] = {}
    for actual, guess in zip(expected, predicted):
        confusion[actual][guess] += 1
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in labels if actual != label)
        false_negative = sum(confusion[label][guess] for guess in labels if guess != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    accuracy = sum(actual == guess for actual, guess in zip(expected, predicted)) / len(expected) if expected else 0.0
    return {
        "frame_accuracy": accuracy,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(per_class),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _anticipation_metrics(
    sequences: list[ActionSequence],
    predicted_sequences: list[list[str]],
) -> dict[str, Any]:
    lead_times: list[float] = []
    deliveries_with_release = 0
    anticipated_deliveries = 0
    for sequence, predicted in zip(sequences, predicted_sequences):
        release_times = [
            timestamp
            for timestamp, label in zip(sequence.timestamps, sequence.labels)
            if label == "release_candidate"
        ]
        if not release_times:
            continue
        deliveries_with_release += 1
        release_time = release_times[0]
        prior_anticipations = [
            timestamp
            for timestamp, label in zip(sequence.timestamps, predicted)
            if label == "delivery_approaching" and timestamp < release_time
        ]
        if prior_anticipations:
            lead_times.append(release_time - min(prior_anticipations))
            anticipated_deliveries += 1
    return {
        "deliveries_with_release_labels": deliveries_with_release,
        "deliveries_anticipated_before_release": anticipated_deliveries,
        "pre_release_anticipation_rate": (
            anticipated_deliveries / deliveries_with_release if deliveries_with_release else None
        ),
        "mean_lead_time_seconds": sum(lead_times) / len(lead_times) if lead_times else None,
        "lead_time_observations": len(lead_times),
    }


def _predict_sequences(
    model: Any,
    torch: Any,
    sequences: list[ActionSequence],
    feature_mean: Any,
    feature_scale: Any,
) -> list[list[str]]:
    results: list[list[str]] = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            values = torch.tensor(sequence.features, dtype=torch.float32)
            values = ((values - feature_mean) / feature_scale).unsqueeze(0)
            logits = model(values)[0]
            indices = logits.argmax(dim=-1).tolist()
            results.append([ACTION_LABELS[index] for index in indices])
    return results


class TemporalActionPredictor:
    """Loads a trained sequence model and predicts action state from pose features."""

    def __init__(self, checkpoint_path: str | Path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Temporal action inference requires PyTorch.") from exc
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Action-model checkpoint not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES or tuple(checkpoint.get("labels", ())) != ACTION_LABELS:
            raise ValueError("Action-model checkpoint feature or label schema does not match this application.")
        self.torch = torch
        self.model = _new_model(torch, len(FEATURE_NAMES), int(checkpoint["hidden_size"]), len(ACTION_LABELS))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        if "feature_mean" not in checkpoint or "feature_scale" not in checkpoint:
            raise ValueError("Action-model checkpoint is missing feature normalization values.")
        self.feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
        self.feature_scale = torch.tensor(checkpoint["feature_scale"], dtype=torch.float32)
        self._features: list[tuple[float, ...]] = []
        self._max_history = 256

    def reset(self) -> None:
        self._features.clear()

    def predict(self, features: dict[str, float | None]) -> tuple[str, float]:
        values: list[float] = []
        for name in FEATURE_NAMES:
            value = features.get(name)
            numeric = float(value) if value is not None else 0.0
            if not math.isfinite(numeric):
                numeric = 0.0
            values.append(numeric)
        self._features.append(tuple(values))
        self._features = self._features[-self._max_history :]
        with self.torch.no_grad():
            sequence = self.torch.tensor(self._features, dtype=self.torch.float32)
            sequence = ((sequence - self.feature_mean) / self.feature_scale).unsqueeze(0)
            probabilities = self.torch.softmax(self.model(sequence)[0, -1], dim=0)
            class_index = int(probabilities.argmax().item())
            confidence = float(probabilities[class_index].item())
        return ACTION_LABELS[class_index], confidence


def train_action_model(
    data_path: str | Path,
    output_path: str | Path,
    epochs: int = 40,
    hidden_size: int = 32,
    seed: int = 17,
) -> dict[str, Any]:
    if epochs <= 0 or hidden_size <= 0:
        raise ValueError("epochs and hidden_size must be positive.")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Temporal action training requires PyTorch.") from exc

    sequences = load_action_sequences(data_path)
    training, validation = _grouped_split(sequences, seed)
    torch.manual_seed(seed)
    random_generator = random.Random(seed)
    label_to_index = {label: index for index, label in enumerate(ACTION_LABELS)}
    model = _new_model(torch, len(FEATURE_NAMES), hidden_size, len(ACTION_LABELS))
    training_features = torch.tensor(
        [vector for sequence in training for vector in sequence.features],
        dtype=torch.float32,
    )
    feature_mean = training_features.mean(dim=0)
    feature_scale = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        order = list(range(len(training)))
        random_generator.shuffle(order)
        for sequence_index in order:
            sequence = training[sequence_index]
            values = torch.tensor(sequence.features, dtype=torch.float32)
            values = ((values - feature_mean) / feature_scale).unsqueeze(0)
            targets = torch.tensor([label_to_index[label] for label in sequence.labels], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(values)[0]
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()

    predicted_sequences = _predict_sequences(model, torch, validation, feature_mean, feature_scale)
    expected = [label for sequence in validation for label in sequence.labels]
    predicted = [label for sequence in predicted_sequences for label in sequence]
    metrics = _class_metrics(expected, predicted)
    metrics.update(_anticipation_metrics(validation, predicted_sequences))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "labels": ACTION_LABELS,
            "hidden_size": hidden_size,
            "seed": seed,
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
        },
        target,
    )
    metrics_path = target.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "completed",
        "model_path": str(target),
        "metrics_path": str(metrics_path),
        "training_delivery_groups": len({item.delivery_id for item in training}),
        "validation_delivery_groups": len({item.delivery_id for item in validation}),
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "epochs": epochs,
        "metrics": metrics,
        "claim": "Unvalidated until reviewed on a representative, held-out cricket dataset.",
    }


def evaluate_action_model(data_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Temporal action evaluation requires PyTorch.") from exc

    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Action-model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES or tuple(checkpoint.get("labels", ())) != ACTION_LABELS:
        raise ValueError("Action-model checkpoint feature or label schema does not match this application.")
    if "feature_mean" not in checkpoint or "feature_scale" not in checkpoint:
        raise ValueError("Action-model checkpoint is missing feature normalization values.")
    sequences = load_action_sequences(data_path)
    model = _new_model(torch, len(FEATURE_NAMES), int(checkpoint["hidden_size"]), len(ACTION_LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    feature_scale = torch.tensor(checkpoint["feature_scale"], dtype=torch.float32)
    predicted_sequences = _predict_sequences(model, torch, sequences, feature_mean, feature_scale)
    expected = [label for sequence in sequences for label in sequence.labels]
    predicted = [label for sequence in predicted_sequences for label in sequence]
    metrics = _class_metrics(expected, predicted)
    metrics.update(_anticipation_metrics(sequences, predicted_sequences))
    return {
        "sequences": len(sequences),
        "delivery_groups": len({sequence.delivery_id for sequence in sequences}),
        "metrics": metrics,
        "claim": "Evaluation reflects only the supplied dataset and is not a generalization guarantee.",
    }
