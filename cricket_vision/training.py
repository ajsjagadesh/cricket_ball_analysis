from __future__ import annotations

from pathlib import Path


def train_ball_detector(data_yaml: str | Path, model: str = "yolo11n.pt", epochs: int = 100, image_size: int = 1280) -> dict[str, str | int]:
    """Train an Ultralytics detector on a user-provided, labeled dataset."""

    data_path = Path(data_yaml)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if epochs <= 0 or image_size <= 0:
        raise ValueError("epochs and image_size must be positive.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ball training requires the optional 'ultralytics' package.") from exc

    detector = YOLO(model)
    result = detector.train(data=str(data_path), epochs=epochs, imgsz=image_size)
    return {
        "status": "completed",
        "data": str(data_path),
        "epochs": epochs,
        "image_size": image_size,
        "results": str(result.save_dir),
    }
