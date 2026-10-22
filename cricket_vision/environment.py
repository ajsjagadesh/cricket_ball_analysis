from __future__ import annotations

import importlib.util
import json
import platform
from typing import Any


def _safe_module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"import failed: {exc.__class__.__name__}: {exc}"
    version_attr = getattr(module, "__version__", None)
    return str(version_attr) if version_attr is not None else "version unknown"


def get_environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv": _safe_module_version("cv2"),
        "numpy": _safe_module_version("numpy"),
        "yaml": _safe_module_version("yaml"),
        "pytest": _safe_module_version("pytest"),
        "pytorch": _safe_module_version("torch"),
    }

    try:
        from cricket_vision.cpp import NATIVE_EXTENSION_AVAILABLE

        summary["cpp_extension_available"] = NATIVE_EXTENSION_AVAILABLE
    except ImportError:
        summary["cpp_extension_available"] = False

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            summary["torch_cuda_available"] = bool(torch.cuda.is_available())
            summary["torch_cuda_version"] = torch.version.cuda
            summary["torch_device_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                summary["torch_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - diagnostic path only
            summary["torch_error"] = f"{exc.__class__.__name__}: {exc}"

    return summary


def print_environment_summary() -> str:
    return json.dumps(get_environment_summary(), indent=2, sort_keys=True)
