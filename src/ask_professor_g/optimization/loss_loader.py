from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np


def load_loss_function(path: str | Path) -> Callable[[np.ndarray, np.ndarray], float]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"ask_professor_g_loss_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import loss module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _extract_calculate_loss(module, path)


def _extract_calculate_loss(module: ModuleType, path: Path) -> Callable[[np.ndarray, np.ndarray], float]:
    if not hasattr(module, "calculate_loss"):
        raise AttributeError(f"{path} must define calculate_loss(pose_mat, point_cloud)")
    func = module.calculate_loss
    if not callable(func):
        raise TypeError(f"{path}: calculate_loss must be callable")
    return func


def validate_loss_function(
    path: str | Path,
    *,
    point_cloud: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Load a generated loss and verify that it can evaluate one finite sample."""
    func = load_loss_function(path)
    sample_pose = np.eye(4, dtype=np.float64)
    sample_points = point_cloud
    if sample_points is None:
        sample_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            dtype=np.float64,
        )
    value = func(sample_pose, sample_points)
    try:
        value_float = float(value)
    except Exception as exc:
        raise TypeError(f"{Path(path)}: calculate_loss must return a numeric scalar") from exc
    if not np.isfinite(value_float):
        raise ValueError(f"{Path(path)}: calculate_loss returned a non-finite value")
    return func
