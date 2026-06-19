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

