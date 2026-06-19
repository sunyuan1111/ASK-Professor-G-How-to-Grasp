from __future__ import annotations

import numpy as np


def downsample_cloud(points: np.ndarray, max_points: int = 2048) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[indices]

