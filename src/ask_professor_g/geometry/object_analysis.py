from __future__ import annotations

from pathlib import Path

import numpy as np

from .probing import load_point_cloud


def analyze_object_geometry(point_cloud_path: str | Path) -> dict:
    points = load_point_cloud(point_cloud_path)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "num_points": int(len(points)),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": (maxs - mins).tolist(),
    }

