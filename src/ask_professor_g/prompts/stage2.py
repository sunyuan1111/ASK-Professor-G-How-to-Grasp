from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def build_stage2_prompt(
    *,
    gripper: dict[str, Any],
    stage1_processed: dict[str, Any],
    output_path: str | Path,
) -> tuple[str, str]:
    system = (
        "You are Prof. G compiling a Stage 2 objective for CEM grasp optimization. "
        "Return only Python code. No markdown, no explanation, no surrounding text."
    )
    context = {
        "gripper": gripper,
        "stage1_processed": stage1_processed,
    }
    user = (
        "Implement calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float. "
        "Lower is better. The function must be deterministic and may only depend on numpy. "
        "The returned code must start with import numpy as np or def calculate_loss. "
        "It must define calculate_loss exactly once. If candidate data is sparse, implement a "
        "general loss that favors poses near the Stage 1 validated contact/search regions, encourages "
        "contact with the object point cloud, penalizes collision-like penetration, and preserves semantic intent. "
        "The objective should be decomposable into target, contact, collision/clearance, orientation, and semantic-priority terms. "
        "Primary strategies and robust functional parts must be preferred over secondary/fragile shade or decorative regions "
        "unless geometry makes the primary infeasible. "
        "Use the Stage 1 ranges as constants; do not create new grasp regions or ignore the verified 3D targets. "
        "Return a finite float for every valid pose_mat, and add large finite penalties for NaN/Inf inputs. "
        "Bake constants from Context JSON into the code; do not parse JSON at runtime.\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2)}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n", encoding="utf-8")
    return system, user
