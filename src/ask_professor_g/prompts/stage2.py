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
        "pose_mat[:3, 3] is the optimized TCP/contact/search center, not the gripper base. "
        "The loss receives only this 4x4 pose and cannot observe normalized synergy values such as s0. "
        "The returned code must start with import numpy as np or def calculate_loss. "
        "It must define calculate_loss exactly once. If candidate data is sparse, implement a "
        "general loss that favors poses near the Stage 1 validated contact/search regions, encourages "
        "contact with the object point cloud, penalizes collision-like penetration, and preserves semantic intent. "
        "The objective should be decomposable into target, surface-clearance, inward-penetration, orientation, "
        "and semantic-priority terms. Primary strategies and robust functional parts must be preferred over "
        "secondary/fragile shade or decorative regions unless geometry makes the primary infeasible. "
        "Use the Stage 1 ranges as constants; do not create new grasp regions or ignore the verified 3D targets. "
        "If a grasp contains source_clearance_point, optimize toward that point rather than the raw source_3d_point. "
        "If a grasp contains measured_normal_used or measured_normal, penalize dot(position - source_3d_point, normal) < surface_clearance_m. "
        "Prefer a small positive clearance from the point cloud, roughly 0.003-0.010 m; do not minimize distance by sliding the TCP into the object. "
        "Strongly penalize candidates that move behind the local surface normal, candidates with nearest object distance below 0.0015 m, "
        "and candidates far from the selected validated strategy. Use stage1_processed.gripper_axes as authoritative: local_tcp_axis should point from the gripper base toward the surface, "
        "i.e. opposite the outward surface normal, and local_closing_axis should align with each grasp's closing_direction_used across the narrow object dimension. "
        "Opening/synergy feasibility has already been calibrated from Stage 1 measured_width, safety margin, and the gripper URDF/gLPCA; do not try to optimize s0 inside this loss. "
        "Return a finite float for every valid pose_mat, and add large finite penalties for NaN/Inf inputs. "
        "Bake constants from Context JSON into the code; do not parse JSON at runtime.\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2)}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n", encoding="utf-8")
    return system, user


def build_stage2_repair_prompt(
    *,
    original_user_prompt: str,
    invalid_code: str,
    validation_error: str,
    output_path: str | Path,
) -> tuple[str, str]:
    system = (
        "You are Prof. G repairing a Stage 2 objective for CEM grasp optimization. "
        "Return only corrected Python code. No markdown, no explanation, no surrounding text."
    )
    user = (
        "The previous Stage 2 Python loss failed validation. Rewrite the whole file so it is self-contained "
        "and passes the runtime smoke test. Keep the same public interface exactly:\n"
        "calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float\n\n"
        "Hard requirements:\n"
        "- Only depend on numpy.\n"
        "- Define every constant used by calculate_loss in the returned file.\n"
        "- Do not reference undefined names such as _STAGE1_TARGETS unless you define them first.\n"
        "- Do not load files, parse JSON, call APIs, print, or use randomness at runtime.\n"
        "- Return a finite numeric float for a valid 4x4 pose matrix and an Nx3 point cloud.\n"
        "- Return a large finite penalty for malformed, NaN, or Inf inputs.\n"
        "- Preserve the original Stage 1 semantic intent, validated 3D targets, Stage 1-calibrated gripper opening assumptions, "
        "contact encouragement, positive surface clearance, inward-penetration penalty, and orientation preference.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Invalid code:\n```python\n{invalid_code}\n```\n\n"
        f"Original Stage 2 task and context:\n{original_user_prompt}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n", encoding="utf-8")
    return system, user
