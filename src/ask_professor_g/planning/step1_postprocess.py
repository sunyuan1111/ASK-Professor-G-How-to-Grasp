from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..gripper_axes import rotation_from_normal_and_closing, serializable_gripper_axes
from ..visualization.real_gripper import calibrated_s0_for_opening_width


def postprocess_step1(
    *,
    input_path: str | Path,
    output_path: str | Path,
    gripper_name: str,
    gripper_max_opening_width: float = 0.11,
    geometry_path: str | Path | None = None,
    gripper: dict[str, Any] | None = None,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> dict[str, Any]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    grasps = data.get("grasps", data if isinstance(data, list) else [])
    geometry = None
    if geometry_path and Path(geometry_path).exists():
        geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    if geometry:
        grasps = _align_grasps_to_geometry(
            grasps,
            geometry,
            gripper_max_opening_width=gripper_max_opening_width,
            gripper=gripper,
            gripper_urdf_path=gripper_urdf_path,
            gripper_glpca_path=gripper_glpca_path,
        )
    if (not grasps) and geometry:
        grasps = _fallback_grasps_from_geometry(
            geometry,
            gripper_max_opening_width=gripper_max_opening_width,
            gripper=gripper,
            gripper_urdf_path=gripper_urdf_path,
            gripper_glpca_path=gripper_glpca_path,
        )
    processed = {
        "object_perception": data.get("object_perception", {}),
        "gripper": gripper_name,
        "geometry_source": str(geometry_path) if geometry_path else None,
        "gripper_axes": serializable_gripper_axes(gripper),
        "postprocess_info": {
            "position_semantics": "wrist_pose_relative.pos_xyz_m stores TCP/contact-center search ranges; the real gripper base is recovered downstream using tcp_offset.",
            "anti_penetration": "Search centers are shifted outward along the measured surface normal and constrained to a narrow range around that clearance point.",
            "orientation_semantics": "RPY is recomputed from measured_normal and Stage 0 closing_direction using the current gripper local TCP and closing axes.",
        },
        "grasps": [],
    }
    for idx, grasp in enumerate(grasps):
        item = dict(grasp)
        item.setdefault("type", f"grasp_{idx}")
        item.setdefault("category", "candidate")
        item.setdefault("grasp_mode", "pinch")
        item.setdefault("synergy_config", {"s0": [0.0, 0.1], "s1": [0.0, 0.0], "s2": [0.0, 0.0]})
        item.setdefault(
            "wrist_pose_relative",
            {
                "pos_xyz_m": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
                "orn_rpy_deg": {"r": [0.0, 0.0], "p": [0.0, 0.0], "y": [0.0, 0.0]},
            },
        )
        processed["grasps"].append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    return processed


def _align_grasps_to_geometry(
    grasps: list[dict[str, Any]],
    geometry: dict[str, Any],
    *,
    gripper_max_opening_width: float,
    gripper: dict[str, Any] | None = None,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    strategies = geometry.get("selected_strategies", [])
    if not grasps or not strategies:
        return grasps

    aligned: list[dict[str, Any]] = []
    used_strategy_names: set[str] = set()
    for idx, grasp in enumerate(grasps):
        strategy = _match_strategy(grasp, strategies, fallback_index=idx)
        if not strategy or not strategy.get("adjusted_3d_point"):
            aligned.append(grasp)
            continue

        item = _grasp_from_strategy(
            strategy,
            template=grasp,
            variation_index=0,
            gripper_max_opening_width=gripper_max_opening_width,
            gripper=gripper,
            gripper_urdf_path=gripper_urdf_path,
            gripper_glpca_path=gripper_glpca_path,
        )
        used_strategy_names.add(str(strategy.get("strategy")))
        aligned.append(item)

    for strategy in strategies:
        name = str(strategy.get("strategy"))
        if name in used_strategy_names:
            continue
        fallback = _fallback_grasps_from_geometry(
            {"selected_strategies": [strategy]},
            gripper_max_opening_width=gripper_max_opening_width,
            gripper=gripper,
            gripper_urdf_path=gripper_urdf_path,
            gripper_glpca_path=gripper_glpca_path,
        )
        if fallback:
            aligned.append(fallback[0])
    return aligned


def _grasp_from_strategy(
    strategy: dict[str, Any],
    *,
    template: dict[str, Any] | None,
    variation_index: int,
    gripper_max_opening_width: float,
    gripper: dict[str, Any] | None = None,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> dict[str, Any]:
    item = dict(template or {})
    point = np.asarray(strategy["adjusted_3d_point"][:3], dtype=float)
    width = float(strategy.get("measured_width") or 0.025)
    normal = _correct_normal_direction(strategy.get("measured_normal"), strategy.get("approach_direction"), strategy.get("strategy"))
    closing = _project_closing_direction(strategy.get("closing_direction"), normal)
    clearance = _surface_clearance(width=width, category=str(item.get("category", "")), strategy=str(strategy.get("strategy", "")))
    clearance_point = point + normal * clearance
    is_primary = strategy.get("display_id") == 1 or str(strategy.get("priority", "")).lower() == "high"
    category = str(item.get("category", "")).lower()
    item["category"] = category if category in {"primary", "secondary"} else ("primary" if is_primary else "secondary")
    item.setdefault("type", f"{strategy.get('strategy', 'validated')}_{variation_index + 1}")
    item.setdefault("grasp_mode", "pinch")
    item["source_stage0_strategy"] = strategy.get("strategy")
    item["source_3d_point"] = point.tolist()
    item["source_clearance_point"] = clearance_point.tolist()
    item["source_measured_width"] = width
    item["target_part"] = strategy.get("target_part")
    item["measured_normal"] = strategy.get("measured_normal")
    item["measured_normal_used"] = normal.tolist()
    item["closing_direction"] = strategy.get("closing_direction")
    item["closing_direction_used"] = closing.tolist()
    item["approach_direction"] = strategy.get("approach_direction")
    item["surface_clearance_m"] = float(clearance)
    item["postprocess_notes"] = (
        "Search box is centered on an outward clearance point. "
        "source_3d_point remains the measured surface anchor from RGB-D probing."
    )

    half_range = 0.0045 if item["category"] == "primary" else 0.006
    # Let later variations explore slightly more without including a large chunk of object interior.
    half_range += min(0.0015 * variation_index, 0.003)
    item["wrist_pose_relative"] = {
        "pos_xyz_m": _range_dict(clearance_point, half_range),
        "orn_rpy_deg": _orientation_ranges_from_normal_and_closing(normal, closing, gripper=gripper),
    }
    _, axis_debug = rotation_from_normal_and_closing(normal, closing, gripper)
    item["orientation_source"] = "measured_normal_closing_direction_and_gripper_axes"
    item["gripper_axis_alignment"] = axis_debug

    max_opening = max(float(gripper_max_opening_width), 1e-6)
    # Keep the pre-grasp mesh visibly open around the measured width.
    configured_margin = None
    if gripper:
        margin_value = gripper.get("pregrasp_opening_margin_m", gripper.get("opening_safety_margin_m"))
        if margin_value is not None:
            try:
                configured_margin = float(margin_value)
            except (TypeError, ValueError):
                configured_margin = None
    base_margin = max(0.012, min(0.020, 0.40 * width))
    safety_margin = max(base_margin, configured_margin or 0.0)
    desired_opening = min(max_opening, max(width + safety_margin, 0.018))
    s0_center, calibrated_opening, calibration = calibrated_s0_for_opening_width(
        desired_opening=desired_opening,
        gripper=gripper or {"max_opening_width": max_opening},
        urdf_path=gripper_urdf_path,
        glpca_path=gripper_glpca_path,
    )
    s0_half = 0.04 if item["category"] == "primary" else 0.055
    item["source_desired_opening_m"] = float(desired_opening)
    item["source_calibrated_opening_m"] = float(calibrated_opening)
    item["source_calibrated_s0"] = float(s0_center)
    item["opening_calibration"] = calibration
    item["opening_safety_margin_m"] = float(calibrated_opening - width)
    item["synergy_config"] = {
        "s0": [max(0.0, s0_center - s0_half), min(1.0, s0_center + s0_half)],
        "s1": item.get("synergy_config", {}).get("s1", [0.0, 0.0]),
        "s2": item.get("synergy_config", {}).get("s2", [0.0, 0.0]),
    }
    return item


def _match_strategy(grasp: dict[str, Any], strategies: list[dict[str, Any]], fallback_index: int) -> dict[str, Any] | None:
    text = " ".join(str(grasp.get(key, "")) for key in ["type", "source_stage0_strategy", "target_part"]).lower()
    for strategy in strategies:
        name = str(strategy.get("strategy", "")).lower()
        target = str(strategy.get("target_part", "")).lower()
        if (name and name in text) or (target and target in text):
            return strategy
    if fallback_index < len(strategies):
        return strategies[fallback_index]
    return strategies[0] if strategies else None


def _fallback_grasps_from_geometry(
    geometry: dict[str, Any],
    *,
    gripper_max_opening_width: float = 0.11,
    gripper: dict[str, Any] | None = None,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    grasps: list[dict[str, Any]] = []
    for strategy in geometry.get("selected_strategies", [])[:2]:
        if not strategy.get("adjusted_3d_point"):
            continue
        for variation in range(2):
            template = {
                "type": f"{strategy.get('strategy', 'validated')}_{variation + 1}",
                "category": "primary" if strategy.get("display_id") == 1 else "secondary",
                "grasp_mode": "pinch",
                "synergy_config": {"s1": [0.0, 0.0], "s2": [0.0, 0.0]},
            }
            grasps.append(
                _grasp_from_strategy(
                    strategy,
                    template=template,
                    variation_index=variation,
                    gripper_max_opening_width=gripper_max_opening_width,
                    gripper=gripper,
                    gripper_urdf_path=gripper_urdf_path,
                    gripper_glpca_path=gripper_glpca_path,
                )
            )
    return grasps


def _range_dict(center: np.ndarray, half_range: float) -> dict[str, list[float]]:
    center = np.asarray(center, dtype=float)
    return {
        "x": [float(center[0] - half_range), float(center[0] + half_range)],
        "y": [float(center[1] - half_range), float(center[1] + half_range)],
        "z": [float(center[2] - half_range), float(center[2] + half_range)],
    }


def _normalize_vector(value: Any, fallback: list[float]) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        vector = np.asarray(fallback, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        vector = np.asarray(fallback, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        vector = np.asarray(fallback, dtype=float)
        norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _approach_outward_normal(approach: Any) -> np.ndarray | None:
    key = str(approach or "").lower().replace("_", "")
    mapping = {
        "topdown": np.array([0.0, 0.0, 1.0]),
        "sidex": np.array([1.0, 0.0, 0.0]),
        "sidenegx": np.array([-1.0, 0.0, 0.0]),
        "sidey": np.array([0.0, 1.0, 0.0]),
        "sidenegy": np.array([0.0, -1.0, 0.0]),
    }
    return mapping.get(key)


def _correct_normal_direction(normal: Any, approach: Any, strategy: Any) -> np.ndarray:
    measured = _normalize_vector(normal, [0.0, 0.0, 1.0])
    expected = _approach_outward_normal(approach)
    strategy_text = str(strategy or "").lower()
    if expected is None and "top" in strategy_text:
        expected = np.array([0.0, 0.0, 1.0])
    if expected is None:
        return measured
    if float(np.dot(measured, expected)) < -0.35:
        measured = -measured
    elif abs(float(np.dot(measured, expected))) < 0.25:
        # The local PCA normal can be unstable near edges/handles; use the semantic approach side.
        measured = expected
    return measured / max(float(np.linalg.norm(measured)), 1e-8)


def _project_closing_direction(closing: Any, normal: np.ndarray) -> np.ndarray:
    closing_vec = _normalize_vector(closing, [1.0, 0.0, 0.0])
    projected = closing_vec - float(np.dot(closing_vec, normal)) * normal
    if float(np.linalg.norm(projected)) < 1e-7:
        ref = np.array([0.0, 1.0, 0.0]) if abs(float(normal[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        projected = ref - float(np.dot(ref, normal)) * normal
    return projected / max(float(np.linalg.norm(projected)), 1e-8)


def _surface_clearance(*, width: float, category: str, strategy: str) -> float:
    text = f"{category} {strategy}".lower()
    base = 0.005
    if any(term in text for term in ["handle", "root", "rim", "secondary"]):
        base = 0.0065
    if width > 0.04:
        base += 0.0015
    return float(np.clip(base, 0.004, 0.009))


def _orientation_ranges_from_normal_and_closing(
    normal: np.ndarray,
    closing: np.ndarray,
    *,
    gripper: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    rotation, _ = rotation_from_normal_and_closing(normal, closing, gripper)
    rpy = _matrix_to_rpy_deg(rotation)
    half_width = np.array([12.0, 12.0, 16.0], dtype=float)
    return {
        "r": [float(rpy[0] - half_width[0]), float(rpy[0] + half_width[0])],
        "p": [float(rpy[1] - half_width[1]), float(rpy[1] + half_width[1])],
        "y": [float(rpy[2] - half_width[2]), float(rpy[2] + half_width[2])],
    }


def _matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    # Inverse of R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching optimization.cem.state_to_pose_matrix.
    sy = -float(rotation[2, 0])
    pitch = math.asin(float(np.clip(sy, -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))
