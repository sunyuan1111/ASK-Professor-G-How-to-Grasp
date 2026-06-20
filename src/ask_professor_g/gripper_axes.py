from __future__ import annotations

from typing import Any

import numpy as np


_AXIS_LABELS = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=float),
    "X": np.array([1.0, 0.0, 0.0], dtype=float),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
}


def normalize_vector(value: Any, fallback: Any) -> np.ndarray:
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


def axis_label_to_vector(value: Any, fallback: Any = "+Z") -> np.ndarray:
    if isinstance(value, str):
        key = value.strip().upper().replace(" ", "")
        if key in _AXIS_LABELS:
            return _AXIS_LABELS[key].copy()
    if isinstance(fallback, str):
        fallback_key = fallback.strip().upper().replace(" ", "")
        fallback_vector = _AXIS_LABELS.get(fallback_key, _AXIS_LABELS["+Z"])
    else:
        fallback_vector = np.asarray(fallback, dtype=float)
    return normalize_vector(value, fallback_vector)


def project_to_plane(vector: Any, normal: Any, fallback: Any = "+X") -> np.ndarray:
    normal_vec = normalize_vector(normal, [0.0, 0.0, 1.0])
    vector_vec = normalize_vector(vector, fallback)
    projected = vector_vec - float(np.dot(vector_vec, normal_vec)) * normal_vec
    if float(np.linalg.norm(projected)) < 1e-7:
        ref = axis_label_to_vector(fallback, "+X")
        if abs(float(np.dot(ref, normal_vec))) > 0.92:
            ref = np.array([0.0, 1.0, 0.0]) if abs(float(normal_vec[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        projected = ref - float(np.dot(ref, normal_vec)) * normal_vec
    return projected / max(float(np.linalg.norm(projected)), 1e-8)


def local_gripper_axes(gripper: dict[str, Any] | None) -> dict[str, np.ndarray | str]:
    gripper = gripper or {}
    offset_key = "tcp_offset"
    offset = gripper.get("tcp_offset")
    hand_type = str(gripper.get("hand_type", "")).lower()
    if hand_type in {"dexterous", "underactuated"} and gripper.get("tcp_wrap") is not None:
        offset_key = "tcp_wrap"
        offset = gripper.get("tcp_wrap")

    try:
        tcp_axis = np.asarray(offset, dtype=float)
    except (TypeError, ValueError):
        tcp_axis = np.zeros(3, dtype=float)
    if tcp_axis.shape != (3,) or not np.all(np.isfinite(tcp_axis)) or float(np.linalg.norm(tcp_axis)) < 1e-8:
        offset_key = "palm_normal"
        tcp_axis = axis_label_to_vector(gripper.get("palm_normal", "+Z"), "+Z")
    tcp_axis = normalize_vector(tcp_axis, [0.0, 0.0, 1.0])

    closing_axis = axis_label_to_vector(gripper.get("closing_direction", "+Y"), "+Y")
    closing_axis = project_to_plane(closing_axis, tcp_axis, fallback=gripper.get("width_axis", "+X"))

    width_axis = np.cross(closing_axis, tcp_axis)
    if float(np.linalg.norm(width_axis)) < 1e-7:
        width_axis = project_to_plane(gripper.get("width_axis", "+X"), tcp_axis, fallback="+X")
    width_axis = width_axis / max(float(np.linalg.norm(width_axis)), 1e-8)
    closing_axis = np.cross(tcp_axis, width_axis)
    closing_axis = closing_axis / max(float(np.linalg.norm(closing_axis)), 1e-8)

    local_frame = np.column_stack([width_axis, closing_axis, tcp_axis])
    if float(np.linalg.det(local_frame)) < 0.0:
        width_axis = -width_axis
        local_frame = np.column_stack([width_axis, closing_axis, tcp_axis])

    return {
        "local_width_axis": width_axis,
        "local_closing_axis": closing_axis,
        "local_tcp_axis": tcp_axis,
        "local_frame": local_frame,
        "tcp_axis_source": offset_key,
        "closing_axis_source": str(gripper.get("closing_direction", "+Y")),
    }


def serializable_gripper_axes(gripper: dict[str, Any] | None) -> dict[str, Any]:
    axes = local_gripper_axes(gripper)
    return {
        "local_width_axis": np.asarray(axes["local_width_axis"], dtype=float).round(6).tolist(),
        "local_closing_axis": np.asarray(axes["local_closing_axis"], dtype=float).round(6).tolist(),
        "local_tcp_axis": np.asarray(axes["local_tcp_axis"], dtype=float).round(6).tolist(),
        "tcp_axis_source": axes["tcp_axis_source"],
        "closing_axis_source": axes["closing_axis_source"],
        "semantics": {
            "local_tcp_axis": "base-to-TCP/contact direction; after placement it should point toward the object surface",
            "local_closing_axis": "jaw pinch/compression axis; after placement it should align with Stage 0 closing_direction",
            "local_width_axis": "orthogonal axis completing a right-handed local frame",
        },
    }


def rotation_from_normal_and_closing(
    normal: Any,
    closing: Any,
    gripper: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normal_vec = normalize_vector(normal, [0.0, 0.0, 1.0])
    target_tcp_axis = -normal_vec
    target_closing_axis = project_to_plane(closing, target_tcp_axis, fallback="+X")
    target_width_axis = np.cross(target_closing_axis, target_tcp_axis)
    target_width_axis = target_width_axis / max(float(np.linalg.norm(target_width_axis)), 1e-8)
    target_closing_axis = np.cross(target_tcp_axis, target_width_axis)
    target_closing_axis = target_closing_axis / max(float(np.linalg.norm(target_closing_axis)), 1e-8)

    axes = local_gripper_axes(gripper)
    local_frame = np.asarray(axes["local_frame"], dtype=float)
    target_frame = np.column_stack([target_width_axis, target_closing_axis, target_tcp_axis])
    rotation = target_frame @ np.linalg.inv(local_frame)
    if float(np.linalg.det(rotation)) < 0.0:
        target_width_axis = -target_width_axis
        target_frame = np.column_stack([target_width_axis, target_closing_axis, target_tcp_axis])
        rotation = target_frame @ np.linalg.inv(local_frame)

    world_closing = rotation @ np.asarray(axes["local_closing_axis"], dtype=float)
    world_tcp = rotation @ np.asarray(axes["local_tcp_axis"], dtype=float)
    debug = {
        "local_width_axis": np.asarray(axes["local_width_axis"], dtype=float).tolist(),
        "local_closing_axis": np.asarray(axes["local_closing_axis"], dtype=float).tolist(),
        "local_tcp_axis": np.asarray(axes["local_tcp_axis"], dtype=float).tolist(),
        "target_width_axis": target_width_axis.tolist(),
        "target_closing_axis": target_closing_axis.tolist(),
        "target_tcp_axis": target_tcp_axis.tolist(),
        "closing_alignment_dot": float(np.dot(world_closing, target_closing_axis)),
        "tcp_axis_alignment_dot": float(np.dot(world_tcp, target_tcp_axis)),
        "tcp_axis_source": axes["tcp_axis_source"],
        "closing_axis_source": axes["closing_axis_source"],
    }
    return rotation, debug
