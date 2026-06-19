from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _box_vertices(center: np.ndarray, size: Iterable[float], rotation: np.ndarray | None = None) -> np.ndarray:
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0
    corners = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=float,
    )
    if rotation is not None:
        corners = corners @ rotation.T
    return corners + center[None, :]


def _write_box(handle, vertices: np.ndarray, name: str, material: str, start_index: int) -> int:
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    handle.write(f"o {name}\nusemtl {material}\n")
    for vertex in vertices:
        handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
    for face in faces:
        shifted = [idx + start_index - 1 for idx in face]
        handle.write("f " + " ".join(str(idx) for idx in shifted) + "\n")
    return start_index + len(vertices)


def _load_obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    points = np.asarray(vertices, dtype=float)
    return points.min(axis=0), points.max(axis=0)


def _copy_obj_geometry(handle, path: Path, *, object_name: str = "object_mesh") -> int:
    vertex_count = 0
    handle.write(f"o {object_name}\nusemtl object_gray\n")
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if line.startswith(("mtllib ", "usemtl ")):
                continue
            if line.startswith("o ") or line.startswith("g "):
                handle.write("# " + line)
                continue
            if line.startswith("v "):
                vertex_count += 1
            handle.write(line)
    handle.write("\n")
    return vertex_count


def _candidate_point(candidate: Iterable[float], mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    x, y = candidate
    span = np.maximum(maxs - mins, 1e-9)
    return np.array([mins[0] + float(x) * span[0], mins[1] + float(y) * span[1], float((mins[2] + maxs[2]) / 2)])


def _material_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "newmtl grasp_red",
                "Kd 0.95 0.10 0.08",
                "newmtl object_gray",
                "Kd 0.68 0.68 0.68",
                "newmtl grasp_blue",
                "Kd 0.10 0.30 0.95",
                "newmtl grasp_green",
                "Kd 0.10 0.70 0.25",
                "newmtl grasp_yellow",
                "Kd 0.95 0.70 0.10",
                "newmtl gripper",
                "Kd 0.05 0.05 0.05",
                "newmtl gripper_best",
                "Kd 0.00 0.65 0.90",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_stage0_points_obj(
    *,
    object_mesh_path: str | Path,
    stage0_path: str | Path,
    output_dir: str | Path,
    geometry_path: str | Path | None = None,
) -> Path:
    object_mesh_path = Path(object_mesh_path)
    stage0 = json.loads(Path(stage0_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_name = "stage0_grasp_points.obj"
    mtl_name = "grasp_scene.mtl"
    scene_path = output_dir / obj_name
    shutil.copy2(object_mesh_path, output_dir / "object_mesh.obj")
    _material_file(output_dir / mtl_name)

    mins, maxs = _load_obj_bounds(object_mesh_path)
    diag = float(np.linalg.norm(maxs - mins))
    marker_size = max(diag * 0.025, 0.003)
    colors = ["grasp_red", "grasp_blue", "grasp_green", "grasp_yellow"]
    next_index = 1
    with scene_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {mtl_name}\n")
        next_index = _copy_obj_geometry(handle, object_mesh_path, object_name="object_mesh") + 1
        markers = _markers_from_geometry(geometry_path) if geometry_path else []
        if not markers:
            markers = []
            for pidx, proposal in enumerate(stage0.get("proposals", [])):
                for cidx, candidate in enumerate(proposal.get("candidate_points", [])):
                    markers.append(
                        {
                            "center": _candidate_point(candidate, mins, maxs),
                            "name": f"candidate_{pidx + 1}_{cidx + 1}_{proposal.get('strategy', 'grasp')}",
                            "material": colors[pidx % len(colors)],
                        }
                    )
        for marker in markers:
            center = np.asarray(marker["center"], dtype=float)
            vertices = _box_vertices(center, [marker_size] * 3)
            next_index = _write_box(
                handle,
                vertices,
                marker["name"],
                marker["material"],
                next_index,
            )
    return scene_path


def _markers_from_geometry(geometry_path: str | Path | None) -> list[dict[str, Any]]:
    if not geometry_path or not Path(geometry_path).exists():
        return []
    data = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    markers: list[dict[str, Any]] = []
    status_material = {
        "VALID": "grasp_green",
        "TOO_WIDE": "grasp_yellow",
        "TOO_THIN": "grasp_yellow",
        "LOW_CONFIDENCE_PROJECTION": "grasp_blue",
        "INVALID_DEPTH": "grasp_red",
        "EMPTY_REGION": "grasp_red",
    }
    for item in data.get("audit_results", []):
        strategy = item.get("strategy", "grasp")
        for candidate in item.get("candidate_results", []):
            point = candidate.get("point_3d")
            if point is None:
                continue
            status = candidate.get("audit_status") or candidate.get("status") or item.get("audit_status", "UNKNOWN")
            markers.append(
                {
                    "center": point,
                    "name": f"proposal_{item.get('id', '?')}_candidate_{candidate.get('candidate_index', 0)}_{strategy}",
                    "material": status_material.get(status, "grasp_red"),
                }
            )
    return markers


def export_optimized_grasps_obj(
    *,
    object_mesh_path: str | Path,
    stage3_path: str | Path,
    output_dir: str | Path,
    top_n: int = 5,
) -> Path:
    object_mesh_path = Path(object_mesh_path)
    stage3 = json.loads(Path(stage3_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "optimized_grasps.obj"
    mtl_name = "grasp_scene.mtl"
    shutil.copy2(object_mesh_path, output_dir / "object_mesh.obj")
    _material_file(output_dir / mtl_name)

    mins, maxs = _load_obj_bounds(object_mesh_path)
    diag = float(np.linalg.norm(maxs - mins))
    jaw_length = max(diag * 0.28, 0.04)
    jaw_width = max(diag * 0.035, 0.006)
    jaw_depth = max(diag * 0.08, 0.012)
    default_opening = max(diag * 0.18, 0.035)

    next_index = 1
    with scene_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {mtl_name}\n")
        next_index = _copy_obj_geometry(handle, object_mesh_path, object_name="object_mesh") + 1
        for idx, grasp in enumerate(stage3.get("grasps", [])[:top_n]):
            state = np.asarray(grasp["result"], dtype=float)
            center = state[:3]
            rotation = _rpy_to_matrix(float(state[3]), float(state[4]), float(state[5]))
            material = "gripper_best" if idx == 0 else "gripper"
            opening = float(np.clip(state[6] * 0.11 if len(state) > 6 else default_opening, 0.012, 0.11))
            for side, sign in [("left", -1.0), ("right", 1.0)]:
                local_offset = np.array([sign * opening / 2.0, 0.0, 0.0])
                jaw_center = center + rotation @ local_offset
                vertices = _box_vertices(jaw_center, [jaw_width, jaw_length, jaw_depth], rotation)
                next_index = _write_box(
                    handle,
                    vertices,
                    f"rank_{idx + 1}_{side}_{grasp.get('type', 'grasp')}",
                    material,
                    next_index,
                )
            palm_center = center + rotation @ np.array([0.0, -jaw_length * 0.45, 0.0])
            palm_vertices = _box_vertices(palm_center, [opening + jaw_width, jaw_width, jaw_depth], rotation)
            next_index = _write_box(
                handle,
                palm_vertices,
                f"rank_{idx + 1}_palm_{grasp.get('type', 'grasp')}",
                material,
                next_index,
            )
    return scene_path
