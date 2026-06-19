from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def normalized_to_pixel(
    normalized_coords: list[float] | tuple[float, float],
    image_width: int,
    image_height: int,
    origin: str = "top_left",
) -> tuple[int, int]:
    x_norm, y_norm = normalized_coords
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    col = int(round(x_norm * (image_width - 1)))
    if origin == "top_left":
        row = int(round(y_norm * (image_height - 1)))
    else:
        row = int(round((1.0 - y_norm) * (image_height - 1)))
    return row, col


def pixel_to_normalized(row: int, col: int, image_width: int, image_height: int) -> list[float]:
    return [col / max(image_width - 1, 1), row / max(image_height - 1, 1)]


def load_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        points = np.load(path)
    elif ext in {".xyz", ".txt"}:
        points = np.loadtxt(path)
    elif ext in {".obj", ".ply", ".stl", ".off"}:
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError("Install trimesh to load mesh point clouds.") from exc
        mesh = trimesh.load(path, force="mesh")
        points = np.asarray(mesh.vertices)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape (N, 3+), got {points.shape}")
    return points[:, :3].astype(np.float32)


def estimate_local_geometry(
    points: np.ndarray,
    center: np.ndarray,
    radius: float = 0.02,
    closing_direction: list[float] | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    distances = np.linalg.norm(points - center[None, :], axis=1)
    radii = [radius, radius * 2.0, radius * 3.0, radius * 5.0]
    local = np.empty((0, 3), dtype=float)
    used_radius = radius
    for candidate_radius in radii:
        local = points[distances <= candidate_radius]
        used_radius = candidate_radius
        if len(local) >= 10:
            break
    if len(local) < 3:
        local = points[np.argsort(distances)[: min(len(points), 64)]]
        used_radius = float(distances[np.argsort(distances)[min(len(local) - 1, 63)]]) if len(local) else radius
    if len(local) < 3:
        return {"status": False, "reason": "not enough local points", "num_points": int(len(local))}

    local_center = local.mean(axis=0)
    centered = local - local_center
    _, values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    position_vector = center - points.mean(axis=0)
    if np.dot(normal, position_vector) < 0:
        normal = -normal
    axis = vh[0]

    secondary = vh[1] if vh.shape[0] > 1 else vh[0]
    pca_span = float(np.ptp(centered @ secondary))
    aabb_span = float(np.max(local.max(axis=0) - local.min(axis=0)))
    width = pca_span
    closing_width = None
    if closing_direction is not None:
        closing = np.asarray(closing_direction, dtype=float)
        norm = float(np.linalg.norm(closing))
        if norm > 1e-8:
            closing = closing / norm
            closing_width = float(np.ptp(centered @ closing))
            width = closing_width

    if width <= 1e-8:
        width = min(aabb_span, float(np.linalg.norm(local.max(axis=0) - local.min(axis=0))))

    geometry_type = "mixed"
    if values[0] > 1e-12 and len(values) >= 3:
        ratio_21 = float(values[1] / values[0])
        ratio_32 = float(values[2] / values[1]) if values[1] > 1e-12 else 0.0
        if ratio_21 < 0.15:
            geometry_type = "linear"
        elif ratio_32 < 0.15:
            geometry_type = "planar"
        elif ratio_21 > 0.5 and ratio_32 > 0.5:
            geometry_type = "volumetric"
    return {
        "status": True,
        "width": width,
        "closing_width": closing_width,
        "pca_width": pca_span,
        "aabb_width": aabb_span,
        "normal": normal.tolist(),
        "axis": axis.tolist(),
        "num_points": int(len(local)),
        "eigenvalues": values.tolist(),
        "geometry_type": geometry_type,
        "radius": float(used_radius),
    }


def _project_points_from_camera(points: np.ndarray, camera: dict[str, Any], image_width: int, image_height: int) -> np.ndarray:
    if _is_perspective_camera(camera):
        world_to_camera = np.asarray(camera.get("extrinsic") or np.linalg.inv(np.asarray(camera["camera_pose"])), dtype=float)
        intrinsic = camera.get("intrinsic", {})
        fx = float(intrinsic.get("fx", image_width))
        fy = float(intrinsic.get("fy", image_height))
        cx = float(intrinsic.get("cx", image_width / 2.0))
        cy = float(intrinsic.get("cy", image_height / 2.0))
        points_h = np.concatenate([points, np.ones((len(points), 1), dtype=float)], axis=1)
        cam = (world_to_camera @ points_h.T).T[:, :3]
        z = -cam[:, 2]
        z = np.maximum(z, 1e-9)
        u = fx * cam[:, 0] / z + cx
        v = cy - fy * cam[:, 1] / z
        return np.stack([u, v], axis=1)
    if camera.get("projection_type") == "orthographic_xz":
        mins = np.asarray(camera["bounds_min"], dtype=float)
        maxs = np.asarray(camera["bounds_max"], dtype=float)
        margin = int(camera.get("margin_px", min(image_width, image_height) * 0.12))
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        margin = int(min(image_width, image_height) * 0.12)
    span = np.maximum(maxs - mins, 1e-9)
    u = margin + (points[:, 0] - mins[0]) / span[0] * (image_width - 2 * margin)
    v = image_height - margin - (points[:, 2] - mins[2]) / span[2] * (image_height - 2 * margin)
    return np.stack([u, v], axis=1)


def _nearest_3d_from_pixel(points: np.ndarray, projected: np.ndarray, row: int, col: int) -> tuple[np.ndarray, float]:
    target = np.array([col, row], dtype=float)
    distances = np.linalg.norm(projected - target[None, :], axis=1)
    idx = int(np.argmin(distances))
    return points[idx], float(distances[idx])


def _is_perspective_camera(camera: dict[str, Any]) -> bool:
    projection_type = camera.get("projection_type")
    if projection_type == "orthographic_xz":
        return False
    if projection_type:
        return projection_type == "pyrender_perspective"
    return "camera_pose" in camera and "extrinsic" in camera


def _valid_depth(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _nearest_depth(depth_map: np.ndarray, row: int, col: int, search_radius: int = 8) -> tuple[float | None, int | None, int | None, float | None]:
    height, width = depth_map.shape
    if not (0 <= row < height and 0 <= col < width):
        return None, None, None, None
    value = float(depth_map[row, col])
    if _valid_depth(value):
        return value, row, col, 0.0

    best: tuple[float, int, int, float] | None = None
    for radius in range(1, search_radius + 1):
        r0 = max(0, row - radius)
        r1 = min(height - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(width - 1, col + radius)
        window = depth_map[r0 : r1 + 1, c0 : c1 + 1]
        valid = np.argwhere(np.isfinite(window) & (window > 0))
        if len(valid) == 0:
            continue
        offsets = valid + np.array([r0, c0])
        distances = np.linalg.norm(offsets - np.array([row, col]), axis=1)
        nearest = int(np.argmin(distances))
        nr, nc = int(offsets[nearest, 0]), int(offsets[nearest, 1])
        best = (float(depth_map[nr, nc]), nr, nc, float(distances[nearest]))
        break
    if best is None:
        return None, None, None, None
    return best


def deproject_pixel_pyrender(col: int, row: int, depth: float, camera: dict[str, Any]) -> np.ndarray:
    intrinsic = camera["intrinsic"]
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    camera_pose = np.asarray(camera["camera_pose"], dtype=float)
    x_cam = (col - cx) * depth / fx
    y_cam = -(row - cy) * depth / fy
    z_cam = -depth
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=float)
    return (camera_pose @ point_cam)[:3]


def build_pointcloud_from_depth_pyrender(depth_map: np.ndarray, camera: dict[str, Any], stride: int = 1) -> np.ndarray:
    height, width = depth_map.shape
    intrinsic = camera["intrinsic"]
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    camera_pose = np.asarray(camera["camera_pose"], dtype=float)

    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth_map[::stride, ::stride]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    cols_valid = cols[valid].astype(float)
    rows_valid = rows[valid].astype(float)
    depth_valid = sampled_depth[valid].astype(float)
    x_cam = (cols_valid - cx) * depth_valid / fx
    y_cam = -(rows_valid - cy) * depth_valid / fy
    z_cam = -depth_valid
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
    return (camera_pose @ points_cam.T).T[:, :3].astype(np.float32)


def _point_from_depth_pixel(
    *,
    row: int,
    col: int,
    depth_map: np.ndarray | None,
    camera: dict[str, Any],
    fallback_points: np.ndarray,
    fallback_projected: np.ndarray,
) -> tuple[np.ndarray | None, float, str, dict[str, Any]]:
    if depth_map is not None and _is_perspective_camera(camera):
        depth, depth_row, depth_col, pixel_error = _nearest_depth(depth_map, row, col)
        if depth is not None and depth_row is not None and depth_col is not None:
            return (
                deproject_pixel_pyrender(depth_col, depth_row, depth, camera),
                float(pixel_error or 0.0),
                "pyrender_depth",
                {"depth": float(depth), "depth_pixel": {"row": int(depth_row), "col": int(depth_col)}},
            )
        return None, float("inf"), "pyrender_depth", {"depth": None, "depth_pixel": None}
    center, pixel_error = _nearest_3d_from_pixel(fallback_points, fallback_projected, row, col)
    return center, pixel_error, "nearest_projected_point_cloud", {"depth": None, "depth_pixel": None}


def _audit_status(geometry: dict[str, Any], gripper_limits: dict[str, float], pixel_error: float) -> tuple[str, str]:
    if not geometry.get("status"):
        return "EMPTY_REGION", geometry.get("reason", "local geometry unavailable")
    width = float(geometry.get("width") or 0.0)
    max_width = float(gripper_limits.get("max_width", gripper_limits.get("max_opening_width", 0.08)))
    min_width = float(gripper_limits.get("min_width", gripper_limits.get("min_opening_width", 0.0)))
    if pixel_error > 80:
        return "LOW_CONFIDENCE_PROJECTION", f"nearest visible point is {pixel_error:.1f}px away"
    if width > max_width:
        return "TOO_WIDE", f"local width {width:.4f}m > gripper max {max_width:.4f}m"
    if width < min_width:
        return "TOO_THIN", f"local width {width:.4f}m < gripper min {min_width:.4f}m"
    return "VALID", "passes projection and local width checks"


def simple_geometry_probe(
    stage0: dict[str, Any],
    point_cloud_path: str | Path,
    *,
    output_path: str | Path,
    camera_path: str | Path | None = None,
    depth_path: str | Path | None = None,
    gripper_limits: dict[str, float] | None = None,
    image_width: int = 800,
    image_height: int = 800,
) -> dict[str, Any]:
    model_points = load_point_cloud(point_cloud_path)
    camera: dict[str, Any] = {}
    if camera_path and Path(camera_path).exists():
        camera = json.loads(Path(camera_path).read_text(encoding="utf-8"))
        image_size = camera.get("image_size", [image_width, image_height])
        image_width = int(camera.get("image_width", image_size[0]))
        image_height = int(camera.get("image_height", image_size[1]))
    depth_map: np.ndarray | None = None
    if depth_path and Path(depth_path).exists():
        depth_map = np.load(depth_path)
        image_height, image_width = int(depth_map.shape[0]), int(depth_map.shape[1])

    visible_points = model_points
    if depth_map is not None and _is_perspective_camera(camera):
        stride = max(1, int(max(depth_map.shape) // 400))
        depth_points = build_pointcloud_from_depth_pyrender(depth_map, camera, stride=stride)
        if len(depth_points) >= 16:
            visible_points = depth_points

    projected = _project_points_from_camera(model_points, camera, image_width, image_height)
    gripper_limits = gripper_limits or {}
    proposals = stage0.get("proposals", [])
    audit_results: list[dict[str, Any]] = []

    for proposal in proposals:
        proposal_result: dict[str, Any] | None = None
        candidate_results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(proposal.get("candidate_points", [])):
            row, col = normalized_to_pixel(candidate, image_width, image_height)
            center, pixel_error, projection_source, projection_meta = _point_from_depth_pixel(
                row=row,
                col=col,
                depth_map=depth_map,
                camera=camera,
                fallback_points=model_points,
                fallback_projected=projected,
            )
            if center is None:
                candidate_result = {
                    "candidate_index": idx,
                    "candidate_2d_normalized": candidate,
                    "pixel": {"row": row, "col": col},
                    "point_3d": None,
                    "pixel_error": pixel_error,
                    "projection_source": projection_source,
                    "audit_status": "INVALID_DEPTH",
                    "reason": "no valid depth found near candidate pixel",
                    "geometry": {"status": False, "reason": "no valid depth"},
                    **projection_meta,
                }
                candidate_results.append(candidate_result)
                continue
            geometry = estimate_local_geometry(
                visible_points,
                center,
                closing_direction=proposal.get("closing_direction"),
            )
            status, reason = _audit_status(geometry, gripper_limits, pixel_error)
            candidate_result = {
                "candidate_index": idx,
                "candidate_2d_normalized": candidate,
                "pixel": {"row": row, "col": col},
                "point_3d": center.tolist(),
                "pixel_error": pixel_error,
                "projection_source": projection_source,
                "audit_status": status,
                "reason": reason,
                "geometry": geometry,
                **projection_meta,
            }
            candidate_results.append(candidate_result)
            if status == "VALID" and proposal_result is None:
                proposal_result = candidate_result

        if proposal_result is None and candidate_results:
            proposal_result = candidate_results[0]

        if proposal_result is None:
            audit_results.append(
                {
                    "id": proposal.get("id"),
                    "strategy": proposal.get("strategy", f"proposal_{proposal.get('id', 0)}"),
                    "priority": proposal.get("priority", "Low"),
                    "audit_status": "INVALID_INPUT",
                    "reason": "no candidate_points provided",
                    "candidate_results": [],
                }
            )
            continue

        geom = proposal_result.get("geometry", {})
        audit_results.append(
            {
                "id": proposal.get("id"),
                "strategy": proposal.get("strategy", f"proposal_{proposal.get('id', 0)}"),
                "target_part": proposal.get("target_part", ""),
                "priority": proposal.get("priority", "Medium"),
                "reasoning": proposal.get("reasoning", ""),
                "approach_direction": proposal.get("approach_direction", "Unknown"),
                "closing_direction": proposal.get("closing_direction", [1, 0, 0]),
                "audit_status": proposal_result["audit_status"],
                "reason": proposal_result["reason"],
                "selected_candidate_index": proposal_result["candidate_index"],
                "selected_2d_normalized": proposal_result["candidate_2d_normalized"],
                "selected_pixel": proposal_result["pixel"],
                "adjusted_3d_point": proposal_result["point_3d"],
                "projection_source": proposal_result.get("projection_source"),
                "depth": proposal_result.get("depth"),
                "depth_pixel": proposal_result.get("depth_pixel"),
                "measured_width": geom.get("width"),
                "measured_normal": geom.get("normal"),
                "measured_axis": geom.get("axis"),
                "num_local_points": geom.get("num_points"),
                "geometry_type": geom.get("geometry_type", "local_pca"),
                "geometry_radius": geom.get("radius"),
                "width_method": "closing_direction" if geom.get("closing_width") is not None else "local_pca",
                "candidate_results": candidate_results,
            }
        )

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    valid = [item for item in audit_results if item["audit_status"] == "VALID"]
    valid.sort(key=lambda item: priority_order.get(item.get("priority", "Low"), 3))
    selected = []
    for idx, item in enumerate(valid[:5], start=1):
        copied = dict(item)
        copied["display_id"] = idx
        selected.append(copied)

    payload = {
        "source": "pyrender_depth_geometry_probe" if depth_map is not None and _is_perspective_camera(camera) else "orthographic_rgbd_geometry_probe",
        "input_stage0_json": stage0.get("object_analysis", ""),
        "camera_file": str(camera_path) if camera_path else None,
        "depth_file": str(depth_path) if depth_path else None,
        "rendering_engine": "PyRender" if depth_map is not None and _is_perspective_camera(camera) else "FallbackOrthographic",
        "pointcloud_source": "rendered_depth" if visible_points is not model_points else str(point_cloud_path),
        "gripper_limits": gripper_limits,
        "audit_results": audit_results,
        "selected_strategies": selected,
        "summary": {
            "total_proposals": len(audit_results),
            "valid_grasps": len(valid),
            "selected_count": len(selected),
            "rejected": sum(1 for item in audit_results if item["audit_status"] != "VALID"),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
