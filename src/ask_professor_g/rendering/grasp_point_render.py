from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_grasp_points(
    *,
    mesh_path: str | Path,
    geometry_path: str | Path,
    camera_path: str | Path,
    output_path: str | Path,
    show_normals: bool = True,
) -> Path:
    """Render the object mesh with 3D grasp points and local normals.

    This mirrors the old `render_grasp_points.py` behavior, but is safe for the public
    pipeline: PyRender is optional and a deterministic 2D evidence fallback is emitted when
    OpenGL is unavailable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_with_pyrender(
            mesh_path=Path(mesh_path),
            geometry_path=Path(geometry_path),
            camera_path=Path(camera_path),
            output_path=output_path,
            show_normals=show_normals,
        )
    except Exception as exc:
        _render_fallback_from_pixels(
            geometry_path=Path(geometry_path),
            camera_path=Path(camera_path),
            output_path=output_path,
            reason=str(exc),
        )
    return output_path


def _load_geometry(geometry_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(geometry_path.read_text(encoding="utf-8"))
    if data.get("selected_strategies"):
        return data["selected_strategies"], data
    return data.get("audit_results", []), data


def _sphere_mesh(trimesh, center: np.ndarray, radius: float, color: list[int]):
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = color
    return sphere


def _arrow_mesh(trimesh, start: np.ndarray, direction: np.ndarray, length: float, radius: float, color: list[int]):
    direction = direction.astype(float)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    cylinder = trimesh.creation.cylinder(radius=radius, height=length * 0.7)
    cylinder.apply_translation([0.0, 0.0, length * 0.35])
    cone = trimesh.creation.cone(radius=radius * 2.2, height=length * 0.3)
    cone.apply_translation([0.0, 0.0, length * 0.85])
    arrow = trimesh.util.concatenate([cylinder, cone])
    arrow.visual.vertex_colors = color

    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(direction, z_axis):
        rotation = np.eye(4)
    elif np.allclose(direction, -z_axis):
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        cross = np.cross(z_axis, direction)
        dot = float(np.dot(z_axis, direction))
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        rotation = trimesh.transformations.rotation_matrix(angle, cross)
    arrow.apply_transform(rotation)
    arrow.apply_translation(start)
    return arrow


def _axis_meshes(trimesh, length: float, radius: float):
    specs = [
        ([1, 0, 0], [0, 1, 0], np.pi / 2, [255, 0, 0, 255]),
        ([0, 1, 0], [1, 0, 0], -np.pi / 2, [0, 190, 0, 255]),
        ([0, 0, 1], None, 0.0, [0, 0, 255, 255]),
    ]
    meshes = []
    for direction, rot_axis, angle, color in specs:
        cyl = trimesh.creation.cylinder(radius=radius, height=length)
        if rot_axis is not None:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(angle, rot_axis))
        cyl.apply_translation(np.asarray(direction, dtype=float) * length / 2.0)
        cyl.visual.vertex_colors = color
        meshes.append(cyl)
    return meshes


def _render_with_pyrender(
    *,
    mesh_path: Path,
    geometry_path: Path,
    camera_path: Path,
    output_path: Path,
    show_normals: bool,
) -> None:
    if os.name != "nt":
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import pyrender
    import trimesh

    strategies, _ = _load_geometry(geometry_path)
    camera_params = json.loads(camera_path.read_text(encoding="utf-8"))
    image_width = int(camera_params.get("image_width", camera_params.get("image_size", [800, 800])[0]))
    image_height = int(camera_params.get("image_height", camera_params.get("image_size", [800, 800])[1]))

    object_mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))
    object_mesh.visual.vertex_colors = [180, 180, 180, 255]
    bounds = np.asarray(object_mesh.bounds, dtype=float)
    diag = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-4)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.34, 0.34, 0.34])
    scene.add(pyrender.Mesh.from_trimesh(object_mesh, smooth=True))
    for mesh in _axis_meshes(trimesh, length=diag * 0.35, radius=diag * 0.008):
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    status_colors = {
        "VALID": [0, 205, 80, 255],
        "TOO_WIDE": [255, 200, 0, 255],
        "TOO_THIN": [255, 130, 0, 255],
        "LOW_CONFIDENCE_PROJECTION": [160, 80, 220, 255],
        "INVALID_DEPTH": [255, 0, 0, 255],
        "EMPTY_REGION": [120, 120, 120, 255],
        "ALL_CANDIDATES_FAILED": [120, 0, 0, 255],
    }
    for item in strategies:
        point_3d = item.get("adjusted_3d_point")
        if point_3d is None:
            continue
        point = np.asarray(point_3d, dtype=float)
        color = status_colors.get(item.get("audit_status", "VALID"), [255, 255, 255, 255])
        radius = diag * (0.032 if item.get("audit_status") == "VALID" else 0.022)
        scene.add(pyrender.Mesh.from_trimesh(_sphere_mesh(trimesh, point, radius=radius, color=color)))

        normal = item.get("measured_normal")
        if show_normals and normal is not None:
            arrow = _arrow_mesh(
                trimesh,
                point,
                np.asarray(normal, dtype=float),
                length=diag * 0.18,
                radius=diag * 0.006,
                color=[30, 70, 230, 255],
            )
            scene.add(pyrender.Mesh.from_trimesh(arrow, smooth=False))

    intrinsic = camera_params.get("intrinsic", {})
    fy = float(intrinsic.get("fy", image_height))
    yfov = float(2.0 * np.arctan(image_height / (2.0 * fy)))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=image_width / image_height)
    if "camera_pose" in camera_params:
        camera_pose = np.asarray(camera_params["camera_pose"], dtype=float)
    else:
        camera_pose = np.linalg.inv(np.asarray(camera_params["extrinsic"], dtype=float))
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))
    pose2 = np.eye(4)
    pose2[:3, 3] = [0.4, -0.2, 0.8]
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=pose2)

    renderer = pyrender.OffscreenRenderer(image_width, image_height)
    color, _ = renderer.render(scene)
    renderer.delete()
    image = Image.fromarray(color).convert("RGB")
    _draw_legend(image, "3D geometry validation: mesh + lifted grasp points + normals")
    _draw_projected_labels(image, strategies, camera_pose, intrinsic)
    image.save(output_path)


def _project_world_to_pixel(
    point_3d: list[float] | tuple[float, float, float],
    camera_pose: np.ndarray,
    intrinsic: dict[str, Any],
) -> tuple[int, int] | None:
    point = np.array([float(point_3d[0]), float(point_3d[1]), float(point_3d[2]), 1.0], dtype=float)
    point_cam = np.linalg.inv(camera_pose) @ point
    z = -float(point_cam[2])
    if z <= 1e-8:
        return None
    fx = float(intrinsic.get("fx", 1.0))
    fy = float(intrinsic.get("fy", 1.0))
    cx = float(intrinsic.get("cx", 0.0))
    cy = float(intrinsic.get("cy", 0.0))
    u = fx * float(point_cam[0]) / z + cx
    v = cy - fy * float(point_cam[1]) / z
    return int(round(u)), int(round(v))


def _draw_projected_labels(image: Image.Image, strategies: list[dict[str, Any]], camera_pose: np.ndarray, intrinsic: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in strategies:
        point = item.get("adjusted_3d_point")
        if point is None:
            continue
        projected = _project_world_to_pixel(point, camera_pose, intrinsic)
        if projected is None:
            continue
        x, y = projected
        if not (-30 <= x < width + 30 and -30 <= y < height + 30):
            continue
        measured_width = item.get("measured_width")
        width_text = f"{measured_width * 1000:.0f}mm" if isinstance(measured_width, (int, float)) else "n/a"
        label = f"#{item.get('display_id', item.get('id', '?'))} {item.get('audit_status', 'VALID')} {width_text}"
        x0 = min(max(x + 13, 0), max(width - 180, 0))
        y0 = min(max(y - 18, 0), max(height - 24, 0))
        draw.rectangle((x0, y0, x0 + max(118, len(label) * 7), y0 + 23), fill=(0, 0, 0))
        draw.text((x0 + 5, y0 + 4), label, fill=(20, 220, 100), font=_font(12))


def _draw_legend(image: Image.Image, title: str, extra: str | None = None) -> None:
    draw = ImageDraw.Draw(image)
    height = 94 if extra is None else 118
    draw.rectangle((8, 8, 510, height), fill=(0, 0, 0))
    draw.text((18, 18), title, fill=(255, 255, 255), font=_font(16))
    draw.text((18, 44), "green = valid selected 3D point, blue arrow = local normal", fill=(230, 230, 230), font=_font(13))
    draw.text((18, 66), "same camera as Stage 0 RGB-D observation", fill=(230, 230, 230), font=_font(13))
    if extra:
        draw.text((18, 90), extra[:78], fill=(245, 200, 120), font=_font(12))


def _render_fallback_from_pixels(
    *,
    geometry_path: Path,
    camera_path: Path,
    output_path: Path,
    reason: str,
) -> None:
    strategies, data = _load_geometry(geometry_path)
    camera_params = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.exists() else {}
    width = int(camera_params.get("image_width", camera_params.get("image_size", [800, 800])[0]))
    height = int(camera_params.get("image_height", camera_params.get("image_size", [800, 800])[1]))
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    _draw_legend(image, "3D geometry validation fallback: lifted points on render pixels", f"PyRender failed: {reason}")
    colors = {"VALID": (25, 180, 70), "TOO_WIDE": (235, 180, 20), "TOO_THIN": (235, 130, 30)}
    for item in data.get("audit_results", strategies):
        pixel = item.get("selected_pixel") or {}
        x = int(pixel.get("col", 0))
        y = int(pixel.get("row", 0))
        status = item.get("audit_status", "INVALID_DEPTH")
        color = colors.get(status, (200, 70, 70))
        point = item.get("adjusted_3d_point")
        radius = 14 if status == "VALID" else 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
        label = f"#{item.get('id')} {status}"
        if point:
            label += f" [{point[0]:.3f},{point[1]:.3f},{point[2]:.3f}]"
        draw.rectangle((x + 16, y - 15, x + 16 + max(120, len(label) * 7), y + 9), fill=(0, 0, 0))
        draw.text((x + 20, y - 13), label, fill=color, font=_font(12))
    image.save(output_path)


def write_grasp_visualization_placeholder(*, output_path: str | Path) -> Path:
    """Backward-compatible wrapper retained for older imports."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "No geometry file supplied. Run the geometry stage first.", fill="black", font=_font())
    image.save(output_path)
    return output_path
