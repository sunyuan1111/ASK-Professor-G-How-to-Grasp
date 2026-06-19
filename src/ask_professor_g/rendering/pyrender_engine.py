from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry.probing import load_point_cloud


def _font(size: int = 15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def project_points_xz(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], size: int, margin: int) -> np.ndarray:
    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-9)
    u = margin + (points[:, 0] - mins[0]) / span[0] * (size - 2 * margin)
    v = size - margin - (points[:, 2] - mins[2]) / span[2] * (size - 2 * margin)
    return np.stack([u, v], axis=1)


def render_observation(mesh_path: str | Path, point_cloud_path: str | Path, *, output_dir: str | Path, size: int = 800) -> dict:
    """Render the Stage 0 RGB-D observation.

    This is the public entrypoint used by the pipeline. It tries to use the migrated PyRender
    renderer first because that matches the original project: RGB image with axes, clean depth
    buffer without axes, and camera calibration for 2D-to-3D lifting. On machines without a
    working OpenGL/PyRender stack it falls back to the deterministic orthographic renderer.
    """
    try:
        return render_pyrender_observation(mesh_path, output_dir=output_dir, size=size)
    except Exception as exc:
        result = render_placeholder(point_cloud_path, output_dir=output_dir, size=size)
        result["fallback_reason"] = f"PyRender unavailable or failed: {exc}"
        return result


def render_pyrender_observation(mesh_path: str | Path, *, output_dir: str | Path, size: int = 800) -> dict:
    if os.name != "nt":
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import trimesh
    import pyrender

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.visual.kind is None:
        mesh.visual.vertex_colors = [185, 185, 185, 255]

    center = np.asarray(mesh.centroid, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    extent = bounds[1] - bounds[0]
    scene_size = max(float(np.max(extent)), 1e-4)
    camera_distance = scene_size * 2.5
    camera_position = center + np.array([0.5 * camera_distance, 0.8 * camera_distance, 0.5 * camera_distance])
    camera_pose = _look_at(camera_position, center, up=np.array([0.0, 0.0, 1.0]))
    yfov = np.deg2rad(45.0)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.32, 0.32, 0.32])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    for axis_mesh in _axis_meshes(trimesh, length=scene_size * 0.55, radius=scene_size * 0.014):
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)
    _add_lights(scene, pyrender)

    renderer = pyrender.OffscreenRenderer(size, size)
    color, _ = renderer.render(scene)
    rgb_path = output_dir / "view_front_iso_rgb.png"
    Image.fromarray(color).save(rgb_path)

    # Re-render clean depth without axis geometry.
    depth_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.32, 0.32, 0.32])
    depth_scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    depth_scene.add(camera, pose=camera_pose)
    _add_lights(depth_scene, pyrender)
    _, depth = renderer.render(depth_scene)
    renderer.delete()

    depth_path = output_dir / "view_front_iso_depth.npy"
    np.save(depth_path, depth.astype(np.float32))
    _save_depth_preview(depth, output_dir / "view_front_iso_depth_vis.png")

    intrinsic = _intrinsic_from_yfov(size, yfov)
    extrinsic = np.linalg.inv(camera_pose)
    camera_payload = {
        "projection_type": "pyrender_perspective",
        "image_size": [size, size],
        "image_width": size,
        "image_height": size,
        "intrinsic": {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "intrinsic_matrix": intrinsic.tolist(),
        },
        "extrinsic": extrinsic.tolist(),
        "camera_pose": camera_pose.tolist(),
        "depth_type": "pyrender z-buffer distance in meters along camera -Z",
        "mesh_bounds": bounds.tolist(),
        "mesh_centroid": center.tolist(),
    }
    camera_path = output_dir / "view_front_iso_camera.json"
    camera_path.write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")

    return {
        "renderer": "pyrender",
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "depth_vis": str(output_dir / "view_front_iso_depth_vis.png"),
        "camera": str(camera_path),
        "size": size,
    }


def _look_at(camera_position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = camera_position - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position
    return pose


def _intrinsic_from_yfov(size: int, yfov: float) -> np.ndarray:
    fy = size / (2.0 * np.tan(yfov / 2.0))
    fx = fy
    cx = size / 2.0
    cy = size / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _axis_meshes(trimesh_module, *, length: float, radius: float):
    axes = []
    specs = [
        ([1, 0, 0], [0, 1, 0], np.pi / 2, [255, 0, 0, 255]),
        ([0, 1, 0], [1, 0, 0], -np.pi / 2, [0, 180, 0, 255]),
        ([0, 0, 1], None, 0.0, [0, 0, 255, 255]),
    ]
    for direction, rot_axis, angle, color in specs:
        cyl = trimesh_module.creation.cylinder(radius=radius, height=length)
        if rot_axis is not None:
            cyl.apply_transform(trimesh_module.transformations.rotation_matrix(angle, rot_axis))
        cyl.apply_translation(np.asarray(direction, dtype=float) * length / 2.0)
        cyl.visual.vertex_colors = color
        axes.append(cyl)

        cone = trimesh_module.creation.cone(radius=radius * 2.5, height=length * 0.16)
        if rot_axis is not None:
            cone.apply_transform(trimesh_module.transformations.rotation_matrix(angle, rot_axis))
        cone.apply_translation(np.asarray(direction, dtype=float) * (length + length * 0.08))
        cone.visual.vertex_colors = color
        axes.append(cone)
    origin = trimesh_module.creation.uv_sphere(radius=radius * 2.0)
    origin.visual.vertex_colors = [90, 90, 90, 255]
    axes.append(origin)
    return axes


def _add_lights(scene, pyrender_module) -> None:
    light = pyrender_module.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    pose = np.eye(4)
    scene.add(light, pose=pose)
    light2 = pyrender_module.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    pose2 = np.eye(4)
    pose2[:3, 3] = [0.2, -0.4, 0.8]
    scene.add(light2, pose=pose2)


def _save_depth_preview(depth: np.ndarray, path: Path) -> None:
    valid = depth[depth > 0]
    if len(valid) == 0:
        Image.new("L", depth.shape[::-1], 0).save(path)
        return
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[depth > 0] = (depth[depth > 0] - valid.min()) / (valid.max() - valid.min() + 1e-8)
    Image.fromarray(np.uint8(np.clip(norm, 0, 1) * 255)).save(path)


def render_placeholder(mesh_or_cloud_path: str | Path, *, output_dir: str | Path, size: int = 800) -> dict:
    """Create a deterministic RGB-D style observation from the object point cloud.

    This is a headless replacement for the original PyRender path. It intentionally writes the
    same core artifacts used by the method: an RGB observation, a depth/projection file, and
    calibration metadata. The projection is orthographic X-Z, so Stage 0 image points can be
    back-projected deterministically to the nearest visible 3D point.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_point_cloud(mesh_or_cloud_path)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    margin = int(size * 0.12)
    projected = project_points_xz(points, (mins, maxs), size, margin)

    image = Image.new("RGB", (size, size), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    order = np.argsort(points[:, 1])
    y_span = max(float(maxs[1] - mins[1]), 1e-9)
    for idx in order:
        x, y = projected[idx]
        shade = int(80 + 145 * (points[idx, 1] - mins[1]) / y_span)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(shade, shade, shade + 18))

    # Axis overlay matching the paper/old pipeline convention.
    origin = (margin, size - margin)
    axis_len = int(size * 0.16)
    draw.line((origin[0], origin[1], origin[0] + axis_len, origin[1]), fill=(220, 38, 38), width=4)
    draw.line((origin[0], origin[1], origin[0] + int(axis_len * 0.65), origin[1] - int(axis_len * 0.45)), fill=(22, 163, 74), width=4)
    draw.line((origin[0], origin[1], origin[0], origin[1] - axis_len), fill=(37, 99, 235), width=4)
    draw.text((origin[0] + axis_len + 6, origin[1] - 8), "+X", fill=(180, 20, 20), font=_font())
    draw.text((origin[0] + int(axis_len * 0.65) + 6, origin[1] - int(axis_len * 0.45) - 8), "+Y", fill=(20, 130, 50), font=_font())
    draw.text((origin[0] - 10, origin[1] - axis_len - 24), "+Z", fill=(25, 80, 190), font=_font())
    draw.text((24, 24), "RGB observation for Stage 0 (orthographic X-Z view)", fill=(20, 24, 32), font=_font(18))

    rgb_path = output_dir / "view_front_iso_rgb.png"
    image.save(rgb_path)

    depth = np.full((size, size), np.nan, dtype=np.float32)
    for idx in order:
        u, v = projected[idx]
        col = int(round(u))
        row = int(round(v))
        if 0 <= row < size and 0 <= col < size:
            depth[row, col] = points[idx, 1]
    depth_path = output_dir / "view_front_iso_depth.npy"
    np.save(depth_path, depth)
    camera = {
        "projection_type": "orthographic_xz",
        "intrinsic": {"fx": size, "fy": size, "cx": size / 2.0, "cy": size / 2.0},
        "camera_pose": np.eye(4).tolist(),
        "image_size": [size, size],
        "margin_px": margin,
        "bounds_min": mins.tolist(),
        "bounds_max": maxs.tolist(),
        "note": "Headless orthographic observation. u maps to X, v maps to Z, depth stores Y.",
    }
    camera_path = output_dir / "view_front_iso_camera.json"
    camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")
    return {"rgb": str(rgb_path), "depth": str(depth_path), "camera": str(camera_path), "size": size}
