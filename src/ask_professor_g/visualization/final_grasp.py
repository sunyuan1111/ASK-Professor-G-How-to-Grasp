from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry.probing import load_point_cloud
from ..gripper_axes import local_gripper_axes, project_to_plane
from .real_gripper import build_real_gripper_meshes, estimate_parallel_opening_width


def _font(size: int = 15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _box_mesh(trimesh, center: np.ndarray, size: np.ndarray, rotation: np.ndarray, color: list[int]):
    mesh = trimesh.creation.box(extents=size)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    mesh.visual.vertex_colors = color
    return mesh


def _solid_color_mesh(mesh: Any, color: list[int]) -> Any:
    solid = mesh.copy()
    try:
        import trimesh

        solid.visual = trimesh.visual.ColorVisuals(mesh=solid, vertex_colors=color)
    except Exception:
        solid.visual.vertex_colors = color
    return solid


def _box_vertices(center: np.ndarray, size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
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
    return corners @ rotation.T + center[None, :]


def _proxy_boxes(
    *,
    state: np.ndarray,
    gripper: dict[str, Any],
    object_diag: float,
) -> tuple[list[dict[str, Any]], float]:
    center = state[:3]
    rotation = _rpy_to_matrix(float(state[3]), float(state[4]), float(state[5]))
    axes = local_gripper_axes(gripper)
    local_closing = np.asarray(axes["local_closing_axis"], dtype=float)
    local_tcp = np.asarray(axes["local_tcp_axis"], dtype=float)
    local_span = np.asarray(axes["local_width_axis"], dtype=float)
    local_finger = -local_tcp
    local_finger = project_to_plane(local_finger, local_closing, fallback=local_span)
    local_depth = np.cross(local_closing, local_finger)
    local_depth = local_depth / max(float(np.linalg.norm(local_depth)), 1e-8)
    local_box_rotation = np.column_stack([local_closing, local_finger, local_depth])
    if float(np.linalg.det(local_box_rotation)) < 0.0:
        local_depth = -local_depth
        local_box_rotation = np.column_stack([local_closing, local_finger, local_depth])
    box_rotation = rotation @ local_box_rotation

    max_opening = float(gripper.get("max_opening_width", 0.11))
    s0 = float(np.clip(state[6] if len(state) > 6 else 0.3, 0.0, 1.0))
    opening = float(np.clip(s0 * max_opening, 0.012, max_opening))
    finger_length = min(float(gripper.get("finger_length", max(object_diag * 0.25, 0.04))), max(object_diag * 0.24, 0.032))
    finger_thickness = float(gripper.get("finger_thickness", max(object_diag * 0.035, 0.006)))
    jaw_depth = max(finger_thickness * 0.9, object_diag * 0.035)
    palm_thickness = max(finger_thickness * 1.2, object_diag * 0.035)

    boxes: list[dict[str, Any]] = []
    for side, sign in [("left", -1.0), ("right", 1.0)]:
        local_offset = sign * opening / 2.0 * local_closing - finger_length * 0.25 * local_tcp
        jaw_center = center + rotation @ local_offset
        size = np.array([finger_thickness, finger_length, jaw_depth], dtype=float)
        boxes.append(
            {
                "name": f"{side}_finger",
                "center": jaw_center,
                "size": size,
                "rotation": box_rotation,
                "vertices": _box_vertices(jaw_center, size, box_rotation),
            }
        )

    palm_center = center - rotation @ (local_tcp * finger_length * 0.68)
    palm_size = np.array([opening + finger_thickness * 2.2, palm_thickness * 0.82, jaw_depth * 1.05], dtype=float)
    boxes.append(
        {
            "name": "palm",
            "center": palm_center,
            "size": palm_size,
            "rotation": box_rotation,
            "vertices": _box_vertices(palm_center, palm_size, box_rotation),
        }
    )
    return boxes, opening


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


def _project_world_to_pixel(point_3d: np.ndarray, camera_pose: np.ndarray, intrinsic: dict[str, Any]) -> tuple[int, int] | None:
    point = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=float)
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


def _draw_overlay(
    image: Image.Image,
    best: dict[str, Any],
    camera_pose: np.ndarray,
    intrinsic: dict[str, Any],
    opening: float,
    *,
    gripper_label: str = "gripper proxy",
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 560, 120), fill=(0, 0, 0))
    draw.text((18, 18), f"Final CEM grasp: object mesh + {gripper_label}", fill=(255, 255, 255), font=_font(17))
    draw.text((18, 46), f"rank #1: {best.get('type', 'unknown')}", fill=(40, 220, 150), font=_font(14))
    draw.text(
        (18, 70),
        f"loss={float(best.get('loss', 0.0)):.4f} raw={float(best.get('raw_loss', 0.0)):.4f} opening={opening * 1000:.1f}mm",
        fill=(230, 230, 230),
        font=_font(13),
    )
    draw.text((18, 94), "blue/cyan = gripper, green sphere = optimized TCP/contact center", fill=(230, 230, 230), font=_font(13))

    state = np.asarray(best.get("result", []), dtype=float)
    if len(state) >= 3:
        projected = _project_world_to_pixel(state[:3], camera_pose, intrinsic)
        if projected is not None:
            x, y = projected
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(20, 220, 110), outline=(0, 0, 0), width=2)
            draw.rectangle((x + 12, y - 15, x + 178, y + 10), fill=(0, 0, 0))
            draw.text((x + 17, y - 12), "optimized center", fill=(20, 220, 110), font=_font(12))


def render_final_grasp(
    *,
    object_mesh_path: str | Path,
    stage3_path: str | Path,
    camera_path: str | Path,
    gripper: dict[str, Any],
    output_path: str | Path,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_final_grasp_pyrender(
            object_mesh_path=Path(object_mesh_path),
            stage3_path=Path(stage3_path),
            camera_path=Path(camera_path),
            gripper=gripper,
            output_path=output_path,
            gripper_urdf_path=Path(gripper_urdf_path) if gripper_urdf_path else None,
            gripper_glpca_path=Path(gripper_glpca_path) if gripper_glpca_path else None,
        )
    except Exception as exc:
        _render_fallback(stage3_path=Path(stage3_path), output_path=output_path, reason=str(exc))
    return output_path


def render_final_grasp_real_views(
    *,
    object_mesh_path: str | Path,
    stage3_path: str | Path,
    gripper: dict[str, Any],
    output_path: str | Path,
    gripper_urdf_path: str | Path,
    gripper_glpca_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_final_grasp_real_views_pyrender(
            object_mesh_path=Path(object_mesh_path),
            stage3_path=Path(stage3_path),
            gripper=gripper,
            output_path=output_path,
            gripper_urdf_path=Path(gripper_urdf_path),
            gripper_glpca_path=Path(gripper_glpca_path) if gripper_glpca_path else None,
        )
    except Exception as exc:
        _render_fallback(stage3_path=Path(stage3_path), output_path=output_path, reason=str(exc))
    return output_path


def _render_final_grasp_pyrender(
    *,
    object_mesh_path: Path,
    stage3_path: Path,
    camera_path: Path,
    gripper: dict[str, Any],
    output_path: Path,
    gripper_urdf_path: Path | None = None,
    gripper_glpca_path: Path | None = None,
) -> None:
    if os.name != "nt":
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import pyrender
    import trimesh

    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    grasps = stage3.get("grasps", [])
    if not grasps:
        raise ValueError("No grasps in step3 output.")
    best = grasps[0]
    state = np.asarray(best["result"], dtype=float)
    center = state[:3]
    rotation = _rpy_to_matrix(float(state[3]), float(state[4]), float(state[5]))

    object_mesh = trimesh.load(object_mesh_path, force="mesh")
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))
    object_mesh = _solid_color_mesh(object_mesh, [185, 185, 185, 255])
    bounds = np.asarray(object_mesh.bounds, dtype=float)
    diag = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-4)

    boxes, opening = _proxy_boxes(state=state, gripper=gripper, object_diag=diag)
    real_gripper = None
    if gripper_urdf_path and gripper_urdf_path.exists():
        real_gripper = build_real_gripper_meshes(
            state=state,
            gripper=gripper,
            urdf_path=gripper_urdf_path,
            glpca_path=gripper_glpca_path,
            color=[70, 145, 255, 215],
        )

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.34, 0.34, 0.34])
    scene.add(pyrender.Mesh.from_trimesh(object_mesh, smooth=True))
    for mesh in _axis_meshes(trimesh, length=diag * 0.35, radius=diag * 0.008):
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    gripper_label = "gripper proxy"
    if real_gripper is not None:
        gripper_label = f"real gripper mesh ({real_gripper.source})"
        for mesh in real_gripper.meshes:
            scene.add(pyrender.Mesh.from_trimesh(_solid_color_mesh(mesh, [70, 145, 255, 215]), smooth=False))
    else:
        gripper_color = [0, 185, 230, 205]
        dark_color = [10, 60, 75, 230]
        for box in boxes:
            color = dark_color if box["name"] == "palm" else gripper_color
            mesh = _box_mesh(
                trimesh,
                box["center"],
                box["size"],
                box["rotation"],
                color,
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    tcp = trimesh.creation.icosphere(subdivisions=2, radius=diag * 0.026)
    tcp.apply_translation(center)
    tcp.visual.vertex_colors = [20, 220, 110, 255]
    scene.add(pyrender.Mesh.from_trimesh(tcp, smooth=True))

    camera_params = json.loads(camera_path.read_text(encoding="utf-8"))
    image_width = int(camera_params.get("image_width", camera_params.get("image_size", [800, 800])[0]))
    image_height = int(camera_params.get("image_height", camera_params.get("image_size", [800, 800])[1]))
    intrinsic = camera_params.get("intrinsic", {})
    fy = float(intrinsic.get("fy", image_height))
    yfov = float(2.0 * np.arctan(image_height / (2.0 * fy)))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=image_width / image_height)
    if "camera_pose" in camera_params and camera_params.get("projection_type") != "orthographic_xz":
        camera_pose = np.asarray(camera_params["camera_pose"], dtype=float)
    else:
        camera_pose = _fallback_camera_pose(bounds)
    scene.add(camera, pose=camera_pose)

    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=np.eye(4))
    pose2 = np.eye(4)
    pose2[:3, 3] = [0.4, -0.2, 0.8]
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=pose2)

    renderer = pyrender.OffscreenRenderer(image_width, image_height)
    color, _ = renderer.render(scene)
    renderer.delete()
    image = Image.fromarray(color).convert("RGB")
    _draw_overlay(image, best, camera_pose, intrinsic, opening, gripper_label=gripper_label)
    image.save(output_path)


def _render_final_grasp_real_views_pyrender(
    *,
    object_mesh_path: Path,
    stage3_path: Path,
    gripper: dict[str, Any],
    output_path: Path,
    gripper_urdf_path: Path,
    gripper_glpca_path: Path | None = None,
) -> None:
    if os.name != "nt":
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import pyrender
    import trimesh

    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    grasps = stage3.get("grasps", [])
    if not grasps:
        raise ValueError("No grasps in step3 output.")
    best = grasps[0]
    state = np.asarray(best["result"], dtype=float)
    tcp = state[:3]

    object_mesh = trimesh.load(object_mesh_path, force="mesh")
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))
    object_mesh = _solid_color_mesh(object_mesh, [185, 185, 185, 255])
    object_bounds = np.asarray(object_mesh.bounds, dtype=float)
    object_diag = max(float(np.linalg.norm(object_bounds[1] - object_bounds[0])), 1e-4)

    real_gripper = build_real_gripper_meshes(
        state=state,
        gripper=gripper,
        urdf_path=gripper_urdf_path,
        glpca_path=gripper_glpca_path,
        color=[70, 145, 255, 215],
    )
    all_vertices = [np.asarray(object_mesh.vertices, dtype=float)]
    all_vertices.extend(np.asarray(mesh.vertices, dtype=float) for mesh in real_gripper.meshes)
    combined_points = np.concatenate(all_vertices, axis=0)
    combined_bounds = np.stack([combined_points.min(axis=0), combined_points.max(axis=0)], axis=0)
    combined_center = combined_bounds.mean(axis=0)
    combined_radius = max(float(np.linalg.norm(combined_bounds[1] - combined_bounds[0]) / 2.0), 1e-4)

    panel_specs = [
        ("Whole scene", combined_center, combined_radius * 1.15, np.array([0.7, -0.9, 0.45]), np.array([0.0, 0.0, 1.0])),
        ("Top view", combined_center, combined_radius * 1.05, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        ("Side view", combined_center, combined_radius * 1.05, np.array([1.0, 0.0, 0.25]), np.array([0.0, 0.0, 1.0])),
        ("Contact crop", tcp, max(object_diag * 0.72, 0.06), np.array([0.8, -0.9, 0.38]), np.array([0.0, 0.0, 1.0])),
    ]

    panels: list[Image.Image] = []
    for title, target, radius, direction, up in panel_specs:
        panel = _render_real_view_panel(
            pyrender=pyrender,
            trimesh=trimesh,
            object_mesh=object_mesh,
            gripper_meshes=real_gripper.meshes,
            tcp=tcp,
            title=title,
            target=np.asarray(target, dtype=float),
            radius=float(radius),
            direction=direction,
            up=up,
        )
        panels.append(panel)

    canvas = Image.new("RGB", (1240, 1320), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 24), "Final Grasp With Real Gripper Mesh", fill=(10, 15, 25), font=_font(26))
    draw.text(
        (34, 62),
        "Object mesh + URDF visual meshes. Green sphere marks the optimized TCP/contact center.",
        fill=(70, 78, 92),
        font=_font(15),
    )
    positions = [(20, 108), (630, 108), (20, 714), (630, 714)]
    for panel, position in zip(panels, positions):
        canvas.paste(panel, position)
    canvas.save(output_path)


def _render_real_view_panel(
    *,
    pyrender: Any,
    trimesh: Any,
    object_mesh: Any,
    gripper_meshes: list[Any],
    tcp: np.ndarray,
    title: str,
    target: np.ndarray,
    radius: float,
    direction: np.ndarray,
    up: np.ndarray,
) -> Image.Image:
    image_size = 590
    yfov = np.deg2rad(42.0)
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    distance = max(radius / np.tan(yfov / 2.0), radius * 2.0)
    camera_pose = _look_at(target + direction * distance, target, up=up)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.34, 0.34, 0.34])
    scene.add(pyrender.Mesh.from_trimesh(_solid_color_mesh(object_mesh, [185, 185, 185, 255]), smooth=True))
    for mesh in gripper_meshes:
        scene.add(pyrender.Mesh.from_trimesh(_solid_color_mesh(mesh, [70, 145, 255, 215]), smooth=False))

    marker = trimesh.creation.icosphere(subdivisions=2, radius=max(radius * 0.035, 0.0025))
    marker.apply_translation(tcp)
    marker.visual.vertex_colors = [20, 220, 110, 255]
    scene.add(pyrender.Mesh.from_trimesh(marker, smooth=True))

    camera = pyrender.PerspectiveCamera(yfov=float(yfov), aspectRatio=1.0, znear=0.001, zfar=10.0)
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=camera_pose)
    light_pose = np.eye(4)
    light_pose[:3, 3] = target + np.array([0.25, -0.35, 0.5])
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.8), pose=light_pose)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    color, _ = renderer.render(scene)
    renderer.delete()
    image = Image.fromarray(color).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image_size, 36), fill=(0, 0, 0))
    draw.text((14, 9), title, fill=(255, 255, 255), font=_font(15))
    return image


def _fallback_camera_pose(bounds: np.ndarray) -> np.ndarray:
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    scene_size = max(float(np.max(extent)), 1e-4)
    camera_distance = scene_size * 2.5
    position = center + np.array([0.5 * camera_distance, 0.8 * camera_distance, 0.5 * camera_distance])
    return _look_at(position, center, up=np.array([0.0, 0.0, 1.0]))


def _look_at(camera_position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = camera_position - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position
    return pose


def _render_fallback(*, stage3_path: Path, output_path: Path, reason: str) -> None:
    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    image = Image.new("RGB", (900, 520), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    draw.text((24, 22), "Final grasp render fallback", fill=(10, 15, 25), font=_font(24))
    draw.text((24, 58), f"PyRender failed: {reason[:100]}", fill=(120, 60, 40), font=_font(13))
    for idx, grasp in enumerate(stage3.get("grasps", [])[:5]):
        y = 105 + idx * 58
        state = ", ".join(f"{v:.3f}" for v in grasp.get("result", [])[:9])
        draw.text((34, y), f"#{idx + 1} {grasp.get('type')} loss={float(grasp.get('loss', 0.0)):.4f}", fill=(20, 30, 45), font=_font(15))
        draw.text((54, y + 24), state, fill=(70, 78, 92), font=_font(12))
    image.save(output_path)


def draw_final_grasp_diagnostics(
    *,
    point_cloud_path: str | Path,
    stage3_path: str | Path,
    stage1_processed_path: str | Path,
    gripper: dict[str, Any],
    output_path: str | Path,
    report_path: str | Path,
    gripper_urdf_path: str | Path | None = None,
    gripper_glpca_path: str | Path | None = None,
) -> dict[str, Any]:
    points = load_point_cloud(point_cloud_path)
    stage3 = json.loads(Path(stage3_path).read_text(encoding="utf-8"))
    stage1 = json.loads(Path(stage1_processed_path).read_text(encoding="utf-8"))
    grasps = stage3.get("grasps", [])
    if not grasps:
        raise ValueError("No grasps in step3 output.")

    best = grasps[0]
    state = np.asarray(best["result"], dtype=float)
    source = _find_stage1_source(best, stage1)
    surface_target = np.asarray(source.get("source_3d_point", state[:3]), dtype=float)
    target = np.asarray(source.get("source_clearance_point", surface_target), dtype=float)
    measured_width = source.get("source_measured_width")
    normal_value = source.get("measured_normal_used") or source.get("measured_normal")
    normal = _normal_or_none(normal_value)
    signed_clearance = None if normal is None else float(np.dot(state[:3] - surface_target, normal))
    requested_clearance = source.get("surface_clearance_m")
    axis_metrics = _axis_alignment_metrics(state, gripper, source, normal)

    object_diag = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-4)
    boxes, proxy_opening = _proxy_boxes(state=state, gripper=gripper, object_diag=object_diag)
    opening, opening_estimation = estimate_parallel_opening_width(
        state=state,
        gripper=gripper,
        urdf_path=gripper_urdf_path,
        glpca_path=gripper_glpca_path,
    )
    if opening_estimation.get("source") == "linear_config_fallback":
        opening = proxy_opening
    all_box_vertices = np.concatenate([box["vertices"] for box in boxes], axis=0)
    all_points = np.concatenate([points, all_box_vertices, state[:3][None, :], target[None, :], surface_target[None, :]], axis=0)
    bounds = (all_points.min(axis=0), all_points.max(axis=0))
    padding = object_diag * 0.12
    bounds = (bounds[0] - padding, bounds[1] + padding)

    nearest_dist = float(np.min(np.linalg.norm(points - state[:3][None, :], axis=1)))
    target_dist = float(np.linalg.norm(state[:3] - target))
    width_margin = None if measured_width is None else float(opening - float(measured_width))
    report = {
        "best_type": best.get("type"),
        "loss": best.get("loss"),
        "optimized_center": state[:3].tolist(),
        "source_target": target.tolist(),
        "surface_target": surface_target.tolist(),
        "target_distance_m": target_dist,
        "nearest_object_distance_m": nearest_dist,
        "surface_signed_clearance_m": signed_clearance,
        "requested_surface_clearance_m": requested_clearance,
        "opening_m": opening,
        "proxy_opening_m": proxy_opening,
        "opening_estimation": opening_estimation,
        "measured_width_m": measured_width,
        "opening_minus_measured_width_m": width_margin,
        "category": best.get("category"),
        **axis_metrics,
        "diagnostics": {
            "target_alignment": "OK" if target_dist <= 0.015 else "CHECK",
            "opening": "OK" if width_margin is None or width_margin >= -0.003 else "TOO_NARROW",
            "surface_distance": "OK" if nearest_dist <= 0.02 else "FAR_FROM_OBJECT",
            "surface_clearance": _clearance_status(signed_clearance, requested_clearance),
            "closing_axis_alignment": _alignment_status(axis_metrics.get("closing_alignment_dot")),
            "tcp_axis_alignment": _alignment_status(axis_metrics.get("tcp_axis_alignment_dot")),
        },
    }

    image = Image.new("RGBA", (1500, 940), (248, 249, 251, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((34, 26), "Final Grasp Diagnostic Views", fill=(10, 15, 25, 255), font=_font(26))
    draw.text(
        (34, 62),
        "Object point cloud with CEM gripper proxy. Green = optimized TCP, magenta = clearance target, orange = raw surface point.",
        fill=(70, 78, 92, 255),
        font=_font(15),
    )

    panels = [
        ((42, 112, 482, 702), (0, 1), "XY top view"),
        ((530, 112, 970, 702), (0, 2), "XZ front view"),
        ((1018, 112, 1458, 702), (1, 2), "YZ side view"),
    ]
    for box, axes, title in panels:
        _draw_projection_panel(draw, points, boxes, state[:3], target, bounds, box, axes, title, surface_target=surface_target)

    _draw_report(draw, report, top_left=(46, 742))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _normal_or_none(value: Any) -> np.ndarray | None:
    try:
        normal = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        return None
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    return normal / norm


def _clearance_status(signed_clearance: float | None, requested_clearance: Any) -> str:
    if signed_clearance is None:
        return "UNKNOWN"
    requested = float(requested_clearance or 0.004)
    if signed_clearance < -0.001:
        return "INSIDE"
    if signed_clearance + 0.002 < requested:
        return "LOW_CLEARANCE"
    return "OK"


def _alignment_status(dot_value: Any) -> str:
    if dot_value is None:
        return "UNKNOWN"
    return "OK" if float(dot_value) >= 0.85 else "CHECK"


def _axis_alignment_metrics(
    state: np.ndarray,
    gripper: dict[str, Any],
    source: dict[str, Any],
    normal: np.ndarray | None,
) -> dict[str, Any]:
    axes = local_gripper_axes(gripper)
    rotation = _rpy_to_matrix(float(state[3]), float(state[4]), float(state[5]))
    local_closing = np.asarray(axes["local_closing_axis"], dtype=float)
    local_tcp = np.asarray(axes["local_tcp_axis"], dtype=float)
    world_closing = rotation @ local_closing
    world_tcp = rotation @ local_tcp
    metrics: dict[str, Any] = {
        "gripper_local_closing_axis": local_closing.tolist(),
        "gripper_local_tcp_axis": local_tcp.tolist(),
        "world_closing_axis": world_closing.tolist(),
        "world_tcp_axis": world_tcp.tolist(),
    }
    closing_value = source.get("closing_direction_used") or source.get("closing_direction")
    if closing_value is not None and normal is not None:
        target_tcp = -normal
        target_closing = project_to_plane(closing_value, target_tcp, fallback="+X")
        metrics["target_closing_axis"] = target_closing.tolist()
        metrics["closing_alignment_dot"] = float(np.dot(world_closing, target_closing))
    if normal is not None:
        target_tcp = -normal
        metrics["target_tcp_axis"] = target_tcp.tolist()
        metrics["tcp_axis_alignment_dot"] = float(np.dot(world_tcp, target_tcp))
    return metrics


def _find_stage1_source(best: dict[str, Any], stage1: dict[str, Any]) -> dict[str, Any]:
    best_type = str(best.get("type", ""))
    for grasp in stage1.get("grasps", []):
        if str(grasp.get("type", "")) == best_type:
            return grasp
    for grasp in stage1.get("grasps", []):
        if str(grasp.get("type", "")).split("_")[0] in best_type:
            return grasp
    return {}


def _project_points(points: np.ndarray, axes: tuple[int, int], bounds: tuple[np.ndarray, np.ndarray], box: tuple[int, int, int, int]) -> np.ndarray:
    mins, maxs = bounds
    x0, y0, x1, y1 = box
    span = np.maximum(maxs - mins, 1e-9)
    xs = (points[:, axes[0]] - mins[axes[0]]) / span[axes[0]]
    ys = (points[:, axes[1]] - mins[axes[1]]) / span[axes[1]]
    px = x0 + xs * (x1 - x0)
    py = y1 - ys * (y1 - y0)
    return np.stack([px, py], axis=1)


def _draw_projection_panel(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    boxes: list[dict[str, Any]],
    center: np.ndarray,
    target: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    box: tuple[int, int, int, int],
    axes: tuple[int, int],
    title: str,
    *,
    surface_target: np.ndarray | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6, outline=(35, 41, 51, 255), width=2, fill=(255, 255, 255, 255))
    draw.text((x0 + 12, y0 + 10), title, fill=(20, 24, 31, 255), font=_font(16))
    inner = (x0 + 18, y0 + 42, x1 - 18, y1 - 18)
    projected_points = _project_points(points, axes, bounds, inner)
    step = max(1, len(points) // 3500)
    for px, py in projected_points[::step]:
        draw.point((float(px), float(py)), fill=(165, 170, 178, 210))

    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    for gripper_box in boxes:
        projected = _project_points(gripper_box["vertices"], axes, bounds, inner)
        fill = (0, 188, 230, 88) if gripper_box["name"] != "palm" else (5, 75, 90, 62)
        outline = (0, 120, 150, 230)
        for face in face_indices:
            polygon = [(float(projected[idx, 0]), float(projected[idx, 1])) for idx in face]
            draw.polygon(polygon, fill=fill)
            draw.line(polygon + [polygon[0]], fill=outline, width=2)

    center_px = _project_points(center[None, :], axes, bounds, inner)[0]
    target_px = _project_points(target[None, :], axes, bounds, inner)[0]
    draw.ellipse((center_px[0] - 7, center_px[1] - 7, center_px[0] + 7, center_px[1] + 7), fill=(20, 220, 110, 255), outline=(0, 0, 0, 255), width=2)
    draw.line((target_px[0] - 8, target_px[1], target_px[0] + 8, target_px[1]), fill=(220, 40, 210, 255), width=3)
    draw.line((target_px[0], target_px[1] - 8, target_px[0], target_px[1] + 8), fill=(220, 40, 210, 255), width=3)
    if surface_target is not None:
        surface_px = _project_points(surface_target[None, :], axes, bounds, inner)[0]
        draw.ellipse((surface_px[0] - 5, surface_px[1] - 5, surface_px[0] + 5, surface_px[1] + 5), outline=(255, 140, 0, 255), width=2)
        draw.line((surface_px[0], surface_px[1], target_px[0], target_px[1]), fill=(255, 140, 0, 170), width=1)
    draw.line((center_px[0], center_px[1], target_px[0], target_px[1]), fill=(40, 80, 210, 180), width=1)


def _draw_report(draw: ImageDraw.ImageDraw, report: dict[str, Any], top_left: tuple[int, int]) -> None:
    x, y = top_left
    draw.rounded_rectangle((x, y, x + 1412, y + 156), radius=6, fill=(255, 255, 255, 255), outline=(35, 41, 51, 255), width=2)
    draw.text((x + 16, y + 14), f"Best grasp: {report['best_type']}", fill=(10, 15, 25, 255), font=_font(17))
    rows = [
        ("loss", f"{float(report.get('loss', 0.0)):.4f}"),
        ("target distance", f"{report['target_distance_m'] * 1000:.1f} mm ({report['diagnostics']['target_alignment']})"),
        ("nearest object distance", f"{report['nearest_object_distance_m'] * 1000:.1f} mm ({report['diagnostics']['surface_distance']})"),
        ("opening", f"{report['opening_m'] * 1000:.1f} mm ({report['diagnostics']['opening']})"),
        ("measured width", "n/a" if report["measured_width_m"] is None else f"{float(report['measured_width_m']) * 1000:.1f} mm"),
        ("opening margin", "n/a" if report["opening_minus_measured_width_m"] is None else f"{float(report['opening_minus_measured_width_m']) * 1000:.1f} mm"),
        ("surface clearance", "n/a" if report["surface_signed_clearance_m"] is None else f"{float(report['surface_signed_clearance_m']) * 1000:.1f} mm ({report['diagnostics']['surface_clearance']})"),
        ("closing axis", "n/a" if report.get("closing_alignment_dot") is None else f"{float(report['closing_alignment_dot']):.3f} ({report['diagnostics']['closing_axis_alignment']})"),
        ("tcp axis", "n/a" if report.get("tcp_axis_alignment_dot") is None else f"{float(report['tcp_axis_alignment_dot']):.3f} ({report['diagnostics']['tcp_axis_alignment']})"),
    ]
    for idx, (label, value) in enumerate(rows):
        col = idx % 3
        row = idx // 3
        tx = x + 18 + col * 450
        ty = y + 54 + row * 42
        draw.text((tx, ty), label, fill=(70, 78, 92, 255), font=_font(13))
        draw.text((tx + 160, ty), value, fill=(20, 24, 31, 255), font=_font(14))
