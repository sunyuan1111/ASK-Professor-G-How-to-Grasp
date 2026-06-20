from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..gripper_axes import local_gripper_axes


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class VisualSpec:
    link: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray
    color: list[int] | None


@dataclass(frozen=True)
class URDFVisualModel:
    root_link: str
    links: set[str]
    joints: list[JointSpec]
    visuals: list[VisualSpec]


@dataclass(frozen=True)
class RealGripperMeshes:
    meshes: list[Any]
    joint_config: dict[str, float]
    base_transform: np.ndarray
    source: str


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def pose_from_xyz_rpy(xyz: list[float] | np.ndarray, rpy: list[float] | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def state_to_pose_matrix(state: np.ndarray) -> np.ndarray:
    return pose_from_xyz_rpy(state[:3], state[3:6])


def state_to_gripper_base_transform(state: np.ndarray, gripper: dict[str, Any]) -> np.ndarray:
    """Treat state[:3] as the optimized TCP/contact center and recover the URDF base pose."""
    transform = state_to_pose_matrix(state)
    hand_type = str(gripper.get("hand_type", "")).lower()
    if hand_type in {"dexterous", "underactuated"} and gripper.get("tcp_wrap") is not None:
        offset = gripper.get("tcp_wrap", [0.0, 0.0, 0.0])
    else:
        offset = gripper.get("tcp_offset", [0.0, 0.0, 0.0])
    tcp_offset = np.asarray(offset, dtype=float)
    transform[:3, 3] = np.asarray(state[:3], dtype=float) - transform[:3, :3] @ tcp_offset
    return transform


def load_urdf_visual_model(urdf_path: str | Path) -> URDFVisualModel:
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    links = {node.attrib["name"] for node in root.findall("link") if "name" in node.attrib}
    material_colors = _global_material_colors(root)

    joints: list[JointSpec] = []
    child_links: set[str] = set()
    for node in root.findall("joint"):
        parent_node = node.find("parent")
        child_node = node.find("child")
        if parent_node is None or child_node is None:
            continue
        parent = parent_node.attrib.get("link", "")
        child = child_node.attrib.get("link", "")
        child_links.add(child)
        axis_node = node.find("axis")
        limit_node = node.find("limit")
        axis = _parse_vector(axis_node.attrib.get("xyz", "1 0 0") if axis_node is not None else "1 0 0")
        lower = float(limit_node.attrib["lower"]) if limit_node is not None and "lower" in limit_node.attrib else None
        upper = float(limit_node.attrib["upper"]) if limit_node is not None and "upper" in limit_node.attrib else None
        joints.append(
            JointSpec(
                name=node.attrib.get("name", f"{parent}_to_{child}"),
                joint_type=node.attrib.get("type", "fixed"),
                parent=parent,
                child=child,
                origin=_origin_transform(node.find("origin")),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )

    visuals: list[VisualSpec] = []
    for link_node in root.findall("link"):
        link_name = link_node.attrib.get("name", "")
        for visual_node in link_node.findall("visual"):
            geometry_node = visual_node.find("geometry")
            mesh_node = geometry_node.find("mesh") if geometry_node is not None else None
            if mesh_node is None or "filename" not in mesh_node.attrib:
                continue
            visuals.append(
                VisualSpec(
                    link=link_name,
                    mesh_path=_resolve_mesh_path(mesh_node.attrib["filename"], urdf_path),
                    origin=_origin_transform(visual_node.find("origin")),
                    scale=_parse_vector(mesh_node.attrib.get("scale", "1 1 1")),
                    color=_visual_color(visual_node, material_colors),
                )
            )

    roots = sorted(links - child_links)
    root_link = roots[0] if roots else (sorted(links)[0] if links else "base_link")
    return URDFVisualModel(root_link=root_link, links=links, joints=joints, visuals=visuals)


def joint_config_from_state(
    *,
    state: np.ndarray,
    model: URDFVisualModel,
    gripper: dict[str, Any],
    glpca_path: str | Path | None = None,
) -> tuple[dict[str, float], str]:
    if glpca_path and Path(glpca_path).exists():
        try:
            return _joint_config_from_glpca(state=state, model=model, gripper=gripper, glpca_path=Path(glpca_path))
        except Exception:
            pass
    return _joint_config_from_limits(state=state, model=model), "joint_limits"


def build_real_gripper_meshes(
    *,
    state: np.ndarray,
    gripper: dict[str, Any],
    urdf_path: str | Path,
    glpca_path: str | Path | None = None,
    color: list[int] | None = None,
) -> RealGripperMeshes:
    import trimesh

    urdf_path = Path(urdf_path)
    model = load_urdf_visual_model(urdf_path)
    cfg, source = joint_config_from_state(state=state, model=model, gripper=gripper, glpca_path=glpca_path)
    base_transform = state_to_gripper_base_transform(state, gripper)
    link_transforms = _link_transforms(model, cfg, base_transform)
    meshes = []
    for visual in model.visuals:
        if not visual.mesh_path.exists():
            continue
        mesh = trimesh.load_mesh(visual.mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh = mesh.copy()
        scale_transform = np.eye(4, dtype=float)
        scale_transform[0, 0] = float(visual.scale[0])
        scale_transform[1, 1] = float(visual.scale[1])
        scale_transform[2, 2] = float(visual.scale[2])
        mesh.apply_transform(scale_transform)
        mesh.apply_transform(link_transforms.get(visual.link, base_transform) @ visual.origin)
        mesh.visual.vertex_colors = color or visual.color or [80, 150, 255, 220]
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No visual meshes could be loaded from gripper URDF: {urdf_path}")
    return RealGripperMeshes(meshes=meshes, joint_config=cfg, base_transform=base_transform, source=source)


def concatenate_meshes(meshes: list[Any]) -> Any:
    import trimesh

    return trimesh.util.concatenate([mesh.copy() for mesh in meshes])


def _joint_config_from_glpca(
    *,
    state: np.ndarray,
    model: URDFVisualModel,
    gripper: dict[str, Any],
    glpca_path: Path,
) -> tuple[dict[str, float], str]:
    data = np.load(glpca_path, allow_pickle=True)
    required = {"U", "theta_mean", "joint_names"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"gLPCA file missing required keys: {glpca_path}")
    U = np.asarray(data["U"], dtype=float)
    theta_mean = np.asarray(data["theta_mean"], dtype=float)
    scale = np.asarray(data["scale"], dtype=float) if "scale" in data.files else np.ones_like(theta_mean)
    joint_names = [str(name) for name in data["joint_names"].tolist()]
    s_limits = np.asarray(data["s_limits"], dtype=float) if "s_limits" in data.files else None
    k_dim = int(U.shape[1])
    s_vec = np.zeros(k_dim, dtype=float)
    provided = max(0, len(state) - 6)
    invert_s0 = bool(gripper.get("s0_invert", gripper.get("invert_s0", False)))

    for idx in range(k_dim):
        if idx < provided:
            value = float(state[6 + idx])
        elif s_limits is not None:
            value = float((s_limits[idx, 0] + s_limits[idx, 1]) / 2.0)
        else:
            value = 0.0

        if idx == 0 and s_limits is not None and 0.0 <= value <= 1.0:
            lo, hi = float(s_limits[idx, 0]), float(s_limits[idx, 1])
            alpha = 1.0 - value if invert_s0 else value
            value = lo + alpha * (hi - lo)
        s_vec[idx] = value

    theta = theta_mean + scale * (U @ s_vec)
    name_to_theta = {name: float(theta[idx]) for idx, name in enumerate(joint_names)}
    cfg: dict[str, float] = {}
    for joint in _actuated_joints(model):
        if joint.name in name_to_theta:
            cfg[joint.name] = _clip_joint(name_to_theta[joint.name], joint)
        else:
            cfg[joint.name] = _limit_midpoint(joint)
    return cfg, "glpca"


def _joint_config_from_limits(*, state: np.ndarray, model: URDFVisualModel) -> tuple[dict[str, float], str]:
    s0 = float(state[6]) if len(state) > 6 else 0.5
    s0 = max(0.0, min(1.0, s0))
    cfg: dict[str, float] = {}
    for joint in _actuated_joints(model):
        if joint.lower is None or joint.upper is None:
            cfg[joint.name] = 0.0
            continue
        lo, hi = float(joint.lower), float(joint.upper)
        if hi <= 0.0:
            q = -(abs(hi) + s0 * (abs(lo) - abs(hi)))
        elif lo >= 0.0:
            q = lo + s0 * (hi - lo)
        else:
            q = lo + s0 * (hi - lo)
        cfg[joint.name] = _clip_joint(q, joint)
    return cfg, "joint_limits"


def _actuated_joints(model: URDFVisualModel) -> list[JointSpec]:
    return [joint for joint in model.joints if joint.joint_type in {"revolute", "continuous", "prismatic"}]


def _link_transforms(
    model: URDFVisualModel,
    joint_config: dict[str, float],
    root_transform: np.ndarray,
) -> dict[str, np.ndarray]:
    by_parent: dict[str, list[JointSpec]] = {}
    for joint in model.joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    transforms: dict[str, np.ndarray] = {model.root_link: root_transform}
    queue = [model.root_link]
    while queue:
        parent = queue.pop(0)
        parent_transform = transforms[parent]
        for joint in by_parent.get(parent, []):
            q = float(joint_config.get(joint.name, 0.0))
            transforms[joint.child] = parent_transform @ joint.origin @ _joint_motion(joint, q)
            queue.append(joint.child)
    return transforms


def _joint_motion(joint: JointSpec, value: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    axis = np.asarray(joint.axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    if joint.joint_type == "prismatic":
        transform[:3, 3] = axis * value
    elif joint.joint_type in {"revolute", "continuous"}:
        transform[:3, :3] = _axis_angle_matrix(axis, value)
    return transform


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def _origin_transform(origin_node: ET.Element | None) -> np.ndarray:
    if origin_node is None:
        return np.eye(4, dtype=float)
    xyz = _parse_vector(origin_node.attrib.get("xyz", "0 0 0"))
    rpy = _parse_vector(origin_node.attrib.get("rpy", "0 0 0"))
    return pose_from_xyz_rpy(xyz, rpy)


def _parse_vector(text: str) -> np.ndarray:
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"Expected 3-vector, got: {text}")
    return np.asarray(values, dtype=float)


def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    filename = filename.replace("\\", "/")
    if filename.startswith("file://"):
        filename = filename[7:]
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
        parts = filename.split("/", 1)
        filename = parts[1] if len(parts) == 2 else parts[0]
    path = Path(filename)
    if path.is_absolute():
        return path
    return (urdf_path.parent / path).resolve()


def _global_material_colors(root: ET.Element) -> dict[str, list[int]]:
    colors: dict[str, list[int]] = {}
    for material in root.findall("material"):
        name = material.attrib.get("name")
        color_node = material.find("color")
        if name and color_node is not None and "rgba" in color_node.attrib:
            colors[name] = _rgba_to_color(color_node.attrib["rgba"])
    return colors


def _visual_color(visual_node: ET.Element, material_colors: dict[str, list[int]]) -> list[int] | None:
    material = visual_node.find("material")
    if material is None:
        return None
    color_node = material.find("color")
    if color_node is not None and "rgba" in color_node.attrib:
        return _rgba_to_color(color_node.attrib["rgba"])
    name = material.attrib.get("name")
    return material_colors.get(name or "")


def _rgba_to_color(text: str) -> list[int]:
    values = [float(part) for part in text.split()]
    if len(values) == 3:
        values.append(1.0)
    rgba = [max(0, min(255, int(round(value * 255)))) for value in values[:4]]
    return rgba


def _clip_joint(value: float, joint: JointSpec) -> float:
    if joint.lower is not None:
        value = max(float(joint.lower), value)
    if joint.upper is not None:
        value = min(float(joint.upper), value)
    return float(value)


def _limit_midpoint(joint: JointSpec) -> float:
    if joint.lower is not None and joint.upper is not None:
        return float((joint.lower + joint.upper) / 2.0)
    return 0.0



def estimate_parallel_opening_width(
    *,
    state: np.ndarray,
    gripper: dict[str, Any],
    urdf_path: str | Path | None = None,
    glpca_path: str | Path | None = None,
) -> tuple[float, dict[str, Any]]:
    """Estimate current jaw opening from URDF actuated link poses.

    The public pipeline treats s0 as normalized opening, but gLPCA may be nonlinear or
    inverted. Measuring link separation along the configured local closing axis keeps
    diagnostics and planning tied to the rendered hand.
    """
    fallback = _fallback_opening_width(state, gripper)
    if urdf_path is None or not Path(urdf_path).exists():
        return fallback, {"source": "linear_config_fallback"}
    try:
        model = load_urdf_visual_model(urdf_path)
        joint_config, source = joint_config_from_state(
            state=state,
            model=model,
            gripper=gripper,
            glpca_path=glpca_path,
        )
        transforms = _link_transforms(model, joint_config, np.eye(4, dtype=float))
        actuated_children = [joint.child for joint in _actuated_joints(model) if joint.child in transforms]
        if len(actuated_children) < 2:
            return fallback, {"source": "linear_config_fallback", "reason": "fewer_than_two_actuated_links"}
        closing_axis = np.asarray(local_gripper_axes(gripper)["local_closing_axis"], dtype=float)
        closing_axis = closing_axis / max(float(np.linalg.norm(closing_axis)), 1e-9)
        projections = [float(np.dot(transforms[child][:3, 3], closing_axis)) for child in actuated_children]
        opening = max(projections) - min(projections)
        if not np.isfinite(opening) or opening < 0.0:
            return fallback, {"source": "linear_config_fallback", "reason": "invalid_projection"}
        return float(opening), {
            "source": f"urdf_{source}",
            "actuated_links": actuated_children,
            "joint_config": {name: float(value) for name, value in joint_config.items()},
            "projection_axis": closing_axis.tolist(),
            "link_projections": projections,
        }
    except Exception as exc:
        return fallback, {"source": "linear_config_fallback", "reason": str(exc)}


def calibrated_s0_for_opening_width(
    *,
    desired_opening: float,
    gripper: dict[str, Any],
    urdf_path: str | Path | None = None,
    glpca_path: str | Path | None = None,
    num_samples: int = 101,
) -> tuple[float, float, dict[str, Any]]:
    desired = max(0.0, float(desired_opening))
    samples: list[tuple[float, float]] = []
    for s0 in np.linspace(0.0, 1.0, max(3, int(num_samples))):
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(s0), 0.0, 0.0], dtype=float)
        opening, _ = estimate_parallel_opening_width(
            state=state,
            gripper=gripper,
            urdf_path=urdf_path,
            glpca_path=glpca_path,
        )
        samples.append((float(s0), float(opening)))

    finite = [(s0, width) for s0, width in samples if np.isfinite(width)]
    if not finite:
        fallback_s0 = float(np.clip(desired / max(float(gripper.get("max_opening_width", 0.08)), 1e-6), 0.0, 1.0))
        return fallback_s0, _fallback_opening_width(np.array([0, 0, 0, 0, 0, 0, fallback_s0], dtype=float), gripper), {"source": "linear_config_fallback"}

    preference = str(
        gripper.get("s0_prefer_open_branch", gripper.get("s0_branch_preference", ""))
    ).strip().lower()
    split = float(gripper.get("s0_branch_split", 0.5))

    feasible = [(s0, width) for s0, width in finite if width + 1e-5 >= desired]
    if feasible:
        preferred = feasible
        if preference in {"high", "open", "larger", "max"}:
            high_branch = [(s0, width) for s0, width in feasible if s0 >= split]
            if high_branch:
                preferred = high_branch
        elif preference in {"low", "closed", "smaller", "min"}:
            low_branch = [(s0, width) for s0, width in feasible if s0 <= split]
            if low_branch:
                preferred = low_branch
        # Prefer the closest feasible opening; break ties toward the configured branch direction.
        tie_direction = -1.0 if preference in {"high", "open", "larger", "max"} else 1.0
        s0, width = min(preferred, key=lambda item: (abs(item[1] - desired), tie_direction * item[0]))
    else:
        # Desired opening exceeds calibrated range; use the widest available opening.
        s0, width = max(finite, key=lambda item: item[1])
    return float(s0), float(width), {
        "source": "sampled_urdf_opening_curve" if urdf_path else "sampled_linear_fallback",
        "desired_opening_m": desired,
        "selected_s0": float(s0),
        "selected_opening_m": float(width),
        "s0_branch_preference": preference or None,
        "max_sampled_opening_m": max(width for _, width in finite),
        "min_sampled_opening_m": min(width for _, width in finite),
    }


def _fallback_opening_width(state: np.ndarray, gripper: dict[str, Any]) -> float:
    max_opening = max(float(gripper.get("max_opening_width", 0.08)), 1e-6)
    s0 = float(np.clip(state[6] if len(state) > 6 else 0.5, 0.0, 1.0))
    return float(np.clip(s0 * max_opening, 0.0, max_opening))
