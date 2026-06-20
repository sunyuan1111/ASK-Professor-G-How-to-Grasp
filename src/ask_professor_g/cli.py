from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import load_config
from .data_registry import DataRegistry
from .geometry.probing import load_point_cloud, simple_geometry_probe
from .llm.base import LLMClient, build_llm_client
from .llm.parser import extract_python
from .optimization.loss_loader import validate_loss_function
from .optimization.run_optimization import run_cem_optimization
from .planning.step1_postprocess import postprocess_step1
from .prompts.stage0 import build_stage0_prompt
from .prompts.stage1 import build_stage1_prompt
from .prompts.stage2 import build_stage2_prompt, build_stage2_repair_prompt
from .rendering.grasp_point_render import render_grasp_points
from .rendering.pyrender_engine import render_observation
from .schemas import RunContext
from .visualization.final_grasp import (
    draw_final_grasp_diagnostics,
    render_final_grasp,
    render_final_grasp_real_views,
)
from .visualization.plots import draw_optimization_overview, draw_stage0_geometry_overview
from .visualization.obj_export import export_optimized_grasps_obj, export_stage0_points_obj
from .visualization.evidence import draw_geometry_audit_on_image, draw_stage0_points_on_image


ALL_STAGES = ["render", "stage0", "geometry", "stage1", "stage2", "optimize"]


def _parse_stages(value: str | None) -> list[str]:
    if not value:
        return ALL_STAGES.copy()
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in ALL_STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    return stages


def _copy_cached(example_dir: Path, name: str, run_dir: Path) -> Path:
    src = example_dir / name
    dst = run_dir / name
    if not src.exists():
        raise FileNotFoundError(f"Cached example missing: {src}")
    shutil.copy2(src, dst)
    return dst


def _default_run_dir(output_root: Path, gripper: str, obj: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{stamp}_{gripper}_{obj}"


def _render_stage(context: RunContext, obj: Any) -> dict:
    result = render_observation(obj.mesh_path, obj.point_cloud_path, output_dir=context.render_dir)
    (context.render_dir / "render_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _target_from_geometry_item(item: dict[str, Any]) -> dict[str, Any] | None:
    point = item.get("adjusted_3d_point")
    if not point:
        return None
    category_weight = 0.0 if item.get("display_id") == 1 or str(item.get("priority", "")).lower() == "high" else 0.35
    normal = item.get("measured_normal")
    width = float(item.get("measured_width") or 0.025)
    clearance = float(np.clip(0.005 + (0.0015 if width > 0.04 else 0.0), 0.004, 0.008))
    return {
        "name": item.get("strategy", "validated"),
        "point": [float(v) for v in point[:3]],
        "clearance_point": _clearance_point(point, normal, clearance),
        "normal": _finite_vector_or_none(normal),
        "closing_direction": _finite_vector_or_none(item.get("closing_direction")),
        "local_tcp_axis": [0.0, 0.0, 1.0],
        "local_closing_axis": [0.0, 1.0, 0.0],
        "clearance": clearance,
        "width": width,
        "category_weight": float(category_weight),
    }


def _clearance_point(point: Any, normal: Any, clearance: float) -> list[float]:
    point_arr = np.asarray(point[:3], dtype=float)
    normal_vec = _finite_vector_or_none(normal)
    if normal_vec is None:
        return point_arr.tolist()
    normal_arr = np.asarray(normal_vec, dtype=float)
    return (point_arr + normal_arr * float(clearance)).tolist()


def _finite_vector_or_none(value: Any) -> list[float] | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return (vector / norm).tolist()


def _write_cached_loss_from_geometry(geometry_path: Path, output_path: Path) -> None:
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    targets = []
    for item in geometry.get("selected_strategies", [])[:4]:
        target = _target_from_geometry_item(item)
        if target is not None:
            targets.append(target)
    if not targets:
        targets = [{"name": "object_center", "point": [0.0, 0.0, 0.06], "clearance_point": [0.0, 0.0, 0.06], "normal": None, "clearance": 0.006, "width": 0.02, "category_weight": 0.0}]
    _write_loss_code(targets, output_path)


def _write_fallback_loss_from_stage1(stage1_processed_path: Path, output_path: Path) -> None:
    stage1 = json.loads(stage1_processed_path.read_text(encoding="utf-8"))
    axes = stage1.get("gripper_axes", {})
    local_tcp_axis = _finite_vector_or_none(axes.get("local_tcp_axis")) or [0.0, 0.0, 1.0]
    local_closing_axis = _finite_vector_or_none(axes.get("local_closing_axis")) or [0.0, 1.0, 0.0]
    targets = []
    for grasp in stage1.get("grasps", [])[:6]:
        point = grasp.get("source_3d_point")
        if not point:
            pos = grasp.get("wrist_pose_relative", {}).get("pos_xyz_m", {})
            if not pos:
                continue
            point = [
                float(pos.get("x", [0.0, 0.0])[0] + pos.get("x", [0.0, 0.0])[1]) / 2.0,
                float(pos.get("y", [0.0, 0.0])[0] + pos.get("y", [0.0, 0.0])[1]) / 2.0,
                float(pos.get("z", [0.0, 0.0])[0] + pos.get("z", [0.0, 0.0])[1]) / 2.0,
            ]
        clearance_point = grasp.get("source_clearance_point") or point
        normal = grasp.get("measured_normal_used") or grasp.get("measured_normal")
        clearance = float(grasp.get("surface_clearance_m") or 0.006)
        category = str(grasp.get("category", "")).lower()
        category_weight = 0.0 if category == "primary" else 0.35
        text = " ".join(str(grasp.get(key, "")).lower() for key in ["type", "target_part", "source_stage0_strategy"])
        if any(term in text for term in ["stem", "handle", "shaft"]):
            category_weight -= 0.08
        if any(term in text for term in ["shade", "fragile", "decorative"]):
            category_weight += 0.25
        targets.append(
            {
                "name": grasp.get("type", "candidate"),
                "point": [float(v) for v in point[:3]],
                "clearance_point": [float(v) for v in clearance_point[:3]],
                "normal": _finite_vector_or_none(normal),
                "closing_direction": _finite_vector_or_none(grasp.get("closing_direction_used") or grasp.get("closing_direction")),
                "local_tcp_axis": local_tcp_axis,
                "local_closing_axis": local_closing_axis,
                "clearance": clearance,
                "width": float(grasp.get("source_measured_width") or 0.02),
                "opening_margin": float(grasp.get("opening_safety_margin_m") or 0.004),
                "category_weight": float(max(0.0, category_weight)),
            }
        )
    if not targets:
        targets = [{"name": "object_center", "point": [0.0, 0.0, 0.06], "clearance_point": [0.0, 0.0, 0.06], "normal": None, "clearance": 0.006, "width": 0.02, "category_weight": 0.0}]
    _write_loss_code(targets, output_path)


def _write_loss_code(targets: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "import numpy as np",
        "",
        f"TARGETS = {repr(targets)}",
        "",
        "",
        "def _safe_points(point_cloud):",
        "    points = np.asarray(point_cloud[:, :3], dtype=float)",
        "    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.all(np.isfinite(points)):",
        "        return None",
        "    return points",
        "",
        "",
        "def calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float:",
        "    if pose_mat.shape != (4, 4) or not np.all(np.isfinite(pose_mat)):",
        "        return 1e6",
        "    position = np.asarray(pose_mat[:3, 3], dtype=float)",
        "    points = _safe_points(point_cloud)",
        "    if points is None:",
        "        return 1e6",
        "    distances = np.linalg.norm(points - position[None, :], axis=1)",
        "    nearest_dist = float(np.min(distances))",
        "    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())",
        "    target_terms = []",
        "    normal_terms = []",
        "    orientation_terms = []",
        "    for target in TARGETS:",
        "        raw_point = np.asarray(target.get('point', target.get('clearance_point')), dtype=float)",
        "        clearance_point = np.asarray(target.get('clearance_point', target.get('point')), dtype=float)",
        "        normal_value = target.get('normal')",
        "        semantic = float(target.get('category_weight', 0.0))",
        "        target_terms.append(float(np.linalg.norm(position - clearance_point)) + semantic)",
        "        if normal_value is not None:",
        "            normal = np.asarray(normal_value, dtype=float)",
        "            normal = normal / max(float(np.linalg.norm(normal)), 1e-9)",
        "            required = float(target.get('clearance', 0.006))",
        "            signed = float(np.dot(position - raw_point, normal))",
        "            normal_terms.append(max(0.0, required - signed) ** 2 * 90.0)",
        "            normal_terms.append(abs(signed - required) * 0.45)",
        "            local_tcp = np.asarray(target.get('local_tcp_axis', [0.0, 0.0, 1.0]), dtype=float)",
        "            local_tcp = local_tcp / max(float(np.linalg.norm(local_tcp)), 1e-9)",
        "            tcp_axis = np.asarray(pose_mat[:3, :3] @ local_tcp, dtype=float)",
        "            tcp_axis = tcp_axis / max(float(np.linalg.norm(tcp_axis)), 1e-9)",
        "            orientation_terms.append(max(0.0, float(np.dot(tcp_axis, normal)) + 0.1) * 0.12)",
        "        closing_value = target.get('closing_direction')",
        "        if closing_value is not None:",
        "            local_closing = np.asarray(target.get('local_closing_axis', [0.0, 1.0, 0.0]), dtype=float)",
        "            local_closing = local_closing / max(float(np.linalg.norm(local_closing)), 1e-9)",
        "            closing_axis = np.asarray(pose_mat[:3, :3] @ local_closing, dtype=float)",
        "            closing_axis = closing_axis / max(float(np.linalg.norm(closing_axis)), 1e-9)",
        "            desired = np.asarray(closing_value, dtype=float)",
        "            desired = desired / max(float(np.linalg.norm(desired)), 1e-9)",
        "            if normal_value is not None:",
        "                desired = desired - float(np.dot(desired, normal)) * normal",
        "                desired = desired / max(float(np.linalg.norm(desired)), 1e-9)",
        "            orientation_terms.append(max(0.0, 0.92 - abs(float(np.dot(closing_axis, desired)))) * 0.45)",
        "    semantic_target = min(target_terms) if target_terms else 1.0",
        "    normal_penalty = min(normal_terms) if normal_terms else 0.0",
        "    orientation_penalty = min(orientation_terms) if orientation_terms else 0.0",
        "    contact_term = abs(nearest_dist - 0.006)",
        "    too_close_penalty = max(0.0, 0.0025 - nearest_dist) ** 2 * 160.0",
        "    too_far_penalty = max(0.0, nearest_dist - 0.018) * 0.5",
        "    height_penalty = max(0.0, z_min - position[2]) + max(0.0, position[2] - z_max)",
        "    radial_penalty = max(0.0, float(np.linalg.norm(position[:2])) - 0.08) * 0.15",
        "    loss = semantic_target + 0.55 * contact_term + too_close_penalty + too_far_penalty + normal_penalty + orientation_penalty",
        "    loss += 0.5 * height_penalty + radial_penalty",
        "    if not np.isfinite(loss):",
        "        return 1e6",
        "    return float(loss)",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _stage2_validation_cloud(point_cloud_path: str | Path, max_points: int = 512) -> np.ndarray:
    points = load_point_cloud(point_cloud_path)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return points


def _format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _run_stage2_with_repair(
    *,
    client: LLMClient,
    gripper: dict[str, Any],
    stage1_processed_path: Path,
    point_cloud_path: str | Path,
    run_dir: Path,
    prompts_dir: Path,
    step2_path: Path,
    max_attempts: int = 2,
) -> dict[str, Any]:
    stage1_processed = json.loads(stage1_processed_path.read_text(encoding="utf-8"))
    system, user = build_stage2_prompt(
        gripper=gripper,
        stage1_processed=stage1_processed,
        output_path=prompts_dir / "stage2.txt",
    )
    validation_cloud = _stage2_validation_cloud(point_cloud_path)
    attempts: list[dict[str, Any]] = []
    current_system, current_user = system, user
    last_error = ""
    attempt_count = max(1, max_attempts)

    for attempt in range(1, attempt_count + 1):
        prompt_path = prompts_dir / ("stage2.txt" if attempt == 1 else f"stage2_repair_{attempt - 1}.txt")
        raw_path = run_dir / ("step2_loss.raw.txt" if attempt == 1 else f"step2_loss.repair{attempt - 1}.raw.txt")
        candidate_written = False
        invalid_code = ""
        try:
            raw_response = client.generate_text(current_system, current_user)
            raw_path.write_text(raw_response, encoding="utf-8")
            code = extract_python(raw_response)
            step2_path.write_text(code + "\n", encoding="utf-8")
            candidate_written = True
            validate_loss_function(step2_path, point_cloud=validation_cloud)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "loaded_and_smoke_tested",
                    "prompt": str(prompt_path),
                    "raw_response": str(raw_path),
                }
            )
            metadata = {
                "source": "llm",
                "status": "loaded_and_smoke_tested",
                "max_attempts": attempt_count,
                "attempts": attempts,
            }
            (run_dir / "step2_loss_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata
        except Exception as exc:
            last_error = _format_exception(exc)
            invalid_path: Path | None = None
            if candidate_written:
                invalid_path = run_dir / (
                    "step2_loss.invalid.py" if attempt == 1 else f"step2_loss.repair{attempt - 1}.invalid.py"
                )
                shutil.copy2(step2_path, invalid_path)
                invalid_code = step2_path.read_text(encoding="utf-8")
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed_validation_or_request",
                    "prompt": str(prompt_path),
                    "error": last_error,
                    "raw_response": str(raw_path) if raw_path.exists() else None,
                    "invalid_code": str(invalid_path) if invalid_path else None,
                }
            )
            if attempt < attempt_count:
                if invalid_code:
                    current_system, current_user = build_stage2_repair_prompt(
                        original_user_prompt=user,
                        invalid_code=invalid_code[-12000:],
                        validation_error=last_error[-4000:],
                        output_path=prompts_dir / f"stage2_repair_{attempt}.txt",
                    )
                else:
                    current_system, current_user = system, user

    _write_fallback_loss_from_stage1(stage1_processed_path, step2_path)
    validate_loss_function(step2_path, point_cloud=validation_cloud)
    metadata = {
        "source": "fallback_from_stage1",
        "status": "llm_stage2_failed_after_repair_attempts",
        "max_attempts": attempt_count,
        "attempts": attempts,
        "last_error": last_error,
        "fallback": str(step2_path),
    }
    (run_dir / "step2_loss_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def run_pipeline(args: argparse.Namespace) -> Path:
    config = load_config(args.config, gripper=args.gripper, obj=args.object)
    registry = DataRegistry(config)
    gripper = registry.gripper()
    obj = registry.object()
    missing = registry.validate_paths(gripper, obj)
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(
        config.output_root, gripper.key, obj.key
    )
    context = RunContext(
        repo_root=config.repo_root,
        run_dir=run_dir,
        gripper_name=gripper.key,
        object_name=obj.key,
    )
    context.ensure_dirs()
    stages = _parse_stages(args.stages)
    example_dir = Path(args.example).resolve() if args.example else config.repo_root / "examples" / "cached"

    if args.cached:
        _render_stage(context, obj)
        for name in ["stage0_output.json", "stage1_output_processed.json", "step2_loss.py"]:
            _copy_cached(example_dir, name, context.run_dir)
        if "stage1" in stages and not (context.run_dir / "stage1_output.json").exists():
            shutil.copy2(context.run_dir / "stage1_output_processed.json", context.run_dir / "stage1_output.json")

    if "render" in stages and not args.cached:
        _render_stage(context, obj)

    stage0_path = context.run_dir / "stage0_output.json"
    if "stage0" in stages and not args.cached:
        prompt = build_stage0_prompt(gripper.raw, obj.raw, output_path=context.prompts_dir / "stage0.json")
        client = build_llm_client(config.llm)
        stage0 = client.generate_json(
            prompt["system"],
            prompt["user"],
            image_path=str(context.render_dir / "view_front_iso_rgb.png"),
        )
        stage0_path.write_text(json.dumps(stage0, indent=2), encoding="utf-8")

    if stage0_path.exists() and (context.render_dir / "view_front_iso_rgb.png").exists():
        draw_stage0_points_on_image(
            image_path=context.render_dir / "view_front_iso_rgb.png",
            stage0_path=stage0_path,
            output_path=context.visualizations_dir / "stage0_2d_points.png",
        )

    geometry_path = context.run_dir / "geometry_probing_results.json"
    if "geometry" in stages:
        stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
        simple_geometry_probe(
            stage0,
            obj.point_cloud_path,
            output_path=geometry_path,
            camera_path=context.render_dir / "view_front_iso_camera.json",
            depth_path=context.render_dir / "view_front_iso_depth.npy",
            gripper_limits={
                "max_width": gripper.max_opening_width,
                "min_width": gripper.min_opening_width,
            },
        )
        draw_geometry_audit_on_image(
            image_path=context.render_dir / "view_front_iso_rgb.png",
            geometry_path=geometry_path,
            output_path=context.visualizations_dir / "stage0_3d_validation.png",
        )
        render_grasp_points(
            mesh_path=obj.mesh_path,
            geometry_path=geometry_path,
            camera_path=context.render_dir / "view_front_iso_camera.json",
            output_path=context.visualizations_dir / "grasp_points_visualization.png",
        )
        draw_stage0_geometry_overview(
            point_cloud_path=obj.point_cloud_path,
            stage0_path=stage0_path,
            geometry_path=geometry_path,
            output_path=context.visualizations_dir / "stage0_geometry_overview.png",
        )
        export_stage0_points_obj(
            object_mesh_path=obj.mesh_path,
            stage0_path=stage0_path,
            geometry_path=geometry_path,
            output_dir=context.visualizations_dir / "obj_scene",
        )

    stage1_path = context.run_dir / "stage1_output.json"
    stage1_processed_path = context.run_dir / "stage1_output_processed.json"
    if args.cached and geometry_path.exists() and stage1_path.exists():
        postprocess_step1(
            input_path=stage1_path,
            output_path=stage1_processed_path,
            gripper_name=gripper.key,
            gripper_max_opening_width=gripper.max_opening_width,
            geometry_path=geometry_path,
            gripper=gripper.raw,
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )

    if "stage1" in stages and not args.cached:
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        prompt = build_stage1_prompt(
            gripper=gripper.raw,
            obj=obj.raw,
            geometry_results=geometry,
            output_path=context.prompts_dir / "stage1.json",
        )
        client = build_llm_client(config.llm)
        stage1 = client.generate_json(prompt["system"], prompt["user"])
        stage1_path.write_text(json.dumps(stage1, indent=2), encoding="utf-8")
        postprocess_step1(
            input_path=stage1_path,
            output_path=stage1_processed_path,
            gripper_name=gripper.key,
            gripper_max_opening_width=gripper.max_opening_width,
            geometry_path=geometry_path,
            gripper=gripper.raw,
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )

    step2_path = context.run_dir / "step2_loss.py"
    if args.cached and stage1_processed_path.exists():
        _write_fallback_loss_from_stage1(stage1_processed_path, step2_path)
        validate_loss_function(step2_path, point_cloud=_stage2_validation_cloud(obj.point_cloud_path))
    elif args.cached and geometry_path.exists():
        _write_cached_loss_from_geometry(geometry_path, step2_path)
        validate_loss_function(step2_path, point_cloud=_stage2_validation_cloud(obj.point_cloud_path))

    if "stage2" in stages and not args.cached:
        client = build_llm_client(config.llm)
        _run_stage2_with_repair(
            client=client,
            gripper=gripper.raw,
            stage1_processed_path=stage1_processed_path,
            point_cloud_path=obj.point_cloud_path,
            run_dir=context.run_dir,
            prompts_dir=context.prompts_dir,
            step2_path=step2_path,
            max_attempts=args.stage2_max_attempts,
        )

    if "optimize" in stages:
        run_cem_optimization(
            stage1_processed_path=stage1_processed_path,
            step2_loss_path=step2_path,
            point_cloud_path=obj.point_cloud_path,
            output_path=context.run_dir / "step3_output.json",
            object_name=obj.key,
            gripper_name=gripper.key,
            top_k=config.top_k,
            cem_settings=config.cem,
            seed=config.seed,
        )
        draw_optimization_overview(
            point_cloud_path=obj.point_cloud_path,
            stage3_path=context.run_dir / "step3_output.json",
            output_path=context.visualizations_dir / "optimization_overview.png",
        )
        export_optimized_grasps_obj(
            object_mesh_path=obj.mesh_path,
            stage3_path=context.run_dir / "step3_output.json",
            output_dir=context.visualizations_dir / "obj_scene",
            gripper=gripper.raw,
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )
        render_final_grasp(
            object_mesh_path=obj.mesh_path,
            stage3_path=context.run_dir / "step3_output.json",
            camera_path=context.render_dir / "view_front_iso_camera.json",
            gripper=gripper.raw,
            output_path=context.visualizations_dir / "final_grasp_render.png",
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )
        render_final_grasp_real_views(
            object_mesh_path=obj.mesh_path,
            stage3_path=context.run_dir / "step3_output.json",
            gripper=gripper.raw,
            output_path=context.visualizations_dir / "final_grasp_real_views.png",
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )
        draw_final_grasp_diagnostics(
            point_cloud_path=obj.point_cloud_path,
            stage3_path=context.run_dir / "step3_output.json",
            stage1_processed_path=stage1_processed_path,
            gripper=gripper.raw,
            output_path=context.visualizations_dir / "final_grasp_diagnostics.png",
            report_path=context.run_dir / "final_grasp_report.json",
            gripper_urdf_path=gripper.path("urdf"),
            gripper_glpca_path=gripper.path("glpca") if gripper.raw.get("glpca") else None,
        )

    return context.run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASK Professor G grasping pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gripper", default=None)
    parser.add_argument("--object", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stages", default=None, help="Comma-separated stages.")
    parser.add_argument(
        "--stage2-max-attempts",
        type=int,
        default=2,
        help="Maximum Stage 2 LLM generation/repair attempts.",
    )
    parser.add_argument("--cached", action="store_true", help="Use examples/cached instead of online LLM.")
    parser.add_argument("--example", default=None, help="Cached example directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = run_pipeline(args)
    print(f"Run complete: {run_dir}")
