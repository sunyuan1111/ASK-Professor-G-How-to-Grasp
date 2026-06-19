from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .data_registry import DataRegistry
from .geometry.probing import simple_geometry_probe
from .llm.base import build_llm_client
from .llm.parser import extract_python
from .optimization.loss_loader import load_loss_function
from .optimization.run_optimization import run_cem_optimization
from .planning.step1_postprocess import postprocess_step1
from .prompts.stage0 import build_stage0_prompt
from .prompts.stage1 import build_stage1_prompt
from .prompts.stage2 import build_stage2_prompt
from .rendering.grasp_point_render import render_grasp_points
from .rendering.pyrender_engine import render_observation
from .schemas import RunContext
from .visualization.final_grasp import draw_final_grasp_diagnostics, render_final_grasp
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


def _write_cached_loss_from_geometry(geometry_path: Path, output_path: Path) -> None:
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    targets = []
    for item in geometry.get("selected_strategies", [])[:4]:
        point = item.get("adjusted_3d_point")
        if not point:
            continue
        category_weight = 0.0 if item.get("display_id") == 1 or str(item.get("priority", "")).lower() == "high" else 0.35
        targets.append(
            {
                "name": item.get("strategy", "validated"),
                "point": [float(v) for v in point[:3]],
                "category_weight": float(category_weight),
            }
        )
    if not targets:
        targets = [{"name": "object_center", "point": [0.0, 0.0, 0.06], "category_weight": 0.0}]
    lines = [
        "import numpy as np",
        "",
        f"TARGETS = {repr(targets)}",
        "",
        "",
        "def calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float:",
        "    if pose_mat.shape != (4, 4) or not np.all(np.isfinite(pose_mat)):",
        "        return 1e6",
        "    position = np.asarray(pose_mat[:3, 3], dtype=float)",
        "    points = np.asarray(point_cloud[:, :3], dtype=float)",
        "    if len(points) == 0:",
        "        return 1e6",
        "    nearest_dist = float(np.min(np.linalg.norm(points - position[None, :], axis=1)))",
        "    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())",
        "    target_terms = []",
        "    for target in TARGETS:",
        "        target_point = np.asarray(target['point'], dtype=float)",
        "        target_terms.append(float(np.linalg.norm(position - target_point)) + float(target['category_weight']))",
        "    semantic_target = min(target_terms)",
        "    contact_term = abs(nearest_dist - 0.006)",
        "    penetration_penalty = max(0.0, 0.002 - nearest_dist) * 10.0",
        "    height_penalty = max(0.0, z_min - position[2]) + max(0.0, position[2] - z_max)",
        "    radial_penalty = max(0.0, float(np.linalg.norm(position[:2])) - 0.04)",
        "    approach_bonus = -0.015 * abs(float(pose_mat[2, 2]))",
        "    return float(semantic_target + 0.45 * contact_term + 0.4 * penetration_penalty + 0.5 * height_penalty + 0.2 * radial_penalty + approach_bonus)",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
            geometry_path=geometry_path,
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
            geometry_path=geometry_path,
        )

    step2_path = context.run_dir / "step2_loss.py"
    if args.cached and geometry_path.exists():
        _write_cached_loss_from_geometry(geometry_path, step2_path)
        load_loss_function(step2_path)

    if "stage2" in stages and not args.cached:
        stage1_processed = json.loads(stage1_processed_path.read_text(encoding="utf-8"))
        system, user = build_stage2_prompt(
            gripper=gripper.raw,
            stage1_processed=stage1_processed,
            output_path=context.prompts_dir / "stage2.txt",
        )
        client = build_llm_client(config.llm)
        code = extract_python(client.generate_text(system, user))
        step2_path.write_text(code + "\n", encoding="utf-8")
        load_loss_function(step2_path)

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
        )
        render_final_grasp(
            object_mesh_path=obj.mesh_path,
            stage3_path=context.run_dir / "step3_output.json",
            camera_path=context.render_dir / "view_front_iso_camera.json",
            gripper=gripper.raw,
            output_path=context.visualizations_dir / "final_grasp_render.png",
        )
        draw_final_grasp_diagnostics(
            point_cloud_path=obj.point_cloud_path,
            stage3_path=context.run_dir / "step3_output.json",
            stage1_processed_path=stage1_processed_path,
            gripper=gripper.raw,
            output_path=context.visualizations_dir / "final_grasp_diagnostics.png",
            report_path=context.run_dir / "final_grasp_report.json",
        )

    return context.run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASK Professor G grasping pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gripper", default=None)
    parser.add_argument("--object", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stages", default=None, help="Comma-separated stages.")
    parser.add_argument("--cached", action="store_true", help="Use examples/cached instead of online LLM.")
    parser.add_argument("--example", default=None, help="Cached example directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = run_pipeline(args)
    print(f"Run complete: {run_dir}")
