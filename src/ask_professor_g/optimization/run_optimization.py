from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.probing import load_point_cloud
from .cem import CEMOptimizer
from .loss_loader import load_loss_function


def _to_range(value: Any) -> list[float]:
    if isinstance(value, list):
        if len(value) == 1:
            return [float(value[0]), float(value[0])]
        return [float(value[0]), float(value[1])]
    return [float(value), float(value)]


def _expand_if_singleton(bounds: list[float], delta: float) -> list[float]:
    if abs(bounds[0] - bounds[1]) < 1e-12:
        return [bounds[0] - delta, bounds[1] + delta]
    return bounds


def _bounded_center(bounds: list[float], *, default: float = 0.0) -> float:
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError):
        return float(default)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float(default)
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip((lo + hi) / 2.0, lo, hi))


def parse_grasp_state(grasp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = grasp["wrist_pose_relative"]["pos_xyz_m"]
    rpy = grasp["wrist_pose_relative"]["orn_rpy_deg"]
    syn = grasp.get("synergy_config", {})

    pos_ranges = [
        _expand_if_singleton(_to_range(pos["x"]), 0.002),
        _expand_if_singleton(_to_range(pos["y"]), 0.002),
        _expand_if_singleton(_to_range(pos["z"]), 0.005),
    ]
    rpy_ranges = [
        np.deg2rad(_expand_if_singleton(_to_range(rpy["r"]), 10.0)),
        np.deg2rad(_expand_if_singleton(_to_range(rpy["p"]), 10.0)),
        np.deg2rad(_expand_if_singleton(_to_range(rpy["y"]), 10.0)),
    ]
    synergy_ranges = [
        _expand_if_singleton(_to_range(syn.get("s0", [0.0, 0.0])), 0.02),
        _expand_if_singleton(_to_range(syn.get("s1", [0.0, 0.0])), 0.02),
        _expand_if_singleton(_to_range(syn.get("s2", [0.0, 0.0])), 0.02),
    ]

    lower = np.array(
        [rng[0] for rng in pos_ranges]
        + [float(rng[0]) for rng in rpy_ranges]
        + [rng[0] for rng in synergy_ranges],
        dtype=np.float64,
    )
    upper = np.array(
        [rng[1] for rng in pos_ranges]
        + [float(rng[1]) for rng in rpy_ranges]
        + [rng[1] for rng in synergy_ranges],
        dtype=np.float64,
    )
    mean = (lower + upper) / 2.0
    std = np.maximum((upper - lower) / 2.0, 1e-4)
    return mean, std, lower, upper


def apply_fixed_synergy_center(state: np.ndarray, grasp: dict[str, Any]) -> np.ndarray:
    """Keep gripper synergy values at Stage 1 calibrated centers.

    Stage 2 losses receive only a 4x4 pose matrix, so CEM cannot observe s0/s1/s2.
    Fixing these dimensions avoids random drift toward visually too-small openings.
    """
    result = np.asarray(state, dtype=np.float64).copy()
    if result.size < 9:
        return result
    syn = grasp.get("synergy_config", {})
    result[6] = _bounded_center(_to_range(syn.get("s0", [result[6], result[6]])), default=float(result[6]))
    result[7] = _bounded_center(_to_range(syn.get("s1", [result[7], result[7]])), default=float(result[7]))
    result[8] = _bounded_center(_to_range(syn.get("s2", [result[8], result[8]])), default=float(result[8]))
    return result


def semantic_priority_penalty(grasp: dict[str, Any]) -> float:
    category = str(grasp.get("category", "")).lower()
    if category == "primary":
        category_penalty = 0.0
    elif category == "secondary":
        category_penalty = 0.35
    else:
        category_penalty = 0.2

    text = " ".join(
        str(grasp.get(key, "")).lower()
        for key in ["type", "source_stage0_strategy", "target_part", "category"]
    )
    fragile_terms = ["shade", "rim", "top cap", "decorative", "fragile"]
    robust_terms = ["stem", "handle", "grip", "base support", "functional"]

    def has_term(term: str) -> bool:
        if " " in term:
            return term in text
        return re.search(rf"\b{re.escape(term)}\b", text) is not None

    fragile_penalty = 0.25 if any(has_term(term) for term in fragile_terms) else 0.0
    robust_bonus = -0.10 if any(has_term(term) for term in robust_terms) else 0.0
    if category == "primary" and fragile_penalty == 0.0 and robust_bonus < 0.0:
        return 0.0
    return max(0.0, category_penalty + fragile_penalty + robust_bonus)


def run_cem_optimization(
    *,
    stage1_processed_path: str | Path,
    step2_loss_path: str | Path,
    point_cloud_path: str | Path,
    output_path: str | Path,
    object_name: str,
    gripper_name: str,
    top_k: int = 1,
    cem_settings: dict[str, Any] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    stage1 = json.loads(Path(stage1_processed_path).read_text(encoding="utf-8"))
    point_cloud = load_point_cloud(point_cloud_path)
    loss_func = load_loss_function(step2_loss_path)
    settings = cem_settings or {}
    optimizer = CEMOptimizer(
        num_samples=int(settings.get("num_samples", 128)),
        num_elites=int(settings.get("num_elites", 16)),
        max_iterations=int(settings.get("max_iterations", 5)),
        seed=seed,
    )

    optimized: list[dict[str, Any]] = []
    history: dict[str, list[float]] = {}
    for idx, grasp in enumerate(stage1.get("grasps", [])):
        mean, std, lower, upper = parse_grasp_state(grasp)
        state, raw_loss, cem_history = optimizer.optimize(
            mean,
            std,
            point_cloud,
            loss_func,
            clamp_min=lower,
            clamp_max=upper,
        )
        state = apply_fixed_synergy_center(state, grasp)
        name = grasp.get("type", f"grasp_{idx}")
        sem_penalty = semantic_priority_penalty(grasp)
        adjusted_loss = raw_loss + sem_penalty
        optimized.append(
            {
                "type": name,
                "loss": adjusted_loss,
                "raw_loss": raw_loss,
                "semantic_penalty": sem_penalty,
                "result": state.tolist(),
                "cem_history": cem_history,
                "category": grasp.get("category"),
            }
        )
        history[name] = cem_history

    optimized.sort(key=lambda item: item["loss"])
    payload = {
        "object": object_name,
        "gripper": gripper_name,
        "grasps": optimized[:top_k],
        "total_optimized": len(optimized),
        "top_k": top_k,
        "optimization_history": history,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
