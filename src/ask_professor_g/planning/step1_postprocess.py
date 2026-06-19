from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def postprocess_step1(
    *,
    input_path: str | Path,
    output_path: str | Path,
    gripper_name: str,
    geometry_path: str | Path | None = None,
) -> dict[str, Any]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    grasps = data.get("grasps", data if isinstance(data, list) else [])
    geometry = None
    if geometry_path and Path(geometry_path).exists():
        geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    if geometry:
        grasps = _align_grasps_to_geometry(grasps, geometry)
    if (not grasps) and geometry:
        grasps = _fallback_grasps_from_geometry(geometry)
    processed = {
        "object_perception": data.get("object_perception", {}),
        "gripper": gripper_name,
        "geometry_source": str(geometry_path) if geometry_path else None,
        "grasps": [],
    }
    for idx, grasp in enumerate(grasps):
        item = dict(grasp)
        item.setdefault("type", f"grasp_{idx}")
        item.setdefault("category", "candidate")
        item.setdefault("grasp_mode", "pinch")
        item.setdefault("synergy_config", {"s0": [0.0, 0.1], "s1": [0.0, 0.0], "s2": [0.0, 0.0]})
        item.setdefault(
            "wrist_pose_relative",
            {
                "pos_xyz_m": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
                "orn_rpy_deg": {"r": [0.0, 0.0], "p": [0.0, 0.0], "y": [0.0, 0.0]},
            },
        )
        processed["grasps"].append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    return processed


def _align_grasps_to_geometry(grasps: list[dict[str, Any]], geometry: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = geometry.get("selected_strategies", [])
    if not grasps or not strategies:
        return grasps

    aligned: list[dict[str, Any]] = []
    used_strategy_names: set[str] = set()
    for idx, grasp in enumerate(grasps):
        strategy = _match_strategy(grasp, strategies, fallback_index=idx)
        if not strategy or not strategy.get("adjusted_3d_point"):
            aligned.append(grasp)
            continue

        item = dict(grasp)
        point = [float(v) for v in strategy["adjusted_3d_point"][:3]]
        width = float(strategy.get("measured_width") or 0.025)
        is_primary = strategy.get("display_id") == 1 or str(strategy.get("priority", "")).lower() == "high"
        category = str(item.get("category", "")).lower()
        item["category"] = category if category in {"primary", "secondary"} else ("primary" if is_primary else "secondary")
        item["source_stage0_strategy"] = strategy.get("strategy")
        item["source_3d_point"] = point
        item["source_measured_width"] = width
        item["target_part"] = strategy.get("target_part")
        item["measured_normal"] = strategy.get("measured_normal")
        item["closing_direction"] = strategy.get("closing_direction")
        item["approach_direction"] = strategy.get("approach_direction")

        margin = 0.008 if item["category"] == "primary" else 0.012
        item["wrist_pose_relative"] = {
            "pos_xyz_m": {
                "x": [point[0] - margin, point[0] + margin],
                "y": [point[1] - margin, point[1] + margin],
                "z": [point[2] - margin, point[2] + margin],
            },
            "orn_rpy_deg": _orientation_ranges_from_strategy(strategy),
        }
        s0_center = max(0.04, min(0.95, width / 0.11))
        item["synergy_config"] = {
            "s0": [max(0.0, s0_center - 0.04), min(1.0, s0_center + 0.08)],
            "s1": item.get("synergy_config", {}).get("s1", [0.0, 0.0]),
            "s2": item.get("synergy_config", {}).get("s2", [0.0, 0.0]),
        }
        used_strategy_names.add(str(strategy.get("strategy")))
        aligned.append(item)

    for strategy in strategies:
        name = str(strategy.get("strategy"))
        if name in used_strategy_names:
            continue
        fallback = _fallback_grasps_from_geometry({"selected_strategies": [strategy]})
        if fallback:
            aligned.append(fallback[0])
    return aligned


def _match_strategy(grasp: dict[str, Any], strategies: list[dict[str, Any]], fallback_index: int) -> dict[str, Any] | None:
    text = " ".join(str(grasp.get(key, "")) for key in ["type", "source_stage0_strategy", "target_part"]).lower()
    for strategy in strategies:
        name = str(strategy.get("strategy", "")).lower()
        target = str(strategy.get("target_part", "")).lower()
        if (name and name in text) or (target and target in text):
            return strategy
    if fallback_index < len(strategies):
        return strategies[fallback_index]
    return strategies[0] if strategies else None


def _orientation_ranges_from_strategy(strategy: dict[str, Any]) -> dict[str, list[float]]:
    approach = str(strategy.get("approach_direction", "")).lower()
    if "top" in approach:
        return {"r": [160.0, 200.0], "p": [-15.0, 15.0], "y": [-35.0, 35.0]}
    if "sidex" in approach or "sidenegx" in approach:
        return {"r": [80.0, 110.0], "p": [-10.0, 10.0], "y": [-20.0, 20.0]}
    if "sidey" in approach or "sidenegy" in approach:
        return {"r": [80.0, 110.0], "p": [-10.0, 10.0], "y": [70.0, 110.0]}
    return {"r": [70.0, 110.0], "p": [-15.0, 15.0], "y": [-35.0, 35.0]}


def _fallback_grasps_from_geometry(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    grasps: list[dict[str, Any]] = []
    for strategy in geometry.get("selected_strategies", [])[:2]:
        point = strategy.get("adjusted_3d_point")
        if not point:
            continue
        width = float(strategy.get("measured_width") or 0.02)
        s0_center = max(0.02, min(1.0, width / 0.11))
        for variation in range(2):
            delta = 0.004 * variation
            grasps.append(
                {
                    "type": f"{strategy.get('strategy', 'validated')}_{variation + 1}",
                    "category": "primary" if strategy.get("display_id") == 1 else "secondary",
                    "grasp_mode": "pinch",
                    "synergy_config": {
                        "s0": [max(0.0, s0_center - 0.05), min(1.0, s0_center + 0.08)],
                        "s1": [0.0, 0.0],
                        "s2": [0.0, 0.0],
                    },
                    "wrist_pose_relative": {
                        "pos_xyz_m": {
                            "x": [point[0] - 0.006 - delta, point[0] + 0.006 + delta],
                            "y": [point[1] - 0.006 - delta, point[1] + 0.006 + delta],
                            "z": [point[2] - 0.006, point[2] + 0.010 + delta],
                        },
                        "orn_rpy_deg": {
                            "r": [70.0, 110.0],
                            "p": [-15.0, 15.0],
                            "y": [-30.0, 30.0],
                        },
                    },
                    "source_stage0_strategy": strategy.get("strategy"),
                    "source_3d_point": point,
                    "source_measured_width": width,
                }
            )
    return grasps
