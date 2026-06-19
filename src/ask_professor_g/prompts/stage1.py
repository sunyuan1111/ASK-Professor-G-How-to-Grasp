from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_stage1_prompt(
    *,
    gripper: dict[str, Any],
    obj: dict[str, Any],
    geometry_results: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    context = {
        "gripper": gripper,
        "object": obj,
        "geometry_results": geometry_results,
    }
    selected_count = len(geometry_results.get("selected_strategies", []))
    payload = {
        "system": (
            "You are Prof. G, generating Stage 1 pose-synergy search regions for CEM. Return valid JSON only. "
            "Use only Stage 0 strategies that passed RGB-D 3D geometry verification. Do not invent new target parts."
        ),
        "user": (
            "Generate grasp search boxes from the verified 3D candidates. "
            "Important: pos_xyz_m should be centered on the validated CONTACT POINT with small variations, "
            "not an arbitrary wrist pose. This is the paper's expansion from visual proposals to pose-synergy boxes.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "object_perception": {"what_i_see": "string", "graspable_parts": ["string"]},\n'
            '  "grasps": [\n'
            "    {\n"
            '      "type": "string",\n'
            '      "category": "primary|secondary",\n'
            '      "grasp_mode": "pinch|palm",\n'
            '      "synergy_config": {"s0": [0.0, 0.5], "s1": [0.0, 0.0], "s2": [0.0, 0.0]},\n'
            '      "wrist_pose_relative": {\n'
            '        "pos_xyz_m": {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.02, 0.12]},\n'
            '        "orn_rpy_deg": {"r": [70, 110], "p": [-15, 15], "y": [-30, 30]}\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Generate only variations of selected_strategies from geometry_results.\n"
            "- If one selected strategy exists, generate 3-4 variations. If two or more exist, use the best 1-2 strategies.\n"
            "- Preserve the selected strategy name as the type prefix.\n"
            "- Center pos_xyz_m around adjusted_3d_point with about +/-0.01m variation.\n"
            "- Use measured_normal, measured_axis, approach_direction, and closing_direction to choose plausible RPY ranges.\n"
            "- Keep pose ranges narrow enough for CEM but wide enough to refine contact.\n"
            "- Set s0 from measured_width and gripper opening; keep secondary synergies near 0 unless needed.\n"
            "- Prefer High-priority functional selected_strategies unless their geometry is invalid or too wide.\n"
            "- Output numeric ranges only. No prose outside JSON.\n\n"
            f"Selected validated strategies count: {selected_count}\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        ),
        "gripper": gripper,
        "object": obj,
        "geometry_results": geometry_results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
