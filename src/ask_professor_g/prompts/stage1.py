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
    gripper_axis_context = {
        "local_closing_axis": gripper.get("closing_direction", "+Y"),
        "local_palm_normal": gripper.get("palm_normal", "+Z"),
        "tcp_offset": gripper.get("tcp_offset", gripper.get("tcp_wrap", [0.0, 0.0, 0.0])),
        "max_opening_width": gripper.get("max_opening_width"),
        "rule": "Stage 0 closing_direction is a world-frame direction. The postprocessor aligns this gripper local_closing_axis to that world direction and aligns the TCP/base-to-contact axis toward the object surface.",
    }
    context = {
        "gripper": gripper,
        "gripper_axis_context": gripper_axis_context,
        "object": obj,
        "geometry_results": geometry_results,
    }
    selected_count = len(geometry_results.get("selected_strategies", []))
    payload = {
        "system": (
            "You are Prof. G, generating Stage 1 pose-synergy search regions for CEM. Return valid JSON only. "
            "Use only Stage 0 strategies that passed RGB-D 3D geometry verification. Do not invent new target parts. "
            "The field name wrist_pose_relative is legacy: in this public pipeline pos_xyz_m is a TCP/contact-center "
            "search box, not the gripper base pose. Downstream code will recover the real gripper base from TCP offset."
        ),
        "user": (
            "Generate grasp search boxes from the verified 3D candidates. "
            "Important: pos_xyz_m should be centered on the validated CONTACT POINT / TCP search center with small variations, "
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
            "- Treat adjusted_3d_point as the measured surface contact anchor. pos_xyz_m is a TCP/contact-center range, not the wrist/base link.\n"
            "- If measured_normal is available, keep the TCP/contact center on the outside of the surface: shift 0.003-0.008 m along measured_normal, never inward.\n"
            "- Keep pos_xyz_m variations narrow: about +/-0.004-0.008 m around the validated point or outward clearance point. Avoid ranges that include the object interior.\n"
            "- Use measured_normal, measured_axis, approach_direction, and closing_direction to choose plausible RPY ranges; the postprocessor will recompute a geometry-aligned center.\n"
            "- For side grasps, do not place the palm/base through the object body. The TCP should be near the surface and the base should remain outside along the approach side.\n"
            "- For handles, prefer the exposed middle span. Avoid handle roots unless the geometry result shows clear outside access for the palm and both fingers.\n"
            "- Set s0/opening from measured_width plus a safety margin of 3-8 mm so the object fits between the fingers before closing.\n"
            "- Keep secondary synergies near 0 unless needed by the gripper description.\n"
            "- Prefer High-priority functional selected_strategies unless their geometry is invalid, too wide, or collision-prone.\n"
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
