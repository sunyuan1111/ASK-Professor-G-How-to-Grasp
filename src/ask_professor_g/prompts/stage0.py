from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_stage0_prompt(gripper: dict[str, Any], obj: dict[str, Any], *, output_path: str | Path) -> dict[str, Any]:
    context = {
        "gripper": gripper,
        "object": obj,
        "image_coordinate_convention": "candidate_points are normalized [x, y] in [0, 1], origin top-left",
    }
    payload = {
        "system": (
            "You are Prof. G, an expert robotic vision and grasping planner. Return valid JSON only. "
            "Stage 0 must produce semantic 2D grasp target regions from the RGB observation, not final grasps. "
            "These visual proposals are the only semantic seed for the later RGB-D 2D-to-3D verifier."
        ),
        "user": (
            "Analyze the attached RGB observation. The image contains an axis overlay: red=+X, green=+Y, blue=+Z. "
            "Propose visual semantic grasp regions as normalized 2D candidate points. Treat this as the paper's "
            "Object Image step: you are not allowed to use hidden masks, labels, or ground-truth keypoints. "
            "Prefer task-safe functional regions and avoid fragile, dangerous, or semantically forbidden parts.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "object_analysis": "string",\n'
            '  "graspable_parts": ["string"],\n'
            '  "proposals": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "strategy": "string",\n'
            '      "approach_direction": "SideX|SideY|TopDown|Other",\n'
            '      "closing_direction": [1, 0, 0],\n'
            '      "reasoning": "string",\n'
            '      "target_part": "string",\n'
            '      "candidate_points": [[0.5, 0.5]],\n'
            '      "estimated_part_width": "string",\n'
            '      "priority": "High|Medium|Low",\n'
            '      "risk_factors": ["string"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Provide 2-5 distinct semantic strategies.\n"
            "- Provide 3-8 candidate_points per strategy, ordered best-first and slightly spread around the visible target.\n"
            "- candidate_points are normalized [x, y] with origin at the image top-left.\n"
            "- Every candidate point must lie on a visible object surface, not on the background or coordinate axes.\n"
            "- Include at least one robust primary functional grasp if visible, such as a stem, handle, waist, body, or stable base support.\n"
            "- approach_direction must be one of TopDown, SideX, SideNegX, SideY, SideNegY.\n"
            "- closing_direction is a world-frame 3D pinch axis perpendicular to approach_direction.\n"
            "- Do not output wrist pose, RPY, or synergy in Stage 0.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        ),
        "gripper": gripper,
        "object": obj,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
