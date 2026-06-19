from __future__ import annotations


def summarize_stage1(stage1: dict) -> str:
    return f"{len(stage1.get('grasps', []))} grasp strategies"

