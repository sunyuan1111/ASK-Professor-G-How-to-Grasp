from __future__ import annotations


def summarize_stage0(stage0: dict) -> str:
    return f"{len(stage0.get('proposals', []))} semantic grasp proposals"

