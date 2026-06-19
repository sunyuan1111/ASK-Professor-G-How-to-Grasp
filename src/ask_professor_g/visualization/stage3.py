from __future__ import annotations


def summarize_stage3(stage3: dict) -> str:
    best = stage3.get("grasps", [{}])[0]
    return f"best={best.get('type', 'none')} loss={best.get('loss', 'n/a')}"

