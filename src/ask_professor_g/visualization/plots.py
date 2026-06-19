from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry.probing import load_point_cloud


def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _project(points: np.ndarray, axes: tuple[int, int], bounds: tuple[np.ndarray, np.ndarray], box):
    mins, maxs = bounds
    x0, y0, x1, y1 = box
    span = np.maximum(maxs - mins, 1e-9)
    xs = (points[:, axes[0]] - mins[axes[0]]) / span[axes[0]]
    ys = (points[:, axes[1]] - mins[axes[1]]) / span[axes[1]]
    px = x0 + xs * (x1 - x0)
    py = y1 - ys * (y1 - y0)
    return np.stack([px, py], axis=1)


def _draw_cloud(draw: ImageDraw.ImageDraw, projected: np.ndarray, color=(185, 190, 197), step: int = 1):
    for x, y in projected[::step]:
        draw.point((float(x), float(y)), fill=color)


def _draw_panel(draw: ImageDraw.ImageDraw, box, title: str):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(35, 41, 51), width=2)
    draw.text((x0 + 8, y0 + 8), title, fill=(20, 24, 31), font=_font(16))


def _candidate_point_from_normalized(candidate: Iterable[float], points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    x, y = candidate
    return np.array([mins[0] + float(x) * span[0], mins[1] + float(y) * span[1], points[:, 2].mean()])


def draw_stage0_geometry_overview(
    *,
    point_cloud_path: str | Path,
    stage0_path: str | Path,
    geometry_path: str | Path,
    output_path: str | Path,
) -> Path:
    points = load_point_cloud(point_cloud_path)
    stage0 = json.loads(Path(stage0_path).read_text(encoding="utf-8"))
    geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    image = Image.new("RGB", (1400, 860), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    draw.text((32, 24), "Semantic Grasp Candidates and 3D Geometry Probing", fill=(10, 15, 25), font=title_font)
    draw.text((32, 58), "Object: 3D_Dollhouse_Lamp    Gripper: wsg_50", fill=(70, 78, 92), font=_font(15))

    bounds = (points.min(axis=0), points.max(axis=0))
    xy_box = (48, 100, 660, 760)
    xz_box = (740, 100, 1352, 760)
    _draw_panel(draw, xy_box, "XY projection")
    _draw_panel(draw, xz_box, "XZ projection")
    xy = _project(points, (0, 1), bounds, xy_box)
    xz = _project(points, (0, 2), bounds, xz_box)
    step = max(1, len(points) // 2500)
    _draw_cloud(draw, xy, step=step)
    _draw_cloud(draw, xz, step=step)

    colors = [(220, 38, 38), (37, 99, 235), (22, 163, 74), (202, 138, 4), (147, 51, 234)]
    legend_y = 790
    for pidx, proposal in enumerate(stage0.get("proposals", [])):
        color = colors[pidx % len(colors)]
        label = proposal.get("strategy", f"proposal {pidx + 1}")[:42]
        draw.ellipse((48 + pidx * 250, legend_y, 62 + pidx * 250, legend_y + 14), fill=color)
        draw.text((68 + pidx * 250, legend_y - 2), label, fill=(30, 35, 45), font=_font(13))
        for candidate in proposal.get("candidate_points", []):
            point = _candidate_point_from_normalized(candidate, points)
            xy_p = _project(point[None, :], (0, 1), bounds, xy_box)[0]
            xz_p = _project(point[None, :], (0, 2), bounds, xz_box)[0]
            for px, py in [xy_p, xz_p]:
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline=(255, 255, 255), width=2)

    for candidate in geometry.get("candidates", []):
        point = np.array(candidate["point_3d"], dtype=float)
        xy_p = _project(point[None, :], (0, 1), bounds, xy_box)[0]
        xz_p = _project(point[None, :], (0, 2), bounds, xz_box)[0]
        for px, py in [xy_p, xz_p]:
            draw.rectangle((px - 3, py - 3, px + 3, py + 3), outline=(0, 0, 0), width=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def draw_optimization_overview(
    *,
    point_cloud_path: str | Path,
    stage3_path: str | Path,
    output_path: str | Path,
) -> Path:
    points = load_point_cloud(point_cloud_path)
    stage3 = json.loads(Path(stage3_path).read_text(encoding="utf-8"))
    image = Image.new("RGB", (1400, 900), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    draw.text((32, 24), "CEM Optimized Grasps", fill=(10, 15, 25), font=_font(24))
    draw.text((32, 58), f"Top {len(stage3.get('grasps', []))} grasps sorted by generated loss", fill=(70, 78, 92), font=_font(15))

    bounds = (points.min(axis=0), points.max(axis=0))
    xy_box = (48, 104, 660, 720)
    xz_box = (740, 104, 1352, 720)
    _draw_panel(draw, xy_box, "XY projection with optimized wrist positions")
    _draw_panel(draw, xz_box, "XZ projection with optimized wrist positions")
    step = max(1, len(points) // 2500)
    _draw_cloud(draw, _project(points, (0, 1), bounds, xy_box), step=step)
    _draw_cloud(draw, _project(points, (0, 2), bounds, xz_box), step=step)

    colors = [(220, 38, 38), (37, 99, 235), (22, 163, 74), (202, 138, 4), (147, 51, 234)]
    rows_y = 748
    draw.text((48, rows_y), "Rank", fill=(30, 35, 45), font=_font(14))
    draw.text((120, rows_y), "Type", fill=(30, 35, 45), font=_font(14))
    draw.text((520, rows_y), "Loss", fill=(30, 35, 45), font=_font(14))
    draw.text((650, rows_y), "State [x, y, z, roll, pitch, yaw, s0, s1, s2]", fill=(30, 35, 45), font=_font(14))

    for idx, grasp in enumerate(stage3.get("grasps", [])):
        color = colors[idx % len(colors)]
        state = np.array(grasp["result"], dtype=float)
        point = state[:3]
        xy_p = _project(point[None, :], (0, 1), bounds, xy_box)[0]
        xz_p = _project(point[None, :], (0, 2), bounds, xz_box)[0]
        radius = 10 if idx == 0 else 7
        for px, py in [xy_p, xz_p]:
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=(255, 255, 255), width=2)
            draw.text((px + 10, py - 10), str(idx + 1), fill=color, font=_font(15))
        y = rows_y + 28 + idx * 27
        draw.ellipse((50, y + 4, 62, y + 16), fill=color)
        draw.text((72, y), str(idx + 1), fill=(30, 35, 45), font=_font(13))
        draw.text((120, y), grasp.get("type", "unknown")[:45], fill=(30, 35, 45), font=_font(13))
        draw.text((520, y), f"{float(grasp.get('loss', 0.0)):.4f}", fill=(30, 35, 45), font=_font(13))
        state_text = ", ".join(f"{v:.3f}" for v in state)
        draw.text((650, y), state_text, fill=(30, 35, 45), font=_font(13))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

