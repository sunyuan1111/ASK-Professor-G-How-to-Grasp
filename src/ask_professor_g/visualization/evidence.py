from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_stage0_points_on_image(
    *,
    image_path: str | Path,
    stage0_path: str | Path,
    output_path: str | Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    data = json.loads(Path(stage0_path).read_text(encoding="utf-8"))
    colors = {"High": (40, 190, 80), "Medium": (245, 150, 30), "Low": (230, 60, 60)}

    draw.rectangle((8, 8, 330, 124), fill=(0, 0, 0))
    draw.text((18, 18), "Stage 0: VLM 2D grasp proposals", fill=(255, 255, 255), font=_font(17))
    draw.text((18, 46), "solid = primary point", fill=(230, 230, 230), font=_font(13))
    draw.text((18, 68), "ring = backup candidate", fill=(230, 230, 230), font=_font(13))
    draw.text((18, 92), "green/orange/red = priority", fill=(230, 230, 230), font=_font(13))

    for proposal in data.get("proposals", []):
        color = colors.get(proposal.get("priority", "Medium"), colors["Medium"])
        prop_id = str(proposal.get("id", "?"))
        primary = None
        for idx, point in enumerate(proposal.get("candidate_points", [])):
            if not isinstance(point, list) or len(point) != 2:
                continue
            x = int(float(point[0]) * (width - 1))
            y = int(float(point[1]) * (height - 1))
            if idx == 0:
                primary = (x, y)
                draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color, outline=(255, 255, 255), width=3)
                draw.rectangle((x + 14, y - 16, x + 58, y + 8), fill=(0, 0, 0))
                draw.text((x + 19, y - 14), f"#{prop_id}", fill=color, font=_font(16))
            else:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
                if primary:
                    draw.line((primary[0], primary[1], x, y), fill=tuple(int(c * 0.65) for c in color), width=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def draw_geometry_audit_on_image(
    *,
    image_path: str | Path,
    geometry_path: str | Path,
    output_path: str | Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    data = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    status_colors = {
        "VALID": (25, 180, 70),
        "TOO_WIDE": (235, 180, 20),
        "TOO_THIN": (235, 130, 30),
        "LOW_CONFIDENCE_PROJECTION": (160, 80, 220),
        "EMPTY_REGION": (150, 150, 150),
        "INVALID_INPUT": (230, 50, 50),
    }

    draw.rectangle((8, 8, 430, 132), fill=(0, 0, 0))
    draw.text((18, 18), "Stage 0 -> 3D geometry validation", fill=(255, 255, 255), font=_font(17))
    summary = data.get("summary", {})
    draw.text(
        (18, 46),
        f"valid={summary.get('valid_grasps', 0)} selected={summary.get('selected_count', 0)} rejected={summary.get('rejected', 0)}",
        fill=(230, 230, 230),
        font=_font(13),
    )
    draw.text((18, 70), "labels show proposal id and measured local width", fill=(230, 230, 230), font=_font(13))
    draw.text((18, 94), "green selected markers become Stage 1 search regions", fill=(230, 230, 230), font=_font(13))

    selected_ids = {item.get("id") for item in data.get("selected_strategies", [])}
    for item in data.get("audit_results", []):
        pixel = item.get("selected_pixel") or {}
        x = int(pixel.get("col", 0))
        y = int(pixel.get("row", 0))
        status = item.get("audit_status", "INVALID_INPUT")
        color = status_colors.get(status, (255, 255, 255))
        radius = 15 if item.get("id") in selected_ids else 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
        if status == "VALID":
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        width = item.get("measured_width")
        width_text = f"{width * 1000:.0f}mm" if isinstance(width, (int, float)) else "n/a"
        label = f"#{item.get('id')} {status} {width_text}"
        text_w = max(130, len(label) * 7)
        draw.rectangle((x + 15, y - 16, x + 15 + text_w, y + 8), fill=(0, 0, 0))
        draw.text((x + 20, y - 14), label, fill=color, font=_font(13))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

