from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ask_professor_g.geometry.probing import load_point_cloud  # noqa: E402
from ask_professor_g.optimization.cem import CEMOptimizer  # noqa: E402
from ask_professor_g.optimization.loss_loader import load_loss_function  # noqa: E402
from ask_professor_g.optimization.run_optimization import parse_grasp_state  # noqa: E402


COLORS = [
    (0, 109, 119),
    (191, 63, 47),
    (229, 184, 63),
    (53, 105, 177),
    (88, 80, 141),
    (38, 151, 94),
    (215, 116, 43),
    (70, 91, 107),
    (156, 89, 182),
    (42, 157, 143),
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _short_label(name: str) -> str:
    lower = name.lower()
    if "mid stem" in lower or "vertical shaft" in lower:
        return "mid-stem"
    if "outer base" in lower or "base wall" in lower:
        return "base-wall"
    if "lower-middle" in lower:
        return "lower-stem"
    if "lampshade" in lower:
        return "shade"
    return name.split(".")[0][:18]


def _normalize(history: list[float]) -> list[float]:
    if not history:
        return []
    base = max(float(history[0]), 1e-12)
    return [float(value) / base for value in history]


def _run_trace(
    *,
    grasp: dict[str, Any],
    trace_idx: int,
    point_cloud: np.ndarray,
    loss_func: Any,
    iterations: int,
    samples: int,
    elites: int,
    seed: int,
) -> dict[str, Any]:
    mean, std, lower, upper = parse_grasp_state(grasp)
    optimizer = CEMOptimizer(
        num_samples=samples,
        num_elites=elites,
        max_iterations=iterations,
        seed=seed + trace_idx * 17,
    )
    state, loss, history = optimizer.optimize(
        mean,
        std,
        point_cloud,
        loss_func,
        clamp_min=lower,
        clamp_max=upper,
    )
    if len(history) < iterations:
        history = history + [history[-1]] * (iterations - len(history))
    return {
        "trace": trace_idx + 1,
        "label": _short_label(str(grasp.get("type", f"grasp_{trace_idx}"))),
        "type": grasp.get("type", f"grasp_{trace_idx}"),
        "seed": seed + trace_idx * 17,
        "loss": float(loss),
        "result": np.asarray(state, dtype=float).tolist(),
        "history": [float(value) for value in history[:iterations]],
        "normalized_history": _normalize(history[:iterations]),
    }


def generate_traces(
    *,
    run_dir: Path,
    point_cloud_path: Path,
    iterations: int,
    samples: int,
    elites: int,
    num_traces: int,
    seed: int,
) -> list[dict[str, Any]]:
    stage1_path = run_dir / "stage1_output_processed.json"
    loss_path = run_dir / "step2_loss.py"
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    grasps = stage1.get("grasps", [])
    if not grasps:
        raise ValueError(f"No grasps found in {stage1_path}")

    point_cloud = load_point_cloud(point_cloud_path)
    loss_func = load_loss_function(loss_path)
    traces = []
    for trace_idx in range(num_traces):
        grasp = grasps[trace_idx % len(grasps)]
        traces.append(
            _run_trace(
                grasp=grasp,
                trace_idx=trace_idx,
                point_cloud=point_cloud,
                loss_func=loss_func,
                iterations=iterations,
                samples=samples,
                elites=elites,
                seed=seed,
            )
        )
    return traces


def _value_to_y(value: float, bounds: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    lo, hi = bounds
    _, y0, _, y1 = box
    return y1 - (float(value) - lo) / max(hi - lo, 1e-9) * (y1 - y0)


def _draw_axis(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    y_bounds: tuple[float, float],
    x_ticks: list[tuple[float, str]],
    y_ticks: list[float],
) -> None:
    x0, y0, x1, y1 = box
    grid = (232, 236, 229)
    ink = (35, 41, 51)
    muted = (88, 99, 94)
    for y_value in y_ticks:
        y = _value_to_y(y_value, y_bounds, box)
        draw.line((x0, y, x1, y), fill=grid, width=1)
        draw.text((x0 - 52, y - 9), f"{y_value:.2f}", fill=muted, font=_font(14))
    for x, label in x_ticks:
        draw.line((x, y0, x, y1), fill=(240, 242, 238), width=1)
        draw.text((x - 11, y1 + 16), label, fill=muted, font=_font(14))
    draw.line((x0, y1, x1, y1), fill=ink, width=2)
    draw.line((x0, y0, x0, y1), fill=ink, width=2)


def _draw_boxplot(
    draw: ImageDraw.ImageDraw,
    *,
    traces: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    stages: list[int],
    y_bounds: tuple[float, float],
) -> None:
    x0, y0, x1, y1 = box
    x_positions = np.linspace(x0 + 70, x1 - 45, len(stages))
    _draw_axis(
        draw,
        box,
        y_bounds=y_bounds,
        x_ticks=[(float(x), str(stage)) for x, stage in zip(x_positions, stages)],
        y_ticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    for x, stage in zip(x_positions, stages):
        values = np.asarray([trace["normalized_history"][stage - 1] for trace in traces], dtype=float)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        low, high = float(values.min()), float(values.max())
        y_low = _value_to_y(low, y_bounds, box)
        y_high = _value_to_y(high, y_bounds, box)
        y_q1 = _value_to_y(float(q1), y_bounds, box)
        y_med = _value_to_y(float(med), y_bounds, box)
        y_q3 = _value_to_y(float(q3), y_bounds, box)
        color = COLORS[min(stage // 5, len(COLORS) - 1)]
        draw.line((x, y_high, x, y_low), fill=(70, 78, 92), width=2)
        draw.line((x - 22, y_high, x + 22, y_high), fill=(70, 78, 92), width=2)
        draw.line((x - 22, y_low, x + 22, y_low), fill=(70, 78, 92), width=2)
        draw.rounded_rectangle((x - 36, y_q3, x + 36, y_q1), radius=5, fill=color, outline=(35, 41, 51), width=2)
        draw.line((x - 40, y_med, x + 40, y_med), fill=(255, 253, 246), width=3)

    draw.text(((x0 + x1) // 2 - 95, y1 + 48), "CEM iteration", fill=(88, 99, 94), font=_font(16))


def _draw_convergence(
    draw: ImageDraw.ImageDraw,
    *,
    traces: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    y_bounds: tuple[float, float],
    iterations: int,
) -> None:
    x0, y0, x1, y1 = box
    x_positions = np.linspace(x0, x1, iterations)
    _draw_axis(
        draw,
        box,
        y_bounds=y_bounds,
        x_ticks=[
            (float(x_positions[0]), "1"),
            (float(x_positions[4]), "5"),
            (float(x_positions[9]), "10"),
            (float(x_positions[14]), "15"),
            (float(x_positions[-1]), str(iterations)),
        ],
        y_ticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    for idx, trace in enumerate(traces):
        values = trace["normalized_history"]
        pts = [
            (float(x_positions[i]), _value_to_y(float(values[i]), y_bounds, box))
            for i in range(min(iterations, len(values)))
        ]
        color = COLORS[idx % len(COLORS)]
        width = 4 if idx == 0 else 2
        for p0, p1 in zip(pts, pts[1:]):
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
        for point_idx in [0, 4, 9, 14, 19]:
            if point_idx < len(pts):
                x, y = pts[point_idx]
                radius = 5 if idx == 0 else 3
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.text(((x0 + x1) // 2 - 95, y1 + 48), "CEM iteration", fill=(88, 99, 94), font=_font(16))


def draw_cem_figure(
    *,
    traces: list[dict[str, Any]],
    output_path: Path,
    iterations: int,
) -> Path:
    image = Image.new("RGB", (1800, 980), (255, 253, 246))
    draw = ImageDraw.Draw(image)
    ink = (24, 33, 31)
    muted = (88, 99, 94)
    line = (216, 221, 211)
    panel = (255, 255, 255)

    def panel_box(box: tuple[int, int, int, int]) -> None:
        draw.rounded_rectangle(box, radius=18, fill=panel, outline=line, width=2)

    draw.text((58, 42), "CEM optimization behavior", fill=ink, font=_font(42))
    draw.text(
        (60, 98),
        "Twenty-iteration refinement reduces the generated objective before final grasp selection.",
        fill=muted,
        font=_font(21),
    )

    left_panel = (58, 154, 858, 820)
    right_panel = (904, 154, 1742, 820)
    panel_box(left_panel)
    panel_box(right_panel)
    draw.text((92, 190), "Best objective distribution", fill=ink, font=_font(27))
    draw.text((92, 224), "boxplots at selected CEM stages", fill=muted, font=_font(17))
    draw.text((938, 190), "Convergence curves", fill=ink, font=_font(27))
    draw.text((938, 224), "10 representative traces", fill=muted, font=_font(17))

    y_bounds = (0.0, 1.05)
    _draw_boxplot(
        draw,
        traces=traces,
        box=(150, 300, 792, 704),
        stages=[1, 5, 10, 15, 20],
        y_bounds=y_bounds,
    )
    _draw_convergence(
        draw,
        traces=traces,
        box=(1000, 300, 1680, 704),
        y_bounds=y_bounds,
        iterations=iterations,
    )

    summary = (58, 856, 1742, 930)
    panel_box(summary)
    start_values = np.asarray([trace["history"][0] for trace in traces], dtype=float)
    final_values = np.asarray([trace["history"][-1] for trace in traces], dtype=float)
    normalized_final = np.asarray([trace["normalized_history"][-1] for trace in traces], dtype=float)
    median_drop = float((1.0 - np.median(normalized_final)) * 100.0)
    best_idx = int(np.argmin(final_values))
    items = [
        ("traces", str(len(traces))),
        ("iterations", str(iterations)),
        ("median objective drop", f"{median_drop:.1f}%"),
        ("best final objective", f"{final_values[best_idx]:.6f}"),
        ("best trace", traces[best_idx]["label"]),
    ]
    for idx, (label, value) in enumerate(items):
        x = 92 + idx * 320
        draw.text((x, 878), label, fill=muted, font=_font(15))
        draw.text((x, 902), value, fill=ink if idx != 2 else (38, 151, 94), font=_font(21))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the docs CEM convergence figure.")
    parser.add_argument("--run-dir", default="runs/readme_lamp_wsg50_stem_demo_20260621")
    parser.add_argument("--point-cloud", default="data/objects/3D_Dollhouse_Lamp/points_sampled_1024.xyz")
    parser.add_argument("--output", default="docs/media/lamp_wsg50_readme_clean/cem_refinement.png")
    parser.add_argument("--trace-json", default="docs/media/lamp_wsg50_readme_clean/cem_refinement_traces.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--elites", type=int, default=50)
    parser.add_argument("--traces", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    traces = generate_traces(
        run_dir=(REPO_ROOT / args.run_dir).resolve(),
        point_cloud_path=(REPO_ROOT / args.point_cloud).resolve(),
        iterations=args.iterations,
        samples=args.samples,
        elites=args.elites,
        num_traces=args.traces,
        seed=args.seed,
    )
    output_path = draw_cem_figure(
        traces=traces,
        output_path=(REPO_ROOT / args.output).resolve(),
        iterations=args.iterations,
    )
    trace_path = (REPO_ROOT / args.trace_json).resolve()
    trace_path.write_text(json.dumps({"traces": traces}, indent=2), encoding="utf-8")
    print(output_path)
    print(trace_path)


if __name__ == "__main__":
    main()
