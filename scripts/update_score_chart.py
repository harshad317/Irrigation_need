from __future__ import annotations

import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results.tsv"
OUTPUT_PATH = ROOT / "artifacts" / "champion_score_history.svg"

WIDTH = 960
HEIGHT = 540
PLOT_LEFT = 90
PLOT_RIGHT = 40
PLOT_TOP = 70
PLOT_BOTTOM = 90
PLOT_WIDTH = WIDTH - PLOT_LEFT - PLOT_RIGHT
PLOT_HEIGHT = HEIGHT - PLOT_TOP - PLOT_BOTTOM


def load_points():
    if not RESULTS_PATH.exists():
        return []

    points = []
    champion_score = float("-inf")
    with RESULTS_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                score = float(row["val_balanced_accuracy_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if score <= champion_score + 1e-12:
                continue
            champion_score = score
            points.append(
                {
                    "index": len(points) + 1,
                    "run": row.get("run", f"run_{len(points) + 1}"),
                    "timestamp": row.get("timestamp", ""),
                    "score": score,
                }
            )
    return points


def to_x(position, total_points):
    if total_points == 1:
        return PLOT_LEFT + PLOT_WIDTH / 2
    return PLOT_LEFT + (position / (total_points - 1)) * PLOT_WIDTH


def to_y(score, score_min, score_max):
    if score_max <= score_min:
        return PLOT_TOP + PLOT_HEIGHT / 2
    normalized = (score - score_min) / (score_max - score_min)
    return PLOT_TOP + (1 - normalized) * PLOT_HEIGHT


def build_svg(points):
    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">'
            '<rect width="100%" height="100%" fill="#f7f7f5" />'
            '<text x="50%" y="50%" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#444">'
            "No champion scores recorded yet"
            "</text></svg>"
        )

    scores = [point["score"] for point in points]
    raw_min = min(scores)
    raw_max = max(scores)
    padding = max(0.0005, (raw_max - raw_min) * 0.25 or 0.001)
    score_min = raw_min - padding
    score_max = raw_max + padding

    grid_values = [score_min + (score_max - score_min) * idx / 4 for idx in range(5)]
    polyline_points = []
    circles = []
    labels = []
    x_labels = []

    for idx, point in enumerate(points):
        x_pos = to_x(idx, len(points))
        y_pos = to_y(point["score"], score_min, score_max)
        polyline_points.append(f"{x_pos:.1f},{y_pos:.1f}")
        circles.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="5.5" fill="#d9480f" '
            'stroke="#ffffff" stroke-width="2" />'
        )
        labels.append(
            f'<text x="{x_pos:.1f}" y="{y_pos - 14:.1f}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333">'
            f'{point["score"]:.6f}</text>'
        )
        x_labels.append(
            f'<text x="{x_pos:.1f}" y="{HEIGHT - 45}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#555">'
            f'{escape(point["run"])}</text>'
        )

    grid_lines = []
    y_labels = []
    for grid_value in grid_values:
        y_pos = to_y(grid_value, score_min, score_max)
        grid_lines.append(
            f'<line x1="{PLOT_LEFT}" y1="{y_pos:.1f}" x2="{WIDTH - PLOT_RIGHT}" '
            f'y2="{y_pos:.1f}" stroke="#d9d9d3" stroke-width="1" />'
        )
        y_labels.append(
            f'<text x="{PLOT_LEFT - 12}" y="{y_pos + 4:.1f}" text-anchor="end" '
            'font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#555">'
            f"{grid_value:.6f}</text>"
        )

    last_point = points[-1]
    subtitle = (
        f'Latest champion: {last_point["score"]:.6f} '
        f'({escape(last_point["run"])})'
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f7f7f5" />',
        '<rect x="28" y="28" width="904" height="484" rx="18" fill="#ffffff" '
        'stroke="#dfdfd8" stroke-width="1.5" />',
        '<text x="50%" y="62" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="28" fill="#1f2933">'
        "Champion Balanced Accuracy"
        "</text>",
        f'<text x="50%" y="88" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#52606d">'
        f"{subtitle}</text>",
        f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" '
        f'y2="{PLOT_TOP + PLOT_HEIGHT}" stroke="#6b7280" stroke-width="1.5" />',
        f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP + PLOT_HEIGHT}" '
        f'x2="{PLOT_LEFT + PLOT_WIDTH}" y2="{PLOT_TOP + PLOT_HEIGHT}" '
        'stroke="#6b7280" stroke-width="1.5" />',
        *grid_lines,
        *y_labels,
        f'<polyline fill="none" stroke="#d9480f" stroke-width="3" '
        f'points="{" ".join(polyline_points)}" />',
        *circles,
        *labels,
        *x_labels,
        '<text x="50%" y="515" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#52606d">'
        "Strict champion improvements from results.tsv"
        "</text>",
        '<text x="24" y="24" font-family="Helvetica, Arial, sans-serif" font-size="1" '
        'fill="transparent">champion-score-chart</text>',
        "</svg>",
    ]
    return "".join(svg)


def main():
    points = load_points()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_svg(points))
    latest = points[-1]["score"] if points else None
    print(f"Chart updated: {OUTPUT_PATH}")
    if latest is not None:
        print(f"Latest champion score: {latest:.6f}")


if __name__ == "__main__":
    main()
