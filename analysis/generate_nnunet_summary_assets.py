import html
import json
import math
import random
from pathlib import Path


SOURCE_ROOT = Path(r"D:\KLEIN\nnunet")
OUTPUT_DIR = Path(__file__).resolve().parent / "notion_summary_assets"

EXPERIMENTS = [
    ("CT 3D 500", "CT_3d_500epochs", "#8FAADC"),
    ("CT 3D 1000", "CT_3d_1000epochs", "#4472C4"),
    ("CT+PET 2D 500", "CT_PET_2d_500epochs", "#70AD47"),
    ("CT+PET 3D 500", "CTPET_3d_500epochs", "#ED7D31"),
    ("CT+PET 3D 1000", "CTPET_3d_1000epochs", "#C00000"),
]


def mean(values):
    return sum(values) / len(values)


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def load_experiment(label, folder, color):
    with (SOURCE_ROOT / folder / "summary.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    positive_dice = []
    negative_perfect = 0
    negative_false_positive = 0
    false_positive_voxels = []

    for case in data["metric_per_case"]:
        metrics = case["metrics"]["1"]
        if metrics["n_ref"] > 0:
            positive_dice.append(float(metrics["Dice"]))
        elif metrics["n_pred"] == 0:
            negative_perfect += 1
        else:
            negative_false_positive += 1
            false_positive_voxels.append(float(metrics["n_pred"]))

    positive_mean = mean(positive_dice)
    positive_std = math.sqrt(
        mean([(value - positive_mean) ** 2 for value in positive_dice])
    )
    negative_total = negative_perfect + negative_false_positive
    return {
        "label": label,
        "folder": folder,
        "color": color,
        "summary_dice": float(data["foreground_mean"]["Dice"]),
        "positive_dice": positive_dice,
        "positive_mean": positive_mean,
        "positive_median": percentile(positive_dice, 0.5),
        "positive_std": positive_std,
        "q1": percentile(positive_dice, 0.25),
        "q3": percentile(positive_dice, 0.75),
        "minimum": min(positive_dice),
        "maximum": max(positive_dice),
        "negative_perfect": negative_perfect,
        "negative_false_positive": negative_false_positive,
        "specificity": negative_perfect / negative_total,
        "mean_false_positive_voxels": (
            mean(false_positive_voxels) if false_positive_voxels else 0.0
        ),
    }


def svg_text(x, y, text, size=16, weight="normal", anchor="middle", fill="#222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(text)}</text>'
    )


def save_svg(name, width, height, content):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        + "".join(content)
        + "</svg>"
    )
    (OUTPUT_DIR / name).write_text(svg, encoding="utf-8")


def plot_dashboard(results):
    width, height = 1800, 720
    content = [
        svg_text(900, 44, "CT / CT+PET nnU-Net Experiment Dashboard", 27, "bold"),
        svg_text(
            900,
            75,
            "Fold 0: 720 training + 180 validation cases (96 positive, 84 negative)",
            16,
            fill="#555",
        ),
    ]
    panels = [
        ("Positive-case Dice", "Dice (%)", "positive"),
        ("Negative-case Specificity", "Specificity (%)", "specificity"),
        ("nnU-Net summary Dice", "Dice (%)", "summary"),
    ]
    panel_width = 540
    start_x = 60
    gap = 45
    chart_top = 145
    chart_height = 410
    baseline = chart_top + chart_height

    for panel_index, (title, ylabel, kind) in enumerate(panels):
        left = start_x + panel_index * (panel_width + gap)
        content.extend(
            [
                svg_text(left + panel_width / 2, 120, title, 20, "bold"),
                f'<line x1="{left}" y1="{baseline}" x2="{left + panel_width}" '
                f'y2="{baseline}" stroke="#555"/>',
                f'<line x1="{left}" y1="{chart_top}" x2="{left}" '
                f'y2="{baseline}" stroke="#555"/>',
            ]
        )
        for tick in range(0, 101, 20):
            y = baseline - chart_height * tick / 100
            content.append(
                f'<line x1="{left}" y1="{y}" x2="{left + panel_width}" y2="{y}" '
                f'stroke="#ddd" stroke-dasharray="3 4"/>'
            )
            content.append(svg_text(left - 10, y + 5, str(tick), 12, anchor="end"))
        content.append(
            f'<text x="{left - 45}" y="{chart_top + chart_height / 2}" '
            f'font-family="Arial" font-size="14" text-anchor="middle" '
            f'transform="rotate(-90 {left - 45} {chart_top + chart_height / 2})">'
            f"{html.escape(ylabel)}</text>"
        )

        slot = panel_width / len(results)
        for index, item in enumerate(results):
            center = left + slot * (index + 0.5)
            if kind == "positive":
                values = [
                    ("M", item["positive_mean"] * 100, 0.75),
                    ("Md", item["positive_median"] * 100, 0.4),
                ]
                for offset, (short, value, opacity) in zip((-15, 15), values):
                    bar_height = chart_height * value / 100
                    content.append(
                        f'<rect x="{center + offset - 12}" y="{baseline - bar_height}" '
                        f'width="24" height="{bar_height}" fill="{item["color"]}" '
                        f'fill-opacity="{opacity}"/>'
                    )
                    content.append(
                        svg_text(center + offset, baseline - bar_height - 8, f"{value:.1f}", 11)
                    )
                    content.append(svg_text(center + offset, baseline + 18, short, 10))
            else:
                value = (
                    item["specificity"] * 100
                    if kind == "specificity"
                    else item["summary_dice"] * 100
                )
                bar_height = chart_height * value / 100
                content.append(
                    f'<rect x="{center - 22}" y="{baseline - bar_height}" width="44" '
                    f'height="{bar_height}" fill="{item["color"]}" fill-opacity="0.82"/>'
                )
                content.append(
                    svg_text(center, baseline - bar_height - 8, f"{value:.1f}", 12)
                )
            content.append(
                f'<text x="{center}" y="{baseline + 52}" font-family="Arial" '
                f'font-size="11" text-anchor="end" '
                f'transform="rotate(-28 {center} {baseline + 52})">'
                f'{html.escape(item["label"])}</text>'
            )

    content.append(
        svg_text(
            900,
            695,
            "Positive Dice evaluates lesion-containing cases; specificity measures "
            "whether negative cases remain prediction-free.",
            15,
            fill="#555",
        )
    )
    save_svg("experiment_dashboard.svg", width, height, content)


def plot_distributions(results):
    width, height = 1500, 780
    left, right, top, bottom = 100, 40, 90, 130
    chart_width = width - left - right
    chart_height = height - top - bottom
    baseline = top + chart_height
    content = [
        svg_text(
            width / 2,
            45,
            "Positive-case Dice Distributions (N=96 per experiment)",
            26,
            "bold",
        )
    ]
    for tick in range(0, 101, 10):
        y = baseline - chart_height * tick / 100
        content.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" '
            f'stroke="#ddd" stroke-dasharray="3 4"/>'
        )
        content.append(svg_text(left - 12, y + 5, str(tick), 13, anchor="end"))
    content.append(
        f'<line x1="{left}" y1="{baseline}" x2="{width - right}" '
        f'y2="{baseline}" stroke="#555"/>'
    )
    content.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="#555"/>'
    )
    content.append(
        f'<text x="32" y="{top + chart_height / 2}" font-family="Arial" '
        f'font-size="16" text-anchor="middle" '
        f'transform="rotate(-90 32 {top + chart_height / 2})">Dice (%)</text>'
    )

    rng = random.Random(20260612)
    slot = chart_width / len(results)
    for index, item in enumerate(results):
        center = left + slot * (index + 0.5)
        scale_y = lambda value: baseline - chart_height * value
        for value in item["positive_dice"]:
            x = center + rng.uniform(-42, 42)
            y = scale_y(value)
            content.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" '
                f'fill="{item["color"]}" fill-opacity="0.35"/>'
            )

        q1_y = scale_y(item["q1"])
        q3_y = scale_y(item["q3"])
        median_y = scale_y(item["positive_median"])
        min_y = scale_y(item["minimum"])
        max_y = scale_y(item["maximum"])
        content.extend(
            [
                f'<line x1="{center}" y1="{max_y}" x2="{center}" y2="{min_y}" '
                f'stroke="#555" stroke-width="2"/>',
                f'<line x1="{center - 18}" y1="{max_y}" x2="{center + 18}" '
                f'y2="{max_y}" stroke="#555" stroke-width="2"/>',
                f'<line x1="{center - 18}" y1="{min_y}" x2="{center + 18}" '
                f'y2="{min_y}" stroke="#555" stroke-width="2"/>',
                f'<rect x="{center - 38}" y="{q3_y}" width="76" '
                f'height="{q1_y - q3_y}" fill="{item["color"]}" '
                f'fill-opacity="0.75" stroke="{item["color"]}" stroke-width="2"/>',
                f'<line x1="{center - 38}" y1="{median_y}" x2="{center + 38}" '
                f'y2="{median_y}" stroke="#111" stroke-width="4"/>',
            ]
        )
        content.append(
            svg_text(
                center,
                top - 15,
                f'Mean {item["positive_mean"] * 100:.1f}%',
                13,
                fill=item["color"],
            )
        )
        content.append(
            f'<text x="{center}" y="{baseline + 45}" font-family="Arial" '
            f'font-size="14" text-anchor="end" '
            f'transform="rotate(-20 {center} {baseline + 45})">'
            f'{html.escape(item["label"])}</text>'
        )
    save_svg("positive_dice_distributions.svg", width, height, content)


def plot_workflow():
    width, height = 1800, 470
    stages = [
        ("Pipeline check", "RetinaOCT", "100 → 1000 epochs", "#D9EAF7"),
        ("Data engineering", "900 CT/PET volumes", "labels 255 → 1", "#DDEBF7"),
        ("Controlled split", "5-fold split", "Fold 0: 720 / 180", "#E2F0D9"),
        ("Modality study", "CT vs CT+PET", "same validation cases", "#FFF2CC"),
        ("Dimension study", "2D vs 3D fullres", "500 vs 1000 epochs", "#FCE4D6"),
        ("Clinical evaluation", "Positive Dice", "+ negative specificity", "#F4CCCC"),
    ]
    content = [
        svg_text(width / 2, 52, "End-to-end Experimental Workflow", 27, "bold")
    ]
    box_width, box_height = 245, 220
    gap = 45
    start_x = 45
    y = 120
    for index, (title, line1, line2, color) in enumerate(stages):
        x = start_x + index * (box_width + gap)
        content.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" '
                f'rx="16" fill="{color}" stroke="#666" stroke-width="2"/>',
                svg_text(x + box_width / 2, y + 55, title, 19, "bold"),
                svg_text(x + box_width / 2, y + 115, line1, 17),
                svg_text(x + box_width / 2, y + 153, line2, 16, fill="#555"),
            ]
        )
        if index < len(stages) - 1:
            arrow_x = x + box_width
            content.append(
                f'<line x1="{arrow_x + 8}" y1="{y + box_height / 2}" '
                f'x2="{arrow_x + gap - 10}" y2="{y + box_height / 2}" '
                f'stroke="#555" stroke-width="3"/>'
            )
            content.append(
                f'<polygon points="{arrow_x + gap - 10},{y + box_height / 2 - 8} '
                f'{arrow_x + gap},{y + box_height / 2} '
                f'{arrow_x + gap - 10},{y + box_height / 2 + 8}" fill="#555"/>'
            )
    save_svg("workflow_overview.svg", width, height, content)


def save_metrics(results):
    serializable = []
    for item in results:
        row = {key: value for key, value in item.items() if key != "positive_dice"}
        row["positive_count"] = len(item["positive_dice"])
        row["negative_count"] = (
            item["negative_perfect"] + item["negative_false_positive"]
        )
        serializable.append(row)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [load_experiment(*experiment) for experiment in EXPERIMENTS]
    save_metrics(results)
    plot_dashboard(results)
    plot_distributions(results)
    plot_workflow()
    print(f"Generated summary assets in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
