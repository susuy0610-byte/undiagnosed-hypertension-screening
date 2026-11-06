from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/knhanes_primary_ml_v1"
OUT = ROOT / "manuscript/figures/figure_2_roc_curves.png"
AUC_OUT = ROOT / "manuscript/figures/figure_2_roc_curves_auc_values.csv"

W, H = 1200, 540
SCALE = 2
MODEL_SETS = ["integrated", "male", "female"]
PANEL_TITLES = {"integrated": "Integrated", "male": "Male", "female": "Female"}

FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

DATASETS = {
    "Development": {
        "label": "Development OOF",
        "color": "#002855",
        "style": "solid",
        "path": lambda model_set: ANALYSIS / f"oof_predictions/{model_set}__core_sdoh_hr__xgboost__oof.csv",
        "outcome": "undiagnosed_htn",
    },
    "Temporal": {
        "label": "Temporal validation",
        "color": "#008C95",
        "style": "dash",
        "path": lambda model_set: ANALYSIS
        / f"time_validation_primary_core_sdoh_hr_best_single_20260813/{model_set}__core_sdoh_hr__xgboost__2021_2022_predictions.csv",
        "outcome": "undiagnosed_htn",
    },
    "HSE": {
        "label": "HSE external validation",
        "color": "#D55E00",
        "style": "dashdot",
        "path": lambda model_set: ANALYSIS
        / f"external_validation_hse_pooled_2018_2019_2021_2022_primary_core_sdoh_hr_best_single_20260813_20260819/{model_set}__core_sdoh_hr__xgboost__hse_pooled_predictions.csv",
        "outcome": "outcome",
    },
}


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD_FONT if bold else FONT), size * SCALE)


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_roc(model_set, dataset_key):
    spec = DATASETS[dataset_key]
    data = pd.read_csv(spec["path"](model_set))
    data = data[[spec["outcome"], "pred_calibrated"]].dropna()
    fpr, tpr, _ = roc_curve(data[spec["outcome"]], data["pred_calibrated"])
    return fpr, tpr, auc(fpr, tpr)


def point_on_segment(p1, p2, distance):
    x1, y1 = p1
    x2, y2 = p2
    seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if seg_len == 0:
        return p1
    frac = distance / seg_len
    return x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)


def draw_styled_line(draw, points, fill, width, style="solid"):
    if len(points) < 2:
        return
    if style == "solid":
        draw.line(points, fill=fill, width=width, joint="curve")
        return

    if style == "dash":
        pattern = [10 * SCALE, 5 * SCALE]
    elif style == "dashdot":
        pattern = [12 * SCALE, 4 * SCALE, 3 * SCALE, 4 * SCALE]
    else:
        pattern = [12 * SCALE, 8 * SCALE]

    pattern_idx = 0
    draw_on = True
    remaining = pattern[pattern_idx]
    for p1, p2 in zip(points[:-1], points[1:]):
        x1, y1 = p1
        x2, y2 = p2
        seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        used = 0
        current = p1
        while used < seg_len:
            step = min(remaining, seg_len - used)
            nxt = point_on_segment(p1, p2, used + step)
            if draw_on:
                draw.line([current, nxt], fill=fill, width=width)
            used += step
            current = nxt
            remaining -= step
            if remaining <= 1e-6:
                pattern_idx = (pattern_idx + 1) % len(pattern)
                remaining = pattern[pattern_idx]
                draw_on = not draw_on


def draw_centered_text(draw, xy, text, text_font, fill):
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=text_font, fill=fill)


def draw_rotated_label(canvas, xy, text, text_font, fill):
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=text_font)
    label = Image.new("RGBA", (bbox[2] + 12 * SCALE, bbox[3] + 12 * SCALE), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((6 * SCALE, 6 * SCALE), text, font=text_font, fill=fill)
    rotated = label.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (int(xy[0] - rotated.width / 2), int(xy[1] - rotated.height / 2)))


def main():
    canvas = Image.new("RGBA", (W * SCALE, H * SCALE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = font(24, bold=True)
    auc_line_font = font(18)
    axis_font = font(23)
    tick_font = font(17)
    legend_font = font(21)

    plot_size = 305 * SCALE
    top = 108 * SCALE
    lefts = [102 * SCALE, 467 * SCALE, 832 * SCALE]
    auc_rows = []

    for left, model_set in zip(lefts, MODEL_SETS):
        right = left + plot_size
        bottom = top + plot_size

        auc_values = {}
        for dataset_key, spec in DATASETS.items():
            fpr, tpr, auc_value = load_roc(model_set, dataset_key)
            auc_values[dataset_key] = auc_value
            auc_rows.append({"model_set": model_set, "dataset": spec["label"], "auroc": auc_value})

        title_x = (left + right) / 2
        draw_centered_text(draw, (title_x, top - 78 * SCALE), PANEL_TITLES[model_set], title_font, (20, 20, 20))
        auc_line = (
            f"Dev {auc_values['Development']:.4f} | "
            f"Temporal {auc_values['Temporal']:.4f} | "
            f"HSE {auc_values['HSE']:.4f}"
        )
        draw_centered_text(draw, (title_x, top - 46 * SCALE), auc_line, auc_line_font, (50, 50, 50))

        # Chance line and axes.
        draw_styled_line(draw, [(left, bottom), (right, top)], fill=(215, 215, 215), width=1 * SCALE, style="dash")
        draw.line([(left, top), (left, bottom), (right, bottom)], fill=(105, 105, 105), width=2 * SCALE)

        for tick in [0, 0.25, 0.50, 0.75, 1.00]:
            x = left + tick * plot_size
            y = bottom - tick * plot_size
            draw.line([(x, bottom), (x, bottom + 5 * SCALE)], fill=(80, 80, 80), width=1 * SCALE)
            draw.line([(left - 5 * SCALE, y), (left, y)], fill=(80, 80, 80), width=1 * SCALE)
            label = f"{tick:.2f}".rstrip("0").rstrip(".")
            draw_centered_text(draw, (x, bottom + 19 * SCALE), label, tick_font, (45, 45, 45))
            if model_set == "integrated" and tick != 0:
                draw.text((left - 34 * SCALE, y - 8 * SCALE), label, font=tick_font, fill=(45, 45, 45))

        for dataset_key, spec in DATASETS.items():
            fpr, tpr, auc_value = load_roc(model_set, dataset_key)
            points = [(left + x * plot_size, bottom - y * plot_size) for x, y in zip(fpr, tpr)]
            draw_styled_line(
                draw,
                points,
                fill=hex_to_rgb(spec["color"]),
                width=3 * SCALE,
                style=spec["style"],
            )

    draw_centered_text(draw, (W * SCALE / 2, 462 * SCALE), "1 - Specificity", axis_font, (35, 35, 35))
    draw_rotated_label(canvas, (20 * SCALE, 260 * SCALE), "Sensitivity", axis_font, (35, 35, 35))

    # Common legend.
    legend_y = 490 * SCALE
    total_width = 0
    legend_items = list(DATASETS.values())
    item_widths = []
    for spec in legend_items:
        text_width = draw.textbbox((0, 0), spec["label"], font=legend_font)[2]
        width = 66 * SCALE + text_width
        item_widths.append(width)
        total_width += width
    total_width += 30 * SCALE * (len(legend_items) - 1)
    start_x = (W * SCALE - total_width) / 2
    x = start_x
    for spec, item_width in zip(legend_items, item_widths):
        line_y = legend_y + 9 * SCALE
        draw_styled_line(
            draw,
            [(x, line_y), (x + 44 * SCALE, line_y)],
            fill=hex_to_rgb(spec["color"]),
            width=3 * SCALE,
            style=spec["style"],
        )
        draw.text((x + 55 * SCALE, legend_y), spec["label"], font=legend_font, fill=(35, 35, 35))
        x += item_width + 30 * SCALE

    if OUT.exists():
        backup = OUT.with_name("figure_2_roc_curves_pre_publication_redesign_20260824.png")
        if not backup.exists():
            backup.write_bytes(OUT.read_bytes())
    final = canvas.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    final.save(OUT, format="PNG", optimize=True)
    pd.DataFrame(auc_rows).to_csv(AUC_OUT, index=False)


if __name__ == "__main__":
    main()
