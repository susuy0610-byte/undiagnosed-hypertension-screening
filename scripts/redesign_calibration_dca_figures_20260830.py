#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/matplotlib-cache-knhanes-hypertension",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd
from PIL import Image
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "knhanes_primary_ml_v1"
HSE_DIR = (
    ANALYSIS_DIR
    / "external_validation_hse_pooled_2018_2019_2021_2022_primary_core_sdoh_hr_best_single_20260813_20260819"
)
FIG_DIR = ROOT / "manuscript" / "figures"
JMIR_DIR = ROOT / "manuscript" / "jmir_figures_20260824"
DOCX = ROOT / "manuscript" / "hypertension_introduction_pubmed_references_20260821.docx"
BACKUP_DIR = FIG_DIR / "pre_redesign_backup_20260830"

CAL_PNG = FIG_DIR / "figure_4_hse_calibration_curves.png"
CAL_PDF = FIG_DIR / "figure_4_hse_calibration_curves.pdf"
DCA_PNG = FIG_DIR / "figure_5_hse_decision_curve.png"
DCA_PDF = FIG_DIR / "figure_5_hse_decision_curve.pdf"

JMIR_CAL_PNG = JMIR_DIR / "Figure_3.png"
JMIR_CAL_PDF = JMIR_DIR / "Figure_3.pdf"
JMIR_DCA_PNG = JMIR_DIR / "Figure_4.png"
JMIR_DCA_PDF = JMIR_DIR / "Figure_4.pdf"

WORD_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}

MODEL_COLORS = {
    "integrated": "#002855",
    "male": "#008C95",
    "female": "#D55E00",
}
MODEL_TITLES = {"integrated": "Integrated", "male": "Male", "female": "Female"}


def compact_tick(value, _pos=None):
    if abs(value) < 1e-10:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def set_publication_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.labelcolor": "#111827",
            "text.color": "#111827",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def backup_outputs() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in [CAL_PNG, CAL_PDF, DCA_PNG, DCA_PDF, JMIR_CAL_PNG, JMIR_CAL_PDF, JMIR_DCA_PNG, JMIR_DCA_PDF]:
        if path.exists():
            backup = BACKUP_DIR / path.name
            if not backup.exists():
                shutil.copy2(path, backup)


def optimize_png(path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white.alpha_composite(image)
    white.convert("RGB").save(path, format="PNG", optimize=True)


def copy_exact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def draw_calibration() -> None:
    set_publication_theme()

    model_sets = ["integrated", "male", "female"]
    ideal_color = "#B8C0CC"

    cal_data = {}
    max_axis = 0.0
    for model_set in model_sets:
        path = HSE_DIR / f"{model_set}__core_sdoh_hr__xgboost__hse_pooled_calibration_plot_data.csv"
        df = pd.read_csv(path)
        cal_data[model_set] = df
        max_axis = max(max_axis, float(df["mean_pred"].max()), float(df["observed"].max()))

    axis_max = min(1.0, max(0.45, max_axis + 0.04))
    tick_values = [0, 0.1, 0.2, 0.3, 0.4]
    if axis_max > 0.5:
        tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.3), dpi=100, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, model_set in zip(axes, model_sets):
        df = cal_data[model_set]
        observed_color = MODEL_COLORS[model_set]
        ax.plot(
            [0, axis_max],
            [0, axis_max],
            color=ideal_color,
            lw=1.1,
            linestyle=(0, (4, 4)),
            label="Ideal",
            zorder=1,
        )
        ax.plot(
            df["mean_pred"],
            df["observed"],
            color=observed_color,
            lw=2.0,
            marker="o",
            markersize=4.5,
            markerfacecolor=observed_color,
            markeredgecolor=observed_color,
            markeredgewidth=1.2,
            label="Observed",
            zorder=2,
        )
        ax.set_title(MODEL_TITLES[model_set], pad=12)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", width=0.8, length=3.5, pad=4)

    axes[0].set_ylabel("Observed event rate", labelpad=10)
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.supxlabel("Mean predicted risk", y=0.045, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.18, wspace=0.16)
    fig.savefig(CAL_PNG, dpi=100)
    fig.savefig(CAL_PDF)
    plt.close(fig)
    optimize_png(CAL_PNG)
    copy_exact(CAL_PNG, JMIR_CAL_PNG)
    copy_exact(CAL_PDF, JMIR_CAL_PDF)


def draw_dca() -> None:
    set_publication_theme()
    dca = pd.read_csv(ANALYSIS_DIR / "hse_pooled_dca_full_curve_20260819.csv")
    dca = dca[(dca["stage"] == "as_is") & (dca["threshold"].between(0.05, 0.30))].copy()

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=100)
    fig.patch.set_facecolor("white")

    for model_set in ["integrated", "male", "female"]:
        grp = dca[dca["model_set"] == model_set].sort_values("threshold")
        ax.plot(
            grp["threshold"],
            grp["net_benefit_model"],
            lw=2.2,
            color=MODEL_COLORS[model_set],
            label=MODEL_TITLES[model_set],
            solid_capstyle="round",
            zorder=3,
        )

    ref = dca[dca["model_set"] == "integrated"].sort_values("threshold")
    ax.plot(
        ref["threshold"],
        ref["net_benefit_treat_all"],
        color="#8A8F98",
        lw=1.35,
        linestyle=(0, (5, 4)),
        label="Treat all",
        zorder=1,
    )
    ax.plot(
        ref["threshold"],
        ref["net_benefit_treat_none"],
        color="#4B5563",
        lw=1.25,
        linestyle=(0, (1, 3)),
        label="Treat none",
        zorder=1,
    )

    ax.set_xlabel("Threshold probability", labelpad=14)
    ax.set_ylabel("Net benefit", labelpad=10)
    ax.set_xlim(0.05, 0.30)
    ax.set_ylim(-0.235, 0.165)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_yticks([-0.20, -0.10, 0.00, 0.10, 0.15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if abs(v) < 1e-10 else f"{v:.2f}".rstrip("0").rstrip(".")))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3.5, pad=4, labelsize=15)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=5,
        frameon=False,
        fontsize=15,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.085, right=0.965, top=0.965, bottom=0.32)
    fig.savefig(DCA_PNG, dpi=100)
    fig.savefig(DCA_PDF)
    plt.close(fig)
    optimize_png(DCA_PNG)
    copy_exact(DCA_PNG, JMIR_DCA_PNG)
    copy_exact(DCA_PDF, JMIR_DCA_PDF)


def patch_docx_media() -> list[str]:
    media_bytes = {
        "word/media/image3.png": CAL_PNG.read_bytes(),
        "word/media/image4.png": DCA_PNG.read_bytes(),
    }
    dimensions = {
        "media/image3.png": Image.open(CAL_PNG).size,
        "media/image4.png": Image.open(DCA_PNG).size,
    }
    display_width_emu = int(6.55 * 914400)

    tmp = DOCX.with_suffix(".tmp_figures_20260830.docx")
    replaced = []
    with ZipFile(DOCX, "r") as zin, ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
        relmap = {}
        if "word/_rels/document.xml.rels" in zin.namelist():
            rels = etree.fromstring(zin.read("word/_rels/document.xml.rels"))
            for rel in rels:
                relmap[rel.get("Id")] = rel.get("Target")

        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                root = etree.fromstring(data)
                for blip in root.xpath("//a:blip[@r:embed]", namespaces=WORD_NS):
                    target = relmap.get(blip.get(f"{{{WORD_NS['r']}}}embed"))
                    if target not in dimensions:
                        continue
                    width, height = dimensions[target]
                    display_height_emu = int(display_width_emu * height / width)
                    inline = blip.xpath("./ancestor::wp:inline[1]", namespaces=WORD_NS)
                    if inline:
                        extent = inline[0].find("wp:extent", namespaces=WORD_NS)
                        if extent is not None:
                            extent.set("cx", str(display_width_emu))
                            extent.set("cy", str(display_height_emu))
                    for ext in blip.xpath("./ancestor::pic:pic[1]/pic:spPr/a:xfrm/a:ext", namespaces=WORD_NS):
                        ext.set("cx", str(display_width_emu))
                        ext.set("cy", str(display_height_emu))
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
            elif item.filename in media_bytes:
                data = media_bytes[item.filename]
                replaced.append(item.filename)
            zout.writestr(item, data)
    tmp.replace(DOCX)
    return replaced


def main() -> None:
    backup_outputs()
    draw_calibration()
    draw_dca()
    replaced = patch_docx_media()
    for path in [CAL_PNG, DCA_PNG, JMIR_CAL_PNG, JMIR_DCA_PNG]:
        image = Image.open(path)
        print(f"{path} {image.size[0]}x{image.size[1]} {path.stat().st_size} bytes")
    for path in [CAL_PDF, DCA_PDF, JMIR_CAL_PDF, JMIR_DCA_PDF]:
        print(f"{path} {path.stat().st_size} bytes")
    print("replaced_docx_media=" + ",".join(replaced))


if __name__ == "__main__":
    main()
