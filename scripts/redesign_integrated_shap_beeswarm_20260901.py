from __future__ import annotations

import json
import shutil
import zipfile
import colorsys
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "analysis" / "knhanes_primary_ml_v1" / "interpretability_core_sdoh_hr_20260814"
FIG_DIR = ROOT / "manuscript" / "figures"
JMIR_DIR = ROOT / "manuscript" / "jmir_figures_20260824"
DOCX_PATH = ROOT / "manuscript" / "hypertension_introduction_pubmed_references_20260821.docx"
BACKUP_DIR = FIG_DIR / "pre_shap_redesign_backup_20260901"

SHAP_VALUES_PATH = SHAP_DIR / "integrated__core_sdoh_hr__xgboost__participant_grouped_shap_values.csv"
RANKING_PATH = SHAP_DIR / "shap_original_predictor_importance.csv"

LABELS = {
    "age": "Age",
    "bmi": "Body mass index",
    "waist_cm": "Waist circumference",
    "heart_rate": "Heart rate",
    "sex": "Sex",
    "diabetes_dx": "Diabetes",
    "aerobic_activity": "Aerobic activity",
    "self_rated_health": "Self-rated health",
    "activity_limitation": "Activity limitation",
    "family_history_htn": "Family history of hypertension",
    "current_smoking": "Current smoking",
    "alcohol_frequency": "Alcohol frequency",
    "drink_amount": "Alcohol consumption",
    "binge_drinking": "Binge drinking",
    "education_level": "Education level",
    "household_income_q": "Household income",
    "employment_status": "Employment status",
    "marital_status": "Marital status",
    "food_security_status": "Food security",
}

OUTPUTS = {
    "main_top12_png": FIG_DIR / "figure_3_grouped_shap_beeswarm.png",
    "top12_png": FIG_DIR / "figure_3_grouped_shap_beeswarm_top12.png",
    "top12_pdf": FIG_DIR / "figure_3_grouped_shap_beeswarm_top12.pdf",
    "all_png": FIG_DIR / "figure_3_grouped_shap_beeswarm_all_predictors.png",
    "all_pdf": FIG_DIR / "figure_3_grouped_shap_beeswarm_all_predictors.pdf",
    "jmir_top12_png": JMIR_DIR / "Figure_5.png",
    "jmir_top12_pdf": JMIR_DIR / "Figure_5.pdf",
}


def backup_existing_files() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUTS.values():
        if path.exists():
            dst = BACKUP_DIR / path.name
            if not dst.exists():
                shutil.copy2(path, dst)
    if DOCX_PATH.exists():
        dst = BACKUP_DIR / DOCX_PATH.name
        if not dst.exists():
            shutil.copy2(DOCX_PATH, dst)


def load_locked_order() -> list[str]:
    ranking = pd.read_csv(RANKING_PATH, encoding="utf-8-sig")
    integrated = ranking.loc[
        (ranking["model_set"] == "integrated")
        & (ranking["variable_set"] == "core_sdoh_hr")
        & (ranking["algorithm"] == "xgboost")
    ].copy()
    integrated = integrated.sort_values("rank")
    predictors = integrated["predictor"].tolist()
    if not predictors:
        raise RuntimeError("No integrated core_sdoh_hr XGBoost SHAP ranking found.")
    return predictors


def load_shap_values() -> pd.DataFrame:
    df = pd.read_csv(SHAP_VALUES_PATH, encoding="utf-8-sig")
    shap_cols = [c for c in df.columns if c.startswith("shap__") and c != "shap_sum_predictors"]
    if not shap_cols:
        raise RuntimeError("No SHAP value columns found.")
    return df


def collect_values(df: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    cols = [f"shap__{p}" for p in predictors]
    values = df[cols].to_numpy(dtype=float).ravel()
    return values[np.isfinite(values)]


def make_cmap() -> LinearSegmentedColormap:
    def boost_muted(hex_color: str, sat_factor: float = 1.18, contrast_factor: float = 1.08) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
        h, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
        saturation = min(1.0, saturation * sat_factor)
        lightness = 0.5 + (lightness - 0.5) * contrast_factor
        lightness = max(0.0, min(1.0, lightness))
        rr, gg, bb = colorsys.hls_to_rgb(h, lightness, saturation)
        return f"#{round(rr * 255):02X}{round(gg * 255):02X}{round(bb * 255):02X}"

    return LinearSegmentedColormap.from_list(
        "publication_shap_diverging",
        [boost_muted("#3D63A3"), "#F7F7F7", boost_muted("#C86B2D")],
        N=256,
    )


def plot_beeswarm(
    df: pd.DataFrame,
    predictors: list[str],
    out_png: Path,
    out_pdf: Path,
    all_values_for_norm: np.ndarray,
    height_px: int,
) -> None:
    width_px = 1200
    dpi = 100
    rng = np.random.default_rng(20260901)

    max_abs = float(np.nanmax(np.abs(all_values_for_norm)))
    max_abs = max(max_abs, 0.1)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = make_cmap()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.295, 0.13, 0.56, 0.76], facecolor="white")

    plot_predictors = list(reversed(predictors))
    for y_idx, predictor in enumerate(plot_predictors):
        col = f"shap__{predictor}"
        vals = df[col].dropna().to_numpy(dtype=float)
        order = rng.permutation(len(vals))
        vals = vals[order]
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            vals,
            np.full(len(vals), y_idx, dtype=float) + jitter,
            c=vals,
            cmap=cmap,
            norm=norm,
            s=3.8,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    x_min = float(np.nanmin(collect_values(df, predictors)))
    x_max = float(np.nanmax(collect_values(df, predictors)))
    x_pad = max((x_max - x_min) * 0.045, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(-0.7, len(predictors) - 0.3)

    ax.axvline(0, color="#4B5563", lw=1.15, zorder=1)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, alpha=0.65, zorder=0)
    ax.grid(axis="y", visible=False)

    ax.set_yticks(range(len(plot_predictors)))
    ax.set_yticklabels([LABELS.get(p, p) for p in plot_predictors], fontsize=13)
    ax.set_xlabel("SHAP value", fontsize=15, labelpad=10)
    ax.tick_params(axis="x", labelsize=12, width=0.8, length=4, colors="#111827")
    ax.tick_params(axis="y", width=0.8, length=4, colors="#111827", pad=6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.85)
        ax.spines[spine].set_color("#4B5563")

    cax = fig.add_axes([0.902, 0.22, 0.012, 0.58])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Signed grouped contribution", fontsize=11, labelpad=9)
    cbar.ax.tick_params(labelsize=10, width=0.7, length=3)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#6B7280")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def sync_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def replace_docx_image5(image_path: Path) -> None:
    if not DOCX_PATH.exists():
        return

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(DOCX_PATH, "r") as zin:
            zin.extractall(tmp_dir)

        media_path = tmp_dir / "word" / "media" / "image5.png"
        if not media_path.exists():
            raise RuntimeError("Expected DOCX SHAP image word/media/image5.png was not found.")
        shutil.copy2(image_path, media_path)

        # Keep the embedded figure at manuscript text width while preserving the new image aspect ratio.
        img = Image.open(image_path)
        width_in = 6.55
        height_in = width_in * img.height / img.width
        cx = str(int(round(width_in * 914400)))
        cy = str(int(round(height_in * 914400)))

        namespaces = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        }
        rels_xml = tmp_dir / "word" / "_rels" / "document.xml.rels"
        rels_tree = ET.parse(rels_xml)
        image5_rids = {
            rel.attrib["Id"]
            for rel in rels_tree.getroot()
            if rel.attrib.get("Target") == "media/image5.png"
        }
        if not image5_rids:
            raise RuntimeError("Could not resolve relationship id for word/media/image5.png.")

        document_xml = tmp_dir / "word" / "document.xml"
        tree = ET.parse(document_xml)
        root = tree.getroot()
        for inline in root.findall(".//wp:inline", namespaces) + root.findall(".//wp:anchor", namespaces):
            blips = inline.findall(".//a:blip", namespaces)
            if not any(blip.attrib.get(f"{{{namespaces['r']}}}embed") in image5_rids for blip in blips):
                continue
            extent = inline.find("wp:extent", namespaces)
            if extent is not None:
                extent.attrib["cx"] = cx
                extent.attrib["cy"] = cy
            for ext in inline.findall(".//a:ext", namespaces):
                ext.attrib["cx"] = cx
                ext.attrib["cy"] = cy

        tree.write(document_xml, encoding="utf-8", xml_declaration=True)

        backup_docx = DOCX_PATH.with_suffix(".docx.pre_shap_redesign_20260901.bak")
        if not backup_docx.exists():
            shutil.copy2(DOCX_PATH, backup_docx)
        with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for path in tmp_dir.rglob("*"):
                if path.is_file():
                    zout.write(path, path.relative_to(tmp_dir).as_posix())


def image_info(path: Path) -> dict[str, object]:
    img = Image.open(path)
    return {
        "path": str(path),
        "width_px": int(img.width),
        "height_px": int(img.height),
        "size_bytes": int(path.stat().st_size),
    }


def main() -> None:
    backup_existing_files()

    df = load_shap_values()
    locked_predictors = load_locked_order()
    all_values = collect_values(df, locked_predictors)

    top12 = locked_predictors[:12]
    plot_beeswarm(
        df=df,
        predictors=top12,
        out_png=OUTPUTS["top12_png"],
        out_pdf=OUTPUTS["top12_pdf"],
        all_values_for_norm=all_values,
        height_px=740,
    )
    sync_png(OUTPUTS["top12_png"], OUTPUTS["main_top12_png"])
    sync_png(OUTPUTS["top12_png"], OUTPUTS["jmir_top12_png"])
    sync_png(OUTPUTS["top12_pdf"], OUTPUTS["jmir_top12_pdf"])

    plot_beeswarm(
        df=df,
        predictors=locked_predictors,
        out_png=OUTPUTS["all_png"],
        out_pdf=OUTPUTS["all_pdf"],
        all_values_for_norm=all_values,
        height_px=800,
    )

    replace_docx_image5(OUTPUTS["top12_png"])

    report = {
        "data_sources_read_only": {
            "participant_grouped_shap_values": str(SHAP_VALUES_PATH),
            "locked_predictor_ranking": str(RANKING_PATH),
        },
        "color_encoding_confirmed_from_existing_code": "Point color encodes the signed grouped SHAP contribution itself, not the raw feature value.",
        "model_retraining": False,
        "shap_recalculation": False,
        "ranking_recalculation": False,
        "top12_predictors_from_locked_ranking": top12,
        "all_predictors_from_locked_ranking": locked_predictors,
        "outputs": {name: image_info(path) if path.suffix.lower() == ".png" else str(path) for name, path in OUTPUTS.items()},
        "backup_dir": str(BACKUP_DIR),
    }
    out_report = FIG_DIR / "integrated_shap_beeswarm_redesign_20260901_report.json"
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
