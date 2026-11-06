from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "knhanes_primary_ml_v1"
HSE_DIR = (
    ANALYSIS_DIR
    / "external_validation_hse_pooled_2018_2019_2021_2022_primary_core_sdoh_hr_best_single_20260813_20260819"
)
OUT = ROOT / "manuscript" / "figures" / "figure_4_hse_calibration_curves.png"


def compact_tick(value, pos):
    return f"{value:.1f}".rstrip("0").rstrip(".")


def compact_y_tick(value, pos):
    if abs(value) < 1e-9:
        return ""
    return compact_tick(value, pos)


def set_plot_theme():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
        }
    )


def main():
    set_plot_theme()
    colors = {"integrated": "#2563EB", "male": "#059669", "female": "#DC2626"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), dpi=350, sharex=True, sharey=True)
    max_axis = 0.0
    cal_data = {}
    for model_set in ["integrated", "male", "female"]:
        data_path = HSE_DIR / f"{model_set}__core_sdoh_hr__xgboost__hse_pooled_calibration_plot_data.csv"
        df = pd.read_csv(data_path)
        cal_data[model_set] = df
        max_axis = max(max_axis, float(df["mean_pred"].max()), float(df["observed"].max()))
    max_axis = min(1.0, max(0.35, max_axis + 0.06))

    for ax, model_set in zip(axes, ["integrated", "male", "female"]):
        df = cal_data[model_set]
        ax.plot([0, max_axis], [0, max_axis], color="#64748B", lw=1.8, ls="--", label="Ideal")
        ax.plot(
            df["mean_pred"],
            df["observed"],
            marker="o",
            ms=8,
            lw=3,
            color=colors[model_set],
            label="Observed",
        )
        ax.set_title(model_set.capitalize(), fontweight="bold", pad=12)
        ax.set_xlabel("Mean predicted risk")
        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.grid(alpha=0.23, linewidth=0.8)

    axes[0].set_ylabel("Observed event rate")
    axes[0].yaxis.set_major_formatter(FuncFormatter(compact_y_tick))
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
