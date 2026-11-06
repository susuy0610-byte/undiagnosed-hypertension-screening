#!/usr/bin/env python3
"""Decision Curve Analysis for HSE 2018-2022 pooled external validation.

Compares Model (as-is, KNHANES-fitted calibrator) vs Model (recalibrated,
intercept+slope on HSE) vs Treat-all vs Treat-none, across the pre-specified
clinical threshold range 0.05-0.30. DCA is threshold-based, so unlike AUROC it
CAN change after recalibration even though ranking does not; both stages are
reported explicitly for that reason. No retraining, retuning, or frozen
threshold-value change is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402

STAMP = "20260819"
RECAL_DIR = ROOT / f"analysis/knhanes_primary_ml_v1/external_validation_hse_pooled_recalibration_{STAMP}"
REPORTS = ROOT / "reports"

STAGES = {
    "as_is": "1_as_is_knhanes_calibrator",
    "recalibrated": "3_intercept_slope_hse_pooled",
}


def useful_range(dca: pd.DataFrame) -> tuple[float, float] | None:
    """Largest contiguous threshold band where model beats BOTH treat-all and treat-none."""
    beats_both = (dca["net_benefit_model"] > dca["net_benefit_treat_all"]) & (dca["net_benefit_model"] > 0)
    if not beats_both.any():
        return None
    thr = dca.loc[beats_both, "threshold"]
    return float(thr.min()), float(thr.max())


def main() -> int:
    all_dca_rows = []
    summary_rows = []

    for model_set in ["integrated", "male", "female"]:
        for stage_label, stage_tag in STAGES.items():
            path = RECAL_DIR / f"{model_set}__{stage_tag}__predictions.csv"
            df = pd.read_csv(path)
            y = df["outcome"].astype(int).to_numpy()
            p = df["pred"].to_numpy(dtype=float)
            dca = pipe.dca_table(y, p)
            dca.insert(0, "stage", stage_label)
            dca.insert(0, "model_set", model_set)
            all_dca_rows.append(dca)

            band = useful_range(dca)
            key_thresholds = dca[dca["threshold"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])].copy()
            for _, row in key_thresholds.iterrows():
                summary_rows.append({
                    "model_set": model_set, "stage": stage_label, "threshold": row["threshold"],
                    "net_benefit_model": row["net_benefit_model"], "net_benefit_treat_all": row["net_benefit_treat_all"],
                    "net_benefit_treat_none": row["net_benefit_treat_none"],
                    "model_beats_all": bool(row["net_benefit_model"] > row["net_benefit_treat_all"]),
                    "model_beats_none": bool(row["net_benefit_model"] > 0),
                })
            mean_nb_range = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
            summary_rows.append({
                "model_set": model_set, "stage": stage_label, "threshold": "mean_0.05_0.30",
                "net_benefit_model": mean_nb_range, "net_benefit_treat_all": float("nan"), "net_benefit_treat_none": 0.0,
                "model_beats_all": None, "model_beats_none": None,
            })
            summary_rows.append({
                "model_set": model_set, "stage": stage_label, "threshold": "useful_range",
                "net_benefit_model": band[0] if band else float("nan"),
                "net_benefit_treat_all": band[1] if band else float("nan"),
                "net_benefit_treat_none": None, "model_beats_all": None, "model_beats_none": None,
            })

    dca_full = pd.concat(all_dca_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pipe.write_csv(dca_full, REPORTS.parent / "analysis/knhanes_primary_ml_v1" / f"hse_pooled_dca_full_curve_{STAMP}.csv")
    pipe.write_csv(summary, REPORTS / f"hse_pooled_dca_summary_{STAMP}.csv")

    def md_table(d: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
        v = d[cols].copy()
        for c in v.columns:
            if pd.api.types.is_float_dtype(v[c]):
                v[c] = v[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in v.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    report_path = REPORTS / f"hse_pooled_dca_results_{STAMP}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# HSE 2018-2022 Pooled Decision Curve Analysis - core_sdoh_hr\n\n")
        f.write("Scope: DCA only, on already-computed pooled external predictions. Compares the as-is (KNHANES-fitted calibrator) and fully recalibrated (intercept+slope, HSE pooled) probability stages. No retraining, retuning, or frozen-threshold-value change performed.\n\n")
        f.write("## Net Benefit at Key Clinical Thresholds\n\n")
        key_rows = summary[~summary["threshold"].isin(["mean_0.05_0.30", "useful_range"])].copy()
        key_rows["threshold"] = key_rows["threshold"].astype(float)
        f.write(md_table(key_rows, ["model_set", "stage", "threshold", "net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none", "model_beats_all", "model_beats_none"]))
        f.write("\n\n## Mean Net Benefit (0.05-0.30) and Useful-Range Band\n\n")
        for model_set in ["integrated", "male", "female"]:
            for stage_label in STAGES:
                mean_row = summary[(summary["model_set"] == model_set) & (summary["stage"] == stage_label) & (summary["threshold"] == "mean_0.05_0.30")].iloc[0]
                band_row = summary[(summary["model_set"] == model_set) & (summary["stage"] == stage_label) & (summary["threshold"] == "useful_range")].iloc[0]
                band_txt = "none" if pd.isna(band_row["net_benefit_model"]) else f"{band_row['net_benefit_model']:.2f}-{band_row['net_benefit_treat_all']:.2f}"
                f.write(f"- **{model_set} / {stage_label}**: mean net benefit (0.05-0.30) = {mean_row['net_benefit_model']:.4f}; threshold band where model beats both treat-all and treat-none = {band_txt}\n")
        f.write("\n## Clinical Interpretation\n\n")
        f.write("- Net benefit here translates to: for every 100 people, how many true undiagnosed-hypertension cases are caught (weighted by the harm-of-referral at each threshold) versus referring everyone or no one to have their blood pressure measured.\n")
        f.write("- Because blood pressure measurement itself is low-cost/low-risk, the clinically relevant threshold range skews low (0.05-0.30); net benefit at very high thresholds is less relevant for a screening/referral use case.\n")
        f.write("- Recalibration does not change AUROC/ranking, but it can shift how many people cross a FIXED threshold like 0.10 or 0.20, which is why the as-is vs recalibrated DCA rows can differ even though the two stages have identical AUROC.\n")
        f.write("\n## Output Files\n\n")
        f.write(f"- Full DCA curve (all thresholds, both stages): `analysis/knhanes_primary_ml_v1/hse_pooled_dca_full_curve_{STAMP}.csv`\n")
        f.write(f"- Key-threshold summary: `reports/hse_pooled_dca_summary_{STAMP}.csv`\n")

    print(summary[summary["threshold"] != "useful_range"].to_string(index=False))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
