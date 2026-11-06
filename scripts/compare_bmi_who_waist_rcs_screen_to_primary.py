#!/usr/bin/env python3
"""Compare the BMI WHO + waist RCS screen against existing primary OOF metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCREEN_OUT = ROOT / "analysis" / "knhanes_bmi_who_waist_rcs_6alg_screen_20260812"
PRIMARY_OUT = ROOT / "analysis" / "knhanes_primary_ml_v1"
VARIABLE_SET = "core_sdoh_hr_bmi_who_waist_rcs"
PRIMARY_VARIABLE_SET = "core_sdoh_hr"
ALGORITHMS = ["logistic", "lasso", "random_forest", "extra_trees", "xgboost", "lightgbm"]
METRIC_COLS = [
    "auroc_calibrated",
    "auprc_calibrated",
    "brier_calibrated",
    "calibration_intercept_raw",
    "calibration_slope_raw",
    "mean_net_benefit_0.05_0.30",
]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    screen_metrics_path = SCREEN_OUT / "metrics" / "oof_metrics_all_models.csv"
    primary_metrics_path = PRIMARY_OUT / "metrics" / "oof_metrics_all_models.csv"
    screen_thresholds_path = SCREEN_OUT / "metrics" / "oof_thresholds_all_models.csv"
    primary_thresholds_path = PRIMARY_OUT / "metrics" / "oof_thresholds_all_models.csv"

    screen = pd.read_csv(screen_metrics_path)
    primary = pd.read_csv(primary_metrics_path)

    screen = screen[(screen["variable_set"] == VARIABLE_SET) & (screen["algorithm"].isin(ALGORITHMS))].copy()
    primary = primary[(primary["variable_set"] == PRIMARY_VARIABLE_SET)].copy()

    screen_rank = screen.sort_values(["model_set", "auroc_calibrated"], ascending=[True, False]).copy()
    screen_rank["rank_within_model_set"] = screen_rank.groupby("model_set")["auroc_calibrated"].rank(
        method="first", ascending=False
    )
    write_csv(screen_rank, SCREEN_OUT / "metrics" / "screen_oof_ranked_by_auroc.csv")

    primary_same_alg = primary[primary["algorithm"].isin([a for a in ALGORITHMS if a != "extra_trees"])].copy()
    merged = screen.merge(
        primary_same_alg,
        on=["model_set", "algorithm"],
        how="left",
        suffixes=("_screen", "_primary"),
    )
    for col in METRIC_COLS:
        left = f"{col}_screen"
        right = f"{col}_primary"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{col}"] = merged[left] - merged[right]
    write_csv(merged, SCREEN_OUT / "metrics" / "screen_vs_primary_same_algorithm.csv")

    primary_best = (
        primary[primary["algorithm"] != "him_benchmark"]
        .sort_values(["model_set", "auroc_calibrated"], ascending=[True, False])
        .groupby("model_set", as_index=False)
        .head(1)
    )
    best_merge = screen_rank.merge(
        primary_best,
        on="model_set",
        how="left",
        suffixes=("_screen", "_primary_best"),
    )
    for col in METRIC_COLS:
        left = f"{col}_screen"
        right = f"{col}_primary_best"
        if left in best_merge.columns and right in best_merge.columns:
            best_merge[f"delta_vs_primary_best_{col}"] = best_merge[left] - best_merge[right]
    write_csv(best_merge, SCREEN_OUT / "metrics" / "screen_vs_primary_best.csv")

    if screen_thresholds_path.exists() and primary_thresholds_path.exists():
        screen_thr = pd.read_csv(screen_thresholds_path)
        primary_thr = pd.read_csv(primary_thresholds_path)
        screen_thr = screen_thr[
            (screen_thr["variable_set"] == VARIABLE_SET) & (screen_thr["algorithm"].isin(ALGORITHMS))
        ].copy()
        primary_thr = primary_thr[(primary_thr["variable_set"] == PRIMARY_VARIABLE_SET)].copy()
        threshold_merge = screen_thr.merge(
            primary_thr,
            on=["model_set", "algorithm", "threshold_type"],
            how="left",
            suffixes=("_screen", "_primary"),
        )
        for col in ["sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "referral_rate"]:
            left = f"{col}_screen"
            right = f"{col}_primary"
            if left in threshold_merge.columns and right in threshold_merge.columns:
                threshold_merge[f"delta_{col}"] = threshold_merge[left] - threshold_merge[right]
        write_csv(threshold_merge, SCREEN_OUT / "metrics" / "screen_vs_primary_thresholds_same_algorithm.csv")

    print(f"Wrote comparison files under {SCREEN_OUT / 'metrics'}")


if __name__ == "__main__":
    main()
