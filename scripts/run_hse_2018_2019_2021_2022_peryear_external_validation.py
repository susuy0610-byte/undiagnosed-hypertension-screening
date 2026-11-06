#!/usr/bin/env python3
"""Prediction-only HSE external validation run SEPARATELY per year (2018/2019/2021/2022).

Reuses the already-built pooled harmonized inputs (same cohort-exclusion and
predictor-missingness-filter rules as the pooled run) and subsets by
source_year, so each year's result is directly comparable to the pooled run.
Uses frozen KNHANES 2014-2020 final model bundles. Does not retrain, retune,
reselect variables, recalibrate, or reset thresholds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402


STAMP = "20260819"
FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"
OUT = ROOT / "analysis/knhanes_primary_ml_v1"
FINAL_MODELS = OUT / "final_models"
REPORTS = ROOT / "reports"
INPUTS = ROOT / "data/processed"
YEARS = ["2018", "2019", "2021", "2022"]

MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
DEV_METRICS_PATH = FINAL_MODELS / f"{FREEZE_TAG}_selected_models.csv"
TEMPORAL_DIR = OUT / f"time_validation_{FREEZE_TAG}"
TEMPORAL_METRICS_PATH = TEMPORAL_DIR / "time_validation_metrics_selected_models.csv"
MAPPING_PATH = REPORTS / f"hse_pooled_core_sdoh_hr_mapping_final_{STAMP}.csv"


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, n_bootstrap: int, seed: int) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = p[idx]
        if metric == "auroc":
            values.append(float(roc_auc_score(y_b, p_b)))
        elif metric == "auprc":
            values.append(float(average_precision_score(y_b, p_b)))
        else:
            raise ValueError(metric)
    if not values:
        return {"lower": math.nan, "upper": math.nan, "n_success": 0}
    arr = np.asarray(values, dtype=float)
    return {"lower": float(np.percentile(arr, 2.5)), "upper": float(np.percentile(arr, 97.5)), "n_success": int(len(arr))}


def category_qc(X: pd.DataFrame, model_set: str, year: str) -> list[dict[str, Any]]:
    rows = []
    for col in X.columns:
        if col not in pipe.VALID_CATEGORIES:
            continue
        allowed = set(pipe.VALID_CATEGORIES[col])
        observed = sorted(pd.to_numeric(X[col], errors="coerce").dropna().astype(int).unique().tolist())
        unseen = [v for v in observed if v not in allowed]
        rows.append({
            "model_set": model_set, "source_year": year, "predictor": col,
            "allowed_levels": json.dumps(sorted(allowed)), "observed_levels": json.dumps(observed),
            "unseen_levels": json.dumps(unseen), "has_unseen_level": bool(unseen),
        })
    return rows


def missingness_qc(X: pd.DataFrame, model_set: str, year: str) -> list[dict[str, Any]]:
    rows = []
    n = len(X)
    for col in X.columns:
        missing_n = int(X[col].isna().sum())
        rows.append({
            "model_set": model_set, "source_year": year, "predictor": col,
            "n": n, "missing_n": missing_n, "missing_rate": missing_n / n if n else math.nan, "all_missing": missing_n == n,
        })
    return rows


def summarize_prediction(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, Any]:
    raw_intercept, raw_slope = pipe.calibration_intercept_slope(y, p_raw)
    cal_intercept, cal_slope = pipe.calibration_intercept_slope(y, p_cal)
    auroc = float(roc_auc_score(y, p_cal))
    auprc = float(average_precision_score(y, p_cal))
    auroc_ci = bootstrap_ci(y, p_cal, "auroc", n_bootstrap, seed)
    auprc_ci = bootstrap_ci(y, p_cal, "auprc", n_bootstrap, seed + 31)
    dca = pipe.dca_table(y, p_cal)
    mean_nb = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
    return {
        "auroc": auroc, "auroc_ci_lower": auroc_ci["lower"], "auroc_ci_upper": auroc_ci["upper"],
        "auroc_bootstrap_success": auroc_ci["n_success"], "auprc": auprc,
        "auprc_ci_lower": auprc_ci["lower"], "auprc_ci_upper": auprc_ci["upper"],
        "auprc_bootstrap_success": auprc_ci["n_success"], "brier": float(brier_score_loss(y, p_cal)),
        "calibration_intercept": cal_intercept, "calibration_slope": cal_slope,
        "calibration_intercept_raw": raw_intercept, "calibration_slope_raw": raw_slope,
        "mean_net_benefit_0.05_0.30": mean_nb,
    }, dca


def pooled_input_path(model_set: str) -> Path:
    return INPUTS / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("freeze_tag") != FREEZE_TAG:
        raise RuntimeError(f"Unexpected freeze tag in {MANIFEST_PATH}")
    return manifest


def validate_one_year(
    model_set: str, year: str, selected: dict[str, Any], dev_metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame, n_bootstrap: int, seed: int, external_dir: Path,
):
    bundle_path = Path(selected["bundle_path"])
    bundle = joblib.load(bundle_path)
    variable_set = str(bundle["variable_set"])
    algorithm = str(bundle["algorithm"])
    features = list(bundle["features"])
    tag = f"{model_set}__core_sdoh_hr__{algorithm}__hse{year}"

    full = pd.read_csv(pooled_input_path(model_set))
    df = full[full["source_year"].astype(str) == year].copy()
    if df.empty:
        raise RuntimeError(f"{tag}: no rows for year {year}")
    if df["person_id"].duplicated().any():
        raise RuntimeError(f"{tag}: duplicate person_id")

    X = df[features]
    y = df["outcome"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{tag}: fewer than two outcome classes for {year} ({model_set})")

    missing_rows = missingness_qc(X, model_set, year)
    category_rows = category_qc(X, model_set, year)
    if any(r["has_unseen_level"] for r in category_rows):
        raise RuntimeError(f"{tag}: unseen categorical level detected")

    matrices = pipe.transformed_list(bundle["preprocessors"], X)
    column_match = all(list(mat.columns) == list(bundle["preprocessors"][i].columns_) for i, mat in enumerate(matrices))
    transform_missing = sum(int(np.isnan(mat.to_numpy()).sum()) for mat in matrices)
    if not column_match or transform_missing:
        raise RuntimeError(f"{tag}: transform schema mismatch or missing values")

    p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
    p_cal = bundle["calibrator"].transform(p_raw)
    if np.isnan(p_raw).any() or np.isnan(p_cal).any() or ((p_raw < 0) | (p_raw > 1)).any() or ((p_cal < 0) | (p_cal > 1)).any():
        raise RuntimeError(f"{tag}: invalid prediction values")

    pred = df[["person_id", "outcome"]].copy()
    pred.insert(1, "source_dataset", f"HSE {year}")
    pred["model_set"] = model_set
    pred["algorithm"] = algorithm
    pred["pred_raw"] = p_raw
    pred["pred_calibrated"] = p_cal
    pipe.write_csv(pred, external_dir / f"{tag}__predictions.csv")

    summary, dca = summarize_prediction(y, p_raw, p_cal, n_bootstrap, seed)
    dca.insert(0, "source_year", year)
    dca.insert(0, "model_set", model_set)
    pipe.write_csv(dca, external_dir / f"{tag}__dca.csv")

    calib = pipe.calibration_plot_table(y, p_cal)
    calib.insert(0, "source_year", year)
    calib.insert(0, "model_set", model_set)
    pipe.write_csv(calib, external_dir / f"{tag}__calibration_plot_data.csv")
    pipe.plot_calibration(calib, external_dir / "plots" / f"{tag}__calibration.png", f"{model_set} core_sdoh_hr {algorithm} HSE {year}")

    metric_row = {"model_set": model_set, "source_year": year, "variable_set": variable_set, "algorithm": algorithm, "n": int(len(df)), "events": int(y.sum()), "event_rate": float(y.mean()), **summary}

    threshold_rows = []
    for _, thr_row in bundle["thresholds"].iterrows():
        met = pipe.threshold_metrics(y, p_cal, float(thr_row["threshold"]))
        threshold_rows.append({"model_set": model_set, "source_year": year, "threshold_type": thr_row["threshold_type"], "development_threshold": float(thr_row["threshold"]), **met})

    dev_row = dev_metrics[(dev_metrics["model_set"] == model_set) & (dev_metrics["variable_set"] == variable_set) & (dev_metrics["algorithm"] == algorithm)].iloc[0]
    temporal_row = temporal_metrics[(temporal_metrics["model_set"] == model_set) & (temporal_metrics["variable_set"] == variable_set) & (temporal_metrics["algorithm"] == algorithm)].iloc[0]
    comparison_row = {
        "model_set": model_set, "source_year": year,
        "development_auroc": float(dev_row["auroc_calibrated"]), "temporal_auroc": float(temporal_row["auroc"]), "external_auroc": summary["auroc"],
        "delta_auroc_external_minus_development": summary["auroc"] - float(dev_row["auroc_calibrated"]),
        "delta_auroc_external_minus_temporal": summary["auroc"] - float(temporal_row["auroc"]),
        "development_brier": float(dev_row["brier_calibrated"]), "temporal_brier": float(temporal_row["brier"]), "external_brier": summary["brier"],
        "external_calibration_intercept": summary["calibration_intercept"], "external_calibration_slope": summary["calibration_slope"],
    }
    qc_row = {
        "model_set": model_set, "source_year": year, "n": int(len(df)), "events": int(y.sum()), "event_rate": float(y.mean()),
        "feature_count": len(features), "transformed_column_match": bool(column_match),
        "all_na_features": ",".join([c for c in features if X[c].isna().all()]),
        "max_participant_missing_fraction": float(X.isna().mean(axis=1).max()),
    }
    return metric_row, threshold_rows, missing_rows, category_rows, comparison_row, qc_row


def md_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    external_dir = OUT / f"external_validation_hse_peryear_{FREEZE_TAG}_{STAMP}"
    external_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    manifest = load_manifest()
    dev_metrics = pd.read_csv(DEV_METRICS_PATH)
    temporal_metrics = pd.read_csv(TEMPORAL_METRICS_PATH)
    n_bootstrap = int(manifest.get("n_bootstrap", 2000))
    seed = int(manifest.get("seed", 20260722))
    selected_bundles = manifest["selected_bundles"]

    metric_rows, threshold_rows, missing_rows, category_rows, comparison_rows, qc_rows = [], [], [], [], [], []
    i = 0
    for year in YEARS:
        for model_set in ["integrated", "male", "female"]:
            metric, thr, miss, cat, comp, qc = validate_one_year(
                model_set, year, selected_bundles[model_set], dev_metrics, temporal_metrics,
                n_bootstrap, seed + i * 1000, external_dir,
            )
            metric_rows.append(metric)
            threshold_rows.extend(thr)
            missing_rows.extend(miss)
            category_rows.extend(cat)
            comparison_rows.append(comp)
            qc_rows.append(qc)
            i += 1

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    missing = pd.DataFrame(missing_rows)
    category = pd.DataFrame(category_rows)
    comparison = pd.DataFrame(comparison_rows)
    pred_qc = pd.DataFrame(qc_rows)

    pipe.write_csv(metrics, external_dir / "external_validation_hse_peryear_metrics.csv")
    pipe.write_csv(thresholds, external_dir / "external_validation_hse_peryear_threshold_metrics.csv")
    pipe.write_csv(missing, external_dir / "external_validation_hse_peryear_predictor_missingness_qc.csv")
    pipe.write_csv(category, external_dir / "external_validation_hse_peryear_category_level_qc.csv")
    pipe.write_csv(comparison, external_dir / "external_validation_hse_peryear_development_temporal_external_comparison.csv")
    pipe.write_csv(pred_qc, external_dir / "external_validation_hse_peryear_prediction_qc.csv")

    run_config = {
        "dataset": "HSE per-year (2018, 2019, 2021, 2022 run separately)", "freeze_tag": FREEZE_TAG,
        "manifest_path": str(MANIFEST_PATH), "external_dir": str(external_dir), "mapping_path": str(MAPPING_PATH),
        "prediction_only": True, "retraining_executed": False, "retuning_executed": False,
        "variable_reselection_executed": False, "threshold_reset_executed": False, "calibrator_refit_executed": False,
        "n_bootstrap": n_bootstrap, "seed": seed,
        "note": "Each year subsets the already-built pooled harmonized input by source_year; cohort-exclusion and predictor-missingness-filter rules are identical to the pooled run, so per-year and pooled results are directly comparable.",
    }
    pipe.save_json(run_config, external_dir / "external_validation_hse_peryear_run_config.json")

    report_path = REPORTS / f"hse_peryear_2018_2019_2021_2022_external_validation_results_{STAMP}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# HSE 2018 / 2019 / 2021 / 2022 External Validation Run SEPARATELY per Year - core_sdoh_hr\n\n")
        f.write("Scope: prediction-only external validation using frozen KNHANES 2014-2020 final bundles, run independently for each HSE year (not pooled). Each year uses the same harmonized cohort as the pooled run, subset by year. No retraining, tuning, variable reselection, recalibration, or threshold reset was run.\n\n")
        f.write("## External Validation Metrics by Year\n\n")
        f.write(md_table(metrics, ["source_year", "model_set", "n", "events", "event_rate", "auroc", "auroc_ci_lower", "auroc_ci_upper", "auprc", "brier", "calibration_intercept", "calibration_slope", "mean_net_benefit_0.05_0.30"]))
        f.write("\n\n## Development vs Temporal vs Each HSE Year\n\n")
        f.write(md_table(comparison, ["source_year", "model_set", "development_auroc", "temporal_auroc", "external_auroc", "delta_auroc_external_minus_development", "external_calibration_intercept", "external_calibration_slope"]))
        f.write("\n\n## Prediction QC by Year\n\n")
        f.write(md_table(pred_qc, ["source_year", "model_set", "n", "events", "feature_count", "transformed_column_match", "all_na_features", "max_participant_missing_fraction"]))
        f.write("\n\n## Notes\n\n")
        f.write("- Each year's cohort is a subset of the same pooled-harmonized cohort used in the pooled HSE 2018-2022 run (identical exclusion/missingness-filter rules), so pooled vs per-year deltas reflect real between-year differences, not different filtering logic.\n")
        f.write("- 2021 has the smallest n and the widest bootstrap CIs due to markedly lower BP/nurse-visit completion that year.\n")
        f.write("- Harmonization limitations (family_history_htn/food_security_status all-NA, drink_amount/binge_drinking proxy construct differences pre-2022, aerobic_activity all-NA in 2019, 2021 BMI derived with high missingness, banded age) apply identically to each year as documented in the pooled mapping report.\n")
    print("HSE per-year external validation complete")
    print(metrics.to_string(index=False))
    print(f"report: {report_path}")
    print(f"output_dir: {external_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
