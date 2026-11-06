#!/usr/bin/env python3
"""Prediction-only HSE 2022 external validation for frozen core_sdoh_hr models.

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


STAMP = "20260817"
FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"
OUT = ROOT / "analysis/knhanes_primary_ml_v1"
FINAL_MODELS = OUT / "final_models"
EXTERNAL_DIR = OUT / f"external_validation_hse2022_{FREEZE_TAG}_{STAMP}"
REPORTS = ROOT / "reports"
INPUTS = ROOT / "data/processed"

MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
DEV_METRICS_PATH = FINAL_MODELS / f"{FREEZE_TAG}_selected_models.csv"
TEMPORAL_DIR = OUT / f"time_validation_{FREEZE_TAG}"
TEMPORAL_METRICS_PATH = TEMPORAL_DIR / "time_validation_metrics_selected_models.csv"
MAPPING_PATH = REPORTS / f"hse_2022_core_sdoh_hr_mapping_final_{STAMP}.csv"
FLOW_PATH = REPORTS / f"hse_2022_core_sdoh_hr_flow_{STAMP}.csv"
INPUT_COMPAT_PATH = REPORTS / f"hse_2022_core_sdoh_hr_input_compatibility_{STAMP}.csv"
DECISION_NOTES_PATH = REPORTS / f"hse_2022_external_mapping_decision_notes_{STAMP}.md"


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
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "upper": float(np.percentile(arr, 97.5)),
        "n_success": int(len(arr)),
    }


def category_qc(X: pd.DataFrame, model_set: str, variable_set: str, algorithm: str) -> list[dict[str, Any]]:
    rows = []
    for col in X.columns:
        if col not in pipe.VALID_CATEGORIES:
            continue
        allowed = set(pipe.VALID_CATEGORIES[col])
        observed = sorted(pd.to_numeric(X[col], errors="coerce").dropna().astype(int).unique().tolist())
        unseen = [v for v in observed if v not in allowed]
        rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "predictor": col,
                "allowed_levels": json.dumps(sorted(allowed)),
                "observed_levels": json.dumps(observed),
                "unseen_levels": json.dumps(unseen),
                "has_unseen_level": bool(unseen),
            }
        )
    return rows


def missingness_qc(X: pd.DataFrame, model_set: str, variable_set: str, algorithm: str) -> list[dict[str, Any]]:
    rows = []
    n = len(X)
    for col in X.columns:
        missing_n = int(X[col].isna().sum())
        rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "predictor": col,
                "n": n,
                "missing_n": missing_n,
                "missing_rate": missing_n / n if n else math.nan,
                "all_missing": missing_n == n,
            }
        )
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
        "auroc": auroc,
        "auroc_ci_lower": auroc_ci["lower"],
        "auroc_ci_upper": auroc_ci["upper"],
        "auroc_bootstrap_success": auroc_ci["n_success"],
        "auprc": auprc,
        "auprc_ci_lower": auprc_ci["lower"],
        "auprc_ci_upper": auprc_ci["upper"],
        "auprc_bootstrap_success": auprc_ci["n_success"],
        "brier": float(brier_score_loss(y, p_cal)),
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "calibration_intercept_raw": raw_intercept,
        "calibration_slope_raw": raw_slope,
        "mean_net_benefit_0.05_0.30": mean_nb,
    }


def input_path(model_set: str) -> Path:
    return INPUTS / f"hse_2022_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("freeze_tag") != FREEZE_TAG:
        raise RuntimeError(f"Unexpected freeze tag in {MANIFEST_PATH}")
    return manifest


def validate_one(
    model_set: str,
    selected: dict[str, Any],
    dev_metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bundle_path = Path(selected["bundle_path"])
    bundle = joblib.load(bundle_path)
    variable_set = str(bundle["variable_set"])
    algorithm = str(bundle["algorithm"])
    combo = pipe.model_combo_id(model_set, variable_set, algorithm)
    features = list(bundle["features"])

    if variable_set != "core_sdoh_hr":
        raise RuntimeError(f"{combo}: non-primary variable set in frozen bundle")
    if model_set != str(bundle["model_set"]):
        raise RuntimeError(f"{combo}: bundle model_set mismatch")
    if len(bundle.get("preprocessors", [])) != 20 or len(bundle.get("models", [])) != 20:
        raise RuntimeError(f"{combo}: expected 20 preprocessors and 20 models")

    df = pd.read_csv(input_path(model_set))
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{combo}: missing input columns {missing_cols}")
    if df["person_id"].duplicated().any():
        raise RuntimeError(f"{combo}: duplicate person_id in external input")

    X = df[features]
    y = df["outcome"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{combo}: external outcome has fewer than two classes")

    missing_rows = missingness_qc(X, model_set, variable_set, algorithm)
    category_rows = category_qc(X, model_set, variable_set, algorithm)
    if any(row["has_unseen_level"] for row in category_rows):
        raise RuntimeError(f"{combo}: unseen categorical level detected")

    matrices = pipe.transformed_list(bundle["preprocessors"], X)
    column_match = all(list(mat.columns) == list(bundle["preprocessors"][i].columns_) for i, mat in enumerate(matrices))
    transform_missing = sum(int(np.isnan(mat.to_numpy()).sum()) for mat in matrices)
    if not column_match:
        raise RuntimeError(f"{combo}: transformed feature schema mismatch")
    if transform_missing:
        raise RuntimeError(f"{combo}: transformed matrix contains missing values")

    p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
    p_cal = bundle["calibrator"].transform(p_raw)
    pred_missing = int(np.isnan(p_raw).sum() + np.isnan(p_cal).sum())
    pred_out_of_range = int(((p_raw < 0) | (p_raw > 1) | (p_cal < 0) | (p_cal > 1)).sum())
    if pred_missing or pred_out_of_range:
        raise RuntimeError(f"{combo}: invalid prediction values")

    pred = df[["person_id", "outcome"]].copy()
    pred.insert(1, "source_dataset", "HSE 2022")
    pred["model_set"] = model_set
    pred["variable_set"] = variable_set
    pred["algorithm"] = algorithm
    pred["pred_raw"] = p_raw
    pred["pred_calibrated"] = p_cal
    pipe.write_csv(pred, EXTERNAL_DIR / f"{combo}__hse2022_predictions.csv")

    dca = pipe.dca_table(y, p_cal)
    dca.insert(0, "algorithm", algorithm)
    dca.insert(0, "variable_set", variable_set)
    dca.insert(0, "model_set", model_set)
    pipe.write_csv(dca, EXTERNAL_DIR / f"{combo}__hse2022_dca.csv")

    calib = pipe.calibration_plot_table(y, p_cal)
    calib.insert(0, "algorithm", algorithm)
    calib.insert(0, "variable_set", variable_set)
    calib.insert(0, "model_set", model_set)
    pipe.write_csv(calib, EXTERNAL_DIR / f"{combo}__hse2022_calibration_plot_data.csv")
    pipe.plot_calibration(
        calib,
        EXTERNAL_DIR / "plots" / f"{combo}__hse2022_calibration.png",
        f"{model_set} core_sdoh_hr {algorithm} HSE 2022",
    )

    summary = summarize_prediction(y, p_raw, p_cal, n_bootstrap, seed)
    metric_row = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "n": int(len(df)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        **summary,
    }

    threshold_rows = []
    for _, thr_row in bundle["thresholds"].iterrows():
        met = pipe.threshold_metrics(y, p_cal, float(thr_row["threshold"]))
        threshold_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "threshold_type": thr_row["threshold_type"],
                "development_threshold": float(thr_row["threshold"]),
                **met,
            }
        )

    dev_row = dev_metrics[
        (dev_metrics["model_set"] == model_set)
        & (dev_metrics["variable_set"] == variable_set)
        & (dev_metrics["algorithm"] == algorithm)
    ].iloc[0]
    temporal_row = temporal_metrics[
        (temporal_metrics["model_set"] == model_set)
        & (temporal_metrics["variable_set"] == variable_set)
        & (temporal_metrics["algorithm"] == algorithm)
    ].iloc[0]
    comparison_row = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "development_n": int(dev_row["n"]),
        "development_events": int(dev_row["events"]),
        "development_event_rate": float(dev_row["events"]) / float(dev_row["n"]),
        "temporal_n": int(temporal_row["n"]),
        "temporal_events": int(temporal_row["events"]),
        "temporal_event_rate": float(temporal_row["event_rate"]),
        "external_n": int(len(df)),
        "external_events": int(y.sum()),
        "external_event_rate": float(y.mean()),
        "development_auroc": float(dev_row["auroc_calibrated"]),
        "temporal_auroc": float(temporal_row["auroc"]),
        "external_auroc": summary["auroc"],
        "delta_auroc_external_minus_development": summary["auroc"] - float(dev_row["auroc_calibrated"]),
        "delta_auroc_external_minus_temporal": summary["auroc"] - float(temporal_row["auroc"]),
        "development_auprc": float(dev_row["auprc_calibrated"]),
        "temporal_auprc": float(temporal_row["auprc"]),
        "external_auprc": summary["auprc"],
        "delta_auprc_external_minus_development": summary["auprc"] - float(dev_row["auprc_calibrated"]),
        "delta_auprc_external_minus_temporal": summary["auprc"] - float(temporal_row["auprc"]),
        "development_brier": float(dev_row["brier_calibrated"]),
        "temporal_brier": float(temporal_row["brier"]),
        "external_brier": summary["brier"],
        "delta_brier_external_minus_development": summary["brier"] - float(dev_row["brier_calibrated"]),
        "delta_brier_external_minus_temporal": summary["brier"] - float(temporal_row["brier"]),
        "development_calibration_intercept": float(dev_row["calibration_intercept_calibrated"]),
        "temporal_calibration_intercept": float(temporal_row["calibration_intercept"]),
        "external_calibration_intercept": summary["calibration_intercept"],
        "development_calibration_slope": float(dev_row["calibration_slope_calibrated"]),
        "temporal_calibration_slope": float(temporal_row["calibration_slope"]),
        "external_calibration_slope": summary["calibration_slope"],
    }
    qc_row = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "bundle_path": str(bundle_path),
        "input_path": str(input_path(model_set)),
        "n": int(len(df)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "feature_count": len(features),
        "n_preprocessors": len(bundle["preprocessors"]),
        "n_models": len(bundle["models"]),
        "transformed_column_match": bool(column_match),
        "transformed_missing_count": int(transform_missing),
        "prediction_missing_count": int(pred_missing),
        "prediction_out_of_range_count": int(pred_out_of_range),
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


def write_report(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    comparison: pd.DataFrame,
    pred_qc: pd.DataFrame,
    miss: pd.DataFrame,
    category: pd.DataFrame,
) -> Path:
    mapping = pd.read_csv(MAPPING_PATH)
    flow = pd.read_csv(FLOW_PATH)
    compat = pd.read_csv(INPUT_COMPAT_PATH)
    usable = mapping.groupby("mapping_class")["knhanes_variable"].apply(lambda s: ", ".join(s)).reset_index()
    high_missing = miss[(miss["missing_rate"] >= 0.30) | (miss["all_missing"])].copy()
    high_missing["missing_percent"] = high_missing["missing_rate"] * 100
    report_path = REPORTS / f"hse_2022_external_validation_results_{STAMP}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# HSE 2022 External Validation - core_sdoh_hr\n\n")
        f.write("Scope: prediction-only external validation using frozen KNHANES 2014-2020 final bundles. No retraining, Optuna tuning, variable reselection, recalibration, or threshold reset was run.\n\n")
        f.write("## Frozen Models\n\n")
        f.write("- Integrated: `core_sdoh_hr + XGBoost`\n")
        f.write("- Male: `core_sdoh_hr + XGBoost`\n")
        f.write("- Female: `core_sdoh_hr + XGBoost`\n")
        f.write("- MICE/preprocessing/model ensemble: 20 frozen imputations/preprocessors and 20 frozen estimators per model set.\n")
        f.write("- Thresholds: frozen from 2014-2020 OOF predictions.\n\n")
        f.write("## HSE 2022 Cohort Flow\n\n")
        f.write(md_table(flow, ["step", "n_remaining", "n_excluded_step"], digits=0))
        f.write("\n\n## Model-Set Input Compatibility\n\n")
        f.write(md_table(compat, ["model_set", "n_before_predictor_missing_filter", "n_after_predictor_missing_filter", "n_excluded_predictor_missing_fraction_gt_50pct", "feature_count", "max_participant_missing_fraction_after_filter"]))
        f.write("\n\n## External Validation Metrics\n\n")
        f.write(md_table(metrics, ["model_set", "n", "events", "event_rate", "auroc", "auroc_ci_lower", "auroc_ci_upper", "auprc", "auprc_ci_lower", "auprc_ci_upper", "brier", "calibration_intercept", "calibration_slope", "mean_net_benefit_0.05_0.30"]))
        f.write("\n\n## Frozen Threshold Performance\n\n")
        f.write(md_table(thresholds, ["model_set", "threshold_type", "development_threshold", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "referral_rate"]))
        f.write("\n\n## Development vs Temporal vs HSE 2022\n\n")
        f.write(md_table(comparison, ["model_set", "development_auroc", "temporal_auroc", "external_auroc", "delta_auroc_external_minus_development", "development_auprc", "temporal_auprc", "external_auprc", "development_brier", "temporal_brier", "external_brier", "external_calibration_intercept", "external_calibration_slope"]))
        f.write("\n\n## Prediction QC\n\n")
        f.write(md_table(pred_qc, ["model_set", "n", "events", "feature_count", "n_preprocessors", "n_models", "transformed_column_match", "transformed_missing_count", "prediction_missing_count", "prediction_out_of_range_count", "all_na_features", "max_participant_missing_fraction"]))
        f.write("\n\n## High Missingness Predictors\n\n")
        f.write(md_table(high_missing, ["model_set", "predictor", "missing_n", "missing_percent", "all_missing"]))
        f.write("\n\n## Mapping Classes\n\n")
        f.write(md_table(usable, ["mapping_class", "knhanes_variable"]))
        f.write("\n\n## Harmonization Notes\n\n")
        f.write("- Age: HSE exact age is unavailable in the EUL file. The cohort is restricted to `age16g5>=3` (20+), and age bands are mapped to midpoints with 80+ top-coded to 80 to match KNHANES training scale.\n")
        f.write("- Heart rate: mean of `PULS2OM` and `PULS3OM`; pulse-pressure variables are not used.\n")
        f.write("- Binge drinking: `AUD6Mre` preserves the 0-5 frequency scale but uses a 6+ drink threshold, not the KNHANES sex-specific 7+/5+ threshold.\n")
        f.write("- `family_history_htn` and `food_security_status` are all-NA because no acceptable HSE direct/proxy variables were found; no substitution was made.\n")
        f.write("- Education has no separate middle-school category in `educ2`; below-degree is mapped to high-school level for frozen coding compatibility.\n")
        f.write("\n## Output Files\n\n")
        for path in [
            EXTERNAL_DIR / "external_validation_hse2022_metrics_selected_models.csv",
            EXTERNAL_DIR / "external_validation_hse2022_threshold_metrics_selected_models.csv",
            EXTERNAL_DIR / "external_validation_hse2022_development_temporal_external_comparison.csv",
            EXTERNAL_DIR / "external_validation_hse2022_prediction_qc.csv",
            EXTERNAL_DIR / "external_validation_hse2022_predictor_missingness_qc.csv",
            EXTERNAL_DIR / "external_validation_hse2022_category_level_qc.csv",
            EXTERNAL_DIR / "plots",
            DECISION_NOTES_PATH,
        ]:
            f.write(f"- `{path}`\n")
    return report_path


def main() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    manifest = load_manifest()
    dev_metrics = pd.read_csv(DEV_METRICS_PATH)
    temporal_metrics = pd.read_csv(TEMPORAL_METRICS_PATH)
    n_bootstrap = int(manifest.get("n_bootstrap", 2000))
    seed = int(manifest.get("seed", 20260722))

    metric_rows = []
    threshold_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    comparison_rows = []
    qc_rows = []

    selected_bundles = manifest["selected_bundles"]
    for i, model_set in enumerate(["integrated", "male", "female"]):
        metric, thr, miss, cat, comp, qc = validate_one(
            model_set,
            selected_bundles[model_set],
            dev_metrics,
            temporal_metrics,
            n_bootstrap,
            seed + i * 1000,
        )
        metric_rows.append(metric)
        threshold_rows.extend(thr)
        missing_rows.extend(miss)
        category_rows.extend(cat)
        comparison_rows.append(comp)
        qc_rows.append(qc)

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    missing = pd.DataFrame(missing_rows)
    category = pd.DataFrame(category_rows)
    comparison = pd.DataFrame(comparison_rows)
    pred_qc = pd.DataFrame(qc_rows)

    pipe.write_csv(metrics, EXTERNAL_DIR / "external_validation_hse2022_metrics_selected_models.csv")
    pipe.write_csv(thresholds, EXTERNAL_DIR / "external_validation_hse2022_threshold_metrics_selected_models.csv")
    pipe.write_csv(missing, EXTERNAL_DIR / "external_validation_hse2022_predictor_missingness_qc.csv")
    pipe.write_csv(category, EXTERNAL_DIR / "external_validation_hse2022_category_level_qc.csv")
    pipe.write_csv(comparison, EXTERNAL_DIR / "external_validation_hse2022_development_temporal_external_comparison.csv")
    pipe.write_csv(pred_qc, EXTERNAL_DIR / "external_validation_hse2022_prediction_qc.csv")

    run_config = {
        "dataset": "HSE 2022",
        "freeze_tag": FREEZE_TAG,
        "manifest_path": str(MANIFEST_PATH),
        "external_dir": str(EXTERNAL_DIR),
        "mapping_path": str(MAPPING_PATH),
        "input_compatibility_path": str(INPUT_COMPAT_PATH),
        "prediction_only": True,
        "retraining_executed": False,
        "retuning_executed": False,
        "variable_reselection_executed": False,
        "threshold_reset_executed": False,
        "calibrator_refit_executed": False,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
    pipe.save_json(run_config, EXTERNAL_DIR / "external_validation_hse2022_run_config.json")
    report_path = write_report(metrics, thresholds, comparison, pred_qc, missing, category)
    print("HSE 2022 external validation complete")
    print(metrics.to_string(index=False))
    print(f"report: {report_path}")
    print(f"output_dir: {EXTERNAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
