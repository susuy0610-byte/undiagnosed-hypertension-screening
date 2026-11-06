#!/usr/bin/env python3
"""Prediction-only HSE 2018+2019 (pre-COVID pooled) external validation for frozen core_sdoh_hr models.

Pre-specified sensitivity split (pre-COVID vs COVID-era HSE years), not chosen
based on which individual year scored highest. Uses frozen KNHANES 2014-2020
final model bundles. Does not retrain, retune, reselect variables, recalibrate,
or reset thresholds.
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
TAG = "precovid_2018_2019"
FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"
OUT = ROOT / "analysis/knhanes_primary_ml_v1"
FINAL_MODELS = OUT / "final_models"
EXTERNAL_DIR = OUT / f"external_validation_hse_{TAG}_{FREEZE_TAG}_{STAMP}"
REPORTS = ROOT / "reports"
INPUTS = ROOT / "data/processed"

MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
DEV_METRICS_PATH = FINAL_MODELS / f"{FREEZE_TAG}_selected_models.csv"
TEMPORAL_DIR = OUT / f"time_validation_{FREEZE_TAG}"
TEMPORAL_METRICS_PATH = TEMPORAL_DIR / "time_validation_metrics_selected_models.csv"


def bootstrap_ci(y, p, metric, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        if metric == "auroc":
            values.append(float(roc_auc_score(y_b, p_b)))
        elif metric == "auprc":
            values.append(float(average_precision_score(y_b, p_b)))
    if not values:
        return {"lower": math.nan, "upper": math.nan}
    arr = np.asarray(values)
    return {"lower": float(np.percentile(arr, 2.5)), "upper": float(np.percentile(arr, 97.5))}


def input_path(model_set: str) -> Path:
    return INPUTS / f"hse_{TAG}_pooled_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def validate_one(model_set, selected, dev_metrics, temporal_metrics, n_bootstrap, seed):
    bundle_path = Path(selected["bundle_path"])
    bundle = joblib.load(bundle_path)
    variable_set, algorithm = str(bundle["variable_set"]), str(bundle["algorithm"])
    features = list(bundle["features"])
    assert variable_set == "core_sdoh_hr"
    assert model_set == str(bundle["model_set"])

    df = pd.read_csv(input_path(model_set))
    if df["person_id"].duplicated().any():
        raise RuntimeError(f"{model_set}: duplicate person_id")
    X = df[features]
    y = df["outcome"].astype(int).to_numpy()

    matrices = pipe.transformed_list(bundle["preprocessors"], X)
    column_match = all(list(mat.columns) == list(bundle["preprocessors"][i].columns_) for i, mat in enumerate(matrices))
    transform_missing = sum(int(np.isnan(mat.to_numpy()).sum()) for mat in matrices)
    if not column_match or transform_missing:
        raise RuntimeError(f"{model_set}: transform schema mismatch or missing values")

    p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
    p_cal = bundle["calibrator"].transform(p_raw)

    cal_intercept, cal_slope = pipe.calibration_intercept_slope(y, p_cal)
    dca = pipe.dca_table(y, p_cal)
    mean_nb = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
    auroc_ci = bootstrap_ci(y, p_cal, "auroc", n_bootstrap, seed)

    metric_row = {
        "model_set": model_set, "n": int(len(df)), "events": int(y.sum()), "event_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p_cal)), "auroc_ci_lower": auroc_ci["lower"], "auroc_ci_upper": auroc_ci["upper"],
        "auprc": float(average_precision_score(y, p_cal)), "brier": float(brier_score_loss(y, p_cal)),
        "calibration_intercept": cal_intercept, "calibration_slope": cal_slope, "mean_net_benefit_0.05_0.30": mean_nb,
    }

    dev_row = dev_metrics[(dev_metrics["model_set"] == model_set) & (dev_metrics["variable_set"] == variable_set) & (dev_metrics["algorithm"] == algorithm)].iloc[0]
    temporal_row = temporal_metrics[(temporal_metrics["model_set"] == model_set) & (temporal_metrics["variable_set"] == variable_set) & (temporal_metrics["algorithm"] == algorithm)].iloc[0]
    comparison_row = {
        "model_set": model_set, "development_auroc": float(dev_row["auroc_calibrated"]), "temporal_auroc": float(temporal_row["auroc"]),
        "external_auroc": metric_row["auroc"], "delta_auroc_external_minus_development": metric_row["auroc"] - float(dev_row["auroc_calibrated"]),
    }

    pred = df[["person_id", "source_year", "outcome"]].copy()
    pred["pred_raw"] = p_raw
    pred["pred_calibrated"] = p_cal
    pipe.write_csv(pred, EXTERNAL_DIR / f"{model_set}__core_sdoh_hr__{algorithm}__hse_{TAG}_predictions.csv")
    return metric_row, comparison_row


def main() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    dev_metrics = pd.read_csv(DEV_METRICS_PATH)
    temporal_metrics = pd.read_csv(TEMPORAL_METRICS_PATH)
    n_bootstrap = int(manifest.get("n_bootstrap", 2000))
    seed = int(manifest.get("seed", 20260722))
    selected_bundles = manifest["selected_bundles"]

    metric_rows, comparison_rows = [], []
    for i, model_set in enumerate(["integrated", "male", "female"]):
        metric, comp = validate_one(model_set, selected_bundles[model_set], dev_metrics, temporal_metrics, n_bootstrap, seed + i * 1000)
        metric_rows.append(metric)
        comparison_rows.append(comp)

    metrics = pd.DataFrame(metric_rows)
    comparison = pd.DataFrame(comparison_rows)
    pipe.write_csv(metrics, EXTERNAL_DIR / f"external_validation_hse_{TAG}_metrics.csv")
    pipe.write_csv(comparison, EXTERNAL_DIR / f"external_validation_hse_{TAG}_dev_temporal_comparison.csv")

    print(f"HSE {TAG} external validation complete")
    print(metrics.to_string(index=False))
    print(f"output_dir: {EXTERNAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
