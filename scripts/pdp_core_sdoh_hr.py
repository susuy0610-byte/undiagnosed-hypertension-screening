#!/usr/bin/env python3
"""Partial Dependence Plots for the frozen core_sdoh_hr models (age, bmi, waist_cm).

For each model_set (integrated/male/female) and each target predictor, sweep a
grid of values, override that column for every KNHANES 2014-2020 development
participant (holding all other predictors at their observed values), run the
SAME frozen preprocessors + models used in development/external validation,
and average the predicted probability across participants at each grid point.

Prediction-only diagnostic: no retraining, retuning, or frozen-model change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402

FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"
OUT = ROOT / "analysis/knhanes_primary_ml_v1"
FINAL_MODELS = OUT / "final_models"
PDP_DIR = OUT / "pdp_core_sdoh_hr_20260820"
REPORTS = ROOT / "reports"
MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
DEV_PATH = ROOT / "data/processed/knhanes_development_2014_2020_primary.csv"

TARGET_VARS = {
    "age": {"low_pct": 1, "high_pct": 99, "n_points": 15},
    "bmi": {"low_pct": 1, "high_pct": 99, "n_points": 15},
    "waist_cm": {"low_pct": 1, "high_pct": 99, "n_points": 15},
}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def pdp_for_variable(bundle: dict, X: pd.DataFrame, var: str, grid: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for val in grid:
        X_mod = X.copy()
        X_mod[var] = val
        matrices = pipe.transformed_list(bundle["preprocessors"], X_mod)
        p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
        p_cal = bundle["calibrator"].transform(p_raw)
        rows.append({
            "grid_value": float(val),
            "mean_pred_raw": float(np.mean(p_raw)),
            "mean_pred_calibrated": float(np.mean(p_cal)),
            "std_pred_calibrated": float(np.std(p_cal)),
        })
    return rows


def main() -> int:
    PDP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    selected_bundles = manifest["selected_bundles"]
    dev = pd.read_csv(DEV_PATH)

    all_rows = []
    for model_set in ["integrated", "male", "female"]:
        bundle_path = Path(selected_bundles[model_set]["bundle_path"])
        bundle = joblib.load(bundle_path)
        features = list(bundle["features"])
        assert bundle["variable_set"] == "core_sdoh_hr"
        assert bundle["model_set"] == model_set

        if model_set == "integrated":
            X_full = dev.copy()
        elif model_set == "male":
            X_full = dev[dev["sex"] == 1].copy()
        else:
            X_full = dev[dev["sex"] == 2].copy()
        X = X_full[features]

        for var, cfg in TARGET_VARS.items():
            observed = X[var].dropna()
            low = np.percentile(observed, cfg["low_pct"])
            high = np.percentile(observed, cfg["high_pct"])
            grid = np.linspace(low, high, cfg["n_points"])
            print(f"[{model_set}] PDP for {var}: grid {low:.1f}-{high:.1f} ({cfg['n_points']} points), n={len(X)}")
            rows = pdp_for_variable(bundle, X, var, grid)
            for r in rows:
                r["model_set"] = model_set
                r["variable"] = var
                r["algorithm"] = bundle["algorithm"]
                r["n"] = len(X)
            all_rows.extend(rows)
            df_var = pd.DataFrame(rows)
            pipe.write_csv(df_var, PDP_DIR / f"{model_set}__{var}__pdp.csv")

    full = pd.DataFrame(all_rows)
    pipe.write_csv(full, PDP_DIR / "pdp_all_groups_all_variables.csv")

    report_path = REPORTS / "pdp_core_sdoh_hr_results_20260820.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Partial Dependence Plots - core_sdoh_hr (age, bmi, waist_cm)\n\n")
        f.write("Scope: prediction-only diagnostic using the frozen KNHANES 2014-2020 final bundles (development cohort, same participants used for development OOF). No retraining, retuning, or frozen-model change.\n\n")
        for model_set in ["integrated", "male", "female"]:
            f.write(f"## {model_set}\n\n")
            for var in TARGET_VARS:
                sub = full[(full.model_set == model_set) & (full.variable == var)][["grid_value", "mean_pred_calibrated"]]
                f.write(f"### {var}\n\n")
                f.write("| grid_value | mean_pred_calibrated |\n| --- | --- |\n")
                for _, r in sub.iterrows():
                    f.write(f"| {r['grid_value']:.2f} | {r['mean_pred_calibrated']:.4f} |\n")
                f.write("\n")
        f.write("## Output Files\n\n")
        f.write(f"- Full PDP table: `{PDP_DIR.relative_to(ROOT)}/pdp_all_groups_all_variables.csv`\n")

    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
