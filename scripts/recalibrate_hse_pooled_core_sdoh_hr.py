#!/usr/bin/env python3
"""Recalibrate frozen core_sdoh_hr predictions on the HSE 2018-2022 pooled external cohort.

Follows the protocol's pre-specified recalibration sequence (Sec 8.6):
  1. as-is: frozen KNHANES-development-fitted calibrator applied to HSE (already run).
  2. intercept-only: fix slope=1 on the logit scale, refit only the intercept
     using HSE outcomes (adjusts for prevalence/case-mix shift only).
  3. slope+intercept (full local logistic recalibration): refit both on HSE.

Recalibration is a monotonic transform of the raw model score, so it changes
calibration (Brier, calibration slope/intercept) but NEVER AUROC/AUPRC/ranking.
No retraining, retuning, variable reselection, or frozen-threshold value change
is performed; frozen threshold performance is only re-evaluated on the
recalibrated probabilities for reporting.
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
from scipy.optimize import minimize_scalar
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
POOLED_DIR = OUT / f"external_validation_hse_pooled_2018_2019_2021_2022_{FREEZE_TAG}_{STAMP}"
RECAL_DIR = OUT / f"external_validation_hse_pooled_recalibration_{STAMP}"
REPORTS = ROOT / "reports"
MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    c = np.clip(p, EPS, 1 - EPS)
    return np.log(c / (1 - c))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_intercept_only(y: np.ndarray, p_raw: np.ndarray) -> float:
    z = logit(p_raw)

    def neg_ll(a: float) -> float:
        pred = sigmoid(a + z)
        pred = np.clip(pred, EPS, 1 - EPS)
        return -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    res = minimize_scalar(neg_ll, bounds=(-10, 10), method="bounded")
    return float(res.x)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        if metric == "auroc":
            values.append(float(roc_auc_score(y_b, p_b)))
        elif metric == "brier":
            values.append(float(brier_score_loss(y_b, p_b)))
    if not values:
        return {"lower": math.nan, "upper": math.nan}
    arr = np.asarray(values)
    return {"lower": float(np.percentile(arr, 2.5)), "upper": float(np.percentile(arr, 97.5))}


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    cal = pipe.LogitCalibrator().fit(y, p)
    return cal.intercept_, cal.slope_


def stage_metrics(y: np.ndarray, p: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, Any]:
    intercept, slope = calibration_slope_intercept(y, p)
    dca = pipe.dca_table(y, p)
    mean_nb = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
    auroc_ci = bootstrap_ci(y, p, "auroc", n_bootstrap, seed)
    brier_ci = bootstrap_ci(y, p, "brier", n_bootstrap, seed + 17)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auroc_ci_lower": auroc_ci["lower"],
        "auroc_ci_upper": auroc_ci["upper"],
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_ci_lower": brier_ci["lower"],
        "brier_ci_upper": brier_ci["upper"],
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_net_benefit_0.05_0.30": mean_nb,
    }


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    RECAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    n_bootstrap = int(manifest.get("n_bootstrap", 2000))
    seed = int(manifest.get("seed", 20260722))
    selected_bundles = manifest["selected_bundles"]

    stage_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    recal_params: dict[str, Any] = {}

    for i, model_set in enumerate(["integrated", "male", "female"]):
        bundle_path = Path(selected_bundles[model_set]["bundle_path"])
        bundle = joblib.load(bundle_path)
        algorithm = str(bundle["algorithm"])
        pred_path = POOLED_DIR / f"{model_set}__core_sdoh_hr__{algorithm}__hse_pooled_predictions.csv"
        df = pd.read_csv(pred_path)
        y = df["outcome"].astype(int).to_numpy()
        p_raw = df["pred_raw"].to_numpy(dtype=float)
        p_asis = df["pred_calibrated"].to_numpy(dtype=float)  # frozen KNHANES-fitted calibrator, already reported

        # Stage 2: intercept-only local recalibration (slope fixed at 1 on logit scale).
        a_only = fit_intercept_only(y, p_raw)
        p_intercept_only = sigmoid(a_only + logit(p_raw))

        # Stage 3: full local logistic (intercept + slope) recalibration.
        local_cal = pipe.LogitCalibrator().fit(y, p_raw)
        p_full = local_cal.transform(p_raw)

        seed_i = seed + i * 1000
        for stage_name, p_stage in [
            ("1_as_is_knhanes_calibrator", p_asis),
            ("2_intercept_only_hse_pooled", p_intercept_only),
            ("3_intercept_slope_hse_pooled", p_full),
        ]:
            m = stage_metrics(y, p_stage, n_bootstrap, seed_i)
            stage_rows.append({"model_set": model_set, "n": int(len(df)), "events": int(y.sum()), "stage": stage_name, **m})

            df_out = df[["person_id", "source_year", "outcome"]].copy()
            df_out["pred"] = p_stage
            pipe.write_csv(df_out, RECAL_DIR / f"{model_set}__{stage_name}__predictions.csv")

            # Frozen threshold performance re-evaluated on this stage's probabilities.
            for _, thr_row in bundle["thresholds"].iterrows():
                met = pipe.threshold_metrics(y, p_stage, float(thr_row["threshold"]))
                threshold_rows.append({
                    "model_set": model_set, "stage": stage_name,
                    "threshold_type": thr_row["threshold_type"], "frozen_threshold_value": float(thr_row["threshold"]),
                    **met,
                })

        recal_params[model_set] = {
            "intercept_only_a": a_only,
            "full_intercept": local_cal.intercept_,
            "full_slope": local_cal.slope_,
        }

    stages = pd.DataFrame(stage_rows)
    thresholds = pd.DataFrame(threshold_rows)
    pipe.write_csv(stages, RECAL_DIR / "hse_pooled_recalibration_stage_metrics.csv")
    pipe.write_csv(thresholds, RECAL_DIR / "hse_pooled_recalibration_threshold_metrics.csv")
    pipe.save_json(recal_params, RECAL_DIR / "hse_pooled_recalibration_parameters.json")

    def md_table(d: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
        v = d[cols].copy()
        for c in v.columns:
            if pd.api.types.is_float_dtype(v[c]):
                v[c] = v[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in v.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    report_path = REPORTS / f"hse_pooled_recalibration_results_{STAMP}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# HSE 2018-2022 Pooled Recalibration - core_sdoh_hr\n\n")
        f.write("Scope: recalibration only, using the already-executed pooled external prediction (`pred_raw`). No retraining, retuning, variable reselection, or change to the frozen threshold VALUES was performed; frozen threshold performance is re-evaluated on recalibrated probabilities for reporting only.\n\n")
        f.write("## Stages\n\n")
        f.write("1. `1_as_is_knhanes_calibrator`: frozen KNHANES-development-fitted calibrator applied as-is (already reported in the pooled external validation run).\n")
        f.write("2. `2_intercept_only_hse_pooled`: slope fixed at 1 (logit scale), intercept refit on HSE pooled outcomes only (adjusts for prevalence/case-mix shift).\n")
        f.write("3. `3_intercept_slope_hse_pooled`: full local logistic recalibration (intercept and slope both refit on HSE pooled outcomes).\n\n")
        f.write("## Stage Metrics\n\n")
        f.write(md_table(stages, ["model_set", "stage", "n", "events", "auroc", "auroc_ci_lower", "auroc_ci_upper", "brier", "brier_ci_lower", "brier_ci_upper", "calibration_intercept", "calibration_slope", "mean_net_benefit_0.05_0.30"]))
        f.write("\n\n## Recalibration Parameters\n\n")
        f.write("```json\n" + json.dumps(recal_params, indent=2) + "\n```\n\n")
        f.write("## Frozen-Threshold Performance by Stage\n\n")
        f.write(md_table(thresholds, ["model_set", "stage", "threshold_type", "frozen_threshold_value", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "referral_rate"]))
        f.write("\n\n## Interpretation Notes\n\n")
        f.write("- AUROC is identical across all three stages for a given model_set by construction (recalibration is a monotonic transform); any tiny differences beyond rounding indicate a bug, not real discrimination change.\n")
        f.write("- Stage 3 calibration slope/intercept are approximately 1/0 by construction (it is fit on this exact sample); Stage 2's residual slope deviation from 1 shows how much of the miscalibration is shape (slope), not just level (intercept).\n")
        f.write("- Frozen thresholds were never changed; only how many people they flag (sensitivity/specificity/referral_rate) shifts as the underlying probabilities are recalibrated.\n")
    print(stages.to_string(index=False))
    print(f"report: {report_path}")
    print(f"output_dir: {RECAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
