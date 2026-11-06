#!/usr/bin/env python3
"""
Temporal-validation-only runner for frozen KNHANES final model bundles.

This script loads the final 2014-2020 bundles and applies their frozen
preprocessors, MICE imputers, estimators, OOF calibrators, and thresholds to
KNHANES 2021-2022. It does not refit, tune, select variables, or change
thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402


SELECTED_BUNDLES = {
    "integrated": ("core_sdoh_hr", "xgboost"),
    "male": ("core_sdoh_hr", "lasso"),
    "female": ("core_sdoh_hr", "xgboost"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen KNHANES final models to 2021-2022 only.")
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "knhanes_primary_ml_v1"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def setup_logging(out: Path) -> Path:
    log_path = out / "logs" / f"temporal_validation_only_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)
    return log_path


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str, n_bootstrap: int, seed: int) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    values = []
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


def category_qc(df: pd.DataFrame, features: list[str], model_set: str, variable_set: str, algorithm: str) -> list[dict[str, Any]]:
    rows = []
    for col in features:
        if col in pipe.CONTINUOUS:
            continue
        valid = set(pipe.VALID_CATEGORIES[col])
        observed = df[col].dropna()
        if observed.empty:
            levels: list[Any] = []
        else:
            levels = sorted(pd.Series(np.rint(observed.astype(float)).astype(int)).unique().tolist())
        unseen = [x for x in levels if x not in valid]
        rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "predictor": col,
                "allowed_levels": json.dumps(sorted(valid), ensure_ascii=False),
                "observed_levels": json.dumps(levels, ensure_ascii=False),
                "unseen_levels": json.dumps(unseen, ensure_ascii=False),
                "has_unseen_level": bool(unseen),
            }
        )
    return rows


def missingness_qc(df: pd.DataFrame, features: list[str], model_set: str, variable_set: str, algorithm: str) -> list[dict[str, Any]]:
    n = len(df)
    rows = []
    for col in features:
        missing_n = int(df[col].isna().sum())
        rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "predictor": col,
                "n": n,
                "missing_n": missing_n,
                "missing_rate": missing_n / n if n else math.nan,
            }
        )
    return rows


def summarize_prediction(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, Any]:
    raw_intercept, raw_slope = pipe.calibration_intercept_slope(y, p_raw)
    cal_intercept, cal_slope = pipe.calibration_intercept_slope(y, p_cal)
    auroc = float(roc_auc_score(y, p_cal))
    auprc = float(average_precision_score(y, p_cal))
    auroc_ci = bootstrap_ci(y, p_cal, "auroc", n_bootstrap, seed)
    auprc_ci = bootstrap_ci(y, p_cal, "auprc", n_bootstrap, seed + 17)
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


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    pipe.ensure_dirs(out)
    tv_dir = out / "time_validation"
    tv_dir.mkdir(parents=True, exist_ok=True)
    (tv_dir / "plots").mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out)

    time_df = pipe.load_dataset(pipe.TIME_PATH)
    metrics_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    prediction_qc_rows: list[dict[str, Any]] = []

    dev_metrics = pd.read_csv(out / "metrics" / "oof_metrics_all_models.csv")

    for idx, (model_set, (variable_set, algorithm)) in enumerate(SELECTED_BUNDLES.items()):
        combo = pipe.model_combo_id(model_set, variable_set, algorithm)
        bundle_path = out / "final_models" / f"{combo}__final_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing frozen bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)

        if bundle["model_set"] != model_set or bundle["variable_set"] != variable_set or bundle["algorithm"] != algorithm:
            raise RuntimeError(f"Bundle identity mismatch for {bundle_path}")

        expected_features = pipe.predictors_for(model_set, variable_set)
        if list(bundle["features"]) != expected_features:
            raise RuntimeError(f"Frozen feature list mismatch for {combo}")
        for i, preprocessor in enumerate(bundle["preprocessors"]):
            if list(preprocessor.features) != expected_features:
                raise RuntimeError(f"Preprocessor feature mismatch for {combo} imputation {i}")

        df = pipe.subset_for_model_set(time_df, model_set)
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f"Missing predictors for {combo}: {missing_cols}")

        X = df[expected_features]
        y = df["undiagnosed_htn"].astype(int).to_numpy()
        missingness_rows.extend(missingness_qc(X, expected_features, model_set, variable_set, algorithm))
        category_rows.extend(category_qc(X, expected_features, model_set, variable_set, algorithm))

        matrices = pipe.transformed_list(bundle["preprocessors"], X)
        column_match = all(list(mat.columns) == list(bundle["preprocessors"][i].columns_) for i, mat in enumerate(matrices))
        if not column_match:
            raise RuntimeError(f"Transformed matrix column mismatch for {combo}")

        p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
        p_cal = bundle["calibrator"].transform(p_raw)
        pred_missing = int(np.isnan(p_cal).sum() + np.isnan(p_raw).sum())
        pred_out_of_range = int(((p_cal < 0) | (p_cal > 1) | (p_raw < 0) | (p_raw > 1)).sum())
        prediction_qc_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "n": int(len(df)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "prediction_missing_count": pred_missing,
                "prediction_out_of_range_count": pred_out_of_range,
                "transformed_column_match": bool(column_match),
                "n_preprocessors": len(bundle["preprocessors"]),
                "n_models": len(bundle["models"]),
            }
        )
        if pred_missing or pred_out_of_range:
            raise RuntimeError(f"Invalid predictions for {combo}")

        pred_out = df[["participant_id", "source_year", "sex", "weight", "strata", "psu", "undiagnosed_htn"]].copy()
        pred_out["model_set"] = model_set
        pred_out["variable_set"] = variable_set
        pred_out["algorithm"] = algorithm
        pred_out["pred_raw"] = p_raw
        pred_out["pred_calibrated"] = p_cal
        pipe.write_csv(pred_out, tv_dir / f"{combo}__2021_2022_predictions.csv")

        summary = summarize_prediction(y, p_raw, p_cal, args.n_bootstrap, args.seed + idx * 1009)
        metrics_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "n": int(len(df)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                **summary,
            }
        )

        thresholds = bundle["thresholds"].copy()
        for _, thr_row in thresholds.iterrows():
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

        dca = pipe.dca_table(y, p_cal)
        dca.insert(0, "algorithm", algorithm)
        dca.insert(0, "variable_set", variable_set)
        dca.insert(0, "model_set", model_set)
        pipe.write_csv(dca, tv_dir / f"{combo}__2021_2022_dca.csv")

        calib = pipe.calibration_plot_table(y, p_cal)
        calib.insert(0, "algorithm", algorithm)
        calib.insert(0, "variable_set", variable_set)
        calib.insert(0, "model_set", model_set)
        pipe.write_csv(calib, tv_dir / f"{combo}__2021_2022_calibration_plot_data.csv")
        pipe.plot_calibration(
            calib,
            tv_dir / "plots" / f"{combo}__2021_2022_calibration.png",
            f"{pipe.LABELS[model_set]} {pipe.LABELS[variable_set]} {pipe.LABELS[algorithm]} temporal",
        )

        dev_row = dev_metrics[
            (dev_metrics["model_set"] == model_set)
            & (dev_metrics["variable_set"] == variable_set)
            & (dev_metrics["algorithm"] == algorithm)
        ].iloc[0]
        comparison_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "development_n": int(dev_row["n"]),
                "development_events": int(dev_row["events"]),
                "temporal_n": int(len(df)),
                "temporal_events": int(y.sum()),
                "development_auroc": float(dev_row["auroc_calibrated"]),
                "temporal_auroc": summary["auroc"],
                "delta_auroc_temporal_minus_development": summary["auroc"] - float(dev_row["auroc_calibrated"]),
                "development_auprc": float(dev_row["auprc_calibrated"]),
                "temporal_auprc": summary["auprc"],
                "delta_auprc_temporal_minus_development": summary["auprc"] - float(dev_row["auprc_calibrated"]),
                "development_brier": float(dev_row["brier_calibrated"]),
                "temporal_brier": summary["brier"],
                "delta_brier_temporal_minus_development": summary["brier"] - float(dev_row["brier_calibrated"]),
                "development_calibration_intercept": float(dev_row["calibration_intercept_calibrated"]),
                "temporal_calibration_intercept": summary["calibration_intercept"],
                "delta_calibration_intercept_temporal_minus_development": summary["calibration_intercept"]
                - float(dev_row["calibration_intercept_calibrated"]),
                "development_calibration_slope": float(dev_row["calibration_slope_calibrated"]),
                "temporal_calibration_slope": summary["calibration_slope"],
                "delta_calibration_slope_temporal_minus_development": summary["calibration_slope"]
                - float(dev_row["calibration_slope_calibrated"]),
            }
        )
        logging.info("Temporal validation complete for %s", combo)

    pipe.write_csv(pd.DataFrame(metrics_rows), tv_dir / "time_validation_metrics_selected_models.csv")
    pipe.write_csv(pd.DataFrame(threshold_rows), tv_dir / "time_validation_threshold_metrics_selected_models.csv")
    pipe.write_csv(pd.DataFrame(comparison_rows), tv_dir / "time_validation_development_comparison.csv")
    pipe.write_csv(pd.DataFrame(missingness_rows), tv_dir / "time_validation_predictor_missingness_qc.csv")
    pipe.write_csv(pd.DataFrame(category_rows), tv_dir / "time_validation_category_level_qc.csv")
    pipe.write_csv(pd.DataFrame(prediction_qc_rows), tv_dir / "time_validation_prediction_qc.csv")
    pipe.save_json(
        {
            "script": str(Path(__file__).resolve()),
            "stage": "temporal-validation-only",
            "output_dir": str(out),
            "time_validation_path": str(pipe.TIME_PATH),
            "selected_bundles": SELECTED_BUNDLES,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "log_path": str(log_path),
            "refit_or_tuning_executed": False,
            "external_validation_executed": False,
        },
        tv_dir / "time_validation_run_config.json",
    )
    logging.info("Temporal validation-only complete")


if __name__ == "__main__":
    main()
