#!/usr/bin/env python3
"""Freeze core_sdoh_hr best single models and run temporal validation only.

Primary decision for 2026-08-13:
- predictor set: core_sdoh_hr
- no ensemble/stacking for primary
- select the best single algorithm within each analysis population by existing
  2014-2020 OOF AUROC

This script refits only missing/mismatched final bundles on the full 2014-2020
development data with the already selected hyperparameters. It does not run CV,
Optuna tuning, variable selection, NHANES external validation, or any threshold
re-estimation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
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


VARIABLE_SET = "core_sdoh_hr"
MODEL_SETS = ["integrated", "male", "female"]
FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze primary core_sdoh_hr best single models and run temporal validation.")
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "knhanes_primary_ml_v1"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--m-imputations", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite-bundles", action="store_true")
    return parser.parse_args()


def setup_logging(out: Path) -> Path:
    log_path = out / "logs" / f"{FREEZE_TAG}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)
    return log_path


def choose_best_single_models(out: Path) -> pd.DataFrame:
    metrics = pd.read_csv(out / "metrics" / "oof_metrics_all_models.csv")
    part = metrics[metrics["variable_set"].eq(VARIABLE_SET)].copy()
    if part.empty:
        raise RuntimeError(f"No OOF metrics found for {VARIABLE_SET}")
    selected = []
    for model_set in MODEL_SETS:
        rows = part[part["model_set"].eq(model_set)].copy()
        if rows.empty:
            raise RuntimeError(f"No OOF metrics found for {model_set}/{VARIABLE_SET}")
        rows = rows[~rows["algorithm"].astype(str).str.contains("benchmark", case=False, na=False)]
        best = rows.sort_values(["auroc_raw", "auprc_raw", "brier_raw"], ascending=[False, False, True]).iloc[0].copy()
        selected.append(best)
    return pd.DataFrame(selected).reset_index(drop=True)


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


def aggregate_existing_params(out: Path, model_set: str, variable_set: str, algorithm: str) -> dict[str, Any]:
    best_paths = sorted((out / "tuning").glob(f"{model_set}__{variable_set}__{algorithm}__fold*__best.json"))
    if not best_paths:
        return {}
    params_by_fold = [pipe.read_json(p)["params"] for p in best_paths]
    return pipe.aggregate_params(params_by_fold)


def bundle_qc(
    bundle: dict[str, Any],
    model_set: str,
    variable_set: str,
    algorithm: str,
    features: list[str],
    params: dict[str, Any],
    thresholds: pd.DataFrame,
    m_imputations: int,
) -> tuple[bool, list[str]]:
    issues = []
    if bundle.get("model_set") != model_set:
        issues.append("model_set mismatch")
    if bundle.get("variable_set") != variable_set:
        issues.append("variable_set mismatch")
    if bundle.get("algorithm") != algorithm:
        issues.append("algorithm mismatch")
    if list(bundle.get("features", [])) != features:
        issues.append("feature list mismatch")
    if len(bundle.get("preprocessors", [])) != m_imputations:
        issues.append("preprocessor count mismatch")
    if len(bundle.get("models", [])) != m_imputations:
        issues.append("model count mismatch")
    for idx, preprocessor in enumerate(bundle.get("preprocessors", [])):
        if list(getattr(preprocessor, "features", [])) != features:
            issues.append(f"preprocessor {idx} features mismatch")
            break
    if dict(bundle.get("params", {})) != dict(params):
        issues.append("hyperparameter mismatch")
    if "calibrator" not in bundle:
        issues.append("missing calibrator")
    if "thresholds" not in bundle:
        issues.append("missing thresholds")
    else:
        b_thr = bundle["thresholds"][["threshold_type", "threshold"]].reset_index(drop=True)
        e_thr = thresholds[["threshold_type", "threshold"]].reset_index(drop=True)
        if len(b_thr) != len(e_thr) or not np.allclose(b_thr["threshold"].astype(float), e_thr["threshold"].astype(float)):
            issues.append("threshold mismatch")
        if b_thr["threshold_type"].astype(str).tolist() != e_thr["threshold_type"].astype(str).tolist():
            issues.append("threshold type mismatch")
    return not issues, issues


def refit_or_load_bundle(
    out: Path,
    dev: pd.DataFrame,
    model_set: str,
    variable_set: str,
    algorithm: str,
    seed: int,
    m_imputations: int,
    n_jobs: int,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    combo = pipe.model_combo_id(model_set, variable_set, algorithm)
    bundle_path = out / "final_models" / f"{combo}__final_bundle.joblib"
    config_path = out / "final_models" / f"{combo}__final_config.json"
    features = pipe.predictors_for(model_set, variable_set)
    params = aggregate_existing_params(out, model_set, variable_set, algorithm)
    calibrator = pipe.calibrator_lookup(out, model_set, variable_set, algorithm)
    thresholds = pipe.threshold_lookup(out, model_set, variable_set, algorithm)
    train_df = pipe.subset_for_model_set(dev, model_set)
    y_train = train_df["undiagnosed_htn"].astype(int).to_numpy()

    status = "loaded_existing"
    issues: list[str] = []
    bundle: dict[str, Any] | None = None
    if bundle_path.exists() and not overwrite:
        bundle = joblib.load(bundle_path)
        ok, issues = bundle_qc(bundle, model_set, variable_set, algorithm, features, params, thresholds, m_imputations)
        if not ok:
            logging.info("Existing bundle failed QC for %s: %s", combo, "; ".join(issues))
            bundle = None
            status = "refit_after_qc_mismatch"
    else:
        status = "refit_missing_or_overwrite"

    if bundle is None:
        X_train = train_df[features]
        preprocessors = pipe.fit_preprocessors(
            X_train,
            features,
            m_imputations,
            seed + 9000,
            pipe.rcs_variables_for(variable_set),
        )
        matrices_train = pipe.transformed_list(preprocessors, X_train)
        models = pipe.fit_model_ensemble(algorithm, params, matrices_train, y_train, seed + 9999, n_jobs)
        bundle = {
            "model_set": model_set,
            "variable_set": variable_set,
            "algorithm": algorithm,
            "features": features,
            "rcs_variables": pipe.rcs_variables_for(variable_set),
            "rcs_knot_probs": list(pipe.RCS_KNOT_PROBS),
            "params": params,
            "preprocessors": preprocessors,
            "models": models,
            "calibrator": calibrator,
            "thresholds": thresholds,
            "locked_variables_path": str(pipe.LOCKED_VARIABLES_PATH),
            "primary_freeze_tag": FREEZE_TAG,
            "primary_freeze_decision": "best_single_model_by_existing_2014_2020_oof_auroc",
        }
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, bundle_path, compress=3)
        pipe.save_json(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "features": features,
                "rcs_variables": pipe.rcs_variables_for(variable_set),
                "rcs_knot_probs": list(pipe.RCS_KNOT_PROBS),
                "params": params,
                "primary_freeze_tag": FREEZE_TAG,
                "primary_freeze_decision": "best_single_model_by_existing_2014_2020_oof_auroc",
            },
            config_path,
        )
        logging.info("Saved final bundle: %s", bundle_path)

    ok, post_issues = bundle_qc(bundle, model_set, variable_set, algorithm, features, params, thresholds, m_imputations)
    if not ok:
        raise RuntimeError(f"Final bundle QC failed for {combo}: {post_issues}")

    qc_row = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "bundle_status": status,
        "bundle_path": str(bundle_path),
        "config_path": str(config_path),
        "n": int(len(train_df)),
        "events": int(y_train.sum()),
        "event_rate": float(y_train.mean()),
        "n_features": int(len(features)),
        "features": json.dumps(features, ensure_ascii=False),
        "params": json.dumps(params, ensure_ascii=False),
        "n_preprocessors": int(len(bundle["preprocessors"])),
        "n_models": int(len(bundle["models"])),
        "thresholds": thresholds[["threshold_type", "threshold"]].to_json(orient="records", force_ascii=False),
        "qc_passed": True,
        "qc_issues": json.dumps(post_issues, ensure_ascii=False),
    }
    return bundle, qc_row


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


def validate_temporal(
    out: Path,
    tv_dir: Path,
    time_df: pd.DataFrame,
    dev_metrics: pd.DataFrame,
    bundle: dict[str, Any],
    model_set: str,
    variable_set: str,
    algorithm: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    combo = pipe.model_combo_id(model_set, variable_set, algorithm)
    features = list(bundle["features"])
    df = pipe.subset_for_model_set(time_df, model_set)
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing predictors for {combo}: {missing_cols}")
    X = df[features]
    y = df["undiagnosed_htn"].astype(int).to_numpy()
    missingness_rows = missingness_qc(X, features, model_set, variable_set, algorithm)
    category_rows = category_qc(X, features, model_set, variable_set, algorithm)

    matrices = pipe.transformed_list(bundle["preprocessors"], X)
    column_match = all(
        list(mat.columns) == list(bundle["preprocessors"][i].columns_)
        for i, mat in enumerate(matrices)
    )
    if not column_match:
        raise RuntimeError(f"Transformed matrix column mismatch for {combo}")

    p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
    p_cal = bundle["calibrator"].transform(p_raw)
    pred_missing = int(np.isnan(p_cal).sum() + np.isnan(p_raw).sum())
    pred_out_of_range = int(((p_cal < 0) | (p_cal > 1) | (p_raw < 0) | (p_raw > 1)).sum())
    prediction_qc = {
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
    if pred_missing or pred_out_of_range:
        raise RuntimeError(f"Invalid predictions for {combo}")

    pred_out = df[["participant_id", "source_year", "sex", "weight", "strata", "psu", "undiagnosed_htn"]].copy()
    pred_out["model_set"] = model_set
    pred_out["variable_set"] = variable_set
    pred_out["algorithm"] = algorithm
    pred_out["pred_raw"] = p_raw
    pred_out["pred_calibrated"] = p_cal
    pipe.write_csv(pred_out, tv_dir / f"{combo}__2021_2022_predictions.csv")

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
    comparison_row = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "development_n": int(dev_row["n"]),
        "development_events": int(dev_row["events"]),
        "development_event_rate": float(dev_row["events"]) / float(dev_row["n"]),
        "temporal_n": int(len(df)),
        "temporal_events": int(y.sum()),
        "temporal_event_rate": float(y.mean()),
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
    return metric_row, threshold_rows, missingness_rows, category_rows, prediction_qc, comparison_row


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    headers = [str(c) for c in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    selected: pd.DataFrame,
    bundle_qc_df: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    prediction_qc: pd.DataFrame,
) -> None:
    selected_cols = ["model_set", "variable_set", "algorithm", "n", "events", "auroc_raw", "auprc_raw", "brier_raw"]
    temporal_cols = [
        "model_set",
        "algorithm",
        "n",
        "events",
        "event_rate",
        "auroc",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "auprc",
        "auprc_ci_lower",
        "auprc_ci_upper",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "mean_net_benefit_0.05_0.30",
    ]
    comparison_cols = [
        "model_set",
        "algorithm",
        "development_auroc",
        "temporal_auroc",
        "delta_auroc_temporal_minus_development",
        "development_auprc",
        "temporal_auprc",
        "delta_auprc_temporal_minus_development",
        "development_brier",
        "temporal_brier",
        "development_calibration_slope",
        "temporal_calibration_slope",
    ]
    threshold_cols = [
        "model_set",
        "algorithm",
        "threshold_type",
        "development_threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
        "referral_rate",
    ]
    qc_cols = ["model_set", "algorithm", "bundle_status", "n", "events", "n_features", "n_preprocessors", "n_models", "qc_passed"]
    pred_qc_cols = [
        "model_set",
        "algorithm",
        "prediction_missing_count",
        "prediction_out_of_range_count",
        "transformed_column_match",
        "n_preprocessors",
        "n_models",
    ]

    text = [
        "# Primary Core+SDOH+HR Freeze And Temporal Validation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Primary predictor set fixed to `core_sdoh_hr`.",
        "- Primary model uses the best single algorithm by existing 2014-2020 OOF AUROC within each analysis population.",
        "- Ensemble/stacking is not used for the primary model.",
        "- No new CV, Optuna tuning, threshold re-estimation, or NHANES external validation was run.",
        "",
        "## Selected Primary Models",
        "",
        markdown_table(selected[selected_cols], floatfmt=".4f"),
        "",
        "## Final Bundle QC",
        "",
        markdown_table(bundle_qc_df[qc_cols], floatfmt=".4f"),
        "",
        "## Temporal Validation Performance",
        "",
        markdown_table(temporal_metrics[temporal_cols], floatfmt=".4f"),
        "",
        "## Development OOF vs Temporal",
        "",
        markdown_table(comparison[comparison_cols], floatfmt=".4f"),
        "",
        "## Frozen Threshold Performance In Temporal Validation",
        "",
        markdown_table(threshold_metrics[threshold_cols], floatfmt=".4f"),
        "",
        "## Prediction QC",
        "",
        markdown_table(prediction_qc[pred_qc_cols], floatfmt=".4f"),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    pipe.ensure_dirs(out)
    log_path = setup_logging(out)
    tv_dir = out / f"time_validation_{FREEZE_TAG}"
    tv_dir.mkdir(parents=True, exist_ok=True)
    (tv_dir / "plots").mkdir(parents=True, exist_ok=True)

    selected = choose_best_single_models(out)
    logging.info("Selected primary models:\n%s", selected[["model_set", "variable_set", "algorithm", "auroc_raw"]].to_string(index=False))

    dev = pipe.load_dataset(pipe.DEV_PATH)
    time_df = pipe.load_dataset(pipe.TIME_PATH)
    dev_metrics = pd.read_csv(out / "metrics" / "oof_metrics_all_models.csv")

    bundle_qc_rows = []
    temporal_metric_rows = []
    threshold_rows_all = []
    missingness_rows_all = []
    category_rows_all = []
    prediction_qc_rows = []
    comparison_rows = []
    selected_bundle_map = {}

    for idx, row in selected.iterrows():
        model_set = str(row["model_set"])
        variable_set = str(row["variable_set"])
        algorithm = str(row["algorithm"])
        bundle, qc_row = refit_or_load_bundle(
            out,
            dev,
            model_set,
            variable_set,
            algorithm,
            seed=args.seed,
            m_imputations=args.m_imputations,
            n_jobs=args.n_jobs,
            overwrite=args.overwrite_bundles,
        )
        bundle_qc_rows.append(qc_row)
        selected_bundle_map[model_set] = {
            "variable_set": variable_set,
            "algorithm": algorithm,
            "bundle_path": qc_row["bundle_path"],
            "bundle_status": qc_row["bundle_status"],
        }

        metric_row, threshold_rows, missingness_rows, category_rows, prediction_qc, comparison_row = validate_temporal(
            out,
            tv_dir,
            time_df,
            dev_metrics,
            bundle,
            model_set,
            variable_set,
            algorithm,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + idx * 1009,
        )
        temporal_metric_rows.append(metric_row)
        threshold_rows_all.extend(threshold_rows)
        missingness_rows_all.extend(missingness_rows)
        category_rows_all.extend(category_rows)
        prediction_qc_rows.append(prediction_qc)
        comparison_rows.append(comparison_row)
        logging.info("Temporal validation complete for %s__%s__%s", model_set, variable_set, algorithm)

    selected_out = out / "final_models" / f"{FREEZE_TAG}_selected_models.csv"
    bundle_qc_out = out / "final_models" / f"{FREEZE_TAG}_bundle_qc.csv"
    manifest_out = out / "final_models" / f"{FREEZE_TAG}_freeze_manifest.json"
    pipe.write_csv(selected, selected_out)
    bundle_qc_df = pd.DataFrame(bundle_qc_rows)
    pipe.write_csv(bundle_qc_df, bundle_qc_out)
    pipe.save_json(
        {
            "freeze_tag": FREEZE_TAG,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "core_sdoh_hr primary; no ensemble; best single model by existing 2014-2020 OOF AUROC",
            "development_path": str(pipe.DEV_PATH),
            "time_validation_path": str(pipe.TIME_PATH),
            "m_imputations": args.m_imputations,
            "seed": args.seed,
            "target_sensitivity": args.target_sensitivity,
            "n_bootstrap": args.n_bootstrap,
            "selected_bundles": selected_bundle_map,
            "external_validation_executed": False,
            "new_cv_or_optuna_tuning_executed": False,
            "log_path": str(log_path),
            "temporal_validation_dir": str(tv_dir),
        },
        manifest_out,
    )

    temporal_metrics = pd.DataFrame(temporal_metric_rows)
    threshold_metrics = pd.DataFrame(threshold_rows_all)
    missingness_qc = pd.DataFrame(missingness_rows_all)
    category_qc_df = pd.DataFrame(category_rows_all)
    prediction_qc = pd.DataFrame(prediction_qc_rows)
    comparison = pd.DataFrame(comparison_rows)
    pipe.write_csv(temporal_metrics, tv_dir / "time_validation_metrics_selected_models.csv")
    pipe.write_csv(threshold_metrics, tv_dir / "time_validation_threshold_metrics_selected_models.csv")
    pipe.write_csv(missingness_qc, tv_dir / "time_validation_predictor_missingness_qc.csv")
    pipe.write_csv(category_qc_df, tv_dir / "time_validation_category_level_qc.csv")
    pipe.write_csv(prediction_qc, tv_dir / "time_validation_prediction_qc.csv")
    pipe.write_csv(comparison, tv_dir / "time_validation_development_comparison.csv")
    pipe.save_json(
        {
            "freeze_tag": FREEZE_TAG,
            "script": str(Path(__file__).resolve()),
            "stage": "primary-freeze-and-temporal-validation-only",
            "selected_bundles": selected_bundle_map,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "refit_missing_or_mismatched_final_bundles_only": True,
            "new_cv_or_optuna_tuning_executed": False,
            "threshold_reestimation_executed": False,
            "external_validation_executed": False,
            "log_path": str(log_path),
        },
        tv_dir / "time_validation_run_config.json",
    )

    report_path = ROOT / "reports" / f"{FREEZE_TAG}_temporal_validation_report.md"
    write_report(report_path, selected, bundle_qc_df, temporal_metrics, comparison, threshold_metrics, prediction_qc)
    print(f"Wrote selected models: {selected_out}")
    print(f"Wrote freeze manifest: {manifest_out}")
    print(f"Wrote temporal validation dir: {tv_dir}")
    print(f"Wrote report: {report_path}")
    print(temporal_metrics[["model_set", "algorithm", "n", "events", "event_rate", "auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]].to_string(index=False))


if __name__ == "__main__":
    main()
