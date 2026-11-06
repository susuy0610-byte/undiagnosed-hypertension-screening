#!/usr/bin/env python3
"""
Final-refit-only runner for the locked KNHANES primary ML pipeline.

This script intentionally does not load or evaluate the 2021-2022 temporal
validation data. It reads completed 2014-2020 OOF predictions, applies the
locked model-selection rule, freezes OOF calibrators/thresholds, and refits
the selected models on the full 2014-2020 development data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402


def fast_threshold_rows(y: np.ndarray, p: np.ndarray, target_sensitivity: float) -> list[dict[str, Any]]:
    """Vectorized equivalent of pipe.choose_thresholds."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    thresholds = np.unique(np.r_[np.linspace(0.001, 0.999, 999), p])

    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(y_sorted)
    total_pos = float(y_sorted.sum())
    total_neg = float(n - total_pos)

    idx = np.searchsorted(p_sorted, thresholds, side="left")
    pred_pos = (n - idx).astype(float)
    cpos_rev = np.r_[np.cumsum(y_sorted[::-1])[::-1], 0]
    tp = cpos_rev[idx].astype(float)
    fp = pred_pos - tp
    fn = total_pos - tp
    tn = total_neg - fp

    sensitivity = np.divide(tp, total_pos, out=np.zeros_like(tp), where=total_pos != 0)
    specificity = np.divide(tn, total_neg, out=np.zeros_like(tn), where=total_neg != 0)
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    npv = np.divide(tn, tn + fn, out=np.zeros_like(tn), where=(tn + fn) != 0)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    referral_rate = pred_pos / float(n)

    def row_at(i: int, threshold_type: str) -> dict[str, Any]:
        return {
            "threshold": float(thresholds[i]),
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]),
            "ppv": float(ppv[i]),
            "npv": float(npv[i]),
            "balanced_accuracy": float(balanced_accuracy[i]),
            "referral_rate": float(referral_rate[i]),
            "tp": float(tp[i]),
            "fp": float(fp[i]),
            "tn": float(tn[i]),
            "fn": float(fn[i]),
            "threshold_type": threshold_type,
        }

    sens_ok = np.where(sensitivity >= target_sensitivity)[0]
    if len(sens_ok):
        sensitivity_idx = int(sens_ok[-1])
    else:
        sensitivity_idx = int(np.argmin(np.abs(sensitivity - target_sensitivity)))

    youden = sensitivity + specificity - 1.0
    best_youden = float(youden.max())
    youden_idx = int(np.where(np.isclose(youden, best_youden, rtol=1e-12, atol=1e-12))[0][-1])

    return [
        row_at(sensitivity_idx, f"sensitivity_{target_sensitivity:.2f}"),
        row_at(youden_idx, "youden"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected KNHANES final refits without temporal validation.")
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "knhanes_primary_ml_v1"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--m-imputations", type=int, default=20)
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> pipe.Config:
    return pipe.Config(
        output_dir=args.output_dir,
        seed=args.seed,
        n_folds=args.n_folds,
        inner_folds=3,
        m_imputations=args.m_imputations,
        n_trials=40,
        target_sensitivity=args.target_sensitivity,
        n_jobs=args.n_jobs,
        stage="final-refit-only",
        model_sets=pipe.MODEL_SETS,
        variable_sets=pipe.VARIABLE_SETS,
        algorithms=pipe.ALGORITHMS,
        outer_folds=None,
        overwrite=args.overwrite,
    )


def setup_logging(out: Path) -> Path:
    log_path = out / "logs" / f"final_refit_only_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)
    return log_path


def summarize_oof_fast(config: pipe.Config, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for model_set in config.model_sets:
        for variable_set in config.variable_sets:
            for algorithm in config.algorithms:
                if not pipe.valid_model_combo(variable_set, algorithm):
                    continue
                oof = pipe.collect_oof(out, model_set, variable_set, algorithm, config.n_folds)
                if oof is None:
                    logging.info("OOF incomplete or absent for %s %s %s; skipping", model_set, variable_set, algorithm)
                    continue

                y = oof["undiagnosed_htn"].astype(int).to_numpy()
                p_raw = oof["pred_raw"].astype(float).to_numpy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    summary, p_cal, calibrator = pipe.summarize_candidate(y, p_raw, config.target_sensitivity)

                combo = pipe.model_combo_id(model_set, variable_set, algorithm)
                pipe.save_json(
                    {"intercept": calibrator.intercept_, "slope": calibrator.slope_},
                    out / "metrics" / f"{combo}__calibrator.json",
                )

                metric_rows.append(
                    {
                        "model_set": model_set,
                        "variable_set": variable_set,
                        "algorithm": algorithm,
                        "n": len(oof),
                        "events": int(y.sum()),
                        **summary,
                    }
                )
                for row in fast_threshold_rows(y, p_cal, config.target_sensitivity):
                    threshold_rows.append({"model_set": model_set, "variable_set": variable_set, "algorithm": algorithm, **row})

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No complete OOF predictions found.")
    thresholds = pd.DataFrame(threshold_rows)
    ranked = pipe.rank_candidates(metrics)

    pipe.write_csv(metrics, out / "metrics" / "oof_metrics_all_models.csv")
    pipe.write_csv(thresholds, out / "metrics" / "oof_thresholds_all_models.csv")
    pipe.write_csv(ranked, out / "metrics" / "oof_model_selection_ranked.csv")
    pipe.write_csv(
        ranked[ranked["selected_within_model_set"]].copy(),
        out / "final_models" / "final_refit_selected_models.csv",
    )
    logging.info("Summarized %d complete OOF candidates", len(metrics))
    return ranked, thresholds


def aggregate_selected_params(out: Path, model_set: str, variable_set: str, algorithm: str) -> dict[str, Any]:
    best_paths = sorted((out / "tuning").glob(f"{model_set}__{variable_set}__{algorithm}__fold*__best.json"))
    if not best_paths:
        return {}
    params_by_fold = [pipe.read_json(path)["params"] for path in best_paths]
    return pipe.aggregate_params(params_by_fold)


def refit_selected(config: pipe.Config, out: Path, ranked: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    dev = pipe.load_dataset(pipe.DEV_PATH)
    selected = ranked[ranked["selected_within_model_set"]].copy()
    if len(selected) != 3:
        raise RuntimeError(f"Expected exactly 3 selected model sets; found {len(selected)}.")

    summary_rows: list[dict[str, Any]] = []
    for _, row in selected.sort_values("model_set").iterrows():
        model_set = str(row["model_set"])
        variable_set = str(row["variable_set"])
        algorithm = str(row["algorithm"])
        combo = pipe.model_combo_id(model_set, variable_set, algorithm)
        final_path = out / "final_models" / f"{combo}__final_bundle.joblib"
        config_path = out / "final_models" / f"{combo}__final_config.json"
        features = pipe.predictors_for(model_set, variable_set)

        train_df = pipe.subset_for_model_set(dev, model_set)
        X_train = train_df[features]
        y_train = train_df["undiagnosed_htn"].astype(int).to_numpy()
        n = int(len(train_df))
        events = int(y_train.sum())
        params = aggregate_selected_params(out, model_set, variable_set, algorithm)

        if final_path.exists() and not config.overwrite:
            bundle = joblib.load(final_path)
            logging.info("Loaded existing final bundle: %s", final_path)
        else:
            logging.info(
                "Final refit start: %s features=%d n=%d events=%d m=%d params=%s",
                combo,
                len(features),
                n,
                events,
                config.m_imputations,
                params,
            )
            preprocessors = pipe.fit_preprocessors(X_train, features, config.m_imputations, config.seed + 9000)
            matrices_train = pipe.transformed_list(preprocessors, X_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models = pipe.fit_model_ensemble(algorithm, params, matrices_train, y_train, config.seed + 9999, config.n_jobs)

            calibrator = pipe.calibrator_lookup(out, model_set, variable_set, algorithm)
            frozen_thresholds = thresholds[
                (thresholds["model_set"] == model_set)
                & (thresholds["variable_set"] == variable_set)
                & (thresholds["algorithm"] == algorithm)
            ].copy()
            if frozen_thresholds.empty:
                raise RuntimeError(f"No frozen thresholds found for {combo}.")

            bundle = {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "features": features,
                "params": params,
                "preprocessors": preprocessors,
                "models": models,
                "calibrator": calibrator,
                "thresholds": frozen_thresholds,
                "locked_variables_path": str(pipe.LOCKED_VARIABLES_PATH),
                "final_refit": {
                    "development_path": str(pipe.DEV_PATH),
                    "n": n,
                    "events": events,
                    "seed": config.seed,
                    "m_imputations": config.m_imputations,
                    "target_sensitivity": config.target_sensitivity,
                    "n_folds_from_oof": config.n_folds,
                },
            }
            joblib.dump(bundle, final_path, compress=3)
            pipe.save_json(
                {
                    "model_set": model_set,
                    "variable_set": variable_set,
                    "algorithm": algorithm,
                    "features": features,
                    "params": params,
                    "thresholds": frozen_thresholds.to_dict(orient="records"),
                    "calibrator": {"intercept": calibrator.intercept_, "slope": calibrator.slope_},
                    "final_refit": bundle["final_refit"],
                },
                config_path,
            )
            logging.info("Saved final bundle: %s", final_path)

        summary_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "n": n,
                "events": events,
                "event_rate": events / n,
                "n_features": len(features),
                "params": params,
                "final_bundle": str(final_path),
                "final_config": str(config_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    pipe.write_csv(summary, out / "final_models" / "final_refit_summary.csv")
    return summary


def main() -> None:
    args = parse_args()
    config = make_config(args)
    out = Path(config.output_dir)
    pipe.ensure_dirs(out)
    log_path = setup_logging(out)
    pipe.save_json(
        {
            "script": str(Path(__file__).resolve()),
            "stage": "final-refit-only",
            "output_dir": str(out),
            "seed": config.seed,
            "n_folds": config.n_folds,
            "m_imputations": config.m_imputations,
            "target_sensitivity": config.target_sensitivity,
            "n_jobs": config.n_jobs,
            "log_path": str(log_path),
            "temporal_validation_executed": False,
        },
        out / "config" / "final_refit_only_config.json",
    )
    ranked, thresholds = summarize_oof_fast(config, out)
    summary = refit_selected(config, out, ranked, thresholds)
    logging.info("Final refit-only complete:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
