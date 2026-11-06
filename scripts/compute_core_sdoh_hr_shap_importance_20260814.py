#!/usr/bin/env python3
"""Compute TreeSHAP and XGBoost native variable importance for frozen primary models.

Scope:
- Uses frozen final bundles for core_sdoh_hr best single XGBoost models.
- Uses KNHANES 2014-2020 development data for interpretation.
- Does not refit, tune, change predictors, run temporal validation, or run
  external validation.

Implementation notes:
- The `shap` Python package is not required. XGBoost's native
  `pred_contribs=True` is TreeSHAP for tree boosters.
- SHAP values are computed on the raw XGBoost margin scale before OOF
  calibration.
- For MICE m=20 final bundles, SHAP values are computed for each imputation
  model and aggregated. Original-predictor SHAP is obtained by summing the
  transformed feature contributions belonging to the same predictor first, then
  taking absolute values.
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
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import knhanes_primary_ml_pipeline as pipe  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FREEZE_TAG = "primary_core_sdoh_hr_best_single_20260813"
INTERP_TAG = "core_sdoh_hr_shap_importance_20260814"
MODEL_SPECS = {
    "integrated": ("core_sdoh_hr", "xgboost"),
    "male": ("core_sdoh_hr", "xgboost"),
    "female": ("core_sdoh_hr", "xgboost"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP/importance for frozen core_sdoh_hr XGBoost models.")
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "knhanes_primary_ml_v1"))
    parser.add_argument("--dev-path", default=str(pipe.DEV_PATH))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 means full eligible development subset. Positive value uses stratified sample per model_set.",
    )
    return parser.parse_args()


def setup_logging(out: Path) -> Path:
    log_path = out / "logs" / f"{INTERP_TAG}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)
    return log_path


def predictor_for_transformed_column(col: str, features: list[str]) -> str:
    if col.startswith("num__"):
        body = col[len("num__") :]
        for feature in sorted(features, key=len, reverse=True):
            if body == feature or body.startswith(f"{feature}_rcs"):
                return feature
    if col.startswith("cat__"):
        body = col[len("cat__") :]
        for feature in sorted(features, key=len, reverse=True):
            if body.startswith(f"{feature}_"):
                return feature
    raise ValueError(f"Cannot map transformed column to predictor: {col}")


def feature_name_for_score_key(key: str, columns: list[str]) -> str:
    if key in columns:
        return key
    if key.startswith("f") and key[1:].isdigit():
        idx = int(key[1:])
        if 0 <= idx < len(columns):
            return columns[idx]
    raise KeyError(f"Unknown XGBoost feature key: {key}")


def stratified_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    frac = max_rows / len(df)
    parts = []
    for _, part in df.groupby("undiagnosed_htn", sort=False):
        n = max(1, int(round(len(part) * frac)))
        parts.append(part.sample(n=min(n, len(part)), random_state=seed))
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def empty_feature_accumulator(columns: list[str]) -> dict[str, np.ndarray]:
    n_cols = len(columns)
    return {
        "abs_sum": np.zeros(n_cols, dtype=float),
        "sum": np.zeros(n_cols, dtype=float),
        "sq_sum": np.zeros(n_cols, dtype=float),
    }


def empty_group_accumulator(features: list[str]) -> dict[str, np.ndarray]:
    n_features = len(features)
    return {
        "abs_sum": np.zeros(n_features, dtype=float),
        "sum": np.zeros(n_features, dtype=float),
        "sq_sum": np.zeros(n_features, dtype=float),
    }


def normalized_rank(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["normalized_importance"] = out.groupby(group_cols)[value_col].transform(
        lambda s: s / s.sum() if float(s.sum()) > 0 else np.nan
    )
    out["rank"] = out.groupby(group_cols)[value_col].rank(method="min", ascending=False).astype(int)
    return out.sort_values(group_cols + ["rank"])


def save_bar_plot(df: pd.DataFrame, title: str, value_col: str, label_col: str, path: Path, top_n: int = 15) -> None:
    part = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, max(4, 0.34 * len(part) + 1.4)))
    plt.barh(part[label_col].astype(str), part[value_col].astype(float), color="#356f9f")
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def compute_for_model_set(
    out: Path,
    interp_dir: Path,
    dev: pd.DataFrame,
    model_set: str,
    variable_set: str,
    algorithm: str,
    seed: int,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    combo = pipe.model_combo_id(model_set, variable_set, algorithm)
    bundle_path = out / "final_models" / f"{combo}__final_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    bundle = joblib.load(bundle_path)
    if bundle.get("model_set") != model_set or bundle.get("variable_set") != variable_set or bundle.get("algorithm") != algorithm:
        raise RuntimeError(f"Bundle identity mismatch: {bundle_path}")
    if bundle.get("primary_freeze_tag") != FREEZE_TAG:
        logging.warning("Bundle %s has freeze tag %s, expected %s", combo, bundle.get("primary_freeze_tag"), FREEZE_TAG)

    features = list(bundle["features"])
    df_all = pipe.subset_for_model_set(dev, model_set).reset_index(drop=True)
    df = stratified_sample(df_all, max_rows=max_rows, seed=seed).reset_index(drop=True)
    X = df[features]
    y = df["undiagnosed_htn"].astype(int).to_numpy()

    preprocessors = bundle["preprocessors"]
    models = bundle["models"]
    if len(preprocessors) != len(models):
        raise RuntimeError(f"Preprocessor/model count mismatch for {combo}")
    if len(preprocessors) != 20:
        logging.warning("%s has %s preprocessors/models, expected MICE m=20", combo, len(preprocessors))

    first_matrix = preprocessors[0].transform(X)
    columns = list(first_matrix.columns)
    transformed_to_predictor = [predictor_for_transformed_column(c, features) for c in columns]
    feature_to_idx = {feature: idx for idx, feature in enumerate(features)}
    group_indices = [[i for i, pred in enumerate(transformed_to_predictor) if pred == feature] for feature in features]
    if any(not idxs for idxs in group_indices):
        missing_groups = [feature for feature, idxs in zip(features, group_indices) if not idxs]
        raise RuntimeError(f"No transformed columns for predictors: {missing_groups}")

    feat_acc = empty_feature_accumulator(columns)
    group_acc = empty_group_accumulator(features)
    group_contrib_sum_for_rows = np.zeros((len(df), len(features)), dtype=float)
    native_rows: list[dict[str, Any]] = []
    bias_values = []
    margin_check_max_abs = []
    n_total = 0

    for imp_idx, (preprocessor, model) in enumerate(zip(preprocessors, models), start=1):
        matrix = preprocessor.transform(X)
        if list(matrix.columns) != columns:
            raise RuntimeError(f"Transformed columns differ across MICE preprocessors for {combo} imputation {imp_idx}")
        booster = model.get_booster()
        dmat = xgb.DMatrix(matrix, feature_names=columns)
        contrib = booster.predict(dmat, pred_contribs=True)
        if contrib.shape[1] != len(columns) + 1:
            raise RuntimeError(f"Unexpected SHAP contribution shape for {combo}: {contrib.shape}")
        phi = np.asarray(contrib[:, :-1], dtype=float)
        bias = np.asarray(contrib[:, -1], dtype=float)
        margin = np.asarray(booster.predict(dmat, output_margin=True), dtype=float)
        margin_check_max_abs.append(float(np.max(np.abs(phi.sum(axis=1) + bias - margin))))
        bias_values.append(float(np.mean(bias)))

        feat_acc["abs_sum"] += np.abs(phi).sum(axis=0)
        feat_acc["sum"] += phi.sum(axis=0)
        feat_acc["sq_sum"] += np.square(phi).sum(axis=0)
        group_phi = np.zeros((len(df), len(features)), dtype=float)
        for group_idx, col_indices in enumerate(group_indices):
            group_phi[:, group_idx] = phi[:, col_indices].sum(axis=1)
        group_acc["abs_sum"] += np.abs(group_phi).sum(axis=0)
        group_acc["sum"] += group_phi.sum(axis=0)
        group_acc["sq_sum"] += np.square(group_phi).sum(axis=0)
        group_contrib_sum_for_rows += group_phi
        n_total += len(df)

        for importance_type in ["gain", "weight", "cover", "total_gain", "total_cover"]:
            scores = booster.get_score(importance_type=importance_type)
            transformed_values = {c: 0.0 for c in columns}
            for key, value in scores.items():
                transformed_values[feature_name_for_score_key(key, columns)] = float(value)
            predictor_values = {feature: 0.0 for feature in features}
            for col, value in transformed_values.items():
                predictor_values[predictor_for_transformed_column(col, features)] += float(value)
            for feature, value in predictor_values.items():
                native_rows.append(
                    {
                        "model_set": model_set,
                        "variable_set": variable_set,
                        "algorithm": algorithm,
                        "importance_type": importance_type,
                        "mice_imputation": imp_idx,
                        "predictor": feature,
                        "importance": value,
                    }
                )

        logging.info("%s SHAP complete for MICE imputation %s/%s", combo, imp_idx, len(preprocessors))

    feature_rows = []
    for col_idx, col in enumerate(columns):
        feature_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "transformed_feature": col,
                "predictor": transformed_to_predictor[col_idx],
                "n_rows": int(len(df)),
                "m_imputations": int(len(preprocessors)),
                "mean_abs_shap": float(feat_acc["abs_sum"][col_idx] / n_total),
                "mean_shap": float(feat_acc["sum"][col_idx] / n_total),
                "rms_shap": float(math.sqrt(feat_acc["sq_sum"][col_idx] / n_total)),
            }
        )
    transformed_df = normalized_rank(pd.DataFrame(feature_rows), "mean_abs_shap", ["model_set"])

    group_rows = []
    for group_idx, feature in enumerate(features):
        group_rows.append(
            {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "predictor": feature,
                "n_rows": int(len(df)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "m_imputations": int(len(preprocessors)),
                "mean_abs_group_shap": float(group_acc["abs_sum"][group_idx] / n_total),
                "mean_group_shap": float(group_acc["sum"][group_idx] / n_total),
                "rms_group_shap": float(math.sqrt(group_acc["sq_sum"][group_idx] / n_total)),
                "n_transformed_features": int(len(group_indices[group_idx])),
            }
        )
    grouped_df = normalized_rank(pd.DataFrame(group_rows), "mean_abs_group_shap", ["model_set"])

    native_raw = pd.DataFrame(native_rows)
    native_df = (
        native_raw.groupby(["model_set", "variable_set", "algorithm", "importance_type", "predictor"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
            nonzero_mice_models=("importance", lambda s: int((s > 0).sum())),
            m_imputations=("importance", "size"),
        )
    )
    native_df = normalized_rank(native_df, "mean_importance", ["model_set", "importance_type"])

    row_group_contrib = group_contrib_sum_for_rows / len(preprocessors)
    wide = df[["participant_id", "source_year", "sex", "undiagnosed_htn"]].copy()
    wide.insert(0, "model_set", model_set)
    wide.insert(1, "variable_set", variable_set)
    wide.insert(2, "algorithm", algorithm)
    for feature_idx, feature in enumerate(features):
        wide[f"shap__{feature}"] = row_group_contrib[:, feature_idx]
    wide["shap_sum_predictors"] = row_group_contrib.sum(axis=1)
    pipe.write_csv(wide, interp_dir / f"{combo}__participant_grouped_shap_values.csv")

    qc = {
        "model_set": model_set,
        "variable_set": variable_set,
        "algorithm": algorithm,
        "bundle_path": str(bundle_path),
        "n_development_subset": int(len(df_all)),
        "n_rows_used": int(len(df)),
        "events_used": int(y.sum()),
        "event_rate_used": float(y.mean()),
        "sampled": bool(len(df) < len(df_all)),
        "n_predictors": int(len(features)),
        "n_transformed_features": int(len(columns)),
        "m_imputations": int(len(preprocessors)),
        "features": features,
        "transformed_columns": columns,
        "mean_bias_margin": float(np.mean(bias_values)),
        "max_margin_reconstruction_abs_error": float(np.max(margin_check_max_abs)),
    }
    return grouped_df, transformed_df, native_df, wide, qc


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


def write_report(report_path: Path, grouped: pd.DataFrame, native: pd.DataFrame, qc_rows: list[dict[str, Any]], interp_dir: Path) -> None:
    lines = [
        "# Core+SDOH+HR SHAP And Variable Importance",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Frozen primary `core_sdoh_hr` final bundles were used.",
        "- Integrated, male, and female primary models are XGBoost.",
        "- TreeSHAP values were computed with XGBoost native `pred_contribs=True`.",
        "- SHAP values are on the raw XGBoost margin scale before OOF calibration.",
        "- No model refit, hyperparameter tuning, temporal validation, or external validation was run.",
        "",
        "## Top Original Predictors By Mean Absolute Grouped SHAP",
        "",
    ]
    show_cols = ["model_set", "rank", "predictor", "mean_abs_group_shap", "normalized_importance", "mean_group_shap", "n_transformed_features"]
    for model_set in ["integrated", "male", "female"]:
        lines.append(f"### {model_set}")
        lines.append("")
        lines.append(markdown_table(grouped[grouped["model_set"].eq(model_set)][show_cols].head(15), floatfmt=".4f"))
        lines.append("")
    lines += [
        "## Top Original Predictors By XGBoost Native Gain",
        "",
    ]
    native_gain = native[native["importance_type"].eq("gain")].copy()
    gain_cols = ["model_set", "rank", "predictor", "mean_importance", "normalized_importance", "nonzero_mice_models"]
    for model_set in ["integrated", "male", "female"]:
        lines.append(f"### {model_set}")
        lines.append("")
        lines.append(markdown_table(native_gain[native_gain["model_set"].eq(model_set)][gain_cols].head(15), floatfmt=".4f"))
        lines.append("")
    lines += [
        "## QC",
        "",
    ]
    qc_df = pd.DataFrame(qc_rows)
    qc_cols = ["model_set", "n_development_subset", "n_rows_used", "events_used", "n_predictors", "n_transformed_features", "m_imputations", "max_margin_reconstruction_abs_error"]
    lines.append(markdown_table(qc_df[qc_cols], floatfmt=".6f"))
    lines += [
        "",
        "## Output Directory",
        "",
        f"`{interp_dir}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    interp_dir = out / "interpretability_core_sdoh_hr_20260814"
    interp_dir.mkdir(parents=True, exist_ok=True)
    (interp_dir / "plots").mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out)

    dev = pipe.load_dataset(Path(args.dev_path))
    all_grouped = []
    all_transformed = []
    all_native = []
    qc_rows = []

    for idx, (model_set, (variable_set, algorithm)) in enumerate(MODEL_SPECS.items()):
        grouped, transformed, native, _wide, qc = compute_for_model_set(
            out=out,
            interp_dir=interp_dir,
            dev=dev,
            model_set=model_set,
            variable_set=variable_set,
            algorithm=algorithm,
            seed=args.seed + idx * 101,
            max_rows=args.max_rows,
        )
        all_grouped.append(grouped)
        all_transformed.append(transformed)
        all_native.append(native)
        qc_rows.append(qc)
        save_bar_plot(
            grouped,
            f"{model_set} core_sdoh_hr XGBoost grouped SHAP",
            "mean_abs_group_shap",
            "predictor",
            interp_dir / "plots" / f"{model_set}__grouped_shap_top_predictors.png",
        )
        gain = native[native["importance_type"].eq("gain")].copy()
        save_bar_plot(
            gain,
            f"{model_set} core_sdoh_hr XGBoost native gain",
            "mean_importance",
            "predictor",
            interp_dir / "plots" / f"{model_set}__native_gain_top_predictors.png",
        )

    grouped_df = pd.concat(all_grouped, ignore_index=True)
    transformed_df = pd.concat(all_transformed, ignore_index=True)
    native_df = pd.concat(all_native, ignore_index=True)
    pipe.write_csv(grouped_df, interp_dir / "shap_original_predictor_importance.csv")
    pipe.write_csv(transformed_df, interp_dir / "shap_transformed_feature_importance.csv")
    pipe.write_csv(native_df, interp_dir / "xgboost_native_importance_by_predictor.csv")
    pipe.save_json(
        {
            "interpretability_tag": INTERP_TAG,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(out),
            "interpretability_dir": str(interp_dir),
            "dev_path": str(args.dev_path),
            "model_specs": MODEL_SPECS,
            "shap_method": "xgboost native pred_contribs TreeSHAP",
            "shap_scale": "raw XGBoost margin before OOF calibration",
            "max_rows": args.max_rows,
            "new_model_training_executed": False,
            "temporal_validation_executed": False,
            "external_validation_executed": False,
            "log_path": str(log_path),
            "qc": qc_rows,
        },
        interp_dir / "interpretability_run_config.json",
    )
    report_path = ROOT / "reports" / "core_sdoh_hr_shap_variable_importance_20260814.md"
    write_report(report_path, grouped_df, native_df, qc_rows, interp_dir)
    print(f"Wrote {report_path}")
    print(grouped_df.groupby("model_set").head(10)[["model_set", "rank", "predictor", "mean_abs_group_shap", "normalized_importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
