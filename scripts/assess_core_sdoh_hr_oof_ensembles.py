#!/usr/bin/env python3
"""Assess OOF-only ensemble/stacking potential for the locked core_sdoh_hr set.

This script reads existing development OOF predictions only. It does not refit
any base model, rerun CV, tune hyperparameters, or touch temporal/external data.
Fold-stacked meta-models are fit on 4 OOF folds and evaluated on the held-out OOF
fold to avoid evaluating a stacker on the same rows used to fit it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from knhanes_primary_ml_pipeline import (
    calibration_intercept_slope,
    choose_thresholds,
    dca_table,
    summarize_candidate,
)


MODEL_SETS = ["integrated", "male", "female"]
VARIABLE_SET = "core_sdoh_hr"
KEY_COLUMNS = ["participant_id", "source_year", "sex", "fold", "undiagnosed_htn"]


@dataclass(frozen=True)
class Candidate:
    name: str
    pred: np.ndarray
    algorithms: list[str]
    kind: str


def read_oof(out_dir: Path, model_set: str, variable_set: str, algorithm: str) -> pd.DataFrame:
    parts = []
    for fold in range(1, 6):
        path = out_dir / "oof_predictions" / f"{model_set}__{variable_set}__{algorithm}__fold{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        keep = KEY_COLUMNS + ["pred_raw"]
        parts.append(df[keep].copy())
    df_all = pd.concat(parts, ignore_index=True)
    duplicated = df_all.duplicated(KEY_COLUMNS).sum()
    if duplicated:
        raise ValueError(f"{model_set}/{algorithm} has duplicated OOF rows: {duplicated}")
    return df_all.rename(columns={"pred_raw": f"pred__{algorithm}"})


def build_matrix(out_dir: Path, metrics: pd.DataFrame, model_set: str) -> tuple[pd.DataFrame, list[str]]:
    rows = metrics[(metrics["model_set"] == model_set) & (metrics["variable_set"] == VARIABLE_SET)].copy()
    rows = rows.sort_values("auroc_raw", ascending=False)
    algorithms = rows["algorithm"].tolist()
    merged: pd.DataFrame | None = None
    for algorithm in algorithms:
        df_alg = read_oof(out_dir, model_set, VARIABLE_SET, algorithm)
        if merged is None:
            merged = df_alg
        else:
            merged = merged.merge(df_alg, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged is None:
        raise ValueError(f"No OOF data for {model_set}")
    pred_cols = [f"pred__{a}" for a in algorithms]
    if merged[pred_cols].isna().any().any():
        missing = merged[pred_cols].isna().sum()
        raise ValueError(f"Missing OOF predictions for {model_set}: {missing[missing > 0].to_dict()}")
    return merged, algorithms


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fold_stack_predictions(
    df: pd.DataFrame,
    algorithms: list[str],
    use_logit_features: bool,
    seed: int,
) -> np.ndarray:
    pred_cols = [f"pred__{a}" for a in algorithms]
    x_all = df[pred_cols].to_numpy(dtype=float)
    if use_logit_features:
        x_all = logit(x_all)
    y = df["undiagnosed_htn"].astype(int).to_numpy()
    folds = df["fold"].astype(int).to_numpy()
    out = np.full(len(df), np.nan, dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        meta = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed + int(fold),
        )
        meta.fit(x_all[train], y[train])
        out[valid] = meta.predict_proba(x_all[valid])[:, 1]
    if np.isnan(out).any():
        raise ValueError("Fold-stacked predictions contain missing values")
    return out


def metric_row(model_set: str, candidate: Candidate, y: np.ndarray, target_sensitivity: float) -> dict[str, object]:
    summary, p_cal, _ = summarize_candidate(y, candidate.pred, target_sensitivity)
    row: dict[str, object] = {
        "model_set": model_set,
        "variable_set": VARIABLE_SET,
        "candidate": candidate.name,
        "kind": candidate.kind,
        "algorithms": "+".join(candidate.algorithms),
        "n": int(len(y)),
        "events": int(y.sum()),
    }
    row.update(summary)
    for thr in choose_thresholds(y, p_cal, target_sensitivity):
        prefix = str(thr["threshold_type"])
        for key in ["threshold", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "referral_rate"]:
            row[f"{prefix}__{key}"] = float(thr[key])
    return row


def bootstrap_auc_delta(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_new: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    if not deltas:
        return np.nan, np.nan, np.nan
    arr = np.asarray(deltas)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def candidate_predictions(df: pd.DataFrame, algorithms_by_rank: list[str], seed: int) -> list[Candidate]:
    pred = {alg: df[f"pred__{alg}"].to_numpy(dtype=float) for alg in algorithms_by_rank}
    top2 = algorithms_by_rank[:2]
    top3 = algorithms_by_rank[:3]
    all_algs = algorithms_by_rank

    candidates: list[Candidate] = [
        Candidate("avg_top2", np.mean([pred[a] for a in top2], axis=0), top2, "simple_average"),
        Candidate("avg_top3", np.mean([pred[a] for a in top3], axis=0), top3, "simple_average"),
        Candidate("avg_all7", np.mean([pred[a] for a in all_algs], axis=0), all_algs, "simple_average"),
        Candidate(
            "fold_stack_logistic_top3",
            fold_stack_predictions(df, top3, use_logit_features=False, seed=seed),
            top3,
            "fold_stacking",
        ),
        Candidate(
            "fold_stack_logit_top3",
            fold_stack_predictions(df, top3, use_logit_features=True, seed=seed),
            top3,
            "fold_stacking",
        ),
        Candidate(
            "fold_stack_logistic_all7",
            fold_stack_predictions(df, all_algs, use_logit_features=False, seed=seed),
            all_algs,
            "fold_stacking",
        ),
        Candidate(
            "fold_stack_logit_all7",
            fold_stack_predictions(df, all_algs, use_logit_features=True, seed=seed),
            all_algs,
            "fold_stacking",
        ),
    ]
    return candidates


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small no-dependency Markdown table writer."""
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
    path: Path,
    base_best: pd.DataFrame,
    ens: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# Core+SDOH+HR OOF Ensemble/Stacking Diagnostic")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Read existing KNHANES 2014-2020 development OOF predictions only.")
    lines.append("- Predictor set fixed to `core_sdoh_hr`.")
    lines.append("- No base-model refit, Optuna tuning, temporal validation, or external validation was run.")
    lines.append("- Stacking diagnostics used 4 OOF folds to fit the meta-model and the held-out OOF fold for evaluation.")
    lines.append("")
    lines.append("## Current Best Base Models")
    lines.append("")
    lines.append(markdown_table(base_best, floatfmt=".4f"))
    lines.append("")
    lines.append("## Best OOF Ensemble/Stacking Candidate By Group")
    lines.append("")
    lines.append(markdown_table(comparison, floatfmt=".4f"))
    lines.append("")
    lines.append("## All Ensemble/Stacking Candidates")
    lines.append("")
    show_cols = [
        "model_set",
        "candidate",
        "kind",
        "algorithms",
        "auroc_raw",
        "auprc_raw",
        "brier_raw",
        "calibration_intercept_raw",
        "calibration_slope_raw",
        "mean_net_benefit_0.05_0.30",
    ]
    lines.append(markdown_table(ens[show_cols].sort_values(["model_set", "auroc_raw"], ascending=[True, False]), floatfmt=".4f"))
    lines.append("")
    lines.append("## QC")
    lines.append("")
    for key, value in diagnostics.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("OOF-only ensemble/stacking is useful only if the held-out-fold stacking or simple averaging improves AUROC meaningfully over the best single base model without worsening Brier/calibration in a way that complicates validation. Small gains within the bootstrap CI around zero should be treated as insufficient evidence for switching to an ensemble.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analysis/knhanes_primary_ml_v1")
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    metrics_path = out_dir / "metrics" / "oof_metrics_all_models.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["variable_set"] == VARIABLE_SET].copy()

    base_best_rows = []
    ensemble_rows = []
    comparison_rows = []
    diagnostics: dict[str, object] = {}

    for model_set in MODEL_SETS:
        df, algorithms = build_matrix(out_dir, metrics, model_set)
        y = df["undiagnosed_htn"].astype(int).to_numpy()
        pred_cols = [f"pred__{a}" for a in algorithms]
        diagnostics[f"{model_set}_n"] = int(len(df))
        diagnostics[f"{model_set}_events"] = int(y.sum())
        diagnostics[f"{model_set}_folds"] = ",".join(str(x) for x in sorted(df["fold"].unique()))
        diagnostics[f"{model_set}_algorithms_by_auroc"] = ",".join(algorithms)
        diagnostics[f"{model_set}_prediction_missing"] = int(df[pred_cols].isna().sum().sum())
        diagnostics[f"{model_set}_duplicate_keys"] = int(df.duplicated(KEY_COLUMNS).sum())

        base = metrics[metrics["model_set"] == model_set].sort_values("auroc_raw", ascending=False).iloc[0]
        base_alg = str(base["algorithm"])
        base_pred = df[f"pred__{base_alg}"].to_numpy(dtype=float)
        base_best_rows.append(
            {
                "model_set": model_set,
                "best_base_algorithm": base_alg,
                "n": int(base["n"]),
                "events": int(base["events"]),
                "auroc_raw": safe_float(base["auroc_raw"]),
                "auprc_raw": safe_float(base["auprc_raw"]),
                "brier_raw": safe_float(base["brier_raw"]),
                "calibration_intercept_raw": safe_float(base["calibration_intercept_raw"]),
                "calibration_slope_raw": safe_float(base["calibration_slope_raw"]),
                "mean_net_benefit_0.05_0.30": safe_float(base["mean_net_benefit_0.05_0.30"]),
            }
        )

        candidates = candidate_predictions(df, algorithms, seed=args.seed)
        for cand in candidates:
            ensemble_rows.append(metric_row(model_set, cand, y, args.target_sensitivity))

        ens_model = pd.DataFrame([r for r in ensemble_rows if r["model_set"] == model_set])
        best_ens = ens_model.sort_values("auroc_raw", ascending=False).iloc[0]
        best_ens_pred = next(c.pred for c in candidates if c.name == best_ens["candidate"])
        boot_mean, boot_low, boot_high = bootstrap_auc_delta(
            y,
            base_pred,
            best_ens_pred,
            n_boot=args.n_bootstrap,
            seed=args.seed + 100 * (MODEL_SETS.index(model_set) + 1),
        )
        comparison_rows.append(
            {
                "model_set": model_set,
                "best_base_algorithm": base_alg,
                "base_auroc": safe_float(base["auroc_raw"]),
                "best_ensemble_candidate": best_ens["candidate"],
                "ensemble_kind": best_ens["kind"],
                "ensemble_algorithms": best_ens["algorithms"],
                "ensemble_auroc": safe_float(best_ens["auroc_raw"]),
                "delta_auroc": safe_float(best_ens["auroc_raw"]) - safe_float(base["auroc_raw"]),
                "bootstrap_delta_mean": boot_mean,
                "bootstrap_delta_95ci_low": boot_low,
                "bootstrap_delta_95ci_high": boot_high,
                "base_auprc": safe_float(base["auprc_raw"]),
                "ensemble_auprc": safe_float(best_ens["auprc_raw"]),
                "base_brier": safe_float(base["brier_raw"]),
                "ensemble_brier": safe_float(best_ens["brier_raw"]),
                "base_cal_slope": safe_float(base["calibration_slope_raw"]),
                "ensemble_cal_slope": safe_float(best_ens["calibration_slope_raw"]),
                "base_mean_nb_0.05_0.30": safe_float(base["mean_net_benefit_0.05_0.30"]),
                "ensemble_mean_nb_0.05_0.30": safe_float(best_ens["mean_net_benefit_0.05_0.30"]),
            }
        )

    base_best_df = pd.DataFrame(base_best_rows)
    ens_df = pd.DataFrame(ensemble_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    metrics_out = out_dir / "metrics"
    base_best_df.to_csv(metrics_out / "core_sdoh_hr_oof_base_best_for_ensemble.csv", index=False)
    ens_df.to_csv(metrics_out / "core_sdoh_hr_oof_ensemble_stacking_metrics.csv", index=False)
    comparison_df.to_csv(metrics_out / "core_sdoh_hr_oof_ensemble_stacking_vs_base.csv", index=False)
    (metrics_out / "core_sdoh_hr_oof_ensemble_stacking_qc.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path = Path("reports") / "core_sdoh_hr_oof_ensemble_stacking_diagnostic_20260813.md"
    write_report(report_path, base_best_df, ens_df, comparison_df, diagnostics)
    print(f"Wrote {report_path}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
