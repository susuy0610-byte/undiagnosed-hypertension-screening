#!/usr/bin/env python3
"""Decompose the external AUROC drop: family_history_htn blinding vs residual transportability gap.

Method (prediction-only, no retraining): take the KNHANES 2021-2022 temporal
validation cohort (same population, same frozen model, already-reported
temporal AUROC with the real family_history_htn value observed). Create a
counterfactual copy where family_history_htn is set to fully missing (NaN)
for every participant, exactly matching how it enters the frozen model when
applied to NHANES/HSE (where it is a 100%-missing external predictor). Run
both versions through the SAME frozen preprocessors/models/calibrator and
compare AUROC.

This isolates, in a fully matched Korean population, how much AUROC is lost
purely from losing the family_history_htn signal (same imputation mechanism
as used externally), independent of any genuine cross-country transportability
gap. The residual (real external drop - blinding-only drop) is attributable to
factors other than this one missing predictor.
"""

from __future__ import annotations

import json
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
DECOMP_DIR = OUT / f"family_history_blinding_decomposition_{STAMP}"
REPORTS = ROOT / "reports"
MANIFEST_PATH = FINAL_MODELS / f"{FREEZE_TAG}_freeze_manifest.json"
TEMPORAL_PATH = ROOT / "data/processed/knhanes_time_validation_2021_2022_check_only.csv"

# Real, already-reported external AUROC (frozen model, as-is calibrator) for context.
EXTERNAL_AUROC = {
    "nhanes_2021_2023": {"integrated": 0.6910, "male": 0.6592, "female": 0.7204},
    "hse_pooled_2018_2022": {"integrated": 0.7204, "male": 0.6714, "female": 0.7460},
}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_one(model_set: str, selected: dict[str, Any], df_full: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = Path(selected["bundle_path"])
    bundle = joblib.load(bundle_path)
    features = list(bundle["features"])
    assert bundle["variable_set"] == "core_sdoh_hr"
    assert bundle["model_set"] == model_set

    if model_set == "integrated":
        df = df_full.copy()
    elif model_set == "male":
        df = df_full[df_full["sex"] == 1].copy()
    elif model_set == "female":
        df = df_full[df_full["sex"] == 2].copy()
    else:
        raise ValueError(model_set)

    y = df["undiagnosed_htn"].astype(int).to_numpy()

    def predict(input_df: pd.DataFrame) -> np.ndarray:
        X = input_df[features]
        matrices = pipe.transformed_list(bundle["preprocessors"], X)
        p_raw = pipe.mean_predict_from_models(bundle["models"], matrices)
        return bundle["calibrator"].transform(p_raw)

    # Real (observed family_history_htn, as originally collected in KNHANES).
    p_real = predict(df)

    # Counterfactual: family_history_htn forced to fully missing for everyone,
    # matching how the frozen MICE preprocessors receive it from NHANES/HSE.
    df_blind = df.copy()
    df_blind["family_history_htn"] = np.nan
    p_blind = predict(df_blind)

    def summarize(y_: np.ndarray, p_: np.ndarray) -> dict[str, float]:
        return {
            "auroc": float(roc_auc_score(y_, p_)),
            "auprc": float(average_precision_score(y_, p_)),
            "brier": float(brier_score_loss(y_, p_)),
        }

    real_metrics = summarize(y, p_real)
    blind_metrics = summarize(y, p_blind)

    row = {
        "model_set": model_set, "n": int(len(df)), "events": int(y.sum()),
        "temporal_auroc_real_family_history": real_metrics["auroc"],
        "temporal_auroc_family_history_blinded": blind_metrics["auroc"],
        "delta_auroc_blinding_only": blind_metrics["auroc"] - real_metrics["auroc"],
        "temporal_brier_real": real_metrics["brier"],
        "temporal_brier_blinded": blind_metrics["brier"],
    }

    pred_out = df[["participant_id"]].copy() if "participant_id" in df.columns else pd.DataFrame(index=df.index)
    pred_out["outcome"] = y
    pred_out["pred_real_family_history"] = p_real
    pred_out["pred_family_history_blinded"] = p_blind
    pipe.write_csv(pred_out, DECOMP_DIR / f"{model_set}__family_history_blinding_predictions.csv")

    return row, {
        "model_set": model_set,
        "family_history_htn_missing_rate_before_blinding": float(df["family_history_htn"].isna().mean()),
        "family_history_htn_missing_rate_after_blinding": float(df_blind["family_history_htn"].isna().mean()),
    }


def main() -> int:
    DECOMP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    selected_bundles = manifest["selected_bundles"]
    df_full = pd.read_csv(TEMPORAL_PATH)

    rows, qc_rows = [], []
    for model_set in ["integrated", "male", "female"]:
        row, qc = run_one(model_set, selected_bundles[model_set], df_full)
        rows.append(row)
        qc_rows.append(qc)

    decomposition = pd.DataFrame(rows)
    for ds_key, ds_label in [("nhanes_2021_2023", "NHANES 2021-2023"), ("hse_pooled_2018_2022", "HSE 2018-2022 pooled")]:
        decomposition[f"real_external_auroc_{ds_key}"] = decomposition["model_set"].map(EXTERNAL_AUROC[ds_key])
        decomposition[f"total_drop_{ds_key}"] = decomposition["temporal_auroc_real_family_history"] - decomposition[f"real_external_auroc_{ds_key}"]
        decomposition[f"drop_explained_by_blinding_{ds_key}"] = decomposition["temporal_auroc_real_family_history"] - decomposition["temporal_auroc_family_history_blinded"]
        decomposition[f"pct_of_drop_explained_by_family_history_{ds_key}"] = (
            decomposition[f"drop_explained_by_blinding_{ds_key}"] / decomposition[f"total_drop_{ds_key}"] * 100
        )
        decomposition[f"residual_unexplained_drop_{ds_key}"] = decomposition[f"total_drop_{ds_key}"] - decomposition[f"drop_explained_by_blinding_{ds_key}"]

    pipe.write_csv(decomposition, REPORTS / f"family_history_blinding_decomposition_{STAMP}.csv")
    pipe.write_csv(pd.DataFrame(qc_rows), REPORTS / f"family_history_blinding_qc_{STAMP}.csv")

    def md_table(d: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
        v = d[cols].copy()
        for c in v.columns:
            if pd.api.types.is_float_dtype(v[c]):
                v[c] = v[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in v.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    report_path = REPORTS / f"family_history_blinding_decomposition_{STAMP}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# AUROC Drop Decomposition: family_history_htn Blinding vs Residual Transportability Gap\n\n")
        f.write("Scope: prediction-only diagnostic using the frozen core_sdoh_hr models on the KNHANES 2021-2022 temporal validation cohort. No retraining, retuning, or frozen-model change. `family_history_htn` is set to fully missing for a counterfactual copy of the SAME Korean temporal cohort, isolating the AUROC cost of losing this one predictor from any cross-country transportability effect.\n\n")
        f.write("## Step 1: Blinding-Only Effect (same Korean population, real vs family_history_htn-blinded)\n\n")
        f.write(md_table(decomposition, ["model_set", "n", "events", "temporal_auroc_real_family_history", "temporal_auroc_family_history_blinded", "delta_auroc_blinding_only", "temporal_brier_real", "temporal_brier_blinded"]))
        for ds_key, ds_label in [("nhanes_2021_2023", "NHANES 2021-2023"), ("hse_pooled_2018_2022", "HSE 2018-2022 pooled")]:
            f.write(f"\n\n## Step 2: Decomposition vs Real External Result - {ds_label}\n\n")
            f.write(md_table(decomposition, ["model_set", "temporal_auroc_real_family_history", f"real_external_auroc_{ds_key}", f"total_drop_{ds_key}", f"drop_explained_by_blinding_{ds_key}", f"pct_of_drop_explained_by_family_history_{ds_key}", f"residual_unexplained_drop_{ds_key}"]))
        f.write("\n\n## Interpretation\n\n")
        f.write("- `delta_auroc_blinding_only` is the AUROC change from losing family_history_htn ALONE, holding population, model, and every other predictor fixed (same Korean temporal cohort). This is the cleanest possible estimate of that one predictor's marginal contribution.\n")
        f.write("- `total_drop` is the full observed gap between KNHANES temporal AUROC (real family history) and the real external AUROC on NHANES/HSE.\n")
        f.write("- `pct_of_drop_explained_by_family_history` = how much of the total external drop is attributable to the missing predictor alone vs `residual_unexplained_drop`, which reflects genuine cross-population/measurement/case-mix transportability effects not explained by this one variable.\n")
        f.write("- A small `delta_auroc_blinding_only` relative to a much larger `total_drop` means family_history_htn imputation is a minor contributor and the bulk of the external drop is genuine transportability decay, not an artifact of this specific missing predictor.\n")
        f.write("\n## Output Files\n\n")
        f.write(f"- Decomposition table: `reports/family_history_blinding_decomposition_{STAMP}.csv`\n")
        f.write(f"- Per-participant predictions (real vs blinded): `analysis/knhanes_primary_ml_v1/family_history_blinding_decomposition_{STAMP}/\n")

    print(decomposition.to_string(index=False))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
