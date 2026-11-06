#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="analysis/knhanes_bmi_who_waist_rcs_6alg_screen_20260812"
SEED="${SEED:-20260722}"
N_TRIALS="${N_TRIALS:-5}"

"$PYTHON_BIN" scripts/knhanes_primary_ml_pipeline.py \
  --stage cv \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  --n-folds 5 \
  --inner-folds 3 \
  --m-imputations 20 \
  --n-trials "$N_TRIALS" \
  --target-sensitivity 0.80 \
  --n-jobs -1 \
  --model-sets integrated male female \
  --variable-sets core_sdoh_hr_bmi_who_waist_rcs \
  --algorithms logistic lasso random_forest extra_trees xgboost lightgbm

"$PYTHON_BIN" scripts/knhanes_primary_ml_pipeline.py \
  --stage summarize \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  --n-folds 5 \
  --inner-folds 3 \
  --m-imputations 20 \
  --n-trials "$N_TRIALS" \
  --target-sensitivity 0.80 \
  --n-jobs -1 \
  --model-sets integrated male female \
  --variable-sets core_sdoh_hr_bmi_who_waist_rcs \
  --algorithms logistic lasso random_forest extra_trees xgboost lightgbm

"$PYTHON_BIN" scripts/compare_bmi_who_waist_rcs_screen_to_primary.py
