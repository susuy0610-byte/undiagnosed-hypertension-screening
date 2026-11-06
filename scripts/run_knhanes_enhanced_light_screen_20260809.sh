#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_DIR="analysis/knhanes_enhanced_light_screen_20260809"
mkdir -p "${OUT_DIR}/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${OUT_DIR}/logs/light_screen_run_${STAMP}.out"
PID_FILE="${OUT_DIR}/logs/enhanced_light_screen_shell.pid"

echo "$$" > "${PID_FILE}"
echo "START_KST=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${RUN_LOG}"
echo "RUN_LOG=${RUN_LOG}" | tee -a "${RUN_LOG}"

.venv/bin/python scripts/knhanes_primary_ml_pipeline.py \
  --stage cv \
  --output-dir "${OUT_DIR}" \
  --dev-path data/processed/knhanes_development_2014_2020_conservative_enhanced.csv \
  --time-path data/processed/knhanes_time_validation_2021_2022_conservative_enhanced.csv \
  --variable-sets enhanced_conservative \
  --model-sets integrated male female \
  --algorithms logistic lasso xgboost \
  --n-trials 10 \
  --m-imputations 20 \
  --n-folds 5 \
  --inner-folds 3 \
  --seed 20260722 \
  --n-jobs -1 2>&1 | tee -a "${RUN_LOG}"

echo "END_KST=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${RUN_LOG}"
