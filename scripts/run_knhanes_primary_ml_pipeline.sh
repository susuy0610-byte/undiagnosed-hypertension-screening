#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
STAGE="${STAGE:-cv}"
N_TRIALS="${N_TRIALS:-40}"
export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib_cache"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing .venv Python. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/knhanes_primary_ml_pipeline.py \
  --stage "$STAGE" \
  --output-dir analysis/knhanes_primary_ml_v1 \
  --seed 20260722 \
  --n-folds 5 \
  --inner-folds 3 \
  --m-imputations 20 \
  --n-trials "$N_TRIALS" \
  --target-sensitivity 0.80 \
  --n-jobs -1 \
  "$@"
