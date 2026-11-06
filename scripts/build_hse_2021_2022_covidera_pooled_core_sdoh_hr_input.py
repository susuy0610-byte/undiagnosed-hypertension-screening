#!/usr/bin/env python3
"""Build HSE 2018+2019 (pre-COVID) POOLED harmonized inputs for frozen core_sdoh_hr validation.

Pre-specified sensitivity split: COVID-era years (2021, 2022) vs COVID-era years
(2021, 2022), grounded in HSE's own documented 2021 mode change (remote
interviews, reduced nurse-visit/measurement completion), NOT chosen based on
which individual year scored highest. Reuses the exact same harmonization
functions already QC'd for the full 4-year pool
(build_hse_2018_2019_2021_2022_pooled_core_sdoh_hr_input.py) - only the set of
years pooled changes.

This script creates participant-level external-validation input files only. It
does not train, tune, calibrate, select variables, set thresholds, or run model
prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_hse_2018_2019_2021_2022_pooled_core_sdoh_hr_input as full4  # noqa: E402

PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
STAMP = "20260819"
YEARS = ["2021", "2022"]
TAG = "covidera_2021_2022"


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    harmonized_parts = []
    for year in YEARS:
        raw = pd.read_stata(full4.RAW_FILES[year], convert_categoricals=False)
        harmonized_parts.append(full4.harmonize_year(raw, year))
    harmonized = pd.concat(harmonized_parts, ignore_index=True)
    if harmonized["person_id"].duplicated().any():
        raise RuntimeError("Duplicate person_id after year-prefixing; check id construction")
    harmonized.to_csv(PROCESSED / f"hse_{TAG}_pooled_core_sdoh_hr_harmonized_qc_{STAMP}.csv", index=False)

    cohort, flow = full4.apply_cohort_rules(harmonized)
    flow_by_year_rows = [{"source_year": y, "n_in_final_cohort_before_predictor_filter": int((cohort["source_year"] == y).sum())} for y in YEARS]
    pd.DataFrame(flow_by_year_rows).to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_cohort_by_year_{STAMP}.csv", index=False)

    inputs: dict[str, pd.DataFrame] = {}
    compat_rows = []
    for model_set in ["integrated", "male", "female"]:
        model_df, qc = full4.participant_missing_filter(cohort, model_set)
        inputs[model_set] = model_df
        compat_rows.append(qc)
        model_df.to_csv(PROCESSED / f"hse_{TAG}_pooled_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv", index=False)

    flow.to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_flow_{STAMP}.csv", index=False)
    pd.DataFrame(compat_rows).to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_input_compatibility_{STAMP}.csv", index=False)
    full4.missingness_table(inputs).to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_predictor_missingness_{STAMP}.csv", index=False)
    full4.category_table(inputs).to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_category_level_qc_{STAMP}.csv", index=False)
    full4.mapping_table().to_csv(REPORTS / f"hse_{TAG}_core_sdoh_hr_mapping_final_{STAMP}.csv", index=False)

    print(f"HSE {'+'.join(YEARS)} (COVID-era) pooled harmonized inputs created")
    print(flow.to_string(index=False))
    print(pd.DataFrame(flow_by_year_rows).to_string(index=False))
    print(pd.DataFrame(compat_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
