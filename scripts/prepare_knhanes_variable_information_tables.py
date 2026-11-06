#!/usr/bin/env python3
"""Prepare KNHANES variable-information tables for workbook authoring.

The output mimics the HSE `Variables` / `Value Labels` workbook structure, but
uses KNHANES final model-development variables. This script only writes
documentation tables; it does not alter raw data, train models, tune models, or
run validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/knhanes_variable_information_20260819"
MAPPING_PATH = ROOT / "reports/knhanes_variable_mapping_primary.csv"
DEV_PATH = ROOT / "data/processed/knhanes_development_2014_2020_primary.csv"


SOURCE_BY_VARIABLE = {
    "participant_id": ("Identification", "Participant metadata", "HNxx_ALL", "2014-2015 `id`; 2016-2020 `ID`."),
    "year": ("Metadata", "Survey year", "HNxx_ALL", "Analysis year variable."),
    "sex": ("Demographics", "Basic demographics", "HNxx_ALL", "Integrated model only; excluded from sex-specific models."),
    "age": ("Demographics", "Basic demographics", "HNxx_ALL", "Age is top-coded at 80 years in KNHANES."),
    "sbp_final": ("Eligibility/outcome", "Blood pressure measurement", "HNxx_ALL", "Outcome derivation only; excluded from predictors."),
    "dbp_final": ("Eligibility/outcome", "Blood pressure measurement", "HNxx_ALL", "Outcome derivation only; excluded from predictors."),
    "known_htn_dx": ("Eligibility/outcome", "Physician diagnosis", "HNxx_ALL", "Exclusion/outcome derivation only; excluded from predictors."),
    "antihypertensive_use": ("Eligibility/outcome", "Medication use", "HNxx_ALL", "Exclusion/outcome derivation only; excluded from predictors."),
    "pregnancy": ("Eligibility/exclusion", "Pregnancy", "HNxx_ALL", "Pregnancy exclusion only; excluded from predictors."),
    "bmi": ("Core predictor", "Examination/anthropometry", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "waist_cm": ("Core predictor", "Examination/anthropometry", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "diabetes_dx": ("Core predictor", "Health interview/chronic disease", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "aerobic_activity": ("Core predictor", "Health interview/physical activity", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "self_rated_health": ("Core predictor", "Health interview/general health", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "activity_limitation": ("Core predictor", "Health interview/activity limitation", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "heart_rate": ("Core+SDOH+HR predictor", "Examination/pulse", "HNxx_ALL", "Primary final predictor set includes heart rate."),
    "family_history_htn": ("Core predictor", "Examination/family history", "HNxx_ALL", "Derived from parent/sibling hypertension family-history items."),
    "current_smoking": ("Core predictor", "Health interview/smoking", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "alcohol_frequency": ("Core predictor", "Health interview/alcohol", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "drink_amount": ("Core predictor", "Health interview/alcohol", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "binge_drinking": ("Core predictor", "Health interview/alcohol", "HNxx_ALL", "Sex-specific raw binge variables are combined before modeling."),
    "education_level": ("SDOH predictor", "Health interview/education", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "household_income_q": ("SDOH predictor", "Health interview/income", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "employment_status": ("SDOH predictor", "Health interview/employment", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "marital_status": ("SDOH predictor", "Health interview/marital status", "HNxx_ALL", "Primary `core_sdoh_hr` predictor."),
    "food_security_status": ("SDOH predictor", "Nutrition survey/food security", "HNxx_ALL; HNxx_24RC documentation", "Primary `core_sdoh_hr` predictor; nutrition-weight sensitivity may be needed for descriptive nutrition analyses."),
    "dyslipidemia_dx_or_med": ("Benchmark-only", "Health interview/chronic disease", "HNxx_ALL", "HIM non-invasive benchmark only; not final primary predictor."),
    "cancer_history": ("Benchmark-only", "Health interview/cancer history", "HNxx_ALL", "HIM non-invasive benchmark only; not final primary predictor."),
    "total_energy_kcal": ("Benchmark-only", "Nutrition survey/intake", "HNxx_24RC", "HIM non-invasive benchmark only; not final primary predictor."),
    "weight": ("Survey design", "Survey weight", "HNxx_ALL", "Descriptive/survey-design metadata, not a model predictor."),
    "strata": ("Survey design", "Survey design stratum", "HNxx_ALL", "Survey-design metadata, not a model predictor."),
    "psu": ("Survey design", "Primary sampling unit", "HNxx_ALL", "Survey-design metadata, not a model predictor."),
}

DEV_COLUMN_MAP = {
    "known_htn_dx": "known_htn_dx_raw",
    "antihypertensive_use": "antihypertensive_use_raw",
    "pregnancy": "pregnancy_raw",
}

VALUE_LABELS = {
    "sex": {1: "male", 2: "female"},
    "undiagnosed_htn": {0: "no", 1: "yes"},
    "elevated_bp_14090": {0: "no", 1: "yes"},
    "known_htn_dx": {0: "no", 1: "yes", 8: "not_applicable", 9: "unknown/no_response"},
    "antihypertensive_use": {1: "use", 2: "use", 3: "use", 4: "use", 5: "non_use", 8: "not_applicable", 9: "unknown/no_response"},
    "pregnancy": {0: "no", 1: "yes", 8: "not_applicable/male"},
    "diabetes_dx": {0: "no", 1: "yes"},
    "aerobic_activity": {0: "does_not_meet_aerobic_activity", 1: "meets_aerobic_activity"},
    "self_rated_health": {1: "very_good", 2: "good", 3: "fair", 4: "poor", 5: "very_poor"},
    "activity_limitation": {0: "no", 1: "yes"},
    "family_history_htn": {0: "no_parent_or_sibling_history", 1: "any_parent_or_sibling_history"},
    "current_smoking": {0: "non_current", 1: "current"},
    "alcohol_frequency": {
        0: "none_or_never",
        1: "less_than_monthly",
        2: "monthly",
        3: "2_4_per_month",
        4: "2_3_per_week",
        5: "4plus_per_week",
    },
    "drink_amount": {
        0: "none_or_not_applicable",
        1: "1_2_drinks",
        2: "3_4_drinks",
        3: "5_6_drinks",
        4: "7_9_drinks",
        5: "10plus_drinks",
    },
    "binge_drinking": {
        0: "none_or_not_applicable",
        1: "never",
        2: "less_than_monthly",
        3: "monthly",
        4: "weekly",
        5: "almost_daily",
    },
    "education_level": {1: "elementary_or_less", 2: "middle_school", 3: "high_school", 4: "college_plus"},
    "household_income_q": {1: "low", 2: "mid_low", 3: "mid_high", 4: "high"},
    "employment_status": {1: "employed", 2: "not_employed"},
    "marital_status": {
        0: "never_married",
        1: "married_living_together",
        2: "married_separated",
        3: "widowed",
        4: "divorced",
    },
    "food_security_status": {1: "secure", 2: "mild_insecure", 3: "moderate_insecure", 4: "severe_insecure"},
    "dyslipidemia_dx_or_med": {0: "no", 1: "yes"},
    "cancer_history": {0: "no", 1: "yes"},
}


def missing_note(dev: pd.DataFrame, variable: str) -> str:
    column = DEV_COLUMN_MAP.get(variable, variable)
    if column not in dev.columns:
        return "Development missingness not applicable/not in processed development file."
    missing_n = int(dev[column].isna().sum())
    missing_pct = missing_n / len(dev) * 100
    return f"Development 2014-2020 missing: {missing_n:,}/{len(dev):,} ({missing_pct:.2f}%)."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING_PATH)
    dev = pd.read_csv(DEV_PATH)

    variable_rows = []
    raw_mapping_rows = []
    for row in mapping.to_dict("records"):
        variable = row["final_variable"]
        topic, section, source, source_note = SOURCE_BY_VARIABLE.get(
            variable,
            (row["role"], "KNHANES source variable", "HNxx_ALL", ""),
        )
        notes = [
            f"Raw variable(s): {row['raw_variable']}.",
            f"Coding/derivation: {row['coding_or_derivation']}.",
            f"Role: {row['role']}.",
            f"Final model input: {row['included_in_final_model_input']}.",
            source_note,
            missing_note(dev, variable),
        ]
        variable_rows.append(
            {
                "Variable": variable,
                "Description": row["description"],
                "Topic": topic,
                "Section": section,
                "Source": source,
                "Notes": " ".join(x for x in notes if x),
            }
        )
        raw_mapping_rows.append(
            {
                "final_variable": variable,
                "raw_variable": row["raw_variable"],
                "description": row["description"],
                "coding_or_derivation": row["coding_or_derivation"],
                "role": row["role"],
                "included_in_final_model_input": row["included_in_final_model_input"],
                "topic": topic,
                "section": section,
                "source": source,
            }
        )

    value_rows = []
    variable_lookup = {row["Variable"]: row for row in variable_rows}
    for variable, labels in VALUE_LABELS.items():
        meta = variable_lookup.get(variable)
        if not meta:
            continue
        for value, label in labels.items():
            value_rows.append(
                {
                    "Variable": variable,
                    "Topic": meta["Topic"],
                    "Section": meta["Section"],
                    "Source": meta["Source"],
                    "Value": value,
                    "Value Label": label,
                }
            )

    primary_input_rows = []
    primary_vars = [
        "age",
        "bmi",
        "waist_cm",
        "heart_rate",
        "sex",
        "diabetes_dx",
        "aerobic_activity",
        "self_rated_health",
        "activity_limitation",
        "family_history_htn",
        "current_smoking",
        "alcohol_frequency",
        "drink_amount",
        "binge_drinking",
        "education_level",
        "household_income_q",
        "employment_status",
        "marital_status",
        "food_security_status",
    ]
    variable_by_name = {row["final_variable"]: row for row in mapping.to_dict("records")}
    for variable in primary_vars:
        row = variable_by_name[variable]
        primary_input_rows.append(
            {
                "Variable": variable,
                "Description": row["description"],
                "Raw variable(s)": row["raw_variable"],
                "Integrated model": "Yes",
                "Male model": "No" if variable == "sex" else "Yes",
                "Female model": "No" if variable == "sex" else "Yes",
                "Coding/derivation": row["coding_or_derivation"],
                "Development missingness": missing_note(dev, variable).replace("Development 2014-2020 missing: ", ""),
            }
        )

    intro_rows = [
        ["Item", "Value"],
        ["Dataset", "KNHANES 2014-2020 development dataset"],
        ["Purpose", "Variable information workbook for frozen primary `core_sdoh_hr` model inputs and related outcome/exclusion/design variables"],
        ["Reference format", "Mirrors HSE `9469_hse_2022_eul_variable_information_v1.xlsx` with `Variables` and `Value Labels` sheets"],
        ["Final primary predictor set", "core_sdoh_hr"],
        ["Integrated model predictors", "19 predictors including sex"],
        ["Sex-specific model predictors", "18 predictors; sex excluded"],
        ["Outcome", "undiagnosed hypertension: SBP >=140 or DBP >=90 among participants without prior diagnosed hypertension and not using antihypertensive medication"],
        ["Development N/events", "N=30,323; events=3,528"],
        ["Source mapping file", str(MAPPING_PATH.relative_to(ROOT))],
        ["Development data file", str(DEV_PATH.relative_to(ROOT))],
    ]

    payload = {
        "introduction": intro_rows,
        "variables": variable_rows,
        "value_labels": value_rows,
        "primary_model_input": primary_input_rows,
        "raw_mapping": raw_mapping_rows,
    }
    out_json = OUT_DIR / "knhanes_variable_information_payload_20260819.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(variable_rows).to_csv(OUT_DIR / "knhanes_variable_information_variables_20260819.csv", index=False)
    pd.DataFrame(value_rows).to_csv(OUT_DIR / "knhanes_variable_information_value_labels_20260819.csv", index=False)
    pd.DataFrame(primary_input_rows).to_csv(OUT_DIR / "knhanes_primary_model_input_variables_20260819.csv", index=False)
    print(out_json)


if __name__ == "__main__":
    main()
