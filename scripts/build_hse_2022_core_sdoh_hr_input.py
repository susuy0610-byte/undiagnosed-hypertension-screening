#!/usr/bin/env python3
"""Build HSE 2022 harmonized inputs for frozen core_sdoh_hr validation.

This script creates participant-level external-validation input files only. It
does not train, tune, calibrate, select variables, set thresholds, or run model
prediction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata/stata/stata13/hse_2022_eul_v1.dta"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
STAMP = "20260817"

PREDICTORS_INTEGRATED = [
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
PREDICTORS_SEX_SPECIFIC = [x for x in PREDICTORS_INTEGRATED if x != "sex"]

VALID_CATEGORIES = {
    "sex": [1, 2],
    "diabetes_dx": [0, 1],
    "aerobic_activity": [0, 1],
    "self_rated_health": [1, 2, 3, 4, 5],
    "activity_limitation": [0, 1],
    "family_history_htn": [0, 1],
    "current_smoking": [0, 1],
    "alcohol_frequency": [0, 1, 2, 3, 4, 5],
    "drink_amount": [0, 1, 2, 3, 4, 5],
    "binge_drinking": [0, 1, 2, 3, 4, 5],
    "education_level": [1, 2, 3, 4],
    "household_income_q": [1, 2, 3, 4],
    "employment_status": [1, 2],
    "marital_status": [0, 1, 2, 3, 4],
    "food_security_status": [1, 2, 3, 4],
}

NEGATIVE_MISSING = {-99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -9, -8, -7, -6, -5, -4, -3, -2, -1}


def clean_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x.isin(NEGATIVE_MISSING))


def yes_no_12(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[x == 1] = 1.0
    out[x == 2] = 0.0
    return out


def zero_one(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[x == 1] = 1.0
    out[x == 0] = 0.0
    return out


def age_midpoint(age16g5: pd.Series) -> pd.Series:
    raw = clean_numeric(age16g5)
    mapping = {
        3: 22.5,
        4: 27.5,
        5: 32.5,
        6: 37.5,
        7: 42.5,
        8: 47.5,
        9: 52.5,
        10: 57.5,
        11: 62.5,
        12: 67.5,
        13: 72.5,
        14: 77.5,
        # KNHANES development top-coded age at 80, so HSE 80+ groups are
        # harmonized to 80 instead of extrapolated outside the training range.
        15: 80.0,
        16: 80.0,
        17: 80.0,
    }
    return raw.map(mapping).astype("float")


def valid_measure(s: pd.Series, low: float, high: float) -> pd.Series:
    x = clean_numeric(s)
    return x.where(x.between(low, high), np.nan)


def mean_available(df: pd.DataFrame, cols: list[str], low: float, high: float) -> pd.Series:
    values = [valid_measure(df[col], low, high) for col in cols if col in df.columns]
    if not values:
        return pd.Series(np.nan, index=df.index, dtype="float")
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def derive_aerobic_activity(df: pd.DataFrame) -> pd.Series:
    mvpa = clean_numeric(df["mvpamwk"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[mvpa.notna()] = (mvpa[mvpa.notna()] >= 150).astype(float)
    return out


def derive_activity_limitation(df: pd.DataFrame) -> pd.Series:
    # limitill: 1 limiting longstanding illness, 2 non-limiting LI, 3 no LI.
    raw = clean_numeric(df["limitill"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw.isin([2, 3])] = 0.0
    return out


def derive_current_smoking(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["cigsta3_19"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw.isin([2, 3])] = 0.0
    return out


def derive_alcohol_frequency(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["dnoft_22"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 8] = 0.0
    out[raw.isin([6, 7])] = 1.0
    out[raw == 5] = 2.0
    out[raw == 4] = 3.0
    out[raw == 3] = 4.0
    out[raw.isin([1, 2])] = 5.0
    return out


def derive_drink_amount(df: pd.DataFrame, alcohol_frequency: pd.Series) -> pd.Series:
    raw = clean_numeric(df["AUDTyDy"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[alcohol_frequency == 0] = 0.0
    out[raw.isin([1, 2, 3, 4, 5])] = raw[raw.isin([1, 2, 3, 4, 5])]
    return out


def derive_binge_drinking(df: pd.DataFrame, alcohol_frequency: pd.Series) -> pd.Series:
    raw = clean_numeric(df["AUD6Mre"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[alcohol_frequency == 0] = 0.0
    out[raw.isin([1, 2, 3, 4, 5])] = raw[raw.isin([1, 2, 3, 4, 5])]
    return out


def derive_education(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["educ2"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 3] = 1.0
    out[raw == 2] = 3.0
    out[raw == 1] = 4.0
    return out


def derive_income_quartile(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["eqv5"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw == 2] = 2.0
    out[raw == 3] = 3.0
    out[raw.isin([4, 5])] = 4.0
    return out


def derive_employment(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["activb2"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw.isin([2, 3])] = 1.0
    out[raw.isin([1, 4, 5, 6, 7, 8, 9, 95])] = 2.0
    return out


def derive_marital(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["marstatd_22"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 0.0
    out[raw.isin([2, 6])] = 1.0
    out[raw == 3] = 2.0
    out[raw == 5] = 3.0
    out[raw == 4] = 4.0
    return out


def derive_pregnancy(df: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(df["PregNTJ"], errors="coerce")
    out = pd.Series(0.0, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw == -8] = np.nan
    return out


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["person_id"] = clean_numeric(df["seriala"]).astype("Int64").astype("string")
    out["source_dataset"] = "HSE 2022"
    out["age_band_raw"] = clean_numeric(df["age16g5"])
    out["age"] = age_midpoint(df["age16g5"])
    out["sex"] = clean_numeric(df["Sex"]).where(clean_numeric(df["Sex"]).isin([1, 2]), np.nan)
    out["bmi"] = valid_measure(df["bmival"], 10, 80)
    out["waist_cm"] = valid_measure(df["wstval"], 40, 220)
    out["heart_rate"] = mean_available(df, ["PULS2OM", "PULS3OM"], 30, 220)
    out["sbp_final"] = valid_measure(df["omsysval"], 50, 300)
    out["dbp_final"] = valid_measure(df["omdiaval"], 20, 200)
    out["known_htn_dx"] = yes_no_12(df["bp1"])
    out["antihypertensive_use"] = zero_one(df["bpmedd2"])
    out["pregnancy"] = derive_pregnancy(df)
    out["diabetes_dx"] = yes_no_12(df["diabete2"])
    out["aerobic_activity"] = derive_aerobic_activity(df)
    out["self_rated_health"] = clean_numeric(df["GenHelf"]).where(clean_numeric(df["GenHelf"]).isin([1, 2, 3, 4, 5]), np.nan)
    out["activity_limitation"] = derive_activity_limitation(df)
    out["family_history_htn"] = np.nan
    out["current_smoking"] = derive_current_smoking(df)
    out["alcohol_frequency"] = derive_alcohol_frequency(df)
    out["drink_amount"] = derive_drink_amount(df, out["alcohol_frequency"])
    out["binge_drinking"] = derive_binge_drinking(df, out["alcohol_frequency"])
    out["education_level"] = derive_education(df)
    out["household_income_q"] = derive_income_quartile(df)
    out["employment_status"] = derive_employment(df)
    out["marital_status"] = derive_marital(df)
    out["food_security_status"] = np.nan
    out["outcome"] = ((out["sbp_final"] >= 140) | (out["dbp_final"] >= 90)).astype(float)

    # QC raw columns retained for traceability, not used by frozen model.
    for col in [
        "age16g5",
        "PULS2OM",
        "PULS3OM",
        "omsysval",
        "omdiaval",
        "bp1",
        "bpmedd2",
        "PregNTJ",
        "mvpamwk",
        "AUDTyDy",
        "AUD6Mre",
        "limitill",
        "dnoft_22",
    ]:
        out[f"raw_{col}"] = df[col]
    return out


def flow_count(rows: list[dict[str, Any]], name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    rows.append({"step": name, "n_remaining": int(len(after)), "n_excluded_step": int(len(before) - len(after))})


def apply_cohort_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cur = df.copy()
    rows.append({"step": "raw_hse_2022_eul", "n_remaining": int(len(cur)), "n_excluded_step": 0})
    before = cur
    cur = cur[cur["age_band_raw"] >= 3].copy()
    flow_count(rows, "age_20plus_due_to_public_5yr_age_band", before, cur)
    before = cur
    cur = cur[cur["sex"].isin([1, 2])].copy()
    flow_count(rows, "valid_sex", before, cur)
    before = cur
    cur = cur[cur["sbp_final"].notna() & cur["dbp_final"].notna()].copy()
    flow_count(rows, "valid_blood_pressure_measurement", before, cur)
    before = cur
    cur = cur[cur["known_htn_dx"].isin([0, 1])].copy()
    flow_count(rows, "known_hypertension_diagnosis_determinable", before, cur)
    before = cur
    cur = cur[cur["known_htn_dx"] != 1].copy()
    flow_count(rows, "exclude_self_reported_hypertension_diagnosis", before, cur)
    before = cur
    cur = cur[cur["antihypertensive_use"].isin([0, 1])].copy()
    flow_count(rows, "antihypertensive_medication_determinable", before, cur)
    before = cur
    cur = cur[cur["antihypertensive_use"] != 1].copy()
    flow_count(rows, "exclude_antihypertensive_medication_use", before, cur)
    before = cur
    cur = cur[cur["pregnancy"].fillna(0) != 1].copy()
    flow_count(rows, "exclude_current_pregnancy", before, cur)
    return cur, pd.DataFrame(rows)


def participant_missing_filter(df: pd.DataFrame, model_set: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model_set == "integrated":
        features = PREDICTORS_INTEGRATED
        sub = df.copy()
    elif model_set == "male":
        features = PREDICTORS_SEX_SPECIFIC
        sub = df[df["sex"] == 1].copy()
    elif model_set == "female":
        features = PREDICTORS_SEX_SPECIFIC
        sub = df[df["sex"] == 2].copy()
    else:
        raise ValueError(model_set)
    before = len(sub)
    miss_frac = sub[features].isna().mean(axis=1)
    sub = sub[miss_frac <= 0.50].copy()
    after_frac = sub[features].isna().mean(axis=1) if len(sub) else pd.Series([], dtype=float)
    qc = {
        "model_set": model_set,
        "n_before_predictor_missing_filter": int(before),
        "n_after_predictor_missing_filter": int(len(sub)),
        "n_excluded_predictor_missing_fraction_gt_50pct": int(before - len(sub)),
        "feature_count": len(features),
        "max_participant_missing_fraction_after_filter": float(after_frac.max()) if len(sub) else math.nan,
        "mean_participant_missing_fraction_after_filter": float(after_frac.mean()) if len(sub) else math.nan,
    }
    keep_cols = [
        "person_id",
        "source_dataset",
        "age_band_raw",
        "sbp_final",
        "dbp_final",
        "known_htn_dx",
        "antihypertensive_use",
        "pregnancy",
        "outcome",
    ] + PREDICTORS_INTEGRATED
    return sub[keep_cols].copy(), qc


def missingness_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        features = PREDICTORS_INTEGRATED if model_set == "integrated" else PREDICTORS_SEX_SPECIFIC
        for col in features:
            rows.append(
                {
                    "model_set": model_set,
                    "predictor": col,
                    "n": int(len(df)),
                    "missing_n": int(df[col].isna().sum()),
                    "missing_rate": float(df[col].isna().mean()) if len(df) else math.nan,
                    "all_missing": bool(df[col].isna().all()) if len(df) else False,
                }
            )
    return pd.DataFrame(rows)


def category_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        for col, allowed in VALID_CATEGORIES.items():
            if col not in df.columns or (model_set != "integrated" and col == "sex"):
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            observed = sorted(vals.dropna().round().astype(int).unique().tolist())
            rows.append(
                {
                    "model_set": model_set,
                    "predictor": col,
                    "allowed_levels": json.dumps(allowed),
                    "observed_levels": json.dumps(observed),
                    "unseen_levels": json.dumps([x for x in observed if x not in allowed]),
                }
            )
    return pd.DataFrame(rows)


def mapping_table() -> pd.DataFrame:
    rows = [
        ("age", "age16g5", "Proxy", "Include age16g5>=3 (20+); map 20-79 bands to midpoints and 80+ to KNHANES top-code 80."),
        ("sex", "Sex", "Direct", "1 male, 2 female."),
        ("bmi", "bmival", "Direct", "Valid measured BMI kg/m2."),
        ("waist_cm", "wstval", "Direct", "Valid mean waist circumference cm."),
        ("heart_rate", "mean(PULS2OM, PULS3OM)", "Harmonization", "Mean of 2nd and 3rd Omron pulse readings in bpm; 999/out-of-range excluded."),
        ("sbp_final", "omsysval", "Direct", "Outcome only; valid Omron mean systolic BP."),
        ("dbp_final", "omdiaval", "Direct", "Outcome only; valid Omron mean diastolic BP."),
        ("known_htn_dx", "bp1", "Direct", "Doctor diagnosed high BP excluding pregnancy; exclusion/outcome only."),
        ("antihypertensive_use", "bpmedd2", "Direct", "Taking drugs prescribed for blood pressure; exclusion/outcome only."),
        ("pregnancy", "PregNTJ", "Direct", "Currently pregnant 16+; exclusion only."),
        ("diabetes_dx", "diabete2", "Direct", "Doctor diagnosed diabetes excluding pregnant."),
        ("aerobic_activity", "mvpamwk", "Harmonization", "IPAQ weekly MVPA minutes >=150."),
        ("self_rated_health", "GenHelf", "Direct", "1 very good to 5 very bad."),
        ("activity_limitation", "limitill", "Harmonization", "Limiting longstanding illness=1; non-limiting/no LI=0."),
        ("family_history_htn", "not available", "Not available", "Kept all-NA; no proxy substitution."),
        ("current_smoking", "cigsta3_19", "Harmonization", "Current cigarette smoker=1; ex/never=0."),
        ("alcohol_frequency", "dnoft_22", "Harmonization", "Past-year drinking frequency collapsed to KNHANES 0-5 direction."),
        ("drink_amount", "AUDTyDy", "Direct/Harmonization", "Typical drinks on drinking day; 1-5 maps to frozen 1-5, non-drinker=0."),
        ("binge_drinking", "AUD6Mre", "Harmonization", "AUDIT six-plus drinks frequency; frequency scale maps to 0-5 but threshold differs from KNHANES sex-specific definition."),
        ("education_level", "educ2", "Harmonization", "Degree=4, below degree=3, none=1; middle-school category cannot be separated."),
        ("household_income_q", "eqv5", "Harmonization", "Equivalised income quintile; lowest=1, second=2, middle=3, upper two quintiles=4."),
        ("employment_status", "activb2", "Harmonization", "Paid/self-employed/unpaid family business=1; all other activity statuses=2."),
        ("marital_status", "marstatd_22", "Harmonization", "Single=0, married/civil/cohabiting=1, separated=2, widowed=3, divorced=4."),
        ("food_security_status", "not available", "Not available", "Kept all-NA; no benefit/income proxy substitution."),
    ]
    return pd.DataFrame(rows, columns=["knhanes_variable", "hse_source", "mapping_class", "rule"])


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_stata(RAW, convert_categoricals=False)
    harmonized = harmonize(raw)
    cohort, flow = apply_cohort_rules(harmonized)
    harmonized.to_csv(PROCESSED / f"hse_2022_core_sdoh_hr_harmonized_qc_{STAMP}.csv", index=False)

    inputs: dict[str, pd.DataFrame] = {}
    compat_rows = []
    for model_set in ["integrated", "male", "female"]:
        model_df, qc = participant_missing_filter(cohort, model_set)
        inputs[model_set] = model_df
        compat_rows.append(qc)
        model_df.to_csv(PROCESSED / f"hse_2022_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv", index=False)

    flow.to_csv(REPORTS / f"hse_2022_core_sdoh_hr_flow_{STAMP}.csv", index=False)
    pd.DataFrame(compat_rows).to_csv(REPORTS / f"hse_2022_core_sdoh_hr_input_compatibility_{STAMP}.csv", index=False)
    missingness_table(inputs).to_csv(REPORTS / f"hse_2022_core_sdoh_hr_predictor_missingness_{STAMP}.csv", index=False)
    category_table(inputs).to_csv(REPORTS / f"hse_2022_core_sdoh_hr_category_level_qc_{STAMP}.csv", index=False)
    mapping_table().to_csv(REPORTS / f"hse_2022_core_sdoh_hr_mapping_final_{STAMP}.csv", index=False)

    print("HSE 2022 harmonized inputs created")
    print(flow.to_string(index=False))
    print(pd.DataFrame(compat_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
