#!/usr/bin/env python3
"""Build HSE 2018/2019/2021/2022 POOLED harmonized inputs for frozen core_sdoh_hr validation.

Extends the established HSE 2022 single-year build (build_hse_2022_core_sdoh_hr_input.py)
to four pooled years, using the exact raw variable names confirmed in
reports/hse_2018_2019_2021_2022_raw_core_sdoh_hr_mapping_qc_20260819.csv.

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
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
STAMP = "20260819"

RAW_FILES = {
    "2018": ROOT / "data/raw/hse/2018/UKDA-8649-stata/stata/stata13/hse_2018_eul_15082022.dta",
    "2019": ROOT / "data/raw/hse/2019/UKDA-8860-stata/stata/stata13/hse_2019_eul_20211006.dta",
    "2021": ROOT / "data/raw/hse/2021/UKDA-9319-stata/stata/stata13/hse_2021_eul_v1.dta",
    "2022": ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata/stata/stata13/hse_2022_eul_v1.dta",
}

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
        3: 22.5, 4: 27.5, 5: 32.5, 6: 37.5, 7: 42.5, 8: 47.5, 9: 52.5,
        10: 57.5, 11: 62.5, 12: 67.5, 13: 72.5, 14: 77.5,
        # KNHANES development top-coded age at 80.
        15: 80.0, 16: 80.0, 17: 80.0,
    }
    return raw.map(mapping).astype("float")


def valid_measure(s: pd.Series, low: float, high: float) -> pd.Series:
    x = clean_numeric(s)
    return x.where(x.between(low, high), np.nan)


def mean_available(df: pd.DataFrame, cols: list[str], low: float, high: float) -> pd.Series:
    values = [valid_measure(df[c], low, high) for c in cols if c in df.columns]
    if not values:
        return pd.Series(np.nan, index=df.index, dtype="float")
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def derive_bmi(df: pd.DataFrame, year: str) -> pd.Series:
    if year in {"2018", "2019", "2022"}:
        return valid_measure(df["bmival"], 10, 80)
    # 2021: no BMIVal in the raw file (COVID-year protocol change). Derive from
    # final measured/estimated height and weight. This is the only measured-BMI
    # source available for 2021; it is retained rather than falling back to
    # self-reported BMI (bmisr*), which would mix a different construct in.
    height = valid_measure(df["esthtn"], 120, 230)
    weight = valid_measure(df["estwtn"], 25, 250)
    bmi = weight / ((height / 100) ** 2)
    return bmi.where(bmi.between(10, 80), np.nan)


def derive_aerobic_activity(df: pd.DataFrame, year: str) -> pd.Series:
    if year == "2019":
        return pd.Series(np.nan, index=df.index, dtype="float")
    if year == "2018":
        mvpa = clean_numeric(df["mvpamwk"])
    elif year == "2021":
        # Proxy: sport-only MVPA minutes/week (narrower than the IPAQ-based
        # all-domain MVPA used in 2018/2022; likely under-counts activity).
        mvpa = clean_numeric(df["mins10spta"])
    else:
        mvpa = clean_numeric(df["mvpamwk"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[mvpa.notna()] = (mvpa[mvpa.notna()] >= 150).astype(float)
    return out


def derive_activity_limitation(df: pd.DataFrame) -> pd.Series:
    raw = clean_numeric(df["limitill"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw.isin([2, 3])] = 0.0
    return out


def derive_current_smoking(df: pd.DataFrame, year: str) -> pd.Series:
    if year == "2018":
        # cigst1 category order is Never/Ex-reg/Ex-occ/Current (1/2/3/4) -
        # DIFFERENT from cigsta3_19's Current/Ex-Reg/Never-Reg (1/2/3) order.
        raw = clean_numeric(df["cigst1"])
        out = pd.Series(np.nan, index=df.index, dtype="float")
        out[raw == 4] = 1.0
        out[raw.isin([1, 2, 3])] = 0.0
        return out
    raw = clean_numeric(df["cigsta3_19"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw.isin([2, 3])] = 0.0
    return out


def derive_alcohol_frequency(df: pd.DataFrame, year: str) -> pd.Series:
    col = {"2018": "dnoft", "2019": "dnoft_19", "2021": "dnoft_19", "2022": "dnoft_22"}[year]
    raw = clean_numeric(df[col])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 8] = 0.0
    out[raw.isin([6, 7])] = 1.0
    out[raw == 5] = 2.0
    out[raw == 4] = 3.0
    out[raw == 3] = 4.0
    out[raw.isin([1, 2])] = 5.0
    return out


def derive_drink_amount(df: pd.DataFrame, year: str, alcohol_frequency: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float")
    if year == "2022":
        raw = clean_numeric(df["audtydy"])
        out[alcohol_frequency == 0] = 0.0
        out[raw.isin([1, 2, 3, 4, 5])] = raw[raw.isin([1, 2, 3, 4, 5])]
        return out
    # Proxy for 2018/2019/2021: units drunk on heaviest day in last 7 days,
    # banded 1-7 (NOT the same construct as "typical drinking day"; heaviest-day
    # units skew higher than a typical day). Capped at 5 to match frozen scale.
    col = "d7unitwgrp" if year == "2018" else "d7unitwgrp_19"
    raw = clean_numeric(df[col])
    out[alcohol_frequency == 0] = 0.0
    has_band = raw.between(1, 7)
    out[has_band] = raw[has_band].clip(upper=5)
    return out


def derive_binge_drinking(df: pd.DataFrame, year: str, alcohol_frequency: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float")
    if year == "2022":
        raw = clean_numeric(df["aud6mre"])
        out[alcohol_frequency == 0] = 0.0
        out[raw.isin([1, 2, 3, 4, 5])] = raw[raw.isin([1, 2, 3, 4, 5])]
        return out
    # Proxy for 2018/2019/2021: alcohol-unit risk-limit band derived from
    # heaviest-day units (0-3), NOT an AUDIT 6+-drinks frequency. Different
    # construct (risk-band classification vs. binge frequency); passed through
    # as-is since 0-3 is a valid subset of the frozen 0-5 category range.
    col = "alclimit07b" if year == "2018" else "alclimit7b_19"
    raw = clean_numeric(df[col])
    out[alcohol_frequency == 0] = 0.0
    has_band = raw.isin([0, 1, 2, 3])
    out[has_band] = raw[has_band]
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


def derive_marital(df: pd.DataFrame, year: str) -> pd.Series:
    col = "marstatd_22" if year == "2022" else "marstatd"
    raw = clean_numeric(df[col])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 0.0
    out[raw.isin([2, 6])] = 1.0
    out[raw == 3] = 2.0
    out[raw == 5] = 3.0
    out[raw == 4] = 4.0
    return out


def derive_pregnancy(df: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(df["pregntj"], errors="coerce")
    out = pd.Series(0.0, index=df.index, dtype="float")
    out[raw == 1] = 1.0
    out[raw == -8] = np.nan
    return out


def harmonize_year(df: pd.DataFrame, year: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    out["person_id"] = year + "_" + clean_numeric(df["seriala"]).astype("Int64").astype("string")
    out["source_dataset"] = f"HSE {year}"
    out["source_year"] = year
    out["age_band_raw"] = clean_numeric(df["age16g5"])
    out["age"] = age_midpoint(df["age16g5"])
    out["sex"] = clean_numeric(df["sex"]).where(clean_numeric(df["sex"]).isin([1, 2]), np.nan)
    out["bmi"] = derive_bmi(df, year)
    out["waist_cm"] = valid_measure(df["wstval"], 40, 220)
    out["heart_rate"] = mean_available(df, ["puls2om", "puls3om"], 30, 220)
    out["sbp_final"] = valid_measure(df["omsysval"], 50, 300)
    out["dbp_final"] = valid_measure(df["omdiaval"], 20, 200)
    out["known_htn_dx"] = yes_no_12(df["bp1"])
    out["antihypertensive_use"] = zero_one(df["bpmedd2"])
    out["pregnancy"] = derive_pregnancy(df)
    out["diabetes_dx"] = yes_no_12(df["diabete2"])
    out["aerobic_activity"] = derive_aerobic_activity(df, year)
    out["self_rated_health"] = clean_numeric(df["genhelf"]).where(clean_numeric(df["genhelf"]).isin([1, 2, 3, 4, 5]), np.nan)
    out["activity_limitation"] = derive_activity_limitation(df)
    out["family_history_htn"] = np.nan
    out["current_smoking"] = derive_current_smoking(df, year)
    out["alcohol_frequency"] = derive_alcohol_frequency(df, year)
    out["drink_amount"] = derive_drink_amount(df, year, out["alcohol_frequency"])
    out["binge_drinking"] = derive_binge_drinking(df, year, out["alcohol_frequency"])
    out["education_level"] = derive_education(df)
    out["household_income_q"] = derive_income_quartile(df)
    out["employment_status"] = derive_employment(df)
    out["marital_status"] = derive_marital(df, year)
    out["food_security_status"] = np.nan
    out["outcome"] = ((out["sbp_final"] >= 140) | (out["dbp_final"] >= 90)).astype(float)
    return out


def flow_count(rows: list[dict[str, Any]], name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    rows.append({"step": name, "n_remaining": int(len(after)), "n_excluded_step": int(len(before) - len(after))})


def apply_cohort_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cur = df.copy()
    rows.append({"step": "raw_hse_2018_2019_2021_2022_pooled", "n_remaining": int(len(cur)), "n_excluded_step": 0})
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
        "person_id", "source_dataset", "source_year", "age_band_raw",
        "sbp_final", "dbp_final", "known_htn_dx", "antihypertensive_use", "pregnancy", "outcome",
    ] + PREDICTORS_INTEGRATED
    return sub[keep_cols].copy(), qc


def missingness_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        features = PREDICTORS_INTEGRATED if model_set == "integrated" else PREDICTORS_SEX_SPECIFIC
        for col in features:
            rows.append({
                "model_set": model_set, "predictor": col, "n": int(len(df)),
                "missing_n": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()) if len(df) else math.nan,
                "all_missing": bool(df[col].isna().all()) if len(df) else False,
            })
    return pd.DataFrame(rows)


def missingness_by_year_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        features = PREDICTORS_INTEGRATED if model_set == "integrated" else PREDICTORS_SEX_SPECIFIC
        for year, sub in df.groupby("source_year"):
            for col in features:
                rows.append({
                    "model_set": model_set, "source_year": year, "predictor": col, "n": int(len(sub)),
                    "missing_n": int(sub[col].isna().sum()),
                    "missing_rate": float(sub[col].isna().mean()) if len(sub) else math.nan,
                })
    return pd.DataFrame(rows)


def category_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        for col, allowed in VALID_CATEGORIES.items():
            if col not in df.columns or (model_set != "integrated" and col == "sex"):
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            observed = sorted(vals.dropna().round().astype(int).unique().tolist())
            rows.append({
                "model_set": model_set, "predictor": col,
                "allowed_levels": json.dumps(allowed), "observed_levels": json.dumps(observed),
                "unseen_levels": json.dumps([x for x in observed if x not in allowed]),
            })
    return pd.DataFrame(rows)


def mapping_table() -> pd.DataFrame:
    rows = [
        ("age", "age16g5 (all years)", "Proxy", "Include age16g5>=3 (20+); map bands to midpoints, 80+ top-coded to 80."),
        ("sex", "sex", "Direct", "1 male, 2 female."),
        ("bmi", "BMIVal(2018/2019/2022); EstHtN+EstWtN(2021)", "Direct/Harmonization", "2021 lacks BMIVal (COVID-year protocol change); derived from measured height/weight, 76.9% missing in 2021."),
        ("waist_cm", "wstval (all years)", "Direct", "Valid mean waist circumference cm."),
        ("heart_rate", "mean(PULS2OM, PULS3OM) (all years)", "Harmonization", "Mean of 2nd/3rd Omron pulse readings."),
        ("sbp_final", "omsysval (all years)", "Direct", "Outcome only."),
        ("dbp_final", "omdiaval (all years)", "Direct", "Outcome only."),
        ("known_htn_dx", "bp1 (all years)", "Direct", "Doctor diagnosed high BP excluding pregnancy."),
        ("antihypertensive_use", "bpmedd2 (all years)", "Direct", "Prescribed BP medication."),
        ("pregnancy", "PregNTJ (all years)", "Direct", "Currently pregnant 16+."),
        ("diabetes_dx", "diabete2 (all years)", "Direct", "Doctor diagnosed diabetes excluding pregnant."),
        ("aerobic_activity", "MVPAmWk(2018,2022); mins10spta(2021, proxy); not available(2019)", "Harmonization/Proxy", "IPAQ MVPA>=150 min/wk; 2021 sport-only MVPA is narrower construct; 2019 all-NA."),
        ("self_rated_health", "GenHelf (all years)", "Direct", "1 very good to 5 very bad."),
        ("activity_limitation", "limitill (all years)", "Harmonization", "Limiting longstanding illness=1; else 0."),
        ("family_history_htn", "not available (all years)", "Not available", "Kept all-NA; no proxy substitution."),
        ("current_smoking", "cigst1(2018, DIFFERENT category order); cigsta3_19(2019/2021/2022)", "Harmonization", "2018 cigst1 order is Never/Ex-reg/Ex-occ/Current(1-4); mapped so current=raw4. Other years current=raw1."),
        ("alcohol_frequency", "dnoft(2018); dnoft_19(2019,2021); dnoft_22(2022)", "Harmonization", "Past-year drinking frequency collapsed to KNHANES 0-5 direction; assumed consistent HSE-standard coding across year-suffixed variable family."),
        ("drink_amount", "AUDTyDy(2022, direct); d7unitwgrp/d7unitwgrp_19(2018/2019/2021, proxy)", "Direct/Proxy", "2022 typical-day drinks 1-5 direct. Other years use heaviest-day-in-7-days units banded 1-7 capped at 5 - construct differs (heaviest day skews higher than typical day)."),
        ("binge_drinking", "AUD6Mre(2022, direct); alclimit07B/alclimit7B_19(2018/2019/2021, proxy)", "Harmonization/Proxy", "2022 AUDIT 6+-drinks frequency 0-5. Other years use an alcohol-unit risk-limit band (0-3) derived from heaviest-day units - different construct (risk classification, not binge frequency), passed through as valid subset of 0-5 scale."),
        ("education_level", "educ2 (all years)", "Harmonization", "Degree=4, below degree=3, none=1."),
        ("household_income_q", "eqv5 (all years)", "Harmonization", "Equivalised income quintile collapsed to quartile."),
        ("employment_status", "activb2 (all years)", "Harmonization", "Paid/self-employed/unpaid family business=1; else 2."),
        ("marital_status", "marstatD(2018/2019/2021); marstatd_22(2022)", "Harmonization", "Single=0, married/civil/cohabiting=1, separated=2, widowed=3, divorced=4."),
        ("food_security_status", "not available (all years)", "Not available", "Kept all-NA; no proxy substitution."),
    ]
    return pd.DataFrame(rows, columns=["knhanes_variable", "hse_source_by_year", "mapping_class", "rule"])


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    harmonized_parts = []
    for year, path in RAW_FILES.items():
        raw = pd.read_stata(path, convert_categoricals=False)
        harmonized_parts.append(harmonize_year(raw, year))
    harmonized = pd.concat(harmonized_parts, ignore_index=True)
    if harmonized["person_id"].duplicated().any():
        raise RuntimeError("Duplicate person_id after year-prefixing; check id construction")
    harmonized.to_csv(PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_harmonized_qc_{STAMP}.csv", index=False)

    cohort, flow = apply_cohort_rules(harmonized)
    flow_by_year_rows = []
    for year in RAW_FILES:
        n = int((cohort["source_year"] == year).sum())
        flow_by_year_rows.append({"source_year": year, "n_in_final_cohort_before_predictor_filter": n})
    pd.DataFrame(flow_by_year_rows).to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_cohort_by_year_{STAMP}.csv", index=False)

    inputs: dict[str, pd.DataFrame] = {}
    compat_rows = []
    for model_set in ["integrated", "male", "female"]:
        model_df, qc = participant_missing_filter(cohort, model_set)
        inputs[model_set] = model_df
        compat_rows.append(qc)
        model_df.to_csv(PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_{model_set}_input_missing_as_na_{STAMP}.csv", index=False)

    flow.to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_flow_{STAMP}.csv", index=False)
    pd.DataFrame(compat_rows).to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_input_compatibility_{STAMP}.csv", index=False)
    missingness_table(inputs).to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_predictor_missingness_{STAMP}.csv", index=False)
    missingness_by_year_table(inputs).to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_predictor_missingness_by_year_{STAMP}.csv", index=False)
    category_table(inputs).to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_category_level_qc_{STAMP}.csv", index=False)
    mapping_table().to_csv(REPORTS / f"hse_pooled_core_sdoh_hr_mapping_final_{STAMP}.csv", index=False)

    print("HSE 2018/2019/2021/2022 pooled harmonized inputs created")
    print(flow.to_string(index=False))
    print(pd.DataFrame(flow_by_year_rows).to_string(index=False))
    print(pd.DataFrame(compat_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
