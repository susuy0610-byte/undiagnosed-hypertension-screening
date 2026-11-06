#!/usr/bin/env python3
"""
Build conservative enhanced KNHANES datasets without overwriting the locked
primary inputs.

The enhanced variables are intentionally limited to non-invasive, AoU-mappable
items:

- height_cm
- weight_kg
- waist_to_height_ratio
- central_obesity
- dyslipidemia_dx_or_med, already present in the primary benchmark fields

No BP, hypertension diagnosis, antihypertensive medication, or derived outcome
variable is added as a predictor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_knhanes_primary_dataset as base


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DEV_OUT = PROCESSED_DIR / "knhanes_development_2014_2020_conservative_enhanced.csv"
TIME_OUT = PROCESSED_DIR / "knhanes_time_validation_2021_2022_conservative_enhanced.csv"

ENHANCED_PREDICTORS_INTEGRATED = [
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "heart_rate",
    "diabetes_dx",
    "dyslipidemia_dx_or_med",
    "central_obesity",
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

ENHANCED_OUTPUT_COLS = [
    "participant_id",
    "source_year",
    "year",
    "sex",
    "age",
    "undiagnosed_htn",
    "elevated_bp_14090",
    "bmi",
    "waist_cm",
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "central_obesity",
    "diabetes_dx",
    "dyslipidemia_dx_or_med",
    "aerobic_activity",
    "self_rated_health",
    "activity_limitation",
    "heart_rate",
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
    "cancer_history",
    "total_energy_kcal",
    "weight",
    "strata",
    "psu",
    "sbp_final",
    "dbp_final",
    "known_htn_dx_raw",
    "antihypertensive_use_raw",
    "pregnancy_raw",
    "predictor_missing_count",
    "predictor_missing_fraction",
    "enhanced_predictor_missing_count",
    "enhanced_predictor_missing_fraction",
]


def add_enhanced_variables(harmonized: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = harmonized.copy()
    out["height_cm"] = base.optional_numeric(raw, "HE_ht").where(base.optional_numeric(raw, "HE_ht") > 0, np.nan)
    out["weight_kg"] = base.optional_numeric(raw, "HE_wt").where(base.optional_numeric(raw, "HE_wt") > 0, np.nan)
    valid_ratio = out["height_cm"].notna() & (out["height_cm"] > 0) & out["waist_cm"].notna()
    out["waist_to_height_ratio"] = np.nan
    out.loc[valid_ratio, "waist_to_height_ratio"] = out.loc[valid_ratio, "waist_cm"] / out.loc[valid_ratio, "height_cm"]

    out["central_obesity"] = np.nan
    male = out["sex"] == 1
    female = out["sex"] == 2
    out.loc[male & out["waist_cm"].notna(), "central_obesity"] = (out.loc[male & out["waist_cm"].notna(), "waist_cm"] >= 90).astype(float)
    out.loc[female & out["waist_cm"].notna(), "central_obesity"] = (out.loc[female & out["waist_cm"].notna(), "waist_cm"] >= 85).astype(float)
    return out


def read_harmonized_enhanced(years: list[int]) -> pd.DataFrame:
    base.RAW_COLUMNS = list(dict.fromkeys(base.RAW_COLUMNS + ["HE_ht", "HE_wt"]))
    frames = []
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for year in years:
            raw, _ = base.read_sav_year(year, temp_dir)
            harmonized = base.harmonize(raw)
            frames.append(add_enhanced_variables(harmonized, raw))
    return pd.concat(frames, ignore_index=True)


def enhanced_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, part in [
        ("integrated", df),
        ("male", df[df["sex"] == 1]),
        ("female", df[df["sex"] == 2]),
    ]:
        predictors = [x for x in ENHANCED_PREDICTORS_INTEGRATED if not (group_name in {"male", "female"} and x == "sex")]
        for variable in predictors:
            rows.append(
                {
                    "model_set": group_name,
                    "variable": variable,
                    "n": len(part),
                    "missing_n": int(part[variable].isna().sum()),
                    "missing_percent": float(part[variable].isna().mean() * 100) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def enhanced_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["height_cm", "weight_kg", "waist_to_height_ratio"]:
        x = df[variable].dropna()
        rows.append(
            {
                "variable": variable,
                "n": int(x.size),
                "missing_percent": float(df[variable].isna().mean() * 100),
                "min": float(x.min()) if x.size else np.nan,
                "median": float(x.median()) if x.size else np.nan,
                "max": float(x.max()) if x.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_enhanced_missing_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["enhanced_predictor_missing_count"] = out[ENHANCED_PREDICTORS_INTEGRATED].isna().sum(axis=1)
    out["enhanced_predictor_missing_fraction"] = out["enhanced_predictor_missing_count"] / len(ENHANCED_PREDICTORS_INTEGRATED)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_summary(dev: pd.DataFrame, time_df: pd.DataFrame) -> None:
    lines = [
        "# KNHANES Conservative Enhanced Dataset",
        "",
        "This dataset preserves the locked primary cohort and outcome rules, then adds a small non-invasive enhanced predictor set.",
        "",
        "## Added Predictors",
        "",
        "| Predictor | Source/derivation | Rationale | AoU mapping expectation |",
        "|---|---|---|---|",
        "| `height_cm` | `HE_ht` | body size context beyond BMI | AoU Physical Measurements height |",
        "| `weight_kg` | `HE_wt` | body size context beyond BMI | AoU Physical Measurements weight |",
        "| `waist_to_height_ratio` | `waist_cm / height_cm` | central adiposity adjusted for height | derive from AoU waist and height |",
        "| `central_obesity` | waist >=90 cm men, >=85 cm women | Korean abdominal obesity criterion | derive from AoU waist and sex |",
        "| `dyslipidemia_dx_or_med` | `DI2_dg` or `DI2_2` | non-invasive comorbidity history/treatment | AoU survey/EHR dyslipidemia or high cholesterol mapping |",
        "",
        "## Dataset Size",
        "",
        f"- Development N: {len(dev):,}; events: {int(dev['undiagnosed_htn'].sum()):,}; event rate: {dev['undiagnosed_htn'].mean():.4f}",
        f"- Temporal N: {len(time_df):,}; events: {int(time_df['undiagnosed_htn'].sum()):,}; event rate: {time_df['undiagnosed_htn'].mean():.4f}",
        "",
        "## Safeguards",
        "",
        "- Existing primary inputs and frozen final models are not overwritten.",
        "- SBP/DBP, hypertension diagnosis, antihypertensive medication, pregnancy, and outcome flags remain excluded from predictors.",
        "- Nutrition energy and cancer history remain benchmark-only; they are not included in this conservative enhanced set.",
    ]
    (REPORT_DIR / "knhanes_conservative_enhanced_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dev_raw = read_harmonized_enhanced(base.DEVELOPMENT_YEARS)
    dev, _ = base.apply_eligibility(dev_raw)
    dev = add_enhanced_missing_fraction(dev)

    time_raw = read_harmonized_enhanced(base.TIME_VALIDATION_YEARS)
    time_df, _ = base.apply_eligibility(time_raw)
    time_df = add_enhanced_missing_fraction(time_df)

    write_csv(dev[ENHANCED_OUTPUT_COLS], DEV_OUT)
    write_csv(time_df[ENHANCED_OUTPUT_COLS], TIME_OUT)
    write_csv(enhanced_missingness(dev), REPORT_DIR / "knhanes_conservative_enhanced_missingness_2014_2020.csv")
    write_csv(enhanced_missingness(time_df), REPORT_DIR / "knhanes_conservative_enhanced_missingness_2021_2022.csv")
    write_csv(enhanced_distribution(dev), REPORT_DIR / "knhanes_conservative_enhanced_distribution_2014_2020.csv")
    write_csv(enhanced_distribution(time_df), REPORT_DIR / "knhanes_conservative_enhanced_distribution_2021_2022.csv")
    write_summary(dev, time_df)

    print(f"Wrote {DEV_OUT}")
    print(f"Wrote {TIME_OUT}")
    print(f"Development N={len(dev):,}, events={int(dev['undiagnosed_htn'].sum()):,}")
    print(f"Temporal N={len(time_df):,}, events={int(time_df['undiagnosed_htn'].sum()):,}")


if __name__ == "__main__":
    main()
