#!/usr/bin/env python3
"""
Build a defensible non-invasive enhanced v2 KNHANES dataset.

This keeps the locked primary cohort/outcome/exclusion rules and adds only
non-invasive predictors that are available across KNHANES 2014-2020 development
years and KNHANES 2021-2022 temporal-validation years.

No SBP/DBP, hypertension diagnosis, antihypertensive medication, pregnancy, or
derived outcome variable is added as a predictor.
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

DEV_OUT = PROCESSED_DIR / "knhanes_development_2014_2020_enhanced_v2.csv"
TIME_OUT = PROCESSED_DIR / "knhanes_time_validation_2021_2022_enhanced_v2.csv"

RAW_EXTRA_COLUMNS = [
    "HE_ht",
    "HE_wt",
    "BE3_31",
    "BE5_1",
    "BP1",
    "BO1",
    "BO1_1",
    "DN1_dg",
    "HE_IHDfh1",
    "HE_IHDfh2",
    "HE_IHDfh3",
    "HE_STRfh1",
    "HE_STRfh2",
    "HE_STRfh3",
    "LW_ms",
]

ENHANCED_V2_COMMON_PREDICTORS = [
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "age_squared",
    "bmi_squared",
    "age_bmi_interaction",
    "age_waist_interaction",
    "walking_days",
    "strength_days",
    "heart_rate",
    "diabetes_dx",
    "dyslipidemia_dx_or_med",
    "central_obesity",
    "bmi_category",
    "stress_level",
    "subjective_body_shape",
    "weight_change_1yr",
    "kidney_disease_dx",
    "family_history_ihd",
    "family_history_stroke",
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

ENHANCED_V2_FEMALE_ONLY_PREDICTORS = ["menstrual_status"]

ENHANCED_V2_OUTPUT_COLS = [
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
    "age_squared",
    "bmi_squared",
    "age_bmi_interaction",
    "age_waist_interaction",
    "walking_days",
    "strength_days",
    "central_obesity",
    "bmi_category",
    "stress_level",
    "subjective_body_shape",
    "weight_change_1yr",
    "kidney_disease_dx",
    "family_history_ihd",
    "family_history_stroke",
    "menstrual_status",
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
    "enhanced_v2_predictor_missing_count",
    "enhanced_v2_predictor_missing_fraction",
]


def positive_numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    x = base.optional_numeric(raw, column)
    return x.where(x > 0, np.nan)


def recode_walking_days(raw: pd.DataFrame) -> pd.Series:
    x = base.optional_numeric(raw, "BE3_31")
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    valid = x.isin(range(1, 9))
    out[valid] = x[valid] - 1
    return out


def recode_strength_days(raw: pd.DataFrame) -> pd.Series:
    x = base.optional_numeric(raw, "BE5_1")
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    for code, days in {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}.items():
        out[x == code] = days
    return out


def recode_stress_level(raw: pd.DataFrame) -> pd.Series:
    return base.map_code(base.optional_numeric(raw, "BP1"), [1, 2, 3, 4])


def recode_subjective_body_shape(raw: pd.DataFrame) -> pd.Series:
    return base.map_code(base.optional_numeric(raw, "BO1"), [1, 2, 3, 4, 5])


def recode_weight_change(raw: pd.DataFrame) -> pd.Series:
    return base.map_code(base.optional_numeric(raw, "BO1_1"), [1, 2, 3])


def derive_binary_dx_optional(raw: pd.DataFrame, column: str) -> pd.Series:
    x = base.optional_numeric(raw, column)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[x == 1] = 1
    out[x == 0] = 0
    return out


def derive_family_history_any(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    values = pd.DataFrame({col: base.optional_numeric(raw, col) for col in cols})
    any_yes = (values == 1).any(axis=1)
    all_no_or_structural = values.apply(lambda row: all(v in {0, 8} for v in row.dropna()), axis=1)
    no_missing = values.notna().all(axis=1)

    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[any_yes] = 1
    out[(~any_yes) & all_no_or_structural & no_missing] = 0
    return out


def derive_bmi_category(bmi: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=bmi.index, dtype="float")
    out[(bmi > 0) & (bmi < 18.5)] = 1
    out[(bmi >= 18.5) & (bmi < 23.0)] = 2
    out[(bmi >= 23.0) & (bmi < 25.0)] = 3
    out[(bmi >= 25.0) & (bmi < 30.0)] = 4
    out[bmi >= 30.0] = 5
    return out


def recode_menstrual_status(raw: pd.DataFrame, harmonized: pd.DataFrame) -> pd.Series:
    x = base.optional_numeric(raw, "LW_ms")
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    female = harmonized["sex"] == 2
    out[female & x.isin([2, 3, 4, 5, 6, 7])] = x[female & x.isin([2, 3, 4, 5, 6, 7])]
    return out


def add_enhanced_v2_variables(harmonized: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = harmonized.copy()
    out["height_cm"] = positive_numeric(raw, "HE_ht")
    out["weight_kg"] = positive_numeric(raw, "HE_wt")

    valid_ratio = out["height_cm"].notna() & (out["height_cm"] > 0) & out["waist_cm"].notna()
    out["waist_to_height_ratio"] = np.nan
    out.loc[valid_ratio, "waist_to_height_ratio"] = out.loc[valid_ratio, "waist_cm"] / out.loc[valid_ratio, "height_cm"]

    out["central_obesity"] = np.nan
    male = out["sex"] == 1
    female = out["sex"] == 2
    out.loc[male & out["waist_cm"].notna(), "central_obesity"] = (out.loc[male & out["waist_cm"].notna(), "waist_cm"] >= 90).astype(float)
    out.loc[female & out["waist_cm"].notna(), "central_obesity"] = (out.loc[female & out["waist_cm"].notna(), "waist_cm"] >= 85).astype(float)

    out["age_squared"] = out["age"] ** 2
    out["bmi_squared"] = out["bmi"] ** 2
    out["age_bmi_interaction"] = out["age"] * out["bmi"]
    out["age_waist_interaction"] = out["age"] * out["waist_cm"]
    out["bmi_category"] = derive_bmi_category(out["bmi"])

    out["walking_days"] = recode_walking_days(raw)
    out["strength_days"] = recode_strength_days(raw)
    out["stress_level"] = recode_stress_level(raw)
    out["subjective_body_shape"] = recode_subjective_body_shape(raw)
    out["weight_change_1yr"] = recode_weight_change(raw)
    out["kidney_disease_dx"] = derive_binary_dx_optional(raw, "DN1_dg")
    out["family_history_ihd"] = derive_family_history_any(raw, ["HE_IHDfh1", "HE_IHDfh2", "HE_IHDfh3"])
    out["family_history_stroke"] = derive_family_history_any(raw, ["HE_STRfh1", "HE_STRfh2", "HE_STRfh3"])
    out["menstrual_status"] = recode_menstrual_status(raw, out)
    return out


def read_harmonized_enhanced_v2(years: list[int]) -> pd.DataFrame:
    base.RAW_COLUMNS = list(dict.fromkeys(base.RAW_COLUMNS + RAW_EXTRA_COLUMNS))
    frames = []
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for year in years:
            raw, _ = base.read_sav_year(year, temp_dir)
            harmonized = base.harmonize(raw)
            frames.append(add_enhanced_v2_variables(harmonized, raw))
    return pd.concat(frames, ignore_index=True)


def predictors_for_model_set(model_set: str) -> list[str]:
    predictors = list(ENHANCED_V2_COMMON_PREDICTORS)
    if model_set in {"male", "female"}:
        predictors = [x for x in predictors if x != "sex"]
    if model_set == "female":
        predictors += ENHANCED_V2_FEMALE_ONLY_PREDICTORS
    return predictors


def enhanced_v2_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_set, part in [
        ("integrated", df),
        ("male", df[df["sex"] == 1]),
        ("female", df[df["sex"] == 2]),
    ]:
        for variable in predictors_for_model_set(model_set):
            rows.append(
                {
                    "model_set": model_set,
                    "variable": variable,
                    "n": len(part),
                    "missing_n": int(part[variable].isna().sum()),
                    "missing_percent": float(part[variable].isna().mean() * 100) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def enhanced_v2_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    continuous = [
        "height_cm",
        "weight_kg",
        "waist_to_height_ratio",
        "age_squared",
        "bmi_squared",
        "age_bmi_interaction",
        "age_waist_interaction",
        "walking_days",
        "strength_days",
    ]
    for variable in continuous:
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


def add_enhanced_v2_missing_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    predictors = ENHANCED_V2_COMMON_PREDICTORS
    out["enhanced_v2_predictor_missing_count"] = out[predictors].isna().sum(axis=1)
    out["enhanced_v2_predictor_missing_fraction"] = out["enhanced_v2_predictor_missing_count"] / len(predictors)
    return out


def apply_enhanced_v2_missing_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["enhanced_v2_predictor_missing_fraction"] <= 0.50].copy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_mapping_report(dev_before: pd.DataFrame, dev: pd.DataFrame, time_before: pd.DataFrame, time_df: pd.DataFrame) -> None:
    rows = [
        ("height_cm", "HE_ht", "measured height in cm", "Direct/AoU PM", "included"),
        ("weight_kg", "HE_wt", "measured weight in kg", "Direct/AoU PM", "included"),
        ("waist_to_height_ratio", "HE_wc / HE_ht", "central adiposity adjusted for height", "Derived", "included"),
        ("age_squared", "age^2", "nonlinear age effect", "Derived", "included"),
        ("bmi_squared", "bmi^2", "nonlinear BMI effect", "Derived", "included"),
        ("age_bmi_interaction", "age * bmi", "age-dependent adiposity risk", "Derived", "included"),
        ("age_waist_interaction", "age * waist_cm", "age-dependent central adiposity risk", "Derived", "included"),
        ("bmi_category", "HE_BMI", "Asian/Korean BMI risk categories", "Derived", "included"),
        ("walking_days", "BE3_31", "weekly walking-day frequency, 0-7 days", "Harmonization", "included"),
        ("strength_days", "BE5_1", "weekly muscle-strengthening frequency, 0-5+ days", "Harmonization", "included"),
        ("stress_level", "BP1", "usual perceived stress, 1 high to 4 low", "Harmonization", "included"),
        ("subjective_body_shape", "BO1", "self-perceived body shape, 1 very thin to 5 very obese", "Harmonization", "included"),
        ("weight_change_1yr", "BO1_1", "no change / weight loss / weight gain in prior year", "Proxy/Harmonization", "included"),
        ("kidney_disease_dx", "DN1_dg", "physician-diagnosed kidney disease/renal failure", "Direct/condition history", "included"),
        ("family_history_ihd", "HE_IHDfh1/2/3", "parent/sibling ischemic heart disease history", "Proxy/Harmonization", "included"),
        ("family_history_stroke", "HE_STRfh1/2/3", "parent/sibling stroke history", "Proxy/Harmonization", "included"),
        ("menstrual_status", "LW_ms", "female-only menstrual/menopause status", "Proxy/Harmonization", "female model only"),
        ("depressive_symptoms", "BP5", "2-week depressive feeling", "Potential AoU mental-health proxy", "excluded: structural missing in 2014/2016/2018/2020"),
        ("sleep_hours", "BP8 / Total_slp_* / BP16_*", "sleep duration", "Potential proxy", "excluded: inconsistent 2021 total sleep availability"),
        ("personal_stroke_or_ihd_dx", "DI3/DI4/DI5/DI6", "self-reported stroke/IHD diagnosis", "Direct condition history", "excluded: 2022 availability gap"),
        ("nutrition meal variables", "L_BR_FQ/L_OUT_FQ/N_*", "meal pattern/diet", "Proxy", "excluded: nutrition-module weight/external harmonization burden"),
    ]
    mapping = pd.DataFrame(rows, columns=["predictor", "source_or_derivation", "rationale", "aou_mapping_expectation", "decision"])
    write_csv(mapping, REPORT_DIR / "knhanes_enhanced_v2_variable_mapping_20260810.csv")

    lines = [
        "# KNHANES Enhanced V2 Non-Invasive Dataset",
        "",
        "Date: 2026-08-10",
        "",
        "This v2 dataset preserves the locked primary cohort, outcome, and leakage exclusions, then adds defensible non-invasive predictors.",
        "",
        "## Dataset Size",
        "",
        f"- Development after locked eligibility: N={len(dev_before):,}; events={int(dev_before['undiagnosed_htn'].sum()):,}",
        f"- Development after enhanced v2 >50% predictor-missing exclusion: N={len(dev):,}; events={int(dev['undiagnosed_htn'].sum()):,}; event rate={dev['undiagnosed_htn'].mean():.4f}",
        f"- Temporal after locked eligibility: N={len(time_before):,}; events={int(time_before['undiagnosed_htn'].sum()):,}",
        f"- Temporal after enhanced v2 >50% predictor-missing exclusion: N={len(time_df):,}; events={int(time_df['undiagnosed_htn'].sum()):,}; event rate={time_df['undiagnosed_htn'].mean():.4f}",
        "",
        "## Safeguards",
        "",
        "- Existing primary, final, temporal, and enhanced light-screen outputs are not overwritten.",
        "- SBP/DBP, hypertension diagnosis, antihypertensive medication, pregnancy, and outcome flags remain excluded from predictors.",
        "- Participants with >50% missingness across enhanced v2 common predictors are excluded before modeling.",
        "- Variables missing or unstable in 2021-2022 were excluded from this main v2 candidate set.",
        "- `menstrual_status` is included only for the female model.",
        "",
        "## Added Variable Decisions",
        "",
        "| Predictor | Source/derivation | Rationale | AoU expectation | Decision |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| `" + "` | `".join(row[:2]) + "` | " + " | ".join(row[2:]) + " |")
    (REPORT_DIR / "knhanes_enhanced_v2_summary_20260810.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dev_raw = read_harmonized_enhanced_v2(base.DEVELOPMENT_YEARS)
    dev, _ = base.apply_eligibility(dev_raw)
    dev = add_enhanced_v2_missing_fraction(dev)
    dev_before_missing_exclusion = dev.copy()
    dev = apply_enhanced_v2_missing_exclusion(dev)

    time_raw = read_harmonized_enhanced_v2(base.TIME_VALIDATION_YEARS)
    time_df, _ = base.apply_eligibility(time_raw)
    time_df = add_enhanced_v2_missing_fraction(time_df)
    time_before_missing_exclusion = time_df.copy()
    time_df = apply_enhanced_v2_missing_exclusion(time_df)

    write_csv(dev[ENHANCED_V2_OUTPUT_COLS], DEV_OUT)
    write_csv(time_df[ENHANCED_V2_OUTPUT_COLS], TIME_OUT)
    write_csv(enhanced_v2_missingness(dev), REPORT_DIR / "knhanes_enhanced_v2_missingness_2014_2020.csv")
    write_csv(enhanced_v2_missingness(time_df), REPORT_DIR / "knhanes_enhanced_v2_missingness_2021_2022.csv")
    write_csv(enhanced_v2_distribution(dev), REPORT_DIR / "knhanes_enhanced_v2_distribution_2014_2020.csv")
    write_csv(enhanced_v2_distribution(time_df), REPORT_DIR / "knhanes_enhanced_v2_distribution_2021_2022.csv")
    write_mapping_report(dev_before_missing_exclusion, dev, time_before_missing_exclusion, time_df)

    print(f"Wrote {DEV_OUT}")
    print(f"Wrote {TIME_OUT}")
    print(f"Development N={len(dev):,}, events={int(dev['undiagnosed_htn'].sum()):,}")
    print(f"Temporal N={len(time_df):,}, events={int(time_df['undiagnosed_htn'].sum()):,}")


if __name__ == "__main__":
    main()
