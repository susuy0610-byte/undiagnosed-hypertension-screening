#!/usr/bin/env python3
"""
Build KNHANES enhanced v3 non-invasive predictors.

Enhanced v3 starts from enhanced v2 and adds defensible non-invasive blocks:
sedentary time, detailed physical activity, smoking intensity/history,
granular SDOH, and female-only reproductive history.

No blood pressure measurement, hypertension diagnosis, antihypertensive
medication, pregnancy, or outcome-derived field is used as a predictor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_knhanes_primary_dataset as base
import build_knhanes_enhanced_v2_dataset as v2


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DEV_OUT = PROCESSED_DIR / "knhanes_development_2014_2020_enhanced_v3.csv"
TIME_OUT = PROCESSED_DIR / "knhanes_time_validation_2021_2022_enhanced_v3.csv"

RAW_EXTRA_COLUMNS = list(
    dict.fromkeys(
        v2.RAW_EXTRA_COLUMNS
        + [
            "BE8_1",
            "BE8_2",
            "BE3_32",
            "BE3_33",
            "BE3_71",
            "BE3_72",
            "BE3_73",
            "BE3_74",
            "BE3_75",
            "BE3_76",
            "BE3_77",
            "BE3_78",
            "BE3_81",
            "BE3_82",
            "BE3_83",
            "BE3_84",
            "BE3_85",
            "BE3_86",
            "BE3_87",
            "BE3_88",
            "BE3_91",
            "BE3_92",
            "BE3_93",
            "BE3_94",
            "BS1_1",
            "BS3_2",
            "BS3_3",
            "BS6_2",
            "BS6_3",
            "BS6_4",
            "BS8_2",
            "BS9_2",
            "BS12_1",
            "BS12_2",
            "cfam",
            "house",
            "live_t",
            "apt_t",
            "allownc",
            "tins",
            "npins",
            "incm",
            "incm5",
            "ho_incm5",
            "ainc",
            "occp",
            "EC_pedu_1",
            "EC_pedu_2",
            "town_t",
            "LW_ms_a",
            "LW_mp_a",
            "LW_mt",
            "LW_mt_a1",
            "LW_mt_a2",
            "LW_oc",
            "LW_br",
            "LW_br_ch",
            "LW_br_dur",
        ]
    )
)

V3_COMMON_CONTINUOUS = [
    "sedentary_minutes_day",
    "walking_minutes_week",
    "work_vigorous_minutes_week",
    "leisure_vigorous_minutes_week",
    "work_moderate_minutes_week",
    "leisure_moderate_minutes_week",
    "transport_minutes_week",
    "total_activity_minutes_week",
    "current_cigarettes_day",
    "current_smoking_days_month",
    "former_smoking_months",
    "former_cigarettes_day",
    "quit_months",
    "household_size",
    "monthly_household_income",
]

V3_COMMON_CATEGORICAL = [
    "lifetime_smoking_level",
    "secondhand_smoke_work",
    "secondhand_smoke_home",
    "ecig_ever",
    "ecig_current",
    "personal_income_q",
    "personal_income_quintile",
    "household_income_quintile",
    "house_ownership",
    "housing_type",
    "apartment_type",
    "basic_livelihood_support",
    "health_insurance_type",
    "private_insurance",
    "occupation_group",
    "father_education",
    "mother_education",
    "urban_rural",
]

V3_FEMALE_CONTINUOUS = [
    "menarche_age",
    "menopause_age",
    "first_birth_age",
    "last_birth_age",
    "breastfeeding_months",
    "breastfed_children_count",
]

V3_FEMALE_CATEGORICAL = [
    "parity_experience",
    "oral_contraceptive_use",
    "breastfeeding_experience",
]

ENHANCED_V3_COMMON_PREDICTORS = (
    v2.ENHANCED_V2_COMMON_PREDICTORS + V3_COMMON_CONTINUOUS + V3_COMMON_CATEGORICAL
)
ENHANCED_V3_FEMALE_ONLY_PREDICTORS = (
    v2.ENHANCED_V2_FEMALE_ONLY_PREDICTORS + V3_FEMALE_CONTINUOUS + V3_FEMALE_CATEGORICAL
)

ENHANCED_V3_OUTPUT_COLS = [
    x
    for x in v2.ENHANCED_V2_OUTPUT_COLS
    if x not in {"enhanced_v2_predictor_missing_count", "enhanced_v2_predictor_missing_fraction"}
] + V3_COMMON_CONTINUOUS + V3_COMMON_CATEGORICAL + V3_FEMALE_CONTINUOUS + V3_FEMALE_CATEGORICAL + [
    "enhanced_v2_predictor_missing_count",
    "enhanced_v2_predictor_missing_fraction",
    "enhanced_v3_predictor_missing_count",
    "enhanced_v3_predictor_missing_fraction",
]


def raw_num(raw: pd.DataFrame, column: str) -> pd.Series:
    return base.optional_numeric(raw, column)


def valid_range(raw: pd.DataFrame, column: str, lo: float, hi: float) -> pd.Series:
    x = raw_num(raw, column)
    missing_codes = {88, 99, 888, 999, 8888, 9999, 88888, 99999, 999999}
    x = x.where(~x.isin(missing_codes), np.nan)
    return x.where((x >= lo) & (x <= hi), np.nan)


def map_binary(raw: pd.DataFrame, column: str, yes: set[int], no: set[int]) -> pd.Series:
    x = raw_num(raw, column)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[x.isin(yes)] = 1
    out[x.isin(no)] = 0
    return out


def map_category(raw: pd.DataFrame, column: str, valid: list[int]) -> pd.Series:
    return base.map_code(raw_num(raw, column), valid)


def duration_minutes(raw: pd.DataFrame, hour_col: str, minute_col: str) -> pd.Series:
    hours = valid_range(raw, hour_col, 0, 24)
    minutes = valid_range(raw, minute_col, 0, 59)
    out = hours * 60 + minutes
    return out.where(out <= 24 * 60, np.nan)


def weekly_minutes_with_yes_no(raw: pd.DataFrame, yes_col: str, days_col: str, hour_col: str, minute_col: str) -> pd.Series:
    yes = raw_num(raw, yes_col)
    days = valid_range(raw, days_col, 1, 7)
    daily = duration_minutes(raw, hour_col, minute_col)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[yes == 2] = 0
    ok = (yes == 1) & days.notna() & daily.notna()
    out[ok] = days[ok] * daily[ok]
    return out.where(out <= 7 * 24 * 60, np.nan)


def walking_minutes_week(raw: pd.DataFrame) -> pd.Series:
    days = v2.recode_walking_days(raw)
    daily = duration_minutes(raw, "BE3_32", "BE3_33")
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[days == 0] = 0
    ok = (days > 0) & daily.notna()
    out[ok] = days[ok] * daily[ok]
    return out.where(out <= 7 * 24 * 60, np.nan)


def sedentary_minutes_day(raw: pd.DataFrame) -> pd.Series:
    out = duration_minutes(raw, "BE8_1", "BE8_2")
    return out.where(out <= 24 * 60, np.nan)


def current_cigarettes_day(raw: pd.DataFrame) -> pd.Series:
    smoking = raw_num(raw, "BS3_1")
    cigs = valid_range(raw, "BS3_2", 0, 100)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[smoking.isin([3, 8])] = 0
    out[smoking.isin([1, 2]) & cigs.notna()] = cigs[smoking.isin([1, 2]) & cigs.notna()]
    return out


def current_smoking_days_month(raw: pd.DataFrame) -> pd.Series:
    smoking = raw_num(raw, "BS3_1")
    days = valid_range(raw, "BS3_3", 1, 31)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[smoking == 1] = 30
    out[smoking == 2] = days[smoking == 2]
    out[smoking.isin([3, 8])] = 0
    return out


def noncurrent_exposure(raw: pd.DataFrame, column: str, hi: float) -> pd.Series:
    smoking = raw_num(raw, "BS3_1")
    value = valid_range(raw, column, 0, hi)
    out = pd.Series(np.nan, index=raw.index, dtype="float")
    out[smoking.isin([1, 8])] = 0
    out[value.notna()] = value[value.notna()]
    return out


def female_age(raw: pd.DataFrame, harmonized: pd.DataFrame, column: str, lo: float, hi: float) -> pd.Series:
    x = valid_range(raw, column, lo, hi)
    return x.where(harmonized["sex"] == 2, np.nan)


def female_binary(raw: pd.DataFrame, harmonized: pd.DataFrame, column: str, yes: set[int], no: set[int]) -> pd.Series:
    out = map_binary(raw, column, yes, no)
    return out.where(harmonized["sex"] == 2, np.nan)


def add_enhanced_v3_variables(harmonized: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = v2.add_enhanced_v2_variables(harmonized, raw)

    out["sedentary_minutes_day"] = sedentary_minutes_day(raw)
    out["walking_minutes_week"] = walking_minutes_week(raw)
    out["work_vigorous_minutes_week"] = weekly_minutes_with_yes_no(raw, "BE3_71", "BE3_72", "BE3_73", "BE3_74")
    out["leisure_vigorous_minutes_week"] = weekly_minutes_with_yes_no(raw, "BE3_75", "BE3_76", "BE3_77", "BE3_78")
    out["work_moderate_minutes_week"] = weekly_minutes_with_yes_no(raw, "BE3_81", "BE3_82", "BE3_83", "BE3_84")
    out["leisure_moderate_minutes_week"] = weekly_minutes_with_yes_no(raw, "BE3_85", "BE3_86", "BE3_87", "BE3_88")
    out["transport_minutes_week"] = weekly_minutes_with_yes_no(raw, "BE3_91", "BE3_92", "BE3_93", "BE3_94")
    out["total_activity_minutes_week"] = (
        out["walking_minutes_week"].fillna(0)
        + out["transport_minutes_week"].fillna(0)
        + out["work_moderate_minutes_week"].fillna(0)
        + out["leisure_moderate_minutes_week"].fillna(0)
        + 2 * out["work_vigorous_minutes_week"].fillna(0)
        + 2 * out["leisure_vigorous_minutes_week"].fillna(0)
    )
    all_activity_missing = out[
        [
            "walking_minutes_week",
            "transport_minutes_week",
            "work_moderate_minutes_week",
            "leisure_moderate_minutes_week",
            "work_vigorous_minutes_week",
            "leisure_vigorous_minutes_week",
        ]
    ].isna().all(axis=1)
    out.loc[all_activity_missing, "total_activity_minutes_week"] = np.nan

    out["lifetime_smoking_level"] = map_category(raw, "BS1_1", [1, 2, 3])
    out["current_cigarettes_day"] = current_cigarettes_day(raw)
    out["current_smoking_days_month"] = current_smoking_days_month(raw)
    out["former_smoking_months"] = noncurrent_exposure(raw, "BS6_2", 900)
    out["former_cigarettes_day"] = noncurrent_exposure(raw, "BS6_3", 100)
    out["quit_months"] = noncurrent_exposure(raw, "BS6_4", 900)
    out["secondhand_smoke_work"] = map_category(raw, "BS8_2", [1, 2, 8])
    out["secondhand_smoke_home"] = map_category(raw, "BS9_2", [1, 2, 3, 8])
    out["ecig_ever"] = map_binary(raw, "BS12_1", {1}, {2})
    out["ecig_current"] = map_binary(raw, "BS12_2", {1}, {2, 8})

    out["household_size"] = valid_range(raw, "cfam", 1, 20)
    out["monthly_household_income"] = valid_range(raw, "ainc", 0, 1500)
    out["personal_income_q"] = map_category(raw, "incm", [1, 2, 3, 4])
    out["personal_income_quintile"] = map_category(raw, "incm5", [1, 2, 3, 4, 5])
    out["household_income_quintile"] = map_category(raw, "ho_incm5", [1, 2, 3, 4, 5])
    out["house_ownership"] = map_category(raw, "house", [1, 2, 3])
    out["housing_type"] = map_category(raw, "live_t", [1, 2, 3, 4, 5])
    out["apartment_type"] = map_category(raw, "apt_t", [1, 2])
    out["basic_livelihood_support"] = map_category(raw, "allownc", [10, 20])
    out["health_insurance_type"] = map_category(raw, "tins", [10, 20, 30])
    out["private_insurance"] = map_category(raw, "npins", [1, 2])
    out["occupation_group"] = map_category(raw, "occp", [1, 2, 3, 4, 5, 6, 7])
    out["father_education"] = map_category(raw, "EC_pedu_1", [1, 2, 3, 4, 5, 6, 7, 8])
    out["mother_education"] = map_category(raw, "EC_pedu_2", [1, 2, 3, 4, 5, 6, 7, 8])
    out["urban_rural"] = map_category(raw, "town_t", [1, 2])

    out["menarche_age"] = female_age(raw, out, "LW_ms_a", 8, 26)
    out["menopause_age"] = female_age(raw, out, "LW_mp_a", 17, 70)
    out["parity_experience"] = female_binary(raw, out, "LW_mt", {1}, {2, 8})
    out["first_birth_age"] = female_age(raw, out, "LW_mt_a1", 13, 55)
    out["last_birth_age"] = female_age(raw, out, "LW_mt_a2", 13, 60)
    out["oral_contraceptive_use"] = female_binary(raw, out, "LW_oc", {1}, {2})
    out["breastfeeding_experience"] = female_binary(raw, out, "LW_br", {1}, {2, 8})
    out["breastfed_children_count"] = valid_range(raw, "LW_br_ch", 1, 20).where(out["sex"] == 2, np.nan)
    out.loc[(out["sex"] == 2) & raw_num(raw, "LW_br_ch").isin([88]), "breastfed_children_count"] = 0
    out["breastfeeding_months"] = valid_range(raw, "LW_br_dur", 0, 600).where(out["sex"] == 2, np.nan)
    out.loc[(out["sex"] == 2) & raw_num(raw, "LW_br").isin([2, 8]), "breastfeeding_months"] = 0

    return out


def read_harmonized_enhanced_v3(years: list[int]) -> pd.DataFrame:
    base.RAW_COLUMNS = list(dict.fromkeys(base.RAW_COLUMNS + RAW_EXTRA_COLUMNS))
    frames = []
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for year in years:
            raw, _ = base.read_sav_year(year, temp_dir)
            harmonized = base.harmonize(raw)
            frames.append(add_enhanced_v3_variables(harmonized, raw))
    return pd.concat(frames, ignore_index=True)


def predictors_for_model_set(model_set: str) -> list[str]:
    predictors = list(ENHANCED_V3_COMMON_PREDICTORS)
    if model_set in {"male", "female"}:
        predictors = [x for x in predictors if x != "sex"]
    if model_set == "female":
        predictors += ENHANCED_V3_FEMALE_ONLY_PREDICTORS
    return predictors


def add_enhanced_v3_missing_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    predictors = ENHANCED_V3_COMMON_PREDICTORS
    out["enhanced_v3_predictor_missing_count"] = out[predictors].isna().sum(axis=1)
    out["enhanced_v3_predictor_missing_fraction"] = out["enhanced_v3_predictor_missing_count"] / len(predictors)
    return out


def apply_enhanced_v3_missing_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["enhanced_v3_predictor_missing_fraction"] <= 0.50].copy()


def missingness(df: pd.DataFrame) -> pd.DataFrame:
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


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in V3_COMMON_CONTINUOUS + V3_FEMALE_CONTINUOUS:
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(dev_before: pd.DataFrame, dev: pd.DataFrame, time_before: pd.DataFrame, time_df: pd.DataFrame) -> None:
    rows = [
        ("sedentary_minutes_day", "BE8_1/BE8_2", "sedentary time", "Direct/harmonization", "included"),
        ("walking_minutes_week", "BE3_31/32/33", "walking volume", "Harmonization", "included"),
        ("work/leisure/transport activity minutes", "BE3_71-BE3_94", "domain-specific activity volume", "Harmonization", "included"),
        ("smoking intensity/history", "BS1_1/BS3_2/BS3_3/BS6_2/BS6_3/BS6_4", "dose and former exposure", "Harmonization", "included"),
        ("secondhand/e-cigarette exposure", "BS8_2/BS9_2/BS12_1/BS12_2", "smoking exposure granularity", "Proxy/harmonization", "included"),
        ("granular SDOH", "incm/incm5/ho_incm5/ainc/cfam/house/live_t/apt_t/allownc/tins/npins/occp/EC_pedu_1/2/town_t", "income, housing, insurance, occupation, childhood SES, urbanicity", "Harmonization/proxy", "included"),
        ("female reproductive history", "LW_ms_a/LW_mp_a/LW_mt/LW_mt_a1/2/LW_oc/LW_br*", "female-specific non-invasive risk context", "Proxy/harmonization", "female model only"),
        ("region", "region", "17 administrative regions", "AoU mapping weak, possible geographic overfit", "excluded"),
        ("activity limitation reason: hypertension", "LQ4_08", "explicit hypertension-related reason", "Leakage risk", "excluded"),
    ]
    mapping = pd.DataFrame(rows, columns=["block_or_predictor", "source_or_derivation", "rationale", "aou_mapping_expectation", "decision"])
    write_csv(mapping, REPORT_DIR / "knhanes_enhanced_v3_variable_mapping_20260811.csv")

    lines = [
        "# KNHANES Enhanced V3 Non-Invasive Dataset",
        "",
        "Date: 2026-08-11",
        "",
        "Enhanced v3 preserves the locked primary cohort/outcome/leakage exclusions and adds non-invasive lifestyle intensity, smoking intensity/history, granular SDOH, and female-only reproductive history.",
        "",
        "## Dataset Size",
        "",
        f"- Development after locked eligibility: N={len(dev_before):,}; events={int(dev_before['undiagnosed_htn'].sum()):,}",
        f"- Development after enhanced v3 >50% common-predictor-missing exclusion: N={len(dev):,}; events={int(dev['undiagnosed_htn'].sum()):,}; event rate={dev['undiagnosed_htn'].mean():.4f}",
        f"- Temporal after locked eligibility: N={len(time_before):,}; events={int(time_before['undiagnosed_htn'].sum()):,}",
        f"- Temporal after enhanced v3 >50% common-predictor-missing exclusion: N={len(time_df):,}; events={int(time_df['undiagnosed_htn'].sum()):,}; event rate={time_df['undiagnosed_htn'].mean():.4f}",
        "",
        "## Safeguards",
        "",
        "- No SBP/DBP, hypertension diagnosis, antihypertensive medication, pregnancy, or outcome-derived field is used as a predictor.",
        "- Region and hypertension-specific activity-limitation reason are excluded.",
        "- Female reproductive variables are included only for the female model.",
        "",
        "## Added Blocks",
        "",
        "| Block/predictor | Source/derivation | Rationale | AoU expectation | Decision |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| `" + "` | `".join(row[:2]) + "` | " + " | ".join(row[2:]) + " |")
    (REPORT_DIR / "knhanes_enhanced_v3_summary_20260811.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dev_raw = read_harmonized_enhanced_v3(base.DEVELOPMENT_YEARS)
    dev, _ = base.apply_eligibility(dev_raw)
    dev = v2.add_enhanced_v2_missing_fraction(dev)
    dev = add_enhanced_v3_missing_fraction(dev)
    dev_before_missing_exclusion = dev.copy()
    dev = apply_enhanced_v3_missing_exclusion(dev)

    time_raw = read_harmonized_enhanced_v3(base.TIME_VALIDATION_YEARS)
    time_df, _ = base.apply_eligibility(time_raw)
    time_df = v2.add_enhanced_v2_missing_fraction(time_df)
    time_df = add_enhanced_v3_missing_fraction(time_df)
    time_before_missing_exclusion = time_df.copy()
    time_df = apply_enhanced_v3_missing_exclusion(time_df)

    write_csv(dev[ENHANCED_V3_OUTPUT_COLS], DEV_OUT)
    write_csv(time_df[ENHANCED_V3_OUTPUT_COLS], TIME_OUT)
    write_csv(missingness(dev), REPORT_DIR / "knhanes_enhanced_v3_missingness_2014_2020.csv")
    write_csv(missingness(time_df), REPORT_DIR / "knhanes_enhanced_v3_missingness_2021_2022.csv")
    write_csv(distribution(dev), REPORT_DIR / "knhanes_enhanced_v3_distribution_2014_2020.csv")
    write_csv(distribution(time_df), REPORT_DIR / "knhanes_enhanced_v3_distribution_2021_2022.csv")
    write_report(dev_before_missing_exclusion, dev, time_before_missing_exclusion, time_df)

    print(f"Wrote {DEV_OUT}")
    print(f"Wrote {TIME_OUT}")
    print(f"Development N={len(dev):,}, events={int(dev['undiagnosed_htn'].sum()):,}")
    print(f"Temporal N={len(time_df):,}, events={int(time_df['undiagnosed_htn'].sum()):,}")


if __name__ == "__main__":
    main()
