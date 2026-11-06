#!/usr/bin/env python3
"""
Build the KNHANES 2014-2020 primary development dataset for the
undiagnosed hypertension screening model.

This script only creates the locked analysis input and descriptive/QC
reports. It does not fit prediction models.
"""

from __future__ import annotations

import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyreadstat


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "knhanes"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DEVELOPMENT_YEARS = list(range(2014, 2021))
TIME_VALIDATION_YEARS = [2021, 2022]

RAW_COLUMNS = [
    "id",
    "ID",
    "year",
    "sex",
    "age",
    "HE_sbp",
    "HE_dbp",
    "HE_sbp1",
    "HE_sbp2",
    "HE_sbp3",
    "HE_dbp1",
    "HE_dbp2",
    "HE_dbp3",
    "HE_PLS",
    "HE_PLS_15",
    "HE_PLS_30",
    "HE_mPLS",
    "DI1_dg",
    "DI1_2",
    "HE_HPdr",
    "HE_prg",
    "HE_BMI",
    "HE_wc",
    "DE1_dg",
    "DI2_dg",
    "DI2_2",
    "DC1_dg",
    "DC2_dg",
    "DC3_dg",
    "DC4_dg",
    "DC5_dg",
    "DC6_dg",
    "DC7_dg",
    "DC11_dg",
    "HE_HPfh1",
    "HE_HPfh2",
    "HE_HPfh3",
    "D_1_1",
    "LQ4_00",
    "pa_aerobic",
    "LF_SAFE",
    "LF_SECUR_G",
    "BS3_1",
    "BD1_11",
    "BD2_1",
    "BD2_31",
    "BD2_32",
    "edu",
    "educ",
    "ho_incm",
    "EC1_1",
    "marri_1",
    "marri_2",
    "N_EN",
    "wt_itvex",
    "kstrata",
    "psu",
]

PREDICTORS_INTEGRATED = [
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "diabetes_dx",
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
]

PREDICTORS_SEX_SPECIFIC = [x for x in PREDICTORS_INTEGRATED if x != "sex"]

HIM_NONINVASIVE_INTEGRATED = [
    "age",
    "sex",
    "household_income_q",
    "education_level",
    "diabetes_dx",
    "dyslipidemia_dx_or_med",
    "cancer_history",
    "current_smoking",
    "alcohol_frequency",
    "drink_amount",
    "binge_drinking",
    "total_energy_kcal",
    "bmi",
    "waist_cm",
]

HIM_NONINVASIVE_MALE = [x for x in HIM_NONINVASIVE_INTEGRATED if x != "sex"]
HIM_NONINVASIVE_FEMALE = [x for x in HIM_NONINVASIVE_MALE if x != "current_smoking"]
ALL_CANDIDATE_PREDICTORS = sorted(set(PREDICTORS_INTEGRATED + HIM_NONINVASIVE_INTEGRATED))

VARIABLE_SET_PREDICTORS = {
    "core": [
        "age",
        "sex",
        "bmi",
        "waist_cm",
        "diabetes_dx",
        "aerobic_activity",
        "self_rated_health",
        "activity_limitation",
        "family_history_htn",
        "current_smoking",
        "alcohol_frequency",
        "drink_amount",
        "binge_drinking",
    ],
    "core_sdoh": [
        "age",
        "sex",
        "bmi",
        "waist_cm",
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
    ],
    "core_sdoh_hr": PREDICTORS_INTEGRATED,
    "him_noninvasive": HIM_NONINVASIVE_INTEGRATED,
}

LEAKAGE_OR_OUTCOME_ONLY = [
    "HE_sbp",
    "HE_dbp",
    "HE_sbp1",
    "HE_sbp2",
    "HE_sbp3",
    "HE_dbp1",
    "HE_dbp2",
    "HE_dbp3",
    "DI1_dg",
    "DI1_2",
    "HE_HPdr",
    "undiagnosed_htn",
    "elevated_bp_14090",
]


MAPPING_ROWS = [
    ("participant_id", "id/ID", "개인 ID", "2014-2015 id; 2016-2020 ID", "identifier", "No"),
    ("year", "year", "조사연도", "same", "metadata", "No"),
    ("sex", "sex", "성별", "1 male, 2 female", "predictor", "Yes, integrated model only"),
    ("age", "age", "만나이", "1-79; 80 means 80+ top-coded", "predictor", "Yes"),
    ("sbp_final", "HE_sbp", "최종 수축기혈압(2,3차 평균)", "mmHg", "outcome only", "No"),
    ("dbp_final", "HE_dbp", "최종 이완기혈압(2,3차 평균)", "mmHg", "outcome only", "No"),
    ("known_htn_dx", "DI1_dg", "고혈압 의사진단 여부", "0 no, 1 yes, 8 not applicable, 9 unknown/no response", "exclusion/outcome", "No"),
    ("antihypertensive_use", "DI1_2", "혈압조절제 복용", "1-4 use, 5 non-use, 8 not applicable, 9 unknown/no response", "exclusion/outcome", "No"),
    ("pregnancy", "HE_prg", "임신여부", "0 no, 1 yes, 8 not applicable male", "exclusion", "No"),
    ("bmi", "HE_BMI", "체질량지수", "kg/m2", "predictor", "Yes"),
    ("waist_cm", "HE_wc", "허리둘레", "cm", "predictor", "Yes"),
    ("diabetes_dx", "DE1_dg", "당뇨병 의사진단 여부", "0 no, 1 yes, else missing", "predictor", "Yes, Core"),
    ("aerobic_activity", "pa_aerobic", "유산소 신체활동 실천율", "0 no, 1 yes", "predictor", "Yes, Core"),
    ("self_rated_health", "D_1_1", "주관적 건강상태/건강인지", "1 very good to 5 very poor", "predictor", "Yes, Core"),
    ("activity_limitation", "LQ4_00", "활동제한 여부", "1 yes -> 1; 2 no -> 0", "predictor", "Yes, Core"),
    (
        "heart_rate",
        "HE_PLS/HE_PLS_15/HE_PLS_30",
        "맥박수",
        "2014-2018 HE_PLS 15초 x4; 2019 HE_PLS_15 15초 x4; 2020+ HE_PLS_30 30초 x2; HE_mPLS 60초 fallback",
        "predictor",
        "Yes, Core+SDOH+HR",
    ),
    ("family_history_htn", "HE_HPfh1/2/3", "부/모/형제자매 고혈압 가족력", "derived any yes", "predictor", "Yes"),
    ("current_smoking", "BS3_1", "현재 일반담배/궐련 흡연", "1/2 current, 3/8 non-current, 9 missing", "predictor", "Yes"),
    ("alcohol_frequency", "BD1_11", "최근 1년 음주빈도", "8 recoded to none/never; 9 missing", "predictor", "Yes"),
    ("drink_amount", "BD2_1", "1회 음주량", "8 recoded to none/not applicable; 9 missing", "predictor", "Yes"),
    ("binge_drinking", "BD2_31/BD2_32", "폭음빈도", "male BD2_31, female BD2_32; 8 recoded none/not applicable", "predictor", "Yes"),
    ("education_level", "edu", "교육수준 재분류", "1 elementary or less, 2 middle, 3 high, 4 college+", "predictor", "Yes"),
    ("household_income_q", "ho_incm", "가구소득 4분위", "1 low to 4 high", "predictor", "Yes"),
    ("employment_status", "EC1_1", "경제활동 상태", "1 employed, 2 not employed, 8/9 missing for adult model", "predictor", "Yes"),
    ("marital_status", "marri_1/marri_2", "혼인상태", "derived never/married-separated/widowed/divorced", "predictor", "Yes"),
    ("food_security_status", "LF_SAFE/LF_SECUR_G", "식생활형편/식품안정성", "1-4 common ordered category", "predictor", "Yes, Core+SDOH"),
    ("dyslipidemia_dx_or_med", "DI2_dg/DI2_2", "이상지질혈증 의사진단 또는 약복용", "derived binary", "benchmark", "Yes, Hypertension Index non-invasive benchmark"),
    ("cancer_history", "DC1_dg/DC2_dg/DC3_dg/DC4_dg/DC5_dg/DC6_dg/DC7_dg/DC11_dg", "암 의사진단 병력", "any cancer diagnosis yes", "benchmark", "Yes, Hypertension Index non-invasive benchmark"),
    ("total_energy_kcal", "N_EN", "총 에너지 섭취량", "kcal/day, nutrition-derived", "benchmark", "Yes, Hypertension Index non-invasive benchmark"),
    ("weight", "wt_itvex", "건강설문·검진조사 가중치", "survey/exam weight", "survey design", "No"),
    ("strata", "kstrata", "분산추정 층", "survey design strata", "survey design", "No"),
    ("psu", "psu", "조사구번호", "primary sampling unit", "survey design", "No"),
]


CAT_LABELS = {
    "sex": {1: "male", 2: "female"},
    "diabetes_dx": {0: "no", 1: "yes"},
    "aerobic_activity": {0: "no", 1: "yes"},
    "self_rated_health": {1: "very_good", 2: "good", 3: "fair", 4: "poor", 5: "very_poor"},
    "activity_limitation": {0: "no", 1: "yes"},
    "family_history_htn": {0: "no", 1: "yes"},
    "current_smoking": {0: "non-current", 1: "current"},
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
    "education_level": {1: "elementary_or_less", 2: "middle", 3: "high", 4: "college_plus"},
    "household_income_q": {1: "low", 2: "mid_low", 3: "mid_high", 4: "high"},
    "employment_status": {1: "employed", 2: "not_employed"},
    "food_security_status": {1: "secure", 2: "mild_insecure", 3: "moderate_insecure", 4: "severe_insecure"},
    "dyslipidemia_dx_or_med": {0: "no", 1: "yes"},
    "cancer_history": {0: "no", 1: "yes"},
    "marital_status": {
        0: "never_married",
        1: "married_living_together",
        2: "married_separated",
        3: "widowed",
        4: "divorced",
    },
    "undiagnosed_htn": {0: "no", 1: "yes"},
}


@dataclass(frozen=True)
class YearFile:
    year: int
    path: Path
    member: str | None


def year_file(year: int) -> YearFile:
    yy = str(year)[2:]
    if year <= 2021:
        path = RAW_DIR / f"HN{yy}_ALL(SPSS).zip"
        return YearFile(year=year, path=path, member=None)
    path = RAW_DIR / f"HN{yy}_all.sav"
    if not path.exists():
        path = RAW_DIR / f"HN{yy}_ALL.sav"
    return YearFile(year=year, path=path, member=None)


def read_sav_year(year: int, temp_dir: Path) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    yf = year_file(year)
    if not yf.path.exists():
        raise FileNotFoundError(f"Missing raw file for {year}: {yf.path}")

    sav_path = yf.path
    if yf.path.suffix.lower() == ".zip":
        with zipfile.ZipFile(yf.path) as zf:
            sav_members = [x for x in zf.namelist() if x.lower().endswith(".sav")]
            if len(sav_members) != 1:
                raise ValueError(f"Expected one .sav in {yf.path}, found {sav_members}")
            sav_path = Path(zf.extract(sav_members[0], temp_dir))

    _, meta = pyreadstat.read_sav(str(sav_path), metadataonly=True)
    available = [col for col in RAW_COLUMNS if col in meta.column_names]
    df, meta = pyreadstat.read_sav(str(sav_path), usecols=available)
    df["source_year"] = year
    return df, meta


def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def map_code(s: pd.Series, valid: Iterable[int]) -> pd.Series:
    valid_set = set(valid)
    x = as_num(s)
    return x.where(x.isin(valid_set), np.nan)


def derive_binary_dx(s: pd.Series) -> pd.Series:
    x = as_num(s)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[x == 1] = 1
    out[x == 0] = 0
    return out


def derive_family_history(df: pd.DataFrame) -> pd.Series:
    cols = ["HE_HPfh1", "HE_HPfh2", "HE_HPfh3"]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    any_yes = (values == 1).any(axis=1)
    all_no_or_structural = values.apply(lambda row: all(v in {0, 8} for v in row.dropna()), axis=1)
    no_missing = values.notna().all(axis=1)

    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[any_yes] = 1
    out[(~any_yes) & all_no_or_structural & no_missing] = 0
    return out


def derive_marital_status(df: pd.DataFrame) -> pd.Series:
    marri_1 = as_num(df["marri_1"])
    marri_2 = as_num(df["marri_2"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[marri_1 == 2] = 0
    out[(marri_1 == 1) & (marri_2 == 1)] = 1
    out[(marri_1 == 1) & (marri_2 == 2)] = 2
    out[(marri_1 == 1) & (marri_2 == 3)] = 3
    out[(marri_1 == 1) & (marri_2 == 4)] = 4
    return out


def derive_alcohol_frequency(df: pd.DataFrame) -> pd.Series:
    raw = as_num(df["BD1_11"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw.isin([1, 8])] = 0
    out[raw == 2] = 1
    out[raw == 3] = 2
    out[raw == 4] = 3
    out[raw == 5] = 4
    out[raw == 6] = 5
    return out


def derive_drink_amount(df: pd.DataFrame) -> pd.Series:
    raw_amount = as_num(df["BD2_1"])
    raw_freq = as_num(df["BD1_11"])
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw_freq.isin([1, 8]) | (raw_amount == 8)] = 0
    out[raw_amount.isin([1, 2, 3, 4, 5])] = raw_amount[raw_amount.isin([1, 2, 3, 4, 5])]
    return out


def derive_binge_drinking(df: pd.DataFrame) -> pd.Series:
    sex = as_num(df["sex"])
    male_raw = as_num(df["BD2_31"])
    female_raw = as_num(df["BD2_32"])
    raw_freq = as_num(df["BD1_11"])
    raw = pd.Series(np.nan, index=df.index, dtype="float")
    raw[sex == 1] = male_raw[sex == 1]
    raw[sex == 2] = female_raw[sex == 2]

    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw_freq.isin([1, 8]) | (raw == 8)] = 0
    out[raw.isin([1, 2, 3, 4, 5])] = raw[raw.isin([1, 2, 3, 4, 5])]
    return out


def optional_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float")
    return as_num(df[column])


def derive_activity_limitation(df: pd.DataFrame) -> pd.Series:
    raw = optional_numeric(df, "LQ4_00")
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[raw == 1] = 1
    out[raw == 2] = 0
    return out


def derive_food_security(df: pd.DataFrame) -> pd.Series:
    safe = optional_numeric(df, "LF_SAFE")
    security_group = optional_numeric(df, "LF_SECUR_G")
    out = safe.where(safe.isin([1, 2, 3, 4]), np.nan)
    return out.combine_first(security_group.where(security_group.isin([1, 2, 3, 4]), np.nan))


def derive_dyslipidemia(df: pd.DataFrame) -> pd.Series:
    dx = optional_numeric(df, "DI2_dg")
    medication = optional_numeric(df, "DI2_2")
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[(dx == 1) | medication.isin([1, 2, 3, 4])] = 1
    out[(dx == 0) & medication.isin([5, 8])] = 0
    return out


def derive_cancer_history(df: pd.DataFrame) -> pd.Series:
    cols = ["DC1_dg", "DC2_dg", "DC3_dg", "DC4_dg", "DC5_dg", "DC6_dg", "DC7_dg", "DC11_dg"]
    values = pd.DataFrame({col: optional_numeric(df, col) for col in cols})
    any_yes = (values == 1).any(axis=1)
    any_observed = values.notna().any(axis=1)
    all_observed_no = (values.fillna(0) == 0).all(axis=1) & any_observed
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out[any_yes] = 1
    out[all_observed_no & ~any_yes] = 0
    return out


def derive_heart_rate_bpm(df: pd.DataFrame) -> pd.Series:
    pulse_15 = optional_numeric(df, "HE_PLS").combine_first(optional_numeric(df, "HE_PLS_15"))
    pulse_30 = optional_numeric(df, "HE_PLS_30")
    pulse_60 = optional_numeric(df, "HE_mPLS")
    heart_rate = pulse_15 * 4.0
    heart_rate = heart_rate.combine_first(pulse_30 * 2.0)
    return heart_rate.combine_first(pulse_60)


def harmonize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    id_col = "ID" if "ID" in df.columns else "id"

    out = pd.DataFrame(index=df.index)
    out["participant_id"] = df[id_col].astype("string")
    out["year"] = as_num(df["year"]).fillna(as_num(df["source_year"]))
    out["source_year"] = as_num(df["source_year"])
    out["sex"] = map_code(df["sex"], [1, 2])
    out["age"] = as_num(df["age"])
    out["sbp_final"] = as_num(df["HE_sbp"])
    out["dbp_final"] = as_num(df["HE_dbp"])
    out["known_htn_dx_raw"] = as_num(df["DI1_dg"])
    out["antihypertensive_use_raw"] = as_num(df["DI1_2"])
    out["pregnancy_raw"] = as_num(df["HE_prg"])
    out["bmi"] = as_num(df["HE_BMI"])
    out["waist_cm"] = as_num(df["HE_wc"])
    out["diabetes_dx"] = derive_binary_dx(optional_numeric(df, "DE1_dg"))
    out["aerobic_activity"] = map_code(optional_numeric(df, "pa_aerobic"), [0, 1])
    out["self_rated_health"] = map_code(optional_numeric(df, "D_1_1"), [1, 2, 3, 4, 5])
    out["activity_limitation"] = derive_activity_limitation(df)
    out["heart_rate"] = derive_heart_rate_bpm(df)
    out["family_history_htn"] = derive_family_history(df)
    out["current_smoking"] = pd.Series(np.nan, index=df.index, dtype="float")
    bs = as_num(df["BS3_1"])
    out.loc[bs.isin([1, 2]), "current_smoking"] = 1
    out.loc[bs.isin([3, 8]), "current_smoking"] = 0
    out["alcohol_frequency"] = derive_alcohol_frequency(df)
    out["drink_amount"] = derive_drink_amount(df)
    out["binge_drinking"] = derive_binge_drinking(df)
    out["education_level"] = map_code(df["edu"], [1, 2, 3, 4])
    out["household_income_q"] = map_code(df["ho_incm"], [1, 2, 3, 4])
    out["employment_status"] = map_code(df["EC1_1"], [1, 2])
    out["marital_status"] = derive_marital_status(df)
    out["food_security_status"] = derive_food_security(df)
    out["dyslipidemia_dx_or_med"] = derive_dyslipidemia(df)
    out["cancer_history"] = derive_cancer_history(df)
    out["total_energy_kcal"] = optional_numeric(df, "N_EN").where(optional_numeric(df, "N_EN") > 0, np.nan)
    out["weight"] = as_num(df["wt_itvex"])
    out["strata"] = as_num(df["kstrata"])
    out["psu"] = df["psu"].astype("string")

    out["known_htn_dx"] = np.where(out["known_htn_dx_raw"] == 1, 1, np.where(out["known_htn_dx_raw"] == 0, 0, np.nan))
    out["antihypertensive_use"] = np.where(
        out["antihypertensive_use_raw"].isin([1, 2, 3, 4]),
        1,
        np.where(out["antihypertensive_use_raw"].isin([5, 8]), 0, np.nan),
    )
    out["pregnancy"] = np.where(
        out["pregnancy_raw"] == 1,
        1,
        np.where(out["pregnancy_raw"].isin([0, 8]), 0, np.nan),
    )
    out["elevated_bp_14090"] = ((out["sbp_final"] >= 140) | (out["dbp_final"] >= 90)).astype("float")
    out.loc[out["sbp_final"].isna() | out["dbp_final"].isna(), "elevated_bp_14090"] = np.nan
    return out


def read_harmonized(years: list[int]) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    frames = []
    columns_by_year = {}
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for year in years:
            raw, meta = read_sav_year(year, temp_dir)
            columns_by_year[year] = list(raw.columns)
            frames.append(harmonize(raw))
    return pd.concat(frames, ignore_index=True), columns_by_year


def add_flow_step(rows: list[dict], step: str, before_n: int, after_df: pd.DataFrame) -> None:
    after_n = len(after_df)
    rows.append(
        {
            "step": len(rows),
            "criterion": step,
            "remaining_n": after_n,
            "excluded_at_step_n": before_n - after_n,
            "remaining_percent_of_start": None,
        }
    )


def apply_eligibility(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    current = df.copy()
    start_n = len(current)
    rows.append(
        {
            "step": 0,
            "criterion": "Raw KNHANES ALL participants",
            "remaining_n": start_n,
            "excluded_at_step_n": 0,
            "remaining_percent_of_start": 100.0,
        }
    )

    before = len(current)
    current = current[current["age"] >= 19].copy()
    add_flow_step(rows, "Age >= 19 years", before, current)

    before = len(current)
    current = current[~((current["sex"] == 2) & (current["pregnancy"] == 1))].copy()
    add_flow_step(rows, "Exclude currently pregnant women", before, current)

    before = len(current)
    current = current[current["sbp_final"].notna() & current["dbp_final"].notna()].copy()
    add_flow_step(rows, "Exclude missing final SBP or DBP", before, current)

    before = len(current)
    current = current[current["known_htn_dx_raw"] != 1].copy()
    current = current[~current["antihypertensive_use_raw"].isin([1, 2, 3, 4])].copy()
    add_flow_step(rows, "Exclude prior hypertension diagnosis or antihypertensive medication use", before, current)

    before = len(current)
    current = current[(current["known_htn_dx"] == 0) & (current["antihypertensive_use"] == 0)].copy()
    add_flow_step(rows, "Exclude indeterminate diagnosis/medication status", before, current)

    current["undiagnosed_htn"] = current["elevated_bp_14090"].astype(int)
    current["predictor_missing_count"] = current[ALL_CANDIDATE_PREDICTORS].isna().sum(axis=1)
    current["predictor_missing_fraction"] = current["predictor_missing_count"] / len(ALL_CANDIDATE_PREDICTORS)

    before = len(current)
    current = current[current["predictor_missing_fraction"] <= 0.50].copy()
    add_flow_step(rows, "Exclude >50% missing among primary predictors", before, current)

    flow = pd.DataFrame(rows)
    flow["remaining_percent_of_start"] = flow["remaining_n"] / start_n * 100
    return current.reset_index(drop=True), flow


def flow_by_year(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year, part in df.groupby("source_year"):
        _, flow = apply_eligibility(part.reset_index(drop=True))
        flow.insert(0, "year", int(year))
        pieces.append(flow)
    return pd.concat(pieces, ignore_index=True)


def weighted_mean_binary(y: pd.Series, w: pd.Series) -> float:
    ok = y.notna() & w.notna() & (w > 0)
    if ok.sum() == 0:
        return math.nan
    return float(np.average(y[ok], weights=w[ok]))


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", "overall", df)]
    groups.extend((("year", int(year), part) for year, part in df.groupby("source_year")))
    groups.extend((("sex", CAT_LABELS["sex"].get(int(sex), sex), part) for sex, part in df.groupby("sex")))
    for group_type, group_value, part in groups:
        y = part["undiagnosed_htn"]
        rows.append(
            {
                "group_type": group_type,
                "group": group_value,
                "n": len(part),
                "events_n": int(y.sum()),
                "unweighted_prevalence_percent": y.mean() * 100,
                "weighted_prevalence_percent": weighted_mean_binary(y, part["weight"]) * 100,
                "weight_nonmissing_n": int(part["weight"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "undiagnosed_htn",
        *ALL_CANDIDATE_PREDICTORS,
        "weight",
        "strata",
        "psu",
    ]
    rows = []
    for var in variables:
        miss_n = int(df[var].isna().sum())
        rows.append(
            {
                "variable": var,
                "n": len(df),
                "missing_n": miss_n,
                "missing_percent": miss_n / len(df) * 100,
                "nonmissing_n": len(df) - miss_n,
            }
        )
    return pd.DataFrame(rows)


def continuous_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["age", "bmi", "waist_cm", "heart_rate", "total_energy_kcal", "sbp_final", "dbp_final"]:
        s = df[var].dropna()
        rows.append(
            {
                "variable": var,
                "n_nonmissing": int(s.size),
                "missing_n": int(df[var].isna().sum()),
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "median": s.median(),
                "q1": s.quantile(0.25),
                "q3": s.quantile(0.75),
                "min": s.min(),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def categorical_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = [
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
        "food_security_status",
        "dyslipidemia_dx_or_med",
        "cancer_history",
        "marital_status",
        "undiagnosed_htn",
    ]
    for var in variables:
        counts = df[var].value_counts(dropna=False).sort_index()
        for value, n in counts.items():
            missing = pd.isna(value)
            label = "missing" if missing else CAT_LABELS.get(var, {}).get(int(value), str(value))
            rows.append(
                {
                    "variable": var,
                    "value": "NA" if missing else int(value),
                    "label": label,
                    "n": int(n),
                    "percent": int(n) / len(df) * 100,
                }
            )
    return pd.DataFrame(rows)


def estimated_design_columns(df: pd.DataFrame, predictors: list[str]) -> int:
    continuous = {"age", "bmi", "waist_cm", "heart_rate", "total_energy_kcal"}
    total = 0
    for pred in predictors:
        if pred in continuous:
            total += 1
        else:
            levels = df[pred].dropna().nunique()
            total += max(1, int(levels) - 1)
    return total


def predictors_for_model_report(model_set: str, variable_set: str) -> list[str]:
    if variable_set == "him_noninvasive":
        if model_set == "male":
            return HIM_NONINVASIVE_MALE
        if model_set == "female":
            return HIM_NONINVASIVE_FEMALE
        return HIM_NONINVASIVE_INTEGRATED
    predictors = list(VARIABLE_SET_PREDICTORS[variable_set])
    if model_set in {"male", "female"}:
        predictors = [x for x in predictors if x != "sex"]
    return predictors


def model_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in [
        ("integrated", df),
        ("male", df[df["sex"] == 1]),
        ("female", df[df["sex"] == 2]),
    ]:
        for variable_set in ["core", "core_sdoh", "core_sdoh_hr", "him_noninvasive"]:
            predictors = predictors_for_model_report(name, variable_set)
            events = int(part["undiagnosed_htn"].sum())
            nonevents = int(len(part) - events)
            cols = estimated_design_columns(part, predictors)
            rows.append(
                {
                    "model_set": name,
                    "variable_set": variable_set,
                    "n": len(part),
                    "events_n": events,
                    "nonevents_n": nonevents,
                    "candidate_predictor_count": len(predictors),
                    "estimated_one_hot_design_columns": cols,
                    "events_per_design_column": events / cols if cols else math.nan,
                    "nonevents_per_design_column": nonevents / cols if cols else math.nan,
                    "logistic_feasible_rule_of_thumb": events >= 10 * cols and nonevents >= 10 * cols,
                    "tree_boosting_feasible_rule_of_thumb": events >= 100 and nonevents >= 100,
                }
            )
    return pd.DataFrame(rows)


def survey_design_validity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, part in df.groupby("source_year"):
        rows.append(
            {
                "year": int(year),
                "n": len(part),
                "weight_missing_n": int(part["weight"].isna().sum()),
                "weight_nonpositive_n": int(((part["weight"].notna()) & (part["weight"] <= 0)).sum()),
                "strata_missing_n": int(part["strata"].isna().sum()),
                "strata_unique_n": int(part["strata"].nunique(dropna=True)),
                "psu_missing_n": int(part["psu"].isna().sum()),
                "psu_unique_n": int(part["psu"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def raw_code_distribution(years: list[int]) -> pd.DataFrame:
    rows = []
    raw_vars = [
        "DI1_dg",
        "DI1_2",
        "HE_prg",
        "HE_HPfh1",
        "HE_HPfh2",
        "HE_HPfh3",
        "BS3_1",
        "BD1_11",
        "BD2_1",
        "BD2_31",
        "BD2_32",
        "DE1_dg",
        "DI2_dg",
        "DI2_2",
        "D_1_1",
        "LQ4_00",
        "pa_aerobic",
        "LF_SAFE",
        "LF_SECUR_G",
        "edu",
        "ho_incm",
        "EC1_1",
        "marri_1",
        "marri_2",
    ]
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        for year in years:
            raw, _ = read_sav_year(year, temp_dir)
            for var in raw_vars:
                if var not in raw.columns:
                    rows.append({"year": year, "raw_variable": var, "value": "MISSING_VARIABLE", "n": np.nan})
                    continue
                counts = as_num(raw[var]).value_counts(dropna=False).sort_index()
                for value, n in counts.items():
                    rows.append(
                        {
                            "year": year,
                            "raw_variable": var,
                            "value": "NA" if pd.isna(value) else int(value),
                            "n": int(n),
                        }
                    )
    return pd.DataFrame(rows)


def time_validation_feasibility(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tv_df, cols_by_year = read_harmonized(years)
    tv_final, tv_flow = apply_eligibility(tv_df)
    required_final = set(
        ALL_CANDIDATE_PREDICTORS
        + ["sbp_final", "dbp_final", "known_htn_dx_raw", "antihypertensive_use_raw", "pregnancy_raw", "weight", "strata", "psu"]
    )
    rows = []
    for year in years:
        available_raw = set(cols_by_year[year])
        for final_var, raw_var, description, _, role, _ in MAPPING_ROWS:
            if final_var in {"participant_id", "year"}:
                ok = True
                raw_status = "id/ID present" if final_var == "participant_id" else "year present"
            elif final_var == "heart_rate":
                raw_parts = ["HE_PLS", "HE_PLS_15", "HE_PLS_30", "HE_mPLS"]
                available_parts = [part for part in raw_parts if part in available_raw]
                ok = bool(available_parts)
                raw_status = ",".join(available_parts) if available_parts else "missing"
            elif final_var == "food_security_status":
                raw_parts = ["LF_SAFE", "LF_SECUR_G"]
                available_parts = [part for part in raw_parts if part in available_raw]
                ok = bool(available_parts)
                raw_status = ",".join(available_parts) if available_parts else "missing"
            elif final_var == "cancer_history":
                raw_parts = ["DC1_dg", "DC2_dg", "DC3_dg", "DC4_dg", "DC5_dg", "DC6_dg", "DC7_dg", "DC11_dg"]
                available_parts = [part for part in raw_parts if part in available_raw]
                ok = bool(available_parts)
                raw_status = ",".join(available_parts) if available_parts else "missing"
            elif "/" in raw_var:
                raw_parts = raw_var.split("/")
                if raw_var == "HE_HPfh1/2/3":
                    raw_parts = ["HE_HPfh1", "HE_HPfh2", "HE_HPfh3"]
                elif raw_var == "BD2_31/BD2_32":
                    raw_parts = ["BD2_31", "BD2_32"]
                elif raw_var == "marri_1/marri_2":
                    raw_parts = ["marri_1", "marri_2"]
                elif raw_var == "id/ID":
                    raw_parts = ["id", "ID"]
                ok = all(part in available_raw for part in raw_parts) if raw_var != "id/ID" else any(part in available_raw for part in raw_parts)
                raw_status = ",".join(part for part in raw_parts if part in available_raw)
            else:
                ok = raw_var in available_raw
                raw_status = raw_var if ok else "missing"
            rows.append(
                {
                    "year": year,
                    "final_variable": final_var,
                    "raw_variable": raw_var,
                    "role": role,
                    "available_for_time_validation": ok,
                    "raw_status": raw_status,
                    "description": description,
                }
            )
    return tv_final, tv_flow, pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_summary(
    final_df: pd.DataFrame,
    flow: pd.DataFrame,
    missing: pd.DataFrame,
    prevalence: pd.DataFrame,
    feasibility: pd.DataFrame,
    tv_final: pd.DataFrame,
    tv_var_check: pd.DataFrame,
) -> None:
    def small_markdown_table(table: pd.DataFrame) -> str:
        shown = table.copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(shown.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in shown.astype(str).to_numpy()]
        return "\n".join([header, sep, *body])

    overall_prev = prevalence[(prevalence["group_type"] == "overall") & (prevalence["group"] == "overall")].iloc[0]
    max_pred_missing = missing[missing["variable"].isin(ALL_CANDIDATE_PREDICTORS)]["missing_percent"].max()
    unavailable_tv = tv_var_check[~tv_var_check["available_for_time_validation"]]
    unavailable_primary_tv = unavailable_tv[unavailable_tv["role"] == "predictor"]

    lines = [
        "# KNHANES Primary Development Dataset Summary",
        "",
        "## Locked Rules",
        "",
        "- Development years: 2014-2020",
        "- Adult analytic population: age >= 19 years",
        "- Exclusions: current pregnancy, missing final SBP/DBP, prior hypertension diagnosis, antihypertensive medication use, indeterminate diagnosis/medication status, >50% missing among primary predictors",
        "- Outcome: final SBP >= 140 mmHg or final DBP >= 90 mmHg among participants with no prior hypertension diagnosis and no antihypertensive medication use",
        "- Leakage variables excluded from predictors: final and repeated BP measurements, hypertension diagnosis, antihypertensive medication use, same-day hypertension medication, and derived outcome flags",
        "",
        "## Key Results",
        "",
        f"- Final development N: {len(final_df):,}",
        f"- Events: {int(final_df['undiagnosed_htn'].sum()):,}",
        f"- Unweighted prevalence: {overall_prev['unweighted_prevalence_percent']:.2f}%",
        f"- Weighted prevalence: {overall_prev['weighted_prevalence_percent']:.2f}%",
        f"- Maximum predictor missingness after exclusions: {max_pred_missing:.2f}%",
        f"- Time-validation check N after same eligibility screen, 2021-2022: {len(tv_final):,}",
        f"- Time-validation unavailable primary predictor variables: {len(unavailable_primary_tv):,}",
        f"- Time-validation unavailable benchmark-only variables: {len(unavailable_tv) - len(unavailable_primary_tv):,}",
        "",
        "## Model Feasibility",
        "",
        small_markdown_table(feasibility),
        "",
        "## Flow",
        "",
        small_markdown_table(flow),
        "",
    ]
    (REPORT_DIR / "knhanes_development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.DataFrame(
        MAPPING_ROWS,
        columns=["final_variable", "raw_variable", "description", "coding_or_derivation", "role", "included_in_final_model_input"],
    )
    write_csv(mapping, REPORT_DIR / "knhanes_variable_mapping_primary.csv")

    dev_raw_harmonized, _ = read_harmonized(DEVELOPMENT_YEARS)
    final_df, flow = apply_eligibility(dev_raw_harmonized)
    flow_year = flow_by_year(dev_raw_harmonized)
    prevalence = prevalence_table(final_df)
    missing = missingness_table(final_df)
    continuous = continuous_distribution(final_df)
    categorical = categorical_distribution(final_df)
    feasibility = model_feasibility(final_df)
    design_validity = survey_design_validity(final_df)
    raw_codes = raw_code_distribution(DEVELOPMENT_YEARS)
    tv_final, tv_flow, tv_var_check = time_validation_feasibility(TIME_VALIDATION_YEARS)
    tv_prevalence = prevalence_table(tv_final)
    tv_missing = missingness_table(tv_final)
    tv_design_validity = survey_design_validity(tv_final)

    output_cols = [
        "participant_id",
        "source_year",
        "year",
        "sex",
        "age",
        "undiagnosed_htn",
        "elevated_bp_14090",
        "bmi",
        "waist_cm",
        "diabetes_dx",
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
        "dyslipidemia_dx_or_med",
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
    ]
    write_csv(final_df[output_cols], PROCESSED_DIR / "knhanes_development_2014_2020_primary.csv")
    write_csv(tv_final[output_cols], PROCESSED_DIR / "knhanes_time_validation_2021_2022_check_only.csv")

    write_csv(flow, REPORT_DIR / "knhanes_development_flow_overall.csv")
    write_csv(flow_year, REPORT_DIR / "knhanes_development_flow_by_year.csv")
    write_csv(prevalence, REPORT_DIR / "knhanes_development_outcome_prevalence.csv")
    write_csv(missing, REPORT_DIR / "knhanes_development_missingness_final.csv")
    write_csv(continuous, REPORT_DIR / "knhanes_development_distribution_continuous.csv")
    write_csv(categorical, REPORT_DIR / "knhanes_development_distribution_categorical.csv")
    write_csv(feasibility, REPORT_DIR / "knhanes_model_feasibility.csv")
    write_csv(design_validity, REPORT_DIR / "knhanes_development_survey_design_validity.csv")
    write_csv(raw_codes, REPORT_DIR / "knhanes_raw_code_distribution_2014_2020.csv")
    write_csv(tv_flow, REPORT_DIR / "knhanes_time_validation_flow_2021_2022.csv")
    write_csv(tv_prevalence, REPORT_DIR / "knhanes_time_validation_outcome_prevalence_2021_2022_check_only.csv")
    write_csv(tv_missing, REPORT_DIR / "knhanes_time_validation_missingness_2021_2022_check_only.csv")
    write_csv(tv_design_validity, REPORT_DIR / "knhanes_time_validation_survey_design_validity_2021_2022_check_only.csv")
    write_csv(tv_var_check, REPORT_DIR / "knhanes_time_validation_variable_check_2021_2022.csv")

    write_summary(final_df, flow, missing, prevalence, feasibility, tv_final, tv_var_check)


if __name__ == "__main__":
    main()
