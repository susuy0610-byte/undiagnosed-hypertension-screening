#!/usr/bin/env python3
"""QC HSE 2022 mapping feasibility for the frozen core_sdoh_hr model.

This script is intentionally read-only with respect to raw HSE data. It does not
build a prediction input file and does not run any frozen model.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata"
DTA = RAW_DIR / "stata/stata13/hse_2022_eul_v1.dta"
XLSX = RAW_DIR / "mrdoc/excel/9469_hse_2022_eul_variable_information_v1.xlsx"
OUT_DIR = ROOT / "reports"
OUT_DATE = "20260817"

MISSING_CODES = {-99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -9, -8, -7, -6, -5, -4, -3, -2, -1}


def clean_label(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).replace("\n", " ").strip()


def read_metadata() -> tuple[dict[str, str], dict[str, dict[Any, str]], pd.DataFrame]:
    with pd.io.stata.StataReader(DTA) as reader:
        var_labels = {k: clean_label(v) for k, v in reader.variable_labels().items()}
        value_labels = reader.value_labels()

    try:
        variables = pd.read_excel(XLSX, sheet_name="Variables")
    except Exception:
        variables = pd.DataFrame()
    if not variables.empty:
        variables.columns = [str(c).strip() for c in variables.columns]
    return var_labels, value_labels, variables


def is_missing(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_numeric_dtype(series):
        mask = mask | series.isin(MISSING_CODES)
    return mask


def valid_numeric(series: pd.Series) -> pd.Series:
    mask = ~is_missing(series)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series[mask], errors="coerce")
    return pd.to_numeric(series[mask], errors="coerce")


def label_for(value_labels: dict[str, dict[Any, str]], var: str, value: Any) -> str:
    labels = value_labels.get(var, {})
    if not labels:
        return ""
    candidates = [value]
    try:
        candidates.append(int(value))
    except Exception:
        pass
    try:
        candidates.append(float(value))
    except Exception:
        pass
    for candidate in candidates:
        if candidate in labels:
            return clean_label(labels[candidate])
    return ""


def first_existing(columns: list[str], *names: str) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in names:
        hit = lower.get(name.lower())
        if hit is not None:
            return hit
    return None


def find_candidates(
    columns: list[str],
    var_labels: dict[str, str],
    patterns: list[str],
    limit: int = 60,
) -> list[dict[str, str]]:
    compiled = [re.compile(p, re.I) for p in patterns]
    out = []
    for col in columns:
        haystack = f"{col} {var_labels.get(col, '')}"
        if any(p.search(haystack) for p in compiled):
            out.append({"variable": col, "label": var_labels.get(col, "")})
    return out[:limit]


def top_value_summary(df: pd.DataFrame, value_labels: dict[str, dict[Any, str]], var: str, n: int = 8) -> str:
    if var not in df.columns:
        return ""
    s = df[var]
    vc = s.value_counts(dropna=False).head(n)
    parts = []
    for value, count in vc.items():
        if pd.isna(value):
            raw = "NA"
        elif isinstance(value, float) and value.is_integer():
            raw = str(int(value))
        else:
            raw = str(value)
        lab = label_for(value_labels, var, value)
        if lab:
            parts.append(f"{raw}={lab} (n={int(count)})")
        else:
            parts.append(f"{raw} (n={int(count)})")
    return "; ".join(parts)


def variable_missing(df: pd.DataFrame, var: str) -> tuple[int, float]:
    if var not in df.columns:
        return len(df), 100.0
    miss = is_missing(df[var])
    return int(miss.sum()), float(miss.mean() * 100)


def numeric_range(df: pd.DataFrame, var: str) -> str:
    if var not in df.columns:
        return ""
    vals = valid_numeric(df[var]).dropna()
    if vals.empty:
        return ""
    return f"min={vals.min():.2f}, median={vals.median():.2f}, max={vals.max():.2f}"


def derived_mean_missing(df: pd.DataFrame, vars_: list[str]) -> tuple[int, float, str]:
    available = [v for v in vars_ if v in df.columns]
    if not available:
        return len(df), 100.0, ""
    temp = []
    for var in available:
        s = pd.to_numeric(df[var], errors="coerce")
        s = s.mask(is_missing(df[var]))
        temp.append(s)
    mat = pd.concat(temp, axis=1)
    derived = mat.mean(axis=1, skipna=True)
    miss = derived.isna()
    vals = derived[~miss]
    rng = ""
    if not vals.empty:
        rng = f"min={vals.min():.2f}, median={vals.median():.2f}, max={vals.max():.2f}"
    return int(miss.sum()), float(miss.mean() * 100), rng


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    shown = df.loc[:, columns].copy()
    for col in shown.columns:
        shown[col] = (
            shown[col]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("|", "\\|", regex=False)
        )
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, separator, *body])


def main() -> None:
    if not DTA.exists():
        raise FileNotFoundError(DTA)
    if not XLSX.exists():
        raise FileNotFoundError(XLSX)

    var_labels, value_labels, _variables = read_metadata()
    df = pd.read_stata(DTA, convert_categoricals=False)
    columns = list(df.columns)
    n_total = len(df)

    exact_age = first_existing(columns, "age", "ageyr", "ageyrs", "ageyears")
    age_proxy = first_existing(columns, "age16g5", "ag16g10", "age35g", "ag8g2")
    age_selected = exact_age or age_proxy
    age_class = "Direct" if exact_age else "Proxy"
    age_note = (
        "Exact continuous age found."
        if exact_age
        else "Exact continuous age was not found in the EUL Stata file; closest available option is grouped age."
    )

    selected = {
        "person_id": [first_existing(columns, "seriala")],
        "age": [age_selected],
        "sex": [first_existing(columns, "Sex")],
        "bmi": [first_existing(columns, "bmival", "bmival2")],
        "waist_cm": [first_existing(columns, "wstval")],
        "heart_rate": [first_existing(columns, "PULS2OM"), first_existing(columns, "PULS3OM")],
        "sbp_final": [first_existing(columns, "omsysval", "omsyst")],
        "dbp_final": [first_existing(columns, "omdiaval", "omdiast")],
        "known_htn_dx": [first_existing(columns, "bp1", "DocBP", "OthBP")],
        "antihypertensive_use": [first_existing(columns, "bpmedd2", "hypertakg2", "hyperatakg2")],
        "pregnancy": [first_existing(columns, "PregNTJ", "PregNowB20", "NCPregJ")],
        "diabetes_dx": [first_existing(columns, "diabete2", "Diabetes", "EverDi")],
        "aerobic_activity": [first_existing(columns, "mvpamwk", "mvpamwkg")],
        "self_rated_health": [first_existing(columns, "GenHelf", "genhelf2")],
        "activity_limitation": [first_existing(columns, "ReducAct")],
        "family_history_htn": [],
        "current_smoking": [first_existing(columns, "cigsta3_19", "SmokeNow", "CSmokeNow")],
        "alcohol_frequency": [first_existing(columns, "dnoft_22", "ADrinkOf")],
        "drink_amount": [first_existing(columns, "AUDTyDy", "totalwu_22", "totalwug2_22")],
        "binge_drinking": [first_existing(columns, "AUD6Mre", "alclimit7b_22", "d7many_22")],
        "education_level": [first_existing(columns, "educ2", "topqual4", "topqual3", "EducEnd")],
        "household_income_q": [first_existing(columns, "eqv5", "hhinc3", "jntinc2")],
        "employment_status": [first_existing(columns, "activb2", "StWork")],
        "marital_status": [first_existing(columns, "marstatd_22")],
        "food_security_status": [],
    }
    selected = {k: [v for v in vs if v] for k, vs in selected.items()}

    rows = []

    def add_row(
        role: str,
        kn: str,
        required: str,
        hse_vars: list[str],
        mapping_class: str,
        rule: str,
        qc_flag: str,
        notes: str,
    ) -> None:
        labels = "; ".join(f"{v}: {var_labels.get(v, '')}" for v in hse_vars)
        if kn == "heart_rate" and hse_vars:
            miss_n, miss_pct, rng = derived_mean_missing(df, hse_vars)
            coding = "Derived mean QC from selected pulse readings"
        elif not hse_vars:
            miss_n, miss_pct, rng = n_total, 100.0, ""
            coding = "Not available in selected raw variables"
        else:
            miss_n, miss_pct = variable_missing(df, hse_vars[0])
            rng = numeric_range(df, hse_vars[0])
            coding = top_value_summary(df, value_labels, hse_vars[0])
        rows.append(
            {
                "role": role,
                "knhanes_final_variable": kn,
                "required_for_model": required,
                "hse_selected_variable": "; ".join(hse_vars),
                "hse_variable_label": labels,
                "mapping_class": mapping_class,
                "raw_missing_n_or_all_na_plan": miss_n,
                "raw_missing_pct_or_all_na_plan": round(miss_pct, 2),
                "range_or_numeric_summary": rng,
                "value_coding_top_values": coding,
                "proposed_harmonization_rule": rule,
                "qc_flag": qc_flag,
                "notes": notes,
            }
        )

    add_row("id", "person_id", "yes", selected["person_id"], "Direct", "Use `seriala` as participant identifier.", "OK", "Needed for deduplication and joins.")
    add_row("predictor", "age", "yes", selected["age"], age_class, "Use exact age if available; otherwise use pre-specified grouped-age midpoint only if accepted as proxy.", "LIMITATION" if age_class == "Proxy" else "OK", age_note)
    add_row("predictor", "sex", "integrated only", selected["sex"], "Direct", "Map male/female to KNHANES coding; omit in sex-specific models.", "OK", "")
    add_row("predictor", "bmi", "yes", selected["bmi"], "Direct", "Use valid measured BMI kg/m2.", "OK", "")
    add_row("predictor", "waist_cm", "yes", selected["waist_cm"], "Direct", "Use valid mean waist circumference in cm.", "OK", "")
    add_row("predictor", "heart_rate", "yes", selected["heart_rate"], "Harmonization", "Use mean of 2nd and 3rd Omron pulse readings in bpm; do not use `ompuls/ompulval` because labels refer to pulse pressure.", "OK" if selected["heart_rate"] else "BLOCKER", "HSE has raw pulse readings; average derived to match BP-reading convention.")
    add_row("outcome/exclusion", "sbp_final", "outcome only", selected["sbp_final"], "Direct", "Use valid Omron mean systolic BP, based on HSE valid derived variable.", "OK", "Never use as predictor.")
    add_row("outcome/exclusion", "dbp_final", "outcome only", selected["dbp_final"], "Direct", "Use valid Omron mean diastolic BP, based on HSE valid derived variable.", "OK", "Never use as predictor.")
    add_row("outcome/exclusion", "known_htn_dx", "exclusion/outcome", selected["known_htn_dx"], "Direct", "Exclude doctor/nurse-diagnosed high blood pressure excluding pregnancy-only hypertension.", "OK", "Candidate `bp1` is a derived diagnosis variable excluding pregnancy.")
    add_row("outcome/exclusion", "antihypertensive_use", "exclusion/outcome", selected["antihypertensive_use"], "Direct", "Exclude prescribed drug use for blood pressure.", "OK", "Use current/prescribed BP-medication variable; review high missingness as skip pattern, not necessarily unknown.")
    add_row("outcome/exclusion", "pregnancy", "exclusion", selected["pregnancy"], "Direct", "Exclude currently pregnant respondents.", "OK", "Pregnant participants may already lack BP measurement.")
    add_row("predictor", "diabetes_dx", "yes", selected["diabetes_dx"], "Direct", "Use doctor-diagnosed diabetes excluding gestational-only diabetes if derived variable supports it.", "OK", "")
    add_row("predictor", "aerobic_activity", "yes", selected["aerobic_activity"], "Harmonization", "Use HSE IPAQ-derived weekly MVPA minutes and pre-specify >=150 min/week as KNHANES-compatible aerobic activity yes/no.", "OK" if selected["aerobic_activity"] else "REVIEW", "Prefer `mvpamwk` over `mvpamwkg` because the grouped HSE variable uses 30 min/week, not the usual 150 min/week guideline threshold.")
    add_row("predictor", "self_rated_health", "yes", selected["self_rated_health"], "Direct", "Map ordered self-rated general health categories.", "OK", "")
    add_row("predictor", "activity_limitation", "yes", selected["activity_limitation"], "Harmonization", "Use whether illness/disability reduces day-to-day activities.", "OK", "")
    add_row("predictor", "family_history_htn", "yes", selected["family_history_htn"], "Not available", "Keep frozen predictor column as missing if external validation proceeds; do not substitute unrelated family-history variables.", "LIMITATION", "No parent/sibling hypertension family history variable was selected from raw search.")
    add_row("predictor", "current_smoking", "yes", selected["current_smoking"], "Harmonization", "Map current smoker vs non-current.", "OK", "")
    add_row("predictor", "alcohol_frequency", "yes", selected["alcohol_frequency"], "Harmonization", "Map alcohol drinking frequency in past 12 months to KNHANES ordinal frequency.", "OK", "")
    add_row("predictor", "drink_amount", "yes", selected["drink_amount"], "Direct", "Use AUDIT typical number of alcoholic drinks on a drinking day and map ordered categories to KNHANES usual drink amount.", "OK" if selected["drink_amount"] else "REVIEW", "`AUDTyDy` is closer to KNHANES usual amount than weekly alcohol units.")
    add_row("predictor", "binge_drinking", "yes", selected["binge_drinking"], "Harmonization", "Use AUDIT frequency of six or more drinks on one occasion and map ordered frequency to KNHANES binge-drinking frequency.", "REVIEW", "AUDIT Q3 is close but not identical to KNHANES sex-specific binge thresholds.")
    add_row("predictor", "education_level", "yes", selected["education_level"], "Harmonization", "Collapse HSE education/qualification to KNHANES 4-level education.", "OK", "")
    add_row("predictor", "household_income_q", "yes", selected["household_income_q"], "Harmonization", "Use equivalised household income quintile; collapse or rank-map to KNHANES household income quartile.", "OK", "")
    add_row("predictor", "employment_status", "yes", selected["employment_status"], "Harmonization", "Map activity status/current paid work to employed vs not employed.", "OK", "Candidate `activb2` is individual activity status, not household reference person status.")
    add_row("predictor", "marital_status", "yes", selected["marital_status"], "Harmonization", "Collapse marital status to KNHANES categories.", "OK", "")
    add_row("predictor", "food_security_status", "yes", selected["food_security_status"], "Not available", "Keep frozen predictor column as missing if external validation proceeds; fruit/vegetable intake is not a food-security proxy.", "LIMITATION", "No food insecurity/security item was selected from raw search.")

    mapping = pd.DataFrame(rows)
    mapping_path = OUT_DIR / f"hse_2022_core_sdoh_hr_mapping_qc_{OUT_DATE}.csv"
    mapping.to_csv(mapping_path, index=False)

    dist_rows = []
    all_vars = sorted({v for vs in selected.values() for v in vs})
    for var in all_vars:
        miss_n, miss_pct = variable_missing(df, var)
        dist_rows.append(
            {
                "variable": var,
                "label": var_labels.get(var, ""),
                "missing_n": miss_n,
                "missing_pct": round(miss_pct, 2),
                "range": numeric_range(df, var),
                "top_values": top_value_summary(df, value_labels, var, n=12),
            }
        )

    # Add targeted discovery lists for variables not directly selected or needing review.
    discovery_specs = {
        "age": [r"^age", r"^ag\d", r"\bage\b", r"age group"],
        "family_history_htn": [r"family", r"father", r"mother", r"parent", r"sibling", r"hypertension", r"high blood pressure"],
        "food_security_status": [r"food secur", r"food insuff", r"hunger", r"afford food", r"food bank", r"eat enough"],
        "alcohol_quantity": [r"audit", r"usual", r"typical", r"units", r"heaviest", r"drink.*day", r"limit", r"binge", r"six"],
        "physical_activity": [r"mvpa", r"ipaq", r"aerobic", r"moderate", r"vigorous", r"activity", r"walk"],
    }
    discovery_rows = []
    for topic, patterns in discovery_specs.items():
        for item in find_candidates(columns, var_labels, patterns, limit=80):
            discovery_rows.append({"topic": topic, **item})
    discovery = pd.DataFrame(discovery_rows)
    discovery_path = OUT_DIR / f"hse_2022_core_sdoh_hr_discovery_candidates_{OUT_DATE}.csv"
    discovery.to_csv(discovery_path, index=False)

    dist_path = OUT_DIR / f"hse_2022_core_sdoh_hr_selected_variable_distributions_{OUT_DATE}.csv"
    pd.DataFrame(dist_rows).to_csv(dist_path, index=False)

    ok_predictors = mapping.query("role == 'predictor' and mapping_class in ['Direct', 'Harmonization']").shape[0]
    proxy_predictors = mapping.query("role == 'predictor' and mapping_class == 'Proxy'").shape[0]
    unavailable_predictors = mapping.query("role == 'predictor' and mapping_class == 'Not available'").shape[0]
    limitations = mapping[mapping["qc_flag"].isin(["LIMITATION", "REVIEW", "BLOCKER"])]

    md_path = OUT_DIR / f"hse_2022_core_sdoh_hr_mapping_qc_{OUT_DATE}.md"
    md = []
    md.append("# HSE 2022 Mapping QC for Frozen `core_sdoh_hr` Model\n")
    md.append(f"Date: {OUT_DATE[:4]}-{OUT_DATE[4:6]}-{OUT_DATE[6:]}\n")
    md.append("Scope: HSE 2022 UK Data Service participant-level Stata file and official variable-information Excel codebook. This QC does not run external prediction, retraining, tuning, or validation metrics.\n")
    md.append("## Source Files\n")
    md.append(f"- Stata data: `{DTA.relative_to(ROOT)}`\n")
    md.append(f"- Variable information: `{XLSX.relative_to(ROOT)}`\n")
    md.append(f"- Rows in raw Stata file: {n_total:,}\n")
    md.append("- Missingness percentages below are raw-file QC percentages. They include children, participants without a nurse visit, and questionnaire skip patterns; they are not final external-cohort missingness estimates.\n")
    md.append("## Summary\n")
    md.append(f"- Direct/harmonizable predictors: {ok_predictors}\n")
    md.append(f"- Proxy predictors needing pre-specified rules: {proxy_predictors}\n")
    md.append(f"- Not available predictors: {unavailable_predictors}\n")
    md.append("- Outcome/exclusion variables for SBP, DBP, diagnosed hypertension, BP medication, and pregnancy are present as raw or derived HSE variables.\n")
    md.append("- Important limitation: exact continuous age was not found in the EUL Stata file during this QC; grouped age is available and would require a pre-specified proxy rule.\n")
    md.append("- Because only grouped age is available, the KNHANES adult criterion of age >=19 cannot be reproduced exactly. A conservative HSE rule would start at the 20-24 group; including the 16-19 group would mix 16-18-year-old participants with 19-year-olds.\n")
    md.append("- Important limitation: `family_history_htn` and `food_security_status` were not found as direct raw HSE variables.\n")
    md.append("- Important correction to the preliminary mapping: HSE variables labelled `ompuls/ompulval` refer to pulse pressure; heart rate should be derived from `PULS2OM` and `PULS3OM` pulse readings in bpm.\n")
    md.append("## Final Mapping QC Table\n")
    md.append(markdown_table(mapping, [
        "role",
        "knhanes_final_variable",
        "hse_selected_variable",
        "mapping_class",
        "raw_missing_pct_or_all_na_plan",
        "qc_flag",
        "proposed_harmonization_rule",
    ]))
    md.append("\n## Variables Requiring Review or Limitation Notes\n")
    if limitations.empty:
        md.append("No limitations flagged.\n")
    else:
        md.append(markdown_table(limitations, [
            "knhanes_final_variable",
            "hse_selected_variable",
            "mapping_class",
            "qc_flag",
            "notes",
        ]))
    md.append("\n## Selected Raw Variable Distributions\n")
    md.append(markdown_table(pd.DataFrame(dist_rows), [
        "variable",
        "missing_pct",
        "range",
        "top_values",
    ]))
    md.append("\n## Discovery Candidate Files\n")
    md.append(f"- Mapping QC CSV: `{mapping_path.relative_to(ROOT)}`\n")
    md.append(f"- Selected-variable distribution CSV: `{dist_path.relative_to(ROOT)}`\n")
    md.append(f"- Candidate discovery CSV: `{discovery_path.relative_to(ROOT)}`\n")
    md.append("## Current Judgment\n")
    md.append("HSE 2022 is technically promising for outcome construction and most predictors, but strict external validation of the frozen `core_sdoh_hr` model is not fully direct because at least `family_history_htn`, `food_security_status`, and exact continuous age are not directly available in the EUL raw file. Prediction should not be run until the age proxy/inclusion rule, all-missing handling for unavailable predictors, and binge-drinking harmonization rule are explicitly locked.\n")
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Loaded HSE 2022 rows: {n_total:,}, columns: {df.shape[1]:,}")
    print(f"Wrote {mapping_path}")
    print(f"Wrote {dist_path}")
    print(f"Wrote {discovery_path}")
    print(f"Wrote {md_path}")
    print("\nLimitations/review rows:")
    print(limitations[[
        "knhanes_final_variable",
        "hse_selected_variable",
        "mapping_class",
        "qc_flag",
        "raw_missing_pct_or_all_na_plan",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
