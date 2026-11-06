#!/usr/bin/env python3
"""QC HSE 2018/2019/2021/2022 raw-variable mapping to core_sdoh_hr.

This is a read-only QC script for external-validation preparation. It reads
local UK Data Service EUL Stata files, checks raw variable availability, and
writes mapping/missingness reports. It does not build prediction inputs, run
models, retrain, tune, recalibrate, or change thresholds.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STAMP = "20260819"

HSE_FILES = {
    "2018": ROOT / "data/raw/hse/2018/UKDA-8649-stata/stata/stata13/hse_2018_eul_15082022.dta",
    "2019": ROOT / "data/raw/hse/2019/UKDA-8860-stata/stata/stata13/hse_2019_eul_20211006.dta",
    "2021": ROOT / "data/raw/hse/2021/UKDA-9319-stata/stata/stata13/hse_2021_eul_v1.dta",
    "2022": ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata/stata/stata13/hse_2022_eul_v1.dta",
}

NEGATIVE_MISSING = {
    -99,
    -98,
    -97,
    -96,
    -95,
    -94,
    -93,
    -92,
    -91,
    -90,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
}


MAPPING_SPECS: list[dict[str, Any]] = [
    {
        "role": "id",
        "knhanes_variable": "person_id",
        "candidates": ["seriala"],
        "class_if_found": "Direct",
        "rule": "Use archive respondent serial number as participant identifier within each HSE year.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "age",
        "candidates": ["age", "ageyr", "ageyrs", "ageyears"],
        "fallback_candidates": ["age16g5"],
        "class_if_found": "Direct",
        "class_if_fallback": "Proxy",
        "rule": "Requested KNHANES adult criterion is age >=19. HSE EUL raw files do not expose exact age; if only age16g5 is available, 19-year-olds cannot be isolated from the 16-19 band. Conservative proxy remains age16g5 >=3 (20+).",
    },
    {
        "role": "predictor",
        "knhanes_variable": "sex",
        "candidates": ["sex"],
        "class_if_found": "Direct",
        "rule": "Use HSE sex variable; omit from sex-specific frozen model inputs.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "bmi",
        "candidates": ["bmival", "bmival2"],
        "fallback_candidates": ["esthtn+estwtn", "bmival_from_height_weight", "bmisr_adj"],
        "class_if_found": "Direct",
        "class_if_fallback": "Harmonization",
        "rule": "Prefer HSE valid measured BMI. If absent, derive from final measured height/weight when available; self-reported adjusted BMI is only a secondary proxy.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "waist_cm",
        "candidates": ["wstval"],
        "class_if_found": "Direct",
        "rule": "Use valid mean waist circumference in cm.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "heart_rate",
        "candidates": ["puls2om+puls3om"],
        "class_if_found": "Harmonization",
        "rule": "Use mean of second and third Omron pulse readings in bpm.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "sbp_final",
        "candidates": ["omsysval"],
        "class_if_found": "Direct",
        "rule": "Use valid mean Omron systolic BP for outcome only; never use as predictor.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "dbp_final",
        "candidates": ["omdiaval"],
        "class_if_found": "Direct",
        "rule": "Use valid mean Omron diastolic BP for outcome only; never use as predictor.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "known_htn_dx",
        "candidates": ["bp1"],
        "class_if_found": "Direct",
        "rule": "Use doctor-diagnosed high blood pressure excluding pregnancy-only hypertension for exclusion/outcome definition.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "antihypertensive_use",
        "candidates": ["bpmedd2"],
        "class_if_found": "Direct",
        "rule": "Use prescribed blood-pressure drug use for exclusion/outcome definition.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "pregnancy",
        "candidates": ["pregntj"],
        "class_if_found": "Direct",
        "rule": "Exclude currently pregnant participants when identifiable.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "diabetes_dx",
        "candidates": ["diabete2", "diabete2r"],
        "class_if_found": "Direct",
        "rule": "Use doctor-diagnosed diabetes excluding pregnancy-only diabetes when available.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "aerobic_activity",
        "candidates": ["mvpamwk"],
        "fallback_candidates": ["mins10spta", "recs21"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Proxy",
        "rule": "Prefer IPAQ MVPA minutes/week and define >=150 min/week. 2021 alternatives are not identical; 2019 lacks a clear adult MVPA variable in the raw file.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "self_rated_health",
        "candidates": ["genhelf"],
        "class_if_found": "Direct",
        "rule": "Map ordered self-rated general health categories.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "activity_limitation",
        "candidates": ["limitill"],
        "fallback_candidates": ["reducact"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Harmonization",
        "rule": "Use limiting longstanding illness; ReducAct is retained as a close fallback if needed.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "family_history_htn",
        "candidates": [],
        "class_if_missing": "Not available",
        "rule": "No participant-level family history of hypertension variable identified. Do not substitute parental smoking or parental BMI.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "current_smoking",
        "candidates": ["cigsta3_19", "cigst1"],
        "fallback_candidates": ["cignow", "cignow_19", "smokelmon"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Proxy",
        "rule": "Map current smoker vs non-current from cigarette-smoking status.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "alcohol_frequency",
        "candidates": ["dnoft_22", "dnoft_19", "dnoft"],
        "class_if_found": "Harmonization",
        "rule": "Map drinking frequency in past 12 months to KNHANES ordinal alcohol frequency.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "drink_amount",
        "candidates": ["audtydy"],
        "fallback_candidates": ["d7unitwgrp_22", "d7unitwgrp_19", "d7unitwgrp"],
        "class_if_found": "Direct",
        "class_if_fallback": "Proxy",
        "rule": "AUDTyDy is closest to usual drinks/day but only appears in 2022. For pooled years, heaviest-day alcohol units in the last 7 days are only a proxy and should be flagged.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "binge_drinking",
        "candidates": ["aud6mre"],
        "fallback_candidates": ["alclimit7b_22", "alclimit7b_19", "alclimit07b"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Proxy",
        "rule": "AUDIT frequency of >=6 drinks is closest but only appears in 2022. Alcohol-limit/heaviest-day proxies are not equivalent to KNHANES sex-specific binge frequency.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "education_level",
        "candidates": ["educ2"],
        "fallback_candidates": ["topqual4", "topqual3"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Harmonization",
        "rule": "Collapse HSE highest qualification to KNHANES education levels.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "household_income_q",
        "candidates": ["eqv5"],
        "class_if_found": "Harmonization",
        "rule": "Use equivalised household income quintile; rank-map/collapse to KNHANES income quartile if prediction input is built.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "employment_status",
        "candidates": ["activb2"],
        "fallback_candidates": ["activb3", "stwork"],
        "class_if_found": "Harmonization",
        "class_if_fallback": "Harmonization",
        "rule": "Map activity status/current paid work to employed vs not employed.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "marital_status",
        "candidates": ["marstatd_22", "marstatd"],
        "class_if_found": "Harmonization",
        "rule": "Collapse marital status including cohabitees to KNHANES marital categories.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "food_security_status",
        "candidates": [],
        "class_if_missing": "Not available",
        "rule": "No direct food insecurity/security item identified. Do not substitute fruit/vegetable intake.",
    },
]


def clean_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).replace("\n", " ").strip()


def read_metadata(path: Path) -> tuple[dict[str, str], list[str]]:
    with pd.io.stata.StataReader(path) as reader:
        labels = {name: clean_label(label) for name, label in reader.variable_labels().items()}
    return labels, list(labels)


def lower_lookup(columns: list[str]) -> dict[str, str]:
    return {col.lower(): col for col in columns}


def first_existing(columns: list[str], candidates: list[str]) -> list[str]:
    lookup = lower_lookup(columns)
    hits = []
    for candidate in candidates:
        parts = candidate.split("+")
        if len(parts) > 1:
            found = [lookup.get(part.lower()) for part in parts]
            if all(found):
                hits.extend([x for x in found if x])
                return list(dict.fromkeys(hits))
        else:
            found = lookup.get(candidate.lower())
            if found:
                hits.append(found)
                return list(dict.fromkeys(hits))
    return []


def clean_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(NEGATIVE_MISSING))


def is_missing(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return series.isna() | values.isin(NEGATIVE_MISSING)


def derived_values(df: pd.DataFrame, variable: str, cols: list[str]) -> pd.Series:
    if variable == "heart_rate" and len(cols) >= 2:
        values = [clean_numeric(df[col]).where(clean_numeric(df[col]).between(30, 220)) for col in cols[:2]]
        return pd.concat(values, axis=1).mean(axis=1, skipna=True)
    if variable == "bmi" and {"EstHtN", "EstWtN"}.issubset(set(cols)):
        height = clean_numeric(df["EstHtN"]).where(clean_numeric(df["EstHtN"]).between(120, 230))
        weight = clean_numeric(df["EstWtN"]).where(clean_numeric(df["EstWtN"]).between(25, 250))
        return weight / ((height / 100) ** 2)
    if variable == "bmi" and {"esthtn", "estwtn"}.issubset({c.lower() for c in cols}):
        height_col = next(c for c in cols if c.lower() == "esthtn")
        weight_col = next(c for c in cols if c.lower() == "estwtn")
        height = clean_numeric(df[height_col]).where(clean_numeric(df[height_col]).between(120, 230))
        weight = clean_numeric(df[weight_col]).where(clean_numeric(df[weight_col]).between(25, 250))
        return weight / ((height / 100) ** 2)
    if cols:
        return clean_numeric(df[cols[0]])
    return pd.Series(np.nan, index=df.index)


def summarize_series(series: pd.Series) -> tuple[int, float, str, str]:
    missing = series.isna()
    non_missing = series[~missing]
    missing_n = int(missing.sum())
    missing_pct = float(missing.mean() * 100) if len(series) else 100.0
    numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
    if numeric.empty:
        range_summary = ""
    else:
        range_summary = f"min={numeric.min():.2f}, median={numeric.median():.2f}, max={numeric.max():.2f}"
    vc = non_missing.value_counts(dropna=False).head(8)
    top_values = "; ".join([f"{idx} (n={int(count)})" for idx, count in vc.items()])
    return missing_n, round(missing_pct, 2), range_summary, top_values


def selected_columns_for_year(labels: dict[str, str], columns: list[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for spec in MAPPING_SPECS:
        variable = spec["knhanes_variable"]
        hits = first_existing(columns, spec.get("candidates", []))
        mapping_class = spec.get("class_if_found", "Direct") if hits else ""
        source = "primary"
        if not hits and spec.get("fallback_candidates"):
            hits = first_existing(columns, spec["fallback_candidates"])
            mapping_class = spec.get("class_if_fallback", "Proxy") if hits else ""
            source = "fallback"
        if not hits:
            mapping_class = spec.get("class_if_missing", "Not available")
            source = "missing"
        selected[variable] = {
            "role": spec["role"],
            "selected_variables": hits,
            "mapping_class": mapping_class,
            "source": source,
            "rule": spec["rule"],
        }
    return selected


def qc_flag(variable: str, mapping_class: str, selected_vars: list[str], year: str) -> str:
    if variable == "age":
        return "AGE_19_NOT_EXACT"
    if mapping_class == "Not available":
        return "LIMITATION"
    if variable in {"drink_amount", "binge_drinking"} and mapping_class == "Proxy":
        return "REVIEW_PROXY"
    if variable == "aerobic_activity" and mapping_class == "Proxy":
        return "REVIEW_PROXY"
    if not selected_vars:
        return "MISSING"
    return "OK"


def adult_counts(df: pd.DataFrame, selected: dict[str, dict[str, Any]]) -> dict[str, int]:
    age_vars = selected["age"]["selected_variables"]
    if not age_vars:
        return {"n_age16plus": 0, "n_age20plus_proxy": 0, "n_age19_exact": 0}
    age_col = age_vars[0]
    age = clean_numeric(df[age_col])
    if age_col.lower() == "age16g5":
        return {
            "n_age16plus": int((age >= 1).sum()),
            "n_age20plus_proxy": int((age >= 3).sum()),
            "n_age19_exact": -1,
        }
    return {
        "n_age16plus": int((age >= 16).sum()),
        "n_age20plus_proxy": int((age >= 20).sum()),
        "n_age19_exact": int((age >= 19).sum()),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    shown = df.loc[:, columns].copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    for col in shown.columns:
        shown[col] = shown[col].astype(str).str.replace("\n", " ", regex=False).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, separator, *body])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []

    for year, path in HSE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        labels, columns = read_metadata(path)
        df = pd.read_stata(path, convert_categoricals=False)
        selected = selected_columns_for_year(labels, columns)
        counts = adult_counts(df, selected)

        bp_vars = selected["sbp_final"]["selected_variables"] + selected["dbp_final"]["selected_variables"]
        bp_valid = 0
        if len(bp_vars) >= 2:
            bp_valid = int((clean_numeric(df[bp_vars[0]]).notna() & clean_numeric(df[bp_vars[1]]).notna()).sum())

        year_rows.append(
            {
                "year": year,
                "raw_n": len(df),
                **counts,
                "n_valid_sbp_dbp_raw": bp_valid,
                "note": "n_age19_exact=-1 means exact >=19 cannot be computed because only grouped age is released.",
                "stata_file": str(path.relative_to(ROOT)),
            }
        )

        for variable, info in selected.items():
            selected_vars = info["selected_variables"]
            role = info["role"]
            labels_joined = "; ".join([f"{v}: {labels.get(v, '')}" for v in selected_vars])
            if selected_vars:
                values = derived_values(df, variable, selected_vars)
                missing_n, missing_pct, range_summary, top_values = summarize_series(values)
            else:
                missing_n, missing_pct, range_summary, top_values = len(df), 100.0, "", ""
            flag = qc_flag(variable, info["mapping_class"], selected_vars, year)
            mapping_rows.append(
                {
                    "year": year,
                    "role": role,
                    "knhanes_variable": variable,
                    "hse_selected_variable": "; ".join(selected_vars) if selected_vars else "",
                    "hse_variable_label": labels_joined,
                    "mapping_class": info["mapping_class"],
                    "selection_source": info["source"],
                    "raw_missing_n": missing_n,
                    "raw_missing_pct": missing_pct,
                    "range_or_summary": range_summary,
                    "top_values": top_values,
                    "qc_flag": flag,
                    "harmonization_rule": info["rule"],
                }
            )
            distribution_rows.append(
                {
                    "year": year,
                    "knhanes_variable": variable,
                    "hse_selected_variable": "; ".join(selected_vars) if selected_vars else "",
                    "range_or_summary": range_summary,
                    "top_values": top_values,
                }
            )

    mapping = pd.DataFrame(mapping_rows)
    year_summary = pd.DataFrame(year_rows)
    distributions = pd.DataFrame(distribution_rows)

    variable_summary = (
        mapping.groupby("knhanes_variable", as_index=False)
        .agg(
            role=("role", "first"),
            years_with_direct_or_harmonization=(
                "mapping_class",
                lambda s: int(s.isin(["Direct", "Harmonization"]).sum()),
            ),
            years_with_proxy=("mapping_class", lambda s: int((s == "Proxy").sum())),
            years_not_available=("mapping_class", lambda s: int((s == "Not available").sum())),
            hse_selected_by_year=(
                "hse_selected_variable",
                lambda s: " | ".join([f"{yr}:{val or 'NA'}" for yr, val in zip(mapping.loc[s.index, "year"], s)]),
            ),
            qc_flags=("qc_flag", lambda s: "; ".join(sorted(set(s)))),
        )
    )

    mapping_path = REPORTS / f"hse_2018_2019_2021_2022_raw_core_sdoh_hr_mapping_qc_{STAMP}.csv"
    year_path = REPORTS / f"hse_2018_2019_2021_2022_raw_core_sdoh_hr_year_summary_{STAMP}.csv"
    distribution_path = REPORTS / f"hse_2018_2019_2021_2022_raw_core_sdoh_hr_selected_distributions_{STAMP}.csv"
    summary_path = REPORTS / f"hse_2018_2019_2021_2022_raw_core_sdoh_hr_variable_summary_{STAMP}.csv"
    mapping.to_csv(mapping_path, index=False)
    year_summary.to_csv(year_path, index=False)
    distributions.to_csv(distribution_path, index=False)
    variable_summary.to_csv(summary_path, index=False)

    blockers = variable_summary[
        (variable_summary["years_not_available"] > 0)
        | variable_summary["qc_flags"].str.contains("AGE_19_NOT_EXACT|REVIEW_PROXY|LIMITATION", regex=True)
    ].copy()

    md = []
    md.append("# HSE 2018/2019/2021/2022 Raw Mapping QC for `core_sdoh_hr`\n")
    md.append(f"Date: {STAMP[:4]}-{STAMP[4:6]}-{STAMP[6:]}\n")
    md.append("Scope: Read-only QC of local UK Data Service EUL Stata files. This script does not run external prediction, retraining, tuning, recalibration, threshold selection, or validation metrics.\n")
    md.append("## Source Files\n")
    for year, path in HSE_FILES.items():
        md.append(f"- {year}: `{path.relative_to(ROOT)}`\n")
    md.append("## Age Criterion Finding\n")
    md.append("- Requested change: use age >=19 to match the KNHANES adult criterion.\n")
    md.append("- Actual raw-file finding: none of the four EUL Stata files contains exact continuous age. All four expose grouped age, mainly `age16g5`.\n")
    md.append("- Consequence: exact age >=19 cannot be reproduced. The 16-19 band mixes ages 16, 17, 18, and 19, so including it would violate the adult-only criterion. The conservative pre-specified proxy remains `age16g5 >= 3`, corresponding to 20+.\n")
    md.append("## Year-Level Counts\n")
    md.append(markdown_table(year_summary, ["year", "raw_n", "n_age16plus", "n_age20plus_proxy", "n_age19_exact", "n_valid_sbp_dbp_raw"]))
    md.append("\n## Variable-Level Pooled Feasibility\n")
    md.append(markdown_table(variable_summary, ["knhanes_variable", "role", "years_with_direct_or_harmonization", "years_with_proxy", "years_not_available", "qc_flags"], max_rows=None))
    md.append("\n## Main QC Limitations / Review Items\n")
    if blockers.empty:
        md.append("- No blockers or review items detected.\n")
    else:
        md.append(markdown_table(blockers, ["knhanes_variable", "role", "years_with_direct_or_harmonization", "years_with_proxy", "years_not_available", "qc_flags"], max_rows=None))
    md.append("\n## Key Interpretation\n")
    md.append("- HSE 2018, 2019, 2021, and 2022 can be read as participant-level EUL raw microdata, but exact age >=19 is not available in the released Stata files.\n")
    md.append("- `family_history_htn` and `food_security_status` remain unavailable in all four years; they would need to remain fully missing if the frozen KNHANES model is applied without changing predictors.\n")
    md.append("- `drink_amount` and `binge_drinking` are not consistently measured using the 2022 AUDIT variables across all years. A pooled HSE analysis would require pre-specified proxy rules using last-7-day/heaviest-day alcohol variables, and this should be reported as a harmonization limitation.\n")
    md.append("- `aerobic_activity` is not consistently available: 2018/2022 have IPAQ MVPA minutes/week, 2021 has different physical-activity derived variables, and 2019 lacks a clear adult MVPA variable in this QC.\n")
    md.append("- 2021 BMI does not have `BMIVal`; measured BMI can be harmonized from final measured height/weight if sufficient non-missing values exist, otherwise self-reported adjusted BMI would be a weaker proxy.\n")
    md.append("## Output Files\n")
    md.append(f"- Mapping QC CSV: `{mapping_path.relative_to(ROOT)}`\n")
    md.append(f"- Variable summary CSV: `{summary_path.relative_to(ROOT)}`\n")
    md.append(f"- Year summary CSV: `{year_path.relative_to(ROOT)}`\n")
    md.append(f"- Selected distributions CSV: `{distribution_path.relative_to(ROOT)}`\n")

    md_path = REPORTS / f"hse_2018_2019_2021_2022_raw_core_sdoh_hr_mapping_qc_{STAMP}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {mapping_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {year_path}")
    print(f"Wrote {distribution_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
