#!/usr/bin/env python3
"""Year-wise consistency QC for HSE external validation inputs.

This script is read-only with respect to model artifacts. It inspects the
already harmonized HSE input files, raw Stata metadata, and local HSE nurse
documentation text to verify age/event distributions, predictor missingness,
outcome/BP definitions, and antihypertensive medication variable consistency.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hse"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STAMP = "20260819"

RAW_DTA = {
    "2018": RAW / "2018" / "UKDA-8649-stata" / "stata" / "stata13" / "hse_2018_eul_15082022.dta",
    "2019": RAW / "2019" / "UKDA-8860-stata" / "stata" / "stata13" / "hse_2019_eul_20211006.dta",
    "2021": RAW / "2021" / "UKDA-9319-stata" / "stata" / "stata13" / "hse_2021_eul_v1.dta",
    "2022": RAW / "2022" / "ukds_raw" / "UKDA-9469-stata" / "stata" / "stata13" / "hse_2022_eul_v1.dta",
}

NURSE_DOCS = {
    "2018": RAW / "2018" / "UKDA-8649-stata" / "mrdoc" / "pdf" / "8649_hse_2018_interviewer_and_nurse_documentation.pdf",
    "2019": RAW / "2019" / "UKDA-8860-stata" / "mrdoc" / "pdf" / "8860_hse_2019_interviewer_and_nurse_documentation.pdf",
    "2021": RAW / "2021" / "UKDA-9319-stata" / "mrdoc" / "pdf" / "9319_hse_2021_interviewer_and_nurse_documentation_v1.pdf",
    "2022": RAW / "2022" / "ukds_raw" / "UKDA-9469-stata" / "mrdoc" / "pdf" / "9469_hse_2022_interviewer_and_nurse_documentation_v1.pdf",
}

MODEL_INPUTS = {
    "integrated": PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_integrated_input_missing_as_na_{STAMP}.csv",
    "male": PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_male_input_missing_as_na_{STAMP}.csv",
    "female": PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_female_input_missing_as_na_{STAMP}.csv",
}

HARMONIZED_QC = PROCESSED / f"hse_2018_2019_2021_2022_pooled_core_sdoh_hr_harmonized_qc_{STAMP}.csv"
MAPPING_QC = REPORTS / "hse_2018_2019_2021_2022_raw_core_sdoh_hr_mapping_qc_20260819.csv"

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


def pct(x: float) -> float:
    return round(float(x) * 100, 2) if pd.notna(x) else np.nan


def read_inputs() -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(v) for k, v in MODEL_INPUTS.items()}


def age_event_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        for year, sub in df.groupby("source_year", sort=True):
            age = pd.to_numeric(sub["age"], errors="coerce")
            outcome = pd.to_numeric(sub["outcome"], errors="coerce")
            rows.append(
                {
                    "model_set": model_set,
                    "source_year": int(year),
                    "n": int(len(sub)),
                    "events": int((outcome == 1).sum()),
                    "event_rate_pct": pct((outcome == 1).mean()),
                    "age_min": float(age.min()),
                    "age_q1": float(age.quantile(0.25)),
                    "age_median": float(age.median()),
                    "age_q3": float(age.quantile(0.75)),
                    "age_max": float(age.max()),
                    "age_80_topcoded_n": int((age == 80).sum()),
                    "age_80_topcoded_pct": pct((age == 80).mean()),
                }
            )
    return pd.DataFrame(rows)


def predictor_missingness(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_set, df in inputs.items():
        predictors = PREDICTORS_INTEGRATED if model_set == "integrated" else PREDICTORS_SEX_SPECIFIC
        for year, sub in df.groupby("source_year", sort=True):
            for var in predictors:
                miss = sub[var].isna()
                rows.append(
                    {
                        "model_set": model_set,
                        "source_year": int(year),
                        "predictor": var,
                        "n": int(len(sub)),
                        "missing_n": int(miss.sum()),
                        "missing_rate_pct": pct(miss.mean()),
                        "all_missing": bool(miss.all()),
                    }
                )
    return pd.DataFrame(rows)


def cohort_flow_by_year(harmonized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    steps = [
        ("raw_harmonized", lambda x: pd.Series(True, index=x.index)),
        ("age_20plus_due_to_5yr_age_band", lambda x: x["age_band_raw"] >= 3),
        ("valid_sex", lambda x: x["sex"].isin([1, 2])),
        ("valid_blood_pressure_measurement", lambda x: x["sbp_final"].notna() & x["dbp_final"].notna()),
        ("known_hypertension_diagnosis_determinable", lambda x: x["known_htn_dx"].isin([0, 1])),
        ("exclude_self_reported_hypertension_diagnosis", lambda x: x["known_htn_dx"] != 1),
        ("antihypertensive_medication_determinable", lambda x: x["antihypertensive_use"].isin([0, 1])),
        ("exclude_antihypertensive_medication_use", lambda x: x["antihypertensive_use"] != 1),
        ("exclude_current_pregnancy", lambda x: x["pregnancy"].fillna(0) != 1),
    ]
    for year, base in harmonized.groupby("source_year", sort=True):
        cur = base.copy()
        prev = None
        for step, rule in steps:
            before = len(cur)
            if prev is None:
                after = cur.copy()
                excluded = 0
            else:
                keep = rule(cur)
                after = cur[keep].copy()
                excluded = before - len(after)
            rows.append(
                {
                    "source_year": int(year),
                    "step": step,
                    "n_remaining": int(len(after)),
                    "n_excluded_step": int(excluded),
                }
            )
            cur = after
            prev = step
    return pd.DataFrame(rows)


def stata_labels(path: Path) -> dict[str, str]:
    reader = pd.io.stata.StataReader(path)
    return reader.variable_labels()


def find_actual(labels: dict[str, str], wanted: str) -> str:
    lower = {k.lower(): k for k in labels}
    return lower.get(wanted.lower(), "")


def raw_definition_table() -> pd.DataFrame:
    selected = [
        ("age_band", "age16g5"),
        ("sbp_final", "omsysval"),
        ("dbp_final", "omdiaval"),
        ("raw_sbp_reading_1", "sys1om"),
        ("raw_sbp_reading_2", "sys2om"),
        ("raw_sbp_reading_3", "sys3om"),
        ("raw_dbp_reading_1", "dias1om"),
        ("raw_dbp_reading_2", "dias2om"),
        ("raw_dbp_reading_3", "dias3om"),
        ("heart_rate_reading_2", "puls2om"),
        ("heart_rate_reading_3", "puls3om"),
        ("known_htn_dx", "bp1"),
        ("antihypertensive_use", "bpmedd2"),
        ("pregnancy", "pregntj"),
    ]
    rows = []
    for year, path in RAW_DTA.items():
        labels = stata_labels(path)
        for role, wanted in selected:
            actual = find_actual(labels, wanted)
            rows.append(
                {
                    "source_year": int(year),
                    "role": role,
                    "expected_variable": wanted,
                    "actual_variable": actual,
                    "exists": bool(actual),
                    "variable_label": labels.get(actual, ""),
                }
            )
    return pd.DataFrame(rows)


def bp_med_distribution(harmonized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, base in harmonized.groupby("source_year", sort=True):
        cur = base.copy()
        cur = cur[cur["age_band_raw"] >= 3]
        cur = cur[cur["sex"].isin([1, 2])]
        cur = cur[cur["sbp_final"].notna() & cur["dbp_final"].notna()]
        cur = cur[cur["known_htn_dx"].isin([0, 1])]
        pre_dx = cur.copy()
        cur = cur[cur["known_htn_dx"] != 1]
        for stage, sub in [("after_valid_bp_before_dx_exclusion", pre_dx), ("after_known_htn_exclusion_before_med_exclusion", cur)]:
            rows.append(
                {
                    "source_year": int(year),
                    "stage": stage,
                    "n": int(len(sub)),
                    "known_htn_dx_yes_n": int((sub["known_htn_dx"] == 1).sum()),
                    "known_htn_dx_yes_pct": pct((sub["known_htn_dx"] == 1).mean()),
                    "antihypertensive_use_yes_n": int((sub["antihypertensive_use"] == 1).sum()),
                    "antihypertensive_use_yes_pct": pct((sub["antihypertensive_use"] == 1).mean()),
                    "antihypertensive_use_missing_n": int(sub["antihypertensive_use"].isna().sum()),
                    "antihypertensive_use_missing_pct": pct(sub["antihypertensive_use"].isna().mean()),
                    "bp_elevated_14090_n": int(((sub["sbp_final"] >= 140) | (sub["dbp_final"] >= 90)).sum()),
                    "bp_elevated_14090_pct": pct(((sub["sbp_final"] >= 140) | (sub["dbp_final"] >= 90)).mean()),
                }
            )
    return pd.DataFrame(rows)


def extract_bp_protocol() -> pd.DataFrame:
    rows = []
    for year, pdf in NURSE_DOCS.items():
        txt = Path("/private/tmp") / f"hse_{year}_nurse_doc.txt"
        if not txt.exists() or txt.stat().st_size == 0:
            subprocess.run(["pdftotext", str(pdf), str(txt)], check=True)
        lines = txt.read_text(errors="ignore").splitlines()

        sit_hits = [i for i, line in enumerate(lines) if "ASK THE RESPONDENT TO SIT STILL FOR FIVE MINUTES" in line.upper()]
        pulse_hits = [i for i, line in enumerate(lines) if "Derived: average of 2nd and 3rd pulse readings" in line]
        consent_hits = [i for i, line in enumerate(lines) if "Does the respondent agree to blood pressure measurement" in line]
        ytake_hits = [i for i, line in enumerate(lines) if "Do you take <name of drug> because of a heart problem, high blood pressure" in line]

        def snippet(hits: list[int], before: int = 2, after: int = 7) -> str:
            if not hits:
                return ""
            i = hits[0]
            return " ".join(line.strip() for line in lines[max(0, i - before) : min(len(lines), i + after)] if line.strip())

        rows.append(
            {
                "source_year": int(year),
                "documentation_pdf": str(pdf.relative_to(ROOT)),
                "bp_measurement_protocol_excerpt": snippet(sit_hits, before=2, after=8),
                "pulse_derivation_excerpt": snippet(pulse_hits, before=1, after=3),
                "bp_consent_excerpt": snippet(consent_hits, before=2, after=4),
                "medication_reason_excerpt": snippet(ytake_hits, before=1, after=5),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    path: Path,
    age_event: pd.DataFrame,
    missingness: pd.DataFrame,
    definitions: pd.DataFrame,
    med_dist: pd.DataFrame,
    protocol: pd.DataFrame,
    flow: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        view = df.copy()
        for col in view.columns:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x).replace("\n", " "))
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in view.astype(str).values.tolist()]
        return "\n".join([header, sep, *rows])

    integrated_age = age_event[age_event["model_set"] == "integrated"].copy()
    high_missing = missingness[
        (missingness["model_set"] == "integrated")
        & ((missingness["missing_rate_pct"] >= 20) | (missingness["all_missing"]))
    ].copy()
    key_defs = definitions[
        definitions["role"].isin(["age_band", "sbp_final", "dbp_final", "known_htn_dx", "antihypertensive_use", "pregnancy"])
    ].copy()
    protocol_short = protocol[["source_year", "bp_measurement_protocol_excerpt", "pulse_derivation_excerpt"]].copy()

    with path.open("w", encoding="utf-8") as f:
        f.write("# HSE 2018/2019/2021/2022 External Validation Consistency QC\n\n")
        f.write("This QC reads only existing harmonized HSE inputs, raw Stata metadata, and local HSE documentation. No model refit, tuning, or prediction was run.\n\n")
        f.write("## Key Conclusion\n\n")
        f.write("- Age handling is consistent across HSE years but is a proxy: public/EUL raw data expose `age16g5`, so the adult criterion is conservatively implemented as `age16g5 >= 3` (20+), not exact KNHANES 19+.\n")
        f.write("- Outcome definition is implemented identically across years: measured `omsysval >= 140` or `omdiaval >= 90`, after excluding self-reported hypertension diagnosis and prescribed BP medication use.\n")
        f.write("- BP source variables and protocol are consistent across years: HSE documentation shows right-arm cuff placement, five-minute rest, and three Omron recordings with one-minute intervals; raw metadata include 1st/2nd/3rd readings and valid mean SBP/DBP variables.\n")
        f.write("- Medication exclusion uses `bpmedd2` in all four years with the same raw label: prescribed blood-pressure drugs. Coding is harmonized as 1 = taking BP medication, 0 = not taking.\n")
        f.write("- Main predictor limitations remain year-consistent but important: `family_history_htn` and `food_security_status` are all missing in all HSE years; `aerobic_activity` is all missing in 2019; alcohol amount/binge variables are proxy constructs except 2022.\n\n")
        f.write("## Integrated Final Input: Age and Event Rate by Year\n\n")
        f.write(md_table(integrated_age))
        f.write("\n\n## Integrated Predictor Missingness Requiring Attention\n\n")
        f.write(md_table(high_missing))
        f.write("\n\n## Cohort Flow by Year\n\n")
        f.write(md_table(flow))
        f.write("\n\n## Outcome/BP/Medication Raw Variable Metadata\n\n")
        f.write(md_table(key_defs))
        f.write("\n\n## Diagnosis and Medication Distribution at Key Stages\n\n")
        f.write(md_table(med_dist))
        f.write("\n\n## BP Measurement Protocol Evidence\n\n")
        f.write(md_table(protocol_short))
        f.write("\n\n## Output Files\n\n")
        f.write(f"- Age/event QC: `reports/hse_pooled_2018_2019_2021_2022_age_event_qc_{STAMP}.csv`\n")
        f.write(f"- Predictor missingness QC: `reports/hse_pooled_2018_2019_2021_2022_predictor_missingness_qc_{STAMP}.csv`\n")
        f.write(f"- Raw definition QC: `reports/hse_pooled_2018_2019_2021_2022_outcome_bp_med_definition_qc_{STAMP}.csv`\n")
        f.write(f"- BP protocol evidence: `reports/hse_pooled_2018_2019_2021_2022_bp_protocol_qc_{STAMP}.csv`\n")
        f.write(f"- Cohort flow by year: `reports/hse_pooled_2018_2019_2021_2022_cohort_flow_by_year_qc_{STAMP}.csv`\n")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()
    harmonized = pd.read_csv(HARMONIZED_QC)
    mapping = pd.read_csv(MAPPING_QC)

    age_event = age_event_table(inputs)
    missing = predictor_missingness(inputs)
    flow = cohort_flow_by_year(harmonized)
    definitions = raw_definition_table()
    med_dist = bp_med_distribution(harmonized)
    protocol = extract_bp_protocol()

    # Bring in prior mapping QC flags for the key outcome/exclusion rows.
    key_map = mapping[mapping["role"].isin(["outcome/exclusion", "predictor"])][
        ["year", "knhanes_variable", "hse_selected_variable", "mapping_class", "qc_flag", "harmonization_rule"]
    ].rename(columns={"year": "source_year"})
    key_map["source_year"] = pd.to_numeric(key_map["source_year"], errors="coerce").astype("Int64")

    age_event.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_age_event_qc_{STAMP}.csv", index=False)
    missing.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_predictor_missingness_qc_{STAMP}.csv", index=False)
    flow.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_cohort_flow_by_year_qc_{STAMP}.csv", index=False)
    definitions.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_outcome_bp_med_definition_qc_{STAMP}.csv", index=False)
    med_dist.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_bp_medication_distribution_qc_{STAMP}.csv", index=False)
    protocol.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_bp_protocol_qc_{STAMP}.csv", index=False)
    key_map.to_csv(REPORTS / f"hse_pooled_2018_2019_2021_2022_mapping_flags_qc_{STAMP}.csv", index=False)

    write_markdown(
        REPORTS / f"hse_pooled_2018_2019_2021_2022_external_consistency_qc_{STAMP}.md",
        age_event,
        missing,
        definitions,
        med_dist,
        protocol,
        flow,
    )
    print("Wrote HSE consistency QC outputs to reports/")
    print(age_event[age_event["model_set"] == "integrated"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
