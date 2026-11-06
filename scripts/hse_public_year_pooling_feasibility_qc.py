#!/usr/bin/env python3
"""Compare public HSE data dictionaries for pooling feasibility.

This is a documentation/codebook QC helper only. It does not build an
external-validation dataset and does not run prediction.
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STAMP = "20260817"


@dataclass
class VarInfo:
    name: str
    label: str


HSE_2018_RTF = ROOT / "data/raw/hse/2018/ukds_docs/hse_2018_eul_15082022_ukda_data_dictionary.rtf"
HSE_2019_RTF = ROOT / "data/raw/hse/2019/ukds_docs/hse_2019_eul_20211006_ukda_data_dictionary.rtf"
HSE_2021_RTF = ROOT / "data/raw/hse/2021/ukds_docs/hse_2021_eul_v1_ukda_data_dictionary.rtf"
HSE_2022_XLSX = ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata/mrdoc/excel/9469_hse_2022_eul_variable_information_v1.xlsx"


CORE_SDOH_HR_MAPPING = [
    {
        "role": "id",
        "knhanes_variable": "person_id",
        "hse_2022_selected": ["seriala"],
        "candidate_names": ["seriala", "SerialA", "Seriala"],
        "notes": "Participant identifier only.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "age",
        "hse_2022_selected": ["age16g5"],
        "candidate_names": ["age16g5", "Age16g5", "age16g10", "age16g7"],
        "notes": "Age is 5-year bands in HSE EUL; KNHANES exact 19+ is approximated as 20+.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "sex",
        "hse_2022_selected": ["Sex"],
        "candidate_names": ["Sex", "sex"],
        "notes": "Omit from sex-specific models.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "bmi",
        "hse_2022_selected": ["bmival"],
        "candidate_names": ["bmival", "BMIVal", "bmi"],
        "notes": "Measured BMI kg/m2.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "waist_cm",
        "hse_2022_selected": ["wstval"],
        "candidate_names": ["wstval", "waist", "waistval"],
        "notes": "Measured waist circumference cm.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "heart_rate",
        "hse_2022_selected": ["PULS2OM", "PULS3OM"],
        "candidate_names": ["PULS2OM", "PULS3OM", "puls2om", "puls3om", "pulse2om", "pulse3om"],
        "notes": "Use mean of 2nd and 3rd Omron pulse readings, not pulse-pressure variables.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "sbp_final",
        "hse_2022_selected": ["omsysval"],
        "candidate_names": ["omsysval", "OMSysVal", "sysval", "systolic"],
        "notes": "Outcome only.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "dbp_final",
        "hse_2022_selected": ["omdiaval"],
        "candidate_names": ["omdiaval", "OMDiaVal", "diaval", "diastolic"],
        "notes": "Outcome only.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "known_htn_dx",
        "hse_2022_selected": ["bp1"],
        "candidate_names": ["bp1", "EverBP"],
        "notes": "Doctor/nurse diagnosed high BP excluding pregnancy-only hypertension.",
    },
    {
        "role": "outcome/exclusion",
        "knhanes_variable": "antihypertensive_use",
        "hse_2022_selected": ["bpmedd2"],
        "candidate_names": ["bpmedd2", "bpmedd", "hypertakg2", "hypertakg"],
        "notes": "Prescribed BP medication.",
    },
    {
        "role": "exclusion",
        "knhanes_variable": "pregnancy",
        "hse_2022_selected": ["PregNTJ"],
        "candidate_names": ["PregNTJ", "PregNT", "pregntj", "pregnt"],
        "notes": "Current pregnancy exclusion.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "diabetes_dx",
        "hse_2022_selected": ["diabete2"],
        "candidate_names": ["diabete2", "diabete", "diabetes"],
        "notes": "Doctor-diagnosed diabetes.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "aerobic_activity",
        "hse_2022_selected": ["mvpamwk"],
        "candidate_names": ["mvpamwk", "mvpamwkg", "mvpa", "mvpamwk2"],
        "notes": "Prefer continuous weekly MVPA minutes where present; derive >=150 min/week.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "self_rated_health",
        "hse_2022_selected": ["GenHelf"],
        "candidate_names": ["GenHelf", "genhelf", "Genhelf2", "genhelf2"],
        "notes": "Ordered self-rated general health.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "activity_limitation",
        "hse_2022_selected": ["limitill"],
        "candidate_names": ["limitill", "LimitIll", "ReducAct", "reducact"],
        "notes": "Actual HSE 2022 input-build script used limitill.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "family_history_htn",
        "hse_2022_selected": [],
        "candidate_names": ["family_history_htn", "parentbp", "bpfa", "bpmo", "hypertension family"],
        "notes": "Not found in HSE 2022; do not substitute own hypertension or medication.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "current_smoking",
        "hse_2022_selected": ["cigsta3_19"],
        "candidate_names": ["cigsta3_19", "cigsta3", "cigst2", "cigsta"],
        "notes": "Current cigarette smoker vs non-current.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "alcohol_frequency",
        "hse_2022_selected": ["dnoft_22"],
        "candidate_names": ["dnoft_22", "dnoft3", "dnoft", "drinkfreq"],
        "notes": "Past-year drinking frequency.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "drink_amount",
        "hse_2022_selected": ["AUDTyDy"],
        "candidate_names": ["AUDTyDy", "audtydy", "AUDITdy", "typday"],
        "notes": "Typical number of drinks on a drinking day.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "binge_drinking",
        "hse_2022_selected": ["AUD6Mre"],
        "candidate_names": ["AUD6Mre", "aud6mre", "sixmore", "alclimit7b"],
        "notes": "AUDIT 6+ drinks frequency; threshold differs from KNHANES sex-specific binge definition.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "education_level",
        "hse_2022_selected": ["educ2"],
        "candidate_names": ["educ2", "educend", "topqual3", "topqual"],
        "notes": "Qualification-based education harmonization.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "household_income_q",
        "hse_2022_selected": ["eqv5"],
        "candidate_names": ["eqv5", "eqvinc5", "hhincome", "incomeq"],
        "notes": "Equivalised household income quintile collapsed to quartile scale.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "employment_status",
        "hse_2022_selected": ["activb2"],
        "candidate_names": ["activb2", "activb", "ecact", "econact"],
        "notes": "Employed/self-employed/unpaid family business vs other.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "marital_status",
        "hse_2022_selected": ["marstatd_22"],
        "candidate_names": ["marstatd_22", "marstatd", "marstat", "marstat2"],
        "notes": "Marital status collapsed to KNHANES categories.",
    },
    {
        "role": "predictor",
        "knhanes_variable": "food_security_status",
        "hse_2022_selected": [],
        "candidate_names": ["foodsec", "food security", "food insecurity", "foodbank", "hunger"],
        "notes": "Not found in HSE 2022; do not substitute income/benefits/diet.",
    },
]


KEYWORD_PATTERNS = {
    "age": r"\bage\b|age 16|\(d\) age",
    "bmi": r"\bbmi\b|body mass",
    "waist_cm": r"waist",
    "heart_rate": r"pulse reading|heart rate|\bpulse\b|omron pulse",
    "sbp_final": r"systolic|blood pressure",
    "dbp_final": r"diastolic|blood pressure",
    "known_htn_dx": r"high blood pressure|hypertension",
    "antihypertensive_use": r"blood pressure.*drug|drug.*blood pressure|hypertension.*med",
    "pregnancy": r"pregnan",
    "diabetes_dx": r"diabet",
    "aerobic_activity": r"mvpa|moderate.*vigorous|physical activity",
    "self_rated_health": r"general health|self.*health",
    "activity_limitation": r"limiting longstanding|reduces day-to-day|activity",
    "family_history_htn": r"family.*blood pressure|parent.*blood pressure|mother.*blood pressure|father.*blood pressure|hypertension family",
    "current_smoking": r"current.*smok|cigarette smoker|regular cigarette",
    "alcohol_frequency": r"drink.*frequency|how often.*drink|days.*alcohol",
    "drink_amount": r"typical.*drink|number.*drinks|drinking day",
    "binge_drinking": r"six.*drinks|6.*drinks|binge|one occasion",
    "education_level": r"education|qualification|degree",
    "household_income_q": r"income|equivalised",
    "employment_status": r"paid employment|self-employed|activity status|economic",
    "marital_status": r"marital|civil partnership|cohabit",
    "food_security_status": r"food security|food insecurity|food bank|hunger|ran out of food",
}


RESPONDENT_ONLY_VARIABLES = {
    "bmi",
    "waist_cm",
    "heart_rate",
    "sbp_final",
    "dbp_final",
    "known_htn_dx",
    "antihypertensive_use",
    "pregnancy",
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
    "employment_status",
}

NON_RESPONDENT_RE = re.compile(
    r"parent/guardian|father|mother|partner|child|children|8-15|aged 0-15|"
    r"\bmoth_|\bfath_|\bpart_|\bkcig|\bp[123]_",
    re.IGNORECASE,
)


def rtf_to_text(path: Path) -> str:
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_text_dictionary(text: str) -> dict[str, VarInfo]:
    pattern = re.compile(
        r"Pos\. =\s*\d+\s+Variable =\s*([^\t\n]+)\s+Variable label =\s*(.*?)(?=\nThis variable is)",
        re.IGNORECASE | re.DOTALL,
    )
    variables: dict[str, VarInfo] = {}
    for m in pattern.finditer(text):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        label = re.sub(r"\s+", " ", m.group(2)).strip()
        variables[name.lower()] = VarInfo(name=name, label=label)
    return variables


def parse_2022_excel(path: Path) -> dict[str, VarInfo]:
    df = pd.read_excel(path, sheet_name="Variables")
    variables: dict[str, VarInfo] = {}
    for _, row in df.iterrows():
        name = str(row.get("Variable", "")).strip()
        if not name or name.lower() == "nan":
            continue
        label = str(row.get("Description", "")).strip()
        variables[name.lower()] = VarInfo(name=name, label=label)
    return variables


def exact_matches(varmap: dict[str, VarInfo], candidate_names: list[str]) -> list[VarInfo]:
    hits: list[VarInfo] = []
    seen = set()
    for cand in candidate_names:
        key = cand.lower()
        if key in varmap and key not in seen:
            hits.append(varmap[key])
            seen.add(key)
    return hits


def keyword_candidates(varmap: dict[str, VarInfo], knhanes_variable: str, limit: int = 8) -> list[VarInfo]:
    pat = KEYWORD_PATTERNS.get(knhanes_variable, knhanes_variable)
    rx = re.compile(pat, re.IGNORECASE)
    hits = []
    for info in varmap.values():
        haystack = f"{info.name} {info.label}"
        if knhanes_variable in RESPONDENT_ONLY_VARIABLES and NON_RESPONDENT_RE.search(haystack):
            continue
        if rx.search(haystack):
            hits.append(info)
    hits.sort(key=lambda x: (0 if knhanes_variable.lower() in x.name.lower() else 1, x.name.lower()))
    return hits[:limit]


def join_hits(hits: list[VarInfo]) -> str:
    if not hits:
        return ""
    return "; ".join(f"{h.name}: {h.label}" for h in hits)


def availability_status(required_count: int, hit_count: int, knhanes_variable: str) -> str:
    if knhanes_variable in {"family_history_htn", "food_security_status"}:
        return "not_available_expected"
    if required_count == 0:
        return "not_required"
    if hit_count >= required_count:
        return "exact_all_required"
    if hit_count > 0:
        return "partial_exact"
    return "no_exact_match"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    maps = {
        "2018_public_dictionary": parse_text_dictionary(rtf_to_text(HSE_2018_RTF)),
        "2019_public_dictionary": parse_text_dictionary(rtf_to_text(HSE_2019_RTF)),
        "2021_public_dictionary": parse_text_dictionary(rtf_to_text(HSE_2021_RTF)),
        "2022_raw_codebook": parse_2022_excel(HSE_2022_XLSX),
    }

    rows = []
    for spec in CORE_SDOH_HR_MAPPING:
        for year_key, varmap in maps.items():
            exact = exact_matches(varmap, spec["candidate_names"])
            selected_hits = exact_matches(varmap, spec["hse_2022_selected"])
            candidates = keyword_candidates(varmap, spec["knhanes_variable"])
            rows.append(
                {
                    "year_source": year_key,
                    "role": spec["role"],
                    "knhanes_variable": spec["knhanes_variable"],
                    "hse_2022_selected_variable": "; ".join(spec["hse_2022_selected"]) or "not available",
                    "status": availability_status(len(spec["hse_2022_selected"]), len(selected_hits), spec["knhanes_variable"]),
                    "selected_exact_found": join_hits(selected_hits),
                    "candidate_exact_found": join_hits(exact),
                    "keyword_candidates": join_hits(candidates),
                    "notes": spec["notes"],
                }
            )

    csv_path = REPORTS / f"hse_2018_2019_2021_2022_core_sdoh_hr_public_dict_mapping_qc_{STAMP}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = REPORTS / f"hse_2018_2019_2021_2022_pooling_feasibility_{STAMP}.md"
    lines = []
    lines.append("# HSE 2018/2019/2021/2022 `core_sdoh_hr` Pooling Feasibility QC\n")
    lines.append(f"Date: {STAMP}\n")
    lines.append("\n## Scope\n")
    lines.append("- This is a read-only mapping feasibility check.\n")
    lines.append("- HSE 2018, 2019, and 2021 are checked using public UKDS data dictionary RTF files, not participant-level raw data.\n")
    lines.append("- HSE 2022 is checked using the local UKDS raw-codebook Excel used in the prior 2022 mapping.\n")
    lines.append("- HSE 2021 participant-level raw data is not locally available, so 2021 value labels, missingness, and final cohort construction are not yet QCed.\n")
    lines.append("- No external prediction, validation metric, retraining, tuning, threshold selection, or frozen-model modification was run.\n")
    lines.append("\n## Download Status\n")
    lines.append("- HSE 2018 participant-level raw data: not downloaded; UKDS account download page returned an unknown-error state.\n")
    lines.append("- HSE 2019 participant-level raw data: not downloaded; UKDS account download page returned an unknown-error state.\n")
    lines.append("- HSE 2021 participant-level raw data: not downloaded in this workspace; public UKDS dictionary only.\n")
    lines.append("- HSE 2018 public data dictionary: downloaded and parsed.\n")
    lines.append("- HSE 2019 public data dictionary: downloaded and parsed.\n")
    lines.append("- HSE 2021 public data dictionary: downloaded and parsed.\n")
    lines.append("- HSE 2022 participant-level raw/codebook: already available locally.\n")
    lines.append("\n## High-Level Judgment\n")
    lines.append("- HSE 2018 and 2019 do not show the same measurement/outcome variable set as HSE 2022 in the public dictionaries; they may not be suitable for the frozen hypertension outcome without raw-file confirmation.\n")
    lines.append("- HSE 2021 appears closer to HSE 2022 for clinical measurements based on public documentation context, but participant-level raw/codebook QC is still required before pooling.\n")
    lines.append("- The biggest expected recurring limitations remain exact age, `family_history_htn`, and `food_security_status`.\n")
    lines.append("- Suffix/version differences are expected for smoking, alcohol frequency, marital status, and some derived variables; these need year-specific harmonization rules rather than assuming a single variable name across years.\n")
    lines.append("- Activity limitation should be checked against `limitill` because the final HSE 2022 input-generation code used `limitill`.\n")
    lines.append("- HSE 2021 still needs separate UKDS raw download/documentation QC before pooling.\n")
    lines.append("\n## Pooling Decision Before Raw Download\n")
    lines.append("- HSE 2018: not recommended for pooled frozen hypertension external validation based on the public dictionary, because measured SBP/DBP outcome variables were not found. Self-reported `bp1` cannot replace measured BP outcome.\n")
    lines.append("- HSE 2019: not recommended for pooled frozen hypertension external validation based on the public dictionary, because measured SBP/DBP outcome variables were not found and many adult anthropometric/behavior variables are absent or module-specific.\n")
    lines.append("- HSE 2021: potentially closer to 2022 because measured Omron SBP/DBP and pulse readings are present in the public dictionary, but BMI/waist and several behavior/SDOH variables were not confirmed in the public dictionary; raw participant-level QC is required before deciding pooling.\n")
    lines.append("- HSE 2022: already mapped using participant-level raw/codebook and remains the only HSE year locally ready for external validation input construction.\n")
    lines.append("- Practical recommendation: do not pool 2018/2019 with 2022 for the current frozen measured-BP outcome unless the actual raw files unexpectedly contain measured BP variables absent from the public dictionaries. Prioritize HSE 2021 raw download/QC if a larger HSE external cohort is needed.\n")
    lines.append("\n## Variable Availability Summary\n")
    lines.append("| KNHANES variable | 2018 public dictionary | 2019 public dictionary | 2021 public dictionary | 2022 raw codebook | pooling note |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for spec in CORE_SDOH_HR_MAPPING:
        cells = []
        for year_key in ["2018_public_dictionary", "2019_public_dictionary", "2021_public_dictionary", "2022_raw_codebook"]:
            r = next(
                row
                for row in rows
                if row["year_source"] == year_key and row["knhanes_variable"] == spec["knhanes_variable"]
            )
            if r["selected_exact_found"]:
                cell = r["selected_exact_found"]
            elif r["candidate_exact_found"]:
                cell = f"candidate: {r['candidate_exact_found']}"
            elif r["keyword_candidates"]:
                first = r["keyword_candidates"].split("; ")[0]
                cell = f"keyword candidate: {first}"
            else:
                cell = "not found"
            cell = cell.replace("|", "/")
            cells.append(cell)
        note = spec["notes"].replace("|", "/")
        lines.append(f"| {spec['knhanes_variable']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {note} |\n")
    lines.append("\n## Files Written\n")
    lines.append(f"- CSV detail table: `{csv_path}`\n")
    lines.append(f"- Markdown report: `{md_path}`\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
