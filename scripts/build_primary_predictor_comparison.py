from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE_TAG = "20260812"
TMP_DIR = Path("/private/tmp/knhanes_predictor_comparison_20260812")
REPORTS = ROOT / "reports"


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ast_list_assignments(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            out[name] = list(value)
    return out


def unique(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def without_sex(seq: list[str]) -> list[str]:
    return [x for x in seq if x != "sex"]


def build_sets() -> dict[str, list[str]]:
    lists = ast_list_assignments(ROOT / "scripts" / "knhanes_primary_ml_pipeline.py")
    core_cont = lists["CORE_CONTINUOUS"]
    core_cat = lists["CORE_CATEGORICAL_ALL"]
    sdoh = lists["SDOH_CATEGORICAL"]
    enhanced_cons_cont = lists["ENHANCED_CONSERVATIVE_CONTINUOUS"]
    enhanced_cons_cat = lists["ENHANCED_CONSERVATIVE_CATEGORICAL"]
    v2_cont = lists["ENHANCED_V2_CONTINUOUS"]
    v2_cat = lists["ENHANCED_V2_CATEGORICAL"]
    v2_female = lists["ENHANCED_V2_FEMALE_ONLY_CATEGORICAL"]
    v3_cont = lists["ENHANCED_V3_CONTINUOUS"]
    v3_cat = lists["ENHANCED_V3_CATEGORICAL"]
    v3_female_cont = lists["ENHANCED_V3_FEMALE_ONLY_CONTINUOUS"]
    v3_female_cat = lists["ENHANCED_V3_FEMALE_ONLY_CATEGORICAL"]

    lean_core_cat = [x for x in core_cat if x != "family_history_htn"]
    lean_cont = lists["NHANES_LEAN23_CONTINUOUS"]

    sets = {
        "primary_core_integrated": unique(core_cont + core_cat),
        "primary_core_sdoh_integrated": unique(core_cont + core_cat + sdoh),
        "primary_core_sdoh_hr_integrated": unique(core_cont + ["heart_rate"] + core_cat + sdoh),
        "primary_core_sdoh_hr_sex_specific": without_sex(unique(core_cont + ["heart_rate"] + core_cat + sdoh)),
        "enhanced_conservative_integrated": unique(
            core_cont + enhanced_cons_cont + ["heart_rate"] + core_cat + sdoh + enhanced_cons_cat
        ),
        "nhanes_lean23_integrated": unique(
            core_cont + lean_cont + ["heart_rate"] + lean_core_cat + sdoh + enhanced_cons_cat
        ),
        "nhanes_lean23_sex_specific": without_sex(
            unique(core_cont + lean_cont + ["heart_rate"] + lean_core_cat + sdoh + enhanced_cons_cat)
        ),
        "full25_discussed_integrated": unique(
            core_cont + lean_cont + ["heart_rate"] + lean_core_cat + sdoh + enhanced_cons_cat + ["height_cm", "weight_kg"]
        ),
        "full25_discussed_sex_specific": without_sex(
            unique(core_cont + lean_cont + ["heart_rate"] + lean_core_cat + sdoh + enhanced_cons_cat + ["height_cm", "weight_kg"])
        ),
        "enhanced_v2_integrated": unique(core_cont + v2_cont + ["heart_rate"] + core_cat + sdoh + v2_cat),
        "enhanced_v2_female": unique(without_sex(core_cont + v2_cont + ["heart_rate"] + core_cat + sdoh + v2_cat) + v2_female),
        "enhanced_v3_integrated": unique(core_cont + v2_cont + v3_cont + ["heart_rate"] + core_cat + sdoh + v2_cat + v3_cat),
        "enhanced_v3_female": unique(
            without_sex(core_cont + v2_cont + v3_cont + ["heart_rate"] + core_cat + sdoh + v2_cat + v3_cat)
            + v2_female
            + v3_female_cont
            + v3_female_cat
        ),
    }
    return sets


def variable_metadata() -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    primary = read_csv("reports/knhanes_variable_mapping_primary.csv")
    for _, r in primary.iterrows():
        var = str(r.get("final_variable", ""))
        if not var:
            continue
        meta[var] = {
            "description": str(r.get("description", "")),
            "source_or_derivation": str(r.get("raw_variable", "")),
            "coding": str(r.get("coding_or_derivation", "")),
            "source_doc": "knhanes_variable_mapping_primary.csv",
        }

    v2 = read_csv("reports/knhanes_enhanced_v2_variable_mapping_20260810.csv")
    for _, r in v2.iterrows():
        var = str(r.get("predictor", ""))
        if not var:
            continue
        meta.setdefault(var, {})
        meta[var].update(
            {
                "description": meta[var].get("description") or str(r.get("rationale", "")),
                "source_or_derivation": meta[var].get("source_or_derivation") or str(r.get("source_or_derivation", "")),
                "coding": meta[var].get("coding") or str(r.get("decision", "")),
                "source_doc": meta[var].get("source_doc") or "knhanes_enhanced_v2_variable_mapping_20260810.csv",
            }
        )

    v3 = read_csv("reports/knhanes_enhanced_v3_current_predictor_codebook_20260811.csv")
    if not v3.empty:
        for _, r in v3.iterrows():
            var = str(r.get("최종 predictor명", ""))
            if not var:
                continue
            meta.setdefault(var, {})
            meta[var].update(
                {
                    "description": meta[var].get("description") or str(r.get("변수 내용", "")),
                    "source_or_derivation": meta[var].get("source_or_derivation") or str(r.get("원자료 변수명 / 파생식", "")),
                    "coding": meta[var].get("coding") or str(r.get("값/단위 및 재코딩", "")),
                    "source_doc": meta[var].get("source_doc") or "knhanes_enhanced_v3_current_predictor_codebook_20260811.csv",
                }
            )
    return meta


def missingness_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    files = [
        ("primary", "reports/knhanes_development_missingness_final.csv"),
        ("conservative", "reports/knhanes_conservative_enhanced_missingness_2014_2020.csv"),
        ("v2", "reports/knhanes_enhanced_v2_missingness_2014_2020.csv"),
        ("v3", "reports/knhanes_enhanced_v3_missingness_2014_2020.csv"),
    ]
    for label, rel in files:
        df = read_csv(rel)
        if df.empty:
            continue
        if "model_set" in df.columns:
            df = df[df["model_set"].eq("integrated")]
        for _, r in df.iterrows():
            var = str(r["variable"])
            out.setdefault(var, {})
            out[var][f"{label}_missing_n"] = str(int(r["missing_n"]))
            out[var][f"{label}_missing_percent"] = f"{float(r['missing_percent']):.4f}"
    return out


def nhanes_map() -> dict[str, dict[str, str]]:
    df = read_csv("reports/nhanes_external_mapping_qc_20260811.csv")
    out: dict[str, dict[str, str]] = {}
    if df.empty:
        return out
    for _, r in df.iterrows():
        var = str(r.get("knhanes_variable", ""))
        if not var:
            continue
        out[var] = {
            "nhanes_mapping_class": str(r.get("mapping_class", "")),
            "nhanes_decision": str(r.get("nhanes_external_decision", "")),
            "nhanes_notes": str(r.get("notes", "")),
        }
    return out


def primary_block(var: str, sets: dict[str, list[str]]) -> str:
    if var in sets["primary_core_integrated"]:
        return "Core"
    if var in sets["primary_core_sdoh_integrated"]:
        return "SDOH add-on"
    if var in sets["primary_core_sdoh_hr_integrated"]:
        return "HR add-on"
    return "Not primary"


def difference_note(var: str, sets: dict[str, list[str]]) -> str:
    primary = var in sets["primary_core_sdoh_hr_integrated"]
    lean = var in sets["nhanes_lean23_integrated"]
    full25 = var in sets["full25_discussed_integrated"]
    if primary and lean:
        return "Primary 유지; lean23에도 포함"
    if primary and not lean:
        if var == "family_history_htn":
            return "Primary에는 포함되지만 NHANES recent public mapping blocker로 lean23에서 제외"
        return "Primary에는 포함되지만 lean23에서 제외"
    if not primary and lean:
        return "Primary 대비 lean23 추가 후보"
    if not primary and full25:
        return "lean23에는 없고 full25 논의에서만 추가 후보"
    return "Primary 외 enhanced 탐색 후보"


def yesno(condition: bool) -> str:
    return "Yes" if condition else "No"


def build_tables() -> dict[str, list[dict[str, Any]]]:
    sets = build_sets()
    meta = variable_metadata()
    miss = missingness_map()
    nhanes = nhanes_map()

    union = unique(
        sets["primary_core_sdoh_hr_integrated"]
        + sets["enhanced_conservative_integrated"]
        + sets["nhanes_lean23_integrated"]
        + sets["full25_discussed_integrated"]
        + sets["enhanced_v2_integrated"]
        + sets["enhanced_v2_female"]
        + sets["enhanced_v3_integrated"]
        + sets["enhanced_v3_female"]
    )

    order_rank = {v: i for i, v in enumerate(sets["primary_core_sdoh_hr_integrated"])}
    rows = []
    for var in union:
        m = meta.get(var, {})
        ms = miss.get(var, {})
        nm = nhanes.get(var, {})
        row = {
            "predictor": var,
            "description": m.get("description", ""),
            "source_or_derivation": m.get("source_or_derivation", ""),
            "coding_or_value": m.get("coding", ""),
            "primary_block": primary_block(var, sets),
            "primary_core_sdoh_hr_integrated": yesno(var in sets["primary_core_sdoh_hr_integrated"]),
            "primary_core_sdoh_hr_sex_specific": yesno(var in sets["primary_core_sdoh_hr_sex_specific"]),
            "nhanes_lean23_integrated": yesno(var in sets["nhanes_lean23_integrated"]),
            "full25_discussed_integrated": yesno(var in sets["full25_discussed_integrated"]),
            "enhanced_conservative_integrated": yesno(var in sets["enhanced_conservative_integrated"]),
            "enhanced_v2_integrated": yesno(var in sets["enhanced_v2_integrated"]),
            "enhanced_v2_female": yesno(var in sets["enhanced_v2_female"]),
            "enhanced_v3_integrated": yesno(var in sets["enhanced_v3_integrated"]),
            "enhanced_v3_female": yesno(var in sets["enhanced_v3_female"]),
            "difference_vs_primary": difference_note(var, sets),
            "primary_missing_n": ms.get("primary_missing_n", ""),
            "primary_missing_percent": ms.get("primary_missing_percent", ""),
            "v2_missing_n": ms.get("v2_missing_n", ""),
            "v2_missing_percent": ms.get("v2_missing_percent", ""),
            "v3_missing_n": ms.get("v3_missing_n", ""),
            "v3_missing_percent": ms.get("v3_missing_percent", ""),
            "nhanes_mapping_class": nm.get("nhanes_mapping_class", ""),
            "nhanes_decision": nm.get("nhanes_decision", ""),
            "nhanes_notes": nm.get("nhanes_notes", ""),
            "source_doc": m.get("source_doc", ""),
            "_sort_primary": order_rank.get(var, 10_000),
            "_sort_name": var,
        }
        rows.append(row)
    rows = sorted(rows, key=lambda r: (r["_sort_primary"], r["primary_block"] == "Not primary", r["_sort_name"]))
    for r in rows:
        r.pop("_sort_primary", None)
        r.pop("_sort_name", None)

    set_rows = []
    for name, features in sets.items():
        set_rows.append(
            {
                "set_name": name,
                "n_predictors": len(features),
                "predictors": ", ".join(features),
            }
        )

    missing_rows = []
    for label, rel in [
        ("primary", "reports/knhanes_development_missingness_final.csv"),
        ("conservative", "reports/knhanes_conservative_enhanced_missingness_2014_2020.csv"),
        ("v2", "reports/knhanes_enhanced_v2_missingness_2014_2020.csv"),
        ("v3", "reports/knhanes_enhanced_v3_missingness_2014_2020.csv"),
        ("lean23_by_group", "reports/nhanes_lean23_missingness_by_group_20260812.csv"),
    ]:
        df = read_csv(rel)
        if df.empty:
            continue
        for _, r in df.iterrows():
            rec = {"source_set": label}
            rec.update({k: r[k] for k in df.columns})
            missing_rows.append(rec)

    nhanes_rows = []
    nhanes_df = read_csv("reports/nhanes_external_mapping_qc_20260811.csv")
    if not nhanes_df.empty:
        nhanes_rows = nhanes_df.to_dict(orient="records")

    readme_rows = [
        {"item": "created_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M KST")},
        {"item": "purpose", "value": "기존 후보 predictor set과 locked primary predictor list 비교"},
        {"item": "primary_reference", "value": "reports/knhanes_primary_variables_locked.md; scripts/knhanes_primary_ml_pipeline.py"},
        {"item": "primary_definition", "value": "Primary = Core+SDOH+HR; integrated 19 predictors, sex-specific 18 predictors"},
        {"item": "existing_sets_compared", "value": "enhanced_conservative, enhanced_v2, enhanced_v3, nhanes_lean23, full25_discussed"},
        {"item": "note", "value": "full25_discussed is not a coded pipeline variable_set; it is lean23 plus height_cm and weight_kg."},
    ]

    return {
        "README": readme_rows,
        "Primary_vs_existing": rows,
        "Set_definitions": set_rows,
        "Missingness_sources": missing_rows,
        "NHANES_mapping": nhanes_rows,
    }


def write_outputs() -> None:
    tables = build_tables()
    REPORTS.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    main_csv = REPORTS / f"primary_vs_existing_predictor_comparison_{DATE_TAG}.csv"
    set_csv = REPORTS / f"primary_vs_existing_predictor_set_definitions_{DATE_TAG}.csv"
    pd.DataFrame(tables["Primary_vs_existing"]).to_csv(main_csv, index=False)
    pd.DataFrame(tables["Set_definitions"]).to_csv(set_csv, index=False)

    payload = {"date_tag": DATE_TAG, "tables": tables}
    (TMP_DIR / "comparison_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(main_csv)
    print(set_csv)
    print(TMP_DIR / "comparison_payload.json")


if __name__ == "__main__":
    write_outputs()
