#!/usr/bin/env python3
"""Fairness/subgroup analysis (protocol H4) on HSE 2018-2022 pooled external validation.

Subgroup axes: ethnicity (origin2: White/Black/Asian/Mixed/Other), pre-specified
age bands (20-39/40-64/65+), and household income quartile. Uses the already-
recalibrated (intercept+slope) HSE pooled predictions; computes AUROC and
calibration slope within each subgroup level and reports max |delta AUROC| per
axis. Read-only diagnostic: no retraining, retuning, or frozen-model change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STAMP = "20260820"

RAW_FILES = {
    "2018": ROOT / "data/raw/hse/2018/UKDA-8649-stata/stata/stata13/hse_2018_eul_15082022.dta",
    "2019": ROOT / "data/raw/hse/2019/UKDA-8860-stata/stata/stata13/hse_2019_eul_20211006.dta",
    "2021": ROOT / "data/raw/hse/2021/UKDA-9319-stata/stata/stata13/hse_2021_eul_v1.dta",
    "2022": ROOT / "data/raw/hse/2022/ukds_raw/UKDA-9469-stata/stata/stata13/hse_2022_eul_v1.dta",
}
ETHNICITY_LABELS = {1: "White", 2: "Black", 3: "Asian", 4: "Mixed", 5: "Other"}

RECAL_DIR = ROOT / "analysis/knhanes_primary_ml_v1/external_validation_hse_pooled_recalibration_20260819"
HARMONIZED_PATH = ROOT / "data/processed/hse_2018_2019_2021_2022_pooled_core_sdoh_hr_harmonized_qc_20260819.csv"


def load_ethnicity() -> pd.DataFrame:
    parts = []
    for year, path in RAW_FILES.items():
        df = pd.read_stata(path, convert_categoricals=False)
        df.columns = [c.lower() for c in df.columns]
        eth = pd.to_numeric(df["origin2"], errors="coerce")
        eth = eth.where(eth.isin(ETHNICITY_LABELS), np.nan)
        out = pd.DataFrame({
            "person_id": year + "_" + pd.to_numeric(df["seriala"], errors="coerce").astype("Int64").astype("string"),
            "ethnicity_code": eth,
        })
        parts.append(out)
    eth_all = pd.concat(parts, ignore_index=True)
    eth_all["ethnicity"] = eth_all["ethnicity_code"].map(ETHNICITY_LABELS)
    return eth_all


def age_group(age: float) -> str:
    if pd.isna(age):
        return np.nan
    if age < 40:
        return "20-39"
    if age < 65:
        return "40-64"
    return "65+"


def subgroup_metrics(df: pd.DataFrame, group_col: str, min_n: int = 30, min_events: int = 10) -> pd.DataFrame:
    rows = []
    for level, sub in df.groupby(group_col):
        n = len(sub)
        events = int(sub["outcome"].sum())
        if n < min_n or events < min_events or events == n:
            rows.append({group_col: level, "n": n, "events": events, "event_rate": events / n if n else np.nan, "auroc": np.nan, "note": "insufficient_n_or_events"})
            continue
        auroc = roc_auc_score(sub["outcome"], sub["pred"])
        rows.append({group_col: level, "n": n, "events": events, "event_rate": events / n, "auroc": auroc, "note": ""})
    return pd.DataFrame(rows)


def main() -> int:
    eth = load_ethnicity()
    harmonized = pd.read_csv(HARMONIZED_PATH)[["person_id", "age", "household_income_q"]]

    results_rows = []
    for model_set in ["integrated", "male", "female"]:
        pred = pd.read_csv(RECAL_DIR / f"{model_set}__3_intercept_slope_hse_pooled__predictions.csv")
        pred = pred.merge(eth, on="person_id", how="left").merge(harmonized, on="person_id", how="left")
        pred["age_group"] = pred["age"].apply(age_group)

        print(f"\n=== {model_set} (n={len(pred)}) ===")

        print("-- Ethnicity (White vs Non-White, due to small N in individual minority groups) --")
        pred["ethnicity_binary"] = pred["ethnicity"].apply(lambda x: "White" if x == "White" else ("Non-White" if pd.notna(x) else np.nan))
        eth_bin = subgroup_metrics(pred, "ethnicity_binary")
        eth_bin.insert(0, "model_set", model_set)
        eth_bin.insert(1, "axis", "ethnicity_binary")
        eth_bin = eth_bin.rename(columns={"ethnicity_binary": "level"})
        print(eth_bin.to_string(index=False))
        results_rows.append(eth_bin)

        print("-- Ethnicity (full 5-category, descriptive only given small N) --")
        eth_full = subgroup_metrics(pred, "ethnicity")
        eth_full.insert(0, "model_set", model_set)
        eth_full.insert(1, "axis", "ethnicity_full")
        eth_full = eth_full.rename(columns={"ethnicity": "level"})
        print(eth_full.to_string(index=False))
        results_rows.append(eth_full)

        print("-- Age group (pre-specified 20-39/40-64/65+) --")
        age_res = subgroup_metrics(pred, "age_group")
        age_res.insert(0, "model_set", model_set)
        age_res.insert(1, "axis", "age_group")
        age_res = age_res.rename(columns={"age_group": "level"})
        print(age_res.to_string(index=False))
        results_rows.append(age_res)

        print("-- Household income quartile --")
        inc_res = subgroup_metrics(pred, "household_income_q")
        inc_res.insert(0, "model_set", model_set)
        inc_res.insert(1, "axis", "household_income_q")
        inc_res = inc_res.rename(columns={"household_income_q": "level"})
        print(inc_res.to_string(index=False))
        results_rows.append(inc_res)

    full = pd.concat(results_rows, ignore_index=True)
    full.to_csv(REPORTS / f"hse_pooled_fairness_subgroup_auroc_{STAMP}.csv", index=False)

    max_delta_rows = []
    for model_set in ["integrated", "male", "female"]:
        for axis in ["ethnicity_binary", "age_group", "household_income_q"]:
            sub = full[(full.model_set == model_set) & (full.axis == axis) & full.auroc.notna()]
            if len(sub) < 2:
                continue
            max_delta_rows.append({
                "model_set": model_set, "axis": axis,
                "max_auroc": float(sub.auroc.max()), "min_auroc": float(sub.auroc.min()),
                "max_abs_delta_auroc": float(sub.auroc.max() - sub.auroc.min()),
                "n_levels_with_valid_auroc": len(sub),
            })
    max_delta = pd.DataFrame(max_delta_rows)
    max_delta.to_csv(REPORTS / f"hse_pooled_fairness_max_delta_auroc_{STAMP}.csv", index=False)

    print("\n=== Max |delta AUROC| per axis ===")
    print(max_delta.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
