#!/usr/bin/env python3
"""DeLong's test for correlated ROC curves: frozen primary core_sdoh_hr model
vs the mandatory HIM non-invasive benchmark, for all three model_sets, on the
SAME participants per group (paired OOF predictions). Read-only statistical
test on already-computed OOF predictions; no retraining, retuning, or model
change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = ROOT / "analysis/knhanes_primary_ml_v1/oof_predictions"
REPORTS = ROOT / "reports"

FROZEN_ALGORITHM = {"integrated": "xgboost", "male": "xgboost", "female": "xgboost"}


def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = predictions[:, y == 1]
    neg = predictions[:, y == 0]
    m, n, k = pos.shape[1], neg.shape[1], predictions.shape[0]
    ordered = np.concatenate([pos, neg], axis=1)
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(ordered[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def run_one(model_set: str) -> dict:
    algo = FROZEN_ALGORITHM[model_set]
    primary = pd.read_csv(OOF_DIR / f"{model_set}__core_sdoh_hr__{algo}__oof.csv").sort_values("participant_id").reset_index(drop=True)
    him = pd.read_csv(OOF_DIR / f"{model_set}__him_noninvasive__him_benchmark__oof.csv").sort_values("participant_id").reset_index(drop=True)
    assert (primary["participant_id"].to_numpy() == him["participant_id"].to_numpy()).all(), f"{model_set}: participant order mismatch"
    assert (primary["undiagnosed_htn"].to_numpy() == him["undiagnosed_htn"].to_numpy()).all(), f"{model_set}: outcome mismatch"

    y = primary["undiagnosed_htn"].astype(int).to_numpy()
    p_primary = primary["pred_calibrated"].to_numpy(dtype=float)
    p_him = him["pred_calibrated"].to_numpy(dtype=float)

    aucs, cov = fast_delong(np.vstack([p_primary, p_him]), y)
    delta = float(aucs[0] - aucs[1])
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = delta / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "model_set": model_set, "algorithm": algo, "n": int(len(y)), "events": int(y.sum()),
        "auroc_primary": float(aucs[0]), "auroc_him_benchmark": float(aucs[1]), "delta_auroc": delta,
        "se_delta_delong": se, "z_statistic": float(z), "p_value_two_sided": p_value,
        "delta_95ci_lower": delta - 1.96 * se, "delta_95ci_upper": delta + 1.96 * se,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main() -> int:
    rows = [run_one(ms) for ms in ["integrated", "male", "female"]]
    df = pd.DataFrame(rows)
    out_path = REPORTS / "all_groups_primary_vs_him_delong_test_20260820.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
