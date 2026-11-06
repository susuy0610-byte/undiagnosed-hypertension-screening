#!/usr/bin/env python3
"""DeLong's test for correlated ROC curves: female core_sdoh_hr (primary) vs
the mandatory HIM non-invasive benchmark, on the SAME participants (paired OOF
predictions). Read-only statistical test on already-computed OOF predictions;
no retraining, retuning, or model change.

Implements the fast DeLong algorithm (DeLong, DeLong & Clarke-Pearson, 1988)
via midranks, as commonly used for comparing two correlated AUCs (e.g. pROC's
roc.test in R).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = ROOT / "analysis/knhanes_primary_ml_v1/oof_predictions"
REPORTS = ROOT / "reports"


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
    """predictions: (k models, n samples). Returns AUCs and their covariance matrix."""
    pos = predictions[:, y == 1]
    neg = predictions[:, y == 0]
    m = pos.shape[1]
    n = neg.shape[1]
    k = predictions.shape[0]

    ordered = np.concatenate([pos, neg], axis=1)  # tz ranks must be over [positives, negatives] in this order
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


def main() -> int:
    primary = pd.read_csv(OOF_DIR / "female__core_sdoh_hr__xgboost__oof.csv")
    him = pd.read_csv(OOF_DIR / "female__him_noninvasive__him_benchmark__oof.csv")

    primary = primary.sort_values("participant_id").reset_index(drop=True)
    him = him.sort_values("participant_id").reset_index(drop=True)
    assert (primary["participant_id"].to_numpy() == him["participant_id"].to_numpy()).all(), "participant order mismatch"
    assert (primary["undiagnosed_htn"].to_numpy() == him["undiagnosed_htn"].to_numpy()).all(), "outcome mismatch between the two OOF files"

    y = primary["undiagnosed_htn"].astype(int).to_numpy()
    p_primary = primary["pred_calibrated"].to_numpy(dtype=float)
    p_him = him["pred_calibrated"].to_numpy(dtype=float)

    preds = np.vstack([p_primary, p_him])
    aucs, cov = fast_delong(preds, y)

    auc_primary, auc_him = aucs[0], aucs[1]
    delta = auc_primary - auc_him
    var_delta = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se_delta = float(np.sqrt(var_delta))
    z = delta / se_delta
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lower = delta - 1.96 * se_delta
    ci_upper = delta + 1.96 * se_delta

    result = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "auroc_primary_core_sdoh_hr_xgboost": float(auc_primary),
        "auroc_him_noninvasive_benchmark": float(auc_him),
        "delta_auroc": float(delta),
        "se_delta_delong": se_delta,
        "z_statistic": float(z),
        "p_value_two_sided": p_value,
        "delta_95ci_lower": float(ci_lower),
        "delta_95ci_upper": float(ci_upper),
        "significant_at_0.05": bool(p_value < 0.05),
    }

    out_df = pd.DataFrame([result])
    out_path = REPORTS / "female_primary_vs_him_delong_test_20260820.csv"
    out_df.to_csv(out_path, index=False)

    print("DeLong test: female core_sdoh_hr (XGBoost, calibrated) vs HIM non-invasive benchmark")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
