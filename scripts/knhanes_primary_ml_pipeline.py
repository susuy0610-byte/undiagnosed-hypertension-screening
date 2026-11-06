#!/usr/bin/env python3
"""
KNHANES primary ML pipeline for undiagnosed hypertension screening.

Default full-run sequence:
1. KNHANES 2014-2020 5-fold OOF development analysis.
2. Fold-internal MICE m=20 preprocessing and hyperparameter tuning.
3. OOF metric, DCA, calibration, and threshold summaries.
4. Final refit on full 2014-2020 development data.
5. Frozen 2021-2022 temporal validation.

The script is checkpointed by analysis population, variable set, algorithm,
and fold. Re-running the same command skips completed artifacts unless
--overwrite is supplied. Increasing --n-trials resumes the Optuna study from
the completed trial count and refreshes the fold model/OOF artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = ROOT / "data" / "processed" / "knhanes_development_2014_2020_primary.csv"
TIME_PATH = ROOT / "data" / "processed" / "knhanes_time_validation_2021_2022_check_only.csv"
LOCKED_VARIABLES_PATH = ROOT / "reports" / "knhanes_primary_variables_locked.md"

CONTINUOUS = [
    "age",
    "bmi",
    "waist_cm",
    "heart_rate",
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "age_squared",
    "bmi_squared",
    "age_bmi_interaction",
    "age_waist_interaction",
    "walking_days",
    "strength_days",
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
    "menarche_age",
    "menopause_age",
    "first_birth_age",
    "last_birth_age",
    "breastfeeding_months",
    "breastfed_children_count",
    "total_energy_kcal",
]
CORE_CONTINUOUS = ["age", "bmi", "waist_cm"]
CORE_BMI_WHO_CONTINUOUS = ["age", "waist_cm"]
ENHANCED_CONSERVATIVE_CONTINUOUS = ["height_cm", "weight_kg", "waist_to_height_ratio"]
NHANES_LEAN23_CONTINUOUS = ["waist_to_height_ratio", "walking_days", "strength_days"]
ENHANCED_V2_CONTINUOUS = [
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "age_squared",
    "bmi_squared",
    "age_bmi_interaction",
    "age_waist_interaction",
    "walking_days",
    "strength_days",
]
# Minimal ablation: only the nonlinear/interaction terms from enhanced_v2, no other
# added predictors (height/weight/WHtR/walking/strength etc excluded). Used to isolate
# whether enhanced_v2's small OOF edge is attributable to these 4 terms specifically.
NONLINEAR_ONLY_CONTINUOUS = [
    "age_squared",
    "bmi_squared",
    "age_bmi_interaction",
    "age_waist_interaction",
]
CORE_CATEGORICAL_ALL = [
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
]
CORE_BMI_WHO_CATEGORICAL_ALL = CORE_CATEGORICAL_ALL + ["bmi_who_category"]
SDOH_CATEGORICAL = [
    "education_level",
    "household_income_q",
    "employment_status",
    "marital_status",
    "food_security_status",
]
ENHANCED_CONSERVATIVE_CATEGORICAL = ["dyslipidemia_dx_or_med", "central_obesity"]
ENHANCED_V2_CATEGORICAL = [
    "dyslipidemia_dx_or_med",
    "central_obesity",
    "bmi_category",
    "stress_level",
    "subjective_body_shape",
    "weight_change_1yr",
    "kidney_disease_dx",
    "family_history_ihd",
    "family_history_stroke",
]
ENHANCED_V2_FEMALE_ONLY_CATEGORICAL = ["menstrual_status"]
ENHANCED_V3_CONTINUOUS = [
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
ENHANCED_V3_CATEGORICAL = [
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
ENHANCED_V3_FEMALE_ONLY_CONTINUOUS = [
    "menarche_age",
    "menopause_age",
    "first_birth_age",
    "last_birth_age",
    "breastfeeding_months",
    "breastfed_children_count",
]
ENHANCED_V3_FEMALE_ONLY_CATEGORICAL = [
    "parity_experience",
    "oral_contraceptive_use",
    "breastfeeding_experience",
]
CATEGORICAL_ALL = CORE_CATEGORICAL_ALL + SDOH_CATEGORICAL
NHANES_LEAN23_CORE_CATEGORICAL = [x for x in CORE_CATEGORICAL_ALL if x != "family_history_htn"]

RCS_KNOT_PROBS = (0.05, 0.35, 0.65, 0.95)
RCS_VARIABLES_BY_SET = {
    "core_sdoh_hr_bmi_who_waist_rcs": ["waist_cm"],
}

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

VARIABLE_SETS = [
    "core",
    "core_sdoh",
    "core_sdoh_hr",
    "enhanced_conservative",
    "nhanes_lean23",
    "core_sdoh_hr_bmi_who_waist_rcs",
    "core_sdoh_hr_nonlinear",
    "enhanced_v2",
    "enhanced_v3",
    "him_noninvasive",
    "app_lean8",
    "app_lean6",
]

# Candidate concise variable sets for the web-app deployment model (protocol
# 9.2: 5-8 self-reportable variables, heart_rate excluded for self-measurement
# difficulty). app_lean8 follows the protocol's a priori 9.2 candidate list;
# app_lean6 drops diabetes_dx and current_smoking (lowest SHAP rank among the
# 8, rank 11/13 across all three model_sets) to test how much is lost by
# going leaner still. Both are ablation-only variable sets for the app model,
# not a replacement for the frozen primary core_sdoh_hr research model.
APP_LEAN8_CONTINUOUS = ["age", "bmi", "waist_cm"]
APP_LEAN8_CATEGORICAL = ["sex", "family_history_htn", "alcohol_frequency", "diabetes_dx", "current_smoking"]
APP_LEAN6_CONTINUOUS = ["age", "bmi", "waist_cm"]
APP_LEAN6_CATEGORICAL = ["sex", "family_history_htn", "alcohol_frequency"]
DEFAULT_VARIABLE_SETS = [x for x in VARIABLE_SETS if x != "core_sdoh_hr_bmi_who_waist_rcs"]

MODEL_SETS = ["integrated", "male", "female"]
ALGORITHMS = [
    "logistic",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "adaboost",
    "him_benchmark",
]
DEFAULT_ALGORITHMS = [x for x in ALGORITHMS if x != "extra_trees"]
NON_TUNABLE_ALGORITHMS = ["logistic", "him_benchmark"]
TUNABLE_ALGORITHMS = [x for x in ALGORITHMS if x not in NON_TUNABLE_ALGORITHMS]

VALID_CATEGORIES = {
    "sex": [1, 2],
    "diabetes_dx": [0, 1],
    "aerobic_activity": [0, 1],
    "self_rated_health": [1, 2, 3, 4, 5],
    "activity_limitation": [0, 1],
    "family_history_htn": [0, 1],
    "current_smoking": [0, 1],
    "alcohol_frequency": [0, 1, 2, 3, 4, 5],
    "drink_amount": [0, 1, 2, 3, 4, 5],
    "binge_drinking": [0, 1, 2, 3, 4, 5],
    "education_level": [1, 2, 3, 4],
    "household_income_q": [1, 2, 3, 4],
    "employment_status": [1, 2],
    "marital_status": [0, 1, 2, 3, 4],
    "food_security_status": [1, 2, 3, 4],
    "dyslipidemia_dx_or_med": [0, 1],
    "cancer_history": [0, 1],
    "central_obesity": [0, 1],
    "bmi_category": [1, 2, 3, 4, 5],
    "bmi_who_category": [1, 2, 3, 4, 5, 6],
    "stress_level": [1, 2, 3, 4],
    "subjective_body_shape": [1, 2, 3, 4, 5],
    "weight_change_1yr": [1, 2, 3],
    "kidney_disease_dx": [0, 1],
    "family_history_ihd": [0, 1],
    "family_history_stroke": [0, 1],
    "menstrual_status": [2, 3, 4, 5, 6, 7],
    "lifetime_smoking_level": [1, 2, 3],
    "secondhand_smoke_work": [1, 2, 8],
    "secondhand_smoke_home": [1, 2, 3, 8],
    "ecig_ever": [0, 1],
    "ecig_current": [0, 1],
    "personal_income_q": [1, 2, 3, 4],
    "personal_income_quintile": [1, 2, 3, 4, 5],
    "household_income_quintile": [1, 2, 3, 4, 5],
    "house_ownership": [1, 2, 3],
    "housing_type": [1, 2, 3, 4, 5],
    "apartment_type": [1, 2],
    "basic_livelihood_support": [10, 20],
    "health_insurance_type": [10, 20, 30],
    "private_insurance": [1, 2],
    "occupation_group": [1, 2, 3, 4, 5, 6, 7],
    "father_education": [1, 2, 3, 4, 5, 6, 7, 8],
    "mother_education": [1, 2, 3, 4, 5, 6, 7, 8],
    "urban_rural": [1, 2],
    "parity_experience": [0, 1],
    "oral_contraceptive_use": [0, 1],
    "breastfeeding_experience": [0, 1],
}

LABELS = {
    "integrated": "Integrated",
    "male": "Male-only",
    "female": "Female-only",
    "core": "Core",
    "core_sdoh": "Core + SDOH",
    "core_sdoh_hr": "Core + SDOH + heart rate",
    "enhanced_conservative": "Conservative enhanced non-invasive",
    "nhanes_lean23": "NHANES-compatible lean 23",
    "core_sdoh_hr_bmi_who_waist_rcs": "Core + SDOH + HR with WHO BMI category + waist RCS",
    "core_sdoh_hr_nonlinear": "Core + SDOH + HR with age/BMI/waist nonlinear+interaction terms only",
    "app_lean8": "App-deployment concise 8-variable candidate (protocol 9.2 self-reportable list)",
    "app_lean6": "App-deployment concise 6-variable candidate (age/sex/BMI/waist/family_history/alcohol only)",
    "enhanced_v2": "Enhanced v2 defensible non-invasive",
    "enhanced_v3": "Enhanced v3 lifestyle + granular SDOH",
    "him_noninvasive": "Hypertension Index non-invasive subset",
    "logistic": "Logistic regression",
    "lasso": "LASSO",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "adaboost": "AdaBoost",
    "him_benchmark": "Hypertension Index benchmark",
}


@dataclass
class Config:
    output_dir: str
    dev_path: str
    time_path: str
    seed: int
    n_folds: int
    inner_folds: int
    m_imputations: int
    n_trials: int
    target_sensitivity: float
    n_jobs: int
    stage: str
    model_sets: list[str]
    variable_sets: list[str]
    algorithms: list[str]
    outer_folds: list[int] | None
    overwrite: bool


class MICEOneHotPreprocessor:
    def __init__(
        self,
        features: list[str],
        seed: int,
        max_iter: int = 10,
        rcs_variables: list[str] | None = None,
        rcs_knot_probs: tuple[float, ...] = RCS_KNOT_PROBS,
    ):
        self.features = list(features)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.continuous = [x for x in CONTINUOUS if x in self.features]
        self.categorical = [x for x in self.features if x not in self.continuous]
        self.rcs_variables = [x for x in (rcs_variables or []) if x in self.continuous]
        self.rcs_knot_probs = tuple(float(x) for x in rcs_knot_probs)
        self.rcs_knots: dict[str, list[float]] = {}
        self.rcs_mean: dict[str, list[float]] = {}
        self.rcs_std: dict[str, list[float]] = {}
        self.cat_modes: dict[str, int] = {}
        self.cont_mean: dict[str, float] = {}
        self.cont_std: dict[str, float] = {}
        self.columns_: list[str] = []
        self.imputer: IterativeImputer | None = None

    def fit(self, X: pd.DataFrame) -> "MICEOneHotPreprocessor":
        data = X[self.features].astype(float)
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=self.max_iter,
            random_state=self.seed,
            skip_complete=False,
        )
        imputed = pd.DataFrame(self.imputer.fit_transform(data), columns=self.features, index=X.index)
        fixed = self._fix_categories(imputed, fit_modes=True)
        for col in self.continuous:
            mean = float(fixed[col].mean())
            std = float(fixed[col].std(ddof=0))
            self.cont_mean[col] = mean
            self.cont_std[col] = std if std > 0 else 1.0
        for col in self.rcs_variables:
            knots = self._compute_rcs_knots(fixed[col])
            self.rcs_knots[col] = knots
            basis = self._rcs_basis(fixed[col], knots)
            self.rcs_mean[col] = [float(basis[:, i].mean()) for i in range(basis.shape[1])]
            self.rcs_std[col] = [
                float(basis[:, i].std(ddof=0)) if float(basis[:, i].std(ddof=0)) > 0 else 1.0
                for i in range(basis.shape[1])
            ]
        matrix = self._encode(fixed)
        self.columns_ = list(matrix.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None:
            raise RuntimeError("Preprocessor is not fitted.")
        data = X[self.features].astype(float)
        imputed = pd.DataFrame(self.imputer.transform(data), columns=self.features, index=X.index)
        fixed = self._fix_categories(imputed, fit_modes=False)
        matrix = self._encode(fixed)
        return matrix.reindex(columns=self.columns_, fill_value=0.0)

    def _fix_categories(self, X: pd.DataFrame, fit_modes: bool) -> pd.DataFrame:
        fixed = X.copy()
        for col in self.categorical:
            allowed = np.array(VALID_CATEGORIES[col], dtype=int)
            rounded = np.rint(fixed[col].to_numpy(dtype=float)).astype(int)
            if fit_modes:
                observed = X[col].dropna()
                observed = np.rint(observed.to_numpy(dtype=float)).astype(int)
                observed = observed[np.isin(observed, allowed)]
                self.cat_modes[col] = int(pd.Series(observed).mode().iloc[0]) if observed.size else int(allowed[0])
            mode = self.cat_modes[col]
            rounded[~np.isin(rounded, allowed)] = mode
            fixed[col] = rounded
        return fixed

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        if self.continuous:
            cont = pd.DataFrame(index=X.index)
            for col in self.continuous:
                cont[f"num__{col}"] = (X[col].astype(float) - self.cont_mean[col]) / self.cont_std[col]
            for col in self.rcs_variables:
                basis = self._rcs_basis(X[col], self.rcs_knots[col])
                for i in range(basis.shape[1]):
                    cont[f"num__{col}_rcs{i + 1}"] = (basis[:, i] - self.rcs_mean[col][i]) / self.rcs_std[col][i]
            pieces.append(cont)
        for col in self.categorical:
            # Reference coding: drop the first, prespecified category.
            for level in VALID_CATEGORIES[col][1:]:
                pieces.append(pd.DataFrame({f"cat__{col}_{level}": (X[col].astype(int) == level).astype(float)}, index=X.index))
        return pd.concat(pieces, axis=1)

    def _compute_rcs_knots(self, values: pd.Series) -> list[float]:
        observed = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        unique = np.unique(observed)
        if unique.size < 4:
            return []
        knots = np.quantile(observed, self.rcs_knot_probs)
        if np.unique(knots).size < 4:
            knots = np.linspace(float(unique.min()), float(unique.max()), num=4)
        return [float(x) for x in knots]

    @staticmethod
    def _rcs_basis(values: pd.Series | np.ndarray, knots: list[float]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if len(knots) < 4:
            return np.empty((len(x), 0))
        k = np.asarray(knots, dtype=float)
        last = k[-1]
        penultimate = k[-2]
        denom = last - penultimate
        if denom <= 0:
            return np.empty((len(x), 0))
        basis_cols = []
        for knot in k[:-2]:
            col = np.maximum(x - knot, 0.0) ** 3
            col -= np.maximum(x - penultimate, 0.0) ** 3 * ((last - knot) / denom)
            col += np.maximum(x - last, 0.0) ** 3 * ((penultimate - knot) / denom)
            basis_cols.append(col)
        return np.vstack(basis_cols).T


class LogitCalibrator:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.intercept_: float = 0.0
        self.slope_: float = 1.0
        self.fitted_: bool = False

    def fit(self, y: np.ndarray, p: np.ndarray) -> "LogitCalibrator":
        z = self._logit(p).reshape(-1, 1)
        lr = logistic_model({"penalty": "none"}, seed=0)
        lr.fit(z, y)
        self.intercept_ = float(lr.intercept_[0])
        self.slope_ = float(lr.coef_[0][0])
        self.fitted_ = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            return np.asarray(p, dtype=float)
        z = self.intercept_ + self.slope_ * self._logit(p)
        return 1.0 / (1.0 + np.exp(-z))

    def _logit(self, p: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(p, dtype=float), self.eps, 1.0 - self.eps)
        return np.log(clipped / (1.0 - clipped))


def ensure_dirs(out: Path) -> None:
    for sub in [
        "config",
        "folds",
        "logs",
        "preprocessors",
        "inner_cache",
        "tuning",
        "cv_models",
        "oof_predictions",
        "metrics",
        "plots",
        "final_models",
        "time_validation",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def setup_logging(out: Path) -> None:
    log_path = out / "logs" / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def add_derived_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bmi_who_category" not in out.columns and "bmi" in out.columns:
        bmi = pd.to_numeric(out["bmi"], errors="coerce")
        category = pd.Series(np.nan, index=out.index, dtype="float")
        category[bmi < 18.5] = 1
        category[(bmi >= 18.5) & (bmi < 25.0)] = 2
        category[(bmi >= 25.0) & (bmi < 30.0)] = 3
        category[(bmi >= 30.0) & (bmi < 35.0)] = 4
        category[(bmi >= 35.0) & (bmi < 40.0)] = 5
        category[bmi >= 40.0] = 6
        out["bmi_who_category"] = category
    return out


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_derived_predictors(df)
    df["psu_group"] = df["source_year"].astype(str) + "_" + df["psu"].astype(str)
    return df


def subset_for_model_set(df: pd.DataFrame, model_set: str) -> pd.DataFrame:
    if model_set == "male":
        return df[df["sex"] == 1].copy()
    if model_set == "female":
        return df[df["sex"] == 2].copy()
    return df.copy()


def predictors_for(model_set: str, variable_set: str) -> list[str]:
    if variable_set == "him_noninvasive":
        if model_set == "male":
            return list(HIM_NONINVASIVE_MALE)
        if model_set == "female":
            return list(HIM_NONINVASIVE_FEMALE)
        return list(HIM_NONINVASIVE_INTEGRATED)
    if variable_set == "core":
        features = CORE_CONTINUOUS + CORE_CATEGORICAL_ALL
    elif variable_set == "core_sdoh":
        features = CORE_CONTINUOUS + CORE_CATEGORICAL_ALL + SDOH_CATEGORICAL
    elif variable_set == "core_sdoh_hr":
        features = CORE_CONTINUOUS + ["heart_rate"] + CORE_CATEGORICAL_ALL + SDOH_CATEGORICAL
    elif variable_set == "core_sdoh_hr_bmi_who_waist_rcs":
        features = CORE_BMI_WHO_CONTINUOUS + ["heart_rate"] + CORE_BMI_WHO_CATEGORICAL_ALL + SDOH_CATEGORICAL
    elif variable_set == "core_sdoh_hr_nonlinear":
        features = (
            CORE_CONTINUOUS
            + NONLINEAR_ONLY_CONTINUOUS
            + ["heart_rate"]
            + CORE_CATEGORICAL_ALL
            + SDOH_CATEGORICAL
        )
    elif variable_set == "app_lean8":
        features = APP_LEAN8_CONTINUOUS + APP_LEAN8_CATEGORICAL
    elif variable_set == "app_lean6":
        features = APP_LEAN6_CONTINUOUS + APP_LEAN6_CATEGORICAL
    elif variable_set == "enhanced_conservative":
        features = (
            CORE_CONTINUOUS
            + ENHANCED_CONSERVATIVE_CONTINUOUS
            + ["heart_rate"]
            + CORE_CATEGORICAL_ALL
            + SDOH_CATEGORICAL
            + ENHANCED_CONSERVATIVE_CATEGORICAL
        )
    elif variable_set == "nhanes_lean23":
        features = (
            CORE_CONTINUOUS
            + NHANES_LEAN23_CONTINUOUS
            + ["heart_rate"]
            + NHANES_LEAN23_CORE_CATEGORICAL
            + SDOH_CATEGORICAL
            + ENHANCED_CONSERVATIVE_CATEGORICAL
        )
    elif variable_set == "enhanced_v2":
        features = (
            CORE_CONTINUOUS
            + ENHANCED_V2_CONTINUOUS
            + ["heart_rate"]
            + CORE_CATEGORICAL_ALL
            + SDOH_CATEGORICAL
            + ENHANCED_V2_CATEGORICAL
        )
    elif variable_set == "enhanced_v3":
        features = (
            CORE_CONTINUOUS
            + ENHANCED_V2_CONTINUOUS
            + ENHANCED_V3_CONTINUOUS
            + ["heart_rate"]
            + CORE_CATEGORICAL_ALL
            + SDOH_CATEGORICAL
            + ENHANCED_V2_CATEGORICAL
            + ENHANCED_V3_CATEGORICAL
        )
    else:
        raise ValueError(f"Unknown variable set: {variable_set}")

    if model_set in {"male", "female"}:
        features = [x for x in features if x != "sex"]
    if variable_set == "enhanced_v2" and model_set == "female":
        features = features + ENHANCED_V2_FEMALE_ONLY_CATEGORICAL
    if variable_set == "enhanced_v3" and model_set == "female":
        features = (
            features
            + ENHANCED_V2_FEMALE_ONLY_CATEGORICAL
            + ENHANCED_V3_FEMALE_ONLY_CONTINUOUS
            + ENHANCED_V3_FEMALE_ONLY_CATEGORICAL
        )
    return features


def rcs_variables_for(variable_set: str) -> list[str]:
    return list(RCS_VARIABLES_BY_SET.get(variable_set, []))


def model_combo_id(model_set: str, variable_set: str, algorithm: str | None = None, fold: int | None = None) -> str:
    pieces = [model_set, variable_set]
    if algorithm is not None:
        pieces.append(algorithm)
    if fold is not None:
        pieces.append(f"fold{fold}")
    return "__".join(pieces)


def valid_model_combo(variable_set: str, algorithm: str) -> bool:
    if variable_set == "him_noninvasive":
        return algorithm == "him_benchmark"
    return algorithm != "him_benchmark"


def create_or_load_folds(df: pd.DataFrame, config: Config, out: Path) -> pd.DataFrame:
    fold_path = out / "folds" / f"fold_assignments_{config.n_folds}fold_seed{config.seed}.csv"
    if fold_path.exists() and not config.overwrite:
        existing = pd.read_csv(fold_path)
        current_ids = set(df["participant_id"].astype(str))
        existing_ids = set(existing["participant_id"].astype(str))
        duplicate_ids = int(existing["participant_id"].duplicated().sum())
        if current_ids == existing_ids and duplicate_ids == 0:
            logging.info("Loading existing fold assignments: %s", fold_path)
            return existing
        logging.warning(
            "Existing fold file does not match current dataset "
            "(current n=%s, existing n=%s, duplicate ids=%s). Regenerating folds.",
            len(current_ids),
            len(existing_ids),
            duplicate_ids,
        )

    y = df["undiagnosed_htn"].astype(int).to_numpy()
    groups = df["psu_group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    rows = []
    for fold, (_, valid_idx) in enumerate(splitter.split(df, y, groups), start=1):
        rows.append(
            pd.DataFrame(
                {
                    "participant_id": df.iloc[valid_idx]["participant_id"].to_numpy(),
                    "fold": fold,
                }
            )
        )
    folds = pd.concat(rows, ignore_index=True)
    merged = df[["participant_id", "undiagnosed_htn", "source_year", "sex", "psu_group"]].merge(folds, on="participant_id", how="left")
    if merged["fold"].isna().any():
        raise RuntimeError("Fold assignment failed for some rows.")
    check = (
        merged.groupby("fold")
        .agg(n=("participant_id", "size"), events=("undiagnosed_htn", "sum"), psu_groups=("psu_group", "nunique"))
        .reset_index()
    )
    check["event_rate"] = check["events"] / check["n"]
    write_csv(merged, fold_path)
    write_csv(check, out / "folds" / "fold_balance.csv")
    logging.info("Created fold assignments: %s", fold_path)
    return merged


def fit_preprocessors(
    X: pd.DataFrame,
    features: list[str],
    m: int,
    seed: int,
    rcs_variables: list[str] | None = None,
) -> list[MICEOneHotPreprocessor]:
    preprocessors = []
    for i in range(m):
        prep = MICEOneHotPreprocessor(
            features=features,
            seed=seed + 1009 * (i + 1),
            rcs_variables=rcs_variables,
        )
        prep.fit(X)
        preprocessors.append(prep)
    return preprocessors


def transformed_list(preprocessors: list[MICEOneHotPreprocessor], X: pd.DataFrame) -> list[pd.DataFrame]:
    return [prep.transform(X) for prep in preprocessors]


def logistic_model(params: dict[str, Any], seed: int) -> LogisticRegression:
    penalty = params.get("penalty", "none")
    if penalty == "none":
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=seed)
    if penalty == "l1":
        return LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(params["C"]),
            max_iter=5000,
            random_state=seed,
        )
    if penalty == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
            random_state=seed,
        )
    raise ValueError(f"Unknown logistic penalty: {penalty}")


def make_estimator(algorithm: str, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    if algorithm in {"logistic", "him_benchmark"}:
        return logistic_model({"penalty": "none"}, seed)
    if algorithm == "lasso":
        return logistic_model({"penalty": "l1", **params}, seed)
    if algorithm == "elastic_net":
        return logistic_model({"penalty": "elasticnet", **params}, seed)
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=n_jobs,
        )
    if algorithm == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=n_jobs,
        )
    if algorithm == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )
    if algorithm == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            max_depth=-1 if params["max_depth"] == 0 else int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_samples=int(params["min_child_samples"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if algorithm == "adaboost":
        stump = DecisionTreeClassifier(max_depth=int(params["max_depth"]), random_state=seed)
        return AdaBoostClassifier(
            estimator=stump,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            random_state=seed,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def suggest_params(trial: optuna.Trial, algorithm: str) -> dict[str, Any]:
    if algorithm == "lasso":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if algorithm == "elastic_net":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if algorithm == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [0, 3, 5, 8, 12, 16, 24]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if algorithm == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [0, 3, 5, 8, 12, 16, 24]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if algorithm == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    if algorithm == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "max_depth": trial.suggest_categorical("max_depth", [0, 3, 5, 8, 12, 16]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    if algorithm == "adaboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
        }
    raise ValueError(f"No search space for {algorithm}")


def mean_predict_from_models(models: list[Any], matrices: list[pd.DataFrame]) -> np.ndarray:
    preds = [model.predict_proba(mat)[:, 1] for model, mat in zip(models, matrices)]
    return np.mean(np.vstack(preds), axis=0)


def fit_model_ensemble(
    algorithm: str,
    params: dict[str, Any],
    matrices_train: list[pd.DataFrame],
    y_train: np.ndarray,
    seed: int,
    n_jobs: int,
) -> list[Any]:
    models = []
    for i, mat in enumerate(matrices_train):
        estimator = make_estimator(algorithm, params, seed + i * 997, n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(mat, y_train)
        models.append(estimator)
    return models


def participant_ids(df: pd.DataFrame) -> list[str]:
    return df["participant_id"].astype(str).tolist()


def cache_metadata(
    config: Config,
    model_set: str,
    variable_set: str,
    fold: int,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    meta = {
        "model_set": model_set,
        "variable_set": variable_set,
        "fold": fold,
        "features": list(features),
        "rcs_variables": rcs_variables_for(variable_set),
        "rcs_knot_probs": list(RCS_KNOT_PROBS),
        "m_imputations": config.m_imputations,
        "seed": config.seed,
        "inner_folds": config.inner_folds,
        "train_participant_ids": participant_ids(train_df),
    }
    if valid_df is not None:
        meta["valid_participant_ids"] = participant_ids(valid_df)
    return meta


def cache_matches(obj: dict[str, Any], expected: dict[str, Any]) -> bool:
    return obj.get("metadata") == expected


def load_or_create_outer_cache(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    config: Config,
    out: Path,
    model_set: str,
    variable_set: str,
    fold: int,
) -> dict[str, Any]:
    cache_path = out / "preprocessors" / f"{model_combo_id(model_set, variable_set, fold=fold)}.joblib"
    expected = cache_metadata(config, model_set, variable_set, fold, features, train_df, valid_df)
    if cache_path.exists() and not config.overwrite:
        obj = joblib.load(cache_path)
        if cache_matches(obj, expected):
            logging.info("Loaded outer MICE cache: %s", cache_path)
            return obj
        logging.warning("Outer MICE cache metadata mismatch; rebuilding: %s", cache_path)

    X_train = train_df[features]
    X_valid = valid_df[features]
    preprocessors = fit_preprocessors(
        X_train,
        features,
        config.m_imputations,
        config.seed + fold * 100,
        rcs_variables_for(variable_set),
    )
    obj = {
        "metadata": expected,
        "features": features,
        "preprocessors": preprocessors,
        "matrices_train": transformed_list(preprocessors, X_train),
        "matrices_valid": transformed_list(preprocessors, X_valid),
    }
    joblib.dump(obj, cache_path, compress=3)
    logging.info("Saved outer MICE cache: %s", cache_path)
    return obj


def load_or_create_inner_cache(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    config: Config,
    out: Path,
    model_set: str,
    variable_set: str,
    fold: int,
) -> list[dict[str, Any]]:
    cache_path = out / "inner_cache" / f"{model_combo_id(model_set, variable_set, fold=fold)}__inner.joblib"
    expected = cache_metadata(config, model_set, variable_set, fold, features, train_df)
    if cache_path.exists() and not config.overwrite:
        obj = joblib.load(cache_path)
        if cache_matches(obj, expected):
            logging.info("Loaded inner MICE cache: %s", cache_path)
            return obj["inner_data"]
        logging.warning("Inner MICE cache metadata mismatch; rebuilding: %s", cache_path)

    X_train = train_df[features].reset_index(drop=True)
    groups = train_df["psu_group"].astype(str).reset_index(drop=True).to_numpy()
    y_train = np.asarray(y_train, dtype=int)
    inner_splits = make_inner_splits(y_train, groups, config.inner_folds, config.seed + fold * 13)
    inner_data = []
    for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        X_inner_train = X_train.iloc[inner_train_idx]
        X_inner_valid = X_train.iloc[inner_valid_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_valid = y_train[inner_valid_idx]
        inner_preprocessors = fit_preprocessors(
            X_inner_train,
            features,
            config.m_imputations,
            config.seed + fold * 10000 + inner_fold * 1000,
            rcs_variables_for(variable_set),
        )
        inner_data.append(
            {
                "inner_fold": inner_fold,
                "train_idx": inner_train_idx,
                "valid_idx": inner_valid_idx,
                "matrices_train": transformed_list(inner_preprocessors, X_inner_train),
                "matrices_valid": transformed_list(inner_preprocessors, X_inner_valid),
                "y_train": y_inner_train,
                "y_valid": y_inner_valid,
            }
        )

    joblib.dump({"metadata": expected, "inner_data": inner_data}, cache_path, compress=3)
    logging.info("Saved inner MICE cache: %s", cache_path)
    return inner_data


def metric_safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan


def metric_safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan


def completed_trial_count(out: Path, model_set: str, variable_set: str, algorithm: str, fold: int) -> int:
    if algorithm not in TUNABLE_ALGORITHMS:
        return 0
    combo = model_combo_id(model_set, variable_set, algorithm, fold)
    storage_path = out / "tuning" / f"{combo}__optuna.sqlite3"
    if storage_path.exists():
        try:
            study = optuna.load_study(study_name=combo, storage=f"sqlite:///{storage_path}")
            return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        except Exception as exc:  # pragma: no cover - checkpoint fallback
            logging.warning("Could not read Optuna storage %s: %s", storage_path, exc)
    trial_path = out / "tuning" / f"{combo}__trials.csv"
    if trial_path.exists():
        trials = pd.read_csv(trial_path)
        if "state" in trials.columns:
            return int((trials["state"] == "COMPLETE").sum())
        return int(len(trials))
    return 0


def tune_algorithm(
    algorithm: str,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    config: Config,
    out: Path,
    model_set: str,
    variable_set: str,
    fold: int,
) -> dict[str, Any]:
    combo = model_combo_id(model_set, variable_set, algorithm, fold)
    trial_path = out / "tuning" / f"{combo}__trials.csv"
    best_path = out / "tuning" / f"{combo}__best.json"
    storage_path = out / "tuning" / f"{combo}__optuna.sqlite3"
    if best_path.exists() and trial_path.exists() and not config.overwrite:
        completed = completed_trial_count(out, model_set, variable_set, algorithm, fold)
        if completed >= config.n_trials:
            return read_json(best_path)

    y_train = np.asarray(y_train, dtype=int)
    inner_data = load_or_create_inner_cache(train_df, y_train, features, config, out, model_set, variable_set, fold)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, algorithm)
        fold_scores = []
        for inner in inner_data:
            models = fit_model_ensemble(
                algorithm,
                params,
                inner["matrices_train"],
                inner["y_train"],
                seed=config.seed + fold * 100000 + int(inner["inner_fold"]) * 1000 + trial.number,
                n_jobs=config.n_jobs,
            )
            pred = mean_predict_from_models(models, inner["matrices_valid"])
            auroc = metric_safe_auc(inner["y_valid"], pred)
            auprc = metric_safe_auprc(inner["y_valid"], pred)
            brier = float(brier_score_loss(inner["y_valid"], pred))
            trial.set_user_attr(f"inner{inner['inner_fold']}_auroc", auroc)
            trial.set_user_attr(f"inner{inner['inner_fold']}_auprc", auprc)
            trial.set_user_attr(f"inner{inner['inner_fold']}_brier", brier)
            fold_scores.append(auroc)
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=config.seed + fold * 101)
    storage = f"sqlite:///{storage_path}"
    study_name = combo
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, study_name=study_name, load_if_exists=True)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining_trials = max(0, config.n_trials - len(completed_trials))
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    write_csv(trials, trial_path)
    best = {
        "algorithm": algorithm,
        "fold": fold,
        "model_set": model_set,
        "variable_set": variable_set,
        "best_value": study.best_value,
        "params": study.best_params,
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "requested_trials": config.n_trials,
    }
    save_json(best, best_path)
    return best


def make_inner_splits(y: np.ndarray, groups: np.ndarray, inner_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2 for inner CV.")
    if len(np.unique(groups)) >= inner_folds:
        splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def fold_cv_complete(out: Path, model_set: str, variable_set: str, algorithm: str, fold: int, config: Config) -> bool:
    combo = model_combo_id(model_set, variable_set, algorithm, fold)
    oof_path = out / "oof_predictions" / f"{combo}.csv"
    cv_model_path = out / "cv_models" / f"{combo}.joblib"
    if not (oof_path.exists() and cv_model_path.exists()):
        return False
    if algorithm not in TUNABLE_ALGORITHMS:
        return True
    return completed_trial_count(out, model_set, variable_set, algorithm, fold) >= config.n_trials


def run_cv_for_algorithm(
    df: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    model_set: str,
    variable_set: str,
    algorithm: str,
    config: Config,
    out: Path,
) -> None:
    if not valid_model_combo(variable_set, algorithm):
        logging.info("Skipping invalid model combination: %s %s", variable_set, algorithm)
        return
    features = predictors_for(model_set, variable_set)
    df_set = subset_for_model_set(df, model_set)
    folds = fold_assignments[["participant_id", "fold"]]
    df_set = df_set.merge(folds, on="participant_id", how="left")
    if df_set["fold"].isna().any():
        raise RuntimeError(f"Missing fold assignment for {model_set}.")

    requested_folds = config.outer_folds or list(range(1, config.n_folds + 1))
    for fold in requested_folds:
        combo = model_combo_id(model_set, variable_set, algorithm, fold)
        oof_path = out / "oof_predictions" / f"{combo}.csv"
        cv_model_path = out / "cv_models" / f"{combo}.joblib"
        if fold_cv_complete(out, model_set, variable_set, algorithm, fold, config) and not config.overwrite:
            logging.info("Skipping completed %s %s %s fold %s", model_set, variable_set, algorithm, fold)
            continue

        t0 = time.time()
        train_df = df_set[df_set["fold"] != fold].copy()
        valid_df = df_set[df_set["fold"] == fold].copy()
        y_train = train_df["undiagnosed_htn"].astype(int).to_numpy()
        y_valid = valid_df["undiagnosed_htn"].astype(int).to_numpy()

        logging.info(
            "Start %s %s %s fold %s: train n=%s valid n=%s events train=%s valid=%s features=%s",
            model_set,
            variable_set,
            algorithm,
            fold,
            len(train_df),
            len(valid_df),
            int(y_train.sum()),
            int(y_valid.sum()),
            len(features),
        )

        outer_cache = load_or_create_outer_cache(train_df, valid_df, features, config, out, model_set, variable_set, fold)
        preprocessors = outer_cache["preprocessors"]
        matrices_train = outer_cache["matrices_train"]
        matrices_valid = outer_cache["matrices_valid"]

        if algorithm in NON_TUNABLE_ALGORITHMS:
            best = {"algorithm": algorithm, "fold": fold, "model_set": model_set, "variable_set": variable_set, "params": {}}
        else:
            best = tune_algorithm(algorithm, train_df, y_train, features, config, out, model_set, variable_set, fold)

        models = fit_model_ensemble(
            algorithm,
            best["params"],
            matrices_train,
            y_train,
            seed=config.seed + fold * 10000,
            n_jobs=config.n_jobs,
        )
        pred = mean_predict_from_models(models, matrices_valid)
        pred_df = valid_df[
            ["participant_id", "source_year", "sex", "weight", "strata", "psu", "fold", "undiagnosed_htn"]
        ].copy()
        pred_df["model_set"] = model_set
        pred_df["variable_set"] = variable_set
        pred_df["algorithm"] = algorithm
        pred_df["pred_raw"] = pred
        write_csv(pred_df, oof_path)
        joblib.dump(
            {
                "algorithm": algorithm,
                "model_set": model_set,
                "variable_set": variable_set,
                "fold": fold,
                "features": features,
                "params": best["params"],
                "models": models,
            },
            cv_model_path,
            compress=3,
        )
        logging.info("Done %s %s %s fold %s in %.1fs", model_set, variable_set, algorithm, fold, time.time() - t0)


def collect_oof(out: Path, model_set: str, variable_set: str, algorithm: str, n_folds: int) -> pd.DataFrame | None:
    paths = [
        out / "oof_predictions" / f"{model_combo_id(model_set, variable_set, algorithm, fold)}.csv"
        for fold in range(1, n_folds + 1)
    ]
    if not all(p.exists() for p in paths):
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    cal = LogitCalibrator().fit(y, p)
    return cal.intercept_, cal.slope_


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else math.nan
    spec = tn / (tn + fp) if (tn + fp) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    return {
        "threshold": float(threshold),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": (sens + spec) / 2.0,
        "referral_rate": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def choose_thresholds(y: np.ndarray, p: np.ndarray, target_sensitivity: float) -> list[dict[str, Any]]:
    thresholds = np.unique(np.r_[np.linspace(0.001, 0.999, 999), p])
    rows = []
    for thr in thresholds:
        met = threshold_metrics(y, p, float(thr))
        rows.append(met)
    table = pd.DataFrame(rows)
    sens_ok = table[table["sensitivity"] >= target_sensitivity]
    if len(sens_ok):
        sens_row = sens_ok.sort_values(["threshold"], ascending=False).iloc[0].to_dict()
    else:
        sens_row = table.iloc[(table["sensitivity"] - target_sensitivity).abs().argmin()].to_dict()
    sens_row["threshold_type"] = f"sensitivity_{target_sensitivity:.2f}"

    table["youden"] = table["sensitivity"] + table["specificity"] - 1
    youden_row = table.sort_values(["youden", "threshold"], ascending=[False, False]).iloc[0].to_dict()
    youden_row["threshold_type"] = "youden"
    return [sens_row, youden_row]


def dca_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    n = len(y)
    prevalence = y.mean()
    rows = []
    for threshold in np.round(np.arange(0.01, 0.501, 0.01), 3):
        pred = p >= threshold
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        net_benefit = tp / n - fp / n * (threshold / (1.0 - threshold))
        treat_all = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": net_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_plot_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), q=n_bins, duplicates="drop")
    table = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=False).agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        observed=("y", "mean"),
    )
    return table.reset_index(drop=True)


def plot_calibration(calib: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.plot(calib["mean_pred"], calib["observed"], marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed outcome rate")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def summarize_candidate(y: np.ndarray, p_raw: np.ndarray, target_sensitivity: float) -> tuple[dict[str, Any], np.ndarray, LogitCalibrator]:
    raw_intercept, raw_slope = calibration_intercept_slope(y, p_raw)
    calibrator = LogitCalibrator().fit(y, p_raw)
    p_cal = calibrator.transform(p_raw)
    cal_intercept, cal_slope = calibration_intercept_slope(y, p_cal)
    dca = dca_table(y, p_cal)
    mean_nb = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
    summary = {
        "auroc_raw": metric_safe_auc(y, p_raw),
        "auprc_raw": metric_safe_auprc(y, p_raw),
        "brier_raw": float(brier_score_loss(y, p_raw)),
        "calibration_intercept_raw": raw_intercept,
        "calibration_slope_raw": raw_slope,
        "auroc_calibrated": metric_safe_auc(y, p_cal),
        "auprc_calibrated": metric_safe_auprc(y, p_cal),
        "brier_calibrated": float(brier_score_loss(y, p_cal)),
        "calibration_intercept_calibrated": cal_intercept,
        "calibration_slope_calibrated": cal_slope,
        "mean_net_benefit_0.05_0.30": mean_nb,
        "calibrator_intercept": calibrator.intercept_,
        "calibrator_slope": calibrator.slope_,
    }
    return summary, p_cal, calibrator


def summarize_with_fixed_calibrator(y: np.ndarray, p_raw: np.ndarray, calibrator: LogitCalibrator) -> tuple[dict[str, Any], np.ndarray]:
    raw_intercept, raw_slope = calibration_intercept_slope(y, p_raw)
    p_cal = calibrator.transform(p_raw)
    cal_intercept, cal_slope = calibration_intercept_slope(y, p_cal)
    dca = dca_table(y, p_cal)
    mean_nb = float(dca[(dca["threshold"] >= 0.05) & (dca["threshold"] <= 0.30)]["net_benefit_model"].mean())
    summary = {
        "auroc_raw": metric_safe_auc(y, p_raw),
        "auprc_raw": metric_safe_auprc(y, p_raw),
        "brier_raw": float(brier_score_loss(y, p_raw)),
        "calibration_intercept_raw": raw_intercept,
        "calibration_slope_raw": raw_slope,
        "auroc_calibrated": metric_safe_auc(y, p_cal),
        "auprc_calibrated": metric_safe_auprc(y, p_cal),
        "brier_calibrated": float(brier_score_loss(y, p_cal)),
        "calibration_intercept_calibrated": cal_intercept,
        "calibration_slope_calibrated": cal_slope,
        "mean_net_benefit_0.05_0.30": mean_nb,
        "calibrator_intercept": calibrator.intercept_,
        "calibrator_slope": calibrator.slope_,
    }
    return summary, p_cal


def rank_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["is_benchmark"] = ranked["algorithm"] == "him_benchmark"
    ranked["slope_error"] = (ranked["calibration_slope_raw"] - 1.0).abs()
    ranked["intercept_abs"] = ranked["calibration_intercept_raw"].abs()
    interpretability = {
        "logistic": 1,
        "lasso": 1,
        "elastic_net": 2,
        "adaboost": 4,
        "random_forest": 5,
        "extra_trees": 5,
        "lightgbm": 6,
        "xgboost": 6,
        "him_benchmark": 1,
    }
    ranked["interpretability_rank"] = ranked["algorithm"].map(interpretability)
    rank_terms = {
        "auroc_calibrated": False,
        "auprc_calibrated": False,
        "brier_calibrated": True,
        "slope_error": True,
        "intercept_abs": True,
        "mean_net_benefit_0.05_0.30": False,
        "interpretability_rank": True,
    }
    weights = {
        "auroc_calibrated": 0.25,
        "auprc_calibrated": 0.15,
        "brier_calibrated": 0.20,
        "slope_error": 0.15,
        "intercept_abs": 0.10,
        "mean_net_benefit_0.05_0.30": 0.10,
        "interpretability_rank": 0.05,
    }
    ranked["selection_score"] = 0.0
    for col, ascending in rank_terms.items():
        rank_col = f"rank_{col}"
        ranked[rank_col] = ranked.groupby("model_set")[col].rank(method="min", ascending=ascending)
        ranked["selection_score"] += ranked[rank_col] * weights[col]
    ranked["selected_within_model_set"] = False
    ranked["selected_within_model_variable_set"] = False
    for model_set, part in ranked.groupby("model_set"):
        part = part[~part["is_benchmark"]]
        if part.empty:
            continue
        idx = part.sort_values(["selection_score", "brier_calibrated", "algorithm"]).index[0]
        ranked.loc[idx, "selected_within_model_set"] = True
    for (model_set, variable_set), part in ranked.groupby(["model_set", "variable_set"]):
        part = part[~part["is_benchmark"]]
        if part.empty:
            continue
        idx = part.sort_values(["selection_score", "brier_calibrated", "algorithm"]).index[0]
        ranked.loc[idx, "selected_within_model_variable_set"] = True
    return ranked.sort_values(["model_set", "variable_set", "selection_score"])


def summarize_oof(config: Config, out: Path) -> pd.DataFrame:
    metric_rows = []
    threshold_rows = []
    for model_set in config.model_sets:
        for variable_set in config.variable_sets:
            for algorithm in config.algorithms:
                if not valid_model_combo(variable_set, algorithm):
                    continue
                combo = model_combo_id(model_set, variable_set, algorithm)
                oof = collect_oof(out, model_set, variable_set, algorithm, config.n_folds)
                if oof is None:
                    logging.warning("OOF incomplete for %s %s %s; skipping summary", model_set, variable_set, algorithm)
                    continue
                y = oof["undiagnosed_htn"].astype(int).to_numpy()
                p_raw = oof["pred_raw"].to_numpy()
                summary, p_cal, calibrator = summarize_candidate(y, p_raw, config.target_sensitivity)
                oof["pred_calibrated"] = p_cal
                write_csv(oof, out / "oof_predictions" / f"{combo}__oof.csv")
                save_json(
                    {"intercept": calibrator.intercept_, "slope": calibrator.slope_},
                    out / "metrics" / f"{combo}__calibrator.json",
                )
                dca = dca_table(y, p_cal)
                dca.insert(0, "algorithm", algorithm)
                dca.insert(0, "variable_set", variable_set)
                dca.insert(0, "model_set", model_set)
                write_csv(dca, out / "metrics" / f"{combo}__dca.csv")
                calib = calibration_plot_table(y, p_cal)
                calib.insert(0, "algorithm", algorithm)
                calib.insert(0, "variable_set", variable_set)
                calib.insert(0, "model_set", model_set)
                write_csv(calib, out / "metrics" / f"{combo}__calibration_plot_data.csv")
                plot_calibration(
                    calib,
                    out / "plots" / f"{combo}__calibration.png",
                    f"{LABELS[model_set]} {LABELS[variable_set]} {LABELS[algorithm]}",
                )
                metric_rows.append(
                    {
                        "model_set": model_set,
                        "variable_set": variable_set,
                        "algorithm": algorithm,
                        "n": len(oof),
                        "events": int(y.sum()),
                        **summary,
                    }
                )
                for row in choose_thresholds(y, p_cal, config.target_sensitivity):
                    threshold_rows.append({"model_set": model_set, "variable_set": variable_set, "algorithm": algorithm, **row})

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No complete OOF predictions found.")
    thresholds = pd.DataFrame(threshold_rows)
    ranked = rank_candidates(metrics)
    write_csv(metrics, out / "metrics" / "oof_metrics_all_models.csv")
    write_csv(thresholds, out / "metrics" / "oof_thresholds_all_models.csv")
    write_csv(ranked, out / "metrics" / "oof_model_selection_ranked.csv")
    return ranked


def selected_candidates(out: Path) -> pd.DataFrame:
    path = out / "metrics" / "oof_model_selection_ranked.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run summarize stage first: {path}")
    ranked = pd.read_csv(path)
    return ranked[ranked["selected_within_model_set"] == True].copy()


def threshold_lookup(out: Path, model_set: str, variable_set: str, algorithm: str) -> pd.DataFrame:
    tbl = pd.read_csv(out / "metrics" / "oof_thresholds_all_models.csv")
    return tbl[
        (tbl["model_set"] == model_set)
        & (tbl["variable_set"] == variable_set)
        & (tbl["algorithm"] == algorithm)
    ].copy()


def calibrator_lookup(out: Path, model_set: str, variable_set: str, algorithm: str) -> LogitCalibrator:
    obj = read_json(out / "metrics" / f"{model_combo_id(model_set, variable_set, algorithm)}__calibrator.json")
    cal = LogitCalibrator()
    cal.intercept_ = float(obj["intercept"])
    cal.slope_ = float(obj["slope"])
    cal.fitted_ = True
    return cal


def final_refit_and_validate(config: Config, out: Path) -> None:
    dev = load_dataset(Path(config.dev_path))
    time_df = load_dataset(Path(config.time_path))
    selected = selected_candidates(out)
    validation_metric_rows = []
    validation_threshold_rows = []

    for _, row in selected.iterrows():
        model_set = row["model_set"]
        variable_set = row["variable_set"]
        algorithm = row["algorithm"]
        combo = model_combo_id(model_set, variable_set, algorithm)
        final_path = out / "final_models" / f"{combo}__final_bundle.joblib"
        features = predictors_for(model_set, variable_set)

        if final_path.exists() and not config.overwrite:
            bundle = joblib.load(final_path)
            logging.info("Loaded existing final bundle: %s", final_path)
        else:
            train_df = subset_for_model_set(dev, model_set)
            X_train = train_df[features]
            y_train = train_df["undiagnosed_htn"].astype(int).to_numpy()
            preprocessors = fit_preprocessors(
                X_train,
                features,
                config.m_imputations,
                config.seed + 9000,
                rcs_variables_for(variable_set),
            )
            matrices_train = transformed_list(preprocessors, X_train)
            params = {}
            best_paths = sorted((out / "tuning").glob(f"{model_set}__{variable_set}__{algorithm}__fold*__best.json"))
            if best_paths:
                # Use the most frequently selected value for categorical/integer params and median for floats.
                params_by_fold = [read_json(p)["params"] for p in best_paths]
                params = aggregate_params(params_by_fold)
            models = fit_model_ensemble(algorithm, params, matrices_train, y_train, config.seed + 9999, config.n_jobs)
            calibrator = calibrator_lookup(out, model_set, variable_set, algorithm)
            thresholds = threshold_lookup(out, model_set, variable_set, algorithm)
            bundle = {
                "model_set": model_set,
                "variable_set": variable_set,
                "algorithm": algorithm,
                "features": features,
                "rcs_variables": rcs_variables_for(variable_set),
                "rcs_knot_probs": list(RCS_KNOT_PROBS),
                "params": params,
                "preprocessors": preprocessors,
                "models": models,
                "calibrator": calibrator,
                "thresholds": thresholds,
                "locked_variables_path": str(LOCKED_VARIABLES_PATH),
            }
            joblib.dump(bundle, final_path, compress=3)
            save_json(
                {
                    "model_set": model_set,
                    "variable_set": variable_set,
                    "algorithm": algorithm,
                    "features": features,
                    "rcs_variables": rcs_variables_for(variable_set),
                    "rcs_knot_probs": list(RCS_KNOT_PROBS),
                    "params": params,
                },
                out / "final_models" / f"{combo}__final_config.json",
            )
            logging.info("Saved final bundle: %s", final_path)

        val_df = subset_for_model_set(time_df, model_set)
        X_val = val_df[bundle["features"]]
        y_val = val_df["undiagnosed_htn"].astype(int).to_numpy()
        matrices_val = transformed_list(bundle["preprocessors"], X_val)
        p_raw = mean_predict_from_models(bundle["models"], matrices_val)
        p_cal = bundle["calibrator"].transform(p_raw)
        pred_out = val_df[["participant_id", "source_year", "sex", "weight", "strata", "psu", "undiagnosed_htn"]].copy()
        pred_out["model_set"] = model_set
        pred_out["variable_set"] = variable_set
        pred_out["algorithm"] = algorithm
        pred_out["pred_raw"] = p_raw
        pred_out["pred_calibrated"] = p_cal
        write_csv(pred_out, out / "time_validation" / f"{combo}__2021_2022_predictions.csv")

        summary, _ = summarize_with_fixed_calibrator(y_val, p_raw, bundle["calibrator"])
        validation_metric_rows.append(
            {"model_set": model_set, "variable_set": variable_set, "algorithm": algorithm, "n": len(val_df), "events": int(y_val.sum()), **summary}
        )
        thresholds = bundle["thresholds"]
        for _, thr_row in thresholds.iterrows():
            met = threshold_metrics(y_val, p_cal, float(thr_row["threshold"]))
            validation_threshold_rows.append(
                {
                    "model_set": model_set,
                    "variable_set": variable_set,
                    "algorithm": algorithm,
                    "threshold_type": thr_row["threshold_type"],
                    "development_threshold": float(thr_row["threshold"]),
                    **met,
                }
            )
        dca = dca_table(y_val, p_cal)
        dca.insert(0, "algorithm", algorithm)
        dca.insert(0, "variable_set", variable_set)
        dca.insert(0, "model_set", model_set)
        write_csv(dca, out / "time_validation" / f"{combo}__2021_2022_dca.csv")

    write_csv(pd.DataFrame(validation_metric_rows), out / "time_validation" / "time_validation_metrics_selected_models.csv")
    write_csv(pd.DataFrame(validation_threshold_rows), out / "time_validation" / "time_validation_threshold_metrics_selected_models.csv")


def aggregate_params(params_by_fold: list[dict[str, Any]]) -> dict[str, Any]:
    if not params_by_fold:
        return {}
    keys = sorted({key for params in params_by_fold for key in params})
    out = {}
    for key in keys:
        values = [params[key] for params in params_by_fold if key in params]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            if all(float(v).is_integer() for v in values):
                out[key] = int(round(float(np.median(values))))
            else:
                out[key] = float(np.median(values))
        else:
            out[key] = pd.Series(values, dtype=object).mode(dropna=False).iloc[0]
    return out


def write_run_readme(config: Config, out: Path) -> None:
    text = f"""# KNHANES Primary ML Pipeline Run

This folder contains checkpointed outputs for the locked primary model input.

## Configuration

- Development input: `{config.dev_path}`
- Temporal input: `{config.time_path}`
- Seed: {config.seed}
- MICE imputations: {config.m_imputations}
- CV folds: {config.n_folds}
- Inner tuning CV folds: {config.inner_folds}
- Optuna trials per tunable model/fold: {config.n_trials}
- Target sensitivity threshold rule: {config.target_sensitivity}
- Analysis populations: {', '.join(config.model_sets)}
- Variable sets: {', '.join(config.variable_sets)}
- Algorithms: {', '.join(config.algorithms)}
- Configured candidate grid: requested analysis populations x requested predictor sets x requested algorithms
- Benchmark: Hypertension Index non-invasive subset with untuned logistic model only
- Outer folds requested: {'all' if config.outer_folds is None else ', '.join(map(str, config.outer_folds))}

## Key Artifacts

- Fold assignments: `folds/fold_assignments_{config.n_folds}fold_seed{config.seed}.csv`
- OOF predictions: `oof_predictions/`
- Inner MICE caches: `inner_cache/`
- Trial logs and Optuna SQLite checkpoints: `tuning/`
- OOF metrics: `metrics/oof_metrics_all_models.csv`
- Thresholds: `metrics/oof_thresholds_all_models.csv`
- Ranked selection: `metrics/oof_model_selection_ranked.csv`
- Final frozen bundles: `final_models/`
- Temporal validation: `time_validation/`
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def dry_run(config: Config, out: Path) -> None:
    dev = load_dataset(Path(config.dev_path))
    folds = create_or_load_folds(dev, config, out)
    fold_balance = pd.read_csv(out / "folds" / "fold_balance.csv")
    logging.info("Dry run complete. Development shape=%s", dev.shape)
    logging.info("Fold balance:\n%s", fold_balance.to_string(index=False))


def run_cv(config: Config, out: Path) -> None:
    dev = load_dataset(Path(config.dev_path))
    fold_assignments = create_or_load_folds(dev, config, out)
    for model_set in config.model_sets:
        for variable_set in config.variable_sets:
            for algorithm in config.algorithms:
                run_cv_for_algorithm(dev, fold_assignments, model_set, variable_set, algorithm, config, out)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run KNHANES primary ML pipeline.")
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "knhanes_primary_ml_v1"))
    parser.add_argument("--dev-path", default=str(DEV_PATH))
    parser.add_argument("--time-path", default=str(TIME_PATH))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--m-imputations", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--stage", choices=["dry-run", "cv", "summarize", "finalize", "all"], default="cv")
    parser.add_argument("--model-sets", nargs="+", choices=MODEL_SETS, default=MODEL_SETS)
    parser.add_argument("--variable-sets", nargs="+", choices=VARIABLE_SETS, default=DEFAULT_VARIABLE_SETS)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--outer-folds", nargs="+", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    out = Path(config.output_dir)
    ensure_dirs(out)
    setup_logging(out)
    save_json(asdict(config), out / "config" / "pipeline_config.json")
    write_run_readme(config, out)
    logging.info("Pipeline config: %s", asdict(config))

    if config.stage == "dry-run":
        dry_run(config, out)
        return
    if config.stage in {"cv", "all"}:
        run_cv(config, out)
    if config.stage in {"summarize", "all"}:
        summarize_oof(config, out)
    if config.stage in {"finalize", "all"}:
        final_refit_and_validate(config, out)


if __name__ == "__main__":
    main()
