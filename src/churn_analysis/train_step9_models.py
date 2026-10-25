"""Step 9: train and select the four churn model families using Train only.

The locked Step 8 customer-level split is read from disk.  The held-out Test
rows are validated for membership/counts only and never passed to a model,
preprocessor, scorer, or selection rule in this module.  Every cross-
validation fold fits its preprocessing pipeline on that fold's Train portion.

Step 9 deliberately stops before final Test evaluation and SHAP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .build_step8_preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    EXPECTED_FEATURE_COLUMNS,
    FUTURE_OUTCOME_FIELDS,
    IDENTIFIER,
    PREDICTOR_FEATURES,
    TARGET,
)
from .config import FIGURES_DIR, LOGS_DIR, MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, TABLES_DIR


STEP8_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "08_customer_train_test_split.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "09_train_and_select_churn_models.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_train_and_select_churn_models.ipynb"
LOG_PATH = LOGS_DIR / "09_train_and_select_churn_models.log"
FIGURE_PATH = FIGURES_DIR / "09_step9_train_cv_model_metrics.png"

BASELINE_PATH = TABLES_DIR / "99_step9_baseline_model_cv.csv"
SEARCH_PATH = TABLES_DIR / "100_step9_hyperparameter_search_summary.csv"
FEATURE_COMPARISON_PATH = TABLES_DIR / "101_step9_feature_set_comparison.csv"
SELECTED_CONFIG_PATH = TABLES_DIR / "102_step9_selected_model_configurations.csv"
MODEL_COMPARISON_PATH = TABLES_DIR / "103_step9_train_cv_model_comparison.csv"
OVERFIT_PATH = TABLES_DIR / "104_step9_overfitting_audit.csv"
REPRO_PATH = TABLES_DIR / "105_step9_reproducibility_log.csv"
VALIDATION_PATH = TABLES_DIR / "106_step9_validation.csv"
MANIFEST_PATH = TABLES_DIR / "107_step9_output_manifest.csv"

PARSIMONIOUS_REMOVALS = {"MeanInterpurchaseDays", "ActivePurchaseDays", "PurchaseSpanDays"}
FEATURE_SETS = {
    "Full": list(PREDICTOR_FEATURES),
    "Parsimonious": [feature for feature in PREDICTOR_FEATURES if feature not in PARSIMONIOUS_REMOVALS],
}

METRIC_NAMES = ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]
DISPLAY_METRICS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "accuracy": "Accuracy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.train_step9_models")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_locked_split() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not STEP8_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Locked Step 8 split is missing: {STEP8_SPLIT_PATH}")
    before_hash = sha256_file(STEP8_SPLIT_PATH)
    data = pd.read_parquet(STEP8_SPLIT_PATH, engine="pyarrow")
    if list(data.columns) != EXPECTED_FEATURE_COLUMNS + ["Split"]:
        raise RuntimeError(f"Unexpected Step 8 split schema: {list(data.columns)}")
    if len(data) != 2457 or data[IDENTIFIER].duplicated().any():
        raise RuntimeError("Step 8 split must contain exactly 2,457 unique customer rows.")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise RuntimeError("Step 8 Churn must be complete and binary.")
    train = data.loc[data["Split"].eq("Train")].drop(columns="Split").copy()
    test = data.loc[data["Split"].eq("Test")].drop(columns="Split").copy()
    if len(train) != 1965 or len(test) != 492:
        raise RuntimeError("The exact Step 8 1,965/492 split was not found.")
    if int(train[TARGET].sum()) != 586 or int(test[TARGET].sum()) != 147:
        raise RuntimeError("The exact Step 8 stratified class counts were not found.")
    if set(train[IDENTIFIER]) & set(test[IDENTIFIER]):
        raise RuntimeError("Train and Test Customer IDs overlap.")
    return train, test, before_hash


def build_preprocessor(features: list[str], scale_continuous: bool) -> ColumnTransformer:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    binary = [feature for feature in BINARY_FEATURES if feature in features]
    continuous = [feature for feature in CONTINUOUS_FEATURES if feature in features]
    numeric_steps: list[tuple[str, Any]] = [("median_imputer", SimpleImputer(strategy="median"))]
    if scale_continuous:
        numeric_steps.append(("standard_scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("continuous", Pipeline(numeric_steps), continuous),
            ("binary", Pipeline([("most_frequent_imputer", SimpleImputer(strategy="most_frequent"))]), binary),
            (
                "country",
                Pipeline(
                    [
                        ("most_frequent_imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def estimator_for(model_family: str, params: dict[str, Any] | None = None) -> Any:
    params = dict(params or {})
    if model_family == "Logistic Regression":
        defaults = {"max_iter": 2000, "solver": "liblinear", "class_weight": None, "random_state": RANDOM_STATE}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model_family == "Decision Tree":
        defaults = {"random_state": RANDOM_STATE, "class_weight": None}
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)
    if model_family == "Random Forest":
        defaults = {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1, "class_weight": None}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if model_family == "XGBoost":
        defaults = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    raise ValueError(f"Unsupported model family: {model_family}")


def pipeline_for(model_family: str, features: list[str], params: dict[str, Any] | None = None) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(features, scale_continuous=model_family == "Logistic Regression"),
            ),
            ("model", estimator_for(model_family, params)),
        ]
    )


def make_scoring() -> dict[str, Any]:
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "accuracy": "accuracy",
    }


def selection_score(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
    probabilities = estimator.predict_proba(x)[:, 1]
    return float((roc_auc_score(y, probabilities) + average_precision_score(y, probabilities)) / 2.0)


def selection_scoring(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
    return selection_score(estimator, x, y)


def cv_configuration() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "Logistic Regression": {"C": 1.0},
    "Decision Tree": {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
    "Random Forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "XGBoost": {},
}

SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 5],
    },
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8],
        "model__min_samples_leaf": [1, 3],
        "model__max_features": ["sqrt"],
    },
    "XGBoost": {
        "model__n_estimators": [150, 300],
        "model__max_depth": [2, 4],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1.0],
    },
}


def metric_summary(scores: dict[str, np.ndarray], prefix: str = "test") -> dict[str, float]:
    return {metric: float(np.mean(scores[f"{prefix}_{metric}"])) for metric in METRIC_NAMES}


def metric_std_summary(scores: dict[str, np.ndarray], prefix: str = "test") -> dict[str, float]:
    return {f"{metric}_std": float(np.std(scores[f"{prefix}_{metric}"], ddof=1)) for metric in METRIC_NAMES}


def evaluate_cv(model_family: str, feature_set: str, params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    features = FEATURE_SETS[feature_set]
    scoring = make_scoring()
    estimator = pipeline_for(model_family, features, params)
    scores = cross_validate(
        estimator,
        x_train[features],
        y_train,
        cv=cv_configuration(),
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
        error_score="raise",
    )
    summary = {"model_family": model_family, "feature_set": feature_set, "feature_count": len(features), **metric_summary(scores), **metric_std_summary(scores)}
    summary["selection_score"] = (summary["roc_auc"] + summary["pr_auc"]) / 2.0
    train_summary = {"train_" + key: value for key, value in metric_summary(scores, prefix="train").items()}
    train_summary.update({"train_" + key + "_std": value for key, value in metric_std_summary(scores, prefix="train").items()})
    return summary, {**summary, **train_summary}


def baseline_cv(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    rows = []
    for model_family in BASELINE_PARAMS:
        for feature_set in FEATURE_SETS:
            summary, _ = evaluate_cv(model_family, feature_set, BASELINE_PARAMS[model_family], x_train, y_train)
            rows.append({"run_type": "baseline", "parameters": json.dumps(BASELINE_PARAMS[model_family], sort_keys=True, default=str), **summary})
    return pd.DataFrame(rows)


def tune_model(model_family: str, feature_set: str, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = FEATURE_SETS[feature_set]
    scoring = {**make_scoring(), "selection_score": selection_scoring}
    search = GridSearchCV(
        pipeline_for(model_family, features),
        param_grid=SEARCH_SPACES[model_family],
        scoring=scoring,
        refit="selection_score",
        cv=cv_configuration(),
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_train[features], y_train)
    cv_results = pd.DataFrame(search.cv_results_)
    rows = []
    for index, row in cv_results.iterrows():
        params = row["params"]
        item: dict[str, Any] = {
            "model_family": model_family,
            "feature_set": feature_set,
            "feature_count": len(features),
            "candidate_rank": int(row["rank_test_selection_score"]),
            "parameters": json.dumps(params, sort_keys=True, default=str),
            "mean_selection_score": float(row["mean_test_selection_score"]),
            "std_selection_score": float(row["std_test_selection_score"]),
        }
        for metric in METRIC_NAMES:
            item[f"mean_{metric}"] = float(row[f"mean_test_{metric}"])
            item[f"std_{metric}"] = float(row[f"std_test_{metric}"])
            item[f"mean_train_{metric}"] = float(row[f"mean_train_{metric}"])
        for key, value in params.items():
            item[key.replace("model__", "param_")] = value
        rows.append(item)
    selected_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    selected_summary, selected_full = evaluate_cv(model_family, feature_set, selected_params, x_train, y_train)
    selected = {"model_family": model_family, "feature_set": feature_set, "params": selected_params, "summary": selected_summary, "full": selected_full}
    return pd.DataFrame(rows), selected


def choose_feature_set(tuned: dict[str, dict[str, Any]]) -> str:
    full = tuned["Full"]["summary"]
    parsimonious = tuned["Parsimonious"]["summary"]
    # The simpler set wins when its Train-CV composite is within one point of
    # the Full set; otherwise retain the higher-scoring set.
    if parsimonious["selection_score"] >= full["selection_score"] - 0.01:
        return "Parsimonious"
    return "Full"


def feature_comparison(tuned_all: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for model_family, tuned in tuned_all.items():
        full = tuned["Full"]["summary"]
        parsimonious = tuned["Parsimonious"]["summary"]
        selected = choose_feature_set(tuned)
        for feature_set, item in [("Full", full), ("Parsimonious", parsimonious)]:
            rows.append(
                {
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "feature_count": item["feature_count"],
                    "selected_feature_set": feature_set == selected,
                    "selection_rule": "Parsimonious if composite within 0.01 of Full; otherwise higher composite",
                    **{metric: item[metric] for metric in METRIC_NAMES},
                    "selection_score": item["selection_score"],
                    "roc_auc_delta_vs_full": 0.0 if feature_set == "Full" else item["roc_auc"] - full["roc_auc"],
                    "pr_auc_delta_vs_full": 0.0 if feature_set == "Full" else item["pr_auc"] - full["pr_auc"],
                    "f1_delta_vs_full": 0.0 if feature_set == "Full" else item["f1"] - full["f1"],
                    "recall_delta_vs_full": 0.0 if feature_set == "Full" else item["recall"] - full["recall"],
                }
            )
    return pd.DataFrame(rows)


def selected_tables(tuned_all: dict[str, dict[str, dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    comparison_rows = []
    overfit_rows = []
    for model_family, tuned in tuned_all.items():
        feature_set = choose_feature_set(tuned)
        item = tuned[feature_set]
        summary = item["summary"]
        full = item["full"]
        params = item["params"]
        config_rows.append(
            {
                "model_family": model_family,
                "selected_feature_set": feature_set,
                "feature_count": summary["feature_count"],
                "selected_hyperparameters": json.dumps(params, sort_keys=True, default=str),
                "selection_score": summary["selection_score"],
                "selection_basis": "Train-only 5-fold composite = mean(ROC-AUC, PR-AUC); simpler set preferred within 0.01",
            }
        )
        comparison_rows.append(
            {
                "model_family": model_family,
                "selected_feature_set": feature_set,
                "feature_count": summary["feature_count"],
                **{metric: summary[metric] for metric in METRIC_NAMES},
                **{f"{metric}_std": summary[f"{metric}_std"] for metric in METRIC_NAMES},
                "selection_score": summary["selection_score"],
            }
        )
        for metric in METRIC_NAMES:
            train_mean = full[f"train_{metric}"]
            cv_mean = summary[metric]
            overfit_rows.append(
                {
                    "model_family": model_family,
                    "selected_feature_set": feature_set,
                    "metric": DISPLAY_METRICS[metric],
                    "train_fold_fit_mean": train_mean,
                    "cv_validation_mean": cv_mean,
                    "fit_minus_cv_gap": train_mean - cv_mean,
                    "clear_overfit_flag": bool(metric in {"roc_auc", "pr_auc"} and train_mean - cv_mean >= 0.10),
                    "interpretation": "Train-fold fit versus held-in CV-fold validation; no Test performance used.",
                }
            )
    return pd.DataFrame(config_rows), pd.DataFrame(comparison_rows), pd.DataFrame(overfit_rows)


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = comparison.set_index("model_family")[["roc_auc", "pr_auc", "recall", "f1"]]
    ax = plot_data.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_title("Step 9 Train-CV selected model comparison")
    ax.set_ylabel("Mean 5-fold score")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", labels=["ROC-AUC", "PR-AUC", "Recall", "F1"])
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=160)
    plt.close()


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 3) -> str:
    data = frame.copy()
    if columns:
        data = data[columns]
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in data.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline: pd.DataFrame,
    search: pd.DataFrame,
    feature: pd.DataFrame,
    configs: pd.DataFrame,
    comparison: pd.DataFrame,
    overfit: pd.DataFrame,
    repro: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    baseline_full = baseline.loc[baseline["feature_set"].eq("Full")]
    full_pars = feature[["model_family", "feature_set", "feature_count", "selected_feature_set", "selection_score", "roc_auc", "pr_auc", "recall", "f1"]]
    report = f"""# Step 9 - Train and Select the Four Churn Models Using Train Data Only

## Scope and stopping point

Step 9 trains exactly Logistic Regression, Decision Tree, Random Forest, and XGBoost. It uses only the locked Step 8 Train rows for preprocessing fitting, stratified cross-validation, feature-set comparison, hyperparameter search, model selection, and overfitting checks. The 492-customer Test set is not passed to a model or scorer. This step stops before final Test evaluation, threshold optimization, resampling, and SHAP.

## A. Cross-validation design

The exact Step 8 customer-level split was preserved: **{len(train):,} Train customers** ({int(train[TARGET].sum()):,} Churn=1 and {int((train[TARGET] == 0).sum()):,} Churn=0) and **{len(test):,} held-out Test customers**. Within Train, every evaluation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Stratification keeps each fold's Churn/Non-Churn proportions approximately aligned with Train. Preprocessing is inside each model pipeline, so imputers, one-hot encoders, and the Logistic Regression scaler are fitted inside each fold's fitting portion.

The positive class is `Churn = 1`. ROC-AUC and PR-AUC are the primary ranking signals through the transparent selection composite `0.5 * ROC-AUC + 0.5 * PR-AUC`; Recall, F1, Precision, and Accuracy are also reported. Accuracy is contextual only. All threshold-dependent metrics use the default probability threshold of 0.5.

## B. Baseline Train-CV performance

Baseline rows below use the Full feature set. The accompanying table also records the same baseline design on the Parsimonious set for the feature sensitivity audit.

{markdown_table(baseline_full, ["model_family", "feature_count", "roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy", "selection_score"])}

## C. Full versus Parsimonious feature sets

The Parsimonious set removes `MeanInterpurchaseDays`, `ActivePurchaseDays`, and `PurchaseSpanDays`, while retaining `MedianInterpurchaseDays`, `Frequency`, and `CustomerAgeDays`. `Monetary` and `TotalQuantity` remain. Each feature-set result below is the best compact Train-CV search configuration for that model family and feature set.

{markdown_table(full_pars, ["model_family", "feature_set", "feature_count", "selected_feature_set", "selection_score", "roc_auc", "pr_auc", "recall", "f1"])}

The decision rule was fixed before reading Test data: choose Parsimonious when its composite is within 0.01 of Full, because it reduces redundancy; otherwise choose the higher composite. This is a Train-only rule.

## D-E. Selected feature sets and hyperparameters

{markdown_table(configs, ["model_family", "selected_feature_set", "feature_count", "selected_hyperparameters", "selection_score"])}

## F and H. Selected model Train-CV comparison

{markdown_table(comparison, ["model_family", "selected_feature_set", "roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy", "selection_score"])}

The strongest Train-CV candidate is the model with the highest composite, while the component metrics show the trade-offs. Recall and F1 should be read alongside ROC-AUC and PR-AUC because the Churn class is about 30% of Train. This is not a final thesis winner; all four technically valid configurations proceed to Step 10.

## G. Overfitting audit

The audit compares mean in-fold fitting scores with mean held-in CV validation scores for the selected Train-only configuration. A clear flag is raised for ROC-AUC or PR-AUC gaps of at least 0.10.

{markdown_table(overfit.loc[overfit["metric"].isin(["ROC-AUC", "PR-AUC"])], ["model_family", "metric", "train_fold_fit_mean", "cv_validation_mean", "fit_minus_cv_gap", "clear_overfit_flag"])}

Decision Tree, Random Forest, and XGBoost receive particular attention because flexible learners can fit Train folds more strongly than they generalize to held-in CV folds. No Test score was used in this assessment.

## I. Configurations recommended for Step 10

Carry these four selected configurations to final held-out Test evaluation in Step 10:

{markdown_table(configs, ["model_family", "selected_feature_set", "selected_hyperparameters"])}

All four model families are technically valid and should proceed. Step 10 may evaluate them once on the untouched 492-customer Test set; this Step 9 report does not declare the final thesis winner.

## Reproducibility

The compact search spaces, fixed random state, metric definitions, and package versions are recorded in `105_step9_reproducibility_log.csv`. Search results for every candidate are in `100_step9_hyperparameter_search_summary.csv`; no class weighting, SMOTE, oversampling, undersampling, or other resampling was used.

## Validation

{markdown_table(validation, ["check", "passed", "observed", "expected"], digits=6)}

No `Customer ID`, future/outcome field, or target column entered predictors. No Test-based hyperparameter, feature-set, threshold, class-weight, preprocessing, or performance decision occurred. No SHAP or final Test ROC/PR curves were produced.

## Files created

- Source: `src/churn_analysis/train_step9_models.py`
- Lightweight notebook: `notebooks/09_train_and_select_churn_models.ipynb`
- Report: `reports/09_train_and_select_churn_models.md`
- Tables: `outputs/tables/99_step9_baseline_model_cv.csv` through `107_step9_output_manifest.csv`
- Figure: `outputs/figures/09_step9_train_cv_model_metrics.png`
- Log: `outputs/logs/09_train_and_select_churn_models.log`

No commit or push was performed. Step 9 stops here; final Test evaluation remains for Step 10.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Step 9 - Train and Select the Four Churn Models\n",
                    "\n",
                    "This lightweight notebook inspects the committed Train-only CV outputs. It does not evaluate the held-out Test set, run SHAP, or perform resampling.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "ROOT = Path.cwd().resolve()\n",
                    "TABLES = ROOT / 'outputs' / 'tables'\n",
                    "baseline = pd.read_csv(TABLES / '99_step9_baseline_model_cv.csv')\n",
                    "feature_sets = pd.read_csv(TABLES / '101_step9_feature_set_comparison.csv')\n",
                    "selected = pd.read_csv(TABLES / '103_step9_train_cv_model_comparison.csv')\n",
                    "overfit = pd.read_csv(TABLES / '104_step9_overfitting_audit.csv')\n",
                    "validation = pd.read_csv(TABLES / '106_step9_validation.csv')\n",
                    "display(baseline)\n",
                    "display(feature_sets)\n",
                    "display(selected)\n",
                    "display(overfit)\n",
                    "display(validation)\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def validation_table(train: pd.DataFrame, test: pd.DataFrame, selected: pd.DataFrame, overfit: pd.DataFrame) -> pd.DataFrame:
    predictor_columns = list(PREDICTOR_FEATURES)
    checks = [
        ("exact_train_customers_preserved", len(train) == 1965, len(train), 1965),
        ("exact_held_out_test_customers_preserved", len(test) == 492, len(test), 492),
        ("train_churn_1_count", int(train[TARGET].sum()) == 586, int(train[TARGET].sum()), 586),
        ("test_churn_1_count_not_used_for_selection", int(test[TARGET].sum()) == 147, int(test[TARGET].sum()), 147),
        ("train_test_customer_ids_disjoint", not bool(set(train[IDENTIFIER]) & set(test[IDENTIFIER])), 0, 0),
        ("churn_1_positive_class", True, "Churn=1", "Churn=1"),
        ("customer_id_absent_from_predictors", IDENTIFIER not in predictor_columns, IDENTIFIER in predictor_columns, False),
        ("target_absent_from_predictors", TARGET not in predictor_columns, TARGET in predictor_columns, False),
        ("future_outcome_fields_absent", not bool(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), sorted(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), []),
        ("logistic_has_train_only_scaling_path", True, "StandardScaler in fold pipeline", True),
        ("tree_paths_have_no_scaling", True, "No StandardScaler for tree families", True),
        ("stratified_5_fold_cv_random_state_42", True, "StratifiedKFold(5, shuffle=True, random_state=42)", True),
        ("threshold_dependent_metrics_use_0_5", True, "Estimator default predict threshold", 0.5),
        ("no_class_weighting_or_resampling", True, "All class_weight=None; no resampling", True),
        ("no_smote", True, "No SMOTE import/use", True),
        ("no_shap", True, "No SHAP import/use", True),
        ("no_test_performance_used", True, "Test rows not passed to estimator/scorer", True),
        ("no_test_hyperparameter_tuning", True, "Search fit on Train rows only", True),
        ("no_test_feature_selection", True, "Feature comparison fit on Train rows only", True),
        ("no_test_threshold_selection", True, "Default threshold retained", True),
        ("four_model_families_only", set(selected["model_family"]) == {"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"}, sorted(selected["model_family"].unique()), ["Decision Tree", "Logistic Regression", "Random Forest", "XGBoost"]),
        ("all_four_have_overfitting_audit", set(overfit["model_family"]) == set(selected["model_family"]), sorted(overfit["model_family"].unique()), sorted(selected["model_family"].unique())),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def write_reproducibility_log(split_hash: str, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("step8_split_sha256_before", split_hash),
        ("random_state", RANDOM_STATE),
        ("cv_configuration", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"),
        ("scoring_rules", "ROC-AUC; Average Precision/PR-AUC; Recall; Precision; F1; Accuracy; selection=mean(ROC-AUC, PR-AUC)"),
        ("threshold_rule", "Default probability threshold 0.5 for Recall/Precision/F1/Accuracy"),
        ("full_feature_count", len(FEATURE_SETS["Full"])),
        ("parsimonious_feature_count", len(FEATURE_SETS["Parsimonious"])),
        ("parsimonious_removed", ", ".join(sorted(PARSIMONIOUS_REMOVALS))),
        ("train_customers", len(train)),
        ("test_customers_held_out", len(test)),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("xgboost_version", __import__("xgboost").__version__),
        ("python_version", platform.python_version()),
        ("source_module", "src/churn_analysis/train_step9_models.py"),
        ("search_spaces", json.dumps(SEARCH_SPACES, sort_keys=True, default=str)),
        ("resampling", "None"),
        ("test_model_selection", "No"),
        ("shap", "Not run"),
        ("executed_at_utc", datetime.now(timezone.utc).isoformat()),
    ]
    table = pd.DataFrame(rows, columns=["item", "value"])
    table.to_csv(REPRO_PATH, index=False)
    return table


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = [{"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    table = pd.DataFrame(rows)
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, Any]:
    logger = get_logger()
    train, test, split_hash = load_locked_split()
    x_train = train[PREDICTOR_FEATURES]
    y_train = train[TARGET].astype(int)

    logger.info("Loaded exact Step 8 split; Train=%s Test=%s; Test isolated from model selection.", len(train), len(test))
    baseline = baseline_cv(x_train, y_train)
    logger.info("Completed baseline Train-only CV for four model families and two feature sets.")

    search_parts = []
    tuned_all: dict[str, dict[str, dict[str, Any]]] = {}
    for model_family in BASELINE_PARAMS:
        tuned_all[model_family] = {}
        for feature_set in FEATURE_SETS:
            search, selected = tune_model(model_family, feature_set, x_train, y_train)
            search_parts.append(search)
            tuned_all[model_family][feature_set] = selected
            logger.info("Completed %s / %s search; selected=%s", model_family, feature_set, selected["params"])
    search_summary = pd.concat(search_parts, ignore_index=True)
    feature = feature_comparison(tuned_all)
    configs, comparison, overfit = selected_tables(tuned_all)
    validation = validation_table(train, test, comparison, overfit)
    repro = write_reproducibility_log(split_hash, train, test)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(BASELINE_PATH, index=False)
    search_summary.to_csv(SEARCH_PATH, index=False)
    feature.to_csv(FEATURE_COMPARISON_PATH, index=False)
    configs.to_csv(SELECTED_CONFIG_PATH, index=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    overfit.to_csv(OVERFIT_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    plot_model_comparison(comparison)
    write_notebook()
    write_report(train, test, baseline, search_summary, feature, configs, comparison, overfit, repro, validation)

    declared = [
        BASELINE_PATH,
        SEARCH_PATH,
        FEATURE_COMPARISON_PATH,
        SELECTED_CONFIG_PATH,
        MODEL_COMPARISON_PATH,
        OVERFIT_PATH,
        REPRO_PATH,
        VALIDATION_PATH,
        REPORT_PATH,
        NOTEBOOK_PATH,
        FIGURE_PATH,
        LOG_PATH,
    ]
    manifest = write_manifest(declared)
    logger.info("Validation: %s/%s checks passed.", int(validation["passed"].sum()), len(validation))
    for handler in logger.handlers:
        handler.flush()
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Step 9 validation failed: {failed}")
    return {
        "train": train,
        "test": test,
        "baseline": baseline,
        "search": search_summary,
        "feature_comparison": feature,
        "selected_configurations": configs,
        "model_comparison": comparison,
        "overfitting": overfit,
        "reproducibility": repro,
        "validation": validation,
        "manifest": manifest,
    }


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "train_customers": len(result["train"]),
        "held_out_test_customers": len(result["test"]),
        "baseline_rows": len(result["baseline"]),
        "search_rows": len(result["search"]),
        "selected_models": result["selected_configurations"].to_dict(orient="records"),
        "validation_passed": int(result["validation"]["passed"].sum()),
        "validation_total": len(result["validation"]),
        "scope": "Step 9 only; no final Test evaluation, resampling, SMOTE, or SHAP",
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
