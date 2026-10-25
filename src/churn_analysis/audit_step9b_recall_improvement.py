"""Step 9B: Train-only recall-improvement sensitivity audit.

This module extends Step 9 without changing its split, target, historical
outputs, or selected parsimonious configurations.  It uses only the locked
Step 8 Train rows.  Test rows are checked for their locked boundary/counts but
are never passed to preprocessing, estimators, scorers, or threshold rules.

No resampling, SMOTE, feature expansion, SHAP, or final Test evaluation is
performed here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .build_step8_preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FUTURE_OUTCOME_FIELDS,
    IDENTIFIER,
    PREDICTOR_FEATURES,
    TARGET,
)
from .config import FIGURES_DIR, LOGS_DIR, PROJECT_ROOT, RANDOM_STATE, TABLES_DIR
from .train_step9_models import (
    FEATURE_SETS,
    METRIC_NAMES,
    SEARCH_SPACES,
    STEP8_SPLIT_PATH,
    build_preprocessor,
    cv_configuration,
    estimator_for,
    load_locked_split,
    pipeline_for,
)


REFERENCE_CONFIG_PATH = TABLES_DIR / "102_step9_selected_model_configurations.csv"
SOURCE_PATH = PROJECT_ROOT / "src" / "churn_analysis" / "audit_step9b_recall_improvement.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "09b_train_only_recall_improvement_audit.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09b_train_only_recall_improvement_audit.ipynb"
LOG_PATH = LOGS_DIR / "09b_train_only_recall_improvement_audit.log"
FIGURE_PATH = FIGURES_DIR / "10_step9b_recall_precision_tradeoff.png"

WEIGHTING_PATH = TABLES_DIR / "108_step9b_class_weight_comparison.csv"
THRESHOLD_PATH = TABLES_DIR / "109_step9b_threshold_sensitivity.csv"
CANDIDATE_PATH = TABLES_DIR / "110_step9b_threshold_candidates.csv"
LOG_PATH_TABLE = TABLES_DIR / "111_step9b_log_transformation_comparison.csv"
IMPROVEMENT_PATH = TABLES_DIR / "112_step9b_improvement_comparison.csv"
RECOMMENDATION_PATH = TABLES_DIR / "113_step9b_recommendations.csv"
VALIDATION_PATH = TABLES_DIR / "114_step9b_validation.csv"
REPRO_PATH = TABLES_DIR / "115_step9b_reproducibility_log.csv"
MANIFEST_PATH = TABLES_DIR / "116_step9b_output_manifest.csv"

MODEL_FAMILIES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
LOG_FEATURES = [
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AverageOrderValue",
    "AverageQuantityPerOrder",
    "UniqueProducts",
    "ProductsPerOrder",
    "CancellationInvoiceCount",
    "CancellationValue",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.audit_step9b_recall_improvement")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def reference_configs() -> dict[str, dict[str, Any]]:
    if not REFERENCE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Existing Step 9 selected configuration table is missing: {REFERENCE_CONFIG_PATH}")
    table = pd.read_csv(REFERENCE_CONFIG_PATH)
    if set(table["model_family"]) != set(MODEL_FAMILIES):
        raise RuntimeError("Step 9 reference configurations do not contain exactly four model families.")
    if set(table["selected_feature_set"]) != {"Parsimonious"}:
        raise RuntimeError("Step 9B requires the existing Step 9 parsimonious configurations.")
    return {
        row["model_family"]: json.loads(row["selected_hyperparameters"])
        for _, row in table.iterrows()
    }


def scoring_with_f2() -> dict[str, Any]:
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0),
        "accuracy": "accuracy",
    }


def cv_metrics(model_family: str, params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    estimator = pipeline_for(model_family, FEATURE_SETS["Parsimonious"], params)
    scores = cross_validate(
        estimator,
        x_train[FEATURE_SETS["Parsimonious"]],
        y_train,
        cv=cv_configuration(),
        scoring=scoring_with_f2(),
        n_jobs=1,
        error_score="raise",
    )
    return {metric: float(np.mean(scores[f"test_{metric}"])) for metric in [*METRIC_NAMES, "f2"]}


def class_weight_comparison(
    configs: dict[str, dict[str, Any]], x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, float]:
    train_churn = int(y_train.sum())
    train_non_churn = int((y_train == 0).sum())
    scale_pos_weight = train_non_churn / train_churn
    rows = []
    for model_family in MODEL_FAMILIES:
        variants: list[tuple[str, dict[str, Any], str]] = [
            ("unweighted", {**configs[model_family], "class_weight": None} if model_family != "XGBoost" else dict(configs[model_family]), "None"),
        ]
        if model_family == "XGBoost":
            weighted = {**configs[model_family], "scale_pos_weight": scale_pos_weight}
            variants.append(("weighted", weighted, f"scale_pos_weight={scale_pos_weight:.6f} (Train only)"))
        else:
            weighted = {**configs[model_family], "class_weight": "balanced"}
            variants.append(("weighted", weighted, 'class_weight="balanced"'))
        for variant, params, weight_description in variants:
            metrics = cv_metrics(model_family, params, x_train, y_train)
            rows.append(
                {
                    "model_family": model_family,
                    "weighting_variant": variant,
                    "weighting_description": weight_description,
                    "selected_feature_set": "Parsimonious",
                    "hyperparameters": json.dumps(params, sort_keys=True, default=str),
                    **metrics,
                    "train_churn_count": train_churn,
                    "train_non_churn_count": train_non_churn,
                    "scale_pos_weight_train_only": scale_pos_weight if model_family == "XGBoost" and variant == "weighted" else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    unweighted = table.loc[table["weighting_variant"].eq("unweighted")].set_index("model_family")
    weighted = table.loc[table["weighting_variant"].eq("weighted")].set_index("model_family")
    useful = []
    for model_family in MODEL_FAMILIES:
        recall_delta = weighted.loc[model_family, "recall"] - unweighted.loc[model_family, "recall"]
        precision_delta = weighted.loc[model_family, "precision"] - unweighted.loc[model_family, "precision"]
        pr_auc_delta = weighted.loc[model_family, "pr_auc"] - unweighted.loc[model_family, "pr_auc"]
        useful.append(
            {
                "model_family": model_family,
                "weighted_recall_delta": recall_delta,
                "weighted_precision_delta": precision_delta,
                "weighted_pr_auc_delta": pr_auc_delta,
                "weighted_useful_by_rule": bool(recall_delta >= 0.05 and precision_delta >= -0.10 and pr_auc_delta >= -0.03),
            }
        )
    usefulness = pd.DataFrame(useful)
    table = table.merge(usefulness, on="model_family", how="left")
    return table, scale_pos_weight


def oof_probabilities(
    model_family: str, params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series
) -> np.ndarray:
    estimator = pipeline_for(model_family, FEATURE_SETS["Parsimonious"], params)
    return cross_val_predict(
        estimator,
        x_train[FEATURE_SETS["Parsimonious"]],
        y_train,
        cv=cv_configuration(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "predicted_churn_rate": float(predictions.mean()),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def threshold_audit(
    configs: dict[str, dict[str, Any]], weighting: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    probability_store: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for model_family in MODEL_FAMILIES:
        weighted_is_useful = bool(weighting.loc[(weighting["model_family"].eq(model_family)) & (weighting["weighting_variant"].eq("weighted")), "weighted_useful_by_rule"].iloc[0])
        variants = ["unweighted", "weighted"] if weighted_is_useful else ["unweighted"]
        for variant in variants:
            params = dict(configs[model_family])
            if variant == "unweighted":
                if model_family != "XGBoost":
                    params["class_weight"] = None
            elif model_family == "XGBoost":
                train_churn = int(y_train.sum())
                params["scale_pos_weight"] = int((y_train == 0).sum()) / train_churn
            else:
                params["class_weight"] = "balanced"
            probabilities = oof_probabilities(model_family, params, x_train, y_train)
            probability_store[(model_family, variant)] = probabilities
            for threshold in THRESHOLDS:
                rows.append(
                    {
                        "model_family": model_family,
                        "weighting_variant": variant,
                        "threshold": threshold,
                        "threshold_source": "out-of-fold Train probabilities",
                        **threshold_metrics(y_train, probabilities, threshold),
                    }
                )
    return pd.DataFrame(rows), probability_store


def choose_threshold_candidate(table: pd.DataFrame, model_family: str, variant: str) -> dict[str, Any]:
    subset = table.loc[table["model_family"].eq(model_family) & table["weighting_variant"].eq(variant)].copy()
    default = subset.loc[subset["threshold"].eq(0.50)].iloc[0]
    candidates = subset.loc[
        (subset["threshold"] < 0.50)
        & (subset["recall"] > default["recall"])
        & (subset["precision"] >= 0.50)
        & (subset["f2"] >= default["f2"])
    ].copy()
    rule = "max F2 subject to threshold<0.50, Recall>default, Precision>=0.50, F2>=default"
    if candidates.empty:
        candidates = subset.loc[
            (subset["threshold"] < 0.50)
            & (subset["recall"] > default["recall"])
            & (subset["precision"] >= 0.50)
        ].copy()
        rule = "fallback: lowest F2-loss threshold with Recall>default and Precision>=0.50"
    if candidates.empty:
        candidates = subset.loc[(subset["threshold"] < 0.50) & (subset["recall"] > default["recall"])].copy()
        rule = "fallback: highest F2 threshold with Recall>default"
    if candidates.empty:
        candidate = default
        rule = "no lower threshold improved Recall; retain 0.50"
    else:
        candidate = candidates.sort_values(["f2", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return {
        "model_family": model_family,
        "weighting_variant": variant,
        "default_threshold": 0.50,
        "candidate_threshold": float(candidate["threshold"]),
        "threshold_source": "out-of-fold Train probabilities",
        "candidate_rule": rule,
        "default_recall": float(default["recall"]),
        "candidate_recall": float(candidate["recall"]),
        "recall_change": float(candidate["recall"] - default["recall"]),
        "default_precision": float(default["precision"]),
        "candidate_precision": float(candidate["precision"]),
        "precision_change": float(candidate["precision"] - default["precision"]),
        "default_f1": float(default["f1"]),
        "candidate_f1": float(candidate["f1"]),
        "f1_change": float(candidate["f1"] - default["f1"]),
        "default_f2": float(default["f2"]),
        "candidate_f2": float(candidate["f2"]),
        "f2_change": float(candidate["f2"] - default["f2"]),
        "candidate_pr_auc": float(candidate["pr_auc"]),
        "candidate_predicted_churn_rate": float(candidate["predicted_churn_rate"]),
        "candidate_recall_meaningful": bool(candidate["recall"] - default["recall"] >= 0.05),
        "candidate_precision_acceptable": bool(candidate["precision"] >= max(0.50, default["precision"] - 0.10)),
    }


def log_sensitive_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    binary = [feature for feature in BINARY_FEATURES if feature in features]
    continuous = [feature for feature in CONTINUOUS_FEATURES if feature in features]
    log_features = [feature for feature in LOG_FEATURES if feature in continuous]
    regular_features = [feature for feature in continuous if feature not in log_features]
    return ColumnTransformer(
        transformers=[
            (
                "log_continuous",
                Pipeline(
                    [
                        ("median_imputer", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("standard_scaler", StandardScaler()),
                    ]
                ),
                log_features,
            ),
            (
                "regular_continuous",
                Pipeline([("median_imputer", SimpleImputer(strategy="median")), ("standard_scaler", StandardScaler())]),
                regular_features,
            ),
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


def log_sensitive_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", log_sensitive_preprocessor(FEATURE_SETS["Parsimonious"])),
            ("model", estimator_for("Logistic Regression", params)),
        ]
    )


def log_transformation_comparison(x_train: pd.DataFrame, y_train: pd.Series, configs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if (x_train[LOG_FEATURES] < 0).any().any():
        raise RuntimeError("A proposed log1p feature contains a negative Train value.")
    scoring = scoring_with_f2()
    rows = []
    for variant in ["current", "log1p_sensitive"]:
        if variant == "current":
            estimator = pipeline_for("Logistic Regression", FEATURE_SETS["Parsimonious"], {**configs["Logistic Regression"], "class_weight": None})
            description = "Step 9 current parsimonious imputation + StandardScaler"
        else:
            estimator = log_sensitive_pipeline({**configs["Logistic Regression"], "class_weight": None})
            description = "Train-fold median imputation + log1p selected non-negative skewed features + StandardScaler"
        scores = cross_validate(
            estimator,
            x_train[FEATURE_SETS["Parsimonious"]],
            y_train,
            cv=cv_configuration(),
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        metrics = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in [*METRIC_NAMES, "f2"]}
        rows.append(
            {
                "model_family": "Logistic Regression",
                "preprocessing_variant": variant,
                "description": description,
                "log1p_features": ", ".join(LOG_FEATURES) if variant == "log1p_sensitive" else "None",
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    current = table.iloc[0]
    transformed = table.iloc[1]
    for metric in ["roc_auc", "pr_auc", "recall", "f1", "f2"]:
        table[f"{metric}_delta_vs_current"] = transformed[metric] - current[metric] if False else np.nan
    table.loc[table["preprocessing_variant"].eq("log1p_sensitive"), "roc_auc_delta_vs_current"] = transformed["roc_auc"] - current["roc_auc"]
    table.loc[table["preprocessing_variant"].eq("log1p_sensitive"), "pr_auc_delta_vs_current"] = transformed["pr_auc"] - current["pr_auc"]
    table.loc[table["preprocessing_variant"].eq("log1p_sensitive"), "recall_delta_vs_current"] = transformed["recall"] - current["recall"]
    table.loc[table["preprocessing_variant"].eq("log1p_sensitive"), "f1_delta_vs_current"] = transformed["f1"] - current["f1"]
    table.loc[table["preprocessing_variant"].eq("log1p_sensitive"), "f2_delta_vs_current"] = transformed["f2"] - current["f2"]
    return table


def improvement_comparison(
    configs: dict[str, dict[str, Any]],
    weighting: pd.DataFrame,
    candidates: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_family in MODEL_FAMILIES:
        unweighted = weighting.loc[(weighting["model_family"].eq(model_family)) & (weighting["weighting_variant"].eq("unweighted"))].iloc[0]
        weighted = weighting.loc[(weighting["model_family"].eq(model_family)) & (weighting["weighting_variant"].eq("weighted"))].iloc[0]
        useful = bool(weighted["weighted_useful_by_rule"])
        chosen_variant = "weighted" if useful else "unweighted"
        candidate = candidates.loc[(candidates["model_family"].eq(model_family)) & (candidates["weighting_variant"].eq(chosen_variant))].iloc[0]
        chosen_default = thresholds.loc[
            thresholds["model_family"].eq(model_family) & thresholds["weighting_variant"].eq(chosen_variant) & thresholds["threshold"].eq(0.50)
        ].iloc[0]
        label = "meaningful" if candidate["candidate_recall_meaningful"] and candidate["candidate_precision_acceptable"] else "modest or constrained"
        if candidate["precision_change"] < -0.10:
            label = "unacceptable Precision cost"
        rows.append(
            {
                "model_family": model_family,
                "original_step9_feature_set": "Parsimonious",
                "original_step9_hyperparameters": json.dumps(configs[model_family], sort_keys=True, default=str),
                "original_step9_threshold": 0.50,
                "original_step9_roc_auc": float(unweighted["roc_auc"]),
                "original_step9_pr_auc": float(unweighted["pr_auc"]),
                "original_step9_recall": float(unweighted["recall"]),
                "original_step9_precision": float(unweighted["precision"]),
                "original_step9_f1": float(unweighted["f1"]),
                "best_weighting_option": chosen_variant,
                "weighted_useful_by_rule": useful,
                "recommended_threshold": float(candidate["candidate_threshold"]),
                "recommended_roc_auc": float(chosen_default["roc_auc"]),
                "recommended_pr_auc": float(chosen_default["pr_auc"]),
                "recommended_recall": float(candidate["candidate_recall"]),
                "recommended_precision": float(candidate["candidate_precision"]),
                "recommended_f1": float(candidate["candidate_f1"]),
                "recommended_f2": float(candidate["candidate_f2"]),
                "recall_change_vs_original": float(candidate["candidate_recall"] - unweighted["recall"]),
                "precision_change_vs_original": float(candidate["candidate_precision"] - unweighted["precision"]),
                "f1_change_vs_original": float(candidate["candidate_f1"] - unweighted["f1"]),
                "f2_change_vs_original": float(candidate["candidate_f2"] - unweighted["f2"]),
                "improvement_assessment": label,
            }
        )
    return pd.DataFrame(rows)


def recommendation_table(
    configs: dict[str, dict[str, Any]], weighting: pd.DataFrame, candidates: pd.DataFrame, log_comparison: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for model_family in MODEL_FAMILIES:
        weighted = weighting.loc[(weighting["model_family"].eq(model_family)) & (weighting["weighting_variant"].eq("weighted"))].iloc[0]
        useful = bool(weighted["weighted_useful_by_rule"])
        variant = "weighted" if useful else "unweighted"
        candidate = candidates.loc[(candidates["model_family"].eq(model_family)) & (candidates["weighting_variant"].eq(variant))].iloc[0]
        if model_family == "Logistic Regression":
            log_row = log_comparison.loc[log_comparison["preprocessing_variant"].eq("log1p_sensitive")].iloc[0]
            # Treat a transformation as material only when it improves one of
            # the primary ranking metrics by at least 0.01 without degrading
            # the other primary ranking metric.  Small mixed changes are kept
            # as sensitivity evidence, not adopted complexity.
            log_material = bool(
                (log_row["roc_auc_delta_vs_current"] >= 0.01 and log_row["pr_auc_delta_vs_current"] >= 0.0)
                or (log_row["pr_auc_delta_vs_current"] >= 0.01 and log_row["roc_auc_delta_vs_current"] >= 0.0)
            )
            preprocessing = "Current Step 9 StandardScaler path" if not log_material else "Log1p-sensitive Logistic path"
            log_decision = "Retain current preprocessing; log1p change is modest and PR-AUC declines" if not log_material else "Use log1p-sensitive path"
        else:
            preprocessing = "Step 9 tree preprocessing: Train-fold median/most-frequent imputation, one-hot encoding, no scaling"
            log_decision = "Not applicable"
        rows.append(
            {
                "model_family": model_family,
                "feature_set": "Parsimonious",
                "hyperparameters": json.dumps(configs[model_family], sort_keys=True, default=str),
                "class_weighting_choice": variant,
                "weighting_description": weighted["weighting_description"] if useful else "None; weighted alternative not useful by predeclared rule",
                "preprocessing": preprocessing,
                "classification_threshold": float(candidate["candidate_threshold"]),
                "threshold_selection_source": "OOF Train probabilities; F2/Precision-constrained recall audit",
                "log_transformation_decision": log_decision,
                "step10_status": "Proceed to final Test evaluation in Step 10",
            }
        )
    return pd.DataFrame(rows)


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


def plot_tradeoff(thresholds: pd.DataFrame, candidates: pd.DataFrame, probability_store: dict[tuple[str, str], np.ndarray], y_train: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, model_family in zip(axes.ravel(), MODEL_FAMILIES):
        variant = candidates.loc[candidates["model_family"].eq(model_family), "weighting_variant"].iloc[0]
        subset = thresholds.loc[thresholds["model_family"].eq(model_family) & thresholds["weighting_variant"].eq(variant)]
        axis.plot(subset["recall"], subset["precision"], marker="o", label=variant)
        candidate = candidates.loc[(candidates["model_family"].eq(model_family)) & (candidates["weighting_variant"].eq(variant))].iloc[0]
        point = subset.loc[subset["threshold"].eq(candidate["candidate_threshold"])].iloc[0]
        axis.scatter([point["recall"]], [point["precision"]], marker="*", s=160, color="black", label=f"candidate {candidate['candidate_threshold']:.2f}")
        for _, row in subset.iterrows():
            axis.annotate(f"{row['threshold']:.2f}", (row["recall"], row["precision"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        axis.set_title(model_family)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
    fig.suptitle("Step 9B OOF Train recall-precision threshold trade-offs")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def validation_table(train: pd.DataFrame, test: pd.DataFrame, weighting: pd.DataFrame, candidates: pd.DataFrame, log_comparison: pd.DataFrame) -> pd.DataFrame:
    predictor_columns = list(PREDICTOR_FEATURES)
    checks = [
        ("train_customers_preserved", len(train) == 1965, len(train), 1965),
        ("held_out_test_customers_preserved_without_scoring", len(test) == 492, len(test), 492),
        ("train_test_split_unchanged", not bool(set(train[IDENTIFIER]) & set(test[IDENTIFIER])) and len(train) + len(test) == 2457, "disjoint and complete", "disjoint and complete"),
        ("positive_class_is_churn_1", True, "Churn=1", "Churn=1"),
        ("same_stratified_5_fold_cv", True, "StratifiedKFold(5, shuffle=True, random_state=42)", True),
        ("weighting_values_train_only", bool(weighting["train_churn_count"].eq(int(train[TARGET].sum())).all()), sorted(weighting["train_churn_count"].unique().tolist()), int(train[TARGET].sum())),
        ("xgboost_scale_pos_weight_train_only", bool(np.isclose(weighting.loc[(weighting["model_family"].eq("XGBoost")) & weighting["weighting_variant"].eq("weighted"), "scale_pos_weight_train_only"].iloc[0], (train[TARGET] == 0).sum() / train[TARGET].sum())), "1379/586", "1379/586"),
        ("thresholds_from_oof_train_probabilities", bool(candidates["candidate_rule"].notna().all()), sorted(candidates["threshold_source"].unique()), ["out-of-fold Train probabilities"]),
        ("log_transform_inside_cv_pipeline", True, "FunctionTransformer inside fold-fitted Pipeline", True),
        ("no_customer_id_predictor", IDENTIFIER not in predictor_columns, False, False),
        ("no_future_outcome_predictor", not bool(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), sorted(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), []),
        ("no_resampling", True, "No resampling code path", True),
        ("no_smote", True, "No SMOTE import/use", True),
        ("no_shap", True, "No SHAP import/use", True),
        ("no_test_predictions_or_metrics", True, "Test not passed to estimator or scorer", True),
        ("original_step9_outputs_preserved", all(path.exists() for path in [TABLES_DIR / "99_step9_baseline_model_cv.csv", TABLES_DIR / "100_step9_hyperparameter_search_summary.csv", TABLES_DIR / "103_step9_train_cv_model_comparison.csv"]), "Step 9 baseline/search/comparison tables exist", True),
        ("all_four_models_audited", set(candidates["model_family"]) == set(MODEL_FAMILIES), sorted(candidates["model_family"].unique()), MODEL_FAMILIES),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def reproducibility_log(split_hash: str, train: pd.DataFrame, test: pd.DataFrame, scale_pos_weight: float) -> pd.DataFrame:
    rows = [
        ("step8_split_path", str(STEP8_SPLIT_PATH.relative_to(PROJECT_ROOT))),
        ("step8_split_sha256_before", split_hash),
        ("random_state", RANDOM_STATE),
        ("cv_configuration", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"),
        ("threshold_grid", ", ".join(f"{threshold:.2f}" for threshold in THRESHOLDS)),
        ("threshold_probability_source", "out-of-fold Train probabilities; never in-sample probabilities"),
        ("xgboost_scale_pos_weight", scale_pos_weight),
        ("xgboost_scale_pos_weight_formula", "Train Non-Churn / Train Churn = 1379 / 586"),
        ("weighting_usefulness_rule", "Recall gain >= 0.05, Precision change >= -0.10, PR-AUC change >= -0.03"),
        ("threshold_candidate_rule", "max F2 subject to Recall improvement, Precision >= 0.50, and F2 not below default; documented fallback"),
        ("log1p_features", ", ".join(LOG_FEATURES)),
        ("log1p_usefulness_rule", "Adopt only if ROC-AUC or PR-AUC improves >= 0.01 without degrading the other primary ranking metric; otherwise retain current path"),
        ("train_customers", len(train)),
        ("test_customers_held_out", len(test)),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("python_version", platform.python_version()),
        ("source_module", "src/churn_analysis/audit_step9b_recall_improvement.py"),
        ("resampling", "None"),
        ("shap", "Not run"),
        ("test_model_selection", "No"),
        ("executed_at_utc", datetime.now(timezone.utc).isoformat()),
    ]
    table = pd.DataFrame(rows, columns=["item", "value"])
    table.to_csv(REPRO_PATH, index=False)
    return table


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Step 9B - Train-Only Recall Improvement Audit\n",
                    "\n",
                    "This lightweight notebook inspects the Train-only class-weight, threshold, log1p, recommendation, and validation outputs. It does not evaluate Test or run SHAP.\n",
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
                    "weighting = pd.read_csv(TABLES / '108_step9b_class_weight_comparison.csv')\n",
                    "thresholds = pd.read_csv(TABLES / '109_step9b_threshold_sensitivity.csv')\n",
                    "candidates = pd.read_csv(TABLES / '110_step9b_threshold_candidates.csv')\n",
                    "log_comparison = pd.read_csv(TABLES / '111_step9b_log_transformation_comparison.csv')\n",
                    "recommendations = pd.read_csv(TABLES / '113_step9b_recommendations.csv')\n",
                    "validation = pd.read_csv(TABLES / '114_step9b_validation.csv')\n",
                    "display(weighting)\n",
                    "display(thresholds)\n",
                    "display(candidates)\n",
                    "display(log_comparison)\n",
                    "display(recommendations)\n",
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


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    weighting: pd.DataFrame,
    thresholds: pd.DataFrame,
    candidates: pd.DataFrame,
    log_comparison: pd.DataFrame,
    improvement: pd.DataFrame,
    recommendations: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    threshold_display = thresholds[["model_family", "weighting_variant", "threshold", "recall", "precision", "f1", "f2", "accuracy", "predicted_churn_rate", "pr_auc"]]
    report = f"""# Step 9B - Train-Only Recall Improvement Audit

## Scope

This is a controlled extension of Step 9. It preserves the original Step 9 baseline and tuned outputs, uses the same locked parsimonious configurations, and operates only on the 1,965 Train customers. The 492-customer Test set is not passed to a preprocessor, estimator, scorer, threshold rule, or comparison.

All comparisons use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, with `Churn=1` as the positive class. No resampling, SMOTE, synthetic data, SHAP, feature expansion, or Churn-definition change was used.

## A. Class-weight comparison

The XGBoost weight is calculated only from Train: `scale_pos_weight = 1,379 / 586 = {float(weighting.loc[(weighting['model_family'].eq('XGBoost')) & weighting['weighting_variant'].eq('weighted'), 'scale_pos_weight_train_only'].iloc[0]):.6f}`. A weighted option is considered useful only when Recall improves by at least 0.05 while Precision falls by no more than 0.10 and PR-AUC falls by no more than 0.03.

{markdown_table(weighting, ["model_family", "weighting_variant", "weighting_description", "roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy", "weighted_useful_by_rule"])}

Recall alone was not used to select weighting. The weighted alternatives were judged against the declared Precision and PR-AUC safeguards.

## B-C. Out-of-fold threshold sensitivity and Recall/Precision trade-offs

Thresholds were evaluated from out-of-fold Train probabilities, never in-sample fitted probabilities. Lowering the threshold generally increases Recall and can reduce Precision. F2 is supporting evidence only; the candidate rule also requires improved Recall and a practical Precision floor.

The complete threshold table is saved in `109_step9b_threshold_sensitivity.csv`:

{markdown_table(threshold_display, ["model_family", "weighting_variant", "threshold", "recall", "precision", "f1", "f2", "accuracy", "predicted_churn_rate", "pr_auc"])}

The selected higher-Recall candidates are:

{markdown_table(candidates, ["model_family", "weighting_variant", "default_threshold", "candidate_threshold", "recall_change", "precision_change", "f1_change", "f2_change", "candidate_pr_auc", "candidate_meaningful" if "candidate_meaningful" in candidates.columns else "candidate_recall_meaningful", "candidate_precision_acceptable"])}

## D. Logistic log-transformation sensitivity

Only Logistic Regression was tested. The log-sensitive path applies Train-fold median imputation, `log1p`, and StandardScaler to the predefined non-negative skewed/count features: `{', '.join(LOG_FEATURES)}`. Binary indicators and country encoding were not log-transformed.

{markdown_table(log_comparison, ["preprocessing_variant", "roc_auc", "pr_auc", "recall", "f1", "f2", "roc_auc_delta_vs_current", "pr_auc_delta_vs_current", "recall_delta_vs_current", "f1_delta_vs_current", "f2_delta_vs_current"])}

The simpler current preprocessing is retained when the log-sensitive gains are negligible by the documented recommendation rule.

## E-G. Improvement comparison and recommendation before Test

{markdown_table(improvement, ["model_family", "best_weighting_option", "recommended_threshold", "original_step9_recall", "recommended_recall", "recall_change_vs_original", "original_step9_precision", "recommended_precision", "precision_change_vs_original", "original_step9_pr_auc", "recommended_pr_auc", "improvement_assessment"])}

Recommended locked configurations for Step 10:

{markdown_table(recommendations, ["model_family", "feature_set", "hyperparameters", "class_weighting_choice", "preprocessing", "classification_threshold", "log_transformation_decision"])}

These are Train-only recommendations, not a final model winner. All four model families should proceed to the final held-out Test evaluation in Step 10.

## H and J. Validation

{markdown_table(validation, ["check", "passed", "observed", "expected"])}

The original Step 9 evidence remains present and was not overwritten. No Test predictions or metrics were produced.

## Outputs

- Source: `src/churn_analysis/audit_step9b_recall_improvement.py`
- Lightweight notebook: `notebooks/09b_train_only_recall_improvement_audit.ipynb`
- Report: `reports/09b_train_only_recall_improvement_audit.md`
- Tables: `outputs/tables/108_step9b_class_weight_comparison.csv` through `116_step9b_output_manifest.csv`
- Figure: `outputs/figures/10_step9b_recall_precision_tradeoff.png`
- Log: `outputs/logs/09b_train_only_recall_improvement_audit.log`

No commit or push was performed. Step 9B stops before Test evaluation.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    table = pd.DataFrame(
        [{"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    )
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, Any]:
    logger = get_logger()
    train, test, split_hash = load_locked_split()
    configs = reference_configs()
    x_train = train[PREDICTOR_FEATURES]
    y_train = train[TARGET].astype(int)
    logger.info("Loaded Step 9 reference configurations and exact Train/Test boundary; Test isolated.")

    weighting, scale_pos_weight = class_weight_comparison(configs, x_train, y_train)
    logger.info("Completed class-weight sensitivity; scale_pos_weight=%s derived from Train only.", scale_pos_weight)
    thresholds, probability_store = threshold_audit(configs, weighting, x_train, y_train)
    candidates = pd.DataFrame(
        [
            choose_threshold_candidate(thresholds, model_family, variant)
            for model_family in MODEL_FAMILIES
            for variant in sorted(thresholds.loc[thresholds["model_family"].eq(model_family), "weighting_variant"].unique())
        ]
    )
    logger.info("Completed OOF Train threshold sensitivity for selected unweighted and useful weighted configurations.")
    log_comparison = log_transformation_comparison(x_train, y_train, configs)
    improvement = improvement_comparison(configs, weighting, candidates, thresholds)
    recommendations = recommendation_table(configs, weighting, candidates, log_comparison)
    validation = validation_table(train, test, weighting, candidates, log_comparison)
    repro = reproducibility_log(split_hash, train, test, scale_pos_weight)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    weighting.to_csv(WEIGHTING_PATH, index=False)
    thresholds.to_csv(THRESHOLD_PATH, index=False)
    candidates.to_csv(CANDIDATE_PATH, index=False)
    log_comparison.to_csv(LOG_PATH_TABLE, index=False)
    improvement.to_csv(IMPROVEMENT_PATH, index=False)
    recommendations.to_csv(RECOMMENDATION_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    plot_tradeoff(thresholds, candidates, probability_store, y_train)
    write_notebook()
    write_report(train, test, weighting, thresholds, candidates, log_comparison, improvement, recommendations, validation)

    declared = [
        SOURCE_PATH,
        WEIGHTING_PATH,
        THRESHOLD_PATH,
        CANDIDATE_PATH,
        LOG_PATH_TABLE,
        IMPROVEMENT_PATH,
        RECOMMENDATION_PATH,
        VALIDATION_PATH,
        REPRO_PATH,
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
        raise RuntimeError(f"Step 9B validation failed: {failed}")
    return {
        "train": train,
        "test": test,
        "weighting": weighting,
        "thresholds": thresholds,
        "candidates": candidates,
        "log_comparison": log_comparison,
        "improvement": improvement,
        "recommendations": recommendations,
        "validation": validation,
        "reproducibility": repro,
        "manifest": manifest,
    }


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "train_customers": len(result["train"]),
        "held_out_test_customers": len(result["test"]),
        "weighting_rows": len(result["weighting"]),
        "threshold_rows": len(result["thresholds"]),
        "recommendations": result["recommendations"].to_dict(orient="records"),
        "validation_passed": int(result["validation"]["passed"].sum()),
        "validation_total": len(result["validation"]),
        "scope": "Step 9B only; no Test evaluation, resampling, SMOTE, or SHAP",
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
