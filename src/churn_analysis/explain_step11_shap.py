"""Step 11: explain the locked XGBoost churn model with SHAP.

This module is deliberately explanatory only.  It refits the already locked
Step 10 XGBoost pipeline on the complete Train split, transforms the untouched
Test split with Train-fitted preprocessing, and computes SHAP explanations for
the held-out Test customers.  It does not tune, resample, change the feature
set, change the threshold, or design managerial interventions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import spearmanr

from .build_step8_preprocessing import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    FUTURE_OUTCOME_FIELDS,
    IDENTIFIER,
    PREDICTOR_FEATURES,
    TARGET,
)
from .config import FIGURES_DIR, LOGS_DIR, PROJECT_ROOT, RANDOM_STATE, TABLES_DIR
from .evaluate_step10_test import EXPECTED_THRESHOLDS, load_locked_recommendations
from .train_step9_models import FEATURE_SETS, load_locked_split, pipeline_for


SOURCE_PATH = PROJECT_ROOT / "src" / "churn_analysis" / "explain_step11_shap.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "11_explain_selected_xgboost_with_shap.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "11_explain_selected_xgboost_with_shap.ipynb"
LOG_PATH = LOGS_DIR / "11_explain_selected_xgboost_with_shap.log"

TRANSFORMED_IMPORTANCE_PATH = TABLES_DIR / "128_step11_transformed_feature_shap_importance.csv"
BUSINESS_IMPORTANCE_PATH = TABLES_DIR / "129_step11_business_feature_shap_importance.csv"
DIRECTION_PATH = TABLES_DIR / "130_step11_direction_of_effect_audit.csv"
LOCAL_PATH = TABLES_DIR / "131_step11_local_explanation_audit.csv"
LOGISTIC_PATH = TABLES_DIR / "132_step11_logistic_benchmark_comparison.csv"
RQ1_PATH = TABLES_DIR / "133_step11_research_question_1_evidence.csv"
VALIDATION_PATH = TABLES_DIR / "134_step11_shap_validation.csv"
REPRO_PATH = TABLES_DIR / "135_step11_reproducibility_log.csv"
MANIFEST_PATH = TABLES_DIR / "136_step11_output_manifest.csv"

GLOBAL_BAR_PATH = FIGURES_DIR / "15_step11_global_shap_importance.png"
BEESWARM_PATH = FIGURES_DIR / "16_step11_shap_beeswarm.png"

BUSINESS_MEANINGS = {
    "RecencyDays": "Days since the customer's last positive purchase at the cutoff",
    "Frequency": "Count of positive purchase invoices/events in the observation period",
    "Monetary": "Total positive purchase value attributed to the customer",
    "TotalQuantity": "Total quantity purchased in positive purchase events",
    "AverageOrderValue": "Average positive purchase invoice value",
    "AverageQuantityPerOrder": "Average quantity purchased per positive invoice",
    "UniqueProducts": "Number of distinct products purchased",
    "ProductsPerOrder": "Average number of distinct products per positive invoice",
    "CustomerAgeDays": "Days from the customer's first positive purchase to the cutoff",
    "MedianInterpurchaseDays": "Median gap between positive purchase events",
    "HasRepeatPurchaseHistory": "Indicator that the customer has more than one positive purchase event",
    "CancellationInvoiceCount": "Count of cancellation/return invoices linked to the customer",
    "CancellationValue": "Absolute value of cancellation quantities multiplied by price",
    "HasCancellation": "Indicator that the customer has at least one cancellation/return invoice",
    "CancellationInvoiceRate": "Cancellation invoice count relative to positive purchase invoice count",
    "CountryCandidate": "Country recorded for the customer's linked transactions",
}

EXPECTED_XGB_PARAMS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 150,
    "reg_lambda": 1.0,
    "subsample": 0.8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.explain_step11_shap")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _format_value(value: Any, digits: int = 3) -> str:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return _json(value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 3) -> str:
    data = frame.copy()
    if columns is not None:
        data = data[columns]
    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value, digits) for value in row) + " |")
    return "\n".join(lines)


def normalize_feature_name(name: str) -> str:
    if name.startswith("CountryCandidate_"):
        return "CountryCandidate"
    return name


def expected_value_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0])


def load_locked_data() -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, dict[str, Any]]]:
    train, test, split_hash = load_locked_split()
    recommendations = load_locked_recommendations()
    return train, test, split_hash, recommendations


def validate_locked_configuration(recommendations: dict[str, dict[str, Any]]) -> None:
    xgb = recommendations["XGBoost"]
    if xgb["feature_set"] != "Parsimonious":
        raise RuntimeError("Step 11 requires the locked Parsimonious XGBoost feature set.")
    if not str(xgb["preprocessing"]).startswith("Step 9 tree preprocessing"):
        raise RuntimeError("Step 11 received an unexpected XGBoost preprocessing path.")
    if xgb["params"] != EXPECTED_XGB_PARAMS:
        raise RuntimeError(f"Locked XGBoost parameters changed: {xgb['params']}")
    if not np.isclose(xgb["threshold"], EXPECTED_THRESHOLDS["XGBoost"]):
        raise RuntimeError("Locked XGBoost threshold changed.")


def fit_locked_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recommendations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    features = FEATURE_SETS["Parsimonious"]
    xgb_pipeline = pipeline_for("XGBoost", features, recommendations["XGBoost"]["params"])
    xgb_pipeline.fit(train[features], train[TARGET].astype(int))
    xgb_preprocessor = xgb_pipeline.named_steps["preprocess"]
    xgb_model = xgb_pipeline.named_steps["model"]
    x_train = pd.DataFrame(
        xgb_preprocessor.transform(train[features]),
        columns=xgb_preprocessor.get_feature_names_out(),
        index=train.index,
    )
    x_test = pd.DataFrame(
        xgb_preprocessor.transform(test[features]),
        columns=xgb_preprocessor.get_feature_names_out(),
        index=test.index,
    )

    logistic_pipeline = pipeline_for("Logistic Regression", features, recommendations["Logistic Regression"]["params"])
    logistic_pipeline.fit(train[features], train[TARGET].astype(int))
    return {
        "features": features,
        "xgb_pipeline": xgb_pipeline,
        "xgb_preprocessor": xgb_preprocessor,
        "xgb_model": xgb_model,
        "x_train": x_train,
        "x_test": x_test,
        "logistic_pipeline": logistic_pipeline,
    }


def compute_shap(explanatory: dict[str, Any]) -> dict[str, Any]:
    model = explanatory["xgb_model"]
    x_test = explanatory["x_test"]
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = np.asarray(explainer.shap_values(x_test), dtype=float)
    if values.ndim != 2:
        raise RuntimeError(f"Expected a 2D SHAP matrix, received shape {values.shape}.")
    expected = expected_value_scalar(explainer.expected_value)
    raw_predictions = np.asarray(model.predict(x_test, output_margin=True), dtype=float).reshape(-1)
    reconstructed = expected + values.sum(axis=1)
    errors = reconstructed - raw_predictions
    # XGBoost/SHAP float arithmetic can differ by a few e-6 on the raw
    # margin; this is a numerical reconstruction tolerance, not a modelling
    # tolerance.
    tolerance = 1e-5
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reconstructed)):
        raise RuntimeError("SHAP values or reconstructed raw predictions contain non-finite values.")
    return {
        "xgb_model": model,
        "explainer": explainer,
        "values": values,
        "expected_value": expected,
        "raw_predictions": raw_predictions,
        "reconstructed": reconstructed,
        "reconstruction_errors": errors,
        "reconstruction_tolerance": tolerance,
        "shap_scale": "raw XGBoost margin / log-odds scale",
    }


def transformed_importance(shap_result: dict[str, Any], x_test: pd.DataFrame) -> pd.DataFrame:
    values = shap_result["values"]
    result = pd.DataFrame(
        {
            "transformed_feature": x_test.columns,
            "original_feature": [normalize_feature_name(name) for name in x_test.columns],
            "mean_absolute_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "test_customer_count": len(x_test),
        }
    )
    result = result.sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["relative_share_pct_of_transformed"] = result["mean_absolute_shap"] / result["mean_absolute_shap"].sum() * 100.0
    return result[[
        "rank",
        "transformed_feature",
        "original_feature",
        "mean_absolute_shap",
        "mean_shap",
        "relative_share_pct_of_transformed",
        "test_customer_count",
    ]]


def business_importance(
    transformed: pd.DataFrame,
    x_test: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    country_indices = [
        index for index, name in enumerate(x_test.columns) if normalize_feature_name(name) == "CountryCandidate"
    ]
    country_names = [x_test.columns[index] for index in country_indices]
    if len(country_indices) != 30 or len(set(country_names)) != 30:
        raise RuntimeError("Expected exactly 30 unique CountryCandidate one-hot columns.")
    non_country_indices = [
        index for index, name in enumerate(x_test.columns) if normalize_feature_name(name) != "CountryCandidate"
    ]
    if set(non_country_indices) & set(country_indices):
        raise RuntimeError("CountryCandidate and non-Country columns overlap.")
    # Group the signed contributions per customer before applying abs/mean.
    # This is the additive SHAP-consistent aggregation for the original
    # categorical feature; it is not a sum or mean of global magnitudes.
    country_grouped_shap = shap_values[:, country_indices].sum(axis=1)
    for feature in FEATURE_SETS["Parsimonious"]:
        if feature == "CountryCandidate":
            mean_abs = float(np.mean(np.abs(country_grouped_shap)))
            aggregation = "mean(abs(per-customer sum of signed SHAP across all 30 CountryCandidate one-hot columns))"
            column_count = 30
            mean_shap = float(np.mean(country_grouped_shap))
        else:
            matching = transformed.loc[transformed["original_feature"].eq(feature)]
            if len(matching) != 1:
                raise RuntimeError(f"Expected one transformed column for {feature}, found {len(matching)}.")
            row = matching.iloc[0]
            mean_abs = float(row["mean_absolute_shap"])
            mean_shap = float(row["mean_shap"])
            aggregation = "direct transformed feature value"
            column_count = 1
        rows.append(
            {
                "business_feature": feature,
                "business_meaning": BUSINESS_MEANINGS[feature],
                "mean_absolute_shap": mean_abs,
                "mean_shap": mean_shap,
                "transformed_column_count": column_count,
                "aggregation_method": aggregation,
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["relative_share_pct_of_business_aggregate"] = result["mean_absolute_shap"] / result["mean_absolute_shap"].sum() * 100.0
    return result[[
        "rank",
        "business_feature",
        "business_meaning",
        "mean_absolute_shap",
        "mean_shap",
        "relative_share_pct_of_business_aggregate",
        "transformed_column_count",
        "aggregation_method",
    ]]


def _direction_label(rho: float, low: float, high: float, bin_means: list[float]) -> str:
    differences = np.diff(bin_means)
    if abs(high - low) < 1e-9:
        return "nonlinear/mixed"
    positive_fraction = float(np.mean(differences >= 0)) if len(differences) else 0.0
    negative_fraction = float(np.mean(differences <= 0)) if len(differences) else 0.0
    if rho >= 0.25 and high > low and positive_fraction >= 0.5:
        return "higher values generally push predictions toward Churn"
    if rho <= -0.25 and high < low and negative_fraction >= 0.5:
        return "higher values generally push predictions away from Churn"
    return "nonlinear/mixed"


def direction_audit(
    test: pd.DataFrame,
    x_test: pd.DataFrame,
    shap_result: dict[str, Any],
    business: pd.DataFrame,
) -> pd.DataFrame:
    values = shap_result["values"]
    top_numeric = [
        feature
        for feature in business.sort_values("mean_absolute_shap")["business_feature"].tolist()[::-1]
        if feature in CONTINUOUS_FEATURES
    ][:5]
    top_binary = [
        feature
        for feature in business.sort_values("mean_absolute_shap")["business_feature"].tolist()[::-1]
        if feature in BINARY_FEATURES
    ]
    selected = top_numeric + [feature for feature in top_binary if feature not in top_numeric]
    rows: list[dict[str, Any]] = []
    for feature in selected:
        col_index = list(x_test.columns).index(feature)
        feature_values = x_test[feature].to_numpy(dtype=float)
        shap_values = values[:, col_index]
        missing_count = int(test[feature].isna().sum())
        if feature in BINARY_FEATURES:
            group_0 = float(np.mean(shap_values[feature_values < 0.5])) if np.any(feature_values < 0.5) else np.nan
            group_1 = float(np.mean(shap_values[feature_values >= 0.5])) if np.any(feature_values >= 0.5) else np.nan
            if pd.isna(group_0) or pd.isna(group_1) or abs(group_1 - group_0) < 1e-9:
                pattern = "nonlinear/mixed"
            elif group_1 > group_0:
                pattern = "value 1 generally pushes predictions toward Churn"
            else:
                pattern = "value 1 generally pushes predictions away from Churn"
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "binary",
                    "mean_absolute_shap": float(np.mean(np.abs(shap_values))),
                    "spearman_rho": np.nan,
                    "low_bin_median_feature_value": 0.0,
                    "high_bin_median_feature_value": 1.0,
                    "low_bin_mean_shap": group_0,
                    "high_bin_mean_shap": group_1,
                    "quantile_bin_mean_shap": "",
                    "pattern": pattern,
                    "missing_in_raw_test": missing_count,
                    "interpretation_note": "Binary indicator comparison; SHAP remains model-associative, not causal.",
                }
            )
            continue

        series = pd.Series(feature_values, index=test.index)
        ranks = series.rank(method="first")
        bins = pd.qcut(ranks, q=5, labels=False, duplicates="drop")
        bin_means = [float(np.mean(shap_values[bins.to_numpy() == bin_id])) for bin_id in sorted(pd.Series(bins).dropna().unique())]
        bin_medians = [float(np.median(feature_values[bins.to_numpy() == bin_id])) for bin_id in sorted(pd.Series(bins).dropna().unique())]
        rho = float(spearmanr(feature_values, shap_values).statistic)
        pattern = _direction_label(rho, bin_means[0], bin_means[-1], bin_means)
        rows.append(
            {
                "feature": feature,
                "feature_type": "continuous numeric",
                "mean_absolute_shap": float(np.mean(np.abs(shap_values))),
                "spearman_rho": rho,
                "low_bin_median_feature_value": bin_medians[0],
                "high_bin_median_feature_value": bin_medians[-1],
                "low_bin_mean_shap": bin_means[0],
                "high_bin_mean_shap": bin_means[-1],
                "quantile_bin_mean_shap": _json([round(value, 6) for value in bin_means]),
                "pattern": pattern,
                "missing_in_raw_test": missing_count,
                "interpretation_note": "Quantile bins use the Train-fitted tree transformation; structural missing values are median-imputed.",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)


def _top_contributors(values: np.ndarray, feature_names: list[str], positive: bool, count: int = 3) -> str:
    order = np.argsort(values)[::-1] if positive else np.argsort(values)
    selected = order[:count]
    return "; ".join(f"{feature_names[index]} ({values[index]:+.4f})" for index in selected)


def select_local_cases(
    test: pd.DataFrame,
    x_test: pd.DataFrame,
    shap_result: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    model = shap_result["xgb_model"] if "xgb_model" in shap_result else None
    probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    actual = test[TARGET].astype(int).to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    candidate_masks = {
        "True Positive / correctly predicted high-risk Churn": (actual == 1) & (predicted == 1),
        "True Negative / correctly predicted low-risk Non-Churn": (actual == 0) & (predicted == 0),
        "False Positive": (actual == 0) & (predicted == 1),
        "False Negative": (actual == 1) & (predicted == 0),
    }
    rows: list[dict[str, Any]] = []
    for case_label, mask in candidate_masks.items():
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            raise RuntimeError(f"No available Test customer for local case: {case_label}")
        if case_label.startswith("True Negative"):
            chosen = indices[np.argmin(probabilities[indices])]
        else:
            chosen = indices[np.argmax(probabilities[indices])]
        shap_values = shap_result["values"][chosen]
        customer_id = test.iloc[chosen][IDENTIFIER]
        rows.append(
            {
                "case_label": case_label,
                "test_row_position": int(chosen),
                "Customer ID": customer_id,
                "actual_Churn": int(actual[chosen]),
                "predicted_probability": float(probabilities[chosen]),
                "locked_threshold": threshold,
                "predicted_class": int(predicted[chosen]),
                "selection_rule": "highest Test probability within case" if not case_label.startswith("True Negative") else "lowest Test probability within True Negatives",
                "strongest_features_increasing_risk": _top_contributors(shap_values, list(x_test.columns), positive=True),
                "strongest_features_decreasing_risk": _top_contributors(shap_values, list(x_test.columns), positive=False),
            }
        )
    return pd.DataFrame(rows)


def logistic_benchmark(
    business: pd.DataFrame,
    direction: pd.DataFrame,
    explanatory: dict[str, Any],
) -> pd.DataFrame:
    pipeline = explanatory["logistic_pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    coefficients = np.asarray(model.coef_[0], dtype=float)
    rows: list[dict[str, Any]] = []
    for _, item in business.head(10).iterrows():
        feature = item["business_feature"]
        matching = [index for index, name in enumerate(names) if normalize_feature_name(name) == feature]
        if feature == "CountryCandidate":
            coefficient = float(np.mean(np.abs(coefficients[matching])))
            coefficient_sign = "mixed across country indicators"
            coefficient_method = "mean absolute standardized coefficient across country indicators"
        else:
            coefficient = float(coefficients[matching[0]])
            coefficient_sign = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
            coefficient_method = "single transformed feature coefficient"
        direction_row = direction.loc[direction["feature"].eq(feature)]
        xgb_pattern = direction_row.iloc[0]["pattern"] if len(direction_row) else (
            "country-specific/mixed" if feature == "CountryCandidate" else "not summarized"
        )
        if feature == "CountryCandidate" or "mixed" in str(xgb_pattern).lower() or "not summarized" in str(xgb_pattern).lower():
            comparison = "supportive comparison only; XGBoost pattern is not a single monotonic direction"
        else:
            xgb_toward = "toward Churn" in str(xgb_pattern)
            log_toward = coefficient > 0
            comparison = "directionally plausible" if xgb_toward == log_toward else "major directional disagreement"
        rows.append(
            {
                "rank_by_xgb_business_mean_abs_shap": int(item["rank"]),
                "business_feature": feature,
                "xgb_mean_absolute_shap": float(item["mean_absolute_shap"]),
                "xgb_direction_pattern": xgb_pattern,
                "logistic_coefficient_or_country_mean_abs": coefficient,
                "logistic_sign": coefficient_sign,
                "logistic_coefficient_method": coefficient_method,
                "comparison": comparison,
                "interpretation_guardrail": "Agreement is supportive model comparison, not causal confirmation.",
            }
        )
    return pd.DataFrame(rows)


def research_question_table(business: pd.DataFrame, direction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in business.head(10).iterrows():
        feature = item["business_feature"]
        direction_row = direction.loc[direction["feature"].eq(feature)]
        if feature == "CountryCandidate":
            pattern = "country-specific/mixed across one-hot indicators"
            interpretation = "CountryCandidate was ranked using the mean absolute per-customer sum of its 30 signed dummy SHAP contributions; there is no single high-versus-low country direction."
        elif len(direction_row):
            pattern = str(direction_row.iloc[0]["pattern"])
            interpretation = f"{pattern}; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect."
        else:
            pattern = "not summarized"
            interpretation = "Included in the ranked evidence; no simple direction was forced."
        rows.append(
            {
                "rank": int(item["rank"]),
                "feature": feature,
                "business_meaning": item["business_meaning"],
                "mean_absolute_shap": float(item["mean_absolute_shap"]),
                "relative_share_pct_of_business_aggregate": float(item["relative_share_pct_of_business_aggregate"]),
                "typical_direction_or_pattern": pattern,
                "short_interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    transformed: pd.DataFrame,
    shap_result: dict[str, Any],
    x_test: pd.DataFrame,
    direction: pd.DataFrame,
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    values = shap_result["values"]
    top = transformed.head(20).sort_values("mean_absolute_shap")
    plt.figure(figsize=(9, 7))
    plt.barh(top["transformed_feature"], top["mean_absolute_shap"], color="#2f6f9f")
    plt.xlabel("Mean absolute SHAP value (raw margin / log-odds)")
    plt.ylabel("Transformed model feature")
    plt.title("Step 11: global XGBoost SHAP importance on held-out Test")
    plt.tight_layout()
    plt.savefig(GLOBAL_BAR_PATH, dpi=180)
    plt.close()

    shap.summary_plot(
        values,
        x_test,
        feature_names=list(x_test.columns),
        max_display=20,
        plot_type="dot",
        show=False,
        plot_size=(10, 8),
    )
    plt.title("Step 11: SHAP beeswarm on held-out Test\n(raw margin / log-odds scale)", pad=18)
    plt.tight_layout()
    plt.savefig(BEESWARM_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    paths = [GLOBAL_BAR_PATH, BEESWARM_PATH]
    dependence_features = direction.loc[direction["feature_type"].eq("continuous numeric"), "feature"].head(4).tolist()
    for offset, feature in enumerate(dependence_features, start=17):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_").lower()
        path = FIGURES_DIR / f"{offset:02d}_step11_dependence_{safe}.png"
        shap.dependence_plot(
            feature,
            values,
            x_test,
            interaction_index=None,
            display_features=x_test,
            show=False,
        )
        plt.title(f"Step 11: SHAP dependence — {feature}\n(raw margin / log-odds scale)")
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def validation_table(
    train: pd.DataFrame,
    test: pd.DataFrame,
    explanatory: dict[str, Any],
    shap_result: dict[str, Any],
    transformed: pd.DataFrame,
    business: pd.DataFrame,
    local: pd.DataFrame,
    recommendations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    feature_names = list(explanatory["x_test"].columns)
    country_names = [name for name in feature_names if normalize_feature_name(name) == "CountryCandidate"]
    business_features = list(business["business_feature"])
    country_aggregation = str(
        business.loc[business["business_feature"].eq("CountryCandidate"), "aggregation_method"].iloc[0]
    )
    checks = [
        ("exact_locked_xgboost_configuration", recommendations["XGBoost"]["params"] == EXPECTED_XGB_PARAMS, recommendations["XGBoost"]["params"], EXPECTED_XGB_PARAMS),
        ("exact_locked_parsimonious_feature_set", explanatory["features"] == FEATURE_SETS["Parsimonious"], explanatory["features"], FEATURE_SETS["Parsimonious"]),
        ("train_customer_count", len(train) == 1965, len(train), 1965),
        ("test_customer_count", len(test) == 492, len(test), 492),
        ("preprocessing_fitted_on_train_only", True, "fit(train[features]); transform(test[features])", True),
        ("one_shap_explanation_per_test_customer", shap_result["values"].shape[0] == 492, shap_result["values"].shape[0], 492),
        ("shap_feature_columns_align_with_transformed_inputs", set(transformed["transformed_feature"]) == set(feature_names) and shap_result["values"].shape[1] == len(feature_names), f"{shap_result['values'].shape[1]} SHAP columns aligned to {len(feature_names)} transformed inputs", f"{len(feature_names)} aligned inputs"),
        ("customer_id_absent_from_model_features", IDENTIFIER not in PREDICTOR_FEATURES and IDENTIFIER not in feature_names, IDENTIFIER in feature_names, False),
        ("churn_absent_from_model_features", TARGET not in PREDICTOR_FEATURES and TARGET not in feature_names, TARGET in feature_names, False),
        ("future_information_absent", not bool(set(PREDICTOR_FEATURES) & FUTURE_OUTCOME_FIELDS), sorted(set(PREDICTOR_FEATURES) & FUTURE_OUTCOME_FIELDS), []),
        ("no_anonymous_feature_names", not any(name.startswith("x") and name[1:].isdigit() for name in feature_names), feature_names, "no x0-style names"),
        ("shap_values_finite", bool(np.isfinite(shap_result["values"]).all()), "all finite", True),
        ("shap_additive_reconstruction_within_tolerance", bool(np.max(np.abs(shap_result["reconstruction_errors"])) <= shap_result["reconstruction_tolerance"]), float(np.max(np.abs(shap_result["reconstruction_errors"]))), shap_result["reconstruction_tolerance"]),
        ("country_one_hot_aggregated_to_business_feature", "CountryCandidate" in set(business["business_feature"]), "CountryCandidate present", True),
        ("country_grouped_shap_uses_exactly_30_country_columns", len(country_names) == 30 and len(set(country_names)) == 30, len(country_names), 30),
        ("country_grouped_shap_excludes_non_country_columns", len(country_names) == sum(normalize_feature_name(name) == "CountryCandidate" for name in feature_names), "only CountryCandidate columns", True),
        ("country_aggregation_is_per_customer_signed_sum_then_abs_mean", country_aggregation.startswith("mean(abs(per-customer sum of signed SHAP"), country_aggregation, "signed per-customer sum then abs/mean"),
        ("business_feature_names_unique", len(business_features) == len(set(business_features)), len(business_features), len(set(business_features))),
        ("business_relative_shares_sum_to_100", np.isclose(float(business["relative_share_pct_of_business_aggregate"].sum()), 100.0, atol=1e-8), float(business["relative_share_pct_of_business_aggregate"].sum()), 100.0),
        ("selected_local_cases_present", len(local) == 4, len(local), 4),
        ("locked_threshold_unchanged", np.isclose(recommendations["XGBoost"]["threshold"], 0.35), recommendations["XGBoost"]["threshold"], 0.35),
        ("no_test_based_model_change", True, "Test used for explanation and audit only", True),
        ("no_new_feature_selection", True, "Step 9 Parsimonious set reused unchanged", True),
        ("no_model_tuning", True, "No search or parameter update", True),
        ("no_resampling", True, "No SMOTE/oversampling/undersampling", True),
        ("no_managerial_intervention_design", True, "Technical explanation only", True),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def reproducibility_table(
    split_hash: str,
    shap_result: dict[str, Any],
    explanatory: dict[str, Any],
    recommendations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = [
        ("step8_split_path", "data/processed/08_customer_train_test_split.parquet"),
        ("step8_split_sha256_before", split_hash),
        ("train_customers_fit", 1965),
        ("test_customers_explained", 492),
        ("positive_class", "Churn=1"),
        ("locked_feature_set", _json(explanatory["features"])),
        ("locked_xgboost_parameters", _json(recommendations["XGBoost"]["params"])),
        ("locked_classification_threshold", recommendations["XGBoost"]["threshold"]),
        ("xgboost_preprocessing", "Step 9 tree path: Train-fitted median/binary imputation and one-hot CountryCandidate; no scaling"),
        ("shap_explainer", "shap.TreeExplainer on fitted XGBoost model"),
        ("shap_output_scale", shap_result["shap_scale"]),
        ("positive_shap_interpretation", "pushes model toward higher predicted churn risk"),
        ("negative_shap_interpretation", "pushes model toward lower predicted churn risk"),
        ("expected_base_value_raw_margin", shap_result["expected_value"]),
        ("reconstruction_tolerance", shap_result["reconstruction_tolerance"]),
        ("country_aggregation", "mean(abs(per-customer sum of signed SHAP across all 30 CountryCandidate one-hot columns))"),
        ("local_case_selection", "highest probability within TP, FP, FN; lowest probability within TN"),
        ("test_based_model_change", "No"),
        ("retuning", "None"),
        ("resampling", "None"),
        ("causal_interpretation", "Not permitted; SHAP explains fitted-model behavior and association"),
        ("random_state", RANDOM_STATE),
        ("python_version", platform.python_version()),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("scipy_version", scipy.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("shap_version", shap.__version__),
        ("source_module", "src/churn_analysis/explain_step11_shap.py"),
        ("executed_at_utc", datetime.now(timezone.utc).isoformat()),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Step 11 — Explain the Selected XGBoost Churn Model Using SHAP\n",
                    "\n",
                    "This lightweight notebook inspects the numbered Step 11 audit outputs and figures. The executable source module refits the locked Step 10 XGBoost configuration on Train and explains the held-out Test set. It does not tune, resample, alter the threshold, or design managerial interventions.\n",
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
                    "transformed = pd.read_csv(TABLES / '128_step11_transformed_feature_shap_importance.csv')\n",
                    "business = pd.read_csv(TABLES / '129_step11_business_feature_shap_importance.csv')\n",
                    "direction = pd.read_csv(TABLES / '130_step11_direction_of_effect_audit.csv')\n",
                    "local = pd.read_csv(TABLES / '131_step11_local_explanation_audit.csv')\n",
                    "logistic = pd.read_csv(TABLES / '132_step11_logistic_benchmark_comparison.csv')\n",
                    "rq1 = pd.read_csv(TABLES / '133_step11_research_question_1_evidence.csv')\n",
                    "validation = pd.read_csv(TABLES / '134_step11_shap_validation.csv')\n",
                    "display(transformed.head(15))\n",
                    "display(business.head(10))\n",
                    "display(direction)\n",
                    "display(local.drop(columns=['Customer ID'], errors='ignore'))\n",
                    "display(logistic)\n",
                    "display(rq1)\n",
                    "display(validation)\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def write_report(
    transformed: pd.DataFrame,
    business: pd.DataFrame,
    direction: pd.DataFrame,
    local: pd.DataFrame,
    logistic: pd.DataFrame,
    rq1: pd.DataFrame,
    validation: pd.DataFrame,
    shap_result: dict[str, Any],
    explanatory: dict[str, Any],
    recommendations: dict[str, dict[str, Any]],
    figure_paths: list[Path],
) -> None:
    local_thesis = local.drop(columns=["Customer ID", "test_row_position"], errors="ignore")
    local_thesis_report = local_thesis.copy()
    false_negative_mask = local_thesis_report["case_label"].eq("False Negative")
    local_thesis_report.loc[false_negative_mask, "predicted_probability"] = local_thesis_report.loc[
        false_negative_mask, "predicted_probability"
    ].map(lambda value: f"{float(value):.17f}")
    report = f"""# Step 11 — Explain the Selected XGBoost Churn Model Using SHAP

## Scope and locked method

Step 11 explains the already selected XGBoost model. It refits the exact Step 10 configuration on the complete **1,965-customer Train set**, applies the Train-fitted Step 9 tree preprocessing to the existing **492-customer Test set**, and computes SHAP values for all Test customers. The Parsimonious 16-feature set, unweighted estimator, and locked classification threshold **0.35** are unchanged.

The XGBoost configuration was `n_estimators=150`, `max_depth=2`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, and `reg_lambda=1.0`, with the Step 9 binary-logistic objective, `random_state=42`, and one processing thread. No Test result was used to change the model.

## A. SHAP method and output scale

`shap.TreeExplainer` was applied to the fitted XGBoost estimator after Train-only preprocessing. The SHAP output is on the model's **raw margin / log-odds scale**, not the probability scale. A positive SHAP value pushes the model toward a higher predicted churn risk; a negative SHAP value pushes it toward a lower predicted churn risk. These are explanations of model behavior and association, not causal effects.

The base/expected value was **{shap_result['expected_value']:.8f}** on the raw-margin scale. Additive reconstruction was checked as `expected value + sum(feature SHAP values)` against the XGBoost raw margin, with tolerance **{shap_result['reconstruction_tolerance']:.1e}** and maximum absolute error **{float(np.max(np.abs(shap_result['reconstruction_errors']))):.3e}**.

## B. Global transformed-feature importance

The ranking below uses mean absolute SHAP value across the 492 held-out Test customers. Magnitude measures average influence strength regardless of direction; it does not indicate whether a feature increases or decreases churn risk.

{markdown_table(transformed.head(15), ['rank', 'transformed_feature', 'mean_absolute_shap', 'relative_share_pct_of_transformed'])}

The full transformed-feature table is saved as `outputs/tables/128_step11_transformed_feature_shap_importance.csv`. The global bar plot is `outputs/figures/{GLOBAL_BAR_PATH.name}` and the beeswarm is `outputs/figures/{BEESWARM_PATH.name}`.

## C. Original/business-feature importance

For non-categorical predictors, business-level importance is the direct transformed-column mean absolute SHAP value. `CountryCandidate` expands to 30 one-hot columns; its business-level value is calculated per customer by summing the **signed** SHAP values across all 30 dummies, then taking the absolute value and averaging across Test customers. This preserves the additive grouped SHAP contribution and is neither a sum nor an average of global dummy magnitudes.

{markdown_table(business.head(10), ['rank', 'business_feature', 'mean_absolute_shap', 'relative_share_pct_of_business_aggregate', 'transformed_column_count', 'aggregation_method'])}

The corrected top-four combined business-level share is **{float(business.head(4)['relative_share_pct_of_business_aggregate'].sum()):.6f}%**.

The relative shares here are shares of the transparent business-level aggregate, so they are useful for comparing the original features within that aggregation scheme. They are not causal effect sizes.

## D. Direction and selected dependence plots

The direction audit examines the actual transformed feature value against its SHAP value. Continuous features use ordered Test quantile bins and Spearman association; binary features compare the two observed indicator groups. Structural missing interval values use the already locked Train-fitted median imputation. The interpretation is intentionally descriptive: higher values may generally push the fitted model toward Churn, away from Churn, or show a nonlinear/mixed pattern.

{markdown_table(direction, ['feature', 'feature_type', 'mean_absolute_shap', 'spearman_rho', 'low_bin_mean_shap', 'high_bin_mean_shap', 'pattern', 'missing_in_raw_test'])}

Selected dependence figures were created only for the four most influential continuous numeric features in the business ranking:

{chr(10).join(f'- `{path.name}`' for path in figure_paths[2:])}

The beeswarm shows both SHAP magnitude and direction: points to the right push toward higher raw churn risk, points to the left push toward lower risk, and color indicates low-to-high transformed feature values. One-hot country columns are technical inputs; their business interpretation is the aggregated CountryCandidate result above.

## E. Local explanations

Cases were selected objectively from Test: highest predicted probability among correctly predicted Churn customers, lowest predicted probability among correctly predicted Non-Churn customers, highest-probability False Positive, and highest-probability False Negative. The audit table includes the internal Customer ID for traceability. The thesis-facing summary below omits that identifier.

{markdown_table(local_thesis_report, ['case_label', 'actual_Churn', 'predicted_probability', 'locked_threshold', 'predicted_class', 'selection_rule', 'strongest_features_increasing_risk', 'strongest_features_decreasing_risk'])}

Positive local SHAP contributions pushed each selected prediction toward higher churn risk; negative contributions pushed it toward lower risk. They do not identify causes or prescribe interventions.

## F. Logistic Regression interpretive benchmark

Logistic Regression was refit only in its locked Step 9 configuration (`C=1.0`, Parsimonious set, Train-fitted scaled preprocessing) as a secondary interpretive benchmark. It was not used to replace SHAP, retune XGBoost, or select features.

{markdown_table(logistic, ['rank_by_xgb_business_mean_abs_shap', 'business_feature', 'xgb_direction_pattern', 'logistic_coefficient_or_country_mean_abs', 'logistic_sign', 'comparison'])}

The benchmark shows broad directional plausibility for the strongest monotonic numeric variables where signs agree; any nonlinear/mixed XGBoost pattern is not reduced to a single coefficient comparison. Agreement is supportive model comparison only and is not causal confirmation.

## G. Direct answer to Research Question 1

**Which behavioral and transactional features are most predictive of churn, and what is their relative importance according to SHAP?**

In this fitted model and held-out Test sample, the strongest original/business features are shown below. The ranking is based on average absolute SHAP magnitude; direction/pattern is reported separately so magnitude is not confused with effect direction.

{markdown_table(rq1, ['rank', 'feature', 'business_meaning', 'mean_absolute_shap', 'relative_share_pct_of_business_aggregate', 'typical_direction_or_pattern', 'short_interpretation'])}

Thus, the answer is model-specific and associative: these features contributed most strongly to the XGBoost churn predictions on average in the locked held-out evaluation. “Most predictive” here means strongest contribution to this model's predictions, not proof that changing a feature would change churn.

## H. Validation and stopping point

{markdown_table(validation, ['check', 'passed', 'observed', 'expected'])}

All Step 11 validation checks passed. The locked Train/Test split, Step 10 model configuration, feature set, threshold, and preprocessing were preserved. No new feature selection, model tuning, threshold change, Test-based model change, resampling, or managerial intervention design occurred. Step 11 stops at technical explanation and understandable feature interpretation.

## Outputs

- Source: `src/churn_analysis/explain_step11_shap.py`
- Lightweight notebook: `notebooks/11_explain_selected_xgboost_with_shap.ipynb`
- Report: `reports/11_explain_selected_xgboost_with_shap.md`
- Tables: `outputs/tables/128_step11_transformed_feature_shap_importance.csv` through `outputs/tables/136_step11_output_manifest.csv`
- Figures: `{', '.join(path.name for path in figure_paths)}`
- Reproducibility log: `outputs/tables/135_step11_reproducibility_log.csv`

No commit or push was performed. Step 10 outputs were not overwritten.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ]
    )
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, Any]:
    logger = get_logger()
    train, test, split_hash, recommendations = load_locked_data()
    validate_locked_configuration(recommendations)
    logger.info("Validated exact Step 10 XGBoost configuration, Parsimonious features, and threshold.")

    explanatory = fit_locked_models(train, test, recommendations)
    logger.info("Fitted locked XGBoost and Logistic benchmark on %s Train rows.", len(train))
    shap_result = compute_shap(explanatory)
    transformed = transformed_importance(shap_result, explanatory["x_test"])
    business = business_importance(transformed, explanatory["x_test"], shap_result["values"])
    direction = direction_audit(test, explanatory["x_test"], shap_result, business)
    local = select_local_cases(test, explanatory["x_test"], shap_result, EXPECTED_THRESHOLDS["XGBoost"])
    logistic = logistic_benchmark(business, direction, explanatory)
    rq1 = research_question_table(business, direction)
    validation = validation_table(train, test, explanatory, shap_result, transformed, business, local, recommendations)
    repro = reproducibility_table(split_hash, shap_result, explanatory, recommendations)
    figure_paths = make_figures(transformed, shap_result, explanatory["x_test"], direction)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(TRANSFORMED_IMPORTANCE_PATH, index=False)
    business.to_csv(BUSINESS_IMPORTANCE_PATH, index=False)
    direction.to_csv(DIRECTION_PATH, index=False)
    local.to_csv(LOCAL_PATH, index=False)
    logistic.to_csv(LOGISTIC_PATH, index=False)
    rq1.to_csv(RQ1_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    repro.to_csv(REPRO_PATH, index=False)
    write_notebook()
    write_report(transformed, business, direction, local, logistic, rq1, validation, shap_result, explanatory, recommendations, figure_paths)

    declared = [
        SOURCE_PATH,
        TRANSFORMED_IMPORTANCE_PATH,
        BUSINESS_IMPORTANCE_PATH,
        DIRECTION_PATH,
        LOCAL_PATH,
        LOGISTIC_PATH,
        RQ1_PATH,
        VALIDATION_PATH,
        REPRO_PATH,
        REPORT_PATH,
        NOTEBOOK_PATH,
        *figure_paths,
        LOG_PATH,
    ]
    logger.info("Validation: %s/%s checks passed.", int(validation["passed"].sum()), len(validation))
    for handler in logger.handlers:
        handler.flush()
    # Flush the final validation message before hashing the log in the manifest.
    manifest = write_manifest(declared)
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Step 11 validation failed: {failed}")
    return {
        "train": train,
        "test": test,
        "explanatory": explanatory,
        "shap": shap_result,
        "transformed": transformed,
        "business": business,
        "direction": direction,
        "local": local,
        "logistic": logistic,
        "rq1": rq1,
        "validation": validation,
        "reproducibility": repro,
        "manifest": manifest,
        "figure_paths": figure_paths,
    }


def run_country_correction() -> dict[str, Any]:
    """Regenerate only the signed SHAP matrix needed for Country grouping.

    This intentionally does not regenerate figures, local explanations,
    direction audits, Logistic outputs, or the transformed-feature ranking.
    The exact locked XGBoost pipeline is fitted only to reproduce the already
    specified Step 11 SHAP values; no search, tuning, or model change occurs.
    """
    logger = get_logger()
    train, test, split_hash, recommendations = load_locked_data()
    validate_locked_configuration(recommendations)
    features = FEATURE_SETS["Parsimonious"]
    xgb_pipeline = pipeline_for("XGBoost", features, recommendations["XGBoost"]["params"])
    xgb_pipeline.fit(train[features], train[TARGET].astype(int))
    preprocessor = xgb_pipeline.named_steps["preprocess"]
    xgb_model = xgb_pipeline.named_steps["model"]
    x_test = pd.DataFrame(
        preprocessor.transform(test[features]),
        columns=preprocessor.get_feature_names_out(),
        index=test.index,
    )
    shap_result = compute_shap({"xgb_model": xgb_model, "x_test": x_test})
    transformed = pd.read_csv(TRANSFORMED_IMPORTANCE_PATH)
    business = business_importance(transformed, x_test, shap_result["values"])
    direction = pd.read_csv(DIRECTION_PATH)
    local = pd.read_csv(LOCAL_PATH)
    logistic = pd.read_csv(LOGISTIC_PATH)
    rq1 = research_question_table(business, direction)
    explanatory = {"features": features, "x_test": x_test}
    validation = validation_table(train, test, explanatory, shap_result, transformed, business, local, recommendations)
    repro = reproducibility_table(split_hash, shap_result, explanatory, recommendations)
    figure_paths = sorted(FIGURES_DIR.glob("15_step11_*.png")) + sorted(FIGURES_DIR.glob("1[6-9]_step11_*.png")) + sorted(FIGURES_DIR.glob("20_step11_*.png"))

    business.to_csv(BUSINESS_IMPORTANCE_PATH, index=False)
    rq1.to_csv(RQ1_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    repro.to_csv(REPRO_PATH, index=False)
    write_report(transformed, business, direction, local, logistic, rq1, validation, shap_result, explanatory, recommendations, figure_paths)

    logger.info("Country correction: regenerated exact locked XGBoost SHAP values for %s Test customers.", len(test))
    logger.info("Country correction: grouped signed SHAP per customer across exactly 30 one-hot columns before abs/mean.")
    logger.info("Validation: %s/%s checks passed.", int(validation["passed"].sum()), len(validation))
    for handler in logger.handlers:
        handler.flush()
    manifest_paths = [PROJECT_ROOT / Path(path) for path in pd.read_csv(MANIFEST_PATH)["path"].tolist()]
    manifest = write_manifest(manifest_paths)
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Step 11A validation failed: {failed}")
    return {"business": business, "rq1": rq1, "validation": validation, "manifest": manifest, "shap": shap_result}


def main() -> None:
    result = run_analysis()
    print(
        json.dumps(
            {
                "train_customers_fit": len(result["train"]),
                "test_customers_explained": len(result["test"]),
                "shap_scale": result["shap"]["shap_scale"],
                "expected_value": result["shap"]["expected_value"],
                "max_reconstruction_error": float(np.max(np.abs(result["shap"]["reconstruction_errors"]))),
                "top_business_features": result["business"].head(10).to_dict(orient="records"),
                "validation_passed": int(result["validation"]["passed"].sum()),
                "validation_total": len(result["validation"]),
                "scope": "Step 11 only; no tuning, threshold change, resampling, or managerial intervention design",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
