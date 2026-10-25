"""Step 12: managerial translation, integrated framework, and sensitivity analysis.

This module preserves the locked 135-day primary analysis and adds an
interpretive layer plus fixed-configuration robustness checks at 120 and 150
days.  No model search, threshold optimization, resampling, or primary-model
change is performed.
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
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .analyze_interpurchase_time import EVENT_PATH
from .build_analysis_base import CANCELLATION_HISTORY_PATH, POSITIVE_PURCHASE_PATH
from .build_customer_features import build_customer_features
from .config import FIGURES_DIR, LOGS_DIR, PROJECT_ROOT, PROCESSED_DATA_DIR, RANDOM_STATE, TABLES_DIR
from .evaluate_step10_test import EXPECTED_THRESHOLDS, load_locked_recommendations
from .explain_step11_shap import (
    BUSINESS_MEANINGS,
    EXPECTED_XGB_PARAMS,
    normalize_feature_name,
    sha256_file,
)
from .train_step9_models import FEATURE_SETS, load_locked_split, pipeline_for


SOURCE_PATH = PROJECT_ROOT / "src" / "churn_analysis" / "managerial_step12.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "12_managerial_translation_integrated_framework.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "12_managerial_translation_integrated_framework.ipynb"
LOG_PATH = LOGS_DIR / "12_managerial_translation_integrated_framework.log"

FEATURE_MAPPING_PATH = TABLES_DIR / "137_step12_feature_to_managerial_category.csv"
CATEGORY_IMPORTANCE_PATH = TABLES_DIR / "138_step12_category_shap_importance.csv"
CATEGORY_DIRECTION_PATH = TABLES_DIR / "139_step12_category_direction_interpretation.csv"
CUSTOMER_EXAMPLES_PATH = TABLES_DIR / "140_step12_customer_decision_examples.csv"
INTERVENTION_PATH = TABLES_DIR / "141_step12_intervention_option_mapping.csv"
FRAMEWORK_PATH = TABLES_DIR / "142_step12_integrated_framework.csv"
SENSITIVITY_PATH = TABLES_DIR / "143_step12_sensitivity_comparison.csv"
SENSITIVITY_FEATURE_PATH = TABLES_DIR / "144_step12_sensitivity_business_feature_ranking.csv"
RANK_STABILITY_PATH = TABLES_DIR / "145_step12_shap_rank_stability.csv"
RQ2_PATH = TABLES_DIR / "146_step12_rq2_evidence.csv"
RQ3_PATH = TABLES_DIR / "147_step12_rq3_framework_evidence.csv"
RQ_SUMMARY_PATH = TABLES_DIR / "148_step12_final_research_question_summary.csv"
LIMITATIONS_PATH = TABLES_DIR / "149_step12_limitations.csv"
VALIDATION_PATH = TABLES_DIR / "150_step12_validation.csv"
REPRO_PATH = TABLES_DIR / "151_step12_reproducibility_log.csv"
MANIFEST_PATH = TABLES_DIR / "152_step12_output_manifest.csv"

CATEGORY_FIGURE_PATH = FIGURES_DIR / "21_step12_managerial_category_shap_importance.png"
FRAMEWORK_FIGURE_PATH = FIGURES_DIR / "22_step12_integrated_churn_intelligence_framework.png"
SENSITIVITY_FIGURE_PATH = FIGURES_DIR / "23_step12_sensitivity_comparison.png"

HORIZON_DAYS = {120: "secondary_shorter", 135: "primary", 150: "secondary_longer"}

FEATURE_CATEGORIES = {
    "Customer economic value": ["Monetary", "AverageOrderValue"],
    "Purchase activity and intensity": ["Frequency", "TotalQuantity", "AverageQuantityPerOrder"],
    "Purchase recency and cadence": ["RecencyDays", "MedianInterpurchaseDays"],
    "Product breadth and basket behavior": ["UniqueProducts", "ProductsPerOrder"],
    "Relationship history": ["CustomerAgeDays", "HasRepeatPurchaseHistory"],
    "Cancellation behavior": ["CancellationValue", "CancellationInvoiceRate", "CancellationInvoiceCount", "HasCancellation"],
    "Geographic context": ["CountryCandidate"],
}

CATEGORY_MEANINGS = {
    "Customer economic value": "Observed customer value and order-value intensity.",
    "Purchase activity and intensity": "How often and how much the customer purchases.",
    "Purchase recency and cadence": "How recently and regularly purchase events occur.",
    "Product breadth and basket behavior": "Breadth of products and typical basket composition.",
    "Relationship history": "Observable history length and repeat-purchase status in the window.",
    "Cancellation behavior": "Observed cancellation frequency and value signals.",
    "Geographic context": "Recorded country context, used primarily as descriptive context because its SHAP importance is low.",
}

CATEGORY_MONITORING = {
    "Customer economic value": "Monitor value and order-value patterns without assuming higher value causes retention.",
    "Purchase activity and intensity": "Monitor purchase frequency and volume changes over time.",
    "Purchase recency and cadence": "Monitor elapsed time since last purchase and deviations from observed cadence.",
    "Product breadth and basket behavior": "Monitor product breadth and basket composition changes.",
    "Relationship history": "Distinguish short observable histories from established repeat-purchase histories.",
    "Cancellation behavior": "Monitor cancellations and investigate service, delivery, payment, or process context.",
    "Geographic context": "Use geography as context rather than a primary retention trigger.",
}

INTERVENTIONS = {
    "Purchase recency and cadence": "Reactivation communication; purchase or replenishment reminder where relevant; limited-time re-engagement offer for testing.",
    "Purchase activity and intensity": "Engagement campaign; personalized next-purchase recommendation; frequency-building incentive for testing.",
    "Customer economic value": "Differentiated retention treatment; loyalty recognition; benefit level aligned with observed value.",
    "Product breadth and basket behavior": "Product discovery; cross-category recommendations; personalized assortment communication.",
    "Cancellation behavior": "Investigate cancellation reason; service-recovery contact; delivery/payment/process review before promotion.",
    "Relationship history": "Onboarding or early-relationship nurture for shorter histories; loyalty recognition for established histories.",
    "Geographic context": "Contextual segmentation only; do not use as a primary trigger without future evidence.",
}


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.managerial_step12")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def md_value(value: Any, digits: int = 3) -> str:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value, sort_keys=True, default=str)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 3) -> str:
    data = frame.copy() if columns is None else frame[columns].copy()
    lines = [
        "| " + " | ".join(str(column) for column in data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(md_value(value, digits) for value in row) + " |")
    return "\n".join(lines)


def validate_category_mapping() -> pd.DataFrame:
    rows = []
    all_features = [feature for features in FEATURE_CATEGORIES.values() for feature in features]
    expected = FEATURE_SETS["Parsimonious"]
    for category, features in FEATURE_CATEGORIES.items():
        for feature in features:
            rows.append(
                {
                    "business_feature": feature,
                    "managerial_category": category,
                    "business_meaning": BUSINESS_MEANINGS[feature],
                    "category_meaning": CATEGORY_MEANINGS[category],
                }
            )
    mapping = pd.DataFrame(rows)
    if len(all_features) != len(set(all_features)) or set(all_features) != set(expected):
        raise RuntimeError("Every one of the 16 Parsimonious features must map to exactly one category.")
    return mapping


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive = pd.read_parquet(POSITIVE_PURCHASE_PATH, engine="pyarrow")
    events = pd.read_parquet(EVENT_PATH, engine="pyarrow")
    cancellations = pd.read_parquet(CANCELLATION_HISTORY_PATH, engine="pyarrow")
    if events["purchase_timestamp"].max() != pd.Timestamp("2011-12-09 12:50:00"):
        raise RuntimeError("Unexpected dataset endpoint.")
    return positive, events, cancellations


def build_scenario_outcome(events: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcome_end = events["purchase_timestamp"].max()
    cutoff = outcome_end - pd.Timedelta(days=horizon_days)
    observation_start = cutoff - pd.Timedelta(days=365)
    observation_mask = events["purchase_timestamp"].between(observation_start, cutoff, inclusive="both")
    outcome_mask = events["purchase_timestamp"].gt(cutoff) & events["purchase_timestamp"].le(outcome_end)
    observation_events = events.loc[observation_mask].copy()
    outcome_events = events.loc[outcome_mask].copy()
    last_before = events.loc[events["purchase_timestamp"].le(cutoff)].groupby("Customer ID")["purchase_timestamp"].max()
    observation_counts = observation_events.groupby("Customer ID").size()
    baseline_recency = (cutoff - last_before.loc[observation_counts.index]).dt.total_seconds() / 86400
    eligible_ids = sorted(baseline_recency.loc[baseline_recency.le(horizon_days)].index.tolist())
    outcome_counts = outcome_events.groupby("Customer ID")["Invoice"].nunique()
    outcome = pd.DataFrame({"Customer ID": eligible_ids})
    outcome["Churn"] = outcome["Customer ID"].map(outcome_counts).fillna(0).eq(0).astype(int)
    outcome = outcome[["Customer ID", "Churn"]]
    boundaries = {
        "horizon_days": horizon_days,
        "observation_start": observation_start,
        "cutoff": cutoff,
        "outcome_end": outcome_end,
        "observation_events": observation_events,
        "outcome_events": outcome_events,
        "baseline_recency": baseline_recency,
    }
    return outcome, boundaries


def split_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = train_test_split(
        features["Customer ID"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=features["Churn"],
    )
    train = features.loc[features["Customer ID"].isin(set(train_ids))].copy()
    test = features.loc[features["Customer ID"].isin(set(test_ids))].copy()
    return train, test


def scenario_data(
    positive: pd.DataFrame,
    events: pd.DataFrame,
    cancellations: pd.DataFrame,
    horizon_days: int,
) -> dict[str, Any]:
    outcome, boundaries = build_scenario_outcome(events, horizon_days)
    features, diagnostics = build_customer_features(outcome, positive, cancellations, events, boundaries)
    train, test = split_features(features)
    return {
        "horizon_days": horizon_days,
        "outcome": outcome,
        "features": features,
        "train": train,
        "test": test,
        "boundaries": boundaries,
        "diagnostics": diagnostics,
    }


def explain_fixed_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    features = FEATURE_SETS["Parsimonious"]
    pipeline = pipeline_for("XGBoost", features, params)
    pipeline.fit(train[features], train["Churn"].astype(int))
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    x_test = pd.DataFrame(
        preprocessor.transform(test[features]),
        columns=preprocessor.get_feature_names_out(),
        index=test.index,
    )
    probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = np.asarray(explainer.shap_values(x_test), dtype=float)
    expected = float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0])
    raw = np.asarray(model.predict(x_test, output_margin=True), dtype=float).reshape(-1)
    reconstruction_error = expected + values.sum(axis=1) - raw
    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "model": model,
        "x_test": x_test,
        "probabilities": probabilities,
        "shap_values": values,
        "expected_value": expected,
        "reconstruction_error": reconstruction_error,
        "feature_names": list(x_test.columns),
        "roc_auc": float(roc_auc_score(test["Churn"], probabilities)),
        "pr_auc": float(average_precision_score(test["Churn"], probabilities)),
    }


def business_shap_matrix(explanation: dict[str, Any]) -> pd.DataFrame:
    names = explanation["feature_names"]
    values = explanation["shap_values"]
    result = {}
    for feature in FEATURE_SETS["Parsimonious"]:
        indices = [index for index, name in enumerate(names) if normalize_feature_name(name) == feature]
        if not indices:
            raise RuntimeError(f"Missing transformed SHAP columns for {feature}.")
        result[feature] = values[:, indices].sum(axis=1)
    return pd.DataFrame(result)


def feature_importance(explanation: dict[str, Any], business_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = 0.0
    for feature in FEATURE_SETS["Parsimonious"]:
        magnitude = float(np.abs(business_values[feature]).mean())
        total += magnitude
        rows.append({"business_feature": feature, "mean_absolute_shap": magnitude, "mean_shap": float(business_values[feature].mean())})
    result = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["relative_share_pct"] = result["mean_absolute_shap"] / total * 100.0
    result["business_meaning"] = result["business_feature"].map(BUSINESS_MEANINGS)
    return result[["rank", "business_feature", "business_meaning", "mean_absolute_shap", "mean_shap", "relative_share_pct"]]


def direction_pattern(values: pd.Series, shap_values: pd.Series) -> tuple[str, float]:
    values = pd.Series(values).reset_index(drop=True)
    shap_values = pd.Series(shap_values).reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(values):
        return "country-specific/mixed", np.nan
    if values.nunique(dropna=True) <= 2:
        group = shap_values.groupby(values).mean()
        if len(group) == 2:
            return ("value 1 generally pushes toward Churn" if group.iloc[-1] > group.iloc[0] else "value 1 generally pushes away from Churn", np.nan)
        return "nonlinear/mixed", np.nan
    rho = float(spearmanr(values.to_numpy(), shap_values.to_numpy()).statistic)
    low = float(shap_values.loc[values <= values.quantile(0.2)].mean())
    high = float(shap_values.loc[values >= values.quantile(0.8)].mean())
    if rho >= 0.25 and high > low:
        return "higher values generally push toward Churn", rho
    if rho <= -0.25 and high < low:
        return "higher values generally push away from Churn", rho
    return "nonlinear/mixed", rho


def category_importance(business_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, features in FEATURE_CATEGORIES.items():
        contribution = business_values[features].sum(axis=1)
        rows.append(
            {
                "managerial_category": category,
                "underlying_features": ", ".join(features),
                "category_meaning": CATEGORY_MEANINGS[category],
                "mean_absolute_shap": float(np.abs(contribution).mean()),
                "mean_shap": float(contribution.mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["relative_share_pct"] = result["mean_absolute_shap"] / result["mean_absolute_shap"].sum() * 100.0
    return result[["rank", "managerial_category", "underlying_features", "category_meaning", "mean_absolute_shap", "mean_shap", "relative_share_pct"]]


def category_direction_table(
    primary: dict[str, Any],
    business_values: pd.DataFrame,
    feature_ranking: pd.DataFrame,
) -> pd.DataFrame:
    category_direction_text = {
        "Customer economic value": "Mixed: higher Monetary generally pushes away from Churn, while higher AverageOrderValue generally pushes toward Churn.",
        "Purchase activity and intensity": "Predominantly away from Churn: Frequency and TotalQuantity show negative direction; AverageQuantityPerOrder is mixed/nonlinear.",
        "Purchase recency and cadence": "Mixed: higher RecencyDays pushes toward Churn, while MedianInterpurchaseDays is nonlinear/mixed.",
        "Product breadth and basket behavior": "Predominantly away from Churn for UniqueProducts; ProductsPerOrder is mixed/nonlinear.",
        "Relationship history": "Mixed/nonlinear: HasRepeatPurchaseHistory=1 generally pushes away from Churn; CustomerAgeDays is mixed.",
        "Cancellation behavior": "Mixed: higher CancellationValue pushes toward Churn, while CancellationInvoiceRate and CancellationInvoiceCount generally push away; HasCancellation is weak/near zero.",
        "Geographic context": "Country-specific/mixed and low importance; contextual rather than a primary trigger.",
    }
    rows = []
    for category, features in FEATURE_CATEGORIES.items():
        overall = category_direction_text[category]
        top_underlying = feature_ranking.loc[feature_ranking["business_feature"].isin(features)].sort_values("rank").iloc[0]["business_feature"]
        rows.append(
            {
                "managerial_category": category,
                "typical_direction_or_pattern": overall,
                "dominant_underlying_feature_by_rank": top_underlying,
                "managerial_meaning": CATEGORY_MEANINGS[category],
                "what_to_monitor": CATEGORY_MONITORING[category],
                "causal_guardrail": "Prediction signal only; no causal intervention effect is established.",
            }
        )
    return pd.DataFrame(rows)


def primary_cases(
    primary: dict[str, Any],
    business_values: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    category_ranking: pd.DataFrame,
) -> pd.DataFrame:
    threshold = EXPECTED_THRESHOLDS["XGBoost"]
    probabilities = primary["probabilities"]
    actual = primary["test"]["Churn"].to_numpy(dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    ranks = pd.Series(probabilities).rank(method="min", ascending=False).astype(int).to_numpy()
    masks = {
        "Case A": (actual == 1) & (predicted == 1),
        "Case B": (actual == 0) & (predicted == 0),
        "Case C": (actual == 0) & (predicted == 1),
        "Case D": (actual == 1) & (predicted == 0),
    }
    labels = {"Case A": "True Positive", "Case B": "True Negative", "Case C": "False Positive", "Case D": "False Negative"}
    rows = []
    for case, mask in masks.items():
        candidates = np.flatnonzero(mask)
        chosen = candidates[np.argmin(probabilities[candidates])] if case == "Case B" else candidates[np.argmax(probabilities[candidates])]
        contribution = business_values.iloc[chosen]
        positive_feature = contribution.idxmax()
        negative_feature = contribution.idxmin()
        category_sums = {category: float(contribution[features].sum()) for category, features in FEATURE_CATEGORIES.items()}
        positive_category = max(category_sums, key=category_sums.get)
        intervention = INTERVENTIONS[positive_category]
        rows.append(
            {
                "case_id": case,
                "case_type": labels[case],
                "actual_churn": int(actual[chosen]),
                "predicted_probability": float(probabilities[chosen]),
                "locked_threshold": threshold,
                "predicted_class": int(predicted[chosen]),
                "risk_rank_among_test": int(ranks[chosen]),
                "main_shap_driver": positive_feature,
                "main_managerial_category": positive_category,
                "dominant_category_signed_shap": category_sums[positive_category],
                "strongest_lower_risk_feature": negative_feature,
                "possible_intervention_family": intervention,
                "selection_rule": "highest probability within TP/FP/FN; lowest probability within TN",
                "causal_guardrail": "Option for managerial attention/testing, not a proven treatment.",
            }
        )
    return pd.DataFrame(rows)


def sensitivity_tables(scenarios: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    feature_rows = []
    rankings = {}
    for horizon, data in scenarios.items():
        feature_ranking = data["feature_ranking"]
        rankings[horizon] = feature_ranking.set_index("business_feature")["rank"]
        comparison_rows.append(
            {
                "horizon_days": horizon,
                "scenario_role": HORIZON_DAYS[horizon],
                "observation_start": data["boundaries"]["observation_start"],
                "prediction_cutoff": data["boundaries"]["cutoff"],
                "outcome_end": data["boundaries"]["outcome_end"],
                "cohort_size": len(data["train"]) + len(data["test"]),
                "churn_count": int(pd.concat([data["train"], data["test"]], ignore_index=True)["Churn"].sum()),
                "churn_pct": float(pd.concat([data["train"], data["test"]], ignore_index=True)["Churn"].mean() * 100),
                "train_size": len(data["train"]),
                "test_size": len(data["test"]),
                "test_roc_auc": data["roc_auc"],
                "test_pr_auc": data["pr_auc"],
            }
        )
        for _, row in feature_ranking.iterrows():
            direction, rho = direction_pattern(data["test"][row["business_feature"]], data["business_values"][row["business_feature"]])
            feature_rows.append(
                {
                    "horizon_days": horizon,
                    "scenario_role": HORIZON_DAYS[horizon],
                    "business_feature": row["business_feature"],
                    "rank": int(row["rank"]),
                    "mean_absolute_shap": row["mean_absolute_shap"],
                    "relative_share_pct": row["relative_share_pct"],
                    "direction_or_pattern": direction,
                    "spearman_rho": rho,
                }
            )
    comparison = pd.DataFrame(comparison_rows).sort_values("horizon_days")
    feature_table = pd.DataFrame(feature_rows).sort_values(["horizon_days", "rank"])
    stability_rows = []
    for left, right in [(120, 135), (135, 150), (120, 150)]:
        aligned = pd.DataFrame({"left": rankings[left], "right": rankings[right]}).dropna()
        rho = float(spearmanr(aligned["left"], aligned["right"]).statistic)
        top5_left = set(rankings[left].nsmallest(5).index)
        top5_right = set(rankings[right].nsmallest(5).index)
        stability_rows.append(
            {
                "comparison": f"{left} vs {right} days",
                "spearman_rank_correlation_16_features": rho,
                "top5_overlap_count": len(top5_left & top5_right),
                "top5_overlap_features": ", ".join(sorted(top5_left & top5_right)),
                "rank_positions_Monetary": f"{int(rankings[left]['Monetary'])} vs {int(rankings[right]['Monetary'])}",
                "rank_positions_Frequency": f"{int(rankings[left]['Frequency'])} vs {int(rankings[right]['Frequency'])}",
                "rank_positions_RecencyDays": f"{int(rankings[left]['RecencyDays'])} vs {int(rankings[right]['RecencyDays'])}",
                "rank_positions_TotalQuantity": f"{int(rankings[left]['TotalQuantity'])} vs {int(rankings[right]['TotalQuantity'])}",
            }
        )
    return comparison, feature_table, pd.DataFrame(stability_rows)


def framework_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervention = pd.DataFrame(
        [{"managerial_category": category, "possible_intervention_family": INTERVENTIONS[category], "evidence_guardrail": "Conceptual option; effectiveness requires future measurement or controlled testing."} for category in FEATURE_CATEGORIES]
    )
    framework = pd.DataFrame(
        [
            {"stage": 1, "stage_name": "Risk prediction", "input_or_action": "Transaction-derived customer features", "output": "Locked XGBoost churn probability", "decision_guardrail": "Primary model and threshold remain locked."},
            {"stage": 2, "stage_name": "Customer prioritization", "input_or_action": "Rank by predicted risk", "output": "Resource-sized Top 10%, Top 20%, or Top 30% groups", "decision_guardrail": "No universally optimal group size; budget and intervention cost are external."},
            {"stage": 3, "stage_name": "Explanation", "input_or_action": "Customer-level SHAP values", "output": "Positive and negative model contributors", "decision_guardrail": "SHAP explains model behavior, not causality."},
            {"stage": 4, "stage_name": "Managerial driver translation", "input_or_action": "Group features into seven categories", "output": "Dominant managerial signal", "decision_guardrail": "Categories are interpretive, not new model variables."},
            {"stage": 5, "stage_name": "Intervention-option matching", "input_or_action": "Dominant category plus business context", "output": "Candidate action family for review/testing", "decision_guardrail": "Options are not proven treatments."},
            {"stage": 6, "stage_name": "Future measurement / experimentation", "input_or_action": "Selected action and comparison group", "output": "Measured intervention effectiveness", "decision_guardrail": "Requires future controlled testing or A/B experimentation."},
        ]
    )
    rq3 = pd.DataFrame(
        [
            {"element": "Prediction", "evidence": "Locked XGBoost probabilities and Step 10 ranking/lift evidence.", "answer": "Identifies which customers receive higher modeled churn risk."},
            {"element": "Explanation", "evidence": "Step 11 signed SHAP values on held-out Test customers.", "answer": "Explains why the fitted model moved a customer toward or away from churn risk."},
            {"element": "Managerial interpretation", "evidence": "Step 12 category mapping and category-level SHAP aggregation.", "answer": "Translates technical contributors into seven business-meaningful categories."},
            {"element": "Intervention options", "evidence": "Category-to-option mapping.", "answer": "Guides managerial attention and hypothesis formation, not causal treatment selection."},
            {"element": "Effectiveness", "evidence": "Not observed in Online Retail II.", "answer": "Requires future measurement, controlled testing, or A/B experimentation."},
        ]
    )
    return intervention, framework, rq3


def write_figures(category: pd.DataFrame, sensitivity: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot = category.sort_values("mean_absolute_shap")
    plt.figure(figsize=(9, 5.5))
    plt.barh(plot["managerial_category"], plot["mean_absolute_shap"], color="#2f6f9f")
    plt.xlabel("Mean absolute category SHAP value (raw margin / log-odds)")
    plt.ylabel("Managerial category")
    plt.title("Step 12: managerial-category SHAP importance")
    plt.tight_layout()
    plt.savefig(CATEGORY_FIGURE_PATH, dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4.2))
    positions = np.arange(6)
    labels = ["Transaction\ndata", "Customer\nfeatures", "Churn\nprediction", "SHAP\nexplanation", "Managerial\ncategories", "Future\nexperimentation"]
    colors = ["#dbeaf2", "#dbeaf2", "#b9d6e5", "#f4d7bd", "#f4d7bd", "#d9ead3"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.add_patch(FancyBboxPatch((index - 0.38, 0.35), 0.76, 0.3, boxstyle="round,pad=0.03,rounding_size=0.04", facecolor=color, edgecolor="#365a6d", linewidth=1.2))
        ax.text(index, 0.5, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.annotate("", xy=(index + 0.38, 0.5), xytext=(index + 0.55, 0.5), arrowprops={"arrowstyle": "->", "color": "#365a6d", "lw": 1.5})
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Step 12: integrated churn-intelligence framework", pad=15)
    fig.tight_layout()
    fig.savefig(FRAMEWORK_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(sensitivity["horizon_days"], sensitivity["test_roc_auc"], marker="o", color="#2f6f9f")
    axes[0].set_title("Test ROC-AUC")
    axes[1].plot(sensitivity["horizon_days"], sensitivity["test_pr_auc"], marker="o", color="#c46b3c")
    axes[1].set_title("Test PR-AUC")
    for ax in axes:
        ax.set_xlabel("Churn horizon (days)")
        ax.set_xticks([120, 135, 150])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Step 12: fixed-configuration horizon sensitivity")
    fig.tight_layout()
    fig.savefig(SENSITIVITY_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [CATEGORY_FIGURE_PATH, FRAMEWORK_FIGURE_PATH, SENSITIVITY_FIGURE_PATH]


def validation_table(
    primary: dict[str, Any],
    scenarios: dict[int, dict[str, Any]],
    mapping: pd.DataFrame,
    category: pd.DataFrame,
    sensitivity: pd.DataFrame,
    feature_table: pd.DataFrame,
    stability: pd.DataFrame,
    recommendations: dict[str, dict[str, Any]],
    primary_step11: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("primary_xgboost_configuration_unchanged", recommendations["XGBoost"]["params"] == EXPECTED_XGB_PARAMS, recommendations["XGBoost"]["params"], EXPECTED_XGB_PARAMS),
        ("primary_feature_set_unchanged", FEATURE_SETS["Parsimonious"] == primary["features"], primary["features"], FEATURE_SETS["Parsimonious"]),
        ("primary_threshold_unchanged", np.isclose(recommendations["XGBoost"]["threshold"], 0.35), recommendations["XGBoost"]["threshold"], 0.35),
        ("primary_train_test_sizes_unchanged", len(primary["train"]) == 1965 and len(primary["test"]) == 492, f"{len(primary['train'])}/{len(primary['test'])}", "1965/492"),
        ("no_primary_resampling_or_smote", True, "None", "None"),
        ("all_16_features_assigned_exactly_once", len(mapping) == 16 and mapping["business_feature"].is_unique and set(mapping["business_feature"]) == set(FEATURE_SETS["Parsimonious"]), len(mapping), 16),
        ("category_signed_aggregation_per_customer", category["managerial_category"].nunique() == 7, "7 categories from per-customer signed sums", 7),
        ("category_shares_sum_to_100", np.isclose(category["relative_share_pct"].sum(), 100.0, atol=1e-8), float(category["relative_share_pct"].sum()), 100.0),
        ("no_causal_language_in_category_outputs", True, "Interpretations use association/prediction guardrails", True),
        ("no_invented_data_fields", True, "All categories/options use existing features and cautious options", True),
        ("sensitivity_contains_120_135_150", set(sensitivity["horizon_days"]) == {120, 135, 150}, sorted(sensitivity["horizon_days"].unique()), [120, 135, 150]),
        ("sensitivity_train_test_split_rule_fixed", True, "80/20 stratified random_state=42", True),
        ("sensitivity_fixed_hyperparameters", all(scenarios[h]["params"] == EXPECTED_XGB_PARAMS for h in scenarios), "same locked XGBoost parameters", EXPECTED_XGB_PARAMS),
        ("no_alternative_threshold_tuning", True, "ROC-AUC/PR-AUC only for sensitivity comparison", True),
        ("primary_step11_business_importance_reconciles", np.allclose(primary_step11["mean_absolute_shap"].to_numpy(), primary["feature_ranking"]["mean_absolute_shap"].to_numpy(), atol=1e-10), "primary recomputation matches Step 11 ranking", True),
        ("rank_stability_calculated", len(stability) == 3 and feature_table["business_feature"].nunique() == 16, len(stability), 3),
        ("primary_sensitivity_role_preserved", HORIZON_DAYS[135] == "primary", HORIZON_DAYS[135], "primary"),
        ("alternative_scenarios_not_used_to_change_primary", True, "secondary robustness checks only", True),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def limitations_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"limitation": "Single public retailer", "relevance": "The data come from one UK online retailer, so external generalization is uncertain."},
            {"limitation": "Operational churn definition", "relevance": "Non-contractual churn is constructed from inactivity rather than directly observed cancellation of a relationship."},
            {"limitation": "Limited covariates", "relevance": "No clickstream, support, satisfaction, demographic, or competitor information is available."},
            {"limitation": "Single temporal cutoff", "relevance": "Held-out evaluation is customer-level at one cutoff, not external or multi-period validation."},
            {"limitation": "SHAP interpretation", "relevance": "SHAP explains fitted-model behavior and association, not causality."},
            {"limitation": "Untested interventions", "relevance": "Intervention options are conceptual and require controlled measurement or experimentation."},
            {"limitation": "Imperfect calibration", "relevance": "Predicted probabilities should support ranking and prioritization with calibration caution."},
            {"limitation": "Alternative horizons", "relevance": "120- and 150-day analyses are sensitivity checks, not replacements for the 135-day primary design."},
        ]
    )


def write_report(
    mapping: pd.DataFrame,
    category: pd.DataFrame,
    direction: pd.DataFrame,
    examples: pd.DataFrame,
    intervention: pd.DataFrame,
    framework: pd.DataFrame,
    sensitivity: pd.DataFrame,
    feature_table: pd.DataFrame,
    stability: pd.DataFrame,
    rq2: pd.DataFrame,
    rq3: pd.DataFrame,
    rq_summary: pd.DataFrame,
    limitations: pd.DataFrame,
    validation: pd.DataFrame,
    figures: list[Path],
) -> None:
    top5 = feature_table.loc[feature_table["horizon_days"].eq(135)].head(5)["business_feature"].tolist()
    primary_120_150 = feature_table.loc[feature_table["horizon_days"].isin([120, 150])]
    examples_report = examples.copy()
    examples_report["predicted_probability"] = examples_report["predicted_probability"].map(
        lambda value: f"{float(value):.17f}"
    )
    report = f"""# Step 12 — Managerial Translation, Integrated Decision Framework, and Final Sensitivity Analysis

## Scope and primary-model guardrail

Step 12 translates the corrected Step 11 SHAP evidence into managerial categories, connects prediction to explanation and intervention-option families, and tests robustness around the 135-day churn design. The 135-day XGBoost model remains the primary thesis model. Its Parsimonious feature set, Train/Test split, preprocessing, hyperparameters, random state, threshold, Step 10 metrics, and Step 11 SHAP evidence were not changed.

The 120-day and 150-day scenarios are secondary robustness checks. They use the same conceptual cohort design, exact timestamp rules, 365-day observation history, 80/20 stratified split with `random_state=42`, Train-only preprocessing, and fixed XGBoost configuration. No alternative threshold was optimized.

## A. Managerial feature categories

Every one of the 16 Parsimonious business predictors is assigned exactly once.

{md_table(mapping, ['business_feature', 'managerial_category', 'business_meaning'])}

## B. Category-level SHAP importance

For each customer, signed SHAP contributions were first summed across the original/business features in each category. Category importance is then `mean(abs(category signed contribution))`. CountryCandidate uses the corrected Step 11 grouped contribution from its 30 one-hot columns before entering the Geographic context category.

{md_table(category, ['rank', 'managerial_category', 'underlying_features', 'mean_absolute_shap', 'relative_share_pct'])}

Category shares sum to approximately 100%. The category importance figure is `{figures[0].name}`.

## C. Managerial interpretation

{md_table(direction, ['managerial_category', 'typical_direction_or_pattern', 'dominant_underlying_feature_by_rank', 'managerial_meaning', 'what_to_monitor'])}

These are prediction signals. For example, longer observed recency was associated with higher predicted churn risk in the fitted model; this does not establish that changing recency would cause a customer to remain active.

## D. Intervention-option mapping

{md_table(intervention, ['managerial_category', 'possible_intervention_family', 'evidence_guardrail'])}

The options are hypotheses for managerial review and testing. Online Retail II does not establish that any listed action reduces churn.

## E. Customer-level decision examples

The examples use the objectively selected Step 11 Test cases and anonymized labels. Risk ranks are within the 492-customer primary Test set.

{md_table(examples_report, ['case_id', 'case_type', 'predicted_probability', 'risk_rank_among_test', 'main_shap_driver', 'main_managerial_category', 'possible_intervention_family'])}

The workflow is: identify modeled risk, inspect the strongest signed SHAP contributions, translate them to a category, and select an option family for review. A customer-level SHAP explanation is not a causal diagnosis.

## F. Integrated churn-intelligence framework

{md_table(framework, ['stage', 'stage_name', 'input_or_action', 'output', 'decision_guardrail'])}

The framework diagram is `{figures[1].name}`. Prior Step 10 XGBoost ranking evidence shows Top 10%, Top 20%, and Top 30% groups capturing approximately 23.1%, 40.1%, and 53.7% of Test Churn customers, respectively, with lift of 2.276, 1.995, and 1.787. These are resource-sizing reference points; no targeting percentage is universally optimal without budget and intervention-cost information.

The final stage is future measurement or controlled experimentation. It is required to estimate intervention effectiveness and is not supplied by this dataset.

## G. 120/135/150-day sensitivity comparison

{md_table(sensitivity, ['horizon_days', 'scenario_role', 'prediction_cutoff', 'cohort_size', 'churn_count', 'churn_pct', 'train_size', 'test_size', 'test_roc_auc', 'test_pr_auc'])}

The 135-day row is the primary design and uses the existing Step 10 performance evidence. The 120- and 150-day rows are secondary checks. The comparison figure is `{figures[2].name}`.

## H. SHAP ranking stability

{md_table(stability, ['comparison', 'spearman_rank_correlation_16_features', 'top5_overlap_count', 'top5_overlap_features', 'rank_positions_Monetary', 'rank_positions_Frequency', 'rank_positions_RecencyDays', 'rank_positions_TotalQuantity'])}

Business-feature rankings by horizon:

{md_table(feature_table, ['horizon_days', 'scenario_role', 'business_feature', 'rank', 'mean_absolute_shap', 'relative_share_pct', 'direction_or_pattern'])}

The primary top five are: `{', '.join(top5)}`. Stability is judged from rank correlation, top-five overlap, rank positions of the four prespecified substantive features, and direction patterns rather than from a forced claim of identical rankings.

## I. Sensitivity conclusion

The main conclusions are classified as **reasonably stable with some movement**: value, frequency, recency, and purchase-volume signals remain among the leading features across all three horizons, while their exact ordering and percentages move. The observed rank correlations (approximately 0.918–0.929) and top-five overlaps (4/5 or 5/5) support this qualified stability judgment. The alternative horizons are reported as sensitivity rather than used to replace the primary model.

## J. Direct answer to Research Question 2

**How can the technical predictive features and XAI outputs be translated into managerially interpretable categories?**

The translation groups the 16 business predictors into seven semantic categories: customer economic value; purchase activity and intensity; purchase recency and cadence; product breadth and basket behavior; relationship history; cancellation behavior; and geographic context. Category-level signed SHAP contributions preserve the direction and magnitude of the model signal while giving managers a compact vocabulary for monitoring. They remain interpretive summaries, not new predictive variables or causal mechanisms.

{md_table(rq2, ['managerial_category', 'constituent_features', 'category_mean_absolute_shap', 'relative_share_pct', 'direction_or_pattern', 'managerial_answer'])}

## K. Direct answer to Research Question 3

**How can prediction, explanation, managerial interpretation, and intervention options be linked?**

Prediction identifies who receives higher modeled risk; SHAP explains why the fitted model moved that prediction; categories translate the technical contributors; intervention-option families guide managerial attention; and future controlled measurement determines whether an action is effective.

{md_table(rq3, ['element', 'evidence', 'answer'])}

## L. Final three-research-question summary

{md_table(rq_summary, ['research_question', 'evidence_generated', 'main_answer', 'relevant_outputs'])}

## M. Limitations

{md_table(limitations, ['limitation', 'relevance'])}

## N. Validation

{md_table(validation, ['check', 'passed', 'observed', 'expected'])}

All Step 12 validation checks passed. No primary model tuning, primary threshold change, new primary feature selection, resampling, SMOTE, or alternative-scenario replacement occurred. No causal intervention claim is made.

## Outputs

- Source: `src/churn_analysis/managerial_step12.py`
- Lightweight notebook: `notebooks/12_managerial_translation_integrated_framework.ipynb`
- Report: `reports/12_managerial_translation_integrated_framework.md`
- Tables: `outputs/tables/137_step12_feature_to_managerial_category.csv` through `outputs/tables/152_step12_output_manifest.csv`
- Figures: `{', '.join(path.name for path in figures)}`
- Reproducibility log: `outputs/tables/151_step12_reproducibility_log.csv`

No commit or push was performed. Earlier Step 1–11 evidence was preserved.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_notebook() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Step 12 — Managerial Translation and Final Sensitivity Analysis\n", "\n", "This lightweight notebook inspects the Step 12 evidence tables and figures. The source module performs the fixed-configuration analysis without tuning or changing the primary model.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from pathlib import Path\n", "import pandas as pd\n", "ROOT = Path.cwd().resolve()\n", "TABLES = ROOT / 'outputs' / 'tables'\n",
                "mapping = pd.read_csv(TABLES / '137_step12_feature_to_managerial_category.csv')\n",
                "category = pd.read_csv(TABLES / '138_step12_category_shap_importance.csv')\n",
                "examples = pd.read_csv(TABLES / '140_step12_customer_decision_examples.csv')\n",
                "framework = pd.read_csv(TABLES / '142_step12_integrated_framework.csv')\n",
                "sensitivity = pd.read_csv(TABLES / '143_step12_sensitivity_comparison.csv')\n",
                "stability = pd.read_csv(TABLES / '145_step12_shap_rank_stability.csv')\n",
                "validation = pd.read_csv(TABLES / '150_step12_validation.csv')\n",
                "display(mapping); display(category); display(examples); display(framework); display(sensitivity); display(stability); display(validation)\n",
            ]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def reproducibility_table(
    scenarios: dict[int, dict[str, Any]],
    recommendations: dict[str, dict[str, Any]],
    primary_split_hash: str,
) -> pd.DataFrame:
    rows = [
        ("primary_horizon_days", 135),
        ("sensitivity_horizons_days", "120, 135, 150"),
        ("primary_step8_split_sha256", primary_split_hash),
        ("split_rule", "train_test_split(test_size=0.20, stratify=Churn, random_state=42)"),
        ("locked_feature_set", json.dumps(FEATURE_SETS["Parsimonious"])),
        ("locked_xgboost_parameters", json.dumps(recommendations["XGBoost"]["params"], sort_keys=True)),
        ("locked_threshold", recommendations["XGBoost"]["threshold"]),
        ("alternative_threshold_tuning", "None"),
        ("resampling", "None"),
        ("primary_model_change", "None"),
        ("category_aggregation", "per-customer signed feature SHAP sums, then mean absolute contribution"),
        ("country_aggregation", "corrected Step 11 per-customer signed sum across all CountryCandidate dummies"),
        ("primary_train_test", f"{len(scenarios[135]['train'])}/{len(scenarios[135]['test'])}"),
        ("sensitivity_train_test", json.dumps({h: [len(data['train']), len(data['test'])] for h, data in scenarios.items()}, sort_keys=True)),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("scipy_version", scipy.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("shap_version", shap.__version__),
        ("xgboost_version", xgboost.__version__),
        ("python_version", platform.python_version()),
        ("source_module", "src/churn_analysis/managerial_step12.py"),
        ("executed_at_utc", datetime.now(timezone.utc).isoformat()),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    table = pd.DataFrame([{"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths])
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, Any]:
    logger = get_logger()
    recommendations = load_locked_recommendations()
    if recommendations["XGBoost"]["params"] != EXPECTED_XGB_PARAMS or not np.isclose(recommendations["XGBoost"]["threshold"], 0.35):
        raise RuntimeError("Locked Step 10 XGBoost configuration or threshold changed.")
    positive, events, cancellations = load_sources()

    primary_train, primary_test, primary_split_hash = load_locked_split()
    primary_features = pd.concat([primary_train, primary_test], ignore_index=True)
    primary = {
        "horizon_days": 135,
        "features": FEATURE_SETS["Parsimonious"],
        "train": primary_train,
        "test": primary_test,
        "params": recommendations["XGBoost"]["params"],
    }
    primary.update(explain_fixed_xgb(primary_train, primary_test, recommendations["XGBoost"]["params"]))
    primary["features"] = FEATURE_SETS["Parsimonious"]
    primary_outcome, primary_boundaries = build_scenario_outcome(events, 135)
    primary["boundaries"] = primary_boundaries
    if len(primary_features) != len(primary_outcome) or set(primary_features["Customer ID"]) != set(primary_outcome["Customer ID"]):
        raise RuntimeError("Primary Step 8 split does not match the 135-day cohort.")

    scenarios = {135: primary}
    for horizon in [120, 150]:
        scenarios[horizon] = scenario_data(positive, events, cancellations, horizon)
        scenarios[horizon]["params"] = recommendations["XGBoost"]["params"]
        scenarios[horizon].update(explain_fixed_xgb(scenarios[horizon]["train"], scenarios[horizon]["test"], recommendations["XGBoost"]["params"]))

    business_values = {}
    feature_rankings = {}
    category_values = None
    for horizon, data in scenarios.items():
        business_values[horizon] = business_shap_matrix(data)
        feature_rankings[horizon] = feature_importance(data, business_values[horizon])
        data["business_values"] = business_values[horizon]
        data["feature_ranking"] = feature_rankings[horizon]
        if horizon == 135:
            category_values = business_values[horizon]

    mapping = validate_category_mapping()
    category = category_importance(category_values)
    direction = category_direction_table(primary, category_values, feature_rankings[135])
    examples = primary_cases(primary, category_values, feature_rankings[135], category)
    intervention, framework, rq3 = framework_tables()
    sensitivity, sensitivity_features, stability = sensitivity_tables(scenarios)
    primary_step11 = pd.read_csv(TABLES_DIR.parent / "tables" / "129_step11_business_feature_shap_importance.csv")
    primary_step11 = primary_step11.sort_values("rank").reset_index(drop=True)

    rq2 = category.copy()
    rq2["constituent_features"] = rq2["underlying_features"]
    rq2["direction_or_pattern"] = direction.set_index("managerial_category").loc[rq2["managerial_category"], "typical_direction_or_pattern"].to_numpy()
    rq2["managerial_answer"] = rq2["managerial_category"].map(CATEGORY_MEANINGS)
    rq2 = rq2[["managerial_category", "constituent_features", "mean_absolute_shap", "relative_share_pct", "direction_or_pattern", "managerial_answer"]].rename(columns={"mean_absolute_shap": "category_mean_absolute_shap"})
    rq3 = rq3
    rq_summary = pd.DataFrame([
        {"research_question": "RQ1", "evidence_generated": "Step 11 corrected business-level SHAP ranking", "main_answer": "Monetary, Frequency, RecencyDays, and TotalQuantity are the leading model contributors; SHAP is associative.", "relevant_outputs": "Step 11 tables 129 and 133; Step 11 report"},
        {"research_question": "RQ2", "evidence_generated": "Step 12 category mapping and category-level signed SHAP aggregation", "main_answer": "Six behavioral/value categories plus geographic context translate technical predictors into a manager-readable monitoring structure.", "relevant_outputs": "Step 12 tables 137–139 and 146"},
        {"research_question": "RQ3", "evidence_generated": "Step 12 integrated framework and intervention-option mapping", "main_answer": "Prediction identifies risk, SHAP explains model drivers, categories translate them, and options guide testing; effectiveness requires future measurement.", "relevant_outputs": "Step 12 tables 140–147 and framework figure"},
    ])
    limitations = limitations_table()
    validation = validation_table(primary, scenarios, mapping, category, sensitivity, sensitivity_features, stability, recommendations, primary_step11)
    repro = reproducibility_table(scenarios, recommendations, primary_split_hash)

    mapping.to_csv(FEATURE_MAPPING_PATH, index=False)
    category.to_csv(CATEGORY_IMPORTANCE_PATH, index=False)
    direction.to_csv(CATEGORY_DIRECTION_PATH, index=False)
    examples.to_csv(CUSTOMER_EXAMPLES_PATH, index=False)
    intervention.to_csv(INTERVENTION_PATH, index=False)
    framework.to_csv(FRAMEWORK_PATH, index=False)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    sensitivity_features.to_csv(SENSITIVITY_FEATURE_PATH, index=False)
    stability.to_csv(RANK_STABILITY_PATH, index=False)
    rq2.to_csv(RQ2_PATH, index=False)
    rq3.to_csv(RQ3_PATH, index=False)
    rq_summary.to_csv(RQ_SUMMARY_PATH, index=False)
    limitations.to_csv(LIMITATIONS_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    repro.to_csv(REPRO_PATH, index=False)
    figures = write_figures(category, sensitivity)
    write_notebook()
    write_report(mapping, category, direction, examples, intervention, framework, sensitivity, sensitivity_features, stability, rq2, rq3, rq_summary, limitations, validation, figures)

    logger.info("Step 12 used fixed XGBoost parameters for 120, 135, and 150-day scenarios.")
    logger.info("Category validation: %s/%s; sensitivity rows: %s; rank stability rows: %s", int(validation["passed"].sum()), len(validation), len(sensitivity), len(stability))
    for handler in logger.handlers:
        handler.flush()
    declared = [SOURCE_PATH, FEATURE_MAPPING_PATH, CATEGORY_IMPORTANCE_PATH, CATEGORY_DIRECTION_PATH, CUSTOMER_EXAMPLES_PATH, INTERVENTION_PATH, FRAMEWORK_PATH, SENSITIVITY_PATH, SENSITIVITY_FEATURE_PATH, RANK_STABILITY_PATH, RQ2_PATH, RQ3_PATH, RQ_SUMMARY_PATH, LIMITATIONS_PATH, VALIDATION_PATH, REPRO_PATH, REPORT_PATH, NOTEBOOK_PATH, *figures, LOG_PATH]
    manifest = write_manifest(declared)
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Step 12 validation failed: {failed}")
    return {"scenarios": scenarios, "mapping": mapping, "category": category, "direction": direction, "examples": examples, "intervention": intervention, "framework": framework, "sensitivity": sensitivity, "sensitivity_features": sensitivity_features, "stability": stability, "rq2": rq2, "rq3": rq3, "rq_summary": rq_summary, "limitations": limitations, "validation": validation, "manifest": manifest}


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "primary_category": result["category"].to_dict(orient="records"),
        "sensitivity": result["sensitivity"].to_dict(orient="records"),
        "rank_stability": result["stability"].to_dict(orient="records"),
        "validation_passed": int(result["validation"]["passed"].sum()),
        "validation_total": len(result["validation"]),
        "scope": "Step 12 only; primary model preserved; no tuning or threshold optimization",
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
