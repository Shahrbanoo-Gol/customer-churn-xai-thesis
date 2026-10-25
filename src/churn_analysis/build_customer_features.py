"""Step 7: build and audit customer-level predictive features.

This module is deliberately limited to descriptive feature engineering from
the locked Step 6 observation window.  It does not split data, preprocess
features for a model, train a model, or inspect SHAP values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_analysis_base import (
    CANCELLATION_HISTORY_PATH,
    CANONICAL_PATH,
    POSITIVE_PURCHASE_PATH,
    find_raw_workbook,
    sha256_file,
)
from .analyze_interpurchase_time import EVENT_PATH
from .config import FIGURES_DIR, LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR


STEP6_OUTCOME_PATH = PROCESSED_DATA_DIR / "06_primary_customer_outcome_table.parquet"
FEATURE_PATH = PROCESSED_DATA_DIR / "07_customer_features.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "07_customer_feature_engineering_and_audit.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_customer_feature_engineering_and_audit.ipynb"
LOG_PATH = LOGS_DIR / "07_customer_feature_engineering_and_audit.log"

QUALITY_PATH = TABLES_DIR / "82_step7_feature_quality_audit.csv"
COUNTRY_PATH = TABLES_DIR / "83_step7_country_audit.csv"
CORRELATION_PATH = TABLES_DIR / "84_step7_correlation_audit.csv"
RECONCILIATION_PATH = TABLES_DIR / "85_step7_reconciliation_audit.csv"
HASH_PATH = TABLES_DIR / "86_step7_input_hashes.csv"
MANIFEST_PATH = TABLES_DIR / "87_step7_output_manifest.csv"

RFM_FIGURE_PATH = FIGURES_DIR / "07_core_rfm_distributions.png"
SKEWNESS_FIGURE_PATH = FIGURES_DIR / "08_feature_skewness_audit.png"

EXPECTED_STEP6_COLUMNS = [
    "Customer ID",
    "observation_start_timestamp",
    "cutoff_timestamp",
    "outcome_end_timestamp",
    "last_positive_purchase_before_cutoff_timestamp",
    "baseline_recency_days",
    "eligible_observation_window",
    "primary_cohort_eligibility",
    "first_positive_purchase_in_outcome_timestamp",
    "positive_purchase_events_in_outcome",
    "Churn",
]

FEATURE_COLUMNS = [
    "Customer ID",
    "RecencyDays",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AverageOrderValue",
    "AverageQuantityPerOrder",
    "ActivePurchaseDays",
    "UniqueProducts",
    "ProductsPerOrder",
    "CustomerAgeDays",
    "PurchaseSpanDays",
    "MeanInterpurchaseDays",
    "MedianInterpurchaseDays",
    "HasRepeatPurchaseHistory",
    "CancellationInvoiceCount",
    "CancellationValue",
    "HasCancellation",
    "CancellationInvoiceRate",
    "CountryCandidate",
    "Churn",
]

NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in {"Customer ID", "CountryCandidate", "Churn"}]

FEATURE_DEFINITIONS = {
    "RecencyDays": (
        "Days since the latest positive purchase at the cutoff.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "(cutoff - latest positive invoice-event timestamp) in elapsed days; this reconciles to Step 6 baseline recency.",
    ),
    "Frequency": (
        "Number of positive-purchase invoices in the observation window.",
        "Step 4 positive-purchase base within the observation window.",
        "Unique Invoice count per customer; line items are not counted as orders.",
    ),
    "Monetary": (
        "Total positive purchase value in the observation window.",
        "Step 4 positive-purchase base within the observation window.",
        "Sum of Quantity * Price across positive-purchase lines.",
    ),
    "TotalQuantity": (
        "Total units purchased in positive purchases.",
        "Step 4 positive-purchase base within the observation window.",
        "Sum of positive-purchase Quantity.",
    ),
    "AverageOrderValue": (
        "Mean positive-purchase invoice value.",
        "Monetary and Frequency from the observation window.",
        "Monetary / Frequency.",
    ),
    "AverageQuantityPerOrder": (
        "Mean units purchased per positive invoice.",
        "TotalQuantity and Frequency from the observation window.",
        "TotalQuantity / Frequency.",
    ),
    "ActivePurchaseDays": (
        "Number of calendar dates with a positive purchase.",
        "Step 4 positive-purchase InvoiceDate within the observation window.",
        "Distinct normalized InvoiceDate per customer.",
    ),
    "UniqueProducts": (
        "Number of distinct products purchased.",
        "Step 4 positive-purchase StockCode within the observation window.",
        "Distinct non-null StockCode count per customer.",
    ),
    "ProductsPerOrder": (
        "Mean distinct-product intensity per positive invoice.",
        "Step 4 positive-purchase StockCode and Invoice within the observation window.",
        "For each invoice, count distinct non-null StockCode values; average those invoice-level counts per customer.",
    ),
    "CustomerAgeDays": (
        "Observable customer history from first observed purchase to cutoff.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "(cutoff - first positive invoice-event timestamp) in elapsed days; this is not necessarily true lifetime tenure.",
    ),
    "PurchaseSpanDays": (
        "Elapsed time between first and last positive purchases in the observation window.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "(latest positive invoice-event timestamp - earliest positive invoice-event timestamp) in elapsed days.",
    ),
    "MeanInterpurchaseDays": (
        "Mean elapsed time between consecutive positive-purchase invoice events.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "Mean adjacent timestamp gap per customer; missing when the customer has only one event.",
    ),
    "MedianInterpurchaseDays": (
        "Median elapsed time between consecutive positive-purchase invoice events.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "Median adjacent timestamp gap per customer; missing when the customer has only one event.",
    ),
    "HasRepeatPurchaseHistory": (
        "Whether at least two positive-purchase invoice events are observed.",
        "Step 5 invoice-level positive-purchase events within the observation window.",
        "1 when invoice-event count >= 2, otherwise 0.",
    ),
    "CancellationInvoiceCount": (
        "Number of distinct identified cancellation invoices.",
        "Step 4 customer-linked cancellation history within the observation window.",
        "Unique cancellation Invoice count per customer.",
    ),
    "CancellationValue": (
        "Absolute monetary value represented by identified cancellation lines.",
        "Step 4 customer-linked cancellation history within the observation window.",
        "Sum of abs(Quantity * Price) across cancellation lines; cancellations are not netted against purchases.",
    ),
    "HasCancellation": (
        "Whether at least one identified cancellation invoice is observed.",
        "Step 4 customer-linked cancellation history within the observation window.",
        "1 when CancellationInvoiceCount > 0, otherwise 0.",
    ),
    "CancellationInvoiceRate": (
        "Share of identified purchase/cancellation invoices that are cancellations.",
        "Observation-period positive purchases and customer-linked cancellations.",
        "CancellationInvoiceCount / (Frequency + CancellationInvoiceCount); denominator is valid because both terms count distinct invoices.",
    ),
    "CountryCandidate": (
        "Raw customer-level country candidate based on observation-period purchases.",
        "Step 4 positive-purchase Country within the observation window.",
        "Most frequent observed country per customer; unique top country required, with deterministic lexical ordering only as a tie audit.",
    ),
}


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.build_customer_features")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def input_paths() -> dict[str, Path]:
    return {
        "step6_outcome_table": STEP6_OUTCOME_PATH,
        "positive_purchase_base": POSITIVE_PURCHASE_PATH,
        "cancellation_history": CANCELLATION_HISTORY_PATH,
        "purchase_events": EVENT_PATH,
        "canonical_source": CANONICAL_PATH,
        "raw_workbook": find_raw_workbook(),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, pd.Timestamp]]:
    paths = input_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Step 7 inputs are missing: {missing}")

    outcome = pd.read_parquet(paths["step6_outcome_table"], engine="pyarrow")
    positive = pd.read_parquet(paths["positive_purchase_base"], engine="pyarrow")
    cancellations = pd.read_parquet(paths["cancellation_history"], engine="pyarrow")
    events = pd.read_parquet(paths["purchase_events"], engine="pyarrow")

    if list(outcome.columns) != EXPECTED_STEP6_COLUMNS:
        raise RuntimeError(f"Unexpected Step 6 schema: {list(outcome.columns)}")
    if len(outcome) != 2457 or outcome["Customer ID"].duplicated().any():
        raise RuntimeError("Step 6 input is not the locked 2,457-customer one-row-per-customer outcome table.")
    if outcome["Churn"].isna().any() or not outcome["Churn"].isin([0, 1]).all():
        raise RuntimeError("Step 6 Churn values are missing or outside {0, 1}.")
    boundaries = {
        "observation_start": pd.Timestamp("2010-07-27 12:50:00"),
        "cutoff": pd.Timestamp("2011-07-27 12:50:00"),
        "outcome_end": pd.Timestamp("2011-12-09 12:50:00"),
    }
    for name, timestamp in boundaries.items():
        column = {"observation_start": "observation_start_timestamp", "cutoff": "cutoff_timestamp", "outcome_end": "outcome_end_timestamp"}[name]
        if not outcome[column].eq(timestamp).all():
            raise RuntimeError(f"Step 6 {column} does not match the locked boundary {timestamp}.")

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return outcome, positive, cancellations, events, hashes, boundaries


def build_customer_features(
    outcome: pd.DataFrame,
    positive: pd.DataFrame,
    cancellations: pd.DataFrame,
    events: pd.DataFrame,
    boundaries: dict[str, pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build one observation-only feature row for every locked cohort customer."""

    ids = outcome["Customer ID"].tolist()
    id_set = set(ids)
    start = boundaries["observation_start"]
    cutoff = boundaries["cutoff"]

    positive_obs = positive.loc[
        positive["Customer ID"].isin(id_set)
        & positive["InvoiceDate"].between(start, cutoff, inclusive="both")
    ].copy()
    positive_obs["line_value"] = positive_obs["Quantity"] * positive_obs["Price"]

    if positive_obs.empty or positive_obs["Customer ID"].nunique() != len(outcome):
        raise RuntimeError("Observation-period positive purchases do not cover every Step 6 customer.")

    invoice_stats = positive_obs.groupby(["Customer ID", "Invoice"], sort=False).agg(
        invoice_value=("line_value", "sum"),
        invoice_quantity=("Quantity", "sum"),
        invoice_product_count=("StockCode", "nunique"),
        invoice_date=("InvoiceDate", "min"),
    )
    customer_stats = positive_obs.groupby("Customer ID", sort=False).agg(
        first_positive_purchase=("InvoiceDate", "min"),
        last_positive_purchase=("InvoiceDate", "max"),
        monetary=("line_value", "sum"),
        total_quantity=("Quantity", "sum"),
        active_purchase_days=("InvoiceDate", lambda values: values.dt.normalize().nunique()),
        unique_products=("StockCode", "nunique"),
    )
    customer_stats["frequency"] = positive_obs.groupby("Customer ID")["Invoice"].nunique()
    customer_stats["average_order_value"] = customer_stats["monetary"] / customer_stats["frequency"]
    customer_stats["average_quantity_per_order"] = customer_stats["total_quantity"] / customer_stats["frequency"]
    customer_stats["products_per_order"] = invoice_stats.groupby(level=0)["invoice_product_count"].mean()
    customer_stats["recency_days"] = (cutoff - customer_stats["last_positive_purchase"]).dt.total_seconds() / 86400
    customer_stats["customer_age_days"] = (cutoff - customer_stats["first_positive_purchase"]).dt.total_seconds() / 86400
    customer_stats["purchase_span_days"] = (
        customer_stats["last_positive_purchase"] - customer_stats["first_positive_purchase"]
    ).dt.total_seconds() / 86400

    event_obs = events.loc[
        events["Customer ID"].isin(id_set)
        & events["purchase_timestamp"].between(start, cutoff, inclusive="both")
    ].copy()
    event_obs = event_obs.sort_values(["Customer ID", "purchase_timestamp", "Invoice"])
    event_counts = event_obs.groupby("Customer ID").size().rename("event_count")
    event_time_stats = event_obs.groupby("Customer ID")["purchase_timestamp"].agg(
        first_event="min", last_event="max"
    )
    # Step 6 baseline recency is defined from the validated invoice-event
    # timestamp.  Use the same event-level time basis for all elapsed-time
    # features so temporal features reconcile exactly to Step 6.
    customer_stats["first_positive_purchase"] = event_time_stats["first_event"]
    customer_stats["last_positive_purchase"] = event_time_stats["last_event"]
    customer_stats["recency_days"] = (cutoff - customer_stats["last_positive_purchase"]).dt.total_seconds() / 86400
    customer_stats["customer_age_days"] = (cutoff - customer_stats["first_positive_purchase"]).dt.total_seconds() / 86400
    customer_stats["purchase_span_days"] = (
        customer_stats["last_positive_purchase"] - customer_stats["first_positive_purchase"]
    ).dt.total_seconds() / 86400
    event_obs["interpurchase_days"] = (
        event_obs.groupby("Customer ID")["purchase_timestamp"].diff().dt.total_seconds() / 86400
    )
    interpurchase = event_obs.dropna(subset=["interpurchase_days"]).groupby("Customer ID")["interpurchase_days"].agg(
        mean_interpurchase_days="mean", median_interpurchase_days="median"
    )

    cancellation_obs = cancellations.loc[
        cancellations["Customer ID"].isin(id_set)
        & cancellations["InvoiceDate"].between(start, cutoff, inclusive="both")
    ].copy()
    cancellation_obs["absolute_line_value"] = (
        cancellation_obs["Quantity"] * cancellation_obs["Price"]
    ).abs()
    cancellation_stats = cancellation_obs.groupby("Customer ID").agg(
        cancellation_invoice_count=("Invoice", "nunique"),
        cancellation_value=("absolute_line_value", "sum"),
    )

    country_counts = (
        positive_obs.dropna(subset=["Country"])
        .groupby(["Customer ID", "Country"])
        .size()
        .rename("country_purchase_rows")
        .reset_index()
    )
    country_counts = country_counts.sort_values(["Customer ID", "country_purchase_rows", "Country"], ascending=[True, False, True])
    country_top = country_counts.drop_duplicates("Customer ID", keep="first").set_index("Customer ID")
    top_ties = country_counts.groupby("Customer ID")["country_purchase_rows"].transform("max").eq(country_counts["country_purchase_rows"])
    tie_counts = country_counts.loc[top_ties].groupby("Customer ID").size()
    country_top["CountryCandidate"] = country_top["Country"].where(tie_counts.reindex(country_top.index).fillna(0).eq(1), pd.NA)
    country_unique_counts = positive_obs.groupby("Customer ID")["Country"].nunique(dropna=True).rename("observed_country_count")

    features = pd.DataFrame({"Customer ID": ids}).set_index("Customer ID")
    features["RecencyDays"] = customer_stats["recency_days"]
    features["Frequency"] = customer_stats["frequency"]
    features["Monetary"] = customer_stats["monetary"]
    features["TotalQuantity"] = customer_stats["total_quantity"]
    features["AverageOrderValue"] = customer_stats["average_order_value"]
    features["AverageQuantityPerOrder"] = customer_stats["average_quantity_per_order"]
    features["ActivePurchaseDays"] = customer_stats["active_purchase_days"]
    features["UniqueProducts"] = customer_stats["unique_products"]
    features["ProductsPerOrder"] = customer_stats["products_per_order"]
    features["CustomerAgeDays"] = customer_stats["customer_age_days"]
    features["PurchaseSpanDays"] = customer_stats["purchase_span_days"]
    features["MeanInterpurchaseDays"] = interpurchase["mean_interpurchase_days"]
    features["MedianInterpurchaseDays"] = interpurchase["median_interpurchase_days"]
    features["HasRepeatPurchaseHistory"] = event_counts.ge(2).astype("int64")
    features["CancellationInvoiceCount"] = cancellation_stats["cancellation_invoice_count"].reindex(features.index).fillna(0).astype("int64")
    features["CancellationValue"] = cancellation_stats["cancellation_value"].reindex(features.index).fillna(0.0)
    features["HasCancellation"] = features["CancellationInvoiceCount"].gt(0).astype("int64")
    features["CancellationInvoiceRate"] = features["CancellationInvoiceCount"] / (
        features["Frequency"] + features["CancellationInvoiceCount"]
    )
    features["CountryCandidate"] = country_top["CountryCandidate"]
    features["Churn"] = outcome.set_index("Customer ID")["Churn"]
    features = features.reset_index()[FEATURE_COLUMNS]

    diagnostics = {
        "positive_observation": positive_obs,
        "event_observation": event_obs,
        "cancellation_observation": cancellation_obs,
        "country_unique_counts": country_unique_counts,
        "country_top": country_top,
        "event_counts": event_counts,
    }
    return features, diagnostics


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
    return str(value)


def build_quality_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nonnegative = set(NUMERIC_FEATURES) - {"CancellationInvoiceRate"}
    binary = {"HasRepeatPurchaseHistory", "HasCancellation"}
    for name in FEATURE_COLUMNS:
        if name in {"Customer ID", "Churn"}:
            continue
        series = features[name]
        definition = FEATURE_DEFINITIONS[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        numeric_values = pd.to_numeric(series, errors="coerce") if numeric else pd.Series(dtype=float)
        missing = int(series.isna().sum())
        row = {
            "feature_name": name,
            "business_meaning": definition[0],
            "source_data": definition[1],
            "calculation_rule": definition[2],
            "data_type": str(series.dtype),
            "missing_count": missing,
            "missing_pct": missing / len(features) * 100,
            "zero_count": int(numeric_values.eq(0).sum()) if numeric else np.nan,
            "zero_pct": float(numeric_values.eq(0).mean() * 100) if numeric else np.nan,
            "min": numeric_values.min() if numeric else np.nan,
            "p01": numeric_values.quantile(0.01) if numeric else np.nan,
            "p05": numeric_values.quantile(0.05) if numeric else np.nan,
            "median": numeric_values.median() if numeric else np.nan,
            "mean": numeric_values.mean() if numeric else np.nan,
            "p75": numeric_values.quantile(0.75) if numeric else np.nan,
            "p95": numeric_values.quantile(0.95) if numeric else np.nan,
            "p99": numeric_values.quantile(0.99) if numeric else np.nan,
            "max": numeric_values.max() if numeric else np.nan,
            "n_unique": int(series.nunique(dropna=True)),
            "skewness": numeric_values.skew() if numeric and numeric_values.notna().sum() > 2 else np.nan,
            "zero_inflated": bool(numeric and numeric_values.eq(0).mean() >= 0.20),
            "strongly_skewed": bool(numeric and abs(numeric_values.skew()) >= 2),
            "structurally_missing": bool(name in {"MeanInterpurchaseDays", "MedianInterpurchaseDays"}),
            "structural_missing_reason": "Only one positive-purchase event; no adjacent interpurchase interval exists." if name in {"MeanInterpurchaseDays", "MedianInterpurchaseDays"} else "",
            "impossible_value_count": int((numeric_values.lt(0) & numeric_values.notna()).sum()) if name in nonnegative else 0,
            "extreme_value_note": "Maximum exceeds P99 by more than 3x; inspect as a long-tail/outlier candidate." if numeric and numeric_values.notna().any() and numeric_values.max() > 3 * numeric_values.quantile(0.99) else "",
        }
        if name in binary:
            row["impossible_value_count"] = int((~series.dropna().isin([0, 1])).sum())
        if name == "CancellationInvoiceRate":
            row["impossible_value_count"] = int(((series < 0) | (series > 1)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_country_audit(features: pd.DataFrame, diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    counts = diagnostics["country_unique_counts"].reindex(features["Customer ID"]).fillna(0)
    country_top = diagnostics["country_top"].copy()
    candidate_counts = features["CountryCandidate"].value_counts(dropna=False)
    total = len(features)
    rows = [
        {"section": "summary", "metric": "primary_cohort_customers", "country": "", "customers": total, "percentage": 100.0, "notes": "Locked Step 6 cohort."},
        {"section": "summary", "metric": "customers_with_exactly_one_observed_country", "country": "", "customers": int(counts.eq(1).sum()), "percentage": float(counts.eq(1).mean() * 100), "notes": "Country associations counted from observation-period positive purchases."},
        {"section": "summary", "metric": "customers_with_more_than_one_observed_country", "country": "", "customers": int(counts.gt(1).sum()), "percentage": float(counts.gt(1).mean() * 100), "notes": "These customers have multiple observed country values."},
        {"section": "summary", "metric": "customers_with_no_observed_country", "country": "", "customers": int(counts.eq(0).sum()), "percentage": float(counts.eq(0).mean() * 100), "notes": "No non-null Country in observation-period positive purchases."},
        {"section": "summary", "metric": "customers_with_unique_dominant_country_candidate", "country": "", "customers": int(features["CountryCandidate"].notna().sum()), "percentage": float(features["CountryCandidate"].notna().mean() * 100), "notes": "Raw candidate included because the observed dominant country is unique for each customer in this cohort."},
    ]
    ranked = candidate_counts.drop(labels=[x for x in candidate_counts.index if pd.isna(x)], errors="ignore").sort_values(ascending=False)
    for rank, (country, count) in enumerate(ranked.items(), start=1):
        rows.append({
            "section": "dominant_country_frequency",
            "metric": "dominant_country",
            "country": country,
            "customers": int(count),
            "percentage": float(count / total * 100),
            "notes": f"Rank {rank}; CountryCandidate is the unique modal observed country.",
        })
    if len(ranked):
        rows.append({
            "section": "summary",
            "metric": "largest_country_category",
            "country": ranked.index[0],
            "customers": int(ranked.iloc[0]),
            "percentage": float(ranked.iloc[0] / total * 100),
            "notes": "Concentration of the largest raw country candidate category.",
        })
    return pd.DataFrame(rows)


def build_correlation_audit(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson")
    rows = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            rows.append({
                "feature_a": left,
                "feature_b": right,
                "pearson_correlation": float(value),
                "absolute_correlation": float(abs(value)),
                "high_correlation_abs_ge_0_90": bool(abs(value) >= 0.90),
                "interpretation": "Derived/component redundancy is plausible; retain for audit and defer feature selection." if abs(value) >= 0.90 else "Descriptive pairwise association only.",
            })
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def build_reconciliation_audit(
    features: pd.DataFrame,
    outcome: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    boundaries: dict[str, pd.Timestamp],
    hashes_before: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, bool]]:
    positive_obs = diagnostics["positive_observation"]
    event_obs = diagnostics["event_observation"]
    cancellation_obs = diagnostics["cancellation_observation"]
    expected_churn = outcome.set_index("Customer ID")["Churn"].sort_index()
    observed_churn = features.set_index("Customer ID")["Churn"].sort_index()
    recency = features.set_index("Customer ID")["RecencyDays"]
    baseline = outcome.set_index("Customer ID")["baseline_recency_days"]

    forbidden_outcome_columns = {
        "observation_start_timestamp", "cutoff_timestamp", "outcome_end_timestamp",
        "last_positive_purchase_before_cutoff_timestamp", "baseline_recency_days",
        "eligible_observation_window", "primary_cohort_eligibility",
        "first_positive_purchase_in_outcome_timestamp", "positive_purchase_events_in_outcome",
    }
    checks: dict[str, tuple[bool, object, object, str]] = {
        "final_rows_exactly_2457": (len(features) == 2457, len(features), 2457, "One output row is required for each locked Step 6 customer."),
        "Customer_ID_unique": (features["Customer ID"].is_unique, int(features["Customer ID"].nunique()), 2457, "Customer ID is preserved only as the row key."),
        "Churn_exactly_matches_Step6": (expected_churn.equals(observed_churn), "equal" if expected_churn.equals(observed_churn) else "different", "equal", "Churn is copied from Step 6 and never recalculated."),
        "positive_observation_has_no_post_cutoff_rows": (bool(positive_obs["InvoiceDate"].le(boundaries["cutoff"]).all()), str(positive_obs["InvoiceDate"].max()), str(boundaries["cutoff"]), "Positive feature rows are restricted to the closed observation interval."),
        "positive_observation_respects_observation_start": (bool(positive_obs["InvoiceDate"].ge(boundaries["observation_start"]).all()), str(positive_obs["InvoiceDate"].min()), str(boundaries["observation_start"]), "Positive feature rows are restricted to the closed observation interval."),
        "event_observation_has_no_post_cutoff_rows": (bool(event_obs["purchase_timestamp"].le(boundaries["cutoff"]).all()), str(event_obs["purchase_timestamp"].max()), str(boundaries["cutoff"]), "Interpurchase features use only Step 5 events through cutoff."),
        "cancellation_observation_has_no_post_cutoff_rows": (bool(cancellation_obs.empty or cancellation_obs["InvoiceDate"].le(boundaries["cutoff"]).all()), str(cancellation_obs["InvoiceDate"].max()) if not cancellation_obs.empty else "none", str(boundaries["cutoff"]), "Cancellation features use only customer-linked observation-period cancellations."),
        "RecencyDays_reconciles_with_Step6": (bool(np.allclose(recency.sort_index(), baseline.sort_index(), rtol=0, atol=1e-9)), float((recency - baseline).abs().max()), 0.0, "Latest positive purchase and elapsed-day calculation reconcile to Step 6 baseline recency."),
        "RecencyDays_at_most_135": (bool(recency.le(135).all()), float(recency.max()), 135, "Primary Population B eligibility requires baseline recency <=135 days."),
        "Frequency_at_least_1": (bool(features["Frequency"].ge(1).all()), int(features["Frequency"].min()), 1, "Every primary-cohort customer has observation-period positive purchase activity."),
        "Monetary_positive": (bool(features["Monetary"].gt(0).all()), float(features["Monetary"].min()), "> 0", "Positive quantities and prices make every purchase value positive."),
        "TotalQuantity_positive": (bool(features["TotalQuantity"].gt(0).all()), int(features["TotalQuantity"].min()), "> 0", "Positive-purchase base contains Quantity > 0."),
        "AverageOrderValue_positive": (bool(features["AverageOrderValue"].gt(0).all()), float(features["AverageOrderValue"].min()), "> 0", "Monetary and Frequency are positive."),
        "cancellation_features_observation_only": (bool(cancellation_obs["Customer ID"].isin(features["Customer ID"]).all()), int(cancellation_obs["Customer ID"].nunique()), int(features["HasCancellation"].sum()), "Cancellation source is customer-linked and limited to the cohort; expected count is customers with observed cancellations."),
        "no_future_outcome_columns_in_feature_table": (not bool(forbidden_outcome_columns.intersection(features.columns)), sorted(forbidden_outcome_columns.intersection(features.columns)), [], "Only Churn is carried from the Step 6 outcome table."),
    }
    after_hashes = {name: sha256_file(path) for name, path in input_paths().items()}
    for name, before in hashes_before.items():
        checks[f"{name}_hash_unchanged"] = (before == after_hashes[name], after_hashes[name], before, "Source/processed input hash remained unchanged during Step 7.")

    rows = [
        {"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "evidence": evidence}
        for name, (passed, observed, expected, evidence) in checks.items()
    ]
    audit = pd.DataFrame(rows)
    return audit, {name: bool(result[0]) for name, result in checks.items()}


def write_figures(features: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, column, title in zip(
        axes.flat,
        ["RecencyDays", "Frequency", "Monetary", "TotalQuantity"],
        ["Recency", "Frequency", "Monetary value", "Total quantity"],
    ):
        ax.hist(features[column].dropna(), bins=35, color="#2f6f8f", edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Customers")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Step 7: observation-period core RFM feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(RFM_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    quality = build_quality_audit(features).sort_values("skewness", key=lambda s: s.abs(), ascending=False)
    numeric_quality = quality.loc[quality["skewness"].notna()].head(12).sort_values("skewness")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#b55d4c" if value >= 2 else "#5b8e7d" for value in numeric_quality["skewness"]]
    ax.barh(numeric_quality["feature_name"], numeric_quality["skewness"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axvline(2, color="#b55d4c", linestyle="--", linewidth=0.8, label="strong skewness threshold")
    ax.axvline(-2, color="#b55d4c", linestyle="--", linewidth=0.8)
    ax.set_title("Step 7: largest absolute feature skewness values")
    ax.set_xlabel("Pearson moment skewness")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SKEWNESS_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [RFM_FIGURE_PATH, SKEWNESS_FIGURE_PATH]


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    if limit is not None:
        selected = selected.head(limit)
    def cell(value: object) -> str:
        if isinstance(value, (list, tuple, set, dict)):
            return str(value).replace("|", "\\|").replace("\n", " ")
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for values in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join(lines)


def write_notebook() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Step 7 - Build and Audit Customer-Level Predictive Features\n",
                "\n",
                "This lightweight notebook inspects the reproducible Step 7 outputs. It uses the locked Step 6 cohort and observation-only features; it does not split, encode, scale, rebalance, train models, or run SHAP."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "sys.path.insert(0, str(ROOT / 'src'))\n",
                "from churn_analysis.build_customer_features import FEATURE_PATH\n",
                "features = pd.read_parquet(FEATURE_PATH)\n",
                "TABLES = ROOT / 'outputs' / 'tables'\n",
                "quality = pd.read_csv(TABLES / '82_step7_feature_quality_audit.csv')\n",
                "country = pd.read_csv(TABLES / '83_step7_country_audit.csv')\n",
                "correlations = pd.read_csv(TABLES / '84_step7_correlation_audit.csv')\n",
                "reconciliation = pd.read_csv(TABLES / '85_step7_reconciliation_audit.csv')"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Feature table and quality audit"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(features.head())\n",
                "display(features.describe(include='all').T)\n",
                "display(quality)"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Country, correlation, and reconciliation audits"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(country)\n",
                "display(correlations.head(20))\n",
                "display(reconciliation)"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": [
                "Figures: `outputs/figures/07_core_rfm_distributions.png` and `outputs/figures/08_feature_skewness_audit.png`. The processed parquet is intentionally Git-ignored."
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def write_report(
    features: pd.DataFrame,
    quality: pd.DataFrame,
    country: pd.DataFrame,
    correlations: pd.DataFrame,
    reconciliation: pd.DataFrame,
    hashes: pd.DataFrame,
    metadata: dict[str, str],
) -> None:
    core = quality.loc[quality["feature_name"].isin(["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "ActivePurchaseDays", "UniqueProducts"])].copy()
    missing = quality.loc[quality["missing_count"].gt(0)].copy()
    high = correlations.loc[correlations["high_correlation_abs_ge_0_90"]].head(15)
    country_summary = country.loc[country["section"].eq("summary")]
    largest = country.loc[country["metric"].eq("largest_country_category")]
    passed = int(reconciliation["passed"].sum())
    report = f"""# Step 7 - Build and Audit Customer-Level Predictive Features

## Scope and locked temporal design

Step 7 uses exactly the Step 6 primary cohort of **{len(features):,} customers**. The observation interval is **[2010-07-27 12:50:00, 2011-07-27 12:50:00]**. Features use only positive purchases, invoice-level purchase events, and customer-linked cancellations inside that interval. The Step 6 `Churn` value is copied unchanged; no future outcome field is used as a predictor.

No train/test split, categorical encoding, scaling, class rebalancing, SMOTE, model training, or SHAP analysis was performed.

## 1. Feature table

The ignored processed output is `data/processed/07_customer_features.parquet`, with exactly one row per Step 6 customer. `Customer ID` is retained only as the row key. `CountryCandidate` is a raw categorical audit candidate and is not encoded.

| feature | definition |
| --- | --- |
| RecencyDays | Days from latest positive purchase to cutoff. |
| Frequency | Unique positive-purchase invoice count. |
| Monetary | Sum of positive-purchase `Quantity * Price`. |
| TotalQuantity | Sum of positive-purchase units. |
| AverageOrderValue | Monetary divided by Frequency. |
| AverageQuantityPerOrder | TotalQuantity divided by Frequency. |
| ActivePurchaseDays | Distinct calendar dates with a positive purchase. |
| UniqueProducts | Distinct purchased `StockCode` count. |
| ProductsPerOrder | Mean distinct `StockCode` count per positive invoice. |
| CustomerAgeDays | First observed positive purchase to cutoff; observable history in this window, not necessarily true lifetime tenure. |
| PurchaseSpanDays | First-to-last positive purchase elapsed days in the observation window. |
| MeanInterpurchaseDays | Mean adjacent positive invoice-event gap; missing for one-event customers. |
| MedianInterpurchaseDays | Median adjacent positive invoice-event gap; missing for one-event customers. |
| HasRepeatPurchaseHistory | 1 when at least two positive invoice events are observed. |
| CancellationInvoiceCount | Unique identified cancellation invoice count. |
| CancellationValue | Sum of absolute cancellation line values. |
| HasCancellation | 1 when at least one cancellation invoice is observed. |
| CancellationInvoiceRate | Cancellation invoices / (positive invoices + cancellation invoices). |
| CountryCandidate | Unique modal observed country, retained raw for audit only. |
| Churn | Step 6 label copied without recalculation. |

## 2. Core feature statistics

{markdown_table(core, ['feature_name', 'data_type', 'missing_count', 'min', 'median', 'mean', 'max', 'n_unique'])}

The complete business meaning, source, calculation rule, percentiles, skewness, zero inflation, and impossible-value audit is in `82_step7_feature_quality_audit.csv`.

## 3. Missing and structural-missing audit

{markdown_table(missing, ['feature_name', 'missing_count', 'missing_pct', 'structurally_missing', 'structural_missing_reason']) if not missing.empty else 'No missing engineered features.'}

`MeanInterpurchaseDays` and `MedianInterpurchaseDays` are structurally missing for **{int(features['HasRepeatPurchaseHistory'].eq(0).sum())} customers** with only one positive purchase event. They were retained as missing; no zero, mean, or median imputation was performed. Customers with repeat history: **{int(features['HasRepeatPurchaseHistory'].sum()):,}**.

## 4. Cancellation-feature results

Observation-period cancellation features are zero for customers without an identified customer-linked C-prefix invoice. The cohort has **{int(features['HasCancellation'].sum()):,} customers with cancellation activity** ({features['HasCancellation'].mean() * 100:.2f}%). Total cancellation invoices are **{int(features['CancellationInvoiceCount'].sum()):,}**, total absolute cancellation value is **{features['CancellationValue'].sum():,.2f}**, and the customer-level cancellation rate has median **{features['CancellationInvoiceRate'].median():.4f}** and maximum **{features['CancellationInvoiceRate'].max():.4f}**. These values use only cancellations through the cutoff and are not netted against purchases.

## 5. Country audit

{markdown_table(country_summary, ['metric', 'country', 'customers', 'percentage', 'notes'])}

The largest country candidate category is **{largest.iloc[0]['country']}** with **{int(largest.iloc[0]['customers']):,} customers ({largest.iloc[0]['percentage']:.2f}%)**. The distribution is therefore highly concentrated, but Country remains a raw candidate for later modeling audit because it is observed and unencoded here; its eventual modeling value must be assessed empirically rather than decided from intuition alone. The full dominant-country distribution is in `83_step7_country_audit.csv`.

## 6. Distribution and outlier findings

The core RFM distributions are in `outputs/figures/07_core_rfm_distributions.png`; the compact skewness view is in `outputs/figures/08_feature_skewness_audit.png`. The quality table marks zero-inflated features using a zero share of at least 20% and strongly skewed numeric features using absolute skewness at least 2. Monetary, TotalQuantity, Frequency, and related order-value measures are expected to be right-skewed, with large maxima relative to their medians. Cancellation indicators/value are zero-inflated. Extreme values are retained for audit; no outliers were automatically removed and no transformations were applied.

## 7. Numeric correlation audit

The correlation audit excludes `Customer ID`, `CountryCandidate`, and `Churn`; it is descriptive only and is not a feature-selection rule.

{markdown_table(high, ['feature_a', 'feature_b', 'pearson_correlation', 'interpretation']) if not high.empty else 'No absolute correlations at or above 0.90.'}

High associations are expected where one feature is mathematically derived from another: Monetary and AverageOrderValue relate through Frequency; TotalQuantity and AverageQuantityPerOrder relate through Frequency; ProductsPerOrder relates to UniqueProducts and Frequency; and the purchase-history time measures share first/last purchase timestamps. No feature was removed automatically.

## 8. Reconciliation and temporal-leakage validation

**{passed}/{len(reconciliation)} reconciliation checks passed.**

{markdown_table(reconciliation, ['check', 'passed', 'observed', 'expected'])}

The feature table contains no post-cutoff transaction-derived fields and no Step 6 outcome-window validation columns. The only outcome value carried forward is `Churn`. Input hashes before and after execution are recorded in `86_step7_input_hashes.csv`; the canonical source, raw workbook, and committed Step 4-6 inputs remained unchanged.

## 9. Outputs and stopping point

- Source module: `src/churn_analysis/build_customer_features.py`
- Lightweight notebook: `notebooks/07_customer_feature_engineering_and_audit.ipynb`
- Report: `reports/07_customer_feature_engineering_and_audit.md`
- Audit tables: `outputs/tables/82_step7_feature_quality_audit.csv` through `87_step7_output_manifest.csv`
- Figures: `outputs/figures/07_core_rfm_distributions.png` and `08_feature_skewness_audit.png`
- Reproducibility log: `outputs/logs/07_customer_feature_engineering_and_audit.log`
- Ignored processed feature table: `data/processed/07_customer_features.parquet`

Step 7 stops after customer-level feature construction, quality auditing, descriptive country review, correlation auditing, and reconciliation. No split, encoding, scaling, resampling, model training, or SHAP analysis occurred.

Input hashes at start:

{markdown_table(hashes, ['input', 'path', 'sha256_before', 'sha256_after', 'unchanged'])}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_hash_table(hashes_before: dict[str, str]) -> pd.DataFrame:
    rows = []
    after = {name: sha256_file(path) for name, path in input_paths().items()}
    for name, path in input_paths().items():
        rows.append({
            "input": name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256_before": hashes_before[name],
            "sha256_after": after[name],
            "unchanged": hashes_before[name] == after[name],
        })
    table = pd.DataFrame(rows)
    table.to_csv(HASH_PATH, index=False)
    return table


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    table = pd.DataFrame(rows)
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, object]:
    logger = get_logger()
    outcome, positive, cancellations, events, hashes_before, boundaries = load_inputs()
    features, diagnostics = build_customer_features(outcome, positive, cancellations, events, boundaries)
    quality = build_quality_audit(features)
    country = build_country_audit(features, diagnostics)
    correlations = build_correlation_audit(features)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_PATH, engine="pyarrow", index=False)
    quality.to_csv(QUALITY_PATH, index=False)
    country.to_csv(COUNTRY_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)
    figures = write_figures(features)
    write_notebook()
    hashes = write_hash_table(hashes_before)
    reconciliation, checks = build_reconciliation_audit(features, outcome, diagnostics, boundaries, hashes_before)
    reconciliation.to_csv(RECONCILIATION_PATH, index=False)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Step 7 reconciliation failed: {failed}")

    write_report(features, quality, country, correlations, reconciliation, hashes, hashes_before)
    logger.info("Step 7 built the observation-only customer feature table.")
    logger.info("Customers: %s; features excluding key and Churn: %s", len(features), len(FEATURE_COLUMNS) - 2)
    logger.info("Repeat-history customers: %s; structural interpurchase missing customers: %s", int(features["HasRepeatPurchaseHistory"].sum()), int(features["HasRepeatPurchaseHistory"].eq(0).sum()))
    logger.info("Cancellation customers: %s; high absolute correlations: %s", int(features["HasCancellation"].sum()), int(correlations["high_correlation_abs_ge_0_90"].sum()))
    logger.info("Validation checks: %s", checks)
    for handler in logger.handlers:
        handler.flush()

    declared = [FEATURE_PATH, QUALITY_PATH, COUNTRY_PATH, CORRELATION_PATH, RECONCILIATION_PATH, HASH_PATH, *figures, REPORT_PATH, NOTEBOOK_PATH, LOG_PATH]
    manifest = write_manifest(declared)
    return {
        "features": features,
        "quality": quality,
        "country": country,
        "correlations": correlations,
        "reconciliation": reconciliation,
        "hashes": hashes,
        "checks": checks,
        "manifest": manifest,
    }


def main() -> None:
    result = run_analysis()
    features = result["features"]
    print(json.dumps({
        "customers": len(features),
        "feature_columns": FEATURE_COLUMNS,
        "churn_1": int(features["Churn"].eq(1).sum()),
        "churn_0": int(features["Churn"].eq(0).sum()),
        "repeat_history_customers": int(features["HasRepeatPurchaseHistory"].sum()),
        "structural_interpurchase_missing_customers": int(features["HasRepeatPurchaseHistory"].eq(0).sum()),
        "cancellation_customers": int(features["HasCancellation"].sum()),
        "validation": result["checks"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
