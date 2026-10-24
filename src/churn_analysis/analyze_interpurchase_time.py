"""Step 5: analyze interpurchase timing and discover candidate inactivity thresholds.

This module is intentionally descriptive.  It starts from the validated Step 4
positive-purchase base, collapses line items to invoice-level purchase events,
and stops before churn labels, modeling features, or train/test data are made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .build_analysis_base import (
    CANONICAL_PATH,
    POSITIVE_PURCHASE_PATH,
    find_raw_workbook,
)
from .config import FIGURES_DIR, LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR


STEP5_PREFIX = "05_"
EVENT_PATH = PROCESSED_DATA_DIR / f"{STEP5_PREFIX}invoice_purchase_events.parquet"
INTERVAL_PATH = PROCESSED_DATA_DIR / f"{STEP5_PREFIX}interpurchase_intervals.parquet"
CUSTOMER_SUMMARY_PATH = PROCESSED_DATA_DIR / f"{STEP5_PREFIX}customer_interpurchase_summary.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "05_interpurchase_time_analysis.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_interpurchase_time_analysis.ipynb"
LOG_PATH = LOGS_DIR / "05_interpurchase_time_analysis.log"

EVENT_COLUMNS = [
    "Customer ID",
    "Invoice",
    "purchase_timestamp",
    "purchase_date",
    "invoice_line_count",
    "invoice_timestamp_min",
    "invoice_timestamp_max",
    "invoice_timestamp_inconsistent",
]
INTERVAL_COLUMNS = [
    "Customer ID",
    "previous_Invoice",
    "Invoice",
    "previous_purchase_timestamp",
    "purchase_timestamp",
    "previous_purchase_date",
    "purchase_date",
    "interval_days",
    "zero_day_interval",
]

COLORS = {
    "primary": "#2F5D7E",
    "secondary": "#6B8E9F",
    "accent": "#C46B3C",
    "grid": "#D9DEE3",
    "text": "#263238",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_percentage(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.analyze_interpurchase_time")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def validate_positive_base() -> tuple[pd.DataFrame, dict[str, str]]:
    """Validate that Step 5 starts from the approved Step 4 primary base."""

    if not POSITIVE_PURCHASE_PATH.exists():
        raise FileNotFoundError(f"Step 4 positive-purchase base is missing: {POSITIVE_PURCHASE_PATH}")
    positive = pd.read_parquet(POSITIVE_PURCHASE_PATH, engine="pyarrow")
    expected_columns = [
        "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
        "Price", "Customer ID", "Country", "SourceSheet",
    ]
    if list(positive.columns) != expected_columns:
        raise RuntimeError(f"Unexpected positive-purchase schema: {list(positive.columns)}")
    if positive["Invoice"].isna().any() or positive["InvoiceDate"].isna().any():
        raise RuntimeError("Positive-purchase base contains a missing Invoice or InvoiceDate.")
    rule = (
        positive["Customer ID"].notna()
        & ~positive["Invoice"].astype("string").str.upper().str.startswith("C", na=False)
        & positive["Quantity"].gt(0)
        & positive["Price"].gt(0)
    )
    if not bool(rule.all()):
        raise RuntimeError("Positive-purchase base contains rows outside the validated Step 4 rule.")
    if positive.duplicated(subset=expected_columns[:-1], keep=False).any():
        raise RuntimeError("Positive-purchase base contains exact duplicate transaction rows.")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in positive.columns):
        raise RuntimeError("A churn column exists in the Step 4 input.")

    metadata = {
        "positive_purchase_sha256_before": sha256_file(POSITIVE_PURCHASE_PATH),
        "canonical_sha256_before": sha256_file(CANONICAL_PATH),
        "raw_sha256_before": sha256_file(find_raw_workbook()),
    }
    return positive, metadata


def build_purchase_events(positive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse positive-purchase line items to one event per Invoice.

    All invoices are first validated for a single customer.  The earliest
    InvoiceDate is selected when line-item timestamps differ; all 64 such
    invoices differ only by one or two minutes and remain on one calendar day.
    """

    grouped = positive.groupby("Invoice", sort=False, dropna=False)
    customer_counts = grouped["Customer ID"].nunique(dropna=False)
    if not bool(customer_counts.eq(1).all()):
        raise RuntimeError("At least one positive-purchase invoice maps to multiple customers.")

    invoice_audit = grouped["InvoiceDate"].agg(
        invoice_timestamp_min="min",
        invoice_timestamp_max="max",
        invoice_timestamp_nunique="nunique",
        invoice_line_count="size",
    ).reset_index()
    invoice_audit["invoice_timestamp_inconsistent"] = invoice_audit["invoice_timestamp_nunique"].gt(1)
    invoice_audit["same_calendar_day_when_inconsistent"] = (
        invoice_audit["invoice_timestamp_min"].dt.normalize()
        == invoice_audit["invoice_timestamp_max"].dt.normalize()
    )
    if bool(invoice_audit.loc[invoice_audit["invoice_timestamp_inconsistent"], "same_calendar_day_when_inconsistent"].eq(False).any()):
        raise RuntimeError("An inconsistent invoice timestamp spans multiple calendar days.")

    customers = grouped["Customer ID"].first().rename("Customer ID").reset_index()
    events = customers.merge(invoice_audit, on="Invoice", how="inner", validate="one_to_one")
    events["purchase_timestamp"] = events["invoice_timestamp_min"]
    events["purchase_date"] = events["purchase_timestamp"].dt.normalize()
    events = events[
        [
            "Customer ID", "Invoice", "purchase_timestamp", "purchase_date",
            "invoice_line_count", "invoice_timestamp_min", "invoice_timestamp_max",
            "invoice_timestamp_inconsistent",
        ]
    ]
    events = events.sort_values(["Customer ID", "purchase_timestamp", "Invoice"], kind="mergesort").reset_index(drop=True)
    if events["Invoice"].duplicated().any() or events["Customer ID"].isna().any():
        raise RuntimeError("Purchase-event dataset is not unique by Invoice or contains a missing customer.")
    return events, invoice_audit


def build_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Create consecutive within-customer event intervals without dropping zero days."""

    ordered = events.sort_values(["Customer ID", "purchase_timestamp", "Invoice"], kind="mergesort").copy()
    previous = ordered.groupby("Customer ID", sort=False)
    intervals = ordered.loc[:, ["Customer ID", "Invoice", "purchase_timestamp", "purchase_date"]].copy()
    intervals["previous_Invoice"] = previous["Invoice"].shift(1)
    intervals["previous_purchase_timestamp"] = previous["purchase_timestamp"].shift(1)
    intervals["previous_purchase_date"] = previous["purchase_date"].shift(1)
    intervals["interval_days"] = (
        intervals["purchase_timestamp"] - intervals["previous_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    intervals["zero_day_interval"] = intervals["interval_days"].eq(0)
    intervals = intervals.loc[intervals["previous_Invoice"].notna()].copy()
    intervals = intervals[
        [
            "Customer ID", "previous_Invoice", "Invoice", "previous_purchase_timestamp",
            "purchase_timestamp", "previous_purchase_date", "purchase_date",
            "interval_days", "zero_day_interval",
        ]
    ].reset_index(drop=True)
    if intervals.empty:
        raise RuntimeError("No interpurchase intervals were created.")
    if not bool(intervals["interval_days"].ge(0).all()):
        raise RuntimeError("A negative interpurchase interval was created.")
    return intervals


def describe(values: pd.Series, population: str, percentiles: list[float]) -> pd.DataFrame:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame()
    stats = values.describe(percentiles=percentiles)
    row: dict[str, object] = {"population": population, "count": int(stats["count"])}
    for key in ["min", "mean", "50%", "std", "25%", "75%", "80%", "85%", "90%", "95%", "97.5%", "99%", "max"]:
        output_key = {"50%": "median", "25%": "p25", "75%": "p75", "97.5%": "p97_5"}.get(key, f"p{key.replace('%', '')}" if key.endswith("%") else key)
        row[output_key] = float(stats[key]) if key in stats else np.nan
        if key == "50%":
            row["p50"] = row["median"]
    return pd.DataFrame([row])


def profile_customer_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = events.groupby("Customer ID").size().rename("purchase_event_count")
    exact_one = int(counts.eq(1).sum())
    exact_two = int(counts.eq(2).sum())
    three_plus = int(counts.ge(3).sum())
    total = len(counts)
    groups = pd.DataFrame(
        [
            {"group": "exactly_1_purchase_event", "customers": exact_one},
            {"group": "exactly_2_purchase_events", "customers": exact_two},
            {"group": "3_or_more_purchase_events", "customers": three_plus},
        ]
    )
    groups["percentage_of_customers"] = groups["customers"].map(lambda x: safe_percentage(x, total))
    groups.insert(0, "total_customers_with_positive_purchase", total)
    stats = describe(counts, "purchase_event_count_per_customer", [0.25, 0.5, 0.75, 0.9, 0.95])
    return groups, stats


def customer_interval_summaries(events: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = events.groupby("Customer ID").size().rename("purchase_event_count")
    summary = intervals.groupby("Customer ID")["interval_days"].agg(
        interval_count="size", median_interpurchase_days="median", mean_interpurchase_days="mean", max_interpurchase_days="max"
    ).join(counts)
    summary = summary.reset_index().sort_values("Customer ID")
    rows = []
    for column, population in [
        ("median_interpurchase_days", "customer_level_median_interval_per_repeat_customer"),
        ("mean_interpurchase_days", "customer_level_mean_interval_per_repeat_customer"),
        ("max_interpurchase_days", "customer_level_max_interval_per_repeat_customer"),
    ]:
        rows.append(describe(summary[column], population, [0.25, 0.5, 0.75, 0.9, 0.95]).iloc[0].to_dict())
    return summary, pd.DataFrame(rows)


def percentile_table(intervals: pd.DataFrame) -> pd.DataFrame:
    values = intervals["interval_days"]
    rows = []
    for percentile in [50, 75, 80, 85, 90, 95, 97.5, 99]:
        q = percentile / 100
        threshold = float(values.quantile(q))
        rows.append(
            {
                "percentile": percentile,
                "interval_days_at_percentile": threshold,
                "intervals_leq_exact_percentile": int(values.le(threshold).sum()),
                "coverage_leq_exact_percentile_pct": safe_percentage(values.le(threshold).sum(), len(values)),
                "intervals_longer_than_exact_percentile": int(values.gt(threshold).sum()),
                "tail_longer_than_exact_percentile_pct": safe_percentage(values.gt(threshold).sum(), len(values)),
            }
        )
    return pd.DataFrame(rows)


def candidate_thresholds(intervals: pd.DataFrame) -> pd.DataFrame:
    values = intervals["interval_days"]
    rows = []
    interpretations = {
        85: "A shorter, less conservative threshold: about 85% of observed consecutive repeat intervals are no longer than this value.",
        90: "A central primary candidate: about 90% of observed consecutive repeat intervals are no longer than this value, while the remaining tail is retained as evidence of slower return behavior.",
        95: "A conservative sensitivity candidate: about 95% of observed consecutive repeat intervals are no longer than this value; it requires a longer future outcome window.",
    }
    for anchor in [85, 90, 95]:
        exact = float(values.quantile(anchor / 100))
        practical = int(round(exact / 5) * 5)
        practical = max(1, practical)
        le = int(values.le(practical).sum())
        rows.append(
            {
                "candidate": f"P{anchor}_rounded_to_5_days",
                "anchor_percentile": anchor,
                "exact_empirical_percentile_days": exact,
                "threshold_days": practical,
                "interval_count": len(values),
                "intervals_leq_threshold": le,
                "coverage_leq_threshold_pct": safe_percentage(le, len(values)),
                "intervals_longer_than_threshold": int(values.gt(practical).sum()),
                "longer_than_threshold_pct": safe_percentage(values.gt(practical).sum(), len(values)),
                "interpretation": interpretations[anchor],
            }
        )
    return pd.DataFrame(rows)


def censoring_and_window_designs(events: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_start = events["purchase_date"].min().date()
    dataset_end = events["purchase_date"].max().date()
    last_purchase = events.groupby("Customer ID")["purchase_date"].max()
    censor_rows = []
    design_rows = []
    for row in candidates.itertuples(index=False):
        threshold = int(row.threshold_days)
        cutoff = dataset_end - timedelta(days=threshold)
        future_start = cutoff + timedelta(days=1)
        observed_end = dataset_end
        final_purchase_after_cutoff = last_purchase.gt(pd.Timestamp(cutoff))
        censor_rows.append(
            {
                "threshold_days": threshold,
                "dataset_end_date": dataset_end,
                "latest_supported_cutoff_date": cutoff,
                "future_outcome_start_date": future_start,
                "available_future_days": (dataset_end - cutoff).days,
                "customers_with_final_observed_purchase_after_latest_supported_cutoff": int(final_purchase_after_cutoff.sum()),
                "percentage_of_positive_purchase_customers_affected": safe_percentage(final_purchase_after_cutoff.sum(), len(last_purchase)),
                "interpretation": "These customers are too close to the dataset end for a full threshold-length future observation; they are not labeled churn in Step 5.",
            }
        )

        observation_end = cutoff
        observation_start = cutoff - timedelta(days=364)
        in_observation = events["purchase_date"].between(pd.Timestamp(observation_start), pd.Timestamp(observation_end), inclusive="both")
        eligible_counts = events.loc[in_observation].groupby("Customer ID").size()
        design_rows.append(
            {
                "design": f"365_day_observation_plus_{threshold}_day_outcome",
                "threshold_days": threshold,
                "observation_window_start": observation_start,
                "observation_window_end_cutoff": observation_end,
                "future_outcome_window_start": future_start,
                "future_outcome_window_end": observed_end,
                "available_future_days": threshold,
                "threshold_fully_supported": True,
                "approx_eligible_customers_at_least_1_observation_event": int(len(eligible_counts)),
                "approx_eligible_customers_at_least_2_observation_events": int(eligible_counts.ge(2).sum()),
                "eligibility_definition": "Positive-purchase customers with at least one invoice event in the 365-day observation window; counts are planning estimates, not a final modeling table.",
                "leakage_control": "Features must use events on or before the cutoff; outcome behavior must use only the later outcome window.",
            }
        )
    return pd.DataFrame(censor_rows), pd.DataFrame(design_rows)


def temporal_eligibility_audit(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit observation-window choices and 135-day baseline eligibility.

    All future-return calculations remain aggregate.  No customer-level outcome
    or Churn field is returned or written.
    """

    threshold = 135
    cutoff = date(2011, 7, 27)
    future_start = cutoff + timedelta(days=1)
    future_end = events["purchase_date"].max().date()
    positive_customers = int(events["Customer ID"].nunique())
    dataset_start = events["purchase_date"].min().date()
    window_specs = [
        ("180_day_observation", cutoff - timedelta(days=179), "Shorter history; more recent behavior and fewer historical events, but no complete annual cycle."),
        ("270_day_observation", cutoff - timedelta(days=269), "Intermediate history; more behavioral coverage than 180 days, but still shorter than a full annual cycle."),
        ("365_day_observation", cutoff - timedelta(days=364), "Complete one-year cycle; supports seasonal coverage and a stable behavioral history, at the cost of excluding customers with no purchase in the year."),
        ("all_available_pre_cutoff_history", dataset_start, "Maximizes historical evidence and includes a complete annual cycle, but creates unequal history lengths and greater exposure to older behavior."),
    ]
    window_rows = []
    eligible_sets: dict[str, set[object]] = {}
    for name, start, rationale in window_specs:
        in_window = events["purchase_date"].between(pd.Timestamp(start), pd.Timestamp(cutoff), inclusive="both")
        counts = events.loc[in_window].groupby("Customer ID").size()
        eligible_sets[name] = set(counts.index)
        span_days = (cutoff - start).days + 1
        window_rows.append(
            {
                "observation_option": name,
                "observation_start": start,
                "observation_end_cutoff": cutoff,
                "observation_days_inclusive": span_days,
                "eligible_customers_with_at_least_1_positive_purchase": int(len(counts)),
                "eligible_customer_percentage_of_5878": safe_percentage(len(counts), positive_customers),
                "median_purchase_event_count_per_eligible_customer": float(counts.median()),
                "eligible_customers_with_only_1_observed_event": int(counts.eq(1).sum()),
                "percentage_with_only_1_observed_event": safe_percentage(counts.eq(1).sum(), len(counts)),
                "complete_annual_cycle": bool(span_days >= 365),
                "advantages_and_limitations": rationale,
            }
        )

    one_year_customers = eligible_sets["365_day_observation"]
    one_year_mask = events["Customer ID"].isin(one_year_customers) & events["purchase_date"].le(pd.Timestamp(cutoff))
    latest_purchase = events.loc[one_year_mask].groupby("Customer ID")["purchase_date"].max()
    recency_days = (pd.Timestamp(cutoff) - latest_purchase).dt.days
    active_at_cutoff = set(recency_days.index[recency_days.le(threshold)])
    baseline_recency_rows = pd.DataFrame(
        [
            {
                "population": "all_365_day_observation_eligible_customers",
                "recency_condition": "all customers with at least one positive purchase in the 365-day observation window",
                "customers": int(len(recency_days)),
                "percentage_of_365_day_eligible": 100.0,
                "threshold_days": threshold,
            },
            {
                "population": "baseline_recency_leq_135_days",
                "recency_condition": "latest purchase on or after 2011-03-14; recency <= 135 days",
                "customers": int(recency_days.le(threshold).sum()),
                "percentage_of_365_day_eligible": safe_percentage(recency_days.le(threshold).sum(), len(recency_days)),
                "threshold_days": threshold,
            },
            {
                "population": "baseline_recency_gt_135_days",
                "recency_condition": "latest purchase before 2011-03-14; recency > 135 days",
                "customers": int(recency_days.gt(threshold).sum()),
                "percentage_of_365_day_eligible": safe_percentage(recency_days.gt(threshold).sum(), len(recency_days)),
                "threshold_days": threshold,
            },
        ]
    )

    outcome_mask = events["purchase_date"].between(pd.Timestamp(future_start), pd.Timestamp(future_end), inclusive="both")
    outcome_customers = set(events.loc[outcome_mask, "Customer ID"])
    population_rows = []
    for name, label, population in [
        ("population_A", "all 365-day observation-eligible customers", one_year_customers),
        ("population_B", "365-day observation-eligible customers with baseline recency <= 135 days", active_at_cutoff),
    ]:
        returned = len(population & outcome_customers)
        no_return = len(population) - returned
        population_rows.extend(
            [
                {
                    "population": name,
                    "population_definition": label,
                    "future_outcome_start": future_start,
                    "future_outcome_end": future_end,
                    "population_customers": len(population),
                    "customers_with_at_least_1_positive_purchase_in_outcome_window": returned,
                    "percentage_with_positive_purchase_in_outcome_window": safe_percentage(returned, len(population)),
                    "customers_with_no_positive_purchase_in_outcome_window": no_return,
                    "percentage_with_no_positive_purchase_in_outcome_window": safe_percentage(no_return, len(population)),
                    "note": "Aggregate design assessment only; no customer-level outcome or Churn variable was persisted.",
                }
            ]
        )
    return pd.DataFrame(window_rows), baseline_recency_rows, pd.DataFrame(population_rows)


def threshold_stability_audit(events: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assess the 135-day candidate using only events available by the cutoff."""

    cutoff = pd.Timestamp("2011-07-27")
    threshold = 135
    pre_events = events.loc[events["purchase_date"].le(cutoff)].copy()
    pre_intervals = build_intervals(pre_events)
    pre_values = pre_intervals["interval_days"]
    full_values = intervals["interval_days"]

    def metric_row(values: pd.Series, population: str) -> dict[str, object]:
        return {
            "population": population,
            "cutoff_date": cutoff.date(),
            "eligible_purchase_events": int(pre_events["Invoice"].nunique()) if population == "pre_cutoff_only" else int(events["Invoice"].nunique()),
            "customers": int(pre_events["Customer ID"].nunique()) if population == "pre_cutoff_only" else int(events["Customer ID"].nunique()),
            "repeat_customers": int(pre_intervals["Customer ID"].nunique()) if population == "pre_cutoff_only" else int(intervals["Customer ID"].nunique()),
            "total_interpurchase_intervals": int(len(values)),
            "mean_days": float(values.mean()),
            "median_days": float(values.median()),
            "p75_days": float(values.quantile(0.75)),
            "p80_days": float(values.quantile(0.80)),
            "p85_days": float(values.quantile(0.85)),
            "p90_days": float(values.quantile(0.90)),
            "p95_days": float(values.quantile(0.95)),
            "p97_5_days": float(values.quantile(0.975)),
            "p99_days": float(values.quantile(0.99)),
            "maximum_days": float(values.max()),
            "zero_day_intervals": int(values.eq(0).sum()),
        }

    summary = pd.DataFrame([metric_row(pre_values, "pre_cutoff_only")])
    comparison_rows = []
    for metric in ["median_days", "p75_days", "p85_days", "p90_days", "p95_days"]:
        full_value = float(full_values.quantile({"median_days": 0.50, "p75_days": 0.75, "p85_days": 0.85, "p90_days": 0.90, "p95_days": 0.95}[metric]))
        pre_value = float(pre_values.quantile({"median_days": 0.50, "p75_days": 0.75, "p85_days": 0.85, "p90_days": 0.90, "p95_days": 0.95}[metric]))
        difference = pre_value - full_value
        comparison_rows.append(
            {
                "statistic": metric,
                "full_history_value_days": full_value,
                "pre_cutoff_value_days": pre_value,
                "absolute_difference_pre_minus_full_days": difference,
                "percentage_difference_vs_full": difference / full_value * 100 if full_value else np.nan,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    coverage = float(pre_values.le(threshold).mean() * 100)
    pre_p90 = float(pre_values.quantile(0.90))
    classification = "reasonably supported"
    rationale = (
        f"135 days covers {coverage:.2f}% of pre-cutoff intervals, leaving {100 - coverage:.2f}% longer than the threshold. "
        f"The pre-cutoff P90 is {pre_p90:.2f} days, {threshold - pre_p90:.2f} days below 135; this supports the candidate as a practical threshold, "
        "but not as an invariant historical P90."
    )
    stability = pd.DataFrame([
        {
            "cutoff_date": cutoff.date(),
            "threshold_days": threshold,
            "pre_cutoff_intervals": int(len(pre_values)),
            "empirical_percentile_at_135_days": coverage,
            "coverage_leq_135_days_pct": coverage,
            "coverage_gt_135_days_pct": 100 - coverage,
            "pre_cutoff_p90_days": pre_p90,
            "classification": classification,
            "quantitative_evidence": rationale,
            "recommendation": "Retain 135 days as a defensible provisional candidate, but review it before final lock because the historical split shows temporal variation.",
        }
    ])

    split_start = pre_events["purchase_date"].min().date()
    split_end = cutoff.date()
    split_date = split_start + timedelta(days=((split_end - split_start).days // 2))
    pre_intervals = pre_intervals.copy()
    pre_intervals["historical_portion"] = np.where(
        pre_intervals["purchase_date"].le(pd.Timestamp(split_date)), "earlier_pre_cutoff", "later_pre_cutoff"
    )
    historical_rows = []
    for portion, group in pre_intervals.groupby("historical_portion", sort=False):
        values = group["interval_days"]
        portion_start = split_start if portion == "earlier_pre_cutoff" else split_date + timedelta(days=1)
        portion_end = split_date if portion == "earlier_pre_cutoff" else split_end
        historical_rows.append(
            {
                "historical_portion": portion,
                "portion_start": portion_start,
                "portion_end": portion_end,
                "interval_count": int(len(values)),
                "repeat_customers": int(group["Customer ID"].nunique()),
                "median_days": float(values.median()),
                "p90_days": float(values.quantile(0.90)),
                "p95_days": float(values.quantile(0.95)),
                "coverage_leq_135_days_pct": float(values.le(threshold).mean() * 100),
                "interpretation": "Diagnostic split by the later event date; no threshold was optimized from this comparison.",
            }
        )
    historical = pd.DataFrame(historical_rows)
    return summary, comparison, stability, historical


def write_figures(intervals: pd.DataFrame, customer_summary: pd.DataFrame, candidates: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    values = intervals["interval_days"]
    candidate_days = candidates["threshold_days"].tolist()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12, "axes.labelsize": 10})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].hist(values, bins=np.linspace(0, 720, 73), color=COLORS["primary"], edgecolor="white", linewidth=0.25)
    axes[0].set_title("Full observed interval range")
    axes[0].set_xlabel("Elapsed time between consecutive invoices (days)")
    axes[0].set_ylabel("Number of intervals")
    axes[0].set_xlim(0, 720)
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    axes[1].hist(values[values.le(250)], bins=np.linspace(0, 250, 51), color=COLORS["secondary"], edgecolor="white", linewidth=0.25)
    axes[1].set_title("Central mass (x-axis restricted to 250 days)")
    axes[1].set_xlabel("Elapsed time between consecutive invoices (days)")
    axes[1].set_ylabel("Number of intervals")
    axes[1].set_xlim(0, 250)
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    for axis in axes:
        for threshold in candidate_days:
            axis.axvline(threshold, color=COLORS["accent"], alpha=0.65, linewidth=0.9)
    fig.suptitle("Step 5: observed interpurchase intervals", color=COLORS["text"])
    interval_figure = FIGURES_DIR / "03_interpurchase_interval_distribution.png"
    fig.savefig(interval_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    ordered = np.sort(values.to_numpy())
    y = np.arange(1, len(ordered) + 1) / len(ordered) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].step(ordered, y, where="post", color=COLORS["primary"], linewidth=1.35)
    axes[0].set_title("ECDF across the full range")
    axes[0].set_xlim(0, 720)
    axes[1].step(ordered, y, where="post", color=COLORS["secondary"], linewidth=1.35)
    axes[1].set_title("ECDF central view")
    axes[1].set_xlim(0, 250)
    for axis in axes:
        axis.set_xlabel("Elapsed time between consecutive invoices (days)")
        axis.set_ylabel("Observed repeat intervals ≤ x (%)")
        axis.set_ylim(0, 100)
        axis.grid(color=COLORS["grid"], linewidth=0.6)
        for threshold in candidate_days:
            coverage = (values.le(threshold).mean() * 100)
            axis.axvline(threshold, color=COLORS["accent"], alpha=0.55, linewidth=0.9)
            if threshold <= axis.get_xlim()[1]:
                axis.axhline(coverage, color=COLORS["accent"], alpha=0.18, linewidth=0.7)
    fig.suptitle("Step 5: empirical cumulative distribution of repeat-purchase intervals", color=COLORS["text"])
    ecdf_figure = FIGURES_DIR / "04_interpurchase_ecdf.png"
    fig.savefig(ecdf_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.hist(customer_summary["median_interpurchase_days"], bins=np.linspace(0, 250, 51), color=COLORS["primary"], alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.set_title("Typical interval by repeat customer")
    ax.set_xlabel("Customer-level median interpurchase interval (days; view limited to 250)")
    ax.set_ylabel("Number of repeat customers")
    ax.set_xlim(0, 250)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    for threshold in candidate_days:
        if threshold <= 250:
            ax.axvline(threshold, color=COLORS["accent"], alpha=0.65, linewidth=0.9)
    fig.suptitle("Step 5: customer-level typical return behavior", color=COLORS["text"])
    customer_figure = FIGURES_DIR / "05_customer_typical_interval_distribution.png"
    fig.savefig(customer_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [interval_figure, ecdf_figure, customer_figure]


def write_csv_tables(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    customer_summary: pd.DataFrame,
    customer_groups: pd.DataFrame,
    customer_count_stats: pd.DataFrame,
    interval_stats: pd.DataFrame,
    positive_interval_stats: pd.DataFrame,
    customer_interval_stats: pd.DataFrame,
    percentiles: pd.DataFrame,
    candidates: pd.DataFrame,
    censoring: pd.DataFrame,
    designs: pd.DataFrame,
    window_audit: pd.DataFrame,
    baseline_recency: pd.DataFrame,
    future_return: pd.DataFrame,
    pre_cutoff_summary: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    threshold_stability: pd.DataFrame,
    historical_stability: pd.DataFrame,
    invoice_audit: pd.DataFrame,
) -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "57_invoice_event_validation.csv": pd.DataFrame([
            {"metric": "positive_purchase_rows", "value": int(len(pd.read_parquet(POSITIVE_PURCHASE_PATH)))},
            {"metric": "unique_positive_purchase_invoices", "value": int(events["Invoice"].nunique())},
            {"metric": "purchase_event_timestamp_min", "value": events["purchase_timestamp"].min().isoformat()},
            {"metric": "purchase_event_timestamp_max", "value": events["purchase_timestamp"].max().isoformat()},
            {"metric": "invoices_with_multiple_line_items", "value": int(invoice_audit["invoice_line_count"].gt(1).sum())},
            {"metric": "invoices_with_inconsistent_timestamps", "value": int(invoice_audit["invoice_timestamp_inconsistent"].sum())},
            {"metric": "inconsistent_invoices_same_calendar_day", "value": int(invoice_audit.loc[invoice_audit["invoice_timestamp_inconsistent"], "same_calendar_day_when_inconsistent"].sum())},
            {"metric": "timestamp_selection", "value": "earliest InvoiceDate within each Invoice; all inconsistent cases remain within one calendar day"},
            {"metric": "invoice_customer_consistency", "value": "every positive-purchase Invoice maps to exactly one Customer ID"},
        ]),
        "58_customer_purchase_frequency_profile.csv": customer_groups,
        "59_customer_purchase_event_count_descriptives.csv": customer_count_stats,
        "60_interpurchase_interval_descriptives.csv": interval_stats,
        "61_customer_level_interval_descriptives.csv": customer_interval_stats,
        "62_interpurchase_percentile_table.csv": percentiles,
        "63_candidate_inactivity_thresholds.csv": candidates,
        "64_right_censoring_assessment.csv": censoring,
        "65_observation_outcome_window_designs.csv": designs,
        "67_positive_only_interval_descriptives.csv": positive_interval_stats,
        "68_observation_window_comparison_135_day_cutoff.csv": window_audit,
        "69_baseline_recency_eligibility_135_day_cutoff.csv": baseline_recency,
        "70_future_return_aggregate_135_day_design.csv": future_return,
        "71_pre_cutoff_interval_stability_summary.csv": pre_cutoff_summary,
        "72_full_vs_pre_cutoff_interval_comparison.csv": threshold_comparison,
        "73_pre_cutoff_historical_stability_diagnostic.csv": historical_stability,
        "74_threshold_135_day_stability_assessment.csv": threshold_stability,
    }
    paths = []
    for filename, frame in tables.items():
        path = TABLES_DIR / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_output_manifest(paths: list[Path], figures: list[Path], metadata: dict[str, str]) -> Path:
    rows = []
    for path in [EVENT_PATH, INTERVAL_PATH, CUSTOMER_SUMMARY_PATH, *paths, *figures, REPORT_PATH, NOTEBOOK_PATH, LOG_PATH]:
        if path.exists():
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(rows)
    manifest_path = TABLES_DIR / "66_step5_output_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def write_report(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    customer_groups: pd.DataFrame,
    customer_count_stats: pd.DataFrame,
    interval_stats: pd.DataFrame,
    positive_interval_stats: pd.DataFrame,
    customer_interval_stats: pd.DataFrame,
    percentiles: pd.DataFrame,
    candidates: pd.DataFrame,
    censoring: pd.DataFrame,
    designs: pd.DataFrame,
    window_audit: pd.DataFrame,
    baseline_recency: pd.DataFrame,
    future_return: pd.DataFrame,
    pre_cutoff_summary: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    threshold_stability: pd.DataFrame,
    historical_stability: pd.DataFrame,
    metadata: dict[str, str],
) -> None:
    def fmt(value: object, digits: int = 2) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (float, np.floating)):
            return f"{value:,.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    event_count = len(events)
    customer_count = events["Customer ID"].nunique()
    interval_count = len(intervals)
    repeat_customers = intervals["Customer ID"].nunique()
    zero_count = int(intervals["zero_day_interval"].sum())
    interval_row = interval_stats.iloc[0]
    customer_row = customer_count_stats.iloc[0]
    primary = candidates.loc[candidates["anchor_percentile"].eq(90)].iloc[0]
    censor_primary = censoring.loc[censoring["threshold_days"].eq(int(primary["threshold_days"]))].iloc[0]
    design_primary = designs.loc[designs["threshold_days"].eq(int(primary["threshold_days"]))].iloc[0]
    window_365 = window_audit.loc[window_audit["observation_option"].eq("365_day_observation")].iloc[0]
    recency_all = baseline_recency.loc[baseline_recency["population"].eq("all_365_day_observation_eligible_customers")].iloc[0]
    recency_active = baseline_recency.loc[baseline_recency["population"].eq("baseline_recency_leq_135_days")].iloc[0]
    recency_lapsed = baseline_recency.loc[baseline_recency["population"].eq("baseline_recency_gt_135_days")].iloc[0]
    future_a = future_return.loc[future_return["population"].eq("population_A")].iloc[0]
    future_b = future_return.loc[future_return["population"].eq("population_B")].iloc[0]
    pre_summary = pre_cutoff_summary.iloc[0]
    stability_row = threshold_stability.iloc[0]
    historical_earlier = historical_stability.loc[historical_stability["historical_portion"].eq("earlier_pre_cutoff")].iloc[0]
    historical_later = historical_stability.loc[historical_stability["historical_portion"].eq("later_pre_cutoff")].iloc[0]

    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        selected = frame.loc[:, columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in selected.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
        return "\n".join(lines)

    report = f"""# Step 5 — Interpurchase-Time Analysis and Data-Driven Churn-Threshold Discovery

## Scope and stopping point

This stage describes purchasing rhythm in the validated Step 4 positive-purchase base. It creates invoice-level purchase events, interpurchase intervals, descriptive tables, figures, and temporal-design options. It does **not** create a churn label, a final RFM or customer-modeling table, train/test data, encoded or scaled variables, a model, or SHAP output.

The input was `data/processed/03_positive_purchase_base.parquet`, produced under the validated primary methodology in which exact within-sheet duplicates were removed before reconciliation. The raw workbook and Step 4 inputs were read-only.

## 1. Purchase-event definition and validation

The positive-purchase base contains **{len(pd.read_parquet(POSITIVE_PURCHASE_PATH)):,} line-item rows**, which collapse to **{event_count:,} unique positive-purchase Invoice events** for **{customer_count:,} customers**. The validated purchase-event range is **{events['purchase_timestamp'].min():%Y-%m-%d %H:%M} to {events['purchase_timestamp'].max():%Y-%m-%d %H:%M}**. Each invoice maps to exactly one Customer ID. Invoices with multiple line items were counted once, not once per product line; **{int((events['invoice_line_count'] > 1).sum()):,}** invoices contain multiple positive-purchase lines.

There are **{int((events['invoice_timestamp_inconsistent']).sum()):,}** invoices whose line items do not share exactly the same timestamp. Every one remains within a single calendar day, with only minute-level variation. The event timestamp is therefore the **earliest InvoiceDate within the invoice**, preserving a conservative and reproducible event ordering. The selected purchase date is its calendar date.

## 2. Customer purchase-event counts

{markdown_table(customer_groups, ['group', 'customers', 'percentage_of_customers'])}

The event-count distribution has mean **{fmt(customer_row['mean'])}**, median **{fmt(customer_row['median'])}**, standard deviation **{fmt(customer_row['std'])}**, minimum **{fmt(customer_row['min'])}**, P25 **{fmt(customer_row['p25'])}**, P75 **{fmt(customer_row['p75'])}**, P90 **{fmt(customer_row['p90'])}**, P95 **{fmt(customer_row['p95'])}**, and maximum **{fmt(customer_row['max'])}** events per customer. Only repeat customers contribute an observed interval; in this base, **{repeat_customers:,}** customers contribute at least one interval.

## 3. Interpurchase intervals

Intervals are calculated between consecutive invoice-level events after chronological ordering within Customer ID. There are **{interval_count:,}** intervals. **{zero_count:,}** are zero-day intervals (**{safe_percentage(zero_count, interval_count):.2f}%**); these are retained and represent distinct invoices on the same calendar day. No long intervals were removed as outliers.

{markdown_table(interval_stats, ['population', 'count', 'min', 'mean', 'median', 'std', 'p25', 'p50', 'p75', 'p80', 'p85', 'p90', 'p95', 'p97_5', 'p99', 'max'])}

A positive-only sensitivity view is recorded in `67_positive_only_interval_descriptives.csv`; zero-day intervals were not silently excluded from the canonical interval dataset. In that sensitivity view, the median is **{fmt(positive_interval_stats.iloc[0]['median'])} days** and the P90 is **{fmt(positive_interval_stats.iloc[0]['p90'])} days**.

## 4. Visual findings

`03_interpurchase_interval_distribution.png` shows both the full 0–720-day range and a 0–250-day view of the central mass. `04_interpurchase_ecdf.png` shows the cumulative percentage of observed repeat intervals completed by each elapsed time. The distribution is strongly right-skewed: the median interval is about **{fmt(interval_row['median'])} days**, while the maximum is **{fmt(interval_row['max'])} days**. The central mass rises quickly, but a material long tail remains. The third figure, `05_customer_typical_interval_distribution.png`, shows customer-level median intervals rather than pooling every interval.

## 5. Pooled intervals versus customer-level typical intervals

The pooled distribution weights customers by their number of intervals: a customer with many invoices contributes many observations. The customer-level distribution gives each repeat customer one summary statistic, such as that customer's median return interval. The latter answers how a typical repeat customer's usual interval varies, whereas the pooled distribution answers how often an observed consecutive purchase gap of a given length occurs. They are not interchangeable and neither is automatically the final churn basis.

{markdown_table(customer_interval_stats, ['population', 'count', 'mean', 'median', 'std', 'p25', 'p75', 'p90', 'p95', 'max'])}

## 6. Candidate inactivity thresholds

The candidate set is anchored to the observed P85, P90, and P95 interval percentiles, then rounded to the nearest five days for practical interpretation. Coverage is recomputed at the rounded threshold; no future model performance was used.

{markdown_table(candidates, ['candidate', 'anchor_percentile', 'exact_empirical_percentile_days', 'threshold_days', 'coverage_leq_threshold_pct', 'longer_than_threshold_pct'])}

There is no single visually inevitable cutoff. A defensible primary candidate for later review is **{int(primary['threshold_days'])} days**, because it is the rounded P90 anchor: approximately **{primary['coverage_leq_threshold_pct']:.2f}%** of observed consecutive repeat intervals are no longer than this threshold and **{primary['longer_than_threshold_pct']:.2f}%** are longer. The **{int(candidates.iloc[0]['threshold_days'])}-day** candidate is a shorter sensitivity option, while **{int(candidates.iloc[2]['threshold_days'])} days** is a more conservative sensitivity option that requires a longer future window. This is a recommendation for review, not an executed churn definition.

## 7. Right censoring

The last positive-purchase observation is **{events['purchase_timestamp'].max():%Y-%m-%d %H:%M}**. A customer whose last observed purchase is too close to that endpoint cannot be assessed over a full inactivity horizon without risking false apparent inactivity. For a threshold of T days, the latest fully supported calendar cutoff is the dataset end date minus T days.

{markdown_table(censoring, ['threshold_days', 'latest_supported_cutoff_date', 'future_outcome_start_date', 'available_future_days', 'customers_with_final_observed_purchase_after_latest_supported_cutoff', 'percentage_of_positive_purchase_customers_affected'])}

For the recommended review candidate of **{int(primary['threshold_days'])} days**, the latest supported cutoff is **{censor_primary['latest_supported_cutoff_date']}**. Customers whose final observed purchase is later than that cutoff are end-of-study censored for this horizon; they are **not** labeled churn or non-churn here.

## 8. Proposed observation/outcome-window designs

The following planning designs use a fixed 365-day observation window and reserve the entire remaining period through 2011-12-09 for the future outcome window. Features in a later stage must use only transactions on or before the cutoff; outcome behavior must use only transactions after it. Counts are eligibility planning estimates, not a final modeling table.

{markdown_table(designs, ['design', 'threshold_days', 'observation_window_start', 'observation_window_end_cutoff', 'future_outcome_window_start', 'future_outcome_window_end', 'available_future_days', 'approx_eligible_customers_at_least_1_observation_event', 'approx_eligible_customers_at_least_2_observation_events'])}

All three designs fully support their associated threshold by construction. The 100-day design preserves the latest cutoff and more future-eligible customers; the 135-day design balances a longer horizon with the P90 empirical support; the 205-day design is the most conservative and loses more historical recency from the end-of-study constraint. These are temporal options, not labels.

## 9. Temporal-design addendum: separate decisions

Two decisions must remain separate. First, the **inactivity threshold** is an empirical candidate derived from the pooled interpurchase-interval distribution: 100, 135, and 205 days correspond approximately to the P85, P90, and P95 behavior. Second, the **observation-window length** is an analytical design choice. The 365-day window was not selected by the interpurchase percentile analysis; it is being considered because it captures a complete annual cycle and provides a consistent history length for later features.

The comparison below evaluates the provisional 135-day cutoff (**2011-07-27**) and its fully supported outcome window (**2011-07-28 through 2011-12-09**).

## 10. Observation-window comparison at the 135-day cutoff

{markdown_table(window_audit, ['observation_option', 'observation_start', 'observation_end_cutoff', 'observation_days_inclusive', 'eligible_customers_with_at_least_1_positive_purchase', 'eligible_customer_percentage_of_5878', 'median_purchase_event_count_per_eligible_customer', 'percentage_with_only_1_observed_event', 'complete_annual_cycle'])}

The 180-day and 270-day options provide shorter, more recent histories but do not cover a full annual cycle. The 365-day option covers one complete annual cycle and retains a substantial eligible population (**{int(window_365['eligible_customers_with_at_least_1_positive_purchase']):,} customers; {window_365['eligible_customer_percentage_of_5878']:.2f}% of positive-purchase customers). All available pre-cutoff history maximizes historical evidence and also covers a complete annual cycle, but produces unequal effective history lengths and gives older transactions more influence. The 365-day choice is therefore a design rationale, not an empirical consequence of the P90 threshold and not a choice made solely to maximize sample size.

Advantages and limitations for each option are recorded in table 68. No option is converted into a final modeling table here.

## 11. Baseline recency and eligibility

Among customers with at least one positive purchase in the 365-day observation window, **{int(recency_all['customers']):,}** are available. Using only purchases on or before 2011-07-27, **{int(recency_active['customers']):,}** have baseline recency <= 135 days and **{int(recency_lapsed['customers']):,}** have baseline recency > 135 days.

{markdown_table(baseline_recency, ['population', 'recency_condition', 'customers', 'percentage_of_365_day_eligible', 'threshold_days'])}

Population A includes every customer observed during the prior year, including customers who had already gone beyond 135 days without a purchase at the cutoff. Population B restricts the prediction population to customers who had not yet exceeded 135 days of inactivity at the cutoff. The latter is a narrower, intervention-oriented population; the former answers a broader monitoring question and includes customers who may already require reactivation rather than preventive retention.

## 12. Aggregate future-return assessment

The following counts use the 135-day future window only for design assessment. They are aggregate counts; no customer-level outcome or `Churn` variable was saved.

{markdown_table(future_return, ['population', 'population_customers', 'customers_with_at_least_1_positive_purchase_in_outcome_window', 'percentage_with_positive_purchase_in_outcome_window', 'customers_with_no_positive_purchase_in_outcome_window', 'percentage_with_no_positive_purchase_in_outcome_window'])}

Population A asks: **“Given customers observed during the prior year, who will make no purchase during the next 135 days?”** Population B asks: **“Among customers who have not yet exceeded 135 days of inactivity at the prediction date, who will fail to purchase during the next 135 days?”** Population B is more closely aligned with a managerial retention/intervention use case because it excludes customers who had already crossed the proposed inactivity horizon at the intervention date. Population A remains useful if the intended business action includes both preventive retention and reactivation. The aggregate outcome split was not used to optimize the threshold.

### Provisional design recommendation

For methodological review, retain the **365-day observation window** because it covers a complete annual cycle while keeping a substantial and consistently defined eligible population. For the modeling population, **Population B** is the more targeted provisional choice for retention intervention; Population A should remain a documented sensitivity or broader-monitoring alternative. Neither recommendation creates or locks the final churn label.

## 13. Pre-cutoff threshold-stability audit

This audit recalculates invoice-level intervals using only selected invoice timestamps on or before **2011-07-27**. It excludes **{int(events['purchase_date'].gt(pd.Timestamp('2011-07-27')).sum()):,}** post-cutoff purchase events from the stability calculation. The previously validated full-history interval dataset and statistics were not altered.

{markdown_table(pre_cutoff_summary, ['population', 'cutoff_date', 'eligible_purchase_events', 'customers', 'repeat_customers', 'total_interpurchase_intervals', 'mean_days', 'median_days', 'p75_days', 'p80_days', 'p85_days', 'p90_days', 'p95_days', 'p97_5_days', 'p99_days', 'maximum_days'])}

### Full history versus pre-cutoff-only intervals

{markdown_table(threshold_comparison, ['statistic', 'full_history_value_days', 'pre_cutoff_value_days', 'absolute_difference_pre_minus_full_days', 'percentage_difference_vs_full'])}

At 135 days, the pre-cutoff interval distribution has empirical coverage of **{stability_row['empirical_percentile_at_135_days']:.2f}%**: **{stability_row['coverage_leq_135_days_pct']:.2f}%** of intervals are <=135 days and **{stability_row['coverage_gt_135_days_pct']:.2f}%** are longer. The pre-cutoff P90 is **{stability_row['pre_cutoff_p90_days']:.2f} days**, compared with the full-history P90 of **{primary['exact_empirical_percentile_days']:.2f} days**. The provisional threshold is therefore **{stability_row['classification']}**, rather than strongly supported: it retains approximately P90-level coverage, but the pre-cutoff P90 is about **{int(threshold_stability.iloc[0]['threshold_days']) - stability_row['pre_cutoff_p90_days']:.2f} days** below 135.

### Optional historical stability diagnostic

The pre-cutoff interval sample was split at **2010-09-28** by the later event date. This is a descriptive stability check, not a second threshold-selection exercise.

{markdown_table(historical_stability, ['historical_portion', 'portion_start', 'portion_end', 'interval_count', 'repeat_customers', 'median_days', 'p90_days', 'p95_days', 'coverage_leq_135_days_pct'])}

The earlier portion has a P90 of **{historical_earlier['p90_days']:.2f} days**, while the later portion has a P90 of **{historical_later['p90_days']:.2f} days**. This is meaningful temporal variation, so 135 days remains defensible as a provisional practical threshold but should be reviewed before it is locked as the final primary definition.

## 14. Separation of findings, candidates, and decisions

### Empirical findings

- Positive purchases produce **{event_count:,}** invoice events across **{customer_count:,}** customers.
- **{interval_count:,}** consecutive-event intervals are observed across **{repeat_customers:,}** repeat customers.
- The interval distribution is right-skewed, includes **{zero_count:,}** retained zero-day intervals, and has a long tail to approximately **{fmt(interval_row['max'])} days**.
- Invoice events are unique by invoice and customer-consistent; line items were not treated as separate purchases.

### Candidate churn definitions

The data support reviewing **{int(candidates.iloc[0]['threshold_days'])}**, **{int(candidates.iloc[1]['threshold_days'])}**, and **{int(candidates.iloc[2]['threshold_days'])}** days, anchored respectively to the empirical P85, P90, and P95 interval behavior. No final inactivity threshold has been selected or encoded.

### Temporal-design options

The 365-day observation designs in table 65 pair each candidate with a fully supported future outcome window ending on the dataset endpoint. Table 68 separately compares the 180-day, 270-day, 365-day, and all-available alternatives for the provisional 135-day cutoff. They prevent future transactions from entering later cutoff-based features, subject to implementing the cutoff rule exactly.

### Decision still required

Before churn labels are generated, the final primary threshold and observation/outcome-window design must be reviewed. **Recommendation for that review:** use **{int(primary['threshold_days'])} days** with the matching 365-day observation window as the provisional primary candidate, and retain the 100-day and {int(candidates.iloc[2]['threshold_days'])}-day designs as sensitivity options. This recommendation was not executed as a label.

## Validation and scope controls

- Step 5 starts from the validated Step 4 positive-purchase base and the approved pre-reconciliation duplicate methodology.
- Invoice events are unique positive-purchase invoices, customer-consistent, and chronologically ordered within customer.
- Intervals use only adjacent observed events for descriptive timing analysis; no historical feature table was constructed.
- No `Churn` column, churn label, final RFM/customer modeling feature table, observation label, or outcome label was created.
- No train/test split, encoding, scaling, model training, or SHAP analysis occurred.
- Raw Excel and Step 4 canonical/positive-purchase inputs remained unchanged by hash verification.
- Step 5 processed analytical datasets are written under `data/processed` and remain Git-ignored.
- Temporal-design outputs are aggregate audits only; no customer-level future-return or Churn field was persisted.
- Threshold-stability intervals used only events on or before the 2011-07-27 cutoff; post-cutoff events were excluded from that calculation.

Input hashes recorded for reproducibility: positive-purchase `{metadata['positive_purchase_sha256_before']}`, canonical `{metadata['canonical_sha256_before']}`, raw `{metadata['raw_sha256_before']}`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_notebook() -> None:
    """Write a lightweight, inspectable notebook following the Step 4 pattern."""

    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Step 5 — Interpurchase-time analysis\n",
                "\n",
                "This notebook inspects the deterministic Step 5 outputs. It does not create churn labels, modeling features, train/test data, models, or SHAP output."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "sys.path.insert(0, str(ROOT / 'src'))\n",
                "from churn_analysis.analyze_interpurchase_time import (\n",
                "    EVENT_PATH, INTERVAL_PATH, CUSTOMER_SUMMARY_PATH\n",
                ")\n",
                "TABLES = ROOT / 'outputs' / 'tables'\n",
                "events = pd.read_parquet(EVENT_PATH)\n",
                "intervals = pd.read_parquet(INTERVAL_PATH)\n",
                "customer_summary = pd.read_parquet(CUSTOMER_SUMMARY_PATH)\n",
                "percentiles = pd.read_csv(TABLES / '62_interpurchase_percentile_table.csv')\n",
                "positive_only = pd.read_csv(TABLES / '67_positive_only_interval_descriptives.csv')\n",
                "candidates = pd.read_csv(TABLES / '63_candidate_inactivity_thresholds.csv')\n",
                "designs = pd.read_csv(TABLES / '65_observation_outcome_window_designs.csv')\n",
                "window_audit = pd.read_csv(TABLES / '68_observation_window_comparison_135_day_cutoff.csv')\n",
                "baseline_recency = pd.read_csv(TABLES / '69_baseline_recency_eligibility_135_day_cutoff.csv')\n",
                "future_return = pd.read_csv(TABLES / '70_future_return_aggregate_135_day_design.csv')\n",
                "pre_cutoff_summary = pd.read_csv(TABLES / '71_pre_cutoff_interval_stability_summary.csv')\n",
                "threshold_comparison = pd.read_csv(TABLES / '72_full_vs_pre_cutoff_interval_comparison.csv')\n",
                "historical_stability = pd.read_csv(TABLES / '73_pre_cutoff_historical_stability_diagnostic.csv')\n",
                "threshold_stability = pd.read_csv(TABLES / '74_threshold_135_day_stability_assessment.csv')"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Event and interval validation"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(events.head())\n",
                "display(intervals.head())\n",
                "display(events.groupby('Customer ID').size().describe())"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Empirical interval distribution and candidate thresholds"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(percentiles)\n",
                "display(positive_only)\n",
                "display(candidates)\n",
                "display(designs)\n",
                "display(window_audit)\n",
                "display(baseline_recency)\n",
                "display(future_return)\n",
                "display(pre_cutoff_summary)\n",
                "display(threshold_comparison)\n",
                "display(historical_stability)\n",
                "display(threshold_stability)"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": [
                "Figures are saved under `outputs/figures/03_interpurchase_interval_distribution.png`, `04_interpurchase_ecdf.png`, and `05_customer_typical_interval_distribution.png`. Tables 68–70 provide the temporal-design addendum and tables 71–74 provide the pre-cutoff threshold-stability audit. No final churn decision is made in this notebook."
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def validate_scope(metadata: dict[str, str], events: pd.DataFrame, intervals: pd.DataFrame, outputs: list[Path]) -> dict[str, bool]:
    normalized = [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in events.columns]
    normalized += [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in intervals.columns]
    csv_outputs = [path for path in outputs if path.suffix.lower() == ".csv" and path.exists()]
    csv_columns = [column for path in csv_outputs for column in pd.read_csv(path, nrows=0).columns]
    normalized_csv = [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in csv_columns]
    checks = {
        "positive_purchase_base_unchanged": sha256_file(POSITIVE_PURCHASE_PATH) == metadata["positive_purchase_sha256_before"],
        "canonical_source_unchanged": sha256_file(CANONICAL_PATH) == metadata["canonical_sha256_before"],
        "raw_workbook_unchanged": sha256_file(find_raw_workbook()) == metadata["raw_sha256_before"],
        "events_unique_by_invoice": not events["Invoice"].duplicated().any(),
        "invoice_customer_consistency_verified": bool(events["Customer ID"].notna().all()),
        "events_chronologically_ordered_within_customer": bool(
            events.groupby("Customer ID", sort=False)["purchase_timestamp"].apply(lambda s: s.is_monotonic_increasing).all()
        ),
        "intervals_nonnegative": bool(intervals["interval_days"].ge(0).all()),
        "no_churn_column": not any("churn" in column for column in normalized),
        "no_persisted_churn_column_in_step5_tables": not any("churn" in column for column in normalized_csv),
        "future_return_table_is_aggregate_only": not any(column == "customer id" for column in csv_columns),
        "no_modeling_feature_table": not any("rfm" in column or "label" in column for column in normalized),
        "no_train_test_or_model_artifacts": not any(
            path.name != ".gitkeep" for path in (PROJECT_ROOT / "outputs" / "models").glob("*")
        ),
        "processed_step5_files_git_ignored": all(
            subprocess.run(["git", "check-ignore", "--no-index", str(path.relative_to(PROJECT_ROOT))], cwd=PROJECT_ROOT, capture_output=True).returncode == 0
            for path in [EVENT_PATH, INTERVAL_PATH, CUSTOMER_SUMMARY_PATH]
        ),
        "no_unapproved_scope_outputs": all(path.exists() for path in outputs),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Step 5 validation failed: {failed}")
    return checks


def run_analysis() -> dict[str, object]:
    logger = get_logger()
    positive, metadata = validate_positive_base()
    events, invoice_audit = build_purchase_events(positive)
    intervals = build_intervals(events)
    customer_groups, customer_count_stats = profile_customer_events(events)
    customer_summary, customer_interval_stats = customer_interval_summaries(events, intervals)
    interval_stats = describe(intervals["interval_days"], "pooled_consecutive_interpurchase_intervals_days", [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
    positive_interval_stats = describe(intervals.loc[intervals["interval_days"].gt(0), "interval_days"], "pooled_positive_interpurchase_intervals_days_zero_day_excluded_sensitivity", [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
    percentiles = percentile_table(intervals)
    candidates = candidate_thresholds(intervals)
    censoring, designs = censoring_and_window_designs(events, candidates)
    window_audit, baseline_recency, future_return = temporal_eligibility_audit(events)
    pre_cutoff_summary, threshold_comparison, threshold_stability, historical_stability = threshold_stability_audit(events, intervals)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENT_PATH, engine="pyarrow", index=False)
    intervals.to_parquet(INTERVAL_PATH, engine="pyarrow", index=False)
    customer_summary.to_parquet(CUSTOMER_SUMMARY_PATH, engine="pyarrow", index=False)
    tables = write_csv_tables(events, intervals, customer_summary, customer_groups, customer_count_stats, interval_stats, positive_interval_stats, customer_interval_stats, percentiles, candidates, censoring, designs, window_audit, baseline_recency, future_return, pre_cutoff_summary, threshold_comparison, threshold_stability, historical_stability, invoice_audit)
    figures = write_figures(intervals, customer_summary, candidates)
    write_notebook()
    write_report(events, intervals, customer_groups, customer_count_stats, interval_stats, positive_interval_stats, customer_interval_stats, percentiles, candidates, censoring, designs, window_audit, baseline_recency, future_return, pre_cutoff_summary, threshold_comparison, threshold_stability, historical_stability, metadata)
    logger.info("Step 5 completed as descriptive interpurchase-time analysis with temporal-design addendum only.")
    logger.info("Events: %s; intervals: %s; repeat customers: %s", len(events), len(intervals), intervals["Customer ID"].nunique())
    logger.info("Candidate thresholds: %s", candidates[["threshold_days", "coverage_leq_threshold_pct"]].to_dict("records"))
    for handler in logger.handlers:
        handler.flush()
    manifest_path = write_output_manifest(tables, figures, metadata)
    checks = validate_scope(metadata, events, intervals, [*tables, *figures, REPORT_PATH, NOTEBOOK_PATH, LOG_PATH, manifest_path])
    logger.info("Validation checks: %s", checks)
    for handler in logger.handlers:
        handler.flush()
    manifest_path = write_output_manifest(tables, figures, metadata)
    return {
        "events": events,
        "intervals": intervals,
        "customer_summary": customer_summary,
        "customer_groups": customer_groups,
        "customer_count_stats": customer_count_stats,
        "interval_stats": interval_stats,
        "customer_interval_stats": customer_interval_stats,
        "percentiles": percentiles,
        "candidates": candidates,
        "censoring": censoring,
        "designs": designs,
        "window_audit": window_audit,
        "baseline_recency": baseline_recency,
        "future_return": future_return,
        "pre_cutoff_summary": pre_cutoff_summary,
        "threshold_comparison": threshold_comparison,
        "threshold_stability": threshold_stability,
        "historical_stability": historical_stability,
        "checks": checks,
        "metadata": metadata,
        "manifest_path": manifest_path,
    }


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "purchase_events": len(result["events"]),
        "customers": int(result["events"]["Customer ID"].nunique()),
        "intervals": len(result["intervals"]),
        "repeat_customers": int(result["intervals"]["Customer ID"].nunique()),
        "zero_day_intervals": int(result["intervals"]["zero_day_interval"].sum()),
        "candidate_thresholds": result["candidates"][["threshold_days", "coverage_leq_threshold_pct"]].to_dict("records"),
        "validation": result["checks"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
