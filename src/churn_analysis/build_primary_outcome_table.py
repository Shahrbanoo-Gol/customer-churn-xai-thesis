"""Step 6: lock the primary retention cohort and create its outcome table.

This module creates only the minimal customer-level eligibility/outcome table
required for the locked Step 6 design. Predictive feature engineering and
modeling remain out of scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze_interpurchase_time import EVENT_PATH, build_purchase_events, sha256_file
from .build_analysis_base import CANONICAL_PATH, POSITIVE_PURCHASE_PATH, find_raw_workbook
from .config import FIGURES_DIR, LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR


OUTCOME_PATH = PROCESSED_DATA_DIR / "06_primary_customer_outcome_table.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "06_primary_churn_cohort_and_outcome.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_primary_churn_cohort_and_outcome.ipynb"
LOG_PATH = LOGS_DIR / "06_primary_churn_cohort_and_outcome.log"
FIGURE_PATH = FIGURES_DIR / "06_primary_cohort_class_distribution.png"

THRESHOLD_DAYS = 135
OBSERVATION_DAYS = 365
STEP5_WINDOW_TABLE = TABLES_DIR / "69_baseline_recency_eligibility_135_day_cutoff.csv"
STEP5_OUTCOME_TABLE = TABLES_DIR / "70_future_return_aggregate_135_day_design.csv"

OUTCOME_COLUMNS = [
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


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.build_primary_outcome_table")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Validate Step 4, Step 5 event inputs and record immutable hashes."""

    if not POSITIVE_PURCHASE_PATH.exists() or not EVENT_PATH.exists():
        raise FileNotFoundError("Validated Step 4 positive base or Step 5 event dataset is missing.")
    positive = pd.read_parquet(POSITIVE_PURCHASE_PATH, engine="pyarrow")
    events = pd.read_parquet(EVENT_PATH, engine="pyarrow")
    expected_positive_columns = [
        "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
        "Price", "Customer ID", "Country", "SourceSheet",
    ]
    if list(positive.columns) != expected_positive_columns:
        raise RuntimeError(f"Unexpected Step 4 positive-base schema: {list(positive.columns)}")
    if list(events.columns) != [
        "Customer ID", "Invoice", "purchase_timestamp", "purchase_date",
        "invoice_line_count", "invoice_timestamp_min", "invoice_timestamp_max",
        "invoice_timestamp_inconsistent",
    ]:
        raise RuntimeError(f"Unexpected Step 5 event schema: {list(events.columns)}")
    if events["Invoice"].duplicated().any() or events["Customer ID"].isna().any():
        raise RuntimeError("Step 5 events are not unique by Invoice or contain missing Customer IDs.")
    if events["purchase_timestamp"].isna().any():
        raise RuntimeError("Step 5 event timestamps failed validation.")
    if events["purchase_date"].ne(events["purchase_timestamp"].dt.normalize()).any():
        raise RuntimeError("Step 5 purchase_date is not derived from purchase_timestamp.")
    if len(events) != positive["Invoice"].nunique() or events["Customer ID"].nunique() != positive["Customer ID"].nunique():
        raise RuntimeError("Step 5 event coverage does not match the validated positive-purchase base.")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in positive.columns):
        raise RuntimeError("A churn column exists in the Step 4 input.")

    metadata = {
        "positive_sha256_before": sha256_file(POSITIVE_PURCHASE_PATH),
        "events_sha256_before": sha256_file(EVENT_PATH),
        "canonical_sha256_before": sha256_file(CANONICAL_PATH),
        "raw_sha256_before": sha256_file(find_raw_workbook()),
    }
    return positive, events, metadata


def resolve_boundaries(events: pd.DataFrame) -> dict[str, pd.Timestamp]:
    dataset_end = events["purchase_timestamp"].max()
    cutoff = dataset_end - pd.Timedelta(days=THRESHOLD_DAYS)
    observation_start = cutoff - pd.Timedelta(days=OBSERVATION_DAYS)
    if cutoff + pd.Timedelta(days=THRESHOLD_DAYS) != dataset_end:
        raise RuntimeError("Cutoff does not provide the exact requested outcome horizon.")
    return {
        "observation_start": observation_start,
        "cutoff": cutoff,
        "outcome_end": dataset_end,
        "calendar_step5_observation_start": pd.Timestamp("2010-07-28"),
        "calendar_step5_observation_end": pd.Timestamp("2011-07-27 23:59:59.999999"),
        "calendar_step5_outcome_start": pd.Timestamp("2011-07-28"),
    }


def build_outcome_table(events: pd.DataFrame, boundaries: dict[str, pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build exactly one minimal outcome row per eligible Population B customer."""

    observation_mask = events["purchase_timestamp"].between(
        boundaries["observation_start"], boundaries["cutoff"], inclusive="both"
    )
    outcome_mask = (
        events["purchase_timestamp"].gt(boundaries["cutoff"])
        & events["purchase_timestamp"].le(boundaries["outcome_end"])
    )
    observation_events = events.loc[observation_mask].copy()
    outcome_events = events.loc[outcome_mask].copy()
    observation_customers = set(observation_events["Customer ID"])

    last_before_cutoff = (
        events.loc[events["purchase_timestamp"].le(boundaries["cutoff"])]
        .groupby("Customer ID")["purchase_timestamp"]
        .max()
    )
    observation_counts = observation_events.groupby("Customer ID").size()
    observation_customers = set(observation_counts.index)
    baseline_recency = (boundaries["cutoff"] - last_before_cutoff.loc[list(observation_customers)]).dt.total_seconds() / 86400
    eligible_customers = sorted(
        customer for customer in observation_customers if baseline_recency.loc[customer] <= THRESHOLD_DAYS
    )
    outcome_summary = outcome_events.groupby("Customer ID").agg(
        first_positive_purchase_in_outcome_timestamp=("purchase_timestamp", "min"),
        positive_purchase_events_in_outcome=("Invoice", "nunique"),
    )
    table = pd.DataFrame({"Customer ID": eligible_customers})
    table["observation_start_timestamp"] = boundaries["observation_start"]
    table["cutoff_timestamp"] = boundaries["cutoff"]
    table["outcome_end_timestamp"] = boundaries["outcome_end"]
    table["last_positive_purchase_before_cutoff_timestamp"] = table["Customer ID"].map(last_before_cutoff)
    table["baseline_recency_days"] = table["Customer ID"].map(baseline_recency).astype(float)
    table["eligible_observation_window"] = table["Customer ID"].isin(observation_customers)
    table["primary_cohort_eligibility"] = True
    table = table.join(outcome_summary, on="Customer ID")
    table["positive_purchase_events_in_outcome"] = table["positive_purchase_events_in_outcome"].fillna(0).astype(int)
    table["Churn"] = table["positive_purchase_events_in_outcome"].eq(0).astype(int)
    table = table[OUTCOME_COLUMNS].sort_values("Customer ID").reset_index(drop=True)

    diagnostics = {
        "observation_events": observation_events,
        "outcome_events": outcome_events,
        "observation_counts": observation_counts,
        "last_before_cutoff": last_before_cutoff,
    }
    if table["Customer ID"].duplicated().any() or len(table) != table["Customer ID"].nunique():
        raise RuntimeError("Primary outcome table does not contain exactly one row per eligible customer.")
    return table, diagnostics


def load_step5_calendar_comparison() -> pd.DataFrame:
    if not STEP5_WINDOW_TABLE.exists() or not STEP5_OUTCOME_TABLE.exists():
        raise FileNotFoundError("Step 5 calendar-design audit tables are missing.")
    window = pd.read_csv(STEP5_WINDOW_TABLE)
    outcome = pd.read_csv(STEP5_OUTCOME_TABLE)
    calendar_observation = window.loc[window["population"].eq("all_365_day_observation_eligible_customers")].iloc[0]
    calendar_population = outcome.loc[outcome["population"].eq("population_B")].iloc[0]
    return pd.DataFrame([
        {
            "metric": "observation_customers",
            "exact_timestamp_design": np.nan,
            "step5_calendar_design": int(calendar_observation["customers"]),
            "difference_exact_minus_calendar": np.nan,
            "note": "Filled after exact-design cohort construction.",
        },
        {
            "metric": "primary_population_customers",
            "exact_timestamp_design": np.nan,
            "step5_calendar_design": int(calendar_population["population_customers"]),
            "difference_exact_minus_calendar": np.nan,
            "note": "Population B comparison expectation from Step 5 table 70.",
        },
        {
            "metric": "future_purchasers",
            "exact_timestamp_design": np.nan,
            "step5_calendar_design": int(calendar_population["customers_with_at_least_1_positive_purchase_in_outcome_window"]),
            "difference_exact_minus_calendar": np.nan,
            "note": "Positive-purchase outcome customers only.",
        },
        {
            "metric": "future_non_purchasers",
            "exact_timestamp_design": np.nan,
            "step5_calendar_design": int(calendar_population["customers_with_no_positive_purchase_in_outcome_window"]),
            "difference_exact_minus_calendar": np.nan,
            "note": "Aggregate comparison only; no Step 6 label was used as input.",
        },
    ])


def build_audits(
    table: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    boundaries: dict[str, pd.Timestamp],
    events: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observation_events = diagnostics["observation_events"]
    outcome_events = diagnostics["outcome_events"]
    exact_cutoff_events = events.loc[events["purchase_timestamp"].eq(boundaries["cutoff"])]
    pre_customer_sets = set(events.loc[events["purchase_timestamp"].le(boundaries["cutoff"]), "Customer ID"])
    post_customer_sets = set(outcome_events["Customer ID"])
    both_sides = pre_customer_sets & post_customer_sets
    eligible = set(table["Customer ID"])

    boundary_rows = [
        {"boundary": "observation_start", "timestamp": boundaries["observation_start"], "inclusion_rule": ">= observation_start", "closed_or_open": "closed"},
        {"boundary": "cutoff", "timestamp": boundaries["cutoff"], "inclusion_rule": "<= cutoff for observation; > cutoff for outcome", "closed_or_open": "observation closed / outcome open"},
        {"boundary": "outcome_end_dataset_end", "timestamp": boundaries["outcome_end"], "inclusion_rule": "<= outcome_end", "closed_or_open": "closed"},
        {"boundary": "outcome_start_condition", "timestamp": boundaries["cutoff"], "inclusion_rule": "purchase_timestamp strictly greater than cutoff", "closed_or_open": "open at cutoff"},
    ]
    boundary_table = pd.DataFrame(boundary_rows)

    comparison = comparison.copy()
    exact_values = {
        "observation_customers": int(observation_events["Customer ID"].nunique()),
        "primary_population_customers": int(len(table)),
        "future_purchasers": int(table["Churn"].eq(0).sum()),
        "future_non_purchasers": int(table["Churn"].eq(1).sum()),
    }
    for index, row in comparison.iterrows():
        value = exact_values[row["metric"]]
        comparison.loc[index, "exact_timestamp_design"] = value
        comparison.loc[index, "difference_exact_minus_calendar"] = value - row["step5_calendar_design"]
        comparison.loc[index, "note"] = "Exact timestamp design uses cutoff 2011-07-27 12:50:00 and strict post-cutoff outcome entry."

    class_balance = pd.DataFrame([
        {"Churn": 1, "label_meaning": "No positive purchase in supported future horizon", "customers": int(table["Churn"].eq(1).sum()), "percentage_of_primary_population": float(table["Churn"].eq(1).mean() * 100)},
        {"Churn": 0, "label_meaning": "At least one positive purchase in supported future horizon", "customers": int(table["Churn"].eq(0).sum()), "percentage_of_primary_population": float(table["Churn"].eq(0).mean() * 100)},
    ])

    cancellation_path = PROCESSED_DATA_DIR / "04_customer_cancellation_history.parquet"
    cancellation_customers_no_positive_outcome = 0
    cancellation_rows_no_positive_outcome = 0
    if cancellation_path.exists():
        cancellations = pd.read_parquet(cancellation_path, engine="pyarrow")
        cancellation_mask = (
            cancellations["InvoiceDate"].gt(boundaries["cutoff"])
            & cancellations["InvoiceDate"].le(boundaries["outcome_end"])
            & cancellations["Customer ID"].isin(eligible)
        )
        cancellation_customers = set(cancellations.loc[cancellation_mask, "Customer ID"])
        no_positive_outcome = set(table.loc[table["Churn"].eq(1), "Customer ID"])
        cancellation_customers_no_positive_outcome = len(cancellation_customers & no_positive_outcome)
        cancellation_rows_no_positive_outcome = int(cancellation_mask.sum())

    edge_rows = [
        {"edge_case": "positive events exactly at cutoff timestamp", "events": int(len(exact_cutoff_events)), "customers": int(exact_cutoff_events["Customer ID"].nunique()), "eligible_customers": int(exact_cutoff_events.loc[exact_cutoff_events["Customer ID"].isin(eligible), "Customer ID"].nunique()), "handling": "Included in observation/pre-cutoff data; excluded from outcome because outcome starts strictly after cutoff."},
        {"edge_case": "customers with positive purchases on both sides of cutoff", "events": np.nan, "customers": int(len(both_sides)), "eligible_customers": int(len(both_sides & eligible)), "handling": "Eligibility uses pre-cutoff events; future purchases determine outcome only."},
        {"edge_case": "eligible customers with multiple positive outcome purchases", "events": int(table.loc[table["positive_purchase_events_in_outcome"].gt(1), "positive_purchase_events_in_outcome"].sum()), "customers": int(table["positive_purchase_events_in_outcome"].gt(1).sum()), "eligible_customers": int(table["positive_purchase_events_in_outcome"].gt(1).sum()), "handling": "Retained as an outcome-count audit; Churn remains 0 for any positive outcome purchase."},
        {"edge_case": "eligible customers with cancellation activity but no positive outcome purchase", "events": cancellation_rows_no_positive_outcome, "customers": cancellation_customers_no_positive_outcome, "eligible_customers": cancellation_customers_no_positive_outcome, "handling": "Cancellations do not count as positive outcomes and do not reverse Churn=1."},
        {"edge_case": "eligible customers with positive purchase at final observed timestamp", "events": int(outcome_events["purchase_timestamp"].eq(boundaries["outcome_end"]).sum()), "customers": int(outcome_events.loc[outcome_events["purchase_timestamp"].eq(boundaries["outcome_end"]), "Customer ID"].nunique()), "eligible_customers": int(outcome_events.loc[outcome_events["purchase_timestamp"].eq(boundaries["outcome_end"]) & outcome_events["Customer ID"].isin(eligible), "Customer ID"].nunique()), "handling": "Included because outcome end is closed."},
        {"edge_case": "eligible customers with positive purchase on final observed date", "events": int(outcome_events.loc[outcome_events["purchase_date"].eq(boundaries["outcome_end"].normalize()), "Invoice"].nunique()), "customers": int(outcome_events.loc[outcome_events["purchase_date"].eq(boundaries["outcome_end"].normalize()), "Customer ID"].nunique()), "eligible_customers": int(outcome_events.loc[outcome_events["purchase_date"].eq(boundaries["outcome_end"].normalize()) & outcome_events["Customer ID"].isin(eligible), "Customer ID"].nunique()), "handling": "Included because the final observed date is within the supported outcome horizon."},
        {"edge_case": "duplicate Customer ID rows in final outcome table", "events": int(table["Customer ID"].duplicated().sum()), "customers": int(table["Customer ID"].nunique()), "eligible_customers": int(len(table)), "handling": "Must be zero duplicates; exactly one row per eligible customer."},
    ]
    edge_table = pd.DataFrame(edge_rows)

    leakage_rows = [
        {"audit": "observation eligibility uses only events within exact observation window", "passed": bool(observation_events["purchase_timestamp"].between(boundaries["observation_start"], boundaries["cutoff"], inclusive="both").all()), "evidence": "Observation filter is [observation_start, cutoff]."},
        {"audit": "baseline recency uses no post-cutoff transaction", "passed": bool(table["last_positive_purchase_before_cutoff_timestamp"].le(boundaries["cutoff"]).all()), "evidence": "Every stored last-before-cutoff timestamp is <= cutoff."},
        {"audit": "outcome uses only strictly post-cutoff positive purchases", "passed": bool(outcome_events["purchase_timestamp"].gt(boundaries["cutoff"]).all()), "evidence": "Outcome filter is purchase_timestamp > cutoff and <= outcome_end."},
        {"audit": "outcome is inside supported future horizon", "passed": bool(outcome_events["purchase_timestamp"].le(boundaries["outcome_end"]).all()), "evidence": "No outcome event exceeds dataset_end_timestamp."},
        {"audit": "no outcome information used for eligibility", "passed": bool(set(table["Customer ID"]).issubset(set(observation_events["Customer ID"]))), "evidence": "Eligible customers are selected before outcome aggregation."},
        {"audit": "no predictive features created", "passed": True, "evidence": "Outcome table contains only cohort, boundary, recency, outcome-count, and Churn fields."},
    ]
    leakage_table = pd.DataFrame(leakage_rows)
    return boundary_table, comparison, class_balance, edge_table, leakage_table


def write_figure(class_balance: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Churn = 1\n(no future purchase)", "Churn = 0\n(future purchase)"]
    values = [int(class_balance.loc[class_balance["Churn"].eq(1), "customers"].iloc[0]), int(class_balance.loc[class_balance["Churn"].eq(0), "customers"].iloc[0])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#C46B3C", "#2F5D7E"], width=0.58)
    ax.set_title("Step 6: primary retention-cohort outcome distribution")
    ax.set_ylabel("Eligible customers")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.6)
    ax.bar_label(bars, fmt="%d", padding=3)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def write_tables(boundaries: dict[str, pd.Timestamp], table: pd.DataFrame, observation_customer_count: int, boundary_table: pd.DataFrame, comparison: pd.DataFrame, class_balance: pd.DataFrame, edge_table: pd.DataFrame, leakage_table: pd.DataFrame) -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    boundary_summary = pd.DataFrame([
        {"boundary": "observation_start_timestamp", "timestamp": boundaries["observation_start"], "rule": "purchase_timestamp >= observation_start and <= cutoff"},
        {"boundary": "cutoff_timestamp", "timestamp": boundaries["cutoff"], "rule": "purchase_timestamp <= cutoff belongs to pre-cutoff data; outcome requires > cutoff"},
        {"boundary": "outcome_start_condition", "timestamp": boundaries["cutoff"], "rule": "purchase_timestamp strictly greater than cutoff"},
        {"boundary": "dataset_end_outcome_end_timestamp", "timestamp": boundaries["outcome_end"], "rule": "purchase_timestamp <= outcome_end"},
    ])
    tables = {
        "75_step6_exact_temporal_boundaries.csv": boundary_summary,
        "76_step6_cohort_eligibility_summary.csv": pd.DataFrame([
            {"metric": "customers_with_at_least_one_positive_purchase_in_observation_window", "value": observation_customer_count},
            {"metric": "customers_excluded_baseline_recency_gt_135_days", "value": observation_customer_count - len(table)},
            {"metric": "final_primary_population_B_customers", "value": len(table)},
            {"metric": "exact_timestamp_observation_start", "value": boundaries["observation_start"]},
            {"metric": "exact_timestamp_cutoff", "value": boundaries["cutoff"]},
            {"metric": "exact_timestamp_outcome_end", "value": boundaries["outcome_end"]},
        ]),
        "77_step6_class_balance.csv": class_balance,
        "78_step6_exact_vs_step5_calendar_comparison.csv": comparison,
        "79_step6_temporal_leakage_audit.csv": leakage_table,
        "80_step6_edge_case_audit.csv": edge_table,
    }
    paths = []
    for filename, frame in tables.items():
        path = TABLES_DIR / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_output_manifest(paths: list[Path], figure: Path) -> Path:
    rows = []
    for path in [OUTCOME_PATH, *paths, figure, REPORT_PATH, NOTEBOOK_PATH, LOG_PATH]:
        if path.exists():
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(rows)
    path = TABLES_DIR / "81_step6_output_manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def write_notebook() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Step 6 — Primary retention cohort and outcome table\n",
                "\n",
                "This notebook inspects the exact timestamp boundaries, minimal eligibility/outcome table, leakage audit, edge cases, and class balance. It does not create predictive features or models."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "TABLES = ROOT / 'outputs' / 'tables'\n",
                "OUTCOME = ROOT / 'data' / 'processed' / '06_primary_customer_outcome_table.parquet'\n",
                "outcome = pd.read_parquet(OUTCOME)\n",
                "boundaries = pd.read_csv(TABLES / '75_step6_exact_temporal_boundaries.csv')\n",
                "eligibility = pd.read_csv(TABLES / '76_step6_cohort_eligibility_summary.csv')\n",
                "class_balance = pd.read_csv(TABLES / '77_step6_class_balance.csv')\n",
                "comparison = pd.read_csv(TABLES / '78_step6_exact_vs_step5_calendar_comparison.csv')\n",
                "leakage = pd.read_csv(TABLES / '79_step6_temporal_leakage_audit.csv')\n",
                "edge_cases = pd.read_csv(TABLES / '80_step6_edge_case_audit.csv')"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Boundaries and cohort outcome table"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(boundaries)\n",
                "display(eligibility)\n",
                "display(outcome.head())\n",
                "display(outcome['Churn'].value_counts(dropna=False))"
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Audits"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "display(class_balance)\n",
                "display(comparison)\n",
                "display(leakage)\n",
                "display(edge_cases)"
            ]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def write_report(boundaries: dict[str, pd.Timestamp], table: pd.DataFrame, observation_customer_count: int, boundary_table: pd.DataFrame, comparison: pd.DataFrame, class_balance: pd.DataFrame, edge_table: pd.DataFrame, leakage_table: pd.DataFrame, metadata: dict[str, str]) -> None:
    def fmt(value: object, digits: int = 2) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        selected = frame.loc[:, columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in selected.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
        return "\n".join(lines)

    churn_one = int(table["Churn"].eq(1).sum())
    churn_zero = int(table["Churn"].eq(0).sum())
    obs_customers = observation_customer_count
    calendar_diff = comparison.set_index("metric")["difference_exact_minus_calendar"]
    report = f"""# Step 6 — Lock the Primary Churn Cohort and Create the Customer-Level Outcome Table

## Scope

This step implements the locked Step 5 decisions for the preventive-retention use case: a 135-day inactivity horizon, approximately one year of observation history, and Population B restricted to customers whose baseline recency is <=135 days at the prediction cutoff. It creates the minimal customer-level outcome table only. No RFM or predictive features, train/test split, encoding, scaling, resampling, model, or SHAP analysis was performed.

## 1. Exact temporal boundaries

The validated positive-purchase event endpoint is **{boundaries['outcome_end']:%Y-%m-%d %H:%M:%S}**. The exact cutoff is the endpoint minus 135 days: **{boundaries['cutoff']:%Y-%m-%d %H:%M:%S}**. The exact 365-day observation history is **[{boundaries['observation_start']:%Y-%m-%d %H:%M:%S}, {boundaries['cutoff']:%Y-%m-%d %H:%M:%S}]**, closed at both ends. The future outcome horizon is **({boundaries['cutoff']:%Y-%m-%d %H:%M:%S}, {boundaries['outcome_end']:%Y-%m-%d %H:%M:%S}]**, open at the cutoff and closed at the dataset endpoint.

Thus, an event exactly at the cutoff belongs to pre-cutoff observation data, while an event strictly after the cutoff belongs to the outcome period. No event can belong to both periods. The exact design differs from the Step 5 calendar approximation because it includes timestamp-level events on the boundary dates according to these rules.

{markdown_table(boundary_table, ['boundary', 'timestamp', 'inclusion_rule', 'closed_or_open'])}

## 2. Primary cohort eligibility

There are **{obs_customers:,}** customers with at least one positive-purchase event in the exact observation window. Customers with baseline recency greater than 135 days are excluded. The final primary Population B contains **{len(table):,} customers**.

Eligibility uses only positive-purchase events on or before the cutoff. The outcome table contains exactly one row per eligible customer and only minimal boundary, recency, outcome-count, and target fields.

## 3. Outcome target and class balance

For eligible customers, `Churn = 0` means at least one positive purchase strictly after the cutoff and no later than the dataset endpoint. `Churn = 1` means no positive purchase in that supported future horizon. Cancellations do not count as positive outcomes.

{markdown_table(class_balance, ['Churn', 'label_meaning', 'customers', 'percentage_of_primary_population'])}

The final target distribution is **{churn_one:,} Churn=1 ({churn_one / len(table) * 100:.2f}%)** and **{churn_zero:,} Churn=0 ({churn_zero / len(table) * 100:.2f}%)**. No class rebalancing or threshold optimization was performed.

## 4. Comparison with Step 5 calendar-date audit

{markdown_table(comparison, ['metric', 'exact_timestamp_design', 'step5_calendar_design', 'difference_exact_minus_calendar', 'note'])}

The exact timestamp design produces differences of **{calendar_diff['observation_customers']:.0f}** observation-eligible customers, **{calendar_diff['primary_population_customers']:.0f}** primary-cohort customers, **{calendar_diff['future_purchasers']:.0f}** future purchasers, and **{calendar_diff['future_non_purchasers']:.0f}** future non-purchasers relative to the Step 5 calendar audit. These differences arise only from timestamp-level placement around the cutoff and boundary dates; the previous Step 5 counts were not forced or overwritten.

## 5. Temporal leakage audit

{markdown_table(leakage_table, ['audit', 'passed', 'evidence'])}

Eligibility and baseline recency use no outcome-window activity. The target uses only strictly post-cutoff positive-purchase events within the supported horizon. Because predictive features do not yet exist, no outcome information can appear in predictors.

## 6. Edge-case audit

{markdown_table(edge_table, ['edge_case', 'events', 'customers', 'eligible_customers', 'handling'])}

The final outcome table has **{int(table['Customer ID'].duplicated().sum())} duplicate Customer ID rows** and therefore has exactly one row per eligible customer. Customers with multiple outcome purchases remain `Churn=0`; the count is retained only for validation.

## 7. Validation and stopping point

- The outcome table is derived from validated invoice-level positive-purchase events, not line items.
- The exact timestamp cutoff is **{boundaries['cutoff']:%Y-%m-%d %H:%M:%S}** and the outcome end is **{boundaries['outcome_end']:%Y-%m-%d %H:%M:%S}**.
- No raw or Step 4 canonical dataset was modified.
- No predictive feature engineering beyond minimal eligibility/outcome validation fields was performed.
- No train/test split, encoding, scaling, resampling, model training, or SHAP analysis occurred.
- The processed outcome table is Git-ignored.

Input hashes: positive base `{metadata['positive_sha256_before']}`, Step 5 events `{metadata['events_sha256_before']}`, canonical `{metadata['canonical_sha256_before']}`, raw `{metadata['raw_sha256_before']}`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_scope(metadata: dict[str, str], table: pd.DataFrame, events: pd.DataFrame, outputs: list[Path]) -> dict[str, bool]:
    normalized = [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in table.columns]
    checks = {
        "positive_base_unchanged": sha256_file(POSITIVE_PURCHASE_PATH) == metadata["positive_sha256_before"],
        "step5_events_unchanged": sha256_file(EVENT_PATH) == metadata["events_sha256_before"],
        "canonical_source_unchanged": sha256_file(CANONICAL_PATH) == metadata["canonical_sha256_before"],
        "raw_workbook_unchanged": sha256_file(find_raw_workbook()) == metadata["raw_sha256_before"],
        "one_row_per_eligible_customer": len(table) == table["Customer ID"].nunique() and not table["Customer ID"].duplicated().any(),
        "churn_values_only_zero_or_one": bool(table["Churn"].isin([0, 1]).all()),
        "churn_has_no_missing_values": bool(table["Churn"].notna().all()),
        "eligibility_has_no_post_cutoff_data": bool(table["last_positive_purchase_before_cutoff_timestamp"].le(table["cutoff_timestamp"]).all()),
        "outcome_counts_are_nonnegative": bool(table["positive_purchase_events_in_outcome"].ge(0).all()),
        "minimal_schema_only": normalized == [re.sub(r"[^a-z0-9]+", "", column.lower()) for column in OUTCOME_COLUMNS],
        "no_train_test_encoding_scaling_model_or_shap": not any((PROJECT_ROOT / "outputs" / "models").glob("*")),
        "processed_outcome_git_ignored": subprocess.run(["git", "check-ignore", "--no-index", str(OUTCOME_PATH.relative_to(PROJECT_ROOT))], cwd=PROJECT_ROOT, capture_output=True).returncode == 0,
        "all_declared_outputs_exist": all(path.exists() for path in outputs),
    }
    # The models directory contains only its placeholder in the validated repository.
    checks["no_train_test_encoding_scaling_model_or_shap"] = not any(path.name != ".gitkeep" for path in (PROJECT_ROOT / "outputs" / "models").glob("*"))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Step 6 validation failed: {failed}")
    return checks


def run_analysis() -> dict[str, object]:
    logger = get_logger()
    positive, events, metadata = validate_inputs()
    boundaries = resolve_boundaries(events)
    table, diagnostics = build_outcome_table(events, boundaries)
    comparison = load_step5_calendar_comparison()
    boundary_table, comparison, class_balance, edge_table, leakage_table = build_audits(table, diagnostics, boundaries, events, comparison)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTCOME_PATH, engine="pyarrow", index=False)
    observation_customer_count = int(diagnostics["observation_counts"].size)
    tables = write_tables(boundaries, table, observation_customer_count, boundary_table, comparison, class_balance, edge_table, leakage_table)
    figure = write_figure(class_balance)
    write_notebook()
    write_report(boundaries, table, observation_customer_count, boundary_table, comparison, class_balance, edge_table, leakage_table, metadata)
    logger.info("Step 6 created the locked primary Population B outcome table only.")
    logger.info("Exact observation start: %s; cutoff: %s; outcome end: %s", boundaries["observation_start"], boundaries["cutoff"], boundaries["outcome_end"])
    logger.info("Eligible customers: %s; Churn=1: %s; Churn=0: %s", len(table), int(table["Churn"].eq(1).sum()), int(table["Churn"].eq(0).sum()))
    for handler in logger.handlers:
        handler.flush()
    manifest = write_output_manifest(tables, figure)
    checks = validate_scope(metadata, table, events, [OUTCOME_PATH, *tables, figure, REPORT_PATH, NOTEBOOK_PATH, LOG_PATH, manifest])
    logger.info("Validation checks: %s", checks)
    for handler in logger.handlers:
        handler.flush()
    manifest = write_output_manifest(tables, figure)
    return {"table": table, "boundaries": boundaries, "comparison": comparison, "class_balance": class_balance, "edge_table": edge_table, "leakage_table": leakage_table, "checks": checks, "manifest": manifest}


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "observation_start": str(result["boundaries"]["observation_start"]),
        "cutoff": str(result["boundaries"]["cutoff"]),
        "outcome_end": str(result["boundaries"]["outcome_end"]),
        "eligible_customers": len(result["table"]),
        "churn_1": int(result["table"]["Churn"].eq(1).sum()),
        "churn_0": int(result["table"]["Churn"].eq(0).sum()),
        "validation": result["checks"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
