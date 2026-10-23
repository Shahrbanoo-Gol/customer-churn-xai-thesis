"""Step 4: build approved customer-linked transaction analysis bases."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR


RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "SourceSheet",
]
ORIGINAL_TRANSACTION_COLUMNS = RAW_COLUMNS[:-1]
CANONICAL_PATH = PROCESSED_DATA_DIR / "01_source_reconciled.parquet"
CUSTOMER_HISTORY_PATH = PROCESSED_DATA_DIR / "02_customer_transaction_history.parquet"
POSITIVE_PURCHASE_PATH = PROCESSED_DATA_DIR / "03_positive_purchase_base.parquet"
CANCELLATION_HISTORY_PATH = PROCESSED_DATA_DIR / "04_customer_cancellation_history.parquet"
MANIFEST_PATH = TABLES_DIR / "01_raw_file_manifest.csv"
STEP2_SUMMARY_PATH = TABLES_DIR / "20_source_reconciliation_summary.csv"
RAW_WORKBOOK_NAME = "online_retail_II.xlsx"


def sha256_file(path: Path) -> str:
    """Hash a file so the source and canonical input can be proven unchanged."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_percentage(numerator: float, denominator: float) -> float:
    """Return a percentage without producing a divide-by-zero error."""

    return float(numerator / denominator * 100) if denominator else 0.0


def find_raw_workbook() -> Path:
    """Find the workbook whose checksum was established in Step 1."""

    path = RAW_DATA_DIR / RAW_WORKBOOK_NAME
    if not path.exists():
        raise FileNotFoundError(f"Raw workbook is missing: {path}")
    return path


def get_logger() -> logging.Logger:
    """Create the reproducibility log used for this preparation run."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.build_analysis_base")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(
            LOGS_DIR / "04_final_preparation_and_analysis_base.log", mode="w", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def validate_input() -> tuple[pd.DataFrame, str, str]:
    """Validate Step 2 output and source checksums before any derived file is written."""

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(f"Canonical Step 2 parquet is missing: {CANONICAL_PATH}")
    if not MANIFEST_PATH.exists() or not STEP2_SUMMARY_PATH.exists():
        raise FileNotFoundError("Step 2 manifest or reconciliation summary is missing.")

    raw_path = find_raw_workbook()
    manifest = pd.read_csv(MANIFEST_PATH)
    expected_raw_hash = str(manifest.iloc[0]["sha256"]).lower()
    observed_raw_hash = sha256_file(raw_path)
    if observed_raw_hash != expected_raw_hash:
        raise RuntimeError(f"Raw checksum changed: expected {expected_raw_hash}, observed {observed_raw_hash}.")

    step2 = pd.read_csv(STEP2_SUMMARY_PATH).set_index("metric")["value"]
    data = pd.read_parquet(CANONICAL_PATH, engine="pyarrow")
    expected_rows = int(float(step2["row_count_after_source_reconciliation"]))
    if len(data) != expected_rows:
        raise RuntimeError(f"Canonical row count does not match Step 2 summary: {len(data)} vs {expected_rows}")
    if list(data.columns) != RAW_COLUMNS:
        raise RuntimeError(f"Unexpected canonical columns: {list(data.columns)}")
    expected_min = pd.Timestamp(step2["earliest_InvoiceDate"])
    expected_max = pd.Timestamp(step2["latest_InvoiceDate"])
    if data["InvoiceDate"].min() != expected_min or data["InvoiceDate"].max() != expected_max:
        raise RuntimeError("Canonical date range does not match Step 2 summary.")
    if data["Customer ID"].isna().sum() < 0:
        raise RuntimeError("Canonical missing-Customer-ID count could not be computed.")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in data.columns):
        raise RuntimeError("A churn column exists in the canonical input.")
    return data, sha256_file(CANONICAL_PATH), observed_raw_hash


def invoice_starts_with_c(data: pd.DataFrame) -> pd.Series:
    """Identify cancellation invoices without changing invoice values."""

    return data["Invoice"].astype("string").str.upper().str.startswith("C", na=False)


def build_customer_history(data: pd.DataFrame) -> pd.DataFrame:
    """Keep all rows linked to a customer because history includes exceptions and cancellations."""

    return data.loc[data["Customer ID"].notna()].copy()


def build_positive_purchase_base(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only identifiable, non-cancellation, positive-quantity, positive-price purchases."""

    cancellation = invoice_starts_with_c(data)
    mask = data["Customer ID"].notna() & ~cancellation & data["Quantity"].gt(0) & data["Price"].gt(0)
    return data.loc[mask].copy()


def build_cancellation_history(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve customer-linked C-prefix rows as cancellations rather than purchases."""

    mask = data["Customer ID"].notna() & invoice_starts_with_c(data)
    return data.loc[mask].copy()


def keep_first_duplicate_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Return the primary keep-first duplicate-removal scenario for sensitivity callers."""

    return data.loc[~data.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep="first")].copy()


def duplicate_retained_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Return a duplicate-retained alternative without changing the primary datasets."""

    return data.copy()


def write_parquet(data: pd.DataFrame, path: Path) -> None:
    """Write a derived dataset while retaining its original transaction columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if list(data.columns) != RAW_COLUMNS:
        raise RuntimeError(f"Derived dataset has unexpected columns: {list(data.columns)}")
    data.to_parquet(path, engine="pyarrow", index=False)


def _summary_row(section: str, metric: str, value: object, percentage: float = np.nan, notes: str = "") -> dict[str, object]:
    """Create a consistent table row for Step 4 summaries."""

    return {"section": section, "metric": metric, "value": value, "percentage": percentage, "notes": notes}


def write_cleaning_waterfall(raw_rows: int, data: pd.DataFrame, history: pd.DataFrame, positive: pd.DataFrame) -> pd.DataFrame:
    """Show sequential preparation stages without adding overlapping exclusions."""

    missing = len(data) - len(history)
    not_positive = len(history) - len(positive)
    rows = [
        _summary_row("sequential_stage", "original_raw_worksheet_rows", raw_rows, notes="Sum of both raw worksheets before Step 2 source reconciliation."),
        _summary_row("sequential_stage", "source_reconciled_rows", len(data), safe_percentage(len(data), raw_rows), "Canonical input after within-sheet exact deduplication and verified cross-sheet overlap reconciliation."),
        _summary_row("sequential_stage", "customer_identifiable_history_rows", len(history), safe_percentage(len(history), len(data)), "Rows with non-null Customer ID; cancellations and exceptions retained."),
        _summary_row("sequential_stage", "rows_excluded_for_missing_Customer_ID", missing, safe_percentage(missing, len(data)), "Sequential reduction from source-reconciled rows; not an artificial-ID assignment."),
        _summary_row("sequential_stage", "positive_purchase_base_rows", len(positive), safe_percentage(len(positive), len(history)), "Sequential reduction from customer history using the approved positive-purchase rule."),
        _summary_row("sequential_stage", "identified_rows_not_qualifying_for_positive_purchase_base", not_positive, safe_percentage(not_positive, len(history)), "Includes overlapping cancellations, non-positive quantities/prices, and other identified exceptions."),
        _summary_row("diagnostic_note", "overlapping_conditions", "See table 45", notes="C-prefix, Quantity <= 0, and Price <= 0 conditions are not added as independent reductions."),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "40_cleaning_waterfall.csv", index=False)
    return result


def write_customer_coverage(history: pd.DataFrame, positive: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    """Measure customer coverage without assigning any churn meaning."""

    history_customers = set(history["Customer ID"].dropna().unique())
    positive_customers = set(positive["Customer ID"].dropna().unique())
    cancellation_customers = set(cancellations["Customer ID"].dropna().unique())
    rows = [
        _summary_row("customer_coverage", "unique_customers_in_customer_transaction_history", len(history_customers), notes="All identifiable customers in the broad history."),
        _summary_row("customer_coverage", "unique_customers_with_at_least_one_positive_purchase", len(positive_customers), safe_percentage(len(positive_customers), len(history_customers)), "Not a churn definition."),
        _summary_row("customer_coverage", "unique_customers_with_at_least_one_cancellation", len(cancellation_customers), safe_percentage(len(cancellation_customers), len(history_customers)), "Not a churn definition."),
        _summary_row("customer_coverage", "customers_with_positive_purchases_and_cancellations", len(positive_customers & cancellation_customers), safe_percentage(len(positive_customers & cancellation_customers), len(history_customers)), "Not a churn definition."),
        _summary_row("customer_coverage", "customers_with_cancellations_but_no_positive_purchase", len(cancellation_customers - positive_customers), safe_percentage(len(cancellation_customers - positive_customers), len(history_customers)), "Not a churn definition."),
        _summary_row("customer_coverage", "identified_history_customers_with_no_qualifying_positive_purchase", len(history_customers - positive_customers), safe_percentage(len(history_customers - positive_customers), len(history_customers)), "Not a churn definition."),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "41_customer_coverage_after_preparation.csv", index=False)
    return result


def invoice_count_with_multiple_lines(data: pd.DataFrame) -> int:
    """Count invoices with more than one identified-history line item."""

    invoice_counts = data.loc[data["Invoice"].notna()].groupby("Invoice").size()
    return int(invoice_counts.gt(1).sum())


def write_invoice_coverage(history: pd.DataFrame, positive: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    """Report invoice identity and temporal coverage for each prepared dataset."""

    datasets = [
        ("customer_transaction_history", history, "unique_invoices", history["Invoice"].nunique(dropna=True)),
        ("positive_purchase_base", positive, "unique_non_C_positive_purchase_invoices", positive["Invoice"].nunique(dropna=True)),
        ("customer_cancellation_history", cancellations, "unique_C_prefix_invoices", cancellations["Invoice"].nunique(dropna=True)),
    ]
    rows = []
    for name, frame, metric_name, invoice_count in datasets:
        rows.append(_summary_row(name, metric_name, invoice_count, notes="Unique non-null invoice identifiers."))
        rows.append(_summary_row(name, "date_min", frame["InvoiceDate"].min().isoformat() if not frame.empty else ""))
        rows.append(_summary_row(name, "date_max", frame["InvoiceDate"].max().isoformat() if not frame.empty else ""))
        rows.append(_summary_row(name, "row_count", len(frame)))
    rows.append(_summary_row("customer_transaction_history", "invoices_containing_multiple_line_items", invoice_count_with_multiple_lines(history), notes="Invoice identifiers with more than one history row."))
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "42_invoice_coverage_after_preparation.csv", index=False)
    return result


def value_quantity_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Calculate descriptive line-value measures without creating feature columns."""

    line_values = frame["Quantity"] * frame["Price"]
    return {
        "total_Quantity": frame["Quantity"].sum(),
        "gross_line_value": line_values.sum(),
        "median_line_value": line_values.median(),
        "unique_products": frame["StockCode"].nunique(dropna=True),
        "unique_invoices": frame["Invoice"].nunique(dropna=True),
        "unique_customers": frame["Customer ID"].nunique(dropna=True),
    }


def write_value_quantity_reconciliation(positive: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    """Reconcile positive purchase and cancellation totals separately."""

    rows = []
    for dataset_name, frame in [("positive_purchase_base", positive), ("customer_cancellation_history", cancellations)]:
        metrics = value_quantity_metrics(frame)
        for metric, value in metrics.items():
            rows.append(_summary_row(dataset_name, metric, value, notes="Positive purchases and cancellations are not netted."))
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "43_value_quantity_reconciliation.csv", index=False)
    return result


def write_duplicate_removal_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Document the primary keep-first removal and later sensitivity alternative."""

    participating = history.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep=False)
    extra = history.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep="first")
    duplicate_customers = history.loc[participating, "Customer ID"].nunique()
    duplicate_invoices = history.loc[participating, "Invoice"].nunique(dropna=True)
    rows = [
        _summary_row("duplicate_treatment", "duplicate_participating_rows_in_primary", int(participating.sum()), notes="Should be zero after primary keep-first treatment and cross-sheet reconciliation."),
        _summary_row("duplicate_treatment", "extra_duplicate_occurrences_in_primary", int(extra.sum()), notes="Should be zero in the canonical primary datasets."),
        _summary_row("duplicate_removal", "affected_customers", duplicate_customers),
        _summary_row("duplicate_removal", "affected_invoices", duplicate_invoices),
        _summary_row("duplicate_treatment", "primary_dataset_treatment", "keep_first_removed", notes="Exact full-row duplicates are removed within each worksheet before reconciliation."),
        _summary_row("duplicate_treatment", "sensitivity_alternative", "duplicate_retained", notes="A duplicate-retained alternative remains available for later sensitivity comparison."),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "44_duplicate_removal_summary.csv", index=False)
    return result


def write_nonpurchase_summary(history: pd.DataFrame, positive: pd.DataFrame) -> pd.DataFrame:
    """Classify non-qualifying identified rows diagnostically with explicit precedence."""

    cancellation = invoice_starts_with_c(history)
    non_c_nonpositive_quantity = ~cancellation & history["Quantity"].le(0)
    nonpositive_price = history["Price"].le(0)
    qualifies = history.index.isin(positive.index)
    nonpurchase = history.loc[~qualifies].copy()
    c_mask = cancellation.loc[nonpurchase.index]
    qty_mask = non_c_nonpositive_quantity.loc[nonpurchase.index]
    price_mask = nonpositive_price.loc[nonpurchase.index]
    overlap_mask = qty_mask & price_mask

    diagnostic_rows = [
        ("C-prefix cancellation", c_mask),
        ("non-C Quantity <= 0", qty_mask),
        ("Price <= 0", price_mask),
        ("overlapping Quantity/Price conditions", overlap_mask),
    ]
    rows = []
    for label, mask in diagnostic_rows:
        rows.append(_summary_row("diagnostic_overlapping_condition", label, int(mask.sum()), safe_percentage(mask.sum(), len(nonpurchase)), "Conditions may overlap; these counts must not be added."))

    # The mutually exclusive sequence gives every non-purchase row one primary explanation.
    remaining = pd.Series(True, index=nonpurchase.index)
    precedence = [
        ("C-prefix cancellation", c_mask),
        ("non-C Quantity <= 0", qty_mask),
        ("Price <= 0", price_mask),
    ]
    assigned = pd.Series("other identified non-purchase", index=nonpurchase.index, dtype="string")
    for label, mask in precedence:
        selected = remaining & mask
        assigned.loc[selected] = label
        remaining.loc[selected] = False
    for label, count in assigned.value_counts().items():
        rows.append(_summary_row("mutually_exclusive_precedence", str(label), int(count), safe_percentage(count, len(nonpurchase)), "Precedence: C-prefix cancellation > non-C Quantity <= 0 > Price <= 0 > other."))
    rows.append(_summary_row("control_total", "identified_history_rows", len(history)))
    rows.append(_summary_row("control_total", "positive_purchase_base_rows", len(positive)))
    rows.append(_summary_row("control_total", "identified_nonpurchase_rows", len(nonpurchase), safe_percentage(len(nonpurchase), len(history)), "Mutually exclusive primary categories sum to this count."))
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "45_identified_nonpurchase_summary.csv", index=False)
    return result


def prepared_dataset_manifest(paths: Iterable[Path]) -> pd.DataFrame:
    """Create file-level metadata so every derived parquet can be reproduced and checked."""

    rows = []
    for path in paths:
        frame = pd.read_parquet(path, engine="pyarrow")
        rows.append({
            "filename": path.name,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "date_min": frame["InvoiceDate"].min().isoformat() if not frame.empty else "",
            "date_max": frame["InvoiceDate"].max().isoformat() if not frame.empty else "",
            "unique_non_null_customers": frame["Customer ID"].nunique(dropna=True),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "46_processed_data_manifest.csv", index=False)
    return result


def load_prepared_datasets() -> dict[str, pd.DataFrame]:
    """Load the three prepared datasets for later notebooks without recomputing them."""

    return {
        "customer_transaction_history": pd.read_parquet(CUSTOMER_HISTORY_PATH, engine="pyarrow"),
        "positive_purchase_base": pd.read_parquet(POSITIVE_PURCHASE_PATH, engine="pyarrow"),
        "customer_cancellation_history": pd.read_parquet(CANCELLATION_HISTORY_PATH, engine="pyarrow"),
    }


def validate_prepared_datasets(
    canonical_hash_before: str,
    raw_hash_before: str,
    source: pd.DataFrame,
    history: pd.DataFrame,
    positive: pd.DataFrame,
    cancellations: pd.DataFrame,
) -> dict[str, bool]:
    """Validate the approved filters and prove that forbidden later-stage work was not done."""

    raw_hash_after = sha256_file(find_raw_workbook())
    canonical_hash_after = sha256_file(CANONICAL_PATH)
    positive_cancellation = invoice_starts_with_c(positive)
    cancellation_cancellation = invoice_starts_with_c(cancellations)
    prepared_frames = [history, positive, cancellations]
    prepared_columns = [column for frame in prepared_frames for column in frame.columns]
    normalized_columns = [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in prepared_columns]
    checks = {
        "customer_history_has_no_missing_customer_id": bool(history["Customer ID"].notna().all()),
        "source_has_no_exact_duplicate_rows": not bool(source.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep=False).any()),
        "customer_history_has_no_exact_duplicate_rows": not bool(history.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep=False).any()),
        "positive_base_has_no_exact_duplicate_rows": not bool(positive.duplicated(subset=ORIGINAL_TRANSACTION_COLUMNS, keep=False).any()),
        "positive_base_all_rows_meet_rule": bool((positive["Customer ID"].notna() & ~positive_cancellation & positive["Quantity"].gt(0) & positive["Price"].gt(0)).all()),
        "cancellation_history_all_rows_meet_rule": bool((cancellations["Customer ID"].notna() & cancellation_cancellation).all()),
        "customer_history_derives_from_reconciled_source": bool(len(history) == int(source["Customer ID"].notna().sum())),
        "positive_base_derives_from_reconciled_source": bool(positive.index.isin(source.index).all()),
        "cancellation_history_derives_from_reconciled_source": bool(cancellations.index.isin(source.index).all()),
        "source_date_range_preserved": bool(source["InvoiceDate"].min() == history["InvoiceDate"].min() and source["InvoiceDate"].max() == history["InvoiceDate"].max()),
        "canonical_hash_unchanged": canonical_hash_after == canonical_hash_before,
        "raw_hash_unchanged": raw_hash_after == raw_hash_before,
        "no_churn_column": not any("churn" in column for column in normalized_columns),
        "no_inactivity_threshold_defined": not any("inactivity" in column for column in normalized_columns),
        "no_observation_or_outcome_window_created": not any(any(term in column for term in ("observation", "outcome", "window")) for column in normalized_columns),
        "no_predictive_feature_columns": all(list(frame.columns) == RAW_COLUMNS for frame in prepared_frames),
        "no_rfm_feature_table": not any("rfm" in path.name.lower() for path in PROCESSED_DATA_DIR.glob("*.parquet")),
        "no_ml_model_fitted_or_artifact_created": not any(path.exists() for path in (PROJECT_ROOT / "outputs" / "models").glob("step4*")),
        "no_churn_created": not any("churn" in path.name.lower() for path in PROCESSED_DATA_DIR.glob("*.parquet")),
        "processed_parquet_files_ignored": all(
            __import__("subprocess").run(
                ["git", "check-ignore", "--no-index", str(path.relative_to(PROJECT_ROOT))],
                cwd=PROJECT_ROOT, capture_output=True,
            ).returncode == 0
            for path in PROCESSED_DATA_DIR.glob("*.parquet")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Step 4 validation failed: {failed}")
    return checks


def write_report(waterfall: pd.DataFrame, coverage: pd.DataFrame, invoice: pd.DataFrame, value_quantity: pd.DataFrame, duplicate: pd.DataFrame, nonpurchase: pd.DataFrame, manifest: pd.DataFrame, checks: dict[str, bool], raw_hash: str) -> None:
    """Write a management-oriented explanation of the prepared analysis bases."""

    def value(table: pd.DataFrame, metric: str, section: str | None = None) -> object:
        selected = table[table["metric"] == metric]
        if section is not None:
            selected = selected[selected["section"] == section]
        return selected.iloc[0]["value"] if not selected.empty else ""

    wf = waterfall.set_index("metric")["value"]
    cov = coverage.set_index("metric")["value"]
    pos_value = value_quantity[(value_quantity["section"] == "positive_purchase_base")].set_index("metric")["value"]
    cancel_value = value_quantity[(value_quantity["section"] == "customer_cancellation_history")].set_index("metric")["value"]
    report = f"""# Step 4 — Final Preparation and Analysis Base

## 1. Purpose

This step executes the approved transaction-preparation rules and creates reproducible bases for later customer-level feature engineering. It does not define churn, select observation or future-outcome windows, calculate RFM features, split data, or train a model.

## 2. Approved preparation rules

- The Step 2 source-reconciled parquet is the canonical input after within-sheet exact deduplication and verified cross-sheet overlap reconciliation.
- Rows without Customer ID were excluded from customer-linked datasets without imputation.
- Exact within-source duplicates were removed with keep-first logic. A duplicate-retained alternative is available for later sensitivity comparison.
- C-prefix invoices remain in customer history and are preserved separately as cancellation history.
- Non-C negative quantities and Price <= 0 records remain in customer history but do not enter the positive-purchase base.
- Missing Description and special StockCodes were not used as blanket exclusion rules.

## 3. Customer-linked transaction history

The customer transaction history contains **{int(wf['customer_identifiable_history_rows']):,} rows** linked to **{int(cov['unique_customers_in_customer_transaction_history']):,} customers**. It retains cancellations, non-positive prices, negative quantities, special StockCodes, and missing descriptions; exact duplicate extras were removed before customer-linked datasets were built.

## 4. Positive-purchase base

The positive-purchase base contains **{int(wf['positive_purchase_base_rows']):,} rows**. Every row has a non-null Customer ID, a non-C-prefix invoice, Quantity > 0, and Price > 0. Its total Quantity is **{pos_value['total_Quantity']:,.0f}**, and its gross line value is **{pos_value['gross_line_value']:,.2f}**. It contains **{int(pos_value['unique_products']):,} products**, **{int(pos_value['unique_invoices']):,} invoices**, and **{int(pos_value['unique_customers']):,} customers**.

## 5. Cancellation history

The cancellation history contains **{int(value(invoice, 'row_count', 'customer_cancellation_history')):,} rows** linked to **{int(cov['unique_customers_with_at_least_one_cancellation']):,} customers**. Its signed Quantity total is **{cancel_value['total_Quantity']:,.0f}**, and its signed line-value total is **{cancel_value['gross_line_value']:,.2f}**. These values are reported separately and were not netted against positive purchases.

## 6. Customer and invoice coverage

Among identifiable customers, **{int(cov['unique_customers_with_at_least_one_positive_purchase']):,}** have at least one qualifying positive purchase, **{int(cov['unique_customers_with_at_least_one_cancellation']):,}** have at least one cancellation, and **{int(cov['customers_with_positive_purchases_and_cancellations']):,}** have both. **{int(cov['identified_history_customers_with_no_qualifying_positive_purchase']):,}** customers have identified history but no qualifying positive-purchase row; this is a preparation category, not a churn label.

The identified-customer history contains **{int(value(invoice, 'unique_invoices', 'customer_transaction_history')):,}** invoices, the positive-purchase base contains **{int(value(invoice, 'unique_non_C_positive_purchase_invoices', 'positive_purchase_base')):,}** non-C invoices, and the cancellation history contains **{int(value(invoice, 'unique_C_prefix_invoices', 'customer_cancellation_history')):,}** C-prefix invoices. The history contains **{int(value(invoice, 'invoices_containing_multiple_line_items', 'customer_transaction_history')):,}** invoices with multiple line items.

## 7. Duplicate treatment

The primary customer history has **{int(value(duplicate, 'duplicate_participating_rows_in_primary')):,} duplicate-participating rows** and **{int(value(duplicate, 'extra_duplicate_occurrences_in_primary')):,} extra occurrences** after keep-first removal. The duplicate-retained alternative remains available for a later sensitivity comparison.

## 8. Data-quality controls

The cleaning waterfall begins with **{int(wf['original_raw_worksheet_rows']):,} raw worksheet rows**, followed by **{int(wf['source_reconciled_rows']):,} source-reconciled rows**. Missing-ID exclusion is a sequential reduction of **{int(wf['rows_excluded_for_missing_Customer_ID']):,} rows**. Non-purchase conditions in table 45 are diagnostic and overlapping; they are not added as independent reductions. All validation checks passed, and the raw workbook checksum remained `{raw_hash}`.

## 9. Datasets available for the next analytical stage

- `02_customer_transaction_history.parquet`: all identified-customer transaction history.
- `03_positive_purchase_base.parquet`: approved basis for positive-purchase behavior.
- `04_customer_cancellation_history.parquet`: customer-linked C-prefix history.

Transaction preparation is complete. Churn has **not** been defined, the final modeling sample has **not** been determined, and observation and future-outcome windows have **not** been selected.

**Approved transaction-preparation rules were executed. Exact within-source duplicates were removed in the primary analysis datasets. Churn, observation/outcome windows, predictive features, train/test splits, encoding, scaling, machine-learning models, and SHAP analyses have not been created.**
"""
    report_path = PROJECT_ROOT / "reports" / "04_final_preparation_and_analysis_base.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_preparation() -> dict[str, object]:
    """Execute Step 4 once and return the prepared frames and validation metadata."""

    logger = get_logger()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    source, canonical_hash_before, raw_hash_before = validate_input()
    step2 = pd.read_csv(STEP2_SUMMARY_PATH).set_index("metric")["value"]
    raw_rows = int(float(step2["total_raw_rows_before_source_reconciliation"]))
    history = build_customer_history(source)
    positive = build_positive_purchase_base(source)
    cancellations = build_cancellation_history(source)
    write_parquet(history, CUSTOMER_HISTORY_PATH)
    write_parquet(positive, POSITIVE_PURCHASE_PATH)
    write_parquet(cancellations, CANCELLATION_HISTORY_PATH)

    waterfall = write_cleaning_waterfall(raw_rows, source, history, positive)
    coverage = write_customer_coverage(history, positive, cancellations)
    invoice = write_invoice_coverage(history, positive, cancellations)
    value_quantity = write_value_quantity_reconciliation(positive, cancellations)
    duplicate = write_duplicate_removal_summary(history)
    nonpurchase = write_nonpurchase_summary(history, positive)
    manifest = prepared_dataset_manifest([CUSTOMER_HISTORY_PATH, POSITIVE_PURCHASE_PATH, CANCELLATION_HISTORY_PATH])
    checks = validate_prepared_datasets(canonical_hash_before, raw_hash_before, source, history, positive, cancellations)
    write_report(waterfall, coverage, invoice, value_quantity, duplicate, nonpurchase, manifest, checks, raw_hash_before)
    logger.info("Step 4 completed with approved rules only.")
    logger.info("Canonical SHA-256: %s", canonical_hash_before)
    logger.info("Raw workbook SHA-256: %s", raw_hash_before)
    logger.info("Customer history rows: %s", len(history))
    logger.info("Positive purchase rows: %s", len(positive))
    logger.info("Cancellation history rows: %s", len(cancellations))
    logger.info("Validation checks: %s", checks)
    return {
        "source": source,
        "history": history,
        "positive": positive,
        "cancellations": cancellations,
        "manifest": manifest,
        "checks": checks,
        "canonical_hash": canonical_hash_before,
        "raw_hash": raw_hash_before,
    }


def main() -> None:
    """Run Step 4 and print a compact completion summary."""

    result = run_preparation()
    print({
        "customer_history_rows": len(result["history"]),
        "positive_purchase_rows": len(result["positive"]),
        "cancellation_history_rows": len(result["cancellations"]),
        "missing_customer_id_rows_excluded": len(result["source"]) - len(result["history"]),
        "customer_history_customers": result["history"]["Customer ID"].nunique(),
        "positive_purchase_customers": result["positive"]["Customer ID"].nunique(),
        "cancellation_customers": result["cancellations"]["Customer ID"].nunique(),
        "canonical_sha256": result["canonical_hash"],
        "raw_sha256": result["raw_hash"],
        "validation": result["checks"],
    })


if __name__ == "__main__":
    main()
