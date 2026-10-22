"""Step 2: source reconciliation and non-destructive cleaning audit."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audit_raw import _file_snapshot, find_raw_dataset, load_raw_partitions, sha256_file
from .config import FIGURES_DIR, LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR


RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
SHEET_1 = "Year 2009-2010"
SHEET_2 = "Year 2010-2011"


def _logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.reconcile_and_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(
            LOGS_DIR / "02_source_reconciliation_and_cleaning_audit.log", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _safe_percentage(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    if limit is not None:
        selected = selected.head(limit)
    if selected.empty:
        return "_No rows._"
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )


def _verify_git_and_raw(manifest_path: Path) -> tuple[Path, str]:
    branch = _run_git("branch", "--show-current").stdout.strip()
    if branch != "main":
        raise RuntimeError(f"Step 2 requires Git branch 'main'; detected {branch!r}.")

    source_path = find_raw_dataset()
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError(f"Raw-file manifest is empty: {manifest_path}")
    expected_hash = str(manifest.iloc[0]["sha256"])
    observed_hash = sha256_file(source_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Raw checksum changed. Expected {expected_hash}; observed {observed_hash}."
        )

    ignored = _run_git("check-ignore", "--no-index", "--quiet", str(source_path.relative_to(PROJECT_ROOT)))
    if ignored.returncode != 0:
        raise RuntimeError("The raw workbook is not ignored by Git.")
    tracked = _run_git("ls-files", "--error-unmatch", str(source_path.relative_to(PROJECT_ROOT)))
    if tracked.returncode == 0:
        raise RuntimeError("The raw workbook is tracked by Git.")
    return source_path, observed_hash


def _load_source_frames(source_path: Path) -> dict[str, pd.DataFrame]:
    partitions = load_raw_partitions(source_path)
    frames = {partition["sheet_name"]: partition["frame"] for partition in partitions}
    if set(frames) != {SHEET_1, SHEET_2}:
        raise RuntimeError(f"Expected exactly {SHEET_1!r} and {SHEET_2!r}; observed {list(frames)}")
    for name, frame in frames.items():
        if list(frame.columns) != RAW_COLUMNS:
            raise RuntimeError(f"Unexpected schema in {name}: {list(frame.columns)}")
        if frame["InvoiceDate"].isna().any():
            raise RuntimeError(f"Missing InvoiceDate values prevent safe chronological reconciliation in {name}.")
    return frames


def _signature_counts(frame: pd.DataFrame, count_name: str = "count") -> pd.DataFrame:
    return (
        frame.groupby(RAW_COLUMNS, dropna=False, sort=False)
        .size()
        .rename(count_name)
        .reset_index()
    )


def _multiset_counts_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_counts = _signature_counts(left, "left_count")
    right_counts = _signature_counts(right, "right_count")
    merged = left_counts.merge(right_counts, on=RAW_COLUMNS, how="outer", sort=False)
    merged[["left_count", "right_count"]] = merged[["left_count", "right_count"]].fillna(0).astype("int64")
    return bool((merged["left_count"] == merged["right_count"]).all())


def _parquet_storage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use stable textual storage for mixed-type identifier columns without changing row membership."""

    storage = frame.copy()
    for column in ("Invoice", "StockCode", "Description", "Country", "SourceSheet"):
        if column in storage.columns:
            storage[column] = storage[column].astype("string")
    return storage


def _expected_multiplicity_reconciliation(sheet1: pd.DataFrame, sheet2: pd.DataFrame) -> pd.DataFrame:
    counts1 = _signature_counts(sheet1, "sheet1_count")
    counts2 = _signature_counts(sheet2, "sheet2_count")
    merged = counts1.merge(counts2, on=RAW_COLUMNS, how="outer", sort=False)
    merged[["sheet1_count", "sheet2_count"]] = merged[["sheet1_count", "sheet2_count"]].fillna(0).astype("int64")
    merged["expected_count_after_cross_sheet_reconciliation"] = (
        merged["sheet1_count"]
        + merged["sheet2_count"]
        - merged[["sheet1_count", "sheet2_count"]].min(axis=1)
    )
    return merged


def _reconcile_source(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sheet1 = frames[SHEET_1].copy()
    sheet2 = frames[SHEET_2].copy()
    sheet1_max = sheet1["InvoiceDate"].max()
    sheet2_tail = sheet2.loc[sheet2["InvoiceDate"] > sheet1_max].copy()
    sheet1["SourceSheet"] = SHEET_1
    sheet2_tail["SourceSheet"] = SHEET_2
    reconciled = pd.concat([sheet1, sheet2_tail], ignore_index=True)
    reconciled = reconciled.sort_values("InvoiceDate", kind="mergesort").reset_index(drop=True)
    metadata = {
        "sheet_1_rows": len(sheet1),
        "sheet_2_rows": len(sheet2),
        "raw_total_rows": len(sheet1) + len(sheet2),
        "sheet_1_max_datetime": sheet1_max,
        "sheet_2_rows_retained_after_boundary": len(sheet2_tail),
        "cross_sheet_repeated_occurrences_removed": len(sheet2) - len(sheet2_tail),
        "reconciled_rows": len(reconciled),
        "sheet_2_first_retained_datetime": sheet2_tail["InvoiceDate"].min() if not sheet2_tail.empty else pd.NaT,
    }
    return reconciled, metadata


def _validate_reconciliation(
    sheet1: pd.DataFrame,
    sheet2: pd.DataFrame,
    reconciled: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    expected_counts = _expected_multiplicity_reconciliation(sheet1, sheet2)
    actual_counts = _signature_counts(reconciled, "actual_count")
    comparison = expected_counts.merge(actual_counts, on=RAW_COLUMNS, how="outer", sort=False)
    comparison[["expected_count_after_cross_sheet_reconciliation", "actual_count"]] = comparison[
        ["expected_count_after_cross_sheet_reconciliation", "actual_count"]
    ].fillna(0).astype("int64")
    if not (comparison["expected_count_after_cross_sheet_reconciliation"] == comparison["actual_count"]).all():
        raise RuntimeError("Preferred boundary reconciliation does not match multiplicity-aware reconciliation.")
    if set(reconciled.columns) != set(RAW_COLUMNS + ["SourceSheet"]):
        raise RuntimeError(f"Unexpected reconciled columns: {list(reconciled.columns)}")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in reconciled.columns):
        raise RuntimeError("A churn variable was unexpectedly created.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage_frame = _parquet_storage_frame(reconciled)
    storage_frame.to_parquet(output_path, engine="pyarrow", index=False)
    reloaded = pd.read_parquet(output_path, engine="pyarrow")
    if list(reloaded.columns) != RAW_COLUMNS + ["SourceSheet"]:
        raise RuntimeError("Parquet output changed the expected column structure.")
    if not _multiset_counts_equal(storage_frame[RAW_COLUMNS], reloaded[RAW_COLUMNS]):
        raise RuntimeError("Parquet round-trip changed one or more original raw-variable values.")
    return comparison


def _write_reconciliation_summary(reconciled: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    metrics = [
        ("raw_sheet_1_rows", metadata["sheet_1_rows"], "All rows retained from Year 2009-2010."),
        ("raw_sheet_2_rows", metadata["sheet_2_rows"], "Rows originally present in Year 2010-2011."),
        ("total_raw_rows_before_source_reconciliation", metadata["raw_total_rows"], "Sum of both raw worksheets."),
        ("cross_sheet_repeated_occurrences_removed", metadata["cross_sheet_repeated_occurrences_removed"], "Only the second representation of the proven overlap."),
        ("row_count_after_source_reconciliation", len(reconciled), "Chronologically sorted source-reconciled rows."),
        ("earliest_InvoiceDate", reconciled["InvoiceDate"].min().isoformat(), "Observed in reconciled data."),
        ("latest_InvoiceDate", reconciled["InvoiceDate"].max().isoformat(), "Observed in reconciled data."),
        ("unique_Invoice_count", reconciled["Invoice"].nunique(dropna=True), "Non-null unique invoice identifiers."),
        ("unique_non_null_Customer_ID_count", reconciled["Customer ID"].nunique(dropna=True), "Non-null unique customer identifiers."),
    ]
    summary = pd.DataFrame(metrics, columns=["metric", "value", "notes"])
    summary.to_csv(TABLES_DIR / "20_source_reconciliation_summary.csv", index=False)
    return summary


def _audit_date_continuity(reconciled: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    dates = reconciled["InvoiceDate"].dt.normalize()
    date_index = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = reconciled.assign(_calendar_date=dates).groupby("_calendar_date").size()
    daily_sheet = reconciled.assign(_calendar_date=dates).pivot_table(
        index="_calendar_date", columns="SourceSheet", values="Invoice", aggfunc="size", fill_value=0
    )
    daily_sheet = daily_sheet.reindex(date_index, fill_value=0)
    boundary_date = metadata["sheet_1_max_datetime"].normalize()
    boundary_gap_hours = (
        (metadata["sheet_2_first_retained_datetime"] - metadata["sheet_1_max_datetime"]).total_seconds() / 3600
        if pd.notna(metadata["sheet_2_first_retained_datetime"])
        else np.nan
    )
    same_boundary_day = (
        pd.notna(metadata["sheet_2_first_retained_datetime"])
        and metadata["sheet_2_first_retained_datetime"].normalize() == boundary_date
    )
    rows = []
    for date in date_index:
        row_count = int(daily.get(date, 0))
        source1_count = int(daily_sheet.get(SHEET_1, pd.Series(index=date_index, dtype=int)).get(date, 0))
        source2_count = int(daily_sheet.get(SHEET_2, pd.Series(index=date_index, dtype=int)).get(date, 0))
        if row_count == 0:
            classification = "normal_no_transaction_day"
            note = "No transaction rows observed; no date was imputed."
        elif date == boundary_date:
            classification = "source_boundary_date_with_transactions"
            note = (
                "First-sheet rows end at the source boundary; second-sheet retained rows continue on the same calendar day."
                if same_boundary_day
                else "Source boundary date requires review of the transition timing."
            )
        else:
            classification = "transaction_day"
            note = "Observed transaction rows."
        rows.append(
            {
                "record_type": "daily",
                "calendar_date": date.date().isoformat(),
                "reconciled_row_count": row_count,
                "year_2009_2010_rows": source1_count,
                "year_2010_2011_retained_rows": source2_count,
                "source_sheets_present": ", ".join(
                    source for source, count in ((SHEET_1, source1_count), (SHEET_2, source2_count)) if count
                ),
                "is_source_boundary_date": date == boundary_date,
                "gap_classification": classification,
                "boundary_gap_hours": boundary_gap_hours,
                "notes": note,
            }
        )
    continuity = pd.DataFrame(rows)
    continuity.to_csv(TABLES_DIR / "21_date_continuity_audit.csv", index=False)

    focused = continuity[
        pd.to_datetime(continuity["calendar_date"]).between(
            boundary_date - pd.Timedelta(days=7), boundary_date + pd.Timedelta(days=7), inclusive="both"
        )
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    x = pd.to_datetime(focused["calendar_date"])
    ax.plot(x, focused["year_2009_2010_rows"], marker="o", linewidth=2, label=SHEET_1)
    ax.plot(x, focused["year_2010_2011_retained_rows"], marker="o", linewidth=2, label=f"{SHEET_2} retained")
    ax.axvline(boundary_date, color="black", linestyle="--", linewidth=1, label="Source boundary date")
    ax.set_xlabel("Calendar day")
    ax.set_ylabel("Reconciled transaction rows")
    ax.set_title("Source-reconciled daily rows around the worksheet boundary")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_source_reconciled_boundary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return continuity


def _audit_duplicates(reconciled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _signature_counts(reconciled[RAW_COLUMNS], "multiplicity")
    duplicate_groups = groups[groups["multiplicity"] > 1].copy()
    duplicate_groups["extra_occurrences"] = duplicate_groups["multiplicity"] - 1
    records: list[dict[str, Any]] = [
        {"record_type": "summary", "scope": "overall", "period": "", "metric": "total_rows", "value": len(reconciled)},
        {"record_type": "summary", "scope": "overall", "period": "", "metric": "rows_participating_in_duplicate_groups_keep_false", "value": int(duplicate_groups["multiplicity"].sum())},
        {"record_type": "summary", "scope": "overall", "period": "", "metric": "extra_duplicate_occurrences_keep_first", "value": int(duplicate_groups["extra_occurrences"].sum())},
        {"record_type": "summary", "scope": "overall", "period": "", "metric": "unique_duplicate_signatures", "value": len(duplicate_groups)},
        {"record_type": "summary", "scope": "overall", "period": "", "metric": "maximum_duplicate_group_multiplicity", "value": int(duplicate_groups["multiplicity"].max()) if not duplicate_groups.empty else 0},
    ]
    multiplicity_dist = duplicate_groups.groupby("multiplicity").size().reset_index(name="number_of_duplicate_signatures")
    for row in multiplicity_dist.itertuples(index=False):
        records.append({
            "record_type": "multiplicity_distribution", "scope": "overall", "period": "", "metric": "", "value": "",
            "group_multiplicity": int(row.multiplicity), "number_of_duplicate_signatures": int(row.number_of_duplicate_signatures),
            "rows_in_duplicate_groups": int(row.multiplicity * row.number_of_duplicate_signatures),
        })
    duplicate_groups["year"] = duplicate_groups["InvoiceDate"].dt.year
    duplicate_groups["month"] = duplicate_groups["InvoiceDate"].dt.to_period("M").astype(str)
    for period_name, series in (("year", duplicate_groups["year"]), ("month", duplicate_groups["month"])):
        grouped = duplicate_groups.groupby(series).agg(
            duplicate_participating_rows=("multiplicity", "sum"),
            extra_duplicate_occurrences=("extra_occurrences", "sum"),
            unique_duplicate_signatures=("multiplicity", "size"),
        ).reset_index()
        for row in grouped.itertuples(index=False):
            records.append({
                "record_type": f"by_{period_name}", "scope": "reconciled", "period": str(getattr(row, period_name)),
                "metric": "duplicate_summary", "value": "", "duplicate_participating_rows": int(row.duplicate_participating_rows),
                "extra_duplicate_occurrences": int(row.extra_duplicate_occurrences), "unique_duplicate_signatures": int(row.unique_duplicate_signatures),
            })
    audit = pd.DataFrame(records)
    audit.to_csv(TABLES_DIR / "22_within_source_duplicate_audit.csv", index=False)
    return audit, duplicate_groups


def _audit_duplicate_context(reconciled: pd.DataFrame, duplicate_groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def summary(metric: str, value: Any, notes: str) -> None:
        rows.append({"record_type": "summary", "metric": metric, "value": value, "notes": notes})

    duplicate_count = len(duplicate_groups)
    summary("duplicate_signature_count", duplicate_count, "Unique exact eight-field signatures with multiplicity greater than one.")
    summary("duplicate_groups_same_Invoice", duplicate_count, "All exact duplicate signatures include Invoice, so each group has one Invoice value.")
    summary("duplicate_groups_same_StockCode_within_Invoice", duplicate_count, "All exact duplicate signatures include Invoice and StockCode.")
    summary("duplicate_groups_identical_InvoiceDate", duplicate_count, "All exact duplicate signatures include InvoiceDate.")
    summary("distinct_Invoices_with_duplicate_groups", duplicate_groups["Invoice"].nunique(dropna=True), "Distinct non-null invoice identifiers associated with duplicate signatures.")
    summary("duplicate_participating_rows", int(duplicate_groups["multiplicity"].sum()), "Rows counted using duplicated(keep=False).")
    summary("extra_duplicate_occurrences", int(duplicate_groups["extra_occurrences"].sum()), "Rows beyond the first occurrence using duplicated(keep='first').")

    for column, label in (
        ("InvoiceDate", "calendar_date"),
        ("Customer ID", "customer_id"),
        ("StockCode", "stock_code"),
        ("Country", "country"),
    ):
        grouped = duplicate_groups.copy()
        if column == "InvoiceDate":
            grouped["_category"] = grouped[column].dt.date.astype(str)
        else:
            grouped["_category"] = grouped[column].astype("string").fillna("<MISSING>")
        concentration = grouped.groupby("_category", dropna=False).agg(
            duplicate_signature_count=("multiplicity", "size"),
            duplicate_participating_rows=("multiplicity", "sum"),
        ).reset_index().sort_values("duplicate_participating_rows", ascending=False).head(10)
        for category, signature_count, participating_rows in concentration.itertuples(index=False, name=None):
            rows.append({
                "record_type": "concentration", "metric": label, "category": category,
                "value": int(participating_rows),
                "percentage": _safe_percentage(participating_rows, duplicate_groups["multiplicity"].sum()),
                "notes": f"{int(signature_count)} duplicate signatures in this category.",
            })

    examples = duplicate_groups.sort_values(
        ["multiplicity", "extra_occurrences", "InvoiceDate"], ascending=[False, False, True]
    ).head(20)
    for row in examples.itertuples(index=False):
        example = {"record_type": "example", "metric": "largest_duplicate_group", "value": int(row.multiplicity), "notes": "Representative largest exact duplicate group."}
        for column in RAW_COLUMNS:
            example[column] = getattr(row, column.replace(" ", "_"), None)
        example["duplicate_group_size"] = int(row.multiplicity)
        example["extra_occurrences"] = int(row.extra_occurrences)
        rows.append(example)
    context = pd.DataFrame(rows)
    context.to_csv(TABLES_DIR / "23_duplicate_context_examples.csv", index=False)
    return context


def _behavior_rates(frame: pd.DataFrame) -> dict[str, float]:
    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    price = pd.to_numeric(frame["Price"], errors="coerce")
    c_prefix = frame["Invoice"].astype("string").str.startswith("C", na=False)
    return {
        "c_prefix_rate": _safe_percentage(int(c_prefix.sum()), len(frame)),
        "negative_quantity_rate": _safe_percentage(int((quantity < 0).sum()), len(frame)),
        "zero_price_rate": _safe_percentage(int((price == 0).sum()), len(frame)),
        "negative_price_rate": _safe_percentage(int((price < 0).sum()), len(frame)),
    }


def _audit_customer_missingness(reconciled: pd.DataFrame) -> pd.DataFrame:
    customer_missing = reconciled["Customer ID"].isna()
    rows: list[dict[str, Any]] = []
    for status, mask in (("missing", customer_missing), ("identified", ~customer_missing)):
        subset = reconciled.loc[mask]
        item = {"record_type": "summary", "period": "overall", "customer_id_status": status, "row_count": len(subset), "percentage_of_total_rows": _safe_percentage(len(subset), len(reconciled))}
        item.update(_behavior_rates(subset))
        rows.append(item)
    monthly = reconciled.assign(_month=reconciled["InvoiceDate"].dt.to_period("M").astype(str))
    for month, subset in monthly.groupby("_month", sort=True):
        missing_count = int(subset["Customer ID"].isna().sum())
        rows.append({
            "record_type": "monthly", "period": month, "customer_id_status": "all_rows",
            "row_count": len(subset), "missing_customer_id_rows": missing_count,
            "missing_customer_id_percentage": _safe_percentage(missing_count, len(subset)),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "24_customer_id_missingness_audit.csv", index=False)
    return audit


def _audit_cancellations(reconciled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantity = pd.to_numeric(reconciled["Quantity"], errors="coerce")
    c_prefix = reconciled["Invoice"].astype("string").str.startswith("C", na=False)
    negative = quantity < 0
    categories = [
        ("C-prefix AND Quantity < 0", c_prefix & negative, "C-prefix", "negative"),
        ("C-prefix AND Quantity >= 0", c_prefix & ~negative, "C-prefix", "non-negative"),
        ("non-C invoice AND Quantity < 0", ~c_prefix & negative, "non-C", "negative"),
        ("non-C invoice AND Quantity >= 0", ~c_prefix & ~negative, "non-C", "non-negative"),
    ]
    rows = [
        {"category": label, "count": int(mask.sum()), "percentage": _safe_percentage(int(mask.sum()), len(reconciled)), "invoice_type": invoice_type, "quantity_sign": quantity_sign}
        for label, mask, invoice_type, quantity_sign in categories
    ]
    relationship = pd.DataFrame(rows)
    relationship.to_csv(TABLES_DIR / "25_cancellation_quantity_relationship.csv", index=False)

    example_rows = []
    for label, mask, _, _ in categories[1:3]:
        subset = reconciled.loc[mask, RAW_COLUMNS + ["SourceSheet"]].head(20).copy()
        subset.insert(0, "exception_type", label)
        subset.insert(1, "total_exception_rows", int(mask.sum()))
        example_rows.append(subset)
    examples = pd.concat(example_rows, ignore_index=True) if example_rows else pd.DataFrame()
    examples.to_csv(TABLES_DIR / "26_cancellation_exception_examples.csv", index=False)
    return relationship, examples


def _audit_nonpositive_prices(reconciled: pd.DataFrame) -> pd.DataFrame:
    price = pd.to_numeric(reconciled["Price"], errors="coerce")
    categories = [("Price < 0", price < 0), ("Price = 0", price == 0), ("Price > 0", price > 0)]
    rows: list[dict[str, Any]] = []
    for label, mask in categories:
        rows.append({"record_type": "summary", "price_category": label, "count": int(mask.sum()), "percentage_of_reconciled_rows": _safe_percentage(int(mask.sum()), len(reconciled))})

    nonpositive = reconciled.loc[price <= 0].copy()
    nonpositive["cancellation_status"] = np.where(
        nonpositive["Invoice"].astype("string").str.startswith("C", na=False), "C-prefix", "non-C"
    )
    pattern_columns = ["StockCode", "Description", "Quantity", "Customer ID", "Country", "cancellation_status", "Price"]
    if not nonpositive.empty:
        patterns = nonpositive.groupby(pattern_columns, dropna=False, sort=False).agg(
            pattern_row_count=("Invoice", "size"),
            example_Invoice=("Invoice", "first"),
            example_InvoiceDate=("InvoiceDate", "first"),
        ).reset_index()
        patterns["record_type"] = "pattern"
        patterns["price_category"] = np.where(patterns["Price"] < 0, "Price < 0", "Price = 0")
        patterns["count"] = patterns["pattern_row_count"]
        patterns["percentage_of_reconciled_rows"] = patterns["count"].map(lambda value: _safe_percentage(value, len(reconciled)))
        patterns["distinct_pattern_count"] = len(patterns)
        rows.extend(patterns.to_dict("records"))
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "27_nonpositive_price_audit.csv", index=False)
    return audit


def _stockcode_pattern(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value)
    if text.isdigit():
        return "numeric_only"
    if re.search(r"\s", text):
        return "contains_space"
    if re.search(r"[^A-Za-z0-9]", text):
        return "contains_symbol"
    return "alphanumeric"


def _description_signal(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    for keyword in ("postage", "post", "discount", "manual", "bank", "charge", "fee", "adjust", "amazon", "dotcom", "carriage", "gift", "sample", "credit"):
        if keyword in text:
            return keyword
    return ""


def _audit_special_stockcodes(reconciled: pd.DataFrame) -> pd.DataFrame:
    working = reconciled.copy()
    working["_stock_key"] = working["StockCode"].astype("string")
    working["_code_pattern"] = working["StockCode"].map(_stockcode_pattern)
    working["_description_signal"] = working["Description"].map(_description_signal)
    working["_c_prefix"] = working["Invoice"].astype("string").str.startswith("C", na=False)
    quantity = pd.to_numeric(working["Quantity"], errors="coerce")
    price = pd.to_numeric(working["Price"], errors="coerce")
    working["_negative_quantity"] = quantity < 0
    working["_nonpositive_price"] = price <= 0
    grouped = working.groupby("_stock_key", dropna=False, sort=False)
    summary = grouped.agg(
        row_count=("Invoice", "size"),
        unique_descriptions=("Description", lambda values: values.nunique(dropna=True)),
        missing_description_count=("Description", lambda values: int(values.isna().sum())),
        negative_quantity_count=("_negative_quantity", "sum"),
        nonpositive_price_count=("_nonpositive_price", "sum"),
        cancellation_prefix_count=("_c_prefix", "sum"),
        unique_non_null_customers=("Customer ID", lambda values: values.nunique(dropna=True)),
        example_description=("Description", "first"),
    ).reset_index()
    signals = working.groupby("_stock_key", dropna=False)["_description_signal"].agg(
        lambda values: next((value for value in values if value), "")
    ).reset_index(name="description_signal")
    patterns = working.groupby("_stock_key", dropna=False)["_code_pattern"].first().reset_index(name="stockcode_pattern")
    summary = summary.merge(signals, on="_stock_key").merge(patterns, on="_stock_key")
    summary["record_type"] = "stockcode"
    summary["observed_stockcode"] = summary["_stock_key"].fillna("<MISSING>")
    summary["diagnostic_signal"] = summary["description_signal"]
    special_code_patterns = {"missing", "contains_space", "contains_symbol"}
    summary.loc[
        summary["diagnostic_signal"].eq("") & summary["stockcode_pattern"].isin(special_code_patterns),
        "diagnostic_signal",
    ] = summary.loc[
        summary["diagnostic_signal"].eq("") & summary["stockcode_pattern"].isin(special_code_patterns),
        "stockcode_pattern",
    ]
    summary["unusual_behavior_count"] = summary[["missing_description_count", "negative_quantity_count", "nonpositive_price_count", "cancellation_prefix_count"]].sum(axis=1)
    summary = summary.sort_values(["unusual_behavior_count", "row_count"], ascending=False)
    summary.to_csv(TABLES_DIR / "28_special_stockcode_audit.csv", index=False)
    return summary


def _audit_country_profile(reconciled: pd.DataFrame) -> pd.DataFrame:
    working = reconciled.copy()
    working["_country_key"] = working["Country"].astype("string").fillna("<MISSING>")
    profile = working.groupby("_country_key", sort=False).agg(
        transaction_row_count=("Invoice", "size"),
        unique_non_null_customers=("Customer ID", lambda values: values.nunique(dropna=True)),
        unique_non_null_invoices=("Invoice", lambda values: values.nunique(dropna=True)),
    ).reset_index().rename(columns={"_country_key": "Country"})
    profile["percentage_of_rows"] = profile["transaction_row_count"].map(lambda value: _safe_percentage(value, len(reconciled)))
    profile = profile.sort_values("transaction_row_count", ascending=False).reset_index(drop=True)
    profile.to_csv(TABLES_DIR / "29_country_profile.csv", index=False)
    return profile


def _audit_decision_matrix(
    reconciled: pd.DataFrame,
    metadata: dict[str, Any],
    duplicate_groups: pd.DataFrame,
    customer_audit: pd.DataFrame,
    cancellation_audit: pd.DataFrame,
    price_audit: pd.DataFrame,
    stock_audit: pd.DataFrame,
) -> pd.DataFrame:
    total = len(reconciled)
    quantity = pd.to_numeric(reconciled["Quantity"], errors="coerce")
    price = pd.to_numeric(reconciled["Price"], errors="coerce")
    c_prefix = reconciled["Invoice"].astype("string").str.startswith("C", na=False)
    missing_customer = int(reconciled["Customer ID"].isna().sum())
    missing_description = int(reconciled["Description"].isna().sum())
    special_rows = int(
        stock_audit.loc[
            stock_audit["diagnostic_signal"].ne("")
            | stock_audit["stockcode_pattern"].isin(["missing", "contains_space", "contains_symbol"]),
            "row_count",
        ].sum()
    )
    duplicate_rows = int(duplicate_groups["multiplicity"].sum())
    cancellation_rows = int(c_prefix.sum())
    rows = [
        {"issue": "cross-sheet overlap", "observed_count": metadata["cross_sheet_repeated_occurrences_removed"], "evidence": "Complete exact overlap established in Step 1A; second representation removed by deterministic boundary rule.", "potential_effect_on_customer_features": "Would double-count transactions, invoices, and customer activity if retained.", "candidate_treatment": "Retain Year 2009-2010 and Year 2010-2011 rows after the first-sheet maximum InvoiceDate.", "decision_status": "Executed"},
        {"issue": "within-source exact duplicates", "observed_count": duplicate_rows, "evidence": "Exact duplicate groups remain after cross-sheet reconciliation.", "potential_effect_on_customer_features": "May inflate purchase frequency, quantity, revenue, or product counts.", "candidate_treatment": "Review duplicate context and define a documented rule before final preprocessing.", "decision_status": "Requires methodological review"},
        {"issue": "missing Customer ID", "observed_count": missing_customer, "evidence": "Customer identifiers are absent for a measurable share of transaction rows.", "potential_effect_on_customer_features": "Rows cannot be assigned directly to customer-level records.", "candidate_treatment": "Evaluate exclusion from customer-level modeling with sensitivity reporting.", "decision_status": "Recommended for next step"},
        {"issue": "missing Description", "observed_count": missing_description, "evidence": "Description is missing in the reconciled raw data.", "potential_effect_on_customer_features": "Limits product-description diagnostics but not necessarily transaction-level totals.", "candidate_treatment": "Retain while deciding whether description is used in later features.", "decision_status": "Requires methodological review"},
        {"issue": "C-prefix cancellations", "observed_count": cancellation_rows, "evidence": "Invoice identifiers beginning with C were observed and quantified.", "potential_effect_on_customer_features": "May affect purchase, return, and monetary behavior measures.", "candidate_treatment": "Define cancellation treatment in the next analytical stage.", "decision_status": "Recommended for next step"},
        {"issue": "negative Quantity without C-prefix", "observed_count": int((~c_prefix & (quantity < 0)).sum()), "evidence": "Negative quantities occur on both C-prefix and non-C invoices.", "potential_effect_on_customer_features": "Could affect item and revenue summaries if treated inconsistently.", "candidate_treatment": "Investigate and document before deciding on exclusions.", "decision_status": "Requires methodological review"},
        {"issue": "zero Price", "observed_count": int((price == 0).sum()), "evidence": "Zero-price rows are present and may represent special transaction behavior.", "potential_effect_on_customer_features": "Can affect revenue and price-derived features.", "candidate_treatment": "Inspect product and transaction context; do not automatically remove.", "decision_status": "Requires methodological review"},
        {"issue": "negative Price", "observed_count": int((price < 0).sum()), "evidence": "Negative-price rows are rare but present.", "potential_effect_on_customer_features": "Can affect monetary summaries and may indicate adjustments.", "candidate_treatment": "Review representative records before any rule is chosen.", "decision_status": "Requires methodological review"},
        {"issue": "special/non-merchandise StockCode patterns", "observed_count": special_rows, "evidence": "StockCode form and description signals identify diagnostic patterns without classifying them as errors.", "potential_effect_on_customer_features": "Administrative or service rows may distort product and purchase measures.", "candidate_treatment": "Create a documented classification rule after reviewing the audit.", "decision_status": "Requires methodological review"},
        {"issue": "invalid InvoiceDate", "observed_count": int(reconciled["InvoiceDate"].isna().sum()), "evidence": "No missing InvoiceDate values were found in the reconciled data.", "potential_effect_on_customer_features": "None observed at this stage.", "candidate_treatment": "Retain; continue validating dates in later processing.", "decision_status": "Retain"},
    ]
    matrix = pd.DataFrame(rows)
    matrix["observed_percentage"] = [
        _safe_percentage(row.observed_count, metadata["raw_total_rows"] if row.issue == "cross-sheet overlap" else total)
        for row in matrix.itertuples(index=False)
    ]
    matrix = matrix[["issue", "observed_count", "observed_percentage", "evidence", "potential_effect_on_customer_features", "candidate_treatment", "decision_status"]]
    matrix.to_csv(TABLES_DIR / "30_cleaning_decision_matrix.csv", index=False)
    return matrix


def _write_report(
    source_path: Path,
    source_hash: str,
    summary: pd.DataFrame,
    continuity: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    duplicate_context: pd.DataFrame,
    customer_audit: pd.DataFrame,
    cancellation_audit: pd.DataFrame,
    price_audit: pd.DataFrame,
    stock_audit: pd.DataFrame,
    country_profile: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    summary_map = dict(zip(summary["metric"], summary["value"]))
    missing_customer = customer_audit[(customer_audit["record_type"] == "summary") & (customer_audit["customer_id_status"] == "missing")]
    duplicate_summary = duplicate_audit[duplicate_audit["record_type"] == "summary"]
    duplicate_map = dict(zip(duplicate_summary["metric"], duplicate_summary["value"]))
    c2 = cancellation_audit[cancellation_audit["category"].isin(["C-prefix AND Quantity >= 0", "non-C invoice AND Quantity < 0"])]
    price_summary = price_audit[price_audit["record_type"] == "summary"]
    special_top = stock_audit.head(15)
    boundary_rows = continuity[continuity["is_source_boundary_date"]]
    report = f"""# Step 2 — Source Reconciliation and Cleaning Audit

## 1. Purpose

This step constructs a chronological source-reconciled transaction dataset after the complete exact worksheet overlap established in Step 1A. It executes only that source-reconciliation rule and audits remaining raw-data conditions. It does not create churn labels, RFM features, predictive features, train/test splits, or machine-learning models.

The source file was `{source_path.relative_to(PROJECT_ROOT)}` with SHA-256 `{source_hash}`. The raw workbook remained read-only.

## 2. Source reconciliation

The deterministic rule retained all `{SHEET_1}` rows and retained only `{SHEET_2}` rows with `InvoiceDate` strictly after the maximum `{SHEET_1}` timestamp (`{metadata['sheet_1_max_datetime'].isoformat()}`). This removed `{metadata['cross_sheet_repeated_occurrences_removed']:,}` rows solely because they were the second representation of the proven cross-sheet overlap.

{_markdown_table(summary)}

The reconciled Parquet file contains the original eight variables plus the technical `SourceSheet` provenance field. Because the source identifier columns contain mixed numeric/text values, `Invoice` and `StockCode` use stable textual Parquet storage while preserving their displayed source values. No global `drop_duplicates()` operation was used.

## 3. Validation of the reconciled timeline

The reconciled data spans `{summary_map['earliest_InvoiceDate']}` through `{summary_map['latest_InvoiceDate']}`. The preferred boundary rule was validated against multiplicity-aware full-row counts: the resulting transaction multiset matched exactly.

There were **{int((continuity['reconciled_row_count'] == 0).sum())}** calendar dates without transaction rows; these were reported and not imputed. Around the source boundary:

{_markdown_table(boundary_rows, ['calendar_date', 'reconciled_row_count', 'year_2009_2010_rows', 'year_2010_2011_retained_rows', 'boundary_gap_hours', 'notes'])}

## 4. Remaining exact duplicates

After source reconciliation, `{duplicate_map.get('rows_participating_in_duplicate_groups_keep_false', 0):,}` rows participate in exact duplicate groups under `duplicated(keep=False)`, and `{duplicate_map.get('extra_duplicate_occurrences_keep_first', 0):,}` are extra occurrences beyond the first under `duplicated(keep='first')`. There are `{duplicate_map.get('unique_duplicate_signatures', 0):,}` duplicate signatures, with maximum multiplicity `{duplicate_map.get('maximum_duplicate_group_multiplicity', 0)}`.

Duplicate groups necessarily have the same Invoice, StockCode, and InvoiceDate because those are included in the exact eight-field signature. Their concentrations and representative largest groups are documented in `outputs/tables/23_duplicate_context_examples.csv`.

## 5. Missing Customer ID

Rows without Customer ID: `{int(missing_customer.iloc[0]['row_count']):,}` ({float(missing_customer.iloc[0]['percentage_of_total_rows']):.4f}%). They were retained. The monthly rate and comparisons of cancellation, quantity-sign, and price-sign behavior between missing and identified customers are in `outputs/tables/24_customer_id_missingness_audit.csv`.

## 6. Cancellations and negative quantities

{_markdown_table(cancellation_audit)}

The two exception categories were retained for review:

{_markdown_table(c2)}

No cancellation or negative-quantity rule was executed.

## 7. Non-positive prices

{_markdown_table(price_summary)}

Distinct non-positive-price patterns and representative records are saved in `outputs/tables/27_nonpositive_price_audit.csv`. Zero-price rows were not automatically classified as invalid.

## 8. Special transaction/product codes

The StockCode audit reports code form, description signals, missing descriptions, negative quantities, non-positive prices, and cancellation-prefix counts. The most behaviorally unusual observed codes/patterns were:

{_markdown_table(special_top, ['observed_stockcode', 'stockcode_pattern', 'description_signal', 'row_count', 'missing_description_count', 'negative_quantity_count', 'nonpositive_price_count', 'cancellation_prefix_count'], limit=15)}

These are diagnostic patterns only; no merchandise/non-merchandise classification was executed.

## 9. Geographic profile

{_markdown_table(country_profile.head(15))}

The complete country profile includes transaction rows, unique non-null customers, and unique non-null invoices for every observed country.

## 10. Cleaning-decision matrix

{_markdown_table(decision_matrix)}

## 11. Decisions requiring approval before final preprocessing

Only the proven cross-sheet overlap was executed. Before final preprocessing, methodological approval is still required for within-source exact duplicates, missing Customer ID rows, missing descriptions, cancellation treatment, negative quantities without a C-prefix, non-positive prices, and special/non-merchandise StockCode patterns. These decisions may affect customer-level feature construction and must be documented before modeling.

No causal relationships are inferred from these descriptive audits.
"""
    (PROJECT_ROOT / "reports" / "02_source_reconciliation_and_cleaning_audit.md").write_text(report, encoding="utf-8")


def run_step2_audit() -> dict[str, Any]:
    """Reconcile the proven worksheet overlap and audit remaining raw conditions."""

    logger = _logger()
    logger.info("Starting Step 2 source reconciliation and cleaning audit")
    for directory in (TABLES_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    source_path, source_hash = _verify_git_and_raw(TABLES_DIR / "01_raw_file_manifest.csv")
    raw_before = _file_snapshot(RAW_DATA_DIR)
    processed_before = _file_snapshot(PROCESSED_DATA_DIR)
    frames = _load_source_frames(source_path)
    sheet1, sheet2 = frames[SHEET_1], frames[SHEET_2]

    reconciled, metadata = _reconcile_source(frames)
    output_path = PROCESSED_DATA_DIR / "01_source_reconciled.parquet"
    _validate_reconciliation(sheet1, sheet2, reconciled, output_path)
    summary = _write_reconciliation_summary(reconciled, metadata)
    continuity = _audit_date_continuity(reconciled, metadata)
    duplicate_audit, duplicate_groups = _audit_duplicates(reconciled)
    duplicate_context = _audit_duplicate_context(reconciled, duplicate_groups)
    customer_audit = _audit_customer_missingness(reconciled)
    cancellation_audit, cancellation_examples = _audit_cancellations(reconciled)
    price_audit = _audit_nonpositive_prices(reconciled)
    stock_audit = _audit_special_stockcodes(reconciled)
    country_profile = _audit_country_profile(reconciled)
    decision_matrix = _audit_decision_matrix(
        reconciled, metadata, duplicate_groups, customer_audit, cancellation_audit, price_audit, stock_audit
    )
    _write_report(
        source_path, source_hash, summary, continuity, duplicate_audit, duplicate_context,
        customer_audit, cancellation_audit, price_audit, stock_audit, country_profile,
        decision_matrix, metadata,
    )

    if sha256_file(source_path) != source_hash:
        raise RuntimeError("Raw checksum changed during Step 2.")
    if raw_before != _file_snapshot(RAW_DATA_DIR):
        raise RuntimeError("A file in data/raw changed or was created during Step 2.")
    if _file_snapshot(PROCESSED_DATA_DIR).get("01_source_reconciled.parquet") is None:
        raise RuntimeError("The source-reconciled Parquet output was not created.")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in reconciled.columns):
        raise RuntimeError("A churn variable exists in the reconciled output.")
    persisted = pd.read_parquet(output_path, engine="pyarrow")
    if _multiset_counts_equal(_parquet_storage_frame(reconciled)[RAW_COLUMNS], persisted[RAW_COLUMNS]) is False:
        raise RuntimeError("The persisted reconciled dataset cannot reproduce the in-memory reconciled multiset.")

    processed_ignore = _run_git("check-ignore", "--no-index", "--quiet", "data/processed/01_source_reconciled.parquet")
    if processed_ignore.returncode != 0:
        raise RuntimeError("The generated processed Parquet dataset is not ignored by Git.")
    gitkeep_ignore = _run_git("check-ignore", "--no-index", "data/processed/.gitkeep")
    if gitkeep_ignore.returncode == 0:
        raise RuntimeError("data/processed/.gitkeep must remain trackable.")
    status = _run_git("status", "--short").stdout.strip()
    logger.info("Git status after Step 2:\n%s", status)
    logger.info("Raw checksum unchanged: %s", source_hash)
    logger.info("Source-reconciled row count: %s", len(reconciled))
    logger.info("Validation passed: only the proven cross-sheet overlap was reconciled")
    logger.info("Validation passed: original eight raw variables preserved")
    logger.info("Validation passed: no churn labels, predictive features, or model objects created")
    logger.info("Step 2 completed successfully")
    return {
        "source_path": source_path,
        "source_hash": source_hash,
        "reconciled": reconciled,
        "metadata": metadata,
        "summary": summary,
        "continuity": continuity,
        "duplicate_audit": duplicate_audit,
        "duplicate_context": duplicate_context,
        "customer_audit": customer_audit,
        "cancellation_audit": cancellation_audit,
        "cancellation_examples": cancellation_examples,
        "price_audit": price_audit,
        "stock_audit": stock_audit,
        "country_profile": country_profile,
        "decision_matrix": decision_matrix,
        "processed_path": output_path,
    }


if __name__ == "__main__":
    run_step2_audit()

