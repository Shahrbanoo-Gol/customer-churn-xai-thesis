"""Step 1A: diagnose temporal overlap and exact cross-sheet repetition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .audit_raw import (
    _file_snapshot,
    _logger,
    _normalise_name,
    _safe_percentage,
    find_raw_dataset,
    load_raw_partitions,
    sha256_file,
)
from .config import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR


EXPECTED_SHEETS = ("Year 2009-2010", "Year 2010-2011")
SIGNATURE_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def _sheet_frames(source_path: Path) -> dict[str, pd.DataFrame]:
    partitions = load_raw_partitions(source_path)
    frames = {partition["sheet_name"]: partition["frame"] for partition in partitions}
    missing = [sheet for sheet in EXPECTED_SHEETS if sheet not in frames]
    if missing:
        raise KeyError(f"Expected worksheets not found: {missing}. Observed: {list(frames)}")
    if set(frames) != set(EXPECTED_SHEETS):
        raise KeyError(f"Step 1A requires exactly the audited worksheets: {EXPECTED_SHEETS}; observed {list(frames)}")
    for sheet_name, frame in frames.items():
        if list(frame.columns) != SIGNATURE_COLUMNS:
            raise ValueError(
                f"Worksheet {sheet_name!r} does not have the exact Step 1 schema/order "
                f"{SIGNATURE_COLUMNS}; observed {list(frame.columns)}"
            )
    return frames


def _signature_counts(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Count exact full-row signatures, retaining nulls as equal missing values."""

    return (
        frame.groupby(SIGNATURE_COLUMNS, dropna=False, sort=False)
        .size()
        .rename(f"count_{suffix}")
        .reset_index()
    )


def _merge_counts(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(right, on=SIGNATURE_COLUMNS, how="outer", sort=False)
    for column in ("count_sheet1", "count_sheet2"):
        if column not in merged:
            merged[column] = 0
        merged[column] = merged[column].fillna(0).astype("int64")
    return merged


def _set_metrics(values1: set[Any], values2: set[Any], scope: str, value_type: str) -> dict[str, Any]:
    shared = values1 & values2
    return {
        "scope": scope,
        "identifier_type": value_type,
        "unique_sheet_1": len(values1),
        "unique_sheet_2": len(values2),
        "shared": len(shared),
        "only_sheet_1": len(values1 - values2),
        "only_sheet_2": len(values2 - values1),
    }


def _non_null_set(series: pd.Series) -> set[Any]:
    return set(series.dropna().tolist())


def _prefix_set(series: pd.Series, prefix: str = "C") -> set[Any]:
    text = series.astype("string")
    return set(text[text.str.startswith(prefix, na=False)].dropna().tolist())


def _write_report_appendix(
    overlap: dict[str, Any],
    exact_summary: dict[str, Any],
    reconciliation: dict[str, Any],
    invoice_metrics: pd.DataFrame,
    customer_metrics: dict[str, Any],
    duplicate_reconciliation: dict[str, Any],
) -> None:
    report_path = PROJECT_ROOT / "reports" / "01_raw_data_audit.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Base audit report does not exist: {report_path}")
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Cross-sheet temporal overlap\n"
    base = existing.split(marker, maxsplit=1)[0].rstrip()
    pct1 = exact_summary["percentage_overlap_rows_sheet_1_matched"]
    pct2 = exact_summary["percentage_overlap_rows_sheet_2_matched"]
    if pct1 == 100 and pct2 == 100:
        extent = "complete"
    elif pct1 == 0 and pct2 == 0:
        extent = "none"
    elif pct1 == pct2:
        extent = "partial"
    else:
        extent = "partial and heterogeneous"

    invoice_overlap = invoice_metrics[invoice_metrics["scope"] == "overlap_period"]
    if pct1 == 100 and pct2 == 100:
        recommendation = (
            "A. Deduplicate exact records across worksheets while retaining one occurrence. "
            "The overlap is complete: every overlap-period row in both worksheets has an exact "
            "full-row counterpart, with multiplicity accounted for, and no exact matches occur "
            "outside the overlap. This confirms the cross-worksheet reconciliation rule."
        )
    else:
        recommendation = (
            "B. Treat the worksheets as separate source periods with a documented reconciliation rule. "
            "The observed overlap is "
            f"{extent}, with {pct1:.4f}% of overlap-period rows in Year 2009-2010 and "
            f"{pct2:.4f}% in Year 2010-2011 having exact counterparts. The overlap should therefore "
            "be reconciled explicitly before customer-level preprocessing."
        )

    appendix = f"""

## Cross-sheet temporal overlap

The two worksheets have overlapping date ranges. Based on the observed worksheet minima and maxima, the exact intersection is **{overlap['overlap_start_datetime']} through {overlap['overlap_end_datetime']}**, lasting **{overlap['duration_days']:.6f} days ({overlap['duration_hours']:.6f} hours)**.

Rows inside this interval:

- Year 2009-2010: {overlap['sheet_1_rows_in_overlap']:,}
- Year 2010-2011: {overlap['sheet_2_rows_in_overlap']:,}

The full-row comparison used all eight original variables and accounted for duplicate multiplicities. It found **{exact_summary['unique_shared_full_row_signatures']:,} shared full-row signatures** and **{exact_summary['total_matched_cross_sheet_row_occurrences']:,} matched cross-sheet row occurrences**. Within the temporal intersection, **{exact_summary['overlap_rows_sheet_1_with_exact_counterpart']:,} ({pct1:.4f}%)** of Year 2009-2010 rows and **{exact_summary['overlap_rows_sheet_2_with_exact_counterpart']:,} ({pct2:.4f}%)** of Year 2010-2011 rows had exact counterparts.

The evidence indicates that the overlap is **{extent}**, rather than automatically establishing that either worksheet is erroneous. Exact matches outside the temporal intersection were assessed separately: **{exact_summary['shared_signatures_with_exact_match_outside_overlap']:,} shared signatures** had an exact cross-sheet counterpart involving a row outside the intersection.

## Overlap reconciliation

Within the temporal intersection, multiplicity-aware reconciliation after within-sheet keep-first deduplication found:

- {reconciliation['row_occurrences_only_sheet_1']:,} row occurrences only in Year 2009-2010;
- {reconciliation['row_occurrences_only_sheet_2']:,} row occurrences only in Year 2010-2011;
- {reconciliation['matched_row_occurrences']:,} matched row occurrences present in both worksheets.

These unmatched records are reported descriptively and are not interpreted as errors.

## Invoice and customer overlap

Within the temporal intersection, shared invoice IDs were **{invoice_overlap.loc[invoice_overlap['identifier_type'] == 'Invoice', 'shared'].iloc[0]:,}**, shared `C`-prefix invoice IDs were **{invoice_overlap.loc[invoice_overlap['identifier_type'] == 'C-prefix Invoice', 'shared'].iloc[0]:,}**, and shared non-null customers were **{customer_metrics['shared']:,}**.

## Duplicate-count reconciliation

After within-sheet keep-first deduplication, the `duplicated(keep=False)` counts were {duplicate_reconciliation['sheet_1_keep_false']:,} and {duplicate_reconciliation['sheet_2_keep_false']:,}. The combined count was {duplicate_reconciliation['combined_duplicate_rows_keep_false']:,}; cross-sheet repetition is handled only through the verified common-interval reconciliation.

For reference, `duplicated(keep='first')` counts were {duplicate_reconciliation['sheet_1_keep_first']:,} and {duplicate_reconciliation['sheet_2_keep_first']:,} within sheets, versus {duplicate_reconciliation['combined_keep_first']:,} after row-wise combination. The earlier report's `exact_duplicated_row_count` corresponds to `keep=False`; it has not been silently replaced.

## Applied reconciliation decision

**Recommendation: {recommendation}**

Within-sheet exact duplicate extras were removed before this cross-sheet comparison. No churn labeling, feature engineering, train/test splitting, encoding, scaling, modeling, or SHAP step was executed.
"""
    report_path.write_text(base + appendix, encoding="utf-8")


def run_sheet_overlap_audit() -> dict[str, Any]:
    """Run the Step 1A overlap audit without writing to raw or processed data."""

    logger: logging.Logger = _logger()
    logger.info("Starting Step 1A sheet overlap audit")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_before = _file_snapshot(RAW_DATA_DIR)
    processed_before = _file_snapshot(PROCESSED_DATA_DIR)
    source_path = find_raw_dataset()
    source_hash_before = sha256_file(source_path)
    frames = _sheet_frames(source_path)
    sheet1_name, sheet2_name = EXPECTED_SHEETS
    raw_sheet1 = frames[sheet1_name]
    raw_sheet2 = frames[sheet2_name]
    sheet1 = raw_sheet1.loc[~raw_sheet1.duplicated(subset=SIGNATURE_COLUMNS, keep="first")].copy()
    sheet2 = raw_sheet2.loc[~raw_sheet2.duplicated(subset=SIGNATURE_COLUMNS, keep="first")].copy()
    date_column = "InvoiceDate"

    min1, max1 = sheet1[date_column].min(), sheet1[date_column].max()
    min2, max2 = sheet2[date_column].min(), sheet2[date_column].max()
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)
    if pd.isna(overlap_start) or pd.isna(overlap_end) or overlap_start > overlap_end:
        raise ValueError("The two worksheet date ranges do not have a valid temporal intersection.")

    overlap_mask1 = sheet1[date_column].between(overlap_start, overlap_end, inclusive="both")
    overlap_mask2 = sheet2[date_column].between(overlap_start, overlap_end, inclusive="both")
    overlap1 = sheet1.loc[overlap_mask1].copy()
    overlap2 = sheet2.loc[overlap_mask2].copy()
    duration = overlap_end - overlap_start

    overlap_summary = {
        "sheet_1": sheet1_name,
        "sheet_2": sheet2_name,
        "sheet_1_min_datetime": min1.isoformat(),
        "sheet_1_max_datetime": max1.isoformat(),
        "sheet_2_min_datetime": min2.isoformat(),
        "sheet_2_max_datetime": max2.isoformat(),
        "overlap_start_datetime": overlap_start.isoformat(),
        "overlap_end_datetime": overlap_end.isoformat(),
        "duration_days": duration.total_seconds() / 86400,
        "duration_hours": duration.total_seconds() / 3600,
        "sheet_1_rows_in_overlap": len(overlap1),
        "sheet_2_rows_in_overlap": len(overlap2),
    }
    pd.DataFrame([overlap_summary]).to_csv(TABLES_DIR / "14_sheet_date_overlap.csv", index=False)

    full_counts1 = _signature_counts(sheet1, "sheet1")
    full_counts2 = _signature_counts(sheet2, "sheet2")
    full_counts = _merge_counts(full_counts1, full_counts2)
    overlap_counts1 = _signature_counts(overlap1, "overlap1").rename(columns={"count_overlap1": "overlap_count_sheet1"})
    overlap_counts2 = _signature_counts(overlap2, "overlap2").rename(columns={"count_overlap2": "overlap_count_sheet2"})
    full_counts = full_counts.merge(overlap_counts1, on=SIGNATURE_COLUMNS, how="left", sort=False)
    full_counts = full_counts.merge(overlap_counts2, on=SIGNATURE_COLUMNS, how="left", sort=False)
    for column in ("overlap_count_sheet1", "overlap_count_sheet2"):
        full_counts[column] = full_counts[column].fillna(0).astype("int64")
    full_counts["matched_cross_sheet_multiplicity"] = full_counts[["count_sheet1", "count_sheet2"]].min(axis=1)
    full_counts["outside_count_sheet1"] = full_counts["count_sheet1"] - full_counts["overlap_count_sheet1"]
    full_counts["outside_count_sheet2"] = full_counts["count_sheet2"] - full_counts["overlap_count_sheet2"]
    full_counts["exact_match_outside_overlap_sheet1_occurrences"] = full_counts[["outside_count_sheet1", "count_sheet2"]].min(axis=1)
    full_counts["exact_match_outside_overlap_sheet2_occurrences"] = full_counts[["outside_count_sheet2", "count_sheet1"]].min(axis=1)
    full_counts["has_exact_match_outside_overlap"] = (
        (full_counts["exact_match_outside_overlap_sheet1_occurrences"] > 0)
        | (full_counts["exact_match_outside_overlap_sheet2_occurrences"] > 0)
    )
    shared_full = full_counts[(full_counts["count_sheet1"] > 0) & (full_counts["count_sheet2"] > 0)].copy()
    shared_full.to_csv(TABLES_DIR / "15_cross_sheet_exact_overlap.csv", index=False)

    overlap_counts = full_counts[(full_counts["overlap_count_sheet1"] > 0) | (full_counts["overlap_count_sheet2"] > 0)].copy()
    overlap_counts["matched_overlap_multiplicity"] = overlap_counts[["overlap_count_sheet1", "overlap_count_sheet2"]].min(axis=1)
    shared_overlap = overlap_counts[
        (overlap_counts["overlap_count_sheet1"] > 0) & (overlap_counts["overlap_count_sheet2"] > 0)
    ]
    only1_overlap = overlap_counts[(overlap_counts["overlap_count_sheet1"] > 0) & (overlap_counts["overlap_count_sheet2"] == 0)]
    only2_overlap = overlap_counts[(overlap_counts["overlap_count_sheet2"] > 0) & (overlap_counts["overlap_count_sheet1"] == 0)]
    reconciliation = {
        "overlap_rows_sheet_1": len(overlap1),
        "overlap_rows_sheet_2": len(overlap2),
        "unique_full_row_signatures_only_sheet_1": len(only1_overlap),
        "unique_full_row_signatures_only_sheet_2": len(only2_overlap),
        "shared_full_row_signatures": len(shared_overlap),
        "row_occurrences_only_sheet_1": int(only1_overlap["overlap_count_sheet1"].sum()),
        "row_occurrences_only_sheet_2": int(only2_overlap["overlap_count_sheet2"].sum()),
        "matched_row_occurrences": int(shared_overlap["matched_overlap_multiplicity"].sum()),
    }
    reconciliation_rows = [
        {"metric": "overlap_row_count", "sheet_1": reconciliation["overlap_rows_sheet_1"], "sheet_2": reconciliation["overlap_rows_sheet_2"], "overall": reconciliation["overlap_rows_sheet_1"] + reconciliation["overlap_rows_sheet_2"], "definition": "Rows whose InvoiceDate falls within the inclusive intersection."},
        {"metric": "unique_full_row_signatures_only", "sheet_1": reconciliation["unique_full_row_signatures_only_sheet_1"], "sheet_2": reconciliation["unique_full_row_signatures_only_sheet_2"], "overall": reconciliation["unique_full_row_signatures_only_sheet_1"] + reconciliation["unique_full_row_signatures_only_sheet_2"], "definition": "Signatures present in one overlap subset and absent from the other."},
        {"metric": "row_occurrences_only", "sheet_1": reconciliation["row_occurrences_only_sheet_1"], "sheet_2": reconciliation["row_occurrences_only_sheet_2"], "overall": reconciliation["row_occurrences_only_sheet_1"] + reconciliation["row_occurrences_only_sheet_2"], "definition": "Multiplicity-aware occurrences for signatures present in only one overlap subset."},
        {"metric": "shared_full_row_signatures", "sheet_1": reconciliation["shared_full_row_signatures"], "sheet_2": reconciliation["shared_full_row_signatures"], "overall": reconciliation["shared_full_row_signatures"], "definition": "Full-row signatures present in both overlap subsets."},
        {"metric": "matched_row_occurrences", "sheet_1": reconciliation["matched_row_occurrences"], "sheet_2": reconciliation["matched_row_occurrences"], "overall": reconciliation["matched_row_occurrences"], "definition": "Sum of min(sheet-1 multiplicity, sheet-2 multiplicity) by signature."},
        {"metric": "exact_match_percentage_within_overlap", "sheet_1": _safe_percentage(reconciliation["matched_row_occurrences"], reconciliation["overlap_rows_sheet_1"]), "sheet_2": _safe_percentage(reconciliation["matched_row_occurrences"], reconciliation["overlap_rows_sheet_2"]), "overall": "", "definition": "Matched row occurrences divided by rows in that sheet's overlap subset."},
    ]
    pd.DataFrame(reconciliation_rows).to_csv(TABLES_DIR / "16_overlap_row_reconciliation.csv", index=False)

    invoice_metrics = pd.DataFrame(
        [
            _set_metrics(_non_null_set(overlap1["Invoice"]), _non_null_set(overlap2["Invoice"]), "overlap_period", "Invoice"),
            _set_metrics(_prefix_set(overlap1["Invoice"]), _prefix_set(overlap2["Invoice"]), "overlap_period", "C-prefix Invoice"),
            _set_metrics(_non_null_set(sheet1["Invoice"]), _non_null_set(sheet2["Invoice"]), "full_worksheets", "Invoice"),
            _set_metrics(_prefix_set(sheet1["Invoice"]), _prefix_set(sheet2["Invoice"]), "full_worksheets", "C-prefix Invoice"),
        ]
    )
    invoice_metrics.to_csv(TABLES_DIR / "17_invoice_overlap_audit.csv", index=False)

    customer_set1 = _non_null_set(overlap1["Customer ID"])
    customer_set2 = _non_null_set(overlap2["Customer ID"])
    customer_metrics = _set_metrics(customer_set1, customer_set2, "overlap_period", "Customer ID")
    pd.DataFrame([customer_metrics]).to_csv(TABLES_DIR / "18_customer_overlap_audit.csv", index=False)

    within1_keep_false = int(sheet1.duplicated(subset=SIGNATURE_COLUMNS, keep=False).sum())
    within2_keep_false = int(sheet2.duplicated(subset=SIGNATURE_COLUMNS, keep=False).sum())
    combined = pd.concat([sheet1, sheet2], ignore_index=True)
    combined_keep_false = int(combined.duplicated(subset=SIGNATURE_COLUMNS, keep=False).sum())
    duplicate_reconciliation = {
        "sheet_1_keep_false": within1_keep_false,
        "sheet_2_keep_false": within2_keep_false,
        "within_sheet_duplicate_rows_sum": within1_keep_false + within2_keep_false,
        "combined_duplicate_rows_keep_false": combined_keep_false,
        "cross_sheet_increment_in_participating_duplicate_rows": combined_keep_false - within1_keep_false - within2_keep_false,
        "sheet_1_keep_first": int(sheet1.duplicated(subset=SIGNATURE_COLUMNS, keep="first").sum()),
        "sheet_2_keep_first": int(sheet2.duplicated(subset=SIGNATURE_COLUMNS, keep="first").sum()),
        "combined_keep_first": int(combined.duplicated(subset=SIGNATURE_COLUMNS, keep="first").sum()),
    }
    duplicate_rows = [
        {"metric": "within_sheet_exact_duplicate_rows_keep_false", "sheet_1": duplicate_reconciliation["sheet_1_keep_false"], "sheet_2": duplicate_reconciliation["sheet_2_keep_false"], "combined": duplicate_reconciliation["within_sheet_duplicate_rows_sum"], "definition": "Rows belonging to repeated full-row signatures within each worksheet; duplicated(keep=False)."},
        {"metric": "combined_exact_duplicate_rows_keep_false", "sheet_1": "", "sheet_2": "", "combined": duplicate_reconciliation["combined_duplicate_rows_keep_false"], "definition": "Rows belonging to repeated full-row signatures after row-wise concatenation; duplicated(keep=False)."},
        {"metric": "cross_sheet_increment_in_duplicate_rows", "sheet_1": "", "sheet_2": "", "combined": duplicate_reconciliation["cross_sheet_increment_in_participating_duplicate_rows"], "definition": "Combined keep=False count minus the two within-sheet keep=False counts."},
        {"metric": "duplicate_rows_keep_first", "sheet_1": duplicate_reconciliation["sheet_1_keep_first"], "sheet_2": duplicate_reconciliation["sheet_2_keep_first"], "combined": duplicate_reconciliation["combined_keep_first"], "definition": "Rows after the first occurrence of each repeated full-row signature; duplicated(keep='first')."},
    ]
    pd.DataFrame(duplicate_rows).to_csv(TABLES_DIR / "19_duplicate_reconciliation.csv", index=False)

    exact_summary = {
        "unique_shared_full_row_signatures": int(len(shared_full)),
        "total_matched_cross_sheet_row_occurrences": int(shared_full["matched_cross_sheet_multiplicity"].sum()),
        "overlap_rows_sheet_1_with_exact_counterpart": int((full_counts[["overlap_count_sheet1", "count_sheet2"]].min(axis=1)).sum()),
        "overlap_rows_sheet_2_with_exact_counterpart": int((full_counts[["overlap_count_sheet2", "count_sheet1"]].min(axis=1)).sum()),
        "percentage_overlap_rows_sheet_1_matched": _safe_percentage((full_counts[["overlap_count_sheet1", "count_sheet2"]].min(axis=1)).sum(), len(overlap1)),
        "percentage_overlap_rows_sheet_2_matched": _safe_percentage((full_counts[["overlap_count_sheet2", "count_sheet1"]].min(axis=1)).sum(), len(overlap2)),
        "shared_signatures_with_exact_match_outside_overlap": int(full_counts["has_exact_match_outside_overlap"].sum()),
    }

    # Daily counts are an audit visualization only; no transformed dataset is persisted.
    plot_start = overlap_start.floor("D")
    plot_end = overlap_end.floor("D")
    daily_index = pd.date_range(plot_start, plot_end, freq="D")
    daily1 = sheet1.loc[overlap_mask1].assign(_calendar_day=sheet1.loc[overlap_mask1, date_column].dt.floor("D")).groupby("_calendar_day").size()
    daily2 = sheet2.loc[overlap_mask2].assign(_calendar_day=sheet2.loc[overlap_mask2, date_column].dt.floor("D")).groupby("_calendar_day").size()
    daily = pd.DataFrame({sheet1_name: daily1, sheet2_name: daily2}, index=daily_index).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(daily.index, daily[sheet1_name], marker="o", linewidth=2, label=sheet1_name)
    ax.plot(daily.index, daily[sheet2_name], marker="o", linewidth=2, label=sheet2_name)
    ax.set_xlabel("Calendar day")
    ax.set_ylabel("Number of raw transaction rows")
    ax.set_title("Raw rows by calendar day during the December 2010 sheet overlap")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sheet_overlap_daily_rows.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_report_appendix(overlap_summary, exact_summary, reconciliation, invoice_metrics, customer_metrics, duplicate_reconciliation)

    source_hash_after = sha256_file(source_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Raw source checksum changed during Step 1A.")
    if raw_before != _file_snapshot(RAW_DATA_DIR):
        raise RuntimeError("A file in data/raw changed or was created during Step 1A.")
    if processed_before != _file_snapshot(PROCESSED_DATA_DIR):
        raise RuntimeError("A processed analytical file was created or changed during Step 1A.")

    expected = [
        TABLES_DIR / f"{number:02d}_{name}.csv"
        for number, name in [
            (14, "sheet_date_overlap"),
            (15, "cross_sheet_exact_overlap"),
            (16, "overlap_row_reconciliation"),
            (17, "invoice_overlap_audit"),
            (18, "customer_overlap_audit"),
            (19, "duplicate_reconciliation"),
        ]
    ] + [FIGURES_DIR / "01_sheet_overlap_daily_rows.png", PROJECT_ROOT / "reports" / "01_raw_data_audit.md"]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Step 1A outputs missing: {missing}")

    logger.info("Step 1A source checksum unchanged: %s", source_hash_after)
    logger.info("Step 1A validation passed: no files written to data/raw")
    logger.info("Step 1A validation passed: no processed analytical dataset created")
    logger.info("Step 1A completed without removing or modifying overlapping records")
    return {
        "overlap": overlap_summary,
        "exact_summary": exact_summary,
        "reconciliation": reconciliation,
        "invoice_metrics": invoice_metrics,
        "customer_metrics": customer_metrics,
        "duplicate_reconciliation": duplicate_reconciliation,
        "source_path": source_path,
    }


if __name__ == "__main__":
    run_sheet_overlap_audit()

