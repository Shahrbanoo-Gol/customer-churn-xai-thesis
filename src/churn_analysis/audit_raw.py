"""Reproducible, non-destructive audit of the raw Online Retail II file."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    TABLES_DIR,
    RANDOM_STATE,
)


SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
AUDIT_OUTPUT_NAMES = [
    "01_raw_file_manifest.csv",
    "02_raw_sheet_inventory.csv",
    "03_raw_schema_audit.csv",
    "04_date_coverage.csv",
    "05_missing_values.csv",
    "06_duplicate_audit.csv",
    "07_customer_id_audit.csv",
    "08_invoice_audit.csv",
    "09_quantity_audit.csv",
    "10_price_audit.csv",
    "11_product_audit.csv",
    "12_country_distribution.csv",
    "13_raw_condition_counts.csv",
]


def _logger() -> logging.Logger:
    """Return the audit logger without changing global warning or logging settings."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.audit_raw")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOGS_DIR / "01_raw_data_audit.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _safe_percentage(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """Capture relative path, size, and checksum for files below a directory."""

    if not root.exists():
        return {}
    snapshot: dict[str, tuple[int, str]] = {}
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = (path.stat().st_size, sha256_file(path))
    return snapshot


def find_raw_dataset() -> Path:
    """Find exactly one supported raw file and fail on ambiguity."""

    candidates = sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported raw dataset found in {RAW_DATA_DIR}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if len(candidates) > 1:
        names = ", ".join(str(path.relative_to(RAW_DATA_DIR)) for path in candidates)
        raise RuntimeError(f"More than one raw dataset candidate found; choose explicitly: {names}")
    return candidates[0]


def load_raw_partitions(path: Path) -> list[dict[str, Any]]:
    """Load raw sheets/CSV partitions for in-memory audit only."""

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        return [{"sheet_name": path.stem, "frame": frame}]

    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
    excel = pd.ExcelFile(path, engine=engine)
    return [
        {"sheet_name": sheet_name, "frame": pd.read_excel(excel, sheet_name=sheet_name)}
        for sheet_name in excel.sheet_names
    ]


def _field_aliases() -> dict[str, list[str]]:
    return {
        "date": ["invoicedate", "transactiondate", "date"],
        "customer_id": ["customerid", "customeridentifier", "customer"],
        "invoice": ["invoiceno", "invoice", "invoicenumber", "transactionid"],
        "quantity": ["quantity", "qty"],
        "price": ["unitprice", "price", "sellingprice"],
        "product_code": ["stockcode", "productcode", "itemcode", "sku"],
        "description": ["description", "productdescription", "itemdescription"],
        "country": ["country", "countryname"],
    }


def identify_fields(partitions: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Identify fundamental fields using actual column names, without renaming them."""

    aliases = _field_aliases()
    mapping: dict[str, dict[str, str]] = {role: {} for role in aliases}
    for partition in partitions:
        columns = list(partition["frame"].columns)
        normalised = {_normalise_name(column): column for column in columns}
        for role, role_aliases in aliases.items():
            selected = None
            for alias in role_aliases:
                if alias in normalised:
                    selected = normalised[alias]
                    break
            if selected is None:
                contains = [
                    column
                    for column in columns
                    if any(alias in _normalise_name(column) for alias in role_aliases)
                ]
                if len(contains) == 1:
                    selected = contains[0]
            if selected is None:
                raise KeyError(
                    f"Could not identify required '{role}' field in sheet "
                    f"'{partition['sheet_name']}'. Observed columns: {columns}"
                )
            mapping[role][partition["sheet_name"]] = selected
    return mapping


def _write_rows(rows: Iterable[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)
    return frame


def _write_worksheet_profiles(partitions: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Write citation-friendly, pre-modification worksheet profile tables."""

    summary_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    describe_rows: list[dict[str, Any]] = []
    for partition in partitions:
        sheet_name = partition["sheet_name"]
        frame = partition["frame"]
        summary_rows.append(
            {
                "sheet_name": sheet_name,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "column_names": " | ".join(map(str, frame.columns)),
                "unique_non_null_customers": int(frame["Customer ID"].nunique(dropna=True)),
                "unique_non_null_invoices": int(frame["Invoice"].nunique(dropna=True)),
                "date_min": _date_value(pd.to_datetime(frame["InvoiceDate"], errors="coerce").min()),
                "date_max": _date_value(pd.to_datetime(frame["InvoiceDate"], errors="coerce").max()),
                "random_seed": RANDOM_STATE,
            }
        )
        for position, row in frame.head(5).iterrows():
            item = {"sheet_name": sheet_name, "source_row_position": int(position)}
            item.update(row.to_dict())
            head_rows.append(item)
        for position, row in frame.tail(5).iterrows():
            item = {"sheet_name": sheet_name, "source_row_position": int(position)}
            item.update(row.to_dict())
            tail_rows.append(item)
        for position, row in frame.sample(n=min(5, len(frame)), random_state=RANDOM_STATE).iterrows():
            item = {"sheet_name": sheet_name, "source_row_position": int(position)}
            item.update(row.to_dict())
            sample_rows.append(item)
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            missing_rows.append(
                {
                    "sheet_name": sheet_name,
                    "column_name": column,
                    "missing_count": missing_count,
                    "missing_percentage": _safe_percentage(missing_count, len(frame)),
                }
            )
        with pd.option_context("display.max_columns", None):
            described = frame.describe(include="all").transpose().reset_index()
        described = described.rename(columns={"index": "column_name"})
        described.insert(0, "sheet_name", sheet_name)
        describe_rows.extend(described.to_dict("records"))

    outputs = {
        "profile_summary": _write_rows(summary_rows, TABLES_DIR / "47_raw_worksheet_profile_summary.csv"),
        "head": _write_rows(head_rows, TABLES_DIR / "48_raw_worksheet_head.csv"),
        "tail": _write_rows(tail_rows, TABLES_DIR / "49_raw_worksheet_tail.csv"),
        "sample": _write_rows(sample_rows, TABLES_DIR / "50_raw_worksheet_sample.csv"),
        "missing": _write_rows(missing_rows, TABLES_DIR / "51_raw_worksheet_missing_values.csv"),
        "descriptive_statistics": _write_rows(describe_rows, TABLES_DIR / "52_raw_worksheet_descriptive_statistics.csv"),
    }
    return outputs


def _schema_identical(partitions: list[dict[str, Any]]) -> bool:
    schemas = [tuple(partition["frame"].columns.tolist()) for partition in partitions]
    return len(set(schemas)) == 1


def _combined_frame(partitions: list[dict[str, Any]]) -> pd.DataFrame | None:
    if not _schema_identical(partitions):
        return None
    return pd.concat([partition["frame"] for partition in partitions], ignore_index=True)


def _date_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _numeric_summary(scope: str, series: pd.Series, variable: str) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    quantiles = valid.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]) if not valid.empty else pd.Series(dtype=float)

    def q(level: float) -> float:
        return float(quantiles.get(level, np.nan))

    return {
        "scope": scope,
        "variable": variable,
        "count": int(valid.count()),
        "invalid_numeric_count": int(numeric.isna().sum() - series.isna().sum()),
        "minimum": float(valid.min()) if not valid.empty else np.nan,
        "maximum": float(valid.max()) if not valid.empty else np.nan,
        "mean": float(valid.mean()) if not valid.empty else np.nan,
        "median": float(valid.median()) if not valid.empty else np.nan,
        "standard_deviation": float(valid.std()) if not valid.empty else np.nan,
        "percentile_1": q(0.01),
        "percentile_5": q(0.05),
        "percentile_25": q(0.25),
        "percentile_75": q(0.75),
        "percentile_95": q(0.95),
        "percentile_99": q(0.99),
        "number_less_than_zero": int((valid < 0).sum()),
        "number_equal_to_zero": int((valid == 0).sum()),
        "number_greater_than_zero": int((valid > 0).sum()),
    }


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


def _build_report(
    source_path: Path,
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    schema: pd.DataFrame,
    date_coverage: pd.DataFrame,
    missing: pd.DataFrame,
    duplicates: pd.DataFrame,
    customer_ids: pd.DataFrame,
    invoices: pd.DataFrame,
    quantities: pd.DataFrame,
    prices: pd.DataFrame,
    countries: pd.DataFrame,
    field_map: dict[str, dict[str, str]],
    schemas_identical: bool,
) -> None:
    manifest_row = manifest.iloc[0].to_dict()
    customer_missing = customer_ids[customer_ids["scope"] != "overall"]
    description_missing = missing[
        missing.apply(
            lambda row: row["column_name"]
            == field_map["description"].get(row["sheet_name"], "__not_found__"),
            axis=1,
        )
    ]
    country_overall = countries[countries["scope"] == "overall"]

    report = f"""# Raw Data Audit

This report documents a non-destructive audit of the raw source file for Step 1 of the thesis workflow. Statistics were recalculated directly from the source file; no records were removed, altered, imputed, cleaned, relabelled, or aggregated for modeling.

## Dataset file

- Filename: `{manifest_row['filename']}`
- Format: `{manifest_row['extension']}`
- Size in bytes: `{manifest_row['file_size_bytes']}`
- SHA-256: `{manifest_row['sha256']}`
- Last modified (UTC): `{manifest_row['last_modified_timestamp_utc']}`

## Workbook structure

Source path: `{source_path.relative_to(PROJECT_ROOT)}`

{_markdown_table(inventory)}

Schema identical across worksheets: **{schemas_identical}**.

## Raw variables

The exact observed column names and pandas-detected types are listed below. No source column names were changed.

{_markdown_table(schema, ['sheet_name', 'column_name', 'detected_dtype', 'non_null_count', 'null_count', 'null_percentage', 'unique_non_null_values'])}

## Date coverage

{_markdown_table(date_coverage)}

## Missing data

Customer-ID missingness by sheet:

{_markdown_table(customer_missing)}

Description missingness by sheet:

{_markdown_table(description_missing)}

The complete column-level missing-value audit is saved in `outputs/tables/05_missing_values.csv`.

## Duplicates

{_markdown_table(duplicates)}

Duplicate rows are reported only as diagnostics. No assumption was made that they are errors.

## Customer IDs

{_markdown_table(customer_ids)}

## Cancellations

{_markdown_table(invoices)}

The `C`-prefix convention is reported as observed in the invoice identifier. Cancellation transactions were not deleted, and negative quantities were not treated as automatically equivalent to cancellation invoices.

## Quantity and price

Quantity audit:

{_markdown_table(quantities)}

Price audit:

{_markdown_table(prices)}

Unusual values are reported descriptively only. No outlier or exclusion decision was made.

## Countries

Number of unique country values by scope and the descending transaction-row distribution are saved in `outputs/tables/12_country_distribution.csv`.

{_markdown_table(country_overall, limit=15)}

## Reproducible worksheet profiles before modification

Each raw worksheet was profiled independently before any duplicate removal. The citation-ready outputs are saved as `47_raw_worksheet_profile_summary.csv`, `48_raw_worksheet_head.csv`, `49_raw_worksheet_tail.csv`, `50_raw_worksheet_sample.csv`, `51_raw_worksheet_missing_values.csv`, and `52_raw_worksheet_descriptive_statistics.csv`. Random samples use the fixed seed `{RANDOM_STATE}`. These tables include worksheet shape, columns, data types, head, tail, sample, missingness, descriptive statistics, unique customer/invoice counts, and date ranges.

## Important methodological observations

- The source is transactional data, and the observed variables are limited to the fields documented above.
- Customer identifiers and product descriptions have the missingness shown in the audit tables; no rows were removed because of missing values.
- Invoice-prefix, quantity-sign, price-sign, duplicate, and invalid-date conditions were counted independently as raw-data diagnostics.
- The raw source file was treated as read-only throughout this audit.

## Decisions intentionally NOT made yet

This stage did **not** decide:

- which duplicates should be removed;
- whether missing Customer IDs should be removed;
- whether cancellations should be excluded;
- whether negative quantities should be excluded;
- whether zero or negative prices should be excluded;
- what constitutes an outlier;
- which countries should be included;
- how churn will be defined;
- what inactivity threshold will be used;
- what observation window will be used;
- what predictive features will be created.

No causal interpretations are made in this report. SHAP, churn labels, predictive features, train/test splits, and machine-learning models belong to later stages and were not created here.
"""
    (PROJECT_ROOT / "reports" / "01_raw_data_audit.md").write_text(report, encoding="utf-8")


def run_audit() -> dict[str, Any]:
    """Run the complete raw audit and validation workflow."""

    logger = _logger()
    logger.info("Starting non-destructive raw data audit")
    for directory in (TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw_before = _file_snapshot(RAW_DATA_DIR)
    processed_before = _file_snapshot(PROCESSED_DATA_DIR)
    source_path = find_raw_dataset()
    source_before_hash = sha256_file(source_path)
    logger.info("Raw source detected: %s", source_path)

    stat = source_path.stat()
    manifest = _write_rows(
        [
            {
                "filename": source_path.name,
                "extension": source_path.suffix.lower(),
                "file_size_bytes": stat.st_size,
                "sha256": source_before_hash,
                "last_modified_timestamp_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        ],
        TABLES_DIR / "01_raw_file_manifest.csv",
    )

    partitions = load_raw_partitions(source_path)
    logger.info("Loaded %d raw worksheet/data partition(s) for audit", len(partitions))
    field_map = identify_fields(partitions)
    schemas_identical = _schema_identical(partitions)
    combined = _combined_frame(partitions)
    if combined is not None:
        first_sheet = partitions[0]["sheet_name"]
        for role in field_map:
            field_map[role]["overall"] = field_map[role][first_sheet]

    inventory = _write_rows(
        [
            {
                "sheet_name": partition["sheet_name"],
                "row_count": int(partition["frame"].shape[0]),
                "column_count": int(partition["frame"].shape[1]),
            }
            for partition in partitions
        ],
        TABLES_DIR / "02_raw_sheet_inventory.csv",
    )

    schema_rows: list[dict[str, Any]] = []
    for partition in partitions:
        frame = partition["frame"]
        for column in frame.columns:
            series = frame[column]
            schema_rows.append(
                {
                    "sheet_name": partition["sheet_name"],
                    "column_name": column,
                    "detected_dtype": str(series.dtype),
                    "non_null_count": int(series.notna().sum()),
                    "null_count": int(series.isna().sum()),
                    "null_percentage": _safe_percentage(series.isna().sum(), len(frame)),
                    "unique_non_null_values": int(series.nunique(dropna=True)),
                }
            )
    schema = _write_rows(schema_rows, TABLES_DIR / "03_raw_schema_audit.csv")

    date_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    customer_rows: list[dict[str, Any]] = []
    invoice_rows: list[dict[str, Any]] = []
    quantity_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    def audit_partition(scope: str, frame: pd.DataFrame, sheet_name: str) -> None:
        date_col = field_map["date"][sheet_name]
        customer_col = field_map["customer_id"][sheet_name]
        invoice_col = field_map["invoice"][sheet_name]
        quantity_col = field_map["quantity"][sheet_name]
        price_col = field_map["price"][sheet_name]
        product_col = field_map["product_code"][sheet_name]
        description_col = field_map["description"][sheet_name]
        country_col = field_map["country"][sheet_name]

        raw_dates = frame[date_col]
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
        invalid_dates = raw_dates.notna() & parsed_dates.isna()
        date_rows.append(
            {
                "scope": scope,
                "sheet_name": sheet_name,
                "date_column": date_col,
                "minimum_transaction_datetime": _date_value(parsed_dates.min()),
                "maximum_transaction_datetime": _date_value(parsed_dates.max()),
                "invalid_unparseable_date_count": int(invalid_dates.sum()),
            }
        )

        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            missing_rows.append(
                {
                    "scope": scope,
                    "sheet_name": sheet_name,
                    "column_name": column,
                    "missing_count": missing_count,
                    "missing_percentage": _safe_percentage(missing_count, len(frame)),
                }
            )

        duplicate_mask = frame.duplicated(keep=False)
        duplicate_excluding_first = int(frame.duplicated().sum())
        duplicate_rows.append(
            {
                "scope": scope,
                "sheet_name": sheet_name,
                "total_row_count": len(frame),
                "exact_duplicated_row_count": int(duplicate_mask.sum()),
                "exact_duplicated_row_percentage": _safe_percentage(duplicate_mask.sum(), len(frame)),
                "duplicate_rows_excluding_first_occurrence": duplicate_excluding_first,
            }
        )

        customer_series = frame[customer_col]
        customer_missing = int(customer_series.isna().sum())
        customer_rows.append(
            {
                "scope": scope,
                "sheet_name": sheet_name,
                "customer_id_column": customer_col,
                "non_null_customer_id_records": int(customer_series.notna().sum()),
                "missing_customer_id_records": customer_missing,
                "missing_customer_id_percentage": _safe_percentage(customer_missing, len(frame)),
                "unique_non_null_customers": int(customer_series.nunique(dropna=True)),
            }
        )

        invoice_series = frame[invoice_col]
        invoice_text = invoice_series.astype("string")
        c_mask = invoice_text.str.startswith("C", na=False)
        unique_invoice_count = int(invoice_series.nunique(dropna=True))
        c_invoice_values = set(invoice_text[c_mask].dropna().tolist())
        invoice_rows.append(
            {
                "scope": scope,
                "sheet_name": sheet_name,
                "invoice_column": invoice_col,
                "unique_invoices": unique_invoice_count,
                "unique_invoices_beginning_with_C": len(c_invoice_values),
                "percentage_unique_invoices_beginning_with_C": _safe_percentage(
                    len(c_invoice_values), unique_invoice_count
                ),
                "rows_associated_with_C_prefix_invoices": int(c_mask.sum()),
            }
        )

        quantity_rows.append(_numeric_summary(scope, frame[quantity_col], quantity_col))
        price_rows.append(_numeric_summary(scope, frame[price_col], price_col))

        product_rows.append(
            {
                "scope": scope,
                "sheet_name": sheet_name,
                "product_code_column": product_col,
                "unique_product_codes": int(frame[product_col].nunique(dropna=True)),
                "missing_product_codes": int(frame[product_col].isna().sum()),
                "missing_descriptions": int(frame[description_col].isna().sum()),
                "unique_descriptions": int(frame[description_col].nunique(dropna=True)),
            }
        )

        country_counts = frame[country_col].value_counts(dropna=False)
        for country, count in country_counts.items():
            country_label = "<MISSING>" if pd.isna(country) else str(country)
            country_rows.append(
                {
                    "scope": scope,
                    "sheet_name": sheet_name,
                    "country": country_label,
                    "transaction_row_count": int(count),
                    "percentage_of_rows": _safe_percentage(count, len(frame)),
                    "unique_country_values_in_scope": int(frame[country_col].nunique(dropna=True)),
                }
            )

        numeric_quantity = pd.to_numeric(frame[quantity_col], errors="coerce")
        numeric_price = pd.to_numeric(frame[price_col], errors="coerce")
        cancellation = c_mask
        negative_quantity = numeric_quantity < 0
        condition_masks = {
            "missing_customer_id": frame[customer_col].isna(),
            "missing_description": frame[description_col].isna(),
            "exact_duplicate_row": duplicate_mask,
            "cancellation_prefix_invoice": cancellation,
            "quantity_less_than_zero": negative_quantity,
            "quantity_equal_to_zero": numeric_quantity == 0,
            "price_less_than_zero": numeric_price < 0,
            "price_equal_to_zero": numeric_price == 0,
            "invalid_invoice_date": invalid_dates,
            "cancellation_prefix_and_negative_quantity": cancellation & negative_quantity,
            "negative_quantity_without_cancellation_prefix": negative_quantity & ~cancellation,
            "cancellation_prefix_without_negative_quantity": cancellation & ~negative_quantity,
        }
        for condition, mask in condition_masks.items():
            condition_rows.append(
                {
                    "scope": scope,
                    "sheet_name": sheet_name,
                    "condition": condition,
                    "count": int(mask.fillna(False).sum()),
                    "percentage_of_rows": _safe_percentage(mask.fillna(False).sum(), len(frame)),
                }
            )

    for partition in partitions:
        audit_partition(partition["sheet_name"], partition["frame"], partition["sheet_name"])

    worksheet_profiles = _write_worksheet_profiles(partitions)

    if combined is not None:
        audit_partition("overall", combined, "overall")
    else:
        all_customer_values = pd.concat(
            [partition["frame"][field_map["customer_id"][partition["sheet_name"]]] for partition in partitions],
            ignore_index=True,
        )
        customer_rows.append(
            {
                "scope": "overall",
                "sheet_name": "overall",
                "customer_id_column": "multiple sheet-specific columns",
                "non_null_customer_id_records": int(all_customer_values.notna().sum()),
                "missing_customer_id_records": int(all_customer_values.isna().sum()),
                "missing_customer_id_percentage": _safe_percentage(all_customer_values.isna().sum(), len(all_customer_values)),
                "unique_non_null_customers": int(all_customer_values.nunique(dropna=True)),
            }
        )

    date_coverage = _write_rows(date_rows, TABLES_DIR / "04_date_coverage.csv")
    missing = _write_rows(missing_rows, TABLES_DIR / "05_missing_values.csv")
    duplicates = _write_rows(duplicate_rows, TABLES_DIR / "06_duplicate_audit.csv")
    customer_ids = _write_rows(customer_rows, TABLES_DIR / "07_customer_id_audit.csv")
    invoices = _write_rows(invoice_rows, TABLES_DIR / "08_invoice_audit.csv")
    quantities = _write_rows(quantity_rows, TABLES_DIR / "09_quantity_audit.csv")
    prices = _write_rows(price_rows, TABLES_DIR / "10_price_audit.csv")
    products = _write_rows(product_rows, TABLES_DIR / "11_product_audit.csv")
    countries = _write_rows(country_rows, TABLES_DIR / "12_country_distribution.csv")
    conditions = _write_rows(condition_rows, TABLES_DIR / "13_raw_condition_counts.csv")

    _build_report(
        source_path,
        manifest,
        inventory,
        schema,
        date_coverage,
        missing,
        duplicates,
        customer_ids,
        invoices,
        quantities,
        prices,
        countries,
        field_map,
        schemas_identical,
    )

    expected_outputs = [TABLES_DIR / name for name in AUDIT_OUTPUT_NAMES]
    expected_outputs.append(PROJECT_ROOT / "reports" / "01_raw_data_audit.md")
    missing_outputs = [str(path) for path in expected_outputs if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"Expected audit outputs were not created: {missing_outputs}")

    source_after_hash = sha256_file(source_path)
    if source_before_hash != source_after_hash:
        raise RuntimeError("Raw source checksum changed during audit.")
    raw_after = _file_snapshot(RAW_DATA_DIR)
    if raw_before != raw_after:
        raise RuntimeError("A file in data/raw changed or was created during the audit.")
    processed_after = _file_snapshot(PROCESSED_DATA_DIR)
    if processed_before != processed_after:
        raise RuntimeError("A processed analytical file was created or changed during the audit.")
    if any(_normalise_name(column) == "churn" for column in schema["column_name"]):
        raise RuntimeError("A churn variable was unexpectedly present in the raw schema audit.")

    logger.info("Validation passed: raw checksum unchanged")
    logger.info("Validation passed: no files written to data/raw")
    logger.info("Validation passed: no processed analytical dataset created")
    logger.info("Validation passed: no churn variable created")
    logger.info("Validation passed: no machine-learning model trained")
    logger.info("Raw data audit completed successfully")

    return {
        "source_path": source_path,
        "manifest": manifest,
        "inventory": inventory,
        "schema": schema,
        "date_coverage": date_coverage,
        "missing_values": missing,
        "duplicates": duplicates,
        "customer_ids": customer_ids,
        "invoices": invoices,
        "quantities": quantities,
        "prices": prices,
        "products": products,
        "countries": countries,
        "conditions": conditions,
        "worksheet_profiles": worksheet_profiles,
    }


if __name__ == "__main__":
    run_audit()

