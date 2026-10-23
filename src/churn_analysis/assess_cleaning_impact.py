"""Step 3: quantify candidate cleaning-rule impacts without applying them."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR


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
CANONICAL_PATH = PROCESSED_DATA_DIR / "01_source_reconciled.parquet"
MANIFEST_PATH = TABLES_DIR / "01_raw_file_manifest.csv"
STEP2_SUMMARY_PATH = TABLES_DIR / "20_source_reconciliation_summary.csv"
STEP2_MISSINGNESS_PATH = TABLES_DIR / "24_customer_id_missingness_audit.csv"
STEP2_SPECIAL_PATH = TABLES_DIR / "28_special_stockcode_audit.csv"


def safe_percentage(numerator: float, denominator: float) -> float:
    """Return a percentage, using zero when the denominator is zero."""

    return float(numerator / denominator * 100) if denominator else 0.0


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_raw_workbook() -> Path:
    """Locate the raw workbook used by the earlier audit steps."""

    candidates = [RAW_DATA_DIR / "online_retail_II.xlsx"]
    candidates.extend(sorted(RAW_DATA_DIR.glob("*.xlsx")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No raw workbook found under {RAW_DATA_DIR}")


def validate_inputs() -> tuple[pd.DataFrame, str, str]:
    """Validate the canonical parquet, Step 2 metadata, and raw checksum."""

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(f"Canonical input is missing: {CANONICAL_PATH}")
    if not MANIFEST_PATH.exists() or not STEP2_SUMMARY_PATH.exists():
        raise FileNotFoundError("Step 2 manifest or reconciliation summary is missing.")

    source_path = find_raw_workbook()
    manifest = pd.read_csv(MANIFEST_PATH)
    expected_raw_hash = str(manifest.iloc[0]["sha256"]).lower()
    observed_raw_hash = sha256_file(source_path)
    if observed_raw_hash != expected_raw_hash:
        raise RuntimeError(
            f"Raw checksum changed: expected {expected_raw_hash}, observed {observed_raw_hash}."
        )

    summary = pd.read_csv(STEP2_SUMMARY_PATH).set_index("metric")["value"]
    frame = pd.read_parquet(CANONICAL_PATH, engine="pyarrow")
    if len(frame) != int(float(summary["row_count_after_source_reconciliation"])):
        raise RuntimeError("Canonical row count does not match Step 2 reconciliation summary.")
    if list(frame.columns) != RAW_COLUMNS + ["SourceSheet"]:
        raise RuntimeError(f"Unexpected canonical schema: {list(frame.columns)}")
    expected_min = pd.Timestamp(summary["earliest_InvoiceDate"])
    expected_max = pd.Timestamp(summary["latest_InvoiceDate"])
    if frame["InvoiceDate"].min() != expected_min or frame["InvoiceDate"].max() != expected_max:
        raise RuntimeError("Canonical date range does not match Step 2 reconciliation summary.")
    if any("churn" in re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in frame.columns):
        raise RuntimeError("Canonical input unexpectedly contains a churn label or feature.")
    return frame, sha256_file(CANONICAL_PATH), observed_raw_hash


def add_diagnostic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add in-memory masks and monetary measures without changing the input file."""

    data = frame.copy()
    invoice_text = data["Invoice"].astype("string")
    description_text = data["Description"].fillna("").astype("string").str.upper()
    stockcode_text = data["StockCode"].fillna("").astype("string").str.upper()
    data["_identified"] = data["Customer ID"].notna()
    data["_invoice_present"] = data["Invoice"].notna() & invoice_text.str.strip().ne("")
    data["_c_prefix"] = invoice_text.str.upper().str.startswith("C", na=False)
    data["_gross_value"] = data["Quantity"] * data["Price"]
    data["_positive_purchase"] = data["_identified"] & data["Quantity"].gt(0) & data["Price"].gt(0)
    data["_positive_value"] = data["_gross_value"].where(data["_positive_purchase"], 0.0)
    data["_positive_quantity"] = data["Quantity"].where(data["_positive_purchase"], 0)
    data["_duplicate"] = data.duplicated(subset=RAW_COLUMNS, keep=False)
    data["_duplicate_extra"] = data.duplicated(subset=RAW_COLUMNS, keep="first")
    data["_description_upper"] = description_text
    data["_stockcode_upper"] = stockcode_text
    return data


def write_missing_customer_impact(data: pd.DataFrame) -> pd.DataFrame:
    """Profile rows that cannot be linked to an identifiable customer."""

    missing = data.loc[~data["_identified"]].copy()
    total_rows = len(data)
    rows = []

    def add(dimension: str, category: str, metric: str, value: float, percentage: float = np.nan, notes: str = "") -> None:
        rows.append({"section": "summary", "dimension": dimension, "category": category, "metric": metric,
                     "value": value, "percentage": percentage, "notes": notes})

    add("overall", "all rows", "total_rows", total_rows)
    add("overall", "identified", "rows_retained", int(data["_identified"].sum()), safe_percentage(data["_identified"].sum(), total_rows))
    add("overall", "unidentified", "rows_excluded_from_customer_level_modeling", len(missing), safe_percentage(len(missing), total_rows),
        "Rows remain in the transaction history but cannot be linked to a customer record.")
    add("overall", "unidentified", "unique_invoices", missing.loc[missing["_invoice_present"], "Invoice"].nunique())
    add("overall", "unidentified", "quantity_total", missing["Quantity"].sum())
    add("overall", "unidentified", "gross_value_signed", missing["_gross_value"].sum())
    add("overall", "unidentified", "positive_purchase_gross_value", missing["_positive_value"].sum())
    add("overall", "unidentified", "date_min", missing["InvoiceDate"].min().isoformat())
    add("overall", "unidentified", "date_max", missing["InvoiceDate"].max().isoformat())

    monthly = data.assign(month=data["InvoiceDate"].dt.to_period("M")).groupby("month", observed=True)
    monthly_missing = missing.assign(month=missing["InvoiceDate"].dt.to_period("M")).groupby("month", observed=True).size()
    for month, group in monthly:
        missing_rows = int(monthly_missing.get(month, 0))
        add("month", str(month), "unidentified_rows", missing_rows, safe_percentage(missing_rows, len(group)))
        add("month", str(month), "total_rows", len(group))

    country_total = data.groupby(data["Country"].fillna("[missing]"), dropna=False).size()
    country_missing = missing.groupby(missing["Country"].fillna("[missing]"), dropna=False).size()
    for country, count in country_missing.sort_values(ascending=False).items():
        add("country", str(country), "unidentified_rows", int(count), safe_percentage(count, country_total.get(country, 0)))
        add("country", str(country), "total_rows", int(country_total.get(country, 0)))

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "31_missing_customer_impact.csv", index=False)
    return result


def write_duplicate_impact(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Document the already deduplicated primary canonical source."""

    retained = data.loc[~data["_duplicate_extra"]].copy()
    scenarios = {"primary_deduplicated": data}
    summary_rows = []
    for scenario, frame in scenarios.items():
        identified = frame["_identified"]
        summary_rows.extend([
            {"scenario": scenario, "metric": "total_rows", "value": len(frame), "notes": "All canonical rows in the temporary scenario."},
            {"scenario": scenario, "metric": "identified_customer_rows", "value": int(identified.sum()), "notes": "Rows with a non-null Customer ID."},
            {"scenario": scenario, "metric": "unique_customers", "value": frame.loc[identified, "Customer ID"].nunique(), "notes": "Unique non-null Customer IDs."},
            {"scenario": scenario, "metric": "unique_invoices", "value": frame.loc[frame["_invoice_present"], "Invoice"].nunique(), "notes": "Unique non-null invoice identifiers; duplicates are not expected to change this."},
            {"scenario": scenario, "metric": "aggregate_positive_purchase_monetary_value", "value": frame["_positive_value"].sum(), "notes": "Identified rows with Quantity > 0 and Price > 0."},
            {"scenario": scenario, "metric": "aggregate_positive_purchase_quantity", "value": frame["_positive_quantity"].sum(), "notes": "Identified rows with Quantity > 0 and Price > 0."},
            {"scenario": scenario, "metric": "duplicate_extra_occurrences", "value": int(frame["_duplicate_extra"].sum()), "notes": "Primary canonical source after keep-first duplicate removal; expected zero."},
        ])

    before = data.loc[data["_identified"]].groupby("Customer ID").agg(
        rows_before=("Customer ID", "size"), value_before=("_positive_value", "sum"), quantity_before=("_positive_quantity", "sum")
    )
    after = retained.loc[retained["_identified"]].groupby("Customer ID").agg(
        rows_after=("Customer ID", "size"), value_after=("_positive_value", "sum"), quantity_after=("_positive_quantity", "sum")
    )
    customers = before.join(after, how="outer").fillna(0)
    for column in ["rows_before", "rows_after", "quantity_before", "quantity_after"]:
        customers[column] = customers[column].astype(int)
    customers["rows_removed"] = customers["rows_before"] - customers["rows_after"]
    customers["value_change"] = customers["value_after"] - customers["value_before"]
    customers["quantity_change"] = customers["quantity_after"] - customers["quantity_before"]
    customers["row_count_change"] = customers["rows_after"] - customers["rows_before"]
    customers["value_relative_change_pct"] = np.where(customers["value_before"].ne(0), customers["value_change"] / customers["value_before"] * 100, np.nan)
    customers["quantity_relative_change_pct"] = np.where(customers["quantity_before"].ne(0), customers["quantity_change"] / customers["quantity_before"] * 100, np.nan)
    customers["row_count_relative_change_pct"] = np.where(customers["rows_before"].ne(0), customers["row_count_change"] / customers["rows_before"] * 100, np.nan)
    customers["affected"] = customers["rows_removed"].gt(0)
    affected = customers.loc[customers["affected"]].copy()
    affected.insert(0, "Customer ID", affected.index)
    affected = affected.reset_index(drop=True)

    affected_count = len(affected)
    identified_customers = data.loc[data["_identified"], "Customer ID"].nunique()
    summary_rows.extend([
        {"scenario": "primary_check", "metric": "customers_affected", "value": affected_count, "notes": "Expected zero because duplicate extras were removed before the canonical source was built."},
        {"scenario": "primary_check", "metric": "percentage_of_identified_customers_affected", "value": safe_percentage(affected_count, identified_customers), "notes": "Expected zero in the deduplicated primary canonical source."},
        {"scenario": "primary_check", "metric": "unique_invoice_count_change", "value": retained.loc[retained["_invoice_present"], "Invoice"].nunique() - data.loc[data["_invoice_present"], "Invoice"].nunique(), "notes": "Expected zero because the canonical source contains no duplicate extras."},
    ])
    for metric_name in ["value_relative_change_pct", "quantity_relative_change_pct", "row_count_relative_change_pct"]:
        distribution = affected[metric_name].dropna()
        for statistic, value in distribution.describe(percentiles=[0.25, 0.5, 0.75]).items():
            summary_rows.append({
                "scenario": "affected_customer_distribution",
                "metric": metric_name + "_" + statistic,
                "value": float(value),
                "notes": "Expected-empty distribution in the deduplicated primary canonical source.",
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "32_duplicate_impact_summary.csv", index=False)
    affected.to_csv(TABLES_DIR / "33_duplicate_customer_impact.csv", index=False)
    return summary, affected


def write_cancellation_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize C-prefix cancellation activity among identified customers."""

    identified = data.loc[data["_identified"]].copy()
    cancellation = identified.loc[identified["_c_prefix"]]
    customer_counts = cancellation.groupby("Customer ID").size()
    all_customers = identified["Customer ID"].nunique()
    all_invoices = identified.loc[identified["_invoice_present"], "Invoice"].nunique()
    rows = [
        {"section": "summary", "metric": "identified_customers_with_at_least_one_C_prefix_invoice", "value": len(customer_counts), "percentage": safe_percentage(len(customer_counts), all_customers), "notes": "Cancellation-linked customers."},
        {"section": "summary", "metric": "identified_customers_without_C_prefix_invoice", "value": all_customers - len(customer_counts), "percentage": safe_percentage(all_customers - len(customer_counts), all_customers), "notes": "Identified customers with no C-prefix row."},
        {"section": "summary", "metric": "unique_C_prefix_invoices", "value": cancellation.loc[cancellation["_invoice_present"], "Invoice"].nunique(), "percentage": safe_percentage(cancellation.loc[cancellation["_invoice_present"], "Invoice"].nunique(), all_invoices), "notes": "Rate uses unique identified-customer invoice identifiers as denominator."},
        {"section": "summary", "metric": "cancellation_rows", "value": len(cancellation), "percentage": safe_percentage(len(cancellation), len(identified)), "notes": "Share of identified transaction rows."},
        {"section": "summary", "metric": "cancellation_rate_at_invoice_level_pct", "value": safe_percentage(cancellation.loc[cancellation["_invoice_present"], "Invoice"].nunique(), all_invoices), "percentage": np.nan, "notes": "Unique C-prefix invoices divided by unique identified invoices."},
        {"section": "summary", "metric": "cancellation_rate_by_customer_pct", "value": safe_percentage(len(customer_counts), all_customers), "percentage": np.nan, "notes": "Customers with at least one C-prefix invoice divided by identified customers."},
    ]
    distribution = customer_counts.value_counts().sort_index()
    for count, customers in distribution.items():
        rows.append({"section": "customer_count_distribution", "metric": "customers", "value": int(customers), "percentage": safe_percentage(customers, len(customer_counts)), "category": f"{count}_cancellation_rows", "notes": "Distribution among customers with at least one cancellation row."})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "34_customer_cancellation_profile.csv", index=False)
    return result


def description_is_adjustment(text: pd.Series) -> pd.Series:
    """Flag descriptions with an explicit administrative or inventory signal."""

    pattern = r"manual|adjust|damag|lost|missing|break|sample|postage|discount|bank charge|fee|mailout|sold as|wrong|error|crushed|wet|smashed|found"
    return text.str.contains(pattern, case=False, regex=True, na=False)


def write_non_c_negative_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Audit the 3,393 negative-quantity rows without a C-prefix invoice."""

    subset = data.loc[~data["_c_prefix"] & data["Quantity"].lt(0)].copy()
    if len(subset) != 3393:
        raise RuntimeError(f"Expected 3,393 non-C negative-Quantity rows; observed {len(subset)}.")
    rows = []
    rows.append({"section": "summary", "category": "all", "metric": "rows", "value": len(subset), "percentage": 100.0, "notes": "Required Step 2 exception population."})
    for label, mask in [("identified", subset["_identified"]), ("without_Customer_ID", ~subset["_identified"])]:
        rows.append({"section": "summary", "category": label, "metric": "rows", "value": int(mask.sum()), "percentage": safe_percentage(mask.sum(), len(subset)), "notes": "Customer ID availability."})
    rows.extend([
        {"section": "summary", "category": "identified", "metric": "unique_customers", "value": subset.loc[subset["_identified"], "Customer ID"].nunique(), "percentage": np.nan, "notes": "Unique identified customers affected."},
        {"section": "summary", "category": "all", "metric": "unique_invoices", "value": subset.loc[subset["_invoice_present"], "Invoice"].nunique(), "percentage": np.nan, "notes": "Unique invoice identifiers."},
        {"section": "summary", "category": "all", "metric": "quantity_total", "value": subset["Quantity"].sum(), "percentage": np.nan, "notes": "Signed quantity total."},
        {"section": "summary", "category": "all", "metric": "negative_price_rows", "value": int(subset["Price"].lt(0).sum()), "percentage": safe_percentage(subset["Price"].lt(0).sum(), len(subset)), "notes": "Negative Price is separate from negative Quantity."},
        {"section": "summary", "category": "all", "metric": "zero_price_rows", "value": int(subset["Price"].eq(0).sum()), "percentage": safe_percentage(subset["Price"].eq(0).sum(), len(subset)), "notes": "Zero Price rows."},
        {"section": "summary", "category": "all", "metric": "positive_price_rows", "value": int(subset["Price"].gt(0).sum()), "percentage": safe_percentage(subset["Price"].gt(0).sum(), len(subset)), "notes": "Positive Price rows."},
    ])
    adjustment_flag = description_is_adjustment(subset["_description_upper"])
    rows.append({"section": "summary", "category": "description_signal", "metric": "apparently_administrative_or_inventory_adjustment_rows", "value": int(adjustment_flag.sum()), "percentage": safe_percentage(adjustment_flag.sum(), len(subset)), "notes": "Transparent keyword screen; uncertain cases are not automatically classified."})

    for column, section, limit in [("StockCode", "stockcode_pattern", 25), ("Description", "description_pattern", 25), ("Country", "country_distribution", 50)]:
        values = subset[column].fillna("[missing]").astype(str).value_counts().head(limit)
        for value, count in values.items():
            rows.append({"section": section, "category": value, "metric": "rows", "value": int(count), "percentage": safe_percentage(count, len(subset)), "notes": "Top observed values."})
    for statistic, value in subset["Price"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).items():
        rows.append({"section": "price_distribution", "category": "all", "metric": statistic, "value": float(value), "percentage": np.nan, "notes": "Price distribution among non-C negative-Quantity rows."})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "35_non_c_negative_quantity_audit.csv", index=False)
    return result


def write_nonpositive_price_impact(data: pd.DataFrame) -> pd.DataFrame:
    """Quantify the customer-level footprint of Price <= 0 rows."""

    subset = data.loc[data["Price"].le(0)].copy()
    identified = subset["_identified"]
    identified_customers = subset.loc[identified, "Customer ID"].nunique()
    identified_all = data.loc[data["_identified"]]
    positive_customers = identified_all.loc[identified_all["_positive_purchase"], "Customer ID"].nunique()
    only_nonpositive = set(subset.loc[identified, "Customer ID"].unique()) - set(identified_all.loc[identified_all["_positive_purchase"], "Customer ID"].unique())
    rows = [
        {"section": "summary", "category": "all", "metric": "nonpositive_price_rows", "value": len(subset), "percentage": safe_percentage(len(subset), len(data)), "notes": "Price <= 0."},
        {"section": "summary", "category": "identified", "metric": "rows", "value": int(identified.sum()), "percentage": safe_percentage(identified.sum(), len(subset)), "notes": "Rows with non-null Customer ID."},
        {"section": "summary", "category": "unidentified", "metric": "rows", "value": int((~identified).sum()), "percentage": safe_percentage((~identified).sum(), len(subset)), "notes": "Rows without Customer ID."},
        {"section": "summary", "category": "identified", "metric": "unique_customers_affected", "value": identified_customers, "percentage": safe_percentage(identified_customers, data["Customer ID"].nunique()), "notes": "Share uses all identifiable customers."},
        {"section": "summary", "category": "all", "metric": "unique_invoices", "value": subset.loc[subset["_invoice_present"], "Invoice"].nunique(), "percentage": np.nan, "notes": "Unique invoice identifiers."},
        {"section": "summary", "category": "identified", "metric": "unique_invoices", "value": subset.loc[identified & subset["_invoice_present"], "Invoice"].nunique(), "percentage": np.nan, "notes": "Unique invoice identifiers with Customer ID."},
        {"section": "summary", "category": "all", "metric": "price_negative_rows", "value": int(subset["Price"].lt(0).sum()), "percentage": safe_percentage(subset["Price"].lt(0).sum(), len(subset)), "notes": "Price sign distribution."},
        {"section": "summary", "category": "all", "metric": "price_zero_rows", "value": int(subset["Price"].eq(0).sum()), "percentage": safe_percentage(subset["Price"].eq(0).sum(), len(subset)), "notes": "Price sign distribution."},
        {"section": "summary", "category": "all", "metric": "quantity_positive_rows", "value": int(subset["Quantity"].gt(0).sum()), "percentage": safe_percentage(subset["Quantity"].gt(0).sum(), len(subset)), "notes": "Quantity sign distribution."},
        {"section": "summary", "category": "all", "metric": "quantity_zero_rows", "value": int(subset["Quantity"].eq(0).sum()), "percentage": safe_percentage(subset["Quantity"].eq(0).sum(), len(subset)), "notes": "Quantity sign distribution."},
        {"section": "summary", "category": "all", "metric": "quantity_negative_rows", "value": int(subset["Quantity"].lt(0).sum()), "percentage": safe_percentage(subset["Quantity"].lt(0).sum(), len(subset)), "notes": "Quantity sign distribution."},
        {"section": "summary", "category": "all", "metric": "C_prefix_rows", "value": int(subset["_c_prefix"].sum()), "percentage": safe_percentage(subset["_c_prefix"].sum(), len(subset)), "notes": "Cancellation-prefix status."},
        {"section": "summary", "category": "all", "metric": "non_C_rows", "value": int((~subset["_c_prefix"]).sum()), "percentage": safe_percentage((~subset["_c_prefix"]).sum(), len(subset)), "notes": "Cancellation-prefix status."},
        {"section": "summary", "category": "all", "metric": "meaningful_positive_purchase_value", "value": subset["_positive_value"].sum(), "percentage": 0.0, "notes": "None can satisfy Price > 0, so these rows contribute no positive-purchase value."},
        {"section": "summary", "category": "identified", "metric": "customers_with_only_nonpositive_price_records", "value": len(only_nonpositive), "percentage": safe_percentage(len(only_nonpositive), identified_customers), "notes": "Estimated customers with no identified positive-purchase candidate row."},
        {"section": "summary", "category": "identified", "metric": "customers_with_positive_purchase_and_nonpositive_price_records", "value": len(set(subset.loc[identified, "Customer ID"].unique()) - only_nonpositive), "percentage": safe_percentage(len(set(subset.loc[identified, "Customer ID"].unique()) - only_nonpositive), identified_customers), "notes": "These customers retain other positive-purchase records if Price <= 0 rows are excluded later."},
    ]
    for column, section, limit in [("StockCode", "stockcode_pattern", 25), ("Description", "description_pattern", 25)]:
        values = subset[column].fillna("[missing]").astype(str).value_counts().head(limit)
        for value, count in values.items():
            rows.append({"section": section, "category": value, "metric": "rows", "value": int(count), "percentage": safe_percentage(count, len(subset)), "notes": "Top observed values."})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "36_nonpositive_price_customer_impact.csv", index=False)
    return result


def special_pattern_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Return transparent, overlapping diagnostic masks for special-code patterns."""

    code = data["_stockcode_upper"]
    desc = data["_description_upper"]
    return {
        "Manual": code.eq("M") | desc.str.contains(r"\bMANUAL\b", na=False),
        "Postage": code.isin(["POST", "DOT"]) | desc.str.contains(r"POSTAGE|CARRIAGE|SHIPPING", na=False),
        "Discount": code.eq("D") | desc.str.contains(r"\bDISCOUNT\b", na=False),
        "Sample": code.eq("S") | desc.str.contains(r"\bSAMPLES?\b", na=False),
        "dotcom-related": code.eq("DOT") | desc.str.contains(r"DOTCOM|DOT COM", na=False),
        "adjustments": desc.str.contains(r"ADJUST|MAILOUT|LOST|DAMAG|MISSING|BROKEN|WRONG|ERROR|SOLD AS", na=False),
        "gift-related": desc.str.contains(r"GIFT", na=False) | code.str.contains(r"GIFT", na=False),
        "bank/fee-related": code.str.contains(r"BANK|FEE|CHARGE", na=False) | desc.str.contains(r"BANK|FEE|CHARGE", na=False),
    }


def write_special_stockcode_impact(data: pd.DataFrame) -> pd.DataFrame:
    """Profile Step 2 special-code patterns without declaring them invalid."""

    total_positive_value = data["_positive_value"].sum()
    rows = []
    for pattern, mask in special_pattern_masks(data).items():
        subset = data.loc[mask]
        rows.extend([
            {"pattern": pattern, "metric": "row_count", "value": len(subset), "percentage": safe_percentage(len(subset), len(data)), "notes": "Patterns may overlap; counts are not intended to sum to the dataset total."},
            {"pattern": pattern, "metric": "identified_customers", "value": subset.loc[subset["_identified"], "Customer ID"].nunique(), "percentage": np.nan, "notes": "Unique non-null Customer IDs."},
            {"pattern": pattern, "metric": "unique_invoices", "value": subset.loc[subset["_invoice_present"], "Invoice"].nunique(), "percentage": np.nan, "notes": "Unique invoice identifiers."},
            {"pattern": pattern, "metric": "quantity_positive_rows", "value": int(subset["Quantity"].gt(0).sum()), "percentage": safe_percentage(subset["Quantity"].gt(0).sum(), len(subset)), "notes": "Quantity sign."},
            {"pattern": pattern, "metric": "quantity_negative_rows", "value": int(subset["Quantity"].lt(0).sum()), "percentage": safe_percentage(subset["Quantity"].lt(0).sum(), len(subset)), "notes": "Quantity sign."},
            {"pattern": pattern, "metric": "price_positive_rows", "value": int(subset["Price"].gt(0).sum()), "percentage": safe_percentage(subset["Price"].gt(0).sum(), len(subset)), "notes": "Price sign."},
            {"pattern": pattern, "metric": "price_nonpositive_rows", "value": int(subset["Price"].le(0).sum()), "percentage": safe_percentage(subset["Price"].le(0).sum(), len(subset)), "notes": "Price sign."},
            {"pattern": pattern, "metric": "gross_value_signed", "value": subset["_gross_value"].sum(), "percentage": np.nan, "notes": "Quantity multiplied by Price where numeric."},
            {"pattern": pattern, "metric": "positive_purchase_gross_value", "value": subset["_positive_value"].sum(), "percentage": safe_percentage(subset["_positive_value"].sum(), total_positive_value), "notes": "Share of total identified positive-purchase gross value."},
        ])
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "37_special_stockcode_impact.csv", index=False)
    return result


def write_transaction_roles(data: pd.DataFrame) -> pd.DataFrame:
    """Assign temporary diagnostic roles using documented single-role precedence."""

    role = pd.Series("Other / requires review", index=data.index, dtype="string")
    role.loc[data["_identified"] & data["Quantity"].gt(0) & data["Price"].gt(0)] = "Positive purchase candidate"
    role.loc[data["Price"].le(0)] = "Non-positive-price record"
    role.loc[(~data["_c_prefix"]) & data["Quantity"].lt(0)] = "Non-C negative adjustment"
    role.loc[data["_c_prefix"]] = "Cancellation"
    data["_primary_role"] = role
    rows = []
    for name, group in data.groupby("_primary_role", sort=False, observed=True):
        rows.append({"row_type": "primary_role", "role": name, "row_count": len(group), "identified_customers": group.loc[group["_identified"], "Customer ID"].nunique(), "unique_invoices": group.loc[group["_invoice_present"], "Invoice"].nunique(), "notes": "Precedence: Cancellation > Non-C negative adjustment > Non-positive-price record > Positive purchase candidate > Other/review; missing IDs remain an overlapping diagnostic flag."})
    flag_masks = {
        "Positive purchase candidate": data["_identified"] & data["Quantity"].gt(0) & data["Price"].gt(0),
        "Cancellation": data["_c_prefix"],
        "Non-C negative adjustment": ~data["_c_prefix"] & data["Quantity"].lt(0),
        "Non-positive-price record": data["Price"].le(0),
        "Unidentified-customer record": ~data["_identified"],
    }
    for name, mask in flag_masks.items():
        group = data.loc[mask]
        rows.append({"row_type": "overlapping_diagnostic_role", "role": name, "row_count": len(group), "identified_customers": group.loc[group["_identified"], "Customer ID"].nunique(), "unique_invoices": group.loc[group["_invoice_present"], "Invoice"].nunique(), "notes": "Overlapping diagnostic role; not a predictive feature and not a final cleaning rule."})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "38_transaction_role_summary.csv", index=False)
    return result


def write_decision_table(data: pd.DataFrame, duplicate_summary: pd.DataFrame, cancellation: pd.DataFrame, non_c: pd.DataFrame, nonprice: pd.DataFrame, special: pd.DataFrame) -> pd.DataFrame:
    """Write evidence-based recommendations after the approved duplicate rule."""

    def metric(scenario: str, name: str) -> float:
        values = duplicate_summary.loc[(duplicate_summary["scenario"] == scenario) & (duplicate_summary["metric"] == name), "value"]
        return float(values.iloc[0]) if not values.empty else np.nan

    identified_customers = data["Customer ID"].nunique()
    dup_affected = metric("primary_check", "customers_affected")
    dup_pct = metric("primary_check", "percentage_of_identified_customers_affected")
    c_customers = int(cancellation.loc[cancellation["metric"] == "identified_customers_with_at_least_one_C_prefix_invoice", "value"].iloc[0])
    nonc_customers = int(non_c.loc[(non_c["category"] == "identified") & (non_c["metric"] == "unique_customers"), "value"].iloc[0])
    nonprice_customers = int(nonprice.loc[(nonprice["category"] == "identified") & (nonprice["metric"] == "unique_customers_affected"), "value"].iloc[0])
    total_positive = data["_positive_value"].sum()
    special_rows = int(special.loc[special["metric"] == "row_count", "value"].sum())
    rows = [
        {"issue": "Missing Customer ID", "evidence": f"{int((~data['_identified']).sum()):,} rows ({safe_percentage((~data['_identified']).sum(), len(data)):.2f}%) and {data.loc[~data['_identified'], 'Invoice'].nunique():,} invoices lack a customer link.", "customers_affected": "Not identifiable for excluded rows", "invoices_affected": int(data.loc[~data['_identified'], "Invoice"].nunique()), "metric_impact": "Customer-level activity, value, and frequency cannot be assigned without an identifier.", "recommended_treatment": "Exclude from customer-level modeling tables; retain in transaction-level audit totals and report sensitivity.", "confidence": "High", "reason": "A missing identifier prevents a defensible customer-to-transaction link; no artificial IDs should be assigned."},
        {"issue": "Cross-sheet overlap", "evidence": "Step 2 removed the verified second representation after within-sheet deduplication and the canonical validation passes.", "customers_affected": "Already resolved", "invoices_affected": "Already resolved", "metric_impact": "No new Step 3 action is indicated.", "recommended_treatment": "Retain the Step 2 reconciliation.", "confidence": "High", "reason": "The common interval was confirmed as exact full-row repetition after deduplication."},
        {"issue": "Exact duplicates", "evidence": f"The primary canonical source contains {int(metric('primary_deduplicated', 'duplicate_extra_occurrences')):,} removable extra occurrences after Step 2 keep-first processing.", "customers_affected": int(dup_affected), "invoices_affected": 0, "metric_impact": "The primary analysis has already removed exact duplicate extras; a duplicate-retained alternative can be evaluated later.", "recommended_treatment": "Use keep-first removal for the primary analysis and retain a documented duplicate-retained sensitivity alternative.", "confidence": "High", "reason": "This is the explicitly approved methodological change."},
        {"issue": "C-prefix cancellations", "evidence": f"{c_customers:,} identified customers have at least one cancellation-linked row; cancellation rows should not be treated as ordinary purchases.", "customers_affected": c_customers, "invoices_affected": int(cancellation.loc[cancellation["metric"] == "unique_C_prefix_invoices", "value"].iloc[0]), "metric_impact": "Including cancellations in purchase metrics can distort frequency, quantity, and value.", "recommended_treatment": "Exclude from ordinary positive-purchase metrics but preserve separately for cancellation-behavior features and audit reporting.", "confidence": "High", "reason": "The C-prefix is a documented transaction signal and is customer-linked in this population."},
        {"issue": "Non-C negative Quantity", "evidence": f"3,393 rows affect {nonc_customers:,} identifiable customers; {int(non_c.loc[(non_c['category'] == 'description_signal') & (non_c['metric'] == 'apparently_administrative_or_inventory_adjustment_rows'), 'value'].iloc[0]):,} rows have an apparent administrative/inventory description signal.", "customers_affected": nonc_customers, "invoices_affected": int(non_c.loc[(non_c["category"] == "all") & (non_c["metric"] == "unique_invoices"), "value"].iloc[0]), "metric_impact": "Negative quantities can reduce ordinary item totals and may represent returns, adjustments, or data-entry events.", "recommended_treatment": "Exclude from ordinary positive-purchase metrics provisionally, retain in transaction history, and review uncertain patterns before finalizing.", "confidence": "Moderate", "reason": "The population is small and heterogeneous; keyword evidence is diagnostic, not proof of a single meaning."},
        {"issue": "Price <= 0", "evidence": f"{len(data.loc[data['Price'].le(0)]):,} rows affect {nonprice_customers:,} identifiable customers; they contribute no meaningful positive-purchase value under Quantity > 0 and Price > 0.", "customers_affected": nonprice_customers, "invoices_affected": int(nonprice.loc[(nonprice["category"] == "all") & (nonprice["metric"] == "unique_invoices"), "value"].iloc[0]), "metric_impact": "They should not contribute to product-purchase value; excluding them could remove only-record customers from a future purchase table.", "recommended_treatment": "Exclude from positive-purchase value and product-purchase metrics; retain separately for administrative/exception analysis and document customers with no positive purchase.", "confidence": "High", "reason": "Price <= 0 cannot generate a positive purchase value, while the remaining records may still be analytically relevant."},
        {"issue": "Special StockCodes", "evidence": f"Diagnostic pattern screens identify {special_rows:,} overlapping pattern-row observations; pattern gross values and customer footprints are reported separately, with total positive-purchase value {total_positive:,.2f}.", "customers_affected": "Pattern-specific; see table 37", "invoices_affected": "Pattern-specific; see table 37", "metric_impact": "Administrative/service records can distort merchandise metrics, but not every flagged record is invalid.", "recommended_treatment": "Keep in general transaction history; exclude only clearly administrative categories from merchandise metrics after documented review; retain uncertain patterns for sensitivity analysis.", "confidence": "Moderate", "reason": "Description and StockCode signals are informative but overlapping and not definitive for every row."},
    ]
    result = pd.DataFrame(rows, columns=["issue", "evidence", "customers_affected", "invoices_affected", "metric_impact", "recommended_treatment", "confidence", "reason"])
    result.to_csv(TABLES_DIR / "39_cleaning_impact_decision_table.csv", index=False)
    return result


def format_value(value: object) -> str:
    """Format report values compactly."""

    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Render a small DataFrame as a Markdown table."""

    selected = frame.loc[:, list(columns)].copy()
    lines = ["| " + " | ".join(selected.columns) + " |", "| " + " | ".join("---" for _ in selected.columns) + " |"]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(data: pd.DataFrame, duplicate_summary: pd.DataFrame, duplicate_customers: pd.DataFrame, cancellation: pd.DataFrame, non_c: pd.DataFrame, nonprice: pd.DataFrame, roles: pd.DataFrame, decisions: pd.DataFrame, raw_hash: str) -> None:
    """Write the management-oriented Step 3 assessment report."""

    identified = int(data["_identified"].sum())
    total_customers = data.loc[data["_identified"], "Customer ID"].nunique()
    dup_a = duplicate_summary.loc[duplicate_summary["scenario"] == "primary_deduplicated"].set_index("metric")["value"]
    dup_distribution = duplicate_summary.loc[duplicate_summary["scenario"] == "affected_customer_distribution"].set_index("metric")["value"]
    c_customers = int(cancellation.loc[cancellation["metric"] == "identified_customers_with_at_least_one_C_prefix_invoice", "value"].iloc[0])
    non_c_customers = int(non_c.loc[(non_c["category"] == "identified") & (non_c["metric"] == "unique_customers"), "value"].iloc[0])
    nonprice_customers = int(nonprice.loc[(nonprice["category"] == "identified") & (nonprice["metric"] == "unique_customers_affected"), "value"].iloc[0])
    role_report = roles.loc[roles["row_type"] == "primary_role", ["role", "row_count", "identified_customers", "unique_invoices"]]

    report = f"""# Step 3 — Cleaning Impact Assessment

## Executive conclusion

This assessment measures the consequences of candidate cleaning rules before final cleaning. The canonical source-reconciled dataset contains **{len(data):,} rows**, **{total_customers:,} identifiable customers**, and spans **{data['InvoiceDate'].min():%Y-%m-%d} to {data['InvoiceDate'].max():%Y-%m-%d}**. No rule was executed in this step.

The main management implication is that transaction-level history should remain available for audit and sensitivity analysis, while customer-level purchase measures should be built only from records that can be linked to an identified customer and that represent positive-priced positive-quantity purchases. Cancellations and administrative exceptions should be preserved separately rather than silently treated as ordinary purchases.

## A. Missing Customer ID

{len(data.loc[~data['_identified']]):,} rows ({safe_percentage((~data['_identified']).sum(), len(data)):.2f}%) have no Customer ID, representing {data.loc[~data['_identified'], 'Invoice'].nunique():,} invoice identifiers. The remaining **{identified:,} rows ({safe_percentage(identified, len(data)):.2f}%)** are identifiable and represent **{total_customers:,} customers**. Unidentified rows span **{data.loc[~data['_identified'], 'InvoiceDate'].min():%Y-%m-%d} to {data.loc[~data['_identified'], 'InvoiceDate'].max():%Y-%m-%d}** and have signed quantity total **{data.loc[~data['_identified'], 'Quantity'].sum():,.0f}**.

These rows cannot be assigned to a customer-level modeling record without inventing an identifier. The recommendation is therefore to exclude them from customer-level modeling tables, while retaining them in transaction-level totals and reporting their monthly and country distribution in `31_missing_customer_impact.csv`.

## B. Exact duplicates

The primary source has already removed duplicate occurrences after the first using the original eight transaction fields. The table below documents the current primary state; a duplicate-retained alternative is reserved for later sensitivity analysis.

{markdown_table(duplicate_summary.loc[duplicate_summary['scenario'].eq('primary_deduplicated'), ['scenario', 'metric', 'value']], ['scenario', 'metric', 'value'])}

The primary canonical source contains **{int(dup_a['duplicate_extra_occurrences']):,} removable extra occurrences** and therefore has no remaining exact duplicate extras. The duplicate-retained alternative is not used for primary counts and may be compared later if required.

## C. Customer-linked cancellation profile

There are **{c_customers:,} identifiable customers** with at least one C-prefix invoice. C-prefix rows are not ordinary purchases: including them in purchase metrics can distort frequency, item quantity, and monetary behavior. The recommendation is to exclude them from ordinary positive-purchase metrics but preserve them as a separate cancellation-behavior signal for later consideration. The cancellation table includes invoice-level and customer-level rates plus the distribution of cancellation counts.

## D. Non-C negative Quantity

The required exception population contains **3,393 rows**, affecting **{non_c_customers:,} identifiable customers** and {int(non_c.loc[(non_c['category'] == 'all') & (non_c['metric'] == 'unique_invoices'), 'value'].iloc[0]):,} invoices. A transparent description screen finds **{int(non_c.loc[(non_c['category'] == 'description_signal') & (non_c['metric'] == 'apparently_administrative_or_inventory_adjustment_rows'), 'value'].iloc[0]):,} rows** with an apparent administrative or inventory-adjustment signal. This is evidence for review, not an automatic classification. The recommendation is to exclude these rows from ordinary positive-purchase measures provisionally, retain them in transaction history, and review uncertain cases before finalizing.

## E. Price <= 0

There are **{len(data.loc[data['Price'].le(0)]):,} rows** with Price <= 0, affecting **{nonprice_customers:,} identifiable customers**. They contribute **0** to the positive-purchase value measure because that measure requires Quantity > 0 and Price > 0. They should therefore be excluded from merchandise purchase-value metrics, while being retained separately for exception analysis. The table estimates customers whose identified records contain no positive-purchase candidate and would therefore have no record in a future positive-purchase table.

## F. Special StockCodes

Manual, postage, discount, sample, dotcom, adjustment, gift, and bank/fee signals were profiled using overlapping diagnostic screens. The screens are not validity judgments: a special code can be customer-linked and have a positive price, and an ordinary-looking code can still represent an exception. The recommendation is to retain the full history, exclude only clearly administrative categories from merchandise metrics after documentation, and sensitivity-test uncertain patterns. See `37_special_stockcode_impact.csv` for row counts, customer/invoice footprints, quantity and price signs, and positive-purchase value shares.

## G. Candidate transaction roles

The temporary primary-role counts below use this precedence: **Cancellation → Non-C negative adjustment → Non-positive-price record → Positive purchase candidate → Other/requires review**. Diagnostic flags are also reported because categories can overlap, especially with missing Customer IDs. These roles are not predictive features and are not final cleaning rules.

{markdown_table(role_report, ['role', 'row_count', 'identified_customers', 'unique_invoices'])}

## H. Recommended treatment and confidence

{markdown_table(decisions[['issue', 'recommended_treatment', 'confidence']], ['issue', 'recommended_treatment', 'confidence'])}

Confidence is **High** where the record cannot support a defensible customer link or positive purchase value. Confidence is **Moderate** where the observed pattern is clear but its business meaning remains ambiguous, especially exact duplicates, non-C negative quantities, and special StockCodes.

## Methodological ambiguity remaining

The primary analysis now uses keep-first removal for exact eight-field duplicates. The remaining ambiguity is whether a later sensitivity analysis should compare against a duplicate-retained alternative. Additional ambiguity remains for non-C negative quantities and overlapping special-code signals: descriptions suggest administrative behavior in some cases, but the dataset does not prove the business meaning of every row. These issues should be handled through documented review and sensitivity analysis, not silent deletion.

## Scope validation

- Source-reconciled row count and Step 2 date range were validated.
- Raw workbook SHA-256 remained `{raw_hash}`.
- The canonical parquet reflects the approved permanent within-sheet duplicate-removal rule; no later cleaning rule was applied here.
- No churn definition, observation/outcome window, predictive feature set, train/test split, or machine-learning model was created.

**The canonical source-reconciled dataset was read-only during this audit; the approved within-sheet keep-first rule had already been applied in Step 2. No churn definition, predictive feature set, train/test split, encoding, scaling, or machine-learning model was created.**
"""
    report_path = PROJECT_ROOT / "reports" / "03_cleaning_impact_assessment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def validate_scope(canonical_hash_before: str, raw_hash: str) -> None:
    """Confirm the canonical input and scope restrictions after analysis."""

    if sha256_file(CANONICAL_PATH) != canonical_hash_before:
        raise RuntimeError("Canonical source-reconciled parquet changed during Step 3.")
    if sha256_file(find_raw_workbook()) != raw_hash:
        raise RuntimeError("Raw workbook checksum changed during Step 3.")
    if any(path.exists() for path in [PROJECT_ROOT / "outputs" / "models" / "step3_model.pkl"]):
        raise RuntimeError("Unexpected Step 3 model artifact exists.")


def main() -> None:
    """Run the complete deterministic Step 3 assessment."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data, canonical_hash, raw_hash = validate_inputs()
    diagnostic = add_diagnostic_columns(data)
    missing = write_missing_customer_impact(diagnostic)
    duplicate_summary, duplicate_customers = write_duplicate_impact(diagnostic)
    cancellation = write_cancellation_profile(diagnostic)
    non_c = write_non_c_negative_audit(diagnostic)
    nonprice = write_nonpositive_price_impact(diagnostic)
    special = write_special_stockcode_impact(diagnostic)
    roles = write_transaction_roles(diagnostic)
    decisions = write_decision_table(diagnostic, duplicate_summary, cancellation, non_c, nonprice, special)
    write_report(diagnostic, duplicate_summary, duplicate_customers, cancellation, non_c, nonprice, roles, decisions, raw_hash)
    validate_scope(canonical_hash, raw_hash)
    print(json.dumps({
        "canonical_rows": len(diagnostic),
        "identified_rows": int(diagnostic["_identified"].sum()),
        "identified_customers": int(diagnostic.loc[diagnostic["_identified"], "Customer ID"].nunique()),
        "duplicate_extra_occurrences": int(diagnostic["_duplicate_extra"].sum()),
        "duplicate_affected_customers": int(len(duplicate_customers)),
        "cancellation_rows": int(diagnostic["_c_prefix"].sum()),
        "non_c_negative_rows": int((~diagnostic["_c_prefix"] & diagnostic["Quantity"].lt(0)).sum()),
        "nonpositive_price_rows": int(diagnostic["Price"].le(0).sum()),
        "canonical_sha256": canonical_hash,
        "raw_sha256": raw_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
