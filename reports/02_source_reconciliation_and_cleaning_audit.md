# Step 2 — Source Reconciliation and Cleaning Audit

## 1. Purpose

This step first removes additional exact full-row occurrences within each raw worksheet (`keep='first'`) using the eight original variables, then verifies and removes the second copy of the duplicated common worksheet interval. It audits the resulting source-reconciled transaction dataset without creating churn labels, RFM features, predictive features, train/test splits, or machine-learning models.

The source file was `data\raw\online_retail_II.xlsx` with SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. The raw workbook remained read-only.

## 2. Source reconciliation

The deterministic rule retained all deduplicated `Year 2009-2010` rows and retained only deduplicated `Year 2010-2011` rows with `InvoiceDate` strictly after the maximum `Year 2009-2010` timestamp (`2010-12-09T20:01:00`). Within-sheet duplicate extras were removed before this boundary rule. The common interval was confirmed as exact full-row repetition after deduplication, and `22,202` second-worksheet rows were removed.

| metric | value | notes |
| --- | --- | --- |
| raw_sheet_1_rows | 525461 | All rows retained from Year 2009-2010. |
| raw_sheet_2_rows | 541910 | Rows originally present in Year 2010-2011. |
| total_raw_rows_before_source_reconciliation | 1067371 | Sum of both raw worksheets. |
| within_sheet_duplicate_extras_removed | 12133 | Additional exact full-row occurrences removed before worksheet reconciliation. |
| deduplicated_sheet_1_rows | 518596 | Year 2009-2010 rows after keep='first' within-sheet deduplication. |
| deduplicated_sheet_2_rows | 536642 | Year 2010-2011 rows after keep='first' within-sheet deduplication. |
| cross_sheet_overlap_rows_after_deduplication | 22202 | Distinct exact full-row matches in the common interval after within-sheet deduplication. |
| cross_sheet_repeated_occurrences_removed | 22202 | Second worksheet rows in the verified common interval removed after deduplication. |
| cross_sheet_exact_matches_outside_common_interval | 0 | Exact cross-sheet matches involving retained post-boundary rows; expected zero. |
| row_count_after_source_reconciliation | 1033036 | Chronologically sorted source-reconciled rows. |
| earliest_InvoiceDate | 2009-12-01T07:45:00 | Observed in reconciled data. |
| latest_InvoiceDate | 2011-12-09T12:50:00 | Observed in reconciled data. |
| unique_Invoice_count | 53628 | Non-null unique invoice identifiers. |
| unique_non_null_Customer_ID_count | 5942 | Non-null unique customer identifiers. |

The reconciled Parquet file contains the original eight variables plus the technical `SourceSheet` provenance field. Because the source identifier columns contain mixed numeric/text values, `Invoice` and `StockCode` use stable textual Parquet storage while preserving their displayed source values. `SourceSheet` is not part of duplicate or cross-sheet equality comparisons.

## 3. Validation of the reconciled timeline

The reconciled data spans `2009-12-01T07:45:00` through `2011-12-09T12:50:00`. The preferred boundary rule was validated against multiplicity-aware full-row counts: the resulting transaction multiset matched exactly.

There were **135** calendar dates without transaction rows; these were reported and not imputed. Around the source boundary:

| calendar_date | reconciled_row_count | year_2009_2010_rows | year_2010_2011_retained_rows | boundary_gap_hours | notes |
| --- | --- | --- | --- | --- | --- |
| 2010-12-09 | 2881 | 2881 | 0 | 13.533333333333333 | Source boundary date requires review of the transition timing. |

## 4. Exact duplicates after primary deduplication

After within-sheet deduplication and cross-sheet reconciliation, `0` rows participate in exact duplicate groups under `duplicated(keep=False)`, and `0` are extra occurrences beyond the first under `duplicated(keep='first')`. The primary source therefore contains no exact duplicate signature under the eight original variables. The pre-removal per-sheet counts and retained row counts are in `outputs/tables/56_within_sheet_duplicate_audit_before_after.csv`.

Duplicate groups necessarily have the same Invoice, StockCode, and InvoiceDate because those are included in the exact eight-field signature. Their concentrations and representative largest groups are documented in `outputs/tables/23_duplicate_context_examples.csv`.

## 5. Missing Customer ID

Rows without Customer ID: `235,151` (22.7631%). They were retained. The monthly rate and comparisons of cancellation, quantity-sign, and price-sign behavior between missing and identified customers are in `outputs/tables/24_customer_id_missingness_audit.csv`.

## 6. Cancellations and negative quantities

| category | count | percentage | invoice_type | quantity_sign |
| --- | --- | --- | --- | --- |
| C-prefix AND Quantity < 0 | 19103 | 1.84920951447965 | C-prefix | negative |
| C-prefix AND Quantity >= 0 | 1 | 9.68020475569099e-05 | C-prefix | non-negative |
| non-C invoice AND Quantity < 0 | 3393 | 0.3284493473605954 | non-C | negative |
| non-C invoice AND Quantity >= 0 | 1010539 | 97.82224433611219 | non-C | non-negative |

The two exception categories were retained for review:

| category | count | percentage | invoice_type | quantity_sign |
| --- | --- | --- | --- | --- |
| C-prefix AND Quantity >= 0 | 1 | 9.68020475569099e-05 | C-prefix | non-negative |
| non-C invoice AND Quantity < 0 | 3393 | 0.3284493473605954 | non-C | negative |

No cancellation or negative-quantity rule was executed.

## 7. Non-positive prices

| record_type | price_category | count | percentage_of_reconciled_rows | StockCode | Description | Quantity | Customer ID | Country | cancellation_status | Price | pattern_row_count | example_Invoice | example_InvoiceDate | distinct_pattern_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | Price < 0 | 5 | 0.00048401023778454964 |  |  |  |  |  |  |  |  |  |  |  |
| summary | Price = 0 | 6014 | 0.5821675140072563 |  |  |  |  |  |  |  |  |  |  |  |
| summary | Price > 0 | 1027017 | 99.41734847575496 |  |  |  |  |  |  |  |  |  |  |  |

Distinct non-positive-price patterns and representative records are saved in `outputs/tables/27_nonpositive_price_audit.csv`. Zero-price rows were not automatically classified as invalid.

## 8. Special transaction/product codes

The StockCode audit reports code form, description signals, missing descriptions, negative quantities, non-positive prices, and cancellation-prefix counts. The most behaviorally unusual observed codes/patterns were:

| observed_stockcode | stockcode_pattern | description_signal | row_count | missing_description_count | negative_quantity_count | nonpositive_price_count | cancellation_prefix_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M | alphanumeric | manual | 1387 | 0 | 533 | 7 | 534 |
| 22423 | numeric_only |  | 4306 | 1 | 352 | 15 | 341 |
| POST | alphanumeric | postage | 2086 | 7 | 228 | 7 | 228 |
| 22138 | numeric_only |  | 2139 | 2 | 211 | 4 | 209 |
| 21232 | numeric_only |  | 2666 | 0 | 187 | 1 | 187 |
| D | alphanumeric | discount | 173 | 0 | 168 | 0 | 168 |
| 21843 | numeric_only |  | 1689 | 0 | 159 | 5 | 155 |
| 85123A | alphanumeric |  | 5653 | 0 | 136 | 8 | 134 |
| 79323W | alphanumeric |  | 478 | 1 | 128 | 2 | 126 |
| 21527 | numeric_only |  | 957 | 3 | 117 | 4 | 114 |
| S | alphanumeric | sample | 101 | 0 | 98 | 0 | 98 |
| 22197 | numeric_only |  | 2465 | 4 | 92 | 9 | 90 |
| 22960 | numeric_only |  | 1250 | 1 | 88 | 3 | 87 |
| 82483 | numeric_only |  | 1220 | 2 | 87 | 3 | 85 |
| 79323P | alphanumeric |  | 354 | 1 | 88 | 2 | 86 |

These are diagnostic patterns only; no merchandise/non-merchandise classification was executed.

## 9. Geographic profile

| Country | transaction_row_count | unique_non_null_customers | unique_non_null_invoices | percentage_of_rows |
| --- | --- | --- | --- | --- |
| United Kingdom | 948321 | 5410 | 49108 | 91.79941454121638 |
| EIRE | 17667 | 5 | 806 | 1.7102017741879276 |
| Germany | 17339 | 107 | 1095 | 1.6784507025892612 |
| France | 14025 | 95 | 746 | 1.3576487169856617 |
| Netherlands | 5137 | 23 | 250 | 0.4972721182998463 |
| Spain | 3754 | 41 | 188 | 0.3633948865286399 |
| Switzerland | 3175 | 22 | 123 | 0.307346500993189 |
| Belgium | 3110 | 29 | 183 | 0.30105436790198986 |
| Portugal | 2528 | 24 | 124 | 0.24471557622386828 |
| Australia | 1890 | 15 | 117 | 0.18295586988255977 |
| Channel Islands | 1646 | 14 | 79 | 0.15933617027867375 |
| Italy | 1507 | 17 | 92 | 0.14588068566826326 |
| Sweden | 1362 | 19 | 129 | 0.1318443887725113 |
| Norway | 1308 | 13 | 53 | 0.12661707820443818 |
| Cyprus | 1157 | 11 | 45 | 0.11199996902334479 |

The complete country profile includes transaction rows, unique non-null customers, and unique non-null invoices for every observed country.

## 10. Cleaning-decision matrix

| issue | observed_count | observed_percentage | evidence | potential_effect_on_customer_features | candidate_treatment | decision_status |
| --- | --- | --- | --- | --- | --- | --- |
| cross-sheet overlap | 22202 | 2.0800640077348924 | The common interval was confirmed as exact full-row repetition after within-sheet deduplication; the second representation was removed by the deterministic boundary rule. | Would double-count transactions, invoices, and customer activity if retained. | Retain Year 2009-2010 and Year 2010-2011 rows after the first-sheet maximum InvoiceDate. | Executed |
| within-source exact duplicates | 12133 | 1.1744992430079881 | Exact full-row duplicate extras were recomputed separately within each raw worksheet. | May inflate purchase frequency, quantity, revenue, or product counts. | Remove only additional occurrences with keep='first' before worksheet reconciliation; retain a duplicate-retained sensitivity alternative. | Executed |
| missing Customer ID | 235151 | 22.763098285054927 | Customer identifiers are absent for a measurable share of transaction rows. | Rows cannot be assigned directly to customer-level records. | Retain in the transaction source; exclude from customer-level datasets without imputation. | Applied downstream |
| missing Description | 4275 | 0.4138287533057899 | Description is missing in the reconciled raw data. | Limits product-description diagnostics but not necessarily transaction-level totals. | Retain the transaction; do not impute Description. | Applied |
| C-prefix cancellations | 19104 | 1.849306316527207 | Invoice identifiers beginning with C were observed and quantified. | May affect purchase, return, and monetary behavior measures. | Preserve identified cancellations separately and exclude them from the positive-purchase base. | Applied downstream |
| negative Quantity without C-prefix | 3393 | 0.3284493473605954 | Negative quantities occur on both C-prefix and non-C invoices. | Could affect item and revenue summaries if treated inconsistently. | Investigate and document before deciding on exclusions. | Requires methodological review |
| zero Price | 6014 | 0.5821675140072563 | Zero-price rows are present and may represent special transaction behavior. | Can affect revenue and price-derived features. | Inspect product and transaction context; do not automatically remove. | Requires methodological review |
| negative Price | 5 | 0.00048401023778454964 | Negative-price rows are rare but present. | Can affect monetary summaries and may indicate adjustments. | Review representative records before any rule is chosen. | Requires methodological review |
| special/non-merchandise StockCode patterns | 62767 | 6.075974119004565 | StockCode form and description signals identify diagnostic patterns without classifying them as errors. | Administrative or service rows may distort product and purchase measures. | Create a documented classification rule after reviewing the audit. | Requires methodological review |
| invalid InvoiceDate | 0 | 0.0 | No missing InvoiceDate values were found in the reconciled data. | None observed at this stage. | Retain; continue validating dates in later processing. | Retain |

## 11. Applied methodological decisions

The primary analysis removes exact within-sheet duplicate extras before reconciling worksheets, and removes only the verified second copy of the common cross-sheet interval. Missing Customer ID rows and missing Description rows remain in the reconciled transaction source; missing-ID rows are excluded only from customer-linked datasets. C-prefix cancellations, non-C negative quantities, non-positive prices, and special StockCodes remain available for separate audits and are not blanket-deleted.

No causal relationships are inferred from these descriptive audits.
