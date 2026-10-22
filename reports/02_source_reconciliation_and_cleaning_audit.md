# Step 2 — Source Reconciliation and Cleaning Audit

## 1. Purpose

This step constructs a chronological source-reconciled transaction dataset after the complete exact worksheet overlap established in Step 1A. It executes only that source-reconciliation rule and audits remaining raw-data conditions. It does not create churn labels, RFM features, predictive features, train/test splits, or machine-learning models.

The source file was `data\raw\online_retail_II.xlsx` with SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. The raw workbook remained read-only.

## 2. Source reconciliation

The deterministic rule retained all `Year 2009-2010` rows and retained only `Year 2010-2011` rows with `InvoiceDate` strictly after the maximum `Year 2009-2010` timestamp (`2010-12-09T20:01:00`). This removed `22,523` rows solely because they were the second representation of the proven cross-sheet overlap.

| metric | value | notes |
| --- | --- | --- |
| raw_sheet_1_rows | 525461 | All rows retained from Year 2009-2010. |
| raw_sheet_2_rows | 541910 | Rows originally present in Year 2010-2011. |
| total_raw_rows_before_source_reconciliation | 1067371 | Sum of both raw worksheets. |
| cross_sheet_repeated_occurrences_removed | 22523 | Only the second representation of the proven overlap. |
| row_count_after_source_reconciliation | 1044848 | Chronologically sorted source-reconciled rows. |
| earliest_InvoiceDate | 2009-12-01T07:45:00 | Observed in reconciled data. |
| latest_InvoiceDate | 2011-12-09T12:50:00 | Observed in reconciled data. |
| unique_Invoice_count | 53628 | Non-null unique invoice identifiers. |
| unique_non_null_Customer_ID_count | 5942 | Non-null unique customer identifiers. |

The reconciled Parquet file contains the original eight variables plus the technical `SourceSheet` provenance field. Because the source identifier columns contain mixed numeric/text values, `Invoice` and `StockCode` use stable textual Parquet storage while preserving their displayed source values. No global `drop_duplicates()` operation was used.

## 3. Validation of the reconciled timeline

The reconciled data spans `2009-12-01T07:45:00` through `2011-12-09T12:50:00`. The preferred boundary rule was validated against multiplicity-aware full-row counts: the resulting transaction multiset matched exactly.

There were **135** calendar dates without transaction rows; these were reported and not imputed. Around the source boundary:

| calendar_date | reconciled_row_count | year_2009_2010_rows | year_2010_2011_retained_rows | boundary_gap_hours | notes |
| --- | --- | --- | --- | --- | --- |
| 2010-12-09 | 2891 | 2891 | 0 | 13.533333333333333 | Source boundary date requires review of the transition timing. |

## 4. Remaining exact duplicates

After source reconciliation, `22,813` rows participate in exact duplicate groups under `duplicated(keep=False)`, and `11,812` are extra occurrences beyond the first under `duplicated(keep='first')`. There are `11,001` duplicate signatures, with maximum multiplicity `20`.

Duplicate groups necessarily have the same Invoice, StockCode, and InvoiceDate because those are included in the exact eight-field signature. Their concentrations and representative largest groups are documented in `outputs/tables/23_duplicate_context_examples.csv`.

## 5. Missing Customer ID

Rows without Customer ID: `235,287` (22.5188%). They were retained. The monthly rate and comparisons of cancellation, quantity-sign, and price-sign behavior between missing and identified customers are in `outputs/tables/24_customer_id_missingness_audit.csv`.

## 6. Cancellations and negative quantities

| category | count | percentage | invoice_type | quantity_sign |
| --- | --- | --- | --- | --- |
| C-prefix AND Quantity < 0 | 19164 | 1.83414238243266 | C-prefix | negative |
| C-prefix AND Quantity >= 0 | 1 | 9.570770102445524e-05 | C-prefix | non-negative |
| non-C invoice AND Quantity < 0 | 3393 | 0.3247362295759766 | non-C | negative |
| non-C invoice AND Quantity >= 0 | 1022290 | 97.84102568029034 | non-C | non-negative |

The two exception categories were retained for review:

| category | count | percentage | invoice_type | quantity_sign |
| --- | --- | --- | --- | --- |
| C-prefix AND Quantity >= 0 | 1 | 9.570770102445524e-05 | C-prefix | non-negative |
| non-C invoice AND Quantity < 0 | 3393 | 0.3247362295759766 | non-C | negative |

No cancellation or negative-quantity rule was executed.

## 7. Non-positive prices

| record_type | price_category | count | percentage_of_reconciled_rows | StockCode | Description | Quantity | Customer ID | Country | cancellation_status | Price | pattern_row_count | example_Invoice | example_InvoiceDate | distinct_pattern_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | Price < 0 | 5 | 0.0004785385051222761 |  |  |  |  |  |  |  |  |  |  |  |
| summary | Price = 0 | 6024 | 0.5765431909713183 |  |  |  |  |  |  |  |  |  |  |  |
| summary | Price > 0 | 1038819 | 99.42297827052356 |  |  |  |  |  |  |  |  |  |  |  |

Distinct non-positive-price patterns and representative records are saved in `outputs/tables/27_nonpositive_price_audit.csv`. Zero-price rows were not automatically classified as invalid.

## 8. Special transaction/product codes

The StockCode audit reports code form, description signals, missing descriptions, negative quantities, non-positive prices, and cancellation-prefix counts. The most behaviorally unusual observed codes/patterns were:

| observed_stockcode | stockcode_pattern | description_signal | row_count | missing_description_count | negative_quantity_count | nonpositive_price_count | cancellation_prefix_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M | alphanumeric | manual | 1398 | 0 | 534 | 7 | 535 |
| 22423 | numeric_only |  | 4327 | 1 | 353 | 15 | 342 |
| POST | alphanumeric | postage | 2086 | 7 | 228 | 7 | 228 |
| 22138 | numeric_only |  | 2152 | 2 | 211 | 4 | 209 |
| 21232 | numeric_only |  | 2709 | 0 | 189 | 1 | 189 |
| D | alphanumeric | discount | 173 | 0 | 168 | 0 | 168 |
| 21843 | numeric_only |  | 1693 | 0 | 159 | 5 | 155 |
| 85123A | alphanumeric |  | 5711 | 0 | 136 | 8 | 134 |
| 79323W | alphanumeric |  | 478 | 1 | 128 | 2 | 126 |
| 21527 | numeric_only |  | 972 | 3 | 117 | 4 | 114 |
| S | alphanumeric | sample | 102 | 0 | 99 | 0 | 99 |
| 22197 | numeric_only |  | 2479 | 4 | 92 | 9 | 90 |
| 82483 | numeric_only |  | 1240 | 2 | 90 | 3 | 88 |
| 22960 | numeric_only |  | 1259 | 1 | 88 | 3 | 87 |
| 79323P | alphanumeric |  | 354 | 1 | 88 | 2 | 86 |

These are diagnostic patterns only; no merchandise/non-merchandise classification was executed.

## 9. Geographic profile

| Country | transaction_row_count | unique_non_null_customers | unique_non_null_invoices | percentage_of_rows |
| --- | --- | --- | --- | --- |
| United Kingdom | 959983 | 5410 | 49108 | 91.87776595255961 |
| EIRE | 17689 | 5 | 806 | 1.6929735234215888 |
| Germany | 17363 | 107 | 1095 | 1.6617728128876161 |
| France | 14059 | 95 | 746 | 1.345554568702816 |
| Netherlands | 5138 | 23 | 250 | 0.49174616786365094 |
| Spain | 3766 | 41 | 188 | 0.36043520205809837 |
| Switzerland | 3183 | 22 | 123 | 0.30463761236084097 |
| Belgium | 3111 | 29 | 183 | 0.2977466578870802 |
| Portugal | 2540 | 24 | 124 | 0.24309756060211632 |
| Australia | 1891 | 15 | 117 | 0.18098326263724485 |
| Channel Islands | 1647 | 14 | 79 | 0.15763058358727777 |
| Italy | 1509 | 17 | 92 | 0.14442292084590297 |
| Sweden | 1364 | 19 | 129 | 0.13054530419735694 |
| Norway | 1308 | 13 | 53 | 0.12518567293998745 |
| Cyprus | 1176 | 11 | 45 | 0.11255225640475934 |

The complete country profile includes transaction rows, unique non-null customers, and unique non-null invoices for every observed country.

## 10. Cleaning-decision matrix

| issue | observed_count | observed_percentage | evidence | potential_effect_on_customer_features | candidate_treatment | decision_status |
| --- | --- | --- | --- | --- | --- | --- |
| cross-sheet overlap | 22523 | 2.1101378995681914 | Complete exact overlap established in Step 1A; second representation removed by deterministic boundary rule. | Would double-count transactions, invoices, and customer activity if retained. | Retain Year 2009-2010 and Year 2010-2011 rows after the first-sheet maximum InvoiceDate. | Executed |
| within-source exact duplicates | 22813 | 2.183379783470897 | Exact duplicate groups remain after cross-sheet reconciliation. | May inflate purchase frequency, quantity, revenue, or product counts. | Review duplicate context and define a documented rule before final preprocessing. | Requires methodological review |
| missing Customer ID | 235287 | 22.518777850941 | Customer identifiers are absent for a measurable share of transaction rows. | Rows cannot be assigned directly to customer-level records. | Evaluate exclusion from customer-level modeling with sensitivity reporting. | Recommended for next step |
| missing Description | 4275 | 0.4091504218795461 | Description is missing in the reconciled raw data. | Limits product-description diagnostics but not necessarily transaction-level totals. | Retain while deciding whether description is used in later features. | Requires methodological review |
| C-prefix cancellations | 19165 | 1.8342380901336843 | Invoice identifiers beginning with C were observed and quantified. | May affect purchase, return, and monetary behavior measures. | Define cancellation treatment in the next analytical stage. | Recommended for next step |
| negative Quantity without C-prefix | 3393 | 0.3247362295759766 | Negative quantities occur on both C-prefix and non-C invoices. | Could affect item and revenue summaries if treated inconsistently. | Investigate and document before deciding on exclusions. | Requires methodological review |
| zero Price | 6024 | 0.5765431909713183 | Zero-price rows are present and may represent special transaction behavior. | Can affect revenue and price-derived features. | Inspect product and transaction context; do not automatically remove. | Requires methodological review |
| negative Price | 5 | 0.0004785385051222761 | Negative-price rows are rare but present. | Can affect monetary summaries and may indicate adjustments. | Review representative records before any rule is chosen. | Requires methodological review |
| special/non-merchandise StockCode patterns | 63339 | 6.06203007518797 | StockCode form and description signals identify diagnostic patterns without classifying them as errors. | Administrative or service rows may distort product and purchase measures. | Create a documented classification rule after reviewing the audit. | Requires methodological review |
| invalid InvoiceDate | 0 | 0.0 | No missing InvoiceDate values were found in the reconciled data. | None observed at this stage. | Retain; continue validating dates in later processing. | Retain |

## 11. Decisions requiring approval before final preprocessing

Only the proven cross-sheet overlap was executed. Before final preprocessing, methodological approval is still required for within-source exact duplicates, missing Customer ID rows, missing descriptions, cancellation treatment, negative quantities without a C-prefix, non-positive prices, and special/non-merchandise StockCode patterns. These decisions may affect customer-level feature construction and must be documented before modeling.

No causal relationships are inferred from these descriptive audits.
