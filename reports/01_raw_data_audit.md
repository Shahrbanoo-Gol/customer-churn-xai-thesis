# Raw Data Audit

This report documents a non-destructive audit of the raw source file for Step 1 of the thesis workflow. Statistics were recalculated directly from the source file; no records were removed, altered, imputed, cleaned, relabelled, or aggregated for modeling.

## Dataset file

- Filename: `online_retail_II.xlsx`
- Format: `.xlsx`
- Size in bytes: `45622278`
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`
- Last modified (UTC): `2023-05-22T11:50:22+00:00`

## Workbook structure

Source path: `data\raw\online_retail_II.xlsx`

| sheet_name | row_count | column_count |
| --- | --- | --- |
| Year 2009-2010 | 525461 | 8 |
| Year 2010-2011 | 541910 | 8 |

Schema identical across worksheets: **True**.

## Raw variables

The exact observed column names and pandas-detected types are listed below. No source column names were changed.

| sheet_name | column_name | detected_dtype | non_null_count | null_count | null_percentage | unique_non_null_values |
| --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Invoice | object | 525461 | 0 | 0.0 | 28816 |
| Year 2009-2010 | StockCode | object | 525461 | 0 | 0.0 | 4632 |
| Year 2009-2010 | Description | object | 522533 | 2928 | 0.5572249891048051 | 4681 |
| Year 2009-2010 | Quantity | int64 | 525461 | 0 | 0.0 | 825 |
| Year 2009-2010 | InvoiceDate | datetime64[us] | 525461 | 0 | 0.0 | 25296 |
| Year 2009-2010 | Price | float64 | 525461 | 0 | 0.0 | 1606 |
| Year 2009-2010 | Customer ID | float64 | 417534 | 107927 | 20.53948818275762 | 4383 |
| Year 2009-2010 | Country | str | 525461 | 0 | 0.0 | 40 |
| Year 2010-2011 | Invoice | object | 541910 | 0 | 0.0 | 25900 |
| Year 2010-2011 | StockCode | object | 541910 | 0 | 0.0 | 4070 |
| Year 2010-2011 | Description | object | 540456 | 1454 | 0.26831023601705084 | 4223 |
| Year 2010-2011 | Quantity | int64 | 541910 | 0 | 0.0 | 722 |
| Year 2010-2011 | InvoiceDate | datetime64[us] | 541910 | 0 | 0.0 | 23260 |
| Year 2010-2011 | Price | float64 | 541910 | 0 | 0.0 | 1630 |
| Year 2010-2011 | Customer ID | float64 | 406830 | 135080 | 24.92664833643963 | 4372 |
| Year 2010-2011 | Country | str | 541910 | 0 | 0.0 | 38 |

## Date coverage

| scope | sheet_name | date_column | minimum_transaction_datetime | maximum_transaction_datetime | invalid_unparseable_date_count |
| --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | InvoiceDate | 2009-12-01T07:45:00 | 2010-12-09T20:01:00 | 0 |
| Year 2010-2011 | Year 2010-2011 | InvoiceDate | 2010-12-01T08:26:00 | 2011-12-09T12:50:00 | 0 |
| overall | overall | InvoiceDate | 2009-12-01T07:45:00 | 2011-12-09T12:50:00 | 0 |

## Missing data

Customer-ID missingness by sheet:

| scope | sheet_name | customer_id_column | non_null_customer_id_records | missing_customer_id_records | missing_customer_id_percentage | unique_non_null_customers |
| --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | Customer ID | 417534 | 107927 | 20.53948818275762 | 4383 |
| Year 2010-2011 | Year 2010-2011 | Customer ID | 406830 | 135080 | 24.92664833643963 | 4372 |

Description missingness by sheet:

| scope | sheet_name | column_name | missing_count | missing_percentage |
| --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | Description | 2928 | 0.5572249891048051 |
| Year 2010-2011 | Year 2010-2011 | Description | 1454 | 0.26831023601705084 |
| overall | overall | Description | 4382 | 0.4105414143723223 |

The complete column-level missing-value audit is saved in `outputs/tables/05_missing_values.csv`.

## Duplicates

| scope | sheet_name | total_row_count | exact_duplicated_row_count | exact_duplicated_row_percentage | duplicate_rows_excluding_first_occurrence |
| --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | 525461 | 13283 | 2.5278755226363137 | 6865 |
| Year 2010-2011 | Year 2010-2011 | 541910 | 10147 | 1.8724511450240815 | 5268 |
| overall | overall | 1067371 | 67242 | 6.299777678051961 | 34335 |

Duplicate rows are reported only as diagnostics. No assumption was made that they are errors.

## Customer IDs

| scope | sheet_name | customer_id_column | non_null_customer_id_records | missing_customer_id_records | missing_customer_id_percentage | unique_non_null_customers |
| --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | Customer ID | 417534 | 107927 | 20.53948818275762 | 4383 |
| Year 2010-2011 | Year 2010-2011 | Customer ID | 406830 | 135080 | 24.92664833643963 | 4372 |
| overall | overall | Customer ID | 824364 | 243007 | 22.766872999172733 | 5942 |

## Cancellations

| scope | sheet_name | invoice_column | unique_invoices | unique_invoices_beginning_with_C | percentage_unique_invoices_beginning_with_C | rows_associated_with_C_prefix_invoices |
| --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Year 2009-2010 | Invoice | 28816 | 4592 | 15.935591338145475 | 10206 |
| Year 2010-2011 | Year 2010-2011 | Invoice | 25900 | 3836 | 14.810810810810812 | 9288 |
| overall | overall | Invoice | 53628 | 8292 | 15.46207205191318 | 19494 |

The `C`-prefix convention is reported as observed in the invoice identifier. Cancellation transactions were not deleted, and negative quantities were not treated as automatically equivalent to cancellation invoices.

## Quantity and price

Quantity audit:

| scope | variable | count | invalid_numeric_count | minimum | maximum | mean | median | standard_deviation | percentile_1 | percentile_5 | percentile_25 | percentile_75 | percentile_95 | percentile_99 | number_less_than_zero | number_equal_to_zero | number_greater_than_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Quantity | 525461 | 0 | -9600.0 | 19152.0 | 10.337667305470815 | 3.0 | 107.42411026879778 | -3.0 | 1.0 | 1.0 | 10.0 | 30.0 | 120.0 | 12326 | 0 | 513135 |
| Year 2010-2011 | Quantity | 541910 | 0 | -80995.0 | 80995.0 | 9.552233765754462 | 3.0 | 218.08095694392432 | -2.0 | 1.0 | 1.0 | 10.0 | 29.0 | 100.0 | 10624 | 0 | 531286 |
| overall | Quantity | 1067371 | 0 | -80995.0 | 80995.0 | 9.9388984711033 | 3.0 | 172.7057940767534 | -3.0 | 1.0 | 1.0 | 10.0 | 30.0 | 100.0 | 22950 | 0 | 1044421 |

Price audit:

| scope | variable | count | invalid_numeric_count | minimum | maximum | mean | median | standard_deviation | percentile_1 | percentile_5 | percentile_25 | percentile_75 | percentile_95 | percentile_99 | number_less_than_zero | number_equal_to_zero | number_greater_than_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Year 2009-2010 | Price | 525461 | 0 | -53594.36 | 25111.09 | 4.688834478676819 | 2.1 | 146.1269139529265 | 0.21 | 0.42 | 1.25 | 4.21 | 10.17 | 19.95 | 3 | 3687 | 521771 |
| Year 2010-2011 | Price | 541910 | 0 | -11062.06 | 38970.0 | 4.611138332933514 | 2.08 | 96.75976549366526 | 0.19 | 0.42 | 1.25 | 4.13 | 9.95 | 18.0 | 2 | 2515 | 539393 |
| overall | Price | 1067371 | 0 | -53594.36 | 38970.0 | 4.64938772741624 | 2.1 | 123.5530587214632 | 0.21 | 0.42 | 1.25 | 4.15 | 9.95 | 18.0 | 5 | 6202 | 1061164 |

Unusual values are reported descriptively only. No outlier or exclusion decision was made.

## Countries

Number of unique country values by scope and the descending transaction-row distribution are saved in `outputs/tables/12_country_distribution.csv`.

| scope | sheet_name | country | transaction_row_count | percentage_of_rows | unique_country_values_in_scope |
| --- | --- | --- | --- | --- | --- |
| overall | overall | United Kingdom | 981330 | 91.93897904290074 | 43 |
| overall | overall | EIRE | 17866 | 1.6738322476439778 | 43 |
| overall | overall | Germany | 17624 | 1.6511597185983131 | 43 |
| overall | overall | France | 14330 | 1.342550996794929 | 43 |
| overall | overall | Netherlands | 5140 | 0.4815570218789905 | 43 |
| overall | overall | Spain | 3811 | 0.35704548840094025 | 43 |
| overall | overall | Switzerland | 3189 | 0.29877146746538924 | 43 |
| overall | overall | Belgium | 3123 | 0.2925880504529353 | 43 |
| overall | overall | Portugal | 2620 | 0.2454629177671119 | 43 |
| overall | overall | Australia | 1913 | 0.17922540522461264 | 43 |
| overall | overall | Channel Islands | 1664 | 0.15589705922308175 | 43 |
| overall | overall | Italy | 1534 | 0.1437176014712785 | 43 |
| overall | overall | Norway | 1455 | 0.1363162386836442 | 43 |
| overall | overall | Sweden | 1364 | 0.12779061825738192 | 43 |
| overall | overall | Cyprus | 1176 | 0.11017724858554336 | 43 |

## Reproducible worksheet profiles before modification

Each raw worksheet was profiled independently before any duplicate removal. The citation-ready outputs are saved as `47_raw_worksheet_profile_summary.csv`, `48_raw_worksheet_head.csv`, `49_raw_worksheet_tail.csv`, `50_raw_worksheet_sample.csv`, `51_raw_worksheet_missing_values.csv`, and `52_raw_worksheet_descriptive_statistics.csv`. Random samples use the fixed seed `42`. These tables include worksheet shape, columns, data types, head, tail, sample, missingness, descriptive statistics, unique customer/invoice counts, and date ranges.

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
