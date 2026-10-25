# Step 7 - Build and Audit Customer-Level Predictive Features

## Scope and locked temporal design

Step 7 uses exactly the Step 6 primary cohort of **2,457 customers**. The observation interval is **[2010-07-27 12:50:00, 2011-07-27 12:50:00]**. Features use only positive purchases, invoice-level purchase events, and customer-linked cancellations inside that interval. The Step 6 `Churn` value is copied unchanged; no future outcome field is used as a predictor.

No train/test split, categorical encoding, scaling, class rebalancing, SMOTE, model training, or SHAP analysis was performed.

## 1. Feature table

The ignored processed output is `data/processed/07_customer_features.parquet`, with exactly one row per Step 6 customer. `Customer ID` is retained only as the row key. `CountryCandidate` is a raw categorical audit candidate and is not encoded.

| feature | definition |
| --- | --- |
| RecencyDays | Days from latest positive purchase to cutoff. |
| Frequency | Unique positive-purchase invoice count. |
| Monetary | Sum of positive-purchase `Quantity * Price`. |
| TotalQuantity | Sum of positive-purchase units. |
| AverageOrderValue | Monetary divided by Frequency. |
| AverageQuantityPerOrder | TotalQuantity divided by Frequency. |
| ActivePurchaseDays | Distinct calendar dates with a positive purchase. |
| UniqueProducts | Distinct purchased `StockCode` count. |
| ProductsPerOrder | Mean distinct `StockCode` count per positive invoice. |
| CustomerAgeDays | First observed positive purchase to cutoff; observable history in this window, not necessarily true lifetime tenure. |
| PurchaseSpanDays | First-to-last positive purchase elapsed days in the observation window. |
| MeanInterpurchaseDays | Mean adjacent positive invoice-event gap; missing for one-event customers. |
| MedianInterpurchaseDays | Median adjacent positive invoice-event gap; missing for one-event customers. |
| HasRepeatPurchaseHistory | 1 when at least two positive invoice events are observed. |
| CancellationInvoiceCount | Unique identified cancellation invoice count. |
| CancellationValue | Sum of absolute cancellation line values. |
| HasCancellation | 1 when at least one cancellation invoice is observed. |
| CancellationInvoiceRate | Cancellation invoices / (positive invoices + cancellation invoices). |
| CountryCandidate | Unique modal observed country, retained raw for audit only. |
| Churn | Step 6 label copied without recalculation. |

## 2. Core feature statistics

| feature_name | data_type | missing_count | min | median | mean | max | n_unique |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RecencyDays | float64 | 0 | 0.0000 | 43.0181 | 50.8599 | 134.9563 | 2395 |
| Frequency | int64 | 0 | 1.0000 | 3.0000 | 5.8339 | 183.0000 | 58 |
| Monetary | float64 | 0 | 2.9000 | 1,070.3800 | 2,784.0437 | 251,520.8200 | 2442 |
| TotalQuantity | int64 | 0 | 1.0000 | 606.0000 | 1,638.3691 | 178,396.0000 | 1518 |
| AverageOrderValue | float64 | 0 | 2.9000 | 303.6743 | 396.2912 | 14,844.7667 | 2442 |
| AverageQuantityPerOrder | float64 | 0 | 1.0000 | 167.0000 | 239.0863 | 22,071.7500 | 1708 |
| ActivePurchaseDays | int64 | 0 | 1.0000 | 3.0000 | 5.1933 | 124.0000 | 49 |
| UniqueProducts | int64 | 0 | 1.0000 | 52.0000 | 77.6577 | 1,604.0000 | 313 |

The complete business meaning, source, calculation rule, percentiles, skewness, zero inflation, and impossible-value audit is in `82_step7_feature_quality_audit.csv`.

## 3. Missing and structural-missing audit

| feature_name | missing_count | missing_pct | structurally_missing | structural_missing_reason |
| --- | --- | --- | --- | --- |
| MeanInterpurchaseDays | 445 | 18.1115 | True | Only one positive-purchase event; no adjacent interpurchase interval exists. |
| MedianInterpurchaseDays | 445 | 18.1115 | True | Only one positive-purchase event; no adjacent interpurchase interval exists. |

`MeanInterpurchaseDays` and `MedianInterpurchaseDays` are structurally missing for **445 customers** with only one positive purchase event. They were retained as missing; no zero, mean, or median imputation was performed. Customers with repeat history: **2,012**.

## 4. Cancellation-feature results

Observation-period cancellation features are zero for customers without an identified customer-linked C-prefix invoice. The cohort has **1,144 customers with cancellation activity** (46.56%). Total cancellation invoices are **3,095**, total absolute cancellation value is **282,432.64**, and the customer-level cancellation rate has median **0.0000** and maximum **0.7500**. These values use only cancellations through the cutoff and are not netted against purchases.

## 5. Country audit

| metric | country | customers | percentage | notes |
| --- | --- | --- | --- | --- |
| primary_cohort_customers |  | 2457 | 100.0000 | Locked Step 6 cohort. |
| customers_with_exactly_one_observed_country |  | 2449 | 99.6744 | Country associations counted from observation-period positive purchases. |
| customers_with_more_than_one_observed_country |  | 8 | 0.3256 | These customers have multiple observed country values. |
| customers_with_no_observed_country |  | 0 | 0.0000 | No non-null Country in observation-period positive purchases. |
| customers_with_unique_dominant_country_candidate |  | 2457 | 100.0000 | Raw candidate included because the observed dominant country is unique for each customer in this cohort. |
| largest_country_category | United Kingdom | 2229 | 90.7204 | Concentration of the largest raw country candidate category. |

The largest country candidate category is **United Kingdom** with **2,229 customers (90.72%)**. The distribution is therefore highly concentrated, but Country remains a raw candidate for later modeling audit because it is observed and unencoded here; its eventual modeling value must be assessed empirically rather than decided from intuition alone. The full dominant-country distribution is in `83_step7_country_audit.csv`.

## 6. Distribution and outlier findings

The core RFM distributions are in `outputs/figures/07_core_rfm_distributions.png`; the compact skewness view is in `outputs/figures/08_feature_skewness_audit.png`. The quality table marks zero-inflated features using a zero share of at least 20% and strongly skewed numeric features using absolute skewness at least 2. Monetary, TotalQuantity, Frequency, and related order-value measures are expected to be right-skewed, with large maxima relative to their medians. Cancellation indicators/value are zero-inflated. Extreme values are retained for audit; no outliers were automatically removed and no transformations were applied.

## 7. Numeric correlation audit

The correlation audit excludes `Customer ID`, `CountryCandidate`, and `Churn`; it is descriptive only and is not a feature-selection rule.

| feature_a | feature_b | pearson_correlation | interpretation |
| --- | --- | --- | --- |
| MeanInterpurchaseDays | MedianInterpurchaseDays | 0.9806 | Derived/component redundancy is plausible; retain for audit and defer feature selection. |
| Frequency | ActivePurchaseDays | 0.9719 | Derived/component redundancy is plausible; retain for audit and defer feature selection. |
| CustomerAgeDays | PurchaseSpanDays | 0.9490 | Derived/component redundancy is plausible; retain for audit and defer feature selection. |

High associations are expected where one feature is mathematically derived from another: Monetary and AverageOrderValue relate through Frequency; TotalQuantity and AverageQuantityPerOrder relate through Frequency; ProductsPerOrder relates to UniqueProducts and Frequency; and the purchase-history time measures share first/last purchase timestamps. No feature was removed automatically.

## 8. Reconciliation and temporal-leakage validation

**21/21 reconciliation checks passed.**

| check | passed | observed | expected |
| --- | --- | --- | --- |
| final_rows_exactly_2457 | True | 2457 | 2457 |
| Customer_ID_unique | True | 2457 | 2457 |
| Churn_exactly_matches_Step6 | True | equal | equal |
| positive_observation_has_no_post_cutoff_rows | True | 2011-07-27 12:50:00 | 2011-07-27 12:50:00 |
| positive_observation_respects_observation_start | True | 2010-07-27 13:14:00 | 2010-07-27 12:50:00 |
| event_observation_has_no_post_cutoff_rows | True | 2011-07-27 12:50:00 | 2011-07-27 12:50:00 |
| cancellation_observation_has_no_post_cutoff_rows | True | 2011-07-27 12:40:00 | 2011-07-27 12:50:00 |
| RecencyDays_reconciles_with_Step6 | True | 0.0000 | 0.0000 |
| RecencyDays_at_most_135 | True | 134.9563 | 135 |
| Frequency_at_least_1 | True | 1 | 1 |
| Monetary_positive | True | 2.9000 | > 0 |
| TotalQuantity_positive | True | 1 | > 0 |
| AverageOrderValue_positive | True | 2.9000 | > 0 |
| cancellation_features_observation_only | True | 1144 | 1144 |
| no_future_outcome_columns_in_feature_table | True | [] | [] |
| step6_outcome_table_hash_unchanged | True | 166e9cec31785968a3304c7a0442e34b05de990cca92ccf516481d6f2d2f7b99 | 166e9cec31785968a3304c7a0442e34b05de990cca92ccf516481d6f2d2f7b99 |
| positive_purchase_base_hash_unchanged | True | 825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503 | 825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503 |
| cancellation_history_hash_unchanged | True | 7a90c3159f0ac7a810f47ec05a51b257d0c4f9359ea6d8eeaa6b3584a0a76023 | 7a90c3159f0ac7a810f47ec05a51b257d0c4f9359ea6d8eeaa6b3584a0a76023 |
| purchase_events_hash_unchanged | True | 4c6fee4faaa9214eff99e446fe7b2956473b7d250cd8852997799f3b4c8dc5b8 | 4c6fee4faaa9214eff99e446fe7b2956473b7d250cd8852997799f3b4c8dc5b8 |
| canonical_source_hash_unchanged | True | a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42 | a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42 |
| raw_workbook_hash_unchanged | True | bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980 | bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980 |

The feature table contains no post-cutoff transaction-derived fields and no Step 6 outcome-window validation columns. The only outcome value carried forward is `Churn`. Input hashes before and after execution are recorded in `86_step7_input_hashes.csv`; the canonical source, raw workbook, and committed Step 4-6 inputs remained unchanged.

## 9. Outputs and stopping point

- Source module: `src/churn_analysis/build_customer_features.py`
- Lightweight notebook: `notebooks/07_customer_feature_engineering_and_audit.ipynb`
- Report: `reports/07_customer_feature_engineering_and_audit.md`
- Audit tables: `outputs/tables/82_step7_feature_quality_audit.csv` through `87_step7_output_manifest.csv`
- Figures: `outputs/figures/07_core_rfm_distributions.png` and `08_feature_skewness_audit.png`
- Reproducibility log: `outputs/logs/07_customer_feature_engineering_and_audit.log`
- Ignored processed feature table: `data/processed/07_customer_features.parquet`

Step 7 stops after customer-level feature construction, quality auditing, descriptive country review, correlation auditing, and reconciliation. No split, encoding, scaling, resampling, model training, or SHAP analysis occurred.

Input hashes at start:

| input | path | sha256_before | sha256_after | unchanged |
| --- | --- | --- | --- | --- |
| step6_outcome_table | data\processed\06_primary_customer_outcome_table.parquet | 166e9cec31785968a3304c7a0442e34b05de990cca92ccf516481d6f2d2f7b99 | 166e9cec31785968a3304c7a0442e34b05de990cca92ccf516481d6f2d2f7b99 | True |
| positive_purchase_base | data\processed\03_positive_purchase_base.parquet | 825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503 | 825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503 | True |
| cancellation_history | data\processed\04_customer_cancellation_history.parquet | 7a90c3159f0ac7a810f47ec05a51b257d0c4f9359ea6d8eeaa6b3584a0a76023 | 7a90c3159f0ac7a810f47ec05a51b257d0c4f9359ea6d8eeaa6b3584a0a76023 | True |
| purchase_events | data\processed\05_invoice_purchase_events.parquet | 4c6fee4faaa9214eff99e446fe7b2956473b7d250cd8852997799f3b4c8dc5b8 | 4c6fee4faaa9214eff99e446fe7b2956473b7d250cd8852997799f3b4c8dc5b8 | True |
| canonical_source | data\processed\01_source_reconciled.parquet | a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42 | a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42 | True |
| raw_workbook | data\raw\online_retail_II.xlsx | bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980 | bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980 | True |
