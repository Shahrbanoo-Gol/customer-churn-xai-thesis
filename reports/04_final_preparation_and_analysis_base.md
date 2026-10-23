# Step 4 — Final Preparation and Analysis Base

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

The customer transaction history contains **797,885 rows** linked to **5,942 customers**. It retains cancellations, non-positive prices, negative quantities, special StockCodes, and missing descriptions; exact duplicate extras were removed before customer-linked datasets were built.

## 4. Positive-purchase base

The positive-purchase base contains **779,425 rows**. Every row has a non-null Customer ID, a non-C-prefix invoice, Quantity > 0, and Price > 0. Its total Quantity is **10,513,952**, and its gross line value is **17,374,804.27**. It contains **4,631 products**, **36,969 invoices**, and **5,878 customers**.

## 5. Cancellation history

The cancellation history contains **18,390 rows** linked to **2,572 customers**. Its signed Quantity total is **-472,976**, and its signed line-value total is **-1,084,812.98**. These values are reported separately and were not netted against positive purchases.

## 6. Customer and invoice coverage

Among identifiable customers, **5,878** have at least one qualifying positive purchase, **2,572** have at least one cancellation, and **2,511** have both. **64** customers have identified history but no qualifying positive-purchase row; this is a preparation category, not a churn label.

The identified-customer history contains **44,876** invoices, the positive-purchase base contains **36,969** non-C invoices, and the cancellation history contains **7,901** C-prefix invoices. The history contains **37,421** invoices with multiple line items.

## 7. Duplicate treatment

The primary customer history has **0 duplicate-participating rows** and **0 extra occurrences** after keep-first removal. The duplicate-retained alternative remains available for a later sensitivity comparison.

## 8. Data-quality controls

The cleaning waterfall begins with **1,067,371 raw worksheet rows**, followed by **1,033,036 source-reconciled rows**. Missing-ID exclusion is a sequential reduction of **235,151 rows**. Non-purchase conditions in table 45 are diagnostic and overlapping; they are not added as independent reductions. All validation checks passed, and the raw workbook checksum remained `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.

## 9. Datasets available for the next analytical stage

- `02_customer_transaction_history.parquet`: all identified-customer transaction history.
- `03_positive_purchase_base.parquet`: approved basis for positive-purchase behavior.
- `04_customer_cancellation_history.parquet`: customer-linked C-prefix history.

Transaction preparation is complete. Churn has **not** been defined, the final modeling sample has **not** been determined, and observation and future-outcome windows have **not** been selected.

**Approved transaction-preparation rules were executed. Exact within-source duplicates were removed in the primary analysis datasets. Churn, observation/outcome windows, predictive features, train/test splits, encoding, scaling, machine-learning models, and SHAP analyses have not been created.**
