# Step 3 — Cleaning Impact Assessment

## Executive conclusion

This assessment measures the consequences of candidate cleaning rules before final cleaning. The canonical source-reconciled dataset contains **1,044,848 rows**, **5,942 identifiable customers**, and spans **2009-12-01 to 2011-12-09**. No rule was executed in this step.

The main management implication is that transaction-level history should remain available for audit and sensitivity analysis, while customer-level purchase measures should be built only from records that can be linked to an identified customer and that represent positive-priced positive-quantity purchases. Cancellations and administrative exceptions should be preserved separately rather than silently treated as ordinary purchases.

## A. Missing Customer ID

235,287 rows (22.52%) have no Customer ID, representing 8,752 invoice identifiers. The remaining **809,561 rows (77.48%)** are identifiable and represent **5,942 customers**. Unidentified rows span **2009-12-01 to 2011-12-09** and have signed quantity total **354,170**.

These rows cannot be assigned to a customer-level modeling record without inventing an identifier. The recommendation is therefore to exclude them from customer-level modeling tables, while retaining them in transaction-level totals and reporting their monthly and country distribution in `31_missing_customer_impact.csv`.

## B. Exact duplicates

The temporary comparison uses the original eight transaction fields and removes only duplicate occurrences after the first. It does not modify the parquet. The comparison is:

| scenario | metric | value |
| --- | --- | --- |
| A_duplicates_retained | total_rows | 1,044,848.00 |
| A_duplicates_retained | identified_customer_rows | 809,561.00 |
| A_duplicates_retained | unique_customers | 5,942.00 |
| A_duplicates_retained | unique_invoices | 53,628.00 |
| A_duplicates_retained | aggregate_positive_purchase_monetary_value | 17,431,551.18 |
| A_duplicates_retained | aggregate_positive_purchase_quantity | 10,547,799.00 |
| A_duplicates_retained | duplicate_extra_occurrences | 11,812.00 |
| B_duplicate_occurrences_removed | total_rows | 1,033,036.00 |
| B_duplicate_occurrences_removed | identified_customer_rows | 797,885.00 |
| B_duplicate_occurrences_removed | unique_customers | 5,942.00 |
| B_duplicate_occurrences_removed | unique_invoices | 53,628.00 |
| B_duplicate_occurrences_removed | aggregate_positive_purchase_monetary_value | 17,374,804.27 |
| B_duplicate_occurrences_removed | aggregate_positive_purchase_quantity | 10,513,952.00 |
| B_duplicate_occurrences_removed | duplicate_extra_occurrences | 0.00 |

Keep-first removes **11,812 rows**. It affects **1,594 of 5,942 identifiable customers (26.83%)**. The unique invoice count changes by **0**, so the measured risk is concentrated in row counts, quantity, and monetary totals rather than invoice identity. Among affected customers, the median relative change is **-1.36%** for positive-purchase value, **-1.18%** for positive-purchase quantity, and **-2.55%** for transaction-row count; the 25th–75th percentile range is **-2.77% to -0.58%** for value. Because identical rows may be source duplication or genuine repeated line items, sensitivity testing is recommended before a final rule is chosen.

## C. Customer-linked cancellation profile

There are **2,572 identifiable customers** with at least one C-prefix invoice. C-prefix rows are not ordinary purchases: including them in purchase metrics can distort frequency, item quantity, and monetary behavior. The recommendation is to exclude them from ordinary positive-purchase metrics but preserve them as a separate cancellation-behavior signal for later consideration. The cancellation table includes invoice-level and customer-level rates plus the distribution of cancellation counts.

## D. Non-C negative Quantity

The required exception population contains **3,393 rows**, affecting **0 identifiable customers** and 3,393 invoices. A transparent description screen finds **375 rows** with an apparent administrative or inventory-adjustment signal. This is evidence for review, not an automatic classification. The recommendation is to exclude these rows from ordinary positive-purchase measures provisionally, retain them in transaction history, and review uncertain cases before finalizing.

## E. Price <= 0

There are **6,029 rows** with Price <= 0, affecting **51 identifiable customers**. They contribute **0** to the positive-purchase value measure because that measure requires Quantity > 0 and Price > 0. They should therefore be excluded from merchandise purchase-value metrics, while being retained separately for exception analysis. The table estimates customers whose identified records contain no positive-purchase candidate and would therefore have no record in a future positive-purchase table.

## F. Special StockCodes

Manual, postage, discount, sample, dotcom, adjustment, gift, and bank/fee signals were profiled using overlapping diagnostic screens. The screens are not validity judgments: a special code can be customer-linked and have a positive price, and an ordinary-looking code can still represent an exception. The recommendation is to retain the full history, exclude only clearly administrative categories from merchandise metrics after documentation, and sensitivity-test uncertain patterns. See `37_special_stockcode_impact.csv` for row counts, customer/invoice footprints, quantity and price signs, and positive-purchase value shares.

## G. Candidate transaction roles

The temporary primary-role counts below use this precedence: **Cancellation → Non-C negative adjustment → Non-positive-price record → Positive purchase candidate → Other/requires review**. Diagnostic flags are also reported because categories can overlap, especially with missing Customer IDs. These roles are not predictive features and are not final cleaning rules.

| role | row_count | identified_customers | unique_invoices |
| --- | --- | --- | --- |
| Positive purchase candidate | 791,045 | 5,878 | 36,969 |
| Cancellation | 19,165 | 2,572 | 8,292 |
| Non-C negative adjustment | 3,393 | 0 | 3,393 |
| Other / requires review | 228,609 | 0 | 3,108 |
| Non-positive-price record | 2,636 | 51 | 1,991 |

## H. Recommended treatment and confidence

| issue | recommended_treatment | confidence |
| --- | --- | --- |
| Missing Customer ID | Exclude from customer-level modeling tables; retain in transaction-level audit totals and report sensitivity. | High |
| Cross-sheet overlap | Retain the Step 2 reconciliation; do not revisit. | High |
| Exact duplicates | Sensitivity-test retained versus keep-first scenarios before selecting a final rule; do not execute this step's scenario as a final cleaning rule. | Moderate |
| C-prefix cancellations | Exclude from ordinary positive-purchase metrics but preserve separately for cancellation-behavior features and audit reporting. | High |
| Non-C negative Quantity | Exclude from ordinary positive-purchase metrics provisionally, retain in transaction history, and review uncertain patterns before finalizing. | Moderate |
| Price <= 0 | Exclude from positive-purchase value and product-purchase metrics; retain separately for administrative/exception analysis and document customers with no positive purchase. | High |
| Special StockCodes | Keep in general transaction history; exclude only clearly administrative categories from merchandise metrics after documented review; retain uncertain patterns for sensitivity analysis. | Moderate |

Confidence is **High** where the record cannot support a defensible customer link or positive purchase value. Confidence is **Moderate** where the observed pattern is clear but its business meaning remains ambiguous, especially exact duplicates, non-C negative quantities, and special StockCodes.

## Methodological ambiguity remaining

The main unresolved issue is whether identical eight-field rows are repeated source records or legitimate repeated line items. Additional ambiguity remains for non-C negative quantities and overlapping special-code signals: descriptions suggest administrative behavior in some cases, but the dataset does not prove the business meaning of every row. These issues should be handled through documented review and sensitivity analysis, not silent deletion.

## Scope validation

- Source-reconciled row count and Step 2 date range were validated.
- Raw workbook SHA-256 remained `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
- The canonical parquet was read only; no permanent row removal occurred.
- No churn definition, observation/outcome window, predictive feature set, train/test split, or machine-learning model was created.

**No final cleaning rule was executed in this step. The canonical source-reconciled dataset was not modified. No churn definition, predictive feature set, train/test split, or machine-learning model was created.**
