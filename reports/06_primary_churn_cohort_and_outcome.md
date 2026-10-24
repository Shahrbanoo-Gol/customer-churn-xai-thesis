# Step 6 — Lock the Primary Churn Cohort and Create the Customer-Level Outcome Table

## Scope

This step implements the locked Step 5 decisions for the preventive-retention use case: a 135-day inactivity horizon, approximately one year of observation history, and Population B restricted to customers whose baseline recency is <=135 days at the prediction cutoff. It creates the minimal customer-level outcome table only. No RFM or predictive features, train/test split, encoding, scaling, resampling, model, or SHAP analysis was performed.

## 1. Exact temporal boundaries

The validated positive-purchase event endpoint is **2011-12-09 12:50:00**. The exact cutoff is the endpoint minus 135 days: **2011-07-27 12:50:00**. The exact 365-day observation history is **[2010-07-27 12:50:00, 2011-07-27 12:50:00]**, closed at both ends. The future outcome horizon is **(2011-07-27 12:50:00, 2011-12-09 12:50:00]**, open at the cutoff and closed at the dataset endpoint.

Thus, an event exactly at the cutoff belongs to pre-cutoff observation data, while an event strictly after the cutoff belongs to the outcome period. No event can belong to both periods. The exact design differs from the Step 5 calendar approximation because it includes timestamp-level events on the boundary dates according to these rules.

| boundary | timestamp | inclusion_rule | closed_or_open |
| --- | --- | --- | --- |
| observation_start | 2010-07-27 12:50:00 | >= observation_start | closed |
| cutoff | 2011-07-27 12:50:00 | <= cutoff for observation; > cutoff for outcome | observation closed / outcome open |
| outcome_end_dataset_end | 2011-12-09 12:50:00 | <= outcome_end | closed |
| outcome_start_condition | 2011-07-27 12:50:00 | purchase_timestamp strictly greater than cutoff | open at cutoff |

## 2. Primary cohort eligibility

There are **4,325** customers with at least one positive-purchase event in the exact observation window. Customers with baseline recency greater than 135 days are excluded. The final primary Population B contains **2,457 customers**.

Eligibility uses only positive-purchase events on or before the cutoff. The outcome table contains exactly one row per eligible customer and only minimal boundary, recency, outcome-count, and target fields.

## 3. Outcome target and class balance

For eligible customers, `Churn = 0` means at least one positive purchase strictly after the cutoff and no later than the dataset endpoint. `Churn = 1` means no positive purchase in that supported future horizon. Cancellations do not count as positive outcomes.

| Churn | label_meaning | customers | percentage_of_primary_population |
| --- | --- | --- | --- |
| 1 | No positive purchase in supported future horizon | 733 | 29.83 |
| 0 | At least one positive purchase in supported future horizon | 1,724 | 70.17 |

The final target distribution is **733 Churn=1 (29.83%)** and **1,724 Churn=0 (70.17%)**. No class rebalancing or threshold optimization was performed.

## 4. Comparison with Step 5 calendar-date audit

| metric | exact_timestamp_design | step5_calendar_design | difference_exact_minus_calendar | note |
| --- | --- | --- | --- | --- |
| observation_customers | 4,325 | 4,327 | -2 | Exact timestamp design uses cutoff 2011-07-27 12:50:00 and strict post-cutoff outcome entry. |
| primary_population_customers | 2,457 | 2,463 | -6 | Exact timestamp design uses cutoff 2011-07-27 12:50:00 and strict post-cutoff outcome entry. |
| future_purchasers | 1,724 | 1,727 | -3 | Exact timestamp design uses cutoff 2011-07-27 12:50:00 and strict post-cutoff outcome entry. |
| future_non_purchasers | 733 | 736 | -3 | Exact timestamp design uses cutoff 2011-07-27 12:50:00 and strict post-cutoff outcome entry. |

The exact timestamp design produces differences of **-2** observation-eligible customers, **-6** primary-cohort customers, **-3** future purchasers, and **-3** future non-purchasers relative to the Step 5 calendar audit. These differences arise only from timestamp-level placement around the cutoff and boundary dates; the previous Step 5 counts were not forced or overwritten.

## 5. Temporal leakage audit

| audit | passed | evidence |
| --- | --- | --- |
| observation eligibility uses only events within exact observation window | True | Observation filter is [observation_start, cutoff]. |
| baseline recency uses no post-cutoff transaction | True | Every stored last-before-cutoff timestamp is <= cutoff. |
| outcome uses only strictly post-cutoff positive purchases | True | Outcome filter is purchase_timestamp > cutoff and <= outcome_end. |
| outcome is inside supported future horizon | True | No outcome event exceeds dataset_end_timestamp. |
| no outcome information used for eligibility | True | Eligible customers are selected before outcome aggregation. |
| no predictive features created | True | Outcome table contains only cohort, boundary, recency, outcome-count, and Churn fields. |

Eligibility and baseline recency use no outcome-window activity. The target uses only strictly post-cutoff positive-purchase events within the supported horizon. Because predictive features do not yet exist, no outcome information can appear in predictors.

## 6. Edge-case audit

| edge_case | events | customers | eligible_customers | handling |
| --- | --- | --- | --- | --- |
| positive events exactly at cutoff timestamp | 1 | 1 | 1 | Included in observation/pre-cutoff data; excluded from outcome because outcome starts strictly after cutoff. |
| customers with positive purchases on both sides of cutoff | nan | 2,464 | 1,724 | Eligibility uses pre-cutoff events; future purchases determine outcome only. |
| eligible customers with multiple positive outcome purchases | 5,403 | 1,163 | 1,163 | Retained as an outcome-count audit; Churn remains 0 for any positive outcome purchase. |
| eligible customers with cancellation activity but no positive outcome purchase | 2,968 | 12 | 12 | Cancellations do not count as positive outcomes and do not reverse Churn=1. |
| eligible customers with positive purchase at final observed timestamp | 1 | 1 | 0 | Included because outcome end is closed. |
| eligible customers with positive purchase on final observed date | 41 | 35 | 28 | Included because the final observed date is within the supported outcome horizon. |
| duplicate Customer ID rows in final outcome table | 0 | 2,457 | 2,457 | Must be zero duplicates; exactly one row per eligible customer. |

The final outcome table has **0 duplicate Customer ID rows** and therefore has exactly one row per eligible customer. Customers with multiple outcome purchases remain `Churn=0`; the count is retained only for validation.

## 7. Validation and stopping point

- The outcome table is derived from validated invoice-level positive-purchase events, not line items.
- The exact timestamp cutoff is **2011-07-27 12:50:00** and the outcome end is **2011-12-09 12:50:00**.
- No raw or Step 4 canonical dataset was modified.
- No predictive feature engineering beyond minimal eligibility/outcome validation fields was performed.
- No train/test split, encoding, scaling, resampling, model training, or SHAP analysis occurred.
- The processed outcome table is Git-ignored.

Input hashes: positive base `825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503`, Step 5 events `4c6fee4faaa9214eff99e446fe7b2956473b7d250cd8852997799f3b4c8dc5b8`, canonical `a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42`, raw `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
