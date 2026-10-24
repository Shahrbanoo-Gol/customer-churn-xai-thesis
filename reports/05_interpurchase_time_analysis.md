# Step 5 — Interpurchase-Time Analysis and Data-Driven Churn-Threshold Discovery

## Scope and stopping point

This stage describes purchasing rhythm in the validated Step 4 positive-purchase base. It creates invoice-level purchase events, interpurchase intervals, descriptive tables, figures, and temporal-design options. It does **not** create a churn label, a final RFM or customer-modeling table, train/test data, encoded or scaled variables, a model, or SHAP output.

The input was `data/processed/03_positive_purchase_base.parquet`, produced under the validated primary methodology in which exact within-sheet duplicates were removed before reconciliation. The raw workbook and Step 4 inputs were read-only.

## 1. Purchase-event definition and validation

The positive-purchase base contains **779,425 line-item rows**, which collapse to **36,969 unique positive-purchase Invoice events** for **5,878 customers**. The validated purchase-event range is **2009-12-01 07:45 to 2011-12-09 12:50**. Each invoice maps to exactly one Customer ID. Invoices with multiple line items were counted once, not once per product line; **33,918** invoices contain multiple positive-purchase lines.

There are **64** invoices whose line items do not share exactly the same timestamp. Every one remains within a single calendar day, with only minute-level variation. The event timestamp is therefore the **earliest InvoiceDate within the invoice**, preserving a conservative and reproducible event ordering. The selected purchase date is its calendar date.

## 2. Customer purchase-event counts

| group | customers | percentage_of_customers |
| --- | --- | --- |
| exactly_1_purchase_event | 1,623 | 27.61 |
| exactly_2_purchase_events | 944 | 16.06 |
| 3_or_more_purchase_events | 3,311 | 56.33 |

The event-count distribution has mean **6.29**, median **3.00**, standard deviation **13.01**, minimum **1.00**, P25 **1.00**, P75 **7.00**, P90 **13.00**, P95 **21.00**, and maximum **398.00** events per customer. Only repeat customers contribute an observed interval; in this base, **4,255** customers contribute at least one interval.

## 3. Interpurchase intervals

Intervals are calculated between consecutive invoice-level events after chronological ordering within Customer ID. There are **31,091** intervals. **181** are zero-day intervals (**0.58%**); these are retained and represent distinct invoices on the same calendar day. No long intervals were removed as outliers.

| population | count | min | mean | median | std | p25 | p50 | p75 | p80 | p85 | p90 | p95 | p97_5 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_consecutive_interpurchase_intervals_days | 31,091 | 0.00 | 51.69 | 24.74 | 75.79 | 6.98 | 24.74 | 61.81 | 76.99 | 99.21 | 134.88 | 206.43 | 295.86 | 368.74 | 714.15 |

A positive-only sensitivity view is recorded in `67_positive_only_interval_descriptives.csv`; zero-day intervals were not silently excluded from the canonical interval dataset. In that sensitivity view, the median is **24.95 days** and the P90 is **135.13 days**.

## 4. Visual findings

`03_interpurchase_interval_distribution.png` shows both the full 0–720-day range and a 0–250-day view of the central mass. `04_interpurchase_ecdf.png` shows the cumulative percentage of observed repeat intervals completed by each elapsed time. The distribution is strongly right-skewed: the median interval is about **24.74 days**, while the maximum is **714.15 days**. The central mass rises quickly, but a material long tail remains. The third figure, `05_customer_typical_interval_distribution.png`, shows customer-level median intervals rather than pooling every interval.

## 5. Pooled intervals versus customer-level typical intervals

The pooled distribution weights customers by their number of intervals: a customer with many invoices contributes many observations. The customer-level distribution gives each repeat customer one summary statistic, such as that customer's median return interval. The latter answers how a typical repeat customer's usual interval varies, whereas the pooled distribution answers how often an observed consecutive purchase gap of a given length occurs. They are not interchangeable and neither is automatically the final churn basis.

| population | count | mean | median | std | p25 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| customer_level_median_interval_per_repeat_customer | 4,255 | 92.71 | 55.97 | 101.18 | 28.08 | 118.19 | 220.39 | 322.16 | 714.15 |
| customer_level_mean_interval_per_repeat_customer | 4,255 | 102.30 | 72.34 | 98.28 | 38.58 | 129.84 | 221.05 | 321.95 | 714.15 |
| customer_level_max_interval_per_repeat_customer | 4,255 | 174.93 | 151.83 | 122.64 | 80.06 | 244.98 | 352.24 | 391.15 | 714.15 |

## 6. Candidate inactivity thresholds

The candidate set is anchored to the observed P85, P90, and P95 interval percentiles, then rounded to the nearest five days for practical interpretation. Coverage is recomputed at the rounded threshold; no future model performance was used.

| candidate | anchor_percentile | exact_empirical_percentile_days | threshold_days | coverage_leq_threshold_pct | longer_than_threshold_pct |
| --- | --- | --- | --- | --- | --- |
| P85_rounded_to_5_days | 85 | 99.21 | 100 | 85.10 | 14.90 |
| P90_rounded_to_5_days | 90 | 134.88 | 135 | 90.04 | 9.96 |
| P95_rounded_to_5_days | 95 | 206.43 | 205 | 94.94 | 5.06 |

There is no single visually inevitable cutoff. A defensible primary candidate for later review is **135 days**, because it is the rounded P90 anchor: approximately **90.04%** of observed consecutive repeat intervals are no longer than this threshold and **9.96%** are longer. The **100-day** candidate is a shorter sensitivity option, while **205 days** is a more conservative sensitivity option that requires a longer future window. This is a recommendation for review, not an executed churn definition.

## 7. Right censoring

The last positive-purchase observation is **2011-12-09 12:50**. A customer whose last observed purchase is too close to that endpoint cannot be assessed over a full inactivity horizon without risking false apparent inactivity. For a threshold of T days, the latest fully supported calendar cutoff is the dataset end date minus T days.

| threshold_days | latest_supported_cutoff_date | future_outcome_start_date | available_future_days | customers_with_final_observed_purchase_after_latest_supported_cutoff | percentage_of_positive_purchase_customers_affected |
| --- | --- | --- | --- | --- | --- |
| 100 | 2011-08-31 | 2011-09-01 | 100 | 2,973 | 50.58 |
| 135 | 2011-07-27 | 2011-07-28 | 135 | 3,217 | 54.73 |
| 205 | 2011-05-18 | 2011-05-19 | 205 | 3,617 | 61.53 |

For the recommended review candidate of **135 days**, the latest supported cutoff is **2011-07-27**. Customers whose final observed purchase is later than that cutoff are end-of-study censored for this horizon; they are **not** labeled churn or non-churn here.

## 8. Proposed observation/outcome-window designs

The following planning designs use a fixed 365-day observation window and reserve the entire remaining period through 2011-12-09 for the future outcome window. Features in a later stage must use only transactions on or before the cutoff; outcome behavior must use only transactions after it. Counts are eligibility planning estimates, not a final modeling table.

| design | threshold_days | observation_window_start | observation_window_end_cutoff | future_outcome_window_start | future_outcome_window_end | available_future_days | approx_eligible_customers_at_least_1_observation_event | approx_eligible_customers_at_least_2_observation_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 365_day_observation_plus_100_day_outcome | 100 | 2010-09-01 | 2011-08-31 | 2011-09-01 | 2011-12-09 | 100 | 4,335 | 2,761 |
| 365_day_observation_plus_135_day_outcome | 135 | 2010-07-28 | 2011-07-27 | 2011-07-28 | 2011-12-09 | 135 | 4,327 | 2,790 |
| 365_day_observation_plus_205_day_outcome | 205 | 2010-05-19 | 2011-05-18 | 2011-05-19 | 2011-12-09 | 205 | 4,335 | 2,795 |

All three designs fully support their associated threshold by construction. The 100-day design preserves the latest cutoff and more future-eligible customers; the 135-day design balances a longer horizon with the P90 empirical support; the 205-day design is the most conservative and loses more historical recency from the end-of-study constraint. These are temporal options, not labels.

## 9. Temporal-design addendum: separate decisions

Two decisions must remain separate. First, the **inactivity threshold** is an empirical candidate derived from the pooled interpurchase-interval distribution: 100, 135, and 205 days correspond approximately to the P85, P90, and P95 behavior. Second, the **observation-window length** is an analytical design choice. The 365-day window was not selected by the interpurchase percentile analysis; it is being considered because it captures a complete annual cycle and provides a consistent history length for later features.

The comparison below evaluates the provisional 135-day cutoff (**2011-07-27**) and its fully supported outcome window (**2011-07-28 through 2011-12-09**).

## 10. Observation-window comparison at the 135-day cutoff

| observation_option | observation_start | observation_end_cutoff | observation_days_inclusive | eligible_customers_with_at_least_1_positive_purchase | eligible_customer_percentage_of_5878 | median_purchase_event_count_per_eligible_customer | percentage_with_only_1_observed_event | complete_annual_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 180_day_observation | 2011-01-29 | 2011-07-27 | 180 | 2,771 | 47.14 | 2.00 | 47.13 | False |
| 270_day_observation | 2010-10-31 | 2011-07-27 | 270 | 3,609 | 61.40 | 2.00 | 39.79 | False |
| 365_day_observation | 2010-07-28 | 2011-07-27 | 365 | 4,327 | 73.61 | 2.00 | 35.52 | True |
| all_available_pre_cutoff_history | 2009-12-01 | 2011-07-27 | 604 | 5,126 | 87.21 | 3.00 | 30.36 | True |

The 180-day and 270-day options provide shorter, more recent histories but do not cover a full annual cycle. The 365-day option covers one complete annual cycle and retains a substantial eligible population (**4,327 customers; 73.61% of positive-purchase customers). All available pre-cutoff history maximizes historical evidence and also covers a complete annual cycle, but produces unequal effective history lengths and gives older transactions more influence. The 365-day choice is therefore a design rationale, not an empirical consequence of the P90 threshold and not a choice made solely to maximize sample size.

Advantages and limitations for each option are recorded in table 68. No option is converted into a final modeling table here.

## 11. Baseline recency and eligibility

Among customers with at least one positive purchase in the 365-day observation window, **4,327** are available. Using only purchases on or before 2011-07-27, **2,463** have baseline recency <= 135 days and **1,864** have baseline recency > 135 days.

| population | recency_condition | customers | percentage_of_365_day_eligible | threshold_days |
| --- | --- | --- | --- | --- |
| all_365_day_observation_eligible_customers | all customers with at least one positive purchase in the 365-day observation window | 4,327 | 100.00 | 135 |
| baseline_recency_leq_135_days | latest purchase on or after 2011-03-14; recency <= 135 days | 2,463 | 56.92 | 135 |
| baseline_recency_gt_135_days | latest purchase before 2011-03-14; recency > 135 days | 1,864 | 43.08 | 135 |

Population A includes every customer observed during the prior year, including customers who had already gone beyond 135 days without a purchase at the cutoff. Population B restricts the prediction population to customers who had not yet exceeded 135 days of inactivity at the cutoff. The latter is a narrower, intervention-oriented population; the former answers a broader monitoring question and includes customers who may already require reactivation rather than preventive retention.

## 12. Aggregate future-return assessment

The following counts use the 135-day future window only for design assessment. They are aggregate counts; no customer-level outcome or `Churn` variable was saved.

| population | population_customers | customers_with_at_least_1_positive_purchase_in_outcome_window | percentage_with_positive_purchase_in_outcome_window | customers_with_no_positive_purchase_in_outcome_window | percentage_with_no_positive_purchase_in_outcome_window |
| --- | --- | --- | --- | --- | --- |
| population_A | 4,327 | 2,364 | 54.63 | 1,963 | 45.37 |
| population_B | 2,463 | 1,727 | 70.12 | 736 | 29.88 |

Population A asks: **“Given customers observed during the prior year, who will make no purchase during the next 135 days?”** Population B asks: **“Among customers who have not yet exceeded 135 days of inactivity at the prediction date, who will fail to purchase during the next 135 days?”** Population B is more closely aligned with a managerial retention/intervention use case because it excludes customers who had already crossed the proposed inactivity horizon at the intervention date. Population A remains useful if the intended business action includes both preventive retention and reactivation. The aggregate outcome split was not used to optimize the threshold.

### Provisional design recommendation

For methodological review, retain the **365-day observation window** because it covers a complete annual cycle while keeping a substantial and consistently defined eligible population. For the modeling population, **Population B** is the more targeted provisional choice for retention intervention; Population A should remain a documented sensitivity or broader-monitoring alternative. Neither recommendation creates or locks the final churn label.

## 13. Pre-cutoff threshold-stability audit

This audit recalculates invoice-level intervals using only selected invoice timestamps on or before **2011-07-27**. It excludes **8,580** post-cutoff purchase events from the stability calculation. The previously validated full-history interval dataset and statistics were not altered.

| population | cutoff_date | eligible_purchase_events | customers | repeat_customers | total_interpurchase_intervals | mean_days | median_days | p75_days | p80_days | p85_days | p90_days | p95_days | p97_5_days | p99_days | maximum_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre_cutoff_only | 2011-07-27 | 28,389 | 5,126 | 3,570 | 23,263 | 46.37 | 23.92 | 58.10 | 72.08 | 91.99 | 123.00 | 176.82 | 229.98 | 305.98 | 595.90 |

### Full history versus pre-cutoff-only intervals

| statistic | full_history_value_days | pre_cutoff_value_days | absolute_difference_pre_minus_full_days | percentage_difference_vs_full |
| --- | --- | --- | --- | --- |
| median_days | 24.74 | 23.92 | -0.83 | -3.35 |
| p75_days | 61.81 | 58.10 | -3.71 | -6.00 |
| p85_days | 99.21 | 91.99 | -7.22 | -7.28 |
| p90_days | 134.88 | 123.00 | -11.87 | -8.80 |
| p95_days | 206.43 | 176.82 | -29.60 | -14.34 |

At 135 days, the pre-cutoff interval distribution has empirical coverage of **91.54%**: **91.54%** of intervals are <=135 days and **8.46%** are longer. The pre-cutoff P90 is **123.00 days**, compared with the full-history P90 of **134.88 days**. The provisional threshold is therefore **reasonably supported**, rather than strongly supported: it retains approximately P90-level coverage, but the pre-cutoff P90 is about **12.00 days** below 135.

### Optional historical stability diagnostic

The pre-cutoff interval sample was split at **2010-09-28** by the later event date. This is a descriptive stability check, not a second threshold-selection exercise.

| historical_portion | portion_start | portion_end | interval_count | repeat_customers | median_days | p90_days | p95_days | coverage_leq_135_days_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earlier_pre_cutoff | 2009-12-01 | 2010-09-28 | 10,013 | 2,207 | 20.90 | 90.91 | 124.86 | 96.17 |
| later_pre_cutoff | 2010-09-29 | 2011-07-27 | 13,250 | 3,132 | 27.80 | 151.39 | 213.47 | 88.04 |

The earlier portion has a P90 of **90.91 days**, while the later portion has a P90 of **151.39 days**. This is meaningful temporal variation, so 135 days remains defensible as a provisional practical threshold but should be reviewed before it is locked as the final primary definition.

## 14. Separation of findings, candidates, and decisions

### Empirical findings

- Positive purchases produce **36,969** invoice events across **5,878** customers.
- **31,091** consecutive-event intervals are observed across **4,255** repeat customers.
- The interval distribution is right-skewed, includes **181** retained zero-day intervals, and has a long tail to approximately **714.15 days**.
- Invoice events are unique by invoice and customer-consistent; line items were not treated as separate purchases.

### Candidate churn definitions

The data support reviewing **100**, **135**, and **205** days, anchored respectively to the empirical P85, P90, and P95 interval behavior. No final inactivity threshold has been selected or encoded.

### Temporal-design options

The 365-day observation designs in table 65 pair each candidate with a fully supported future outcome window ending on the dataset endpoint. Table 68 separately compares the 180-day, 270-day, 365-day, and all-available alternatives for the provisional 135-day cutoff. They prevent future transactions from entering later cutoff-based features, subject to implementing the cutoff rule exactly.

### Decision still required

Before churn labels are generated, the final primary threshold and observation/outcome-window design must be reviewed. **Recommendation for that review:** use **135 days** with the matching 365-day observation window as the provisional primary candidate, and retain the 100-day and 205-day designs as sensitivity options. This recommendation was not executed as a label.

## Validation and scope controls

- Step 5 starts from the validated Step 4 positive-purchase base and the approved pre-reconciliation duplicate methodology.
- Invoice events are unique positive-purchase invoices, customer-consistent, and chronologically ordered within customer.
- Intervals use only adjacent observed events for descriptive timing analysis; no historical feature table was constructed.
- No `Churn` column, churn label, final RFM/customer modeling feature table, observation label, or outcome label was created.
- No train/test split, encoding, scaling, model training, or SHAP analysis occurred.
- Raw Excel and Step 4 canonical/positive-purchase inputs remained unchanged by hash verification.
- Step 5 processed analytical datasets are written under `data/processed` and remain Git-ignored.
- Temporal-design outputs are aggregate audits only; no customer-level future-return or Churn field was persisted.
- Threshold-stability intervals used only events on or before the 2011-07-27 cutoff; post-cutoff events were excluded from that calculation.

Input hashes recorded for reproducibility: positive-purchase `825bccc6ebdcbf594001729ad50e5d7a292ef2eb0456459dca2b7d15e30ee503`, canonical `a97e6f7279276417727b76bf7ee326b0653c0cc3b8f83526a24682086a8ace42`, raw `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
