# Chapter 4.5 descriptive visualization report

## Scope and data source

The figures use the finalized customer-level feature table at `data\processed\07_customer_features.parquet`. No features were rebuilt from raw transactions. The table contains exactly **2,457 customers**, with one row per unique `Customer ID`; `Churn` matches the Step 6 outcome table exactly. The dataset hash was unchanged during this descriptive step.

All figures are white-background PNG files exported at 300 DPI. The analysis is descriptive: it does not establish causality, determine feature importance, or replace the later model-development and SHAP analyses.

## Figure 1 — Core RFM distributions

File: `outputs\figures\chapter4_5_rfm_distributions.png`

1. **What it shows.** Separate empirical histograms for `RecencyDays`, `Frequency`, and `Monetary`, with mean and median reference lines and the full cohort sample size.
2. **Main numerical observation.** `RecencyDays` has median 43.02 days and mean 50.86 days, while `Frequency` has median 3 and mean 5.83. `Monetary` has median 1,070.38 and mean 2,784.04; its maximum is 251,520.82.
3. **Managerial interpretation.** Most customers are relatively low-to-moderate frequency and value customers, while a smaller group contributes substantially higher purchasing activity or economic value. The median is therefore a useful description of a typical customer alongside the mean.
4. **What must not be inferred.** The high-value tail is not automatically erroneous, and the figure does not show that high monetary value or frequency causes churn.

## Figure 2 — Selected behavioral feature distributions and skewness

Files: `outputs\figures\chapter4_5_selected_feature_distributions.png` and `outputs\figures\chapter4_5_selected_feature_distributions_log_view.png`

1. **What it shows.** Original-scale histograms for `TotalQuantity`, `AverageOrderValue`, `UniqueProducts`, and `CancellationValue`, annotated with mean, median, and sample skewness. The companion view uses `log1p` only to make the shape of long-tailed distributions easier to inspect.
2. **Main numerical observation.** Skewness is positive for all four selected features: `TotalQuantity` 16.09, `AverageOrderValue` 14.92, `UniqueProducts` 5.67, and `CancellationValue` 24.66. The original stored values were not log-transformed, imputed, winsorized, truncated, or otherwise changed.
3. **Managerial interpretation.** Customer behavior is concentrated among lower/moderate activity levels, with a smaller group showing much larger baskets, product breadth, order values, or cancellation values. This supports segment-aware descriptive monitoring.
4. **What must not be inferred.** The log1p panels are visualization transformations only; they are not evidence that a transformed feature was used in the model pipeline, and skewness alone is not evidence of invalid data.

## Figure 3 — Customer heterogeneity and extreme values

File: `outputs\figures\chapter4_5_customer_heterogeneity_boxplots.png`

1. **What it shows.** Separate boxplots for `Frequency`, `Monetary`, `TotalQuantity`, `UniqueProducts`, and `AverageOrderValue`. Every observation remains included; a logarithmic axis is used only for display.
2. **Main numerical observation.** The medians are `Frequency` 3.00, `Monetary` 1,070.38, `TotalQuantity` 606.00, `UniqueProducts` 52.00, `AverageOrderValue` 303.67. The corresponding maxima extend to `Frequency` 183.00, `Monetary` 251,520.82, `TotalQuantity` 178,396.00, `UniqueProducts` 1,604.00, `AverageOrderValue` 14,844.77.
3. **Managerial interpretation.** The cohort is heterogeneous: a small number of customers exhibit substantially higher purchasing activity or economic value than the typical customer. These customers may merit separate descriptive review or differentiated service design.
4. **What must not be inferred.** Extreme observations must not be treated as errors merely because they are large, and the boxplots do not identify causal drivers or predictive importance.

## Figure 4 — Feature correlation heatmap

File: `outputs\figures\chapter4_5_feature_correlation_heatmap.png`  
Companion table: `outputs\tables\chapter4_5_high_correlations.csv`

1. **What it shows.** Pearson correlations among numeric customer-level predictors, excluding `Customer ID`, `Churn`, and raw `CountryCandidate`. Strong absolute correlations (|r| ≥ 0.80) are annotated.
2. **Main numerical observation.** The strongest associations are:

| Feature A | Feature B | Pearson r | Pairwise n |
| --- | --- | ---: | ---: |
| MeanInterpurchaseDays | MedianInterpurchaseDays | 0.9806 | 2,012 |
| Frequency | ActivePurchaseDays | 0.9719 | 2,457 |
| CustomerAgeDays | PurchaseSpanDays | 0.9490 | 2,457 |
| Monetary | TotalQuantity | 0.8793 | 2,457 |
| HasCancellation | CancellationInvoiceRate | 0.8361 | 2,457 |

The `Monetary`–`TotalQuantity` correlation is 0.8793, which is lower than the approximate 0.920 reference value supplied for the completed analysis.
3. **Managerial interpretation.** Strong correlations indicate overlapping descriptive information and may signal redundancy when constructing a parsimonious feature set. This is useful for later modeling review and stakeholder interpretation.
4. **What must not be inferred.** Correlation does not establish causality, and it does not by itself determine which feature should be removed or which feature will be important for churn prediction.

## Figure 5 — Cancellation behavior

File: `outputs\figures\chapter4_5_cancellation_customer_counts.png`

1. **What it shows.** A bar chart comparing customers with at least one observed cancellation against customers without observed cancellation activity.
2. **Main numerical observation.** 1,144 customers (46.56%) have at least one cancellation, while 1,313 (53.44%) do not.
3. **Managerial interpretation.** Cancellation activity is present for a substantial share of the cohort, so it is a meaningful descriptive dimension for customer-behavior profiling and service-process review.
4. **What must not be inferred.** The chart does not show that cancellation causes churn or that customers with cancellations are necessarily dissatisfied.

## Statistical summary and structural missingness

The descriptive statistics used for the plotted quantitative features are in `outputs\tables\chapter4_5_visual_summary.csv`. `MeanInterpurchaseDays` and `MedianInterpurchaseDays` each have **445 missing values** because these customers have only one positive purchase event, so no adjacent interpurchase interval exists. This is structural missingness, not an ordinary data-quality failure; no imputation was performed.

## Reference-value verification

- **RecencyDays median:** actual `43.0181`; expected reference approximately `43.0200`; consistent within rounding.
- **RecencyDays mean:** actual `50.8599`; expected reference approximately `50.8600`; consistent within rounding.
- **RecencyDays maximum:** actual `134.9563`; expected reference approximately `134.9600`; consistent within rounding.
- **Frequency median:** actual `3.0000`; expected reference approximately `3.0000`; consistent within rounding.
- **Frequency mean:** actual `5.8339`; expected reference approximately `5.8300`; consistent within rounding.
- **Frequency maximum:** actual `183.0000`; expected reference approximately `183.0000`; consistent within rounding.
- **Monetary median:** actual `1,070.3800`; expected reference approximately `1,070.3800`; consistent within rounding.
- **Monetary mean:** actual `2,784.0437`; expected reference approximately `2,784.0400`; consistent within rounding.
- **Monetary maximum:** actual `251,520.8200`; expected reference approximately `251,520.8200`; consistent within rounding.
- **HasCancellation count:** actual `1,144.0000`; expected reference approximately `1,144.0000`; consistent within rounding.

The expected reference for `Monetary`–`TotalQuantity` was approximately 0.920; the finalized table yields 0.8793. This is the only material discrepancy identified against the supplied reference values.

## Validation completed

- Customer rows: **2,457**; unique `Customer ID`: **2,457**.
- `Churn`: unchanged and exactly equal to `06_primary_customer_outcome_table.parquet`.
- Existing Step 7 reconciliation checks for no post-cutoff positive-purchase, event, and cancellation rows: **passed**.
- Feature-table schema and values: source hash unchanged before/after (`d1781ea30db17701c0d243ea2371bac5f59ce85c6fe148ce8dc5c02f85c72768`); no feature was deleted or altered.
- No models were retrained and no preprocessing decisions were changed.
- Existing figures were not overwritten; only new `chapter4_5_*` files were created.

