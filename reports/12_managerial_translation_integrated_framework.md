# Step 12 — Managerial Translation, Integrated Decision Framework, and Final Sensitivity Analysis

## Scope and primary-model guardrail

Step 12 translates the corrected Step 11 SHAP evidence into managerial categories, connects prediction to explanation and intervention-option families, and tests robustness around the 135-day churn design. The 135-day XGBoost model remains the primary thesis model. Its Parsimonious feature set, Train/Test split, preprocessing, hyperparameters, random state, threshold, Step 10 metrics, and Step 11 SHAP evidence were not changed.

The 120-day and 150-day scenarios are secondary robustness checks. They use the same conceptual cohort design, exact timestamp rules, 365-day observation history, 80/20 stratified split with `random_state=42`, Train-only preprocessing, and fixed XGBoost configuration. No alternative threshold was optimized.

## A. Managerial feature categories

Every one of the 16 Parsimonious business predictors is assigned exactly once.

| business_feature | managerial_category | business_meaning |
| --- | --- | --- |
| Monetary | Customer economic value | Total positive purchase value attributed to the customer |
| AverageOrderValue | Customer economic value | Average positive purchase invoice value |
| Frequency | Purchase activity and intensity | Count of positive purchase invoices/events in the observation period |
| TotalQuantity | Purchase activity and intensity | Total quantity purchased in positive purchase events |
| AverageQuantityPerOrder | Purchase activity and intensity | Average quantity purchased per positive invoice |
| RecencyDays | Purchase recency and cadence | Days since the customer's last positive purchase at the cutoff |
| MedianInterpurchaseDays | Purchase recency and cadence | Median gap between positive purchase events |
| UniqueProducts | Product breadth and basket behavior | Number of distinct products purchased |
| ProductsPerOrder | Product breadth and basket behavior | Average number of distinct products per positive invoice |
| CustomerAgeDays | Relationship history | Days from the customer's first positive purchase to the cutoff |
| HasRepeatPurchaseHistory | Relationship history | Indicator that the customer has more than one positive purchase event |
| CancellationValue | Cancellation behavior | Absolute value of cancellation quantities multiplied by price |
| CancellationInvoiceRate | Cancellation behavior | Cancellation invoice count relative to positive purchase invoice count |
| CancellationInvoiceCount | Cancellation behavior | Count of cancellation/return invoices linked to the customer |
| HasCancellation | Cancellation behavior | Indicator that the customer has at least one cancellation/return invoice |
| CountryCandidate | Geographic context | Country recorded for the customer's linked transactions |

## B. Category-level SHAP importance

For each customer, signed SHAP contributions were first summed across the original/business features in each category. Category importance is then `mean(abs(category signed contribution))`. CountryCandidate uses the corrected Step 11 grouped contribution from its 30 one-hot columns before entering the Geographic context category.

| rank | managerial_category | underlying_features | mean_absolute_shap | relative_share_pct |
| --- | --- | --- | --- | --- |
| 1 | Purchase activity and intensity | Frequency, TotalQuantity, AverageQuantityPerOrder | 0.619 | 41.138 |
| 2 | Customer economic value | Monetary, AverageOrderValue | 0.433 | 28.764 |
| 3 | Purchase recency and cadence | RecencyDays, MedianInterpurchaseDays | 0.258 | 17.169 |
| 4 | Product breadth and basket behavior | UniqueProducts, ProductsPerOrder | 0.070 | 4.680 |
| 5 | Cancellation behavior | CancellationValue, CancellationInvoiceRate, CancellationInvoiceCount, HasCancellation | 0.066 | 4.370 |
| 6 | Relationship history | CustomerAgeDays, HasRepeatPurchaseHistory | 0.044 | 2.956 |
| 7 | Geographic context | CountryCandidate | 0.014 | 0.923 |

Category shares sum to approximately 100%. The category importance figure is `21_step12_managerial_category_shap_importance.png`.

## C. Managerial interpretation

| managerial_category | typical_direction_or_pattern | dominant_underlying_feature_by_rank | managerial_meaning | what_to_monitor |
| --- | --- | --- | --- | --- |
| Customer economic value | Mixed: higher Monetary generally pushes away from Churn, while higher AverageOrderValue generally pushes toward Churn. | Monetary | Observed customer value and order-value intensity. | Monitor value and order-value patterns without assuming higher value causes retention. |
| Purchase activity and intensity | Predominantly away from Churn: Frequency and TotalQuantity show negative direction; AverageQuantityPerOrder is mixed/nonlinear. | Frequency | How often and how much the customer purchases. | Monitor purchase frequency and volume changes over time. |
| Purchase recency and cadence | Mixed: higher RecencyDays pushes toward Churn, while MedianInterpurchaseDays is nonlinear/mixed. | RecencyDays | How recently and regularly purchase events occur. | Monitor elapsed time since last purchase and deviations from observed cadence. |
| Product breadth and basket behavior | Predominantly away from Churn for UniqueProducts; ProductsPerOrder is mixed/nonlinear. | UniqueProducts | Breadth of products and typical basket composition. | Monitor product breadth and basket composition changes. |
| Relationship history | Mixed/nonlinear: HasRepeatPurchaseHistory=1 generally pushes away from Churn; CustomerAgeDays is mixed. | CustomerAgeDays | Observable history length and repeat-purchase status in the window. | Distinguish short observable histories from established repeat-purchase histories. |
| Cancellation behavior | Mixed: higher CancellationValue pushes toward Churn, while CancellationInvoiceRate and CancellationInvoiceCount generally push away; HasCancellation is weak/near zero. | CancellationValue | Observed cancellation frequency and value signals. | Monitor cancellations and investigate service, delivery, payment, or process context. |
| Geographic context | Country-specific/mixed and low importance; contextual rather than a primary trigger. | CountryCandidate | Recorded country context, used primarily as descriptive context because its SHAP importance is low. | Use geography as context rather than a primary retention trigger. |

These are prediction signals. For example, longer observed recency was associated with higher predicted churn risk in the fitted model; this does not establish that changing recency would cause a customer to remain active.

## D. Intervention-option mapping

| managerial_category | possible_intervention_family | evidence_guardrail |
| --- | --- | --- |
| Customer economic value | Differentiated retention treatment; loyalty recognition; benefit level aligned with observed value. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Purchase activity and intensity | Engagement campaign; personalized next-purchase recommendation; frequency-building incentive for testing. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Purchase recency and cadence | Reactivation communication; purchase or replenishment reminder where relevant; limited-time re-engagement offer for testing. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Product breadth and basket behavior | Product discovery; cross-category recommendations; personalized assortment communication. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Relationship history | Onboarding or early-relationship nurture for shorter histories; loyalty recognition for established histories. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Cancellation behavior | Investigate cancellation reason; service-recovery contact; delivery/payment/process review before promotion. | Conceptual option; effectiveness requires future measurement or controlled testing. |
| Geographic context | Contextual segmentation only; do not use as a primary trigger without future evidence. | Conceptual option; effectiveness requires future measurement or controlled testing. |

The options are hypotheses for managerial review and testing. Online Retail II does not establish that any listed action reduces churn.

## E. Customer-level decision examples

The examples use the objectively selected Step 11 Test cases and anonymized labels. Risk ranks are within the 492-customer primary Test set.

| case_id | case_type | predicted_probability | risk_rank_among_test | main_shap_driver | main_managerial_category | possible_intervention_family |
| --- | --- | --- | --- | --- | --- | --- |
| Case A | True Positive | 0.85396915674209595 | 1 | RecencyDays | Purchase activity and intensity | Engagement campaign; personalized next-purchase recommendation; frequency-building incentive for testing. |
| Case B | True Negative | 0.02081900276243687 | 492 | MedianInterpurchaseDays | Cancellation behavior | Investigate cancellation reason; service-recovery contact; delivery/payment/process review before promotion. |
| Case C | False Positive | 0.78399145603179932 | 7 | Monetary | Customer economic value | Differentiated retention treatment; loyalty recognition; benefit level aligned with observed value. |
| Case D | False Negative | 0.34993505477905273 | 217 | Monetary | Customer economic value | Differentiated retention treatment; loyalty recognition; benefit level aligned with observed value. |

The workflow is: identify modeled risk, inspect the strongest signed SHAP contributions, translate them to a category, and select an option family for review. A customer-level SHAP explanation is not a causal diagnosis.

## F. Integrated churn-intelligence framework

| stage | stage_name | input_or_action | output | decision_guardrail |
| --- | --- | --- | --- | --- |
| 1 | Risk prediction | Transaction-derived customer features | Locked XGBoost churn probability | Primary model and threshold remain locked. |
| 2 | Customer prioritization | Rank by predicted risk | Resource-sized Top 10%, Top 20%, or Top 30% groups | No universally optimal group size; budget and intervention cost are external. |
| 3 | Explanation | Customer-level SHAP values | Positive and negative model contributors | SHAP explains model behavior, not causality. |
| 4 | Managerial driver translation | Group features into seven categories | Dominant managerial signal | Categories are interpretive, not new model variables. |
| 5 | Intervention-option matching | Dominant category plus business context | Candidate action family for review/testing | Options are not proven treatments. |
| 6 | Future measurement / experimentation | Selected action and comparison group | Measured intervention effectiveness | Requires future controlled testing or A/B experimentation. |

The framework diagram is `22_step12_integrated_churn_intelligence_framework.png`. Prior Step 10 XGBoost ranking evidence shows Top 10%, Top 20%, and Top 30% groups capturing approximately 23.1%, 40.1%, and 53.7% of Test Churn customers, respectively, with lift of 2.276, 1.995, and 1.787. These are resource-sizing reference points; no targeting percentage is universally optimal without budget and intervention-cost information.

The final stage is future measurement or controlled experimentation. It is required to estimate intervention effectiveness and is not supplied by this dataset.

## G. 120/135/150-day sensitivity comparison

| horizon_days | scenario_role | prediction_cutoff | cohort_size | churn_count | churn_pct | train_size | test_size | test_roc_auc | test_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | secondary_shorter | 2011-08-11 12:50:00 | 2281 | 668 | 29.285 | 1824 | 457 | 0.739 | 0.547 |
| 135 | primary | 2011-07-27 12:50:00 | 2457 | 733 | 29.833 | 1965 | 492 | 0.777 | 0.578 |
| 150 | secondary_longer | 2011-07-12 12:50:00 | 2554 | 773 | 30.266 | 2043 | 511 | 0.748 | 0.545 |

The 135-day row is the primary design and uses the existing Step 10 performance evidence. The 120- and 150-day rows are secondary checks. The comparison figure is `23_step12_sensitivity_comparison.png`.

## H. SHAP ranking stability

| comparison | spearman_rank_correlation_16_features | top5_overlap_count | top5_overlap_features | rank_positions_Monetary | rank_positions_Frequency | rank_positions_RecencyDays | rank_positions_TotalQuantity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 120 vs 135 days | 0.918 | 4 | Frequency, Monetary, RecencyDays, TotalQuantity | 2 vs 1 | 1 vs 2 | 3 vs 3 | 4 vs 4 |
| 135 vs 150 days | 0.929 | 4 | Frequency, Monetary, RecencyDays, TotalQuantity | 1 vs 3 | 2 vs 2 | 3 vs 4 | 4 vs 1 |
| 120 vs 150 days | 0.926 | 5 | Frequency, MedianInterpurchaseDays, Monetary, RecencyDays, TotalQuantity | 2 vs 3 | 1 vs 2 | 3 vs 4 | 4 vs 1 |

Business-feature rankings by horizon:

| horizon_days | scenario_role | business_feature | rank | mean_absolute_shap | relative_share_pct | direction_or_pattern |
| --- | --- | --- | --- | --- | --- | --- |
| 120 | secondary_shorter | Frequency | 1 | 0.559 | 31.778 | higher values generally push away from Churn |
| 120 | secondary_shorter | Monetary | 2 | 0.350 | 19.907 | higher values generally push away from Churn |
| 120 | secondary_shorter | RecencyDays | 3 | 0.225 | 12.775 | higher values generally push toward Churn |
| 120 | secondary_shorter | TotalQuantity | 4 | 0.203 | 11.540 | higher values generally push away from Churn |
| 120 | secondary_shorter | MedianInterpurchaseDays | 5 | 0.103 | 5.876 | nonlinear/mixed |
| 120 | secondary_shorter | UniqueProducts | 6 | 0.059 | 3.347 | higher values generally push away from Churn |
| 120 | secondary_shorter | AverageQuantityPerOrder | 7 | 0.059 | 3.345 | nonlinear/mixed |
| 120 | secondary_shorter | AverageOrderValue | 8 | 0.038 | 2.163 | higher values generally push toward Churn |
| 120 | secondary_shorter | CustomerAgeDays | 9 | 0.037 | 2.120 | nonlinear/mixed |
| 120 | secondary_shorter | ProductsPerOrder | 10 | 0.037 | 2.104 | nonlinear/mixed |
| 120 | secondary_shorter | CancellationValue | 11 | 0.024 | 1.385 | higher values generally push toward Churn |
| 120 | secondary_shorter | CountryCandidate | 12 | 0.023 | 1.326 | country-specific/mixed |
| 120 | secondary_shorter | CancellationInvoiceRate | 13 | 0.021 | 1.196 | higher values generally push away from Churn |
| 120 | secondary_shorter | CancellationInvoiceCount | 14 | 0.020 | 1.139 | higher values generally push away from Churn |
| 120 | secondary_shorter | HasRepeatPurchaseHistory | 15 | 0.000 | 0.000 | value 1 generally pushes away from Churn |
| 120 | secondary_shorter | HasCancellation | 16 | 0.000 | 0.000 | value 1 generally pushes away from Churn |
| 135 | primary | Monetary | 1 | 0.466 | 26.034 | higher values generally push away from Churn |
| 135 | primary | Frequency | 2 | 0.441 | 24.649 | higher values generally push away from Churn |
| 135 | primary | RecencyDays | 3 | 0.252 | 14.071 | higher values generally push toward Churn |
| 135 | primary | TotalQuantity | 4 | 0.213 | 11.883 | higher values generally push away from Churn |
| 135 | primary | AverageOrderValue | 5 | 0.093 | 5.191 | higher values generally push toward Churn |
| 135 | primary | CancellationValue | 6 | 0.070 | 3.884 | higher values generally push toward Churn |
| 135 | primary | MedianInterpurchaseDays | 7 | 0.056 | 3.107 | nonlinear/mixed |
| 135 | primary | UniqueProducts | 8 | 0.054 | 2.999 | higher values generally push away from Churn |
| 135 | primary | CustomerAgeDays | 9 | 0.043 | 2.399 | nonlinear/mixed |
| 135 | primary | AverageQuantityPerOrder | 10 | 0.043 | 2.375 | nonlinear/mixed |
| 135 | primary | ProductsPerOrder | 11 | 0.030 | 1.697 | nonlinear/mixed |
| 135 | primary | CountryCandidate | 12 | 0.014 | 0.776 | country-specific/mixed |
| 135 | primary | CancellationInvoiceRate | 13 | 0.007 | 0.412 | higher values generally push away from Churn |
| 135 | primary | HasRepeatPurchaseHistory | 14 | 0.005 | 0.304 | value 1 generally pushes away from Churn |
| 135 | primary | CancellationInvoiceCount | 15 | 0.004 | 0.219 | higher values generally push away from Churn |
| 135 | primary | HasCancellation | 16 | 0.000 | 0.000 | value 1 generally pushes away from Churn |
| 150 | secondary_longer | TotalQuantity | 1 | 0.451 | 23.612 | higher values generally push away from Churn |
| 150 | secondary_longer | Frequency | 2 | 0.365 | 19.085 | higher values generally push away from Churn |
| 150 | secondary_longer | Monetary | 3 | 0.340 | 17.817 | higher values generally push away from Churn |
| 150 | secondary_longer | RecencyDays | 4 | 0.205 | 10.719 | higher values generally push toward Churn |
| 150 | secondary_longer | MedianInterpurchaseDays | 5 | 0.114 | 5.967 | nonlinear/mixed |
| 150 | secondary_longer | UniqueProducts | 6 | 0.113 | 5.907 | higher values generally push away from Churn |
| 150 | secondary_longer | AverageOrderValue | 7 | 0.094 | 4.925 | higher values generally push toward Churn |
| 150 | secondary_longer | CustomerAgeDays | 8 | 0.069 | 3.629 | nonlinear/mixed |
| 150 | secondary_longer | CancellationValue | 9 | 0.053 | 2.791 | higher values generally push toward Churn |
| 150 | secondary_longer | CancellationInvoiceRate | 10 | 0.040 | 2.114 | nonlinear/mixed |
| 150 | secondary_longer | AverageQuantityPerOrder | 11 | 0.028 | 1.483 | nonlinear/mixed |
| 150 | secondary_longer | ProductsPerOrder | 12 | 0.012 | 0.627 | nonlinear/mixed |
| 150 | secondary_longer | CountryCandidate | 13 | 0.011 | 0.574 | country-specific/mixed |
| 150 | secondary_longer | HasRepeatPurchaseHistory | 14 | 0.011 | 0.560 | value 1 generally pushes away from Churn |
| 150 | secondary_longer | CancellationInvoiceCount | 15 | 0.004 | 0.189 | higher values generally push away from Churn |
| 150 | secondary_longer | HasCancellation | 16 | 0.000 | 0.000 | value 1 generally pushes away from Churn |

The primary top five are: `Monetary, Frequency, RecencyDays, TotalQuantity, AverageOrderValue`. Stability is judged from rank correlation, top-five overlap, rank positions of the four prespecified substantive features, and direction patterns rather than from a forced claim of identical rankings.

## I. Sensitivity conclusion

The main conclusions are classified as **reasonably stable with some movement**: value, frequency, recency, and purchase-volume signals remain among the leading features across all three horizons, while their exact ordering and percentages move. The observed rank correlations (approximately 0.918–0.929) and top-five overlaps (4/5 or 5/5) support this qualified stability judgment. The alternative horizons are reported as sensitivity rather than used to replace the primary model.

## J. Direct answer to Research Question 2

**How can the technical predictive features and XAI outputs be translated into managerially interpretable categories?**

The translation groups the 16 business predictors into seven semantic categories: customer economic value; purchase activity and intensity; purchase recency and cadence; product breadth and basket behavior; relationship history; cancellation behavior; and geographic context. Category-level signed SHAP contributions preserve the direction and magnitude of the model signal while giving managers a compact vocabulary for monitoring. They remain interpretive summaries, not new predictive variables or causal mechanisms.

| managerial_category | constituent_features | category_mean_absolute_shap | relative_share_pct | direction_or_pattern | managerial_answer |
| --- | --- | --- | --- | --- | --- |
| Purchase activity and intensity | Frequency, TotalQuantity, AverageQuantityPerOrder | 0.619 | 41.138 | Predominantly away from Churn: Frequency and TotalQuantity show negative direction; AverageQuantityPerOrder is mixed/nonlinear. | How often and how much the customer purchases. |
| Customer economic value | Monetary, AverageOrderValue | 0.433 | 28.764 | Mixed: higher Monetary generally pushes away from Churn, while higher AverageOrderValue generally pushes toward Churn. | Observed customer value and order-value intensity. |
| Purchase recency and cadence | RecencyDays, MedianInterpurchaseDays | 0.258 | 17.169 | Mixed: higher RecencyDays pushes toward Churn, while MedianInterpurchaseDays is nonlinear/mixed. | How recently and regularly purchase events occur. |
| Product breadth and basket behavior | UniqueProducts, ProductsPerOrder | 0.070 | 4.680 | Predominantly away from Churn for UniqueProducts; ProductsPerOrder is mixed/nonlinear. | Breadth of products and typical basket composition. |
| Cancellation behavior | CancellationValue, CancellationInvoiceRate, CancellationInvoiceCount, HasCancellation | 0.066 | 4.370 | Mixed: higher CancellationValue pushes toward Churn, while CancellationInvoiceRate and CancellationInvoiceCount generally push away; HasCancellation is weak/near zero. | Observed cancellation frequency and value signals. |
| Relationship history | CustomerAgeDays, HasRepeatPurchaseHistory | 0.044 | 2.956 | Mixed/nonlinear: HasRepeatPurchaseHistory=1 generally pushes away from Churn; CustomerAgeDays is mixed. | Observable history length and repeat-purchase status in the window. |
| Geographic context | CountryCandidate | 0.014 | 0.923 | Country-specific/mixed and low importance; contextual rather than a primary trigger. | Recorded country context, used primarily as descriptive context because its SHAP importance is low. |

## K. Direct answer to Research Question 3

**How can prediction, explanation, managerial interpretation, and intervention options be linked?**

Prediction identifies who receives higher modeled risk; SHAP explains why the fitted model moved that prediction; categories translate the technical contributors; intervention-option families guide managerial attention; and future controlled measurement determines whether an action is effective.

| element | evidence | answer |
| --- | --- | --- |
| Prediction | Locked XGBoost probabilities and Step 10 ranking/lift evidence. | Identifies which customers receive higher modeled churn risk. |
| Explanation | Step 11 signed SHAP values on held-out Test customers. | Explains why the fitted model moved a customer toward or away from churn risk. |
| Managerial interpretation | Step 12 category mapping and category-level SHAP aggregation. | Translates technical contributors into seven business-meaningful categories. |
| Intervention options | Category-to-option mapping. | Guides managerial attention and hypothesis formation, not causal treatment selection. |
| Effectiveness | Not observed in Online Retail II. | Requires future measurement, controlled testing, or A/B experimentation. |

## L. Final three-research-question summary

| research_question | evidence_generated | main_answer | relevant_outputs |
| --- | --- | --- | --- |
| RQ1 | Step 11 corrected business-level SHAP ranking | Monetary, Frequency, RecencyDays, and TotalQuantity are the leading model contributors; SHAP is associative. | Step 11 tables 129 and 133; Step 11 report |
| RQ2 | Step 12 category mapping and category-level signed SHAP aggregation | Six behavioral/value categories plus geographic context translate technical predictors into a manager-readable monitoring structure. | Step 12 tables 137–139 and 146 |
| RQ3 | Step 12 integrated framework and intervention-option mapping | Prediction identifies risk, SHAP explains model drivers, categories translate them, and options guide testing; effectiveness requires future measurement. | Step 12 tables 140–147 and framework figure |

## M. Limitations

| limitation | relevance |
| --- | --- |
| Single public retailer | The data come from one UK online retailer, so external generalization is uncertain. |
| Operational churn definition | Non-contractual churn is constructed from inactivity rather than directly observed cancellation of a relationship. |
| Limited covariates | No clickstream, support, satisfaction, demographic, or competitor information is available. |
| Single temporal cutoff | Held-out evaluation is customer-level at one cutoff, not external or multi-period validation. |
| SHAP interpretation | SHAP explains fitted-model behavior and association, not causality. |
| Untested interventions | Intervention options are conceptual and require controlled measurement or experimentation. |
| Imperfect calibration | Predicted probabilities should support ranking and prioritization with calibration caution. |
| Alternative horizons | 120- and 150-day analyses are sensitivity checks, not replacements for the 135-day primary design. |

## N. Validation

| check | passed | observed | expected |
| --- | --- | --- | --- |
| primary_xgboost_configuration_unchanged | True | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} |
| primary_feature_set_unchanged | True | ["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "UniqueProducts", "ProductsPerOrder", "CustomerAgeDays", "MedianInterpurchaseDays", "HasRepeatPurchaseHistory", "CancellationInvoiceCount", "CancellationValue", "HasCancellation", "CancellationInvoiceRate", "CountryCandidate"] | ["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "UniqueProducts", "ProductsPerOrder", "CustomerAgeDays", "MedianInterpurchaseDays", "HasRepeatPurchaseHistory", "CancellationInvoiceCount", "CancellationValue", "HasCancellation", "CancellationInvoiceRate", "CountryCandidate"] |
| primary_threshold_unchanged | True | 0.350 | 0.350 |
| primary_train_test_sizes_unchanged | True | 1965/492 | 1965/492 |
| no_primary_resampling_or_smote | True | None | None |
| all_16_features_assigned_exactly_once | True | 16 | 16 |
| category_signed_aggregation_per_customer | True | 7 categories from per-customer signed sums | 7 |
| category_shares_sum_to_100 | True | 100.000 | 100.000 |
| no_causal_language_in_category_outputs | True | Interpretations use association/prediction guardrails | True |
| no_invented_data_fields | True | All categories/options use existing features and cautious options | True |
| sensitivity_contains_120_135_150 | True | ["120", "135", "150"] | [120, 135, 150] |
| sensitivity_train_test_split_rule_fixed | True | 80/20 stratified random_state=42 | True |
| sensitivity_fixed_hyperparameters | True | same locked XGBoost parameters | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} |
| no_alternative_threshold_tuning | True | ROC-AUC/PR-AUC only for sensitivity comparison | True |
| primary_step11_business_importance_reconciles | True | primary recomputation matches Step 11 ranking | True |
| rank_stability_calculated | True | 3 | 3 |
| primary_sensitivity_role_preserved | True | primary | primary |
| alternative_scenarios_not_used_to_change_primary | True | secondary robustness checks only | True |

All Step 12 validation checks passed. No primary model tuning, primary threshold change, new primary feature selection, resampling, SMOTE, or alternative-scenario replacement occurred. No causal intervention claim is made.

## Outputs

- Source: `src/churn_analysis/managerial_step12.py`
- Lightweight notebook: `notebooks/12_managerial_translation_integrated_framework.ipynb`
- Report: `reports/12_managerial_translation_integrated_framework.md`
- Tables: `outputs/tables/137_step12_feature_to_managerial_category.csv` through `outputs/tables/152_step12_output_manifest.csv`
- Figures: `21_step12_managerial_category_shap_importance.png, 22_step12_integrated_churn_intelligence_framework.png, 23_step12_sensitivity_comparison.png`
- Reproducibility log: `outputs/tables/151_step12_reproducibility_log.csv`

No commit or push was performed. Earlier Step 1–11 evidence was preserved.
