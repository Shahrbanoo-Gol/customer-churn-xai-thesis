# Step 11 — Explain the Selected XGBoost Churn Model Using SHAP

## Scope and locked method

Step 11 explains the already selected XGBoost model. It refits the exact Step 10 configuration on the complete **1,965-customer Train set**, applies the Train-fitted Step 9 tree preprocessing to the existing **492-customer Test set**, and computes SHAP values for all Test customers. The Parsimonious 16-feature set, unweighted estimator, and locked classification threshold **0.35** are unchanged.

The XGBoost configuration was `n_estimators=150`, `max_depth=2`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, and `reg_lambda=1.0`, with the Step 9 binary-logistic objective, `random_state=42`, and one processing thread. No Test result was used to change the model.

## A. SHAP method and output scale

`shap.TreeExplainer` was applied to the fitted XGBoost estimator after Train-only preprocessing. The SHAP output is on the model's **raw margin / log-odds scale**, not the probability scale. A positive SHAP value pushes the model toward a higher predicted churn risk; a negative SHAP value pushes it toward a lower predicted churn risk. These are explanations of model behavior and association, not causal effects.

The base/expected value was **-0.86082393** on the raw-margin scale. Additive reconstruction was checked as `expected value + sum(feature SHAP values)` against the XGBoost raw margin, with tolerance **1.0e-05** and maximum absolute error **2.617e-06**.

## B. Global transformed-feature importance

The ranking below uses mean absolute SHAP value across the 492 held-out Test customers. Magnitude measures average influence strength regardless of direction; it does not indicate whether a feature increases or decreases churn risk.

| rank | transformed_feature | mean_absolute_shap | relative_share_pct_of_transformed |
| --- | --- | --- | --- |
| 1 | Monetary | 0.466 | 26.024 |
| 2 | Frequency | 0.441 | 24.639 |
| 3 | RecencyDays | 0.252 | 14.066 |
| 4 | TotalQuantity | 0.213 | 11.878 |
| 5 | AverageOrderValue | 0.093 | 5.189 |
| 6 | CancellationValue | 0.070 | 3.882 |
| 7 | MedianInterpurchaseDays | 0.056 | 3.106 |
| 8 | UniqueProducts | 0.054 | 2.998 |
| 9 | CustomerAgeDays | 0.043 | 2.398 |
| 10 | AverageQuantityPerOrder | 0.043 | 2.374 |
| 11 | ProductsPerOrder | 0.030 | 1.696 |
| 12 | CountryCandidate_France | 0.009 | 0.496 |
| 13 | CancellationInvoiceRate | 0.007 | 0.411 |
| 14 | HasRepeatPurchaseHistory | 0.005 | 0.304 |
| 15 | CountryCandidate_United Kingdom | 0.005 | 0.284 |

The full transformed-feature table is saved as `outputs/tables/128_step11_transformed_feature_shap_importance.csv`. The global bar plot is `outputs/figures/15_step11_global_shap_importance.png` and the beeswarm is `outputs/figures/16_step11_shap_beeswarm.png`.

## C. Original/business-feature importance

For non-categorical predictors, business-level importance is the direct transformed-column mean absolute SHAP value. `CountryCandidate` expands to 30 one-hot columns; its business-level value is calculated per customer by summing the **signed** SHAP values across all 30 dummies, then taking the absolute value and averaging across Test customers. This preserves the additive grouped SHAP contribution and is neither a sum nor an average of global dummy magnitudes.

| rank | business_feature | mean_absolute_shap | relative_share_pct_of_business_aggregate | transformed_column_count | aggregation_method |
| --- | --- | --- | --- | --- | --- |
| 1 | Monetary | 0.466 | 26.034 | 1 | direct transformed feature value |
| 2 | Frequency | 0.441 | 24.649 | 1 | direct transformed feature value |
| 3 | RecencyDays | 0.252 | 14.071 | 1 | direct transformed feature value |
| 4 | TotalQuantity | 0.213 | 11.883 | 1 | direct transformed feature value |
| 5 | AverageOrderValue | 0.093 | 5.191 | 1 | direct transformed feature value |
| 6 | CancellationValue | 0.070 | 3.884 | 1 | direct transformed feature value |
| 7 | MedianInterpurchaseDays | 0.056 | 3.107 | 1 | direct transformed feature value |
| 8 | UniqueProducts | 0.054 | 2.999 | 1 | direct transformed feature value |
| 9 | CustomerAgeDays | 0.043 | 2.399 | 1 | direct transformed feature value |
| 10 | AverageQuantityPerOrder | 0.043 | 2.375 | 1 | direct transformed feature value |

The corrected top-four combined business-level share is **76.637843%**.

The relative shares here are shares of the transparent business-level aggregate, so they are useful for comparing the original features within that aggregation scheme. They are not causal effect sizes.

## D. Direction and selected dependence plots

The direction audit examines the actual transformed feature value against its SHAP value. Continuous features use ordered Test quantile bins and Spearman association; binary features compare the two observed indicator groups. Structural missing interval values use the already locked Train-fitted median imputation. The interpretation is intentionally descriptive: higher values may generally push the fitted model toward Churn, away from Churn, or show a nonlinear/mixed pattern.

| feature | feature_type | mean_absolute_shap | spearman_rho | low_bin_mean_shap | high_bin_mean_shap | pattern | missing_in_raw_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Monetary | continuous numeric | 0.466 | -0.939 | 0.557 | -0.762 | higher values generally push predictions away from Churn | 0 |
| Frequency | continuous numeric | 0.441 | -0.937 | 0.418 | -1.037 | higher values generally push predictions away from Churn | 0 |
| RecencyDays | continuous numeric | 0.252 | 0.962 | -0.373 | 0.426 | higher values generally push predictions toward Churn | 0 |
| TotalQuantity | continuous numeric | 0.213 | -0.974 | 0.243 | -0.434 | higher values generally push predictions away from Churn | 0 |
| AverageOrderValue | continuous numeric | 0.093 | 0.728 | -0.065 | 0.167 | higher values generally push predictions toward Churn | 0 |
| HasRepeatPurchaseHistory | binary | 0.005 |  | 0.012 | -0.004 | value 1 generally pushes predictions away from Churn | 0 |
| HasCancellation | binary | 0.000 |  | 0.000 | 0.000 | nonlinear/mixed | 0 |

Selected dependence figures were created only for the four most influential continuous numeric features in the business ranking:

- `17_step11_dependence_monetary.png`
- `18_step11_dependence_frequency.png`
- `19_step11_dependence_recencydays.png`
- `20_step11_dependence_totalquantity.png`

The beeswarm shows both SHAP magnitude and direction: points to the right push toward higher raw churn risk, points to the left push toward lower risk, and color indicates low-to-high transformed feature values. One-hot country columns are technical inputs; their business interpretation is the aggregated CountryCandidate result above.

## E. Local explanations

Cases were selected objectively from Test: highest predicted probability among correctly predicted Churn customers, lowest predicted probability among correctly predicted Non-Churn customers, highest-probability False Positive, and highest-probability False Negative. The audit table includes the internal Customer ID for traceability. The thesis-facing summary below omits that identifier.

| case_label | actual_Churn | predicted_probability | locked_threshold | predicted_class | selection_rule | strongest_features_increasing_risk | strongest_features_decreasing_risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True Positive / correctly predicted high-risk Churn | 1 | 0.854 | 0.350 | 1 | highest Test probability within case | RecencyDays (+0.6404); Frequency (+0.5163); ProductsPerOrder (+0.4192) | HasRepeatPurchaseHistory (-0.0095); HasCancellation (+0.0000); CountryCandidate_Australia (+0.0000) |
| True Negative / correctly predicted low-risk Non-Churn | 0 | 0.021 | 0.350 | 0 | lowest Test probability within True Negatives | MedianInterpurchaseDays (+0.1156); CancellationValue (+0.0412); CustomerAgeDays (+0.0153) | Frequency (-1.0833); Monetary (-0.8351); TotalQuantity (-0.6543) |
| False Positive | 0 | 0.784 | 0.350 | 1 | highest Test probability within case | Monetary (+0.8506); Frequency (+0.4275); RecencyDays (+0.4145) | CancellationValue (-0.0443); CancellationInvoiceRate (-0.0041); CountryCandidate_Australia (+0.0000) |
| False Negative | 1 | 0.34993505477905273 | 0.350 | 0 | highest Test probability within case | Monetary (+0.3693); TotalQuantity (+0.2071); UniqueProducts (+0.0665) | ProductsPerOrder (-0.1349); AverageOrderValue (-0.0754); RecencyDays (-0.0716) |

Positive local SHAP contributions pushed each selected prediction toward higher churn risk; negative contributions pushed it toward lower risk. They do not identify causes or prescribe interventions.

## F. Logistic Regression interpretive benchmark

Logistic Regression was refit only in its locked Step 9 configuration (`C=1.0`, Parsimonious set, Train-fitted scaled preprocessing) as a secondary interpretive benchmark. It was not used to replace SHAP, retune XGBoost, or select features.

| rank_by_xgb_business_mean_abs_shap | business_feature | xgb_direction_pattern | logistic_coefficient_or_country_mean_abs | logistic_sign | comparison |
| --- | --- | --- | --- | --- | --- |
| 1 | Monetary | higher values generally push predictions away from Churn | -0.349 | negative | directionally plausible |
| 2 | Frequency | higher values generally push predictions away from Churn | -1.270 | negative | directionally plausible |
| 3 | RecencyDays | higher values generally push predictions toward Churn | 0.330 | positive | directionally plausible |
| 4 | TotalQuantity | higher values generally push predictions away from Churn | -1.100 | negative | directionally plausible |
| 5 | AverageOrderValue | higher values generally push predictions toward Churn | -0.129 | negative | major directional disagreement |
| 6 | CancellationValue | not summarized | 0.717 | positive | supportive comparison only; XGBoost pattern is not a single monotonic direction |
| 7 | MedianInterpurchaseDays | not summarized | 0.186 | positive | supportive comparison only; XGBoost pattern is not a single monotonic direction |
| 8 | UniqueProducts | not summarized | 0.040 | positive | supportive comparison only; XGBoost pattern is not a single monotonic direction |
| 9 | CustomerAgeDays | not summarized | -0.139 | negative | supportive comparison only; XGBoost pattern is not a single monotonic direction |
| 10 | AverageQuantityPerOrder | not summarized | 0.035 | positive | supportive comparison only; XGBoost pattern is not a single monotonic direction |

The benchmark shows broad directional plausibility for the strongest monotonic numeric variables where signs agree; any nonlinear/mixed XGBoost pattern is not reduced to a single coefficient comparison. Agreement is supportive model comparison only and is not causal confirmation.

## G. Direct answer to Research Question 1

**Which behavioral and transactional features are most predictive of churn, and what is their relative importance according to SHAP?**

In this fitted model and held-out Test sample, the strongest original/business features are shown below. The ranking is based on average absolute SHAP magnitude; direction/pattern is reported separately so magnitude is not confused with effect direction.

| rank | feature | business_meaning | mean_absolute_shap | relative_share_pct_of_business_aggregate | typical_direction_or_pattern | short_interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Monetary | Total positive purchase value attributed to the customer | 0.466 | 26.034 | higher values generally push predictions away from Churn | higher values generally push predictions away from Churn; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect. |
| 2 | Frequency | Count of positive purchase invoices/events in the observation period | 0.441 | 24.649 | higher values generally push predictions away from Churn | higher values generally push predictions away from Churn; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect. |
| 3 | RecencyDays | Days since the customer's last positive purchase at the cutoff | 0.252 | 14.071 | higher values generally push predictions toward Churn | higher values generally push predictions toward Churn; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect. |
| 4 | TotalQuantity | Total quantity purchased in positive purchase events | 0.213 | 11.883 | higher values generally push predictions away from Churn | higher values generally push predictions away from Churn; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect. |
| 5 | AverageOrderValue | Average positive purchase invoice value | 0.093 | 5.191 | higher values generally push predictions toward Churn | higher values generally push predictions toward Churn; SHAP describes how this feature contributed to predictions in this fitted model, not a causal effect. |
| 6 | CancellationValue | Absolute value of cancellation quantities multiplied by price | 0.070 | 3.884 | not summarized | Included in the ranked evidence; no simple direction was forced. |
| 7 | MedianInterpurchaseDays | Median gap between positive purchase events | 0.056 | 3.107 | not summarized | Included in the ranked evidence; no simple direction was forced. |
| 8 | UniqueProducts | Number of distinct products purchased | 0.054 | 2.999 | not summarized | Included in the ranked evidence; no simple direction was forced. |
| 9 | CustomerAgeDays | Days from the customer's first positive purchase to the cutoff | 0.043 | 2.399 | not summarized | Included in the ranked evidence; no simple direction was forced. |
| 10 | AverageQuantityPerOrder | Average quantity purchased per positive invoice | 0.043 | 2.375 | not summarized | Included in the ranked evidence; no simple direction was forced. |

Thus, the answer is model-specific and associative: these features contributed most strongly to the XGBoost churn predictions on average in the locked held-out evaluation. “Most predictive” here means strongest contribution to this model's predictions, not proof that changing a feature would change churn.

## H. Validation and stopping point

| check | passed | observed | expected |
| --- | --- | --- | --- |
| exact_locked_xgboost_configuration | True | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} |
| exact_locked_parsimonious_feature_set | True | ["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "UniqueProducts", "ProductsPerOrder", "CustomerAgeDays", "MedianInterpurchaseDays", "HasRepeatPurchaseHistory", "CancellationInvoiceCount", "CancellationValue", "HasCancellation", "CancellationInvoiceRate", "CountryCandidate"] | ["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "UniqueProducts", "ProductsPerOrder", "CustomerAgeDays", "MedianInterpurchaseDays", "HasRepeatPurchaseHistory", "CancellationInvoiceCount", "CancellationValue", "HasCancellation", "CancellationInvoiceRate", "CountryCandidate"] |
| train_customer_count | True | 1965 | 1965 |
| test_customer_count | True | 492 | 492 |
| preprocessing_fitted_on_train_only | True | fit(train[features]); transform(test[features]) | True |
| one_shap_explanation_per_test_customer | True | 492 | 492 |
| shap_feature_columns_align_with_transformed_inputs | True | 45 SHAP columns aligned to 45 transformed inputs | 45 aligned inputs |
| customer_id_absent_from_model_features | True | False | False |
| churn_absent_from_model_features | True | False | False |
| future_information_absent | True | [] | [] |
| no_anonymous_feature_names | True | ["RecencyDays", "Frequency", "Monetary", "TotalQuantity", "AverageOrderValue", "AverageQuantityPerOrder", "UniqueProducts", "ProductsPerOrder", "CustomerAgeDays", "MedianInterpurchaseDays", "CancellationInvoiceCount", "CancellationValue", "CancellationInvoiceRate", "HasCancellation", "HasRepeatPurchaseHistory", "CountryCandidate_Australia", "CountryCandidate_Austria", "CountryCandidate_Bahrain", "CountryCandidate_Belgium", "CountryCandidate_Brazil", "CountryCandidate_Canada", "CountryCandidate_Channel Islands", "CountryCandidate_Cyprus", "CountryCandidate_Denmark", "CountryCandidate_EIRE", "CountryCandidate_European Community", "CountryCandidate_Finland", "CountryCandidate_France", "CountryCandidate_Germany", "CountryCandidate_Greece", "CountryCandidate_Iceland", "CountryCandidate_Israel", "CountryCandidate_Italy", "CountryCandidate_Japan", "CountryCandidate_Malta", "CountryCandidate_Netherlands", "CountryCandidate_Norway", "CountryCandidate_Poland", "CountryCandidate_Portugal", "CountryCandidate_Singapore", "CountryCandidate_Spain", "CountryCandidate_Sweden", "CountryCandidate_Switzerland", "CountryCandidate_United Kingdom", "CountryCandidate_Unspecified"] | no x0-style names |
| shap_values_finite | True | all finite | True |
| shap_additive_reconstruction_within_tolerance | True | 0.000 | 0.000 |
| country_one_hot_aggregated_to_business_feature | True | CountryCandidate present | True |
| country_grouped_shap_uses_exactly_30_country_columns | True | 30 | 30 |
| country_grouped_shap_excludes_non_country_columns | True | only CountryCandidate columns | True |
| country_aggregation_is_per_customer_signed_sum_then_abs_mean | True | mean(abs(per-customer sum of signed SHAP across all 30 CountryCandidate one-hot columns)) | signed per-customer sum then abs/mean |
| business_feature_names_unique | True | 16 | 16 |
| business_relative_shares_sum_to_100 | True | 100.000 | 100.000 |
| selected_local_cases_present | True | 4 | 4 |
| locked_threshold_unchanged | True | 0.350 | 0.350 |
| no_test_based_model_change | True | Test used for explanation and audit only | True |
| no_new_feature_selection | True | Step 9 Parsimonious set reused unchanged | True |
| no_model_tuning | True | No search or parameter update | True |
| no_resampling | True | No SMOTE/oversampling/undersampling | True |
| no_managerial_intervention_design | True | Technical explanation only | True |

All Step 11 validation checks passed. The locked Train/Test split, Step 10 model configuration, feature set, threshold, and preprocessing were preserved. No new feature selection, model tuning, threshold change, Test-based model change, resampling, or managerial intervention design occurred. Step 11 stops at technical explanation and understandable feature interpretation.

## Outputs

- Source: `src/churn_analysis/explain_step11_shap.py`
- Lightweight notebook: `notebooks/11_explain_selected_xgboost_with_shap.ipynb`
- Report: `reports/11_explain_selected_xgboost_with_shap.md`
- Tables: `outputs/tables/128_step11_transformed_feature_shap_importance.csv` through `outputs/tables/136_step11_output_manifest.csv`
- Figures: `15_step11_global_shap_importance.png, 16_step11_shap_beeswarm.png, 17_step11_dependence_monetary.png, 18_step11_dependence_frequency.png, 19_step11_dependence_recencydays.png, 20_step11_dependence_totalquantity.png`
- Reproducibility log: `outputs/tables/135_step11_reproducibility_log.csv`

No commit or push was performed. Step 10 outputs were not overwritten.
