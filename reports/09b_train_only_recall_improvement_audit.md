# Step 9B - Train-Only Recall Improvement Audit

## Scope

This is a controlled extension of Step 9. It preserves the original Step 9 baseline and tuned outputs, uses the same locked parsimonious configurations, and operates only on the 1,965 Train customers. The 492-customer Test set is not passed to a preprocessor, estimator, scorer, threshold rule, or comparison.

All comparisons use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, with `Churn=1` as the positive class. No resampling, SMOTE, synthetic data, SHAP, feature expansion, or Churn-definition change was used.

## A. Class-weight comparison

The XGBoost weight is calculated only from Train: `scale_pos_weight = 1,379 / 586 = 2.353242`. A weighted option is considered useful only when Recall improves by at least 0.05 while Precision falls by no more than 0.10 and PR-AUC falls by no more than 0.03.

| model_family | weighting_variant | weighting_description | roc_auc | pr_auc | recall | precision | f1 | accuracy | weighted_useful_by_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unweighted | None | 0.785 | 0.600 | 0.423 | 0.594 | 0.492 | 0.741 | False |
| Logistic Regression | weighted | class_weight="balanced" | 0.784 | 0.602 | 0.751 | 0.494 | 0.594 | 0.694 | False |
| Decision Tree | unweighted | None | 0.766 | 0.524 | 0.427 | 0.595 | 0.493 | 0.741 | False |
| Decision Tree | weighted | class_weight="balanced" | 0.756 | 0.508 | 0.765 | 0.473 | 0.583 | 0.672 | False |
| Random Forest | unweighted | None | 0.781 | 0.576 | 0.379 | 0.602 | 0.460 | 0.739 | False |
| Random Forest | weighted | class_weight="balanced" | 0.779 | 0.569 | 0.732 | 0.491 | 0.586 | 0.692 | False |
| XGBoost | unweighted | None | 0.789 | 0.595 | 0.428 | 0.589 | 0.493 | 0.741 | False |
| XGBoost | weighted | scale_pos_weight=2.353242 (Train only) | 0.787 | 0.595 | 0.749 | 0.488 | 0.590 | 0.689 | False |

Recall alone was not used to select weighting. The weighted alternatives were judged against the declared Precision and PR-AUC safeguards.

## B-C. Out-of-fold threshold sensitivity and Recall/Precision trade-offs

Thresholds were evaluated from out-of-fold Train probabilities, never in-sample fitted probabilities. Lowering the threshold generally increases Recall and can reduce Precision. F2 is supporting evidence only; the candidate rule also requires improved Recall and a practical Precision floor.

The complete threshold table is saved in `109_step9b_threshold_sensitivity.csv`:

| model_family | weighting_variant | threshold | recall | precision | f1 | f2 | accuracy | predicted_churn_rate | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unweighted | 0.250 | 0.819 | 0.458 | 0.587 | 0.707 | 0.656 | 0.534 | 0.592 |
| Logistic Regression | unweighted | 0.300 | 0.754 | 0.494 | 0.597 | 0.682 | 0.696 | 0.455 | 0.592 |
| Logistic Regression | unweighted | 0.350 | 0.667 | 0.523 | 0.587 | 0.632 | 0.720 | 0.380 | 0.592 |
| Logistic Regression | unweighted | 0.400 | 0.596 | 0.554 | 0.574 | 0.587 | 0.736 | 0.321 | 0.592 |
| Logistic Regression | unweighted | 0.450 | 0.510 | 0.579 | 0.543 | 0.523 | 0.744 | 0.263 | 0.592 |
| Logistic Regression | unweighted | 0.500 | 0.423 | 0.593 | 0.494 | 0.449 | 0.741 | 0.213 | 0.592 |
| Decision Tree | unweighted | 0.250 | 0.800 | 0.466 | 0.589 | 0.700 | 0.667 | 0.512 | 0.545 |
| Decision Tree | unweighted | 0.300 | 0.775 | 0.477 | 0.590 | 0.689 | 0.679 | 0.484 | 0.545 |
| Decision Tree | unweighted | 0.350 | 0.708 | 0.478 | 0.570 | 0.646 | 0.682 | 0.442 | 0.545 |
| Decision Tree | unweighted | 0.400 | 0.505 | 0.543 | 0.523 | 0.512 | 0.726 | 0.277 | 0.545 |
| Decision Tree | unweighted | 0.450 | 0.502 | 0.552 | 0.525 | 0.511 | 0.730 | 0.271 | 0.545 |
| Decision Tree | unweighted | 0.500 | 0.427 | 0.592 | 0.496 | 0.452 | 0.741 | 0.215 | 0.545 |
| Random Forest | unweighted | 0.250 | 0.836 | 0.458 | 0.591 | 0.717 | 0.655 | 0.545 | 0.564 |
| Random Forest | unweighted | 0.300 | 0.765 | 0.479 | 0.589 | 0.683 | 0.682 | 0.476 | 0.564 |
| Random Forest | unweighted | 0.350 | 0.688 | 0.511 | 0.586 | 0.643 | 0.710 | 0.402 | 0.564 |
| Random Forest | unweighted | 0.400 | 0.604 | 0.532 | 0.565 | 0.588 | 0.723 | 0.339 | 0.564 |
| Random Forest | unweighted | 0.450 | 0.488 | 0.555 | 0.520 | 0.500 | 0.731 | 0.262 | 0.564 |
| Random Forest | unweighted | 0.500 | 0.379 | 0.600 | 0.464 | 0.409 | 0.739 | 0.188 | 0.564 |
| XGBoost | unweighted | 0.250 | 0.833 | 0.464 | 0.596 | 0.718 | 0.663 | 0.535 | 0.586 |
| XGBoost | unweighted | 0.300 | 0.765 | 0.490 | 0.597 | 0.687 | 0.692 | 0.466 | 0.586 |
| XGBoost | unweighted | 0.350 | 0.695 | 0.523 | 0.597 | 0.652 | 0.720 | 0.396 | 0.586 |
| XGBoost | unweighted | 0.400 | 0.613 | 0.544 | 0.576 | 0.598 | 0.731 | 0.336 | 0.586 |
| XGBoost | unweighted | 0.450 | 0.524 | 0.565 | 0.544 | 0.532 | 0.738 | 0.276 | 0.586 |
| XGBoost | unweighted | 0.500 | 0.428 | 0.591 | 0.497 | 0.453 | 0.741 | 0.216 | 0.586 |

The selected higher-Recall candidates are:

| model_family | weighting_variant | default_threshold | candidate_threshold | recall_change | precision_change | f1_change | f2_change | candidate_pr_auc | candidate_recall_meaningful | candidate_precision_acceptable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unweighted | 0.500 | 0.350 | 0.244 | -0.070 | 0.093 | 0.184 | 0.592 | True | True |
| Decision Tree | unweighted | 0.500 | 0.400 | 0.078 | -0.049 | 0.027 | 0.060 | 0.545 | True | True |
| Random Forest | unweighted | 0.500 | 0.350 | 0.309 | -0.089 | 0.122 | 0.234 | 0.564 | True | True |
| XGBoost | unweighted | 0.500 | 0.350 | 0.266 | -0.067 | 0.100 | 0.199 | 0.586 | True | True |

## D. Logistic log-transformation sensitivity

Only Logistic Regression was tested. The log-sensitive path applies Train-fold median imputation, `log1p`, and StandardScaler to the predefined non-negative skewed/count features: `Frequency, Monetary, TotalQuantity, AverageOrderValue, AverageQuantityPerOrder, UniqueProducts, ProductsPerOrder, CancellationInvoiceCount, CancellationValue`. Binary indicators and country encoding were not log-transformed.

| preprocessing_variant | roc_auc | pr_auc | recall | f1 | f2 | roc_auc_delta_vs_current | pr_auc_delta_vs_current | recall_delta_vs_current | f1_delta_vs_current | f2_delta_vs_current |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 0.785 | 0.600 | 0.423 | 0.492 | 0.448 |  |  |  |  |  |
| log1p_sensitive | 0.792 | 0.597 | 0.439 | 0.503 | 0.462 | 0.007 | -0.003 | 0.015 | 0.011 | 0.014 |

The simpler current preprocessing is retained when the log-sensitive gains are negligible by the documented recommendation rule.

## E-G. Improvement comparison and recommendation before Test

| model_family | best_weighting_option | recommended_threshold | original_step9_recall | recommended_recall | recall_change_vs_original | original_step9_precision | recommended_precision | precision_change_vs_original | original_step9_pr_auc | recommended_pr_auc | improvement_assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | unweighted | 0.350 | 0.423 | 0.667 | 0.244 | 0.594 | 0.523 | -0.070 | 0.600 | 0.592 | meaningful |
| Decision Tree | unweighted | 0.400 | 0.427 | 0.505 | 0.078 | 0.595 | 0.543 | -0.052 | 0.524 | 0.545 | meaningful |
| Random Forest | unweighted | 0.350 | 0.379 | 0.688 | 0.309 | 0.602 | 0.511 | -0.091 | 0.576 | 0.564 | meaningful |
| XGBoost | unweighted | 0.350 | 0.428 | 0.695 | 0.266 | 0.589 | 0.523 | -0.066 | 0.595 | 0.586 | meaningful |

Recommended locked configurations for Step 10:

| model_family | feature_set | hyperparameters | class_weighting_choice | preprocessing | classification_threshold | log_transformation_decision |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Parsimonious | {"C": 1.0} | unweighted | Current Step 9 StandardScaler path | 0.350 | Retain current preprocessing; log1p change is modest and PR-AUC declines |
| Decision Tree | Parsimonious | {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 2} | unweighted | Step 9 tree preprocessing: Train-fold median/most-frequent imputation, one-hot encoding, no scaling | 0.400 | Not applicable |
| Random Forest | Parsimonious | {"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 400} | unweighted | Step 9 tree preprocessing: Train-fold median/most-frequent imputation, one-hot encoding, no scaling | 0.350 | Not applicable |
| XGBoost | Parsimonious | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} | unweighted | Step 9 tree preprocessing: Train-fold median/most-frequent imputation, one-hot encoding, no scaling | 0.350 | Not applicable |

These are Train-only recommendations, not a final model winner. All four model families should proceed to the final held-out Test evaluation in Step 10.

## H and J. Validation

| check | passed | observed | expected |
| --- | --- | --- | --- |
| train_customers_preserved | True | 1965 | 1965 |
| held_out_test_customers_preserved_without_scoring | True | 492 | 492 |
| train_test_split_unchanged | True | disjoint and complete | disjoint and complete |
| positive_class_is_churn_1 | True | Churn=1 | Churn=1 |
| same_stratified_5_fold_cv | True | StratifiedKFold(5, shuffle=True, random_state=42) | True |
| weighting_values_train_only | True | [586] | 586 |
| xgboost_scale_pos_weight_train_only | True | 1379/586 | 1379/586 |
| thresholds_from_oof_train_probabilities | True | ['out-of-fold Train probabilities'] | ['out-of-fold Train probabilities'] |
| log_transform_inside_cv_pipeline | True | FunctionTransformer inside fold-fitted Pipeline | True |
| no_customer_id_predictor | True | False | False |
| no_future_outcome_predictor | True | [] | [] |
| no_resampling | True | No resampling code path | True |
| no_smote | True | No SMOTE import/use | True |
| no_shap | True | No SHAP import/use | True |
| no_test_predictions_or_metrics | True | Test not passed to estimator or scorer | True |
| original_step9_outputs_preserved | True | Step 9 baseline/search/comparison tables exist | True |
| all_four_models_audited | True | ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'] | ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'] |

The original Step 9 evidence remains present and was not overwritten. No Test predictions or metrics were produced.

## Outputs

- Source: `src/churn_analysis/audit_step9b_recall_improvement.py`
- Lightweight notebook: `notebooks/09b_train_only_recall_improvement_audit.ipynb`
- Report: `reports/09b_train_only_recall_improvement_audit.md`
- Tables: `outputs/tables/108_step9b_class_weight_comparison.csv` through `116_step9b_output_manifest.csv`
- Figure: `outputs/figures/10_step9b_recall_precision_tradeoff.png`
- Log: `outputs/logs/09b_train_only_recall_improvement_audit.log`

No commit or push was performed. Step 9B stops before Test evaluation.
