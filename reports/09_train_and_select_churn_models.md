# Step 9 - Train and Select the Four Churn Models Using Train Data Only

## Scope and stopping point

Step 9 trains exactly Logistic Regression, Decision Tree, Random Forest, and XGBoost. It uses only the locked Step 8 Train rows for preprocessing fitting, stratified cross-validation, feature-set comparison, hyperparameter search, model selection, and overfitting checks. The 492-customer Test set is not passed to a model or scorer. This step stops before final Test evaluation, threshold optimization, resampling, and SHAP.

## A. Cross-validation design

The exact Step 8 customer-level split was preserved: **1,965 Train customers** (586 Churn=1 and 1,379 Churn=0) and **492 held-out Test customers**. Within Train, every evaluation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Stratification keeps each fold's Churn/Non-Churn proportions approximately aligned with Train. Preprocessing is inside each model pipeline, so imputers, one-hot encoders, and the Logistic Regression scaler are fitted inside each fold's fitting portion.

The positive class is `Churn = 1`. ROC-AUC and PR-AUC are the primary ranking signals through the transparent selection composite `0.5 * ROC-AUC + 0.5 * PR-AUC`; Recall, F1, Precision, and Accuracy are also reported. Accuracy is contextual only. All threshold-dependent metrics use the default probability threshold of 0.5.

## B. Baseline Train-CV performance

Baseline rows below use the Full feature set. The accompanying table also records the same baseline design on the Parsimonious set for the feature sensitivity audit.

| model_family | feature_count | roc_auc | pr_auc | recall | precision | f1 | accuracy | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 19 | 0.789 | 0.602 | 0.433 | 0.591 | 0.498 | 0.740 | 0.695 |
| Decision Tree | 19 | 0.743 | 0.485 | 0.398 | 0.548 | 0.453 | 0.720 | 0.614 |
| Random Forest | 19 | 0.751 | 0.519 | 0.396 | 0.548 | 0.457 | 0.723 | 0.635 |
| XGBoost | 19 | 0.764 | 0.552 | 0.442 | 0.546 | 0.487 | 0.724 | 0.658 |

## C. Full versus Parsimonious feature sets

The Parsimonious set removes `MeanInterpurchaseDays`, `ActivePurchaseDays`, and `PurchaseSpanDays`, while retaining `MedianInterpurchaseDays`, `Frequency`, and `CustomerAgeDays`. `Monetary` and `TotalQuantity` remain. Each feature-set result below is the best compact Train-CV search configuration for that model family and feature set.

| model_family | feature_set | feature_count | selected_feature_set | selection_score | roc_auc | pr_auc | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Full | 19 | False | 0.695 | 0.789 | 0.602 | 0.433 | 0.498 |
| Logistic Regression | Parsimonious | 16 | True | 0.692 | 0.785 | 0.600 | 0.423 | 0.492 |
| Decision Tree | Full | 19 | False | 0.640 | 0.762 | 0.518 | 0.394 | 0.459 |
| Decision Tree | Parsimonious | 16 | True | 0.645 | 0.766 | 0.524 | 0.427 | 0.493 |
| Random Forest | Full | 19 | False | 0.676 | 0.780 | 0.573 | 0.403 | 0.473 |
| Random Forest | Parsimonious | 16 | True | 0.678 | 0.781 | 0.576 | 0.379 | 0.460 |
| XGBoost | Full | 19 | False | 0.691 | 0.788 | 0.595 | 0.428 | 0.491 |
| XGBoost | Parsimonious | 16 | True | 0.692 | 0.789 | 0.595 | 0.428 | 0.493 |

The decision rule was fixed before reading Test data: choose Parsimonious when its composite is within 0.01 of Full, because it reduces redundancy; otherwise choose the higher composite. This is a Train-only rule.

## D-E. Selected feature sets and hyperparameters

| model_family | selected_feature_set | feature_count | selected_hyperparameters | selection_score |
| --- | --- | --- | --- | --- |
| Logistic Regression | Parsimonious | 16 | {"C": 1.0} | 0.692 |
| Decision Tree | Parsimonious | 16 | {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 2} | 0.645 |
| Random Forest | Parsimonious | 16 | {"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 400} | 0.678 |
| XGBoost | Parsimonious | 16 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} | 0.692 |

## F and H. Selected model Train-CV comparison

| model_family | selected_feature_set | roc_auc | pr_auc | recall | precision | f1 | accuracy | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Parsimonious | 0.785 | 0.600 | 0.423 | 0.594 | 0.492 | 0.741 | 0.692 |
| Decision Tree | Parsimonious | 0.766 | 0.524 | 0.427 | 0.595 | 0.493 | 0.741 | 0.645 |
| Random Forest | Parsimonious | 0.781 | 0.576 | 0.379 | 0.602 | 0.460 | 0.739 | 0.678 |
| XGBoost | Parsimonious | 0.789 | 0.595 | 0.428 | 0.589 | 0.493 | 0.741 | 0.692 |

The strongest Train-CV candidate is the model with the highest composite, while the component metrics show the trade-offs. Recall and F1 should be read alongside ROC-AUC and PR-AUC because the Churn class is about 30% of Train. This is not a final thesis winner; all four technically valid configurations proceed to Step 10.

## G. Overfitting audit

The audit compares mean in-fold fitting scores with mean held-in CV validation scores for the selected Train-only configuration. A clear flag is raised for ROC-AUC or PR-AUC gaps of at least 0.10.

| model_family | metric | train_fold_fit_mean | cv_validation_mean | fit_minus_cv_gap | clear_overfit_flag |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | ROC-AUC | 0.801 | 0.785 | 0.016 | False |
| Logistic Regression | PR-AUC | 0.621 | 0.600 | 0.020 | False |
| Decision Tree | ROC-AUC | 0.795 | 0.766 | 0.030 | False |
| Decision Tree | PR-AUC | 0.570 | 0.524 | 0.046 | False |
| Random Forest | ROC-AUC | 0.919 | 0.781 | 0.138 | True |
| Random Forest | PR-AUC | 0.835 | 0.576 | 0.259 | True |
| XGBoost | ROC-AUC | 0.848 | 0.789 | 0.059 | False |
| XGBoost | PR-AUC | 0.702 | 0.595 | 0.107 | True |

Decision Tree, Random Forest, and XGBoost receive particular attention because flexible learners can fit Train folds more strongly than they generalize to held-in CV folds. No Test score was used in this assessment.

## I. Configurations recommended for Step 10

Carry these four selected configurations to final held-out Test evaluation in Step 10:

| model_family | selected_feature_set | selected_hyperparameters |
| --- | --- | --- |
| Logistic Regression | Parsimonious | {"C": 1.0} |
| Decision Tree | Parsimonious | {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 2} |
| Random Forest | Parsimonious | {"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 3, "n_estimators": 400} |
| XGBoost | Parsimonious | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "reg_lambda": 1.0, "subsample": 0.8} |

All four model families are technically valid and should proceed. Step 10 may evaluate them once on the untouched 492-customer Test set; this Step 9 report does not declare the final thesis winner.

## Reproducibility

The compact search spaces, fixed random state, metric definitions, and package versions are recorded in `105_step9_reproducibility_log.csv`. Search results for every candidate are in `100_step9_hyperparameter_search_summary.csv`; no class weighting, SMOTE, oversampling, undersampling, or other resampling was used.

## Validation

| check | passed | observed | expected |
| --- | --- | --- | --- |
| exact_train_customers_preserved | True | 1965 | 1965 |
| exact_held_out_test_customers_preserved | True | 492 | 492 |
| train_churn_1_count | True | 586 | 586 |
| test_churn_1_count_not_used_for_selection | True | 147 | 147 |
| train_test_customer_ids_disjoint | True | 0 | 0 |
| churn_1_positive_class | True | Churn=1 | Churn=1 |
| customer_id_absent_from_predictors | True | False | False |
| target_absent_from_predictors | True | False | False |
| future_outcome_fields_absent | True | [] | [] |
| logistic_has_train_only_scaling_path | True | StandardScaler in fold pipeline | True |
| tree_paths_have_no_scaling | True | No StandardScaler for tree families | True |
| stratified_5_fold_cv_random_state_42 | True | StratifiedKFold(5, shuffle=True, random_state=42) | True |
| threshold_dependent_metrics_use_0_5 | True | Estimator default predict threshold | 0.5 |
| no_class_weighting_or_resampling | True | All class_weight=None; no resampling | True |
| no_smote | True | No SMOTE import/use | True |
| no_shap | True | No SHAP import/use | True |
| no_test_performance_used | True | Test rows not passed to estimator/scorer | True |
| no_test_hyperparameter_tuning | True | Search fit on Train rows only | True |
| no_test_feature_selection | True | Feature comparison fit on Train rows only | True |
| no_test_threshold_selection | True | Default threshold retained | True |
| four_model_families_only | True | ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'] | ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'] |
| all_four_have_overfitting_audit | True | ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'] | ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'] |

No `Customer ID`, future/outcome field, or target column entered predictors. No Test-based hyperparameter, feature-set, threshold, class-weight, preprocessing, or performance decision occurred. No SHAP or final Test ROC/PR curves were produced.

## Files created

- Source: `src/churn_analysis/train_step9_models.py`
- Lightweight notebook: `notebooks/09_train_and_select_churn_models.ipynb`
- Report: `reports/09_train_and_select_churn_models.md`
- Tables: `outputs/tables/99_step9_baseline_model_cv.csv` through `107_step9_output_manifest.csv`
- Figure: `outputs/figures/09_step9_train_cv_model_metrics.png`
- Log: `outputs/logs/09_train_and_select_churn_models.log`

No commit or push was performed. Step 9 stops here; final Test evaluation remains for Step 10.
