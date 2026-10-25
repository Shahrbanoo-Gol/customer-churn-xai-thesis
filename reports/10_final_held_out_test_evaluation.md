# Step 10 - Final Held-Out Test Evaluation of the Four Churn Models

## Scope and locked design

This is the first final evaluation on the untouched **492-customer Test set**. Exactly 1,965 Train customers were used to fit each complete pipeline once. The locked Step 9B Parsimonious feature set, hyperparameters, unweighted configurations, Step 9 preprocessing, and thresholds were used unchanged. No Test result was used to retune, change features, change thresholds, change hyperparameters, or modify a model.

The locked thresholds are: Logistic Regression **0.35**, Decision Tree **0.40**, Random Forest **0.35**, and XGBoost **0.35**. `Churn=1` is the positive class. No resampling, SMOTE, oversampling, undersampling, SHAP, or Accuracy-only conclusion was used.

## A. Test ranking performance

| model_family | Test ROC-AUC | Test PR-AUC / Average Precision | Brier score |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.758 | 0.527 | 0.176 |
| Decision Tree | 0.747 | 0.487 | 0.179 |
| Random Forest | 0.773 | 0.559 | 0.170 |
| XGBoost | 0.777 | 0.578 | 0.169 |

ROC-AUC and PR-AUC assess ranking discrimination without selecting a classification threshold.

## B-C. Test classification performance

At the default 0.50 threshold:

| model_family | threshold | recall | precision | f1 | f2 | accuracy | specificity | predicted_churn_count | predicted_churn_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.500 | 0.449 | 0.569 | 0.502 | 0.469 | 0.734 | 0.855 | 116 | 23.577 |
| Decision Tree | 0.500 | 0.476 | 0.515 | 0.495 | 0.483 | 0.709 | 0.809 | 136 | 27.642 |
| Random Forest | 0.500 | 0.415 | 0.587 | 0.486 | 0.441 | 0.738 | 0.875 | 104 | 21.138 |
| XGBoost | 0.500 | 0.456 | 0.588 | 0.513 | 0.477 | 0.742 | 0.864 | 114 | 23.171 |

At the locked Step 9B thresholds:

| model_family | threshold | recall | precision | f1 | f2 | accuracy | specificity | predicted_churn_count | predicted_churn_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.350 | 0.701 | 0.484 | 0.572 | 0.643 | 0.687 | 0.681 | 213 | 43.293 |
| Decision Tree | 0.400 | 0.497 | 0.503 | 0.500 | 0.498 | 0.703 | 0.791 | 145 | 29.472 |
| Random Forest | 0.350 | 0.721 | 0.484 | 0.579 | 0.657 | 0.687 | 0.672 | 219 | 44.512 |
| XGBoost | 0.350 | 0.714 | 0.486 | 0.579 | 0.653 | 0.689 | 0.678 | 216 | 43.902 |

## D. Confusion-matrix interpretation

There are **147 actual Test Churn customers**. At the locked thresholds:

| model_family | threshold | true_positive | false_negative | false_positive | true_negative | actual_test_churn_identified | actual_test_churn_missed | unnecessary_retention_alerts_false_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.350 | 103 | 44 | 110 | 235 | 103 | 44 | 110 |
| Decision Tree | 0.400 | 73 | 74 | 72 | 273 | 73 | 74 | 72 |
| Random Forest | 0.350 | 106 | 41 | 113 | 232 | 106 | 41 | 113 |
| XGBoost | 0.350 | 105 | 42 | 111 | 234 | 105 | 42 | 111 |

True Positives are correctly identified Churn customers; False Negatives are missed Churn customers; False Positives are Non-Churn customers incorrectly flagged and represent potential intervention cost; True Negatives are correctly cleared Non-Churn customers.

## E. Train-CV versus Test generalization

| model_family | train_cv_roc_auc | test_roc_auc | roc_auc_difference_test_minus_train_cv | train_cv_pr_auc | test_pr_auc | pr_auc_difference_test_minus_train_cv | generalization_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.785 | 0.758 | -0.027 | 0.600 | 0.527 | -0.073 | More noticeable PR-AUC degradation; ROC-AUC degradation remains modest. |
| Decision Tree | 0.766 | 0.747 | -0.019 | 0.524 | 0.487 | -0.037 | Modest degradation, with a larger PR-AUC drop than ROC-AUC drop. |
| Random Forest | 0.781 | 0.773 | -0.008 | 0.576 | 0.559 | -0.017 | Close ranking generalization; both ROC-AUC and PR-AUC drops are small. |
| XGBoost | 0.789 | 0.777 | -0.012 | 0.595 | 0.578 | -0.017 | Close ranking generalization; both ROC-AUC and PR-AUC drops are small. |

The Train-CV/Test differences are interpreted by magnitude and jointly across ROC-AUC and PR-AUC. The Random Forest and XGBoost results are read alongside the Step 9 overfitting audit rather than assumed to generalize because of model complexity.

## F. ROC and Precision-Recall curves

The ROC comparison is saved as `outputs/figures/11_step10_test_roc_curves.png`. The Precision-Recall comparison is saved as `outputs/figures/12_step10_test_precision_recall_curves.png` and includes the Test Churn prevalence reference. These curves are descriptive Test evaluation only and were not used to retune models.

## G. High-risk ranking, capture, and lift

Top-risk groups use descending Test predicted probability; group sizes use the ceiling of 10%, 20%, and 30% of 492. Capture Rate is the percentage of all actual Churn customers contained in the group. Lift is group Churn rate divided by the overall Test Churn rate.

| model_family | risk_group | selected_customer_count | group_churn_count | group_churn_rate | churn_capture_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Top 10% | 50 | 30 | 0.600 | 0.204 | 2.008 |
| Logistic Regression | Top 20% | 99 | 61 | 0.616 | 0.415 | 2.062 |
| Logistic Regression | Top 30% | 148 | 79 | 0.534 | 0.537 | 1.787 |
| Decision Tree | Top 10% | 50 | 30 | 0.600 | 0.204 | 2.008 |
| Decision Tree | Top 20% | 99 | 55 | 0.556 | 0.374 | 1.859 |
| Decision Tree | Top 30% | 148 | 75 | 0.507 | 0.510 | 1.696 |
| Random Forest | Top 10% | 50 | 34 | 0.680 | 0.231 | 2.276 |
| Random Forest | Top 20% | 99 | 58 | 0.586 | 0.395 | 1.961 |
| Random Forest | Top 30% | 148 | 78 | 0.527 | 0.531 | 1.764 |
| XGBoost | Top 10% | 50 | 34 | 0.680 | 0.231 | 2.276 |
| XGBoost | Top 20% | 99 | 59 | 0.596 | 0.401 | 1.995 |
| XGBoost | Top 30% | 148 | 79 | 0.534 | 0.537 | 1.787 |

This is a ranking audit only; no final managerial targeting percentage is selected here. The lift figure is `outputs/figures/14_step10_test_risk_lift.png`.

## H. Calibration audit

| model_family | brier_score | calibration_bins | calibration_interpretation |
| --- | --- | --- | --- |
| Logistic Regression | 0.176 | 8 | Lower Brier is better; calibration is separate from ranking discrimination. |
| Decision Tree | 0.179 | 7 | Lower Brier is better; calibration is separate from ranking discrimination. |
| Random Forest | 0.170 | 8 | Lower Brier is better; calibration is separate from ranking discrimination. |
| XGBoost | 0.169 | 9 | Lower Brier is better; calibration is separate from ranking discrimination. |

Discrimination asks whether higher-risk customers rank above safer customers. Calibration asks whether predicted probabilities match observed frequencies. No Test recalibration was performed. Reliability curves are saved as `outputs/figures/13_step10_test_calibration_curves.png`.

## I-K. Overall comparison and recommendation

The final recommendation uses observed Test evidence across ROC-AUC, PR-AUC, locked-threshold Recall, F2, Top-20% lift, generalization stability, overfitting evidence, and interpretability. Accuracy is contextual only.

| model_family | role | test_roc_auc | test_pr_auc | locked_recall | locked_precision | locked_f1 | locked_f2 | top20_lift | generalization_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Secondary benchmark | 0.758 | 0.527 | 0.701 | 0.484 | 0.572 | 0.643 | 2.062 | More noticeable PR-AUC degradation; ROC-AUC degradation remains modest. |
| Decision Tree | Additional Test-evaluated model | 0.747 | 0.487 | 0.497 | 0.503 | 0.500 | 0.498 | 1.859 | Modest degradation, with a larger PR-AUC drop than ROC-AUC drop. |
| Random Forest | Additional Test-evaluated model | 0.773 | 0.559 | 0.721 | 0.484 | 0.579 | 0.657 | 1.961 | Close ranking generalization; both ROC-AUC and PR-AUC drops are small. |
| XGBoost | Primary model for subsequent SHAP | 0.777 | 0.578 | 0.714 | 0.486 | 0.579 | 0.653 | 1.995 | Close ranking generalization; both ROC-AUC and PR-AUC drops are small. |

XGBoost is the primary candidate because it has the highest Test ROC-AUC and PR-AUC while remaining close to Random Forest on locked-threshold Recall, F2, and ranking lift. Random Forest is therefore a close performance comparator. Logistic Regression has the strongest Top-20% lift and interpretability value, but lower Test discrimination and more noticeable PR-AUC degradation; it is retained as the secondary benchmark. Decision Tree is weaker across the combined Test evidence.

The primary and secondary roles above are recommendations for subsequent SHAP analysis and benchmarking only. No SHAP analysis was run in Step 10. All four models remain fully evaluated and documented.

## Validation

| check | passed | observed | expected |
| --- | --- | --- | --- |
| exact_train_customers_used_for_fitting | True | 1965 | 1965 |
| exact_test_customers_used_for_final_evaluation | True | 492 | 492 |
| test_churn_count | True | 147 | 147 |
| locked_step8_split_disjoint_and_complete | True | disjoint and complete | disjoint and complete |
| locked_parsimonious_feature_set | True | Parsimonious for all four | Parsimonious for all four |
| locked_thresholds_unchanged | True | {'Logistic Regression': 0.35, 'Decision Tree': 0.4, 'Random Forest': 0.35, 'XGBoost': 0.35} | {'Logistic Regression': 0.35, 'Decision Tree': 0.4, 'Random Forest': 0.35, 'XGBoost': 0.35} |
| preprocessing_fit_on_train_only | True | four fit calls received Train rows only | True |
| test_transform_predict_only_after_train_fit | True | Test used only after Train fit | True |
| no_customer_id_predictor | True | False | False |
| no_future_outcome_predictor | True | [] | [] |
| no_test_based_tuning | True | No search or post-Test modification | True |
| no_resampling | True | No resampling code path | True |
| no_smote | True | No SMOTE import/use | True |
| no_shap | True | No SHAP import/use | True |
| original_step9_and_step9b_evidence_preserved | True | Step 9/9B outputs exist | True |
| four_locked_models_evaluated | True | 4 | 4 |

No Test-based tuning occurred. Test was used only for the final locked evaluation after Train fitting.

## Outputs

- Source: `src/churn_analysis/evaluate_step10_test.py`
- Lightweight notebook: `notebooks/10_final_held_out_test_evaluation.ipynb`
- Report: `reports/10_final_held_out_test_evaluation.md`
- Tables: `outputs/tables/117_step10_test_model_comparison.csv` through `127_step10_output_manifest.csv`
- Figures: `outputs/figures/11_step10_test_roc_curves.png`, `12_step10_test_precision_recall_curves.png`, `13_step10_test_calibration_curves.png`, and `14_step10_test_risk_lift.png`
- Log: `outputs/logs/10_final_held_out_test_evaluation.log`

No commit or push was performed. Step 10 stops before SHAP.
