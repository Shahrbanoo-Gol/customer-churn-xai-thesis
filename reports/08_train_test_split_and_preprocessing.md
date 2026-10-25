# Step 8 - Train/Test Split and Leakage-Safe Preprocessing Design

## Scope and stopping point

Step 8 uses the exact Step 7 customer feature table with **2,457 customers**, one row per customer. It creates a reproducible **80% Train / 20% Test**, customer-level, stratified split with `random_state=42`. All customers belong to the same locked prediction snapshot and cutoff, so this is a **customer-level cross-sectional split**, not a temporal split.

This step stops before predictive modeling. No Logistic Regression, Decision Tree, Random Forest, XGBoost, other predictive model, SHAP, resampling, SMOTE, oversampling, undersampling, or class weighting was run.

## A-B. Train/Test sizes and class distributions

| split | customers | Churn=1 | Churn=0 | Churn=1 % | Churn=0 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,965 | 586 | 1,379 | 29.82% | 70.18% |
| Test | 492 | 147 | 345 | 29.88% | 70.12% |

The full locked snapshot contains **733 Churn=1** and **1,724 Churn=0** customers. Train and Test have no overlapping `Customer ID` values and together contain all **2,457** customers exactly once.

## C. Predictor list before preprocessing

`Churn` remains unchanged as the target. `Customer ID` is retained only as an identifier and is excluded from `X`. The complete predictor set before preprocessing has **19 features**:

`RecencyDays, Frequency, Monetary, TotalQuantity, AverageOrderValue, AverageQuantityPerOrder, ActivePurchaseDays, UniqueProducts, ProductsPerOrder, CustomerAgeDays, PurchaseSpanDays, MeanInterpurchaseDays, MedianInterpurchaseDays, HasRepeatPurchaseHistory, CancellationInvoiceCount, CancellationValue, HasCancellation, CancellationInvoiceRate, CountryCandidate`

No Step 6 outcome-window validation fields are included.

## D. Train-only redundancy findings and feature-set decision

Numeric correlations were recalculated from Train only, excluding `Customer ID`, `CountryCandidate`, and `Churn`. No Test values or target values were used.

| feature_a | feature_b | pearson_correlation_train_only | absolute_correlation_train_only |
| --- | --- | --- | --- |
| MeanInterpurchaseDays | MedianInterpurchaseDays | 0.9813259738326594 | 0.9813259738326594 |
| Frequency | ActivePurchaseDays | 0.9716789343085176 | 0.9716789343085176 |
| CustomerAgeDays | PurchaseSpanDays | 0.9497695653082904 | 0.9497695653082904 |
| Monetary | TotalQuantity | 0.9197681555774321 | 0.9197681555774321 |

The complete feature set is retained in both primary preprocessing paths. The proposed parsimonious set is advisory for a later modeling sensitivity analysis: remove `MeanInterpurchaseDays` while retaining the more robust median interval summary; remove `ActivePurchaseDays` while retaining standard invoice `Frequency`; and remove `PurchaseSpanDays` while retaining the clearer customer-history/tenure measure `CustomerAgeDays`. These are business/statistical simplifications, not target-driven selections, and no feature was deleted in Step 8.

## E. Structural missing interpurchase features

The only expected missing numeric values are the two interpurchase summaries for customers with one positive purchase event. They were not replaced with zero. Median imputation was fitted on Train only and then applied to both Train and Test; `HasRepeatPurchaseHistory` was retained so the model can distinguish structurally unavailable intervals.

| feature | train_missing_before | test_missing_before | train_derived_median | structurally_missing_in_step7 |
| --- | --- | --- | --- | --- |
| RecencyDays | 0 | 0 | 41.97708333333333 | False |
| Frequency | 0 | 0 | 4.0 | False |
| Monetary | 0 | 0 | 1099.23 | False |
| TotalQuantity | 0 | 0 | 620.0 | False |
| AverageOrderValue | 0 | 0 | 302.79 | False |
| AverageQuantityPerOrder | 0 | 0 | 168.0 | False |
| ActivePurchaseDays | 0 | 0 | 3.0 | False |
| UniqueProducts | 0 | 0 | 52.0 | False |
| ProductsPerOrder | 0 | 0 | 17.1 | False |
| CustomerAgeDays | 0 | 0 | 285.8305555555556 | False |
| PurchaseSpanDays | 0 | 0 | 224.20763888888888 | False |
| MeanInterpurchaseDays | 351 | 94 | 57.47708333333334 | True |
| MedianInterpurchaseDays | 351 | 94 | 48.14982638888889 | True |
| CancellationInvoiceCount | 0 | 0 | 0.0 | False |
| CancellationValue | 0 | 0 | 0.0 | False |
| CancellationInvoiceRate | 0 | 0 | 0.0 | False |

## F. Country One-Hot results

`CountryCandidate` remains categorical. `OneHotEncoder(handle_unknown='ignore')` was fitted on Train only. It learned **30 categories** and created **30 one-hot columns**. Test-only categories, if any, are safely transformed to all-zero country indicators rather than causing a failure.

| metric | value | category |
| --- | --- | --- |
| train_categories_learned | 30 | Australia \| Austria \| Bahrain \| Belgium \| Brazil \| Canada \| Channel Islands \| Cyprus \| Denmark \| EIRE \| European Community \| Finland \| France \| Germany \| Greece \| Iceland \| Israel \| Italy \| Japan \| Malta \| Netherlands \| Norway \| Poland \| Portugal \| Singapore \| Spain \| Sweden \| Switzerland \| United Kingdom \| Unspecified |
| test_unique_categories | 18 | Australia \| Austria \| Belgium \| Denmark \| EIRE \| Finland \| France \| Germany \| Italy \| Netherlands \| Poland \| Portugal \| Spain \| Sweden \| Switzerland \| USA \| United Kingdom \| Unspecified |
| test_only_category_count | 1 | USA |
| one_hot_columns_created | 30 | CountryCandidate one-hot columns |
| handle_unknown | ignore | unknown Test categories transform to all-zero country indicators |
| customer_id_encoded | False | Customer ID excluded before preprocessing |

## G. Logistic Regression preprocessing design

The complete predictor set is passed through a Train-fitted `ColumnTransformer`: continuous numeric predictors receive Train-only median imputation followed by Train-only `StandardScaler`; the two binary indicators remain 0/1 after missing-value handling; `CountryCandidate` receives Train-fitted one-hot encoding. The scaling statistics are therefore computed only from Train.

## H. Tree-model preprocessing design

The same complete predictor set receives Train-only median imputation for continuous numeric values, binary missing-value handling, and Train-only one-hot encoding for `CountryCandidate`. No `StandardScaler` is applied because Decision Tree, Random Forest, and XGBoost candidates do not require common numerical scale.

Skewed variables such as `Monetary`, `Frequency`, `TotalQuantity`, `AverageOrderValue`, product counts, and cancellation measures retain their original values. No automatic log transform, winsorization, clipping, or extreme-value removal was applied.

## I. Transformed feature count and mapping

Both paths produced **48 transformed columns** with identical Train/Test column names. The mapping in `95_step8_transformed_feature_name_mapping.csv` maps every original feature to its technical model input; `CountryCandidate` expands into one readable column per Train-learned category. No anonymous `x0`, `x1`, etc. names were used.

## J. Leakage audit

The leakage audit contains **27/27 passed checks**. It verifies identifier/target exclusion, absence of future outcome fields, Train/Test disjointness and coverage, Train-only correlation decisions, Train-only imputation/encoding/scaling, aligned transformed columns, finite transformed values, and the absence of resampling/model/SHAP operations.

## K. Validation results

| check | passed | observed | expected |
| --- | --- | --- | --- |
| exact_2457_customers_before_split | True | 2457 | 2457 |
| one_row_per_customer_before_split | True | 2457 | 2457 unique IDs |
| step7_churn_1_count | True | 733 | 733 |
| step7_churn_0_count | True | 1724 | 1724 |
| train_test_disjoint | True | 0 | 0 overlapping Customer ID values |
| train_test_cover_all_customers_exactly_once | True | 2457 | 2457-customer union; train + test = 2457 |
| customer_id_absent_from_predictors | True | False | False |
| churn_absent_from_predictors | True | False | False |
| future_outcome_fields_absent | True | [] | [] |
| correlation_target_unused | True | False | False |
| correlation_test_unused | True | False | False |
| logistic_train_test_columns_align | True | 48 | same Train/Test transformed feature count |
| tree_train_test_columns_align | True | 48 | same Train/Test transformed feature count |
| logistic_no_missing_or_infinite | True | finite arrays | True |
| tree_no_missing_or_infinite | True | finite arrays | True |
| preprocessing_fit_rows_equal_train | True | 1965 | 1965 |
| test_transform_only | True | Test passed to transform after Train fit | True |
| train_only_imputation | True | SimpleImputer statistics are Train-derived | True |
| train_only_country_categories | True | OneHotEncoder categories are Train-derived | True |
| train_only_scaler_statistics | True | StandardScaler is inside Train-fitted Logistic preprocessor | True |
| tree_preprocessing_not_standardized | True | Tree path has no StandardScaler | True |
| no_automatic_outlier_removal | True | Original skewed/extreme values retained | True |
| no_automatic_skewness_transform | True | No log/winsorize/clip transform | True |
| no_resampling_or_class_weighting | True | No resampling or class weighting code path | True |
| no_model_training | True | No predictive estimator fit or score invoked | True |
| no_shap | True | No SHAP import/use or explanation run | True |
| step7_feature_table_unchanged | True | d1781ea30db17701c0d243ea2371bac5f59ce85c6fe148ce8dc5c02f85c72768 | d1781ea30db17701c0d243ea2371bac5f59ce85c6fe148ce8dc5c02f85c72768 |

The Step 7 parquet hash remained unchanged after Step 8. The split artifact is ignored processed data; all audit tables, report, notebook, and log are numbered Step 8 outputs.

## L. Files created/changed

- Source: `src/churn_analysis/build_step8_preprocessing.py`
- Lightweight notebook: `notebooks/08_train_test_split_and_preprocessing.ipynb`
- Report: `reports/08_train_test_split_and_preprocessing.md`
- Ignored processed split: `data/processed/08_customer_train_test_split.parquet`
- Tables: `outputs/tables/88_step8_train_test_split_audit.csv` through `98_step8_output_manifest.csv`
- Reproducibility log: `outputs/logs/08_train_test_split_and_preprocessing.log`

The committed Step 1-7 evidence and the original Step 7 feature table were not overwritten.

## M-N. Repository status and explicit stopping confirmation

No commit or push was performed. No resampling, SMOTE, predictive model training, or SHAP occurred. Step 8 is complete and the workflow does not proceed beyond preprocessing design.

Reproducibility metadata:

| item | value |
| --- | --- |
| step7_feature_table_sha256_before | d1781ea30db17701c0d243ea2371bac5f59ce85c6fe148ce8dc5c02f85c72768 |
| step7_feature_table_sha256_after | d1781ea30db17701c0d243ea2371bac5f59ce85c6fe148ce8dc5c02f85c72768 |
| random_state | 42 |
| split_rule | train_test_split(test_size=0.20, stratify=Churn, random_state=42) |
| full_customers | 2457 |
| train_customers | 1965 |
| test_customers | 492 |
| pandas_version | 3.1.0 |
| scikit_learn_version | 1.9.1 |
| python_version | 3.11.0 |
| source_module | src/churn_analysis/build_step8_preprocessing.py |
| executed_at_utc | 2026-10-25T17:55:12.331584+00:00 |
| scope_guard | No resampling, model training, or SHAP |
