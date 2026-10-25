# Customer Churn XAI Thesis

This project supports a reproducible MBA thesis on customer churn intelligence in non-contractual online retail. It will use the UCI Online Retail II dataset to develop a customer-level churn prediction workflow and translate model results into explainable and managerially relevant outputs.

Online Retail II is transactional data. The unit of analysis will ultimately be the customer, with customer-level analytical records derived from the underlying transactions. The planned workflow is: raw data → preprocessing → customer-level features → churn definition → ML models → SHAP → managerial interpretation.

The analysis may use only information available in Online Retail II. No clickstream, webpage visit, abandoned-cart, customer-satisfaction, competitor-activity, or other variables absent from the source dataset may be invented. SHAP is used to explain the locked model's predictions; SHAP values must not be interpreted as proof of causality.

## Current Stage

The current repository contains the reproducible raw audit, worksheet reconciliation, transaction-quality audits, customer-analysis preparation bases, Step 5 interpurchase-time analysis, Step 6 primary retention cohort/outcome table, Step 7 observation-period customer feature engineering and audit, Step 8 leakage-safe split/preprocessing design, Step 9 Train-only model training/selection and recall audit, Step 10 final held-out Test evaluation, and Step 11 SHAP explanation of the locked selected XGBoost model. Step 8 creates the locked 2,457-customer stratified Train/Test split. Step 9 and Step 9B select the four locked model configurations, assess weighting/threshold sensitivity, and preserve Train-only evidence. Step 10 evaluates those locked configurations on Test and recommends XGBoost as the primary model, with Logistic Regression as an interpretable benchmark. Step 11 explains XGBoost on the held-out Test set; managerial intervention design remains intentionally deferred.

Files in data/raw must never be manually edited. All transformations must be performed programmatically and written to data/processed.

## Methodological Guardrails

- Never use future information to construct predictive features.
- Observation data and future churn outcome must be temporally separated.
- Test data must not influence preprocessing, feature selection, model tuning, or threshold selection.
- Raw data must remain unchanged.
- The primary analysis removes additional exact full-row duplicates within each raw worksheet using the eight original source variables and `keep='first'` logic before worksheet reconciliation. `SourceSheet` is technical lineage only and is never part of duplicate equality.
- Rows with missing Customer ID remain in the reconciled transaction source but are excluded from customer-linked datasets; missing Description rows remain when transaction information is otherwise valid.
- All random processes must use recorded random seeds.
- Every major analytical step must save reproducible outputs.
- Do not introduce variables not available in the source dataset.
- Do not interpret SHAP values as causal effects.

## Planned Project Structure

```text
customer-churn-xai-thesis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── churn_analysis/
│       ├── __init__.py
│       └── config.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── logs/
├── reports/
├── prompts/
├── README.md
├── requirements.txt
├── .gitignore
└── .python-version
```

