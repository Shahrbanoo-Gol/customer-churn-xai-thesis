# Customer Churn XAI Thesis

This project supports a reproducible MBA thesis on customer churn intelligence in non-contractual online retail. It will use the UCI Online Retail II dataset to develop a customer-level churn prediction workflow and translate model results into explainable and managerially relevant outputs.

Online Retail II is transactional data. The unit of analysis will ultimately be the customer, with customer-level analytical records derived from the underlying transactions. The planned workflow is: raw data → preprocessing → customer-level features → churn definition → ML models → SHAP → managerial interpretation.

The analysis may use only information available in Online Retail II. No clickstream, webpage visit, abandoned-cart, customer-satisfaction, competitor-activity, or other variables absent from the source dataset may be invented. SHAP will later be used to explain model predictions; SHAP values must not be interpreted as proof of causality.

## Current Stage

This repository currently contains project scaffolding and configuration only. Dataset analysis, data cleaning, churn-label construction, feature engineering, and model training are intentionally deferred to later analytical stages.

Files in data/raw must never be manually edited. All transformations must be performed programmatically and written to data/processed.

## Methodological Guardrails

- Never use future information to construct predictive features.
- Observation data and future churn outcome must be temporally separated.
- Test data must not influence preprocessing, feature selection, model tuning, or threshold selection.
- Raw data must remain unchanged.
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

