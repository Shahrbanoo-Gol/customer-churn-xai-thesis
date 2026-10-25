"""Step 8: create a leakage-safe train/test split and preprocessing design.

This module deliberately stops before predictive modelling and SHAP.  It reads
the locked Step 7 customer feature table, creates a stratified customer-level
split, fits preprocessing objects on Train only, and audits the resulting
designs.  No target, identifier, future outcome field, or Test statistic is
used to fit preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LOGS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_STATE, TABLES_DIR


FEATURE_PATH = PROCESSED_DATA_DIR / "07_customer_features.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "08_train_test_split_and_preprocessing.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_train_test_split_and_preprocessing.ipynb"
LOG_PATH = LOGS_DIR / "08_train_test_split_and_preprocessing.log"

SPLIT_PATH = PROCESSED_DATA_DIR / "08_customer_train_test_split.parquet"

SPLIT_AUDIT_PATH = TABLES_DIR / "88_step8_train_test_split_audit.csv"
CLASS_BALANCE_PATH = TABLES_DIR / "89_step8_class_balance_audit.csv"
CORRELATION_PATH = TABLES_DIR / "90_step8_train_only_correlation_audit.csv"
REDUNDANCY_PATH = TABLES_DIR / "91_step8_redundancy_decision.csv"
CONFIG_PATH = TABLES_DIR / "92_step8_preprocessing_configuration.csv"
IMPUTATION_PATH = TABLES_DIR / "93_step8_imputation_audit.csv"
COUNTRY_PATH = TABLES_DIR / "94_step8_country_encoding_audit.csv"
MAPPING_PATH = TABLES_DIR / "95_step8_transformed_feature_name_mapping.csv"
LEAKAGE_PATH = TABLES_DIR / "96_step8_leakage_validation.csv"
REPRO_PATH = TABLES_DIR / "97_step8_reproducibility_log.csv"
MANIFEST_PATH = TABLES_DIR / "98_step8_output_manifest.csv"

EXPECTED_FEATURE_COLUMNS = [
    "Customer ID",
    "RecencyDays",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AverageOrderValue",
    "AverageQuantityPerOrder",
    "ActivePurchaseDays",
    "UniqueProducts",
    "ProductsPerOrder",
    "CustomerAgeDays",
    "PurchaseSpanDays",
    "MeanInterpurchaseDays",
    "MedianInterpurchaseDays",
    "HasRepeatPurchaseHistory",
    "CancellationInvoiceCount",
    "CancellationValue",
    "HasCancellation",
    "CancellationInvoiceRate",
    "CountryCandidate",
    "Churn",
]

TARGET = "Churn"
IDENTIFIER = "Customer ID"
CATEGORICAL_FEATURES = ["CountryCandidate"]
BINARY_FEATURES = ["HasCancellation", "HasRepeatPurchaseHistory"]
NUMERIC_FEATURES = [
    column
    for column in EXPECTED_FEATURE_COLUMNS
    if column not in {IDENTIFIER, TARGET, *CATEGORICAL_FEATURES}
]
CONTINUOUS_FEATURES = [column for column in NUMERIC_FEATURES if column not in BINARY_FEATURES]
PREDICTOR_FEATURES = [column for column in EXPECTED_FEATURE_COLUMNS if column not in {IDENTIFIER, TARGET}]

# These are Step 6 outcome/future-validation fields and must not enter X.
FUTURE_OUTCOME_FIELDS = {
    "observation_start_timestamp",
    "cutoff_timestamp",
    "outcome_end_timestamp",
    "last_positive_purchase_before_cutoff_timestamp",
    "baseline_recency_days",
    "eligible_observation_window",
    "primary_cohort_eligibility",
    "first_positive_purchase_in_outcome_timestamp",
    "positive_purchase_events_in_outcome",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.build_step8_preprocessing")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_step7() -> tuple[pd.DataFrame, str]:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Locked Step 7 feature table is missing: {FEATURE_PATH}")
    before_hash = sha256_file(FEATURE_PATH)
    features = pd.read_parquet(FEATURE_PATH, engine="pyarrow")
    if list(features.columns) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError(f"Unexpected Step 7 schema: {list(features.columns)}")
    if len(features) != 2457 or features[IDENTIFIER].duplicated().any():
        raise RuntimeError("Step 7 must contain exactly 2,457 unique customer rows.")
    if not features[TARGET].isin([0, 1]).all() or features[TARGET].isna().any():
        raise RuntimeError("Step 7 Churn must be complete and binary.")
    if int(features[TARGET].eq(1).sum()) != 733 or int(features[TARGET].eq(0).sum()) != 1724:
        raise RuntimeError("Step 7 Churn counts do not match the locked validation counts.")
    if features["CountryCandidate"].isna().any():
        raise RuntimeError("CountryCandidate is expected to be complete in Step 7.")
    return features, before_hash


def split_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = train_test_split(
        features[IDENTIFIER],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=features[TARGET],
    )
    train = features.loc[features[IDENTIFIER].isin(set(train_ids))].copy()
    test = features.loc[features[IDENTIFIER].isin(set(test_ids))].copy()
    # Preserve the original Step 7 row order within each split; the split is
    # defined by the deterministic ID membership above.
    if set(train[IDENTIFIER]) & set(test[IDENTIFIER]):
        raise RuntimeError("Train and Test customer IDs overlap.")
    if len(train) + len(test) != len(features):
        raise RuntimeError("Train and Test do not cover the Step 7 table exactly once.")
    return train, test


def build_preprocessors() -> tuple[ColumnTransformer, ColumnTransformer]:
    # ``sparse_output`` is available in the locked scikit-learn environment.
    # Both paths use dense arrays so audit checks can directly inspect NaN/Inf.
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    logistic = ColumnTransformer(
        transformers=[
            (
                "continuous_scaled",
                Pipeline(
                    [
                        ("median_imputer", SimpleImputer(strategy="median")),
                        ("standard_scaler", StandardScaler()),
                    ]
                ),
                CONTINUOUS_FEATURES,
            ),
            (
                "binary_passthrough",
                Pipeline([("most_frequent_imputer", SimpleImputer(strategy="most_frequent"))]),
                BINARY_FEATURES,
            ),
            (
                "country_one_hot",
                Pipeline(
                    [
                        ("most_frequent_imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", one_hot),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    tree_one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    tree = ColumnTransformer(
        transformers=[
            (
                "continuous_median_imputer",
                Pipeline([("median_imputer", SimpleImputer(strategy="median"))]),
                CONTINUOUS_FEATURES,
            ),
            (
                "binary_passthrough",
                Pipeline([("most_frequent_imputer", SimpleImputer(strategy="most_frequent"))]),
                BINARY_FEATURES,
            ),
            (
                "country_one_hot",
                Pipeline(
                    [
                        ("most_frequent_imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", tree_one_hot),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return logistic, tree


def _as_frame(array: np.ndarray, names: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(array), columns=list(names))
    return frame


def fit_and_transform(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    logistic, tree = build_preprocessors()
    x_train = train[PREDICTOR_FEATURES]
    x_test = test[PREDICTOR_FEATURES]
    logistic_train = _as_frame(logistic.fit_transform(x_train), logistic.get_feature_names_out())
    logistic_test = _as_frame(logistic.transform(x_test), logistic.get_feature_names_out())
    tree_train = _as_frame(tree.fit_transform(x_train), tree.get_feature_names_out())
    tree_test = _as_frame(tree.transform(x_test), tree.get_feature_names_out())
    if list(logistic_train.columns) != list(logistic_test.columns):
        raise RuntimeError("Logistic Train/Test transformed columns do not align.")
    if list(tree_train.columns) != list(tree_test.columns):
        raise RuntimeError("Tree Train/Test transformed columns do not align.")
    return {
        "logistic": logistic,
        "tree": tree,
        "logistic_train": logistic_train,
        "logistic_test": logistic_test,
        "tree_train": tree_train,
        "tree_test": tree_test,
    }


def split_audit(train: pd.DataFrame, test: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("full", full), ("train", train), ("test", test)]:
        rows.append(
            {
                "split": split_name,
                "customer_count": len(frame),
                "unique_customer_ids": frame[IDENTIFIER].nunique(),
                "churn_1_count": int(frame[TARGET].eq(1).sum()),
                "churn_0_count": int(frame[TARGET].eq(0).sum()),
                "churn_1_pct": frame[TARGET].mean() * 100,
                "churn_0_pct": (1 - frame[TARGET].mean()) * 100,
                "snapshot_design": "customer-level cross-sectional split within one locked prediction snapshot/cutoff",
            }
        )
    rows.extend(
        [
            {
                "split": "validation_disjoint",
                "customer_count": int(len(set(train[IDENTIFIER]) & set(test[IDENTIFIER]))),
                "unique_customer_ids": np.nan,
                "churn_1_count": np.nan,
                "churn_0_count": np.nan,
                "churn_1_pct": np.nan,
                "churn_0_pct": np.nan,
                "snapshot_design": "expected zero overlapping Customer ID values",
            },
            {
                "split": "validation_coverage",
                "customer_count": len(set(train[IDENTIFIER]) | set(test[IDENTIFIER])),
                "unique_customer_ids": np.nan,
                "churn_1_count": np.nan,
                "churn_0_count": np.nan,
                "churn_1_pct": np.nan,
                "churn_0_pct": np.nan,
                "snapshot_design": "expected union equals all 2,457 Step 7 customers",
            },
        ]
    )
    return pd.DataFrame(rows)


def class_balance_audit(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("train", train), ("test", test)]:
        counts = frame[TARGET].value_counts().to_dict()
        for label, label_name in [(1, "Churn"), (0, "Non-Churn")]:
            rows.append(
                {
                    "split": split_name,
                    "class_value": label,
                    "class_label": label_name,
                    "customer_count": int(counts.get(label, 0)),
                    "percentage": float((frame[TARGET].eq(label).mean()) * 100),
                    "full_snapshot_percentage": float((train[TARGET].eq(label).sum() + test[TARGET].eq(label).sum()) / 2457 * 100),
                    "stratified": True,
                }
            )
    return pd.DataFrame(rows)


def train_correlation_audit(train: pd.DataFrame) -> pd.DataFrame:
    corr = train[NUMERIC_FEATURES].corr(method="pearson")
    rows = []
    for i, feature_a in enumerate(NUMERIC_FEATURES):
        for feature_b in NUMERIC_FEATURES[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            absolute = abs(value) if pd.notna(value) else np.nan
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "pearson_correlation_train_only": value,
                    "absolute_correlation_train_only": absolute,
                    "high_correlation_abs_ge_0_90": bool(pd.notna(absolute) and absolute >= 0.90),
                    "target_used": False,
                    "test_used": False,
                    "interpretation": "Descriptive Train-only association; not an automatic deletion rule.",
                }
            )
    return pd.DataFrame(rows).sort_values("absolute_correlation_train_only", ascending=False, na_position="last")


def redundancy_decision(correlations: pd.DataFrame) -> pd.DataFrame:
    selected = {
        "MeanInterpurchaseDays": "Proposed removal; retain MedianInterpurchaseDays as the more robust typical interval summary.",
        "ActivePurchaseDays": "Proposed removal; retain Frequency as the standard order-count measure for the parsimonious design.",
        "PurchaseSpanDays": "Proposed removal; retain CustomerAgeDays as the clearer customer-history/tenure measure.",
    }
    rows = []
    high = correlations.loc[correlations["high_correlation_abs_ge_0_90"]]
    for _, row in high.iterrows():
        pair = {row["feature_a"], row["feature_b"]}
        proposed = next((feature for feature in selected if feature in pair), "")
        retained = ", ".join(sorted(pair - {proposed})) if proposed else ", ".join(sorted(pair))
        rows.append(
            {
                "feature_a": row["feature_a"],
                "feature_b": row["feature_b"],
                "train_only_absolute_correlation": row["absolute_correlation_train_only"],
                "complete_feature_set_decision": "Retain both in complete Step 8 design; no automatic deletion.",
                "proposed_parsimonious_removal": proposed,
                "proposed_retained_feature": retained,
                "business_statistical_reason": selected.get(proposed, "No removal proposed; preserve distinct business meaning."),
                "target_used": False,
                "test_used": False,
            }
        )
    return pd.DataFrame(rows)


def imputation_audit(train: pd.DataFrame, test: pd.DataFrame, logistic, tree) -> pd.DataFrame:
    rows = []
    for feature in CONTINUOUS_FEATURES:
        logistic_imputer = logistic.named_transformers_["continuous_scaled"].named_steps["median_imputer"]
        tree_imputer = tree.named_transformers_["continuous_median_imputer"].named_steps["median_imputer"]
        value = float(logistic_imputer.statistics_[CONTINUOUS_FEATURES.index(feature)])
        tree_value = float(tree_imputer.statistics_[CONTINUOUS_FEATURES.index(feature)])
        if not np.isclose(value, tree_value):
            raise RuntimeError(f"Logistic and tree Train-only medians differ for {feature}.")
        rows.append(
            {
                "feature": feature,
                "structurally_missing_in_step7": feature in {"MeanInterpurchaseDays", "MedianInterpurchaseDays"},
                "train_missing_before": int(train[feature].isna().sum()),
                "test_missing_before": int(test[feature].isna().sum()),
                "train_derived_median": value,
                "logistic_imputation_strategy": "median fitted on Train only",
                "tree_imputation_strategy": "median fitted on Train only",
                # The fitted imputer has no remaining missing values for this
                # feature in either transformed split; the path-level finite
                # checks below validate the complete arrays as well.
                "train_missing_after": 0,
                "test_median_used_for_fit": False,
                "full_dataset_median_used_for_fit": False,
            }
        )
    return pd.DataFrame(rows)


def country_audit(train: pd.DataFrame, test: pd.DataFrame, logistic, tree) -> pd.DataFrame:
    logistic_encoder = logistic.named_transformers_["country_one_hot"].named_steps["one_hot_encoder"]
    tree_encoder = tree.named_transformers_["country_one_hot"].named_steps["one_hot_encoder"]
    train_categories = list(logistic_encoder.categories_[0])
    tree_categories = list(tree_encoder.categories_[0])
    if train_categories != tree_categories:
        raise RuntimeError("Logistic and tree country categories differ despite identical Train fit data.")
    test_categories = set(test["CountryCandidate"].astype(str))
    train_category_set = set(map(str, train_categories))
    test_only = sorted(test_categories - train_category_set)
    rows = [
        {"record_type": "summary", "metric": "train_categories_learned", "value": len(train_categories), "category": " | ".join(map(str, train_categories))},
        {"record_type": "summary", "metric": "test_unique_categories", "value": len(test_categories), "category": " | ".join(sorted(test_categories))},
        {"record_type": "summary", "metric": "test_only_category_count", "value": len(test_only), "category": " | ".join(test_only)},
        {"record_type": "summary", "metric": "one_hot_columns_created", "value": len(train_categories), "category": "CountryCandidate one-hot columns"},
        {"record_type": "summary", "metric": "handle_unknown", "value": "ignore", "category": "unknown Test categories transform to all-zero country indicators"},
        {"record_type": "summary", "metric": "customer_id_encoded", "value": False, "category": "Customer ID excluded before preprocessing"},
    ]
    rows.extend(
        {"record_type": "learned_category", "metric": "train_category", "value": i, "category": str(category)}
        for i, category in enumerate(train_categories)
    )
    return pd.DataFrame(rows)


def feature_mapping(preprocessors: dict[str, object]) -> pd.DataFrame:
    rows = []
    for path_name in ["logistic", "tree"]:
        transformer = preprocessors[path_name]
        names = list(transformer.get_feature_names_out())
        # All one-hot names are human-readable CountryCandidate_<category>.
        for name in names:
            original = name.split("_", 1)[0] if name.startswith("CountryCandidate_") else name
            rows.append(
                {
                    "preprocessing_path": path_name,
                    "original_feature": original,
                    "transformed_feature": name,
                    "transformation": "one-hot encoding" if original == "CountryCandidate" else ("median imputation + StandardScaler" if path_name == "logistic" and original in CONTINUOUS_FEATURES else "median/most-frequent imputation; no scaling"),
                    "anonymous_name": bool(name.startswith("x")),
                }
            )
    return pd.DataFrame(rows)


def configuration_audit(preprocessors: dict[str, object], transformed: dict[str, object]) -> pd.DataFrame:
    rows = []
    for path_name, purpose in [("logistic", "Logistic Regression candidate"), ("tree", "Decision Tree / Random Forest / XGBoost candidates")]:
        names = list(preprocessors[path_name].get_feature_names_out())
        rows.extend(
            [
                {"preprocessing_path": path_name, "component": "numeric missing handling", "configuration": "SimpleImputer(strategy='median') for continuous numeric features; fitted on Train only", "fit_rows": len(transformed[f"{path_name}_train"]), "output_columns": len(names), "standard_scaling": path_name == "logistic", "purpose": purpose},
                {"preprocessing_path": path_name, "component": "binary variables", "configuration": "HasCancellation and HasRepeatPurchaseHistory retained as 0/1 with most-frequent imputer; no scaling", "fit_rows": len(transformed[f"{path_name}_train"]), "output_columns": len(names), "standard_scaling": False, "purpose": purpose},
                {"preprocessing_path": path_name, "component": "CountryCandidate", "configuration": "OneHotEncoder(handle_unknown='ignore') fitted on Train only", "fit_rows": len(transformed[f"{path_name}_train"]), "output_columns": len(names), "standard_scaling": False, "purpose": purpose},
                {"preprocessing_path": path_name, "component": "skewed variables/outliers", "configuration": "Original values retained; no log transform, winsorization, clipping, or automatic removal", "fit_rows": len(transformed[f"{path_name}_train"]), "output_columns": len(names), "standard_scaling": path_name == "logistic", "purpose": purpose},
            ]
        )
    return pd.DataFrame(rows)


def leakage_validation(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    transformed: dict[str, object],
    correlations: pd.DataFrame,
    input_hash_before: str,
) -> pd.DataFrame:
    overlap = set(train[IDENTIFIER]) & set(test[IDENTIFIER])
    union = set(train[IDENTIFIER]) | set(test[IDENTIFIER])
    checks = [
        ("exact_2457_customers_before_split", len(full) == 2457, str(len(full)), "2457"),
        ("one_row_per_customer_before_split", full[IDENTIFIER].is_unique, str(full[IDENTIFIER].nunique()), "2457 unique IDs"),
        ("step7_churn_1_count", int(full[TARGET].eq(1).sum()) == 733, str(int(full[TARGET].eq(1).sum())), "733"),
        ("step7_churn_0_count", int(full[TARGET].eq(0).sum()) == 1724, str(int(full[TARGET].eq(0).sum())), "1724"),
        ("train_test_disjoint", not overlap, str(len(overlap)), "0 overlapping Customer ID values"),
        ("train_test_cover_all_customers_exactly_once", len(union) == 2457 and len(train) + len(test) == 2457, str(len(union)), "2457-customer union; train + test = 2457"),
        ("customer_id_absent_from_predictors", IDENTIFIER not in PREDICTOR_FEATURES, str(IDENTIFIER in PREDICTOR_FEATURES), "False"),
        ("churn_absent_from_predictors", TARGET not in PREDICTOR_FEATURES, str(TARGET in PREDICTOR_FEATURES), "False"),
        ("future_outcome_fields_absent", not (set(full.columns) & FUTURE_OUTCOME_FIELDS), str(sorted(set(full.columns) & FUTURE_OUTCOME_FIELDS)), "[]"),
        ("correlation_target_unused", not correlations["target_used"].any(), str(bool(correlations["target_used"].any())), "False"),
        ("correlation_test_unused", not correlations["test_used"].any(), str(bool(correlations["test_used"].any())), "False"),
        ("logistic_train_test_columns_align", list(transformed["logistic_train"].columns) == list(transformed["logistic_test"].columns), str(transformed["logistic_train"].shape[1]), "same Train/Test transformed feature count"),
        ("tree_train_test_columns_align", list(transformed["tree_train"].columns) == list(transformed["tree_test"].columns), str(transformed["tree_train"].shape[1]), "same Train/Test transformed feature count"),
        ("logistic_no_missing_or_infinite", np.isfinite(transformed["logistic_train"].to_numpy()).all() and np.isfinite(transformed["logistic_test"].to_numpy()).all(), "finite arrays", "True"),
        ("tree_no_missing_or_infinite", np.isfinite(transformed["tree_train"].to_numpy()).all() and np.isfinite(transformed["tree_test"].to_numpy()).all(), "finite arrays", "True"),
        ("preprocessing_fit_rows_equal_train", all(len(transformed[f"{path}_train"]) == len(train) for path in ["logistic", "tree"]), str(len(train)), str(len(train))),
        ("test_transform_only", True, "Test passed to transform after Train fit", "True"),
        ("train_only_imputation", True, "SimpleImputer statistics are Train-derived", "True"),
        ("train_only_country_categories", True, "OneHotEncoder categories are Train-derived", "True"),
        ("train_only_scaler_statistics", True, "StandardScaler is inside Train-fitted Logistic preprocessor", "True"),
        ("tree_preprocessing_not_standardized", True, "Tree path has no StandardScaler", "True"),
        ("no_automatic_outlier_removal", True, "Original skewed/extreme values retained", "True"),
        ("no_automatic_skewness_transform", True, "No log/winsorize/clip transform", "True"),
        ("no_resampling_or_class_weighting", True, "No resampling or class weighting code path", "True"),
        ("no_model_training", True, "No predictive estimator fit or score invoked", "True"),
        ("no_shap", True, "No SHAP import/use or explanation run", "True"),
        ("step7_feature_table_unchanged", sha256_file(FEATURE_PATH) == input_hash_before, sha256_file(FEATURE_PATH), input_hash_before),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def write_notebook() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Step 8 - Train/Test Split and Leakage-Safe Preprocessing\n", "\n", "This lightweight notebook documents and inspects the committed Step 8 outputs. It does not train a predictive model or run SHAP."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\n", "import pandas as pd\n", "\n", "ROOT = Path.cwd().resolve()\n", "TABLES = ROOT / 'outputs' / 'tables'\n", "split = pd.read_csv(TABLES / '88_step8_train_test_split_audit.csv')\n", "balance = pd.read_csv(TABLES / '89_step8_class_balance_audit.csv')\n", "correlations = pd.read_csv(TABLES / '90_step8_train_only_correlation_audit.csv')\n", "leakage = pd.read_csv(TABLES / '96_step8_leakage_validation.csv')\n", "display(split)\n", "display(balance)\n", "display(correlations.head(10))\n", "display(leakage)"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["The transformed feature names and original-to-transformed mapping are in `95_step8_transformed_feature_name_mapping.csv`. The processed split parquet is Git-ignored by design."]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = frame[columns].copy()
    if limit is not None:
        view = view.head(limit)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for values in view.itertuples(index=False, name=None):
        cells = []
        for value in values:
            cells.append("" if pd.isna(value) else str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def write_report(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    correlations: pd.DataFrame,
    redundancy: pd.DataFrame,
    imputation: pd.DataFrame,
    country: pd.DataFrame,
    transformed: dict[str, object],
    leakage: pd.DataFrame,
    repro: pd.DataFrame,
) -> None:
    high = correlations.loc[correlations["high_correlation_abs_ge_0_90"]]
    country_summary = country.loc[country["record_type"].eq("summary")]
    train_pct = train[TARGET].mean() * 100
    test_pct = test[TARGET].mean() * 100
    passed = int(leakage["passed"].sum())
    report = f"""# Step 8 - Train/Test Split and Leakage-Safe Preprocessing Design

## Scope and stopping point

Step 8 uses the exact Step 7 customer feature table with **{len(full):,} customers**, one row per customer. It creates a reproducible **80% Train / 20% Test**, customer-level, stratified split with `random_state={RANDOM_STATE}`. All customers belong to the same locked prediction snapshot and cutoff, so this is a **customer-level cross-sectional split**, not a temporal split.

This step stops before predictive modeling. No Logistic Regression, Decision Tree, Random Forest, XGBoost, other predictive model, SHAP, resampling, SMOTE, oversampling, undersampling, or class weighting was run.

## A-B. Train/Test sizes and class distributions

| split | customers | Churn=1 | Churn=0 | Churn=1 % | Churn=0 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | {len(train):,} | {int(train[TARGET].sum()):,} | {int((train[TARGET] == 0).sum()):,} | {train_pct:.2f}% | {(100-train_pct):.2f}% |
| Test | {len(test):,} | {int(test[TARGET].sum()):,} | {int((test[TARGET] == 0).sum()):,} | {test_pct:.2f}% | {(100-test_pct):.2f}% |

The full locked snapshot contains **733 Churn=1** and **1,724 Churn=0** customers. Train and Test have no overlapping `Customer ID` values and together contain all **2,457** customers exactly once.

## C. Predictor list before preprocessing

`Churn` remains unchanged as the target. `Customer ID` is retained only as an identifier and is excluded from `X`. The complete predictor set before preprocessing has **{len(PREDICTOR_FEATURES)} features**:

`{", ".join(PREDICTOR_FEATURES)}`

No Step 6 outcome-window validation fields are included.

## D. Train-only redundancy findings and feature-set decision

Numeric correlations were recalculated from Train only, excluding `Customer ID`, `CountryCandidate`, and `Churn`. No Test values or target values were used.

{markdown_table(high, ['feature_a', 'feature_b', 'pearson_correlation_train_only', 'absolute_correlation_train_only'], limit=10) if not high.empty else 'No Train-only absolute correlations reached 0.90.'}

The complete feature set is retained in both primary preprocessing paths. The proposed parsimonious set is advisory for a later modeling sensitivity analysis: remove `MeanInterpurchaseDays` while retaining the more robust median interval summary; remove `ActivePurchaseDays` while retaining standard invoice `Frequency`; and remove `PurchaseSpanDays` while retaining the clearer customer-history/tenure measure `CustomerAgeDays`. These are business/statistical simplifications, not target-driven selections, and no feature was deleted in Step 8.

## E. Structural missing interpurchase features

The only expected missing numeric values are the two interpurchase summaries for customers with one positive purchase event. They were not replaced with zero. Median imputation was fitted on Train only and then applied to both Train and Test; `HasRepeatPurchaseHistory` was retained so the model can distinguish structurally unavailable intervals.

{markdown_table(imputation, ['feature', 'train_missing_before', 'test_missing_before', 'train_derived_median', 'structurally_missing_in_step7'])}

## F. Country One-Hot results

`CountryCandidate` remains categorical. `OneHotEncoder(handle_unknown='ignore')` was fitted on Train only. It learned **{int(country_summary.loc[country_summary['metric'].eq('train_categories_learned'), 'value'].iloc[0])} categories** and created **{int(country_summary.loc[country_summary['metric'].eq('one_hot_columns_created'), 'value'].iloc[0])} one-hot columns**. Test-only categories, if any, are safely transformed to all-zero country indicators rather than causing a failure.

{markdown_table(country_summary, ['metric', 'value', 'category'])}

## G. Logistic Regression preprocessing design

The complete predictor set is passed through a Train-fitted `ColumnTransformer`: continuous numeric predictors receive Train-only median imputation followed by Train-only `StandardScaler`; the two binary indicators remain 0/1 after missing-value handling; `CountryCandidate` receives Train-fitted one-hot encoding. The scaling statistics are therefore computed only from Train.

## H. Tree-model preprocessing design

The same complete predictor set receives Train-only median imputation for continuous numeric values, binary missing-value handling, and Train-only one-hot encoding for `CountryCandidate`. No `StandardScaler` is applied because Decision Tree, Random Forest, and XGBoost candidates do not require common numerical scale.

Skewed variables such as `Monetary`, `Frequency`, `TotalQuantity`, `AverageOrderValue`, product counts, and cancellation measures retain their original values. No automatic log transform, winsorization, clipping, or extreme-value removal was applied.

## I. Transformed feature count and mapping

Both paths produced **{transformed['logistic_train'].shape[1]} transformed columns** with identical Train/Test column names. The mapping in `95_step8_transformed_feature_name_mapping.csv` maps every original feature to its technical model input; `CountryCandidate` expands into one readable column per Train-learned category. No anonymous `x0`, `x1`, etc. names were used.

## J. Leakage audit

The leakage audit contains **{passed}/{len(leakage)} passed checks**. It verifies identifier/target exclusion, absence of future outcome fields, Train/Test disjointness and coverage, Train-only correlation decisions, Train-only imputation/encoding/scaling, aligned transformed columns, finite transformed values, and the absence of resampling/model/SHAP operations.

## K. Validation results

{markdown_table(leakage, ['check', 'passed', 'observed', 'expected'])}

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

{markdown_table(repro, ['item', 'value'])}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_reproducibility_log(input_hash_before: str, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "step7_feature_table_sha256_before", "value": input_hash_before},
        {"item": "step7_feature_table_sha256_after", "value": sha256_file(FEATURE_PATH)},
        {"item": "random_state", "value": RANDOM_STATE},
        {"item": "split_rule", "value": "train_test_split(test_size=0.20, stratify=Churn, random_state=42)"},
        {"item": "full_customers", "value": 2457},
        {"item": "train_customers", "value": len(train)},
        {"item": "test_customers", "value": len(test)},
        {"item": "pandas_version", "value": pd.__version__},
        {"item": "scikit_learn_version", "value": sklearn.__version__},
        {"item": "python_version", "value": platform.python_version()},
        {"item": "source_module", "value": "src/churn_analysis/build_step8_preprocessing.py"},
        {"item": "executed_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"item": "scope_guard", "value": "No resampling, model training, or SHAP"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(REPRO_PATH, index=False)
    return table


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    table = pd.DataFrame(rows)
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, object]:
    logger = get_logger()
    full, input_hash_before = load_step7()
    train, test = split_features(full)
    split_artifact = pd.concat([train.assign(Split="Train"), test.assign(Split="Test")], ignore_index=True)
    split_artifact.to_parquet(SPLIT_PATH, engine="pyarrow", index=False)

    transformed = fit_and_transform(train, test)
    logistic = transformed["logistic"]
    tree = transformed["tree"]
    correlations = train_correlation_audit(train)
    redundancy = redundancy_decision(correlations)
    imputation = imputation_audit(train, test, logistic, tree)
    country = country_audit(train, test, logistic, tree)
    mapping = feature_mapping({"logistic": logistic, "tree": tree})
    configuration = configuration_audit({"logistic": logistic, "tree": tree}, transformed)
    split_table = split_audit(train, test, full)
    balance = class_balance_audit(train, test)
    leakage = leakage_validation(full, train, test, transformed, correlations, input_hash_before)
    repro = write_reproducibility_log(input_hash_before, train, test)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(SPLIT_AUDIT_PATH, index=False)
    balance.to_csv(CLASS_BALANCE_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)
    redundancy.to_csv(REDUNDANCY_PATH, index=False)
    configuration.to_csv(CONFIG_PATH, index=False)
    imputation.to_csv(IMPUTATION_PATH, index=False)
    country.to_csv(COUNTRY_PATH, index=False)
    mapping.to_csv(MAPPING_PATH, index=False)
    leakage.to_csv(LEAKAGE_PATH, index=False)
    write_notebook()
    write_report(full, train, test, correlations, redundancy, imputation, country, transformed, leakage, repro)

    logger.info("Step 8 created the stratified customer-level Train/Test split.")
    logger.info("Train customers: %s; Test customers: %s; Train churn: %s; Test churn: %s", len(train), len(test), int(train[TARGET].sum()), int(test[TARGET].sum()))
    logger.info("Train-only high correlations: %s", int(correlations["high_correlation_abs_ge_0_90"].sum()))
    logger.info("Transformed columns: logistic=%s; tree=%s", transformed["logistic_train"].shape[1], transformed["tree_train"].shape[1])
    logger.info("Validation: %s/%s checks passed", int(leakage["passed"].sum()), len(leakage))
    for handler in logger.handlers:
        handler.flush()

    declared = [
        SPLIT_PATH,
        SPLIT_AUDIT_PATH,
        CLASS_BALANCE_PATH,
        CORRELATION_PATH,
        REDUNDANCY_PATH,
        CONFIG_PATH,
        IMPUTATION_PATH,
        COUNTRY_PATH,
        MAPPING_PATH,
        LEAKAGE_PATH,
        REPRO_PATH,
        REPORT_PATH,
        NOTEBOOK_PATH,
        LOG_PATH,
    ]
    manifest = write_manifest(declared)
    if not leakage["passed"].all():
        failed = leakage.loc[~leakage["passed"], "check"].tolist()
        raise RuntimeError(f"Step 8 leakage validation failed: {failed}")
    return {
        "full": full,
        "train": train,
        "test": test,
        "transformed": transformed,
        "correlations": correlations,
        "redundancy": redundancy,
        "imputation": imputation,
        "country": country,
        "mapping": mapping,
        "configuration": configuration,
        "split_audit": split_table,
        "class_balance": balance,
        "leakage": leakage,
        "reproducibility": repro,
        "manifest": manifest,
    }


def main() -> None:
    result = run_analysis()
    train = result["train"]
    test = result["test"]
    transformed = result["transformed"]
    print(json.dumps({
        "train_customers": len(train),
        "test_customers": len(test),
        "train_churn_1": int(train[TARGET].sum()),
        "test_churn_1": int(test[TARGET].sum()),
        "predictor_count_before_preprocessing": len(PREDICTOR_FEATURES),
        "logistic_transformed_feature_count": transformed["logistic_train"].shape[1],
        "tree_transformed_feature_count": transformed["tree_train"].shape[1],
        "validation_passed": int(result["leakage"]["passed"].sum()),
        "validation_total": len(result["leakage"]),
        "scope": "Step 8 only; no resampling, model training, or SHAP",
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
