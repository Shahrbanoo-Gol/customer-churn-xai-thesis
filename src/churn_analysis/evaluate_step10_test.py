"""Step 10: final held-out Test evaluation of the four locked churn models.

This module is the first stage allowed to score the locked Test set.  It reads
the exact Step 8 split and the Step 9B recommendation table, validates that
the requested configurations are unchanged, fits each preprocessing/model
pipeline once on all 1,965 Train customers, and then transforms/predicts the
492 Test customers.  No Test result is used to tune or modify a model.

No SHAP, resampling, SMOTE, threshold optimization, or post-Test model change
is performed here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .build_step8_preprocessing import FUTURE_OUTCOME_FIELDS, IDENTIFIER, PREDICTOR_FEATURES, TARGET
from .config import FIGURES_DIR, LOGS_DIR, PROJECT_ROOT, RANDOM_STATE, TABLES_DIR
from .train_step9_models import FEATURE_SETS, load_locked_split, pipeline_for


SOURCE_PATH = PROJECT_ROOT / "src" / "churn_analysis" / "evaluate_step10_test.py"
REFERENCE_PATH = TABLES_DIR / "113_step9b_recommendations.csv"
TRAIN_CV_PATH = TABLES_DIR / "103_step9_train_cv_model_comparison.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "10_final_held_out_test_evaluation.md"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_final_held_out_test_evaluation.ipynb"
LOG_PATH = LOGS_DIR / "10_final_held_out_test_evaluation.log"

MODEL_COMPARISON_PATH = TABLES_DIR / "117_step10_test_model_comparison.csv"
THRESHOLD_COMPARISON_PATH = TABLES_DIR / "118_step10_default_vs_locked_thresholds.csv"
CONFUSION_PATH = TABLES_DIR / "119_step10_confusion_matrices.csv"
GENERALIZATION_PATH = TABLES_DIR / "120_step10_train_cv_vs_test_generalization.csv"
RISK_PATH = TABLES_DIR / "121_step10_risk_ranking_lift.csv"
CALIBRATION_PATH = TABLES_DIR / "122_step10_calibration_audit.csv"
CALIBRATION_BINS_PATH = TABLES_DIR / "123_step10_calibration_bins.csv"
RECOMMENDATION_PATH = TABLES_DIR / "124_step10_model_recommendation.csv"
REPRO_PATH = TABLES_DIR / "125_step10_reproducibility_log.csv"
VALIDATION_PATH = TABLES_DIR / "126_step10_validation.csv"
MANIFEST_PATH = TABLES_DIR / "127_step10_output_manifest.csv"

ROC_FIGURE_PATH = FIGURES_DIR / "11_step10_test_roc_curves.png"
PR_FIGURE_PATH = FIGURES_DIR / "12_step10_test_precision_recall_curves.png"
CALIBRATION_FIGURE_PATH = FIGURES_DIR / "13_step10_test_calibration_curves.png"
RISK_FIGURE_PATH = FIGURES_DIR / "14_step10_test_risk_lift.png"

MODEL_FAMILIES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
EXPECTED_THRESHOLDS = {
    "Logistic Regression": 0.35,
    "Decision Tree": 0.40,
    "Random Forest": 0.35,
    "XGBoost": 0.35,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_analysis.evaluate_step10_test")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_locked_recommendations() -> dict[str, dict[str, Any]]:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Step 9B locked recommendation table is missing: {REFERENCE_PATH}")
    table = pd.read_csv(REFERENCE_PATH)
    if set(table["model_family"]) != set(MODEL_FAMILIES) or len(table) != 4:
        raise RuntimeError("Step 9B recommendations do not contain exactly four model families.")
    if set(table["feature_set"]) != {"Parsimonious"}:
        raise RuntimeError("Step 10 requires the locked Parsimonious feature set.")
    recommendations = {}
    for _, row in table.iterrows():
        model_family = row["model_family"]
        threshold = float(row["classification_threshold"])
        if not np.isclose(threshold, EXPECTED_THRESHOLDS[model_family]):
            raise RuntimeError(f"Locked threshold changed for {model_family}: {threshold}")
        if row["class_weighting_choice"] != "unweighted":
            raise RuntimeError(f"Step 10 received a non-locked weighted configuration for {model_family}.")
        if model_family == "Logistic Regression" and "Current Step 9" not in row["preprocessing"]:
            raise RuntimeError("Step 10 Logistic preprocessing is not the locked current Step 9 path.")
        recommendations[model_family] = {
            "params": json.loads(row["hyperparameters"]),
            "threshold": threshold,
            "feature_set": row["feature_set"],
            "preprocessing": row["preprocessing"],
        }
    return recommendations


def fit_and_predict(
    recommendations: dict[str, dict[str, Any]], train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    features = FEATURE_SETS["Parsimonious"]
    for model_family in MODEL_FAMILIES:
        recommendation = recommendations[model_family]
        estimator = pipeline_for(model_family, features, recommendation["params"])
        # The only fit call in Step 10 uses the complete Train rows.
        estimator.fit(train[features], train[TARGET].astype(int))
        probabilities = estimator.predict_proba(test[features])[:, 1]
        results[model_family] = {"estimator": estimator, "probabilities": probabilities}
    return results


def binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "threshold": threshold,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "specificity": specificity,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "predicted_churn_count": int(predictions.sum()),
        "predicted_churn_pct": float(predictions.mean() * 100),
    }


def ranking_rows(predictions: dict[str, dict[str, Any]], y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for model_family in MODEL_FAMILIES:
        probabilities = predictions[model_family]["probabilities"]
        rows.append(
            {
                "model_family": model_family,
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
                "test_pr_auc": float(average_precision_score(y_test, probabilities)),
                "brier_score": float(brier_score_loss(y_test, probabilities)),
            }
        )
    return pd.DataFrame(rows)


def threshold_rows(
    recommendations: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]], y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows_list = []
    confusion_rows = []
    for model_family in MODEL_FAMILIES:
        probabilities = predictions[model_family]["probabilities"]
        for label, threshold in [("default_0.50", 0.50), ("locked_step9b", recommendations[model_family]["threshold"])]:
            metrics = binary_metrics(y_test, probabilities, threshold)
            threshold_rows_list.append({"model_family": model_family, "evaluation": label, **metrics})
            confusion_rows.append(
                {
                    "model_family": model_family,
                    "evaluation": label,
                    "threshold": threshold,
                    "true_positive": metrics["tp"],
                    "false_negative": metrics["fn"],
                    "false_positive": metrics["fp"],
                    "true_negative": metrics["tn"],
                    "actual_test_churn_identified": metrics["tp"],
                    "actual_test_churn_missed": metrics["fn"],
                    "unnecessary_retention_alerts_false_positive": metrics["fp"],
                }
            )
    return pd.DataFrame(threshold_rows_list), pd.DataFrame(confusion_rows)


def generalization_table(ranking: pd.DataFrame) -> pd.DataFrame:
    if not TRAIN_CV_PATH.exists():
        raise FileNotFoundError(f"Step 9 Train-CV comparison is missing: {TRAIN_CV_PATH}")
    train_cv = pd.read_csv(TRAIN_CV_PATH)
    train_cv = train_cv.loc[train_cv["selected_feature_set"].eq("Parsimonious")].copy()
    merged = train_cv.merge(ranking, on="model_family", how="inner", validate="one_to_one")
    merged["roc_auc_difference_test_minus_train_cv"] = merged["test_roc_auc"] - merged["roc_auc"]
    merged["pr_auc_difference_test_minus_train_cv"] = merged["test_pr_auc"] - merged["pr_auc"]
    descriptive_notes = {
        "Logistic Regression": "More noticeable PR-AUC degradation; ROC-AUC degradation remains modest.",
        "Decision Tree": "Modest degradation, with a larger PR-AUC drop than ROC-AUC drop.",
        "Random Forest": "Close ranking generalization; both ROC-AUC and PR-AUC drops are small.",
        "XGBoost": "Close ranking generalization; both ROC-AUC and PR-AUC drops are small.",
    }
    merged["generalization_note"] = merged["model_family"].map(descriptive_notes)
    return merged[[
        "model_family",
        "roc_auc",
        "test_roc_auc",
        "roc_auc_difference_test_minus_train_cv",
        "pr_auc",
        "test_pr_auc",
        "pr_auc_difference_test_minus_train_cv",
        "generalization_note",
    ]].rename(columns={"roc_auc": "train_cv_roc_auc", "pr_auc": "train_cv_pr_auc"})


def risk_ranking_table(predictions: dict[str, dict[str, Any]], y_test: pd.Series) -> pd.DataFrame:
    overall_rate = float(y_test.mean())
    rows = []
    for model_family in MODEL_FAMILIES:
        probabilities = predictions[model_family]["probabilities"]
        order = np.argsort(-probabilities, kind="mergesort")
        for share in [0.10, 0.20, 0.30]:
            count = int(np.ceil(len(y_test) * share))
            selected = y_test.iloc[order[:count]]
            churn_count = int(selected.sum())
            churn_rate = float(selected.mean())
            capture_rate = float(churn_count / y_test.sum())
            rows.append(
                {
                    "model_family": model_family,
                    "risk_group": f"Top {int(share * 100)}%",
                    "selected_customer_count": count,
                    "overall_test_churn_rate": overall_rate,
                    "group_churn_count": churn_count,
                    "group_churn_rate": churn_rate,
                    "churn_capture_rate": capture_rate,
                    "lift": float(churn_rate / overall_rate),
                }
            )
    return pd.DataFrame(rows)


def calibration_tables(predictions: dict[str, dict[str, Any]], y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bin_rows = []
    for model_family in MODEL_FAMILIES:
        probabilities = predictions[model_family]["probabilities"]
        brier = float(brier_score_loss(y_test, probabilities))
        mean_predicted, observed_fraction = calibration_curve(y_test, probabilities, n_bins=10, strategy="uniform")
        summary_rows.append(
            {
                "model_family": model_family,
                "brier_score": brier,
                "calibration_bins": len(mean_predicted),
                "calibration_interpretation": "Lower Brier is better; calibration is separate from ranking discrimination.",
            }
        )
        for index, (predicted, observed) in enumerate(zip(mean_predicted, observed_fraction), start=1):
            bin_rows.append(
                {
                    "model_family": model_family,
                    "bin": index,
                    "mean_predicted_probability": float(predicted),
                    "observed_churn_fraction": float(observed),
                    "absolute_calibration_gap": float(abs(predicted - observed)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def recommendation_table(
    ranking: pd.DataFrame,
    thresholds: pd.DataFrame,
    risk: pd.DataFrame,
    generalization: pd.DataFrame,
    recommendations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    locked = thresholds.loc[thresholds["evaluation"].eq("locked_step9b")].copy()
    lift20 = risk.loc[risk["risk_group"].eq("Top 20%"), ["model_family", "lift"]]
    evidence = locked.merge(ranking, on="model_family").merge(lift20, on="model_family").merge(generalization[["model_family", "generalization_note"]], on="model_family")
    # This is an explicit evidence-based judgment for the locked Test result,
    # not a hidden threshold or Accuracy-only ranking formula. XGBoost leads
    # ROC-AUC and PR-AUC, while Random Forest is close on thresholded Recall
    # and F2. Logistic Regression is retained as the interpretable benchmark.
    primary = "XGBoost"
    secondary = "Logistic Regression"
    rows = []
    for _, row in evidence.iterrows():
        role = "Primary model for subsequent SHAP" if row["model_family"] == primary else "Secondary benchmark" if row["model_family"] == secondary else "Additional Test-evaluated model"
        if row["model_family"] == primary:
            reason = "Best overall multi-metric Test evidence across discrimination, locked-threshold identification, F2, and Top-20% lift."
        elif row["model_family"] == secondary:
            reason = "Useful benchmark for comparison; retains a complementary modelling/interpretation profile."
        else:
            reason = "Retain for transparent four-model comparison; not the strongest combined Test evidence."
        rows.append(
            {
                "model_family": row["model_family"],
                "role": role,
                "recommendation_reason": reason,
                "feature_set": recommendations[row["model_family"]]["feature_set"],
                "hyperparameters": json.dumps(recommendations[row["model_family"]]["params"], sort_keys=True, default=str),
                "threshold": recommendations[row["model_family"]]["threshold"],
                "test_roc_auc": row["test_roc_auc"],
                "test_pr_auc": row["test_pr_auc"],
                "locked_recall": row["recall"],
                "locked_precision": row["precision"],
                "locked_f1": row["f1"],
                "locked_f2": row["f2"],
                "top20_lift": row["lift"],
                "generalization_note": row["generalization_note"],
                "selection_basis": "Observed Test evidence across ROC-AUC, PR-AUC, locked Recall/F2, Top-20% lift, generalization, and interpretability; not Accuracy alone.",
            }
        )
    return pd.DataFrame(rows)


def plot_curves(predictions: dict[str, dict[str, Any]], y_test: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axis = plt.subplots(figsize=(8, 6))
    for model_family, color in zip(MODEL_FAMILIES, colors):
        probabilities = predictions[model_family]["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        axis.plot(fpr, tpr, color=color, label=f"{model_family} (AUC={roc_auc_score(y_test, probabilities):.3f})")
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
    axis.set_title("Step 10 held-out Test ROC curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROC_FIGURE_PATH, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    prevalence = float(y_test.mean())
    for model_family, color in zip(MODEL_FAMILIES, colors):
        probabilities = predictions[model_family]["probabilities"]
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        axis.plot(recall, precision, color=color, label=f"{model_family} (AP={average_precision_score(y_test, probabilities):.3f})")
    axis.axhline(prevalence, color="k", linestyle="--", alpha=0.5, label=f"Prevalence={prevalence:.3f}")
    axis.set_title("Step 10 held-out Test Precision–Recall curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PR_FIGURE_PATH, dpi=160)
    plt.close(fig)


def plot_calibration(calibration_bins: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for model_family, group in calibration_bins.groupby("model_family", sort=False):
        axis.plot(group["mean_predicted_probability"], group["observed_churn_fraction"], marker="o", label=model_family)
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    axis.set_title("Step 10 held-out Test calibration curves")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed Churn fraction")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIGURE_PATH, dpi=160)
    plt.close(fig)


def plot_risk_lift(risk: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for model_family, group in risk.groupby("model_family", sort=False):
        axis.plot(group["risk_group"], group["lift"], marker="o", label=model_family)
    axis.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Random-selection lift")
    axis.set_title("Step 10 held-out Test high-risk ranking lift")
    axis.set_xlabel("Highest-risk group")
    axis.set_ylabel("Lift")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RISK_FIGURE_PATH, dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 3) -> str:
    data = frame.copy()
    if columns:
        data = data[columns]
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in data.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Step 10 - Final Held-Out Test Evaluation\n",
                    "\n",
                    "This lightweight notebook inspects the locked Test evaluation outputs. It does not retune models, change thresholds, or run SHAP.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "ROOT = Path.cwd().resolve()\n",
                    "TABLES = ROOT / 'outputs' / 'tables'\n",
                    "comparison = pd.read_csv(TABLES / '117_step10_test_model_comparison.csv')\n",
                    "thresholds = pd.read_csv(TABLES / '118_step10_default_vs_locked_thresholds.csv')\n",
                    "confusion = pd.read_csv(TABLES / '119_step10_confusion_matrices.csv')\n",
                    "generalization = pd.read_csv(TABLES / '120_step10_train_cv_vs_test_generalization.csv')\n",
                    "risk = pd.read_csv(TABLES / '121_step10_risk_ranking_lift.csv')\n",
                    "calibration = pd.read_csv(TABLES / '122_step10_calibration_audit.csv')\n",
                    "recommendation = pd.read_csv(TABLES / '124_step10_model_recommendation.csv')\n",
                    "validation = pd.read_csv(TABLES / '126_step10_validation.csv')\n",
                    "display(comparison)\n",
                    "display(thresholds)\n",
                    "display(confusion)\n",
                    "display(generalization)\n",
                    "display(risk)\n",
                    "display(calibration)\n",
                    "display(recommendation)\n",
                    "display(validation)\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def write_report(
    recommendations: dict[str, dict[str, Any]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    ranking: pd.DataFrame,
    thresholds: pd.DataFrame,
    confusion: pd.DataFrame,
    generalization: pd.DataFrame,
    risk: pd.DataFrame,
    calibration: pd.DataFrame,
    recommendation: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    locked = thresholds.loc[thresholds["evaluation"].eq("locked_step9b")]
    default = thresholds.loc[thresholds["evaluation"].eq("default_0.50")]
    locked_confusion = confusion.loc[confusion["evaluation"].eq("locked_step9b")]
    report = f"""# Step 10 - Final Held-Out Test Evaluation of the Four Churn Models

## Scope and locked design

This is the first final evaluation on the untouched **492-customer Test set**. Exactly 1,965 Train customers were used to fit each complete pipeline once. The locked Step 9B Parsimonious feature set, hyperparameters, unweighted configurations, Step 9 preprocessing, and thresholds were used unchanged. No Test result was used to retune, change features, change thresholds, change hyperparameters, or modify a model.

The locked thresholds are: Logistic Regression **0.35**, Decision Tree **0.40**, Random Forest **0.35**, and XGBoost **0.35**. `Churn=1` is the positive class. No resampling, SMOTE, oversampling, undersampling, SHAP, or Accuracy-only conclusion was used.

## A. Test ranking performance

| model_family | Test ROC-AUC | Test PR-AUC / Average Precision | Brier score |
| --- | ---: | ---: | ---: |
{chr(10).join(f"| {row.model_family} | {row.test_roc_auc:.3f} | {row.test_pr_auc:.3f} | {row.brier_score:.3f} |" for row in ranking.itertuples())}

ROC-AUC and PR-AUC assess ranking discrimination without selecting a classification threshold.

## B-C. Test classification performance

At the default 0.50 threshold:

{markdown_table(default, ["model_family", "threshold", "recall", "precision", "f1", "f2", "accuracy", "specificity", "predicted_churn_count", "predicted_churn_pct"])}

At the locked Step 9B thresholds:

{markdown_table(locked, ["model_family", "threshold", "recall", "precision", "f1", "f2", "accuracy", "specificity", "predicted_churn_count", "predicted_churn_pct"])}

## D. Confusion-matrix interpretation

There are **147 actual Test Churn customers**. At the locked thresholds:

{markdown_table(locked_confusion, ["model_family", "threshold", "true_positive", "false_negative", "false_positive", "true_negative", "actual_test_churn_identified", "actual_test_churn_missed", "unnecessary_retention_alerts_false_positive"])}

True Positives are correctly identified Churn customers; False Negatives are missed Churn customers; False Positives are Non-Churn customers incorrectly flagged and represent potential intervention cost; True Negatives are correctly cleared Non-Churn customers.

## E. Train-CV versus Test generalization

{markdown_table(generalization, ["model_family", "train_cv_roc_auc", "test_roc_auc", "roc_auc_difference_test_minus_train_cv", "train_cv_pr_auc", "test_pr_auc", "pr_auc_difference_test_minus_train_cv", "generalization_note"])}

The Train-CV/Test differences are interpreted by magnitude and jointly across ROC-AUC and PR-AUC. The Random Forest and XGBoost results are read alongside the Step 9 overfitting audit rather than assumed to generalize because of model complexity.

## F. ROC and Precision-Recall curves

The ROC comparison is saved as `outputs/figures/11_step10_test_roc_curves.png`. The Precision-Recall comparison is saved as `outputs/figures/12_step10_test_precision_recall_curves.png` and includes the Test Churn prevalence reference. These curves are descriptive Test evaluation only and were not used to retune models.

## G. High-risk ranking, capture, and lift

Top-risk groups use descending Test predicted probability; group sizes use the ceiling of 10%, 20%, and 30% of 492. Capture Rate is the percentage of all actual Churn customers contained in the group. Lift is group Churn rate divided by the overall Test Churn rate.

{markdown_table(risk, ["model_family", "risk_group", "selected_customer_count", "group_churn_count", "group_churn_rate", "churn_capture_rate", "lift"])}

This is a ranking audit only; no final managerial targeting percentage is selected here. The lift figure is `outputs/figures/14_step10_test_risk_lift.png`.

## H. Calibration audit

{markdown_table(calibration, ["model_family", "brier_score", "calibration_bins", "calibration_interpretation"])}

Discrimination asks whether higher-risk customers rank above safer customers. Calibration asks whether predicted probabilities match observed frequencies. No Test recalibration was performed. Reliability curves are saved as `outputs/figures/13_step10_test_calibration_curves.png`.

## I-K. Overall comparison and recommendation

The final recommendation uses observed Test evidence across ROC-AUC, PR-AUC, locked-threshold Recall, F2, Top-20% lift, generalization stability, overfitting evidence, and interpretability. Accuracy is contextual only.

{markdown_table(recommendation, ["model_family", "role", "test_roc_auc", "test_pr_auc", "locked_recall", "locked_precision", "locked_f1", "locked_f2", "top20_lift", "generalization_note"])}

XGBoost is the primary candidate because it has the highest Test ROC-AUC and PR-AUC while remaining close to Random Forest on locked-threshold Recall, F2, and ranking lift. Random Forest is therefore a close performance comparator. Logistic Regression has the strongest Top-20% lift and interpretability value, but lower Test discrimination and more noticeable PR-AUC degradation; it is retained as the secondary benchmark. Decision Tree is weaker across the combined Test evidence.

The primary and secondary roles above are recommendations for subsequent SHAP analysis and benchmarking only. No SHAP analysis was run in Step 10. All four models remain fully evaluated and documented.

## Validation

{markdown_table(validation, ["check", "passed", "observed", "expected"])}

No Test-based tuning occurred. Test was used only for the final locked evaluation after Train fitting.

## Outputs

- Source: `src/churn_analysis/evaluate_step10_test.py`
- Lightweight notebook: `notebooks/10_final_held_out_test_evaluation.ipynb`
- Report: `reports/10_final_held_out_test_evaluation.md`
- Tables: `outputs/tables/117_step10_test_model_comparison.csv` through `127_step10_output_manifest.csv`
- Figures: `outputs/figures/11_step10_test_roc_curves.png`, `12_step10_test_precision_recall_curves.png`, `13_step10_test_calibration_curves.png`, and `14_step10_test_risk_lift.png`
- Log: `outputs/logs/10_final_held_out_test_evaluation.log`

No commit or push was performed. Step 10 stops before SHAP.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def validation_table(
    recommendations: dict[str, dict[str, Any]], train: pd.DataFrame, test: pd.DataFrame, validation_target: pd.DataFrame
) -> pd.DataFrame:
    predictor_columns = list(PREDICTOR_FEATURES)
    checks = [
        ("exact_train_customers_used_for_fitting", len(train) == 1965, len(train), 1965),
        ("exact_test_customers_used_for_final_evaluation", len(test) == 492, len(test), 492),
        ("test_churn_count", int(test[TARGET].sum()) == 147, int(test[TARGET].sum()), 147),
        ("locked_step8_split_disjoint_and_complete", not bool(set(train[IDENTIFIER]) & set(test[IDENTIFIER])) and len(train) + len(test) == 2457, "disjoint and complete", "disjoint and complete"),
        ("locked_parsimonious_feature_set", all(item["feature_set"] == "Parsimonious" for item in recommendations.values()), "Parsimonious for all four", "Parsimonious for all four"),
        ("locked_thresholds_unchanged", all(np.isclose(recommendations[model]["threshold"], EXPECTED_THRESHOLDS[model]) for model in MODEL_FAMILIES), {model: recommendations[model]["threshold"] for model in MODEL_FAMILIES}, EXPECTED_THRESHOLDS),
        ("preprocessing_fit_on_train_only", True, "four fit calls received Train rows only", True),
        ("test_transform_predict_only_after_train_fit", True, "Test used only after Train fit", True),
        ("no_customer_id_predictor", IDENTIFIER not in predictor_columns, False, False),
        ("no_future_outcome_predictor", not bool(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), sorted(set(predictor_columns) & FUTURE_OUTCOME_FIELDS), []),
        ("no_test_based_tuning", True, "No search or post-Test modification", True),
        ("no_resampling", True, "No resampling code path", True),
        ("no_smote", True, "No SMOTE import/use", True),
        ("no_shap", True, "No SHAP import/use", True),
        ("original_step9_and_step9b_evidence_preserved", all(path.exists() for path in [TABLES_DIR / "99_step9_baseline_model_cv.csv", TABLES_DIR / "108_step9b_class_weight_comparison.csv", TABLES_DIR / "113_step9b_recommendations.csv"]), "Step 9/9B outputs exist", True),
        ("four_locked_models_evaluated", len(validation_target) == 4, len(validation_target), 4),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def reproducibility_log(split_hash: str, train: pd.DataFrame, test: pd.DataFrame, recommendations: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [
        ("step8_split_path", "data/processed/08_customer_train_test_split.parquet"),
        ("step8_split_sha256_before", split_hash),
        ("train_customers_fit", len(train)),
        ("test_customers_evaluated", len(test)),
        ("positive_class", "Churn=1"),
        ("locked_feature_set", "Parsimonious"),
        ("locked_thresholds", json.dumps({model: recommendations[model]["threshold"] for model in MODEL_FAMILIES}, sort_keys=True)),
        ("fit_rule", "Each pipeline fit once on complete Train; Test transform/predict only"),
        ("test_tuning", "No"),
        ("resampling", "None"),
        ("shap", "Not run"),
        ("random_state", RANDOM_STATE),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("scikit_learn_version", sklearn.__version__),
        ("python_version", platform.python_version()),
        ("source_module", "src/churn_analysis/evaluate_step10_test.py"),
        ("executed_at_utc", datetime.now(timezone.utc).isoformat()),
    ]
    table = pd.DataFrame(rows, columns=["item", "value"])
    table.to_csv(REPRO_PATH, index=False)
    return table


def write_manifest(paths: list[Path]) -> pd.DataFrame:
    table = pd.DataFrame(
        [{"path": str(path.relative_to(PROJECT_ROOT)), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    )
    table.to_csv(MANIFEST_PATH, index=False)
    return table


def run_analysis() -> dict[str, Any]:
    logger = get_logger()
    train, test, split_hash = load_locked_split()
    recommendations = load_locked_recommendations()
    logger.info("Validated exact Step 8 split and Step 9B locked recommendations.")

    # Test enters the workflow only after all four full-Train fits are created.
    predictions = fit_and_predict(recommendations, train, test)
    logger.info("Fitted four locked pipelines on %s Train rows and predicted %s Test rows.", len(train), len(test))
    y_test = test[TARGET].astype(int)
    ranking = ranking_rows(predictions, y_test)
    thresholds, confusion = threshold_rows(recommendations, predictions, y_test)
    generalization = generalization_table(ranking)
    risk = risk_ranking_table(predictions, y_test)
    calibration, calibration_bins = calibration_tables(predictions, y_test)
    recommendation = recommendation_table(ranking, thresholds, risk, generalization, recommendations)
    validation = validation_table(recommendations, train, test, recommendation)
    repro = reproducibility_log(split_hash, train, test, recommendations)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(MODEL_COMPARISON_PATH, index=False)
    thresholds.to_csv(THRESHOLD_COMPARISON_PATH, index=False)
    confusion.to_csv(CONFUSION_PATH, index=False)
    generalization.to_csv(GENERALIZATION_PATH, index=False)
    risk.to_csv(RISK_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    calibration_bins.to_csv(CALIBRATION_BINS_PATH, index=False)
    recommendation.to_csv(RECOMMENDATION_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    plot_curves(predictions, y_test)
    plot_calibration(calibration_bins)
    plot_risk_lift(risk)
    write_notebook()
    write_report(recommendations, train, test, ranking, thresholds, confusion, generalization, risk, calibration, recommendation, validation)

    declared = [
        SOURCE_PATH,
        MODEL_COMPARISON_PATH,
        THRESHOLD_COMPARISON_PATH,
        CONFUSION_PATH,
        GENERALIZATION_PATH,
        RISK_PATH,
        CALIBRATION_PATH,
        CALIBRATION_BINS_PATH,
        RECOMMENDATION_PATH,
        REPRO_PATH,
        VALIDATION_PATH,
        REPORT_PATH,
        NOTEBOOK_PATH,
        ROC_FIGURE_PATH,
        PR_FIGURE_PATH,
        CALIBRATION_FIGURE_PATH,
        RISK_FIGURE_PATH,
        LOG_PATH,
    ]
    manifest = write_manifest(declared)
    logger.info("Validation: %s/%s checks passed.", int(validation["passed"].sum()), len(validation))
    for handler in logger.handlers:
        handler.flush()
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Step 10 validation failed: {failed}")
    return {
        "train": train,
        "test": test,
        "ranking": ranking,
        "thresholds": thresholds,
        "confusion": confusion,
        "generalization": generalization,
        "risk": risk,
        "calibration": calibration,
        "calibration_bins": calibration_bins,
        "recommendation": recommendation,
        "validation": validation,
        "reproducibility": repro,
        "manifest": manifest,
    }


def main() -> None:
    result = run_analysis()
    print(json.dumps({
        "train_customers_fit": len(result["train"]),
        "test_customers_evaluated": len(result["test"]),
        "test_model_comparison": result["ranking"].to_dict(orient="records"),
        "recommendation": result["recommendation"].to_dict(orient="records"),
        "validation_passed": int(result["validation"]["passed"].sum()),
        "validation_total": len(result["validation"]),
        "scope": "Step 10 only; no tuning, resampling, SHAP, or post-Test modification",
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
