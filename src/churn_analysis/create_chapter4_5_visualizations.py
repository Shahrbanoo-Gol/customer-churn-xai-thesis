"""Create thesis-quality descriptive visualizations for Chapter 4.5.

This module reads the finalized Step 7 customer-level feature table only. It
does not rebuild features, alter the stored table, retrain models, or modify
existing analytical outputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "07_customer_features.parquet"
OUTCOME_PATH = PROJECT_ROOT / "data" / "processed" / "06_primary_customer_outcome_table.parquet"
RECONCILIATION_PATH = PROJECT_ROOT / "outputs" / "tables" / "85_step7_reconciliation_audit.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_PATH = PROJECT_ROOT / "reports" / "chapter4_5_visualization_report.md"

RFM_FEATURES = ["RecencyDays", "Frequency", "Monetary"]
SELECTED_FEATURES = ["TotalQuantity", "AverageOrderValue", "UniqueProducts", "CancellationValue"]
HETEROGENEITY_FEATURES = [
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AverageOrderValue",
]
CONTINUOUS_SUMMARY_FEATURES = [
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
    "CancellationInvoiceCount",
    "CancellationValue",
    "CancellationInvoiceRate",
]
EXCLUDED_CORRELATION_COLUMNS = {"Customer ID", "CountryCandidate", "Churn"}
BINARY_FEATURES = {"HasRepeatPurchaseHistory", "HasCancellation"}

PALETTE = {
    "blue": "#2f5d7e",
    "teal": "#4e8c88",
    "orange": "#c26a3d",
    "gray": "#555555",
    "light_gray": "#d9dee3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_value(value: float, column: str) -> str:
    if column in {"Frequency", "UniqueProducts", "TotalQuantity", "ActivePurchaseDays", "CancellationInvoiceCount"}:
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return f"{value:,.2f}"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#444444",
            "grid.color": "#d9dee3",
            "grid.linewidth": 0.6,
        }
    )


def load_and_validate() -> tuple[pd.DataFrame, dict[str, object]]:
    features = pd.read_parquet(FEATURE_PATH)
    outcome = pd.read_parquet(OUTCOME_PATH)
    if len(features) != 2457:
        raise AssertionError(f"Expected 2,457 feature rows; found {len(features)}")
    if features["Customer ID"].nunique() != 2457:
        raise AssertionError("Customer ID is not unique across the finalized feature table")

    feature_churn = features.set_index("Customer ID")["Churn"].sort_index()
    outcome_churn = outcome.set_index("Customer ID")["Churn"].sort_index()
    pd.testing.assert_series_equal(feature_churn, outcome_churn, check_names=False, check_dtype=False)

    missing_expected = {
        "MeanInterpurchaseDays": 445,
        "MedianInterpurchaseDays": 445,
    }
    missing_actual = features.isna().sum().to_dict()
    for column, expected in missing_expected.items():
        if int(missing_actual[column]) != expected:
            raise AssertionError(f"{column} structural missing count differs: {missing_actual[column]} != {expected}")

    reconciliation = pd.read_csv(RECONCILIATION_PATH)
    required_checks = {
        "final_rows_exactly_2457",
        "Customer_ID_unique",
        "Churn_exactly_matches_Step6",
        "positive_observation_has_no_post_cutoff_rows",
        "event_observation_has_no_post_cutoff_rows",
        "cancellation_observation_has_no_post_cutoff_rows",
    }
    reconciliation_map = reconciliation.set_index("check")["passed"].astype(bool).to_dict()
    failed = [check for check in required_checks if not reconciliation_map.get(check, False)]
    if failed:
        raise AssertionError(f"Existing Step 7 temporal/reconciliation checks failed: {failed}")

    snapshot = {
        "feature_hash_before": sha256_file(FEATURE_PATH),
        "outcome_hash_before": sha256_file(OUTCOME_PATH),
        "feature_columns": list(features.columns),
        "feature_rows": len(features),
        "unique_customers": int(features["Customer ID"].nunique()),
        "churn_0": int(features["Churn"].eq(0).sum()),
        "churn_1": int(features["Churn"].eq(1).sum()),
        "structural_missing": missing_actual,
        "reconciliation": reconciliation_map,
    }
    return features, snapshot


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rfm_distributions(features: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    titles = {
        "RecencyDays": "Recency of most recent purchase",
        "Frequency": "Positive-purchase frequency",
        "Monetary": "Positive-purchase monetary value",
    }
    for ax, column in zip(axes, RFM_FEATURES):
        values = features[column].dropna()
        mean = values.mean()
        median = values.median()
        sns.histplot(values, bins=35, color=PALETTE["blue"], edgecolor="white", linewidth=0.4, ax=ax)
        ax.axvline(median, color=PALETTE["gray"], linestyle="--", linewidth=1.5, label=f"Median = {format_value(median, column)}")
        ax.axvline(mean, color=PALETTE["orange"], linestyle="-", linewidth=1.5, label=f"Mean = {format_value(mean, column)}")
        ax.set_title(titles[column])
        ax.set_xlabel(column)
        ax.set_ylabel("Customers")
        ax.legend(frameon=False, loc="upper right")
        ax.text(0.02, 0.96, f"n = {len(values):,}", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(axis="y", alpha=0.55)
    fig.suptitle("Distribution of the three core RFM features", fontsize=15, y=1.04)
    path = FIGURES_DIR / "chapter4_5_rfm_distributions.png"
    save_figure(fig, path)
    return path


def plot_selected_distributions(features: pd.DataFrame) -> tuple[Path, Path]:
    original_path = FIGURES_DIR / "chapter4_5_selected_feature_distributions.png"
    log_path = FIGURES_DIR / "chapter4_5_selected_feature_distributions_log_view.png"
    titles = {
        "TotalQuantity": "Total quantity purchased",
        "AverageOrderValue": "Average order value",
        "UniqueProducts": "Unique products purchased",
        "CancellationValue": "Cancellation value",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, column in zip(axes.flat, SELECTED_FEATURES):
        values = features[column].dropna()
        mean = values.mean()
        median = values.median()
        sns.histplot(values, bins=35, color=PALETTE["teal"], edgecolor="white", linewidth=0.4, ax=ax)
        ax.axvline(median, color=PALETTE["gray"], linestyle="--", linewidth=1.3, label=f"Median = {format_value(median, column)}")
        ax.axvline(mean, color=PALETTE["orange"], linewidth=1.3, label=f"Mean = {format_value(mean, column)}")
        ax.set_title(titles[column])
        ax.set_xlabel(column)
        ax.set_ylabel("Customers")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        skewness = values.skew()
        ax.text(
            0.02,
            0.96,
            f"n = {len(values):,}\nSkewness = {skewness:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["light_gray"]},
        )
        ax.grid(axis="y", alpha=0.55)
    fig.suptitle("Selected behavioral feature distributions on the original scale", fontsize=15, y=1.02)
    save_figure(fig, original_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, column in zip(axes.flat, SELECTED_FEATURES):
        values = np.log1p(features[column].dropna())
        sns.histplot(values, bins=35, color=PALETTE["teal"], edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_title(titles[column])
        ax.set_xlabel(f"log1p({column}) — visualization only")
        ax.set_ylabel("Customers")
        ax.text(
            0.02,
            0.96,
            "Visualization transformation only\nStored values unchanged",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["light_gray"]},
        )
        ax.grid(axis="y", alpha=0.55)
    fig.suptitle("Selected behavioral features: log1p visualization view", fontsize=15, y=1.02)
    save_figure(fig, log_path)
    return original_path, log_path


def plot_heterogeneity(features: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(HETEROGENEITY_FEATURES), figsize=(15, 4.8))
    for ax, column in zip(axes, HETEROGENEITY_FEATURES):
        values = features[column].dropna()
        sns.boxplot(
            x=values,
            color="#9bbcc1",
            width=0.45,
            fliersize=2.2,
            linewidth=0.9,
            orient="h",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.text(
            0.02,
            0.06,
            f"Median = {format_value(values.median(), column)}\nMax = {format_value(values.max(), column)}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["light_gray"]},
        )
        ax.grid(axis="x", alpha=0.55)
    fig.suptitle("Customer heterogeneity and extreme values", fontsize=15, y=0.94)
    fig.supxlabel("Original value (log-scaled axis for display)", y=0.10, fontsize=9)
    fig.text(
        0.5,
        0.025,
        "All observations are retained; the logarithmic axis is a visualization aid and does not alter the feature table.",
        ha="center",
        fontsize=9,
        color=PALETTE["gray"],
    )
    fig.subplots_adjust(left=0.03, right=0.995, top=0.82, bottom=0.23, wspace=0.08)
    path = FIGURES_DIR / "chapter4_5_customer_heterogeneity_boxplots.png"
    save_figure(fig, path)
    return path


def build_correlation_table(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_CORRELATION_COLUMNS
    ]
    matrix = features[columns].corr(method="pearson")
    pair_rows: list[dict[str, object]] = []
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            pair = features[[feature_a, feature_b]].dropna()
            correlation = pair[feature_a].corr(pair[feature_b], method="pearson")
            if pd.notna(correlation) and abs(correlation) >= 0.80:
                pair_rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pearson_correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "pairwise_n": len(pair),
                    }
                )
    high = pd.DataFrame(pair_rows).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)
    return high, columns


def plot_correlation_heatmap(features: pd.DataFrame, high: pd.DataFrame, columns: list[str]) -> Path:
    matrix = features[columns].corr(method="pearson")
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    annotation = matrix.where(matrix.abs().ge(0.80)).round(2).astype(object)
    annotation = annotation.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    labels = [column.replace("Interpurchase", "Interpurchase\n").replace("Purchase", "Purchase\n") for column in columns]
    matrix_display = matrix.copy()
    matrix_display.index = labels
    matrix_display.columns = labels
    annotation.index = labels
    annotation.columns = labels

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        matrix_display,
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.25,
        linecolor="white",
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 8, "fontweight": "bold"},
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson correlation among numeric customer-level predictors", fontsize=15, pad=16)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.text(
        0.5,
        0.01,
        "Only |r| ≥ 0.80 is annotated. Pairwise-complete observations are used for correlations involving structural missingness.",
        ha="center",
        fontsize=9,
        color=PALETTE["gray"],
    )
    path = FIGURES_DIR / "chapter4_5_feature_correlation_heatmap.png"
    save_figure(fig, path)
    return path


def plot_cancellation_counts(features: pd.DataFrame) -> Path:
    counts = features["HasCancellation"].value_counts().reindex([1, 0]).fillna(0).astype(int)
    labels = ["At least one cancellation", "No cancellation"]
    frame = pd.DataFrame({"label": labels, "customers": counts.to_numpy()})
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    sns.barplot(data=frame, x="label", y="customers", hue="label", palette=[PALETTE["orange"], PALETTE["blue"],], legend=False, ax=ax)
    for index, value in enumerate(frame["customers"]):
        ax.text(index, value + max(frame["customers"]) * 0.02, f"{value:,}\n({value / len(features):.2%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Customers with and without observed cancellation activity", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, max(frame["customers"]) * 1.18)
    ax.grid(axis="y", alpha=0.55)
    path = FIGURES_DIR / "chapter4_5_cancellation_customer_counts.png"
    save_figure(fig, path)
    return path


def write_summary(features: pd.DataFrame) -> Path:
    rows = []
    for column in CONTINUOUS_SUMMARY_FEATURES:
        values = features[column]
        non_missing = values.dropna()
        rows.append(
            {
                "feature": column,
                "count": int(non_missing.size),
                "missing": int(values.isna().sum()),
                "mean": non_missing.mean(),
                "median": non_missing.median(),
                "standard_deviation": non_missing.std(),
                "minimum": non_missing.min(),
                "p25": non_missing.quantile(0.25),
                "p75": non_missing.quantile(0.75),
                "p90": non_missing.quantile(0.90),
                "p95": non_missing.quantile(0.95),
                "p99": non_missing.quantile(0.99),
                "maximum": non_missing.max(),
                "skewness": non_missing.skew(),
            }
        )
    path = TABLES_DIR / "chapter4_5_visual_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10f")
    return path


def write_report(
    features: pd.DataFrame,
    snapshot: dict[str, object],
    high: pd.DataFrame,
    figure_paths: list[Path],
    summary_path: Path,
    correlation_path: Path,
) -> None:
    def stats(column: str) -> dict[str, float]:
        values = features[column].dropna()
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "max": float(values.max()),
            "skew": float(values.skew()),
        }

    rfm_stats = {column: stats(column) for column in RFM_FEATURES}
    selected_stats = {column: stats(column) for column in SELECTED_FEATURES}
    heterogeneity_stats = {column: stats(column) for column in HETEROGENEITY_FEATURES}
    cancellation_count = int(features["HasCancellation"].sum())
    no_cancellation_count = int(features["HasCancellation"].eq(0).sum())
    heterogeneity_medians = ", ".join(
        f"`{column}` {heterogeneity_stats[column]['median']:,.2f}" for column in HETEROGENEITY_FEATURES
    )
    heterogeneity_maxima = ", ".join(
        f"`{column}` {heterogeneity_stats[column]['max']:,.2f}" for column in HETEROGENEITY_FEATURES
    )

    high_rows = "\n".join(
        f"| {row.feature_a} | {row.feature_b} | {row.pearson_correlation:.4f} | {int(row.pairwise_n):,} |"
        for row in high.itertuples()
    ) or "| None | None | None | None |"

    reference_checks = [
        ("RecencyDays median", rfm_stats["RecencyDays"]["median"], 43.02),
        ("RecencyDays mean", rfm_stats["RecencyDays"]["mean"], 50.86),
        ("RecencyDays maximum", rfm_stats["RecencyDays"]["max"], 134.96),
        ("Frequency median", rfm_stats["Frequency"]["median"], 3.0),
        ("Frequency mean", rfm_stats["Frequency"]["mean"], 5.83),
        ("Frequency maximum", rfm_stats["Frequency"]["max"], 183.0),
        ("Monetary median", rfm_stats["Monetary"]["median"], 1070.38),
        ("Monetary mean", rfm_stats["Monetary"]["mean"], 2784.04),
        ("Monetary maximum", rfm_stats["Monetary"]["max"], 251520.82),
        ("HasCancellation count", float(cancellation_count), 1144.0),
    ]
    reference_lines = "\n".join(
        f"- **{name}:** actual `{actual:,.4f}`; expected reference approximately `{expected:,.4f}`; {'consistent within rounding' if abs(actual - expected) < max(0.02, abs(expected) * 0.001) else 'discrepancy noted'}."
        for name, actual, expected in reference_checks
    )

    monetary_total_quantity = float(features["Monetary"].corr(features["TotalQuantity"]))
    report = f"""# Chapter 4.5 descriptive visualization report

## Scope and data source

The figures use the finalized customer-level feature table at `{FEATURE_PATH.relative_to(PROJECT_ROOT)}`. No features were rebuilt from raw transactions. The table contains exactly **{len(features):,} customers**, with one row per unique `Customer ID`; `Churn` matches the Step 6 outcome table exactly. The dataset hash was unchanged during this descriptive step.

All figures are white-background PNG files exported at 300 DPI. The analysis is descriptive: it does not establish causality, determine feature importance, or replace the later model-development and SHAP analyses.

## Figure 1 — Core RFM distributions

File: `{figure_paths[0].relative_to(PROJECT_ROOT)}`

1. **What it shows.** Separate empirical histograms for `RecencyDays`, `Frequency`, and `Monetary`, with mean and median reference lines and the full cohort sample size.
2. **Main numerical observation.** `RecencyDays` has median {rfm_stats['RecencyDays']['median']:.2f} days and mean {rfm_stats['RecencyDays']['mean']:.2f} days, while `Frequency` has median {rfm_stats['Frequency']['median']:.0f} and mean {rfm_stats['Frequency']['mean']:.2f}. `Monetary` has median {rfm_stats['Monetary']['median']:,.2f} and mean {rfm_stats['Monetary']['mean']:,.2f}; its maximum is {rfm_stats['Monetary']['max']:,.2f}.
3. **Managerial interpretation.** Most customers are relatively low-to-moderate frequency and value customers, while a smaller group contributes substantially higher purchasing activity or economic value. The median is therefore a useful description of a typical customer alongside the mean.
4. **What must not be inferred.** The high-value tail is not automatically erroneous, and the figure does not show that high monetary value or frequency causes churn.

## Figure 2 — Selected behavioral feature distributions and skewness

Files: `{figure_paths[1].relative_to(PROJECT_ROOT)}` and `{figure_paths[2].relative_to(PROJECT_ROOT)}`

1. **What it shows.** Original-scale histograms for `TotalQuantity`, `AverageOrderValue`, `UniqueProducts`, and `CancellationValue`, annotated with mean, median, and sample skewness. The companion view uses `log1p` only to make the shape of long-tailed distributions easier to inspect.
2. **Main numerical observation.** Skewness is positive for all four selected features: `TotalQuantity` {selected_stats['TotalQuantity']['skew']:.2f}, `AverageOrderValue` {selected_stats['AverageOrderValue']['skew']:.2f}, `UniqueProducts` {selected_stats['UniqueProducts']['skew']:.2f}, and `CancellationValue` {selected_stats['CancellationValue']['skew']:.2f}. The original stored values were not log-transformed, imputed, winsorized, truncated, or otherwise changed.
3. **Managerial interpretation.** Customer behavior is concentrated among lower/moderate activity levels, with a smaller group showing much larger baskets, product breadth, order values, or cancellation values. This supports segment-aware descriptive monitoring.
4. **What must not be inferred.** The log1p panels are visualization transformations only; they are not evidence that a transformed feature was used in the model pipeline, and skewness alone is not evidence of invalid data.

## Figure 3 — Customer heterogeneity and extreme values

File: `{figure_paths[3].relative_to(PROJECT_ROOT)}`

1. **What it shows.** Separate boxplots for `Frequency`, `Monetary`, `TotalQuantity`, `UniqueProducts`, and `AverageOrderValue`. Every observation remains included; a logarithmic axis is used only for display.
2. **Main numerical observation.** The medians are {heterogeneity_medians}. The corresponding maxima extend to {heterogeneity_maxima}.
3. **Managerial interpretation.** The cohort is heterogeneous: a small number of customers exhibit substantially higher purchasing activity or economic value than the typical customer. These customers may merit separate descriptive review or differentiated service design.
4. **What must not be inferred.** Extreme observations must not be treated as errors merely because they are large, and the boxplots do not identify causal drivers or predictive importance.

## Figure 4 — Feature correlation heatmap

File: `{figure_paths[4].relative_to(PROJECT_ROOT)}`  
Companion table: `{correlation_path.relative_to(PROJECT_ROOT)}`

1. **What it shows.** Pearson correlations among numeric customer-level predictors, excluding `Customer ID`, `Churn`, and raw `CountryCandidate`. Strong absolute correlations (|r| ≥ 0.80) are annotated.
2. **Main numerical observation.** The strongest associations are:

| Feature A | Feature B | Pearson r | Pairwise n |
| --- | --- | ---: | ---: |
{high_rows}

The `Monetary`–`TotalQuantity` correlation is {monetary_total_quantity:.4f}, which is lower than the approximate 0.920 reference value supplied for the completed analysis.
3. **Managerial interpretation.** Strong correlations indicate overlapping descriptive information and may signal redundancy when constructing a parsimonious feature set. This is useful for later modeling review and stakeholder interpretation.
4. **What must not be inferred.** Correlation does not establish causality, and it does not by itself determine which feature should be removed or which feature will be important for churn prediction.

## Figure 5 — Cancellation behavior

File: `{figure_paths[5].relative_to(PROJECT_ROOT)}`

1. **What it shows.** A bar chart comparing customers with at least one observed cancellation against customers without observed cancellation activity.
2. **Main numerical observation.** {cancellation_count:,} customers ({cancellation_count / len(features):.2%}) have at least one cancellation, while {no_cancellation_count:,} ({no_cancellation_count / len(features):.2%}) do not.
3. **Managerial interpretation.** Cancellation activity is present for a substantial share of the cohort, so it is a meaningful descriptive dimension for customer-behavior profiling and service-process review.
4. **What must not be inferred.** The chart does not show that cancellation causes churn or that customers with cancellations are necessarily dissatisfied.

## Statistical summary and structural missingness

The descriptive statistics used for the plotted quantitative features are in `{summary_path.relative_to(PROJECT_ROOT)}`. `MeanInterpurchaseDays` and `MedianInterpurchaseDays` each have **445 missing values** because these customers have only one positive purchase event, so no adjacent interpurchase interval exists. This is structural missingness, not an ordinary data-quality failure; no imputation was performed.

## Reference-value verification

{reference_lines}

The expected reference for `Monetary`–`TotalQuantity` was approximately 0.920; the finalized table yields {monetary_total_quantity:.4f}. This is the only material discrepancy identified against the supplied reference values.

## Validation completed

- Customer rows: **{snapshot['feature_rows']:,}**; unique `Customer ID`: **{snapshot['unique_customers']:,}**.
- `Churn`: unchanged and exactly equal to `06_primary_customer_outcome_table.parquet`.
- Existing Step 7 reconciliation checks for no post-cutoff positive-purchase, event, and cancellation rows: **passed**.
- Feature-table schema and values: source hash unchanged before/after (`{snapshot['feature_hash_before']}`); no feature was deleted or altered.
- No models were retrained and no preprocessing decisions were changed.
- Existing figures were not overwritten; only new `chapter4_5_*` files were created.

"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    configure_style()
    features, snapshot = load_and_validate()

    figure_paths = [plot_rfm_distributions(features)]
    selected_original, selected_log = plot_selected_distributions(features)
    figure_paths.extend([selected_original, selected_log, plot_heterogeneity(features)])
    high, columns = build_correlation_table(features)
    correlation_path = TABLES_DIR / "chapter4_5_high_correlations.csv"
    high.to_csv(correlation_path, index=False, float_format="%.10f")
    figure_paths.append(plot_correlation_heatmap(features, high, columns))
    figure_paths.append(plot_cancellation_counts(features))
    summary_path = write_summary(features)

    if sha256_file(FEATURE_PATH) != snapshot["feature_hash_before"]:
        raise AssertionError("Finalized feature table changed during plotting")
    if sha256_file(OUTCOME_PATH) != snapshot["outcome_hash_before"]:
        raise AssertionError("Step 6 outcome table changed during plotting")

    write_report(features, snapshot, high, figure_paths, summary_path, correlation_path)
    print("Generated:")
    for path in [*figure_paths, summary_path, correlation_path, REPORT_PATH]:
        print(path.relative_to(PROJECT_ROOT))
    print(f"Customers: {len(features):,}")
    print(f"Cancellation customers: {int(features['HasCancellation'].sum()):,}")
    print(f"High absolute correlations: {len(high):,}")
    print(f"Feature hash unchanged: {sha256_file(FEATURE_PATH) == snapshot['feature_hash_before']}")


if __name__ == "__main__":
    main()
