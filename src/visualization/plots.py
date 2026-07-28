"""Plotting Utilities for LLM Fingerprinting Model Evaluation.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Model Evaluation & Comparison
Description:
    Reusable plotting functions for:
        - Confusion matrices (matplotlib + seaborn heatmap)
        - ROC curves (one-vs-rest, multi-class)
        - Feature importance bar charts
        - Model comparison bar/radar charts (plotly)
        - Feature distribution plots

All figures are saved to the figures/ directory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for notebooks & scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE: List[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
FIGURE_DPI: int = 150
FIGURE_SIZE_DEFAULT: Tuple[float, float] = (10.0, 7.0)


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save a matplotlib figure and close it.

    Args:
        fig: matplotlib Figure to save.
        out_path: Destination file path (.png or .pdf).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", out_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Integer confusion matrix array (n_classes × n_classes).
        class_names: Ordered list of class label strings.
        title: Figure title.
        out_path: Destination .png file path.
        normalize: If True, normalise rows to percentages.

    Returns:
        matplotlib Figure object.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt_str = ".2%"
    else:
        cm_plot = cm.astype(float)
        fmt_str = ".0f"

    n = len(class_names)
    fig_size = (max(8, n * 0.9), max(6, n * 0.8))
    fig, ax = plt.subplots(figsize=fig_size)

    if sns is not None:
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt_str,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(n)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_figure(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> plt.Figure:
    """Plot one-vs-rest ROC curves for all classes.

    Args:
        y_test: True integer label array.
        y_proba: Class probability matrix (n_samples × n_classes).
        class_names: Ordered list of class label strings.
        title: Figure title.
        out_path: Destination .png file path.

    Returns:
        matplotlib Figure object.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    for i, (cls_name, color) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{cls_name} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str,
    out_path: Path,
    top_n: int = 30,
) -> plt.Figure:
    """Plot a horizontal bar chart of top feature importances.

    Args:
        importances: 1-D array of importance scores.
        feature_names: Corresponding feature name strings.
        title: Figure title.
        out_path: Destination .png file path.
        top_n: Number of top features to display.

    Returns:
        matplotlib Figure object.
    """
    idx = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top_names))]
    ax.barh(top_names, top_vals, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
) -> plt.Figure:
    """Plot a grouped bar chart comparing models on a single metric.

    Args:
        comparison_df: DataFrame with columns [model_name, feature_set, <metric>].
        metric: Column name of the metric to plot.
        title: Figure title.
        out_path: Destination .png file path.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = comparison_df.apply(
        lambda r: f"{r['model_name']}\n({r['feature_set']})", axis=1
    )
    values = comparison_df[metric].values
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_figure(fig, out_path)
    return fig


def plot_feature_comparison_heatmap(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    title: str,
    out_path: Path,
) -> plt.Figure:
    """Plot a heatmap of metric × model/feature-set pairs.

    Args:
        comparison_df: DataFrame with model_name, feature_set, and metric columns.
        metrics: List of metric column names to include.
        title: Figure title.
        out_path: Destination .png file path.

    Returns:
        matplotlib Figure object.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    pivot = comparison_df.copy()
    pivot["label"] = (
        pivot["model_name"].str.replace("_", " ").str.title()
        + " / "
        + pivot["feature_set"]
    )
    pivot = pivot.set_index("label")[metrics]

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.8 + 3, len(pivot) * 0.7 + 2))
    if sns is not None:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            vmin=0,
            vmax=1,
        )
    else:
        im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_figure(fig, out_path)
    return fig
