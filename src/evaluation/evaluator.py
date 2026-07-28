"""Model Evaluation Engine.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Model Evaluation
Description:
    Computes a comprehensive set of classification metrics, generates
    evaluation artefacts (confusion matrices, ROC curves, classification
    reports), and produces cross-model comparison tables.

Metrics computed:
    - Accuracy
    - Precision (macro, weighted)
    - Recall (macro, weighted)
    - F1 Score (macro, weighted, per-class)
    - ROC-AUC (one-vs-rest, macro)
    - Training time, prediction time, memory usage
"""

import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Computes and stores evaluation metrics for a single model run.

    Attributes:
        model_name: Name of the evaluated model.
        feature_set: Name of the feature set used.
        classes: Array of class name strings.
        results_: Dict of computed metric values (populated after evaluate()).
    """

    def __init__(
        self,
        model_name: str,
        feature_set: str,
        classes: np.ndarray,
    ) -> None:
        """Initialise ModelEvaluator.

        Args:
            model_name: Human-readable model identifier.
            feature_set: Name of the feature set (tfidf, char, style, emb).
            classes: Array of class label strings.
        """
        self.model_name = model_name
        self.feature_set = feature_set
        self.classes = classes
        self.results_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Primary evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        estimator: Any,
        X_test: Any,
        y_test: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        train_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute all classification metrics.

        Args:
            estimator: Fitted sklearn (or compatible) estimator.
            X_test: Test feature matrix.
            y_test: True integer label array.
            y_pred: Optional precomputed predictions (computed if None).
            y_proba: Optional precomputed class probabilities.
            train_time: Training wall-clock time in seconds.

        Returns:
            Dict of metric name → value.
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )

        # Prediction timing
        t0 = time.perf_counter()
        if y_pred is None:
            y_pred = estimator.predict(X_test)
        pred_time = time.perf_counter() - t0

        # Memory profiling
        tracemalloc.start()
        _ = estimator.predict(X_test[:1])
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Per-class F1
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        # ROC-AUC (requires probabilities)
        roc_auc: Optional[float] = None
        if y_proba is None and hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)
        if y_proba is not None and len(np.unique(y_test)) > 1:
            try:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
            except Exception:
                roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report_str = classification_report(
            y_test, y_pred,
            target_names=self.classes,
            zero_division=0,
        )

        self.results_ = {
            "model_name": self.model_name,
            "feature_set": self.feature_set,
            "accuracy": round(acc, 6),
            "precision_macro": round(prec_macro, 6),
            "precision_weighted": round(prec_weighted, 6),
            "recall_macro": round(rec_macro, 6),
            "recall_weighted": round(rec_weighted, 6),
            "f1_macro": round(f1_macro, 6),
            "f1_weighted": round(f1_weighted, 6),
            "f1_per_class": {
                cls: round(float(f), 6)
                for cls, f in zip(self.classes, f1_per_class)
            },
            "roc_auc_macro": round(roc_auc, 6) if roc_auc is not None else None,
            "train_time_s": round(train_time, 4),
            "pred_time_s": round(pred_time, 4),
            "peak_memory_mb": round(peak_mem / 1024 / 1024, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
            "y_pred": y_pred,
            "y_test": y_test,
            "y_proba": y_proba,
        }

        logger.info(
            "[%s | %s] acc=%.4f  f1_macro=%.4f  f1_weighted=%.4f  pred_time=%.3fs",
            self.model_name, self.feature_set, acc, f1_macro, f1_weighted, pred_time,
        )
        return self.results_

    def save_classification_report(self, out_path: Path) -> None:
        """Write the classification report string to a text file.

        Args:
            out_path: Target .txt file path.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.results_.get("classification_report", "No report available.")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"Model: {self.model_name}\n")
            fh.write(f"Feature Set: {self.feature_set}\n\n")
            fh.write(report)
        logger.info("Classification report saved → %s", out_path)

    def to_series(self) -> pd.Series:
        """Return scalar metrics as a pandas Series (excludes arrays).

        Returns:
            pandas Series with one entry per scalar metric.
        """
        exclude_keys = {"confusion_matrix", "classification_report",
                        "y_pred", "y_test", "y_proba", "f1_per_class"}
        return pd.Series({
            k: v for k, v in self.results_.items()
            if k not in exclude_keys
        })


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

class ModelComparator:
    """Aggregates and ranks results across multiple ModelEvaluator instances.

    Attributes:
        evaluators: List of ModelEvaluator instances with completed results.
    """

    def __init__(self) -> None:
        """Initialise with an empty evaluator list."""
        self.evaluators: List[ModelEvaluator] = []

    def add(self, evaluator: ModelEvaluator) -> None:
        """Register a completed ModelEvaluator.

        Args:
            evaluator: ModelEvaluator with results_ populated.
        """
        self.evaluators.append(evaluator)
        logger.info(
            "Registered [%s | %s] for comparison.",
            evaluator.model_name, evaluator.feature_set,
        )

    def comparison_table(self) -> pd.DataFrame:
        """Build a comprehensive comparison DataFrame across all models.

        Returns:
            DataFrame sorted by f1_macro descending.
        """
        rows = [ev.to_series() for ev in self.evaluators]
        df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
        df = df.reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def best_model(self, metric: str = "f1_macro") -> ModelEvaluator:
        """Return the ModelEvaluator with the highest value for ``metric``.

        Args:
            metric: Metric name to rank by.

        Returns:
            ModelEvaluator with the best score on the specified metric.
        """
        best = max(
            self.evaluators,
            key=lambda ev: ev.results_.get(metric, 0.0) or 0.0,
        )
        logger.info(
            "Best model by %s: [%s | %s] = %.4f",
            metric, best.model_name, best.feature_set,
            best.results_.get(metric, 0.0),
        )
        return best

    def save_comparison_table(self, out_path: Path) -> None:
        """Save the comparison table to a CSV file.

        Args:
            out_path: Target .csv file path.
        """
        df = self.comparison_table()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Comparison table saved → %s", out_path)
