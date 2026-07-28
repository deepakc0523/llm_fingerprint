"""Classical ML Classifiers for LLM Fingerprinting.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Model Training
Description:
    Concrete implementations of the four classical classifiers:
        - LogisticRegressionClassifier
        - LinearSVMClassifier
        - RandomForestClassifier
        - XGBoostClassifier

    Each inherits from FingerprintClassifier and only overrides _build_model().
"""

import logging
from typing import Any, Dict

from src.models.base_model import FingerprintClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionClassifier(FingerprintClassifier):
    """Multinomial Logistic Regression classifier.

    Best suited for sparse, high-dimensional feature spaces such as
    TF-IDF word and character n-gram matrices.

    Strengths:
        - Fast training and inference.
        - Probabilistic output (calibrated probabilities).
        - Excellent baseline for high-dimensional sparse features.
        - Easy to interpret via coefficient inspection.

    Weaknesses:
        - Assumes linear decision boundary.
        - May underfit on complex non-linear patterns.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise Logistic Regression classifier.

        Args:
            cfg: The ``logistic_regression`` section from training.yaml.
        """
        super().__init__(cfg=cfg, name="logistic_regression")

    def _build_model(self) -> None:
        """Instantiate the sklearn LogisticRegression estimator."""
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            C=self.cfg.get("C", 1.0),
            max_iter=self.cfg.get("max_iter", 1000),
            solver=self.cfg.get("solver", "lbfgs"),
            multi_class=self.cfg.get("multi_class", "auto"),
            class_weight=self.cfg.get("class_weight", "balanced"),
            random_state=self.cfg.get("random_state", 42),
            n_jobs=-1,
        )
        logger.info("[logistic_regression] Model built with cfg: %s", self.cfg)


# ---------------------------------------------------------------------------
# Linear SVM
# ---------------------------------------------------------------------------

class LinearSVMClassifier(FingerprintClassifier):
    """Linear Support Vector Machine classifier (LinearSVC).

    Highly effective for high-dimensional sparse text classification.
    Operates without kernel tricks, making it extremely scalable.

    Strengths:
        - State-of-the-art performance on TF-IDF features.
        - Memory and time efficient with sparse matrices.
        - Robust to the curse of dimensionality.

    Weaknesses:
        - No direct probabilistic output (requires CalibratedClassifierCV).
        - Hard margin formulation can be sensitive to outliers.
        - Does not support kernel transformations.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise Linear SVM classifier.

        Args:
            cfg: The ``linear_svm`` section from training.yaml.
        """
        super().__init__(cfg=cfg, name="linear_svm")

    def _build_model(self) -> None:
        """Instantiate the sklearn LinearSVC estimator wrapped in CalibratedClassifierCV."""
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        base = LinearSVC(
            C=self.cfg.get("C", 1.0),
            max_iter=self.cfg.get("max_iter", 2000),
            class_weight=self.cfg.get("class_weight", "balanced"),
            random_state=self.cfg.get("random_state", 42),
        )
        # Wrap in Platt scaling to enable predict_proba
        self.model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        logger.info("[linear_svm] Model built (CalibratedClassifierCV). cfg: %s", self.cfg)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestClassifier(FingerprintClassifier):
    """Random Forest ensemble classifier.

    Excels at dense feature spaces such as stylometric and embedding
    features.  Provides built-in feature importance estimates.

    Strengths:
        - Robust to noisy features and irrelevant attributes.
        - Inherent feature importance (Gini importance).
        - No feature scaling required.
        - Handles non-linear decision boundaries naturally.

    Weaknesses:
        - Slower training than linear models on large datasets.
        - High memory usage due to storing multiple trees.
        - May overfit on small datasets with many features.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise Random Forest classifier.

        Args:
            cfg: The ``random_forest`` section from training.yaml.
        """
        super().__init__(cfg=cfg, name="random_forest")

    def _build_model(self) -> None:
        """Instantiate the sklearn RandomForestClassifier estimator."""
        from sklearn.ensemble import RandomForestClassifier as SKLearnRF

        self.model = SKLearnRF(
            n_estimators=self.cfg.get("n_estimators", 300),
            max_depth=self.cfg.get("max_depth", 20),
            min_samples_split=self.cfg.get("min_samples_split", 5),
            min_samples_leaf=self.cfg.get("min_samples_leaf", 2),
            max_features=self.cfg.get("max_features", "sqrt"),
            class_weight=self.cfg.get("class_weight", "balanced"),
            n_jobs=self.cfg.get("n_jobs", -1),
            random_state=self.cfg.get("random_state", 42),
        )
        logger.info("[random_forest] Model built with cfg: %s", self.cfg)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostClassifier(FingerprintClassifier):
    """XGBoost gradient boosting classifier.

    State-of-the-art performance on tabular and dense features.
    Combines gradient boosted trees with regularisation for high accuracy.

    Strengths:
        - Top performance on structured/tabular features.
        - Built-in L1/L2 regularisation prevents overfitting.
        - Handles missing values natively.
        - Native feature importance (gain, weight, cover).

    Weaknesses:
        - Slower training than Random Forest on large corpora.
        - Sensitive to hyperparameter choices.
        - Less effective than linear models on extremely sparse TF-IDF matrices.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise XGBoost classifier.

        Args:
            cfg: The ``xgboost`` section from training.yaml.
        """
        super().__init__(cfg=cfg, name="xgboost")

    def _build_model(self) -> None:
        """Instantiate the XGBClassifier estimator."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc

        self.model = XGBClassifier(
            n_estimators=self.cfg.get("n_estimators", 300),
            learning_rate=self.cfg.get("learning_rate", 0.05),
            max_depth=self.cfg.get("max_depth", 6),
            subsample=self.cfg.get("subsample", 0.8),
            colsample_bytree=self.cfg.get("colsample_bytree", 0.8),
            eval_metric=self.cfg.get("eval_metric", "mlogloss"),
            random_state=self.cfg.get("random_state", 42),
            n_jobs=self.cfg.get("n_jobs", -1),
            use_label_encoder=False,
        )
        logger.info("[xgboost] Model built with cfg: %s", self.cfg)
