"""Base Model Interface for LLM Fingerprinting Classifiers.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Model Training
Description:
    Abstract base class and shared utilities for all classical ML classifiers
    used in the Fingerprint project.  Every model (Logistic Regression,
    Linear SVM, Random Forest, XGBoost) inherits from FingerprintClassifier.
"""

import abc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

logger = logging.getLogger(__name__)


class FingerprintClassifier(abc.ABC):
    """Abstract base class for all LLM Fingerprinting classifiers.

    Subclasses must implement ``_build_model()``.
    All other methods (fit, predict, evaluate, save, load) are inherited.

    Attributes:
        cfg: Model-specific configuration dict from training.yaml.
        model: The underlying sklearn (or compatible) estimator.
        label_encoder_: Fitted LabelEncoder for decoding predictions.
        train_time_: Wall-clock training time in seconds.
        feature_set_: Name of the feature set used (tfidf, char, style, emb).
    """

    def __init__(self, cfg: Dict[str, Any], name: str) -> None:
        """Initialise the base classifier.

        Args:
            cfg: Model configuration dict.
            name: Human-readable model name (e.g. "logistic_regression").
        """
        self.cfg = cfg
        self.name = name
        self.model: Optional[Any] = None
        self.label_encoder_: Optional[Any] = None
        self.train_time_: float = 0.0
        self.feature_set_: str = "unknown"
        self._build_model()

    # ------------------------------------------------------------------
    # Abstract method — subclasses must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_model(self) -> None:
        """Instantiate the underlying sklearn estimator and assign to self.model."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        feature_set: str = "unknown",
    ) -> "FingerprintClassifier":
        """Fit the classifier on training data.

        Args:
            X_train: Feature matrix (sparse CSR or dense numpy array).
            y_train: Integer-encoded label array.
            feature_set: Name of the feature set (logged/stored for tracking).

        Returns:
            Self (for method chaining).
        """
        self.feature_set_ = feature_set
        logger.info(
            "[%s] Training on %s features, %d samples ...",
            self.name,
            feature_set,
            X_train.shape[0],
        )
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.train_time_ = time.perf_counter() - t0
        logger.info(
            "[%s] Training complete in %.2fs", self.name, self.train_time_
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Predict class indices for feature matrix X.

        Args:
            X: Feature matrix (sparse CSR or dense numpy array).

        Returns:
            1-D integer array of predicted class indices.
        """
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        """Predict class probabilities for feature matrix X.

        Args:
            X: Feature matrix (sparse CSR or dense numpy array).

        Returns:
            2-D probability array (n_samples × n_classes), or None if model
            does not support ``predict_proba``.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "decision_function"):
            # Convert decision function scores to pseudo-probabilities
            scores = self.model.decision_function(X)
            from scipy.special import softmax
            return softmax(scores, axis=1)
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path, suffix: str = "") -> Path:
        """Persist the fitted model to disk using joblib.

        Args:
            out_dir: Directory to save the model file.
            suffix: Optional suffix appended to the filename stem.

        Returns:
            Path to the saved .joblib file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.name}{('_' + suffix) if suffix else ''}.joblib"
        out_path = out_dir / fname
        payload = {
            "model": self.model,
            "name": self.name,
            "cfg": self.cfg,
            "train_time": self.train_time_,
            "feature_set": self.feature_set_,
        }
        joblib.dump(payload, out_path)
        logger.info("[%s] Model saved → %s", self.name, out_path)
        return out_path

    @classmethod
    def load(cls, model_path: Path) -> "FingerprintClassifier":
        """Load a previously saved model from a .joblib file.

        Args:
            model_path: Path to the .joblib model file.

        Returns:
            Reconstructed FingerprintClassifier instance.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        payload = joblib.load(model_path)

        # Reconstruct shell without calling _build_model
        instance = cls.__new__(cls)
        instance.name = payload["name"]
        instance.cfg = payload["cfg"]
        instance.model = payload["model"]
        instance.train_time_ = payload["train_time"]
        instance.feature_set_ = payload["feature_set"]
        instance.label_encoder_ = None
        logger.info("[%s] Model loaded from %s", instance.name, model_path)
        return instance
