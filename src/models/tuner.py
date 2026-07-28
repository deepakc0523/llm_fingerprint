"""Hyperparameter Tuning Engine.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Hyperparameter Tuning
Description:
    Provides a unified interface for three tuning strategies:
        1. GridSearchCV     — exhaustive grid search
        2. RandomizedSearchCV — random sampling from parameter distributions
        3. Optuna           — Bayesian TPE optimisation (most efficient)

    All tuning runs are tracked, compared, and the best parameters
    are returned in a consistent format.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter search spaces
# ---------------------------------------------------------------------------

LR_PARAM_GRID: Dict[str, List[Any]] = {
    "C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000, 2000],
}

LR_PARAM_DIST: Dict[str, Any] = {
    "C": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000, 2000],
}

SVM_PARAM_GRID: Dict[str, List[Any]] = {
    "base_estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0],
    "base_estimator__max_iter": [1000, 2000, 5000],
}

RF_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}

RF_PARAM_DIST: Dict[str, Any] = {
    "n_estimators": [100, 150, 200, 250, 300, 400, 500],
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 3, 5, 7, 10],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 3, 4],
}

XGB_PARAM_DIST: Dict[str, Any] = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 1.5, 2.0],
}


class HyperparameterTuner:
    """Unified hyperparameter tuning interface.

    Supports GridSearchCV, RandomizedSearchCV, and Optuna strategies.

    Attributes:
        cfg: Tuning sub-config dict from training.yaml.
        method: Tuning strategy name ("grid", "random", "optuna").
        cv_folds: Number of cross-validation folds.
        scoring: Sklearn scoring string for optimisation objective.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.
        results_: List of result dicts from each tuning run.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise HyperparameterTuner.

        Args:
            cfg: The ``tuning`` section from training.yaml.
        """
        self.cfg = cfg
        self.method: str = cfg.get("method", "optuna")
        self.cv_folds: int = int(cfg.get("cv_folds", 5))
        self.scoring: str = cfg.get("scoring", "f1_macro")
        self.n_jobs: int = int(cfg.get("n_jobs", -1))
        self.n_iter: int = int(cfg.get("n_iter", 20))
        self.n_trials: int = int(cfg.get("n_trials", 50))
        self.random_state: int = int(cfg.get("random_state", 42))
        self.timeout: Optional[int] = cfg.get("timeout", 3600)
        self.results_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune_grid(
        self,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        X_train: Any,
        y_train: np.ndarray,
        model_name: str = "model",
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Run GridSearchCV exhaustive hyperparameter search.

        Args:
            estimator: Unfitted sklearn estimator.
            param_grid: Parameter grid dict.
            X_train: Training feature matrix.
            y_train: Training label array.
            model_name: Name for logging/tracking purposes.

        Returns:
            Tuple of (best_estimator, best_params, best_score).
        """
        from sklearn.model_selection import GridSearchCV

        logger.info("[%s] Starting GridSearchCV ...", model_name)
        t0 = time.perf_counter()

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        best_params = search.best_params_
        best_score = search.best_score_
        logger.info(
            "[%s] GridSearchCV complete. Best %s=%.4f  params=%s  time=%.1fs",
            model_name, self.scoring, best_score, best_params, elapsed,
        )
        self.results_.append({
            "model": model_name, "method": "grid",
            "best_score": best_score, "best_params": best_params, "time": elapsed,
        })
        return search.best_estimator_, best_params, best_score

    def tune_random(
        self,
        estimator: Any,
        param_distributions: Dict[str, Any],
        X_train: Any,
        y_train: np.ndarray,
        model_name: str = "model",
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Run RandomizedSearchCV random hyperparameter sampling.

        Args:
            estimator: Unfitted sklearn estimator.
            param_distributions: Parameter distribution dict.
            X_train: Training feature matrix.
            y_train: Training label array.
            model_name: Name for logging/tracking purposes.

        Returns:
            Tuple of (best_estimator, best_params, best_score).
        """
        from sklearn.model_selection import RandomizedSearchCV

        logger.info("[%s] Starting RandomizedSearchCV (n_iter=%d) ...", model_name, self.n_iter)
        t0 = time.perf_counter()

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        best_params = search.best_params_
        best_score = search.best_score_
        logger.info(
            "[%s] RandomizedSearchCV complete. Best %s=%.4f  params=%s  time=%.1fs",
            model_name, self.scoring, best_score, best_params, elapsed,
        )
        self.results_.append({
            "model": model_name, "method": "random",
            "best_score": best_score, "best_params": best_params, "time": elapsed,
        })
        return search.best_estimator_, best_params, best_score

    def tune_optuna(
        self,
        objective_fn: Any,
        model_name: str = "model",
    ) -> Tuple[Dict[str, Any], float]:
        """Run Optuna Bayesian TPE hyperparameter optimisation.

        Args:
            objective_fn: Callable that accepts an optuna.Trial and returns float score.
            model_name: Name for logging/tracking purposes.

        Returns:
            Tuple of (best_params_dict, best_score).
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("optuna is not installed. Run: pip install optuna") from exc

        logger.info("[%s] Starting Optuna (n_trials=%d) ...", model_name, self.n_trials)
        t0 = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0

        best_params = study.best_params
        best_score = study.best_value
        logger.info(
            "[%s] Optuna complete. Best %s=%.4f  params=%s  time=%.1fs",
            model_name, self.scoring, best_score, best_params, elapsed,
        )
        self.results_.append({
            "model": model_name, "method": "optuna",
            "best_score": best_score, "best_params": best_params,
            "time": elapsed, "n_trials": len(study.trials),
        })
        return best_params, best_score

    def compare_results(self) -> "pd.DataFrame":
        """Return a summary DataFrame comparing all tuning runs.

        Returns:
            pandas DataFrame with one row per tuning run.
        """
        import pandas as pd
        if not self.results_:
            logger.warning("No tuning results to compare.")
            return pd.DataFrame()
        return pd.DataFrame(self.results_).sort_values(
            "best_score", ascending=False
        ).reset_index(drop=True)

    def save_results(self, out_path: Path) -> None:
        """Save tuning results summary to a CSV file.

        Args:
            out_path: Path to the output CSV file.
        """
        df = self.compare_results()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Tuning results saved → %s", out_path)
