"""General-Purpose Utilities for the Fingerprint Project.

Project : Fingerprint — LLM Fingerprinting Framework
Description:
    Shared helper functions used across multiple stages:
        - YAML configuration loading
        - Data splitting utilities
        - Reproducibility helpers
        - Path resolution helpers
        - Notebook display helpers
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for full reproducibility.

    Sets seeds for Python's random module, numpy, and (if available)
    PyTorch and CUDA.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("Global random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# YAML / Config helpers
# ---------------------------------------------------------------------------

def load_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_json(data: Dict[str, Any], out_path: Path) -> None:
    """Serialise a dict to a JSON file.

    Args:
        data: Dict to serialise (must contain JSON-serialisable values).
        out_path: Destination .json file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("JSON saved → %s", out_path)


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dict.

    Raises:
        FileNotFoundError: If json_path does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_test_val_split(
    X: Any,
    y: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.10,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature matrix and labels into train / validation / test sets.

    Args:
        X: Feature matrix (sparse CSR or dense numpy array).
        y: Integer label array.
        test_size: Fraction of data for the test set.
        val_size: Fraction of data for the validation set (taken from train).
        stratify: If True, preserve class proportions in each split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    from sklearn.model_selection import train_test_split

    strat = y if stratify else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )
    strat_val = y_train_val if stratify else None
    val_frac_of_train = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac_of_train,
        stratify=strat_val,
        random_state=random_state,
    )

    logger.info(
        "Split → train=%d  val=%d  test=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    """Return the project root directory (parent of the src/ directory).

    Returns:
        Absolute Path to the project root.
    """
    # Walk up from this file's location until we find configs/ or src/
    candidate = Path(__file__).resolve()
    for _ in range(6):
        candidate = candidate.parent
        if (candidate / "configs").exists() or (candidate / "notebooks").exists():
            return candidate
    return Path.cwd()


def make_output_dirs(*dirs: Path) -> None:
    """Create a list of directories if they do not exist.

    Args:
        *dirs: Positional Path arguments to create.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Created %d output directories.", len(dirs))


# ---------------------------------------------------------------------------
# Notebook display helpers
# ---------------------------------------------------------------------------

def display_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Convert a metrics dict to a display-friendly pandas DataFrame.

    Args:
        metrics: Dict of metric name → scalar value.

    Returns:
        Single-column DataFrame with metric names as index.
    """
    exclude = {"confusion_matrix", "classification_report",
               "y_pred", "y_test", "y_proba", "f1_per_class"}
    rows = {k: v for k, v in metrics.items() if k not in exclude}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])


def print_section_header(title: str, width: int = 70) -> None:
    """Print a formatted section header to stdout.

    Args:
        title: Section title string.
        width: Total header width in characters.
    """
    line = "=" * width
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")
