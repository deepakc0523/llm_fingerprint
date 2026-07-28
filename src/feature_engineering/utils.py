"""Feature Engineering Utilities and Configuration Object.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Feature Engineering
Author  : Senior NLP Research Team
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureEngineeringConfig:
    """Strongly typed configuration for the Feature Engineering stage."""

    project_name: str
    stage: str
    random_seed: int

    # Input
    pipeline_a_path: Path
    pipeline_b_path: Path

    # Output directories
    base_dir: Path
    tfidf_dir: Path
    char_dir: Path
    style_dir: Path
    emb_dir: Path

    # Column names
    text_col: str
    label_col: str

    # Sub-configs (raw dicts — parsed by each extractor)
    tfidf_cfg: Dict[str, Any]
    char_ngrams_cfg: Dict[str, Any]
    stylometric_cfg: Dict[str, Any]
    embeddings_cfg: Dict[str, Any]

    # Logging
    log_file: str = "logs/feature_engineering.log"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FeatureEngineeringConfig":
        """Load and validate configuration from a YAML file.

        Args:
            yaml_path: Absolute or relative path to feature_engineering.yaml.

        Returns:
            Populated FeatureEngineeringConfig instance.

        Raises:
            FileNotFoundError: If yaml_path does not exist.
        """
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}"
            )
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        inp = data.get("input", {})
        out = data.get("output", {})
        cols = data.get("columns", {})
        log_cfg = data.get("logging", {})

        return cls(
            project_name=data.get("project_name", "Fingerprint"),
            stage=data.get("stage", "feature_engineering"),
            random_seed=int(data.get("random_seed", 42)),
            pipeline_a_path=Path(inp.get("pipeline_a_path", "")),
            pipeline_b_path=Path(inp.get("pipeline_b_path", "")),
            base_dir=Path(out.get("base_dir", "data/features")),
            tfidf_dir=Path(out.get("tfidf_dir", "data/features/tfidf")),
            char_dir=Path(out.get("char_dir", "data/features/char")),
            style_dir=Path(out.get("style_dir", "data/features/style")),
            emb_dir=Path(out.get("emb_dir", "data/features/embedding")),
            text_col=cols.get("text", "generated_text"),
            label_col=cols.get("label", "model_label"),
            tfidf_cfg=data.get("tfidf", {}),
            char_ngrams_cfg=data.get("char_ngrams", {}),
            stylometric_cfg=data.get("stylometric", {}),
            embeddings_cfg=data.get("embeddings", {}),
            log_file=log_cfg.get("log_file", "logs/feature_engineering.log"),
            log_level=log_cfg.get("level", "INFO"),
        )

    def create_output_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in [self.base_dir, self.tfidf_dir, self.char_dir,
                  self.style_dir, self.emb_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Output directories verified/created.")


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

def setup_logger(log_file: str, level: str = "INFO") -> None:
    """Configure root logger to write to console and a log file.

    Args:
        log_file: Path string for the log file.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info("Logger initialised. Log file: %s", log_path)


def save_feature_matrix(
    matrix: Any,
    labels: np.ndarray,
    label_encoder: Any,
    out_dir: Path,
    stem: str,
) -> None:
    """Persist a feature matrix (sparse or dense) and its labels.

    Saves:
        <stem>.npz       — sparse matrix (scipy) or dense array (numpy)
        labels_<stem>.npy — integer label array
        classes_<stem>.npy — class name array from LabelEncoder

    Args:
        matrix: scipy sparse matrix or numpy ndarray.
        labels: 1-D numpy integer array of encoded class indices.
        label_encoder: Fitted sklearn LabelEncoder instance.
        out_dir: Target directory.
        stem: File name stem (e.g. "tfidf_fingerprint").
    """
    import scipy.sparse as sp

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / f"{stem}.npz"
    labels_path = out_dir / f"labels_{stem}.npy"
    classes_path = out_dir / f"classes_{stem}.npy"

    if sp.issparse(matrix):
        sp.save_npz(str(matrix_path), matrix)
    else:
        np.savez_compressed(str(matrix_path), data=matrix)

    np.save(str(labels_path), labels)
    np.save(str(classes_path), label_encoder.classes_)

    logger.info("Saved matrix → %s  (%s rows)", matrix_path, labels.shape[0])


def load_feature_matrix(
    matrix_path: Path,
    labels_path: Path,
) -> Tuple[Any, np.ndarray]:
    """Load a previously saved feature matrix and label array.

    Args:
        matrix_path: Path to .npz feature matrix file.
        labels_path: Path to .npy label array file.

    Returns:
        Tuple of (matrix, labels_array).
    """
    import scipy.sparse as sp

    matrix = sp.load_npz(str(matrix_path))
    labels = np.load(str(labels_path))
    logger.info("Loaded %s  [%d samples]", matrix_path.name, labels.shape[0])
    return matrix, labels


def encode_labels(
    series: "pd.Series",
) -> Tuple[np.ndarray, Any]:
    """Encode a string label Series to integer indices.

    Args:
        series: pandas Series of string class labels.

    Returns:
        Tuple of (encoded_integer_array, fitted_LabelEncoder).
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    encoded = le.fit_transform(series.values)
    logger.info(
        "Label encoding: %d classes → %s", len(le.classes_), list(le.classes_)
    )
    return encoded, le
