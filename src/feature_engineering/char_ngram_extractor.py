"""Character N-Gram Feature Extractor.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Feature Engineering
Description:
    Extracts character-level n-gram TF-IDF feature matrices.
    Supports individual n-gram ranges and combined (stacked) matrices.
    Character n-grams are one of the most discriminative feature sets
    for authorship attribution and LLM fingerprinting tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)


class CharNGramExtractor:
    """Extracts character-level n-gram TF-IDF feature matrices.

    Supports:
        - Individual n-gram ranges (e.g., bigrams only, trigrams only).
        - Combined / horizontally stacked matrices from multiple ranges.

    Attributes:
        cfg: Char n-gram sub-configuration dict from feature_engineering.yaml.
        vectorizers: Dict mapping ngram_range tuple → fitted TfidfVectorizer.
        combined_vectorizer: Single vectorizer fitted on the widest range (when combine=False).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise with character n-gram config dict.

        Args:
            cfg: The ``char_ngrams`` section from feature_engineering.yaml.
        """
        self.cfg = cfg
        self.vectorizers: Dict[Tuple[int, int], TfidfVectorizer] = {}
        self.combined_vectorizer: Optional[TfidfVectorizer] = None
        self._combine: bool = cfg.get("combine", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: pd.Series) -> sp.csr_matrix:
        """Fit all n-gram vectorizers on ``texts`` and return feature matrix.

        If ``combine=True`` in config, individual matrices are horizontally
        stacked into a single feature matrix.

        Args:
            texts: 1-D pandas Series of preprocessed text documents.

        Returns:
            CSR sparse feature matrix (n_samples × total_features).
        """
        logger.info(
            "Fitting character n-gram vectorizers on %d documents ...", len(texts)
        )
        ngram_ranges: List[List[int]] = self.cfg.get("ngram_ranges", [[2, 4]])
        matrices: List[sp.csr_matrix] = []

        for rng in ngram_ranges:
            key = (rng[0], rng[1])
            vec = self._build_vectorizer(key)
            matrix = vec.fit_transform(texts)
            self.vectorizers[key] = vec
            matrices.append(matrix)
            logger.info(
                "  char(%d,%d): shape=%s  vocab=%d",
                key[0], key[1], matrix.shape, len(vec.vocabulary_),
            )

        if self._combine and len(matrices) > 1:
            combined = sp.hstack(matrices, format="csr")
            logger.info("Combined char n-gram matrix: %s", combined.shape)
            return combined

        # Return single matrix if not combining or only one range
        return matrices[0]

    def transform(self, texts: pd.Series) -> sp.csr_matrix:
        """Transform ``texts`` using already-fitted vectorizers.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            CSR sparse feature matrix.

        Raises:
            RuntimeError: If vectorizers have not been fitted yet.
        """
        if not self.vectorizers:
            raise RuntimeError(
                "CharNGramExtractor must be fitted before calling transform()."
            )
        matrices = [
            vec.transform(texts) for vec in self.vectorizers.values()
        ]
        if self._combine and len(matrices) > 1:
            return sp.hstack(matrices, format="csr")
        return matrices[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Persist all fitted vectorizers to ``out_dir``.

        Args:
            out_dir: Directory where vectorizer .joblib files are written.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        for (lo, hi), vec in self.vectorizers.items():
            fname = f"char_ngram_{lo}_{hi}_vectorizer.joblib"
            joblib.dump(vec, out_dir / fname)
        logger.info("Char n-gram vectorizers saved to %s", out_dir)

    @classmethod
    def load(cls, out_dir: Path, cfg: Dict[str, Any]) -> "CharNGramExtractor":
        """Load fitted vectorizers from ``out_dir``.

        Args:
            out_dir: Directory containing .joblib vectorizer files.
            cfg: Original char n-gram config dict.

        Returns:
            Reconstructed CharNGramExtractor with fitted vectorizers.
        """
        instance = cls(cfg=cfg)
        for path in sorted(out_dir.glob("char_ngram_*_vectorizer.joblib")):
            parts = path.stem.replace("char_ngram_", "").replace("_vectorizer", "")
            lo, hi = (int(x) for x in parts.split("_"))
            instance.vectorizers[(lo, hi)] = joblib.load(path)
        logger.info("Char n-gram vectorizers loaded from %s", out_dir)
        return instance

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, matrix: sp.csr_matrix) -> Dict[str, Any]:
        """Return summary statistics for the combined n-gram matrix.

        Args:
            matrix: CSR sparse matrix to inspect.

        Returns:
            Dict with shape, sparsity, and per-range vocabulary sizes.
        """
        total = matrix.shape[0] * matrix.shape[1]
        sparsity = round(1.0 - matrix.nnz / total, 6) if total > 0 else 1.0
        vocab_sizes = {
            f"char_{lo}_{hi}": len(vec.vocabulary_)
            for (lo, hi), vec in self.vectorizers.items()
        }
        return {
            "combined_shape": matrix.shape,
            "sparsity": sparsity,
            "total_features": matrix.shape[1],
            "vocab_per_range": vocab_sizes,
            "n_ranges": len(self.vectorizers),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vectorizer(self, ngram_range: Tuple[int, int]) -> TfidfVectorizer:
        """Build a TfidfVectorizer for a given character n-gram range.

        Args:
            ngram_range: (min_n, max_n) tuple.

        Returns:
            Configured (unfitted) TfidfVectorizer.
        """
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=self.cfg.get("max_features", 30000),
            min_df=self.cfg.get("min_df", 2),
            max_df=self.cfg.get("max_df", 0.95),
            sublinear_tf=self.cfg.get("sublinear_tf", True),
            strip_accents="unicode",
            decode_error="replace",
        )
