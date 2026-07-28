"""TF-IDF Feature Extractor.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Feature Engineering
Description:
    Extracts word-level and character-level TF-IDF feature matrices from
    preprocessed text corpora.  Both Pipeline A (Fingerprint-Preserving) and
    Pipeline B (Traditional NLP) outputs are supported.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)


class TFIDFExtractor:
    """Fits and transforms TF-IDF feature matrices (word and char variants).

    Attributes:
        cfg: TF-IDF sub-configuration dict from feature_engineering.yaml.
        word_vectorizer: Fitted word-level TfidfVectorizer.
        char_vectorizer: Fitted character-level TfidfVectorizer.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise with TF-IDF configuration dict.

        Args:
            cfg: The ``tfidf`` section from feature_engineering.yaml.
        """
        self.cfg = cfg
        self.word_vectorizer: Optional[TfidfVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        texts: pd.Series,
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Fit vectorizers on ``texts`` and return word + char matrices.

        Args:
            texts: 1-D pandas Series of preprocessed text documents.

        Returns:
            Tuple of (word_tfidf_matrix, char_tfidf_matrix), both CSR sparse.
        """
        logger.info("Fitting TF-IDF vectorizers on %d documents ...", len(texts))

        word_cfg = self.cfg.get("word", {})
        char_cfg = self.cfg.get("char", {})

        self.word_vectorizer = self._build_vectorizer(word_cfg, analyzer="word")
        self.char_vectorizer = self._build_vectorizer(char_cfg, analyzer="char_wb")

        word_matrix = self.word_vectorizer.fit_transform(texts)
        logger.info(
            "Word TF-IDF matrix: %s  vocab=%d",
            word_matrix.shape,
            len(self.word_vectorizer.vocabulary_),
        )

        char_matrix = self.char_vectorizer.fit_transform(texts)
        logger.info(
            "Char TF-IDF matrix: %s  vocab=%d",
            char_matrix.shape,
            len(self.char_vectorizer.vocabulary_),
        )

        return word_matrix, char_matrix

    def transform(
        self,
        texts: pd.Series,
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Transform ``texts`` using already-fitted vectorizers.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            Tuple of (word_tfidf_matrix, char_tfidf_matrix).

        Raises:
            RuntimeError: If vectorizers have not been fitted yet.
        """
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError(
                "TFIDFExtractor must be fitted before calling transform()."
            )
        return (
            self.word_vectorizer.transform(texts),
            self.char_vectorizer.transform(texts),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Persist fitted vectorizer objects to ``out_dir``.

        Args:
            out_dir: Directory where vectorizer .joblib files are written.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.word_vectorizer, out_dir / "tfidf_word_vectorizer.joblib")
        joblib.dump(self.char_vectorizer, out_dir / "tfidf_char_vectorizer.joblib")
        logger.info("TF-IDF vectorizers saved to %s", out_dir)

    @classmethod
    def load(cls, out_dir: Path) -> "TFIDFExtractor":
        """Load a previously saved TFIDFExtractor from ``out_dir``.

        Args:
            out_dir: Directory containing .joblib vectorizer files.

        Returns:
            Reconstructed TFIDFExtractor with fitted vectorizers.
        """
        instance = cls(cfg={})
        instance.word_vectorizer = joblib.load(
            out_dir / "tfidf_word_vectorizer.joblib"
        )
        instance.char_vectorizer = joblib.load(
            out_dir / "tfidf_char_vectorizer.joblib"
        )
        logger.info("TF-IDF vectorizers loaded from %s", out_dir)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vectorizer(
        cfg: Dict[str, Any],
        analyzer: str,
    ) -> TfidfVectorizer:
        """Instantiate a TfidfVectorizer from a config sub-dict.

        Args:
            cfg: Parameter dict (max_features, ngram_range, min_df, …).
            analyzer: sklearn analyzer string ("word" or "char_wb").

        Returns:
            Configured (unfitted) TfidfVectorizer.
        """
        ngram_range = cfg.get("ngram_range", [1, 2])
        return TfidfVectorizer(
            max_features=cfg.get("max_features", 50000),
            ngram_range=tuple(ngram_range),
            min_df=cfg.get("min_df", 2),
            max_df=cfg.get("max_df", 0.95),
            sublinear_tf=cfg.get("sublinear_tf", True),
            analyzer=analyzer,
            strip_accents="unicode",
            decode_error="replace",
        )

    def get_feature_names(self, kind: str = "word") -> np.ndarray:
        """Return feature name array for the specified vectorizer.

        Args:
            kind: ``"word"`` or ``"char"``.

        Returns:
            numpy array of feature name strings.
        """
        if kind == "word":
            return np.array(self.word_vectorizer.get_feature_names_out())
        return np.array(self.char_vectorizer.get_feature_names_out())

    def get_statistics(self, word_matrix: sp.csr_matrix, char_matrix: sp.csr_matrix) -> Dict[str, Any]:
        """Return a summary statistics dict for the extracted matrices.

        Args:
            word_matrix: Word-level TF-IDF CSR matrix.
            char_matrix: Char-level TF-IDF CSR matrix.

        Returns:
            Dict with shape, sparsity, and vocabulary size information.
        """
        def _sparsity(m: sp.csr_matrix) -> float:
            total = m.shape[0] * m.shape[1]
            return round(1.0 - m.nnz / total, 6) if total > 0 else 1.0

        return {
            "word_shape": word_matrix.shape,
            "word_vocab_size": len(self.word_vectorizer.vocabulary_),
            "word_sparsity": _sparsity(word_matrix),
            "char_shape": char_matrix.shape,
            "char_vocab_size": len(self.char_vectorizer.vocabulary_),
            "char_sparsity": _sparsity(char_matrix),
        }
