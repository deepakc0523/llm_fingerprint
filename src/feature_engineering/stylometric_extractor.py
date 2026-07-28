"""Stylometric Feature Extractor.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Feature Engineering
Description:
    Extracts hand-crafted stylometric (writing-style) features from text.
    These features capture lexical richness, punctuation patterns,
    structural properties, and readability metrics — all of which
    are discriminative signals for LLM fingerprinting.

Feature Groups:
    1. Lexical      — vocabulary richness, word length, sentence length
    2. Punctuation  — ratios of individual punctuation characters
    3. Structural   — paragraph count, whitespace patterns, digit density
    4. Syntactic    — question/exclamation density, pronoun/conjunction usage
    5. Readability  — Flesch Reading Ease, Flesch-Kincaid Grade Level
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
_DIGIT_RE = re.compile(r"\d")
_CONJUNCTION_RE = re.compile(
    r"\b(and|but|or|nor|for|yet|so|although|because|since|while|whereas|however|"
    r"therefore|furthermore|moreover|nevertheless|consequently|thus)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(
    r"\b(i|me|my|myself|we|our|ours|ourselves|you|your|yours|yourself|"
    r"he|him|his|himself|she|her|hers|herself|it|its|itself|"
    r"they|them|their|theirs|themselves)\b",
    re.IGNORECASE,
)


class StylometricExtractor:
    """Extracts stylometric feature vectors from a text corpus.

    Each document is mapped to a fixed-length numerical feature vector
    containing lexical, punctuation, structural, syntactic, and readability
    measurements.

    Attributes:
        cfg: Stylometric sub-configuration dict from feature_engineering.yaml.
        feature_names_: List of feature name strings (populated after fit).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise with stylometric configuration dict.

        Args:
            cfg: The ``stylometric`` section from feature_engineering.yaml.
        """
        self.cfg = cfg
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Compute stylometric feature matrix for all documents.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            Dense float32 numpy array of shape (n_samples, n_features).
        """
        logger.info(
            "Extracting stylometric features from %d documents ...", len(texts)
        )
        rows = [self._extract_one(doc) for doc in texts]
        matrix = np.array(rows, dtype=np.float32)

        if not self.feature_names_:
            self.feature_names_ = self._build_feature_names()

        logger.info(
            "Stylometric matrix: %s  (%d features)", matrix.shape, matrix.shape[1]
        )
        return matrix

    def get_feature_dataframe(self, texts: pd.Series) -> pd.DataFrame:
        """Return stylometric features as a named pandas DataFrame.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            DataFrame with one column per feature.
        """
        matrix = self.fit_transform(texts)
        return pd.DataFrame(matrix, columns=self.feature_names_)

    # ------------------------------------------------------------------
    # Single-document feature extraction
    # ------------------------------------------------------------------

    def _extract_one(self, text: str) -> List[float]:
        """Compute all enabled stylometric features for a single document.

        Args:
            text: Raw text string.

        Returns:
            List of float feature values.
        """
        if not isinstance(text, str) or not text.strip():
            return [0.0] * len(self._build_feature_names())

        feats: List[float] = []
        n_chars = len(text)

        # ── Lexical features ───────────────────────────────────────────
        words = _WORD_RE.findall(text)
        n_words = len(words)
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        n_sentences = max(len(sentences), 1)

        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        avg_sent_len = n_words / n_sentences

        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / n_words if n_words > 0 else 0.0

        word_counts: Dict[str, int] = {}
        for w in words:
            word_counts[w.lower()] = word_counts.get(w.lower(), 0) + 1
        hapax = sum(1 for c in word_counts.values() if c == 1)
        hapax_ratio = hapax / n_words if n_words > 0 else 0.0

        feats += [avg_word_len, avg_sent_len, ttr, hapax_ratio, float(n_words)]

        # ── Punctuation features ───────────────────────────────────────
        punctuation_chars = self.cfg.get(
            "punctuation_chars", [".", ",", "!", "?", ";", ":", "-", '"', "'"]
        )
        for ch in punctuation_chars:
            feats.append(text.count(ch) / n_chars if n_chars > 0 else 0.0)

        digits_count = len(_DIGIT_RE.findall(text))
        feats.append(digits_count / n_chars if n_chars > 0 else 0.0)

        uppercase_count = sum(1 for c in text if c.isupper())
        feats.append(uppercase_count / n_chars if n_chars > 0 else 0.0)

        # ── Structural features ────────────────────────────────────────
        paragraphs = [p for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
        n_paragraphs = max(len(paragraphs), 1)
        avg_para_len = n_words / n_paragraphs

        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / n_chars if n_chars > 0 else 0.0

        feats += [
            float(n_sentences),
            float(n_paragraphs),
            avg_para_len,
            whitespace_ratio,
            float(n_chars),
        ]

        # ── Syntactic features ─────────────────────────────────────────
        question_count = text.count("?")
        exclamation_count = text.count("!")
        conjunction_count = len(_CONJUNCTION_RE.findall(text))
        pronoun_count = len(_PRONOUN_RE.findall(text))

        feats += [
            question_count / n_sentences,
            exclamation_count / n_sentences,
            conjunction_count / n_words if n_words > 0 else 0.0,
            pronoun_count / n_words if n_words > 0 else 0.0,
        ]

        # ── Readability features ───────────────────────────────────────
        syllable_count = sum(self._count_syllables(w) for w in words)
        if n_words > 0 and n_sentences > 0:
            flesch_ease = (
                206.835
                - 1.015 * (n_words / n_sentences)
                - 84.6 * (syllable_count / n_words)
            )
            flesch_kincaid = (
                0.39 * (n_words / n_sentences)
                + 11.8 * (syllable_count / n_words)
                - 15.59
            )
        else:
            flesch_ease = 0.0
            flesch_kincaid = 0.0

        feats += [
            np.clip(flesch_ease, -100.0, 121.0),
            np.clip(flesch_kincaid, 0.0, 18.0),
        ]

        return feats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for an English word using vowel groups.

        Args:
            word: Single English word string.

        Returns:
            Estimated syllable count (minimum 1).
        """
        word = word.lower().rstrip("e")
        vowels = re.findall(r"[aeiou]+", word)
        return max(len(vowels), 1)

    def _build_feature_names(self) -> List[str]:
        """Build ordered list of feature name strings.

        Returns:
            List of feature name strings matching the order produced by
            ``_extract_one``.
        """
        names: List[str] = []

        # Lexical
        names += [
            "avg_word_length",
            "avg_sentence_length",
            "type_token_ratio",
            "hapax_legomena_ratio",
            "word_count",
        ]

        # Punctuation
        punctuation_chars = self.cfg.get(
            "punctuation_chars", [".", ",", "!", "?", ";", ":", "-", '"', "'"]
        )
        for ch in punctuation_chars:
            safe = ch.replace('"', "dquote").replace("'", "squote")
            names.append(f"punct_ratio_{safe}")
        names += ["digit_ratio", "uppercase_ratio"]

        # Structural
        names += [
            "sentence_count",
            "paragraph_count",
            "avg_paragraph_length",
            "whitespace_ratio",
            "char_count",
        ]

        # Syntactic
        names += [
            "question_density",
            "exclamation_density",
            "conjunction_ratio",
            "pronoun_ratio",
        ]

        # Readability
        names += [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
        ]

        return names

    def get_statistics(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Return summary statistics for the stylometric feature matrix.

        Args:
            matrix: Dense numpy array (n_samples × n_features).

        Returns:
            Dict with shape, mean, std, and min/max per feature.
        """
        df = pd.DataFrame(matrix, columns=self.feature_names_)
        return {
            "shape": matrix.shape,
            "n_features": matrix.shape[1],
            "feature_means": df.mean().to_dict(),
            "feature_stds": df.std().to_dict(),
            "feature_mins": df.min().to_dict(),
            "feature_maxs": df.max().to_dict(),
        }
